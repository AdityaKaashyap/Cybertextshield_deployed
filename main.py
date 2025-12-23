from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn.functional as F
import joblib
import re
from torch_geometric.data import Data
from torch_geometric.nn import HypergraphConv
from db import save_smish_to_db

# -----------------------------
# 1. Request & Response Models
# -----------------------------
class MessageRequest(BaseModel):
    messages: list[str]
    senders: list[str] | None = None  # OPTIONAL sender ids


class PredictionResponse(BaseModel):
    predictions: list[str]
    probabilities: list[float]


# -----------------------------
# 2. HGNN Model
# -----------------------------
class HGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, dropout=0.5):
        super(HGNN, self).__init__()
        self.hconv1 = HypergraphConv(in_channels, hidden_channels)
        self.hconv2 = HypergraphConv(hidden_channels, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.hconv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.hconv2(x, edge_index)
        return x


# -----------------------------
# 3. Preprocessing
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text


def build_hypergraph_single(N):
    """
    Builds a minimal hypergraph for batch prediction.
    Even for 1 SMS, HGNN needs a graph structure.
    """
    if N == 1:
        return torch.tensor([[0], [0]], dtype=torch.long)

    # Fully connected graph for simplicity
    edges = []
    for i in range(N):
        for j in range(N):
            edges.append([i, j])

    return torch.tensor(edges, dtype=torch.long).t()


# -----------------------------
# 4. App Init
# -----------------------------
app = FastAPI(title="HGNN Smishing Detection API")

vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = HGNN(in_channels=2000, hidden_channels=64, num_classes=2).to(device)
model.load_state_dict(torch.load("hgnn_smish_model.pth", map_location=device))
model.eval()

SAFE_SUFFIXES = ("-G", "-S", "-P", "-T")


# -----------------------------
# 5. PREDICTION ENDPOINT
# -----------------------------
@app.post("/predict", response_model=PredictionResponse)
def predict(request: MessageRequest):

    try:
        messages = request.messages
        senders = request.senders or ["UNKNOWN"] * len(messages)
        N = len(messages)

        predictions = [""] * N
        probabilities = [0.0] * N

        ml_indices = []

        # 1️⃣ RULE-BASED FILTER
        for i in range(N):
            sender = (senders[i] or "").upper()
            if sender.endswith(SAFE_SUFFIXES):
                predictions[i] = "ham"
                probabilities[i] = 0.0
            else:
                ml_indices.append(i)

        # 2️⃣ ML PREDICTION FOR REMAINING
        if ml_indices:
            msgs = [messages[i] for i in ml_indices]
            cleaned = [clean_text(m) for m in msgs]

            X = vectorizer.transform(cleaned).toarray()
            X_tensor = torch.tensor(X, dtype=torch.float).to(device)

            edge_index = build_hypergraph_single(len(ml_indices)).to(device)

            with torch.no_grad():
                out = model(X_tensor, edge_index)
                prob = F.softmax(out, dim=1)[:, 1].cpu().numpy()
                pred_idx = out.argmax(dim=1).cpu().numpy()
                labels = label_encoder.inverse_transform(pred_idx)

            for k, idx in enumerate(ml_indices):
                label = labels[k]
                prob_val = float(prob[k])

                predictions[idx] = label
                probabilities[idx] = prob_val

                # ✅ SAVE ONLY SMISH
                if label.lower() == "smish":
                    save_smish_to_db(
                        message=messages[idx],
                        prediction=label,
                        probability=prob_val
                    )


        return PredictionResponse(predictions=predictions, probabilities=probabilities)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------
# 6. Root endpoint
# -----------------------------
@app.get("/")
def root():
    return {"message": "HGNN Model API Running"}


# -----------------------------
# 7. Run server
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
