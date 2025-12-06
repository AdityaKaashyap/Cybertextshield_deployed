from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn.functional as F
import joblib
import re
from torch_geometric.data import Data
from torch_geometric.nn import HypergraphConv

from db import save_smish_to_db  # DB logger


# -----------------------------
# 1. Request & Response Models
# -----------------------------
class MessageRequest(BaseModel):
    messages: list[str]
    senders: list[str] | None = None   # NEW FIELD ADDED

class PredictionResponse(BaseModel):
    predictions: list[str]
    probabilities: list[float]


# -----------------------------
# 2. HGNN Model Class
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
# 3. Text Preprocessing
# -----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text


def build_hypergraph(X, K=10):
    from sklearn.neighbors import NearestNeighbors
    N = X.shape[0]
    nbrs = NearestNeighbors(n_neighbors=min(K+1, N), metric="cosine").fit(X)
    _, indices = nbrs.kneighbors(X)

    edge_index = []
    for i in range(N):
        for j in indices[i]:
            edge_index.append([j, i])

    return torch.tensor(edge_index, dtype=torch.long).t()


def make_data(X):
    edge_index = build_hypergraph(X)
    return Data(x=torch.tensor(X, dtype=torch.float), edge_index=edge_index)


# -----------------------------
# 4. FastAPI App Initialization
# -----------------------------
app = FastAPI(title="HGNN Smishing Detection API")

vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HGNN(in_channels=2000, hidden_channels=64, num_classes=2).to(device)
model.load_state_dict(torch.load("hgnn_smish_model.pth", map_location=device))
model.eval()


# -----------------------------
# 5. Prediction Endpoint
# -----------------------------
@app.post("/predict", response_model=PredictionResponse)
def predict(request: MessageRequest):
    try:
        senders = request.senders or ["UNKNOWN"] * len(request.messages)

        # Preallocate results
        predictions = [""] * len(request.messages)
        probabilities = [0.0] * len(request.messages)

        SAFE_SUFFIXES = ("-G", "-S", "-P", "-T")
        rule_based_flags = []

        # 1️⃣ Apply RULE-BASED detection first
        for i, sender in enumerate(senders):
            sender = sender.upper()

            if sender.endswith(SAFE_SUFFIXES):
                predictions[i] = "ham"
                probabilities[i] = 0.0
                rule_based_flags.append(True)
            else:
                rule_based_flags.append(False)

        # 2️⃣ Prepare ML model inputs for remaining messages
        messages_to_predict = [
            msg for msg, is_rule in zip(request.messages, rule_based_flags) if not is_rule
        ]

        if messages_to_predict:
            cleaned = [clean_text(m) for m in messages_to_predict]
            X = vectorizer.transform(cleaned).toarray()
            data = make_data(X).to(device)

            with torch.no_grad():
                out = model(data.x, data.edge_index)
                probs = F.softmax(out, dim=1)[:, 1].cpu().numpy()
                preds = out.argmax(dim=1).cpu().numpy()
                labels = label_encoder.inverse_transform(preds)

            # Fill values back into correct positions
            idx = 0
            for i in range(len(request.messages)):
                if not rule_based_flags[i]:
                    predictions[i] = labels[idx]
                    probabilities[i] = float(probs[idx])
                    idx += 1

        # 3️⃣ Save smish results to DB
        for raw_msg, pred, prob in zip(request.messages, predictions, probabilities):
            if pred.lower() == "smish":
                save_smish_to_db(raw_msg, prob, pred)

        return PredictionResponse(
            predictions=predictions,
            probabilities=probabilities
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------
# 6. Root Test Endpoint
# -----------------------------
@app.get("/")
def root():
    return {"message": "HGNN Smishing Detection API is running."}
