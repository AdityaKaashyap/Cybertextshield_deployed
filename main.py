from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn.functional as F
import joblib
import re
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import HypergraphConv
from pymongo import MongoClient
from datetime import datetime
from db import save_smish_to_db



# -----------------------------
# 1. Define Request & Response Models
# -----------------------------
class MessageRequest(BaseModel):
    messages: list[str]

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
# 3. Preprocessing Functions
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
# 4. Initialize FastAPI
# -----------------------------
app = FastAPI(title="HGNN Smishing Detection API")

# Load preprocessing objects
vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Load model
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
        messages = [clean_text(m) for m in request.messages]
        X = vectorizer.transform(messages).toarray()
        data = make_data(X)
        data = data.to(device)

        with torch.no_grad():
            out = model(data.x, data.edge_index)
            probs = F.softmax(out, dim=1)[:, 1].cpu().numpy()
            preds = out.argmax(dim=1).cpu().numpy()
            labels = label_encoder.inverse_transform(preds)

        # -----------------------------
        # Save smish messages to MongoDB
        # -----------------------------
        for msg, pred, prob in zip(request.messages, labels, probs):
            if pred.lower() == "smish":
                collection.insert_one({
                    "message": msg,
                    "prediction": pred,
                    "probability": float(prob),
                    "timestamp": datetime.utcnow()
                })

        return PredictionResponse(
            predictions=labels.tolist(),
            probabilities=probs.tolist()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------
# 6. Test Endpoint
# -----------------------------
@app.get("/")
def root():
    return {"message": "HGNN Smishing Detection API is running."}
