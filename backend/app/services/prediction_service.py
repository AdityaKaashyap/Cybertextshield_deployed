import torch
import torch.nn.functional as F
import joblib
import numpy as np
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
from app.models.hgnn_model import HGNN
from app.utils.preprocessing import clean_text
import os

# Load artifacts
ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "hgnn_smish_model.pth")
VECTORIZER_PATH = os.path.join(ARTIFACT_DIR, "tfidf_vectorizer.pkl")
ENCODER_PATH = os.path.join(ARTIFACT_DIR, "label_encoder.pkl")

vectorizer = joblib.load(VECTORIZER_PATH)
label_encoder = joblib.load(ENCODER_PATH)

# Model params (must match training!)
IN_CHANNELS = vectorizer.max_features
HIDDEN_CHANNELS = 64
NUM_CLASSES = len(label_encoder.classes_)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HGNN(in_channels=IN_CHANNELS, hidden_channels=HIDDEN_CHANNELS, num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Build small hypergraph for single inference
def build_hypergraph(X, K=3):
    N = X.shape[0]
    nbrs = NearestNeighbors(n_neighbors=min(K+1, N), metric="cosine").fit(X)
    _, indices = nbrs.kneighbors(X)
    edge_index = []
    for i in range(N):
        for j in indices[i]:
            edge_index.append([j, i])
    return torch.tensor(edge_index, dtype=torch.long).t()

def predict_message(message: str):
    # Preprocess
    text = clean_text(message)
    X = vectorizer.transform([text]).toarray()
    
    # Build hypergraph for single sample
    edge_index = build_hypergraph(X, K=1)  # K=1 for single prediction
    data = Data(x=torch.tensor(X, dtype=torch.float), edge_index=edge_index).to(device)

    with torch.no_grad():
        out = model(data.x, data.edge_index)
        probs = F.softmax(out, dim=1).cpu().numpy()[0]
        pred_idx = np.argmax(probs)
        prediction = label_encoder.inverse_transform([pred_idx])[0]
        confidence = float(probs[pred_idx])

    return prediction, confidence
