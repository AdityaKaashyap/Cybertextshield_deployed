from pymongo import MongoClient
from datetime import datetime

# -----------------------------
# MongoDB Connection (LOCALHOST)
# -----------------------------
client = MongoClient("mongodb://127.0.0.1:27017/")
db = client["smish_db"]
smish_collection = db["detected_smish"]

def save_smish_to_db(message: str, probability: float):
    """Store detected smish message in MongoDB."""
    smish_collection.insert_one({
        "message": message,
        "probability": float(probability),
        "detected_at": datetime.utcnow()
    })
