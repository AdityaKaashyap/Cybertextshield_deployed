from pymongo import MongoClient
from datetime import datetime

# -----------------------------------
# MongoDB Connection (Localhost)
# -----------------------------------
try:
    client = MongoClient("mongodb://127.0.0.1:27017/", serverSelectionTimeoutMS=3000)
    client.server_info()  # Force connection
    db = client["smish_db_collector"]
    smish_collection = db["detected_smish"]
    print("✅ MongoDB connected.")
except Exception as e:
    print("❌ MongoDB NOT connected:", e)
    smish_collection = None


def save_smish_to_db(message: str, probability: float, prediction: str):
    """Insert detected smish message into MongoDB if connected."""
    if smish_collection is None:
        print("⚠️ MongoDB unavailable — smish not saved.")
        return

    smish_collection.insert_one({
        "message": message,
        "prediction": prediction,
        "probability": float(probability),
        "timestamp": datetime.utcnow()
    })
