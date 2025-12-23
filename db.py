from pymongo import MongoClient
from datetime import datetime
from dotenv import load_dotenv
import os
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

# -----------------------------------
# MongoDB Connection (Localhost)
# -----------------------------------
try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
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
