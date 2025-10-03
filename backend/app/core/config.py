import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

class Settings:
    PROJECT_NAME: str = "Smishing Detection API"
    VERSION: str = "1.0"
    DESCRIPTION: str = "API for detecting smishing SMS using HGNN"

    # MongoDB (Atlas cluster or local)
    MONGO_URI: str = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    MONGO_DB: str = os.getenv("MONGO_DB", "smishing_db")

    # JWT
    SECRET_KEY: str = os.getenv("SECRET_KEY", "ADGP2026")
    ALGORITHM: str = os.getenv("ALGORITHM", "HS256")

settings = Settings()
