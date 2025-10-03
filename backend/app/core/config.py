import os

class Settings:
    PROJECT_NAME: str = "Smishing Detection API"
    VERSION: str = "1.0"
    DESCRIPTION: str = "API for detecting smishing SMS using HGNN"
    
    # MongoDB
    MONGO_URI: str = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    MONGO_DB: str = os.getenv("MONGO_DB", "smishing_db")

        # JWT
    SECRET_KEY: str = os.getenv("SECRET_KEY", "ADGP2026")  # ⚠️ change in prod
    ALGORITHM: str = "HS256"
    
settings = Settings()