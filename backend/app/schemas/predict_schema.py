from pydantic import BaseModel

class PredictRequest(BaseModel):
    message: str

class PredictResponse(BaseModel):
    message: str
    prediction: str
    confidence: float

