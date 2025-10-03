from fastapi import APIRouter, Depends
from app.schemas.predict_schema import PredictRequest, PredictResponse
from app.services.prediction_service import predict_message
from app.core.auth import get_current_user  # 🔹 import JWT dependency

router = APIRouter()

@router.post("/", response_model=PredictResponse)
def predict(request: PredictRequest, current_user: str = Depends(get_current_user)):
    """
    Predict whether the message is ham or smish.
    Requires the user to be logged in (JWT in Authorization header).
    """
    prediction, confidence = predict_message(request.message)
    return PredictResponse(
        message=request.message,
        prediction=prediction,
        confidence=confidence
    )
