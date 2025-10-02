from fastapi import APIRouter
from app.schemas.predict_schema import PredictRequest, PredictResponse
from app.services.prediction_service import predict_message

router = APIRouter()

@router.post("/", response_model=PredictResponse)
def predict(request: PredictRequest):
    prediction, confidence = predict_message(request.message)
    return PredictResponse(
        message=request.message,
        prediction=prediction,
        confidence=confidence
    )
