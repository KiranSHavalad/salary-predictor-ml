from fastapi import APIRouter

from api.schemas import (
    SalaryPredictionRequest,
    SalaryPredictionResponse,
)

from api.dependencies import pipeline
from src.predict import SalaryPredictor

router = APIRouter()

predictor = SalaryPredictor(pipeline)


@router.post(
    "/predict",
    response_model=SalaryPredictionResponse,
)
def predict_salary(request: SalaryPredictionRequest):

    prediction = predictor.predict(request.model_dump())

    return SalaryPredictionResponse(
        predicted_salary=prediction
    )