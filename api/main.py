# API entry point placeholder.
from fastapi import FastAPI

from api.routes import router

app = FastAPI(
    title="Salary Prediction API",
    version="1.0.0",
    description="Machine Learning API for predicting employee salaries",
)

app.include_router(router)


@app.get("/")
def health():
    return {
        "status": "healthy",
        "message": "Salary Prediction API is running."
    }