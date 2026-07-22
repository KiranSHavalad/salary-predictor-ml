import pandas as pd
from pathlib import Path

from utils import load_pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "salary_prediction_pipeline.pkl"


class SalaryPredictor:

    def __init__(self, model_path):
        self.pipeline = load_pipeline(model_path)

    def predict(self, employee_data):
        df = pd.DataFrame(employee_data)
        return self.pipeline.predict(df)


def predict_salary(employee_data):
    """
    Predict salary for one or more employees.
    """

    pipeline = load_pipeline(MODEL_PATH)

    input_df = pd.DataFrame(employee_data)

    predictions = pipeline.predict(input_df)

    return predictions


if __name__ == "__main__":

    predictor = SalaryPredictor(MODEL_PATH)

    employee = {
        "work_year": [2025],
        "experience_level": ["SE"],
        "employment_type": ["FT"],
        "job_title": ["Data Scientist"],
        "employee_residence": ["US"],
        "remote_ratio": [100],
        "company_location": ["US"],
        "company_size": ["M"],
    }

    prediction = predictor.predict(employee)

    print(prediction)
