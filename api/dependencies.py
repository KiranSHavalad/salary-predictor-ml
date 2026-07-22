from pathlib import Path

from src.utils import load_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "salary_prediction_pipeline.pkl"

pipeline = load_pipeline(MODEL_PATH)
