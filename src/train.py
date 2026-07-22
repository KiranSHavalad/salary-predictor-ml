from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path

from preprocessing import (
    load_dataset,
    select_features,
)

from evaluate import (
    evaluate_model,
    compare_predictions,
    create_evaluation_report
)

from pipeline import build_pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def train():

    df = load_dataset(PROJECT_ROOT / "data" / "raw" / "salaries.csv")

    X, y = select_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    pipeline = build_pipeline()

    pipeline.fit(X_train, y_train)

    metrics = evaluate_model(
        pipeline,
        X_test,
        y_test,
    )

    print(metrics)

    comparison = compare_predictions(
        pipeline,
        X_test,
        y_test,
    )

    print(comparison)

    joblib.dump(
        pipeline,
        PROJECT_ROOT / "models" / "salary_prediction_pipeline.pkl",
    )

    print("Pipeline trained successfully!")
    report = create_evaluation_report(metrics)

    print(report)

    return pipeline, X_test, y_test


if __name__ == "__main__":
    train()
