from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

import pandas as pd


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained regression model.
    """

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, predictions)

    print("=" * 50)
    print("Model Evaluation Report")
    print("=" * 50)
    print(f"MAE  : {mae:.2f}")
    print(f"MSE  : {mse:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.4f}")

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
    }


def compare_predictions(model, X_test, y_test, n=10):
    """
    Compare actual and predicted values.
    """

    predictions = model.predict(X_test)

    comparison = pd.DataFrame(
        {
            "Actual Salary": y_test.values,
            "Predicted Salary": predictions,
        }
    )

    return comparison.head(n)


def create_evaluation_report(metrics):
    report = pd.DataFrame(
        {
            "Metric": metrics.keys(),
            "Value": metrics.values(),
        }
    )

    return report