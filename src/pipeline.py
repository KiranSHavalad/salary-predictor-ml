from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

from preprocessing import build_preprocessor


def build_pipeline():
    """
    Create complete ML pipeline.
    """

    preprocessor = build_preprocessor()

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LinearRegression()),
        ]
    )

    return pipeline