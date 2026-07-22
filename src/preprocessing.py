import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def load_dataset(path):
    """
    Load dataset from CSV.
    """
    return pd.read_csv(path)


def select_features(df):
    """
    Separate input features and target variable.
    """

    features = [
        "work_year",
        "experience_level",
        "employment_type",
        "job_title",
        "employee_residence",
        "remote_ratio",
        "company_location",
        "company_size",
    ]

    target = "salary_in_usd"

    X = df[features].copy()
    y = df[target].copy()

    return X, y


def build_preprocessor():
    """
    Build preprocessing pipeline.
    """

    numeric_features = [
        "work_year",
        "remote_ratio",
    ]

    categorical_features = [
        "experience_level",
        "employment_type",
        "job_title",
        "employee_residence",
        "company_location",
        "company_size",
    ]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor