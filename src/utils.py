import joblib


def save_pipeline(pipeline, model_path):
    """
    Save the trained pipeline to disk.
    """
    joblib.dump(pipeline, model_path)


def load_pipeline(model_path):
    """
    Load a trained pipeline from disk.
    """
    return joblib.load(model_path)