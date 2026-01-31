import joblib
from pathlib import Path


def predict_data(X):
    """Predict class labels for the input data using the saved model.
    """
    model_path = Path("../model") / "predict_wine_dataset_random_forest.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            "No trained model found. Run the training script: `python train.py` to create model/predict_wine_dataset_random_forest.pkl"
        )
    model = joblib.load(str(model_path))
    y_pred = model.predict(X)
    return y_pred
