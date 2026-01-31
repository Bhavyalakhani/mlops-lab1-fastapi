from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from pathlib import Path
from data import load_data, split_data
import logging


def fit_model(X_train, y_train):
    """
    Train a RandomForestClassifier and save the model to a file.

    Args:
        X_train : Training data features.
        y_train : Training data target values.
    """
    rf_model = RandomForestClassifier(n_estimators=100, random_state=12)
    rf_model.fit(X_train, y_train)
    model_dir = Path("../model")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "predict_wine_dataset_random_forest.pkl"
    joblib.dump(rf_model, str(model_path))

    return rf_model, model_path

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    model, model_path = fit_model(X_train, y_train)

    # Evaluate on the test set and log the accuracy
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logging.info(f"Saved trained model to: {model_path}")
    logging.info(f"Model accuracy on test set: {acc:.4f}")
