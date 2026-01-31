import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split


def load_data():
    """
    Load the Wine dataset and return the features and target values.
    Returns:
        X : The features of the wine dataset.
        y : The target values of the wine dataset.
    """
    data = load_wine()
    X = data.data
    y = data.target
    return X, y

def split_data(X, y):
    """
    Split the data into training and testing sets.
    Args:
        X : The features of the wie dataset.
        y : The target values of the wine dataset.
    Returns:
        X_train, X_test, y_train, y_test (tuple): The split dataset.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)
    return X_train, X_test, y_train, y_test