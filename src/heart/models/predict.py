"""
Module for performing model-related operations.

This module provides functions for training, predicting, and evaluating regression models.
"""
from typing import Dict, Union
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

SklearnRegressionModel = Union[RandomForestRegressor, LogisticRegression]


def predict_model(
    model: SklearnRegressionModel, features: pd.DataFrame
) -> np.ndarray:
    """
    Predicts the target values using the trained model.

    Args:
        model: Trained regression model.
        features: Input features for prediction.

    Returns:
        Predicted target values.
    """
    predicts = model.predict(features)
    return predicts


def evaluate_model(
    predicts: np.ndarray, target: pd.Series
) -> Dict[str, float]:
    """
    Evaluates the performance of the model predictions.

    Args:
        predicts: Predicted target values.
        target: True target values.

    Returns:
        Dictionary with evaluation metrics.
    """
    return {
        "accuracy": accuracy_score(target, predicts),
    }
