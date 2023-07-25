"""
Module for model-related functionality.

This module provides functions for training and serializing regression models.
"""
from typing import Union
import pickle
import pandas as pd
from hydra.utils import instantiate
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

SklearnRegressionModel = Union[RandomForestClassifier, LogisticRegression]


def train_model(
    model_params, train_features: pd.DataFrame, target: pd.Series
) -> SklearnRegressionModel:
    """
    Trains a regression model using the provided features and target.

    Args:
        model_params: Parameters for model instantiation.
        train_features: Input features for training.
        target: Target variable for training.

    Returns:
        Trained regression model.
    """
    #model = instantiate(model_params).fit(train_features, target)
    model = instantiate(model_params).fit(train_features, target.ravel())
    return model


def serialize_model(model: SklearnRegressionModel, output: str) -> str:
    """
    Serializes the trained model to a file.

    Args:
        model: Trained regression model.
        output: Output file path.

    Returns:
        Path to the serialized model file.
    """
    with open(output, "wb") as file:
        pickle.dump(model, file)
    return output
