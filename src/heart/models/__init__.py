"""
Module for model-related functionality.

This module provides functions for training and serializing regression models,
as well as predicting and evaluating model performance.

Functions:
    train_model: Train a regression model.
    serialize_model: Serialize a trained model to a file.
    predict_model: Make predictions using a trained model.
    evaluate_model: Evaluate model performance.
"""
from .fit import (
    train_model,
    serialize_model,
)

from .predict import (
    predict_model,
    evaluate_model,
)

__all__ = [
    "train_model",
    "serialize_model",
    "evaluate_model",
    "predict_model",
]
