"""
Module containing model configurations.

This module defines data classes for configuring different machine learning models,
such as Random Forest and Logistic Regression. Each configuration class specifies
the parameters for model instantiation.

Classes:
    RFConfig: Configuration for Random Forest model.
    LogregConfig: Configuration for Logistic Regression model.
    ModelConfig: Configuration for a generic model.
"""
from typing import Any
from dataclasses import dataclass, field


@dataclass
class RFConfig:
    """
    Configuration for Random Forest model.

    Args:
        n_estimators: Number of decision trees in the forest.
        max_depth: Maximum depth of the trees.
        random_state: Random seed for reproducibility.
    """
    _target_: str = field(default='sklearn.ensemble.RandomForestClassifier')
    n_estimators: int = field(default=100)
    max_depth: int = field(default=3)
    random_state: int = field(default=42)


@dataclass
class LogregConfig:
    """
    Configuration for Logistic Regression model.

    Args:
        penalty: Type of regularization penalty.
        solver: Algorithm to use in the optimization problem.
        C: Inverse of regularization strength.
        random_state: Random seed for reproducibility.
        max_iter: Maximum number of iterations.
    """
    _target_: str = field(default='sklearn.linear_model.LogisticRegression')
    penalty: str = field(default='l1')
    solver: str = field(default='liblinear')
    C: float = field(default=1.0)
    random_state: int = field(default=42)
    max_iter: int = field(default=100)


@dataclass
class ModelConfig:
    """
    Configuration for a generic model.

    Args:
        model_name: Name of the model.
        model_params: Parameters for model instantiation.
    """
    model_name: str
    model_params: Any
