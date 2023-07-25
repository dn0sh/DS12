"""
Module for defining feature-related entities.
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class FeatureConfig:
    """
    Configuration class for feature-related settings.

    Attributes:
        categorical_features (List[str]): List of categorical feature names.
        numerical_features (List[str]): List of numerical feature names.
        target_col (List[str]): List of target variable names.
        use_log_trick (bool, optional): Flag indicating whether to apply logarithmic transformation
            to numerical features. Defaults to True.
        features_to_drop (Optional[List[str]], optional): List of feature names to drop.
            Defaults to None.
    """
    categorical_features: List[str]
    numerical_features: List[str]
    target_col: List[str]
    use_log_trick: bool = field(default=True)
    features_to_drop: Optional[List[str]] = None
