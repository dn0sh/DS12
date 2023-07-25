"""
Module for building and processing features.

This module provides functions and transformers for processing categorical and numerical features,
as well as building a column transformer for feature transformation.

Functions:
    process_categorical_features: Process categorical features.
    build_categorical_pipeline: Build a pipeline for processing categorical features.
    process_numerical_features: Process numerical features.
    build_numerical_pipeline: Build a pipeline for processing numerical features.
    make_features: Apply a column transformer to create features.
    extract_target: Extract the target variable from a DataFrame.
    build_transformer: Build a column transformer for processing features.

Classes:
    OutlierRemover: Custom transformer for removing outliers from numerical features.
    MyTransform: Custom transformer for numerical feature transformation.
"""
from typing import List
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd


def process_categorical_features(categorical_df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes the categorical features by applying a pipeline of transformations.

    Args:
        categorical_df: DataFrame containing the categorical features.

    Returns:
        Transformed DataFrame with categorical features.
    """
    categorical_pipeline = build_categorical_pipeline()
    return pd.DataFrame(categorical_pipeline.fit_transform(categorical_df).toarray())


def build_categorical_pipeline() -> Pipeline:
    """
    Builds a pipeline for processing categorical features.

    Returns:
        Categorical feature processing pipeline.
    """
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder()),
        ]
    )
    return categorical_pipeline


def process_numerical_features(numerical_df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes the numerical features by applying a pipeline of transformations.

    Args:
        numerical_df: DataFrame containing the numerical features.

    Returns:
        Transformed DataFrame with numerical features.
    """
    num_pipeline = build_numerical_pipeline()
    return pd.DataFrame(num_pipeline.fit_transform(numerical_df))


def build_numerical_pipeline() -> Pipeline:
    """
    Builds a pipeline for processing numerical features.

    Returns:
        Numerical feature processing pipeline.
    """
    num_pipeline = Pipeline(
        [
            ("OutlierRemover", OutlierRemover()),
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )
    return num_pipeline


def make_features(transformer: ColumnTransformer, input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the column transformer to the input DataFrame to create features.

    Args:
        transformer: ColumnTransformer for transforming the input features.
        input_df: Input DataFrame.

    Returns:
        Transformed DataFrame with features.
    """
    return pd.DataFrame(transformer.transform(input_df))


def extract_target(input_df: pd.DataFrame, target_col: List[str]) -> pd.Series:
    """
    Extracts the target variable from the DataFrame.

    Args:
        input_df: DataFrame containing the target variable.
        target_col: List of target column names.

    Returns:
        Series with the target variable.
    """
    target = input_df[target_col].values
    return target


class OutlierRemover(BaseEstimator, TransformerMixin):
    """
    Custom transformer to remove outliers from numerical features.

    Args:
        factor: The factor used for outlier detection.
    """
    def __init__(self, factor=1.5):
        self.factor = factor

    def outlier_removal(self, data_frame: pd.DataFrame):
        """
        Removes outliers from the input DataFrame.

        Args:
            data_frame: Input DataFrame.

        Returns:
            DataFrame with outliers replaced by NaN values.
        """
        data_frame = pd.Series(data_frame).copy()
        first_quartile = data_frame.quantile(0.25)
        third_quartile = data_frame.quantile(0.75)
        iqr = third_quartile - first_quartile
        lower_bound = first_quartile - (self.factor * iqr)
        upper_bound = third_quartile + (self.factor * iqr)
        data_frame.loc[((data_frame < lower_bound) | (data_frame > upper_bound))] = np.nan
        return pd.Series(data_frame)

    def fit(self, x_data, y_data=None):
        """
        Fit method of the OutlierRemover transformer.

        Args:
            x_data: Input features.
            y_data: Target variable (default: None).

        Returns:
            The fitted transformer object.
        """
        print(x_data, y_data)
        return self

    def transform(self, input_array: np.array):
        """
        Applies the transformer to the input array.

        Args:
            input_array: Input array.

        Returns:
            Transformed array with outliers replaced by NaN values.
        """
        return pd.DataFrame(input_array).apply(self.outlier_removal)


def build_transformer(categorical_features: List[str],
                      numerical_features: List[str]) -> ColumnTransformer:
    """
    Builds a column transformer for processing categorical and numerical features.

    Args:
        categorical_features: List of categorical feature names.
        numerical_features: List of numerical feature names.

    Returns:
        ColumnTransformer object for feature transformation.
    """
    transformer = ColumnTransformer(
        [
            (
                "categorical_pipeline",
                build_categorical_pipeline(),
                #[c for c in categorical_features],
                list(categorical_features)
            ),
            (
                "numerical_pipeline",
                build_numerical_pipeline(),
                list(numerical_features)
            )
        ]
    )
    return transformer
