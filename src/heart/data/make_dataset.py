"""
Module for dataset handling and splitting.

This module provides functions for reading dataset from a file
  and splitting it into train and test subsets.

Functions:
    read_data: Read a dataset from a file.
    split_train_test_data: Split a dataset into train and test subsets.
"""
from typing import Tuple
from sklearn.model_selection import train_test_split
import pandas as pd


def read_data(dataset_path: str) -> pd.DataFrame:
    """
    Read a dataset from the specified file.

    Args:
        dataset_path: The path to the dataset file.

    Returns:
        A DataFrame containing the loaded data.
    """
    data = pd.read_csv(dataset_path)
    return data


# def split_train_test_data():
def split_train_test_data(dataset: pd.DataFrame,
                          test_size: float,
                          random_state: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a dataset into random train and test subsets.

    Args:
        dataset: The DataFrame containing the dataset.
        test_size: The proportion of the dataset to be used for the test subset.
        random_state: The random seed for the random number generator.

    Returns:
        A tuple of two DataFrames: the training subset and the test subset.
    """
    train, test = train_test_split(dataset, test_size=test_size, random_state=random_state)
    return train, test
