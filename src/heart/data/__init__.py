"""
Module for data-related functionality.

This module provides functions for reading data from files
  and splitting it into train and test sets.
"""

from .make_dataset import read_data, split_train_test_data

__all__ = ["read_data", "split_train_test_data"]
