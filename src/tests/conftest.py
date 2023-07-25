import os
import tempfile
from typing import List

import pandas as pd

import pytest

from heart.data.make_dataset import read_data
from heart.entities import (
    LogregConfig, RFConfig, SplittingConfig, FeatureConfig, DatasetConfig
)
from tests.data_generator import generate_dataset


@pytest.fixture(scope='session')
def dataset_path() -> str:
    data = generate_dataset()
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp:
        data.to_csv(temp)
    return temp.name


@pytest.fixture(scope='session')
def dataset(dataset_path) -> pd.DataFrame:
    data = read_data(dataset_path)
    return data


@pytest.fixture(scope='session')
def target_col():
    return "target"


@pytest.fixture(scope='session')
def categorical_features() -> List[str]:
    return [
        "sex",
        "cp",
        "fbs",
        "restecg",
        "exang",
        "slope",
        "ca",
        "thal",
    ]


@pytest.fixture(scope='session')
def numerical_features() -> List[str]:
    return [
        "age",
        "trestbps",
        "chol",
        "thalach",
        "oldpeak",
    ]


@pytest.fixture(scope='session')
def features_to_drop() -> List[str]:
    return []


@pytest.fixture(scope='session')
def features_to_drop() -> List[str]:
    return []


@pytest.fixture(scope='class')
def log_reg_model() -> LogregConfig:
    return LogregConfig(
            _target_='sklearn.linear_model.LogisticRegression',
            penalty='l1',
            solver='liblinear',
            C=1.0,
            random_state=42,
            max_iter=100,
    )


@pytest.fixture(scope='class')
def rf_model() -> RFConfig:
    return RFConfig(
        _target_='sklearn.ensemble.RandomForestClassifier',
        max_depth=100,
        n_estimators=100,
        random_state=42
    )


@pytest.fixture(scope='session')
def split_config_v1() -> SplittingConfig:
    return SplittingConfig(
        test_size=0.25,
        random_state=42
    )


@pytest.fixture(scope='session')
def feature_param_v1(categorical_features,
                     numerical_features,
                     target_col,
                     features_to_drop
                     ) -> FeatureConfig:
    return FeatureConfig(
        categorical_features,
        numerical_features,
        target_col,
        True,
        features_to_drop
    )


@pytest.fixture(scope='session')
def dataset_config_v1() -> DatasetConfig:
    return DatasetConfig(
        input_data_path=dataset_path
    )
