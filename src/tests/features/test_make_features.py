from typing import List

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from heart.data.make_dataset import read_data
from heart.entities.feature import FeatureConfig
from heart.features import build_transformer, extract_target, make_features
from heart.features.build_features import OutlierRemover


@pytest.fixture
def feature_params(
    categorical_features: List[str],
    features_to_drop: List[str],
    numerical_features: List[str],
    target_col: str,
) -> FeatureConfig:
    params = FeatureConfig(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        features_to_drop=features_to_drop,
        target_col=[target_col],
    )
    return params


def test_make_features(
    feature_params: FeatureConfig, dataset: pd.DataFrame,
):
    transformer = build_transformer(feature_params.categorical_features, feature_params.numerical_features)
    transformer.fit(dataset)
    features = make_features(transformer, dataset)
    assert not pd.isnull(features).any().any()
    assert all(x not in features.columns for x in feature_params.features_to_drop)


def test_outlier_transformer(dataset: pd.DataFrame, numerical_features: List[str]):

    outlier_remover_100 = OutlierRemover(1.5)
    result = outlier_remover_100.fit_transform(dataset[numerical_features])
    q1 = dataset[numerical_features].quantile(0.25)
    q3 = dataset[numerical_features].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    result = result.fillna(result.mean())
    assert sum(((result < lower_bound) | (result > upper_bound)).sum()) == 0


def test_extract_features(feature_params: FeatureConfig, dataset: pd.DataFrame):

    target = extract_target(dataset, feature_params.target_col)
    assert_allclose(
        dataset[feature_params.target_col], target
    )
