import os
import pickle
from typing import List, Tuple

import pandas as pd
import pytest

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from heart.entities import FeatureConfig
from heart.features import build_transformer, extract_target, make_features
from heart.models.fit import serialize_model, train_model


@pytest.fixture
def features_and_target(
    dataset: pd.DataFrame, categorical_features: List[str], numerical_features: List[str]
) -> Tuple[pd.DataFrame, pd.Series]:
    params = FeatureConfig(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        features_to_drop=['fbs'],
        target_col=["target"],
    )
    transformer = build_transformer(categorical_features, numerical_features)
    transformer.fit(dataset)
    features = make_features(transformer, dataset)
    target = extract_target(dataset, params.target_col)
    return features, target


@pytest.mark.parametrize(
    "model, model_class",
    [
        pytest.param(pytest.lazy_fixture('log_reg_model'), LogisticRegression, id="age"),
        pytest.param(pytest.lazy_fixture('rf_model'), RandomForestClassifier, id="rf"),
    ],
)
def test_train_model(features_and_target: Tuple[pd.DataFrame, pd.Series], model, model_class):
    features, target = features_and_target
    model = train_model(model, features, target)
    assert isinstance(model, model_class)

