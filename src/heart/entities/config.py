"""
Module for defining configuration entities.
"""
from dataclasses import dataclass, field
#from hydra.utils import instantiate
from hydra.core.config_store import ConfigStore
from .models import ModelConfig
from .feature import FeatureConfig


@dataclass
class DatasetConfig:
    """
    Configuration class for dataset-related settings.

    Attributes:
        input_data_path (str): Path to the input data file.
    """
    input_data_path: str


@dataclass
class SplittingConfig:
    """
    Configuration class for dataset splitting settings.

    Attributes:
        test_size (float, optional): The proportion of the dataset to include in the test split.
            Defaults to 0.25.
        random_state (int, optional): The random state for reproducibility. Defaults to 42.
    """
    test_size: float = field(default=0.25)
    random_state: int = field(default=42)


@dataclass
class TrainingPipelineConfig:
    """
    Configuration class for the training pipeline.

    Attributes:
        model (ModelConfig): Configuration for the model.
        dataset (DatasetConfig): Configuration for the dataset.
        feature (FeatureConfig): Configuration for the features.
        split (SplittingConfig): Configuration for the data splitting.
    """
    model: ModelConfig
    dataset: DatasetConfig
    feature: FeatureConfig
    split: SplittingConfig


cs = ConfigStore.instance()
cs.store(name="base_config", node=TrainingPipelineConfig)
