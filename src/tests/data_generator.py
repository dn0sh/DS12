import numpy as np
import pandas as pd


def __generate_by_normal_distribution(size: int, loc: int = 1, scale: int = 1) -> np.ndarray:
    return np.random.normal(loc=loc, scale=scale, size=size).astype(int)


def __generate_by_binomial_distribution(size: int, n: int = 1, p: float = 0.5) -> np.ndarray:
    return np.random.binomial(n=n, p=p, size=size).astype(int)


def __generate_random(size: int, low: int = 0, high: int = 1) -> np.ndarray:
    return np.random.randint(low=low, high=high, size=size).astype(int)


def generate_dataset(dataset_size=100, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    fake_dataset = pd.DataFrame()
    fake_dataset["age"] = __generate_by_normal_distribution(size=dataset_size, loc=45, scale=6)
    fake_dataset["sex"] = __generate_by_binomial_distribution(size=dataset_size, n=1, p=0.4)
    fake_dataset["cp"] = __generate_random(size=dataset_size, low=0, high=3)
    fake_dataset["trestbps"] = __generate_by_normal_distribution(size=dataset_size, loc=120, scale=15)
    fake_dataset["chol"] = __generate_by_normal_distribution(size=dataset_size, loc=234, scale=50)
    fake_dataset["fbs"] = __generate_by_binomial_distribution(size=dataset_size, n=1, p=0.20)
    fake_dataset["restecg"] = __generate_random(size=dataset_size, low=0, high=3)
    fake_dataset["thalach"] = __generate_by_normal_distribution(size=dataset_size, loc=120, scale=18)
    fake_dataset["exang"] = __generate_by_binomial_distribution(size=dataset_size, n=1, p=0.45)
    fake_dataset["oldpeak"] = np.clip(
        __generate_by_normal_distribution(size=dataset_size, loc=1, scale=3), 0, None
    ).astype(int)
    fake_dataset["slope"] = __generate_random(size=dataset_size, low=0, high=4)
    fake_dataset["ca"] = __generate_random(size=dataset_size, low=0, high=3)
    fake_dataset["thal"] = __generate_random(size=dataset_size, low=0, high=7)
    fake_dataset["target"] = __generate_by_binomial_distribution(size=dataset_size, n=1, p=0.37)
    return fake_dataset
