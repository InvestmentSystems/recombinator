import numpy as np
import pytest


@pytest.fixture()
def original_iid_sample_size() -> int:
    return 100


@pytest.fixture()
def number_of_iid_boostrap_replications() -> int:
    return 100000


@pytest.fixture()
def number_of_time_series_boostrap_replications() -> int:
    return 100000


@pytest.fixture()
def iid_sample(original_iid_sample_size: int):
    n = original_iid_sample_size
    np.random.seed(1)
    x = np.abs(np.random.randn(n))

    return x
