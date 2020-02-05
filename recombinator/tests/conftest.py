import numpy as np
import pytest


@pytest.fixture()
def original_iid_sample_size() -> int:
    return 100


@pytest.fixture()
def original_time_series_sample_size() -> int:
    return 1000


@pytest.fixture()
def number_of_boostrap_replications() -> int:
    return 100000


@pytest.fixture()
def number_of_time_series_boostrap_replications() -> int:
    return 10000


@pytest.fixture()
def iid_sample(original_iid_sample_size: int) -> np.ndarray:
    n = original_iid_sample_size
    np.random.seed(1)
    x = np.abs(np.random.randn(n))

    return x


@pytest.fixture()
def time_series_sample(original_time_series_sample_size: int) -> np.ndarray:
    np.random.seed(1)

    # number of time periods
    T = original_time_series_sample_size

    # draw random errors
    e = np.random.randn(T)
    y = np.zeros((T,))

    # y is an AR(1) with phi_1 = 0.5
    phi_1 = 0.5
    y[0] = e[0] * np.sqrt(1.0 / (1.0 - phi_1 ** 2))
    for t in range(1, T):
        y[t] = phi_1 * y[t - 1] + e[t]

    return y
