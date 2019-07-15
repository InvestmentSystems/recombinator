import numpy as np
import typing as tp


from recombinator.iid_bootstrap import \
    iid_balanced_bootstrap, \
    iid_bootstrap, \
    iid_bootstrap_vectorized, \
    iid_bootstrap_via_choice, \
    iid_bootstrap_via_loop, \
    iid_bootstrap_with_antithetic_resampling

from recombinator.statistics import \
    estimate_confidence_interval_from_bootstrap, \
    estimate_standard_error_from_bootstrap


def _test_iid_bootstrap_generic(bootstrap_function: tp.Callable,
                                x: np.ndarray,
                                replications: int):
    n = len(x)
    percentile = 75
    original_statistic = np.percentile(x, percentile)

    x_resampled \
        = bootstrap_function(x, replications=replications)

    assert x_resampled.shape == (replications, n)

    resampled_statistic \
        = np.percentile(x_resampled,
                        percentile,
                        axis=1)

    estimated_standard_error \
        = estimate_standard_error_from_bootstrap(
            bootstrap_estimates=resampled_statistic,
            original_estimate=original_statistic)

    assert 0.10 <= estimated_standard_error <= 0.12

    lower_ci, upper_ci \
        = estimate_confidence_interval_from_bootstrap(
            bootstrap_estimates=resampled_statistic,
            confidence_level=95)

    assert 0.82 <= lower_ci <= 0.86
    assert 1.1 <= upper_ci <= 1.2


def test_iid_bootstrap(iid_sample: np.ndarray,
                       number_of_iid_boostrap_replications: int):
    _test_iid_bootstrap_generic(
        bootstrap_function=iid_bootstrap,
        x=iid_sample,
        replications=number_of_iid_boostrap_replications)


def test_iid_bootstrap_via_loop(iid_sample: np.ndarray,
                                number_of_iid_boostrap_replications: int):
    _test_iid_bootstrap_generic(
        bootstrap_function=iid_bootstrap_via_loop,
        x=iid_sample,
        replications=number_of_iid_boostrap_replications)


def test_iid_bootstrap_vectorized(iid_sample: np.ndarray,
                                  number_of_iid_boostrap_replications: int):
    _test_iid_bootstrap_generic(
        bootstrap_function=iid_bootstrap_vectorized,
        x=iid_sample,
        replications=number_of_iid_boostrap_replications)


def test_iid_bootstrap_via_choice(iid_sample: np.ndarray,
                                  number_of_iid_boostrap_replications: int):
    _test_iid_bootstrap_generic(
        bootstrap_function=iid_bootstrap_via_choice,
        x=iid_sample,
        replications=number_of_iid_boostrap_replications)


def test_iid_balanced_bootstrap(iid_sample: np.ndarray,
                                number_of_iid_boostrap_replications: int):
    _test_iid_bootstrap_generic(
        bootstrap_function=iid_balanced_bootstrap,
        x=iid_sample,
        replications=number_of_iid_boostrap_replications)


def test_iid_antithetic_bootstrap(iid_sample: np.ndarray,
                                  number_of_iid_boostrap_replications: int):
    _test_iid_bootstrap_generic(
        bootstrap_function=iid_bootstrap_with_antithetic_resampling,
        x=iid_sample,
        replications=number_of_iid_boostrap_replications)
