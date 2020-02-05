import math
import numpy as np
import typing as tp

from statsmodels.tsa.ar_model import AR

from recombinator.block_bootstrap import moving_block_bootstrap
from recombinator.block_bootstrap import moving_block_bootstrap_vectorized
from recombinator.block_bootstrap import circular_block_bootstrap
from recombinator.block_bootstrap import circular_block_bootstrap_vectorized
from recombinator.block_bootstrap import stationary_bootstrap
from recombinator.optimal_block_length import optimal_block_length
from recombinator.tapered_block_bootstrap import tapered_block_bootstrap
from recombinator.tapered_block_bootstrap import \
    tapered_block_bootstrap_vectorized

from recombinator.statistics import (
    estimate_confidence_interval_from_bootstrap,
    estimate_standard_error_from_bootstrap
)

from recombinator.tests.rng_link_tools import (
    numpy_to_numba_rng_link_generic_tester,
    numba_to_numpy_rng_link_generic_tester
)


def _test_block_bootstrap_generic(
        bootstrap_function: tp.Callable,
        y: np.ndarray,
        block_length: tp.Union[int, float],
        replications: int,
        lower_ci_bounds: tp.Tuple[float, float] = (0.35, 0.38),
        upper_ci_bounds: tp.Tuple[float, float] = (0.48, 0.50),
        bse_bounds: tp.Tuple[float, float] = (0.05, 0.065),
        test_rng_link: bool = False):
    T = len(y)
    # original estimate
    ar = AR(y)
    estimate_from_original_data = ar.fit(maxlag=1)

    y_star \
        = bootstrap_function(x=y,
                             block_length=block_length,
                             replications=replications)

    assert y_star.shape == (replications, T)

    estimates_from_bootstrap = []
    ar_estimates_from_bootstrap = np.zeros((len(y_star),))

    for b in range(len(y_star)):
        y_bootstrap = np.array(y_star[b, :].squeeze())
        ar_bootstrap = AR(y_bootstrap)
        estimate_from_bootstrap = ar_bootstrap.fit(maxlag=1)
        estimates_from_bootstrap.append(estimate_from_bootstrap)
        ar_estimates_from_bootstrap[b] = estimate_from_bootstrap.params[1]

    lower_ci, upper_ci \
        = estimate_confidence_interval_from_bootstrap(
            bootstrap_estimates=ar_estimates_from_bootstrap,
            confidence_level=95)

    assert lower_ci_bounds[0] <= lower_ci <= lower_ci_bounds[1]
    assert upper_ci_bounds[0] <= upper_ci <= upper_ci_bounds[1]

    bootstrap_standard_error \
        = estimate_standard_error_from_bootstrap(
            bootstrap_estimates=ar_estimates_from_bootstrap,
            original_estimate=estimate_from_original_data.params[1])

    assert bse_bounds[0] <= bootstrap_standard_error <= bse_bounds[1]

    if test_rng_link:
        numpy_to_numba_rng_link_generic_tester(
            x=y,
            bootstrap_function=bootstrap_function,
            number_of_boostrap_replications=replications,
            block_length=block_length)

        numba_to_numpy_rng_link_generic_tester(
            x=y,
            bootstrap_function=bootstrap_function,
            number_of_boostrap_replications=replications,
            block_length=block_length)


def test_moving_block_bootstrap(
        time_series_sample: np.ndarray,
        number_of_time_series_boostrap_replications: int):
    y = time_series_sample
    B = number_of_time_series_boostrap_replications
    b_star = optimal_block_length(y)
    block_length = math.ceil(b_star[0].b_star_cb)

    lower_ci_bounds = (0.35, 0.38)
    upper_ci_bounds = (0.48, 0.50)
    bse_bounds = (0.03, 0.07)

    _test_block_bootstrap_generic(
        bootstrap_function=moving_block_bootstrap,
        y=time_series_sample,
        block_length=block_length,
        replications=number_of_time_series_boostrap_replications,
        lower_ci_bounds=lower_ci_bounds,
        upper_ci_bounds=upper_ci_bounds,
        bse_bounds=bse_bounds,
        test_rng_link=True)

    _test_block_bootstrap_generic(
        bootstrap_function=moving_block_bootstrap_vectorized,
        y=time_series_sample,
        block_length=block_length,
        replications=number_of_time_series_boostrap_replications,
        lower_ci_bounds=lower_ci_bounds,
        upper_ci_bounds=upper_ci_bounds,
        bse_bounds=bse_bounds)


def test_circular_block_bootstrap(
        time_series_sample: np.ndarray,
        number_of_time_series_boostrap_replications: int):
    y = time_series_sample
    B = number_of_time_series_boostrap_replications
    b_star = optimal_block_length(y)
    block_length = math.ceil(b_star[0].b_star_cb)

    lower_ci_bounds = (0.34, 0.37)
    upper_ci_bounds = (0.46, 0.50)
    bse_bounds = (0.03, 0.07)

    _test_block_bootstrap_generic(
        bootstrap_function=circular_block_bootstrap,
        y=time_series_sample,
        block_length=block_length,
        replications=number_of_time_series_boostrap_replications,
        lower_ci_bounds=lower_ci_bounds,
        upper_ci_bounds=upper_ci_bounds,
        bse_bounds=bse_bounds,
        test_rng_link=True)

    _test_block_bootstrap_generic(
        bootstrap_function=circular_block_bootstrap_vectorized,
        y=time_series_sample,
        block_length=block_length,
        replications=number_of_time_series_boostrap_replications,
        lower_ci_bounds=lower_ci_bounds,
        upper_ci_bounds=upper_ci_bounds,
        bse_bounds=bse_bounds)


def test_stationary_bootstrap(
        time_series_sample: np.ndarray,
        number_of_time_series_boostrap_replications: int):
    y = time_series_sample
    B = number_of_time_series_boostrap_replications
    b_star = optimal_block_length(y)
    block_length = b_star[0].b_star_sb

    lower_ci_bounds = (0.35, 0.38)
    upper_ci_bounds = (0.46, 0.50)
    bse_bounds = (0.03, 0.07)

    _test_block_bootstrap_generic(
        bootstrap_function=stationary_bootstrap,
        y=time_series_sample,
        block_length=block_length,
        replications=number_of_time_series_boostrap_replications,
        lower_ci_bounds=lower_ci_bounds,
        upper_ci_bounds=upper_ci_bounds,
        bse_bounds=bse_bounds,
        test_rng_link=True)


def test_tapered_block_bootstrap(
        time_series_sample: np.ndarray,
        number_of_time_series_boostrap_replications: int):
    y = time_series_sample
    B = number_of_time_series_boostrap_replications
    b_star = optimal_block_length(y)
    block_length = math.ceil(b_star[0].b_star_cb)

    lower_ci_bounds = (0.38, 0.42)
    upper_ci_bounds = (0.52, 0.54)
    bse_bounds = (0.03, 0.07)

    _test_block_bootstrap_generic(
        bootstrap_function=tapered_block_bootstrap,
        y=time_series_sample,
        block_length=block_length,
        replications=number_of_time_series_boostrap_replications,
        lower_ci_bounds=lower_ci_bounds,
        upper_ci_bounds=upper_ci_bounds,
        bse_bounds=bse_bounds,
        test_rng_link=True)

    _test_block_bootstrap_generic(
        bootstrap_function=tapered_block_bootstrap_vectorized,
        y=time_series_sample,
        block_length=block_length,
        replications=number_of_time_series_boostrap_replications,
        lower_ci_bounds=lower_ci_bounds,
        upper_ci_bounds=upper_ci_bounds,
        bse_bounds=bse_bounds)
