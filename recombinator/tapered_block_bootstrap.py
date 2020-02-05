import numba
import numpy as np
import typing as tp

from .numba_rng_tools import rng_link
from .utilities import (
    _verify_shape_of_bootstrap_input_data_and_get_dimensions,
    _grab_sub_samples_from_indices,
    _verify_block_bootstrap_arguments,
    BlockBootstrapType,
    _generate_block_start_indices_and_successive_indices
)


@numba.njit
def trapezoid(t: float, c: float) -> float:
    """
    Function computes a trapezoid on [0, 1]
    Args:
        t: real valued argument
        c: parameter indicating the beginning of the flat part

    Returns:
        function value of the trapezoid with parameter c evaluated at t

    """
    if 0 <= t < c:
        return t/c
    elif c <= t < 1-c:
        return 1
    elif 1-c <= t < 1:
        return (1-t)/c
    else:
        return 0.0


@numba.njit
def compute_weights(block_length: int, c: float = 0.43) \
        -> tp.Tuple[np.ndarray, np.ndarray]:
    """
    This function computes the weights to rescale the samples in a tapered block
    bootstrap from the demeaned time-series. The unscaled weights represent the
    naive rescaling. The rescaled weights are chosen such that the variance of
    the bootstrapped  time-series matches that of the data and should be used
    for rescaling the bootstrapped samples.
    The procedure assumes a trapezoidal window.

    Args:
        block_length: the block length
        c: partameter describing the beginning of the flat part of the
           trapezoid.

    Returns:
        a Tuple of unscaled_weights and scaled_weights

    """
    js = list(range(1, block_length+1))
    unscaled_weights \
        = np.array([trapezoid((j - 0.5)/block_length, c) for j in js])
    l2_norm_of_weights = np.sqrt(np.sum(unscaled_weights**2))
    scaled_weights \
        = unscaled_weights * np.sqrt(block_length) / l2_norm_of_weights
    return unscaled_weights, scaled_weights


@rng_link()
@numba.njit
def _tapered_block_bootstrap_internal(block_length: int,
                                      replications: int,
                                      sub_sample_length: int,
                                      T: int,
                                      k: int,
                                      y: np.ndarray,
                                      replace: bool,
                                      link_rngs: bool) \
        -> np.ndarray:
    # compute the number of blocks (k in the original paper)
    number_of_blocks = int(np.ceil(sub_sample_length / block_length))

    # length of the output samples
    output_length = number_of_blocks * block_length

    # compute weights from window function
    unscaled_weights, scaled_weights = compute_weights(block_length)

    # allocate output array
    y_star = np.zeros((replications, output_length, k), dtype=y.dtype)

    block_start_indices = np.arange(np.int32(T - block_length))

    for b in range(replications):
        for m in range(number_of_blocks):
            u = np.random.choice(block_start_indices, replace=replace)
            for j in range(block_length):
                scaled_weight = scaled_weights[j]
                unweighted_observation = y[u + j, :]
                weighted_observation = scaled_weight * unweighted_observation
                y_star[b, m * block_length + j, :] = weighted_observation

    return y_star


def tapered_block_bootstrap(
        x: np.ndarray,
        block_length: int,
        replications: int,
        sub_sample_length: tp.Optional[int] = None,
        replace: bool = True,
        link_rngs: bool = True) \
        -> np.ndarray:
    """
    This function creates samples from a data series using the tapered block
    bootstrap. It samples overlapping blocks of fixed length. Observations are
    first demeaned and then rescaled according to a window function. After the
    simulation, the mean is added back to each observation.

    Args:
        x: input data, a NumPy array with one axis
        block_length: the block length
        replications: the number of bootstrap samples to generate
        sub_sample_length: length of the sub-samples to generate
        replace: whether to sample the same block more than once in a single
                 replication
        link_rngs: whether to synchronize the states of Numba's and Numpy's
                   random number generators

    Returns: a NumPy array with shape (replications, int(len(x)/block_length))
             of bootstrapped sub-samples
    """

    # length of the data sample
    # T = len(x)
    T, k = _verify_shape_of_bootstrap_input_data_and_get_dimensions(x)

    if sub_sample_length is None:
        sub_sample_length = T

    _verify_block_bootstrap_arguments(
        x=x,
        block_length=block_length,
        replications=replications,
        replace=replace,
        bootstrap_type=BlockBootstrapType.TAPERED_BLOCK,
        sub_sample_length=sub_sample_length)

    if block_length > T:
        raise ValueError(
            'The argument block_length must not exceed the size of the data.')

    # compute the mean of x
    x_bar = np.mean(x, axis=0)

    # demean observations
    y = x - x_bar
    if k == 1:
        y = y.reshape((T, k, 1))

    # compute the number of blocks (k in the original paper)
    number_of_blocks = int(np.ceil(sub_sample_length / block_length))

    # length of the output samples
    output_length = number_of_blocks * block_length

    y_star = _tapered_block_bootstrap_internal(
                block_length=block_length,
                replications=replications,
                sub_sample_length=sub_sample_length,
                T=T,
                k=k,
                y=y,
                replace=replace,
                link_rngs=link_rngs)

    if k == 1:
        y_star = y_star.reshape((replications, output_length))

    # add back the mean
    x_star = y_star + x_bar

    # cut off observations exceeding the sub-sample length
    x_star = x_star[:, :sub_sample_length]

    return x_star


def tapered_block_bootstrap_vectorized(
        x: np.ndarray,
        block_length: int,
        replications: int,
        sub_sample_length: tp.Optional[int] = None,
        replace: bool = True,
        num_pack=np,
        choice=np.random.choice) \
        -> np.ndarray:
    """
    This function creates samples from a data series using the tapered block
    bootstrap. It samples overlapping blocks of fixed length. Observations are
    first demeaned and then rescaled according to a window function. After the
    simulation, the mean is added back to each observation.

    Args:
        x: Input data, a NumPy array with one or two axes. If y has two axes,
           the Txk array is interpreted as a multidimensional time-series with T
           observations and k variables.
        block_length: the block length
        replications: the number of bootstrap samples to generate
        sub_sample_length: length of the sub-samples to generate
        replace: whether to sample the same block more than once in a single
                 replication
        num_pack: a module compatible with NumPy (function uses vstack and sort)
        choice: a function compatible with numpy.random.choice

    Returns: a NumPy array with shape (replications, int(len(x)/block_length))
             of bootstrapped sub-samples

    """
    T, k = _verify_shape_of_bootstrap_input_data_and_get_dimensions(x)

    if sub_sample_length is None:
        sub_sample_length = T

    _verify_block_bootstrap_arguments(
                x=x,
                block_length=block_length,
                replications=replications,
                replace=replace,
                bootstrap_type=BlockBootstrapType.TAPERED_BLOCK,
                sub_sample_length=sub_sample_length)

    if block_length > T:
        raise ValueError(
            'The argument block_length must not exceed the size of the data.')

    # compute the mean of x
    x_bar = num_pack.mean(x, axis=0)

    # demean observations
    y = x - x_bar

    # compute weights from window function
    unscaled_weights, scaled_weights = compute_weights(block_length)

    # compute the number of blocks (k in the original paper)
    number_of_blocks = int(np.ceil(sub_sample_length / block_length))

    # replicate scaled weights
    replicated_scaled_weights = num_pack.tile(num_pack.array(scaled_weights),
                                              number_of_blocks)

    if k == 1:
        replicated_scaled_weights = replicated_scaled_weights.reshape((1, -1))
    else:
        replicated_scaled_weights \
            = replicated_scaled_weights.reshape((1, -1, 1))

    block_start_indices = num_pack.array(np.arange(np.int32(T - block_length)))
    # generate a 1-d array containing the sequence of integers from
    # 0 to block_length-1 with shape (1, block_length)
    successive_indices \
        = num_pack.array(np.arange(block_length, dtype=int)
                         .reshape((1, 1, block_length)))

    # generate a random array of block start indices with shape
    # (np.ceil(T/block_length), 1)
    indices = choice(block_start_indices,
                     size=(number_of_blocks,
                           replications,
                           1),
                     replace=replace)

    # add successive indices to starting indices
    indices = (indices + successive_indices)

    # transform to col vector and and remove excess
    indices = indices.reshape((replications, -1))

    # y-star sample simulation
    y_star = _grab_sub_samples_from_indices(y, indices)
    y_star *= replicated_scaled_weights

    # cut off observations exceeding the sub-sample length
    y_star = y_star[:, :sub_sample_length]

    # add back the mean
    x_star = y_star + x_bar

    return x_star
