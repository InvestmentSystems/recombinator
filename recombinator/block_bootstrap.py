import math
import numba
import numpy as np
import typing as tp

from .utilities import \
    _verify_shape_of_bootstrap_input_data_and_get_dimensions, \
    _grab_sub_samples_from_indices, \
    _verify_block_bootstrap_arguments, \
    BlockBootstrapType


# ------------------------------------------------------------------------------
# stationary bootstrap
@numba.njit
def _stationary_bootstrap_loop(block_length: float,
                               replications: int,
                               sub_sample_length: int,
                               u: np.ndarray,
                               T: int) -> np.ndarray:
    """
    This function implements the inner loop for stationary bootstraps. It is JIT
    compiled via Numba for performance.

    Args:
        block_length: the average length of each block in a stationary bootstrap
        replications: the number of (sub-)samples to generate
        sub_sample_length: length of the sub-samples to generate
        u: an integer Numpy array with shape (replications, sub_sample_length)
           containing (random) indices of the the beginning of the first block
           in the first column.
        T: the length of the data series that is sub-sampled

    Returns: an integer Numpy array with shape (replications, sub_sample_length)
             containing the index into the source data series for each element
             of each sub-sample.
    """

    for b in range(replications):
        for t in range(1, sub_sample_length):
            if np.random.rand() < 1.0 / block_length:
                # end current block and randomly pick start index for next block
                u[b, t] = np.ceil(T * np.random.rand())
            else:
                # continue current block for another time-step
                u[b, t] = u[b, t - 1] + 1

    return u


def stationary_bootstrap(x: np.ndarray,
                         block_length: float,
                         replications: int,
                         sub_sample_length: tp.Optional[int] = None) \
        -> np.ndarray:
    """
    This function creates sub-samples from a data series via stationary
    bootstrapping.

    Args:
        x: Input data, a NumPy array with one or two axes. If y has two axes,
           the Txk array is interpreted as a multidimensional time-series with T
           observations and k variables.
        block_length: the block length
        replications: the number of (sub-)samples to generate
        sub_sample_length: length of the sub-samples to generate

    Returns: a NumPy array with shape (replications, sub_sample_length) of
             bootstrapped sub-samples
    """

    _verify_block_bootstrap_arguments(
        x=x,
        block_length=block_length,
        replications=replications,
        replace=True,
        bootstrap_type=BlockBootstrapType.STATIONARY,
        sub_sample_length=sub_sample_length)

    T, k = _verify_shape_of_bootstrap_input_data_and_get_dimensions(x)

    if not sub_sample_length:
        sub_sample_length = T

    # repeat the input series to allow blocks to wrap around in a circular
    # fashion
    if x.ndim == 1:
        x = np.hstack((x, x))
    else:
        x = np.vstack((x, x))

    # allocate array for the indices into the source array
    u = np.zeros((replications, sub_sample_length), dtype=np.int)

    # randomly initialize the beginning of the first block in each sub-sample
    u[:, 0] = np.random.randint(T, size=(replications,))

    # Loop over B bootstraps
    u = _stationary_bootstrap_loop(block_length=block_length,
                                   replications=replications,
                                   sub_sample_length=sub_sample_length,
                                   u=u,
                                   T=T)

    # return sub-samples
    return _grab_sub_samples_from_indices(x, u)


# ------------------------------------------------------------------------------
# moving block and circular bootstrap - Numba versions
@numba.njit
def _general_block_bootstrap_loop(block_length: int,
                                  replications: int,
                                  block_start_indices: np.ndarray,
                                  successive_indices: np.ndarray,
                                  sub_sample_length: tp.Optional[int] = None,
                                  replace: bool = True) -> np.ndarray:
    """
    This function implements the inner loop for moving block or circular block
    bootstrapping. It is JIT compiled via Numba for performance.

    Args:
        block_length: the block length
        replications: the number of (sub-)samples to generate
        block_start_indices: a one-dimensional Numpy array containing the first
                             index of each block in into the array y
        successive_indices: an integer Numpy array of shape (1, block_length)
                           containing the range(0, block_length)
        sub_sample_length: length of the sub-samples to generate
        replace: whether to sample the same block more than once in a single
                 replication

    Returns: an integer Numpy array with shape (replications, sub_sample_length)
             containing the index into the source data series for each element
             of each sub-sample.

    """
    # generate array of indeces into the original time series that describe
    # the composition of the generate sub-samples
    u = np.zeros((replications, sub_sample_length), dtype=np.int32)

    # loop over replications bootstraps
    for b in range(replications):
        # generate a random array of block start indices with shape
        # (np.ceil(sub_sample_length/block_length), 1)
        u_tmp \
            = np.random.choice(block_start_indices,
                               size=(math.ceil(sub_sample_length
                                               / block_length),
                                     1),
                               replace=replace)

        # add successive indices to starting indices
        u_tmp = (u_tmp + successive_indices)

        # transform to col vector and and remove excess
        u[b, :] = u_tmp.reshape((-1,))[:sub_sample_length]

    return u


def general_block_bootstrap(x: np.ndarray,
                            block_length: int,
                            replications: int,
                            sub_sample_length: tp.Optional[int] = None,
                            replace: bool = True,
                            circular: bool = False) -> np.ndarray:
    """
    This function creates sub-samples from a data series via block based
    bootstrapping using either the circular or moving block scheme.

    Args:
        x: Input data, a NumPy array with one or two axes. If y has two axes,
           the Txk array is interpreted as a multidimensional time-series with T
           observations and k variables.
        block_length: the block size
        replications: the number of (sub-)samples to generate
        sub_sample_length: length of the sub-samples to generate
        replace: whether to sample the same block more than once in a single
                 replication
        circular: whether to use circular or moving block bootstrapping

    Returns: a NumPy array with shape (replications, sub_sample_length) of
             bootstrapped sub-samples
    """

    T, k = _verify_shape_of_bootstrap_input_data_and_get_dimensions(x)

    # if block_length > T:
    #     raise ValueError(
    #         'The argument block_length must not exceed the size of the data.')

    if not sub_sample_length:
        sub_sample_length = T

    if circular:
        bootstrap_type = BlockBootstrapType.CIRCULAR_BLOCK
    else:
        bootstrap_type = BlockBootstrapType.MOVING_BLOCK

    _verify_block_bootstrap_arguments(x=x,
                                      block_length=block_length,
                                      replications=replications,
                                      replace=replace,
                                      bootstrap_type=bootstrap_type,
                                      sub_sample_length=sub_sample_length)

    # generate a list of block start indices
    block_start_indices = list(range(0, T, block_length))

    if not circular:
        if max(block_start_indices) + block_length >= T:
            block_start_indices.pop()
    else:
        # replicate time-series for wrap-around
        if x.ndim == 1:
            x = np.hstack((x, x))
        else:
            x = np.vstack((x, x))

    block_start_indices = np.array(block_start_indices)

    # generate a 1-d array containing the sequence of integers from 0 to m-1
    # with shape (1, block_length)
    successive_indices \
        = np.arange(block_length, dtype=int).reshape((1, block_length))

    u = _general_block_bootstrap_loop(block_length=block_length,
                                      replications=replications,
                                      block_start_indices=block_start_indices,
                                      successive_indices=successive_indices,
                                      sub_sample_length=sub_sample_length,
                                      replace=replace)

    # return sub-samples
    return _grab_sub_samples_from_indices(x, u)


def moving_block_bootstrap(x: np.ndarray,
                           block_length: int,
                           replications: int,
                           sub_sample_length: tp.Optional[int] = None,
                           replace: bool = True) -> np.ndarray:
    """
    This function creates sub-samples from a data series via moving block
    bootstrapping.

    Args:
        x: Input data, a NumPy array with one or two axes. If y has two axes,
           the Txk array is interpreted as a multidimensional time-series with T
           observations and k variables.
        block_length: the block size
        replications: the number of (sub-)samples to generate
        sub_sample_length: length of the sub-samples to generate
        replace: whether to sample the same block more than once in a single
                 replication

    Returns: a NumPy array with shape (replications, sub_sample_length) of
             bootstrapped sub-samples
    """

    return general_block_bootstrap(x=x,
                                   block_length=block_length,
                                   replications=replications,
                                   sub_sample_length=sub_sample_length,
                                   replace=replace,
                                   circular=False)


def circular_block_bootstrap(x: np.ndarray,
                             block_length: int,
                             replications: int,
                             sub_sample_length: tp.Optional[int] = None,
                             replace: bool = True) -> np.ndarray:
    """
    This function creates sub-samples from a data series via circular block
    bootstrapping.

    Args:
        x: Input data, a NumPy array with one or two axes. If y has two axes,
           the Txk array is interpreted as a multidimensional time-series with T
           observations and k variables.
        block_length: the block size
        replications: the number of (sub-)samples to generate
        sub_sample_length: length of the sub-samples to generate
        replace: whether to sample the same block more than once in a single
                 replication

    Returns: a NumPy array with shape (replications, sub_sample_length) of
             bootstrapped sub-samples
    """

    return general_block_bootstrap(x=x,
                                   block_length=block_length,
                                   replications=replications,
                                   sub_sample_length=sub_sample_length,
                                   replace=replace,
                                   circular=True)


# ------------------------------------------------------------------------------
# moving block bootstrap - vectorized version
def moving_block_bootstrap_vectorized(
        x: np.ndarray,
        block_length: int,
        replications: int,
        sub_sample_length: tp.Optional[int] = None) \
        -> np.ndarray:
    """
    This function creates sub-samples from a data series via moving block
    bootstrapping. It relies on a vectorized implementation.

    Args:
        x: Input data, a NumPy array with one or two axes. If y has two axes,
           the Txk array is interpreted as a multidimensional time-series with T
           observations and k variables.
        block_length: the block size
        replications: the number of (sub-)samples to generate
        sub_sample_length: length of the sub-samples to generate

    Returns: a NumPy array with shape (replications, sub_sample_length) of
             bootstrapped sub-samples
    """

    T, k = _verify_shape_of_bootstrap_input_data_and_get_dimensions(x)

    if not sub_sample_length:
        sub_sample_length = T

    _verify_block_bootstrap_arguments(
                    x=x,
                    block_length=block_length,
                    replications=replications,
                    replace=True,
                    bootstrap_type=BlockBootstrapType.MOVING_BLOCK,
                    sub_sample_length=sub_sample_length)

    # generate a list of block start indices
    block_start_indices = list(range(0, T, block_length))
    if max(block_start_indices) + block_length >= T:
        block_start_indices.pop()

    # generate a 1-d array containing the sequence of integers from
    # 0 to block_length-1 with shape (1, block_length)
    successive_indices \
        = np.arange(block_length, dtype=int).reshape((1, 1, block_length))

    # generate a random array of block start indices with shape
    # (np.ceil(sub_sample_length/block_length), 1)
    u = np.random.choice(block_start_indices,
                         size=(math.ceil(sub_sample_length
                                         / block_length),
                               replications, 1))

    # add successive indices to starting indices
    u = (u + successive_indices)

    # transform to col vector and and remove excess
    u = u.reshape((replications, -1))[:, :sub_sample_length]

    # return sub-samples
    return _grab_sub_samples_from_indices(x, u)


# ------------------------------------------------------------------------------
# circular block bootstrap - vectorized version
def circular_block_bootstrap_vectorized(
        x: np.ndarray,
        block_length: int,
        replications: int,
        sub_sample_length: tp.Optional[int] = None) \
        -> np.ndarray:
    """
    This function creates sub-samples from a data series via circular block
    bootstrapping.

    Args:
        x: Input data, a NumPy array with one or two axes. If y has two axes,
           the Txk array is interpreted as a multidimensional time-series with T
           observations and k variables.
        block_length: the block size
        replications: the number of (sub-)samples to generate
        sub_sample_length: length of the sub-samples to generate

    Returns: a NumPy array with shape (replications, sub_sample_length) of
             bootstrapped sub-samples
    """
    T, k = _verify_shape_of_bootstrap_input_data_and_get_dimensions(x)

    if not sub_sample_length:
        sub_sample_length = T

    _verify_block_bootstrap_arguments(
                x=x,
                block_length=block_length,
                replications=replications,
                replace=True,
                bootstrap_type=BlockBootstrapType.CIRCULAR_BLOCK,
                sub_sample_length=sub_sample_length)

    # replicate time-series for wrap-around
    if x.ndim == 1:
        x = np.hstack((x, x))
    else:
        x = np.vstack((x, x))

    # generate a list of block start indices
    block_start_indices = list(range(T))

    # generate a 1-d array containing the sequence of integers from
    # 0 to block_length-1 with shape (1, block_length)
    successive_indices \
        = np.arange(block_length, dtype=int).reshape((1, 1, block_length))

    # generate a random array of block start indices with shape
    # (np.ceil(T/block_length), 1)
    u = np.random.choice(block_start_indices,
                         size=(math.ceil(sub_sample_length / block_length),
                               replications,
                               1))

    # add successive indices to starting indices
    u = (u + successive_indices)

    # transform to col vector and and remove excess
    u = u.reshape((replications, -1))[:, :sub_sample_length]

    # return sub-samples
    return _grab_sub_samples_from_indices(x, u)
