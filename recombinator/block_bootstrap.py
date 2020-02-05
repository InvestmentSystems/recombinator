import math
import numba
import numpy as np
import typing as tp

from .numba_rng_tools import rng_link
from .utilities import (
    _generate_block_start_indices_and_successive_indices,
    _verify_shape_of_bootstrap_input_data_and_get_dimensions,
    _grab_sub_samples_from_indices,
    _verify_block_bootstrap_arguments,
    BlockBootstrapType
)


# ------------------------------------------------------------------------------
# stationary bootstrap
@rng_link()
@numba.njit
def _stationary_bootstrap_loop(block_length: float,
                               replications: int,
                               sub_sample_length: int,
                               indices: np.ndarray,
                               T: int,
                               link_rngs: bool) -> np.ndarray:
    """
    This function implements the inner loop for stationary bootstraps. It is JIT
    compiled via Numba for performance.

    Args:
        block_length: the average length of each block in a stationary bootstrap
        replications: the number of (sub-)samples to generate
        sub_sample_length: length of the sub-samples to generate
        indices: an integer Numpy array with shape (replications, sub_sample_length)
           containing (random) indices of the the beginning of the first block
           in the first column.
        T: the length of the data series that is sub-sampled
        link_rngs: whether to synchronize the states of Numba's and Numpy's
                   random number generators

    Returns: an integer Numpy array with shape (replications, sub_sample_length)
             containing the index into the source data series for each element
             of each sub-sample.
    """

    for b in range(replications):
        for t in range(1, sub_sample_length):
            if np.random.rand() < 1.0 / block_length:
                # end current block and randomly pick start index for next block
                indices[b, t] = np.ceil(T * np.random.rand())
            else:
                # continue current block for another time-step
                indices[b, t] = indices[b, t - 1] + 1

    return indices


def stationary_bootstrap(x: np.ndarray,
                         block_length: float,
                         replications: int,
                         sub_sample_length: tp.Optional[int] = None,
                         link_rngs: bool = True) \
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
        link_rngs: whether to synchronize the states of Numba's and Numpy's
                   random number generators

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

    T, _ = _verify_shape_of_bootstrap_input_data_and_get_dimensions(x)

    if not sub_sample_length:
        sub_sample_length = T

    # repeat the input series to allow blocks to wrap around in a circular
    # fashion
    if x.ndim == 1:
        x = np.hstack((x, x))
    else:
        x = np.vstack((x, x))

    # allocate array for the indices into the source array
    indices = np.zeros((replications, sub_sample_length), dtype=np.int)

    # randomly initialize the beginning of the first block in each sub-sample
    indices[:, 0] = np.random.randint(T, size=(replications,))

    # Loop over B bootstraps
    indices = _stationary_bootstrap_loop(block_length=block_length,
                                         replications=replications,
                                         sub_sample_length=sub_sample_length,
                                         indices=indices,
                                         T=T,
                                         link_rngs=link_rngs)

    # return sub-samples
    return _grab_sub_samples_from_indices(x, indices)


# ------------------------------------------------------------------------------
# moving block and circular bootstrap - Numba versions
@rng_link()
@numba.njit
def _general_block_bootstrap_loop(block_length: int,
                                  replications: int,
                                  block_start_indices: np.ndarray,
                                  successive_indices: np.ndarray,
                                  sub_sample_length: tp.Optional[int],
                                  replace: bool,
                                  link_rngs: bool) -> np.ndarray:
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
        link_rngs: whether to synchronize the states of Numba's and Numpy's
                   random number generators

    Returns: an integer Numpy array with shape (replications, sub_sample_length)
             containing the index into the source data series for each element
             of each sub-sample.

    """
    # generate array of indeces into the original time series that describe
    # the composition of the generate sub-samples
    indices = np.zeros((replications, sub_sample_length), dtype=np.int32)

    # loop over replications bootstraps
    for b in range(replications):
        # generate a random array of block start indices with shape
        # (np.ceil(sub_sample_length/block_length), 1)
        tmp_indices \
            = np.random.choice(block_start_indices,
                               size=(math.ceil(sub_sample_length
                                               / block_length),
                                     1),
                               replace=replace)

        # ToDo: Move that outside the loop
        # add successive indices to starting indices
        tmp_indices = (tmp_indices + successive_indices)

        # transform to col vector and and remove excess
        indices[b, :] = tmp_indices.reshape((-1,))[:sub_sample_length]

    return indices


def _general_block_bootstrap(x: np.ndarray,
                             block_length: int,
                             replications: int,
                             sub_sample_length: tp.Optional[int] = None,
                             replace: bool = True,
                             circular: bool = False,
                             link_rngs: bool = True) -> np.ndarray:
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
        link_rngs: whether to synchronize the states of Numba's and Numpy's
                   random number generators

    Returns: a NumPy array with shape (replications, sub_sample_length) of
             bootstrapped sub-samples
    """
    T, _ = _verify_shape_of_bootstrap_input_data_and_get_dimensions(x)

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

    block_start_indices, successive_indices \
        = _generate_block_start_indices_and_successive_indices(
                    sample_length=T,
                    block_length=block_length,
                    circular=circular,
                    successive_3d=False)

    if circular:
        # replicate time-series for wrap-around
        if x.ndim == 1:
            x = np.hstack((x, x))
        else:
            x = np.vstack((x, x))

    indices \
        = _general_block_bootstrap_loop(block_length=block_length,
                                        replications=replications,
                                        block_start_indices=block_start_indices,
                                        successive_indices=successive_indices,
                                        sub_sample_length=sub_sample_length,
                                        replace=replace,
                                        link_rngs=link_rngs)

    # return sub-samples
    return _grab_sub_samples_from_indices(x, indices)


def moving_block_bootstrap(x: np.ndarray,
                           block_length: int,
                           replications: int,
                           sub_sample_length: tp.Optional[int] = None,
                           replace: bool = True,
                           link_rngs: bool = True) -> np.ndarray:
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
        link_rngs: whether to synchronize the states of Numba's and Numpy's
                   random number generators

    Returns: a NumPy array with shape (replications, sub_sample_length) of
             bootstrapped sub-samples
    """

    return _general_block_bootstrap(x=x,
                                    block_length=block_length,
                                    replications=replications,
                                    sub_sample_length=sub_sample_length,
                                    replace=replace,
                                    circular=False,
                                    link_rngs=link_rngs)


def circular_block_bootstrap(x: np.ndarray,
                             block_length: int,
                             replications: int,
                             sub_sample_length: tp.Optional[int] = None,
                             replace: bool = True,
                             link_rngs: bool = True) -> np.ndarray:
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
        link_rngs: whether to synchronize the states of Numba's and Numpy's
                   random number generators

    Returns: a NumPy array with shape (replications, sub_sample_length) of
             bootstrapped sub-samples
    """

    return _general_block_bootstrap(x=x,
                                    block_length=block_length,
                                    replications=replications,
                                    sub_sample_length=sub_sample_length,
                                    replace=replace,
                                    circular=True,
                                    link_rngs=link_rngs)


def _generalized_block_bootstrap_vectorized(
        x: np.ndarray,
        block_length: int,
        replications: int,
        circular: bool,
        sub_sample_length: tp.Optional[int] = None,
        replace: bool = True,
        num_pack=np,
        choice=np.random.choice) \
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
        replace: whether to sample the same block more than once in a single
                 replication
        num_pack: a module compatible with NumPy (function uses vstack and sort)
        choice: a function compatible with numpy.random.choice

    Returns: a NumPy array with shape (replications, sub_sample_length) of
             bootstrapped sub-samples
    """

    T, _ = _verify_shape_of_bootstrap_input_data_and_get_dimensions(x)

    if not sub_sample_length:
        sub_sample_length = T

    if circular:
        bootstrap_type = BlockBootstrapType.CIRCULAR_BLOCK
    else:
        bootstrap_type = BlockBootstrapType.MOVING_BLOCK

    _verify_block_bootstrap_arguments(
                x=x,
                block_length=block_length,
                replications=replications,
                replace=replace,
                bootstrap_type=bootstrap_type,
                sub_sample_length=sub_sample_length)

    if circular:
        # replicate time-series for wrap-around
        if x.ndim == 1:
            x = num_pack.hstack((x, x))
        else:
            x = num_pack.vstack((x, x))

    block_start_indices, successive_indices \
        = _generate_block_start_indices_and_successive_indices(
                    sample_length=T,
                    block_length=block_length,
                    circular=circular,
                    successive_3d=True)

    # print(f'block_start_indices.shape before = {block_start_indices.shape}')
    # print(f'successive_indices.shape before = {successive_indices.shape}')
    block_start_indices = num_pack.array(block_start_indices)
    successive_indices = num_pack.array((successive_indices))
    # print(f'block_start_indices.shape after = {block_start_indices.shape}')
    # print(f'successive_indices.shape after = {successive_indices.shape}')

    # generate a random array of block start indices with shape
    # (np.ceil(T/block_length), 1)
    indices = choice(block_start_indices,
                     size=(math.ceil(sub_sample_length
                                     / block_length),
                           replications,
                           1),
                     replace=replace)

    # add successive indices to starting indices
    indices = (indices + successive_indices)
    # print(indices)
    # print(f'indices.shape = {indices.shape}')

    # transform to col vector and and remove excess
    indices = indices.reshape((replications, -1))[:, :sub_sample_length]
    # print(indices)

    # return sub-samples
    return _grab_sub_samples_from_indices(x, indices)


def circular_block_bootstrap_vectorized(
        x: np.ndarray,
        block_length: int,
        replications: int,
        sub_sample_length: tp.Optional[int] = None,
        replace: bool = True,
        num_pack=np,
        choice=np.random.choice) \
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
        replace: whether to sample the same block more than once in a single
                 replication
        num_pack: a module compatible with NumPy (function uses vstack and sort)
        choice: a function compatible with numpy.random.choice

    Returns: a NumPy array with shape (replications, sub_sample_length) of
             bootstrapped sub-samples
    """

    return _generalized_block_bootstrap_vectorized(
                x=x,
                block_length=block_length,
                replications=replications,
                circular=True,
                sub_sample_length=sub_sample_length,
                replace=replace,
                num_pack=num_pack,
                choice=choice)


def moving_block_bootstrap_vectorized(
        x: np.ndarray,
        block_length: int,
        replications: int,
        sub_sample_length: tp.Optional[int] = None,
        replace: bool = True,
        num_pack=np,
        choice=np.random.choice) \
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
        replace: whether to sample the same block more than once in a single
                 replication
        num_pack: a module compatible with NumPy (function uses vstack and sort)
        choice: a function compatible with numpy.random.choice


    Returns: a NumPy array with shape (replications, sub_sample_length) of
             bootstrapped sub-samples
    """

    return _generalized_block_bootstrap_vectorized(
                x=x,
                block_length=block_length,
                replications=replications,
                circular=False,
                sub_sample_length=sub_sample_length,
                replace=replace,
                num_pack=num_pack,
                choice=choice)
