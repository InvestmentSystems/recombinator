from enum import Enum
from fractions import Fraction
import math
import numpy as np
import typing as tp


def calculate_number_of_blocks(sample_length: int,
                               block_length: int,
                               overhang: bool = False) -> int:
    """
    This function calculates the number of blocks contained
    in a sample with a given block length.

    Args:
        sample_length: the lenght of the sample to be split into blocks
        block_length: the length of each block
        overhang: whether the end of the sample that would not fit another
                  block should be cut off or combined with elements from
                  the beginning of the sample to form another block.

    Returns: number of blocks
    """

    if overhang:
        return math.ceil(sample_length / block_length)
    else:
        return math.floor(sample_length / block_length)


def number_of_permutations(sub_sample_length: int,
                           block_length: int,
                           number_of_blocks: int,
                           replacement: bool = False) -> int:
    """
    Computes the number of possible permutations of a certain number of
    blocks with a given length in creating sub-samples of a particular
    length if the block order matters.

    Args:
        sub_sample_length: the length of the sub-sample to be generated
        block_length: the length of a single block
        number_of_blocks: the number of blocks to sample from
        replacement: whether to sample a given block more than once

    Returns: number of permutations
    """

    blocks_to_sample = math.ceil(sub_sample_length / block_length)

    if replacement:
        return number_of_blocks ** blocks_to_sample
    else:
        if blocks_to_sample > number_of_blocks:
            raise ValueError(
                f'The number of blocks to sample ({blocks_to_sample}) to '
                f'create a subsample of length {sub_sample_length} without '
                f'replacement exceeds the number of blocks available '
                f'({number_of_blocks}).')

        n_permutations = Fraction(math.factorial(number_of_blocks),
                            math.factorial(number_of_blocks - blocks_to_sample))
        return int(n_permutations)


def number_of_combinations(sub_sample_length: int,
                           block_length: int,
                           number_of_blocks: int,
                           replacement: bool = False) -> int:
    """
    Computes the number of possible combinations of a certain number of
    blocks with a given length in creating sub-samples of a particular
    length if the block order does not matter.

    Args:
        sub_sample_length: the length of the sub-sample to be generated
        block_length: the length of a single block
        number_of_blocks: the number of blocks to sample from
        replacement: whether to sample a given block more than once

    Returns: number of combinations
    """

    blocks_to_sample = math.ceil(sub_sample_length / block_length)

    if replacement:
        n_combinations \
            = Fraction(
                int(math.factorial(blocks_to_sample + number_of_blocks - 1)),
                math.factorial(blocks_to_sample)
                * math.factorial(number_of_blocks - 1))
        return int(n_combinations)
    else:
        n_combinations \
            = Fraction(math.factorial(number_of_blocks),
                       math.factorial(number_of_blocks - blocks_to_sample)
                       * math.factorial(blocks_to_sample))
        return int(n_combinations)


# @numba.njit
def _verify_shape_of_bootstrap_input_data_and_get_dimensions(x: np.ndarray) \
        -> tp.Tuple[int, int]:
    if x.ndim == 1:
        # the input data is a one-dimensional time-series
        T = len(x)
        k = 1
    elif x.ndim == 2:
        # the data is a multi-dimensional time-series
        T, k = x.shape
    else:
        raise ValueError('The argument y must be a NumPy array with one '
                         '(for a univariate time-series) or two axes '
                         '(for a multi-variate time-series).')

    return T, k


def _grab_sub_samples_from_indices(x: np.ndarray,
                                   u: np.ndarray) -> np.ndarray:
    """
    This function selects sub-samples of the data array y at the indices in u.
    There are B replications.
    Args:
        x: a NumPy array of shape (T, k) or (T, )
        u: a NumPy array of shape (B, T) of indices into the data array y

    Returns:

    """

    if x.ndim == 1:
        return x[u]
    else:
        return x[u, :]


class BlockBootstrapType(Enum):
    MOVING_BLOCK = 0
    CIRCULAR_BLOCK = 1
    STATIONARY = 2
    TAPERED_BLOCK = 3


def _verify_block_bootstrap_arguments_internal(
        x: np.ndarray,
        block_length: tp.Union[int, float],
        replications: int,
        replace: bool,
        stationary_bootstrap: bool = False,
        sub_sample_length: tp.Optional[int] = None) -> None:
    """
    This function checks that the arguments passed to block bootstrap functions
    satisfy certain constraints. If a violation is detected, an exception is
    raised.

    Args:
        x: the source data array
        block_length: the length of each block to sample
        replications: the number of samples to bootstrap
        replace: whether blocks can appear more than once in a sample
        stationary_bootstrap: whether this is a stationary bootstrap
        sub_sample_length: the length of each bootstrap sub-sample

    Returns: None

    """
    T, k = _verify_shape_of_bootstrap_input_data_and_get_dimensions(x)

    if stationary_bootstrap:
        if not isinstance(block_length, float) and \
                not isinstance(block_length, int):
            raise ValueError('The block_length must be int or float.')
    else:
        if not isinstance(block_length, int):
            raise ValueError('The block_length must be int.')

    if not sub_sample_length:
        sub_sample_length = T

    if block_length <= 0:
        raise ValueError('The argument block_length must be strictly positive.')

    if replications <= 0:
        raise ValueError('The argument replications must be strictly positive.')

    if sub_sample_length <= 0:
        raise ValueError('The argument sub_sample_length must be strictly '
                         'positive.')

    if block_length > T:
        raise ValueError(
            'The argument block_length must not exceed the size of the data.')

    if sub_sample_length > T:
        if not replace or stationary_bootstrap:
            raise ValueError(f'The argument '
                             f'sub_sample_length={sub_sample_length} must not '
                             f'exceed the length of the data={T}.')


def _verify_block_bootstrap_arguments(
        x: np.ndarray,
        block_length: tp.Union[int, float],
        replications: int,
        replace: bool,
        bootstrap_type: BlockBootstrapType,
        sub_sample_length: tp.Optional[int] = None) -> None:
    """
    This function checks that the arguments passed to block bootstrap functions
    satisfy certain constraints. If a violation is detected, an exception is
    raised.

    Args:
        x: the source data array
        block_length: the length of each block to sample
        replications: the number of samples to bootstrap
        replace: whether blocks can appear more than once in a sample
        bootstrap_type: see class BlockBootstrapType
        sub_sample_length: the length of each bootstrap sub-sample

    Returns: None

    """
    stationary_bootstrap = (bootstrap_type is BlockBootstrapType.STATIONARY)
    _verify_block_bootstrap_arguments_internal(
        x,
        block_length=block_length,
        replications=replications,
        replace=replace,
        stationary_bootstrap=stationary_bootstrap,
        sub_sample_length=sub_sample_length)


def _verify_iid_bootstrap_arguments(
                x: np.ndarray,
                replications: int,
                replace: bool,
                sub_sample_length: tp.Optional[int] = None) -> None:
    _verify_block_bootstrap_arguments(
                        x=x,
                        block_length=1,
                        replications=replications,
                        replace=replace,
                        bootstrap_type=BlockBootstrapType.MOVING_BLOCK,
                        sub_sample_length=sub_sample_length)


def _generate_block_start_indices_and_successive_indices(sample_length: int,
                                                         block_length: int,
                                                         circular: bool,
                                                         successive_3d: bool) \
        -> tp.Tuple[np.ndarray, np.ndarray]:
    block_start_indices = list(range(0, sample_length, block_length))
    if not circular:
        if max(block_start_indices) + block_length >= sample_length:
            block_start_indices = block_start_indices[:-1]

    # generate a 1-d array containing the sequence of integers from
    # 0 to block_length-1 with shape (1, block_length)
    if successive_3d:
        successive_indices \
            = np.arange(block_length, dtype=int).reshape((1, 1, block_length))
    else:
        successive_indices \
            = np.arange(block_length, dtype=int).reshape((1, block_length))

    return np.array(block_start_indices), successive_indices