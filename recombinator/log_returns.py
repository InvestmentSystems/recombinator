import numpy as np
import pandas as pd
import typing as tp

from .iid_bootstrap import iid_bootstrap
from .block_bootstrap import (
    moving_block_bootstrap,
    circular_block_bootstrap,
    stationary_bootstrap,
    BlockBootstrapType
)

from .tapered_block_bootstrap import (
    tapered_block_bootstrap,
    tapered_block_bootstrap_vectorized
)


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def calculate_number_of_complete_blocks(sample_length: int,
                                        block_length: int) -> int:
    """
    This function computes the number of blocks that fit into a sample of a
    given length.
    """

    return int(sample_length / block_length)


def _get_number_of_observations_from_1d_or_2d_array(a: np.ndarray) -> int:
    """
    This function retrieves the number of observations in a numpy array with
    shape (T, ) or (k, T), where k is the dimensionality of the data.

    Args:
        a: numpy data array

    Returns: number of observations

    """
    if a.ndim == 1:
        T = len(a)
    elif a.ndim == 2:
        _, T = a.shape
    else:
        raise ValueError('The data array must be a NumPy array with one or '
                         'two axes')

    return T


def remove_incomplete_blocks(log_returns: np.ndarray,
                             number_of_observations_to_aggregate: int) \
        -> np.ndarray:
    """
    This function takes a NumPy array of observations and a positive integer of
    observations to be aggregated. It cuts off the remaining observations at the
    end that do not form a complete block.
    Args:
        log_returns: a numpy with shape (T, ) or (k, T)
        number_of_observations_to_aggregate: the block length

    Returns: a numpy array with the remaining observations cut off

    """
    T \
        = _get_number_of_observations_from_1d_or_2d_array(log_returns)

    number_of_complete_blocks \
        = calculate_number_of_complete_blocks(
            T,
            number_of_observations_to_aggregate)
    end_index = number_of_complete_blocks * number_of_observations_to_aggregate

    if log_returns.ndim == 1:
        return log_returns[:end_index]
    elif log_returns.ndim == 2:
        return log_returns[:, :end_index]
    else:
        assert False


def aggregate_log_returns(log_returns: np.ndarray,
                          number_of_observations_to_aggregate: int,
                          rolling: bool = False,
                          verbose: bool = True) \
        -> tp.Tuple[np.ndarray, np.ndarray]:
    """
    Args:
        log_returns: a numpy array of log returns with shape (T, ) or (k, T),
                     where k is number of securities or portfolios
        number_of_observations_to_aggregate: the length of blocks to aggregate
        rolling: If True: the step size to move the summation window by is one
                          observation.
                 If False: the step size is number_of_observations_to_aggregate
                           resulting in sums of returns from non-overlapping
                           observations.
        verbose: Whether to print the number of observations that were cut off
                 from the sample at the end because they were not sufficient to
                 form another aggregation period.
                 This is 0 if rolling equals True.

    Returns:
        Tuple of (aggregated_log_returns, retained_log_returns).
        The aggregated_log_returns are the sums of
        'number_of_observations_to_aggregate' log returns.
    """

    if number_of_observations_to_aggregate < 1:
        raise ValueError("The number of observations to aggregate must be a "
                         "positive integer.")

    if number_of_observations_to_aggregate == 1:
        return log_returns, log_returns

    n_observations \
        = _get_number_of_observations_from_1d_or_2d_array(log_returns)

    if rolling:
        if log_returns.ndim == 1:
            log_returns_df = pd.Series(log_returns)
            aggregated_log_returns \
                = (log_returns_df
                   .rolling(number_of_observations_to_aggregate)
                   .sum()
                   .dropna()
                   .values)
        elif log_returns.ndim == 2:
            rolling_log_returns \
                = rolling_window(log_returns,
                                 number_of_observations_to_aggregate)
            aggregated_log_returns = rolling_log_returns.sum(axis=2)
        else:
            raise ValueError(
                "The argument log_returns must be a one- or "
                "two-dimensional array")
        retained_log_returns = log_returns
    else:
        number_of_complete_blocks \
            = calculate_number_of_complete_blocks(
                n_observations,
                number_of_observations_to_aggregate)
        end_index \
            = number_of_complete_blocks * number_of_observations_to_aggregate

        if log_returns.ndim == 1:
            # the input is a single time series sample
            retained_log_returns \
                = log_returns[:end_index]
            aggregated_log_returns \
                = (retained_log_returns
                   .reshape(number_of_complete_blocks,
                            number_of_observations_to_aggregate)
                   .sum(axis=1))
        elif log_returns.ndim == 2:
            # the input are many time series samples
            number_of_complete_blocks \
                = calculate_number_of_complete_blocks(
                    n_observations,
                    number_of_observations_to_aggregate)
            retained_log_returns \
                = log_returns[:, :end_index]

            aggregated_log_returns \
                = (retained_log_returns
                   .reshape(-1,
                            number_of_complete_blocks,
                            number_of_observations_to_aggregate)
                   .sum(axis=2))
        else:
            raise ValueError(
                'The argument log_returns must be a NumPy array with '
                'one or two axes')

    if verbose:
        number_of_cut_off_observations \
            = n_observations % number_of_observations_to_aggregate
        print(f'Number of cut off observations '
              f'= {number_of_cut_off_observations}')

    return aggregated_log_returns, retained_log_returns


def aggregate_multi_dimensional_log_returns(
        log_returns: np.ndarray,
        number_of_observations_to_aggregate: int):
    """
    This function takes a numpy array with shape (replications, T, dimensions)
    and sums the T observations in each replication and along each dimension in
    blocks. The function does not discard observations that do not fit a block.

    Args:
        log_returns: a numpy array with shape (replications, T, dimensions)
        number_of_observations_to_aggregate: number of observations to sum

    Returns: a numpy array with shape (replications,
                                       number_of_blocks,
                                       dimension)

    """
    replications, T, dimensions = log_returns.shape

    number_of_complete_blocks \
        = calculate_number_of_complete_blocks(
            sample_length=T,
            block_length=number_of_observations_to_aggregate)

    if number_of_complete_blocks * number_of_observations_to_aggregate != T:
        raise ValueError('T / number_of_observations_to_aggregate must be an '
                         'integer')

    aggregated_log_returns \
        = (log_returns
           .reshape((replications,
                    number_of_complete_blocks,
                    number_of_observations_to_aggregate,
                    dimensions))
           .sum(axis=2))

    return aggregated_log_returns


def _calculate_number_of_observations_to_be_cut_off_at_the_end(
        sample_length: int,
        number_of_observations_to_aggregate: int,
        verbose: bool = False) -> int:
    """
    This function supports the calculation of statistics from a single
    time-series by shifting the window by number_of_observations_to_aggregate
    different integer values to obtain different estimates for the same
    statistic.

    Specifically, the function calculates how many observations need to be
    discarded at the end if number_of_observations_to_aggregate different
    shifted windows are considered.

    Args:
        sample_length: the length of the time-series
        number_of_observations_to_aggregate: the block length
        verbose: whether to print debug information

    Returns:

    """
    if sample_length < 2 * number_of_observations_to_aggregate - 1:
        raise ValueError(
                f'The sample length is too short. At least '
                f'{2 * number_of_observations_to_aggregate - 1} observations'
                f' are required.')

    observations_remaining_at_the_end \
        = sample_length % number_of_observations_to_aggregate

    observations_to_be_cut_off_at_the_end \
        = (observations_remaining_at_the_end + 1) \
        % number_of_observations_to_aggregate

    if verbose:
        print(f'observations remaining at the end '
              f'= {observations_remaining_at_the_end}')
        print(observations_to_be_cut_off_at_the_end)

    return observations_to_be_cut_off_at_the_end


def resample_and_aggregate(
        log_returns: np.ndarray,
        block_length: tp.Union[int, float],
        replications: int,
        number_of_observations_to_aggregate: int,
        block_bootstrap_type: BlockBootstrapType
        = BlockBootstrapType.STATIONARY,
        sub_sample_length: int = None,
        rolling: bool = False,
        verbose: bool = True) -> tp.Tuple:
    """
    This function
    1.) computes aggregated returns from of an original time-series of
    log returns discarding insufficient observations
    2.) and draws new bootstrap samples from those original observations not
    discarded and aggregates these resampled returns

    Args:
        log_returns: a numpy array of log returns with one axis
        block_length: the block length for bootstrapping
        replications: the number of bootstrap samples to simulate
        number_of_observations_to_aggregate: the number of log returns to
                aggregate in a block
        block_bootstrap_type: The block bootstrap method to use
                (see class BlockBootstrapType)
        sub_sample_length: the length of each bootstrap sub-samples to generate
                (optional parameter which defaults to the same length as the
                original time-series)
        rolling: whether to perform time aggregation on a rolling basis or
                whether to use non-overlapping observations

    Returns: numpy arrays of the aggregated log returns from the original
    sample, the aggregated resampled log returns, and the original log returns
    that were not discarded
    """

    # Aggregate log returns in the original sample and obtain a series of
    # retained log returns where the tail end of observations that do not
    # fit an additional block have been cut off.

    aggregated_log_returns, retained_log_returns \
        = aggregate_log_returns(
            log_returns=log_returns,
            number_of_observations_to_aggregate
            =number_of_observations_to_aggregate,
            rolling=rolling,
            verbose=verbose)

    if not sub_sample_length:
        sub_sample_length = len(retained_log_returns)

    # draw sample time-series of log returns of the same length as the
    # original data from retained log returns
    if block_length == 1:
        resampled_log_returns \
            = iid_bootstrap(retained_log_returns,
                            replications=replications,
                            sub_sample_length=sub_sample_length)

    elif block_bootstrap_type == BlockBootstrapType.MOVING_BLOCK:
        resampled_log_returns \
            = moving_block_bootstrap(retained_log_returns,
                                     block_length=block_length,
                                     replications=replications,
                                     sub_sample_length=sub_sample_length)

    elif block_bootstrap_type == BlockBootstrapType.CIRCULAR_BLOCK:
        resampled_log_returns \
            = circular_block_bootstrap(retained_log_returns,
                                       block_length=block_length,
                                       replications=replications,
                                       sub_sample_length=sub_sample_length)

    elif block_bootstrap_type == BlockBootstrapType.STATIONARY:
        resampled_log_returns \
            = stationary_bootstrap(retained_log_returns,
                                   block_length=block_length,
                                   replications=replications,
                                   sub_sample_length=sub_sample_length)
    elif block_bootstrap_type == BlockBootstrapType.TAPERED_BLOCK:
        resampled_log_returns \
            = tapered_block_bootstrap(retained_log_returns,
                                      block_length=block_length,
                                      replications=replications,
                                      sub_sample_length=sub_sample_length)
    else:
        raise ValueError(
            f'block_bootstrap_type {block_bootstrap_type} is not supported')

    # aggregate log-returns in each bootstrapped sample
    aggregated_resampled_log_returns, _ \
        = aggregate_log_returns(
            log_returns=resampled_log_returns,
            number_of_observations_to_aggregate
            =number_of_observations_to_aggregate,
            rolling=rolling,
            verbose=verbose)

    # return the aggregated returns from the bootstrap,
    # the aggregated log returns from the original data,
    # and the retained log returns which the bootstrap is based on
    return (aggregated_resampled_log_returns,
            aggregated_log_returns,
            retained_log_returns)


def resample_and_aggregate_multidimensional_log_returns(
        log_returns: np.ndarray,
        block_length: tp.Union[
            int, float],
        replications: int,
        number_of_observations_to_aggregate: int,
        block_bootstrap_type: BlockBootstrapType
        = BlockBootstrapType.STATIONARY) \
        -> np.ndarray:
    """
    This function resamples a multi-dimensional time-series of log returns and
    sums the resampled returns in blocks.

    Args:
        log_returns: a numpy array of log returns with shape (T, dimension)
        block_length: block length for bootstrap
        replications: the number of bootstrap samples to generate
        number_of_observations_to_aggregate: the number of returns to aggregate
        block_bootstrap_type: The block bootstrap method to use
                              (see class BlockBootstrapType)

    Returns: a numpy array of dimension (replications,
                                         number_of_blocks,
                                         dimension)

    """

    T, dim = log_returns.shape
    number_of_complete_blocks \
        = calculate_number_of_complete_blocks(
            sample_length=T,
            block_length=number_of_observations_to_aggregate)
    number_of_observations_to_retain \
        = number_of_complete_blocks * number_of_observations_to_aggregate

    if block_length == 1:
        resampled_log_returns \
            = iid_bootstrap(log_returns,
                            replications=replications,
                            sub_sample_length=number_of_observations_to_retain)

    elif block_bootstrap_type == BlockBootstrapType.MOVING_BLOCK:
        resampled_log_returns \
            = moving_block_bootstrap(
                log_returns,
                block_length=block_length,
                replications=replications,
                sub_sample_length=number_of_observations_to_retain)

    elif block_bootstrap_type == BlockBootstrapType.CIRCULAR_BLOCK:
        resampled_log_returns \
            = circular_block_bootstrap(
                log_returns,
                block_length=block_length,
                replications=replications,
                sub_sample_length=number_of_observations_to_retain)

    elif block_bootstrap_type == BlockBootstrapType.STATIONARY:
        resampled_log_returns \
            = stationary_bootstrap(
                log_returns,
                block_length=block_length,
                replications=replications,
                sub_sample_length=number_of_observations_to_retain)
    elif block_bootstrap_type == BlockBootstrapType.TAPERED_BLOCK:
        # resampled_log_returns \
        #     = tapered_block_bootstrap(
        #         log_returns,
        #         block_length=block_length,
        #         replications=replications,
        #         sub_sample_length=number_of_observations_to_retain)
        resampled_log_returns \
            = tapered_block_bootstrap_vectorized(
                log_returns,
                block_length=block_length,
                replications=replications,
                sub_sample_length=number_of_observations_to_retain)
    else:
        raise ValueError(
            f'block_bootstrap_type {block_bootstrap_type} is not supported')

    aggregated_resampled_log_returns \
        = aggregate_multi_dimensional_log_returns(
            log_returns=resampled_log_returns,
            number_of_observations_to_aggregate
            =number_of_observations_to_aggregate)

    return aggregated_resampled_log_returns
