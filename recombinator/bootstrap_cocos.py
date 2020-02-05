import cocos.numerics as cn
import typing as tp


from .block_bootstrap import (
    moving_block_bootstrap_vectorized,
    circular_block_bootstrap_vectorized
)

from .iid_bootstrap import (
    iid_bootstrap_vectorized,
    iid_bootstrap_via_choice,
    iid_bootstrap_with_antithetic_resampling
)

from .tapered_block_bootstrap import tapered_block_bootstrap_vectorized


def iid_bootstrap_cocos(x: cn.ndarray,
                        replications: int,
                        sub_sample_length: tp.Optional[int] = None) \
        -> cn.ndarray:
    """
    This function performs an i.i.d. non-balanced bootstrap via
    a vectorized implementation.

    Args:
        x: (n,) or (n, k) dimensional Cocos array of input data
        replications: the number of samples to generate
        sub_sample_length: the length of the bootstrapped samples to generate

    Returns:
        a (replications, n) or (replications, n, k) dimensional Cocos array
    """

    return iid_bootstrap_vectorized(x=x,
                                    replications=replications,
                                    sub_sample_length=sub_sample_length)


def iid_bootstrap_via_choice_cocos(
            x: cn.ndarray,
            replications: int,
            sub_sample_length: tp.Optional[int] = None) \
        -> cn.ndarray:
    """
    This function performs an i.i.d. non-balanced bootstrap via
    NumPy's random choice function.

    Args:
        x: (n,) dimensional Cocos array of input data
        replications: the number of samples to generate
        sub_sample_length: the length of the bootstrapped samples to generate

    Returns:
        a (replications, n) dimensional Cocos array
    """

    return iid_bootstrap_via_choice(x=x,
                                    replications=replications,
                                    sub_sample_length=sub_sample_length)


def iid_bootstrap_with_antithetic_resampling_cocos(
            x: cn.ndarray,
            replications: int,
            sub_sample_length: tp.Optional[int] = None) \
        -> cn.ndarray:
    """
    This function performs an i.i.d. non-balanced bootstrap with antithetic
    sampling.

    Args:
        x: (n,) dimensional Cocos array of input data
        replications: the number of samples to generate
        sub_sample_length: the length of the bootstrapped samples to generate

    Returns:
        a (replications, n) dimensional Cocos array
    """

    return iid_bootstrap_with_antithetic_resampling(
                x=x,
                replications=replications,
                sub_sample_length=sub_sample_length,
                num_pack=cn,
                randint=cn.random.randint)


def circular_block_bootstrap_cocos(
        x: cn.ndarray,
        block_length: int,
        replications: int,
        sub_sample_length: tp.Optional[int] = None,
        replace: bool = True) \
        -> cn.ndarray:
    """
    This function creates sub-samples from a data series via circular block
    bootstrapping.

    Args:
        x: Input data, a Cocos array with one or two axes. If y has two axes,
           the Txk array is interpreted as a multidimensional time-series with T
           observations and k variables.
        block_length: the block size
        replications: the number of (sub-)samples to generate
        sub_sample_length: length of the sub-samples to generate
        replace: whether to sample the same block more than once in a single
                 replication

    Returns: a Cocos array with shape (replications, sub_sample_length) of
             bootstrapped sub-samples
    """

    return circular_block_bootstrap_vectorized(
                x=x,
                block_length=block_length,
                replications=replications,
                sub_sample_length=sub_sample_length,
                replace=replace,
                num_pack=cn,
                choice=cn.random.choice)


def moving_block_bootstrap_cocos(
        x: cn.ndarray,
        block_length: int,
        replications: int,
        sub_sample_length: tp.Optional[int] = None,
        replace: bool = True) \
        -> cn.ndarray:
    """
    This function creates sub-samples from a data series via moving block
    bootstrapping. It relies on a vectorized implementation.

    Args:
        x: Input data, a Cocos array with one or two axes. If y has two axes,
           the Txk array is interpreted as a multidimensional time-series with T
           observations and k variables.
        block_length: the block size
        replications: the number of (sub-)samples to generate
        sub_sample_length: length of the sub-samples to generate
        replace: whether to sample the same block more than once in a single
                 replication


    Returns: a Cocos array with shape (replications, sub_sample_length) of
             bootstrapped sub-samples
    """
    return moving_block_bootstrap_vectorized(
                x=x,
                block_length=block_length,
                replications=replications,
                sub_sample_length=sub_sample_length,
                replace=replace,
                num_pack=cn,
                choice=cn.random.choice)


def tapered_block_bootstrap_cocos(
        x: cn.ndarray,
        block_length: int,
        replications: int,
        sub_sample_length: tp.Optional[int] = None,
        replace: bool = True) \
        -> cn.ndarray:
    """
    This function creates samples from a data series using the tapered block
    bootstrap. It samples overlapping blocks of fixed length. Observations are
    first demeaned and then rescaled according to a window function. After the
    simulation, the mean is added back to each observation.

    Args:
        x: Input data, a Cocos array with one or two axes. If y has two axes,
           the Txk array is interpreted as a multidimensional time-series with T
           observations and k variables.
        block_length: the block length
        replications: the number of bootstrap samples to generate
        sub_sample_length: length of the sub-samples to generate
        replace: whether to sample the same block more than once in a single
                 replication

    Returns: a Cocos array with shape (replications, int(len(x)/block_length))
             of bootstrapped sub-samples

    """

    return tapered_block_bootstrap_vectorized(
                    x=x,
                    block_length=block_length,
                    replications=replications,
                    sub_sample_length=sub_sample_length,
                    replace=replace)
