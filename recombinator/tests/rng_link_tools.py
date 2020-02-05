import numba
import numpy as np
import typing as tp

from recombinator.numba_rng_tools import rng_link


def _construct_kwargs(x: np.ndarray,
                     number_of_boostrap_replications: int,
                     block_length: tp.Optional[int] = None) \
        -> tp.Dict[str, tp.Any]:
    kwargs = {'x': x,
              'replications': number_of_boostrap_replications,
              'link_rngs': True}
    if block_length is not None:
        kwargs['block_length'] = block_length

    return kwargs


@rng_link()
@numba.njit
def numba_rand(link_rngs: bool):
    return np.random.rand()


def numpy_to_numba_rng_link_generic_tester(
        x: np.ndarray,
        bootstrap_function: tp.Callable,
        number_of_boostrap_replications: int,
        block_length: tp.Optional[int] = None):
    kwargs = \
        _construct_kwargs(x=x,
                          number_of_boostrap_replications
                          =number_of_boostrap_replications,
                          block_length=block_length)

    np.random.seed(42)
    resampled_numba_1 = \
        bootstrap_function(**kwargs)

    np.random.seed(42)
    resampled_numba_2 = \
        bootstrap_function(**kwargs)

    np.random.seed(42)
    np.random.rand()
    resampled_numba_3 = \
        bootstrap_function(**kwargs)

    assert np.allclose(resampled_numba_1, resampled_numba_2)
    assert not np.allclose(resampled_numba_1, resampled_numba_3)


def numba_to_numpy_rng_link_generic_tester(
        x: np.ndarray,
        bootstrap_function: tp.Callable,
        number_of_boostrap_replications: int,
        block_length: tp.Optional[int] = None):
    kwargs = \
        _construct_kwargs(x=x,
                          number_of_boostrap_replications
                          =number_of_boostrap_replications,
                          block_length=block_length)

    np.random.seed(42)
    numpy_rand_1 = np.random.rand()

    np.random.seed(42)
    bootstrap_function(**kwargs)

    numpy_rand_2 = np.random.rand()

    assert not np.allclose(numpy_rand_1, numpy_rand_2)
