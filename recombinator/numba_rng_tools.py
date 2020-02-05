"""
This module is adapted from Alex Gramfort's work <alexandre.gramfort@inria.fr>
published on GitHub at https://github.com/numba/numba/issues/3249.
"""

import numpy as np
from numba import _helperlib


def check_random_state(seed):
    """
    Turn seed into a np.random.RandomState instance.

    If seed is None, return the RandomState singleton used by np.random.
    If seed is an int, return a new RandomState instance seeded with seed.
    If seed is already a RandomState instance, return it.
    Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (int, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError(f'{seed} cannot be used to seed a '
                     f'numpy.random.RandomState instance')


def _copy_np_state(r, ptr):
    """
    Copy state of Numpy random *r* to Numba state *ptr*.
    """
    ints, index = r.get_state()[1:3]
    _helperlib.rnd_set_state(ptr, (index, [int(x) for x in ints]))
    return ints, index


def _copyback_np_state(r, ptr):
    """
    Copy state of of Numba state *ptr* to Numpy random *r*
    """
    index, ints = _helperlib.rnd_get_state(ptr)
    r.set_state(('MT19937', ints, index, 0, 0.0))


def get_np_state_ptr():
    """
    Get the Numba state *ptr*
    """
    return _helperlib.rnd_get_np_state_ptr()


class rng_link:
    """
    This decorator links the state of the random number generators in Numpy and
    Numba. To achieve this, it copies Numpy's random state to Numba before the
    function call and copies the state in the opposite direction afterwards.
    """
    def __init__(self, random_state=None):
        self.random_state = random_state

    def __call__(self, func):
        def new_func(*args, **kwargs):
            link_numba_rng = kwargs.get('link_rngs')
            if link_numba_rng:
                r = check_random_state(self.random_state)
                ptr = get_np_state_ptr()
                _copy_np_state(r, ptr)

            out = func(*args, **kwargs)

            if link_numba_rng:
                _copyback_np_state(r, ptr)

            return out
        return new_func
