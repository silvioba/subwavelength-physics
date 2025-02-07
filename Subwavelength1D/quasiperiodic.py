import numpy as np

from Subwavelength1D.classic import ClassicFiniteSWP1D

from Subwavelength1D.nonreciprocal import NonReciprocalFiniteSWP1D, NonReciprocalPeriodicSWP1D
from Subwavelength1D.disordered import DisorderedClassicFiniteSWP1D, DisorderedNonReciprocalFiniteSWP1D

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm

from typing import Literal, Callable, Tuple, Self, List, override
import itertools

from Utils.settings import settings as settings

from Utils.utils_general import *

plt.rcParams.update(settings.matplotlib_params)


def disordered_system_from_fibonacci_tiling(
    blocks: List[Tuple[List[int | float]]],
    n_tiles: int,
    cls=DisorderedClassicFiniteSWP1D,
    **params,
):
    replacement_dict = {
        0: [0, 1],
        1: [0],
    }
    tiling = [1]
    for i in range(n_tiles):
        tiling = list(itertools.chain.from_iterable(
            [replacement_dict[x] for x in tiling]))
    return cls.from_blocks(
        blocks=blocks,
        idxs=tiling,
        **params,
    )


def disordered_system_from_mathieu(
    n_blocks: int,
    A: float = 1,
    irrational_factor: float = (1+np.sqrt(5))/2,
    cls=DisorderedClassicFiniteSWP1D,
    **params,
):
    def get_mathieu_block(j):
        return ([1, 1], [1+A*np.cos(2*np.pi*j*irrational_factor), 2])

    blocks = [get_mathieu_block(j) for j in range(n_blocks)]
    idxs = list(range(n_blocks))
    return cls.from_blocks(
        blocks=blocks,
        idxs=idxs,
        **params,
    )


def disordered_system_from_random_mathieu(
    n_blocks: int,
    A: float = 1,
    cls=DisorderedClassicFiniteSWP1D,
    **params,
):
    def get_mathieu_block(p):
        return ([1, 1], [1+A*np.cos(p), 2])

    np.random.random()
    blocks = [get_mathieu_block(np.random.uniform(0, 2*np.pi))
              for j in range(n_blocks)]
    idxs = list(range(n_blocks))
    return cls.from_blocks(
        blocks=blocks,
        idxs=idxs,
        **params,
    )


def construct_hyperuniform_binary_sequence(n_chunks: int):
    ss = []
    for i in range(n_chunks):
        s = np.random.choice(2)
        ss.extend([s, 1-s])
    return ss


def construct_bound_length_sequence(n_symbols: int, weights: List[float], bounds: List[int], n_reps: int,):
    ss = np.zeros(n_reps, dtype=int)
    for i in range(n_reps):
        # Check if reached the repetition bound
        last_s = ss[i-1] if i > 0 else None
        if last_s is not None and i >= bounds[last_s] and all(t == last_s for t in ss[i-bounds[last_s]:i]):
            # If so, remove the last symbol from the weights
            t_weights = weights.copy()
            t_weights[last_s] = 0
            t_weights = t_weights / np.sum(t_weights)
            s = np.random.choice(n_symbols, p=t_weights)
        else:
            s = np.random.choice(n_symbols, p=weights)

        ss[i] = s
    return ss


def construct_softmax_uniformed_sequence(n_symbols: int, n_reps: int, beta: float = 1):
    def get_weigths_from_counts(counts):
        exps = np.exp(beta*counts)
        return exps / np.sum(exps)

    ss = np.zeros(n_reps, dtype=int)
    for i in range(n_reps):
        expected_count = i/n_symbols
        counts = np.bincount(ss[:i], minlength=n_symbols)
        weights = get_weigths_from_counts(expected_count - counts)
        s = np.random.choice(n_symbols, p=weights)
        ss[i] = s
    return ss


def autocovariance(x, max_lag=None):
    """
    Compute the sample autocovariance of a (0-1) binary sequence x.

    Parameters
    ----------
    x : array_like
        Input 1D sequence of 0s and 1s.
    max_lag : int, optional
        Maximum lag for which autocovariance is calculated.
        If None, defaults to len(x) - 1.

    Returns
    -------
    covs : ndarray
        1D array of autocovariance values for lags 0..max_lag.
    """

    x = np.asarray(x, dtype=float)
    n = len(x)
    if max_lag is None:
        max_lag = n - 1

    # Compute mean
    mean_x = np.mean(x)
    # Center the sequence around zero
    x_centered = x - mean_x

    # Use 'full' cross-correlation, then trim. For a sequence x_centered,
    # np.correlate(x_centered, x_centered, 'full') yields an array of length 2n-1:
    #    [ (x[0]*x[0] + ... ), ..., (x[n-1]*x[0]), (x[0]*x[n-1]), ..., (x[n-1]*x[n-1]) ]
    # The zero-lag is at index n-1.
    c = np.correlate(x_centered, x_centered, mode='full')

    # We only want the part from lag=0 up to lag=max_lag
    # The element for lag k is c[n-1 + k]
    c = c[n-1: n-1 + max_lag + 1]

    # Typically, the sample autocovariance for lag k is
    #   (1 / n) * sum_{t=k+1..n} [(x_t - mean_x)*(x_{t-k} - mean_x)]
    # Here, we divide by n to get the biased estimator.
    # For the unbiased estimator, you might divide by (n - k) or (n - |k|).
    c = c / n

    return c
