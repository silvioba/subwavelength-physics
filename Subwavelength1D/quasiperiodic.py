import numpy as np
from scipy.signal import correlate

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
    """
    Create a disordered SWP system based on a Fibonacci tiling pattern.

    Parameters
    ----------
    blocks : List[Tuple[List[int | float]]]
        List of block configurations, each defined as (lengths, properties).
    n_tiles : int
        Number of iterations in the Fibonacci sequence construction.
    cls : class, default=DisorderedClassicFiniteSWP1D
        Class to instantiate for the system.
    **params : dict
        Additional parameters to pass to the class constructor.

    Returns
    -------
    cls
        Instance of the specified disordered SWP system class arranged in a Fibonacci pattern.
    """
    tiling = construct_fibonacci_sequence(n_tiles)
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
    """
    Create a disordered SWP system with properties following a Mathieu-like modulation.

    Parameters
    ----------
    n_blocks : int
        Number of blocks in the system.
    A : float, default=1
        Amplitude of the cosine modulation.
    irrational_factor : float, default=(1+sqrt(5))/2
        Irrational number (golden ratio) used to create quasiperiodic pattern.
    cls : class, default=DisorderedClassicFiniteSWP1D
        Class to instantiate for the system.
    **params : dict
        Additional parameters to pass to the class constructor.

    Returns
    -------
    cls
        Instance of the specified disordered SWP system class with Mathieu-modulated properties.
    """
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
    """
    Create a disordered SWP system with random Mathieu-like modulated properties.

    Parameters
    ----------
    n_blocks : int
        Number of blocks in the system.
    A : float, default=1
        Amplitude of the cosine modulation.
    cls : class, default=DisorderedClassicFiniteSWP1D
        Class to instantiate for the system.
    **params : dict
        Additional parameters to pass to the class constructor.

    Returns
    -------
    cls
        Instance of the specified disordered SWP system class with randomly modulated properties.
    """
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


def construct_fibonacci_sequence(n_iterates: int):
    """
    Generate a Fibonacci sequence through iterative substitution.

    Parameters
    ----------
    n_iterates : int
        Number of iterations for the substitution process.

    Returns
    -------
    List[int]
        A Fibonacci sequence of 0s and 1s.

    Notes
    -----
    Uses the substitution rules: 0 → [0,1] and 1 → [0]
    Starting with [1] and applying the rules n_iterates times.
    """
    replacement_dict = {
        0: [1],
        1: [1, 0],
    }
    tiling = [0]
    for i in range(n_iterates):
        tiling = list(itertools.chain.from_iterable(
            [replacement_dict[x] for x in tiling]))
    return tiling


def construct_hyperuniform_binary_sequence(n_chunks: int, seed=42):
    """
    Generate a hyperuniform binary sequence with controlled disorder.

    Parameters
    ----------
    n_chunks : int
        Number of pairs of binary symbols.

    Returns
    -------
    List[int]
        A hyperuniform binary sequence of 0s and 1s with total length 2*n_chunks.

    Notes
    -----
    Creates a sequence where each chunk contains one 0 and one 1,
    ensuring that density fluctuations are minimized.
    """
    np.random.seed(seed)
    ss = []
    for i in range(n_chunks):
        s = np.random.choice(2)
        ss.extend([s, 1-s])
    # Roll to make the sequence homogenous
    ss = np.roll(ss, np.random.randint(0, 2))
    return ss


def construct_bound_length_sequence(n_symbols: int, weights: List[float], bounds: List[int], n_reps: int, seed=42):
    """
    Generate a sequence with bounds on consecutive repetitions of the same symbol.

    Parameters
    ----------
    n_symbols : int
        Number of distinct symbols in the sequence.
    weights : List[float]
        Probability weights for selecting each symbol.
    bounds : List[int]
        Maximum number of consecutive repetitions allowed for each symbol.
    n_reps : int
        Length of the sequence to generate.

    Returns
    -------
    numpy.ndarray
        A sequence of integers respecting the repetition constraints.

    Notes
    -----
    When a symbol reaches its bound for consecutive repetitions,
    it's temporarily removed from the selection pool.
    """
    np.random.seed(seed)
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


def construct_softmax_uniformed_sequence(n_symbols: int, n_reps: int, beta: float = 1, seed=42):
    """
    Generate a sequence with dynamically adjusted probabilities to maintain uniform distribution.

    Parameters
    ----------
    n_symbols : int
        Number of distinct symbols to use.
    n_reps : int
        Length of the sequence to generate.
    beta : float, default=1
        Temperature parameter controlling the strength of the uniformity enforcement.
        Higher values create more uniform distributions.

    Returns
    -------
    numpy.ndarray
        A sequence with balanced occurrences of each symbol.

    Notes
    -----
    Uses a softmax function to adjust selection probabilities based on
    the current count deficit for each symbol.
    """
    def get_weigths_from_counts(counts):
        exps = np.exp(beta*counts)
        return exps / np.sum(exps)

    np.random.seed(seed)
    ss = np.zeros(n_reps, dtype=int)
    for i in range(n_reps):
        expected_count = i/n_symbols
        counts = np.bincount(ss[:i], minlength=n_symbols)
        weights = get_weigths_from_counts(expected_count - counts)
        s = np.random.choice(n_symbols, p=weights)
        ss[i] = s
    return ss


def autocovariance(x):
    """
    Calculate the autocovariance of a sequence.

    Parameters
    ----------
    x : array_like
        Input sequence.

    Returns
    -------
    numpy.ndarray
        The normalized autocovariance of the input sequence.

    Notes
    -----
    The autocovariance is normalized by the length of the input sequence.
    """
    mean = np.mean(x)
    c = correlate(x-mean, x-mean, mode="same")
    return c/len(x)
