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
        return ([1, 1], [1+A*p, 2])

    np.random.random()
    blocks = [get_mathieu_block(np.random.uniform(-1, 1))
              for j in range(n_blocks)]
    idxs = list(range(n_blocks))
    return cls.from_blocks(
        blocks=blocks,
        idxs=idxs,
        **params,
    )
