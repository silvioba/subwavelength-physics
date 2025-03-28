import numpy as np
from scipy.signal import correlate

from Subwavelength1D.classic import ClassicFiniteSWP1D

from Subwavelength1D.nonreciprocal import NonReciprocalFiniteSWP1D, NonReciprocalPeriodicSWP1D
from Subwavelength1D.disordered import DisorderedClassicFiniteSWP1D, DisorderedNonReciprocalFiniteSWP1D

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm

from functools import cache

from typing import Literal, Callable, Tuple, Self, List, override
import itertools

from Utils.settings import settings as settings

from Utils.utils_general import *

plt.rcParams.update(settings.matplotlib_params)


def get_metaatom_occurrence(sequence, metaatoms):
    # Add a termination symbol
    sequence = tuple(sequence + [0])
    occurrences = [0] * len(metaatoms)
    index = 0
    while index < len(sequence):
        for ma_idx in range(len(metaatoms)-1, -1, -1):
            metaatom = metaatoms[ma_idx]
            if sequence[index:index+len(metaatom)] == metaatom:
                # print(f"Found {metaatom} ({ma_idx}) at index {index}")
                occurrences[metaatoms.index(metaatom)] += 1
                index += len(metaatom)
                break

    return occurrences


@cache
def get_metaatom_solutions(bulk_block, defect_block, metaatom, d=10, midgap=1.5):
    # print(metaatom)
    dp_dirichlet = DisorderedClassicFiniteSWP1D.from_blocks(
        [bulk_block, defect_block],
        [0]*d+list(metaatom)+[0]*d,
        v_in=1, v_out=1
    )
    D = dp_dirichlet.get_spectral_range_capacitance_matrix()
    return list(D[D > midgap])


def get_metaatom_spectrum(sequence: List[int],
                          metaatoms: Tuple[Tuple[int]] = None,
                          bulk_block: Tuple[Tuple[int | float]] = None,
                          defect_block: Tuple[Tuple[int | float]] = None,
                          d=10,
                          midgap=1.5
                          ):
    if not bulk_block:
        bulk_block = ((2,), (2,))

    if not defect_block:
        defect_block = ((1, 1), (1, 2))

    if not metaatoms:
        metaatoms = [
            (0,), (1,),
            (1, 1, 0),
            (1, 0, 1, 0), (1, 1, 1, 0),
            (1, 1, 1, 1, 0),
            (1, 1, 1, 1, 1, 0), (1, 1, 0, 1, 1, 0),
        ]

    occs = get_metaatom_occurrence(sequence, metaatoms)

    D_calc = []
    for i, num_occ in enumerate(occs):
        ds = get_metaatom_solutions(bulk_block, defect_block,
                                    metaatoms[i], d=d, midgap=midgap)*num_occ
        D_calc.extend(ds)

    return D_calc


def generate_metaatoms(max_len, max_bulk):
    metaatoms = [(0,), (1,), (1, 1, 0)]
    for l in range(3, max_len+1):
        l_interior = l-2
        for ma in itertools.product((0, 1), repeat=l_interior):
            if ma.count(0) <= max_bulk:
                metaatoms.append(tuple([1]+list(ma)+[1, 0]))

    return metaatoms
