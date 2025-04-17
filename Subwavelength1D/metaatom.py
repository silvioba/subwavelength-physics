import time
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
        else:
            raise ValueError(
                f"Metaatom {sequence[index:index+len(metaatoms[0])]} not found in {metaatoms}")

    return occurrences


def get_metaatom_occurrence_dict(sequence, metaatoms):
    """
    Calculates metaatom occurrences using a dictionary for faster lookups.

    Assumes metaatoms is sorted by ascending sequence length.
    """

    # --- Preprocessing ---
    # Ensure metaatoms are tuples for hashing
    metaatoms_tuple = [tuple(ma) for ma in metaatoms]
    metaatom_map = {ma: idx for idx, ma in enumerate(metaatoms_tuple)}

    if not metaatoms_tuple:
        if sequence:
            raise ValueError("No metaatoms provided to parse the sequence.")
        else:
            return []  # Empty sequence, empty metaatoms -> 0 occurrences

    max_metaatom_len = len(metaatoms_tuple[-1]) if metaatoms_tuple else 0

    # --- Processing ---
    # Add a termination symbol (using None as it's unlikely to be in the data)
    # Using a tuple for potential slicing performance benefits
    # Use None or another unique sentinel
    sequence_tuple = tuple(sequence + [0])
    seq_len = len(sequence_tuple)
    occurrences = [0] * len(metaatoms)
    index = 0

    while index < seq_len:  # Stop before the sentinel
        found_match = False
        # Iterate lengths from longest possible down to 1
        # Start with min(max_metaatom_len, remaining sequence length)
        for length in range(min(max_metaatom_len, seq_len - index), 0, -1):
            sub_sequence = sequence_tuple[index: index + length]

            if sub_sequence in metaatom_map:
                ma_idx = metaatom_map[sub_sequence]
                occurrences[ma_idx] += 1
                index += length
                found_match = True
                break  # Found the longest match for this position

        if not found_match:
            # If the loop completes without finding any match (even length 1)
            # Check if the single element itself is a metaatom if not caught above
            # (The loop range already covers length 1)
            raise ValueError(
                f"Sequence segment starting at index {index} "
                f"with element '{sequence_tuple[index]}' does not match "
                f"the start of any known metaatom."
            )
            # Or, if single elements *not* part of a longer metaatom should be skipped:
            # index += 1 # Skip the unmatchable element (depends on desired behavior)

    # Check if we consumed the entire original sequence
    if index != len(sequence):
        # This can happen if the loop exited early but didn't reach the end
        # typically because the remaining part couldn't be matched.
        # The ValueError above should catch this, but adding a check for safety.
        pass  # The ValueError inside the loop should handle incomplete parsing

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
                          midgap=1.5,
                          use_hashmap: bool = True,
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

    if use_hashmap:
        # Use the dictionary-based method for faster lookups
        occs = get_metaatom_occurrence_dict(sequence, metaatoms)
    else:
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
