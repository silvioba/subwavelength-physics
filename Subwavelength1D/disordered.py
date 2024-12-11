import numpy as np

from Subwavelength1D.classic import (
    ClassicFiniteSWP1D,
    ClassicFiniteSWP1D,
    convert_finite_into_periodic,
)

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm

from typing import Literal, Callable, Tuple, Self, List, override

from Utils.settings import settings as settings

from Utils.utils_general import *

plt.rcParams.update(settings.matplotlib_params)


class DisorderedClassicFiniteSWP1D(ClassicFiniteSWP1D):
    """
    A class representing a disordered classical finite subwavelength problem in 1D.

    This class extends the `ClassicFiniteSWP1D` class to handle disordered systems of resonators.
    Attributes:
        idxs (List[int] | None): Indices indicating the order in which to use the blocks.
        blocks (List[Tuple[List[int | float]]] | None): Contains the blocks of resonators.
    Methods:
        from_finite_wave_problem(fwp):
            Constructs an instance from a finite wave problem.
        from_blocks(blocks, idxs, **params):
            Constructs a finite classical system of disordered blocks of resonators.
        from_blocks_random(blocks, n_reps, weights=None, seed=42, **params):
            Constructs a finite classical system of disordered blocks of resonators by randomly picking blocks.
    """

    def __init__(self, **params):
        super().__init__(**params)
        self.idxs = None
        self.blocks = None

    def __str__(self):
        return super().__str__()

    @classmethod
    def from_finite_wave_problem(cls, fwp):
        """
        Create an instance of the class from a finite wave problem.

        Args:
            fwp (FiniteWaveProblem): An instance of FiniteWaveProblem containing the necessary parameters.

        Returns:
            cls: An instance of the class with parameters initialized from the finite wave problem.
        """
        return cls(N=fwp.N, l=fwp.l, s=fwp.s, v_in=fwp.v_in)

    @classmethod
    def from_blocks(
        cls, blocks: List[Tuple[List[int | float]]], idxs: List[int], **params
    ) -> Self:
        """
        Constructs a finite classical system of disorded blocks of resonators

        Args:
            blocks (List[List[int  |  float]]): contains the blocks, the inner is given by ([l1,l2,l3], [s1,s2,s3])
            idxs (List[int]): orders in which to use the blocks

        Returns:
            Self: OneDimensionalClassicDisorderedFiniteSWLProblem
        """
        l = []
        s = []
        for idx in idxs:
            ll, ss = blocks[idx]
            l = l + ll
            s = s + ss
        s = s[:-1]
        c = cls(N=len(l), l=np.array(l), s=np.array(s), **params)
        c.__setattr__("idxs", idxs)
        c.__setattr__("blocks", blocks)
        return c

    @classmethod
    def from_blocks_random(
        cls,
        blocks: List[Tuple[List[int | float]]],
        n_reps: int,
        weights: List[float] | None = None,
        seed=42,
        **params,
    ) -> Self:
        """
        Constructs a finite classical system of disorded blocks of resonators by random picking blocks


        Args:
            blocks (List[Tuple[List[int  |  float]]]): contains the blocks, the inner is given by ([l1,l2,l3], [s1,s2,s3])
            n_reps (int): total number of blocks in the final structure
            weights (List[float] | None, optional): probabilities to choose the blocks. If none the probabilities are uniform. Defaults to None.
            seed (int, optional): Random seed for reproducibility. Defaults to 42.

        Returns:
            Self: _description_
        """
        if weights:
            assert len(weights) == len(
                blocks
            ), "len(weights) must be same of len(blocks)"
            assert sum(weights) == 1, "sum(weights) must be 1"

        np.random.seed(seed)
        return cls.from_blocks(
            blocks=blocks,
            idxs=np.random.choice(len(blocks), n_reps, p=weights),
            **params,
        )
