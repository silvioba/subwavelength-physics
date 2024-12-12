import numpy as np

from Subwavelength1D.classic import ClassicFiniteSWP1D

from Subwavelength1D.nonreciprocal import NonReciprocalFiniteSWP1D, NonReciprocalPeriodicSWP1D

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm

from typing import Literal, Callable, Tuple, Self, List, override

from Utils.settings import settings as settings

from Utils.utils_general import *

plt.rcParams.update(settings.matplotlib_params)


class DisorderedCommon:
    pass


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
            Self: DisorderedClassicFiniteSWP1D instance with the specified blocks
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
            Self: DisorderedClassicFiniteSWP1D instance with randomly picked blocks
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

    def get_block_list(self):
        """Get the list of len(self.idxs) containing the corresponding blocks, as specified by self.idx.

        Returns:
            List: 
        """
        return [self.blocks[idx] for idx in self.idxs]

    def get_block_index_at_x(self, x):
        """Get the index idx such that x lies in the interval of the idx-th block.

        Args:
            x (float): Space coordinate

        Returns:
            int: Corresponding index
        """
        assert x >= self.xi[0] and x <= self.xi[-1]
        endpoint = 0
        for i, block in enumerate(self.get_block_list()):
            endpoint += sum(block[0]) + sum(block[1])
            if x < endpoint:
                return i

    def resonators_before_index(self, idx):
        """Get the total number of resonators BEFORE the block of specified index. Note that blocks may recall multiple resonators.

        Args:
            idx (int): Index specifying the block of interest

        Returns:
            int: Number of resonators before the block
        """
        assert 0 <= idx < len(self.idxs)
        return sum([len(block[0]) for block in self.get_block_list()[:idx]])

    def get_block_index_at_resonator(self, j):
        """For a given resonator index j, get the index of the block containing the resonator.

        Args:
            j (int): Index specifying the resonator of interest

        Returns:
            int: Index of the block containing said resonator
        """
        assert 0 <= j < self.N
        res_idx = 0
        for i, block in enumerate(self.get_block_list()):
            res_idx += len(block[0])
            if j < res_idx:
                return i

    def get_block_bounds(self, idx):
        """For a given block index, get the bounds of the block in terms of space coordinates.

        Args:
            idx (int): Index specifying the block of interest

        Returns:
            (float, float): left and right space bounds of said block
        """
        assert 0 <= idx < len(self.idxs)
        left = 0
        right = 0
        for i, block in enumerate(self.get_block_list()):
            right = left + sum(block[0]) + sum(block[1])
            if i == idx:
                return left, right
            left = right


class DisorderedNonReciprocalFiniteSWP1D(NonReciprocalFiniteSWP1D):
    pass
