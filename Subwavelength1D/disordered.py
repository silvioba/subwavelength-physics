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

    def get_block_list(self):
        return [self.blocks[idx] for idx in self.idxs]

    def get_block_index_at_x(self, x):
        assert x >= self.xi[0] and x <= self.xi[-1]
        endpoint = 0
        for i, block in enumerate(self.get_block_list()):
            endpoint += sum(block[0]) + sum(block[1])
            if x < endpoint:
                return i

    def resonators_before_index(self, idx):
        assert 0 <= idx < self.N
        return sum([len(block[0]) for block in self.get_block_list()[:idx]])

    def get_block_index_at_resonator(self, j):
        assert 0 <= j < self.N
        res_idx = 0
        for i, block in enumerate(self.get_block_list()):
            res_idx += len(block[0])
            if j < res_idx:
                return i

    def get_block_bounds(self, idx):
        assert 0 <= idx < len(self.idxs)
        left = 0
        right = 0
        for i, block in enumerate(self.get_block_list()):
            right = left + sum(block[0]) + sum(block[1])
            if i == idx:
                return left, right
            left = right

    def plot_variance_band_functions(
        self,
        s_N=1,
        nalpha=100,
        ax=None,
        semilogy=False,
        generalised=False,
        only_background=False,
        **kwargs,
    ):
        pwp = convert_finite_into_periodic(self, s_N=s_N)
        alphas, bands = pwp.get_band_data(generalised=generalised, nalpha=nalpha)
        bands = np.real(bands)

        variances = np.var(bands, axis=0) * self.N**2
        variance_lowest = variances[0]
        means = np.mean(bands, axis=0)

        mask_big_jump = np.diff(means) > 0.2
        idxs = np.arange(len(mask_big_jump))[mask_big_jump]
        idxs += 1

        def custom_colormap_with_lognorm(a, vmin, vmax):
            """
            Generates a custom colormap with graded red above `a` and graded blue below `a`,
            with logarithmic normalization.

            Parameters:
            a (float): The threshold value for the color transition (in log space).
            vmin (float): The minimum value for normalization (must be > 0).
            vmax (float): The maximum value for normalization (must be > 0).

            Returns:
            tuple: (LinearSegmentedColormap, LogNorm) for custom plotting.
            """
            # Define the color transitions for the blue-to-white and white-to-red gradient
            colors = [
                (0.0, (0.9, 0.9, 1.0)),  # Very light blue (near-white)
                (0.5, (0.0, 0.0, 1.0)),  # Blue
                # (0.5, (1.0, 1.0, 1.0)),  # White at the transition point
                # (0.5, (0.5, 0.0, 0.5)),  # Purple at the transition point
                (0.5, (0.0, 0.0, 1.0)),  # Blue
                (1.0, (1.0, 0.0, 0.0)),  # Red
            ]

            # Create the colormap
            cmap = LinearSegmentedColormap.from_list("custom_colormap", colors)

            # Log normalization with midpoint adjustment
            class MidPointLogNorm(LogNorm):
                def __init__(self, vmin, vmax, midpoint, clip=False):
                    self.midpoint = midpoint
                    super().__init__(vmin, vmax, clip)

                def __call__(self, value, clip=None):
                    # Adjust log normalization for smooth scaling around the midpoint
                    log_v = np.log10(value)
                    log_vmin, log_vmax, log_midpoint = map(
                        np.log10, [self.vmin, self.vmax, self.midpoint]
                    )
                    norm_value = (log_v - log_vmin) / (log_vmax - log_vmin)
                    norm_value = np.where(
                        log_v <= log_midpoint,
                        (log_v - log_vmin) / (log_midpoint - log_vmin) * 0.5,
                        0.5 + (log_v - log_midpoint) / (log_vmax - log_midpoint) * 0.5,
                    )
                    return np.ma.masked_array(norm_value)

            return cmap, MidPointLogNorm(vmin=vmin, vmax=vmax, midpoint=a)

        variances_trans = variances

        if ax is None:
            fig, ax = plt.subplots()
        if only_background:
            idxs = np.insert(idxs, 0, 0)
            idxs = np.append(idxs, -1)
            for i in range(len(idxs) - 1):
                means_sel = means[idxs[i] : idxs[i + 1]]
                X, Y = np.meshgrid(
                    means_sel,
                    np.array(
                        [kwargs.get("vlims", [0, 1])[0], kwargs.get("vlims", [0, 1])[1]]
                    ),
                )
                Z = np.ones_like(Y, dtype=float)
                for j in range(Z.shape[0]):
                    Z[j, :] = variances_trans[idxs[i] : idxs[i + 1]]

                cmap, norm = custom_colormap_with_lognorm(
                    variance_lowest,
                    vmin=np.min(variances_trans) + 1e-14,
                    vmax=np.max(variances_trans),
                )
                pcm = ax.pcolormesh(
                    X,
                    Y,
                    Z,
                    norm=(norm if semilogy else None),
                    cmap=cmap,
                    shading="nearest",
                    # edgecolors="black",
                )
            if kwargs.get("colorbar"):
                plt.colorbar(pcm, ax=ax, extend="max")

        else:
            if semilogy:
                ax.semilogy(means, variances, "k.")
            else:
                ax.plot(means, variances, "k.")
        return ax
