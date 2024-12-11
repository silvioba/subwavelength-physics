import numpy as np
from Subwavelength1D.swp import (
    FiniteSWP1D,
    PeriodicSWP1D,
)


import Subwavelength1D.utils_propagation as utils_propagation

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.axes import Axes

from typing import Literal, Callable, Tuple, Self, List, override

import copy

from Utils.settings import settings

import Utils.utils_general as utils

plt.rcParams.update(settings.matplotlib_params)


def check_parameters_inconsistencies(fwp: FiniteSWP1D):
    if not (np.abs(fwp.k_in - fwp.omega / fwp.v_in) < 1e-6).all():
        raise ValueError("k_in does not equal omega / v_in")
    if not (np.abs(fwp.k_out - fwp.omega / fwp.v_out) < 1e-6).all():
        raise ValueError("k_in does not equal omega / v_in")


class ClassicFiniteSWP1D(FiniteSWP1D):
    """
    Base class for acoustic subwavelength wave problem. Subclass of OneDimensionalFiniteSWLProblem

    Initially modelled on [1] (see README), subsequently extended
    """

    def __init__(self, **pars):
        super().__init__(**pars)

    def __str__(self):
        return super().__str__() + "\nPhysics:      Classic system"

    def set_params(self, **params):

        for key, val in params.items():
            if hasattr(self, key):
                setattr(self, key, val)
            else:
                raise AttributeError(
                    f"{self.__class__.__name__} has no attribute '{key}'"
                )
        check_parameters_inconsistencies(self)

    @classmethod
    def get_SSH(cls, i: int, s1: float | int, s2: float | int, **params) -> Self:
        """
        Creates and returns an instance of the class with an SSH geometry configuration.

        SSH geometry is defined by alternating spacings of the form:
        `[s1, s2, s1, s2, ...]` and `[s2, s1, s2, s1, ...]` repeated `i` times.
        These patterns are concatenated to form the complete spacing array.

        Args:
            i (int): Number of repetitions of the spacing pattern. Must be a positive integer.
            s1 (float | int): The first spacing value in the alternating pattern.
            s2 (float | int): The second spacing value in the alternating pattern.
            **params: Additional keyword arguments to pass to the class constructor.

        Returns:
            Self: An instance of the class with the specified SSH geometry configuration.
        """
        if i < 1:
            raise ValueError("i must be a positive integer")
        if s1 <= 0 or s2 <= 0:
            raise ValueError("s1 and s2 must be positive")

        N = 4 * i + 1
        return cls(N=N, l=1, s=np.array(i * [s1, s2] + i * [s2, s1]), **params)

    @override
    def get_capacitance_matrix(self) -> np.ndarray:
        """
        Computes the capacitance matrix C from eq (1.13) in [1]. Only depends on the spacings between resonators.

        Returns:
            np.ndarray
        """

        d1 = np.concatenate(
            (
                [1 / self.s[0]],
                1 / self.s[:-1] + 1 / self.s[1:],
                [1 / self.s[-1]],
            )
        )
        d2 = -1 / self.s
        C = np.diag(d1) + np.diag(d2, 1) + np.diag(d2, -1)
        return C

    @override
    def get_generalised_capacitance_matrix(self) -> np.ndarray:
        """
        Computes the capacitance matrix C from eq (1.13) in [1] premultiplied by V^2 L^{-1} where V is the diagonal matrix of wave speeds inside the resonators and L the matrix of lengths of the resonators

        Returns:
            np.ndarray:
        """
        C = self.get_capacitance_matrix()
        L = np.diag(1 / self.l)
        V = np.diag(self.v_in)
        return V**2 @ L @ C

    def get_greens_matrix(self, k):
        return np.linalg.inv(
            self.get_generalised_capacitance_matrix() - k * np.eye(self.N)
        )

    def compute_propagation_matrix(
        self, space_from_end: float = 1.0, subwavelength: bool = True
    ) -> np.ndarray:
        """
        Computes the propagation matrix for the finite subwavelength wave problem.

        Args:
            space_from_end (float, optional): The spacing from the last resonator to the end of the domain. Defaults to 1.0.
            subwavelength (bool, optional): Whether to use the subwavelength approximation. Defaults to True.

        Raises:
            ValueError: If omega is not set.
            NotImplementedError: If the wave number inside and outside the structure are not the same.

        Returns:
            np.ndarray: The propagation matrix.
        """
        if self.omega is None:
            raise ValueError("omega must be set, is currently None")
        if np.linalg.norm(self.k_in - np.ones(self.N) * self.k_out) > 1e-8:
            raise NotImplementedError(
                "Propagation matrix is implemented only for structure with same wave number inside and outside."
            )

        pm = np.eye(2)
        for i in range(self.N - 1):
            p = utils_propagation.propagation_matrix_single(
                l=self.l[i],
                s=self.s[i],
                k=self.k_in[i],
                delta=self.delta,
                subwavelength=subwavelength,
            )
            pm = p @ pm
        p = utils_propagation.propagation_matrix_single(
            l=self.l[-1],
            s=space_from_end,
            k=self.k_in[i],
            delta=self.delta,
            subwavelength=subwavelength,
        )
        pm = p @ pm
        return pm


class ClassicPeriodicSWP1D(PeriodicSWP1D):
    """
    Base class for acoustic subwavelength wave problem. Subclass of OneDimensionalPeriodicSWLProblem

    Initially modelled on [2] (see README), subsequently extended
    """

    def __init__(self, **pars):
        super().__init__(**pars)

    def get_capacitance_matrix(self) -> Callable[[float], np.ndarray]:
        """
        Computes the capacitance matrix C from Lemma 4.7 in [2]. Only depends on the spacings between resonators.

        The alpha parameter is the Bloch wave number and must be in [-np.pi, np.pi)

        Returns:
            Callable[[float], np.ndarray]: map alpha -> C^alpha
        """

        d1 = np.concatenate(
            (
                [1 / self.s[0] + 1 / self.s[-1]],
                1 / self.s[:-2] + 1 / self.s[1:-1],
                [1 / self.s[-1] + 1 / self.s[0]],
            )
        )
        d2 = -1 / self.s[:-1]
        C0 = np.zeros((self.N, self.N), dtype=complex)
        C0 += np.diag(d1) + np.diag(d2, 1) + np.diag(d2, -1)

        def C(alpha) -> np.ndarray:
            if not -np.pi <= alpha <= np.pi:
                raise ValueError(f"alpha must be in [-pi, pi), you provided {alpha}")
            C0[0, -1] += -np.exp(-1j * alpha) / self.s[-1]
            C0[-1, 0] += -np.exp(1j * alpha) / self.s[-1]
            return C0

        return C

    def get_generalised_capacitance_matrix(self) -> Callable[[float], np.ndarray]:
        """
        Computes the generalised capacitance matrix as a function of the Bloch wave number alpha.

        Returns:
            Callable[[float], np.ndarray]: A function that maps alpha to the generalised capacitance matrix.
        """

        L = np.diag(1 / self.l)
        V = np.diag(self.v_in)

        return lambda alpha: V**2 @ L @ self.get_capacitance_matrix()(alpha)

    def get_band_data(
        self, generalised: bool = True, nalpha: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the band data of the capacitance matrix

        Args:
            generalised (bool, optional): Wheter to use the generalised capacitance matrix. Defaults to True.
            nalpha (int, optional): number of samples in [-pi, pi). Defaults to 100.

        Returns:
            np.ndarray: np.linspace(-np.pi, np.pi, nalpha)
            np.ndarray: (nalpha, self.N) array with band data
        """
        alphas = np.linspace(-np.pi, np.pi, nalpha)
        if generalised:
            C = self.get_generalised_capacitance_matrix()
            bands = np.zeros((nalpha, self.N), dtype=complex)
            for i, alpha in enumerate(alphas):
                D, S = np.linalg.eig(C(alpha))
                D, S = utils.sort_by_eva_real(D, S)
                bands[i, :] = D

        else:
            bands = np.zeros((nalpha, self.N), dtype=float)
            C = self.get_capacitance_matrix()
            for i, alpha in enumerate(alphas):
                D, S = np.linalg.eigh(C(alpha))
                bands[i, :] = D

        return alphas, bands

    def plot_band_functions(
        self, generalised=True, nalpha=100, ax: Axes | None = None
    ) -> Tuple:
        """
        Plots the band functions of the capacitance matrix

        Args:
            generalised (bool, optional): Wheter to use the generalised capacitance matrix. Defaults to False.
            nalpha (int, optional): number of samples in [-pi, pi). Defaults to 100.
            ax (Axes | None, optional): matplotlib ax on which to plot. None means do a new plot. Defaults to None.

        Returns:
            Tuple: fig, ax matplotlib
        """
        alphas, bands = self.get_band_data(generalised, nalpha)
        if ax is None:
            fig, ax = plt.subplots(figsize=settings.figure_size)
        ax.plot(alphas, bands, "k-")
        ax.set_xticks([-np.pi, 0, np.pi], [r"$-\pi$", r"$0$", r"$\pi$"])
        ax.set_ylabel(r"$\lambda_i$")
        return ax


def convert_periodic_into_finite(
    periodic_problem: ClassicPeriodicSWP1D, i: int
) -> ClassicFiniteSWP1D:
    """
    Returns a Classical Finite wave problem with the same properties as a periodic system

    Args:
        periodic_problem (ClassicPeriodicSWP1D): the periodic system
        i (int): number of cells

    Returns:
        ClassicFiniteSWP1D: classical finite subwavelength wave problem with the same properties
    """
    periodic_problem = copy.deepcopy(periodic_problem)
    return ClassicFiniteSWP1D(
        N=periodic_problem.N * i,
        l=np.concatenate([periodic_problem.l for _ in range(i)]),
        s=np.concatenate([periodic_problem.s for _ in range(i)])[:-1],
        v_in=np.concatenate([periodic_problem.v_in for _ in range(i)]),
        v_out=periodic_problem.v_out,
        delta=periodic_problem.delta,
        omega=periodic_problem.omega,
        uin=periodic_problem.uin,
        duin=periodic_problem.duin,
    )


def convert_finite_into_periodic(
    finite_problem: ClassicFiniteSWP1D, s_N: float | int
) -> ClassicPeriodicSWP1D:
    """
    Returns a Classical Periodic wave problem with the same properties as a finite system

    Args:
        finite_problem (ClassicFiniteSWP1D): ClassicFiniteSWP1D
        s_N (float | int): extra distance between the last resonator of one cell and the first resonator of the next cell

    Returns:
        ClassicPeriodicSWP1D
    """
    if s_N <= 0:
        raise ValueError("s_N must be positive")
    return ClassicPeriodicSWP1D(
        N=finite_problem.N,
        l=finite_problem.l,
        s=np.concatenate([finite_problem.s, [s_N]]),
        v_in=finite_problem.v_in,
        v_out=finite_problem.v_out,
        delta=finite_problem.delta,
        omega=finite_problem.omega,
        uin=finite_problem.uin,
        duin=finite_problem.duin,
    )
