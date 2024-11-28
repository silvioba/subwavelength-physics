import numpy as np
from Subwavelength1D.swp import (
    FiniteSWP1D,
    PeriodicSWP1D,
)


import Subwavelength1D.utils_propagation as utils_propagation

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap, LogNorm

from typing import Literal, Callable, Tuple, Self, List, override

from Subwavelength1D.settings import settings

from Subwavelength1D.utils_general import *

plt.rcParams.update(settings.matplotlib_params)


def check_parameters_inconsitencies(fwp: FiniteSWP1D):
    if not (np.abs(fwp.k_in - fwp.omega / fwp.v_in) < 1e-6).all():
        raise ValueError("k_in does not equal omega / v_in")
    if not (np.abs(fwp.k_out - fwp.omega / fwp.v_out) < 1e-6).all():
        raise ValueError("k_in does not equal omega / v_in")


class ClassicFiniteSWP1D(FiniteSWP1D):
    """
    Base class for acoustic subwavelength wave problem. Subclass of OneDimensionalFiniteSWLProblem

    Initially modelled on https://doi.org/10.1137/22M1503841 (subsequently referred as [1]), subsequently extended
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
        check_parameters_inconsitencies(self)

    @classmethod
    def get_SSH(cls, i: int, s1: float | int, s2: float | int, **params):
        N = 4 * i + 1
        return cls(N=N, l=1, s=np.array(i * [s1, s2] + i * [s2, s1]), **params)

    @override
    def get_capacitance_matrix(self) -> np.ndarray:
        """
        Computes the capacitance matrix C from eq (1.13) in [1]. Only depends on the spacings between resonators.

        Returns:
            np.ndarray:
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
        Computes the capacitance matrix C from eq (1.13) in [1] premultiplied by V^2 L^{-1} where V is the diagonal matrix of wavespeeds inside the resonators and L the matrix of lengths of the resonators

        Returns:
            np.ndarray:
        """
        C = self.get_capacitance_matrix()
        L = np.diag(1 / self.l)
        V = np.diag(self.v_in)
        return V**2 @ L @ C

    def get_greens_matrix(self, k):
        return np.linalg.inv(self.get_generalised_capacitance_matrix()-k*np.eye(self.N))

    def compute_propagation_matrix(
        self, space_from_end=1, subwavelength=True
    ) -> np.ndarray:
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

    Initially modelled on https://doi.org/10.1137/23M1549419 (subsequently referred as [2]), subsequently extended
    """

    def __init__(
        self,
        N: int,
        l: np.ndarray | float,
        s: np.ndarray | float,
        v_in: np.ndarray | float | complex = 1,
        v_out: float = 1,
        delta: float = 0.001,
        omega: float | complex | None = None,
        uin=lambda x: np.sin(x),
        duin=lambda x: np.cos(x),
    ):
        super().__init__(
            N=N,
            l=l,
            s=s,
            v_in=v_in,
            v_out=v_out,
            delta=delta,
            omega=omega,
            uin=uin,
            duin=duin,
        )

    def get_capacitance_matrix(self) -> Callable[[float], np.ndarray]:
        """
        Computes the capacitance matrix C from Lemma 4.7 in [2]. Only depends on the spacings between resonators.

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

        def C(alpha):
            C0[0, -1] += -np.exp(-1j * alpha) / self.s[-1]
            C0[-1, 0] += -np.exp(1j * alpha) / self.s[-1]
            return C0

        return C

    def get_generalised_capacitance_matrix(self) -> Callable[[float], np.ndarray]:

        L = np.diag(1 / self.l)
        V = np.diag(self.v_in)

        return lambda alpha: V**2 @ L @ self.get_capacitance_matrix()(alpha)

    def get_band_data(
        self, generalised=False, nalpha=100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the band data of the capacitance matrix

        Args:
            generalised (bool, optional): doesnt work yet. Defaults to False.
            nalpha (int, optional): number of samples in the first BZ. Defaults to 100.

        Returns:
            np.ndarray: np.linspace(-np.pi, np.pi, nalpha)
            np.ndarray: (nalpha, self.N) array with band data
        """
        alphas = np.linspace(-np.pi, np.pi, nalpha)
        if generalised:
            C = self.get_generalised_capacitance_matrix()
            bands = np.zeros((nalpha, self.N), dtype=complex)
            for i, alpha in enumerate(alphas):
                # raise NotImplementedError("Eigenvalues are not sorted yet")
                D, S = np.linalg.eig(C(alpha))
                bands[i, :] = np.sort(np.real(D))

        else:
            bands = np.zeros((nalpha, self.N), dtype=float)
            C = self.get_capacitance_matrix()
            for i, alpha in enumerate(alphas):
                D, S = np.linalg.eigh(C(alpha))
                bands[i, :] = D

        return alphas, bands

    def plot_band_functions(self, generalised=False, nalpha=100, figax=None) -> Tuple:
        """
        Plots the band functions of the capacitance matrix

        Args:
            generalised (bool, optional): doesnt work yet. Defaults to False.
            nalpha (int, optional): number of samples in the first BZ. Defaults to 100.
            figax (_type_, optional): Tuple (fig,ax) on which to plot. None means do a new plot. Defaults to None.

        Returns:
            Tuple: fig, ax matplotlib
        """
        alphas, bands = self.get_band_data(generalised, nalpha)
        if figax is None:
            fig, ax = plt.subplots(figsize=settings.figure_size)
        else:
            fig, ax = figax
        ax.plot(alphas, bands, "k-")
        ax.set_xticks([-np.pi, 0, np.pi], [r"$-\pi$", r"$0$", r"$\pi$"])
        # ax.set_xlabel("Site index $i$")
        ax.set_ylabel(r"$\lambda_i$")
        return fig, ax


def convert_periodic_into_finite(
    periodic_problem: ClassicPeriodicSWP1D, i: int
) -> ClassicFiniteSWP1D:
    """
    Returns a Classical Finite wave problem with the same properties as the input

    Args:
        periodic_problem (OneDimensionalClassicPeriodicSWLProblem): _description_
        i (int): number of cells

    Returns:
        OneDimensionalClassicFiniteSWLProblem: _description_
    """
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
    Returns a Classical Periodic wave problem with the same properties as the input

    Args:
        finite_problem (OneDimensionalClassicFiniteSWLProblem): _description_
        s_N (float | int): extra distance between the last resonator of one cell and the first resonator of the next cell

    Returns:
        OneDimensionalClassicPeriodicSWLProblem
    """
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