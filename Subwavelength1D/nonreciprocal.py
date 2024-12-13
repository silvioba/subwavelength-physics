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

import copy

from Utils.settings import settings

from Utils.utils_general import *

plt.rcParams.update(settings.matplotlib_params)


def check_parameters_inconsitencies(fwp: FiniteSWP1D):
    if not (np.abs(fwp.k_in - fwp.omega / fwp.v_in) < 1e-6).all():
        raise ValueError("k_in does not equal omega / v_in")
    if not (np.abs(fwp.k_out - fwp.omega / fwp.v_out) < 1e-6).all():
        raise ValueError("k_in does not equal omega / v_in")


class NonReciprocalFiniteSWP1D(FiniteSWP1D):
    """
    Base class for nonreciprocal finite acoustic subwavelength wave problem. Subclass of OneDimensionalFiniteSWLProblem

    Initially modelled on https://arxiv.org/abs/2306.15587 (subsequently referred as [1]), subsequently extended
    """

    def __init__(self, gammas=1, **pars):
        super().__init__(**pars)
        if isinstance(gammas, (int, float)):
            gammas = np.ones(self.N) * gammas
        self.gammas = np.array(gammas, dtype=float)

    def __str__(self):
        return super().__str__() + "\nPhysics:      Non-reciprocal system"

    def set_params(self, **params):
        for key, val in params.items():
            if hasattr(self, key):
                setattr(self, key, val)
            else:
                raise AttributeError(
                    f"{self.__class__.__name__} has no attribute '{key}'"
                )
        check_parameters_inconsitencies(self)

    @override
    def get_capacitance_matrix(self) -> np.ndarray:
        """
        Computes the gauge capacitance matrix C from eq (20) in [1]. Note that the paper contains wrong indicies. This implementation is corrected.

        Returns:
            np.ndarray:
        """
        assert self.N > 1, "N must be greater than 1 to compute capacitance matrix"
        upper_diag = - self.gammas[:-1] * self.l[:-1] / (
            self.s * (1-np.exp(-self.gammas[:-1]*self.l[:-1])))
        lower_diag = self.gammas[1:] * self.l[1:] / (
            self.s * (1-np.exp(self.gammas[1:]*self.l[1:])))

        first_coef = self.gammas[0] * self.l[0] / (
            self.s[0] * (1-np.exp(-self.gammas[0]*self.l[0])))
        last_coef = - self.gammas[-1] * self.l[-1] / (
            self.s[-1] * (1-np.exp(self.gammas[-1]*self.l[-1])))

        center_diag = self.gammas[1:-1] * self.l[1:-1] * (
            1/(self.s[1:]*(1-np.exp(-self.gammas[1:-1] * self.l[1:-1]))) - 1/(self.s[:-1]*(1-np.exp(self.gammas[1:-1]*self.l[1:-1]))))

        center_diag = np.concatenate([[first_coef], center_diag, [last_coef]])

        C = np.diag(center_diag) + np.diag(upper_diag, 1) + \
            np.diag(lower_diag, -1)
        return C

    @override
    def get_generalised_capacitance_matrix(self) -> np.ndarray:
        """
        Computes the gauge capacitance matrix C premultiplied by V^2 L^{-1} where V is the diagonal matrix of wavespeeds inside the resonators and L the matrix of lengths of the resonators

        Returns:
            np.ndarray:
        """
        C = self.get_capacitance_matrix()
        L = np.diag(1 / self.l)
        V = np.diag(self.v_in)
        return V**2 @ L @ C

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


class NonReciprocalPeriodicSWP1D(PeriodicSWP1D):
    """
    Base class for nonreciprocal periodic acoustic subwavelength wave problem. Subclass of OneDimensionalFiniteSWLProblem

    Initially modelled on https://arxiv.org/abs/2306.15587 (subsequently referred as [1]), subsequently extended
    """

    def __init__(self, gammas=1, **pars):
        super().__init__(**pars)
        if isinstance(gammas, (int, float)):
            gammas = np.ones(self.N) * gammas
        self.gammas = np.array(gammas, dtype=float)

    def get_capacitance_matrix(self) -> Callable[[float], np.ndarray]:
        """
        Computes the capacitance matrix C from Definition in [1]. Note that the paper contains wrong indicies. This implementation is corrected.

        Returns:
            Callable[[float], np.ndarray]: map alpha -> C^alpha
        """

        if self.N == 1:
            C0 = np.array([[
                self.gammas[0] * self.l[0] * (
                    1/(self.s[0]*(1-np.exp(-self.gammas[0] * self.l[0]))) - 1/(self.s[-1]*(1-np.exp(self.gammas[0]*self.l[0]))))
            ]])
        else:
            upper_diag = - self.gammas[:-1] * self.l[:-1] / (
                self.s[:-1] * (1-np.exp(-self.gammas[:-1]*self.l[:-1])))
            lower_diag = self.gammas[1:] * self.l[1:] / (
                self.s[:-1] * (1-np.exp(self.gammas[1:]*self.l[1:])))

            first_coef = self.gammas[0] * self.l[0] * (
                1/(self.s[0]*(1-np.exp(-self.gammas[0] * self.l[0]))) - 1/(self.s[-1]*(1-np.exp(self.gammas[0]*self.l[0]))))
            last_coef = self.gammas[-1] * self.l[-1] * (
                1/(self.s[-1]*(1-np.exp(-self.gammas[-1] * self.l[-1]))) - 1/(self.s[-2]*(1-np.exp(self.gammas[-1]*self.l[-1]))))

            center_diag = self.gammas[1:-1] * self.l[1:-1] * (
                1/(self.s[1:-1]*(1-np.exp(-self.gammas[1:-1] * self.l[1:-1]))) - 1/(self.s[0:-2]*(1-np.exp(self.gammas[1:-1]*self.l[1:-1]))))

            center_diag = np.concatenate(
                [[first_coef], center_diag, [last_coef]])

            C0 = np.diag(center_diag) + np.diag(upper_diag, 1) + \
                np.diag(lower_diag, -1)

        C0 = C0.astype(complex)

        def C(alpha):
            C0[0, -1] += np.exp(-1j * alpha) * self.gammas[0] * self.l[0] / (
                self.s[-1] * (1-np.exp(self.gammas[0]*self.l[0])))
            C0[-1, 0] += -np.exp(1j * alpha) * self.gammas[-1] * self.l[-1] / (
                self.s[-1] * (1-np.exp(-self.gammas[-1]*self.l[-1])))
            return C0

        return C

    def get_generalised_capacitance_matrix(self) -> Callable[[float], np.ndarray]:

        L = np.diag(1 / self.l)
        V = np.diag(self.v_in)

        return lambda alpha: V**2 @ L @ self.get_capacitance_matrix()(alpha)


def convert_periodic_into_finite(
    periodic_problem: NonReciprocalPeriodicSWP1D, i: int
) -> NonReciprocalFiniteSWP1D:
    """
    Returns a Nonreciprocal Finite wave problem with the same properties as the input

    Args:
        periodic_problem (NonReciprocalPeriodicSWP1D): _description_
        i (int): number of cells

    Returns:
        NonReciprocalFiniteSWP1D: _description_
    """
    periodic_problem = copy.deepcopy(periodic_problem)
    return NonReciprocalFiniteSWP1D(
        N=periodic_problem.N * i,
        l=np.concatenate([periodic_problem.l for _ in range(i)]),
        s=np.concatenate([periodic_problem.s for _ in range(i)])[:-1],
        gammas=np.concatenate([periodic_problem.gammas for _ in range(i)]),
        v_in=np.concatenate([periodic_problem.v_in for _ in range(i)]),
        v_out=periodic_problem.v_out,
        delta=periodic_problem.delta,
        omega=periodic_problem.omega,
        uin=periodic_problem.uin,
        duin=periodic_problem.duin,
    )


def convert_finite_into_periodic(
    finite_problem: NonReciprocalFiniteSWP1D, s_N: float | int
) -> NonReciprocalPeriodicSWP1D:
    """
    Returns a Nonreciprocal Periodic wave problem with the same properties as the input

    Args:
        finite_problem (NonReciprocalFiniteSWP1D): _description_
        s_N (float | int): extra distance between the last resonator of one cell and the first resonator of the next cell

    Returns:
        NonReciprocalPeriodicSWP1D
    """
    finite_problem = copy.deepcopy(finite_problem)
    return NonReciprocalPeriodicSWP1D(
        N=finite_problem.N,
        l=finite_problem.l,
        s=np.concatenate([finite_problem.s, [s_N]]),
        gammas=finite_problem.gammas,
        v_in=finite_problem.v_in,
        v_out=finite_problem.v_out,
        delta=finite_problem.delta,
        omega=finite_problem.omega,
        uin=finite_problem.uin,
        duin=finite_problem.duin,
    )
