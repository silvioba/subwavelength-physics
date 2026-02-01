import numpy as np
import scipy as sci
from Subwavelength1D.swp import (
    FiniteSWP1D,
    PeriodicSWP1D,
)

import Utils.utils_propagation as utils_propagation

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap, LogNorm

from typing import Literal, Callable, Tuple, Self, List
from typing_extensions import override

import copy

from Utils.settings import settings

import Utils.utils_general as utils

plt.rcParams.update(settings.matplotlib_params)


def check_parameters_inconsitencies(fwp: FiniteSWP1D):
    if not (np.abs(fwp.k_in - fwp.omega / fwp.v_in) < 1e-6).all():
        raise ValueError("k_in does not equal omega / v_in")
    if not (np.abs(fwp.k_out - fwp.omega / fwp.v_out) < 1e-6).all():
        raise ValueError("k_in does not equal omega / v_in")


def _Q(x, r1, r2):
    return np.array([
        [np.exp(r1 * x), np.exp(r2 * x)],
        [r1 * np.exp(r1 * x), r2 * np.exp(r2 * x)]
    ])


class NonSubwavelengthNonReciprocalFiniteSWP1D(FiniteSWP1D):
    """
    Base class for nonreciprocal finite acoustic subwavelength wave problem. Subclass of OneDimensionalFiniteSWLProblem

    Initially modelled on https://arxiv.org/abs/2306.15587 (subsequently referred as [1]), subsequently extended
    """

    def __init__(self, gammas=1, **pars):
        super().__init__(**pars)
        if isinstance(gammas, (int, float)):
            gammas = np.ones(self.N) * gammas
        self.gammas = np.array(gammas, dtype=float)

    @override
    def get_physics(self):
        return "Non-subwavelength, Non-reciprocal"

    def set_params(self, **params):
        for key, val in params.items():
            if hasattr(self, key):
                setattr(self, key, val)
            else:
                raise AttributeError(
                    f"{self.__class__.__name__} has no attribute '{key}'"
                )
        check_parameters_inconsitencies(self)

    def get_resonator_propagation_matrix(self, j, space_from_end: float = 1.0, subwavelength=False, symmetrised: bool = True):
        if self.omega is None:
            raise ValueError("omega must be set, is currently None")
        if np.linalg.norm(self.k_in - np.ones(self.N) * self.k_out) > 1e-8:
            raise NotImplementedError(
                "Propagation matrix is implemented only for structure with same wave number inside and outside."
            )
        if subwavelength:
            raise NotImplementedError(
                "Use NonReciprocalFiniteSWP1D instead."
            )

        if j == self.N - 1:
            p = utils_propagation.nonreciprocal_propagation_matrix_single(
                l=self.l[-1],
                s=space_from_end,
                gamma=self.gammas[-1],
                z=self.omega,
                delta=self.delta,
                symmetrised=symmetrised,
                subwavelength=False
            )
        else:
            p = utils_propagation.nonreciprocal_propagation_matrix_single(
                l=self.l[j],
                s=self.s[j],
                gamma=self.gammas[j],
                z=self.omega,
                delta=self.delta,
                symmetrised=symmetrised,
                subwavelength=False
            )
        return p

    @override
    def compute_propagation_matrix(
        self, space_from_end: float = 1.0, subwavelength: bool = False, symmetrised: bool = True
    ) -> np.ndarray:
        pm = np.eye(2)
        for j in range(self.N):
            p = self.get_resonator_propagation_matrix(
                j=j, space_from_end=space_from_end, subwavelength=subwavelength, symmetrised=symmetrised)
            pm = p @ pm
        return pm

    def compute_resonance_BC_mismatch(self):
        P = self.compute_propagation_matrix(space_from_end=0, symmetrised=True)
        u_r = P @ np.array([1, -1j*self.omega])
        u_r = u_r / u_r[0]
        return np.abs(u_r[1] / (-1j*self.omega) - 1)

    def compute_Lyapunov_exponent(self, space_from_end=1, rescale_every=20, max_N=None, return_parts=False) -> float:
        """
        Computes the Lyapunov exponent for the finite subwavelength wave problem.
        The Lyapunov exponent is a measure of the exponential growth rate of the wave function.
        It is computed using the propagation matrix.
        Raises:
            ValueError: If omega is not set.
            NotImplementedError: If the wave number inside and outside the structure are not the same.
        Returns:
            float: The Lyapunov exponent.
        """
        if max_N is None:
            max_N = self.N
        pm = np.eye(2, dtype=float)
        log_norm_sum = 0.0
        gamma_li_sum = 0.0
        for j in range(max_N):
            p = self.get_resonator_propagation_matrix(
                j=j, space_from_end=space_from_end, symmetrised=True)
            pm = p @ pm
            gamma_li_sum += self.gammas[j] * self.l[j]
            if rescale_every is not None and ((j+1) % rescale_every == 0):
                norm = np.linalg.norm(pm)
                log_norm_sum += np.log(norm)
                pm /= norm

        log_norm_sum += np.log(np.linalg.norm(pm))
        if return_parts:
            return log_norm_sum/max_N, 1/(2 * max_N) * gamma_li_sum
        else:
            return log_norm_sum/max_N - 1/(2 * max_N) * gamma_li_sum

    def solve_u(self, alpha_0=None):
        assert self.omega is not None, "omega must be set, is currently None"

        alphas = np.zeros((self.N + 1, 2), dtype=complex)
        aas = np.zeros((self.N, 2), dtype=complex)
        if alpha_0 is None:
            alphas[0] = [0.0, 1.0]
        else:
            alphas[0] = alpha_0

        D = np.diag([1, self.delta])
        Dinv = np.diag([1, 1/self.delta])

        def _Xi(gamma, omega):
            nu = np.sqrt(complex((gamma/2)**2-omega**2))
            return -gamma / 2 + nu, -gamma / 2 - nu

        for j in range(self.N):
            # alpha outside to u inside
            u_outside_m = _Q(
                self.xim[j], 1j*self.omega, -1j*self.omega) @ alphas[j]
            u_inside_m = D @ u_outside_m

            # calculate a inside
            xi1, xi2 = _Xi(self.gammas[j], self.omega)
            aas[j] = np.linalg.inv(_Q(self.xim[j], xi1, xi2)) @ u_inside_m

            # a inside to alpha outside
            u_inside_p = _Q(self.xip[j], xi1, xi2) @ aas[j]
            u_outside_p = Dinv @ u_inside_p
            alphas[j+1] = np.linalg.inv(_Q(self.xip[j],
                                        1j*self.omega, -1j*self.omega)) @ u_outside_p

        self.alphas = alphas
        self.aas = aas

    def Gamma(self, x):
        # Returns \int_0^x gamma(x') dx'
        j = np.searchsorted(self.xi, x) - 1
        if j % 2 == 0:
            # Inside resonator
            i = (j // 2)
            xi1 = self.xi[j]
            return np.sum(self.gammas[:i] * self.l[:i]) + self.gammas[i] * (x - xi1)
        else:
            # Outside resonator
            i = ((j + 1) // 2)
            xi1 = self.xi[j]
            return np.sum(self.gammas[:i] * self.l[:i])

    def u(self, x, return_inside=False):
        def _Xi(gamma, omega):
            nu = np.sqrt(complex((gamma/2)**2-omega**2))
            return -gamma / 2 + nu, -gamma / 2 - nu

        # Find j such that self.xi[j] < x < self.xi[j+1]
        j = np.searchsorted(self.xi, x) - 1
        if j % 2 == 0:
            # Inside resonator
            i = (j // 2)
            xi1, xi2 = _Xi(self.gammas[i], self.omega)
            u = self.aas[i][0]*np.exp(xi1*x) + self.aas[i][1]*np.exp(xi2*x)
            if return_inside:
                return u, True
            else:
                return u
        else:
            # Outside resonator
            i = ((j + 1) // 2)
            u = self.alphas[i][0]*np.exp(1j*self.omega*x) + \
                self.alphas[i][1]*np.exp(-1j*self.omega*x)
            if return_inside:
                return u, False
            else:
                return u
