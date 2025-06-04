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

from typing import Literal, Callable, Tuple, Self, List, override

import copy

from Utils.settings import settings

import Utils.utils_general as utils

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

    def __get_capacitance_diagonal(self) -> np.ndarray:
        assert self.N > 1, "N must be greater than 1 to compute capacitance matrix"
        first_coef = self.gammas[0] * self.l[0] / (
            self.s[0] * (1-np.exp(-self.gammas[0]*self.l[0])))
        last_coef = - self.gammas[-1] * self.l[-1] / (
            self.s[-1] * (1-np.exp(self.gammas[-1]*self.l[-1])))

        center_diag = self.gammas[1:-1] * self.l[1:-1] * (
            1/(self.s[1:]*(1-np.exp(-self.gammas[1:-1] * self.l[1:-1]))) - 1/(self.s[:-1]*(1-np.exp(self.gammas[1:-1]*self.l[1:-1]))))

        center_diag = np.concatenate([[first_coef], center_diag, [last_coef]])

        # Handle the case where gammas are very small to avoid 0/0
        low_gammas = np.nonzero(np.abs(self.gammas) < 1e-14)[0]
        for i in low_gammas:
            if i == 0:
                center_diag[0] = 1 / self.s[0]
            elif i == self.N - 1:
                center_diag[-1] = 1 / self.s[-1]
            else:
                center_diag[i] = 1 / self.s[i-1] + 1 / self.s[i]
        return center_diag

    def __get_capacitance_upper_offdiagonal(self) -> np.ndarray:
        assert self.N > 1, "N must be greater than 1 to compute capacitance matrix"
        upper_diag = - self.gammas[:-1] * self.l[:-1] / (
            self.s * (1-np.exp(-self.gammas[:-1]*self.l[:-1])))
        # Handle the case where gammas are very small to avoid 0/0
        low_gammas = np.nonzero(np.abs(self.gammas[:-1]) < 1e-14)[0]
        for i in low_gammas:
            upper_diag[i] = - 1 / self.s[i]
        return upper_diag

    def __get_capacitance_lower_offdiagonal(self) -> np.ndarray:
        assert self.N > 1, "N must be greater than 1 to compute capacitance matrix"
        lower_diag = self.gammas[1:] * self.l[1:] / (
            self.s * (1-np.exp(self.gammas[1:]*self.l[1:])))
        # Handle the case where gammas are very small to avoid 0/0
        low_gammas = np.nonzero(np.abs(self.gammas[1:]) < 1e-14)[0]
        for i in low_gammas:
            lower_diag[i] = - 1 / self.s[i]
        return lower_diag

    @override
    def get_capacitance_matrix(self) -> np.ndarray:
        """
        Computes the gauge capacitance matrix C from eq (20) in [1]. Note that the paper contains wrong indicies. This implementation is corrected.

        Returns:
            np.ndarray:
        """
        center_diag = self.__get_capacitance_diagonal()
        upper_diag = self.__get_capacitance_upper_offdiagonal()
        lower_diag = self.__get_capacitance_lower_offdiagonal()

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
        return self.get_material_matrix() @ self.get_capacitance_matrix()

    def get_symmetrised_generalised_capacitance_matrix(self) -> np.ndarray:
        """
        Computes the gauge capacitance matrix C premultiplied by V^2 L^{-1} where V is the diagonal matrix of wavespeeds inside the resonators and L the matrix of lengths of the resonators

        Returns:
            np.ndarray:
        """
        V = self.get_material_matrix(return_only_list=True)
        center_diag = self.__get_capacitance_diagonal()
        upper_diag = self.__get_capacitance_upper_offdiagonal()
        lower_diag = self.__get_capacitance_lower_offdiagonal()

        a = V*center_diag
        b = V[:-1]*upper_diag
        c = V[1:]*lower_diag

        d = np.sign(b)*np.sqrt(b*c)

        C = np.diag(a) + np.diag(d, 1) + \
            np.diag(d, -1)

        return C

    @override
    def compute_sorted_eigs_capacitance_matrix(
        self,
        eigenvalues_only=False,
        generalised=True,
        real_symmetrisation_acceleratrion=True,
        sorting: Literal[
            "eve_middle_localization",
            "eve_localization",
            "eva_real",
            "eva_imag",
            "eve_abs",
            "eva_first_val",
        ] = "eva_real",
    ) -> Tuple[np.ndarray, np.ndarray]:
        if real_symmetrisation_acceleratrion:
            if generalised:
                V = self.get_material_matrix(return_only_list=True)
                center_diag = self.__get_capacitance_diagonal()
                upper_diag = self.__get_capacitance_upper_offdiagonal()
                lower_diag = self.__get_capacitance_lower_offdiagonal()

                a = V*center_diag
                b = V[:-1]*upper_diag
                c = V[1:]*lower_diag

                d = np.sign(b)*np.sqrt(b*c)
            else:
                a = self.__get_capacitance_diagonal()
                b = self.__get_capacitance_upper_offdiagonal()
                c = self.__get_capacitance_lower_offdiagonal()

                d = np.sign(b)*np.sqrt(b*c)
            if eigenvalues_only:
                D = sci.linalg.eigh_tridiagonal(
                    a, d, eigvals_only=True,
                )
                S = None
            else:
                D, St = sci.linalg.eigh_tridiagonal(
                    a, d, eigvals_only=False,
                )
                cp = np.sqrt(np.concatenate(([1.], np.cumprod(c/b))))
                CP = np.diag(cp)
                S = CP@St
                S = S / np.linalg.norm(S, axis=0)
        else:
            if generalised:
                mat = self.get_generalised_capacitance_matrix()
            else:
                mat = self.get_capacitance_matrix()

            if eigenvalues_only:
                D = np.linalg.eigvals(mat)
                S = None
            else:
                D, S = np.linalg.eig(mat)

        D, S = utils.sort_by_method(D, S, sorting)
        return D, S

    def get_resonator_propagation_matrix(self, j, space_from_end: float = 1.0, regularised: bool = True):
        if self.omega is None:
            raise ValueError("omega must be set, is currently None")
        if np.linalg.norm(self.k_in - np.ones(self.N) * self.k_out) > 1e-8:
            raise NotImplementedError(
                "Propagation matrix is implemented only for structure with same wave number inside and outside."
            )

        if j == self.N - 1:
            p = utils_propagation.nonreciprocal_subwavelength_propagation_matrix_single(
                l=self.l[-1],
                s=space_from_end,
                gamma=self.gammas[-1],
                lbda=self.k_in[-1],
                regularised=True
            )
        else:
            p = utils_propagation.nonreciprocal_subwavelength_propagation_matrix_single(
                l=self.l[j],
                s=self.s[j],
                gamma=self.gammas[j],
                lbda=self.k_in[j],
                regularised=True
            )
        return p

    def compute_propagation_matrix(
        self, space_from_end: float = 1.0, regularised: bool = True
    ) -> np.ndarray:
        pm = np.eye(2)
        for j in range(self.N):
            p = self.get_resonator_propagation_matrix(
                j=j, space_from_end=space_from_end, regularised=regularised)
            pm = p @ pm
        return pm

    def compute_Lyapunov_exponent(self, space_from_end=1, rescale_every=20, max_N=None) -> float:
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
                j=j, space_from_end=space_from_end, regularised=True)
            pm = p @ pm
            gamma_li_sum += self.gammas[j] * self.l[j]
            if rescale_every is not None and ((j+1) % rescale_every == 0):
                norm = np.linalg.norm(pm)
                log_norm_sum += np.log(norm)
                pm /= norm

        log_norm_sum += np.log(np.linalg.norm(pm))
        return log_norm_sum/max_N - 1/(2 * max_N) * gamma_li_sum


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

    @override
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

    @override
    def compute_generalised_capacitance_matrix(self) -> Callable[[float], np.ndarray]:
        """
        Computes the generalised capacitance matrix as a function of the Bloch wave number alpha.

        Returns:
            Callable[[float], np.ndarray]: A function that maps alpha to the generalised capacitance matrix.
        """
        return lambda alpha: self.get_material_matrix() @ self.get_capacitance_matrix()(alpha)

    @override
    def compute_sorted_eigs_capacitance_matrix(
        self,
        eigenvalues_only=False,
        generalised=True,
        sorting: Literal[
            "eve_middle_localization",
            "eve_localization",
            "eva_real",
            "eva_imag",
            "eve_abs",
            "eva_first_val",
        ] = "eva_real",
    ) -> Callable[[float], Tuple[np.ndarray, np.ndarray]]:
        def eig(alpha):
            if generalised:
                mat = self.compute_generalised_capacitance_matrix()(alpha)
            else:
                mat = self.get_capacitance_matrix()(alpha)

            if eigenvalues_only:
                D = np.linalg.eigvals(mat)
                S = None
            else:
                D, S = np.linalg.eig(mat)
            return D, S

        return eig


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
