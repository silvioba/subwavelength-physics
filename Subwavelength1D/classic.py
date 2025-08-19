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
from matplotlib.axes import Axes

from typing import Literal, Callable, Tuple, Self, List
from typing_extensions import override

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
    FiniteSWP1D.__doc__ + """
    Base class for acoustic subwavelength wave problem. Subclass of OneDimensionalFiniteSWLProblem

    Initially modelled on [1] (see README), subsequently extended
    """

    def __init__(self, **pars):
        super().__init__(**pars)

    @override
    def get_physics(self):
        return "Classic"

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

    def __get_capacitance_diagonal(self) -> np.ndarray:
        """
        Computes the diagonal of the capacitance matrix C from eq (1.13) in [1]. Only depends on the spacings between resonators.

        Returns:
            np.ndarray
        """
        assert self.N > 1, "N must be greater than 1 to compute capacitance matrix"
        d1 = np.concatenate(
            (
                [1 / self.s[0]],
                1 / self.s[:-1] + 1 / self.s[1:],
                [1 / self.s[-1]],
            )
        )
        return d1

    def __get_capacitance_offdiagonal(self) -> np.ndarray:
        """
        Computes the off-diagonal of the capacitance matrix C from eq (1.13) in [1]. Only depends on the spacings between resonators.

        Returns:
            np.ndarray
        """
        assert self.N > 1, "N must be greater than 1 to compute capacitance matrix"
        d2 = -1 / self.s
        return d2

    @override
    def get_capacitance_matrix(self) -> np.ndarray:
        """
        Computes the capacitance matrix C from eq (1.13) in [1]. Only depends on the spacings between resonators.

        Returns:
            np.ndarray
        """
        d1 = self.__get_capacitance_diagonal()
        d2 = self.__get_capacitance_offdiagonal()
        C = np.diag(d1) + np.diag(d2, 1) + np.diag(d2, -1)
        return C

    @override
    def get_generalised_capacitance_matrix(self) -> np.ndarray:
        """
        Computes the capacitance matrix C from eq (1.13) in [1] premultiplied by V^2 L^{-1} where V is the diagonal matrix of wave speeds inside the resonators and L the matrix of lengths of the resonators

        Returns:
            np.ndarray:
        """
        return self.get_material_matrix() @ self.get_capacitance_matrix()

    @override
    def get_periodized_system(self, sN=None):
        """Get the periodized system of the disordered system by calculating s_N and converting the finite system into a periodic one.

        Returns:
            pwp: Periodized system
        """
        if sN is None:
            raise ValueError(
                "sN must be provided to convert the finite system into a periodic one"
            )
        pwp = convert_finite_into_periodic(self, sN)
        return pwp

    @override
    def compute_sorted_eigs_capacitance_matrix(
        self,
        eigenvalues_only: bool = False,
        generalised: bool = True,
        hermitian_acceleration: bool = True,
        sorting: Literal[
            "eve_middle_localization",
            "eve_localization",
            "eva_real",
            "eva_imag",
            "eve_abs",
            "eva_first_val",
        ] = "eva_real",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Computes the eigendecomposition of the capacitance matrix

        Args:
            eigenvalues_only (bool, optional): Compute only eigenvalues. Defaults to False.
            generalised (bool, optional): Use the generalised capacitance matrix. Defaults to True.
            hermitian_acceleration (bool, optional): Use acceleration in case of hermitian matrix. Defaults to True.
            sorting (Literal[ &quot;eve_middle_localization&quot;, &quot;eve_localization&quot;, &quot;eva_real&quot;, &quot;eva_imag&quot;, &quot;eve_abs&quot;, &quot;eva_first_val&quot;, ], optional): _description_. Defaults to "eva_real".

        Returns:
            Tuple[np.ndarray, np.ndarray]: matrix of eigenvalues , matrix of eigenvectors
        """
        if hermitian_acceleration:
            if generalised:
                Vl = self.get_material_matrix(
                    inverted=False, perform_sqrt=True, return_only_list=True)
                cdiag = self.__get_capacitance_diagonal()*(Vl**2)
                coffdiag = self.__get_capacitance_offdiagonal()*(
                    Vl[:-1]*Vl[1:])
                if eigenvalues_only:
                    D = sci.linalg.eigh_tridiagonal(
                        cdiag, coffdiag, eigvals_only=True
                    )
                    S = None
                else:
                    D, St = sci.linalg.eigh_tridiagonal(cdiag, coffdiag)
                    S = Vl.reshape(-1, 1) * St
                    S = S / np.linalg.norm(S, axis=0)
            else:
                cdiag = self.__get_capacitance_diagonal()
                coffdiag = self.__get_capacitance_offdiagonal()
                if eigenvalues_only:
                    D = sci.linalg.eigh_tridiagonal(
                        cdiag, coffdiag, eigvals_only=True,
                    )
                    S = None
                else:
                    D, S = sci.linalg.eigh_tridiagonal(cdiag, coffdiag)
        else:
            if generalised:
                D, S = np.linalg.eig(self.get_generalised_capacitance_matrix())
            else:
                D, S = np.linalg.eigh(self.get_capacitance_matrix())

        D, S = utils.sort_by_method(D, S, sorting)
        return D, S

    def compute_spectral_range_capacitance_matrix(
        self,
        select='a',
        select_range=None,
        eigenvalues_only=True,
        sorting: Literal[
            "eve_middle_localization",
            "eve_localization",
            "eva_real",
            "eva_imag",
            "eve_abs",
            "eva_first_val",
        ] = "eva_real",
    ) -> np.ndarray:
        """
        Calculates the eigenvalues and optionally eigenvectors of the generalized capacitance matrix.

        This function computes the eigenvalues and eigenvectors of the generalized capacitance matrix,
        which is scaled by material properties. The computation is performed efficiently using
        scipy's tridiagonal eigenvalue solver.

            - 'v': Eigenvalues in the specified value range will be computed
        eigenvalues_only : bool, default=True
            If True, only eigenvalues are returned. If False, both eigenvalues and eigenvectors are returned.
        sorting : str, default="eva_real"
            Method for sorting eigenvalues and eigenvectors:
            - "eve_middle_localization": Sort by eigenvector localization at the middle of the domain
            - "eve_localization": Sort by eigenvector localization
            - "eva_real": Sort by real part of eigenvalues
            - "eva_imag": Sort by imaginary part of eigenvalues
            - "eve_abs": Sort by absolute value of eigenvectors
            - "eva_first_val": Sort by first value of eigenvalues

        tuple or np.ndarray
            If eigenvalues_only=True, returns np.ndarray of eigenvalues.
            If eigenvalues_only=False, returns a tuple (D, S) where:
                - D: np.ndarray of eigenvalues
                - S: np.ndarray of eigenvectors, normalized and properly scaled

        The generalized capacitance matrix is constructed by scaling the regular capacitance matrix
        with material properties. The eigenvalues and eigenvectors are computed using 
        scipy.linalg.eigh_tridiagonal for efficiency, as the capacitance matrix has a tridiagonal structure.

        The eigenvectors are scaled by the material properties and then normalized.
        """

        Vl = self.get_material_matrix(
            inverted=False, perform_sqrt=True, return_only_list=True)
        cdiag = self.__get_capacitance_diagonal()*(Vl**2)
        coffdiag = self.__get_capacitance_offdiagonal()*(
            Vl[:-1]*Vl[1:])
        if eigenvalues_only:
            D = sci.linalg.eigh_tridiagonal(
                cdiag, coffdiag, eigvals_only=True,
                select=select, select_range=select_range
            )
            S = None
        else:
            D, St = sci.linalg.eigh_tridiagonal(
                cdiag, coffdiag, eigvals_only=False,
                select=select, select_range=select_range
            )
            S = Vl.reshape(-1, 1) * St
            S = S / np.linalg.norm(S, axis=0)

        D, S = utils.sort_by_method(D, S, sorting)
        return D, S

    def compute_greens_matrix(self, k: float):
        """Computes the green Matrix / descrete green function

        Computes C - k*id

        Args:
            k (float): see description

        Returns:
            np.array: descrete green function
        """
        return np.linalg.inv(
            self.get_generalised_capacitance_matrix() - k * np.eye(self.N)
        )

    def get_resonator_propagation_matrix(self, j, space_from_end: float = 1.0, subwavelength: bool = True):
        """
        Computes the propagation matrix for a single resonator.

        Args:
            j (int): The index of the resonator.
            space_from_end (float, optional): The spacing from the last resonator to the end of the domain. Defaults to 1.0.
            subwavelength (bool, optional): Whether to use the subwavelength approximation. Defaults to True.

        Returns:
            np.ndarray: The propagation matrix for the j-th resonator.
        """
        if self.omega is None:
            raise ValueError("omega must be set, is currently None")
        if np.linalg.norm(self.k_in - np.ones(self.N) * self.k_out) > 1e-8:
            raise NotImplementedError(
                "Propagation matrix is implemented only for structure with same wave number inside and outside."
            )

        if j == self.N - 1:
            p = utils_propagation.propagation_matrix_single(
                l=self.l[-1],
                s=space_from_end,
                k=self.k_in[-1],
                delta=self.delta,
                subwavelength=subwavelength,
            )
        else:
            p = utils_propagation.propagation_matrix_single(
                l=self.l[j],
                s=self.s[j],
                k=self.k_in[j],
                delta=self.delta,
                subwavelength=subwavelength,
            )
        return p

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

        pm = np.eye(2)
        for j in range(self.N):
            p = self.get_resonator_propagation_matrix(
                j=j, space_from_end=space_from_end, subwavelength=subwavelength)
            pm = p @ pm
        return pm

    def compute_Lyapunov_exponent(self, space_from_end=1, subwavelength: bool = True, rescale_every=20) -> float:
        """
        Computes the Lyapunov exponent for the finite subwavelength wave problem.
        The Lyapunov exponent is a measure of the exponential growth rate of the wave function.
        It is computed using the propagation matrix.
        Args:
            subwavelength (bool, optional): Whether to use the subwavelength approximation. Defaults to True.
        Raises:
            ValueError: If omega is not set.
            NotImplementedError: If the wave number inside and outside the structure are not the same.
        Returns:
            float: The Lyapunov exponent.
        """
        pm = np.eye(2, dtype=float)
        log_norm_sum = 0.0
        for j in range(self.N):
            p = self.get_resonator_propagation_matrix(
                j=j, space_from_end=space_from_end, subwavelength=subwavelength)
            pm = p @ pm
            if rescale_every is not None and ((j+1) % rescale_every == 0):
                norm = np.linalg.norm(pm)
                log_norm_sum += np.log(norm)
                pm /= norm

        log_norm_sum += np.log(np.linalg.norm(pm))
        return log_norm_sum/self.N

    def compute_reflection_and_transmission(self, subwavelength: bool = True) -> Tuple[float, float]:
        """
        Computes the transmission and reflection coefficients for the finite subwavelength wave problem.
        We do this by using the Q matrix to go to the A B basis and then applying the boundary conditions u_in,L = 1, u_in,R = 0.

        Args:
            subwavelength (bool, optional): Whether to use the subwavelength approximation. Defaults to True.

        Returns:
            Tuple[float, float]: The transmission and reflection coefficients.
        """
        pm = self.compute_propagation_matrix(
            subwavelength=subwavelength, space_from_end=0)
        Q0 = utils_propagation.get_Q_matrix(self.k_out, 0)
        QL = utils_propagation.get_Q_matrix(self.k_out, self.L)
        M = np.linalg.inv(QL) @ pm @ Q0
        Rtot = - M[1, 0] / M[1, 1]
        Ttot = M[0, 0] + M[0, 1] * Rtot
        return np.abs(Rtot), np.abs(Ttot)


class ClassicPeriodicSWP1D(PeriodicSWP1D):
    """
    Base class for acoustic subwavelength wave problem. Subclass of OneDimensionalPeriodicSWLProblem

    Initially modelled on [2] (see README), subsequently extended
    """

    def __init__(self, **pars):
        super().__init__(**pars)

    @override
    def get_physics(self):
        return "Classic"

    @override
    def get_capacitance_matrix(self) -> Callable[[float], np.ndarray]:
        """
        Computes the capacitance matrix C from Lemma 4.7 in [2]. Only depends on the spacings between resonators.

        The alpha parameter is the Bloch wave number and must be in [-np.pi, np.pi)

        Returns:
            Callable[[float], np.ndarray]: map alpha -> C^alpha
        """
        if self.N == 1:
            C0 = np.array([[
                1/self.s[0] + 1/self.s[-1]
            ]])
        else:
            d1 = np.concatenate(
                (
                    [1 / self.s[0] + 1 / self.s[-1]],
                    1 / self.s[1:-1] + 1 / self.s[0:-2],
                    [1 / self.s[-1] + 1 / self.s[-2]],
                )
            )
            d2 = -1 / self.s[:-1]
            C0 = np.zeros((self.N, self.N), dtype=complex)
            C0 += np.diag(d1) + np.diag(d2, 1) + np.diag(d2, -1)

        C0 = C0.astype(complex)

        def C(alpha) -> np.ndarray:
            if not -np.pi <= alpha <= np.pi:
                raise ValueError(
                    f"alpha must be in [-pi, pi), you provided {alpha}")
            C0[0, -1] += -np.exp(-1j * alpha) / self.s[-1]
            C0[-1, 0] += -np.exp(1j * alpha) / self.s[-1]
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
        eigenvalues_only: bool = False,
        generalised: bool = True,
        hermitian_acceleration: bool = True,
        sorting: Literal[
            "eve_middle_localization",
            "eve_localization",
            "eva_real",
            "eva_imag",
            "eve_abs",
            "eva_first_val",
        ] = "eva_real",
    ) -> Callable[[float], Tuple[np.ndarray, np.ndarray]]:
        """Compute the eigenpais of the generalised capacitance matrix.

        Returns a callable function with taking a float alpha and returning D(alpha) and V(alpha)

        Args:
            eigenvalues_only (bool, optional): Compute only eigenvalues. Defaults to False.
            generalised (bool, optional): Use the generalised capacitance matrix. Defaults to True.
            hermitian_acceleration (bool, optional): In case of hermitian system, use acceleration. Defaults to True.
            sorting (Literal[ &quot;eve_middle_localization&quot;, &quot;eve_localization&quot;, &quot;eva_real&quot;, &quot;eva_imag&quot;, &quot;eve_abs&quot;, &quot;eva_first_val&quot;, ], optional): _description_. Defaults to "eva_real".

        Returns:
            Callable[[float], Tuple[np.ndarray, np.ndarray]]: D(alpha), V(alpha)
        """
        def eig(alpha):
            if hermitian_acceleration:
                if generalised:
                    V = self.get_material_matrix(inverted=True)
                    C = self.get_capacitance_matrix()(alpha)
                    if eigenvalues_only:
                        D = sci.linalg.eigvalsh(C, b=V)
                        S = None
                    else:
                        D, S = sci.linalg.eigh(C, b=V)
                else:
                    if eigenvalues_only:
                        D = sci.linalg.eigvalsh(
                            self.get_capacitance_matrix()(alpha))
                        S = None
                    else:
                        D, S = sci.linalg.eigh(
                            self.get_capacitance_matrix()(alpha))
            else:
                if generalised:
                    if eigenvalues_only:
                        D = np.linalg.eigvals(
                            self.compute_generalised_capacitance_matrix()(alpha))
                        S = None
                    else:
                        D, S = np.linalg.eig(
                            self.compute_generalised_capacitance_matrix()(alpha))
                else:
                    if eigenvalues_only:
                        D = np.linalg.eigvalsh(
                            self.get_capacitance_matrix()(alpha))
                        S = None
                    else:
                        D, S = np.linalg.eigh(
                            self.get_capacitance_matrix()(alpha))

            D, S = utils.sort_by_method(D, S, sorting)
            return D, S
        return eig

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
    finite_problem = copy.deepcopy(finite_problem)
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
