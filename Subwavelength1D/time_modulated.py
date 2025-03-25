import numpy as np
import scipy as sci
from scipy.sparse.linalg import eigs
from Subwavelength1D.swp import FiniteSWP1D
from typing import Literal, Callable, Tuple, Self, List, override
from Utils.utils_general import sort_by_eva_real
import Utils.utils_general as utils

import time


class TimeModulatedFiniteSWP1D(FiniteSWP1D):
    """
    A time-modulated extension of the FiniteSWP1D class that uses a spectral (Fourier)
    approach to compute the \"approximate spectrum\" of a capacitance matrix under
    time-dependent material parameters.

    This class implements a method mirroring the MATLAB code:
    .. code-block:: matlab

        function w_out = get_capacitance_approx_spec(epsilon_kappa, phase_kappa, ...
                epsilon_rho, phase_rho, Omega, delta, vr, li, C)
            % ...
            % w_out = sort(eigs(mat, 2*N, \'smallestabs\'), \'ComparisonMethod\', \'real\');
        end

    We override ``get_sorted_eigs_capacitance_matrix`` so that it constructs and solves
    a large block-structured eigenvalue problem in the Floquet sense.

    Notes:
        - We only return the eigenvalues ``w_out`` (as a 1D array) and ``None`` for the
          eigenvectors to keep consistency with the parent method\'s signature
          ``-> Tuple[np.ndarray, np.ndarray]``.
        - This method *ignores* the usual arguments like ``generalised=True`` or
          ``sorting=...``, because the system is time-dependent. The \"sorting\" in the
          MATLAB code is effectively \"by real part\" of the smallest-magnitude eigenvalues.
        - By default we truncate the *time* expansion at ±``N_fourier`` and the *material
          modulation* expansions at ±``M=1`` (fundamental). You can modify these defaults
          as needed.
    """

    def __init__(self, epsilon_kappa: float = 0, phase_kappa: List[float] = None, epsilon_rho: float = 0, phase_rho: List[float] = None, big_omega: float | None = None, **pars):
        super().__init__(**pars)

        self.set_epsilon_kappa(epsilon_kappa)
        self.phase_kappa = np.array(
            phase_kappa, dtype=float) if phase_kappa is not None else np.zeros(self.N)
        self.epsilon_rho = epsilon_rho
        self.phase_rho = np.array(
            phase_rho, dtype=float) if phase_rho is not None else np.zeros(self.N)
        self.big_omega = big_omega if big_omega is not None else np.sqrt(
            self.delta)

    def set_epsilon_kappa(self, epsilon_kappa: float | int | complex | np.ndarray):
        if (
            isinstance(epsilon_kappa, float)
            or isinstance(epsilon_kappa, int)
            or isinstance(epsilon_kappa, complex)
        ):
            epsilon_kappa = (
                np.ones(self.N, dtype=complex if isinstance(
                    epsilon_kappa, complex) else float) * epsilon_kappa
            )
        self.epsilon_kappa = epsilon_kappa

    def __get_capacitance_diagonal(self) -> np.ndarray:
        """
        Computes the diagonal of the capacitance matrix C from eq (1.13) in [1]. Only depends on the spacings between resonators.

        Returns:
            np.ndarray
        """
        assert self.N > 1, 'N must be greater than 1 to compute capacitance matrix'
        d1 = np.concatenate(
            ([1 / self.s[0]], 1 / self.s[:(-1)] + 1 / self.s[1:], [1 / self.s[(-1)]]))
        return d1

    def __get_capacitance_offdiagonal(self) -> np.ndarray:
        """
        Computes the off-diagonal of the capacitance matrix C from eq (1.13) in [1]. Only depends on the spacings between resonators.

        Returns:
            np.ndarray
        """
        assert self.N > 1, 'N must be greater than 1 to compute capacitance matrix'
        d2 = (-1) / self.s
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
        C = np.diag(d1) + np.diag(d2, 1) + np.diag(d2, (-1))
        return C

    @override
    def get_generalised_capacitance_matrix(self) -> np.ndarray:
        """
        Computes the capacitance matrix C from eq (1.13) in [1] premultiplied by V^2 L^{-1} where V is the diagonal matrix of wave speeds inside the resonators and L the matrix of lengths of the resonators

        Returns:
            np.ndarray:
        """
        return self.get_material_matrix() @ self.get_capacitance_matrix()

    def __get_toeplitz_form_from_fourier_coeffs(self, coeffs: np.ndarray, M: int | None = None):
        coeff_M = len(coeffs) // 2
        if M is None:
            M = coeff_M
        T = np.zeros((2 * M + 1, 2 * M + 1), dtype=complex)
        for k in range(-M, M + 1):
            coeff = coeffs[k + coeff_M] if 0 <= k + \
                coeff_M < len(coeffs) else 0
            T += np.diag([coeff] * (2 * M + 1 - abs(k)), k)
        return T

    def get_spectral_matrix(self, N_fourier: int = 4, generalised: bool = True, hot: bool = False):
        # Get the appropriate capacitance matrix
        if generalised:
            GCM = self.delta * self.get_generalised_capacitance_matrix()
        else:
            GCM = self.delta * self.get_capacitance_matrix()
        Nres = self.N  # Number of resonators
        Mexp = 2 * N_fourier + 1  # Total number of Fourier coefficients
        dim_block = Nres * Mexp  # Total size of the matrices in Fourier space

        # Construct big_Omega, corresponding to the time derivative
        ns = np.arange(-N_fourier, N_fourier + 1)
        Omega_block = np.diag((-1j) * ns * self.big_omega)
        I_Nres = np.eye(Nres, dtype=complex)
        big_Omega = np.kron(I_Nres, Omega_block)

        Mmod = 1  # Order of Fourier coefficients of the inverse of the material modulation
        # Calculate the Fourier coefficients of the inverse of the material modulation
        kappa_inv_coeffs = np.zeros((2 * Mmod + 1, Nres), dtype=complex)
        for i in range(Nres):
            # For each resonator we have a Fourier expansion of the inverse material modulation:
            # (self.epsilon_kappa / 2.0 * np.exp(-1j * self.phase_kappa[i]), 1, self.epsilon_kappa / 2.0 * np.exp(1j * self.phase_kappa[i]))
            kappa_inv_coeffs[Mmod, i] = 1.0
            kappa_inv_coeffs[Mmod - 1, i] = self.epsilon_kappa[i] / \
                2.0 * np.exp(-1j * self.phase_kappa[i])
            kappa_inv_coeffs[Mmod + 1, i] = self.epsilon_kappa[i] / \
                2.0 * np.exp(1j * self.phase_kappa[i])

        # Helper function to get the slice of length (Mexp) containing all Fourier coefficients corresponding to the i-th resonator
        def blk_slice(i):
            r1 = i * Mexp
            r2 = (i + 1) * Mexp
            return slice(r1, r2)

        # Construct big_K, corresponding to the material modulation
        # To that end we have to calculate the inverse of the Fourier coefficients of the inverse material modulation obtained above
        big_K = np.zeros((dim_block, dim_block), dtype=complex)
        for i in range(Nres):
            # Using the inverse material modulation Fourier coefficients we construct the Toeplitz matrix
            # corresponding to the i-th resonator
            Ki = self.__get_toeplitz_form_from_fourier_coeffs(
                kappa_inv_coeffs[:, i], N_fourier)
            # Taking the inverse yields the Fourier coefficients of the (uninverted) material modulation
            Ki_inv = np.linalg.inv(Ki)
            slc = blk_slice(i)
            big_K[slc, slc] = Ki_inv

        # Construct big_GCM, corresponding to the capacitance matrix
        I_Mexp = np.eye(Mexp, dtype=complex)
        big_GCM = np.kron(GCM, I_Mexp)

        # Construct the matrix
        mat_upperleft = (-1j) * big_Omega
        mat_upperright = 1j * big_K
        mat_lowerleft = (-1j) * big_GCM
        # If hot is True, we have to include the higher order terms
        if hot:
            alpha = self.delta * self.get_material_matrix()
            big_alpha = np.kron(alpha, I_Mexp)
            mat_lowerright = (-1j) * (big_alpha @ big_K + big_Omega)
        else:
            mat_lowerright = (-1j) * big_Omega

        top = np.concatenate([mat_upperleft, mat_upperright], axis=1)
        bottom = np.concatenate([mat_lowerleft, mat_lowerright], axis=1)
        mat = np.concatenate([top, bottom], axis=0)
        return mat

    @override
    def get_sorted_eigs_capacitance_matrix(
            self,
            N_fourier: int = 4,
            generalised: bool = True,
            hot: bool = False,
            return_eigenvectors: bool = True,
            sparse: bool = True,
            sigma_factor: float = 5,
            debug_time: bool = False,
            sorting: Literal[
                "eve_middle_localization",
                "eve_localization",
                "eva_real",
                "eva_imag",
                "eve_abs",
                "eva_first_val",
            ] = "eva_real"):

        if debug_time:
            t0 = time.perf_counter()

        mat = self.get_spectral_matrix(
            N_fourier=N_fourier, generalised=generalised, hot=hot)

        if debug_time:
            t1 = time.perf_counter()
            print(f'Construction of spectral matrix took {t1-t0:.9f}s')

        n_eigs = self.N

        if sparse:
            # To ensure numerical stability we have to add a small sigma ideally in (lambda_0, lambda_1) where lambda_0 is the eigenvalue with the smallest positive real part
            # because we know that there must be self.N eigenvalues in the first brillouin zone (0, self.big_omega) we choose sigma as below
            sigma = self.big_omega / (sigma_factor*self.N)
            sol = sci.sparse.linalg.eigs(
                mat, k=2*n_eigs, which='LM', sigma=sigma, return_eigenvectors=return_eigenvectors)
            if debug_time:
                t2 = time.perf_counter()
                print(f'Calculation of eigenvalues took {t2-t1:.9f}s')

            D = sol[0] if return_eigenvectors else sol
            S = sol[1] if return_eigenvectors else None

            D, S = sort_by_eva_real(D, S)

            # We look for the indices where D has almost 0 real part and then choose the middle one to split between positive and negative solutions
            i_zeros = []
            for i, d in enumerate(D):
                if np.isclose(np.real(d), 0):
                    i_zeros.append(i)
            i_base = i_zeros[len(i_zeros) // 2]

            # Then, starting from i_base, we return the N eigenvalues with the smallest positive real part
            # These should be all the eigenvalues in the first brillouin zone
            idx = np.arange(i_base, i_base+n_eigs)
            D = D[idx]
            S = S[:, idx] if return_eigenvectors else None
            if debug_time:
                t3 = time.perf_counter()
                print(f'Eva selection took {t3-t2:.9f}s')
        else:
            # Without sparse acceleration we just calculate all the eigenvalues, select the 2Nres ones with smallest absulute real part (to avoid folding)
            # and of those select the Nres ones with largest real part
            if return_eigenvectors:
                D, S = np.linalg.eig(mat)
            else:
                D = np.linalg.eigvals(mat)
                S = None

            if debug_time:
                t2 = time.perf_counter()
                print(f'Calculation of eigenvalues took {t2-t1:.9f}s')

            idx_selection = np.argsort(np.abs(D.real))[:n_eigs*2]
            D = D[idx_selection]
            S = S[:, idx_selection] if return_eigenvectors else None
            D_sorted, S_sorted = sort_by_eva_real(D, S)
            D, S = (D_sorted[-n_eigs:], (S_sorted[:, -n_eigs:]
                    if S_sorted is not None else None))

            if debug_time:
                t3 = time.perf_counter()
                print(f'Eva selection took {t3-t2:.9f}s')

        D, S = utils.sort_by_method(D, S, sorting)
        if debug_time:
            t4 = time.perf_counter()
            print(f'Final sorting took {t4-t3:.9f}s')
        return D, S
