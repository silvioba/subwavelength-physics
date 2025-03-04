import numpy as np
import scipy as sci
from scipy.sparse.linalg import eigs
from Subwavelength1D.swp import FiniteSWP1D
from typing import Literal, Callable, Tuple, Self, List, override
from Utils.utils_general import sort_by_eva_real


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
        self.epsilon_kappa = epsilon_kappa
        self.phase_kappa = np.array(
            phase_kappa, dtype=float) if phase_kappa is not None else np.zeros(self.N)
        self.epsilon_rho = epsilon_rho
        self.phase_rho = np.array(
            phase_rho, dtype=float) if phase_rho is not None else np.zeros(self.N)
        self.big_omega = big_omega if big_omega is not None else np.sqrt(
            self.delta)

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

    @override
    def get_sorted_eigs_capacitance_matrix(self, N_fourier: int = 4, n_eigs: int | None = None, generalised: bool = True, which: str = 'SM', maxiter: int = 1000, return_eigenvectors: bool = False):
        """
        Computes the approximate spectrum (Floquet exponents) of the time-modulated
        capacitance system:

        .. math::
            \\rho(t)^{-1} \\frac{d}{dt}\\Bigl(\\kappa(t)^{-1}\\frac{d}{dt}[\\rho(t)\\,\\Psi]\\Bigr) + \\delta\\,\\mathrm{diag}\\Bigl(\\frac{1}{l_i}\\Bigr)\\,\\mathrm{diag}(v_{r,i}^2)\\,C \\,\\Psi \\;=\\; 0,

        using the spectral (Fourier) method in time.  It solves a large block-diagonal
        eigenvalue problem for the expansions of :math:`\\Psi(t)`, restricted to
        Fourier harmonics :math:`|n|\\leq N_{\\text{fourier}}`.

        Parameters
        ----------
        N_fourier : int
            Highest Fourier order for the solution expansions.  The large system
            dimension is :math:`2\\,N\\,(2\\,N_{\\mathrm{fourier}}+1)`.
        n_eigs : int or None
            Number of smallest-magnitude eigenvalues to extract. Defaults to 2*N.
        use_generalised : bool
            If True, we use the \"generalised\" matrix  :math:`\\delta\\,\\mathrm{diag}(\\tfrac{1}{l_i})\\,\\mathrm{diag}(v_{r,i}^2)\\,C`.
            If False, we use just the simpler matrix :math:`C`.
        which : {\"SM\", \"LM\", ...}
            Passed to `scipy.sparse.linalg.eigs` - \"SM\" means smallest magnitude,
            \"LM\" largest magnitude, etc.
        return_eigenvectors : bool
            If True, we also return the large matrix\'s eigenvectors. (Be aware of the memory cost!)

        Returns
        -------
        w_out : np.ndarray
            Sorted 1D array of the requested eigenvalues (length = n_eigs).
        None or np.ndarray
            By default None. If ``return_eigenvectors=True``, returns the matrix
            of eigenvectors corresponding to ``w_out``.

        Notes
        -----
        - The code closely follows the MATLAB approach.  The complex matrix dimension
          is :math:`2 (N \\times NN)` with :math:`NN = 2\\,N_{\\mathrm{fourier}} + 1`.
        - The final sort is by real part of the eigenvalues.
        """
        if generalised:
            GCM = self.delta * self.get_generalised_capacitance_matrix()
        else:
            GCM = self.delta * self.get_capacitance_matrix()
        Nres = self.N
        if n_eigs is None:
            n_eigs = 2 * Nres
        M = 1
        R_mod = np.zeros((2 * M + 1, Nres), dtype=complex)
        K_mod = np.zeros((2 * M + 1, Nres), dtype=complex)
        for i in range(Nres):
            R_mod[M, i] = 1.0
            R_mod[M - 1, i] = self.epsilon_rho / 2.0 * \
                np.exp((-1j) * self.phase_rho[i])
            R_mod[M + 1, i] = self.epsilon_rho / \
                2.0 * np.exp(1j * self.phase_rho[i])
            K_mod[M, i] = 1.0
            K_mod[M - 1, i] = self.epsilon_kappa / 2.0 * \
                np.exp((-1j) * self.phase_kappa[i])
            K_mod[M + 1, i] = self.epsilon_kappa / \
                2.0 * np.exp(1j * self.phase_kappa[i])
        ns = np.arange(-N_fourier, N_fourier + 1)
        NN = 2 * N_fourier + 1
        O_block = np.diag(ns * self.big_omega)
        dim_block = Nres * NN
        iK = np.zeros((dim_block, dim_block), dtype=complex)
        R_mat = np.zeros((dim_block, dim_block), dtype=complex)
        iR_mat = np.zeros((dim_block, dim_block), dtype=complex)

        def blk_slice(i):
            r1 = i * NN
            r2 = (i + 1) * NN
            return slice(r1, r2)
        for i in range(Nres):
            Ki = self.__get_toeplitz_form_from_fourier_coeffs(
                K_mod[:, i], N_fourier)
            Ri = self.__get_toeplitz_form_from_fourier_coeffs(
                R_mod[:, i], N_fourier)
            Ki_inv = np.linalg.inv(Ki)
            Ri_inv = np.linalg.inv(Ri)
            slc = blk_slice(i)
            iK[slc, slc] = Ki_inv
            R_mat[slc, slc] = Ri
            iR_mat[slc, slc] = Ri_inv
        I_NN = np.eye(NN, dtype=complex)
        big_GCM = np.kron(GCM, I_NN)
        iRcR = iR_mat @ big_GCM @ R_mat
        block_size = dim_block
        zero_block = np.zeros((block_size, block_size), dtype=complex)
        big_I_N = np.eye(Nres, dtype=complex)
        big_O = np.kron(big_I_N, O_block)
        mat_upperleft = -big_O
        mat_upperright = zero_block
        mat_lowerleft = zero_block
        mat_lowerright = -big_O
        mat_upperright = mat_upperright - 1j * iK
        mat_lowerleft = mat_lowerleft + 1j * iRcR
        top = np.concatenate([mat_upperleft, mat_upperright], axis=1)
        bottom = np.concatenate([mat_lowerleft, mat_lowerright], axis=1)
        mat = np.concatenate([top, bottom], axis=0)
        dim_mat = mat.shape[0]
        if n_eigs >= dim_mat:
            raise ValueError(
                f'Requested n_eigs={n_eigs} but matrix dimension is {dim_mat}. Must have n_eigs < matrix dimension.')
        D, S = np.linalg.eig(mat)
        idx_selection = np.argsort(np.abs(D.real))[:n_eigs]
        D = D[idx_selection]
        S = S[:, idx_selection] if return_eigenvectors else None
        D_sorted, S_sorted = sort_by_eva_real(D, S)
        return (D_sorted[-n_eigs // 2:], S_sorted[:, -n_eigs // 2:] if S_sorted is not None else None)

    def get_spectral_mat(self, N_fourier: int = 4, n_eigs: int | None = None, generalised: bool = True):
        if generalised:
            GCM = self.delta * self.get_generalised_capacitance_matrix()
        else:
            GCM = self.delta * self.get_capacitance_matrix()
        Nres = self.N
        Mexp = 2 * N_fourier + 1
        dim_block = Nres * Mexp
        if n_eigs is None:
            n_eigs = 2 * Nres
        Mmod = 1
        k_mod = np.zeros((2 * Mmod + 1, Nres), dtype=complex)
        for i in range(Nres):
            k_mod[Mmod, i] = 1.0
            k_mod[Mmod - 1, i] = self.epsilon_kappa / \
                2.0 * np.exp((-1j) * self.phase_kappa[i])
            k_mod[Mmod + 1, i] = self.epsilon_kappa / \
                2.0 * np.exp(1j * self.phase_kappa[i])
        ns = np.arange(-N_fourier, N_fourier + 1)
        Omega_block = np.diag((-1j) * ns * self.big_omega)
        I_Nres = np.eye(Nres, dtype=complex)
        big_Omega = np.kron(I_Nres, Omega_block)

        def blk_slice(i):
            r1 = i * Mexp
            r2 = (i + 1) * Mexp
            return slice(r1, r2)
        big_iK = np.zeros((dim_block, dim_block), dtype=complex)
        for i in range(Nres):
            Ki = self.__get_toeplitz_form_from_fourier_coeffs(
                k_mod[:, i], N_fourier)
            Ki_inv = np.linalg.inv(Ki)
            slc = blk_slice(i)
            big_iK[slc, slc] = Ki_inv
        I_Mexp = np.eye(Mexp, dtype=complex)
        big_GCM = np.kron(GCM, I_Mexp)
        mat_upperleft = (-1j) * big_Omega
        mat_upperright = 1j * big_iK
        mat_lowerleft = (-1j) * big_GCM
        mat_lowerright = (-1j) * big_Omega
        top = np.concatenate([mat_upperleft, mat_upperright], axis=1)
        bottom = np.concatenate([mat_lowerleft, mat_lowerright], axis=1)
        mat = np.concatenate([top, bottom], axis=0)
        return mat

    def simplified_get_sorted_eigs_capacitance_matrix(self, N_fourier: int = 4, n_eigs: int | None = None, generalised: bool = True, which: str = 'SM', maxiter: int = 1000, return_eigenvectors: bool = False):
        if generalised:
            GCM = self.delta * self.get_generalised_capacitance_matrix()
        else:
            GCM = self.delta * self.get_capacitance_matrix()
        Nres = self.N
        Mexp = 2 * N_fourier + 1
        dim_block = Nres * Mexp
        if n_eigs is None:
            n_eigs = 2 * Nres
        Mmod = 1
        k_mod = np.zeros((2 * Mmod + 1, Nres), dtype=complex)
        for i in range(Nres):
            k_mod[Mmod, i] = 1.0
            k_mod[Mmod - 1, i] = self.epsilon_kappa / \
                2.0 * np.exp((-1j) * self.phase_kappa[i])
            k_mod[Mmod + 1, i] = self.epsilon_kappa / \
                2.0 * np.exp(1j * self.phase_kappa[i])
        ns = np.arange(-N_fourier, N_fourier + 1)
        Omega_block = np.diag((-1j) * ns * self.big_omega)
        I_Nres = np.eye(Nres, dtype=complex)
        big_Omega = np.kron(I_Nres, Omega_block)

        def blk_slice(i):
            r1 = i * Mexp
            r2 = (i + 1) * Mexp
            return slice(r1, r2)
        big_iK = np.zeros((dim_block, dim_block), dtype=complex)
        for i in range(Nres):
            Ki = self.__get_toeplitz_form_from_fourier_coeffs(
                k_mod[:, i], N_fourier)
            Ki_inv = np.linalg.inv(Ki)
            slc = blk_slice(i)
            big_iK[slc, slc] = Ki_inv
        I_Mexp = np.eye(Mexp, dtype=complex)
        big_GCM = np.kron(GCM, I_Mexp)
        mat_upperleft = (-1j) * big_Omega
        mat_upperright = 1j * big_iK
        mat_lowerleft = (-1j) * big_GCM
        mat_lowerright = (-1j) * big_Omega
        top = np.concatenate([mat_upperleft, mat_upperright], axis=1)
        bottom = np.concatenate([mat_lowerleft, mat_lowerright], axis=1)
        mat = np.concatenate([top, bottom], axis=0)
        dim_mat = mat.shape[0]
        if n_eigs >= dim_mat:
            raise ValueError(
                f'Requested n_eigs={n_eigs} but matrix dimension is {dim_mat}. Must have n_eigs < matrix dimension.')
        D, S = np.linalg.eig(mat)
        idx_selection = np.argsort(np.abs(D.real))[:n_eigs]
        D = D[idx_selection]
        S = S[:, idx_selection] if return_eigenvectors else None
        D_sorted, S_sorted = sort_by_eva_real(D, S)
        return (D_sorted[-n_eigs // 2:], S_sorted[:, -n_eigs // 2:] if S_sorted is not None else None)

    def simplified_get_sorted_eigs_capacitance_matrix_hot(self, N_fourier: int = 4, n_eigs: int | None = None, generalised: bool = True, which: str = 'SM', maxiter: int = 1000, return_eigenvectors: bool = False):
        if generalised:
            GCM = self.delta * self.get_generalised_capacitance_matrix()
        else:
            GCM = self.delta * self.get_capacitance_matrix()
        Nres = self.N
        Mexp = 2 * N_fourier + 1
        dim_block = Nres * Mexp
        if n_eigs is None:
            n_eigs = 2 * Nres
        Mmod = 1
        k_mod = np.zeros((2 * Mmod + 1, Nres), dtype=complex)
        for i in range(Nres):
            k_mod[Mmod, i] = 1.0
            k_mod[Mmod - 1, i] = self.epsilon_kappa / \
                2.0 * np.exp((-1j) * self.phase_kappa[i])
            k_mod[Mmod + 1, i] = self.epsilon_kappa / \
                2.0 * np.exp(1j * self.phase_kappa[i])
        ns = np.arange(-N_fourier, N_fourier + 1)
        Omega_block = np.diag((-1j) * ns * self.big_omega)
        I_Nres = np.eye(Nres, dtype=complex)
        big_Omega = np.kron(I_Nres, Omega_block)

        def blk_slice(i):
            r1 = i * Mexp
            r2 = (i + 1) * Mexp
            return slice(r1, r2)
        big_K = np.zeros((dim_block, dim_block), dtype=complex)
        big_iK = np.zeros((dim_block, dim_block), dtype=complex)
        for i in range(Nres):
            Ki = self.__get_toeplitz_form_from_fourier_coeffs(
                k_mod[:, i], N_fourier)
            Ki_inv = np.linalg.inv(Ki)
            slc = blk_slice(i)
            big_K[slc, slc] = Ki
            big_iK[slc, slc] = Ki_inv
        I_Mexp = np.eye(Mexp, dtype=complex)
        big_GCM = np.kron(GCM, I_Mexp)
        alpha = self.delta * self.get_material_matrix()
        big_alpha = np.kron(alpha, I_Mexp)
        mat_upperleft = (-1j) * big_Omega
        mat_upperright = 1j * big_iK
        mat_lowerleft = (-1j) * big_GCM
        mat_lowerright = (-1j) * (big_alpha @ big_iK + big_Omega)
        top = np.concatenate([mat_upperleft, mat_upperright], axis=1)
        bottom = np.concatenate([mat_lowerleft, mat_lowerright], axis=1)
        mat = np.concatenate([top, bottom], axis=0)
        dim_mat = mat.shape[0]
        if n_eigs >= dim_mat:
            raise ValueError(
                f'Requested n_eigs={n_eigs} but matrix dimension is {dim_mat}. Must have n_eigs < matrix dimension.')
        D, S = np.linalg.eig(mat)
        idx_selection = np.argsort(np.abs(D.real))[:n_eigs]
        D = D[idx_selection]
        S = S[:, idx_selection] if return_eigenvectors else None
        D_sorted, S_sorted = sort_by_eva_real(D, S)
        return (D_sorted[-n_eigs // 2:], S_sorted[:, -n_eigs // 2:] if S_sorted is not None else None)
