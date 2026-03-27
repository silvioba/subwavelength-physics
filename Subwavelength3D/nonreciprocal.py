import numpy as np
import scipy as sci

from Subwavelength3D.classic_finite import ClassicFiniteSWP3D, flat_index
import Utils.utils_general as utils

from typing import Literal, Tuple
from typing_extensions import override

from scipy.special import sph_harm


def compute_gauge_harmonics(gamma: float, R: float, N_multipole: int, N_quad: int = 200) -> np.ndarray:
    """Expand exp(gamma * R * sin(theta) * cos(phi)) in spherical harmonics.

    The function exp(gamma * R * x_hat) where x_hat is the unit vector along x
    is expanded as sum_{l,m} f1_{lm} Y_l^m(theta, phi).

    Uses scipy convention: sph_harm(m, l, theta_azimuthal, phi_polar).

    Args:
        gamma: Gauge potential strength.
        R: Radius of the resonator (all radii assumed equal).
        N_multipole: Maximum multipole order (l goes from 0 to N_multipole-1).
        N_quad: Number of quadrature points per angular dimension.

    Returns:
        np.ndarray: Expansion coefficients of shape (N_multipole**2,).
    """
    theta_az = np.linspace(0, 2 * np.pi, N_quad)
    phi_polar = np.linspace(0, np.pi, N_quad)
    THETA, PHI = np.meshgrid(theta_az, phi_polar)

    # exp(gamma * R * x_hat) = exp(gamma * R * sin(phi_polar) * cos(theta_az))
    fun = np.exp(gamma * R * np.sin(PHI) * np.cos(THETA))

    N_block = N_multipole ** 2
    f1 = np.zeros(N_block, dtype=complex)

    idx = 0
    for l in range(N_multipole):
        for m in range(-l, l + 1):
            Y = sph_harm(m, l, THETA, PHI)
            integrand = fun * np.sin(PHI) * np.conj(Y)
            val = np.trapz(np.trapz(integrand, phi_polar, axis=0), theta_az)
            f1[idx] = val
            idx += 1

    return f1


def compute_normalization_integral(gamma: float, R: float, N_quad: int = 200) -> float:
    """Compute the volume integral of exp(gamma * r * sin(theta) * cos(phi)) over a sphere.

    int_A = integral over [0,R] x [0,pi] x [0,2pi] of
        exp(gamma * r * sin(theta) * cos(phi)) * r^2 * sin(theta) dr dtheta dphi

    For gamma=0, this reduces to 4/3 * pi * R^3.

    Args:
        gamma: Gauge potential strength.
        R: Radius of the sphere.
        N_quad: Number of quadrature points per dimension.

    Returns:
        float: The normalization integral value.
    """
    r_vals = np.linspace(0, R, N_quad)
    theta_vals = np.linspace(0, np.pi, N_quad)   # polar angle (inclination)
    phi_vals = np.linspace(0, 2 * np.pi, N_quad)  # azimuthal angle

    # Build 3D grid: (r, theta, phi)
    # Using scipy convention: x = r * sin(theta) * cos(phi)
    R_grid, TH_grid, PH_grid = np.meshgrid(r_vals, theta_vals, phi_vals, indexing='ij')

    integrand = np.exp(gamma * R_grid * np.sin(TH_grid) * np.cos(PH_grid)) * R_grid**2 * np.sin(TH_grid)

    # Integrate over phi, then theta, then r
    val = np.trapz(np.trapz(np.trapz(integrand, phi_vals, axis=2), theta_vals, axis=1), r_vals, axis=0)
    return float(np.real(val))


class NonReciprocalFiniteSWP3D(ClassicFiniteSWP3D):
    """3D finite system of spherical resonators with imaginary gauge potential.

    The gauge potential gamma is applied along the x-direction, modifying the
    test functions and normalization in the capacitance matrix computation.
    This produces a non-Hermitian capacitance matrix exhibiting the skin effect.

    All radii must be equal (consistent with the underlying theory).

    Based on the MATLAB implementation in Claude/tmp/skin_effect/.
    """

    def __init__(self, gamma: float = 0.0, **pars):
        super().__init__(**pars)
        self.gamma = float(gamma)
        if not np.allclose(self.radii, self.radii[0]):
            raise ValueError("NonReciprocalFiniteSWP3D requires all radii to be equal")

    def __str__(self):
        return (
            f"Three Dimensional Finite system with {self.N} resonators.\n"
            f"Physics: Non-reciprocal (gamma={self.gamma})\n"
            f"Geometry: The first centers are\n{self.centers[:5]}\n"
            f"and the radii are {self.radii[0]}"
        )

    @override
    def get_capacitance_matrix(
        self,
        N_multipole: int = 2,
        method: Literal['general'] = 'general',
        k0: float = 1e-6,
        N_quad: int = 200,
        **kwargs,
    ) -> np.ndarray:
        """Compute the non-reciprocal capacitance matrix.

        The gauge potential modifies the test functions used in the boundary
        integral formulation. The single-layer potential matrix S is unchanged;
        only the right-hand sides (psi) and test functions (phi) are modified.

        For resonator j:
            phi_j block = R^2 * exp(gamma * cx_j) * f1
            psi_j block = f2 (indicator function expansion)

        Then C[i,j] = -conj(phi_i) @ solve(S, psi_j).

        Args:
            N_multipole: Maximum multipole order.
            method: Only 'general' is supported for non-reciprocal systems.
            k0: Wavenumber (small, quasi-static regime).
            N_quad: Number of quadrature points for harmonic expansion.

        Returns:
            np.ndarray: Complex N x N capacitance matrix (non-Hermitian when gamma != 0).
        """
        parameters = {
            "N_multipole": N_multipole,
            "method": method,
            "k0": k0,
            "N_quad": N_quad,
            "gamma": self.gamma,
        }

        if self._capacitance_matrix is not None and self._capacitance_matrix_parameters == parameters:
            return self._capacitance_matrix

        if method != 'general':
            raise ValueError(
                f"Only method='general' is supported for non-reciprocal systems, got '{method}'"
            )

        R = self.radii[0]
        N_block = N_multipole ** 2
        M = self.N * N_block
        cx = self.centers[:, 0]  # x-coordinates of centers

        # Compute single-layer potential matrix (unchanged by gauge)
        S = self.compute_general_single_layer_potential_matrix(N_multipole, k0)

        # Gauge harmonic expansion: exp(gamma * R * sin(theta) * cos(phi))
        f1 = compute_gauge_harmonics(self.gamma, R, N_multipole, N_quad)

        # Indicator function expansion: only monopole (l=0, m=0) component
        f2 = np.zeros(N_block, dtype=complex)
        f2[0] = np.sqrt(4 * np.pi)

        # LU factorize S for efficient multiple solves
        lu_piv = sci.linalg.lu_factor(S)

        # Build phi and solve for psi
        C = np.zeros((self.N, self.N), dtype=complex)
        phis = np.zeros((M, self.N), dtype=complex)
        psis = np.zeros((M, self.N), dtype=complex)

        for j in range(self.N):
            # Modified test function: phi_j = R^2 * exp(gamma * cx_j) * f1
            phi_j = np.zeros(M, dtype=complex)
            phi_j[N_block * j: N_block * (j + 1)] = R**2 * np.exp(self.gamma * cx[j]) * f1
            phis[:, j] = phi_j

            # Standard RHS (indicator function)
            psi_rhs = np.zeros(M, dtype=complex)
            psi_rhs[N_block * j: N_block * (j + 1)] = f2
            psis[:, j] = sci.linalg.lu_solve(lu_piv, psi_rhs)

        # Assemble capacitance matrix: C[i,j] = -conj(phi_i) @ psi_sol_j
        for j in range(self.N):
            for i in range(self.N):
                C[i, j] = -np.conj(phis[:, i]) @ psis[:, j]

        if self.cache_capacitance_matrix:
            self._capacitance_matrix = C
            self._capacitance_matrix_parameters = parameters

        return C

    @override
    def get_material_matrix(
        self,
        inverted: bool = False,
        perform_sqrt: bool = False,
        return_only_list: bool = False,
        N_quad: int = 200,
    ) -> np.ndarray:
        """Get the gauge-modified material matrix.

        Instead of using the standard volume 4/3*pi*R^3, uses the normalization
        A_norm_j = exp(gamma * cx_j) * int_A, where int_A is the volume integral
        of exp(gamma * r * sin(theta) * cos(phi)) over the sphere.

        Args:
            inverted: If True, return the inverse.
            perform_sqrt: If True, return sqrt of diagonal entries.
            return_only_list: If True, return diagonal as 1D array.
            N_quad: Number of quadrature points for normalization integral.

        Returns:
            np.ndarray: Material matrix (diagonal) or its diagonal entries.
        """
        R = self.radii[0]
        cx = self.centers[:, 0]

        int_A = compute_normalization_integral(self.gamma, R, N_quad)
        A_norm = np.exp(self.gamma * cx) * int_A

        if perform_sqrt:
            diag = self.v_in / np.sqrt(A_norm)
        else:
            diag = np.power(self.v_in, 2) / A_norm

        if inverted:
            diag = 1.0 / diag

        if return_only_list:
            return diag
        else:
            return np.diag(diag)

    @override
    def compute_sorted_eigs_capacitance_matrix(
        self,
        eigenvalues_only: bool = False,
        sorting: Literal[
            "eve_middle_localization",
            "eve_localization",
            "eva_real",
            "eva_imag",
            "eve_abs",
            "eva_first_val",
        ] = "eva_real",
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute eigenvalues/vectors of the generalised capacitance matrix.

        Uses numpy.linalg.eig (not eigh) since the GCM is non-Hermitian.

        Args:
            eigenvalues_only: If True, only return eigenvalues.
            sorting: Method for sorting eigenvalues/vectors.
            **kwargs: Passed to get_capacitance_matrix().

        Returns:
            Tuple of (eigenvalues, eigenvectors) sorted by the specified method.
        """
        GCM = self.get_generalised_capacitance_matrix(**kwargs)

        if eigenvalues_only:
            D = np.linalg.eigvals(GCM)
            S = None
        else:
            D, S = np.linalg.eig(GCM)
            S /= np.linalg.norm(S, axis=0)

        D, S = utils.sort_by_method(D, S, sorting)
        return D, S
