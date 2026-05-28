"""
Non-reciprocal 3D subwavelength systems with imaginary gauge potential.

Provides:
- NonReciprocalFiniteSWP3D: finite system, gauge along x-axis
- NonReciprocalPeriodicSWP3D: periodic system, gauge along chain axis (x₁)

For the periodic class, the gauge is applied along the chain direction
(x₁-axis). Internally, the multipole expansion uses z as the polar axis,
so the gauge function exp(γ x₁) on the sphere boundary becomes
exp(γ R cos θ) in the rotated frame. This has only m=0 spherical harmonic
components, which couple to the μ=0 lattice sums and produce the
asymmetry Ĉ(−α) ≠ Ĉ(α) needed for the skin effect.
"""

import numpy as np
import scipy as sci

from Subwavelength3D.classic_finite import ClassicFiniteSWP3D
from Subwavelength3D.classic_periodic import ClassicPeriodicFWP3D
import Utils.utils_general as utils

from typing import Literal, Tuple
from typing_extensions import override

from scipy.special import sph_harm


# ============================================================================
# Gauge harmonic expansions
# ============================================================================

def compute_gauge_harmonics(gamma: float, R: float, N_multipole: int, N_quad: int = 200) -> np.ndarray:
    """Expand exp(gamma * R * sin(theta) * cos(phi)) in spherical harmonics.

    This is the gauge along the x-axis, for use with the FINITE system
    where resonators can be at arbitrary positions.

    Uses scipy convention: sph_harm(m, l, theta_azimuthal, phi_polar).

    Args:
        gamma: Gauge potential strength.
        R: Radius of the resonator.
        N_multipole: Maximum multipole order.
        N_quad: Number of quadrature points per angular dimension.

    Returns:
        np.ndarray: Expansion coefficients of shape (N_multipole**2,).
    """
    theta_az = np.linspace(0, 2 * np.pi, N_quad)
    phi_polar = np.linspace(0, np.pi, N_quad)
    THETA, PHI = np.meshgrid(theta_az, phi_polar)

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


def compute_gauge_harmonics_chain_axis(
    gamma: float, R: float, N_multipole: int, N_quad: int = 200
) -> np.ndarray:
    """Expand exp(gamma * R * cos(theta)) in spherical harmonics.

    This is the gauge along the CHAIN axis (z-axis in the multipole frame),
    for use with the PERIODIC system where the chain direction = gauge direction.

    The function exp(γ R cos θ) is azimuthally symmetric, so only m=0
    components are nonzero. This is crucial: the m=0 components couple to
    the μ=0 lattice sums, and the l=1,m=0 component with the antisymmetric
    (odd-n) lattice sum produces the Ĉ(−α) ≠ Ĉ(α) asymmetry.

    Args:
        gamma: Gauge potential strength.
        R: Radius of the resonator.
        N_multipole: Maximum multipole order.
        N_quad: Number of quadrature points.

    Returns:
        np.ndarray: Expansion coefficients of shape (N_multipole**2,).
    """
    theta_az = np.linspace(0, 2 * np.pi, N_quad)
    phi_polar = np.linspace(0, np.pi, N_quad)
    THETA, PHI = np.meshgrid(theta_az, phi_polar)

    # exp(gamma * R * cos(phi_polar)) -- gauge along z (= chain axis in multipole frame)
    # scipy: z = r * cos(phi_polar)
    fun = np.exp(gamma * R * np.cos(PHI))

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

    For gamma=0, this reduces to 4/3 * pi * R^3.
    This is for the x-axis gauge (finite system).
    """
    r_vals = np.linspace(0, R, N_quad)
    theta_vals = np.linspace(0, np.pi, N_quad)
    phi_vals = np.linspace(0, 2 * np.pi, N_quad)

    R_grid, TH_grid, PH_grid = np.meshgrid(r_vals, theta_vals, phi_vals, indexing='ij')
    integrand = np.exp(gamma * R_grid * np.sin(TH_grid) * np.cos(PH_grid)) * R_grid**2 * np.sin(TH_grid)

    val = np.trapz(np.trapz(np.trapz(integrand, phi_vals, axis=2), theta_vals, axis=1), r_vals, axis=0)
    return float(np.real(val))


def compute_normalization_integral_chain_axis(gamma: float, R: float, N_quad: int = 200) -> float:
    """Compute the volume integral of exp(gamma * r * cos(theta)) over a sphere.

    For gamma=0, this reduces to 4/3 * pi * R^3.
    This is for the chain-axis gauge (periodic system).
    """
    r_vals = np.linspace(0, R, N_quad)
    theta_vals = np.linspace(0, np.pi, N_quad)
    phi_vals = np.linspace(0, 2 * np.pi, N_quad)

    R_grid, TH_grid, PH_grid = np.meshgrid(r_vals, theta_vals, phi_vals, indexing='ij')
    # exp(gamma * r * cos(theta)) -- z-axis gauge
    integrand = np.exp(gamma * R_grid * np.cos(TH_grid)) * R_grid**2 * np.sin(TH_grid)

    val = np.trapz(np.trapz(np.trapz(integrand, phi_vals, axis=2), theta_vals, axis=1), r_vals, axis=0)
    return float(np.real(val))


# ============================================================================
# Shared gauge assembly
# ============================================================================

def _assemble_gauge_capacitance_matrix(
    S: np.ndarray,
    N: int,
    N_multipole: int,
    R: float,
    gamma: float,
    chain_coords: np.ndarray,
    f1: np.ndarray,
    f2: np.ndarray,
) -> np.ndarray:
    """Assemble the gauge capacitance matrix given a single-layer potential matrix S.

    Computes: C[i,j] = -conj(phi_i) @ solve(S, psi_j)
    where phi_i = R² * exp(gamma * chain_coord_i) * f1 in block i
    and   psi_j = f2 in block j.
    """
    N_block = N_multipole ** 2
    M_total = N * N_block

    lu_piv = sci.linalg.lu_factor(S)

    C = np.zeros((N, N), dtype=complex)
    phis = np.zeros((M_total, N), dtype=complex)
    psis = np.zeros((M_total, N), dtype=complex)

    for j in range(N):
        phi_j = np.zeros(M_total, dtype=complex)
        phi_j[N_block * j: N_block * (j + 1)] = R**2 * np.exp(gamma * chain_coords[j]) * f1
        phis[:, j] = phi_j

        psi_rhs = np.zeros(M_total, dtype=complex)
        psi_rhs[N_block * j: N_block * (j + 1)] = f2
        psis[:, j] = sci.linalg.lu_solve(lu_piv, psi_rhs)

    for j in range(N):
        for i in range(N):
            C[i, j] = -np.conj(phis[:, i]) @ psis[:, j]

    return C


# ============================================================================
# NonReciprocalFiniteSWP3D (gauge along x-axis)
# ============================================================================

class NonReciprocalFiniteSWP3D(ClassicFiniteSWP3D):
    """3D finite system with imaginary gauge potential along the x-axis.

    All radii must be equal.
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
        """Compute the non-reciprocal capacitance matrix for a finite system.

        The gauge is along the x-axis: exp(gamma * x_1).
        """
        parameters = {
            "N_multipole": N_multipole, "method": method,
            "k0": k0, "N_quad": N_quad, "gamma": self.gamma,
        }

        if self._capacitance_matrix is not None and self._capacitance_matrix_parameters == parameters:
            return self._capacitance_matrix

        if method != 'general':
            raise ValueError(f"Only method='general' is supported, got '{method}'")

        R = self.radii[0]
        N_block = N_multipole ** 2

        S = self.compute_general_single_layer_potential_matrix(N_multipole, k0)

        f1 = compute_gauge_harmonics(self.gamma, R, N_multipole, N_quad)
        f2 = np.zeros(N_block, dtype=complex)
        f2[0] = np.sqrt(4 * np.pi)

        cx = self.centers[:, 0]  # x-coordinates (gauge direction)

        C = _assemble_gauge_capacitance_matrix(S, self.N, N_multipole, R, self.gamma, cx, f1, f2)

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
        """Gauge-modified material matrix. Uses A_norm_j = exp(gamma * cx_j) * int_A."""
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
            "eve_middle_localization", "eve_localization",
            "eva_real", "eva_imag", "eve_abs", "eva_first_val",
        ] = "eva_real",
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Eigenvalues/vectors of the GCM. Uses np.linalg.eig (non-Hermitian)."""
        GCM = self.get_generalised_capacitance_matrix(**kwargs)
        if eigenvalues_only:
            D = np.linalg.eigvals(GCM)
            S = None
        else:
            D, S = np.linalg.eig(GCM)
            S /= np.linalg.norm(S, axis=0)
        D, S = utils.sort_by_method(D, S, sorting)
        return D, S


# ============================================================================
# NonReciprocalPeriodicSWP3D (gauge along chain axis = x₁)
# ============================================================================

class NonReciprocalPeriodicSWP3D(ClassicPeriodicFWP3D):
    """3D periodic system with imaginary gauge potential along the chain axis.

    Implements Definition 5.1 from the JEMS paper. The gauge potential gamma
    is applied along the chain direction (x₁-axis). In the multipole frame
    (where the chain maps to the z-axis), the gauge function on the sphere
    boundary is exp(γ R cos θ), which has only m=0 spherical harmonic
    components.

    The key property: the l≥1, m=0 gauge harmonics couple to the odd-parity
    lattice sums (S_n with odd n), producing Ĉ(−α) ≠ Ĉ(α). This asymmetry
    gives a Laurent polynomial with c_n ≠ c_{-n}, yielding a symbol curve
    with nonzero interior and hence the non-Hermitian skin effect.

    All radii must be equal.
    """

    def __init__(self, gamma: float = 0.0, **pars):
        super().__init__(**pars)
        self.gamma = float(gamma)
        if not np.allclose(self.radii, self.radii[0]):
            raise ValueError("NonReciprocalPeriodicSWP3D requires all radii to be equal")
        self._gauge_harmonics_cache = {}

    def __str__(self):
        return (
            f"Three Dimensional Periodic system with {self.N} resonators per cell.\n"
            f"Physics: Non-reciprocal (gamma={self.gamma})\n"
            f"Lattice period: L={self.L}\n"
            f"Radii: {self.radii[0]}"
        )

    @override
    def get_capacitance_matrix(
        self,
        alpha: float,
        N_multipole: int = 2,
        N_quad: int = 100,
        method: str = 'lattice_sums',
    ) -> np.ndarray:
        """Compute the gauge quasiperiodic capacitance matrix Ĉ^{α,γ}.

        Uses the quasiperiodic SLP S^{α,0} and gauge harmonics for the
        chain-axis gauge exp(γ R cos θ).

        Args:
            alpha: Bloch wave number.
            N_multipole: Maximum multipole order.
            N_quad: Quadrature points for gauge harmonics.
            method: 'lattice_sums' (default) or 'epstein' (monopole, gamma=0 only).

        Returns:
            np.ndarray: Complex N x N gauge capacitance matrix.
        """
        # Auto-optimize: use Epstein for gamma=0 (exact monopole, much faster)
        if method == 'lattice_sums' and abs(self.gamma) < 1e-15 and abs(alpha) > 1e-15:
            method = 'epstein'

        if method == 'epstein':
            if abs(self.gamma) > 1e-15:
                raise ValueError(
                    "method='epstein' only supports gamma=0. "
                    "For non-reciprocal systems, use method='lattice_sums'."
                )
            from Subwavelength3D import epstein
            return epstein.compute_capacitance_matrix_epstein(
                self.centers, self.radii, self.L, alpha)

        R = self.radii[0]
        N_block = N_multipole ** 2

        S = self.compute_single_layer_potential_matrix(
            N_multipole=N_multipole, alpha=alpha)

        # Chain-axis gauge harmonics: cached since they don't depend on alpha
        cache_key = (self.gamma, R, N_multipole, N_quad)
        if cache_key not in self._gauge_harmonics_cache:
            self._gauge_harmonics_cache[cache_key] = compute_gauge_harmonics_chain_axis(
                self.gamma, R, N_multipole, N_quad)
        f1 = self._gauge_harmonics_cache[cache_key]

        f2 = np.zeros(N_block, dtype=complex)
        f2[0] = np.sqrt(4 * np.pi)

        # Chain coordinates (x₁ positions within the unit cell)
        cx = self.centers[:, 0]

        return _assemble_gauge_capacitance_matrix(
            S, self.N, N_multipole, R, self.gamma, cx, f1, f2)

    def get_material_matrix(
        self,
        inverted: bool = False,
        perform_sqrt: bool = False,
        return_only_list: bool = False,
        N_quad: int = 200,
    ) -> np.ndarray:
        """Gauge-modified material matrix for the periodic system."""
        R = self.radii[0]
        cx = self.centers[:, 0]

        int_A = compute_normalization_integral_chain_axis(self.gamma, R, N_quad)
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
    def get_generalised_capacitance_matrix(self, alpha: float, **kwargs) -> np.ndarray:
        """Compute V @ C(alpha)."""
        return self.get_material_matrix() @ self.get_capacitance_matrix(alpha=alpha, **kwargs)

    def compute_sorted_eigs_capacitance_matrix(
        self,
        alpha: float,
        eigenvalues_only: bool = False,
        sorting: Literal[
            "eve_middle_localization", "eve_localization",
            "eva_real", "eva_imag", "eve_abs", "eva_first_val",
        ] = "eva_real",
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Eigenvalues/vectors of the GCM at a given alpha."""
        GCM = self.get_generalised_capacitance_matrix(alpha=alpha, **kwargs)
        if eigenvalues_only:
            D = np.linalg.eigvals(GCM)
            S = None
        else:
            D, S = np.linalg.eig(GCM)
            S /= np.linalg.norm(S, axis=0)
        D, S = utils.sort_by_method(D, S, sorting)
        return D, S

    def compute_band_structure(
        self,
        alphas: np.ndarray,
        N_multipole: int = 2,
        N_quad: int = 200,
    ) -> np.ndarray:
        """Compute eigenvalues across an array of alpha values."""
        eigs = np.zeros((len(alphas), self.N), dtype=complex)
        for k, alpha in enumerate(alphas):
            D, _ = self.compute_sorted_eigs_capacitance_matrix(
                alpha=alpha, N_multipole=N_multipole, N_quad=N_quad)
            eigs[k] = D
        return eigs
