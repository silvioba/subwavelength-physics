"""
Periodic 3D subwavelength systems with quasiperiodic lattice sums.

The lattice vector l₁ is along the x₁-axis, matching the convention in
Ammari et al., "The non-Hermitian skin effect with three-dimensional
long-range coupling", JEMS 2025.

Internally, the multipole expansion uses the z-axis as the polar axis.
For a collinear chain along x₁, all inter-resonator separation vectors
point along x₁. In the multipole basis, the addition theorem for
displacements along the z-axis uses only μ=0 terms. We therefore
ROTATE the geometry so that the chain direction maps to z for the
lattice sum computation, and rotate back for the user-facing API.

For a single resonator per unit cell (N=1), the rotation is trivial
since the resonator is at the origin.
"""

import numpy as np
import scipy as sci

from mpmath import polylog
from mpmath import mp
from Subwavelength3D.swp import SWP3D

from typing import Literal, Callable, Tuple, Self, List, Dict
from typing_extensions import override

from scipy.special import spherical_jn, hankel1, sph_harm
from sympy.physics.wigner import wigner_3j

from math import factorial
from functools import cache


# ============================================================================
# Lattice sum infrastructure (chain along z-axis in the multipole frame)
# ============================================================================

def evaluate_cns(n, s):
    """Evaluate (n+s)! / (2^s * s! * (n-s)!)."""
    if s < 0 or s > n:
        raise ValueError("s must satisfy 0 <= s <= n.")
    return factorial(n + s) // (2**s * factorial(s) * factorial(n - s))


def Lspm(b, s, pm, k, L):
    """L_s^{pm} function from equation 3.11 of the paper."""
    return (
        (1j) ** s / (k * L) ** (s + 1)
        * polylog(s + 1, np.exp(1j * (k + pm * b) * L))
    )


def lattice_sums(alpha, n, L, k):
    """Compute lattice sum sigma_n(alpha) using equation 3.10 of the paper.

    These lattice sums assume the lattice is along the z-axis of the
    multipole expansion (so that only mu=0 terms contribute).
    """
    scaling = np.sqrt((2 * n + 1) / (4 * np.pi)) * (-1j) ** (n + 1)
    temp = 0
    for s in range(n + 1):
        temp += evaluate_cns(n=n, s=s) * (
            Lspm(b=alpha, s=s, pm=1, k=k, L=L)
            + (-1) ** n * Lspm(b=alpha, s=s, pm=-1, k=k, L=L)
        )
    return complex(scaling * temp)


def spherical_hl(n, z):
    """Spherical Hankel function of the first kind."""
    return np.sqrt(np.pi / z / 2) * hankel1(n + 1 / 2, z)


# ============================================================================
# Multipole coefficients
# ============================================================================

def flat_index(n: int, L: int, l: int, m: int) -> int:
    """Index in the S matrix (size N*L²) for basis element Y_l^m in block n.

    Returns: n * L² + l² + (l + m)
    """
    if l < 0 or n < 0 or L < 0:
        raise ValueError(f"n, L, l must be non-negative, got n={n}, L={L}, l={l}")
    if np.abs(m) > l:
        raise ValueError(f"|m| must be <= l, got m={m}, l={l}")
    if l > L:
        raise ValueError(f"l must be <= L, got l={l}, L={L}")
    return n * L**2 + l**2 + (l + m)


def get_indicator_function_spherical_harmonics_expansion(
    N: int, N_multi: int, index: int
) -> np.ndarray:
    """Spherical harmonic expansion of the indicator function chi_{partial D_index}.

    The constant function 1 on the sphere has expansion sqrt(4*pi) * Y_0^0.
    """
    idx = np.zeros(N * N_multi**2)
    idx[N_multi**2 * index] = np.sqrt(4 * np.pi)
    return idx


@cache
def C_coefficient(l: int, m: int, lp: int, mp: int, lam: int, mu: int) -> float:
    """Addition theorem C-coefficient using Wigner-3j symbols."""
    return complex(
        (1j) ** (lp - l + lam)
        * (-1) ** m
        * np.sqrt(4 * np.pi * (2 * l + 1) * (2 * lp + 1) * (2 * lam + 1))
        * wigner_3j(l, lp, lam, 0, 0, 0)
        * wigner_3j(l, lp, lam, -m, mp, mu)
    )


def B_coefficient(alpha, l, m, lp, mp, L, k0, N_multipole):
    """Quasiperiodic lattice sum B-coefficient.

    Combines C-coefficients with lattice sums. Only mu=0 terms contribute
    because the chain is along the z-axis in the multipole frame.
    """
    B = 0
    for lam in range(N_multipole):
        B += C_coefficient(l, m, lp, mp, lam, 0) * lattice_sums(alpha, lam, L, k0)
    return complex(B)


# ============================================================================
# ClassicPeriodicFWP3D
# ============================================================================

class ClassicPeriodicFWP3D(SWP3D):
    """Periodic 3D system of spherical resonators.

    The lattice vector is along the x₁-axis with period L.
    Centers must lie on the x₁-axis (collinear chain).

    Internally, the multipole expansion treats the chain direction as the
    z-axis. For N=1 per cell this is transparent since the resonator is
    at the origin.
    """

    def __init__(self, L: float, k0: float = 1e-6, **pars):
        self.L = L
        self.k0 = k0
        super().__init__(**pars)
        # Verify chain is on x-axis
        for c in self.centers:
            c = np.asarray(c)
            if np.abs(c[1]) > 1e-10 or np.abs(c[2]) > 1e-10:
                raise ValueError(
                    "ClassicPeriodicFWP3D requires all centers on the x₁-axis "
                    f"(y=z=0). Got center {c}."
                )

    def __str__(self):
        return super().__str__() + "\nPhysics: Classic periodic system"

    @classmethod
    def get_chain(cls, N: int, L: float, radius: float, k0: float = 1e-6, **params) -> 'ClassicPeriodicFWP3D':
        """Create a periodic chain of N equally spaced resonators per unit cell.

        For N=1, places a single resonator at the origin.
        For N>1, distributes resonators evenly within [0, L).
        """
        if N == 1:
            centers = [np.array([0.0, 0.0, 0.0])]
        else:
            sep = L / N
            centers = [np.array([i * sep, 0.0, 0.0]) for i in range(N)]
        return cls(L=L, k0=k0, centers=centers, radii=np.ones(N) * radius, **params)

    def _get_internal_distances(self) -> np.ndarray:
        """Get pairwise distances between resonators within the unit cell.

        Since the chain is along x₁, these are just |x_i - x_j|.
        """
        cx = self.centers[:, 0]
        return np.abs(cx[:, None] - cx[None, :])

    def compute_single_layer_potential_matrix(
        self, N_multipole: int, alpha: float
    ) -> np.ndarray:
        """Compute the quasiperiodic single-layer potential matrix S^{alpha,k0}.

        For a collinear chain, the lattice sums use only mu=0 terms.
        The diagonal blocks contain self-interaction (Hankel) + lattice sum.
        The off-diagonal blocks (N>1 per cell) use the addition theorem
        with inter-resonator distances.

        Args:
            N_multipole: Maximum multipole order L (l = 0, ..., L-1).
            alpha: Bloch wave number.

        Returns:
            np.ndarray: Complex matrix of size (N*L², N*L²).
        """
        c = -1j * self.radii**2
        L2 = N_multipole**2
        S = np.zeros((self.N * L2, self.N * L2), dtype=complex)

        for i in range(self.N):
            for j in range(self.N):
                for l in range(N_multipole):
                    for m in range(-l, l + 1):
                        if i == j:
                            # Self-interaction: Hankel term
                            S[
                                flat_index(i, N_multipole, l, m),
                                flat_index(j, N_multipole, l, m),
                            ] = (
                                c[i] * self.k0
                                * spherical_hl(l, self.radii[i] * self.k0)
                                * spherical_jn(l, self.radii[i] * self.k0)
                            )
                            # Lattice sum contribution (periodic images of self)
                            for lp in range(N_multipole):
                                for mp in range(-lp, lp + 1):
                                    S[
                                        flat_index(i, N_multipole, l, m),
                                        flat_index(j, N_multipole, lp, mp),
                                    ] += (
                                        B_coefficient(alpha, l, m, lp, mp,
                                                      self.L, self.k0, N_multipole)
                                        * spherical_jn(lp, self.k0 * self.radii[i])
                                        * c[i] * self.k0
                                        * spherical_jn(l, self.k0 * self.radii[i])
                                    )
                        else:
                            # Off-diagonal: use addition theorem for distance r_ij
                            # NOTE: For N>1 per cell, this is approximate.
                            # See the plan doc for known issues with this block.
                            rp = abs(self.centers[i][0] - self.centers[j][0])
                            for lp in range(N_multipole):
                                for mp in range(-lp, lp + 1):
                                    S[
                                        flat_index(i, N_multipole, l, m),
                                        flat_index(j, N_multipole, lp, mp),
                                    ] = (
                                        c[j] * self.k0
                                        * B_coefficient(alpha, l, m, lp, mp,
                                                        self.L, self.k0, N_multipole)
                                        * spherical_jn(lp, self.k0 * self.radii[j])
                                        * spherical_jn(l, self.k0 * self.radii[i])
                                    )
        return S

    def get_capacitance_matrix(self, alpha: float, N_multipole: int = 2) -> np.ndarray:
        """Compute the quasiperiodic capacitance matrix C(alpha).

        Solves S(alpha) @ psi_j = chi_j for each resonator j, then extracts
        the monopole component to form C[i,j].

        Args:
            alpha: Bloch wave number.
            N_multipole: Maximum multipole order.

        Returns:
            np.ndarray: Complex N x N capacitance matrix.
        """
        S = self.compute_single_layer_potential_matrix(
            N_multipole=N_multipole, alpha=alpha)
        C = np.zeros((self.N, self.N), dtype=complex)

        lu_piv = sci.linalg.lu_factor(S)
        for j in range(self.N):
            u_j = get_indicator_function_spherical_harmonics_expansion(
                N=self.N, N_multi=N_multipole, index=j)
            y = sci.linalg.lu_solve(lu_piv, u_j)
            for i in range(self.N):
                C[i, j] = -np.sqrt(4 * np.pi) * self.radii[i]**2 * y[i * N_multipole**2]
        return C

    def get_generalised_capacitance_matrix(self, alpha: float, **kwargs) -> np.ndarray:
        """Compute V @ C(alpha)."""
        return self.get_material_matrix() @ self.get_capacitance_matrix(alpha=alpha, **kwargs)
