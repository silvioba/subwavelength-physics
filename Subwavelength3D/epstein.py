"""
Epstein zeta accelerated quasiperiodic capacitance matrix computation.

Uses epsteinlib to evaluate the quasiperiodic single-layer potential
directly via the Epstein zeta function, avoiding the polylogarithm-based
lattice sum computation. This is faster and handles intra-cell distances
correctly for multi-resonator unit cells.

Restriction: monopole only (N_multipole=1 equivalent). The Epstein zeta
function encodes the quasiperiodic Green's function:

    G^{alpha}(x) = -(1/4pi) * Z_{Lambda,1}(x | -alpha/(2*pi))

where Z_{Lambda,nu}(x|y) = sum'_{z in Lambda} e^{-2pi*i*y*z} / |x - z|^nu.
"""

import numpy as np

try:
    from epsteinlib import epstein_zeta
except ImportError:
    raise ImportError(
        "epsteinlib is required for the Epstein zeta method. "
        "Install with: pip install epsteinlib"
    )


def compute_single_layer_potential_matrix_epstein(
    centers: np.ndarray,
    radii: np.ndarray,
    L: float,
    alpha: float,
) -> np.ndarray:
    """Compute the monopole quasiperiodic single-layer potential matrix via Epstein zeta.

    For a 1D periodic chain (period L) of N spherical resonators on the x-axis:
        S[i,i] = -R_i - R_i^2 * Z_{Lambda,1}(0 | -alpha/(2*pi))
        S[i,j] = -R_j^2 * Z_{Lambda,1}(r_i - r_j | -alpha/(2*pi))   for i != j

    Args:
        centers: Resonator centers, shape (N, 3). Must be on the x-axis.
        radii: Resonator radii, shape (N,).
        L: Lattice period.
        alpha: Bloch wave number. Must be nonzero.

    Returns:
        np.ndarray: Complex N x N single-layer potential matrix.

    Raises:
        ValueError: If alpha is zero (Epstein zeta diverges for 1D nu=1 at y=0).
    """
    if abs(alpha) < 1e-15:
        raise ValueError(
            "alpha must be nonzero for the Epstein zeta method "
            "(the 1D Epstein zeta with nu=1 diverges at y=0)."
        )

    centers = np.asarray(centers).reshape(-1, 3)
    radii = np.asarray(radii, dtype=float)
    N = len(radii)
    cx = centers[:, 0]

    A = np.array([[L]])
    y = np.array([-alpha / (2 * np.pi)])

    # Self-interaction Epstein zeta (shared by all diagonal entries)
    Z_self = epstein_zeta(1.0, A, np.array([0.0]), y)

    S = np.zeros((N, N), dtype=complex)

    for i in range(N):
        S[i, i] = -radii[i] - radii[i]**2 * Z_self
        for j in range(N):
            if i != j:
                x_ij = np.array([cx[i] - cx[j]])
                S[i, j] = -radii[j]**2 * epstein_zeta(1.0, A, x_ij, y)

    return S


def compute_capacitance_matrix_epstein(
    centers: np.ndarray,
    radii: np.ndarray,
    L: float,
    alpha: float,
) -> np.ndarray:
    """Compute the monopole quasiperiodic capacitance matrix via Epstein zeta.

    C[i,j] = -4*pi * R_i^2 * (S^{-1})_{i,j}

    This is equivalent to ClassicPeriodicFWP3D.get_capacitance_matrix with
    N_multipole=1, but faster.

    Args:
        centers: Resonator centers, shape (N, 3). Must be on the x-axis.
        radii: Resonator radii, shape (N,).
        L: Lattice period.
        alpha: Bloch wave number. Must be nonzero.

    Returns:
        np.ndarray: Complex N x N capacitance matrix.
    """
    radii = np.asarray(radii, dtype=float)
    S = compute_single_layer_potential_matrix_epstein(centers, radii, L, alpha)
    S_inv = np.linalg.inv(S)
    C = -4 * np.pi * radii[:, None]**2 * S_inv
    return C
