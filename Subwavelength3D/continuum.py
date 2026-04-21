"""Exact (non-subwavelength, nonzero-delta) resonance solver for finite 3D
systems of spherical resonators.

Reference: Frequency-dependent capacitance matrix formulation for Fabry-Perot
resonances. Part II: Three-dimensional systems (eq. 2.4).

The resonance characteristic equation is

    A(omega, delta) (psi, phi)^T = 0

with block operator

    A = [[ S_tilde^omega_D,           -S^k_D                       ],
         [ -1/2 I + K_tilde^{omega,*}, -delta_{partial D}(1/2 I + K^{k,*}) ]]

where k = omega / v (exterior) and S_tilde^omega_D, K_tilde^{omega,*}_D use
the per-resonator interior wavenumber k_j = omega / v_{in,j}.

We discretise on a spherical-harmonic basis of size L = N_multipole**2 per
resonator (total system dim = 2 * N * L) and seek resonances by root-finding
on omega -> det(A(omega, delta)) using Muller's method with deflation.
"""

import numpy as np
from functools import cache
from typing import Tuple, List, Dict, Sequence

from scipy.special import spherical_jn
from scipy.optimize import brentq

from Subwavelength3D.swp import SWP3D, _get_consistent_parameter
from Subwavelength3D.classic_finite import (
    spherical_hl,
    flat_index,
    A_coefficient_general,
)
from Utils.utils_general import find_roots_muller


def _spherical_hl_derivative(l: int, z: complex) -> complex:
    """Derivative of the spherical Hankel function of the first kind.

    Uses h_l'(z) = -h_{l+1}(z) + (l/z) h_l(z), valid for all l >= 0.
    """
    return -spherical_hl(l + 1, z) + (l / z) * spherical_hl(l, z)


# ---------------------------------------------------------------------------
# Per-block matrix-element helpers (cached scalar functions).
#
# Index convention (matches S_coefficient_* in classic_finite.py):
#     matrix[(i, l, m), (j, lp, mp)] at row resonator i, column resonator j,
#     with xb = centers[j] - centers[i].
# ---------------------------------------------------------------------------


@cache
def S_tilde_diagonal(l: int, k_i: complex, Ri: float) -> complex:
    """Self-block of S_tilde^omega on resonator i, using interior k_i."""
    return -1j * k_i * Ri**2 * spherical_jn(l, k_i * Ri) * spherical_hl(l, k_i * Ri)


@cache
def S_tilde_offdiagonal(l: int, m: int, lp: int, mp: int, k_i: complex,
                        xb: Tuple[float, float, float], Ri: float, Rj: float,
                        max_lam: int) -> complex:
    """Off-diagonal block (i != j) of S_tilde^omega; row wavenumber k_i is
    used for the transfer kernel (the paper's piecewise operator definition).
    """
    return (-1j * k_i * Rj**2
            * A_coefficient_general(l, m, lp, mp, k_i, xb, max_lam)
            * spherical_jn(l, k_i * Ri) * spherical_jn(lp, k_i * Rj))


@cache
def S_exterior_diagonal(l: int, k: complex, Ri: float) -> complex:
    """Self-block of S^k (exterior wavenumber)."""
    return -1j * k * Ri**2 * spherical_jn(l, k * Ri) * spherical_hl(l, k * Ri)


@cache
def S_exterior_offdiagonal(l: int, m: int, lp: int, mp: int, k: complex,
                           xb: Tuple[float, float, float], Ri: float, Rj: float,
                           max_lam: int) -> complex:
    """Off-diagonal block of S^k (exterior wavenumber)."""
    return (-1j * k * Rj**2
            * A_coefficient_general(l, m, lp, mp, k, xb, max_lam)
            * spherical_jn(l, k * Ri) * spherical_jn(lp, k * Rj))


@cache
def K_tilde_star_diagonal(l: int, k_i: complex, Ri: float) -> complex:
    """Self-block of (-1/2 I + K_tilde^{omega,*}) on resonator i, interior k_i.

    Equals the interior normal derivative of S_tilde^omega[Y_l^m] on partial D_i,
    i.e. -i k_i^2 R_i^2 h_l(k_i R_i) j_l'(k_i R_i).
    """
    return (-1j * k_i**2 * Ri**2
            * spherical_hl(l, k_i * Ri)
            * spherical_jn(l, k_i * Ri, derivative=True))


@cache
def K_tilde_star_offdiagonal(l: int, m: int, lp: int, mp: int, k_i: complex,
                             xb: Tuple[float, float, float], Ri: float, Rj: float,
                             max_lam: int) -> complex:
    """Off-diagonal block of (-1/2 I + K_tilde^{omega,*}); receiver-side j_l' factor."""
    return (-1j * k_i**2 * Rj**2
            * A_coefficient_general(l, m, lp, mp, k_i, xb, max_lam)
            * spherical_jn(l, k_i * Ri, derivative=True)
            * spherical_jn(lp, k_i * Rj))


@cache
def K_exterior_star_diagonal(l: int, k: complex, Ri: float) -> complex:
    """Self-block of (1/2 I + K^{k,*}) on resonator i, exterior wavenumber.

    Equals the exterior normal derivative of S^k[Y_l^m] on partial D_i,
    i.e. -i k^2 R_i^2 j_l(k R_i) h_l'(k R_i).
    """
    return (-1j * k**2 * Ri**2
            * spherical_jn(l, k * Ri)
            * _spherical_hl_derivative(l, k * Ri))


@cache
def K_exterior_star_offdiagonal(l: int, m: int, lp: int, mp: int, k: complex,
                                xb: Tuple[float, float, float], Ri: float, Rj: float,
                                max_lam: int) -> complex:
    """Off-diagonal block of (1/2 I + K^{k,*}); exterior wavenumber, j_l' on receiver."""
    return (-1j * k**2 * Rj**2
            * A_coefficient_general(l, m, lp, mp, k, xb, max_lam)
            * spherical_jn(l, k * Ri, derivative=True)
            * spherical_jn(lp, k * Rj))


# ---------------------------------------------------------------------------
# Neumann-eigenvalue helpers (non-subwavelength leading-order approximation).
#
# For a ball of radius R, the Neumann eigenvalues of -Laplace are
#     mu_{l,n} = (alpha_{l,n} / R)^2,
# where alpha_{l,n} is the n-th non-negative root of j_l'(alpha) = 0.
# The eigenspace at (l,n) is Span{j_l(alpha r/R) Y_l^m : m=-l,...,l}
# with dimension 2l+1.
# ---------------------------------------------------------------------------


@cache
def _spherical_jn_prime_zeros(l: int, n_max: int) -> Tuple[float, ...]:
    """Return the first ``n_max`` positive roots of j_l'(x) = 0.

    Strategy: scan a fine grid for sign changes of ``spherical_jn(l, x,
    derivative=True)`` and refine each bracket with ``brentq``.

    Args:
        l: Angular-momentum index (l >= 0).
        n_max: Number of roots to return.

    Returns:
        Tuple of ``n_max`` floats (cached).
    """
    return _spherical_jn_generic_zeros(l, n_max, derivative=True)


@cache
def _spherical_jn_zeros(l: int, n_max: int) -> Tuple[float, ...]:
    """Return the first ``n_max`` positive roots of j_l(x) = 0.

    These are the interior Dirichlet eigenvalues of -Laplace on the unit ball
    (up to the factor 1/R) and therefore the fictitious-frequency set of the
    exterior single-layer operator S^k.

    For ``l = 0`` the closed form is ``n * pi``; for ``l >= 1`` no closed form
    exists and we scan+bracket (via ``brentq``) the same way as for the
    derivative zeros.
    """
    return _spherical_jn_generic_zeros(l, n_max, derivative=False)


def _spherical_jn_generic_zeros(l: int, n_max: int,
                                derivative: bool) -> Tuple[float, ...]:
    """Shared zero-finder for ``j_l`` (``derivative=False``) or ``j_l'``
    (``derivative=True``). Returns the first ``n_max`` positive roots."""
    if n_max <= 0:
        return tuple()

    # Upper bound is comfortably above the n_max-th zero for reasonable l,n.
    # Both zeros of j_l and j_l' grow like ~ (n + l/2) * pi.
    x_max = float((n_max + l + 4) * np.pi)
    n_grid = max(4000, 200 * (n_max + l + 2))
    xs = np.linspace(1e-6, x_max, n_grid)

    def f(x):
        return spherical_jn(l, x, derivative=derivative)

    vals = f(xs)
    sign_changes = np.where(np.sign(vals[:-1]) * np.sign(vals[1:]) < 0)[0]

    roots: List[float] = []
    for idx in sign_changes:
        a, b = xs[idx], xs[idx + 1]
        try:
            r = brentq(f, a, b, xtol=1e-12, rtol=1e-12)
        except ValueError:
            continue
        if r > 1e-6:  # positive root
            roots.append(r)
        if len(roots) == n_max:
            break

    if len(roots) < n_max:
        raise RuntimeError(
            f"Found only {len(roots)}/{n_max} positive roots of "
            f"j_{l}{'‎′' if derivative else ''}(x)=0 on [0, {x_max:g}]. "
            f"Increase the search range or grid density."
        )

    return tuple(roots)


def _compute_neumann_eigenvalues_for_balls(
    radii: Sequence[float],
    v_in: Sequence[complex],
    l_max: int,
    n_max: int,
) -> List[Dict]:
    """Enumerate Neumann eigenfrequencies of -Laplace on a collection of balls.

    Shared implementation used by both :class:`ContinuumFiniteSWP3D` and
    :class:`ContinuumPeriodicSWP3D` (in ``continuum_periodic.py``).

    For each ball ``j`` (radius ``radii[j]``, interior wave speed ``v_in[j]``),
    and each ``(l, n)`` with ``0 <= l <= l_max`` and ``0 <= n < n_max``, the
    Neumann eigenfrequency of the acoustic Helmholtz operator on the ball is

        omega_{j,l,n} = v_in[j] * alpha_{l,n} / radii[j],

    where ``alpha_{l,n}`` is the ``n``-th positive root of ``j_l'(x) = 0``.
    The trivial (l=0, n=0, alpha=0) constant mode is omitted because
    :func:`_spherical_jn_prime_zeros` returns only positive roots.

    Args:
        radii: Sequence of ball radii, length ``N``.
        v_in: Sequence of interior wave speeds, length ``N`` (may be complex).
        l_max: Maximum angular-momentum index (inclusive).
        n_max: Number of radial modes to keep per (j, l).

    Returns:
        List of dicts (unsorted across resonators), with keys
        ``j, l, n, alpha, omega, multiplicity, kappa``.
    """
    radii = np.asarray(radii)
    v_in = np.asarray(v_in)
    N = len(radii)
    records: List[Dict] = []
    for l in range(l_max + 1):
        zeros = _spherical_jn_prime_zeros(l, n_max)
        for n, alpha in enumerate(zeros):
            for j in range(N):
                Rj = float(radii[j])
                vj = (complex(v_in[j]) if np.iscomplexobj(v_in)
                      else float(v_in[j]))
                omega_jln = vj * alpha / Rj
                records.append({
                    'j': j, 'l': l, 'n': n, 'alpha': alpha,
                    'omega': omega_jln,
                    'multiplicity': 2 * l + 1,
                    'kappa': _neumann_trace_kappa(l, alpha, Rj),
                })
    return records


def _find_omega0_candidates_for_balls(
    radii: Sequence[float],
    v_in: Sequence[complex],
    l_max: int,
    n_max: int,
    tol: float = 1e-8,
) -> List[Dict]:
    """Group per-resonator Neumann eigenfrequencies into ``omega_0`` clusters.

    Shared implementation used by both :class:`ContinuumFiniteSWP3D` and
    :class:`ContinuumPeriodicSWP3D`.

    Two records share the same ``omega_0`` if their ``omega`` values agree to
    ``tol``. Within a cluster each ``(j, l, n)`` record is expanded into
    ``2 l + 1`` index-set entries (one per ``m = -l,...,l``).

    Returns:
        List of dicts with keys ``omega_0, index_set, kappa, size``.
    """
    records = _compute_neumann_eigenvalues_for_balls(radii, v_in, l_max, n_max)
    if not records:
        return []

    records.sort(key=lambda r: (np.real(r['omega']), np.imag(r['omega'])))

    clusters: List[List[Dict]] = []
    current: List[Dict] = [records[0]]
    for rec in records[1:]:
        if abs(rec['omega'] - current[-1]['omega']) <= tol:
            current.append(rec)
        else:
            clusters.append(current)
            current = [rec]
    clusters.append(current)

    out: List[Dict] = []
    for cluster in clusters:
        omega_values = np.array([c['omega'] for c in cluster])
        index_set: List[Tuple[int, int, int, int]] = []
        kappa_list: List[float] = []
        for c in cluster:
            for m in range(-c['l'], c['l'] + 1):
                index_set.append((c['j'], c['l'], c['n'], m))
                kappa_list.append(c['kappa'])
        out.append({
            'omega_0': complex(np.mean(omega_values)),
            'index_set': index_set,
            'kappa': np.array(kappa_list),
            'size': len(index_set),
        })
    return out


def _complex_to_real_sh_unitary(
    index_set: Sequence[Tuple[int, int, int, int]]
) -> np.ndarray:
    """Unitary mapping the complex-SH basis to the real-SH basis on ``index_set``.

    Within every ``(j, l, n)`` block the ``index_set`` is ordered
    ``m = -l, ..., +l``. The returned matrix ``U`` acts on that ordering so
    that ``U @ v`` expresses a complex-SH coefficient vector in the real-SH
    basis, using the standard Condon-Shortley convention

        Y_l^{0, R}  = Y_l^0,
        Y_l^{+m, R} = (1 / sqrt 2) ( Y_l^{-m} + (-1)^m Y_l^{+m} ),   m > 0,
        Y_l^{-m, R} = (i / sqrt 2) ( Y_l^{-m} - (-1)^m Y_l^{+m} ),   m > 0.

    The real-SH slots use the same ``m``-ordering as the complex-SH input
    (``m = -l, ..., +l``).

    ``U`` is unitary (``U U^H = I``), so the similarity ``U C U^H`` preserves
    spectra. It is used in :meth:`ContinuumFiniteSWP3D.get_frequency_dependent_capacitance`
    (and its periodic counterpart) to optionally return ``C`` in the real-SH
    basis where Prop. 3.13 / Prop. 4.3 of the paper manifest as ordinary
    matrix complex-symmetry / Hermiticity.

    Args:
        index_set: Sequence of ``(j, l, n, m)`` tuples, organised in
            contiguous ``2 l + 1`` blocks per ``(j, l, n)``.

    Returns:
        ``(M, M)`` complex unitary with ``M = len(index_set)``.
    """
    M = len(index_set)
    U = np.zeros((M, M), dtype=complex)
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    i = 0
    while i < M:
        _j, l, _n, _m = index_set[i]
        block = 2 * l + 1
        # Sanity: the next ``block`` entries must all share (j, l, n) and have
        # m-ordering -l, ..., +l. We don't validate strictly (caller contract)
        # but this assumption is baked into the contiguous 3-block loop below.
        for out_m in range(-l, l + 1):
            row = i + (out_m + l)
            if out_m == 0:
                U[row, i + l] = 1.0
            elif out_m > 0:
                abs_m = out_m
                col_minus = i + (-abs_m + l)
                col_plus = i + (abs_m + l)
                U[row, col_minus] = inv_sqrt2
                U[row, col_plus] = inv_sqrt2 * ((-1) ** abs_m)
            else:
                abs_m = -out_m
                col_minus = i + (-abs_m + l)
                col_plus = i + (abs_m + l)
                U[row, col_minus] = 1j * inv_sqrt2
                U[row, col_plus] = -1j * inv_sqrt2 * ((-1) ** abs_m)
        i += block
    return U


def _neumann_trace_kappa(l: int, alpha: float, R: float) -> float:
    """Boundary-trace normalization constant kappa_{l,n,j} for the L^2(D_j)-
    normalized Neumann eigenfunction u(r,theta,phi) = N j_l(alpha r/R) Y_l^m.

    The boundary trace on partial D_j is kappa * Y_l^m, with

        kappa = N * j_l(alpha),
        N^{-2} = (R^3 / 2) * (1 - l(l+1)/alpha^2) * j_l(alpha)^2    (alpha > 0).

    Equivalently (choosing the positive square root / matching sign),

        kappa = sign(j_l(alpha)) * sqrt(2 / (R^3 * (1 - l(l+1)/alpha^2))).

    This is valid only for alpha > 0 (we exclude the (l,n)=(0,0) constant mode
    at the enumeration stage).
    """
    if alpha <= 0:
        raise ValueError("kappa_{l,n,j} is only defined for alpha > 0 (not the "
                         "constant mode).")
    jl_alpha = spherical_jn(l, alpha)
    denom = 1.0 - l * (l + 1) / (alpha ** 2)
    if denom <= 0:
        # Should never happen: at a Neumann zero, alpha^2 > l(l+1) unless
        # alpha=0 (and (l,n)=(0,0) is explicitly excluded upstream).
        raise ValueError(
            f"Degenerate normalization at (l={l}, alpha={alpha}): "
            "alpha^2 - l(l+1) <= 0."
        )
    mag = np.sqrt(2.0 / (R ** 3 * denom))
    return float(np.sign(jl_alpha) * mag)


# ---------------------------------------------------------------------------
# ContinuumFiniteSWP3D
# ---------------------------------------------------------------------------


class ContinuumFiniteSWP3D(SWP3D):
    """Finite system of spherical resonators with the exact (nonzero-delta,
    non-subwavelength) resonance problem.

    Instances expose :meth:`get_A_matrix` to assemble the discretised block
    operator A(omega, delta) of eq. (2.4) and :meth:`compute_resonances` to
    find its characteristic values by Muller's method.

    The background (exterior) wave speed is hard-coded to ``v = 1``; interior
    wave speeds are taken from :attr:`v_in` (per-resonator).
    """

    def __init__(self, **pars):
        super().__init__(**pars)
        self.v = 1.0

    def __str__(self):
        return super().__str__() + "\nPhysics: Continuum (exact, nonzero-delta)"

    def _build_S_tilde_block(self, omega: complex, N_multipole: int) -> np.ndarray:
        """Assemble the S_tilde^omega block (piecewise interior wavenumber)."""
        L = N_multipole ** 2
        max_lam = N_multipole + 1
        B = np.zeros((self.N * L, self.N * L), dtype=complex)
        for i in range(self.N):
            k_i = omega / self.v_in[i]
            for j in range(self.N):
                for l in range(N_multipole):
                    for m in range(-l, l + 1):
                        for lp in range(N_multipole):
                            for mp in range(-lp, lp + 1):
                                row = flat_index(i, N_multipole, l, m)
                                col = flat_index(j, N_multipole, lp, mp)
                                if i == j:
                                    if l == lp and m == mp:
                                        B[row, col] = S_tilde_diagonal(l, k_i, self.radii[i])
                                else:
                                    xb = tuple(self.centers[j] - self.centers[i])
                                    B[row, col] = S_tilde_offdiagonal(
                                        l, m, lp, mp, k_i, xb,
                                        self.radii[i], self.radii[j], max_lam,
                                    )
        return B

    def _build_S_exterior_block(self, omega: complex, N_multipole: int) -> np.ndarray:
        """Assemble the exterior S^k block (single exterior wavenumber)."""
        L = N_multipole ** 2
        max_lam = N_multipole + 1
        k = omega / self.v
        B = np.zeros((self.N * L, self.N * L), dtype=complex)
        for i in range(self.N):
            for j in range(self.N):
                for l in range(N_multipole):
                    for m in range(-l, l + 1):
                        for lp in range(N_multipole):
                            for mp in range(-lp, lp + 1):
                                row = flat_index(i, N_multipole, l, m)
                                col = flat_index(j, N_multipole, lp, mp)
                                if i == j:
                                    if l == lp and m == mp:
                                        B[row, col] = S_exterior_diagonal(l, k, self.radii[i])
                                else:
                                    xb = tuple(self.centers[j] - self.centers[i])
                                    B[row, col] = S_exterior_offdiagonal(
                                        l, m, lp, mp, k, xb,
                                        self.radii[i], self.radii[j], max_lam,
                                    )
        return B

    def _build_K_tilde_star_block(self, omega: complex, N_multipole: int) -> np.ndarray:
        """Assemble the (-1/2 I + K_tilde^{omega,*}) block."""
        L = N_multipole ** 2
        max_lam = N_multipole + 1
        B = np.zeros((self.N * L, self.N * L), dtype=complex)
        for i in range(self.N):
            k_i = omega / self.v_in[i]
            for j in range(self.N):
                for l in range(N_multipole):
                    for m in range(-l, l + 1):
                        for lp in range(N_multipole):
                            for mp in range(-lp, lp + 1):
                                row = flat_index(i, N_multipole, l, m)
                                col = flat_index(j, N_multipole, lp, mp)
                                if i == j:
                                    if l == lp and m == mp:
                                        B[row, col] = K_tilde_star_diagonal(l, k_i, self.radii[i])
                                else:
                                    xb = tuple(self.centers[j] - self.centers[i])
                                    B[row, col] = K_tilde_star_offdiagonal(
                                        l, m, lp, mp, k_i, xb,
                                        self.radii[i], self.radii[j], max_lam,
                                    )
        return B

    def _build_K_exterior_star_block(self, omega: complex, delta_arr: np.ndarray,
                                     N_multipole: int) -> np.ndarray:
        """Assemble the -delta_{partial D} (1/2 I + K^{k,*}) block.

        The minus sign and per-row delta factor are applied here so the caller
        can concatenate blocks directly.
        """
        L = N_multipole ** 2
        max_lam = N_multipole + 1
        k = omega / self.v
        B = np.zeros((self.N * L, self.N * L), dtype=complex)
        for i in range(self.N):
            delta_i = delta_arr[i]
            for j in range(self.N):
                for l in range(N_multipole):
                    for m in range(-l, l + 1):
                        for lp in range(N_multipole):
                            for mp in range(-lp, lp + 1):
                                row = flat_index(i, N_multipole, l, m)
                                col = flat_index(j, N_multipole, lp, mp)
                                if i == j:
                                    if l == lp and m == mp:
                                        B[row, col] = (
                                            -delta_i
                                            * K_exterior_star_diagonal(l, k, self.radii[i])
                                        )
                                else:
                                    xb = tuple(self.centers[j] - self.centers[i])
                                    B[row, col] = (
                                        -delta_i
                                        * K_exterior_star_offdiagonal(
                                            l, m, lp, mp, k, xb,
                                            self.radii[i], self.radii[j], max_lam,
                                        )
                                    )
        return B

    def get_A_matrix(self, omega: complex, delta, N_multipole: int = 2) -> np.ndarray:
        """Assemble the (2 N L, 2 N L) discretisation of A(omega, delta).

        Args:
            omega: Complex angular frequency at which to evaluate A.
            delta: Density contrast; scalar (broadcast) or length-N array.
            N_multipole: Number of l values kept per resonator (basis size L = N_multipole**2).

        Returns:
            (2 N L, 2 N L) complex ndarray. Rows/cols 0:NL correspond to psi,
            NL:2NL to phi.
        """
        delta_arr = _get_consistent_parameter(delta, self.N).astype(complex)

        S_tilde = self._build_S_tilde_block(omega, N_multipole)
        S_ext = self._build_S_exterior_block(omega, N_multipole)
        K_tilde = self._build_K_tilde_star_block(omega, N_multipole)
        K_ext = self._build_K_exterior_star_block(omega, delta_arr, N_multipole)

        top = np.concatenate([S_tilde, -S_ext], axis=1)
        bot = np.concatenate([K_tilde, K_ext], axis=1)
        return np.concatenate([top, bot], axis=0)

    def characteristic_determinant(self, omega: complex, delta,
                                   N_multipole: int = 2,
                                   filter_spurious: bool = False,
                                   spurious_l_max: int = None,
                                   spurious_n_max: int = 3) -> complex:
        """det A(omega, delta). The resonant frequencies are its roots.

        Args:
            omega: Complex angular frequency.
            delta: Density contrast; scalar or length-N array.
            N_multipole: Spherical-harmonic truncation.
            filter_spurious: If True, divide out the known spurious zeros
                of det A at ``omega = v * beta_{l,n} / R_i`` (interior
                Dirichlet eigenfrequencies of the exterior single-layer
                operator). Each zero is deflated with its full multiplicity
                (2 l + 1 per (i, l, n) tuple, summed over coincident tuples);
                see :meth:`spurious_omega_candidates`. Default False.
            spurious_l_max: Largest l used when listing the spurious set.
                Defaults to ``N_multipole - 1``.
            spurious_n_max: Number of radial zeros per l to include. Default 3.

        Returns:
            det A(omega, delta) (optionally pre-deflated by the spurious
            polynomial).
        """
        val = np.linalg.det(self.get_A_matrix(omega, delta, N_multipole))
        if not filter_spurious:
            return val

        l_max = (spurious_l_max if spurious_l_max is not None
                 else max(0, int(N_multipole) - 1))
        spurious, multiplicities = self.spurious_omega_candidates(
            l_max, spurious_n_max, with_multiplicities=True)
        # Analytic pre-deflation: each spurious s is a zero of det A with
        # multiplicity m, so divide by (omega - s) ** m.
        for s, m in zip(spurious, multiplicities):
            val = val / (omega - s) ** int(m)
        return val

    def spurious_omega_candidates(self, l_max: int,
                                  n_max: int,
                                  with_multiplicities: bool = False):
        """Fictitious (spurious) resonances of the single-layer formulation.

        The exterior single-layer operator S^k entering the (1,2) block of
        A(omega, delta) is singular at the interior Dirichlet eigenfrequencies
        of each individual ball D_i: the ``(l,m)`` diagonal self-entry
        ``~ j_l(k R_i) * h_l(k R_i)`` vanishes whenever ``j_l(k R_i) = 0``.
        The same j_l factor appears in the (2,2) block, so for N = 1 the
        corresponding phi column of A is identically zero and det A vanishes
        independently of delta; for N >= 2 the same frequencies remain near-
        singular points where Muller's method tends to report fictitious roots.

        The spurious set is

            omega = v * beta_{l,n} / R_i,     0 <= l <= l_max, 1 <= n <= n_max,
                                              1 <= i <= N,

        where ``beta_{l,n}`` is the n-th positive zero of ``j_l`` and ``v`` is
        the background (exterior) wave speed.

        For each tuple (i, l, n) the (l, m) columns m = -l,...,l of A all
        vanish simultaneously, so the zero of det A at that omega has
        multiplicity 2 l + 1. When several tuples coincide (e.g. resonators
        sharing a radius, or tuned geometries) the per-tuple multiplicities
        add.

        Args:
            l_max: Highest angular-momentum index to enumerate (inclusive).
            n_max: Number of radial zeros per l to enumerate.
            with_multiplicities: If True, return ``(omegas, multiplicities)``
                as parallel arrays. If False (default, back-compat), return
                just ``omegas``.

        Returns:
            Sorted 1D float ndarray of distinct candidate frequencies, or a
            tuple ``(omegas, multiplicities)`` of parallel 1D arrays when
            ``with_multiplicities=True``.
        """
        per_tuple = []   # list of (omega, multiplicity)
        for l in range(l_max + 1):
            # Positive zeros of the spherical Bessel j_l, found by bracketed
            # brentq (same strategy as _spherical_jn_prime_zeros).
            betas = _spherical_jn_zeros(l, n_max)
            mult_l = 2 * l + 1
            for beta in betas:
                for i in range(self.N):
                    R_i = float(self.radii[i])
                    omega = float(self.v) * float(beta) / R_i
                    per_tuple.append((omega, mult_l))

        if not per_tuple:
            empty = np.array([], dtype=float)
            if with_multiplicities:
                return empty, np.array([], dtype=int)
            return empty

        # Deduplicate close values (different (l,n,i) can coincide for tuned
        # geometries). Sum multiplicities at coincident frequencies.
        mult_by_key: Dict[float, int] = {}
        for omega, m in per_tuple:
            key = round(omega, 12)
            mult_by_key[key] = mult_by_key.get(key, 0) + m

        omegas = np.array(sorted(mult_by_key.keys()), dtype=float)
        mults = np.array([mult_by_key[o] for o in omegas], dtype=int)

        if with_multiplicities:
            return omegas, mults
        return omegas

    def compute_resonances(self, x0: complex, N_roots: int, delta,
                           N_multipole: int = 2,
                           perturbation: float = 1e-3,
                           filter_spurious: bool = False,
                           spurious_l_max: int = None,
                           spurious_n_max: int = 3,
                           spurious_tol: float = 1e-6) -> np.ndarray:
        """Find N_roots roots of omega -> det A(omega, delta) near x0.

        Args:
            x0: Initial guess for the first root.
            N_roots: Number of roots to find (subsequent roots use deflation).
            delta: Density contrast; scalar or length-N array.
            N_multipole: Spherical-harmonic truncation.
            perturbation: Muller's method bracket half-width. Also controls
                the offset applied when seeking the next root after deflation.
            filter_spurious: If True, pre-deflate the characteristic
                determinant by the fictitious interior-Dirichlet eigenvalues
                of the exterior single-layer operator (see
                :meth:`spurious_omega_candidates`). This suppresses spurious
                zeros of det A at ``omega = v * beta_{l,n} / R_i``. Default
                False to preserve legacy behaviour.
            spurious_l_max: Largest l used when listing the spurious set.
                Defaults to ``N_multipole - 1`` (the same truncation as the
                underlying basis).
            spurious_n_max: Number of radial zeros per l to include. Default 3.
            spurious_tol: After Muller returns, any root closer than this to a
                spurious candidate is reported via a ``warnings.warn`` call
                (the root is still returned; filtering is assumed to have
                happened via the pre-deflation above).

        Returns:
            Length-N_roots complex ndarray of roots.
        """
        def f(w):
            return self.characteristic_determinant(
                w, delta, N_multipole,
                filter_spurious=filter_spurious,
                spurious_l_max=spurious_l_max,
                spurious_n_max=spurious_n_max,
            )

        roots = find_roots_muller(f, x0, N_roots, perturbation=perturbation)

        if filter_spurious and len(roots):
            spurious = self.spurious_omega_candidates(
                spurious_l_max if spurious_l_max is not None
                else max(0, int(N_multipole) - 1),
                spurious_n_max)
            if len(spurious):
                import warnings
                for r in roots:
                    min_dist = float(np.min(np.abs(np.asarray(spurious) - r)))
                    if min_dist < spurious_tol:
                        warnings.warn(
                            f"compute_resonances: root {r} is within "
                            f"{min_dist:.2e} of a spurious candidate despite "
                            f"filter_spurious=True.",
                            RuntimeWarning,
                        )
        return roots

    # -----------------------------------------------------------------
    # Non-subwavelength leading-order approximation (Theorem 3.7 / eq. (3.8)).
    #
    # For a Neumann eigenfrequency omega_0 of -Laplace on at least one
    # interior domain D_j, the resonances of the full operator satisfy
    #     omega_n = omega_0 + lambda_n(omega_0) + O(delta^2),
    # where lambda_n(omega_0) are eigenvalues of the frequency-dependent
    # capacitance matrix C(omega_0) defined by eq. (3.8).
    # -----------------------------------------------------------------

    def compute_neumann_eigenvalues(self, l_max: int,
                                    n_max: int) -> List[Dict]:
        """Enumerate (per-resonator) Neumann eigenfrequencies of -Laplace.

        For each resonator j and each (l, n) with 0 <= l <= l_max,
        0 <= n < n_max, excluding the trivial (l=0, n=0, alpha=0) mode,
        returns a record containing:
            j, l, n, alpha, omega = v_in[j] * alpha / radii[j],
            multiplicity = 2 l + 1, kappa = kappa_{l,n,j}.

        Args:
            l_max: Maximum angular-momentum index (inclusive).
            n_max: Number of radial modes to keep per (j, l).

        Returns:
            List of dicts (unsorted across resonators; grouped by l, n within
            each resonator).
        """
        # ``_spherical_jn_prime_zeros`` returns only the POSITIVE roots of
        # j_l'(x)=0. The constant mode (l=0, alpha=0) corresponding to the
        # zero Neumann eigenvalue is therefore automatically excluded, and
        # ``n=0`` here labels the first *positive* root alpha_{l,0} for each l.
        return _compute_neumann_eigenvalues_for_balls(
            self.radii, self.v_in, l_max, n_max
        )

    def find_omega0_candidates(self, l_max: int, n_max: int,
                               tol: float = 1e-8) -> List[Dict]:
        """Group per-resonator Neumann eigenfrequencies into omega_0 clusters.

        Two records are considered to share the same omega_0 if their omega
        values agree to ``tol`` in absolute value. Within each cluster we
        expand each (j, l, n) record into (2 l + 1) index-set entries, one
        for each m = -l, ..., l.

        Args:
            l_max: Maximum angular-momentum index (inclusive).
            n_max: Number of radial modes to keep per (j, l).
            tol: Absolute tolerance for grouping frequencies.

        Returns:
            List of dicts with keys:
                'omega_0':   cluster mean (complex),
                'index_set': list of tuples (j, l, n, m) of size ``size``,
                'kappa':     np.ndarray of kappa values aligned with index_set,
                'size':      total multiplicity m = |index_set|.
        """
        return _find_omega0_candidates_for_balls(
            self.radii, self.v_in, l_max, n_max, tol
        )

    def _build_K_exterior_star_block_raw(self, omega: complex,
                                         N_multipole: int) -> np.ndarray:
        """Raw (1/2 I + K^{k,*}) block without the -delta_i prefactor.

        Implemented by reusing ``_build_K_exterior_star_block`` with
        ``delta_arr = -np.ones(N)`` (the existing function folds in -delta_i,
        and passing -1 cancels the sign to recover the raw operator).
        """
        delta_neg_ones = -np.ones(self.N, dtype=complex)
        return self._build_K_exterior_star_block(omega, delta_neg_ones, N_multipole)

    def get_frequency_dependent_capacitance(
        self,
        omega_0: complex,
        index_set: Sequence[Tuple[int, int, int, int]],
        kappa: np.ndarray,
        N_multipole: int,
        delta,
        *,
        real_sh_basis: bool = False,
    ) -> np.ndarray:
        """Assemble the frequency-dependent capacitance matrix C(omega_0).

        Discretises eq. (3.8):
            C[(i,m_i),(j,m_j)](omega_0) = -(delta_i v_i^2 / (2 omega_0))
                    * int_{partial D_i} g_{i,m_i} * d/dn V_{j,m_j}|_+ dsigma,

        where g_{j,ell} are L^2(D_j)-normalized Neumann eigenmode traces
        (extended by zero) and V_{j,ell} is the outgoing exterior Helmholtz
        solution at wavenumber k_0 = omega_0 / v with V|_{partial D} = g_{j,ell}.

        Basis convention
        ----------------
        By default the returned matrix is expressed in the *complex*
        spherical-harmonic basis ``{Y_l^m}``, orthonormal under the Hermitian
        L^2 pairing. Proposition 3.13 of the paper ("``C(omega_0)^T =
        C(omega_0)``") is proved using the *non-Hermitian* pairing applied to
        the *real-valued* Neumann traces, and therefore does NOT manifest as
        ``C == C.T`` in the default basis. Instead it takes the form
        ``C^T = J C J`` with ``J[(l, m), (l', m')] = (-1)^m * delta_{l l'} *
        delta_{m, -m'}``.

        Set ``real_sh_basis=True`` to return the matrix in the real spherical
        harmonic basis (``U C U^H`` with the unitary ``U`` from
        :func:`_complex_to_real_sh_unitary`), where Prop. 3.13 reduces to the
        ordinary complex-symmetry ``C == C.T``. Eigenvalues — and hence
        :meth:`compute_nonsubwavelength_resonances` — are unaffected by this
        choice.

        Args:
            omega_0: Reference (Neumann-eigen) frequency (complex-capable).
            index_set: Sequence of (j, l, n, m) tuples describing the Neumann
                index set I-tilde (returned by ``find_omega0_candidates``).
            kappa: Array of kappa values aligned with ``index_set``.
            N_multipole: Spherical-harmonic truncation (must exceed max l in
                the index set).
            delta: Density contrast (scalar or length-N array).
            real_sh_basis: If True, return ``U C U^H`` where ``U`` maps the
                complex-SH basis to the real-SH basis. The result is then
                complex-symmetric (``C == C.T``) as claimed by Prop. 3.13.
                Default False (preserves the library's native complex-SH
                basis).

        Returns:
            Complex ndarray of shape (m, m) with m = len(index_set).
        """
        if len(index_set) == 0:
            return np.zeros((0, 0), dtype=complex)

        l_max_idx = max(l for (_, l, _, _) in index_set)
        if N_multipole <= l_max_idx:
            raise ValueError(
                f"N_multipole (={N_multipole}) must exceed max l in the index "
                f"set (={l_max_idx})."
            )

        delta_arr = _get_consistent_parameter(delta, self.N).astype(complex)

        # Discretise S^{k_0} and raw (1/2 I + K^{k_0,*}).
        S_ext = self._build_S_exterior_block(omega_0, N_multipole)
        K_ext_raw = self._build_K_exterior_star_block_raw(omega_0, N_multipole)

        NL = self.N * N_multipole ** 2
        m = len(index_set)

        # RHS matrix: one column per element of index_set, unit basis vector
        # at flat_index(j, N_multipole, l, m).
        G = np.zeros((NL, m), dtype=complex)
        for c, (jc, lc, _nc, mc) in enumerate(index_set):
            G[flat_index(jc, N_multipole, lc, mc), c] = 1.0

        # Lambda_ext [g] = (1/2 I + K^{k_0,*}) S^{k_0,-1} [g]:
        try:
            tilde_G = np.linalg.solve(S_ext, G)
        except np.linalg.LinAlgError:
            # Spurious interior resonance of the exterior Dirichlet problem;
            # fall back to least-squares.
            tilde_G, *_ = np.linalg.lstsq(S_ext, G, rcond=None)
        D = K_ext_raw @ tilde_G  # shape (NL, m)

        # Integral reduction (paper's eq. (3.8) uses a pairing under which
        # the spherical harmonics are orthonormal --- equivalent to real
        # spherical harmonics or to the Hermitian L^2 inner product on
        # complex SHs; for real-valued Neumann modes the two coincide):
        #     <conj(Y_l^m), Y_{l'}^{m'}>_{partial D_i}
        #       = R_i^2 * delta_{l l'} delta_{m m'}.
        # => row r (index (i, l_r, _, m_r)) picks off D[tau_r, :] with
        #    tau_r = flat_index(i, N_multipole, l_r, m_r), times R_i^2. No
        #    (-1)^m factor arises because we pair against conj(g_{i,m}).
        row_prefactor = np.zeros(m, dtype=complex)
        tau = np.zeros(m, dtype=int)
        for r, (ir, lr, _nr, mr) in enumerate(index_set):
            Ri = float(self.radii[ir])
            vi = complex(self.v_in[ir]) if np.iscomplexobj(self.v_in) else float(self.v_in[ir])
            row_prefactor[r] = (
                -delta_arr[ir] * (vi ** 2) / (2.0 * omega_0)
                * kappa[r] * (Ri ** 2)
            )
            tau[r] = flat_index(ir, N_multipole, lr, mr)

        # Column prefactor: kappa[c].
        # C = (row_prefactor[:, None] * D[tau, :]) * kappa[None, :]
        C = row_prefactor[:, None] * D[tau, :] * kappa[None, :]

        if real_sh_basis:
            U = _complex_to_real_sh_unitary(index_set)
            C = U @ C @ U.conj().T
        return C

    def compute_nonsubwavelength_resonances(
        self,
        omega_0: complex,
        index_set: Sequence[Tuple[int, int, int, int]],
        kappa: np.ndarray,
        N_multipole: int,
        delta,
    ) -> np.ndarray:
        """Leading-order non-subwavelength resonances omega_n = omega_0 + lambda_n(omega_0).

        Args:
            omega_0: Reference Neumann eigenfrequency.
            index_set: Sequence of (j, l, n, m) tuples (from ``find_omega0_candidates``).
            kappa: Array of kappa values aligned with ``index_set``.
            N_multipole: Spherical-harmonic truncation.
            delta: Density contrast (scalar or length-N array).

        Returns:
            Complex ndarray of shape (m,) with m = len(index_set).
        """
        C = self.get_frequency_dependent_capacitance(
            omega_0, index_set, kappa, N_multipole, delta
        )
        lambdas = np.linalg.eigvals(C)
        return omega_0 + lambdas
