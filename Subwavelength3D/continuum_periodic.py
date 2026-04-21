"""Exact (non-subwavelength, nonzero-delta) resonance solver for
crystalline 3D systems of spherical resonators.

Periodic counterpart of :mod:`Subwavelength3D.continuum`
(class :class:`ContinuumFiniteSWP3D`). Implements Section 4 of the paper

    "Frequency-dependent capacitance matrix formulation for Fabry-Perot
    resonances. Part II: Three-dimensional systems"

on a full 3D Bravais lattice

    Lambda = { m_1 l_1 + m_2 l_2 + m_3 l_3 : m in Z^3 }

with quasi-momentum ``alpha`` in the first Brillouin zone ``Y*``.

Poisson-sum (reciprocal-lattice) Green's function
-------------------------------------------------
For ``xi_q := q + alpha`` with ``q`` in the reciprocal lattice
``Lambda* = { n_1 b_1 + n_2 b_2 + n_3 b_3 }`` (where
``b_i . l_j = 2 pi delta_{ij}``), the quasiperiodic standing-wave
Green's function is

    G^{alpha, omega}(x) = (1/|Y|) sum_{q in Lambda*}
        exp(i xi_q . x) / (omega^2 - |xi_q|^2),

with principal-value convention at ``|xi_q| = omega`` (the diffraction
thresholds). Below all diffraction thresholds the sum is absolutely
convergent, real-valued (for real ``omega``, real ``alpha``), and the
resulting single-layer operator ``S^{alpha, omega}_D`` is **Hermitian** on
``L^2(partial D)`` — matching Prop. 4.3 of the paper. The lack of an
outgoing-radiation imaginary part reflects the physical fact that a 3D
crystal fills all of space: there is no free-space channel into which
energy can radiate below the first diffraction threshold.

Matrix element in the spherical-harmonic basis
-----------------------------------------------
Using the Rayleigh expansion ``exp(i xi . R y) =
4 pi sum_{l,m} i^l j_l(|xi| R) Y_l^m^*(hat xi) Y_l^m(y)`` at each
resonator centre, the matrix element of the single-layer operator acting
on ``Y_l^m`` on ball ``j`` (radius ``R``) and tested against ``Y_L^M`` on
ball ``i`` (radius ``R``) becomes

    S^alpha_{(i,L,M),(j,l,m)}(omega)
      = ((4 pi)^2 R^4 / |Y|) sum_q
          exp(i xi_q . (x_i - x_j))
              * i^{L - l} * j_L(|xi_q| R) * j_l(|xi_q| R)
              * Y_L^M(hat xi_q) * conj(Y_l^m(hat xi_q))
              / (omega^2 - |xi_q|^2).

For the normal-derivative blocks ``K^{alpha, omega, *}_D`` the receiver
bessel ``j_L(|xi| R)`` is replaced by ``|xi| j_L'(|xi| R)``, and a
jump-relation constant ``+1/2 I`` (exterior limit) or ``-1/2 I``
(interior limit) is added on the ``(i, L, M) = (j, l, m)`` diagonal.

This single closed form covers all four blocks of the block operator
``A^alpha(omega, delta)``:

    A = [[ S_tilde^{alpha, omega},          -S^{alpha, k}                 ],
         [ -1/2 I + K_tilde^{alpha, omega, *}, -delta (1/2 I + K^{alpha, k, *}) ]]

with ``k_b = omega / v_b`` (interior, identical for all resonators),
``k = omega / v = omega`` (exterior), and ``delta`` the density contrast.

Section 4 of the paper assumes **identical resonators** within the unit
cell (single radius ``R``, single interior wave speed ``v_b``); the
constructor enforces this.
"""

from __future__ import annotations

import itertools
import warnings
from typing import Dict, List, Sequence, Tuple

import numpy as np

from scipy.special import spherical_jn, sph_harm

from Subwavelength3D.swp import SWP3D, _get_consistent_parameter
from Subwavelength3D.classic_finite import flat_index
from Subwavelength3D.continuum import (
    _complex_to_real_sh_unitary,
    _compute_neumann_eigenvalues_for_balls,
    _find_omega0_candidates_for_balls,
)
from Utils.utils_general import find_roots_muller


def _enumerate_lattice_shells(m_cutoff: int) -> np.ndarray:
    """Integer triples ``(n_1, n_2, n_3)`` with ``max_i |n_i| <= m_cutoff``.

    Used both for direct-lattice and reciprocal-lattice enumeration.

    Returns:
        Array of shape ``(M, 3)`` of integer triples, sorted by Euclidean
        norm so that ``(0, 0, 0)`` is the first entry.
    """
    m_cap = int(m_cutoff)
    ns = np.arange(-m_cap, m_cap + 1)
    triples = np.array(list(itertools.product(ns, ns, ns)), dtype=int)
    norms = np.linalg.norm(triples.astype(float), axis=1)
    return triples[np.argsort(norms, kind='stable')]


def _sph_harm_physics(l: int, m: int, direction: np.ndarray) -> np.ndarray:
    """``Y_l^m`` in physics convention, evaluated at unit 3-vectors.

    Args:
        l, m: Spherical-harmonic indices with ``|m| <= l``.
        direction: ``(M, 3)`` array of unit vectors.

    Returns:
        Complex array of shape ``(M,)`` with ``Y_l^m(polar, azimuth)``.
        ``scipy.special.sph_harm(m, l, azimuth, polar)`` is used.
    """
    # cos(polar) = n_z (clipped to [-1, 1] for numerical safety)
    cos_polar = np.clip(direction[:, 2], -1.0, 1.0)
    polar = np.arccos(cos_polar)
    azimuth = np.arctan2(direction[:, 1], direction[:, 0])
    return sph_harm(m, l, azimuth, polar)


class ContinuumPeriodicSWP3D(SWP3D):
    """Crystalline (fully 3D periodic) counterpart of
    :class:`ContinuumFiniteSWP3D`.

    The quasiperiodic layer-potential matrix elements are computed via the
    reciprocal-lattice Poisson-sum representation of the standing-wave
    Green's function (principal-value convention), which is Hermitian
    below the first diffraction threshold.

    Args:
        centers: ``(N, 3)`` resonator centres inside the fundamental cell.
        radii: Scalar or ``(N,)`` array. Must be uniform (single ``R``).
        v_in: Scalar or ``(N,)`` array. Must be uniform (single ``v_b``).
        lattice_vectors: ``(3, 3)`` array whose rows are ``l_1, l_2, l_3``.
        lattice_shell_cutoff: Dimensionless reciprocal-lattice cutoff (float
            or int). The sum retains every reciprocal vector ``q`` with
            ``|q| <= lattice_shell_cutoff * max_i |b_i|``, where ``b_i`` are
            the reciprocal-lattice basis vectors. For cubic cells all
            ``|b_i|`` agree and this reduces to a single-sphere cap
            comparable to the legacy integer box cap. For anisotropic cells
            the bound is adapted per axis so that the physical reciprocal
            resolution is consistent in every direction (crucial for
            elongated cells such as a chain along one axis embedded in a
            transversely large cell). Default ``4``.

    Attributes:
        R (float): Common ball radius.
        v_b (complex): Common interior wave speed.
        v (float): Exterior wave speed (hard-coded to ``1``).
        lattice_vectors (np.ndarray): ``(3, 3)`` with rows ``l_i``.
        reciprocal_vectors (np.ndarray): ``(3, 3)`` with rows ``b_i``
            (``b_i . l_j = 2 pi delta_{ij}``).
        cell_volume (float): ``|det(lattice_vectors)|``.
        lattice_shell_cutoff (float): Dimensionless reciprocal-shell radius.
        q_cutoff (float): Physical ``|q|`` cap actually used in the sum
            (``lattice_shell_cutoff * max_i |b_i|``).
    """

    def __init__(
        self,
        *,
        centers,
        radii=1.0,
        v_in=1.0,
        lattice_vectors,
        lattice_shell_cutoff: float = 4.0,
        **pars,
    ):
        super().__init__(centers=centers, radii=radii, v_in=v_in, **pars)

        radii_arr = np.asarray(self.radii, dtype=float)
        if not np.allclose(radii_arr, radii_arr[0], rtol=0.0, atol=1e-12):
            raise ValueError(
                "ContinuumPeriodicSWP3D requires identical resonators "
                "(radii must all be equal). Got "
                f"radii = {radii_arr}."
            )
        v_in_arr = np.asarray(self.v_in)
        if not np.allclose(v_in_arr, v_in_arr[0], rtol=0.0, atol=1e-12):
            raise ValueError(
                "ContinuumPeriodicSWP3D requires identical resonators "
                "(v_in must all be equal). Got "
                f"v_in = {v_in_arr}."
            )
        self.R: float = float(radii_arr[0])
        self.v_b: complex = (
            complex(v_in_arr[0]) if np.iscomplexobj(v_in_arr)
            else float(v_in_arr[0])
        )
        self.v = 1.0

        self.lattice_vectors = np.asarray(lattice_vectors, dtype=float)
        if self.lattice_vectors.shape != (3, 3):
            raise ValueError(
                "lattice_vectors must be (3, 3); got "
                f"{self.lattice_vectors.shape}."
            )
        self.cell_volume = float(abs(np.linalg.det(self.lattice_vectors)))
        if self.cell_volume < 1e-14:
            raise ValueError("lattice_vectors are degenerate (det ~ 0).")

        # Reciprocal basis: rows b_i satisfying l_j . b_i = 2 pi delta_{ij}.
        # With rows of L being l_i, we need B = 2 pi (L^{-1})^T.
        self.reciprocal_vectors = (
            2.0 * np.pi * np.linalg.inv(self.lattice_vectors).T
        )

        self.lattice_shell_cutoff = float(lattice_shell_cutoff)
        if self.lattice_shell_cutoff < 0:
            raise ValueError(
                f"lattice_shell_cutoff must be nonnegative; got "
                f"{self.lattice_shell_cutoff}."
            )
        # Physical reciprocal-space radius used for filtering.
        b_norms = np.linalg.norm(self.reciprocal_vectors, axis=1)
        self._b_norms = b_norms
        self.q_cutoff = float(self.lattice_shell_cutoff * b_norms.max())

        # Cached q-grid data (q-vectors themselves — independent of alpha / k).
        self._reciprocal_triples: np.ndarray | None = None
        self._q_vectors: np.ndarray | None = None

    def __str__(self):
        return (
            super().__str__()
            + "\nPhysics: Continuum periodic (exact, nonzero-delta)"
            + f"\nLattice (rows = l_i):\n{self.lattice_vectors}"
            + f"\n|Y| = {self.cell_volume:.6g}, "
            + f"reciprocal shell cutoff = {self.lattice_shell_cutoff}"
        )

    # ------------------------------------------------------------------
    # Reciprocal-lattice enumeration
    # ------------------------------------------------------------------

    def _ensure_reciprocal(self) -> Tuple[np.ndarray, np.ndarray]:
        """Cache integer triples and their reciprocal-lattice vectors.

        Enumerates all integer triples ``(n_1, n_2, n_3)`` whose associated
        reciprocal vector ``q = sum_i n_i b_i`` satisfies
        ``|q| <= q_cutoff``. For isotropic cells this is essentially the
        inscribed sphere of the legacy integer-box cap; for anisotropic
        cells the per-axis integer bounds adjust so that reciprocal-space
        coverage is consistent in every direction.
        """
        if self._reciprocal_triples is not None:
            return self._reciprocal_triples, self._q_vectors

        Q = self.q_cutoff
        if Q <= 0.0:
            # Only the q = 0 vector.
            triples = np.zeros((1, 3), dtype=int)
            q_vecs = np.zeros((1, 3), dtype=float)
            self._reciprocal_triples = triples
            self._q_vectors = q_vecs
            return triples, q_vecs

        # Per-axis integer bounds so that every |q| <= Q is enumerated.
        per_axis = np.ceil(Q / self._b_norms).astype(int)
        # Enumerate the anisotropic bounding box, then filter to |q| <= Q.
        n1 = np.arange(-per_axis[0], per_axis[0] + 1)
        n2 = np.arange(-per_axis[1], per_axis[1] + 1)
        n3 = np.arange(-per_axis[2], per_axis[2] + 1)
        N1, N2, N3 = np.meshgrid(n1, n2, n3, indexing='ij')
        triples_all = np.stack([N1.ravel(), N2.ravel(), N3.ravel()], axis=1)
        q_all = triples_all.astype(float) @ self.reciprocal_vectors
        q_norm = np.linalg.norm(q_all, axis=1)
        keep = q_norm <= Q + 1e-12
        triples = triples_all[keep]
        q_vecs = q_all[keep]
        # Sort by |q| so q = 0 is first (matches _enumerate_lattice_shells).
        order = np.argsort(np.linalg.norm(q_vecs, axis=1), kind='stable')
        self._reciprocal_triples = triples[order]
        self._q_vectors = q_vecs[order]
        return self._reciprocal_triples, self._q_vectors

    # ------------------------------------------------------------------
    # Core Poisson-sum block assembly
    # ------------------------------------------------------------------

    def _assemble_poisson_block(
        self,
        alpha: np.ndarray,
        k: complex,
        N_multipole: int,
        *,
        receiver_derivative: bool,
        jump_sign: float,
    ) -> np.ndarray:
        """Assemble a single layer-potential block via the Poisson sum.

        Args:
            alpha: Quasi-momentum ``(3,)``.
            k: Wavenumber (``k_b = omega / v_b`` for interior blocks,
                ``k = omega / v`` for exterior blocks).
            N_multipole: Truncation; basis dim per resonator is
                ``L = N_multipole**2``.
            receiver_derivative: If True, replace ``j_L(|xi| R)`` by
                ``|xi| j_L'(|xi| R)`` on the receiver side (for ``K^*``
                blocks).
            jump_sign: Self-diagonal jump added to the integral operator
                (``0.0`` for single-layer; ``-0.5`` for interior
                ``K_tilde^*``; ``+0.5`` for exterior ``K^*``).

        Returns:
            Complex ndarray of shape ``(N L, N L)``.
        """
        N = self.N
        L_basis = N_multipole ** 2
        NL = N * L_basis
        R = self.R
        alpha = np.asarray(alpha, dtype=float).reshape(3,)

        _, q_vecs = self._ensure_reciprocal()
        xi = q_vecs + alpha[None, :]                          # (M, 3)
        xi_mag = np.linalg.norm(xi, axis=1)                   # (M,)

        # Split into singular (xi = 0) and non-singular contributions.
        singular_mask = xi_mag < 1e-12
        ns_mask = ~singular_mask

        # Diffraction-threshold proximity warning (non-singular subset only).
        if ns_mask.any():
            denom_ns_abs = np.abs(k ** 2 - xi_mag[ns_mask] ** 2)
            rel_min = (
                np.min(denom_ns_abs) / max(abs(k) ** 2, 1e-300)
            )
            if rel_min < 1e-10:
                warnings.warn(
                    "ContinuumPeriodicSWP3D: near-diffraction threshold "
                    f"(min |k^2 - |xi_q|^2| / |k|^2 = {rel_min:.3e}); "
                    "Poisson-sum accuracy will degrade.",
                    RuntimeWarning,
                )

        # Non-singular xi_q arithmetic ----------------------------------
        xi_ns = xi[ns_mask]
        xi_mag_ns = xi_mag[ns_mask]
        M_ns = xi_mag_ns.shape[0]

        # Unit directions for spherical harmonics.
        xi_unit = xi_ns / xi_mag_ns[:, None]                  # (M_ns, 3)

        # Y_l^m(hat xi_q) for every basis index, vectorised over q.
        # Layout: Y[idx_lm, q] with idx_lm = flat (l, m) index.
        LM = [(l, m) for l in range(N_multipole) for m in range(-l, l + 1)]
        Y_grid = np.zeros((L_basis, M_ns), dtype=complex)
        for idx_lm, (l, m) in enumerate(LM):
            Y_grid[idx_lm, :] = _sph_harm_physics(l, m, xi_unit)

        # Spherical Bessel tables: source uses j_l(|xi| R); receiver either
        # j_L(|xi| R) or |xi| j_L'(|xi| R).
        kR_ns = xi_mag_ns * R                                 # (M_ns,)
        source_bessel = np.zeros((N_multipole, M_ns), dtype=float)
        for l in range(N_multipole):
            source_bessel[l, :] = spherical_jn(l, kR_ns)
        if receiver_derivative:
            receiver_bessel = np.zeros((N_multipole, M_ns), dtype=float)
            for l in range(N_multipole):
                receiver_bessel[l, :] = (
                    xi_mag_ns * spherical_jn(l, kR_ns, derivative=True)
                )
        else:
            receiver_bessel = source_bessel

        # i^L table.
        i_powers = 1j ** np.arange(N_multipole)               # (N_multipole,)

        # Resonator phases e^{i xi_q . x_i}: shape (N, M_ns).
        phase_i = np.exp(1j * self.centers @ xi_ns.T)

        # Denominator 1/(k^2 - |xi_q|^2), cast to complex to preserve
        # complex k.
        denom_ns = (k ** 2 - xi_mag_ns ** 2).astype(complex)  # (M_ns,)

        prefactor = (4.0 * np.pi) ** 2 * R ** 4 / self.cell_volume

        # Build the block. We sum over (L, M) outer × (l, m) inner basis
        # indices, with the (N, N) spatial matrix assembled per (L,M,l,m)
        # as a quadratic form in phase_i.
        B = np.zeros((NL, NL), dtype=complex)

        # Precompute "receiver factor per q per L" and "source factor per
        # q per l" to avoid recomputing basis-indexed bessels.
        for idx_LM, (L, M_sh) in enumerate(LM):
            recv_fac_L = i_powers[L] * receiver_bessel[L, :]  # (M_ns,)
            Y_L = Y_grid[idx_LM, :]                            # (M_ns,)
            for idx_lm, (l, m_sh) in enumerate(LM):
                src_fac_l = (
                    np.conj(i_powers[l]) * source_bessel[l, :]
                )                                               # (M_ns,)
                # Note: i^{L-l} = i^L * conj(i^l) = i^L * (-i)^l.
                conjY_l = np.conj(Y_grid[idx_lm, :])
                # factor_q for this (L,M)<-(l,m): shape (M_ns,)
                factor_q = (
                    recv_fac_L * src_fac_l * Y_L * conjY_l / denom_ns
                )
                # N x N block: sum_q phase_i[i, q] * factor_q[q] * conj(phase_i[j, q])
                weighted = phase_i * factor_q[None, :]         # (N, M_ns)
                block_NN = prefactor * (weighted @ np.conj(phase_i.T))
                # Place into global B.
                rows = np.arange(N) * L_basis + idx_LM
                cols = np.arange(N) * L_basis + idx_lm
                B[np.ix_(rows, cols)] += block_NN

        # xi = 0 contribution. Only arises when alpha + q = 0 for some
        # enumerated q (e.g., alpha = 0 with q = 0). For j_l(0) = delta_{l,0}
        # the only surviving matrix element of the single-layer block is at
        # (L=l=0, M=m=0). Evaluating the general formula in the limit
        # ``xi -> 0`` with ``Y_0^0 = 1/sqrt(4 pi)`` and the
        # ``(4 pi)^2 R^4 / |Y|`` prefactor gives
        #
        #     (4 pi)^2 R^4 / |Y| * (1 / (4 pi)) / k^2
        #       = 4 pi R^4 / (|Y| k^2).
        #
        # The receiver-derivative version contributes zero because
        # ``|xi| j_L'(0) = 0`` for every L. Phases at xi=0 are all unity.
        if singular_mask.any() and not receiver_derivative:
            count_sing = int(singular_mask.sum())
            extra = (
                4.0 * np.pi * R ** 4
                / (self.cell_volume * (k ** 2 + 0j))
            )
            extra *= count_sing
            idx_00 = 0  # (l=0, m=0) comes first in our ordering
            for i in range(N):
                for j in range(N):
                    B[i * L_basis + idx_00, j * L_basis + idx_00] += extra

        # Jump-relation self-diagonal.
        if jump_sign != 0.0:
            diag_idx = np.arange(NL)
            B[diag_idx, diag_idx] += jump_sign

        return B

    # ------------------------------------------------------------------
    # Public block builders
    # ------------------------------------------------------------------

    def _build_S_alpha_tilde_block(
        self, alpha: np.ndarray, omega: complex, N_multipole: int,
    ) -> np.ndarray:
        """Interior single-layer ``S_tilde^{alpha, omega}`` (wavenumber ``k_b``)."""
        k_b = omega / self.v_b
        return self._assemble_poisson_block(
            alpha, k_b, N_multipole,
            receiver_derivative=False, jump_sign=0.0,
        )

    def _build_S_alpha_exterior_block(
        self, alpha: np.ndarray, omega: complex, N_multipole: int,
    ) -> np.ndarray:
        """Exterior single-layer ``S^{alpha, k}`` with ``k = omega / v``."""
        k = omega / self.v
        return self._assemble_poisson_block(
            alpha, k, N_multipole,
            receiver_derivative=False, jump_sign=0.0,
        )

    def _build_K_alpha_tilde_star_block(
        self, alpha: np.ndarray, omega: complex, N_multipole: int,
    ) -> np.ndarray:
        """``(-1/2 I + K_tilde^{alpha, omega, *})`` block (interior limit)."""
        k_b = omega / self.v_b
        return self._assemble_poisson_block(
            alpha, k_b, N_multipole,
            receiver_derivative=True, jump_sign=-0.5,
        )

    def _build_K_alpha_exterior_star_block_raw(
        self, alpha: np.ndarray, omega: complex, N_multipole: int,
    ) -> np.ndarray:
        """Raw ``(1/2 I + K^{alpha, k, *})`` block (exterior limit), without
        the ``-delta_i`` prefactor used in ``A^alpha``.
        """
        k = omega / self.v
        return self._assemble_poisson_block(
            alpha, k, N_multipole,
            receiver_derivative=True, jump_sign=+0.5,
        )

    def _build_K_alpha_exterior_star_block(
        self,
        alpha: np.ndarray,
        omega: complex,
        delta_arr: np.ndarray,
        N_multipole: int,
    ) -> np.ndarray:
        """``-delta_i (1/2 I + K^{alpha, k, *})`` block.

        The ``-delta_i`` prefactor is applied row-wise at block-assembly
        time, matching the finite-case convention.
        """
        raw = self._build_K_alpha_exterior_star_block_raw(
            alpha, omega, N_multipole
        )
        L = N_multipole ** 2
        row_delta = np.repeat(delta_arr, L)                       # (NL,)
        return (-row_delta)[:, None] * raw

    # ------------------------------------------------------------------
    # Public assembly / characteristic equation
    # ------------------------------------------------------------------

    def get_A_matrix(
        self,
        alpha: np.ndarray,
        omega: complex,
        delta,
        N_multipole: int = 2,
    ) -> np.ndarray:
        """Assemble the ``(2 N L, 2 N L)`` block matrix ``A^alpha(omega, delta)``.

        Args:
            alpha: Quasi-momentum ``(3,)`` in reciprocal space.
            omega: Complex frequency.
            delta: Density contrast (scalar or length-``N`` array).
            N_multipole: Spherical-harmonic truncation
                (basis dim ``L = N_multipole**2`` per resonator).

        Returns:
            Complex ndarray of shape ``(2 N L, 2 N L)``.
        """
        alpha = np.asarray(alpha, dtype=float).reshape(3,)
        delta_arr = _get_consistent_parameter(delta, self.N).astype(complex)

        S_tilde = self._build_S_alpha_tilde_block(alpha, omega, N_multipole)
        S_ext = self._build_S_alpha_exterior_block(alpha, omega, N_multipole)
        K_tilde = self._build_K_alpha_tilde_star_block(alpha, omega, N_multipole)
        K_ext = self._build_K_alpha_exterior_star_block(
            alpha, omega, delta_arr, N_multipole
        )

        top = np.concatenate([S_tilde, -S_ext], axis=1)
        bot = np.concatenate([K_tilde, K_ext], axis=1)
        return np.concatenate([top, bot], axis=0)

    def characteristic_determinant(
        self,
        alpha: np.ndarray,
        omega: complex,
        delta,
        N_multipole: int = 2,
    ) -> complex:
        """``det A^alpha(omega, delta)``; its zeros are the Bloch resonances."""
        return np.linalg.det(self.get_A_matrix(alpha, omega, delta, N_multipole))

    def compute_resonances(
        self,
        alpha: np.ndarray,
        x0: complex,
        N_roots: int,
        delta,
        N_multipole: int = 2,
        perturbation: float = 1e-3,
        tol: float = 1e-10,
    ) -> np.ndarray:
        """Find ``N_roots`` roots of ``omega -> det A^alpha(omega, delta)``."""
        def f(w):
            return self.characteristic_determinant(
                alpha, w, delta, N_multipole
            )
        return find_roots_muller(
            f, x0, N_roots, perturbation=perturbation, tol=tol
        )

    # ------------------------------------------------------------------
    # Neumann-mode enumeration (shared helpers from continuum.py)
    # ------------------------------------------------------------------

    def compute_neumann_eigenvalues(self, l_max: int, n_max: int) -> List[Dict]:
        """Per-resonator Neumann eigenfrequencies of the ball interior."""
        return _compute_neumann_eigenvalues_for_balls(
            self.radii, self.v_in, l_max, n_max
        )

    def find_omega0_candidates(
        self, l_max: int, n_max: int, tol: float = 1e-8,
    ) -> List[Dict]:
        """Cluster Neumann eigenfrequencies into ``omega_0`` candidates."""
        return _find_omega0_candidates_for_balls(
            self.radii, self.v_in, l_max, n_max, tol
        )

    # ------------------------------------------------------------------
    # Frequency-dependent capacitance matrix (Def. 4.1 / Theorem 4.2)
    # ------------------------------------------------------------------

    def get_frequency_dependent_capacitance(
        self,
        alpha: np.ndarray,
        omega_0: complex,
        index_set: Sequence[Tuple[int, int, int, int]],
        kappa: np.ndarray,
        N_multipole: int,
        delta,
        *,
        hermitian_projection: bool = True,
        real_sh_basis: bool = False,
    ) -> np.ndarray:
        """Quasiperiodic frequency-dependent capacitance matrix (Def. 4.1).

        Direct analogue of
        :meth:`Subwavelength3D.continuum.ContinuumFiniteSWP3D.get_frequency_dependent_capacitance`,
        using the quasiperiodic exterior ``S^{alpha, k_0}`` and
        ``(1/2 I + K^{alpha, k_0, *})`` in place of their free-space
        counterparts.

        Proposition 4.3 of the paper guarantees that the exact
        (infinite-lattice) ``C^alpha(omega_0)`` is Hermitian for real
        ``alpha`` and real ``omega_0``. The truncated Poisson sum used
        internally can introduce a small non-Hermitian component whose
        size reflects the truncation error of the reciprocal-lattice
        series. By default we project onto the Hermitian subspace
        (``C <- (C + C^H) / 2``) so that the returned matrix respects
        the theoretical property exactly; the symmetric part is the best
        Hermitian approximation in Frobenius norm and coincides with the
        exact ``C`` in the infinite-lattice limit. Pass
        ``hermitian_projection=False`` to obtain the raw block-assembly
        output (useful for diagnosing truncation errors).

        Args:
            alpha: Quasi-momentum ``(3,)``.
            omega_0: Reference Neumann eigenfrequency (usually real,
                below the first diffraction threshold for Hermiticity).
            index_set: Sequence of ``(j, l, n, m)`` tuples (from
                :meth:`find_omega0_candidates`).
            kappa: Parallel array of ``kappa`` normalisation constants.
            N_multipole: Must exceed the maximum ``l`` in the index set.
            delta: Density contrast (scalar or length-``N`` array).
            hermitian_projection: If True (default) return
                ``(C + C^H) / 2`` so that the result is Hermitian by
                construction (Prop. 4.3).
            real_sh_basis: If True, conjugate the final matrix by the
                complex-to-real spherical-harmonic unitary ``U`` (see
                :func:`Subwavelength3D.continuum._complex_to_real_sh_unitary`).
                In the real-SH basis, Prop. 4.3's Hermiticity for real
                ``alpha`` coincides with ordinary complex-symmetry — and
                matches the convention under which Prop. 3.13 of the paper
                is stated for the finite problem. Eigenvalues are preserved
                by the unitary transform. Default False.

        Returns:
            Complex ndarray of shape ``(m, m)`` with ``m = len(index_set)``.
        """
        alpha = np.asarray(alpha, dtype=float).reshape(3,)
        if len(index_set) == 0:
            return np.zeros((0, 0), dtype=complex)

        l_max_idx = max(l for (_, l, _, _) in index_set)
        if N_multipole <= l_max_idx:
            raise ValueError(
                f"N_multipole (={N_multipole}) must exceed max l in the "
                f"index set (={l_max_idx})."
            )
        delta_arr = _get_consistent_parameter(delta, self.N).astype(complex)

        S_ext = self._build_S_alpha_exterior_block(alpha, omega_0, N_multipole)
        K_ext_raw = self._build_K_alpha_exterior_star_block_raw(
            alpha, omega_0, N_multipole
        )

        NL = self.N * N_multipole ** 2
        m = len(index_set)

        # RHS: unit vectors at the basis indices (j, l, m).
        G = np.zeros((NL, m), dtype=complex)
        for c, (jc, lc, _nc, mc) in enumerate(index_set):
            G[flat_index(jc, N_multipole, lc, mc), c] = 1.0

        # Lambda_ext[g] = (1/2 I + K^*) S^{-1} [g].
        try:
            tilde_G = np.linalg.solve(S_ext, G)
        except np.linalg.LinAlgError:
            tilde_G, *_ = np.linalg.lstsq(S_ext, G, rcond=None)
        D = K_ext_raw @ tilde_G                                  # (NL, m)

        row_prefactor = np.zeros(m, dtype=complex)
        tau = np.zeros(m, dtype=int)
        for r, (ir, lr, _nr, mr) in enumerate(index_set):
            Ri = float(self.radii[ir])
            vi = (
                complex(self.v_in[ir])
                if np.iscomplexobj(self.v_in)
                else float(self.v_in[ir])
            )
            row_prefactor[r] = (
                -delta_arr[ir] * (vi ** 2) / (2.0 * omega_0)
                * kappa[r] * (Ri ** 2)
            )
            tau[r] = flat_index(ir, N_multipole, lr, mr)

        C = row_prefactor[:, None] * D[tau, :] * kappa[None, :]

        if hermitian_projection:
            # Prop. 4.3: the exact infinite-lattice ``C^alpha(omega_0)`` is
            # Hermitian for real ``alpha`` and real ``omega_0``. The truncated
            # Poisson sum introduces a small anti-Hermitian component whose
            # size reflects the reciprocal-lattice truncation error
            # (empirically ~ 10^-3 at ``lattice_shell_cutoff=3`` for generic
            # irrational alpha). Projecting onto the Hermitian subspace
            # returns the best Frobenius-norm Hermitian approximation, which
            # coincides with the exact ``C`` in the infinite-lattice limit
            # and restores the theoretical property at any truncation.
            C = 0.5 * (C + C.conj().T)
        if real_sh_basis:
            U = _complex_to_real_sh_unitary(index_set)
            C = U @ C @ U.conj().T
        return C

    def compute_nonsubwavelength_resonances(
        self,
        alpha: np.ndarray,
        omega_0: complex,
        index_set: Sequence[Tuple[int, int, int, int]],
        kappa: np.ndarray,
        N_multipole: int,
        delta,
    ) -> np.ndarray:
        """Leading-order Bloch resonances (Theorem 4.2):

            omega_n(alpha) = omega_0 + lambda_n(C^alpha(omega_0)) + o(delta).
        """
        C = self.get_frequency_dependent_capacitance(
            alpha, omega_0, index_set, kappa, N_multipole, delta
        )
        return omega_0 + np.linalg.eigvals(C)
