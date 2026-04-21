"""Unit tests for :class:`ContinuumPeriodicSWP3D` in
``Subwavelength3D.continuum_periodic``.
"""
import unittest

import numpy as np
from parameterized import parameterized

from Subwavelength3D.continuum import (
    ContinuumFiniteSWP3D,
    _complex_to_real_sh_unitary,
)
from Subwavelength3D.continuum_periodic import (
    ContinuumPeriodicSWP3D,
    _enumerate_lattice_shells,
)


def _simple_cubic_one_ball(
    a: float = 3.0, R: float = 1.0, v_b: float = 1.0,
    shell: int = 3,
) -> ContinuumPeriodicSWP3D:
    """Cubic Bravais cell with a single resonator at the origin."""
    return ContinuumPeriodicSWP3D(
        centers=[[0.0, 0.0, 0.0]], radii=R, v_in=v_b,
        lattice_vectors=np.diag([a, a, a]),
        lattice_shell_cutoff=shell,
    )


class IdenticalResonatorGuard(unittest.TestCase):
    """Theorem 4.2 assumes identical resonators within the unit cell."""

    def test_rejects_heterogeneous_radii(self):
        with self.assertRaises(ValueError):
            ContinuumPeriodicSWP3D(
                centers=[[0, 0, 0], [1, 0, 0]],
                radii=[1.0, 1.5], v_in=1.0,
                lattice_vectors=np.diag([3.0, 3.0, 3.0]),
            )

    def test_rejects_heterogeneous_v_in(self):
        with self.assertRaises(ValueError):
            ContinuumPeriodicSWP3D(
                centers=[[0, 0, 0], [1, 0, 0]],
                radii=1.0, v_in=[1.0, 0.9],
                lattice_vectors=np.diag([3.0, 3.0, 3.0]),
            )

    def test_accepts_uniform_arrays(self):
        cn = ContinuumPeriodicSWP3D(
            centers=[[0, 0, 0], [1.0, 0, 0]],
            radii=[1.0, 1.0], v_in=[1.2, 1.2],
            lattice_vectors=np.diag([3.0, 3.0, 3.0]),
        )
        self.assertEqual(cn.R, 1.0)
        self.assertEqual(cn.v_b, 1.2)
        self.assertEqual(cn.N, 2)


class LatticeEnumeration(unittest.TestCase):

    def test_zero_triple_first(self):
        triples = _enumerate_lattice_shells(2)
        np.testing.assert_array_equal(triples[0], [0, 0, 0])

    def test_count_is_cube(self):
        for m_cap in [0, 1, 2, 3]:
            triples = _enumerate_lattice_shells(m_cap)
            self.assertEqual(triples.shape, ((2 * m_cap + 1) ** 3, 3))


class AssemblySanity(unittest.TestCase):

    @parameterized.expand([
        [2, 2],  # N_multipole, shell
        [3, 1],
        [3, 3],
    ])
    def test_A_matrix_shape_and_finite(self, N_multipole, shell):
        cn = _simple_cubic_one_ball(shell=shell)
        alpha = np.array([0.1, 0.0, 0.2])
        A = cn.get_A_matrix(
            alpha, omega=0.5 + 0j, delta=1e-3, N_multipole=N_multipole,
        )
        expected_dim = 2 * 1 * N_multipole ** 2
        self.assertEqual(A.shape, (expected_dim, expected_dim))
        self.assertTrue(np.all(np.isfinite(A)))

    def test_two_resonator_cell_finite(self):
        cn = ContinuumPeriodicSWP3D(
            centers=[[0, 0, 0], [1.2, 0, 0]], radii=0.5, v_in=1.0,
            lattice_vectors=np.diag([3.0, 3.0, 3.0]),
            lattice_shell_cutoff=2,
        )
        A = cn.get_A_matrix(
            np.array([0.3, 0.2, 0.1]),
            omega=0.5 + 0j, delta=1e-3, N_multipole=2,
        )
        self.assertEqual(A.shape, (2 * 2 * 4, 2 * 2 * 4))
        self.assertTrue(np.all(np.isfinite(A)))

    def test_q_zero_only_closed_form(self):
        """Poisson-sum sanity check. With ``lattice_shell_cutoff = 0`` and
        ``alpha = 0`` the sum retains only the ``q = 0`` reciprocal vector.
        Since ``j_l(0) = delta_{l, 0}`` and ``Y_0^0 = 1 / sqrt(4 pi)``, the
        interior single-layer block reduces to a matrix whose only nonzero
        entries are at ``(l = l' = 0, m = m' = 0)`` with value
        ``4 pi R^4 / (|Y| * k_b^2)`` for every resonator pair ``(i, j)``.
        """
        R = 1.0
        v_b = 1.0
        L_cell = 5.0
        N_multipole = 3
        omega = 1.3 + 0j
        k_b = omega / v_b

        periodic = ContinuumPeriodicSWP3D(
            centers=[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            radii=R, v_in=v_b,
            lattice_vectors=np.diag([L_cell, L_cell, L_cell]),
            lattice_shell_cutoff=0,
        )
        S_tilde = periodic._build_S_alpha_tilde_block(
            np.zeros(3), omega, N_multipole,
        )
        expected = 4.0 * np.pi * R ** 4 / (L_cell ** 3 * k_b ** 2)
        Lb = N_multipole ** 2
        # Every (l=0, m=0)(l'=0, m'=0) cross-resonator entry equals ``expected``.
        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(
                    S_tilde[i * Lb + 0, j * Lb + 0], expected,
                    places=12,
                )
        # All other entries must vanish.
        mask = np.ones_like(S_tilde, dtype=bool)
        for i in range(2):
            for j in range(2):
                mask[i * Lb + 0, j * Lb + 0] = False
        self.assertLess(np.abs(S_tilde[mask]).max(), 1e-12)


class DeltaScaling(unittest.TestCase):
    """Theorem 4.2 says shift = ``delta * lambda_n(C^alpha) + o(delta)``."""

    def test_delta_linear_leading_order(self):
        cn = _simple_cubic_one_ball()
        cands = cn.find_omega0_candidates(l_max=1, n_max=1)
        c0 = cands[0]
        alpha = np.array([0.3, 0.0, 0.0])
        shifts = {}
        for delta in [1e-2, 1e-3, 1e-4]:
            eigs = cn.compute_nonsubwavelength_resonances(
                alpha, c0['omega_0'], c0['index_set'], c0['kappa'],
                N_multipole=3, delta=delta,
            )
            shifts[delta] = np.max(np.abs(eigs - c0['omega_0']))
        # Linear delta-scaling: shift(1e-4) / shift(1e-3) should be ~0.1.
        ratio1 = shifts[1e-3] / shifts[1e-2]
        ratio2 = shifts[1e-4] / shifts[1e-3]
        self.assertAlmostEqual(ratio1, 0.1, delta=0.01)
        self.assertAlmostEqual(ratio2, 0.1, delta=0.01)


class Hermiticity(unittest.TestCase):
    """Proposition 4.3: for real ``alpha`` and real reference Neumann
    frequency, ``C^alpha(omega_0)`` is Hermitian.
    """

    @parameterized.expand([
        [np.array([0.0, 0.0, 0.0])],
        [np.array([0.5, 0.0, 0.0])],
        [np.array([0.2, 0.3, 0.1])],
    ])
    def test_C_is_hermitian(self, alpha):
        cn = _simple_cubic_one_ball()
        cands = cn.find_omega0_candidates(l_max=1, n_max=1)
        c0 = cands[0]
        C = cn.get_frequency_dependent_capacitance(
            alpha, c0['omega_0'], c0['index_set'], c0['kappa'],
            N_multipole=3, delta=1e-3,
        )
        norm_C = np.linalg.norm(C)
        self.assertGreater(norm_C, 1e-10)
        hermitian_resid = np.linalg.norm(C - C.conj().T) / norm_C
        self.assertLess(hermitian_resid, 1e-8)


class FiniteToPeriodicConvergence(unittest.TestCase):
    """Central acceptance test: as ``N -> infinity`` the real parts of the
    finite-chain frequency-dependent-capacitance eigenvalues approach the
    periodic band traced by :meth:`ContinuumPeriodicSWP3D.compute_nonsubwavelength_resonances`
    swept over the 1D Brillouin zone.

    Ignoring the imaginary part (chain-edge radiation loss, an O(1)
    finite-size effect) is essential here.
    """

    def _build_band(self, periodic, omega_0, index_set, kappa,
                    L_z, N_multipole, delta, n_alpha):
        alpha_zs = np.linspace(-np.pi / L_z, np.pi / L_z, n_alpha)
        band = []
        for az in alpha_zs:
            alpha_vec = np.array([0.0, 0.0, az])
            lam = periodic.compute_nonsubwavelength_resonances(
                alpha_vec, omega_0, index_set, kappa,
                N_multipole=N_multipole, delta=delta,
            )
            band.append(lam)
        return np.array(band)

    def test_one_dim_chain_real_parts_converge(self):
        L_z, L_xy = 5.0, 80.0
        R = 1.0
        v_b = 1.0
        delta = 1e-3
        N_multipole = 3

        periodic = ContinuumPeriodicSWP3D(
            centers=[[0, 0, 0]], radii=R, v_in=v_b,
            lattice_vectors=np.diag([L_xy, L_xy, L_z]),
            lattice_shell_cutoff=3,
        )
        per_cands = periodic.find_omega0_candidates(l_max=1, n_max=1)
        c0 = per_cands[0]

        band = self._build_band(
            periodic, c0['omega_0'], c0['index_set'], c0['kappa'],
            L_z, N_multipole, delta, n_alpha=81,
        )
        band_re = band.real.flatten()

        # Evaluate the finite chain for N in a couple of sizes.
        dists_by_N = {}
        for N in [10, 20]:
            centers = np.array([[0, 0, i * L_z] for i in range(N)])
            finite = ContinuumFiniteSWP3D(centers=centers, radii=R, v_in=v_b)
            fcands = finite.find_omega0_candidates(l_max=1, n_max=1)
            C_fin = finite.get_frequency_dependent_capacitance(
                fcands[0]['omega_0'], fcands[0]['index_set'], fcands[0]['kappa'],
                N_multipole=N_multipole, delta=delta,
            )
            fin_freqs_re = (fcands[0]['omega_0'] + np.linalg.eigvals(C_fin)).real
            dists = np.array([np.min(np.abs(band_re - fe)) for fe in fin_freqs_re])
            dists_by_N[N] = dists

        # Median distance (bulk behaviour) must be << band width.
        band_width = band_re.max() - band_re.min()
        for N, dists in dists_by_N.items():
            self.assertLess(
                np.median(dists), 0.3 * band_width,
                msg=(f"N={N}: median(finite-to-periodic Re distance) = "
                     f"{np.median(dists):.3e} should be << band width "
                     f"{band_width:.3e}"),
            )

        # 80%-quantile should also stay well within the band.
        for N, dists in dists_by_N.items():
            self.assertLess(
                np.quantile(dists, 0.8), 0.5 * band_width,
                msg=(f"N={N}: 80%-quantile distance = "
                     f"{np.quantile(dists, 0.8):.3e} should be < "
                     f"{0.5 * band_width:.3e}"),
            )


class RealSHBasisOption(unittest.TestCase):
    """``get_frequency_dependent_capacitance(..., real_sh_basis=True)`` locks
    in the public-API analogue of the finite-case test
    :class:`test_3D_continuum.ComplexSymmetryProp313.test_real_sh_basis_option_returns_complex_symmetric`.

    In the real-SH basis, Prop. 4.3 Hermiticity (for real ``alpha``) coincides
    with ordinary complex-symmetry, so ``real_sh_basis=True`` combined with
    ``hermitian_projection=True`` returns a matrix that is simultaneously
    symmetric AND Hermitian, i.e. real-symmetric.
    """

    def test_option_matches_manual_transform(self):
        periodic = _simple_cubic_one_ball()
        c0 = periodic.find_omega0_candidates(l_max=1, n_max=1)[0]
        alpha = np.array([0.3, 0.1, 0.2])

        C_default = periodic.get_frequency_dependent_capacitance(
            alpha, c0['omega_0'], c0['index_set'], c0['kappa'],
            N_multipole=3, delta=1e-3,
        )
        C_real = periodic.get_frequency_dependent_capacitance(
            alpha, c0['omega_0'], c0['index_set'], c0['kappa'],
            N_multipole=3, delta=1e-3, real_sh_basis=True,
        )
        U = _complex_to_real_sh_unitary(c0['index_set'])
        np.testing.assert_allclose(C_real, U @ C_default @ U.conj().T, atol=1e-12)

    def test_real_sh_basis_is_real_symmetric_at_real_alpha(self):
        periodic = _simple_cubic_one_ball()
        c0 = periodic.find_omega0_candidates(l_max=1, n_max=1)[0]
        alpha = np.array([0.2, 0.3, 0.1])
        C_real = periodic.get_frequency_dependent_capacitance(
            alpha, c0['omega_0'], c0['index_set'], c0['kappa'],
            N_multipole=3, delta=1e-3, real_sh_basis=True,
        )
        # Hermiticity (Prop. 4.3) + complex symmetry => real symmetric.
        resid_sym = np.linalg.norm(C_real - C_real.T) / np.linalg.norm(C_real)
        resid_her = np.linalg.norm(C_real - C_real.conj().T) / np.linalg.norm(C_real)
        self.assertLess(resid_sym, 1e-8)
        self.assertLess(resid_her, 1e-8)
        # Imaginary part must therefore be at the noise floor.
        self.assertLess(
            np.linalg.norm(C_real.imag) / np.linalg.norm(C_real), 1e-8,
        )

    def test_eigenvalues_invariant_under_option(self):
        periodic = _simple_cubic_one_ball()
        c0 = periodic.find_omega0_candidates(l_max=1, n_max=1)[0]
        alpha = np.array([0.3, 0.0, 0.0])
        kwargs = dict(
            alpha=alpha, omega_0=c0['omega_0'], index_set=c0['index_set'],
            kappa=c0['kappa'], N_multipole=3, delta=1e-3,
        )
        eigs_default = np.sort_complex(np.linalg.eigvals(
            periodic.get_frequency_dependent_capacitance(**kwargs)))
        eigs_real = np.sort_complex(np.linalg.eigvals(
            periodic.get_frequency_dependent_capacitance(
                **kwargs, real_sh_basis=True)))
        np.testing.assert_allclose(eigs_default, eigs_real, atol=1e-10)


if __name__ == '__main__':
    unittest.main()
