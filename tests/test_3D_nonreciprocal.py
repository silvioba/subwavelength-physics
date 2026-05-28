import unittest
import numpy as np
from parameterized import parameterized

from Subwavelength3D.nonreciprocal import (
    NonReciprocalFiniteSWP3D,
    compute_gauge_harmonics,
    compute_normalization_integral,
)
from Subwavelength3D.classic_finite import ClassicFiniteSWP3D


class GaugeHarmonicsTests(unittest.TestCase):

    def test_gamma_zero_only_monopole(self):
        """For gamma=0, exp(0)=1, so only the Y_0^0 component is nonzero."""
        f1 = compute_gauge_harmonics(gamma=0.0, R=0.3, N_multipole=3, N_quad=200)
        expected_monopole = np.sqrt(4 * np.pi)
        self.assertAlmostEqual(abs(f1[0] - expected_monopole), 0, places=3)
        for i in range(1, len(f1)):
            self.assertAlmostEqual(abs(f1[i]), 0, places=3,
                                   msg=f"f1[{i}] should be zero for gamma=0")

    def test_nonzero_gamma_has_higher_harmonics(self):
        """For gamma != 0, higher-order components should be nonzero."""
        f1 = compute_gauge_harmonics(gamma=1.0, R=0.3, N_multipole=3, N_quad=200)
        # l=1 components should be nonzero
        has_nonzero = any(abs(f1[i]) > 1e-6 for i in range(1, 4))
        self.assertTrue(has_nonzero, "Higher harmonics should be nonzero for gamma != 0")


class NormalizationTests(unittest.TestCase):

    def test_gamma_zero_gives_volume(self):
        """For gamma=0, the normalization integral should equal 4/3*pi*R^3."""
        R = 0.5
        int_A = compute_normalization_integral(gamma=0.0, R=R, N_quad=200)
        expected = 4 / 3 * np.pi * R**3
        np.testing.assert_allclose(int_A, expected, rtol=1e-3,
                                   err_msg="Normalization integral should be volume for gamma=0")

    def test_positive_gamma_increases_integral(self):
        """For positive gamma, the integral should be larger than the volume."""
        R = 0.3
        vol = 4 / 3 * np.pi * R**3
        int_A = compute_normalization_integral(gamma=2.0, R=R, N_quad=200)
        self.assertGreater(int_A, vol)


class CapacitanceMatrixTests(unittest.TestCase):

    def _make_chain(self, N, gamma=0.0):
        centers = np.array([[i, 0, 0] for i in range(N)], dtype=float)
        radii = 0.3 * np.ones(N)
        return NonReciprocalFiniteSWP3D(gamma=gamma, centers=centers, radii=radii)

    @parameterized.expand([
        [2, 1],
        [3, 1],
        [2, 2],
    ])
    def test_gamma_zero_recovers_classical_capacitance(self, N, N_multipole):
        """With gamma=0, the non-reciprocal C should match the classical C."""
        centers = np.array([[i * 3, 0, 0] for i in range(N)], dtype=float)
        radii = 0.3 * np.ones(N)

        nr = NonReciprocalFiniteSWP3D(gamma=0.0, centers=centers, radii=radii)
        C_nr = nr.get_capacitance_matrix(N_multipole=N_multipole, N_quad=200)

        cl = ClassicFiniteSWP3D(centers=centers, radii=radii)
        C_cl = cl.get_capacitance_matrix(N_multipole=N_multipole, method='general')

        np.testing.assert_allclose(
            np.real(C_nr), C_cl,
            atol=1e-3, rtol=1e-3,
            err_msg="gamma=0 non-reciprocal C should match classical C"
        )

    @parameterized.expand([
        [3, 1, -1.0],
        [4, 1, 1.0],
        [3, 2, -0.5],
    ])
    def test_capacitance_matrix_asymmetric(self, N, N_multipole, gamma):
        """With gamma != 0, the capacitance matrix should not be symmetric."""
        system = self._make_chain(N, gamma=gamma)
        C = system.get_capacitance_matrix(N_multipole=N_multipole, N_quad=200)
        diff = np.linalg.norm(C - C.T)
        self.assertGreater(diff, 1e-6,
                           msg="Capacitance matrix should be asymmetric for gamma != 0")

    @parameterized.expand([
        [2, 1],
        [3, 1],
    ])
    def test_eigenvalues_real_for_gamma_zero(self, N, N_multipole):
        """Eigenvalues should be real when gamma=0."""
        centers = np.array([[i * 3, 0, 0] for i in range(N)], dtype=float)
        radii = 0.3 * np.ones(N)
        system = NonReciprocalFiniteSWP3D(gamma=0.0, centers=centers, radii=radii)
        D, S = system.compute_sorted_eigs_capacitance_matrix(
            N_multipole=N_multipole, N_quad=200
        )
        np.testing.assert_allclose(
            np.imag(D), 0, atol=1e-3,
            err_msg="Eigenvalues should be real for gamma=0"
        )

    def test_skin_effect_localization(self):
        """For a chain with gamma != 0, eigenmodes should show localization."""
        N = 10
        centers = np.array([[i, 0, 0] for i in range(N)], dtype=float)
        radii = 0.3 * np.ones(N)

        system = NonReciprocalFiniteSWP3D(gamma=-2.0, centers=centers, radii=radii)
        D, S = system.compute_sorted_eigs_capacitance_matrix(
            N_multipole=1, N_quad=200
        )

        # Check that the first eigenmode has non-uniform amplitude
        # (localized modes should have higher amplitude on one side)
        mode = np.abs(S[:, 0])
        ratio = mode.max() / (mode.min() + 1e-15)
        self.assertGreater(ratio, 1.5,
                           msg="First eigenmode should show localization for gamma != 0")

    @parameterized.expand([
        [
            np.array([[0, 0, 0], [3, 0, 0]]),
            [1, 0],
            1,
            -1.0,
        ],
        [
            np.array([[0, 0, 0], [3, 0, 0], [6, 0, 0]]),
            [0, 2, 1],
            1,
            -0.5,
        ],
    ])
    def test_permutation_consistency(self, centers, permutation, N_multipole, gamma):
        """Permuting resonator indices should create a similarity transform on C."""
        radii = 0.3 * np.ones(len(centers))

        C1 = NonReciprocalFiniteSWP3D(
            gamma=gamma, centers=centers, radii=radii
        ).get_capacitance_matrix(N_multipole=N_multipole, N_quad=200)

        perm = np.array(permutation)
        permuted_centers = centers[perm]
        C2 = NonReciprocalFiniteSWP3D(
            gamma=gamma, centers=permuted_centers, radii=radii
        ).get_capacitance_matrix(N_multipole=N_multipole, N_quad=200)

        inv_perm = np.argsort(perm)
        np.testing.assert_allclose(
            C1, C2[:, inv_perm][inv_perm, :],
            atol=1e-4,
            err_msg="Capacitance matrices should be related by permutation similarity"
        )


class ConstructorValidation(unittest.TestCase):

    def test_unequal_radii_raises(self):
        """Unequal radii should raise ValueError."""
        with self.assertRaises(ValueError):
            NonReciprocalFiniteSWP3D(
                gamma=1.0,
                centers=[[0, 0, 0], [3, 0, 0]],
                radii=[0.3, 0.5],
            )

    def test_chain_factory(self):
        """get_chain should work with gamma parameter."""
        system = NonReciprocalFiniteSWP3D.get_chain(N=5, sep=1.0, radius=0.3, gamma=-1.0)
        self.assertEqual(system.N, 5)
        self.assertAlmostEqual(system.gamma, -1.0)


if __name__ == "__main__":
    unittest.main()
