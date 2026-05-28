import unittest
import numpy as np
from parameterized import parameterized

from Subwavelength3D.nonreciprocal import (
    NonReciprocalPeriodicSWP3D,
    NonReciprocalFiniteSWP3D,
    compute_gauge_harmonics_chain_axis,
)
from Subwavelength3D.classic_periodic import ClassicPeriodicFWP3D


class GammaZeroRecovery(unittest.TestCase):
    """gamma=0 should recover the classical periodic capacitance matrix."""

    @parameterized.expand([
        (0.1,),
        (0.3,),
        (-0.2,),
    ])
    def test_single_resonator_gamma_zero(self, alpha):
        centers = [np.array([0, 0, 0])]
        radii = [1.0]
        L = 10
        k0 = 1e-5

        classical = ClassicPeriodicFWP3D(centers=centers, radii=radii, L=L, k0=k0)
        C_classic = classical.get_capacitance_matrix(alpha=alpha, N_multipole=2)

        nonrecip = NonReciprocalPeriodicSWP3D(
            gamma=0.0, centers=centers, radii=radii, L=L, k0=k0)
        C_nr = nonrecip.get_capacitance_matrix(alpha=alpha, N_multipole=2, N_quad=200)

        np.testing.assert_allclose(
            C_nr, C_classic, atol=1e-3, rtol=1e-3,
            err_msg="gamma=0 should recover classical periodic C(alpha)")

    @parameterized.expand([
        (0.15,),
        (0.5,),
    ])
    def test_two_resonators_gamma_zero(self, alpha):
        centers = [np.array([1, 0, 0]), np.array([5, 0, 0])]
        radii = [1.0, 1.0]
        L = 10
        k0 = 1e-5

        classical = ClassicPeriodicFWP3D(centers=centers, radii=radii, L=L, k0=k0)
        C_classic = classical.get_capacitance_matrix(alpha=alpha, N_multipole=1)

        nonrecip = NonReciprocalPeriodicSWP3D(
            gamma=0.0, centers=centers, radii=radii, L=L, k0=k0)
        C_nr = nonrecip.get_capacitance_matrix(alpha=alpha, N_multipole=1, N_quad=200)

        np.testing.assert_allclose(
            C_nr, C_classic, atol=1e-3, rtol=1e-3,
            err_msg="gamma=0 should recover classical periodic C(alpha)")


class SymbolAsymmetry(unittest.TestCase):
    """For gamma != 0, the symbol Ĉ(alpha) should be asymmetric: Ĉ(-alpha) != Ĉ(alpha)."""

    def test_symbol_asymmetry_single_resonator(self):
        """Key test: the periodic GCM should satisfy Ĉ(-α) ≠ Ĉ(α) for γ ≠ 0."""
        centers = [np.array([0, 0, 0])]
        radii = [0.3]
        L = 1.0
        k0 = 1e-5
        alpha = 0.5

        for gamma in [1.0, 2.0]:
            sys3 = NonReciprocalPeriodicSWP3D(
                gamma=gamma, centers=centers, radii=radii, L=L, k0=k0)
            C_plus = sys3.get_generalised_capacitance_matrix(
                alpha=alpha, N_multipole=2, N_quad=200)
            C_minus = sys3.get_generalised_capacitance_matrix(
                alpha=-alpha, N_multipole=2, N_quad=200)

            diff = np.abs(C_plus[0, 0] - C_minus[0, 0])
            self.assertGreater(
                diff, 1e-2,
                msg=f"Ĉ(+α) should differ from Ĉ(-α) for gamma={gamma}")

    def test_symbol_symmetric_for_gamma_zero(self):
        """For γ=0, the symbol should be symmetric: Ĉ(-α) = Ĉ(α)."""
        centers = [np.array([0, 0, 0])]
        radii = [0.3]
        L = 1.0
        k0 = 1e-5
        alpha = 0.5

        sys3 = NonReciprocalPeriodicSWP3D(
            gamma=0.0, centers=centers, radii=radii, L=L, k0=k0)
        C_plus = sys3.get_generalised_capacitance_matrix(
            alpha=alpha, N_multipole=2, N_quad=200)
        C_minus = sys3.get_generalised_capacitance_matrix(
            alpha=-alpha, N_multipole=2, N_quad=200)

        np.testing.assert_allclose(
            C_plus, C_minus, atol=1e-6,
            err_msg="Ĉ(+α) should equal Ĉ(-α) for gamma=0")


class GaugeHarmonicsChainAxis(unittest.TestCase):
    """Test the chain-axis gauge harmonic expansion."""

    def test_only_m_zero_components(self):
        """exp(gamma*R*cos(theta)) should have only m=0 spherical harmonic components."""
        f1 = compute_gauge_harmonics_chain_axis(gamma=1.0, R=0.3, N_multipole=3, N_quad=200)
        idx = 0
        for l in range(3):
            for m in range(-l, l + 1):
                if m != 0:
                    self.assertAlmostEqual(
                        abs(f1[idx]), 0, places=5,
                        msg=f"f1[l={l},m={m}] should be zero (only m=0 expected)")
                idx += 1

    def test_gamma_zero_monopole_only(self):
        """For gamma=0, only the l=0 component should be nonzero."""
        f1 = compute_gauge_harmonics_chain_axis(gamma=0.0, R=0.3, N_multipole=3, N_quad=200)
        expected_monopole = np.sqrt(4 * np.pi)
        self.assertAlmostEqual(abs(f1[0] - expected_monopole), 0, places=3)


class ScalarCapacitance(unittest.TestCase):
    def test_scalar_output(self):
        sys3 = NonReciprocalPeriodicSWP3D(
            gamma=1.0, centers=[[0, 0, 0]], radii=[0.3], L=2, k0=1e-5)
        C = sys3.get_capacitance_matrix(alpha=0.5, N_multipole=2, N_quad=200)
        self.assertEqual(C.shape, (1, 1))


class BandStructure(unittest.TestCase):
    def test_eigenvalues_vary_with_alpha(self):
        sys3 = NonReciprocalPeriodicSWP3D(
            gamma=1.0, centers=[[0, 0, 0]], radii=[0.3], L=2, k0=1e-5)
        alphas = np.linspace(0.1, 1.0, 5)
        eigs = sys3.compute_band_structure(alphas, N_multipole=2, N_quad=200)
        self.assertGreater(np.max(np.abs(np.diff(eigs[:, 0]))), 1e-6)


class ConstructorValidation(unittest.TestCase):
    def test_unequal_radii_raises(self):
        with self.assertRaises(ValueError):
            NonReciprocalPeriodicSWP3D(
                gamma=1.0, centers=[[0, 0, 0], [3, 0, 0]], radii=[0.3, 0.5], L=5)

    def test_gamma_attribute(self):
        sys3 = NonReciprocalPeriodicSWP3D(
            gamma=-1.5, centers=[[0, 0, 0]], radii=[0.3], L=2)
        self.assertAlmostEqual(sys3.gamma, -1.5)

    def test_off_axis_center_raises(self):
        """Centers not on x-axis should raise."""
        with self.assertRaises(ValueError):
            NonReciprocalPeriodicSWP3D(
                gamma=1.0, centers=[[0, 0, 1]], radii=[0.3], L=2)


if __name__ == "__main__":
    unittest.main()
