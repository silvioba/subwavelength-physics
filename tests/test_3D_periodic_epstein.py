import unittest
import numpy as np
from parameterized import parameterized

from Subwavelength3D.classic_periodic import ClassicPeriodicFWP3D
from Subwavelength3D.epstein import (
    compute_single_layer_potential_matrix_epstein,
    compute_capacitance_matrix_epstein,
)


class EpsteinMatchesLatticeSums(unittest.TestCase):
    """For N=1 per cell, Epstein should match lattice sums (N_multipole=1) exactly."""

    @parameterized.expand([
        (0.1,), (0.3,), (0.5,), (1.0,), (2.0,), (3.0,),
    ])
    def test_N1_capacitance_matches(self, alpha):
        R = 0.3; L = 1.0; k0 = 1e-5
        cp = ClassicPeriodicFWP3D(
            centers=[np.array([0, 0, 0])], radii=[R], L=L, k0=k0)

        C_lattice = cp.get_capacitance_matrix(
            alpha=alpha, N_multipole=1, method='lattice_sums')
        C_epstein = cp.get_capacitance_matrix(
            alpha=alpha, method='epstein')

        np.testing.assert_allclose(
            C_epstein, C_lattice, rtol=1e-6,
            err_msg=f"Epstein should match lattice sums for N=1 at alpha={alpha}")

    @parameterized.expand([
        (0.2,), (0.7,), (1.5,),
    ])
    def test_N1_GCM_matches(self, alpha):
        R = 0.3; L = 1.0; k0 = 1e-5
        cp = ClassicPeriodicFWP3D(
            centers=[np.array([0, 0, 0])], radii=[R], L=L, k0=k0)

        GCM_lattice = cp.get_generalised_capacitance_matrix(
            alpha=alpha, N_multipole=1, method='lattice_sums')
        GCM_epstein = cp.get_generalised_capacitance_matrix(
            alpha=alpha, method='epstein')

        np.testing.assert_allclose(
            GCM_epstein, GCM_lattice, rtol=1e-6,
            err_msg=f"Epstein GCM should match lattice sums for N=1 at alpha={alpha}")


class EpsteinBandStructure(unittest.TestCase):
    """Band structure from Epstein should match lattice sums across the BZ."""

    def test_band_sweep_N1(self):
        R = 0.3; L = 1.0; k0 = 1e-5
        cp = ClassicPeriodicFWP3D(
            centers=[np.array([0, 0, 0])], radii=[R], L=L, k0=k0)

        alphas = np.linspace(0.1, np.pi / L - 0.1, 20)
        for alpha in alphas:
            C_lat = cp.get_capacitance_matrix(
                alpha=alpha, N_multipole=1, method='lattice_sums')[0, 0]
            C_ep = cp.get_capacitance_matrix(
                alpha=alpha, method='epstein')[0, 0]
            np.testing.assert_allclose(
                C_ep, C_lat, rtol=1e-5,
                err_msg=f"Band mismatch at alpha={alpha:.4f}")


class EpsteinSymmetryN2(unittest.TestCase):
    """For N=2 per cell, Epstein C should satisfy Lemma 6.3.1 properties."""

    @parameterized.expand([
        (0.3,), (0.7,), (1.5,),
    ])
    def test_symmetry_properties(self, alpha):
        R = 1.0; L = 10.0
        centers = [np.array([1, 0, 0]), np.array([5, 0, 0])]
        C = compute_capacitance_matrix_epstein(
            np.array(centers), np.array([R, R]), L, alpha)

        # C[0,0] = C[1,1] (equal radii, symmetric positions)
        np.testing.assert_allclose(
            C[0, 0], C[1, 1], atol=1e-8,
            err_msg="Diagonal entries should be equal for equal radii")

        # C[0,1] = conj(C[1,0])
        np.testing.assert_allclose(
            C[0, 1], np.conj(C[1, 0]), atol=1e-8,
            err_msg="Off-diagonal should satisfy C[0,1] = conj(C[1,0])")

        # Im(C[0,0]) ≈ 0
        self.assertLess(
            abs(C[0, 0].imag), 1e-6,
            msg="Diagonal should be real")


class EpsteinDistinctRadii(unittest.TestCase):
    """Tests for N=2 per cell with distinct radii."""

    @parameterized.expand([
        (0.3,), (0.7,), (1.5,),
    ])
    def test_methods_match_distinct_radii(self, alpha):
        """Epstein and lattice_sums should agree for distinct radii."""
        R1, R2 = 0.5, 1.0; L = 10.0; k0 = 1e-5
        centers = [np.array([1, 0, 0]), np.array([5, 0, 0])]
        cp = ClassicPeriodicFWP3D(
            centers=centers, radii=[R1, R2], L=L, k0=k0)

        C_lat = cp.get_capacitance_matrix(
            alpha=alpha, N_multipole=1, method='lattice_sums')
        C_ep = cp.get_capacitance_matrix(
            alpha=alpha, method='epstein')

        np.testing.assert_allclose(
            C_ep, C_lat, rtol=1e-6,
            err_msg=f"Methods should agree for distinct radii at alpha={alpha}")

    @parameterized.expand([
        (0.3,), (1.0,),
    ])
    def test_distinct_radii_different_diagonal(self, alpha):
        """With distinct radii, diagonal entries should differ."""
        R1, R2 = 0.5, 1.0; L = 10.0
        centers = [np.array([1, 0, 0]), np.array([5, 0, 0])]
        C = compute_capacitance_matrix_epstein(
            np.array(centers), np.array([R1, R2]), L, alpha)

        self.assertGreater(
            abs(C[0, 0] - C[1, 1]), 1.0,
            msg="Diagonal entries should differ for distinct radii")

    @parameterized.expand([
        (0.3,), (1.0,),
    ])
    def test_distinct_radii_real_diagonal(self, alpha):
        """Diagonal entries should be real even with distinct radii."""
        R1, R2 = 0.5, 1.0; L = 10.0
        centers = [np.array([1, 0, 0]), np.array([5, 0, 0])]
        C = compute_capacitance_matrix_epstein(
            np.array(centers), np.array([R1, R2]), L, alpha)

        self.assertLess(abs(C[0, 0].imag), 1e-6)
        self.assertLess(abs(C[1, 1].imag), 1e-6)

    def test_spectral_convergence_distinct_radii(self):
        """Finite system eigenvalues should lie within periodic band ranges."""
        from Subwavelength3D.classic_finite import ClassicFiniteSWP3D

        R1, R2 = 0.5, 1.0; L = 10.0; k0 = 1e-5
        centers_cell = [np.array([1, 0, 0]), np.array([5, 0, 0])]
        cp = ClassicPeriodicFWP3D(
            centers=centers_cell, radii=[R1, R2], L=L, k0=k0)
        V = cp.get_material_matrix()

        # Periodic band ranges
        alphas = np.linspace(0.01, np.pi / L - 0.01, 50)
        all_eigs = []
        for a in alphas:
            C = cp.get_capacitance_matrix(alpha=a, method='epstein')
            all_eigs.append(np.sort(np.linalg.eigvals(V @ C).real))
        all_eigs = np.array(all_eigs)
        band_mins = all_eigs.min(axis=0)
        band_maxs = all_eigs.max(axis=0)

        # Finite system: 30 unit cells
        N_cells = 30
        centers_fin = []
        radii_fin = []
        for cell in range(N_cells):
            centers_fin.append([cell * L + 1.0, 0, 0])
            radii_fin.append(R1)
            centers_fin.append([cell * L + 5.0, 0, 0])
            radii_fin.append(R2)

        sys_fin = ClassicFiniteSWP3D(
            centers=np.array(centers_fin), radii=np.array(radii_fin))
        D_fin, _ = sys_fin.compute_sorted_eigs_capacitance_matrix(
            N_multipole=1, method='general')

        # Most eigenvalues should fall within the band ranges (with some tolerance)
        margin = 0.5
        in_bands = 0
        for ev in D_fin.real:
            for b in range(2):
                if band_mins[b] - margin <= ev <= band_maxs[b] + margin:
                    in_bands += 1
                    break
        fraction = in_bands / len(D_fin)
        self.assertGreater(
            fraction, 0.8,
            msg=f"At least 80% of finite eigs should be in bands, got {fraction:.1%}")


class EpsteinEdgeCases(unittest.TestCase):

    def test_alpha_zero_raises(self):
        with self.assertRaises(ValueError):
            compute_capacitance_matrix_epstein(
                np.array([[0, 0, 0]]), np.array([0.3]), 1.0, alpha=0.0)

    def test_alpha_near_zero_raises(self):
        with self.assertRaises(ValueError):
            compute_capacitance_matrix_epstein(
                np.array([[0, 0, 0]]), np.array([0.3]), 1.0, alpha=1e-16)


if __name__ == "__main__":
    unittest.main()
