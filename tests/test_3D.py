import unittest
import numpy as np
from parameterized import parameterized

from Subwavelength3D.classic import (
    ClassicFiniteFWP3D,
    ClassicPeriodicFWP3D,
    flat_index,
    lattice_sums,
    C_coefficient,
    B_coefficient,
)


class CapacitanceMatrixConstruction(unittest.TestCase):

    @parameterized.expand([[0, 1], [0, 5], [2, 1], [3, 5], [3, 1], [3, 2]])
    def test_get_S_index_continuity(self, N, L):
        total_number_base_functions = L**2

        total = total_number_base_functions * N
        all_indices = []
        for n in range(N):
            for l in range(L):
                for m in range(-l, l + 1):
                    # print(n, L, l, m, "-->", get_S_index(n=n, L=L, l=l, m=m))
                    all_indices.append(flat_index(n=n, L=L, l=l, m=m))
        np.testing.assert_array_equal(np.arange(total), np.array(all_indices))

    @parameterized.expand([[2, 1], [3, 5], [3, 2], [3, 1]])
    def test_get_S_index_upper_bound(self, N, L):
        total_number_base_functions = L**2

        total = total_number_base_functions * N
        sima = 0
        for n in range(N):
            for l in range(L):
                for m in range(-l, l + 1):
                    si = flat_index(n=n, L=L, l=l, m=m)
                    self.assertGreaterEqual(si, 0)
                    self.assertLess(si, total)
                    sima = max(sima, si)
        self.assertEqual(sima, total - 1)


class CapacitanceMatrix(unittest.TestCase):

    @parameterized.expand(
        [
            [
                [
                    np.array([0, 0, 1]),
                    np.array([0, 0, 5]),
                    np.array([0, 0, 9]),
                    np.array([0, 0, 20]),
                ],
                np.array([1, 1, 1, 1]),
                1e-5,
                1,
            ],
            [
                [
                    np.array([0, 0, 1]),
                    np.array([0, 0, 5]),
                    np.array([0, 0, 9]),
                    np.array([0, 0, 20]),
                ],
                np.array([1, 1, 1, 1]),
                1e-5,
                2,
            ],
        ],
    )
    def test_diagonally_dominant(self, centers, radii, k0, N_multipole):
        C = ClassicFiniteFWP3D(
            centers=centers, radii=radii, k0=k0
        ).get_capacitance_matrix(N_multipole=N_multipole)
        for i in range(C.shape[0]):
            self.assertGreater(
                C[i, i].real, 0, f"Diagonal element ({i}, {i}) is not positive"
            )
            for j in range(C.shape[1]):
                if i != j:
                    self.assertLess(
                        C[i, j].real,
                        0,
                        f"Off-Diagonal element ({i}, {j}) is not negative",
                    )
            self.assertGreater(
                C[i, i],
                np.sum(np.abs(C[i, :])) - C[i, i],
                "Matrix is not diagonally dominant",
            )


class QuasiPeriodicCapacitanceMatrixConstruction(unittest.TestCase):

    @parameterized.expand(
        [
            (
                0.25,  # alpha
                0,  # N_multipole - 1
                10,  # L
                1e-5,  # k
                -2.8209e-01 + 3.6152e03j,  # old_code_result
            ),
            (
                0.1735,  # alpha
                0,  # N_multipole - 1
                13,  # L
                1e-5,  # k
                -2.8209e-01 + 2.5676e03j,  # old_code_result
            ),
            (
                0.1735,  # alpha
                1,  # N_multipole - 1
                13,  # L
                1e-5,  # k
                1e7 * (3.3804 + 0j),  # old_code_result
            ),
            (
                0.0785,  # alpha
                0,  # Nmulti
                10,  # L
                1e-05,  # k0
                -2.8209e-01 - 1.5114e03j,  # old_code_result
            ),
            (
                0.0785,  # alpha
                1,  # N_multipole - 1
                10,  # L
                1e-05,  # k0
                1e7 * (9.5939 - 0.0000j),  # old_code_result
            ),
        ]
    )
    def test_lattice_sums(self, alpha, n, L, k, old_code_result):
        res = lattice_sums(alpha=alpha, n=n, L=L, k=k)
        np.testing.assert_approx_equal(
            res.real,
            old_code_result.real,
            significant=5,
            err_msg="Real part of the lattice sum does not match",
        )
        if abs(res.imag - old_code_result.imag) > 1e-5:
            np.testing.assert_approx_equal(
                res.imag,
                old_code_result.imag,
                significant=5,
                err_msg="Imaginary part of the lattice sum does not match",
            )

    def test_C_coef(self):
        """Compare new and old code for C_coefficent"""
        self.assertAlmostEqual(
            C_coefficient(l=0, m=0, lp=0, mp=0, lam=0, mu=0), 3.5449, places=4
        )
        self.assertAlmostEqual(
            C_coefficient(l=1, m=0, lp=0, mp=0, lam=0, mu=0), 0, places=4
        )
        self.assertAlmostEqual(
            C_coefficient(l=0, m=0, lp=1, mp=0, lam=0, mu=0), 0, places=4
        )
        self.assertAlmostEqual(
            C_coefficient(l=1, m=0, lp=1, mp=0, lam=0, mu=0), 3.5449, places=4
        )
        self.assertAlmostEqual(
            C_coefficient(l=0, m=0, lp=0, mp=0, lam=1, mu=0), 0, places=4
        )
        self.assertAlmostEqual(
            C_coefficient(l=1, m=0, lp=0, mp=0, lam=1, mu=0), 3.5449, places=4
        )
        self.assertAlmostEqual(
            C_coefficient(l=0, m=0, lp=1, mp=0, lam=1, mu=0), -3.5449, places=4
        )

    @parameterized.expand(
        [
            (
                0.3142,  # alpha
                1,  # n_multipole
                10,  # L
                1e-7,  # k
                -1.0000e00 + 1.3863e06j,  # old_code_result
            ),
            (
                0.1571,  # alpha
                1,  # n_multipole
                10,  # L
                1e-7,  # k
                -1.0000e00 + 6.9315e05j,  # old_code_result
            ),
            (
                0.2356,  # alpha
                1,  # n_multipole
                10,  # L
                1e-7,  # k
                -1.0000e00 + 1.2279e06j,  # old_code_result
            ),
        ]
    )
    def test_B_coef(self, alpha, n, L, k0, old_code_result):
        out = B_coefficient(alpha, l=0, m=0, lp=0, mp=0, L=L, k0=k0, N_multipole=n)
        np.testing.assert_approx_equal(
            out.real,
            old_code_result.real,
            significant=5,
            err_msg="Real part of B does not match old code result",
        )
        np.testing.assert_approx_equal(
            out.imag,
            old_code_result.imag,
            significant=5,
            err_msg="Imag part of B does not match old code result",
        )


class QuasiPeriodicCapacitanceMatrix(unittest.TestCase):

    @parameterized.expand(
        [
            (
                [np.array([0, 0, 1]), np.array([0, 0, 5])],  # centers
                np.array([1, 1, 1, 1]),  # radii
                10,  # L
                1e-5,  # k0
                1,  # N_multipole
                0.0754,  # alpha
                np.array(
                    [
                        [-0.9654 - 0.0000j, -0.0654 + 0.0000j],
                        [-0.0654 + 0.0000j, -0.9654 - 0.0000j],
                    ]
                ),
            )
        ]
    )
    def test_properties_2x2_S_matrix(
        self, centers, radii, L, k0, N_multipole, alpha, expected_S
    ):
        """"""
        pwp = ClassicPeriodicFWP3D(centers=centers, radii=radii, L=L, k0=k0)
        S = pwp.compute_single_layer_potential_matrix_bruteforce(
            N_multipole=N_multipole, alpha=alpha
        )
        np.testing.assert_allclose(S, expected_S, atol=1e-4)

    @parameterized.expand(
        [
            (
                [np.array([0, 0, 1]), np.array([0, 0, 5])],  # centers
                np.array([1, 1, 1, 1]),  # radii
                10,  # L
                1e-5,  # k0
                1,  # N_multipole
            )
        ]
    )
    def test_properties_2x2_capacitance(self, centers, radii, L, k0, N_multipole):
        """
        Tests for properties of Lemma 6.3.1. in E.O. Hiltunen Doctoral Thesis
        """
        pwp = ClassicPeriodicFWP3D(centers=centers, radii=radii, L=L, k0=k0)
        alphas = np.linspace(-np.pi / L, np.pi / L, 100)
        for alpha in alphas:
            C = pwp.get_capacitance_matrix(alpha=alpha, N_multipole=N_multipole)
            self.assertLess(np.abs(C[0, 0].imag), 1e-5)
            self.assertAlmostEqual(C[0, 0], C[1, 1], places=5)
            self.assertAlmostEqual(C[0, 1], np.conj(C[1, 0]), places=5)
