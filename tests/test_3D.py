import unittest
import numpy as np
from parameterized import parameterized

from Subwavelength3D.classic import (
    ClassicFiniteFWP3D,
    ClassicPeriodicFWP3D,
    flat_index,
    lattice_sums,
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
                0,  # n
                10,  # L
                1e-5,  # k
                -2.8209e-01 + 3.6152e03j,  # old_code_result
            ),
            (
                0.1735,  # alpha
                0,  # n
                13,  # L
                1e-5,  # k
                -2.8209e-01 + 2.5676e03j,  # old_code_result
            ),
            (
                0.1735,  # alpha
                1,  # n
                13,  # L
                1e-5,  # k
                1e7 * (3.3804 + 0.0000j),  # old_code_result
            ),
        ]
    )
    def test_lattice_sums(self, alpha, n, L, k, old_code_result):
        res = lattice_sums(alpha=alpha, n=n, L=L, k=k)
        if old_code_result.imag != 0:
            self.assertLess(
                np.abs((old_code_result.imag - res.imag) / old_code_result.imag),
                1e-4,
                f"Relative error of imaginary part is too large: got: {res.imag}, expected: {old_code_result.imag}",
            )
        else:
            self.assertLess(
                np.abs((old_code_result.imag - res.imag)),
                1e-4,
                f"Relative error of imaginary part is too large: got: {res.imag}, expected: {old_code_result.imag}",
            )
        self.assertLess(
            np.abs((old_code_result.real - res.real) / old_code_result.real),
            1e-4,
            f"Relative error of imaginary part is too large: got: {res.imag}, expected: {old_code_result.imag}",
        )


class QuasiPeriodicCapacitanceMatrix(unittest.TestCase):

    @parameterized.expand(
        [
            (
                [np.array([0, 0, 1]), np.array([0, 0, 5])],  # centers
                np.array([1, 1, 1, 1]),  # radii
                10,  # L
                1e-5,  # k0
                2,  # N_multipole
                0.0706,  # alpha
                np.array(
                    [
                        [
                            -0.9660 - 0.0000j,
                            -0.0000 - 0.0002j,
                            -0.0660 + 0.0027j,
                            -0.0058 - 0.0002j,
                        ],
                        [
                            0.0000 + 0.0002j,
                            -0.3333 + 0.0000j,
                            0.0058 + 0.0002j,
                            -0.0000 + 0.0000j,
                        ],
                        [
                            -0.0660 - 0.0027j,
                            0.0058 - 0.0002j,
                            -0.9660 - 0.0000j,
                            -0.0000 - 0.0002j,
                        ],
                        [
                            -0.0058 + 0.0002j,
                            -0.0000 - 0.0000j,
                            0.0000 + 0.0002j,
                            -0.3333 + 0.0000j,
                        ],
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
        """Tests for properties of Lemma 6.3.1. in E.O. Hiltunen Doctoral Thesis"""
        pwp = ClassicPeriodicFWP3D(centers=centers, radii=radii, L=L, k0=k0)
        alphas = np.linspace(-np.pi / L, np.pi / L, 100)
        for alpha in alphas:
            C = pwp.get_capacitance_matrix(alpha=alpha, N_multipole=N_multipole)
            self.assertLess(np.abs(C[0, 0].imag), 1e-5)
            self.assertAlmostEqual(C[0, 0], C[1, 1], places=5)
            self.assertAlmostEqual(C[0, 1], np.conj(C[1, 0]), places=5)
