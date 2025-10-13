import unittest
import numpy as np
from parameterized import parameterized

from Subwavelength3D.classic_finite import (
    ClassicFiniteSWP3D,
    flat_index,
)
import Subwavelength3D.fmm as fmm

import Utils.utils_general as utils


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


class ColinearCapacitanceMatrix(unittest.TestCase):

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
                3,
            ],
            [
                [
                    np.array([0, 0, 1]),
                    np.array([3, 0, 5]),
                    np.array([-5, 0, 9]),
                    np.array([0, 1, 20]),
                ],
                np.array([1, 1.2, 1.1, 1]),
                1,
            ],
            [
                [
                    np.array([0, 0, 1]),
                    np.array([3, 0, 5]),
                    np.array([-5, 0, 9]),
                    np.array([0, 1, 20]),
                ],
                np.array([1, 1.2, 1.1, 1]),
                3,
            ],
        ],
    )
    def test_diagonally_dominant(self, centers, radii, N_multipole):
        """
        Checks that the capacitance matrix is diagonally dominant
        """
        C = ClassicFiniteSWP3D(
            centers=centers, radii=radii
        ).get_capacitance_matrix(N_multipole=N_multipole, method='general')
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
                3,
            ],
            [
                [
                    np.array([0, 0, 1]),
                    np.array([3, 0, 5]),
                    np.array([-5, 0, 9]),
                    np.array([0, 1, 20]),
                ],
                np.array([1, 1.2, 1.1, 1]),
                1,
            ],
            [
                [
                    np.array([0, 0, 1]),
                    np.array([3, 0, 5]),
                    np.array([-5, 0, 9]),
                    np.array([0, 1, 20]),
                ],
                np.array([1, 1.2, 1.1, 1]),
                3,
            ],
        ],
    )
    def test_hermitian(self, centers, radii, N_multipole):
        """
        Checks that the capacitance matrix is hermitian
        """
        C = ClassicFiniteSWP3D(
            centers=centers, radii=radii
        ).get_capacitance_matrix(N_multipole=N_multipole, method='general')
        np.testing.assert_allclose(
            C, np.conj(C.T),
            rtol=0, atol=1e-4, err_msg="Capacitance matrix is not Hermitian"
        )

    # DEPRECATED Accelerated colinear is no longer maintained
    # @parameterized.expand(
    #     [
    #         [
    #             [
    #                 np.array([0, 0, 1]),
    #                 np.array([0, 0, 5]),
    #                 np.array([0, 0, 9]),
    #                 np.array([0, 0, 20]),
    #             ],
    #             np.array([1, 1, 1, 1]),
    #             1e-5,
    #             1,
    #         ],
    #         [
    #             [
    #                 np.array([0, 0, 1]),
    #                 np.array([0, 0, 5]),
    #                 np.array([0, 0, 9]),
    #                 np.array([0, 0, 20]),
    #             ],
    #             np.array([1, 1, 1, 1]),
    #             1e-5,
    #             2,
    #         ],
    #     ],
    # )
    # def test_caching(self, centers, radii, k0, N_multipole):
    #     C_bruteforce = ClassicFiniteSWP3D(
    #         centers=centers, radii=radii, k0=k0
    #     ).get_capacitance_matrix(N_multipole=N_multipole, accelerated=False)
    #     C_fast = ClassicFiniteSWP3D(
    #         centers=centers, radii=radii, k0=k0
    #     ).get_capacitance_matrix(N_multipole=N_multipole, accelerated=True)
    #     np.testing.assert_allclose(
    #         C_bruteforce, C_fast,
    #         err_msg="Capacitance matrices from bruteforce and fast methods do not match",
    #     )


class ConsistencyTests(unittest.TestCase):

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
                2,
            ],
            [
                [
                    np.array([0, 0, 1]),
                    np.array([0, 0, 5]),
                    np.array([0, 0, 9]),
                    np.array([0, 0, 20]),
                ],
                np.array([1, 1, 1, 1]),
                3,
            ],
            [
                [
                    np.array([0, 0, 1]),
                    np.array([0, 0, 5]),
                    np.array([0, 0, -9]),
                    np.array([0, 0, -3]),
                ],
                np.array([1, 1, 1, 1]),
                3,
            ],
            [
                [
                    np.array([0, 0, 1]),
                    np.array([0, 0, 5]),
                    np.array([0, 0, -9]),
                    np.array([0, 0, -3]),
                ],
                np.array([0.5, 1, 2, 1]),
                1,
            ],
            [
                [
                    np.array([0, 0, 1]),
                    np.array([0, 0, 5]),
                    np.array([0, 0, -9]),
                    np.array([0, 0, -3]),
                ],
                np.array([0.5, 1, 3, 1]),
                3,
            ],
        ],
    )
    def test_colinear_and_general_consistent(self, centers, radii, N_multipole):
        """
        Checks that the capacitance matrix for a chain of resonators computed with the
        colinear simplification is the same as when this semplification is not used

        """
        dp = ClassicFiniteSWP3D(
            centers=centers, radii=radii
        )

        Ccol = dp.get_capacitance_matrix(
            N_multipole=N_multipole, method="colinear")

        Scol = dp.compute_colinear_single_layer_potential_matrix_bruteforce(
            N_multipole=N_multipole)

        Cnoncol = dp.get_capacitance_matrix(
            N_multipole=N_multipole, method="general")

        Snoncol = dp.compute_general_single_layer_potential_matrix(
            N_multipole=N_multipole)

        np.testing.assert_allclose(
            Ccol, Cnoncol,
            err_msg="Capacitance matrices are not close in the colinear case",
        )

        np.testing.assert_allclose(
            Scol, Snoncol,
            err_msg="Single layer potentials are not close in the colinear case",
        )

    @parameterized.expand(
        [
            [
                [
                    np.array([0, 0, 0]),
                    np.array([0, 0, 4]),
                ],
                [
                    np.array([0, 0, 0]),
                    np.array([0, 4, 0]),
                ],
                1,
            ],
            [
                [
                    np.array([0, 0, 0]),
                    np.array([0, 0, 4]),
                ],
                [
                    np.array([0, 0, 0]),
                    np.array([0, 4, 0]),
                ],
                2,
            ],
            [
                [
                    np.array([0, 0, 0]),
                    np.array([0, 0, 4]),
                ],
                [
                    np.array([0, 0, 0]),
                    np.array([4, 0, 0]),
                ],
                1,
            ],
            [
                [
                    np.array([0, 0, 0]),
                    np.array([0, 0, 4]),
                ],
                [
                    np.array([0, 0, 0]),
                    np.array([4, 0, 0]),
                ],
                2,
            ],
            [
                [
                    np.array([0, 0, 4]),
                    np.array([0, 4, 0]),
                    np.array([4, 0, 0]),
                ],
                [
                    np.array([4, 0, 0]),
                    np.array([0, 4, 0]),
                    np.array([0, 0, 4]),
                ],
                1,
            ],
            [
                [
                    np.array([0, 0, 4]),
                    np.array([0, 4, 0]),
                    np.array([4, 0, 0]),
                ],
                [
                    np.array([4, 0, 0]),
                    np.array([0, 4, 0]),
                    np.array([0, 0, 4]),
                ],
                2,
            ],
        ]
    )
    def test_symmetry_consistency(self, c1, c2, N_multipole):
        """
        Checks that if a system is rotated or mirrored in space the capacitance matrix remains the same
        """
        N = len(c1)
        radii = np.ones(N)
        d1 = ClassicFiniteSWP3D(
            centers=c1, radii=radii
        ).get_capacitance_matrix(N_multipole=N_multipole, method="general")

        d2 = ClassicFiniteSWP3D(
            centers=c2, radii=radii
        ).get_capacitance_matrix(N_multipole=N_multipole, method="general")

        np.testing.assert_allclose(
            d1, d2,
            err_msg="Capacitance matrices are not close in the symmetry case",
        )

    @parameterized.expand(
        [
            [
                [
                    np.array([1, 2, 3]),
                    np.array([-4, 5, -6]),
                ],
                [1, 1],
                [1, 0],
                1,
            ],
            [
                [
                    np.array([1, 2, 3]),
                    np.array([-4, 5, -6]),
                ],
                [1, 1],
                [1, 0],
                2,
            ],
            [
                [
                    np.array([1, 2, 3]),
                    np.array([-4, 5, -6]),
                ],
                [1, 2],
                [1, 0],
                1,
            ],
            [
                [
                    np.array([1, 2, 3]),
                    np.array([-4, 5, -6]),
                ],
                [1, 2],
                [1, 0],
                2,
            ],
            [
                [
                    np.array([4, 0, 0]),
                    np.array([0, 4, 0]),
                    np.array([0, 0, 4]),
                ],
                [1, 2, 0.5],
                [0, 2, 1],
                1,
            ],
            [
                [
                    np.array([4, 0, 0]),
                    np.array([0, 4, 0]),
                    np.array([0, 0, 4]),
                ],
                [1, 2, 0.5],
                [1, 2, 0],
                1,
            ],
            [
                [
                    np.array([4, 0, 0]),
                    np.array([0, 4, 0]),
                    np.array([0, 0, 4]),
                ],
                [1, 2, 0.5],
                [0, 2, 1],
                2,
            ],
            [
                [
                    np.array([4, 0, 0]),
                    np.array([0, 4, 0]),
                    np.array([0, 0, 4]),
                ],
                [1, 2, 0.5],
                [1, 2, 0],
                2,
            ],
            [
                [
                    np.array([4, 0, 0]),
                    np.array([0, 4, 0]),
                    np.array([0, 0, 4]),
                ],
                [1, 2, 0.5],
                [0, 2, 1],
                3,
            ],
            [
                [
                    np.array([4, 0, 0]),
                    np.array([0, 4, 0]),
                    np.array([0, 0, 4]),
                ],
                [1, 2, 0.5],
                [1, 2, 0],
                3,
            ],
        ]
    )
    def test_permutation_consistency(self, centers, radii, permutation, N_multipole):
        """
        Checks that if the index of some resonator is permuted then the matrix 
        associated to that permutation creates a similarity transformation between matrices
        """
        d1 = ClassicFiniteSWP3D(
            centers=centers, radii=radii
        ).get_capacitance_matrix(N_multipole=N_multipole, method="general")

        permuted_centers = np.array(centers)[permutation]
        permuted_radii = np.array(radii)[permutation]
        d2 = ClassicFiniteSWP3D(
            centers=permuted_centers, radii=permuted_radii
        ).get_capacitance_matrix(N_multipole=N_multipole, method="general")

        inverse_permutation = np.argsort(permutation)
        np.testing.assert_allclose(
            d1, d2[:, inverse_permutation][inverse_permutation, :],
            err_msg="Capacitance matrices are not close in the permutation case",
        )

    @parameterized.expand(
        [
            [
                np.array([
                    [0, 0, 1],
                    [0, 0, 5],
                    [0, 0, 9],
                    [0, 0, 20],
                ]),
                np.array([1, 1, 1, 1]),
                1,
            ],
            [
                np.array([
                    [0, 0, 1],
                    [0, 0, 5],
                    [0, 0, 9],
                    [0, 0, 20],
                ]),
                np.array([1, 1/2, 1/3, 1/4]),
                1,
            ],
            [
                np.array([
                    [0, 0.3, 1],
                    [5, 0, 5.1],
                    [0, -1, 9],
                    [0.01, 1, 20],
                ]),
                np.array([2, 1.1, 2, 0.5]),
                1,
            ],
            [
                np.array([
                    [0, 0, 1],
                    [0, 0, 5],
                    [0, 0, 9],
                    [0, 0, 20],
                ]),
                np.array([1, 1, 1, 1]),
                2,
            ],
            [
                np.array([
                    [0, 0, 1],
                    [0, 0, 5],
                    [0, 0, 9],
                    [0, 0, 20],
                ]),
                np.array([1, 1/2, 1/3, 1/4]),
                2,
            ],
            [
                np.array([
                    [0, 0.3, 1],
                    [5, 0, 5.1],
                    [0, -1, 9],
                    [0.01, 1, 20],
                ]),
                np.array([2, 1.1, 3, 0.5]),
                2,
            ],
        ]
    )
    def test_general_and_fmm_consistent(self, centers, radii, N_multipole):
        """
        Checks that the computaiton of the matrix using single layer potentials and using fmm is the same
        """
        Cnoncol = np.real(ClassicFiniteSWP3D(
            centers=centers, radii=radii
        ).get_capacitance_matrix(N_multipole=N_multipole, method="general"))

        dipole = N_multipole > 1
        Cfmm = fmm.compute_capacitance_matrix(
            centers, radii, dipole=dipole, eps=1e-3)

        np.testing.assert_allclose(
            Cnoncol, Cfmm,
            err_msg="Capacitance matrices are not close",
            atol=1e-4,
            rtol=1e-4
        )

    @parameterized.expand(
        [
            [
                np.array([
                    [0, 0, 1],
                    [0, 0, 5],
                    [0, 0, 9],
                    [0, 0, 20],
                ]),
                np.array([1, 1, 1, 1]),
                1,
            ],
            [
                np.array([
                    [0, 0, 1],
                    [0, 0, 5],
                    [0, 0, 9],
                    [0, 0, 20],
                ]),
                np.array([1, 1/2, 1/3, 1/4]),
                1,
            ],
            [
                np.array([
                    [0, 0, 1],
                    [0, 0, 5],
                    [0, 0, 9],
                    [0, 0, 20],
                ]),
                np.array([1, 1/2, 1/3, 1/4]),
                2,
            ],
            [
                np.array([
                    [0, 0.3, 1],
                    [5, 0, 5.1],
                    [0, -1, 9],
                    [0.01, 1, 20],
                ]),
                np.array([2, 1.1, 2, 0.5]),
                1,
            ],
            [
                np.array([
                    [0, 0, 1],
                    [0, 0, 5],
                    [0, 0, 9],
                    [0, 0, 20],
                ]),
                np.array([1, 1, 1, 1]),
                np.array([1, 1, 1, 1]),
            ],
            [
                np.array([
                    [0, 0, 1],
                    [0, 0, 5],
                    [0, 0, 9],
                    [0, 0, 20],
                ]),
                np.array([1, 1/2, 1/3, 1/4]),
                np.array([2, 1, 0.5, 3]),
            ],
            [
                np.array([
                    [0, 0.3, 1],
                    [5, 0, 5.1],
                    [0, -1, 9],
                    [0.01, 1, 20],
                ]),
                np.array([2, 1.1, 3, 0.5]),
                np.array([2, 1, 0.5, 3]),
            ],
        ]
    )
    def test_eigenvalue_calculation(self, centers, radii, v_in):
        N_multipole = 2

        dp3 = ClassicFiniteSWP3D(
            centers=centers, radii=radii, v_in=v_in
        )
        V = dp3.get_material_matrix()
        C = dp3.get_capacitance_matrix(
            N_multipole=N_multipole, method="general")

        D, S = utils.sort_by_method(*np.linalg.eig(V@C), method="eva_real")
        D = np.real(D)
        S = np.real(S)

        Dm, Sm = dp3.compute_sorted_eigs_capacitance_matrix(
            N_multipole=N_multipole, method="general")
        np.testing.assert_allclose(
            D, Dm,
            err_msg="Eigenvalues are not close"
        )
        np.testing.assert_allclose(
            utils.unique_eigenvector_phases(
                S), utils.unique_eigenvector_phases(Sm),
            err_msg="Eigenvectors are not close"
        )
