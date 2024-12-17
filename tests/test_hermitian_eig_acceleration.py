import unittest
import numpy as np
from parameterized import parameterized

from Subwavelength1D.nonreciprocal import NonReciprocalFiniteSWP1D, NonReciprocalPeriodicSWP1D
from Subwavelength1D.classic import ClassicFiniteSWP1D, ClassicPeriodicSWP1D

from Utils.utils_general import unique_eigenvector_phases


class HermitianEigAccelerationTests(unittest.TestCase):

    @parameterized.expand([
        ([1, 2, 3], [2, 3], [1, 2, 1]),
        ([2, 2, 4], [1, 1], [0.5, 10, 2]),
        ([1, 2, 3], [2, 3], [1, 2, 1]),
        ([2, 2, 4], [1, 1], [0.5, 10, 2]),
    ])
    def test_material_matrix(self, ls, ss, v_in):
        cp = ClassicFiniteSWP1D(N=len(ls), l=ls, s=ss, v_in=v_in, v_out=1)
        V = np.array([
            [v_in[0]**2/ls[0], 0, 0],
            [0, v_in[1]**2/ls[1], 0],
            [0, 0, v_in[2]**2/ls[2]]
        ])
        V_inv = np.linalg.inv(V)
        V_sqrt = np.sqrt(V)
        V_sqrt_inv = np.linalg.inv(V_sqrt)

        np.testing.assert_allclose(
            cp.get_material_matrix(),
            V
        )
        np.testing.assert_allclose(
            cp.get_material_matrix(inverted=True),
            V_inv
        )
        np.testing.assert_allclose(
            cp.get_material_matrix(perform_sqrt=True),
            V_sqrt
        )
        np.testing.assert_allclose(
            cp.get_material_matrix(inverted=True, perform_sqrt=True),
            V_sqrt_inv
        )

    @parameterized.expand([
        ([1, 2, 3], [2, 3], [1, 2, 1], False),
        ([2, 2, 4], [1, 1], [0.5, 10, 2], False),
        ([1, 2, 3], [2, 3], [1, 2, 1], True),
        ([2, 2, 4], [1, 1], [0.5, 10, 2], True),
    ])
    def test_finite_hermitian_acceleration(self, ls, ss, v_in, generalized):
        cp = ClassicFiniteSWP1D(
            N=len(ls), l=ls, s=ss, v_in=v_in, v_out=1)

        D1, S1 = cp.get_sorted_eigs_capacitance_matrix(
            hermitian_acceleration=False, generalised=generalized)
        D2, S2 = cp.get_sorted_eigs_capacitance_matrix(
            hermitian_acceleration=True, generalised=generalized)

        S1 = unique_eigenvector_phases(S1)
        S2 = unique_eigenvector_phases(S2)
        np.testing.assert_allclose(
            D1,
            D2,
            atol=1e-4
        )
        np.testing.assert_allclose(
            S1,
            S2,
            atol=1e-4
        )

    @parameterized.expand([
        ([1, 2, 3], [5, 2, 3], [1, 2, 1], 0.1, False),
        ([1, 2, 3], [5, 2, 3], [1, 2, 1], 2, False),
        ([2, 2, 4], [1, 1, 7], [0.5, 10, 2], 0.3, False),
        ([2, 2, 4], [1, 1, 7], [0.5, 10, 2], -0.5, False),
        ([1, 2, 3], [5, 2, 3], [1, 2, 1], 0.1, True),
        ([1, 2, 3], [5, 2, 3], [1, 2, 1], 2, True),
        ([2, 2, 4], [1, 1, 7], [0.5, 10, 2], 0.3, True),
        ([2, 2, 4], [1, 1, 7], [0.5, 10, 2], -0.5, True),
    ])
    def test_periodic_hermitian_acceleration(self, ls, ss, v_in, alpha, generalized):
        cp = ClassicPeriodicSWP1D(
            N=len(ls), l=ls, s=ss, v_in=v_in, v_out=1)

        D1, S1 = cp.get_sorted_eigs_capacitance_matrix(
            hermitian_acceleration=False, generalised=generalized)(alpha)
        D2, S2 = cp.get_sorted_eigs_capacitance_matrix(
            hermitian_acceleration=True, generalised=generalized)(alpha)
        S1 = unique_eigenvector_phases(S1)
        S2 = unique_eigenvector_phases(S2)
        np.testing.assert_allclose(
            D1,
            D2,
            atol=1e-4
        )
        np.testing.assert_allclose(
            S1,
            S2,
            atol=1e-4
        )
