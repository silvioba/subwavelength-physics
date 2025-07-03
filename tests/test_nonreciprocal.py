import unittest
import numpy as np
from parameterized import parameterized

from Subwavelength1D.nonreciprocal import NonReciprocalFiniteSWP1D, NonReciprocalPeriodicSWP1D
from Subwavelength1D.classic import ClassicFiniteSWP1D, ClassicPeriodicSWP1D

from Utils.utils_general import unique_eigenvector_phases


class NonReciprocalTests(unittest.TestCase):

    @parameterized.expand([
        ([1, 2, 3], [2, 3]),
        ([2, 2, 4], [1, 1])
    ])
    def test_asymptotically_valid_capacitance_finite(self, ls, ss):
        cp = ClassicFiniteSWP1D(
            N=len(ls), l=ls, s=ss, v_in=1, v_out=1)
        nrp = NonReciprocalFiniteSWP1D(
            N=len(ls), gammas=1e-8, l=ls, s=ss, v_in=1, v_out=1)
        np.testing.assert_allclose(
            cp.get_capacitance_matrix(),
            nrp.get_capacitance_matrix(),
            atol=1e-4
        )
        np.testing.assert_allclose(
            cp.get_generalised_capacitance_matrix(),
            nrp.get_generalised_capacitance_matrix(),
            atol=1e-4
        )

    @parameterized.expand([
        ([1, 2, 3], [2, 3], [0, 0, 0]),
        ([2, 2, 4], [1, 1], [0, 1e-12, 0])
    ])
    def test_valid_zero_handling_capacitance_finite(self, ls, ss, gg):
        cp = ClassicFiniteSWP1D(
            N=len(ls), l=ls, s=ss, v_in=1, v_out=1)
        nrp = NonReciprocalFiniteSWP1D(
            N=len(ls), gammas=gg, l=ls, s=ss, v_in=1, v_out=1)
        np.testing.assert_allclose(
            cp.get_capacitance_matrix(),
            nrp.get_capacitance_matrix(),
            atol=1e-4
        )
        np.testing.assert_allclose(
            cp.get_generalised_capacitance_matrix(),
            nrp.get_generalised_capacitance_matrix(),
            atol=1e-4
        )

    @parameterized.expand([
        ([1, 2, 3], [2, 3, 4], 0.1, False),
        ([1, 2, 3], [2, 3, 4], -2, False),
        ([1, 2, 3], [2, 3, 4], 0.3, True),
        ([1, 2, 3], [2, 3, 4], -1.5, True),
        ([1], [2], 0.1, False),
        ([1], [2], 0.1, True),
        ([1, 2], [2, 3], 0.2, False),
        ([1, 2], [2, 3], 0.2, True),
    ])
    def test_asymptotically_valid_capacitance_periodic(self, ls, ss, alpha, general):
        cp = ClassicPeriodicSWP1D(
            N=len(ls), l=ls, s=ss, v_in=1, v_out=1)
        nrp = NonReciprocalPeriodicSWP1D(
            N=len(ls), gammas=1e-4, l=ls, s=ss, v_in=1, v_out=1)
        if general:
            np.testing.assert_allclose(
                cp.compute_generalised_capacitance_matrix()(alpha),
                nrp.compute_generalised_capacitance_matrix()(alpha),
                atol=1e-4
            )
        else:
            np.testing.assert_allclose(
                cp.get_capacitance_matrix()(alpha),
                nrp.get_capacitance_matrix()(alpha),
                atol=1e-4
            )

    @parameterized.expand([
        ([1, 2, 3], [2, 3], [2, 1, -1], [3, 2, 1], True),
        ([1, 2, 3], [2, 3], [2, 1, -1], [1, 2, 1], True),
        ([2, 2, 4], [1, 1], [0, 1, 2], [0.5, 10, 2], True),
        ([1, 2, 3], [2, 3], [0, 1, 2], [1, 2, 1], False),
        ([2, 2, 4], [1, 1], [2, 1, -1], [0.5, 10, 2], False),
    ])
    def test_finite_nonreciprocal_acceleration(self, ls, ss, gg, v_in, generalized):
        cp = NonReciprocalFiniteSWP1D(
            N=len(ls), l=ls, s=ss, gammas=gg, v_in=v_in, v_out=1)

        D1, S1 = cp.compute_sorted_eigs_capacitance_matrix(
            real_symmetrisation_acceleration=True, generalised=generalized)
        D2, S2 = cp.compute_sorted_eigs_capacitance_matrix(
            real_symmetrisation_acceleration=False, generalised=generalized)

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
