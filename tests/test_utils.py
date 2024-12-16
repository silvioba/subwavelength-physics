import unittest
import numpy as np
from parameterized import parameterized

from Subwavelength1D.nonreciprocal import NonReciprocalFiniteSWP1D, NonReciprocalPeriodicSWP1D
from Utils.utils_general import EigenvectorPathTracker, sort_by_method


class UtilsTests(unittest.TestCase):

    @parameterized.expand([
        ([1, 2, 3], [2, 3], [3, 2, 1]),
        ([2, 2, 4], [1, 1], [-1, -1, -2])
    ])
    def test_eigenvalue_path_tracker_identical(self, ls, ss, gg):
        nrp = NonReciprocalFiniteSWP1D(
            N=len(ls), l=ls, s=ss, gammas=gg, v_in=1, v_out=1)
        ept = EigenvectorPathTracker()
        D, S = nrp.get_sorted_eigs_capacitance_matrix()
        D, S = ept.next(D, S)
        D2, S2 = nrp.get_sorted_eigs_capacitance_matrix(sorting="eva_imag")
        D2, S2 = ept.next(D2, S2)
        np.testing.assert_allclose(
            D,
            D2,
            atol=1e-4
        )

    @parameterized.expand([
        ([1, 2, 3], [2, 3], [3, 2, 1]),
        ([2, 2, 4], [1, 1], [-1, -1, -2])
    ])
    def test_eigenvalue_path_tracker_small_perturb(self, ls, ss, gg):
        nrp = NonReciprocalFiniteSWP1D(
            N=len(ls), l=ls, s=ss, gammas=gg, v_in=1, v_out=1)
        nrp2 = NonReciprocalFiniteSWP1D(
            N=len(ls), gammas=np.array(gg)+1e-4, l=ls, s=ss, v_in=1, v_out=1)
        ept = EigenvectorPathTracker()
        D, S = nrp.get_sorted_eigs_capacitance_matrix()
        D, S = ept.next(D, S)
        D2, S2 = nrp2.get_sorted_eigs_capacitance_matrix("eva_imag")
        D2, S2 = ept.next(D2, S2)
        np.testing.assert_allclose(
            D,
            D2,
            atol=1e-4
        )
