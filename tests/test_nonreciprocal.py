import unittest
import numpy as np
from parameterized import parameterized

from Subwavelength1D.nonreciprocal import NonReciprocalFiniteSWP1D, NonReciprocalPeriodicSWP1D
from Subwavelength1D.classic import ClassicFiniteSWP1D, ClassicPeriodicSWP1D


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
                cp.get_generalised_capacitance_matrix()(alpha),
                nrp.get_generalised_capacitance_matrix()(alpha),
                atol=1e-4
            )
        else:
            np.testing.assert_allclose(
                cp.get_capacitance_matrix()(alpha),
                nrp.get_capacitance_matrix()(alpha),
                atol=1e-4
            )
