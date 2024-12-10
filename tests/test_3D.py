import unittest
import numpy as np
from parameterized import parameterized

from Subwavelength3D.classic import ClassicFiniteFWP3D, flat_index


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
