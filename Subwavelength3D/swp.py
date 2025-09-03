import numpy as np

import copy
from typing import Literal, Callable, Tuple, Self, List


def _get_consistent_parameter(param, N):
    if (
        isinstance(param, float)
        or isinstance(param, int)
        or isinstance(param, complex)
    ):
        return (
            np.ones(N, dtype=complex if isinstance(
                param, complex) else float) * param
        )
    else:
        return np.array(param, dtype=float).reshape((N,))  # (N,)


class SWP3D:
    """
    Base class for a three-dimensional subwavelength problem
    """

    def __init__(
        self,
        centers: List | np.ndarray,
        radii: np.ndarray | List | float | int | complex = 1,
        v_in: np.ndarray | List | float | int | complex = 1,
        cache_capacitance_matrix: bool = True,
    ):
        self.centers = np.array(centers).reshape((-1, 3))  # (N,3)
        self.N = len(centers)

        self.radii = _get_consistent_parameter(radii, self.N)  # (N,)
        self.v_in = _get_consistent_parameter(v_in, self.N)  # (N,)

        self.cache_capacitance_matrix = cache_capacitance_matrix
        self._capacitance_matrix = None
        self._capacitance_matrix_parameters = None

    def __str__(self):
        return f"Three Dimensional Finite system with {self.N} resonators.\nGeometry:     The first centers are {self.centers[:5]} and the first radii are {self.radii[:5]}."

    def __repr__(self):
        return self.__str__()

    def get_material_matrix(self, inverted=False, perform_sqrt=False, return_only_list=False) -> np.ndarray:
        """
        Get the material matrix such that VCu = lambda u is a solution to the subwavelength problem.

        Returns:
            np.ndarray: The material matrix.
        """
        diag = (
            np.power(self.v_in, 2) / (4/3*np.pi*np.power(self.radii, 3))
        ) if not perform_sqrt else (
            self.v_in/np.sqrt(4/3*np.pi*np.power(self.radii, 3))
        )
        if inverted:
            diag = 1/diag
        if return_only_list:
            return diag
        else:
            return np.diag(diag)
