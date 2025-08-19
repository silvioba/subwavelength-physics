import numpy as np

import copy
from typing import Literal, Callable, Tuple, Self, List


class SWP3D:
    """
    Base class for a three-dimensional subwavelength problem
    """

    def __init__(
        self,
        centers: List | np.ndarray,
        radii: List | np.ndarray,
        v_in: np.ndarray | float | int | complex = 1
    ):
        self.centers = np.array(centers).reshape((-1, 3))  # (N,3)
        self.N = len(centers)
        self.radii = np.array(radii).reshape((self.N,))  # (N,)
        self.v_in = v_in

        if (
            isinstance(v_in, float)
            or isinstance(v_in, int)
            or isinstance(v_in, complex)
        ):
            v_in = (
                np.ones(self.N, dtype=complex if isinstance(
                    v_in, complex) else float) * v_in
            )

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
