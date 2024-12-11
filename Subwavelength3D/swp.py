import numpy as np

import copy
from typing import Literal, Callable, Tuple, Self, List


class SWP3D:
    """
    Base class for a three-dimensional subwavelength problem
    """

    def __init__(self, centers, radii, k0):
        self.centers = centers  # List of 3d np array
        self.radii = radii  # Np array
        self.N = len(centers)
        self.k0 = k0

    def __str__(self):
        return f"Three Dimensional Finite system with {self.N} resonators.\nGeometry:     The first centers are {self.centers[:5]} and the first radii are {self.radii[:5]}."

    def __repr__(self):
        return self.__str__()
