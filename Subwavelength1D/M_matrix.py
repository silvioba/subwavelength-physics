import numpy as np
import scipy as sci
from Subwavelength1D.swp import (
    FiniteSWP1D,
    PeriodicSWP1D,
)

import Utils.utils_general as utils

from typing import Literal, Callable, Tuple, Self, List, override

import copy


# This file contains the M-matrix classes which prentend to be subwavelength systems in order
# to extend all the logic from SWP1D to M-matrix systems.

class FiniteBandedMMatrix(FiniteSWP1D):
    def __init__(self, diagonals: List[np.ndarray]):
        self.diagonals = diagonals
        self.N = len(diagonals[0])
        self.L = 1

    @override
    def get_capacitance_matrix(self) -> np.ndarray:
        M = np.diag(self.diagonals[0])
        for i in range(1, len(self.diagonals)):
            M += np.diag(self.diagonals[i][:-i], i) + \
                np.diag(self.diagonals[i][:-i], -i)
        return M

    def get_generalised_capacitance_matrix(self) -> np.ndarray:
        return self.get_capacitance_matrix()

    @override
    def get_periodized_system(self, sN=None):
        assert sN is None
        return PeriodicBandedMMatrix(self.diagonals)

    @override
    def compute_sorted_eigs_capacitance_matrix(
        self,
        eigenvalues_only=False,
        generalised=True,
        sorting: Literal[
            "eve_middle_localization",
            "eve_localization",
            "eva_real",
            "eva_imag",
            "eve_abs",
            "eva_first_val",
        ] = "eva_real",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Assumes that the M-matrix is Hermitian.
        """
        C = self.get_generalised_capacitance_matrix()
        if eigenvalues_only:
            D = np.linalg.eigvalsh(C)
            S = None
        else:
            D, S = np.linalg.eigh(C)

        D, S = utils.sort_by_method(D, S, sorting)
        return D, S


class PeriodicBandedMMatrix(PeriodicSWP1D):
    def __init__(self, diagonals: List[np.ndarray]):
        self.diagonals = diagonals
        self.N = len(diagonals[0])
        self.L = 1

    @classmethod
    def from_finite(cls, finite: FiniteSWP1D) -> Self:
        return cls(copy.deepcopy(finite.diagonals))

    @override
    def get_capacitance_matrix(self) -> Callable[[float], np.ndarray]:
        def C(alpha):
            M = np.diag(np.array(self.diagonals[0], dtype=complex))
            n = M.shape[0]
            for i in range(1, len(self.diagonals)):
                M += np.diag(self.diagonals[i][:-i], i) + \
                    np.diag(self.diagonals[i][:-i], -i)
                qp = np.exp(-1j*alpha)*np.diag(self.diagonals[i][-i:], n-i)
                M += qp + np.conj(qp).T
            return M
        return C

    @override
    def compute_generalised_capacitance_matrix(self) -> Callable[[float], np.ndarray]:
        return self.get_capacitance_matrix()

    @override
    def compute_sorted_eigs_capacitance_matrix(
        self,
        eigenvals_only=False,
        generalised=True,
        sorting: Literal[
            "eve_middle_localization",
            "eve_localization",
            "eva_real",
            "eva_imag",
            "eve_abs",
            "eva_first_val",
        ] = "eva_real",
    ) -> Callable[[float], Tuple[np.ndarray, np.ndarray]]:
        def f(alpha):
            C = self.get_capacitance_matrix()(alpha)
            if eigenvals_only:
                D = np.linalg.eigvalsh(C)
                S = None
            else:
                D, S = np.linalg.eigh(C)

            D, S = utils.sort_by_method(D, S, sorting)
            return D, S
        return f
