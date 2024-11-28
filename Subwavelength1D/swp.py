import numpy as np

import copy
from typing import Literal, Callable, Tuple, Self, List

from Subwavelength1D.utils_general import *


class SWP1D:
    """
    Base class for a one-dimensional subwavelength problem
    """

    def __init__(
        self,
        l: np.ndarray | float,
        s: np.ndarray | float,
        v_in: np.ndarray | float | complex | None = None,
        v_out: float | None = None,
        delta: float | None = None,
        omega: float | complex | None = None,
        uin=lambda x: np.sin(x),
        duin=lambda x: np.cos(x),
    ):
        self.N = len(l)
        self.l = l
        self.s = s
        self.v_in = v_in
        self.v_out = v_out

        self.delta = delta
        self.omega = omega

        if self.omega:
            self.set_omega(self.omega)
        else:
            self.k_in, self.k_in = None, None

        self.uin = uin
        self.duin = duin

        self.L = np.sum(self.l) + np.sum(self.s)
        # ds is the array with the distances between the interesting points
        ds = np.zeros(len(self.l)+len(self.s))
        ds[::2] = self.l
        ds[1::2] = self.s

        self.xi = np.insert(np.cumsum(ds), 0, 0)
        # We go to self.xi[0:-1:2] to avoid the last element in the periodic case
        self.xiCol = np.column_stack((self.xi[0:-1:2], self.xi[1::2]))
        self.xim = self.xiCol[:, 0]
        self.xip = self.xiCol[:, 1]

    def set_omega(self, omega: float | complex):
        self.omega = omega
        self.k_in = self.omega / self.v_in
        self.k_out = self.omega / self.v_out

    def get_pertubed_copy(
        self,
        p: float,
        perturb_param: Literal["spacing", "sizes", "material"] = "spacing",
        p_sampling: Literal["uniform", "positive", "loguniform"] = "uniform"
    ):

        dp_perturbed = copy.deepcopy(self)
        perturb_array: np.array = None
        if perturb_param == "spacing":
            perturb_array = dp_perturbed.s
        elif perturb_param == "sizes":
            perturb_array = dp_perturbed.l
        elif perturb_param == "material":
            perturb_array = dp_perturbed.v_in

        if p_sampling == "uniform":
            perturbation = np.random.uniform(-p, p, len(perturb_array))
            perturb_array[:] += perturbation
        elif p_sampling == "positive":
            perturbation = np.random.uniform(0, p, len(perturb_array))
            perturb_array[:] += perturbation
        elif p_sampling == "loguniform":
            perturbation = np.random.uniform(10**-p, 10**p, len(perturb_array))
            perturb_array[:] *= perturbation
        return dp_perturbed


class FiniteSWP1D(SWP1D):
    """
    Base class for a one-dimensional subwavelength problem with a finite number of resonators
    """

    def __init__(
        self,
        N: int,
        l: np.ndarray | float,
        s: np.ndarray | float,
        v_in: np.ndarray | float | complex | None = None,
        v_out: float | None = None,
        delta: float | None = None,
        omega: float | complex | None = None,
        uin=lambda x: np.sin(x),
        duin=lambda x: np.cos(x),
    ):

        # Convert to array if input is given as a constant over all resonators
        if isinstance(l, float) or isinstance(l, int):
            l = np.ones(N, dtype=float) * l

        if isinstance(s, float) or isinstance(s, int):
            s = np.ones(N - 1, dtype=float) * s

        if (
            isinstance(v_in, float)
            or isinstance(v_in, int)
            or isinstance(v_in, complex)
        ):
            v_in = (
                np.ones(N, dtype=complex if isinstance(
                    v_in, complex) else float) * v_in
            )

        if (
            isinstance(v_out, float)
            or isinstance(v_out, int)
            or isinstance(v_out, complex)
        ):
            v_out = (
                np.ones(N, dtype=complex if isinstance(
                    v_out, complex) else float)
                * v_in
            )

        assert (
            len(l) == N
        ), f"The len of the l array (currently {len(self.l)}) must be equal to N={N}"
        assert (
            len(s) == N - 1
        ), f"The len of s array (currently {len(self.s)}) must be equal to N={N-1}"
        if v_in is not None:
            assert len(v_in) == N, "The l of v_in array must be equal to N"

        super().__init__(l, s, v_in, v_out, delta, omega, uin, duin)

    def __str__(self):
        return f"One Dimensional Finite system with {self.N} resonators.\nGeometry:     The first lengths are {self.l[:5]} and the first spacings are {self.s[:5]}."

    def __repr__(self):
        return self.__str__()

    def get_capacitance_matrix(self) -> np.ndarray:
        raise NotImplementedError

    def get_generalised_capacitance_matrix(self) -> np.ndarray:
        raise NotImplementedError

    def get_sorted_eigs_capacitance_matrix(
        self, generalised=True, sorting: Literal["real", "abs"] = "real"
    ) -> np.ndarray:
        if generalised:
            D, S = np.linalg.eig(self.get_generalised_capacitance_matrix())
            if sorting == "real":
                idx = np.argsort(np.real(D))
                return D[idx], S[:, idx]
            elif sorting == "abs":
                idx = np.argsort(np.abs(D))
                return D[idx], S[:, idx]
        else:
            D, S = np.linalg.eigh(self.get_capacitance_matrix())
            if sorting == "real":
                return D, S
        if sorting == "abs":
            return D[np.argsort(np.abs(D))], S[:, np.argsort(np.abs(D))]

    def plot_eigenvalues(self, generalised=True, sort=Literal["real"], colorfunc=None, ax=None):
        """
        Plots the eigenvalues of the capacitance matrix.

        Args:
            generalised (bool, optional): If True, computes the eigenvalues of get_generalised_capacitance_matrix , else get_capacitance_matrix. Defaults to True.
            sort (_type_, optional): Sorting for the eigenvalues. If generalised is False, the value is ignored and "real" is used. Defaults to Literal["real"].
        """
        if generalised:
            D, _ = np.linalg.eig(self.get_generalised_capacitance_matrix())
            D = np.sort(D)
        else:
            D, _ = np.linalg.eigh(self.get_capacitance_matrix())

        plot_eigenvalues(D, colorfunc, ax)


class PeriodicSWP1D(SWP1D):
    """
    Base class for a one-dimensional subwavel problem with an infition number of resonators with quasi periodic boundary conditons
    """

    def __init__(
        self,
        N: int,
        l: np.ndarray | float,
        s: np.ndarray | float,
        v_in: np.ndarray | float | complex = 1,
        v_out: float = 1,
        delta: float = 0.001,
        omega: float | complex | None = None,
        uin=lambda x: np.sin(x),
        duin=lambda x: np.cos(x),
    ):

        # Convert to array if input is given as a constant over all resonators
        if isinstance(l, float) or isinstance(l, int):
            l = np.ones(N, dtype=float) * l
        else:
            l = np.array(l)

        if isinstance(s, float) or isinstance(s, int):
            s = np.ones(N, dtype=float) * s
        else:
            s = np.array(s)

        if (
            isinstance(v_in, float)
            or isinstance(v_in, int)
            or isinstance(v_in, complex)
        ):
            v_in = (
                np.ones(N, dtype=complex if isinstance(
                    v_in, complex) else float) * v_in
            )

        if (
            isinstance(v_out, float)
            or isinstance(v_out, int)
            or isinstance(v_out, complex)
        ):
            v_out = (
                np.ones(N, dtype=complex if isinstance(
                    v_out, complex) else float)
                * v_in
            )

        assert len(l) == N, "The l of the l array must be equal to N"
        assert len(s) == N, "The l of s array must be equal to N-1"
        if v_in is not None:
            assert len(v_in) == N, "The l of v_in array must be equal to N"

        super().__init__(l, s, v_in, v_out, delta, omega, uin, duin)

    def __str__(self):
        return f"One Dimensional Periodic system with {self.N} resonators.\nGeometry:     The first lengths are {self.l[:5]} and the first spacings are {self.s[:5]}."

    def __repr__(self):
        return self.__str__()

    def get_capacitance_matrix(self) -> Callable[[float], np.ndarray]:
        raise NotImplementedError

    def get_generalised_capacitance_matrix(self) -> Callable[[float], np.ndarray]:
        raise NotImplementedError
