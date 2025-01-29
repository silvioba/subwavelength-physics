import numpy as np
import scipy as sci

import copy
from typing import Literal, Callable, Tuple, Self, List

import Utils.utils_general as utils

import matplotlib.pyplot as plt
import Utils.settings as settings


class SWP1D:
    """
    Base class for a one-dimensional subwavelength problem
    """

    def __init__(
        self,
        N: int,
        l: np.ndarray,
        s: np.ndarray,
        v_in: np.ndarray | None = None,
        v_out: float | None = None,
        delta: float | None = None,
        omega: float | complex | None = None,
        uin=lambda x: np.sin(x),
        duin=lambda x: np.cos(x),
    ):
        """
        Initializes the parameters for the subwavelength physics model.

        Args:
            N (int): Number of resonators.
            l (np.ndarray | float): Array or float representing the lengths.
            s (np.ndarray | float): Array or float representing the spacings.
            v_in (np.ndarray | float | complex | None, optional): Input velocity. Defaults to None.
            v_out (float | None, optional): Output velocity. Defaults to None.
            delta (float | None, optional): Delta parameter. Defaults to None.
            omega (float | complex | None, optional): Omega parameter. Defaults to None.
            uin (callable, optional): Function for initial condition. Defaults to lambda x: np.sin(x).
            duin (callable, optional): Function for derivative of initial condition. Defaults to lambda x: np.cos(x).

        Attributes:
            N (int): Number of elements in l.
            l (np.ndarray | float): Lengths.
            s (np.ndarray | float): Spacings.
            v_in (np.ndarray | float | complex | None): Input velocity.
            v_out (float | None): Output velocity.
            delta (float | None): Delta parameter.
            omega (float | complex | None): Omega parameter.
            k_in (float | None): Wave number for input.
            uin (callable): Function for initial condition.
            duin (callable): Function for derivative of initial condition.
            L (float): Total length including spacings.
            xi (np.ndarray): Cumulative sum of lengths and spacings.
            xiCol (np.ndarray): Column stack of xi for interesting points.
            xim (np.ndarray): Start points of interesting intervals.
            xip (np.ndarray): End points of interesting intervals.
        """
        self.N = N
        self.l = l
        self.s = s
        assert len(l) == N, "The l of the l array must be equal to N"
        self.set_geometry(l, s)

        self.v_in = v_in
        self.v_out = v_out

        self.delta = delta

        self.omega = None
        self.k_in = None
        self.k_out = None

        if omega:
            self.set_omega(omega)
        else:
            self.k_in, self.k_in = None, None

        self.uin = uin
        self.duin = duin

    def set_geometry(self, l: np.ndarray | float, s: np.ndarray | float):
        self.l = l.astype(float)
        self.s = s.astype(float)

        self.L = np.sum(self.l) + np.sum(self.s)
        # ds is the array with the distances between the interesting points
        ds = np.zeros(len(self.l) + len(self.s))
        ds[::2] = self.l
        ds[1::2] = self.s

        self.xi = np.insert(np.cumsum(ds), 0, 0)
        # We go to self.xi[0:-1:2] to avoid the last element in the periodic case
        self.xiCol = np.column_stack((self.xi[0:-1:2], self.xi[1::2]))
        self.xim = self.xiCol[:, 0]
        self.xip = self.xiCol[:, 1]

    def set_omega(self, omega: float | complex):
        """
        Set the angular frequency (omega) and update the corresponding wave numbers.

        Args:
            omega (float | complex): The angular frequency to set. It can be a real or complex number.

        Attributes:
            omega (float | complex): The angular frequency.
            k_in (float | complex): The wave number inside the medium, calculated as omega divided by the velocity inside the medium (v_in).
            k_out (float | complex): The wave number outside the medium, calculated as omega divided by the velocity outside the medium (v_out).
        """

        self.omega = omega
        self.k_in = self.omega / self.v_in
        self.k_out = self.omega / self.v_out

    def get_material_matrix(self, inverted=False, perform_sqrt=False, return_only_list=False) -> np.ndarray:
        """
        Get the material matrix such that :math:`VCu = \lambda u` is a solution to the subwavelength problem.

        Returns:
            np.ndarray: The material matrix.
        """
        diag = (
            np.power(self.v_in, 2) / self.l
        ) if not perform_sqrt else (
            self.v_in/np.sqrt(self.l)
        )
        if inverted:
            diag = 1/diag
        if return_only_list:
            return diag
        else:
            return np.diag(diag)

    def get_pertubed_copy(
        self,
        p: float,
        perturb_param: Literal["spacing", "sizes", "material"] = "spacing",
        p_sampling: Literal["uniform", "positive", "loguniform"] = "uniform",
    ):
        """
        Generate a perturbed copy of the current object.

        Args:
            p (float): The perturbation magnitude.
            perturb_param (Literal["spacing", "sizes", "material"], optional):
            The parameter to perturb. Defaults to "spacing".
            p_sampling (Literal["uniform", "positive", "loguniform"], optional):
            The sampling method for perturbation. Defaults to "uniform".

        Returns:
            dp_perturbed: A deep copy of the current object with the specified perturbations applied.
        """

        dp_perturbed = copy.deepcopy(self)
        perturb_array: np.array = None
        if perturb_param == "spacing":
            perturb_array = dp_perturbed.s.copy()
        elif perturb_param == "sizes":
            perturb_array = dp_perturbed.l.copy()
        elif perturb_param == "material":
            perturb_array = dp_perturbed.v_in.copy()

        if p_sampling == "uniform":
            perturbation = np.random.uniform(-p, p, len(perturb_array))
            perturb_array += perturbation
        elif p_sampling == "positive":
            perturbation = np.random.uniform(0, p, len(perturb_array))
            perturb_array += perturbation
        elif p_sampling == "loguniform":
            perturbation = np.random.uniform(10**-p, 10**p, len(perturb_array))
            perturb_array *= perturbation

        if perturb_param == "spacing":
            dp_perturbed.set_geometry(dp_perturbed.l, perturb_array)
        elif perturb_param == "sizes":
            dp_perturbed.set_geometry(perturb_array, dp_perturbed.s)
        elif perturb_param == "material":
            dp_perturbed.v_in = perturb_array

        return dp_perturbed

    def plot_geometry(self):
        fig, ax = plt.subplots()
        for xs in self.xiCol:
            ax.plot(xs, np.ones_like(xs), c="k")
        return fig, ax


class FiniteSWP1D(SWP1D):
    """
        A class representing a one-dimensional finite system with subwavelength physics.
        N (int): Number of resonators.
        l (np.ndarray | float): Lengths of the resonators. If a float is provided, it is assumed to be constant for all resonators.
        s (np.ndarray | float): Spacings between the resonators. If a float is provided, it is assumed to be constant for all spacings.
        v_in (np.ndarray | float | complex | None, optional): Input voltages. If a float or complex is provided, it is assumed to be constant for all inputs. Defaults to None.
        v_out (float | None, optional): Output voltage. If a float is provided, it is assumed to be constant for all outputs. Defaults to None.
        delta (float | None, optional): Delta parameter. Defaults to None.
        omega (float | complex | None, optional): Omega parameter. Defaults to None.
        uin (callable, optional): Function for the input voltage. Defaults to lambda x: np.sin(x).
        duin (callable, optional): Function for the derivative of the input voltage. Defaults to lambda x: np.cos(x).
    Attributes:
        N (int): Number of resonators.
        l (np.ndarray): Lengths of the resonators.
        s (np.ndarray): Spacings between the resonators.
        v_in (np.ndarray | None): Input voltages.
        v_out (np.ndarray | None): Output voltage.
        delta (float | None): Delta parameter.
        omega (float | complex | None): Omega parameter.
        uin (callable): Function for the input voltage.
        duin (callable): Function for the derivative of the input voltage.
    Methods:
        __str__(): Returns a string representation of the object.
        __repr__(): Returns a string representation of the object.
        get_capacitance_matrix() -> np.ndarray: Abstract method to get the capacitance matrix.
        get_generalised_capacitance_matrix() -> np.ndarray: Abstract method to get the generalised capacitance matrix.
        get_sorted_eigs_capacitance_matrix(generalised=True, sorting: Literal["real", "abs"] = "real") -> np.ndarray: Returns the sorted eigenvalues and eigenvectors of the capacitance matrix.
        plot_eigenvalues(generalised=True, sort=Literal["real"], colorfunc=None, ax=None): Plots the eigenvalues of the capacitance matrix.
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
        if isinstance(l, list):
            l = np.array(l)

        if isinstance(s, float) or isinstance(s, int):
            s = np.ones(N - 1, dtype=float) * s
        if isinstance(s, list):
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

        assert (
            len(l) == N
        ), f"The len of the l array (currently {len(self.l)}) must be equal to N={N}"
        assert (
            len(s) == N - 1
        ), f"The len of s array (currently {len(self.s)}) must be equal to N-1={N-1}"
        if v_in is not None:
            assert len(v_in) == N, "The l of v_in array must be equal to N"

        super().__init__(N, l, s, v_in, v_out, delta, omega, uin, duin)

    def __str__(self):
        return f"One Dimensional Finite system with {self.N} resonators.\nGeometry:     The first lengths are {self.l[:5]} and the first spacings are {self.s[:5]}."

    def __repr__(self):
        return self.__str__()

    def get_capacitance_matrix(self) -> np.ndarray:
        raise NotImplementedError

    def get_generalised_capacitance_matrix(self) -> np.ndarray:
        raise NotImplementedError

    def get_greens_matrix(self, k: int | float) -> np.ndarray:
        return np.linalg.inv(
            self.get_generalised_capacitance_matrix() - k * np.eye(self.N)
        )

    def get_sorted_eigs_capacitance_matrix(
        self,
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
        raise NotImplementedError

    def plot_eigenvalues(
        self,
        generalised=True,
        real=True,
        colorfunc=None,
        ax=None,
    ):
        """
        Plots the eigenvalues of the capacitance matrix.

        Args:
            generalised (bool, optional): If True, computes the eigenvalues of get_generalised_capacitance_matrix , else get_capacitance_matrix. Defaults to True.
            sorting (_type_, optional): Sorting for the eigenvalues. If generalised is False, the value is ignored and "real" is used. Defaults to Literal["real"].
        """
        if generalised:
            D, _ = np.linalg.eig(self.get_generalised_capacitance_matrix())
            D = np.sort(D)
        else:
            D, _ = np.linalg.eigh(self.get_capacitance_matrix())

        return utils.plot_eigenvalues(D, colorfunc, real, ax)


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

        super().__init__(
            N=N,
            l=l,
            s=s,
            v_in=v_in,
            v_out=v_out,
            delta=delta,
            omega=omega,
            uin=uin,
            duin=duin,
        )

    def __str__(self):
        return f"One Dimensional Periodic system with {self.N} resonators.\nGeometry:     The first lengths are {self.l[:5]} and the first spacings are {self.s[:5]}."

    def __repr__(self):
        return self.__str__()

    def get_capacitance_matrix(self) -> Callable[[float], np.ndarray]:
        raise NotImplementedError

    def get_generalised_capacitance_matrix(self) -> Callable[[float], np.ndarray]:
        raise NotImplementedError

    def get_sorted_eigs_capacitance_matrix(
        self,
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
        raise NotImplementedError

    def get_band_data(
        self, generalised=True, nalpha=100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the band data of the capacitance matrix

        Args:
            generalised (bool, optional): doesnt work yet. Defaults to False.
            nalpha (int, optional): number of samples in the first BZ. Defaults to 100.

        Returns:
            np.ndarray: np.linspace(-np.pi, np.pi, nalpha)
            np.ndarray: (nalpha, self.N) array with band data
        """
        alphas = np.linspace(-np.pi, np.pi, nalpha)

        C = (
            self.get_generalised_capacitance_matrix()
            if generalised
            else self.get_capacitance_matrix()
        )

        bands = np.zeros((nalpha, self.N), dtype=complex)
        for i, alpha in enumerate(alphas):
            D, S = utils.sort_by_eva_real(*np.linalg.eig(C(alpha)))
            bands[i, :] = D

        return alphas, bands
