import numpy as np
import scipy as sci

from mpmath import polylog
from mpmath import mp
from Subwavelength3D.swp import SWP3D
import Subwavelength3D.fmm as fmm
from Utils.settings import settings
import Utils.utils_general as utils

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap, LogNorm

from typing import Literal, Callable, Tuple, Self, List, Dict
from typing_extensions import override

from scipy.special import spherical_jn, hankel1, sph_harm
from sympy.physics.wigner import wigner_3j
from scipy.linalg import block_diag

from math import factorial

from joblib import Parallel, delayed  # For parallelism

from functools import cache


def spherical_hl(n, z):
    return np.sqrt(np.pi / z / 2) * hankel1(n + 1 / 2, z)


def flat_index(n: int, L: int, l: int, m: int) -> int:
    """
    Computes the index in the matrix S (of size N * L**2) of the basis element Y_l^m in the block n

    Args:
        n (int): Resonator index, zero based
        L (int): total number of multipoles
        l (int): l of Y_l^m, starts with 0
        m (int): m of Y_l^m starts with -l

    Returns:
        int: index equal to n * L**2 + l**2 + (l + m)
    """
    if l < 0 or n < 0 or L < 0:
        raise ValueError(
            f"n, L, l must be non-negative integers, you provided n={n}, L={L}, l={l}"
        )
    if np.abs(m) > l:
        raise ValueError(f"m must be -l <= m <= l, you provided m={m}, l={l}")
    if l > L:
        raise ValueError(f"l must be l <= L, you provided l={l}, L={L}")

    # We identify the base function Y^l_m by e_i with i = l**2 + (l+m)
    # In the previous blocks there are: n*\sum_{l=0}^{L-1+}\sum_{m=-l}^{l} 1
    from_other_blocks = n * L**2
    # In the current block before the index l,m there are: \sum_{ll=0}^{l-1}\sum_{mm=-l}^{l} 1 + \sum_{mm=-l}^{m} 1
    current_block = l**2 + (l + m)
    return from_other_blocks + current_block


def get_mask_block(N: int, N_multi: int, index: int) -> np.ndarray:
    """
    Returns a mask corresponding to chi_index

    Args:
        N (int): Total number of resonators
        N_multi (int): Number of multipoles used
        index (int): index of the resonator that we are interested in

    Returns:
        np.ndarray: array with zeros except of N_multi**2 elements corresponding to the resonator <index>
    """
    idx = np.zeros(N * N_multi**2)
    idx[N_multi**2 * index: N_multi**2 * (index + 1)] = 1

    return idx


def get_indicator_function_spherical_harmonics_expansion(N: int, N_multi: int, index: int) -> np.ndarray:
    idx = np.zeros(N * N_multi**2)
    idx[N_multi**2 * index] = np.sqrt(4 * np.pi)

    return idx


def estimate_time():
    pass


def cartesian_to_spherical(
    cartesian_coords: np.ndarray
) -> Tuple[float, float, float]:
    """
    Converts Cartesian coordinates to spherical coordinates.

    Args:
        cartesian_coords (np.ndarray): Cartesian coordinates (x, y, z)

    Returns:
        Tuple[float, float, float]: Spherical coordinates (r, theta, phi)
    """
    x, y, z = cartesian_coords
    r = np.sqrt(x**2 + y**2 + z**2)
    # Polar angle
    phi = np.arccos(z / r) if r != 0 else 0
    # Azimuthal angle
    theta = np.arctan2(y, x) if r != 0 else 0
    return r, theta, phi


def spherical_to_cartesian(
    r: float, theta: float, phi: float
) -> np.ndarray:
    """
    Converts spherical coordinates to Cartesian coordinates.

    Args:
        r (float): Radius
        theta (float): Polar angle (colatitude)
        phi (float): Azimuthal angle

    Returns:
        np.ndarray: Cartesian coordinates (x, y, z)
    """
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z])


@cache
def C_coefficient(l: int, m: int, lp: int, mp: int, lam: int, mu: int) -> float:
    """
    C coefficent used for the addition theorem as presented on page 42 of [3]

    Returns:
        float
    """
    return complex(
        (1j) ** (lp - l + lam)
        * (-1.0) ** m
        * np.sqrt(4 * np.pi * (2 * l + 1) * (2 * lp + 1) * (2 * lam + 1))
        * wigner_3j(l, lp, lam, 0, 0, 0)
        * wigner_3j(l, lp, lam, -m, mp, mu)
    )


@cache
def A_coefficient_colinear(l: int, m: int, lp: int, mp: int, k: float, rb: float, max_lam: int):
    A = 0
    for lam in range(max_lam):
        A += np.sqrt((2*lam+1)/(4*np.pi)) * C_coefficient(l,
                                                          m, lp, mp, lam, 0) * spherical_hl(lam, k*rb)
    return A


@cache
def A_coefficient_general(l: int, m: int, lp: int, mp: int, k: float, xb: float, max_lam: int):
    rb, thetab, phib = cartesian_to_spherical(xb)
    A = 0
    for lam in range(max_lam):
        for mu in range(-lam, lam + 1):
            A += C_coefficient(l,
                               m, lp, mp, lam, mu) * spherical_hl(lam, k*rb)*sph_harm(mu, lam, thetab, phib)
    # assert np.allclose(A, A_coefficient_colinear(l, m, lp, mp, k, rb, max_lam))
    return A


@cache
def S_coefficient_offdiagonal(l: int, m: int, lp: int, mp: int, k: float, xb: float, Ri: float, Rj: float, max_lam: int):
    c = -1j * Rj**2 * k
    return (
        c
        * A_coefficient_general(
            l=l, m=m, lp=lp, mp=mp, k=k, xb=xb, max_lam=max_lam
        )
        * spherical_jn(l, k * Ri)
        * spherical_jn(lp, k * Rj)
    )


@cache
def S_coefficient_diagonal(l: int, m: int, k: float, Ri: float):
    c = -1j * Ri**2 * k
    return (
        c
        * spherical_jn(l, k * Ri)
        * spherical_hl(l, k * Ri)
    )


class ClassicFiniteFWP3D(SWP3D):

    def __init__(self, **pars):
        super().__init__(**pars)

    def __str__(self):
        return super().__str__() + "\nPhysics:      Classic system"

    @classmethod
    def get_SSH(
        cls, i: int, r: float, s1: float | int, s2: float | int, **params
    ) -> Self:
        """ """
        if i < 1:
            raise ValueError("i must be a positive integer")
        if s1 <= 0 or s2 <= 0:
            raise ValueError("s1 and s2 must be positive")

        centers = []
        z = 0
        for k in range(i):
            centers.append(np.array([0, 0, z]))
            z += s1 + 2 * r
            centers.append(np.array([0, 0, z]))
            z += s2 + 2 * r
        centers.append(np.array([0, 0, z]))
        z += s2 + 2 * r
        for k in range(i):
            centers.append(np.array([0, 0, z]))
            z += s1 + 2 * r
            centers.append(np.array([0, 0, z]))
            z += s2 + 2 * r

        N = 4 * i + 1
        return cls(radii=np.ones(N) * r, centers=centers, **params)

    def compute_colinear_single_layer_potential_matrix_bruteforce(
        self, N_multipole: int, k0: float = 1e-6
    ) -> np.ndarray:
        """
        Computes the discrete approximation of the single layer potential matrix

        Args:
            N_multipole (int): Number of multipole to use

        Raises:
            NotImplementedError: Currently the formula works only for chain of resonators on the z-Axis

        Returns:
            np.ndarray: a N_multipole**2 * self.N array composed of self.N blocks representing the single layer potential. This is the matrix at the bottom of page 43 [3]
        """

        c = -1j * np.power(self.radii, 2)

        # For each l there are 2l+1 Y^l_m functions. Summing up we get to the following number
        total_number_base_functions = N_multipole**2

        S = np.zeros(
            (
                total_number_base_functions * self.N,
                total_number_base_functions * self.N,
            ),
            dtype=complex,
        )

        # Remark that S[i,j] = Se_i[j]
        # We extensively use (A.3)
        for i in range(self.N):
            for j in range(self.N):
                for l in range(N_multipole):
                    for m in range(-l, l + 1):
                        for lp in range(N_multipole):
                            for mp in range(-lp, lp + 1):
                                if i != j:
                                    rp = np.linalg.norm(
                                        self.centers[j] - self.centers[i]
                                    )
                                    S[
                                        flat_index(i, N_multipole, l, m),
                                        flat_index(j, N_multipole, lp, mp),
                                    ] = (
                                        c[j]
                                        * k0
                                        * A_coefficient_colinear(
                                            l=l, m=m, lp=lp, mp=mp, k=k0, rb=rp, max_lam=N_multipole
                                        )
                                        * spherical_jn(lp, k0 * self.radii[j])
                                        * spherical_jn(l, k0 * self.radii[i])
                                    )
                                else:
                                    if l == lp and m == mp:
                                        # print(
                                        #     i,
                                        #     N_multipole,
                                        #     l,
                                        #     m,
                                        #     "-->",
                                        #     flat_index(i, N_multipole, l, m),
                                        # )
                                        S[
                                            flat_index(i, N_multipole, l, m),
                                            flat_index(j, N_multipole, lp, mp),
                                        ] = (
                                            c[j]
                                            * k0
                                            * spherical_hl(l, self.radii[i] * k0)
                                            * spherical_jn(l, self.radii[i] * k0)
                                        )
        return S

    def compute_general_single_layer_potential_matrix(
        self, N_multipole: int, k0: float = 1e-6
    ) -> np.ndarray:
        """
        Computes the discrete approximation of the single layer potential matrix for general spheres not necessarily on the z-axis.

        Args:
            N_multipole (int): Number of multipole to use

        Returns:
            np.ndarray: a N_multipole**2 * self.N array composed of self.N blocks representing the single layer potential. This is the matrix at the bottom of page 43 [3]
        """

        # For each l there are 2l+1 Y^l_m functions. Summing up we get to the following number
        total_number_base_functions = N_multipole**2

        S = np.zeros(
            (
                total_number_base_functions * self.N,
                total_number_base_functions * self.N,
            ),
            dtype=complex,
        )

        # Remark that S[i,j] = Se_i[j]
        # We extensively use (A.3)
        for i in range(self.N):
            for j in range(self.N):
                for l in range(N_multipole):
                    for m in range(-l, l + 1):
                        for lp in range(N_multipole):
                            for mp in range(-lp, lp + 1):
                                if i != j:
                                    rp = np.linalg.norm(
                                        self.centers[i] - self.centers[j]
                                    )
                                    S[
                                        flat_index(i, N_multipole, l, m),
                                        flat_index(j, N_multipole, lp, mp),
                                    ] = S_coefficient_offdiagonal(
                                        l=l,
                                        m=m,
                                        lp=lp,
                                        mp=mp,
                                        k=k0,
                                        xb=tuple(
                                            self.centers[j]-self.centers[i]),
                                        Ri=self.radii[i],
                                        Rj=self.radii[j],
                                        max_lam=N_multipole+1
                                    )
                                else:
                                    if l == lp and m == mp:
                                        S[
                                            flat_index(i, N_multipole, l, m),
                                            flat_index(j, N_multipole, lp, mp),
                                        ] = S_coefficient_diagonal(l=l, m=m, k=k0, Ri=self.radii[i])
        return S

    def _compute_capactance_matrix_classical(self, N_multipole, colinear, k0=1e-6) -> np.ndarray:
        if colinear:
            for c in self.centers:
                if sum(np.abs(c[:2])) > 0:
                    raise ValueError(
                        "Sphere centers must be colinear along the z-axis for colinear single layer potential matrix computation."
                    )

            S = self.compute_colinear_single_layer_potential_matrix_bruteforce(
                N_multipole=N_multipole
            )
        else:
            S = self.compute_general_single_layer_potential_matrix(
                N_multipole=N_multipole
            )

        C = np.zeros((self.N, self.N), dtype=complex)
        for j in range(self.N):
            u_j = get_indicator_function_spherical_harmonics_expansion(
                N=self.N, N_multi=N_multipole, index=j)
            y = np.linalg.solve(S, u_j)
            for i in range(self.N):
                C[i, j] = - np.sqrt(4 * np.pi) * \
                    self.radii[i]**2 * y[i*N_multipole**2]
        return np.real(C)

    def get_capacitance_matrix(
            self,
            N_multipole=2,
            method: Literal['fmm', 'colinear', 'general'] = 'fmm',
            k0=1e-6,
            eps=1e-3,
            N_jobs=12
    ) -> np.ndarray:
        if method == 'colinear':
            return self._compute_capactance_matrix_classical(
                N_multipole=N_multipole, colinear=True, k0=k0)
        elif method == 'general':
            return self._compute_capactance_matrix_classical(
                N_multipole=N_multipole, colinear=False, k0=k0)
        elif method == 'fmm':
            assert N_multipole <= 2, "FMM method only supports up to dipole (N_multipole=2)"
            dipole = (N_multipole == 2)
            return fmm.compute_capacitance_matrix_accelerated(
                self.centers, self.radii, eps=eps, dipole=dipole, n_jobs=N_jobs
            )
        else:
            raise ValueError(f"Unknown method: {method}")

    def get_generalised_capacitance_matrix(self,
                                           N_multipole=2,
                                           method: Literal['fmm',
                                                           'colinear', 'general'] = 'fmm',
                                           k0=1e-6,
                                           eps=1e-3,
                                           N_jobs=12) -> np.ndarray:
        return self.get_material_matrix() @ self.get_capacitance_matrix(N_multipole=N_multipole, method=method, k0=k0, eps=eps, N_jobs=N_jobs)

    @override
    def compute_sorted_eigs_capacitance_matrix(
        self,
        eigenvalues_only=False,
        sorting: Literal[
            "eve_middle_localization",
            "eve_localization",
            "eva_real",
            "eva_imag",
            "eve_abs",
            "eva_first_val",
        ] = "eva_real",
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        Vinv = self.get_material_matrix(inverted=True)
        C = self.get_capacitance_matrix(**kwargs)

        if eigenvalues_only:
            D = sci.linalg.eigh(C, b=Vinv, eigvals_only=True)
        else:
            D, S = sci.linalg.eigh(C, b=Vinv)

        D, S = utils.sort_by_method(D, S, sorting)
        return D, S
