import numpy as np
from Subwavelength3D.swp import SWP3D

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap, LogNorm

from typing import Literal, Callable, Tuple, Self, List, override, Dict

from scipy.special import spherical_jn, hankel1
from sympy.physics.wigner import wigner_3j
from scipy.linalg import block_diag

from joblib import Parallel, delayed  # For parallelism


# plt.rcParams.update(settings.matplotlib_params)


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
    idx[N_multi**2 * index : N_multi**2 * (index + 1)] = 1

    return idx


def estimate_time():
    pass


def C_coefficent(l: int, m: int, lp: int, mp: int, lam: int, mu: int) -> float:
    """
    C coefficent used for the addition theorem as presented on page 42 of [3]

    Returns:
        float
    """
    return (
        (1j) ** (lp - l + lam)
        * (-1) ** m
        * np.sqrt(4 * np.pi * (2 * l + 1) * (2 * lp + 1) * (2 * lam + 1))
        * wigner_3j(l, lp, lam, 0, 0, 0)
        * wigner_3j(l, lp, lam, -m, mp, mu)
    )


def precompute_C_and_A_coefficients_pairwise(
    max_l: int, pairwise_distances: np.ndarray, k0: float
):
    """
    Precomputes C_coefficients and A_coefficients only at the unique pairwise distances.

    Args:
        max_l (int): Maximum degree of spherical harmonics (l and lp).
        pairwise_distances (np.ndarray): Unique pairwise distances where A_coefficients are needed.
        k0 (float): Wavenumber used for scaling distances.

    Returns:
        tuple: (C_cache, A_cache)
            - C_cache (np.ndarray): Precomputed C_coefficients, 6D array of shape
              (max_l, 2*max_l+1, max_l, 2*max_l+1, 2*max_l, 2*max_l+1).
            - A_cache (dict): Dictionary mapping distances to A_coefficients.
              Keys are distances, and values are 4D numpy arrays of shape
              (max_l, 2*max_l+1, max_l, 2*max_l+1).
    """
    # Precompute C_coefficients
    C_cache = np.zeros(
        (max_l, 2 * max_l + 1, max_l, 2 * max_l + 1, 2 * max_l, 2 * max_l + 1),
        dtype=complex,
    )
    for l in range(max_l):
        for m in range(-l, l + 1):
            for lp in range(max_l):
                for mp in range(-lp, lp + 1):
                    for lam in range(abs(l - lp), l + lp + 1):
                        for mu in range(-lam, lam + 1):
                            C_cache[l, m + max_l, lp, mp + max_l, lam, mu + max_l] = (
                                (1j) ** (lp - l + lam)
                                * (-1) ** m
                                * np.sqrt(
                                    4
                                    * np.pi
                                    * (2 * l + 1)
                                    * (2 * lp + 1)
                                    * (2 * lam + 1)
                                )
                                * float(wigner_3j(l, lp, lam, 0, 0, 0))
                                * float(wigner_3j(l, lp, lam, -m, mp, mu))
                            )

    # Precompute A_coefficients for unique pairwise distances
    unique_distances = np.unique(pairwise_distances)
    A_cache = {}
    for z in unique_distances:
        if z < 1e-5:
            A_cache[z] = 0
            continue
        A_array = np.zeros((max_l, 2 * max_l + 1, max_l, 2 * max_l + 1), dtype=complex)
        for l in range(max_l):
            for m in range(-l, l + 1):
                for lp in range(max_l):
                    for mp in range(-lp, lp + 1):
                        A = 0
                        for lam in range(abs(l - lp), l + lp + 1):
                            A += (
                                np.sqrt((2 * lam + 1) / (4 * np.pi))
                                * C_cache[l, m + max_l, lp, mp + max_l, lam, 0 + max_l]
                                * spherical_hl(lam, k0 * z)
                            )
                        A_array[l, m + max_l, lp, mp + max_l] = A
        A_cache[z] = A_array

    return C_cache, A_cache


class ClassicFiniteFWP3D(SWP3D):

    def __init__(self, **pars):
        super().__init__(**pars)

    def __str__(self):
        return super().__str__() + "\nPhysics:      Classic system"

    def compute_single_layer_potential_matrix_bruteforce(
        self, N_multipole: int
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

        for c in self.centers:
            if sum(np.abs(c[:2])) > 0:
                raise NotImplementedError(
                    "Discretized single layer potential is only implemented for chains of resonators on the z-Axis using the simplified expression for the addition theorem."
                )
            # TODO We can use product formula for the other cases

        def A_coefficent(z, l, m, lp, mp) -> float:
            A = 0
            for lam in range(N_multipole):
                A += (
                    np.sqrt((2 * lam + 1) / 4 / np.pi)
                    * C_coefficent(l, m, lp, mp, lam, 0)
                    * spherical_hl(lam, z)
                )
            return A

        c = -1j * self.radii**2

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
                                    ] = (
                                        c[i]
                                        * self.k0
                                        * A_coefficent(
                                            z=self.k0 * rp, l=l, m=m, lp=lp, mp=mp
                                        )
                                        * spherical_jn(lp, self.k0 * rp)
                                        * spherical_jn(l, self.k0 * self.radii[i])
                                    )
                                else:
                                    if l == lp and m == mp:
                                        S[
                                            flat_index(i, N_multipole, l, m),
                                            flat_index(j, N_multipole, lp, mp),
                                        ] = (
                                            c[i]
                                            * self.k0
                                            * spherical_hl(l, self.radii[i] * self.k0)
                                            * spherical_jn(l, self.radii[i] * self.k0)
                                        )
        return S

    def compute_single_layer_potential_matrix(self, N_multipole: int) -> np.ndarray:
        """
        Computes the discrete approximation of the single layer potential matrix

        Args:
            N_multipole (int): Number of multipole to use

        Raises:
            NotImplementedError: Currently the formula works only for chain of resonators on the z-Axis

        Returns:
            np.ndarray: a N_multipole**2 * self.N array composed of self.N blocks representing the single layer potential. This is the matrix at the bottom of page 43 [3]
        """

        for c in self.centers:
            if sum(np.abs(c[:2])) > 0:
                raise NotImplementedError(
                    "Discretized single layer potential is only implemented for chains of resonators on the z-Axis using the simplified expression for the addition theorem."
                )
            # TODO We can use product formula for the other cases

        c = -1j * self.radii**2

        N = self.N
        L = N_multipole
        S_size = N * L**2  # Size of the flattened matrix
        S = np.zeros((S_size, S_size), dtype=complex)  # Dense matrix

        # Convert self.centers from a list of (1, 3) arrays to a numpy array of shape (N, 3)
        centers_array = np.squeeze(np.array(self.centers))  # Shape (N, 3)

        # Compute pairwise distances using broadcasting
        pairwise_distances = np.linalg.norm(
            centers_array[:, None, :] - centers_array[None, :, :], axis=-1
        )

        # Precompute coefficients
        C_cache, A_cache = precompute_C_and_A_coefficients_pairwise(
            L, pairwise_distances, self.k0
        )

        # Precompute spherical Bessel and Hankel functions for radii
        precomputed_jn = {
            r: {l: spherical_jn(l, r * self.k0) for l in range(L)} for r in self.radii
        }
        precomputed_hl = {
            r: {l: spherical_hl(l, r * self.k0) for l in range(L)} for r in self.radii
        }

        for i in range(N):
            for j in range(N):
                if i == j:
                    # Diagonal terms (self-interaction)
                    for l in range(L):
                        for m in range(-l, l + 1):
                            idx = flat_index(i, L, l, m)
                            S[idx, idx] = (
                                c[i]
                                * self.k0
                                * precomputed_hl[self.radii[i]][l]
                                * precomputed_jn[self.radii[i]][l]
                            )
                else:
                    # Use precomputed A_cache for pairwise distance
                    d = pairwise_distances[i, j]
                    A_array = A_cache[d]
                    for l in range(L):
                        for m in range(-l, l + 1):
                            for lp in range(L):
                                for mp in range(-lp, lp + 1):
                                    idx_i = flat_index(i, L, l, m)
                                    idx_j = flat_index(j, L, lp, mp)

                                    # Use precomputed A coefficient
                                    A_value = A_array[l, m + L, lp, mp + L]

                                    # Update matrix value
                                    S[idx_i, idx_j] = (
                                        c[i]
                                        * self.k0
                                        * A_value
                                        * precomputed_jn[self.radii[i]][l]
                                    )

        return S

    def get_capacitance_matrix(self, N_multipole=1) -> np.ndarray:
        S = self.compute_single_layer_potential_matrix(N_multipole=N_multipole)
        C = np.zeros((self.N, self.N), dtype=complex)

        # TODO: S should be symmetric in a classical system. Then we could use cholsesky
        # TODO: test if LU would be better
        Q, R = np.linalg.qr(S)
        for j in range(self.N):
            u_j = get_mask_block(N=self.N, N_multi=N_multipole, index=j)
            y = np.linalg.lstsq(S, u_j, rcond=None)[0]  # np.linalg.solve(R, Q.T @ u_j)
            for i in range(self.N):
                u_i = get_mask_block(N=self.N, N_multi=N_multipole, index=i)
                C[i, j] = u_i.T @ y * (-4 * np.pi * self.radii[i] ** 2)
        return C

    def get_generalised_capacitance_matrix(self) -> np.ndarray:
        pass
