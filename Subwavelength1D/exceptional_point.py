"""Systems with complex velocities near exceptional points."""

import numpy as np
import scipy as sci
from scipy.linalg import null_space, pinv

from Subwavelength1D.classic import *
from Subwavelength1D.swp import FiniteSWP1D

from Subwavelength1D.nonreciprocal import NonReciprocalFiniteSWP1D, NonReciprocalPeriodicSWP1D

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm

from typing import Literal, Callable, Tuple, Self, List
from typing_extensions import override

import itertools
import copy

from Utils.settings import settings as settings

from Utils.utils_general import *


class EPArrayClassicFiniteSWP1D(ClassicFiniteSWP1D):
    """Classical system with complex velocities parametrised near exceptional points."""

    def __init__(self, N_cells, ep_idx=0, eps=0, **kwargs):
        self.N_cells = N_cells
        self.ep_idx = ep_idx
        self.theta = np.pi * ep_idx / (2*2*N_cells)
        self.eps = eps
        N = 2*N_cells + 1
        l = [1]*N
        s = [1]*(N-1)
        s[-1] = 1/eps if eps != 0 else np.inf
        v_in = [np.exp(1j * self.theta),
                np.exp(-1j * self.theta)]*N_cells+[1]

        super().__init__(N=N, l=l, s=s, v_in=v_in, **kwargs)

    def __str__(self):
        return f"EPArrayClassicFiniteSWP1D(N_cells={self.N_cells}, ep_idx={self.ep_idx}, eps={self.eps})"

    def set_eps(self, eps):
        self.eps = eps
        self.s[-1] = 1/eps if eps != 0 else np.inf
        self.set_geometry(self.l, self.s)

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
    ) -> Tuple[np.ndarray, np.ndarray]:
        # We override the eigenvalue computation because the system is no longer Hermitian
        C = self.get_generalised_capacitance_matrix()
        if eigenvalues_only:
            D = np.linalg.eigvals(C)
            S = None
        else:
            D, S = np.linalg.eig(C)

        D, S = utils.sort_by_method(D, S, sorting)
        return D, S

    def get_unperturbed_copy(self):
        unperturbed = copy.deepcopy(self)
        unperturbed.set_eps(0)
        return unperturbed

    def get_perturbation_matrix(self):
        """Returns Delta such that C = C0 + eps*Delta where C0 is the unperturbed generalised capacitance matrix.
        """
        Delta = np.zeros((self.N, self.N), dtype=complex)
        Delta[-2, -2] = self.v_in[-2]
        Delta[-2, -1] = -self.v_in[-2]
        Delta[-1, -2] = -1
        Delta[-1, -1] = 1
        return Delta

    def find_eps(self, order=3, tol=1e-4):
        def split(c):
            return [np.real(c), np.imag(c)]

        unperturbed = self.get_unperturbed_copy()
        D, S = unperturbed.compute_sorted_eigs_capacitance_matrix()

        # Compute the pairwise distances between the eigenvalues
        seps = sci.spatial.distance.squareform(
            sci.spatial.distance.pdist(np.array(list(map(split, D)))))
        np.fill_diagonal(seps, np.inf)

        # Find the indices of the eigenvalues that are close to each other (ignoring the first two)
        ep_indices = np.unique(np.where(seps < tol)[0])[2:]
        assert len(
            ep_indices) == order, f"Found {len(ep_indices)} eigenvalues close to each other, expected {order}"

        ep = np.mean(D[ep_indices])

        regular_indices = np.setdiff1d(np.arange(len(D)), ep_indices)

        return ep, ep_indices, regular_indices

    def get_ep_convergence_constant(self, eps=1e-5, order=3):
        # This should be done exactly in the future
        ep, ep_indices, regular_indices = self.find_eps(order=order)
        cp = self.get_unperturbed_copy()
        cp.set_eps(eps)
        D, _ = cp.compute_sorted_eigs_capacitance_matrix(eigenvalues_only=True)
        alpha = np.mean(np.power(np.abs(D[ep_indices] - ep), order)/eps)
        return alpha

    def get_regular_convergence_constant(self, index, eps=1e-5):
        # This should be done exactly in the future
        cp = self.get_unperturbed_copy()
        D0, _ = cp.compute_sorted_eigs_capacitance_matrix(
            eigenvalues_only=True)
        cp.set_eps(eps)
        D, _ = cp.compute_sorted_eigs_capacitance_matrix(
            eigenvalues_only=True)
        alpha = np.abs(D[index] - D0[index])/eps

        return alpha

    def compute_alpha(self, index, order=3):
        def _jordan_chain(A, lam, k=3):
            """Return right Jordan chain [r0,…,r_{k-1}] for (A,lam)."""
            Z = A - lam*np.eye(A.shape[0], dtype=A.dtype)
            r = [null_space(Z)[:, 0]]                 # r0
            for j in range(1, k):
                # r_j solves Z r_j = r_{j-1}
                r.append(pinv(Z) @ r[j-1])
            return r

        def _left_chain(A, lam, k=3):
            """Return left Jordan chain [l0,…,l_{k-1}]  (columns, not rows!)."""
            ZH = (A - lam*np.eye(A.shape[0], dtype=A.dtype)).conj().T
            l = [null_space(ZH)[:, 0]]                # l0
            for j in range(1, k):
                l.append(pinv(ZH) @ l[j-1])          # ZH l_j = l_{j-1}
            return l

        def _biorthogonalise(l_chain, r_chain):
            """Rescale the LEFT chain so  l_i^* r_j = δ_{ij}."""
            L = np.column_stack(l_chain)
            R = np.column_stack(r_chain)
            G = np.conj(L).T @ R
            S = np.linalg.inv(G).conj().T       # G^{-H}
            Lnew = L @ S
            return [Lnew[:, i] for i in range(len(r_chain))]

        cp = self.get_unperturbed_copy()
        D0, _ = cp.compute_sorted_eigs_capacitance_matrix(
            eigenvalues_only=True)
        C = cp.get_generalised_capacitance_matrix()
        Delta = self.get_perturbation_matrix()

        # Compute the right and left Jordan chains
        r = _jordan_chain(C, D0[index], order)
        l = _left_chain(C, D0[index], order)
        # Compute the biorthogonalisation of the left chain
        l = _biorthogonalise(l, r)
        return np.abs(np.vdot(l[order-1], Delta @ r[0]))
