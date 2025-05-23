import numpy as np
import scipy as sci

from Subwavelength1D.classic import *
from Subwavelength1D.swp import FiniteSWP1D

from Subwavelength1D.nonreciprocal import NonReciprocalFiniteSWP1D, NonReciprocalPeriodicSWP1D

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm

from typing import Literal, Callable, Tuple, Self, List, override
import itertools
import copy

from Utils.settings import settings as settings

from Utils.utils_general import *


class EPArrayClassicFiniteSWP1D(ClassicFiniteSWP1D):
    def __init__(self, N_cells, ep_idx=0, eps=0, **kwargs):
        N = 2*N_cells + 1
        self.ep_idx = ep_idx
        self.theta = np.pi * ep_idx / (2*2*N_cells)
        self.eps = eps
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

    def get_ep_convergence_constant(self, eps=1e-5):
        # This should be done exactly in the future
        ep, ep_indices, regular_indices = self.find_eps()
        cp = self.get_unperturbed_copy()
        cp.set_eps(eps)
