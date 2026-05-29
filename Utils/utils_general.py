"""General utilities: eigenvalue sorting, root-finding, and eigenvector tracking."""

from functools import wraps
import warnings
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.axes import Axes

from Utils.settings import settings

from typing import Literal, Callable, Tuple, Self, List
from typing_extensions import override

plt.rcParams.update(settings.matplotlib_params)


def unique_eigenvector_phases(S):
    S = S / S[0, :]
    S = S / np.linalg.norm(S, axis=0)
    return S


def remove_edgemodes(D, S, num=1):
    """Removes edge-localized modes based on the amplitude at the edges.

    Args:
        D (np.ndarray): Array of eigenvalues.
        S (np.ndarray): Array of eigenvectors.
        num (int, optional): Number of edge modes to remove. Defaults to 1.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Arrays of eigenvalues and eigenvectors with edge modes removed.
    """

    sidx = np.argsort(np.abs(S[0, :])+np.abs(S[-1, :]))
    edge_idx = sidx[-num:]
    keep_idx = np.setdiff1d(np.arange(D.shape[0]), edge_idx)
    return D[keep_idx], S[:, keep_idx]


def mullers_method(f, x0, x1, x2, tol=1e-7, max_iter=100):
    """
    Find a root of function f using Muller's method given three starting points.
    """
    res = 0
    for i in range(max_iter):
        f0, f1, f2 = f(x0), f(x1), f(x2)

        h1 = x1 - x0
        h2 = x2 - x1

        d1 = (f1 - f0) / h1
        d2 = (f2 - f1) / h2

        d = (d2 - d1) / (h2 + h1)

        b = d2 + h2 * d
        D = np.sqrt(b**2 - 4 * f2 * d + 0j)  # +0j ensures complex sqrt

        # Choose the sign that maximizes the denominator
        if np.abs(b - D) < np.abs(b + D):
            E = b + D
        else:
            E = b - D

        h = -2 * f2 / E
        p = x2 + h

        if np.abs(h) < tol:
            return p

        x0, x1, x2 = x1, x2, p

    warnings.warn("Muller's method did not converge")
    return p


def find_roots_muller(f, z0, n_roots, perturbation=1e-2):
    """
    Find n_roots of f starting near z0 using Muller's method with deflation.
    """
    roots = []

    def deflated_f(z):
        val = f(z)
        for r in roots:
            val /= (z - r)
        return val

    current_z = z0

    for _ in range(n_roots):
        # Generate 3 initial points near the current guess
        p0 = current_z
        p1 = current_z + perturbation
        p2 = current_z - perturbation

        root = mullers_method(deflated_f, p0, p1, p2)
        roots.append(root)

        # Update starting point for next root (simple heuristic: move slightly away)
        current_z = root + perturbation * 2

    return np.array(roots)


def sort_by_eve_middle_localization(D, S):
    N = D.shape[0]
    sidx = np.argsort(np.linalg.norm(S[N // 3: 2 * N // 3, :], axis=0))
    return D[sidx], S[:, sidx]


def sort_by_eve_localization(D, S):
    sidx = np.argsort(-np.linalg.norm(S, axis=0, ord=1))
    return D[sidx], S[:, sidx]


def sort_by_eva_real(D, S):
    sidx = np.argsort(np.real(D))
    return D[sidx], (S[:, sidx] if S is not None else None)


def sort_by_eva_imag(D, S):
    sidx = np.argsort(np.imag(D))
    return D[sidx], (S[:, sidx] if S is not None else None)


def sort_by_eva_abs(D, S):
    sidx = np.argsort(np.abs(D))
    return D[sidx], (S[:, sidx] if S is not None else None)


def sort_by_eve_first_val(D, S):
    sidx = np.argsort(np.abs(S[0, :]))
    return D[sidx], S[:, sidx]


sorting_methods = [
    "eve_middle_localization",
    "eve_localization",
    "eva_real",
    "eva_imag",
    "eve_abs",
    "eva_first_val",
]


def sort_by_method(
    D: np.ndarray,
    S: np.ndarray,
    method: Literal[
        "eve_middle_localization",
        "eve_localization",
        "eva_real",
        "eva_imag",
        "eva_abs",
        "eve_first_val",
    ],
):
    """
    Sorts the eigenvalues and eigenvectors based on the specified method.

    Args:
        D (np.ndarray): Array of eigenvalues.
        S (np.ndarray): Array of eigenvectors.
        method (Literal["eve_middle_localization", "eve_localization", "eva_real", "eva_imag", "eva_abs", "eve_first_val"]): Sorting method.

    Raises:
        ValueError: If an unknown sorting method is provided.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Sorted eigenvalues and eigenvectors.
    """
    if method == "eve_middle_localization":
        return sort_by_eve_middle_localization(D, S)
    elif method == "eve_localization":
        return sort_by_eve_localization(D, S)
    elif method == "eva_real":
        return sort_by_eva_real(D, S)
    elif method == "eva_imag":
        return sort_by_eva_imag(D, S)
    elif method == "eva_abs":
        return sort_by_eva_abs(D, S)
    elif method == "eve_first_val":
        return sort_by_eve_first_val(D, S)
    else:
        raise ValueError("Unknown sorting method")


class EigenvectorPathTracker:
    """
    A class to track the path of eigenvectors and eigenvalues through iterations. Tracking is achieved using dot product similarity on the eigenvectors.
    Necessarily fails at exceptional points or when the step size is too large.
    Attributes:
        initial_sorting_method (str): The method used for initial sorting of eigenvalues and eigenvectors.
        D (np.array): The array of eigenvalues.
        S (np.array): The array of eigenvectors.
    Methods:
        __init__(initial_sorting_method="eva_real"):
            Initializes the EigenvectorPathTracker with a specified initial sorting method.
        next(D: np.array, S: np.array):
            Sorts D and S to match the stored eigenvectors.
    """

    def __init__(self, initial_sorting_method="eva_real"):
        self.initial_sorting_method = initial_sorting_method
        self.D = None
        self.S = None

    def _initial(self, D: np.array, S: np.array):
        D, S = sort_by_method(D, S, self.initial_sorting_method)
        self.D, self.S = D.copy(), S.copy()
        return D, S

    def next(self, D: np.array, S: np.array):
        if self.D is None:
            return self._initial(D, S)
        else:
            Dn = np.zeros_like(D)
            Sn = np.zeros_like(S)
            correlations = np.abs(S.T @ self.S)
            for i in range(D.shape[0]):
                # Finding the eigenvector pair with the highest correlation
                idx = np.unravel_index(
                    np.argmax(correlations, axis=None), correlations.shape)
                new_idx = idx[0]
                old_idx = idx[1]
                # Asserting no index reuse
                assert Dn[old_idx] == 0
                # Storing the eigenvalue and eigenvector at the appropriate index
                Dn[old_idx], Sn[:, old_idx] = D[new_idx], S[:, new_idx]
                # Setting the correlations to zero so that the pair is not used again
                correlations[new_idx] = 0
                correlations[:, old_idx] = 0
            self.D, self.S = Dn.copy(), Sn.copy()
            return Dn, Sn
