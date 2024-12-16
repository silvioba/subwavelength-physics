import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.axes import Axes

from Utils.settings import settings

from typing import Literal, Callable, Tuple, Self, List, override

plt.rcParams.update(settings.matplotlib_params)


def unique_eigenvector_phases(S):
    S = S / S[0, :]
    S = S / np.linalg.norm(S, axis=0)
    return S


def sort_by_eve_middle_localization(D, S):
    N = D.shape[0]
    sidx = np.argsort(np.linalg.norm(S[N // 3: 2 * N // 3, :], axis=0))
    return D[sidx], S[:, sidx]


def sort_by_eve_localization(D, S):
    sidx = np.argsort(-np.linalg.norm(S, axis=0, ord=1))
    return D[sidx], S[:, sidx]


def sort_by_eva_real(D, S):
    sidx = np.argsort(np.real(D))
    return D[sidx], S[:, sidx]


def sort_by_eva_imag(D, S):
    sidx = np.argsort(np.imag(D))
    return D[sidx], S[:, sidx]


def sort_by_eva_abs(D, S):
    sidx = np.argsort(np.abs(D))
    return D[sidx], S[:, sidx]


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
        self.inital_sorting_method = initial_sorting_method
        self.D = None
        self.S = None

    def _initial(self, D: np.array, S: np.array):
        D, S = sort_by_method(D, S, self.inital_sorting_method)
        self.D, self.S = D.copy(), S.copy()
        return D, S

    def next(self, D: np.array, S: np.array):
        if self.D is None:
            return self._initial(D, S)
        else:
            Dn = np.zeros_like(D)
            Sn = np.zeros_like(S)
            for i in range(D.shape[0]):
                # Finding the index of the previous eigenvector that is closest to the current one
                idx = np.argmax(np.abs(S[:, i].T @ self.S))
                # Asserting no index reuse
                assert Dn[idx] == 0
                # Storing the eigenvalue and eigenvector at the appropriate index
                Dn[idx], Sn[:, idx] = D[i], S[:, i]
                # Setting the used eigenvector to zero so that it is not used again
                self.S[:, idx] = 0
            self.D, self.S = Dn.copy(), Sn.copy()
            return Dn, Sn


def plot_eigenvalues(D, colorfunc=None, real=True, ax: Axes | None = None) -> Axes:
    """
    Plots the eigenvalues.

    Args:
        D (np.ndarray): Array of eigenvalues.
        colorfunc (Callable, optional): Function to determine the color of the points. Defaults to None.
        ax (Axes | None, optional): Matplotlib Axes object. Defaults to None.
    """
    if ax is None:
        fig, ax = plt.subplots()
    if real:
        if colorfunc:
            ax.scatter(np.arange(len(D)), D, c=colorfunc(D), marker=".")
        else:
            ax.scatter(np.arange(len(D)), D, c="black", marker=".")
        ax.set_xlabel("Site index $i$")
        ax.set_ylabel(r"$\lambda_i$")
    else:
        if colorfunc:
            ax.scatter(np.real(D), np.imag(D), c=colorfunc(D), marker=".")
        else:
            ax.scatter(np.real(D), np.imag(D), c="black", marker=".")
        ax.set_xlabel(r"$\Re \lambda_i$")
        ax.set_ylabel(r"$\Im \lambda_i$")
    return ax
