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
    sidx = np.argsort(np.linalg.norm(S[N // 3 : 2 * N // 3, :], axis=0))
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


def plot_eigenvalues(D, colorfunc=None, ax: Axes | None = None) -> Axes:
    """
    Plots the eigenvalues.

    Args:
        D (np.ndarray): Array of eigenvalues.
        colorfunc (Callable, optional): Function to determine the color of the points. Defaults to None.
        ax (Axes | None, optional): Matplotlib Axes object. Defaults to None.
    """
    if ax is None:
        fig, ax = plt.subplots()
    if colorfunc:
        ax.scatter(np.arange(len(D)), D, c=colorfunc(D), marker=".")
    else:
        ax.scatter(np.arange(len(D)), D, c="black", marker=".")
    ax.set_xlabel("Site index $i$")
    ax.set_ylabel(r"$\lambda_i$")
    return ax
