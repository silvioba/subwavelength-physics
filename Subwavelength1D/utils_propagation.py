import numpy as np
from typing import Literal, Callable, Tuple, Self, List
import matplotlib.pyplot as plt

from Subwavelength1D.classic import FiniteSWP1D


def get_subwavelength_propagation_matrix_single(l, s, k):
    return np.array([[1 - l * s * k, s], [-l * k, 1]])


def propagation_matrix_single(
    l: int | float,
    s: int | float,
    k: int | float,
    delta: int | float = 1e-3,
    subwavelength: bool = True,
) -> np.ndarray:
    """
    Computes the propagation matrix from A to B in a structure like

     |--l--|--s--|
     [-----]     [-----]
    ^A          ^B

    Args:
        l (int | float): length of the resonator
        s (int | float): length in free space
        k (int | float): wave number
        delta (int | float): derivative transmission parameter
        subwavelength (bool): whether to use the subwavelength approximation

    Returns:
        np.ndarray: propagation matrix from A to B
    """
    if subwavelength:
        return get_subwavelength_propagation_matrix_single(l, s, k)
    ckl = np.cos(k * l)
    skl = np.sin(k * l)
    cks = np.cos(k * s)
    sks = np.sin(k * s)
    return np.array(
        [
            [
                ckl * ckl - (1 / delta) * skl * sks,
                (delta / k) * cks * skl + (1 / k) * ckl * sks,
            ],
            [(-k / delta) * cks * skl + k * ckl * sks, ckl * cks - delta * skl * sks],
        ]
    )


def propagation_matrix_block(
    block: Tuple[List[int | float]],
    k: int | float,
    delta: int | float = 1e-3,
    subwavelength: bool = True,
) -> np.ndarray:
    mat = np.eye(2)
    ll, ss = block
    for i in range(len(ll)):
        mat = propagation_matrix_single(ll[i], ss[i], k, delta, subwavelength) @ mat
    return mat


def propagation_matrix_block_function(
    block: Tuple[List[int | float]],
    delta: int | float = 1e-3,
    subwavelength: bool = True,
) -> Callable[[int | float], np.ndarray]:
    return lambda k: propagation_matrix_block(
        block, k, delta=delta, subwavelength=subwavelength
    )


def plot_propagation_eigenvalues(
    fswp: FiniteSWP1D,
    k_min: float = 1e-1,
    k_max: int = 5,
    n_pts: int = 100,
    ax=None,
    semilogy=True,
    space_from_end=1,
    subwavelength=False,
    only_large=False,
    color=None,
):
    """
    Plots the eigenvaues of a propagation matrix

    Args:
        fswp (OneDimensionalFiniteSWLProblem): Finite subwavelength problem
        k_min (float, optional): minimal value for the wave number. Defaults to 1e-1.
        k_max (int, optional): Maximal value for the wave number. Defaults to 5.
        n_pts (int, optional): Number of sample to take in the interval [k_min, k_max]. Defaults to 100.
        ax: matplotlib ax to plot on. Defaults to None.
        semilogy (bool, optional): Uses semilogy in the plot. Defaults to True.
    """

    if ax is None:
        fig, ax = plt.subplots(1, 1)
    ks = np.linspace(k_min, k_max, n_pts)
    eves = np.zeros((len(ks), 2 if not only_large else 1), dtype=complex)
    for i, k in enumerate(ks):
        fswp.set_params(
            k_in=np.ones(fswp.N) * k,
            k_out=k,
            v_in=np.ones(fswp.N) * fswp.omega / k,
            v_out=fswp.omega / k,
        )
        D, S = np.linalg.eig(
            fswp.compute_propagation_matrix(
                space_from_end=space_from_end, subwavelength=subwavelength
            )
        )
        if only_large:
            eves[i] = D[np.argmax(np.abs(D))]
        else:
            eves[i] = np.sort(np.abs(D))
    if color and only_large:
        assert len(color) == 1
        ax.semilogy(ks, np.abs(eves[:, 0]), color[0])
    elif color and not only_large:
        assert len(color) == 2
        ax.semilogy(ks, np.abs(eves[:, 0]), color[0])
        ax.semilogy(ks, np.abs(eves[:, 1]), color[1])
    else:
        ax.semilogy(ks, np.abs(eves[:, 0]), "b-")
        ax.semilogy(ks, np.abs(eves[:, 1]), "r-")
    return ax
