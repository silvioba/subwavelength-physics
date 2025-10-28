import numpy as np
from typing import Literal, Callable, Tuple, Self, List
import matplotlib.pyplot as plt

from Subwavelength1D.classic import FiniteSWP1D


def get_Q_matrix(k: int | float, x: int | float) -> np.ndarray:
    """Let u(x) = Ae^{ikx} + Be^{-ikx}. Then the Q matrix is defined as the matrix such that Q(k,x)@(A,B)^T = (u(x), u'(x))^T.

    Args:
        k (int | float): Wave number
        x (int | float, optional): Spatal position for change of basis.

    Returns:
        np.ndarray: The Q matrix
    """
    return np.array([[np.exp(1j * k * x), np.exp(-1j * k * x)], [1j * k * np.exp(1j * k * x), -1j * k * np.exp(-1j * k * x)]])


def get_subwavelength_propagation_matrix_single(l, s, lbda):
    return np.array([[1 - l * s * lbda, s], [-l * lbda, 1]])


def nonreciprocal_nonsubwavelength_propagation_matrix_single(
    l: int | float,
    s: int | float,
    gamma: int | float,
    omega: int | float,
    delta: int | float,
    symmetrised=True,
):
    nu = np.sqrt(complex((gamma/2)**2-omega**2))

    def Psi(a, b):
        return (a*np.cos(omega*s)+b*np.sin(omega*s))/nu

    P = np.array([
        [
            np.cos(omega*s)*np.cosh(nu*l)-1/delta *
            Psi(-delta*gamma/2, omega)*np.sinh(nu*l),
            1/omega*np.cosh(nu*l)*np.sin(omega*s)+delta/omega *
            Psi(omega, -gamma/(2*delta))*np.sinh(nu*l)
        ],
        [
            -omega*np.cosh(nu*l)*np.sin(omega*s)-omega/delta *
            Psi(omega, delta*gamma/2)*np.sinh(nu*l),
            np.cos(omega*s)*np.cosh(nu*l)-delta *
            Psi(gamma/(2*delta), omega)*np.sinh(nu*l)
        ]
    ])

    if symmetrised:
        return P
    else:
        return np.exp(-l*gamma/2)*P


def nonreciprocal_subwavelength_propagation_matrix_single(
    l: int | float,
    s: int | float,
    gamma: int | float,
    lbda: int | float,
    symmetrised=True,
) -> np.ndarray:
    def f(z): return z / (1-np.exp(-z))
    P = np.array([[1-l*s*lbda/f(gamma*l), np.exp(-gamma*l)*s],
                  [-l*lbda/f(gamma*l), np.exp(-gamma*l)]])
    if symmetrised:
        P = np.exp(gamma * l / 2) * P
    return P


def propagation_matrix_free_space(
    s: int | float,
    k: int | float,
    delta: int | float = 1e-3,
    subwavelength: bool = True,
) -> np.ndarray:
    """
    Computes the propagation matrix from A to B in empty space like
    |--s--| 
    ^A    ^B

    Args:
        s (int | float): length in free space
        k (int | float): wave number
        delta (int | float): derivative transmission parameter
        subwavelength (bool): whether to use the subwavelength approximation

    Returns:
        np.ndarray: propagation matrix from A to B
    """
    if subwavelength:
        return np.array([[1, s], [0, 1]])
    else:
        raise NotImplementedError(
            "Non-subwavelength propagation matrix for free space not implemented yet.")


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
            [(-k / delta) * cks * skl + k * ckl *
             sks, ckl * cks - delta * skl * sks],
        ]
    )


def propagation_matrix_block(
    block: Tuple[Tuple[int | float]],
    k: int | float,
    delta: int | float = 1e-3,
    subwavelength: bool = True,
) -> np.ndarray:
    mat = np.eye(2)
    ll, ss = block
    if len(ss) == len(ll) + 1:
        # Block with pre and post spacing
        mat = propagation_matrix_free_space(
            ss[0], k, delta, subwavelength) @ mat
        ss = ss[1:]
    for i in range(len(ll)):
        mat = propagation_matrix_single(
            ll[i], ss[i], k, delta, subwavelength) @ mat
    return mat


def propagation_matrix_nonreciprocal_block(
    block: Tuple[Tuple[int | float]],
    k: int | float,
    symmetrised: bool = True,
) -> np.ndarray:
    mat = np.eye(2)
    ll, ss, gamma = block
    if len(ss) == len(ll) + 1:
        # Block with pre and post spacing
        mat = propagation_matrix_free_space(
            ss[0], k, subwavelength=True) @ mat
        ss = ss[1:]
    for i in range(len(ll)):
        mat = nonreciprocal_subwavelength_propagation_matrix_single(
            ll[i], ss[i], gamma[i], k, symmetrised=symmetrised) @ mat
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
        fswp.set_omega(k)
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
