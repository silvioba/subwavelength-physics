"""Propagation matrices and Q-matrices for 1D resonator systems."""

import numpy as np
from typing import Literal, Callable, Tuple, Self, List
import matplotlib.pyplot as plt


def get_Q_matrix(z: int | float, x: int | float) -> np.ndarray:
    """Let u(x) = Ae^{ikx} + Be^{-ikx}. Then the Q matrix is defined as the matrix such that Q(z,x)@(A,B)^T = (u(x), u'(x))^T.

    Args:
        z (int | float): Wave number
        x (int | float, optional): Spatal position for change of basis.

    Returns:
        np.ndarray: The Q matrix
    """
    return _Q(x, 1j * z, -1j * z)


def _Q(x, r1, r2):
    return np.array([
        [np.exp(r1 * x), np.exp(r2 * x)],
        [r1 * np.exp(r1 * x), r2 * np.exp(r2 * x)]
    ])


def propagation_matrix_free_space(
    s: int | float,
    z: int | float,
    vo: int | float = 1,
    subwavelength: bool = True,
) -> np.ndarray:
    """
    Computes the propagation matrix from A to B in empty space like
    |--s--| 
    ^A    ^B

    Args:
        s (int | float): length in free space
        z (int | float): wave number
        delta (int | float): derivative transmission parameter
        subwavelength (bool): whether to use the subwavelength approximation

    Returns:
        np.ndarray: propagation matrix from A to B
    """
    if subwavelength:
        return np.array([[1, s], [0, 1]])
    else:
        phase = s * z / vo

        cos_p = np.cos(phase)
        sin_p = np.sin(phase)

        M11 = cos_p
        M12 = vo * sin_p / z
        M21 = -z * sin_p / vo
        M22 = cos_p

        return np.array([[M11, M12], [M21, M22]])


def propagation_matrix_single(
    l: int | float,
    s: int | float,
    vi: int | float,
    vo: int | float,
    z: int | float,
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
        z (int | float): frequency (corresponds to either lambda if subwavelength or omega if not)
        delta (int | float): derivative transmission parameter
        subwavelength (bool): whether to use the subwavelength approximation

    Returns:
        np.ndarray: propagation matrix from A to B
    """
    if subwavelength:
        vi2 = vi ** 2

        M11 = 1 - (l * s * z) / vi2
        M12 = s
        M21 = -(l * z) / vi2
        M22 = 1
        return np.array([[M11, M12], [M21, M22]])
    else:
        # Precompute phase terms
        phi_i = l * z / vi      # phase inside
        phi_o = s * z / vo     # phase outside

        cos_i = np.cos(phi_i)
        sin_i = np.sin(phi_i)
        cos_o = np.cos(phi_o)
        sin_o = np.sin(phi_o)

        # Matrix elements
        M11 = cos_i * cos_o - (vo * sin_i * sin_o) / (vi * delta)
        M12 = (vi * delta * cos_o * sin_i + vo * cos_i * sin_o) / z
        M21 = -z * cos_o * sin_i / (vi * delta) - z * cos_i * sin_o / vo
        M22 = cos_i * cos_o - (vi * delta * sin_i * sin_o) / vo

        return np.array([[M11, M12], [M21, M22]])


def nonreciprocal_propagation_matrix_single(
    l: int | float,
    s: int | float,
    z: int | float,
    gamma: int | float,
    delta: int | float = 1e-3,
    symmetrised: bool = True,
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
        z (int | float): frequency (corresponds to either lambda if subwavelength or omega if not)
        delta (int | float): derivative transmission parameter
        subwavelength (bool): whether to use the subwavelength approximation

    Returns:
        np.ndarray: propagation matrix from A to B
    """
    if subwavelength:
        def f(z): return z / (1-np.exp(-z))
        P = np.array([[1-l*s*z/f(gamma*l), np.exp(-gamma*l)*s],
                      [-l*z/f(gamma*l), np.exp(-gamma*l)]])
        if symmetrised:
            P = np.exp(gamma * l / 2) * P
        return P
    else:
        nu = np.sqrt(complex((gamma/2)**2-z**2))

        def Psi(a, b):
            return (a*np.cos(z*s)+b*np.sin(z*s))/nu

        P = np.array([
            [
                np.cos(z*s)*np.cosh(nu*l)-1/delta *
                Psi(-delta*gamma/2, z)*np.sinh(nu*l),
                1/z*np.cosh(nu*l)*np.sin(z*s)+delta/z *
                Psi(z, -gamma/(2*delta))*np.sinh(nu*l)
            ],
            [
                -z*np.cosh(nu*l)*np.sin(z*s)-z/delta *
                Psi(z, delta*gamma/2)*np.sinh(nu*l),
                np.cos(z*s)*np.cosh(nu*l)-delta *
                Psi(gamma/(2*delta), z)*np.sinh(nu*l)
            ]
        ])

    if symmetrised:
        return P
    else:
        return np.exp(-l*gamma/2)*P


def propagation_matrix_block(
    block: Tuple[Tuple[int | float]],
    z: int | float,
    vo: int | float = 1,
    delta: int | float = 1e-3,
    subwavelength: bool = True,
) -> np.ndarray:
    mat = np.eye(2)
    ll, ss = block
    if len(ss) == len(ll) + 1:
        # Block with pre and post spacing
        mat = propagation_matrix_free_space(
            ss[0], z, vo=vo, subwavelength=subwavelength) @ mat
        ss = ss[1:]
    for i in range(len(ll)):
        mat = propagation_matrix_single(
            l=ll[i], s=ss[i], vi=1, vo=vo, z=z, delta=delta, subwavelength=subwavelength) @ mat
    return mat


def propagation_matrix_nonreciprocal_block(
    block: Tuple[Tuple[int | float]],
    z: int | float,
    vo: int | float = 1,
    symmetrised: bool = True,
    subwavelength: bool = True,
) -> np.ndarray:
    if vo != 1:
        raise NotImplementedError(
            "Only vo=1 is implemented for nonreciprocal propagation matrices."
        )
    mat = np.eye(2)
    ll, ss, gamma = block
    if len(ss) == len(ll) + 1:
        # Block with pre and post spacing
        mat = propagation_matrix_free_space(
            ss[0], z, vo=vo, subwavelength=subwavelength) @ mat
        ss = ss[1:]
    for i in range(len(ll)):
        mat = nonreciprocal_propagation_matrix_single(
            l=ll[i], s=ss[i], z=z, gamma=gamma[i], symmetrised=symmetrised, subwavelength=subwavelength) @ mat
    return mat


def propagation_matrix_block_function(
    block: Tuple[List[int | float]],
    delta: int | float = 1e-3,
    subwavelength: bool = True,
) -> Callable[[int | float], np.ndarray]:
    return lambda z: propagation_matrix_block(
        block, z, delta=delta, subwavelength=subwavelength
    )


def plot_propagation_eigenvalues(
    fswp,
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
    for i, z in enumerate(ks):
        fswp.set_omega(z)
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
