import numpy as np
import scipy as sci

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import matplotlib.cm as cm

from Utils.settings import settings as settings

from Utils.utils_general import *
import Utils.utils_propagation as utils_propagation

import Subwavelength1D.swp as swp
import Subwavelength1D.classic as classic
import Subwavelength1D.disordered as disordered
from Subwavelength1D.metaatom import *
from Subwavelength1D.quasiperiodic import *

from typing import Literal, Callable, Tuple, Self, List, override

import copy
from tqdm import tqdm


from itertools import product
from collections import deque
plt.rcParams.update(settings.matplotlib_params)


def visualize_spectrum(sp: swp.FiniteSWP1D, j, semilogy=False, axes=None):
    if not axes:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    D, S = sp.compute_sorted_eigs_capacitance_matrix()
    assert np.allclose(np.imag(D), 0), "Eigenvalues are not real"
    axes[0].plot(D, 'k.')
    axes[0].plot(j, D[j], 'ro')
    if semilogy:
        sv = np.abs(S[:, j])
        axes[1].semilogy(sv, 'k-')
    else:
        sv = np.real(S[:, j])
        axes[1].plot(sv, 'k-')
    if semilogy:
        axes[1].set_yscale('log')


def visualize_spectrum_with_winding(dp: disordered.DisorderedNonReciprocalFiniteSWP1D, j,
                                    D=None, S=None,
                                    axes=None,
                                    semilogy=False,
                                    nalpha=100,
                                    real_symmetrisation_acceleratrion=True):
    if not axes:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    if D is None or S is None:
        D, S = dp.compute_sorted_eigs_capacitance_matrix(
            real_symmetrisation_acceleratrion=real_symmetrisation_acceleratrion)
    # assert np.allclose(np.imag(D), 0), "Eigenvalues are not real"
    plot_eigenvalues(D, real=False, ax=axes[0])
    dp.plot_winding_regions(ax=axes[0], nalpha=nalpha)
    axes[0].plot(np.real(D[j]), np.imag(D[j]), 'ro')
    if semilogy:
        sv = np.abs(S[:, j])
        axes[1].semilogy(sv, 'k-')
    else:
        sv = np.real(S[:, j])
        axes[1].plot(sv, 'k-')


def visualize_spectrum_with_blockcolors(sp: DisorderedClassicFiniteSWP1D, j, semilogy=False, fig=None, y_cutoff=None):
    if fig is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    else:
        axes = fig.axes

    D, S = sp.compute_sorted_eigs_capacitance_matrix()
    assert np.allclose(np.imag(D), 0), "Eigenvalues are not real"
    D_cut = D[D > y_cutoff] if y_cutoff else D
    axes[0].plot(D_cut, 'k.')
    j_cut = len(D_cut) - j
    axes[0].plot(j_cut, D_cut[j_cut], 'ro')
    print("Lambda:", D_cut[j_cut])

    color_list = ["blue", "red", "green"]
    cc = [
        color_list[sp.idxs[sp.get_block_index_at_resonator(jj)]] for jj in range(sp.N)
    ]
    if semilogy:
        sv = np.abs(S[:, sp.N-j])
        axes[1].semilogy(sv, 'k-')
    else:
        sv = np.real(S[:, sp.N-j])
        axes[1].plot(sv, 'k-')
    axes[1].scatter(np.arange(sp.N), sv, c=cc, s=10, zorder=2)


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
        ax.set_xlabel("Index $i$")
        ax.set_ylabel(r"$\lambda_i$")
    else:
        if colorfunc:
            ax.scatter(np.real(D), np.imag(D), c=colorfunc(D), marker=".")
        else:
            ax.scatter(np.real(D), np.imag(D), c="black", marker=".")
        ax.set_xlabel(r"$\Re \lambda_i$")
        ax.set_ylabel(r"$\Im \lambda_i$")
    return ax
