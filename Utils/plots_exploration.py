import numpy as np
import scipy as sci

import matplotlib as mpl
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
import Subwavelength3D.classic_finite as classic_finite_3D
import Subwavelength1D.disordered as disordered
from Subwavelength1D.metaatom import *
from Subwavelength1D.quasiperiodic import *

from typing import Literal, Callable, Tuple, Self, List
from typing_extensions import override

import copy
from tqdm import tqdm


from itertools import product
from collections import deque
plt.rcParams.update(settings.matplotlib_params)


def visualize_spectrum(sp: swp.FiniteSWP1D, j, D=None, S=None, semilogy=False, axes=None):
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    else:
        fig = axes[0].get_figure()
    if D is None or S is None:
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

    return fig, axes


def visualize_spectrum_2D(sp: classic_finite_3D.ClassicFiniteSWP3D, j, D=None, S=None, axes=None, radius_scaling=1):
    # assert np.allclose(sp.centers[:, 2], 0), "All resonator centers must lie in the xy-plane (z=0)"

    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    else:
        fig = axes[0].get_figure()
    if D is None or S is None:
        D, S = sp.compute_sorted_eigs_capacitance_matrix()
    assert np.allclose(np.imag(D), 0), "Eigenvalues are not real"

    axes[0].plot(D, 'k.')
    axes[0].plot(j, D[j], 'ro')
    axes[0].set_xlabel("Index $i$")
    axes[0].set_ylabel(r"$\lambda_i$")

    sv = np.real(S[:, j])
    vmax = np.max(np.abs(sv))
    norm = mpl.colors.Normalize(vmin=-vmax, vmax=vmax)
    cmap = plt.get_cmap('RdBu_r')

    xy = sp.centers[:, :2]
    patches = []
    for i in range(sp.N):
        circle = mpl.patches.Circle(xy[i], radius=sp.radii[i]*radius_scaling)
        patches.append(circle)
    pc = mpl.collections.PatchCollection(patches, cmap=cmap, norm=norm,
                                         edgecolors='k', linewidths=0.5)
    pc.set_array(sv)
    axes[1].add_collection(pc)
    axes[1].autoscale_view()
    axes[1].set_aspect('equal')
    axes[1].set_xlabel("$x$")
    axes[1].set_ylabel("$y$")
    fig.colorbar(pc, ax=axes[1], label=f"Eigenvector $v_{{{j}}}$")

    return fig, axes


def visualize_spectrum_complex(sp: swp.FiniteSWP1D, j, D=None, S=None, semilogy=False, axes=None, ylim=None):
    if not axes:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    if D is None or S is None:
        D, S = sp.compute_sorted_eigs_capacitance_matrix()
    # assert np.allclose(np.imag(D), 0), "Eigenvalues are not real"
    plot_eigenvalues(D, real=False, ax=axes[0])
    axes[0].plot(np.real(D[j]), np.imag(D[j]), 'ro')
    if ylim is not None:
        axes[0].set_ylim(ylim)
    if semilogy:
        sv = np.abs(S[:, j])
        axes[1].semilogy(sv, 'k-')
    else:
        sv = S[:, j]
        axes[1].plot(sv.real, 'r-')
        axes[1].plot(sv.imag, 'b-')
    return fig, axes


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
            real_symmetrisation_acceleration=real_symmetrisation_acceleratrion)
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


def get_gap(mat, k_min=1e-1, k_max=5, n_pts=1000):
    ks = np.linspace(k_min, k_max, n_pts)
    gap = np.empty(len(ks), dtype=bool)
    for i, k in enumerate(ks):
        D, S = sort_by_eva_abs(*np.linalg.eig(mat(k)))
        gap[i] = np.imag(D[0]) < 1e-5
    return gap, ks[gap]


def get_gap_counts(dp: disordered.DisorderedCommon, k_min=1e-1, k_max=5, n_pts=1000):
    ks = np.linspace(k_min, k_max, n_pts)
    block_gaps = np.zeros((len(dp.blocks), n_pts), dtype=bool)
    for i, block in enumerate(dp.blocks):
        if dp.get_physics() == "Non-reciprocal":
            def matfun(lbda): return utils_propagation.propagation_matrix_nonreciprocal_block(
                block, lbda)
        else:
            matfun = utils_propagation.propagation_matrix_block_function(
                block, subwavelength=True)
        gap_truth, gap_ks = get_gap(matfun, k_min, k_max, n_pts)
        block_gaps[i] = gap_truth

    return np.count_nonzero(block_gaps, axis=0), ks


def get_gap_changes(dp: disordered.DisorderedCommon, k_min=1e-1, k_max=5, n_pts=1000):
    counts, ks = get_gap_counts(dp, k_min, k_max, n_pts)
    changes = []
    count = 0
    changes.append((0, 0))
    for i in range(1, len(counts)):
        if counts[i] != count:
            changes.append((ks[i], counts[i]))
            count = counts[i]
    return changes


def shade_regions(dp: DisorderedClassicFiniteSWP1D, k_min=1e-3, k_max=5, n_pts=100, horizontal=False, offset=0, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    change_points = get_gap_changes(dp, k_min, k_max, n_pts)

    def colorfun(gap_count):
        D = len(dp.blocks)
        if gap_count == 0:
            return "green"
        elif gap_count == D:
            return "red"
        else:
            return "orange"

    for i in range(len(change_points)):
        k = change_points[i][0]
        gap_count = change_points[i][1]
        if i == len(change_points)-1:
            next_k = k_max
        else:
            next_k = change_points[i+1][0]
        if horizontal:
            ax.axhspan(k, next_k,  alpha=0.3, facecolor=colorfun(
                gap_count), edgecolor=None)
        else:
            ax.axvspan(k, next_k,  alpha=0.3, facecolor=colorfun(
                gap_count), edgecolor=None)

        # Create legend handles
    red_patch = mpl.patches.Patch(
        color='red', alpha=0.3, label='Shared bandgap')
    green_patch = mpl.patches.Patch(
        color='green', alpha=0.3, label='Shared pass band')
    orange_patch = mpl.patches.Patch(
        color='orange', alpha=0.3, label='Hybridisation region')

    # Add to legend
    return [red_patch, green_patch, orange_patch]


def get_region_function(dp: disordered.DisorderedCommon, k_min=1e-1, k_max=5, n_pts=1000, colors_out=True):
    gap_counts, ks = get_gap_counts(dp, k_min, k_max, n_pts)

    @np.vectorize
    def region_function(k):
        n_gaps = gap_counts[np.argmin(np.abs(ks-k))]
        if n_gaps == 0:
            return "green" if colors_out else "pass"
        elif n_gaps == len(dp.blocks):
            return "red" if colors_out else "gap"
        else:
            return "orange" if colors_out else "hybrid"
    return region_function, np.linspace(k_min, k_max, n_pts)


def plot_regions(dp: DisorderedClassicFiniteSWP1D, k_min=1e-3, k_max=5, n_pts=100, horizontal=False, offset=0, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    block_gaps, ks = get_gap_counts(dp, k_min, k_max, n_pts)
    shared_pass = (block_gaps == 0)
    shared_gap = (block_gaps == len(dp.blocks))
    hybrid = (np.logical_not(shared_pass) & np.logical_not(shared_gap))
    if horizontal:
        ax.plot(ks[shared_gap], np.zeros(np.sum(shared_gap))+offset, 'r.')
        ax.plot(ks[shared_pass], np.zeros(np.sum(shared_pass))+offset, 'b.')
        ax.plot(ks[hybrid], np.zeros(np.sum(hybrid))+offset, 'm.')
    else:
        ax.plot(np.zeros(np.sum(shared_gap))+offset, ks[shared_gap], 'r.')
        ax.plot(np.zeros(np.sum(shared_pass))+offset, ks[shared_pass], 'b.')
        ax.plot(np.zeros(np.sum(hybrid))+offset, ks[hybrid], 'm.')


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
