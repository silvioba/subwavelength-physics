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


def calculate_fractal_DoS(background_block, defect_block, weights, spectum_cutoff=1.5, max_order=6, n_padding=10, p_cutoff=0):
    def get_probability_from_sequence(seq):
        return np.prod([weights[s] for s in seq])

    density_of_states = deque()
    for seq in product([0, 1], repeat=max_order):
        p = get_probability_from_sequence(seq)
        if p < p_cutoff:
            continue
        dp = disordered.DisorderedClassicFiniteSWP1D.from_blocks(
            [background_block, defect_block],
            [0]*n_padding+list(seq)+[0]*n_padding,
            v_in=1, v_out=1
        )
        DD = dp.get_spectral_range_capacitance_matrix()
        DD = DD[DD > spectum_cutoff]
        for d in DD:
            density_of_states.append((d, p))

    return list(density_of_states)


def plot_metaatoms(basic_block=None, dimer_block=None, metaatoms=None, dos_min=1.5, dos_max=3.5, ax=None, styles=None):
    if not basic_block:
        basic_block = ((2,), (2,))

    if not dimer_block:
        dimer_block = ((1, 1), (1, 2))

    if not metaatoms:
        metaatoms = [[1]*i for i in range(1, 5)] + \
            [[1, 0, 1]] + [[1, 1, 0, 1, 1]]

    if not styles:
        styles = [("k", "--"), ("r", "--"), ("g", "--"), ("b", "--"),
                  ("k", ":"), ("r", ":"), ("g", ":"), ("b", ":")]

    for i, metaatom in enumerate(metaatoms):
        metaatom = tuple(metaatom)
        D = get_metaatom_solutions(
            basic_block, dimer_block, metaatom, midgap=dos_min)
        label = '('+','.join("2" if s == 1 else "1" for s in metaatom)+')'
        color, linestyle = styles[i]
        for d in D:
            ax.axvline(d, color=color, linestyle=linestyle,
                       alpha=0.3, label=label)

    handles, labels = ax.get_legend_handles_labels()
    labels, ids = np.unique(labels, return_index=True)
    handles = [handles[i] for i in ids]
    return handles, labels


def ecdf_integrated_difference(D1, D2, a, b, n=1000):
    ecdf1 = sci.stats.ecdf(D1)
    ecdf2 = sci.stats.ecdf(D2)

    pts = np.linspace(a, b, n)
    return np.trapz((ecdf1.cdf.evaluate(pts)-ecdf2.cdf.evaluate(pts))**2, pts)


def plot_iDoS(sp: ClassicFiniteSWP1D = None, D=None, dos_min=1.5, dos_max=3.5, label=None, fig=None, ax=None, color='k'):
    if fig is None:
        fig, ax = plt.subplots(1, 1, figsize=(
            settings.figure_width, settings.figure_height), constrained_layout=True)

    if D is None:
        assert sp is not None
        D = sp.get_spectral_range_capacitance_matrix(
            select='v', select_range=(dos_min, dos_max))

    ax.ecdf(D, color=color, label=label)


def plot_DoS(sp: ClassicFiniteSWP1D = None, D=None, basic_block=None, dimer_block=None, metaatoms=None, dos_min=1.5, dos_max=3.5, bins=200, fig=None, ax=None, styles=None):
    if fig is None:
        fig, ax = plt.subplots(1, 1, figsize=(
            settings.figure_width, settings.figure_height), constrained_layout=True)

    handles, lables = plot_metaatoms(basic_block, dimer_block, metaatoms,
                                     dos_min, dos_max, ax, styles)
    if D is None:
        assert sp is not None
        D = sp.get_spectral_range_capacitance_matrix(
            select='v', select_range=(dos_min, dos_max))

    hist, edges = np.histogram(D, bins=bins)
    binsize = edges[1] - edges[0]
    hist = hist / (len(D) * binsize)

    ax.stairs(hist, edges, color='k', fill=True, zorder=10)

    # ax.hist(D[D > dos_min], density=True, bins=bins, zorder=10)
    ax.set_yscale("log")

    return handles, lables


def plot_averaged_DoS(get_realization: Callable, basic_block=None, dimer_block=None, metaatoms=None, dos_min=1.5, dos_max=3.5, bins=400, n_realizations=10, fig=None, styles=None):
    if not fig:
        fig, ax = plt.subplots(1, 1, figsize=(
            settings.figure_width, settings.figure_height), constrained_layout=True)
    else:
        ax = fig.gca()

    Ds = []
    for i in range(n_realizations):
        sp = get_realization()
        D = sp.get_sorted_spectrum_capacitance_matrix(
            select='v', select_range=(dos_min, dos_max))
        Ds.extend(D)

    Ds = np.array(Ds, dtype=float)

    ax.hist(Ds, bins=bins, zorder=0)
    ax.set_yscale("log")

    plot_metaatoms(basic_block, dimer_block, metaatoms,
                   dos_min, dos_max, ax, styles)

    ax.set_xlabel("Eigenvalue $\\lambda$")
    ax.set_ylabel("Density of states")


def plot_hyperuniform_fft_autocovariance(N=100000, avg_window=1000, fig=None):
    if fig is None:
        fig, ax = plt.subplots()
    else:
        ax = fig.gca()

    random_seq = np.random.choice([0, 1], size=N)
    random_acvr = autocovariance(random_seq)

    hyperuniform_seq = construct_hyperuniform_binary_sequence(N//2)
    hyperuniform_acvr = autocovariance(hyperuniform_seq)

    bound_seq = construct_bound_length_sequence(2, [0.5, 0.5], [2, 2], N)
    bound_acvr = autocovariance(bound_seq)

    softmax_seq = construct_softmax_uniformed_sequence(2, N, beta=5)
    softmax_acvr = autocovariance(softmax_seq)

    random_movavg = np.convolve(np.abs(np.fft.ifft(random_acvr)),
                                np.ones(avg_window), mode='valid')
    hyperuniform_movavg = np.convolve(np.abs(np.fft.ifft(hyperuniform_acvr)),
                                      np.ones(avg_window), mode='valid')
    bound_movavg = np.convolve(np.abs(np.fft.ifft(bound_acvr)),
                               np.ones(avg_window), mode='valid')
    softmax_movavg = np.convolve(np.abs(np.fft.ifft(softmax_acvr)),
                                 np.ones(avg_window), mode='valid')

    n_pts = len(random_movavg)
    x = np.linspace(0, 2*np.pi, n_pts)

    ax.plot(x, random_movavg, label="Random")
    ax.plot(x, hyperuniform_movavg, label="Chunk")
    ax.plot(x, bound_movavg, label="Bound length")
    ax.plot(x, softmax_movavg, label="Softmax")
    ax.set_xlabel("Wave number $k$")
    ax.set_ylabel(r"$\widehat{K}(k)$")
    ax.legend(loc="best")
