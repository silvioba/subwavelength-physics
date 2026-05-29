import numpy as np
import scipy as sci
import copy

from Subwavelength1D.swp import (
    FiniteSWP1D,
    PeriodicSWP1D,
)

from Subwavelength1D.classic import (
    ClassicPeriodicSWP1D,
    ClassicFiniteSWP1D,
    convert_finite_into_periodic
)

from Subwavelength1D.time_modulated import TimeModulatedFiniteSWP1D
import Utils.utils_propagation as utils_propagation
from Utils.utils_general import *

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import path, patches
from matplotlib.colors import LinearSegmentedColormap, LogNorm

from matplotlib import colormaps

from itertools import product

from collections import deque

from typing import Literal, Callable, Tuple, Self, List

from dataclasses import dataclass


plt.rcParams.update(settings.matplotlib_params)


def calculate_and_visualize_spectrum(tmswp: TimeModulatedFiniteSWP1D, j, N_fourier=4, semilogy=False, fig=None, ylim=None, xlim=None):
    if fig is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    else:
        axes = fig.get_axes()

    D, S = tmswp.compute_sorted_eigs_capacitance_matrix(
        return_eigenvectors=True, N_fourier=N_fourier)

    print(f"Folding Bound:{tmswp.big_omega/2}")
    print(f"Biggest real part eva: {D[-1]}")

    D, S = sort_by_eva_imag(D, S)

    N = tmswp.N

    S_norms = np.zeros((N, N))
    for i in range(N):
        reshaped = np.reshape(S[:(2*N_fourier+1)*N, i], (N, 2*N_fourier+1))
        S_norms[:, i] = np.linalg.norm(reshaped, axis=1)

    visualize_spectrum(D, S_norms, j, semilogy=semilogy, fig=fig)

    if ylim is not None:
        axes[0].set_ylim(ylim)
    if xlim is not None:
        axes[0].set_xlim(xlim)


def visualize_spectrum(D, S_norms, j, semilogy=False, fig=None):
    if fig is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    else:
        axes = fig.get_axes()

    axes[0].plot(np.real(D), np.imag(D), 'k.')
    axes[0].plot(np.real(D[j]), np.imag(D[j]), 'ro')
    if semilogy:
        sv = np.abs(S_norms[:, j])
        axes[1].semilogy(sv, 'k-')
    else:
        sv = np.real(S_norms[:, j])
        axes[1].plot(sv, 'k-')


def plot_logslope_vs_eva_imag(tmswp: TimeModulatedFiniteSWP1D, N_fourier=4, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()

    D, S = tmswp.compute_sorted_eigs_capacitance_matrix(
        return_eigenvectors=True, N_fourier=N_fourier)
    D, S = sort_by_eva_imag(D, S)

    N = tmswp.N

    S_norms = np.zeros((N, N))
    for i in range(N):
        reshaped = np.reshape(S[:(2*N_fourier+1)*N, i], (N, 2*N_fourier+1))
        S_norms[:, i] = np.linalg.norm(reshaped, axis=1)

    xx = np.arange(N)

    slopes = np.zeros(N)
    for i in range(N):
        slopes[i] = sci.stats.linregress(xx, np.log(S_norms[:, i])).slope

    ax.scatter(np.imag(D), slopes, s=4)
    ax.set_title(f"Modulation strength: {tmswp.epsilon_kappa}")
    ax.set_ylabel('Log slope of the mode')
    ax.set_xlabel('Imaginary part of eigenvalue')
    return ax


def plot_spectrum_with_slopecolor(tmswp: TimeModulatedFiniteSWP1D, N_fourier=4, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()

    D, S = tmswp.compute_sorted_eigs_capacitance_matrix(
        return_eigenvectors=True, N_fourier=N_fourier)

    print(f"Folding Bound:{tmswp.big_omega/2}")
    print(f"Biggest real part eva: {np.real(D[-1])}")

    D, S = sort_by_eva_imag(D, S)

    N = tmswp.N

    S_norms = np.zeros((N, N))
    for i in range(N):
        reshaped = np.reshape(S[:(2*N_fourier+1)*N, i], (N, 2*N_fourier+1))
        S_norms[:, i] = np.linalg.norm(reshaped, axis=1)

    xx = np.arange(N)

    slopes = np.zeros(N)
    for i in range(N):
        slopes[i] = sci.stats.linregress(xx, np.log(S_norms[:, i])).slope

    ax.scatter(np.real(D), np.imag(D), c=slopes,
               s=4, cmap='coolwarm', vmin=-0.25, vmax=0.25)

    # Draw colorbar
    cbar = ax.figure.colorbar(ax.collections[0], ax=ax)
    # ax.set_title(f"Modulation strength: {tmswp.epsilon_kappa}")
    # ax.set_ylabel('Log slope of the mode')
    # ax.set_xlabel('Imaginary part of eigenvalue')
    return ax, cbar
