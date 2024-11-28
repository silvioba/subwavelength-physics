#  Copyright (c) 2023.
#  Copyright held by Silvio Barandun ETH Zurich
#  All rights reserved.


import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as colors

from Subwavelength1D.settings import settings

plt.rcParams.update(settings.matplotlib_params)

def unique_eigenvector_phases(S):
    S = S / S[0, :]
    S = S / np.linalg.norm(S, axis=0)
    return S

def sort_by_EM_middle_localization(D, S):
    N = D.shape[0]
    sidx = np.argsort(np.linalg.norm(S[N // 3 : 2 * N // 3, :], axis=0))
    return D[sidx], S[:, sidx]


def sort_by_EM_localization(D, S):
    sidx = np.argsort(-np.linalg.norm(S, axis=0, ord=1))
    return D[sidx], S[:, sidx]


def sort_by_EF_real(D, S):
    sidx = np.argsort(np.real(D))
    return D[sidx], S[:, sidx]


def sort_by_EF_imag(D, S):
    sidx = np.argsort(np.imag(D))
    return D[sidx], S[:, sidx]


def sort_by_EF_abs(D, S):
    sidx = np.argsort(np.abs(D))
    return D[sidx], S[:, sidx]


def sort_by_EM_first_val(D, S):
    sidx = np.argsort(np.abs(S[0, :]))
    return D[sidx], S[:, sidx]

def plot_eigenvalues(D, colorfunc=None, ax=None):
    if ax is None:
            fig, ax = plt.subplots()
    if colorfunc:
        ax.scatter(np.arange(len(D)), D, c=colorfunc(D), marker=".")
    else:
        ax.scatter(np.arange(len(D)), D,  c="black", marker=".")
    ax.set_xlabel("Site index $i$")
    ax.set_ylabel(r"$\lambda_i$")