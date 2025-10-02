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

from typing import Literal, Callable, Tuple, Self, List
from typing_extensions import override

import copy
from tqdm import tqdm


from itertools import product
from collections import deque

plt.rcParams.update(settings.matplotlib_params)


def visualize_projective_source_sink(lbda, dp, alpha=0):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    u = np.array([np.cos(alpha), np.sin(alpha)])
    dp.set_omega(lbda)

    tt = np.linspace(-np.pi, np.pi, 100)
    ax.plot(np.cos(tt), np.sin(tt), 'k--', alpha=0.5)
    ax.set_aspect('equal')

    Ptot = dp.compute_propagation_matrix(space_from_end=dp.get_sN())
    un = Ptot @ u
    un = un / un[0]
    un = un / np.linalg.norm(un)

    ax.plot(u[0], u[1], 'ko')
    ax.plot(un[0], un[1], 'mo')

    Dtot, Stot = sort_by_eva_abs(*np.linalg.eig(Ptot))
    Stot = unique_eigenvector_phases(Stot)
    # Total source / stable
    ax.plot(Stot[0, 0], Stot[1, 0], 'rx')
    # Total sink / unstable
    ax.plot(Stot[0, 1], Stot[1, 1], 'bx')

    P1 = utils_propagation.propagation_matrix_block(dp.blocks[0], lbda)
    D1, S1 = sort_by_eva_abs(*np.linalg.eig(P1))
    S1 = unique_eigenvector_phases(S1)
    # Block 1 source / stable
    ax.plot(S1[0, 0], S1[1, 0], 'r^')
    # Block 1 sink / unstable
    ax.plot(S1[0, 1], S1[1, 1], 'b^')

    P2 = utils_propagation.propagation_matrix_block(dp.blocks[1], lbda)
    D2, S2 = sort_by_eva_abs(*np.linalg.eig(P2))
    S2 = unique_eigenvector_phases(S2)
    # Block 2 source / stable
    ax.plot(S2[0, 0], S2[1, 0], 'rv')
    # Block 2 sink / unstable
    ax.plot(S2[0, 1], S2[1, 1], 'bv')
