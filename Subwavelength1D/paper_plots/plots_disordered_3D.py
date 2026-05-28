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

from Subwavelength3D.classic_finite import *
from Subwavelength3D.fmm import *

from typing import Literal, Callable, Tuple, Self, List
from typing_extensions import override

import copy
from tqdm import tqdm


from itertools import product
from collections import deque
plt.rcParams.update(settings.matplotlib_params)


def create_linear_SWP3D(N, sep=1.0, radius=1.0):
    center_sep = sep + 2*radius

    radii = np.array([radius]*N)
    centers = np.array([[0, 0, z*center_sep] for z in range(N)])
    return ClassicFiniteSWP3D(radii=radii, centers=centers)


def create_linear_dimer_SWP3D(N_cells, radius=1.0, intra_sep=1.0, inter_sep=5.0):
    radii = np.array([radius]*2*N_cells)

    spacings = [2*radius + intra_sep, 2*radius + inter_sep]*N_cells
    z_positions = np.cumsum([0]+spacings)[:-1]
    centers = np.array([[0, 0, z] for z in z_positions])
    return ClassicFiniteSWP3D(radii=radii, centers=centers)


def create_screen_SWP3D(Nx, Ny, sep_x=1.0, sep_y=1.0, radius=1.0):
    center_sep_x = sep_x + 2*radius
    center_sep_y = sep_y + 2*radius

    radii = np.array([radius]*Nx*Ny)
    centers = np.array([[x*center_sep_x, y*center_sep_y, 0]
                       for x in range(Nx) for y in range(Ny)])
    return ClassicFiniteSWP3D(radii=radii, centers=centers)


def create_double_screen_SWP3D(Nx, Ny, sep_x=1.0, sep_y=1.0, sep_z=1.0, radius=1.0):
    center_sep_x = sep_x + 2*radius
    center_sep_y = sep_y + 2*radius
    center_sep_z = sep_z + 2*radius

    centers = np.zeros((Nx, Ny, 2, 3))

    for x in range(Nx):
        for y in range(Ny):
            centers[x, y, 0] = [x*center_sep_x,
                                y*center_sep_y, 0]
            centers[x, y, 1] = [x*center_sep_x,
                                y*center_sep_y, center_sep_z]

    centers_flat = centers.reshape((-1, 3))
    radii_flat = np.array([radius]*(Nx*Ny*2))
    return ClassicFiniteSWP3D(radii=radii_flat, centers=centers_flat)


def periodic_2D_constructor(N1, N2, v1=None, v2=None, unit_cell=None):
    """Constructs a periodic 2D lattice of 3D resonators.

    Args:
        N1 (int): Number of unit cells along the v1 direction.
        N2 (int): Number of unit cells along the v2 direction.
        v1 (np.ndarray | int | float | None): Lattice vector for the first direction.
            If scalar, interpreted as (v1, 0, 0). Defaults to (3, 0, 0).
        v2 (np.ndarray | int | float | None): Lattice vector for the second direction.
            If scalar, interpreted as (0, v2, 0). Defaults to (0, 3, 0).
        unit_cell (tuple | None): Unit cell definition as a tuple of
            ((x1, y1, z1), r1), ..., ((xd, yd, zd), rd)) pairs where
            (xi, yi, zi) is the resonator offset within the unit cell and ri
            is the resonator radius. Defaults to a single resonator at the
            origin with radius 1.

    Returns:
        ClassicFiniteSWP3D: The assembled periodic system.
    """
    if v1 is None:
        v1 = np.array([3, 0, 0])
    elif isinstance(v1, (int, float)):
        v1 = np.array([v1, 0, 0])
    else:
        v1 = np.asarray(v1, dtype=float)
    if v2 is None:
        v2 = np.array([0, 3, 0])
    elif isinstance(v2, (int, float)):
        v2 = np.array([0, v2, 0])
    else:
        v2 = np.asarray(v2, dtype=float)
    if unit_cell is None:
        unit_cell = (((0, 0, 0), 1.0),)

    unit_cell_offset = np.array(
        [np.asarray(offset, dtype=float) for offset, _ in unit_cell])
    unit_cell_radii = np.array([r for _, r in unit_cell])
    N_unit = len(unit_cell)

    centers = np.zeros((N1, N2, N_unit, 3))
    radii = np.zeros((N1, N2, N_unit))
    for i in range(N1):
        for j in range(N2):
            centers[i, j] = unit_cell_offset + i*v1 + j*v2
            radii[i, j] = unit_cell_radii

    centers_flat = centers.reshape((-1, 3))
    radii_flat = radii.reshape((-1,))
    return ClassicFiniteSWP3D(radii=radii_flat, centers=centers_flat)


def create_linear_blockdisordered_SWP3D(blocks, idxs) -> ClassicFiniteSWP3D:
    radii = []
    centers = []
    current_z = 0.0
    for i in idxs:
        block_len = len(blocks[i][0])
        for j in range(block_len):
            r = blocks[i][0][j]
            s = blocks[i][1][j]
            radii.append(r)
            centers.append((0.0, 0.0, current_z))
            current_z += s

    return ClassicFiniteSWP3D(centers=np.array(centers), radii=np.array(radii))


def blockdisordered_2D_constructor(N1, N2, v1, v2, blocks, weights=None, seed=42):
    """Constructs a disordered 2D lattice of 3D resonators by randomly assigning blocks to unit cells.

    At each lattice site (i, j), a block is randomly chosen from the provided list
    and its resonators are placed at the corresponding offsets shifted by i*v1 + j*v2.

    Args:
        N1 (int): Number of unit cells along the v1 direction.
        N2 (int): Number of unit cells along the v2 direction.
        v1 (np.ndarray | int | float): Lattice vector for the first direction.
            If scalar, interpreted as (v1, 0, 0).
        v2 (np.ndarray | int | float): Lattice vector for the second direction.
            If scalar, interpreted as (0, v2, 0).
        blocks (list): List of block definitions. Each block is a tuple of
            ((x1, y1, z1), r1), ..., ((xd, yd, zd), rd)) pairs where
            (xi, yi, zi) is the resonator offset within the unit cell and ri
            is the resonator radius.
        weights (list[float] | None): Probabilities for choosing each block.
            Must sum to 1 and have the same length as blocks. If None, uniform
            probabilities are used.
        seed (int): Random seed for reproducibility.

    Returns:
        ClassicFiniteSWP3D: The assembled disordered system.
    """
    if isinstance(v1, (int, float)):
        v1 = np.array([v1, 0, 0])
    else:
        v1 = np.asarray(v1, dtype=float)
    if isinstance(v2, (int, float)):
        v2 = np.array([0, v2, 0])
    else:
        v2 = np.asarray(v2, dtype=float)

    if weights is not None:
        assert len(weights) == len(
            blocks), "len(weights) must equal len(blocks)"
        assert np.isclose(sum(weights), 1), "weights must sum to 1"

    np.random.seed(seed)
    block_indices = np.random.choice(len(blocks), size=(N1, N2), p=weights)

    centers = []
    radii = []
    for i in range(N1):
        for j in range(N2):
            origin = i * v1 + j * v2
            block = blocks[block_indices[i, j]]
            for offset, r in block:
                centers.append(origin + np.asarray(offset, dtype=float))
                radii.append(r)

    return ClassicFiniteSWP3D(
        radii=np.array(radii),
        centers=np.array(centers).reshape((-1, 3)),
    )


def get_capacitance_like_matrix(N, alpha=1, p=0.5, seed=42, normed=True, fill_diagonal=True):
    np.random.seed(seed)
    C = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(N):
            if i != j:
                C[i, j] = -1/(abs(i-j)**alpha)

    scaling = np.array([(i+1)**alpha for i in range(N)])
    if fill_diagonal:
        diag = 2*sum(1/scaling)
        np.fill_diagonal(C, diag)

    if normed:
        C = C / np.linalg.norm(C, ord='fro')

    V = np.random.uniform(1-p, 1+p, size=N)
    return C, V


def get_random_banded_GOE(N, alpha=1, seed=42):
    np.random.seed(seed)
    M = np.zeros((N, N), dtype=float)
    for i in range(-N+1, N):
        l = N-np.abs(i)
        band = np.random.normal(0, 1/(1+np.abs(i))**alpha, size=l)
        M = M + np.diag(band, k=i)
    return (M+M.T)/2


def get_average_IPR(S):
    ipr = np.linalg.norm(S, axis=0, ord=4)**4
    return np.mean(ipr), np.std(ipr)


def get_IPR_slope_at_noise_for_swp(swps: List[ClassicFiniteSWP3D], p: float = 0):
    NN = []
    mean_iprs = []
    for swp in swps:
        print(swp.N)
        NN.append(swp.N)
        np.random.seed(42)
        v_in = 1+np.random.uniform(-p, p, swp.N)
        swp.v_in = v_in
        D, S = swp.compute_sorted_eigs_capacitance_matrix()
        mean_ipr, std_ipr = get_average_IPR(S)
        mean_iprs.append(mean_ipr)
    slope, _ = np.polyfit(np.log(NN), np.log(mean_iprs), 1)
    return slope


def get_IPR_slope_at_noise_for_generic_C(NN, alpha: float = 1, p: float = 0, fill_diagonal=True):
    mean_iprs = []
    for N in NN:
        # print(N)
        C, V = get_capacitance_like_matrix(
            N, alpha=alpha, p=p, fill_diagonal=fill_diagonal)
        D, S = sci.linalg.eigh(C, b=np.diag(1/V))
        mean_ipr, std_ipr = get_average_IPR(S)
        mean_iprs.append(mean_ipr)
    slope, _ = np.polyfit(np.log(NN), np.log(mean_iprs), 1)
    return slope


def get_IPR_slope_at_alpha_for_generic_banded(NN, alpha: float = 1):
    mean_iprs = []
    for N in NN:
        G = get_random_banded_GOE(N, alpha=alpha)
        D, S = sci.linalg.eigh(G)
        mean_ipr, std_ipr = get_average_IPR(S)
        mean_iprs.append(mean_ipr)
    slope, _ = np.polyfit(np.log(NN), np.log(mean_iprs), 1)
    return slope


def get_C_decay(C, i, cutoff=0):
    ii = []
    cc = []
    N = C.shape[0]

    for j in range(cutoff, N-cutoff):
        if i != j:
            ii.append(np.abs(i-j))
            cc.append(C[i, j])

    return ii, cc


def get_decay_rate_per_row(C, cutoff=0):
    decay_rates = []
    N = C.shape[0]
    for i in range(N):
        if i % 100 == 0:
            print(f"Processing row {i}")
        ii, cc = get_C_decay(C, i, cutoff=cutoff)
        if ii and cc:
            slope, _ = np.polyfit(np.log(ii), np.log(np.abs(cc)), 1)
            decay_rates.append(-slope)
    return decay_rates


def flatten_index(ix, iy, Nx, Ny):
    return ix*Ny + iy


def unflatten_index(i, Nx, Ny):
    ix = i // Ny
    iy = i % Ny
    return ix, iy


def get_C_decay_screen(C, ix, iy, Nx, Ny):
    ii = []
    cc = []
    N = C.shape[0]

    for jx in range(Nx):
        for jy in range(Ny):
            if (ix, iy) != (jx, jy):
                ii.append(np.sqrt((ix-jx)**2 + (iy-jy)**2))
                cc.append(C[flatten_index(ix, iy, Nx, Ny),
                          flatten_index(jx, jy, Nx, Ny)])

    return ii, cc


def get_decay_rate_per_row_screen(C, Nx, Ny):
    decay_rates = []
    N = Nx * Ny
    for i in range(N):
        if i % 100 == 0:
            print(f"Processing row {i}")
        ix, iy = unflatten_index(i, Nx, Ny)
        ii, cc = get_C_decay_screen(C, ix, iy, Nx, Ny)
        if ii and cc:
            slope, _ = np.polyfit(np.log(ii), np.log(np.abs(cc)), 1)
            decay_rates.append(slope)
    return decay_rates
