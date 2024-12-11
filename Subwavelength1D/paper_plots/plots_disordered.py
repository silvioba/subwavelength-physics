import numpy as np

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import matplotlib.cm as cm

from Utils.settings import settings as settings

from Utils.utils_general import *

import Subwavelength1D.classic as classic
import Subwavelength1D.disordered as disordered
from typing import Literal, Callable, Tuple, Self, List, override

import copy

plt.rcParams.update(settings.matplotlib_params)


def plot_variance_band_functions(
    dwp: disordered.DisorderedClassicFiniteSWP1D,
    s_N: float = 1,
    nalpha: int = 100,
    ax: Axes | None = None,
    semilogy: bool = False,
    generalised: bool = False,
    only_background: bool = False,
    **kwargs,
):
    """
    Plots the variance of band functions.

    Args:
        s_N (float, optional): Scaling factor for periodic conversion. Defaults to 1.
        nalpha (int, optional): Number of alpha values to sample. Defaults to 100.
        ax (Axes | None, optional): Matplotlib Axes object to plot on. Defaults to None.
        semilogy (bool, optional): Whether to use a logarithmic scale for the y-axis. Defaults to False.
        generalised (bool, optional): Whether to use the generalised capacitance matrix. Defaults to False.
        only_background (bool, optional): Whether to plot only the background. Defaults to False.

    Returns:
        Axes: Matplotlib Axes object with the plot.
    """
    pwp = classic.convert_finite_into_periodic(dwp, s_N=s_N)
    alphas, bands = pwp.get_band_data(generalised=generalised, nalpha=nalpha)
    bands = np.real(bands)

    variances = np.var(bands, axis=0) * dwp.N**2
    variance_lowest = variances[0]
    means = np.mean(bands, axis=0)

    mask_big_jump = np.diff(means) > 0.5
    idxs = np.arange(len(mask_big_jump))[mask_big_jump]
    idxs += 1

    variances_trans = variances

    if ax is None:
        fig, ax = plt.subplots()
    if only_background:
        idxs = np.insert(idxs, 0, 0)
        idxs = np.append(idxs, -1)

        for i in range(len(idxs) - 1):
            means_sel = means[idxs[i] : idxs[i + 1]]
            if len(means_sel) == 0:
                continue
            X, Y = np.meshgrid(
                means_sel,
                np.array(
                    [kwargs.get("vlims", [0, 1])[0], kwargs.get("vlims", [0, 1])[1]]
                ),
            )
            Z = np.ones_like(Y, dtype=float)
            for j in range(Z.shape[0]):
                Z[j, :] = variances_trans[idxs[i] : idxs[i + 1]]

            cmap, norm = custom_colormap_with_lognorm(
                variance_lowest,
                vmin=1e-17,
                vmax=np.max(variances_trans),
            )
            pcm = ax.pcolormesh(
                X,
                Y,
                Z,
                norm=(norm if semilogy else None),
                cmap=cmap,
                shading="nearest",
            )
        if kwargs.get("colorbar"):
            plt.colorbar(pcm, ax=ax, extend="max")

    else:
        if semilogy:
            ax.semilogy(means, variances, "k.")
        else:
            ax.plot(means, variances, "k.")
    return ax


def custom_colormap_with_lognorm(a, vmin, vmax, include_white=False):
    """
    Generates a custom colormap with graded red above `a` and graded blue below `a`,
    with logarithmic normalization.

    Parameters:
    a (float): The threshold value for the color transition (in log space).
    vmin (float): The minimum value for normalization (must be > 0).
    vmax (float): The maximum value for normalization (must be > 0).

    Returns:
    tuple: (LinearSegmentedColormap, LogNorm) for custom plotting.
    """
    if include_white:
        colors = [
            (0.0, (1.0, 1.0, 1.0)),  # Very light blue (near-white)
            (0.1, (0.85, 0.85, 1.0)),  # Very light blue (near-white)
            (0.4, (0.0, 0.0, 1.0)),  # Blue
            (0.5, (0.5, 0.0, 0.5)),  # Purple at the transition point
            (1.0, (1.0, 0.0, 0.0)),  # Red
        ]
    else:
        colors = [
            (0.0, (0.85, 0.85, 1.0)),  # Very light blue (near-white)
            (0.4, (0.0, 0.0, 1.0)),  # Blue
            (0.5, (0.5, 0.0, 0.5)),  # Purple at the transition point
            (1.0, (1.0, 0.0, 0.0)),  # Red
        ]

    # Create the colormap
    cmap = LinearSegmentedColormap.from_list("custom_colormap", colors)

    # Log normalization with midpoint adjustment
    class MidPointLogNorm(LogNorm):
        def __init__(self, vmin, vmax, midpoint, clip=False):
            self.midpoint = midpoint
            super().__init__(vmin, vmax, clip)

        def __call__(self, value, clip=None):
            # Adjust log normalization for smooth scaling around the midpoint
            log_v = np.log10(value)
            log_vmin, log_vmax, log_midpoint = map(
                np.log10, [self.vmin, self.vmax, self.midpoint]
            )
            norm_value = (log_v - log_vmin) / (log_vmax - log_vmin)
            norm_value = np.where(
                log_v <= log_midpoint,
                (log_v - log_vmin) / (log_midpoint - log_vmin) * 0.5,
                0.5 + (log_v - log_midpoint) / (log_vmax - log_midpoint) * 0.5,
            )
            return np.ma.masked_array(norm_value)

    return cmap, MidPointLogNorm(vmin=vmin, vmax=vmax, midpoint=a)


def plot_band_function_variance_as_color(
    pwp: classic.PeriodicSWP1D,
    fig: Figure | None = None,
    ax: Axes | None = None,
    generalised: bool = True,
    nalpha: int = 100,
    xticks: bool = True,
    yticks: bool = True,
) -> Tuple[Figure, Axes]:
    """
    Plots the band function of a classic quasi periodic problem coloring the bands based on their variance

    Args:
        pwp (classic.PeriodicSWP1D): Classic Periodic Wave Problem
        fig (Figure | None): matplotlib figure to plot on. Defaults to None
        ax (Axes | None): matplotlib axes to plot on. Defaults to None
        generalised (bool, optional): whether to use the generalised capacitance matrix. Defaults to True.
        nalpha (int, optional): Number of quasifrequencies to sample in [-np.pi, np.pi). Defaults to 100.
        xticks (bool, optional): whether to plot xticks. Defaults to True.
        yticks (bool, optional): whether to plot yticks. Defaults to True.

    Returns:
        Tuple[Figure, Axes]: Matplotlib figure and axes plotted on
    """
    alphas, bands = pwp.get_band_data(generalised, nalpha)
    vars = np.var(bands, axis=0)
    if ax is None:
        fig, ax = plt.subplots(figsize=settings.figure_size)
    cmap, norm = custom_colormap_with_lognorm(
        vars[0],
        vmin=1e-14,
        vmax=np.max(vars),
    )

    for i in range(bands.shape[1]):  # Loop through each line
        ax.plot(alphas, bands[:, i], color=cmap(norm(vars[i])))

    if not yticks:
        ax.set_yticks([])
    if not xticks:
        ax.set_xticks([])
    else:
        ax.set_xticks([-np.pi, 0, np.pi], [r"$-\pi$", r"$0$", r"$\pi$"])

    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    return fig, ax


def plot_variance_vs_localisation(
    fwp: classic.ClassicFiniteSWP1D,
    s_N: int,
    fig: Figure | None = None,
    ax: Axes | None = None,
    generalised: bool = True,
    nalpha: int = 100,
    c: str | None = None,
) -> Tuple[Figure, Axes]:
    """
    For a classical finite system of resonators plots the variance of the band functions of the corresponding quasi periodic system against the localisation degree (in form of np.linalg.norm(ord=np.inf)) of the corresponding eigenvector

    Args:
        fwp (classic.ClassicFiniteSWP1D): Classic Finite subwavelength problem
        s_N (int): Distance to be imposed between cells in the quasiperiodic problem
        fig (Figure | None, optional): matplotlib figure to plot on. Defaults to None.
        ax (Axes | None, optional): matplotlib axes to plot on. Defaults to None.
        generalised (bool, optional):  whether to use the generalised capacitance matrix. Defaults to True.
        nalpha (int, optional): number of quasifrequencies to sample in [-np.pi, np.pi). Defaults to 100.
        c (str | None, optional): color and marker to use for plot. Defaults to None.

    Returns:
        Tuple[Figure, Axes]: scatter plot of the variance versus localisation with semilogx axes
    """
    D, S = fwp.get_sorted_eigs_capacitance_matrix(generalised=generalised)
    pwp = classic.convert_finite_into_periodic(finite_problem=fwp, s_N=s_N)
    alphas, bands = pwp.get_band_data(generalised, nalpha)
    vars = np.var(bands, axis=0)
    ax.semilogx(vars, np.linalg.norm(S, ord=np.inf, axis=0), c)
    return fig, ax


def plot_defect(
    dwp: disordered.DisorderedClassicFiniteSWP1D,
    p: float,
    j: int,
    fig: Figure | None = None,
    axs: Axes | None = None,
    selection_criteria: List = None,
    colors: List = None,
) -> Tuple[Figure, Axes]:
    """
    Plots on the left the eigenvalues with highlight on the ones that enter the band gap. On the right the two eigenvectors corresponding to the eigenvalues in the band gap

    Args:
        dwp (disordered.DisorderedClassicFiniteSWP1D): Disordered Classic subwavelength problem
        p (float): perturbation to be applied to the wave speed in resonator j
        j (int): resonator index to perturb
        fig (Figure | None, optional): matplotlib figure to plot on. Defaults to None.
        axs (Axes | None, optional): matplotlib axis to plot on needs to have 2 columns and 1 row. Defaults to None.
        selection_criteria (List): the selected eigenvalues are the ones closes to the elements of this list
        colors (List): colors to use for the selected eigenvalues

    Returns:
        Tuple[Figure, Axes]: Figure with two subfigures
    """
    if not fig and not axs:
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    dwp.v_in[j] = p

    capmat = dwp.get_generalised_capacitance_matrix()

    D, S = np.linalg.eig(capmat)
    idx = np.argsort(D.real)
    D = D[idx]
    S = S[:, idx]

    for i, sel in enumerate(selection_criteria):
        defect_idx = np.argmin(np.abs(D - sel))

        defect_eva = D[defect_idx]

        print(f"First selected eigenvalue {defect_eva} has index {defect_idx}")
        axes[0].plot(D, "ko", mfc="none")
        axes[0].plot(defect_idx, defect_eva, f"{colors[i]}x")
        axes[1].semilogy(np.abs(S[:, defect_idx]), f"{colors[i]}-")

    return fig, axes


def plot_double_defect(
    dwp: disordered.DisorderedClassicFiniteSWP1D,
    epss: np.ndarray,
    idx_perturbations: List,
    generalised=True,
):
    """
    Plots the eigenvalue separation of the two largest eigenvalues depending on the perturbation size

    Args:
        dwp (disordered.DisorderedClassicFiniteSWP1D): Disordered Classic Finite subwavelength problem to perturb
        epss (np.ndarray): epsilons used for perturbation
        idx_perturbations (List): indices to perturb
        generalised (bool, optional): whether to use the generalised capacitance matrix. Defaults to True.
    """

    Ds_bottom = np.zeros((epss.shape[0], 2))
    v_in0 = copy.deepcopy(dwp.v_in)

    for i, eps in enumerate(epss):
        v_in = copy.deepcopy(v_in0)
        for j in idx_perturbations:
            v_in[j] = v_in[j] + eps
        dwp.v_in = v_in

        D, S = dwp.get_sorted_eigs_capacitance_matrix(generalised=generalised)
        Ds_bottom[i, :] = D[-1], D[-2]

    fig, ax = plt.subplots(figsize=settings.figure_sizeh)

    _ = ax.plot(epss, Ds_bottom[:, 0], "b", label=r"$\lambda_{N}$")
    _ = ax.plot(epss, Ds_bottom[:, 1], "r", label=r"$\lambda_{N-1}$")
    _ = ax.set_xlabel(r"Perturbation strength $\eta$")
    _ = ax.legend()
    return fig, ax


def plot_defect_offshoot(
    dp: disordered.DisorderedClassicFiniteSWP1D,
    perturbation_indicies: List,
    selection_criteria: List[float],
    band_edges: List[Tuple],
    p_max: float = 5,
    n_ps: int = 20,
    log: bool = False,
    axes=None,
    marker_list=None,
):
    """
    Shows the defect mode frequeny dependence on the kind of block that the perturbations affects
    """
    if axes is None:
        fig, axes = plt.subplots(
            1,
            len(perturbation_indicies),
            figsize=(6 * len(perturbation_indicies), 6),
            squeeze=True,
            sharey=False,
        )

    if marker_list is None:
        marker_list = ["r-", "b-", "g-", "k-"]

    for ai, j in enumerate(perturbation_indicies):
        ps = np.linspace(0, p_max, n_ps)
        defect_frequencies = np.zeros((n_ps, len(selection_criteria)))
        for pi, p in enumerate(ps):
            dp_defected = copy.deepcopy(dp)
            dp_defected.v_in[j] += p

            capmat = dp_defected.get_generalised_capacitance_matrix()
            D, S = np.linalg.eig(capmat)
            idx_D = np.argsort(D.real)
            D = D[idx_D]
            S = S[:, idx_D]

            for mi, midgap in enumerate(selection_criteria):
                defect_idx = np.argmin(np.abs(D - midgap))
                defect_eve = D[defect_idx]
                defect_frequencies[pi, mi] = defect_eve

        for dfi in range(len(selection_criteria)):
            axes[ai].plot(ps, defect_frequencies[:, dfi], marker_list[dfi])
        # axes[ai].plot(ps, np.sum(defect_frequencies, axis=1), "k-")
        axes[ai].semilogy() if log else None
        for band_edge in band_edges:
            axes[ai].fill_between(
                ps, band_edge[0], band_edge[1], color="gray", alpha=0.5, edgecolor=None
            )
        axes[ai].set_ylim(bottom=band_edges[0][0])
        axes[ai].legend(
            ["Lower eigenvalue", "Upper eigenvalue", "Bands"],
            loc="upper left",
            frameon=False,
        )

    for ax in axes:
        ax.set_xlabel("Positive defect $\\eta$")
    axes[0].set_ylabel("Defect eigenvalues")
    return axes


def plot_variance_density_histogram(
    dp: disordered.DisorderedClassicFiniteSWP1D,
    nalpha: int = 10,
    p: float = 0.0,
    p_sampling="uniform",
    perturb_param="material",
    n_realizations=1,
    bins=None,
    semilogy=True,
    sqrt_eva=False,
    ax=None,
    **kwargs,
):
    if not bins:
        bins = dp.N // 2

    pwp = classic.convert_finite_into_periodic(dp, dp.blocks[dp.idxs[-1]][1][-1])

    bands_total = np.zeros((nalpha, dp.N))

    def get_average_perturbed_bands(p):
        band_realizations = np.zeros((n_realizations, nalpha, dp.N))
        for i in range(n_realizations):
            pwp_perturbed = pwp.get_pertubed_copy(
                p, perturb_param=perturb_param, p_sampling=p_sampling
            )
            alphas, bands = pwp_perturbed.get_band_data(generalised=True, nalpha=nalpha)
            band_realizations[i, :, :] = np.real(bands)
        return np.mean(band_realizations, axis=0)

    bands_total = get_average_perturbed_bands(p)

    variances = np.var(bands_total, axis=0) * dp.N**2
    variance_lowest = variances[0]

    means = np.mean(bands_total, axis=0)

    if sqrt_eva:
        means = np.sqrt(means)

    if ax is None:
        fig, ax = plt.subplots()

    mean_hist, bin_edges = np.histogram(
        means, bins=bins, range=(0, np.max(means) + 1e-3)
    )
    digitized = np.digitize(means, bin_edges)
    average_variance_over_bin = np.zeros_like(mean_hist, dtype=float)

    for i in range(dp.N):
        average_variance_over_bin[digitized[i] - 1] += variances[i]

    average_variance_over_bin /= mean_hist

    cmap, norm = custom_colormap_with_lognorm(
        variance_lowest,
        vmin=np.min(variances),
        vmax=np.max(variances),
    )

    pcm = ax.bar(
        bin_edges[:-1],
        mean_hist,
        width=bin_edges[1] - bin_edges[0],
        color=cmap(norm(average_variance_over_bin)),
        edgecolor="black",
    )

    if kwargs.get("colorbar"):
        cmap2, norm = custom_colormap_with_lognorm(
            variance_lowest,
            vmin=np.min(variances),
            vmax=np.max(variances),
        )
        mp = plt.cm.ScalarMappable(norm=norm, cmap=cmap2)
        cbar = plt.colorbar(mp, ax=ax)


def plot_variance_perturbation_heatmap(
    dp: disordered.DisorderedClassicFiniteSWP1D,
    nalpha=10,
    p_max=0.5,
    n_ps=10,
    p_sampling="uniform",
    perturb_param="material",
    n_realizations=10,
    gap_threshold=1,
    gap_bound=10,
    semilogy=False,
    ax=None,
    **kwargs,
):
    pwp = classic.convert_finite_into_periodic(dp, dp.blocks[dp.idxs[-1]][1][-1])

    bands_total = np.zeros((n_ps, nalpha, dp.N))

    def get_average_perturbed_bands(p):
        band_realizations = np.zeros((n_realizations, nalpha, dp.N))
        for i in range(n_realizations):
            pwp_perturbed = pwp.get_pertubed_copy(
                p, perturb_param=perturb_param, p_sampling=p_sampling
            )
            alphas, bands = pwp_perturbed.get_band_data(generalised=True, nalpha=nalpha)
            band_realizations[i, :, :] = np.real(bands)
        return np.mean(band_realizations, axis=0)

    ps = np.linspace(0, p_max, n_ps)
    for i, p in enumerate(ps):
        bands_total[i] = get_average_perturbed_bands(p)

    variances = np.var(bands_total, axis=1) * dp.N**2
    variance_lowest = variances[0, 0]

    means = np.mean(bands_total, axis=1)

    if ax is None:
        fig, ax = plt.subplots(figsize=settings.figure_size)

    X = means
    Y = np.repeat(ps[:, np.newaxis], dp.N, axis=1)
    Z = variances

    cmap, norm = custom_colormap_with_lognorm(
        variance_lowest,
        vmin=np.min(variances) / 10**2,
        vmax=np.max(variances),
    )

    gap_indices = np.logical_and(
        np.logical_or(
            np.append(
                np.abs(X[:, 1:] - X[:, :-1]) > gap_threshold, [[False]] * n_ps, axis=1
            ),
            np.append(
                [[False]] * n_ps, np.abs(X[:, 1:] - X[:, :-1]) > gap_threshold, axis=1
            ),
        ),
        X < gap_bound,
    )
    Z[gap_indices] = np.min(variances) / 10**2

    pcm = ax.pcolormesh(
        X,
        Y,
        Z,
        norm=(norm if semilogy else None),
        cmap=cmap,
        shading="gouraud",
        edgecolors=None,
    )
    ax.set_xlabel("Eigenvalue $\\lambda$")
    ax.set_ylabel("Perturbation strength $\\sigma$")

    if kwargs.get("colorbar"):
        cmap2, norm = custom_colormap_with_lognorm(
            variance_lowest,
            vmin=np.min(variances),
            vmax=np.max(variances),
            include_white=True,
        )
        mp = plt.cm.ScalarMappable(norm=norm, cmap=cmap2)
        cbar = plt.colorbar(mp, ax=ax)
