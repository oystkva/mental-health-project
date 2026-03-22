import os, sys
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib.pyplot import title
from matplotlib.lines import Line2D
from tqdm import tqdm

from src.config import DATA_DIR

from src.atlas_config import list_networks

from src.functional_connectivity import fisher_z2r

from src.permutation_test import (
    load_mean_zFCs,
    load_zFCs,
    compute_global_abs_max
)
from src.data_loader import load_perm_test_results

from matplotlib.colors import LinearSegmentedColormap
##### custom cmap #####
clist = [(0.1, 0.6, 1.0), (0.0, 0.0, 0.0), (1.0, 0.6, 0.1)]
cmap = LinearSegmentedColormap.from_list("cmap_name", clist, N=201)
#######################

#region Helper functions for plot functions
def star_count(p, alphas=(0.05, 0.01, 0.001)):
    # P-value significance overlay (upper triangle only)
        if p < alphas[2]:
            return 3
        elif p < alphas[1]:
            return 2
        elif p < alphas[0]:
            return 1
        return 0

def _add_sig_rectangles(ax, p_mat, alpha_levels, n, upper_triangle_only=True):
    """
    Draw rectangle outlines on cells where p-values are significant.
    Colors: *** (alpha[2]) green, ** (alpha[1]) grey, * (alpha[0]) black.
    """
    a05, a01, a001 = alpha_levels  # expects (0.05, 0.01, 0.001)

    for r in range(n):
        for c in range(n):
            if upper_triangle_only and c < r:
                continue

            # Choose edge color by significance tier
            if p_mat[r, c] < a001:
                edge = ('white', 1.0)
            elif p_mat[r, c] < a01:
                edge = ('white', 0.8)
            elif p_mat[r, c] < a05:
                edge = ('white', 0.5)
            else:
                continue

            rect = patches.Rectangle(
                (c - 0.5, r - 0.5),
                1.0, 1.0,
                linewidth=0.8,
                edgecolor=edge,
                facecolor="none",
            )
            ax.add_patch(rect)

def _apply_axis_labels(ax, networks, x_labels=True, y_labels=True, hide_ticks=False):
    n = len(networks)
    if x_labels:
        ax.set_xticks(np.arange(n))
        ax.set_xticklabels(networks, rotation=90, fontsize=7)
    else:
        ax.set_xticks([])
    if y_labels:
        ax.set_yticks(np.arange(n))
        ax.set_yticklabels(networks, fontsize=7)
    else:
        ax.set_yticks([])
    if hide_ticks:
        ax.tick_params(axis='both', which='both', length=0)

def _set_labels_(axes, row_labels: list[str], col_labels: list[str]):
    if len(row_labels) != axes.shape[0] or len(col_labels) != axes.shape[1]:
        raise ValueError(
            f"Label count mismatch: expected {axes.shape[0]} row labels and "
            f"{axes.shape[1]} column labels, but got {len(row_labels)} and "
            f"{len(col_labels)}."
        )
    for row in range(axes.shape[0]):
        label = axes[row, 0].set_ylabel(row_labels[row], fontsize=12, labelpad=10)
        label.set_bbox({'facecolor': '#fef2e2', 'boxstyle': 'round,pad=0.3'})
    for col in range(axes.shape[1]):
        label = axes[0, col].set_title(col_labels[col], fontsize=12, y=1.03)
        label.set_bbox({'facecolor': '#fef2e2', 'boxstyle': 'round,pad=0.3'})

#endregion

def plot_perm_test_result(
    test_type: str,
    task_type: str,
    band_type: str,
    out_dir: str = DATA_DIR,
    network_means: bool = True,
    alpha: tuple = (0.05, 0.01, 0.001),
    atlas_type: Optional[str] = None,
    title: Optional[str] = None,
    decomp_method: str = "memd"
):
    """
    Plot permutation test result matrix (delta or delta-delta) with optional significance overlay.

    Args:
        test_type: 'hc_mdd', 'atlas_comparison', 'band_comparison', or 'method_comparison'
        task_type: 'restAP'/'restPA'
        atlas_type: 'Schaefer400'/'Yan2023' (ignored for atlas_comparison if desired)
        band_type: 'full'/'slow5'/'slow4'/'slow3'
        out_dir: directory to save figure
        network_means: use network-level results
        alpha: significance threshold
        title: optional plot title
        decomp_method: decomposition method used ('memd', 'bandpass', etc.) to determine the directory structure for loading results
    """
    delta_obs, p_values = load_perm_test_results(
        test_type=test_type,
        task_type=task_type,
        band_type=band_type,
        atlas_type=atlas_type,
        network_means=network_means,
        decomp_method=decomp_method
    )

    plt.figure(figsize=(7, 6))

    vmax = np.max(np.abs(delta_obs))
    im = plt.imshow(delta_obs, cmap=cmap, vmin=-vmax, vmax=vmax)

    #region set title
    if title is None:
        if test_type == "hc_mdd":
            title = f"HC − MDD ({atlas_type}, {band_type})"
        elif test_type == "atlas_comparison":
            title = f"Atlas interaction: (HC−MDD) Yan − Schaefer ({band_type})"
        elif test_type == "band_comparison":
            title = f"Band interaction: (HC−MDD) {band_type} − full ({atlas_type})"
        elif test_type == "method_comparison":
            title = f"Method interaction: (HC−MDD) MEMD − Band-pass ({atlas_type}, {band_type})"

    if title:
        plt.title(title)
    #endregion

    plt.xlabel("Network")
    plt.ylabel("Network")

    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label("Δ" if test_type == "hc_mdd" else "ΔΔ")

    #region Significance overlay (upper triangle only)
    for i in range(delta_obs.shape[0]):
        for j in range(i + 1, delta_obs.shape[1]):
            for k, a in reversed(list(enumerate(alpha))):
                if p_values[i, j] < a:
                    plt.text(j, i, "*" * (k + 1), ha="center", va="center", color="white", fontsize=8)
                    plt.text(i, j, "*" * (k + 1), ha="center", va="center", color="white", fontsize=8)
                    break
    #endregion

    #region save path
    decomp_dir = decomp_method if test_type != "method_comparison" else "method_comparison"
    save_path = os.path.join(out_dir, "permutation_test_results", decomp_dir, "plots", test_type if test_type != "method_comparison" else "")
    os.makedirs(save_path, exist_ok=True)
    file_name = f"perm_test_{test_type}_{task_type}_{f'{atlas_type}_' if atlas_type else ''}{band_type}_{'networks' if network_means else 'full_parcels'}.svg"
    save_path = os.path.join(save_path, file_name)
    #endregion

    network_labels = list(list_networks().keys())
    plt.xticks(ticks=np.arange(len(network_labels)), labels=network_labels, rotation=90, fontsize=7)
    plt.yticks(ticks=np.arange(len(network_labels)), labels=network_labels, fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, format='svg')
    plt.close()

    return


def plot_combined_with_runs(
    test_type: str,
    atlas_type: str,
    band_type: str,
    decomp_method: str,
    out_dir: str = DATA_DIR,
    upper_triangle_only: bool = True,
    use_fdr_pvals: bool = False,
) -> None:
    """
    Plot combined FC matrix on the left, with restAP and restPA on the right
    stacked vertically, and a shared colorbar on the far right.

    Layout:
        [ Combined | AP ]
        [ Combined | PA ]   [colorbar]

    Args:
        test_type: 'hc_mdd', 'atlas_comparison', 'band_comparison', or 'method_comparison'
        atlas_type: 'Schaefer400'/'Yan2023' (ignored for atlas_comparison if desired)
        band_type: 'full'/'slow5'/'slow4'/'slow3'
        decomp_method: decomposition method used ('memd', 'bandpass', etc.) to determine the directory structure for loading results
        title_combined, title_AP, title_PA: panel titles.
        cbar_label: colorbar label.
        cmap: matplotlib colormap.
        out_path: optional save path.
    """

    #region load data
    delta_obs, p_values = load_perm_test_results(
        test_type=test_type,
        task_type="combined",
        band_type=band_type,
        atlas_type=atlas_type,
        decomp_method=decomp_method,
        use_fdr_pvals=use_fdr_pvals
    )

    delta_obs_PA, p_values_PA = load_perm_test_results(
        test_type=test_type,
        task_type="restPA",
        band_type=band_type,
        atlas_type=atlas_type,
        decomp_method=decomp_method,
        use_fdr_pvals=use_fdr_pvals
    )

    delta_obs_AP, p_values_AP = load_perm_test_results(
        test_type=test_type,
        task_type="restAP",
        band_type=band_type,
        atlas_type=atlas_type,
        decomp_method=decomp_method,
        use_fdr_pvals=use_fdr_pvals
    )
    #endregion

    if delta_obs.shape != delta_obs_AP.shape or delta_obs.shape != delta_obs_PA.shape:
        raise ValueError(
            f"All FC matrices must have the same shape, got "
            f"{delta_obs.shape}, {delta_obs_AP.shape}, {delta_obs_PA.shape}"
        )

    vmax = compute_global_abs_max(test_type=test_type)
    vmin = -vmax

    # Plot

    fig = plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(
        2, 3,
        width_ratios=[2.4, 1.2, 0.08],
        height_ratios=[1, 1],
        wspace=0.05,
        hspace=0.05,
        figure=fig
    )

    ax_combined = fig.add_subplot(gs[:, 0])   # spans both rows
    ax_ap = fig.add_subplot(gs[0, 1])
    ax_pa = fig.add_subplot(gs[1, 1])
    cax = fig.add_subplot(gs[:, 2])

    if upper_triangle_only:
        mask = np.triu(np.ones_like(delta_obs, dtype=bool), k=0)
        delta_obs = np.where(mask, delta_obs, np.nan)
        delta_obs_AP = np.where(mask, delta_obs_AP, np.nan)
        delta_obs_PA = np.where(mask, delta_obs_PA, np.nan)

    im = ax_combined.imshow(delta_obs, cmap=cmap, vmin=vmin, vmax=vmax)
    ax_ap.imshow(delta_obs_AP, cmap=cmap, vmin=vmin, vmax=vmax)
    ax_pa.imshow(delta_obs_PA, cmap=cmap, vmin=vmin, vmax=vmax)

    alphas = (0.05, 0.01, 0.001)


    #region significant p-value stars
    for i in range(delta_obs.shape[0]):
        for j in range(i, delta_obs.shape[1]):

            s_comb = star_count(p_values[i, j], alphas)
            s_ap   = star_count(p_values_AP[i, j], alphas)
            s_pa   = star_count(p_values_PA[i, j], alphas)

            # Combined plot: always just white stars
            if s_comb > 0:
                ax_combined.text(
                    j, i, "*" * s_comb,
                    ha="center", va="center",
                    color="white", fontsize=8
                )
                if not upper_triangle_only:
                    ax_combined.text(
                        i, j, "*" * s_comb,
                        ha="center", va="center",
                        color="white", fontsize=8
                    )

            # AP plot:
            # - white stars = overlap with combined
            # - colored stars = extra significance beyond combined
            if s_ap > 0:
                overlap_ap = min(s_ap, s_comb)
                extra_ap = max(0, s_ap - s_comb)

                if overlap_ap > 0:
                    ax_ap.text(
                        j, i, "*" * overlap_ap + " " * extra_ap,
                        ha="center", va="center",
                        color="white", fontsize=5
                    )
                    if not upper_triangle_only:
                        ax_ap.text(
                            i, j, "*" * overlap_ap + " " * extra_ap,
                            ha="center", va="center",
                            color="white", fontsize=5
                        )

                if extra_ap > 0:
                    ax_ap.text(
                        j, i, " " * overlap_ap + "*" * extra_ap,
                        ha="center", va="center",
                        color="lime", fontsize=5
                    )
                    if not upper_triangle_only:
                        ax_ap.text(
                            i, j, " " * overlap_ap + "*" * extra_ap,
                            ha="center", va="center",
                            color="lime", fontsize=5
                        )

            # PA plot:
            # - white stars = overlap with combined
            # - colored stars = extra significance beyond combined
            if s_pa > 0:
                overlap_pa = min(s_pa, s_comb)
                extra_pa = max(0, s_pa - s_comb)

                if overlap_pa > 0:
                    ax_pa.text(
                        j, i, "*" * overlap_pa + " " * extra_pa,
                        ha="center", va="center",
                        color="white", fontsize=5
                    )
                    if not upper_triangle_only:
                        ax_pa.text(
                            i, j, "*" * overlap_pa + " " * extra_pa,
                            ha="center", va="center",
                            color="white", fontsize=5
                        )

                if extra_pa > 0:
                    ax_pa.text(
                        j, i, " " * overlap_pa + "*" * extra_pa,
                        ha="center", va="center",
                        color="lime", fontsize=5
                    )
                    if not upper_triangle_only:
                        ax_pa.text(
                            i, j, " " * overlap_pa + "*" * extra_pa,
                            ha="center", va="center",
                            color="lime", fontsize=5
                        )
    #endregion

    #region figure title, legend and colorbar
    if test_type == "hc_mdd":
        sup_title = r"$\Delta$ (HC − MDD) for " + f"{band_type}{(' decomposed with ' + decomp_method.upper()) if band_type != 'full' else ''}{' - FDR corrected' if use_fdr_pvals else ''}"
        cbar_label = r"$\Delta$ (HC − MDD) of mean zFC"
    elif test_type == "atlas_comparison":
        sup_title = r"$\Delta\Delta$ Atlas Interaction: (HC − MDD) Yan − Schaefer for " + f"{band_type}{(' decomposed with ' + decomp_method.upper()) if band_type != 'full' else ''}{' - FDR corrected' if use_fdr_pvals else ''}"
        cbar_label = r"$\Delta\Delta$ (Yan − Schaefer) of mean zFC"
    elif test_type == "band_comparison":
        sup_title = r"$\Delta\Delta$ Band Interaction: (HC − MDD) " + f"{band_type} − full{(' decomposed with ' + decomp_method.upper()) if band_type != 'full' else ''}{' - FDR corrected' if use_fdr_pvals else ''}"
        cbar_label = r"$\Delta\Delta$ (" + f"{band_type} − full" + r") of mean zFC"
    elif test_type == "method_comparison":
        sup_title = r"$\Delta\Delta$ Method Interaction: (HC − MDD) MEMD − Band-pass filtered for " + f"{band_type}{' - FDR corrected' if use_fdr_pvals else ''}"
        cbar_label = r"$\Delta\Delta$ (MEMD − Band-pass) of mean zFC"
    else:
        sup_title = f"Permutation Test Results: {test_type}, {atlas_type if atlas_type else 'both atlases'}, {band_type}{' decomposed with ' + decomp_method.upper() if band_type != 'full' else ''}{' - FDR corrected' if use_fdr_pvals else ''}"
        cbar_label = "Value"
    legend_elements = [
        Line2D(
            [0], [0],
            linestyle='None',
            color='black',
            markersize=0,
            label='p-values: * < .05, ** < .01, *** < .001'
        ),
        Line2D(
            [0], [0],
            linestyle='None',
            marker='$*$',
            color='lime',
            markersize=3,
            label='stronger level of significance (beyond combined)'
        )
    ]

    fig.suptitle(sup_title, fontsize=10)
    fig.legend(
        handles=legend_elements,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.95),
        ncol=2,
        frameon=False,
        fontsize=6
    )
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(cbar_label, fontsize=10)
    #endregion

    #region axes titles and ticks
    # Titles
    ax_combined.set_title("Combined", fontsize=13)
    ax_ap.set_title("restAP", fontsize=11)
    ax_pa.set_title("restPA", fontsize=11, y=-0.2, pad=10)

    # Labels only on combined
    network_labels = list_networks()
    ticks = np.arange(len(network_labels))
    ax_combined.set_xticks(ticks)
    ax_combined.set_yticks(ticks)
    ax_combined.set_xticklabels(network_labels, rotation=90, fontsize=7)
    ax_combined.set_yticklabels(network_labels, fontsize=7)
    
    # Remove axes from AP/PA
    _apply_axis_labels(ax_ap, networks=None, x_labels=False, y_labels=False)
    _apply_axis_labels(ax_pa, networks=None, x_labels=False, y_labels=False)
    #endregion

    plt.tight_layout(rect=(0, 0, 1, 0.96))

    decomp_dir = decomp_method if test_type != "method_comparison" else "method_comparison"
    save_path = os.path.join(out_dir, "permutation_test_results", decomp_dir, "plots", test_type if test_type != "method_comparison" else "")
    os.makedirs(save_path, exist_ok=True)

    if use_fdr_pvals:
        out_path = os.path.join(save_path, f"combined_perm_test_{test_type}_{band_type}_fdr_corrected.svg")
    else:
        out_path = os.path.join(save_path, f"combined_perm_test_{test_type}_{band_type}.svg")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, format=os.path.splitext(out_path)[1][1:] or "svg", bbox_inches="tight")

    return


def plot_perm_test_results_grid(
    task_type: str,
    atlas_type: str,
    network_means: bool = True,
    alpha: tuple[float, float, float] = (0.05, 0.01, 0.001),
    test_stat_sign: str = "hc_mdd",
    out_dir: str = DATA_DIR,
    decomp_method: str = "memd"
) -> str:
    """
    Creates a 4x3 grid plot:
      rows: full, slow3, slow4, slow5
      cols: [mean rFC HC, delta_obs with significance stars, mean rFC MDD]
    """

    band_types = ("full", "slow3", "slow4", "slow5")
    networks = list_networks()
    n = len(networks)
    if n == 0:
        raise ValueError("list_networks returned an empty list; cannot label axes.")

    #region load data and compute means per band
    rFC_HC_means = []
    rFC_MDD_means = []
    deltas = []
    pvals = []

    for band in band_types:
        if task_type == "combined":
            Z_HC = load_mean_zFCs(
                atlas_type=atlas_type, 
                band_type=band, 
                group="HC", 
                network_means=network_means,
                decomp_method=decomp_method
            )
            Z_MDD = load_mean_zFCs(
                atlas_type=atlas_type, 
                band_type=band, 
                group="MDD",
                network_means=network_means,
                decomp_method=decomp_method
            )
        else:
            Z_HC = load_zFCs(
                task_type=task_type, 
                atlas_type=atlas_type, 
                band_type=band, 
                group="HC", 
                network_means=network_means,
                decomp_method=decomp_method
            )
            Z_MDD = load_zFCs(
                task_type=task_type, 
                atlas_type=atlas_type, 
                band_type=band, 
                group="MDD",
                network_means=network_means,
                decomp_method=decomp_method
            )

        if Z_HC.ndim != 3 or Z_MDD.ndim != 3:
            raise ValueError(
                f"Expected Z arrays with shape (n_subj, n, n). "
                f"Got HC={Z_HC.shape}, MDD={Z_MDD.shape}"
            )

        if Z_HC.shape[1] != n:
            raise ValueError(
                f"Network count mismatch: matrices have n={Z_HC.shape[1]} "
                f"but list_networks has n={n}"
            )

        # mean z-FC
        zHC_mean = np.mean(Z_HC, axis=0)
        zMDD_mean = np.mean(Z_MDD, axis=0)

        # Convert to rFC for plotting
        rHC_mean = fisher_z2r(zHC_mean)
        rMDD_mean = fisher_z2r(zMDD_mean)

        # Load permutation test results
        delta_obs, p_values = load_perm_test_results(
            test_type="hc_mdd",
            task_type=task_type, 
            atlas_type=atlas_type, 
            band_type=band, 
            network_means=network_means,
            decomp_method=decomp_method
        )

        if delta_obs.shape != (n, n) or p_values.shape != (n, n):
            raise ValueError(
                f"Permutation results must be (n,n). "
                f"Got delta={delta_obs.shape}, p={p_values.shape}"
            )

        rFC_HC_means.append(rHC_mean)
        rFC_MDD_means.append(rMDD_mean)
        deltas.append(delta_obs)
        pvals.append(p_values)

    rFC_HC_means = np.stack(rFC_HC_means, axis=0)
    rFC_MDD_means = np.stack(rFC_MDD_means, axis=0)
    deltas = np.stack(deltas, axis=0)
    pvals = np.stack(pvals, axis=0)
    #endregion

    #region figure
    fig, axes = plt.subplots(4, 3, figsize=(18, 22), constrained_layout=True)

    mean_vmin, mean_vmax = -1, 1
    delta_abs_max = float(np.max(np.abs(deltas)))
    if delta_abs_max == 0:
        delta_abs_max = 1e-6

    delta_title = "Δ mean z-FC (HC − MDD)" if test_stat_sign == "hc_mdd" else "Δ mean z-FC (MDD − HC)"
    #endregion

    #region plot
    for i, band in enumerate(band_types):
        # HC mean rFC
        ax0 = axes[i, 0]
        im0 = ax0.imshow(rFC_HC_means[i], cmap=cmap, vmin=mean_vmin, vmax=mean_vmax)
        ax0.set_title(f"HC mean rFC ({band})")
        _apply_axis_labels(ax0, networks)
        _add_sig_rectangles(ax0, pvals[i], alpha, n, upper_triangle_only=True)
        fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.02)

        # Delta + significance stars
        ax1 = axes[i, 1]
        im1 = ax1.imshow(deltas[i], cmap=cmap, vmin=-delta_abs_max, vmax=delta_abs_max)
        ax1.set_title(f"{delta_title} ({band})")
        _apply_axis_labels(ax1, networks)
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.02)

        # MDD mean rFC
        ax2 = axes[i, 2]
        im2 = ax2.imshow(rFC_MDD_means[i], cmap=cmap, vmin=mean_vmin, vmax=mean_vmax)
        ax2.set_title(f"MDD mean rFC ({band})")
        _apply_axis_labels(ax2, networks)
        _add_sig_rectangles(ax2, pvals[i], alpha, n, upper_triangle_only=True)
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.02)
        
        p = pvals[i]
        for r in range(n):
            for c in range(n):
                if p[r, c] < alpha[2]:
                    ax1.text(c, r, "***", ha="center", va="center", fontsize=7, color='white')
                elif p[r, c] < alpha[1]:
                    ax1.text(c, r, "**", ha="center", va="center", fontsize=7, color='white')
                elif p[r, c] < alpha[0]:
                    ax1.text(c, r, "*", ha="center", va="center", fontsize=7, color='white')

    fig.suptitle(
        f"Permutation test results "
        f"(task={task_type}, atlas={atlas_type}, "
        f"{'networks' if network_means else 'parcels'})",
        fontsize=16,
    )
    #endregion

    out_name = f"perm_grid_{task_type}_{atlas_type}_{'networks' if network_means else 'parcels'}.svg"
    out_path = os.path.join(out_dir, "permutation_test_results", decomp_method, "plots", "hc_mdd", out_name)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.savefig(out_path, format='svg')
    plt.close(fig)

    return out_path


def plot_perm_test_results_grid_AP_PA(
    atlas_type: str,
    network_means: bool = True,
    alpha: tuple[float, float, float] = (0.05, 0.01, 0.001),
    test_stat_sign: str = "hc_mdd",
    out_dir: str = DATA_DIR,
    decomp_method: str = "memd"
) -> str:
    """
    Creates two side-by-side 4x3 grid plots for restAP and restPA:
      rows: full, slow3, slow4, slow5
      cols: [mean FC HC, delta_obs with significance stars, mean FC MDD]
    """

    band_types = ("full", "slow3", "slow4", "slow5")
    networks = list_networks()
    n = len(networks)
    if n == 0:
        raise ValueError("list_networks returned an empty list; cannot label axes.")

    #region load data and compute means per band
    rFC_HC_PA_means = []
    rFC_MDD_PA_means = []
    rFC_HC_AP_means = []
    rFC_MDD_AP_means = []
    deltas_PA = []
    pvals_PA = []
    deltas_AP = []
    pvals_AP = []

    for band in band_types:
        Z_HC_PA = load_zFCs(
            task_type="restPA", 
            atlas_type=atlas_type, 
            band_type=band, 
            group="HC", 
            network_means=network_means,
            decomp_method=decomp_method
        )
        Z_MDD_PA = load_zFCs(
            task_type="restPA", 
            atlas_type=atlas_type, 
            band_type=band, 
            group="MDD",
            network_means=network_means,
            decomp_method=decomp_method
        )
        Z_HC_AP = load_zFCs(
            task_type="restAP", 
            atlas_type=atlas_type, 
            band_type=band, 
            group="HC", 
            network_means=network_means,
            decomp_method=decomp_method
        )
        Z_MDD_AP = load_zFCs(
            task_type="restAP", 
            atlas_type=atlas_type, 
            band_type=band, 
            group="MDD",
            network_means=network_means,
            decomp_method=decomp_method
        )

        if Z_HC_PA.ndim != 3 or Z_MDD_PA.ndim != 3:
            raise ValueError(
                f"Expected Z arrays with shape (n_subj, n, n). "
                f"Got HC={Z_HC_PA.shape}, MDD={Z_MDD_PA.shape}"
            )

        if Z_HC_PA.shape[1] != n:
            raise ValueError(
                f"Network count mismatch: matrices have n={Z_HC_PA.shape[1]} "
                f"but list_networks has n={n}"
            )

        if Z_HC_AP.ndim != 3 or Z_MDD_AP.ndim != 3:
            raise ValueError(
                f"Expected Z arrays with shape (n_subj, n, n). "
                f"Got HC={Z_HC_AP.shape}, MDD={Z_MDD_AP.shape}"
            )
        
        if Z_HC_AP.shape[1] != n:
            raise ValueError(
                f"Network count mismatch: matrices have n={Z_HC_AP.shape[1]} "
                f"but list_networks has n={n}"
            )

        # mean z-FC
        zHC_PA_mean = np.mean(Z_HC_PA, axis=0)
        zMDD_PA_mean = np.mean(Z_MDD_PA, axis=0)
        zHC_AP_mean = np.mean(Z_HC_AP, axis=0)
        zMDD_AP_mean = np.mean(Z_MDD_AP, axis=0)

        # Convert to rFC for plotting
        rHC_PA_mean = fisher_z2r(zHC_PA_mean)
        rMDD_PA_mean = fisher_z2r(zMDD_PA_mean)
        rHC_AP_mean = fisher_z2r(zHC_AP_mean)
        rMDD_AP_mean = fisher_z2r(zMDD_AP_mean)

        # Load permutation test results
        delta_obs_PA, p_values_PA = load_perm_test_results(
            test_type="hc_mdd",
            task_type="restPA", 
            atlas_type=atlas_type, 
            band_type=band, 
            network_means=network_means,
            decomp_method=decomp_method
        )
        delta_obs_AP, p_values_AP = load_perm_test_results(
            test_type="hc_mdd",
            task_type="restAP", 
            atlas_type=atlas_type, 
            band_type=band, 
            network_means=network_means,
            decomp_method=decomp_method
        )

        if delta_obs_PA.shape != (n, n) or p_values_PA.shape != (n, n):
            raise ValueError(
                f"Permutation results must be (n,n). "
                f"Got delta={delta_obs_PA.shape}, p={p_values_PA.shape}"
            )
        if delta_obs_AP.shape != (n, n) or p_values_AP.shape != (n, n):
            raise ValueError(
                f"Permutation results must be (n,n). "
                f"Got delta={delta_obs_AP.shape}, p={p_values_AP.shape}"
            )

        rFC_HC_PA_means.append(rHC_PA_mean)
        rFC_MDD_PA_means.append(rMDD_PA_mean)
        deltas_PA.append(delta_obs_PA)
        pvals_PA.append(p_values_PA)
        rFC_HC_AP_means.append(rHC_AP_mean)
        rFC_MDD_AP_means.append(rMDD_AP_mean)
        deltas_AP.append(delta_obs_AP)
        pvals_AP.append(p_values_AP)

    rFC_HC_PA_means = np.stack(rFC_HC_PA_means, axis=0)
    rFC_MDD_PA_means = np.stack(rFC_MDD_PA_means, axis=0)
    deltas_PA = np.stack(deltas_PA, axis=0)
    pvals_PA = np.stack(pvals_PA, axis=0)
    rFC_HC_AP_means = np.stack(rFC_HC_AP_means, axis=0)
    rFC_MDD_AP_means = np.stack(rFC_MDD_AP_means, axis=0)
    deltas_AP = np.stack(deltas_AP, axis=0)
    pvals_AP = np.stack(pvals_AP, axis=0)
    #endregion

    #region figure
    custom_grey = "#DDDDDD"   # white
    custom_white = "#FFFFFF"   # light grey

    fig = plt.figure(figsize=(32, 20), constrained_layout=True)
    gs = gridspec.GridSpec(5, 9, figure=fig, width_ratios=[0.2,1,1,1,0.2,1,1,1,0.2], height_ratios=[0.1,1,1,1,1])
    axes = np.empty((4, 6), dtype=object)
    for i in range(4):
        for j in range(3):
            axes[i, j] = fig.add_subplot(gs[i+1, j+1])
            axes[i, j + 3] = fig.add_subplot(gs[i+1, j+1+4])
    for pos in [(0.13, 0.975), (0.415, 0.975), (0.585, 0.975), (0.87, 0.975)]:
        group = "FC HC" if pos[0] % 0.5 < 0.25 else "FC MDD"
        fig.text(x=pos[0], y=pos[1], s=group, fontsize=16, ha='center', va='center')
    title1 = fig.text(x=0.275, y=0.975, s=" Δ mean z-FC (HC − MDD) restPA ", fontsize=16, ha='center', va='center')
    title2 = fig.text(x=0.725, y=0.975, s=" Δ mean z-FC (HC − MDD) restAP ", fontsize=16, ha='center', va='center')
    title1.set_bbox({'facecolor': custom_grey, 'boxstyle': 'round,pad=0.5', 'edgecolor': custom_white})
    title2.set_bbox({'facecolor': custom_white, 'boxstyle': 'round,pad=0.5', 'edgecolor': custom_grey})

    mean_vmin, mean_vmax = -1, 1
    delta_abs_max = float(np.max(np.abs(deltas_PA)))
    if delta_abs_max == 0:
        delta_abs_max = 1e-6

    delta_title = "Δ mean z-FC (HC − MDD)" if test_stat_sign == "hc_mdd" else "Δ mean z-FC (MDD − HC)"

    subplot_label_rectprops = {'facecolor': '#fef2e2', 'boxstyle': 'round,pad=0.3'}
    fig.set_facecolor(custom_grey)
    rect = patches.Rectangle(
        (0, 0),          # bottom-left in figure coordinates
        0.5,             # width (half figure)
        1,               # height (full figure)
        transform=fig.transFigure,
        facecolor=custom_white,
        zorder=-1
    )

    title = fig.suptitle(
        f"Permutation test results ("
        f"{'networks' if network_means else 'parcels'})",
        fontsize=16,
        horizontalalignment='center',
    )
    title.set_bbox({'facecolor': '#fef2e2', 'boxstyle': 'round,pad=0.5'})
    fig.patches.append(rect)
    #endregion

    #region band plots
    for i, band in enumerate(band_types):
        # HC PA mean rFC
        ax0 = axes[i, 0]
        im0 = ax0.imshow(rFC_HC_PA_means[i], cmap=cmap, vmin=mean_vmin, vmax=mean_vmax)
        # ax0.set_title(f"HC PA mean rFC ({band})")
        if i == len(band_types) - 1:
            _apply_axis_labels(ax0, networks, hide_ticks=True)
        else:
            _apply_axis_labels(ax0, networks, x_labels=False)
        text = ax0.set_title(band, rotation=90, fontsize=16, x = -0.25, y = 0.5)
        text.set_bbox(subplot_label_rectprops)
        _add_sig_rectangles(ax0, pvals_PA[i], alpha, n, upper_triangle_only=True)
        if i == 0:
            fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.02, location='top')

        # Delta PA + significance stars
        ax1 = axes[i, 1]
        im1 = ax1.imshow(deltas_PA[i], cmap=cmap, vmin=-delta_abs_max, vmax=delta_abs_max)
        # ax1.set_title(f"{delta_title} ({band})")
        # _apply_axis_labels(ax1)
        if i == len(band_types) - 1:
            _apply_axis_labels(ax1, networks, y_labels=False)
        else:
            _apply_axis_labels(ax1, networks, x_labels=False, y_labels=False)
        if i == 0:
            fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.02, location='top')

        # MDD PA mean rFC
        ax2 = axes[i, 2]
        im2 = ax2.imshow(rFC_MDD_PA_means[i], cmap=cmap, vmin=mean_vmin, vmax=mean_vmax)
        # ax2.set_title(f"MDD PA mean rFC ({band})")
        # _apply_axis_labels(ax2)
        if i == len(band_types) - 1:
            _apply_axis_labels(ax2, networks, y_labels=False)
        else:
            _apply_axis_labels(ax2, networks, x_labels=False, y_labels=False)
        _add_sig_rectangles(ax2, pvals_PA[i], alpha, n, upper_triangle_only=True)
        if i == 0:
            fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.02, location='top')
        
        p = pvals_PA[i]
        for r in range(n):
            for c in range(n):
                if p[r, c] < alpha[2]:
                    ax1.text(c, r, "***", ha="center", va="center", fontsize=7, color='white')
                elif p[r, c] < alpha[1]:
                    ax1.text(c, r, "**", ha="center", va="center", fontsize=7, color='white')
                elif p[r, c] < alpha[0]:
                    ax1.text(c, r, "*", ha="center", va="center", fontsize=7, color='white')

        # HC AP mean rFC
        ax3 = axes[i, 3]
        im3 = ax3.imshow(rFC_HC_AP_means[i], cmap=cmap, vmin=mean_vmin, vmax=mean_vmax)
        # ax3.set_title(f"HC AP mean rFC ({band})")
        # _apply_axis_labels(ax3)
        if i == len(band_types) - 1:
            _apply_axis_labels(ax3, networks, y_labels=False)
        else:
            _apply_axis_labels(ax3, networks, x_labels=False, y_labels=False)
        _add_sig_rectangles(ax3, pvals_AP[i], alpha, n, upper_triangle_only=True)
        if i == 0:
            fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.02, location='top')

        # Delta AP + significance stars
        ax4 = axes[i, 4]
        im4 = ax4.imshow(deltas_AP[i], cmap=cmap, vmin=-delta_abs_max, vmax=delta_abs_max)
        # ax4.set_title(f"{delta_title} ({band})")
        # _apply_axis_labels(ax4)
        if i == len(band_types) - 1:
            _apply_axis_labels(ax4, networks, y_labels=False)
        else:
            _apply_axis_labels(ax4, networks, x_labels=False, y_labels=False)
        if i == 0:
            fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.02, location='top')

        # MDD AP mean rFC
        ax5 = axes[i, 5]
        im5 = ax5.imshow(rFC_MDD_AP_means[i], cmap=cmap, vmin=mean_vmin, vmax=mean_vmax)
        text = ax5.set_title(band, rotation=90, fontsize=16, x = 1.25, y = 0.5, color=custom_grey)
        text.set_bbox({'facecolor': custom_grey, 'edgecolor': custom_grey, 'boxstyle': 'round,pad=0.3'})
        # ax5.set_title(f"MDD AP mean rFC ({band})")
        # _apply_axis_labels(ax5)
        if i == 0:
            ax5.yaxis.tick_right()
            ax5.set_yticks(np.arange(n))
            ax5.set_yticklabels(networks,  fontsize=7, color=custom_grey)
            ax5.tick_params(length=0)
            ax5.set_xticks([])
        elif i == len(band_types) - 1:
            _apply_axis_labels(ax5, networks, y_labels=False)
        else:
            _apply_axis_labels(ax5, networks, x_labels=False, y_labels=False)
        _add_sig_rectangles(ax5, pvals_AP[i], alpha, n, upper_triangle_only=True)
        if i == 0:
            fig.colorbar(im5, ax=ax5, fraction=0.046, pad=0.02, location='top')

        p = pvals_AP[i]
        for r in range(n):
            for c in range(n):
                if p[r, c] < alpha[2]:
                    ax4.text(c, r, "***", ha="center", va="center", fontsize=7, color='white')
                elif p[r, c] < alpha[1]:
                    ax4.text(c, r, "**", ha="center", va="center", fontsize=7, color='white')
                elif p[r, c] < alpha[0]:
                    ax4.text(c, r, "*", ha="center", va="center", fontsize=7, color='white')
    #endregion
    
    out_name = f"perm_grid_PA_AP_{atlas_type}_{'networks' if network_means else 'parcels'}.svg"
    out_path = os.path.join(out_dir, "permutation_test_results", decomp_method, "plots", "hc_mdd", out_name)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.savefig(out_path, format='svg')
    plt.close(fig)

    return out_path


def plot_delta_atlas_comp(
    task_type: str,
    network_means: bool = True,
    alpha: tuple[float, float, float] = (0.05, 0.01, 0.001),
    out_dir: str = DATA_DIR,
    decomp_method: str = "memd"
):
    """
    Creates a 2x4 grid plot:
      rows: Schaefer400, Yan2023 (delta_obs)
      cols: full, slow3, slow4, slow5
    """
    band_types = ("full", "slow3", "slow4", "slow5")
    networks = list_networks()
    n = len(networks)
    if n == 0:
        raise ValueError("list_networks returned an empty list; cannot label axes.")

    #region load data
    deltas_Yan2023 = []
    pvals_Yan2023 = []
    deltas_Schaefer400 = []
    pvals_Schaefer400 = []

    for band in band_types:
        for atlas in ("Yan2023", "Schaefer400"):
            # Load permutation test results
            delta_obs, p_values = load_perm_test_results(
                test_type="hc_mdd",
                task_type=task_type, 
                atlas_type=atlas, 
                band_type=band, 
                network_means=network_means,
                decomp_method=decomp_method
            )

            if delta_obs.shape != (n, n) or p_values.shape != (n, n):
                    raise ValueError(
                        f"Permutation results must be (n,n). "
                        f"Got delta={delta_obs.shape}, p={p_values.shape}"
                    )
            if atlas == "Yan2023":
                deltas_Yan2023.append(delta_obs)
                pvals_Yan2023.append(p_values)
            elif atlas == "Schaefer400":
                deltas_Schaefer400.append(delta_obs)
                pvals_Schaefer400.append(p_values)
            else:
                raise ValueError(f"Unexpected atlas type: {atlas}")
        
    deltas_Yan2023 = np.stack(deltas_Yan2023, axis=0)
    pvals_Yan2023 = np.stack(pvals_Yan2023, axis=0)
    deltas_Schaefer400 = np.stack(deltas_Schaefer400, axis=0)
    pvals_Schaefer400 = np.stack(pvals_Schaefer400, axis=0)
    #endregion

    #region plot
    fig, axes = plt.subplots(2, 4, figsize=(20, 10.4), constrained_layout=True)
    delta_abs_max = float(np.max(np.abs([deltas_Yan2023, deltas_Schaefer400])))
    if delta_abs_max == 0:
        delta_abs_max = 1e-6

    for i, band in enumerate(band_types):
        # Yan2023 delta
        ax0 = axes[0, i]
        im0 = ax0.imshow(deltas_Yan2023[i], cmap=cmap, vmin=-delta_abs_max, vmax=delta_abs_max)
        _apply_axis_labels(ax0, networks, x_labels=False, y_labels=False)
        
        p = pvals_Yan2023[i]
        for r in range(n):
            for c in range(n):
                if p[r, c] < alpha[2]:
                    pcolor = 'green'
                elif p[r, c] < alpha[1]:
                    pcolor = 'cyan'
                elif p[r, c] < alpha[0]:
                    pcolor = 'white'
                else:
                    continue
                ax0.text(c, r, round(p[r, c], 3), ha="center", va="center", fontsize=4, color=pcolor)
        # Schaefer400 delta
        ax1 = axes[1, i]
        im1 = ax1.imshow(deltas_Schaefer400[i], cmap=cmap, vmin=-delta_abs_max, vmax=delta_abs_max)
        _apply_axis_labels(ax1, networks, x_labels=False, y_labels=False)

        p = pvals_Schaefer400[i]
        for r in range(n):
            for c in range(n):
                if p[r, c] < alpha[2]:
                    pcolor = 'green'
                elif p[r, c] < alpha[1]:
                    pcolor = 'cyan'
                elif p[r, c] < alpha[0]:
                    pcolor = 'white'
                else:
                    continue
                ax1.text(c, r, round(p[r, c], 3), ha="center", va="center", fontsize=4, color=pcolor)
    
    _set_labels_(
        axes,
        row_labels=["Yan2023", "Schaefer400"],
        col_labels=list(band_types),
    )
    fig.set_constrained_layout_pads(w_pad=0.05, h_pad=0.05, hspace=0, wspace=0)
    fig.suptitle(
        r"$\Delta$ mean z-FC (HC$\minus$MDD)"
        f"\t(task={task_type})",
        fontsize=16,
    )
    #endregion

    out_name = f"perm_delta_atlas_comp_{task_type}_{'networks' if network_means else 'parcels'}.svg"
    out_path = os.path.join(out_dir, "permutation_test_results", decomp_method, "plots", "atlas_comparison", out_name)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, format='svg')
    plt.close(fig)

    return