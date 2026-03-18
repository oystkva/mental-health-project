import os
from matplotlib.pyplot import title
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from tqdm import tqdm

##### custom cmap #####
from matplotlib.colors import LinearSegmentedColormap
clist = [(0.1, 0.6, 1.0), (0.0, 0.0, 0.0), (1.0, 0.6, 0.1)]
cmap = LinearSegmentedColormap.from_list("cmap_name", clist, N=201)
#######################

from config import DATA_DIR, LOG_DIR, PLOT_DIR

from utils import log_message

from data_loader import list_networks

from functional_connectivity import fisher_z2r

def permutation_test(Z_A: np.ndarray, Z_B: np.ndarray, n_permutations: int = 10_000, test_dir: str = 'upper_tailed', seed = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform a permutation test between two groups of Fisher Z-transformed functional connectivity matrices.
    Args:
        Z_A (np.ndarray): Fisher Z-transformed FC matrices for group A (shape: n_A x regions x regions).
        Z_B (np.ndarray): Fisher Z-transformed FC matrices for group B (shape: n_B x regions x regions).
        n_permutations (int): Number of permutations to perform.
        test_dir (str): Type of test ('upper_tailed', 'lower_tailed', 'two_tailed').
        seed (int): Random seed for reproducibility.
    Returns:
        delta_obs (np.ndarray): Observed difference in means between groups.
        p_values (np.ndarray): P-values from the permutation test.
    """

    # assert test_dir in ['upper_tailed', 'lower_tailed', 'two_tailed'], "test_dir must be 'upper_tailed', 'lower_tailed', or 'two_tailed'"
    if test_dir not in ['upper_tailed', 'lower_tailed', 'two_tailed']:
        raise ValueError("test_dir must be 'upper_tailed', 'lower_tailed', or 'two_tailed'")
    
    rng = np.random.default_rng(seed)
    n_A = Z_A.shape[0]
    n_B = Z_B.shape[0]
    N = n_A + n_B
    Z = np.concatenate([Z_A, Z_B], axis=0)
    delta_obs = np.mean(Z_A, axis=0) - np.mean(Z_B, axis=0)
    # t_obs = delta_obs / (np.sqrt(np.var(Z_A, axis=0, ddof=1)/n_A + np.var(Z_B, axis=0, ddof=1)/n_B) + 1e-10)
    permutations = np.array([rng.permutation(N) for _ in range(n_permutations)])
    idxA = permutations[:, :n_A]
    idxB = permutations[:, n_A:]

    mean_A = np.mean(Z[idxA, :, :], axis=1)
    mean_B = np.mean(Z[idxB, :, :], axis=1)

    # var_A = np.var(Z[idxA, :, :], axis=1, ddof=1)
    # var_B = np.var(Z[idxB, :, :], axis=1, ddof=1)
    # se = np.sqrt((var_A / n_A) + (var_B / n_B))

    delta_perm = mean_A - mean_B
    # t_perm = delta_perm / (se + 1e-10)

    # se_z = 1 / np.sqrt(n_samples - 3)
    # z_score = mean_fc / se_z
    # p_val = 2 * (1 - norm.cdf(abs(z_score)))
    # p_val_matrix[i, j] = p_val
    # p_val_matrix[j, i] = p_val  # Symmetric matrix

    if test_dir == 'upper_tailed':
        p_values = (np.sum(delta_perm >= delta_obs[None, :, :], axis=0) + 1)/(n_permutations + 1)
    elif test_dir == 'lower_tailed':
        p_values = (np.sum(delta_perm <= delta_obs[None, :, :], axis=0) + 1)/(n_permutations + 1)
    else:  # two_tailed
        p_values = (np.sum(np.abs(delta_perm) >= np.abs(delta_obs[None, :, :]), axis=0) + 1)/(n_permutations + 1)
    return delta_obs, p_values


def permutation_test_delta_delta(
    Z_A_X: np.ndarray, 
    Z_A_Y: np.ndarray, 
    Z_B_X: np.ndarray, 
    Z_B_Y: np.ndarray, 
    n_permutations: int = 10_000, 
    test_dir: str = 'upper_tailed', 
    seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform a permutation test on the difference of differences (delta-delta) between two groups of Fisher Z-transformed FC matrices.
    Args:
        Z_A_X (np.ndarray): FC matrices for group A, condition X (shape: n_A x regions x regions).
        Z_A_Y (np.ndarray): FC matrices for group A, condition Y (shape: n_A x regions x regions).
        Z_B_X (np.ndarray): FC matrices for group B, condition X (shape: n_B x regions x regions).
        Z_B_Y (np.ndarray): FC matrices for group B, condition Y (shape: n_B x regions x regions).
        n_permutations (int): Number of permutations to perform.
        test_dir (str): Type of test ('upper_tailed', 'lower_tailed', 'two_tailed').
        seed (int): Random seed for reproducibility.|
    Returns:
        delta_obs (np.ndarray): Observed difference in means between groups.
        p_values (np.ndarray): P-values from the permutation test.
    """
    if test_dir not in ["upper_tailed", "lower_tailed", "two_tailed"]:
        raise ValueError("test_dir must be 'upper_tailed', 'lower_tailed', or 'two_tailed'")

    rng = np.random.default_rng(seed)

    # Trailing-dim checks (allow any FC shape: (R,R), (E,), (K,K), etc.)
    trailing = Z_A_X.shape[1:]
    for name, arr in [
        ("Z_B_X", Z_B_X),
        ("Z_A_Y", Z_A_Y),
        ("Z_B_Y", Z_B_Y),
    ]:
        if arr.shape[1:] != trailing:
            raise ValueError(f"{name} trailing dims {arr.shape[1:]} do not match {trailing}.")
        
    # Observed delta-delta
    delta_X_obs = np.mean(Z_A_X, axis=0) - np.mean(Z_B_X, axis=0)
    delta_Y_obs = np.mean(Z_A_Y, axis=0) - np.mean(Z_B_Y, axis=0)
    delta_delta_obs = delta_X_obs - delta_Y_obs

    n_Ax, n_Bx = Z_A_X.shape[0], Z_B_X.shape[0]
    Nx = n_Ax + n_Bx
    Zx = np.concatenate([Z_A_X, Z_B_X], axis=0)

    n_Ay, n_By = Z_A_Y.shape[0], Z_B_Y.shape[0]
    Ny = n_Ay + n_By
    if Nx != Ny:
        raise ValueError(f"Total sample size for X (Nx={Nx}) does not match total sample size for Y (Ny={Ny}).")
    Zy = np.concatenate([Z_A_Y, Z_B_Y], axis=0)

    perms = np.array([rng.permutation(Nx) for _ in range(n_permutations)])

    idxA = perms[:, :n_Ax]
    idxB = perms[:, n_Ax:]

    mean_A_X = np.mean(Zx[idxA], axis=1)
    mean_B_X = np.mean(Zx[idxB], axis=1)
    mean_A_Y = np.mean(Zy[idxA], axis=1)
    mean_B_Y = np.mean(Zy[idxB], axis=1)

    delta_X_perm = mean_A_X - mean_B_X
    delta_Y_perm = mean_A_Y - mean_B_Y
    delta_delta_perm = delta_X_perm - delta_Y_perm

    if test_dir == "upper_tailed":
        p_values = (np.sum(delta_delta_perm >= delta_delta_obs[None], axis=0) + 1) / (n_permutations + 1)
    elif test_dir == "lower_tailed":
        p_values = (np.sum(delta_delta_perm <= delta_delta_obs[None], axis=0) + 1) / (n_permutations + 1)
    else:  # two-tailed
        p_values = (np.sum(np.abs(delta_delta_perm) >= np.abs(delta_delta_obs[None, ...]), axis=0) + 1) / (n_permutations + 1)

    return delta_delta_obs, p_values


def load_zFCs(
    task_type: str, 
    atlas_type: str, 
    band_type: str, 
    group: str, 
    network_means: bool = True,
    decomp_method: str = "memd",
) -> np.ndarray:
    """
    Load Fisher Z-transformed functional connectivity matrices for a given group.
    Args:
        task_type (str): Task type ('restAP'/'restPA'/'combined').
        atlas_type (str): Atlas type ('Schaefer400'/'Yan2023').
        band_type (str): Band type ('full'/'slow5'/'slow4'/'slow3').
        group (str): Group identifier ('MDD'/'HC').
        network_means (bool): Whether to use network means or parcel-level data (default: True).
        decomp_method (str): Decomposition method used ('memd', 'bandpass', etc.) to determine the directory structure.
    Returns:
        np.ndarray: Fisher Z-transformed FC matrices for the specified group.
    """
    parcellation = "networks" if network_means else "full_parcels"
    if decomp_method == "memd" or band_type == "full":
        load_dir = os.path.join(DATA_DIR, "zFC_matrices", task_type, parcellation)
    elif decomp_method == "bandpass" and band_type != "full":
        load_dir = os.path.join(DATA_DIR, "zFC_matrices", task_type + "_bandpass", parcellation)
    zFCs = np.empty((0, ))
    for root, _, files in list(os.walk(load_dir)):
        for file in files:
            if file.endswith(".npy"):
                if all(param in file for param in [group, atlas_type, band_type]):
                    file_path = os.path.join(root, file)
                    loaded_zFCs = np.load(file_path)[np.newaxis, ...]
                    if zFCs.size == 0:
                        zFCs = loaded_zFCs
                    else:
                        zFCs = np.concatenate((zFCs, loaded_zFCs), axis=0)
    return zFCs


def load_mean_zFCs(
    atlas_type: str,
    band_type: str,
    group: str,
    network_means: bool = True,
    decomp_method: str = "memd",
) -> np.ndarray:
    """Load mean Fisher Z-transformed functional connectivity matrix for a given group.
    Args:
        atlas_type (str): Atlas type ('Schaefer400'/'Yan2023').
        band_type (str): Band type ('full'/'slow5'/'slow4'/'slow3').
        group (str): Group identifier ('MDD'/'HC').
        network_means (bool): Whether to use network means or parcel-level data (default: True).
        decomp_method (str): Decomposition method used ('memd', 'bandpass', etc.) to determine the directory structure.
    Returns:
        np.ndarray: Mean Fisher Z-transformed FC matrix for the specified group.
    """
    zFCs_AP = load_zFCs("restAP", atlas_type, band_type, group, network_means, decomp_method)
    zFCs_PA = load_zFCs("restPA", atlas_type, band_type, group, network_means, decomp_method)
    mean_zFC = np.mean(np.stack([zFCs_AP, zFCs_PA]), axis=0)
    return mean_zFC


def perm_test_HC_MDD(
    task_type: str, 
    atlas_type: str, 
    band_type: str, 
    network_means: bool = True, 
    n_permutations: int = 10_000, 
    test_dir: str = 'upper_tailed',
    decomp_method: str = "memd",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform permutation test between HC and MDD groups for given task, atlas, and band type.
    Args:
        task_type (str): Task type ('restAP'/'restPA'/'combined').
        atlas_type (str): Atlas type ('Schaefer400'/'Yan2023').
        band_type (str): Band type ('full'/'slow5'/'slow4'/'slow3').
        network_means (bool): Whether to use network means or parcel-level data (default: True).
        n_permutations (int): Number of permutations to perform.
        test_dir (str): Type of test ('upper_tailed', 'lower_tailed', 'two_tailed').
        decomp_method (str): Decomposition method used ('memd', 'bandpass', etc.).
    Returns:
        delta_obs (np.ndarray): Observed difference in means between groups.
        p_values (np.ndarray): P-values from the permutation test.
    """
    log_path = os.path.join(LOG_DIR, "permutation_tests.log")
    log_message(f"Starting permutation test for {task_type}, {atlas_type}, {band_type}, network_means={network_means}, decomp_method={decomp_method}", log_path)

    if task_type == "combined":
        Z_HC = load_mean_zFCs(atlas_type, band_type, "HC", network_means, decomp_method)
        Z_MDD = load_mean_zFCs(atlas_type, band_type, "MDD", network_means, decomp_method)
    else:
        Z_HC = load_zFCs(task_type, atlas_type, band_type, "HC", network_means, decomp_method)
        Z_MDD = load_zFCs(task_type, atlas_type, band_type, "MDD", network_means, decomp_method)

    delta_obs, p_values = permutation_test(Z_HC, Z_MDD, n_permutations, test_dir)

    log_message(f"Results: delta_obs shape: {delta_obs.shape}, p_values shape: {p_values.shape}", log_path)

    network_label = "networks" if network_means else "full_parcels"

    os.makedirs(os.path.join(DATA_DIR, "permutation_test_results", decomp_method, "hc_mdd"), exist_ok=True)
    np.save(os.path.join(DATA_DIR, "permutation_test_results", decomp_method, "hc_mdd", f"delta_obs_{task_type}_{atlas_type}_{band_type}_{network_label}.npy"), delta_obs)
    np.save(os.path.join(DATA_DIR, "permutation_test_results", decomp_method, "hc_mdd",  f"p_values_{task_type}_{atlas_type}_{band_type}_{network_label}.npy"), p_values)
    #TODO: remove?
    # # plot comparison of mean FC matrices with delta_obs between groups
    # fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    # im0 = ax[0].imshow(fisher_z2r(np.mean(Z_HC, axis=0)), cmap='bwr', vmin=-1, vmax=1)
    # ax[0].set_title("Mean FC - HC")
    # plt.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04, orientation='horizontal')
    # im1 = ax[1].imshow(delta_obs, cmap='bwr', vmin=-np.max(np.abs(delta_obs)), vmax=np.max(np.abs(delta_obs)))
    # ax[1].set_title("Delta Obs (HC - MDD)")
    # plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04, orientation='horizontal')
    # im2 = ax[2].imshow(fisher_z2r(np.mean(Z_MDD, axis=0)), cmap='bwr', vmin=-1, vmax=1)
    # ax[2].set_title("Mean FC - MDD")
    # plt.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04, orientation='horizontal')
    # for row in range(p_values.shape[0]):
    #     for col in range(p_values.shape[1]):
    #         rect0 = patches.Rectangle(
    #             (col - 0.5, row - 0.5),  # lower-left corner
    #             1, 1,                   # width, height
    #             linewidth=1.2,
    #             edgecolor='black',
    #             facecolor='none'
    #         )
    #         rect1 = patches.Rectangle(
    #             (col - 0.5, row - 0.5),  # lower-left corner
    #             1, 1,                   # width, height
    #             linewidth=1.2,
    #             edgecolor='black',
    #             facecolor='none'
    #         )
    #         if p_values[row, col] < 0.001:
    #             rect0.set_edgecolor('green')
    #             rect1.set_edgecolor('green')
    #             ax[0].add_patch(rect0)
    #             ax[1].text(col, row, '***', ha='center', va='center', color='black', fontsize=12)
    #             ax[2].add_patch(rect1)
    #         elif p_values[row, col] < 0.01:
    #             rect0.set_edgecolor('grey')
    #             rect1.set_edgecolor('grey')
    #             ax[0].add_patch(rect0)
    #             ax[1].text(col, row, '**', ha='center', va='center', color='black', fontsize=12)
    #             ax[2].add_patch(rect1)
    #         elif p_values[row, col] < 0.05:
    #             rect0.set_edgecolor('black')
    #             rect1.set_edgecolor('black')
    #             ax[0].add_patch(rect0)
    #             ax[1].text(col, row, '*', ha='center', va='center', color='black', fontsize=12)
    #             ax[2].add_patch(rect1)

    # plt.suptitle(f"Permutation Test Results: {task_type}, {atlas_type}, {band_type}, {'Networks' if network_means else 'Full Parcels'}")
    # if os.path.exists(os.path.join(PLOT_DIR, 'permutation_test_results')) is False:
    #     os.makedirs(os.path.join(PLOT_DIR, 'permutation_test_results'))
    # plt.savefig(os.path.join(PLOT_DIR, 'permutation_test_results', f"permutation_test_{task_type}_{atlas_type}_{band_type}_{'networks' if network_means else 'full_parcels'}.png"))
    # plt.close()

    return delta_obs, p_values


def perm_test_atlas(
    task_type: str, 
    band_type: str,
    network_means: bool = True, 
    n_permutations: int = 10_000, 
    test_dir: str = 'upper_tailed',
    decomp_method: str = 'memd'
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform permutation test between the results of the permutation tests for the Schaefer400 and Yan2023 atlases.
    Args:
        task_type (str): Task type ('restAP'/'restPA'/'combined').
        band_type (str): Band type ('full'/'slow5'/'slow4'/'slow3').
        network_means (bool): Whether to use network means or parcel-level data (default: True).
        n_permutations (int): Number of permutations to perform.
        test_dir (str): Type of test ('upper_tailed', 'lower_tailed', 'two_tailed').
        decomp_method (str): Decomposition method used ('memd', 'bandpass', etc.).
    Returns:
        delta_obs (np.ndarray): Observed difference in means between atlases.
        p_values (np.ndarray): P-values from the permutation test.
    """
    log_path = os.path.join(LOG_DIR, "permutation_tests.log")
    log_message(f"Starting permutation test between atlases for {task_type}, {band_type}, network_means={network_means}", log_path)

    if task_type == "combined":
        Z_HC_yan = load_mean_zFCs("Yan2023", band_type, "HC", network_means, decomp_method)
        Z_MDD_yan = load_mean_zFCs("Yan2023", band_type, "MDD", network_means, decomp_method)

        Z_HC_sch = load_mean_zFCs("Schaefer400", band_type, "HC", network_means, decomp_method)
        Z_MDD_sch = load_mean_zFCs("Schaefer400", band_type, "MDD", network_means, decomp_method)
    else:
        Z_HC_yan = load_zFCs(task_type, "Yan2023", band_type, "HC", network_means, decomp_method)
        Z_MDD_yan = load_zFCs(task_type, "Yan2023", band_type, "MDD", network_means, decomp_method)
        
        Z_HC_sch = load_zFCs(task_type, "Schaefer400", band_type, "HC", network_means, decomp_method)
        Z_MDD_sch = load_zFCs(task_type, "Schaefer400", band_type, "MDD", network_means, decomp_method)

    delta_delta_obs, p_values = permutation_test_delta_delta(
        Z_HC_yan, 
        Z_HC_sch, 
        Z_MDD_yan, 
        Z_MDD_sch, 
        n_permutations, 
        test_dir, 
        seed=42
    )
    
    log_message(f"Results: delta_delta_obs shape: {delta_delta_obs.shape}, p_values shape: {p_values.shape}", log_path)
    
    os.makedirs(os.path.join(DATA_DIR, "permutation_test_results", decomp_method, "atlas_comparison"), exist_ok=True)
    np.save(os.path.join(DATA_DIR, "permutation_test_results", decomp_method, "atlas_comparison", f"delta_delta_obs_{task_type}_{band_type}_{'networks' if network_means else 'full_parcels'}.npy"), delta_delta_obs)
    np.save(os.path.join(DATA_DIR, "permutation_test_results", decomp_method, "atlas_comparison",  f"p_values_{task_type}_{band_type}_{'networks' if network_means else 'full_parcels'}.npy"), p_values)  

    return delta_delta_obs, p_values


def perm_test_slow_band(
    task_type: str, 
    atlas_type: str, 
    slow_band: str, 
    network_means: bool = True, 
    n_permutations: int = 5000, 
    test_dir: str = 'upper_tailed',
    decomp_method: str = 'memd'
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform permutation test between the results of the permutation tests for slow band n and full band.
    Args:
        task_type (str): Task type ('restAP'/'restPA'/'combined').
        atlas_type (str): Atlas type ('Schaefer400'/'Yan2023').
        slow_band (str): Slow band type ('slow5'/'slow4'/'slow3').
        network_means (bool): Whether to use network means or parcel-level data (default: True).
        n_permutations (int): Number of permutations to perform.
        test_dir (str): Type of test ('upper_tailed', 'lower_tailed', 'two_tailed').
        decomp_method (str): Decomposition method used ('memd', 'bandpass', etc.).
    Returns:
        delta_delta_obs (np.ndarray): Observed difference in means between bands.
        p_values (np.ndarray): P-values from the permutation test.
    """
    log_path = os.path.join(LOG_DIR, "permutation_tests.log")
    log_message(f"Starting permutation test between {slow_band} and full band for {task_type}, {atlas_type}, network_means={network_means}", log_path)

    if task_type == "combined":
        Z_HC_full = load_mean_zFCs(atlas_type, "full", "HC", network_means, decomp_method)
        Z_MDD_full = load_mean_zFCs(atlas_type, "full", "MDD", network_means, decomp_method)
        
        Z_HC_slow = load_mean_zFCs(atlas_type, slow_band, "HC", network_means, decomp_method)
        Z_MDD_slow = load_mean_zFCs(atlas_type, slow_band, "MDD", network_means, decomp_method)
    else:
        Z_HC_full = load_zFCs(task_type, atlas_type, "full", "HC", network_means, decomp_method)
        Z_MDD_full = load_zFCs(task_type, atlas_type, "full", "MDD", network_means, decomp_method)
        
        Z_HC_slow = load_zFCs(task_type, atlas_type, slow_band, "HC", network_means, decomp_method)
        Z_MDD_slow = load_zFCs(task_type, atlas_type, slow_band, "MDD", network_means, decomp_method)

    delta_delta_obs, p_values = permutation_test_delta_delta(
        Z_HC_slow, 
        Z_HC_full, 
        Z_MDD_slow, 
        Z_MDD_full, 
        n_permutations, 
        test_dir, 
        seed=42
    )

    log_message(f"Results: delta_delta_obs shape: {delta_delta_obs.shape}, p_values shape: {p_values.shape}", log_path)
    
    os.makedirs(os.path.join(DATA_DIR, "permutation_test_results", decomp_method, "band_comparison"), exist_ok=True)
    np.save(os.path.join(DATA_DIR, "permutation_test_results", decomp_method, "band_comparison", f"delta_delta_obs_{task_type}_{atlas_type}_{slow_band}_{'networks' if network_means else 'full_parcels'}.npy"), delta_delta_obs)
    np.save(os.path.join(DATA_DIR, "permutation_test_results", decomp_method, "band_comparison",  f"p_values_{task_type}_{atlas_type}_{slow_band}_{'networks' if network_means else 'full_parcels'}.npy"), p_values) 

    return delta_delta_obs, p_values

def perm_test_memd_bandpass(
    task_type: str,
    atlas_type: str,
    slow_band: str,
    network_means: bool = True,
    n_permutations: int = 10_000,
    test_dir: str = 'upper_tailed',
) -> tuple[np.ndarray, np.ndarray]:
    """Perform permutation test between MEMD and band-pass results for given task, atlas, and band type.
    Args:
        task_type (str): Task type ('restAP'/'restPA'/'combined').
        atlas_type (str): Atlas type ('Schaefer400'/'Yan2023').
        slow_band (str): Band type ('full'/'slow5'/'slow4'/'slow3').
        network_means (bool): Whether to use network means or parcel-level data (default: True).
        n_permutations (int): Number of permutations to perform.
        test_dir (str): Type of test ('upper_tailed', 'lower_tailed', 'two_tailed').
    Returns:
        delta_delta_obs (np.ndarray): Observed difference in means between methods.
        p_values (np.ndarray): P-values from the permutation test.
    """
    log_path = os.path.join(LOG_DIR, "permutation_tests.log")
    log_message(f"Starting permutation test between MEMD and band-pass for {task_type}, {atlas_type}, {slow_band}, network_means={network_means}", log_path)

    if task_type == "combined":
        Z_HC_MEMD = load_mean_zFCs(atlas_type, slow_band, "HC", network_means, decomp_method="memd")
        Z_MDD_MEMD = load_mean_zFCs(atlas_type, slow_band, "MDD", network_means, decomp_method="memd")

        Z_HC_BP = load_mean_zFCs(atlas_type, slow_band, "HC", network_means, decomp_method="bandpass")
        Z_MDD_BP = load_mean_zFCs(atlas_type, slow_band, "MDD", network_means, decomp_method="bandpass")
    else:
        Z_HC_MEMD = load_zFCs(task_type, atlas_type, slow_band, "HC", network_means, decomp_method="memd")
        Z_MDD_MEMD = load_zFCs(task_type, atlas_type, slow_band, "MDD", network_means, decomp_method="memd")
        
        Z_HC_BP = load_zFCs(task_type, atlas_type, slow_band, "HC", network_means, decomp_method="bandpass")
        Z_MDD_BP = load_zFCs(task_type, atlas_type, slow_band, "MDD", network_means, decomp_method="bandpass")
    # difference per subject

    delta_delta_obs, p_values = permutation_test_delta_delta(
        Z_HC_MEMD, 
        Z_HC_BP, 
        Z_MDD_MEMD, 
        Z_MDD_BP, 
        n_permutations, 
        test_dir, 
        seed=42
    )
    log_message(f"Results: delta_delta_obs shape: {delta_delta_obs.shape}, p_values shape: {p_values.shape}", log_path)
    
    os.makedirs(os.path.join(DATA_DIR, "permutation_test_results", "method_comparison"), exist_ok=True)
    np.save(os.path.join(DATA_DIR, "permutation_test_results", "method_comparison", f"delta_delta_obs_{task_type}_{atlas_type}_{slow_band}_{'networks' if network_means else 'full_parcels'}.npy"), delta_delta_obs)
    np.save(os.path.join(DATA_DIR, "permutation_test_results", "method_comparison",  f"p_values_{task_type}_{atlas_type}_{slow_band}_{'networks' if network_means else 'full_parcels'}.npy"), p_values) 

    return delta_delta_obs, p_values
    

def load_perm_test_results(
    test_type: str, 
    task_type: str, 
    band_type: str, 
    atlas_type: str = None, 
    network_means: bool = True,
    decomp_method: str = "memd",
    use_fdr_pvals: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load permutation test results for given task, atlas, and band type.
    Args:
        test_type (str): Type of test ('hc_mdd', 'atlas_comparison', 'band_comparison', 'method_comparison').
        task_type (str): Task type ('restAP'/'restPA'/'combined').
        band_type (str): Band type ('full'/'slow5'/'slow4'/'slow3').
        atlas_type (str | None): Atlas type ('Schaefer400'/'Yan2023') or None if not applicable (atlas comparison).
        network_means (bool): Whether to use network means or parcel-level data (default: True).
        decomp_method (str): Decomposition method used ('memd', 'bandpass', etc.).
    Returns:
        delta_obs (np.ndarray): Observed difference in means between groups.
        p_values (np.ndarray): P-values from the permutation test.
    """


    file_name_id = ""
    if test_type == "method_comparison":
        if band_type == "full":
            raise ValueError("Method comparison is not applicable for full band type since it only applies to decomposed bands.")
        file_name_id = f"{task_type}_{atlas_type}_{band_type}_{'networks' if network_means else 'full_parcels.npy'}"
    elif test_type == "hc_mdd":
        if atlas_type is None:
            raise ValueError("atlas_type must be provided for 'hc_mdd' test_type")
        file_name_id = f"{task_type}_{atlas_type}_{band_type}_{'networks' if network_means else 'full_parcels.npy'}"
    elif test_type == "atlas_comparison":
        file_name_id = f"{task_type}_{band_type}_{'networks' if network_means else 'full_parcels.npy'}"
    elif test_type == "band_comparison":
        if atlas_type is None:
            raise ValueError("atlas_type must be provided for 'band_comparison' test_type")
        file_name_id = f"{task_type}_{atlas_type}_{band_type}_{'networks' if network_means else 'full_parcels.npy'}"
    else:
        print(test_type, task_type, band_type, atlas_type, network_means, decomp_method)
        raise ValueError(f"test_type must be 'hc_mdd', 'atlas_comparison', or 'band_comparison'. test_type provided: {test_type}")

    decomp_dir = decomp_method
    if band_type == "full":
        decomp_dir = "memd"  # full band results are stored under memd directory for now since they are the same for both methods
    elif test_type == "method_comparison":
        decomp_dir = ""  # method comparison results are stored under their own directory since they involve both methods

    delta_obs_path = os.path.join(
        DATA_DIR, "permutation_test_results",
        decomp_dir, test_type,
        f"delta{'_delta' if test_type != 'hc_mdd' else ''}_obs_{file_name_id}.npy"
    )
    p_values_path = os.path.join(
        DATA_DIR, "permutation_test_results", 
        decomp_dir, test_type, 
        f"p_{'fdr_' if use_fdr_pvals else ''}values_{file_name_id}.npy"
    )
    delta_obs = np.load(delta_obs_path)
    p_values = np.load(p_values_path)

    return delta_obs, p_values


def compute_global_abs_max(
        test_type: str,
) -> float:
    """Compute the global maximum absolute value across multiple .npy files containing FC matrices.
    Args:
        test_type: 'hc_mdd', 'atlas_comparison', 'band_comparison', or 'method_comparison' to determine which set of results to load.

    Returns:
        float: The global maximum absolute value across all matrices.
    """
    match_str = "delta"
    path = os.path.join(DATA_DIR, "permutation_test_results")
    if test_type == "method_comparison":
        file_paths = [
            os.path.join(path, "method_comparison", f)
            for f in os.listdir(os.path.join(path, "method_comparison"))
            if f.endswith(".npy") and (match_str in f)
        ]       
    elif test_type in ["hc_mdd", "atlas_comparison", "band_comparison"]:
        file_paths = [
            os.path.join(path, decomp_dir, test_type, f)
            for decomp_dir in ["memd", "bandpass"]  # both memd and band_pass directories contain relevant results for these test types
            for f in os.listdir(os.path.join(path, decomp_dir, test_type))
            if f.endswith(".npy") and (match_str in f)
        ]
    else:
        raise ValueError(f"test_type must be 'hc_mdd', 'atlas_comparison', 'band_comparison', or 'method_comparison'. test_type provided: {test_type}")


    return max(
        np.max(np.abs(np.load(f)))
        for f in file_paths
    )


def plot_perm_test_results_old(
    delta_obs: np.ndarray,
    p_values: np.ndarray,
    alpha: tuple[float, float, float] = (0.05, 0.01, 0.001)
):
    """
    Plot the results of the permutation test.
    Args:
        delta_obs (np.ndarray): Observed difference in means between groups. Shape: (4 x regions x regions) where 4 corresponds to different band types.
        p_values (np.ndarray): P-values from the permutation test. Shape: (4 x regions x regions) where 4 corresponds to different band types.
        alpha (list): Significance levels for thresholding p-values.
    """
    significant_mask1 = p_values < alpha[0]
    significant_mask2 = p_values < alpha[1]
    significant_mask3 = p_values < alpha[2]
    significant_mask = significant_mask1.astype(int) + significant_mask2.astype(int)*2 + significant_mask3.astype(int)*3

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title("Observed Difference in Means (Delta Obs)")
    plt.imshow(delta_obs, cmap='bwr', vmin=-np.max(np.abs(delta_obs)), vmax=np.max(np.abs(delta_obs)))
    plt.colorbar(label='Delta Obs')
    plt.xlabel('Regions')
    plt.ylabel('Regions')

    plt.subplot(1, 2, 2)
    plt.title(f"P-values (Significant at alpha={alpha})")
    plt.imshow(significant_mask, cmap='gray_r')
    plt.colorbar(label='Significant (1=True, 0=False)')
    plt.xlabel('Regions')
    plt.ylabel('Regions')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "permutation_test_results.svg"), format='svg')


def plot_perm_test_result(
    test_type: str,
    task_type: str,
    band_type: str,
    out_dir: str = DATA_DIR,
    network_means: bool = True,
    alpha: tuple = (0.05, 0.01, 0.001),
    atlas_type: str = None,
    title: str = None,
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

    if title is None:
        if test_type == "hc_mdd":
            title = f"HC − MDD ({atlas_type}, {band_type})"
        elif test_type == "atlas_comparison":
            title = f"Atlas interaction: (HC−MDD) Yan − Schaefer ({band_type})"
        elif test_type == "band_comparison":
            title = f"Band interaction: (HC−MDD) {band_type} − full ({atlas_type})"
        elif test_type == "method_comparison":
            title = f"Method interaction: (HC−MDD) MEMD − Band-pass ({atlas_type}, {band_type})"

    plt.title(title)
    plt.xlabel("Network")
    plt.ylabel("Network")

    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label("Δ" if test_type == "hc_mdd" else "ΔΔ")

    # Significance overlay (upper triangle only)
    for i in range(delta_obs.shape[0]):
        for j in range(i + 1, delta_obs.shape[1]):
            for k, a in reversed(list(enumerate(alpha))):
                if p_values[i, j] < a:
                    plt.text(j, i, "*" * (k + 1), ha="center", va="center", color="white", fontsize=8)
                    plt.text(i, j, "*" * (k + 1), ha="center", va="center", color="white", fontsize=8)
                    break

    decomp_dir = decomp_method if test_type != "method_comparison" else "method_comparison"
    save_path = os.path.join(out_dir, "permutation_test_results", decomp_dir, "plots", test_type if test_type != "method_comparison" else "")
    os.makedirs(save_path, exist_ok=True)
    file_name = f"perm_test_{test_type}_{task_type}_{f'{atlas_type}_' if atlas_type else ''}{band_type}_{'networks' if network_means else 'full_parcels'}.svg"
    save_path = os.path.join(save_path, file_name)

    network_labels = list_networks()
    plt.xticks(ticks=np.arange(len(network_labels)), labels=network_labels, rotation=90, fontsize=7)
    plt.yticks(ticks=np.arange(len(network_labels)), labels=network_labels, fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, format='svg')
    plt.close()




def plot_combined_with_runs(
    test_type: str,
    atlas_type: str,
    band_type: str,
    decomp_method: str,
    out_dir: str = DATA_DIR,
    upper_triangle_only: bool = True,
    use_fdr_pvals: bool = False,
):
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

    # P-value significance overlay (upper triangle only)
    def star_count(p, alphas=(0.05, 0.01, 0.001)):
        if p < alphas[2]:
            return 3
        elif p < alphas[1]:
            return 2
        elif p < alphas[0]:
            return 1
        return 0


    alphas = (0.05, 0.01, 0.001)

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
                
    from matplotlib.lines import Line2D

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
    fig.suptitle(sup_title, fontsize=10)

    fig.legend(
        handles=legend_elements,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.95),
        ncol=2,
        frameon=False,
        fontsize=6
    )

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
    for ax in [ax_ap, ax_pa]:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")

    # Shared colorbar on far right
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(cbar_label, fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    decomp_dir = decomp_method if test_type != "method_comparison" else "method_comparison"
    save_path = os.path.join(out_dir, "permutation_test_results", decomp_dir, "plots", test_type if test_type != "method_comparison" else "")
    os.makedirs(save_path, exist_ok=True)

    if use_fdr_pvals:
        out_path = os.path.join(save_path, f"combined_perm_test_{test_type}_{band_type}_fdr_corrected.svg")
    else:
        out_path = os.path.join(save_path, f"combined_perm_test_{test_type}_{band_type}.svg")

    print(f"Saving combined plot to {out_path}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, format=os.path.splitext(out_path)[1][1:] or "svg", bbox_inches="tight")



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

    # --- Networks / labels ---
    networks = list_networks()
    n = len(networks)
    if n == 0:
        raise ValueError("list_networks returned an empty list; cannot label axes.")

    rFC_HC_means = []
    rFC_MDD_means = []
    deltas = []
    pvals = []

    # --- Load + compute means per band ---

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

    # --- Figure layout ---
    fig, axes = plt.subplots(4, 3, figsize=(18, 22), constrained_layout=True)

    mean_vmin, mean_vmax = -1, 1
    delta_abs_max = float(np.max(np.abs(deltas)))
    if delta_abs_max == 0:
        delta_abs_max = 1e-6

    def _apply_axis_labels(ax):
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(networks, rotation=90, fontsize=7)
        ax.set_yticklabels(networks, fontsize=7)

    def _add_sig_rectangles(ax, p_mat, alpha_levels, upper_triangle_only=True):
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

    delta_title = "Δ mean z-FC (HC − MDD)" if test_stat_sign == "hc_mdd" else "Δ mean z-FC (MDD − HC)"

    # --- Plot per band ---
    for i, band in enumerate(band_types):
        # HC mean rFC
        ax0 = axes[i, 0]
        im0 = ax0.imshow(rFC_HC_means[i], cmap=cmap, vmin=mean_vmin, vmax=mean_vmax)
        ax0.set_title(f"HC mean rFC ({band})")
        _apply_axis_labels(ax0)
        _add_sig_rectangles(ax0, pvals[i], alpha, upper_triangle_only=True)
        fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.02)

        # Delta + significance stars
        ax1 = axes[i, 1]
        im1 = ax1.imshow(deltas[i], cmap=cmap, vmin=-delta_abs_max, vmax=delta_abs_max)
        ax1.set_title(f"{delta_title} ({band})")
        _apply_axis_labels(ax1)
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.02)

        # MDD mean rFC
        ax2 = axes[i, 2]
        im2 = ax2.imshow(rFC_MDD_means[i], cmap=cmap, vmin=mean_vmin, vmax=mean_vmax)
        ax2.set_title(f"MDD mean rFC ({band})")
        _apply_axis_labels(ax2)
        _add_sig_rectangles(ax2, pvals[i], alpha, upper_triangle_only=True)
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

    out_name = f"perm_grid_{task_type}_{atlas_type}_{'networks' if network_means else 'parcels'}.svg"
    out_path = os.path.join(out_dir, "permutation_test_results", decomp_method, "plots", "hc_mdd", out_name)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.savefig(out_path, format='svg')
    plt.close(fig)

    return out_path


def plot_perm_test_results_grid2(
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
      cols: [mean FC HC, delta_obs with significance stars, mean FC MDD]
    """

    band_types = ("full", "slow3", "slow4", "slow5")

    # --- Networks / labels ---
    networks = list_networks()
    n = len(networks)
    if n == 0:
        raise ValueError("list_networks returned an empty list; cannot label axes.")

    rFC_HC_PA_means = []
    rFC_MDD_PA_means = []
    rFC_HC_AP_means = []
    rFC_MDD_AP_means = []
    deltas_PA = []
    pvals_PA = []
    deltas_AP = []
    pvals_AP = []

    # --- Load + compute means per band ---
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

    # --- Figure layout ---
    # fig, axes = plt.subplots(figsize=(32, 22), constrained_layout=True)
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

    def _apply_axis_labels(ax):
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(networks, rotation=90, fontsize=7)
        ax.set_yticklabels(networks, fontsize=7)
        ax.tick_params(axis='both', which='both', length=0)
    
    def _apply_axis_labels_xonly(ax):
        ax.set_xticks(np.arange(n))
        ax.set_xticklabels(networks, rotation=90, fontsize=7)
        ax.set_yticks([])
        ax.tick_params(axis='both', which='both', length=0)

    def _apply_axis_labels_yonly(ax):
        ax.set_yticks(np.arange(n))
        ax.set_yticklabels(networks, fontsize=7)
        ax.set_xticks([])
        ax.tick_params(axis='both', which='both', length=0)
    
    def _apply_no_axis_labels(ax):
        ax.set_xticks([])
        ax.set_yticks([])

    def _add_sig_rectangles(ax, p_mat, alpha_levels, upper_triangle_only=True):
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
    fig.patches.append(rect)

    # --- Plot per band ---
    for i, band in enumerate(band_types):
        # HC PA mean rFC
        ax0 = axes[i, 0]
        im0 = ax0.imshow(rFC_HC_PA_means[i], cmap=cmap, vmin=mean_vmin, vmax=mean_vmax)
        # ax0.set_title(f"HC PA mean rFC ({band})")
        if i == len(band_types) - 1:
            _apply_axis_labels(ax0)
        else:
            _apply_axis_labels_yonly(ax0)
        text = ax0.set_title(band, rotation=90, fontsize=16, x = -0.25, y = 0.5)
        text.set_bbox(subplot_label_rectprops)
        _add_sig_rectangles(ax0, pvals_PA[i], alpha, upper_triangle_only=True)
        if i == 0:
            fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.02, location='top')

        # Delta PA + significance stars
        ax1 = axes[i, 1]
        im1 = ax1.imshow(deltas_PA[i], cmap=cmap, vmin=-delta_abs_max, vmax=delta_abs_max)
        # ax1.set_title(f"{delta_title} ({band})")
        # _apply_axis_labels(ax1)
        if i == len(band_types) - 1:
            _apply_axis_labels_xonly(ax1)
        else:
            _apply_no_axis_labels(ax1)
        if i == 0:
            fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.02, location='top')

        # MDD PA mean rFC
        ax2 = axes[i, 2]
        im2 = ax2.imshow(rFC_MDD_PA_means[i], cmap=cmap, vmin=mean_vmin, vmax=mean_vmax)
        # ax2.set_title(f"MDD PA mean rFC ({band})")
        # _apply_axis_labels(ax2)
        if i == len(band_types) - 1:
            _apply_axis_labels_xonly(ax2)
        else:
            _apply_no_axis_labels(ax2)
        _add_sig_rectangles(ax2, pvals_PA[i], alpha, upper_triangle_only=True)
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
            _apply_axis_labels_xonly(ax3)
        else:
            _apply_no_axis_labels(ax3)
        _add_sig_rectangles(ax3, pvals_AP[i], alpha, upper_triangle_only=True)
        if i == 0:
            fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.02, location='top')

        # Delta AP + significance stars
        ax4 = axes[i, 4]
        im4 = ax4.imshow(deltas_AP[i], cmap=cmap, vmin=-delta_abs_max, vmax=delta_abs_max)
        # ax4.set_title(f"{delta_title} ({band})")
        # _apply_axis_labels(ax4)
        if i == len(band_types) - 1:
            _apply_axis_labels_xonly(ax4)
        else:
            _apply_no_axis_labels(ax4)
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
            _apply_axis_labels_xonly(ax5)
        else:
            _apply_no_axis_labels(ax5)
        _add_sig_rectangles(ax5, pvals_AP[i], alpha, upper_triangle_only=True)
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

    title = fig.suptitle(
        f"Permutation test results ("
        f"{'networks' if network_means else 'parcels'})",
        fontsize=16,
        horizontalalignment='center',
    )
    title.set_bbox({'facecolor': '#fef2e2', 'boxstyle': 'round,pad=0.5'})
    


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
    test_stat_sign: str = "hc_mdd",
    out_dir: str = DATA_DIR,
    decomp_method: str = "memd"
):
    """
    Creates a 2x4 grid plot:
      rows: Schaefer400, Yan2023 (delta_obs)
      cols: full, slow3, slow4, slow5
    """
    band_types = ("full", "slow3", "slow4", "slow5")

    # --- Networks / labels ---
    networks = list_networks()
    n = len(networks)
    if n == 0:
        raise ValueError("list_networks returned an empty list; cannot label axes.")

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

    # --- Figure layout ---
    fig, axes = plt.subplots(2, 4, figsize=(20, 10.4), constrained_layout=True)
    delta_abs_max = float(np.max(np.abs([deltas_Yan2023, deltas_Schaefer400])))
    if delta_abs_max == 0:
        delta_abs_max = 1e-6

    def _remove_ticks_(ax):
        ax.set_xticks([])
        ax.set_yticks([])

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

    for i, band in enumerate(band_types):
        # Yan2023 delta
        ax0 = axes[0, i]
        im0 = ax0.imshow(deltas_Yan2023[i], cmap=cmap, vmin=-delta_abs_max, vmax=delta_abs_max)
        _remove_ticks_(ax0)
        
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
        _remove_ticks_(ax1)

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
    out_name = f"perm_delta_atlas_comp_{task_type}_{'networks' if network_means else 'parcels'}.svg"
    out_path = os.path.join(out_dir, "permutation_test_results", decomp_method, "plots", "atlas_comparison", out_name)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, format='svg')
    plt.close(fig)