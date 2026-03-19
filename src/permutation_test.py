import os
import numpy as np


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

