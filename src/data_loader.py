import os, sys
import numpy as np
import pandas as pd
import h5py as h5
from typing import Optional
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import PROJECT_ROOT, DATA_DIR
from src.utils import subjectkey_to_subdir
from src.atlas_config import list_networks

def recursively_load(data: dict, name: str, obj) -> None:
    if isinstance(obj, h5.Dataset):
        data[name] = obj[()]
    elif isinstance(obj, h5.Group):
        for key, val in obj.items():
            recursively_load(data, f"{name}/{key}", val)


def load_h5_file(file_path: str) -> dict:
    """
    Load an h5 file and return its contents as a dictionary.
    Args:
        file_path (str): Path to the h5 file.
    Returns:
        dict: Dictionary containing the h5 file contents.
    """
    with h5.File(file_path, 'r') as h5_file:
        data = {}
        recursively_load(data, '', h5_file)
    return data


def load_subject_list(file_path: str) -> list[str]:
    """
    Load a list of subject keys from a text file.
    Args:
        file_path (str): Path to the text file containing subject keys.
    Returns:
        list[str]: List of subject keys.
    """
    with open(file_path, 'r') as f:
        subject_keys = [line.strip() for line in f if line.strip()]
    return subject_keys


def list_bold_h5_file_paths(
    data_dir: str, 
    subject_keys: list[str], 
    group: str, 
    task_type: str, 
    cortical_atlas: str = "Yan2023"
) -> list[str]:
    """
    List fMRI file paths for given subject keys and run types. Prints suubjects where no files were found.
    Args:
        data_dir (str): Base directory containing subject folders.
        subject_keys (list[str]): List of subject keys.
        group (str): Subject group, e.g., "MDD" or "HC".
        cortical_atlas (str): Name of the cortical atlas used in file naming.
    Returns:
        list[str]: List of fMRI file paths.
    """
    fmri_files = []
    files_found = {subject: [] for subject in subject_keys}
    for subject in subject_keys:
        if os.path.exists(os.path.join(data_dir, cortical_atlas, task_type, f"{group}_{subject}_{task_type}_run01_{cortical_atlas}_BOLD_signals.h5")):
            fmri_files.append(os.path.join(data_dir, cortical_atlas, task_type, f"{group}_{subject}_{task_type}_run01_{cortical_atlas}_BOLD_signals.h5"))
            files_found[subject].append(1)
        if os.path.exists(os.path.join(data_dir, cortical_atlas, task_type, f"{group}_{subject}_{task_type}_run02_{cortical_atlas}_BOLD_signals.h5")):
            fmri_files.append(os.path.join(data_dir, cortical_atlas, task_type, f"{group}_{subject}_{task_type}_run02_{cortical_atlas}_BOLD_signals.h5"))
            files_found[subject].append(1)
        if not any(files_found[subject]):
            files_found[subject].append(0)
    print(f"Found {len(fmri_files)} files out of {len(subject_keys)} expected.")            
    if any(sum(found) == 0 for found in files_found.values()):
        missing_subjects = [subject for subject, found in files_found.items() if sum(found) == 0]
        print(f"Missing files for subjects:")
        for subject in missing_subjects:
            print(f" - {subject}")
    
    return fmri_files


def list_fmri_nii_file_paths(
    data_dir: str,
    subjectkey: str,
    fmri_run_types: list[list[str]],
) -> list[str]:
    """
    Find and return fMRI NIfTI file paths for given subject keys and run types.
    Args:
        data_dir (str): Base directory containing subject folders.
        subject_keys (list[str]): List of subject keys.
        fmri_run_types (list[list[str]]): List of fMRI run type specifications. Each specification is a list of strings that should be part of the filename. List should be structured as [['restAP', 'run-01', 'MNI152NLin2009cAsym_res-2_desc-preproc_bold']] where restAP/restPA and space-MNI152NLin2009cAsym_res-2_desc-preproc_bold should be included and run-01/run-02 can used if desired.
    Returns:
        list[str]: List of fMRI NIfTI file paths.
    """
    fmri_file_paths = []
    subdir = subjectkey_to_subdir(subjectkey)
    subject_dir = os.path.join(data_dir, subdir, "func")
    if not os.path.exists(subject_dir):
        print(f"No func directory for subject {subdir} at {subject_dir}, skipping.")

    for root, _, files in os.walk(subject_dir):
        for f in files:
            if f.endswith(".nii.gz"):
                if any(all(run_type_part in f for run_type_part in run_type) for run_type in fmri_run_types):
                    fmri_file_paths.append(os.path.join(root, f))
    print(f"Found {len(fmri_file_paths)} NIfTI files for {subjectkey}.")
    return fmri_file_paths


def save_BOLD_signals_h5(
    bold_signals: dict,
    out_dir: str,
    subjectkey: str,
    atlas_paths: dict,
    TR: float = 0.8,
    n_runs: int = 1,
):
    """
    Save parcellated BOLD signals to HDF5 files.
    Args:
        bold_signals (dict): Dictionary with run file paths as keys and parcellated BOLD time series as values.
        out_dir (str): Directory to save the HDF5 file.
        subjectkey (str): Subject identifier.
        atlas_paths (dict): Dictionary of atlas names and their file paths.
        TR (float): Repetition time of the fMRI data.
        n_runs (int): Number of fMRI runs.
    Returns:
        None
    """
    print(f"[IO] Saving BOLD signals for subject {subjectkey} to {out_dir}...")

    with h5.File(out_dir, "w") as f:
        for run_name, data in bold_signals.items():
            # Store each run in its own dataset
            f.create_dataset(os.path.basename(run_name).replace(".nii.gz", ""), data=data)
        f.attrs["subject"] = subjectkey
        f.attrs["atlas"] = ", ".join([os.path.basename(a) for a in atlas_paths.values()])
        f.attrs["TR"] = TR
        f.attrs["n_runs"] = n_runs
        print(f"[IO] Saved: {out_dir}")


def load_zFCs(
    group: str, 
    task_type: str, 
    band_type: str, 
    include_all_runs: bool = False,
    atlas_type: str = "Yan2023",
    network_means: bool = True,
    decomp_method: str = "memd",
    vectorize: bool = False,
) -> np.ndarray:
    """
    Load Fisher Z-transformed functional connectivity matrices for a given group.
    Args:
        group (str): Group identifier ('MDD'/'HC').
        task_type (str): Task type ('restAP'/'restPA').
        band_type (str): Band type ('full'/'slow5'/'slow4'/'slow3').
        include_all_runs (bool): Whether to use all available runs for each subject or just the first (run-01).
        atlas_type (str): Atlas type ('Schaefer400'/'Yan2023').
        network_means (bool): Whether to use network means or parcel-level data (default: True).
        decomp_method (str): Decomposition method used ('memd', 'bandpass', etc.) to determine the directory structure.
        vectorize (bool): Vectorize the zFCs (for ML appliances) 
    Returns:
        np.ndarray: Fisher Z-transformed FC matrices for the specified group.
    """
    parcellation = "networks" if network_means else "full_parcels"
    if decomp_method == "memd" or band_type == "full":
        load_dir = os.path.join(DATA_DIR, "zFC_matrices", "with_diagonal", atlas_type, task_type, parcellation)
    elif decomp_method == "bandpass" and band_type != "full":
        load_dir = os.path.join(DATA_DIR, "zFC_matrices", "with_diagonal", atlas_type, task_type + "_bandpass", parcellation)
    zFCs = np.empty((0, ))
    run = "run01" if not include_all_runs else ""
    for root, _, files in list(os.walk(load_dir)):
        for f in files:
            if f.endswith(".npy"):
                if all(param in f for param in [group, atlas_type, band_type, run]):
                    file_path = os.path.join(root, f)
                    loaded_zFCs = np.load(file_path)[np.newaxis, ...]
                    if zFCs.size == 0:
                        zFCs = loaded_zFCs
                    else:
                        zFCs = np.concatenate((zFCs, loaded_zFCs), axis=0)
    if vectorize:
        return vectorize_zFCs(zFCs)
    return zFCs


def load_mean_zFCs(
    group: str,
    band_type: str,
    include_all_runs: bool = False,
    atlas_type: str = "Yan2023",
    network_means: bool = True,
    decomp_method: str = "memd",
    vectorize: bool = False,
) -> np.ndarray:
    """Load Fisher Z-transformed functional connectivity matrices for a given group, computed as the mean across restAP and restPA runs.
    Args:
        group (str): Group identifier ('MDD'/'HC').
        band_type (str): Band type ('full'/'slow5'/'slow4'/'slow3').
        include_all_runs (bool): Whether to use all available runs for each subject or just the first (run-01).
        atlas_type (str): Atlas type ('Schaefer400'/'Yan2023').
        network_means (bool): Whether to use network means or parcel-level data (default: True).
        decomp_method (str): Decomposition method used ('memd', 'bandpass', etc.) to determine the directory structure.
        vectorize (bool): Vectorize the zFCs (for ML appliances) 
    Returns:
        np.ndarray: Fisher Z-transformed FC matrices for the specified group, averaged across restAP and restPA runs.
    """
    parcellation = "networks" if network_means else "full_parcels"
    if decomp_method == "memd" or band_type == "full":
        load_dir = os.path.join(DATA_DIR, "zFC_matrices", atlas_type, "restAP", parcellation)
    elif decomp_method == "bandpass" and band_type != "full":
        load_dir = os.path.join(DATA_DIR, "zFC_matrices", atlas_type, "restAP_bandpass", parcellation)
    mean_zFCs = np.empty((0, ))
    for root, _, files in list(os.walk(load_dir)):
        for f in files:
            if f.endswith(".npy"):
                if all(param in f for param in [group, atlas_type, band_type, "run01"]):
                    file_path = os.path.join(root, f)
                    zFC_AP = np.load(file_path)[np.newaxis, ...]
                    if include_all_runs and os.path.isfile(file_path.replace("run01", "run02")):
                        zFC_AP = np.mean(np.concatenate((zFC_AP, np.load(file_path.replace("run01", "run02"))[np.newaxis, ...])), axis=0)[np.newaxis, ...] # avg runs first
                    file_path = file_path.replace("restAP", "restPA")
                    zFC_PA = np.load(file_path)[np.newaxis, ...]
                    if include_all_runs and os.path.isfile(file_path.replace("run01", "run02")):
                        zFC_PA = np.mean(np.concatenate((zFC_PA, np.load(file_path.replace("run01", "run02"))[np.newaxis, ...])), axis=0)[np.newaxis, ...] # avg runs first
                    avg_zFC = np.mean(np.concatenate((zFC_AP, zFC_PA), axis=0), axis=0)                                            # avg AP and PA
                    if mean_zFCs.size == 0:
                        mean_zFCs = avg_zFC[np.newaxis, ...]
                    else:
                        mean_zFCs = np.concatenate((mean_zFCs, avg_zFC[np.newaxis, ...]), axis=0)
    if vectorize:
        return vectorize_zFCs(mean_zFCs)
    return mean_zFCs


def load_perm_test_results(
    test_type: str, 
    task_type: str, 
    band_type: str, 
    atlas_type: Optional[str] = None, 
    network_means: bool = True,
    decomp_method: str = "memd",
    use_fdr_pvals: bool = False,
    include_all_runs: bool = False
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
        include_all_runs (bool): Whether to use all available runs for each subject or just the first (run-01).
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

    run_dir = "all_runs" if include_all_runs else "single_run" 

    delta_obs_path = os.path.join(
        DATA_DIR, "permutation_test_results", 
        run_dir, decomp_dir, test_type,
        f"delta{'_delta' if test_type != 'hc_mdd' else ''}_obs_{file_name_id}.npy"
    )
    p_values_path = os.path.join(
        DATA_DIR, "permutation_test_results", 
        run_dir, decomp_dir, test_type, 
        f"p_{'fdr_' if use_fdr_pvals else ''}values_{file_name_id}.npy"
    )
    delta_obs = np.load(delta_obs_path)
    p_values = np.load(p_values_path)

    return delta_obs, p_values


def vectorize_zFCs(zFCs: np.ndarray) -> np.ndarray:
    """
    Vectorize zFC matrices.
    
    Args:
        zFCs (np.ndarray): Fisher-z transformed FC array of shape (N, N) or an array of these of shape (S, N, N)
    Output:
        np.ndarray: vectorizations of the zFC matrices of shape (M, ) or (S, M) with M = (N^2+N)/2 [Upper Triangle Elements]
    """
    single_output = False
    if zFCs.ndim == 2:
        zFCs = zFCs[np.newaxis, ...]
        single_output = True
    
    N, n, _ = zFCs.shape

    mask = np.triu_indices(n, k=0)
    zFC_vecs = zFCs[:, mask[0], mask[1]]

    if single_output:
        return zFC_vecs[0]
    return zFC_vecs


def load_zFC_df(
    band_type: str = 'all',
    task_type: str = 'restAP',
    include_all_runs: bool = False
):
    n = list(list_networks().keys())
    
    if band_type == 'all':
        feature_labels = []
        # for band_label in ['s3', 's4', 's5']: 
        band_label = "3 bands"   
        for idx, i in enumerate(n):
            feature_labels.extend([f'{i} - {j} ({band_label})' for j in n[idx:]])
        if task_type == 'all':
            X_HC = np.concatenate([
                load_zFCs(group='HC', task_type='restAP', band_type='slow3', include_all_runs=include_all_runs, vectorize=True),
                load_zFCs(group='HC', task_type='restAP', band_type='slow4', include_all_runs=include_all_runs, vectorize=True),
                load_zFCs(group='HC', task_type='restAP', band_type='slow5', include_all_runs=include_all_runs, vectorize=True),
                load_zFCs(group='HC', task_type='restPA', band_type='slow3', include_all_runs=include_all_runs, vectorize=True),
                load_zFCs(group='HC', task_type='restPA', band_type='slow4', include_all_runs=include_all_runs, vectorize=True),
                load_zFCs(group='HC', task_type='restPA', band_type='slow5', include_all_runs=include_all_runs, vectorize=True) 
            ], axis=0)
            X_MDD = np.concatenate([
                load_zFCs(group='MDD', task_type='restAP', band_type='slow3', include_all_runs=include_all_runs, vectorize=True),
                load_zFCs(group='MDD', task_type='restAP', band_type='slow4', include_all_runs=include_all_runs, vectorize=True),
                load_zFCs(group='MDD', task_type='restAP', band_type='slow5', include_all_runs=include_all_runs, vectorize=True),
                load_zFCs(group='MDD', task_type='restPA', band_type='slow3', include_all_runs=include_all_runs, vectorize=True),
                load_zFCs(group='MDD', task_type='restPA', band_type='slow4', include_all_runs=include_all_runs, vectorize=True),
                load_zFCs(group='MDD', task_type='restPA', band_type='slow5', include_all_runs=include_all_runs, vectorize=True) 
            ], axis=0)
        else:
            X_HC = np.concatenate([
                load_zFCs(group='HC', task_type=task_type, band_type='slow3', include_all_runs=include_all_runs, vectorize=True),
                load_zFCs(group='HC', task_type=task_type, band_type='slow4', include_all_runs=include_all_runs, vectorize=True),
                load_zFCs(group='HC', task_type=task_type, band_type='slow5', include_all_runs=include_all_runs, vectorize=True) 
            ], axis=0)
            X_MDD = np.concatenate([
                load_zFCs(group='MDD', task_type=task_type, band_type='slow3', include_all_runs=include_all_runs, vectorize=True),
                load_zFCs(group='MDD', task_type=task_type, band_type='slow4', include_all_runs=include_all_runs, vectorize=True),
                load_zFCs(group='MDD', task_type=task_type, band_type='slow5', include_all_runs=include_all_runs, vectorize=True) 
            ], axis=0)
    else:
        band_label = band_type[0]+band_type[-1]
        feature_labels = []
        for idx, i in enumerate(n):
            feature_labels.extend([f'{i} - {j} ({band_label})' for j in n[idx:]])
        if task_type == 'all':
            X_HC = np.concatenate([
                load_zFCs(group='HC', task_type='restAP', band_type=band_type, include_all_runs=include_all_runs, vectorize=True),
                load_zFCs(group='HC', task_type='restPA', band_type=band_type, include_all_runs=include_all_runs, vectorize=True)
            ])
            X_MDD = np.concatenate([
                load_zFCs(group='MDD', task_type='restAP', band_type=band_type, include_all_runs=include_all_runs, vectorize=True),
                load_zFCs(group='MDD', task_type='restPA', band_type=band_type, include_all_runs=include_all_runs, vectorize=True)
            ])
        else:
            X_HC = load_zFCs(group='HC', task_type=task_type, band_type=band_type, include_all_runs=include_all_runs, vectorize=True)
            X_MDD = load_zFCs(group='MDD', task_type=task_type, band_type=band_type, include_all_runs=include_all_runs, vectorize=True)
    
    X = np.concatenate([X_HC, X_MDD], axis=0)
    df = pd.DataFrame(X, columns=feature_labels)
    labels = np.concatenate([
        np.zeros(len(X_HC), dtype=bool),
        np.ones(len(X_MDD), dtype=bool)
    ], axis=0)

    # TODO?: Add subjectkeys
    # subjectkeys = []
    # df.insert(loc=0, column='subjectkey', value=subjectkeys)
    df['MDD'] = labels

    return df