import os, sys
import numpy as np
import pandas as pd
import h5py as h5
from typing import Optional

from src.config import PROJECT_ROOT

def recursively_load(data: dict, name: str, obj) -> None:
    if isinstance(obj, h5.Dataset):
        data[name] = obj[()]
    elif isinstance(obj, h5.Group):
        for key, val in obj.items():
            recursively_load(data, f"{name}/{key}", val)


def subjectkey_to_subdir(subject_key: str) -> str:
    """
    Convert a subject key to its corresponding subdirectory name.
    Args:
        subject_key (str): Subject key on form 'NDAR_XXXXXXXXXXX'
    Returns:
        str: Corresponding subdirectory name on form 'sub-NDARXXXXXXXXXXX'
    """
    if not subject_key.startswith("NDAR_"):
        raise ValueError(f"Invalid subject key format: {subject_key}")
    return "sub-" + subject_key.replace("NDAR_", "NDAR")


def extract_subjectkey_from_subdir(subdir: str) -> str:
    """
    Extract the subject key from a subdirectory name.
    Args:
        subdir (str): Subdirectory name on form 'sub-NDARXXXXXXXXXXX_.....'
    Returns:
        str: Corresponding subject key on form 'NDAR_XXXXXXXXXXX'
    """
    if "sub-NDAR" in subdir:
        start_idx = subdir.index("sub-NDAR") + len("sub-")
        subjectkey = subdir[start_idx:].split("_")[0]
        return subjectkey.replace("NDAR", "NDAR_")
    else:
        raise ValueError(f"Invalid subdirectory format: {subdir}")


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


def list_bold_h5_file_paths(data_dir: str, subject_keys: list[str], group: str, task_type: str, cortical_atlas: str = "Schaefer400") -> list[str]:
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
        if os.path.exists(os.path.join(data_dir, task_type, f"{group}_{subject}_{task_type}_run01_{cortical_atlas}_BOLD_signals.h5")):
            fmri_files.append(os.path.join(data_dir, task_type, f"{group}_{subject}_{task_type}_run01_{cortical_atlas}_BOLD_signals.h5"))
            files_found[subject].append(1)
        if os.path.exists(os.path.join(data_dir, task_type, f"{group}_{subject}_{task_type}_run02_{cortical_atlas}_BOLD_signals.h5")):
            fmri_files.append(os.path.join(data_dir, task_type, f"{group}_{subject}_{task_type}_run02_{cortical_atlas}_BOLD_signals.h5"))
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
        fmri_run_types (list[list[str]]): List of fMRI run type specifications. Each specification is a list of strings that should be part of the filename.
        List should be structured as [['restAP', 'run-01', 'MNI152NLin2009cAsym_res-2_desc-preproc_bold']] where restAP/restPA and space-MNI152NLin2009cAsym_res-2_desc-preproc_bold should be included and run-01/run-02 can used if desired.
    Returns:
        list[str]: List of fMRI NIfTI file paths.
    """
    fmri_file_paths = []
    subdir = subjectkey_to_subdir(subjectkey)
    subject_dir = os.path.join(data_dir, subdir, "func")
    if not os.path.exists(subject_dir):
        print(f"No func directory for subject {subdir} at {subject_dir}, skipping.")

    for root, _, files in os.walk(subject_dir):
        for file in files:
            if file.endswith(".nii.gz"):
                if any(all(run_type_part in file for run_type_part in run_type) for run_type in fmri_run_types):
                    fmri_file_paths.append(os.path.join(root, file))
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


def extract_run_id(path: str) -> str:
    """
    Extract the run identifier from a file path of saved data from this project (non-imported data).
    Args:
        path (str): File path.
    Returns:
        str: Extracted run identifier.
    e.g. /path/to/HC_NDAR_INVWU297KRB_restPA_run01_Schaefer400_BOLD_signals.h5 -> 'HC_NDAR_INVWU297KRB_restPA_run01_Schaefer400'
    """
    atlases = ["Schaefer400", "Yan2023"]

    base = os.path.splitext(os.path.basename(path))[0]
    parts = base.split("_")

    for i, token in enumerate(parts):
        if token in atlases:
            return "_".join(parts[:i + 1])

    raise ValueError(f"No known atlas token {atlases} found in filename: {base}")


def list_networks(Yan2023: bool = False) -> dict[str, list[int]]:
    network_map = {}
    if Yan2023:
        network_map.update(list_Yan17_networks())
    else:
        network_map.update(list_Schaefer17_networks())
    network_map.update(list_Tian3_networks())
    network_map.update(list_Buckner1_networks())
    return network_map


def list_Yan17_networks() -> dict[str, list[int]]:
    network_map = {}
    with open(os.path.join(PROJECT_ROOT, "src", "brain_atlases", "Yan2023_homotopic_400Parcels_Kong2022_17Networks_info.txt"), "r") as f:
        lines = f.readlines()
    lines = [network.strip().split("_") for network in lines]
    networks_labels = list(set([line[2] for line in lines[::2]]))
    networks = []
    for i in range(0, len(lines), 2):
        networks.append([int(lines[i+1][0].split(" ")[0])-1, lines[i][2]])
    for label in networks_labels:
        network_map[label] = [network[0] for network in networks if network[1] == label]
    return network_map


def list_Schaefer17_networks() -> dict[str, list[int]]:
    network_map = {}
    with open(os.path.join(PROJECT_ROOT, "src", "brain_atlases", "Schaefer2018_400Parcels_Kong2022_17Networks_order.txt"), "r") as f:
        lines = f.readlines()
    lines = [network.strip().split('\t') for network in lines]
    # networks_labels = list(set([line[1].split("_")[2] for line in lines]))
    # print(networks_labels)
    for i in range(len(lines)):
        if lines[i][1].split("_")[2] not in network_map:
            network_map[lines[i][1].split("_")[2]] = []
        network_map[lines[i][1].split("_")[2]].extend([int(lines[i][0]) - 1])
    return network_map


def list_Tian3_networks() -> dict[str, list[int]]:
    """
    Labels found on: https://github.com/yetianmed/subcortex/blob/master/Group-Parcellation/3T/Cortex-Subcortex/Schaefer2018_400Parcels_17Networks_order_Tian_Subcortex_S2_label.txt
    """
    network_map = {
        "MedialTemporal": [], 
        "Striatal": [], 
        "Thalamic": []
    }
    with open(os.path.join(PROJECT_ROOT, "src", "brain_atlases", "Tian_Subcortex_S2_3T_info.txt"), "r") as f:
        lines = f.readlines()
    lines = [network.strip() for network in lines]
    for i in range(0, len(lines), 2):
        label = lines[i]
        if "HIP" in label.upper() or "AMY" in label.upper():
            network_map["MedialTemporal"].extend([int(lines[i+1].split(" ")[0]) - 1 + 400])
        elif "THA" in label.upper():
            network_map["Thalamic"].extend([int(lines[i+1].split(" ")[0]) - 1 + 400])
        else:
            network_map["Striatal"].extend([int(lines[i+1].split(" ")[0]) - 1 + 400]) 
    return network_map


def list_Buckner1_networks() -> dict[str, list[int]]:
    return {
        "Cerebellar": [432, 433]
    }


print(list_Tian3_networks())