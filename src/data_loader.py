import os, sys
import numpy as np
import pandas as pd
import h5py as h5
from typing import Optional

from src.config import PROJECT_ROOT
from src.utils import subjectkey_to_subdir

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
