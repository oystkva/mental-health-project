import sys, os
import warnings
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

import MEMD_all as memd_utils

from src.data_loader import (
    load_h5_file, 
    load_subject_list,
    list_bold_h5_file_paths,
)
from src.config import (
    TR,
    N_CPUs,
    #region path configurations
    BOLD_DIR,
    LOG_DIR,
    #endregion
)
from src.utils import extract_run_id, extract_subjectkey_from_subdir


#region Memd function
# =============================================================================

##### Customized memd function (adapted from src.memd_all.py) to include progress bar (no other changes) ######

def memd(*args):
    x, seq, t, ndir, N_dim, N, sd, sd2, tol, nbit, MAXITERATIONS, stop_crit, stp_cnt = memd_utils.set_value(args)

    r = x
    n_imf = 1
    q = []
    # === Add tqdm to main loop ===
    with tqdm(desc="Extracting IMFs", unit="IMF") as pbar:
        while memd_utils.stop_emd(r, seq, ndir, N_dim) == False:
            # current mode
            m = r

            # computation of mean and stopping criterion
            if stop_crit == 'stop':
                stop_sift, env_mean = memd_utils.stop(m, t, sd, sd2, tol, seq, ndir, N, N_dim)
            else:
                counter = 0
                stop_sift, env_mean, counter = memd_utils.fix(m, t, seq, ndir, stp_cnt, counter, N, N_dim)

            # In case the current mode is so small that machine precision can cause
            # spurious extrema to appear
            if np.max(np.abs(m)) < (1e-10)*(np.max(np.abs(x))):
                if stop_sift == False:
                    warnings.warn('forced stop of EMD : too small amplitude', category=UserWarning)
                else:
                    print('forced stop of EMD : too small amplitude')
                break

            # sifting loop
            while stop_sift == False and nbit < MAXITERATIONS:
                # sifting
                m = m - env_mean

                # computation of mean and stopping criterion
                if stop_crit =='stop':
                    stop_sift, env_mean = memd_utils.stop(m, t, sd, sd2, tol, seq, ndir, N, N_dim)
                else:
                    stop_sift, env_mean, counter = memd_utils.fix(m, t, seq, ndir, stp_cnt, counter, N, N_dim)

                nbit = nbit + 1

                if nbit == (MAXITERATIONS-1) and  nbit > 100:
                    warnings.warn('forced stop of sifting : too many iterations', category=UserWarning)

            q.append(m.transpose())

            n_imf = n_imf + 1
            r = r - m
            nbit = 0

            # === Update progress bar ===
            pbar.update(1)
            pbar.set_postfix({"IMFs extracted": n_imf-1})

    # Stores the residue
    q.append(r.transpose())
    q = np.asarray(q)
    #sprintf('Elapsed time: %f\n',toc);

    return q

# =============================================================================
#endregion

def create_subject_tensor(subject_file: str) -> tuple:
    """
    Create a 3D tensor from fMRI data files for multiple subjects.
    Args:
        subject_file (str): Path to the subject's fMRI h5 file.
    Returns:
        tuple: (list of subject keys, 3D numpy array of fMRI data)
    """
    subjectkeys = []
    data = np.array([])
    fmri_data = load_h5_file(subject_file)
    for key in fmri_data.keys():
        bold_signals = fmri_data[key]
        if bold_signals.shape[0] != 434:
            continue
        elif data.size == 0:
            data = bold_signals[np.newaxis, :, :]
            subjectkeys.append(key)
        else:
            data = np.concatenate((data, bold_signals[np.newaxis, :, :]), axis=0)
            subjectkeys.append(key)
    return subjectkeys, data

def subject_memd_decomposition(subjectkey: str, data: np.ndarray, task_type: str, out_dir: str, run_id: str) -> None:
    """
    Find IMFs using MEMD and save the results to disk.
    Args:
        subjectkey (str): Subject key identifier.
        data (np.ndarray): fMRI data array.
        out_dir (str): Directory to save the results.
        run_id (str): Run identifier.
    """
    subject = extract_subjectkey_from_subdir(subjectkey)
    if subject not in run_id:
        raise ValueError(f"Subject key {subjectkey} does not match run ID {run_id}.")
    imfs = memd(data[0], 4*434)
    np.save(os.path.join(out_dir, task_type, f"{run_id}_MEMD_imfs.npy"), imfs)
    return

#TODO: Add logging

def run_memd_pipeline(subject_list: list[str], out_dir: str, group: str, task_type: str = "restPA", cortical_atlas: str = "Schaefer400", n_parallels: int = N_CPUs) -> None:
    """
    Run the MEMD decomposition pipeline for a list of subjects.
    Args:
        subject_list (list[str]): List of subject keys.
        group (str): Subject group identifier (e.g., "MDD" or "HC").
        task_type (str): Type of fMRI task (e.g., "restPA").
        cortical_atlas (str): Name of the cortical atlas used.
        out_dir (str): Directory to save the processed results.
        n_parallels (int): Number of parallel jobs to run.
    Returns:
        None
        Results are saved to data/MEMD_processed/.
    """
    files = list_bold_h5_file_paths(BOLD_DIR, subject_list, group=group, task_type=task_type, cortical_atlas=cortical_atlas)
    
    data = np.array([])
    subject_keys = []
    for file in files:
        subject_key, subject_data = create_subject_tensor(file)
        subject_keys.extend(subject_key)
        if data.size == 0:
            data = subject_data[np.newaxis, :, :]
        else:
            data = np.concatenate((data, subject_data[np.newaxis, :, :]), axis=0)

    Parallel(n_jobs=n_parallels)(
        delayed(subject_memd_decomposition)(
            subject_keys[idx], 
            data[idx],
            task_type,
            out_dir, 
            extract_run_id(files[idx])
        ) for idx in tqdm(range(len(subject_keys)))
    )
    return