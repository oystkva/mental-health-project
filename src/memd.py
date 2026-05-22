import sys, os
import warnings
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import src.MEMD_all as memd_utils

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
from src.utils import (
    extract_run_id, 
    extract_subjectkey_from_subdir, 
    log_message
)


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

def create_subject_tensor(subject_file: str) -> tuple[str, np.ndarray]:
    """
    Create a 3D tensor from fMRI data files for a single subject run.
    Args:
        subject_file (str): Path to the subject's fMRI h5 file.
    Returns:
        tuple: (subject key, 2D numpy array of fMRI data of shape (C, T) which usually is (434, 488))
    """
    fmri_data = load_h5_file(subject_file)
    if len(fmri_data) != 1:
        raise ValueError(f"Expected exactly one run in the fMRI data, but found {len(fmri_data.values())}.")
    key = next(iter(fmri_data.keys()))
    return key, fmri_data[key]


def subject_memd_decomposition(
    subjectkey: str, 
    data: np.ndarray, 
    out_dir: str, 
    task_type: str,
    atlas: str,  
    run_id: str
) -> None:
    """
    Find IMFs using MEMD and save the results to disk.
    Args:
        subjectkey (str): Subject key identifier.
        data (np.ndarray): fMRI data array of shape (C, T). Usually (434, 488).
        out_dir (str): Directory to save the results.
        task_type (str): Type of fMRI task (restAP/restPA).
        cortical_atlas (str): Name of the cortical atlas used.
        run_id (str): Run identifier.
    """
    log_path = os.path.join(LOG_DIR, "memd.log")
    log_message(f"Starting MEMD for run {run_id}", log_path)

    subject = extract_subjectkey_from_subdir(subjectkey)
    if subject not in run_id:
        raise ValueError(f"Subject key {subjectkey} does not match run ID {run_id}.")
    imfs = memd(data, 4*434)
    np.save(os.path.join(out_dir, atlas, task_type, f"{run_id}_MEMD_imfs.npy"), imfs)

    log_message(f"{run_id} decomposed", log_path)
    return

#TODO: Add logging

def run_memd_pipeline(
    subject_list: list[str], 
    out_dir: str, 
    group: str, 
    task_type: str = "restPA", 
    cortical_atlas: str = "Yan2023", 
    n_parallels: int = N_CPUs
) -> None:
    """
    Run the MEMD decomposition pipeline for a list of subjects.
    Args:
        subject_list (list[str]): List of subject keys.
        group (str): Subject group identifier (e.g., "MDD" or "HC").
        task_type (str): Type of fMRI task (restAP/restPA).
        cortical_atlas (str): Name of the cortical atlas used.
        out_dir (str): Directory to save the processed results.
        n_parallels (int): Number of parallel jobs to run.
    Returns:
        None
        Results are saved to data/MEMD_processed/.
    """
    log_path = os.path.join(LOG_DIR, "memd.log")

    log_message(f"Starting MEMD pipeline for {group} - {task_type} runs with atlas {cortical_atlas}", log_path)
    files = list_bold_h5_file_paths(
        data_dir=BOLD_DIR, 
        subject_keys=subject_list, 
        group=group, 
        task_type=task_type, 
        cortical_atlas=cortical_atlas
    )
    
    
    memd_out_dir = os.path.join(out_dir, cortical_atlas, task_type)
    os.makedirs(memd_out_dir, exist_ok=True)

    jobs = []

    for f in files:
        run_id = extract_run_id(f)
        out_file = f"{run_id}_MEMD_imfs.npy"

        if out_file in os.listdir(memd_out_dir):
            log_message(f"Skipping {run_id} as it has already been processed.", log_path)
            continue

        subject_key, subject_data = create_subject_tensor(f)

        jobs.append({
            "subjectkey": subject_key,
            "data": subject_data,
            "run_id": run_id,
        })    
    
    Parallel(n_jobs=n_parallels)(
        delayed(subject_memd_decomposition)(
            subjectkey=job["subjectkey"],
            data=job["data"],
            out_dir=out_dir,
            task_type=task_type,
            atlas=cortical_atlas,
            run_id=job["run_id"]
        ) for job in tqdm(jobs)
    )
    return