import os, sys
from joblib import Parallel, delayed
import numpy as np
from tqdm import tqdm
from typing import Tuple
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.utils import (
    log_message,
    extract_run_id
)
from src.atlas_config import list_networks
from src.config import (
    TR,
    N_CPUs,
    BOLD_DIR,
    DATA_DIR,
    LOG_DIR,
)
from src.data_loader import load_h5_file
from src.slow_band_extraction import seperate_slow_band_signals, extract_slow_band_signals

#region fisher Z transformation functions
def fisher_r2z(r: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """
    Apply Fisher r-to-z transformation.
    Args:
        r (np.ndarray): Correlation coefficients.
    Returns:
        np.ndarray: Fisher Z-transformed values.
    """
    return np.arctanh(np.clip(r, -1 + eps, 1 - eps))

def fisher_z2r(z: np.ndarray) -> np.ndarray:
    """
    Apply inverse Fisher z-to-r transformation.
    Args:
        z (np.ndarray): Fisher Z-transformed values.
    Returns:
        np.ndarray: Correlation coefficients.
    """
    return np.tanh(z)
#endregion

#region zFC calculation functions
def calculate_zFC(fmri_data: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Calculate the Fisher Z-transformed functional connectivity (FC) matrix.
    Args:
        fmri_data (np.ndarray): 2D array where each row is a brain region's time series.
    Returns:
        np.ndarray: 2D Fisher Z-transformed FC matrix.
        int: Number of samples used in the calculation.
    """
    if fmri_data.shape[0] == 0:
        FC = np.zeros((434, 434))
        return fisher_r2z(FC), 0

    if len(fmri_data.shape) == 2:
        return fisher_r2z(np.corrcoef(fmri_data)), fmri_data.shape[1]

    avg_zFC = np.zeros((fmri_data.shape[1], fmri_data.shape[1]))
    for i in range(fmri_data.shape[0]):
        FC = np.corrcoef(fmri_data[i])
        zFC = fisher_r2z(FC)
        avg_zFC += zFC
        # std = 1 / np.sqrt(fmri_data.shape[1] - 3)

    n_samples = fmri_data.shape[2]
    avg_zFC /= fmri_data.shape[0]
    return avg_zFC, n_samples

def calculate_zFC_parcel(fmri_data: np.ndarray) -> np.ndarray:
    """
    Calculate the functional connectivity (FC) matrix using parcelled fMRI data.
    Args:
        fmri_data (np.ndarray): 2D array where each row is a brain parcel's time series.
    Returns:
        np.ndarray: 2D FC matrix.
    """
    zfc_matrix, _ = calculate_zFC(fmri_data)
    return zfc_matrix

def calculate_zFC_network(fmri_data: np.ndarray, atlas: str = "Yan2023") -> np.ndarray:
    """
    Calculate mean FC values within and between networks.
    Args:
        fmri_data (np.ndarray): 2D array where each row is a brain parcel's time series.
    Returns:
        np.ndarray: Mean FC values matrix between networks.
        np.ndarray: P-value matrix for the mean FC values.
    """
    networks = list_networks(atlas = atlas)
    network_labels = list(networks.keys())

    n_networks = len(network_labels)
    zfc_matrix, n_samples = calculate_zFC(fmri_data)
    mean_zfc_matrix = np.zeros((n_networks, n_networks))
    # p_val_matrix = np.zeros((n_networks, n_networks))

    for i, label_i in enumerate(network_labels):
        regions_i = networks[label_i]
        for j, label_j in enumerate(network_labels[:i+1]):
            regions_j = networks[label_j]
            submatrix = zfc_matrix[np.ix_(regions_i, regions_j)]
            mean_fc = np.mean(submatrix)
            mean_zfc_matrix[i, j] = mean_fc
            mean_zfc_matrix[j, i] = mean_fc  # Symmetric matrix
            # se_z = 1 / np.sqrt(n_samples - 3)
            # z_score = mean_fc / se_z
            # p_val = 2 * (1 - norm.cdf(abs(z_score)))
            # p_val_matrix[i, j] = p_val
            # p_val_matrix[j, i] = p_val  # Symmetric matrix

    return mean_zfc_matrix
#endregion

#region subject zFC calculation
def calculate_subject_bold_zFC(bold_path: str, out_dir: str) -> None:
    """
    Calculate and save the zFC parcel and mean network matrices for a subject's BOLD signals.
    Args:
        bold_path (str): Path to the subject's BOLD signals h5 file.
        out_dir (str): Directory to save the results.
    Returns:
        None
    Saved files:
        - {run_id}_zFC_full.npy : Full parcel zFC matrix.
        - {run_id}_zFC_full_mean.npy : Mean network zFC matrix.
    """
    run_id = extract_run_id(bold_path)
    task_type = "restPA" if "restPA" in bold_path else "restAP"

    bold_signals = load_h5_file(os.path.join(BOLD_DIR, task_type, bold_path))
    zFC = calculate_zFC_parcel(bold_signals[list(bold_signals.keys())[0]])
    np.save(os.path.join(out_dir, "full_parcels", f"{run_id}_zFC_full.npy"), zFC)
    
    zFC_mean = calculate_zFC_network(zFC)
    np.save(os.path.join(out_dir, "networks", f"{run_id}_zFC_full_mean.npy"), zFC_mean)
    return


def calculate_subject_band_zFC(imf_path: str, memd_dir: str, out_dir: str, TR: float = TR) -> None:
    """
    Calculate and save the zFC parcel and mean network matrices for a subject's slow band signals extracted from src.memd IMFs.
    Args:
        imf_path (str): Path to the subject's IMF signals npy file.
        memd_dir (str): Directory where the MEMD processed files are stored.
        out_dir (str): Directory to save the zFC results.
        TR (float): Repetition time of the fMRI data.
    Returns:
        None
    """
    run_id = extract_run_id(imf_path)
    task_type = "restPA" if "restPA" in imf_path else "restAP"

    imfs = np.load(os.path.join(memd_dir, task_type, imf_path))
    band_signals = seperate_slow_band_signals(run_id, imfs, TR=TR)

    ### Calculate zFC for each band with all parcels and save
    zFC_slow5 = calculate_zFC_parcel(band_signals['slow-5'])
    zFC_slow4 = calculate_zFC_parcel(band_signals['slow-4'])
    zFC_slow3 = calculate_zFC_parcel(band_signals['slow-3'])
    np.save(os.path.join(out_dir, "full_parcels", f"{run_id}_zFC_slow5.npy"), zFC_slow5)
    np.save(os.path.join(out_dir, "full_parcels", f"{run_id}_zFC_slow4.npy"), zFC_slow4)
    np.save(os.path.join(out_dir, "full_parcels", f"{run_id}_zFC_slow3.npy"), zFC_slow3)

    ### Calculate zFC for each band with mean networks and save
    zFC_slow5_mean = calculate_zFC_network(zFC_slow5)
    zFC_slow4_mean = calculate_zFC_network(zFC_slow4)
    zFC_slow3_mean = calculate_zFC_network(zFC_slow3)
    np.save(os.path.join(out_dir, "networks", f"{run_id}_zFC_slow5_mean.npy"), zFC_slow5_mean)
    np.save(os.path.join(out_dir, "networks", f"{run_id}_zFC_slow4_mean.npy"), zFC_slow4_mean)
    np.save(os.path.join(out_dir, "networks", f"{run_id}_zFC_slow3_mean.npy"), zFC_slow3_mean)
    return


def calculate_subject_bandpass_bold_zFC(bold_path: str, out_dir: str, TR: float = TR) -> None:
    """
    Calculate and save the zFC parcel and mean network matrices for a subject's band-pass filtered BOLD signals.
    Args:
        bold_path (str): Path to the subject's BOLD signals h5 file.
        out_dir (str): Directory to save the results.
        TR (float): Repetition time of the fMRI data.
    Returns:
        None
    Saved files:
        - {run_id}_zFC_bandpass_full.npy : Full parcel zFC matrix for band-pass filtered signals.
        - {run_id}_zFC_bandpass_full_mean.npy : Mean network zFC matrix for band-pass filtered signals.
    """
    bands = ['slow-5', 'slow-4', 'slow-3']
    run_id = extract_run_id(bold_path)
    task_type = "restPA" if "restPA" in bold_path else "restAP"

    bold_signals = load_h5_file(os.path.join(BOLD_DIR, task_type, bold_path))
    band_signals = extract_slow_band_signals(run_id, bold_signals[list(bold_signals.keys())[0]], TR=TR)
    for band in bands:
        zFC_band = calculate_zFC_parcel(band_signals[band])
        np.save(os.path.join(out_dir, "full_parcels", f"{run_id}_zFC_bandpass_{band}.npy"), zFC_band)
        zFC_band_mean = calculate_zFC_network(zFC_band)
        np.save(os.path.join(out_dir, "networks", f"{run_id}_zFC_bandpass_{band}_mean.npy"), zFC_band_mean)
    return
#endregion


#TODO: Proper handling of all runs. Currently, the code is set up to process only one run type at a time (either restPA or restAP) to avoid mismatches between BOLD and IMF files. This should be improved in the future to allow simultaneous processing of multiple run types without errors. (Also a log message if-else statement that says all can be processed or only specified run types can be processed, which is not currently implemented.) Applies to all functions beneath:
def run_zFC_pipeline(run_types: list, task_type: str, memd_dir: str, out_dir: str, n_parallels: int = N_CPUs, TR: float = TR) -> None:
    """
    Run the zFC calculation pipeline for all subjects in the processed directory.
    Args:
        run_types (list): List of run types to process. Format: [[GROUP1, TASK1, RUN1, CORTICALATLAS1], [GROUP2, TASK2, RUN2, CORTICALATLAS2], ...] (one list per run type if multiple). If all run types are to be processed, pass an empty list [].
        task_type (str): Type of fMRI task (e.g., "restPA").
        memd_dir (str): Directory where the MEMD processed files are stored.
        out_dir (str): Directory to save the processed results.
        n_parallels (int): Number of parallel jobs to run.
        TR (float): Repetition time of the fMRI data.
    Returns:
        None
        Results are saved to data/zFC_matrices/.
    """
    if task_type == "restPA" and "restAP" in run_types or task_type == "restAP" and "restPA" in run_types:
        raise ValueError("Cannot process both 'restPA' and 'restAP' run types simultaneously.")
    
    #region Configure output directories
    save_dir = os.path.join(out_dir, task_type)
    os.makedirs(os.path.join(save_dir, "full_parcels"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "networks"), exist_ok=True)
    BOLD_run_dir = os.path.join(BOLD_DIR, task_type)
    IMF_run_dir = os.path.join(memd_dir, task_type)
    zFC_log_file = os.path.join(LOG_DIR, "zFC_matrices.log")
    #endregion
    #region Initial log message
    msg = f"Starting zFC calculation pipeline with {n_parallels} parallel jobs.\n"
    if not run_types:
        msg += "Processing all run types."
    else:
        msg += f"Processing specified run types: {run_types}."
    log_message(msg, file_path=zFC_log_file)
    #endregion
    #region Filter out BOLD files to process
    bold_files = sorted(path for path in os.listdir(BOLD_run_dir) if path.endswith('.h5'))
    filtered_bold_paths = []

    if not run_types:
        filtered_bold_paths = bold_files
    else:    
        for i in range(len(bold_files)):
            GROUP, TASK, RUN, ATLAS = run_types[0][0], run_types[0][1], run_types[0][2], run_types[0][3]
            if GROUP in bold_files[i] and ATLAS in bold_files[i] and RUN in bold_files[i] and TASK in bold_files[i]:
                filtered_bold_paths.append(bold_files[i])
            else:
                continue
    print(filtered_bold_paths[0])
    print(f"Processing {len(filtered_bold_paths)} BOLD files for full zFC calculation.")
    #endregion

    Parallel(n_jobs=n_parallels)(
        delayed(calculate_subject_bold_zFC)(bold_path, save_dir) for bold_path in tqdm(filtered_bold_paths)
    )

    log_message(
        "Completed calculation of zFC matrices for the bold signals.\nBeginning slow band signal extraction and zFC calculation for each band.",
        file_path=zFC_log_file
    )
    
    #region Filter out IMF files to process
    imf_files = sorted([path for path in os.listdir(IMF_run_dir) if path.endswith('.npy') and not path.endswith('bold_MEMD_imfs.npy')])
    filtered_imf_paths = []
    
    if not run_types:
        filtered_imf_paths = imf_files
    else:
        for i in range(len(imf_files)):
            GROUP, TASK, RUN, ATLAS = run_types[0][0], run_types[0][1], run_types[0][2], run_types[0][3]
            print(GROUP, TASK, RUN, ATLAS)
            print(imf_files[i])
            if GROUP in imf_files[i] and ATLAS in imf_files[i] and RUN in imf_files[i] and TASK in imf_files[i]:
                filtered_imf_paths.append(imf_files[i])
            else:
                continue
    print(f"Processing {len(filtered_imf_paths)} IMF files for slow band zFC calculation.")
    #endregion

    Parallel(n_jobs=n_parallels)(
        delayed(calculate_subject_band_zFC)(imf_path, memd_dir, save_dir, TR) for imf_path in tqdm(filtered_imf_paths)
    )

    log_message(
        "Completed calculation of zFC matrices for the slow band signals.",
        file_path=zFC_log_file
    )

    # imf_ids = {extract_run_id(f) for f in imf_files}
    # bold_ids = {extract_run_id(f) for f in bold_files[:-2]}
    # sym_diff = imf_ids ^ bold_ids
    # if sym_diff:
    #     intersection = imf_ids & bold_ids        
    #     message = (
    #         f"Warning: Mismatch between IMF files and BOLD files.\n"
    #         f"Proceeding with the {len(intersection)} matching files out of {len(imf_ids)} IMF files and {len(bold_ids)} BOLD files.\n"
    #         f"Mismatched IDs: {sym_diff}\n"
    #     )
    #     log_message(message, file_path=zFC_log_file)
    #     imf_files = [f for f in imf_files if extract_run_id(f) in intersection]
    #     bold_files = [f for f in bold_files if extract_run_id(f) in intersection]

    return


def run_bandpass_zFC_pipeline(run_types: list, task_type: str, out_dir: str, n_parallels: int = N_CPUs, TR: float = TR) -> None:
    """
    Run the zFC calculation pipeline using band-pass filtered signals for all subjects in the processed directory.
    Args:
        run_types (list): List of run types to process. Format: [[GROUP1, TASK1, RUN1, CORTICALATLAS1], [GROUP2, TASK2, RUN2, CORTICALATLAS2], ...] (one list per run type if multiple). If all run types are to be processed, pass an empty list [].
        task_type (str): Type of fMRI task (e.g., "restPA").
        bold_dir (str): Directory where the BOLD signals are stored.
        out_dir (str): Directory to save the processed results.
        n_parallels (int): Number of parallel jobs to run.
        TR (float): Repetition time of the fMRI data.
    Returns:
        None
        Results are saved to data/zFC_matrices/*bandpass/.
    """
    if task_type == "restPA" and "restAP" in run_types or task_type == "restAP" and "restPA" in run_types:
        raise ValueError("Cannot process both 'restPA' and 'restAP' run types simultaneously.")
    
    #region Configure output directories
    save_dir = os.path.join(out_dir, f"{task_type}_bandpass")
    os.makedirs(os.path.join(save_dir, "full_parcels"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "networks"), exist_ok=True)
    BOLD_run_dir = os.path.join(BOLD_DIR, task_type)
    zFC_log_file = os.path.join(LOG_DIR, "zFC_matrices_bandpass.log")
    #endregion
    #region Initial log message
    msg = f"Starting band-pass zFC calculation pipeline with {n_parallels} parallel jobs.\n"
    if not run_types:
        msg += "Processing all run types."
    else:
        msg += f"Processing specified run types: {run_types}."
    log_message(msg, file_path=zFC_log_file)
    #endregion
    #region Filter out BOLD files to process
    bold_files = sorted(path for path in os.listdir(BOLD_run_dir) if path.endswith('.h5'))
    filtered_bold_paths = []

    if not run_types:
        filtered_bold_paths = bold_files
    else:
        for i in range(len(bold_files)):
            GROUP, TASK, RUN, ATLAS = run_types[0][0], run_types[0][1], run_types[0][2], run_types[0][3]
            if GROUP in bold_files[i] and ATLAS in bold_files[i] and RUN in bold_files[i] and TASK in bold_files[i]:
                filtered_bold_paths.append(bold_files[i])
            else:
                continue
    print(f"Processing {len(filtered_bold_paths)} BOLD files for band-pass zFC calculation.")
    #endregion

    Parallel(n_jobs=n_parallels)(
        delayed(calculate_subject_bandpass_bold_zFC)(bold_path, save_dir, TR) for bold_path in tqdm(filtered_bold_paths)
    )

    log_message(
        "Completed calculation of zFC matrices for the band-pass filtered signals.",
        file_path=zFC_log_file
    )
    return
    