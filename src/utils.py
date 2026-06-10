import os, sys
from datetime import datetime
import numpy as np
from pathlib import Path
import random
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import PROJECT_ROOT

def log_message(message: str, file_path: str) -> None:
    """
    Log a message with a timestamp.
    Args:
        message (str): Message to log.
    """
    if os.path.exists(file_path) == False:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(file_path, 'a') as f:
        f.write(f"[{now}] {message}\n")


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


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility. 
    Should be updated if CUDA is used.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)