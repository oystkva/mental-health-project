import os, sys
import numpy as np
from scipy.signal import hilbert, butter, filtfilt
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import LOG_DIR, TR
from src.utils import log_message

def assign_imfs_to_slow_bands(imfs: np.ndarray, TR: float = TR) -> tuple:
    """
    Assign IMFs to slow frequency bands based on their center frequencies.
    Args:
        imfs (np.ndarray): Array of shape (n_imfs, n_channels, n_timepoints).
        TR (float): Repetition time in seconds.
    Returns:
        tuple: A tuple containing:
            - dict: Mapping of band names to lists of IMF indices.
            - np.ndarray: Array of center frequencies for each IMF.
    """
    n_imfs, C, T = imfs.shape
    
    bands = {
        "slow-5": (0.01, 0.027),
        "slow-4": (0.027, 0.073),
        "slow-3": (0.073, 0.198),
    }

    band_names = list(bands.keys())
    band_to_imfs = {b: [] for b in band_names}
    imfs_centre_freqs = np.zeros(n_imfs, dtype=float)

    dt = TR

    for k in range(n_imfs):
        x = imfs[k]

        analytic = hilbert(x, axis=-1)
        amp = np.abs(analytic)
        phase = np.unwrap(np.angle(analytic), axis=-1)

        dphi_dt = np.gradient(phase, dt, axis=-1)
        inst_freq = dphi_dt / (2.0 * np.pi)

        inst_freq = np.maximum(inst_freq, 0.0)

        energy = amp ** 2
        total_energy = energy.sum()

        fc = (energy *inst_freq).sum() / total_energy
        imfs_centre_freqs[k] = fc

        for band_name, (f_low, f_high) in bands.items():
            if f_low <= fc < f_high:
                band_to_imfs[band_name].append(k)                
                break

    return band_to_imfs, imfs_centre_freqs

def seperate_slow_band_signals(run_id: str, imfs: np.ndarray, TR: float = TR) -> dict:
    """
    Separate slow band signals based on assigned IMFs.
    Args:
        run (str): Run identifier.
        imfs (np.ndarray): Array of shape (n_imfs, n_channels, n_timepoints).
        TR (float): Repetition time in seconds.
    Returns:
        dict: Mapping of band names to arrays of signals.
    """
    band_to_imfs, imfs_centre_freqs = assign_imfs_to_slow_bands(imfs, TR)

    msg = f"Run {run_id}: IMF center frequencies: {[round(float(f), 5) for f in imfs_centre_freqs]}\n"
    msg += f"Slow-3 IMFs: {band_to_imfs['slow-3']} with center freqs {[round(float(imfs_centre_freqs[i]), 4) for i in band_to_imfs['slow-3']]}\n"
    msg += f"Slow-4 IMFs: {band_to_imfs['slow-4']} with center freqs {[round(float(imfs_centre_freqs[i]), 4) for i in band_to_imfs['slow-4']]}\n"
    msg += f"Slow-5 IMFs: {band_to_imfs['slow-5']} with center freqs {[round(float(imfs_centre_freqs[i]), 4) for i in band_to_imfs['slow-5']]}\n"
    log_message(msg, file_path=os.path.join(LOG_DIR, "slow_band_extraction.log"))

    band_signals = {}
    for band_name, imf_indices in band_to_imfs.items():
        if imf_indices:
            band_signals[band_name] = imfs[imf_indices]
        else:
            band_signals[band_name] = None
    return band_signals


def bandpass_filter(signal: np.ndarray, low_freq: float, high_freq: float, TR: float = TR) -> np.ndarray:
    """
    Apply a band-pass filter to the signal.
    Args:
        signal (np.ndarray): Input signal of shape (n_channels, n_timepoints).
        low_freq (float): Low cutoff frequency in Hz.
        high_freq (float): High cutoff frequency in Hz.
        TR (float): Repetition time in seconds.
    Returns:
        np.ndarray: Band-pass filtered signal of shape (n_channels, n_timepoints).
    """

    fs = 1.0 / TR
    nyquist = fs / 2.0
    low = low_freq / nyquist
    high = high_freq / nyquist

    b, a = butter(N=4, Wn=[low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal, axis=-1)
    
    return filtered_signal


def extract_slow_band_signals(run_id: str, signals: np.ndarray, TR: float = TR) -> dict:
    """
    Extract slow band signals by applying band-pass filters to the original signal.
    Args:
        run_id (str): Run identifier.
        signals (np.ndarray): Array of shape (n_channels, n_timepoints).
        TR (float): Repetition time in seconds.
    Returns:
        dict: Mapping of band names to arrays of signals.
    """
    bands = {
        "slow5": (0.01, 0.027),
        "slow4": (0.027, 0.073),
        "slow3": (0.073, 0.198),
    }

    msg = f"Run {run_id}: Extracting slow band signals using band-pass filters with TR={TR}s\n"
    log_message(msg, file_path=os.path.join(LOG_DIR, "slow_band_extraction.log"))

    # init dictionary to hold band signals
    band_signals = {band_name: np.zeros_like(signals) for band_name in bands.keys()}
    for band_name, (f_low, f_high) in bands.items():
        for ch in range(signals.shape[0]):
            band_signals[band_name][ch] = bandpass_filter(signals[ch], f_low, f_high, TR)
            
    return band_signals