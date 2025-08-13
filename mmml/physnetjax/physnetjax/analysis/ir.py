"""
Statistics and Miscellaneous utilities for scientific calculations.
"""

from typing import List, Tuple

import ase.units as units
import matplotlib.pyplot as plt
import numpy as np
import pint
from scipy.signal import find_peaks
from statsmodels.tsa.stattools import acovf

ureg = pint.UnitRegistry()

# Constants
C = (units._c * ureg("m/s")).to("cm/s").magnitude
HBAR = (5.29e-12 * ureg("cm^-1 s")).magnitude
KB = 3.1668114e-6  # Boltzmann constant in atomic units (Hartree/K)
BETA = 1.0 / (KB * 300)  # atomic units


def autocorrelation_ft(
    series: np.ndarray, timestep: float, verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the autocorrelation function of a time series using the Fourier transform.

    Args:
        series (np.ndarray): Input time series data.
        timestep (float): Time step between data points.
        verbose (bool, optional): If True, print additional information. Defaults to False.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Frequency range and corresponding spectra.
    """
    jiffy = 0.01 / units._c * 1e12  # Time for light to travel 1cm in fs
    nframes = series.shape[0]
    nfreq = int(nframes / 2) + 1
    freq = np.arange(nfreq) / float(nframes) / timestep * jiffy

    if verbose:
        print(f"Nframes: {nframes}")

    # Dipole-Dipole autocorrelation function
    acv = sum(acovf(series[:, i], fft=True) for i in range(3))

    if verbose:
        print(f"ACV shape: {acv.shape}")

    acv *= np.blackman(nframes)
    spectra = np.abs(np.fft.rfftn(acv))

    return freq, spectra


def intensity_correction(
    freq: np.ndarray, spectra: np.ndarray, volume: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply intensity correction to the spectra.

    Args:
        freq (np.ndarray): Frequency range.
        spectra (np.ndarray): Input spectra.
        volume (float): Volume for normalization.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Corrected frequency range and spectra.
    """
    twopiomega = 2 * np.pi * freq
    exp_corr = 1 - np.exp(-BETA * HBAR * freq)
    three_h_c_v = 3 * HBAR * C * volume

    spectra = spectra * (twopiomega * exp_corr) / three_h_c_v

    # Scale the spectra so the integral is 1 between 0 and 4500 cm^-1
    spectra /= np.trapz(spectra, freq)

    return freq, spectra


def read_dat_file(filename: str) -> np.ndarray:
    """
    Read a .dat file and return the data as a numpy array.

    Args:
        filename (str): Path to the .dat file.

    Returns:
        np.ndarray: Data from the file.
    """
    return np.loadtxt(filename)


def rolling_avg(
    freq: np.ndarray, spectra: np.ndarray, window: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the rolling average of a data set.

    Args:
        freq (np.ndarray): Frequency range.
        spectra (np.ndarray): Input spectra.
        window (int, optional): Size of the rolling window. Defaults to 10.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Smoothed frequency range and spectra.
    """
    freq = freq[window:]
    spectra = np.convolve(spectra, np.ones(window), "valid") / window
    return freq, spectra[1:]


def assign_peaks(
    spectra: np.ndarray, n_peaks: int = 10, height: float = 0.1
) -> List[int]:
    """
    Find peaks in the spectra.

    Args:
        spectra (np.ndarray): Input spectra.
        n_peaks (int, optional): Number of peaks to find. Defaults to 10.
        height (float, optional): Minimum height of peaks. Defaults to 0.1.

    Returns:
        List[int]: Indices of detected peaks.
    """
    distance = len(spectra) // n_peaks
    peaks, _ = find_peaks(spectra, threshold=None, height=height, distance=distance)
    return peaks.tolist()
