"""Vibrational spectra (IR, Raman, VCD) from MD trajectories and EF models."""

from mmml.spectra.spectra_md import (
    autocorrelation,
    compute_magnetic_dipoles,
    compute_polarizability_batched,
    correlation_to_spectrum,
    cross_correlation,
    extract_properties,
    extract_properties_hdf5,
    get_args,
    load_hdf5_trajectory,
    main,
    polarizability_autocorrelation,
    raman_to_spectrum,
)

__all__ = [
    "autocorrelation",
    "compute_magnetic_dipoles",
    "compute_polarizability_batched",
    "correlation_to_spectrum",
    "cross_correlation",
    "extract_properties",
    "extract_properties_hdf5",
    "get_args",
    "load_hdf5_trajectory",
    "main",
    "polarizability_autocorrelation",
    "raman_to_spectrum",
]
