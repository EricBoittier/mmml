#!/usr/bin/env python3
"""
Example: How to spoof/simulate CLI arguments for plot_ir_raman.py

This shows how to call the plotting functions programmatically without argparse.
"""

from pathlib import Path
from types import SimpleNamespace

from plot_ir_raman import main_with_args


def spoof_args(
    input_file: str | Path,
    output_dir: str | Path | None = None,
    freq_range: tuple[float, float] = (0, 4500),
    no_peaks: bool = False,
    peak_threshold: float = 0.05,
    dpi: int = 300,
) -> SimpleNamespace:
    """
    Create a mock argparse.Namespace object with CLI arguments.
    
    Parameters
    ----------
    input_file : str or Path
        Input NPZ file path
    output_dir : str or Path, optional
        Output directory (default: same as input file)
    freq_range : tuple
        (min_freq, max_freq) frequency range
    no_peaks : bool
        Don't mark peaks
    peak_threshold : float
        Minimum relative intensity for peaks
    dpi : int
        Figure resolution
    
    Returns
    -------
    SimpleNamespace
        Mock args object compatible with argparse.Namespace
    """
    input_path = Path(input_file)
    if output_dir is None:
        output_dir = input_path.parent
    else:
        output_dir = Path(output_dir)
    
    return SimpleNamespace(
        input=input_path,
        output_dir=output_dir,
        freq_range=list(freq_range),
        no_peaks=no_peaks,
        peak_threshold=peak_threshold,
        dpi=dpi,
    )


# Example usage:
if __name__ == "__main__":
    # Spoof CLI arguments programmatically
    args = spoof_args(
        input_file="ir_raman.npz",  # Your NPZ file
        output_dir="./plots",        # Output directory
        freq_range=(0, 4500),       # Frequency range
        no_peaks=False,              # Show peaks
        peak_threshold=0.05,         # Peak detection threshold
        dpi=300,                     # Resolution
    )
    
    # Call the main function with spoofed args
    main_with_args(args)
    
    # Or create args inline:
    # args = SimpleNamespace(
    #     input=Path("ir_raman.npz"),
    #     output_dir=Path("./plots"),
    #     freq_range=[0, 4500],
    #     no_peaks=False,
    #     peak_threshold=0.05,
    #     dpi=300,
    # )
    # main_with_args(args)

