#!/usr/bin/env python3
"""Backward-compatible entry point; implementation lives in mmml.spectra.spectra_md."""

from mmml.spectra.spectra_md import *  # noqa: F403
from mmml.spectra.spectra_md import main

if __name__ == "__main__":
    main()
