#!/usr/bin/env python3
"""Backward-compatible entry point; implementation lives in karml.spectra.spectra_md."""

from karml.spectra.spectra_md import *  # noqa: F403
from karml.spectra.spectra_md import main

if __name__ == "__main__":
    main()
