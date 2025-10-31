"""Molecular Mechanics and Machine Learned Force Fields"""

# Add imports here
from .mmml import *

# Handle version import gracefully
try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.0+dev"
