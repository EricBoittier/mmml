"""Molecular Mechanics and Machine Learned Force Fields"""

# Add imports here
from .karml import *

import sys

if "karml.models.physnetjax" not in sys.modules:
    from karml.models import physnetjax
    sys.modules["karml.models.physnetjax"] = physnetjax

# Compatibility: karml.dcmnet -> karml.models.dcmnet
if "karml.dcmnet" not in sys.modules:
    from karml.models import dcmnet
    sys.modules["karml.dcmnet"] = dcmnet

# Handle version import gracefully
try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.0+dev"
