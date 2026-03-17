"""Molecular Mechanics and Machine Learned Force Fields"""

# Add imports here
from .mmml import *

# Compatibility: mmml.pycharmmInterface -> mmml.interfaces.pycharmmInterface
import sys
if "mmml.pycharmmInterface" not in sys.modules:
    from mmml.interfaces import pycharmmInterface
    sys.modules["mmml.pycharmmInterface"] = pycharmmInterface

# Compatibility: mmml.physnetjax -> mmml.models.physnetjax
if "mmml.physnetjax" not in sys.modules:
    from mmml.models import physnetjax
    sys.modules["mmml.physnetjax"] = physnetjax

# Compatibility: mmml.dcmnet -> mmml.models.dcmnet
if "mmml.dcmnet" not in sys.modules:
    from mmml.models import dcmnet
    sys.modules["mmml.dcmnet"] = dcmnet

# Handle version import gracefully
try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.0+dev"
