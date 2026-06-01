"""Molecular Mechanics and Machine Learned Force Fields"""

# Add imports here
from .mmml import *

# Legacy ``mmml.pycharmmInterface`` imports are handled by ``mmml/pycharmmInterface/``
# (lazy redirects to ``mmml.interfaces.pycharmmInterface``). Do not register the
# interfaces package as ``mmml.pycharmmInterface`` in ``sys.modules`` here: that
# breaks ``alias_mod is canonical_mod`` for submodules loaded via the legacy path.

import sys

if "mmml.models.physnetjax" not in sys.modules:
    from mmml.models import physnetjax
    sys.modules["mmml.models.physnetjax"] = physnetjax

# Compatibility: mmml.dcmnet -> mmml.models.dcmnet
if "mmml.dcmnet" not in sys.modules:
    from mmml.models import dcmnet
    sys.modules["mmml.dcmnet"] = dcmnet

# Handle version import gracefully
try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.0+dev"
