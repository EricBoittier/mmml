"""Distributed charge multipole network utilities."""

# The package exposes submodules such as :mod:`mmml.dcmnet.dcmnet`, but we avoid
# importing them eagerly to keep optional dependencies lazy.  Users can import
# the submodules explicitly (``from mmml.dcmnet import dcmnet``) without the
# heavy side effects that ``from . import *`` introduced.

__all__: list[str] = []
