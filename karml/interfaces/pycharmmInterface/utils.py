from mmml.interfaces.pycharmmInterface.import_pycharmm import *
import os
import ase
import numpy as np

def get_Z_from_psf():
    import pycharmm.psf as psf

    masses = psf.get_amass()
    Z = []
    for m in masses:
        mdif = (ase.data.atomic_masses_common - m)**2
        Z .append( np.argmin(mdif) )
    return Z

def set_up_directories(base: str | os.PathLike[str] | None = None) -> None:
    """Create legacy make-res / make-box layout dirs (pdb, psf, xyz, res, dcd).

    Only call this from paths that still write relative ``pdb/…``, ``psf/…``,
    etc. (``mmml make-res``, ``mmml make-box``, ``generate_coordinates``).
    ``md-system`` / MLpot use flat files under ``--output-dir`` and must not
    create these in CWD via :func:`ensure_charmm_session_ready`.
    """
    root = os.fspath(base) if base is not None else "."
    for name in ("pdb", "res", "dcd", "psf", "xyz"):
        os.makedirs(os.path.join(root, name), exist_ok=True)