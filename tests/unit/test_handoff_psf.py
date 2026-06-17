"""PSF layout checks when continuing MD from cross-backend handoff."""

import numpy as np
import pytest

from mmml.cli.run.md_handoff import _validate_handoff_psf_layout


def test_validate_handoff_psf_layout_accepts_matching_cluster():
    z = np.array([6, 1, 1, 1, 17] * 2, dtype=int)
    _validate_handoff_psf_layout(
        psf_atomic_numbers=z.copy(),
        psf_atoms_per_list=[5, 5],
        psf_residue_labels=["DCM", "DCM"],
        atomic_numbers=z,
        atoms_per_list=[5, 5],
        residue_labels=["DCM", "DCM"],
    )


def test_validate_handoff_psf_layout_rejects_z_mismatch():
    z = np.array([6, 1, 1, 1, 17, 6, 1, 1, 1, 17], dtype=int)
    with pytest.raises(RuntimeError, match="Z mismatch"):
        _validate_handoff_psf_layout(
            psf_atomic_numbers=np.ones(len(z), dtype=int),
            psf_atoms_per_list=[5, 5],
            psf_residue_labels=["DCM", "DCM"],
            atomic_numbers=z,
            atoms_per_list=[5, 5],
            residue_labels=["DCM", "DCM"],
        )
