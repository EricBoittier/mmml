"""PSF layout checks when continuing MD from cross-backend handoff."""

import numpy as np
import pytest

from mmml.cli.run.md_handoff import (
    _live_psf_matches_handoff,
    _validate_handoff_psf_layout,
    ensure_psf_for_handoff_cluster,
)


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


def test_live_psf_matches_handoff(monkeypatch):
    charges = np.array([0.1, -0.1, 0.0], dtype=float)

    class _PSF:
        @staticmethod
        def get_charges():
            return charges

    monkeypatch.setattr("mmml.cli.run.md_handoff.psf", _PSF, raising=False)
    import mmml.cli.run.md_handoff as handoff_mod

    monkeypatch.setattr(handoff_mod, "psf", _PSF, raising=False)

    def _import_psf():
        return _PSF

    monkeypatch.setattr(
        handoff_mod,
        "_live_psf_matches_handoff",
        lambda n_atoms: (
            n_atoms == charges.size and np.all(np.isfinite(charges))
        ),
    )
    assert handoff_mod._live_psf_matches_handoff(3)
    assert not handoff_mod._live_psf_matches_handoff(4)


def test_ensure_psf_reuses_live_psf_without_rebuild(monkeypatch):
    z = np.array([6, 1, 1, 1, 17], dtype=int)
    built = {"called": False}

    def _fail_build(*_args, **_kwargs):
        built["called"] = True
        raise AssertionError("should not rebuild PSF")

    monkeypatch.setattr(
        "mmml.cli.run.md_handoff._live_psf_matches_handoff",
        lambda n_atoms: n_atoms == len(z),
    )
    monkeypatch.setattr(
        "mmml.cli.run.md_handoff._build_cluster_psf_topology_only",
        _fail_build,
    )

    ensure_psf_for_handoff_cluster(
        composition=[("DCM", 1)],
        atomic_numbers=z,
        atoms_per_list=[5],
        residue_labels=["DCM"],
        quiet=True,
    )
    assert built["called"] is False
