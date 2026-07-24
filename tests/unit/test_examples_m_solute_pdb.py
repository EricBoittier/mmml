"""Unit tests for examples/m NPZ → CGenFF solute PDB export."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
GEOM = REPO / "examples" / "m" / "_geometry.py"
NPZ = REPO / "examples" / "m" / "nh3_ch3cl_filtered.npz"


def _load_geometry():
    spec = importlib.util.spec_from_file_location("examples_m_geometry", GEOM)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(not NPZ.is_file(), reason="examples/m dataset NPZ not present")
def test_write_solute_pdb_amm1_ch3cl(tmp_path: Path) -> None:
    geom = _load_geometry()
    out = tmp_path / "solute.pdb"
    path = geom.write_solute_pdb(out, NPZ, index=0)
    text = path.read_text(encoding="utf-8")
    atom_lines = [ln for ln in text.splitlines() if ln.startswith("ATOM")]
    assert len(atom_lines) == 9
    assert sum("AMM1" in ln for ln in atom_lines) == 4
    assert sum("CH3CL" in ln for ln in atom_lines) == 5
    assert "TER" in text.splitlines()
    # Atom names expected by CGenFF / make-box
    assert any("N1" in ln for ln in atom_lines)
    assert any("CL1" in ln for ln in atom_lines)
    # Full CH3CL (5 chars) must survive whitespace parse — not truncated to CH3C
    from mmml.interfaces.pycharmmInterface.mlpot.setup import (
        _residue_sequence_from_pdb,
    )

    assert _residue_sequence_from_pdb(path) == ["AMM1", "CH3CL"]
    z, r = geom.load_dimer_frame(NPZ, index=0)
    assert len(z) == 9
    assert r.shape == (9, 3)
    assert np.all(np.isfinite(r))
