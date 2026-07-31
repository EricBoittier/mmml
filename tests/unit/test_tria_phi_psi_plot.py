"""Unit tests for trialanine gas/solvent Ramachandran plotting."""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _load_plot_mod():
    path = REPO / "scripts" / "plot_tria_phi_psi_gas_solvent.py"
    spec = importlib.util.spec_from_file_location("plot_tria_phi_psi_gas_solvent", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_demo_gas_solvent_figure(tmp_path: Path) -> None:
    mod = _load_plot_mod()
    g, s = mod._write_demo_csvs(tmp_path / "csv")
    out = tmp_path / "fig.png"
    mod.plot_gas_solvent(g, s, out, vmax=40.0)
    assert out.is_file()
    assert out.stat().st_size > 1000
