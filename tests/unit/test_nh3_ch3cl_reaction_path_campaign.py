"""Unit tests for workflows/nh3_ch3cl_reaction_path matrix expansion."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

WORKFLOW = Path(__file__).resolve().parents[2] / "workflows" / "nh3_ch3cl_reaction_path"
SCRIPTS = WORKFLOW / "scripts"


def _load() -> ModuleType:
    name = "nh3_ch3cl_reaction_path_campaign_lib"
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    path = SCRIPTS / "campaign_lib.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


cl = _load()


def test_smoke_config_uses_model_ext_and_small_matrix() -> None:
    cfg = cl.load_config(WORKFLOW / "config.smoke.yaml")
    assert cfg["checkpoint"].endswith("model_ext.json")
    assert cl.seeds(cfg) == [0]
    assert cl.temperatures(cfg) == [300]
    assert cl.solvents(cfg) == ["tip3"]
    targets = cl.expand_targets(cfg, cfg["output_root"])
    assert any(t.endswith("neb/status.json") for t in targets)
    assert any("/T300/seed0/" in t for t in targets)
    assert any(t.endswith("dmc/react/seed0/status.json") for t in targets)
    assert not any("adumb_" in t for t in targets)  # disabled in smoke


def test_full_config_expands_seed_temperature_solvent() -> None:
    cfg = cl.load_config(WORKFLOW / "config.yaml")
    assert cfg["checkpoint"] == "examples/m/model_ext.json"
    assert cl.seeds(cfg) == [0, 1, 2]
    assert cl.temperatures(cfg) == [250, 300, 350]
    targets = cl.expand_targets(cfg, cfg["output_root"])
    # umbrella_gas: 2 variants × 3 T × 3 seeds
    gas = [t for t in targets if "/umbrella_gas/" in t and t.endswith("/status.json") and "/mbar/" not in t]
    assert len(gas) == 2 * 3 * 3
    # umbrella_sol: 3 solvents × 2 × 3 × 3
    sol = [t for t in targets if "/umbrella_sol/" in t and t.endswith("/status.json") and "/mbar/" not in t]
    assert len(sol) == 3 * 2 * 3 * 3
    # dmc: 2 basins × 3 seeds (no T)
    dmc = [t for t in targets if "/dmc/" in t]
    assert len(dmc) == 2 * 3
    assert all("/T" not in t for t in dmc)


def test_prod_config_uses_model_ext() -> None:
    cfg = cl.load_config(WORKFLOW / "config.prod.yaml")
    assert cfg["checkpoint"] == "examples/m/model_ext.json"
    assert cl.checkpoint_path(cfg) == "examples/m/model_ext.json"
    assert set(cl.umbrella_variants(cfg)) == {"dt1", "dt05", "dt025"}
    for name in cl.umbrella_variants(cfg):
        v = cfg["umbrella"]["variants"][name]
        assert float(v["xi_min"]) >= 2.0
        assert float(v["k_ev_A2"]) <= 5.0


def test_require_umbrella_products(tmp_path: Path) -> None:
    import importlib.util

    name = "nh3_run_job_require_products"
    path = SCRIPTS / "run_job.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)

    out = tmp_path / "umb"
    out.mkdir()
    try:
        mod._require_umbrella_products(out)
        raise AssertionError("expected FileNotFoundError")
    except FileNotFoundError as exc:
        assert "umbrella_snapshots.npz" in str(exc)
    (out / "umbrella_snapshots.npz").write_bytes(b"x")
    (out / "umbrella_summary.json").write_text("{}", encoding="utf-8")
    mod._require_umbrella_products(out)


def test_job_mbar_cli_has_no_output_dir_flag() -> None:
    """Regression: umbrella-mbar rejects --output-dir (broke mbar_gas on studix)."""
    from mmml.cli.misc.umbrella_mbar import build_parser

    parser = build_parser()
    with __import__("pytest").raises(SystemExit):
        parser.parse_args(["--run-dir", "/tmp", "--output-dir", "/tmp/mbar"])
