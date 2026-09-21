"""CLI + driver for DiffTRe liquid-observable fitting (toy Hamiltonian)."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import jax
import numpy as np
import pytest

from mmml.cli.misc.fit_liquid import build_parser, main
from mmml.fit.driver import dry_run, jsonable, theta_sidecar_payload, toy_hybrid_handles
from mmml.fit.frames import FrameSet, density_g_cm3, load_frames
from mmml.fit.lj_theta import LjTypeMap, init_theta
from mmml.models.mm_lj_scales import load_mm_lj_scales_sidecar

APM = 2
Z = np.array([6, 8, 6, 8], dtype=np.int32)
MASS = 2 * (12.011 + 15.999)


@pytest.fixture(autouse=True)
def _x64():
    with jax.enable_x64(True):
        yield


def _write_jaxmd_h5(path: Path, n_frames: int = 4, T_K: float = 250.0) -> None:
    rng = np.random.default_rng(0)
    box = np.tile([10.0, 11.0, 12.0], (n_frames, 1)) * rng.uniform(0.95, 1.05, (n_frames, 1))
    pos = rng.uniform(0.0, 9.0, (n_frames, 4, 3))
    pos[:, 1] = pos[:, 0] + [1.2, 0.0, 0.0]
    pos[:, 3] = pos[:, 2] + [0.0, 1.1, 0.0]
    rho = density_g_cm3(MASS, box)
    with h5py.File(path, "w") as f:
        f.create_dataset("positions", data=pos.astype(np.float32))
        f.create_dataset("potential_energy", data=np.linspace(-1.0, -1.2, n_frames))
        f.create_dataset("density_g_cm3", data=rho)
        f.attrs["atomic_numbers"] = Z
        f.attrs["temperature_target"] = T_K


def test_parser_prog_and_stages():
    p = build_parser()
    assert p.prog == "mmml fit-liquid"
    dry = p.parse_args(
        ["dry-run", "--frames", "x.h5", "--atoms-per-molecule", "10", "--molecule", "ACO", "--toy"]
    )
    assert dry.stage == "dry-run" and dry.n_steps == 2 and dry.toy
    fit = p.parse_args(
        ["fit", "--cache", "c.npz", "--molecule", "ACO", "--toy", "--out-json", "t.json", "--n-steps", "8"]
    )
    assert fit.n_steps == 8 and fit.sidecar is None


def test_help_does_not_import_jax():
    import ast
    from pathlib import Path as P

    src = (P(__file__).resolve().parents[2] / "mmml" / "cli" / "misc" / "fit_liquid.py").read_text()
    tree = ast.parse(src)
    imports = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.extend(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module.split(".")[0])
    assert "jax" not in imports
    assert "mmml" not in imports


def test_bundled_reference_is_used_when_machine_file_is_absent(monkeypatch):
    from mmml.fit.targets import BUNDLED_REFERENCE, DEFAULT_REFERENCE_JSON, load_reference

    monkeypatch.delenv("MMML_EXP_REFERENCE_JSON", raising=False)
    if DEFAULT_REFERENCE_JSON.is_file():
        pytest.skip("machine-local experimental JSON is present")
    ref = load_reference()
    assert "ACO" in ref and ref["ACO"]["T_boil_K"]["value"] == BUNDLED_REFERENCE["ACO"]["T_boil_K"]["value"]


def test_dry_run_on_jaxmd_npt_h5(tmp_path):
    h5 = tmp_path / "run_npt.h5"
    _write_jaxmd_h5(h5)
    frames = load_frames(h5, APM)
    assert frames.n_frames == 4 and frames.temperature_K == 250.0
    out = tmp_path / "cache.npz"
    result = dry_run(frames, "ACO", n_steps=2, cache_path=out, overwrite=True)
    payload = result.as_json()
    assert result.n_frames == 4
    assert np.isfinite(result.loss)
    assert payload["check"][0]["ess_fraction"] == pytest.approx(1.0, abs=1e-5)
    assert payload["history"][0]["ess_fraction"] > 0.2
    assert Path(out).is_file()
    sidecar = theta_sidecar_payload(result.theta, result.type_names, result.lam)
    path = tmp_path / "hybrid_mm.json"
    path.write_text(json.dumps(sidecar))
    loaded = load_mm_lj_scales_sidecar(path)
    assert loaded is not None
    assert list(loaded["cgenff_type_names"]) == list(result.type_names)


def test_cli_dry_run_and_fit_from_cache(tmp_path):
    h5 = tmp_path / "run_npt.h5"
    _write_jaxmd_h5(h5)
    cache = tmp_path / "cache.npz"
    report = tmp_path / "dry.json"
    rc = main(
        [
            "dry-run",
            "--frames",
            str(h5),
            "--atoms-per-molecule",
            str(APM),
            "--molecule",
            "ACO",
            "--toy",
            "--n-steps",
            "2",
            "--cache-out",
            str(cache),
            "--out-json",
            str(report),
        ]
    )
    assert rc == 0
    data = json.loads(report.read_text())
    assert data["n_frames"] == 4
    assert np.isfinite(data["loss"])

    out = tmp_path / "theta.json"
    sidecar = tmp_path / "scales.json"
    rc = main(
        [
            "fit",
            "--cache",
            str(cache),
            "--molecule",
            "ACO",
            "--toy",
            "--n-steps",
            "1",
            "--out-json",
            str(out),
            "--sidecar",
            str(sidecar),
        ]
    )
    assert rc == 0
    assert out.is_file() and sidecar.is_file()
    loaded = load_mm_lj_scales_sidecar(sidecar)
    assert loaded is not None
    assert "ml_dimer_scale" in json.loads(sidecar.read_text())


def test_cli_refuses_production_path_without_toy():
    with pytest.raises(SystemExit, match="--toy"):
        main(["dry-run", "--frames", "x.h5", "--atoms-per-molecule", "10", "--molecule", "ACO"])


def test_toy_handles_match_init_theta_types():
    rng = np.random.default_rng(1)
    box = np.tile([12.0, 12.0, 12.0], (2, 1))
    pos = rng.uniform(0.0, 8.0, (2, 4, 3))
    frames = FrameSet(
        positions=pos,
        box=box,
        atomic_numbers=Z,
        masses=np.array([12.011, 15.999, 12.011, 15.999]),
        temperature_K=250.0,
        n_molecules=2,
        density_g_cm3=density_g_cm3(MASS, box),
        u_ref=np.full(2, np.nan),
    )
    handles = toy_hybrid_handles(frames)
    type_map = LjTypeMap.from_atc(handles.update_fn.at_codes, handles.update_fn.atc_names)
    theta = init_theta(type_map)
    assert theta["log_eps"].shape == (2,)
    assert jsonable({"a": np.array(np.nan), "b": 1}) == {"a": None, "b": 1}
