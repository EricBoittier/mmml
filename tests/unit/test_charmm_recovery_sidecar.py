"""Unit tests for CHARMM bonded recovery sidecar."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mmml.interfaces.pycharmmInterface.mlpot.charmm_recovery_sidecar import (
    SidecarRecoveryManifest,
    build_sidecar_manifest,
    run_charmm_recovery_sidecar,
)
from mmml.interfaces.pycharmmInterface.mlpot.dynamics import BondedMmMiniConfig


def test_sidecar_manifest_round_trip(tmp_path: Path) -> None:
    manifest = SidecarRecoveryManifest(
        psf=str(tmp_path / "box.psf"),
        input_crd=str(tmp_path / "in.crd"),
        output_crd=str(tmp_path / "out.crd"),
        output_result=str(tmp_path / "result.json"),
        use_pbc=True,
        box_side_A=28.0,
        nstep_sd=100,
        nprint=10,
        tolenr=1e-3,
        tolgrd=1e-3,
        include_vdw=False,
        verbose=False,
    )
    path = manifest.write(tmp_path / "manifest.json")
    loaded = SidecarRecoveryManifest.load(path)
    assert loaded.psf == manifest.psf
    assert loaded.box_side_A == pytest.approx(28.0)
    assert loaded.nstep_sd == 100


def test_build_sidecar_manifest_exports_psf_and_crd(tmp_path: Path) -> None:
    import sys

    ctx = MagicMock()
    ctx.use_pbc = True
    ctx.charmm_cubic_box_side_A = 30.0
    ctx.topology_psf_path = None
    cfg = BondedMmMiniConfig(nstep_sd=50, verbose=False, include_vdw=False)
    psf = tmp_path / "live.psf"
    psf.write_text("psf", encoding="utf-8")

    fake_write = MagicMock()
    with patch.dict(
        sys.modules,
        {
            "pycharmm": MagicMock(write=fake_write),
            "pycharmm.write": fake_write,
        },
    ):
        with patch(
            "mmml.interfaces.pycharmmInterface.mlpot.bonded_jax_recovery.resolve_recovery_psf_source",
            return_value=MagicMock(path=psf, temporary=False, cleanup=lambda: None),
        ):
            manifest = build_sidecar_manifest(ctx, cfg, tmp_path)
    assert Path(manifest.psf) == psf.resolve()
    assert Path(manifest.input_crd) == (tmp_path / "input.crd").resolve()
    fake_write.coor_card.assert_called_once()


def test_run_charmm_recovery_sidecar_applies_output_crd(tmp_path: Path) -> None:
    ctx = MagicMock()
    cfg = BondedMmMiniConfig(nstep_sd=10, verbose=False, include_vdw=False)
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    out_pos = pos + 0.1

    def _fake_run(cmd, env, capture_output, text):
        work = Path(cmd[-1]).parent if cmd[-1].endswith(".json") else tmp_path
        for part in cmd:
            if part.endswith("manifest.json"):
                work = Path(part).parent
                break
        manifest = SidecarRecoveryManifest.load(work / "manifest.json")
        Path(manifest.output_result).write_text(
            json.dumps({"grms": 2.5}),
            encoding="utf-8",
        )
        lines = ["2 EXT", "1 0.1 0.2 0.3", "2 1.1 0.2 0.3"]
        Path(manifest.output_crd).write_text("\n".join(lines) + "\n", encoding="utf-8")
        return MagicMock(returncode=0, stdout="", stderr="")

    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.get_charmm_positions_array",
        side_effect=[pos, out_pos],
    ):
        with patch(
            "mmml.interfaces.pycharmmInterface.mlpot.charmm_recovery_sidecar.build_sidecar_manifest",
            wraps=lambda ctx, config, work_dir, topology_psf=None: SidecarRecoveryManifest(
                psf=str(tmp_path / "box.psf"),
                input_crd=str(work_dir / "input.crd"),
                output_crd=str(work_dir / "output.crd"),
                output_result=str(work_dir / "result.json"),
                use_pbc=False,
                box_side_A=None,
                nstep_sd=10,
                nprint=1,
                tolenr=1e-3,
                tolgrd=1e-3,
                include_vdw=False,
                verbose=False,
            ),
        ):
            with patch(
                "mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation.read_crd_coordinates",
                return_value=out_pos,
            ):
                with patch(
                    "mmml.interfaces.pycharmmInterface.mlpot.setup.sync_charmm_positions",
                ) as sync_pos:
                    with patch(
                        "mmml.interfaces.pycharmmInterface.mlpot.charmm_recovery_sidecar.subprocess.run",
                        side_effect=_fake_run,
                    ):
                        grms = run_charmm_recovery_sidecar(
                            ctx,
                            cfg,
                            work_dir=tmp_path,
                        )
    assert grms == pytest.approx(2.5)
    sync_pos.assert_called_once()
