"""Isolated subprocess CHARMM bonded recovery (full CGENFF, no MLpot param swap).

Spawns a fresh ``np=1`` CHARMM worker that loads PSF/CRD snapshots, runs SD with
full CGENFF parameters, and returns optimized coordinates to the parent MLpot session.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import BondedMmMiniConfig
    from mmml.interfaces.pycharmmInterface.mlpot.setup import MlpotContext

PathLike = str | Path


@dataclass(frozen=True)
class SidecarRecoveryManifest:
    psf: str
    input_crd: str
    output_crd: str
    output_result: str
    use_pbc: bool
    box_side_A: float | None
    nstep_sd: int
    nprint: int
    tolenr: float
    tolgrd: float
    include_vdw: bool
    verbose: bool

    def write(self, path: PathLike) -> Path:
        p = Path(path).expanduser().resolve()
        p.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: PathLike) -> SidecarRecoveryManifest:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def sidecar_worker_script() -> Path:
    return _repo_root() / "scripts" / "charmm_bonded_recovery_worker.py"


def mpirun_launcher_script() -> Path:
    return _repo_root() / "scripts" / "mmml-charmm-mpirun.sh"


def _resolve_topology_psf(ctx: Any, topology_psf: PathLike | None) -> Path:
    if topology_psf is not None:
        path = Path(topology_psf).expanduser().resolve()
        if path.is_file():
            return path
    topo = getattr(ctx, "topology_psf_path", None)
    if topo is not None and Path(topo).is_file():
        return Path(topo).expanduser().resolve()
    import pycharmm.write as write

    fd, name = tempfile.mkstemp(suffix=".psf")
    os.close(fd)
    path = Path(name)
    write.psf_card(str(path))
    return path


def _export_sidecar_snapshot(
    ctx: Any,
    work_dir: Path,
    *,
    topology_psf: PathLike | None,
) -> tuple[Path, Path]:
    import pycharmm.write as write

    from mmml.interfaces.pycharmmInterface.mlpot.bonded_jax_recovery import (
        resolve_recovery_psf_source,
    )

    psf_source = resolve_recovery_psf_source(ctx, topology_psf)
    psf_path = Path(psf_source.path).resolve()
    if not psf_path.is_file():
        psf_path = _resolve_topology_psf(ctx, topology_psf)

    input_crd = work_dir / "input.crd"
    write.coor_card(str(input_crd))
    return psf_path, input_crd


def build_sidecar_manifest(
    ctx: Any,
    config: BondedMmMiniConfig,
    work_dir: Path,
    *,
    topology_psf: PathLike | None = None,
) -> SidecarRecoveryManifest:
    psf_path, input_crd = _export_sidecar_snapshot(
        ctx, work_dir, topology_psf=topology_psf
    )
    box_side = getattr(ctx, "charmm_cubic_box_side_A", None) or getattr(
        ctx, "cubic_box_side_A", None
    )
    return SidecarRecoveryManifest(
        psf=str(psf_path),
        input_crd=str(input_crd),
        output_crd=str(work_dir / "output.crd"),
        output_result=str(work_dir / "result.json"),
        use_pbc=bool(getattr(ctx, "use_pbc", False)),
        box_side_A=float(box_side) if box_side is not None else None,
        nstep_sd=int(config.nstep_sd),
        nprint=max(1, int(config.nprint)),
        tolenr=float(config.tolenr),
        tolgrd=float(config.tolgrd),
        include_vdw=bool(getattr(config, "include_vdw", True)),
        verbose=bool(config.verbose),
    )


def _sidecar_command(manifest_path: Path) -> list[str]:
    launcher = mpirun_launcher_script()
    worker = sidecar_worker_script()
    py = sys.executable
    if launcher.is_file():
        return [str(launcher), py, str(worker), "--manifest", str(manifest_path)]
    return [py, str(worker), "--manifest", str(manifest_path)]


def run_charmm_recovery_sidecar(
    ctx: MlpotContext,
    config: BondedMmMiniConfig,
    *,
    topology_psf: PathLike | None = None,
    work_dir: PathLike | None = None,
) -> float | None:
    """Run bonded recovery in a subprocess with an isolated full-CGENFF CHARMM session."""
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import charmm_grms
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
        read_crd_coordinates,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.setup import (
        get_charmm_positions_array,
        sync_charmm_positions,
    )

    pos_before = np.asarray(get_charmm_positions_array(), dtype=np.float64, copy=True)
    cleanup_dir: tempfile.TemporaryDirectory[str] | None = None
    if work_dir is None:
        cleanup_dir = tempfile.TemporaryDirectory(prefix="mmml_charmm_sidecar_")
        work = Path(cleanup_dir.name)
    else:
        work = Path(work_dir).expanduser().resolve()
        work.mkdir(parents=True, exist_ok=True)

    manifest = build_sidecar_manifest(
        ctx, config, work, topology_psf=topology_psf
    )
    manifest_path = manifest.write(work / "manifest.json")
    cmd = _sidecar_command(manifest_path)
    env = os.environ.copy()
    env.setdefault("MMML_MPI_NP", "1")

    if config.verbose:
        print(
            "bonded recovery: CHARMM sidecar (isolated full CGENFF, MLpot session untouched)",
            flush=True,
        )
        print(f"  work_dir={work}", flush=True)

    proc = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()[-2000:]
        raise RuntimeError(
            f"CHARMM recovery sidecar failed (exit {proc.returncode}): {tail}"
        )

    result_path = Path(manifest.output_result)
    if not result_path.is_file():
        raise RuntimeError(f"CHARMM recovery sidecar missing result: {result_path}")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    grms = float(payload.get("grms", float("nan")))

    out_crd = Path(manifest.output_crd)
    pos_after = read_crd_coordinates(out_crd)
    if pos_after is None:
        raise RuntimeError(f"CHARMM recovery sidecar missing output CRD: {out_crd}")
    if pos_after.shape != pos_before.shape:
        raise RuntimeError(
            f"CHARMM recovery sidecar CRD shape mismatch: "
            f"{pos_after.shape} vs {pos_before.shape}"
        )
    sync_charmm_positions(pos_after)

    if config.verbose:
        from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
            _print_bonded_recovery_geometry_diff,
        )

        _print_bonded_recovery_geometry_diff(
            pos_before,
            ctx,
            topology_psf=topology_psf,
            label="bonded recovery (CHARMM sidecar)",
        )
        print(
            f"bonded recovery sidecar end: GRMS={grms:.4f} kcal/mol/Å "
            f"(parent CHARMM GRMS={charmm_grms():.4f})",
            flush=True,
        )
    return grms if np.isfinite(grms) else None
