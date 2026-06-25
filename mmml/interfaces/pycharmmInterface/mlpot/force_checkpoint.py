"""Optional NPZ dumps of CHARMM total forces during MLpot dynamics."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

_active: Optional["ForceCheckpointWriter"] = None


@dataclass
class ForceCheckpointConfig:
    enabled: bool = False
    interval: int = 1
    output_path: Path | None = None
    n_monomers: int = 1
    atoms_per_monomer: int | None = None


@dataclass
class ForceCheckpointWriter:
    """Accumulate force snapshots and flush to NPZ."""

    config: ForceCheckpointConfig
    steps: list[int] = field(default_factory=list)
    forces: list[np.ndarray] = field(default_factory=list)
    positions: list[np.ndarray] = field(default_factory=list)
    ml_forces: list[np.ndarray] = field(default_factory=list)
    _last_saved_step: int = -1

    def should_save(self, step: int) -> bool:
        if not self.config.enabled:
            return False
        interval = max(1, int(self.config.interval))
        return int(step) % interval == 0

    def record(
        self,
        step: int,
        *,
        total_forces: np.ndarray,
        positions: np.ndarray | None = None,
        ml_forces: np.ndarray | None = None,
    ) -> None:
        if not self.should_save(step):
            return
        if int(step) == self._last_saved_step:
            return
        self._last_saved_step = int(step)
        f = np.asarray(total_forces, dtype=np.float64).reshape(-1, 3)
        self.steps.append(int(step))
        self.forces.append(f)
        if positions is not None:
            self.positions.append(np.asarray(positions, dtype=np.float64).reshape(-1, 3))
        if ml_forces is not None:
            self.ml_forces.append(np.asarray(ml_forces, dtype=np.float64).reshape(-1, 3))

    def flush(self) -> Path | None:
        if not self.config.enabled or not self.steps:
            return None
        path = Path(self.config.output_path or "forces.npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "step": np.asarray(self.steps, dtype=np.int64),
            "forces": np.stack(self.forces, axis=0),
        }
        if len(self.positions) == len(self.steps):
            payload["positions"] = np.stack(self.positions, axis=0)
        if len(self.ml_forces) == len(self.steps):
            payload["ml_forces"] = np.stack(self.ml_forces, axis=0)

        n_atoms = int(payload["forces"].shape[1])
        fmag = np.linalg.norm(payload["forces"], axis=2)
        payload["force_mag"] = fmag

        n_mol = int(self.config.n_monomers)
        if self.config.atoms_per_monomer is not None and n_atoms == n_mol * int(
            self.config.atoms_per_monomer
        ):
            per = int(self.config.atoms_per_monomer)
            monomer_id = np.repeat(np.arange(n_mol, dtype=np.int32), per)
            payload["monomer_id"] = monomer_id
            max_per_mol = np.zeros((len(self.steps), n_mol), dtype=np.float64)
            f_com = np.zeros((len(self.steps), n_mol, 3), dtype=np.float64)
            for fi in range(len(self.steps)):
                for mi in range(n_mol):
                    sl = slice(mi * per, (mi + 1) * per)
                    block = payload["forces"][fi, sl, :]
                    max_per_mol[fi, mi] = float(np.linalg.norm(block, axis=1).max())
                    f_com[fi, mi, :] = block.sum(axis=0)
            payload["max_force_per_monomer"] = max_per_mol
            payload["force_sum_per_monomer"] = f_com

        np.savez(path, **payload)
        return path


def configure_force_checkpoint(config: ForceCheckpointConfig | None) -> None:
    global _active
    if config is None or not config.enabled:
        _active = None
        return
    _active = ForceCheckpointWriter(config=config)


def get_force_checkpoint_writer() -> ForceCheckpointWriter | None:
    return _active


def maybe_record_forces(
    step: int,
    *,
    ml_forces: np.ndarray | None = None,
) -> None:
    """Read CHARMM forces/positions and append to the active writer."""
    writer = _active
    if writer is None or not writer.should_save(step):
        return
    try:
        from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
            charmm_positions_angstrom,
            charmm_total_forces_kcalmol_A,
        )

        total = charmm_total_forces_kcalmol_A()
        pos = charmm_positions_angstrom()
    except Exception:
        return
    writer.record(step, total_forces=total, positions=pos, ml_forces=ml_forces)


def flush_force_checkpoint() -> Path | None:
    global _active
    if _active is None:
        return None
    path = _active.flush()
    _active = None
    return path


def resolve_force_checkpoint_config(args: Any, *, n_monomers: int) -> ForceCheckpointConfig:
    enabled = bool(getattr(args, "save_forces_npz", False))
    if not enabled:
        return ForceCheckpointConfig(enabled=False)
    out = getattr(args, "output_dir", None)
    path = Path(out) / "forces.npz" if out is not None else Path("forces.npz")
    per = getattr(args, "atoms_per_monomer", None)
    if per is None and hasattr(args, "composition"):
        # uniform DCM-like clusters
        try:
            import re

            m = re.match(r"^[A-Za-z0-9]+:(\d+)$", str(getattr(args, "composition", "") or ""))
            if m:
                n = int(m.group(1))
                z_len = getattr(args, "_cluster_n_atoms", None)
                if z_len is not None and n > 0:
                    per = int(z_len) // n
        except Exception:
            per = None
    return ForceCheckpointConfig(
        enabled=True,
        interval=int(getattr(args, "forces_npz_interval", 1) or 1),
        output_path=path,
        n_monomers=int(n_monomers),
        atoms_per_monomer=int(per) if per is not None else None,
    )
