"""Numbered minimization / geometry-recovery coordinate snapshots."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

PathLike = str | Path

MinimizeKind = Literal[
    "packmol",
    "MM",
    "MMML",
    "bonded_MM",
    "overlap_rescue",
    "intra_rescue",
]


@dataclass(frozen=True)
class MinimizeSnapshotSpec:
    """One saved geometry checkpoint in a staged MLpot workflow."""

    seq: int
    slug: str
    label: str
    kind: MinimizeKind

    def stem(self, tag: str) -> str:
        return f"{self.seq:02d}_{self.slug}_{tag}"


# Staged workflow checkpoints (fixed sequence).
PACKMOL_CLUSTER = MinimizeSnapshotSpec(
    0, "packmol_cluster", "Packmol sphere placement (pre-minimize)", "packmol"
)
CHARMM_MM_PRE = MinimizeSnapshotSpec(
    1, "charmm_mm", "CHARMM CGENFF SD/ABNR before MLpot (MM only)", "MM"
)
MLPOT_MMML = MinimizeSnapshotSpec(
    2, "mlpot_mmml", "MLpot PhysNet steepest descent (USER / MMML)", "MMML"
)
BONDED_MM_AFTER_MINI = MinimizeSnapshotSpec(
    3,
    "bonded_mm_after_mini",
    "Bonded-only CHARMM SD after MLpot mini (strain recovery)",
    "bonded_MM",
)
BONDED_MM_AFTER_HEAT = MinimizeSnapshotSpec(
    4,
    "bonded_mm_after_heat",
    "Bonded-only CHARMM SD after heat (strain recovery)",
    "bonded_MM",
)


def snapshot_file_paths(out_dir: PathLike, spec: MinimizeSnapshotSpec, tag: str) -> dict[str, Path]:
    """Standard artifact extensions for a minimize snapshot stem."""
    out = Path(out_dir).expanduser().resolve()
    stem = spec.stem(tag)
    return {
        "stem": out / stem,
        "pdb": out / f"{stem}.pdb",
        "crd": out / f"{stem}.crd",
        "psf": out / f"{stem}.psf",
        "dcd": out / f"{stem}.dcd",
        "xyz": out / f"{stem}.xyz",
        "energy_json": out / f"{stem}_energy.json",
    }


def legacy_mlpot_mini_paths(out_dir: PathLike, tag: str) -> dict[str, Path]:
    """Historical filenames (``mini_full_mlpot_*``) kept for scripts and docs."""
    out = Path(out_dir).expanduser().resolve()
    stem = f"mini_full_mlpot_{tag}"
    return {
        "mini_crd": out / f"{stem}.crd",
        "mini_psf": out / f"{stem}.psf",
        "mini_pdb": out / f"{stem}.pdb",
        "mini_dcd": out / f"{stem}.dcd",
        "mini_xyz": out / f"{stem}.xyz",
        "mini_energy_json": out / f"{stem}_energy.json",
    }


def legacy_charmm_mm_dcd(out_dir: PathLike, tag: str) -> Path:
    return Path(out_dir).expanduser().resolve() / f"mini_charmm_mm_{tag}.dcd"


def _slugify_context(context: str) -> str:
    s = re.sub(r"[^\w]+", "_", context.strip().lower())
    return s.strip("_")[:72] or "rescue"


def rescue_snapshot_spec(context: str, *, seq: int) -> MinimizeSnapshotSpec:
    """Dynamic snapshot for overlap / intra rescue during dynamics."""
    slug = f"rescue_{_slugify_context(context)}"
    return MinimizeSnapshotSpec(
        seq,
        slug,
        f"Geometry rescue: {context}",
        "intra_rescue",
    )


class MinimizeArtifactRegistry:
    """Append numbered minimize snapshots and write ``minimize_snapshots_{tag}.json``."""

    def __init__(self, out_dir: PathLike, tag: str) -> None:
        self.out_dir = Path(out_dir).expanduser().resolve()
        self.tag = tag
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._entries: list[dict[str, Any]] = []
        self._next_dynamic_seq = 10

    @property
    def manifest_path(self) -> Path:
        return self.out_dir / f"minimize_snapshots_{self.tag}.json"

    def allocate_rescue_spec(self, context: str) -> MinimizeSnapshotSpec:
        spec = rescue_snapshot_spec(context, seq=self._next_dynamic_seq)
        self._next_dynamic_seq += 1
        return spec

    def record(
        self,
        spec: MinimizeSnapshotSpec,
        written: dict[str, Path],
        *,
        grms_kcalmol_A: float | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        files = {k: str(Path(v).resolve()) for k, v in sorted(written.items())}
        entry: dict[str, Any] = {
            "seq": spec.seq,
            "slug": spec.slug,
            "stem": spec.stem(self.tag),
            "kind": spec.kind,
            "label": spec.label,
            "files": files,
        }
        if grms_kcalmol_A is not None:
            entry["grms_kcalmol_A"] = float(grms_kcalmol_A)
        if extra:
            entry["meta"] = extra
        self._entries.append(entry)
        self.flush()
        if not any(e.get("stem") == entry["stem"] for e in self._entries[:-1]):
            print(
                f"minimize snapshot [{spec.seq:02d}] {spec.kind}: {spec.label} → "
                f"{spec.stem(self.tag)}",
                flush=True,
            )

    def flush(self) -> None:
        payload = {"tag": self.tag, "snapshots": self._entries}
        self.manifest_path.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )


def mirror_legacy_mlpot_files(
    written: dict[str, Path],
    legacy: dict[str, Path],
) -> dict[str, Path]:
    """Copy numbered MMML outputs to legacy ``mini_full_mlpot_*`` names."""
    mapping = {
        "pdb": "mini_pdb",
        "crd": "mini_crd",
        "psf": "mini_psf",
        "xyz": "mini_xyz",
        "energy_json": "mini_energy_json",
    }
    out: dict[str, Path] = {}
    for src_key, dst_key in mapping.items():
        if src_key not in written or dst_key not in legacy:
            continue
        src = Path(written[src_key])
        dst = Path(legacy[dst_key])
        if not src.is_file():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        out[dst_key] = dst.resolve()
    return out


def save_snapshot_from_charmm(
    registry: MinimizeArtifactRegistry | None,
    spec: MinimizeSnapshotSpec,
    *,
    out_dir: PathLike,
    tag: str,
    title: str | None = None,
    show_energy: bool = False,
    grms_kcalmol_A: float | None = None,
    legacy_mlpot: dict[str, Path] | None = None,
    include_psf: bool = True,
) -> dict[str, Path]:
    """Write PDB/CRD (+ optional PSF/energy) from current CHARMM coordinates."""
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import save_minimization_results

    paths = snapshot_file_paths(out_dir, spec, tag)
    written = save_minimization_results(
        pdb_path=paths["pdb"],
        crd_path=paths["crd"],
        psf_path=paths["psf"] if include_psf else None,
        energy_json_path=paths["energy_json"],
        title=title or spec.label,
        show_energy=show_energy,
    )
    if legacy_mlpot is not None:
        written.update(mirror_legacy_mlpot_files(written, legacy_mlpot))
    if registry is not None:
        registry.record(spec, written, grms_kcalmol_A=grms_kcalmol_A)
    return written
