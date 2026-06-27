"""On-disk progress folders for density prep ladder and geometry cleanup."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

PREP_LADDER_SUBDIR = "prep_ladder"
CLEANUP_SUBDIR = "cleanup"

RecoveryCategory = Literal["prep_ladder", "cleanup"]


def resolve_output_dir(args: argparse.Namespace | Any | None) -> Path | None:
    out = getattr(args, "output_dir", None) if args is not None else None
    if out is None:
        return None
    return Path(out).expanduser().resolve()


def resolve_prep_ladder_dir(args: argparse.Namespace | Any | None) -> Path | None:
    """``<output-dir>/prep_ladder`` (or ``--prep-ladder-dir``)."""
    root = resolve_output_dir(args)
    if root is None:
        return None
    if bool(getattr(args, "no_recovery_artifacts", False)):
        return None
    name = str(getattr(args, "prep_ladder_dir", None) or PREP_LADDER_SUBDIR)
    return (root / name).resolve()


def resolve_cleanup_dir(args: argparse.Namespace | Any | None) -> Path | None:
    """``<output-dir>/cleanup`` (or ``--cleanup-dir``)."""
    root = resolve_output_dir(args)
    if root is None:
        return None
    if bool(getattr(args, "no_recovery_artifacts", False)):
        return None
    name = str(getattr(args, "cleanup_dir", None) or CLEANUP_SUBDIR)
    return (root / name).resolve()


def slugify_step_label(label: str) -> str:
    slug = re.sub(r"[^\w]+", "_", str(label).lower()).strip("_")
    return slug[:56] or "step"


@dataclass
class RecoveryProgressStore:
    """Numbered PDB/CRD checkpoints + ``journal.json`` under a recovery subfolder."""

    root: Path
    category: RecoveryCategory
    title: str
    quiet: bool = False
    _index: int = field(default=0, init=False)
    _steps: list[dict[str, Any]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    @classmethod
    def for_prep_ladder(
        cls,
        args: argparse.Namespace | Any | None,
        *,
        title: str = "Density prep ladder",
        quiet: bool = False,
    ) -> RecoveryProgressStore | None:
        directory = resolve_prep_ladder_dir(args)
        if directory is None:
            return None
        return cls(directory, "prep_ladder", title, quiet=quiet)

    @classmethod
    def for_cleanup(
        cls,
        args: argparse.Namespace | Any | None,
        *,
        title: str = "Geometry cleanup",
        quiet: bool = False,
    ) -> RecoveryProgressStore | None:
        directory = resolve_cleanup_dir(args)
        if directory is None:
            return None
        return cls(directory, "cleanup", title, quiet=quiet)

    def record_step(
        self,
        label: str,
        *,
        grms_kcalmol_A: float | None = None,
        box_side_A: float | None = None,
        positions: Any | None = None,
        note: str = "",
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Path]:
        self._index += 1
        slug = slugify_step_label(label)
        stem = f"{self._index:03d}_{slug}"
        files = _save_geometry_checkpoint(
            self.root,
            stem,
            title=str(label),
            positions=positions,
        )
        entry: dict[str, Any] = {
            "index": self._index,
            "label": str(label),
            "slug": slug,
            "stem": stem,
            "files": {k: str(v) for k, v in sorted(files.items())},
        }
        if grms_kcalmol_A is not None:
            entry["hybrid_grms_kcalmol_A"] = float(grms_kcalmol_A)
        if box_side_A is not None:
            entry["box_side_A"] = float(box_side_A)
        if note:
            entry["note"] = str(note)
        if extra:
            entry["meta"] = dict(extra)
        self._steps.append(entry)
        self._write_journal()
        if not self.quiet:
            from mmml.interfaces.pycharmmInterface.mpi_rank_io import rank0_print

            rank0_print(
                f"{self.category}: checkpoint {self._index:03d} → {self.root.name}/{stem} "
                f"({label})",
                flush=True,
            )
        return files

    def finish(self, summary: dict[str, Any] | None = None) -> None:
        from mmml.interfaces.pycharmmInterface.mpi_rank_io import is_mpi_rank_zero

        if not is_mpi_rank_zero():
            return
        payload = {
            "category": self.category,
            "title": self.title,
            "root": str(self.root),
            "steps": list(self._steps),
            "summary": summary or {},
        }
        (self.root / "summary.json").write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
        self._write_journal()

    def _write_journal(self) -> None:
        from mmml.interfaces.pycharmmInterface.mpi_rank_io import is_mpi_rank_zero

        if not is_mpi_rank_zero():
            return
        (self.root / "journal.json").write_text(
            json.dumps(
                {
                    "category": self.category,
                    "title": self.title,
                    "steps": list(self._steps),
                },
                indent=2,
            ),
            encoding="utf-8",
        )


def _save_geometry_checkpoint(
    directory: Path,
    stem: str,
    *,
    title: str,
    positions: Any | None = None,
) -> dict[str, Path]:
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import save_minimization_results

    directory.mkdir(parents=True, exist_ok=True)
    pdb_path = directory / f"{stem}.pdb"
    crd_path = directory / f"{stem}.crd"
    written = save_minimization_results(
        pdb_path=pdb_path,
        crd_path=crd_path,
        positions=positions,
        title=title,
        show_energy=False,
    )
    if "crd" in written:
        shutil.copy2(written["crd"], directory / "latest.crd")
    if "pdb" in written:
        shutil.copy2(written["pdb"], directory / "latest.pdb")
    return written


def add_recovery_artifact_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Recovery artifact folders")
    group.add_argument(
        "--prep-ladder-dir",
        type=str,
        default=PREP_LADDER_SUBDIR,
        help=(
            "Subfolder under --output-dir for density / pre-MLpot ladder checkpoints "
            f"(default: {PREP_LADDER_SUBDIR})."
        ),
    )
    group.add_argument(
        "--cleanup-dir",
        type=str,
        default=CLEANUP_SUBDIR,
        help=(
            "Subfolder under --output-dir for geometry cleanup / overlap rescue "
            f"checkpoints (default: {CLEANUP_SUBDIR})."
        ),
    )
    group.add_argument(
        "--no-recovery-artifacts",
        action="store_true",
        help="Do not write prep_ladder/ or cleanup/ checkpoint folders.",
    )
