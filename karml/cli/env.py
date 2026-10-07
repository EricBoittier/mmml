"""``mmml env`` — resolved paths, bundled checkpoints, and shell export hints."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PRESETS_DIR = _REPO_ROOT / "mmml" / "cli" / "run" / "presets"
_DCM_RESILIENT = _REPO_ROOT / "mmml" / "cli" / "run" / "dcm_liquid_workflow.resilient.example.yaml"


def _repo_root() -> Path:
    try:
        from mmml.interfaces.pycharmmInterface.charmm_paths import mmml_repo_root

        return mmml_repo_root()
    except Exception:
        return _REPO_ROOT


def _resolve_mmml_ckpt() -> tuple[Path | None, str]:
    """Return (path, source) for the checkpoint ``resolve_checkpoint`` would use."""
    explicit = (os.environ.get("MMML_CKPT") or os.environ.get("MMML_CHECKPOINT") or "").strip()
    if explicit:
        p = Path(explicit).expanduser()
        if p.exists():
            return p.resolve(), "MMML_CKPT"
        return None, f"MMML_CKPT (missing: {p})"

    try:
        from mmml.interfaces.pycharmmInterface.mlpot.cli_common import resolve_checkpoint

        return resolve_checkpoint(None), "bundled/default search"
    except FileNotFoundError:
        return None, "not found (set MMML_CKPT or pass --checkpoint)"


def _resolve_spooky_ckpt() -> tuple[Path | None, str]:
    """Return (path, source) for default SpookyNet checkpoint."""
    explicit = (os.environ.get("SPOOKYNET_CKPT") or os.environ.get("SPOOKY_CKPT") or "").strip()
    if explicit:
        p = Path(explicit).expanduser()
        if p.exists():
            return p.resolve(), "SPOOKYNET_CKPT"
        return None, f"SPOOKYNET_CKPT (missing: {p})"

    try:
        from mmml.models.spookynet_calc import resolve_spooky_checkpoint

        return resolve_spooky_checkpoint(), "bundled/default search"
    except Exception:
        return None, "not found (set SPOOKYNET_CKPT or pass checkpoint)"


def _resolve_mbd_ckpt() -> tuple[Path | None, str]:
    """Return (path, source) for default MBD checkpoint."""
    explicit = (os.environ.get("MBD_CKPT") or os.environ.get("MBD_CHECKPOINT") or "").strip()
    if explicit:
        p = Path(explicit).expanduser()
        if p.exists():
            return p.resolve(), "MBD_CKPT"
        return None, f"MBD_CKPT (missing: {p})"

    try:
        from mmml.models.mbd.calculator import resolve_mbd_checkpoint

        return resolve_mbd_checkpoint(), "bundled/default search"
    except Exception:
        return None, "not found (set MBD_CKPT or pass checkpoint)"


def _resolve_multipoles_ckpt() -> tuple[Path | None, str]:
    """Return (path, source) for default Multipoles checkpoint."""
    explicit = (
        os.environ.get("MULTIPOLES_CKPT") or os.environ.get("MULTIPOLES_CHECKPOINT") or ""
    ).strip()
    if explicit:
        p = Path(explicit).expanduser()
        if p.exists():
            return p.resolve(), "MULTIPOLES_CKPT"
        return None, f"MULTIPOLES_CKPT (missing: {p})"

    try:
        from mmml.models.multipoles.electrostatics import resolve_multipoles_checkpoint

        return resolve_multipoles_checkpoint(), "bundled/default search"
    except Exception:
        return None, "not found (set MULTIPOLES_CKPT or pass checkpoint)"


def _charmm_paths() -> dict[str, str | None]:
    try:
        from mmml.interfaces.pycharmmInterface.charmm_paths import bootstrap_charmm_env

        home, lib = bootstrap_charmm_env(repo_root=_repo_root())
        return {
            "CHARMM_HOME": home or None,
            "CHARMM_LIB_DIR": lib or None,
        }
    except Exception as exc:
        return {"CHARMM_HOME": None, "CHARMM_LIB_DIR": None, "error": str(exc)}


def _bundled_checkpoints() -> list[dict[str, Any]]:
    try:
        from mmml.models.physnetjax.defaults import (
            HF_JSON_DIR,
            list_hf_physnet_models,
            resolve_hf_physnet_checkpoint,
        )

        aliases = (
            ("mmml-default", "mmml-default"),
            ("best-forces", "best-forces"),
            ("joint-default", "default"),
        )
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for label, sel in aliases:
            try:
                path = resolve_hf_physnet_checkpoint(sel)
                key = str(path.resolve())
                if key in seen:
                    continue
                seen.add(key)
                out.append({"alias": label, "path": key})
            except (KeyError, OSError):
                continue
        for entry in list_hf_physnet_models():
            file_name = str(entry.get("file") or "")
            if not file_name:
                continue
            path = (HF_JSON_DIR / file_name).resolve()
            key = str(path)
            if key in seen or not path.is_file():
                continue
            seen.add(key)
            out.append(
                {
                    "id": entry.get("id"),
                    "label": entry.get("label"),
                    "path": key,
                }
            )
        return out
    except Exception:
        return []


def collect_env_report() -> dict[str, Any]:
    phys_ckpt, phys_source = _resolve_mmml_ckpt()
    spooky_ckpt, spooky_source = _resolve_spooky_ckpt()
    mbd_ckpt, mbd_source = _resolve_mbd_ckpt()
    mult_ckpt, mult_source = _resolve_multipoles_ckpt()

    charmm = _charmm_paths()
    charmm.pop("error", None)

    return {
        "repo_root": str(_repo_root().resolve()),
        "MMML_CKPT": str(phys_ckpt) if phys_ckpt is not None else None,
        "MMML_CKPT_source": phys_source,
        "MMML_CKPT_set": bool((os.environ.get("MMML_CKPT") or "").strip()),
        "SPOOKYNET_CKPT": str(spooky_ckpt) if spooky_ckpt is not None else None,
        "SPOOKYNET_CKPT_source": spooky_source,
        "SPOOKYNET_CKPT_set": bool((os.environ.get("SPOOKYNET_CKPT") or "").strip()),
        "MBD_CKPT": str(mbd_ckpt) if mbd_ckpt is not None else None,
        "MBD_CKPT_source": mbd_source,
        "MBD_CKPT_set": bool((os.environ.get("MBD_CKPT") or "").strip()),
        "MULTIPOLES_CKPT": str(mult_ckpt) if mult_ckpt is not None else None,
        "MULTIPOLES_CKPT_source": mult_source,
        "MULTIPOLES_CKPT_set": bool((os.environ.get("MULTIPOLES_CKPT") or "").strip()),
        "CHARMM_HOME": charmm.get("CHARMM_HOME"),
        "CHARMM_LIB_DIR": charmm.get("CHARMM_LIB_DIR"),
        "presets_dir": str(_PRESETS_DIR.resolve()) if _PRESETS_DIR.is_dir() else None,
        "dcm_liquid_resilient_yaml": str(_DCM_RESILIENT.resolve())
        if _DCM_RESILIENT.is_file()
        else None,
        "md_system_defaults_doc": str((_repo_root() / "docs" / "md-system-configs.md").resolve()),
        "bundled_checkpoints": _bundled_checkpoints(),
        "model_defaults": {
            "physnet": {
                "name": "PhysNet / Joint MLpot",
                "env_var": "MMML_CKPT",
                "path": str(phys_ckpt) if phys_ckpt is not None else None,
                "source": phys_source,
                "available": phys_ckpt is not None,
            },
            "spookynet": {
                "name": "SpookyNet",
                "env_var": "SPOOKYNET_CKPT",
                "path": str(spooky_ckpt) if spooky_ckpt is not None else None,
                "source": spooky_source,
                "available": spooky_ckpt is not None,
            },
            "mbd": {
                "name": "QCML MBD",
                "env_var": "MBD_CKPT",
                "path": str(mbd_ckpt) if mbd_ckpt is not None else None,
                "source": mbd_source,
                "available": mbd_ckpt is not None,
            },
            "multipoles": {
                "name": "QCML Multipoles",
                "env_var": "MULTIPOLES_CKPT",
                "path": str(mult_ckpt) if mult_ckpt is not None else None,
                "source": mult_source,
                "available": mult_ckpt is not None,
            },
        },
    }


def export_lines(report: dict[str, Any] | None = None) -> list[str]:
    report = report or collect_env_report()
    lines: list[str] = []

    model_vars = (
        ("MMML_CKPT", "MMML_CKPT_set"),
        ("SPOOKYNET_CKPT", "SPOOKYNET_CKPT_set"),
        ("MBD_CKPT", "MBD_CKPT_set"),
        ("MULTIPOLES_CKPT", "MULTIPOLES_CKPT_set"),
    )
    for var, is_set in model_vars:
        val = report.get(var)
        if val and not report.get(is_set):
            lines.append(f"export {var}={val!r}")
        elif val and report.get(is_set):
            lines.append(f"# {var} already set: {val}")

    home = report.get("CHARMM_HOME")
    lib = report.get("CHARMM_LIB_DIR")
    if home and not os.environ.get("CHARMM_HOME"):
        lines.append(f"export CHARMM_HOME={home!r}")
    if lib and not os.environ.get("CHARMM_LIB_DIR"):
        lines.append(f"export CHARMM_LIB_DIR={lib!r}")
    return lines


def render_env_report(report: dict[str, Any]) -> str:
    lines = [
        "MMML Environment & Model Default Parameters Status",
        "==================================================",
        f"repo_root:        {report['repo_root']}",
    ]

    models = report.get("model_defaults") or {}
    if models:
        lines.append("")
        lines.append("Default Model Checkpoints & Parameter Status:")
        lines.append("-" * 50)
        for key in ("physnet", "spookynet", "mbd", "multipoles"):
            info = models.get(key)
            if not info:
                continue
            status = "✓ READY" if info["available"] else "✗ MISSING"
            var = info["env_var"]
            path = info["path"] or f"<unset> ({info['source']})"
            lines.append(f"  [{status}] {info['name']:<18} ({var})")
            lines.append(f"           Path:   {path}")
            lines.append(f"           Source: {info['source']}")

    if report.get("CHARMM_HOME"):
        lines.append("")
        lines.append(f"CHARMM_HOME:      {report['CHARMM_HOME']}")
    if report.get("CHARMM_LIB_DIR"):
        lines.append(f"CHARMM_LIB_DIR:    {report['CHARMM_LIB_DIR']}")
    if report.get("presets_dir"):
        lines.append(f"md-system presets: {report['presets_dir']}")
    if report.get("dcm_liquid_resilient_yaml"):
        lines.append(f"DCM liquid YAML:   {report['dcm_liquid_resilient_yaml']}")

    bundled = report.get("bundled_checkpoints") or []
    if bundled:
        lines.append("")
        lines.append("Bundled PhysNet checkpoints (hf_json/manifest.json):")
        for entry in bundled[:12]:
            alias = entry.get("alias") or entry.get("id") or "?"
            label = entry.get("label")
            suffix = f"  ({label})" if label else ""
            lines.append(f"  {alias}{suffix}")
            lines.append(f"    {entry['path']}")
        if len(bundled) > 12:
            lines.append(f"  ... and {len(bundled) - 12} more")

    exports = export_lines(report)
    if exports:
        lines.append("")
        lines.append("Suggested shell exports (only for unset vars):")
        lines.extend(f"  {line}" for line in exports)
    lines.append("")
    lines.append("Tip: eval \"$(mmml env --export)\"  or  mmml env --export >> ~/.bashrc")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mmml env",
        description=(
            "Show resolved MMML environment paths and status of default model "
            "parameters (PhysNet, SpookyNet, MBD, Multipoles)."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON.",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Print export lines only (for eval \"$(mmml env --export)\").",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = collect_env_report()
    if args.export:
        for line in export_lines(report):
            print(line)
        return 0 if report.get("MMML_CKPT") else 1
    if args.json:
        print(json.dumps(report, indent=2))
        return 0
    print(render_env_report(report))
    models = report.get("model_defaults") or {}
    all_ready = all(m.get("available") for m in models.values()) if models else True
    return 0 if all_ready else 1


if __name__ == "__main__":
    sys.exit(main())
