"""Orchestrate multi-backend cross-check runs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from mmml.analysis.npz_comparison import (
    align_npz_arrays,
    compare_npz_arrays,
    plot_comparison,
    write_comparison_report,
)
from mmml.data.units import normalize_energy_unit, normalize_force_unit
from mmml.interfaces.qc_backends.factory import backend_from_dict, build_backend
from mmml.interfaces.qc_backends.npz_output import (
    BACKEND_NATIVE_UNITS,
    infer_target_units,
    normalize_backend_npz,
    write_backend_metadata,
)
from mmml.interfaces.qc_backends.protocol import BackendSpec
from mmml.interfaces.qc_backends.structures import load_structures


@dataclass
class CrossCheckConfig:
    """Configuration for a cross-check run."""

    structures: Path
    output_dir: Path = Path("cross_check_out")
    reference_npz: Path | None = None
    reference_backend: str | None = None
    backends: list[BackendSpec] = field(default_factory=list)
    max_frames: int | None = None
    stride: int = 1
    charge: int = 0
    spin: int = 0
    multiplicity: int = 1
    properties: frozenset[str] = frozenset({"energy", "forces"})
    save_backend_npz: bool = True
    no_plots: bool = False

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], base_dir: Path | None = None) -> CrossCheckConfig:
        base = base_dir or Path.cwd()

        def _resolve(p: Any) -> Path | None:
            if p is None:
                return None
            path = Path(p)
            if not path.is_absolute():
                path = (base / path).resolve()
            return path

        backends_raw = data.get("backends") or []
        backends = [backend_from_dict(entry) for entry in backends_raw]

        ref = data.get("reference")
        reference_backend = None if ref is None else str(ref)
        reference_npz = _resolve(data.get("reference_npz"))

        mult = data.get("multiplicity")
        if mult is None and "spin" in data:
            mult = int(data["spin"]) + 1

        props = data.get("properties") or ["energy", "forces"]
        return cls(
            structures=_resolve(data["structures"]),  # type: ignore[arg-type]
            output_dir=_resolve(data.get("output") or data.get("output_dir") or "cross_check_out"),  # type: ignore[arg-type]
            reference_npz=reference_npz,
            reference_backend=reference_backend,
            backends=backends,
            max_frames=data.get("max_frames"),
            stride=int(data.get("stride", 1)),
            charge=int(data.get("charge", 0)),
            spin=int(data.get("spin", 0)),
            multiplicity=int(mult if mult is not None else 1),
            properties=frozenset(str(p) for p in props),
            save_backend_npz=bool(data.get("save_backend_npz", True)),
            no_plots=bool(data.get("no_plots", False)),
        )

    @classmethod
    def from_yaml(cls, path: Path) -> CrossCheckConfig:
        data = yaml.safe_load(path.read_text())
        if not isinstance(data, dict):
            raise ValueError(f"Cross-check config must be a mapping: {path}")
        return cls.from_mapping(data, base_dir=path.parent)


class CrossCheckRunner:
    """Run multiple QC backends and compare against a reference."""

    def __init__(self, config: CrossCheckConfig) -> None:
        self.config = config

    def _inject_global_options(self, spec: BackendSpec) -> BackendSpec:
        opts = dict(spec.options)
        opts.setdefault("charge", self.config.charge)
        opts.setdefault("spin", self.config.spin)
        opts.setdefault("multiplicity", self.config.multiplicity)
        return BackendSpec(name=spec.name, options=opts)

    def _load_reference(
        self,
        frames: list,
        structures_arrays: dict[str, np.ndarray],
    ) -> tuple[dict[str, np.ndarray], str]:
        cfg = self.config
        if cfg.reference_npz is not None:
            data = np.load(cfg.reference_npz, allow_pickle=True)
            ref = {k: np.asarray(data[k]) for k in data.files}
            label = cfg.reference_npz.stem
            return ref, label

        ref_name = (cfg.reference_backend or "pyscf").lower()
        spec = BackendSpec(name=ref_name, options={"charge": cfg.charge, "spin": cfg.spin})
        spec = self._inject_global_options(spec)
        backend = build_backend(spec)
        print(f"Computing reference with {backend.method_label}...")
        ref = backend.evaluate_batch(frames, properties=cfg.properties)
        ref = write_backend_metadata(
            ref,
            backend=backend.name,
            method_label=backend.method_label,
            energy_unit=backend.energy_unit,
            force_unit=backend.force_unit,
        )
        return ref, backend.method_label

    def run(self) -> dict[str, Any]:
        cfg = self.config
        cfg.output_dir.mkdir(parents=True, exist_ok=True)

        frames, structures_arrays = load_structures(
            cfg.structures,
            max_frames=cfg.max_frames,
            stride=cfg.stride,
        )
        if not frames:
            raise ValueError(f"No structures loaded from {cfg.structures}")

        reference, ref_label = self._load_reference(frames, structures_arrays)
        target_e_unit, target_f_unit = infer_target_units(reference)
        energy_unit_label = normalize_energy_unit(target_e_unit)
        force_unit_label = {
            "ev": "eV/Å",
            "hartree": "Ha/bohr",
            "kcal_mol": "kcal/mol/Å",
        }.get(energy_unit_label, target_f_unit)

        ref_backend = str(reference.get("_backend", np.array("reference")))
        if isinstance(ref_backend, np.ndarray):
            ref_backend = str(ref_backend.reshape(()))
        ref_normalized = normalize_backend_npz(
            reference,
            backend=ref_backend if ref_backend in BACKEND_NATIVE_UNITS else "pyscf",
            target_energy_unit=target_e_unit,
            target_force_unit=target_f_unit,
            source_energy_unit=target_e_unit if cfg.reference_npz is not None else None,
            source_force_unit=target_f_unit if cfg.reference_npz is not None else None,
        )

        summary: dict[str, Any] = {
            "structures": str(cfg.structures.resolve()),
            "reference": ref_label,
            "n_frames": len(frames),
            "units": {"energy": energy_unit_label, "forces": force_unit_label},
            "backends": {},
            "method_warnings": [],
        }

        if cfg.save_backend_npz:
            ref_path = cfg.output_dir / "reference.npz"
            np.savez_compressed(ref_path, **ref_normalized)

        for spec in cfg.backends:
            spec = self._inject_global_options(spec)
            if spec.name.lower() == (cfg.reference_backend or "").lower() and cfg.reference_npz is None:
                continue

            backend = build_backend(spec)
            print(f"Evaluating backend {backend.method_label}...")
            try:
                pred = backend.evaluate_batch(frames, properties=cfg.properties)
            except Exception as exc:
                summary["backends"][spec.label] = {"error": str(exc)}
                print(f"  backend {spec.label} failed: {exc}")
                continue

            pred = write_backend_metadata(
                pred,
                backend=backend.name,
                method_label=backend.method_label,
                energy_unit=backend.energy_unit,
                force_unit=backend.force_unit,
            )
            pred_norm = normalize_backend_npz(
                pred,
                backend=backend.name,
                target_energy_unit=target_e_unit,
                target_force_unit=target_f_unit,
            )

            if backend.method_label != ref_label:
                summary["method_warnings"].append(
                    f"{backend.method_label} compared to reference {ref_label}"
                )

            aligned = align_npz_arrays(ref_normalized, pred_norm)
            metrics = compare_npz_arrays(
                aligned,
                energy_unit_label=energy_unit_label,
                force_unit_label=force_unit_label,
            )

            backend_dir = cfg.output_dir / spec.name
            backend_dir.mkdir(parents=True, exist_ok=True)
            plot_paths: list[str] = []
            if not cfg.no_plots:
                try:
                    plot_paths = plot_comparison(
                        aligned,
                        metrics,
                        backend_dir,
                        title_prefix=f"{backend.method_label} vs {ref_label} | ",
                        energy_unit=energy_unit_label,
                        force_unit=force_unit_label,
                    )
                except ImportError:
                    pass

            report_path = write_comparison_report(
                metrics,
                backend_dir,
                reference=ref_label,
                predictions=backend.method_label,
                plot_paths=plot_paths,
            )

            if cfg.save_backend_npz:
                np.savez_compressed(backend_dir / "predictions.npz", **pred_norm)

            summary["backends"][spec.label] = {
                "method": backend.method_label,
                "metrics": metrics,
                "report": str(report_path),
            }

        summary_path = cfg.output_dir / "cross_check_summary.json"

        def _json_default(obj: Any) -> Any:
            if isinstance(obj, (np.floating, np.integer)):
                return obj.item()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(type(obj))

        summary_path.write_text(json.dumps(summary, indent=2, default=_json_default))
        print(f"\nWrote summary to {summary_path}")
        return summary
