#!/usr/bin/env python3
"""Audit QCML Orbax shards for schema, physical consistency, and scale."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

from mmml.data.orbax_shards import read_manifest
from mmml.models.multipoles import irrep_blocks_to_traceless


BOHR_TO_ANGSTROM = 0.529177210903
HARTREE_TO_KCAL_MOL = 627.5094740631


def _quantiles(values: list[np.ndarray]) -> dict[str, float]:
    array = np.concatenate([np.asarray(value).reshape(-1) for value in values])
    return {
        "min": float(np.min(array)),
        "q01": float(np.quantile(array, 0.01)),
        "median": float(np.median(array)),
        "q99": float(np.quantile(array, 0.99)),
        "max": float(np.max(array)),
    }


def _shard_paths(cache: Path, max_shards: int | None) -> list[Path]:
    if (cache / "manifest.json").exists():
        paths = [cache / item["path"] for item in read_manifest(cache)["shards"]]
    else:
        paths = [cache]
    return paths[:max_shards]


def _pair_distances(positions: np.ndarray) -> np.ndarray:
    if len(positions) < 2:
        return np.empty(0, dtype=np.float32)
    dst, src = np.triu_indices(len(positions), k=1)
    return np.linalg.norm(positions[src] - positions[dst], axis=-1)


def audit_cache(
    cache: Path,
    *,
    kind: str = "auto",
    max_shards: int | None = None,
    samples_per_shard: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    report: dict[str, Any] = {
        "cache": str(cache),
        "units": {
            "positions": "bohr (QCML schema)",
            "mbd_energy": "hartree (QCML schema)",
            "mbd_polarizabilities": "bohr^3 (QCML schema)",
            "mbd_forces": "assumed hartree/bohr; verify externally",
            "mbd_c6": "assumed hartree*bohr^6; verify externally",
        },
    }
    distances = []
    atom_counts = []
    multipoles = []
    charges = []
    mbd_energy = []
    c6_values = []
    alpha_values = []
    net_force_residuals = []
    torque_residuals = []
    padding_violations = 0
    nonfinite_values = 0
    sampled = 0

    for shard_path in _shard_paths(cache, max_shards):
        shard = {
            key: np.asarray(value)
            for key, value in ocp.PyTreeCheckpointer().restore(shard_path).items()
        }
        if kind == "auto":
            kind = "mbd" if "E_mbd" in shard else "multipoles"
        size = len(shard["R"])
        selected = rng.choice(size, size=min(size, samples_per_shard), replace=False)
        for index in selected:
            mask = shard["atom_mask"][index].astype(bool)
            positions = shard["R"][index, mask]
            atomic_numbers = shard["Z"][index]
            sampled += 1
            atom_counts.append(np.asarray([mask.sum()]))
            pair_distances = _pair_distances(positions)
            if len(pair_distances):
                distances.append(pair_distances)
            padding_violations += int(np.any(shard["R"][index, ~mask] != 0))
            padding_violations += int(np.any(atomic_numbers[~mask] != 0))
            padding_violations += int(np.any(atomic_numbers[mask] <= 0))

            arrays = [positions, atomic_numbers[mask]]
            if kind == "multipoles":
                multipoles.append(shard["multipoles"][index : index + 1])
                charges.append(np.asarray(shard["Q"][index]).reshape(1))
                arrays.append(shard["multipoles"][index])
            else:
                force = shard["F_mbd"][index, mask]
                c6 = shard["C6_mbd"][index, mask]
                alpha = shard["alpha_mbd"][index, mask]
                energy = np.asarray(shard["E_mbd"][index]).reshape(1)
                mbd_energy.append(energy)
                c6_values.append(c6)
                alpha_values.append(alpha)
                force_scale = max(float(np.linalg.norm(force)), 1e-12)
                net_force_residuals.append(
                    np.asarray([np.linalg.norm(force.sum(axis=0)) / force_scale])
                )
                centered = positions - positions.mean(axis=0, keepdims=True)
                torque_residuals.append(
                    np.asarray(
                        [np.linalg.norm(np.cross(centered, force).sum(axis=0)) / force_scale]
                    )
                )
                padding_violations += int(np.any(shard["F_mbd"][index, ~mask] != 0))
                padding_violations += int(np.any(shard["C6_mbd"][index, ~mask] != 0))
                padding_violations += int(np.any(shard["alpha_mbd"][index, ~mask] != 0))
                arrays.extend((force, c6, alpha, energy))
            nonfinite_values += sum(int(not np.all(np.isfinite(array))) for array in arrays)

    report["kind"] = kind
    report["sampled_structures"] = sampled
    report["integrity"] = {
        "padding_violations": padding_violations,
        "arrays_with_nonfinite_values": nonfinite_values,
    }
    report["atom_count"] = _quantiles(atom_counts)
    if distances:
        report["pair_distance_bohr"] = _quantiles(distances)
        report["pair_distance_angstrom"] = {
            key: value * BOHR_TO_ANGSTROM
            for key, value in report["pair_distance_bohr"].items()
        }

    if kind == "multipoles":
        packed = np.concatenate(multipoles)
        converted = irrep_blocks_to_traceless(jnp.asarray(packed))
        quadrupoles = np.asarray(converted["l2_quadrupole_tensor"])
        octupoles = np.asarray(converted["l3_octupole_tensor"])
        report["multipoles"] = {
            "monopole_charge_max_abs_error": float(
                np.max(np.abs(packed[:, 0] - np.concatenate(charges)))
            ),
            "quadrupole_max_abs_trace": float(
                np.max(np.abs(np.trace(quadrupoles, axis1=-2, axis2=-1)))
            ),
            "octupole_max_abs_trace": float(
                np.max(
                    np.abs(np.trace(octupoles, axis1=-2, axis2=-1))
                )
            ),
            "component_scale_native": _quantiles([packed]),
        }
    else:
        energies = np.concatenate(mbd_energy)
        report["mbd"] = {
            "energy_hartree": _quantiles([energies]),
            "energy_kcal_mol": _quantiles([energies * HARTREE_TO_KCAL_MOL]),
            "positive_energy_fraction": float(np.mean(energies > 0)),
            "c6_native": _quantiles(c6_values),
            "polarizability_bohr3": _quantiles(alpha_values),
            "nonpositive_c6_count": int(
                np.sum(np.concatenate(c6_values) <= 0)
            ),
            "nonpositive_polarizability_count": int(
                np.sum(np.concatenate(alpha_values) <= 0)
            ),
            "relative_net_force_residual": _quantiles(net_force_residuals),
            "relative_torque_residual_bohr": _quantiles(torque_residuals),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--kind", choices=("auto", "multipoles", "mbd"), default="auto")
    parser.add_argument("--max-shards", type=int)
    parser.add_argument("--samples-per-shard", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit_cache(
        args.cache,
        kind=args.kind,
        max_shards=args.max_shards,
        samples_per_shard=args.samples_per_shard,
        seed=args.seed,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
