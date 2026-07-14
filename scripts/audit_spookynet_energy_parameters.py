#!/usr/bin/env python3
"""Audit checkpoint leaves that change fixed energy-function components.

The report is JSON-only and does not initialize JAX.  It records optional
CGenFF-LJ scaling heads and compares trainable ZBL parameters with the nominal
Å/eV defaults implemented by :mod:`physnetjax.models.zbl`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


BOHR_TO_ANGSTROM = 0.529177249
ZBL_DEFAULTS = {
    "a_coefficient": 0.8854 * BOHR_TO_ANGSTROM,
    "a_exponent": 0.23,
    "phi_coefficients": np.asarray([0.18175, 0.50986, 0.28022, 0.02817]),
    "phi_exponents": np.asarray([3.19980, 0.94229, 0.40290, 0.20162]),
}
ELEMENTS = {1: "H", 6: "C", 7: "N", 8: "O", 17: "Cl"}


def _nested(tree: dict, *names: str):
    node = tree
    for name in names:
        if not isinstance(node, dict) or name not in node:
            return None
        node = node[name]
    return node


def audit(path: Path) -> dict:
    payload = json.loads(path.read_text())
    config = payload.get("config", {})
    params = payload.get("params", {})
    if set(params) == {"params"}:
        params = params["params"]
    repulsion = params.get("repulsion", {})
    zbl = {}
    warnings = []
    for name, default in ZBL_DEFAULTS.items():
        value = repulsion.get(name)
        if value is None:
            zbl[name] = {"present": False}
            continue
        array = np.asarray(value, dtype=float)
        default_array = np.asarray(default, dtype=float)
        zbl[name] = {
            "present": True,
            "value": array.tolist(),
            "default": default_array.tolist(),
            "max_abs_difference": float(np.max(np.abs(array - default_array))),
        }
    a_value = np.asarray(repulsion.get("a_coefficient", ZBL_DEFAULTS["a_coefficient"]), dtype=float)
    a_ratio = float(np.abs(a_value) / ZBL_DEFAULTS["a_coefficient"])
    exponent = float(np.abs(np.asarray(repulsion.get("a_exponent", ZBL_DEFAULTS["a_exponent"]))))
    if a_ratio < 0.25 or a_ratio > 4.0:
        warnings.append(f"ZBL a_coefficient is {a_ratio:.4g}x its nominal Å default")
    if exponent < 0.05 or exponent > 1.0:
        warnings.append(f"ZBL a_exponent={exponent:.6g} is outside the nominal sanity interval [0.05, 1]")

    dense = params.get("Dense_13")
    global_scale = params.get("global_vdw_scale")
    element_scale = params.get("element_vdw_scale")
    vdw = {
        "atomic_scale_head_present": isinstance(dense, dict),
        "global_scale_present": global_scale is not None,
        "element_scale_present": element_scale is not None,
        "inferred_predict_atomic_vdw_scale": isinstance(dense, dict),
        "inferred_learn_cgenff_vdw_scale": global_scale is not None and element_scale is not None,
    }
    if global_scale is not None:
        vdw["global_scale"] = np.asarray(global_scale, dtype=float).tolist()
    if element_scale is not None:
        values = np.asarray(element_scale, dtype=float)
        vdw["element_scales"] = {symbol: float(values[z]) for z, symbol in ELEMENTS.items()}

    return {
        "checkpoint": str(path),
        "saved_config": {
            key: config.get(key, "<missing>")
            for key in (
                "cutoff",
                "zbl",
                "no_zbl",
                "no_cgenff_vdw",
                "predict_atomic_vdw_scale",
                "learn_cgenff_vdw_scale",
            )
        },
        "unit_contract": {"distance": "angstrom", "energy": "eV"},
        "zbl": zbl,
        "zbl_a_coefficient_ratio_to_default": a_ratio,
        "cgenff_vdw": vdw,
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {path.stem: audit(path) for path in args.checkpoint}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
