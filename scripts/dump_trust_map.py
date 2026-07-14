"""Dump a trained model's neural-interaction trust map as a per-element-pair fingerprint.

The trust map is a learned log-lambda matrix over (H, C, N, O, S, Cl). After
softplus and symmetrisation, lambda_c is the shrinkage the model applies to its neural
interaction energy for element pair c. Small lambda = the data justified a large learned
correction ("trusts the neural term"); large lambda = shrunk to the CGenFF prior ("falls
back to physics"). Reading lambda_c off is a data-provenance fingerprint -- with the
caveat that a chemistry the prior already handles also reads as large lambda, so this is
a trust map, not a membership oracle.

Usage:
    python scripts/dump_trust_map.py path/to/step-000XXXXX_params.json
"""

from __future__ import annotations

import json
import sys

import numpy as np

ELEMENTS = (1, 6, 7, 8, 16, 17)
SYMBOL = {1: "H", 6: "C", 7: "N", 8: "O", 16: "S", 17: "Cl"}
KEY = "neural_interaction_log_lambda"


def _find(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            found = _find(v, key)
            if found is not None:
                return found
    return None


def _softplus(x):
    return np.logaddexp(0.0, x)


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    path = sys.argv[1]
    with open(path) as fh:
        params = json.load(fh)

    raw = _find(params, KEY)
    if raw is None:
        sys.exit(
            f"No '{KEY}' in {path}. The checkpoint was not trained with "
            "--interaction-trust-map."
        )
    log_lambda = np.asarray(raw, dtype=float)
    lam = _softplus(log_lambda)
    lam = 0.5 * (lam + lam.T)

    print(f"Trust map from {path}")
    print("lambda_c = per-element-pair shrinkage of the neural interaction energy")
    print("  small  -> model relies on a learned correction (data-informed)")
    print("  large  -> shrunk to the CGenFF prior (falls back to physics)\n")

    syms = [SYMBOL[z] for z in ELEMENTS]
    header = "       " + "".join(f"{s:>7}" for s in syms)
    print(header)
    for i, si in enumerate(syms):
        row = f"  {si:<4}" + "".join(f"{lam[i, j]:7.3f}" for j in range(len(syms)))
        print(row)

    # ranked unique pairs (upper triangle incl. diagonal)
    pairs = []
    for i in range(len(syms)):
        for j in range(i, len(syms)):
            pairs.append((f"{syms[i]}-{syms[j]}", float(lam[i, j])))
    pairs.sort(key=lambda t: t[1])
    print("\nranked (most trusted / least shrunk first):")
    for name, v in pairs:
        print(f"  {name:<7} lambda = {v:7.3f}")


if __name__ == "__main__":
    main()
