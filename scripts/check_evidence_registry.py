#!/usr/bin/env python3
"""Validate MMML's scientific-claim evidence registry and inline references."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
REGISTRY = REPO / "docs/evidence-registry.yaml"
REFERENCE_RE = re.compile(r"\[evidence:\s*([a-z0-9_]+)\]")


def _evidence_path(raw: str) -> tuple[Path, str | None]:
    path_text, separator, node = raw.partition("::")
    return REPO / path_text, node if separator else None


def check_registry() -> list[str]:
    data = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    claims = data.get("claims", {}) if isinstance(data, dict) else {}
    errors: list[str] = []
    if data.get("schema_version") != 1 or not isinstance(claims, dict):
        return ["docs/evidence-registry.yaml must contain schema_version 1 and claims"]

    for claim_id, claim in claims.items():
        status = claim.get("status")
        evidence = claim.get("evidence", [])
        if status not in {"verified", "generated", "unverified"}:
            errors.append(f"{claim_id}: invalid status {status!r}")
        if status in {"verified", "generated"} and not evidence:
            errors.append(f"{claim_id}: {status} claim has no evidence")
        if status == "unverified" and not claim.get("needed"):
            errors.append(f"{claim_id}: unverified claim must state needed evidence")
        for raw in evidence:
            path, node = _evidence_path(str(raw))
            if not path.exists():
                errors.append(f"{claim_id}: missing evidence path {path.relative_to(REPO)}")
                continue
            if node and node not in path.read_text(encoding="utf-8", errors="replace"):
                errors.append(f"{claim_id}: pytest node {node!r} not found in {path.relative_to(REPO)}")

    referenced: set[str] = set()
    for root in (REPO / "docs", REPO / "workflows"):
        for path in root.rglob("*.md"):
            referenced.update(REFERENCE_RE.findall(path.read_text(encoding="utf-8")))
    for claim_id in sorted(referenced.difference(claims)):
        errors.append(f"inline reference uses unknown evidence id {claim_id!r}")
    return errors


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    errors = check_registry()
    if errors:
        print("\n".join(f"ERROR: {error}" for error in errors))
        return 1
    print("evidence registry: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
