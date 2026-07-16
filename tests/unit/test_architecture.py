"""Low-cost guards for the repository's supported-code boundaries."""

from __future__ import annotations

import ast
from pathlib import Path

from mmml.cli.registry import COMMAND_REGISTRY


ROOT = Path(__file__).resolve().parents[2]


def test_production_code_does_not_import_operational_trees() -> None:
    """Keep scripts, workflows, and tests out of the distributable API."""
    forbidden = ("scripts", "workflows", "tests")
    violations: list[str] = []
    for path in (ROOT / "mmml").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = (alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = (node.module,)
            else:
                continue
            if any(name == prefix or name.startswith(f"{prefix}.") for name in names for prefix in forbidden):
                violations.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert not violations, "production imports operational code:\n" + "\n".join(violations)


def test_non_active_commands_have_replacements_and_near_term_removal_dates() -> None:
    """Compatibility commands must have an explicit, actionable exit plan."""
    for spec in COMMAND_REGISTRY:
        if spec.status == "active":
            continue
        assert spec.replacement, spec.name
        assert spec.removal_date == "2026-09-01", spec.name
