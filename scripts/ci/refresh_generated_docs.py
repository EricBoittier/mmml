"""Refresh or check generated documentation from one entry point.

Write-mode generators update committed pages (CLI reference, package
architecture, crystal literature tables). Check-only generators validate
figures, the evidence registry, and hard-coded recommendation annotations.

CI uses ``--diff``: regenerate, then fail if ``docs/`` or ``mkdocs.yml``
drifted, and write ``generated-docs.patch`` so the stale content is an
artifact rather than a one-line ``stale:`` message.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PATCH = REPO / "generated-docs.patch"

# Generators that rewrite committed files. Each supports ``--check``.
WRITE_GENERATORS: tuple[tuple[str, ...], ...] = (
    ("scripts/generate_cli_docs.py",),
    ("scripts/generate_package_architecture.py",),
    ("scripts/generate_crystal_lit_compare.py",),
)

# Validations that do not rewrite the matching committed page as their
# primary output (figures --check tests presence; the audits are read-only).
CHECK_GENERATORS: tuple[tuple[str, ...], ...] = (
    ("scripts/generate_docs_figures.py", "--check"),
    ("scripts/check_evidence_registry.py",),
    ("scripts/audit_hardcoded_recommendations.py",),
)

DIFF_PATHS: tuple[str, ...] = ("docs", "mkdocs.yml")


def _run(args: tuple[str, ...]) -> int:
    cmd = [sys.executable, str(REPO / args[0]), *args[1:]]
    print("+", " ".join(cmd[1:]), flush=True)
    return subprocess.run(cmd, cwd=REPO, check=False).returncode


def _run_write(*, check: bool) -> int:
    extra = ("--check",) if check else ()
    for args in WRITE_GENERATORS:
        rc = _run((*args, *extra))
        if rc:
            return rc
    return 0


def _run_checks() -> int:
    for args in CHECK_GENERATORS:
        rc = _run(args)
        if rc:
            return rc
    return 0


def working_tree_drift() -> tuple[str, str]:
    """Return (diff_text, untracked_listing) for generated-doc paths."""
    diff = subprocess.run(
        ["git", "diff", "--", *DIFF_PATHS],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "--", *DIFF_PATHS],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )
    return diff.stdout, untracked.stdout


def fail_if_dirty() -> int:
    diff_text, untracked = working_tree_drift()
    if not diff_text and not untracked.strip():
        print("generated docs: working tree matches generators")
        if PATCH.is_file():
            PATCH.unlink()
        return 0
    PATCH.write_text(diff_text, encoding="utf-8")
    if diff_text:
        print(diff_text, end="" if diff_text.endswith("\n") else "\n")
    if untracked.strip():
        print("untracked generated files:")
        print(untracked, end="" if untracked.endswith("\n") else "\n")
    print(f"stale generated docs; patch written to {PATCH.name}", file=sys.stderr)
    print(
        "Regenerate locally: uv run python scripts/ci/refresh_generated_docs.py --write",
        file=sys.stderr,
    )
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check",
        action="store_true",
        help="Fail if committed generated pages are stale (no rewrite)",
    )
    mode.add_argument(
        "--write",
        action="store_true",
        help="Rewrite generated pages, then run read-only checks",
    )
    mode.add_argument(
        "--diff",
        action="store_true",
        help="Rewrite, then fail if git still sees drift (CI regen job)",
    )
    args = parser.parse_args(argv)
    if not (args.check or args.write or args.diff):
        parser.error("one of --check, --write, or --diff is required")

    if args.check:
        rc = _run_write(check=True)
        if rc:
            return rc
        return _run_checks()

    rc = _run_write(check=False)
    if rc:
        return rc
    rc = _run_checks()
    if rc:
        return rc
    if args.diff:
        return fail_if_dirty()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
