#!/usr/bin/env python3
"""Find merged PRs whose content never reached ``main``.

GitHub marks a pull request "MERGED" when it is merged into **its own base**, not
into ``main``. For a stacked PR that is not the same thing, and the gap is easy
to miss because the PR page looks identical either way.

This actually happened here. #167 was merged into
``fix/charmm-api-read-sticky-append`` at 16:04:40, sixteen seconds *after* that
branch had already merged into ``main`` at 16:04:24. The base branch was gone by
then, so #167's three test files never reached ``main`` -- while the PR showed a
purple "Merged" badge. It was only caught by chance, re-basing something else.

A merged PR is "landed" when its merge commit is an ancestor of ``main``. That is
what this checks. Needs ``gh`` authenticated; skips cleanly (exit 0) when ``gh``
is missing, so it can sit in CI without becoming a hard dependency.

Usage::

    python scripts/ci/check_merged_prs_landed.py            # last 50 merged PRs
    python scripts/ci/check_merged_prs_landed.py --limit 200
    python scripts/ci/check_merged_prs_landed.py --base develop
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys


def _run(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout


def _merged_prs(limit: int) -> list[dict] | None:
    code, out = _run(
        [
            "gh", "pr", "list", "--state", "merged", "--limit", str(limit),
            "--json", "number,title,baseRefName,mergeCommit,mergedAt",
        ]
    )
    if code != 0:
        return None
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="main", help="branch everything must land on")
    ap.add_argument("--limit", type=int, default=50, help="how many merged PRs to check")
    args = ap.parse_args(argv)

    if shutil.which("gh") is None:
        print("check_merged_prs_landed: gh not installed; skipping", file=sys.stderr)
        return 0

    prs = _merged_prs(args.limit)
    if prs is None:
        print("check_merged_prs_landed: gh unavailable or not authenticated; skipping",
              file=sys.stderr)
        return 0

    # Prefer the remote ref: a stale local `main` would produce false positives.
    base = args.base
    if _run(["git", "rev-parse", "--verify", f"origin/{base}"])[0] == 0:
        base = f"origin/{base}"

    stranded: list[dict] = []
    unknown: list[dict] = []
    for pr in prs:
        # Only stacked PRs can strand; a PR based on `base` lands by definition.
        if pr.get("baseRefName") == args.base:
            continue
        sha = (pr.get("mergeCommit") or {}).get("oid")
        if not sha:
            unknown.append(pr)
            continue
        if _run(["git", "cat-file", "-e", f"{sha}^{{commit}}"])[0] != 0:
            unknown.append(pr)  # not fetched locally; cannot judge
            continue
        if _run(["git", "merge-base", "--is-ancestor", sha, base])[0] != 0:
            stranded.append(pr)

    for pr in unknown:
        print(
            f"check_merged_prs_landed: #{pr['number']} merge commit not available "
            f"locally; run `git fetch --all` for a complete answer",
            file=sys.stderr,
        )

    if not stranded:
        print(f"check_merged_prs_landed: all merged PRs reached {base}")
        return 0

    print(f"::error::{len(stranded)} merged PR(s) never reached {base}", file=sys.stderr)
    for pr in stranded:
        print(
            f"  #{pr['number']} {pr['title'][:60]}\n"
            f"      merged into {pr['baseRefName']} at {pr.get('mergedAt')}, "
            f"but {(pr.get('mergeCommit') or {}).get('oid', '?')[:12]} is not an "
            f"ancestor of {base}.\n"
            f"      Re-land it: branch off {base}, cherry-pick the work, open a new PR.",
            file=sys.stderr,
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
