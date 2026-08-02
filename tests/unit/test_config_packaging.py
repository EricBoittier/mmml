"""Every config the docs tell a user to run must ship with the package.

`mmml/cli/run/*.example.yaml` is a packaging glob, not a naming style: a config
in that directory without the suffix is absent from an installed wheel, so a
documented `--config mmml/cli/run/<name>.yaml` works from a git checkout and
fails everywhere else. These tests pin that contract so the next config added
there cannot quietly break it.

See docs/configs.md.
"""

from __future__ import annotations

import fnmatch
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
# Directories whose YAML is user-facing and must therefore be installed.
SHIPPED_CONFIG_DIRS = (
    "mmml/cli/run",
    "mmml/cli/misc",
    "mmml/mcp/examples",
    "mmml/mcp/recipes",
)


def _package_data_globs() -> list[str]:
    data = tomllib.loads((REPO / "pyproject.toml").read_text())
    return data["tool"]["setuptools"]["package-data"]["mmml"]


def _is_shipped(rel_to_mmml: str, globs: list[str]) -> bool:
    return any(fnmatch.fnmatch(rel_to_mmml, g) for g in globs)


def _tracked_yaml(directory: str) -> list[Path]:
    return sorted((REPO / directory).glob("*.yaml"))


def test_package_data_globs_are_declared():
    globs = _package_data_globs()
    assert "cli/run/*.example.yaml" in globs
    assert "cli/run/presets/*.yaml" in globs


@pytest.mark.parametrize("directory", SHIPPED_CONFIG_DIRS)
def test_every_config_in_a_shipped_dir_is_actually_shipped(directory):
    globs = _package_data_globs()
    unshipped = [
        p.name
        for p in _tracked_yaml(directory)
        if not _is_shipped(str(p.relative_to(REPO / "mmml")), globs)
    ]
    assert not unshipped, (
        f"{directory} holds config(s) no package-data glob matches, so they are "
        f"missing from an installed wheel: {unshipped}. Rename to *.example.yaml "
        f"or extend package-data in pyproject.toml."
    )


def test_cli_run_uses_the_example_suffix():
    """The suffix is what the glob keys on, so it cannot be optional."""
    offenders = [p.name for p in _tracked_yaml("mmml/cli/run") if not p.name.endswith(".example.yaml")]
    assert not offenders, (
        f"mmml/cli/run/ config(s) missing the .example.yaml suffix: {offenders}. "
        "package-data ships cli/run/*.example.yaml only."
    )


def test_shipped_configs_parse():
    yaml = pytest.importorskip("yaml")
    for directory in SHIPPED_CONFIG_DIRS + ("mmml/cli/run/presets",):
        for path in _tracked_yaml(directory):
            try:
                yaml.safe_load(path.read_text())
            except yaml.YAMLError as exc:
                pytest.fail(f"{path.relative_to(REPO)} does not parse: {exc}")


def test_package_code_references_resolve():
    """`mmml env` reports config paths; a rename must not leave them dangling."""
    from mmml.cli import env

    assert env._DCM_RESILIENT.is_file(), (
        f"{env._DCM_RESILIENT} does not exist — mmml/cli/env.py points at a "
        "config that was moved or renamed."
    )
    assert env._PRESETS_DIR.is_dir()


def test_no_dangling_references_to_cli_run_configs():
    """A rename must not leave scripts pointing at a path that no longer exists.

    Renaming the cli/run configs to `.example.yaml` broke two scripts that build
    the path as `REPO / "mmml" / "cli" / "run" / "<name>.yaml"`, which no plain
    grep for the old filename in a config directory would surface.
    """
    import re
    import subprocess

    run_dir = REPO / "mmml" / "cli" / "run"
    literal = re.compile(r"mmml/cli/run/([\w.\-]+\.yaml)")
    joined = re.compile(r"""["']cli["']\s*/\s*["']run["']\s*/\s*["']([\w.\-]+\.yaml)["']""")

    tracked = subprocess.run(
        ["git", "ls-files", "*.py", "*.sh", "*.md", "*.sbatch"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.split()

    dangling: list[str] = []
    for rel in tracked:
        if rel.startswith("build/"):
            continue
        try:
            text = (REPO / rel).read_text(errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            # An unchecked markdown task item names a config that is *planned*,
            # not one that should already resolve.
            if re.match(r"\s*[-*]\s*\[ \]", line):
                continue
            for pattern in (literal, joined):
                for name in pattern.findall(line):
                    if not (run_dir / name).exists() and not (run_dir / "presets" / name).exists():
                        dangling.append(f"{rel} -> mmml/cli/run/{name}")

    assert not dangling, "reference(s) to a non-existent cli/run config:\n  " + "\n  ".join(
        sorted(set(dangling))
    )


def test_generated_resume_bundles_are_not_tracked():
    """`mmml md-system` writes next_run.* after a failure; they are one machine's
    job state, and a committed one has include: paths that go stale at once."""
    import subprocess

    tracked = subprocess.run(
        ["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=False
    ).stdout.split()
    offenders = [t for t in tracked if Path(t).name.startswith("next_run")]
    assert not offenders, f"generated resume bundle(s) are tracked: {offenders[:8]}"
