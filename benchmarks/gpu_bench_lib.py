"""GPU ASV runner: correctness first, then timing, then a browser-readable report.

The asv suite under ``benchmarks/benchmarks/`` is the source of the timings.
This module wraps it so a GPU job refuses a CPU fallback, checks that the same
kernels produce finite (and, for MM, force–energy consistent) results, and only
then spends the allocation on ``asv run``.

Results publishing stays the existing asv path: ``asv publish`` regenerates
``benchmarks/html/`` from ``benchmarks/results/``. CI does not upload that
HTML; the JSON under ``results/`` is what you commit so later publishes have
history. This module writes a standalone ``gpu-report.html`` next to asv's
index so a run is readable without ``asv preview``.
"""

from __future__ import annotations

import html
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
ASV_BENCH_DIR = Path(__file__).resolve().parent / "benchmarks"
DEFAULT_CKPT = REPO_ROOT / "examples" / "ckpts_json" / "DESdimers_params.json"
HTML_DIR = REPO_ROOT / "benchmarks" / "html"
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"
REPORT_HTML_NAME = "gpu-report.html"
REPORT_JSON_NAME = "gpu-report.json"

_METADATA_JSON_NAMES = frozenset({"machine.json", "benchmarks.json"})


@dataclass(frozen=True, slots=True)
class CheckResult:
    """One pre-timing probe: ``pass``, ``fail``, or ``skip``."""

    name: str
    status: str
    detail: str
    values: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == "pass"


@dataclass(frozen=True, slots=True)
class TimingRow:
    name: str
    params: dict[str, str]
    value: float | None


def all_blocking_checks_passed(checks: Sequence[CheckResult]) -> bool:
    """Skips do not block timing; only an explicit ``fail`` does."""
    return all(c.status != "fail" for c in checks)


def finite_array_report(name: str, value: Any) -> dict[str, Any]:
    arr = np_asarray(value)
    finite = bool(arr.size) and bool(_all_finite(arr))
    abs_max = float(_abs_max(arr)) if arr.size else 0.0
    return {
        "name": name,
        "shape": list(arr.shape),
        "finite": finite,
        "abs_max": abs_max,
    }


def force_energy_relative_error(
    energy_0: float,
    energy_plus: float,
    energy_minus: float,
    *,
    force_component: float,
    eps: float,
) -> float:
    """``|g_fd - g| / max(|g|, |g_fd|, 1e-12)`` from a central difference of E.

    ``force_component`` is ``dE/dx`` (``jax.grad``), matching the asv MM benches,
    not the physical force ``-∇E``. ``energy_0`` documents the expansion point.
    """
    del energy_0
    fd = (float(energy_plus) - float(energy_minus)) / (2.0 * float(eps))
    denom = max(abs(float(force_component)), abs(fd), 1e-12)
    return abs(fd - float(force_component)) / denom


def check_jax_gpu_device(devices: Sequence[Any] | None = None) -> CheckResult:
    """Refuse to time if JAX came up on CPU (classic wasted GPU allocation)."""
    if devices is None:
        try:
            import jax

            devices = list(jax.devices())
        except Exception as exc:  # pragma: no cover - environment-dependent
            return CheckResult("jax_gpu", "fail", f"jax unavailable: {exc}")
    if not devices:
        return CheckResult("jax_gpu", "fail", "no JAX devices")
    platform_name = str(getattr(devices[0], "platform", devices[0])).lower()
    detail = f"{devices[0]}"
    if "gpu" not in platform_name and "cuda" not in platform_name:
        return CheckResult(
            "jax_gpu",
            "fail",
            f"refusing to time on the {platform_name} backend ({detail})",
            values={"platform": platform_name, "devices": [str(d) for d in devices]},
        )
    return CheckResult(
        "jax_gpu",
        "pass",
        detail,
        values={"platform": platform_name, "devices": [str(d) for d in devices]},
    )


def prepare_bench_env(
    repo_root: Path,
    *,
    allow_cpu: bool,
    apply: bool = False,
) -> dict[str, str]:
    """Match ``run_bench.sh`` / ``slurm_bench_gpu.sh`` so numbers stay comparable.

    Returns the planned env. Does not write ``os.environ`` unless ``apply`` is
    true — unit tests that only inspect the dict must not leak ``MMML_CKPT``
    into later cases (that took down ``test_warmup_mlpot_jax_missing_checkpoint_exits``).
    """
    x64 = os.environ.get("MMML_BENCH_X64", "1")
    planned: dict[str, str] = {
        "MMML_BENCH_X64": x64,
        "JAX_ENABLE_X64": os.environ.get("JAX_ENABLE_X64", x64),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "1"),
    }
    mmml_ckpt = os.environ.get("MMML_CKPT")
    bench_ckpt = os.environ.get("MMML_BENCH_CKPT")
    # Keep the two names distinct. Benches prefer MMML_BENCH_CKPT then
    # MMML_CKPT; copying a bench-only override into MMML_CKPT used to
    # poison later checkpoint resolution (and apply=True leaked it).
    if mmml_ckpt:
        planned["MMML_CKPT"] = mmml_ckpt
    if bench_ckpt:
        planned["MMML_BENCH_CKPT"] = bench_ckpt
    if not mmml_ckpt and not bench_ckpt:
        default = Path(repo_root) / "examples" / "ckpts_json" / "DESdimers_params.json"
        planned["MMML_CKPT"] = str(default)
    if not allow_cpu:
        planned["JAX_PLATFORMS"] = os.environ.get("JAX_PLATFORMS", "cuda")
    machine = os.environ.get("ASV_MACHINE")
    if machine:
        planned["ASV_MACHINE"] = machine
    if apply:
        for key, value in planned.items():
            os.environ.setdefault(key, value)
    return planned


def build_asv_run_argv(
    *,
    commit: str,
    bench: str | None = None,
    append_samples: bool = False,
    machine: str | None = None,
) -> list[str]:
    """``--set-commit-hash`` is required under ``environment_type=existing``."""
    argv = ["run", "--set-commit-hash", str(commit)]
    if machine:
        argv.extend(["--machine", str(machine)])
    if bench:
        argv.extend(["--bench", str(bench)])
    if append_samples:
        argv.append("--append-samples")
    return argv


def build_asv_publish_argv() -> list[str]:
    return ["publish"]


def resolve_asv_command(repo_root: Path) -> list[str]:
    venv_asv = Path(repo_root) / ".venv" / "bin" / "asv"
    if venv_asv.is_file() and os.access(venv_asv, os.X_OK):
        return [str(venv_asv)]
    # asv lives in the "dev" extra (pyproject.toml); plain `uv run asv` uses
    # the base env and fails with "Failed to spawn: asv".
    return ["uv", "run", "--extra", "dev", "asv"]


def latest_asv_result_files(results_dir: Path) -> list[Path]:
    if not results_dir.is_dir():
        return []
    files = [
        p
        for p in results_dir.glob("*/*.json")
        if p.name not in _METADATA_JSON_NAMES
    ]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def extract_timing_rows(payload: Mapping[str, Any]) -> list[TimingRow]:
    """Best-effort parse of asv 0.5/0.6 result JSON."""
    raw = payload.get("results") or {}
    if not isinstance(raw, Mapping):
        return []
    rows: list[TimingRow] = []
    for name, entry in raw.items():
        values, param_grid, param_names = _coerce_result_entry(entry)
        if not values:
            if isinstance(entry, (int, float)):
                rows.append(TimingRow(str(name), {}, float(entry)))
            continue
        for idx, value in enumerate(values):
            params = _params_for_index(param_grid, param_names, idx)
            rows.append(
                TimingRow(
                    str(name),
                    params,
                    None if value is None else float(value),
                )
            )
    return rows


def format_bench_value(name: str, value: float | None) -> str:
    if value is None:
        return "—"
    if "ns_per_day" in name:
        return f"{value:.2f} ns/day"
    if ".time_" in name or name.startswith("time_"):
        if abs(value) < 1.0:
            return f"{value * 1000.0:.2f} ms"
        return f"{value:.3f} s"
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}"


def render_gpu_report_html(
    *,
    meta: Mapping[str, Any],
    checks: Sequence[CheckResult],
    timings: Sequence[TimingRow],
    asv_html_href: str | None = None,
) -> str:
    ok = all_blocking_checks_passed(checks)
    banner = "PASS" if ok else "FAIL"
    banner_class = "pass" if ok else "fail"
    check_rows = "\n".join(_html_check_row(c) for c in checks)
    timing_rows = "\n".join(_html_timing_row(t) for t in timings)
    if not timings:
        timing_rows = (
            '<tr><td colspan="3" class="muted">No asv timings in this report '
            "(checks-only, or results JSON not found).</td></tr>"
        )
    asv_link = ""
    if asv_html_href:
        asv_link = (
            f'<p>Full asv graphs: <a href="{html.escape(asv_html_href)}">'
            f"{html.escape(asv_html_href)}</a> "
            f"(or <code>uv run asv preview</code>).</p>"
        )
    commit = html.escape(str(meta.get("commit") or "unknown"))
    host = html.escape(str(meta.get("hostname") or "unknown"))
    gpu = html.escape(str(meta.get("gpu") or "unknown"))
    dirty = "yes" if meta.get("dirty") else "no"
    generated = html.escape(str(meta.get("generated") or _now_iso()))
    x64 = html.escape(str(meta.get("x64") or os.environ.get("MMML_BENCH_X64", "?")))
    platforms = html.escape(str(meta.get("jax_platforms") or os.environ.get("JAX_PLATFORMS", "auto")))
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>mmml GPU benchmark — {banner}</title>
<style>
  :root {{
    --bg: #0f1419; --card: #1a222c; --ink: #e7ecf1; --muted: #93a1b0;
    --pass: #3dd68c; --fail: #ff6b6b; --skip: #f0c14b; --line: #2a3644;
  }}
  body {{
    margin: 0; font-family: ui-sans-serif, system-ui, sans-serif;
    background: var(--bg); color: var(--ink); line-height: 1.45;
  }}
  main {{ max-width: 960px; margin: 0 auto; padding: 2rem 1.25rem 4rem; }}
  h1 {{ font-size: 1.4rem; margin: 0 0 0.4rem; }}
  h2 {{ font-size: 1.05rem; margin: 2rem 0 0.6rem; }}
  .banner {{
    display: inline-block; padding: 0.15rem 0.6rem; border-radius: 999px;
    font-weight: 700; letter-spacing: 0.04em;
  }}
  .banner.pass {{ background: #163524; color: var(--pass); }}
  .banner.fail {{ background: #3a1717; color: var(--fail); }}
  .card {{
    background: var(--card); border: 1px solid var(--line);
    border-radius: 10px; padding: 1rem 1.1rem; margin-top: 0.8rem;
  }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.92rem; }}
  th, td {{ text-align: left; padding: 0.4rem 0.5rem; border-bottom: 1px solid var(--line); vertical-align: top; }}
  th {{ color: var(--muted); font-weight: 600; }}
  .muted {{ color: var(--muted); }}
  .st-pass {{ color: var(--pass); font-weight: 700; }}
  .st-fail {{ color: var(--fail); font-weight: 700; }}
  .st-skip {{ color: var(--skip); font-weight: 700; }}
  code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.88em; }}
  a {{ color: #8cb4ff; }}
  .note {{ color: var(--muted); font-size: 0.92rem; }}
</style>
</head>
<body>
<main>
  <p><span class="banner {banner_class}">{banner}</span></p>
  <h1>mmml GPU benchmark</h1>
  <p class="muted">Correctness probes run before asv timings. Generated {generated}.</p>

  <div class="card">
    <table>
      <tr><th>Commit</th><td><code>{commit}</code></td></tr>
      <tr><th>Dirty tree</th><td>{dirty}</td></tr>
      <tr><th>Host</th><td>{host}</td></tr>
      <tr><th>GPU</th><td>{gpu}</td></tr>
      <tr><th>Precision</th><td>x64={x64}</td></tr>
      <tr><th>JAX_PLATFORMS</th><td><code>{platforms}</code></td></tr>
    </table>
  </div>

  <h2>Correctness</h2>
  <div class="card">
    <table>
      <thead><tr><th>Check</th><th>Status</th><th>Detail</th></tr></thead>
      <tbody>
{check_rows}
      </tbody>
    </table>
  </div>

  <h2>Timings</h2>
  <div class="card">
    <table>
      <thead><tr><th>Benchmark</th><th>Params</th><th>Value</th></tr></thead>
      <tbody>
{timing_rows}
      </tbody>
    </table>
    {asv_link}
  </div>

  <h2>How results are published</h2>
  <div class="card note">
    <p>This repository does <strong>not</strong> upload ASV HTML from CI.
    GitHub Actions builds MkDocs only; the speed suite is manual on a GPU node.</p>
    <p><code>asv publish</code> regenerates <code>benchmarks/html/</code> from
    <code>benchmarks/results/</code>. The HTML directory is gitignored.
    Commit the per-commit JSON under <code>results/&lt;machine&gt;/</code> —
    asv's regression view is only as long as the history that is checked in.</p>
    <p><code>asv.conf.json</code> tracks the <code>main</code> branch. A feature-branch
    run still writes JSON, but <code>asv publish</code> leaves that point off the
    graphs until the branch merges. View locally with
    <code>uv run asv preview</code> or by opening this file in a browser.</p>
  </div>
</main>
</body>
</html>
"""


def render_gpu_report_json(
    *,
    meta: Mapping[str, Any],
    checks: Sequence[CheckResult],
    timings: Sequence[TimingRow],
) -> str:
    payload = {
        "ok": all_blocking_checks_passed(checks),
        "meta": dict(meta),
        "checks": [asdict(c) for c in checks],
        "timings": [asdict(t) for t in timings],
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def collect_run_meta(
    *,
    repo_root: Path,
    allow_cpu: bool = False,
) -> dict[str, Any]:
    env = prepare_bench_env(repo_root, allow_cpu=allow_cpu)
    gpu = probe_nvidia_smi() or _jax_device_label()
    return {
        "commit": git_commit(repo_root),
        "dirty": git_is_dirty(repo_root),
        "hostname": platform.node(),
        "gpu": gpu,
        "x64": env.get("MMML_BENCH_X64", "1"),
        "jax_platforms": env.get("JAX_PLATFORMS", os.environ.get("JAX_PLATFORMS", "auto")),
        "generated": _now_iso(),
        "python": sys.version.split()[0],
        "machine": os.environ.get("ASV_MACHINE") or None,
    }


def probe_nvidia_smi() -> str | None:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    text = proc.stdout.strip()
    return text or None


def git_commit(repo_root: Path) -> str:
    return _git(repo_root, ["rev-parse", "HEAD"]) or "unknown"


def git_is_dirty(repo_root: Path) -> bool:
    out = _git(
        repo_root,
        ["status", "--porcelain", "--", ":!benchmarks/results", ":!benchmarks/html"],
    )
    return bool(out)


@dataclass(frozen=True, slots=True)
class CheckSpec:
    """One probe and the asv ``--bench`` fragments that should trigger it."""

    name: str
    fn: Callable[[], CheckResult]
    tags: frozenset[str]


def correctness_check_catalog() -> tuple[CheckSpec, ...]:
    """Kernel probes. ``jax_gpu`` is always first and not listed here."""
    return (
        CheckSpec(
            "physnet",
            check_physnet_energy_forces,
            frozenset({"bench_ml_physnet", "physnet", "spooky", "zbl"}),
        ),
        CheckSpec(
            "mm_nonbonded",
            check_mm_nonbonded,
            frozenset(
                {
                    "bench_mm_energy",
                    "mm_nonbonded",
                    "mmnonbonded",
                    "ewald",
                    "bench_md_driver",
                    "mdsystemsize",
                }
            ),
        ),
        CheckSpec(
            "neighbors",
            check_neighbors,
            frozenset(
                {
                    "bench_neighbors",
                    "pairlist",
                    "vesin",
                    "verlet",
                    "bench_md_driver",
                    "mdsystemsize",
                }
            ),
        ),
        CheckSpec(
            "shake",
            check_shake_projection,
            frozenset(
                {
                    "bench_constraints",
                    "shake",
                    "constrainednve",
                    "bench_md_driver",
                    "mdsystemsize",
                }
            ),
        ),
        CheckSpec(
            "rattle",
            check_rattle_projection,
            frozenset(
                {
                    "bench_constraints",
                    "rattle",
                    "constrainednve",
                    "bench_md_driver",
                    "mdsystemsize",
                }
            ),
        ),
        CheckSpec(
            "data",
            check_batch_pair_indices,
            frozenset({"bench_data", "batch", "pair_indices", "prepare_batches"}),
        ),
        CheckSpec(
            "calculator",
            check_calculator_fixture,
            frozenset({"bench_calculator", "setup_calculator", "spherical"}),
        ),
    )


def select_correctness_checks(
    *,
    bench: str | None = None,
    only: Sequence[str] | None = None,
) -> list[str]:
    """Names to run. ``jax_gpu`` is always included unless ``only`` lists checks.

    ``--bench bench_ml_physnet`` runs the PhysNet probe only (plus the device
    gate). An unmatched regex keeps the full set so a typo does not skip the
    gate. ``only`` (``--check``) is an explicit allow-list.
    """
    catalog = correctness_check_catalog()
    known = ["jax_gpu", *[spec.name for spec in catalog]]
    if only:
        unknown = [name for name in only if name not in known]
        if unknown:
            raise ValueError(
                f"unknown check(s) {unknown}; known: {', '.join(known)}"
            )
        selected = list(dict.fromkeys(only))
        if "jax_gpu" not in selected:
            selected = ["jax_gpu", *selected]
        return selected
    if not bench or not str(bench).strip():
        return known
    needle = str(bench).strip().lower()
    matched = [
        spec.name
        for spec in catalog
        if _spec_matches_bench(spec, needle)
    ]
    if not matched:
        return known
    return ["jax_gpu", *matched]


def run_correctness_checks(
    *,
    require_gpu: bool = True,
    bench: str | None = None,
    only: Sequence[str] | None = None,
    extra: Sequence[Callable[[], CheckResult]] | None = None,
) -> list[CheckResult]:
    """Device gate, then the kernels the selected asv benches actually time."""
    wanted = select_correctness_checks(bench=bench, only=only)
    catalog = {spec.name: spec for spec in correctness_check_catalog()}
    checks: list[CheckResult] = []
    if "jax_gpu" in wanted:
        if require_gpu:
            checks.append(check_jax_gpu_device())
            if not checks[-1].passed:
                return checks
        else:
            checks.append(
                CheckResult("jax_gpu", "skip", "CPU allowed (--allow-cpu)")
            )
    for name in wanted:
        if name == "jax_gpu":
            continue
        spec = catalog.get(name)
        if spec is None:
            continue
        checks.append(_run_named_check(spec.name, spec.fn))
    if extra:
        for fn in extra:
            checks.append(_run_named_check(getattr(fn, "__name__", "extra"), fn))
    return checks


def check_physnet_energy_forces() -> CheckResult:
    """Tiny PhysNet forward+backward: finite energy and non-zero finite forces."""
    from benchmarks.benchmarks._common import default_checkpoint
    from benchmarks.benchmarks.bench_ml_physnet import _dense_inputs
    from mmml.cli.base import (
        load_physnet_params_and_ef_model,
        resolve_checkpoint_paths,
    )

    # Use the checkpoint's architecture, not the synthetic scaling benchmark's
    # fixed architecture. In particular, feature width and message-pass depth
    # can differ for an MMML_BENCH_CKPT override.
    checkpoint = default_checkpoint().expanduser()
    if checkpoint.is_dir() and (checkpoint / "params.json").is_file():
        checkpoint = checkpoint / "params.json"
    resolved, epoch = resolve_checkpoint_paths(checkpoint)
    params, model = load_physnet_params_and_ef_model(
        resolved, natoms=20, orbax_epoch_dir=epoch
    )
    inputs = _dense_inputs(20)
    energy = model.apply(params, compute_forces=False, **inputs)
    forces = model.apply(params, compute_forces=True, **inputs)
    if hasattr(energy, "energy"):
        energy_arr = energy.energy
    elif isinstance(energy, Mapping):
        energy_arr = energy.get("energy", energy)
    else:
        energy_arr = energy
    if isinstance(forces, Mapping):
        force_arr = forces.get("forces", forces)
    elif hasattr(forces, "forces"):
        force_arr = forces.forces
    else:
        force_arr = forces
    e_info = finite_array_report("energy", energy_arr)
    f_info = finite_array_report("forces", force_arr)
    if not e_info["finite"]:
        return CheckResult("physnet", "fail", "energy is not finite", values=e_info)
    if not f_info["finite"]:
        return CheckResult("physnet", "fail", "forces are not finite", values=f_info)
    if f_info["abs_max"] <= 0.0:
        return CheckResult("physnet", "fail", "forces are all zero", values=f_info)
    return CheckResult(
        "physnet",
        "pass",
        f"energy abs_max={e_info['abs_max']:.4g}; force abs_max={f_info['abs_max']:.4g}",
        values={"energy": e_info, "forces": f_info},
    )


def check_mm_nonbonded() -> CheckResult:
    """Small switched MM energy: finite values + one-component force–energy check."""
    _import_asv_helpers()
    from _common import block, padded_pair_list, require_jax, synthetic_system  # type: ignore

    jax = require_jax()
    import jax.numpy as jnp

    from mmml.md.energy import EnergyContext
    from mmml.md.energy.terms import MMNonbondedTerm
    from mmml.interfaces.pycharmmInterface.mm_system_energy import CharmmNbondSettings

    # 32 waters at liquid density is ~9.9 Å; a 4 Å cutoff stays inside unique MIC.
    cutoff = 4.0
    system, _box = synthetic_system(32)
    settings = CharmmNbondSettings(cutnb=cutoff, ctonnb=3.0, ctofnb=cutoff)
    term = MMNonbondedTerm(settings, lr_solver="mic")
    energy_fn = term.make(system, EnergyContext()).jax_energy_fn
    pairs = padded_pair_list(system, cutoff)
    kw = dict(
        pair_i=jnp.asarray(pairs["pair_i"]),
        pair_j=jnp.asarray(pairs["pair_j"]),
        pair_mask=jnp.asarray(pairs["pair_mask"]),
    )
    energy = jax.jit(lambda R: energy_fn(R, **kw))
    forces = jax.jit(jax.grad(lambda R: energy_fn(R, **kw)))
    r = jnp.asarray(system.R)
    e0 = float(block(energy(r)))
    f0 = block(forces(r))
    e_info = finite_array_report("energy", e0)
    f_info = finite_array_report("forces", f0)
    if not e_info["finite"]:
        return CheckResult("mm_nonbonded", "fail", "energy is not finite", values=e_info)
    if not f_info["finite"]:
        return CheckResult("mm_nonbonded", "fail", "forces are not finite", values=f_info)

    eps = 1e-4
    r_plus = r.at[0, 0].add(eps)
    r_minus = r.at[0, 0].add(-eps)
    e_plus = float(block(energy(r_plus)))
    e_minus = float(block(energy(r_minus)))
    # ``forces`` is ``jax.grad(energy)`` (dE/dR), matching the asv MM benches.
    rel = force_energy_relative_error(
        e0, e_plus, e_minus, force_component=float(np_asarray(f0)[0, 0]), eps=eps
    )
    values = {"energy": e_info, "forces": f_info, "force_energy_rel": rel}
    if rel > 5e-3:
        return CheckResult(
            "mm_nonbonded",
            "fail",
            f"force–energy relative error {rel:.3g} exceeds 5e-3",
            values=values,
        )
    return CheckResult(
        "mm_nonbonded",
        "pass",
        f"energy={e0:.4g}; force–energy rel={rel:.2e}",
        values=values,
    )


def check_shake_projection() -> CheckResult:
    """SHAKE must shrink constraint residuals on a slightly perturbed water box."""
    _import_asv_helpers()
    from _common import block, require_jax, water_box  # type: ignore

    require_jax()
    import jax.numpy as jnp
    import numpy as np

    from mmml.md.constraints import constraint_residuals, shake_positions, tip3_rigid_constraints

    spec = tip3_rigid_constraints(8)
    box = water_box(8, seed=3)
    rng = np.random.default_rng(3)
    reference = jnp.asarray(box["R"])
    perturbed = jnp.asarray(box["R"] + rng.normal(scale=0.02, size=box["R"].shape))
    before = np_asarray(block(constraint_residuals(perturbed, spec)))
    shaken = block(
        shake_positions(perturbed, reference, spec, iterations=40, box=None)
    )
    after = np_asarray(block(constraint_residuals(shaken, spec)))
    before_max = float(np.max(np.abs(before)))
    after_max = float(np.max(np.abs(after)))
    values = {"residual_max_before": before_max, "residual_max_after": after_max}
    if after_max >= before_max:
        return CheckResult(
            "shake",
            "fail",
            f"residuals did not shrink ({before_max:.3g} → {after_max:.3g})",
            values=values,
        )
    if after_max > 1e-6:
        return CheckResult(
            "shake",
            "fail",
            f"residuals still large after SHAKE ({after_max:.3g})",
            values=values,
        )
    return CheckResult(
        "shake",
        "pass",
        f"residual max {before_max:.3g} → {after_max:.3g}",
        values=values,
    )


def check_rattle_projection() -> CheckResult:
    """RATTLE must remove the velocity component along each constrained bond."""
    _import_asv_helpers()
    from _common import block, require_jax, water_box  # type: ignore

    require_jax()
    import jax.numpy as jnp
    import numpy as np

    from mmml.md.constraints import rattle_velocities, tip3_rigid_constraints

    spec = tip3_rigid_constraints(4)
    box = water_box(4, seed=4)
    rng = np.random.default_rng(4)
    positions = jnp.asarray(box["R"])
    velocities = jnp.asarray(rng.normal(scale=0.15, size=box["R"].shape))
    pairs = np.asarray(spec.pairs, dtype=np.int64)

    def _bond_rv(vel) -> float:
        v = np_asarray(vel)
        r = np_asarray(positions)
        rv = []
        for i, j in pairs:
            d = r[i] - r[j]
            dv = v[i] - v[j]
            rv.append(float(np.abs(np.dot(d, dv))))
        return float(max(rv)) if rv else 0.0

    before = _bond_rv(velocities)
    rattled = block(
        rattle_velocities(velocities, positions, spec, iterations=20, box=None)
    )
    after = _bond_rv(rattled)
    values = {"rv_max_before": before, "rv_max_after": after}
    if after >= before and before > 0.0:
        return CheckResult(
            "rattle",
            "fail",
            f"bond-parallel velocity did not shrink ({before:.3g} → {after:.3g})",
            values=values,
        )
    if after > 1e-5:
        return CheckResult(
            "rattle",
            "fail",
            f"bond-parallel velocity still large after RATTLE ({after:.3g})",
            values=values,
        )
    return CheckResult(
        "rattle",
        "pass",
        f"bond |r·Δv| max {before:.3g} → {after:.3g}",
        values=values,
    )


def check_neighbors() -> CheckResult:
    """Tiny unique-MIC pair list: sorted ``i < j`` and matches the NumPy fallback."""
    _import_asv_helpers()
    from _common import water_box  # type: ignore

    from mmml.interfaces.pycharmmInterface.mm_system_energy import (
        _build_pair_indices,
        _build_pair_indices_vectorized,
    )
    import numpy as np

    box = water_box(8, seed=5)
    cutoff = 3.0
    cell = box["box"]
    sides = np.linalg.norm(np.asarray(cell), axis=1)
    if float(np.min(sides)) < 2.0 * cutoff:
        return CheckResult(
            "neighbors",
            "skip",
            f"water_box(8) L={float(np.min(sides)):.2f} Å is not unique-MIC at {cutoff} Å",
        )
    pi, pj = _build_pair_indices(box["R"], cell, frozenset(), cutoff)
    qi, qj = _build_pair_indices_vectorized(box["R"], cell, frozenset(), cutoff)
    pi, pj, qi, qj = (np.asarray(a, dtype=np.int64) for a in (pi, pj, qi, qj))
    if pi.size == 0:
        return CheckResult("neighbors", "fail", "pair list is empty at 3 Å")
    if not bool(np.all(pi < pj)):
        return CheckResult("neighbors", "fail", "pair list is not i < j")
    got = set(zip(pi.tolist(), pj.tolist()))
    ref = set(zip(qi.tolist(), qj.tolist()))
    if got != ref:
        return CheckResult(
            "neighbors",
            "fail",
            f"dispatch/NumPy pair sets differ ({len(got)} vs {len(ref)})",
            values={"n_dispatch": int(pi.size), "n_numpy": int(qi.size)},
        )
    return CheckResult(
        "neighbors",
        "pass",
        f"{int(pi.size)} unique-MIC pairs; dispatch matches NumPy",
        values={"n_pairs": int(pi.size)},
    )


def check_batch_pair_indices() -> CheckResult:
    """``_pair_indices`` layout used by ``prepare_batches_fast`` / ``pair_cache``."""
    from mmml.models.physnetjax.physnetjax.data.batches import _pair_indices

    n_atoms, batch_size = 6, 2
    segs, offsets, dst_2d, src_2d, dst_flat, src_flat = _pair_indices(n_atoms, batch_size)
    n_pairs = n_atoms * (n_atoms - 1)
    values = {
        "dst_2d": list(getattr(dst_2d, "shape", ())),
        "src_flat": list(getattr(src_flat, "shape", ())),
    }
    if tuple(dst_2d.shape) != (batch_size, n_pairs):
        return CheckResult(
            "data",
            "fail",
            f"dst_2d shape {tuple(dst_2d.shape)} != {(batch_size, n_pairs)}",
            values=values,
        )
    if int(dst_flat.size) != batch_size * n_pairs:
        return CheckResult("data", "fail", "flat pair index size mismatch", values=values)
    if int(offsets.shape[0]) != batch_size or int(segs.shape[0]) != batch_size * n_atoms:
        return CheckResult("data", "fail", "segment/offset layout mismatch", values=values)
    return CheckResult(
        "data",
        "pass",
        f"_pair_indices({n_atoms}, {batch_size}) layout ok",
        values=values,
    )


def check_calculator_fixture() -> CheckResult:
    """Checkpoint + ACO geometry the calculator benches load (no full compile)."""
    _import_asv_helpers()
    from _common import aco_cluster, default_checkpoint  # type: ignore

    ckpt = default_checkpoint()
    if not Path(ckpt).is_file():
        return CheckResult(
            "calculator",
            "skip",
            f"checkpoint missing: {ckpt}",
        )
    geom = aco_cluster(2)
    r_info = finite_array_report("R", geom["R"])
    if not r_info["finite"]:
        return CheckResult("calculator", "fail", "ACO cluster coordinates are not finite")
    n_atoms = int(geom["R"].shape[0])
    if n_atoms < 4:
        return CheckResult("calculator", "fail", f"ACO cluster too small ({n_atoms} atoms)")
    return CheckResult(
        "calculator",
        "pass",
        f"ckpt={Path(ckpt).name}; ACO:2 has {n_atoms} atoms",
        values={"checkpoint": str(ckpt), "n_atoms": n_atoms},
    )


def run_asv(
    *,
    repo_root: Path,
    commit: str,
    bench: str | None,
    append_samples: bool,
    machine: str | None,
) -> None:
    cmd = resolve_asv_command(repo_root)
    ensure_asv_machine(cmd, machine)
    argv = build_asv_run_argv(
        commit=commit,
        bench=bench,
        append_samples=append_samples,
        machine=machine,
    )
    print("==> asv " + " ".join(argv), flush=True)
    try:
        subprocess.run([*cmd, *argv], cwd=repo_root, check=True)
    except subprocess.CalledProcessError as exc:
        # asv's own convention (asv/commands/run.py): exit 2 means "one or
        # more individual benchmarks errored", not that the run itself failed
        # to execute. Swallowing that here means a single flaky/OOM'd
        # parameter (observed on a small GPU under memory pressure) no longer
        # discards every other benchmark's results by skipping publish.
        if exc.returncode != 2:
            raise
        print(
            f"WARNING: asv run reported benchmark failures (exit {exc.returncode}); "
            "publishing the results collected so far anyway",
            flush=True,
        )


def publish_asv(*, repo_root: Path) -> None:
    cmd = resolve_asv_command(repo_root)
    print("==> asv publish", flush=True)
    subprocess.run([*cmd, *build_asv_publish_argv()], cwd=repo_root, check=True)


def load_latest_timings(*, repo_root: Path) -> list[TimingRow]:
    results_dir = Path(repo_root) / "benchmarks" / "results"
    for path in latest_asv_result_files(results_dir):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        rows = extract_timing_rows(payload)
        if rows:
            return rows
    return []


def write_reports(
    *,
    output_dir: Path,
    meta: Mapping[str, Any],
    checks: Sequence[CheckResult],
    timings: Sequence[TimingRow],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    asv_index = output_dir / "index.html"
    html_path = output_dir / REPORT_HTML_NAME
    json_path = output_dir / REPORT_JSON_NAME
    html_path.write_text(
        render_gpu_report_html(
            meta=meta,
            checks=checks,
            timings=timings,
            asv_html_href="index.html" if asv_index.is_file() else None,
        ),
        encoding="utf-8",
    )
    json_path.write_text(
        render_gpu_report_json(meta=meta, checks=checks, timings=timings),
        encoding="utf-8",
    )
    return html_path, json_path


def run_gpu_benchmark(args: Any) -> int:
    repo_root = Path(getattr(args, "repo_root", None) or REPO_ROOT)
    output_dir = Path(getattr(args, "output_dir", None) or (repo_root / "benchmarks" / "html"))
    allow_cpu = bool(getattr(args, "allow_cpu", False))
    skip_checks = bool(getattr(args, "skip_checks", False))
    checks_only = bool(getattr(args, "checks_only", False))
    force = bool(getattr(args, "force", False))
    publish = bool(getattr(args, "publish", True))

    prepare_bench_env(repo_root, allow_cpu=allow_cpu, apply=True)
    meta = collect_run_meta(repo_root=repo_root, allow_cpu=allow_cpu)
    if meta.get("dirty"):
        print(
            f"WARNING: working tree is dirty — results will be labelled "
            f"{str(meta.get('commit', ''))[:8]} but describe the tree as it is now.",
            flush=True,
        )

    if skip_checks:
        checks = [CheckResult("correctness", "skip", "skipped (--skip-checks)")]
    else:
        print("==> correctness checks (before timing)", flush=True)
        checks = run_correctness_checks(
            require_gpu=not allow_cpu,
            bench=getattr(args, "bench", None),
            only=getattr(args, "checks", None),
        )
        for check in checks:
            print(f"    [{check.status.upper():4}] {check.name}: {check.detail}", flush=True)

    ok = all_blocking_checks_passed(checks)
    timings: list[TimingRow] = []
    if ok or force:
        if checks_only:
            print("==> checks-only: skipping asv run", flush=True)
        else:
            machine = os.environ.get("ASV_MACHINE") or meta.get("machine")
            run_asv(
                repo_root=repo_root,
                commit=str(meta.get("commit") or git_commit(repo_root)),
                bench=getattr(args, "bench", None),
                append_samples=bool(getattr(args, "append_samples", False)),
                machine=machine,
            )
            if publish:
                publish_asv(repo_root=repo_root)
            timings = load_latest_timings(repo_root=repo_root)
    else:
        print("==> correctness failed; not starting asv run", flush=True)

    html_path, json_path = write_reports(
        output_dir=output_dir,
        meta=meta,
        checks=checks,
        timings=timings,
    )
    print(f"Report written to {html_path}", flush=True)
    print(f"JSON sidecar     {json_path}", flush=True)
    if (output_dir / "index.html").is_file():
        print(f"asv HTML         {output_dir / 'index.html'}", flush=True)
        print("View with:       uv run asv preview", flush=True)
    return 0 if (ok or force) else 1


def ensure_asv_machine(asv_cmd: Sequence[str], machine: str | None) -> None:
    marker = Path.home() / ".asv-machine.json"
    if marker.is_file():
        return
    cmd = [*asv_cmd, "machine", "--yes"]
    if machine:
        cmd.extend(["--machine", machine])
    print("==> registering this machine with asv", flush=True)
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


def np_asarray(value: Any):
    import numpy as np

    if hasattr(value, "__array__") or isinstance(value, (int, float, list, tuple)):
        return np.asarray(value)
    # JAX scalar / DeviceArray without going through jax at module import.
    try:
        return np.asarray(value)
    except Exception:
        return np.asarray([value])


def _all_finite(arr) -> bool:
    import numpy as np

    return bool(np.isfinite(arr).all())


def _abs_max(arr) -> float:
    import numpy as np

    if arr.size == 0:
        return 0.0
    return float(np.nanmax(np.abs(arr)))


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git(repo_root: Path, args: Sequence[str]) -> str:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return ""
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()


def _jax_device_label() -> str:
    try:
        import jax

        return str(jax.devices())
    except Exception:
        return "unknown"


def _import_asv_helpers() -> None:
    path = str(ASV_BENCH_DIR)
    if path not in sys.path:
        sys.path.insert(0, path)


def _spec_matches_bench(spec: CheckSpec, needle: str) -> bool:
    if spec.name.lower() in needle or needle in spec.name.lower():
        return True
    return any(tag in needle or needle in tag for tag in spec.tags)


def _run_named_check(name: str, fn: Callable[[], CheckResult]) -> CheckResult:
    try:
        return fn()
    except Exception as exc:
        message = str(exc)
        skipped = (
            type(exc).__name__ in {"BenchSkipped", "NotImplementedError"}
            or "unavailable" in message.lower()
        )
        status = "skip" if skipped else "fail"
        return CheckResult(name, status, message)


def _coerce_result_entry(
    entry: Any,
) -> tuple[list[float | None], list[list[str]], list[str]]:
    if isinstance(entry, (int, float)):
        return [float(entry)], [], []
    if isinstance(entry, list):
        # asv v2 sometimes stores [result_column, ...]
        first = entry[0] if entry else None
        if isinstance(first, list):
            return [_maybe_float(v) for v in first], [], []
        return [_maybe_float(v) for v in entry], [], []
    if not isinstance(entry, Mapping):
        return [], [], []
    result = entry.get("result", entry.get("value"))
    params = entry.get("params") or []
    names = [str(n) for n in (entry.get("param_names") or [])]
    if isinstance(result, list):
        values = [_maybe_float(v) for v in result]
    elif isinstance(result, (int, float)):
        values = [float(result)]
    else:
        values = []
    grid: list[list[str]] = []
    if isinstance(params, list) and params and isinstance(params[0], list):
        grid = [[str(x) for x in row] for row in params]
    return values, grid, names


def _params_for_index(
    param_grid: Sequence[Sequence[str]],
    param_names: Sequence[str],
    index: int,
) -> dict[str, str]:
    if not param_grid:
        return {}
    # asv stores one list per parameter axis.
    sizes = [len(axis) for axis in param_grid]
    if not sizes or any(s <= 0 for s in sizes):
        return {}
    coords: list[int] = []
    remaining = index
    for size in reversed(sizes):
        coords.append(remaining % size)
        remaining //= size
    coords.reverse()
    out: dict[str, str] = {}
    for axis, coord in enumerate(coords):
        label = param_names[axis] if axis < len(param_names) else f"p{axis}"
        if coord < len(param_grid[axis]):
            out[label] = str(param_grid[axis][coord])
    return out


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, Sequence) and value and isinstance(value[0], (int, float)):
        import statistics

        return float(statistics.median(value))
    return None


def _html_check_row(check: CheckResult) -> str:
    return (
        "<tr>"
        f"<td><code>{html.escape(check.name)}</code></td>"
        f'<td class="st-{html.escape(check.status)}">{html.escape(check.status.upper())}</td>'
        f"<td>{html.escape(check.detail)}</td>"
        "</tr>"
    )


def _html_timing_row(row: TimingRow) -> str:
    params = ", ".join(f"{k}={v}" for k, v in row.params.items()) or "—"
    return (
        "<tr>"
        f"<td><code>{html.escape(row.name)}</code></td>"
        f"<td>{html.escape(params)}</td>"
        f"<td>{html.escape(format_bench_value(row.name, row.value))}</td>"
        "</tr>"
    )
