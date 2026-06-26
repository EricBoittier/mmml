"""JAX-MD PBC neighbor-list defaults (NVE stability + throughput)."""

from __future__ import annotations

import jax.numpy as jnp

from mmml.cli.run.jaxmd_runner import _nl_update_positions, resolve_jaxmd_steps_per_loop_call
from mmml.interfaces.pycharmmInterface.mm_energy_forces import (
    DEFAULT_JAX_MD_SKIN_DISTANCE_A,
    format_mm_pair_update_stats_summary,
    neighbor_pair_cache_should_reuse,
)
import numpy as np


PBC_RECORDING_BLOCK_STEPS = 800
PBC_BOX_A = np.array([40.0, 40.0, 40.0])


def resolve_pbc_loop_steps(jax_md_update_interval: int | None) -> int:
    """Resolve the production PBC case: dynamic MM pairs inside a fixed recording block."""
    return resolve_jaxmd_steps_per_loop_call(
        steps_per_recording=PBC_RECORDING_BLOCK_STEPS,
        use_pbc=True,
        has_update_fn=True,
        jax_md_update_interval=jax_md_update_interval,
    )


def test_default_skin_is_quarter_angstrom():
    assert DEFAULT_JAX_MD_SKIN_DISTANCE_A == 0.25


def test_nl_update_positions_preserves_jax_arrays(monkeypatch):
    monkeypatch.delenv("MMML_MM_NL_FORCE_HOST", raising=False)
    positions = jnp.zeros((2, 3))
    assert _nl_update_positions(positions) is positions


def test_nl_update_positions_force_host_escape_hatch(monkeypatch):
    monkeypatch.setenv("MMML_MM_NL_FORCE_HOST", "1")
    positions = jnp.zeros((2, 3))
    out = _nl_update_positions(positions)
    assert isinstance(out, np.ndarray)


def test_skin_zero_interval_one_never_reuses():
    R = np.zeros((4, 3), dtype=np.float64)
    assert not neighbor_pair_cache_should_reuse(
        calls=1,
        interval=1,
        skin=0.0,
        R=R,
        last_R=R.copy(),
        box=PBC_BOX_A,
        last_box=PBC_BOX_A.copy(),
        have_cache=True,
    )


def test_default_skin_interval_one_reuses_small_step():
    R0 = np.zeros((4, 3), dtype=np.float64)
    R1 = R0.copy()
    R1[0, 0] = 0.1
    assert neighbor_pair_cache_should_reuse(
        calls=1,
        interval=1,
        skin=DEFAULT_JAX_MD_SKIN_DISTANCE_A,
        R=R1,
        last_R=R0,
        box=PBC_BOX_A,
        last_box=PBC_BOX_A.copy(),
        have_cache=True,
    )


def test_resolve_steps_per_loop_call_defaults_to_one_for_pbc_with_update_fn():
    assert resolve_pbc_loop_steps(jax_md_update_interval=None) == 1


def test_resolve_steps_per_loop_call_honors_pbc_update_interval():
    assert resolve_pbc_loop_steps(jax_md_update_interval=10) == 10


def test_resolve_steps_per_loop_call_uses_divisor_for_recording_blocks():
    assert resolve_pbc_loop_steps(jax_md_update_interval=30) == 25


def test_format_mm_pair_update_stats_summary():
    line = format_mm_pair_update_stats_summary(
        {"calls": 1000, "reused": 950, "updates": 50, "reallocs": 0, "fallbacks": 0}
    )
    assert "950/1000 reused (95.0%)" in line
    assert "reallocs=0" in line


def test_jaxmd_and_ase_cli_defaults_use_interval_one_conservative_skin():
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    jaxmd_src = (root / "mmml/cli/run/md_pbc_suite/jaxmd.py").read_text(encoding="utf-8")
    ase_src = (root / "mmml/cli/run/md_pbc_suite/ase.py").read_text(encoding="utf-8")
    assert "DEFAULT_JAX_MD_SKIN_DISTANCE_A" in jaxmd_src
    assert "default=1" in jaxmd_src.split("jax-md-update-interval")[1][:120]
    assert "DEFAULT_JAX_MD_SKIN_DISTANCE_A" in ase_src
    assert "default=1," in ase_src.split('"--jax-md-update-interval"')[1][:200]
    assert "default=1.75" in ase_src.split('"--jax-md-capacity-multiplier"')[1][:200]
