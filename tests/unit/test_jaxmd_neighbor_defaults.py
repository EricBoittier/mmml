"""JAX-MD PBC neighbor-list defaults (NVE stability)."""

from __future__ import annotations

from mmml.cli.run.jaxmd_runner import resolve_jaxmd_steps_per_loop_call


def test_resolve_steps_per_loop_call_is_one_for_pbc_with_update_fn():
    assert (
        resolve_jaxmd_steps_per_loop_call(
            steps_per_recording=800,
            use_pbc=True,
            has_update_fn=True,
            jax_md_update_interval=10,
        )
        == 1
    )


def test_resolve_steps_per_loop_call_allows_chunks_without_pbc():
    steps = resolve_jaxmd_steps_per_loop_call(
        steps_per_recording=1000,
        use_pbc=False,
        has_update_fn=False,
        jax_md_update_interval=100,
    )
    assert steps == 100


def test_jaxmd_and_ase_cli_defaults_use_interval_one_skin_zero():
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    jaxmd_src = (root / "mmml/cli/run/md_pbc_suite/jaxmd.py").read_text(encoding="utf-8")
    ase_src = (root / "mmml/cli/run/md_pbc_suite/ase.py").read_text(encoding="utf-8")
    assert '"--jax-md-update-interval"' in jaxmd_src or '--jax-md-update-interval", type=int, default=1' in jaxmd_src
    assert "default=1" in jaxmd_src.split("jax-md-update-interval")[1][:120]
    assert '"--jax-md-skin-distance"' in jaxmd_src
    assert "default=0.0" in jaxmd_src.split("jax-md-skin-distance")[1][:120]
    assert '"--jax-md-update-interval"' in ase_src
    assert "default=1," in ase_src.split('"--jax-md-update-interval"')[1][:200]
    assert '"--jax-md-skin-distance"' in ase_src
    assert "default=0.0" in ase_src.split('"--jax-md-skin-distance"')[1][:300]

