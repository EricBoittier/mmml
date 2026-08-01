"""`--clip-global` must actually change the gradient clip.

The DES warm start (job 19360535) ran with the hardcoded default of 10.0, which
is loose enough that most steps pass through unclipped. The value was not
reachable from the CLI at all, so "try tighter clipping" was untestable.
"""

from __future__ import annotations

import pytest

jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("optax")

from mmml.models.physnetjax.physnetjax.training.optimizer import get_optimizer


def _applied_norm(clip_kwargs, raw_norm=1000.0):
    """Global norm of the update after one step on a huge gradient."""
    optimizer, _transform, _schedule, _kw = get_optimizer(
        learning_rate=1.0, optimizer="adam", **clip_kwargs
    )
    params = {"w": jnp.zeros((4,))}
    grads = {"w": jnp.full((4,), raw_norm / 2.0)}  # norm = raw_norm
    state = optimizer.init(params)
    updates, _ = optimizer.update(grads, state, params)
    return float(jnp.linalg.norm(updates["w"]))


def test_default_clip_is_ten():
    _o, _t, _s, kw = get_optimizer(learning_rate=1e-3)
    assert kw.get("clip_global") == 10.0


def test_explicit_clip_is_recorded_in_optimizer_kwargs():
    _o, _t, _s, kw = get_optimizer(learning_rate=1e-3, clip_global=1.0)
    assert kw.get("clip_global") == 1.0


def test_tighter_clip_produces_a_smaller_update():
    """Adam rescales, so compare the two clips against each other, not to 1.0."""
    loose = _applied_norm({"clip_global": 10.0})
    tight = _applied_norm({"clip_global": 0.01})
    assert tight < loose, f"tighter clip did not shrink the update ({tight} vs {loose})"


def test_clip_disabled_differs_from_clipped():
    off = _applied_norm({"clip_global": False})
    on = _applied_norm({"clip_global": 0.01})
    assert on != off


@pytest.mark.parametrize("bad", [0.0, -1.0, "1.0"])
def test_invalid_clip_rejected(bad):
    with pytest.raises((ValueError, TypeError)):
        get_optimizer(learning_rate=1e-3, clip_global=bad)


def test_train_model_forwards_clip_global():
    """The CLI value must reach get_optimizer, not be silently dropped."""
    import inspect

    from mmml.models.physnetjax.physnetjax.training import training

    assert "clip_global" in inspect.signature(training.train_model).parameters
    src = inspect.getsource(training.train_model)
    assert "clip_global" in src.split("get_optimizer")[1][:400]


def test_cli_exposes_clip_global():
    from mmml.cli.make.make_training import build_parser

    args = build_parser().parse_args(["--clip-global", "1.0"])
    assert args.clip_global == 1.0
    assert build_parser().parse_args([]).clip_global is None
