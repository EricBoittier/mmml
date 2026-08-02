"""Unit tests for training-run postconditions.

The motivating cases are the two real Q⁰ jobs where the SLURM state was the
wrong signal in opposite directions: 206089 reported COMPLETED while training a
partly-random model, and 206099 reported TIMEOUT after the fix it existed to
verify had already succeeded.
"""

from __future__ import annotations

import json

import pytest

from mmml.utils.training_run_check import (
    check_run,
    parse_epoch_summaries,
    parse_step_metrics,
    parse_warm_start,
)

# Verbatim from artifacts/spooky_q0_distill_smoke/slurm-206104.out (the good run).
GOOD_WARM_START = (
    "Warm-started from /mmhome/boittier/home/mmml/artifacts/spooky_so3lr_charges/"
    "epoch-0002: loaded 41 parameter leaves, initialized 0 new leaves, "
    "skipped 0 incompatible leaves"
)
# Verbatim from slurm-206089.out (the run that reported COMPLETED regardless).
BAD_WARM_START = (
    "Warm-started from /mmhome/boittier/home/mmml/artifacts/spooky_so3lr_charges/"
    "epoch-0002: loaded 36 parameter leaves, initialized 10 new leaves, "
    "skipped 2 incompatible leaves"
)
GOOD_STEP = (
    "epoch 0001 step 000040 [100.0% of 40] loss=116139 E_MAE=339.452 "
    "F_MAE=0.765333 D_MAE=1.02159 Q_MAE=1.47839 avg_N=120.0 "
    "distill(a=0.75) TE_MAE=343.667 TF_MAE=0.82878"
)
BAD_STEP = (
    "epoch 0001 step 000040 [100.0% of 40] loss=2.10468e+14 E_MAE=89741.9 "
    "F_MAE=5829.36 D_MAE=779518 Q_MAE=4.51998e+06 avg_N=120.0 "
    "distill(a=0.75) TE_MAE=89746 TF_MAE=5829.4"
)
GOOD_EPOCH = (
    "epoch 0001 done in 222.1s train_loss=120563 valid_loss=3366.87 "
    "valid_E_MAE=57.3673 valid_F_MAE=0.484697 valid_D_MAE=0.356156 "
    "valid_Q_MAE=0.519599 MBD_|E|=0 MBD_|F|=0"
)


def _workdir(tmp_path, *, checkpoint=True, distillation=True):
    (tmp_path / "run_config.json").write_text("{}")
    if checkpoint:
        ckpt = tmp_path / "epoch-0001"
        ckpt.mkdir()
        (ckpt / "_CHECKPOINT_METADATA").write_text("{}")
    if distillation:
        (tmp_path / "distillation.json").write_text(
            json.dumps(
                {
                    "teacher": {"sha256": "aea839624849" + "0" * 52},
                    "targets": {"energy": True, "forces": True},
                }
            )
        )
    return tmp_path


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_parse_warm_start_reads_the_real_line():
    assert parse_warm_start([GOOD_WARM_START]) == {
        "path": "/mmhome/boittier/home/mmml/artifacts/spooky_so3lr_charges/epoch-0002",
        "loaded": 41,
        "initialized": 0,
        "skipped": 0,
    }


def test_parse_warm_start_returns_none_without_one():
    assert parse_warm_start(["nothing here"]) is None


def test_parse_step_metrics_extracts_named_values():
    (entry,) = parse_step_metrics([GOOD_STEP])
    assert entry["epoch"] == 1
    assert entry["step"] == 40
    assert entry["F_MAE"] == pytest.approx(0.765333)
    assert entry["TF_MAE"] == pytest.approx(0.82878)


def test_parse_epoch_summaries_extracts_validation_metrics():
    (entry,) = parse_epoch_summaries([GOOD_EPOCH])
    assert entry["epoch"] == 1
    assert entry["wall_time_s"] == pytest.approx(222.1)
    assert entry["valid_F_MAE"] == pytest.approx(0.484697)


def test_parse_ignores_unrelated_lines():
    assert parse_step_metrics(["Planned run (provisional): 1 epochs x 40 steps"]) == []
    assert parse_epoch_summaries(["  auto-batch probing pad_atoms=120 ..."]) == []


# ---------------------------------------------------------------------------
# Verdicts on the two real cases
# ---------------------------------------------------------------------------


def test_completed_but_partial_warm_start_is_a_failure(tmp_path):
    """Job 206089: SLURM said COMPLETED; the model was partly random."""
    verdict = check_run(
        _workdir(tmp_path),
        [BAD_WARM_START, BAD_STEP],
        require_steps=40,
        require_distillation=True,
    )
    assert verdict.status == "FAIL"
    failed = {c.name for c in verdict.failures}
    assert "warm_start" in failed
    assert "initialized 10" in next(c.detail for c in verdict.checks if c.name == "warm_start")


def test_timeout_after_success_still_passes(tmp_path):
    """Job 206099: SLURM said TIMEOUT; the warm-start under test had succeeded.

    With no checkpoint written yet, a verdict scoped to the warm-start alone is
    still able to report success -- which is the whole point of not reading the
    job state.
    """
    verdict = check_run(
        _workdir(tmp_path, checkpoint=False, distillation=False),
        [GOOD_WARM_START],
        require_steps=0,
        require_checkpoint=False,
    )
    assert verdict.status == "PASS"
    assert verdict.observed["warm_start"]["loaded"] == 41


def test_a_run_that_never_trained_is_a_failure(tmp_path):
    """Also job 206099, judged as the smoke it was: zero steps is not success."""
    verdict = check_run(
        _workdir(tmp_path, checkpoint=False, distillation=False),
        [GOOD_WARM_START],
        require_steps=40,
        require_checkpoint=False,
    )
    assert verdict.status == "FAIL"
    assert "training_steps" in {c.name for c in verdict.failures}


def test_the_good_run_passes_every_postcondition(tmp_path):
    """Job 206104, the run that was actually sound."""
    verdict = check_run(
        _workdir(tmp_path),
        [GOOD_WARM_START, GOOD_STEP, GOOD_EPOCH],
        require_steps=40,
        require_distillation=True,
        max_force_mae=50.0,
    )
    assert verdict.status == "PASS", verdict.render()
    assert verdict.failures == []


def test_exploded_forces_fail_the_force_bound(tmp_path):
    verdict = check_run(
        _workdir(tmp_path),
        [GOOD_WARM_START, BAD_STEP],
        require_steps=40,
        max_force_mae=50.0,
    )
    assert verdict.status == "FAIL"
    assert "force_mae" in {c.name for c in verdict.failures}


def test_non_finite_metrics_fail(tmp_path):
    verdict = check_run(
        _workdir(tmp_path),
        [GOOD_WARM_START, "epoch 0001 step 000040 [100.0% of 40] loss=nan F_MAE=nan"],
        require_steps=40,
    )
    assert verdict.status == "FAIL"
    assert "metrics_finite" in {c.name for c in verdict.failures}


def test_missing_checkpoint_fails(tmp_path):
    verdict = check_run(
        _workdir(tmp_path, checkpoint=False),
        [GOOD_WARM_START, GOOD_STEP, GOOD_EPOCH],
        require_steps=40,
    )
    assert verdict.status == "FAIL"
    assert "checkpoint" in {c.name for c in verdict.failures}


def test_missing_distillation_provenance_fails_when_required(tmp_path):
    verdict = check_run(
        _workdir(tmp_path, distillation=False),
        [GOOD_WARM_START, GOOD_STEP, GOOD_EPOCH],
        require_steps=40,
        require_distillation=True,
    )
    assert verdict.status == "FAIL"
    assert "distillation_provenance" in {c.name for c in verdict.failures}


def test_partial_warm_start_can_be_allowed_explicitly(tmp_path):
    verdict = check_run(
        _workdir(tmp_path),
        [BAD_WARM_START, GOOD_STEP, GOOD_EPOCH],
        require_steps=40,
        require_full_warm_start=False,
    )
    assert verdict.status == "PASS", verdict.render()


def test_verdict_serializes_for_the_run_record(tmp_path):
    verdict = check_run(
        _workdir(tmp_path), [GOOD_WARM_START, GOOD_STEP, GOOD_EPOCH], require_steps=40
    )
    payload = json.loads(json.dumps(verdict.to_dict()))
    assert payload["status"] == "PASS"
    assert {"name", "ok", "detail"} <= set(payload["checks"][0])
    assert payload["observed"]["checkpoints"] == ["epoch-0001"]
    assert "PASS  run_config" in verdict.render()
