"""PhysNetJax checkpoint save helper."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from mmml.models.physnetjax.physnetjax.restart.restart import save_training_checkpoint


def test_save_training_checkpoint_uses_force_true(tmp_path: Path) -> None:
    checkpointer = MagicMock()
    ckpt = {"epoch": 1, "params": {"w": 1.0}}

    save_training_checkpoint(tmp_path / "epoch-1", ckpt, checkpointer=checkpointer)

    checkpointer.save.assert_called_once()
    args, kwargs = checkpointer.save.call_args
    assert args[0] == tmp_path / "epoch-1"
    assert args[1] == ckpt
    assert kwargs.get("force") is True


def test_save_training_checkpoint_removes_existing_when_force_unsupported(
    tmp_path: Path,
) -> None:
    ckp = tmp_path / "epoch-1"
    ckp.mkdir()
    (ckp / "partial").write_text("stale", encoding="utf-8")

    checkpointer = MagicMock()
    checkpointer.save.side_effect = [TypeError("no force"), None]

    save_training_checkpoint(ckp, {"epoch": 1}, checkpointer=checkpointer)

    assert not ckp.exists()
    assert checkpointer.save.call_count == 2
