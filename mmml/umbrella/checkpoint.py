"""Load PhysNet / SpookyNet params+model for umbrella sampling.

Supports the same artifact types as ``mmml neb`` / ASE calculators:
JSON (e.g. ``examples/m/kl.json``), Orbax ``epoch-*`` trees, and joint pickles.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_params_and_model(
    checkpoint: Path | str,
    *,
    natoms: int | None = None,
    prefer_ema: bool = True,
) -> tuple[Any, Any]:
    """Return ``(params, model)`` ready for ``model.apply``.

    Parameters
    ----------
    checkpoint
        Path to a ``.json`` portable checkpoint, Orbax training root / epoch, or
        joint pickle directory.
    natoms
        Optional override for ``max_padded_atoms`` (Orbax training restarts).
        Ignored when the JSON/config already defines padding; batched umbrella
        eval uses the real atom count via ``batch_segments``.
    prefer_ema
        Prefer ``ema_params`` when restoring Orbax epoch trees (via bundle).
    """
    from mmml.interfaces.calculators.checkpoint_loading import load_checkpoint_bundle
    from mmml.utils.model_checkpoint import (
        build_physnet_from_config,
        infer_trainable_zbl_config,
        normalize_physnet_config,
    )

    del prefer_ema  # bundle already picks ema when present in Orbax restores
    path = Path(checkpoint).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    bundle = load_checkpoint_bundle(path)
    saved = bundle.config
    if "physnet_config" in saved and (
        "dcmnet_config" in saved or "noneq_config" in saved
    ):
        raise ValueError(
            "umbrella sampling supports pure PhysNet/SpookyNet checkpoints only; "
            f"got joint checkpoint at {path}"
        )

    if "physnet_config" in saved:
        model_config = dict(saved["physnet_config"])
    else:
        model_config = dict(saved)

    model_config = infer_trainable_zbl_config(
        normalize_physnet_config(model_config), bundle.params
    )
    if natoms is not None:
        model_config = {
            **model_config,
            "max_padded_atoms": int(natoms),
            "natoms": int(natoms),
        }

    model = build_physnet_from_config(model_config)
    if natoms is not None:
        model.max_padded_atoms = int(natoms)
    return bundle.params, model
