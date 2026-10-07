"""Load PhysNet / SpookyNet / KerNN params+model for umbrella sampling.

Supports the same artifact types as ``mmml neb`` / ASE calculators:
JSON (e.g. ``examples/m/kl.json``), Orbax ``epoch-*`` trees, joint pickles,
and KerNN JSON checkpoints (``model_type: kernnn``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_params_and_model(
    checkpoint: Path | str,
    *,
    natoms: int | None = None,
    prefer_ema: bool = True,
    model: str | None = None,
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
        eval uses the real atom count via ``batch_segments``. For KerNN, sets
        the expected atom count (default 4).
    prefer_ema
        Prefer ``ema_params`` when restoring Orbax epoch trees (via bundle).
    model
        Optional backend name (``physnet`` / ``kernnn``). Auto-detects KerNN JSON.
    """
    from mmml.models.kernnn import (
        KerNNApplyAdapter,
        is_kernnn_checkpoint,
        load_checkpoint,
    )

    path = Path(checkpoint).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    model_name = (model or "").strip().lower()
    if model_name == "kernnn" or (not model_name and is_kernnn_checkpoint(path)):
        ckpt = path
        if ckpt.is_dir():
            for name in ("best.json", "params.json"):
                cand = ckpt / name
                if cand.is_file():
                    ckpt = cand
                    break
        params, config, stats, _ = load_checkpoint(ckpt)
        n = int(natoms) if natoms is not None else 4
        adapter = KerNNApplyAdapter(stats=stats, config=config, n_atoms=n)
        return params, adapter

    from mmml.interfaces.calculators.checkpoint_loading import load_checkpoint_bundle
    from mmml.utils.model_checkpoint import (
        build_physnet_from_config,
        infer_trainable_zbl_config,
        normalize_physnet_config,
    )

    del prefer_ema  # bundle already picks ema when present in Orbax restores
    bundle = load_checkpoint_bundle(path)
    saved = bundle.config
    if "physnet_config" in saved and (
        "dcmnet_config" in saved or "noneq_config" in saved
    ):
        raise ValueError(
            "umbrella sampling supports pure PhysNet/SpookyNet/KerNN checkpoints only; "
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

    phys_model = build_physnet_from_config(model_config)
    if natoms is not None:
        phys_model.max_padded_atoms = int(natoms)
    return bundle.params, phys_model
