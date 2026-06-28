"""Checkpoint detection and loading for MMML inference calculators."""

from __future__ import annotations

import pickle
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import jax.numpy as jnp
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator

from mmml.utils.model_checkpoint import load_model_checkpoint, normalize_flax_params_for_apply

CheckpointFormat = Literal["pickle_joint", "json", "orbax"]


@dataclass(frozen=True)
class LoadedCheckpoint:
    """Model parameters and configuration loaded from disk."""

    params: Any
    config: dict[str, Any]
    source: Path
    format: CheckpointFormat


def validate_checkpoint_path(checkpoint: Path) -> None:
    """Raise ``FileNotFoundError`` when ``checkpoint`` is not a supported artifact."""
    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    detect_checkpoint_format(checkpoint)


def detect_checkpoint_format(checkpoint: Path) -> CheckpointFormat:
    """Classify a checkpoint path as pickle, JSON, or Orbax."""
    path = checkpoint.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    if path.is_file():
        if path.suffix == ".pkl":
            return "pickle_joint"
        if path.suffix == ".json":
            return "json"
        raise FileNotFoundError(
            f"Unsupported checkpoint file type: {path}. "
            "Expected .pkl (joint pickle), .json (portable params), or a checkpoint directory."
        )

    if (path / "params.json").exists() or (path / "model_config.json").exists():
        return "json"
    if (path / "manifest.ocdbt").exists():
        return "orbax"
    if (path / "params" / "manifest.ocdbt").exists():
        return "orbax"

    epoch_dirs = [
        child
        for child in path.glob("epoch-*")
        if child.is_dir() and "tmp" not in child.name and (child / "manifest.ocdbt").exists()
    ]
    if epoch_dirs:
        return "orbax"

    pickle_names = (
        "best_params.pkl",
        "final_params.pkl",
        "checkpoint.pkl",
        "params.pkl",
        "checkpoint_latest.pkl",
        "checkpoint_best.pkl",
    )
    if any((path / name).exists() for name in pickle_names):
        return "pickle_joint"

    if (path / "model_config.pkl").exists():
        return "pickle_joint"

    raise FileNotFoundError(
        f"No supported checkpoint artifacts found in {path}. "
        "Expected params.json, manifest.ocdbt (Orbax), or best_params.pkl (joint pickle)."
    )


def resolve_pickle_checkpoint_file(checkpoint: Path) -> Path:
    """Return a concrete ``.pkl`` file for joint checkpoints."""
    path = checkpoint.expanduser().resolve()
    if path.is_file():
        return path

    for name in (
        "best_params.pkl",
        "final_params.pkl",
        "checkpoint.pkl",
        "params.pkl",
        "checkpoint_latest.pkl",
        "checkpoint_best.pkl",
    ):
        candidate = path / name
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(f"No pickle checkpoint found in {path}")


def resolve_orbax_epoch_dir(checkpoint: Path) -> Path:
    """Resolve an Orbax epoch directory from a file or experiment root."""
    path = checkpoint.expanduser().resolve()
    if (path / "manifest.ocdbt").exists():
        return path

    epoch_dirs = sorted(
        [
            child
            for child in path.glob("epoch-*")
            if child.is_dir() and "tmp" not in child.name and (child / "manifest.ocdbt").exists()
        ],
        key=lambda child: int(child.name.split("-")[-1]) if child.name.startswith("epoch-") else -1,
    )
    if epoch_dirs:
        return epoch_dirs[-1]

    if (path / "params" / "manifest.ocdbt").exists():
        return path

    raise FileNotFoundError(f"No Orbax checkpoint found under {path}")


def _is_mapping(value: Any) -> bool:
    return isinstance(value, Mapping)


def _to_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if _is_mapping(value):
        return dict(value)
    raise TypeError(f"Expected mapping, got {type(value)}")


def _extract_params_tree(tree: Any) -> Any:
    if _is_mapping(tree) and "params" in tree:
        tree = tree["params"]
    while _is_mapping(tree) and "params" in tree and len(_to_dict(tree)) == 1:
        tree = tree["params"]
    if _is_mapping(tree):
        tree_dict = _to_dict(tree)
        model_param_roots = {
            "physnet",
            "dcmnet",
            "noneq_model",
            "charge_mixer",
            "coulomb_lambda",
        }
        if "params" not in tree_dict and any(key in tree_dict for key in model_param_roots):
            return {"params": tree}
    return tree


def _find_key_recursive(tree: Any, key: str) -> Any | None:
    if _is_mapping(tree):
        tree_dict = _to_dict(tree)
        if key in tree_dict:
            return tree_dict[key]
        for value in tree_dict.values():
            found = _find_key_recursive(value, key)
            if found is not None:
                return found
    return None


def _load_config_for_orbax_epoch(epoch_dir: Path) -> dict[str, Any]:
    for config_path in (
        epoch_dir / "model_config.json",
        epoch_dir.parent / "model_config.json",
        epoch_dir / "model_config.pkl",
        epoch_dir.parent / "model_config.pkl",
    ):
        if not config_path.exists():
            continue
        if config_path.suffix == ".json":
            import json

            with config_path.open(encoding="utf-8") as handle:
                loaded = json.load(handle)
        else:
            with config_path.open("rb") as handle:
                loaded = pickle.load(handle)
        if isinstance(loaded, dict):
            return loaded
    return {}


def _restore_orbax_epoch(epoch_dir: Path) -> tuple[Any, dict[str, Any]]:
    try:
        import orbax.checkpoint as ocp
    except ImportError as exc:
        raise ImportError("orbax-checkpoint is required to load Orbax checkpoints") from exc

    restored = ocp.PyTreeCheckpointer().restore(str(epoch_dir))
    if isinstance(restored, dict):
        if "ema_params" in restored:
            params = restored["ema_params"]
        elif "params" in restored:
            params = restored["params"]
        else:
            params = restored
        config = dict(restored.get("model_attributes") or {})
    else:
        params = restored
        config = {}

    if not config:
        config = _load_config_for_orbax_epoch(epoch_dir)

    return normalize_flax_params_for_apply(params), config


def load_checkpoint_bundle(checkpoint: Path) -> LoadedCheckpoint:
    """Load parameters and configuration from pickle, JSON, or Orbax artifacts."""
    path = checkpoint.expanduser().resolve()
    fmt = detect_checkpoint_format(path)

    if fmt == "pickle_joint":
        pickle_path = resolve_pickle_checkpoint_file(path)
        with pickle_path.open("rb") as handle:
            checkpoint_data = pickle.load(handle)
        params = _extract_params_tree(checkpoint_data)

        config_path = pickle_path.parent / "model_config.pkl"
        if not config_path.exists():
            raise FileNotFoundError(f"Model config not found: {config_path}")
        with config_path.open("rb") as handle:
            saved_config = pickle.load(handle)
        if not isinstance(saved_config, dict):
            raise TypeError(f"Expected dict config in {config_path}")
        return LoadedCheckpoint(
            params=params,
            config=saved_config,
            source=pickle_path,
            format=fmt,
        )

    if fmt == "json":
        loaded = load_model_checkpoint(path, use_orbax=False, load_params=True, load_config=True)
        params = loaded.get("params")
        config = loaded.get("config") or {}
        if params is None:
            raise FileNotFoundError(f"params missing from JSON checkpoint at {path}")
        if not isinstance(config, dict):
            config = {}
        return LoadedCheckpoint(
            params=normalize_flax_params_for_apply(params),
            config=config,
            source=path,
            format=fmt,
        )

    orbax_root = resolve_orbax_epoch_dir(path)
    if (orbax_root / "params" / "manifest.ocdbt").exists() or (
        path / "params" / "manifest.ocdbt"
    ).exists():
        loaded = load_model_checkpoint(path, use_orbax=True, load_params=True, load_config=True)
        params = loaded.get("params")
        config = loaded.get("config") or {}
        if params is None:
            raise FileNotFoundError(f"params missing from Orbax checkpoint at {path}")
        if not isinstance(config, dict):
            config = {}
        return LoadedCheckpoint(
            params=normalize_flax_params_for_apply(params),
            config=config,
            source=orbax_root,
            format=fmt,
        )

    params, config = _restore_orbax_epoch(orbax_root)
    return LoadedCheckpoint(
        params=params,
        config=config,
        source=orbax_root,
        format=fmt,
    )


def _import_joint_model_classes() -> tuple[Any, Any]:
    try:
        from mmml.cli.misc.train_joint import (
            JointPhysNetDCMNet,
            JointPhysNetNonEquivariant,
        )
    except Exception:
        import importlib.util
        import sys

        repo_root = Path(__file__).resolve().parents[3]
        trainer_candidates = [
            repo_root / "examples" / "other" / "co2" / "dcmnet_physnet_train" / "trainer.py",
            repo_root / "examples" / "co2" / "dcmnet_physnet_train" / "trainer.py",
        ]
        trainer_path = next((candidate for candidate in trainer_candidates if candidate.exists()), None)
        if trainer_path is None:
            raise FileNotFoundError(
                "Could not locate trainer module for joint PhysNet checkpoints."
            )
        spec = importlib.util.spec_from_file_location("dcmnet_trainer", trainer_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load trainer module from {trainer_path}")
        trainer = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = trainer
        spec.loader.exec_module(trainer)
        JointPhysNetDCMNet = trainer.JointPhysNetDCMNet  # type: ignore[attr-defined]
        JointPhysNetNonEquivariant = trainer.JointPhysNetNonEquivariant  # type: ignore[attr-defined]
    return JointPhysNetDCMNet, JointPhysNetNonEquivariant


def _build_joint_model(
    saved_config: dict[str, Any],
    params: Any,
    *,
    is_noneq: bool,
    disable_physnet_point_coulomb: bool,
) -> tuple[Any, Any, float]:
    JointPhysNetDCMNet, JointPhysNetNonEquivariant = _import_joint_model_classes()

    physnet_config = dict(saved_config["physnet_config"])
    if disable_physnet_point_coulomb:
        physnet_config["include_electrostatics"] = False
    mix_coulomb_energy = saved_config.get("mix_coulomb_energy", False)

    use_noneq = is_noneq or ("noneq_config" in saved_config and "dcmnet_config" not in saved_config)
    if use_noneq:
        if "noneq_config" not in saved_config:
            raise ValueError("Checkpoint config is missing noneq_config for non-equivariant model.")
        model = JointPhysNetNonEquivariant(
            physnet_config=physnet_config,
            noneq_config=saved_config["noneq_config"],
            mix_coulomb_energy=mix_coulomb_energy,
        )
    else:
        if "dcmnet_config" not in saved_config:
            raise ValueError("Checkpoint config is missing dcmnet_config for equivariant model.")
        model = JointPhysNetDCMNet(
            physnet_config=physnet_config,
            dcmnet_config=saved_config["dcmnet_config"],
            mix_coulomb_energy=mix_coulomb_energy,
        )

    if mix_coulomb_energy and _is_mapping(params):
        params_dict = _to_dict(params)
        if "params" in params_dict and _is_mapping(params_dict["params"]):
            inner = _to_dict(params_dict["params"])
            if "coulomb_lambda" not in inner:
                recovered_lambda = _find_key_recursive(params, "coulomb_lambda")
                if recovered_lambda is not None:
                    inner["coulomb_lambda"] = recovered_lambda
                else:
                    inner["coulomb_lambda"] = jnp.array([1.0], dtype=jnp.float32)
                params = {"params": inner}

    cutoff = float(physnet_config.get("cutoff", 6.0))
    return model, params, cutoff


def _build_physnet_ef_calculator(
    saved_config: dict[str, Any],
    params: Any,
    *,
    cutoff: float | None,
) -> Calculator:
    from mmml.utils.model_checkpoint import normalize_physnet_config, physnet_constructor_kwargs
    from mmml.models.physnetjax.physnetjax.calc.helper_mlp import get_ase_calc
    from mmml.models.physnetjax.physnetjax.models.model import PhysNet
    from mmml.models.physnetjax.physnetjax.models.spooky_model import SpookyPhysNet

    if "physnet_config" in saved_config:
        model_config = dict(saved_config["physnet_config"])
    else:
        model_config = dict(saved_config)
    model_config = normalize_physnet_config(model_config)

    is_spooky = str(saved_config.get("model_type", model_config.get("model_type", ""))).lower() == "spooky"
    model_cls = SpookyPhysNet if is_spooky else PhysNet
    filtered_config = physnet_constructor_kwargs(model_config, model_cls)
    if "max_padded_atoms" not in filtered_config:
        raise ValueError(
            "PhysNet checkpoint config is missing required field "
            "'max_padded_atoms' (or legacy 'natoms')."
        )

    model = model_cls(**filtered_config)
    model.max_padded_atoms = int(filtered_config["max_padded_atoms"])

    natoms = int(model.max_padded_atoms)
    template = Atoms(numbers=[1] * natoms, positions=np.zeros((natoms, 3), dtype=float))
    effective_cutoff = cutoff if cutoff is not None else float(filtered_config.get("cutoff", 6.0))
    _ = effective_cutoff
    return get_ase_calc(params, model, template)


def create_calculator_from_checkpoint(
    checkpoint_path: str | Path,
    is_noneq: bool = False,
    cutoff: float | None = None,
    use_dcmnet_dipole: bool = False,
    disable_physnet_point_coulomb: bool = False,
) -> Calculator:
    """Load a trained MMML model and return an ASE calculator."""
    bundle = load_checkpoint_bundle(Path(checkpoint_path))

    if "physnet_config" in bundle.config and (
        "dcmnet_config" in bundle.config or "noneq_config" in bundle.config
    ):
        model, params, default_cutoff = _build_joint_model(
            bundle.config,
            bundle.params,
            is_noneq=is_noneq,
            disable_physnet_point_coulomb=disable_physnet_point_coulomb,
        )
        effective_cutoff = cutoff if cutoff is not None else default_cutoff
        from mmml.interfaces.calculators.simple_inference import SimpleInferenceCalculator

        return SimpleInferenceCalculator(
            model=model,
            params=params,
            cutoff=effective_cutoff,
            use_dcmnet_dipole=use_dcmnet_dipole,
        )

    return _build_physnet_ef_calculator(bundle.config, bundle.params, cutoff=cutoff)
