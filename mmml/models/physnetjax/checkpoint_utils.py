"""Shared PhysNet checkpoint loading and bundled transfer-model listing."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from mmml.models.physnetjax.defaults import list_hf_physnet_models
from mmml.utils.model_checkpoint import load_model_checkpoint


def load_physnet_checkpoint(path: Path | str) -> tuple[dict[str, Any], Optional[dict[str, Any]]]:
    """Load PhysNet parameters and config from Orbax or JSON checkpoint.

    Returns ``(params, config)`` where *config* is EF architecture attrs when available.
    """
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"PhysNet checkpoint not found: {path}")

    if path.suffix == ".json" or (path.is_dir() and (path / "params.json").exists()):
        json_path = path if path.suffix == ".json" else path / "params.json"
        ckpt = load_model_checkpoint(json_path, use_orbax=False, load_config=True)
        params = ckpt["params"]
        config = ckpt.get("config") if isinstance(ckpt.get("config"), dict) else None
        return params, config

    epoch_dir = path
    if path.is_dir():
        epoch_dirs = sorted(
            path.glob("epoch-*/"),
            key=lambda d: int(d.name.split("-")[-1]) if d.name.startswith("epoch-") else -1,
        )
        epoch_dirs = [d for d in epoch_dirs if "tmp" not in d.name]
        if epoch_dirs:
            epoch_dir = epoch_dirs[-1]
    try:
        import orbax.checkpoint as ocp

        checkpointer = ocp.PyTreeCheckpointer()
        restored = checkpointer.restore(str(epoch_dir))
        if isinstance(restored, dict) and "ema_params" in restored:
            params = restored["ema_params"]
        elif isinstance(restored, dict) and "params" in restored:
            params = restored["params"]
        else:
            params = restored
        config = None
        if isinstance(restored, dict) and "model_attributes" in restored:
            config = dict(restored["model_attributes"])
        return params, config
    except Exception as e:
        raise RuntimeError(f"Failed to load PhysNet checkpoint from {path}: {e}") from e


def print_bundled_physnet_models(category: Optional[str] = None) -> None:
    """Print bundled PhysNet transfer-learning choices."""
    models = list_hf_physnet_models(category)
    title = "Bundled PhysNet transfer models"
    if category:
        title += f" ({category})"
    print(f"\n{title}:")
    for entry in models:
        config = entry.get("config", {})
        objectives = entry.get("metadata", {}).get("objectives", {})
        categories = ", ".join(entry.get("categories", []))
        print(
            f"  {entry['id']}: {entry.get('label', entry['file'])}\n"
            f"    file={entry['file']}\n"
            f"    categories={categories}\n"
            f"    charges={config.get('charges')}, electrostatics={config.get('include_electrostatics', False)}, "
            f"features={config.get('features')}, basis={config.get('num_basis_functions')}, "
            f"iterations={config.get('num_iterations')}, max_degree={config.get('max_degree')}\n"
            f"    valid_forces_mae={objectives.get('valid_forces_mae')}, "
            f"valid_energy_mae={objectives.get('valid_energy_mae')}, "
            f"valid_dipole_mae={objectives.get('valid_dipole_mae')}"
        )


# Architecture keys applied when match_checkpoint_architecture is enabled.
CHECKPOINT_ARCH_KEYS = (
    "features",
    "max_degree",
    "num_basis_functions",
    "num_iterations",
    "n_res",
    "cutoff",
    "max_atomic_number",
    "zbl",
    "efa",
    "use_pbc",
    "use_energy_bias",
    "charges",
    "total_charge",
    "include_electrostatics",
)


def apply_checkpoint_architecture(args, config: dict[str, Any]) -> None:
    """Override argparse Namespace model hyperparams from checkpoint config."""
    for key in CHECKPOINT_ARCH_KEYS:
        if key in config and hasattr(args, key):
            setattr(args, key, config[key])
