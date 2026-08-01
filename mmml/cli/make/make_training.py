"""
Sets up a training set for the ML model.

Args: 
    -d, --data the npz file to use for training
    -t, --tag the name of the run
    -m, --model the model to use for training, as .inp file
    -n, --n_train the number of training samples to use
    -v, --n_valid the number of validation samples to use
    -s, --seed the seed for the random number generator
    -b, --batch_size the batch size for training
    -e, --num_epochs the number of epochs to train for
    -l, --learning_rate the learning rate for training
    -w, --energy_weight the weight for the energy loss
    -o, --objective the objective function to optimize
    -r, --restart the restart file to use for training
    -c, --ckpt_dir the directory to save the checkpoints to
"""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from datetime import datetime
import jax

import yaml

# from mmml.models.physnetjax.physnetjax.models import model as model
from mmml.models.physnetjax.defaults import JOINT_TRAINING_CATEGORY, resolve_hf_physnet_model
from mmml.models.physnetjax.checkpoint_utils import (
    apply_checkpoint_architecture,
    load_physnet_checkpoint,
    print_bundled_physnet_models,
)
from mmml.models.physnetjax.physnetjax.models.model import EF
from mmml.models.physnetjax.physnetjax.training.training import train_model
from mmml.models.physnetjax.physnetjax.data.data import prepare_datasets
from mmml.utils.model_checkpoint import normalize_flax_params_for_apply
# from mmml.models.physnetjax.physnetjax.data.batches import prepare_batches_jit

import numpy as np

# YAML / config aliases (e.g. train -> data, output -> ckpt_dir)
CONFIG_ALIASES: Dict[str, str] = {
    "train": "data",
    "train_file": "data",
    "valid": "valid_data",
    "valid_file": "valid_data",
    "output": "ckpt_dir",
    "output_dir": "ckpt_dir",
    "max_epochs": "num_epochs",
    "epochs": "num_epochs",
    "model_file": "model",
    "restart_file": "restart",
}


def _normalize_config_key(key: str) -> str:
    return CONFIG_ALIASES.get(key, key).replace("-", "_")


def _parse_dict_option(val: Any) -> Optional[Dict[str, Any]]:
    if val is None:
        return None
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        val_stripped = val.strip()
        if not val_stripped:
            return None
        if val_stripped.startswith("{"):
            try:
                return json.loads(val_stripped)
            except Exception as e:
                raise ValueError(f"Failed to parse JSON dict from: {val}. Error: {e}")
        path = Path(val_stripped)
        if path.is_file():
            with path.open() as f:
                if path.suffix in (".yaml", ".yml"):
                    return yaml.safe_load(f)
                else:
                    return json.load(f)
        raise ValueError(f"Expected a JSON string or file path for dictionary option, got: {val}")
    return val


def _parse_list_option(val: Any) -> Optional[Sequence[Any]]:
    if val is None:
        return None
    if isinstance(val, (list, tuple)):
        return list(val)
    if isinstance(val, str):
        val_stripped = val.strip()
        if not val_stripped:
            return None
        if val_stripped.startswith("[") or val_stripped.startswith("("):
            try:
                return json.loads(val_stripped)
            except Exception:
                pass
        if "," in val_stripped:
            return [x.strip() for x in val_stripped.split(",")]
        return val_stripped.split()
    return val


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a PhysNetJAX EF model from NPZ data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  mmml physnet-train \\
      --data output/energies_forces_dipoles_train.npz \\
      --ckpt-dir ./ckpts/ama_mp2 \\
      --tag ama_mp2 \\
      --n-train 24000 --n-valid 3000 \\
      --batch-size 32 --num-epochs 2000 \\
      --max-atomic-number 35

  mmml physnet-train --config train.yaml

YAML keys match CLI flags (with optional aliases: train, output, max_epochs).
See mmml/cli/misc/physnet_train.example.yaml for a template.
See mmml/cli/misc/physnet_train_transfer.example.yaml for transfer learning / distillation.
See examples/hybrid_mm_charges/ for hybrid-mm + mm_charge_mode (fixed/latent/fixed_plus_latent).
        """,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML file with training options (CLI flags override file values)",
    )
    parser.add_argument("--data", type=str, default=None, help="Training NPZ file")
    parser.add_argument(
        "--valid-data",
        "--valid_data",
        type=str,
        default=None,
        dest="valid_data",
        help="Optional validation NPZ (use full files; no random re-split)",
    )
    parser.add_argument(
        "--ckpt-dir",
        "--ckpt_dir",
        type=str,
        default=None,
        dest="ckpt_dir",
        help="Checkpoint directory (absolute path used for Orbax)",
    )

    parser.add_argument("--tag", type=str, default="run", help="Run name for checkpoints")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional model JSON to load instead of creating a new EF model",
    )
    parser.add_argument(
        "--n-train",
        "--n_train",
        type=int,
        default=None,
        dest="n_train",
        help=(
            "Training samples to split from --data (default: 1000). Omit when "
            "--valid-data is set: the full files are used."
        ),
    )
    parser.add_argument(
        "--n-valid",
        "--n_valid",
        type=int,
        default=None,
        dest="n_valid",
        help=(
            "Validation samples to split from --data (default: 100). Omit when "
            "--valid-data is set: the full files are used."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", "--batch_size", type=int, default=1, dest="batch_size")
    parser.add_argument("--num-epochs", "--num_epochs", type=int, default=100, dest="num_epochs")
    parser.add_argument(
        "--learning-rate", "--learning_rate", type=float, default=0.001, dest="learning_rate"
    )
    parser.add_argument(
        "--energy-weight", "--energy_weight", type=float, default=1.0, dest="energy_weight"
    )
    parser.add_argument(
        "--forces-weight", "--forces_weight", type=float, default=52.91, dest="forces_weight"
    )
    parser.add_argument(
        "--dipole-weight", "--dipole_weight", type=float, default=27.21, dest="dipole_weight"
    )
    parser.add_argument(
        "--charges-weight", "--charges_weight", type=float, default=14.39, dest="charges_weight"
    )
    parser.add_argument("--objective", type=str, default="valid_loss")
    parser.add_argument(
        "--mm-charge-mode",
        "--mm_charge_mode",
        type=str,
        default=None,
        dest="mm_charge_mode",
        choices=["fixed", "q0", "latent", "q1", "fixed_plus_latent"],
        help=(
            "Hybrid MM Coulomb charges: fixed (q_CGenFF, default), q0 / Q⁰ "
            "(neutralize unperturbed monomer q_ML; train+liquid), latent / q1 / Q¹ "
            "(neutralize AB-perturbed q_ML; dimer-only), or fixed_plus_latent "
            "(q_CGenFF + neutralize(Q¹)). Modes q0/latent/q1/fixed_plus_latent "
            "require --charges. latent/q1/fixed_plus_latent are dimer-only. "
            "See docs/hybrid-mm-charges.md."
        ),
    )
    parser.add_argument(
        "--mm-charge-correction",
        "--mm_charge_correction",
        action="store_true",
        dest="mm_charge_correction",
        help=(
            "Alias for --mm-charge-mode fixed_plus_latent: use the model's "
            "predicted charges as a CORRECTION to fixed CGenFF charges in MM "
            "electrostatics (q_eff = q_cgenff + dq_ML, projected net-zero per "
            "monomer). Requires --charges."
        ),
    )
    parser.add_argument(
        "--hybrid-mm",
        "--hybrid_mm",
        action="store_true",
        dest="hybrid_mm",
        help=(
            "Train on the hybrid ML/MM total the MD calculator evaluates: "
            "E = (1-s)*(E_A+E_B) + s*E_AB + E_MM, where the taper s(r_com) "
            "applies to the dimer interaction and E_MM is switched CGenFF LJ + "
            "electrostatics. Requires a dataset carrying cgenff_type_idx, "
            "mol_id, cgenff_charge and the cgenff_master_* LJ tables. The "
            "handoff is controlled by --ml-switch-width/--mm-switch-on/"
            "--mm-switch-width (same flags and defaults as the MD side)."
        ),
    )
    # ml_switch_width / mm_switch_on / mm_switch_width / --no-complementary-handoff
    # come from the same helper the MD side uses, so the flags, defaults and
    # semantics cannot drift between training and deployment.
    from mmml.interfaces.pycharmmInterface.cutoffs import add_handoff_cutoff_args

    add_handoff_cutoff_args(parser)
    parser.add_argument(
        "--lr-solver",
        "--lr_solver",
        type=str,
        default="mic",
        dest="lr_solver",
        choices=["mic", "nvalchemiops_pme", "ewald"],
        help=(
            "Hybrid-MM long-range Coulomb for training (default: mic). "
            "mic: switched CGenFF LJ+Coulomb pairs. nvalchemiops_pme: full-box "
            "many-to-many PME on fixed CGenFF charges (no exclusions / no "
            "intra subtract; LJ omitted; requires --pme-box-length and "
            "mmml[nvalchemiops-pme]). ewald: same full-box/no-exclusion "
            "contract as nvalchemiops_pme, pure JAX (no external PME library, "
            "no CUDA requirement); requires --pme-box-length. Matches fast MD "
            "periodic_external."
        ),
    )
    parser.add_argument(
        "--pme-box-length",
        "--pme_box_length",
        type=float,
        default=None,
        dest="pme_box_length",
        help=(
            "Cubic box length (Å) for --lr-solver nvalchemiops_pme|ewald "
            "(required for those solvers)."
        ),
    )
    parser.add_argument(
        "--pme-accuracy",
        "--pme_accuracy",
        type=float,
        default=1e-6,
        dest="pme_accuracy",
        help="nvalchemiops_pme/ewald PME accuracy target (default: 1e-6).",
    )
    parser.add_argument(
        "--mm-include-lj",
        "--mm_include_lj",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="mm_include_lj",
        help=(
            "Include CGenFF LJ in hybrid E_MM (default: on for mic). "
            "Forced off when --lr-solver nvalchemiops_pme or ewald."
        ),
    )
    parser.add_argument(
        "--learn-mm-lj-scales",
        "--learn_mm_lj_scales",
        action=argparse.BooleanOptionalAction,
        default=False,
        dest="learn_mm_lj_scales",
        help=(
            "Learn per-CGenFF-type multiplicative scales on master σ and ε "
            "(separate arrays, init 1.0). Only affects mic hybrid E_MM LJ; "
            "ignored when LJ is forced off (ewald / nvalchemiops_pme). "
            "Scales are saved in hybrid_mm.json for MD ep_scale/sig_scale."
        ),
    )
    parser.add_argument(
        "--ema-decay",
        "--ema_decay",
        type=float,
        default=0.999,
        dest="ema_decay",
        help=(
            "Decay for the parameter EMA (default: 0.999). Validation, "
            "checkpointing and restart all use the EMA weights. Set 0 to "
            "disable EMA (saved weights then track the raw parameters)."
        ),
    )
    parser.add_argument("--restart", type=str, default=None, help="Checkpoint path to restart from")

    parser.add_argument(
        "--num-atoms",
        "--num_atoms",
        type=int,
        default=None,
        dest="num_atoms",
        help="Atoms per structure (auto-detected from N/R if omitted)",
    )
    parser.add_argument("--features", type=int, default=64)
    parser.add_argument("--max-degree", "--max_degree", type=int, default=0, dest="max_degree")
    parser.add_argument(
        "--num-basis-functions",
        "--num_basis_functions",
        type=int,
        default=32,
        dest="num_basis_functions",
    )
    parser.add_argument(
        "--num-iterations",
        "--num_iterations",
        type=int,
        default=2,
        dest="num_iterations",
    )
    parser.add_argument(
        "--n-res",
        "--n_res",
        type=int,
        default=2,
        dest="n_res",
        help="Number of refinement residual blocks (not CHARMM residues)",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=6.0,
        help=(
            "PhysNet radial basis cutoff (Angstrom, atom-pair distance). Must be "
            ">= --mm-switch-on: the ML has to be able to see the interaction out "
            "to wherever MM takes over, or it is silently truncated inside the "
            "handoff. Baked into the checkpoint; MD reads it back automatically."
        ),
    )
    # Electrostatics switching distances. These are model fields but were never
    # passed through from the training config, so every run silently took the
    # dataclass defaults regardless of --cutoff. That is how a model trained at
    # cutoff 8.0 ended up switching its Coulomb tail off at 8.0 as well, leaving
    # a dissociating ion pair with no long-range interaction at all -- and 12 %
    # of the Menshutkin training set beyond that radius, structurally unfittable.
    #
    # They are deliberately NOT derived from --cutoff. The radial cutoff decides
    # which atoms exchange features; the electrostatics range decides where the
    # Coulomb tail is truncated, and for a reaction that separates charges the
    # latter should extend well beyond the former. Defaults are unchanged, so
    # existing configs reproduce their previous behaviour exactly.
    parser.add_argument(
        "--switch-start", "--switch_start", type=float, default=1.0,
        help="Short-range Coulomb switch start (Angstrom)",
    )
    parser.add_argument(
        "--switch-end", "--switch_end", type=float, default=10.0,
        help="Short-range Coulomb switch end (Angstrom)",
    )
    parser.add_argument(
        "--electrostatics-off-start", "--electrostatics_off_start",
        type=float, default=8.0,
        help=(
            "Distance at which the electrostatic term begins switching off "
            "(Angstrom). Set this beyond the largest separation the reaction "
            "reaches, or the Coulomb tail vanishes there."
        ),
    )
    parser.add_argument(
        "--electrostatics-off-end", "--electrostatics_off_end",
        type=float, default=10.0,
        help="Distance at which the electrostatic term is fully off (Angstrom)",
    )
    parser.add_argument(
        "--max-atomic-number",
        "--max_atomic_number",
        type=int,
        default=28,
        dest="max_atomic_number",
    )
    parser.add_argument(
        "--zbl",
        action="store_true",
        dest="zbl",
        help="Enable ZBL repulsion in EF model",
    )
    parser.add_argument(
        "--no-zbl",
        action="store_false",
        dest="zbl",
        help="Disable ZBL repulsion in EF model",
    )
    parser.set_defaults(zbl=False)
    parser.add_argument(
        "--trainable-zbl",
        action="store_true",
        default=False,
        help="Opt in to optimizing ZBL screening parameters; fixed ZBL is the default.",
    )

    parser.add_argument(
        "--use-pbc",
        "--use_pbc",
        action="store_true",
        default=False,
        dest="use_pbc",
        help="Use periodic boundary conditions",
    )
    parser.add_argument(
        "--no-pbc",
        action="store_false",
        dest="use_pbc",
        help="Disable periodic boundary conditions",
    )

    parser.add_argument(
        "--no-energy-bias",
        action="store_false",
        dest="use_energy_bias",
        help="Disable per-element energy bias in the model",
    )
    parser.set_defaults(use_energy_bias=True)

    # Optimizer & Schedule Options
    parser.add_argument(
        "--optimizer",
        type=str,
        default=None,
        help="Optimizer string (e.g. 'adam', 'adamw', 'amsgrad')",
    )
    parser.add_argument(
        "--transform",
        type=str,
        default=None,
        help="Transform string (e.g. 'reduce_on_plateau')",
    )
    parser.add_argument(
        "--schedule-fn",
        "--schedule_fn",
        type=str,
        default=None,
        dest="schedule_fn",
        help="Learning rate schedule string (e.g. 'warmup', 'cosine')",
    )

    # Training Control Options
    parser.add_argument(
        "--early-stop-patience",
        "--early_stop_patience",
        type=int,
        default=None,
        dest="early_stop_patience",
        help="Number of epochs to wait for improvement before stopping training",
    )
    parser.add_argument(
        "--best",
        action="store_true",
        default=False,
        help="Only save checkpoint when objective improves",
    )
    parser.add_argument(
        "--no-save-every-epoch",
        action="store_false",
        dest="save_every_epoch",
        help="Disable saving a checkpoint at every epoch",
    )
    parser.set_defaults(save_every_epoch=True)
    parser.add_argument(
        "--profile-epoch-timing",
        action="store_true",
        dest="profile_epoch_timing",
        help="Print per-epoch timing breakdown (batch prep / train / valid / checkpoint)",
    )
    parser.add_argument(
        "--print-freq",
        "--print_freq",
        type=int,
        default=1,
        dest="print_freq",
        help="Printing frequency in epochs",
    )

    # Data & Batching Options
    parser.add_argument(
        "--batch-method",
        "--batch_method",
        type=str,
        default="default",
        dest="batch_method",
        help="Batching method ('default' or 'advanced')",
    )
    parser.add_argument(
        "--batch-args-dict",
        "--batch_args_dict",
        type=str,
        default=None,
        dest="batch_args_dict",
        help="JSON string or file path for advanced batch arguments",
    )
    parser.add_argument(
        "--data-keys",
        "--data_keys",
        type=str,
        nargs="+",
        default=None,
        dest="data_keys",
        help="Keys to load from NPZ file",
    )
    parser.add_argument(
        "--conversion",
        type=str,
        default=None,
        help=(
            "Display-only MAE scaling for energy/forces (JSON string or .json/"
            ".yaml path). Multiplies reported train/valid energy and force MAE "
            "after each epoch; does NOT transform NPZ arrays or affect the loss. "
            "Default when omitted: {\"energy\": 1, \"forces\": 1} (MAE in same "
            "units as the NPZ). Example for kcal/mol display when data are eV: "
            "'{\"energy\": 23.060549, \"forces\": 23.060549}'. Dipole units "
            "are not handled here — convert D/Dxyz before training (e.g. "
            "mmml fix-and-split --dipole-in debye --dipole-out e-angstrom). "
            "See docs/UNITS_SUMMARY.md § physnet-train --conversion."
        ),
    )
    parser.add_argument(
        "--init-params",
        "--init_params",
        type=str,
        default=None,
        dest="init_params",
        help="JSON string or file path to initialize flax parameters",
    )

    # Transfer learning
    parser.add_argument(
        "--physnet-checkpoint",
        "--physnet_checkpoint",
        type=str,
        default=None,
        dest="physnet_checkpoint",
        help="PhysNet checkpoint path (JSON or Orbax) for warm-start transfer learning",
    )
    parser.add_argument(
        "--physnet-transfer-model",
        "--physnet_transfer_model",
        type=str,
        default=None,
        dest="physnet_transfer_model",
        help=(
            "Bundled PhysNet transfer model ID, file stem, or category. "
            f"Defaults to {JOINT_TRAINING_CATEGORY!r} when distillation is enabled."
        ),
    )
    parser.add_argument(
        "--list-physnet-transfer-models",
        action="store_true",
        default=False,
        dest="list_physnet_transfer_models",
        help="List bundled PhysNet transfer-learning models and exit",
    )
    parser.add_argument(
        "--physnet-transfer-category",
        "--physnet_transfer_category",
        type=str,
        default=None,
        dest="physnet_transfer_category",
        help="Filter --list-physnet-transfer-models by manifest category",
    )
    parser.add_argument(
        "--match-checkpoint-architecture",
        action="store_true",
        default=True,
        dest="match_checkpoint_architecture",
        help="Override EF hyperparameters from transfer checkpoint config (default: on)",
    )
    parser.add_argument(
        "--no-match-checkpoint-architecture",
        action="store_false",
        dest="match_checkpoint_architecture",
        help="Do not override EF hyperparameters from transfer checkpoint config",
    )

    # Knowledge distillation
    parser.add_argument(
        "--distill",
        action="store_true",
        default=False,
        help="Enable teacher distillation loss during training",
    )
    parser.add_argument(
        "--distill-alpha",
        "--distill_alpha",
        type=float,
        default=1.0,
        dest="distill_alpha",
        help="Ground-truth loss weight (1.0=GT only, 0.0=teacher only)",
    )
    parser.add_argument(
        "--distill-targets",
        "--distill_targets",
        type=str,
        nargs="+",
        default=None,
        dest="distill_targets",
        help="Distillation targets: energy forces dipole (default: all three)",
    )
    parser.add_argument(
        "--teacher-checkpoint",
        "--teacher_checkpoint",
        type=str,
        default=None,
        dest="teacher_checkpoint",
        help="Teacher checkpoint for distillation (defaults to warm-start checkpoint)",
    )

    # Post-hoc learning curves
    parser.add_argument(
        "--metrics-plot",
        "--metrics_plot",
        type=str,
        default=None,
        dest="metrics_plot",
        help="After training, write learning-curve plot to this path via Orbax checkpoints",
    )
    parser.add_argument(
        "--log-loss",
        action="store_true",
        default=False,
        dest="log_loss",
        help="Use log scale on loss axes when generating --metrics-plot",
    )

    # Data Augmentation Options
    parser.add_argument(
        "--rot-augment",
        "--rot_augment",
        action="store_true",
        default=False,
        dest="rot_augment",
        help="Apply random rotation augmentation to inputs",
    )
    parser.add_argument(
        "--rot-perturbation",
        "--rot_perturbation",
        type=float,
        default=1.0,
        dest="rot_perturbation",
        help="Magnitude of rotation perturbation",
    )

    # Additional Model Options
    parser.add_argument(
        "--charges",
        action="store_true",
        default=False,
        help="Predict atomic charges (useful for dipoles and electrostatics)",
    )
    parser.add_argument(
        "--no-charges",
        action="store_false",
        dest="charges",
        help="Do not predict atomic charges",
    )
    parser.add_argument(
        "--total-charge",
        "--total_charge",
        type=float,
        default=0.0,
        dest="total_charge",
        help="Total charge constraint of the molecular system",
    )
    parser.add_argument(
        "--no-electrostatics",
        action="store_false",
        dest="include_electrostatics",
        help="Disable electrostatics layer in EF model",
    )
    parser.set_defaults(include_electrostatics=True)
    parser.add_argument(
        "--efa",
        action="store_true",
        default=False,
        help="Enable Euclidean Fast Attention (EFA) in the model",
    )
    parser.add_argument(
        "--no-efa",
        action="store_false",
        dest="efa",
        help="Disable Euclidean Fast Attention (EFA)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug flags in EF model",
    )
    parser.add_argument(
        "--no-debug",
        action="store_false",
        dest="debug",
        help="Disable debug flags in EF model",
    )
    parser.set_defaults(debug=False)

    parser.add_argument(
        "--save-config",
        "--save_config",
        type=str,
        default=None,
        dest="save_config",
        help="Write resolved training options to YAML and exit",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress JAX device summary",
    )
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open() as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config file must contain a YAML mapping, got {type(raw).__name__}")
    return raw


def apply_mapping_to_namespace(
    args: argparse.Namespace,
    mapping: Mapping[str, Any],
    *,
    source: str,
) -> None:
    unknown = []
    for raw_key, value in mapping.items():
        key = _normalize_config_key(str(raw_key))
        if not hasattr(args, key):
            unknown.append(str(raw_key))
            continue
        setattr(args, key, value)
    if unknown:
        raise ValueError(
            f"Unknown {source} key(s): {', '.join(sorted(unknown))}. "
            f"Valid keys include: {', '.join(sorted(k for k in vars(args) if not k.startswith('_')))}"
        )


def namespace_from_yaml(path: str | Path) -> argparse.Namespace:
    args = parse_args([])
    apply_mapping_to_namespace(args, load_yaml_config(path), source=f"config '{path}'")
    return args


def parse_train_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI args, optionally seeded from --config YAML."""
    parser = build_parser()
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None)
    pre_args, remaining = pre_parser.parse_known_args(argv)

    defaults = vars(parse_args([]))
    if pre_args.config:
        file_args = vars(namespace_from_yaml(pre_args.config))
        defaults.update(file_args)
    parser.set_defaults(**defaults)
    return parser.parse_args(remaining)


def save_train_config(args: argparse.Namespace, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: v for k, v in sorted(vars(args).items()) if k != "save_config"}
    with out.open("w") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, default_flow_style=False)


def _validate_handoff_within_cutoff(args: argparse.Namespace) -> None:
    """The ML basis must reach as far as the handoff, or the fit has a hole.

    ``--cutoff`` is an atom-pair radial cutoff; ``--mm-switch-on`` is a COM
    distance at which MM takes over. If mm_switch_on > cutoff the model is asked
    to carry the dimer interaction out to a separation its basis cannot see, and
    the interaction does not error -- it silently truncates inside the handoff.
    """
    if not getattr(args, "hybrid_mm", False):
        return
    cutoff = float(getattr(args, "cutoff", 0.0))
    mm_on = float(getattr(args, "mm_switch_on", 0.0))
    if mm_on > cutoff:
        raise ValueError(
            f"--mm-switch-on ({mm_on:g} A) exceeds --cutoff ({cutoff:g} A): the ML "
            f"basis dies before MM takes over, so the dimer interaction is silently "
            f"truncated in the handoff. Set --cutoff >= {mm_on:g}, or lower "
            f"--mm-switch-on to <= {cutoff:g}."
        )


def _warn_electrostatics_shorter_than_cutoff(args: argparse.Namespace) -> None:
    """Electrostatics dying before the radial basis does is almost never wanted.

    If the Coulomb term switches off at or inside ``--cutoff``, any pair beyond
    that distance contributes nothing at all: no message passing and no
    electrostatics. Training points out there are then unfittable, and the model
    does not fail loudly -- it fits what it can and returns a frozen artefact
    beyond. That is what happened to the Menshutkin checkpoint, where the ion
    pair lost its 1/r tail at exactly the radius the graph disconnected.
    """
    if not getattr(args, "include_electrostatics", True):
        return
    cutoff = float(getattr(args, "cutoff", 0.0))
    off_end = float(getattr(args, "electrostatics_off_end", 0.0))
    if off_end <= cutoff:
        print(
            f"WARNING: --electrostatics-off-end ({off_end:g} A) is not beyond "
            f"--cutoff ({cutoff:g} A), so pairs past {cutoff:g} A get neither "
            f"message passing nor electrostatics. If your data contains "
            f"separations beyond that, those points cannot be fitted. Raise "
            f"--electrostatics-off-start/--electrostatics-off-end past the "
            f"largest separation in the dataset."
        )


def validate_train_args(args: argparse.Namespace) -> None:
    if not args.data:
        raise ValueError("--data is required (or set 'data' / 'train' in --config)")
    _validate_handoff_within_cutoff(args)
    _warn_electrostatics_shorter_than_cutoff(args)
    data_paths = normalize_data_paths(args.data)
    if not data_paths:
        raise ValueError("--data must contain at least one NPZ path")

    if args.restart:
        if args.physnet_checkpoint or args.physnet_transfer_model:
            raise ValueError("--restart cannot be combined with transfer-learning checkpoints")
        if args.distill:
            raise ValueError("--restart cannot be combined with --distill")
        if args.teacher_checkpoint:
            raise ValueError("--restart cannot be combined with --teacher-checkpoint")

    if args.physnet_transfer_model and args.physnet_checkpoint:
        raise ValueError(
            "--physnet-transfer-model cannot be combined with --physnet-checkpoint"
        )

    if args.distill:
        if not (0.0 <= args.distill_alpha <= 1.0):
            raise ValueError("--distill-alpha must be between 0 and 1")
        warm_start, teacher = resolve_transfer_checkpoints(args)
        if teacher is None:
            raise ValueError(
                "--distill requires a teacher checkpoint "
                "(set --teacher-checkpoint, --physnet-checkpoint, or --physnet-transfer-model)"
            )

    if args.valid_data:
        # ``n_train``/``n_valid`` default to None so that *omitting* them (what the
        # example config documents) is not mistaken for an explicit request; only a
        # positive value actually conflicts with fixed valid_data splits. 0 is
        # tolerated: it was the workaround while the defaults were 1000/100.
        if (args.n_train or 0) > 0 or (args.n_valid or 0) > 0:
            raise ValueError(
                "With --valid-data, do not set --n-train/--n-valid (full files are used)"
            )
        return
    # Single-file split: fall back to the historical default split sizes.
    if args.n_train is None:
        args.n_train = 1000
    if args.n_valid is None:
        args.n_valid = 100
    if args.n_train < 0 or args.n_valid < 0:
        raise ValueError("--n-train and --n-valid must be >= 0")
    if args.n_train + args.n_valid <= 0:
        raise ValueError("At least one of --n-train or --n-valid must be > 0")


def _build_hybrid_mm_config(args: argparse.Namespace, data_paths: list[str]) -> dict | None:
    """Kwargs for the hybrid ML/MM assembly, or None when the mode is off.

    The ``cgenff_master_*`` LJ tables are ``(n_types,)`` -- not per-sample -- so
    the batching loader skips them (it drops arrays whose first dim != n_samples).
    Load them here and hand them to ``train_model`` as closure state instead.
    """
    if not getattr(args, "hybrid_mm", False):
        return None

    import numpy as _np

    from mmml.models.hybrid_energy import HYBRID_MM_BATCH_KEYS

    if not data_paths:
        raise ValueError("--hybrid-mm requires --data")
    path = data_paths[0]
    with _np.load(path, allow_pickle=True) as d:
        missing = [
            k
            for k in (*HYBRID_MM_BATCH_KEYS, "cgenff_master_sigmas", "cgenff_master_epsilons")
            if k not in d.files
        ]
        if missing:
            raise ValueError(
                f"--hybrid-mm needs CGenFF fields in {path}; missing: {', '.join(missing)}. "
                "Prepare the dataset with scripts/prepare_ml_mm_dataset.py (or the "
                "combined-dataset recipe) so it carries "
                f"{', '.join(HYBRID_MM_BATCH_KEYS)} + master LJ tables."
            )
        sigmas = _np.asarray(d["cgenff_master_sigmas"])
        epsilons = _np.asarray(d["cgenff_master_epsilons"])
        if "N" in d.files:
            n_atoms_est = int(_np.asarray(d["N"]).max())
        elif "R" in d.files:
            n_atoms_est = int(_np.asarray(d["R"]).shape[1])
        else:
            n_atoms_est = 32

    from mmml.models.mm_charge_mode import (
        mm_charge_mode_needs_q_ml,
        resolve_hybrid_mm_charge_mode,
    )

    mode = resolve_hybrid_mm_charge_mode(
        mm_charge_mode=getattr(args, "mm_charge_mode", None),
        charge_correction=bool(getattr(args, "mm_charge_correction", False)),
    )
    lr_solver = str(getattr(args, "lr_solver", "mic") or "mic").strip().lower()
    include_lj = bool(getattr(args, "mm_include_lj", True))
    learn_mm_lj_scales = bool(getattr(args, "learn_mm_lj_scales", False))
    pme_box_length = getattr(args, "pme_box_length", None)
    pme_accuracy = float(getattr(args, "pme_accuracy", 1e-6) or 1e-6)
    pme_real_space_cutoff = None
    if lr_solver == "nvalchemiops_pme":
        from mmml.interfaces.pycharmmInterface.long_range_backend import (
            estimate_nvalchemiops_pme_real_space_cutoff,
            have_nvalchemiops_pme,
            warmup_nvalchemiops_pme_train_worker,
        )

        if not have_nvalchemiops_pme():
            raise ValueError(
                "--lr-solver nvalchemiops_pme requires the nvalchemiops package "
                "(install mmml[nvalchemiops-pme])."
            )
        if pme_box_length is None or float(pme_box_length) <= 0.0:
            raise ValueError(
                "--lr-solver nvalchemiops_pme requires --pme-box-length > 0"
            )
        include_lj = False
        pme_box_length = float(pme_box_length)
        # Static cutoff for jitted train steps (PME params fixed for the run).
        pme_real_space_cutoff = estimate_nvalchemiops_pme_real_space_cutoff(
            box_length_A=pme_box_length,
            accuracy=pme_accuracy,
            n_atoms=n_atoms_est,
        )
        # Compile Warp PME in a spawn child before jit_train_step; nested CUDA
        # JAX inside pure_callback deadlocks the parent GPU XLA executor.
        warmup_nvalchemiops_pme_train_worker(
            n_atoms=int(n_atoms_est),
            box_length_A=pme_box_length,
            accuracy=pme_accuracy,
            real_space_cutoff_A=pme_real_space_cutoff,
        )
    elif lr_solver == "ewald":
        if pme_box_length is None or float(pme_box_length) <= 0.0:
            raise ValueError("--lr-solver ewald requires --pme-box-length > 0")
        include_lj = False
        pme_box_length = float(pme_box_length)
        # No external-package / cutoff-estimation step needed: ewald_hybrid_
        # coulomb.py defaults real_space_cutoff_A to box_length/2 internally
        # when left None, already validated at that setting.
    if lr_solver in ("nvalchemiops_pme", "ewald"):
        learn_mm_lj_scales = False
    if learn_mm_lj_scales and not include_lj:
        learn_mm_lj_scales = False
    cfg = {
        "master_sigmas": sigmas,
        "master_epsilons": epsilons,
        "mm_switch_on": float(args.mm_switch_on),
        "mm_switch_width": float(args.mm_switch_width),
        "ml_switch_width": float(args.ml_switch_width),
        "complementary_handoff": not bool(getattr(args, "no_complementary_handoff", False)),
        "hybrid_hamiltonian": str(getattr(args, "hybrid_hamiltonian", "handoff")),
        "shared_cutoff": (
            getattr(args, "shared_cutoff", None)
            if getattr(args, "shared_cutoff", None) is not None
            else float(getattr(args, "cutoff", 6.0))
        ),
        "mm_charge_mode": mode.value,
        "lr_solver": lr_solver,
        "include_lj": include_lj,
        "learn_mm_lj_scales": learn_mm_lj_scales,
        "pme_box_length": pme_box_length,
        "pme_accuracy": pme_accuracy,
        "pme_real_space_cutoff": pme_real_space_cutoff,
    }
    if mm_charge_mode_needs_q_ml(mode) and not getattr(args, "charges", False):
        raise ValueError(
            f"--mm-charge-mode {mode.value} needs a model with a charge head; "
            "pass --charges (without it the model predicts no charges)."
        )
    if not getattr(args, "quiet", False):
        lj_txt = "LJ+Coulomb" if include_lj else "Coulomb-only"
        pme_txt = ""
        if lr_solver in ("nvalchemiops_pme", "ewald"):
            pme_txt = (
                f", pme_box_length={pme_box_length}, "
                f"pme_accuracy={pme_accuracy}, "
                f"pme_real_space_cutoff={pme_real_space_cutoff}"
            )
        print(
            f"Hybrid ML/MM training: E = (1-s)*(E_A+E_B) + s*E_AB + E_MM  "
            f"({len(sigmas)} CGenFF types; ml_switch_width={cfg['ml_switch_width']}, "
            f"mm_switch_on={cfg['mm_switch_on']}, mm_switch_width={cfg['mm_switch_width']}, "
            f"complementary_handoff={cfg['complementary_handoff']}, "
            f"mm_charge_mode={cfg['mm_charge_mode']}, "
            f"lr_solver={lr_solver}, E_MM={lj_txt}, "
            f"learn_mm_lj_scales={learn_mm_lj_scales}{pme_txt})",
            flush=True,
        )
    return cfg


def normalize_data_paths(data: Any) -> list[str]:
    """Normalize ``data`` config/CLI value to a list of NPZ paths."""
    if data is None:
        return []
    if isinstance(data, str):
        stripped = data.strip()
        if not stripped:
            return []
        if "," in stripped:
            return [part.strip() for part in stripped.split(",") if part.strip()]
        return [stripped]
    if isinstance(data, (list, tuple)):
        paths = []
        for item in data:
            paths.extend(normalize_data_paths(item))
        return paths
    raise ValueError(f"Invalid data path specification: {data!r}")


def resolve_transfer_checkpoints(
    args: argparse.Namespace,
) -> tuple[Optional[Path], Optional[Path]]:
    """Resolve warm-start and teacher checkpoint paths from CLI/config."""
    warm_start_path: Optional[Path] = None
    if args.physnet_transfer_model:
        selected = resolve_hf_physnet_model(args.physnet_transfer_model)
        warm_start_path = Path(selected["path"])
    elif args.physnet_checkpoint:
        warm_start_path = Path(args.physnet_checkpoint)

    teacher_path: Optional[Path] = None
    if args.teacher_checkpoint:
        teacher_path = Path(args.teacher_checkpoint)
    elif args.distill:
        if warm_start_path is not None:
            teacher_path = warm_start_path
        else:
            try:
                selected = resolve_hf_physnet_model(JOINT_TRAINING_CATEGORY)
            except (KeyError, FileNotFoundError, ValueError):
                selected = None
            if selected is not None:
                teacher_path = Path(selected["path"])

    return warm_start_path, teacher_path


def load_transfer_init_params(
    args: argparse.Namespace,
) -> tuple[Optional[dict], Optional[dict], Optional[dict]]:
    """Load warm-start and teacher params; optionally apply checkpoint architecture."""
    explicit_init = _parse_dict_option(args.init_params)
    warm_start_path, teacher_path = resolve_transfer_checkpoints(args)

    warm_config = None
    init_params = explicit_init
    if init_params is None and warm_start_path is not None:
        init_params, warm_config = load_physnet_checkpoint(warm_start_path)
        print(f"Loaded warm-start checkpoint: {warm_start_path}")

    teacher_params = None
    teacher_config = None
    if args.distill and teacher_path is not None:
        if warm_start_path is not None and teacher_path.resolve() == warm_start_path.resolve():
            teacher_params = init_params
            teacher_config = warm_config
        else:
            teacher_params, teacher_config = load_physnet_checkpoint(teacher_path)
        print(f"Loaded teacher checkpoint: {teacher_path}")

    arch_config = warm_config or teacher_config
    if args.match_checkpoint_architecture and arch_config is not None:
        apply_checkpoint_architecture(args, arch_config)

    return init_params, teacher_params, arch_config


def _detect_natoms_from_data(data_paths: Sequence[str], num_atoms: Optional[int]) -> tuple[list[str], int]:
    """Unpad and detect natoms from the first training NPZ path."""
    paths = list(data_paths)
    if not paths:
        raise ValueError("No training data paths provided")
    first_path, natoms = _maybe_unpad_dataset(paths[0], num_atoms)
    paths[0] = first_path
    if num_atoms is None:
        for idx in range(1, len(paths)):
            path, _ = _maybe_unpad_dataset(paths[idx], natoms)
            paths[idx] = path
    return paths, natoms


def _merge_physnet_npz_dicts(chunks: Sequence[dict]) -> dict:
    merged = {}
    for chunk in chunks:
        for key, value in chunk.items():
            if key not in merged:
                merged[key] = value
            else:
                merged[key] = np.concatenate([merged[key], value], axis=0)
    return merged


def _plot_training_metrics_from_run(
    run_ckpt_dir: Path,
    output_path: Path,
    *,
    log_loss: bool,
    tag: str,
) -> None:
    from mmml.cli.misc.extract_checkpoint_metrics import (
        collect_all_metrics,
        plot_training_metrics,
    )

    metrics = collect_all_metrics(run_ckpt_dir, verbose=True)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_training_metrics(
        metrics,
        output_path,
        ckpt_name=tag,
        log_loss=log_loss,
        verbose=True,
    )
    print(f"Wrote learning-curve plot to {output_path}")


def args_from_kwargs(**overrides) -> argparse.Namespace:
    """Create an argparse-like Namespace using defaults, then apply overrides.

    Handy for notebook use: import this module and call `run_notebook(...)`
    with keyword arguments instead of constructing CLI strings.
    """
    args = parse_args([])
    for key, value in overrides.items():
        if not hasattr(args, key):
            raise ValueError(f"Unknown argument: {key}")
        setattr(args, key, value)
    return args


def log_jax_devices():
    """Print a short JAX device summary."""
    devices = jax.local_devices()
    print(devices)
    print(jax.default_backend())
    print(jax.devices())



def to_jsonable(obj: Any):
    """Recursively convert JAX/NumPy objects to JSON-serializable types."""
    # Handle JAX arrays (ArrayImpl) and NumPy arrays
    try:
        import jax
        jax_array_cls = getattr(jax, "Array", None)
        jax_array_types = (jax_array_cls,) if jax_array_cls is not None else tuple()
    except Exception:
        jax_array_types = tuple()

    if isinstance(obj, (np.ndarray,)) or (jax_array_types and isinstance(obj, jax_array_types)):
        return np.asarray(obj).tolist()
    # NumPy scalar types
    if isinstance(obj, np.generic):
        return obj.item()
    # Basic containers
    if isinstance(obj, dict):
        return {to_jsonable(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    # Paths
    if isinstance(obj, Path):
        return str(obj)
    # Fallback: return as-is; json will handle primitives
    return obj

def load_model(model_file):
    # load the model from the file
    with open(model_file, 'r') as f:
        model = EF(**json.load(f))
    return model


def _maybe_unpad_dataset(data_path: str, natoms: Optional[int]) -> tuple[str, int]:
    """Return (path, natoms), optionally writing an unpadded NPZ."""
    data = np.load(data_path, allow_pickle=True)
    if natoms is not None:
        return data_path, natoms

    if "N" in data:
        max_n = int(np.max(data["N"]))
        if "R" in data and len(data["R"].shape) >= 2:
            padded_atoms = int(data["R"].shape[1])
            if padded_atoms > max_n:
                print(f"  ⚠️  Data is PADDED: {padded_atoms} atoms in array (padding: {padded_atoms - max_n})")
                print("  🔧 Auto-removing padding to train efficiently...")
                n_samples = int(np.asarray(data["R"]).shape[0])
                data_unpadded = {}
                for key, value in data.items():
                    arr = np.asarray(value)
                    # Trim every per-sample array whose axis 1 is the padded
                    # atom axis -- not just R/Z/F. The hybrid ML/MM fields
                    # (cgenff_type_idx, mol_id, cgenff_charge, F_cgenff_mm) are
                    # also (n, atoms[, 3]); leaving them at the old width makes
                    # hybrid_forward fail with a broadcasting error between
                    # atom_mask (n*max_n) and the mol_id-derived keep mask
                    # (n*padded_atoms). Shape-driven so new per-atom fields are
                    # handled without another edit here.
                    if (
                        arr.ndim >= 2
                        and arr.shape[0] == n_samples
                        and arr.shape[1] == padded_atoms
                    ):
                        data_unpadded[key] = arr[:, :max_n, ...]
                    else:
                        data_unpadded[key] = value
                unpadded_path = Path(data_path).parent / f"{Path(data_path).stem}_unpadded.npz"
                np.savez_compressed(unpadded_path, **data_unpadded)
                print(f"  ✅ Saved unpadded data to: {unpadded_path}")
                return str(unpadded_path), max_n
            return data_path, padded_atoms
        return data_path, max_n
    if "R" in data and len(data["R"].shape) >= 2:
        return data_path, int(data["R"].shape[1])
    raise ValueError("Could not auto-detect num_atoms from dataset. Please specify --num-atoms.")


def _load_physnet_npz_dict(path: str, natoms: int) -> dict:
    """Load one NPZ split into the dict format expected by train_model."""
    from mmml.models.physnetjax.physnetjax.data.data import make_dicts, prepare_multiple_datasets

    data, keys, _, _ = prepare_multiple_datasets(
        jax.random.PRNGKey(0),
        train_size=0,
        valid_size=0,
        filename=[path],
        natoms=natoms,
        verbose=False,
    )
    n_samples = data[keys.index("R")].shape[0]
    all_idx = np.arange(n_samples)
    train_data, _ = make_dicts(data, keys, all_idx, np.array([], dtype=np.int64))
    return train_data


def main_loop(args):
    seed = args.seed
    data_key, train_key = jax.random.split(jax.random.PRNGKey(seed), 2)

    if args.ckpt_dir is not None:
        ckpt_dir = Path(args.ckpt_dir).resolve()
        print(f"Checkpoint directory (absolute): {ckpt_dir}")
    else:
        ckpt_dir = None

    data_paths, natoms = _detect_natoms_from_data(
        normalize_data_paths(args.data),
        args.num_atoms,
    )
    args.data = data_paths if len(data_paths) > 1 else data_paths[0]
    if args.num_atoms is None:
        print(f"Auto-detected num_atoms = {natoms}")
    else:
        print(f"Using specified num_atoms = {natoms}")

    init_params, teacher_params, _arch_config = load_transfer_init_params(args)
    distill_targets = _parse_list_option(args.distill_targets)
    if args.distill and distill_targets is None:
        distill_targets = ["energy", "forces", "dipole"]

    if args.valid_data:
        valid_path, _ = _maybe_unpad_dataset(args.valid_data, natoms)
        args.valid_data = valid_path
        train_label = data_paths if len(data_paths) > 1 else data_paths[0]
        print(f"Using fixed splits:\n  train: {train_label}\n  valid: {args.valid_data}")
        train_chunks = [_load_physnet_npz_dict(path, natoms) for path in data_paths]
        train_data = _merge_physnet_npz_dicts(train_chunks) if len(train_chunks) > 1 else train_chunks[0]
        valid_data = _load_physnet_npz_dict(args.valid_data, natoms)
    else:
        train_data, valid_data = prepare_datasets(
            data_key, args.n_train, args.n_valid, data_paths, natoms=natoms
        )
    
    if args.model is not None:
        model = load_model(args.model)
    else:
        model_kwargs = dict(
            features=args.features,
            max_degree=args.max_degree,
            num_basis_functions=args.num_basis_functions,
            num_iterations=args.num_iterations,
            n_refinement_blocks=args.n_res,
            cutoff=args.cutoff,
            switch_start=args.switch_start,
            switch_end=args.switch_end,
            electrostatics_off_start=args.electrostatics_off_start,
            electrostatics_off_end=args.electrostatics_off_end,
            max_atomic_number=args.max_atomic_number,
            zbl=args.zbl,
            trainable_zbl=args.trainable_zbl,
            efa=args.efa,
            use_pbc=args.use_pbc,
            use_energy_bias=args.use_energy_bias,
            charges=args.charges,
            total_charge=args.total_charge,
            include_electrostatics=args.include_electrostatics,
            max_padded_atoms=natoms,
            debug=args.debug,
        )
        from mmml.utils.model_checkpoint import physnet_constructor_kwargs

        model = EF(**physnet_constructor_kwargs(model_kwargs, EF))
        try:
            with open("args.model.json", 'w') as f:
                print("Saving model to args.model.json")
                print(model.return_attributes())
                json.dump(model.return_attributes(), f, default=to_jsonable)
        except Exception as e:
            print(e)
            pass
    
    conversion = _parse_dict_option(args.conversion) or {'energy': 1, 'forces': 1}
    batch_args_dict = _parse_dict_option(args.batch_args_dict)
    data_keys_list = _parse_list_option(args.data_keys)
    if data_keys_list is not None:
        data_keys = tuple(data_keys_list)
    else:
        data_keys = ('R', 'Z', 'F', "N", 'E', 'D', 'batch_segments')

    hybrid_mm = _build_hybrid_mm_config(args, data_paths)
    if hybrid_mm is not None:
        from mmml.models.hybrid_energy import HYBRID_MM_BATCH_KEYS

        data_keys = tuple(data_keys) + tuple(
            k for k in HYBRID_MM_BATCH_KEYS if k not in data_keys
        )

    # nvalchemiops PME cannot nest on the same GPU XLA executor as jit_train_step
    # (deadlock), and a spawn child often gets CUDA_ERROR_DEVICE_UNAVAILABLE after
    # the parent has already initialized CUDA.  Default isolate mode runs the
    # jitted train/eval steps on CPU while the PME pure_callback uses GPU.
    from mmml.interfaces.pycharmmInterface.long_range_backend import (
        nvalchemiops_pme_train_wants_cpu_steps,
    )

    _cpu_train = bool(
        hybrid_mm is not None
        and str(hybrid_mm.get("lr_solver", "mic")).lower() == "nvalchemiops_pme"
        and nvalchemiops_pme_train_wants_cpu_steps()
    )
    _train_device_ctx = (
        jax.default_device(jax.devices("cpu")[0]) if _cpu_train else nullcontext()
    )
    if _cpu_train:
        print(
            "nvalchemiops_pme: jit train/eval on CPU; PME callback on GPU "
            "(set MMML_NVALCHEMIOPS_PME_ISOLATE=spawn to try a second-GPU worker).",
            flush=True,
        )

    with _train_device_ctx:
        ema_params, best_loss, run_ckpt_dir = train_model(
            train_key,
            model,
            train_data,
            valid_data,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            num_atoms=natoms,
            energy_weight=args.energy_weight,
            forces_weight=args.forces_weight,
            dipole_weight=args.dipole_weight,
            charges_weight=args.charges_weight,
            restart=args.restart,
            ema_decay=args.ema_decay,
            conversion=conversion,
            print_freq=args.print_freq,
            name=args.tag,
            best=args.best,
            optimizer=args.optimizer,
            transform=args.transform,
            schedule_fn=args.schedule_fn,
            objective=args.objective,
            ckpt_dir=ckpt_dir,
            log_tb=False,
            batch_method=args.batch_method,
            batch_args_dict=batch_args_dict,
            data_keys=data_keys,
            hybrid_mm=hybrid_mm,
            num_epochs=args.num_epochs,
            early_stop_patience=args.early_stop_patience,
            init_params=init_params,
            rot_augment=args.rot_augment,
            rot_perturbation=args.rot_perturbation,
            save_every_epoch=args.save_every_epoch,
            profile_epoch_timing=args.profile_epoch_timing,
            teacher_params=teacher_params if args.distill else None,
            distill_alpha=args.distill_alpha,
            distill_targets=distill_targets,
        )

    if args.metrics_plot and run_ckpt_dir is not None:
        try:
            _plot_training_metrics_from_run(
                Path(run_ckpt_dir),
                Path(args.metrics_plot),
                log_loss=args.log_loss,
                tag=args.tag,
            )
        except Exception as exc:
            print(f"Warning: failed to write metrics plot: {exc}")

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_attrs = model.return_attributes()
    params_inner = normalize_flax_params_for_apply(ema_params, backend="jax")["params"]
    portable = {"params": params_inner, "config": model_attrs}
    params_path = (ckpt_dir / f"params_{args.tag}_{now}.json") if ckpt_dir else Path(f"params_{args.tag}_{now}.json")
    if ckpt_dir:
        params_path.parent.mkdir(parents=True, exist_ok=True)
    with open(params_path, "w") as f:
        print(f"Saving portable checkpoint to {params_path}")
        json.dump(portable, f, default=to_jsonable)

    return ema_params, params_path, run_ckpt_dir
    


def run(args):
    print(args)
    return main_loop(args)


def run_notebook(**kwargs):
    """Convenience entrypoint for notebooks.

    Example:
        from mmml.cli import make_training

        params, params_path = make_training.run_notebook(
            data="train.npz",
            ckpt_dir="/tmp/ckpts",
            tag="run",
            model=None,
            n_train=1000,
            n_valid=100,
            seed=42,
            batch_size=1,
            num_epochs=2,
            learning_rate=0.001,
            energy_weight=1,
            objective="valid_loss",
            restart=None,
            num_atoms=None,
            features=64,
            max_degree=0,
            num_basis_functions=32,
            num_iterations=2,
            n_res=2,
            cutoff=8.0,
            max_atomic_number=28,
        )
    """
    args = args_from_kwargs(**kwargs)
    return run(args)


def main(argv: Optional[Sequence[str]] = None) -> int:
    import sys

    try:
        args = parse_train_args(argv)
        if args.list_physnet_transfer_models:
            print_bundled_physnet_models(args.physnet_transfer_category)
            return 0
        if args.save_config:
            save_train_config(args, args.save_config)
            print(f"Wrote training config to {args.save_config}")
            return 0
        validate_train_args(args)
        if not args.quiet:
            log_jax_devices()
        run(args)
        return 0
    except (ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def cli_main():
    raise SystemExit(main())


if __name__ == "__main__":
    cli_main()
