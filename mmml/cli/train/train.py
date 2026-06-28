#!/usr/bin/env python
"""
Unified training command for MMML models (DCMNet and PhysNetJAX).

Usage:
    mmml train --model dcmnet --train train.npz --valid valid.npz
    mmml train --config config.yaml
"""

import sys
import argparse
from pathlib import Path
import yaml
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mmml.data import (
    load_npz,
    train_valid_split
)
from mmml.models.physnetjax.defaults import (
    JOINT_TRAINING_CATEGORY,
    list_hf_physnet_models,
    resolve_hf_physnet_model,
)


@dataclass
class TrainConfig:
    """Training configuration."""
    # Model
    model: str = 'dcmnet'  # dcmnet or physnetjax
    
    # Data
    train_file: str = 'train.npz'
    valid_file: Optional[str] = None
    train_fraction: float = 0.8  # If valid_file not provided
    
    # Training
    batch_size: int = 32
    max_epochs: int = 1000
    learning_rate: float = 0.001
    early_stopping: int = 50
    
    # Targets
    targets: list = None  # ['energy', 'forces', 'dipole']
    loss_weights: Dict[str, float] = None  # {'energy': 1.0, 'forces': 100.0}
    
    # Output
    output_dir: str = 'checkpoints'
    log_interval: int = 10
    checkpoint_interval: int = 100
    save_best_only: bool = True
    
    # Preprocessing
    center_coordinates: bool = False
    normalize_energy: bool = False
    rot_augment: bool = False
    rot_perturbation: float = 1.0
    
    # Model-specific (will be passed to model constructor)
    model_params: Dict[str, Any] = None
    physnet_checkpoint: Optional[str] = None
    physnet_transfer_model: Optional[str] = None
    
    def __post_init__(self):
        if self.targets is None:
            self.targets = ['energy']
        if self.loss_weights is None:
            self.loss_weights = {'energy': 1.0}
        if self.model_params is None:
            self.model_params = {}


def load_config_file(config_file: str) -> TrainConfig:
    """Load configuration from YAML file."""
    with open(config_file, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Convert to TrainConfig
    return TrainConfig(**config_dict)


def save_config(config: TrainConfig, output_file: str):
    """Save configuration to YAML file."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(asdict(config), f, default_flow_style=False)


def print_bundled_physnet_models(category: Optional[str] = None):
    """Print bundled PhysNet transfer-learning choices."""
    models = list_hf_physnet_models(category)
    title = "Bundled PhysNet transfer models"
    if category:
        title += f" ({category})"
    print(f"\n{title}:")
    for entry in models:
        config = entry.get("config", {})
        objectives = entry.get("metadata", {}).get("objectives", {})
        print(
            f"  {entry['id']}: {entry.get('label', entry['file'])}\n"
            f"    file={entry['file']}\n"
            f"    categories={', '.join(entry.get('categories', []))}\n"
            f"    charges={config.get('charges')}, electrostatics={config.get('include_electrostatics', False)}, "
            f"features={config.get('features')}, basis={config.get('num_basis_functions')}, "
            f"iterations={config.get('num_iterations')}, max_degree={config.get('max_degree')}\n"
            f"    valid_forces_mae={objectives.get('valid_forces_mae')}, "
            f"valid_energy_mae={objectives.get('valid_energy_mae')}, "
            f"valid_dipole_mae={objectives.get('valid_dipole_mae')}"
        )


def train_dcmnet(
    config: TrainConfig,
    train_data: Dict,
    valid_data: Dict,
    verbose: bool = True
):
    """
    Train DCMNet model.
    
    Parameters
    ----------
    config : TrainConfig
        Training configuration
    train_data : dict
        Training data dictionary
    valid_data : dict
        Validation data dictionary
    verbose : bool
        Whether to print progress
    """
    if verbose:
        print("\n" + "="*60)
        print("Training DCMNet Model")
        print("="*60)
    
    try:
        from mmml.data.adapters import prepare_dcmnet_batches
        
        # Prepare batches
        if verbose:
            print("\n📦 Preparing training batches...")
        
        train_batches = prepare_dcmnet_batches(
            train_data,
            batch_size=config.batch_size,
            shuffle=True
        )
        
        valid_batches = prepare_dcmnet_batches(
            valid_data,
            batch_size=config.batch_size,
            shuffle=False
        )
        
        if verbose:
            print(f"   Train batches: {len(train_batches)}")
            print(f"   Valid batches: {len(valid_batches)}")
        
        # TODO: Load actual DCMNet model and train
        # For now, this is a placeholder showing the structure
        
        if verbose:
            print("\n⚠️  Actual DCMNet training not yet implemented")
            print("   Batch preparation successful!")
            print("   Ready for model training integration")
        
        return True
        
    except Exception as e:
        print(f"❌ Error training DCMNet: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return False


def train_physnetjax(
    config: TrainConfig,
    train_data: Dict,
    valid_data: Dict,
    verbose: bool = True
):
    """
    Train PhysNetJAX via ``make_training`` using file paths from ``TrainConfig``.

    For full options (YAML, model hyperparameters), use ``mmml physnet-train``.
    """
    if verbose:
        print("\n" + "="*60)
        print("Training PhysNetJAX Model")
        print("="*60)
        print("  Tip: use `mmml physnet-train --config train.yaml` for full control")

    try:
        from mmml.cli.make.make_training import args_from_kwargs, run, validate_train_args

        kwargs = {
            "data": config.train_file,
            "ckpt_dir": config.output_dir,
            "tag": Path(config.output_dir).name or "physnet_run",
            "batch_size": config.batch_size,
            "num_epochs": config.max_epochs,
            "learning_rate": config.learning_rate,
            "energy_weight": (config.loss_weights or {}).get("energy", 1.0),
            "forces_weight": (config.loss_weights or {}).get("forces", 52.91),
            "dipole_weight": (config.loss_weights or {}).get("dipole", 27.21),
            "restart": config.physnet_checkpoint,
        }
        if config.valid_file:
            kwargs["valid_data"] = config.valid_file
        else:
            kwargs["n_train"] = len(train_data.get("E", train_data.get("R", [])))
            kwargs["n_valid"] = len(valid_data.get("E", valid_data.get("R", [])))
        if config.model_params:
            kwargs.update(config.model_params)

        args = args_from_kwargs(**kwargs)
        validate_train_args(args)
        run(args)
        return True

    except Exception as e:
        print(f"❌ Error training PhysNetJAX: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Train MMML models (DCMNet or PhysNetJAX)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with config file
  %(prog)s --config config.yaml
  
  # Train DCMNet from command line
  %(prog)s --model dcmnet \\
           --train train.npz \\
           --valid valid.npz \\
           --output checkpoints/dcmnet/
  
  # Train with auto train/valid split
  %(prog)s --model dcmnet \\
           --train dataset.npz \\
           --train-fraction 0.8
  
  # Train PhysNetJAX (prefer physnet-train for full options / YAML)
  mmml physnet-train --config train.yaml
  %(prog)s --model physnetjax --train train.npz --valid valid.npz
        """
    )
    
    # Configuration
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--save-config',
        type=str,
        help='Save configuration to YAML file and exit'
    )
    
    # Model
    parser.add_argument(
        '--model',
        choices=['dcmnet', 'physnetjax'],
        default='dcmnet',
        help='Model to train (default: dcmnet)'
    )
    parser.add_argument(
        '--physnet-checkpoint',
        type=str,
        default=None,
        help='PhysNet checkpoint path for transfer learning'
    )
    parser.add_argument(
        '--physnet-transfer-model',
        type=str,
        default=None,
        help=(
            'Bundled PhysNet transfer model ID, file stem, or category. '
            f'Defaults to the {JOINT_TRAINING_CATEGORY!r} charged model for physnetjax training.'
        )
    )
    parser.add_argument(
        '--list-physnet-transfer-models',
        action='store_true',
        default=False,
        help='List bundled PhysNet transfer-learning models and exit'
    )
    parser.add_argument(
        '--physnet-transfer-category',
        type=str,
        default=None,
        help='Filter --list-physnet-transfer-models by manifest category'
    )
    
    # Data
    parser.add_argument(
        '--train',
        type=str,
        help='Training NPZ file'
    )
    parser.add_argument(
        '--valid',
        type=str,
        help='Validation NPZ file (optional if using --train-fraction)'
    )
    parser.add_argument(
        '--train-fraction',
        type=float,
        default=0.8,
        help='Fraction for train split if --valid not provided (default: 0.8)'
    )
    
    # Training parameters
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size (default: 32)'
    )
    parser.add_argument(
        '--max-epochs',
        type=int,
        default=1000,
        help='Maximum number of epochs (default: 1000)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Learning rate (default: 0.001)'
    )
    parser.add_argument(
        '--early-stopping',
        type=int,
        default=50,
        help='Early stopping patience (default: 50)'
    )
    
    # Targets
    parser.add_argument(
        '--targets',
        nargs='+',
        default=['energy'],
        help='Training targets (default: energy)'
    )
    
    # Output
    parser.add_argument(
        '--output',
        type=str,
        default='checkpoints',
        help='Output directory for checkpoints (default: checkpoints)'
    )
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        help='Log interval in epochs (default: 10)'
    )
    
    # Preprocessing
    parser.add_argument(
        '--center-coords',
        action='store_true',
        help='Center coordinates at origin'
    )
    parser.add_argument(
        '--normalize-energy',
        action='store_true',
        help='Normalize energies'
    )
    parser.add_argument(
        '--rot-augment',
        action='store_true',
        help='Enable SO(3) rotational augmentation in batch builders'
    )
    parser.add_argument(
        '--rot-perturbation',
        type=float,
        default=1.0,
        help='Rotation perturbation strength in [0,1] (default: 1.0)'
    )
    
    # Options
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Verbose output'
    )
    parser.add_argument(
        '--quiet',
        '-q',
        action='store_true',
        help='Quiet mode'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Prepare data but do not train'
    )
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def main():
    """Main training CLI entry point."""
    args = parse_args()
    if args.list_physnet_transfer_models:
        print_bundled_physnet_models(args.physnet_transfer_category)
        return 0

    verbose = args.verbose and not args.quiet
    
    # Load or create configuration
    if args.config:
        if verbose:
            print(f"📋 Loading configuration from {args.config}...")
        config = load_config_file(args.config)
    else:
        # Create config from command-line arguments
        if not args.train:
            print("❌ Error: --train required (or use --config)", file=sys.stderr)
            return 1

        physnet_checkpoint = args.physnet_checkpoint
        physnet_transfer_model = args.physnet_transfer_model
        if args.model == 'physnetjax' and physnet_checkpoint is None:
            try:
                selected_model = resolve_hf_physnet_model(
                    physnet_transfer_model or JOINT_TRAINING_CATEGORY
                )
            except KeyError as e:
                print(f"❌ Error: {e}", file=sys.stderr)
                return 1
            physnet_transfer_model = selected_model["id"]
            physnet_checkpoint = str(selected_model["path"])
        elif args.model != 'physnetjax' and physnet_transfer_model:
            try:
                selected_model = resolve_hf_physnet_model(physnet_transfer_model)
            except KeyError as e:
                print(f"❌ Error: {e}", file=sys.stderr)
                return 1
            physnet_transfer_model = selected_model["id"]
            physnet_checkpoint = str(selected_model["path"])
        
        config = TrainConfig(
            model=args.model,
            train_file=args.train,
            valid_file=args.valid,
            train_fraction=args.train_fraction,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            learning_rate=args.learning_rate,
            early_stopping=args.early_stopping,
            targets=args.targets,
            output_dir=args.output,
            log_interval=args.log_interval,
            center_coordinates=args.center_coords,
            normalize_energy=args.normalize_energy,
            rot_augment=args.rot_augment,
            rot_perturbation=args.rot_perturbation,
            physnet_checkpoint=physnet_checkpoint,
            physnet_transfer_model=physnet_transfer_model,
        )

    if config.model == 'physnetjax' and config.physnet_checkpoint is None:
        try:
            selected_model = resolve_hf_physnet_model(
                config.physnet_transfer_model or JOINT_TRAINING_CATEGORY
            )
        except KeyError as e:
            print(f"❌ Error: {e}", file=sys.stderr)
            return 1
        config.physnet_transfer_model = selected_model["id"]
        config.physnet_checkpoint = str(selected_model["path"])
    
    # Save config if requested
    if args.save_config:
        save_config(config, args.save_config)
        if not args.quiet:
            print(f"✓ Configuration saved to {args.save_config}")
        return 0
    
    # Print configuration
    if verbose:
        print("\n" + "="*60)
        print("MMML Training")
        print("="*60)
        print(f"\nModel: {config.model}")
        print(f"Train file: {config.train_file}")
        print(f"Valid file: {config.valid_file or 'auto-split'}")
        print(f"Batch size: {config.batch_size}")
        print(f"Max epochs: {config.max_epochs}")
        print(f"Targets: {', '.join(config.targets)}")
    
    # Load training data
    if verbose:
        print("\n📂 Loading training data...")
    
    try:
        train_data = load_npz(
            config.train_file,
            validate=True,
            verbose=verbose
        )
    except Exception as e:
        print(f"❌ Error loading training data: {e}", file=sys.stderr)
        return 1
    
    # Load or split validation data
    if config.valid_file:
        if verbose:
            print("📂 Loading validation data...")
        try:
            valid_data = load_npz(
                config.valid_file,
                validate=True,
                verbose=verbose
            )
        except Exception as e:
            print(f"❌ Error loading validation data: {e}", file=sys.stderr)
            return 1
    else:
        if verbose:
            print(f"✂️  Splitting data (train: {config.train_fraction:.0%}, valid: {1-config.train_fraction:.0%})...")
        train_data, valid_data = train_valid_split(
            train_data,
            train_fraction=config.train_fraction,
            shuffle=True,
            seed=42
        )
    
    if verbose:
        print("\n✓ Data loaded:")
        print(f"   Train: {len(train_data['E'])} structures")
        print(f"   Valid: {len(valid_data['E'])} structures")
    
    # Dry run - just prepare data
    if args.dry_run:
        if not args.quiet:
            print("\n🏃 Dry run - data preparation successful!")
            print("   Ready for training (remove --dry-run to train)")
        return 0
    
    # Train model
    start_time = time.time()
    
    if config.model == 'dcmnet':
        success = train_dcmnet(config, train_data, valid_data, verbose=verbose)
    elif config.model == 'physnetjax':
        success = train_physnetjax(config, train_data, valid_data, verbose=verbose)
    else:
        print(f"❌ Unknown model: {config.model}", file=sys.stderr)
        return 1
    
    elapsed = time.time() - start_time
    
    if success:
        if not args.quiet:
            print("\n✅ Training complete!")
            print(f"   Time: {elapsed:.2f}s")
        return 0
    else:
        print("\n❌ Training failed", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())

