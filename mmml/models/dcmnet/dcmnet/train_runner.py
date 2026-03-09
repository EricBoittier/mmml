"""
Convenient training runner for DCMNet with multi-batch support.

Usage:
    python train_runner.py --name my_experiment --epochs 100 --batch-size 2
    
Or in Python:
    from mmml.dcmnet.dcmnet.train_runner import run_training
    run_training(name="my_experiment", num_epochs=100)
"""
import argparse
from pathlib import Path
from typing import Optional
import jax

from .training_config import ExperimentConfig, TrainingConfig, ModelConfig, create_default_config
from .training_multibatch import train_model_multibatch
from .data import prepare_datasets
from .analysis import create_model


def run_training(
    name: str = "dcmnet_experiment",
    data_files: list = None,
    num_train: int = 1000,
    num_valid: int = 200,
    num_epochs: int = 100,
    batch_size: int = 1,
    learning_rate: float = 1e-4,
    gradient_accumulation_steps: int = 1,
    n_dcm: int = 2,
    features: int = 16,
    max_degree: int = 2,
    num_iterations: int = 2,
    esp_w: float = 1.0,
    chg_w: float = 0.01,
    use_grad_clip: bool = True,
    grad_clip_norm: float = 2.0,
    use_lr_schedule: bool = True,
    lr_schedule_type: str = "cosine",
    save_every_n_epochs: int = 10,
    output_dir: str = "./experiments",
    random_seed: int = 42,
    restart_checkpoint: Optional[Path] = None,
    num_atoms: int = 60,
    **kwargs
):
    """
    Run training with specified configuration.
    
    Parameters
    ----------
    name : str
        Experiment name
    data_files : list
        List of data files to load
    num_train : int
        Number of training samples
    num_valid : int
        Number of validation samples
    num_epochs : int
        Number of training epochs
    batch_size : int
        Batch size
    learning_rate : float
        Learning rate
    gradient_accumulation_steps : int
        Number of batches to accumulate gradients over
    n_dcm : int
        Number of distributed multipoles per atom
    features : int
        Number of features
    max_degree : int
        Maximum spherical harmonic degree
    num_iterations : int
        Number of message passing iterations
    esp_w : float
        ESP loss weight
    chg_w : float
        Charge loss weight
    use_grad_clip : bool
        Whether to use gradient clipping
    grad_clip_norm : float
        Gradient clipping norm
    use_lr_schedule : bool
        Whether to use learning rate schedule
    lr_schedule_type : str
        Type of LR schedule ('cosine', 'exponential', 'step')
    save_every_n_epochs : int
        Save checkpoint every N epochs
    output_dir : str
        Output directory
    random_seed : int
        Random seed
    restart_checkpoint : Optional[Path]
        Path to checkpoint to restart from
    num_atoms : int
        Maximum number of atoms
    **kwargs
        Additional configuration parameters
        
    Returns
    -------
    tuple
        (final_params, best_valid_loss, exp_dir)
    """
    if data_files is None:
        data_files = ["esp2000.npz"]
    
    # Create configuration
    config = ExperimentConfig(name=name)
    
    # Model config
    config.model = ModelConfig(
        n_dcm=n_dcm,
        features=features,
        max_degree=max_degree,
        num_iterations=num_iterations,
    )
    
    # Training config
    config.training = TrainingConfig(
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_atoms=num_atoms,
        esp_w=esp_w,
        chg_w=chg_w,
        use_grad_clip=use_grad_clip,
        grad_clip_norm=grad_clip_norm,
        gradient_accumulation_steps=gradient_accumulation_steps,
        use_lr_schedule=use_lr_schedule,
        lr_schedule_type=lr_schedule_type,
        save_every_n_epochs=save_every_n_epochs,
        num_train=num_train,
        num_valid=num_valid,
    )
    
    # Other config
    config.data_files = data_files
    config.output_dir = output_dir
    config.random_seed = random_seed
    
    # Update with any additional kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        elif hasattr(config.model, key):
            setattr(config.model, key, value)
        elif hasattr(config.training, key):
            setattr(config.training, key, value)
    
    # Initialize random key
    key = jax.random.PRNGKey(random_seed)
    
    # Load data
    print("\nLoading datasets...")
    key, data_key = jax.random.split(key)
    train_data, valid_data = prepare_datasets(
        data_key,
        num_train=num_train,
        num_valid=num_valid,
        filename=data_files,
        natoms=num_atoms,
    )
    
    print(f"Training samples: {len(train_data['R'])}")
    print(f"Validation samples: {len(valid_data['R'])}")
    
    # Create model
    print("\nCreating model...")
    model = create_model(
        n_dcm=config.model.n_dcm,
        features=config.model.features,
        max_degree=config.model.max_degree,
        num_iterations=config.model.num_iterations,
        num_basis_functions=config.model.num_basis_functions,
        cutoff=config.model.cutoff,
        include_pseudotensors=config.model.include_pseudotensors,
    )
    
    print(f"Model: {config.model.n_dcm} DCM, {config.model.features} features")
    
    # Run training
    print("\n" + "="*90)
    print("Starting Training")
    print("="*90 + "\n")
    
    final_params, best_valid_loss = train_model_multibatch(
        key=key,
        model=model,
        train_data=train_data,
        valid_data=valid_data,
        config=config,
        restart_checkpoint=restart_checkpoint,
    )
    
    exp_dir = config.get_experiment_dir()
    
    print("\n" + "="*90)
    print("Training Complete!")
    print("="*90)
    print(f"Best validation loss: {best_valid_loss:.6e}")
    print(f"Results saved to: {exp_dir}")
    print("="*90 + "\n")
    
    return final_params, best_valid_loss, exp_dir


def main():
    """Command-line interface for training."""
    parser = argparse.ArgumentParser(
        description="Train DCMNet with multi-batch support",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Experiment
    parser.add_argument("--name", type=str, default="dcmnet_experiment",
                        help="Experiment name")
    parser.add_argument("--output-dir", type=str, default="./experiments",
                        help="Output directory")
    parser.add_argument("--random-seed", type=int, default=42,
                        help="Random seed")
    
    # Data
    parser.add_argument("--data-files", nargs="+", default=["esp2000.npz"],
                        help="Data files to load")
    parser.add_argument("--num-train", type=int, default=1000,
                        help="Number of training samples")
    parser.add_argument("--num-valid", type=int, default=200,
                        help="Number of validation samples")
    parser.add_argument("--num-atoms", type=int, default=60,
                        help="Maximum number of atoms")
    
    # Model
    parser.add_argument("--n-dcm", type=int, default=2,
                        help="Number of distributed multipoles per atom")
    parser.add_argument("--features", type=int, default=16,
                        help="Number of features")
    parser.add_argument("--max-degree", type=int, default=2,
                        help="Maximum spherical harmonic degree")
    parser.add_argument("--num-iterations", type=int, default=2,
                        help="Number of message passing iterations")
    
    # Training
    parser.add_argument("--num-epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1,
                        help="Number of batches to accumulate gradients over")
    
    # Loss
    parser.add_argument("--esp-w", type=float, default=1.0,
                        help="ESP loss weight")
    parser.add_argument("--chg-w", type=float, default=0.01,
                        help="Charge loss weight")
    
    # Optimization
    parser.add_argument("--use-grad-clip", action="store_true", default=True,
                        help="Use gradient clipping")
    parser.add_argument("--grad-clip-norm", type=float, default=2.0,
                        help="Gradient clipping norm")
    parser.add_argument("--use-lr-schedule", action="store_true", default=True,
                        help="Use learning rate schedule")
    parser.add_argument("--lr-schedule-type", type=str, default="cosine",
                        choices=["cosine", "exponential", "step"],
                        help="Learning rate schedule type")
    
    # Checkpointing
    parser.add_argument("--save-every-n-epochs", type=int, default=10,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--restart-checkpoint", type=str, default=None,
                        help="Path to checkpoint to restart from")
    
    args = parser.parse_args()
    
    # Convert restart checkpoint to Path if provided
    restart_checkpoint = Path(args.restart_checkpoint) if args.restart_checkpoint else None
    
    # Run training
    run_training(
        name=args.name,
        data_files=args.data_files,
        num_train=args.num_train,
        num_valid=args.num_valid,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        n_dcm=args.n_dcm,
        features=args.features,
        max_degree=args.max_degree,
        num_iterations=args.num_iterations,
        esp_w=args.esp_w,
        chg_w=args.chg_w,
        use_grad_clip=args.use_grad_clip,
        grad_clip_norm=args.grad_clip_norm,
        use_lr_schedule=args.use_lr_schedule,
        lr_schedule_type=args.lr_schedule_type,
        save_every_n_epochs=args.save_every_n_epochs,
        output_dir=args.output_dir,
        random_seed=args.random_seed,
        restart_checkpoint=restart_checkpoint,
        num_atoms=args.num_atoms,
    )


if __name__ == "__main__":
    main()

