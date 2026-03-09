"""
Training configuration management for DCMNet.

Provides dataclasses for managing training hyperparameters, model
configuration, and experiment tracking.
"""
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, Any
import json
import time


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    n_dcm: int = 2
    features: int = 16
    max_degree: int = 2
    num_iterations: int = 2
    num_basis_functions: int = 16
    cutoff: float = 4.0
    max_atomic_number: int = 36
    include_pseudotensors: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """Create from dictionary."""
        return cls(**config_dict)


@dataclass
class TrainingConfig:
    """Training hyperparameters configuration."""
    # Training parameters
    num_epochs: int = 100
    batch_size: int = 1
    learning_rate: float = 1e-4
    num_atoms: int = 60
    
    # Loss weights
    esp_w: float = 1.0
    chg_w: float = 0.01
    
    # Optimization
    use_grad_clip: bool = True
    grad_clip_norm: float = 2.0
    use_ema: bool = True
    ema_decay: float = 0.999
    
    # Multi-batch training
    gradient_accumulation_steps: int = 1  # Number of batches to accumulate gradients over
    use_parallel_batches: bool = False     # Use vmap for parallel batch processing
    
    # Learning rate schedule
    use_lr_schedule: bool = False
    lr_schedule_type: str = "cosine"  # "cosine", "exponential", "step"
    warmup_epochs: int = 5
    min_lr_factor: float = 0.01
    
    # Checkpointing
    save_every_n_epochs: int = 10
    keep_top_k_checkpoints: int = 5
    
    # Logging
    log_every_n_batches: int = 10
    compute_full_stats_every_n_epochs: int = 1
    
    # Data
    num_train: int = 1000
    num_valid: int = 200
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create from dictionary."""
        return cls(**config_dict)


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""
    name: str = "dcmnet_experiment"
    description: str = ""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data_files: list = field(default_factory=lambda: ["esp2000.npz"])
    random_seed: int = 42
    output_dir: str = "./experiments"
    
    # Metadata
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    git_commit: Optional[str] = None
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'description': self.description,
            'model': self.model.to_dict(),
            'training': self.training.to_dict(),
            'data_files': self.data_files,
            'random_seed': self.random_seed,
            'output_dir': self.output_dir,
            'created_at': self.created_at,
            'git_commit': self.git_commit,
            'notes': self.notes
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        """Create from dictionary."""
        model_config = ModelConfig.from_dict(config_dict.get('model', {}))
        training_config = TrainingConfig.from_dict(config_dict.get('training', {}))
        
        return cls(
            name=config_dict.get('name', "dcmnet_experiment"),
            description=config_dict.get('description', ""),
            model=model_config,
            training=training_config,
            data_files=config_dict.get('data_files', ["esp2000.npz"]),
            random_seed=config_dict.get('random_seed', 42),
            output_dir=config_dict.get('output_dir', "./experiments"),
            created_at=config_dict.get('created_at', time.strftime("%Y-%m-%d %H:%M:%S")),
            git_commit=config_dict.get('git_commit', None),
            notes=config_dict.get('notes', "")
        )
    
    def save(self, path: Path):
        """Save configuration to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> 'ExperimentConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def get_experiment_dir(self) -> Path:
        """Get the experiment output directory."""
        exp_dir = Path(self.output_dir) / self.name
        exp_dir.mkdir(parents=True, exist_ok=True)
        return exp_dir


def create_default_config(
    name: str = "dcmnet_default",
    **kwargs
) -> ExperimentConfig:
    """
    Create a default experiment configuration with optional overrides.
    
    Parameters
    ----------
    name : str
        Experiment name
    **kwargs
        Additional arguments to override defaults
        
    Returns
    -------
    ExperimentConfig
        Configured experiment
    """
    config = ExperimentConfig(name=name)
    
    # Update with any provided kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        elif hasattr(config.model, key):
            setattr(config.model, key, value)
        elif hasattr(config.training, key):
            setattr(config.training, key, value)
    
    return config

