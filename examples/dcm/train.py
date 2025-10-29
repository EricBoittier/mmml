"""Train DCMNet model with Hydra configuration."""

import os
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import hydra
from omegaconf import DictConfig, OmegaConf
import logging

from mmml.dcmnet.dcmnet.training import train_model
from mmml.dcmnet.dcmnet.analysis import dcmnet_analysis, prepare_batch
from mmml.dcmnet.dcmnet.data import prepare_batches

from utils import (
    setup_environment,
    prepare_data,
    create_model,
    get_loss_weight,
    print_batch_info,
    setup_wandb,
)

log = logging.getLogger(__name__)


def prepare_batch_for_analysis(data, index=0, natoms=18):
    """Prepare a single batch correctly for dcmnet_analysis."""
    # Extract single item but keep batch dimension
    _dict = {k: np.array(v[[index]]) for k, v in data.items()}
    
    # Use prepare_batches with include_id=False
    batch = prepare_batches(
        jax.random.PRNGKey(0), _dict, batch_size=1,
        include_id=False, num_atoms=natoms
    )[0]
    batch["com"] = np.array([0, 0, 0])
    batch["Dxyz"] = np.array([0, 0, 0])
    return batch


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> float:
    """
    Main training function.
    
    Args:
        cfg: Hydra configuration
    
    Returns:
        Final validation loss
    """
    # Print configuration
    log.info("Configuration:")
    log.info(OmegaConf.to_yaml(cfg))
    
    # Setup environment
    setup_environment(cfg)
    
    # Log JAX devices
    devices = jax.local_devices()
    log.info(f"JAX devices: {devices}")
    log.info(f"JAX backend: {jax.default_backend()}")
    
    # Create model
    model = create_model(cfg)
    log.info(f"Created model with {cfg.model.n_dcm} DCM components")
    
    # Initialize random key
    key = jax.random.PRNGKey(cfg.seed)
    
    # Setup experiment tracking
    wandb = setup_wandb(cfg)
    
    # Bootstrap training loop
    params = None
    n_bootstrap = cfg.training.n_bootstrap
    
    for bootstrap_idx in range(n_bootstrap):
        log.info(f"\n{'='*60}")
        log.info(f"Bootstrap iteration {bootstrap_idx + 1}/{n_bootstrap}")
        log.info(f"{'='*60}")
        
        # Create data key for this bootstrap iteration
        data_key = jax.random.PRNGKey(bootstrap_idx * cfg.seed)
        
        # Prepare datasets
        log.info("Preparing datasets...")
        train_data, valid_data = prepare_data(cfg, data_key, bootstrap_idx)
        
        # Print batch info for first iteration
        if bootstrap_idx == 0:
            batch = {k: v[0:1] if len(v.shape) > 0 else v for k, v in train_data.items()}
            print_batch_info(batch, model)
        
        # Calculate loss weights for this iteration
        esp_w = get_loss_weight(
            cfg.training.esp_weight_schedule,
            cfg.training.esp_weight_start,
            cfg.training.esp_weight_end,
            bootstrap_idx,
            n_bootstrap,
        )
        chg_w = get_loss_weight(
            cfg.training.chg_weight_schedule,
            cfg.training.chg_weight_start,
            cfg.training.chg_weight_end,
            bootstrap_idx,
            n_bootstrap,
        )
        
        log.info(f"Loss weights - ESP: {esp_w:.2f}, Charge: {chg_w:.4f}")
        
        # Train model
        log.info(f"Training for {cfg.training.num_epochs} epochs...")
        params, valid_loss = train_model(
            key=data_key,
            model=model,
            writer=None,
            train_data=train_data,
            valid_data=valid_data,
            num_epochs=cfg.training.num_epochs,
            learning_rate=cfg.training.learning_rate,
            batch_size=cfg.training.batch_size,
            restart_params=params,
            ndcm=model.n_dcm,
            esp_w=esp_w,
            chg_w=chg_w,
            use_grad_clip=cfg.training.use_grad_clip,
            grad_clip_norm=cfg.training.grad_clip_norm,
        )
        
        log.info(f"Bootstrap {bootstrap_idx + 1} - Validation loss: {valid_loss:.6f}")
        
        # Log to wandb if enabled
        if wandb is not None:
            wandb.log({
                "bootstrap_iteration": bootstrap_idx,
                "valid_loss": valid_loss,
                "esp_weight": esp_w,
                "chg_weight": chg_w,
            })
        
        # Evaluate on first training example
        batch = prepare_batch_for_analysis(train_data, index=0, natoms=cfg.data.natoms)
        output = dcmnet_analysis(params, model, batch, cfg.data.natoms)
        
        log.info(f"  RMSE: {output['rmse_model']:.6f}")
        log.info(f"  RMSE (masked): {output['rmse_model_masked']:.6f}")
        
        if wandb is not None:
            wandb.log({
                "bootstrap_iteration": bootstrap_idx,
                "rmse_model": output['rmse_model'],
                "rmse_model_masked": output['rmse_model_masked'],
            })
    
    # Save final parameters
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    params_path = output_dir / "final_params.npz"
    
    log.info(f"\nSaving final parameters to: {params_path}")
    # Convert params to numpy for saving
    params_np = jax.tree_util.tree_map(np.array, params)
    np.savez(params_path, **params_np)
    
    # Save configuration
    config_path = output_dir / "config.yaml"
    with open(config_path, 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))
    
    log.info(f"Training complete! Output directory: {output_dir}")
    
    if wandb is not None:
        wandb.finish()
    
    return float(valid_loss)


if __name__ == "__main__":
    main()

