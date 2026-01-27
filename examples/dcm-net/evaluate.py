"""Evaluate trained DCMNet model and generate visualizations."""

import os
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import matplotlib.pyplot as plt
import patchworklib as pw

from mmml.dcmnet.dcmnet.analysis import dcmnet_analysis
from mmml.dcmnet.dcmnet.data import prepare_batches

from utils import (
    setup_environment,
    prepare_data,
    create_model,
)

log = logging.getLogger(__name__)


def prepare_batch_for_analysis(data, index=0, natoms=18):
    """Prepare a single batch correctly for dcmnet_analysis."""
    _dict = {k: np.array(v[[index]]) for k, v in data.items()}
    batch = prepare_batches(
        jax.random.PRNGKey(0), _dict, batch_size=1,
        include_id=False, num_atoms=natoms
    )[0]
    batch["com"] = np.array([0, 0, 0])
    batch["Dxyz"] = np.array([0, 0, 0])
    return batch


def plot_esp_comparison(batch, output, output_dir: Path, index: int = 0):
    """
    Create ESP comparison plots.
    
    Args:
        batch: Input batch data
        output: Model output from dcmnet_analysis
        output_dir: Directory to save plots
        index: Index to plot
    """
    VMAX = 0.001
    Npoints = 1000
    
    # Create figure with patchworklib
    xy_ax = pw.Brick()
    xy_ax.scatter(batch["esp"], output['esp_pred'], s=1)
    max_val = np.sqrt(max(np.max(batch["esp"]**2), np.max(output['esp_pred']**2)))
    xy_ax.plot(np.linspace(-max_val, max_val, 100), np.linspace(-max_val, max_val, 100))
    xy_ax.set_aspect('equal')
    xy_ax.set_xlabel('True ESP')
    xy_ax.set_ylabel('Predicted ESP')
    xy_ax.set_title('ESP Correlation')
    
    # True ESP spatial plot
    ax_true = pw.Brick()
    vdw_surface_min = np.min(batch["vdw_surface"][0], axis=-1)
    vdw_surface_max = np.max(batch["vdw_surface"][0], axis=-1)
    
    ax_true.scatter(
        batch["vdw_surface"][0][:Npoints, 0],
        batch["vdw_surface"][0][:Npoints, 1],
        c=batch["esp"][0][:Npoints],
        s=0.01,
        vmin=-VMAX, vmax=VMAX
    )
    ax_true.set_aspect('equal')
    ax_true.set_title('True ESP')
    
    # Predicted ESP spatial plot
    ax_pred = pw.Brick()
    ax_pred.scatter(
        batch["vdw_surface"][0][:Npoints, 0],
        batch["vdw_surface"][0][:Npoints, 1],
        c=output['esp_pred'][:Npoints],
        s=0.01,
        vmin=-VMAX, vmax=VMAX
    )
    ax_pred.set_aspect('equal')
    ax_pred.set_title('Predicted ESP')
    
    # Difference plot
    ax_diff = pw.Brick()
    ax_diff.scatter(
        batch["vdw_surface"][0][:Npoints, 0],
        batch["vdw_surface"][0][:Npoints, 1],
        c=batch["esp"][0][:Npoints] - output['esp_pred'][:Npoints],
        s=0.01,
        vmin=-VMAX, vmax=VMAX
    )
    ax_diff.set_aspect('equal')
    ax_diff.set_title('Difference')
    
    # Set consistent axis limits
    for ax in [ax_pred, ax_true, ax_diff]:
        ax.set_xlim(vdw_surface_min[0], -vdw_surface_min[0])
        ax.set_ylim(vdw_surface_min[1], -vdw_surface_min[0])
    
    # Combine plots
    fig = xy_ax | ax_pred | ax_true | ax_diff
    
    # Save
    fig.savefig(output_dir / f"esp_comparison_{index}.png", dpi=300, bbox_inches='tight')
    log.info(f"Saved ESP comparison plot to {output_dir / f'esp_comparison_{index}.png'}")


def plot_monopoles(output, batch, model, output_dir: Path, index: int = 0):
    """
    Plot monopole charges.
    
    Args:
        output: Model output from dcmnet_analysis
        batch: Input batch data
        model: Model instance
        output_dir: Directory to save plots
        index: Index to plot
    """
    n_atoms = int(batch["N"])
    
    # Monopole matrix
    charge_ax = pw.Brick()
    charge_ax.matshow(output["mono"][0][:n_atoms], vmin=-1, vmax=1)
    charge_ax.set_title('Monopoles')
    
    # Sum of monopoles per atom
    scharge_ax = pw.Brick()
    scharge_ax.matshow(output["mono"][0][:n_atoms].sum(axis=-1)[:, None], vmin=-1, vmax=1)
    scharge_ax.axis("off")
    scharge_ax.set_title('Total Charge')
    
    # Combine
    f = (scharge_ax | charge_ax)
    f.add_colorbar(vmin=-1, vmax=1)
    
    # Save
    f.savefig(output_dir / f"monopoles_{index}.png", dpi=300, bbox_inches='tight')
    log.info(f"Saved monopole plot to {output_dir / f'monopoles_{index}.png'}")


def evaluate_dataset(params, model, data, cfg: DictConfig, output_dir: Path, name: str = "train"):
    """
    Evaluate model on entire dataset.
    
    Args:
        params: Model parameters
        model: Model instance
        data: Dataset dictionary
        cfg: Configuration
        output_dir: Directory to save results
        name: Dataset name (for logging)
    
    Returns:
        Dictionary of metrics
    """
    log.info(f"\nEvaluating on {name} dataset...")
    
    rmse_list = []
    rmse_masked_list = []
    
    # Evaluate on all samples
    n_samples = min(len(data["Z"]), 100)  # Limit to 100 samples for speed
    
    for i in range(n_samples):
        batch = prepare_batch_for_analysis(data, index=i, natoms=cfg.data.natoms)
        output = dcmnet_analysis(params, model, batch, cfg.data.natoms)
        
        rmse_list.append(output['rmse_model'])
        rmse_masked_list.append(output['rmse_model_masked'])
    
    metrics = {
        f"{name}_rmse_mean": np.mean(rmse_list),
        f"{name}_rmse_std": np.std(rmse_list),
        f"{name}_rmse_masked_mean": np.mean(rmse_masked_list),
        f"{name}_rmse_masked_std": np.std(rmse_masked_list),
    }
    
    log.info(f"{name.upper()} Results:")
    log.info(f"  RMSE: {metrics[f'{name}_rmse_mean']:.6f} ± {metrics[f'{name}_rmse_std']:.6f}")
    log.info(f"  RMSE (masked): {metrics[f'{name}_rmse_masked_mean']:.6f} ± {metrics[f'{name}_rmse_masked_std']:.6f}")
    
    # Save metrics
    np.savez(output_dir / f"{name}_metrics.npz", **metrics)
    
    return metrics


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main evaluation function.
    
    Args:
        cfg: Hydra configuration
    """
    # Get checkpoint path from command line or use default
    # Usage: python evaluate.py checkpoint_path=/path/to/checkpoint/final_params.npz
    if 'checkpoint_path' not in cfg:
        log.error("Please provide checkpoint_path. Usage: python evaluate.py checkpoint_path=/path/to/checkpoint")
        return
    
    checkpoint_path = Path(cfg.checkpoint_path)
    if not checkpoint_path.exists():
        log.error(f"Checkpoint not found: {checkpoint_path}")
        return
    
    log.info(f"Loading checkpoint from: {checkpoint_path}")
    
    # Setup environment
    setup_environment(cfg)
    
    # Create model
    model = create_model(cfg)
    log.info(f"Created model with {cfg.model.n_dcm} DCM components")
    
    # Load parameters
    params_data = np.load(checkpoint_path, allow_pickle=True)
    # Convert loaded numpy arrays back to JAX arrays, handling nested structures
    params = jax.tree_util.tree_map(jnp.array, dict(params_data))
    log.info("Loaded model parameters")
    
    # Prepare datasets
    data_key = jax.random.PRNGKey(cfg.seed)
    train_data, valid_data = prepare_data(cfg, data_key, bootstrap_idx=0)
    
    # Get output directory
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    log.info(f"Output directory: {output_dir}")
    
    # Evaluate on training set
    train_metrics = evaluate_dataset(params, model, train_data, cfg, output_dir, "train")
    
    # Evaluate on validation set
    valid_metrics = evaluate_dataset(params, model, valid_data, cfg, output_dir, "valid")
    
    # Generate visualizations for first few examples
    log.info("\nGenerating visualizations...")
    for i in range(min(3, len(train_data["Z"]))):
        batch = prepare_batch_for_analysis(train_data, index=i, natoms=cfg.data.natoms)
        output = dcmnet_analysis(params, model, batch, cfg.data.natoms)
        
        plot_esp_comparison(batch, output, output_dir, index=i)
        plot_monopoles(output, batch, model, output_dir, index=i)
    
    log.info(f"\nEvaluation complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()

