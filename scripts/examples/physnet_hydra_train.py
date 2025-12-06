#!/usr/bin/env python3
"""Hydra-based training entrypoint for PhysNetJAX.

Usage examples:
  - Default config with overrides
    python scripts/physnet_hydra_train.py \
      data.train_file=/path/train.npz \
      data.valid_file=/path/valid.npz \
      model.natoms=60 train.batch_size=32 train.max_epochs=200

  - Auto split validation
    python scripts/physnet_hydra_train.py \
      data.train_file=/path/dataset.npz data.train_fraction=0.9

  - Enable advanced batching
    python scripts/physnet_hydra_train.py batching.method=advanced \
      batching.batch_shape=512 batching.batch_nbl_len=16384
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf


def _ensure_standard_keys(data: Dict) -> Dict:
    """Massage loaded data to keys expected by training.

    - Ensure `N` exists (counts of atoms per structure)
    - Map `Dxyz` to `D` when only dipole vector is provided under that name
    """
    out = dict(data)

    if 'N' not in out and 'Z' in out:
        N = np.sum(np.asarray(out['Z']) > 0, axis=1)
        out['N'] = N.astype(np.int32)

    if 'D' not in out and 'Dxyz' in out:
        Dxyz = np.asarray(out['Dxyz'])
        # Accept (n, 3) directly; if (n, atoms, 3), reduce to total dipole per structure
        if Dxyz.ndim == 3:
            out['D'] = Dxyz.sum(axis=1)
        else:
            out['D'] = Dxyz

    return out


@hydra.main(config_path="conf", config_name="physnet", version_base="1.3")
def main(cfg: DictConfig) -> None:
    from mmml.data import load_npz, train_valid_split, DataConfig
    from mmml.physnetjax.physnetjax.models.model import EF
    from mmml.physnetjax.physnetjax.training.training import train_model
    from mmml.physnetjax.physnetjax.directories import BASE_CKPT_DIR
    import jax

    print("Config:\n" + OmegaConf.to_yaml(cfg))

    # Load data
    if not cfg.data.train_file:
        raise ValueError("data.train_file must be provided")

    dc = DataConfig(
        batch_size=cfg.train.batch_size,
        targets=cfg.train.targets,
        num_atoms=cfg.model.natoms,
        center_coordinates=cfg.data.center_coordinates,
        normalize_energy=cfg.data.normalize_energy,
        esp_mask_vdw=False,
    )
    train_raw = load_npz(cfg.data.train_file, config=dc, validate=True, verbose=cfg.train.verbose)

    if cfg.data.valid_file:
        valid_raw = load_npz(cfg.data.valid_file, config=dc, validate=True, verbose=cfg.train.verbose)
    else:
        train_raw, valid_raw = train_valid_split(
            train_raw, train_fraction=cfg.data.train_fraction, shuffle=True, seed=cfg.train.seed
        )

    train_data = _ensure_standard_keys(train_raw)
    valid_data = _ensure_standard_keys(valid_raw)

    # Build model
    model = EF(
        features=cfg.model.features,
        max_degree=cfg.model.max_degree,
        num_iterations=cfg.model.num_iterations,
        num_basis_functions=cfg.model.num_basis_functions,
        cutoff=cfg.model.cutoff,
        max_atomic_number=cfg.model.max_atomic_number,
        charges=cfg.model.charges,
        natoms=cfg.model.natoms,
        total_charge=cfg.model.total_charge,
        n_res=cfg.model.n_res,
        zbl=cfg.model.zbl,
        debug=cfg.model.debug,
        efa=cfg.model.efa,
    )

    # Training params
    key = jax.random.PRNGKey(cfg.train.seed)

    # Decide keys to pass to batches/train step
    data_keys = ["R", "Z", "F", "E", "N", "dst_idx", "src_idx", "batch_segments"]
    if cfg.model.charges and ("D" in train_data or "Dxyz" in train_data):
        data_keys.append("D")

    # Batch method configuration
    batch_method = None
    batch_args = None
    if cfg.batching.method == "advanced":
        batch_method = "advanced"
        batch_args = {
            "batch_shape": int(cfg.batching.batch_shape),
            "batch_nbl_len": int(cfg.batching.batch_nbl_len),
        }
    else:
        batch_method = None  # default batching in train_model

    # Kick off training
    ema_params = train_model(
        key=key,
        model=model,
        train_data=train_data,
        valid_data=valid_data,
        num_epochs=cfg.train.max_epochs,
        learning_rate=cfg.train.learning_rate,
        energy_weight=cfg.loss.energy_weight,
        forces_weight=cfg.loss.forces_weight,
        dipole_weight=cfg.loss.dipole_weight,
        charges_weight=cfg.loss.charges_weight,
        batch_size=cfg.train.batch_size,
        num_atoms=cfg.model.natoms,
        restart=cfg.train.restart,
        print_freq=cfg.logging.print_freq,
        name=cfg.logging.name,
        best=cfg.logging.save_best,
        optimizer=cfg.optimizer.type,
        transform=cfg.optimizer.transform,
        schedule_fn=cfg.optimizer.schedule,
        objective=cfg.train.objective,
        ckpt_dir=BASE_CKPT_DIR,
        log_tb=bool(cfg.logging.tensorboard),
        batch_method=batch_method,
        batch_args_dict=batch_args,
        data_keys=tuple(data_keys),
    )

    print("Training finished. Final EMA params obtained.")


if __name__ == "__main__":
    main()

