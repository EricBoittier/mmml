#!/usr/bin/env python3
import os
import argparse
import time
import numpy as np

# Configure GPU visibility before importing JAX (effective on process start)
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")  # expose both GPUs by default

import jax
devices = jax.local_devices()
print(devices)
print(jax.default_backend())
print(jax.devices())


import jax.numpy as jnp
import jax
import e3x
import os
from mmml.dcmnet.dcmnet_mcts import optimize_dcmnet_combination, DCMNETSelectionEnv
from mmml.dcmnet.dcmnet.analysis import dcmnet_analysis, prepare_batch
from mmml.dcmnet.dcmnet.data import prepare_datasets
from mmml.dcmnet.dcmnet.utils import apply_model
from mmml.dcmnet.dcmnet.models import models, model_params
key = jax.random.PRNGKey(0)

def create_example_data_with_positions():
    num_atoms = 60
    index = 0
    data_path_resolved = "/home/ericb/esp2000.npz"
    data_loaded = np.load(data_path_resolved, 
    allow_pickle=True)
    train_data, valid_data = prepare_datasets(
        key, num_train=1200, num_valid=100,
        filename=[data_path_resolved],
        clean= False, esp_mask=True,
        # clip_esp=True,
    )
    
    batch = prepare_batch(data_path_resolved, index=9)

    def add_to_batch(batch):
        """ add some useful (dummy) quantities to the batch """
        batch['com'] = np.mean(batch['R'].T, axis=-1) # center of mass
        batch["Dxyz"] = batch["R"] - batch["com"]
        # add 'dst_idx', 'src_idx', 'batch_segments',
        if "dst_idx" not in batch:
            dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(num_atoms)
            batch["dst_idx"] = dst_idx
        if "src_idx" not in batch:
            dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(num_atoms)
            batch["src_idx"] = src_idx
        if "batch_segments" not in batch:
            batch["batch_segments"] = np.zeros_like(batch["Z"])
        return batch

    batch = add_to_batch(batch)

    model_charges = {}
    model_positions = {}
    cached_model_charges = os.path.join(os.path.dirname(__file__), "model_charges.npy")
    cached_model_positions = os.path.join(os.path.dirname(__file__), "model_positions.npy")
    if os.path.exists(cached_model_charges) and os.path.exists(cached_model_positions):
        print("=== DCMNET Loading Model Charges and Positions ===")
        model_charges = np.load(cached_model_charges, allow_pickle=True).tolist()
        model_positions = np.load(cached_model_positions, allow_pickle=True).tolist()
    else:
        print("=== DCMNET Creating Model Charges and Positions ===")
        for _model, _params in zip(models, model_params):
            _model.features = 64
            mono, dipo = apply_model(_model, _params, batch, 1)

            dipo = dipo.reshape(60, mono.shape[-1], 3)
            mono = mono.squeeze()[...,None]
            model_charges[_model.n_dcm] = mono
            model_positions[_model.n_dcm] = dipo
        np.save(cached_model_charges, model_charges)
        np.save(cached_model_positions, model_positions)

    molecular_data = batch
    molecular_data["atomic_numbers"] = batch["Z"]
    esp_target = batch["esp"]
    vdw_surface = batch["vdw_surface"]

    print("=== DCMNET Preparing Data ===")
    return molecular_data, esp_target, vdw_surface, model_charges, model_positions


def main():
    parser = argparse.ArgumentParser(description="Run DCMNET MCTS optimization (fast test script)")
    parser.add_argument('--n-simulations', type=int, default=100, help='MCTS simulations per step')
    parser.add_argument('--temperature', type=float, default=1.0, help='Action selection temperature')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--verbose', action='store_true', help='Print progress')
    parser.add_argument('--target-total', type=int, default=None, help='Target total number of charges to select')
    parser.add_argument('--target-span', type=int, default=2, help='Try target±span and pick best')
    parser.add_argument('--save-prefix', type=str, default=None, help='Prefix path to save XYZ and NPY outputs')
    args = parser.parse_args()

    if args.verbose:
        jax.config.update("jax_debug_nans", True)
        jax.config.update("jax_debug_infs", True)
        # jax.config.update("jax_debug_asserts", True)
        print("Running in verbose mode")
        print("=== JAX Debug Mode Enabled ===")


    if args.seed is not None:
        np.random.seed(args.seed)

    molecular_data, esp_target, vdw_surface, model_charges, model_positions = create_example_data_with_positions()

    t0 = time.time()
    print("=== DCMNET Optimizing with MCTS ===")
    selection, loss = optimize_dcmnet_combination(
        molecular_data=molecular_data,
        esp_target=esp_target,
        vdw_surface=vdw_surface,
        model_charges=model_charges,
        model_positions=model_positions,
        n_simulations=args.n_simulations,
        temperature=args.temperature,
        target_total_selected=args.target_total,
        target_span=args.target_span,
    )
    dt = time.time() - t0

    print("=== DCMNET MCTS Test ===")
    print(f"Time: {dt:.2f}s | Loss: {loss}")
    print(f"Selection shape: {selection.shape} | Total selected: {int(selection.sum())}")

    # Optionally save XYZ of selected charges and error info
    if args.save_prefix:
        prefix = args.save_prefix
        os.makedirs(os.path.dirname(prefix) or '.', exist_ok=True)

        # Build a temp env to access mapping and utilities
        from mmml.dcmnet.dcmnet_mcts import DCMNETSelectionEnv
        temp_env = DCMNETSelectionEnv(molecular_data, esp_target, vdw_surface, model_charges, model_positions)
        charge_mapping = temp_env.charge_mapping

        # Gather selected charges: positions and values
        selected_positions = []
        selected_values = []
        selected_indices = []  # (atom_idx, global_charge_idx, model_id, charge_within_model)
        per_model = {}
        for atom_idx in range(selection.shape[0]):
            for gidx in np.where(selection[atom_idx])[0]:
                model_id, cwm = charge_mapping[gidx]
                pos = np.array(model_positions[model_id][atom_idx, cwm])
                val = float(np.array(model_charges[model_id][atom_idx, cwm]))
                selected_positions.append(pos)
                selected_values.append(val)
                selected_indices.append((int(atom_idx), int(gidx), int(model_id), int(cwm)))
                per_model.setdefault(int(model_id), []).append((pos, val, atom_idx, gidx, cwm))

        def write_xyz(path, items):
            # items: list of (pos3, value, meta...)
            with open(path, 'w') as f:
                f.write(f"{len(items)}\n")
                f.write("Q atoms: x y z charge (model_id atom_idx charge_idx)\n")
                for pos, val, aidx, gidx, cwm, mid in items:
                    x, y, z = pos
                    f.write(f"Q {x:.8f} {y:.8f} {z:.8f} {val:.8f} {mid} {aidx} {gidx}:{cwm}\n")

        # Write combined selected charges XYZ
        combined_items = [(p, v, aidx, gidx, cwm, mid) for mid, lst in per_model.items() for (p, v, aidx, gidx, cwm) in lst]
        write_xyz(f"{prefix}_selected.xyz", combined_items)

        # Write per-model selected charges XYZ
        for mid, lst in per_model.items():
            items = [(p, v, aidx, gidx, cwm, mid) for (p, v, aidx, gidx, cwm) in lst]
            write_xyz(f"{prefix}_model{mid+1}_selected.xyz", items)

        # Compute ESP prediction for error info
        if selected_positions:
            charge_values = jnp.array([float(v) for v in selected_values])
            charge_positions = jnp.array(selected_positions)
            esp_pred = temp_env._calculate_esp_from_distributed_charges(
                charge_values, charge_positions, temp_env.vdw_surface)
            esp_pred_np = np.array(esp_pred)
            esp_target_np = np.array(temp_env.esp_target)
            residuals = esp_pred_np - esp_target_np
        else:
            esp_pred_np = np.zeros_like(np.array(temp_env.esp_target))
            residuals = esp_pred_np - np.array(temp_env.esp_target)

        # Serialize mapping and index map for traceability
        charge_mapping_list = [(int(gidx), int(mid), int(cwm)) for gidx, (mid, cwm) in charge_mapping.items()]
        atom_index_map = np.array(getattr(temp_env, 'atom_index_map', np.arange(selection.shape[0])), dtype=int)

        error_info = {
            'loss': float(loss),
            'esp_target': np.array(temp_env.esp_target),
            'esp_pred': esp_pred_np,
            'residuals': residuals,
            'selected_indices': np.array(selected_indices, dtype=int),
            'charge_mapping': charge_mapping_list,
            'atom_index_map': atom_index_map,
            'charge_values': np.array(selected_values, dtype=float),
            'charge_positions': np.array(selected_positions, dtype=float),
            'initial_esp_target': np.array(temp_env.esp_target, dtype=float),
            'initial_vdw_surface': np.array(temp_env.vdw_surface, dtype=float),
            # note: model arrays and molecular_data omitted here to avoid serialization issues
            'initial_esp_target': np.array(temp_env.esp_target, dtype=float),
            'initial_vdw_surface': np.array(temp_env.vdw_surface, dtype=float),
        }
        np.save(f"{prefix}_error.npy", error_info, allow_pickle=True)
        print(f"Saved XYZ and error info with prefix: {prefix}")


if __name__ == '__main__':
    main()
