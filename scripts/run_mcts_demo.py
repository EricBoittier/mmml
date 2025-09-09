#!/usr/bin/env python3
import os
import argparse
import time
import numpy as np

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

    # Downsample surface points for speed in demo if needed
    try:
        # Determine number of surface points
        if hasattr(esp_target, "shape"):
            n_points = int(batch["n_grid"])
            print(f"Downsampling surface points to {n_points}")
        else:
            n_points = len(esp_target)
        sample_size = min(4000, n_points)
        if sample_size < n_points:
            idx = np.random.choice(np.arange(n_points), size=sample_size, replace=False)
            if esp_target.ndim == 2:
                esp_target = esp_target[0, idx]
            else:
                esp_target = esp_target[idx]
            if vdw_surface.ndim == 3:
                vdw_surface = vdw_surface[0, idx, :]
            else:
                vdw_surface = vdw_surface[idx, :]
    except Exception as e:
        print(f"Warning: surface subsampling skipped due to: {e}")

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
    parser.add_argument('--neutrality-lambda', type=float, default=0.0001, help='Penalty weight for net charge neutrality')
    parser.add_argument('--dq-max', type=float, default=0.2, help='Max abs NN charge delta (e)')
    parser.add_argument('--dr-max', type=float, default=0.2, help='Max abs NN displacement delta (Angstrom)')
    parser.add_argument('--refine-steps', type=int, default=0, help='Post-move refinement steps on selected q')
    parser.add_argument('--refine-lr', type=float, default=2e-4, help='Learning rate for refinement steps')
    parser.add_argument('--alpha-e2e', type=float, default=0.0, help='Weight for NN end-to-end ESP loss (0 disables)')
    parser.add_argument('--nn-policy-only', action='store_true', help='Use NN only to shape priors; disable δq/δr corrections')
    parser.add_argument('--swap-per-atom-models', action='store_true', help='Restrict actions to swapping full model for a single atom')
    parser.add_argument('--excess-charge-penalty', type=float, default=0.0, help='Penalty weight for charges beyond target (per excess charge)')
    parser.add_argument('--selection-metric', type=str, default='loss', choices=['loss', 'rmse_per_charge'],
                        help='How to choose the best model among totals: raw loss or RMSE per charge')
    parser.add_argument('--run-dir', type=str, default=None, help='Directory to save per-run artifacts (weights, caches)')
    parser.add_argument('--accept-only-better', action='store_true', help='Only accept a step if loss does not increase')
    parser.add_argument('--mcts-c-puct', type=float, default=None, help='Override MCTS c_puct')
    parser.add_argument('--mcts-root-noise-frac', type=float, default=None, help='Override MCTS root noise fraction')
    parser.add_argument('--mcts-dirichlet-alpha', type=float, default=None, help='Override MCTS dirichlet alpha')
    args = parser.parse_args()


    print(args)

    if args.seed is not None:
        np.random.seed(args.seed)

    molecular_data, esp_target, vdw_surface, model_charges, model_positions = create_example_data_with_positions()

    t0 = time.time()
    print("=== DCMNET Optimizing with MCTS ===")
    # Default run dir if not provided
    run_dir = args.run_dir
    if run_dir is None and args.save_prefix:
        ts = time.strftime('%Y%m%d_%H%M%S')
        base = os.path.splitext(os.path.basename(args.save_prefix))[0]
        run_dir = os.path.join(os.path.dirname(args.save_prefix), f"{base}_{ts}")
        os.makedirs(run_dir, exist_ok=True)

    selection, loss, best_by_total = optimize_dcmnet_combination(
        molecular_data=molecular_data,
        esp_target=esp_target,
        vdw_surface=vdw_surface,
        model_charges=model_charges,
        model_positions=model_positions,
        n_simulations=args.n_simulations,
        temperature=args.temperature,
        target_total_selected=args.target_total,
        target_span=args.target_span,
        verbose=args.verbose,
        neutrality_lambda=args.neutrality_lambda,
        dq_max=args.dq_max,
        dr_max=args.dr_max,
        refine_steps=args.refine_steps,
        refine_lr=args.refine_lr,
        alpha_e2e=args.alpha_e2e,
        nn_policy_only=args.nn_policy_only,
        excess_charge_penalty=args.excess_charge_penalty,
        accept_only_better=args.accept_only_better,
        mcts_c_puct=args.mcts_c_puct,
        mcts_root_noise_frac=args.mcts_root_noise_frac,
        mcts_dirichlet_alpha=args.mcts_dirichlet_alpha,
        selection_metric=args.selection_metric,
        run_dir=run_dir,
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

        def write_xyzq(path, sel_matrix, total_charges, tag):
            """Write XYZQ format: positions as atoms, charges as comments"""
            with open(path, 'w') as f:
                f.write(f"{total_charges}\n")
                f.write(f"# Best model with {total_charges} charges (tag={tag})\n")
                for atom_idx in range(sel_matrix.shape[0]):
                    for gidx in np.where(sel_matrix[atom_idx])[0]:
                        model_id, cwm = charge_mapping[gidx]
                        pos = np.array(model_positions[model_id][atom_idx, cwm])
                        val = float(np.array(model_charges[model_id][atom_idx, cwm]))
                        x, y, z = pos
                        f.write(f"Q {x:.8f} {y:.8f} {z:.8f} # charge={val:.8f} model={model_id} atom={atom_idx}\n")

        # Write best-by-total-charges XYZQ files
        print("=== DCMNET Saving Best-by-Total XYZQ Files ===")
        for total_charges, (sel_matrix, loss_val) in best_by_total.items():
            basename = os.path.basename(prefix)
            xyzq_path = f"{prefix}tag{total_charges}DCM.xyz"
            write_xyzq(xyzq_path, sel_matrix, total_charges, total_charges)
            print(f"Saved {xyzq_path} (loss={loss_val:.6f})")

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
            
            print("=== DCMNET Computing ESP Prediction ===")
            print(f"N charges: {charge_values.shape[0]}")
            print(f"Sum of charge values: {charge_values.sum()}")
            print(f"Charge values: {charge_values}")
            print(f"Charge positions: {charge_positions}")

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

        print("=== DCMNET Saving Error Info ===")
        print(f"Loss: {loss}")
        # rmse
        rmse = np.sqrt(np.mean(residuals**2))
        print(f"RMSE: {rmse}")

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
        # Also save into run_dir if available
        np.save(f"{prefix}_error.npy", error_info, allow_pickle=True)
        if run_dir is not None:
            np.save(os.path.join(run_dir, "error.npy"), error_info, allow_pickle=True)
            # copy xyz files to run_dir
            try:
                import shutil
                for total_charges in best_by_total.keys():
                    xyzq_path = f"{prefix}tag{total_charges}DCM.xyz"
                    if os.path.exists(xyzq_path):
                        shutil.copy2(xyzq_path, os.path.join(run_dir, os.path.basename(xyzq_path)))
            except Exception as e:
                print(f"Warning: failed to copy XYZQ files to run_dir: {e}")
        print(f"Saved XYZ and error info with prefix: {prefix}")


if __name__ == '__main__':
    main()
