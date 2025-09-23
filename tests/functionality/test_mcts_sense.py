import numpy as np
import jax.numpy as jnp
import jax

from mmml.dcmnet.dcmnet_mcts import DCMNETSelectionEnv, CONVERSION_FACTOR, optimize_dcmnet_combination


def small_env():
    # 2 atoms, 1 candidate per atom, 3 surface points
    molecular_data = {
        'atomic_numbers': np.array([6, 1]),
        'positions': np.zeros((2, 3)),
        'dst_idx': np.array([0]),
        'src_idx': np.array([1]),
    }
    sp = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=float)
    vdw_surface = sp
    esp_target = np.zeros((3,), dtype=float)
    # candidates
    model_charges = {0: np.array([[1.0], [1.0]], dtype=float)}
    model_positions = {0: np.array([[[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]], dtype=float)}
    env = DCMNETSelectionEnv(molecular_data, esp_target, vdw_surface, model_charges, model_positions)
    return env


def small_data_for_opt():
    # Returns (molecular_data, esp_target, vdw_surface, model_charges, model_positions)
    env = small_env()
    molecular_data = env.molecular_data
    esp_target = env.esp_target
    vdw_surface = env.vdw_surface
    model_charges = {0: env.model_charges[0]}
    model_positions = {0: env.model_positions[0]}
    return molecular_data, esp_target, vdw_surface, model_charges, model_positions


def test_kernel_values():
    env = small_env()
    # Charges at 0 and 1 on x-axis, surface at 1,2,3
    K = env.K_surface_to_candidates  # (Ns, L)
    # candidate flats are [atom0, atom1]
    # distances row0 to candidate0 is 1, row0 to candidate1 is 0 (protected by epsilon)
    assert np.allclose(K[1, 0], 1.0 / (2.0 * CONVERSION_FACTOR), rtol=1e-6)


def test_jacobian_fd():
    env = small_env()
    J = env.J_surface_to_candidates  # (Ns, 3L)
    L = env.n_atoms * env.total_charges_per_atom
    # finite diff on candidate 0, x-direction
    eps = 1e-4
    sp = env.vdw_surface
    pos = env.candidate_positions.reshape(-1, 3)
    base = np.linalg.norm(sp - pos[0], axis=1)
    base_esp = 1.0 / (np.maximum(base, 1e-6) * CONVERSION_FACTOR)
    pos_pert = pos.copy()
    pos_pert[0, 0] += eps
    pert = np.linalg.norm(sp - pos_pert[0], axis=1)
    pert_esp = 1.0 / (np.maximum(pert, 1e-6) * CONVERSION_FACTOR)
    fd = (pert_esp - base_esp) / eps
    Jx = J[:, 0]  # x component of first candidate
    assert np.allclose(fd, Jx, rtol=1e-2, atol=1e-2)


def test_neutrality_penalty():
    env = small_env()
    # choose both charges, set unbalanced selection
    env.selected_charges[:] = 1
    env.overridden_charge_values[(0, 0)] = 1.0
    env.overridden_charge_values[(1, 0)] = 0.5
    env.neutrality_lambda = 0.0
    loss_no_penalty = env.get_esp_loss()
    # amplify lambda so difference is visible
    env.neutrality_lambda = 1e6
    loss_with_penalty = env.get_esp_loss()
    assert loss_with_penalty > loss_no_penalty


def test_clipping_projection():
    env = small_env()
    env.nn_dq = np.ones(env.n_atoms * env.total_charges_per_atom, dtype=np.float32) * 1.0
    env.nn_dr = np.ones(env.n_atoms * env.total_charges_per_atom * 3, dtype=np.float32) * 1.0
    env.dq_max = 0.05
    env.dr_max = 0.1 / CONVERSION_FACTOR
    env.selected_charges[:] = 1
    loss = env.get_esp_loss()
    assert np.isfinite(loss)


def test_fastloss_consistency():
    env = small_env()
    env.selected_charges[:] = 1
    base = env.get_esp_loss()
    # small symmetric perturbations should not explode
    env.nn_dq = np.zeros(env.n_atoms * env.total_charges_per_atom, dtype=np.float32)
    env.nn_dr = np.zeros(env.n_atoms * env.total_charges_per_atom * 3, dtype=np.float32)
    env.dq_max = 0.05
    env.dr_max = 0.1 / CONVERSION_FACTOR
    l2 = env.get_esp_loss()
    assert np.isfinite(l2)


def test_ablation_baseline_no_nn_not_worse():
    md, tgt, sp, mchg, mpos = small_data_for_opt()
    # initial mono loss
    env0 = DCMNETSelectionEnv(md, tgt, sp, mchg, mpos)
    init_loss = env0.get_esp_loss()
    sel, loss, _ = optimize_dcmnet_combination(
        molecular_data=md,
        esp_target=tgt,
        vdw_surface=sp,
        model_charges=mchg,
        model_positions=mpos,
        verbose=False,
        n_simulations=50,
        temperature=0.5,
        target_total_selected=env0.n_atoms,
        target_span=0,
        neutrality_lambda=0.0,
        dq_max=0.0,
        dr_max=0.0,
    )
    assert np.isfinite(loss)
    assert loss <= init_loss + 1e-8


def test_ablation_neutrality_only_penalty_applies():
    env = small_env()
    env.neutrality_lambda = 1.0
    env.selected_charges[:] = 1
    env.overridden_charge_values[(0, 0)] = 1.0
    env.overridden_charge_values[(1, 0)] = 0.5
    penalized = env.get_esp_loss()
    env.neutrality_lambda = 0.0
    unpenalized = env.get_esp_loss()
    assert penalized >= unpenalized


def test_ablation_corrections_only_runs_finite():
    md, tgt, sp, mchg, mpos = small_data_for_opt()
    env0 = DCMNETSelectionEnv(md, tgt, sp, mchg, mpos)
    init_loss = env0.get_esp_loss()
    sel, loss, _ = optimize_dcmnet_combination(
        molecular_data=md,
        esp_target=tgt,
        vdw_surface=sp,
        model_charges=mchg,
        model_positions=mpos,
        verbose=False,
        n_simulations=50,
        temperature=0.5,
        target_total_selected=env0.n_atoms,
        target_span=0,
        neutrality_lambda=0.0,
        dq_max=0.1,
        dr_max=0.0,
    )
    assert np.isfinite(loss)
    # shouldn't be much worse than init
    assert loss <= init_loss * 2.0


def test_invariant_no_empty_atoms_after_opt():
    md, tgt, sp, mchg, mpos = small_data_for_opt()
    sel, loss, _ = optimize_dcmnet_combination(
        molecular_data=md,
        esp_target=tgt,
        vdw_surface=sp,
        model_charges=mchg,
        model_positions=mpos,
        verbose=False,
        n_simulations=50,
        temperature=0.5,
        target_total_selected=md['atomic_numbers'].size,  # target = n_atoms
        target_span=0,
        neutrality_lambda=0.0,
        dq_max=0.0,
        dr_max=0.0,
    )
    assert sel.shape[0] == md['atomic_numbers'].size
    per_atom = sel.sum(axis=1)
    assert np.all(per_atom > 0)


def test_target_enforcement_exact_total():
    # Build an env with 2 candidates per atom so we can exceed n_atoms
    env_base = small_env()
    md = env_base.molecular_data
    tgt = env_base.esp_target
    sp = env_base.vdw_surface
    # two candidates per atom: duplicate with shifted position
    mchg = {0: np.stack([env_base.model_charges[0].squeeze(), env_base.model_charges[0].squeeze()], axis=1)}
    mpos = {0: np.stack([env_base.model_positions[0].squeeze(), env_base.model_positions[0].squeeze() + np.array([0.5,0,0])], axis=1)}
    env0 = DCMNETSelectionEnv(md, tgt, sp, mchg, mpos)
    target = env0.n_atoms + 1
    sel, loss, _ = optimize_dcmnet_combination(
        molecular_data=md,
        esp_target=tgt,
        vdw_surface=sp,
        model_charges=mchg,
        model_positions=mpos,
        verbose=False,
        n_simulations=200,
        temperature=0.5,
        target_total_selected=target,
        target_span=0,
        neutrality_lambda=0.0,
        dq_max=0.0,
        dr_max=0.0,
    )
    assert int(sel.sum()) == target


def test_units_scaling():
    env = small_env()
    # choose a surface far from candidates to avoid near-zero distances
    sp = np.array([[10.0,0,0],[20.0,0,0],[30.0,0,0]], dtype=float)
    md = env.molecular_data
    mchg = {0: env.model_charges[0]}
    mpos = {0: env.model_positions[0]}
    envA = DCMNETSelectionEnv(md, env.esp_target, sp, mchg, mpos)
    K1 = envA.K_surface_to_candidates.copy()
    s = 2.0
    # scale both candidate positions and surface by s so all distances scale by s
    mpos_s = {0: mpos[0] * s}
    envB = DCMNETSelectionEnv(md, env.esp_target, sp * s, mchg, mpos_s)
    K2 = envB.K_surface_to_candidates
    ratio = K1 / K2
    # K = 1/(r*CF), so if r scales by s, K scales by 1/s everywhere
    assert np.allclose(ratio, s, rtol=1e-2, atol=1e-2)


