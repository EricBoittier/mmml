"""JAX-MD simulation setup and Nose-Hoover chain routines."""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import jax
import jax_md
import numpy as np
from jax import grad, jit, lax
from jax_md import quantity, simulate, space, units
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import jax.numpy as jnp

from mmml.cli.run.summaries import print_forces_summary
from mmml.interfaces.pycharmmInterface.pbc_utils_jax import wrap_groups
from mmml.utils.hdf5_reporter import make_jaxmd_reporter

import ase.io as ase_io
WORSE_COUNT_THRESHOLD = 100

def default_nhc_kwargs(tau, overrides=None):
    """Build Nose-Hoover chain kwargs dict with sensible defaults.

    Args:
        tau: Thermostat coupling timescale (typically ``nhc_tau * dt``).
        overrides: Optional dict to override individual defaults.

    Returns:
        Dict with keys ``chain_length``, ``chain_steps``, ``sy_steps``, ``tau``.
    """
    default_kwargs = {
        'chain_length': 3,
        'chain_steps': 2,
        'sy_steps': 3,
        'tau': tau,
    }
    if overrides is None:
        return default_kwargs
    return {k: overrides.get(k, default_kwargs[k]) for k in default_kwargs}


def _run_npt_diagnostics(
    *,
    state,
    npt_energy_fn,
    jax_md_force_fn,
    apply_fn,
    shift,
    space,
    simulate,
    quantity,
    npt_pair_idx,
    npt_pair_mask,
    npt_pressure,
    unit,
    dt,
    kT,
    grad,
):
    """Run NPT diagnostic tests to locate instabilities. Call with --npt-diagnose."""
    neighbor = (npt_pair_idx, npt_pair_mask)
    box_curr = simulate.npt_box(state)
    R = state.position
    P = state.momentum
    M = state.mass
    N, dim = R.shape

    c = Console()

    # 1. Energy and forces sanity
    E0 = float(npt_energy_fn(R, box=box_curr, neighbor=neighbor))
    real_pos = space.transform(box_curr, R)
    F_calc = jax_md_force_fn(real_pos, mm_pair_idx=npt_pair_idx, mm_pair_mask=npt_pair_mask, box=box_curr)
    F_grad = -grad(lambda r: npt_energy_fn(r, box=box_curr, neighbor=neighbor))(R)
    print_forces_summary(np.asarray(F_calc), energy_eV=E0, console=c)
    t1 = Table(title="[1] Force consistency")
    t1.add_column("Check", style="cyan")
    t1.add_column("Value", style="white")
    t1.add_row("max|F_calc|", f"{float(np.max(np.abs(F_calc))):.6f}")
    t1.add_row("max|F_grad|", f"{float(np.max(np.abs(F_grad))):.6f}")
    t1.add_row("F_calc finite", str(np.all(np.isfinite(F_calc))))
    t1.add_row("F_grad finite", str(np.all(np.isfinite(F_grad))))
    c.print(Panel(t1, title="NPT Diagnostic [1]", border_style="blue"))

    # 2. Perturbation / stress (dUdV)
    vol = float(quantity.volume(dim, box_curr))
    eps_vals = [0.0, 1e-6, 1e-5, 1e-4]
    t2 = Table(title="[2] Stress (dU/dV) via perturbation")
    t2.add_column("ε", style="cyan")
    t2.add_column("E (eV)", style="white")
    for eps in eps_vals:
        pert = 1.0 + eps
        E_pert = float(npt_energy_fn(R, box=box_curr, neighbor=neighbor, perturbation=pert))
        t2.add_row(f"{eps:.0e}", f"{E_pert:.6f}")
    dE = float(npt_energy_fn(R, box=box_curr, neighbor=neighbor, perturbation=1.0 + 1e-5)) - E0
    dUdV_fd = dE / (vol * 1e-5)  # finite-diff approx
    t2.add_row("dUdV (finite diff)", f"{dUdV_fd:.4f} eV/Å³")
    t2.add_row("volume", f"{vol:.2f} Å³")
    c.print(Panel(t2, title="NPT Diagnostic [2]", border_style="blue"))

    # 3. Shift function with fractional R and Cartesian dR
    dR_cart = dt * (P / M)  # small Cartesian displacement
    R_shifted = shift(R, dR_cart, box=box_curr)
    in_cube = np.all((R_shifted >= 0) & (R_shifted < 1.001))
    t3 = Table(title="[3] Shift function")
    t3.add_column("Check", style="cyan")
    t3.add_column("Value", style="white")
    t3.add_row("R_shifted in [0,1)³", str(in_cube))
    t3.add_row("R_shifted finite", str(np.all(np.isfinite(R_shifted))))
    t3.add_row("R_shifted sample [0]", str(np.asarray(R_shifted[0])))
    c.print(Panel(t3, title="NPT Diagnostic [3]", border_style="blue"))

    # 4. exp_iL1-like displacement (barostat scaling term)
    V_b = 0.0  # box velocity at start
    x = V_b * dt
    scale = np.exp(x) - 1
    term1 = R * scale  # fractional * scalar
    term2 = dt * (P / M) * np.exp(x / 2)  # velocity term
    dR_mixed = term1 + term2
    R_after_scale = shift(R, dR_mixed, box=box_curr)
    t4 = Table(title="[4] Barostat scaling term")
    t4.add_column("Check", style="cyan")
    t4.add_column("Value", style="white")
    t4.add_row("x, exp(x)-1", f"{x}, {scale}")
    t4.add_row("max|term1|", f"{float(np.max(np.abs(term1))):.6e}")
    t4.add_row("max|term2|", f"{float(np.max(np.abs(term2))):.6e}")
    t4.add_row("R_after_scale finite", str(np.all(np.isfinite(R_after_scale))))
    t4.add_row("R_after_scale in [0,1)", str(np.all((R_after_scale >= 0) & (R_after_scale < 1.001))))
    c.print(Panel(t4, title="NPT Diagnostic [4]", border_style="blue"))

    # 5. Box and volume
    t5 = Table(title="[5] Box and volume")
    t5.add_column("Property", style="cyan")
    t5.add_column("Value", style="white")
    t5.add_row("box shape", str(np.asarray(box_curr).shape))
    t5.add_row("box diag", str(np.diagonal(np.asarray(box_curr))))
    t5.add_row("box_position (log V/V0)", f"{float(state.box_position)}")
    t5.add_row("box_momentum", f"{float(state.box_momentum)}")
    c.print(Panel(t5, title="NPT Diagnostic [5]", border_style="blue"))

    # 6. State components
    t6 = Table(title="[6] State components")
    t6.add_column("Component", style="cyan")
    t6.add_column("OK", style="white")
    t6.add_row("position finite", str(np.all(np.isfinite(R))))
    t6.add_row("momentum finite", str(np.all(np.isfinite(P))))
    t6.add_row("force finite", str(np.all(np.isfinite(state.force))))
    t6.add_row("mass shape, all positive", f"{M.shape}, {np.all(M > 0)}")
    c.print(Panel(t6, title="NPT Diagnostic [6]", border_style="blue"))

    # 7. Measured vs target pressure (drives box expansion/contraction)
    KE = quantity.kinetic_energy(momentum=P, mass=M)
    t7 = Table(title="[7] Pressure (measured vs target)")
    t7.add_column("Property", style="cyan")
    t7.add_column("Value", style="white")
    try:
        p_meas = quantity.pressure(
            npt_energy_fn, R, box_curr, kinetic_energy=KE,
            neighbor=(npt_pair_idx, npt_pair_mask)
        )
        p_meas_raw = float(p_meas)
        p_tgt_raw = float(npt_pressure)
        BAR_PER_ATM = 1.01325
        unit_p = float(unit["pressure"])
        p_meas_atm = p_meas_raw / (unit_p * BAR_PER_ATM)
        p_tgt_atm = p_tgt_raw / (unit_p * BAR_PER_ATM)
        t7.add_row("P_measured (raw)", f"{p_meas_raw:.6e}")
        t7.add_row("P_target (raw)", f"{p_tgt_raw:.6e}")
        t7.add_row("P_measured (atm)", f"{p_meas_atm:.2f}")
        t7.add_row("P_target (atm)", f"{p_tgt_atm:.2f}")
        t7.add_row("Note", "P_meas > P_tgt → expands; P_meas < P_tgt → contracts")
    except Exception as e:
        t7.add_row("Error", str(e))
    c.print(Panel(t7, title="NPT Diagnostic [7]", border_style="blue"))

    # 8. First step (apply_fn) and NaN location
    neighbor = (npt_pair_idx, npt_pair_mask)
    t8 = Table(title="[8] First NPT step (apply_fn)")
    t8.add_column("Check", style="cyan")
    t8.add_column("Value", style="white")
    try:
        state_one = apply_fn(state, neighbor=neighbor, pressure=npt_pressure)
        pos_ok = np.all(np.isfinite(np.asarray(state_one.position)))
        mom_ok = np.all(np.isfinite(np.asarray(state_one.momentum)))
        box_ok = np.all(np.isfinite(np.asarray(simulate.npt_box(state_one))))
        t8.add_row("position OK", str(pos_ok))
        t8.add_row("momentum OK", str(mom_ok))
        t8.add_row("box OK", str(box_ok))
        if not pos_ok:
            nan_count = np.sum(~np.isfinite(np.asarray(state_one.position)))
            t8.add_row("NaN count", str(nan_count))
            first_nan = np.where(~np.isfinite(np.asarray(state_one.position)))
            if len(first_nan[0]) > 0:
                t8.add_row("First NaN index", f"({first_nan[0][0]}, {first_nan[1][0]})")
    except Exception as e:
        t8.add_row("Error", f"{type(e).__name__}: {e}")
    c.print(Panel(t8, title="NPT Diagnostic [8]", border_style="blue"))
    c.print(Panel("NPT diagnostic complete", title="[bold]NPT Diagnostics (--npt-diagnose)[/bold]", border_style="green"))


def set_up_nhc_sim_routine(
    atoms,
    args,
    spherical_cutoff_calculator,
    get_update_fn,
    CUTOFF_PARAMS,
    n_monomers,
    monomer_offsets,
    Si_mass,
    show_frame=None,
    atoms_template=None,
):
    """Set up the Nose-Hoover chain simulation routine.

    Returns:
        The run_sim function.
    """
    atoms_template = atoms_template if atoms_template is not None else atoms
    T = args.temperature

    @jax.jit
    def evaluate_energies_and_forces(
        atomic_numbers,
        positions,
        mm_pair_idx=None,
        mm_pair_mask=None,
        box=None,
    ):
        return spherical_cutoff_calculator(
            atomic_numbers=atomic_numbers,
            positions=positions,
            n_monomers=n_monomers,
            cutoff_params=CUTOFF_PARAMS,
            doML=True,
            doMM=args.include_mm,
            doML_dimer=not args.skip_ml_dimers,
            debug=args.debug,
            mm_pair_idx=mm_pair_idx,
            mm_pair_mask=mm_pair_mask,
            box=box,
        )

    atomic_numbers = jnp.asarray(atoms.get_atomic_numbers(), dtype=jnp.int32)
    R = jnp.asarray(atoms.get_positions(), dtype=jnp.float32)

    @jit
    def jax_md_energy_fn(position, mm_pair_idx=None, mm_pair_mask=None, box=None, **kwargs):
        position = jnp.asarray(position, dtype=jnp.float32)
        result = evaluate_energies_and_forces(
            atomic_numbers=atomic_numbers,
            positions=position,
            mm_pair_idx=mm_pair_idx,
            mm_pair_mask=mm_pair_mask,
            box=box,
        )
        return result.energy.reshape(-1)[0]

    @jit
    def jax_md_force_fn(position, mm_pair_idx=None, mm_pair_mask=None, box=None, **kwargs):
        """Return forces from calculator (no autodiff). jax.grad(energy_fn) produces NaN."""
        position = jnp.asarray(position, dtype=jnp.float32)
        result = evaluate_energies_and_forces(
            atomic_numbers=atomic_numbers,
            positions=position,
            mm_pair_idx=mm_pair_idx,
            mm_pair_mask=mm_pair_mask,
            box=box,
        )
        return result.forces

    # evaluate_energies_and_forces (initial call - get update_fn if available)
    use_pbc = args.cell is not None
    is_npt = args.ensemble == "npt" and use_pbc
    update_fn = get_update_fn(R, CUTOFF_PARAMS) if get_update_fn else None
    pair_idx, pair_mask = None, None
    # Use (3,) or 3x3 box format for consistency with mm_energy_forces._box_to_cell_3x3
    L_cell = float(args.cell) if args.cell else None
    box_init = jnp.array([L_cell, L_cell, L_cell], dtype=jnp.float32) if L_cell else None
    box_nl = np.array([L_cell, L_cell, L_cell], dtype=np.float64) if L_cell else None
    pbc_box_nl = box_nl  # Capture for run_sim PBC minimization (avoids UnboundLocalError from later box_nl assignments)
    if update_fn is not None and use_pbc:
        if getattr(args, "debug", False):
            print("[nbr] Initial neighbor list update (PBC)")
        if is_npt:
            # NPT: neighbor list uses fractional_coordinates; pass frac pos and box [L,L,L]
            R_frac = np.asarray(R) / L_cell
            pair_idx, pair_mask = update_fn(R_frac, box=box_nl)
        else:
            # NVT/NVE: fixed box, pass box for neighbor list consistency
            pair_idx, pair_mask = update_fn(np.asarray(R), box=box_nl)
    c = Console()
    c.print(Panel("Compiling JAX energy/force (first run may take minutes)...", title="[bold cyan]JAX-MD[/bold cyan]", border_style="cyan"))
    t0 = time.perf_counter()
    result = evaluate_energies_and_forces(
        atomic_numbers=atomic_numbers,
        positions=R,
        mm_pair_idx=pair_idx,
        mm_pair_mask=pair_mask,
        box=box_init,
    )
    elapsed = time.perf_counter() - t0
    init_energy = result.energy.reshape(-1)[0]
    init_forces = np.asarray(result.forces).reshape(-1, 3)
    c.print(Panel(f"Compilation done in {elapsed:.2f} s", title="[bold green]JAX[/bold green]", border_style="green"))
    print_forces_summary(init_forces, energy_eV=float(init_energy), console=c)

    # MIC-only PBC: calculator uses minimum-image convention, no coordinate transform.
    pbc_map_fn = getattr(atoms.calc, "pbc_map", None) if atoms.calc else None
    pbc_info = f"BOXSIZE: {float(args.cell)} Å, PBC: True (MIC-only)" if use_pbc else "free space (no PBC), pbc_map: False"
    c.print(Panel(pbc_info, title="[bold]JAX-MD PBC[/bold]", border_style="blue"))

    # Mutable container for box/pairs so PBC minimization can update pairs for pbc_start_pos
    _pbc_state = {"box": box_init, "pair_idx": pair_idx, "pair_mask": pair_mask}

    # Energy and force: use calculator's explicit forces (jax.grad through calculator gives NaN).
    # MIC-only PBC: no coordinate transform; calculator uses MIC internally.
    if use_pbc and pbc_map_fn is not None:
        @jax.custom_vjp
        def wrapped_energy_fn(position, **kwargs):
            pos = jnp.array(position)
            return jax_md_energy_fn(
                pbc_map_fn(pos),
                mm_pair_idx=_pbc_state["pair_idx"],
                mm_pair_mask=_pbc_state["pair_mask"],
                box=_pbc_state["box"],
            )

        def wrapped_energy_fn_fwd(position, **kwargs):
            pos = jnp.array(position)
            R_mapped = pbc_map_fn(pos)
            E = jax_md_energy_fn(
                R_mapped,
                mm_pair_idx=_pbc_state["pair_idx"],
                mm_pair_mask=_pbc_state["pair_mask"],
                box=_pbc_state["box"],
            )
            return E, (pos, R_mapped)

        def wrapped_energy_fn_bwd(res, g, **kwargs):
            pos, R_mapped = res
            result = evaluate_energies_and_forces(
                atomic_numbers=atomic_numbers,
                positions=R_mapped,
                mm_pair_idx=_pbc_state["pair_idx"],
                mm_pair_mask=_pbc_state["pair_mask"],
                box=_pbc_state["box"],
            )
            F_mapped = result.forces
            F_orig = pbc_map_fn.transform_forces(pos, F_mapped)
            return (F_orig,)

        wrapped_energy_fn.defvjp(wrapped_energy_fn_fwd, wrapped_energy_fn_bwd)
        wrapped_energy_fn = jit(wrapped_energy_fn)

        @jit
        def wrapped_force_fn(position, **kwargs):
            pos = jnp.array(position)
            R_mapped = pbc_map_fn(pos)
            F_mapped = jax_md_force_fn(
                R_mapped,
                mm_pair_idx=_pbc_state["pair_idx"],
                mm_pair_mask=_pbc_state["pair_mask"],
                box=_pbc_state["box"],
            )
            return pbc_map_fn.transform_forces(pos, F_mapped)
    else:
        # MIC-only: capture box and pairs for PBC minimization (Fix A)
        @jit
        def wrapped_energy_fn(position, **kwargs):
            return jax_md_energy_fn(
                jnp.array(position),
                mm_pair_idx=_pbc_state["pair_idx"],
                mm_pair_mask=_pbc_state["pair_mask"],
                box=_pbc_state["box"],
            )

        @jit
        def wrapped_force_fn(position, **kwargs):
            return jax_md_force_fn(
                jnp.array(position),
                mm_pair_idx=_pbc_state["pair_idx"],
                mm_pair_mask=_pbc_state["pair_mask"],
                box=_pbc_state["box"],
            )

    # Shift and displacement for minimization and simulation
    # Minimization: energy/force use Cartesian (MIC). Use space.periodic when PBC so positions
    # stay in box; avoids coordinate mismatch (fractional shift + Cartesian energy → oscillation).
    # Simulation: NPT uses fractional; NVT/NVE use free or periodic.
    is_npt = args.ensemble == "npt" and use_pbc
    L_cell_val = float(args.cell) if args.cell else None
    if is_npt:
        L_npt = L_cell_val
        box_npt = jnp.eye(3, dtype=jnp.float32) * L_npt
        displacement, shift = space.periodic_general(box=box_npt, fractional_coordinates=True)
    else:
        displacement, shift = space.free()

    # Minimization shift: Cartesian + periodic when PBC (matches MIC energy/force)
    if use_pbc and L_cell_val is not None:
        _, shift_min = space.periodic(L_cell_val)
    else:
        shift_min = shift

    unwrapped_init_fn, unwrapped_step_fn = jax_md.minimize.fire_descent(
        wrapped_force_fn, shift_min, dt_start=0.001, dt_max=0.001
    )
    unwrapped_step_fn = jit(unwrapped_step_fn)

    # ========================================================================
    # SIMULATION PARAMETERS (metal units: eV, Å, ps, amu)
    # ========================================================================
    unit = units.metal_unit_system()
    # dt must be in ps: args.timestep is fs, 1 fs = 0.001 ps
    dt_fs = args.timestep
    dt = dt_fs * 0.001
    # NPT: neighbor list must be updated frequently (box changes every step).
    # Using 1000 steps with a stale neighbor list causes wrong forces → NaN.
    steps_per_recording = (
        getattr(args, "steps_per_recording", None)
        or (25 if (args.ensemble == "npt" and use_pbc) else 1000)
    )
    kT = T * unit['temperature']
    rng_key = jax.random.PRNGKey(0)
    c.print(Panel(
        f"Ensemble: {args.ensemble.upper()} | dt={dt} ps ({dt_fs} fs) | kT={kT} ({T} K) | steps_per_recording={steps_per_recording}",
        title="[bold]JAX-MD Simulation[/bold]",
        border_style="cyan",
    ))

    @jit
    def sim(state, neighbor=None, pressure=None):
        """Step function: for NPT pass neighbor and pressure; for NVT/NVE no kwargs."""
        def step_nve(i, s):
            return apply_fn(s)

        def step_npt(i, s):
            return apply_fn(s, neighbor=neighbor, pressure=pressure)

        step_fn = step_npt if (neighbor is not None and pressure is not None) else step_nve
        return lax.fori_loop(0, steps_per_recording, step_fn, state)

    # Select integrator based on ensemble
    if args.ensemble == "npt" and use_pbc:
        if update_fn is None:
            raise ValueError(
                "NPT requires jax_md neighbor list (cell list cannot handle dynamic box). "
                "Ensure jax_md is installed and pbc_cell is set."
            )
        BAR_PER_ATM = 1.01325
        p_atm = getattr(args, 'pressure', 1.0)
        if p_atm <= 0:
            # Preserve initial density: P = N*kT/V (ideal gas) so box stays ~constant
            V_init = float(L_cell_val ** 3)
            p_atm = float(n_monomers * kT / V_init / (unit['pressure'] * BAR_PER_ATM))
            c.print(Panel(f"pressure=0 → density-preserving P={p_atm:.2f} atm (N={n_monomers}, V={V_init:.0f} Å³)", title="[bold]NPT[/bold]", border_style="yellow"))
        # Pressure for npt_nose_hoover: jax_md uses same units as energy/volume.
        # Metal: energy=eV, V=Å³ → pressure in eV/Å³. 1 bar = unit['pressure'] eV/Å³; 1 atm = 1.01325 bar.
        pressure = jnp.array(p_atm * BAR_PER_ATM * unit['pressure'], dtype=jnp.float32)
        # Barostat tau: 10000*dt (2.5 ps at 0.25 fs) avoids NaN from aggressive box scaling
        barostat_tau = getattr(args, 'nhc_barostat_tau', 10000.0) * dt
        nhc_chain_length = getattr(args, 'nhc_chain_length', 3)
        nhc_chain_steps = getattr(args, 'nhc_chain_steps', 2)
        nhc_sy_steps = getattr(args, 'nhc_sy_steps', 3)
        nhc_tau = getattr(args, 'nhc_tau', 100.0)
        nhc_kwargs = {
            'chain_length': nhc_chain_length,
            'chain_steps': nhc_chain_steps,
            'sy_steps': nhc_sy_steps,
        }

        def _npt_energy_fn_raw(frac_pos, box=None, neighbor=None, perturbation=None, **kwargs):
            """Energy in fractional coords: transform to real, then evaluate.
            Supports perturbation=(1+eps) for NPT barostat stress (dU/dV)."""
            box_eff = jnp.asarray(box, dtype=jnp.float32)
            if perturbation is not None:
                # Isotropic: V' = V * perturbation, so L' = L * perturbation^(1/3)
                scale = jnp.power(jnp.asarray(perturbation, dtype=jnp.float32), 1.0 / 3.0)
                box_eff = box_eff * scale
            real_pos = space.transform(box_eff, frac_pos)
            pair_idx, pair_mask = neighbor if neighbor is not None else (None, None)
            result = evaluate_energies_and_forces(
                atomic_numbers=atomic_numbers,
                positions=real_pos,
                mm_pair_idx=pair_idx,
                mm_pair_mask=pair_mask,
                box=box_eff,
            )
            return result.energy.reshape(-1)[0]

        @jax.custom_vjp
        def npt_energy_fn(frac_pos, box=None, neighbor=None, perturbation=None, kT=None, mass=None):
            """NPT energy with custom VJP: use explicit calculator forces (jax.grad gives NaN).
            All kwargs as explicit params so JAX resolve_kwargs can bind them to positions."""
            return _npt_energy_fn_raw(
                frac_pos, box=box, neighbor=neighbor, perturbation=perturbation
            )

        def npt_energy_fn_fwd(frac_pos, box, neighbor, perturbation, kT, mass):
            E = _npt_energy_fn_raw(
                frac_pos, box=box, neighbor=neighbor, perturbation=perturbation
            )
            return E, (frac_pos, box, neighbor, perturbation)

        def npt_energy_fn_bwd(res, g):
            frac_pos, box, neighbor, perturbation = res
            box_eff = jnp.asarray(box, dtype=jnp.float32)
            if perturbation is not None:
                scale = jnp.power(jnp.asarray(perturbation, dtype=jnp.float32), 1.0 / 3.0)
                box_eff = box_eff * scale
            real_pos = space.transform(box_eff, frac_pos)
            pair_idx, pair_mask = neighbor if neighbor is not None else (None, None)
            F = jax_md_force_fn(
                real_pos,
                mm_pair_idx=pair_idx,
                mm_pair_mask=pair_mask,
                box=box_eff,
            )
            # grad(E) = -F; quantity.force = -grad, so we supply -F as grad
            grad_frac = -F * g
            return (grad_frac, None, None, None, None, None)

        npt_energy_fn.defvjp(npt_energy_fn_fwd, npt_energy_fn_bwd)
        npt_energy_fn = jit(npt_energy_fn)
        init_fn, apply_fn = simulate.npt_nose_hoover(
            npt_energy_fn,
            shift,
            dt=dt,
            pressure=pressure,
            kT=kT,
            barostat_kwargs=default_nhc_kwargs(jnp.array(barostat_tau), nhc_kwargs),
            thermostat_kwargs=default_nhc_kwargs(jnp.array(nhc_tau * dt), nhc_kwargs),
        )
        c.print(Panel(
            f"pressure={p_atm:.2f} atm | barostat_tau={barostat_tau:.6f} ps | thermostat tau={nhc_tau * dt:.6f} ps",
            title="[bold]NPT Nose-Hoover[/bold]",
            border_style="green",
        ))
    elif args.ensemble == "nvt":
        nhc_chain_length = getattr(args, 'nhc_chain_length', 3)
        nhc_chain_steps = getattr(args, 'nhc_chain_steps', 2)
        nhc_sy_steps = getattr(args, 'nhc_sy_steps', 3)
        nhc_tau = getattr(args, 'nhc_tau', 100.0)
        nhc_kwargs = {
            'chain_length': nhc_chain_length,
            'chain_steps': nhc_chain_steps,
            'sy_steps': nhc_sy_steps,
        }
        init_fn, apply_fn = simulate.nvt_nose_hoover(
            wrapped_force_fn, shift, dt=dt, kT=kT,
            thermostat_kwargs=default_nhc_kwargs(
                jnp.array(nhc_tau * dt), nhc_kwargs
            ),
        )
        c.print(Panel(
            f"chain_length={nhc_chain_length} | chain_steps={nhc_chain_steps} | sy_steps={nhc_sy_steps} | tau={nhc_tau * dt:.6f} ps",
            title="[bold]NVT Nose-Hoover[/bold]",
            border_style="green",
        ))
    else:  # nve
        init_fn, apply_fn = simulate.nve(wrapped_force_fn, shift, dt)
    apply_fn = jit(apply_fn)

    def run_sim(
        key,
        total_steps=args.nsteps_jaxmd,
        steps_per_recording=steps_per_recording,
        R=R,
        skip_minimization=False,
    ):
        total_records = total_steps // steps_per_recording
        _monomer_groups = [
            jnp.arange(int(monomer_offsets[m]), int(monomer_offsets[m + 1]))
            for m in range(n_monomers)
        ]

        # When ASE already ran (nsteps_ase > 0), R is ASE-minimized; skip JAX-MD minimization
        fire_positions = []
        if skip_minimization:
            minimized_pos = jnp.asarray(R, dtype=jnp.float32)
            c.print(Panel("Skipping minimization (using ASE positions)", title="[bold]JAX-MD Minimization[/bold]", border_style="yellow"))
        else:
            # Translate to center of mass before minimization (use actual masses)
            com = jnp.sum(Si_mass[:, None] * R, axis=0) / Si_mass.sum()
            initial_pos = jnp.asarray(R - com, dtype=jnp.float32)
            # Sanity check: ensure energy/gradient are finite at start; else use R directly
            try:
                _e0 = float(wrapped_energy_fn(initial_pos))
                _f0 = wrapped_force_fn(initial_pos)
                if not (np.isfinite(_e0) and np.all(np.isfinite(np.asarray(_f0)))):
                    initial_pos = jnp.asarray(R, dtype=jnp.float32)
                    c.print(Panel("Non-finite energy/forces at COM-centered pos; using R directly", title="[bold yellow]Warning[/bold yellow]", border_style="yellow"))
            except Exception:
                initial_pos = jnp.asarray(R, dtype=jnp.float32)
                print("Fallback: using R directly for minimization")
            fire_state = unwrapped_init_fn(initial_pos, mass=Si_mass)

            # FIRE minimization with step rejection (reject steps that produce NaN)
            NMIN = getattr(args, "jaxmd_minimize_steps", 1000)
            if NMIN <= 0:
                c.print(Panel("Skipping first minimization (0 steps requested)", title="[bold]JAX-MD Minimization[/bold]", border_style="yellow"))
            else:
                c.print(Panel(f"FIRE minimization ({NMIN} steps)", title="[bold cyan]JAX-MD Minimization[/bold cyan]", border_style="cyan"))
            for i in range(NMIN):
                fire_positions.append(fire_state.position)
                new_state = unwrapped_step_fn(fire_state)
                # Reject step if it produces NaN/Inf positions
                if not jnp.all(jnp.isfinite(new_state.position)):
                    c.print(Panel("FIRE step produced NaN/Inf positions; rejecting and stopping", title="[bold red]Error[/bold red]", border_style="red"))
                    break
                # Check energy/forces at new position before accepting
                energy = float(wrapped_energy_fn(new_state.position))
                max_force = float(jnp.abs(wrapped_force_fn(new_state.position)).max())
                if not (np.isfinite(energy) and np.isfinite(max_force)):
                    print("FIRE step led to NaN/Inf energy or forces; rejecting and stopping")
                    break
                fire_state = new_state

                if i % max(1, NMIN // 10) == 0:
                    c.print(f"  [dim]{i}/{NMIN}[/dim]: E={energy:.6f} eV, max|F|={max_force:.6f}")
            # fire_state always holds last valid position (we reject bad steps)
            minimized_pos = fire_state.position
            if jnp.any(~jnp.isfinite(minimized_pos)) and fire_positions:
                minimized_pos = fire_positions[-1]
                c.print(Panel("Using last valid position from first minimization", title="[bold yellow]Warning[/bold yellow]", border_style="yellow"))
        # save pdb (wrap by monomer when PBC so molecules stay intact)
        min_pdb_path = Path(f"{args.output_prefix}_minimized.pdb")
        min_pdb_path.parent.mkdir(parents=True, exist_ok=True)
        if use_pbc:
            _cell_for_pdb = jnp.asarray(atoms.get_cell()[:], dtype=jnp.float32)
            pos_wrapped = wrap_groups(
                jnp.asarray(minimized_pos), _monomer_groups, _cell_for_pdb, mass=Si_mass
            )
            atoms.set_positions(np.asarray(jax.device_get(pos_wrapped)))
        ase_io.write(str(min_pdb_path), atoms)

        # ========================================================================
        # PBC MINIMIZATION (only when PBC enabled, i.e. cell is set)
        # ========================================================================
        pbc_fire_positions = []
        if not use_pbc:
            md_pos = minimized_pos
            c.print(Panel("No cell: skipping PBC minimization", title="[bold]PBC Minimization[/bold]", border_style="yellow"))
            atoms.set_positions(np.asarray(md_pos))
        else:
            NMIN_PBC = getattr(args, "jaxmd_pbc_minimize_steps", 1000)
            if NMIN_PBC > 0:
                c.print(Panel(f"PBC FIRE minimization ({NMIN_PBC} steps)", title="[bold cyan]PBC Minimization[/bold cyan]", border_style="cyan"))
            # Molecular shift: wrap by monomer after each step so monomers stay intact.
            # space.periodic wraps atoms individually → monomers break across boundaries.
            _cell_jax = jnp.asarray(atoms.get_cell()[:], dtype=jnp.float32)
            _monomer_groups = [
                jnp.arange(int(monomer_offsets[m]), int(monomer_offsets[m + 1]))
                for m in range(n_monomers)
            ]

            def shift_molecular(R, dR, **kwargs):
                return wrap_groups(R + dR, _monomer_groups, _cell_jax, mass=Si_mass)

            pbc_unwrapped_init_fn, pbc_unwrapped_step_fn = jax_md.minimize.fire_descent(
                wrapped_force_fn, shift_molecular, dt_start=0.001, dt_max=0.001
            )
            pbc_unwrapped_step_fn = jit(pbc_unwrapped_step_fn)
            # Start from wrapped positions so we're in the cell (first min can drift)
            if pbc_map_fn is not None:
                pbc_start_pos = pbc_map_fn(minimized_pos)
            else:
                # MIC-only: wrap by monomer into cell
                pbc_start_pos = wrap_groups(
                    jnp.asarray(minimized_pos), _monomer_groups, _cell_jax, mass=Si_mass
                )
            # Recompute neighbor list for wrapped start positions (Fix A)
            if update_fn is not None:
                pbc_pair_idx, pbc_pair_mask = update_fn(
                    np.asarray(pbc_start_pos), box=pbc_box_nl
                )
                _pbc_state["pair_idx"] = pbc_pair_idx
                _pbc_state["pair_mask"] = pbc_pair_mask
            pbc_fire_state = pbc_unwrapped_init_fn(pbc_start_pos, mass=Si_mass)

            # Run PBC minimization (track best; stop early if forces increase - FIRE+unwrapped can wander)
            # Skip when first minimization already failed (minimized_pos invalid)
            if NMIN_PBC <= 0 or jnp.any(~jnp.isfinite(pbc_start_pos)):
                reason = "0 steps requested" if NMIN_PBC <= 0 else "no valid start position"
                print(f"Skipping PBC minimization ({reason})")
                md_pos = pbc_start_pos if jnp.all(jnp.isfinite(pbc_start_pos)) else minimized_pos
            else:
                max_force_start = float(jnp.abs(wrapped_force_fn(pbc_start_pos)).max())
                best_pbc_pos = pbc_start_pos
                best_pbc_max_f = max_force_start
                worsen_count = 0
                prev_max_f = max_force_start
                for i in range(NMIN_PBC):
                    pbc_fire_positions.append(pbc_fire_state.position)
                    new_pbc_state = pbc_unwrapped_step_fn(pbc_fire_state)
                    # Reject step if it produces NaN
                    if not jnp.all(jnp.isfinite(new_pbc_state.position)):
                        c.print(Panel("PBC FIRE step produced NaN; using first-min result", title="[bold red]Error[/bold red]", border_style="red"))
                        break
                    energy = float(wrapped_energy_fn(new_pbc_state.position))
                    max_force = float(jnp.abs(wrapped_force_fn(new_pbc_state.position)).max())
                    if not (np.isfinite(energy) and np.isfinite(max_force)):
                        print("PBC minimization hit NaN energy/forces; using first-min result")
                        break
                    pbc_fire_state = new_pbc_state
                    if max_force < best_pbc_max_f:
                        best_pbc_max_f = max_force
                        best_pbc_pos = pbc_fire_state.position
                        worsen_count = 0
                    else:
                        worsen_count = worsen_count + 1 if max_force > prev_max_f else 0
                    prev_max_f = max_force
                    if i % max(1, NMIN_PBC // 10) == 0:
                        c.print(f"  [dim]{i}/{NMIN_PBC}[/dim]: E={energy:.6f} eV, max|F|={max_force:.6f}")
                    if worsen_count >= WORSE_COUNT_THRESHOLD:
                        c.print(Panel(f"max|F| increased for {WORSE_COUNT_THRESHOLD} steps; stopping early (best max|F|={best_pbc_max_f:.4f})", title="[bold yellow]PBC Minimization[/bold yellow]", border_style="yellow"))
                        break

                # Use first-min result if PBC minimization worsened structure (max_force increased)
                if best_pbc_max_f > max_force_start * 1.1:
                    md_pos = pbc_map_fn(minimized_pos) if pbc_map_fn else minimized_pos
                    print(f"PBC minimization increased max|F| ({max_force_start:.4f} -> {best_pbc_max_f:.4f}); using first-min wrapped structure")
                else:
                    md_pos = best_pbc_pos

            # Save PBC minimized structure (md_pos already wrapped by monomer)
            atoms.set_positions(np.asarray(md_pos))
            pbc_pdb_path = Path(f"{args.output_prefix}_pbc_minimized.pdb")
            pbc_pdb_path.parent.mkdir(parents=True, exist_ok=True)
            ase_io.write(str(pbc_pdb_path), atoms)
            c.print(Panel(f"Complete. Final energy: {float(wrapped_energy_fn(md_pos)):.6f} eV", title="[bold green]PBC Minimization[/bold green]", border_style="green"))

        # Use last valid positions if minimization produced NaN
        if jnp.any(~jnp.isfinite(md_pos)) and pbc_fire_positions:
            md_pos = pbc_fire_positions[-1]
            c.print(Panel("NaN in PBC minimization; using last valid position from PBC", title="[bold yellow]Warning[/bold yellow]", border_style="yellow"))
        if jnp.any(~jnp.isfinite(md_pos)) and fire_positions:
            md_pos = pbc_map_fn(fire_positions[-1]) if (use_pbc and pbc_map_fn) else fire_positions[-1]
            c.print(Panel("Using last valid position from first minimization", title="[bold yellow]Warning[/bold yellow]", border_style="yellow"))
        if jnp.any(~jnp.isfinite(md_pos)):
            c.print(Panel(f"No valid positions for {args.ensemble.upper()}; skipping JAX-MD", title="[bold red]Error[/bold red]", border_style="red"))
            return 0, jnp.array([]).reshape(0, len(md_pos), 3), None

        if args.ensemble == "npt" and use_pbc:
            # NPT: positions in fractional coords; wrap md_pos into cell first, then convert to fractional
            box_curr = box_npt
            _cell_jax = jnp.asarray(atoms.get_cell()[:], dtype=jnp.float32)
            _monomer_groups = [
                jnp.arange(int(monomer_offsets[m]), int(monomer_offsets[m + 1]))
                for m in range(n_monomers)
            ]
            md_pos_wrapped = wrap_groups(
                jnp.asarray(md_pos), _monomer_groups, _cell_jax, mass=Si_mass
            )
            md_pos_frac = md_pos_wrapped / float(args.cell)  # cubic: frac = R / L
            # Neighbor list with fractional_coordinates expects frac pos and box [L,L,L]
            box_nl = np.array([float(args.cell)] * 3, dtype=np.float64)
            pair_idx, pair_mask = update_fn(np.asarray(md_pos_frac), box=box_nl)
            state = init_fn(
                key, md_pos_frac, box=box_curr,
                neighbor=(pair_idx, pair_mask), kT=kT, mass=Si_mass
            )
            npt_pair_idx, npt_pair_mask = pair_idx, pair_mask
            npt_pressure = pressure  # Use same pressure as NPT block (handles --pressure 0)
        elif args.ensemble == "nvt":
            state = init_fn(key, md_pos, mass=Si_mass)
            npt_pair_idx, npt_pair_mask = None, None
            npt_pressure = None
        else:
            state = init_fn(key, md_pos, kT, mass=Si_mass)
            npt_pair_idx, npt_pair_mask = None, None
            npt_pressure = None
        c.print(Panel(f"Momentum initialized for {T} K", title="[bold]JAX-MD[/bold]", border_style="green"))
        nhc_positions = []
        nhc_boxes = []  # NPT: box at each record step (for frac→real when saving)

        # get energy of initial state
        if is_npt and npt_pair_idx is not None:
            box_curr = simulate.npt_box(state)
            energy_initial = float(npt_energy_fn(state.position, box=box_curr, neighbor=(npt_pair_idx, npt_pair_mask)))
        else:
            energy_initial = float(wrapped_energy_fn(state.position))
        # Debug: forces from calculator (used by NVE; jax.grad gives NaN)
        if is_npt and npt_pair_idx is not None:
            box_curr = simulate.npt_box(state)
            real_pos = space.transform(box_curr, state.position)
            forces_jax = jax_md_force_fn(
                real_pos,
                mm_pair_idx=npt_pair_idx,
                mm_pair_mask=npt_pair_mask,
                box=box_curr,
            )
        else:
            forces_jax = wrapped_force_fn(state.position)
        print_forces_summary(np.asarray(forces_jax), energy_eV=energy_initial, console=c)
        # velocity = momentum / mass; position update = R + dt * v (half-step in VV)
        vel = state.momentum / state.mass
        disp_first = dt * vel
        t_vel = Table(title="First-step kinematics")
        t_vel.add_column("Property", style="cyan")
        t_vel.add_column("Value", style="white")
        t_vel.add_row("velocity sample [0]", str(np.asarray(vel[0])))
        t_vel.add_row("disp dt*v [0]", str(np.asarray(disp_first[0])))
        t_vel.add_row("max|disp|", f"{float(jnp.max(jnp.abs(disp_first))):.6f}")
        c.print(Panel(t_vel, title="[bold]JAX-MD First Step[/bold]", border_style="blue"))

        # ========================================================================
        # NPT DIAGNOSTIC TESTS (--npt-diagnose)
        # ========================================================================
        if is_npt and npt_pair_idx is not None and getattr(args, "npt_diagnose", False):
            _run_npt_diagnostics(
                state=state,
                npt_energy_fn=npt_energy_fn,
                jax_md_force_fn=jax_md_force_fn,
                apply_fn=apply_fn,
                shift=shift,
                space=space,
                simulate=simulate,
                quantity=quantity,
                npt_pair_idx=npt_pair_idx,
                npt_pair_mask=npt_pair_mask,
                npt_pressure=npt_pressure,
                unit=unit,
                dt=dt,
                kT=kT,
                grad=grad,
            )

        # Single-step diagnostic: catch NaN on first step (common with wrong mass/units)
        if is_npt and npt_pair_idx is not None:
            state_one = apply_fn(state, neighbor=(npt_pair_idx, npt_pair_mask), pressure=npt_pressure)
        else:
            state_one = apply_fn(state)
        if not jnp.all(jnp.isfinite(state_one.position)):
            t_err = Table(title="First step NaN")
            t_err.add_column("Check", style="red")
            t_err.add_column("Value", style="white")
            t_err.add_row("mass shape", str(state.mass.shape))
            t_err.add_row("mass min/max", f"{float(jnp.min(state.mass)):.4f} / {float(jnp.max(state.mass)):.4f}")
            c.print(Panel(t_err, title="[bold red]ERROR: First step produced NaN positions[/bold red]\nCheck: mass in amu, dt in ps, energy_fn returns eV.", border_style="red"))
            pos_out = space.transform(simulate.npt_box(state), state.position) if is_npt else state.position
            box_out = [np.asarray(jax.device_get(simulate.npt_box(state)))] if is_npt else None
            return 0, np.stack([np.asarray(jax.device_get(pos_out))]), box_out
        if is_npt and npt_pair_idx is not None:
            box_one = simulate.npt_box(state_one)
            e1 = float(npt_energy_fn(state_one.position, box=box_one, neighbor=(npt_pair_idx, npt_pair_mask)))
        else:
            e1 = float(wrapped_energy_fn(state_one.position))
        c.print(Panel(f"First step OK: E_pot={e1:.6f} eV", title="[bold green]JAX-MD[/bold green]", border_style="green"))

        nbr_monitor = getattr(args, "nbr_monitor", False)
        if use_pbc:
            c.print(Panel(f"{n_monomers} monomer groups, wrapping every {steps_per_recording} steps", title="[bold]PBC Wrapping[/bold]", border_style="blue"))
        c.print(Panel(f"Starting {args.ensemble.upper()} simulation", title="[bold cyan]JAX-MD[/bold cyan]", border_style="cyan"))
        if is_npt:
            hdr = "\t\tTime (ps)\tSteps\tE_pot (eV)\tE_tot (eV)\tT (K)\tL (Å)\tV (Å³)\tP_tgt (atm)\tP_meas (atm)\tavg(ns/day)"
            if nbr_monitor:
                hdr += "\tn_valid\tcapacity\tfill%"
            c.print(f"[dim]{hdr}[/dim]")
        else:
            c.print("[dim]\t\tTime (ps)\tSteps\tE_pot (eV)\tE_tot (eV)\tT (K)\tavg(ns/day)[/dim]")

        # ========================================================================
        # HDF5 REPORTER SETUP
        # ========================================================================
        hdf5_path = Path(f"{args.output_prefix}_{args.ensemble}.h5")
        hdf5_path.parent.mkdir(parents=True, exist_ok=True)
        scalar_quantities = ["total_energy", "time_ps"]
        if nbr_monitor and is_npt:
            scalar_quantities.extend(["nbr_n_valid", "nbr_capacity", "nbr_fill_ratio"])
        hdf5_reporter = make_jaxmd_reporter(
            str(hdf5_path),
            n_atoms=len(atoms),
            buffer_size=min(100, total_records),
            include_positions=True,
            include_velocities=True,
            scalar_quantities=scalar_quantities,
            attrs={
                "ensemble": args.ensemble,
                "temperature_target": T,
                "dt_ps": dt,
                "steps_per_recording": steps_per_recording,
                "n_atoms": len(atoms),
                "atomic_numbers": atoms.get_atomic_numbers(),
            },
        )

        # ========================================================================
        # PBC WRAPPING SETUP
        # ========================================================================
        if use_pbc:
            _cell_jax = jnp.asarray(atoms.get_cell()[:], dtype=jnp.float32)
            _monomer_groups = [
                jnp.arange(int(monomer_offsets[m]), int(monomer_offsets[m + 1]))
                for m in range(n_monomers)
            ]

        # ========================================================================
        # MAIN SIMULATION LOOP
        # ========================================================================
        jaxmd_loop_start = time.perf_counter()

        for i in range(total_records):
            if is_npt and update_fn is not None:
                box_curr = simulate.npt_box(state)
                # Neighbor list with fractional_coordinates expects frac pos and box [L,L,L]
                box_nl = np.asarray(box_curr)
                if box_nl.shape == (1,) or box_nl.ndim == 0:
                    L = float(box_nl.reshape(-1)[0])
                    box_nl = np.array([L, L, L], dtype=np.float64)
                if getattr(args, "debug", False) and (i < 3 or i % 50 == 0):
                    print(f"[nbr] NPT record {i}: updating neighbor list, box L={float(box_nl[0]):.4f}")
                npt_pair_idx, npt_pair_mask = update_fn(
                    np.asarray(state.position), box=box_nl
                )
                state = sim(state, neighbor=(npt_pair_idx, npt_pair_mask), pressure=npt_pressure)
            else:
                state = sim(state)

            if use_pbc:
                if is_npt:
                    # NPT: wrap fractional coords to [0,1)
                    box_curr = simulate.npt_box(state)
                    frac_pos = state.position
                    wrapped_frac = frac_pos - jnp.floor(frac_pos)
                    state = state.set(position=wrapped_frac)
                else:
                    wrapped_pos = wrap_groups(
                        state.position, _monomer_groups, _cell_jax, mass=Si_mass
                    )
                    state = state.set(position=wrapped_pos)

            # Store current position (NPT: fractional + box for correct real coords at save)
            if is_npt:
                box_curr = simulate.npt_box(state)
                nhc_positions.append(state.position)
                nhc_boxes.append(box_curr)
            else:
                nhc_positions.append(state.position)

            # Braille viewer: update at each recording block
            if show_frame is not None and atoms_template is not None:
                steps = (i + 1) * steps_per_recording
                if is_npt:
                    box_curr = simulate.npt_box(state)
                    pos_real = space.transform(box_curr, state.position)
                    pos_real = wrap_groups(pos_real, _monomer_groups, box_curr, mass=Si_mass)
                else:
                    pos_real = state.position
                    if use_pbc:
                        pos_real = wrap_groups(pos_real, _monomer_groups, _cell_jax, mass=Si_mass)
                atoms_template.set_positions(np.asarray(jax.device_get(pos_real)))
                show_frame(atoms_template, steps, "jaxmd")

            # Print progress every 10 steps
            nbr_n_valid = nbr_capacity = nbr_fill_ratio = None
            if i % 10 == 0:
                steps = (i + 1) * steps_per_recording
                time_ps = steps * dt
                T_curr = jax_md.quantity.temperature(
                    momentum=state.momentum,
                    mass=state.mass
                ) / unit['temperature']
                temp = float(T_curr)
                if is_npt and npt_pair_idx is not None:
                    box_curr = simulate.npt_box(state)
                    e_pot = float(npt_energy_fn(state.position, box=box_curr, neighbor=(npt_pair_idx, npt_pair_mask)))
                else:
                    e_pot = float(wrapped_energy_fn(state.position))
                e_kin = float(jax_md.quantity.kinetic_energy(
                    momentum=state.momentum,
                    mass=state.mass
                ))
                e_tot = e_pot + e_kin
                elapsed_s = time.perf_counter() - jaxmd_loop_start
                simulated_ns = steps * dt_fs * 1e-6
                if simulated_ns > 0 and elapsed_s > 0:
                    avg_speed_ns_per_day = simulated_ns * 86400.0 / elapsed_s
                else:
                    avg_speed_ns_per_day = float("nan")
                if is_npt and npt_pair_idx is not None:
                    vol = float(quantity.volume(3, box_curr))
                    box_diag = np.diagonal(np.asarray(box_curr)[:3, :3])
                    L = float(box_diag[0]) if box_diag.size > 0 else float("nan")
                    BAR_PER_ATM = 1.01325
                    unit_p = float(unit["pressure"])
                    p_tgt_atm = float(npt_pressure / (unit_p * BAR_PER_ATM))
                    # Measured pressure (virial + kinetic) for diagnostics
                    try:
                        p_meas = quantity.pressure(
                            npt_energy_fn, state.position, box_curr,
                            kinetic_energy=e_kin, neighbor=(npt_pair_idx, npt_pair_mask)
                        )
                        p_meas_atm = float(p_meas / (unit_p * BAR_PER_ATM))
                    except Exception:
                        p_meas_atm = float("nan")
                    line = (
                        f"{time_ps:10.4f}\t{steps:6d}\t{e_pot:10.4f}\t{e_tot:10.4f}\t{temp:10.2f}\t"
                        f"{L:8.2f}\t{vol:10.1f}\t{p_tgt_atm:8.2f}\t{p_meas_atm:8.2f}\t"
                        f"{avg_speed_ns_per_day:10.4f}"
                    )
                    if nbr_monitor:
                        nbr_n_valid = int(np.sum(np.asarray(jax.device_get(npt_pair_mask))))
                        nbr_capacity = npt_pair_idx.shape[0]
                        nbr_fill_ratio = nbr_n_valid / nbr_capacity if nbr_capacity > 0 else 0.0
                        line += f"\t{nbr_n_valid}\t{nbr_capacity}\t{100.0 * nbr_fill_ratio:.1f}%"
                    print(line)
                else:
                    print(
                        f"{time_ps:10.4f}\t{steps:6d}\t{e_pot:10.4f}\t{e_tot:10.4f}\t{temp:10.2f}\t"
                        f"{avg_speed_ns_per_day:10.4f}"
                    )

                # Record to HDF5 (NPT: save real positions via transform, wrap by monomer)
                pos_for_h5 = state.position
                if is_npt:
                    box_curr = simulate.npt_box(state)
                    pos_for_h5 = space.transform(box_curr, state.position)
                    pos_for_h5 = wrap_groups(pos_for_h5, _monomer_groups, box_curr, mass=Si_mass)
                report_kw = dict(
                    potential_energy=e_pot,
                    kinetic_energy=e_kin,
                    temperature=temp,
                    invariant=e_tot,
                    total_energy=e_tot,
                    time_ps=time_ps,
                    positions=pos_for_h5,
                    velocities=state.momentum / state.mass,
                )
                if nbr_monitor and is_npt and npt_pair_idx is not None and nbr_n_valid is not None:
                    report_kw["nbr_n_valid"] = nbr_n_valid
                    report_kw["nbr_capacity"] = nbr_capacity
                    report_kw["nbr_fill_ratio"] = nbr_fill_ratio
                hdf5_reporter.report(**report_kw)

                # Stop on numerical instability (NaN, Inf, or energy blow-up to 0)
                if not np.isfinite(e_pot) or not np.isfinite(temp):
                    print(f"Numerical instability at step {steps}; stopping.")
                    if len(nhc_positions) > 1:
                        nhc_positions = nhc_positions[:-1]
                        if is_npt:
                            nhc_boxes = nhc_boxes[:-1]
                    break
                if e_pot >= 0 and energy_initial < 0:
                    c.print(Panel(f"Energy blow-up at step {steps} (E_pot={e_pot:.4f}); stopping.", title="[bold red]Error[/bold red]", border_style="red"))
                    if len(nhc_positions) > 1:
                        nhc_positions = nhc_positions[:-1]
                        if is_npt:
                            nhc_boxes = nhc_boxes[:-1]
                    break

        hdf5_reporter.close()
        c.print(Panel(str(hdf5_path), title="[bold green]HDF5 trajectory saved[/bold green]", border_style="green"))

        steps_completed = i * steps_per_recording
        c.print(Panel(f"{steps_completed} steps ({steps_completed * dt:.2f} ps)", title="[bold]Simulation complete[/bold]", border_style="green"))

        nhc_positions_out = []
        nhc_boxes_out = []  # NPT: real-space box per frame for trajectory cell
        for idx, R in enumerate(nhc_positions):
            if is_npt:
                # NPT: convert fractional to real using box at this step
                box_i = nhc_boxes[idx]
                R = space.transform(box_i, R)
                # Wrap by monomer so molecules stay intact (frac wrap was per-atom)
                R = wrap_groups(R, _monomer_groups, box_i, mass=Si_mass)
                nhc_boxes_out.append(np.asarray(jax.device_get(box_i)))
            elif use_pbc and pbc_map_fn is not None:
                R = pbc_map_fn(R)
            nhc_positions_out.append(np.asarray(jax.device_get(R)))
        return steps_completed, np.stack(nhc_positions_out), nhc_boxes_out if is_npt else None

    return run_sim
