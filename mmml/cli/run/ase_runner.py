"""
ASE-specific MD and minimization routines.

Extracted from run_sim.py to separate ASE and JAX-MD code paths.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


def wrap_positions_for_pbc(
    positions: np.ndarray,
    *,
    cell: Optional[float],
    hybrid_calc: Any,
    monomer_offsets: np.ndarray,
    masses: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Wrap positions into cell. Uses pbc_map when available; otherwise wrap by monomer (MIC-only).

    Uses mass-weighted center of mass when masses provided.
    """
    if cell is None:
        return positions
    pbc_map_fn = getattr(hybrid_calc, "pbc_map", None)
    if pbc_map_fn is not None and getattr(hybrid_calc, "do_pbc_map", False):
        import jax.numpy as jnp
        import jax
        R_mapped = pbc_map_fn(jnp.asarray(positions))
        return np.asarray(jax.device_get(R_mapped))
    # MIC-only: wrap by monomer into primary cell (COM-based)
    from mmml.interfaces.pycharmmInterface.cell_list import _wrap_groups_np
    cell_matrix = np.diag([float(cell)] * 3) if np.isscalar(cell) else np.asarray(cell, dtype=np.float64)
    if cell_matrix.ndim == 1 and cell_matrix.shape[0] == 3:
        cell_matrix = np.diag(cell_matrix)
    return _wrap_groups_np(
        np.asarray(positions, dtype=np.float64), cell_matrix, monomer_offsets, masses=masses
    )


def minimize_structure(
    atoms: Any,
    *,
    args: Any,
    hybrid_calc: Any,
    monomer_offsets: np.ndarray,
    n_monomers: int,
    atoms_per_monomer_list: list[int],
    simple_physnet_calculator: Any,
    output_prefix: str,
    run_index: int = 0,
    nsteps: int = 60,
    fmax: float = 0.0006,
    charmm: bool = False,
    ase: bool = True,
) -> Any:
    """Minimize structure using CHARMM and/or ASE BFGS."""
    import ase.io as ase_io
    import ase.optimize as ase_opt
    from mmml.interfaces.pycharmmInterface.import_pycharmm import coor
    import pandas as pd
    import pycharmm

    if charmm:
        pycharmm.minimize.run_abnr(nstep=10000, tolenr=1e-6, tolgrd=1e-6)
        pycharmm.lingo.charmm_script("ENER")
        from mmml.interfaces.pycharmmInterface.import_pycharmm import safe_energy_show
        safe_energy_show()
        atoms.set_positions(coor.get_positions())
        atoms = optimize_as_monomers(
            atoms,
            args=args,
            hybrid_calc=hybrid_calc,
            monomer_offsets=monomer_offsets,
            n_monomers=n_monomers,
            atoms_per_monomer_list=atoms_per_monomer_list,
            simple_physnet_calculator=simple_physnet_calculator,
            run_index=run_index,
            nsteps=100,
            fmax=0.0006,
        )

    if ase:
        traj_path = Path(f"{output_prefix}_bfgs_{run_index}.traj")
        traj_path.parent.mkdir(parents=True, exist_ok=True)
        traj = ase_io.Trajectory(str(traj_path), 'w')
        c = Console()
        c.print(Panel(
            f"BFGS: {nsteps} steps, fmax={fmax}",
            title="[bold cyan]ASE Minimization[/bold cyan]",
            border_style="cyan",
        ))
        _ = ase_opt.BFGS(atoms, trajectory=traj).run(fmax=fmax, steps=nsteps)
        xyz = pd.DataFrame(atoms.get_positions(), columns=["x", "y", "z"])
        coor.set_positions(xyz)
        traj.close()
        return atoms

    return atoms


def optimize_as_monomers(
    atoms: Any,
    *,
    args: Any,
    hybrid_calc: Any,
    monomer_offsets: np.ndarray,
    n_monomers: int,
    atoms_per_monomer_list: list[int],
    simple_physnet_calculator: Any,
    run_index: int = 0,
    nsteps: int = 60,
    fmax: float = 0.0006,
) -> Any:
    """Optimize each monomer separately with ASE BFGS."""
    import ase.optimize as ase_opt
    from mmml.interfaces.pycharmmInterface.import_pycharmm import coor
    import pandas as pd

    optimized_atoms_positions = np.zeros_like(atoms.get_positions())
    for i in range(n_monomers):
        off = int(monomer_offsets[i])
        n_i = atoms_per_monomer_list[i]
        monomer_atoms = atoms[off:off + n_i]
        monomer_atoms.calc = simple_physnet_calculator
        _ = ase_opt.BFGS(monomer_atoms).run(fmax=fmax, steps=nsteps)
        optimized_atoms_positions[off:off + n_i] = monomer_atoms.get_positions()

    atoms.set_positions(optimized_atoms_positions)
    if args.cell is not None:
        wrapped = wrap_positions_for_pbc(
            atoms.get_positions(),
            cell=args.cell,
            hybrid_calc=hybrid_calc,
            monomer_offsets=monomer_offsets,
            masses=atoms.get_masses(),
        )
        atoms.set_positions(wrapped)
        xyz = pd.DataFrame(wrapped, columns=["x", "y", "z"])
    else:
        xyz = pd.DataFrame(atoms.get_positions(), columns=["x", "y", "z"])
    coor.set_positions(xyz)
    return atoms


def run_ase_md(
    atoms: Any,
    *,
    args: Any,
    hybrid_calc: Any,
    monomer_offsets: np.ndarray,
    n_monomers: int,
    atoms_per_monomer_list: list[int],
    simple_physnet_calculator: Any,
    run_index: int = 0,
    temperature: float = 300.0,
) -> None:
    """Run ASE MD simulation with Velocity Verlet."""
    import ase
    import ase.io as ase_io
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
    from ase.md.verlet import VelocityVerlet

    if run_index == 0 and getattr(args, "optimize_monomers", False):
        atoms = optimize_as_monomers(
            atoms,
            args=args,
            hybrid_calc=hybrid_calc,
            monomer_offsets=monomer_offsets,
            n_monomers=n_monomers,
            atoms_per_monomer_list=atoms_per_monomer_list,
            simple_physnet_calculator=simple_physnet_calculator,
            run_index=run_index,
            nsteps=100,
            fmax=0.0006,
        )

    c = Console()
    c.print(Panel(f"Pre-BFGS energy: {atoms.get_potential_energy():.6f} eV", title="[bold]ASE[/bold]", border_style="blue"))

    atoms = minimize_structure(
        atoms,
        args=args,
        hybrid_calc=hybrid_calc,
        monomer_offsets=monomer_offsets,
        n_monomers=n_monomers,
        atoms_per_monomer_list=atoms_per_monomer_list,
        simple_physnet_calculator=simple_physnet_calculator,
        output_prefix=args.output_prefix,
        run_index=run_index,
        nsteps=100 if run_index == 0 else 10,
        fmax=0.0006 if run_index == 0 else 0.001,
    )
    if args.cell is not None:
        c.print(Panel(f"Wrapping positions into cell: {args.cell} Å", title="[bold]PBC[/bold]", border_style="blue"))
        wrapped = wrap_positions_for_pbc(
            atoms.get_positions(),
            cell=args.cell,
            hybrid_calc=hybrid_calc,
            monomer_offsets=monomer_offsets,
            masses=atoms.get_masses(),
        )
        atoms.set_positions(wrapped)
        from mmml.interfaces.pycharmmInterface.import_pycharmm import coor
        import pandas as pd
        xyz = pd.DataFrame(wrapped, columns=["x", "y", "z"])
        coor.set_positions(xyz)

    timestep_fs = args.timestep
    num_steps = args.nsteps_ase
    ase_atoms = atoms

    if run_index == 0:
        MaxwellBoltzmannDistribution(ase_atoms, temperature_K=temperature)
        Stationary(ase_atoms)
        ZeroRotation(ase_atoms)

    dt = timestep_fs * ase.units.fs
    c.print(Panel(
        f"Velocity Verlet | dt={dt} (ase units) | T={temperature} K | {num_steps} steps",
        title="[bold cyan]ASE MD[/bold cyan]",
        border_style="cyan",
    ))
    integrator = VelocityVerlet(ase_atoms, timestep=dt)

    traj_filename = f"{args.output_prefix}_ase_{run_index}_{int(temperature)}K.traj"
    Path(traj_filename).parent.mkdir(parents=True, exist_ok=True)
    traj = ase_io.Trajectory(traj_filename, 'w')

    frames = np.zeros((num_steps, len(ase_atoms), 3))
    potential_energy = np.zeros((num_steps,))
    kinetic_energy = np.zeros((num_steps,))
    total_energy = np.zeros((num_steps,))

    breakcount = 0
    ase_loop_start = time.perf_counter()
    for i in range(num_steps):
        integrator.run(1)
        frames[i] = ase_atoms.get_positions()
        potential_energy[i] = ase_atoms.get_potential_energy()
        kinetic_energy[i] = ase_atoms.get_kinetic_energy()
        total_energy[i] = ase_atoms.get_total_energy()

        if i > 10 and (kinetic_energy[i] > 300 or potential_energy[i] > 0):
            t_spike = Table(title="Energy spike - re-minimizing")
            t_spike.add_column("Property", style="red")
            t_spike.add_column("Value", style="white")
            t_spike.add_row("Step", str(i))
            t_spike.add_row("E_kin (eV)", f"{kinetic_energy[i]:.6f}")
            t_spike.add_row("E_pot (eV)", f"{potential_energy[i]:.6f}")
            t_spike.add_row("E_tot (eV)", f"{total_energy[i]:.6f}")
            t_spike.add_row("Breakcount", str(breakcount))
            t_spike.add_row("T (K)", f"{temperature}")
            t_spike.add_row("dt (fs)", f"{timestep_fs}")
            t_spike.add_row("Atoms", str(len(ase_atoms)))
            t_spike.add_row("Monomers", str(n_monomers))
            c.print(Panel(t_spike, title="[bold red]Energy spike detected[/bold red]", border_style="red"))

            minimize_structure(
                ase_atoms,
                args=args,
                hybrid_calc=hybrid_calc,
                monomer_offsets=monomer_offsets,
                n_monomers=n_monomers,
                atoms_per_monomer_list=atoms_per_monomer_list,
                simple_physnet_calculator=simple_physnet_calculator,
                output_prefix=args.output_prefix,
                run_index=f"{run_index}_{breakcount}_{i}_",
                nsteps=20 if run_index == 0 else 10,
                fmax=0.0006 if run_index == 0 else 0.001,
                charmm=True,
            )
            cur_eng = ase_atoms.get_potential_energy()
            c.print(f"[dim]Re-minimized energy: {cur_eng:.6f} eV[/dim]")
            Stationary(ase_atoms)
            ZeroRotation(ase_atoms)
            breakcount += 1
        if breakcount > 1:
            c.print(Panel("Maximum number of breaks reached", title="[bold red]ASE MD[/bold red]", border_style="red"))
            break
        if (i != 0) and (i % args.write_interval == 0):
            if args.cell is not None:
                wrapped = wrap_positions_for_pbc(
                    ase_atoms.get_positions(),
                    cell=args.cell,
                    hybrid_calc=hybrid_calc,
                    monomer_offsets=monomer_offsets,
                    masses=ase_atoms.get_masses(),
                )
                orig_pos = ase_atoms.get_positions().copy()
                ase_atoms.set_positions(wrapped)
                traj.write(ase_atoms)
                ase_atoms.set_positions(orig_pos)
            else:
                traj.write(ase_atoms)
        if args.ensemble == "nvt":
            if (i % args.heating_interval == 0):
                Stationary(ase_atoms)
                ZeroRotation(ase_atoms)
                MaxwellBoltzmannDistribution(ase_atoms, temperature_K=temperature)
                c.print(f"[dim]Temperature adjusted to {temperature} K[/dim]")
        if i % 100 == 0:
            elapsed_s = time.perf_counter() - ase_loop_start
            completed_steps = i + 1
            simulated_ns = completed_steps * timestep_fs * 1e-6
            if simulated_ns > 0 and elapsed_s > 0:
                avg_speed_ns_per_day = simulated_ns * 86400.0 / elapsed_s
                time_per_ns_s = elapsed_s / simulated_ns
                perf_msg = (
                    f"avg_speed {avg_speed_ns_per_day:8.4f} ns/day "
                    f"time_per_ns {time_per_ns_s:10.2f} s/ns"
                )
            else:
                perf_msg = "avg_speed n/a time_per_ns n/a"
            c.print(
                f"[dim]step {i:5d}[/dim] epot {potential_energy[i]: 5.3f} "
                f"ekin {kinetic_energy[i]: 5.3f} etot {total_energy[i]: 5.3f} | "
                f"{perf_msg}"
            )

    traj.close()
    c.print(Panel(str(traj_filename), title="[bold green]Trajectory saved[/bold green]", border_style="green"))
    c.print(Panel("ASE MD simulation complete!", title="[bold green]ASE[/bold green]", border_style="green"))
