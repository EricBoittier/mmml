import io

import ase
import ase.calculators.calculator as ase_calc
import ase.io as ase_io
import ase.optimize as ase_opt
import matplotlib.pyplot as plt
import numpy as np
import py3Dmol
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
from ase.md.verlet import VelocityVerlet


def NVT(
    atoms,
    temperature=300.0,
    timestep_fs=0.1,
    num_steps=100_0,
    htfreq=1000,
    printfreq=1000,
):
    # Draw initial momenta.
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
    Stationary(atoms)  # Remove center of mass translation
    ZeroRotation(atoms)  # Remove rotations

    # Initialize Velocity Verlet integrator.
    integrator = VelocityVerlet(atoms, timestep=timestep_fs * ase.units.fs)

    # Run molecular dynamics.
    frames = np.zeros((num_steps, len(atoms), 3))
    dipoles = np.zeros((num_steps, 1, 3))
    potential_energy = np.zeros((num_steps,))
    kinetic_energy = np.zeros((num_steps,))
    total_energy = np.zeros((num_steps,))
    for i in range(num_steps):
        # Run 1 time step.
        integrator.run(1)
        # Save current frame and keep track of energies.
        frames[i] = atoms.get_positions()
        potential_energy[i] = atoms.get_potential_energy()
        kinetic_energy[i] = atoms.get_kinetic_energy()
        total_energy[i] = atoms.get_total_energy()
        dipoles[i] = atoms.get_dipole_moment()
        # Occasionally print progress.q
        if i % htfreq == 0:
            MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
        if i % printfreq == 0:
            print(
                f"step {i:5d} epot {potential_energy[i]: 5.3f} ekin {kinetic_energy[i]: 5.3f} etot {total_energy[i]: 5.3f}"
            )


def NVE(atoms, temperature=300.0, timestep_fs=0.1, num_steps=100_0, printfreq=1000):
    # Draw initial momenta.
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
    Stationary(atoms)  # Remove center of mass translation
    ZeroRotation(atoms)  # Remove rotations

    # Initialize Velocity Verlet integrator.
    integrator = VelocityVerlet(atoms, timestep=timestep_fs * ase.units.fs)

    # Run molecular dynamics.
    frames = np.zeros((num_steps, len(atoms), 3))
    dipoles = np.zeros((num_steps, 1, 3))
    potential_energy = np.zeros((num_steps,))
    kinetic_energy = np.zeros((num_steps,))
    total_energy = np.zeros((num_steps,))
    for i in range(num_steps):
        # Run 1 time step.
        integrator.run(1)
        # Save current frame and keep track of energies.
        frames[i] = atoms.get_positions()
        potential_energy[i] = atoms.get_potential_energy()
        kinetic_energy[i] = atoms.get_kinetic_energy()
        total_energy[i] = atoms.get_total_energy()
        dipoles[i] = atoms.get_dipole_moment()
        # Occasionally print progress.q
        if i % printfreq == 0:
            print(
                f"step {i:5d} epot {potential_energy[i]: 5.3f} ekin {kinetic_energy[i]: 5.3f} etot {total_energy[i]: 5.3f}"
            )
