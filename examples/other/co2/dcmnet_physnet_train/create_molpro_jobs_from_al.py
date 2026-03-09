#!/usr/bin/env python3
"""
Create Molpro Job Scripts from Active Learning Structures

Converts saved unstable structures to Molpro input files compatible
with existing job submission infrastructure.

Usage:
    python create_molpro_jobs_from_al.py \
        --al-structures ./md_*/active_learning/*.npz \
        --output-dir ./molpro_al_jobs \
        --template ./template.inp \
        --ntasks 16 \
        --mem 132G
"""

import sys
from pathlib import Path
import numpy as np
import argparse
from typing import List, Dict

try:
    from ase import Atoms
    HAS_ASE = True
except ImportError:
    HAS_ASE = False


def load_al_structure(npz_file: Path) -> Dict:
    """Load active learning structure."""
    data = np.load(npz_file, allow_pickle=True)
    return {
        'file': npz_file,
        'positions': data['positions'],
        'atomic_numbers': data['atomic_numbers'],
        'max_force': float(data['max_force']),
        'reason': str(data.get('reason', 'unknown')),
    }


def create_molpro_input_from_structure(atoms: Atoms, tag: str, 
                                       step_x: float = 0.2, 
                                       step_y: float = 0.2, 
                                       step_z: float = 0.2) -> str:
    """
    Create Molpro input for a single structure.
    
    Similar format to the user's existing inputs.
    """
    lines = []
    
    lines.append(f"***, Active Learning Structure {tag}")
    lines.append("memory,1800,m")
    lines.append("angstrom")
    lines.append("symmetry,nosym")
    lines.append("orient,noorient")
    lines.append("")
    lines.append("basis=aug-cc-pVTZ")
    lines.append("")
    
    # Cartesian geometry (not Z-matrix)
    lines.append("geometry={")
    for atom in atoms:
        symbol = atom.symbol
        x, y, z = atom.position
        lines.append(f"  {symbol:2s}, {x:12.8f}, {y:12.8f}, {z:12.8f}")
    lines.append("}")
    lines.append("")
    
    # HF calculation
    lines.append("{df-hf;  wf,charge=0,spin=0}")
    lines.append(f"{{put,molden,{tag}.hf.molden}}")
    lines.append("")
    
    # MP2 energy + dipole
    lines.append("! MP2 energy + dipole (relaxed 1PDM)")
    lines.append("{df-mp2; expec}")
    lines.append("")
    
    # Forces
    lines.append("! Analytic gradient for last method (MP2) -> prints Cartesian forces")
    lines.append("forces")
    lines.append("")
    
    # Cubes
    lines.append("! Dump MP2 density and electrostatic potential cubes")
    lines.append(f"{{cube,cubes/density/density_{tag}.cube; density;   step,{step_x},{step_y},{step_z}}}")
    lines.append(f"{{cube,cubes/esp/esp_{tag}.cube;  potential; step,{step_x},{step_y},{step_z}}}")
    lines.append(f"{{put,molden,{tag}.mp2.molden}}")
    lines.append("")
    
    # Save results to CSV
    lines.append("! Save results to CSV")
    lines.append(f"table, energy, dipx, dipy, dipz, grms, gmax")
    lines.append(f"table, save, logs/results_{tag}.csv")
    lines.append("")
    
    return "\n".join(lines)


def create_sbatch_script(input_files: List[str], job_name: str,
                        ntasks: int = 16, mem: str = '132G',
                        time: str = '24:00:00') -> str:
    """Create SLURM sbatch script."""
    n_jobs = len(input_files)
    
    lines = []
    lines.append("#!/bin/bash")
    lines.append("")
    lines.append(f"#SBATCH --job-name={job_name}")
    lines.append(f"#SBATCH --output=logs/al_%A_%a.out")
    lines.append(f"#SBATCH --error=logs/al_%A_%a.err")
    lines.append(f"#SBATCH --nodes=1")
    lines.append(f"#SBATCH --ntasks={ntasks}")
    lines.append(f"#SBATCH --time={time}")
    lines.append(f"#SBATCH --mem={mem}")
    lines.append(f"#SBATCH --array=0-{n_jobs-1}")
    lines.append("")
    lines.append("# -- Modules / env ------------------------------------------------------------")
    lines.append("module load molpro")
    lines.append("")
    lines.append("# -- Setup --------------------------------------------------------------------")
    lines.append("mkdir -p logs cubes/esp cubes/density")
    lines.append("")
    lines.append("# Map array index to input file")
    lines.append("INPUT_FILES=(")
    for inp in input_files:
        lines.append(f"  {inp}")
    lines.append(")")
    lines.append("")
    lines.append("INP=${INPUT_FILES[$SLURM_ARRAY_TASK_ID]}")
    lines.append("OUT=logs/${INP%.inp}.molpro.out")
    lines.append("")
    lines.append("# Run Molpro")
    lines.append('molpro -n "${SLURM_NTASKS}" "$INP" > "$OUT"')
    lines.append("")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='Create Molpro jobs from AL structures')
    
    parser.add_argument('--al-structures', type=str, nargs='+', required=True,
                       help='Active learning structure files (glob patterns)')
    parser.add_argument('--output-dir', type=Path, required=True,
                       help='Output directory for Molpro inputs and scripts')
    parser.add_argument('--job-name', type=str, default='al_structures',
                       help='SLURM job name')
    parser.add_argument('--ntasks', type=int, default=16,
                       help='Number of tasks per job')
    parser.add_argument('--mem', type=str, default='132G',
                       help='Memory per job')
    parser.add_argument('--time', type=str, default='24:00:00',
                       help='Time limit')
    parser.add_argument('--cube-step', type=float, default=0.2,
                       help='Cube grid spacing (Å)')
    parser.add_argument('--max-structures', type=int, default=None,
                       help='Maximum number of structures to process')
    parser.add_argument('--sort-by-force', action='store_true',
                       help='Prioritize highest-force structures')
    
    args = parser.parse_args()
    
    print("="*70)
    print("MOLPRO JOB GENERATOR FOR ACTIVE LEARNING")
    print("="*70)
    
    # Find all AL structures
    al_files = []
    for pattern in args.al_structures:
        al_files.extend(Path('.').glob(pattern))
    
    al_files = sorted(set(al_files))
    print(f"\nFound {len(al_files)} active learning structures")
    
    if not al_files:
        print("No structures found. Exiting.")
        return
    
    # Load structures
    print(f"\nLoading structures...")
    structures = []
    for npz_file in al_files:
        try:
            struct = load_al_structure(npz_file)
            structures.append(struct)
        except Exception as e:
            print(f"  ⚠️  Failed to load {npz_file.name}: {e}")
    
    print(f"  Loaded {len(structures)} structures")
    
    # Sort by max force if requested
    if args.sort_by_force:
        print(f"\nSorting by max force (highest first)...")
        structures = sorted(structures, key=lambda x: x['max_force'], reverse=True)
    
    # Limit number
    if args.max_structures and len(structures) > args.max_structures:
        print(f"\nLimiting to {args.max_structures} structures...")
        structures = structures[:args.max_structures]
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = args.output_dir / 'logs'
    logs_dir.mkdir(exist_ok=True)
    
    cubes_dir = args.output_dir / 'cubes'
    (cubes_dir / 'esp').mkdir(parents=True, exist_ok=True)
    (cubes_dir / 'density').mkdir(parents=True, exist_ok=True)
    
    # Generate Molpro inputs
    print(f"\nGenerating Molpro input files...")
    input_files = []
    
    for i, struct in enumerate(structures):
        tag = f"al_{i:04d}_f{struct['max_force']:.1f}"
        
        # Create ASE Atoms object
        atoms = Atoms(
            numbers=struct['atomic_numbers'],
            positions=struct['positions']
        )
        
        # Generate Molpro input
        input_content = create_molpro_input_from_structure(
            atoms, tag,
            step_x=args.cube_step,
            step_y=args.cube_step,
            step_z=args.cube_step,
        )
        
        # Write input file
        input_file = args.output_dir / f"{tag}.inp"
        with open(input_file, 'w') as f:
            f.write(input_content)
        
        input_files.append(f"{tag}.inp")
        
        if (i + 1) % 20 == 0:
            print(f"  Generated {i+1}/{len(structures)}...")
    
    print(f"\n✅ Generated {len(input_files)} Molpro input files")
    
    # Create SLURM sbatch script
    print(f"\nGenerating SLURM submission script...")
    sbatch_content = create_sbatch_script(
        input_files,
        job_name=args.job_name,
        ntasks=args.ntasks,
        mem=args.mem,
        time=args.time,
    )
    
    sbatch_file = args.output_dir / f"{args.job_name}.sbatch"
    with open(sbatch_file, 'w') as f:
        f.write(sbatch_content)
    
    print(f"✅ Created SLURM script: {sbatch_file}")
    
    # Summary
    print(f"\n{'='*70}")
    print("READY TO RUN")
    print(f"{'='*70}")
    print(f"\nGenerated files:")
    print(f"  Input files: {args.output_dir}/*.inp ({len(input_files)} files)")
    print(f"  SLURM script: {sbatch_file}")
    print(f"\nTo submit:")
    print(f"  cd {args.output_dir}")
    print(f"  sbatch {sbatch_file.name}")
    print(f"\nAfter jobs complete:")
    print(f"  python convert_molpro_to_training.py \\")
    print(f"    --molpro-outputs {args.output_dir}/logs/*.molpro.out \\")
    print(f"    --cube-dir {args.output_dir}/cubes \\")
    print(f"    --merge-with ../physnet_train_charges/energies_forces_dipoles_train.npz \\")
    print(f"    --output ../physnet_train_charges/energies_forces_dipoles_train_v2.npz")
    print(f"\nThen retrain with augmented data!")


if __name__ == '__main__':
    if not HAS_ASE:
        print("❌ ASE required for this script")
        sys.exit(1)
    
    main()

