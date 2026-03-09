#!/usr/bin/env python3
"""
Prepare QM Input Files from Active Learning Structures

Creates input files for various QM software packages from XYZ files.

Supported:
- ORCA
- Gaussian
- Q-Chem
- Psi4

Usage:
    python prepare_qm_inputs.py \
        --xyz-dir ./qm_candidates \
        --qm-software orca \
        --method PBE0 \
        --basis def2-TZVP \
        --output ./orca_inputs
"""

import sys
from pathlib import Path
import argparse
import json

try:
    from ase.io import read as ase_read
    HAS_ASE = True
except ImportError:
    HAS_ASE = False
    print("❌ ASE required for this script")
    sys.exit(1)


def create_orca_input(atoms, method='PBE0', basis='def2-TZVP', 
                     calc_esp=True, calc_dipole=True) -> str:
    """
    Create ORCA input file content.
    
    Example output:
        ! PBE0 def2-TZVP TightSCF
        ! EnGrad
        
        %elprop
          Polar 1
          Dipole true
        end
        
        * xyz 0 1
        C  0.0  0.0  0.0
        O  0.0  0.0  1.16
        O  0.0  0.0 -1.16
        *
    """
    lines = []
    
    # Header with method and basis
    lines.append(f"! {method} {basis} TightSCF")
    lines.append("! EnGrad  # Energy and gradient")
    lines.append("")
    
    # Electrostatic properties
    if calc_esp or calc_dipole:
        lines.append("%elprop")
        if calc_dipole:
            lines.append("  Polar 1")
            lines.append("  Dipole true")
        if calc_esp:
            lines.append("  # ESP calculation on VDW surface")
            lines.append("  %method")
            lines.append("    COSX true")
            lines.append("  end")
        lines.append("end")
        lines.append("")
    
    # Parallelization
    lines.append("%pal")
    lines.append("  nprocs 8")
    lines.append("end")
    lines.append("")
    
    # Geometry
    # Determine charge and multiplicity (assume neutral singlet for now)
    charge = 0
    mult = 1
    
    lines.append(f"* xyz {charge} {mult}")
    for atom in atoms:
        symbol = atom.symbol
        x, y, z = atom.position
        lines.append(f"{symbol:2s}  {x:12.6f}  {y:12.6f}  {z:12.6f}")
    lines.append("*")
    lines.append("")
    
    return "\n".join(lines)


def create_gaussian_input(atoms, method='PBE0', basis='6-311G**',
                         calc_esp=True, calc_dipole=True) -> str:
    """
    Create Gaussian input file content.
    
    Example:
        %nprocshared=8
        %mem=16GB
        #p PBE0/6-311G** Force Pop=MK
        
        CO2 single point
        
        0 1
        C  0.0  0.0  0.0
        O  0.0  0.0  1.16
        O  0.0  0.0 -1.16
        
    """
    lines = []
    
    # Resource allocation
    lines.append("%nprocshared=8")
    lines.append("%mem=16GB")
    lines.append("")
    
    # Route section
    route = f"#p {method}/{basis} Force"
    if calc_esp:
        route += " Pop=MK"  # Merz-Kollman ESP charges
    if calc_dipole:
        route += " Polar"
    lines.append(route)
    lines.append("")
    
    # Title
    lines.append("Active learning structure - single point")
    lines.append("")
    
    # Charge and multiplicity
    charge = 0
    mult = 1
    lines.append(f"{charge} {mult}")
    
    # Geometry
    for atom in atoms:
        symbol = atom.symbol
        x, y, z = atom.position
        lines.append(f"{symbol:2s}  {x:12.6f}  {y:12.6f}  {z:12.6f}")
    lines.append("")
    
    return "\n".join(lines)


def create_qchem_input(atoms, method='PBE0', basis='def2-TZVP',
                      calc_esp=True, calc_dipole=True) -> str:
    """Create Q-Chem input file."""
    lines = []
    
    # $rem section
    lines.append("$rem")
    lines.append(f"  method          {method}")
    lines.append(f"  basis           {basis}")
    lines.append("  jobtype         force")
    lines.append("  scf_convergence 8")
    if calc_esp:
        lines.append("  esp_charges     mk")
    if calc_dipole:
        lines.append("  dipole          true")
    lines.append("$end")
    lines.append("")
    
    # Molecule section
    lines.append("$molecule")
    lines.append("  0 1")
    for atom in atoms:
        symbol = atom.symbol
        x, y, z = atom.position
        lines.append(f"  {symbol:2s}  {x:12.6f}  {y:12.6f}  {z:12.6f}")
    lines.append("$end")
    lines.append("")
    
    return "\n".join(lines)


def create_psi4_input(atoms, method='PBE0', basis='def2-TZVP',
                     calc_esp=True, calc_dipole=True) -> str:
    """Create Psi4 input file."""
    lines = []
    
    # Memory and threads
    lines.append("memory 16 GB")
    lines.append("set_num_threads(8)")
    lines.append("")
    
    # Molecule
    lines.append("molecule {")
    lines.append("  0 1")
    for atom in atoms:
        symbol = atom.symbol
        x, y, z = atom.position
        lines.append(f"  {symbol:2s}  {x:12.6f}  {y:12.6f}  {z:12.6f}")
    lines.append("}")
    lines.append("")
    
    # Method and basis
    lines.append(f"set basis {basis}")
    lines.append("")
    
    # Calculation
    lines.append(f"energy, wfn = gradient('{method}', return_wfn=True)")
    
    if calc_dipole:
        lines.append("oeprop(wfn, 'DIPOLE', 'QUADRUPOLE')")
    
    if calc_esp:
        lines.append("# ESP calculation would require additional setup")
    
    lines.append("")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='Prepare QM input files')
    
    parser.add_argument('--xyz-dir', type=Path, required=True,
                       help='Directory with XYZ files')
    parser.add_argument('--qm-software', type=str, required=True,
                       choices=['orca', 'gaussian', 'qchem', 'psi4'],
                       help='QM software to create inputs for')
    parser.add_argument('--method', type=str, default='PBE0',
                       help='DFT method (default: PBE0)')
    parser.add_argument('--basis', type=str, default='def2-TZVP',
                       help='Basis set (default: def2-TZVP)')
    parser.add_argument('--calc-esp', action='store_true', default=True,
                       help='Calculate ESP (default: True)')
    parser.add_argument('--calc-dipole', action='store_true', default=True,
                       help='Calculate dipole (default: True)')
    parser.add_argument('--output', type=Path, default=Path('./qm_inputs'),
                       help='Output directory')
    
    args = parser.parse_args()
    
    print("="*70)
    print("QM INPUT FILE GENERATOR")
    print("="*70)
    print(f"Software: {args.qm_software}")
    print(f"Method: {args.method}")
    print(f"Basis: {args.basis}")
    print(f"ESP: {args.calc_esp}")
    print(f"Dipole: {args.calc_dipole}")
    
    # Find all XYZ files
    xyz_files = sorted(args.xyz_dir.glob('*.xyz'))
    print(f"\nFound {len(xyz_files)} XYZ files in {args.xyz_dir}")
    
    if not xyz_files:
        print("No XYZ files found. Exiting.")
        return
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    # File extensions
    ext_map = {
        'orca': '.inp',
        'gaussian': '.com',
        'qchem': '.in',
        'psi4': '.dat',
    }
    
    # Generator functions
    generator_map = {
        'orca': create_orca_input,
        'gaussian': create_gaussian_input,
        'qchem': create_qchem_input,
        'psi4': create_psi4_input,
    }
    
    # Process each structure
    print(f"\nGenerating {args.qm_software.upper()} input files...")
    
    for i, xyz_file in enumerate(xyz_files):
        # Read structure
        atoms = ase_read(xyz_file)
        
        # Create input
        generator = generator_map[args.qm_software]
        input_content = generator(
            atoms,
            method=args.method,
            basis=args.basis,
            calc_esp=args.calc_esp,
            calc_dipole=args.calc_dipole,
        )
        
        # Write input file
        ext = ext_map[args.qm_software]
        input_file = args.output / f"{xyz_file.stem}{ext}"
        with open(input_file, 'w') as f:
            f.write(input_content)
        
        # Also copy metadata JSON if it exists
        json_file = xyz_file.with_suffix('.json')
        if json_file.exists():
            import shutil
            shutil.copy(json_file, args.output / json_file.name)
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(xyz_files)}...")
    
    print(f"\n✅ Generated {len(xyz_files)} input files in {args.output}")
    
    # Print next steps
    print(f"\n{'='*70}")
    print("NEXT STEPS")
    print(f"{'='*70}")
    
    if args.qm_software == 'orca':
        print(f"1. Run ORCA calculations:")
        print(f"   cd {args.output}")
        print(f"   for inp in *.inp; do")
        print(f"     orca $inp > ${{inp%.inp}}.out")
        print(f"   done")
        print(f"")
        print(f"2. Convert outputs:")
        print(f"   python convert_qm_to_npz.py --orca-outputs {args.output}/*.out \\")
        print(f"     --output ./qm_training_data.npz")
    
    elif args.qm_software == 'gaussian':
        print(f"1. Run Gaussian calculations:")
        print(f"   cd {args.output}")
        print(f"   for com in *.com; do")
        print(f"     g16 < $com > ${{com%.com}}.log")
        print(f"   done")
        print(f"")
        print(f"2. Convert outputs:")
        print(f"   python convert_qm_to_npz.py --gaussian-outputs {args.output}/*.log \\")
        print(f"     --output ./qm_training_data.npz")
    
    elif args.qm_software == 'qchem':
        print(f"1. Run Q-Chem calculations:")
        print(f"   cd {args.output}")
        print(f"   for inp in *.in; do")
        print(f"     qchem $inp ${{inp%.in}}.out")
        print(f"   done")
    
    elif args.qm_software == 'psi4':
        print(f"1. Run Psi4 calculations:")
        print(f"   cd {args.output}")
        print(f"   for dat in *.dat; do")
        print(f"     psi4 $dat ${{dat%.dat}}.out")
        print(f"   done")
    
    print(f"\n3. Merge with training data and retrain")


if __name__ == '__main__':
    main()

