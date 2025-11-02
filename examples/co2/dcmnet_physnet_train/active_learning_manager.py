#!/usr/bin/env python3
"""
Active Learning Dataset Manager

Manages structures saved from unstable/challenging simulations for retraining.

Features:
- Collect structures from multiple MD runs
- Filter/deduplicate similar structures
- Export to formats for QM calculations (ORCA, Gaussian, etc.)
- Integrate back into training data

Usage:
    # List all saved structures
    python active_learning_manager.py --list --source ./md_runs/*/active_learning
    
    # Export unique structures for QM calculations
    python active_learning_manager.py --export-xyz --output ./qm_candidates \
        --max-structures 100 --min-force 3.0
    
    # Prepare dataset from QM results
    python active_learning_manager.py --create-dataset \
        --qm-results ./qm_results/*.npz \
        --output ./training_data_v2
"""

import sys
from pathlib import Path
import numpy as np
import argparse
from collections import defaultdict
from typing import List, Dict, Tuple
import json

# ASE for structure manipulation
try:
    from ase import Atoms
    from ase.io import write as ase_write
    HAS_ASE = True
except ImportError:
    HAS_ASE = False
    print("⚠️  ASE not available, limited functionality")


def load_unstable_structure(npz_file: Path) -> Dict:
    """Load a saved unstable structure."""
    data = np.load(npz_file, allow_pickle=True)
    
    return {
        'file': npz_file,
        'positions': data['positions'],
        'velocities': data.get('velocities', None),
        'forces': data['forces'],
        'atomic_numbers': data['atomic_numbers'],
        'energy': float(data['energy']),
        'dipole_physnet': data.get('dipole_physnet', None),
        'dipole_dcmnet': data.get('dipole_dcmnet', None),
        'step': int(data['step']),
        'time_fs': float(data['time_fs']),
        'max_force': float(data['max_force']),
        'max_velocity': float(data['max_velocity']),
        'max_position': float(data['max_position']),
        'temperature': float(data.get('temperature', 0)),
        'ensemble': str(data.get('ensemble', 'unknown')),
        'timestep': float(data.get('timestep', 0)),
        'reason': str(data.get('reason', 'unknown')),
        'instability_type': str(data.get('instability_type', 'unknown')),
    }


def find_all_structures(source_patterns: List[str]) -> List[Dict]:
    """Find all saved structures matching patterns."""
    structures = []
    
    for pattern in source_patterns:
        # Convert pattern to Path and expand glob
        path = Path(pattern)
        
        if path.is_dir():
            # Search for .npz files in directory
            for npz_file in path.glob('*.npz'):
                try:
                    struct = load_unstable_structure(npz_file)
                    structures.append(struct)
                except Exception as e:
                    print(f"⚠️  Failed to load {npz_file}: {e}")
        else:
            # Glob pattern
            for npz_file in Path('.').glob(pattern):
                try:
                    struct = load_unstable_structure(npz_file)
                    structures.append(struct)
                except Exception as e:
                    print(f"⚠️  Failed to load {npz_file}: {e}")
    
    return structures


def compute_structure_similarity(pos1: np.ndarray, pos2: np.ndarray, 
                                 threshold: float = 0.5) -> bool:
    """
    Check if two structures are similar (same up to translation/rotation).
    
    Simple check: RMSD after centering < threshold
    """
    if pos1.shape != pos2.shape:
        return False
    
    # Center both
    pos1_centered = pos1 - pos1.mean(axis=0)
    pos2_centered = pos2 - pos2.mean(axis=0)
    
    # Compute RMSD
    rmsd = np.sqrt(np.mean((pos1_centered - pos2_centered)**2))
    
    return rmsd < threshold


def deduplicate_structures(structures: List[Dict], 
                          rmsd_threshold: float = 0.5) -> List[Dict]:
    """Remove duplicate/similar structures."""
    unique = []
    
    for struct in structures:
        is_unique = True
        for unique_struct in unique:
            if compute_structure_similarity(
                struct['positions'], 
                unique_struct['positions'], 
                rmsd_threshold
            ):
                is_unique = False
                # Keep the one with higher force (more challenging)
                if struct['max_force'] > unique_struct['max_force']:
                    unique.remove(unique_struct)
                    unique.append(struct)
                break
        
        if is_unique:
            unique.append(struct)
    
    return unique


def export_to_xyz(structures: List[Dict], output_dir: Path):
    """Export structures to XYZ files for QM calculations."""
    if not HAS_ASE:
        print("❌ ASE required for XYZ export")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, struct in enumerate(structures):
        atoms = Atoms(
            numbers=struct['atomic_numbers'],
            positions=struct['positions']
        )
        
        # Add metadata as info
        atoms.info['source'] = str(struct['file'])
        atoms.info['max_force'] = struct['max_force']
        atoms.info['reason'] = struct['reason']
        
        xyz_file = output_dir / f'structure_{i:04d}.xyz'
        ase_write(xyz_file, atoms)
        
        # Also save a JSON with full metadata
        metadata_file = output_dir / f'structure_{i:04d}.json'
        metadata = {
            'source': str(struct['file']),
            'max_force': struct['max_force'],
            'max_velocity': struct['max_velocity'],
            'temperature': struct['temperature'],
            'time_fs': struct['time_fs'],
            'reason': struct['reason'],
            'instability_type': struct['instability_type'],
            'model_energy': struct['energy'],
        }
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    print(f"✅ Exported {len(structures)} structures to {output_dir}")
    print(f"   Next steps:")
    print(f"   1. Run QM calculations on {output_dir}/*.xyz")
    print(f"   2. Convert QM results to NPZ format")
    print(f"   3. Use --create-dataset to merge with training data")


def print_structure_summary(structures: List[Dict]):
    """Print summary statistics of collected structures."""
    if not structures:
        print("No structures found.")
        return
    
    print(f"\n{'='*70}")
    print(f"ACTIVE LEARNING STRUCTURE SUMMARY")
    print(f"{'='*70}")
    print(f"Total structures: {len(structures)}")
    
    # Group by reason
    by_reason = defaultdict(list)
    for s in structures:
        by_reason[s['reason']].append(s)
    
    print(f"\nBy failure reason:")
    for reason, structs in sorted(by_reason.items()):
        print(f"  {reason}: {len(structs)}")
    
    # Group by instability type
    by_type = defaultdict(list)
    for s in structures:
        by_type[s['instability_type']].append(s)
    
    print(f"\nBy instability type:")
    for itype, structs in sorted(by_type.items()):
        print(f"  {itype}: {len(structs)}")
    
    # Force statistics
    forces = [s['max_force'] for s in structures]
    print(f"\nMax force statistics (eV/Å):")
    print(f"  Min: {np.min(forces):.2f}")
    print(f"  Max: {np.max(forces):.2f}")
    print(f"  Mean: {np.mean(forces):.2f}")
    print(f"  Median: {np.median(forces):.2f}")
    
    # Temperature distribution
    temps = [s['temperature'] for s in structures]
    unique_temps = set(temps)
    print(f"\nTemperatures: {sorted(unique_temps)} K")
    
    # Molecule types (by number of atoms)
    n_atoms_list = [len(s['atomic_numbers']) for s in structures]
    unique_n_atoms = set(n_atoms_list)
    print(f"\nMolecule sizes: {sorted(unique_n_atoms)} atoms")
    
    print(f"\nTop 10 most challenging structures (by max force):")
    sorted_structs = sorted(structures, key=lambda x: x['max_force'], reverse=True)
    for i, s in enumerate(sorted_structs[:10]):
        print(f"  {i+1}. {s['file'].name}: F_max={s['max_force']:.2f} eV/Å "
              f"({s['reason']}, {s['instability_type']})")


def main():
    parser = argparse.ArgumentParser(description='Active Learning Dataset Manager')
    
    # Input
    parser.add_argument('--source', type=str, nargs='+', required=True,
                       help='Source directories or glob patterns for unstable structures')
    
    # Actions
    parser.add_argument('--list', action='store_true',
                       help='List and summarize all structures')
    parser.add_argument('--export-xyz', action='store_true',
                       help='Export structures to XYZ for QM calculations')
    parser.add_argument('--create-dataset', action='store_true',
                       help='Create training dataset from QM results')
    
    # Filtering
    parser.add_argument('--min-force', type=float, default=0.0,
                       help='Minimum max force (eV/Å) to include')
    parser.add_argument('--max-force', type=float, default=1000.0,
                       help='Maximum max force (eV/Å) to include')
    parser.add_argument('--deduplicate', action='store_true',
                       help='Remove similar structures (RMSD-based)')
    parser.add_argument('--rmsd-threshold', type=float, default=0.5,
                       help='RMSD threshold (Å) for deduplication')
    parser.add_argument('--max-structures', type=int, default=None,
                       help='Maximum number of structures to export')
    
    # Output
    parser.add_argument('--output', type=Path, default=Path('./al_export'),
                       help='Output directory')
    
    # QM data (for --create-dataset)
    parser.add_argument('--qm-results', type=str, nargs='+', default=[],
                       help='QM result files (NPZ format)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("ACTIVE LEARNING DATASET MANAGER")
    print("="*70)
    
    # Find all structures
    print(f"\n1. Searching for structures...")
    print(f"   Sources: {args.source}")
    structures = find_all_structures(args.source)
    print(f"   Found {len(structures)} structures")
    
    if not structures:
        print("No structures found. Exiting.")
        return
    
    # Filter by force
    if args.min_force > 0 or args.max_force < 1000:
        print(f"\n2. Filtering by force range [{args.min_force}, {args.max_force}] eV/Å...")
        structures = [s for s in structures 
                     if args.min_force <= s['max_force'] <= args.max_force]
        print(f"   {len(structures)} structures remain")
    
    # Deduplicate
    if args.deduplicate:
        print(f"\n3. Deduplicating (RMSD threshold: {args.rmsd_threshold} Å)...")
        original_count = len(structures)
        structures = deduplicate_structures(structures, args.rmsd_threshold)
        print(f"   Removed {original_count - len(structures)} duplicates")
        print(f"   {len(structures)} unique structures remain")
    
    # Limit number
    if args.max_structures and len(structures) > args.max_structures:
        print(f"\n4. Limiting to {args.max_structures} most challenging structures...")
        structures = sorted(structures, key=lambda x: x['max_force'], reverse=True)
        structures = structures[:args.max_structures]
    
    # List/summarize
    if args.list:
        print_structure_summary(structures)
    
    # Export to XYZ
    if args.export_xyz:
        print(f"\n{'='*70}")
        print(f"EXPORTING TO XYZ")
        print(f"{'='*70}")
        export_to_xyz(structures, args.output)
    
    # Create training dataset
    if args.create_dataset:
        print(f"\n{'='*70}")
        print(f"CREATING TRAINING DATASET")
        print(f"{'='*70}")
        print("⚠️  Not implemented yet!")
        print("This will merge QM results with existing training data.")
        print("For now, manually convert QM outputs to the training format.")
    
    print(f"\n{'='*70}")
    print("✅ DONE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()

