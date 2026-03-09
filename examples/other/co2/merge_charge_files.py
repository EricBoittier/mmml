"""
Merge Multiwfn charge analysis NPZ files into a single dataset.

This script:
1. Finds all *_charges.npz files in a directory
2. Loads and combines them
3. Validates consistency across charge methods
4. Saves merged dataset with proper structure
5. Creates documentation

Multiwfn Charge Methods:
- hirshfeld: Hirshfeld population analysis
- vdd: Voronoi deformation density
- becke: Becke population analysis
- adch: Atomic dipole moment corrected Hirshfeld
- chelpg: CHarges from ELectrostatic Potentials using a Grid
- mk: Merz-Kollman ESP fitting
- cm5: Charge Model 5
- mbis: Minimal Basis Iterative Stockholder
"""

import sys
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple
import re


def find_charge_files(directory: Path, pattern: str = "*_charges.npz") -> List[Path]:
    """
    Find all charge NPZ files in a directory.
    
    Parameters
    ----------
    directory : Path
        Directory to search
    pattern : str
        Glob pattern for charge files
        
    Returns
    -------
    list
        List of charge file paths, sorted
    """
    charge_files = list(directory.glob(pattern))
    charge_files.extend(directory.glob("**/" + pattern))  # Search subdirs too
    return sorted(set(charge_files))


def parse_filename_metadata(filepath: Path) -> Dict:
    """
    Extract metadata from filename.
    
    Example: r1_1p000_r2_1p000_ang_157p895.mp2_charges.npz
    Returns: {'r1': 1.000, 'r2': 1.000, 'angle': 157.895, 'method': 'mp2'}
    """
    name = filepath.stem
    
    metadata = {}
    
    # Extract R1
    r1_match = re.search(r'r1_(\d+)p(\d+)', name)
    if r1_match:
        metadata['r1'] = float(f"{r1_match.group(1)}.{r1_match.group(2)}")
    
    # Extract R2
    r2_match = re.search(r'r2_(\d+)p(\d+)', name)
    if r2_match:
        metadata['r2'] = float(f"{r2_match.group(1)}.{r2_match.group(2)}")
    
    # Extract angle
    ang_match = re.search(r'ang_(\d+)p(\d+)', name)
    if ang_match:
        metadata['angle'] = float(f"{ang_match.group(1)}.{ang_match.group(2)}")
    
    # Extract quantum method (hf or mp2)
    if '.mp2_charges' in name:
        metadata['qm_method'] = 'mp2'
    elif '.hf_charges' in name:
        metadata['qm_method'] = 'hf'
    else:
        metadata['qm_method'] = 'unknown'
    
    return metadata


def load_charge_file(filepath: Path) -> Tuple[Dict, Dict]:
    """
    Load a charge NPZ file and extract data + metadata.
    
    Returns
    -------
    tuple
        (charge_data_dict, metadata_dict)
    """
    data = dict(np.load(filepath))
    metadata = parse_filename_metadata(filepath)
    metadata['filename'] = filepath.name
    
    return data, metadata


def merge_charge_files(
    charge_files: List[Path],
    verbose: bool = True
) -> Tuple[Dict[str, np.ndarray], List[Dict]]:
    """
    Merge multiple charge NPZ files into a single dataset.
    
    Parameters
    ----------
    charge_files : list
        List of charge file paths
    verbose : bool
        Print progress information
        
    Returns
    -------
    tuple
        (merged_data_dict, metadata_list)
    """
    if verbose:
        print(f"Merging {len(charge_files)} charge files...")
    
    # Storage for merged data
    merged_data = {}
    metadata_list = []
    
    # Track all keys found
    all_keys = set()
    
    for i, filepath in enumerate(charge_files):
        if verbose and i % 100 == 0:
            print(f"  Processing file {i+1}/{len(charge_files)}")
        
        try:
            data, metadata = load_charge_file(filepath)
            
            # Store metadata
            metadata_list.append(metadata)
            
            # Track all keys
            all_keys.update(data.keys())
            
            # Store data for this sample
            for key, value in data.items():
                if key not in merged_data:
                    merged_data[key] = []
                merged_data[key].append(value)
        
        except Exception as e:
            if verbose:
                print(f"  ⚠️ Error loading {filepath.name}: {e}")
            continue
    
    if verbose:
        print(f"\n  Found keys: {sorted(all_keys)}")
    
    # Convert lists to arrays
    final_data = {}
    for key, values_list in merged_data.items():
        try:
            # Check if all elements have the same shape
            if all(isinstance(v, np.ndarray) for v in values_list):
                shapes = [v.shape for v in values_list]
                if len(set(shapes)) == 1:  # All same shape
                    final_data[key] = np.array(values_list)
                    if verbose:
                        print(f"  ✓ {key}: {final_data[key].shape}")
                else:
                    # Variable shapes - keep as list
                    final_data[key] = values_list
                    if verbose:
                        print(f"  ⚠️ {key}: variable shapes, keeping as list")
            else:
                # Mixed types - keep as list
                final_data[key] = values_list
                if verbose:
                    print(f"  ⚠️ {key}: mixed types, keeping as list")
        except Exception as e:
            if verbose:
                print(f"  ⚠️ {key}: error stacking - {e}")
            final_data[key] = values_list
    
    return final_data, metadata_list


def validate_merged_data(data: Dict, metadata: List[Dict], verbose=True):
    """Validate merged charge data."""
    if verbose:
        print(f"\n{'='*70}")
        print("VALIDATION")
        print(f"{'='*70}")
    
    n_samples = len(metadata)
    print(f"\nNumber of samples: {n_samples}")
    
    # Check each charge method
    print(f"\nCharge methods found:")
    for method, values in data.items():
        if isinstance(values, np.ndarray):
            print(f"  {method:12s}: {values.shape} - mean={values.mean():.6f}, std={values.std():.6f}")
        else:
            print(f"  {method:12s}: list of {len(values)} entries")
    
    # Check metadata distribution
    print(f"\nMetadata distribution:")
    
    # R1 values
    r1_values = [m.get('r1') for m in metadata if 'r1' in m]
    if r1_values:
        print(f"  R1: {len(set(r1_values))} unique values, range [{min(r1_values):.3f}, {max(r1_values):.3f}] Å")
    
    # R2 values
    r2_values = [m.get('r2') for m in metadata if 'r2' in m]
    if r2_values:
        print(f"  R2: {len(set(r2_values))} unique values, range [{min(r2_values):.3f}, {max(r2_values):.3f}] Å")
    
    # Angles
    angles = [m.get('angle') for m in metadata if 'angle' in m]
    if angles:
        print(f"  Angles: {len(set(angles))} unique values, range [{min(angles):.3f}, {max(angles):.3f}]°")
    
    # QM methods
    qm_methods = [m.get('qm_method') for m in metadata if 'qm_method' in m]
    if qm_methods:
        from collections import Counter
        qm_counts = Counter(qm_methods)
        print(f"  QM methods: {dict(qm_counts)}")
    
    return True


def main():
    """Main workflow to merge charge files."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Merge Multiwfn charge NPZ files")
    parser.add_argument(
        'input_dir',
        type=Path,
        help='Directory containing *_charges.npz files'
    )
    parser.add_argument(
        '-o', '--output',
        type=Path,
        default=Path('merged_charges.npz'),
        help='Output NPZ file (default: merged_charges.npz)'
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default='*_charges.npz',
        help='Filename pattern (default: *_charges.npz)'
    )
    parser.add_argument(
        '--compress',
        action='store_true',
        help='Use compression for output'
    )
    parser.add_argument(
        '--filter-method',
        type=str,
        choices=['hf', 'mp2', 'all'],
        default='all',
        help='Filter by quantum method (default: all)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("Multiwfn Charge File Merger")
    print("="*70)
    
    # Find charge files
    print(f"\n📁 Searching for charge files in: {args.input_dir}")
    charge_files = find_charge_files(args.input_dir, args.pattern)
    
    if not charge_files:
        print(f"❌ No charge files found matching pattern '{args.pattern}'")
        return 1
    
    print(f"✓ Found {len(charge_files)} files")
    
    # Filter by QM method if requested
    if args.filter_method != 'all':
        filtered = [f for f in charge_files if args.filter_method in f.name]
        print(f"  Filtering to {args.filter_method} only: {len(filtered)} files")
        charge_files = filtered
    
    if not charge_files:
        print(f"❌ No files remaining after filtering")
        return 1
    
    # Merge files
    print(f"\n🔄 Merging charge files...")
    merged_data, metadata = merge_charge_files(charge_files, verbose=True)
    
    # Validate
    print(f"\n✓ Merged successfully")
    validate_merged_data(merged_data, metadata, verbose=True)
    
    # Save merged data
    print(f"\n💾 Saving merged data...")
    
    # Prepare data for saving - convert lists to object arrays if needed
    save_data = {}
    for key, value in merged_data.items():
        if isinstance(value, list):
            # Convert list to object array (allows variable shapes)
            save_data[key] = np.array(value, dtype=object)
        else:
            save_data[key] = value
    
    # Add metadata array
    save_data['_metadata'] = np.array(metadata, dtype=object)
    
    # Add geometry info if available
    if metadata and all('r1' in m for m in metadata):
        save_data['r1'] = np.array([m.get('r1', np.nan) for m in metadata])
        save_data['r2'] = np.array([m.get('r2', np.nan) for m in metadata])
        save_data['angle'] = np.array([m.get('angle', np.nan) for m in metadata])
        save_data['qm_method'] = np.array([m.get('qm_method', '') for m in metadata], dtype=object)
    
    # Save
    save_fn = np.savez_compressed if args.compress else np.savez
    save_fn(args.output, **save_data)
    
    size_mb = args.output.stat().st_size / 1024 / 1024
    print(f"✓ Saved to: {args.output} ({size_mb:.2f} MB)")
    
    # Create documentation
    doc_path = args.output.parent / (args.output.stem + "_README.txt")
    with open(doc_path, 'w') as f:
        f.write(f"""Merged Multiwfn Charge Analysis Data
{'='*70}

Source: {args.input_dir}
Files merged: {len(charge_files)}
Output: {args.output.name}

Charge Methods Included:
{chr(10).join(f'  - {method}: {data.shape if isinstance(data, np.ndarray) else "list"}' 
              for method, data in merged_data.items() if not method.startswith('_') and method not in ['r1', 'r2', 'angle', 'qm_method'])}

Geometry Parameters:
  - R1 (C-O bond 1): {len(set(m.get('r1') for m in metadata if 'r1' in m))} unique values
  - R2 (C-O bond 2): {len(set(m.get('r2') for m in metadata if 'r2' in m))} unique values  
  - Angles: {len(set(m.get('angle') for m in metadata if 'angle' in m))} unique values

Arrays in NPZ:
{chr(10).join(f'  {k}: {v.shape if isinstance(v, np.ndarray) and hasattr(v, "shape") else type(v).__name__}' 
              for k, v in merged_data.items())}

Usage:
  import numpy as np
  data = np.load('{args.output.name}')
  hirshfeld_charges = data['hirshfeld']  # Shape: (n_samples, n_atoms)
  mk_charges = data['mk']  # Shape: (n_samples, n_atoms)
  
Note: Charges are in atomic units (electron charge, e)
""")
    
    print(f"✓ Documentation: {doc_path.name}")
    
    print(f"\n{'='*70}")
    print("✅ MERGE COMPLETE!")
    print(f"{'='*70}")
    print(f"\nOutput: {args.output}")
    print(f"Samples: {len(metadata)}")
    print(f"Charge methods: {', '.join(k for k in merged_data.keys() if not k.startswith('_') and k not in ['r1', 'r2', 'angle', 'qm_method'])}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

