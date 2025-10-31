#!/usr/bin/env python3
"""
Test script for energy preprocessing functionality.
"""

import sys
from pathlib import Path
import numpy as np

# Add mmml to path
repo_root = Path(__file__).parent / "../../.."
sys.path.insert(0, str(repo_root.resolve()))

from mmml.data import DataConfig, load_npz
from mmml.data.preprocessing import (
    convert_energy_units,
    compute_atomic_energies,
    subtract_atomic_energies,
    scale_energies_by_atoms
)

def test_unit_conversion():
    """Test energy unit conversion."""
    print("\n" + "="*70)
    print("Test 1: Energy Unit Conversion")
    print("="*70)
    
    # Test Hartree to eV
    energies_hartree = np.array([1.0, 2.0, 3.0])
    energies_ev = convert_energy_units(energies_hartree, from_unit='hartree', to_unit='eV')
    expected = energies_hartree * 27.211386245988
    
    print(f"Original (Hartree): {energies_hartree}")
    print(f"Converted (eV): {energies_ev}")
    print(f"Expected (eV): {expected}")
    print(f"Match: {np.allclose(energies_ev, expected)}")
    
    # Test round-trip conversion
    back_to_hartree = convert_energy_units(energies_ev, from_unit='eV', to_unit='hartree')
    print(f"Back to Hartree: {back_to_hartree}")
    print(f"Round-trip match: {np.allclose(back_to_hartree, energies_hartree)}")
    
    return np.allclose(energies_ev, expected) and np.allclose(back_to_hartree, energies_hartree)


def test_atomic_energies():
    """Test atomic energy computation and subtraction."""
    print("\n" + "="*70)
    print("Test 2: Atomic Energy Reference Computation")
    print("="*70)
    
    # Create synthetic data with varied compositions
    # C: atomic number 6, O: atomic number 8
    # Need different compositions to properly fit atomic energies
    
    E_C = -1000.0  # eV
    E_O = -2000.0  # eV
    
    # Create molecules: CO2, CO, O2
    # Pad with zeros to max 3 atoms
    atomic_numbers_list = []
    n_atoms_list = []
    energies_list = []
    
    # 5 CO2 molecules: C + 2O
    for _ in range(5):
        atomic_numbers_list.append([6, 8, 8])
        n_atoms_list.append(3)
        energies_list.append(E_C + 2*E_O)
    
    # 5 CO molecules: C + O
    for _ in range(5):
        atomic_numbers_list.append([6, 8, 0])
        n_atoms_list.append(2)
        energies_list.append(E_C + E_O)
    
    # 5 O2 molecules: 2O
    for _ in range(5):
        atomic_numbers_list.append([8, 8, 0])
        n_atoms_list.append(2)
        energies_list.append(2*E_O)
    
    atomic_numbers = np.array(atomic_numbers_list)
    n_atoms = np.array(n_atoms_list)
    energies = np.array(energies_list)
    
    print(f"Synthetic data:")
    print(f"  5 CO2 molecules (C + 2O)")
    print(f"  5 CO molecules (C + O)")
    print(f"  5 O2 molecules (2O)")
    print(f"  True atomic energies: C={E_C:.1f} eV, O={E_O:.1f} eV")
    
    # Compute atomic energies
    atomic_energies = compute_atomic_energies(
        energies,
        atomic_numbers,
        n_atoms,
        method='linear_regression'
    )
    
    print(f"\nComputed atomic energies:")
    for z, e in atomic_energies.items():
        element = 'C' if z == 6 else 'O'
        true_e = E_C if z == 6 else E_O
        print(f"  {element} (Z={z}): {e:.2f} eV (true: {true_e:.2f} eV)")
    
    # Check if computed values match true values
    match_C = np.abs(atomic_energies[6] - E_C) < 0.1
    match_O = np.abs(atomic_energies[8] - E_O) < 0.1
    
    print(f"\nAccuracy:")
    print(f"  C match: {match_C}")
    print(f"  O match: {match_O}")
    
    # Subtract atomic energies
    binding_energies = subtract_atomic_energies(
        energies,
        atomic_numbers,
        n_atoms,
        atomic_energies
    )
    
    print(f"\nAfter subtracting atomic references:")
    print(f"  Binding energies should be ~0 for ideal gases: {np.mean(binding_energies):.4f} eV")
    print(f"  Std: {np.std(binding_energies):.4f} eV")
    
    return match_C and match_O


def test_scale_by_atoms():
    """Test per-atom energy scaling."""
    print("\n" + "="*70)
    print("Test 3: Scale Energies by Number of Atoms")
    print("="*70)
    
    # Create molecules with different numbers of atoms
    energies = np.array([100.0, 200.0, 300.0])
    n_atoms = np.array([2, 4, 6])
    
    print(f"Original energies: {energies}")
    print(f"Number of atoms: {n_atoms}")
    
    per_atom = scale_energies_by_atoms(energies, n_atoms)
    expected = np.array([50.0, 50.0, 50.0])
    
    print(f"Per-atom energies: {per_atom}")
    print(f"Expected: {expected}")
    print(f"Match: {np.allclose(per_atom, expected)}")
    
    return np.allclose(per_atom, expected)


def test_full_pipeline():
    """Test full preprocessing pipeline with real data."""
    print("\n" + "="*70)
    print("Test 4: Full Pipeline with Real CO2 Data")
    print("="*70)
    
    train_file = Path(__file__).parent.parent / "preclassified_data" / "energies_forces_dipoles_train.npz"
    
    if not train_file.exists():
        print(f"⚠️  Warning: Training file not found: {train_file}")
        print("   Skipping full pipeline test")
        return True
    
    print(f"Loading data from: {train_file}")
    
    # Load without preprocessing
    config_no_preprocessing = DataConfig(
        batch_size=32,
        targets=['energy'],
        num_atoms=60,
    )
    data_raw = load_npz(train_file, config=config_no_preprocessing, validate=False, verbose=False)
    
    print(f"\nRaw data statistics:")
    print(f"  Samples: {len(data_raw['E'])}")
    print(f"  Energy mean: {np.mean(data_raw['E']):.2f} eV")
    print(f"  Energy std: {np.std(data_raw['E']):.2f} eV")
    print(f"  Energy range: [{np.min(data_raw['E']):.2f}, {np.max(data_raw['E']):.2f}] eV")
    
    # Test with atomic energy subtraction
    config_subtract = DataConfig(
        batch_size=32,
        targets=['energy'],
        num_atoms=60,
        subtract_atomic_energies=True,
        atomic_energy_method='linear_regression'
    )
    data_subtract = load_npz(train_file, config=config_subtract, validate=False, verbose=False)
    
    print(f"\nWith atomic energy subtraction:")
    print(f"  Energy mean: {np.mean(data_subtract['E']):.2f} eV")
    print(f"  Energy std: {np.std(data_subtract['E']):.2f} eV")
    print(f"  Energy range: [{np.min(data_subtract['E']):.2f}, {np.max(data_subtract['E']):.2f}] eV")
    
    if 'metadata' in data_subtract and len(data_subtract['metadata']) > 0:
        metadata = data_subtract['metadata'][0]
        if 'atomic_energies' in metadata:
            print(f"  Atomic energies: {metadata['atomic_energies']}")
    
    # Test with per-atom scaling
    config_per_atom = DataConfig(
        batch_size=32,
        targets=['energy'],
        num_atoms=60,
        scale_by_atoms=True
    )
    data_per_atom = load_npz(train_file, config=config_per_atom, validate=False, verbose=False)
    
    print(f"\nWith per-atom scaling:")
    print(f"  Energy mean: {np.mean(data_per_atom['E']):.2f} eV/atom")
    print(f"  Energy std: {np.std(data_per_atom['E']):.2f} eV/atom")
    
    # The scaled energies should be roughly 1/3 of original (CO2 has 3 atoms)
    ratio = np.mean(data_raw['E']) / np.mean(data_per_atom['E'])
    print(f"  Ratio (original/per-atom): {ratio:.2f} (expected ~3 for CO2)")
    
    print("\n✅ Full pipeline test completed successfully")
    return True


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("Energy Preprocessing Test Suite")
    print("="*70)
    
    results = {}
    
    try:
        results['unit_conversion'] = test_unit_conversion()
    except Exception as e:
        print(f"❌ Test 1 failed: {e}")
        results['unit_conversion'] = False
    
    try:
        results['atomic_energies'] = test_atomic_energies()
    except Exception as e:
        print(f"❌ Test 2 failed: {e}")
        results['atomic_energies'] = False
    
    try:
        results['scale_by_atoms'] = test_scale_by_atoms()
    except Exception as e:
        print(f"❌ Test 3 failed: {e}")
        results['scale_by_atoms'] = False
    
    try:
        results['full_pipeline'] = test_full_pipeline()
    except Exception as e:
        print(f"❌ Test 4 failed: {e}")
        results['full_pipeline'] = False
    
    # Summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    all_passed = all(results.values())
    print("\n" + "="*70)
    if all_passed:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed")
    print("="*70)
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())

