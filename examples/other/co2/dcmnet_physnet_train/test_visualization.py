#!/usr/bin/env python3
"""
Test script to verify visualization tools work correctly.

Usage:
    python test_visualization.py --checkpoint /path/to/checkpoint
"""

import argparse
import sys
from pathlib import Path
import tempfile
import numpy as np
from ase import Atoms
from ase.io import write

def create_test_molecule():
    """Create a simple CO2 molecule for testing."""
    # CO2: O-C-O linear molecule
    positions = np.array([
        [0.0, 0.0, 0.0],      # C
        [-1.16, 0.0, 0.0],    # O
        [1.16, 0.0, 0.0],     # O
    ])
    
    symbols = ['C', 'O', 'O']
    atoms = Atoms(symbols, positions=positions)
    
    return atoms


def test_model_loading(checkpoint_dir):
    """Test if model can be loaded."""
    print("Testing model loading...")
    
    try:
        import pickle
        
        config_file = checkpoint_dir / 'model_config.pkl'
        params_file = checkpoint_dir / 'best_params.pkl'
        
        if not config_file.exists():
            print(f"  ✗ Config file not found: {config_file}")
            return False
        
        if not params_file.exists():
            print(f"  ✗ Params file not found: {params_file}")
            return False
        
        with open(config_file, 'rb') as f:
            config = pickle.load(f)
        
        print(f"  ✓ Config loaded")
        # Handle both 'F' and 'features' keys
        physnet_features = config['physnet_config'].get('F', config['physnet_config'].get('features', 'N/A'))
        print(f"    PhysNet features: {physnet_features}")
        print(f"    DCM sites: {config['dcmnet_config']['n_dcm']}")
        
        with open(params_file, 'rb') as f:
            params = pickle.load(f)
        
        print(f"  ✓ Parameters loaded")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error loading model: {e}")
        return False


def test_imports():
    """Test if all required packages are available."""
    print("Testing package imports...")
    
    required = {
        'numpy': 'numpy',
        'matplotlib': 'matplotlib',
        'ase': 'ase',
        'jax': 'jax',
        'jaxlib': 'jaxlib',
        'e3x': 'e3x',
        'flax': 'flax',
        'scipy': 'scipy',
    }
    
    all_ok = True
    for name, import_name in required.items():
        try:
            __import__(import_name)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - not found")
            all_ok = False
    
    return all_ok


def test_povray():
    """Test if POV-Ray is available."""
    print("Testing POV-Ray installation...")
    
    import subprocess
    
    try:
        result = subprocess.run(
            ['povray', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0] if result.stdout else "Unknown version"
            print(f"  ✓ POV-Ray found: {version_line}")
            return True
        else:
            print(f"  ✗ POV-Ray error")
            return False
            
    except FileNotFoundError:
        print(f"  ⚠ POV-Ray not found (optional for high-quality renders)")
        print(f"    Install: sudo apt install povray (Ubuntu)")
        print(f"    or: brew install povray (macOS)")
        return False
    except Exception as e:
        print(f"  ⚠ Error checking POV-Ray: {e}")
        return False


def test_matplotlib_viz(checkpoint_dir, test_mol_path):
    """Test matplotlib visualization."""
    print("Testing matplotlib visualization...")
    
    try:
        import matplotlib
        matplotlib.use('Agg')
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / 'test.png'
            
            # Import and run
            sys.path.insert(0, str(Path(__file__).parent))
            from matplotlib_3d_viz import load_model, evaluate_molecule, plot_molecule_3d
            from ase.io import read
            
            atoms = read(test_mol_path)
            model, params, config = load_model(checkpoint_dir)
            result = evaluate_molecule(atoms, model, params, config)
            
            fig, ax = plot_molecule_3d(atoms, result, show_charges=True)
            
            import matplotlib.pyplot as plt
            plt.savefig(output, dpi=100)
            plt.close()
            
            if output.exists():
                print(f"  ✓ Matplotlib visualization works")
                return True
            else:
                print(f"  ✗ Output file not created")
                return False
                
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_povray_viz(checkpoint_dir, test_mol_path):
    """Test POV-Ray visualization."""
    print("Testing POV-Ray visualization...")
    
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            sys.path.insert(0, str(Path(__file__).parent))
            from ase_povray_viz import (
                load_model, evaluate_molecule, 
                write_ase_povray_with_charges
            )
            from ase.io import read
            
            atoms = read(test_mol_path)
            model, params, config = load_model(checkpoint_dir)
            result = evaluate_molecule(atoms, model, params, config)
            
            pov_file = tmpdir / 'test.pov'
            write_ase_povray_with_charges(
                atoms, result, pov_file,
                show_charges=True,
                show_esp=False,
                canvas_width=800,
            )
            
            if pov_file.exists():
                file_size = pov_file.stat().st_size
                print(f"  ✓ POV file generated ({file_size} bytes)")
                return True
            else:
                print(f"  ✗ POV file not created")
                return False
                
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Test visualization tools')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--structure', type=str, default=None,
                       help='Test structure (optional, CO2 used if not provided)')
    
    args = parser.parse_args()
    
    checkpoint_dir = Path(args.checkpoint)
    
    print("\n" + "="*60)
    print("Visualization Tools Test Suite")
    print("="*60 + "\n")
    
    # Test 1: Imports
    print("Test 1: Package Imports")
    print("-" * 60)
    imports_ok = test_imports()
    print()
    
    # Test 2: Model loading
    print("Test 2: Model Loading")
    print("-" * 60)
    model_ok = test_model_loading(checkpoint_dir)
    print()
    
    # Test 3: POV-Ray
    print("Test 3: POV-Ray Installation")
    print("-" * 60)
    povray_ok = test_povray()
    print()
    
    # Prepare test molecule
    if args.structure:
        test_mol_path = args.structure
    else:
        print("Creating test CO2 molecule...")
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xyz', delete=False) as f:
            test_mol = create_test_molecule()
            write(f.name, test_mol)
            test_mol_path = f.name
        print(f"  ✓ Test molecule: {test_mol_path}\n")
    
    # Test 4: Matplotlib visualization
    print("Test 4: Matplotlib Visualization")
    print("-" * 60)
    if imports_ok and model_ok:
        mpl_ok = test_matplotlib_viz(checkpoint_dir, test_mol_path)
    else:
        print("  ⊘ Skipped (dependencies not met)")
        mpl_ok = False
    print()
    
    # Test 5: POV-Ray visualization
    print("Test 5: POV-Ray Scene Generation")
    print("-" * 60)
    if imports_ok and model_ok:
        pov_ok = test_povray_viz(checkpoint_dir, test_mol_path)
    else:
        print("  ⊘ Skipped (dependencies not met)")
        pov_ok = False
    print()
    
    # Summary
    print("="*60)
    print("Test Summary")
    print("="*60)
    
    tests = {
        'Package imports': imports_ok,
        'Model loading': model_ok,
        'POV-Ray installed': povray_ok,
        'Matplotlib viz': mpl_ok,
        'POV-Ray scene': pov_ok,
    }
    
    for name, result in tests.items():
        status = "✓ PASS" if result else ("⚠ WARN" if name == "POV-Ray installed" else "✗ FAIL")
        print(f"  {status:8s} {name}")
    
    print()
    
    all_pass = all([imports_ok, model_ok, mpl_ok, pov_ok])
    
    if all_pass:
        print("🎉 All tests passed! Ready to create visualizations.")
        print()
        print("Try:")
        print(f"  ./matplotlib_3d_viz.py --checkpoint {args.checkpoint} \\")
        print(f"      --structure {test_mol_path} --show-charges --output preview.png")
        return 0
    else:
        print("⚠ Some tests failed. Check errors above.")
        
        if not imports_ok:
            print("\nInstall missing packages:")
            print("  pip install numpy matplotlib ase jax jaxlib flax e3x scipy")
        
        if not model_ok:
            print(f"\nCheck checkpoint directory: {checkpoint_dir}")
        
        return 1


if __name__ == '__main__':
    sys.exit(main())

