#!/usr/bin/env python3
"""
Test CLI tool to verify optional dependencies and core functionality.

Usage:
    python -m mmml.cli.test_deps
"""

import sys


def test_core_imports():
    """Test that core MMML functionality works."""
    print("=" * 70)
    print("TESTING CORE IMPORTS")
    print("=" * 70)
    
    tests = []
    
    # Test model creation
    try:
        from mmml.physnetjax.physnetjax.models.model import EF
        model = EF(features=64, natoms=3)
        tests.append(("Model creation", True, None))
        print("✅ Model creation works")
    except Exception as e:
        tests.append(("Model creation", False, str(e)))
        print(f"❌ Model creation failed: {e}")
    
    # Test restart utilities
    try:
        from mmml.physnetjax.physnetjax.restart.restart import get_params_model
        tests.append(("Restart utilities", True, None))
        print("✅ Restart utilities work")
    except Exception as e:
        tests.append(("Restart utilities", False, str(e)))
        print(f"❌ Restart utilities failed: {e}")
    
    # Test data loading
    try:
        from mmml.data import DataConfig, load_npz
        config = DataConfig()
        tests.append(("Data loading", True, None))
        print("✅ Data loading utilities work")
    except Exception as e:
        tests.append(("Data loading", False, str(e)))
        print(f"❌ Data loading failed: {e}")
    
    # Test training utilities
    try:
        from mmml.physnetjax.physnetjax.utils.pretty_printer import init_table, print_dict_as_table
        table = init_table(doCharges=False)
        tests.append(("Training utilities", True, None))
        print("✅ Training utilities work")
    except Exception as e:
        tests.append(("Training utilities", False, str(e)))
        print(f"❌ Training utilities failed: {e}")
    
    return tests


def check_optional_dependencies():
    """Check which optional dependencies are available."""
    print("\n" + "=" * 70)
    print("CHECKING OPTIONAL DEPENDENCIES")
    print("=" * 70)
    
    available = {}
    
    # Check plotting dependencies
    try:
        from mmml.physnetjax.physnetjax.utils.pretty_printer import HAS_ASCIICHARTPY, HAS_POLARS
        available['asciichartpy'] = HAS_ASCIICHARTPY
        available['polars'] = HAS_POLARS
        print(f"{'✅' if HAS_ASCIICHARTPY else '❌'} asciichartpy: {HAS_ASCIICHARTPY}")
        print(f"{'✅' if HAS_POLARS else '❌'} polars: {HAS_POLARS}")
    except Exception as e:
        print(f"❌ Could not check plotting dependencies: {e}")
        available['asciichartpy'] = False
        available['polars'] = False
    
    return available


def print_summary(core_tests, optional_deps):
    """Print summary of test results."""
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    # Core functionality
    core_passed = sum(1 for _, passed, _ in core_tests if passed)
    core_total = len(core_tests)
    print(f"\nCore Functionality: {core_passed}/{core_total} tests passed")
    
    if core_passed == core_total:
        print("✅ All core functionality working!")
    else:
        print("⚠️  Some core tests failed:")
        for test_name, passed, error in core_tests:
            if not passed:
                print(f"   - {test_name}: {error}")
    
    # Optional dependencies
    print(f"\nOptional Dependencies:")
    plotting_available = optional_deps.get('asciichartpy', False) and optional_deps.get('polars', False)
    
    print(f"  {'✅' if plotting_available else '❌'} Plotting support (asciichartpy + polars)")
    
    # Installation recommendations
    if not plotting_available:
        print("\n💡 Recommendations:")
        print("   For plotting: pip install -e '.[plotting]'")
    else:
        print("\n✨ All optional features available!")
    
    # Return exit code
    return 0 if core_passed == core_total else 1


def main():
    """Run all tests."""
    print("MMML Optional Dependencies Test")
    print("=" * 70)
    
    # Test core functionality
    core_tests = test_core_imports()
    
    # Check optional dependencies
    optional_deps = check_optional_dependencies()
    
    # Print summary
    exit_code = print_summary(core_tests, optional_deps)
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

