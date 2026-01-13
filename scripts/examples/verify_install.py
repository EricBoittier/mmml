#!/usr/bin/env python3
"""
MMML Installation Verification Script

This script checks if MMML and its dependencies are properly installed.
Run this after installation to ensure everything is working correctly.

Usage:
    python verify_install.py
    # or
    uv run python verify_install.py
"""

import sys
import os
from typing import List, Tuple

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def check_package(package_name: str, import_name: str = None) -> Tuple[bool, str]:
    """Check if a package can be imported."""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        return True, "OK"
    except ImportError as e:
        return False, str(e)

def check_gpu() -> Tuple[bool, str]:
    """Check if GPU/CUDA is available."""
    try:
        import jax
        devices = jax.devices()
        gpu_devices = [d for d in devices if d.platform == 'gpu']
        if gpu_devices:
            return True, f"Found {len(gpu_devices)} GPU(s)"
        else:
            return False, "No GPU detected (CPU only)"
    except Exception as e:
        return False, f"Error checking GPU: {e}"

def check_charmm() -> Tuple[bool, str]:
    """Check if CHARMM is properly configured."""
    charmm_home = os.environ.get('CHARMM_HOME')
    charmm_lib = os.environ.get('CHARMM_LIB_DIR')
    
    if not charmm_home or not charmm_lib:
        return False, "CHARMM_HOME or CHARMM_LIB_DIR not set"
    
    if not os.path.exists(charmm_home):
        return False, f"CHARMM_HOME path does not exist: {charmm_home}"
    
    try:
        import pycharmm
        return True, f"CHARMM configured at {charmm_home}"
    except Exception as e:
        return False, f"Error importing pycharmm: {e}"

def print_header(text: str):
    """Print a section header."""
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}{text:^60}{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")

def print_check(name: str, success: bool, message: str = ""):
    """Print a check result."""
    status = f"{GREEN}✓{RESET}" if success else f"{RED}✗{RESET}"
    print(f"{status} {name:40} {message}")

def main():
    """Run all verification checks."""
    print(f"\n{BLUE}MMML Installation Verification{RESET}")
    print("="*60)
    
    # Core dependencies
    print_header("Core Dependencies")
    
    core_packages = [
        ("mmml", "mmml"),
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
        ("ase", "ase"),
    ]
    
    core_ok = True
    for pkg_name, import_name in core_packages:
        success, msg = check_package(pkg_name, import_name)
        print_check(pkg_name, success, msg if not success else "")
        if not success:
            core_ok = False
    
    # JAX ecosystem
    print_header("JAX Ecosystem")
    
    jax_packages = [
        ("jax", "jax"),
        ("jaxlib", "jaxlib"),
        ("flax", "flax"),
        ("optax", "optax"),
        ("dm-haiku", "haiku"),
        ("chex", "chex"),
    ]
    
    jax_ok = True
    for pkg_name, import_name in jax_packages:
        success, msg = check_package(pkg_name, import_name)
        print_check(pkg_name, success, msg if not success else "")
        if not success:
            jax_ok = False
    
    # Check GPU
    print_header("GPU Support")
    gpu_ok, gpu_msg = check_gpu()
    print_check("GPU/CUDA", gpu_ok, gpu_msg)
    
    # Optional dependencies
    print_header("Optional Dependencies")
    
    optional_packages = [
        ("torch (PyTorch)", "torch", "ml"),
        ("pyscf", "pyscf", "quantum"),
        ("plotly", "plotly", "viz"),
        ("mdanalysis", "MDAnalysis", "md"),
        ("wandb", "wandb", "experiments"),
        ("rdkit", "rdkit", "chem"),
    ]
    
    print(f"{YELLOW}Note: Optional packages are not required for core functionality{RESET}\n")
    
    for pkg_name, import_name, group in optional_packages:
        success, msg = check_package(pkg_name, import_name)
        status = f"{GREEN}✓{RESET}" if success else f"{YELLOW}○{RESET}"
        group_str = f"[{group}]"
        print(f"{status} {pkg_name:30} {group_str:15} {'OK' if success else 'Not installed'}")
    
    # Check CHARMM
    print_header("CHARMM Configuration")
    charmm_ok, charmm_msg = check_charmm()
    print_check("CHARMM", charmm_ok, charmm_msg)
    
    if not charmm_ok:
        print(f"\n{YELLOW}To set up CHARMM:{RESET}")
        print("  source CHARMMSETUP")
        print("  # or")
        print("  export CHARMM_HOME=/path/to/mmml/setup/charmm")
        print("  export CHARMM_LIB_DIR=/path/to/mmml/setup/charmm")
    
    # Summary
    print_header("Summary")
    
    all_ok = core_ok and jax_ok and charmm_ok
    
    if all_ok:
        print(f"{GREEN}✓ All core components are properly installed!{RESET}")
        print(f"{GREEN}  MMML is ready to use.{RESET}\n")
        return 0
    else:
        print(f"{RED}✗ Some components are missing or misconfigured:{RESET}")
        if not core_ok:
            print(f"  {RED}• Core dependencies are missing{RESET}")
        if not jax_ok:
            print(f"  {RED}• JAX ecosystem is incomplete{RESET}")
        if not charmm_ok:
            print(f"  {YELLOW}• CHARMM is not configured (optional){RESET}")
        
        print(f"\n{BLUE}Installation instructions:{RESET}")
        print("  See INSTALL.md for detailed installation guide")
        print("  Quick start: make install-gpu")
        print()
        return 1

if __name__ == "__main__":
    sys.exit(main())

