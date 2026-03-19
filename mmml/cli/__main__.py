#!/usr/bin/env python
"""
Main entry point for MMML CLI commands.

Provides a unified interface for all MMML command-line tools.
"""

import sys
import argparse
from pathlib import Path


def main():
    """Main CLI dispatcher."""
    parser = argparse.ArgumentParser(
        prog='mmml',
        description='MMML: Machine Learning for Molecular Modeling',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available commands:
  make-res    Generate residue (PDB, PSF, topology) via PyCHARMM/CGENFF
  make-box    Pack molecules into periodic box (vacuum or solvated)
  run         MM/ML simulation (ASE + JAX-MD with hybrid calculator)
  run-pycharmm  Pure CHARMM heating and equilibration (no ML)
  xml2npz     Convert Molpro XML files to NPZ format
  train       Train DCMNet or PhysNetJAX models (coming soon)
  evaluate    Evaluate trained models (coming soon)
  validate    Validate NPZ files against schema
  fix-and-split  Fix units and create train/valid/test splits from NPZ data
  pyscf-dft   GPU-accelerated DFT calculations (energy, gradient, hessian, etc.)
  pyscf-mp2   GPU-accelerated MP2 (post-HF) calculations
  pyscf-evaluate  Evaluate geometries (E, F, D, ESP) in batch
  verify-esp-alignment  Verify esp-grid alignment in evaluated NPZ (data generation check)
  normal-mode-sample  Sample geometries along vibrational modes
  physnet-md  PhysNet MD sampling (ASE + JAX-MD)
  gui         Start the molecular viewer GUI

Examples:
  mmml make-res --res CYBZ
  mmml make-box --res CYBZ --n 50 --side_length 25.0
  mmml xml2npz input.xml -o output.npz
  mmml xml2npz inputs/*.xml -o dataset.npz --validate
  mmml validate dataset.npz
  mmml fix-and-split --efd data.npz --output-dir ./splits
  mmml fix-and-split --efd data.npz --grid grids.npz --output-dir ./splits
  mmml gui --data-dir ./trajectories
  mmml gui --file simulation.npz
  mmml pyscf-dft --mol "O 0 0 0; H 0.96 0 0; H -0.24 0.93 0" --energy
  mmml normal-mode-sample -i out/04_results.h5 -o out/06_sampled.npz --amplitude 0.1
  mmml pyscf-evaluate -i out/06_sampled.npz -o out/07_evaluated.npz
  mmml physnet-md --checkpoint out/ckpts/cybz_physnet --data out/splits/energies_forces_dipoles_train.npz -o out/

For help on a specific command:
  mmml <command> --help
        """
    )
    
    parser.add_argument(
        'command',
        choices=['make-res', 'make-box', 'run', 'run-pycharmm', 'xml2npz', 'validate', 'train', 'evaluate', 'downstream', 'fix-and-split', 'pyscf-dft', 'pyscf-mp2', 'pyscf-evaluate', 'verify-esp-alignment', 'normal-mode-sample', 'physnet-md', 'gui'],
        help='Command to run'
    )
    parser.add_argument(
        'args',
        nargs=argparse.REMAINDER,
        help='Arguments for the command'
    )
    
    args = parser.parse_args()
    
    # Dispatch to appropriate command
    if args.command == 'make-res':
        from .misc import make_res_cli
        sys.argv = ['mmml make-res'] + args.args
        return make_res_cli.main()

    elif args.command == 'make-box':
        from .misc import make_box_cli
        sys.argv = ['mmml make-box'] + args.args
        return make_box_cli.main()

    elif args.command == 'run':
        from .run.run_sim import main as run_sim_main
        sys.argv = ['mmml run'] + args.args
        return run_sim_main()

    elif args.command == 'run-pycharmm':
        from .run.run_pycharmm import main
        sys.argv = ['mmml run-pycharmm'] + args.args
        return main()

    elif args.command == 'xml2npz':
        from .misc import xml2npz
        sys.argv = ['mmml xml2npz'] + args.args
        return xml2npz.main()
    
    elif args.command == 'validate':
        from ..data.npz_schema import validate_npz
        if not args.args:
            print("Error: Please provide an NPZ file to validate", file=sys.stderr)
            print("Usage: mmml validate <npz_file>", file=sys.stderr)
            return 1
        
        # Validate each file provided
        all_valid = True
        for npz_file in args.args:
            print(f"\n{'='*60}")
            print(f"Validating: {npz_file}")
            print('='*60)
            is_valid, info = validate_npz(npz_file, verbose=True)
            if not is_valid:
                all_valid = False
        
        return 0 if all_valid else 1
    
    elif args.command == 'train':
        from . import train
        sys.argv = ['mmml train'] + args.args
        return train.main()
    
    elif args.command == 'evaluate':
        from . import evaluate
        sys.argv = ['mmml evaluate'] + args.args
        return evaluate.main()
    elif args.command == 'downstream':
        from . import downstream
        sys.argv = ['mmml downstream'] + args.args
        return downstream.main()
    
    elif args.command == 'fix-and-split':
        from .misc import fix_and_split
        sys.argv = ['mmml fix-and-split'] + args.args
        return fix_and_split.main()
    
    elif args.command == 'pyscf-dft':
        from .misc import pyscf_dft
        sys.argv = ['mmml pyscf-dft'] + args.args
        return pyscf_dft.main()

    elif args.command == 'pyscf-mp2':
        from .misc import pyscf_mp2
        sys.argv = ['mmml pyscf-mp2'] + args.args
        return pyscf_mp2.main()

    elif args.command == 'pyscf-evaluate':
        from .misc import pyscf_evaluate
        sys.argv = ['mmml pyscf-evaluate'] + args.args
        return pyscf_evaluate.main()

    elif args.command == 'verify-esp-alignment':
        from .misc import verify_esp_alignment
        sys.argv = ['mmml verify-esp-alignment'] + args.args
        return verify_esp_alignment.main()

    elif args.command == 'normal-mode-sample':
        from .misc import normal_mode_sample
        sys.argv = ['mmml normal-mode-sample'] + args.args
        return normal_mode_sample.main()

    elif args.command == 'physnet-md':
        from .misc import physnet_md
        sys.argv = ['mmml physnet-md'] + args.args
        return physnet_md.main()

    elif args.command == 'gui':
        from . import gui
        sys.argv = ['mmml gui'] + args.args
        return gui.main()
    
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())

