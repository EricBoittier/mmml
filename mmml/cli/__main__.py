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
  xml2npz     Convert Molpro XML files to NPZ format
  train       Train DCMNet or PhysNetJAX models (coming soon)
  evaluate    Evaluate trained models (coming soon)
  validate    Validate NPZ files against schema
  fix-and-split  Fix units and create train/valid/test splits from NPZ data

Examples:
  mmml xml2npz input.xml -o output.npz
  mmml xml2npz inputs/*.xml -o dataset.npz --validate
  mmml validate dataset.npz
  mmml fix-and-split --efd data.npz --output-dir ./splits
  mmml fix-and-split --efd data.npz --grid grids.npz --output-dir ./splits
  
For help on a specific command:
  mmml <command> --help
        """
    )
    
    parser.add_argument(
        'command',
        choices=['xml2npz', 'validate', 'train', 'evaluate', 'downstream', 'fix-and-split'],
        help='Command to run'
    )
    parser.add_argument(
        'args',
        nargs=argparse.REMAINDER,
        help='Arguments for the command'
    )
    
    args = parser.parse_args()
    
    # Dispatch to appropriate command
    if args.command == 'xml2npz':
        from . import xml2npz
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
        from . import fix_and_split
        sys.argv = ['mmml fix-and-split'] + args.args
        return fix_and_split.main()
    
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())

