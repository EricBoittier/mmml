#!/usr/bin/env python
"""
Command-line interface for converting Molpro XML files to NPZ format.

Usage:
    mmml xml2npz input.xml -o output.npz
    mmml xml2npz inputs/*.xml -o dataset.npz
    mmml xml2npz inputs/ -o dataset.npz --recursive
"""

import sys
import argparse
from pathlib import Path
from typing import List
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mmml.data import (
    batch_convert_xml,
    validate_npz,
    MolproConverter
)


def find_xml_files(paths: List[Path], recursive: bool = False) -> List[Path]:
    """
    Find all XML files from given paths.
    
    Parameters
    ----------
    paths : list
        List of files or directories
    recursive : bool
        Whether to search directories recursively
        
    Returns
    -------
    list
        List of XML file paths
    """
    xml_files = []
    
    for path in paths:
        path = Path(path)
        
        if path.is_file():
            if path.suffix.lower() == '.xml':
                xml_files.append(path)
        elif path.is_dir():
            if recursive:
                xml_files.extend(path.rglob('*.xml'))
            else:
                xml_files.extend(path.glob('*.xml'))
    
    return sorted(xml_files)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Convert Molpro XML files to standardized NPZ format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert single file
  %(prog)s output.xml -o data.npz
  
  # Convert multiple files
  %(prog)s file1.xml file2.xml file3.xml -o dataset.npz
  
  # Convert all XML files in directory
  %(prog)s molpro_outputs/ -o dataset.npz
  
  # Recursive search
  %(prog)s data/ -o dataset.npz --recursive
  
  # With validation and summary
  %(prog)s inputs/*.xml -o data.npz --validate --summary summary.json
  
  # Adjust padding for larger molecules
  %(prog)s inputs/*.xml -o data.npz --padding 100
        """
    )
    
    # Input/output arguments
    parser.add_argument(
        'inputs',
        nargs='+',
        type=str,
        help='Input XML file(s) or directory/directories'
    )
    parser.add_argument(
        '-o', '--output',
        required=True,
        type=str,
        help='Output NPZ file path'
    )
    
    # Conversion options
    parser.add_argument(
        '--padding',
        type=int,
        default=60,
        help='Number of atoms to pad to (default: 60)'
    )
    parser.add_argument(
        '--no-variables',
        action='store_true',
        help='Exclude Molpro internal variables from output'
    )
    parser.add_argument(
        '--first-geometry',
        action='store_true',
        help='Use first geometry from files with multiple geometries (default: use last/final)'
    )
    parser.add_argument(
        '--recursive',
        '-r',
        action='store_true',
        help='Recursively search directories for XML files'
    )
    
    # Validation options
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate output NPZ file against schema'
    )
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='Skip validation (faster but not recommended)'
    )
    parser.add_argument(
        '--strict',
        action='store_true',
        help='Use strict validation (fail on warnings)'
    )
    
    # Output options
    parser.add_argument(
        '--summary',
        type=str,
        help='Save conversion summary to JSON file'
    )
    parser.add_argument(
        '--quiet',
        '-q',
        action='store_true',
        help='Suppress progress output'
    )
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Verbose output'
    )
    
    # Advanced options
    parser.add_argument(
        '--continue-on-error',
        action='store_true',
        help='Continue processing even if some files fail'
    )
    parser.add_argument(
        '--max-files',
        type=int,
        help='Maximum number of files to process (for testing)'
    )
    
    args = parser.parse_args()
    
    # Find all XML files
    if not args.quiet:
        print("🔍 Finding XML files...")
    
    input_paths = [Path(p) for p in args.inputs]
    xml_files = find_xml_files(input_paths, recursive=args.recursive)
    
    if not xml_files:
        print("❌ Error: No XML files found!", file=sys.stderr)
        return 1
    
    if not args.quiet:
        print(f"📁 Found {len(xml_files)} XML file(s)")
    
    # Apply max files limit if specified
    if args.max_files:
        xml_files = xml_files[:args.max_files]
        if not args.quiet:
            print(f"⚠️  Limited to first {args.max_files} files")
    
    # Perform conversion
    if not args.quiet:
        print(f"\n🔄 Converting to NPZ format...")
        print(f"   Output: {args.output}")
        print(f"   Padding: {args.padding} atoms")
        print(f"   Variables: {'No' if args.no_variables else 'Yes'}")
        print(f"   Geometry: {'First' if args.first_geometry else 'Last (final)'}")
    
    try:
        success = batch_convert_xml(
            xml_files=[str(f) for f in xml_files],
            output_file=args.output,
            padding_atoms=args.padding,
            include_variables=not args.no_variables,
            use_last_geometry=not args.first_geometry,
            verbose=args.verbose and not args.quiet
        )
        
        if not success:
            print("❌ Conversion failed!", file=sys.stderr)
            return 1
            
    except Exception as e:
        print(f"❌ Error during conversion: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    # Validate if requested
    if args.validate or not args.no_validate:
        if not args.quiet:
            print(f"\n✓ Validating output...")
        
        try:
            is_valid, info = validate_npz(
                args.output,
                strict=args.strict,
                verbose=args.verbose and not args.quiet
            )
            
            if not is_valid:
                print("⚠️  Validation warnings (see above)", file=sys.stderr)
                if args.strict:
                    return 1
            elif not args.quiet:
                print("✓ Validation passed")
                
                # Print summary
                if info:
                    print(f"\n📊 Dataset Summary:")
                    print(f"   Structures: {info['n_structures']}")
                    print(f"   Atoms: {info['n_atoms']}")
                    print(f"   Properties: {', '.join(info['properties'][:10])}")
                    if len(info['properties']) > 10:
                        print(f"                + {len(info['properties']) - 10} more")
                    if 'unique_elements' in info:
                        print(f"   Elements: {info['unique_elements']}")
                    if 'energy_range' in info:
                        emin = info['energy_range']['min']
                        emax = info['energy_range']['max']
                        print(f"   Energy range: [{emin:.6f}, {emax:.6f}] Ha")
                
        except Exception as e:
            print(f"⚠️  Validation error: {e}", file=sys.stderr)
            if args.strict:
                return 1
    
    # Save summary if requested
    if args.summary:
        if not args.quiet:
            print(f"\n💾 Saving summary to {args.summary}...")
        
        try:
            _, info = validate_npz(args.output, verbose=False)
            
            summary = {
                'input_files': len(xml_files),
                'output_file': str(args.output),
                'padding_atoms': args.padding,
                'include_variables': not args.no_variables,
                'dataset_info': info
            }
            
            summary_path = Path(args.summary)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            if not args.quiet:
                print(f"✓ Summary saved")
                
        except Exception as e:
            print(f"⚠️  Could not save summary: {e}", file=sys.stderr)
    
    if not args.quiet:
        print(f"\n✅ Conversion complete!")
        print(f"   Output: {args.output}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

