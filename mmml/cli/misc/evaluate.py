#!/usr/bin/env python
"""
Model evaluation command for MMML.

Usage:
    mmml evaluate --model checkpoint.pkl --data test.npz
    mmml evaluate --model checkpoint.pkl --data test.npz --output results/
"""

import sys
import argparse
from pathlib import Path
import json
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mmml.data import load_npz, get_data_statistics


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    mae: float  # Mean Absolute Error
    rmse: float  # Root Mean Squared Error
    max_error: float  # Maximum error
    r2: float  # R² score
    
    def to_dict(self) -> Dict:
        return {
            'mae': float(self.mae),
            'rmse': float(self.rmse),
            'max_error': float(self.max_error),
            'r2': float(self.r2)
        }


def compute_metrics(predictions: np.ndarray, targets: np.ndarray) -> EvaluationMetrics:
    """
    Compute evaluation metrics.
    
    Parameters
    ----------
    predictions : np.ndarray
        Model predictions
    targets : np.ndarray
        Ground truth values
        
    Returns
    -------
    EvaluationMetrics
        Computed metrics
    """
    errors = predictions - targets
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    max_error = np.max(np.abs(errors))
    
    # R² score
    ss_res = np.sum(errors**2)
    ss_tot = np.sum((targets - np.mean(targets))**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    return EvaluationMetrics(
        mae=mae,
        rmse=rmse,
        max_error=max_error,
        r2=r2
    )


def evaluate_model(
    model_path: str,
    data: Dict[str, np.ndarray],
    properties: List[str],
    batch_size: int = 32,
    verbose: bool = True
) -> Dict[str, EvaluationMetrics]:
    """
    Evaluate model on test data.
    
    Parameters
    ----------
    model_path : str
        Path to model checkpoint
    data : dict
        Test data dictionary
    properties : list
        Properties to evaluate
    batch_size : int
        Batch size for prediction
    verbose : bool
        Whether to print progress
        
    Returns
    -------
    dict
        Dictionary mapping property names to metrics
    """
    if verbose:
        print(f"\n📊 Evaluating model: {model_path}")
        print(f"   Test samples: {len(data['E'])}")
        print(f"   Properties: {', '.join(properties)}")
    
    # TODO: Load actual model and make predictions
    # For now, this is a placeholder showing the structure
    
    results = {}
    
    # Generate dummy predictions for demonstration
    for prop in properties:
        if prop == 'energy' and 'E' in data:
            # Dummy: add small noise to targets
            targets = data['E'].flatten()
            predictions = targets + np.random.randn(len(targets)) * 0.1
            
            metrics = compute_metrics(predictions, targets)
            results[prop] = metrics
            
            if verbose:
                print(f"\n   {prop.capitalize()}:")
                print(f"      MAE:  {metrics.mae:.6f}")
                print(f"      RMSE: {metrics.rmse:.6f}")
                print(f"      R²:   {metrics.r2:.6f}")
        
        elif prop == 'forces' and 'F' in data:
            # Dummy forces metrics
            targets = data['F'].reshape(-1)
            predictions = targets + np.random.randn(len(targets)) * 0.01
            
            metrics = compute_metrics(predictions, targets)
            results[prop] = metrics
            
            if verbose:
                print(f"\n   {prop.capitalize()}:")
                print(f"      MAE:  {metrics.mae:.6f}")
                print(f"      RMSE: {metrics.rmse:.6f}")
    
    if verbose:
        print("\n   ⚠️  Using dummy predictions (actual model loading not implemented)")
    
    return results


def save_results(
    results: Dict[str, EvaluationMetrics],
    output_dir: Path,
    model_name: str = 'model'
):
    """Save evaluation results to JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_dict = {
        'model': model_name,
        'metrics': {
            prop: metrics.to_dict()
            for prop, metrics in results.items()
        }
    }
    
    output_file = output_dir / 'evaluation_results.json'
    with open(output_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    return output_file


def create_report(
    results: Dict[str, EvaluationMetrics],
    output_dir: Path,
    data_stats: Dict
):
    """Create markdown evaluation report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_file = output_dir / 'evaluation_report.md'
    
    with open(report_file, 'w') as f:
        f.write("# Model Evaluation Report\n\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Dataset Statistics\n\n")
        f.write(f"- Structures: {data_stats.get('n_structures', 'N/A')}\n")
        f.write(f"- Atoms: {data_stats.get('n_atoms', 'N/A')}\n")
        if 'unique_elements' in data_stats:
            f.write(f"- Elements: {data_stats['unique_elements']}\n")
        f.write("\n")
        
        f.write("## Evaluation Metrics\n\n")
        f.write("| Property | MAE | RMSE | Max Error | R² |\n")
        f.write("|----------|-----|------|-----------|----|\n")
        
        for prop, metrics in results.items():
            f.write(f"| {prop.capitalize()} | "
                   f"{metrics.mae:.6f} | "
                   f"{metrics.rmse:.6f} | "
                   f"{metrics.max_error:.6f} | "
                   f"{metrics.r2:.6f} |\n")
        
        f.write("\n")
        f.write("## Notes\n\n")
        f.write("- MAE: Mean Absolute Error\n")
        f.write("- RMSE: Root Mean Squared Error\n")
        f.write("- R²: Coefficient of determination\n")
    
    return report_file


def main():
    """Main evaluation CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Evaluate MMML models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation
  %(prog)s --model checkpoint.pkl --data test.npz
  
  # Evaluate specific properties
  %(prog)s --model checkpoint.pkl --data test.npz \\
           --properties energy forces
  
  # Save results and report
  %(prog)s --model checkpoint.pkl --data test.npz \\
           --output results/ --report
  
  # Compare multiple models
  %(prog)s --model model1.pkl model2.pkl --data test.npz \\
           --output comparison/
        """
    )
    
    # Model and data
    parser.add_argument(
        '--model',
        nargs='+',
        required=True,
        help='Path(s) to model checkpoint(s)'
    )
    parser.add_argument(
        '--data',
        required=True,
        help='Test data NPZ file'
    )
    
    # Evaluation options
    parser.add_argument(
        '--properties',
        nargs='+',
        default=['energy'],
        help='Properties to evaluate (default: energy)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for prediction (default: 32)'
    )
    
    # Output options
    parser.add_argument(
        '--output',
        type=str,
        default='results',
        help='Output directory (default: results)'
    )
    parser.add_argument(
        '--report',
        action='store_true',
        help='Generate markdown report'
    )
    parser.add_argument(
        '--save-predictions',
        action='store_true',
        help='Save predictions to file'
    )
    
    # Display options
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Verbose output'
    )
    parser.add_argument(
        '--quiet',
        '-q',
        action='store_true',
        help='Quiet mode'
    )
    
    args = parser.parse_args()
    verbose = args.verbose and not args.quiet
    
    # Load test data
    if verbose:
        print(f"\n📂 Loading test data: {args.data}")
    
    try:
        data = load_npz(args.data, validate=True, verbose=verbose)
        data_stats = get_data_statistics(data)
    except Exception as e:
        print(f"❌ Error loading test data: {e}", file=sys.stderr)
        return 1
    
    if verbose:
        print(f"✓ Loaded {len(data['E'])} test samples")
    
    # Evaluate each model
    output_dir = Path(args.output)
    all_results = {}
    
    for i, model_path in enumerate(args.model):
        model_name = Path(model_path).stem
        
        if len(args.model) > 1 and not args.quiet:
            print(f"\n{'='*60}")
            print(f"Model {i+1}/{len(args.model)}: {model_name}")
            print('='*60)
        
        try:
            results = evaluate_model(
                model_path,
                data,
                args.properties,
                batch_size=args.batch_size,
                verbose=verbose
            )
            
            all_results[model_name] = results
            
            # Save results
            if not args.quiet:
                print(f"\n💾 Saving results...")
            
            model_output_dir = output_dir / model_name if len(args.model) > 1 else output_dir
            
            results_file = save_results(results, model_output_dir, model_name)
            if verbose:
                print(f"   Results: {results_file}")
            
            # Generate report
            if args.report:
                report_file = create_report(results, model_output_dir, data_stats)
                if verbose:
                    print(f"   Report: {report_file}")
            
        except Exception as e:
            print(f"❌ Error evaluating {model_name}: {e}", file=sys.stderr)
            if verbose:
                import traceback
                traceback.print_exc()
            continue
    
    # Summary
    if not args.quiet:
        print(f"\n✅ Evaluation complete!")
        print(f"   Models evaluated: {len(all_results)}")
        print(f"   Results saved to: {output_dir}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

