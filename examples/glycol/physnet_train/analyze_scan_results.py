#!/usr/bin/env python3
"""
Analyze results from hyperparameter scan.

Usage:
    python analyze_scan_results.py [--checkpoints-dir PATH]
"""

import argparse
import sys
from pathlib import Path
import re
import json
import pandas as pd

def extract_config_from_name(name: str):
    """Extract hyperparameters from experiment name."""
    config = {}
    
    # Parse format: co2_scan_f64_i4_b64_r3_bs16_d2
    patterns = {
        'features': r'f(\d+)',
        'iterations': r'i(\d+)',
        'basis': r'b(\d+)',
        'n_res': r'r(\d+)',
        'batch_size': r'bs(\d+)',
        'max_degree': r'd(\d+)'
    }
    
    for param, pattern in patterns.items():
        match = re.search(pattern, name)
        if match:
            config[param] = int(match.group(1))
    
    return config

def read_best_metrics(checkpoint_dir: Path):
    """Read best validation metrics from checkpoint directory."""
    metrics = {}
    
    # Look for best model file or metrics log
    best_file = checkpoint_dir / "best_model_metrics.json"
    if best_file.exists():
        with open(best_file, 'r') as f:
            metrics = json.load(f)
    else:
        # Try to parse from log files
        log_files = list(checkpoint_dir.glob("*.log"))
        if log_files:
            # Parse log file for best metrics (simplified)
            print(f"Warning: No metrics JSON found for {checkpoint_dir.name}, using logs")
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Analyze hyperparameter scan results")
    parser.add_argument('--checkpoints-dir', type=Path, 
                       default=Path.home() / "mmml" / "checkpoints",
                       help='Directory containing checkpoint folders')
    parser.add_argument('--pattern', type=str, default='co2_*',
                       help='Pattern to match experiment names')
    parser.add_argument('--output', type=str, default='scan_results.csv',
                       help='Output CSV file')
    
    args = parser.parse_args()
    
    if not args.checkpoints_dir.exists():
        print(f"Error: Checkpoints directory not found: {args.checkpoints_dir}")
        sys.exit(1)
    
    # Find all experiment directories
    exp_dirs = sorted(args.checkpoints_dir.glob(args.pattern))
    
    if not exp_dirs:
        print(f"No experiments found matching pattern '{args.pattern}'")
        sys.exit(1)
    
    print(f"Found {len(exp_dirs)} experiments")
    
    # Collect results
    results = []
    
    for exp_dir in exp_dirs:
        if not exp_dir.is_dir():
            continue
        
        name = exp_dir.name
        print(f"Processing: {name}")
        
        # Extract configuration
        config = extract_config_from_name(name)
        
        # Read metrics
        metrics = read_best_metrics(exp_dir)
        
        # Combine into result row
        row = {
            'experiment': name,
            **config,
            **metrics
        }
        
        results.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Sort by performance (lower is better for MAE)
    if 'valid_forces_mae' in df.columns:
        df = df.sort_values('valid_forces_mae')
    
    # Display results
    print("\n" + "="*80)
    print("HYPERPARAMETER SCAN RESULTS")
    print("="*80)
    
    if not df.empty:
        # Select key columns to display
        display_cols = ['experiment', 'features', 'max_degree', 'iterations', 'batch_size']
        metric_cols = [c for c in df.columns if 'mae' in c.lower() or 'loss' in c.lower()]
        display_cols.extend(metric_cols)
        
        # Filter to existing columns
        display_cols = [c for c in display_cols if c in df.columns]
        
        print(df[display_cols].to_string(index=False))
        
        # Save to CSV
        df.to_csv(args.output, index=False)
        print(f"\nFull results saved to: {args.output}")
        
        # Print top 5
        print("\n" + "="*80)
        print("TOP 5 MODELS (by validation forces MAE)")
        print("="*80)
        if 'valid_forces_mae' in df.columns:
            print(df[display_cols].head(5).to_string(index=False))
        
        # Model size statistics
        if 'features' in df.columns:
            print("\n" + "="*80)
            print("STATISTICS BY MODEL SIZE")
            print("="*80)
            size_stats = df.groupby('features')[metric_cols].mean()
            print(size_stats)
    else:
        print("No results to display")
    
    print("\nAnalysis complete!")

if __name__ == '__main__':
    main()

