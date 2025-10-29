#!/bin/bash

# Source the settings
source settings.source

# Base command
BASE_CMD="python -m mmml.cli.opt_mmml \
   --dataset filtered_acetone_3-8A.npz \
   --pdbfile \"pdb/init-packmol.pdb\" \
   --checkpoint \$CHECKPOINT \
   --n-monomers 2 \
   --n-atoms-monomer 10 \
   --energy-weight 1.0 \
   --force-weight 1.0 \
   --max-frames 100 \
   --include-mm"

# Grid values
ML_CUTOFFS=(1.0 1.5 2.0 2.5 3.0)
MM_SWITCH_ON=(4.0 5.0 6.0 7.0)
MM_CUTOFFS=(0.5 1.0 1.5 2.0)

# Counter for output files
counter=0

echo "Starting cutoff optimization with $((${#ML_CUTOFFS[@]} * ${#MM_SWITCH_ON[@]} * ${#MM_CUTOFFS[@]})) combinations..."

# Loop over all combinations
for ml_cutoff in "${ML_CUTOFFS[@]}"; do
    for mm_switch_on in "${MM_SWITCH_ON[@]}"; do
        for mm_cutoff in "${MM_CUTOFFS[@]}"; do
            counter=$((counter + 1))
            output_json="cutoff_opt_${counter}_ml${ml_cutoff}_mm${mm_switch_on}_cut${mm_cutoff}.json"
            output_npz="cutoff_opt_${counter}_ml${ml_cutoff}_mm${mm_switch_on}_cut${mm_cutoff}.npz"
            
            echo "Running combination $counter: ml_cutoff=$ml_cutoff, mm_switch_on=$mm_switch_on, mm_cutoff=$mm_cutoff"
            echo "Output: $output_json and $output_npz"
            
            # Run the optimization
            eval "$BASE_CMD \
                --ml-cutoff-grid $ml_cutoff \
                --mm-switch-on-grid $mm_switch_on \
                --mm-cutoff-grid $mm_cutoff \
                --out $output_json \
                --out-npz $output_npz"
            
            echo "Completed combination $counter"
            echo "---"
        done
    done
done

echo "All optimization runs completed!"
echo "Results saved as cutoff_opt_*.json and cutoff_opt_*.npz files"

# Optional: Create a summary script
cat > summarize_results.py << 'EOF'
#!/usr/bin/env python3
import json
import glob
import pandas as pd
import numpy as np

# Load all results
results = []
for file in glob.glob("cutoff_opt_*.json"):
    with open(file, 'r') as f:
        data = json.load(f)
        best = data['best']
        results.append({
            'file': file,
            'ml_cutoff': best['ml_cutoff'],
            'mm_switch_on': best['mm_switch_on'],
            'mm_cutoff': best['mm_cutoff'],
            'mse_energy': best['mse_energy'],
            'mse_forces': best['mse_forces'],
            'objective': best['objective']
        })

# Create DataFrame and sort by objective
df = pd.DataFrame(results)
df = df.sort_values('objective')

print("Optimization Results Summary:")
print("=" * 80)
print(df.to_string(index=False, float_format='%.6f'))

# Find best overall
best_overall = df.iloc[0]
print(f"\nBest overall result:")
print(f"File: {best_overall['file']}")
print(f"ml_cutoff: {best_overall['ml_cutoff']}")
print(f"mm_switch_on: {best_overall['mm_switch_on']}")
print(f"mm_cutoff: {best_overall['mm_cutoff']}")
print(f"Objective: {best_overall['objective']:.6f}")

# Check for identical results
print(f"\nChecking for identical results...")
unique_objectives = df['objective'].nunique()
total_results = len(df)
print(f"Unique objectives: {unique_objectives} out of {total_results}")

if unique_objectives < total_results:
    print("WARNING: Some results are identical!")
    # Group by objective to find duplicates
    duplicates = df.groupby('objective').size()
    duplicates = duplicates[duplicates > 1]
    if len(duplicates) > 0:
        print("Duplicate objectives found:")
        for obj_val, count in duplicates.items():
            print(f"  Objective {obj_val:.6f}: {count} occurrences")
            matching_rows = df[df['objective'] == obj_val]
            print(f"    Cutoff combinations: {matching_rows[['ml_cutoff', 'mm_switch_on', 'mm_cutoff']].to_string(index=False)}")
EOF

echo "Created summarize_results.py to analyze all results"
