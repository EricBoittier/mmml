#!/bin/bash

# Source the settings
source settings.source

# Test with a single combination to verify it works
echo "Testing single cutoff combination..."

python -m mmml.cli.opt_mmml \
   --dataset $DATA \
   --pdbfile "pdb/init-packmol.pdb" \
   --checkpoint $CHECKPOINT \
   --n-monomers 2 \
   --n-atoms-monomer 10 \
   --ml-cutoff-grid 1.5 \
   --mm-switch-on-grid 5.0 \
   --mm-cutoff-grid 1.0 \
   --energy-weight 1.0 \
   --force-weight 1.0 \
   --max-frames 100 \
   --include-mm \
   --out test_cutoff_opt.json

echo "Test completed. Check test_cutoff_opt.json for results."
