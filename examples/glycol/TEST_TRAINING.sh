#!/bin/bash
# Quick test to verify training works
# Run this to confirm everything is fixed

echo "=========================================="
echo "Testing Glycol Training Setup"
echo "=========================================="
echo ""

echo "1. Checking cleaned data..."
python3 -c "
import numpy as np
data = np.load('glycol_cleaned.npz')
print(f'   ✅ File exists: glycol_cleaned.npz')
print(f'   ✅ Structures: {len(data[\"E\"])}')
print(f'   ✅ Fields: {list(data.keys())}')
"

echo ""
echo "2. Running quick training test (2 epochs)..."
python3 /home/ericb/mmml/mmml/cli/make_training.py \
  --data glycol_cleaned.npz \
  --tag test_quick \
  --n_train 50 \
  --n_valid 10 \
  --num_epochs 2 \
  --batch_size 4 \
  --ckpt_dir /tmp/test_glycol_$(date +%s) \
  --quiet 2>&1 | grep -E "(Auto-detecting|Detected|Checkpoint directory|Epoch|Error)"

echo ""
echo "=========================================="
echo "✅ If you see 'Epoch 1' and 'Epoch 2' above,"
echo "   then everything is working correctly!"
echo "=========================================="
