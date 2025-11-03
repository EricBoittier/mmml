#!/bin/bash
# Example script to run molecular visualizations
# Adjust paths as needed

# Configuration
CHECKPOINT="./checkpoints/best_model"
STRUCTURE="./co2.xyz"
OUTPUT_DIR="./visualization_output"

echo "=========================================="
echo "Molecular Visualization Example"
echo "=========================================="
echo ""
echo "This script demonstrates the visualization tools."
echo "Please update the CHECKPOINT and STRUCTURE paths as needed."
echo ""

# Check if structure file exists
if [ ! -f "$STRUCTURE" ]; then
    echo "⚠️  Structure file not found: $STRUCTURE"
    echo "   Please update the STRUCTURE variable in this script."
    exit 1
fi

# Check if checkpoint exists
if [ ! -d "$CHECKPOINT" ]; then
    echo "⚠️  Checkpoint directory not found: $CHECKPOINT"
    echo "   Please update the CHECKPOINT variable in this script."
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Step 1: Quick Matplotlib Preview"
echo "=========================================="
echo ""

python matplotlib_3d_viz.py \
    --checkpoint "$CHECKPOINT" \
    --structure "$STRUCTURE" \
    --output "${OUTPUT_DIR}/preview_matplotlib.png" \
    --show-charges \
    --dpi 150

if [ $? -eq 0 ]; then
    echo "✓ Preview created: ${OUTPUT_DIR}/preview_matplotlib.png"
else
    echo "✗ Preview failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "Step 2: High-Quality POV-Ray Render"
echo "=========================================="
echo ""

# Check if POV-Ray is available
if command -v povray &> /dev/null; then
    echo "POV-Ray found, rendering..."
    
    python ase_povray_viz.py \
        --checkpoint "$CHECKPOINT" \
        --structure "$STRUCTURE" \
        --output-dir "$OUTPUT_DIR" \
        --show-charges \
        --resolution medium \
        --views "0x,0y,0z" "90x,0y,0z"
    
    if [ $? -eq 0 ]; then
        echo "✓ POV-Ray renders created in: $OUTPUT_DIR"
    else
        echo "✗ POV-Ray rendering failed"
    fi
else
    echo "⚠️  POV-Ray not found. Skipping high-quality render."
    echo "   Install with: sudo apt install povray (Ubuntu)"
    echo "   or: brew install povray (macOS)"
fi

echo ""
echo "=========================================="
echo "Step 3 (Optional): ESP Surface"
echo "=========================================="
echo ""

read -p "Render ESP surface? (takes longer) [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if command -v povray &> /dev/null; then
        python ase_povray_viz.py \
            --checkpoint "$CHECKPOINT" \
            --structure "$STRUCTURE" \
            --output-dir "${OUTPUT_DIR}/esp" \
            --show-charges \
            --show-esp \
            --resolution medium \
            --n-surface-points 5000
        
        if [ $? -eq 0 ]; then
            echo "✓ ESP visualization created in: ${OUTPUT_DIR}/esp"
        fi
    else
        echo "POV-Ray required for ESP surface rendering"
    fi
fi

echo ""
echo "=========================================="
echo "✅ Visualization Complete!"
echo "=========================================="
echo ""
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  • View the generated images"
echo "  • Adjust parameters in the scripts for publication quality"
echo "  • See VISUALIZATION_README.md for more options"
echo ""

