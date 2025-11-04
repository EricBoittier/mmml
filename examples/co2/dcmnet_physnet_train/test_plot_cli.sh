#!/bin/bash
# Test script for the plotting CLI
# Demonstrates various usage patterns

set -e

echo "=========================================="
echo "Testing Comparison Results Plotting CLI"
echo "=========================================="
echo ""

# Check if we have any comparison results
RESULTS=$(find comparisons -name "comparison_results.json" -type f 2>/dev/null | head -1)

if [ -z "$RESULTS" ]; then
    echo "⚠️  No comparison results found in comparisons/"
    echo ""
    echo "To generate comparison results, run:"
    echo "  sbatch sbatch/04_compare_models.sbatch"
    echo ""
    echo "Or use compare_models.py directly:"
    echo "  python compare_models.py \\"
    echo "    --train-efd train.npz --train-esp grids_train.npz \\"
    echo "    --valid-efd valid.npz --valid-esp grids_valid.npz \\"
    echo "    --epochs 20 --comparison-name test"
    echo ""
    exit 1
fi

echo "Found comparison results: $RESULTS"
echo ""

# Test 1: Summary only
echo "Test 1: Print summary (no plots)"
echo "----------------------------------------"
python plot_comparison_results.py "$RESULTS" --summary-only
echo ""

# Test 2: Create all plots
echo "Test 2: Create all plots (PNG, 150 DPI)"
echo "----------------------------------------"
OUTPUT_DIR="$(dirname "$RESULTS")/test_plots"
mkdir -p "$OUTPUT_DIR"

python plot_comparison_results.py "$RESULTS" \
    --output-dir "$OUTPUT_DIR" \
    --dpi 150 \
    --format png

echo ""
echo "Plots created in: $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR"/*.png
echo ""

# Test 3: Performance only, high-res
echo "Test 3: Performance plot only (PDF, 300 DPI)"
echo "----------------------------------------"
python plot_comparison_results.py "$RESULTS" \
    --plot-type performance \
    --output-dir "$OUTPUT_DIR" \
    --format pdf \
    --dpi 300

echo ""

# Test 4: Custom colors
echo "Test 4: Custom colors"
echo "----------------------------------------"
python plot_comparison_results.py "$RESULTS" \
    --plot-type overview \
    --output-dir "$OUTPUT_DIR" \
    --colors "#FF6B6B,#4ECDC4" \
    --dpi 150

echo ""

echo "=========================================="
echo "✅ All tests completed successfully!"
echo "=========================================="
echo ""
echo "Generated files in $OUTPUT_DIR:"
ls -1 "$OUTPUT_DIR"
echo ""
echo "View plots with:"
echo "  xdg-open $OUTPUT_DIR/overview_combined.png"
echo ""

