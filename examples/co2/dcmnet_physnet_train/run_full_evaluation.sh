#!/bin/bash
"
Complete evaluation pipeline: evaluate splits and create R plots.

Usage:
    ./run_full_evaluation.sh \
        --checkpoint ./ckpts/model \
        --train-efd ./data/train_efd.npz \
        --valid-efd ./data/valid_efd.npz \
        --train-esp ./data/train_esp.npz \
        --valid-esp ./data/valid_esp.npz \
        --output-dir ./evaluation
"

set -e  # Exit on error

# Default arguments
CHECKPOINT=""
TRAIN_EFD=""
VALID_EFD=""
TEST_EFD=""
TRAIN_ESP=""
VALID_ESP=""
TEST_ESP=""
OUTPUT_DIR="./evaluation"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --train-efd)
            TRAIN_EFD="$2"
            shift 2
            ;;
        --valid-efd)
            VALID_EFD="$2"
            shift 2
            ;;
        --test-efd)
            TEST_EFD="$2"
            shift 2
            ;;
        --train-esp)
            TRAIN_ESP="$2"
            shift 2
            ;;
        --valid-esp)
            VALID_ESP="$2"
            shift 2
            ;;
        --test-esp)
            TEST_ESP="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Check required arguments
if [ -z "$CHECKPOINT" ] || [ -z "$TRAIN_EFD" ] || [ -z "$VALID_EFD" ] || \
   [ -z "$TRAIN_ESP" ] || [ -z "$VALID_ESP" ]; then
    echo "Error: Missing required arguments"
    echo "Usage: $0 --checkpoint <path> --train-efd <path> --valid-efd <path> \\"
    echo "          --train-esp <path> --valid-esp <path> [--test-efd <path>] \\"
    echo "          [--test-esp <path>] [--output-dir <path>]"
    exit 1
fi

echo "=============================================================================="
echo "FULL EVALUATION PIPELINE"
echo "=============================================================================="
echo ""
echo "Checkpoint: $CHECKPOINT"
echo "Train EFD: $TRAIN_EFD"
echo "Valid EFD: $VALID_EFD"
echo "Train ESP: $TRAIN_ESP"
echo "Valid ESP: $VALID_ESP"
[ -n "$TEST_EFD" ] && echo "Test EFD: $TEST_EFD"
[ -n "$TEST_ESP" ] && echo "Test ESP: $TEST_ESP"
echo "Output dir: $OUTPUT_DIR"
echo ""

# Step 1: Run Python evaluation
echo "=============================================================================="
echo "Step 1: Evaluating model on data splits..."
echo "=============================================================================="

PYTHON_CMD="python evaluate_splits.py \
    --checkpoint $CHECKPOINT \
    --train-efd $TRAIN_EFD \
    --valid-efd $VALID_EFD \
    --train-esp $TRAIN_ESP \
    --valid-esp $VALID_ESP \
    --output-dir $OUTPUT_DIR"

if [ -n "$TEST_EFD" ] && [ -n "$TEST_ESP" ]; then
    PYTHON_CMD="$PYTHON_CMD --test-efd $TEST_EFD --test-esp $TEST_ESP"
fi

$PYTHON_CMD

if [ $? -ne 0 ]; then
    echo "Error: Python evaluation failed"
    exit 1
fi

# Step 2: Check if R is available
echo ""
echo "=============================================================================="
echo "Step 2: Creating R plots..."
echo "=============================================================================="

if ! command -v Rscript &> /dev/null; then
    echo "Warning: Rscript not found. Skipping plotting."
    echo "Install R and required packages:"
    echo "  install.packages(c('ggplot2', 'dplyr', 'gridExtra', 'viridis'))"
    echo ""
    echo "Then run manually:"
    echo "  Rscript plot_evaluation_results.R $OUTPUT_DIR"
    exit 0
fi

# Check if required R packages are installed
Rscript -e "if (!require('ggplot2', quietly=TRUE)) stop('ggplot2 not installed'); \
            if (!require('dplyr', quietly=TRUE)) stop('dplyr not installed'); \
            if (!require('gridExtra', quietly=TRUE)) stop('gridExtra not installed'); \
            if (!require('viridis', quietly=TRUE)) stop('viridis not installed')" 2>/dev/null

if [ $? -ne 0 ]; then
    echo "Warning: Some R packages are missing. Installing..."
    Rscript -e "install.packages(c('ggplot2', 'dplyr', 'gridExtra', 'viridis'), repos='https://cloud.r-project.org')"
fi

# Run R plotting script
Rscript plot_evaluation_results.R "$OUTPUT_DIR"

if [ $? -ne 0 ]; then
    echo "Error: R plotting failed"
    exit 1
fi

# Summary
echo ""
echo "=============================================================================="
echo "EVALUATION COMPLETE"
echo "=============================================================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo "  - evaluation_results.csv (raw data)"
echo "  - plots/ (all visualization plots)"
echo ""
echo "Plots created:"
echo "  1D plots: bonds/angles vs errors"
echo "  2D plots: (r1+r2+r1r2, angle) vs errors"
echo ""

