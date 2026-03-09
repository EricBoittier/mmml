#!/bin/bash
#
# Complete workflow script for MD simulation → IR/Raman spectra
#
# This script orchestrates:
# 1. Run MD simulation (jaxmd_dynamics.py)
# 2. Compute dipoles (compute_dipoles_for_traj.py)
# 3. Compute IR/Raman spectra (compute_ir_raman.py)
# 4. Plot results (plot_ir_raman.py)
#
# Usage:
#   ./run_ir_raman_workflow.sh \
#       --checkpoint /path/to/checkpoint \
#       --molecule CO2 \
#       --output-dir ./results \
#       [other options...]
#

set -e  # Exit on error
set -u  # Exit on undefined variable

# Default values
CHECKPOINT=""
MOLECULE="CO2"
OUTPUT_DIR="./ir_raman_results"
TEMPERATURE=300.0
TIMESTEP=0.5
NSTEPS=66000
MULTI_REPLICAS=16
MULTI_TRANSLATION=30.0
CUTOFF=10.0
SEED=0
ENSEMBLE="nve"
BATCH_SIZE=256
SUBSAMPLE=1
FREQ_RANGE_MIN=0
FREQ_RANGE_MAX=4500
APPLY_CORRECTIONS=false
PLOT_ONLY=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --molecule)
            MOLECULE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --timestep)
            TIMESTEP="$2"
            shift 2
            ;;
        --nsteps)
            NSTEPS="$2"
            shift 2
            ;;
        --multi-replicas)
            MULTI_REPLICAS="$2"
            shift 2
            ;;
        --multi-translation)
            MULTI_TRANSLATION="$2"
            shift 2
            ;;
        --cutoff)
            CUTOFF="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --ensemble)
            ENSEMBLE="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --subsample)
            SUBSAMPLE="$2"
            shift 2
            ;;
        --freq-range)
            FREQ_RANGE_MIN="$2"
            FREQ_RANGE_MAX="$3"
            shift 3
            ;;
        --apply-corrections)
            APPLY_CORRECTIONS=true
            shift
            ;;
        --plot-only)
            PLOT_ONLY=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Required:"
            echo "  --checkpoint PATH          Model checkpoint directory"
            echo ""
            echo "Simulation options:"
            echo "  --molecule NAME           Molecule name (default: CO2)"
            echo "  --output-dir PATH         Output directory (default: ./ir_raman_results)"
            echo "  --temperature FLOAT       Temperature in K (default: 300.0)"
            echo "  --timestep FLOAT          Timestep in fs (default: 0.5)"
            echo "  --nsteps INT              Number of MD steps (default: 66000)"
            echo "  --multi-replicas INT      Number of replicas (default: 16)"
            echo "  --multi-translation FLOAT Translation offset in Å (default: 30.0)"
            echo "  --cutoff FLOAT            Neighbor cutoff in Å (default: 10.0)"
            echo "  --seed INT                Random seed (default: 0)"
            echo "  --ensemble NAME           Ensemble: nve or nvt (default: nve)"
            echo ""
            echo "Analysis options:"
            echo "  --batch-size INT          Batch size for dipole computation (default: 256)"
            echo "  --subsample INT           Subsample trajectory (default: 1)"
            echo "  --freq-range MIN MAX      Frequency range in cm⁻¹ (default: 0 4500)"
            echo "  --apply-corrections       Apply frequency-dependent corrections"
            echo ""
            echo "Other:"
            echo "  --plot-only               Skip simulation, only plot existing results"
            echo "  --help                    Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 --checkpoint ./checkpoint --molecule CO2 --nsteps 100000"
            exit 0
            ;;
        *)
            echo -e "${RED}Error: Unknown option $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check required arguments
if [[ -z "$CHECKPOINT" ]]; then
    echo -e "${RED}Error: --checkpoint is required${NC}"
    echo "Use --help for usage information"
    exit 1
fi

if [[ ! -d "$CHECKPOINT" ]]; then
    echo -e "${RED}Error: Checkpoint directory not found: $CHECKPOINT${NC}"
    exit 1
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}IR/Raman Workflow${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Configuration:"
echo "  Checkpoint:    $CHECKPOINT"
echo "  Molecule:      $MOLECULE"
echo "  Output dir:    $OUTPUT_DIR"
echo "  Temperature:   $TEMPERATURE K"
echo "  Timestep:      $TIMESTEP fs"
echo "  Steps:         $NSTEPS"
echo "  Replicas:      $MULTI_REPLICAS"
echo "  Ensemble:      $ENSEMBLE"
echo ""

# File paths
TRAJ_FILE="$OUTPUT_DIR/multi_copy_traj_${MULTI_REPLICAS}x.npz"
METADATA_FILE="$OUTPUT_DIR/multi_copy_metadata.npz"
DIPOLES_FILE="$OUTPUT_DIR/dipoles.npz"
SPECTRA_FILE="$OUTPUT_DIR/ir_raman.npz"

# Step 1: Run MD simulation
if [[ "$PLOT_ONLY" == false ]]; then
    echo -e "${GREEN}[1/4] Running MD simulation...${NC}"
    
    if [[ -f "$TRAJ_FILE" ]]; then
        echo -e "${YELLOW}  Warning: Trajectory file exists, skipping simulation${NC}"
        echo -e "${YELLOW}  Delete $TRAJ_FILE to rerun simulation${NC}"
    else
        python jaxmd_dynamics.py \
            --checkpoint "$CHECKPOINT" \
            --molecule "$MOLECULE" \
            --multi-replicas "$MULTI_REPLICAS" \
            --multi-translation "$MULTI_TRANSLATION" \
            --temperature "$TEMPERATURE" \
            --timestep "$TIMESTEP" \
            --nsteps "$NSTEPS" \
            --cutoff "$CUTOFF" \
            --seed "$SEED" \
            --multi-ensemble "$ENSEMBLE" \
            --output-dir "$OUTPUT_DIR"
        
        if [[ ! -f "$TRAJ_FILE" ]]; then
            echo -e "${RED}Error: Trajectory file not created: $TRAJ_FILE${NC}"
            exit 1
        fi
        
        echo -e "${GREEN}  ✓ Simulation complete${NC}"
    fi
else
    echo -e "${YELLOW}[1/4] Skipping simulation (--plot-only)${NC}"
fi

# Step 2: Compute dipoles
echo ""
echo -e "${GREEN}[2/4] Computing dipoles...${NC}"

if [[ -f "$DIPOLES_FILE" ]]; then
    echo -e "${YELLOW}  Warning: Dipoles file exists, skipping computation${NC}"
    echo -e "${YELLOW}  Delete $DIPOLES_FILE to recompute${NC}"
else
    if [[ ! -f "$TRAJ_FILE" ]]; then
        echo -e "${RED}Error: Trajectory file not found: $TRAJ_FILE${NC}"
        exit 1
    fi
    
    if [[ ! -f "$METADATA_FILE" ]]; then
        echo -e "${RED}Error: Metadata file not found: $METADATA_FILE${NC}"
        exit 1
    fi
    
    python compute_dipoles_for_traj.py \
        --positions "$TRAJ_FILE" \
        --metadata "$METADATA_FILE" \
        --checkpoint "$CHECKPOINT" \
        --output "$DIPOLES_FILE" \
        --batch-size "$BATCH_SIZE"
    
    if [[ ! -f "$DIPOLES_FILE" ]]; then
        echo -e "${RED}Error: Dipoles file not created: $DIPOLES_FILE${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}  ✓ Dipoles computed${NC}"
fi

# Step 3: Compute IR/Raman spectra
echo ""
echo -e "${GREEN}[3/4] Computing IR/Raman spectra...${NC}"

if [[ -f "$SPECTRA_FILE" ]]; then
    echo -e "${YELLOW}  Warning: Spectra file exists, skipping computation${NC}"
    echo -e "${YELLOW}  Delete $SPECTRA_FILE to recompute${NC}"
else
    if [[ ! -f "$DIPOLES_FILE" ]]; then
        echo -e "${RED}Error: Dipoles file not found: $DIPOLES_FILE${NC}"
        exit 1
    fi
    
    python compute_ir_raman.py \
        --positions "$TRAJ_FILE" \
        --metadata "$METADATA_FILE" \
        --dipoles "$DIPOLES_FILE" \
        --checkpoint "$CHECKPOINT" \
        --output "$SPECTRA_FILE" \
        --subsample "$SUBSAMPLE" \
        --batch-size "$BATCH_SIZE"
    
    if [[ ! -f "$SPECTRA_FILE" ]]; then
        echo -e "${RED}Error: Spectra file not created: $SPECTRA_FILE${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}  ✓ Spectra computed${NC}"
fi

# Step 4: Plot results
echo ""
echo -e "${GREEN}[4/4] Plotting spectra...${NC}"

if [[ ! -f "$SPECTRA_FILE" ]]; then
    echo -e "${RED}Error: Spectra file not found: $SPECTRA_FILE${NC}"
    exit 1
fi

PLOT_ARGS=(
    "$SPECTRA_FILE"
    --output-dir "$OUTPUT_DIR"
    --freq-range "$FREQ_RANGE_MIN" "$FREQ_RANGE_MAX"
)

if [[ "$APPLY_CORRECTIONS" == true ]]; then
    PLOT_ARGS+=(--apply-corrections --temperature "$TEMPERATURE")
fi

python plot_ir_raman.py "${PLOT_ARGS[@]}"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✓ Workflow complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Output files:"
echo "  Trajectory:    $TRAJ_FILE"
echo "  Metadata:      $METADATA_FILE"
echo "  Dipoles:       $DIPOLES_FILE"
echo "  Spectra:       $SPECTRA_FILE"
echo "  Plots:         $OUTPUT_DIR/*.png"
echo ""

