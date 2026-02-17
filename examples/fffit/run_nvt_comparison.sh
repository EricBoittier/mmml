#!/usr/bin/env bash
set -euo pipefail
#
# run_nvt_comparison.sh
#
# End-to-end workflow:
#   1. Create a CGenFF residue via make_res.py
#   2. Build a periodic box at a given density via make_box.py
#   3. Run two NVT simulations with run_sim.py:
#       a) Normal  (ML/MM dimer corrections ON)
#       b) No-dimer (ML/MM dimer corrections OFF, --skip-ml-dimers)
#
# The script sources settings.source for site-specific paths (PY, etc.)
# and then overrides with command-line arguments.
#
# Usage:
#   ./run_nvt_comparison.sh --checkpoint /path/to/ckpt --side-length 30
#   ./run_nvt_comparison.sh --res MEOH --density 0.79 --side-length 25 --checkpoint /path/to/ckpt
#   ./run_nvt_comparison.sh --help
#
# Requirements:
#   - settings.source in the working directory (defines PY, and optionally
#     RES, N, L, NATOMS, etc.)
#   - packmol on PATH  (or set in mmml/pycharmmInterface/setupBox.py)
#   - A trained checkpoint directory
#
# Output:
#   workdir/
#     pdb/                  – residue and packed box PDBs
#     psf/                  – PyCHARMM PSF files
#     nvt_normal/           – simulation with ML/MM dimers
#     nvt_nodimer/          – simulation without ML/MM dimers
#

# ======================================================================
# Source site-specific settings first (CLI args override these below)
# ======================================================================
if [[ -f settings.source ]]; then
    echo "[run_nvt_comparison] Sourcing settings.source"
    source settings.source
fi

# ======================================================================
# Defaults (env vars from settings.source take effect here)
# ======================================================================
RES="${RES:-ACO}"
N="${N:-50}"
DENSITY="${DENSITY:-}"           # g/cm³; empty = use --n directly
SIDE_LENGTH="${L:-}"             # required; no default
NATOMS_MONOMER="${NATOMS:-10}"
CHECKPOINT="${CHECKPOINT:-}"

# Simulation defaults
TEMPERATURE="${TEMPERATURE:-298.0}"
TIMESTEP="${TIMESTEP:-0.3}"
NSTEPS_ASE="${NSTEPS_ASE:-5000}"
NSTEPS_JAXMD="${NSTEPS_JAXMD:-50000}"
ENSEMBLE="nvt"
ML_CUTOFF="${ML_CUTOFF:-0.1}"
MM_SWITCH_ON="${MM_SWITCH_ON:-5.0}"
MM_CUTOFF="${MM_CUTOFF:-5.0}"
ENERGY_CATCH="${ENERGY_CATCH:-0.5}"
HEATING_INTERVAL="${HEATING_INTERVAL:-500}"
WRITE_INTERVAL="${WRITE_INTERVAL:-100}"
NHC_TAU="${NHC_TAU:-100.0}"

# Flags
INCLUDE_MM="${INCLUDE_MM:-true}"
DEBUG="${DEBUG:-false}"
CHARMM_HEAT="${CHARMM_HEAT:-false}"
CHARMM_EQUIL="${CHARMM_EQUIL:-false}"
CHARMM_PROD="${CHARMM_PROD:-false}"

# Python CLI path
PY="${PY:-python -m mmml.cli}"
PDBFILE="pdb/init-packmol.pdb"

# ======================================================================
# Parse command-line overrides (these take priority over everything)
# ======================================================================
usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Create a residue, build a periodic box, and run two NVT simulations
(with and without ML/MM dimer corrections).

Options:
  --res NAME            Residue name (default: $RES)
  --side-length L       Box side length in Å (required)
  --n NUM               Number of molecules in the box (default: $N)
  --density DENS        Target density in kg/m³ (overrides --n; computes
                        N from density, side-length, and molecular weight.
                        E.g. water=997, methanol=791, acetone=784)
  --natoms-monomer N    Atoms per monomer (default: $NATOMS_MONOMER)
  --checkpoint PATH     Path to model checkpoint directory (required)
  --temperature T       Temperature in K (default: $TEMPERATURE)
  --timestep DT         Timestep in fs (default: $TIMESTEP)
  --nsteps-ase N        ASE equilibration steps (default: $NSTEPS_ASE)
  --nsteps-jaxmd N      JAX-MD production steps (default: $NSTEPS_JAXMD)
  --ml-cutoff D         ML cutoff distance (default: $ML_CUTOFF)
  --mm-switch-on D      MM switch-on distance (default: $MM_SWITCH_ON)
  --mm-cutoff D         MM cutoff width (default: $MM_CUTOFF)
  --nhc-tau TAU         NHC thermostat tau multiplier (default: $NHC_TAU)
  --charmm-heat         Run CHARMM heating stage before MD
  --charmm-equilibration Run CHARMM equilibration stage before MD
  --charmm-production   Run CHARMM production stage before MD
  --no-mm               Disable MM contributions
  --debug               Enable debug output
  -h, --help            Show this help message
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --res)            RES="$2";              shift 2 ;;
        --n)              N="$2";                shift 2 ;;
        --density)        DENSITY="$2";          shift 2 ;;
        --side-length)    SIDE_LENGTH="$2";      shift 2 ;;
        --natoms-monomer) NATOMS_MONOMER="$2";   shift 2 ;;
        --checkpoint)     CHECKPOINT="$2";       shift 2 ;;
        --temperature)    TEMPERATURE="$2";      shift 2 ;;
        --timestep)       TIMESTEP="$2";         shift 2 ;;
        --nsteps-ase)     NSTEPS_ASE="$2";       shift 2 ;;
        --nsteps-jaxmd)   NSTEPS_JAXMD="$2";     shift 2 ;;
        --ml-cutoff)      ML_CUTOFF="$2";        shift 2 ;;
        --mm-switch-on)   MM_SWITCH_ON="$2";     shift 2 ;;
        --mm-cutoff)      MM_CUTOFF="$2";        shift 2 ;;
        --nhc-tau)        NHC_TAU="$2";          shift 2 ;;
        --charmm-heat)           CHARMM_HEAT="true";   shift ;;
        --charmm-equilibration)  CHARMM_EQUIL="true";  shift ;;
        --charmm-production)     CHARMM_PROD="true";   shift ;;
        --no-mm)          INCLUDE_MM="false";    shift   ;;
        --debug)          DEBUG="true";          shift   ;;
        -h|--help)        usage ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            ;;
    esac
done

# ======================================================================
# Validate
# ======================================================================
if [[ -z "$CHECKPOINT" ]]; then
    echo "ERROR: --checkpoint is required (path to trained model checkpoint)." >&2
    exit 1
fi

if [[ ! -d "$CHECKPOINT" ]]; then
    echo "ERROR: Checkpoint directory does not exist: $CHECKPOINT" >&2
    exit 1
fi

if [[ -z "$SIDE_LENGTH" ]]; then
    echo "ERROR: --side-length is required (box side length in Å)." >&2
    exit 1
fi

echo "========================================================================"
echo "  NVT Comparison Workflow"
echo "========================================================================"
echo "  Residue:           $RES"
echo "  Box side length:   $SIDE_LENGTH Å"
if [[ -n "$DENSITY" ]]; then
echo "  Density:           $DENSITY kg/m³  (will compute N after residue is built)"
else
echo "  N molecules:       $N"
fi
echo "  Atoms/monomer:     $NATOMS_MONOMER"
echo "  Checkpoint:        $CHECKPOINT"
echo "  Temperature:       $TEMPERATURE K"
echo "  Timestep:          $TIMESTEP fs"
echo "  ASE steps:         $NSTEPS_ASE"
echo "  JAX-MD steps:      $NSTEPS_JAXMD"
echo "  ML cutoff:         $ML_CUTOFF"
echo "  MM switch-on:      $MM_SWITCH_ON"
echo "  MM cutoff:         $MM_CUTOFF"
echo "  Include MM:        $INCLUDE_MM"
echo "  NHC tau:           $NHC_TAU"
echo "  CHARMM heat:       $CHARMM_HEAT"
echo "  CHARMM equil:      $CHARMM_EQUIL"
echo "  CHARMM production: $CHARMM_PROD"
echo "========================================================================"

# ======================================================================
# Ensure crystal_image.str is in the working directory (required by CHARMM)
# ======================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CRYSTAL_IMAGE="${SCRIPT_DIR}/crystal_image.str"

if [[ ! -f crystal_image.str ]]; then
    if [[ -f "$CRYSTAL_IMAGE" ]]; then
        cp "$CRYSTAL_IMAGE" crystal_image.str
        echo "[run_nvt_comparison] Copied crystal_image.str into working directory"
    else
        echo "ERROR: crystal_image.str not found in working directory or at ${CRYSTAL_IMAGE}" >&2
        echo "       CHARMM requires this file for periodic image setup." >&2
        exit 1
    fi
else
    echo "[run_nvt_comparison] crystal_image.str already present in working directory"
fi

# ======================================================================
# Step 1: Create residue
# ======================================================================
echo ""
echo "========== STEP 1: Create residue ($RES) =========="
if [[ "$PY" == "python -m mmml.cli" ]]; then
    python -m mmml.cli.make_res --res "$RES" --skip-energy-show \
        2>&1 | tee make_res.log
else
    ${PY}/make_res.py --res "$RES" --skip-energy-show \
        2>&1 | tee make_res.log
fi
echo "[OK] Residue $RES created."

# ======================================================================
# Compute N from density if requested  (density + side_length -> N)
# This runs after make_res so pdb/initial.pdb exists for mass lookup.
# ======================================================================
if [[ -n "$DENSITY" ]]; then
    echo ""
    echo "[run_nvt_comparison] Computing N from density=${DENSITY} kg/m³, L=${SIDE_LENGTH} Å"
    N=$(python3 -c "
import ase, ase.io, sys, os
# Prefer xyz/initial.xyz (has correct atomic numbers from PSF).
# Fall back to pdb/initial.pdb (CHARMM PDB may have wrong element types).
mol = None
for path in ['xyz/initial.xyz', 'pdb/initial.pdb']:
    if os.path.exists(path):
        mol = ase.io.read(path)
        print(f'  Read molecule from {path}', file=sys.stderr)
        break
if mol is None:
    print('ERROR: neither xyz/initial.xyz nor pdb/initial.pdb found after make_res.', file=sys.stderr)
    sys.exit(1)
M = mol.get_masses().sum()           # amu = g/mol
formula = mol.get_chemical_formula()
rho_kgm3 = float('${DENSITY}')      # kg/m³
L_ang    = float('${SIDE_LENGTH}')   # Å
NA = 6.02214076e23
# Volume in m³:  (L_ang * 1e-10)^3
V_m3 = (L_ang * 1e-10) ** 3
# mass_in_box [kg] = rho * V;  convert to g -> * 1000
# moles = mass_g / M;  N = moles * NA
N = rho_kgm3 * V_m3 * 1000.0 * NA / M
N = max(1, round(N))
print(f'{N}', file=sys.stdout)
# Sanity check
print(f'  Formula: {formula}, molecular weight: {M:.2f} g/mol', file=sys.stderr)
print(f'  Box volume: {V_m3:.4e} m³ = {(L_ang**3):.1f} Å³', file=sys.stderr)
print(f'  N molecules: {N}', file=sys.stderr)
if N > 5000:
    print(f'  WARNING: N={N} seems very large for L={L_ang} Å. Check density units (should be kg/m³).', file=sys.stderr)
if N < 2:
    print(f'  WARNING: N={N} seems very small. Check density and side-length.', file=sys.stderr)
")
    echo "[run_nvt_comparison] Computed N = ${N} molecules (density=${DENSITY} kg/m³, L=${SIDE_LENGTH} Å)"
fi

# ======================================================================
# Step 2: Build box
# ======================================================================
echo ""
echo "========== STEP 2: Build box (N=$N, L=$SIDE_LENGTH Å) =========="
if [[ "$PY" == "python -m mmml.cli" ]]; then
    python -m mmml.cli.make_box --res "$RES" --n "$N" --side_length "$SIDE_LENGTH" \
        2>&1 | tee make_box.log
else
    ${PY}/make_box.py --res "$RES" --n "$N" --side_length "$SIDE_LENGTH" \
        2>&1 | tee make_box.log
fi
echo "[OK] Box created: $PDBFILE"

# ======================================================================
# Helper: build run_sim command
# ======================================================================
build_sim_cmd() {
    local prefix="$1"
    local skip_dimers="$2"  # "true" or "false"

    local CMD=""
    if [[ "$PY" == "python -m mmml.cli" ]]; then
        CMD="python -m mmml.cli.run_sim"
    else
        CMD="${PY}/run_sim.py"
    fi

    CMD="$CMD \
  --pdbfile $PWD/${PDBFILE} \
  --checkpoint ${CHECKPOINT} \
  --cell ${SIDE_LENGTH} \
  --ensemble ${ENSEMBLE} \
  --n-monomers ${N} \
  --n-atoms-monomer ${NATOMS_MONOMER} \
  --ml-cutoff ${ML_CUTOFF} \
  --mm-switch-on ${MM_SWITCH_ON} \
  --mm-cutoff ${MM_CUTOFF} \
  --energy-catch ${ENERGY_CATCH} \
  --temperature ${TEMPERATURE} \
  --timestep ${TIMESTEP} \
  --nsteps_jaxmd ${NSTEPS_JAXMD} \
  --nsteps_ase ${NSTEPS_ASE} \
  --heating_interval ${HEATING_INTERVAL} \
  --write_interval ${WRITE_INTERVAL} \
  --nhc-tau ${NHC_TAU} \
  --output-prefix ${prefix}"

    if [[ "$INCLUDE_MM" == "true" ]]; then
        CMD="$CMD --include-mm"
    fi

    if [[ "$skip_dimers" == "true" ]]; then
        CMD="$CMD --skip-ml-dimers"
    fi

    if [[ "$DEBUG" == "true" ]]; then
        CMD="$CMD --debug"
    fi

    echo "$CMD"
}

# ======================================================================
# Step 3a: NVT with ML/MM dimers ON
# ======================================================================
echo ""
echo "========== STEP 3a: NVT simulation (ML/MM dimers ON) =========="
mkdir -p nvt_normal
SIM_CMD_NORMAL=$(build_sim_cmd "nvt_normal/sim" "false")
echo "Running: $SIM_CMD_NORMAL"
eval "$SIM_CMD_NORMAL" 2>&1 | tee nvt_normal/simulation.log
echo "[OK] Normal NVT simulation complete -> nvt_normal/"

# ======================================================================
# Step 3b: NVT with ML/MM dimers OFF
# ======================================================================
echo ""
echo "========== STEP 3b: NVT simulation (ML/MM dimers OFF) =========="
mkdir -p nvt_nodimer
SIM_CMD_NODIMER=$(build_sim_cmd "nvt_nodimer/sim" "true")
echo "Running: $SIM_CMD_NODIMER"
eval "$SIM_CMD_NODIMER" 2>&1 | tee nvt_nodimer/simulation.log
echo "[OK] No-dimer NVT simulation complete -> nvt_nodimer/"

# ======================================================================
# Summary
# ======================================================================
echo ""
echo "========================================================================"
echo "  DONE — NVT Comparison"
echo "========================================================================"
echo ""
echo "  Residue: $RES, N=$N molecules, L=$SIDE_LENGTH Å"
if [[ -n "$DENSITY" ]]; then
echo "  Density: $DENSITY kg/m³ (N computed from density)"
fi
echo ""
echo "  Normal (dimers ON):  nvt_normal/"
echo "  No-dimer (OFF):      nvt_nodimer/"
echo ""
echo "  HDF5 trajectories:"
ls -lh nvt_normal/sim_*.h5  2>/dev/null || echo "    (none found in nvt_normal/)"
ls -lh nvt_nodimer/sim_*.h5 2>/dev/null || echo "    (none found in nvt_nodimer/)"
echo ""
echo "  ASE trajectories:"
ls -lh nvt_normal/*.traj  2>/dev/null || echo "    (none found in nvt_normal/)"
ls -lh nvt_nodimer/*.traj 2>/dev/null || echo "    (none found in nvt_nodimer/)"
echo ""
echo "  Compare with:"
echo "    python -c \""
echo "    import h5py, numpy as np"
echo "    a = h5py.File('nvt_normal/sim_nvt_trajectory.h5','r')"
echo "    b = h5py.File('nvt_nodimer/sim_nvt_trajectory.h5','r')"
echo "    print('Normal  E:', np.array(a['potential_energy'][:10]))"
echo "    print('NoDimer E:', np.array(b['potential_energy'][:10]))"
echo "    \""
echo "========================================================================"
