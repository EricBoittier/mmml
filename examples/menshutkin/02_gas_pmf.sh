#!/usr/bin/env bash
# Gas-phase Menshutkin PMF: umbrella sampling along xi = r(C-Cl) - r(C-N) + MBAR.
#
# This is the validation leg of the campaign. Turan, Brickel & Meuwly
# (J. Phys. Chem. B 126, 1951 (2022)) report a gas-phase barrier of
# 35.8 kcal/mol for NH3 + MeCl with an MS-ARMD surface fitted to
# MP2/6-311++G(2d,2p); model_ext.json is a PhysNet fit to the same chemistry, so
# the two should agree before any solvent is added.
#
#   source examples/menshutkin/_env.sh
#   bash examples/menshutkin/02_gas_pmf.sh              # production
#   SMOKE=1 bash examples/menshutkin/02_gas_pmf.sh      # ~1 min CPU check
#
# Production wants a GPU: ssh gpu09 first (see _env.sh).
set -euo pipefail

# Without this the sampler's stdout is block-buffered when redirected, so the
# log lags thousands of steps behind the run and a dead job looks alive (and a
# live one looks dead). Cost every time it was forgotten today: several wrong
# conclusions about when runs stopped.
export PYTHONUNBUFFERED=1

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/examples/menshutkin/_env.sh"
cd "${ROOT}"

# MENSH_GAS_OUT lets a new run land somewhere else instead of overwriting the
# previous one. Every stage below passes --overwrite, so pointing this at a live
# directory DESTROYS it: on 2026-08-02 a 600-step smoke test run with the
# production paths wiped umbrella_rep1 and the 5-replica MBAR solve behind the
# published 34.56 kcal/mol barrier, which was not recoverable from disk (see
# artifacts/menshutkin/_archive/gas_smoke_20260802/README). Give a fresh run a
# fresh directory; merge or compare afterwards.
OUT="${MENSH_GAS_OUT:-${MENSH_ARTIFACTS}/gas}"
SEEDS="${OUT}/window_seeds.npz"

# PhysNet with hydrogens is unstable above ~0.25 fs in this sampler, so we buy
# Turan's 50 ps/window at a quarter of their 1 fs step rather than matching it.
DT_FS="${DT_FS:-0.25}"

# xi = r(C-Cl) - r(C-N) fixes only the difference of the two distances. Nothing
# in the bias acts along their sum, so the methyl can drift away from both
# partners while xi sits exactly on its target -- a species no reaction-path or
# normal-mode sample contains, and where the fitted surface is unbounded below.
# min(r(C-Cl), r(C-N)) is the invariant that rules it out: across the entire
# training set its maximum is 2.18 A, at every xi. Canonical atom order here is
# Cl=0, N=1, C=2, so the two competing bonds are (2,0) and (2,1).
MENSH_WALL_MIN_BOND="${MENSH_WALL_MIN_BOND:-2,0,2,1,2.25,100}"

# xi does not constrain the N-C-Cl angle either. Without this, product-side
# windows reorient the chloride to hydrogen-bond with the ammonium protons and
# sample a mean angle of 70 deg -- a real structure, but a different basin from
# backside attack, and they do NOT crash, so the profile is silently wrong.
# Healthy reaction-region windows stay at 165-173 deg (min observed 134.6), so
# 130 deg admits all legitimate sampling and excludes the reoriented basin.
MENSH_WALL_ANGLE="${MENSH_WALL_ANGLE:-1,2,0,130,50}"
# The channel restraint, matching the solvated runs. Without it the gas run
# leaves the reaction path exactly at the transition state: measured on the
# 2026-08-01 profile, the sum r(C-Cl)+r(C-N) sat 0.55-0.92 A off the reference
# over xi = +0.4..+1.0, with 63-89 % of frames beyond the 0.60 A tolerance the
# solvated runs enforce. That displaced the apparent TS from the model's own
# PES saddle at xi = +0.67 out to +1.10 and made the gas profile
# non-comparable with the solvated ones (different Hamiltonian).
# A constant --wall-sum does NOT substitute: it is a box, and the path is a
# line inside it.
MENSH_CHANNEL_JSON="${MENSH_CHANNEL_JSON:-${MENSH_ARTIFACTS}/reaction_channel.json}"
MENSH_WALL_CHANNEL="${MENSH_WALL_CHANNEL:-2,0,2,1,${MENSH_CHANNEL_JSON},sum_grid,0.60,50}"
MENSH_WALL_CHANNEL_CN="${MENSH_WALL_CHANNEL_CN:-2,0,2,1,${MENSH_CHANNEL_JSON},cn_grid,0.80,50}"
# Several short independent replicas rather than one long run. This checkpoint
# sustains about 8 ps per window before some window finds a spurious well in the
# fitted surface: 0.5 ps and 8 ps runs were clean, while a 55 ps run left three
# windows of thirty resetting 106, 84 and 42 times while the other 27 sampled
# perfectly. Replicas give the same total sampling, cost one replica rather than
# the campaign when a window fails, and are genuinely uncorrelated -- better for
# error bars than one long trajectory. Failed windows are dropped per replica by
# merge_replicas.py.
if [[ "${SMOKE:-0}" == "1" ]]; then
  N_REPLICAS="${N_REPLICAS:-2}"
  NSTEPS="${NSTEPS:-2000}"        # 0.5 ps per replica
  EQUIL="${EQUIL:-400}"           # 0.1 ps
  SAVEFREQ="${SAVEFREQ:-20}"
else
  N_REPLICAS="${N_REPLICAS:-5}"
  NSTEPS="${NSTEPS:-40000}"       # 10 ps per replica; 5 x 10 = 50 ps total
  EQUIL="${EQUIL:-4000}"          # 1 ps discarded per replica
  SAVEFREQ="${SAVEFREQ:-50}"      # 720 production frames per window per replica
fi

echo "=== 01: seed ${MENSH_N_WINDOWS} windows from the RC scan ==="
uv run python examples/menshutkin/01_seed_windows.py \
  --scan "${MENSH_SCAN}" \
  --xi-min "${MENSH_XI_MIN}" --xi-max "${MENSH_XI_MAX}" \
  --n-windows "${MENSH_N_WINDOWS}" \
  -o "${SEEDS}"

echo
echo "=== 02: ${N_REPLICAS} independent replicas x ${NSTEPS} steps (dt=${DT_FS} fs) ==="
REP_DIRS=()
for rep in $(seq 1 "${N_REPLICAS}"); do
  REP_DIR="${OUT}/umbrella_rep${rep}"
  REP_DIRS+=("${REP_DIR}")
  echo
  echo "--- replica ${rep}/${N_REPLICAS} (seed ${rep}) ---"
  # --seed-mode frames: window k starts from seed k. Stretch-seeding cannot
  # build these geometries because the methyl group inverts along the path.
  uv run mmml umbrella-sample \
    --checkpoint "${MENSH_CKPT}" \
    --structure "${SEEDS}" \
    --seed-mode frames \
    --cv-difference "${MENSH_CV_DIFFERENCE}" \
    --xi-min "${MENSH_XI_MIN}" --xi-max "${MENSH_XI_MAX}" \
    --n-windows "${MENSH_N_WINDOWS}" \
    --k "${MENSH_K_EV}" \
    --temperature "${MENSH_TEMPERATURE}" \
    --timestep "${DT_FS}" \
    --nsteps "${NSTEPS}" \
    --equilibration-steps "${EQUIL}" \
    --savefreq "${SAVEFREQ}" \
    --printfreq "$((SAVEFREQ * 20))" \
    --thermostat langevin \
    --wall-min-bond "${MENSH_WALL_MIN_BOND}" \
    --wall-angle "${MENSH_WALL_ANGLE}" \
    --wall-channel "${MENSH_WALL_CHANNEL}" \
    --wall-channel "${MENSH_WALL_CHANNEL_CN}" \
    --write-window-xyz \
    --seed "${rep}" \
    --output-dir "${REP_DIR}" \
    --overwrite
done

echo
echo "=== 02b: merge replicas (dropping windows that failed, per replica) ==="
uv run python examples/menshutkin/merge_replicas.py "${REP_DIRS[@]}" \
  -o "${OUT}/umbrella"

echo
echo "=== 03: MBAR ==="
uv run mmml umbrella-mbar --run-dir "${OUT}/umbrella" --checkpoint "${MENSH_CKPT}"

echo
echo "=== 04: profile + figures ==="
uv run python examples/menshutkin/03_gas_report.py --run-dir "${OUT}/umbrella"
