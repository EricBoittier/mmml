#!/usr/bin/env bash
set -euo pipefail
WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$WORKFLOW_ROOT/../.." && pwd)"
cd "$REPO_ROOT"

_cfg_raw="${MMML_WORKFLOW_CONFIG:-config.yaml}"
if [[ "$_cfg_raw" = /* ]]; then
  CFG="${_cfg_raw}"
else
  CFG="${WORKFLOW_ROOT}/${_cfg_raw}"
fi
export MMML_WORKFLOW_CONFIG="$CFG"

# shellcheck source=../../../scripts/resolve_mmml_env.sh
source "$REPO_ROOT/scripts/resolve_mmml_env.sh"
mmml_resolve_env "$REPO_ROOT"
PY="${MMML_PYTHON}"

"$PY" -c "
from pathlib import Path
import sys
# Workflow scripts first; campaign_lib adds pbc_solvent_burst/scripts for bulk_density.
sys.path.insert(0, '${WORKFLOW_ROOT}/scripts')
from campaign_lib import (
    cell_bulk_density_fraction,
    campaign_job_order,
    dynamics_campaign_enabled,
    heat_compare_enabled,
    iter_matrix_cells,
    load_config,
    matrix_box_sizes,
    matrix_heat_thermostats,
    matrix_job_count,
    matrix_setup_ids,
    matrix_temperatures,
    parse_dynamics_legs,
    prep_sweep_anchor_cell,
    prep_sweep_enabled,
    prep_sweep_variant_ids,
    resolve_checkpoint,
    slurm_launch_jobs,
    slurm_small_cluster_max_n,
    slurm_tier_enabled,
    slurm_tier_resource_pools,
    warmup_mlpot_enabled,
)
from mmml.cli.run.warmup_mlpot_jax import resolve_warmup_do_mm_for_config
from setup_variants import resolve_setup_variant
from bulk_density import bulk_reference_table, matrix_uses_bulk_density
cfg = load_config(Path('${CFG}'))
ckpt = resolve_checkpoint(str(cfg['checkpoint']))
print('Preflight OK')
print('checkpoint:', ckpt)
print('setups:', matrix_setup_ids(cfg))
for sid in matrix_setup_ids(cfg):
    v = resolve_setup_variant(sid)
    print(f'  {sid}: {v.description}')
print('matrix jobs:', matrix_job_count(cfg))
print('heat compare:', heat_compare_enabled(cfg))
print('heat_thermostats:', matrix_heat_thermostats(cfg) or ['<mini-only>'])
print('dynamics legs:', parse_dynamics_legs(cfg))
if dynamics_campaign_enabled(cfg):
    print('campaign job order:', campaign_job_order(cfg))
print('temperatures:', matrix_temperatures(cfg))
print('box_sizes:', matrix_box_sizes(cfg))
if matrix_uses_bulk_density(cfg):
    print('bulk_density_fractions:', cfg.get('bulk_density_fractions'))
    print('bulk N at L (298 K liquid, DCM monomers per box):')
    print(bulk_reference_table(matrix_box_sizes(cfg)))
else:
    print('cluster_sizes:', cfg.get('cluster_sizes'))
print('solvents:', cfg['solvents'])
print('slurm tiering:', slurm_tier_enabled(cfg))
if slurm_tier_enabled(cfg):
    print('slurm_small_cluster_max_n:', slurm_small_cluster_max_n(cfg))
    print('slurm resource pools:', slurm_tier_resource_pools(cfg))
print('slurm launch -j:', slurm_launch_jobs(cfg))
print()
print('Sample matrix cells:')
for i, cell in enumerate(iter_matrix_cells(cfg)):
    frac = cell_bulk_density_fraction(cell, cfg)
    ht = f' ht={cell.heat_thermostat}' if cell.heat_thermostat else ''
    frac_s = f'{frac:.2f}x bulk' if frac is not None else 'custom N'
    sweep = f' sw={cell.sweep_id}' if cell.sweep_id else ''
    print(f'  {cell.setup_id} DCM:{cell.n_monomers} T={cell.temperature:.0f}K L={cell.box_size:.0f}{ht}{sweep} ({frac_s})')
    if i >= 7:
        print('  ...')
        break
from mmml.interfaces.pycharmmInterface.mlpot.mlpot_limits import (
    NPR_TIERS,
    estimate_ml_atoms,
    pbc_pair_budget_box_side_A,
    required_max_npr,
    select_npr_tier_for_build,
    validate_mlpot_system_size,
)
worst = max(
    iter_matrix_cells(cfg),
    key=lambda c: required_max_npr(
        estimate_ml_atoms(c.n_monomers, solvent=c.solvent),
        pbc=True,
        box_side_A=pbc_pair_budget_box_side_A(
            estimate_ml_atoms(c.n_monomers, solvent=c.solvent),
            c.box_size,
        ),
    ),
)
max_n_ml = estimate_ml_atoms(worst.n_monomers, solvent=worst.solvent)
worst_box = worst.box_size
tier = select_npr_tier_for_build(max_n_ml, pbc=True, box_side_A=worst_box)
budget_box = pbc_pair_budget_box_side_A(max_n_ml, worst_box)
print(
    f'max matrix n_ml={max_n_ml} L={worst_box:g} CHARMM tier={tier} '
    f'(max_Npr={NPR_TIERS[tier]}, PBC pairs)'
)
validate_mlpot_system_size(max_n_ml, pbc=True, box_side_A=budget_box)
from mmml.interfaces.pycharmmInterface.charmm_paths import assert_cgenff_toppar_readable
toppar = assert_cgenff_toppar_readable()
print('CGENFF RTF:', toppar.rtf)
print('CGENFF PRM:', toppar.prm)
from campaign_lib import prep_sweep_enabled, prep_sweep_variant_ids, prep_sweep_anchor_cell, matrix_job_count
if prep_sweep_enabled(cfg):
    anchor = prep_sweep_anchor_cell(cfg)
    print('prep_sweep: enabled')
    print('prep_sweep anchor:', anchor)
    print('prep_sweep variants:', prep_sweep_variant_ids(cfg))
    print('prep_sweep jobs:', matrix_job_count(cfg))
print('workflow_config:', Path('${CFG}'))
allow_hg = cfg.get('allow_high_grms', False)
print('allow_high_grms:', bool(allow_hg))
print('warmup_mlpot_jax:', warmup_mlpot_enabled(cfg))
print('warmup_do_mm:', resolve_warmup_do_mm_for_config(cfg))
"

if ! command -v packmol >/dev/null 2>&1; then
  _packmol_bundled="$("$PY" -c "from mmml.interfaces.pycharmmInterface.packmol_placement import packmol_executable; print(packmol_executable())" 2>/dev/null || true)"
  if [[ -n "$_packmol_bundled" && -x "$_packmol_bundled" ]]; then
    echo "packmol: bundled ${_packmol_bundled}"
  else
    echo "WARNING: packmol not on PATH and bundled binary unavailable." >&2
  fi
fi

if [[ -n "${MMML_CKPT:-}" ]]; then
  echo "MMML_CKPT=${MMML_CKPT} (optional override when config uses \${MMML_CKPT})"
fi

if [[ -n "${MMML_MLPOT_ALLOW_HIGH_GRMS:-}" ]]; then
  echo "NOTE: MMML_MLPOT_ALLOW_HIGH_GRMS=${MMML_MLPOT_ALLOW_HIGH_GRMS} (env override; config allow_high_grms also applies via campaign.yaml)" >&2
fi
