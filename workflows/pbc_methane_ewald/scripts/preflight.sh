#!/usr/bin/env bash
set -euo pipefail
WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$WORKFLOW_ROOT/../.." && pwd)"
cd "$REPO_ROOT"

CFG="${MMML_WORKFLOW_CONFIG:-config.yaml}"
if [[ "${1:-}" == "--config" ]]; then
  CFG="${2:?--config requires path}"
fi
if [[ "$CFG" == */* && "$CFG" != /* ]]; then
  CFG_PATH="$(cd "$(dirname "$CFG")" && pwd)/$(basename "$CFG")"
elif [[ "$CFG" == /* ]]; then
  CFG_PATH="$CFG"
else
  CFG_PATH="${WORKFLOW_ROOT}/${CFG}"
fi

# shellcheck source=../../../scripts/resolve_mmml_env.sh
source "$REPO_ROOT/scripts/resolve_mmml_env.sh"
mmml_resolve_env "$REPO_ROOT"
PY="${MMML_PYTHON}"

"$PY" -c "
from pathlib import Path
import sys
_WF = Path('${WORKFLOW_ROOT}') / 'scripts'
_BURST = Path('${WORKFLOW_ROOT}').parent / 'pbc_solvent_burst' / 'scripts'
# Methane campaign_lib must win over the burst campaign_lib of the same name.
sys.path.insert(0, str(_BURST))
sys.path.insert(0, str(_WF))
from bulk_density import bulk_reference_table, matrix_uses_bulk_density
from campaign_lib import (
    checkpoint_map,
    cell_ml_atoms,
    iter_matrix_cells,
    load_config,
    matrix_backends,
    matrix_box_sizes,
    matrix_embeddings,
    matrix_job_count,
    matrix_temperatures,
    resolve_checkpoint_path,
    slurm_launch_jobs,
    validate_checkpoint,
)
from cleanup_strategy import resolve_cleanup_strategy
from mmml.interfaces.pycharmmInterface.cgenff_residues import require_cgenff_residue_name
from mmml.analysis.residue_geometry import bundled_monomer_pdb

cfg = load_config(Path('${CFG_PATH}'))
require_cgenff_residue_name('METH')
mono = bundled_monomer_pdb('METH')
assert mono is not None and mono.is_file(), mono
strategy = resolve_cleanup_strategy(cfg)
print('Preflight OK')
print('config:', '${CFG_PATH}')
print('cleanup_strategy:', strategy.name)
print('lr_solver:', cfg.get('lr_solver'))
print('mm_nonbond_mode:', cfg.get('mm_nonbond_mode'))
print('ensemble:', cfg.get('ensemble'))
print('matrix jobs:', matrix_job_count(cfg))
print('temperatures:', matrix_temperatures(cfg))
print('box_sizes:', matrix_box_sizes(cfg))
print('backends:', matrix_backends(cfg))
print('checkpoints:')
for slug, raw in checkpoint_map(cfg).items():
    path = resolve_checkpoint_path(raw)
    validate_checkpoint(path)
    print(f'  {slug}: {path}')
if matrix_uses_bulk_density(cfg):
    print('bulk_density_fractions:', cfg.get('bulk_density_fractions'))
    print('bulk N at L (liquid methane monomers per box):')
    print(bulk_reference_table(matrix_box_sizes(cfg)))
print('sample tags:')
for i, cell in enumerate(iter_matrix_cells(cfg)):
    if i >= 8:
        print('  ...')
        break
    print(f'  {cell.solvent}:{cell.n_monomers} T={cell.temperature} '
          f'ckpt={cell.checkpoint_slug} emb={cell.embedding}/{cell.mm_charge_mode} '
          f'backend={cell.backend} '
          f'n_ml={cell_ml_atoms(cell)}')
print('embeddings:', [f'{e}/{m}' for e, m in matrix_embeddings(cfg)])
print('slurm -j:', slurm_launch_jobs(cfg))
print('METH monomer pdb:', mono)
"
