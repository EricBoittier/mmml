# Neighbor-list validation (manual / Slurm)

Standalone scripts to audit MM PBC neighbor lists: reference oracles (brute-force,
[Vesin](https://luthaf.fr/vesin/latest/index.html)), path parity (cell-list, Vesin
backend, jax-md), and `update_mm_pairs` skin/interval cache behavior.

These scripts are **not** collected by pytest (no `test_` prefix). Run locally or
on a GPU/CHARMM node when validating NL changes.

## Prerequisites

```bash
# From repo root
uv sync
uv sync --extra nl-validation   # optional: Vesin reference oracle
export JAX_PLATFORMS=cpu        # if GPU JAX init fails in your shell

bash tests/functionality/neighbor_lists/run_all.sh
# or individual steps:
uv run python tests/functionality/neighbor_lists/02_path_parity.py
```

Script 04 requires PyCHARMM + CGENFF (`CHARMM_HOME`, `CHARMM_LIB_DIR`) and builds a
2×ACO dimer PSF before calling `build_mm_energy_forces_fn`. Scripts 00–03 do not.

## Ladder

| Step | Script | CHARMM | What it checks |
|------|--------|--------|----------------|
| 0 | `00_check_nl_env.py` | no | jax, jax-md, nl_backend, optional vesin |
| 1 | `01_reference_oracle_smoke.py` | no | Vesin vs brute-force reference pairs |
| 2 | `02_path_parity.py` | no | cell-list / Vesin backend / jax-md vs reference |
| 3 | `03_skin_interval_audit.py` | no | `neighbor_pair_cache_should_reuse` (skin=0 box fix) |
| 4 | `04_update_mm_pairs_integration.py` | yes | live `update_mm_pairs` stats (`reused`, `updates`) |

## Quick run

```bash
bash tests/functionality/neighbor_lists/run_all.sh
```

Script 04 is skipped for CHARMM in `run_all.sh`. Run integration manually:

```bash
python tests/functionality/neighbor_lists/04_update_mm_pairs_integration.py
```

## Pass criteria

- **Path parity**: zero symmetric diff vs reference (Vesin or brute-force), except
  documented cutoff-boundary pairs (`dist ≈ cutoff`).
- **skin=0**: reuse when `calls % interval != 0` and box unchanged; **no reuse** when
  box side changes (NPT).
- **skin>0**: reuse when max atom displacement ≤ skin and box stable.

## Production backend selection

Set `MMML_MM_NL_BACKEND` or pass `mm_nl_backend` to `build_mm_energy_forces_fn`:

| Value | Behavior |
|-------|----------|
| `auto` (default) | jax-md incremental updates; Vesin/cell-list for static path and overflow fallback |
| `vesin` | Vesin rebuild (+ cell-list fallback if Vesin fails) |
| `cell_list` | NumPy cell-list only |
| `jax_md` | jax-md primary (fallback prefers Vesin when available) |

See [`NONBOND_LISTS.md`](../../mmml/interfaces/pycharmmInterface/mlpot/NONBOND_LISTS.md).

## Related tools

- [`scripts/validate_mlpot_pair_lists.py`](../../scripts/validate_mlpot_pair_lists.py) — post-MD geometry audit (CRD/DCD)
- [`tests/functionality/mmml_tests/test_ase_jaxmd_pbc_consistency.py`](../mmml_tests/test_ase_jaxmd_pbc_consistency.py) — pytest integration
