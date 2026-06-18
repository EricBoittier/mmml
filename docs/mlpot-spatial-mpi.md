# Spatial ML MPI decomposition (design)

Target architecture for multi-rank MLpot aligned with CHARMM DOMDEC and existing sparse-dimer logic.

## Current state

- **Single rank (`np=1`)** — global sparse dimers, local GPU chunking (`ml_batch_size`, `ml_gpu_count`).
- **CHARMM domdec off** wherever MLpot runs (stability stopgap).
- **`np>1`** — rank-0 MLpot bridge for correctness only; not a performance path.

## Target (Phase 2+)

Each MPI rank:

1. Owns monomers by COM in its spatial domain.
2. Builds halo ghost monomers within `R_halo ≈ mm_switch_on + r_physnet`.
3. Selects active dimers (COM &lt; `mm_switch_on`) visible in `domain ∪ halo`.
4. Evaluates PhysNet on owned monomers + canonically owned dimers only.
5. Scatters forces locally, then **MPI-reduces** ghost atom contributions.

Neither rank-0-global-every-step nor full MLpot replication on all ranks.

## Module layout

Python package: [`mmml/interfaces/pycharmmInterface/mlpot/mpi_spatial/`](../mmml/interfaces/pycharmmInterface/mlpot/mpi_spatial/)

| Module | Role |
|--------|------|
| `domain.py` | Monomer COM → rank, halo masks |
| `active_set.py` | Per-rank monomer + dimer index lists |
| `dedup.py` | Canonical dimer owner at domain boundaries |
| `force_exchange.py` | Ghost force reduction |
| `pool.py` | Optional gather → ML pool → scatter |
| `domdec_info.py` | PyCHARMM / CHARMM DOMDEC API survey |

## DOMDEC integration (Phase 3)

CHARMM DOMDEC exposes decomposition via Fortran (`domdec_common`, `q_domdec`, `q_recip_node`) but **PyCHARMM does not currently export per-rank atom ownership or ghost lists** to Python. Until that API exists, `mpi_spatial.domain` uses a deterministic Python grid from box size and `n_ranks`.

See `domdec_info.py` for the full survey and open questions.

## Medium PBC now

For 500–2000 monomers, stay on [medium PBC single-rank workflow](mlpot-medium-pbc.md) until Phase 2 is wired into `hybrid_mlpot.py`.
