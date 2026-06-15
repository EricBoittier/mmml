# GOAT conformer search with MMML external potential

Global optimization using ORCA [GOAT](https://www.faccts.de/docs/orca/6.0/manual/contents/typical/GOAT.html)
with MMML as the energy/gradient engine (`mmml orca-server` + `mmml orca-client`).

GOAT runs many geometry optimizations (~O(100 × Natoms) by default). Use a **fast ML
checkpoint** and parallelize with `!PAL` / `%GOAT NWORKERS`. The settings below are a
**reduced smoke preset** for ethanol (9 atoms); increase workers and iterations for
production ensembles.

## Prerequisites

Same as `examples/orca/water_opt/`:

1. `mmml orca-server` running with `--warmup`
2. `examples/orca/mmml-orca-client` executable (`chmod +x`) — set `ProgExt` to its **absolute** path (same as `water_opt.inp`)
3. ORCA 6.x

```bash
export MMML_CHECKPOINT=~/mmml/mmml/models/physnetjax/defaults/hf_json/test-f41c04c0-62e3-4785-9018-351ffdc161c4_epoch-251_portable.json

mmml orca-server --checkpoint "$MMML_CHECKPOINT" --warmup -b 127.0.0.1:8888
```

## Run

```bash
cd ~/mmml/examples/orca/goat_ethanol
orca ethanol_goat.inp
```

## Expected outputs

| File | Content |
|------|---------|
| `ethanol.globalminimum.xyz` | Lowest-energy structure found |
| `ethanol.finalensemble.globaliter.*.xyz` | Conformer ensemble (see ORCA manual §6.4.3) |
| `ethanol.goat.*.out` | Per-worker logs if `KEEPWORKERDATA TRUE` |

Pass criteria for a smoke run:

- ORCA completes without client connection errors
- `ethanol.globalminimum.xyz` is written
- Final ensemble lists at least one conformer within the energy window

## Input notes

- `! ExtOpt GOAT PAL4` — required external PES + parallel workers (same pattern as [OET GOAT](https://www.faccts.de/docs/orca/6.1/tutorials/workflows/extopt.html))
- `%method ProgExt` + `Ext_Params "-b host:port"` — MMML client (identical to `water_opt.inp`)
- `%pal nprocs 4` + `%goat NWORKERS 4` — one worker per core
- `%goat MAXITER 32` — reduced from default 128 for smoke runs
- `%geom EnforceStrictConvergence false` — GOAT sets strict convergence by default; ML potentials can be noisy at `TolE` floors

## Production settings

For real conformer work on larger molecules:

```text
! ExtOpt GOAT
! PAL16

%pal
  nprocs 16
end

%goat
  NWORKERS 16
  KEEPWORKERDATA TRUE
end
```

Use a checkpoint trained on your chemistry. The bundled `neutral_best_forces` JSON is for
**plumbing tests only**, not benchmark-quality conformer energies.

## Unit tests (no ORCA)

```bash
pytest tests/unit/test_orca_external.py tests/unit/test_orca_external_server.py -q
```
