# karml second cut

Planning note for the snapshot already pushed to
[EricBoittier/karml](https://github.com/EricBoittier/karml) (`87a7c51`).
That commit dropped bulky files. This note is the code and docs cut on top of
it. Line counts are from that snapshot (7 Oct 2026).

## Shape

PhysNet, DCMNet, efield, and KerNN stay. Training calls those same modules.
Dynamics runs through PyCHARMM MLpot or JAX-MD. ORCA keeps its server.
Metatomic checkpoints load into the MLpot callback. ASE, OpenMM, and the
satellite apps go. Tutorial pages get rewritten. Docstrings stay.

A checkpoint written by training is what both engines evaluate. Outside models
such as PET enter through the metatomic loader into MLpot. They do not get a
separate MD command.

## Stays

| Piece | Size | Role |
| --- | ---: | --- |
| `karml/models` (PhysNet, DCMNet, efield, KerNN, MBD, hybrid MM) | ~45k lines | The implementations. Training and inference share them. |
| `mlpot/` and `md-system` on PyCHARMM | 67k lines | The CHARMM driver. |
| JAX-MD runner, suite, and `jaxmdInterface` | ~8k lines | Second engine. Same models. |
| ORCA server, client, and external wrapper | ~1.1k lines | QC hook. |
| Metatomic loader inside the MLpot callback | inside `mlpot/` | Outside checkpoints, same callback. |
| Docstrings on the models and both engines | — | Stay with the code. |
| Tests of the models, MLpot, JAX-MD, and ORCA | most of `tests/unit` | Stay until a test is only for a removed engine. |

## Goes

| Piece | Size | Why |
| --- | ---: | --- |
| GUI | 5.8k lines | Satellite app. |
| ASE runner and `md_pbc_suite/ase.py` | 3.3k lines | Third MD engine. |
| MCP | 3.3k lines | Satellite app. |
| OpenMM, Blender, OpenFE | 1.5k lines | Extra interfaces. |
| `aseInterface` | 1.1k lines | Goes with the ASE engine. |
| `metatomic-pbc-md` | one command | CHARMM-free side program. PET loads into MLpot instead. |
| `docs/cli` generated reference | 75 pages, 8.1k lines | Grown from the command list. |
| Lab handoffs under `docs/` | 12 pages, 2.7k lines | Campaign notes. |
| Tutorial pages | 11 pages, 3.9k lines | Rewrite later. Do not patch the old ones. |
| Tests that only open `examples/` | 43 files | The examples are already gone. |

About 15k lines of interfaces and apps, plus the tutorial and generated docs.
The model zoo is not part of this removal.

## Docs to write later

A few pages, not the old set:

1. Install, including the CHARMM tarball.
2. Train one in-repo model.
3. Run `md-system` with that checkpoint on PyCHARMM and on JAX-MD.
4. Point ORCA at the server.
5. Load a metatomic checkpoint into the same MLpot callback.

## Already out of the snapshot

Not part of this cut. Left behind when `87a7c51` was pushed:

- `examples/` (347 MB), to be rewritten later
- extracted `setup/charmm/` (333 MB); `setup/charmm.tar.xz` stayed
- figure assets (95 MB)
- regenerable checkpoints, including `pet-omol-s-v1.0.0.pt` (137 MB)
- `devtools/scratch/`
