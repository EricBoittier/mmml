# MD & campaigns

Running dynamics — pure MM, pure ML, or hybrid ML/MM — across the ASE, JAX-MD,
and PyCHARMM backends. `md-system` is the single entry point; a *campaign* is
one YAML describing many runs.

## Happy path

```bash
mmml env                                 # resolved checkpoints + CHARMM paths
mmml configure                           # interactive YAML, or hand-edit
mmml md-system --setup pbc_npt --composition MEOH:5,TIP3:5 --temperature 300
```

Scaling the same config out to a sweep:

```bash
mmml md-system --config campaign.yaml --run-all
```

On a GPU node, warm the JIT cache once before the real run:

```bash
mmml warmup-mlpot-jax --checkpoint "$MMML_CKPT" --n-monomers 20
mmml health-check --require-gpu --live
```

## What's here

**Tutorial** — [Tri-alanine water box](../trialanine-water-box.md) is the
fullest worked path: build, solvate, equilibrate, run.

**How-to**

- [md-system YAML configs](../md-system-configs.md) — single runs, campaigns,
  and condensed-phase builders. Start here for the config schema.
- [Protein force fields](../protein-force-fields.md) — CHARMM36 + jax-md peptides.
- [Species-aware ML/MM interaction policies](../md-interaction-policies.md) —
  deciding which species talk to which potential.
- [Batched umbrella sampling](../umbrella.md) and
  [Trajectory free-energy surfaces](../trajectory-free-energy-surfaces.md) —
  biased sampling and what to do with the trajectories.
- [Cross-backend handoff](../handoff.md) — moving a live system between ASE,
  JAX-MD, and PyCHARMM without losing state.
- [Remote MD runs + live streaming](../remote-md-streaming.md) — watching a
  cluster run from your laptop.

**Worked examples** — five complete studies, from a solvated peptide to a
reactive Menshutkin free-energy surface.

**Commands** — `md-system`, `md-embedding`, `umbrella-sample`, `health-check`,
and the MPI helpers.

## Where this leads

Runs that need a trained potential come from
[Training & sampling](training-sampling.md); how that potential is *assembled*
is [Hybrid ML/MM potentials](hybrid-potentials.md). For partitions, MPI, and
launchers see [Environment & clusters](environment-clusters.md).
