# Training & sampling

Fitting ML potentials — PhysNet, EF (electric field), KerNN, DCMNet — and the
sampling methods that use them: reaction paths, diffusion Monte Carlo, and
active learning.

```bash
mmml fix-and-split --efd data.npz --output-dir ./splits
mmml physnet-train --config train.yaml
mmml physnet-evaluate --checkpoint ckpts/run --test splits/test.npz
```

To distill a small PhysNet from a metatomic PET-MAD teacher on acetone
(dataset + synthetic noise/scans):

```bash
mmml pet-physnet-distill --checkpoint pet-mad.pt --out-dir ./acetone_pet_distill
mmml physnet-train --config ./acetone_pet_distill/physnet-train.yaml
```

See [Metatomic in MMML](../metatomic.md).

Then sample with the trained model:

```bash
mmml neb --config examples/m/yaml/neb.yaml --overwrite
mmml dmc --natm 20 --nwalker 512 --stepsize 5e-4 --nstep 5000 --eqstep 1000 \
  --alpha 1200.0 --checkpoint "$MMML_CKPT" \
  --input mmml/generate/dmc/examples/acetone_dmc.extxyz
```

## What's here

**How-to**

- [Bayesian design of compact PES datasets](../bayesian-pes-design.md) — the
  four intermolecular regions, physical candidate generation, RDF/SOAP
  compression, D-optimal acquisition, and `mmml pes-design` validation.
- [Nudged elastic band (NEB)](../neb.md) — PhysNet minimum-energy paths and
  barrier sampling.
- [Diffusion Monte Carlo (DMC)](../dmc.md) — batched PhysNetJax walkers, with a
  longer production example and the output file layout.

**Commands** — `physnet-train` / `-evaluate` / `-md`, `pet-physnet-distill`,
the `efield-*` and `kernnn-*` families, `neb`, `dmc`, `active-learning`,
`pes-design`, `kernel-fit`, `train-joint`, and the checkpoint utilities
(`orbax-to-json`, `extract-checkpoint-metrics`, `diagnose-lc-outliers`).

## Before you trust a checkpoint

`mmml physnet-evaluate` reports test-set error, which is necessary but not
sufficient — a model can fit its split and still be unusable in MD. The
diagnostics that catch that live elsewhere: `mmml mode-check` (finite
differences, vibrations, kick tests) in [QM & data](qm-data.md), and the
parity reports in [Hybrid ML/MM potentials](hybrid-potentials.md).

## Where this leads

A checkpoint plus a region policy makes a hybrid potential
([Hybrid ML/MM potentials](hybrid-potentials.md)), which `md-system` then runs
([MD & campaigns](md-campaigns.md)).
