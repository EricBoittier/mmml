# Structure & boxes

Everything that produces a **starting structure**: a single residue from CGenFF,
a packed periodic box, or a symmetry-generated crystal. These are the inputs
every MD and QM workflow downstream expects.

## Happy path

```bash
mmml make-res --list-residues            # what topologies are available
mmml make-res --res CYBZ                 # residue -> PDB/PSF/topology
mmml make-box --res CYBZ --n 50 --box-size 25.0
```

For a density-certified liquid box instead of a naive pack:

```bash
mmml liquid-box --composition DCM:206 --target-density-g-cm3 1.326 -o boxes/dcm206
```

## What's here

**Tutorial** — [Structure building](../cli/structure-building.md) walks
`make-res` → `make-box` → `build-crystal` end to end, with rendered structures.

**How-to**

- [Packmol placement](../packmol-placement.md) — the default composition builder,
  and how to control placement when the default is wrong.
- [Liquid box workflow](../liquid-box-workflow.md) — building and *certifying*
  periodic liquid boxes (MM only), including the density check.

**Commands** — `make-res`, `make-box`, `build-crystal`, `liquid-box`.

## Where this leads

A finished box is the input to [MD & campaigns](md-campaigns.md). If you are
building a box to train against rather than to simulate, go to
[QM & data](qm-data.md) instead.
