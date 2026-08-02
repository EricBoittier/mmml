# Bayesian design of compact PES datasets

The purpose of PES dataset design is not to accumulate the largest possible
number of Cartesian geometries. It is to cover the local environments that a
message-passing model sees, while keeping the reference set physical,
non-redundant, and learnable.

`mmml pes-design` performs this compression on an NPZ candidate pool. Its
default representation is deliberately cheap: element-pair RDF channels plus
hashed CGenFF-type pair spectra. Optional SOAP descriptors add angular and
many-body resolution when DScribe is installed.

```bash
mmml pes-design \
  --input candidate_pool.npz \
  --output selected_20000.npz \
  --n-select 20000 \
  --descriptor combined \
  --cutoff 6 \
  --temperatures 300,600,1200 \
  --min-distance 0.75 \
  --max-relative-energy 10 \
  --seed 42
```

The output retains the source NPZ fields and adds
`pes_design_source_index` and `pes_design_score`. The companion report
directory contains `report.json`, a descriptor-space plot, a coverage CDF, and
an RDF-spectrum comparison against an equally sized physically weighted random
sample.

## The four intermolecular regions

Each relevant atom-type pair should be represented across four distinct
regions. These regions have different purposes and should not receive equal
sampling density.

| Region | What it constrains | Recommended sampling |
|---|---|---|
| **Repulsive wall** | Excluded volume, large stabilizing forces, MD crash prevention | Deliberate but sparse; reject the extreme, non-convergent tail |
| **Well and curvature** | Equilibrium structure, density, RDF peaks, binding thermodynamics | Densest coverage, including radial and angular perturbations around the minimum |
| **Shoulder / transition** | Association, rearrangement, entry into and escape from a contact | Moderate coverage over several partners and orientations |
| **Long-range / cutoff** | Correct asymptote and smooth ML/MM truncation | Sparse coverage, with dedicated points on both sides of the cutoff policy |

For LJ-oriented work, a useful coordinate is the contact distance normalized by
the initial force-field size,

\[
\rho_{ij}=r_{ij}/\sigma_{ij}^{(0)}.
\]

This makes chemically different pair types comparable. Approximate starting
windows are `rho < 0.80` (normally reject), `0.80–0.95` (repulsive),
`0.95–1.25` (well and shoulder), and `1.25–2.0` (tail). They are guidelines,
not universal physical constants: calibrate them against cheap energy and force
scans for the chemistry in question.

Use nearest atomic contacts and type-pair distances, not only center-of-mass
distance. Two dimers with the same COM separation can present entirely
different local environments to PhysNet.

## Candidate distribution: physical, but deliberately broader than equilibrium

A useful candidate distribution is a mixture rather than one canonical MD run:

\[
q(x)=\sum_T w_T p_T(x)
    +\sum_k w_k p(x\mid d_k)
    +w_c p_{\mathrm{cluster}}(x).
\]

The terms represent several-temperature Boltzmann samples, biased pair-distance
windows, and clusters or condensed-phase environments. Suitable sources include:

- thermal normal-mode monomers and short MD trajectories;
- randomized dimer directions and orientations;
- restrained windows in the four distance regions;
- trimers and small clusters, which expose non-additive local environments;
- condensed-phase snapshots, especially around rare type pairs.

Cheap MM, GFN2-xTB, or an existing ML ensemble can screen a pool of
`10^5–10^6` candidates before expensive labeling. Reject non-finite structures,
impossible contacts, unconverged cheap calculations, and configurations above a
chosen energy or force ceiling. Keep a controlled repulsive subset—usually
about 5–10%—rather than allowing huge, thermodynamically irrelevant forces to
dominate maximum-likelihood training.

## Bayesian and maximum-likelihood roles

Maximum likelihood fits the geometries already observed. It does not determine
which geometries are worth observing. Dataset construction is therefore posed
as Bayesian optimal experimental design.

After descriptor standardization and PCA, `pes-design` estimates a ridge-prior
posterior variance (a leverage score) for every candidate. It then performs
leverage-weighted mini-batch clustering. This approximates a batch
D-optimal design:

\[
\mathcal A(B)=
\log\det\left(I+\sigma^{-2}\Phi_B^T W_B\Phi_B\right)
+\lambda_u U(B)+\lambda_n N(B).
\]

The log determinant rewards parameter-information gain; `W` carries physical
or multi-temperature Boltzmann weights; uncertainty and novelty reward poorly
known and non-redundant environments. The current CLI implements the
descriptor/leverage/D-optimal part. A model ensemble can later supply `U(B)` in
an iterative active-learning loop.

For energy-and-force models, force disagreement is particularly valuable.
Structures with similar energies may constrain very different coordinate
derivatives. Production acquisition should therefore consider both energy and
force variance whenever an ensemble is available.

## Recommended production loop

1. Generate a very large, cheap candidate pool spanning the four regions.
2. Apply physical contact, energy, force, and convergence filters.
3. Compute pair-RDF/type-pair descriptors; add SOAP where angular resolution is
   worth its cost.
4. Run `mmml pes-design` for a compact initial D-optimal batch.
5. Label that batch at the chosen consistent reference level.
6. Train three to five independently seeded PhysNet models.
7. Sample biased MD and distance windows with the ensemble.
8. Add candidates with high force disagreement and descriptor-space leverage.
9. Repeat until the physically weighted coverage and uncertainty stop improving.

A sensible starting target is `10,000–20,000` diverse dimers plus several
thousand trimers, clusters, and condensed-phase local environments. After
compression, `15,000–30,000` informative reference labels can cover more useful
PES space than hundreds of thousands of nearly duplicate scan structures.

## What “better coverage” means

Every `pes-design` run compares the selected batch with an equally sized random
baseline. Inspect all of the following:

- lower mean and 95th-percentile nearest-selected distance in descriptor space;
- larger D-optimal log determinant;
- selected and candidate RDF spectra that agree globally;
- visible coverage of sparse PCA regions without excessive outlier chasing;
- retained representation in all four distance regions and every trainable
  atom type;
- downstream force error, dimer curves, liquid density, and RDF peaks.

Descriptor coverage is a selection diagnostic, not proof of PES accuracy. The
final validation must use properties outside the selection objective.

## Important limitations

- Hashed CGenFF type-pair channels can collide. Increase
  `--type-hash-bins`, or use explicit pair channels for a small chemistry set.
- An averaged SOAP vector can hide rare atomic environments. For broad
  chemistry, validate atomic SOAP coverage as well as structure averages.
- Boltzmann weights require comparable energy references within each
  composition. Do not compare raw total energies across unrelated
  stoichiometries.
- A dimer-only set cannot represent every condensed-phase many-body
  environment. Include clusters and liquid snapshots before calling the set
  production-ready.
- Rare LJ types should stay fixed at scale 1 unless the selected set contains
  repeated partners and distance-region coverage for them.

See also [trainable hybrid MM LJ scales](hybrid-mm-lj-scales.md) and the
[LJ-scales numbered ladder](https://github.com/EricBoittier/mmml/tree/main/examples/lj_scales).
