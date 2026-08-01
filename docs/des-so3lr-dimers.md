# The SO3LR / DES dimer set and hybrid ML/MM

Where the DES dimer data lives, what chemistry is in it, and exactly which
**CGenFF Lennard-Jones parameters** the hybrid ML/MM machinery
([`prepare-mm-dataset`](hybrid-mm-dataset-preparation.md) →
[trainable LJ scales](hybrid-mm-lj-scales.md)) can reach with it.

Everything on this page is measured, not estimated: a streaming pass over the
full 370,956-frame file
([`scripts/scan_des_chemical_space.py`](https://github.com/EricBoittier/mmml/blob/main/scripts/scan_des_chemical_space.py))
ran the production assignment
([`mmml/data/cgenff_dataset.py`](https://github.com/EricBoittier/mmml/blob/main/mmml/data/cgenff_dataset.py))
on a 1-in-20 sample, and the figures and tables are generated from that scan by
[`scripts/gen_docs_des_chemspace_figures.py`](https://github.com/EricBoittier/mmml/blob/main/scripts/gen_docs_des_chemspace_figures.py).

!!! note "Related"
    [Preparing hybrid ML/MM datasets](hybrid-mm-dataset-preparation.md) ·
    [Trainable hybrid MM LJ scales](hybrid-mm-lj-scales.md) ·
    [Hybrid MM charges](hybrid-mm-charges.md) ·
    [DES dimer pair scans workflow](https://github.com/EricBoittier/mmml/tree/main/workflows/des_dimer_pair_scans)

---

## Where the data is

Everything is on **scicore** (`/scicore/home/meuwly/boitti0000`). None of it is
on `pcstudix`, and none of it is in the repo.

| Path | Size | What it is |
|---|---:|---|
| `~/qcell/qcell_dimers.h5` | 5.5 GB | **The DES dimer set.** Per-structure groups `mol_000001…`, plus a `metadata` group |
| `~/qcell/qcell_xyz/qcell_dimers.xyz` | 748 MB | Same frames as extxyz — the convenient form for streaming/surveying |
| `~/qcell/.h5_cache/qcell_dimers_66998dd44fdb879b` | — | Orbax cache written by `prepare_h5_datasets` |
| `~/trainDES/train.py` | — | The PhysNet run that consumed it (`train_size=270_000`, `natoms=34`, `charge_filter=0.0`) |
| [`examples/ckpts_json/DESdimers_params.json`](https://github.com/EricBoittier/mmml/blob/main/examples/ckpts_json/DESdimers_params.json) | 696 KB | The resulting checkpoint — **in the repo**, and the reference model for [`workflows/des_dimer_pair_scans`](https://github.com/EricBoittier/mmml/tree/main/workflows/des_dimer_pair_scans) |

Sibling sets in the same directory (`qcell_ions_water.h5`, `qcell_sugars.h5`,
`qcell_nucleic_acids.h5`, `qcell_lipids.h5.swp`) are *not* dimers and are out of
scope here.

**Level of theory** is recorded in the file itself
(`metadata/fhi_aims_settings`): FHI-aims, **PBE0** with `many_body_dispersion_nl`,
`atomic_zora scalar` relativistic, tight SCF — i.e. **PBE0+MBD**, the SO3LR
reference level. `metadata/free_atom_energy` carries free-atom energies for
Z = 1…86, so `formation_energy` is already referenced.

Per-structure datasets go well beyond `R/Z/E/F`: `total_forces`,
`formation_energy`, `dipole`, `quadrupole`, `hirshfeld`-style `c6_ratios` /
`a0_ratios`, `homo_energy` / `lumo_energy` / `homo_lumo_gap`, `ks_eigenvalues`,
and a full energy decomposition (`electrostatic_energy`, `hf_energy`,
`vdw_energy`, `total_xc_energy`, …).

!!! warning "The `~/data/so3lr*` files are a different thing"
    On **pcstudix**, `~/data/so3lr_train.extxyz` (9.5 GB) and `~/data/so3lr_test/`
    (gems, md22, qm7x, TorsionNet500) are the SO3LR *training and benchmark* sets.
    They are not the DES dimers and contain no subset labels — you cannot slice
    the dimers back out of them.

---

## What is in it

<figure markdown>
![DES dimer chemical space](images/des-so3lr-dimers/chemical_space.png)
<figcaption>370,956 frames · 1,333 distinct monomers · 3,672 distinct unordered pairs · 20 elements · 2–34 atoms per frame.</figcaption>
</figure>

Reading the panels:

- **(a)** Water is the hub: it is 27% of all monomer occurrences and outruns the
  next most common monomer (CH₄) roughly 15-fold. Colour marks whether the
  hybrid ML/MM path can type the monomer — the biggest gaps left are H₂S and
  CH₂O.
- **(b)** The sampling is deliberately water-centred plus a homodimer diagonal;
  grey cells are pairs that were never generated. This is DES370K-style
  coverage, not a dense all-pairs grid.
- **(c)** 20 elements, including noble gases (He, Ne, Ar, Kr, Xe) and bare ions
  (Na, K, Li, Ca, Mg, F, Cl, Br, I) — none of which CGenFF alone can type. All
  five noble gases and six of the ions are covered by the merged stream files
  below; F⁻, Br⁻ and I⁻ remain uncovered.
- **(d)** **34 atoms is the hard maximum**, which is where `natoms=34` in
  `~/trainDES/train.py` comes from. The spike at 4 atoms is the
  noble-gas/diatomic and small-ion population.

5.2% of frames (19,172) resolve to a **single** covalent component rather than
two — contact ion pairs and close-contact frames where the covalent-radius graph
merges the monomers. `prepare-mm-dataset` drops these.

---

## The overlap with CGenFF

Running the real assignment on a 1-in-20 sample (18,548 frames):

| | frames | share |
|---|---:|---:|
| **Typeable — usable for hybrid ML/MM** | **7,595** | **40.9%** |
| Dropped: composition matched, topology did not | 5,206 | 28.1% |
| Dropped: no template for the composition | 4,218 | 22.7% |
| Dropped: not two covalent components | 946 | 5.1% |
| Dropped: not graph-isomorphic to the chosen template | 583 | 3.1% |

**94 of 1,333 monomer species** are typeable, and they form **1,188 distinct
`RESI` pairs**. Because the lookup is composition-keyed, each typeable formula
resolves to exactly one residue.

### Extending the reference beyond CGenFF

Bare CGenFF types only 32.9% of frames — it has no template for a monatomic ion
or a noble gas, so every ion–water and noble-gas dimer dropped. `load_reference`
merges additional CHARMM stream files listed in `DEF_EXTRA_TOPPAR`:

| Added | Residues | Coverage | New LJ types |
|---|---|---:|---:|
| *(CGenFF alone)* | — | 32.9% | 90 |
| [`toppar_water_ions.str`](https://github.com/EricBoittier/mmml/blob/main/mmml/data/charmm/toppar_water_ions.str) | `CLA` `SOD` `POT` `LIT` `CAL` `MG` | 35.8% | 96 |
| [`toppar_dum_noble_gases.str`](https://github.com/EricBoittier/mmml/blob/main/mmml/data/charmm/toppar_dum_noble_gases.str) | `HE1` `NE1` | 37.8% | 98 |
| [`toppar_noble_gases_literature.str`](https://github.com/EricBoittier/mmml/blob/main/mmml/data/charmm/toppar_noble_gases_literature.str) | `AR1` `KR1` `XE1` | **40.9%** | **101** |

All measured on the identical 18,548-frame sample — a 24% relative increase in
usable data. The merge is strictly additive: an atom type or `RESI` CGenFF
already defines is left alone and new compositions are appended behind the
existing candidates, so assignments that already worked are byte-identical
(verified over 3,000 DCM frames).

!!! warning "Ar, Kr and Xe are not CHARMM parameters"
    CHARMM ships no residue for them anywhere —
    `toppar_dum_noble_gases.str` covers only He and Ne. The third file carries
    **standard literature 12-6 parameters** (Ar σ 3.405 Å / ε 0.238 kcal/mol;
    Kr 3.600 / 0.340; Xe 4.100 / 0.439), converted to CHARMM's
    (ε, R<sub>min</sub>/2) convention and labelled as non-CHARMM in the file
    header. They were **not fitted alongside CGenFF** and their cross terms are
    unvalidated, so noble-gas results are provisional.

    The one consistency check available: σ and ε both rise monotonically across
    the combined series (He 2.637/0.021 < Ne 2.726/0.085 < Ar 3.405/0.238 <
    Kr 3.600/0.340 < Xe 4.100/0.439) despite spanning two sources. The
    alternative in-tree values, from BMS
    (`toppar/non_charmm/par_bms_dec03.inp`), fail that check — their argon ε is
    *smaller* than their neon ε, and their helium ε differs from CHARMM's by
    51× — which is why they are not used.

The two large failure modes are different problems:

- **No template (22.7%)** — the composition has no `RESI` at all. With the ions
  and all five noble gases merged, what remains is small molecules CGenFF does
  not cover (H₂S, CH₂O and H₂S₂ are the three biggest single contributors) plus
  the halides CHARMM has no ion residue for (F⁻, Br⁻, I⁻).
- **Topology mismatch (28.1%)** — a residue with the *same formula* exists, but
  the frame is a different isomer. `_template_to_geometry_permutation` catches
  this by graph isomorphism and the frame is dropped rather than silently
  mistyped. This is the failure mode working correctly, but it means the
  composition-first lookup is leaving usable frames on the table.

Largest untypeable monomers by occurrence. Cl⁻ and all five noble gases have
moved off this list:

| Monomer | occurrences | Monomer | occurrences |
|---|---:|---|---:|
| H₂S | 13,256 | C₃H₆S₂ | 4,613 |
| CH₂O | 13,040 | C₃H₈O₂ | 4,571 |
| H₂S₂ | 7,147 | C₂H₄O | 3,682 |
| C₆H₁₂ | 6,971 | F⁻ | 3,533 |
| C₃H₈S₂ | 6,135 | Br⁻ | 3,353 |
| C₄H₄N₂ | 4,868 | I⁻ | 3,333 |

What is left is now mostly **molecular**, not atomic: sulfur species (H₂S, H₂S₂,
the C₃ dithiols), formaldehyde, and cyclohexane. The only remaining atomic gap
is F⁻/Br⁻/I⁻ (10,219 occurrences), which `toppar_water_ions.str` does not carry
— chloride is its only halide. Those three would need hand-written residues on
the same footing as the literature noble gases.

---

## LJ parameters covered

<figure markdown>
![CGenFF LJ coverage](images/des-so3lr-dimers/lj_coverage.png)
<figcaption>101 of 185 nonbonded types are reachable from the DES dimer set, via 94 residues.</figcaption>
</figure>

This is the figure that matters for [trainable LJ
scales](hybrid-mm-lj-scales.md): a per-type σ/ε scale only receives a gradient
if some training frame contains an atom of that type. **101 of 185** types
qualify; the other 84 are inert no matter how long you train.

The distribution is steep. `HGA3` (aliphatic H) appears in 68% of typeable
frames and `CG331` (methyl C) in 64%, while 20 of the 101 types appear in fewer
than 50 sampled frames — **`NG2S1` appears in 7**, all from a single residue.
Those thin types will move under gradient descent without being meaningfully
constrained by data, which is exactly the regime where the
[σ/ε degeneracy](hybrid-mm-lj-scales.md) bites. Freeze them, exclude them, or
cut the residue list (below) rather than trusting a fitted value.

The eleven merged monatomic types split sharply. The noble gases and chloride
are **well sampled** — `AR` 252, `CLA` 231, `HE` 199, `NE` 181, `KR` 171,
`XE` 165 — and their residues rank 7th, 8th, 15th, 20th, 25th and 26th of 94,
so they comfortably survive the default cut. The metal cations do not: `POT`
103, `SOD` 93, `LIT` 83, `MG` 13, **`CAL` 9**. Treat `CAL` and `MG` as
unfittable at this sample size.

### Most-exercised types

| CGenFF type | σ (Å) | ε (kcal/mol) | sampled frames | residues |
|---|---:|---:|---:|---|
| `HGA3` | 2.3876 | 0.02400 | 5,011 | 54 |
| `CG331` | 3.6527 | 0.07800 | 4,770 | 51 |
| `HT` | 0.4000 | 0.04600 | 3,776 | TIP3 only |
| `OT` | 3.1506 | 0.15210 | 3,776 | TIP3 only |
| `HGP1` | 0.4000 | 0.04600 | 2,558 | 21 |
| `HGA2` | 2.3876 | 0.03500 | 2,262 | 32 |
| `CG321` | 3.5814 | 0.05600 | 1,906 | 28 |
| `OG311` | 3.1449 | 0.19210 | 1,417 | 9 |
| `HGR52` | 1.6036 | 0.04600 | 1,374 | 12 |
| `OG2D1` | 3.0291 | 0.12000 | 1,272 | 10 |

Note that `HT` and `HGP1` share σ = 0.4000 / ε = 0.04600 exactly — distinct
types, one point in the LJ plane. Scaling them independently is only
identifiable because they sit on different molecules.

**Full tables** (generated, 101 types and 94 residues):

- [`docs/images/des-so3lr-dimers/lj_types.md`](images/des-so3lr-dimers/lj_types.md)
  — every reachable type with σ, ε, frame count, and the residues that use it
- [`docs/images/des-so3lr-dimers/resi_coverage.md`](images/des-so3lr-dimers/resi_coverage.md)
  — every covered residue with its atom count and LJ types

### Residues covered, by sample count

All 94, ranked. Sampled frames are 1-in-20; multiply by 20 for the full set.
The **top 50** (above the rule) is the default training cut — see below.
**Bold** entries are residues merged from the extra stream files.

| # | RESI | smp | # | RESI | smp | # | RESI | smp | # | RESI | smp | # | RESI | smp |
|---:|---|---:|---:|---|---:|---:|---|---:|---:|---|---:|---:|---|---:|
| 1 | `TIP3` | 4,518 | 11 | `ACEM` | 205 | 21 | `ETHA` | 180 | 31 | `PYRL` | 123 | 41 | **`POT`** | 107 |
| 2 | `METH` | 662 | 12 | `IMIA` | 204 | 22 | `BUTA` | 177 | 32 | `PYR1` | 117 | 42 | `MGUA` | 106 |
| 3 | `AMM1` | 644 | 13 | `ACO` | 204 | 23 | `PHEN` | 176 | 33 | `PRO2` | 117 | 43 | `CPEN` | 106 |
| 4 | `ETHE` | 435 | 14 | `ETSH` | 204 | 24 | `DMDS` | 174 | 34 | `MAS` | 115 | 44 | `PRLD` | 104 |
| 5 | `FORH` | 344 | 15 | **`HE1`** | 200 | 25 | **`KR1`** | 171 | 35 | `EMS` | 115 | 45 | `THF` | 103 |
| 6 | `MEOH` | 340 | 16 | `MAM1` | 194 | 26 | **`XE1`** | 165 | 36 | `ACET` | 115 | 46 | `IMIM` | 103 |
| 7 | **`AR1`** | 252 | 17 | `MESH` | 192 | 27 | `NH4` | 162 | 37 | `TMAM` | 113 | 47 | **`SOD`** | 95 |
| 8 | **`CLA`** | 241 | 18 | `FORM` | 190 | 28 | `FORA` | 158 | 38 | `DETE` | 110 | 48 | `BTE1` | 87 |
| 9 | `ETOH` | 207 | 19 | `BENZ` | 187 | 29 | `PRPA` | 133 | 39 | `MIMI` | 109 | 49 | **`LIT`** | 83 |
| 10 | `ACEH` | 205 | 20 | **`NE1`** | 182 | 30 | `DMAM` | 129 | 40 | `MAMM` | 109 | 50 | `PRAM` | 81 |

*— top-50 cut —*

```
ETAC PENT HEXA SM073 DMA  THPS DITH DEDS TOLU INDO TRIT NC4  EAMM FETH
FLUB GUAN ACN  NC3  CLET DCM  PRPY DCLE DFET BRET DBRE EIMI MIND EBEN
EIND PROA EIMM EPHE SM158 MP_0 DMEP TFET GLYN MG   SM129 CAL  TCLE AANM
DME  DFB
```

All five noble gases and chloride land in the top 26 — they are ordinary,
well-sampled residues here, not tail entries. The metal cations are weaker:
`POT` (107), `SOD` (95) and `LIT` (83) scrape in, while `MG` (13) and `CAL` (9)
sit deep in the tail and are unfittable.

The 12-species panel in
[`workflows/des_dimer_pair_scans/config.yaml`](https://github.com/EricBoittier/mmml/blob/main/workflows/des_dimer_pair_scans/config.yaml)
(TIP3, DCM, ACO, ETOH, MEOH, ETHA, BENZ, BUTA, IBUT, PENT, NEOP, HEXA) is a
hand-picked subset of this list — every one of those except `IBUT` and `NEOP` is
confirmed present in the data above. That workflow's 78 pairs are a small corner
of the 919 RESI pairs the dataset actually supports.

---

## Preparing a training run

`LJ_DES=1` runs the ladder on this dataset. Step 12 builds the training NPZ:

```
qcell_dimers.h5
   │  scripts/des_h5_to_npz.py --pad 34        (12a)  units already eV / eV·Å
   ▼
des_dimers_raw.npz
   │  mmml prepare-mm-dataset                  (12b)  types, charges, res_name
   ▼
des_dimers_cgenff_all.npz
   │  scripts/filter_mm_dataset_by_residue.py  (12c)  --top 40
   ▼
des_dimers_cgenff_top40.npz  ──▶  05_train.sh  learn_mm_lj_scales
```

```bash
export LJ_DES=1
bash examples/lj_scales/12_des_dataset.sh
LJ_DES=1 LJ_DEVICE=gpu bash examples/lj_scales/05_train.sh
```

| Variable | Default | Meaning |
|---|---|---|
| `LJ_DES_H5` | `~/qcell/qcell_dimers.h5` | source HDF5 |
| `LJ_DES_PAD` | `34` | the measured DES maximum, **not** the ladder's usual 20 |
| `LJ_DES_TOP_RESIDUES` | `40` | residue cut (below) |
| `LJ_DES_ALL_CHARGES` | `0` | `1` admits net-charged (ion) dimers |

**Units need no conversion.** FHI-aims writes eV and eV/Å, which is what the
hybrid MM baseline and the existing lj_scales NPZs already use, and
`formation_energy` is referenced against `metadata/free_atom_energy`. It is the
direct analogue of `E` in `examples/dcm_mp2_psf_order.npz` (which ranges
−43…−21 eV for a 10-atom dimer).

### Why the residue cut

Yield against the cut, from the scan (sampled frames ×20 ≈ full):

| top-N residues | frames | ≈ full | % of typeable | LJ types | thinnest type |
|---:|---:|---:|---:|---:|---:|
| 20 | 3,049 | 61,000 | 46.0% | 32 | 2,600 |
| 25 | 3,616 | 72,300 | 54.5% | 41 | 2,240 |
| 30 | 4,009 | 80,200 | 60.4% | 47 | 2,120 |
| **40** | **4,750** | **95,000** | **71.6%** | **58** | **1,440** |
| 50 | 5,384 | 107,700 | 81.1% | 65 | 1,160 |
| 60 | 5,894 | 117,900 | 88.8% | 74 | 920 |
| 89 (all) | 6,635 | 132,700 | 100% | 96 | **140** |

**40 is the recommended default.** It keeps 72% of the usable frames and 58 LJ
types while holding every one of them above ~1,400 frames. Taking all 89
residues buys 38 more types but drops the sampling floor by an order of
magnitude — those types are exactly the ones that will drift.

A frame is kept only if **both** monomers are in the allowlist, which is why the
frame yield falls faster than the residue count.

### What to watch

1. **Water dominance.** `TIP3` is in 57% of typeable frames — more than five
   times the next residue. Every aggregate error metric on this set is
   substantially a water metric. Step 12 prints the TIP3 share so it cannot
   surprise you at eval time; consider subsampling or a per-pair breakdown.
2. **Ion frames are off by default.** The merged ion residues only occur in
   net-charged dimers, and step 12a filters to neutral (matching
   `~/trainDES/train.py`). `LJ_DES_ALL_CHARGES=1` admits them — a separate
   decision from having the templates, and one that changes what the Coulomb
   baseline has to represent.
3. **`lr_solver: mic`.** Training LJ requires it; under `ewald` the LJ term is
   removed from the hybrid energy and there is nothing to differentiate.

!!! warning "Dormant landmine in the SMILES fallback"
    `DES_SMILES_TO_RESI` in `mmml/data/cgenff_dataset.py` maps `[Ne]`, `[Ar]`,
    `[Kr]`, `[Xe]`, `[Ca+2]`, `[Li+]` and `[F-]` onto **`TIP3`**, and `C#N` onto
    `DMAM`. Those are placeholders, not chemistry. They are currently **inert** —
    `assign_frame_cgenff` calls `match_cgenff_template` without
    `canonical_smiles`, so only the composition path runs, and noble gases and
    ions correctly fail as "no template". Anything that starts passing SMILES
    would silently give argon water's σ/ε. Fix the map before enabling that
    path on this dataset, where noble gases and bare ions are ~5% of frames.

---

## Reproducing

The scan must run where the data is (scicore); the figures render anywhere.

```bash
python scripts/scan_des_chemical_space.py ~/qcell/qcell_xyz/qcell_dimers.xyz \
    --out artifacts/des_chemspace/qcell_dimers_scan.json \
    --stride 1 --cgenff-stride 20
```

```bash
python scripts/gen_docs_des_chemspace_figures.py \
    artifacts/des_chemspace/qcell_dimers_scan.json
```

The scan takes ~10 min single-threaded and writes a ~350 KB JSON; the CGenFF
assignment is ~100× the cost of the composition tally, hence the separate
`--cgenff-stride`. Lower it for a tighter coverage estimate, raise it for a
quick look.
