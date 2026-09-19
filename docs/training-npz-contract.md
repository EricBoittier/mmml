# Training NPZ contract

What `mmml physnet-train` actually consumes. This is **not** the same as the
Molpro/PySCF **ingest** schema used by `mmml validate` and
[`mmml/data/npz_schema.py`](https://github.com/EricBoittier/mmml/blob/main/mmml/data/npz_schema.py).

Units tables: [Units summary](UNITS_SUMMARY.md). Hybrid extras:
[Preparing hybrid ML/MM datasets](hybrid-mm-dataset-preparation.md).
Teacher-labeled acetone NPZ: [`pet-physnet-distill`](cli/commands/pet-physnet-distill.md).

## Two schemas

| | Ingest (`mmml validate`, GUI, `xml2npz`) | Train (`physnet-train`, distill export) |
|---|---|---|
| `R` | Å | Å |
| `E` | Hartree (PySCF/Molpro default) | **eV** |
| `F` | Hartree/Bohr, force = −∇E | **eV/Å**, force = −∇E |
| `D` / `Dxyz` | Debye | **e·Å** |
| Record | optional | `_mmml_units` or `units_manifest.json` |

`fix-and-split` **defaults** convert ingest → train. If the file is already eV
(SPICE-α, `pet-physnet-distill`, rMD17 after conversion), split only:

```bash
mmml fix-and-split --efd data.npz -o ./splits --preserve-units
```

`--conversion` on `physnet-train` scales **printed MAE only**. It does not
rewrite `E`/`F`/`D`.

## Required keys

| Key | Shape | Notes |
|---|---|---|
| `R` | `(n, pad, 3)` | Å |
| `Z` | `(n, pad)` | atomic numbers; `0` = padding |
| `N` | `(n,)` | real atom count (`≤ pad`) |
| `E` | `(n,)` | eV; **meaning** is separate (below) |
| `F` | `(n, pad, 3)` | eV/Å; must be **force**, not ∂E/∂R |

Optional: `D` `(n, 3)` e·Å (omit or set `dipole_weight: 0` if missing).
`polar` is stored by some exporters and **not** trained by PhysNet.

Pad rule: right-pad with `Z=0`, zero `R`/`F`. `physnet-train` auto-detects
`num_atoms` from `R.shape[1]`.

## Energy meaning (not a unit)

| Definition | What `E` / `F` are | Typical source |
|---|---|---|
| **Total** | DFT total (or formation) energy and forces | PySCF, SPICE-α `dft_total_energy` |
| **Interaction** | monomer `E − E_eq`; dimer `E(AB) − E(A) − E(B)` and matching forces | `pet-physnet-distill` default |
| **Hybrid train** | same QM labels **plus** CGenFF fields; trainer assembles `(1−s)(E_A+E_B)+s E_AB+E_MM` | `prepare-mm-dataset` then `--hybrid-mm` |

Do **not** bake `ml_switch_scale` into labels. Do **not** mix total and
interaction rows in one `E` array. For total energies use
`--subtract-atom-energies` so the net does not eat the ~10²–10³ eV offset.

There is no required `kind` / `E_int` key today. Distill writes them as
metadata (`kind` 0=monomer, 1=dimer). Treat missing `E_int` as total-E.

## Gradient vs force

`F` must be −∇E. PySCF evaluate already stores forces. External HDF5 often
stores **gradients** (`dft_total_gradient`, Psi4 `dft total gradient`):

```bash
mmml fix-and-split --efd raw.npz -o ./out --flip-forces   # ingest units
# already eV: negate in the converter, then --preserve-units
```

## Hybrid extras

`--hybrid-mm` needs two covalent monomers per frame plus
`mol_id`, `cgenff_type_idx`, `cgenff_charge`, `cgenff_master_sigmas`,
`cgenff_master_epsilons`. Raw QM NPZs do not have these — run
`mmml prepare-mm-dataset`. Droplets / water shells / 30-mers are **not**
dimers and **not** PBC boxes unless a lattice is present.

## HDF5

`mmml.models.physnetjax.physnetjax.data.read_h5` loads groups named `mol_*`
with `positions`, `formation_energy`, `total_forces`. SPICE-style files
(one group per molecule, `conformations` of shape `(M, N, 3)`) match
**zero** groups — convert first ([SPICE-α](spice-alpha.md)).

## Forbidden mixes

- Hartree `E` with eV `F` (or the reverse)
- Debye `D` while training as e·Å (`--conversion` will not fix this)
- `--use-pbc` / PME on isolated clusters with no `cell`
- `--hybrid-mm` without CGenFF fields
- Passing a metatomic `.pt` as `physnet-train --teacher-checkpoint`
  (labels are already in the NPZ; see `pet-physnet-distill`)

## Check before train

1. `units_manifest.json` or `_mmml_units` says `ev` / `ev_angstrom` / `e_angstrom`.
2. Bond lengths in `R` are ~0.8–2.5 Å.
3. `|E|` for a small organic is hundreds of eV (total) or ≲ 1 eV (interaction),
   not ~40 (Hartree mistaken for eV) and not 10⁴ after a double conversion.
4. `Z.max()` ≤ `--max-atomic-number` (Iodine is 53; the example yaml uses 35).
