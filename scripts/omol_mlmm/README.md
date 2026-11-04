# OMol25 → ML/MM training labels (CGenFF types, residues, charges, LJ)

Pipeline used for the universal ML/MM fine-tune (Nov 2026). Library code lives in `mmml/fftype/`; the scripts here
import `cgenff_envtyper`, `envcharge`, `ljtab` through the small shim modules in each folder. Slurm/sbatch files and some
paths are the pc-studix ones (`/mmhome/boittier/...`) and need adapting elsewhere.

**Reproducibility:** all numbers and `mmml/fftype/data/omol_auto.rtf` were built with the 938-RESI CGenFF in
`mmml@main:setup/charmm/toppar`. Set `MMML_TOPPAR=<that dir>` to reproduce (leave-one-RESI-out 95.1 %). The toppar in
this branch is a newer CGenFF (2331 RESIs): LORO 94.6 % on that broader set; coverage not yet re-measured.

| Step | Script | Output |
|---|---|---|
| Stream full OMol25 tarball, map chemistry + select ML/MM structures | `stream_select.py`, `map_select.py`, `select_nfs.py` | `map_full/`, `select_full/*_sel.npz` |
| Typer validation (leave one RESI out) | `loro.py` | 95.1 % exact |
| Gap census / families on OMol25 4M | `gap_census.py`, `families.py`, `type_counts.py` | untyped 0.25 % with the auto library |
| ML/MM cache (1M, typer v3.1 + auto residues, CHARMM LJ) | `build_full_cache.py` | `caches/omol_mlmm_full1M_cgenff` |
| Frequent molecules → residues | `residues/frag_mols.py`, `build_residues.py`, `check_residues.py`, `retype_residues.py` | 258 RESI, re-typing 100 % self-consistent |
| Charges | `qm/resp.py` (HF/6-31G* ESP), `fit_hybrid.py` (CV), `novel_rule.py`, `apply_charges.py` | transfer+ESP hybrid / analogue-restrained ESP / RESP for hypervalent anions |
| LJ for atoms without a CHARMM type | `ljfit/extract_dimers.py`, `dft_dimers.py` (wB97M-V/def2-TZVPD, gpu4pyscf), `fit_lj.py`, `apply_assignment.py` | existing types assigned by dimer fit (RMSE 7.0 vs 7.6 kcal/mol element default) |

Typer: `CGenFFTyper(extra=OMOL_AUTO_RTF)`; levels 2/1 = r2/r1 signature, 3 = H from parent type, 4/5 = neighbour
back-off, 0/6 = element level, 7 = polar-H rule, 10+ = other CHARMM library, 20+ = OMol25 auto library, -2 = untyped.
gpu4pyscf on CUDA-12.4 drivers needs CUDA 12.4 pip libraries, unversioned `.so` symlinks and a `CUDA_PATH` tree (see `ljfit/dft.sbatch`).
