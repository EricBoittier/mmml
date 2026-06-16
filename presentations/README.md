# Presentations

## COSMO Lab interview (EPFL)

Terminal-style deck on **MMML** — **3 slides**.

```bash
typst compile presentations/cosmo_epfl_mmml.typ
typst watch presentations/cosmo_epfl_mmml.typ
```

Output: `presentations/cosmo_epfl_mmml.pdf`

Requires [Typst](https://typst.app/) and network on first compile (`polylux`, `sicons`).

### Slides

1. **Title & links** — MMML, stack icons, git/docs/tutorial/CI link rows, contributor credits
2. **MLIP workflow** — gpu4pyscf data gen → train → MD/hybrid CHARMM → spectra/conformers
3. **Engineering & MetaTensor** — CI/CD, tests, metatomic PR #262 / issue #228, MMML `orca-server`

### Speaker notes (~45–60 s each)

1. End-to-end MLIP toolkit; point to mmml, tutorial repo, docs, CI; thank contributors.
2. Full practitioner path: GPU QC labelling, training, MD (hybrid is one path), IR/VCD/Raman, GOAT.
3. Shipped with CI and 65+ tests; same ORCA ExtOpt pattern in MMML (JAX) and metatomic (PyTorch).
