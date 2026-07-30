# ACS-style LaTeX: hybrid training and MD Methods

JCTC-targeted Methods draft (`achemso`) explaining hybrid ML/MM training
(`physnet-train`) and condensed-phase MD (`md-system`), with TikZ figures for
COM handoff, training pipeline, MD campaign stages, and charge embeddings.

| File | Role |
|------|------|
| `main.tex` | Article body (achemso / JCTC) |
| `refs.bib` | Bibliography |
| `main.pdf` | Author-generated PDF (build artifact; may be gitignored) |
| `Makefile` | `make pdf` via tectonic |

## Build

```bash
# tectonic from conda-forge (recommended)
micromamba install -c conda-forge tectonic
cd docs/manuscripts/condensed-phase-hybrid-mlmm/latex
make pdf
# or: tectonic -X compile main.tex
```

ACS Paragon Plus expects a ZIP of TeX sources plus an author PDF; see
[Preparing and Submitting Manuscripts Using LaTeX](https://pubs.acs.org/page/4authors/submission/tex.html).

## Figures (TikZ)

1. **Figure 1** — COM handoff / \(s_\mathrm{ML}\), \(s_\mathrm{MM}\)
2. **Figure 2** — Training pipeline → JSON checkpoint → MD
3. **Figure 3** — Phase A/B MD campaign + backends
4. **Figure 4** — Mechanical (`fixed`) vs electrostatic (`q0`) embeddings

Companion markdown: `../methods.md`, `../results.md`.
