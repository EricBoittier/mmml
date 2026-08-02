# ACS / JCTC LaTeX draft (hybrid ML/MM)

Single-file manuscript + in-file Supporting Information.

## Build

```bash
make pdf
# or: tectonic -X compile main.tex
```

## Layout

| Part | Content |
|------|---------|
| Main text | Hybrid energy, lean training/MD, embeddings, **Results** with filled spoof parity + placeholder NVE / scans / density / umbrella / ADUMB panels |
| SI (end of `main.tex`) | CLI flags, YAML sketches, free-energy commands, workflow provenance map |
| `refs.bib` | Expanded bibliography (ML potentials, QM/MM, Ewald/PME, umbrella/MBAR/ADUMB, …) |

Placeholder figures use TikZ frames labeled `(PLACEHOLDER)` until Snakemake workflows export PNGs under `artifacts/`.
