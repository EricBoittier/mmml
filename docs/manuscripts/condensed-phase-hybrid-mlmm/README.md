# Condensed-Phase Simulations Using Hybrid ML/MM Energy Functions

**Working title.** Meuwly group · July 2026  
**Repo status.** Draft outline + Methods/Results stubs tied to Snakemake workflows under `workflows/`.  
**Branch intent.** Paper figures/tables should be regenerable from those workflows; this folder is the narrative glue.

| Doc | Role |
|-----|------|
| [outline.md](outline.md) | Recommended sections, maturity, next steps |
| [methods.md](methods.md) | Methods draft (hybrid energy, PBC, solvers, protocols) |
| [latex/](latex/) | **ACS/JCTC LaTeX** Methods (`achemso`) + TikZ figures; `make pdf` |
| [results.md](results.md) | Results draft (placeholders → workflow outputs) |
| [workflow-map.md](workflow-map.md) | Figure/table ↔ Snakemake workflow matrix |

Related engineering docs (do not duplicate here): [hybrid potential regions](../hybrid-potential-regions.md), [liquid box workflow](../liquid-box-workflow.md), [long-range solvers](../long-range-solver-tutorial.md), [jax-mm-spoof CHARMM parity](../jax-mm-spoof-charmm-parity.md), [validation campaign](../../workflows/validation_campaign/README.md).
