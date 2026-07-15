# md-embedding smoke results (aaa.ama)

Generated: 2026-07-02  
Artifacts: `artifacts/md_embedding/aaa_docs`

Reproduce:

```bash
export CHARMM_HOME=... CHARMM_LIB_DIR=... LD_LIBRARY_PATH=...
JAX_PLATFORMS=cpu uv run python scripts/collect_md_embedding_docs_results.py
```

## Training (PhysNet smoke)

| Quantity | Value |
|----------|-------|
| Epochs | 8 |
| Train frames | 12500 |
| Valid frames | 1249 |
| Best valid loss | 102.65313720703125 |
| Checkpoint JSON | `/home/ericb/mmml/artifacts/md_embedding/aaa_docs/aaa_smoke_params.json` |

![Training loss](../images/examples/md-embedding/training_loss.png)

## Validation metrics (PhysNet vs NPZ labels)

| Metric | Value |
|--------|-------|
| Energy MAE | 6.23443556918873 kcal/mol |
| Energy RMSE | 7.869658358803783 kcal/mol |
| Force MAE | 6.250538485566922 kcal/mol/Å |
| Force RMSE | 8.869026315270178 kcal/mol/Å |
| Eval samples | 6 |

![Energy/force parity](../images/examples/md-embedding/parity_plots.png)

## Build (CHARMM TRIA + TIP3)

| Quantity | Value |
|----------|-------|
| Peptide atoms (TRIA) | 42 |
| Waters | 10 |
| Box side (Å) | 28.0 |
| Bonded total (kcal/mol) | None |

![embedding_box.png](../images/examples/md-embedding/embedding_box.png)

![embedding_peptide.png](../images/examples/md-embedding/embedding_peptide.png)

![peptide_frame0.png](../images/examples/md-embedding/peptide_frame0.png)

See also: [md-embedding design](md-embedding-design.md), [aaa.ama workflow](aaa-ama-workflow.md).
