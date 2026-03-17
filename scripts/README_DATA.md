# Data Download and Setup

## Large files relocated from git-lfs

Some large data files have been removed from the repository to reduce LFS usage. Use the targets below to obtain them.

### `examples/other/testdata/grids_esp.npz` (~858 MB)

This file is **input** to `fix_and_split` for generating `preclassified_data/`. It is not needed if you already have `preclassified_data/` (e.g. from `git lfs pull`).

**To obtain:**

1. **If you have a copy elsewhere:** Copy or symlink it:
   ```bash
   cp /path/to/your/grids_esp.npz examples/other/testdata/
   ```

2. **Generate from source:** Run the CO2 data preparation pipeline (see `examples/other/co2/notes/TRAINING_GUIDE.md`). The `grids_esp.npz` is produced by earlier steps in the workflow.

3. **Skip:** If you only need `preclassified_data/` for training, run `git lfs pull` and use the pre-split data. No need for `grids_esp.npz`.

### Canonical data location

- **`examples/other/co2/preclassified_data/`** – Canonical source for `grids_esp_{train,valid,test}.npz`
- `dcmnet_train/` and `dcmnet_physnet_train/` use symlinks to this directory (no duplicates)
