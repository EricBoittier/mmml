# Git LFS Setup for CO2 Examples

## Overview

Git LFS (Large File Storage) is configured to handle large data files in this repository, particularly the NPZ files containing molecular data.

## What's Tracked by LFS

The following file types are automatically tracked by Git LFS:

### Data Files
- `*.npz` - NumPy compressed archives (all NPZ files)
- `examples/co2/preclassified_data/*.npz` - Specifically the processed data files

### Model Files
- `*.pkl` - Python pickle files (model parameters)
- `*.pth` - PyTorch model files
- `*.h5` - HDF5 files
- `*.hdf5` - HDF5 files (alternate extension)
- `*.ckpt` - Checkpoint files

## File Sizes in Preclassified Data

```
energies_forces_dipoles_train.npz   897 KB
energies_forces_dipoles_valid.npz   114 KB
energies_forces_dipoles_test.npz    114 KB
grids_esp_train.npz                 305 MB  ← Large!
grids_esp_valid.npz                  39 MB
grids_esp_test.npz                   39 MB
split_indices.npz                    79 KB
────────────────────────────────────────
Total:                              ~383 MB
```

## Git LFS Configuration

The `.gitattributes` file contains:

```
*.npz filter=lfs diff=lfs merge=lfs -text
examples/co2/preclassified_data/*.npz filter=lfs diff=lfs merge=lfs -text
*.pkl filter=lfs diff=lfs merge=lfs -text
*.pth filter=lfs diff=lfs merge=lfs -text
*.h5 filter=lfs diff=lfs merge=lfs -text
*.hdf5 filter=lfs diff=lfs merge=lfs -text
*.ckpt filter=lfs diff=lfs merge=lfs -text
```

## Usage

### For Contributors

When you clone this repository:

```bash
# Clone with LFS
git clone <repo-url>

# Or if already cloned, pull LFS files
git lfs pull
```

### Checking LFS Status

```bash
# See which files are tracked by LFS
git lfs ls-files

# See LFS tracking patterns
git lfs track

# Check LFS status
git lfs status
```

### Adding New Large Files

If you add new data files, they'll automatically be tracked if they match the patterns above. To manually track specific files:

```bash
# Track a specific file
git lfs track "path/to/large_file.npz"

# Track a pattern
git lfs track "data/*.npz"

# Stage .gitattributes changes
git add .gitattributes
```

### Committing Large Files

```bash
# Add your files (LFS will handle them automatically)
git add examples/co2/preclassified_data/*.npz

# Commit as normal
git commit -m "Add processed CO2 data"

# Push (LFS files will be uploaded to LFS server)
git push
```

## Benefits of Using LFS

1. **Repository stays fast** - Large files aren't in git history
2. **Selective downloads** - Only fetch large files when needed
3. **Bandwidth efficient** - Don't download large files on every clone
4. **Version tracking** - Still track versions of large files
5. **Seamless workflow** - Works like regular git for most operations

## Common Commands

```bash
# Initialize LFS (already done)
git lfs install

# Pull all LFS files
git lfs pull

# Pull specific LFS files
git lfs pull --include="examples/co2/preclassified_data/*.npz"

# List tracked files
git lfs ls-files

# Get LFS file info
git lfs ls-files --size

# Fetch LFS files without checking out
git lfs fetch

# Prune old LFS files from local cache
git lfs prune
```

## Bandwidth Considerations

### Cloning
- **Without LFS files**: Fast, only code downloaded
- **With LFS files**: Slower, ~383 MB downloaded for CO2 data

### Pulling
- Only changed LFS files are downloaded
- Much faster than re-cloning

## Troubleshooting

### Files not tracked by LFS

```bash
# Check if file matches patterns
git check-attr filter path/to/file.npz

# Migrate existing files to LFS
git lfs migrate import --include="*.npz"
```

### LFS quota exceeded

```bash
# Check LFS usage
git lfs ls-files --size

# Remove large files if needed
git lfs prune --verify-remote
```

### Clone without LFS files

```bash
# Skip LFS files during clone
GIT_LFS_SKIP_SMUDGE=1 git clone <repo-url>

# Later pull LFS files when needed
git lfs pull
```

## Best Practices

1. **Don't commit unnecessarily large files** - Only commit final processed data
2. **Use .gitignore for generated data** - Don't track intermediate files
3. **Document data sources** - Explain where large files come from
4. **Compress when possible** - Use `.npz` instead of `.npy`
5. **Split large datasets** - Separate train/valid/test for easier handling

## File Recommendations

### Track with LFS ✅
- Processed data files (`.npz`, `.h5`)
- Trained model checkpoints (`.pkl`, `.pth`, `.ckpt`)
- Large configuration files
- Pre-computed features

### Don't track (use .gitignore) ❌
- Raw unprocessed data (provide download scripts instead)
- Temporary/intermediate files
- Log files
- Build artifacts
- Compiled binaries

## Current Setup Status

✅ Git LFS installed and initialized
✅ Patterns configured in `.gitattributes`
✅ Ready to track large files automatically
✅ ~383 MB of CO2 data ready for LFS tracking

## Next Steps

When ready to commit:

```bash
# Stage your changes
git add examples/co2/preclassified_data/*.npz
git add .gitattributes

# Commit
git commit -m "Add preclassified CO2 training data with LFS"

# Push (LFS files will be uploaded)
git push
```

## Additional Resources

- [Git LFS Documentation](https://git-lfs.github.com/)
- [GitHub LFS Guide](https://docs.github.com/en/repositories/working-with-files/managing-large-files)
- [Git LFS Tutorial](https://www.atlassian.com/git/tutorials/git-lfs)

---

**Note**: This setup is already complete. Large files will be automatically handled by LFS when you commit and push.

