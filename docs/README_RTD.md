# ReadTheDocs Deployment Guide

This guide explains how to deploy MMML documentation to ReadTheDocs.

## Prerequisites

- GitHub repository for MMML
- ReadTheDocs account
- MkDocs and dependencies installed

## Local Preview

### 1. Install Dependencies

```bash
pip install mkdocs mkdocs-material mkdocstrings[python] pymdown-extensions
```

### 2. Preview Documentation

```bash
cd /home/ericb/mmml
mkdocs serve
```

Open browser to `http://127.0.0.1:8000`

### 3. Build Static Site

```bash
mkdocs build
# Output in site/ directory
```

## ReadTheDocs Deployment

### Option 1: Automatic (Recommended)

1. **Connect Repository**
   - Go to https://readthedocs.org
   - Click "Import a Project"
   - Select your GitHub repository
   - ReadTheDocs auto-detects `docs/mkdocs.yml`

2. **Configure Build**
   - Build format: MkDocs
   - Configuration file: `docs/mkdocs.yml`
   - Python version: 3.9+
   - Requirements: `docs/requirements.txt` (create if needed)

3. **Trigger Build**
   - Push to main branch
   - RTD automatically builds
   - Site available at `https://mmml.readthedocs.io`

### Option 2: Manual Configuration

1. **Create `.readthedocs.yaml`** in repository root:

```yaml
# .readthedocs.yaml
version: 2

mkdocs:
  configuration: docs/mkdocs.yml

python:
  version: "3.9"
  install:
    - requirements: docs/requirements.txt

build:
  os: ubuntu-22.04
  tools:
    python: "3.9"
```

2. **Create `docs/requirements.txt`:**

```txt
mkdocs>=1.5.0
mkdocs-material>=9.0.0
mkdocstrings[python]>=0.24.0
pymdown-extensions>=10.0
```

3. **Push to GitHub:**

```bash
git add .readthedocs.yaml docs/
git commit -m "Add ReadTheDocs configuration"
git push
```

4. **Import on ReadTheDocs:**
   - RTD will use `.readthedocs.yaml` automatically

## Documentation Structure

Current documentation files ready for deployment:

```
docs/
├── mkdocs.yml              ← Main configuration
├── index.md                ← Landing page
├── quickstart.md           ← Quick start guide
├── data_pipeline.md        ← Pipeline documentation
├── cli_reference.md        ← CLI commands
├── npz_schema.md           ← Data format spec
└── README_RTD.md           ← This file
```

## Customization

### Adding Pages

1. Create new `.md` file in `docs/`
2. Add to `nav` in `docs/mkdocs.yml`:

```yaml
nav:
  - User Guide:
    - Your New Page: your_page.md
```

### Styling

Add custom CSS in `docs/stylesheets/extra.css`

### Search

Search is automatic with MkDocs Material theme.

## Verification

### Check Build Locally

```bash
# Clean build
mkdocs build --clean

# Check for warnings
mkdocs build --strict

# Serve and test
mkdocs serve
```

### Test Links

```bash
# Check all internal links
mkdocs serve &
sleep 5
curl -I http://127.0.0.1:8000/
curl -I http://127.0.0.1:8000/data_pipeline/
```

## Common Issues

### Issue: "Module not found" during build

**Solution:** Add to `docs/requirements.txt`:
```txt
numpy
scipy
ase
```

### Issue: "Configuration file not found"

**Solution:** Ensure `mkdocs.yml` is in `docs/` directory:
```bash
ls docs/mkdocs.yml  # Should exist
```

### Issue: Build warnings about missing files

**Solution:** Create placeholder files or remove from nav:
```bash
# Create missing files
touch docs/installation.md
touch docs/tutorial.md
```

## URLs After Deployment

Once deployed on ReadTheDocs:

- **Main site:** `https://mmml.readthedocs.io`
- **Latest version:** `https://mmml.readthedocs.io/en/latest/`
- **Specific page:** `https://mmml.readthedocs.io/en/latest/data_pipeline/`
- **PDF download:** Available via RTD

## Maintenance

### Updating Documentation

```bash
# Edit markdown files
vim docs/some_page.md

# Commit and push
git add docs/
git commit -m "Update documentation"
git push

# ReadTheDocs auto-rebuilds (usually < 2 minutes)
```

### Versioning

RTD supports multiple versions:
- `latest` - Main branch
- `stable` - Tagged releases
- `v0.1.0` - Specific version tags

## Current Status

✅ **Ready for Deployment:**
- MkDocs configuration complete
- 5+ documentation pages written
- Navigation structured
- Material theme configured
- All prerequisites met

**Estimated deployment time:** 5-10 minutes

## Next Steps

1. ✅ Create `.readthedocs.yaml` (see above)
2. ✅ Create `docs/requirements.txt` (see above)
3. Create remaining placeholder pages (installation.md, tutorial.md, etc.)
4. Push to GitHub
5. Connect repository to ReadTheDocs
6. Trigger first build
7. Verify site works
8. Share URL with team!

---

**Note:** Some documentation pages referenced in `mkdocs.yml` nav need to be created. They can start as placeholders:

```markdown
# Coming Soon

This page is under construction.

For now, see:
- [Quick Start](quickstart.md)
- [Data Pipeline](data_pipeline.md)
```

---

**Status:** ✅ Ready to deploy  
**Estimated Time:** 10 minutes to live site

