# QC supplementary cross-check

MMML provides a unified **cross-check** workflow to evaluate the same structures with multiple quantum-chemistry and ML backends and compare energies and forces against a reference.

This complements:

- **`mmml compare-npz`** — low-level NPZ-vs-NPZ metrics when you already have labels and predictions
- **`mmml orca-server` / `orca-client`** — ORCA **drives** MMML ML potentials via ExtOpt (Opt/GOAT). That direction is ORCA → MMML, not ORCA-as-reference.

Cross-check runs **MMML → backends** (PySCF, ORCA QM, xTB, Molpro, ML checkpoint) and writes per-backend metrics, plots, and an optional summary JSON.

## Quick start

```bash
# Precomputed PySCF labels as reference; evaluate ML checkpoint on same geometries
mmml cross-check \
  -i out/06_sampled.npz \
  --reference-npz out/07_evaluated.npz \
  --checkpoint path/to/epoch.pkl \
  -o validation/

# YAML config (multiple backends)
mmml cross-check -c examples/cross_check/cross_check.example.yaml
```

## Backends

| Backend | Role | Dependency |
|---------|------|------------|
| **pyscf** | Primary DFT reference (same stack as `pyscf-evaluate`) | `mmml[quantum-gpu]` |
| **ml** | Checkpoint inference (`SimpleInferenceCalculator`) | core MMML |
| **orca** | Independent QM reference (subprocess EnGrad) | ORCA binary on `PATH` or `$ORCA` |
| **xtb** | Fast GFN-xTB sanity check | `mmml[quantum-crosscheck]` (`tblite`) |
| **molpro** | Live Molpro SP + gradient → XML parse | Molpro binary on `PATH` or `$MOLPRO` |

### Method matching

Cross-check reports record method/basis per backend. Comparing **GFN2-xTB** to **ωB97M-D3** is useful as a **sanity screen**, not a numerical pass/fail oracle. For ML validation, use PySCF labels trained at the same level of theory as the model.

Suggested reference hierarchy:

1. **PySCF** — authoritative for ML vs QM validation
2. **ORCA QM** — independent implementation of matched method/basis (catches gpu4pyscf/PySCF bugs)
3. **xTB** — cheap force sanity on larger neutrals
4. **Molpro** — legacy datasets / methods already in Molpro XML archives
5. **ML** — subject under test, not the reference

## YAML configuration

```yaml
structures: sampled.npz
reference_npz: 07_evaluated.npz   # or: reference: pyscf

backends:
  - name: ml
    checkpoint: epoch.pkl
  - name: xtb
    method: GFN2-xTB
  - name: orca
    method: PBE
    basis: def2-SVP
    template: examples/cross_check/orca_template.inp

max_frames: 50
stride: 1
charge: 0
spin: 0
output: cross_check_out
```

### ORCA arbitrary jobs

Use `template` with placeholders `{xyz}`, `{method}`, `{basis}`, `{charge}`, `{mult}`, `{pal}` for custom ORCA blocks (CPCM, RI-JK, etc.). Default template runs `! METHOD BASIS EnGrad`.

### Molpro templates

Use `template` with `{geometry}`, `{basis}`, `{method}`, `{charge}`, `{mult}`. Default runs RHF + `force` and parses XML via the existing `parse_molpro` stack.

## Output

```
cross_check_out/
  reference.npz              # normalized reference (optional)
  cross_check_summary.json   # metrics table + method warnings
  ml/
    comparison_report.json
    *.png                    # energy/force plots (unless --no-plots)
    predictions.npz
  xtb/
    ...
```

## ORCA ExtOpt vs ORCA QM cross-check

| Feature | ExtOpt (`orca-server`) | Cross-check (`orca` backend) |
|---------|------------------------|------------------------------|
| Direction | ORCA calls MMML ML PES | MMML calls ORCA QM |
| Use case | ML-driven Opt/GOAT in ORCA | Validate PySCF/ML against ORCA DFT |
| Input | ORCA `*.extinp.tmp` | MMML NPZ/XYZ structures |
| Output | `*.engrad` callback to ORCA | NPZ + comparison report |

See also [tests/functionality/orca_external/README.md](../tests/functionality/orca_external/README.md) for ExtOpt smoke workflow.

## Tests

```bash
pytest tests/unit/test_cross_check.py tests/unit/test_orca_qm.py
```

Manual smoke (GPU/QC node with ORCA/Molpro/tblite):

```bash
mmml cross-check -i tests/fixtures/cross_check/water_frames.npz \
  --reference-npz tests/fixtures/cross_check/water_frames.npz \
  --backend xtb --max-frames 1 -o /tmp/xcheck_smoke
```
