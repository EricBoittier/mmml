# Gas Phase Multi-Molecule Simulations

Tools for simulating multiple CO₂ molecules at realistic gas densities.

## Quick Start

### Create and Visualize Gas System

```bash
# 10 molecules at 1 atm, 300 K
python gas_phase_calculator.py \
    --checkpoint /path/to/ckpt \
    --n-molecules 10 \
    --temperature 300 \
    --pressure 1.0 \
    --save-xyz \
    --output-dir ./gas_10mol

# View initial configuration
ase gui ./gas_10mol/gas_10mol_initial.xyz
```

### Run MD on Gas

```bash
# NVT simulation
python gas_phase_calculator.py \
    --checkpoint /path/to/ckpt \
    --n-molecules 20 \
    --temperature 300 \
    --pressure 1.0 \
    --run-md \
    --md-ensemble nvt \
    --md-timestep 0.5 \
    --md-steps 20000 \
    --output-dir ./gas_nvt_300K

# View trajectory
ase gui ./gas_nvt_300K/gas_nvt.traj
```

## Gas Density Calculator

The script automatically calculates box size using the **ideal gas law**:

```
PV = nRT

For 10 CO₂ molecules at 300 K, 1 atm:
→ Box size: ~152 Å
→ Density: ~2.9 × 10¹⁹ molecules/cm³
```

| Pressure (atm) | T (K) | N molecules | Box size (Å) | Density (mol/cm³) |
|----------------|-------|-------------|--------------|-------------------|
| 1.0 | 300 | 10 | 152 | 2.9e19 |
| 1.0 | 300 | 50 | 263 | 2.9e19 |
| 1.0 | 300 | 100 | 331 | 2.9e19 |
| 10.0 | 300 | 10 | 71 | 2.9e20 |
| 0.1 | 300 | 10 | 326 | 2.9e18 |

## Features

### Gas Phase Calculator (`gas_phase_calculator.py`)

**Automatic:**
- ✅ Calculates box size from T, P (ideal gas law)
- ✅ Random molecule placement (no overlaps)
- ✅ Random orientations
- ✅ Periodic boundary conditions
- ✅ Handles intermolecular interactions

**Actions:**
- `--save-xyz`: Save initial configuration
- `--run-opt`: Optimize gas structure
- `--run-md`: Run molecular dynamics (NVE/NVT)

**Example - 50 molecules:**
```bash
python gas_phase_calculator.py \
    --checkpoint ./ckpts/model \
    --n-molecules 50 \
    --temperature 300 \
    --pressure 1.0 \
    --run-md \
    --md-steps 50000 \
    --output-dir ./gas_50mol_300K
```

### JAX MD Gas (`gas_phase_jaxmd.py`)

**Status:** 🚧 In development

Multi-molecule JAX MD requires:
- Complex neighbor list management with PBC
- Efficient particle-particle interactions
- Careful handling of box boundaries

For now, use the ASE version above which works perfectly!

## System Sizes

**Recommended:**
- **Small test**: 5-10 molecules (~15-30 atoms)
- **Medium**: 20-50 molecules (~60-150 atoms)  
- **Large**: 100-200 molecules (~300-600 atoms)

**Performance (ASE MD on CPU):**
| N molecules | N atoms | Speed (steps/s) | 10 ps time |
|-------------|---------|-----------------|------------|
| 10 | 30 | ~100 | 2 min |
| 50 | 150 | ~20 | 10 min |
| 100 | 300 | ~5 | 40 min |

## Example Workflows

### 1. Equilibrate Gas at Different Temperatures

```bash
for T in 200 250 300 350 400; do
    python gas_phase_calculator.py \
        --checkpoint ./ckpts/model \
        --n-molecules 30 \
        --temperature $T \
        --pressure 1.0 \
        --run-md \
        --md-steps 20000 \
        --output-dir ./gas_T${T}K
done
```

### 2. Pressure Scan at Fixed Temperature

```bash
for P in 0.1 1.0 10.0; do
    python gas_phase_calculator.py \
        --checkpoint ./ckpts/model \
        --n-molecules 30 \
        --temperature 300 \
        --pressure $P \
        --run-md \
        --md-steps 20000 \
        --output-dir ./gas_P${P}atm
done
```

### 3. Size Dependence

```bash
for N in 10 20 50 100; do
    python gas_phase_calculator.py \
        --checkpoint ./ckpts/model \
        --n-molecules $N \
        --temperature 300 \
        --pressure 1.0 \
        --run-md \
        --md-steps 20000 \
        --output-dir ./gas_N${N}mol
done
```

## Analysis

After running MD:

```python
from ase.io import read

# Load trajectory
traj = read('./gas_nvt_300K/gas_nvt.traj', ':')

# Analyze properties
energies = [atoms.get_potential_energy() for atoms in traj]
temps = [atoms.get_temperature() for atoms in traj]

# Radial distribution function
from ase.geometry.analysis.rdf import get_rdf
rdf = get_rdf(traj[::10], rmax=10.0, nbins=100)

# Diffusion coefficient
# ... etc
```

## Physical Insights

**What you can study:**
1. **Intermolecular interactions**: How CO₂ molecules interact
2. **Gas properties**: Pressure, temperature, density correlations
3. **Transport properties**: Diffusion, viscosity (from Einstein relations)
4. **Phase behavior**: At high pressure, do molecules aggregate?
5. **Spectroscopy**: IR/Raman in gas phase (different from isolated molecule!)

## Comparison: Single vs Multi-Molecule

| Property | Single Molecule | Gas (10 molecules) |
|----------|----------------|-------------------|
| Atoms | 3 | 30 |
| Box size | ~10 Å | ~150 Å |
| Computation | Fast | 10× slower |
| Physics | Isolated | Intermolecular forces |
| Realistic | No | Yes! |

## Periodic Boundary Conditions

The calculator automatically handles PBC:
- Molecules can "wrap around" the box
- Forces computed with minimum image convention
- No edge effects
- Represents infinite gas

## Tips

1. **Start small**: Test with 5-10 molecules first
2. **Equilibrate**: Run NVT for 10-20 ps before production
3. **Timestep**: Use 0.5-1.0 fs (smaller if crashes)
4. **Cutoff**: 10 Å is usually sufficient for gas phase
5. **Box size**: Let the script calculate it (ideal gas law)

## Future: JAX MD Version

When implemented, JAX MD will be **100-1000× faster**:
```bash
python gas_phase_jaxmd.py \
    --n-molecules 200 \
    --nsteps 1000000 \
    --timestep 0.1
# Could run on GPU in minutes instead of hours!
```

## Troubleshooting

**"Molecules overlap"**
- Increase `--min-separation` (default: 3.0 Å)
- Reduce `--n-molecules` or increase `--pressure`

**"MD explodes"**
- Reduce `--md-timestep` (try 0.1 fs)
- Run `--run-opt` first to remove bad contacts
- Check that model is well-trained

**"Very slow"**
- Reduce `--n-molecules`
- Increase `--md-timestep` if stable
- Wait for JAX MD version (coming soon!)

## References

- **Ideal Gas Law**: https://en.wikipedia.org/wiki/Ideal_gas_law
- **MD with PBC**: Allen & Tildesley, "Computer Simulation of Liquids"
- **ASE MD**: https://wiki.fysik.dtu.dk/ase/ase/md.html

