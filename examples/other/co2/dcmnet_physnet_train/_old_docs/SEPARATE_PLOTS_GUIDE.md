# Separate Plot Types Guide

## 🎨 Updated Visualization Tools

The visualization tools now create **separate, specialized plots** for different purposes:

1. **`molecule`** - Clean molecular structure only (full ball-and-stick)
2. **`molecule+charges`** - Molecule with distributed charges (wireframe + charges)
3. **`charges`** - Charges only (no molecule)
4. **`esp`** - ESP surface only (no molecule)
5. **`molecule+esp`** - Molecule with ESP overlay (wireframe + ESP)

## 🔄 Key Improvements

### Wireframe Mode for Overlays
When showing charges or ESP, the molecule is rendered as a **wireframe** (thin bonds, small atoms) so the colored spheres and surfaces are clearly visible.

### Separate Files for Each Type
Each plot type creates its own file, making it easy to:
- Use in different contexts (molecule structure vs. charge analysis)
- Compare side-by-side
- Choose the best representation for your purpose

## 🚀 Quick Examples

### Generate All Plot Types
```bash
./ase_povray_viz.py \
    --checkpoint /home/ericb/mmml/mmml/physnetjax/ckpts/co2_joint/ \
    --structure analysis_co2/CO2_optimized.xyz \
    --plot-types molecule molecule+charges esp molecule+esp \
    --resolution high \
    --output-dir all_plots
```

**Creates:**
- `view_1_0x_0y_0z_molecule.png` - Clean molecule
- `view_1_0x_0y_0z_molecule_charges.png` - Wireframe + charges
- `view_1_0x_0y_0z_esp.png` - ESP surface only
- `view_1_0x_0y_0z_molecule_esp.png` - Wireframe + ESP

### Just Charges and ESP
```bash
./ase_povray_viz.py \
    --checkpoint YOUR_CHECKPOINT \
    --structure YOUR_MOLECULE.xyz \
    --plot-types molecule+charges molecule+esp \
    --resolution high
```

### Full Suite with All Options
```bash
./ase_povray_viz.py \
    --checkpoint YOUR_CHECKPOINT \
    --structure YOUR_MOLECULE.xyz \
    --plot-types all \
    --resolution high
```
(Creates molecule, molecule+charges, charges, esp, molecule+esp)

### Custom Molecule Style
```bash
./ase_povray_viz.py \
    --checkpoint YOUR_CHECKPOINT \
    --structure YOUR_MOLECULE.xyz \
    --plot-types molecule+charges \
    --molecule-style full \
    --resolution high
```
(Uses full ball-and-stick even with charges - your choice!)

## 📊 Plot Type Details

### 1. **molecule**
- **Purpose**: Show molecular structure clearly
- **Rendering**: Full ball-and-stick
- **Best for**: Geometric analysis, structure overview
- **File suffix**: `_molecule.png`

### 2. **molecule+charges**
- **Purpose**: Show distributed charge locations
- **Rendering**: **Wireframe molecule** + colored charge spheres
- **Colors**: Red = positive, Blue = negative
- **Best for**: Understanding charge distribution
- **File suffix**: `_molecule_charges.png`

### 3. **charges** (charges only)
- **Purpose**: Focus entirely on charge distribution
- **Rendering**: Colored spheres only (no molecule)
- **Best for**: Charge analysis without structural context
- **File suffix**: `_charges_only.png`

### 4. **esp** (ESP surface only)
- **Purpose**: Show electrostatic potential clearly
- **Rendering**: Colored VDW surface only (no molecule)
- **Best for**: Analyzing reactive regions, ESP maps
- **File suffix**: `_esp.png`

### 5. **molecule+esp**
- **Purpose**: Show ESP in molecular context
- **Rendering**: **Wireframe molecule** + colored ESP surface
- **Best for**: Understanding how ESP relates to structure
- **File suffix**: `_molecule_esp.png`

## 🎯 Matplotlib Version

The matplotlib tool also supports plot types:

```bash
./matplotlib_3d_viz.py \
    --checkpoint YOUR_CHECKPOINT \
    --structure YOUR_MOLECULE.xyz \
    --plot-type molecule+charges \
    --molecule-style wireframe \
    --output preview.png
```

**Options:**
- `--plot-type`: molecule, molecule+charges, charges, esp, molecule+esp
- `--molecule-style`: full, wireframe

## 💡 Best Practices

### For Charge Analysis
```bash
# Create clean charge visualization
./ase_povray_viz.py \
    --checkpoint CHECKPOINT \
    --structure MOLECULE \
    --plot-types molecule+charges \
    --molecule-style wireframe \
    --charge-radius 0.2 \
    --transparency 0.6 \
    --resolution high
```
- Use **wireframe** to avoid obscuring charges
- Adjust `--charge-radius` for visibility
- Lower `--transparency` if charges are too faint

### For ESP Analysis
```bash
# Clean ESP map with context
./ase_povray_viz.py \
    --checkpoint CHECKPOINT \
    --structure MOLECULE \
    --plot-types esp molecule+esp \
    --n-surface-points 10000 \
    --resolution high
```
- Generate both ESP-only and molecule+ESP
- Use **wireframe** for overlay (automatic)
- Increase `--n-surface-points` for smoother surface

### For Publication
```bash
# Complete set of high-quality figures
./ase_povray_viz.py \
    --checkpoint CHECKPOINT \
    --structure MOLECULE \
    --plot-types molecule molecule+charges molecule+esp \
    --resolution ultra \
    --views "0x,0y,0z" "90x,0y,0z" \
    --n-surface-points 15000 \
    --output-dir publication_figures
```

### For Presentations
```bash
# Fast, clear figures
./ase_povray_viz.py \
    --checkpoint CHECKPOINT \
    --structure MOLECULE \
    --plot-types molecule molecule+charges \
    --resolution medium \
    --molecule-style wireframe
```

## 🔧 Customization

### Molecule Appearance

| Parameter | Effect | Recommended |
|-----------|--------|-------------|
| `--molecule-style full` | Solid ball-and-stick | Molecule-only plots |
| `--molecule-style wireframe` | Thin bonds, small atoms | With charges/ESP |

### Charge Appearance

| Parameter | Effect | Default | Range |
|-----------|--------|---------|-------|
| `--charge-radius` | Sphere size | 0.15 | 0.1-0.3 |
| `--transparency` | Sphere opacity | 0.5 | 0.3-0.7 |

### ESP Surface

| Parameter | Effect | Default | Range |
|-----------|--------|---------|-------|
| `--n-surface-points` | Surface resolution | 5000 | 2000-15000 |

## 📁 Output Structure

With multiple views and plot types:

```
output_dir/
├── view_1_0x_0y_0z_molecule.png
├── view_1_0x_0y_0z_molecule_charges.png
├── view_1_0x_0y_0z_esp.png
├── view_1_0x_0y_0z_molecule_esp.png
├── view_2_90x_0y_0z_molecule.png
├── view_2_90x_0y_0z_molecule_charges.png
├── view_2_90x_0y_0z_esp.png
├── view_2_90x_0y_0z_molecule_esp.png
├── colorbar_charges.png
├── colorbar_esp.png
└── *.pov files
```

Clear naming makes it easy to find the right figure!

## 🎓 Use Cases

### Comparing Molecules
```bash
for mol in molecules/*.xyz; do
    name=$(basename "$mol" .xyz)
    ./ase_povray_viz.py \
        --checkpoint CHECKPOINT \
        --structure "$mol" \
        --plot-types molecule+charges \
        --molecule-style wireframe \
        --views "45x,30y,0z" \
        --output-dir "charges_${name}"
done
```

### Animation Frames
```bash
for angle in {0..350..10}; do
    ./ase_povray_viz.py \
        --checkpoint CHECKPOINT \
        --structure MOLECULE \
        --plot-types molecule+charges \
        --views "${angle}x,0y,0z" \
        --resolution medium \
        --output-dir animation_frames
done
```

### Side-by-Side Comparison
```bash
# Generate all types for comparison
./ase_povray_viz.py \
    --checkpoint CHECKPOINT \
    --structure MOLECULE \
    --plot-types all \
    --views "45x,30y,0z" \
    --resolution high \
    --output-dir comparison

# Then use image montage or LaTeX to arrange side-by-side
```

## ✅ What Changed

### Old Behavior (Before)
- `--show-charges` and `--show-esp` flags
- Single combined figure
- Full ball-and-stick always
- Hard to see charges behind atoms

### New Behavior (Now) ✨
- `--plot-types` with multiple options
- Separate files for each type
- **Automatic wireframe** for overlays
- Clear visibility of all features
- Flexible customization

## 🔄 Migration Guide

### Old Command
```bash
./ase_povray_viz.py \
    --checkpoint CHECKPOINT \
    --structure MOLECULE \
    --show-charges \
    --show-esp
```

### New Equivalent
```bash
./ase_povray_viz.py \
    --checkpoint CHECKPOINT \
    --structure MOLECULE \
    --plot-types molecule+charges molecule+esp
```

### Advantages of New Approach
1. ✅ Separate, focused figures
2. ✅ Wireframe for better visibility
3. ✅ More flexibility
4. ✅ Easy to generate multiple types
5. ✅ Clear file naming

## 🎉 Summary

The updated tools provide:
- **5 specialized plot types** for different purposes
- **Automatic wireframe** when overlaying charges/ESP
- **Separate files** for each combination
- **Full control** via command-line options

Perfect for creating publication-quality figures that clearly show:
- Molecular structure
- Distributed charges  
- Electrostatic potential
- All combinations thereof

**Start visualizing with the new approach!** 🚀

