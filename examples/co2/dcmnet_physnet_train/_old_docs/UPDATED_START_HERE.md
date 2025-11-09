# 🎨 Molecular Visualization - UPDATED GUIDE

## ✨ What's New?

The visualization tools now create **separate, specialized plots** with **wireframe rendering** for better visibility!

### Five Plot Types:
1. **`molecule`** - Clean structure (ball-and-stick)
2. **`molecule+charges`** - **Wireframe + colored charge spheres** 🆕
3. **`charges`** - Charges only (no molecule)
4. **`esp`** - ESP surface only  
5. **`molecule+esp`** - **Wireframe + ESP overlay** 🆕

### Key Feature: Automatic Wireframe! 🔥
When showing charges or ESP, the molecule automatically renders as a **wireframe** (thin bonds, small atoms) so the colored features are clearly visible!

---

## 🚀 Quick Start - New Commands

### All-in-One: Generate Complete Figure Set
```bash
cd /home/ericb/mmml/examples/co2/dcmnet_physnet_train

./ase_povray_viz.py \
    --checkpoint /home/ericb/mmml/mmml/physnetjax/ckpts/co2_joint/ \
    --structure analysis_co2/CO2_optimized.xyz \
    --plot-types molecule molecule+charges esp molecule+esp \
    --resolution high \
    --output-dir complete_figures
```

**Creates 4 separate files:**
- `*_molecule.png` - Clean molecular structure
- `*_molecule_charges.png` - Wireframe + charges (perfect visibility!)
- `*_esp.png` - ESP surface alone
- `*_molecule_esp.png` - Wireframe + ESP overlay

---

## 📊 Individual Plot Examples

### 1. Molecule Only
```bash
./ase_povray_viz.py \
    --checkpoint /home/ericb/mmml/mmml/physnetjax/ckpts/co2_joint/ \
    --structure analysis_co2/CO2_optimized.xyz \
    --plot-types molecule \
    --resolution high
```
**Use for:** Structure analysis, geometry figures

### 2. Charges on Wireframe (Recommended!) ⭐
```bash
./ase_povray_viz.py \
    --checkpoint /home/ericb/mmml/mmml/physnetjax/ckpts/co2_joint/ \
    --structure analysis_co2/CO2_optimized.xyz \
    --plot-types molecule+charges \
    --resolution high \
    --molecule-style wireframe
```
**Use for:** Charge distribution analysis
**Shows:** Thin molecular wireframe + colored charge spheres
- Red = positive charges
- Blue = negative charges
- Clear visibility!

### 3. ESP with Molecular Context
```bash
./ase_povray_viz.py \
    --checkpoint /home/ericb/mmml/mmml/physnetjax/ckpts/co2_joint/ \
    --structure analysis_co2/CO2_optimized.xyz \
    --plot-types molecule+esp \
    --resolution high \
    --n-surface-points 10000
```
**Use for:** Reactivity analysis, interaction sites
**Shows:** Wireframe + colored ESP surface

### 4. Everything Separate (Publication Set)
```bash
./ase_povray_viz.py \
    --checkpoint /home/ericb/mmml/mmml/physnetjax/ckpts/co2_joint/ \
    --structure analysis_co2/CO2_optimized.xyz \
    --plot-types all \
    --resolution ultra \
    --views "0x,0y,0z" "90x,0y,0z" \
    --output-dir publication
```
**Creates:** All 5 plot types × 2 views = 10 figures!

---

## 🎯 Quick Matplotlib Preview

Preview before rendering with POV-Ray:

```bash
./matplotlib_3d_viz.py \
    --checkpoint /home/ericb/mmml/mmml/physnetjax/ckpts/co2_joint/ \
    --structure analysis_co2/CO2_optimized.xyz \
    --plot-type molecule+charges \
    --molecule-style wireframe \
    --output preview.png
```

Or interactive:
```bash
./matplotlib_3d_viz.py \
    --checkpoint /home/ericb/mmml/mmml/physnetjax/ckpts/co2_joint/ \
    --structure YOUR_MOLECULE.xyz \
    --plot-type molecule+charges \
    --interactive
```

---

## 🎨 What Each Plot Shows

| Plot Type | Molecule Style | Shows | Best For |
|-----------|---------------|-------|----------|
| `molecule` | Full ball-and-stick | Structure only | Geometry |
| `molecule+charges` | **Wireframe** | + Colored spheres | Charge analysis |
| `charges` | None (invisible) | Charges alone | Pure charge view |
| `esp` | None (invisible) | ESP surface alone | ESP maps |
| `molecule+esp` | **Wireframe** | + ESP surface | Reactivity |

---

## ⚙️ Common Options

### Plot Types
```bash
--plot-types molecule                    # Structure only
--plot-types molecule+charges            # Charges with wireframe
--plot-types molecule+esp                # ESP with wireframe
--plot-types molecule molecule+charges   # Both (separate files)
--plot-types all                         # All 5 types
```

### Molecule Style (for molecule+charges)
```bash
--molecule-style wireframe    # Thin, clear (default for overlays)
--molecule-style full         # Full ball-and-stick
```

### Rendering Quality
```bash
--resolution medium      # Fast (~30s/view)
--resolution high        # Good quality (~60s)
--resolution ultra       # Publication (4K, ~120s)
```

### Multiple Views
```bash
--views "0x,0y,0z"                           # Front
--views "0x,0y,0z" "90x,0y,0z"              # Front + side
--views "0x,0y,0z" "90x,0y,0z" "0x,90y,0z"  # 3 views
```

### Fine-Tuning
```bash
--charge-radius 0.2          # Larger charge spheres
--transparency 0.6           # More transparent charges
--n-surface-points 10000     # Smoother ESP surface
```

---

## 💡 Recommended Workflows

### For Charge Analysis
```bash
# Clean wireframe + charges
./ase_povray_viz.py \
    --checkpoint YOUR_CHECKPOINT \
    --structure YOUR_MOLECULE \
    --plot-types molecule+charges \
    --molecule-style wireframe \
    --charge-radius 0.18 \
    --resolution high
```

### For ESP Analysis
```bash
# ESP surface with molecular context
./ase_povray_viz.py \
    --checkpoint YOUR_CHECKPOINT \
    --structure YOUR_MOLECULE \
    --plot-types esp molecule+esp \
    --n-surface-points 12000 \
    --resolution high
```

### For Publication
```bash
# Complete set: structure, charges, ESP
./ase_povray_viz.py \
    --checkpoint YOUR_CHECKPOINT \
    --structure YOUR_MOLECULE \
    --plot-types molecule molecule+charges molecule+esp \
    --resolution ultra \
    --views "45x,30y,0z" \
    --n-surface-points 15000 \
    --output-dir publication_figs
```

### For Quick Check
```bash
# Fast matplotlib preview
./matplotlib_3d_viz.py \
    --checkpoint YOUR_CHECKPOINT \
    --structure YOUR_MOLECULE \
    --plot-type molecule+charges \
    --output quick_check.png
```

---

## 📁 Output File Structure

With `--views "0x,0y,0z" --plot-types molecule molecule+charges esp`:

```
output_dir/
├── view_1_0x_0y_0z_molecule.png          ← Clean structure
├── view_1_0x_0y_0z_molecule_charges.png  ← Wireframe + charges
├── view_1_0x_0y_0z_esp.png               ← ESP surface
├── colorbar_charges.png                   ← Charge legend
├── colorbar_esp.png                       ← ESP legend
└── *.pov (POV-Ray source files)
```

Clear, descriptive filenames!

---

## 🔍 Examples with Your Data

### CO₂ Molecule
```bash
./ase_povray_viz.py \
    --checkpoint /home/ericb/mmml/mmml/physnetjax/ckpts/co2_joint/ \
    --structure analysis_co2/CO2_optimized.xyz \
    --plot-types molecule+charges \
    --resolution high \
    --output-dir co2_charges
```

### Multiple Normal Modes
```bash
for mode in raman_analysis/mode_*.xyz; do
    name=$(basename "$mode" .xyz)
    ./ase_povray_viz.py \
        --checkpoint /home/ericb/mmml/mmml/physnetjax/ckpts/co2_joint/ \
        --structure "$mode" \
        --plot-types molecule+charges \
        --resolution high \
        --output-dir "charges_${name}"
done
```

### Multiple Views of One Molecule
```bash
./ase_povray_viz.py \
    --checkpoint /home/ericb/mmml/mmml/physnetjax/ckpts/co2_joint/ \
    --structure YOUR_MOLECULE.xyz \
    --plot-types molecule+charges molecule+esp \
    --views "0x,0y,0z" "90x,0y,0z" "0x,90y,0z" "45x,45y,0z" \
    --resolution high \
    --output-dir multi_view
```

---

## 🆚 Old vs New

### Before (Old Way)
```bash
--show-charges --show-esp
```
- Single combined image
- Full ball-and-stick (charges hidden behind atoms!)
- Less flexible

### Now (New Way!) ✨
```bash
--plot-types molecule+charges molecule+esp
```
- **Separate specialized files**
- **Automatic wireframe** (charges visible!)
- Multiple plot types in one command
- Professional results

---

## 📚 Documentation

- **This file** - Quick start
- **`SEPARATE_PLOTS_GUIDE.md`** - Detailed plot type guide
- **`VISUALIZATION_README.md`** - Complete reference
- **`VISUALIZATION_SUCCESS.md`** - Test results

All scripts support `--help`:
```bash
./ase_povray_viz.py --help
./matplotlib_3d_viz.py --help
```

---

## ✅ Tested and Working

```
✓ All 5 plot types working
✓ Wireframe mode for overlays
✓ Separate file generation
✓ Multiple views supported
✓ matplotlib and POV-Ray both updated
```

Demo outputs created in: `demo_separate_plots/`

---

## 🚀 Start Creating!

**Recommended first command:**
```bash
cd /home/ericb/mmml/examples/co2/dcmnet_physnet_train

./ase_povray_viz.py \
    --checkpoint /home/ericb/mmml/mmml/physnetjax/ckpts/co2_joint/ \
    --structure analysis_co2/CO2_optimized.xyz \
    --plot-types molecule+charges molecule+esp \
    --resolution high \
    --output-dir my_first_figures
```

This creates beautiful figures with **wireframe molecules** and clearly visible **charges** and **ESP**! 🎉

---

**The wireframe rendering makes all the difference - your charges and ESP are finally visible!** 🌟

