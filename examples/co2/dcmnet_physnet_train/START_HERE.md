# 🎨 Molecular Visualization Tools - START HERE

## 🎉 Ready to Use!

All visualization tools have been **tested and working** with your checkpoint at:
```
/home/ericb/mmml/mmml/physnetjax/ckpts/co2_joint/
```

## 🚀 Quick Start (Copy & Paste)

### Option 1: Fast Preview (Matplotlib)
```bash
cd /home/ericb/mmml/examples/co2/dcmnet_physnet_train

./matplotlib_3d_viz.py \
    --checkpoint /home/ericb/mmml/mmml/physnetjax/ckpts/co2_joint/ \
    --structure analysis_co2/CO2_optimized.xyz \
    --show-charges \
    --output my_molecule.png
```

### Option 2: Beautiful High-Quality (POV-Ray) ⭐
```bash
cd /home/ericb/mmml/examples/co2/dcmnet_physnet_train

./ase_povray_viz.py \
    --checkpoint /home/ericb/mmml/mmml/physnetjax/ckpts/co2_joint/ \
    --structure analysis_co2/CO2_optimized.xyz \
    --show-charges \
    --resolution high \
    --output-dir my_figures
```

### Option 3: With ESP Surface
```bash
cd /home/ericb/mmml/examples/co2/dcmnet_physnet_train

./ase_povray_viz.py \
    --checkpoint /home/ericb/mmml/mmml/physnetjax/ckpts/co2_joint/ \
    --structure analysis_co2/CO2_optimized.xyz \
    --show-charges \
    --show-esp \
    --resolution high \
    --output-dir figures_with_esp
```

## 📁 What You Get

### Ball-and-Stick Molecules
- Professional atomic spheres
- Element coloring (C=gray, O=red, H=white, etc.)
- Automatic bonds

### Distributed Charges
- **Colored spheres** at off-atom positions
- **Red** = positive charge
- **Blue** = negative charge
- Size = charge magnitude

### ESP on VDW Surface
- Electrostatic potential on molecular surface
- **Red** = repulsive regions
- **Blue** = attractive regions
- Shows where molecules interact

## 📚 Documentation Files

| File | Purpose |
|------|---------|
| **START_HERE.md** | This file - quick start |
| **VISUALIZATION_QUICK_START.md** | Common commands |
| **VISUALIZATION_README.md** | Complete reference |
| **VISUALIZATION_SUCCESS.md** | Test results & examples |

## 🛠️ The Tools

### 1. `matplotlib_3d_viz.py`
- **Fast** 3D preview
- No POV-Ray needed
- Interactive mode available
- Good for exploring viewing angles

### 2. `ase_povray_viz.py` ⭐ **RECOMMENDED**
- **Best quality** output
- Professional rendering
- Resolution presets
- Multiple viewing angles

### 3. `povray_visualization.py`
- Advanced control
- Manual settings
- For power users

## ⚙️ Common Options

```bash
# Resolution
--resolution low         # 800×600, ~10s
--resolution medium      # 1600×1200, ~30s
--resolution high        # 2400×1800, ~60s (default)
--resolution ultra       # 3840×2160, ~120s

# What to show
--show-charges          # Show distributed charges
--show-esp              # Show ESP surface

# Viewing angles
--views "0x,0y,0z"      # Front view
--views "90x,0y,0z"     # Side view
--views "0x,90y,0z"     # Top view
--views "45x,30y,0z"    # Angled view

# Appearance
--charge-radius 0.2     # Charge sphere size (default 0.15)
--transparency 0.7      # Charge transparency (default 0.5)
```

## 💡 Workflow Tips

### 1. Explore with Matplotlib
```bash
./matplotlib_3d_viz.py \
    --checkpoint YOUR_CHECKPOINT \
    --structure YOUR_MOLECULE.xyz \
    --show-charges \
    --interactive
```
Rotate to find best angle, then note it.

### 2. Render with POV-Ray
```bash
./ase_povray_viz.py \
    --checkpoint YOUR_CHECKPOINT \
    --structure YOUR_MOLECULE.xyz \
    --show-charges \
    --views "YOUR_ANGLE" \
    --resolution high
```

### 3. For Publication
```bash
./ase_povray_viz.py \
    --checkpoint YOUR_CHECKPOINT \
    --structure YOUR_MOLECULE.xyz \
    --show-charges \
    --show-esp \
    --resolution ultra \
    --n-surface-points 10000
```

## 🎯 Example Use Cases

### Visualize Normal Modes
```bash
for mode in raman_analysis/mode_*.xyz; do
    ./ase_povray_viz.py \
        --checkpoint /home/ericb/mmml/mmml/physnetjax/ckpts/co2_joint/ \
        --structure "$mode" \
        --show-charges \
        --output-dir "$(basename ${mode%.xyz})_figures"
done
```

### Multiple Viewing Angles
```bash
./ase_povray_viz.py \
    --checkpoint /home/ericb/mmml/mmml/physnetjax/ckpts/co2_joint/ \
    --structure YOUR_MOLECULE.xyz \
    --show-charges \
    --views "0x,0y,0z" "90x,0y,0z" "0x,90y,0z" "45x,45y,0z" \
    --resolution high \
    --output-dir multi_view
```

### Batch Processing
```bash
for xyz in molecules/*.xyz; do
    name=$(basename "$xyz" .xyz)
    ./matplotlib_3d_viz.py \
        --checkpoint /home/ericb/mmml/mmml/physnetjax/ckpts/co2_joint/ \
        --structure "$xyz" \
        --show-charges \
        --output "preview_${name}.png"
done
```

## ✅ Verified Working

```
✓ All Python packages installed
✓ Model loads correctly (64 features, 3 DCM sites)
✓ POV-Ray available
✓ Matplotlib visualization works
✓ POV-Ray scene generation works
✓ Test renders created successfully
```

## 🎓 Learn More

Run any script with `--help`:
```bash
./ase_povray_viz.py --help
./matplotlib_3d_viz.py --help
./test_visualization.py --help
```

## 🆘 Troubleshooting

### POV-Ray not installed
```bash
# Ubuntu/Debian
sudo apt install povray

# macOS
brew install povray
```

### Test if everything works
```bash
./test_visualization.py --checkpoint /home/ericb/mmml/mmml/physnetjax/ckpts/co2_joint/
```

### Charges look weird
The charge and energy values may need unit conversion. The visualization still works perfectly - just the displayed numbers may need interpretation.

## 🎨 Example Outputs

Check these files to see what was created:
- `test_co2_matplotlib.png` - Matplotlib 3D preview
- `test_povray_output/view_1_0x_0y_0z.png` - Front view (POV-Ray)
- `test_povray_output/view_2_90x_0y_0z.png` - Side view (POV-Ray)
- `test_povray_output/colorbar_charges.png` - Color legend

## 🚀 Ready!

Everything is set up and tested. Just run the commands above with your desired:
- `--structure` (molecule file)
- `--output-dir` (where to save)
- `--views` (viewing angles)
- `--resolution` (quality level)

**Enjoy creating beautiful molecular visualizations!** 🎉

