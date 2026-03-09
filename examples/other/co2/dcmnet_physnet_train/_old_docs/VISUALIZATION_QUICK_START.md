# Visualization Quick Start Guide

## 🎯 Goal
Create beautiful figures showing molecules with distributed charges and ESP on VDW surfaces.

## 🚀 Quick Commands

### 1. Fast Preview (No POV-Ray Required)
```bash
./matplotlib_3d_viz.py \
    --checkpoint checkpoints/your_model \
    --structure molecule.xyz \
    --show-charges \
    --output preview.png
```

### 2. High-Quality Figure (Recommended) ⭐
```bash
./ase_povray_viz.py \
    --checkpoint checkpoints/your_model \
    --structure molecule.xyz \
    --show-charges \
    --show-esp \
    --resolution high \
    --output-dir ./figures
```

### 3. Publication Quality
```bash
./ase_povray_viz.py \
    --checkpoint checkpoints/your_model \
    --structure molecule.xyz \
    --show-charges \
    --show-esp \
    --resolution ultra \
    --n-surface-points 10000 \
    --output-dir ./publication_figures
```

## 📋 What You Need

1. **Trained model checkpoint directory** containing:
   - `model_config.pkl`
   - `best_params.pkl`

2. **Structure file** (`.xyz`, `.cif`, etc.)

3. **POV-Ray installed** (for high-quality renders):
   ```bash
   # Ubuntu/Debian
   sudo apt install povray
   
   # macOS
   brew install povray
   ```

## 🎨 Output Examples

### With Charges Only
- Molecular structure: ball-and-stick
- Charges: colored spheres (red=positive, blue=negative)
- Size proportional to charge magnitude

### With ESP Surface
- Everything above, plus:
- VDW surface colored by electrostatic potential
- Shows molecular reactivity regions

## 📊 Files Generated

```
output_dir/
├── view_1_0x_0y_0z.png          # Front view
├── view_2_90x_0y_0z.png         # Side view
├── view_3_0x_90y_0z.png         # Top view
├── colorbar_charges.png         # Charge legend
├── colorbar_esp.png             # ESP legend
└── *.pov                        # POV-Ray source files
```

## ⚙️ Common Adjustments

```bash
# Bigger/smaller charge spheres
--charge-radius 0.2              # Default: 0.15

# More/less transparent charges
--transparency 0.7               # Default: 0.5

# Custom viewing angles
--views "45x,45y,0z" "30x,60y,0z"

# Faster rendering
--resolution medium              # Options: low, medium, high, ultra

# Smoother ESP surface
--n-surface-points 8000          # Default: 5000
```

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| "POV-Ray not found" | Install POV-Ray (see above) or use matplotlib version |
| Charges too small | Increase `--charge-radius` to 0.3 |
| Rendering slow | Use `--resolution medium` or `low` |
| ESP surface noisy | Increase `--n-surface-points` to 8000-10000 |
| Out of memory | Reduce `--n-surface-points` to 2000-3000 |

## 🎓 Workflow Tip

1. **Start with matplotlib** for quick preview:
   ```bash
   ./matplotlib_3d_viz.py --checkpoint MODEL --structure FILE --show-charges --interactive
   ```

2. **Rotate view** interactively to find best angle

3. **Note the angle**, then render with POV-Ray for publication:
   ```bash
   ./ase_povray_viz.py --checkpoint MODEL --structure FILE \
       --show-charges --show-esp --resolution high \
       --views "YOUR_ANGLE"
   ```

## 📚 More Info

See `VISUALIZATION_README.md` for:
- Detailed parameter descriptions
- Advanced customization
- Multiple molecule workflows
- Animation generation
- Manual POV-Ray editing

## 🎬 Example for CO₂

```bash
# Quick preview
./matplotlib_3d_viz.py \
    --checkpoint checkpoints/co2_model \
    --structure co2.xyz \
    --show-charges \
    --output co2_preview.png

# High-quality multi-view
./ase_povray_viz.py \
    --checkpoint checkpoints/co2_model \
    --structure co2.xyz \
    --show-charges \
    --show-esp \
    --resolution high \
    --views "0x,0y,0z" "90x,0y,0z" "0x,90y,0z" \
    --output-dir ./co2_figures
```

## 💡 Tips

- **White background** is default (good for papers/slides)
- **Charge colors**: Red = +, Blue = −
- **ESP colors**: Red = repulsive, Blue = attractive
- **Multiple views** help understand 3D structure
- **Ultra resolution** for publication, medium for drafts

---

**Ready to visualize?** Start with the "High-Quality Figure" command above! 🚀

