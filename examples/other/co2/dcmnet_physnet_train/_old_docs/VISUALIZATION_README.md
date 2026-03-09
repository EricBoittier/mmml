# High-Quality Molecular Visualization Guide

This directory contains scripts for creating publication-quality molecular visualizations showing:
- **Ball-and-stick molecular structures**
- **Distributed charges as colored spheres**
- **Electrostatic potential (ESP) on VDW surfaces**

## Scripts Overview

### 1. `ase_povray_viz.py` ⭐ **Recommended**
High-quality POV-Ray rendering using ASE's native POV-Ray writer.

**Features:**
- Professional ball-and-stick rendering
- Colored charge spheres (red = positive, blue = negative)
- ESP mapped on molecular VDW surface
- Multiple viewing angles
- Resolution presets (low/medium/high/ultra)

**Requirements:**
- POV-Ray (see installation below)
- ASE, matplotlib, jax

**Example usage:**
```bash
# Basic visualization with charges
python ase_povray_viz.py \
    --checkpoint checkpoints/best_model \
    --structure co2.xyz \
    --output-dir ./figures \
    --show-charges \
    --resolution high

# With ESP surface
python ase_povray_viz.py \
    --checkpoint checkpoints/best_model \
    --structure co2.xyz \
    --output-dir ./figures \
    --show-charges \
    --show-esp \
    --resolution high \
    --views "0x,0y,0z" "90x,0y,0z" "0x,90y,0z"

# Ultra quality for publication
python ase_povray_viz.py \
    --checkpoint checkpoints/best_model \
    --structure molecule.xyz \
    --show-charges \
    --show-esp \
    --resolution ultra \
    --n-surface-points 10000
```

### 2. `povray_visualization.py`
Custom POV-Ray writer with more control over rendering parameters.

**Features:**
- Full control over camera, lighting, materials
- Custom bond detection
- Adjustable charge and atom sizes
- Multiple predefined views

**Example usage:**
```bash
python povray_visualization.py \
    --checkpoint checkpoints/best_model \
    --structure co2.xyz \
    --output-dir ./povray_custom \
    --show-charges \
    --show-esp \
    --views front side top \
    --width 3840 \
    --height 2160 \
    --quality 11
```

### 3. `matplotlib_3d_viz.py`
Quick 3D visualization using matplotlib (no POV-Ray needed).

**Features:**
- Fast preview without POV-Ray
- Interactive viewing (optional)
- Good for quick checks
- Lower quality than POV-Ray

**Example usage:**
```bash
# Quick preview
python matplotlib_3d_viz.py \
    --checkpoint checkpoints/best_model \
    --structure co2.xyz \
    --output preview.png \
    --show-charges

# Interactive exploration
python matplotlib_3d_viz.py \
    --checkpoint checkpoints/best_model \
    --structure co2.xyz \
    --show-charges \
    --show-esp \
    --interactive
```

## Installation

### POV-Ray Installation

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install povray
```

**macOS:**
```bash
brew install povray
```

**Windows:**
Download from http://www.povray.org/download/

**Verify installation:**
```bash
povray --version
```

### Python Dependencies
All required packages should already be installed if you've set up the training environment:
- `ase` (Atomic Simulation Environment)
- `matplotlib`
- `numpy`
- `jax`
- `scipy`

## Understanding the Visualizations

### Ball-and-Stick Representation
- **Atoms**: Spheres colored by element (CPK or jmol coloring)
  - Carbon (C): Gray
  - Hydrogen (H): White
  - Oxygen (O): Red
  - Nitrogen (N): Blue
- **Bonds**: Cylinders connecting nearby atoms

### Distributed Charges
- **Colored spheres** positioned off-atom centers
- **Color scale**: Red = positive charge, Blue = negative charge
- **Size**: Proportional to charge magnitude
- **Transparency**: Allows viewing of underlying structure

### ESP Surface
- Computed on **1.4× VDW radii** surface
- **Color scale**: Red = positive potential, Blue = negative potential
- Shows regions where test charges experience repulsion/attraction
- Useful for:
  - Predicting interaction sites
  - Understanding reactivity
  - Visualizing charge distribution effects

## Output Files

Each script produces:

1. **POV-Ray scene files** (`.pov`): Text description of scene
2. **Rendered images** (`.png`): Final high-quality images
3. **Colorbar legends**: Separate files explaining color scales
   - `colorbar_charges.png`: Charge scale
   - `colorbar_esp.png`: ESP scale

## Tips for Best Results

### For Publications
1. Use `ase_povray_viz.py` with `--resolution ultra`
2. Enable both `--show-charges` and `--show-esp`
3. Generate multiple views: front, side, top
4. Adjust `--n-surface-points 10000` for smoother ESP surfaces
5. Fine-tune `--charge-radius` and `--transparency` for clarity

### For Presentations
1. Use `--resolution high` (faster rendering)
2. Consider showing charges only (clearer visualization)
3. Use white background (default) for slides

### For Quick Checks
1. Use `matplotlib_3d_viz.py --interactive`
2. Rotate view manually to find best angle
3. Note elevation and azimuth angles
4. Use those angles for final POV-Ray render

## Customization Options

### Common Parameters

| Parameter | Description | Default | Recommended Range |
|-----------|-------------|---------|-------------------|
| `--charge-radius` | Base size of charge spheres | 0.15 | 0.1 - 0.3 |
| `--transparency` | Charge sphere transparency | 0.5 | 0.3 - 0.7 |
| `--n-surface-points` | ESP surface resolution | 5000 | 2000 - 15000 |
| `--atom-scale` | Atom size multiplier | 0.4 | 0.3 - 0.6 |
| `--bond-radius` | Bond cylinder radius | 0.15 | 0.1 - 0.25 |

### View Angles (ASE Rotation String)

- `"0x,0y,0z"`: Front view (no rotation)
- `"90x,0y,0z"`: Side view (rotated around X)
- `"0x,90y,0z"`: Top view (rotated around Y)
- `"45x,45y,0z"`: Diagonal view
- Custom: `"{angle}x,{angle}y,{angle}z"`

## Troubleshooting

### POV-Ray Not Found
**Error:** `FileNotFoundError: povray`

**Solution:** Install POV-Ray (see installation section above)

### Out of Memory
**Error:** Memory error during ESP surface computation

**Solutions:**
- Reduce `--n-surface-points` (e.g., 2000)
- Use matplotlib version for preview

### Rendering Takes Too Long
**Solutions:**
- Use `--resolution medium` or `--resolution low`
- Reduce `--quality` parameter (e.g., `--quality 9`)
- Disable `--show-esp` if not needed
- Reduce `--n-surface-points`

### Charges Too Small/Large
**Solutions:**
- Adjust `--charge-radius` (increase for larger spheres)
- Check charge values in output (should be reasonable)
- Verify model checkpoint is correct

### ESP Surface Looks Noisy
**Solutions:**
- Increase `--n-surface-points` (e.g., 10000)
- Check model predictions (might be actual noise)
- Adjust colorbar range manually if needed

## Example Workflows

### Workflow 1: Quick Preview → Final Render
```bash
# Step 1: Quick preview with matplotlib
python matplotlib_3d_viz.py \
    --checkpoint checkpoints/model \
    --structure molecule.xyz \
    --show-charges \
    --interactive

# Step 2: Final high-quality render
python ase_povray_viz.py \
    --checkpoint checkpoints/model \
    --structure molecule.xyz \
    --show-charges \
    --show-esp \
    --resolution high \
    --output-dir final_figures
```

### Workflow 2: Multiple Molecules
```bash
# Create figures for multiple structures
for xyz in structures/*.xyz; do
    name=$(basename "$xyz" .xyz)
    python ase_povray_viz.py \
        --checkpoint checkpoints/model \
        --structure "$xyz" \
        --show-charges \
        --resolution high \
        --output-dir "figures_${name}"
done
```

### Workflow 3: Animation Frames
```bash
# Generate frames for rotation animation
for angle in {0..360..10}; do
    python ase_povray_viz.py \
        --checkpoint checkpoints/model \
        --structure molecule.xyz \
        --show-charges \
        --views "${angle}x,0y,0z" \
        --resolution medium \
        --output-dir animation_frames
done

# Combine with ffmpeg (if desired)
# ffmpeg -framerate 30 -pattern_type glob -i 'animation_frames/*.png' \
#        -c:v libx264 -pix_fmt yuv420p animation.mp4
```

## Color Schemes

### Charge Coloring
- **Colormap**: RdBu_r (Red-Blue reversed)
- **Positive charges**: Red tones
- **Negative charges**: Blue tones
- **Zero charge**: White/gray

### ESP Coloring
- **Colormap**: RdBu_r
- **Positive potential**: Red (repels positive charges)
- **Negative potential**: Blue (attracts positive charges)
- **Zero potential**: White/gray

## Performance Notes

Typical rendering times (Intel i7, single core):

| Resolution | Image Size | Time per View |
|------------|------------|---------------|
| Low        | 800×600    | ~10 seconds   |
| Medium     | 1600×1200  | ~30 seconds   |
| High       | 2400×1800  | ~60 seconds   |
| Ultra      | 3840×2160  | ~120 seconds  |

Add ~50% more time when `--show-esp` is enabled.

## Advanced: Editing POV-Ray Scenes

The `.pov` files are plain text and can be edited manually:

1. Generate initial scene with script
2. Open `.pov` file in text editor
3. Modify lighting, camera, materials, etc.
4. Re-render manually:
   ```bash
   povray +Iscene.pov +Ooutput.png +W2400 +H1800 +Q11 +A
   ```

Useful POV-Ray resources:
- Official documentation: http://www.povray.org/documentation/
- Tutorial: http://www.povray.org/documentation/3.7.0/t2_0.html

## Citation

If you use these visualization scripts in published work, please cite:
- ASE: https://doi.org/10.1088/1361-648X/aa680e
- POV-Ray: http://www.povray.org/

## Support

For issues or questions:
1. Check this README
2. Review error messages carefully
3. Try matplotlib version first for debugging
4. Verify model checkpoint and structure file are valid

