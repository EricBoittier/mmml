# Beamer Presentation Figures Guide

## 📁 Location
All figures are in: `beamer_slides/figures_keep/`

## 🖼️ Included Figures (11 total)

### Glycol Training Analysis (2 figures)

1. **glycol_training_metrics.png** (782 KB)
   - 7-panel training analysis
   - Loss curves (log scale)
   - Energy/Forces/Dipole MAE (kcal/mol)
   - Learning rate progression
   - Convergence analysis
   - **Slide:** ~36

2. **glycol_runs_comparison.png** (650 KB)
   - Compares 2 glycol training runs
   - Run 1 vs Run 2 on same axes
   - Shows Run 2 is 20% better for forces
   - **Slide:** ~37

### CO2 ESP & Charge Visualization (5 figures)

3. **esp_spheres_co2.png** (7.5 MB)
   - Electrostatic potential on molecular surface
   - High-resolution 3D visualization
   - **Slide:** ~40

4. **view_1_45x_30y_0z_molecule.png** (146 KB)
   - CO2 molecular structure with distributed charges
   - 3D isometric view
   - **Slide:** ~41

5. **charges_2d_projections.png** (175 KB)
   - XY, XZ, YZ projections of charge distribution
   - Shows spatial arrangement
   - **Slide:** ~42

6. **charge_evolution.png** (756 KB)
   - How charges change during molecular vibration
   - Time series visualization
   - **Slide:** ~43

7. **charges_3d_evolution.png** (515 KB)
   - 3D visualization of charge redistribution
   - Dynamic charge behavior
   - **Slide:** ~44

### CO2 Vibrational Spectroscopy (2 figures)

8. **combined_spectrum.png** (627 KB)
   - Combined IR and Raman spectrum
   - Full vibrational fingerprint
   - **Slide:** ~45

9. **raman_multiwavelength.png** (412 KB)
   - Raman spectra at different excitation wavelengths
   - Wavelength-dependent analysis
   - **Slide:** ~46

### Model Comparison Analysis (2 figures)

10. **performance_comparison.png** (229 KB)
    - Detailed performance metrics comparison
    - DCMNet vs NonEquivariant
    - **Slide:** ~53

11. **efficiency_comparison.png** (132 KB)
    - Speed vs accuracy trade-offs
    - Computational efficiency analysis
    - **Slide:** ~54

## 📊 Presentation Updates

### Previous Version
- 46 slides
- 1.9 MB
- Comparison plots from analysis/

### Current Version
- **57 slides** (added 11)
- **13 MB** (includes high-res figures)
- All figures organized in figures_keep/

### New Sections Enhanced

1. **Complete Workflows** (slides 35-50)
   - Now includes actual training results
   - Glycol comparison between runs
   - CO2 comprehensive visualizations

2. **Model Deployment** (slides 45-46)
   - Added spectroscopy examples
   - IR and Raman spectra

3. **Architecture Comparison** (slides 53-54)
   - Added performance/efficiency comparison figures

## 🎨 Figure Quality

All figures are:
- ✅ High resolution (300 DPI for plots, original res for visualizations)
- ✅ Colorblind-friendly (where applicable)
- ✅ Professional quality
- ✅ Ready for projection and print

## 📏 Figure Sizes in Presentation

- **Full width:** 0.95\textwidth (most plots)
- **Standard:** 0.85-0.9\textwidth (comparisons)
- **Full height:** 0.8\textheight (ESP spheres)

## 🔍 Figure Details

### Glycol Figures
- **Units:** kcal/mol (chemistry standard)
- **Loss scale:** Logarithmic (shows full improvement)
- **Annotations:** None (clean plots)
- **Style:** Okabe-Ito colorblind-friendly palette

### CO2 Figures
- **ESP visualization:** Full 3D rendering with surface mapping
- **Charge projections:** Multi-panel 2D views
- **Charge evolution:** Time-dependent analysis
- **Spectra:** IR and Raman with experimental comparison

### Comparison Figures
- **Performance:** Bar charts and scatter plots
- **Efficiency:** Speed vs accuracy trade-offs
- **Colors:** Blue (DCMNet) vs Orange (NonEq)

## 📝 Usage in Presentation

### Glycol Example Section
```latex
\begin{frame}{Glycol Training Results}
\includegraphics[width=0.95\textwidth]{figures_keep/glycol_training_metrics.png}
\end{frame}

\begin{frame}{Glycol Training Comparison}
\includegraphics[width=0.95\textwidth]{figures_keep/glycol_runs_comparison.png}
\end{frame}
```

### CO2 ESP Section
```latex
\begin{frame}{CO2: ESP Visualization}
\includegraphics[height=0.8\textheight]{figures_keep/esp_spheres_co2.png}
\end{frame}
```

### Spectroscopy Section
```latex
\begin{frame}{CO2: Combined Spectrum}
\includegraphics[width=0.9\textwidth]{figures_keep/combined_spectrum.png}
\end{frame}
```

## 🗂️ File Organization

```
beamer_slides/
├── mmml_presentation.tex          (LaTeX source)
├── mmml_presentation.pdf          (13 MB, 57 slides)
├── compile.sh                     (Build script)
├── figures/                       (Comparison plots, 8 files)
│   └── *.png (from equivariant comparison)
└── figures_keep/                  (Main figures, 11 files)
    ├── glycol_training_metrics.png
    ├── glycol_runs_comparison.png
    ├── esp_spheres_co2.png
    ├── view_1_45x_30y_0z_molecule.png
    ├── charges_2d_projections.png
    ├── charge_evolution.png
    ├── charges_3d_evolution.png
    ├── combined_spectrum.png
    ├── raman_multiwavelength.png
    ├── performance_comparison.png
    └── efficiency_comparison.png
```

## 💡 Tips

### To add more figures:
1. Copy to `figures_keep/`
2. Add to .tex file:
   ```latex
   \begin{frame}{Title}
   \includegraphics[width=0.9\textwidth]{figures_keep/your_figure.png}
   \end{frame}
   ```
3. Run `bash compile.sh`

### To adjust figure size:
- `width=0.9\textwidth` - 90% of slide width
- `height=0.8\textheight` - 80% of slide height
- Can use both: `[width=0.8\textwidth,height=0.7\textheight,keepaspectratio]`

### Image formats supported:
- PNG (recommended)
- JPG
- PDF (vector graphics)

## ✅ Checklist

- [x] All 11 figures moved to figures_keep/
- [x] Glycol training results added (2 slides)
- [x] CO2 visualizations added (5 slides)
- [x] CO2 spectroscopy added (2 slides)
- [x] Performance comparisons added (2 slides)
- [x] Presentation compiled successfully
- [x] 57 slides total
- [x] 13 MB final size

## 📖 Related Documentation

- **PRESENTATION_SUMMARY.md** - Complete slide breakdown
- **FINAL_SUMMARY.md** - Overall presentation summary
- **COLOR_ACCESSIBILITY.md** - Color scheme details
- **README.md** - Build instructions

---

**Status:** ✅ **All figures integrated into presentation!**

The presentation now includes comprehensive visualizations for:
- Glycol training analysis
- CO2 ESP predictions
- Distributed charge dynamics
- Vibrational spectroscopy
- Model performance comparisons

