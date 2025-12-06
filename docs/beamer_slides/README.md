# MMML Beamer Presentation

This directory contains a comprehensive LaTeX Beamer presentation showcasing the MMML CLI tools.

## Contents

- `mmml_presentation.tex` - Main presentation file (35+ slides)

## Topics Covered

1. **Introduction**
   - What is MMML?
   - Key features
   - Tool overview

2. **Data Preparation**
   - Data cleaning
   - Data exploration
   - Dataset splitting
   - Automatic padding removal

3. **Model Training**
   - Basic training
   - Training configuration
   - Joint PhysNet+DCMNet
   - Memory-mapped training
   - Multi-state training

4. **Model Evaluation**
   - Model inspection
   - Comprehensive evaluation
   - Training history visualization

5. **Model Deployment**
   - ASE calculator interface
   - Molecular dynamics
   - Vibrational analysis

6. **Complete Workflows**
   - Glycol example (end-to-end)
   - CO2 example (ESP prediction)
   - Best practices

7. **Summary**
   - Tool summary
   - Key innovations
   - Getting started
   - Documentation
   - Future directions

## Building the Presentation

### Requirements

- LaTeX distribution (TeX Live, MiKTeX, etc.)
- Beamer class
- Required packages: listings, xcolor, tikz, booktabs, fontawesome5

### Compilation

```bash
# Navigate to the slides directory
cd beamer_slides/

# Compile with pdflatex (run twice for proper references)
pdflatex mmml_presentation.tex
pdflatex mmml_presentation.tex

# Clean up auxiliary files
rm -f *.aux *.log *.nav *.out *.snm *.toc
```

### Quick compile script

```bash
#!/bin/bash
pdflatex mmml_presentation.tex && \
pdflatex mmml_presentation.tex && \
echo "✅ Presentation built successfully: mmml_presentation.pdf" && \
rm -f *.aux *.log *.nav *.out *.snm *.toc
```

## Features

- **Modern theme** - Madrid theme with custom styling
- **Code snippets** - Syntax-highlighted Bash and Python code
- **Diagrams** - TikZ diagrams for architecture and workflows
- **Tables** - Professional tables with booktabs
- **Icons** - FontAwesome icons for visual appeal
- **Examples** - Real-world examples from glycol and CO2 datasets

## Customization

### Changing the theme

Edit line 6 in `mmml_presentation.tex`:
```latex
\usetheme{Madrid}  % Try: Berlin, Copenhagen, Singapore, etc.
```

### Changing colors

Edit line 7:
```latex
\usecolortheme{default}  % Try: beaver, crane, dolphin, etc.
```

### Aspect ratio

Edit line 1:
```latex
\documentclass[aspectratio=169,10pt]{beamer}  % 16:9
% Or use aspectratio=43 for 4:3
```

## Adding Figures

To add figures from the examples folders:

1. Place figures in `beamer_slides/figures/` directory
2. Add to a slide:

```latex
\begin{frame}{My Figure}
\begin{center}
\includegraphics[width=0.8\textwidth]{figures/my_figure.png}
\end{center}
\end{frame}
```

## Tips

1. **Keep slides concise** - One main point per slide
2. **Use code sparingly** - Show only essential commands
3. **Add pauses** - Use `\pause` for incremental reveals
4. **Test animations** - Ensure transitions work smoothly
5. **Practice timing** - Aim for 1-2 minutes per slide

## Slide Count

- Total slides: 35+
- Estimated presentation time: 45-60 minutes
- Suitable for: Tutorials, workshops, seminars

## License

Same as MMML main project.

