#!/bin/bash
# Compile MMML Beamer Presentation
# This script compiles the presentation and cleans up auxiliary files

echo "🔨 Compiling MMML Beamer Presentation..."
echo "========================================"

# First compilation
echo "📄 First pass..."
pdflatex -interaction=nonstopmode mmml_presentation.tex > compile_pass1.log 2>&1

# Second compilation (for references and TOC)
echo "📄 Second pass..."
pdflatex -interaction=nonstopmode mmml_presentation.tex > compile_pass2.log 2>&1

# Check if PDF was created
if [ -f "mmml_presentation.pdf" ]; then
    echo ""
    echo "✅ Compilation successful!"
    echo "📊 PDF Statistics:"
    pdfinfo mmml_presentation.pdf 2>/dev/null | grep -E "(Pages|File size|PDF version)"
    ls -lh mmml_presentation.pdf
    echo ""
    
    # Clean up auxiliary files
    echo "🧹 Cleaning up auxiliary files..."
    rm -f *.aux *.log *.nav *.out *.snm *.toc *.vrb
    echo "✅ Cleanup complete!"
    echo ""
    echo "📁 Output: mmml_presentation.pdf"
    echo ""
    echo "🎉 Presentation ready to use!"
else
    echo ""
    echo "❌ Compilation failed!"
    echo "Check compile_pass2.log for details:"
    tail -30 compile_pass2.log
    exit 1
fi

