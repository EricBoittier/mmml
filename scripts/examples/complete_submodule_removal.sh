#!/bin/bash
# Complete the dcmnet submodule removal process
# The .gitmodules and .git/config have already been updated

set -e

echo "Step 1: Removing .git file from mmml/mmml/dcmnet..."
if [ -f "mmml/mmml/dcmnet/.git" ]; then
    rm -f "mmml/mmml/dcmnet/.git"
    echo "  Removed mmml/mmml/dcmnet/.git"
else
    echo "  Already removed"
fi

echo "Step 2: Removing .git file from mmml/github/dcmnet (if exists)..."
if [ -f "mmml/github/dcmnet/.git" ]; then
    rm -f "mmml/github/dcmnet/.git"
    echo "  Removed mmml/github/dcmnet/.git"
else
    echo "  Not found or already removed"
fi

echo "Step 3: Removing submodule metadata from .git/modules..."
if [ -d ".git/modules/mmml/mmml/dcmnet" ]; then
    rm -rf ".git/modules/mmml/mmml/dcmnet"
    echo "  Removed .git/modules/mmml/mmml/dcmnet"
else
    echo "  Already removed"
fi

if [ -d ".git/modules/mmml/github/dcmnet" ]; then
    rm -rf ".git/modules/mmml/github/dcmnet"
    echo "  Removed .git/modules/mmml/github/dcmnet"
else
    echo "  Not found or already removed"
fi

echo "Step 4: Removing submodules from git index..."
git rm --cached -r mmml/mmml/dcmnet 2>/dev/null || echo "  mmml/mmml/dcmnet not in index or already removed"
git rm --cached -r mmml/github/dcmnet 2>/dev/null || echo "  mmml/github/dcmnet not in index or already removed"

echo "Step 5: Adding directories back as regular files..."
git add -f mmml/mmml/dcmnet
git add -f mmml/github/dcmnet
git add .gitmodules

echo ""
echo "✓ Submodule removal complete!"
echo "  The dcmnet code has been kept in:"
echo "    - mmml/mmml/dcmnet/"
echo "    - mmml/github/dcmnet/"
echo ""
echo "  These directories are now regular directories, not submodules."
echo ""
echo "Next steps:"
echo "  1. Review changes: git status"
echo "  2. Commit changes: git commit -m 'Remove dcmnet submodules, keep code'"

