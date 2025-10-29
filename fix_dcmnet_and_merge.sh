#!/bin/bash
# Fix dcmnet submodule issues and prepare for merge

set -e

echo "=== Step 1: Add untracked mmml/dcmnet files to git ==="
git add -f mmml/dcmnet/
echo "✓ Added mmml/dcmnet/"

echo ""
echo "=== Step 2: Complete submodule removal for mmml/mmml/dcmnet ==="

# Remove .git file if it still exists
if [ -f "mmml/mmml/dcmnet/.git" ]; then
    rm -f "mmml/mmml/dcmnet/.git"
    echo "✓ Removed mmml/mmml/dcmnet/.git"
fi

# Remove submodule metadata
if [ -d ".git/modules/mmml/mmml/dcmnet" ]; then
    rm -rf ".git/modules/mmml/mmml/dcmnet"
    echo "✓ Removed .git/modules/mmml/mmml/dcmnet"
fi

# Remove from git index and add back as regular directory
git rm --cached -r mmml/mmml/dcmnet 2>/dev/null || echo "  (already removed from index)"
git add -f mmml/mmml/dcmnet/
echo "✓ Added mmml/mmml/dcmnet as regular directory"

echo ""
echo "=== Step 3: Complete submodule removal for mmml/github/dcmnet ==="

# Remove .git file if it exists
if [ -f "mmml/github/dcmnet/.git" ]; then
    rm -f "mmml/github/dcmnet/.git"
    echo "✓ Removed mmml/github/dcmnet/.git"
fi

# Remove submodule metadata
if [ -d ".git/modules/mmml/github/dcmnet" ]; then
    rm -rf ".git/modules/mmml/github/dcmnet"
    echo "✓ Removed .git/modules/mmml/github/dcmnet"
fi

# Remove from git index and add back as regular directory
git rm --cached -r mmml/github/dcmnet 2>/dev/null || echo "  (already removed from index)"
git add -f mmml/github/dcmnet/
echo "✓ Added mmml/github/dcmnet as regular directory"

echo ""
echo "=== Step 4: Stage configuration changes ==="
git add .gitmodules .git/config
echo "✓ Staged .gitmodules and .git/config changes"

echo ""
echo "=== Step 5: Check status ==="
git status

echo ""
echo "============================================"
echo "✓ All dcmnet directories are now tracked!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Commit these changes:"
echo "     git commit -m 'Remove dcmnet submodules, convert to regular directories'"
echo ""
echo "  2. Then retry your merge:"
echo "     git merge <branch>"
echo "     or"
echo "     git pull"

