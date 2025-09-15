# Optional: ensure the same toolchain RTD uses
asdf global python mambaforge-22.9.0-3

# Create the docs env (remove existing if needed)
mamba env remove -n latest -y || true
mamba env create -n latest -f docs/requirements.yaml

# Activate and build the docs
conda activate latest
sphinx-build -b html docs docs/_build/html
# or: make -C docs html

# Open the result
xdg-open docs/_build/html/index.html || open docs/_build/html/index.html
