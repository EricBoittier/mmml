original_wd=$PWD
# Resolve karml root (parent of setup/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
KARML_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$SCRIPT_DIR"

# Only extract if 'charmm' directory does not exist
if [ ! -d "charmm" ]; then
	echo "unzipping"
    	tar -xf charmm.tar.xz
fi

# Partial extracts may lack CMakeLists.txt (needed by scripts/rebuild_charmm_mlpot.sh).
if [ ! -f "charmm/CMakeLists.txt" ]; then
	echo "Restoring charmm/CMakeLists.txt from charmm.tar.xz"
	tar -xf charmm.tar.xz charmm/CMakeLists.txt charmm/tool/cmake
fi

# Optional legacy override (karml auto-discovers setup/charmm when libcharmm is built).
chmhome="export CHARMM_HOME=$PWD/charmm"
chmlib="export CHARMM_LIB_DIR=$PWD/charmm"
echo "$chmhome" > "$KARML_ROOT/CHARMMSETUP"
echo "$chmlib" >> "$KARML_ROOT/CHARMMSETUP"
echo "Wrote optional $KARML_ROOT/CHARMMSETUP (not required for import or pytest)"
cat "$KARML_ROOT/CHARMMSETUP"

# uv must run from project root (where pyproject.toml lives)
cd "$KARML_ROOT"
which uv
if [ $? -ne 0 ]; then
    echo "uv not found, installing uv"
    wget -qO- https://astral.sh/uv/install.sh | sh
else
    echo "uv found"
fi

uv sync
source .venv/bin/activate
uv sync
echo "venv activated"
echo "venv path: $VIRTUAL_ENV"
echo "venv python path: $(which python)"
echo "PyCHARMM paths auto-discover setup/charmm after libcharmm is built (see scripts/rebuild_charmm_mlpot.sh)"
echo "Setup complete"

cd "$original_wd"
