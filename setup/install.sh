original_wd=$PWD
# Resolve mmml root (parent of setup/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
MMML_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$SCRIPT_DIR"

# Only extract if 'charmm' directory does not exist
if [ ! -d "charmm" ]; then
	echo "unzipping"
    	tar -xf charmm.tar.xz
fi

# Set environment variables
chmhome="export CHARMM_HOME=$PWD/charmm"
chmlib="export CHARMM_LIB_DIR=$PWD/charmm"
echo "$chmhome" > "$MMML_ROOT/CHARMMSETUP"
echo "$chmlib" >> "$MMML_ROOT/CHARMMSETUP"
cat "$MMML_ROOT/CHARMMSETUP"
source "$MMML_ROOT/CHARMMSETUP"

# uv must run from project root (where pyproject.toml lives)
cd "$MMML_ROOT"
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
UV_ENV_FILE="$MMML_ROOT/CHARMMSETUP"
echo "UV_ENV_FILE: $UV_ENV_FILE"
source "$UV_ENV_FILE"
echo "Setup complete"

cd "$original_wd"
