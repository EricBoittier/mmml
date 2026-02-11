
original_wd=$PWD
# Resolve mmml root (parent of setup/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
MMML_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# Set environment variables
chmhome="export CHARMM_HOME=$PWD/charmm"
chmlib="export CHARMM_LIB_DIR=$PWD/charmm"
echo "$chmhome" > "$MMML_ROOT/CHARMMSETUP"
echo "$chmlib" >> "$MMML_ROOT/CHARMMSETUP"
cat "$MMML_ROOT/CHARMMSETUP"
source "$MMML_ROOT/CHARMMSETUP"
