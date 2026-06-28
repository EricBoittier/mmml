
original_wd=$PWD
# Resolve mmml root (parent of setup/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
MMML_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SCRIPT_DIR"
# Optional legacy override (mmml auto-discovers setup/charmm when libcharmm is built).
chmhome="export CHARMM_HOME=$PWD/charmm"
chmlib="export CHARMM_LIB_DIR=$PWD/charmm"
echo "$chmhome" > "$MMML_ROOT/CHARMMSETUP"
echo "$chmlib" >> "$MMML_ROOT/CHARMMSETUP"
echo "Wrote optional $MMML_ROOT/CHARMMSETUP (not required for import or pytest)"
cat "$MMML_ROOT/CHARMMSETUP"
