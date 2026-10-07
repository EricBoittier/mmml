
original_wd=$PWD
# Resolve karml root (parent of setup/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
KARML_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SCRIPT_DIR"
# Optional legacy override (karml auto-discovers setup/charmm when libcharmm is built).
chmhome="export CHARMM_HOME=$PWD/charmm"
chmlib="export CHARMM_LIB_DIR=$PWD/charmm"
echo "$chmhome" > "$KARML_ROOT/CHARMMSETUP"
echo "$chmlib" >> "$KARML_ROOT/CHARMMSETUP"
echo "Wrote optional $KARML_ROOT/CHARMMSETUP (not required for import or pytest)"
cat "$KARML_ROOT/CHARMMSETUP"
