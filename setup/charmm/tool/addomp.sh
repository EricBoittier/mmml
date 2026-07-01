#!/bin/bash
for reg in \
           "SHLG70" "IJGNRL" "MISC"   \
           "SHLINF" "ERIOUT" "SHLNOS" "INTDEX" \
           "ROOT"   "SETINT" "XYZ"    "DENS" \
           "FLIPS"  "GEOMPQ" "GOUT"   "POPOUT" \
           "SHLLFO" "SHLSPD" "JMSGYH" "FQ04" "FQ08" \
           "KI2" "KI3" "KI4" "MAXC" \
           "ERIPRM"  \
           "GSPG80" "JMSG80" "SHLLMN" "INDD80" \
           "DSHLNO" "DERPAR" "DERSHL" "DERSKP" "DERINV" \
           "SHLEQU" "SHLTYP" "SHLGNM" "SHLXPN" "SHLNUM" \
           "DERMEM" "GRAD"   "NLRCF " "DFTPAR" \
           "TMVALS" \
           "DLT"
do

    sed -i " /^      \+COMMON *\/${reg} *\//I{
            :cycle
            n
            /^     \S.*/{b cycle}
            iC\$omp threadprivate(\/${reg}\/)
        } " $1
done
