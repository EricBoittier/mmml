#!/bin/csh -f
#
# install_cr.com
#
#-------------------------------------------------------------------------------
#
# Enviromental variables defined by the script install.com of CHARMM: 
#
#       chmroot = CHARMM programm directoy
#
#
# Enviromental variables defined by hand by the user following the CHARMMRATE
# installation steps
#
#       pr     = POLYRATE programm directory
#       crate  = CRATE utility directory
#
#-------------------------------------------------------------------------------
#
# Definition of variables 'polysrc' and 'prate' 
#
#       polysrc = directory where the POLYRATE source code files are stored
#       prate   = directory where the modified POLYRATE source code files
#                 will be stored for the CHARMMRATE program
#
  set polysrc = $pr/src
  set prate = $chmroot/source/prate
#
#
# List of the POLYRATE source code files 
#
  cd $polysrc
  ls -1 *.f  > $prate/list
  cd $prate
#
#
# Changes to be done on each one of the POLYTATE source code files
#
  foreach file (` cat list `)
#
#
#   The files 'main.f' and 'hooks.f' of POLYRATE were replaced by
#   'maino.src' and 'hooks.src'. Both new files are into the CRATE
#   utility.
#
    if ( $file == 'hooks.f' || $file == 'main.f' ) goto 100
#
#
#   Copy of the POLYRATE source code files 
#
    cp $polysrc/$file .
#
#
#   All tab characters and characters beyond column 72 will be removed
#   from each one of the copies of the files
#
    expand $file  |  awk  '{print substr ($0,1,72)}' > $file.2
    mv $file.2 $file
#
#
#   The name and the extension of each file will be identified
#
    set name = `echo $file | awk -F. '{print $1}' `
    set exte = `echo $file | awk -F. '{print $2}' `
#
#
#   The extension 'f' of the source code files of POLYRATE will be
#   replaced by 'src' 
#
    if ( $exte == 'f' ) mv $file $name.src
#
#
#   Changes that have to be done over the following source code files 
#   of POLYRATE: ef.src, energetics.src, fromblas.src, interface.src, 
#   polyag.src, polyrr.src, polysz.src
#
#
#   1.- Changes on 'interface.src'
#
    if ( $name == 'interface' ) then
#
#      The option *finish will be added to terminate the command stream for 
#      POLYRATE from CHARMM
#
       set nlin = `grep -n 'call rrate(string,iend,istrt)' $name.src | awk -F: '{print $1}'`
       sed -n "1,$nlin p" $name.src > $name.src.cab
       @ nlin = ( $nlin + 1 )
       sed -n "$nlin,$ p" $name.src > $name.src.fin
       echo 'c   Stop input stream from CHARMM file.' >> $name.src.cab
       echo "         else if (string(j:j+5).eq.'finish' .or. string(j:j+2).eq.'end')" >> $name.src.cab
       echo '     &        then' >> $name.src.cab
       echo '               go to 1201' >> $name.src.cab
       sed 's/      call intab$/1201  call intab/p' $name.src.fin >> $name.src.cab
       mv $name.src.cab $name.src
#
#      The SPECBASIS options will be removed since it will not be available
#  
       set nlin = `grep -n 'c SPECBASIS' $name.src | awk -F: '{print $1}'`
       sed -n "1,$nlin p" $name.src > $name.src.cab
       @ nlin = ( $nlin + 3 ) 
       sed -n "$nlin,$ p" $name.src >> $name.src.cab
#
#      The function 'case' will be changed to 'casito' to avoid problems 
#      during the compilation
#
       sed 's/case (/casito (/p' $name.src.cab > $name.src
       sed 's/ case(/ casito(/p' $name.src > $name.src.cab
       sed 's/,case/,casito/p' $name.src.cab > $name.src
       sed 's/, case/, casito/p' $name.src > $name.src.cab
       sed 's/case  /casito  /p' $name.src.cab > $name.src
       sed 's/ case$/ casito/p' $name.src > $name.src.cab
#
#      The subroutine 'readic' will be replaced by 'cr_readic' 
#
       sed 's/readic/cr_readic/p' $name.src.cab > $name.src
       sed 's/READIC/CR_READIC/p' $name.src > $name.src.cab
       mv $name.src.cab $name.src
#
#      Files that will not be use anymore will be deleted here
#
       /bin/rm -f $name.src.fin
#
    endif
#
#
#   2.- Changes on 'polyag.src'
#
    if ( $name == 'polyag' ) then
#
#      The subroutine 'readic' will be replaced by 'cr_readic' 
#
       sed 's/READIC/CR_READIC/p' $name.src > $name.src.cab
       sed 's/readic/cr_readic/p' $name.src.cab > $name.src
#
#      The subroutine 'erf' will be replaced by 'cr_erf' 
#
       sed 's/ERF =/CR_ERF =/p' $name.src > $name.src.cab
       sed 's/= ERF/= CR_ERF/p' $name.src.cab > $name.src
       sed 's/-ERF/-CR_ERF/p' $name.src > $name.src.cab
       sed 's/erf (/cr_erf (/p' $name.src.cab > $name.src
#
#      Files that will not be use anymore will be deleted here
#
       rm  -f $name.src.cab 
#
    endif
#
#
#   3.- Changes on 'polysz.src'
#
    if ( $name == 'polysz' ) then
#
#      The subroutine 'erf' will be replaced by 'cr_erf' 
#
       sed 's/ ERF/ CR_ERF/p' $name.src > $name.src.cab
       mv $name.src.cab $name.src 
#
    endif
#
#
#   4.- Changes on 'energetics.src'
#
    if ( $name == 'energetics' ) then
#
#      The subroutine 'deriv2' will be deactivated 
#
       set nlin = `grep -ni 'ne deriv2' $name.src | awk -F: '{print $1}'`
       @ nlin = ( $nlin - 1 )
       sed -n "1,$nlin p" $name.src > $name.src.cab
       sed -n "$nlin,$ p" $name.src > $name.src.fin
       set nlin = `grep -ni 'return' $name.src.fin | head -1 | awk -F: '{print $1}'`
       @ nlin = ( $nlin + 2 ) 
       sed -n "$nlin,$ p" $name.src.fin >> $name.src.cab
       mv $name.src.cab $name.src 
#
#      The subroutine 'deriv2' will be deactivated 
#
       set nlin = `grep -ni 'ne derv24' $name.src | awk -F: '{print $1}'`
       @ nlin = ( $nlin - 1 ) 
       sed -n "1,$nlin p" $name.src > $name.src.cab
       sed -n "$nlin,$ p" $name.src > $name.src.fin
       set nlin = `grep -ni 'return' $name.src.fin | head -1 | awk -F: '{print $1}'`
       @ nlin = ( $nlin + 2 ) 
       sed -n "$nlin,$ p" $name.src.fin >> $name.src.cab
       mv $name.src.cab $name.src 
#
#      Files that will not be use anymore will be deleted here
#
       rm -f $name.src.fin
#
    endif
#
#
#   5.- Changes on 'polyrr.src'
#
    if ( $name == 'polyrr' ) then
#
#      All 'tqlrat' will be replaced by 'cr_tqlrat'
#
       sed 's/TQLRAT/CR_TQLRAT/p' $name.src > $name.src.cab
#
#      All 'tql2' will be replaced by 'cr_tql2'
#
       sed 's/TQL2/CR_TQL2/p' $name.src.cab > $name.src
#
#      Files that will not be use anymore will be deleted here
#
       rm -f $name.src.cab
#
    endif
#
#
#   6.- Changes on 'fromblas.src'
#
    if ( $name == 'fromblas' ) then
#
#      All 'tqlrat' will be replaced by 'cr_tqlrat'
#
       sed 's/tqlrat/cr_tqlrat/p' $name.src > $name.src.cab
#
#      All 'tql2' will be replaced by 'cr_tql2'
#
       sed 's/tql2/cr_tql2/p' $name.src.cab > $name.src
#
#      Files that will not be use anymore will be deleted here
#
       rm -f $name.src.cab
#
    endif
#
#
#   7.- Changes on 'ef.src'
#
    if ( $name == 'ef' ) then
#
#      All 'tqlrat' will be replaced by 'cr_tqlrat'
#
       sed 's/tqlrat/cr_tqlrat/p' $name.src > $name.src.cab
#
#      All 'tql2' will be replaced by 'cr_tql2'
#
       sed 's/tql2/cr_tql2/p' $name.src.cab > $name.src
#
#      Files that will not be use anymore will be deleted here
#
       rm -f $name.src.cab
#
    endif
#
#
# 
  100:
#
#
# End of the 'foreach'  
#
  end
#
#
# Files that will not be use anymore will be deleted here
#
  /bin/rm $prate/list
#
#
#
# Copy of the files 'maino.src' and 'hooks.src' from the CRATE utility
# into the prate directory
#
  cp $crate/maino.src .
  cp $crate/hooks.src .
#
#
# Copy of the POLYRATE common blocks into the prate directory
#
  expand $crate/param.inc | awk  '{print substr ($0,1,72)}' > param.inc
  expand $polysrc/percon.inc | awk  '{print substr ($0,1,72)}' > percon.inc
  expand $polysrc/common.inc | awk  '{print substr ($0,1,72)}' > common.inc
#
#
# Tha..tha..that's all folk...
#
  exit

