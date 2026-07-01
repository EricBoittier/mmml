#!/bin/csh -f
# setmk.com
#-----------------------------------------------------------------------
# 10-Feb-94 able to produce single module makefile
# 05-Feb-94 for c24 developments
#           c22 replaced by chm
# 01-Jan-92 generates CHARMM Module Makefiles
#           use Steve Fleischman's makemod shell script
# 
# Inquiries to Youngdo Won, chmgr@tammy.harvard.edu (617)495-1782
#-----------------------------------------------------------------------
#
if ( $#argv == 0 ) then
  echo 
  echo " Usage: setmk.com host-machine-type"
  echo 
  echo "        host-machine-type is one of the following."
  echo 
  echo "               alpha     DEC 3000 AXP 500"
  echo "               alphamp   DEC Alpha-MP parallel computers"
  echo "               altix     SGI Itanium Altix series (64 bit)"
  echo "               em64t     ifort compilers on em64t/xeon "
  echo "               gnu       GNU compilers"
  echo "               g95       G95 compiler"
  echo "               hal       HAL work stations"
  echo "               hpitanium HP-UX Itaniums"
  echo "               hpux      HP Apollo 9000 Series 700"
  echo "               ibmaix    IBM AIX not parallel"
  echo "               ibmaixmp  IBM AIX parallel"
  echo "               ibmlnxmp  IBM GNU/Linux parallel"
  echo "               itanium   INTEL Itanium 2 using ifort compiler"
  echo "               osx       Mac OSX machines"
  echo "               sgi       SGI IRIS series (32 bit)"
  echo "               sgi64     SGI IRIS series (64 bit)"
  echo "               sun       SUN work stations (32-bit)"
  echo "               sunmpi    SUN work stations running SUN MPI" 
  echo "               sun64     SUN work stations (64-bit)"
  echo "               t3e       Cray T3E MPP"
  echo
  exit
else
  set chm_host = $argv[1]
endif
#
alias pwd "pwd | sed -e 's;/tmp_mnt;;'"
#
# set up CHARMM environment variables
# if not previously set, the current working directory is assumed to
#   be $chmtool.
  setenv chmhost  $chm_host
  setenv chmroot  `cd ..;pwd | sed -e 's;/tmp_mnt;;'`
  setenv chmbuild $chmroot/build/$chm_host
  setenv chmsrc   $chmroot/source
  setenv chmtool  $chmroot/tool
  set path = ($path $chmtool)
if (! -e $chmbuild ) mkdir $chmbuild
#
set makemod = $chmtool/makemod.pl
if (! -e $makemod ) then
  echo " "
  echo " setmk.com> $makemod is not found."
  echo "            can not proceed to set up module makefiles."
  exit 1
endif
#
# build a list of MODULE locations, for establishing dependencies
# list used by finduse.csh, which is called by mainmake and makemod
# F95 conversion project Oct2008 rvenable
echo "reading module declarations"
( chdir $chmsrc; grep -i '^ *module ' */*.F90 | grep -v -i ' procedure ' \
  | sed 's/:/ /' | awk '{print $2, $3, $1}' \
  | grep -i '^module' | sort -i -k2 ) > $chmtool/modules.txt
if (-f $chmtool/tsort.inp) rm $chmtool/tsort.inp
#
# process each module at a time
#
if ($#argv == 1) then
  set mdlist = `ls $chmsrc | grep -v -e CVS -e fcm`
else
  shift
  set mdlist = ($argv)
endif
#
pushd $chmsrc >>! /dev/null
set rc = 0
set problems = ()
foreach module ( $mdlist )
  if (! -d $module) continue
  pushd $module >>! /dev/null
  $makemod -n $module `pwd` $chmsrc $module.mk
  if ($status != 0) then
    set rc = 1
    set problems = ($problems $module.mk)
  endif
  if (-f $module.mk && ! -z $module.mk) then
    diff -qw $chmbuild/$module.mk $module.mk >& /dev/null
    if ($status == 0) then
      if ($#mdlist < 3) echo "$module.mk unchanged"
      rm $module.mk
    else
      mv $module.mk $chmbuild/$module.mk && echo "$module.mk replaced"
    endif
  else
    echo "$module.mk not written"
  endif
  if ( ( $module == "gamint" ) && ( -e gamess ) ) then
    pushd gamess >>! /dev/null
    $makemod -n gamess `pwd` $chmsrc gamess.mk || set rc = 1
    mv gamess.mk $chmbuild/gamess.mk
    popd >>! /dev/null
  endif
  popd >>! /dev/null
end
popd >>! /dev/null
#
if ($rc != 0) then
  echo "WARNING: makefiles may be incomplete: $problems"
endif
tsort < $chmtool/tsort.inp > /dev/null
if ($status != 0) then
  echo "WARNING: tsort found a dependency cycle"
  set rc = 1
endif
exit($rc)
