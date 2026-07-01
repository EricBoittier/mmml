#! /bin/csh -f
# grftest.com : stand-alone CHARMM GRAPHX Test Script
#-----------------------------------------------------------------------
# Change Log:
#
# 04-Aug-95  Introduced with c24a3 --> c24b1
#
# Inquiries to chmgr@tammy.harvard.edu
# Inquiries to rvenable@deimos.cber.nih.gov
# 
#-----------------------------------------------------------------------
#           
# Arguments
# chm_host = $argv[1], the host machine type
# chm_exec = $argv[2], the executable name, if not 'charmm'
#
#-----------------------------------------------------------------------

#
date
echo " "
echo " Checking args, pref.dat, and env var DISPLAY for graphics tests"
#
phase1:
# command line argument parsing
# set up testing environments
#
if ( $#argv == 0 ) then
  echo " "
  echo " Usage: grftest.com  host_machine_type  [ executable_name ]"
  echo " "
  echo "        host_machine_type is one of the following."
  echo "            convex   for Convex computers"
  echo "            hpux     for HP 9000/700 or 9000/800"
  echo "            sgi      for SGI IRIS series"
  echo "            stardent for Stardent computers"
  echo "            ibmrs    for IBM RS/6000 series"
  echo "            sun      for SUN work stations"
  echo "            cray     for Cray X-MP and Y-MP"
  echo " "
  echo "        The default executable_name is 'charmm'"
  exit
#     
else if ( $#argv == 1 ) then
  set chm_host = $argv[1]
#
else if ( $#argv == 2 ) then
  set chm_host = $argv[1]
  set chm_exec = $argv[2]
#
endif
#
setenv chmhost $chm_host
if ( $?chmroot == 0 ) setenv chmroot `cd ..;pwd`
if ( $?chmexec == 0 ) setenv chmexec $chmroot/exec/$chmhost
if ( $?chmtool == 0 ) setenv chmtool $chmroot/tool
if ( $?chmtest == 0 ) setenv chmtest `pwd`
#
if ( $?chm_exec == 0 ) set chm_exec = $chmexec/charmm
if (! -e $chm_exec   ) then
  echo " "
  echo " grftest.com> the testing target $chm_exec does not exist."
  exit 1
endif
#
if (! -e $chmtest/data ) then
  echo " "
  echo " grftest.com> Directory $chmtest/data does not exist."
  exit 1
endif
if (! -e $chmtest/datadir.def ) then
  echo " "
  echo " grftest.com> $chmtest/datadir.def does not exist."
  exit 1
endif
#
# check pref.dat; assumes oneof[ NOGRAPHICS NODISPLAY XDISPLAY GLDISPLAY ]
# also check pref.dat for the APOLLO keyword
# check the env var DISPLAY (must be set for grfxwin.inp)
#
set grftype = `fgrep -e DISPLAY -e GRAPHIC $chmroot/build/$chmhost/pref.dat`
set grfapo  = `fgrep APOLLO $chmroot/build/$chmhost/pref.dat`
switch ( $grftype )
 case NOGRAPHICS:
  echo " "
  echo " grftest.com> Graphics code not compiled"
  echo " grftest.com> NOGRAPHICS keyword found in pref.dat"
  echo " grftest.com> No testcases run, exiting grftest.com"
  exit 1
  breaksw
 case NODISPLAY:
  set testlist = cgrftest/grfnodsp.inp
  breaksw
 case XDISPLAY:
  if ( $?DISPLAY == 0 ) then
   echo " "
   echo " grftest.com> env var DISPLAY not set; cannot open X window"
   echo " grftest.com> only testcase for GRAPHX NOWIN will be run"
   set testlist = cgrftest/grfnowin.inp
  else
   set testlist = ( cgrftest/grfxwin.inp cgrftest/grfnowin.inp )
  endif
  breaksw
 case GLDISPLAY:
  set testlist = ( cgrftest/grfgldsp.inp cgrftest/grfnodsp.inp )
  breaksw
 default:
  if ( $grfapo == APOLLO ) then
   set testlist = ( cgrftest/grfapo.inp cgrftest/grfnodsp.inp )
  else
   echo " "
   echo " grftest.com> No valid graphics keyword in pref.dat, exiting"
   exit 1
  endif
  breaksw
endsw
#
#-----------------------------------------------------------------------
#
echo " "
echo " Checks complete; running $testlist"
phase2:
#
# run testcases
#
foreach testcase ( $testlist )
 set name = $testcase:r
 $chm_exec < $testcase >& $name.out
end
#
echo " "
echo " Testcases complete; be sure to print any .ps files produced, or at"
echo " least preview with ghostscript; files bpti.atm bpti.fdat also created."
echo " "
echo " grftest.com>     N.B.  All output routed to subdir  cgrftest"
echo " "
#
