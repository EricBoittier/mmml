#!/bin/csh -f
# install.com
#-----------------------------------------------------------------------
#
# Change Log:
#    Sep-14 Major string method updates
#  6-Mar-13 DOMDEC added as a switch - APH
# 30-Sep-12 Added GAMUS option to modify makefile and pref.dat for GAMUS - JMS
# 15-Jun-12 Added keyword for G09
# 18-Jan-11 Removed obsolete machine type and compilers from help message - RMV
#           Disabled setting related variables to 1 (true)
#           Likewise for obsolete keywords
#           Fixed -KEYWORD parsing via $opt --> "$opt"
# 01-Jan-09 Added support for the NAG compiler with Linux
# 13-Nov-08 RCW (SDSC) Added support for xt4 machine target (Cray XT4/5)
# 08-Aug-08 NEWRNG key removed
# 21-Jan-08 CGENFF added to FULL - KEVO
# 16-Jun-08 ENSEMBLE installation modified: sero-temperature string method
#           added
# 15-Aug-07 fix allows both 32-bit and 64-bit (x86_64) PGI, PathScale
#           PGI now pgf95 instead of pgf77
#           Unused (in any C code) -Dnographics flag removed
# 15-Aug-06 FEWMFC, PBCUBES, PROTO, SMD, TAMD keys added
#           FORTRAN90 compilation updated
#           Dynamic link and MODPREF added
#           gfortran is the default gnu compiler.
# 15-Feb-06 GFORTRAN, X86_64 switch added
#           em64t, xlf95, mpif90 sections added
#           SQUANTUM installation added
#           ENSEMBLE installation modified
# 15-Aug-05 MNDO97 interfaced. Use W to replace QUANTUM.
# 15-Feb-05 HUGE, XLARGE implemented
#           New feature keys, FIRCHG, SLVSLV, LNTAB1, WCA added to pref.dat
#           SOCKET compilation fixed
# 15-Aug-04 Obsolete platforms cm5, convex, cray, cray_f90, cspp, dec, gws
#           ibmrs, ibmsp, ibmsp3, intel, stardent, t3d and terra removed.
#           EXPAND is default
#           AR_COMMAND macro added
#           64-bit IBM/AIX support
# 15-FEB-04 Added ASPMEMB, CHEQ, DYNVV2, EMAP, GBIM, LRVDW, OVERLAP,
#           SCPISM and TESTENDIAN to the FULL version
#           Added OSX, EFC(GNU) support - RG, SRB, MFC, CLB
#           QQCHEM switch added
#           VECTOR, CRAYVEC, PARVECT supports removed
# 15-Aug-03 GAMESS-UK interface updated, i8 support - PS
#           SUN64 port - BC
# 15-FEB-03 IFC switch added - SRB
#           LAMMPI libraries updated to circa 6.5.6 - SRB
#           Fix to support multiple MPI libraries - SRB
#           CHEQ processing added
# 15-AUG-02 SUN MPI port - RS
#           SCCDFTB switch added - QC
#           EMAP code processing added
# 15-FEB-01 IBMSP3 port and NOLOG switch added - CLB
#           g77 and pgf77 MPI modified - BW
#           Size XXLARGE added - MH
#           PSSP added to FULL version - SB
# 19-Jul-00 Added GAMESS-UK interfacing - PS
#           FLUCQ key added - BW
# 15-Feb-99 c27a2 installation
#           MBO(N)D GNU port added
#           g77/f77/fort77 compilation switch modified
# 15-Sep-98 c27a0 installation
#           setenv MAKE_COMMAND for Cray T3E
#           adumb.mk, cff.mk, mc.mk and shapes.mk added
# 15-Aug-97 c25b1/c26a1 installation
#           Default feature keys are grouped into FULL and LITE for release
#           MBO(N)D key is implemented - RN
#           GAMESS modifications + GNU support for 3 compilers - MH
# 15-FEB-97 c25a3 installation
#           CADPAC jey is implemented - PL
# 15-Aug-96 c25a2 installation
#           GAMESS and PARALLEL installation added
#           POLAR key added
# 15-Feb-96 c24b2 installation
#           MMFF and PBEQ added
# 15-Aug-95 c24b1 installation
#           install switches are upgraded
#           ibmsp, intel, mpi, terra and GNU ports added
# 15-Feb-95 t3d, cspp, cm5 ports added
# 10-Feb-94 executable name charmm replaces the previous version
#             specific charmm{nn}
# 05-Feb-94 CHARMM 24a1 installation
#           main changed to charmm
# 15-Jan-94 CHARMM 23f3 installation
#           flexfort removed and new prefx.f from NIH used
#           add X and stopmark switches
# 21-Feb-93 CHARMM 23 (c23f) installation
# 08-Oct-92 HP-UX support added.
#
# 01-Jan-92 CHARMM 22 Installation Script
#           File lists are constructed from the source directory
#           listing.  pref.dat file can be constructed interactively.
#           Some features are from Stephen H. Fleischman's build scripts.
# 15-Sep-91 CHARMM 22.0.b1 Installation Script
#           add IBM RS/6000, Stardent and Cray Y-MP support
#           testing script is separated out.
# 05-May-91 CHARMM 22.0.b Installation Script
#
# Inquiries to chmgr@tammy.harvard.edu (617)495-4102
#-----------------------------------------------------------------------
#
# Shell variables
#RHS sunmpi added
# chm_host  = ( gnu, osx, em64t, xt4, nag, gpu ) = $argv[1]
#
#-----------------------------------------------------------------------
#
date
echo "Compilation Commands: $0 $*"
if ( ! $?nolog ) then
   echo "$0 $*" > install.`hostname -s`
endif

echo " install.com> WARNING: install.com is deprecated"
echo " install.com> and will be removed from future releases."
echo " install.com> Please use ./configure and CMake."

#
phase1:
# install.com command line argument processing
# get host-mahchine-type and CHARMM size to be installed.
#
set clean_flag = 0
set distclean_flag = 0

set xreq = 0
set nodsp = 0
set pvmset = 0
set mpiset = 0
set ensemble = 0
set qabpo = 0
# VO : string method v
set qstringm = 0 ;
# VI : string method ^
set mpich = 0
set lammpi = 0
set mpispecial = 0
set mpif90 = 0
set nersc = 0
set stopmark = 0
set ibm64 = 0
set qgamess = 0
set qgamessuk = 0
set qcadpac = 0
set qmndo97 = 0
set qmmmsemi = 0
set qqchem = 0
set qg09 = 0
set qqturbo = 0 
set qsccdftb = 0
set qsquantm = 0
set apbs = 0
set fftw = 0
set mkl = 0
set scali = 0
set socket = 0
set f77 = 0
set f2c = 0
set g77 = 0
set xlf95 = 0
set osx = 0
set ifc = 0
set efc = 0
set amd64 = 0
set x86_64 = 0
set gfortran = 0
set pathscale = 0
set ifort = 0
set g95 = 0
set pgf95 = 0
set cfort = 0
set f90 = 0
set nih = 0
set tsri = 0
set polyrate = 0
set full = 1
set longint = 0
set DEBUG = 0
set qpipf = 0
set dynamic = 0
set cmpi_ok = 0
set openmm = 0
set ommplugin = 0
set openmp = 0
set pthread = 0
set domdec = 0
set domdec_gpu = 0
set colfft_sp = 1
# JMS 10/2011
set gamus = 0


#DCM 01/2019
set qdcm=0
# automatic modification of pref.dat
set addprefdat = ()
set delprefdat = ()

if ( $#argv == 0 ) then
  echo " "
  echo "N.B.: This is the new Fortran95 revision of CHARMM; read install.doc"
  echo " "
  echo " Usage: install.com host-machine-type [charmm-size] [Sw] "
  echo " "
  echo "       [1] host-machine-type is one of the following."
  echo "               em64t     ifort compilers on x86_64 Linux"
  echo "               gnu       Linux; GNU compiler by default"
  echo "               gpu       GPU using CUDA library"
  echo "               g95       G95 compiler"
  echo "               nag       Numerical Algorithms Group compiler"
  echo "               osx       Mac OSX machines"
  echo "               xt4       Cray XT4/XT5 using compute node Linux"
  echo " "
  echo "       [3] [Sw] are install switches, which must be specified after"
  echo "           the host-machine-type argument.  You may specify any of the following."
#  echo "           X and NODISP, M and E are mutually exculsive, respectively."
  echo "           Optional args clean or distclean can be used here; "
  echo "           recommended prior to re-installation with changed options. "

  echo "           MPIF90, MPICH, SCALI, and MPISPECIAL are mutually exclusive"
  echo "           additional options to be used with M; MPIF90 is the default."
  echo " "
  echo "           X  include Xlib graphics, along with .ps .pov .fdat files"
#  echo "      NODISP  graphics, no display; make .ps .pov .fdat files only"
#  echo "              The default is neither, i.e. no graphics"
#  echo "           P  links to PVM"
  echo "           M  links to MPI"
  echo "           1  to halt after setting up installation directory."
  echo "           2  to halt after making installation utilities."
#  echo "          i8  to request 64 bit integers"
  echo "           Q  replace QUANTUM with GAMESS."
  echo "           U  replace QUANTUM with GAMESS-UK."
  echo "           C  replace QUANTUM with CADPAC."
  echo "           T  replace QUANTUM with SCCDFTB."
  echo "          QC  replace QUANTUM with Q-CHEM."
  echo "          QT  replace QUANTUM with Turbomole."
  echo "          SQ  replace QUANTUM with SQUANTUM, only with altix/gnu."
  echo "           W  replace QUANTUM with MNDO97, only with altix/gnu."
  echo "          QS  replace QUANTUM with QMMMSEMI (AMBER Semi-empirical QMMM)."
  echo "         G09  replace QUANTUM with Gaussian09."
  echo "        APBS  compile with APBS support."
  echo "        FFTW  compile with FFTW support (adds COLFFT keyword to pref.dat)."
  echo "         MKL  compile with MKL support (adds COLFFT keyword to pref.dat)."
  echo "      OPENMM  add support for OpenMM (see openmm.doc)"
  echo "      DOMDEC  compile with DOMDEC support"
  echo "  DOMDEC_GPU  compile with DOMDEC GPU support (switches on DOMDEC)"
  echo "        PIPF  add support for Polarizable Intermolecular Potential Function"
  echo "       POLYR  add support for POLYRATE interface"
  echo "       GAMUS  add support for GAMUS (requires LAPACK installation)"
  echo "           S  Uses TCP/IP SOCKET library for parallel."
  echo "           E  Builds a version with ENSEMBLE replicas; requires M"
  echo "     STRINGM  String method (requires M)."
  echo "        ABPO  compile with ABPO support (requires M and activates E)."
  echo "        FULL  For FULL featured version (default)."
  echo "        LITE  For a version with reduced functional features."
#  echo "       XLF95  Uses xlf95/MacOSX driven by xlf95 (default is gfortran)."
#  echo "         F77  Uses Absoft/Linux (default is gfortran)."
#  echo "         G77  Uses obsolete GNU g77 (default is gfortran)."
#  echo "         F2C  Uses f2c/Linux driven by fort77(default is gfortran)."
#  echo "         IFC  Uses IA-32 Intel Fortran ifc/Linux (default is gfortran)."
#  echo "         EFC  Uses IA-64 Intel Fortran efc/Linux and forces I8 (default is gfortran)."
  echo "       IFORT  Uses Intel Fortran ifort/Linux for gnu (default is gfortran)."
  echo "         G95  Uses  g95/Linux for gnu (default is gfortran)."
  echo "       PGF95  Uses PGI pgf95/Linux for gnu (default is gfortran)."
  echo "          PS  Uses PathScale Linux compiler for gnu (default is gfortran)."
#  echo "    GFORTRAN  Uses extra keywords for gfortran."
#  echo "      X86_64  Uses extra keywords for X86_64, both AMD64 & EM64T."
#  echo "       AMD64  Uses extra keywords for g77 on AMD64."
  echo "         NIH  Uses extra keywords for NIH."
  echo "        TSRI  Uses extra keywords for TSRI."
  echo "      MPIF90  Relies entirely on mpif90 wrapper for MPI compiling/linking."
  echo "       NERSC  Relies entirely on ftn wrapper for NERSC compiling/linking."
  echo "       MPICH  adds special library options for standard MPICH."
#  echo "      LAMMPI  adds special library options for standard LAM/MPI."
  echo "       SCALI  adds special library options for standard SCALI MPI Connect."
  echo "  MPISPECIAL  prompts for special MPI library options for load."
  echo "          GA  Use GA tools version of GAMESS-UK"
  echo "           D  link dynamically (ifc/ifort)"
  echo "     MODPREF  add/remove keywords from pref.dat (w/ addtl. parameter)"
  echo "                e.g.  +CGENFF to add, -MMFF to remove"
  echo "       keepf  Will keep the preprocessed .f90 files in build/mach."
  echo " DEBUG/debug  Compile with debugging options to compiler (FCD)"
  echo " big_e/lit_e  Use big/little_endian binary I/O if supported by compiler"
  echo " "
  exit
#
else if ( $#argv == 1 ) then
  set chm_host = $argv[1]
else if ( $#argv >= 2 ) then
  set chm_host = $argv[1]
  shift
  foreach opt ( $argv )
    if ( "$opt" == "clean" ) set clean_flag = 1
    if ( "$opt" == "distclean" ) set distclean_flag = 1
    if ( "$opt" == 'X'  ) set xreq = 1
#    if ( "$opt" == 'P'  ) set pvmset = 1
    if ( "$opt" == 'M'  ) then
       set mpiset = 1
       set mpif90 = 1
    endif
    if ( "$opt" == 'E'  ) then
       set ensemble = 1
       set cmpi_ok = 0
    endif
    if ( "$opt" == 'ABPO' ) then
       set qabpo = 1    
       set ensemble = 1
       set cmpi_ok = 0
    endif
# VO : string method v
    if ( "$opt" =~ [Ss][Tt][Rr][Ii][Nn][Gg][Mm] ) then
     set qstringm = 1
    endif
# VO : string method ^
    if ( "$opt" == 'D'  ) set dynamic = 1
    if ( "$opt" == '1'  ) set stopmark = 1
    if ( "$opt" == '2'  ) set stopmark = 2
    if ( "$opt" == 'I8' || "$opt" == 'i8' ) set longint = 1
    if ( "$opt" == 'Q'  ) set qgamess = 1
    if ( "$opt" == 'U'  ) set qgamessuk = 1
    if ( "$opt" == 'C'  ) set qcadpac = 1
    if ( "$opt" == 'APBS' || "$opt" == 'apbs'  ) then
      set apbs = 1
      setenv APBS YES      
    endif     
    if ( "$opt" == 'FFTW' || "$opt" == 'fftw'  ) then
      set fftw = 1
    endif     
    if ( "$opt" == 'MKL' || "$opt" == 'mkl'  ) then
      set mkl = 1
    endif     
    if ( "$opt" == 'S'  ) set socket = 1
    if ( "$opt" == 'T'  ) set qsccdftb = 1
    if ( "$opt" == 'QC' ) set qqchem = 1
    if ( "$opt" == 'G09' ) set qg09 = 1
    if ( "$opt" == 'QT' ) set qqturbo = 1
    if ( "$opt" == 'SQ' ) set qsquantm = 1
    if ( "$opt" == 'W'  ) set qmndo97 = 1
    if ( "$opt" == 'QS'  ) set qmmmsemi = 1
    if ( "$opt" == 'FULL' || "$opt" == 'full' ) set full = 1
    if ( "$opt" == 'LITE' || "$opt" == 'lite' ) set full = 0
#    if ( "$opt" == 'XLF95' || "$opt" == 'xlf95' ) set xlf95 = 1
#    if ( "$opt" == 'F2C') set f2c = 1
#    if ( "$opt" == 'F77') set f77 = 1
#    if ( "$opt" == 'IFC' || "$opt" == 'ifc') set ifc = 1
    if ( "$opt" == 'IFORT' || "$opt" == 'ifort') set ifort = 1
    if ( "$opt" == 'G95' || "$opt" == 'g95') set g95 = 1
#    if ( "$opt" == 'G77' || "$opt" == 'g77') set g77 = 1
#    if ( "$opt" == 'EFC'|| "$opt" == 'efc' ) then
#      set efc = 1
#      set longint = 1
#    endif
    if ( "$opt" == 'PGF95' || "$opt" == 'pgf95' ) set pgf95 = 1
#    if ( "$opt" == 'FORT') set cfort = 1
#    if ( "$opt" == 'F90'|| "$opt" == 'f90' ) set f90 = 1
    if ( "$opt" == 'GFORTRAN'|| "$opt" == 'gfortran' ) set gfortran = 1
#    if ( "$opt" == 'X86_64'|| "$opt" == 'x86_64' ) set x86_64 = 1
#    if ( "$opt" == 'AMD64'|| "$opt" == 'amd64' ) set amd64 = 1
    if ( "$opt" == 'PS'|| "$opt" == 'ps' ) set pathscale = 1
    if ( "$opt" == 'NIH'|| "$opt" == 'nih' ) set nih = 1
    if ( "$opt" == 'NODISP'  ) set nodsp = 1
    if ( ( "$opt" == 'TSRI'|| "$opt" == 'tsri' ) && $full == 0 ) set tsri = 1
    if ( "$opt" == 'big_e'|| "$opt" == 'BIG_E' ) set bigendian = 1
    if ( "$opt" == 'lit_e'|| "$opt" == 'LIT_E' ) set littleendian = 1
    if ( "$opt" == 'keepf'|| "$opt" == 'KEEPF' ) set keep_src
    if ( "$opt" == 'debug' || "$opt" == 'DEBUG'  ) set DEBUG = 1
    if ( "$opt" == 'nolog'|| "$opt" == 'NOLOG' || "$opt" == 'Nolog') set nolog
    if ( "$opt" == 'mpich'|| "$opt" == 'MPICH' ) then
       set mpich = 1
       set mpif90 = 0
    endif
    if ( "$opt" == 'mpif90'|| "$opt" == 'MPIF90' ) set mpif90 = 1
    if ( "$opt" == 'nersc'|| "$opt" == 'NERSC' ) set nersc = 1
#    if ( "$opt" == 'lammpi'|| "$opt" == 'LAMMPI' ) set lammpi = 1
    if ( "$opt" == 'scali'|| "$opt" == 'SCALI' ) then
       set scali = 1
       set mpif90 = 0
    endif
    if ( "$opt" == 'mpispecial'|| "$opt" == 'MPISPECIAL' ) then
       set mpispecial = 1
       set mpif90 = 0
    endif
    if ( "$opt" == 'polyr' || "$opt" == 'POLYR' ) set polyrate = 1
# PJ 06/2005
    if ( "$opt" == 'pipf' || "$opt" == 'PIPF' ) set qpipf = 1
# JMS 06/2011
    if ( "$opt" == 'gamus' || "$opt" == 'GAMUS' ) set gamus = 1
    if ( "$opt" == 'openmm' || "$opt" == 'OPENMM' ) set openmm = 1
    if ( "$opt" == 'domdec' || "$opt" == 'DOMDEC'  ) then
      set domdec = 1
      set pthread = 1
      set cmpi_ok = 0
    endif
    if ( "$opt" == 'domdec_gpu' || "$opt" == 'DOMDEC_GPU'  ) then
      set domdec = 1
      set domdec_gpu = 1
      set openmp = 1
      set pthread = 1
      set cmpi_ok = 0
    endif
# MH 05/2011
# extra stuff for NERSC:
    if ($nersc == 1 ) then
	set mpiset = 1
	set mpif90 = 1
#	set pgf95 = 1 # VO can use NERSC compilers other than pgi & gfortran
#	if ( $gfortran == 1 ) set pgf95 = 0
     endif
# MF 03/2006
# read extra flags in the form of +TAG or -TAG to modify pref.dat
    if ( { ( echo "$opt" | grep "^+" >& /dev/null ) } ) then
      set tag = `echo "$opt" | cut -c2- | sed 'y/abcdefghijklmnopqrstuvwxyz/ABCDEFGHIJKLMNOPQRSTUVWXYZ/'`
      set addprefdat = ( $addprefdat $tag )
    else if ( { ( echo "$opt" | grep "^-" >& /dev/null ) } ) then
      set tag = `echo "$opt" | cut -c2- | sed 'y/abcdefghijklmnopqrstuvwxyz/ABCDEFGHIJKLMNOPQRSTUVWXYZ/'`
      set delprefdat = ( $delprefdat $tag )
    endif
    #EM 01/2019
    if ( "$opt" == 'dcm' || "$opt" == 'DCM' ) set qdcm = 1
  end
endif
#
if ($?bigendian && $?littleendian ) then
    echo " install.com> Only big_e or lit_e can be specified. Not both."
    exit
endif
if ( $?bigendian ) setenv BIG_ENDIAN yes
if ( $?littleendian ) setenv LITTLE_ENDIAN yes

setenv REMOVE_F "/bin/rm -f"
if ( $?keep_src ) setenv REMOVE_F "echo Keeping "
setenv REMOVE_O "echo Keeping "

setenv AR_COMMAND "ar rucv"
#------------------------------------------------------------------------------
#=============================================================================
#                        SANITY CHECKS
#
if ($xreq == 1 && $nodsp == 1 ) then
    echo " install.com> Either X display or NODISP can be specified. Not both."
    exit
endif
#
if ( $scali ==  1 ) set mpiset=1
if ($mpiset == 1 && $pvmset == 1 ) then
    echo " install.com> Only PVM or MPI can be specified. Not both."
    exit
endif
# VO : string method v
if ($qstringm == 1 && ! ( $mpiset == 1)) then
    echo " install.com> STRINGM keyword requires key 'M' (parallel)"
    exit
endif
#
if ($qstringm == 1 && domdec == 1) then
    echo " install.com> WARNING : STRINGM and DOMDEC can be compiled together but cannot be used simultaneously"
endif
# VO : string method ^
#
#                     END of SANITY CHECKS
#---------------------------------------------------------------------------
#
#=============================================================================
#                        ENVIRONMENT SETUP
# set the host machine type environment variable chmhost
#
setenv chmhost $chm_host
#
# set chmroot to point the current directory (~/charmm) and
# set up the CHARMM installation environments.
#
setenv chmroot  `pwd`
echo "chmroot= $chmroot"
setenv chmbuild $chmroot/build/$chm_host
setenv chmexec  $chmroot/exec/$chm_host
setenv chmlib   $chmroot/lib/$chm_host
setenv chmsrc   $chmroot/source
setenv chmtool  $chmroot/tool
setenv chmpar   serial
set path = ( $path $chmtool )

# Separate parallel builds from serial builds
if ( $mpiset == 1 ) then
  setenv chmbuild ${chmbuild}_M
  setenv chmexec ${chmexec}_M
  setenv chmlib ${chmlib}_M
endif
# for Makefiles
setenv LIB $chmlib
setenv EXEC $chmexec

# Pseudosizes clean and distclean remove binaries
if ( $clean_flag == 1 ) then
  if ( -f $chmexec/charmm ) then
    echo "Removing $chmexec/charmm"
    rm $chmexec/charmm
  endif
  if ( -d $chmlib ) then
    echo "Removing $chmlib/*.(a|o)"
    rm $chmlib/*.a $chmlib/*.o
    if ( -d $chmlib/openmm_plugins ) then
      echo "Removing $chmlib/openmm_plugins/*.so"
      rm $chmlib/openmm_plugins/*.so
    endif
  endif
  if ( -d $chmbuild ) then
    echo "Removing $chmbuild/*.(o|mod|f90)"
    rm $chmbuild/*.o $chmbuild/*.mod $chmbuild/*.f90
  endif
  exit
else if ( $distclean_flag == 1 ) then
  if ( -d $chmbuild || -d $chmlib || -d $chmexec ) then
    echo "Removing directories $chmbuild $chmlib $chmexec"
    rm -r $chmbuild $chmlib $chmexec 
  endif
  exit
endif

#
if (! -e $chmroot/exec ) mkdir $chmroot/exec
if (! -e $chmexec      ) mkdir $chmexec
if (! -e $chmroot/lib  ) mkdir $chmroot/lib
if (! -e $chmlib       ) mkdir $chmlib
#
# VO stringm v =================================
if ($qstringm == 1 ) then
# if a Makefile is provided, make CHARMM-compatible source code in the stringm directory
 pushd $chmsrc/stringm
 if ( -f "Makefile" ) then
  echo -n "Making string method source code in $chmsrc/stringm ..."
  if ( ! $?MAKE_COMMAND ) then
   make >& /dev/null
  else
   $MAKE_COMMAND >& /dev/null
  endif
  echo "done"
  popd
 endif
endif
# VO stringm ^ =================================
#
if ( $stopmark == 1 && -e $chmbuild ) then
  echo " "
  echo " install.com> Recreating directory $chmbuild."
  echo " "
  /bin/rm -r $chmbuild
endif
#    -----------OSX ENV VARS ----------------
# Makefile_osx to support gnu gfortran and g95 and xlf95 and ifort (Intel Macs) on Mac OSX
if ( $chm_host == osx ) then
    if ( $xlf95 == 1 ) then
      setenv XLF95_OSX YES
    else if ($ifort == 1 ) then
      setenv INTEL_IFORT YES
      if ( $x86_64 == 1 ) then
	set longint = 1
	setenv X86_64 YES
      endif
 #   else
 #     setenv GNU_G77 YES
    else if ($g95 == 1 ) then
      setenv GNU_G95 YES
    else #($gfortran == 1 ) then
      if ( $MACHTYPE == powerpc ) setenv POWERPC yes
      setenv GNU_GFORTRAN YES
      if ( $x86_64 == 1 ) then
	set longint = 1
	setenv X86_64 YES
      endif
    endif
    if( $mpiset == 1 ) setenv MPI_OSX YES
endif
#
# Makefile_gpu to support CHARMM using GPU (currently NVIDIA/CUDA)
#
if ( $chm_host == gpu ) then
    # make defaults here so they can be used both in md3compile and Makefile_gpu
    if ( $?CUDATK == 0 ) then
        set nvcc = `which nvcc`
        if ( $status == 0 ) then
            set cudabin = $nvcc:h
            setenv CUDATK $cudabin:h
        else
            setenv CUDATK /opt/cuda
            if (! -x $CUDATK/bin/nvcc) then
                echo 'Cannot find CUDA Toolkit'
                echo 'Please set CUDATK to its directory path'
                exit 1
            endif
        endif
        echo "CUDATK = $CUDATK"
    endif
# this is not needed anymore
#    if ( $?CUDASDK == 0 ) then
#        setenv CUDASDK $CUDATK/sdk/C
#        if (! -d $CUDASDK) setenv CUDASDK $HOME/NVIDIA_GPU_Computing_SDK/C
#        if (! -d $CUDASDK) then
#            echo 'Cannot find GPU Computing SDK'
#            echo 'Please set CUDASDK to the "C" directory path'
#            exit 1
#        endif
#        echo "CUDASDK = $CUDASDK"
#    endif
    if ($ifort == 1 ) then
        setenv INTEL_IFORT YES
        setenv CC gcc
        setenv ifort_gpu 1
    else
        setenv GFORTRAN YES
        setenv ifort_gpu 0
    endif
    if ($mpif90 == 1) then
        setenv MPIF90 YES
    endif
# support only 64 bit!
#    set longint = 1
#    setenv X86_64 YES
endif
#
#
#    -------------COMPILER SPECIFIC ENV VARS -----------------
# Makefile_GNU to support Absoft f77 / GNU g77 / f2c fort77 under Linux
if ( $chm_host == gnu ) then
  if ( $f77 == 1 ) then
    setenv ABSOFT_F77_V44 YES
  else if ( $f2c == 1 ) then
    setenv GNU_F2C YES
  else if ( $ifort == 1 ) then
    setenv INTEL_IFORT 1
    setenv INTEL32_IFC_IFORT yes
    if ( $longint == 1 ) then 
	setenv LONGINT 1
	setenv INTEL32_IFC_IFORT_I8 yes
	setenv INTEL32_IFC_IFORT
    endif
  else if ( $g95 == 1 ) then
    setenv GNU_G95 1
  else if ( $ifc == 1 ) then
    setenv INTEL32_IFC YES
    setenv INTEL32_IFC_IFORT yes
  else if ( $efc == 1 ) then
    set longint=1
    setenv INTEL64_EFC YES
  else if ( $g77 == 1 ) then
    setenv GNU_G77 YES
  else if ( $amd64 == 1 ) then
    setenv AMD64 YES
  else if ( $pathscale == 1 ) then
    setenv PATHSCALE YES
    if ( $nersc == 1 ) then
      set x86_64 = 1
    endif
    if ( $x86_64 == 1 ) then
      set longint = 1
      setenv X86_64 YES
    endif
  else if ( $pgf95 == 1 ) then
    setenv PGI_F95 YES
    setenv NO_STOP_MESSAGE 1
    if ( $x86_64 == 1 ) then
      set longint = 1
      setenv X86_64 YES
    endif
  else if ( $cfort == 1 ) then
    set longint=1
    setenv COMPAQ_FORT YES
  else
    setenv GFORTRAN YES
    if ( $x86_64 == 1 ) then
      set longint = 1
      setenv X86_64 YES
    endif
  endif
else if ( $chm_host == g95 ) then
    setenv GNU_G95 1
endif

if ( $DEBUG == 1 ) then
  setenv DEBUG YES
endif  

#
#--------------------------------------------------------------#
#                                                              #   
#    Set up the build directory with makefiles and pref.dat    #
#                                                              #
#--------------------------------------------------------------#
pushd $chmtool >> /dev/null
./setmk.com UNX
popd >> /dev/null
set new_makefile = 0
if ( -e $chmbuild ) then
  foreach srcmk ( $chmroot/build/UNX/*.mk $chmroot/build/UNX/Makefile_$chm_host )
    set destmk = $chmbuild/$srcmk:t
    if ({ test $srcmk -nt $destmk }) then
      if ( $srcmk:t == 'objlibs.mk' ) then
        echo "Warning: $destmk is out of date"
      else
        echo "Updating $destmk"
        cp $srcmk $chmbuild/
        set new_makefile = 1
      endif
    endif
  end
else
  echo " "
  echo " install.com> Directory $chmbuild does not exist."
  echo "              Creating $chmbuild ..."
  mkdir $chmbuild
  if (! -e $chmroot/build/UNX ) then
    echo " install.com> Directory $chmroot/build/UNX does not exist."
    echo "              CHARMM installation can not proceed."
    exit 1
  endif
  cp $chmroot/build/UNX/*.mk $chmbuild
  if ( $qcadpac == 1 ) then
    if ( $mpiset == 1 ) then
     /bin/mv $chmbuild/cadpac.mk $chmbuild/cadpac.mk.$$
     sed -e 's/DEFS =/DEFS = -DMPI -DPARALLEL/' \
      $chmbuild/cadpac.mk.$$ > $chmbuild/cadpac.mk
     /bin/rm -f $chmbuild/cadpac.mk.$$
    endif
  endif

#mfc  if ( $qsquantm == 1 ) then
      cp $chmroot/build/UNX/squantm.mk  $chmbuild/squantm.mk
#mfc  endif

  if ( $qmmmsemi == 1 ) then
      cp $chmroot/build/UNX/qmmmsemi.mk  $chmbuild/qmmmsemi.mk
  endif

  if ( $ibm64 == 1 ) then
    cp $chmroot/build/UNX/Makefile_${chm_host}64 $chmbuild/Makefile_$chm_host
  else
    cp $chmroot/build/UNX/Makefile_$chm_host $chmbuild
  endif
  set new_makefile = 1

endif

if ( $new_makefile == 1 ) then
#
# Modify Makefile_template for graphics
flag:
  if( $xreq == 1 && ( $chm_host == gnu || $chm_host == em64t ) ) then
    if ( $x86_64 == 1 ) then
      sed -e 's@grflib@-L/usr/X11R6/lib64 -lX11@' \
        $chmbuild/Makefile_$chmhost > $chmbuild/Makefile_$$
    else
      sed -e 's@grflib@-L/usr/X11R6/lib -lX11@' \
        $chmbuild/Makefile_$chmhost > $chmbuild/Makefile_$$
    endif
    sed -e 's@gcc @gcc -Dxdisplay @' -e 's@icc @icc -Dxdisplay @' \
      $chmbuild/Makefile_$$ > $chmbuild/Makefile_$chmhost
    /bin/rm $chmbuild/Makefile_$$
  else if( $xreq == 1 && $chm_host == osx ) then
# newer versions of OS X place X11 in /opt instead of /usr
    set xtmp = `find /usr -name libX11.dylib`
    if ( $#xtmp == 0 ) then
      set xtmp = `find /opt -name libX11.dylib`
      if ( $#xtmp == 0 ) then
        echo "X11 requested, and library cannot be found"
        exit
      endif
    endif
    set xlib = $xtmp:h:h
    sed -e "s;grflib;-L$xlib/lib -lX11;" \
        $chmbuild/Makefile_$chmhost > $chmbuild/Makefile_$$
    sed -e "s@gcc @gcc -Dxdisplay -I$xlib/include @" \
        -e "s@icc @icc -Dxdisplay -I$xlib/include @" \
        $chmbuild/Makefile_$$ > $chmbuild/Makefile_$chmhost
    /bin/rm $chmbuild/Makefile_$$
  else if( $xreq == 1 && $chm_host == g95 ) then
    sed -e 's@grflib@-L/usr/X11R6/lib -lX11@' \
      $chmbuild/Makefile_$chmhost > $chmbuild/Makefile_$$
    sed -e 's@gcc @gcc -Dxdisplay @' \
      $chmbuild/Makefile_$$ > $chmbuild/Makefile_$chmhost
    /bin/rm $chmbuild/Makefile_$$
  else if ( $xreq == 1 ) then
    sed -e '/^GLIB =/s@grflib@-lX11@' \
      $chmbuild/Makefile_$chmhost > $chmbuild/Makefile_$$
    sed -e '/^CC =/s@cc @cc -Dxdisplay @' \
        $chmbuild/Makefile_$$ > $chmbuild/Makefile_$chmhost
    /bin/rm $chmbuild/Makefile_$$
  else
    sed -e '/^GLIB =/s@grflib@ @' \
      $chmbuild/Makefile_$chmhost > $chmbuild/Makefile_$$
    sed -e '/^CC =/s@cc @cc @' \
        $chmbuild/Makefile_$$ > $chmbuild/Makefile_$chmhost
    /bin/rm  $chmbuild/Makefile_$$
  endif
# add graphics to the objlibs.mk
  if ( $xreq == 1 ) then
   awk '/gamint/ { print "	$(LIB)/graphics.a \\" } \
        /./       {print $0}' \
                    $chmbuild/objlibs.mk > $chmbuild/objlibs.mk_$$
   if (1 != `grep graphics $chmbuild/objlibs.mk| wc -l` ) \
      /bin/mv $chmbuild/objlibs.mk_$$ $chmbuild/objlibs.mk
  endif
#
# Modify Makefile_template to replace gcc with ecc on Linux w/ efc
  if ( $chm_host == gnu && $efc == 1 ) then
   sed -e 's/CC = gcc/CC = ecc/' \
       $chmbuild/Makefile_$chmhost > $chmbuild/Makefile_$$
   /bin/mv $chmbuild/Makefile_$$ $chmbuild/Makefile_$chmhost
  endif
# Modify Makefile_template to replace gcc with icc on Linux w/ ifort
  if ( $chm_host == gnu && $ifort == 1 ) then
   sed -e 's/CC = gcc/CC = icc/' \
       $chmbuild/Makefile_$chmhost > $chmbuild/Makefile_$$
   /bin/mv $chmbuild/Makefile_$$ $chmbuild/Makefile_$chmhost
  endif
# Modify Makefile_template to replace gcc with cc -Dibmrs for MacOSX
  if ( $chm_host == osx ) then
    if( $xlf95 == 1) then
      sed -e 's/CC = gcc/CC = xlc/' \
          $chmbuild/Makefile_$chmhost > $chmbuild/Makefile_$$
	/bin/mv $chmbuild/Makefile_$$ $chmbuild/Makefile_$chmhost
	sed -e 's/-DCHARMM_GNUC/-Dibmrs -Dosx/' \
	    $chmbuild/Makefile_$chmhost > $chmbuild/Makefile_$$
	/bin/mv $chmbuild/Makefile_$$ $chmbuild/Makefile_$chmhost
    endif
    if( $ifort == 1 ) then
      sed -e 's/CC = gcc/CC = icc/' \
          $chmbuild/Makefile_$chmhost > $chmbuild/Makefile_$$
	/bin/mv $chmbuild/Makefile_$$ $chmbuild/Makefile_$chmhost
    endif
  endif
# Modify Makefile_template for GAMESS
  if ( $qgamess == 1 ) then
   awk '/gamint/ { print "	$(LIB)/gamess.a \\" } \
        /./       {print $0}' \
                    $chmbuild/objlibs.mk > $chmbuild/objlibs.mk_$$
   if (1 != `grep gamess $chmbuild/objlibs.mk| wc -l` ) \
      /bin/mv $chmbuild/objlibs.mk_$$ $chmbuild/objlibs.mk
  endif
# setup for APBS
  if ( $apbs == 1 && $chm_host == gnu ) then
      setenv APBS Y
      if ( ! $?APBS_LIB || ! $?IAPBS_LIB || ! $?APBS_BLAS ) then
        echo "APBS_LIB, IAPBS_LIB and APBS_BLAS variables must be set!"
	exit
      endif           
  endif
  if ( $apbs == 1 && $chm_host != gnu ) then
    echo "APBS build is supported on gnu platform only"
    exit
  endif

# Modify Makefile_template for long integer compilation
  if ( $longint == 1 ) then
      if ( $chm_host == gnu || $chm_host == osx ) then
        echo "x86_64 option deprecated on $chm_host; ignored"
      else
        echo "-i8 not available for this platform"
        echo "Update install.com and build/UNX/Makefile_$chmhost"
        exit -1
      endif
  endif

      
# Modify Makefile_template for GAMESS-UK
  if ( $qgamessuk == 1 ) then
   if ( ${?CHMGUK_USE_GA} == 0) then
      setenv CHMGUK_USE_GA 0
   endif
   if ( $CHMGUK_USE_GA == 1 ) then
     # Check there is a GA build available, may need to force i8
    if  ( $chmhost == gnu && $cfort == 1 ) then
       if ( $longint == 0 ) then
           echo "CHARMM/GAMESS-UK must be build with long integers (i8 keyword) if the GAs are required on" $chmhost
           exit -1
	endif
      endif

    awk '/gukint/ {print "	$(LIB)/gamessukga.a \\"; \
                  print "	$(LIB)/dft.a \\" } \
        /./       {print $0}' \
                    $chmbuild/objlibs.mk > $chmbuild/objlibs.mk_$$

   else
     awk '/gukint/ {print "	$(LIB)/gamessuk.a \\"; \
                  print "	$(LIB)/dft.a \\" } \
        /./       {print $0}' \
                    $chmbuild/objlibs.mk > $chmbuild/objlibs.mk_$$

   endif
    if ( 1 != `grep gamessuk $chmbuild/objlibs.mk|wc -l` ) \
      /bin/mv $chmbuild/objlibs.mk_$$ $chmbuild/objlibs.mk
#
# This section is needed to load additional libraries as needed 
# by GAMESS-UK
# For simplicity cluster network dependencies are left out here,
# so use mpispecial keyword to set comms flags and locations
#
      set BLAS  = ""

      if ( ${?CHMGUK_LIBS} == 1 ) then
          set BLAS=`echo $CHMGUK_LIBS`
      endif

      # Make the edits
      cat $chmbuild/Makefile_$chmhost | \
         sed -e "s@QLIB =@QLIB = $BLAS@" > $chmbuild/Makefile_$$
      mv $chmbuild/Makefile_$$ $chmbuild/Makefile_$chmhost

  endif
# Modify Makefile_template for CADPAC
  if ( $qcadpac == 1 ) then
   awk '/cadint/ { print "	$(LIB)/cadpac.a \\" } \
        /./       {print $0}' \
                    $chmbuild/objlibs.mk > $chmbuild/objlibs.mk_$$
    if ( 1 != `grep cadpack $chmbuild/objlibs.mk|wc -l` ) \
      /bin/mv $chmbuild/objlibs.mk_$$ $chmbuild/objlibs.mk
  endif
# Modify Makefile_template for SCCDFTB
  if ( $qsccdftb == 1 ) then
   awk '/quantum.a/ { print "	$(LIB)/sccdftb.a \\\n \t$(LIB)/sccdftbint.a \\"} \
        /./       {print $0}' \
                    $chmbuild/objlibs.mk > $chmbuild/objlibs.mk_$$
    if ( 1 != `grep sccdftb $chmbuild/objlibs.mk|wc -l` ) \
      /bin/mv $chmbuild/objlibs.mk_$$ $chmbuild/objlibs.mk
  endif
#
  if ( $qgamess == 1 ) then
    switch ($chmhost)
      case gnu:
        cp $chmroot/build/UNX/gmscomp $chmtool/gmscomp_$chmhost
        if ( $cfort == 1 ) then
           mv $chmtool/gmscomp_$chmhost $chmtool/gmscomp_$$
           sed -e 's@gamess-charmm-target@compaq-axp@g' \
                  $chmtool/gmscomp_$$ > $chmtool/gmscomp_$chmhost 
           sed -e 's@f77@fort@g' \
                  $chmtool/gmscomp_$chmhost > $chmtool/gmscomp_$$
        else
           sed -e 's@gamess-charmm-target@linux-pc@g' \
                  $chmtool/gmscomp_$chmhost > $chmtool/gmscomp_$$
        endif
        sed -e 's@activate-type@gnu@g' \
                $chmtool/gmscomp_$$ > $chmtool/gmscomp_$chmhost

#    ----- F77 ---------------------------------         
        if ( $f77 == 1 ) then
#         these modifications could also be implemented in the same way
#         as for the main Makefile_gnu
            sed -e \
      's@GNU-Linux-compiler@f77 -f -N3 -N26 -B108 -O -U -B100 -B101  -N86@g' \
                  $chmtool/gmscomp_$chmhost > $chmtool/gmscomp_$$
#    ----- F2C ---------------------------------         
        else if ( $f2c == 1 ) then
            sed -e 's@GNU-Linux-compiler@fort77 -O2 -w -malign-double@g' \
                    $chmtool/gmscomp_$chmhost > $chmtool/gmscomp_$$
#    ----- IFC ---------------------------------         
        else if ( $ifc == 1 ) then
          if ( $mpiset == 1 && $scali == 0 ) then
            sed -e 's@GNU-Linux-compiler@mpif77 -O3 -tpp7 -axW@g' \
                    $chmtool/gmscomp_$chmhost > $chmtool/gmscomp_$$
          else
            sed -e 's@GNU-Linux-compiler@ifc -O3 -tpp7 -axW@g' \
                   $chmtool/gmscomp_$chmhost > $chmtool/gmscomp_$$
          endif
#    ----- IFORT ---------------------------------         
        else if ( $ifort == 1 ) then
          if ( $mpiset == 1 && $scali == 0 ) then
            sed -e 's@GNU-Linux-compiler@mpif90 -O3 -tpp7 -axW@g' \
                    $chmtool/gmscomp_$chmhost > $chmtool/gmscomp_$$
          else
            sed -e 's@GNU-Linux-compiler@ifort -O3 -tpp7 -axW@g' \
                   $chmtool/gmscomp_$chmhost > $chmtool/gmscomp_$$
          endif
#    ----- PGF77 ---------------------------------         
        else if ( $pgf95 == 1 ) then
          if ( $mpiset == 1 && $scali == 0 ) then
            sed -e \
    's@GNU-Linux-compiler@mpif90 -O2 -Munroll -tp p7 -Mnoframe@g' \
                   $chmtool/gmscomp_$chmhost > $chmtool/gmscomp_$$
          else
            sed -e \
                's@GNU-Linux-compiler@pgf95 -O2 -Munroll -tp p7 -Mnoframe@g' \
                    $chmtool/gmscomp_$chmhost > $chmtool/gmscomp_$$
          endif
        else if ( $cfort == 1 ) then
            sed -e \
          's@GNU-Linux-compiler@fort -i8 -O4 -math_library fast -tune host@g' \
                    $chmtool/gmscomp_$chmhost > $chmtool/gmscomp_$$
#   ----- G95 ---------------------------------         
        else if ( $g95 == 1) then
          if ( $mpiset == 1 ) then
              sed -e 's@GNU-Linux-compiler@mpif90 -O3 @g' \
                      $chmtool/gmscomp_$chmhost > $chmtool/gmscomp_$$
          else
              sed -e 's@GNU-Linux-compiler@g95 -O3@g' \
                      $chmtool/gmscomp_$chmhost > $chmtool/gmscomp_$$
          endif
          exit
#   ----- G77 ---------------------------------         
        else
          if ( $mpiset == 1 ) then
              sed -e 's@GNU-Linux-compiler@mpif77 -malign-double -march=i686 -O2 -fno-globals -Wno-globals@g' \
                      $chmtool/gmscomp_$chmhost > $chmtool/gmscomp_$$
          else
              sed -e 's@GNU-Linux-compiler@g77 -malign-double -march=i686 -O2 -fno-globals -Wno-globals@g' \
                      $chmtool/gmscomp_$chmhost > $chmtool/gmscomp_$$
          endif
        endif
        /bin/mv $chmtool/gmscomp_$$ $chmtool/gmscomp_$chmhost
        chmod +x $chmtool/gmscomp_$chmhost
        breaksw
      default:
        echo "This machine does not support GAMESS"
        exit
        breaksw
    endsw
    if ( $mpiset != 1 ) then
        sed -e 's@PARALLEL=true@PARALLEL=false@g' \
        $chmtool/gmscomp_$chmhost > $chmtool/gmscomp_$$
        /bin/mv $chmtool/gmscomp_$$ $chmtool/gmscomp_$chmhost
        chmod +x $chmtool/gmscomp_$chmhost
    endif
  endif
# --- end of if qgamess==1 ----------------------------

# MH-12: disabled since Makefile_$$ doesn't exist here...
# Modify Makefile_template for Q-CHEM
#  if ( $qqchem == 1 ) then
#    /bin/mv $chmbuild/Makefile_$$ $chmbuild/Makefile_$chmhost
#  endif
# Modify Makefile_template for Turbomole
#  if ( $qqturbo == 1 ) then
#    /bin/mv $chmbuild/Makefile_$$ $chmbuild/Makefile_$chmhost
#  endif

# Modify Makefile_template for G09
  if ( $qg09 == 1 ) then
    /bin/mv $chmbuild/Makefile_$$ $chmbuild/Makefile_$chmhost
  endif

# Modify Makefile_template for SQUANTM
#--mfc--  if ( $qsquantm == 1 ) then
#--mfc--      switch ($chmhost)
#--mfc--        case ibmaixmp:
         awk '/util.a/ { print "	$(LIB)/squantm.a \\" } \
              /./      {print $0}' \
                     $chmbuild/objlibs.mk > $chmbuild/objlibs.mk_$$
         if ( 1 != `grep squantm $chmbuild/objlibs.mk|wc -l` ) \
            /bin/mv $chmbuild/objlibs.mk_$$ $chmbuild/objlibs.mk
#--mfc--          breaksw
#--mfc--        case ibmaix:
#--mfc--          awk '/util.a/ { print "	$(LIB)/squantm.a \\" } \
#--mfc--               /./      {print $0}' \
#--mfc--                      $chmbuild/objlibs.mk > $chmbuild/objlibs.mk_$$
#--mfc--          if ( 1 != `grep squantm objlibs.mk|wc -l` ) \
#--mfc--             /bin/mv $chmbuild/objlibs.mk_$$ $chmbuild/objlibs.mk
#--mfc--          breaksw
#--mfc--        case altix:
#--mfc--          awk '/util.a/ { print "	$(LIB)/squantm.a \\" } \
#--mfc--               /./      {print $0}' \
#--mfc--                      $chmbuild/objlibs.mk > $chmbuild/objlibs.mk_$$
#--mfc--          if ( 1 != `grep squantm objlibs.mk|wc -l` ) \
#--mfc--             /bin/mv $chmbuild/objlibs.mk_$$ $chmbuild/objlibs.mk
#--mfc--          breaksw
#--mfc--        case gnu:
#--mfc--          if ( $ifort == 1 || $f2c == 1 || $ifc == 1 || $gfortran == 1 ) then
#--mfc--             awk '/util.a/ { print "	$(LIB)/squantm.a \\" } \
#--mfc--                  /./      {print $0}' \
#--mfc--                      $chmbuild/objlibs.mk > $chmbuild/objlibs.mk_$$
#--mfc--             if ( 1 != `grep squantm objlibs.mk|wc -l` ) \
#--mfc--                /bin/mv $chmbuild/objlibs.mk_$$ $chmbuild/objlibs.mk
#--mfc--          else
#--mfc--               echo "This compiler does not suport SQUANTM"
#--mfc--               exit
#--mfc--          endif
#--mfc--          breaksw
#--mfc--        case osx:
#--mfc--          if ( $ifort == 1 || $f2c == 1 || $ifc == 1 || $gfortran == 1 ) then
#--mfc--             awk '/util.a/ { print "	$(LIB)/squantm.a \\" } \
#--mfc--                  /./      {print $0}' \
#--mfc--                      $chmbuild/objlibs.mk > $chmbuild/objlibs.mk_$$
#--mfc--             if ( 1 != `grep squantm objlibs.mk|wc -l` ) \
#--mfc--                /bin/mv $chmbuild/objlibs.mk_$$ $chmbuild/objlibs.mk
#--mfc--          else
#--mfc--               echo "This compiler does not suport SQUANTM"
#--mfc--               exit
#--mfc--          endif
#--mfc--          breaksw
#--mfc--        default:
#--mfc--          echo "This machine does not support  SQUANTM"
#--mfc-- #         exit
#--mfc--          breaksw
#--mfc--      endsw
#--mfc-- #--mfc--  endif

# Modify Makefile_template for FFTW
  if ( $fftw == 1 ) then
    if (! $?FFTW_HOME) then
      set fftw_wisdom = `which fftw-wisdom`
      if ($status == 0) then
	set fftw_bin = $fftw_wisdom:h
	setenv FFTW_HOME $fftw_bin:h
      else
	setenv FFTW_HOME /usr/local
      endif
    endif

    set fftw_intf = "include/fftw3.f03"
    if (-f $FFTW_HOME/$fftw_intf) then
      echo "Found $FFTW_HOME/$fftw_intf"
    else
      echo "Cannot find $fftw_intf"
      echo "Please set environment variable FFTW_HOME to a directory where"
      echo "FFTW 3.3 or later is installed with a modern Fortran interface."
      exit 1
    endif

    set fftw_lib_dirs = "-L$FFTW_HOME/lib"
    set fftw_libs = "-lfftw3"
    if (-f $FFTW_HOME/lib/libfftw3f.a) then
      echo "Found single precision FFTW at $FFTW_HOME/lib/libfftw3f.a"
      set fftw_libs = "-lfftw3 -lfftw3f"
    else if (-f $FFTW_HOME/lib64/libfftw3f.a) then
      echo "Found single precision FFTW at $FFTW_HOME/lib64/libfftw3f.a"
      set fftw_lib_dirs = "-L$FFTW_HOME/lib64 $fftw_lib_dirs"
      set fftw_libs = "$fftw_libs -lfftw3f"
    else
      # Disable COLFFT single precision
      set colfft_sp = 0
      echo "Unable to find single precision FFTW"
    endif

    sed -e "s@ADDLIB =@ADDLIB = $fftw_lib_dirs@" \
        -e "s@QLIB =@QLIB = $fftw_libs@" \
        -e "s@INCLUDE =@INCLUDE = -I$FFTW_HOME/include@" \
        < $chmbuild/Makefile_$chmhost > $chmbuild/Makefile_$$


    /bin/mv $chmbuild/Makefile_$$ $chmbuild/Makefile_$chmhost
  endif

# Modify Makefile_template for MKL
  if ( $mkl == 1 ) then

    if (! $?MKLROOT) then
      set mkl_bin = `which ifort`
      if ( $status == 0 ) then
        setenv MKLROOT $mkl_bin:h:h:h/mkl
      else
        setenv MKLROOT /opt/intel/mkl
      endif
    endif

    if (-f $MKLROOT/include/fftw/fftw3.f) then
      echo "Found $MKLROOT/include/fftw/fftw3.f"
    else
      echo "Cannot find fftw3.f in $MKLROOT"
      exit 1
    endif

    if ( $chmhost == 'mic' ) then
      sed -e "s@ADDLIB =@ADDLIB = -L$MKLROOT/lib/mic@" \
	  -e "s@QLIB =@QLIB = -mkl=sequential@" \
          -e "s@INCLUDE =@INCLUDE = -mkl=parallel -I$MKLROOT/include/fftw@" \
	  < $chmbuild/Makefile_$chmhost > $chmbuild/Makefile_$$
    else
      sed -e "s@ADDLIB =@ADDLIB = -L$MKLROOT/lib@" \
	  -e "s@QLIB =@QLIB = -mkl=sequential@" \
          -e "s@INCLUDE =@INCLUDE = -mkl=sequential -I$MKLROOT/include/fftw@" \
	  < $chmbuild/Makefile_$chmhost > $chmbuild/Makefile_$$
    endif
    /bin/mv $chmbuild/Makefile_$$ $chmbuild/Makefile_$chmhost
  endif


  
  if ($openmp == 1 && $chm_host == em64t) then
    set ifort_version = `ifort --version | head -1 | cut -f3 -d' ' \
                                         | cut -f1 -d'.'`
    set openmp_flag = -openmp
    if ($ifort_version >= 15) then
      set openmp_flag = -qopenmp
    endif
    
    sed -e "s@OPENMP_FLAG = .*@OPENMP_FLAG = $openmp_flag@" \
        < $chmbuild/Makefile_$chmhost > $chmbuild/Makefile_$$
    /bin/mv $chmbuild/Makefile_$$ $chmbuild/Makefile_$chmhost
  endif

# Modify Makefile_template for DOMDEC_GPU (=add nvcc compiler)
  if ( $domdec_gpu == 1 ) then
    set nvcc_bin = `which nvcc`
    if ( $status == 0 ) then
	setenv CUDAC_ROOT $nvcc_bin:h:h
    else
        echo "Cannot find nvcc CUDA compiler, make sure you have correctly installed CUDA."
        echo "nvcc compiler is required for compiling CHARMM with DOMDEC_GPU switch."
        exit 1
    endif

    # this used to rely on chmhost being either osx or one of em64t, gnu
    # but some people compile with em64t and gnu on OSX hosts ...
    set actual_host_os = `uname -s`
    set domdec_gpu_addlib = "-L$CUDAC_ROOT/lib"
    set domdec_gpu_qlib = "-lcudart -lnvToolsExt -lcufft"
    if ( "$actual_host_os" == 'Darwin' ) then
      set domdec_gpu_qlib = "$domdec_gpu_qlib -lpthread"
    else
      set domdec_gpu_addlib = "${domdec_gpu_addlib}64"
    endif

    if ( $ifort == 1 ) then
      set domdec_gpu_qlib = "$domdec_gpu_qlib -liomp5"
    endif

    sed -e "s@ADDLIB =@ADDLIB = $domdec_gpu_addlib@" \
        -e "s@QLIB =@QLIB = $domdec_gpu_qlib@" \
        -e "s@INCLUDE =@INCLUDE = -I$CUDAC_ROOT/include@" \
        < $chmbuild/Makefile_$chmhost > $chmbuild/Makefile_$$
    /bin/mv $chmbuild/Makefile_$$ $chmbuild/Makefile_$chmhost

  endif

# Modify Makefile_template for QMMMSEMI
  if ( $qmmmsemi == 1 ) then
     awk '/util.a/ { print "	$(LIB)/qmmmsemi.a \\" } \
          /./      {print $0}' \
              $chmbuild/objlibs.mk > $chmbuild/objlibs.mk_$$
     if ( 1 != `grep qmmmsemi $chmbuild/objlibs.mk|wc -l` ) \
        /bin/mv $chmbuild/objlibs.mk_$$ $chmbuild/objlibs.mk
  endif

# PJ 06/2005
# Modify Makefile_template for PIPF
#--mfc  if ( $qpipf == 1 ) then
   awk '/quantum/ { print "	$(LIB)/pipf.a \\" } \
        /./       {print $0}' \
                    $chmbuild/objlibs.mk > $chmbuild/objlibs.mk_$$
   if ( 1 != `grep pipf $chmbuild/objlibs.mk|wc -l` ) \
              /bin/mv $chmbuild/objlibs.mk_$$ $chmbuild/objlibs.mk
#--mfc   endif

######## Charmm's OpenMM plugin setup ##########
   if ( $openmm == 1 ) then
      # try to find OpenMM install dir
      if ( ! $?OPENMM_DIR ) then
        if ( $?OPENMM_HOME ) then
          setenv OPENMM_DIR $OPENMM_HOME
        else if ( $?OPENMM_PLUGIN_DIR ) then
          setenv OPENMM_DIR $OPENMM_PLUGIN_DIR:h:h
        else
          setenv OPENMM_DIR /usr/local/openmm
        endif
      endif

      # try to find OpenMM packaged-plugin dir
      if ( ! $?OPENMM_PLUGIN_DIR ) then
        setenv OPENMM_PLUGIN_DIR "$OPENMM_DIR/lib/plugins"
      endif
      set omm_lib = $OPENMM_PLUGIN_DIR:h

      setenv charmm_openmm_plugins_dir $chmlib/openmm_plugins
      mkdir "$charmm_openmm_plugins_dir"

      #### Find a reasonable default C++ compiler
      setenv omm_cc `which g++`
      if ($chm_host == em64t || $ifort == 1) then
	      setenv omm_cc `which icc`
        if ( $status != 0 ) then
          echo " install.com> ERROR: Intel C++ compiler icc not found on path"
          echo "              attempting to default to g++"
          setenv omm_cc `which g++`
        endif
      endif
      echo " install.com> using $omm_cc to compile OpenMM Plugins"

      if ( ! $?CUDATK ) then
        if ( $?CUDA_HOME ) then
          setenv CUDATK "$CUDA_HOME"
        else
          set nvcc = `which nvcc`
          if ( $status == 0 ) then
            set cudabin = $nvcc:h
            setenv CUDATK $cudabin:h
          else
            setenv CUDATK /usr/local/cuda
          endif
        endif
      endif

      if ( -d "$CUDATK" ) then
        echo " install.com> using $CUDATK to provide CUDA for OpenMM"
      else
        echo " install.com> ERROR: environment variable CUDATK not found"
        echo "              Please set CUDATK to the root of your CUDA toolkit installation"
      endif

      setenv SHLIB so
      setenv MAC_CUDA ''
      setenv MAC_OPENCL ''
      if ($chm_host == 'osx') then
	      setenv SHLIB dylib
	      setenv MAC_CUDA '-framework CUDA'
	      setenv MAC_OPENCL '-framework OpenCL'
      endif

      set linker_rpath = ""
      set omm_intf = "include/OpenMMFortranModule.f90"
      if (-f $OPENMM_DIR/$omm_intf) then
        echo " install.com> Found $omm_intf"
	      set omm_plug_prefix="$chmlib/openmm_plugins/libOpenMMCharmm"
	      set omm_plug_so="$omm_plug_prefix.so"
	      set ommplugin = 1
	      set linker_rpath = "-Wl,-rpath,$omm_lib,-rpath,$omm_plug_so:h"
        set omm_libs = " -lOpenMMCharmm -lOpenMMGBSW -lOpenMM "
      else
        echo " install.com> ERROR: Cannot find"
        echo "              $OPENMM_DIR/$omm_intf"
        echo "              Please set environment variable"
        echo "              OPENMM_DIR or OPENMM_PLUGIN_DIR"
        echo "              appropriately."
        exit 1
      endif

      # please leave for future reference
      # discover which version of OpenMM we are dealing with
      # set omm_version_forcom = gfortran
      # if ($chm_host == 'em64t' || ($chm_host == 'osx' && $ifort == 1)) then
      #       set omm_version_forcom = ifort
      # endif
      #
      # version program already compiled? Then skip it.
      # if ((! -e $chmbuild/version) || (! -x $chmbuild/version)) then
      #   pushd "$chmbuild" > /dev/null
      #   $omm_version_forcom \
      #     -I"$OPENMM_DIR/include" \
      #     -L"$OPENMM_DIR/lib" -lOpenMM \
      #     -Wl,-rpath,"$OPENMM_DIR/lib" \
      #     "$chmtool/OpenMMFiles/version.f90" \
      #     -o "$chmbuild/version"   
      #   popd > /dev/null
      # endif
      #
      # failsafe default
      # setenv OPENMM_VERSION '0.0'  
      # check whether above compilation worked
      # if ((-e $chmbuild/version) && (-x $chmbuild/version)) then        
      #   setenv OPENMM_VERSION `$chmbuild/version`
      # endif

      # must leave in vvv to compile against openmm >= 7.1 Feb 2017
      setenv omm_defines "-D_GLIBCXX_USE_CXX11_ABI=0"
      grep -q -w getDefaultTemperature "$OPENMM_DIR/include/openmm/MonteCarloBarostat.h"
      if ( $status == 0 ) then
        setenv omm_defines "-DOPENMM_API_UPDATE $omm_defines"
      endif

      set plugin_dir="$chmsrc/openmm/plugins"
      sed -e "s@ADDLIB =@ADDLIB = -L$omm_lib -L$chmlib/openmm_plugins $linker_rpath @" \
          -e "s@QLIB =@QLIB = $omm_libs -ldl @" \
          -e "s@INCLUDE =@INCLUDE = $omm_defines -I"$chmbuild" -I$OPENMM_DIR/include -I$plugin_dir/MonteCarloBarostat2/wrappers @" \
          < $chmbuild/Makefile_$chmhost > $chmbuild/Makefile_$$
      /bin/mv $chmbuild/Makefile_$$ $chmbuild/Makefile_$chmhost

      sed -e "s@omm_cc =@omm_cc = $omm_cc -D_GLIBCXX_USE_CXX11_ABI=0@" \
            < "$chmbuild/../UNX/Makefile_omm" > "$chmbuild/Makefile_omm"
      sed -ie "s@SHLIB =@SHLIB = $SHLIB@" \
            "$chmbuild/Makefile_omm"
      sed -ie "s@MAC_CUDA =@MAC_CUDA = $MAC_CUDA@" \
            "$chmbuild/Makefile_omm"
      sed -ie "s@MAC_OPENCL =@MAC_OPENCL = $MAC_OPENCL@" \
            "$chmbuild/Makefile_omm"
      sed -ie "s@CUDATK =@CUDATK = $CUDATK@" \
            "$chmbuild/Makefile_omm"
      sed -ie "s@OPENMM_DIR =@OPENMM_DIR = $OPENMM_DIR@" \
            "$chmbuild/Makefile_omm"
      sed -ie "s@OPENMM_PLUGIN_DIR =@OPENMM_PLUGIN_DIR = $OPENMM_PLUGIN_DIR@" \
            "$chmbuild/Makefile_omm"
      sed -ie "s@CHARMM_PLUGIN_DIR =@CHARMM_PLUGIN_DIR = $chmlib/openmm_plugins@" \
            "$chmbuild/Makefile_omm"
      if (-f "$chmbuild/Makefile_omme") then
        rm "$chmbuild/Makefile_omme"
      endif
    endif
######## End of Charmm's OpenMM plugin setup ##########

# JMS 5/2012
#  Modify makefiles to include lapack for GAMUS
   if ( $gamus == 1) then
      echo "GAMUS requires an installation of LAPACK.  Please enter the link options for LAPACK (e.g. -lblas -llapack)"
      echo -n ">> "
      set lapackopts = "$<"
      #echo $lapackopts
      #awk -v lapackopts="$lapackopts" '/QLIB =/ { print "QLIB = "lapackopts; next} /./ {print $0}' \
      #          $chmbuild/Makefile_$chmhost > $chmbuild/Makefile_$$
      cat $chmbuild/Makefile_$chmhost | \
               sed -e "s@QLIB =@QLIB = $lapackopts@" > $chmbuild/Makefile_$$
      /bin/mv $chmbuild/Makefile_$$ $chmbuild/Makefile_$chmhost
      awk '/gamint/ { print "\t$(LIB)/gamus.a \\" } \
           /./      {print $0}' \
                    $chmbuild/objlibs.mk > $chmbuild/objlibs.mk_$$
      if ( 1 != `grep gamint objlibs.mk|wc -l` ) \
              /bin/mv $chmbuild/objlibs.mk_$$ $chmbuild/objlibs.mk
   endif

endif  # new_makefile

if ( $stopmark == 1 ) then
  echo " "
  echo " install.com> Installation STOPs after creating "
  echo "              the $chmbuild directory."
  echo " "
  exit 1
endif
#
if (! -e $chmtool  ) then
  echo " "
  echo " install.com> Directory $chmtool does not exist."
  echo "              CHARMM installation can not proceed."
  exit 1
endif
echo " "
echo " install.com> Phase 1 completed."
#
#-----------------------------------------------------------------------
#
phase2:
# (2.1) generate the PREFX preprocessor 
# (2.2) generate the PREFX data file PREF.DAT
#
#
pushd $chmtool >> /dev/null
#

if ( $domdec_gpu == 1 ) then
    setenv USE_CUDAC YES
endif

if ( $openmp == 1) then
    setenv OPENMP YES
endif

if ( $pthread == 1) then
    setenv PTHREAD YES
endif

switch ($chm_host)
  case gnu:
    set forcom = gfortran
    if ( $g95 == 1 ) set forcom = g95
    if ( $f77 == 1 ) set forcom = f77
    if ( $f2c == 1 ) set forcom = fort77
    if ( $g77 == 1 ) set forcom = g77
    if ( $ifc == 1 ) set forcom = ifc
    if ( $efc == 1 ) set forcom = efc
    if ( $ifort == 1 ) set forcom = ifort
    if ( $pgf95 == 1 ) set forcom = pgf95
    if ( $cfort == 1 ) set forcom = fort
    if ( $pathscale == 1 ) set forcom = pathf95
    # $forcom -g -o prefx_$$  prefx.f90
    $forcom -g -o qchem  qchem.f90
    echo "GNU"      >! $chmbuild/pref$$.dat
    if ($g95 == 1) echo "G95" >> $chmbuild/pref$$.dat
    echo "UNIX"     >> $chmbuild/pref$$.dat
    if ( $x86_64 == 1 ) then
	echo "I4BINARY"   >> $chmbuild/pref$$.dat
    endif
    if($pathscale == 1) echo "PATHSCALE"  >> $chmbuild/pref$$.dat
    if($gfortran == 1 ) then
	echo "GFORTRAN"   >> $chmbuild/pref$$.dat
    endif
    if($mpiset == 1 ) then
       if ($cmpi_ok == 1) echo "CMPI" >> $chmbuild/pref$$.dat
       echo "MPI"        >> $chmbuild/pref$$.dat
       echo "PARALLEL"   >> $chmbuild/pref$$.dat
       echo "PARAFULL"   >> $chmbuild/pref$$.dat
    endif
    breaksw
  case g95:
    set forcom = g95
    # $forcom -o prefx_$$  prefx.f90
    echo "GNU"      >! $chmbuild/pref$$.dat
    echo "G95"      >> $chmbuild/pref$$.dat
    echo "UNIX"     >> $chmbuild/pref$$.dat
    if ( $pvmset == 1 || $socket == 1 ) then
        echo "PVM and SOCKET not supported for g95"
        exit -1
    endif
    if ( $x86_64 == 1 ) then
	echo "I4BINARY"   >> $chmbuild/pref$$.dat
    endif
    if($mpiset == 1 ) then
       if ($cmpi_ok == 1) echo "CMPI" >> $chmbuild/pref$$.dat
       echo "MPI"        >> $chmbuild/pref$$.dat
       echo "PARALLEL"   >> $chmbuild/pref$$.dat
       echo "PARAFULL"   >> $chmbuild/pref$$.dat
    endif
    breaksw
#------------------ LINUX NAG --------------------------------
  case nag:
    # nagfor -o prefx_$$  prefx.f90
    echo "GNU"      >!  $chmbuild/pref$$.dat
    echo "NAG"      >>  $chmbuild/pref$$.dat
    echo "UNIX"     >>  $chmbuild/pref$$.dat
    echo "I4BINARY"    >> $chmbuild/pref$$.dat
    if ( $pvmset == 1 ) then
	echo "PVM not supported for em64t"
        exit 2
    endif
    if ( $mpiset == 1 ) then
        echo "MPI"         >> $chmbuild/pref$$.dat
        echo "PARALLEL"    >> $chmbuild/pref$$.dat
        echo "PARAFULL"    >> $chmbuild/pref$$.dat
        echo "IA64_MPI"    >> $chmbuild/pref$$.dat
	setenv MPI_EM64T YES
    endif
    breaksw
#------------------ Pentium 4 EM64T --------------------------------
  case em64t:
    # ifort -o prefx_$$  prefx.f90
    echo "GNU"      >!  $chmbuild/pref$$.dat
    echo "EM64T"    >>  $chmbuild/pref$$.dat
    echo "UNIX"     >>  $chmbuild/pref$$.dat
    echo "I4BINARY"    >> $chmbuild/pref$$.dat
    if ( $pvmset == 1 ) then
	echo "PVM not supported for em64t"
        exit 2
    endif
    if ( $mpiset == 1 ) then
        echo "MPI"         >> $chmbuild/pref$$.dat
        echo "PARALLEL"    >> $chmbuild/pref$$.dat
        echo "PARAFULL"    >> $chmbuild/pref$$.dat
	setenv MPI_EM64T YES
    endif
    # If mpif90 does not exist, assume the new Intel compiler "mpiifort"
    set mpifc = 'which mpif90'
    if ( $status == 1 ) then
        setenv MPIIFORT YES
    endif
    breaksw
#---------- Cray XT4/XT5 runnin compute node linux -----------------
#Use with MPI e.g.
#./install.com xt4 xlarge MPI M nolog
  case xt4:
    # ftn -o prefx_$$  prefx.f90
    echo "GNU"      >!  $chmbuild/pref$$.dat
    echo "XT4"    >>  $chmbuild/pref$$.dat
    echo "UNIX"     >>  $chmbuild/pref$$.dat
    echo "I4BINARY"    >> $chmbuild/pref$$.dat
    if ( $pvmset == 1 ) then
	echo "PVM not supported for Cray XT4"
        exit 2
    endif
    if ( $mpiset == 1 ) then
        echo "MPI"         >> $chmbuild/pref$$.dat
        echo "PARALLEL"    >> $chmbuild/pref$$.dat
        echo "PARAFULL"    >> $chmbuild/pref$$.dat
        echo "IA64_MPI"    >> $chmbuild/pref$$.dat
    endif
    breaksw
#-------------------------- OSX ------------------------------------
  case osx:
    set forcom = gfortran
    if ( $xlf95 == 1 ) then	
	echo "OSX"      >! $chmbuild/pref$$.dat
	set forcom = xlf95
    else if ( $ifort == 1 || $g95 == 1 ) then
	echo "GNU"      >! $chmbuild/pref$$.dat
	if ( $ifort == 1 ) then
	    set forcom = ifort
	    if( $longint == 1 ) then
		echo "I4BINARY"   >>! $chmbuild/pref$$.dat
	    endif
	else if ( $g95 == 1 ) then
	    set forcom = g95
	    echo "G95"   >>! $chmbuild/pref$$.dat
	endif
    else
	echo "GNU"      >! $chmbuild/pref$$.dat
	echo "GFORTRAN"   >>! $chmbuild/pref$$.dat
    endif
      # if ( $xlf95 == 1 ) then 
      #       $forcom -qfixed -o prefx_$$  prefx.f90
      # else
      #       $forcom -o prefx_$$  prefx.f90
      # endif
    echo "UNIX"     >> $chmbuild/pref$$.dat
    if($mpiset == 1 ) then
       echo "MPI"        >> $chmbuild/pref$$.dat
       echo "PARALLEL"   >> $chmbuild/pref$$.dat
       echo "PARAFULL"   >> $chmbuild/pref$$.dat
    endif
    breaksw
  case gpu:
    set forcom = gfortran
    if ( $ifort == 1 ) then
	echo "GNU"      >! $chmbuild/pref$$.dat
        set forcom = ifort
    else
	echo "GNU"      >! $chmbuild/pref$$.dat
#	echo "GFORTRAN"   >>! $chmbuild/pref$$.dat
    endif
    # if (! -e prefx_gpu  ) then
    #     $forcom -o prefx_$$  prefx.f90
    # endif
    echo "UNIX"     >> $chmbuild/pref$$.dat
    echo "GRAPE"    >> $chmbuild/pref$$.dat
#    echo "I4BINARY" >> $chmbuild/pref$$.dat
    if($mpiset == 1 ) then
#      GENCOMM is needed since we usually calculate KSPACE on separate process
       if ($cmpi_ok == 1) echo "CMPI" >> $chmbuild/pref$$.dat
       echo "MPI"        >> $chmbuild/pref$$.dat
       echo "PARALLEL"   >> $chmbuild/pref$$.dat
       echo "GENCOMM"    >> $chmbuild/pref$$.dat
       echo "PARAFULL"   >> $chmbuild/pref$$.dat
       echo "ASYNC_PME"  >> $chmbuild/pref$$.dat
       echo "PARCMD"     >> $chmbuild/pref$$.dat
    endif
    breaksw
  default:
    echo " install.com> Unsupported host-machine-type specified."
    exit
    breaksw
endsw
#
# if ( -e prefx_$$ ) then
#   mv prefx_$$ prefx_$chm_host
#   echo " "
#   echo " install.com> The preprocessor prefx_$chm_host installed."
# endif
#
popd >> /dev/null
#

#
# generic preprocessing keys to build CHARMM
# PUTFXM required in prex.src
  echo "EXPAND"     >> $chmbuild/pref$$.dat
  echo "PUTFCM"     >> $chmbuild/pref$$.dat
#  echo "FCMDIR=fcm" >> $chmbuild/pref$$.dat
#
# specify parallel platform preprocessor keys
# and supported features to non-parallel platforms
  if ( $full == 1 ) then
    echo "ACE"        >> $chmbuild/pref$$.dat
    echo "ADUMB"      >> $chmbuild/pref$$.dat
    echo "AFM"        >> $chmbuild/pref$$.dat
    if ( $apbs == 1 ) then
      echo "APBS"     >> $chmbuild/pref$$.dat
    endif
    echo "ASPENER"    >> $chmbuild/pref$$.dat
    echo "ASPMEMB"    >> $chmbuild/pref$$.dat
    echo "AXD"        >> $chmbuild/pref$$.dat
    echo "BLOCK"      >> $chmbuild/pref$$.dat
    echo "CFF"        >> $chmbuild/pref$$.dat
    echo "CGENFF"     >> $chmbuild/pref$$.dat
    echo "CHEQ"       >> $chmbuild/pref$$.dat
    echo "CMAP"       >> $chmbuild/pref$$.dat
    echo "CONSHELIX"  >> $chmbuild/pref$$.dat
    echo "CPATH"      >> $chmbuild/pref$$.dat
    echo "DIMB"       >> $chmbuild/pref$$.dat
    echo "DMCONS"     >> $chmbuild/pref$$.dat
    echo "DOCK"       >> $chmbuild/pref$$.dat
    echo "DYNVV2"     >> $chmbuild/pref$$.dat
    echo "EMAP"       >> $chmbuild/pref$$.dat
    echo "EPMF"       >> $chmbuild/pref$$.dat
    echo "ESTATS"     >> $chmbuild/pref$$.dat
    echo "FASTEW"     >> $chmbuild/pref$$.dat
    echo "FLEXPARM"   >> $chmbuild/pref$$.dat
    echo "FLUCQ"      >> $chmbuild/pref$$.dat
    echo "FMA"        >> $chmbuild/pref$$.dat
    echo "FOURD"      >> $chmbuild/pref$$.dat
    echo "FSSHK"      >> $chmbuild/pref$$.dat
    echo "GBFIXAT"    >> $chmbuild/pref$$.dat
    echo "GBINLINE"   >> $chmbuild/pref$$.dat
    echo "HFB"        >> $chmbuild/pref$$.dat
    echo "HQBM"       >> $chmbuild/pref$$.dat
    echo "GCMC"       >> $chmbuild/pref$$.dat
    echo "GENETIC"    >> $chmbuild/pref$$.dat
    echo "GRID"       >> $chmbuild/pref$$.dat
    echo "GSBP"       >> $chmbuild/pref$$.dat
    echo "SMBP"       >> $chmbuild/pref$$.dat
    echo "HMCOM"      >> $chmbuild/pref$$.dat
    echo "IMCUBES"    >> $chmbuild/pref$$.dat
    echo "LARMORD"    >> $chmbuild/pref$$.dat
    echo "LONEPAIR"   >> $chmbuild/pref$$.dat
    echo "LRVDW"      >> $chmbuild/pref$$.dat
    echo "MC"         >> $chmbuild/pref$$.dat
    echo "MEHMC"      >> $chmbuild/pref$$.dat
    echo "MMFF"       >> $chmbuild/pref$$.dat
    echo "MMPT"       >> $chmbuild/pref$$.dat
    echo "MOLVIB"     >> $chmbuild/pref$$.dat
    echo "MSMMPT"     >> $chmbuild/pref$$.dat
    echo "MTPL"       >> $chmbuild/pref$$.dat
#    echo "MTS"        >> $chmbuild/pref$$.dat
    echo "MULTCAN"    >> $chmbuild/pref$$.dat
    echo "NBIPS"      >> $chmbuild/pref$$.dat
    echo "OLDDYN"     >> $chmbuild/pref$$.dat
    echo "OPLS"       >> $chmbuild/pref$$.dat
    echo "OVERLAP"    >> $chmbuild/pref$$.dat
    echo "PATHINT"    >> $chmbuild/pref$$.dat
    echo "PBEQ"       >> $chmbuild/pref$$.dat
    echo "PBOUND"     >> $chmbuild/pref$$.dat
    echo "PERT"       >> $chmbuild/pref$$.dat
    echo "PHMD"       >> $chmbuild/pref$$.dat
# PJ 06/2005
    if ($qpipf == 1) then
       echo "PIPF"       >> $chmbuild/pref$$.dat
    endif
#    echo "POLAR"      >> $chmbuild/pref$$.dat
    echo "PM1"        >> $chmbuild/pref$$.dat
    echo "PMEPLSMA"   >> $chmbuild/pref$$.dat
    echo "PNOE"       >> $chmbuild/pref$$.dat
    echo "PRIMO"      >> $chmbuild/pref$$.dat
    echo "PRIMSH"     >> $chmbuild/pref$$.dat
    echo "RDC"        >> $chmbuild/pref$$.dat
    echo "RDFSOL"     >> $chmbuild/pref$$.dat
    echo "REPLICA"    >> $chmbuild/pref$$.dat
    echo "RGYCONS"    >> $chmbuild/pref$$.dat
#    echo "RISM"       >> $chmbuild/pref$$.dat
    echo "RPATH"      >> $chmbuild/pref$$.dat
    echo "RXNCOR"     >> $chmbuild/pref$$.dat
    echo "SASAE"      >> $chmbuild/pref$$.dat
    echo "SCPISM"     >> $chmbuild/pref$$.dat
    echo "SGLD"       >> $chmbuild/pref$$.dat
    echo "SHAPES"     >> $chmbuild/pref$$.dat
    echo "SHELL"      >> $chmbuild/pref$$.dat
    echo "SOFTVDW"    >> $chmbuild/pref$$.dat
#    echo "SPAS"       >> $chmbuild/pref$$.dat
    echo "SSNMR"      >> $chmbuild/pref$$.dat
#    echo "TESTENDIAN" >> $chmbuild/pref$$.dat
    echo "TNPACK"     >> $chmbuild/pref$$.dat
    echo "TPS"        >> $chmbuild/pref$$.dat
    echo "TRAVEL"     >> $chmbuild/pref$$.dat
    echo "TSM"        >> $chmbuild/pref$$.dat
    echo "TSALLIS"    >> $chmbuild/pref$$.dat
# New in c32a2
    echo "FITCHG"     >> $chmbuild/pref$$.dat
    echo "WCA"        >> $chmbuild/pref$$.dat
# New in c33a3
    echo "CHEMPERT"   >> $chmbuild/pref$$.dat
#    echo "CORSOL"     >> $chmbuild/pref$$.dat
#    echo "FEWMFC"     >> $chmbuild/pref$$.dat
#    echo "PBCUBES"    >> $chmbuild/pref$$.dat
    echo "PROTO"      >> $chmbuild/pref$$.dat
    echo "SMD"        >> $chmbuild/pref$$.dat
# New in c35a1
#    echo "ACTBOND"    >> $chmbuild/pref$$.dat
#    echo "COLFFT"     >> $chmbuild/pref$$.dat
    echo "COMP2"      >> $chmbuild/pref$$.dat
    echo "FACTS"      >> $chmbuild/pref$$.dat
    echo "LOOKUP"     >> $chmbuild/pref$$.dat
#    echo "MSCALE"     >> $chmbuild/pref$$.dat
    echo "RMD "       >> $chmbuild/pref$$.dat
    echo "VALBOND "       >> $chmbuild/pref$$.dat
    echo "RXNCONS"    >> $chmbuild/pref$$.dat
#    echo "ZEROM"      >> $chmbuild/pref$$.dat
# New in c35a2
    echo "GNN"        >> $chmbuild/pref$$.dat
# New in c39a2
    echo "MRMD"        >> $chmbuild/pref$$.dat
# New in c41a1
# for mndo97 and squantm cause error from compiling with QCHEM keywords
    if ( $qmndo97 == 0 && $qsquantm == 0 ) then
       echo "QCHEM"      >> $chmbuild/pref$$.dat
    endif
    echo "HDGBVDW"    >> $chmbuild/pref$$.dat
    echo "DENBIAS"    >> $chmbuild/pref$$.dat
    echo "DHDGB"    >> $chmbuild/pref$$.dat
####
   if ( $gamus == 1 ) then
    echo "GAMUS"      >> $chmbuild/pref$$.dat
   endif   
    if ( $mpiset == 0 ) \
       echo "TAMD"       >> $chmbuild/pref$$.dat
    if ( $openmm == 1 ) then
       echo "OPENMM"     >> $chmbuild/pref$$.dat
       if ( $ommplugin == 1 ) then
          echo "OMMPLUGIN" >> $chmbuild/pref$$.dat
       endif
    endif
#---end of new feature keys
    if ( $qgamess == 1 ) then
      echo "GAMESS"     >> $chmbuild/pref$$.dat
    else if ( $qgamessuk == 1 ) then
      echo "GAMESSUK"   >> $chmbuild/pref$$.dat
      if ( $qsquantm == 1 ) then
        echo "SQUANTM"    >> $chmbuild/pref$$.dat
      endif
    else if ( $qcadpac == 1 ) then
      echo "CADPAC"     >> $chmbuild/pref$$.dat
    else if ( $qsccdftb == 1 ) then
      echo "SCCDFTB"    >> $chmbuild/pref$$.dat
    else if ( $qqchem == 1 ) then
      echo "QCHEM"      >> $chmbuild/pref$$.dat
    else if ( $qg09 == 1 ) then
      echo "G09"      >> $chmbuild/pref$$.dat
    else if ( $qqturbo == 1 ) then
      echo "QTURBO"      >> $chmbuild/pref$$.dat
    else if ( $qmndo97 == 1 ) then
      echo "MNDO97"     >> $chmbuild/pref$$.dat
    else if ( $qsquantm == 1 ) then
      if ( $qgamessuk == 1 ) then
        echo "GAMESSUK"   >> $chmbuild/pref$$.dat
      endif
      echo "SQUANTM"    >> $chmbuild/pref$$.dat
    else if ( $qmmmsemi == 1 ) then
      echo "QMMMSEMI"    >> $chmbuild/pref$$.dat
    else
      echo "QUANTUM"    >> $chmbuild/pref$$.dat
    endif
  endif
endif

if ( $fftw == 1 ) then
    echo "FFTW"     >> $chmbuild/pref$$.dat
    if ($colfft_sp == 0) then
       echo "COLFFT_NOSP"     >> $chmbuild/pref$$.dat
    endif
    echo "COLFFT"   >> $chmbuild/pref$$.dat
endif

if ( $mkl == 1 ) then
    echo "MKL"      >> $chmbuild/pref$$.dat
    echo "COLFFT"   >> $chmbuild/pref$$.dat
endif

if ( $domdec == 1 ) then
    echo "DOMDEC"   >> $chmbuild/pref$$.dat
    if ( $fftw == 0 && $mkl == 0 ) then
	echo "COLFFT"   >> $chmbuild/pref$$.dat
    endif
endif

if ( $domdec_gpu == 1 ) then
    echo "DOMDEC_GPU">> $chmbuild/pref$$.dat
    if ( $fftw == 0 && $mkl == 0 ) then
	echo "COLFFT"   >> $chmbuild/pref$$.dat
    endif
endif

#
if ( $nih == 1 ) then
  echo "LONGLINE"    >> $chmbuild/pref$$.dat
  echo "NIH"         >> $chmbuild/pref$$.dat
  echo "SAVEFCM"     >> $chmbuild/pref$$.dat
  echo "SHAPES"      >> $chmbuild/pref$$.dat
  echo "SGLD"        >> $chmbuild/pref$$.dat
#
endif
#
if ( $tsri == 1 ) then
  echo "PMEPLSMA"   >> $chmbuild/pref$$.dat
  echo "IMCUBES"    >> $chmbuild/pref$$.dat
  echo "GBINLINE"   >> $chmbuild/pref$$.dat
  echo "DMCONS"     >> $chmbuild/pref$$.dat
  echo "RGYCONS"    >> $chmbuild/pref$$.dat
endif
# polyrate control
if ( $polyrate ) then
    echo "CHARMMRATE" >> $chmbuild/pref$$.dat
endif

#EM 01/2019
if ($qdcm == 1) then 
    echo "DCM"      >> $chmbuild/pref$$.dat
endif

#
if ( $ensemble ) then
    echo "ENSEMBLE"    >> $chmbuild/pref$$.dat
endif
if ( $qabpo ) then 
     echo "ABPO"     >> $chmbuild/pref$$.dat
endif
if ( $qstringm == 1 ) then 
     echo "STRINGM"      >> $chmbuild/pref$$.dat
     echo "MULTICOM"     >> $chmbuild/pref$$.dat
     echo "NEWBESTFIT"   >> $chmbuild/pref$$.dat
endif

#
# ----------------
if ( $xreq ) then
  echo "XDISPLAY"   >> $chmbuild/pref$$.dat
else if ( $nodsp ) then
  echo "NODISPLAY"  >> $chmbuild/pref$$.dat
else
  echo "NOGRAPHICS" >> $chmbuild/pref$$.dat
endif
echo "END"          >> $chmbuild/pref$$.dat
#
if ( $stopmark == 2 ) then
  echo " "
  echo " install.com> Installation STOPs after creating "
  echo "              the default pref.dat.
  mv $chmbuild/pref$$.dat $chmbuild/pref.dat
  exit 2
endif
#
if (! -e $chmbuild/pref.dat ) then
  # MF 03/2006
  # modify pref.dat according to flags in addprefdat/delprefdat

  if ( $#addprefdat > 0 ) then
     foreach add ( $addprefdat )
       if ( { ( grep ^${add}'$' $chmbuild/pref$$.dat >& /dev/null ) } ) then
          echo $add already exists in pref.dat
       else 
          sed -e '/^END$/i'"\\
${add}" $chmbuild/pref$$.dat > $chmbuild/pref$$_tmp.dat
          mv $chmbuild/pref$$_tmp.dat $chmbuild/pref$$.dat
       endif
     end
  endif

  if ( $#delprefdat > 0 ) then
     foreach del ( $delprefdat )
       grep -v ^${del}'$' $chmbuild/pref$$.dat > $chmbuild/pref$$_tmp.dat
       mv $chmbuild/pref$$_tmp.dat $chmbuild/pref$$.dat
     end
  endif

  mv $chmbuild/pref$$.dat $chmbuild/pref.dat
else
  echo " install.com> pref.dat already exists and contains"
  echo "              the following preprocessor directives."
  echo " "
  cat  $chmbuild/pref.dat
  echo " "
  echo " install.com> The existing pref.dat will be used in this"
  echo "              installation."
  /bin/rm -f $chmbuild/pref$$.dat
endif

# Modify Makefile for PARCMD in pref.dat
if ( $chmhost == gnu || $chmhost == gpu || $chmhost == em64t ) then
  if ( 1 == `grep PARCMD $chmbuild/pref.dat | wc -l` ) then
    if ( 1 != `grep parcmd $chmbuild/Makefile_$chmhost | wc -l` ) then
      sed -e '/^CC =/s@cc @cc -Dparcmd @' \
        $chmbuild/Makefile_$chmhost > $chmbuild/Makefile_$$
      /bin/mv $chmbuild/Makefile_$$ $chmbuild/Makefile_$chmhost
    endif
  endif
endif

#
# prepare the host platform dependent Makefile
# 08-AUG-92, Always re-generate Makefile, ydw.
sed -e "/^ROOT =/s@rootdir@$chmroot@" $chmbuild/Makefile_$chm_host > Makefile_$$
#
# clbiii mods
# Rework of MPI installation procedures to modify machine specific Makefiles.
# Changes introduced below produce following outcome w/ following input:
#
# install.com machine size M
# produces => MSGLIB = -Lmpi -lmpi
#
# install.com machine size M mpich
# produces => MSGLIB = -Lmpi -lfmpich -lmpich
#
# install.com machine size M lammpi
# produces => MSGLIB = -Lmpi -llamf77mpi -lmpi -llam
#
# install.com machine size M mpispecial
# produces => echo "Enter load line arguments for MPI library linking"
#             echo -n ">> "
#             set linkmpi = "$<"
#	      echo "Adding to load line for linking: -Lmpi "$linkmpi
#	      echo "MSGLIB = -Lmpi "$linkmpi >> sed.tmp_$$
#
# Also, modification introduced to result of query about where include and
# libraries are kept, added linking of *.so as well as *.a to mpi/ directory.
#
#
if ( $mpiset == 1 ) then
    set mpigeneral = 0
#
#  Prepare for Makefile modifications
    mv Makefile_$$ Makefile.tmp_$$
    echo "10a\" > sed.tmp_$$  # "
#
    if ( ! ( $mpich == 1 || $lammpi == 1 || $mpispecial == 1 || $scali == 1 ) ) then
	set mpigeneral = 1
    endif
    if ( $mpif90 == 1 ) set mpigeneral = 0
    if ( $mpigeneral == 1 ) then
	echo "MSGLIB = -Lmpi -lmpi" >> sed.tmp_$$
#
    else
	if ( $mpich == 1 ) then
	    echo "MSGLIB = -Lmpi -lfmpich -lmpich -lpthread" >> sed.tmp_$$
	else if ( $lammpi == 1 ) then
	    echo "MSGLIB = -Lmpi -llamf77mpi -lmpi -llam" >> sed.tmp_$$
	else if ( $scali == 1 ) then
	    echo "MSGLIB = -Lmpi -lfmpi -lmpi" >> sed.tmp_$$
# scasci and scirtl are  no longer needed with Scali MPI Connect (v4)
#	    echo "MSGLIB = -Lmpi -lfmpi -lmpi -lscasci -lscirtl" >> sed.tmp_$$
        else if ( $mpif90 == 1 ) then
            echo "MSGLIB = " >> sed.tmp_$$
	else
	    if ($?MPI_SPECIAL == 0) then
  	       echo "Enter load line arguments for MPI library linking"
	       echo -n ">> "
	       set linkmpi = "$<"
	       setenv MPI_SPECIAL $linkmpi
            else 
	       set linkmpi = $MPI_SPECIAL
	    endif
	    echo "Adding to load line for linking: -Lmpi "$linkmpi
	    echo "MSGLIB = -Lmpi "$linkmpi >> sed.tmp_$$
	endif
    endif
    sed -f sed.tmp_$$ Makefile.tmp_$$ > Makefile_$$
    /bin/rm sed.tmp_$$ Makefile.tmp_$$
endif
#
# PS check??
if( $chm_host == gnu &&  $mpiset == 1  && $scali == 0) then
    mv Makefile_$$ Makefile.tmp_$$
    sed -e '/^FC =/s@g77@mpif90 -Impi@g' \
        -e '/^LD =/s@g77@mpif90 -Impi@g' \
        -e '/^LDD =/s@g77@mpif90 -Impi@g' \
        -e '/^FC =/s@pgcc@mpicc@g' \
        -e '/^LD =/s@pgcc@mpicc@g' \
        -e '/^LDD =/s@pgcc@mpicc@g' \
        -e '/^FC =/s@ fort@ mpif90 -fc=fort -Impi@g' \
        -e '/^LD =/s@ fort@ mpif90 -fc=fort -Impi@g' \
        -e '/^LDD =/s@ fort@ mpif90 -fc=fort -Impi@g' \
        -e '/^FC =/s@ccc@mpicc -cc=ccc@g' \
        -e '/^LD =/s@ccc@mpicc -cc=ccc@g' \
        -e '/^LDD =/s@ccc@mpicc -cc=ccc@g' \
        -e '/^FC =/s@ifc@mpif90 -Impi@g' \
        -e '/^LD =/s@ifc@mpif90 -Impi@g' \
        -e '/^LDD =/s@ifc@mpif90 -Impi@g' \
        -e '/^FC =/s@gfortran@mpif90@g' \
        -e '/^LD =/s@gfortran@mpif90@g' \
        -e '/^FC =/s@ifort@mpif90 -Impi@g' \
        -e '/^LD =/s@ifort@mpif90 -Impi@g' \
        -e '/^LDD =/s@ifort@mpif90 -Impi@g' \
        -e '/^FC =/s@pathf95@mpif90 -Impi@g' \
        -e '/^LD =/s@pathf95@mpif90 -Impi@g' \
        -e '/^LDD =/s@pathf95@mpif90 -Impi@g' \
        -e '/^FC =/s@pgf95@mpif90 -Impi@g' \
        -e '/^LD =/s@pgf95@mpif90 -Impi@g' \
        -e '/^LDD =/s@pgf95@mpif90 -Impi@g' < Makefile.tmp_$$ > Makefile_$$
    if ( $nersc == 1 ) then
	mv Makefile_$$ Makefile.tmp_$$
	#sed -e 's/mpif90/ftn/g' -e 's/mpiifort/ftn/g' -e 's/mpicc/CC/g' -e 's/mpiicc/CC/g' -e 's/g++/CC/g' -e 's/gcc/cc/g' < Makefile.tmp_$$ > Makefile_$$
	#sed -e 's/mpif90/ftn -mprefer-avx128/g' -e 's/mpiifort/ftn -mprefer-avx128/g' -e 's/mpicc/CC -mprefer-avx128/g' -e 's/mpiicc/CC -mprefer-avx128/g' -e 's/g++/CC -mprefer-avx128/g' -e 's/gcc/cc -mprefer-avx128/g' < Makefile.tmp_$$ > Makefile_$$
	sed -e 's/mpif90/ftn -msse3/g' -e 's/mpiifort/ftn -msse3/g' -e 's/mpicc/CC -msse3/g' -e 's/mpiicc/CC -msse3/g'  -e 's/g++/cc -msse3/g' < Makefile.tmp_$$ > Makefile_$$
	#sed -e 's/mpif90/ftn/g' -e 's/mpiifort/ftn/g' -e 's/mpicc/CC/g' -e 's/mpiicc/CC/g' -e 's/icc/CC/g' -e 's/gcc/cc/g' -e 's/g++/CC/g' < Makefile.tmp_$$ > Makefile_$$
	#sed -e 's/mpif90/ftn/g' -e 's/mpicc/gcc/g' < Makefile.tmp_$$ > Makefile_$$
    endif
# PGI parallel MPI workaround to use gcc for domdec_gpu.cu; rmv 26aug2013
    if ( $pgf95 == 1 ) then
        mv Makefile_$$ Makefile.tmp_$$
        sed -e '/CUDAC/s/mpicc/gcc/' < Makefile.tmp_$$ > Makefile_$$
    endif
#
    /bin/rm Makefile.tmp_$$
else if( $chm_host == g95 && ( $mpiset == 1 )) then
    mv Makefile_$$ Makefile.tmp_$$
    sed  -e 's/ g95/ mpif90 -Impi/g'  < Makefile.tmp_$$ > Makefile_$$
    /bin/rm Makefile.tmp_$$
else if( $chm_host == em64t && $mpiset == 1 && $nersc == 1 ) then
    mv Makefile_$$ Makefile.tmp_$$
    sed -e 's/mpif90/ftn/g' -e 's/mpiifort/ftn/g' -e 's/mpicc/CC/g' -e 's/mpiicc/CC/g' -e 's/icc/CC/g' -e 's/gcc/CC/g' < Makefile.tmp_$$ > Makefile_$$
    /bin/rm Makefile.tmp_$$
endif
#
# In case pref.dat is old
set ppvmset = 0

# remove static flag if dynamic linking requested
if ( $dynamic == 1 ) then
   mv Makefile_$$ Makefile.tmp_$$
   sed -e 's/-static//g' Makefile.tmp_$$ > Makefile_$$
   /bin/rm Makefile.tmp_$$
endif

mv Makefile_$$ $chmbuild/Makefile

#
# cleaning and informative message out
echo " "
echo " install.com> Phase 2 completed."
#
#-----------------------------------------------------------------------
#
phase3:
# (3.1) Process source and include files
# (3.2) Compile and build the library for each module
# (3.3) Link to produce the executable.
#
echo " "
echo " install.com> Processing CHARMM source on $chm_host..."
echo " "
#
# make CHARMM
if ($?chmhost == 0) then
  echo " "
  echo " install.com> Environment variable chmhost is not defined."
  exit 1
endif
#
if ( $?chmbuild == 0 ) then
  echo " "
  echo " install.com> Environment variable chmbuild is not defined."
  exit 1
endif
pushd $chmbuild >> /dev/null
#
if ( $?chmtool == 0 ) then
  echo " "
  echo " install.com> Environment variable chmtool is not defined."
  exit 1
endif
#
if ($?chmsrc == 0 ) then
  echo " "
  echo " install.com> Environment variable chmsrc is not defined."
  exit 1
endif
#
# PVM links
if ($pvmset == 1 ) then
   /bin/rm -rf pvm
   mkdir pvm
   echo "Enter the absolute path to the pvm include files (pvm3.h and fpvm3.h)"
   echo -n ">> "
   set incpath = $<
   if (! -e $incpath) then
      echo "Error: can't find directory "$incpath
      exit -1
   endif
   cd pvm
   ln -s $incpath/*.h .
   cd ..
   echo "Enter the absolute path to the pvm lib files (lib?pvm3.a)"
   echo -n ">> "
   set libpath = $<
   if (! -e $libpath) then
      echo "Error: can't find directory "$libpath
      exit -1
   endif
   cd pvm
   ln -s $libpath/*.a .
   cd ..
   #MPI links
else if ( ( $mpif90 == 1 ) ) then
   echo "Using mpif90 for compile"
else if ($mpiset == 1  ) then
    if( $?MPI_HOME ) then
	if ( -d $MPI_HOME/include ) then
	    setenv MPI_INCLUDE $MPI_HOME/include
	else
	    echo "MPI_HOME is set but include dir does not exist"
	endif
	if ( -d $MPI_HOME/lib ) then
	    setenv MPI_LIB $MPI_HOME/lib
	else if ( -d $MPI_HOME/lib64 ) then
	    setenv MPI_LIB $MPI_HOME/lib64
	else
	    echo "MPI_HOME is set but lib dir does not exist"
	endif
    endif
   /bin/rm -rf mpi
   mkdir mpi
   if ($?MPI_INCLUDE == 0) then
      echo "Enter the absolute path to the mpi include files (mpif.h, etc.)"
      echo -n ">> "
      set incpath = $<
      setenv MPI_INCLUDE $incpath
   else
      set incpath = $MPI_INCLUDE
   endif
   setenv MPI_INCLUDE $incpath
   if (! -e $incpath) then
      echo "Error: can't find directory "$incpath
      exit -1
   endif
   cd mpi
   ln -s $incpath/*.h .
   cd ..
   if ($?MPI_LIB == 0) then
      echo "Enter the absolute path to the mpi lib files (e.g., libmpi.a(so))"
      echo -n ">> "
      set libpath = $<
      setenv MPI_LIB $libpath
   else
      set libpath = $MPI_LIB
   endif
   setenv MPI_LIB $libpath
   if (! -e $libpath) then
      echo "Error: can't find directory "$libpath
      exit -1
   endif
   cd mpi
   ln -s $libpath/*.a .
   ln -s $libpath/*.so .
   cd ..
endif
#
#/bin/rm -rf fcm
#ln -s $chmsrc/fcm fcm
if ( $status != 0 ) exit -1
setenv MAKELIM 1
#
# have to create empty archives if they don't exist
if ($?chmlib == 0 ) then
  echo " "
  echo " install.com> Environment variable chmlib is not defined."
  exit 1
endif

# Define a REVID to include in the output header
if (-d $chmroot/.git) then
  set revid = "Git commit ID `git rev-parse --short HEAD`"
else if (-d $chmroot/.svn) then
  set revid = "SVN revision `cd $chmroot; svnversion`"
else
  set revid = ""
endif
pushd $chmbuild >> /dev/null
echo "  character(len=*), parameter :: REVID = '$revid'" > revid.tmp
if (-e revid.h) then
  cmp -s revid.h revid.tmp
  if ($status == 0) then
    rm revid.tmp
  else
    mv revid.tmp revid.h
    if (-e iniall.o) rm iniall.o
  endif
else
  mv revid.tmp revid.h
endif
popd >> /dev/null

pushd $chmsrc >> /dev/null
  set mdlist = `/bin/ls -1 | sed '/fcm/d' | sed '/charmm/d'`
popd >> /dev/null
#
if (! -e $chm_host.log) then
   echo "CHARMM Build on $chm_host" `date` > $chm_host.log
else
   echo "CHARMM Build on $chm_host" `date` >> $chm_host.log
endif
#
foreach module ( $mdlist )
  if (! -e $chmlib/$module.a ) $AR_COMMAND $chmlib/$module.a
end
if (! -e $chmlib/stamp)  touch $chmlib/stamp
#
#
if ( 1 == `grep GAMESSUK pref.dat | wc -l` ) then
  set gamess = '-f guk.mk'

  #
  #  Select the correct version of the interface code
  #  to be used in GAMESS-UK build
  #
  setenv CHMGUK_VERSION 5

  # these flags are used help GAMESS-UK find the correct include files
  # and will also influence whether the -fno-second-underscore flag is used 
  # when GAMESS-UK is built  

  if ( $mpich == 1 ) then
    setenv MPIVERSION mpich
  else if ( $lammpi == 1 ) then
    setenv MPIVERSION lam
  else if ( $scali == 1 ) then
    setenv MPIVERSION scali
  else
    setenv MPIVERSION ""
  endif
 
  if ( $chm_host == gnu ) then
    if ( $f77 == 1 ) then
      setenv F77 absoft
    else if ( $f2c == 1 ) then
      # this will need work
      setenv F77 f2c
    else if ( $g77 == 1 ) then
      setenv F77 g77
    else if ( $ifc == 1 ) then
      setenv F77 ifc
    else if ( $efc == 1 ) then
      setenv F77 efc
    else if ( $pgf95 == 1 ) then
      setenv F77 pgf95
    else if ( $g95 == 1 ) then
      setenv F77 g95
    else if ( $gfortran == 1 ) then
      setenv F77 gfortran
    else if ( $cfort == 1 ) then
      # this will need work
      setenv F77 fort
    else
      setenv F77 gfortran
    endif
  endif

else if ( 1 == `grep GAMESS pref.dat | wc -l` ) then
 set gamess = '-f gamess.mk'
 if (! -e $chmtool/actvte.x_$chm_host ) then
   if ( $chm_host == gnu ) then
    $forcom -o $chmtool/actvte.x_$chm_host $chmtool/actvte.f
   else
    f77 -o $chmtool/actvte.x_$chm_host $chmtool/actvte.f
   endif
 endif
else
 set gamess = ''
endif
#
#
if ( 1 == `grep CADPAC pref.dat | wc -l` ) then
 set cadpac = '-f cadpac.mk'
 if ( $cfort == 1 && $chmhost == gnu ) then
  if ( 0 == `grep DGNUALPHA $chmbuild/cadpac.mk | wc -l` ) then
   mv -f $chmbuild/cadpac.mk $chmbuild/cadpac.mk.$$
   sed -e 's/DEFS = /DEFS = -DGNUALPHA /' \
      $chmbuild/cadpac.mk.$$ > $chmbuild/cadpac.mk
   /bin/rm -f $chmbuild/cadpac.mk.$$
  endif
 else if ( $cfort == 0 ) then
  mv -f $chmbuild/cadpac.mk $chmbuild/cadpac.mk.$$
  sed -e 's/-DGNUALPHA //g' \
      $chmbuild/cadpac.mk.$$ > $chmbuild/cadpac.mk
   /bin/rm -f $chmbuild/cadpac.mk.$$
 endif
else
 set cadpac = ''
endif
#

 set sccdftb = '-f sccdftb.mk -f sccdftbint.mk'

#if ( 1 == `grep SCCDFTB pref.dat | wc -l` ) then
# Also process the include file
# echo "Preparing the include file for SCCDFTB ...... "
# $chmtool/ratfor.$chm_host < $chmsrc/sccdftbint/sccdftbsrc/maxima.h > $chmbuild/maxima.inc
# cp $chmtool/ratfor.$chm_host $chmbuild/ratfor
#mfc else
#mfc  set sccdftb = ''
#endif

#
if ( 1 == `grep SQUANTM pref.dat | wc -l` ) then
 set squantm = '-f squantm.mk'
else
 set squantm = ''
endif

if ( 1 == `grep QMMMSEMI pref.dat | wc -l` ) then
 set qmmmsemi = '-f qmmmsemi.mk'
else
 set qmmmsemi = ''
endif

# PJ 06/2005
#mfc pipf code is protected with ##IF now, do not need to exclude the makefile
#mfc if (1 == `grep PIPF pref.dat |wc -l`) then
   set pipf = '-f pipf.mk'
#mfc else
#mfc    set pipf = ''
#mfc endif

# JWB : added line below after commenting out string method code
set stringm = '-f stringm.mk'

#=====================================================================
# VO : string method v
#
# JWB : commented out to improve speed of install.com on compiles subsequent
# to an initial compile. On subsequent compiles, the code below may result in
# a substantial pause before actual compilation begins.
#
# if ( $qstringm == 1 ) then
#  set stringm = '-f stringm.mk'
# else
#  set stringm = ''
#
# now, remove all string files from dependency info in *mk files 
# this needs to be done due to the behavior of mkmod.pl, which generates
# dependency info ignoring the configuration in pref.dat  (this could
# be fixed by running mkmod.pl on preprocessed source);
# also, remove stringm.mk
#
#  rm -f $chmbuild/stringm.mk
#  set stringm_files=(`ls $chmsrc/stringm | grep ".src"`)
#  set mk_files=(`ls $chmbuild | grep ".mk"`)
#  foreach mk_file ($mk_files)
#   set sedcmd = ()
#   foreach stringm_file ($stringm_files)
#    set file_o=(`basename $stringm_file .src`"\.o")
#    set sedcmd = ($sedcmd"s/$file_o//;")
#   end
#   sed -ie "$sedcmd" $chmbuild/$mk_file
#  end
# remove stringm from objlibs
#  sed -ie '/stringm/d' $chmbuild/objlibs.mk
# remove stringm.a lib since it will be empty
#  rm -f $chmlib/stringm.a
endif
# VO : string method ^
#=====================================================================
#
#if ( 1 == `grep NOGRAPHICS pref.dat | wc -l` ) then
# sed -e '/graphics.a/d' objlibs.mk > objlibs.mk.tmp$$
# mv objlibs.mk.tmp$$ objlibs.mk
# set graphics = ' '
#else
 set graphics = '-f graphics.mk'
#endif

# Setup flucq and cadpac makefiles for FlucQ (if enabled)
if ( 1 == `grep FLUCQ pref.dat | wc -l` ) then
 set flucq = '-f flucq.mk'
 awk '/cadint/ { print "	$(LIB)/flucq.a \\" } \
      /./      {print $0}' \
                objlibs.mk > objlibs.mk.tmp$$
 if ( 1 != `grep flucq objlibs.mk | wc -l` ) mv objlibs.mk.tmp$$ objlibs.mk
 if ( 1 == `grep CADPAC pref.dat | wc -l` ) then
  if ( 0 == `grep DFLUCQ $chmbuild/cadpac.mk | wc -l` ) then
   mv -f $chmbuild/cadpac.mk $chmbuild/cadpac.mk.$$
   sed -e 's/DEFS = /DEFS = -DFLUCQ /' \
      $chmbuild/cadpac.mk.$$ > $chmbuild/cadpac.mk
   /bin/rm -f $chmbuild/cadpac.mk.$$
  endif
 endif
else
 set flucq = ''
 if ( 1 == `grep CADPAC pref.dat | wc -l` ) then
  mv -f $chmbuild/cadpac.mk $chmbuild/cadpac.mk.$$
  sed -e 's/-DFLUCQ //g' \
      $chmbuild/cadpac.mk.$$ > $chmbuild/cadpac.mk
  /bin/rm -f $chmbuild/cadpac.mk.$$
 endif
endif

# Setup EMAP processing
# unconditional, makemod doesn't recognize conditions on ##USE
 set emapmk = '-f emap.mk'
 awk '/dynamc/ { print "	$(LIB)/emap.a \\" } \
      /./      {print $0}' \
                objlibs.mk > objlibs.mk.tmp$$
 if ( 1 != `grep emap objlibs.mk | wc -l` ) mv objlibs.mk.tmp$$ objlibs.mk

# Check for socket compile - some machines don't support this
# include only when really needed
if ( 1 == `grep SOCKET pref.dat | wc -l` ) then
     sed -e 's@cc @cc -Dchmsocket @' \
      Makefile > Makefile.tmp$$
 mv Makefile.tmp$$  Makefile
endif

#
# POLYRATE installation control
#
#  The following environment varibles should have benn set (see documentation)
#
#       pr     = POLYRATE programm directory
#       crate  = CRATE utility directory
#
#  The script file install_cr.com and the source files of polyrate 
#  are provided by UofM
#
if ( $polyrate == 1 ) then
   echo " install_cr.com >  run the CRATE utility"
   echo "                   to copy the source code files of POLYRATE"
   if (! $?crate) setenv crate $chmsrc/prate/crate
   if (! $?pr)    setenv pr    $chmsrc/prate/polyrate
   $crate/install_cr.com
   echo " install.com> copy polyrate common blocks into the build directory"
   /bin/cp $chmsrc/prate/param.inc $chmbuild/.
   /bin/cp $chmsrc/prate/percon.inc $chmbuild/.
   /bin/cp $chmsrc/prate/common.inc $chmbuild/.
   /bin/cp $chmsrc/prate/charmmrate.inc $chmbuild/.
# modify Makefile to include prate.a in the LIB list
   awk '/rxncor/ { print "	$(LIB)/prate.a \\" } \
        /./      {print $0}' \
                    $chmbuild/objlibs.mk > $chmbuild/objlibs.mk_$$
   if ( 1 != `grep prate $chmbuild/objlibs.mk|wc -l` ) \
              /bin/mv $chmbuild/objlibs.mk_$$ $chmbuild/objlibs.mk
endif
#

# prepare CUDA stuff for GPU
if ( $chm_host == gpu ) then
    pushd ${chmroot}/tool/gpu >& /dev/null
    if ( $?nolog ) then
        ./md3compile
        set rc = $status
    else
        ./md3compile >& ${chmbuild}/gpu.log
        set rc = $status
        echo "Finished compiling CHARMM/CUDA interface library."
        echo "More info in" ${chmbuild}/gpu.log
    endif
    popd >& /dev/null
    # not good anymore ....
    if ($rc != 0) then
        echo
        echo "There is some problem compiling/running CUDA code for CHARMM"
        echo "make sure you can run CUDA programs and have proper CUDA environment"
        echo "see" ${chmbuild}/gpu.log "for details"
        echo
        exit $rc
    endif
endif


if ( ! $?MAKE_COMMAND ) then
  setenv MAKE_COMMAND make
endif

# copy the colfft_*.h include files to the build directory
# so we do not have to add a custom -I for
# colfft_(func|kernel|util).src
foreach f ($chmsrc/nbonds/*.inc)
  set new_inc = $f:t
  /bin/cp "$f" "$chmbuild/$new_inc"
end

foreach f ($chmsrc/pipf/*.inc)
  set new_inc = $f:t
  /bin/cp "$f" "$chmbuild/$new_inc"
end

/bin/cp "$chmsrc/manip/fsshake_kernel.inc" "$chmbuild"
/bin/cp "$chmsrc/energy/ediff.inc" "$chmbuild"
/bin/cp "$chmsrc/misc/fctblock.inc" "$chmbuild"
/bin/cp "$chmsrc/misc/fctnba.inc" "$chmbuild"
/bin/cp "$chmsrc/misc/fctnma.inc" "$chmbuild"

foreach f ($chmsrc/cff/*.inc)
  set new_inc = $f:t
  /bin/cp "$f" "$chmbuild/$new_inc"
end

foreach f ($chmsrc/image/*.inc)
  set new_inc = $f:t
  /bin/cp "$f" "$chmbuild/$new_inc"
end

if ($domdec == 1) then
  foreach f ($chmsrc/domdec/*.inc)
    set new_domdec_include = $f:t
    /bin/cp "$f" "$chmbuild/$new_domdec_include"
  end
endif

if ($openmm == 1) then # Make Charmms custom OMM plugins
    if ($?nolog) then
        $MAKE_COMMAND -f $chmbuild/Makefile_omm all
        set rc = $status
    else
        date > new.log
        $MAKE_COMMAND -f Makefile_omm all >>& new.log
        set rc = $status
    endif
    if ($rc != 0) then
        echo " install.com> ERROR: failed to build OpenMM CHARMM plugins"
        if ( ! $?nolog ) then
            echo "              Check $chmbuild/$chm_host.log"
            echo "              for installation errors."
            date >> new.log
            cat new.log >> $chm_host.log
        endif
        exit $rc
    else
        echo " install.com> done building OpenMM CHARMM plugins"
        if ( ! $?nolog ) then
            date >> new.log
            cat new.log >> $chm_host.log
        endif
        setenv MACOSX_DEPLOYMENT_TARGET 10.6
    endif
endif

if ( ! -f "$chmbuild/keywords.inc" ) then
  set nwords = `wc -l "$chmbuild/pref.dat" | awk '{print $1}'`
  set nwords = ($nwords - 1)
  echo "integer, parameter :: num_pref_keys = $nwords" \
    >> "$chmbuild/keywords.inc"
  echo "character(len = 12), dimension(num_pref_keys), parameter :: &" \
    >> "$chmbuild/keywords.inc"
  echo "    pref_keys = [ character(len = 12) :: &" \
    >> "$chmbuild/keywords.inc"
  awk 'NF { if ($1 != "END") print "      \"" $1 "\", &" }' \
    "$chmbuild/pref.dat" \
    | sed -e '$s/...$/]/' \
    >> "$chmbuild/keywords.inc"
endif

setenv FCDEFINES `awk '{printf("-DKEY_%s=1 ",$1);}' < pref.dat | sed 's/ -DKEY_END=1//'`

set go_make = "$MAKE_COMMAND -f Makefile -f charmm.mk -f adumb.mk"
set go_make = "$go_make -f api.mk -f blade_api.mk"
set go_make = "$go_make -f cadint.mk -f cff.mk -f correl.mk -f csa.mk"
set go_make = "$go_make -f dimb.mk -f dynamc.mk -f domdec.mk -f domdec_gpu.mk"
set go_make = "$go_make -f energy.mk -f fftdock.mk"
set go_make = "$go_make -f gamint.mk -f gamus.mk -f gukint.mk"
set go_make = "$go_make -f gener.mk -f image.mk -f io.mk -f ltm.mk"
set go_make = "$go_make -f machdep.mk -f ensemble.mk"
set go_make = "$go_make -f mc.mk -f mmff.mk -f manip.mk -f minmiz.mk"
set go_make = "$go_make -f misc.mk -f mndint.mk -f molvib.mk -f mscale.mk"
set go_make = "$go_make -f nbonds.mk -f pert.mk -f quantum.mk -f rxncor.mk"
set go_make = "$go_make -f shapes.mk -f solvation.mk -f util.mk -f memory.mk"
set go_make = "$go_make -f vibran.mk -f zerom.mk -f qmmmsemi.mk -f flucq.mk"
set go_make = "$go_make -f opencl.mk -f openmm.mk"
set go_make = "$go_make $cadpac $emapmk $gamess $graphics"
set go_make = "$go_make $qmmmsemi -f squantm.mk $pipf -f prate.mk $sccdftb $stringm"
if ( $?nolog ) then
   $go_make 
   set rc = $status
else
   date > new.log
   $go_make >>& new.log
   set rc = $status
   date >> new.log
   cat new.log >> $chm_host.log
endif
#
#
if ( -x charmm.exe ) then
  mv charmm.exe $chmexec/charmm
  if ( $DEBUG && $chm_host == osx ) dsymutil $chmexec/charmm
  echo " "
  echo " install.com> CHARMM Installation is completed."
  echo "              The CHARMM executable is $chmexec/charmm."
else if ( $rc != 0 ) then
  echo " "
  echo " install.com> The CHARMM executable"
  echo "              $chmexec/charmm is NOT produced."
  echo "              Check $chmbuild/$chm_host.log"
  echo "              for installation errors."
endif
#
popd >> /dev/null
echo " "
echo " install.com> Phase 3 completed."
#
# clean up
#

echo " install.com> WARNING: install.com is deprecated"
echo " install.com> and will be removed from future releases."
echo " install.com> Please use ./configure and CMake."

date 
exit $rc
