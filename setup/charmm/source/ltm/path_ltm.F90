module pathm
  use chm_kinds
  use dimens_fcm

  implicit none
  character(len=*),private,parameter :: file_name   ="path_ltm.src"

#if KEY_REPLICA==1 && KEY_RPATH==1 /*path_fcm*/
!
! This common includes constants for the REPLICA/PATH method.
!                       BRB - 3/25/94
!
!-----------------------------------------------------------------------
! Generic replica/path variables.
!
!     QPATH  - Is replica path code enabled?
!     QPMASS - Are the atoms mass weighted?
!     QPWEIG - Is the main weighting array used?
!     QPWCOM - Is the comp weighting array used?
!     QPWCM2 - Is the comp2 weighting array used?
!     KRMS   - The rms deviation (from average) force constant.
!     KMNRMS - The rms force constant for minimum vector exceeded.
!     RMNRMS - The rms distance (A) for minimum vector exceeded.
!     KMXRMS - The rms force constant for maximum vector exceeded.
!     RMXRMS - The rms distance (A) for maximum vector exceeded.
!     KANG   - The angle deviation force constant (Kcal/mol)
!     PCOSMX - The value of COS(theta) below
!              which the vectors are restrained.
!     EVWID  - Eigenvalue separation allowed before using a switching
!              function in FROTU (rms best-fit). Units of Angstrom**2
!
!     QPNORT - Do not use rotation best fit for adjacent replicas?
!     QPNOTR - Do not use translation best fit for adjacent replicas?
!     QPCYCL - Use the cyclical method (full circle)
!
!     NATREP - number of atoms in the key replica
!     PWTOT  - total weight of path
!
!     NTREP  - In the parallel REPLICA/PATH this is actual number of
!              replicates. Local number is obtaines from replica.fcm
!     MYNODRP- MYNOD for REPLICA/PATH method
!-----------------------------------------------------------------------
! Cartesian NEB path variables       plm - 7/25/2001  
!     QNEB   - Cartesian NEB flag
!     KNEB   - The NEB force constant 
!-----------------------------------------------------------------------
!     QPPMF  - primt PMF of each replica for each rpath call
! Bestfit NEB path variables
!   Nudged Elastic Band (NEB) Flags JWCHU 6/13/02
!     QPNEB  - Use the NEB method
!     QPETAN - Use the Energy Based Tangent estimation
!     QPCIMG - Use the climbing image method with NEB
!   PATH STATISTICS Flags     ! jwchu
!     QRFIX  - if forces are projected to a reference path
!              default from main set, use comp set if COMP key work is 
!              present
!     REFPX  - reference path array x
!     REFPY  - reference path array y
!     REFPZ  - reference path array x
!     REPEi  - energy of replica i 
!     REPESQi- energy square of replica i 
!     REPFi  - PMF of replica i 
!     REPFSQi- PMF square of replica i 
!     REPELAi- energy of replica i of the latest step 
!     REPFLAi- PMF of replica i of the latest step 
!     REPLENG- step length in real space
!     NPCALL - number of calls of epath, is reset to zero if rpath is called
!     QANAL  - print out statistics and release space
!     QPA_alloc- if the space for the 
!              reference coordinate is to be generated or not.
!              It is .TRUE. for the first rpath call and then turned
!              off after the space is generated.
!     QPROPT - put restraint to keep the image has the same rms distance
!              between neighboring images
!     QMXRMS - add more restraint f the image has the moved too far 
!     DRATIO - ratio for optimizing a replica reference to ra and rb
!              ra/rb=(1-DRATIO)/DRATIO
!     QPRPMF - print RPMF of each replica for each rpath call
!-----------------------------------------------------------------------
! Off-path restraints variables: H. Lee Woodcock
!     QPROPT - Optimize Pathway with EPATHO SUBROUTINE
!     QPATHFLG - Flag to tell when to calculate pmf
!     QPANAL  - Determines if Extended Analysis will be 
!     QCURVC  - Curvature correction for off-path simulations
!     QNOCURVC - Do not do any curvature correction for off-path simul.
!     XPREF  - X coords for reference path
!     YPREF  - Y coords for reference path
!     ZPREF  - Z coords for reference path
!     PMF    - Stores the pmf during off-path simulations
!     FLUC   - Stores the flucuations during off-path simulations
!     XPDIR  - Direction of the X projection
!     YPDIR  - Direction of the Y projection
!     ZPDIR  - Direction of the Z projection
!     PATHWORK- Work computed to go from step to step on rpath
!     XPROJV - Force projection of the current replica 
!     BFREFJI - RMS best-fit (J(ref) -> I(ref))
!     BFREFJK - RMS best-fit (J(ref) -> K(ref))
!     BFREFIK - RMS best-fit (I(ref) -> K(ref))
!     MAXREP  - Maximum number of replicas
!-----------------------------------------------------------------------
! Generic replica/path variables.
! Added generic off-path simulation variables. 
      INTEGER MAXREP
      PARAMETER(MAXREP=1000)
      LOGICAL QPATH,QPMASS,QPWEIG,QPWCOM,QPWCM2,QPNORT,QPNOTR,QPCYCL
      INTEGER NATREP,NTREP,MYNODRP
      real(chm_real) KRMS,KMNRMS,RMNRMS,KMXRMS,RMXRMS,KANG,PCOSMX, &
           EVWID,PWTOT
      real(chm_real),allocatable,dimension(:) :: XPREF,YPREF,ZPREF,PMF, &
           FLUC,XPDIR,YPDIR,ZPDIR,PATHWORK
      real(chm_real) XPROJV,BFREFJI(maxrep),BFREFJK(maxrep),BFREFIK(maxrep)
!
!-----------------------------------------------------------------------
! Cartesian NEB path variables
      LOGICAL QNEB
      real(chm_real) KNEB
!-----------------------------------------------------------------------
! Bestfit NEB path variables
      LOGICAL QPNEB,QPETAN,QPCIMG,QRFIX,QANAL,QPA_alloc ! jwchu
      LOGICAL QPPMF,QPRPMF,QPTAU  ! jwchu
      INTEGER NPCALL
      real(chm_real) DRATIO
!-----------------------------------------------------------------------
! Off-path logical restraint variables
      LOGICAL QPROPT,QPROPT1,QPATHFLG,QCURVC,QNOCURVC
!-----------------------------------------------------------------------
! for kinetic energy and constant length potential
      LOGICAL QPKINE,QPLENG,QPDYNA,QISOKN,QWETHM,QPTEMP,QPKNUDG
      real(chm_real) KPKINE,KPLENG,PLFIX,PTEMP
      INTEGER IPSEED
!-----------------------------------------------------------------------
! for hyperplane restraint
      LOGICAL QPHP
      real(chm_real) KPHP,RPHP

!
contains
  subroutine path_iniall()  !
    qpath=.false.
    qpathflg=.true.
    qpneb=.false.
    qpa_alloc=.true.  
    npcall=0
    return
  end subroutine path_iniall


  subroutine allocate_path()
    use memory
    character(len=*),parameter :: routine_name="allocate_pathm"
    call chmalloc(file_name,routine_name,'xpref   ',maxa  ,crl=xpref)
    call chmalloc(file_name,routine_name,'ypref   ',maxa  ,crl=ypref)
    call chmalloc(file_name,routine_name,'zpref   ',maxa  ,crl=zpref)
    call chmalloc(file_name,routine_name,'pmf     ',maxrep,crl=pmf)
    call chmalloc(file_name,routine_name,'fluc    ',maxrep,crl=fluc)
    call chmalloc(file_name,routine_name,'xpdir   ',maxa  ,crl=xpdir)
    call chmalloc(file_name,routine_name,'ypdir   ',maxa  ,crl=ypdir)
    call chmalloc(file_name,routine_name,'zpdir   ',maxa  ,crl=zpdir)
    call chmalloc(file_name,routine_name,'pathwork',maxrep,crl=pathwork)

    return
  end subroutine allocate_path
#endif /*   (path_fcm)*/

end module pathm

