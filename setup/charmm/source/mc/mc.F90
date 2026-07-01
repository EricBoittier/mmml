module mcc
#if KEY_MC==1
  use chm_types
  use mc, only: mmvtyp
  implicit none

  type(iptr_ptr),save :: IMCGRP(MMVTYP), NOECNP(MMVTYP)
  type(chm_iptr),save :: NTRYP(MMVTYP), MCACCP(MMVTYP)
  type(chm_ptr),save :: EAVEP(MMVTYP), DAVEP(MMVTYP)
  type(chm_ptr),save :: IGAMMA(MMVTYP)
#if KEY_GCMC==1
  type(chm_iptr),save :: IDXGCP(MMVTYP)
  type(chm_iptr),save :: ISCVTY(MMVTYP), NOCVTY(MMVTYP), NGCPCP(MMVTYP)
#endif 

  type(chm_iptr),save :: ALLMVP, ALLGRP
  integer,allocatable,dimension(:) :: IFRATP, IMVGRP, IA2GP 
  integer,allocatable,dimension(:) :: ISKIP, IMVATP
  integer :: NFREAT, NSKIP
  
  !cumulating array for heuristic updates
  real(chm_real),allocatable,dimension(:) :: CUMULHEUR

contains

  SUBROUTINE MCCALL(COMLYN,COMLEN)
    !
    !       Calling routine for Monte Carlo dynamics.
    !
    !       All papers using this routine should cite J. Hu, A. Ma and A. R.
    !       Dinner, Monte Carlo simulations of biomolecules:  The MC module
    !       in CHARMM, J. Comp. Chem. 27: 203-216, 2006.
    !2i
    !       -----------------------------------------------------------------
    !       IMPORTANT NOTE FOR DEVELOPERS:
    !
    !       To add a new type of MC move, it is necessary to add to MKMOVE (in
    !       source/mc/mc.src) a call to a subroutine that applies the move and
    !       to MOVEAD (in source/mc/movead.src) a call to a subroutine that sets
    !       up the IPIVTP, IMVNGP, IBLSTP, MDXP, and QBND arrays.  These arrays
    !       are described in source/fcm/mc.f90.  If these arrays are constructed
    !       correctly, modification of MCCALL, MCLOOP, or MCENER is NOT required
    !       in most circumstances (i.e., those in which application of the move
    !       and evaluation of the change in energy are separable).
    !       -----------------------------------------------------------------
    !
    !       MC Revision History:
    !
    !       Florent Hedin & Markus Meuwly, University of Basel, Switzerland,
    !                      florent.hedin@unibas.ch
    !       October, 2011 - January, 2014
    !         1) Added support for the spatial averaging algorithm (SA-MC) (see details on samc.src)
    !
    !       Aaron R. Dinner, The University of Chicago
    !                        University of California, Berkeley
    !                        University of Oxford
    !                        Harvard University
    !       July, 2004 - June, 2006
    !         2) Added PERT support with Y. Deng and B. Roux
    !         1) Added Wang-Landau sampling (preferential to old multicanonical)
    !       July, 2002 - June, 2004
    !         2) Added GCMC with H.-J. Woo and B. Roux
    !         1) Added Momentum-Enhanced HMC (MEHMC)
    !       July, 2001 - June, 2002
    !         2) Algorithm for generating unit vectors changed
    !         1) Updated ACE1 and ACE2 supported
    !       January, 2001 - June, 2001
    !         6) Path integral energy term supported
    !         5) Tsallis HMC treated correctly
    !         4) Implemented first version of move linking
    !         3) Added writing and reading of dynamics restart files
    !         2) Constant pressure methods introduced (original images or xtal)
    !         1) QM/MM (MOPAC) energy term supported
    !       June, 2000 - December, 2000
    !         3) Group based energy calculations scale better with system size
    !         2) Rigid body translations and rotations of heavy atoms
    !         1) Hybrid Monte Carlo moves added
    !       January, 2000 - June, 2000
    !         4) Heuristic non-bond update
    !         3) Conjugate gradient minimization allowed
    !         2) Added support for ACE energy term
    !         1) Non-random move selection with PICK (not recommended)
    !       July, 1998 - December, 1999
    !         5) Added move group specific temperature scaling factors
    !         4) Added minimization (SD) before applying the acceptance criterion
    !         3) Added Tsallis (generalized) acceptance criterion choice
    !         2) Added NOE constraints
    !         1) Added rigid rotations around center-of-mass
    !       May, 1998 - June, 1998
    !         5) Sped up all atom-based calculations by limiting referencing
    !            of INBLO and IMBLO in non-bonded calculations
    !         4) Suppressed group centering to speed up group-based calculations;
    !            added MC keyword to pref.dat
    !         3) Added Urey-Bradley calculations
    !         2) Fixed image list update problems
    !         1) Compiled in c26a2
    !       March, 1997 - April, 1998
    !         8) Added multicanonical acceptance criterion choice
    !         7) Added user energy calculations
    !            Tested the user call on Lazaridis solvent term
    !         6) Added group based energy calculations
    !         5) Added concerted dihedral rotations.
    !         4) Added move size automatic optimization.
    !         3) Added CHARMM image facilities (some list update problems still)
    !         2) Sped up non-bonded list generation in MCENER
    !         1) Compiled in c25a3
    !       January, 1997
    !         1) Removal of assigned GOTO
    !       January, 1996 - June, 1996
    !         2) Routine completely overhauled --- unless otherwise specified,
    !            routines are by ARD
    !         1) Removal of tabs, addition of VCLOSE, compiled in c24b1
    !
    !       Sung-Sau So, Harvard University
    !       August, 1993
    !         1) Conformation to implicit none standard
    !         2) Implementation of image condition
    !         3) Quanta compatible trajectory handle
    !
    !       Jiali Gao, Harvard University
    !       May, 1991
    !         1) Rigid translations and rotations of pure simple organic liquids
    !
    use chm_kinds
    use chm_types
    use dimens_fcm
    use coord
    use ctitla
    use mc
    use mccent
    use mcio
    use gcmc
    use parallel
    use repdstrmod
    use repdstr
    use psf
    use memory
    use number, only:zero
#if KEY_SAMC==1
    use samc
#endif

    implicit none
    !
    !       Passed Variables
    !
    CHARACTER(len=*) COMLYN
    INTEGER COMLEN
    !
    !       Local Variables
    !
    !       Based on the Command Line
    !         NSTEPS    Maximum number of conformations generated
    !         IACCPF    The acceptance criterion
    !                   0 = Metropolis (default)
    !                   1 = Multicanonical (weights read from IUMULT)
    !                   2 = Tsallis
    !         ISDMC     Random number generator seed
    !         TKELV     Temperature in Kelvin
    !         PRESMC    Pressure in AKMA
    !         ACECUT    Cutoff for ACE self energy to limit update of
    !                   non-bonded pairs to only those with significantly
    !                   changing Born radii.
    !         PICK      Flag for method of selecting moves from the move set
    !                   0 = random (default)
    !                   1 = try each move instance sequentially
    !                   2 = random move group but sequential move
    !                       instances within that group
    !         INBFMC    Frequency of updating the non-bonded list
    !         IMGFMC    Frequency of updating the image list
    !         IEFRQ     Frequency of checking the energy
    !         IARMF     Frequency of updating the move limits with ARM
    !         IDOMCF    Frequency of updating the move limits with DOMC
    !         NSAVC     Save coordinates every NSAVC steps (to IUNCRD)
    !         ISVFRQ    Frequency of saving a structure for restart
    !         IUNWRI    IO unit for restart writing
    !         IUNREA    IO unit for restart reading
    !
    !       Acceptance function parameters for non-canonical sampling
    !         NMULT     Number of bins in multicanonical
    !         MLTLGP    List of weights of bins in multicanonical
    !         ACPAR1    Minimum energy in multicanonical and Tsallis
    !         ACPAR2    Maximum allowed energy in multicanonical
    !                   (1 - q) in Tsallis
    !         ACPAR3    Separation of energy of bins in multicanonical
    !
    !         IUNWLR    Unit number for reading Wang-Landau weights
    !         IUNWLW    Unit number for writing Wang-Landau weights
    !         NWLFRQ    Frequency of checking Wang-Landau convergence
    !         RWLADD    Increment to add for Wang-Landau
    !         WLTOL     Stopping criterion for Wang-Landau
    !         FLTNES    Update criterion for Wang-Landau
    !
    !       For keeping track of the acceptance statistics
    !         NMVTOT    Total number of attempts of a move type
    !         NMVACC    Total number of accepts  of a move type
    !         WTOT      Total of the weight factors for the different move types
    !         WTMP      Array carrying the weights of only the active moves
    !
    !       For the ARM optimization
    !         NTRYP     Number of attempts for each move instance
    !         MCACCP    Number of accepts  for each move instance
    !
    !       For the DOMC optimization
    !         EAVEP     Average energy change for each move instance
    !         DAVEP     Average coordinate change for each move instance
    !
    !       For trajectory writes
    !         NFREAT    Number of free atoms
    !         IFRATP    List of free atoms
    !
    !       Lookup arrays when group based calculations are required
    !         IMCGRP    IMVGP based on groups rather than atoms
    !         IA2GP     Atom to group index look up table for primary atoms
    !
    !       Lookup arrays when ACE calculations are required
    !         MCA14P    Symmetric bonded list to get exclusions right
    !
    !       Lookup arrays for constraints
    !         NOECNP    List of which NOE constraints change for each instance
    !
    !       Lookup arrays for minimization
    !         ALLMVP    List of all free atoms  in the IMVNGP format
    !         ALLGRP    List of all free groups in the IMVNGP format
    !         IGAMMA    The DYNAMC subroutine GAMMA arrays
    !
    !       Scratch arrays for the energy calculations
    !         ISKIP     Bonded term skip list
    !         IMVATP    Moving atoms
    !         IMVGRP    Moving groups
    !
    !       The following variables are passed but not used in the current
    !       implementation
    !         IOAVES    IO unit for printing averages calculated on the fly
    !         IOENGY    IO unit for printing energy
    !
    !       The follwing variables are for all grand canonical simulations
    !         IDXGCP    Non-ordered list of move instance indices that are
    !                   active (i.e., GCMCON(IDXGCP(I)) = .TRUE.).
    !         NIDXGC    Number of active move instances (i.e., length of IDXGCP
    !                   list but not physical size of array).
    !         NGCIN     Number of active move instances in the insertion region
    !         GCMCBF    The factor B = muex/kT - ln<density*volume>.
    !                   See Im, Seefeld, and Roux, Biophys. J. 79, 788 (2000).
    !         XGCMIN    Minimum x for insertion.
    !         YGCMIN    Minimum y for insertion.
    !         ZGCMIN    Minimum z for insertion.
    !         XGCMAX    Maximum x for insertion.
    !         YGCMAX    Maximum y for insertion.
    !         ZGCMAX    Maximum z for insertion.
    !         QGCGRD    Flag which is .TRUE. if the insertion volume is spherical.
    !         NPSUM     Average current number of GC molecules
    !         GCSUM     Running average number of GC molecules
    !         GC2SUM    Running standard deviation of number of GC molecules
    !
    !       The follwing variables are for cavity biased GC simulations
    !         GCCUT2    Square of the cavity radius (based on atom centers).
    !         NGCTRY    Number of points to test for cavity bias insertions.
    !         ISCVTY    Number of cavities of radius GCCUT found.
    !                   This is P_c^N in Mezei (1980) Mol. Phys. 40, 901-906.
    !         NGCPCP    Normalization for ISCVTY.
    !         NGRDBK    Number of blocking groups.
    !         GRDBLK    List   of blocking groups.
    !         NCFB      Number of orientations to attempt
    !
    !       The follwing variables are for grid-based GC simulations
    !         QGCGRD    Flag which is .TRUE. for grid-based insertion.
    !         GRDSIZ    Grid spacing.
    !         NRGRID    Diameter of exclusion in lattice units.
    !         NOCVTY    Number of configurations without cavities.
    !         NGRDCV    Number of cavities on the grid.
    !         NGCBLK    Number of particles blocking a grid site.
    !         LSTGRD    List of grid coordinates where there is a cavity.
    !         LSTIND    List of positions in LSTGRD indexed by grid site
    !         NGRDX(YZ) Dimensions of the insertion grid in lattice units.
    !         NGDTOT    Number of accessible grid points.
    !
    !       The follwing variables are for orientation-biased GC simulations
    !         NIGCMC    Number of atoms in a GCMC molecule
    !
    INTEGER IACCPF,NSTEPS,NSAVC,INBFMC,IEFRQ
    INTEGER ISVFRQ,IOAVES,IUNCRD,IOENGY,IUNWRI,IUNREA
    INTEGER IARMF
    INTEGER IDOMCF
    INTEGER IMGFMC
    INTEGER I, NMVTOT(MMVTYP), NMVACC(MMVTYP), PICK
    type(chm_ptr) :: MLTLGP
    INTEGER NMULT, IUMULT
    type(iptr_ptr) :: MCA14P
    INTEGER NWLFRQ,IUNWLR,IUNWLW
    real(chm_real)  TKELV, PRESMC, ACPAR1, ACPAR2, ACPAR3, ACECUT
    real(chm_real)  WTMP(MMVTYP), WTOT, DX1(3,MMVTYP)
    real(chm_real)  RWLADD,WLTOL,FLTNES
    LOGICAL QRSTRT
#if KEY_GCMC==1 /*gcmc_call_var*/
    INTEGER NIDXGC(MMVTYP), NIGCMC(MMVTYP)
    INTEGER NGCTRY, NCFB, NGCIN(MMVTYP)
    INTEGER NPSUM(MMVTYP)
    INTEGER NGRDCV
    INTEGER NGRDX, NGRDY, NGRDZ, NRGRID
    INTEGER NGCBLK, LSTGRD, LSTIND, NGDTOT, NGRDBK, GRDBLK
    real(chm_real)  GCMCBF, GCCUT2, GRDSIZ
    real(chm_real)  XGCMIN, YGCMIN, ZGCMIN, XGCMAX, YGCMAX, ZGCMAX
    real(chm_real)  GCSUM(MMVTYP), GC2SUM(MMVTYP)
    LOGICAL QGCGRD, QGCSPH
    integer :: ierror
#endif /*   (gcmc_call_var)*/

#if KEY_REPDSTR==1
    IF(QREPDSTR)THEN
      CALL PSETGLOB
      CALL PSYNC()
      CALL PSETLOC
    ENDIF
#endif 

#if KEY_GCMC==1
    ! If MC is called set-up GCMC arrays. cb3      
    if(.not.allocated(gcmcon)) call allocate_gcmc  
#endif

    !       Initialize some local variables
    DO I = 1, NMVTYP
      NMVTOT(I) = 0
      NMVACC(I) = 0
    ENDDO

    !       Set up initial and default values of variables.
    CALL MCINIT(COMLYN,COMLEN,IACCPF,NSTEPS,NSAVC,ISVFRQ,TKELV, &
        ISDMC,INBFMC,IEFRQ,IOAVES,IUNWRI,IUNREA,IUNCRD, &
        IOENGY,IARMF,IDOMCF,IMGFMC,IUMULT,ACPAR1,ACPAR2, &
        PICK,ACECUT,PRESMC,RVOLMC,QRSTRT, &
#if KEY_GCMC==1
        GCMCBF,NGCTRY,GCCUT2,XGCMIN,YGCMIN,ZGCMIN, &
        XGCMAX,YGCMAX,ZGCMAX,QGCGRD,QGCSPH,GRDSIZ, &
        NRGRID,NGRDX,NGRDY,NGRDZ,NCFB, &
#endif 
        NWLFRQ,IUNWLR,IUNWLW,RWLADD,WLTOL,FLTNES)
            
    !       Make sure there is a move set.
    IF ((NMVTYP .EQ. 0) .AND. (NSTEPS .GT. 0)) &
        CALL WRNDIE(-5,'<MCCALL>','NO MOVE SET')

#if KEY_SAMC==1 /* (samc) */
    IF (LSAMC) THEN
        CALL SPINITCHECK(IACCPF)
    ENDIF
#endif /* (samc) */

    !       Read the restart file if necessary
    IF (QRSTRT) CALL MCRDCD(IUNREA,ISDMC,X,Y,Z)

    !       If multicanonical, read the weights
    IF (IACCPF .EQ. 1) &
        CALL RDMULT(IUMULT,MLTLGP,NMULT,ACPAR1,ACPAR2,ACPAR3)

    !       Set up any necessary dynamically allocated arrays.
    !       print *,"calling mcstup"
    CALL MCSTUP(NMVTYP,NMVATM, &
        MVTYPE,ILNMVP, IARMF, &
        IDOMCF,ANISO, WEIGHT,WTOT,WTMP, &
        LCENTR,MCMINN, MCA14P, &
        TKELV,NACMVG,IACMVG &
#if KEY_GCMC==1
        ,GCMCON,NIDXGC, QGCGRD,NGCTRY, &
        NIGCMC &
#endif 
        )
        
        
    ! Allocate an array for storing the cumulated moves for the heuristic update of nonbonded list and images
    IF (INBFMC .LT. 0) THEN
        call chmalloc('mc.src','MCCALL','CUMULHEUR',NATOM,crl=CUMULHEUR)
        IF( ALLOCATED(CUMULHEUR) ) THEN
            CUMULHEUR = ZERO
        ENDIF
    ENDIF
    
    !       print *,"calling mcloop"

    !       Perform Monte Carlo simulation
    CALL MCLOOP(NSTEPS,IACCPF,ISDMC,PICK,TKELV,NMVTYP,MVTYPE, &
        NMVATM,IPIVTP,IMVNGP,ILNBDP, NLIMIT, &
        INBFMC,IEFRQ,IMGFMC,IOAVES,IUNWRI, &
        IUNCRD,IOENGY,NSAVC,ISVFRQ, &
        MBONDT,QLNBND, &
        IARMF, ARMP,ARMA,ARMB,ARMLIM,ARMMAX, &
        IDOMCF,ANISO,DOMCF, WTMP,WTOT, &
        TITLEA,NTITLA,NMVTOT,NMVACC, &
        ACPAR1,ACPAR2,ACPAR3,NMULT,MLTLGP%A, &
        MCMINN,MCMTYP,RMCSTP,RMCMNF,RMCMNG,RMCMNS,TFACT, &
        ACECUT,MCA14P, PRESMC,RVOLMC, &
        NXTMVG,NACMVG,IACMVG,ILNMVP,DX1,X,Y,Z,WMAIN, &
#if KEY_GCMC==1
        QGCMC,GCMCON,IGCMVG,GCMCBF,GCCUT2,NPSUM,GCSUM, &
        GC2SUM,XGCMIN,YGCMIN,ZGCMIN,XGCMAX,YGCMAX,ZGCMAX, &
        QGCGRD,QGCSPH,GRDSIZ,GCBLKR,NGRDX,NGRDY,NGRDZ, &
        NGDTOT,NIGCMC,NGCIN,NIDXGC, NGCTRY,NCFB, &
        NRGRID,NGRDCV, &
#endif 
        NWLFRQ,IUNWLR,IUNWLW,RWLADD,WLTOL,FLTNES)
    
    !       Compute some statistics
    !       print *,"calling mcstat"
    CALL MCSTAT(NMVTYP,NACMVG,IACMVG,MVLABL,NMVTOT,NMVACC)

    !       Free local dynamically allocated arrays.
    CALL MCFREE(NMVTYP,NMVATM, &
        MVTYPE,IMVNGP, IARMF, &
        IDOMCF,ANISO, IACCPF,MLTLGP,NMULT, &
        LCENTR,MCA14P, MCMINN &
#if KEY_GCMC==1
        ,NIDXGC, QGCGRD,NGCTRY &
#endif 
        )
        
        
    ! Deallocate an array for storing the cumulated moves for the heuristic update of nonbonded list and images
    IF (INBFMC .LT. 0) THEN
        call chmdealloc('mc.src','MCSTUP','CUMULHEUR',NATOM,crl=CUMULHEUR)
    ENDIF
    
#if KEY_REPDSTR==1
    IF(QREPDSTR)THEN
      CALL PSETGLOB
      CALL PSYNC()
      CALL PSETLOC
    ENDIF
#endif 
    !       Return to CHARMM control
    RETURN
  END SUBROUTINE MCCALL

  SUBROUTINE MCLOOP(NSTEPS,IACCPF,ISEED,PICK,TKELV,NMVTYP,MVTYPE, &
      NMVATM,IPIVTP,IMVNGP,IBLSTP, NLIMIT, &
      INBFMC,IEFRQ,IMGFMC,IOAVES,IUNWRI, &
      IUNCRD,IOENGY,NSAVC,ISVFRQ, &
      MBONDT,QBND,IARMF, &
      ARMP,ARMA,ARMB,ARMLIM,ARMMAX,IDOMCF,ANISO,DOMCF, &
      WEIGHT,WTOT,TITLEA,NTITLA, &
      NMVTOT,NMVACC,ACPAR1,ACPAR2,ACPAR3,NMULT,MLTLGP, &
      MCMINN,MCMTYP,RMCSTP,RMCMNF,RMCMNG,RMCMNS,TFACT, &
      ACECUT,MCA14P, PRESMC,RVOLMC, &
      NXTMVG,NACMVG,IACMVG,ILNMVP,DX1,X,Y,Z,WMAIN, &
#if KEY_GCMC==1
      QGCMC,GCMCON,IGCMVG,GCMCBF,GCCUT2,NPSUM,GCSUM, &
      GC2SUM,XGCMIN,YGCMIN,ZGCMIN,XGCMAX,YGCMAX, &
      ZGCMAX,QGCGRD,QGCSPH,GRDSIZ,GCBLKR,NGRDX,NGRDY, &
      NGRDZ,NGDTOT,NIGCMC,NGCIN,NIDXGC, NGCTRY, &
      NCFB, NRGRID,NGRDCV, &
#endif 
      NWLFRQ,IUNWLR,IUNWLW,RWLADD,WLTOL,FLTNES)
    !
    !       The main loop in Monte Carlo
    !
    !       Aaron R. Dinner
    !
    use chm_kinds
    use chm_types
    use clcg_mod,only:random
    use dimens_fcm
    use ace_module
    use bases_fcm
    use block_fcm
    use consta
    use deriv
    use energym
    use image
    use inbnd
    use mehmc
    use number
#if KEY_PBEQ==1
    use pbeq, only:qgsbp,gsbp0 
#endif
    use pert
    use psf
    use rxncom
    use stream
    use memory
    use parallel
    use repdstr
    use datstr, only: freedt_nbond, freedt_image
    use mccent
    use mc, only: mdxp
    use mcace
    use mcarmupt
    use mce
    use mcio, only: mcwrcd
    use mcmini, only: minsub
#if KEY_GCMC==1
    use mcmvgcmc  
#endif
    use mcmvhmc
    use mcmvrtrn, only: sclimg
    use mcmvutil, only: r2indx, cntall
    use mcwl
#if KEY_SAMC==1
    use samc
#endif

    implicit none
    !
    !       Passed Variables
    !
    INTEGER NSTEPS, IACCPF, ISEED, PICK
    INTEGER NMVTYP, MBONDT
    INTEGER MVTYPE(NMVTYP), NMVATM(NMVTYP)
    type(iptr_ptr) :: IPIVTP(NMVTYP), IMVNGP(NMVTYP), IBLSTP(NMVTYP)
    INTEGER NLIMIT(NMVTYP)
    INTEGER INBFMC, IMGFMC, IEFRQ
    INTEGER IOAVES,IUNWRI,IUNCRD,IOENGY,NSAVC,ISVFRQ
    INTEGER NTITLA
    INTEGER IARMF
    INTEGER IDOMCF
    INTEGER NMVTOT(NMVTYP), NMVACC(NMVTYP)
    INTEGER NMULT
    real(chm_real) :: MLTLGP(:)
    INTEGER MCMINN(NMVTYP), MCMTYP(NMVTYP)
    type(iptr_ptr) :: MCA14P
    INTEGER NXTMVG(NMVTYP), NACMVG, IACMVG(NMVTYP)
    type(iptr_ptr) :: ILNMVP(NMVTYP)
    INTEGER NWLFRQ,IUNWLR,IUNWLW
    real(chm_real), dimension(:) :: x, y, z, wmain
    real(chm_real)  TKELV, PRESMC, RVOLMC, ACECUT
    real(chm_real)  WEIGHT(NMVTYP), WTOT, TFACT(NMVTYP)
    real(chm_real)  ARMP(NMVTYP), ARMA(NMVTYP), ARMB(NMVTYP),  &
        ARMMAX(NMVTYP)
    real(chm_real)  DOMCF(NMVTYP)
    real(chm_real)  ACPAR1,ACPAR2,ACPAR3
    real(chm_real)  RMCMNS(NMVTYP), RMCMNF(NMVTYP), RMCMNG(NMVTYP)
    real(chm_real)  RMCSTP(NMVTYP)
    real(chm_real)  DX1(3,NMVTYP)
    real(chm_real)  RWLADD,WLTOL,FLTNES
    LOGICAL ARMLIM(NMVTYP), ANISO(NMVTYP), QBND(MBONDT,NMVTYP)
    character(len=80) :: titlea(:)
    
    LOGICAL :: LACC     !true if accepted move

#if KEY_SAMC==1 /* (samc) */
    INTEGER :: II, IJ   !iterators
#endif /* (samc) */
    
#if KEY_GCMC==1 /* (gcmc_loop_var) */
    INTEGER IGCMVG(NMVTYP), NIDXGC(NMVTYP)
    INTEGER NGCTRY, NCFB, NIGCMC(NMVTYP), NGCIN(NMVTYP)
    INTEGER NPSUM(NMVTYP)
    INTEGER NGRDX, NGRDY, NGRDZ, NGDTOT
    INTEGER NGRDBK, NRGRID, NGRDCV
    real(chm_real)  GCCUT2, GCMCBF, GRDSIZ
    real(chm_real)  XGCMIN, YGCMIN, ZGCMIN, XGCMAX, YGCMAX, ZGCMAX
    real(chm_real)  GCSUM(NMVTYP), GC2SUM(NMVTYP)
    LOGICAL QGCMC, GCMCON(:), QGCGRD, QGCSPH, GCBLKR(:)
#endif /*   (gcmc_loop_var)*/
    !
    !       Local Variables
    !
    integer,pointer,dimension(:) :: IMAGEP, IGRPP
    integer,allocatable,dimension(:) :: IMLIMP
    INTEGER ICYCLE, ISTEP
    INTEGER IAMC, NAMC, IGMC, NGMC, MCNBPP, OLDALL
    INTEGER IMVTYP, JMVTYP, IMCURR, IDX, IATOM
    real(chm_real),allocatable,dimension(:) :: IOL1CP, IOL2CP
    type(iptr_ptr) :: MCBLOP, MCBLGP, MCIMLP, MCIMGP, MCATTP, MCGTTP
    type(nonbondDataStructure) BDUMMY
    type(imageDataStructure) BDIMMY

#if KEY_PERT==1
    type(iptr_ptr) :: MCBLORP, MCBLOPP
    type(nonbondDataStructure) BDUMMYP
    type(imageDataStructure) BDIMMYP
    real(chm_real)  EGSBPR
    type(iptr_ptr) :: MCIMLRP, MCIMLPP
    integer,allocatable,dimension(:) :: IMLIMRP, IMLIMPP
    type(nonbondDataStructure) BDUMMYR
    type(imageDataStructure) BDIMMYR

#endif 
    real(chm_real)  ETOT, EMCOLD, EMCNEW, XOL1(3), XOL2(3)
    real(chm_real)  STEP1, CUTHR2, BETA
    LOGICAL QHEUR, LACEMC, QMOVE, LVOL
#if KEY_ACE==1
    !       ACE local variables
    INTEGER I, J
    INTEGER NACEBC, NCUTBC
    real(chm_real)  RSYS, TAU, ESMAX, FACT1, EDUMMY, EACEEL
    LOGICAL CSWIT
    !       Added for ACE2
    real(chm_real)  FACT2, FACT2H, ETURN, ETURNH, MEREF, MEREFH
#endif 
    !       HMC local variables
    INTEGER XOLD, YOLD, ZOLD, XNEW, YNEW, ZNEW, VX, VY, VZ, VK
    real(chm_real)  TEMPI, DEPOT, DEKIN
#if KEY_MEHMC==1
    !       Momentum-Enhanced HMC local variables
    real(chm_real)  BIASME
    LOGICAL QMEUPD
#endif 
    !       Constant pressure local variables
    INTEGER NSCALE
    real(chm_real)  RS

    !       Grand canonical local variables
    real(chm_real)  EGSBPO, EGSBPN
    LOGICAL LGCMC, LGCOLD, LIMVGC
#if KEY_GCMC==1
    integer,pointer,dimension(:) :: TEMPP
    INTEGER NGRNO
    real(chm_real)  CUTNB2, PCFB, EGSBPS
    LOGICAL LGCCB, LGCIN
#endif 
#if KEY_SCCDFTB==1
    real(chm_real) ECRAP         /*qc_010110*/
#endif

    !       Wang-Landau local variables
    integer,allocatable,dimension(:) :: IWLHST
    real(chm_real),allocatable,dimension(:) :: RWLLGF
    INTEGER NWLBIN, IWLBN1, IWLBN2
    LOGICAL ERROR

    !       -----------------------------------------------------------------
    !       Begin initializations

    IMAGEP => null()
#if KEY_GCMC==1
    TEMPP => null()  
#endif
    BETA = ONE/(TKELV*KBOLTZ)
    IAMC = 1
    NAMC = 1

    !       Pointers for the arrays in MCUPDT
    MCBLOP%A => null()
    MCBLGP%A => null()
    MCIMLP%A => null()
    MCIMGP%A => null()
    MCATTP%A => null()
    MCGTTP%A => null()
    call FREEDT_nbond(BDUMMY)
#if KEY_PERT==1
    MCBLORP%A => null()
    MCBLOPP%A => null()
    call FREEDT_nbond(BDUMMYR)
    call FREEDT_nbond(BDUMMYP)
#endif 
    IF (NTRANS .GT. 0) THEN
      call FREEDT_image(BDIMMY)
#if KEY_PERT==1
      MCIMLRP%A => null()
      MCIMLPP%A => null()
      call FREEDT_image(BDIMMYR)
      call FREEDT_image(BDIMMYP)
      IF (QPERT) THEN
          call chmalloc('mc.src','MCLOOP','IMLIMRP',2*NTRANS,intg=IMLIMRP)
          call chmalloc('mc.src','MCLOOP','IMLIMPP',2*NTRANS,intg=IMLIMPP)
          IMLIMRP(1:2*NTRANS) = 0
          IMLIMPP(1:2*NTRANS) = 0
      ENDIF
#endif 
      call chmalloc('mc.src','MCLOOP','IMLIMP',2*NTRANS,intg=IMLIMP)
      IMLIMP(1:2*NTRANS) = 0
    ELSE
      !         For images, this is in MCUPDT because size requried varies
      IF (LGROUP) THEN
          call mc_alloc_centers('mc.src', 'MCLOOP', NGRP)
          CALL CNTALL(NGRP,X,Y,Z,xcent,ycent, &
              zcent,QCENT)
      ELSE
          !            XCENT = 0.0
          !            YCENT = 0.0
          !            ZCENT = 0.0
          !            QCENT = 0.0
      ENDIF
    ENDIF
    !       NOFORC does not do much without fast routines
#if KEY_BLOCK==1
    NOFORC = .TRUE.  
#endif

#if KEY_ACE==1
    !       If necessary, initialize the arrays for ACE
    CALL MCACIN(TAU,RSYS,FACT1,ESMAX, &
        bnbnd%IBLO14,bnbnd%INB14, &
        MCA14P%A, cgsacp, sa14p, &
        ETURN,ETURNH,MEREF,MEREFH,FACT2,FACT2H,CSWIT,X,Y,Z)
#endif 

    IF (INBFMC .LT. 0) CUTHR2 = ((CUTNB - CTOFNB)*0.5)**2
    QHEUR = .FALSE.

    !       Set GCMC logicals even if GCMC code not compiled.
    LGCMC  = .FALSE.
    LIMVGC = .FALSE.
    LGCOLD = .TRUE.
#if KEY_GCMC==1 /*gcmc_ini*/
    CALL GCMCIN(LGCMC,NMVTYP,MVTYPE,QGCGRD,GCMCON,CUTNB2,CUTNB, &
        XGCMIN,YGCMIN,ZGCMIN,XGCMAX,YGCMAX,ZGCMAX,NATOM, &
        NATIM,NTRANS,LIMALL,bimag%IMATTR, &
        NPSUM,GCSUM,GC2SUM,NGCTRY,GCBLKR, NGRDBK, &
        NGRDX,NGRDY,NGRDZ)
    !       Turn on but not off QGCMC in case some atoms are already inactive
    IF (LGCMC) THEN
      QGCMC = .TRUE.

#if KEY_PBEQ==1 /*pbeqtst*/
      IF (QGSBP) THEN
          CALL GSBP0(NATOM,X,Y,Z,CG,EGSBPS,DX,DY,DZ,1,.FALSE.&
#if KEY_SCCDFTB==1
               ,.FALSE.,ECRAP &       /*qc_010110*/
#endif
              )
#if KEY_PERT==1
          IF (QPERT) THEN
            CALL GSBP0(NATOM,X,Y,Z,PPCG, &
                  EGSBPR,DX,DY,DZ,1,.FALSE.&
#if KEY_SCCDFTB==1
                  ,.FALSE.,ECRAP &       /*qc_010110*/
#endif
                  )
            EGSBPS = EGSBPS*LAMDA + EGSBPR*LAMDAM
          ENDIF
#endif 
      ENDIF
#endif /* (pbeqtst)*/
    ENDIF
#endif /*  (gcmc_ini)*/

    !       ARD and Ao Ma 06-06-30
    !       Histogram arrays for Wang-Landau sampling
    IF (IACCPF .EQ. 3) THEN
      NWLBIN = product(NRXSTT(1:NRXNCR))
      call chmalloc('mc.src','MCLOOP','IWLHST',NWLBIN,intg=IWLHST)
      call chmalloc('mc.src','MCLOOP','RWLLGF',NWLBIN,crl=RWLLGF)
      CALL WLINIT(NWLBIN,NRXNCR,NRXSTT,IWLHST,RWLLGF,IUNWLR, &
            IWLBN1)
    ENDIF
    
    CALL MCUPDT(ETOT,MCBLOP,MCBLGP,MCIMLP,MCIMGP,MCATTP, &
        0,ICYCLE,1,1,1,.TRUE.,X,Y,Z,WMAIN,DX,DY,DZ, &
        NATOM,NGRP,BNBND,BIMAG, &
        BDUMMY,BDIMMY,IA2GP,IGPBS, &
        MCGTTP, &
#if KEY_ACE==1
        RSYS,ESMAX,FACT1, &
        FACT2,FACT2H,ETURN,ETURNH,MEREF,MEREFH,CSWIT, &
#endif 
        RVOLMC,TKELV &
#if KEY_GCMC==1
        ,GCMCON,LGCMC,NMVTYP,MVTYPE,QGCGRD,NRGRID, &
        XGCMIN,YGCMIN,ZGCMIN,XGCMAX,YGCMAX,ZGCMAX, &
        NPSUM,GCSUM,GC2SUM,NGCIN,NIDXGC,QGCSPH, &
        IDXGCP,IMVNGP,NGCTRY,GCBLKR, NGRDBK, &
        NGDTOT, NGRDCV, GRDSIZ, &
        NGRDX,NGRDY,NGRDZ &
#endif 
#if KEY_PERT==1
        ,MCBLORP,MCBLOPP,BDUMMYR,BDUMMYP, &
        BNBNDR,BNBNDP &
        ,MCIMLRP,MCIMLPP,BDIMMYR,BDIMMYP, &
        BIMAGP,BIMAGR &
#endif 
        )

    !       Counters for sequential selection of moves
    IDX = 0
    IF (PICK .EQ. 2) THEN
      IMCURR = R2INDX(WTOT*RANDOM(ISEED),WEIGHT,NACMVG)
    ELSE
      IMCURR = 1
    ENDIF
    IMVTYP = IACMVG(IMCURR)

    !       Allocate space for saving old coordinates
    CALL GTOLDC(OLDALL,NMVTYP,ILNMVP,MCMINN,NMVATM)
    call chmalloc('mc.src','MCLOOP','IOL1CP',3*OLDALL,crl=IOL1CP)
#if KEY_ACE==1
    if (LACE) then
      call chmalloc('mc.src','MCLOOP','IOL2CP',3*OLDALL,crl=IOL2CP)
    endif
#endif 

#if KEY_MEHMC==1
! Allocate space if not alreadly allocated
    if(.not.allocated(vmeavx).or. &
         (allocated(vmeavx) .and. size(vmeavx)/=natom)) then
       call allocate_mehmc(natom)
    endif
    !       Momentum-Enhanced Hybrid Monte Carlo initializations and checks
    CALL MEINIT(QMEUPD,VMEAVX,VMEAVY,VMEAVZ,NMVTYP,MCMINN,MCMTYP, &
        NATOM)
    IF (QMEUPD .AND. IACCPF .NE. 0) THEN
      CALL WRNDIE(-5,'<MCLOOP>', &
            'MEHMC requires canonical weighting (IACCept 0)!')
    ENDIF
#endif

#if KEY_SAMC==1 /* (samc) */
    IF (LSAMC) THEN
      LSPDONE  = .FALSE.
      CALL SPALLOCARRAYS(NATOM)
#if KEY_ACE==1 /* (ace_sp) */
      IF (LACE) THEN
        CALL SPACEALLOCARRAYS(NATOM)
      ENDIF
#endif /* (ace_sp) */
    ENDIF
#endif /* (samc) */

    !       End initializations
    !       -----------------------------------------------------------------
    !
    !       -----------------------------------------------------------------
    !       Begin main loop

    DO ISTEP = 1, NSTEPS
        
      CALL MCUPDT(ETOT,MCBLOP,MCBLGP,MCIMLP,MCIMGP,MCATTP, &
            ISTEP,ICYCLE,INBFMC,IEFRQ,IMGFMC,QHEUR, &
            X,Y,Z,WMAIN,DX,DY,DZ,NATOM,NGRP, &
            BNBND,BIMAG,BDUMMY, &
            BDIMMY,IA2GP,IGPBS,MCGTTP, &
#if KEY_ACE==1
            RSYS,ESMAX,FACT1, &
            FACT2,FACT2H,ETURN,ETURNH,MEREF,MEREFH,CSWIT, &
#endif 
            RVOLMC,TKELV &
#if KEY_GCMC==1
            ,GCMCON,LGCMC,NMVTYP,MVTYPE,QGCGRD,NRGRID, &
            XGCMIN,YGCMIN,ZGCMIN,XGCMAX,YGCMAX,ZGCMAX, &
            NPSUM,GCSUM,GC2SUM,NGCIN,NIDXGC,QGCSPH, &
            IDXGCP,IMVNGP,NGCTRY,GCBLKR, NGRDBK, &
            NGDTOT, NGRDCV, GRDSIZ, &
            NGRDX,NGRDY,NGRDZ &
#endif 
#if KEY_PERT==1
            ,MCBLORP,MCBLOPP,BDUMMYR,BDUMMYP, &
            BNBNDR,BNBNDP &
            ,MCIMLRP,MCIMLPP,BDIMMYR,BDIMMYP, &
            BIMAGP,BIMAGR &
#endif 
            )
      
      !         Pick which move instance to do.
      CALL MCPICK(IMVTYP,IDX,PICK,IMCURR,NACMVG,IACMVG,NMVTYP, &
            NMVATM,WTOT,WEIGHT,ISEED &
#if KEY_GCMC==1
            ,MVTYPE,IGCMVG,IDXGCP,NIDXGC  & 
#endif
            )
      NMVTOT(IMVTYP) = NMVTOT(IMVTYP) + 1
      
#if KEY_SAMC==1
        IF (LSAMC .AND. LSPGROUP(IMVTYP)) THEN
            WEPSILON => WEPARR(IMVTYP)
            MEPSILON => MEPARR(IMVTYP)
            NEPSILON => NEPARR(IMVTYP)
        ENDIF
#endif

#if KEY_ACE==1
      !         If minimization is to be done, ACE calculations are done
      !         by standard energy routines rather than MCACEE
      LACEMC = LACE .AND. (MCMINN(IMVTYP) .EQ. 0)
#else /**/
      LACEMC = .FALSE.
#endif 
      !         If a volume move, use standard energy routines and update total.
      LVOL   = MVTYPE(IMVTYP) .EQ. 7
#if KEY_GCMC==1 /*gcmc_test*/
      !         Otherwise see if it is a GCMC move and set some variables
      CALL GCTEST(LIMVGC,LGCOLD,LGCIN,GCMCON,MVTYPE(IMVTYP), &
            IMVNGP(IMVTYP)%A,IGCMVG(IMVTYP),IDX,QGCSPH, &
            XGCMIN,YGCMIN,ZGCMIN,XGCMAX,YGCMAX,ZGCMAX,X,Y,Z)
      !         Skip (reject) deletions outside the insertion region.
      IF (LIMVGC .AND. LGCOLD .AND. .NOT. LGCIN) GOTO 438
#endif /*   (gcmc_test)*/

      !         Save the old coordinates.
      !         If minimization save all free coordinates by swapping the
      !         ALLMVP and IMVNGP pointers.
      CALL SVOLDC(IDX, ILNMVP(IMVTYP)%A, IOL1CP, XOL1,X,Y,Z, &
            MCMINN(IMVTYP), LGCMC)
      
      !         Compute the old energy contribution of the moving atoms.
      IF (MCMINN(IMVTYP).EQ.0 .AND. .NOT. LVOL &
                .AND. .NOT. (LIMVGC .AND. .NOT. LGCOLD)) THEN

#if KEY_PARALLEL==1
          IF(NUMNOD .GT. 1) THEN
             CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,0)
            EMCOLD = EPROP(EPOT)
            CALL PSND8(EMCOLD,1)
          ELSE
#endif 
            CALL MCENER(EMCOLD,MVTYPE(IMVTYP),IDX, ILNMVP(IMVTYP)%A, &
                  IBLSTP(IMVTYP)%A, IMCGRP(IMVTYP)%A, NOECNP(IMVTYP)%A, &
                  ISKIP, IMVATP, IMVGRP, MBONDT,QBND(:,IMVTYP), &
                  MCBLOP%A, MCBLGP%A, MCIMLP%A, MCIMGP%A, .TRUE., &
                  BDUMMY,BDIMMY, MCATTP%A, IAMC,NAMC,IMLIMP, MCA14P%A, &
                  IGMC,NGMC, MCGTTP%A, X,Y,Z,LGCMC, &
#if KEY_PERT==1
                  MCBLORP%A, MCBLOPP%A, BDUMMYR, BDUMMYP,  & 
#endif
#if KEY_PERT==1
                  MCIMLRP%A, MCIMLPP%A, IMLIMRP, IMLIMPP,  & 
#endif
#if KEY_PERT==1
                  BDIMMYR,BDIMMYP,    & 
#endif
                  EGSBPO)
#if KEY_PARALLEL==1
          ENDIF
#endif 
#if KEY_GCMC==1
#if KEY_PBEQ==1
      ELSEIF (QGSBP .AND. LIMVGC .AND. .NOT. LGCOLD) THEN
#if KEY_PARALLEL==1
          IF(NUMNOD .GT. 1) THEN
             CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,0)
            EMCOLD = EPROP(EPOT)
            CALL PSND8(EMCOLD,1)
          ELSE
#endif 
            EMCOLD = EGSBPS
#if KEY_PARALLEL==1
          ENDIF
#endif 

#endif 
#endif 
      ELSE
          EMCOLD = ZERO
      ENDIF

#if KEY_GCMC==1 /*gcmc_obdel*/
      !         Grand canonical orientation bias (deletion)
      IF (LIMVGC .AND. LGCOLD .AND. NCFB .GT. 1) THEN
          CALL MCENOB(PCFB,NCFB,LGCOLD,EMCOLD,EGSBPO,ISEED,IDX, &
              ILNMVP(IMVTYP)%A, NIGCMC(IMVTYP), &
              IBLSTP(IMVTYP)%A, IMCGRP(IMVTYP)%A, NOECNP(IMVTYP)%A, &
               ISKIP, IMVATP, IMVGRP, MBONDT,QBND(:,IMVTYP), &
              MCBLOP%A, MCBLGP%A, MCIMLP%A, MCIMGP%A, BDUMMY, BDIMMY, &
              MCATTP%A, IAMC,NAMC,IMLIMP,NTRANS,IMTRNS, &
              NOROT,bimag%IMATPT, MCA14P%A, IGMC,NGMC, MCGTTP%A, X,Y,Z, &
              BETA,GCMCON &
#if KEY_PERT==1
               ,MCBLORP%A, MCBLOPP%A, BDUMMYR, BDUMMYP  & 
#endif
#if KEY_PERT==1
               ,MCIMLRP%A, MCIMLPP%A, IMLIMRP, IMLIMPP, & 
#endif
#if KEY_PERT==1
               BDIMMYR,BDIMMYP    & 
#endif
              )
      ENDIF
#endif /*   (gcmc_obdel)*/

#if KEY_ACE==1
      !         ---------------------------------------------------------------
      !         Begin first ACE self energy calculation
      IF (LACEMC) THEN
          !           Set up a list of the BSARR that are changing (by any amount)
          CALL STUPBC(IBCHGP, NACEBC, NOTADP, &
              BDUMMY%INBLO,BDUMMY%JNB, &
              IAMC,NAMC)
          !           Get the old contribution to the ACE self energies
          CALL GTESLF(ESR1P, BDUMMY%INBLO, &
              BDUMMY%JNB,IAMC,NAMC,X,Y,Z,ces1, &
              ces2,sig2i,mue4,CSWIT, &
              CTOFNB*CTOFNB,CTONNB*CTONNB)
      ENDIF
      !         End first ACE self energy calculation
      !         ---------------------------------------------------------------
#endif 
      !         ---------------------------------------------------------------
      !         Begin first move linking loop
      NSCALE = 0
      QMOVE  = .FALSE.
      JMVTYP = IMVTYP

#if KEY_SAMC==1 /* (samc) */
        IF (LSAMC .AND. LSPGROUP(IMVTYP)) THEN
          CALL SPCOPYCOORDINATES(NATOM,X,Y,Z,.TRUE.)
        ENDIF
#endif /* (samc) */

      !         Make a random move of the selected type to the selected atom
      NSCALE = 0
10      CALL MKMOVE(ISEED,MVTYPE(JMVTYP),IDX,IATOM,IPIVTP(JMVTYP), &
              IMVNGP(JMVTYP),MDXP(JMVTYP),ANISO(JMVTYP), &
              DX1(1,JMVTYP),QMOVE,NLIMIT(JMVTYP),NSCALE,X,Y,Z &
#if KEY_GCMC==1
              ,GCMCON,NGCTRY,NGCIN(IMVTYP), ISCVTY(IMVTYP)%A, &
              NGCPCP(IMVTYP)%A, GCCUT2,XGCMIN,YGCMIN,ZGCMIN, &
              XGCMAX,YGCMAX,ZGCMAX,LGCCB,MCBLOP,CUTNB2, &
              QGCGRD,QGCSPH,GRDBLK,NGRDBK,NGRDCV, &
              LSTGRD,GRDSIZ,NGRDX,NGRDY,NGRDZ,NGCBLK &
#endif 
              )
              
#if KEY_SAMC==1 /* (samc) */
        IF (LSAMC .AND. LSPGROUP(IMVTYP)) THEN
          CALL SPCOPYCOORDINATES(NATOM,X,Y,Z,.FALSE.)
        ENDIF
#endif /* (samc) */

       IF (NXTMVG(JMVTYP).GT.0) THEN
          JMVTYP = NXTMVG(JMVTYP)
          GOTO 10
       ENDIF
       !         End first move linking loop
       !         ---------------------------------------------------------------

       !         If necessary, update group centers of moving atoms
        IF (QMOVE) THEN
            IF (LGROUP) IGRPP => IMCGRP(IMVTYP)%A(IDX)%A
            IF (NTRANS > 0) IMAGEP => ILNMVP(IMVTYP)%A(IDX)%A
            CALL CNTTRN(XCENT,YCENT,ZCENT,LGROUP,NTRANS,IGRPP,IMAGEP, &
                MCATTP%A, IMTRNS,NOROT,BIMAG, MCGTTP%A, IGPBS, &
                X,Y,Z &
#if KEY_GCMC==1
               ,GCMCON        & 
#endif
                )
        ENDIF

#if KEY_GCMC==1 /*gcmc_upnb*/
        !         Update the non-bonded lists for GCMC
        !         Group based calculations will break this code
        IF (LIMVGC) THEN
            IF (.NOT. LGCOLD) THEN

#if KEY_PARALLEL==1
              IF (NUMNOD .GT. 1) THEN
                  CALL NBONDS(X,Y,Z,BNBND,BIMAG)! new structures:,LNBND,BIMAG,LIMAG)
              ELSE
#endif 
                  TEMPP => ILNMVP(IMVTYP)%A(IDX)%A
                  if (.not. allocated(IMVATP)) call wrndie(-4, 'MCLOOP', 'IMVATP not allocated')
                  CALL GCADNB(MCBLOP%A, IMVATP, TEMPP, CUTNB2, &
                      GCMCON,X,Y,Z &
#if KEY_PERT==1
                     ,QPERT,PERTIP, .TRUE.   & 
#endif
                      )
#if KEY_PERT==1
                  IF (QPERT) THEN
                    CALL GCADNB(MCBLORP%A, IMVATP, TEMPP, CUTNB2, &
                          GCMCON,X,Y,Z,QPERT,PERTIP,.FALSE.)
                    CALL GCADNB(MCBLOPP%A, IMVATP, TEMPP, CUTNB2, &
                          GCMCON,X,Y,Z,QPERT,PERTIP,.FALSE.)
                  ENDIF
#endif 
                  IF (NTRANS .GT. 0) THEN
                    CALL GCADIM(MCIMLP%A, IMVATP, TEMPP, &
                          bimag%IMATTR, MCATTP%A, &
                          CUTNB2,GCMCON,NATOM,NATIM,X,Y,Z &
#if KEY_PERT==1
                        ,QPERT,PERTIP, .TRUE.   & 
#endif
                          )
#if KEY_PERT==1
                    IF (QPERT) THEN
                        CALL GCADIM(MCIMLRP%A, IMVATP, TEMPP, &
                            bimag%IMATTR, MCATTP%A, &
                            CUTNB2,GCMCON,NATOM,NATIM,X,Y,Z &
#if KEY_PERT==1
                           ,QPERT,PERTIP, .FALSE.   & 
#endif
                            )
                        CALL GCADIM(MCIMLPP%A, IMVATP, TEMPP, &
                            bimag%IMATTR, MCATTP%A, &
                            CUTNB2,GCMCON,NATOM,NATIM,X,Y,Z &
#if KEY_PERT==1
                           ,QPERT,PERTIP, .FALSE.   & 
#endif
                            )
                    ENDIF
#endif 
                  ENDIF
#if KEY_PARALLEL==1
              ENDIF
#endif 
            ENDIF
        ENDIF
#endif /*   (gcmc_upnb)*/

      !         Compute the new energy contribution of the moving atoms
      !         Deal with moves that utilize the energy routines via minimization
      !         and dynamics to propagate the structure here.
      DEKIN = ZERO
      IF (MCMINN(IMVTYP) .LT. 0) THEN

          !           Turn the centering back on but do not reallocate these arrays
          LCENTR = -1
          !           If the acceptance criterion is Tsallis, use appropriate dynamics
          CALL MKHMC(TEMPI,ISEED,TKELV,-MCMINN(IMVTYP), &
              MDXP(IMVTYP)%A(IDX), IDX, &
              IMVNGP(IMVTYP)%A, ALLMVP%A, &
              IGAMMA(IMVTYP)%A, (IACCPF.EQ.2),ACPAR1,ACPAR2, &
              BNBND,BIMAG,X,Y,Z &
#if KEY_MEHMC==1
              ,ISTEP,MCMTYP(IMVTYP),RMCSTP(IMVTYP) &
              ,RMCMNF(IMVTYP),QMEUPD,BIASME &
#endif 
              )
          !           Need to check into high frequency correction
          DEPOT = EPROP(EPOT)  - ETOT
          DEKIN = EPROP(TOTKE) - TEMPI
#if KEY_MEHMC==1
          !           If MEHMC, assume canonical (Metropolis) ensemble and put the
          !           acceptance criterion reweighting in KE.
          IF (MCMTYP(IMVTYP).GT.0) DEKIN=DEKIN-(LOG(BIASME))/BETA
#endif 
          LCENTR =  1
      ELSE IF (MCMINN(IMVTYP) .GT. 0) THEN
          !           Turn the centering back on but do not reallocate these arrays
          LCENTR = -1
          STEP1 = RMCSTP(IMVTYP)
          CALL MINSUB(MCMINN(IMVTYP),MCMTYP(IMVTYP),STEP1, &
              RMCMNF(IMVTYP),RMCMNG(IMVTYP),RMCMNS(IMVTYP), &
              BNBND,BIMAG,X,Y,Z)
          DEPOT = EPROP(EPOT) - ETOT
          LCENTR =  1
      ELSE IF (LVOL) THEN
          !           For volume changes, store the PV work in DEKIN
          CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,0)
          RS = EXP(DX1(1,IMVTYP))
          DEPOT = EPROP(EPOT) - ETOT
          DEKIN = PRESMC*RVOLMC*(RS*RS*RS - ONE) &
              - THREE*(NSCALE+1)*(DX1(1,IMVTYP)/BETA)
       ELSE
          IF (QMOVE .AND. .NOT. (LIMVGC .AND. LGCOLD)) THEN
#if KEY_PARALLEL==1
            IF (NUMNOD .GT. 1)THEN
                CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,0)
                EMCNEW=EPROP(EPOT)
                CALL PSND8(EMCNEW,1)
                !                  IF(MYNOD .EQ. 0)WRITE(OUTU,*)'EMCNEW=',EMCNEW
            ELSE
#endif 
                CALL MCENER(EMCNEW,MVTYPE(IMVTYP),IDX, &
                    ILNMVP(IMVTYP)%A, &
                    IBLSTP(IMVTYP)%A, IMCGRP(IMVTYP)%A, NOECNP(IMVTYP)%A, &
                    ISKIP, IMVATP, IMVGRP, MBONDT,QBND(1,IMVTYP), &
                    MCBLOP%A, MCBLGP%A, MCIMLP%A, MCIMGP%A, LIMVGC, &
                    BDUMMY,BDIMMY, MCATTP%A, IAMC,NAMC,IMLIMP, MCA14P%A, &
                    IGMC,NGMC, MCGTTP%A, X,Y,Z,LGCMC, &
#if KEY_PERT==1
                     MCBLORP%A, MCBLOPP%A, BDUMMYR, BDUMMYP,  & 
#endif
#if KEY_PERT==1
                     MCIMLRP%A, MCIMLPP%A, IMLIMRP, IMLIMPP,  & 
#endif
#if KEY_PERT==1
                     BDIMMYR,BDIMMYP,    & 
#endif
                    EGSBPN)
#if KEY_PARALLEL==1
            ENDIF
#endif 

#if KEY_GCMC==1 /*gcmc_obins*/
            !             Grand canonical orientation bias (insertion)
            IF (LIMVGC .AND. .NOT. LGCOLD .AND. NCFB .GT. 1) THEN
                CALL MCENOB(PCFB,NCFB,LGCOLD,EMCNEW,EGSBPN,ISEED,IDX, &
                    ILNMVP(IMVTYP)%A, NIGCMC(IMVTYP), &
                    IBLSTP(IMVTYP)%A, IMCGRP(IMVTYP)%A, NOECNP(IMVTYP)%A, &
                     ISKIP, IMVATP, IMVGRP, MBONDT,QBND(:,IMVTYP), &
                    MCBLOP%A, MCBLGP%A, MCIMLP%A, MCIMGP%A, BDUMMY,BDIMMY, &
                    MCATTP%A, IAMC,NAMC,IMLIMP,NTRANS,IMTRNS, &
                    NOROT,bimag%IMATPT, MCA14P%A, IGMC,NGMC, MCGTTP%A, X,Y,Z, &
                    BETA,GCMCON &
#if KEY_PERT==1
                     ,MCBLORP%A, MCBLOPP%A, BDUMMYR, BDUMMYP  & 
#endif
#if KEY_PERT==1
                     ,MCIMLRP%A, MCIMLPP%A, IMLIMRP, IMLIMPP, & 
#endif
#if KEY_PERT==1
                     BDIMMYR,BDIMMYP    & 
#endif
                    )
            ENDIF

#if KEY_PBEQ==1 /*pbeqtst2*/
          ELSEIF (QGSBP .AND. LIMVGC .AND. LGCOLD) THEN

#if KEY_PARALLEL==1
            IF (NUMNOD .GT. 1)THEN
                CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,0)
                EMCNEW=EPROP(EPOT)
                CALL PSND8(EMCNEW,1)
            ELSE
#endif 
                CALL GSBP0(NATOM,X,Y,Z,CG,EGSBPN,DX,DY,DZ,1,.FALSE.&
#if KEY_SCCDFTB==1
                     ,.FALSE.,ECRAP &         /*qc_010110*/
#endif
                    )
#if KEY_PERT==1
                IF (QPERT) THEN
                  CALL GSBP0(NATOM,X,Y,Z,PPCG,EGSBPR, &
                        DX,DY,DZ,1,.FALSE.&
#if KEY_SCCDFTB==1
                        ,.FALSE.,ECRAP &       /*qc_010110*/
#endif
                        )
                  EGSBPN = EGSBPN*LAMDA + EGSBPR*LAMDAM
                ENDIF
#endif 
                EMCNEW = EGSBPN
#if KEY_PARALLEL==1
            ENDIF
#endif 
#endif /*   (pbeqtst2)*/
#endif /*   (gcmc_obins)*/

          ELSE
            EMCNEW = ZERO
          ENDIF
          DEPOT = EMCNEW - EMCOLD
#if KEY_GCMC==1 /*gcmc_bias*/
          !           Put chemical potential and cavity bias corrections in DEKIN
          IF (LIMVGC) THEN
            !             Use of IOL1CP assumes no minimization or ACE for GCMC
            DEKIN = GCBIAS(ISEED,GCMCBF, IMVNGP(IMVTYP)%A, &
                  IDX,LGCOLD,LGCCB,QGCGRD,NGCTRY, &
                  NCFB,PCFB,NGCIN(IMVTYP), NOCVTY(IMVTYP)%A, &
                  NGCPCP(IMVTYP)%A, ISCVTY(IMVTYP)%A, &
                  NGCBLK,NGRDCV,NGDTOT,NRGRID,GRDSIZ, &
                  NGRDX,NGRDY,NGRDZ,XGCMIN,YGCMIN,ZGCMIN, &
                  LSTGRD,LSTIND, IOL1CP, X,Y,Z)
            !             It will be multiplied by BETA in MCACPT, so divide here
            DEKIN = DEKIN / BETA
          ENDIF
#endif /*   (gcmc_bias)*/

          !           ARD and Ao Ma 06-06-30
          !           Check if move is in bounds for Wang-Landau sampling
          IF (IACCPF .EQ. 3) THEN
            CALL WLUPDT(QMOVE, IWLBN1, IWLBN2, RWLLGF, NWLBIN, &
                  DEKIN, BETA)
          ENDIF


       ENDIF

#if KEY_ACE==1
      !         ---------------------------------------------------------------
      !         Begin ACE total energy calculation
      !         Get the new energy and then switch back to the old coordinates
      !         and get the old energy.  The code is structured in this way for
      !         ACE so that only interactions involving self energies changing
      !         by more than ACECUT are included.
      IF (LACEMC .AND. QMOVE) THEN
          !           Get the new contribution to the ACE self energies
          CALL GTESLF(ESR2P, BDUMMY%INBLO, &
              BDUMMY%JNB,IAMC,NAMC,X,Y,Z,ces1, &
              ces2,sig2i,mue4,CSWIT, &
              CTOFNB*CTOFNB,CTONNB*CTONNB)
          !           Add the contribution and compile a list of the ESARR changing by
          !           more than ACECUT (Mode = 1 flag in last argument).
          CALL ADESLF(IBCUTP, NCUTBC, ACECUT, &
              ESR1P, ESR2P, IBCHGP, NACEBC, 1)

          !           Get the new contribution to the ACE total energy
          CALL MCACEE(EACEEL,IAMC,NAMC, ESARP, ETMPP, &
              BSARP, CG2RP, RSYS, NOTADP, &
              MCBLOP%A, MCA14P%A, IBCHGP, NACEBC, &
              IBCUTP, NCUTBC, FACT2,FACT2H, &
              ETURN,ETURNH,MEREF,MEREFH,CSWIT)
          EMCNEW = EMCNEW + EACEEL

          !           Save the new coordinates
          CALL SVOLDC(IDX, ILNMVP(IMVTYP)%A, IOL2CP, XOL2, &
              X,Y,Z,MCMINN(IMVTYP), LGCMC)

          !           Restore the old coordinates
          CALL RESTOR(IDX, ILNMVP(IMVTYP)%A, IOL1CP, XOL1, &
              X,Y,Z,MCMINN(IMVTYP), LGCMC)
          CALL CNTTRN(XCENT,YCENT,ZCENT,LGROUP,NTRANS,IGRPP,IMAGEP, &
              MCATTP%A, IMTRNS,NOROT,BIMAG, MCGTTP%A, IGPBS, &
              X,Y,Z &
#if KEY_GCMC==1
               ,GCMCON        & 
#endif
              )

          !           Restore the ACE self energies
          CALL ADESLF(IBCUTP, NCUTBC, ACECUT, &
              ESR2P, ESR1P, IBCHGP, NACEBC, 0)

          !           Get the old contribution of the ACE total energy
          CALL MCACEE(EACEEL,IAMC,NAMC, ESARP, ETMPP, &
              BSARP, CG2RP, RSYS, NOTADP, &
              MCBLOP%A, MCA14P%A, IBCHGP, NACEBC, &
              IBCUTP, NCUTBC, FACT2,FACT2H, &
              ETURN,ETURNH,MEREF,MEREFH,CSWIT)
          EMCOLD = EMCOLD + EACEEL
          DEPOT = EMCNEW - EMCOLD
      ENDIF    
      !         End ACE total energy calculation
      !         ---------------------------------------------------------------
#endif

#if KEY_SAMC==1
      LSPDONE=.FALSE.
#endif

      LACC=.FALSE.
      
#if KEY_SAMC==1 /* (samc) */
          IF (LSAMC .AND. LSPGROUP(IMVTYP)) THEN
            CALL SPGENERATE(NATOM,ANISO(IMVTYP),IDX,MDXP(IMVTYP),IMVTYP,IPIVTP(IMVTYP), &
                            IMVNGP(IMVTYP),MVTYPE(IMVTYP),NLIMIT(IMVTYP), &
                            ISEED,.TRUE.)
                            
            CALL SPGENERATE(NATOM,ANISO(IMVTYP),IDX,MDXP(IMVTYP),IMVTYP,IPIVTP(IMVTYP), &
                            IMVNGP(IMVTYP),MVTYPE(IMVTYP),NLIMIT(IMVTYP), &
                            ISEED,.FALSE.)
                         
            DO II=1,MEPSILON
              DO IJ=1,NEPSILON
                
                SPCX => COORINI(II,IJ,1:NATOM)
                SPCY => COORINI(II,IJ,NATOM+1:2*NATOM)
                SPCZ => COORINI(II,IJ,2*NATOM+1:3*NATOM)
                
                CALL MCENER(ENEINI(II,IJ),MVTYPE(IMVTYP),IDX, &
                      ILNMVP(IMVTYP)%A, &
                      IBLSTP(IMVTYP)%A, IMCGRP(IMVTYP)%A, NOECNP(IMVTYP)%A, &
                      ISKIP, IMVATP, IMVGRP, MBONDT,QBND(1,IMVTYP), &
                      MCBLOP%A, MCBLGP%A, MCIMLP%A, MCIMGP%A, LIMVGC, &
                      BDUMMY,BDIMMY, MCATTP%A, IAMC,NAMC,IMLIMP, MCA14P%A, &
                      IGMC,NGMC, MCGTTP%A, SPCX,SPCY,SPCZ,LGCMC, &
#if KEY_PERT==1
                      MCBLORP%A, MCBLOPP%A, BDUMMYR, BDUMMYP,  &
                      MCIMLRP%A, MCIMLPP%A, IMLIMRP, IMLIMPP,  &
                      BDIMMYR,BDIMMYP,    &
#endif
                      EGSBPN)

                SPCX => COORFIN(II,IJ,1:NATOM)
                SPCY => COORFIN(II,IJ,NATOM+1:2*NATOM)
                SPCZ => COORFIN(II,IJ,2*NATOM+1:3*NATOM)

                CALL MCENER(ENEFIN(II,IJ),MVTYPE(IMVTYP),IDX, &
                      ILNMVP(IMVTYP)%A, &
                      IBLSTP(IMVTYP)%A, IMCGRP(IMVTYP)%A, NOECNP(IMVTYP)%A, &
                      ISKIP, IMVATP, IMVGRP, MBONDT,QBND(1,IMVTYP), &
                      MCBLOP%A, MCBLGP%A, MCIMLP%A, MCIMGP%A, LIMVGC, &
                      BDUMMY,BDIMMY, MCATTP%A, IAMC,NAMC,IMLIMP, MCA14P%A, &
                      IGMC,NGMC, MCGTTP%A, SPCX,SPCY,SPCZ,LGCMC, &
#if KEY_PERT==1
                      MCBLORP%A, MCBLOPP%A, BDUMMYR, BDUMMYP,  &
                      MCIMLRP%A, MCIMLPP%A, IMLIMRP, IMLIMPP,  &
                      BDIMMYR,BDIMMYP,    &
#endif
                      EGSBPN)

              ENDDO
            ENDDO
            
#if KEY_ACE==1 /* (samc_ace) */
            IF(LACEMC) THEN
                !evaluate the ACE contribution for all configurations of spatial averaging (cf. mcsamc.src)
                CALL SPACECONTRIBUTION(BDUMMY%INBLO,BDUMMY%JNB,IAMC,NAMC, &       !For GTESLF
                    ces1,ces2,sig2i,mue4,CSWIT,CTOFNB*CTOFNB,CTONNB*CTONNB, &    !For GTESLF
                    IBCUTP,NCUTBC,ACECUT,IBCHGP,NACEBC, &                        !For ADESLF
                    ESARP,ETMPP,BSARP,CG2RP,RSYS,NOTADP, &                       !For MCACEE
                    MCBLOP%A,MCA14P%A,FACT2,FACT2H,ETURN,ETURNH,MEREF,MEREFH,NATOM)    !For MCACEE
            ENDIF
#endif /* (samc_ace) */

            CALL SPGETCRITERION(BETA,NATOM,LACEMC)
            LSPDONE=.TRUE.
          ENDIF
#endif /* (samc) */
      
      !         Apply the acceptance criterion.
      
      LACC=MCACPT(IACCPF,ISEED,BETA/TFACT(IMVTYP),DEPOT, &
            DEKIN,ETOT,ACPAR1,ACPAR2,ACPAR3,NMULT, &
            MLTLGP,(MCMINN(IMVTYP).LT.0))
            
      IF (QMOVE.AND.LACC) THEN

          !           -------------------------------------------------------------
          !           Begin acceptance
          !           Update the total energy and volume
          ETOT = ETOT + DEPOT
          IF (LVOL) RVOLMC = RS*RS*RS*RVOLMC
#if KEY_GCMC==1
          EGSBPS = EGSBPN 
#endif

          !           -------------------------------------------------------------
          !           Begin second move linking loop
          !           If optimization is allowed, update it.
          JMVTYP = IMVTYP
20        IF ((IARMF .GT. 0) .OR. (IDOMCF .GT. 0)) &
              MCACCP(JMVTYP)%A(IDX) = MCACCP(JMVTYP)%A(IDX) + 1
          NMVACC(JMVTYP) = NMVACC(JMVTYP) + 1
          IF (NXTMVG(JMVTYP).GT.0) THEN
            JMVTYP = NXTMVG(JMVTYP)
            GOTO 20
          ENDIF
          !           End second move linking loop
          !           -------------------------------------------------------------
#if KEY_ACE==1
          !           -------------------------------------------------------------
          !           Begin ACE coordinate update
          IF (LACEMC) THEN
            CALL RESTOR(IDX, ILNMVP(IMVTYP)%A, IOL2CP, XOL2, &
                  X,Y,Z,MCMINN(IMVTYP), LGCMC)
            CALL CNTTRN(XCENT,YCENT,ZCENT,LGROUP,NTRANS,IGRPP,IMAGEP, &
                  MCATTP%A, IMTRNS,NOROT,BIMAG, MCGTTP%A, IGPBS, &
                  X,Y,Z &
#if KEY_GCMC==1
                  ,GCMCON        & 
#endif
                  )

            !             Correct the ACE self energies
            CALL ADESLF(IBCUTP, NCUTBC, ACECUT, &
                  ESR1P, ESR2P, IBCHGP, &
                  NACEBC, 0)
            DO I = 1, NACEBC
                J = IBCHGP(I)
                CALL ACELP2(EDUMMY,EDUMMY,J, BSARP, ETMPP, &
                    ESARP, CG2RP, RSYS,ESMAX,FACT1, &
                    khyd,esii,FACT2,FACT2H,ETURN, &
                    ETURNH,MEREF,MEREFH)
            ENDDO
          ELSE IF (LACE) THEN
            !             Ace but minimization was done.
            CALL ACUPDT(RSYS,ESMAX,FACT1, &
                  X,Y,Z,NATOM,BNBND,FACT2,FACT2H,ETURN,ETURNH, &
                  MEREF,MEREFH,CSWIT)
          ENDIF
          !           End ACE coordinate update
          !           -------------------------------------------------------------
#endif 

#if KEY_GCMC==1 /*gcmc_accept*/
          IF (LIMVGC) THEN
            !             If it is a GC move, accept and update the grid
            CALL ACGCMC(MCBLOP%A, ILNMVP(IMVTYP)%A, NTRANS, MCIMLP%A, &
                  MCATTP%A, IDX,GCMCON,LGCOLD,NMVATM(IMVTYP), &
                  NIDXGC(IMVTYP), IDXGCP(IMVTYP)%A, QGCGRD, &
                  NGRDX,NGRDY,NGRDZ,NGCBLK,LSTGRD,LSTIND, &
                  NGRDCV,NRGRID,GRDSIZ,XGCMIN,YGCMIN,ZGCMIN, &
                  NGCIN(IMVTYP),X,Y,Z &
#if KEY_PERT==1
                  ,MCBLORP%A, MCBLOPP%A, QPERT  & 
#endif
                  )
          ELSE IF (LGCMC) THEN
            !             If GC is active, it is still necessary to update appropriate GC
            !             counters when a move type other than GC is made
            IF (MCMINN(IMVTYP) .NE. 0) THEN
                !               A global move (e.g., HMC) was made.  Force a complete update.
                CALL GCMCUP(NMVTYP,MVTYPE,QGCGRD,NRGRID,GCMCON, &
                    XGCMIN,YGCMIN,ZGCMIN,XGCMAX,YGCMAX,ZGCMAX, &
                    NATOM,NATIM,NTRANS,LIMALL,bimag%IMATTR,NPSUM, &
                    GCSUM,GC2SUM,NGCIN,NIDXGC,QGCSPH,IDXGCP, &
                    IMVNGP,NGCTRY,GCBLKR,NGCBLK,NGRDBK,LSTGRD, &
                    LSTIND,NGDTOT,NGRDCV,GRDBLK,GRDSIZ,NGRDX, &
                    NGRDY,NGRDZ,X,Y,Z)
            ELSE
                IF (QGCGRD) THEN
                  !                 Update the grid
                  !                 Use of IOL1CP assumes no minimization or ACE for GCMC
                  CALL NOGCMV(NGCBLK, LSTGRD, LSTIND, &
                        NGRDCV, ILNMVP(IMVTYP)%A, IDX,NGRDX, &
                        NGRDY,NGRDZ,NRGRID,GRDSIZ,XGCMIN,YGCMIN, &
                        ZGCMIN, IOL1CP, X,Y,Z)
                ENDIF
                !               If a GCMC molecule moves, update number in insertion volume
                IF (LGCMC .AND. IGCMVG(IMVTYP).GT.0) THEN
                  CALL NOGCIN(NGCIN(IGCMVG(IMVTYP)),LGCIN,GCMCON, &
                        IMVNGP(IMVTYP)%A, IDX,QGCSPH, &
                        XGCMIN,YGCMIN,ZGCMIN,XGCMAX,YGCMAX,ZGCMAX, &
                        X,Y,Z)
                ENDIF
            ENDIF
          ENDIF
#endif /*   (gcmc_accept)*/

          !           ARD and Ao Ma 06-06-30
          !           Add to histogram for Wang-Landau sampling
          IF (IACCPF .EQ. 3) THEN
            CALL WLADD(IWLBN2, IWLHST, RWLLGF, RWLADD)
            IWLBN1 = IWLBN2
          ENDIF

          IF (INBFMC .LT. 0) THEN
            IF (MOD(ISTEP,-INBFMC).EQ.0) THEN
                QHEUR = MCHEUR(MVTYPE(IMVTYP),IDX, ILNMVP(IMVTYP)%A, &
                    IOL1CP, XOL1,X,Y,Z,MCMINN(IMVTYP), &
                     CUTHR2, ISTEP)
            ENDIF
          ENDIF

          !           End acceptance
          !           -------------------------------------------------------------

      ELSE

          !           -------------------------------------------------------------
          !           Begin rejection
          
          !           Change the coordinates and images back
          IF (.NOT. LACEMC .AND. QMOVE) THEN
            CALL RESTOR(IDX, ILNMVP(IMVTYP)%A, IOL1CP, XOL1, &
                  X,Y,Z,MCMINN(IMVTYP), LGCMC)
            IF (MCMINN(IMVTYP) .GT. 0) THEN
                IGRPP => ALLGRP%A
                IMAGEP => ALLMVP%A
            ENDIF
#if KEY_GCMC==1 /*gcmc_reject*/
            IF (LIMVGC) THEN
                CALL RJGCMC(GCMCON,IDX, IMVNGP(IMVTYP)%A, &
                    ILNMVP(IMVTYP)%A, LGCOLD, MCBLOP%A, NTRANS, &
                    MCIMLP%A, MCATTP%A, QGCGRD,NGRDX,NGRDY,NGRDZ, &
                    NGCBLK,LSTGRD,LSTIND,NGRDCV,NRGRID,GRDSIZ, &
                    XGCMIN,YGCMIN,ZGCMIN,X,Y,Z &
#if KEY_PERT==1
                     ,MCBLORP%A, MCBLOPP%A, QPERT  & 
#endif
                    )
            ENDIF
#endif /*   (gcmc_reject)*/
            IF (LVOL) CALL SCLIMG(ONE/RS,NTRANS,IMTRNS,IMXCEN,IMYCEN, &
                  IMZCEN,XDIM,XUCELL,XTLABC)
            CALL CNTTRN(XCENT,YCENT,ZCENT,LGROUP,NTRANS,IGRPP,IMAGEP, &
                  MCATTP%A, IMTRNS,NOROT,BIMAG, MCGTTP%A, IGPBS, &
                  X,Y,Z &
#if KEY_GCMC==1
                  ,GCMCON        & 
#endif
                  )

          ENDIF

#if KEY_MEHMC==1
          !           Average in zeros for the MEHMC bias vector if reject move
          IF (MCMINN(IMVTYP).LT.0 .AND. MCMTYP(IMVTYP).GT.0) THEN
            CALL MERJCT(VMEAVX,VMEAVY,VMEAVZ,VMEOLX,VMEOLY,VMEOLZ, &
                  -MCMINN(IMVTYP),RMCMNF(IMVTYP),NATOM)
          ENDIF
#endif 

          !           ARD and Ao Ma 06-06-30
          !           Add to histogram for Wang-Landau sampling
          IF (IACCPF .EQ. 3 .AND. QMOVE)  &
              CALL WLADD(IWLBN1, IWLHST, RWLLGF, RWLADD)

          !           End rejection
          !           -------------------------------------------------------------

      ENDIF
#if KEY_ACE==1
      IF (LACEMC) THEN
          !           Clear the arrays for the ACE self energy differences (Mode = 2)
          CALL ADESLF(IBCUTP, NCUTBC, ACECUT, &
              ESR1P, ESR2P, IBCHGP, NACEBC, 2)
      ENDIF
#endif 
      !         ---------------------------------------------------------------
      !         Begin third move linking loop
      !         If optimization is allowed, update it.
      JMVTYP = IMVTYP
      !         If optimization is allowed, update it.
      !         Only one type of optimization allowed at a time.
30     IF (IARMF .GT. 0) THEN
          CALL ARMUPT(IDX,IARMF,ARMP(JMVTYP), &
              ARMA(JMVTYP),ARMB(JMVTYP), &
              NTRYP(JMVTYP)%A, MCACCP(JMVTYP)%A, &
              ARMLIM(JMVTYP),ARMMAX(JMVTYP), &
              MDXP(JMVTYP)%A, ANISO(JMVTYP))
          CALL HMCARM(MVTYPE(JMVTYP), NTRYP(JMVTYP)%A, IDX, &
              IGAMMA(JMVTYP), MDXP(JMVTYP)%A(IDX), &
              X,Y,Z)
      ELSE IF (IDOMCF .GT. 0) THEN
          CALL DOMCUP(IDX,IDOMCF,ARMP(JMVTYP), &
              ARMA(JMVTYP),ARMB(JMVTYP), &
              NTRYP(JMVTYP)%A, MCACCP(JMVTYP)%A, &
              EAVEP(JMVTYP)%A, DAVEP(JMVTYP)%A, &
              ARMLIM(JMVTYP),ARMMAX(JMVTYP), &
              DX1(1,JMVTYP),DEPOT,DOMCF(JMVTYP),ANISO(JMVTYP), &
              BETA, MDXP(JMVTYP)%A)
          CALL HMCARM(MVTYPE(JMVTYP), NTRYP(JMVTYP)%A, IDX, &
              IGAMMA(JMVTYP), MDXP(JMVTYP)%A(IDX), &
              X,Y,Z)
      ENDIF
      IF (NXTMVG(JMVTYP).GT.0) THEN
          JMVTYP = NXTMVG(JMVTYP)
          GOTO 30
      ENDIF
      !         End third move linking loop
      !         ---------------------------------------------------------------

#if KEY_GCMC==1 /*gcmc_stat*/
      !         Collect GCMC statistics.
438    CONTINUE
      IF (LGCMC) THEN
          CALL GCSTAT(NMVTYP,MVTYPE,NGCTRY,QGCGRD,NIDXGC,NGRDCV, &
              NOCVTY,NGCPCP,ISTEP,IEFRQ,NPSUM,GCSUM,GC2SUM, &
              NGCIN)
      ENDIF
#endif /*   (gcmc_stat)*/

      !         Write out any information necessary.
      CALL MCWRCD(IOAVES,IUNWRI,IUNCRD,IOENGY,NSAVC,ISVFRQ, &
            TITLEA,NTITLA,IFRATP,NFREAT,ISTEP,NSTEPS, &
            TKELV,ETOT,ISEED,X,Y,Z &
#if KEY_GCMC==1
            ,GCMCON     & 
#endif
            )

      !         ARD and Ao Ma 06-06-30
      IF (NWLFRQ .GT. 0) THEN
          IF (MOD(ISTEP,NWLFRQ) .EQ. 0) THEN
            CALL WLCHK(IUNWLW, NRXNCR, NWLBIN, IWLHST, RWLLGF, &
                  NRXSTT, FLTNES, RWLADD)
            !             Terminate if tolerance reached
            IF (RWLADD .LT. WLTOL) GOTO 999
          ENDIF
      ENDIF
   
#if KEY_SAMC==1
     ! if we need to extract thermodynamical data for trajectories we have to write 
     ! data for unbiasing in a specific file.
     ! frequency of writing is the same as coordinates saving
      IF ( (NSAVC.GT.0) .AND. (MOD(ISTEP,NSAVC) .EQ. 0) .AND. LUNB .AND. LSAMC ) THEN
        IF (LSPDONE .AND. LSPGROUP(IMVTYP)) THEN
            UNB = EXP(-BETA*(DEPOT+DEKIN))/EXP(-BETA*UNB)
        ELSE
            UNB = ONE
        ENDIF
        IF(MYNOD .EQ. 0) THEN
            WRITE(UNBIOFILE,'(i10,3x,e15.7)') ISTEP,UNB
        ENDIF
      ENDIF
#endif

    ENDDO
    !
    !       End main loop
    !       -----------------------------------------------------------------
999 CONTINUE
    !       -----------------------------------------------------------------
    !       Free arrays allocated in MCLOOP

    IF (NTRANS .GT. 0) THEN
      call chmdealloc('mc.src','MCLOOP','IMLIMP',2*NTRANS,intg=IMLIMP)
#if KEY_PERT==1
      IF (QPERT) THEN
          call chmdealloc('mc.src','MCLOOP','IMLIMRP',2*NTRANS,intg=IMLIMRP)
          call chmdealloc('mc.src','MCLOOP','IMLIMPP',2*NTRANS,intg=IMLIMPP)
      ENDIF
#endif 
    ENDIF

    !       Free the space allocated in MCUPDT
    CALL FRUPDT(MCBLOP,MCBLGP,MCIMLP,MCIMGP,MCATTP, &
        BDUMMY,BDIMMY,NATOM,NGRP, &
        MCGTTP &
#if KEY_GCMC==1
         ,LGCMC        & 
#endif
#if KEY_PERT==1
         ,MCBLORP,MCBLOPP,BDUMMYR,BDUMMYP  & 
#endif
#if KEY_PERT==1
         ,MCIMLRP,MCIMLPP,BDIMMYR,BDIMMYP  & 
#endif
        )

    call chmdealloc('mc.src','MCLOOP','IOL1CP',3*OLDALL,crl=IOL1CP)
#if KEY_ACE==1
    IF (LACE) THEN
      call chmdealloc('mc.src','MCLOOP','IOL2CP',3*OLDALL,crl=IOL2CP)
      CALL FREACE(NATOM, MCA14P%A)
    ENDIF
#endif 

#if KEY_GCMC==1
    IF (QGCGRD .OR. NGCTRY.GT.0) THEN
      call chmdealloc('mc.src','MCLOOP','GRDBLK',NGRDBK,intg=GRDBLK)
    ENDIF
    IF (QGCGRD) THEN
      NGRNO=NGRDX*NGRDY*NGRDZ
      call chmdealloc('mc.src','MCLOOP','NGCBLK',NGRNO,intg=NGCBLK)
      call chmdealloc('mc.src','MCLOOP','LSTGRD',NGRNO,intg=LSTGRD)
      call chmdealloc('mc.src','MCLOOP','LSTIND',NGRNO,intg=LSTIND)
    ENDIF
#endif 

    !       ARD and Ao Ma 06-06-30
    !       Free the histogram arrays for Wang-Landau sampling
    IF (IACCPF .EQ. 3) THEN
      call chmdealloc('mc.src','MCLOOP','IWLHST',NWLBIN,intg=IWLHST)
      call chmdealloc('mc.src','MCLOOP','RWLLGF',NWLBIN,crl=RWLLGF)
      CALL VCLOSE(IUNWLW,'WRITE',ERROR)
    ENDIF

    !       Turn the forces back on for other commands
#if KEY_BLOCK==1
    NOFORC = .FALSE.
#endif

#if KEY_SAMC==1 /* (samc) */
    IF (LSAMC) THEN
      CALL SPDEALLOCARRAYS(NATOM)
#if KEY_ACE==1 /* (ace_sp) */
      IF (LACE) THEN
        CALL SPACEDEALLOCARRAYS(NATOM)
      ENDIF
#endif /* (ace_sp) */
    ENDIF
#endif /* (samc) */
  
    RETURN
  END SUBROUTINE MCLOOP

  SUBROUTINE MCPICK(IMVTYP,IDX,PICK,IMCURR,NACMVG,IACMVG,NMVTYP, &
      NMVATM,WTOT,WEIGHT,ISEED &
#if KEY_GCMC==1
       ,MVTYPE,IGCMVG,IDXGCP,NIDXGC  & 
#endif
      )
    !
    !       Pick the next move instance to try (random or in sequence).
    !
    !       Aaron R. Dinner 00-04-13
    !
    use clcg_mod,only:random
    use chm_kinds
    use chm_types
    use number
    use stream
    use mcmvutil, only: r2indx
    implicit none
    INTEGER IMVTYP, IDX, PICK, NMVTYP, ISEED, IMCURR
    INTEGER NMVATM(NMVTYP), NACMVG, IACMVG(NMVTYP)
    real(chm_real)  WEIGHT(NMVTYP), WTOT
#if KEY_GCMC==1
    INTEGER MVTYPE(NMVTYP), IGCMVG(NMVTYP)
    type(chm_iptr) :: IDXGCP(NMVTYP)
    INTEGER NIDXGC(NMVTYP)
#endif 
    !
    INTEGER N
#if KEY_GCMC==1
    INTEGER I
    LOGICAL LGCIN
    !
    LOGICAL GCINF
#endif 

    !       Random
    IF (PICK .EQ. 0) THEN

10     IMCURR = R2INDX(WTOT*RANDOM(ISEED),WEIGHT,NACMVG)
      IMVTYP = IACMVG(IMCURR)

      !         Pick which site from that move set
#if KEY_GCMC==1 /*gcmc_pick1*/
      !         GCMC move
      IF (MVTYPE(IMVTYP).EQ.8) THEN

          N = NIDXGC(IMVTYP)
          !           If we have hit one of the limits, all are the same.
          IF (N .EQ. 0 .OR.  N .EQ. NMVATM(IMVTYP)) THEN
             if(prnlev>2) CALL WRNDIE(2,'<MCPICK>','REACHED GCMC LIMIT')
            IDX = DBLE(NMVATM(IMVTYP))*RANDOM(ISEED) + ONE
          ELSE IF (RANDOM(ISEED) .GT. HALF) THEN
            !             Deletion
            IDX = DBLE(N)*RANDOM(ISEED) + ONE
          ELSE
            !             Insertion
            IDX = N + DBLE(NMVATM(IMVTYP)-N)*RANDOM(ISEED) + ONE
          ENDIF
          IDX = IDXGCP(IMVTYP)%A(IDX)

          !         Spatial move of a GCMC particle
      ELSE IF (IGCMVG(IMVTYP).GT.0) THEN

          IF (NIDXGC(IGCMVG(IMVTYP)) .GT. 0) THEN
            IDX = DBLE(NIDXGC(IGCMVG(IMVTYP)))*RANDOM(ISEED) + ONE
            IDX = IDXGCP(IGCMVG(IMVTYP))%A(IDX)
          ELSE
            GOTO 10
          ENDIF

          !         The default case (no GCMC relevance)
      ELSE
#endif /*   (gcmc_pick1)*/
          IDX = DBLE(NMVATM(IMVTYP))*RANDOM(ISEED) + ONE
#if KEY_GCMC==1 /*gcmc_pick2*/
      ENDIF
#endif /*   (gcmc_pick2)*/

      !       Sequential
    ELSE IF (PICK .EQ. 1) THEN
      IDX = IDX + 1
      IF (IDX .GT. NMVATM(IMVTYP)) THEN
          IMCURR = IMCURR + 1
          IF (IMCURR .GT. NACMVG) IMCURR = 1
          IMVTYP = IACMVG(IMCURR)
          IDX = 1
      ENDIF

      !       Random group and sequential instances
    ELSE IF (PICK .EQ. 2) THEN
      IDX = IDX + 1
      IF (IDX .GT. NMVATM(IMVTYP)) THEN
          IMCURR = R2INDX(WTOT*RANDOM(ISEED),WEIGHT,NACMVG)
          IMVTYP = IACMVG(IMCURR)
          IDX = 1
      ENDIF

    ELSE
      CALL WRNDIE(-5,'<MCPICK>','UNKNOWN PICK TYPE')

    ENDIF

    RETURN
  END SUBROUTINE MCPICK

  SUBROUTINE RESTOR(IDX,IMVNG,CRDOLD,XOLD,X,Y,Z,MCMINN, &
      LGCMC)
    !
    !       Restores the old coordinates of the moving atoms.
    !       Treats single and multiple atom moves differently.
    !
    !       Aaron R. Dinner
    !
    use chm_kinds
    use chm_types
    use mcmvad, only: single
    implicit none
    !
    !       Passed Variables
    !
    INTEGER IDX
    type(chm_iptr) :: IMVNG(:)
    INTEGER MCMINN
    real(chm_real)  XOLD(3), X(*), Y(*), Z(*), CRDOLD(*)
    LOGICAL LGCMC
    !
    !       Local Variables
    !
    INTEGER I, J, K, N, IAF, IAL, IATOM, NG
    integer,pointer,dimension(:) :: TEMPP
    LOGICAL LMINI

    logical :: is_single_step

    LMINI = MCMINN .GT. 0
    TEMPP => null()

    is_single_step = single(imvng(idx)%a)
    if (is_single_step .and. .not. (lmini .or. lgcmc)) then
      IATOM = IMVNG(IDX)%A(3)
      X(IATOM) = XOLD(1)
      Y(IATOM) = XOLD(2)
      Z(IATOM) = XOLD(3)
    ELSE

      IF (LMINI) THEN
          TEMPP => ALLMVP%A
      ELSE
          TEMPP => IMVNG(IDX)%A
      ENDIF

      NG = TEMPP(1)
      N = TEMPP(NG)
      NG = NG + 2
      J = 0
      DO K = NG, N, 2
          IAF = TEMPP(K-1)
          IAL = TEMPP(K)
          DO I = IAF, IAL
            J = J + 1
            X(I) = CRDOLD(J)
            J = J + 1
            Y(I) = CRDOLD(J)
            J = J + 1
            Z(I) = CRDOLD(J)
          ENDDO
      ENDDO
    ENDIF
    RETURN
  END SUBROUTINE RESTOR

  SUBROUTINE SVOLDC(IDX, IMVNG, CRDOLD, XOLD, X, Y, Z, &
      MCMINN, LGCMC)
    !
    !       Saves the old coordinates of the moving atoms.
    !
    !       Aaron R. Dinner
    !
    use chm_kinds
    use chm_types
    use mcmvad, only: single
    implicit none
    !
    !       Passed Variables
    !
    INTEGER IDX
    type(chm_iptr) :: IMVNG(:)
    INTEGER MCMINN
    real(chm_real)  XOLD(3), X(*), Y(*), Z(*), CRDOLD(*)
    LOGICAL LGCMC
    !
    !       Local Variables
    !
    INTEGER I, J, K, N, IAF, IAL, IATOM, NG
    integer,pointer,dimension(:) :: TEMPP
    LOGICAL LMINI

    logical :: is_single_step

    LMINI = MCMINN .GT. 0
    TEMPP => null()

    !       If it is a single atom move, do not bother with dynamic memory
    !
    is_single_step = single(imvng(idx)%a)
    if (is_single_step .and. .not. (lmini .or. lgcmc)) then
      IATOM = IMVNG(IDX)%A(3)
      XOLD(1) = X(IATOM)
      XOLD(2) = Y(IATOM)
      XOLD(3) = Z(IATOM)
    ELSE

      IF (LMINI) THEN
          TEMPP => ALLMVP%A
      ELSE
          TEMPP => IMVNG(IDX)%A
      ENDIF

      NG = TEMPP(1)
      N = TEMPP(NG)
      NG = NG + 2
      J = 0
      DO K = NG, N, 2
          IAF = TEMPP(K-1)
          IAL = TEMPP(K)
          DO I = IAF, IAL
            J = J + 1
            CRDOLD(J) = X(I)
            J = J + 1
            CRDOLD(J) = Y(I)
            J = J + 1
            CRDOLD(J) = Z(I)
          ENDDO
      ENDDO
    ENDIF
    RETURN
  END SUBROUTINE SVOLDC

  SUBROUTINE GTOLDC(MTOT,NMVTYP,IMVNG,MCMINN,NMVATM)
    !
    !       Allocate space for storing old coordinates.
    !
    !       Aaron R. Dinner
    !
    use chm_types
    implicit none
    !
    !       Passed Variables
    !
    type(iptr_ptr) :: IMVNG(:)
    INTEGER NMVTYP, MCMINN(*)
    INTEGER NMVATM(*), MTOT
    !
    !       Local Variables
    !
    INTEGER I, J, K, N, IAF, IAL, IM, NG, NTOT
    integer,pointer,dimension(:) :: TEMPP
    LOGICAL LMINI

    MTOT = 0
    TEMPP => null()

    DO IM = 1, NMVTYP
      LMINI = (MCMINN(IM) .GT. 0)
      DO I = 1, NMVATM(IM)

          IF (LMINI) THEN
            TEMPP => ALLMVP%A
          ELSE
            TEMPP => IMVNG(IM)%A(I)%A
          ENDIF

          NTOT = NIMVGF(TEMPP)
          IF (NTOT .GT. MTOT) MTOT = NTOT

      ENDDO
    ENDDO

    RETURN
  END SUBROUTINE GTOLDC

  INTEGER FUNCTION NIMVGF(IMVNG)
    !
    !       Count how many atoms move in a single move instance.
    !
    use chm_kinds
    implicit none
    !
    INTEGER IMVNG(*)
    !
    INTEGER NG, N, K, IAF, IAL

    NIMVGF = 0
    NG = IMVNG(1)
    N =  IMVNG(NG)
    NG = NG + 2
    DO K = NG, N, 2
      IAF = IMVNG(K-1)
      IAL = IMVNG(K)
      NIMVGF = NIMVGF + IAL - IAF + 1
    ENDDO

    RETURN
  END FUNCTION NIMVGF

  SUBROUTINE CNTTRN(XCENT,YCENT,ZCENT,LGROUP,NTRANS,IGRPP,IMAGEP, &
      MCATTP,IMTRNS,NOROT,BIMAG,MCGTTP,IGPBS, &
      X,Y,Z &
#if KEY_GCMC==1
       ,GCMCON        & 
#endif
      )
    !
    !       Updates group centers and images after moving primary atoms.
    !
    use chm_kinds
    use chm_types
    use mcimg, only: mcitrn, mcicnt
    implicit none
    !
    real(chm_real),dimension(:) :: XCENT, YCENT, ZCENT
    INTEGER NTRANS
    type(chm_iptr),dimension(:) :: MCATTP
    integer,dimension(:) :: IGRPP, IMAGEP
    real(chm_real)  IMTRNS(*)
    type(imageDataStructure) BIMAG
    type(chm_iptr),dimension(:) :: MCGTTP
    INTEGER IGPBS(*)
    real(chm_real)  X(*), Y(*), Z(*)
    LOGICAL LGROUP, NOROT
#if KEY_GCMC==1
    LOGICAL GCMCON(:)            
#endif
    !
    !       If necessary, update group centers of moving atoms
    IF (LGROUP) THEN
      CALL CNTMVG(IGRPP, X,Y,Z, &
            xcent,ycent,zcent)
    ENDIF
    !
    !       If necessary, update images of moving atoms
    IF (NTRANS .GT. 0) THEN
      CALL MCITRN(X,Y,Z, IMAGEP, MCATTP, NTRANS,IMTRNS, &
            NOROT,bimag%IMATPT &
#if KEY_GCMC==1
            ,GCMCON          & 
#endif
            )
      !         Update centers of group images
      IF (LGROUP) THEN
          CALL MCICNT(xcent,ycent,zcent, &
              IGRPP, MCGTTP, IGPBS, NTRANS, &
              NOROT,bimag%IMATPT,IMTRNS,X,Y,Z)
      ENDIF
    ENDIF
    RETURN
  END SUBROUTINE CNTTRN

  LOGICAL FUNCTION MCHEUR(MT,IDX,IMVNG,CRDOLD,XOLD,X,Y,Z,MCMINN, &
      CUT2,ISTEP)
    !
    !       Applies a heuristic criterion to see if the non-bond list
    !       should be updated at the start of the next step.
    !
    !       Treats single and multiple atom moves differently.
    !
    !       Aaron R. Dinner
    !
    use chm_kinds
    use chm_types
    use mcmvad, only: single
    use stream, only: outu, prnlev
    use number, only: zero
    
    implicit none
    !
    !       Passed Variables
    !
    INTEGER MT, IDX
    type(chm_iptr) :: IMVNG(:)
    INTEGER MCMINN,ISTEP
    real(chm_real) XOLD(3), X(*), Y(*), Z(*), CUT2, CRDOLD(*)
    !
    !       Local Variables
    !
    INTEGER I, J, K, N, IAF, IAL, IATOM, NG
    integer,pointer,dimension(:) :: TEMPP
    real(chm_real)  DXH, DYH, DZH, D2
    LOGICAL LMINI

    logical :: is_single_step

    MCHEUR = .FALSE.

    LMINI = MCMINN .GT. 0
    TEMPP => null()

    is_single_step = single(imvng(idx)%a)
    if (is_single_step .and. .not. lmini) then
      IATOM = IMVNG(IDX)%A(3)
      DXH = X(IATOM) - XOLD(1)
      DYH = Y(IATOM) - XOLD(2)
      DZH = Z(IATOM) - XOLD(3)

      D2 = DXH*DXH + DYH*DYH + DZH*DZH
      CUMULHEUR(IATOM) = CUMULHEUR(IATOM) + D2
      IF (CUMULHEUR(IATOM) .GT. CUT2) THEN
          MCHEUR = .TRUE.
          CUMULHEUR = ZERO
           IF(PRNLEV .GE. 2) WRITE(OUTU,'(a,i10,a,i10)') " MCHEUR> Heuristic update for atom ",IATOM," at step ",ISTEP
          RETURN
      ENDIF

    ELSE

      IF (LMINI) THEN
          TEMPP => ALLMVP%A
      ELSE
          TEMPP => IMVNG(IDX)%A
      ENDIF

      NG = TEMPP(1)
      N = TEMPP(NG)
      NG = NG + 2
      J = 0
      DO K = NG, N, 2
          IAF = TEMPP(K-1)
          IAL = TEMPP(K)
          DO I = IAF, IAL
            J = J + 1
            DXH = X(I) - CRDOLD(J)
            J = J + 1
            DYH = Y(I) - CRDOLD(J)
            J = J + 1
            DZH = Z(I) - CRDOLD(J)

            D2 = DXH*DXH + DYH*DYH + DZH*DZH
            CUMULHEUR(IATOM) = CUMULHEUR(IATOM) + D2
            IF (CUMULHEUR(IATOM) .GT. CUT2) THEN
                MCHEUR = .TRUE.
                CUMULHEUR = ZERO
                 IF(PRNLEV .GE. 2) WRITE(OUTU,'(a,i10,a,i10)') " MCHEUR> Heuristic update for atom ",IATOM," at step ",ISTEP
                RETURN
            ENDIF

          ENDDO
      ENDDO
    ENDIF
    RETURN
  END FUNCTION MCHEUR

  SUBROUTINE MKMOVE(ISEED,MVTYPE,IDX,IATOM,IPIVTP,IMVNGP, &
      MDXP,ANISO,DX1,QMOVE,NLIMIT,NSCALE,X,Y,Z &
#if KEY_GCMC==1
      ,GCMCON,NGCTRY,NGCIN,ISCVTY,NGCPCP,GCCUT2, &
      XGCMIN,YGCMIN,ZGCMIN,XGCMAX,YGCMAX,ZGCMAX, &
        LGCCB,MCBLOP,CUTNB2,QGCGRD,QGCSPH,GRDBLK, &
        NGRDBK,NGRDCV,LSTGRD,GRDSIZ,NGRDX,NGRDY,NGRDZ, &
        NGCBLK &
#endif 
        )
    !
    !       Calls the appropriate routine to make the move.
    !
    !       Aaron R. Dinner
    !
    use chm_kinds
    use chm_types
    use clcg_mod,only:random
    use dimens_fcm
    use consta
    use image
    use corsubs,only:fndu
    use mcmvcrot, only: mkcrot
#if KEY_GCMC==1
    use mcmvgcmc, only: mkgcmc  
#endif
    use mcmvrtrn

    implicit none
    !
    !       Passed Variables
    !
    INTEGER ISEED, MVTYPE,IDX,IATOM
    type(iptr_ptr) :: IPIVTP, IMVNGP
    INTEGER NLIMIT, NSCALE
    type(chm_ptr) :: MDXP
    real(chm_real) DX1(3), X(*), Y(*), Z(*)
    LOGICAL ANISO, QMOVE
#if KEY_GCMC==1
    INTEGER NGCTRY, NGCIN, ISCVTY(:), NGCPCP(:)
    type(iptr_ptr) :: MCBLOP
    integer,dimension(:) :: GRDBLK, LSTGRD, NGCBLK
    INTEGER NGRDBK, NGRDCV, NGRDX, NGRDY, NGRDZ
    real(chm_real)  GCCUT2,XGCMIN,YGCMIN,ZGCMIN, &
        XGCMAX,YGCMAX,ZGCMAX,CUTNB2
    real(chm_real)  GRDSIZ
    LOGICAL LGCCB, GCMCON(:), QGCSPH, QGCGRD
#endif 
    !
    !       Local Variables
    !
    INTEGER I, J, N, IAF, IAL, IPVT1, IPVT2
    integer,pointer,dimension(:) :: TEMPP
    real(chm_real) U(9), RN(3), RS
    LOGICAL LOK, QMVLOC

    QMVLOC = .TRUE.
    TEMPP => null()

    IF (MVTYPE .EQ. 1) THEN

      ! ------- Rigid body translation
      CALL RUSPHR(ISEED,RN)

      if (.not. associated(MDXP%A)) then
          call WRNDIE(-5, '<MKMOVE>', 'MDXP not allocated!')
          return
      endif

      CALL SCLVEC(ANISO, RN, MDXP%A, IDX, DX1)
      TEMPP => IMVNGP%A(IDX)%A
      CALL TRNALL(X,Y,Z, TEMPP, DX1(1),DX1(2),DX1(3))
      IF (.NOT. ANISO) DX1(1) = &
            SQRT(DX1(1)*DX1(1) + DX1(2)*DX1(2) + DX1(3)*DX1(3))

    ELSE IF (MVTYPE .EQ. 2) THEN
      ! ------- Rigid body rotation
      TEMPP => IMVNGP%A(IDX)%A
      CALL MKRROT(DX1(1),X,Y,Z, TEMPP, IPIVTP%A(IDX)%A(1), &
            MDXP%A(IDX), ISEED)


    ELSE IF (MVTYPE .EQ. 3) THEN
      ! ------- Single cartesian atom move
      CALL RUSPHR(ISEED,RN)
      CALL SCLVEC(ANISO, RN, MDXP%A, IDX, DX1)
      TEMPP => IMVNGP%A(IDX)%A
      IATOM = TEMPP(3)
      CALL RIGTRN(X,Y,Z,IATOM,IATOM,DX1(1),DX1(2),DX1(3))
      IF (.NOT. ANISO) DX1(1) = &
            SQRT(DX1(1)*DX1(1) + DX1(2)*DX1(2) + DX1(3)*DX1(3))

    ELSE IF (MVTYPE .EQ. 4) THEN
      ! ------- Torsion
      TEMPP => IPIVTP%A(IDX)%A
      IPVT1 = TEMPP(1)
      IPVT2 = TEMPP(2)
      RN(1) = X(IPVT2) - X(IPVT1)
      RN(2) = Y(IPVT2) - Y(IPVT1)
      RN(3) = Z(IPVT2) - Z(IPVT1)
      DX1(1) = (2.0*RANDOM(ISEED) - 1.0) * MDXP%A(IDX)
      CALL FNDU(U,RN,DX1(1),LOK)

      TEMPP => IMVNGP%A(IDX)%A
      N = TEMPP(2)
      DO I = 4, N, 2
          IAF = TEMPP(I - 1)
          IAL = TEMPP(I)
          DO J=IAF, IAL
            CALL APPLYU(U,X,Y,Z,J,X(IPVT2),Y(IPVT2),Z(IPVT2))
          ENDDO
      ENDDO

    ELSE IF (MVTYPE .EQ. 5) THEN
      ! ------- Concerted Dihedral Rotations
      CALL MKCROT(DX1(1),QMVLOC,ISEED,IDX, IPIVTP%A, IMVNGP%A, &
            MDXP%A, NLIMIT,X,Y,Z)

    ELSE IF (MVTYPE .EQ. 6) THEN
      ! ------- Hybrid Monte Carlo
      DX1(1) = MDXP%A(IDX)

    ELSE IF (MVTYPE .EQ. 7) THEN

      ! ------- Volume change for constant pressure
      IF (NSCALE .EQ. 0) THEN
          DX1(1) = (2.0*RANDOM(ISEED) - 1.0) * MDXP%A(IDX)
          RS = EXP(DX1(1))
      ENDIF
      TEMPP => IMVNGP%A(IDX)%A
      if (associated(TEMPP)) then
          CALL MKVOLU(NSCALE,RS, IPIVTP%A(1)%A(1), TEMPP, NTRANS,IMTRNS, &
              IMXCEN,IMYCEN,IMZCEN,XDIM,XUCELL,XTLABC,X,Y,Z)
      endif

#if KEY_GCMC==1 /*gcmc_move*/
    ELSE IF (MVTYPE .EQ. 8) THEN
      ! ------- Grand canonical insertion or deletion
      TEMPP => IMVNGP%A(IDX)%A
      CALL MKGCMC(GCMCON,LGCCB,ISEED, TEMPP, NGCIN,NGCTRY, &
            GCCUT2,XGCMIN,YGCMIN,ZGCMIN,XGCMAX,YGCMAX,ZGCMAX, &
            ISCVTY,NGCPCP,X,Y,Z, CUTNB2,QGCGRD, &
            QGCSPH,GRDBLK,NGRDBK,NGRDCV,LSTGRD,GRDSIZ, &
            NGRDX,NGRDY,NGRDZ,NGCBLK)
#endif /*   (gcmc_move)*/

    ELSE
      CALL WRNDIE(-5, &
            '<MKMOVE>','INTERNAL ERROR:  UNKNOWN MOVE TYPE')
    ENDIF

    IF (.NOT. QMOVE) QMOVE = QMVLOC

    RETURN
  END SUBROUTINE MKMOVE

  SUBROUTINE MCINIT(COMLYN,COMLEN,IACCPF,NSTEPS,NSAVC,ISVFRQ,TKELV, &
      ISEED,INBFMC,IEFRQ,IOAVES,IUNWRI,IUNREA,IUNCRD, &
      IOENGY,IARMF,IDOMCF,IMGFMC,IUMULT,ACPAR1,ACPAR2, &
      PICK,ACECUT,PRESMC,RVOLMC,QRSTRT, &
#if KEY_GCMC==1
      GCMCBF,NGCTRY,GCCUT2,XGCMIN,YGCMIN,ZGCMIN, &
      XGCMAX,YGCMAX,ZGCMAX,QGCGRD,QGCSPH,GRDSIZ, &
      NRGRID,NGRDX,NGRDY,NGRDZ,NCFB, &
#endif 
      NWLFRQ,IUNWLR,IUNWLW,RWLADD,WLTOL,FLTNES)
    !
    !       Parse the command MC command line and initialize variables.
    !
    !       Aaron R. Dinner
    !

    use clcg_mod,only:rngmodseeds
    use chm_kinds
    use exfunc
    use number
    use consta
    use parallel
#if KEY_PBEQ==1
    use pbeq, only : qgsbp,srdist 
#endif
    use rndnum
    use memory
#if KEY_GCMC==1
    use mcmvgcmc, only: dens2b  
#endif
    use string
#if KEY_SAMC==1
    use samc
#endif

    implicit none
    !
    !       Passed Variables
    !
    CHARACTER(len=*) COMLYN
    INTEGER COMLEN
    INTEGER IACCPF,NSAVC,ISVFRQ,ISEED,NSTEPS,INBFMC,IEFRQ
    INTEGER IOAVES,IUNWRI,IUNREA,IUNCRD,IOENGY
    INTEGER IARMF,IDOMCF, IMGFMC, IUMULT, PICK
    INTEGER NWLFRQ, IUNWLR, IUNWLW
    real(chm_real)  TKELV, PRESMC, RVOLMC, ACPAR1, ACPAR2, ACECUT
    real(chm_real)  RWLADD,WLTOL,FLTNES
    LOGICAL QRSTRT,qpresent
    integer,allocatable,dimension(:) :: lrngseeds
#if KEY_GCMC==1
    INTEGER NGCTRY,NCFB
    INTEGER NGRDX, NGRDY, NGRDZ, NRGRID
    real(chm_real)  GCCUT2, GCMCBF
    real(chm_real)  GRDSIZ
    real(chm_real)  XGCMIN, YGCMIN, ZGCMIN, XGCMAX, YGCMAX, ZGCMAX
    LOGICAL QGCGRD, QGCSPH
    !
    !       Local Variables
    !
    real(chm_real)  R, MUEX, DENS
#endif 

    IACCPF = GTRMI(COMLYN,COMLEN,'IACC', 0)
    PICK   = GTRMI(COMLYN,COMLEN,'PICK', 0)
    ISVFRQ = GTRMI(COMLYN,COMLEN,'ISVF', 0)

    NSTEPS = GTRMI(COMLYN,COMLEN,'NSTE', 0)
    NSAVC  = GTRMI(COMLYN,COMLEN,'NSAV', 0)
    !
    call chmalloc('mc.src','MCINIT','lrngseeds',Nrand,intg=lrngseeds)
    lrngseeds(1:nrand)=rngseeds(1:nrand)
    if(qoldrandom.or.qbrokenclcg)lrngseeds(1:nrand)=iseed
    call gtrmim(nrand,comlyn,comlen,'ISEE',lrngseeds,rngseeds,qpresent)
    call rngmodseeds(qpresent,iseed)
    call chmdealloc('mc.src','MCINIT','lrngseeds',Nrand,intg=lrngseeds)
    !

    !        ISEED  = GTRMI(COMLYN,COMLEN,'ISEE', ISEED)
    !        if (.not.qoldrng) then    !yw 05-Aug-2008
    !           CALL CLCGINIT(ISEED)
    !           ISEED=1
    !        endif
    INBFMC = GTRMI(COMLYN,COMLEN,'INBF', 0)
    IEFRQ  = GTRMI(COMLYN,COMLEN,'IECH', 0)
    IMGFMC = GTRMI(COMLYN,COMLEN,'IMGF', 0)
    IARMF  = GTRMI(COMLYN,COMLEN,'IARM', 0)
    IDOMCF = GTRMI(COMLYN,COMLEN,'IDOM', 0)
    TKELV  = 3.0D+02
    TKELV  = GTRMF(COMLYN,COMLEN,'TEMP', TKELV)

    !       Constant pressure parameters
    RVOLMC = GTRMF(COMLYN,COMLEN,'VOLU', RVOLMC)
    PRESMC = GTRMF(COMLYN,COMLEN,'PRES', ZERO)
    PRESMC = PRESMC*ATMOSP

    !       Tsallis parameters
    ACPAR1 = GTRMF(COMLYN,COMLEN,'EMIN', ZERO)
    ACPAR2 = ONE - GTRMF(COMLYN,COMLEN,'QTSA', ONE)
    IF ((IACCPF .EQ. 2) .AND. (ACPAR2 .EQ. ZERO)) THEN
      CALL WRNDIE(-2,'<MCINIT>','QTSAllis = 1 --> Metropolis')
      IACCPF = 0
    ENDIF

    !       Wang-Landau parameters
    IUNWLR  = GTRMI(COMLYN,COMLEN,'IWLR',-1)
    IUNWLW  = GTRMI(COMLYN,COMLEN,'IWLW',-1)
    NWLFRQ  = GTRMI(COMLYN,COMLEN,'NWLF',0)
    RWLADD  = GTRMF(COMLYN,COMLEN,'WLIN',ZERO)
    WLTOL   = GTRMF(COMLYN,COMLEN,'WLTO',ZERO)
    FLTNES  = GTRMF(COMLYN,COMLEN,'WLUP',ZERO)


    !       ACE cutoff to limit non-bonded calculations
    ACECUT = GTRMF(COMLYN,COMLEN,'ACEC', PT01)

    !       Check if it is a restart
    QRSTRT = INDXA(COMLYN, COMLEN, 'REST') .GT. 0

    !       Get the io-unit numbers
    !       Files must be opened beforehand with OPEN
    IUNCRD  = GTRMI(COMLYN,COMLEN,'IUNC',-1)
    IUMULT  = GTRMI(COMLYN,COMLEN,'IMUL',-1)
    IUNWRI  = GTRMI(COMLYN,COMLEN,'IUNW',-1)
    IUNREA  = GTRMI(COMLYN,COMLEN,'IUNR',-1)

    !       Reserved for future use
    IOAVES  = GTRMI(COMLYN,COMLEN,'IRUN',-1)
    IOENGY  = GTRMI(COMLYN,COMLEN,'IENE',-1)

#if KEY_GCMC==1 /*gcmc_read*/
    !       Grand canonical parameters

    !       Cavity bias parameters
    NGCTRY = GTRMI(COMLYN,COMLEN,'NGCT', 0)
    GCCUT2 = GTRMF(COMLYN,COMLEN,'GCCU', ZERO)

    !       Orientational bias parameters
    NCFB = GTRMI(COMLYN,COMLEN,'NOTB', 1)

    !       Grid-based insertion parameters
    GRDSIZ = GTRMF(COMLYN,COMLEN,'RGRI',-ONE)
    QGCGRD = GRDSIZ .GT. ZERO
    QGCSPH = INDXA(COMLYN,COMLEN,'INSP') .GT. 0

    !       Cavity-bias checks
    IF (NGCTRY.GT.0 .AND. QGCGRD) &
        CALL WRNDIE(-2,'<MCINIT>','NGCTRY > 0 AND RGRID > 0')
    IF ((NGCTRY.GT.0 .OR. QGCGRD) .AND. GCCUT2.LE.ZERO) &
        CALL WRNDIE(-2,'<MCINIT>','ZERO CUTOFF WITH CAVITY BIAS')

    !       GCCUT2 is not yet squared
    IF (QGCGRD) NRGRID=INT(GCCUT2/GRDSIZ+0.01)
    GCCUT2 = GCCUT2*GCCUT2

    IF (QGCSPH) THEN
      !         Use MIN arrays for scratch
      XGCMIN = GTRMF(COMLYN,COMLEN,'INSX',ZERO)
      YGCMIN = GTRMF(COMLYN,COMLEN,'INSY',ZERO)
      ZGCMIN = GTRMF(COMLYN,COMLEN,'INSZ',ZERO)
      R      = GTRMF(COMLYN,COMLEN,'INSR',ZERO)
#if KEY_PBEQ==1
      IF (QGSBP .AND. (R .GE. SRDIST)) CALL WRNDIE(-5,'<GCMC>', &
            'GCMC INSERTION RADIUS LARGER THAN INNER GSBP REGION')
#endif 
      XGCMAX = XGCMIN + R
      YGCMAX = YGCMIN + R
      ZGCMAX = ZGCMIN + R
      XGCMIN = XGCMIN - R
      YGCMIN = YGCMIN - R
      ZGCMIN = ZGCMIN - R
    ELSE
      XGCMIN = GTRMF(COMLYN,COMLEN,'XMIN', ZERO)
      YGCMIN = GTRMF(COMLYN,COMLEN,'YMIN', ZERO)
      ZGCMIN = GTRMF(COMLYN,COMLEN,'ZMIN', ZERO)
      XGCMAX = GTRMF(COMLYN,COMLEN,'XMAX', ZERO)
      YGCMAX = GTRMF(COMLYN,COMLEN,'YMAX', ZERO)
      ZGCMAX = GTRMF(COMLYN,COMLEN,'ZMAX', ZERO)
    ENDIF

    !       Set up grid variables
    IF (QGCGRD) THEN
      NGRDX = INT((XGCMAX-XGCMIN)/GRDSIZ+0.01)
      NGRDY = INT((YGCMAX-YGCMIN)/GRDSIZ+0.01)
      NGRDZ = INT((ZGCMAX-ZGCMIN)/GRDSIZ+0.01)
    ENDIF

    !       The B parameter can either be input directly
    GCMCBF = GTRMF(COMLYN,COMLEN,'GCBF', ZERO)
    !       Or calculated from MU and RHO
    MUEX  = GTRMF(COMLYN,COMLEN,'MUEX', ZERO)
    DENS  = GTRMF(COMLYN,COMLEN,'DENS', ZERO)

    IF (DENS .GT. ZERO) THEN
      GCMCBF = DENS2B(QGCSPH,QGCGRD,NGRDX,NGRDY,NGRDZ,GRDSIZ, &
            XGCMIN,YGCMIN,ZGCMIN,XGCMAX,YGCMAX,ZGCMAX, &
            TKELV,MUEX,DENS,R)
    ENDIF
#endif   /* (gcmc_read) */

#if KEY_SAMC==1 /* (mcsamc_read_params) */
    !       Spatial Averaging parameters
    LSAMC = INDXA(COMLYN,COMLEN,'SAMC') .GT. 0
    IF (LSAMC) THEN
      LUNB = INDXA(COMLYN,COMLEN,'UNB') .GT. 0
      IF (LUNB) THEN
        UNBIOFILE = GTRMI(COMLYN,COMLEN,'IUNB',-1)
      ENDIF
    ENDIF
#endif /* (mcsamc_read_params) */

    RETURN
  END SUBROUTINE MCINIT

  SUBROUTINE MCSTUP(NMVTYP,NMVATM,MVTYPE,IMVNGP, &
      IARMF, IDOMCF, &
      ANISO, WEIGHT,WTOT,WTMP,LCENTR, &
      MCMINN, MCA14P, &
      TKELV,NACMVG,IACMVG &
#if KEY_GCMC==1
      ,GCMCON,NIDXGC, QGCGRD,NGCTRY, &
      NIGCMC &
#endif 
      )
    !
    !       Dynamically allocates space for scratch arrays carried in MC
    !       and initializes them.    See MCCALL for an explanation of the
    !       arrays.
    !
    !       Aaron R. Dinner
    !
    use chm_kinds
    use chm_types
    use dimens_fcm
    use ace_module
    use bases_fcm
    use consta
    use coord
    use psf
    use inbnd
    use memory
    use number
    use stream
    use image
    use cstran_mod,only:mkfrat
    use mc, only: mdxp
    use mce, only: gtsymnb
#if KEY_GCMC==1
    use mcmvgcmc, only: frsind  
#endif
    use mcmvutil, only: imvlst

    implicit none
    !
    !       Passed Variables
    !
    INTEGER NMVTYP, NMVATM(NMVTYP), MVTYPE(NMVTYP)
    type(iptr_ptr) :: IMVNGP(NMVTYP)
    INTEGER IARMF
    INTEGER IDOMCF
    INTEGER MCMINN(NMVTYP), LCENTR
    type(iptr_ptr) :: MCA14P
    INTEGER NACMVG, IACMVG(NMVTYP)
    real(chm_real)  WEIGHT(NMVTYP), WTOT, WTMP(NMVTYP), TKELV
    LOGICAL ANISO(NMVTYP)
#if KEY_GCMC==1
    INTEGER NIDXGC(NMVTYP), NIGCMC(NMVTYP)
    INTEGER NGCTRY
    LOGICAL GCMCON(:), QGCGRD
#endif 
    !
    !       Local Variables
    !
    INTEGER I, J, K, L, N, IAF, IAL
    integer,allocatable,dimension(:) :: IMOVEP, IGROUP
    integer,pointer,dimension(:) :: TEMPP, IGP
    INTEGER NG
    LOGICAL LMINI, LHMC

    TEMPP => null()

    !       Get the total of the weights
    WTOT = 0.0
    DO I = 1, NACMVG
      WTMP(I) = WEIGHT(IACMVG(I))
      WTOT = WTOT + WTMP(I)
    ENDDO

    !       To set up the MC free atom list, go through the possible
    !       moves and tag atoms that are mobile.
    call chmalloc('mc.src','MCSTUP','IMOVEP',NATOM,intg=IMOVEP)
    IMOVEP(1:NATOM) = 1
    DO I = 1, NMVTYP
      DO J = 1, NMVATM(I)
          TEMPP => IMVNGP(I)%A(J)%A
          NG = TEMPP(1)
          N  = TEMPP(NG)
          NG = NG + 2
          DO K = NG, N, 2
            IAF = TEMPP(K-1)
            IAL = TEMPP(K)
            IMOVEP(IAF:IAL) = 0
          ENDDO
      ENDDO
    ENDDO

    !       Check if any of the moves involve minimization since then we
    !       also need to check the standard free atom list.
    LMINI = .FALSE.
    LHMC  = .FALSE.
    DO I = 1, NMVTYP
      IF (MCMINN(I) .GT. 0) LMINI = .TRUE.
      IF (MCMINN(I) .LT. 0) LHMC  = .TRUE.
    ENDDO

    !       Set up the MC free atom list.  If all the atoms are free,
    !       do not allocate the space, since it should never be used.
    NFREAT = 0
    DO I = 1, NATOM
      IF (LMINI .AND. (IMOVE(I) <= 0)) IMOVEP(I) = 0
      J = IMOVEP(I)
      IF (J .EQ. 0) NFREAT = NFREAT + 1
    ENDDO
    IF (NFREAT .LT. NATOM) THEN
      call chmalloc('mc.src','MCSTUP','IFRATP',NFREAT,intg=IFRATP)
      CALL MKFRAT(IMOVEP, NATOM, IFRATP, NFREAT)
    ENDIF
    call chmdealloc('mc.src','MCSTUP','IMOVEP',NATOM,intg=IMOVEP)

    IF (PRNLEV .GE. 2) WRITE (OUTU,'(A,1X,I7)') &
        ' MCSTUP> Number of free atoms =', NFREAT

    IF (LGROUP) THEN
      call chmalloc('mc.src','MCSTUP','IMVGRP',NGRP,intg=IMVGRP)
      IMVGRP(1:NGRP) = 0
    ENDIF

    !       Setup the ISKIP and IMVATP arrays for faster energy calculations
#if KEY_CMAP==1
    NSKIP = MAX(NBOND,NTHETA,NPHI,NIMPHI,NCRTERM)
#else /**/
    NSKIP = MAX(NBOND,NTHETA,NPHI,NIMPHI)
#endif 
    call chmalloc('mc.src','MCSTUP','ISKIP',NSKIP,intg=ISKIP)
    ISKIP(1:NSKIP) = 1
    call chmalloc('mc.src','MCSTUP','IMVATP',NATOM,intg=IMVATP)
    IMVATP(1:NATOM) = 0

    !
    !       If group non-bonded calculations are desired,
    !       setup up group lists for each move based on IMVNGP
    !
    IF (LGROUP) THEN
      call chmalloc('mc.src','MCSTUP','IGROUP',NGRP,intg=IGROUP)
      DO I = 1, NMVTYP
          !           Allocate space here for group storage
          allocate(IMCGRP(I)%A(NMVATM(I)))
          DO J = 1, NMVATM(I)
            TEMPP => IMVNGP(I)%A(J)%A
            IGP => MCA2GF(TEMPP, NGRP, IGPBS, &
                  IMVATP, IGROUP)
            IMCGRP(I)%A(J)%A => IGP
          ENDDO
      ENDDO
      call chmdealloc('mc.src','MCSTUP','IGROUP',NGRP,intg=IGROUP)

      !         For images, set up atom to group list for primary atoms
      IF (NTRANS > 0) THEN
          call chmalloc('mc.src','MCSTUP','IA2GP',NATOM,intg=IA2GP)
          DO I = 1, NGRP
            DO J = IGPBS(I)+1,IGPBS(I+1)
                IA2GP(J) = I
            ENDDO
          ENDDO
      ENDIF

      !         Turn off centering of groups in the energy routines
      LCENTR = 1

    ENDIF
    !
    !       If there are NOE constraints on the molecule, get a list of
    !       constraints that change for each move instance (analogous to
    !       the IBLSTP array but a different structure).
    !
    CALL MCNOEL(NMVTYP, NMVATM, IMVNGP)
    !
    !       If ARM move size optimization is allowed, set up the
    !       bookkeeping arrays.
    !       NTRYP counts the number of tries of that move.
    !       MCACCP counts the number of acceptances of that move.
    !
    IF ((IARMF .GT. 0) .OR. (IDOMCF .GT. 0)) THEN
      DO I = 1, NMVTYP
          call chmalloc('mc.src','MCSTUP','NTRYP(I)',NMVATM(I),intgp=NTRYP(I)%A)
          call chmalloc('mc.src','MCSTUP','MCACCP(I)',NMVATM(I),intgp=MCACCP(I)%A)
          NTRYP(I)%A(1:NMVATM(I)) = 0
          MCACCP(I)%A(1:NMVATM(I)) = 0
      ENDDO
    ENDIF

    IF (IDOMCF .GT. 0) THEN
      DO I = 1, NMVTYP
          IF (ANISO(I)) THEN
            N = 6*NMVATM(I)
            call chmalloc('mc.src','MCSTUP','EAVEP(I)',N,crlp=EAVEP(I)%A)
            EAVEP(I)%A(1:N) = ZERO
            N = 15*NMVATM(I)
            call chmalloc('mc.src','MCSTUP','DAVEP(I)',N,crlp=DAVEP(I)%A)
            DAVEP(I)%A(1:N) = ZERO
          ELSE
            call chmalloc('mc.src','MCSTUP','EAVEP(I)',NMVATM(I),crlp=EAVEP(I)%A)
            call chmalloc('mc.src','MCSTUP','DAVEP(I)',NMVATM(I),crlp=DAVEP(I)%A)
            EAVEP(I)%A(1:NMVATM(I)) = ZERO
            DAVEP(I)%A(1:NMVATM(I)) = ZERO
          ENDIF
      ENDDO
    ENDIF

    !       For minimization, set up lists that include all free atoms
    !       (and groups) in the same format as the moving atom lists.
    IF (LMINI .OR. LHMC) THEN
      CALL FLIP(IMOVE,NATOM)
      CALL IMVLST(ALLMVP, NATOM, IMOVE)
      CALL FLIP(IMOVE,NATOM)
      IF (LGROUP) THEN
          CALL FLIP(IMOVEG,NGRP)
          CALL IMVLST(ALLGRP, NGRP, IMOVEG)
          CALL FLIP(IMOVEG,NGRP)
      ENDIF
    ENDIF

    !       For dynamics integration in Hybrid MC set up the GAMMA arrays.
    !       A different one is used for each HMC call since they depend on
    !       the timestep.
    DO I = 1, NMVTYP
      IF (MCMINN(I) .LT. 0) THEN
          call chmalloc('mc.src','MCSTUP','IGAMMA(I)',4*NATOM,crlp=IGAMMA(I)%A)
          CALL LNGFIL(0,0, IGAMMA(I)%A, ZERO, &
              MDXP(I)%A(1)/TIMFAC, 0,0,0,0,X,Y,Z &
#if KEY_ACE==1
               ,0,0,.FALSE.         & 
#endif
              )
      ELSE
          IGAMMA(I)%A => null()
      ENDIF
    ENDDO

    !       For ACE calculations generate a symmetric bonded list.
    !       It does not need to be updated since the connectivity is constant.
    MCA14P%A => null()
#if KEY_ACE==1
    IF (LGROUP .OR. LACE) THEN
#else /**/
    IF (LGROUP) THEN
#endif 
      CALL GTSYMNB(bnbnd%IBLO14,bnbnd%INB14, &
            MCA14P,NATOM,NATOM,1,NATOM,.TRUE.)
    ENDIF

#if KEY_GCMC==1 /*gcmc_setup*/
    !       Create an active index list.
    DO I = 1, NMVTYP
      NIDXGC(I) = 0
      IF (MVTYPE(I) .EQ. 8) THEN
          !           Grid-based cavity bias
          IF (NGCTRY .GT. 0 .OR. QGCGRD) THEN
            call chmalloc('mc.src','MCSTUP','ISCVTY(I)',NMVATM(I)+1,intgp=ISCVTY(I)%A)
            call chmalloc('mc.src','MCSTUP','NOCVTY(I)',NMVATM(I)+1,intgp=NOCVTY(I)%A)
            call chmalloc('mc.src','MCSTUP','NGCPCP(I)',NMVATM(I)+1,intgp=NGCPCP(I)%A)
            DO J = 1, NMVATM(I)+1
                ISCVTY(I)%A(J) = 0
                NOCVTY(I)%A(J) = 0
                NGCPCP(I)%A(J) = 0
            ENDDO
          ENDIF
          !           Active in the front and inactive in the back of the list.
          call chmalloc('mc.src','MCSTUP','IDXGCP(I)',NMVATM(I),intgp=IDXGCP(I)%A)
          N = 0
          L = NMVATM(I)
          DO J = 1, NMVATM(I)
            K = FRSIND(IMVNGP(I)%A(J)%A)
            IF (GCMCON(K)) THEN
                N = N + 1
                IDXGCP(I)%A(N) = J
            ELSE
                IDXGCP(I)%A(L) = J
                L = L - 1
            ENDIF
          ENDDO
          NIDXGC(I) = N
          NIGCMC(I) = NIMVGF(IMVNGP(I)%A(1)%A)
      ENDIF
    ENDDO
#endif /*  (gcmc_setup)*/
    
    RETURN
  END SUBROUTINE MCSTUP

  SUBROUTINE MCNOEL(NMVTYP, NMVATM, IMVNGP)
    !
    !       Construct a list of NOE constraint terms affected by each
    !       move instance.
    !
    !       Presently works by a nested loop structure.  If this proves
    !       too slow, at the expense of using more memory, one could construct
    !       an atom-to-constraint lookup table (like MKBONDT) and use that
    !       when looping over the move instances.
    !
    !       Aaron R. Dinner
    !       99-07-12
    !
    use clcg_mod,only:random
    use chm_kinds
    use dimens_fcm
    use energym
    use noem
    use memory
    use mcmvutil, only: tagatm

    implicit none

    INTEGER NMVTYP
    INTEGER NMVATM(NMVTYP)
    type(iptr_ptr) :: IMVNGP(NMVTYP)

    INTEGER IMV, JMV, I, J, II, JJ, N, NPOS
    integer,pointer,dimension(:) :: CNSP, TEMPP
    integer,allocatable,dimension(:) :: ILISTP
    LOGICAL LPREV, LINCLD

    TEMPP => null()
    CNSP => null()

    IF ((NOENUM .GT. 0) .AND. QETERM(NOE)) THEN

      call chmalloc('mc.src','MCNOEL','ILISTP',NOENUM,intg=ILISTP)

      DO IMV = 1, NMVTYP

          allocate(NOECNP(IMV)%A(NMVATM(IMV)))

          DO JMV = 1, NMVATM(IMV)

            !             Tag moving atoms
            TEMPP => IMVNGP(IMV)%A(JMV)%A
            CALL TAGATM(IMVATP, TEMPP, .FALSE., 1)

            !             Loop over all the constraints and see which pairs
            !             are affected by this move instance.
            LPREV = .FALSE.
            NPOS = 0
            DO N = 1, NOENUM

                LINCLD = .FALSE.
                DO II = 1, NOEINM(N)
                  I=NOELIS(NOEIPT(N)+II-1)
#if KEY_PNOE==1 /*pnoe1*/
                  IF(IsPNOE(N)) THEN
                      LINCLD = (IMVATP(I) .GT. 0)
                  ELSE
#endif /* (pnoe1)*/
                      DO JJ=1,NOEJNM(N)
                        J=NOELIS(NOEJPT(N)+JJ-1)
                        LINCLD = (IMVATP(I) .NE. IMVATP(J))
                      ENDDO
#if KEY_PNOE==1 /*pnoe2*/
                  ENDIF
#endif /* (pnoe2)*/

                ENDDO

                IF ((.NOT. LPREV) .AND. LINCLD) THEN
                  NPOS = NPOS + 1
                  ILISTP(NPOS) = N
                ELSE IF (LPREV .AND. (.NOT. LINCLD)) THEN
                  NPOS = NPOS + 1
                  ILISTP(NPOS) = N - 1
                ENDIF
                LPREV = LINCLD

            ENDDO
            IF (LPREV) THEN
                NPOS = NPOS + 1
                ILISTP(NPOS) = N - 1
            ENDIF

            !             Allocate space for a more permanent array
            NPOS = NPOS + 1
            call chmalloc('mc.src','MCNOEL','CNSP',NPOS,intgp=CNSP)
            CNSP(1) = NPOS
            DO I = 2, NPOS
                CNSP(I) = ILISTP(I-1)
            ENDDO

            !             Store the address of this array in the NOECNP structure
            NOECNP(IMV)%A(JMV)%A => CNSP

            !             Clear moving atom array
            CALL TAGATM(IMVATP, TEMPP, .TRUE., 0)

          ENDDO
      ENDDO

      call chmdealloc('mc.src','MCNOEL','ILISTP',NOENUM,intg=ILISTP)

    ENDIF

    RETURN
  END SUBROUTINE MCNOEL

  LOGICAL FUNCTION MCACPT(IACCPF,ISEED,BETA,DEPOT,DEKIN,ETOT, &
      ACPAR1,ACPAR2,ACPAR3,NMULT,MLTLGP,QHMC)
    !
    !       Picks the right acceptance criterion and calls it.
    !
    !       Aaron R. Dinner
    !
    use chm_kinds
#if KEY_SAMC==1
    use samc
#endif

    implicit none
    !
    !       Passed Variables
    !
    INTEGER IACCPF, ISEED, NMULT
    real(chm_real) :: MLTLGP(:)
    real(chm_real)  DEPOT, DEKIN, BETA, ETOT
    real(chm_real)  ACPAR1,ACPAR2,ACPAR3
    LOGICAL QHMC
    
    IF (IACCPF .EQ. 0 .OR. IACCPF .EQ. 3) THEN
      MCACPT = METROP(DEPOT+DEKIN,BETA,ISEED)
    ELSE IF (IACCPF .EQ. 1) THEN
      MCACPT = MULTIF(DEPOT,ETOT,ACPAR1,ACPAR2,ACPAR3, &
            NMULT, MLTLGP, ISEED)
    ELSE IF (IACCPF .EQ. 2) THEN
      MCACPT = TSALLF(DEPOT,DEKIN,ETOT,ACPAR1,ACPAR2,BETA,ISEED, &
            QHMC)
#if KEY_SAMC==1
    ELSE IF (IACCPF .EQ.4) THEN
      IF (LSPDONE) THEN
        MCACPT = SPACPT(BETA,ISEED)
      ELSE
        MCACPT = METROP(DEPOT+DEKIN,BETA,ISEED)
      ENDIF
#endif
    ELSE
      CALL WRNDIE(-5,'<MCACPT>','UNKNOWN ACCEPTANCE CRITERION')
    ENDIF

    RETURN
  END FUNCTION MCACPT

  LOGICAL FUNCTION METROP(DE, BETA, ISEED)
    !
    !       Applies the Metropolis criterion
    !
    !       Aaron R. Dinner
    !
    use clcg_mod,only:random
    use chm_kinds
    use exfunc
    use number
    implicit none
    !
    !       Passed Variables
    !
    INTEGER ISEED
    real(chm_real) DE, BETA

    IF (DE .LE. ZERO) THEN
      METROP = .TRUE.
    ELSE
      METROP = (EXP(-BETA*DE) .GE. RANDOM(ISEED))
    ENDIF
    RETURN
  END FUNCTION METROP

  LOGICAL FUNCTION MULTIF(DE,ETOT,EMULTN,EMULTX,DEMULT, &
      NMULT,EMLTLG,ISEED)
    !
    !       Applies the multicanonical acceptance criterion
    !
    !       Aaron R. Dinner
    !
    use clcg_mod,only:random
    use chm_kinds
    use exfunc
    use number
    implicit none

    INTEGER ISEED, NMULT
    real(chm_real)  DE, ETOT, EMULTN, EMULTX, DEMULT
    real(chm_real)  EMLTLG(*)
    !
    INTEGER IE1, IE2
    real(chm_real)  R, ENEW

    MULTIF = .FALSE.

    ENEW = ETOT + DE

    !       If we have gotten outside the limits due to adding error or
    !       changes in the non-bonded list, reject the move automatically if it
    !       takes us further from the interval.
    IF (((ENEW .LT. EMULTN).AND.(ENEW .LT. ETOT)).OR. &
        ((ENEW .GE. EMULTX).AND.(ENEW .GT. ETOT))) RETURN

    IE1 = INT((ETOT - EMULTN)/DEMULT) + 1
    IE2 = INT((ENEW - EMULTN)/DEMULT) + 1
    IE1 = MAX(MIN(IE1,NMULT),1)
    IE2 = MAX(MIN(IE2,NMULT),1)

    R = EMLTLG(IE1)-EMLTLG(IE2)
    IF (R .GE. 0.0) THEN
      MULTIF = .TRUE.
    ELSE
      R = EXP(R)
      MULTIF = (RANDOM(ISEED) .LT. R)
    END IF

    RETURN
  END FUNCTION MULTIF

  LOGICAL FUNCTION TSALLF(DEPOT,DEKIN,ETOT,EMIN,TS1MQ,BETA,ISEED, &
      QHMC)
    !
    !       Applies the Tsallis acceptance criterion.
    !
    !       Aaron R. Dinner
    !
    use clcg_mod,only:random
    use chm_kinds
    use exfunc
    use number
    implicit none

    INTEGER ISEED
    real(chm_real)  DEPOT, DEKIN, ETOT, EMIN, TS1MQ, BETA
    LOGICAL QHMC
    !
    real(chm_real)  R, ENEW, EOLD, QB

    TSALLF = .FALSE.

    EOLD = ETOT - EMIN
    ENEW = EOLD + DEPOT

    IF (.NOT. QHMC .AND. ENEW .LE. EOLD) THEN
      TSALLF = .TRUE.
    ELSE
      QB = TS1MQ*BETA
      R = ((ONE-QB*ENEW)/(ONE-QB*EOLD))
      R = R**((ONE - TS1MQ)/TS1MQ)
      IF (QHMC) R = R*EXP(-BETA*DEKIN)
      TSALLF = (RANDOM(ISEED) .LT. R)
    END IF

    RETURN
  END FUNCTION TSALLF

  SUBROUTINE RUSPHR(ISEED,V)
    !
    !       Returns a random vector within a unit sphere.
    !
    !       Aaron R. Dinner
    !
    use clcg_mod,only:random
    use chm_kinds
    use exfunc
    use number
    implicit none
    INTEGER ISEED
    real(chm_real) V(3)

100 V(1) = TWO*RANDOM(ISEED) - ONE
    V(2) = TWO*RANDOM(ISEED) - ONE
    V(3) = TWO*RANDOM(ISEED) - ONE
    IF (V(1)*V(1)+V(2)*V(2)+V(3)*V(3) .GT. ONE) GOTO 100

    RETURN
  END SUBROUTINE RUSPHR

  SUBROUTINE SCLVEC(ANISO,DT,RMX,IDX,DX)
    !
    !       Scales the vector DT and returns it in DX.
    !
    !       Aaron R. Dinner
    !
    use chm_kinds
    use chm_types
    implicit none
    INTEGER IDX
    real(chm_real) DT(3), DX(3), RMX(:)
    LOGICAL ANISO
    !
    INTEGER I, J
    IF (ANISO) THEN
      I = (IDX - 1)*9
      DX(1) = RMX(I+1)*DT(1)+RMX(I+2)*DT(2)+RMX(I+3)*DT(3)
      DX(2) = RMX(I+4)*DT(1)+RMX(I+5)*DT(2)+RMX(I+6)*DT(3)
      DX(3) = RMX(I+7)*DT(1)+RMX(I+8)*DT(2)+RMX(I+9)*DT(3)
    ELSE
      DO I = 1, 3
          DX(I) = DT(I)*RMX(IDX)
      ENDDO
    ENDIF
    RETURN
  END SUBROUTINE SCLVEC

  FUNCTION MCA2GF(IMVNG,NGRP,IGPBS,IMVATM,IGROUP)
    !
    !       Generates moving group lists from moving atom lists
    !
    !       Aaron R. Dinner
    !
    use chm_kinds
    use memory
    use mcmvutil, only: tagatm

    implicit none
    integer,pointer,dimension(:) :: MCA2GF
    INTEGER IMVNG(:), NGRP, IMVATM(:), IGROUP(:)
    INTEGER IGPBS(*)
    !
    INTEGER I, J, N, NRIG, NTOT, IPV
    integer,allocatable,dimension(:) :: NMAPP, NPAIRP, NPPBSP

    DO I = 1, NGRP
      IGROUP(I) = 0
    ENDDO

    !       Mark the moving atoms with by rigid units
    CALL TAGATM(IMVATM,IMVNG,.FALSE.,1)

    !       Run through the atom list,
    !       If a whole group is in a rigid unit,
    !         assign it that unit number
    !       Else (mixed rigid units)
    !         assign it a new unit number higher than existing
    !
    NRIG = IMVNG(1)
    DO I = 1, NGRP
      IGROUP(I) = IRIGGF(IMVATM,IGPBS(I)+1,IGPBS(I+1),NRIG)
      IF (IGROUP(I).GT.NRIG) NRIG = IGROUP(I)
    ENDDO

    !       Unmark the moving atoms for future use of array
    CALL TAGATM(IMVATM,IMVNG,.TRUE.,0)

    !       Run through IGROUP
    !       Count how many pairs for each group rigid unit
    call chmalloc('mc.src','MCA2GF','NMAPP',NRIG,intg=NMAPP)
    call chmalloc('mc.src','MCA2GF','NPAIRP',NRIG,intg=NPAIRP)
    call chmalloc('mc.src','MCA2GF','NPPBSP',NRIG,intg=NPPBSP)

    NPAIRP(1:NRIG) = 0
    IPV = 0
    DO I = 1, NGRP
      IF (IGROUP(I).NE.IPV) THEN
          IF (IGROUP(I).NE.0) THEN
            N = NPAIRP(IGROUP(I))
            NPAIRP(IGROUP(I)) = N + 1
          ENDIF
          IF (IPV.NE.0) THEN
            N = NPAIRP(IPV)
            NPAIRP(IPV) = N + 1
          ENDIF
      ENDIF
      IPV = IGROUP(I)
    ENDDO
    IF (IGROUP(NGRP).NE.0) THEN
      N = NPAIRP(IGROUP(NGRP))
      NPAIRP(IGROUP(NGRP)) = N + 1
    ENDIF

    !       In case any units were skipped (due to mixing) generate
    !       a mapping of the old unit numbers to contiguous ones.
    !       Indexes start at 2.
    J = 0
    DO I = 1, NRIG
      IF (NPAIRP(I) > 0) THEN
          J = J + 1
          NMAPP(I) = J
          NPAIRP(J) = NPAIRP(I)
      ENDIF
    ENDDO
    NRIG = J
    !       Remap everthing to the new indices
    DO I = 1, NGRP
      IF (IGROUP(I) > 0) IGROUP(I) = NMAPP(IGROUP(I))
    ENDDO

    !       Figure out the total space needed
    NTOT = NRIG + 1
    DO I = 1, NRIG
      NTOT = NTOT + NPAIRP(I)
    ENDDO
    call chmalloc('mc.src','MCA2GF','MCA2GF',NTOT,intgp=MCA2GF)

    !       Finally, fill in the array (same structure as IMVNGP)
    J = NRIG + 1
    MCA2GF(1) = J
    DO I = 1, NRIG
      NPPBSP(I) = J
      J = J + NPAIRP(I)
      MCA2GF(I+1) = J
    ENDDO
    NPAIRP(1:NRIG) = 0
    IPV = 0
    DO I = 1, NGRP
      IF ((IGROUP(I).GT.0).AND.(IGROUP(I).NE.IPV)) THEN
          N = NPAIRP(IGROUP(I)) + 1
          NPAIRP(IGROUP(I)) = N
          N = N + NPPBSP(IGROUP(I))
          MCA2GF(N) = I
      ENDIF
      !         Do not make this an ELSE since not mutually exclusive!
      IF ((IPV.GT.0).AND.(IGROUP(I).NE.IPV)) THEN
          N = NPAIRP(IPV) + 1
          NPAIRP(IPV) = N
          N = N + NPPBSP(IPV)
          MCA2GF(N) = I - 1
      ENDIF
      IPV = IGROUP(I)
    ENDDO
    IF (IGROUP(NGRP).NE.0) THEN
      N = NPAIRP(IPV) + 1
      NPAIRP(IPV) = N
      N = N + NPPBSP(IPV)
      MCA2GF(N) = NGRP
    ENDIF

    call chmdealloc('mc.src','MCA2GF','NPPBSP',NRIG,intg=NPPBSP)
    call chmdealloc('mc.src','MCA2GF','NPAIRP',NRIG,intg=NPAIRP)
    call chmdealloc('mc.src','MCA2GF','NMAPP',NRIG,intg=NMAPP)

    RETURN
  END FUNCTION MCA2GF

  INTEGER FUNCTION IRIGGF(IMVATM,IAF,IAL,NRIG)
    !
    !       Returns the rigid unit number of a group.
    !
    !       Aaron R. Dinner
    !
    use chm_kinds
    implicit none
    INTEGER IMVATM(:), IAF, IAL, NRIG
    !
    INTEGER I, J

    IRIGGF = IMVATM(IAF)
    J = IAF + 1
    DO I = J, IAL
      IF (IMVATM(I).NE.IRIGGF) THEN
          IRIGGF = NRIG + 1
          RETURN
      ENDIF
    ENDDO

    RETURN
  END FUNCTION IRIGGF

  SUBROUTINE MCFREE(NMVTYP, &
      NMVATM,MVTYPE,IMVNGP, &
      IARMF, IDOMCF,ANISO, &
      IACCPF,MLTLGP,NMULT,LCENTR,MCA14P, &
      MCMINN &
#if KEY_GCMC==1
      ,NIDXGC, QGCGRD,NGCTRY &
#endif 
      )
    !
    !       Frees dynamically allocated arrays carried in MC.
    !
    !       Aaron R. Dinner
    !
    use chm_kinds
    use chm_types
    use dimens_fcm
    use ace_module
    use memory
    use image
    use inbnd
    use noem
    use number
    use psf
    use mce, only: frsymnb

    implicit none
    !
    !       Passed Variables
    !
    INTEGER NMVTYP, NMVATM(NMVTYP), MVTYPE(NMVTYP)
    type(iptr_ptr) :: IMVNGP(NMVTYP)
    INTEGER IARMF
    INTEGER IDOMCF
    type(chm_ptr) :: MLTLGP
    INTEGER IACCPF, NMULT, LCENTR
    type(iptr_ptr) :: MCA14P
    INTEGER MCMINN(NMVTYP)
    LOGICAL ANISO(NMVTYP)
#if KEY_GCMC==1
    INTEGER NIDXGC(NMVTYP)
    INTEGER NGCTRY
    LOGICAL QGCGRD
#endif 
    !
    !       Local Variables
    !
    integer,pointer,dimension(:) :: TEMPP
    INTEGER I, J, N, NF
    LOGICAL LALL

    IF (IACCPF .EQ. 1) call chmdealloc('mc.src','MCFREE','MLTLGP',NMULT,crlp=MLTLGP%A)

    IF (NFREAT .LT. NATOM) call chmdealloc('mc.src','MCFREE','IFRATP',NFREAT,intg=IFRATP)

    call chmdealloc('mc.src','MCFREE','IMVATP',NATOM,intg=IMVATP)
    call chmdealloc('mc.src','MCFREE','ISKIP',NSKIP,intg=ISKIP)

    IF (LGROUP) THEN
      call chmdealloc('mc.src','MCFREE','IMVGRP',NGRP,intg=IMVGRP)
      DO I = 1, NMVTYP
          DO J = 1, NMVATM(I)
            TEMPP => IMCGRP(I)%A(J)%A
            N = TEMPP(1)
            N = TEMPP(N)
            call chmdealloc('mc.src','MCFREE','TEMPP',N,intgp=TEMPP)
          ENDDO
          deallocate(IMCGRP(I)%A)
      ENDDO
      IF (NTRANS > 0) call chmdealloc('mc.src','MCFREE','IA2GP',NATOM,intg=IA2GP)

      !         Need to turn centering back on in energy routines
      LCENTR = 0
    ENDIF

    !       NOE constraints
    IF (NOENUM .GT. 0) THEN
      DO I = 1, NMVTYP
          DO J = 1, NMVATM(I)
            TEMPP => NOECNP(I)%A(J)%A
            N = TEMPP(1)
            call chmdealloc('mc.src','MCFREE','TEMPP',N,intgp=TEMPP)
          ENDDO
          deallocate(NOECNP(I)%A)
      ENDDO
    ENDIF

    IF ((IARMF .GT. 0) .OR. (IDOMCF .GT. 0)) THEN
      DO I = 1, NMVTYP
          call chmdealloc('mc.src','MCFREE','NTRYP(I)',NMVATM(I),intgp=NTRYP(I)%A)
          call chmdealloc('mc.src','MCFREE','MCACCP(I)',NMVATM(I),intgp=MCACCP(I)%A)
      ENDDO
    ENDIF

    IF (IDOMCF .GT. 0) THEN
      DO I = 1, NMVTYP
          IF (ANISO(I)) THEN
            N = 6*NMVATM(I)
            call chmdealloc('mc.src','MCFREE','EAVEP(I)',N,crlp=EAVEP(I)%A)
            N = 15*NMVATM(I)
            call chmdealloc('mc.src','MCFREE','DAVEP(I)',N,crlp=DAVEP(I)%A)
          ELSE
            call chmdealloc('mc.src','MCFREE','EAVEP(I)',NMVATM(I),crlp=EAVEP(I)%A)
            call chmdealloc('mc.src','MCFREE','DAVEP(I)',NMVATM(I),crlp=DAVEP(I)%A)
          ENDIF
      ENDDO
    ENDIF

    LALL = .FALSE.
    DO I = 1, NMVTYP
      IF (MCMINN(I) .NE. 0) LALL = .TRUE.
    ENDDO

    IF (LALL) THEN
      N = ALLMVP%A(1)
      N = ALLMVP%A(N)
      call chmdealloc('mc.src','MCFREE','ALLMVP',N,intgp=ALLMVP%A)
      IF (LGROUP) THEN
          N = ALLGRP%A(1)
          N = ALLGRP%A(N)
          call chmdealloc('mc.src','MCFREE','ALLGRP',N,intgp=ALLGRP%A)
      ENDIF
    ENDIF

    DO I = 1, NMVTYP
      IF (MCMINN(I) .LT. 0) THEN
          call chmdealloc('mc.src','MCFREE','IGAMMA(I)',4*NATOM,crlp=IGAMMA(I)%A)
      ENDIF
    ENDDO

#if KEY_ACE==1
    IF (LGROUP .OR. LACE) THEN
#else /**/
    IF (LGROUP) THEN
#endif 
      CALL FRSYMNB(MCA14P,NATOM)
    ENDIF

#if KEY_GCMC==1 /*gcmc_free*/
    DO I = 1, NMVTYP
      IF (MVTYPE(I) .EQ. 8) THEN
          IF (NGCTRY .GT. 0 .OR. QGCGRD) THEN
            call chmdealloc('mc.src','MCFREE','ISCVTY(I)',NMVATM(I)+1,intgp=ISCVTY(I)%A)
            call chmdealloc('mc.src','MCFREE','NOCVTY(I)',NMVATM(I)+1,intgp=NOCVTY(I)%A)
            call chmdealloc('mc.src','MCFREE','NGCPCP(I)',NMVATM(I)+1,intgp=NGCPCP(I)%A)
          ENDIF
          call chmdealloc('mc.src','MCFREE','IDXGCP(I)',NMVATM(I),intgp=IDXGCP(I)%A)
      ENDIF
    ENDDO
#endif /*   (gcmc_free)*/
    
    RETURN
  END SUBROUTINE MCFREE

  SUBROUTINE FLIP(IMVATM,NATOM)
    !
    !       Flips all the 0 and non-zero terms in an array
    !
    !       Aaron R. Dinner
    !
    use chm_kinds
    implicit none
    INTEGER IMVATM(:), NATOM
    INTEGER I
    DO I = 1, NATOM
      IMVATM(I) = 1 - IMVATM(I)
    ENDDO
    
    RETURN
  END SUBROUTINE FLIP
  
  SUBROUTINE CNTMVG(IMVNG,X,Y,Z,XCENT,YCENT,ZCENT)
    !
    !       Get X, Y, Z center of moving groups.
    !
    !       Aaron  R. Dinner
    !
    use chm_kinds
    use dimens_fcm
    use consta
    use number
    use psf
    use mcmvutil, only: cntone
    implicit none
    INTEGER IMVNG(:)
    real(chm_real)  X(*), Y(*), Z(*)
    real(chm_real)  XCENT(*), YCENT(*), ZCENT(*)

    INTEGER I, J, K, IRS, IAF, IAL, NG, NN
    NG = IMVNG(1)
    J = NG + 2
    DO I = 2, NG
      NN = IMVNG(I)
      DO K = J, NN, 2
          IAF = IMVNG(K-1)
          IAL = IMVNG(K)
          DO IRS = IAF, IAL
            CALL CNTONE(XCENT,YCENT,ZCENT,IRS,IGPBS,X,Y,Z)
          ENDDO
      ENDDO
      J = NN + 2
    ENDDO
    RETURN
  END SUBROUTINE CNTMVG

#if KEY_GCMC==1
  SUBROUTINE MCENOB(PCFB,NCFB,LGCOLD,EMC,EGSBP,ISEED,IDX,IMVNG, &
      NIGCMC,IBLSTP,IMCGRP,NOECNP,ISKIP, &
      IMVATM,IMVGRP,MBONDT,QBND,MCBLOP,MCBLGP,MCIMLP, &
      MCIMGP,BDUMMY,BDIMMY,MCATTP,IAMC,NAMC,IMLIMP, &
      NTRANS,IMTRNS,NOROT,IMATPT,MCA14P,IGMC,NGMC, &
      MCGTTP,X,Y,Z,BETA,GCMCON &
#if KEY_PERT==1
       ,MCBLORP, MCBLOPP,BDUMMYR, BDUMMYP  & 
#endif
#if KEY_PERT==1
       ,MCIMLRP,MCIMLPP,IMLIMRP,IMLIMPP,   & 
#endif
#if KEY_PERT==1
       BDIMMYR,BDIMMYP                    & 
#endif
      )
    !
    !       Calculate MC energy multiple times for orientational bias
    !
    use chm_kinds
    use chm_types
    use dimens_fcm
    use consta
    use memory
    use number
#if KEY_PBEQ==1
    use pbeq 
#endif
    use clcg_mod,only: random
    use mce, only: mcener
    use mcmvrtrn, only: mkrrot, mvgcom
    implicit none

    type(chm_iptr) :: IMVNG(:), IBLSTP(:), IMCGRP(:), NOECNP(:)
    INTEGER NCFB, NIGCMC, ISEED, IDX
    INTEGER ISKIP(:)
    INTEGER IMVATM(:), IMVGRP(:), MBONDT
    type(chm_iptr),dimension(:) :: MCBLOP, MCBLGP, MCIMLP, MCIMGP
    type(nonbondDataStructure) :: BDUMMY
    type(imageDataStructure) :: BDIMMY
    type(chm_iptr),dimension(:) :: MCATTP, MCA14P
    INTEGER IAMC, NAMC
    integer,dimension(:) :: IMLIMP
    INTEGER NTRANS,IMATPT(:)
    INTEGER IGMC, NGMC
    type(chm_iptr),dimension(:) :: MCGTTP
#if KEY_PERT==1
    type(chm_iptr),dimension(:) :: MCBLORP, MCBLOPP
    type(nonbondDataStructure) :: BDUMMYR, BDUMMYP
    type(chm_iptr),dimension(:) :: MCIMLRP, MCIMLPP
    integer,dimension(:) :: IMLIMRP, IMLIMPP
    type(imageDataStructure) :: BDIMMYR, BDIMMYP
#endif 
    real(chm_real), dimension(:) :: x, y, z
    real(chm_real)  PCFB,EMC,BETA,EGSBP
    real(chm_real)  IMTRNS(*)
    LOGICAL NOROT
    LOGICAL LGCOLD,QBND(:),GCMCON(:)

    INTEGER I,N,NG,K,IAF,IAL,IC,IK
    INTEGER ICFB, IEF
    integer,pointer :: IMAGEP(:)
    real(chm_real),allocatable,dimension(:) :: IENER, IGSBP, IEFB
    real(chm_real),allocatable,dimension(:,:) :: IXFB
    real(chm_real)  ESM,ESB,EMCOL1,EBF,EBFT,QRSD,ESBD,DEB,EGSBP1
    real(chm_real)  XCM,YCM,ZCM,DX1

    real(chm_real),parameter :: XOLD0(3) = (/ ZERO, ZERO, ZERO /)
    real(chm_real),parameter :: CENT0(1) = (/ ZERO /)
    integer,parameter :: IGPBS0(1) = (/ 0 /)
    integer,parameter :: IGRPP0(1) = (/ 0 /)
    type(chm_iptr) :: MCGTTP0(1)

    IMAGEP => null()

    !       Allocate memory for energies
    call chmalloc('mc.src','MCENOB','IENER',NCFB,crl=IENER)
#if KEY_PBEQ==1
    IF (QGSBP) call chmalloc('mvgcmc.src','MCENOB','IGSBP',NCFB,crl=IGSBP) 
#endif
    !       Allocate memory for Boltzmann factors
    call chmalloc('mc.src','MCENOB','IEFB',NCFB,crl=IEFB)

    !       Allocate memory for coordinates
    call chmalloc('mc.src','MCENOB','IXFB',3*NIGCMC,NCFB,crl=IXFB)

    !       Save the first set of coordinates and energies
    CALL SVOLDC(IDX, IMVNG, IXFB(:,1), XOLD0, X,Y,Z, 0, .TRUE.)
    IENER(1) = EMC
#if KEY_PBEQ==1
    IF (QGSBP) IGSBP(1) = EGSBP 
#endif

    !       Make an additional ncfb-1 rotations
    ESM = EMC
    CALL MVGCOM(XCM,YCM,ZCM, IMVNG(IDX)%A, X,Y,Z)
    IF (NTRANS.GT.0) IMAGEP => IMVNG(IDX)%A
    DO I=2,NCFB
      CALL MKRROT(DX1,X,Y,Z, IMVNG(IDX)%A, -1,ONE8TY,ISEED)
      !         Update images.  Atom (not group) based calculations are assumed.
      !         Also, CNTTRN, calling sequence depends on GCMC keyword.
      CALL CNTTRN(CENT0, CENT0, CENT0, .FALSE., NTRANS, IGRPP0, IMAGEP, MCATTP, IMTRNS, &
            NOROT,BDIMMY,MCGTTP0,IGPBS0,X,Y,Z &
#if KEY_GCMC==1
            ,GCMCON &  
#endif
            )
      !         Parameters assume it is a grand canonical MC
      CALL MCENER(EMCOL1,8,IDX,IMVNG,IBLSTP,IMCGRP,NOECNP, &
            ISKIP, IMVATM, IMVGRP, MBONDT,QBND,MCBLOP,MCBLGP, &
            MCIMLP,MCIMGP,.FALSE.,BDUMMY,BDIMMY,MCATTP,IAMC, &
            NAMC,IMLIMP,MCA14P,IGMC,NGMC,MCGTTP,X,Y,Z,.TRUE., &
#if KEY_PERT==1
            MCBLORP, MCBLOPP,BDUMMYR, BDUMMYP,  & 
#endif
#if KEY_PERT==1
            MCIMLRP,MCIMLPP,IMLIMRP,IMLIMPP,    & 
#endif
#if KEY_PERT==1
            BDIMMYR,BDIMMYP,                    & 
#endif
            EGSBP1)
      IF(.NOT. LGCOLD) THEN
          CALL SVOLDC(IDX, IMVNG, IXFB(:,I), XOLD0, X,Y,Z, 0, .TRUE.)
      ENDIF
      IENER(I) = EMCOL1
#if KEY_PBEQ==1
       IF (QGSBP) IGSBP(I) = EGSBP1 
#endif
      IF (ESM .GT. EMCOL1) ESM = EMCOL1
    ENDDO

    !       Compute relative Boltzmann factors
    EBF = EXP(-BETA*(EMC-ESM))
    IEFB(1) = EBF
    EBFT = EBF
    DO I=2,NCFB
      DEB = IENER(I) - ESM
      EBF = EXP(-BETA*DEB)
      IEFB(I) = EBF
      EBFT = EBFT + EBF
    ENDDO

    IF (LGCOLD) THEN
      !         If it is a deletion, always use first structure
      ICFB = 1
    ELSE
      !         If it is an insertion, pick with weights
      QRSD = RANDOM(ISEED)
      ESBD = ZERO
      ICFB = 0
      DO WHILE (ESBD.LT.QRSD .AND. ICFB.LT.NCFB)
          ICFB = ICFB + 1
          ESBD = ESBD + IEFB(ICFB) / EBFT
      ENDDO
    ENDIF

    !       Set the coordinates and energies to the chosen configuration
    CALL RESTOR(IDX, IMVNG, IXFB(:,ICFB), XOLD0, X,Y,Z, 0, .TRUE.)
    !       Update images.  Atom (not group) based calculations are assumed.
    !       Also, CNTTRN, calling sequence depends on GCMC keyword.
    CALL CNTTRN(CENT0, CENT0, CENT0, .FALSE., NTRANS, IGRPP0, IMAGEP, MCATTP, IMTRNS, &
        NOROT,BDIMMY,MCGTTP0,IGPBS0,X,Y,Z &
#if KEY_GCMC==1
         ,GCMCON &  
#endif
        )
    EMC = IENER(ICFB)
#if KEY_PBEQ==1
    IF (QGSBP) EGSBP = IGSBP(ICFB) 
#endif
    PCFB = EXP(-BETA*(EMC-ESM))/EBFT

    call chmdealloc('mc.src','MCENOB','IXFB',3*NIGCMC,NCFB,crl=IXFB)
    call chmdealloc('mc.src','MCENOB','IEFB',NCFB,crl=IEFB)
#if KEY_PBEQ==1
    IF (QGSBP) call chmdealloc('mvgcmc.src','MCENOB','IGSBP',NCFB,crl=IGSBP) 
#endif
    call chmdealloc('mc.src','MCENOB','IENER',NCFB,crl=IENER)

    RETURN
  END SUBROUTINE MCENOB
#endif 

#endif 
end module mcc

