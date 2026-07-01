module iapbs
  use chm_kinds
  use dimens_fcm
#if KEY_APBS==1 /*apbs*/
!
! APBS module for CHARMM/APBS integration
!
! This code is based on iAPBS 1.0, available at
! http://mccammon.ucsd.edu/iapbs/
! APBS is Adaptive Poisson-Boltzmann Solver, available at
! http://www.poissonboltzmann.org/apbs/
!

!-----------------------------------------------------------------------
!
!                 iAPBS variables definition
!
! Similar type of varibles are bunched together (integer/integer and
! real/real) to mimize fortran/C passing incompatability.
!
! For detailed description of individual variables please see
! APBS Programmer's Guide.
!
!-----------------------------------------------------------------------

      integer MAXION
      parameter (MAXION = 2)

!-----------------------------------------------------------------------
!   i_param members
!
! 1 int type;         : What type of MG calculation?
!                            0: sequential manual
!                            1: sequential auto-focus
!                            2: parallel auto-focus
! 2 int nlev;         : Levels in multigrid hierarchy
! 3 int cmeth;        : Centering method:
!                            0: center on point,
!                            1: center on molecule
! 4 int ccmeth;       : Coarse grid centering method:  0 => center
!                       on point, 1 => center on molecule
! 5 int fcmeth;       : Fine grid centering method:  0 => center on
!                       point, 1 => center on molecule
! 6 int chgm            Types of charge discretization methods
!                            0: Trilinear interpolation of charge to
!                       8 nearest grid points. The traditional method;
!                       not particularly good to use with PBE forces.
!                            1: Cubic B-spline across nearest- and
!                       next-nearest-neighbors.  Mainly for use in
!                       grid-sensitive applications
!                       (such as force calculations).
!
! 7 int nonlin;           : 0 => LPBE, 1 => NPBE, 2: LRPBE,
!                           3: NRPBE, 4: SMPBE
! 8 int bcfl;             : Boundary condition: 0 => zero, 1 => single
!                           Debye-Huckel sphere, 2 => multiple Debye-
!                           Huckel spheres, 4 => focusing
! 9 int srfm;             : Surface calculation method
!                              0: Mol surface for epsilon; inflated VdW
!                                 for kappa; no smoothing
!                              1: As 0 with harmoic average
!                                 smoothing
!                              2: Cubic spline
!10 int calcenergy;       : Energy calculation
!                              0: don't calculate out energy
!                              1: calculate total energy
!                              2: calculate total energy and all energy
!                                 components
!11 int calcforce;        : Atomic forces I/O
!                              0: don't calculate forces
!                              1: calculate net forces on molecule
!                              2: calculate atom-level forces

!12 int wpot              : write potential map
!13 int wchg
!14 int wsmol
!15 wkappa
!16 wdiel
!17 dummy
!18 dummy
!19 dummy
!20 dummy
!21 calcnpenergy
!22 int nion;             : Number of counterion species
!23 int rchg;             : read charge map
!24 int rkappa            : read kappa map
!25 int rdiel             : read diel maps (x, y and z)

      integer i_param(25)

!-----------------------------------------------------------------------
! r_param memebers
! 1 double pdie;          : Solute dielectric
! 2 double sdie;          : Solvent dielectric
! 3 double srad;          : Solvent radius
! 4 double swin;          : Cubic spline window
! 5 double temp;          : Temperature (in K)
! 6 double sdens;         : Vacc sphere density
! 7 double gamma;         : Surface tension for apolar energies/forces
!                           (in kJ/mol/A^2)
! 8 double smvolume
! 9 double smsize
!

      real(chm_real) r_param(9)

!-----------------------------------------------------------------------
!   int dime[3];               /**< Grid dimensions */
!   int pdime[3];              /**< Grid of processors to be used in
!                               * calculation */
!
      integer dime(3), pdime(3)

!-----------------------------------------------------------------------
!   double grid[3];            /**< Grid spacings */
!   double glen[3];            /**< Grid side lengths. */
!   double center[3];          /**< Grid center. If ispart = 0, then this is
!                               * only meaningful if cmeth = 0.  However, if
!                               * ispart = 1 and cmeth = 0, then this is the
!                               * center of the non-disjoint (overlapping)
!                               * partition.  If ispart = 1 and cmeth = 1, then
!                               * this is the vector that must be added to the
!                               * center of the molecule to give the center of
!                               * the non-disjoint partition.  */c
!   double cglen[3];           /**< Coarse grid side lengths */
!   double fglen[3];           /**< Fine grid side lengths */
!   double ccenter[3];         /**< Coarse grid center.  */
!   double fcenter[3];         /**< Fine grid center.  */
!   double ofrac;              /**< Overlap fraction between procs */
!
      real(chm_real) grid(3), glen(3), center(3), cglen(3), fglen(3)
      real(chm_real) ccenter(3), fcenter(3), ofrac

!-----------------------------------------------------------------------
! mobile ion definition
!
!   double ionq[MAXION]; /**< Counterion charges (in e) */
!   double ionc[MAXION]; /**< Counterion concentrations (in M) */
!   double ionr[MAXION]; /**< Counterion radii (in A) */
!
      real(chm_real) ionq(MAXION), ionc(MAXION), ionrr(MAXION)

!-----------------------------------------------------------------------
! atom
!    double position[3];     /**< Atomic position */
!    double radius;          /**< Atomic radius   */
!    double charge;          /**< Atomic charge   */
! these are defined inside of CHARMM

!-----------------------------------------------------------------------
! radii from wmain get saved in a_radius
      real(chm_real),allocatable,dimension(:) :: a_radius

!-----------------------------------------------------------------------
! solvation energy and forces saved for use later
      real(chm_real) senelec, sennp
      real(chm_real),allocatable,dimension(:) :: solvfrcx,solvfrcy,solvfrcz

!-----------------------------------------------------------------------
! some integer variables
!
! napbs - how often we calculate APBS forces during MD/minimization
! umeth - forces update method
! a_debug - debug/verbosity value [0-5]
!
      integer napbs, umeth, apbs_debug

!-----------------------------------------------------------------------
! logical variables
!
! qapbs: were all parameters parsed?
! qfapbs: do we want to calculate solvation forces?
! qaparsed: did we parse all user options?
! dime_updates: update grid dimensions on the fly
      logical qapbs, qfapbs, qaparsed, qdime_updates

contains

   SUBROUTINE APBS_INIT()
      use memory
      use dimens_fcm,only:maxaim

      if(.not.allocated(a_radius)) then      
         call chmalloc('apbs.src','APBS_INIT','a_radius',maxaim,crl=a_radius)
         call chmalloc('apbs.src','APBS_INIT','solvfrcx',maxaim,crl=solvfrcx)
         call chmalloc('apbs.src','APBS_INIT','solvfrcy',maxaim,crl=solvfrcy)
         call chmalloc('apbs.src','APBS_INIT','solvfrcz',maxaim,crl=solvfrcz)
      endif
   END SUBROUTINE APBS_INIT

   SUBROUTINE APBS_CLEANUP()
      use memory
      use dimens_fcm,only:maxaim

      if(allocated(a_radius)) then
         call chmdealloc('apbs.src','APBS_INIT','a_radius',maxaim,crl=a_radius)
         call chmdealloc('apbs.src','APBS_INIT','solvfrcx',maxaim,crl=solvfrcx)
         call chmdealloc('apbs.src','APBS_INIT','solvfrcy',maxaim,crl=solvfrcy)
         call chmdealloc('apbs.src','APBS_INIT','solvfrcz',maxaim,crl=solvfrcz)
      endif
   END SUBROUTINE APBS_CLEANUP

   SUBROUTINE APBS(PBRAD)
!-----------------------------------------------------------------------
! numbers definitions
  use number
  use chutil
  use string
  use comand
! natom, cg (charge)
  use psf
  use stream
! x, y, z coords, wmain
  use coord
  use cvio
  use memory
  use timerm
  use machutil
! forces common block: dx, dy, dz
  use deriv
  use parallel
  use select

      implicit none
      real(chm_real) :: PBRAD(:)
!-----------------------------------------------------------------
! local variables
      integer,allocatable,dimension(:) :: ISLCT
      integer,allocatable,dimension(:) :: lstpbi
      integer rc, apbsdrv
!      integer ncalc(1)
      logical skip, sp_apbs

      INTEGER nonlin, bcfl, nion, srfm, calcenergy, calcforce
      INTEGER calc_type, nlev, cmeth, ccmeth, fcmeth, chgm
      INTEGER calcnpenergy, wpot, wchg, wsmol, wkappa, wdiel
      INTEGER rchg, rkappa, rdiel
      INTEGER apbs_print, radiopt
      real(chm_real) pdie, sdie, srad, swinapbs, tempapbs, gamma
      real(chm_real) sdens,smvolume, smsize
      real(chm_real) maxx, minx, maxy, miny, maxz, minz

      real(chm_real) esenerg(15), npenerg(15)
      real(chm_real) apbsdx(NATOM), apbsdy(NATOM), apbsdz(NATOM)
      real(chm_real) apbsqfx(NATOM), apbsqfy(NATOM), apbsqfz(NATOM)
      real(chm_real) apbsibx(NATOM), apbsiby(NATOM), apbsibz(NATOM)
      real(chm_real) apbsnpx(NATOM), apbsnpy(NATOM), apbsnpz(NATOM)
      real(chm_real) apbsdbx(NATOM), apbsdby(NATOM), apbsdbz(NATOM)

      INTEGER ntpbi
! Local variables
      INTEGER IMODE, I, LISTR, NN
      SAVE
!
      IF (TIMER.GT.1) &
           CALL WRTTIM('PBEQ/APBS solver times:')
      call apbs_init() ! calling this multiple times is harmless
      call chmalloc('apbs.src','APBS','ISLCT',NATOM,intg=ISLCT)
      call chmalloc('apbs.src','APBS','lstpbi',NATOM,intg=lstpbi)
!
!     Atom Selection
!     ==============

      IMODE=0 !implies default = all atoms selected
      CALL SELRPN(COMLYN,COMLEN,ISLCT,NATOM,1,IMODE, &
           .FALSE.,1,' ',0,RESID,RES,IBASE,SEGID,NICTOT,NSEG, &
           .TRUE.,X,Y,Z,.TRUE.,1,PBRAD)

      IF(IMODE.NE.0)THEN
         Write(outu,'(/,3x,a)') &
              'PREP WARNING: Not all atoms selected, is this what you want?'
!        CALL WRNDIE(-1,'<PREP>','ATOM SELECTION PARSING ERROR')
      ENDIF

      CALL MAKIND(NATOM,ISLCT,lstpbi,NTpbi)
      WRITE(OUTU,101)
      WRITE(OUTU,101) 'Calculation with ',NTpbi,' atoms'
 101  FORMAT(3X,A,I6,A)

      call chmdealloc('apbs.src','APBS','ISLCT',NATOM,intg=ISLCT)
      call chmdealloc('apbs.src','APBS','lstpbi',NATOM,intg=lstpbi)

!-----------------------------------------------------------------
!
!     start of APBS section
!
!-----------------------------------------------------------------

! we are in APBS module so set qapbs to .T.
      qapbs = .true.

! get the keywords and assign apbs parameters
      qaparsed = .false.

!   natom  - current number of atoms

!-----------------------------------------------------------------
!
! CHARMM/APBS commands parsing section
!
!-----------------------------------------------------------------

! rokFIXME: also add centering-related (center) keywords

    ! Default values
    ! PBEparm
!      ispara = 0
!   i_pbeparm(1) = 1 ! molid
      nonlin = 0
      bcfl = 1
      nion = 0
      srfm = 2
      pdie = 2.0D0
      sdie = 78.4D0
      srad = 1.4D0
      swinapbs = 0.3D0
      tempapbs = 298.15D0
      gamma = 0.105D0
      sdens = 10.0D0
      calcenergy = 1
      calcforce = 0
      calcnpenergy = 1
      wpot = 0
      wchg = 0
      wsmol = 0
      wkappa = 0
      wdiel= 0
      rchg = 0
      rkappa = 0
      rdiel = 0

    ! MGparm
      calc_type = 1 ! 0 - manual MG, 1- autoMG, 2- parallel MG
      nlev = 4
      cmeth = 1
      ccmeth = 1
      fcmeth = 1
      chgm = 1
  
      ofrac = 0.1D0

    !    ionq  = (/ 1.0, -1.0 /)
    !    ionc  = (/ 0.15, 0.15 /)
    !    ionrr = (/ 2.0, 2.0 /)

      do i = 1,3
         pdime(i) = 0
         dime(i)  = 0
      end do
      do i = 1, 3
         grid(i)    = 0.5D0
         glen(i)    = 0.0D0
         center(i)  = 0.0D0
         cglen(i)   = 0.0D0
         fglen(i)   = 0.0D0
         ccenter(i) = 0.0D0
         fcenter(i) = 0.0D0
      end do

    ! is this a single point energy calculation?
    ! default is no
      sp_apbs = .FALSE.
    ! printing verbosity
      apbs_print = 1
    ! debuging flag
      apbs_debug = 0
    ! radii optimization option
      radiopt = 0
    ! grid dimension recalculation
      qdime_updates = .FALSE.
    ! number of PB steps
      napbs = 0

!-----------------------------------------------------------------
! PBEparm
!-----------------------------------------------------------------

! do we need molID?? probably not
!      i_pbeparm(1) = 1

! pbetype :: LPBE/NPBE
      if (indxa(comlyn, comlen, 'LPBE') .gt. 0) then
         nonlin = 0
      endif
      if (indxa(comlyn, comlen, 'NPBE') .gt. 0) then
         nonlin = 1
      endif

! bcfl :: boundary condition
! defaults to sdh (single Debye-Huckel sphere)
      bcfl = gtrmi(comlyn, comlen, 'BCFL', bcfl)

! pdie :: solute dielectric
! defaults to 2.0
      pdie = gtrmf(comlyn, comlen, 'PDIE', pdie)

! sdie :; solvent dielectric
! defaults to 78.54
      sdie = gtrmf(comlyn, comlen, 'SDIE', sdie)

! srfm :: surface calculation method
! defaults to smol (molecular surface with smoothed harmonic average)
      srfm = gtrmi(comlyn, comlen, 'SRFM', srfm)

! srad :: solvent radius
! defaults to 1.4
      srad = gtrmf(comlyn, comlen, 'SRAD', srad)

! swinapbs :: cubic spline window
! defaults to 0.3
      swinapbs = gtrmf(comlyn, comlen, 'SWIN', swinapbs)

! tempapbs :: temperature (in K)
! defaults to 298.15
      tempapbs = gtrmf(comlyn, comlen, 'TEMP', tempapbs)

! gamma :: surface tension for apolar energies/forces (in kJ/mol/A^2)
! defaults to 0.105
      gamma = gtrmf(comlyn, comlen, 'GAMMA', gamma)

! sdens :: number of grid points per square-angstrom to use in Vacc object
! defaults to 10.0
      sdens = gtrmf(comlyn, comlen, 'SDENS', sdens)

! calcenergy :: energy calculation
! defaults to total energy calculation
      calcenergy = gtrmi(comlyn, comlen, 'CALCE', calcenergy)

! calcforce :: atomic forces I/O
! defaults to no forces calculation
      calcforce = gtrmi(comlyn, comlen, 'CALCF', calcforce)

! dielMap :: read dielectric maps (x, y and z)
      if (indxa(comlyn, comlen, 'RDIEL') .gt. 0) then
         rdiel = 1
      endif

! kappaMap :: read Kappa map
      if (indxa(comlyn, comlen, 'RKAPPA') .gt. 0) then
         rkappa = 1
      endif

! chargeMap :: read charge map
      if (indxa(comlyn, comlen, 'RCHG') .gt. 0) then
         rchg = 1
      endif

! get ions parameters and set nion (i_pbeparm(4))
! (we are considering 2 ions only - this should be enough for 
! most applications)
      if (indx(comlyn, comlen, 'IONQ1', 5) .gt. 0) then
         ionq(1) = gtrmf(comlyn, comlen, 'IONQ1', zero)
         ionc(1) = gtrmf(comlyn, comlen, 'IONC1', zero)
         ionrr(1) = gtrmf(comlyn, comlen, 'IONR1', zero)
         nion = 1
      endif
      if (indx(comlyn, comlen, 'IONQ2', 5) .gt. 0) then
         ionq(2) = gtrmf(comlyn, comlen, 'IONQ2', zero)
         ionc(2) = gtrmf(comlyn, comlen, 'IONC2', zero)
         ionrr(2) = gtrmf(comlyn, comlen, 'IONR2', zero)
         nion = 2
      endif


!-----------------------------------------------------------------
! external files write section
!-----------------------------------------------------------------
! write potential DX file
      if (indxa(comlyn, comlen, 'WPOT') .gt. 0) then
         wpot = 1
      endif
! write charge DX file
      if (indxa(comlyn, comlen, 'WCHG') .gt. 0) then
         wchg = 1
      endif
! write smol DX file
      if (indxa(comlyn, comlen, 'WSMOL') .gt. 0) then
         wsmol = 1
      endif
! write kappa DX file
      if (indxa(comlyn, comlen, 'WKAPPA') .gt. 0) then
         wkappa = 1
      endif
! write diel DX files (x, y and z)
      if (indxa(comlyn, comlen, 'WDIEL') .gt. 0) then
         wdiel = 1
      endif
!-----------------------------------------------------------------
! MGparm
!-----------------------------------------------------------------

! type mgmanual/mgauto
      if (indxa(comlyn, comlen, 'MGMANUAL') .gt. 0) then
         calc_type = 0
      endif
      if (indxa(comlyn, comlen, 'MGAUTO') .gt. 0) then
         calc_type = 1
      endif

! mg-para 
      if (indxa(comlyn, comlen, 'MGPARA') .gt. 0) then
         calc_type = 2
!         ispara = 1
      endif

! nlev :: levels in multigrid hierarchy
! used in mg-manual only, no default
      nlev = gtrmi(comlyn, comlen, 'NLEV', 0)

! centering methods - defaults to centering on molecule
! 0 is centering on point, 1 is on molecule
! if centering on a point - needs CNT, CCN, FCN

! cmeth for manual only (needs cnt also)
      cmeth = gtrmi(comlyn, comlen, 'CMET', 1)

! ccmeth and fcmeth for auto (need ccn and fcn also)
! defaults to 1 (centered on molecule)
      ccmeth = gtrmi(comlyn, comlen, 'CCME', 1)
      fcmeth = gtrmi(comlyn, comlen, 'FCME', 1)

! dime :: grid dimensions
! no default
      dime(1) = gtrmi(comlyn, comlen, 'DIMX', dime(1))
      dime(2) = gtrmi(comlyn, comlen, 'DIMY', dime(2))
      dime(3) = gtrmi(comlyn, comlen, 'DIMZ', dime(3))

! grid center
      if ((indx(comlyn, comlen, 'CNTX', 4) .gt. 0) .and. &
           (indx(comlyn, comlen, 'CNTY', 4) .gt. 0) .and. &
           (indx(comlyn, comlen, 'CNTZ', 4) .gt. 0)) then
         center(1) = gtrmf(comlyn, comlen, 'CNTX', zero)
         center(2) = gtrmf(comlyn, comlen, 'CNTY', zero)
         center(3) = gtrmf(comlyn, comlen, 'CNTZ', zero)
      endif

! ccenter
      if ((indx(comlyn, comlen, 'CCNX', 4) .gt. 0) .and. &
           (indx(comlyn, comlen, 'CCNY', 4) .gt. 0) .and. &
           (indx(comlyn, comlen, 'CCNZ', 4) .gt. 0)) then
         ccenter(1) = gtrmf(comlyn, comlen, 'CCNX', zero)
         ccenter(2) = gtrmf(comlyn, comlen, 'CCNY', zero)
         ccenter(3) = gtrmf(comlyn, comlen, 'CCNZ', zero)
      endif

! fcenter
      if ((indx(comlyn, comlen, 'FCNX', 4) .gt. 0) .and. &
           (indx(comlyn, comlen, 'FCNY', 4) .gt. 0) .and. &
           (indx(comlyn, comlen, 'FCNZ', 4) .gt. 0)) then
         fcenter(1) = gtrmf(comlyn, comlen, 'FCNX', zero)
         fcenter(2) = gtrmf(comlyn, comlen, 'FCNY', zero)
         fcenter(3) = gtrmf(comlyn, comlen, 'FCNZ', zero)
      endif

! cglen
      if ((indx(comlyn, comlen, 'CGLX', 4) .gt. 0) .and. &
           (indx(comlyn, comlen, 'CGLY', 4) .gt. 0) .and. &
           (indx(comlyn, comlen, 'CGLZ', 4) .gt. 0)) then
         cglen(1) = gtrmf(comlyn, comlen, 'CGLX', zero)
         cglen(2) = gtrmf(comlyn, comlen, 'CGLY', zero)
         cglen(3) = gtrmf(comlyn, comlen, 'CGLZ', zero)
      endif

! fglen
      if ((indx(comlyn, comlen, 'FGLX', 4) .gt. 0) .and. &
           (indx(comlyn, comlen, 'FGLY', 4) .gt. 0) .and. &
           (indx(comlyn, comlen, 'FGLZ', 4) .gt. 0)) then
         fglen(1) = gtrmf(comlyn, comlen, 'FGLX', zero)
         fglen(2) = gtrmf(comlyn, comlen, 'FGLY', zero)
         fglen(3) = gtrmf(comlyn, comlen, 'FGLZ', zero)
      endif

! glen
      if ((indx(comlyn, comlen, 'GLNX', 4) .gt. 0) .and. &
           (indx(comlyn, comlen, 'GLNY', 4) .gt. 0) .and. &
           (indx(comlyn, comlen, 'GLNZ', 4) .gt. 0)) then
      glen(1) = gtrmf(comlyn, comlen, 'GLNX', zero)
      glen(2) = gtrmf(comlyn, comlen, 'GLNY', zero)
      glen(3) = gtrmf(comlyn, comlen, 'GLNZ', zero)
      endif

! grid
      if ((indx(comlyn, comlen, 'GRDX', 4) .gt. 0) .and. &
           (indx(comlyn, comlen, 'GRDY', 4) .gt. 0) .and. &
           (indx(comlyn, comlen, 'GRDZ', 4) .gt. 0)) then
         grid(1) = gtrmf(comlyn, comlen, 'GRDX', zero)
         grid(2) = gtrmf(comlyn, comlen, 'GRDY', zero)
         grid(3) = gtrmf(comlyn, comlen, 'GRDZ', zero)
      endif

! chgm/setchgm
! defaults to spl2 (1)
      chgm = gtrmi(comlyn, comlen, 'CHGM', 1)

! setup for mg-para
! pdime :: 
! defaults to 0 which is not correct
      pdime(1) = gtrmi(comlyn, comlen, 'PDIX', pdime(1))
      pdime(2) = gtrmi(comlyn, comlen, 'PDIY', pdime(2))
      pdime(3) = gtrmi(comlyn, comlen, 'PDIZ', pdime(3))

! ofrac :: overlap fraction between procs
! defaults to 0.1
      ofrac = gtrmf(comlyn, comlen, 'OFRA', ofrac)

! do we calculate solvation forces?
      qfapbs = (indxa(comlyn, comlen, 'SFORCE') .gt. 0)

! if yes, skip the first APBS calculation, just set up all variables
      if (qfapbs) then
         skip = .true.
      endif

! how often do we calculate forces in md?
! default is 1 (every step)
      napbs = gtrmi(comlyn, comlen, 'UPDATE', 1)

! forces update method
      umeth = gtrmi(comlyn, comlen, 'UMETHOD', 1)

! get atom positions, radius and charge from a common block (coord.fcm)
!
!      positionx = x
!      positiony = y
!      positionz = z
!      radius = wmain
!      charge = cg

! save wmain radii to a_radius
      do i=1, natom
         a_radius(i) = wmain(i)
      end do

! debug
      apbs_debug = gtrmi(comlyn, comlen, 'DEBUG', apbs_debug)

! ok, we now have all user switches 
      qaparsed = .true.

      i_param(1) = calc_type
      i_param(2) = nlev
      i_param(3) = cmeth
      i_param(4) = ccmeth
      i_param(5) = fcmeth
      i_param(6) = chgm
      i_param(7) = nonlin
      i_param(8) = bcfl
      i_param(9) = srfm
      i_param(10) = calcenergy
      i_param(11) = calcforce
      i_param(12) = wpot
      i_param(13) = wchg
      i_param(14) = wsmol
      i_param(15) = wkappa
      i_param(16) = wdiel
      i_param(17) = 0
      i_param(18) = 0
      i_param(19) = 0
      i_param(20) = 0
      i_param(21) = calcnpenergy
      i_param(22) = nion
      i_param(23) = rchg
      i_param(24) = rkappa
      i_param(25) = rdiel

      r_param(1) = pdie
      r_param(2) = sdie
      r_param(3) = srad
      r_param(4) = swinapbs
      r_param(5) = tempapbs
      r_param(6) = sdens
      r_param(7) = gamma
      r_param(8) = smvolume
      r_param(9) = smsize

! go over atomic coordinates and figure out optimal grid size
      maxx = x(1)
      minx = x(1)
      maxy = y(1)
      miny = y(1)
      maxz = z(1)
      minz = z(1)
      do i = 1, natom
         if(maxx < x(i)+a_radius(i)) maxx = x(i)+a_radius(i)
         if(minx > x(i)-a_radius(i)) minx = x(i)-a_radius(i)
         if(maxy < y(i)+a_radius(i)) maxy = y(i)+a_radius(i)
         if(miny > y(i)-a_radius(i)) miny = y(i)-a_radius(i)
         if(maxz < z(i)+a_radius(i)) maxz = z(i)+a_radius(i)
         if(minz > z(i)-a_radius(i)) minz = z(i)-a_radius(i)
      end do

      write(outu,'(3x, a, 3f8.3)') 'APBS> Molecular dimensions: ', &
           maxx-minx, maxy-miny, maxz-minz

! for mg-manual calculate missing grid parameters
       if ((i_param(1)==0 .OR. i_param(1)==1) .and. dime(1)==0) then
          cglen(1) = 1.7 * (maxx-minx)
          cglen(2) = 1.7 * (maxy-miny)
          cglen(3) = 1.7 * (maxz-minz)
          fglen(1) = 20.0 + (maxx-minx)
          fglen(2) = 20.0 + (maxy-miny)
          fglen(3) = 20.0 + (maxz-minz)

          do i = 1, 3
             if (fglen(i) > cglen(i)) cglen(i) = fglen(i)
          end do
          write(outu, '(3x, a)') &
               'APBS> Grid dime not specified, calculating ...'
          write(outu, '(3x, a)') &
               'APBS> Requesting dime re-calculation on the fly'
          qdime_updates = .TRUE.

          do i = 1, 3
             dime(i) = 32*(int((int((fglen(i)/grid(i)) + 0.5) - 1)/32.0 &
                  + 0.5)) + 1
             if (dime(i) < 33) dime(i) = 33
          end do
       end if

       if (apbs_debug .gt. 0) then
          write(outu, '(3x, a)') 'APBS> Grid values: '
          write(outu, '(3x, a, 3f8.3)') 'APBS> fglen: ', fglen(1), &
               fglen(2), fglen(3)
          write(outu, '(3x, a, 3f8.3)') 'APBS> cglen: ', cglen(1), &
               cglen(2), cglen(3)
          write(outu, '(3x, a, 3i4)')   'APBS> dime: ', dime(1), &
               dime(2),dime(3)
          write(outu, '(3x, a, 3f8.3)') 'APBS> grid: ', grid(1), &
               grid(2),grid(3)
          write(outu, '(3x, a, f10.3)') &
               'APBS> Required memory (in MB): ',dime(1)*dime(2) &
               *dime(3)*200.0/1024/1024
       end if

       do i = 1, 3
          if (grid(i) * dime(i) < fglen(i)) then
             write(outu,  '(3x, a)') &
                  'APBS> WARNING: caclulated grid spacing is larger than requested:'
             write(outu,  '(3x, a, i1, a, f5.3, a, f5.3)') &
                  'APBS> (', i, '): requested: ', grid(i), ' actual: ', &
                  fglen(i)/dime(i)
!             write(outu,  '(3x, a)')
!     +            'APBS> To fix this decrease grd value'
!             call wrndie(-1, '<APBS>',
!     +            'Requested and calculated grid spacing discrapancy')
          end if
       end do

! OK, now we are ready to call the apbs_driver and start the show

       if (skip) then
          write(outu, '(3x, a)') &
               "APBS> Skiping the first APBS calculation, "
          write(outu, '(3x, a)') &
               "APBS> Initializing parameters only."
       else
         rc = apbsdrv(natom, x, y, z, a_radius, cg, &
              r_param, i_param, &
              grid, dime, pdime, glen, center, &
              cglen, fglen, ccenter, fcenter, &
              ofrac, apbs_debug, &
              ionq, ionc, ionrr, &
              esenerg, npenerg, &
              apbsdx, apbsdy, apbsdz, &
              apbsqfx, apbsqfy, apbsqfz, apbsibx, apbsiby, apbsibz, &
              apbsnpx, apbsnpy, apbsnpz, apbsdbx, apbsdby, apbsdbz)

         if (apbs_debug .gt. 0) then
            write(outu, '(3x, a, i3)') "APBS> APBS return code: ", rc
         end if

         if ((apbs_debug .gt. 0) .and. (calcforce .gt. 0)) then
            do i = 1, natom
               write(outu, '(3x, a, i8, 3f13.5, a)') &
                    "APBS> Total force on atom", i, apbsdx(i), apbsdy(i) &
                    , apbsdz(i), " kJ/(mol/A)"
            end do
         endif

         if (apbs_debug .gt. 0) then
            write(outu, '(3x, a, f13.5, a)') "APBS> esEnergy: ", &
                 esenerg(1)/ 4.2D0, " kcal/mol"
            write(outu, '(3x, a, f13.5, a)') "APBS> npEnergy: ", &
                 npenerg(1)/ 4.2D0, " kcal/mol"
            write(outu, '(3x, a, f13.5, a)') "APBS> Total Energy: ", &
                 (esenerg(1) + npenerg(1))/ 4.2D0, " kcal/mol"
         endif


! pass the electrostatic energy value back (in kcal/mol)
         call set_param('ENPB', esenerg(1) / 4.2D0)
      endif

      RETURN
   END SUBROUTINE APBS

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
   subroutine apbsfrc(f_natom, x, y, z, f_cg, enelec, ennp, &
        dx, dy, dz, icall, qprin)
!
! calculates forces by doing two apbs calculation (vacuum/solvent)
!
! this is called by energy.src (SP APBS calculation must be done 
! first, before starting MD to initialize all values)
!-----------------------------------------------------------------------
! numbers definitions
  use number
  use comand
! natom, cg (charge)
  use psf
  use stream
  use memory
  use timerm
  use parallel
!-----------------------------------------------------------------
! local variables
      real(chm_real) apbsdx(natom), apbsdy(natom), apbsdz(natom)
      real(chm_real) solvdx(natom), solvdy(natom), solvdz(natom)
      real(chm_real) vacdx(natom), vacdy(natom), vacdz(natom)
      real(chm_real) dx(natom), dy(natom), dz(natom)
      real(chm_real) x(natom), y(natom), z(natom)
      real(chm_real) enelec, ennp, f_cg(natom) 
      real(chm_real) esenvac, esensolv, npenvac, npensolv
      real(chm_real) esenerg(15), npenerg(15) 
      real(chm_real) sdie, gamma, ionc1, ionc2
      real(chm_real) maxx, minx, maxy, miny, maxz, minz
      real(chm_real) apbsqfx(natom), apbsqfy(natom), apbsqfz(natom)
      real(chm_real) apbsibx(natom), apbsiby(natom), apbsibz(natom)
      real(chm_real) apbsnpx(natom), apbsnpy(natom), apbsnpz(natom)
      real(chm_real) apbsdbx(natom), apbsdby(natom), apbsdbz(natom)
      real(chm_real) solvqfx(natom), solvqfy(natom), solvqfz(natom)
      real(chm_real) solvibx(natom), solviby(natom), solvibz(natom)
      real(chm_real) solvnpx(natom), solvnpy(natom), solvnpz(natom)
      real(chm_real) solvdbx(natom), solvdby(natom), solvdbz(natom)
      real(chm_real) vacqfx(natom), vacqfy(natom), vacqfz(natom)
      real(chm_real) vacibx(natom), vaciby(natom), vacibz(natom)
      real(chm_real) vacnpx(natom), vacnpy(natom), vacnpz(natom)
      real(chm_real) vacdbx(natom), vacdby(natom), vacdbz(natom)

      integer f_natom, i, apbsdrv, icall, rc
      logical qprin
!--------------------------------------------------------------------
!
! forces: two apbs calculations (one in solvent, the other in vacuum),
! solvation force is the difference 
!

! did we parse options first?
      if (.not. qaparsed) then
         call wrndie(-5, '<APBSFRC>', 'Must have APBS options first!')
      endif

! we're doing this calculation only every icall/napbs step
      if (mod(icall, napbs) .eq. 0) then

! must turn on forces in apbs!
         if (i_param(11) .ne. 2 ) then
            call wrndie(-5, '<APBSFRC>', &
                 'Set CALCF to 2 for solvation forces calculation!')
         end if

         if (qdime_updates) then
            maxx = x(1)
            minx = x(1)
            maxy = y(1)
            miny = y(1)
            maxz = z(1)
            minz = z(1)
            do i = 1, natom
               if(maxx < x(i)+a_radius(i)) maxx = x(i)+a_radius(i)
               if(minx > x(i)-a_radius(i)) minx = x(i)-a_radius(i)
               if(maxy < y(i)+a_radius(i)) maxy = y(i)+a_radius(i)
               if(miny > y(i)-a_radius(i)) miny = y(i)-a_radius(i)
               if(maxz < z(i)+a_radius(i)) maxz = z(i)+a_radius(i)
               if(minz > z(i)-a_radius(i)) minz = z(i)-a_radius(i)
            end do

            cglen(1) = 1.7 * (maxx-minx)
            cglen(2) = 1.7 * (maxy-miny)
            cglen(3) = 1.7 * (maxz-minz)
            fglen(1) = 20.0 + (maxx-minx)
            fglen(2) = 20.0 + (maxy-miny)
            fglen(3) = 20.0 + (maxz-minz)

            do i = 1, 3
               if (fglen(i) > cglen(i)) cglen(i) = fglen(i)
            end do

            if (apbs_debug .gt. 0) then
               write(outu,'(3x, a, 3f8.3)') &
                    'APBS> Molecular dimensions: ',maxx-minx, maxy-miny, &
                    maxz-minz
               write(outu, '(3x, a)') &
                    'APBS> Re-calculating grid dimensions ...'
            end if

            do i = 1, 3
               dime(i) = 32*(int((int(fglen(i)/grid(i) + 0.5) - 1)/32.0 &
                    + 0.5)) + 1
               if (dime(i) < 33) dime(i) = 33
            end do
         end if

         if (apbs_debug .gt. 0) then
            write(outu, '(3x, a)') 'APBS> Grid values: '
            write(outu, '(3x, a, 3f8.3)') 'APBS> fglen: ', fglen(1), &
                 fglen(2), fglen(3)
            write(outu, '(3x, a, 3f8.3)') 'APBS> cglen: ', cglen(1), &
                 cglen(2), cglen(3)
            write(outu, '(3x, a, 3i4)')   'APBS> dime: ', dime(1), &
                 dime(2),dime(3)
            write(outu, '(3x, a, 3f8.3)') 'APBS> grid: ', grid(1), &
                 grid(2),grid(3)
            write(outu, '(3x, a, f10.3)') &
                 'APBS> Required memory (in MB): ',dime(1)*dime(2) &
                 *dime(3)*200.0/1024/1024
         end if

!         write(*,*)'APBSFRC> before apbsdrv...'
! first calculation - in solvent 
         rc = apbsdrv(f_natom, x, y, z, a_radius, f_cg, &
              r_param, i_param, &
              grid, dime, pdime, glen, center, &
              cglen, fglen, ccenter, fcenter, &
              ofrac, apbs_debug, &
              ionq, ionc, ionrr, &
              esenerg, npenerg, &
              apbsdx, apbsdy, apbsdz, &
              apbsqfx, apbsqfy, apbsqfz, apbsibx, apbsiby, apbsibz, &
              apbsnpx, apbsnpy, apbsnpz, apbsdbx, apbsdby, apbsdbz)

!         write(*,*)'APBSFRC> after apbsdrv...,ncalc=',ncalc(1)

! total energy in solvent (in kcal/mol)
         esensolv = esenerg(1) / 4.2D0
         npensolv = npenerg(1) / 4.2D0
!         write(*,*)'APBSFRC>after energy...'
         if (apbs_debug .gt. 0) then
            write(outu, '(3x, a, f8.3, a)') &
                 "APBSFRC> Electrostatic energy in solvent: ", &
                 esensolv, " kcal/mol"
            write(outu, '(3x, a, f8.3, a)') &
                 "APBSFRC> Nonpolar energy in solvent: ", &
                 npensolv, " kcal/mol"
         end if

! get the total forces from the solvent calculation
         do i = 1, f_natom
            solvdx(i) = apbsdx(i) / 4.2D0
            solvdy(i) = apbsdy(i) / 4.2D0
            solvdz(i) = apbsdz(i) / 4.2D0

            solvqfx(i) = apbsqfx(i) / 4.2D0
            solvqfy(i) = apbsqfy(i) / 4.2D0
            solvqfz(i) = apbsqfz(i) / 4.2D0

            solvibx(i) = apbsibx(i) / 4.2D0
            solviby(i) = apbsiby(i) / 4.2D0
            solvibz(i) = apbsibz(i) / 4.2D0

            solvnpx(i) = apbsnpx(i) / 4.2D0
            solvnpy(i) = apbsnpy(i) / 4.2D0
            solvnpz(i) = apbsnpz(i) / 4.2D0

            solvdbx(i) = apbsdbx(i) / 4.2D0
            solvdby(i) = apbsdby(i) / 4.2D0
            solvdbz(i) = apbsdbz(i) / 4.2D0
         end do

!-----------------------------------------------------------------

! save sdie and ion concentration
         sdie = r_param(2)
         ionc1 = ionc(1)
         ionc2 = ionc(2)
! set sdie = 1.0
! salt concentration should be 0.0
         r_param(2) = 1.0D0 ! sdie
         ionc(1) = 0.0D0
         ionc(2) = 0.0D0

! second calculation, now in vacuum
         rc = apbsdrv(f_natom, x, y, z, a_radius, f_cg, &
              r_param, i_param, &
              grid, dime, pdime, glen, center, &
              cglen, fglen, ccenter, fcenter, &
              ofrac, apbs_debug, &
              ionq, ionc, ionrr, &
              esenerg, npenerg, &
              apbsdx, apbsdy, apbsdz, &
              apbsqfx, apbsqfy, apbsqfz, apbsibx, apbsiby, apbsibz, &
              apbsnpx, apbsnpy, apbsnpz, apbsdbx, apbsdby, apbsdbz)

! return back the original sdie and ionc concentration values
         r_param(2) = sdie
         ionc(1) = ionc1
         ionc(2) = ionc2

! total energy in vacuum (in kcal/mol)
         esenvac = esenerg(1) / 4.2D0
!         npenvac = npenerg(1) / 4.2D0
! no NP energy in vacuum
         npenvac = 0.0D0
         if (apbs_debug .gt. 0) then
            print *, "APBSFRC> Electrostatic energy in vacuum: ", &
                 esenvac, " kcal/mol"
            print *, "APBSFRC> Nonpolar energy in vacuum: ", &
                 npenvac, " kcal/mol"
         end if

! get the total forces from the vacuum calculation
         do i = 1, f_natom
            vacdx(i) = apbsdx(i) / 4.2D0
            vacdy(i) = apbsdy(i) / 4.2D0
            vacdz(i) = apbsdz(i) / 4.2D0

            vacqfx(i) = apbsqfx(i) / 4.2D0
            vacqfy(i) = apbsqfy(i) / 4.2D0
            vacqfz(i) = apbsqfz(i) / 4.2D0

            vacibx(i) = apbsibx(i) / 4.2D0
            vaciby(i) = apbsiby(i) / 4.2D0
            vacibz(i) = apbsibz(i) / 4.2D0

! the following are zero in vacuum

            vacnpx(i) = 0.0D0
            vacnpy(i) = 0.0D0
            vacnpz(i) = 0.0D0

            vacdbx(i) = 0.0D0
            vacdby(i) = 0.0D0
            vacdbz(i) = 0.0D0

         end do

!-----------------------------------------------------------------

! add calulated total forces to dx, dy, dz in common block
!
         do i = 1, f_natom
            dx(i) = dx(i) - (solvdx(i) - vacdx(i))
            dy(i) = dy(i) - (solvdy(i) - vacdy(i))
            dz(i) = dz(i) - (solvdz(i) - vacdz(i))
         end do

         if (apbs_debug .gt. 1) then
            do i = 1, f_natom
               print *, "APBSFRC> TotalForces:", i, dx(i), dy(i), dz(i)
            end do
            do i = 1, f_natom
               print *, "APBSFRC> SolventForces:", i, solvdx(i), &
                    solvdy(i) , solvdz(i) 
            end do
            do i = 1, f_natom
               print *, "APBSFRC> VacuumForces:", i, vacdx(i), &
                    vacdy(i), vacdz(i)
            end do
            do i = 1, f_natom
               print *, "APBSFRC> SolvForces:", i, solvdx(i) - vacdx(i), &
                    solvdy(i) - vacdy(i), solvdz(i) - vacdz(i)
            end do
            do i = 1, f_natom
               print *, "APBSFRC> qfForces:", i, solvqfx(i) - vacqfx(i), &
                    solvqfy(i) - vacqfy(i), solvqfz(i) - vacqfz(i)
            end do
            do i = 1, f_natom
               print *, "APBSFRC> ibForces:", i, solvibx(i) - vacibx(i), &
                    solviby(i) - vaciby(i), solvibz(i) - vacibz(i)
            end do
            do i = 1, f_natom
               print *, "APBSFRC> npForces:", i, solvnpx(i) - vacnpx(i), &
                    solvnpy(i) - vacnpy(i), solvnpz(i) - vacnpz(i)
            end do
            do i = 1, f_natom
               print *, "APBSFRC> dbForces:", i, solvdbx(i) - vacdbx(i), &
                    solvdby(i) - vacdby(i), solvdbz(i) - vacdbz(i)
            end do

         endif

! total, solvatation energy (in kcal/mol)
         enelec = esensolv - esenvac
         if (apbs_debug .gt. 0) then
            print *, "APBSFRC> Total solvation energy: ", &
                 enelec, " kcal/mol"
         end if

! total, non-polar energy (in kcal/mol)
!         ennp = npensolv - npenvac
         ennp = npensolv
         if (apbs_debug .gt. 0) then
            print *, "APBSFRC> Total non-polar energy: ", &
                 ennp, " kcal/mol"
            print *, "APBSFRC> Total non-polar energy (vacuum): ", &
                 npenvac, " kcal/mol"

         end if

         if (qprin) then
            WRITE(outu,'(3X,A,F13.5,A)') &
                 'The Free Energy of Charging in Solvent  = ', &
                 esensolv,' [KCAL/MOL]'
            WRITE(outu,'(3X,A,F13.5,A)') &
                 'The Free Energy of Charging in vacuum   = ', &
                 esenvac,' [KCAL/MOL]'
            WRITE(outu,'(3X,A,F13.5,A)') &
                 'The Electrostatic Solvation Free Energy = ', &
                 enelec,' [KCAL/MOL]'

            WRITE(outu,'(3X,A,F13.5,A)') &
                 'The Nonpolar Solvation Free Energy = ', &
                 ennp,' [KCAL/MOL]'
         end if

! pass the electrostatic energy value back (in kcal/mol)
! but this doesn't make too much sense in here, does it?
!         call set_param('ENPB', enelec)

! save solvation forces for different update schemes
         do i = 1, f_natom
            solvfrcx(i) = solvdx(i) - vacdx(i)
            solvfrcy(i) = solvdy(i) - vacdy(i)
            solvfrcz(i) = solvdz(i) - vacdz(i)
         end do
         
! save ennp and enelec
         senelec = enelec
         sennp = ennp

! end if section for napbs==icall
! if we don't do APBS calculation just use the old APBS calculated forces
! or update it accordingly
      else
         if (apbs_debug .gt. 0) then
            print *, "APBSFRC> Reusing solvation forces in this step:", &
                 icall
         end if
! select force update method
         if (umeth .eq. 0) then
!     this just uses zero for solvation forces and energies 
!     between updates 
            enelec = 0.0
            ennp = 0.0
         else if (umeth .eq. 1) then
!     this method uses forces from previous APBS step between updates
            do i = 1, f_natom
               dx(i) = dx(i) - solvfrcx(i)
               dy(i) = dy(i) - solvfrcy(i)
               dz(i) = dz(i) - solvfrcz(i)
            end do
            enelec = senelec
            ennp = sennp
         else
            write(outu, '(3x, a)') &
                 "APBSFRC> This method of force updates is not implemented"
         end if

! endif section for napbs/icall comparison
      end if

      RETURN
   end subroutine apbsfrc
#endif /* (apbs)*/
end module iapbs

