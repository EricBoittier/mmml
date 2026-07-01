module solana_mod
  use chm_kinds
  use dimens_fcm
  implicit none

  !  MXSTP  Maximum number of timesteps for correlation functions
  !  INORM(MXSTP)  Normalization counter for correlation functions
  !  MINI  Logical flag indicating minimum image pbc usage
  !  VAC, VNORM  arrays for values of correlation functions
  !  aBOX(aBOX2)  Boxlength (/2) for each dimension when pbcs used in analysis
#if KEY_NOMISC==0 /*solana_fcm*/
  !
  real(chm_real),dimension(50),save :: FRAD, F2RAD
  !
  real(chm_real),save :: XBOX, YBOX, ZBOX, XBOX2, YBOX2, ZBOX2
  LOGICAL,save :: MINI,JUNK

  !     The solvent analysis common blocks.

  INTEGER,PARAMETER :: MXSTP=600
  real(chm_real),save ::  VAC, VNORM 

  !-----------------------------------------------------------------------
  ! COOR ANAL data
  !
  !    IRLP    -  Rank of Legendre polynom
  !    MRLP    -  Maximum Rank of Legendre polynom
  !
  INTEGER IRLP, MRLP
#endif /* (solana_fcm)  NOMISC*/


contains
  
#if KEY_NOMISC==0
  SUBROUTINE SOLANA(QWEIG)
    !-----------------------------------------------------------------------
    !     Sets up atom selection arrays for analysis of
    !     dynamics around a specific protein site.
    !
    !     SYNTAX:
    !     > COOR ANALysis {WATer} RLP <int>  <selection-syntax> -
    !     {XREF <real> YREF <real> ZREF <real>} {CROS} -
    !     {SITE {MULTiple} <selection-syntax> }
    !
    !     This routine parses command stream information and sets up
    !     solvent and site selections.
  use number
  use comand
  use coord
  use ctitla
  use psf
  use chutil,only:makind
  use stream
  use memory
  use select
  use string

    integer,allocatable,dimension(:),target :: space_flg
    INTEGER,pointer,dimension(:) :: FLGSOL=>null(),FLGSIT=>null(), &
         FLGEXV=>null(),SITE=>null()
    INTEGER :: NSITE, NSOLV, NEXV, NWARN
    real(chm_real) :: REF(3)
    LOGICAL QWEIG
    !
    !     Local
    LOGICAL     DONE, CROS, WAT, LMULTI
    INTEGER     I
    character(len=4) WRD
    !
    call chmalloc('solana.src','solana','space_flg',4*natom,intg=space_flg)
    flgsol => space_flg(1:natom)
    flgsit => space_flg(  natom+1:2*natom)
    flgexv => space_flg(2*natom+1:3*natom)
    site   => space_flg(3*natom+1:4*natom)

    !     Initialize varibles
    NSITE = 0
    NEXV = 0
    REF(1) = ZERO
    REF(2) = ZERO
    REF(3) = ZERO
    FLGSIT = ZERO
    
    !
    DO I=1,NATOM
       SITE(I) = 0
    ENDDO
    !     Command parsing
    WAT = .FALSE.
    DONE = .FALSE.
    LMULTI = .FALSE.
    CROS = .FALSE.
    !
    ! Get specification of solvent molecule; always needed
    CALL SELCTA(COMLYN,COMLEN,FLGSOL,X,Y,Z,WMAIN,.TRUE.)
    NSOLV=NSELCT(NATOM,FLGSOL)
    IF(PRNLEV >= 2) WRITE (OUTU,'(I5,A)')NSOLV, &
         ' SOLVENT MOLECULES SELECTED'

10  CONTINUE
    WRD = NEXTA4(COMLYN,COMLEN)
    IF (WRD == 'SOLV'.OR. WRD.EQ.'FINI'.OR.WRD.EQ.'SPEC') THEN
       ! Obsolete but tolerated
       !
    ELSE IF (WRD == 'WATE') THEN
       WAT = .TRUE.
    ELSE IF (WRD == 'CROS') THEN
       !ab...13-Aug-93, Add cross solvent analysis command
       CROS = .TRUE.
       IF (WAT) CALL WRNDIE(-2,'<SOLANA>', &
            'CROSs AND WATer ARE EXCLUSIVE')
       !
       ! Get second set of molecules for cross analysis
       CALL SELCTA(COMLYN,COMLEN,FLGSIT,X,Y,Z,WMAIN,.TRUE.)
       NSITE = NSELCT(NATOM,FLGSIT)
       IF(PRNLEV >= 2) WRITE (OUTU,'(A,2(I5,A))') &
            ' ANALYSIS BETWEEN ',    NSOLV, &
            ' SOLVENT ATOMS A AND ', NSITE, &
            ' SOLVENT ATOMS B'
       !
    ELSE IF (WRD == 'XREF') THEN
       REF(1) = NEXTF(COMLYN,COMLEN)
    ELSE IF (WRD == 'YREF') THEN
       REF(2) = NEXTF(COMLYN,COMLEN)
    ELSE IF (WRD == 'ZREF') THEN
       REF(3) = NEXTF(COMLYN,COMLEN)
    ELSE IF (WRD == 'MULTI') THEN
       LMULTI = .TRUE.
    ELSE IF (WRD == 'RLP') THEN
       IRLP = NEXTI(COMLYN,COMLEN)
       IF(IRLP > MRLP) CALL WRNDIE(-5,'<SOLANA>','Rank too big')
       IF((PRNLEV >= 2).AND.(IRLP > 1)) WRITE(OUTU,'(A,I5)') &
            ' Rank of Legendre Polynomial is ',IRLP
    ELSE IF (WRD == 'SITE') THEN
       IF (CROS) CALL WRNDIE(-2,'<SOLANA>', &
            'CROSs AND SITE ARE EXCLUSIVE')
       LMULTI = (INDXA(COMLYN,COMLEN,'MULT') /= 0)
       CALL SELCTA(COMLYN,COMLEN,FLGSIT,X,Y,Z,WMAIN,.TRUE.)
       !
       !     Set up site pointers
       NSITE = 0
       DO I=1,NATOM
          IF(FLGSIT(I) == 1) THEN
             NSITE = NSITE + 1
             SITE(NSITE) = I
          ENDIF
       ENDDO
       IF(PRNLEV >= 2) THEN
          IF(.NOT.LMULTI) THEN
             WRITE (OUTU,'(I5,A,A)')NSITE, &
                  ' SITES HAVE BEEN SELECTED TO CONSTRUCT', &
                  ' REFERENCE POINT'
          ELSE
             WRITE (OUTU,'(I5,A,A)')NSITE, &
                  ' MULTIPLE SITES HAVE BEEN SELECTED TO', &
                  ' CONSTRUCT DISTRIBUTIONS FOR'
          ENDIF
       ENDIF
       IF(NSITE > NATOM) THEN
          IF(WRNLEV >= 2) WRITE (OUTU,'(A,I5)') &
               'NSITE GREATER THAN NATOM ',NATOM
          CALL DIEWRN(-2)
       ENDIF
    ELSE IF (WRD == 'EXVC') THEN
       CALL SELCTA(COMLYN,COMLEN,FLGEXV,X,Y,Z,WMAIN,.TRUE.)
       ! Warn if the site around which g(r) is wanted is also contained in the
       ! selected set for the excluded volume.
       NWARN=0
       DO I=1,NSITE
          IF(FLGEXV(SITE(I)) == 1) NWARN=NWARN+1
       ENDDO
       CALL MAKIND(NATOM,FLGEXV,FLGEXV,NEXV)
       IF(PRNLEV >= 2)THEN
          WRITE(OUTU,'(I6,A)') NEXV, &
               ' SITES USED FOR EXCLUDED VOLUME CORRECTION'
          IF(NWARN > 0) WRITE(OUTU,'(I10,A/)') &
               NWARN,' ATOMS FOUND BOTH IN SITE and EXCV SETS !!!'
       ENDIF
    ELSE IF (WRD == '    ') THEN
       DONE = .TRUE.
    ELSE
       CALL JOINWD(COMLYN,MXCMSZ,COMLEN,WRD,4)
       DONE = .TRUE.
    ENDIF
    IF (.NOT.(DONE)) GOTO 10
    !
    IF(NSITE == 0 .AND. WRNLEV >= 2) WRITE (OUTU,'(A,3(F8.2))') &
         'NO SITE CHOSEN. REFERENCE POINT FOR ANALYSIS:',REF
    !
    CALL SOLANL(COMLYN,COMLEN,FLGSOL,NSOLV,WAT,CROS,REF,FLGSIT, &
         NSITE,SITE,NATOM,NEXV,FLGEXV,LMULTI,WMAIN,QWEIG)
    !
    call chmdealloc('solana.src','solana','space_flg',4*natom,intg=space_flg)
    nullify(FLGSOL,FLGSIT,FLGEXV,SITE)
    RETURN
  END SUBROUTINE SOLANA

  SUBROUTINE SOLANL(COMLYN,COMLEN,FLGSOL,NSOLV,WAT,CROS,REF,FLGSIT, &
       NSITE,SITE,MXSITE,NEXV,FLGEXV,LMULTI,WMAIN,QWEIG)
    !-----------------------------------------------------------------------
    !     This program does an analysis of solvent dynamics around a
    !     particular reference point specified either by the coordinates
    !     of atoms in SITE or specified by a particular reference, REF.
    !     The program computes (optionally) the solvent velocity
    !     autocorrelation function, mean-square displacement function,
    !     solvent-solvent radial distribution functions and
    !     solvent-reference site radial distribution function,
    !     and the solvent - reference site deformable boundary force.
    !
    !     The required inputs are:
    !     ISPHER = 0 if all configurations are analyzed - not used
    !     ISPHER = 4 if only atoms within RSPHER are to be considered -
    !     not used.
    !     FIRSt    = first trajectory file unit number
    !     NUNIt    = number of files to read
    !     BEGIn    = number of first dynamics step to be read
    !     STOP     = number of last dynamics step to be read
    !     SKIP     = number of dynamics steps to skip between calculations
    !     NCORs    = number of steps to compute vac or msd
    !     RSPIn    = inner radius for vac,msd, analysis around REF
    !     RSPOu    = outer radius for vac,msd, analysis around REF
    !     RDSP     = radius of dynamics sphere, used for densities,kirkwood and dbf
    !     DR       = grid spacing for analysis of rdf's
    !     RRSPhere = radius for rdf analysis (keyword RSPHere is also allowed)
    !                this radius will also be used as default for RDSP
    !     BYGRoup,BYREsidue,BYSEgment = do not use atoms in the same
    !                group,residue or segment for rdf calculations
    !                default: don't care;
    !     MGN      = number of points in g(r) curve
    !     RCUT     = radius of interaction sphere in dbf calculation
    !     ZP0      = initial reference site - dynamics sphere origin separation
    !     NZP      = number of separations to compute dbf
    !     TYP      = for DBF calc 1=oxygen, 1=hydrogen
    !C Following is being added /not yet fully tested!!/. Dec96-June98, LNI
    !     IHIS     = unit for output of 3Dhistogram data, with options:
    !          WEIG     use WMAIN to weight points
    !          DIPO     accumulate dipole vector density
    !          CHARge   accumulate charge density
    !                   default is to just accumulated number density of sel. atoms
    !          NORM value  densities are divided by this value
    !                      (and by number of frames)
    !          XMIN,XMAX,DX
    !          YMIN,YMAX,DY grid dimension&spacing (default +/- 20A,0.5A spacing)
    !          ZMIN,ZMAX,DZ
    !          IPDB = unit for output of "atoms" where ABS(density) exceeds
    !          THREshold value for output of atoms in PDB file format
    !
    !     The atoms indicated by the solvent selection are analyzed. If dipole
    !     data is to be analyzed the selection should contain 1 atom/group - the
    !     groups define what atoms are to be used for the dipole calculation.
    !     This could be automated; also need minimum image combined with orienting
    !     function; strongly recommended that the trajectory is processed
    !     by a MERGE RECENTER ORIENT befor this analysis is undertaken.
    !     The 3Dhistogram is really a density and will be written to the
    !     IHIS file in the crystallographic DN6 electron density format
    !C  IN PROGRESS....
    !
    ! CHANGE OCTOBER 96,L. Nilsson:
    !     The following keywords are used to indicate that a given analysis
    !     is to be performed, and the value is the file unit for output.
    !     A missing keyword (or value <= 0) means not to do the analysis.
    !     IVAC    vac analysis
    !     IGDIst  solvent-solvent rdf analysis; unit for O-O g(r) required
    !            IHH  unit for H-H g(r), IOH unit for O-H g(r)   if needed
    !     ISDIst  solvent-site rdf analysis
    !     IMSD    msd analysis
    !     IMRD    Magnetic Relaxation Dispersion analysis
    !         RRES  cutoff radius for calculation of residence time. if 0 use shell
    !               beteween RSPIN, RSPOUT
    !     IKIRkg  dipole analysis for water solvent
    !     RKIRk   distance dependent Kirkwood factor for water
    !             iff a SITE MULTI selection containing at least two atoms is
    !             given, then a unit-vector pointing from the first to
    !             the second site atoms  will be used in the
    !             scalar product with a unit vector along the water dipoles
    !     NKIRk   number of points in r-dimension for IKIR and RKIR
    !             from r=0 to r=RDSP
    !     IFDBf   deformable boundary force calculation
    !     IFDT    Time dependent dbf (see FBOUND for details)
    !     IDENs   density printout
    !
    !     IFMIN  = 1 (0) periodic boundaries are (aren't) in effect
    !              This is just a flag which when present takes value 1
    !     XBOX   = dimension of simulation box in x direction
    !     YBOX   = dimension of simulation box in y direction
    !     ZBOX   = dimension of simulation box in z direction
    !     NOTE: The above dimensions ar taken from trajectory stored
    !           informtion for crystal runs (w/ charmm22 or later) ln nov-93
    ! NOTE: Trajectory reading changed to be done through READCV as in other
    ! parts of charmm. This leads to: DT, DTC obsolete (taken from traj.),
    ! trajectory files are specified with usual keywords (FIRSTU...) so
    ! old restriction of files 20-29 (30-39) is removed, BEGIN,SKIP,STOP
    ! are now properly processed. For now EITHER velocity analysis OR
    ! coordinate analysis has to be chosen (ie, IVAC means ONLY the
    ! velocity autocorrelation will be computed); no big restriction, which
    ! could be overcome if somebody wants it.
    ! Also make file unit information input mandatory, using keywords above
    ! Lennart Nilsson, October 96
    !     ** Note IVAC excludes all other analysis indicators
    !        (IGDIST,ISDIST,IMSD,IKIRKG,IFDBF)
    !     ** Note IGDIST and ISDIST are mutually exclusive flags **
    !
    !     functions are also printed onto fortran unit 6 as output.
    !
    !**********************************************************************C
    !                                                                      C
    !     Written by C. L. Brooks III                                      C
    !     Spring, 1983                                                     C
    !                                                                      C
    !**********************************************************************C

  use clcg_mod, only: rngmodseeds
  use cheq, only: qcg
  use consta
  use number
  use ctitla
  use cvio
  use param
  use psf
  use stream
  use string
  use rndnum
  use parallel
  use memory
  use param_store, only: set_param

    character(len=*) COMLYN
    INTEGER COMLEN
    !     main
    real(chm_real) RSPHER, RSPIN, RSPOUT, RSP, DR, DT, DENS, DENS2
    real(chm_real) RCUT, RRES, ZP0, DZP, WMAIN(*)
    INTEGER NCDBF, NZP, IFDBF, TYP,IFDBFT
    INTEGER NCORS
    INTEGER NFIRST, NSTEP, NSKIP,FIRSTU,NUNIT
    INTEGER NOXYG
#if KEY_CHEQ==1
    INTEGER LL,LC   
#endif

    INTEGER NSAVC
    real(chm_real)    DELTA
    character(len=4) HDR
    !
    INTEGER NSOLV, NSITE, MXSITE, NEXV
    real(chm_real),allocatable,dimension(:,:) :: sref
    INTEGER FLGSOL(NATOM), FLGSIT(NATOM), SITE(MXSITE), FLGEXV(NATOM)
    LOGICAL WAT, CROS, LMULTI,QFIRST,VEL,ERRORC,QMRD,LWEIG
    real(chm_real) REF(3)

    INTEGER IFMIN
    !
    INTEGER NAA,NAR,NRR,IHYDN,NCONFH
    real(chm_real)  RHYD,TMP
    !
    INTEGER MX, MOX2, MGN, I, J, ISTP, ISTEP,NDEGF,NFILE,NFRAMES
    INTEGER ICNTRL(20)
    !
    integer  istats,nfreat
    real(chm_real),dimension(:),allocatable ::  x,y,z,vac, vnorm, tim
    real(chm_real4),dimension(:),allocatable ::  temp
    integer,allocatable,dimension(:) :: freeat,inorm
    real(chm_real),allocatable,dimension(:,:) :: hx1, hx2, ox2
    !
    integer nkirk
    integer,allocatable,dimension(:) :: ikrkd
    integer,allocatable,dimension(:) :: gnoo, gnoh, gnhh, gnos
    integer ngoo, ngoh, nghh,ngos, ndens, noxdis(61)
    integer ndens2,nconf2
    integer ivac, igdist, isdist, imsd, ikirkg,idens,ihh,ioh
    integer ikirkr
    real(chm_real),allocatable,dimension(:,:) :: dico,mj
    real(chm_real),allocatable,dimension(:) :: dicor
    real(chm_real),allocatable,dimension(:) :: atcor, atcor1,atcor2
    !
#if KEY_CHEQ==1
    INTEGER QVAC                                
#endif
#if KEY_CHEQ==1
    integer,allocatable,dimension(:) :: IVXP    
#endif
    integer,allocatable,dimension(:) :: igrat
    INTEGER IHIST,NGRAT,NXPT,NYPT,NZPT,NDI,NHIST
    integer,allocatable,dimension(:) :: isolv
    real(chm_real4),allocatable,dimension(:,:,:,:) :: hist
    real(chm_real),allocatable,dimension(:),target :: space_rl0
    real(chm_real),allocatable,dimension(:),target :: space_rl1
    real(chm_real),pointer,dimension(:) :: acv=>null(),rs=>null()
    real(chm_real),allocatable,dimension(:,:) :: solute
    real(chm_real),allocatable,dimension(:) :: vvcg
    INTEGER IPDB,ISEED
    integer,allocatable,dimension(:) :: SOLVSET,SITESET
    INTEGER NMCPTS,NPTS,IMRD
    real(chm_real)  XMIN,XMAX,XSTP,YMIN,YMAX,YSTP, &
         ZMIN,ZMAX,ZSTP,THRS,RNORM
    real(chm_real)  DRMC,RPROBE,RDMAX,RKNORM
    LOGICAL QDIP,QWEIG,QCHG,QBYGRP,QBYRES,QBYSEG,QINTRA
#if KEY_FLUCQ==1
    INTEGER IDIP,NUMD,DCNT
    integer,allocatable,dimension(:) :: ddis
    real(chm_real) MIND,MAXD
#endif 

    INTEGER IROTCO,JSOL,NWAT,ROUT,MTOT
    real(chm_real) TLOW,TUP
    real(chm_real) RLIM
    real(chm_real),allocatable,dimension(:,:) :: HHXV,HHYV,HHZV,OHXV,OHYV,OHZV
    INTEGER MAXTOT
    INTEGER NTOT

    real(chm_real),allocatable,dimension(:,:) :: TX,TX1,TX2
    real(chm_real),allocatable,dimension(:) :: TCO,TCO1,TCO2
    LOGICAL QROTP1,QROTP2

    real(chm_real),allocatable,dimension(:,:) :: VX,VY,VZ
    integer,allocatable,dimension(:) :: ISTACK, lrngseeds
    logical qpresent
    integer :: IUNIT

    character(len=8) NAME(2)
    character(len=4) HDR1,HDR2
    DATA NAME / 'oxygen  ','hydrogen'/
    DATA HDR1,HDR2 /'CORD','VELD'/

    LOGICAL QCRYS

    !     begin
    !
    QFIRST=.TRUE.
    !
    call chmalloc('solana.src','SOLANL','X',NATOM,crl=space_rl0)
    call chmalloc('solana.src','SOLANL','X',NATOM,crl=X)
    call chmalloc('solana.src','SOLANL','Y',NATOM,crl=Y)
    call chmalloc('solana.src','SOLANL','Z',NATOM,crl=Z)
    call chmalloc('solana.src','SOLANL','TEMP',NATOM,cr4=TEMP)
    call chmalloc('solana.src','SOLANL','FREEAT',NATOM,intg=FREEAT)
    MX = 3*NATOM
    MOX2 = MX/3
    call chmalloc('solana.src','SOLANL','HX1',3,natom,crl=HX1)
    call chmalloc('solana.src','SOLANL','HX2',3,natom,crl=HX2)
    call chmalloc('solana.src','SOLANL','OX2',3,natom,crl=OX2)
    OX2 = ZERO
    !
    !     now for construction of reference SITE(s)
    IF (LMULTI) THEN
       call chmalloc('solana.src','SOLANL','SREF',3,NSITE,crl=SREF)
    ELSE
       call chmalloc('solana.src','SOLANL','SREF',3,1,crl=SREF)
    ENDIF
    !
    !     read dynamics information
    CALL TRJSPC(COMLYN,COMLEN,NUNIT,FIRSTU,NFIRST,NSKIP,NSTEP)

    IF (reallow) THEN         
       IF(NSTEP  <=  0) THEN
          ! We should try to get a good guess for the last step, if user has
          ! not provided us with a stop value
          IUNIT=FIRSTU+NUNIT-1
          CALL GTICNT(IUNIT,HDR,ICNTRL,QREWIND=.TRUE.)
          NSTEP=ICNTRL(2)+ICNTRL(4)-ICNTRL(3)
       ENDIF
    ENDIF           
    !
    ! Now get some information from actual trajectory;
    ! also let READCV check (&fix) NFIRST,NSKIP,NSTEP
    IUNIT=FIRSTU
    ISTATS=1
    CALL READCV(X,Y,Z, &
#if KEY_CHEQ==1
         CG,QCG,                                     & 
#endif
         TEMP,NATOM, &
         FREEAT,NFREAT, &
         FIRSTU,NUNIT,IUNIT,NFILE,ISTP,ISTATS,NDEGF,DELTA, &
         NFIRST,NSTEP,NSKIP,NSAVC,HDR1,HDR2, &
         TITLEB,NTITLB,.FALSE., (/ ZERO /), .true.)
    DT=DELTA*TIMFAC*NSKIP
    IF (reallow)  REWIND(FIRSTU)
    IF (NSTEP <= 0) THEN
        NSTEP=ICNTSV(2)+ICNTSV(4)-ICNTSV(3)
        QCRYS=(ICNTSV(11) == 1)
    ENDIF
    !
    !     read static analysis information
    DR =  GTRMF(COMLYN,COMLEN,'DR',ZERO)
    ! Allow RSPHere keyword spelling as well
    RSPHER = GTRMF(COMLYN,COMLEN,'RSPH',ZERO)
    RSPHER = GTRMF(COMLYN,COMLEN,'RRSP',RSPHER)
    ! If RSPHER is given we will also need RSP in the density calculation
    ! so use some reasonable(?) value if no explict value is given
    RSP = GTRMF(COMLYN,COMLEN,'RDSP',TEN)
    MGN = GTRMI(COMLYN,COMLEN,'MGN',400)
    DENS = GTRMF(COMLYN,COMLEN,'DENS',ZERO)
    DENS2 = GTRMF(COMLYN,COMLEN,'DEN2',ZERO)
    !SJS
    ! LNI adapted to modified hydration number calculation, routine HYDNUM
    ! October 2002
    IHYDN = GTRMI(COMLYN,COMLEN,'IHYD',0)
    RHYD = GTRMF(COMLYN,COMLEN,'RHYD',ZERO)
    RHYD=RHYD*RHYD
    NCONFH=0
    NAA=0
    NRR=0
    NAR=0
    !SJS
    !
    !     read dynamics analysis information
    NCORS = GTRMI(COMLYN,COMLEN,'NCOR',0)
    RSPIN = GTRMF(COMLYN,COMLEN,'RSPI',ZERO)
    RSPOUT = GTRMF(COMLYN,COMLEN,'RSPO',ZERO)
    !
    ! Read EXcludeVolumeCorrection information
    IF(NEXV > 0)THEN
       NMCPTS=GTRMI(COMLYN,COMLEN,'MCP',1000)
       NPTS=GTRMI(COMLYN,COMLEN,'MCSH',MGN)
       DRMC=MGN*DR/NPTS
       RPROBE=GTRMF(COMLYN,COMLEN,'RPRO',ONEPT5)
!
       iseed=3141593
       call chmalloc('solana.src','SOLANL','lrngseeds',Nrand,intg=lrngseeds)
       lrngseeds(1:nrand)=rngseeds(1:nrand)
       if(qoldrandom.or.qbrokenclcg)lrngseeds(1:nrand)=iseed
       call gtrmim(nrand,comlyn,comlen,'ISEE',lrngseeds,rngseeds,qpresent)
       call rngmodseeds(qpresent,iseed)
       call chmdealloc('solana.src','SOLANL','lrngseeds',Nrand,intg=lrngseeds)
!
!       if (.not.qoldrng) then     !yw 05-Aug-2008
!          CALL CLCGINIT(ISEED)
!          ISEED=1
!       endif

       call chmalloc('solana.src','SOLANL','space_rl1',NPTS+nexv,crl=space_rl1)
       acv => space_rl1(1:npts)
       rs => space_rl1(npts+1 : npts+nexv)
       call chmalloc('solana.src','SOLANL','ISOLUTE',3,NEXV,crl=SOLUTE)

       ! Set up radius information for the excluding atoms
       LWEIG=INDXA(COMLYN,COMLEN,'WEIG') > 0
       CALL RADSET(NEXV,FLGEXV,RPROBE,RS,RDMAX,VDWR,IAC,ITC,LWEIG)
    ELSE
       NMCPTS=0
       NPTS=0
    ENDIF
    !     read dbf analysis information
    IFDBF= GTRMI(COMLYN,COMLEN,'IFDB',0)
    IFDBFT= GTRMI(COMLYN,COMLEN,'IFDT',0)
    RCUT = GTRMF(COMLYN,COMLEN,'RCUT',ZERO)
    ZP0 = GTRMF(COMLYN,COMLEN,'ZP0',ZERO)
    NZP = GTRMI(COMLYN,COMLEN,'NZP',0)
    TYP = GTRMI(COMLYN,COMLEN,'TYP',0)
    !
    !     read analysis flags
    !
    !SJS
    !  RCOR- flag for rotational correlation fn
    !
    QROTP1=.FALSE.
    QROTP2=.FALSE.
    IROTCO = GTRMI(COMLYN,COMLEN,'RCOR',0)
    IF(IROTCO > 0)THEN
       MTOT = (NSTEP-NFIRST+NSKIP)/NSKIP
       MAXTOT=GTRMI(COMLYN,COMLEN,'MAXT',512)
       TLOW = GTRMF(COMLYN,COMLEN,'TIML',ONE)
       TUP = GTRMF(COMLYN,COMLEN,'TIMU',FOUR)
       ROUT = GTRMI(COMLYN,COMLEN,'ROUT',0)
       IRLP = GTRMI(COMLYN,COMLEN,'RLP',1)
       IF(IRLP > MRLP) CALL WRNDIE(-5,'<SOLANA>','Rank too big')
       IF((PRNLEV >= 2).AND.(IRLP > 1)) WRITE(OUTU,'(A,I5)') &
            ' Rank of Legendre Polynomial is ',IRLP
       TLOW=TLOW/DT
       TUP=TUP/DT
       QROTP1= INDXRA(COMLYN,COMLEN,'P1',2,.TRUE.) > 0
       QROTP2= INDXRA(COMLYN,COMLEN,'P2',2,.TRUE.) > 0
       IF(QROTP1.OR.QROTP2)THEN
          IF(QROTP1.AND.QROTP2) &
               CALL WRNDIE(-2,'<SOLANA>','Cannot specify both P1 and P2') 
          ! QROT requires WATEr to make sure that we have access to all three coordinates
          ! so turn on if user has forgotten
          WAT=.TRUE.
       ENDIF
    ENDIF
    !SJS
    IVAC = GTRMI(COMLYN,COMLEN,'IVAC',0)
    !
#if KEY_CHEQ==1
    QVAC =  GTRMI(COMLYN,COMLEN,'QVAC',0)    
#endif
    IMRD = GTRMI(COMLYN,COMLEN,'IMRD',0)
    QMRD=(IMRD > 0)
    IF(QMRD) RRES=GTRMF(COMLYN,COMLEN,'RRES',ZERO)
    ! We don't do multisite for the MRD
    IF(QMRD.AND.LMULTI) &
         CALL WRNDIE(-2,'<SOLANA>','MULTI not allowed with MRD')

    !
    IGDIST = GTRMI(COMLYN,COMLEN,'IGDI',0)
    !      IF(IGDIST  >  0)THEN
    ! also find out about HH and OH output (perhaps needed even if not doing IGDI?)
    IHH= GTRMI(COMLYN,COMLEN,'IHH',0)
    IOH= GTRMI(COMLYN,COMLEN,'IOH',0)
    !      ENDIF
    ISDIST = GTRMI(COMLYN,COMLEN,'ISDI',0)
    IMSD =  GTRMI(COMLYN,COMLEN,'IMSD',0)
    IKIRKG = GTRMI(COMLYN,COMLEN,'IKIR',0)
    IKIRKR = GTRMI(COMLYN,COMLEN,'RKIR',0)
    IF(IKIRKG > 0 .OR. IKIRKR .GT. 0) &
         NKIRK = GTRMI(COMLYN,COMLEN,'NKIR',61)
    IDENS =  GTRMI(COMLYN,COMLEN,'IDEN',0)
    !
    ! Check for various INTERmolecular type flags, needed when the following
    ! is true
    QINTRA=(ISDIST > 0 .AND. LMULTI).OR.(IGDIST.GT.0 .AND. CROS)
    QBYGRP= QINTRA .AND. INDXA(COMLYN,COMLEN,'BYGR') > 0
    QBYRES= QINTRA .AND. INDXA(COMLYN,COMLEN,'BYRE') > 0
    QBYSEG= QINTRA .AND. INDXA(COMLYN,COMLEN,'BYSE') > 0
    QINTRA=(QBYGRP.OR.QBYRES.OR.QBYSEG)
    !     read minimum image information
    IFMIN = 0
    IF (INDXA(COMLYN,COMLEN,'IFMI') /= 0) IFMIN = 1
    XBOX =  GTRMF(COMLYN,COMLEN,'XBOX',ZERO)
    YBOX =  GTRMF(COMLYN,COMLEN,'YBOX',ZERO)
    ZBOX =  GTRMF(COMLYN,COMLEN,'ZBOX',ZERO)
    IF(XBOX /= ZERO.OR.YBOX.NE.ZERO.OR.ZBOX.NE.ZERO) IFMIN=1
    ! We don't do multisite for the MRD
    IF(QMRD.AND.IFMIN == 1) &
         CALL WRNDIE(-2,'<SOLANA>','IFMIN not allowed with MRD')

#if KEY_FLUCQ==1
    ! Read dipole distribution information
    IDIP = GTRMI(COMLYN,COMLEN,'IDIP',0)
    IF (IDIP > 0) THEN
       NUMD=GTRMI(COMLYN,COMLEN,'NUMD',100)
       MIND=GTRMF(COMLYN,COMLEN,'MIND',ZERO)
       MAXD=GTRMF(COMLYN,COMLEN,'MAXD',FOUR)
    ENDIF
#endif 
    ! Read histogram information
    IHIST = GTRMI(COMLYN,COMLEN,'IHIS',0)
    IPDB = GTRMI(COMLYN,COMLEN,'IPDB',0)
    IF(IHIST > 0.OR.IPDB.GT.0)THEN
       ! Options
       QDIP=INDXA(COMLYN,COMLEN,'DIPO') > 0
       QCHG=INDXA(COMLYN,COMLEN,'CHAR') > 0
       NDI=1
       IF(QDIP) NDI=3
       ! Normalization etc
       THRS=GTRMF(COMLYN,COMLEN,'THRE',ZERO)
       RNORM=GTRMF(COMLYN,COMLEN,'NORM',ZERO)
       ! Grid information (default to +/- 20A with 0.5A spacing)
       XMAX=TWENTY
       XMIN=-XMAX
       XMIN=GTRMF(COMLYN,COMLEN,'XMIN',XMIN)
       XMAX=GTRMF(COMLYN,COMLEN,'XMAX',XMAX)
       XSTP=GTRMF(COMLYN,COMLEN,'DX',HALF)
       ! if no info for Y&Z assume cubic setup
       YMIN=GTRMF(COMLYN,COMLEN,'YMIN',XMIN)
       YMAX=GTRMF(COMLYN,COMLEN,'YMAX',XMAX)
       YSTP=GTRMF(COMLYN,COMLEN,'DY',XSTP)
       ZMIN=GTRMF(COMLYN,COMLEN,'ZMIN',XMIN)
       ZMAX=GTRMF(COMLYN,COMLEN,'ZMAX',XMAX)
       ZSTP=GTRMF(COMLYN,COMLEN,'DZ',XSTP)
       !
       NXPT=(XMAX-XMIN)/XSTP + 1
       NYPT=(YMAX-YMIN)/YSTP + 1
       NZPT=(ZMAX-ZMIN)/ZSTP + 1
    ENDIF
    !
    ! Everything should be gobbled up by now:
    CALL XTRANE(COMLYN,COMLEN,'SOLANL')
    !
    ! A first check, velocities or coordinates?
    IF(IVAC > 0 .AND. &
         (IGDIST+ISDIST+IMSD+IKIRKG+IKIRKR+IFDBF) > 0)THEN
       CALL WRNDIE(-2,'<SOLANA>', &
            'VAC ANALYSIS IS NOT SUPPORTED WITH COORDINATE ANALYSIS')
       ! User insists; we suppress IVAC
       IVAC=0
       IF(PRNLEV >= 2) &
            WRITE(OUTU,*) '%%%-SOLANA> SUPPRESSING VAC ANALYSIS %%%'
    ENDIF
    IF(IVAC*IMRD > 0 .OR. IMRD*IMSD.GT.0) CALL WRNDIE(-2,'<SOLANA>', &
         'ONLY ONE OF (IVAC,IMRD,IMSD) IS ALLOWED')
    ! A few checkings   (AB)
    IF (CROS.AND.(IKIRKG+IKIRKR > 0)) CALL WRNDIE(-2,'<SOLANA>', &
         'DIPOLE ANALYSIS IS NOT SUPPORTED WITH CROSs')
    IF (CROS.AND.(ISDIST /= 0)) CALL WRNDIE(-2,'<SOLANA>', &
         'SITE IS NOT SUPPORTED WITH CROSs')
    IF (CROS.AND.(IFDBF /= 0)) CALL WRNDIE(-2,'<SOLANA>', &
         'FORCE CALCULATION IS  NOT SUPPORTED WITH CROSs')
    !
    !     write analysis information
    IF(PRNLEV >= 2) THEN
       WRITE (OUTU,'(A,1X,I7,A,I7)') &
            ' Configurations to be read for steps',NFIRST,' -',NSTEP
       WRITE (OUTU,'(A,1X,I5,A)') &
            ' skipping',NSKIP,' steps between calculations.'
       WRITE (OUTU,'(A,F8.3,A)') &
            'Analysis done for ',RSP,' A density/dynamics sphere'
    ENDIF
    !
    RSP=RSP*RSP
    VEL=.FALSE.
    IF(IVAC /= 0) THEN
       IF(PRNLEV >= 2) THEN
          WRITE (OUTU,'(A,I5,A)') &
               'Solvent vac computed for ',NCORS,' timesteps'
          WRITE (OUTU,'(A,F8.3,A)') &
               'of length ',DT
          !     $      'of length ',DT,' for molecules in shell between'
          IF(RSPOUT > ZERO)THEN
             ! Tell user that there will be no checking against RSPIN or RSPOUT
             CALL WRNDIE(1,'<SOLANA>', &
                  'VAC in shells is currently not supported')
          ENDIF
          !          WRITE (OUTU,'(F8.3,A,F8.3,A)')
          !     $      RSPIN,' and ',RSPOUT,' around reference point'
       ENDIF
       RSPIN = RSPIN**2
       RSPOUT = RSPOUT**2
       VEL=.TRUE.
    ENDIF
    !
    ! Reasonable NCORS?
    IF(NCORS >  (NSTEP-NFIRST)/NSKIP + 1)THEN
       IF(WRNLEV >= 2)THEN
          WRITE(OUTU,'(A,I5,A,I5,A)') 'NCORS=',NCORS, &
               ' is greater than',(NSTEP-NFIRST)/NSKIP + 1, &
               ' the number of steps analyzed. It will be reset.'
       ENDIF
       NCORS=(NSTEP-NFIRST)/NSKIP + 1
    ENDIF
    !
    IF(IMSD /= 0) THEN
       IF(PRNLEV >= 2) THEN
          WRITE (OUTU,'(A,I5,A,F8.3,A)') &
               'Solvent msd computed for',NCORS,' steps of length',DT,' ps'
          IF(RSPOUT > RSPIN)THEN
             WRITE (OUTU,'(A,F8.3,A,F8.3,A)') &
                  'for molecules in shell between',RSPIN,' and',RSPOUT,' A'
             IF(NSITE == 0)THEN
                WRITE(OUTU,'(A)') 'around reference point'
             ELSE
                WRITE(OUTU,'(A)') 'around reference sites'
             ENDIF
          ENDIF
       ENDIF
       RSPIN = RSPIN**2
       RSPOUT = RSPOUT**2
    ENDIF
    !
    IF(QROTP1.OR.QROTP2) THEN
       IF(PRNLEV >= 2) THEN
          IF(QROTP1) WRITE (OUTU,'(A,I5,A,F8.3,A)') &
               'Water-dipole P1 rotational correlation computed for', &
               NCORS,' steps of length',DT,' ps'
          IF(QROTP2) WRITE (OUTU,'(A,I5,A,F8.3,A)') &
               'Water-dipole P2 rotational correlation computed for', &
               NCORS,' steps of length',DT,' ps'
          IF(RSPOUT > RSPIN)THEN
             WRITE (OUTU,'(A,F8.3,A,F8.3,A)') &
                  'for molecules in shell between',RSPIN,' and',RSPOUT,' A'
             IF(NSITE == 0)THEN
                WRITE(OUTU,'(A)') 'around reference point'
             ELSE
                WRITE(OUTU,'(A)') 'around reference sites'
             ENDIF
          ENDIF
       ENDIF
       RSPIN = RSPIN**2
       RSPOUT = RSPOUT**2
       IF(IMRD /= 0)  CALL WRNDIE(-3,'<SOLANA>', &
            'ROTCor P1/P2 not allowed with MRD')  
    ENDIF
    !
    IF(IMRD /= 0) THEN
       IF(PRNLEV >= 2) THEN
          WRITE (OUTU,'(A,I5,A,F10.5,A)') &
               'MRD computed for',NCORS,' steps of length',DT,' ps'
          IF(RSPOUT > RSPIN)THEN
             WRITE (OUTU,'(A,F8.3,A,F8.3,A)') &
                  'for molecules in shell between',RSPIN,' and',RSPOUT,' A'
             IF(NSITE == 0)THEN
                WRITE(OUTU,'(A)') 'around reference point'
             ELSE
                WRITE(OUTU,'(A)') 'around reference sites'
             ENDIF
          ENDIF
          IF(RRES > ZERO) THEN
             WRITE(OUTU,'(A,F8.3,A)') &
                  'Residence time for molecules within ',RRES,' A of reference'
          ENDIF
       ENDIF
       RSPIN = RSPIN**2
       RSPOUT = RSPOUT**2
       RRES=RRES**2
    ELSE
       RRES=ZERO
    ENDIF

    !
    IF(RHYD > 0.0)THEN
       if (prnlev >= 2) WRITE(OUTU,'(A,A,F8.3,A)') 'Computing solvent atoms/molecules', &
            ' within ',SQRT(RHYD),' A of the selected site'
    ENDIF
    !
    IF (IGDIST /= 0) THEN
       IF(PRNLEV >= 2) THEN
          WRITE (OUTU,'(A,F8.3,A)') &
               'Solvent-solvent rdf computed on radial grid ',DR,' A'
          WRITE (OUTU,'(A,F8.3,A)') &
               'Structure analyzed for solvent inside ',RSPHER,' A'
       ENDIF
       RSPHER = RSPHER**2
       call chmalloc('solana.src','SOLANL','GNOO',MGN,intg=GNOO)
       call chmalloc('solana.src','SOLANL','GNOH',MGN,intg=GNOH)
       call chmalloc('solana.src','SOLANL','GNHH',MGN,intg=GNHH)
    ELSE
       call chmalloc('solana.src','SOLANL','GNOO',1,intg=GNOO)
       call chmalloc('solana.src','SOLANL','GNOH',1,intg=GNOH)
       call chmalloc('solana.src','SOLANL','GNHH',1,intg=GNHH)
    ENDIF
    !
    IF(ISDIST /= 0) THEN
       IF(PRNLEV >= 2) THEN
          WRITE (OUTU,'(A,F8.3,A)') &
               'Solvent-site rdf computed on radial grid ',DR,' A'
          WRITE (OUTU,'(A,F8.3,A)') &
               'Structure analyzed for solvent inside ',RSPHER,' A'
       ENDIF
       call chmalloc('solana.src','SOLANL','GNOS',MGN,intg=GNOS)
       IF(IGDIST == 0) RSPHER = RSPHER**2
    ELSE
       call chmalloc('solana.src','SOLANL','GNOS',1,intg=GNOS)
    ENDIF
    !
    IF(IGDIST+ISDIST > 0 .OR. RHYD.GT.0.0)THEN
       call chmalloc('solana.src','SOLANL','SOLVSET',NATOM,intg=SOLVSET)
       call chmalloc('solana.src','SOLANL','SITESET',NATOM,intg=SITESET)
    ENDIF
    IF(QINTRA.AND. PRNLEV > 2)THEN
       WRITE(OUTU,*) 'Suppressing "intramolecular" peaks in g(r)'
       WRITE(OUTU,*) 'based on GROUP,RESIDUE,SEGMENT:', &
            QBYGRP,QBYRES,QBYSEG
    ENDIF
    MINI=.FALSE.
    IF(IFMIN > 0) THEN
       MINI=.TRUE.
       IF(PRNLEV >= 2) THEN
          WRITE (OUTU,'(A)') &
               ' Minimum image convention used in computing structure'
          WRITE (OUTU,'(A,3(F8.3,A))') &
               ' Primary box size',XBOX,' A by',YBOX,' A by',ZBOX,' A'
       ENDIF
       XBOX2=XBOX/TWO
       YBOX2=YBOX/TWO
       ZBOX2=ZBOX/TWO
    ENDIF
    !
    IF(IFDBF > 0) THEN
       IF(PRNLEV >= 2) THEN
          WRITE (OUTU,'(A)')' Deformable Boundary Force on '//NAME(TYP)
          WRITE (OUTU,'(A)')' computed from dynamics simulation'
          WRITE (OUTU,'(A,F8.3,A)') &
               ' for an interaction radius ',RCUT,' A'
          WRITE (OUTU,'(I5,A)') NZP,' points will be computed between '
          WRITE (OUTU,'(F8.3,A,F8.3)') ZP0,' to ',SQRT(RSP)
       ENDIF
       DZP = (SQRT(RSP) - ZP0)/(NZP-1)
       RCUT=RCUT*RCUT
       IF(PRNLEV >= 2..AND. IFDBFT > 0) THEN
          IF(IOLEV >= 1) WRITE(IFDBFT,'(A)') NAME(TYP)
          IF(IOLEV >= 1) WRITE(IFDBFT,'(I5)') NZP
       ENDIF
    ENDIF
    !
    IF(IKIRKG > 0) THEN
       !        NKIRK=(SQRT(RSP)/2.5)**3 + 1
       !        IF(NKIRK  >  61) NKIRK=61
       !         NKIRK=61
       ! This requires WATEr to make sure that we have access to all three
       ! coordinates, so turn it on in case user forgot...unless CROSS has
       ! been requested, in which case we have a problem
       WAT=.TRUE.
       IF(CROS)THEN
          CALL WRNDIE(-2,'<SOLANA>', &
               'KIRK requires WAT, which is not compatible with CROSS')
       ENDIF
       IF(PRNLEV >= 2) THEN
          WRITE (OUTU,'(A)') &
               ' Water dipole-site probability distribution '
          WRITE (OUTU,'(A,F8.3,A)') &
               ' will be computed within a sphere of',SQRT(RSP),' A'
          WRITE (OUTU,'(A,I5,A)') &
               ' divided into ',NKIRK,' equal DR elements'
          !     $          ' divided into ',NKIRK,' equal volume elements'
       ENDIF
       !
       call chmalloc('solana.src','SOLANL','DICO',19,NKIRK,crl=DICO)
       call chmalloc('solana.src','SOLANL','IKRKD',NKIRK,intg=IKRKD)
       dico=zero
       ikrkd=0
    ENDIF
    IF(IKIRKR  >  0)THEN
       WAT=.TRUE.
       IF(CROS)THEN
          CALL WRNDIE(-2,'<SOLANA>', &
               'RKIRK requires WAT, which is not compatible with CROSS')
       ENDIF
       IF(PRNLEV >= 2) THEN
          WRITE (OUTU,'(A)') &
               ' Distance dependent Kirkwood G-factor for water dipoles'
          WRITE (OUTU,'(A,F8.3,A)') &
               ' will be computed within a sphere of',SQRT(RSP),' A'
          WRITE (OUTU,'(A,I5,A)') &
               ' divided into ',NKIRK,' equal DR elements'
       ENDIF
       !
       call chmalloc('solana.src','SOLANL','DICOr',NKIRK+1,crl=DICOr)
       call chmalloc('solana.src','SOLANL','MJ',3,NKIRK+1,crl=MJ)
       dico=zero
    ENDIF
    !
#if KEY_FLUCQ==1
    IF (IDIP > 0) THEN
       IF (PRNLEV >= 2) THEN
          WRITE(OUTU,'(A,I8)') &
               ' Dipole distribution will be plotted for ',NUMD
          WRITE(OUTU,'(A,2F10.2)') &
               ' points in the range ',MIND,MAXD
       ENDIF
    ENDIF
#endif 
    IF(IHIST > 0.OR.IPDB.GT.0)THEN
       IF(PRNLEV >= 2)THEN
          WRITE(OUTU,'(A,I8,A)') &
               ' 3D histogram analysis will be performed on', &
               NDI*NXPT*NYPT*NZPT,' gridpoints in the range:'
          WRITE(OUTU,'(A,3F10.2)') 'X min,max,dx:',XMIN,XMAX,XSTP
          WRITE(OUTU,'(A,3F10.2)') 'Y min,max,dx:',YMIN,YMAX,YSTP
          WRITE(OUTU,'(A,3F10.2)') 'Z min,max,dx:',ZMIN,ZMAX,ZSTP
          IF(QDIP)THEN
             WRITE(OUTU,'(A)') &
                  'Dipoles of selected groups will be accumulated'
          ELSEIF(QCHG)THEN
             WRITE(OUTU,'(A)') &
                  'The charge of selected groups will be accumulated'
          ENDIF
          IF(QWEIG) WRITE(OUTU,'(A)') &
               'WMAIN will be used for weighting'
       ENDIF
    ENDIF
    !
    IF(NEXV > 0)THEN
       IF(PRNLEV >= 2)THEN
          WRITE(OUTU,'(A)') &
               'Excluded volume correction will be calculated around the'
          IF(NSITE > 0)THEN
             WRITE(OUTU,'(I7,A)') NSITE,' SITE atoms using'
          ELSE
             WRITE(OUTU,'(I7,A)') NSOLV,' SOLVENT atoms using'
          ENDIF
          WRITE(OUTU,'(I7,A,I4,F6.2,A)' ) &
               NMCPTS,' Monte Carlo points in ',NPTS,DRMC,'A shells'
          WRITE(OUTU,'(A,F6.3,5X,A,I8)') &
               'Probe radius',RPROBE,'ISEED',ISEED
       ENDIF
    ENDIF
    !     some memory management
    IF(NSOLV*NCORS > 0) THEN
       call chmalloc('solana.src','SOLANL','VX',NCORS,nsolv,crl=VX)
       call chmalloc('solana.src','SOLANL','VY',NCORS,nsolv,crl=VY)
       call chmalloc('solana.src','SOLANL','VZ',NCORS,nsolv,crl=VZ)
       call chmalloc('solana.src','SOLANL','ISTACK',NSOLV,intg=ISTACK)
    ENDIF
    IF(IVAC > 0.OR.IMSD.GT.0.OR.IMRD.GT.0.OR.QROTP1.OR.QROTP2 &
#if KEY_CHEQ==1
         .OR. QVAC > 0    & 
#endif
         ) THEN
       IF(NCORS <= 0)  CALL WRNDIE(-3,'<SOLANA>', &
            'NCORS <= 0 for VAC/MSD/MRD/ROTCOR(P1/P2)')
       call chmalloc('solana.src','SOLANL','VAC',NCORS,crl=VAC)
       call chmalloc('solana.src','SOLANL','INORM',NCORS,intg=INORM)
       call chmalloc('solana.src','SOLANL','VNORM',NCORS,crl=VNORM)
       call chmalloc('solana.src','SOLANL','TIM',NCORS,crl=TIM)
    ENDIF
#if KEY_FLUCQ==1
    IF (IDIP > 0) THEN
       call chmalloc('solana.src','SOLANL','DDIS',NUMD,intg=DDIS)
       ddis=0
       DCNT=0
    ENDIF
#endif 
    IF(IHIST > 0.OR.IPDB.GT.0)THEN
       ! This may be a large dataset, so we try using single precision....?
       call chmalloc('solana.src','SOLANL','HIST',NDI,NXPT,NYPT,NZPT,cr4=HIST)
       call chmalloc('solana.src','SOLANL','ISOLV',NATOM,intg=ISOLV)
    ENDIF
    !SJS
    IF(IROTCO > 0.AND. .NOT. (QROTP1.OR.QROTP2))THEN
       NWAT = NSOLV/3
       call chmalloc('solana.src','SOLANL','HHXV',MAXTOT,NWAT,crl=HHXV)
       call chmalloc('solana.src','SOLANL','HHYV',MAXTOT,NWAT,crl=HHYV)
       call chmalloc('solana.src','SOLANL','HHZV',MAXTOT,NWAT,crl=HHZV)
       call chmalloc('solana.src','SOLANL','OHXV',MAXTOT,NWAT,crl=OHXV)
       call chmalloc('solana.src','SOLANL','OHYV',MAXTOT,NWAT,crl=OHYV)
       call chmalloc('solana.src','SOLANL','OHZV',MAXTOT,NWAT,crl=OHZV)
    endif
    !SJS
    !
    !     initialize
    DO I=1,50
       F2RAD(I)=ZERO
       FRAD(I)=ZERO
    ENDDO

    DO I=1,61
       NOXDIS(I)=0
    ENDDO
    !
    IF(NEXV > 0)THEN
       IF(CROS)CALL WRNDIE(-2,'<SOLANA>', &
            'CROSS is not supported with EXcludedVolumeCorrection')
       CALL ACVOLUM('INIT',ACV,NPTS,DRMC,NSOLV, &
            OX2,NMCPTS,NEXV,SOLUTE,RS,RDMAX,ISEED)
    ENDIF
    NGOO=0
    NGOH=0
    NGHH=0
    NGOS=0
    !
    IF(IHIST  >  0 .OR. IPDB.GT.0)THEN
       CALL HISINI(NXPT,NYPT,NZPT,NDI,HIST,NSOLV,FLGSOL, &
            ISOLV,QDIP,NGRAT)
       NHIST=0
       IF(NGRAT > 0) call chmalloc('solana.src','SOLANL','IGRAT',NGRAT,intg=IGRAT)
    ENDIF
    !
    CALL GNINIT(MGN,ISDIST,IGDIST,GNOO,GNOH,GNHH,GNOS)
    NCDBF=0
    NDENS=0
    NOXYG=0
    NDENS2=0
    NCONF2=0
    !
    IF(IGDIST+ISDIST  >  0)THEN
       CALL MSETINI(QBYGRP,QBYRES,QBYSEG,SOLVSET,SITESET, &
            NSOLV,FLGSOL,NSITE,FLGSIT)
    ENDIF
    IF(RHYD > 0.0 .OR. ISDIST+IGDIST.GT.0)THEN
       ! For hydration analysis we need the flag arrays to indicate
       ! the atom index, which is not compatible with any of QBY* being true
       IF((QBYGRP.OR.QBYRES.OR.QBYSEG).AND.RHYD > 0.0) &
            CALL WRNDIE(-2,'<SOLANA>', &
            'GROUP/RES/SEG flag not compatible with hyd.no. calculation')
       CALL MSETINI(QBYGRP,QBYRES,QBYSEG,SOLVSET,SITESET, &
            NSOLV,FLGSOL,NSITE,FLGSIT)
    ENDIF
    !
    ERRORC=.FALSE.
    ISTEP=0
    NFRAMES=0
    !
    loop100: DO ISTP=NFIRST,NSTEP,NSKIP
       !
       !     get another coordinate file
       CALL GTSOLV(ISTP,X,Y,Z, &
            MOX2,OX2,hx1,hx2,NSOLV,MX,REF, &
            sref,FLGSOL,FLGSIT,SITE,NSITE,MXSITE, &
            NEXV,FLGEXV,SOLUTE, &
            WAT,CROS,LMULTI,QFIRST,NFIRST,NSKIP,NSTEP, &
            NUNIT,FIRSTU,TEMP,FREEAT,ERRORC,VEL &
            ,QCRYS,NFREAT,NSAVC,NFILE &                     
            )


       ! Everything OK by now?
       IF(ERRORC) THEN
          IF(WRNLEV >= 2 .AND. PRNLEV .GE. 2)THEN
             WRITE (OUTU,'(A,I5)') &
                  '%%%-SOLANL> Error reading coordinates at ',ISTP
          ENDIF
          GOTO 999
       ENDIF
       !
       NFRAMES=NFRAMES+1
       !SJS
       IF(IROTCO > 0)THEN
          IF(QROTP1 .OR. QROTP2)THEN
             IF(LMULTI)THEN
                CALL CORFUNCB(ISTEP,NCORS,NSOLV,OX2, &
                     OX2,hx1,hx2, &
                     RSPIN,RSPOUT,RRES,NSITE,sref,VEL,QMRD, &
                     vx,vy,vz,istack, &
                     VAC,inorm,QROTP1,QROTP2)
             ELSE
                CALL CORFUNCB(ISTEP,NCORS,NSOLV,OX2, &
                     OX2,hx1,hx2, &
                     RSPIN,RSPOUT,RRES,1,REF,VEL,QMRD, &
                     vx,vy,vz,istack, &
                     VAC,inorm,QROTP1,QROTP2)
             ENDIF
          ELSE
             CALL ROTVEC(NSOLV,NFRAMES,OX2,hx1,hx2, &
                  MOX2,hhxv,HHYV,HHZV, &
                  OHXV,OHYV,OHZV,MTOT,NWAT)
          ENDIF
       ENDIF
       !SJS
       IF(NEXV > 0)THEN
          IF(NSITE > 0)THEN
             CALL ACVOLUM('ACCU',ACV,NPTS,DRMC,NSITE, &
                  sref,NMCPTS,NEXV,SOLUTE,RS, &
                  RDMAX,ISEED)
          ELSE
             CALL ACVOLUM('ACCU',ACV,NPTS,DRMC,NSOLV, &
                  OX2,NMCPTS,NEXV,SOLUTE,RS, &
                  RDMAX,ISEED)
          ENDIF
       ENDIF
       IF(IKIRKG > 0) THEN
          IF(LMULTI) THEN
             CALL KIRKG(NSOLV,OX2,hx1,hx2, &
                  DICO,RSP,REF,NKIRK,IKRKD,NSITE, &
                  sref,LMULTI)
          ELSE
             CALL KIRKG(NSOLV,OX2,hx1,hx2, &
                  DICO,RSP,REF,NKIRK,IKRKD,1,REF,LMULTI)
          ENDIF
       ENDIF
       IF(IKIRKR > 0)THEN
          CALL RKIRKG(NSOLV,OX2,hx1,hx2,DICOR, &
               MJ,RSP,NKIRK,NSITE,sref)
       ENDIF
       IF(IDENS > 0 .OR. IGDIST.GT.0 .OR. ISDIST .GT. 0)THEN
          IF(CROS)THEN
             ! We need the density of the second selection (the "sites") for proper
             ! normalization
             CALL DENSIT(NSITE,hx1,RSP,REF,NDENS2,NCONF2,NOXDIS)
          ENDIF
          CALL DENSIT(NSOLV,OX2,RSP,REF,NDENS,NOXYG,NOXDIS)
       ENDIF
       !
       IF(ISDIST > 0) THEN
          IF(LMULTI)THEN
             CALL GDIST(NSITE,sref,NSOLV,OX2,CROS, &
                  RSPHER,REF,NGOS,MGN,GNOS,DR,LMULTI, &
                  SITESET,SOLVSET)
          ELSE
             CALL GDIST(1,REF,NSOLV,OX2,CROS, &
                  RSPHER,REF,NGOS,MGN,GNOS,DR,LMULTI, &
                  SITESET,SOLVSET)
          ENDIF
       ENDIF
       !
       IF(IGDIST > 0) THEN
          CALL GDIST(NSOLV,OX2,NSOLV,OX2,CROS, &
               RSPHER,REF,NGOO,MGN,GNOO,DR,LMULTI, &
               SOLVSET,SOLVSET)
          !
          IF(WAT) THEN
             CALL GDIST(NSOLV,OX2,NSOLV,hx1,CROS, &
                  RSPHER,REF,NGOH,MGN,GNOH,DR,LMULTI, &
                  SOLVSET,SOLVSET)
             CALL GDIST(NSOLV,OX2,NSOLV,hx2,CROS, &
                  RSPHER,REF,NGOH,MGN,GNOH,DR,LMULTI, &
                  SOLVSET,SOLVSET)

             CALL GDIST(NSOLV,hx1,NSOLV,hx1,CROS, &
                  RSPHER,REF,NGHH,MGN,GNHH,DR,LMULTI, &
                  solvset,solvset)
             CALL GDIST(NSOLV,hx1,NSOLV,hx2,CROS, &
                  RSPHER,REF,NGHH,MGN,GNHH,DR,LMULTI, &
                  solvset,solvset)
             CALL GDIST(NSOLV,hx2,NSOLV,hx2,CROS, &
                  RSPHER,REF,NGHH,MGN,GNHH,DR,LMULTI, &
                  solvset,solvset)
          ENDIF
          !
          ! CROSs uses NSITE and hx1 for solvent2  (AB)
          IF(CROS) THEN
             ! Note the order here to give correct normalization using the
             ! SOLVent density (not the SITE density)
             CALL GDIST(NSITE,hx1,NSOLV,OX2,CROS, &
                  RSPHER,REF,NGOH,MGN,GNOH,DR,LMULTI, &
                  siteset,solvset)
             ! and for this we have to use the density of the second selection
             CALL GDIST(NSITE,hx1,NSITE,hx1,CROS, &
                  RSPHER,REF,NGHH,MGN,GNHH,DR,LMULTI, &
                  siteset,siteset)
          ENDIF
       ENDIF
       !
       IF(RHYD > 0.0)THEN
          CALL HYDNUM(IHYDN,RHYD,NSOLV,OX2,solvset, &
               NSITE,sref,siteset, &
               DT,NCONFH,NRR,NAR,NAA)
       ENDIF

       !
       IF(IMSD > 0) THEN
          IF(LMULTI)THEN
             CALL CORFUNCB(ISTEP,NCORS,NSOLV,OX2, &
                  OX2,hx1,hx2, &
                  RSPIN,RSPOUT,RRES,NSITE,sref,VEL,QMRD, &
                  vx,vy,vz,istack, &
                  VAC,inorm,QROTP1,QROTP2)
          ELSE
             CALL CORFUNCB(ISTEP,NCORS,NSOLV,OX2, &
                  OX2,hx1,hx2, &
                  RSPIN,RSPOUT,RRES,1,REF,VEL,QMRD, &
                  vx,vy,vz,istack, &
                  VAC,inorm,QROTP1,QROTP2)
          ENDIF
       ENDIF

       !
       IF(IMRD > 0) THEN
          CALL CORFUNCB(ISTEP,NCORS,NSOLV,OX2, &
               OX2,hx1,hx2, &
               RSPIN,RSPOUT,RRES,1,REF,VEL,QMRD, &
               vx,vy,vz,istack, &
               VAC,inorm,QROTP1,QROTP2)
       ENDIF
       !
       IF(IFDBF > 0) THEN
          CALL FBOUND(IFDBFT,NCDBF,NSOLV, &
               OX2,hx1,hx2,SITE, &
               REF,TYP,RSP,RCUT,ZP0,DZP,NZP,XBOX,YBOX,ZBOX)
       ENDIF
       IF(IHIST > 0.OR.IPDB.GT.0)THEN
          CALL SLVHIS(NSOLV,ISOLV,NGRAT,IGRAT, &
               NXPT,XMIN,XMAX,XSTP, &
               NYPT,YMIN,YMAX,YSTP, &
               NZPT,ZMIN,ZMAX,ZSTP,NDI,HIST,NHIST, &
               QDIP,QWEIG,QCHG, &
               X,Y,Z,WMAIN)
       ENDIF
#if KEY_FLUCQ==1
       IF (IDIP > 0) THEN
          CALL SLVDIP(FLGSOL,NUMD,MIND,MAXD,DDIS,DCNT,X,Y,Z)
       ENDIF
#endif 
       !
       ! velocity file
       IF(IVAC > 0 .AND. ISTP.GT.1) THEN
          CALL CORFUNCB(ISTEP,NCORS,NSOLV,OX2, &
               OX2,hx1,hx2, &
               RSPIN,RSPOUT,RRES,1,REF,VEL,QMRD, &
               vx,vy,vz, &
               istack,VAC,inorm,QROTP1,QROTP2)
       ENDIF
    enddo loop100

#if KEY_CHEQ==1
    call chmalloc('solana.src','SOLANL','VVCG',natom,crl=vvcg)
    LC = 0
    DO LL = 1,NATOM
       IF (FLGSOL(LL) /= 0) THEN
          LC = LC + 1
          VVCG(LC) = CG(LL)
       ENDIF
    ENDDO
    !
    ! charge velocity autocorrelation calculation
    !
    IF(QVAC > 0 .AND. ISTP.GT.1) THEN
       CALL CORFUNCQ(ISTEP,NCORS,NSOLV,VVCG,istack,VAC, &
            inorm)
    ENDIF
    call chmdealloc('solana.src','SOLANL','VVCG',natom,crl=vvcg)
#endif 
    !
    !  print out everything
    !  all subroutines check if the unit number is valid
999 CONTINUE
    !SJS
    IF(IROTCO > 0)THEN
       IF(QROTP1.OR.QROTP2)THEN
          CALL PRNROT(ROUT,NCORS,DT,VAC,inorm, &
               VNORM,QROTP2)
       ELSE
          NTOT=2
          DO WHILE(NTOT <= NFRAMES)
             NTOT=NTOT*2
          ENDDO
          NTOT=NTOT/4
          call chmalloc('solana.src','SOLANL','TX',MAXTOT,3,crl=TX)
          call chmalloc('solana.src','SOLANL','TX1',MAXTOT,3,crl=TX1)
          call chmalloc('solana.src','SOLANL','TX2',MAXTOT,3,crl=TX2)
          call chmalloc('solana.src','SOLANL','TCO',MAXTOT,crl=TCO)
          call chmalloc('solana.src','SOLANL','TCO1',MAXTOT,crl=TCO1)
          call chmalloc('solana.src','SOLANL','TCO2',MAXTOT,crl=TCO2)
          call chmalloc('solana.src','SOLANL','ATCOR',MAXTOT,crl=ATCOR)
          call chmalloc('solana.src','SOLANL','ATCOR1',MAXTOT,crl=ATCOR1)
          call chmalloc('solana.src','SOLANL','ATCOR2',MAXTOT,crl=ATCOR2)
          ATCOR = ZERO
          ATCOR1 = ZERO
          ATCOR2 = ZERO
          !
          IF(PRNLEV >= 2)THEN
             WRITE(OUTU,*)
             WRITE(OUTU,*) &
                  '     ROTATIONAL CORRELATION TIME FOR WATER IN PS'
             WRITE(OUTU,*)
             WRITE(OUTU,105) &
                  'SOLA_RCOR>','CORRELATION TIME FOR ',NWAT,'WATERS '
             WRITE(OUTU,*)
             WRITE(OUTU,110)'SOLA_RCOR>','WAG','TWIST','ROCK'
             WRITE(OUTU,*)
105          FORMAT(1X,A10,5X,A22,I3,2X,A6)
110          FORMAT(1X,a10,30X,6X,A3,3X,3X,A5,2X,3X,A4)
          ENDIF
          CALL ROTCOR(NFRAMES,NWAT,hhxv,HHYV,HHZV, &
               OHXV,OHYV,OHZV,NFRAMES,NTOT, &
               TLOW,TUP,IROTCO,ROUT, &
               tx,tx1,tx2,tco,tco1,tco2,dt,atcor,atcor1,atcor2)
       ENDIF
    ENDIF
    !SJS
    IF(NEXV > 0)THEN
       CALL ACVOLUM('FINI',ACV,NPTS,DRMC,NSOLV,OX2, &
            NMCPTS,NEXV,SOLUTE,RS,RDMAX,ISEED)
    ENDIF
    IF(IVAC > 0) &
         CALL PRNTVAC(IVAC,NCORS,DT,VAC,inorm,VNORM)
    ! SAPTEL
#if KEY_CHEQ==1
    IF(QVAC > 0) &
         CALL PRNTVAC(QVAC,NCORS,DT,VAC,inorm,VNORM)
#endif 
    ! SAPTEL
    IF(IMSD > 0) &
         CALL PRNTMSD(IMSD,NCORS,DT,VAC,inorm,VNORM,TIM)
    IF(IMRD > 0) &
         CALL  PRNTMRD(IMRD,NCORS,DT,VAC,vz,inorm)
    CALL PRNTGK(IKIRKG,DICO,RSP,NKIRK,IKRKD)
    IF(IKIRKR > 0)THEN
       IF(NSITE  >=  2)THEN
          RKNORM=NFRAMES
       ELSE
          RKNORM=NSOLV*NFRAMES
       ENDIF
       CALL PRNTGKR(IKIRKR,DICOr,RSP,NKIRK,NSITE,RKNORM)
    ENDIF
    CALL PRNDNS(IDENS,NOXYG,RSP,NDENS,DENS,NOXDIS,NSOLV)
    IF(ISDIST > 0) &
         CALL PRNTG(ISDIST,RSPHER,DR,NGOS,MGN,GNOS,DENS, &
         NPTS,ACV)
    IF(IGDIST > 0) &
         CALL PRNTG(IGDIST,RSPHER,DR,NGOO,MGN,GNOO,DENS, &
         NPTS,acv)
    IF(IOH > 0) &
         CALL PRNTG(IOH,RSPHER,DR,NGOH,MGN,GNOH,DENS, &
         NPTS,acv)
    IF(CROS.AND.IHH > 0)THEN
       CALL PRNDNS(-1,NCONF2,RSP,NDENS2,DENS2,NOXDIS,NSITE)
       CALL PRNTG(IHH,RSPHER,DR,NGHH,MGN,GNHH,DENS2, &
            NPTS,acv)
    ELSE
       IF(IHH > 0) &
            CALL PRNTG(IHH,RSPHER,DR,NGHH,MGN,GNHH,DENS, &
            NPTS,acv)
    ENDIF
    CALL PRNFBND(IFDBF,NCDBF,ZP0,DZP,NZP,TYP)
    IF(IHIST > 0 .OR. IPDB.GT.0) &
         CALL PRNHIST(IHIST,IPDB,NXPT,NYPT,NZPT,NDI,HIST,NHIST, &
         XMIN,XSTP,XMAX,YMIN,YSTP,YMAX,ZMIN,ZSTP,ZMAX, &
         NFRAMES,QDIP,RNORM,THRS)
    !SJS, output modified October 2002, LNI
    IF(RHYD > 0.0)THEN
       IF(PRNLEV >= 2)THEN
          WRITE(OUTU,'(/A,F8.3,A)') &
               'SOLA_HYDN> AVERAGE HYDRATION NUMBERS AT ', &
               SQRT(RHYD),'A CUTOFF'
          WRITE(OUTU,'(A,I10,A/A,I6)') &
               'USING',NCONFH,' CONFIGURATIONS', &
               'SECOND NUMBER IS AVERAGED OVER NUMBER OF SOLUTE ATOMS:',NSITE
          IF(NSITE > 0 .AND.NCONFH.GT.0)THEN
             TMP=ONE/(NSITE*NCONFH)
             WRITE(OUTU,'(/(A,2G12.4))') &
                  ' SOLVENT RESIDUE - SOLUTE:',FLOAT(NRR)/NCONFH,TMP*NRR, &
                  '    SOLVENT ATOM - SOLUTE:',FLOAT(NAR)/NCONFH,TMP*NAR, &
                  ' SOLV. ATOM - SOLUTE ATOM:',FLOAT(NAA)/NCONFH,TMP*NAA
          ENDIF
       ENDIF
       IF(NCONFH > 0)THEN
          TMP=FLOAT(NRR)/NCONFH
          call set_param('NHYDRR',TMP)
          TMP=FLOAT(NAR)/NCONFH
          call set_param('NHYDAR',TMP)
          TMP=FLOAT(NAA)/NCONFH
          call set_param('NHYDAA',TMP)
       ENDIF
    ENDIF
    !SJS
#if KEY_FLUCQ==1
    IF (IDIP > 0) &
         CALL PRNDIP(IDIP,DDIS,DCNT,NUMD,MIND,MAXD)
#endif 
    !
    !     free memory
    IF(IVAC > 0.OR.IMSD.GT.0) THEN
       call chmdealloc('solana.src','SOLANL','VNORM',NCORS,crl=VNORM)
       call chmdealloc('solana.src','SOLANL','INORM',NCORS,intg=INORM)
       call chmdealloc('solana.src','SOLANL','VAC',NCORS,crl=VAC)
       call chmdealloc('solana.src','SOLANL','TIM',NCORS,crl=TIM)
    ENDIF
    !
    IF(NSOLV*NCORS > 0) THEN
       call chmdealloc('solana.src','SOLANL','VX',NCORS,nsolv,crl=VX)
       call chmdealloc('solana.src','SOLANL','VY',NCORS,nsolv,crl=VY)
       call chmdealloc('solana.src','SOLANL','VZ',NCORS,nsolv,crl=VZ)
       call chmdealloc('solana.src','SOLANL','ISTACK',NSOLV,intg=ISTACK)
    ENDIF
    !SJS
    IF(IROTCO > 0.AND. .NOT. (QROTP1.OR.QROTP2) )THEN
       call chmdealloc('solana.src','SOLANL','HHXV',MAXTOT,NWAT,crl=HHXV)
       call chmdealloc('solana.src','SOLANL','HHYV',MAXTOT,NWAT,crl=HHYV)
       call chmdealloc('solana.src','SOLANL','HHZV',MAXTOT,NWAT,crl=HHZV)
       call chmdealloc('solana.src','SOLANL','OHXV',MAXTOT,NWAT,crl=OHXV)
       call chmdealloc('solana.src','SOLANL','OHYV',MAXTOT,NWAT,crl=OHYV)
       call chmdealloc('solana.src','SOLANL','OHZV',MAXTOT,NWAT,crl=OHZV)
       !
       call chmdealloc('solana.src','SOLANL','TX',MAXTOT,3,crl=TX)
       call chmdealloc('solana.src','SOLANL','TX1',MAXTOT,3,crl=TX1)
       call chmdealloc('solana.src','SOLANL','TX2',MAXTOT,3,crl=TX2)
       call chmdealloc('solana.src','SOLANL','TCO',MAXTOT,crl=TCO)
       call chmdealloc('solana.src','SOLANL','TCO1',MAXTOT,crl=TCO1)
       call chmdealloc('solana.src','SOLANL','TCO2',MAXTOT,crl=TCO2)
       call chmdealloc('solana.src','SOLANL','ATCOR',MAXTOT,crl=ATCOR)
       call chmdealloc('solana.src','SOLANL','ATCOR1',MAXTOT,crl=ATCOR1)
       call chmdealloc('solana.src','SOLANL','ATCOR2',MAXTOT,crl=ATCOR2)
    ENDIF
    !SJS
    IF(ISDIST /= 0) THEN
       call chmdealloc('solana.src','SOLANL','GNOS',MGN,intg=GNOS)
    ELSE
       call chmdealloc('solana.src','SOLANL','GNOS',1,intg=GNOS)
    ENDIF
    IF(IGDIST /= 0) THEN
       call chmdealloc('solana.src','SOLANL','GNHH',MGN,intg=GNHH)
       call chmdealloc('solana.src','SOLANL','GNOH',MGN,intg=GNOH)
       call chmdealloc('solana.src','SOLANL','GNOO',MGN,intg=GNOO)
    ELSE
       call chmdealloc('solana.src','SOLANL','GNHH',1,intg=GNHH)
       call chmdealloc('solana.src','SOLANL','GNOH',1,intg=GNOH)
       call chmdealloc('solana.src','SOLANL','GNOO',1,intg=GNOO)
    ENDIF
    IF(IGDIST+ISDIST  >  0 .OR. RHYD.GT.0.0) THEN
       call chmdealloc('solana.src','SOLANL','SOLVSET',NATOM,intg=SOLVSET)
       call chmdealloc('solana.src','SOLANL','SITESET',NATOM,intg=SITESET)
    ENDIF
    IF(IKIRKR  >  0)THEN
       call chmdealloc('solana.src','SOLANL','DICOr',NKIRK+1,crl=DICOr)
       call chmdealloc('solana.src','SOLANL','MJ',3,NKIRK+1,crl=MJ)
    ENDIF
    IF(IKIRKG  >  0)THEN
       call chmdealloc('solana.src','SOLANL','DICO',19,NKIRK,crl=DICO)
       call chmdealloc('solana.src','SOLANL','IKRKD',NKIRK,intg=IKRKD)
    ENDIF
#if KEY_FLUCQ==1
    IF (IDIP > 0) THEN
       call chmdealloc('solana.src','SOLANL','DDIS',NUMD,intg=DDIS)
    ENDIF
#endif 
    IF(NEXV /= 0)THEN
       if (allocated(space_rl1)) then
          call chmdealloc('solana.src','SOLANL','space_rl1', &
               size(space_rl1),crl=space_rl1)
       endif
       nullify(ACV,rs)
       call chmdealloc('solana.src','SOLANL','SOLUTE',3,NEXV,crl=SOLUTE)
    ENDIF
    call chmdealloc('solana.src','SOLANL','OX2',3,natom,crl=OX2)
    call chmdealloc('solana.src','SOLANL','HX2',3,natom,crl=HX2)
    call chmdealloc('solana.src','SOLANL','HX1',3,natom,crl=HX1)

    call chmdealloc('solana.src','SOLANL','Z',NATOM,crl=Z)
    call chmdealloc('solana.src','SOLANL','Y',NATOM,crl=Y)
    call chmdealloc('solana.src','SOLANL','X',NATOM,crl=X)
    call chmdealloc('solana.src','SOLANL','TEMP',NATOM,cr4=TEMP)
    call chmdealloc('solana.src','SOLANL','FREEAT',NATOM,intg=FREEAT)
    !
    IF (LMULTI) THEN
       call chmdealloc('solana.src','SOLANL','SREF',3,NSITE,crl=SREF)
    ELSE
       call chmdealloc('solana.src','SOLANL','SREF',3,1,crl=SREF)
    ENDIF
    IF(IHIST > 0.OR.IPDB.GT.0)THEN
       call chmdealloc('solana.src','SOLANL','HIST',NDI,NXPT,NYPT,NZPT,cr4=HIST)
       call chmdealloc('solana.src','SOLANL','ISOLV',NATOM,intg=ISOLV)
       IF(NGRAT > 0) &
       call chmdealloc('solana.src','SOLANL','IGRAT',NGRAT,intg=IGRAT)
    ENDIF
    !
    RETURN
  END SUBROUTINE SOLANL

  SUBROUTINE GNINIT(MGN,ISDIST,IGDIST,GNOO,GNOH,GNHH,GNOS)
    !-----------------------------------------------------------------------

    INTEGER ISDIST,IGDIST

    INTEGER MGN,GNOO(*),GNOH(*),GNHH(*),GNOS(*)
    !
    INTEGER I
    !
    IF (ISDIST > 0) THEN
       DO I=1,MGN
          GNOS(I) = 0
       ENDDO
    ENDIF
    IF (IGDIST > 0) THEN
       DO I=1,MGN
          GNOO(I)=0
          GNOH(I)=0
          GNHH(I)=0
       ENDDO
    ENDIF
    RETURN
  END SUBROUTINE GNINIT

  SUBROUTINE HISINI(N1,n2,n3,n4,HIST,NSOLV,FLGSOL, &
       ISOLV,QDIP,NGRAT)
    !
    ! Initialization for solvent histogramming
    !
  use consta
  use number
  use psf
  use chutil,only:getres

    INTEGER N1,n2,n3,n4,NSOLV,NGRAT
    INTEGER ISOLV(NATOM),FLGSOL(NATOM)
    LOGICAL QDIP
    REAL(chm_real4) HIST(n1,n2,n3,n4)
    !
    INTEGER I,IS
    !
    ! Make index instead of flag array
    IS=0
    NGRAT=0
    !
    DO I=1,NATOM
       ISOLV(I)=0
       IF(FLGSOL(I) == 1)THEN
          IS=IS+1
          ISOLV(IS)=I
       ENDIF
    ENDDO
    !
    IF(IS  /=  NSOLV) &
         CALL WRNDIE(-2,'<HISINI>','Mismatch in number of solvent atoms')
    !
    IF(QDIP)THEN
       ! How many atoms do we have per group:
       ! Use GETRES to find GROUP information (all we need is a binary search
       ! through IGPBS, which we assume to be organized the same way as IBASE
       IS=GETRES(ISOLV(1),IGPBS,NGRP)
       NGRAT=IGPBS(IS+1)-IGPBS(IS)
       IF(IS  ==  GETRES(ISOLV(2),IGPBS,NGRP) .AND. NSOLV > 1)THEN
          CALL WRNDIE(-2,'<HISINI>', &
               'You should have only one atom/group for dipoles')
       ENDIF
    ENDIF
    HIST=ZERO
    RETURN
  END SUBROUTINE HISINI

#if KEY_FLUCQ==1
  SUBROUTINE PRNDIP(IDIP,DDIS,DCNT,NUMD,MIND,MAXD)
    !
    !     Prints the final dipole distribution

    INTEGER IDIP,NUMD,DDIS(NUMD),DCNT
    real(chm_real) MIND,MAXD

    INTEGER I
    real(chm_real) DIP,DR,CNT
    DR=(MAXD-MIND)/NUMD
    DIP=MIND
    DO I=1,NUMD
       CNT=REAL(DDIS(I))/DR/DCNT
       WRITE(IDIP,10) DIP,CNT
       DIP=DIP+DR
    ENDDO
10  FORMAT(F12.4,F12.4)
    RETURN
  END SUBROUTINE PRNDIP

  SUBROUTINE SLVDIP(FLGSOL,NUMD,MIND,MAXD,DDIS,DCNT,X,Y,Z)
    !
    !     Calculates a dipole distribution
  use psf
  use number
  use corsubs,only:cdipole

    INTEGER FLGSOL(NATOM),NUMD,DCNT
    INTEGER DDIS(NUMD)
    real(chm_real) MIND,MAXD,X(NATOM),Y(NATOM),Z(NATOM)
    real(chm_real) VALUE(3),TDIP,QTOT,DR,dummy(1)
    INTEGER I,J,NGRAT,IGRAT(80),IND
    LOGICAL INRES
    DR=(MAXD-MIND)/NUMD
    DO I=1,NRES
       INRES=.FALSE.
       DO J=IBASE(I)+1,IBASE(I+1)
          IF (FLGSOL(J) /= 0) INRES=.TRUE.
       ENDDO
       IF (INRES) THEN
          NGRAT=0
          DO J=IBASE(I)+1,IBASE(I+1)
             NGRAT=NGRAT+1
             IF (NGRAT > 80) CALL WRNDIE(-5,'<SLVDIP>', &
                  'Hard limit exceeded')
             IGRAT(NGRAT)=J
          ENDDO
          CALL CDIPOLE(NATOM,X,Y,Z,CG,NGRAT,IGRAT,VALUE(1), &
               VALUE(2),VALUE(3),QTOT,.TRUE.,.FALSE.,dummy)
          TDIP=SQRT(VALUE(1)**2+VALUE(2)**2+VALUE(3)**2)
          IF (TDIP >= MIND.AND.TDIP <= MAXD) THEN
             IND=INT((TDIP-MIND)/DR)
             DDIS(IND)=DDIS(IND)+1
             DCNT=DCNT+1
          ENDIF
       ENDIF
    ENDDO
    RETURN
  END SUBROUTINE SLVDIP
#endif /*  FLUCQ*/

  SUBROUTINE SLVHIS(NSOLV,ISOLV,NGRAT,IGRAT, &
       NXP,XMIN,XMAX,XSTP, &
       NYP,YMIN,YMAX,YSTP, &
       NZP,ZMIN,ZMAX,ZSTP,NDI,HIST,NHIST, &
       QDIP,QWEIG,QCHG, &
       X,Y,Z,WMAIN)
    !
    !  Performs actual 3D histogramming of solvent distribution
    !
  use number
  use psf
  use corsubs,only:cdipole
  use chutil,only:getres

    INTEGER NSOLV,NXP,NYP,NZP,NDI,NHIST,NGRAT
    INTEGER ISOLV(NSOLV),IGRAT(NGRAT)
    real(chm_real) XMIN,XMAX,XSTP,YMIN,YMAX,YSTP,ZMIN,ZMAX,ZSTP
    real(chm_real) X(NATOM),Y(NATOM),Z(NATOM),WMAIN(NATOM)
    LOGICAL QDIP,QCHG,QWEIG
    real(chm_real4) HIST(NDI,NXP,NYP,NZP)
    real(chm_real) VALUE(3), QTOT
    !
    INTEGER I,IX,IY,IZ,ID,IS,J
    real(chm_real)  XI,YI,ZI,XSINV,YSINV,ZSINV,dummy(1)
    LOGICAL SKIP
    !
    XSINV=ONE/XSTP
    YSINV=ONE/YSTP
    ZSINV=ONE/ZSTP
    DO I=1,NSOLV
       SKIP=.FALSE.
       ID=ISOLV(I)
       IX=(X(ID)-XMIN)*XSINV +1
       IY=(Y(ID)-YMIN)*YSINV +1
       IZ=(Z(ID)-ZMIN)*ZSINV +1
       IF(IX  <=  0 .OR. IX  >  NXP) SKIP=.TRUE.
       IF(IY  <=  0 .OR. IY  >  NYP) SKIP=.TRUE.
       IF(IZ  <=  0 .OR. IZ  >  NZP) SKIP=.TRUE.
       IF(.NOT. SKIP)THEN
          NHIST=NHIST+1
          IF(.NOT. QDIP)THEN
             VALUE(1)=ONE
             IF(QCHG) VALUE(1)=CG(ID)
             IF(QWEIG) VALUE(1)=VALUE(1)*WMAIN(ID)
             HIST(1,IX,IY,IZ)=HIST(1,IX,IY,IZ)+VALUE(1)
          ELSE
             IS=IGPBS(GETRES(ID,IGPBS,NGRP))+1
             DO J=1,NGRAT
                IGRAT(J)=IS+J-1
             ENDDO
             CALL CDIPOLE(NATOM,X,Y,Z,CG,NGRAT,IGRAT,VALUE(1), &
                  VALUE(2),VALUE(3),QTOT,.TRUE.,.FALSE.,dummy)
             ! maybe this is a crazy way of stepping thru our large matrix...
             IF(QWEIG)THEN
                VALUE(1)=VALUE(1)*WMAIN(ID)
                VALUE(2)=VALUE(2)*WMAIN(ID)
                VALUE(3)=VALUE(3)*WMAIN(ID)
             ENDIF
             HIST(1,IX,IY,IZ)=HIST(1,IX,IY,IZ) + VALUE(1)
             HIST(2,IX,IY,IZ)=HIST(2,IX,IY,IZ) + VALUE(2)
             HIST(3,IX,IY,IZ)=HIST(3,IX,IY,IZ) + VALUE(3)
          ENDIF
       ENDIF
    ENDDO
    RETURN
  END SUBROUTINE SLVHIS

  !-----------------------------------------------------------------------
  SUBROUTINE CORFUNCQ(ISTEP,NCORS,NATOMS,VRED,ISTACK,VAC,INORM)
    !  SAPATEL 10/19/01
  use number
  use stream
  use memory

    INTEGER NCORS,NATOMS
    INTEGER ISTEP,I,IATM,NDIM,K,ISTK,J,JC,IST,JS
    real(chm_real) VRED(*)
    INTEGER ISTACK(NATOMS)
    real(chm_real) VAC(*)
    real(chm_real),allocatable,dimension(:,:) :: VX
    INTEGER INORM(*)
    !
    INTEGER IPRV
    real(chm_real) XD,YD,ZD,XP,YP,ZP

    call chmalloc('solana.src','CORFUNCQ','VX',NCORS,Natoms,crl=VX)
    write(110,*)"IN CORFUNCQ"

    !     Initialize stack pointer, correlation function and normalization
    IF (ISTEP <= 0) THEN
       ISTACK(1:natoms)=-1
       INORM(1:ncors)=0
       VAC(1:ncors)=ZERO
    ENDIF

    ISTEP=ISTEP+1

    !     Loop over atoms
    loop500: DO IATM=1,NATOMS
       !
       IF(ISTACK(IATM) > 0) ISTACK(IATM)=ISTACK(IATM)+1
       !     Initialize stack if appropiate.
       IF(ISTACK(IATM) < 0) ISTACK(IATM)=1
       ISTK=ISTACK(IATM)
       !     Now look for trajectoriesps longer than NCORS steps.
       IF(ISTK <= NCORS) THEN
          !     Fill stack now.
          VX(ISTK,IATM)=VRED(IATM)
          !     Compute correlations.
          DO J=1,ISTK
             JC=ISTK-J+1
             VAC(J)=VAC(J)+VX(JC,IATM)*VX(ISTK,IATM)
             !            IF (IATM == 5) write(OUTU,*) J,VAC(J)
             !            IF (IATM == 5) write(OUTU,*) VX(JC,IATM),VX(ISTK,IATM)
             INORM(J)=INORM(J)+1
          ENDDO
          cycle loop500
       ENDIF
       !     Now the special loop for ISTK > NCORS
       IST=ISTK-NCORS*(ISTK/NCORS)
       IF (IST == 0) IST=NCORS
       !
       !     Fill stack
       VX(IST,IATM)=VRED(IATM)
       !     Compute correlations for bins above IST.
       DO J=1,IST
          JC=IST-J+1
          VAC(J)=VAC(J)+VX(JC,IATM)*VX(IST,IATM)
          INORM(J)=INORM(J)+1
       ENDDO
       !
       !     Now include correlations for bins below IST.
       JS=IST+1
       DO J=JS,NCORS

          VAC(NCORS+JS-J)=VAC(NCORS+JS-J)+VX(J,IATM)*VX(IST,IATM)
          INORM(NCORS+JS-J)=INORM(NCORS+JS-J)+1
       ENDDO

    enddo loop500
    !
    write(OUTU,*) VAC(1), VAC(2), VAC(3), VAC(4)
    call chmdealloc('solana.src','CORFUNCQ','VX',NCORS,Natoms,crl=VX)
    RETURN
  END SUBROUTINE CORFUNCQ

  !---------------------------------------------------------------------

  SUBROUTINE CORFUNCB(ISTEP,NCORS,NATOMS,CORED,VRED,HX1,HX2, &
       RSPIN2,RSPOUT2,RRES2,NSITE,REF,QVEL,QMRD,VX,VY,VZ,ISTACK,VAC, &
       INORM,QROTP1,QROTP2)
    !----------------------------------------------------------------------C
    !     This routine computes the correlation functions (in this         C
    !  case velocity correlation function) for a system of atoms (NATOMS)  C
    !  for NCORS timesteps, for a trajectory of length NSTEPS   It uses a  C
    !  rotating stack to store the required velocities (VX(NCORS,NATOMS),  C
    !  VY(NCORS,NATOMS) and VZ(NCORS,NATOMS)) and accumulates the          C
    !  correlation function in VAC(NCORS)   The normalization for each     C
    !  bin in VAC (i.e., VAC(IBIN)) is accumulated in INORM(NCORS)         C
    !  The coordinates and/or velocities are passed to CORFUN at each      C
    !  timestep form the main calling program   They are stored in         C
    !  CORED(NDIM,NATOMS) and VRED(NDIM,NATOMS), respectively   A stack    C
    !  pointer ISTACK(NATOMS) designates the correct position in the       C
    !  stacks VX, VY, and VZ for each atom   This is also used to          C
    !  re-initialize interrupted trajectories                              C
    !----------------------------------------------------------------------C
    !     Written by   Charles L. Brooks III                               C
    !     Department of Chemistry                                          C
    !     Harvard University                                               C
    !                                                                      C
    !     March, 1983                                                      C
    !----------------------------------------------------------------------C
    !     First dimension working arrays

  use number
  use stream

    INTEGER NCORS,NATOMS,NSITE
    INTEGER ISTEP,I,IATM,NDIM,K,ISTK,J,JC,IST,JS
    real(chm_real) R2,RRES
    real(chm_real) RSPIN2,RSPOUT2,RRES2,REF(3,NSITE)
    real(chm_real) VRED(3,NATOMS),CORED(3,NATOMS), &
         HX1(3,NATOMS),HX2(3,NATOMS)
    real(chm_real) VX(NCORS,NATOMS),VY(NCORS,NATOMS),VZ(NCORS,NATOMS)
    INTEGER ISTACK(NATOMS)
    real(chm_real) VAC(*)
    INTEGER INORM(*)
    LOGICAL QVEL,QMRD, QROK, QROK1, QROTP1, QROTP2, QROT
    !
    INTEGER IPRV
    real(chm_real) XD,YD,ZD,XP,YP,ZP, RR
    !
    !     Initialize stack pointer, correlation function and normalization
    IF (ISTEP <= 0) THEN
       DO I=1,NATOMS
          ISTACK(I)=-1
       ENDDO
       DO I=1,NCORS
          INORM(I)=0
          VAC(I)=ZERO
          VZ(I,1)=ZERO
       ENDDO
    ENDIF
    !
    QROT=QROTP1 .OR. QROTP2
    ISTEP=ISTEP+1
    !
    !     Loop over atoms

    loop500: DO IATM=1,NATOMS
       !
       !     Do checking for interrupt criterion
       ! This does currently only work for coordinates, not velocities!
       ! Is OK for rotational correlation of water and MRD 
       R2=ZERO
       QROK=.FALSE.
       IF (.NOT.QVEL .AND.(RSPOUT2 > RSPIN2)) THEN
          IF (NSITE == 1) THEN
             DO K=1,3
                R2=R2+(CORED(K,IATM) - REF(K,1))**2
             ENDDO
          ELSE
             !
             ! Have to check minimum distance to the site atoms
             XP=CORED(1,IATM)
             YP=CORED(2,IATM)
             ZP=CORED(3,IATM)
             R2= (XP-REF(1,1))**2 + &
                  (YP-REF(2,1))**2 + &
                  (ZP-REF(3,1))**2
             DO K=2,NSITE
                R2= MIN(R2,(XP-REF(1,K))**2 + &
                     (YP-REF(2,K))**2 + &
                     (ZP-REF(3,K))**2 )
             ENDDO
          ENDIF
          QROK=(R2 < RSPOUT2) .AND. (R2 > RSPIN2)
       ENDIF
       !
       !     Now do stack manipulations   First propagate stack.

       IF(RRES2 > ZERO)THEN
          QROK1=R2 < RRES2
       ELSE
          QROK1=QROK
       ENDIF
       IF (.NOT. QROK .AND. .NOT. (QVEL.OR.QMRD)) THEN
          !     Terminate stack if appropriate (no QROK for VAC,and we don't 
          !     terminate stack for MRD)
          IF(ISTACK(IATM) > 0) ISTACK(IATM)=-1
          IF(ISTACK(IATM) < 0) cycle loop500
       ENDIF
       IF(ISTACK(IATM) > 0) ISTACK(IATM)=ISTACK(IATM)+1
       !     Initialize stack if appropiate.
       IF(ISTACK(IATM) < 0) ISTACK(IATM)=1
       ISTK=ISTACK(IATM)
       !     Now look for trajectories longer than NCORS steps.
       IF(ISTK <= NCORS) THEN
          !     Fill stack now.
          ! LNI+  Check for PBC w/ minimum images and correct to remove jumps!
          ! Allen&Tildesley for simple lattice
          !
          IF(MINI .AND. ISTK > 1)THEN
             IPRV=ISTK-1
             XD= VRED(1,IATM) - VX(IPRV,IATM)
             YD= VRED(2,IATM) - VY(IPRV,IATM)
             ZD= VRED(3,IATM) - VZ(IPRV,IATM)
             XD=ANINT(XD/XBOX)*XBOX
             YD=ANINT(YD/YBOX)*YBOX
             ZD=ANINT(ZD/ZBOX)*ZBOX
             VRED(1,IATM)=VRED(1,IATM)-XD
             VRED(2,IATM)=VRED(2,IATM)-YD
             VRED(3,IATM)=VRED(3,IATM)-ZD
             ! need to add HX1  & HX2 corrections also 
             IF(QROT)THEN
                HX1(1,IATM)=HX1(1,IATM)-XD
                HX2(1,IATM)=HX2(1,IATM)-XD
                HX1(2,IATM)=HX1(2,IATM)-YD
                HX2(2,IATM)=HX2(2,IATM)-YD
                HX1(3,IATM)=HX1(3,IATM)-ZD
                HX2(3,IATM)=HX2(3,IATM)-ZD
             ENDIF
             ! LNI- end minimum image correction
          ENDIF
          !
          IF(QMRD)THEN
             ! For MRD we need P2(z-hat)/r**3, which we keep in VX if r is
             !  within RSPIN, RSPOUT shell
             ! Also keep 0 or 1 in VY, depending on in whether r is < RRES (or inbetween
             ! RSPIN, RSPOUT if RRES is not specified)
             ! for computation of the "INTERMITTENT"
             ! (ie, we don't care what the distance has
             ! been in the meantime) residence time correlation fcn in VZ
             !
             IF(QROK) THEN
                VX(ISTK,IATM)= (THREE*(VRED(3,IATM)-REF(3,1))**2-R2)/ &
                     (TWO*R2**(FIVE/TWO))
             ELSE
                VX(ISTK,IATM)=ZERO
             ENDIF
             IF(QROK1)THEN
                VY(ISTK,IATM)= ONE
             ELSE
                VY(ISTK,IATM)=ZERO
             ENDIF
          ELSEIF(QROT) THEN
             ! Get normalized vector along water dipole
             VX(ISTK,IATM)= &
                  HALF* (HX1(1,IATM) + HX2(1,IATM)) - VRED(1,IATM)
             VY(ISTK,IATM)= &
                  HALF* (HX1(2,IATM) + HX2(2,IATM)) - VRED(2,IATM)
             VZ(ISTK,IATM)= &
                  HALF* (HX1(3,IATM) + HX2(3,IATM)) - VRED(3,IATM)
             RR=MAX(RSMALL,VX(ISTK,IATM)**2+VY(ISTK,IATM)**2 + &
                  VZ(ISTK,IATM)**2)
             RR=ONE/SQRT(RR)
             VX(ISTK,IATM)=RR*VX(ISTK,IATM)
             VY(ISTK,IATM)=RR*VY(ISTK,IATM)
             VZ(ISTK,IATM)=RR*VZ(ISTK,IATM)
          ELSE
             VX(ISTK,IATM)=VRED(1,IATM)
             VY(ISTK,IATM)=VRED(2,IATM)
             VZ(ISTK,IATM)=VRED(3,IATM)
          ENDIF
          !
          !     Compute correlations.
          DO J=1,ISTK
             JC=ISTK-J+1
             IF (QVEL) THEN
                VAC(J)=VAC(J)+VX(JC,IATM)*VX(ISTK,IATM)+ &
                     VY(JC,IATM)*VY(ISTK,IATM)+ &
                     VZ(JC,IATM)*VZ(ISTK,IATM)
             ELSEIF(QMRD)THEN
                VAC(J)=VAC(J)+VX(JC,IATM)*VX(ISTK,IATM)
                VZ(J,1)=VZ(J,1)+VY(JC,IATM)*VY(ISTK,IATM)
                ! For normalization we need to accumulate number of frame pairs used for each
                ! time lag
                ! (but the sum over waters is really a sum, so we do it only for the first
                !  atom in each frame)
                IF(IATM == 1) INORM(J)=INORM(J)+1
             ELSEIF(QROT)THEN
                RR=VX(JC,IATM)*VX(ISTK,IATM)+ &
                     VY(JC,IATM)*VY(ISTK,IATM)+ &
                     VZ(JC,IATM)*VZ(ISTK,IATM)
                IF(QROTP1)THEN
                   VAC(J)=VAC(J) + RR
                ELSE
                   ! Form final P2 when function is printed; just collect <x**2> here
                   VAC(J)=VAC(J)+ RR*RR
                ENDIF
             ELSE
                VAC(J)=VAC(J)+(VX(JC,IATM)-VX(ISTK,IATM))**2+ &
                     (VY(JC,IATM)-VY(ISTK,IATM))**2+ &
                     (VZ(JC,IATM)-VZ(ISTK,IATM))**2
             ENDIF
             IF(.NOT. QMRD) INORM(J)=INORM(J)+1
          ENDDO
          cycle loop500
       ENDIF
       !
       !     Now the special loop for ISTK > NCORS
       IST=ISTK-NCORS*(ISTK/NCORS)
       IF (IST == 0) IST=NCORS
       !
       !     Fill stack
       ! LNI+  Check for PBC w/ minimum images and correct to remove jumps!
       ! Allen&Tildesley for simple lattice
       IF(MINI)THEN
          !
          ! Here we have to find out which was really the previous step
          IPRV=IST-1
          IF(IPRV  ==  0) IPRV=NCORS
          XD= VRED(1,IATM) - VX(IPRV,IATM)
          YD= VRED(2,IATM) - VY(IPRV,IATM)
          ZD= VRED(3,IATM) - VZ(IPRV,IATM)
          XD=ANINT(XD/XBOX)*XBOX
          YD=ANINT(YD/YBOX)*YBOX
          ZD=ANINT(ZD/ZBOX)*ZBOX
          !           VX(IST,IATM)=VX(IPRV,IATM)+XD
          !           VY(IST,IATM)=VY(IPRV,IATM)+YD
          !           VZ(IST,IATM)=VZ(IPRV,IATM)+ZD

          VRED(1,IATM)=VRED(1,IATM)-XD
          VRED(2,IATM)=VRED(2,IATM)-YD
          VRED(3,IATM)=VRED(3,IATM)-ZD
          ! For QROT we need to corret HX1 and HX2 as well
          IF(QROT)THEN
             HX1(1,IATM)=HX1(1,IATM)-XD
             HX2(1,IATM)=HX2(1,IATM)-XD
             HX1(2,IATM)=HX1(2,IATM)-YD
             HX2(2,IATM)=HX2(2,IATM)-YD
             HX1(3,IATM)=HX1(3,IATM)-ZD
             HX2(3,IATM)=HX2(3,IATM)-ZD
          ENDIF
          ! LNI- end minimum image correction
       ENDIF
       !
       IF(QMRD)THEN
          IF(QROK) THEN
             VX(IST,IATM)= (THREE*(VRED(3,IATM)-REF(3,1))**2-R2)/ &
                  (TWO*R2**(FIVE/TWO))
          ELSE
             VX(IST,IATM)=ZERO
          ENDIF
          IF(QROK1)THEN
             VY(IST,IATM)=ONE
          ELSE
             VY(IST,IATM)=ZERO
          ENDIF
       ELSEIF(QROT)THEN
          ! Get normalized vector along water dipole
          VX(IST,IATM)= &
               HALF* (HX1(1,IATM) + HX2(1,IATM)) - VRED(1,IATM)
          VY(IST,IATM)= &
               HALF* (HX1(2,IATM) + HX2(2,IATM)) - VRED(2,IATM)
          VZ(IST,IATM)= &
               HALF* (HX1(3,IATM) + HX2(3,IATM)) - VRED(3,IATM)
          RR=MAX(RSMALL,VX(IST,IATM)**2+VY(IST,IATM)**2 + &
               VZ(IST,IATM)**2)
          RR=ONE/SQRT(RR)
          VX(IST,IATM)=RR*VX(IST,IATM)
          VY(IST,IATM)=RR*VY(IST,IATM)
          VZ(IST,IATM)=RR*VZ(IST,IATM)
       ELSE
          VX(IST,IATM)=VRED(1,IATM)
          VY(IST,IATM)=VRED(2,IATM)
          VZ(IST,IATM)=VRED(3,IATM)
       ENDIF
       !
       !     Compute correlations for bins above IST.
       DO J=1,IST
          JC=IST-J+1
          IF (QVEL) THEN
             VAC(J)=VAC(J)+VX(JC,IATM)*VX(IST,IATM)+ &
                  VY(JC,IATM)*VY(IST,IATM)+ &
                  VZ(JC,IATM)*VZ(IST,IATM)
          ELSEIF(QMRD)THEN
             VAC(J)=VAC(J)+ VX(JC,IATM)*VX(IST,IATM)
             VZ(J,1)=VZ(J,1)+ VY(JC,IATM)*VY(IST,IATM)
             IF(IATM == 1) INORM(J)=INORM(J)+1
          ELSEIF(QROT)THEN
             RR=VX(JC,IATM)*VX(IST,IATM)+ &
                  VY(JC,IATM)*VY(IST,IATM)+ &
                  VZ(JC,IATM)*VZ(IST,IATM)
             IF(QROTP1)THEN
                VAC(J)=VAC(J) + RR
             ELSE
                VAC(J)=VAC(J)+ RR*RR 
             ENDIF
          ELSE
             VAC(J)=VAC(J)+(VX(JC,IATM)-VX(IST,IATM))**2+ &
                  (VY(JC,IATM)-VY(IST,IATM))**2+ &
                  (VZ(JC,IATM)-VZ(IST,IATM))**2
          ENDIF
          IF(.NOT. QMRD) INORM(J)=INORM(J)+1
       ENDDO
       !
       !     Now include correlations for bins below IST.
       JS=IST+1
       DO J=JS,NCORS
          IF (QVEL) THEN
             VAC(NCORS+JS-J)=VAC(NCORS+JS-J)+VX(J,IATM)*VX(IST,IATM)+ &
                  VY(J,IATM)*VY(IST,IATM)+ &
                  VZ(J,IATM)*VZ(IST,IATM)
          ELSEIF(QMRD)THEN
             VAC(NCORS+JS-J)=VAC(NCORS+JS-J)+VX(J,IATM)*VX(IST,IATM)
             VZ(NCORS+JS-J,1)=VZ(NCORS+JS-J,1)+VY(J,IATM)*VY(IST,IATM)
             IF(IATM == 1) INORM(NCORS+JS-J)=INORM(NCORS+JS-J)+1
          ELSEIF(QROT)THEN
             RR=VX(J,IATM)*VX(IST,IATM)+ &
                  VY(J,IATM)*VY(IST,IATM)+ &
                  VZ(J,IATM)*VZ(IST,IATM)
             IF(QROTP1)THEN
                VAC(NCORS+JS-J)=VAC(NCORS+JS-J) + RR
             ELSE
                VAC(NCORS+JS-J)=VAC(NCORS+JS-J)+RR*RR
             ENDIF
          ELSE
             VAC(NCORS+JS-J)=VAC(NCORS+JS-J)+ &
                  (VX(J,IATM)-VX(IST,IATM))**2+ &
                  (VY(J,IATM)-VY(IST,IATM))**2+ &
                  (VZ(J,IATM)-VZ(IST,IATM))**2
          ENDIF
          IF(.NOT. QMRD) INORM(NCORS+JS-J)=INORM(NCORS+JS-J)+1
       ENDDO
       !
    enddo loop500
    !
    RETURN
  END SUBROUTINE CORFUNCB

  SUBROUTINE PRNTVAC(IUNIT,NCORS,DT,VAC,INORM,VNORM)
    !-----------------------------------------------------------------------
  use consta
  use number
  use stream

    real(chm_real) VAC(*)
    INTEGER INORM(*)
    real(chm_real) VNORM(*)
    real(chm_real) VINT,VIINT,T,DT,CHNT
    INTEGER I,NCORS,NC,IUNIT
    !
    IF(IUNIT  <=  0) RETURN
    !     Now normalize correlation functions.
    VINT=VAC(1)/INORM(1)
    VIINT=INORM(1)
    IF(PRNLEV >= 2) WRITE(OUTU,15) VINT,VIINT
15  FORMAT(5X,'INITIAL VALUE OF CORRELATION',1X,F12.5, &
         ' # OF POINTS ',F12.5,/)
    !
    DO I=1,NCORS
       IF(INORM(I) > 0) THEN
          VNORM(I)=INORM(I)/VIINT
          VAC(I)=VAC(I)/INORM(I)
       ENDIF
       !       Normalize to 1.0 at t=0.0
       ! No, not a good idea (we are not estimating a correlation
       ! time from the integral of an exponential): Integral(vac) = 3D
       !        VAC(I)=VAC(I)/VINT
       ! instead rescale to A/ps as velocity unit instead of A/akma
       VAC(I)=VAC(I)/TIMFAC**2
    ENDDO
    !
    DO I=1,NCORS
       T=(I-1)*DT
       IF(IOLEV >= 1) WRITE(IUNIT,25)T,VAC(I),VNORM(I)

       IF(PRNLEV >= 2) WRITE (OUTU,25)T,VAC(I)
    ENDDO
25  FORMAT(5X,E12.5,1X,E12.5,1X,E12.5)
    !
    !    ( Compute relaxation time) from integral of correlation function.
    !     Go for the diffusion coefficient D= (integral of correlation fcn)/3
    !     in A**2/ps
    NC=NCORS-1
    CHNT=ZERO
    DO I=2,NC
       CHNT=CHNT+VAC(I)
    ENDDO
    !
    CHNT=DT*(CHNT+.5*(VAC(1)+VAC(NCORS)))
    IF(PRNLEV >= 2) WRITE (OUTU,35) CHNT/THREE
35  FORMAT(5X,'D =',E12.5,' A**2/ps (NB! w/o longterm correction!)')
    !
    RETURN
  END SUBROUTINE PRNTVAC

  SUBROUTINE PRNROT(IUNIT,NCORS,DT,VAC,INORM,VNORM,QROTP2)
    !-----------------------------------------------------------------------
  use consta
  use number
  use stream

    real(chm_real) VAC(*)
    INTEGER INORM(*)
    real(chm_real) VNORM(*)
    real(chm_real) VINT,VIINT,T,DT,CHNT
    INTEGER I,NCORS,NC,IUNIT
    LOGICAL QROTP2
    !
    !     Now normalize correlation functions.
    VINT=VAC(1)/INORM(1)
    VIINT=INORM(1)
    IF(PRNLEV >= 2) WRITE(OUTU,15) VINT,VIINT
15  FORMAT(5X,'INITIAL VALUE OF CORRELATION',1X,G15.6, &
         ' # OF POINTS ',F12.0,/)
    !
    DO I=1,NCORS
       IF(INORM(I) > 0) THEN
          VNORM(I)=INORM(I)/VIINT
          VAC(I)=VAC(I)/INORM(I)
       ENDIF
       IF(QROTP2)THEN
          VAC(I)=HALF*(THREE*VAC(I)-ONE) 
       ELSE
          !       Normalize to 1.0 at t=0.0; may not be necessary
          VAC(I)=VAC(I)/VINT
       ENDIF
    ENDDO
    !
    DO I=1,NCORS
       T=(I-1)*DT
       IF(IOLEV >= 1.AND.IUNIT > 0 ) WRITE(IUNIT,25)T,VAC(I),VNORM(I)
       IF(PRNLEV >= 2) WRITE (OUTU,25)T,VAC(I)
    ENDDO
25  FORMAT(5X,E12.5,1X,E12.5,1X,E12.5)
    !
    !     Compute relaxation time from integral of correlation function.
    NC=NCORS-1
    CHNT=ZERO
    DO I=2,NC
       CHNT=CHNT+VAC(I)
    ENDDO
    !
    CHNT=DT*(CHNT+.5*(VAC(1)+VAC(NCORS)))
    IF(PRNLEV >= 2) WRITE (OUTU,35) CHNT
35  FORMAT(5X,'ESTIMATE OF TAU =',E12.5, &
         ' ps (assuming normalized C(t) decays to zero)')
    !
    RETURN
  END SUBROUTINE PRNROT


  SUBROUTINE PRNTMRD(IUNIT,NCORS,DT,G,Q,INORM)
    !-----------------------------------------------------------------------
  use consta
  use number
  use stream

    real(chm_real) G(*), Q(*)
    real(chm_real) RN, RNZ,DF,DF1
    real(chm_real) T,DT,TAU,TAUZ, VINT, VZINT
    INTEGER I,NCORS,IUNIT,INORM(*)
    !
    IF(IUNIT  <=  0) RETURN
    IF(NCORS < 3)THEN
       IF(PRNLEV >= 2) WRITE(OUTU,*) ' NO MRD OUTPUT, NCORS < 3'
       RETURN
    ENDIF
    VINT=G(1)/INORM(1)
    IF(G(1) <= ZERO) VINT=ONE
    VZINT=Q(1)/INORM(1)
    IF(Q(1) <= ZERO) VZINT=ONE
    IF(PRNLEV >= 2) WRITE(OUTU,15) VINT,VZINT, &
         INORM(1),INORM(2),INORM(NCORS)
15  FORMAT(5X,'INITIAL VALUES OF: CORRELATION= ',G12.5, &
         ' RESIDENCE CORR. ',G12.5,/5X,'INORM(1,2,NCORS):',3I12,/)
    !
    DO I=1,NCORS
       RN=ONE
       RNZ=ONE
       IF(INORM(I) > 0)THEN
          RN= ONE/(VINT*FLOAT(INORM(I)))
          RNZ= ONE/(VZINT*FLOAT(INORM(I)))
       ENDIF
       T=(I-1)*DT
       G(I)=G(I)*RN
       Q(I)=Q(I)*RNZ
    ENDDO
    !
    ! central difference approx. for the derivative, w/ linear extrapolation at ends
    IF(NCORS <= 3) THEN
       DF=Q(NCORS)-Q(1)/(DT*(NCORS-1))
    ELSE
       DF=(TWO*Q(2)-(THREE*Q(1)+Q(3))/TWO)/DT
    ENDIF
    ! Output:
    ! time, dipolar correlation, residence time corrlation, residence distribution
    !
    IF(IOLEV >= 1) THEN
       WRITE(IUNIT,'(5X,5A13)') 'Time','G','G-norm','Q','F'
       WRITE(IUNIT,25) ZERO,G(1)*VINT,G(1),Q(1),-DF
    ENDIF
    IF(PRNLEV >= 2) THEN
       WRITE(OUTU,'(5X,5A13)') &
            'Time','Dip.corr(G)','G-norm','Res.corr(Q)','Res.dist.(F)'
       WRITE (OUTU,25) ZERO,G(1)*VINT,G(1),Q(1),-DF
    ENDIF
    DO I=2,NCORS-1
       DF1=DF
       DF=(Q(I+1)-Q(I-1))/(TWO*DT)
       T=(I-1)*DT
       IF(IOLEV >= 1) WRITE(IUNIT,25) T,G(I)*VINT,G(I),Q(I),-DF
       IF(PRNLEV >= 2) WRITE (OUTU,25)T,G(I)*VINT,G(I),Q(I),-DF
    ENDDO
    DF=TWO*DF-DF1
    T=(NCORS-1)*DT
    IF(IOLEV >= 1) &
         WRITE(IUNIT,25) T,G(NCORS)*VINT,G(NCORS),Q(NCORS),-DF
    IF(PRNLEV >= 2) &
         WRITE(OUTU,25)  T,G(NCORS)*VINT,G(NCORS),Q(NCORS),-DF
    !
25  FORMAT(5X,5E13.5)
    !
    !  Compute relaxation times from integral of NORMALIZED correlation function.
    TAU=(G(1)+G(NCORS))/TWO
    TAUZ=(Q(1)+Q(NCORS))/TWO
    DO I=2,NCORS-1
       TAU=TAU+G(I)
       TAUZ=TAUZ+Q(I)
    ENDDO
    TAU=DT*TAU
    TAUZ=DT*TAUZ
    !
    IF(PRNLEV >= 2) WRITE (OUTU,35) TAU,TAUZ
35  FORMAT(/5X,'ESTIMATED dipolar cross relaxation correlation time', &
         /5X,'TAU =',E12.5,' ps',// &
         5X,'ESTIMATED residence correlation time', &
         /5X,'TAU =',E12.5,' ps',/)
    !
    RETURN
  END SUBROUTINE PRNTMRD

  SUBROUTINE PRNTMSD(IUNIT,NCORS,DT,VAC,INORM,VNORM,TIM)
    !-----------------------------------------------------------------------
  use number
  use stream
  use param_store, only: set_param

  implicit none

    real(chm_real) VAC(*),TIM(*)
    INTEGER INORM(*)
    real(chm_real) VNORM(*)
    real(chm_real) VINT,T,DT
    INTEGER  I,NCORS,IUNIT
    !
    INTEGER NP,N1
    real(chm_real) DC,DCSD,A,ASD,RXY,DC1
    !
    IF(IUNIT  <=  0) RETURN
    !     Now normalize correlation functions.
    VINT=VAC(1)/INORM(1)
    IF(PRNLEV >= 2) WRITE (OUTU,15) VINT,INORM(1)
15  FORMAT(5X,'MSD: INITIAL VALUE OF CORRELATION',1X,F12.5, &
         ' # OF POINTS ',I12,/)
    !
    DO I=1,NCORS
       IF(INORM(I) > 0) VAC(I)=VAC(I)/FLOAT(INORM(I))
    ENDDO
    DO I=1,NCORS
       T=(I-1)*DT
       TIM(I)=T
       IF(IOLEV >= 1) WRITE(IUNIT,25)T,VAC(I)
       IF(PRNLEV >= 2) WRITE (OUTU,25)T,VAC(I)
    ENDDO
25  FORMAT(5X,E12.5,1X,E12.5)
    !
    ! Make a guess of the diffusion coefficient from MSD vs time
    ! using the last 4/5  of MSD(t)
    NP=(4*NCORS)/5
    N1=NCORS+1-NP
    CALL LINREG(NP,TIM(N1),VAC(N1),A,ASD,DC,DCSD,RXY)
    DC=DC/SIX
    call set_param('DCOEFF',DC)
    ! Also convert to some kind of useful dimension, like (cm**2)/s
    DC1=DC*PT0001
    IF(PRNLEV >= 2)THEN
       WRITE(OUTU,30) NP
30     FORMAT(//' ESTIMATE of diffusion coefficient from ',I5, &
            ' last points of MSD(t):')
       WRITE(OUTU,35) DC,DC1,DCSD/SIX,RXY,A
35     FORMAT('  D=',G12.5,' [A**2/ps] (or',G12.5,' [cm**2/s])', &
            /' SD=',G12.5,'     R=',F6.4,'   Y-intercept=',G12.5/)
    ENDIF
    RETURN
  END SUBROUTINE PRNTMSD

  SUBROUTINE RKIRKG(NOXY,OX2,HX1,HX2,DICOR,MJ,RSP,NKIRK, &
       NSITE,SREF)
    !-----------------------------------------------------------------------
  use consta
  use number
  use stream

    !     This routine calculates the distance dependent
    !     the Kirkwood G-factor for all selected waters
    !     G(r)=sum(mui*muj)/mu**2, r < rij
    !     This routine only computes the sum for j>i (avoids counting same thing
    !     twice). To get same data as in the Hocthl paper (the one is from the
    !     dot product with the same water, which also is not done here):
    !     gk(r)=1+2*gk(r)
    !     LNI March 2002 (Hochtl et al JCP 109, 4927)
    !
    !     Passed varibles
    !
    INTEGER NOXY,NKIRK,NSITE
    real(chm_real) OX2(3,NOXY),HX1(3,NOXY),HX2(3,NOXY),MJ(3,NKIRK)
    real(chm_real) DICOR(NKIRK),RSP,SREF(3,NSITE)
    !
    !     Local variables
    !
    real(chm_real) MI(3),REF(3),ROXY,MDOTR,RSPINV,XD,YD,ZD, &
         RNORM,RNORM1
    INTEGER L,K,I,J,MPOINTS,IOFFS
    LOGICAL QSITE
    !
    INTEGER IFMIN
    !
    RSPINV=ONE/RSP
    MPOINTS=NOXY
    QSITE=.FALSE.
    IOFFS=1
    ! Need normalized "dipole" vectors, so find their length:
    RNORM=ZERO
    DO K=1,3
       RNORM=RNORM+ ((HX1(K,1)+HX2(K,1))*HALF - OX2(K,1))**2
    ENDDO
    IF(NSITE  >=  2)THEN
       !
       ! Doing scalar product between site and the waters, find site-vector
       !
       RNORM1=ZERO
       DO K=1,3
          REF(K)= (SREF(K,1)+SREF(K,2))/TWO
          MI(K) =  SREF(K,2) - SREF(K,1)
          RNORM1=RNORM1+MI(K)*MI(K)
       ENDDO
       MPOINTS=1
       IOFFS=0
       QSITE=.TRUE.
    ENDIF
    loop55: DO I=1,MPOINTS
       IF(.NOT. QSITE ) THEN
          DO K=1,3
             REF(K)=OX2(K,I)
             MI(K)= (HX1(K,I)+HX2(K,I))*HALF - OX2(K,I)
          ENDDO
       ENDIF
       DO J=1,NKIRK
          DO K=1,3
             MJ(K,J)=ZERO
          ENDDO
       ENDDO
       !
       loop50: DO J=I+IOFFS,NOXY
          ROXY=ZERO
          XD=OX2(1,J)-REF(1)
          YD=OX2(2,J)-REF(2)
          ZD=OX2(3,J)-REF(3)
          IF(MINI)THEN
             ! Check for PBC w/ minimum images (Allen&Tildesley for simple lattice)
             XD=XD-ANINT(XD/XBOX)*XBOX
             YD=YD-ANINT(YD/YBOX)*YBOX
             ZD=ZD-ANINT(ZD/ZBOX)*ZBOX
          ENDIF
          ROXY=XD*XD + YD*YD + ZD*ZD
          IF(ROXY  <  RSP)THEN
             L=INT(SQRT(ROXY*RSPINV)*NKIRK + ONE)
             DO K=1,3
                MJ(K,L)=MJ(K,L)+(HX1(K,J)+HX2(K,J))*HALF - OX2(K,J)
             ENDDO
             !               KRKD(L)=KRKD(L)+1
          ENDIF
       enddo loop50
       !
       ! Dot product, normalization and accumulation of result
       MDOTR=ZERO
       IF(QSITE)THEN
          RNORM1=ONE/SQRT(RNORM*RNORM1)
       ELSE
          RNORM1=ONE/RNORM
       ENDIF
       DO J=1,NKIRK
          DO K=1,3
             MDOTR=MDOTR + MI(K)*MJ(K,J)
          ENDDO
          DICOR(J)=DICOR(J)+MDOTR * RNORM1
       ENDDO
    enddo loop55
    !
    RETURN
  END SUBROUTINE RKIRKG

  SUBROUTINE PRNTGKR(IUNIT,DICOR,RSP,NKIRK,NSITE,RNORM)
    !-----------------------------------------------------------------------
  use chm_kinds
  use consta
  use number
  use stream
    implicit none
    INTEGER NCONF,I,NKIRK,IUNIT,NSITE
    !     INTEGER KRKD(NKIRK)
    real(chm_real) DICOR(NKIRK),RSP,R,R1,RNORM
    !
    IF(IUNIT  <=  0) RETURN
    R1=SQRT(RSP)
    IF(PRNLEV >= 2)THEN
       IF(NSITE  >=  2)THEN
          WRITE (OUTU,'(A,G11.4)') &
               ' Site-vector:water "Kirkwood G-factor". RNORM=',RNORM
       ELSE
          WRITE (OUTU,'(A,G11.4)') &
               ' Distance dependent Kirkwood G-factor. RNORM=',RNORM
       ENDIF
    ENDIF
    DO I=1,NKIRK
       !        R=(SQRT(RSP)**3/(NKIRK-1)*I)**(1./3.)
       R=R1*(I-1)/NKIRK +R1/(NKIRK*2)
       IF(PRNLEV >= 2)  WRITE (OUTU,'(A,F8.3,A,F8.3)') &
            ' R=',R,' DICOR(R)=',DICOR(I)/RNORM
       IF(IOLEV >= 1) WRITE(IUNIT,'(2F12.3)') R,DICOR(I)/RNORM
    ENDDO
    !
    RETURN
  END SUBROUTINE PRNTGKR


  SUBROUTINE KIRKG(NOXY,OX2,HX1,HX2,DICOR,RSP,REF,NKIRK,KRKD, &
       NSITE,SREF,LMULTI)
    !-----------------------------------------------------------------------
  use chm_kinds
  use consta
  use number
  use stream
    implicit none
    !
    !     This routine calculates a quantity related to
    !     the Kirkwood G-factor   It is also a measure of the
    !     degree of dipole orientation within the simulation sphere
    !     (i.e., at the boundary, specifically).
    !     Added accumulation of dotproduct vs R, LNI nov 1995
    !
    !     Passed varibles
    !
    INTEGER NOXY,NKIRK,KRKD(61),NSITE
    real(chm_real) OX2(3,NOXY),HX1(3,NOXY),HX2(3,NOXY)
    real(chm_real) DICOR(0:18,61),RSP,REF(3),SREF(3,NSITE)
    LOGICAL LMULTI
    !
    !     Local variables
    !
    real(chm_real) MI(3),MAGI,ROXY,MDOTR,RSPINV
    INTEGER L,K,I,J
    !
    RSPINV=ONE/RSP
    loop55: DO J=1,NSITE
       IF(LMULTI)THEN
          REF(1)=SREF(1,J)
          REF(2)=SREF(2,J)
          REF(3)=SREF(3,J)
       ENDIF
       loop50: DO I=1,NOXY
          MAGI=ZERO
          ROXY=ZERO
          MDOTR=ZERO
          DO K=1,3
             ROXY=ROXY + (OX2(K,I) - REF(K))**2
          ENDDO
          IF(ROXY  <  RSP)THEN
             DO K=1,3
                MI(K)=(HX1(K,I)+HX2(K,I))*HALF - OX2(K,I)
                MAGI=MAGI+MI(K)*MI(K)
                MDOTR=MDOTR + MI(K)*(OX2(K,I) - REF(K))
             ENDDO
             MDOTR=MDOTR/SQRT(ROXY*MAGI)
             K=INT(SQRT(ROXY*RSPINV)*NKIRK + ONE)
             !        K=INT(((ROXY/SQRT(RSP))**3)*(NKIRK-1) + ONE)
             DICOR(0,K)=DICOR(0,K)+MDOTR
             L=INT(ABS(ACOS(MDOTR))*RADDEG*PTONE +ONE)
             IF( (L < 1) .OR. (L > 18) ) THEN
                IF(WRNLEV >= 2) WRITE (OUTU,'(A)') &
                     ' KIRKG: Angle out of bounds'
             ELSE
                DICOR(L,K)=DICOR(L,K) + ONE
             ENDIF
             KRKD(K)=KRKD(K)+1
          ENDIF
       enddo loop50
    enddo loop55
    !
    RETURN
  END SUBROUTINE KIRKG

  SUBROUTINE PRNTGK(IUNIT,DICOR,RSP,NKIRK,KRKD)
    !-----------------------------------------------------------------------
  use chm_kinds
  use consta
  use number
  use stream
    implicit none
    INTEGER NCONF,I,NKIRK,J,KRKD(61),IUNIT
    real(chm_real) DICOR(0:18,61),RSP,R,THETA,R1
    !
    IF(IUNIT  <=  0) RETURN
    R1=SQRT(RSP)
    IF(PRNLEV >= 2)THEN
       WRITE (OUTU,'(A,I8,A)') &
            ' Dipole distribution:'
    ENDIF
    DO I=1,NKIRK
       !        R=(SQRT(RSP)**3/(NKIRK-1)*I)**(1./3.)
       R=R1*(I-1)/NKIRK +R1/(NKIRK*2)
       DO J=1,18
          THETA=(J-1 +HALF)*PI/18.0
          !          DICOR(J,I)=DICOR(J,I)/NCONF/SIN(THETA)
          DICOR(J,I)=DICOR(J,I)/MAX(1,KRKD(I))/SIN(THETA)
       ENDDO
       IF(KRKD(I) >  0) DICOR(0,I)=DICOR(0,I)/KRKD(I)
       IF(PRNLEV >= 2)  WRITE (OUTU,'(A,F8.3,A,20F8.3)') &
            ' R=',R,' DICOR(R)=',(DICOR(J,I),J=0,18)
       IF(IOLEV >= 1) WRITE(IUNIT,'(20(F8.3,1X))') &
            R,(DICOR(J,I),J=0,18)
    ENDDO
    !
    RETURN
  END SUBROUTINE PRNTGK

  SUBROUTINE DENSIT(NATX,X,RSPHER2,REF,NOCONF,NOXYG,NOXDIS)
    !-----------------------------------------------------------------------
    !     calculates density and density profile
    !

  use chm_kinds
  use number
    implicit none
    !
    !
    !     input/output
    INTEGER NATX
    real(chm_real)    X(3,NATX), RSPHER2, REF(3)
    INTEGER NOCONF, NOXYG, NOXDIS(61)
    !     local
    INTEGER I, INOX
    real(chm_real)    RS, DDX, DDY, DDZ
    !     begin
    NOCONF=NOCONF+1
    DO I=1,NATX
       DDX=X(1,I) - REF(1)
       DDY=X(2,I) - REF(2)
       DDZ=X(3,I) - REF(3)
       IF(MINI) THEN
          IF( ABS(DDX)  >  XBOX2 ) DDX=DDX - SIGN(XBOX,DDX)
          IF( ABS(DDY)  >  YBOX2 ) DDY=DDY - SIGN(YBOX,DDY)
          IF( ABS(DDZ)  >  ZBOX2 ) DDZ=DDZ - SIGN(ZBOX,DDZ)
       ENDIF
       RS= DDX*DDX + DDY*DDY + DDZ*DDZ
       IF (RS < RSPHER2) THEN
          INOX=INT( SQRT(RS/RSPHER2)**3*SIXTY + ONE )
          NOXDIS(INOX)=NOXDIS(INOX) + 1
          NOXYG=NOXYG+1
       ENDIF
    ENDDO
    !
    RETURN
  END SUBROUTINE DENSIT

  SUBROUTINE GDIST(NATX,X,NATY,Y,CROS,RSPHER,REF,NCONF,MGN, &
       GN,DR,LMULTI,XSET,YSET)
    !-----------------------------------------------------------------------
    !     calculates radial distribution function g(r) xy
    !
  use chm_kinds
  use number
    implicit none
    !
    LOGICAL LMULTI
    !
    !     input/output
    INTEGER NATX
    real(chm_real)    X(3,NATX)
    INTEGER NATY
    real(chm_real)    Y(3,NATY)
    LOGICAL CROS
    real(chm_real)    RSPHER, REF(3)
    INTEGER NCONF, MGN, GN(MGN),XSET(NATX),YSET(NATY)
    real(chm_real)     DR
    !     local
    INTEGER I, J, IRAD
    real(chm_real)    RS, RIJ, DDX, DDY, DDZ, RIJMAX,DR2
    !
    !     begin
    RIJMAX=((MGN+HALF)*DR)**2
    DR2=(DR/TWO)**2
    loop100: DO I=1,NATX
       DDX=X(1,I) - REF(1)
       DDY=X(2,I) - REF(2)
       DDZ=X(3,I) - REF(3)
       IF(MINI) THEN
          IF(ABS(DDX) > XBOX2) DDX=DDX - SIGN(XBOX,DDX)
          IF(ABS(DDY) > YBOX2) DDY=DDY - SIGN(YBOX,DDY)
          IF(ABS(DDZ) > ZBOX2) DDZ=DDZ - SIGN(ZBOX,DDZ)
       ENDIF
       RS= DDX*DDX + DDY*DDY + DDZ*DDZ
       IF (RS < RSPHER) THEN
          loop50: DO J=1,NATY
             !
             ! We don't really know what are in the arrays, better to check
             ! that IRAD (ie, DR/2<Rij<Rijmax) is OK below
             !
             IF ( XSET(I)  /=  YSET(J) ) THEN
                DDX=X(1,I)-Y(1,J)
                DDY=X(2,I)-Y(2,J)
                DDZ=X(3,I)-Y(3,J)
                IF(MINI) THEN
                   IF(ABS(DDX) > XBOX2) DDX=DDX - SIGN(XBOX,DDX)
                   IF(ABS(DDY) > YBOX2) DDY=DDY - SIGN(YBOX,DDY)
                   IF(ABS(DDZ) > ZBOX2) DDZ=DDZ - SIGN(ZBOX,DDZ)
                ENDIF
                RIJ=DDX*DDX+DDY*DDY+DDZ*DDZ
                IF(RIJ < RIJMAX.AND.RIJ > DR2)THEN
                   RIJ=SQRT(RIJ)
                   IRAD=INT(RIJ/DR + HALF)
                   GN(IRAD)=GN(IRAD) + 1
                ENDIF
             ENDIF
          enddo loop50
          NCONF=NCONF+1
       ENDIF
    enddo loop100
    !
    RETURN
  END SUBROUTINE GDIST

  SUBROUTINE HYDNUM(IHYD,RHYD,NATX,X,XSET,NATY,Y,YSET, &
       DT,NCONF,NRR,NAR,NAA)
    !-----------------------------------------------------------------------
    !     calculates hydration number
    !
  use chm_kinds
  use number
  use stream
  use chutil,only:atomid
    implicit none
    !
    LOGICAL LMULTI
    !
    !     input/output
    INTEGER IHYD,NATX,NATY,NCONF,NAA,NRR,NAR
    INTEGER XSET(*),YSET(*)
    real(chm_real)    X(3,NATX),Y(3,NATY),DT,RHYD
    !     local
    INTEGER I, J, M,MRES,NVAUA,NVRUR,NVAUR
    real(chm_real)    S2, DDX, DDY, DDZ, X1,X2,X3
    character(len=8) SID,SIDO,RID,RIDO,REN,AC
    !
    !     begin
    IF(RHYD <= 0.0)RETURN
    NVAUA=0
    NVRUR=0
    NVAUR=0
    MRES=0
    CALL ATOMID(XSET(1),SIDO,RIDO,REN,AC)
    loop100: DO I=1,NATX
       CALL ATOMID(XSET(I),SID,RID,REN,AC)
       IF(RID /= RIDO.OR.SID.NE.SIDO)THEN
          ! new residue, sum up for previous
          IF(MRES > 0) NVRUR=NVRUR+1
          MRES=0
          SIDO=SID
          RIDO=RID
       ENDIF
       ! keep counting
       X1=X(1,I)
       X2=X(2,I)
       X3=X(3,I)
       M=0
       ! for all selected solute atoms
       loop50: DO J=1,NATY
          IF( XSET(I)  /=  YSET(J) )THEN
             DDX=X1-Y(1,J)
             DDY=X2-Y(2,J)
             DDZ=X3-Y(3,J)
             IF(MINI) THEN
                IF(ABS(DDX) > XBOX2) DDX=DDX - SIGN(XBOX,DDX)
                IF(ABS(DDY) > YBOX2) DDY=DDY - SIGN(YBOX,DDY)
                IF(ABS(DDZ) > ZBOX2) DDZ=DDZ - SIGN(ZBOX,DDZ)
             ENDIF
             S2=DDX*DDX+DDY*DDY+DDZ*DDZ
             IF(S2 < RHYD) M=M+1
          ENDIF
       enddo loop50
       !
       IF(M > 0)THEN
          NVAUR=NVAUR+1
          MRES=MRES+1
       ENDIF
       NVAUA=NVAUA+M
    enddo loop100
    ! sum up for last residue
    IF(MRES > 0) NVRUR=NVRUR+1
    !
    NCONF=NCONF+1
    NRR=NRR+NVRUR
    NAR=NAR+NVAUR
    NAA=NAA+NVAUA
    IF(IHYD > 0.AND.PRNLEV >= 2)THEN
       WRITE(IHYD,'(G12.4,3I10)') NCONF*DT,NVRUR,NVAUR,NVAUA
    ENDIF
    RETURN
  END SUBROUTINE HYDNUM

  SUBROUTINE PRNDNS(IUNIT,NOXYG,RSPHER2,NOCONF,DENS,NOXDIS,NSOLV)
    !-----------------------------------------------------------------------
  use chm_kinds
  use consta
  use number
  use stream
    implicit none
    !
    !     input/output
    INTEGER NOXYG, NOXDIS(61),NSOLV,IUNIT
    real(chm_real) RSPHER2, RI
    INTEGER NOCONF
    real(chm_real) DENS, DENSI
    !
    INTEGER I
    real(chm_real) DVOL
    !
    !     calculate the density if the user has not supplied a value and
    !     if NOCONF and RSPHER2 are not zero
    IF(RSPHER2 <= RSMALL .OR. NOCONF .LE. 0) RETURN
    IF(DENS <= ZERO)THEN
       DENS= NOXYG / ((FOUR/THREE) * PI * (SQRT(RSPHER2) )**3 * NOCONF)
    ENDIF
    IF(PRNLEV >= 2) WRITE (OUTU,1200) DENS
1200 FORMAT(/,5X,'AVERAGE DENSITY ',G16.10)
    IF(IUNIT  <=  0) RETURN
    !
    IF(PRNLEV >= 2) WRITE (OUTU,1300)
1300 FORMAT(5X,'NORMALIZED DENSITY PROFILE',/)
    DVOL=SQRT(RSPHER2)**3/SIXTY
    !
    DO I=1,61
       RI=(DVOL*I)**THIRD
       DENSI= NOXDIS(I) / ((FOUR/THREE) * PI *  DVOL * NOCONF )
       DENSI=DENSI/DENS
       IF(IOLEV >= 1)  WRITE(IUNIT,'(F8.3,1X,F8.3)') RI,DENSI
       IF(PRNLEV >= 2) WRITE (OUTU,'(A,F8.3,A,F8.3)') &
            ' R=',RI,' Density',DENSI
    ENDDO
    !
    RETURN
  END SUBROUTINE PRNDNS

  SUBROUTINE PRNTG(IUNIT,RSPHER,DR,NCONF,MGN,GN,DENS,NPTS,ACCV)
    !-----------------------------------------------------------------------
  use chm_kinds
  use consta
  use number
  use stream
    implicit none
    !     input/output
    INTEGER IUNIT
    real(chm_real) RSPHER, DR
    INTEGER NCONF, MGN, GN(MGN),NPTS
    real(chm_real) DENS,ACCV(NPTS)
    !     local
    real(chm_real) R, GNN, NR, GNR, RNORM, ACVTMP
    INTEGER I
    !
    IF(IUNIT  <=  0 .OR. IOLEV  /=  1)  RETURN
    !
    NR=ZERO
    RNORM=FOUR*PI*DENS*NCONF*DR**3
    DO I=1,MGN
       R=I * DR
       GNR = GN(I)
       ! Allow for a thick shell and use the correct integrated
       ! volume 4PI(Ro^3-Ri^3)/3
       GNN = GNR/(RNORM*(I*I+I+THIRD))
       NR=NR+GNR/NCONF
       ! Volume correction applied to denominator. ACCV has been checked > 0
       IF(NPTS > 0) THEN
          ACVTMP=ACCV( (I-1)*NPTS/MGN +1 )
          WRITE(IUNIT,'(2X,5E12.5)') R,GNN/ACVTMP,NR,GNN,ACVTMP
       ELSE
          WRITE(IUNIT,'(2X,5E12.5)') R,GNN,NR
       ENDIF
    ENDDO
    !
    IF(PRNLEV >= 2) WRITE (OUTU,100) NCONF, DR
100 FORMAT(/,5X,'RADIAL DISTRIBUTION ',I8,' CONFIGS DR= ',E15.5,/)
    !
    RETURN
  END SUBROUTINE PRNTG

  SUBROUTINE GTSOLV(ISTP,X,Y,Z,MOX2,OX2,HX1,HX2,NSOLV, &
       MX,REF,SREF,FLGSOL,FLGSIT,SITE,NSITE,MXSITE, &
       NEXV,FLGEXV,SOLUTE,WAT,CROS, &
       LMULTI,QFIRST,NFIRST,NSKIP,NSTEP,NUNIT,FIRSTU, &
       TEMP,FREEAT,ERRORC,VEL &
       ,PQCRYS,PNFREAT,PNSAVC,PNFILE &    
       )
    !-----------------------------------------------------------------------
    !     gets solvent coordinates(velocities) and sets up reference site
    !

  use chm_kinds
  use cheq,only: qcg
  use dimens_fcm
  use number
  use stream
  use cvio
  use ctitla
  use image
  use psf
    implicit none

    !     input
    INTEGER ISTP, MOX2, NSOLV, NSITE, MXSITE
    INTEGER FLGSOL(NATOM),FLGSIT(NATOM), SITE(MXSITE)
    INTEGER NEXV,FLGEXV(NEXV)
    INTEGER NFIRST,NSKIP,NSTEP,NUNIT,FIRSTU
    LOGICAL WAT, CROS, LMULTI,QFIRST,ERRORC,VEL
    LOGICAL PQCRYS
    INTEGER PNFREAT,PNSAVC,PNFILE
    !     output
    real(chm_real) X(NATOM),Y(NATOM),Z(NATOM)
    real(chm_real)    OX2(3,MOX2),HX1(3,MOX2),HX2(3,MOX2), &
         SOLUTE(3,NEXV)
    INTEGER   MX
    real(chm_real)    REF(3), SREF(3,NSITE)
    real(chm_real4) ::     TEMP(NATOM)
    INTEGER   FREEAT(NATOM)
    !     local
    LOGICAL QCRYS
    INTEGER  I, II, III,ICNTRL(20),ISTATS
    INTEGER NSAVV,NFREAT,NDEGF,ISTEP,IUNIT,NFILE
    real(chm_real)    DELTA
    character(len=4) COORHD,VELHD,HDR
    DATA COORHD,VELHD/'CORD','VELD'/
    !lb...add ISTEP to SAVE list (D970710.clb)
    SAVE ISTATS,IUNIT,NFILE,NSAVV,NFREAT,ISTEP
    !mfc..15-FEB-99 fix to c27a1b
    SAVE DELTA
    SAVE QCRYS
    !
    IF(QFIRST)THEN
       IUNIT=FIRSTU
       IF (reallow) THEN
          ISTATS=1
          CALL GTICNT(IUNIT,HDR,ICNTRL,QREWIND=.TRUE.)
          QCRYS=(ICNTRL(11) == 1)
          QFIRST=.FALSE.
       else  
          QCRYS=PQCRYS
          NFREAT=PNFREAT
          NSAVV=PNSAVC
          NFILE=PNFILE
          ISTEP=ISTP
          ISTATS=2
       endif
    ENDIF
    ERRORC=.FALSE.

    if (.not. qfirst) then 
       CALL READCV(X,Y,Z, &
#if KEY_CHEQ==1
            CG,QCG,                                & 
#endif
            TEMP,NATOM,FREEAT,NFREAT, &
            FIRSTU,NUNIT,IUNIT,NFILE, &
            ISTEP,ISTATS,NDEGF,DELTA, &
            NFIRST,NSTEP,NSKIP,NSAVV,COORHD,VELHD, &
            TITLEB,NTITLB,.FALSE., (/ ZERO /), .true.)
    else
       qfirst=.false.
    endif

    IF(ISTEP  /=  ISTP)THEN
       CALL WRNDIE(-2,'<GTSOLV>','READCV returned wrong frame')
       ERRORC=.TRUE.
    ENDIF

    ! Successful read, now manipulate the crystal cell stuff a little bit
    ! ASSUME WE ARE READING TRAJCTORY W/ CRYSTAL AND CHARMM VERSON  >=  22
    ! ln nov-94
    IF(QCRYS)THEN
       CALL XTLLAT(XUCELL,XTLABC)
       ! and make these values be the ones used by solana
       XBOX=XUCELL(1)
       YBOX=XUCELL(2)
       ZBOX=XUCELL(3)
       XBOX2=XBOX/TWO
       YBOX2=YBOX/TWO
       ZBOX2=ZBOX/TWO
    ENDIF
    !
    !     Now set up coordinates/velocities for solvent
    I=0
    II=1
    III=1
    !
    IF (I < NATOM) THEN
70     CONTINUE
       I=I+1
       IF(FLGSOL(I) == 1) THEN
          OX2(1,II)=X(I)
          OX2(2,II)=Y(I)
          OX2(3,II)=Z(I)
          IF(WAT) THEN
             I=I+1
             HX1(1,II)=X(I)
             HX1(2,II)=Y(I)
             HX1(3,II)=Z(I)
             I=I+1
             HX2(1,II)=X(I)
             HX2(2,II)=Y(I)
             HX2(3,II)=Z(I)
          ENDIF
          II=II+1
       ENDIF
       !
       IF (CROS) THEN
          IF (FLGSIT(I) == 1) THEN
             HX1(1,III)=X(I)
             HX1(2,III)=Y(I)
             HX1(3,III)=Z(I)
             III=III+1
          ENDIF
       ENDIF

       !
       IF (I < NATOM) GOTO 70
    ENDIF
    !
    !     Now compute reference position
    IF((NSITE > 0).AND..NOT.CROS) THEN
       IF(.NOT.VEL) THEN
          REF(1)=ZERO
          REF(2)=ZERO
          REF(3)=ZERO
          DO I=1,NSITE
             IF (LMULTI) THEN
                SREF(1,I) = X(SITE(I))
                SREF(2,I) = Y(SITE(I))
                SREF(3,I) = Z(SITE(I))
             ELSE
                REF(1) = REF(1) + X(SITE(I))
                REF(2) = REF(2) + Y(SITE(I))
                REF(3) = REF(3) + Z(SITE(I))
             ENDIF
          ENDDO
          IF(.NOT.LMULTI) THEN
             REF(1) = REF(1)/NSITE
             REF(2) = REF(2)/NSITE
             REF(3) = REF(3)/NSITE
             SREF(1,1) = REF(1)
             SREF(2,1) = REF(2)
             SREF(3,1) = REF(3)
          ENDIF
       ENDIF
    ENDIF
    !
    ! And the solute positions to use for excluded volume corrections
    DO I=1,NEXV
       SOLUTE(1,I)=X(FLGEXV(I))
       SOLUTE(2,I)=Y(FLGEXV(I))
       SOLUTE(3,I)=Z(FLGEXV(I))
    ENDDO
    IF(ISTATS  <  0)THEN
       ! Done reading, clean up
       QFIRST=.TRUE.
       IF(ISTEP  /=  NSTEP) ERRORC=.TRUE.
    ENDIF
    RETURN
  END SUBROUTINE GTSOLV

  SUBROUTINE FBOUND(IUNIT,NCONF,NMOL,O1,H1,H2,SITE,REF,TYP, &
       RSP,RCUT,ZP0,DZP,NZP,XBOX,YBOX,ZBOX)
    !-----------------------------------------------------------------------
    !     This routine computes the forces on a site in water
    !     arising from the interaction due to the water molecules
    !     outside the dynamics sphere but inside the interaction
    !     sphere   This is analgous to the Deformable Boundary
    !     calculation.
    !
    !******** C. L. Brooks III   19-March,1984  Cambridge, MA *********
    !
  use chm_kinds
  use stream
    implicit none
    !     Declarations, Input/Output
    INTEGER NCONF, NMOL, SITE(1), TYP, NZP, IUNIT
    !
    real(chm_real) O1(3,NMOL), H1(3,NMOL), H2(3,NMOL)
    real(chm_real) REF(3)
    real(chm_real) RSP, RCUT, ZP0, DZP
    real(chm_real) XBOX, YBOX, ZBOX
    !
    !
    !     Internal
    real(chm_real) ZPRIME, FB(3), XORI, YORI, ZORI
    real(chm_real) SITST, XIJ, YIJ, ZIJ, RIJ
    real(chm_real) FR, FBT
    !
    INTEGER IMOL, IM, JM, KM, IZP
    !
    !     Main Loop
    NCONF = NCONF + 1
    loop80: DO IZP=1,NZP
       !     Zero force vector for this ZPRIME
       FB(1)=0.0
       FB(2)=0.0
       FB(3)=0.0
       !     ZPRIME is the separation of the reference site from the origin
       ZPRIME=ZP0 + (IZP-1)*DZP
       !     Construction of the dynamics sphere origin (radius of RSP)
       XORI=REF(1)
       YORI=REF(2)
       ZORI=REF(3) - ZPRIME
       !
       loop70: DO IMOL=1,NMOL
          !     Loop over all molecule images
          loop60: DO IM=-1,1
             loop50: DO JM=-1,1
                loop40: DO KM=-1,1
                   !     Compute the distance of this image from the dynamics sphere origin
                   SITST = (O1(1,IMOL) + IM*XBOX - XORI)**2 &
                        + (O1(2,IMOL) + JM*YBOX - YORI)**2 &
                        + (O1(3,IMOL) + KM*ZBOX - ZORI)**2
                   !     Is this molecule in the dynamics sphere?
                   IF (SITST > RSP) THEN
                      !     Compute test site - reference site separation
                      XIJ = (O1(1,IMOL) + IM*XBOX - REF(1))
                      YIJ = (O1(2,IMOL) + JM*YBOX - REF(2))
                      ZIJ = (O1(3,IMOL) + KM*ZBOX - REF(3))
                      RIJ = XIJ**2 + YIJ**2 + ZIJ**2
                      !     Is the test molecule within the interaction sphere of the
                      !     reference site
                      IF (RIJ <= RCUT) THEN
                         !     Now compute test molecule forces on reference site
                         !     First oxygen
                         FR = FSITE(RIJ,TYP)
                         FB(1) = FB(1) - XIJ*FR
                         FB(2) = FB(2) - YIJ*FR
                         FB(3) = FB(3) - ZIJ*FR
                         !     Compute test site - reference site separation
                         XIJ = (H1(1,IMOL) + IM*XBOX - REF(1))
                         YIJ = (H1(2,IMOL) + JM*YBOX - REF(2))
                         ZIJ = (H1(3,IMOL) + KM*ZBOX - REF(3))
                         RIJ = XIJ**2 + YIJ**2 + ZIJ**2
                         !     Next hydrogen 1
                         FR = FSITE(RIJ,TYP+1)
                         FB(1) = FB(1) - XIJ*FR
                         FB(2) = FB(2) - YIJ*FR
                         FB(3) = FB(3) - ZIJ*FR
                         !     Compute test site - reference site separation
                         XIJ = (H2(1,IMOL) + IM*XBOX - REF(1))
                         YIJ = (H2(2,IMOL) + JM*YBOX - REF(2))
                         ZIJ = (H2(3,IMOL) + KM*ZBOX - REF(3))
                         RIJ = XIJ**2 + YIJ**2 + ZIJ**2
                         !     Next hydrogen 2
                         FR = FSITE(RIJ,TYP+1)
                         FB(1) = FB(1) - XIJ*FR
                         FB(2) = FB(2) - YIJ*FR
                         FB(3) = FB(3) - ZIJ*FR
                      ENDIF
                   ENDIF
                enddo loop40
             enddo loop50
          enddo loop60
       enddo loop70
       !     Project out the radial component for this step
       FBT = (REF(1)*FB(1) + REF(2)*FB(2) + REF(3)*FB(3)) &
            / ( SQRT(REF(1)**2 + REF(2)**2 + REF(3)**2) )
       IF(IOLEV >= 1) WRITE(IUNIT,'(F8.3,1X,F8.3)') ZPRIME, FBT
       FRAD(IZP) = FRAD(IZP) + FBT
       F2RAD(IZP) = F2RAD(IZP) + FBT*FBT
    enddo loop80
    !
    RETURN
  END SUBROUTINE FBOUND

  SUBROUTINE PRNFBND(IUNIT,NCONF,ZP0,DZP,NZP,TY)
    !-----------------------------------------------------------------------
  use chm_kinds
  use stream
    implicit none
    real(chm_real) ZP0, DZP, ZPRIME, FFLCT
    INTEGER NCONF, NZP, TY, IZP,IUNIT
    character(len=8) N(2)
    real(chm_real) FRAD, F2RAD
    COMMON /FBND/ FRAD(50), F2RAD(50)
    DATA N /'oxygen  ','hydrogen'/
    !
    IF(IUNIT  <=  0) RETURN
    !
    IF(PRNLEV >= 2) WRITE (OUTU,'(3A)') &
         ' Deformable Boundary Force on ',N(TY), &
         ' simulation and fluctuation in force'
    IF(IOLEV >= 1) WRITE(IUNIT,'(3A)') &
         ' Deformable Boundary Force on ',N(TY), &
         ' simulation and fluctuation in force'
    !
    DO IZP=1,NZP
       ZPRIME = ZP0 + (IZP-1)*DZP
       FRAD(IZP) = FRAD(IZP)/NCONF
       F2RAD(IZP) = F2RAD(IZP)/NCONF
       FFLCT = F2RAD(IZP) - FRAD(IZP)**2
       IF(PRNLEV >= 2) WRITE (OUTU,'(F8.3,1X,F8.3,1X,F8.3)') &
            ZPRIME, FRAD(IZP), FFLCT
       IF(IOLEV >= 1) WRITE(IUNIT,'(F8.3,1X,F8.3,1X,F8.3)') &
            ZPRIME, FRAD(IZP), FFLCT
    ENDDO
    !
    RETURN
  END SUBROUTINE PRNFBND
  !
  FUNCTION FSITE(R2,TYP) result(fsite_1)
    !-----------------------------------------------------------------------
    !     This function routine computes -1/r*du/dr for the
    !     TIP3P potential.
  use chm_kinds
  use consta
    implicit none
    real(chm_real) R2, FTMP, fsite_1
    INTEGER TYP
    real(chm_real) A(5),C(5),Q2(5)
    DATA A/582000.000,4*0.0/
    DATA C/595.0,4*0.0/
    DATA Q2/0.625556,-0.347778,0.173889,0.0,0.0/
    !brb..07-FEB-99 replaced by CCELEC      DATA QTKC/332.17752/
    !
    FTMP = 12.0*A(TYP)/R2**7 - 6.0*C(TYP)/R2**4 &
         + CCELEC*Q2(TYP)/(R2*SQRT(R2))
    FSITE_1 = FTMP
    !
    RETURN
  END FUNCTION FSITE
  !
  SUBROUTINE PRNHIST(IHIST,IPDB,NXP,NYP,NZP,NDI,HIST,NHIST, &
       XMIN,XSTP,XMAX,YMIN,YSTP,YMAX,ZMIN,ZSTP,ZMAX, &
       NFILE,QDIP,RNORM,THRS)
    !
    !  Write the 3D histogram to unit IHIST. Normalization?
    !
  use clcg_mod,only:random
  use chm_kinds
  use number
  use stream
  use memory
    implicit none
    INTEGER IHIST,NXP,NYP,NZP,NDI,NHIST
    INTEGER ICOUNT,ISEED,NFILE,IX,IY,IZ,IPDB
    real(chm_real4) HIST(NDI,NXP,NYP,NZP)
    real(chm_real) XMIN,XSTP,XMAX,YMIN,YSTP,YMAX,ZMIN,ZSTP,ZMAX
    real(chm_real) X,Y,Z,XTOT,YTOT,ZTOT,R
    real(chm_real) BPROB,RPROB,OUTR,INNR,THRS,RNORM,RNF, &
         HIDENS,LODENS,XX
    LOGICAL QDIP
    !
    INTEGER I,J,K,L,ORIGIN(3)
    real(chm_real4),allocatable,dimension(:,:) :: slice
    real(chm_real4) CELL(6)
    !
    IF(IHIST  <=  0 .AND. IPDB .LE. 0) RETURN
    IF(NFILE  <=  0) RETURN
    IF(IOLEV <  1) RETURN
    IF(RNORM <= ZERO) THEN
       RNF=ONE/NFILE
    ELSE
       RNF=ONE/(RNORM*NFILE)
    ENDIF
    IF(IHIST > 0)THEN
       IF(QDIP)THEN
          DO L=1,NZP
             WRITE(IHIST) (((HIST(I,J,K,L)*RNF,I=1,NDI),J=1,NXP),K=1,NYP)
          ENDDO
       ELSE
          ! Output of density in DN6 (OLD)
          ! electron density format
          call chmalloc('solana.src','PRNHIST','SLICE',8,NXP*NYP,cr4=SLICE)
          CELL(1)=XMAX-XMIN
          CELL(2)=YMAX-YMIN
          CELL(3)=ZMAX-ZMIN
          CELL(4)=NINETY
          CELL(5)=NINETY
          CELL(6)=NINETY
          ORIGIN(1)=XMIN/XSTP
          ORIGIN(2)=YMIN/YSTP
          ORIGIN(3)=ZMIN/ZSTP
          CALL DN6MAP(IHIST,HIST,NXP,NYP,NZP,SLICE,CELL,ORIGIN)
          call chmdealloc('solana.src','PRNHIST','SLICE',8,NXP*NYP,cr4=SLICE)
       ENDIF
    ENDIF
    ! Following piece of code is not in use at present, but may be
    ! reinstated. LNI
    !      IF(RNORM < ZERO)THEN
    ! Assume we are working with a sphere centered at the origin
    ! and of radius "OUTR". The grids span the entire sphere. To
    ! calculate the occupation probablilities of the cells lying
    ! within "OUTR" and "INNR" we use a random sampling of the cells.
    !        ICOUNT = 0
    !        BPROB = ZERO
    !        ISEED = 11567
    !        XTOT = XMAX - XMIN
    !        YTOT = YMAX - YMIN
    !        ZTOT = ZMAX - ZMIN
    !        DO I = 1, 5000
    !           X = XTOT*RANDOM(ISEED) + XMIN
    !           Y = YTOT*RANDOM(ISEED) + YMIN
    !           Z = ZTOT*RANDOM(ISEED) + ZMIN
    !           R = SQRT(X*X+Y*Y+Z*Z)
    !           IF(R < OUTR.AND.R > INNR)THEN
    !              IX = INT((X+XMAX)/XSTP) + 1
    !              IY = INT((Y+YMAX)/YSTP) + 1
    !              IZ = INT((Z+ZMAX)/ZSTP) + 1
    !              IF(.NOT.QDIP)THEN
    !                 BPROB = BPROB + HIST(1,IX,IY,IZ)
    !              ELSE
    !              Something ? MAGNITUDE OF DIPOLE MOMENT?....
    !              ENDIF
    !              ICOUNT = ICOUNT + 1
    !           ENDIF
    !         ENDDO
    !         BPROB = BPROB/(ICOUNT*NFILE)
    !         WRITE(OUTU,'(A,F7.5)')'Bulk occupation probability = ',BPROB
    !         RNF=ONE/(BPROB*NFILE)
    !      ENDIF

    ! Write out the relative occupation probabilities greater than
    ! or equal to a threshold value in the form of a coordinate file

    ICOUNT = 0
    HIDENS=MINONE*MEGA
    LODENS=MEGA
    DO K = 1, NZP
       DO J = 1, NYP
          DO I = 1, NXP
             IF(.NOT.QDIP)THEN
                RPROB = HIST(1,I,J,K)*RNF
             ELSE
                RPROB = RNF * SQRT(HIST(1,I,J,K)**2 + &
                     HIST(2,I,J,K)**2 +HIST(3,I,J,K)**2)
             ENDIF
             LODENS=MIN(LODENS,RPROB)
             HIDENS=MAX(HIDENS,RPROB)
             IF(ABS(RPROB) >= THRS .AND. IPDB > 0)THEN
                X = (I-1)*XSTP + XMIN
                Y = (J-1)*YSTP + YMIN
                Z = (K-1)*ZSTP + ZMIN
                ICOUNT = ICOUNT + 1
                IF(.NOT.QDIP)THEN
                   WRITE(IPDB,70)ICOUNT,'HXX','DEN',ICOUNT,X,Y,Z,RPROB
                ELSE
                   WRITE(IPDB,70)ICOUNT,'HYY','DIP',(ICOUNT+1)/2, &
                        X,Y,Z,RPROB
                   ICOUNT=ICOUNT+1
                   X=X-HIST(1,I,J,K)*RNF
                   Y=Y-HIST(2,I,J,K)*RNF
                   Z=Z-HIST(3,I,J,K)*RNF
                   WRITE(IPDB,70)ICOUNT,'HZZ','DIP',(ICOUNT+1)/2, &
                        X,Y,Z,RPROB
70                 FORMAT('HETATM',I5,2X,A3,1X,A3,1X,I5,4X,3F8.3, &
                        6X,F6.2)
                ENDIF
             ENDIF
          ENDDO
       ENDDO
    ENDDO
    IF(QDIP.AND.ICOUNT > 0)THEN
       DO I=1,ICOUNT,2
          ! PDB format requires this double counting of the connectivity
          ! even though most programs seem not to care
          WRITE(IPDB,75) I,I+1,I+1,I
75        FORMAT('CONECT',2I5)
       ENDDO
    ENDIF
    WRITE(IPDB,'(A)') 'END'
    WRITE(OUTU,80) NFILE,ONE/RNF,LODENS,HIDENS
80  FORMAT(//,I10,' frames analyzed. Normalization factor=',G12.5, &
         ' gives:', &
         /' Minimum density=',G12.5,'     maximum density=',G12.5//)
    RETURN
  END SUBROUTINE PRNHIST
  !
  SUBROUTINE LINREG(NP1,X,Y,A,ASD,B,BSD,RXY)
    !
    ! Fit y=A+Bx to NP1 points in X and Y, return also
    ! correlation coefficient RXY and standard deviations of coeffs ASD,BSD
    ! Lennart Nilsson, Oct 96, from Bevington, "Data Reduction" McGraw-Hill
    ! Modified Nov 2003, LNI: NP1<0 means that points are equidistant with spacing x(1),
    ! starting at abscissa zero.

    !
  use chm_kinds
  use number
    implicit none
    INTEGER NP1
    real(chm_real) X(*),Y(*)
    real(chm_real) A,ASD,B,BSD,RXY
    !

    INTEGER I,NP
    real(chm_real) S,D,SX,SX2,SY,SY2,SXY
    !
    NP=ABS(NP1)
    SX=ZERO
    SX2=ZERO
    SY=ZERO
    SY2=ZERO
    SXY=ZERO
    IF(NP1  >  0)THEN
       DO I=1,NP
          SX=SX+X(I)
          SY=SY+Y(I)
          SX2=SX2+X(I)*X(I)
          SY2=SY2+Y(I)*Y(I)
          SXY=SXY+X(I)*Y(I)
       ENDDO
    ELSE
       SX=NP*(NP-1)*X(1)/2
       DO I=1,NP
          SY=SY+Y(I)
          SX2=SX2+X(1)*X(1)*(I-1)*(I-1)
          SY2=SY2+Y(I)*Y(I)
          SXY=SXY+X(1)*(I-1)*Y(I)
       ENDDO
    ENDIF
    ! Currently (almost) no errorchecking
    D=NP*SX2-SX*SX
    IF(NP  >  2 .AND. ABS(D).GT.RSMALL)THEN
       A=(SX2*SY-SX*SXY)/D
       B=(NP*SXY-SX*SY)/D
       S=(SY2+NP*A*A+B*B*SX2-TWO*(A*SY+B*SXY-A*B*SX))/(NP-2)
       ASD=SQRT(S*SX2/D)
       BSD=SQRT(NP*S/D)
       RXY=(NP*SXY-SX*SY)/SQRT(D*(NP*SY2-SY*SY))
    ELSE
       ASD=ZERO
       BSD=ZERO
       RXY=ZERO
       A=ZERO
       B=ZERO
       IF(NP  ==  2)THEN
          IF(NP1  <  0)THEN
             B=(Y(2)-Y(1))/X(1)
             A=Y(1)
          ELSEIF(X(1)  ==  X(2))THEN
             A=ZERO
             B=ZERO
          ELSE
             B=(Y(1)-Y(2))/(X(1)-X(2))
             A=Y(1)-B*X(1)
          ENDIF
       ENDIF
    ENDIF
    RETURN
  END SUBROUTINE LINREG

  SUBROUTINE DN6MAP(IHIST,HIST,NXP,NYP,NZP,SLICE,CELL,ORIGIN)
    !
    ! Writes density data from HIST in DN6 electron
    ! density format (as described by Alwyn Jones and Morten Kjeldgaard)
    ! L.Nilsson, August 1998
    !
  use chm_kinds
  use number
  use stream
    implicit none
    INTEGER IHIST,NXP,NYP,NZP,ORIGIN(3)
    real(chm_real4) CELL(6)
    real(chm_real4) HIST(NXP,NYP,NZP),SLICE(8,NXP*NYP)
    !
    INTEGER I,J,K,L,NGX,NGY,INUM,I1,I2,ICT,JCT
    INTEGER J1,J2,J3,K1,K2,K3
    character(len=512) STR
    INTEGER(chm_int2) :: IREC(256)
    integer(int_byte) :: IREC2(512)
    real(chm_real4) SIGMA,PROD,RHOMAX,RHOMIN,TMP,PLUS
    real(chm_real) SUM,SUM2
    INTEGER FLEN,IERR
    LOGICAL QOPEN,QFORM,QWRITE
    EQUIVALENCE (IREC,IREC2)
    integer my_recl
    !
    IF(IHIST <= 0) RETURN
    ! Find std dev and range of density
    RHOMAX=HIST(1,1,1)
    RHOMIN=HIST(1,1,1)
    SUM=ZERO
    SUM2=SUM
    DO K=1,NZP
       DO J=1,NYP
          DO I=1,NXP
             TMP=HIST(I,J,K)
             RHOMAX=MAX(TMP,RHOMAX)
             RHOMIN=MIN(TMP,RHOMIN)
             SUM=SUM+TMP
             SUM2=SUM2+TMP*TMP
          ENDDO
       ENDDO
    ENDDO
    !
    IF(RHOMAX <= RHOMIN)THEN
       CALL WRNDIE(-1,'<BRIX>','PROBLEM WITH DENSITY SCALING')
       RETURN
    ENDIF
    PROD=255.0/(RHOMAX-RHOMIN)
    PLUS=-RHOMIN*PROD
    SUM=SUM/(NXP*NYP*NZP)
    SIGMA=SQRT(SUM2/(NXP*NYP*NZP) - SUM*SUM)
    !
    ! First let us try to reopen the file with the correct parameters
    ! (direct access 512 byte records - not available as a standard CHARMM
    !  file type)
    !
    CALL VINQRE('UNIT',STR,512,FLEN,QOPEN,QFORM,QWRITE,IHIST)
    ! Don't use the CHARMM VCLOSE here - we will immediately reopen the file
    CLOSE(IHIST)
    ! Here it may be necessary to have different numbers on different machines
    ! We need 512 bytes
#if KEY_GNU==1 || KEY_OSX==1
    my_recl = 512
#else /**/
    my_recl = 128
#endif 
    ! Open a la DN6 (and we will overwrite existing files....)
    OPEN(IHIST,FILE=STR,STATUS='UNKNOWN',ACCESS='DIRECT', &
         FORM='UNFORMATTED', RECL=my_recl)
    !
    ! Header; NB!: the origin should perhaps be something different!!!
    IF(PRNLEV >= 2)THEN
       WRITE(OUTU,905) RHOMIN,RHOMAX
905    FORMAT(' RHOMIN=',G11.4,' RHOMAX=', G11.4)
       WRITE(OUTU,900) ORIGIN,NXP,NYP,NZP,NXP,NYP,NZP,CELL, &
            PROD,INT(PLUS),SIGMA
       !    &               ,CHAR(12)
    ENDIF
900 FORMAT(':-) Origin',3I5,' Extent',3I5,' Grid',3I5, &
         ' Cell ',6F10.3,' Prod',F12.5,' Plus',I8, &
         ' Sigma ',F12.5,1X,1A)
    !      WRITE(IHIST,REC=1) STR
    ! 'OLD' (dn6 standard?) format:
    I1=80
    I2=100
    IREC(1)=ORIGIN(1)
    IREC(2)=ORIGIN(2)
    IREC(3)=ORIGIN(3)
    IREC(4)=NXP
    IREC(5)=NYP
    IREC(6)=NZP
    IREC(7)=NXP
    IREC(8)=NYP
    IREC(9)=NZP
    IREC(10)=I1*CELL(1)
    IREC(11)=I1*CELL(2)
    IREC(12)=I1*CELL(3)
    IREC(13)=I1*CELL(4)
    IREC(14)=I1*CELL(5)
    IREC(15)=I1*CELL(6)
    IREC(16)=I2*PROD
    IREC(17)=PLUS
    IREC(18)=I1
    IREC(19)=I2
    DO I=20,256
       IREC(I)=0
    ENDDO
    WRITE(IHIST,REC=1) IREC
    !
    ! Get number of 8 grid pages in each direction
    NGX=NXP/8
    IF(MOD(NXP,8)  >=  1) NGX=NGX+1
    NGY=NYP/8
    IF(MOD(NYP,8)  >=  1) NGY=NGY+1
    !
    ! Now try to get this out in the 8x8x8 brick style
    !
    INUM=1
    DO I=1,NZP
       I1=MOD(I,8)
       IF(I1 == 0) I1=8
       ICT=0
       ! Fill SLICE array, properly scaled
       DO K=1,NYP
          DO J=1,NXP
             ICT=ICT+1
             SLICE(I1,ICT)=HIST(J,K,I)*PROD+PLUS
          ENDDO
       ENDDO
       IF(I1 == 8.OR.I.EQ.NZP)THEN
          ! Output time. Put together records as needed and write out
          DO J=1,NGY
             J1=(J-1)*8 + 1
             J2=J*8
             DO K=1,NGX
                K1=(K-1)*8+1
                K2=K*8
                ICT=0
                DO L=1,8
                   DO J3=J1,J2
                      JCT=(J3-1)*NXP+K1-1
                      DO K3=K1,K2
                         ICT=ICT+1
                         JCT=JCT+1
                         IF(J3 > NYP.OR.K3.GT.NXP.OR.L.GT.I1)THEN
                            IREC2(ICT)=0
                         ELSE
                            IREC2(ICT)=INT(SLICE(L,JCT))
                         ENDIF
                      ENDDO
                   ENDDO
                ENDDO
                INUM=INUM+1
                WRITE(IHIST,REC=INUM) IREC
                !                  WRITE(IHIST,REC=INUM) STR
             ENDDO
          ENDDO
       ENDIF
    ENDDO
    RETURN
  END SUBROUTINE DN6MAP

  SUBROUTINE MSETINI(QBYGRP,QBYRES,QBYSEG,SOLVSET,SITESET, &
       NSOLV,FLGSOL,NSITE,FLGSIT)
    !
    ! Initialize flag arrays showing which group,residue or segment
    ! the selected solvent/site atoms belong to. If neither option is turned on
    ! then all atoms will have a unique indicator, which is the atom's index
    ! number in the real coordinate arrays so it may be used to track atom
    ! properties also for the subsets collected by GTSOLV
    ! Lennart Nilsson, October 1998
    !
  use chm_kinds
  use consta
  use dimens_fcm
  use number
  use psf
  use chutil,only:getres,getseg
    implicit none
    !
    INTEGER SOLVSET(NATOM),SITESET(NATOM)
    INTEGER NSOLV,NSITE,FLGSOL(natom),FLGSIT(NATOM)
    LOGICAL QBYGRP,QBYRES,QBYSEG,CROS
    !
    INTEGER I,K,KK,IRES
    !
    K=0
    KK=0
    DO I=1,NATOM
       IF(FLGSOL(I) == 1)THEN
          K=K+1
          IRES=GETRES(I,IBASE,NRES)
          IF(QBYGRP)THEN
             SOLVSET(K)=GETRES(I,IGPBS,NGRP)
          ELSEIF(QBYRES)THEN
             SOLVSET(K)=IRES
          ELSEIF(QBYSEG)THEN
             SOLVSET(K)=GETSEG(IRES,NICTOT,NSEG)
          ELSE
             SOLVSET(K)=I
          ENDIF
       ENDIF
       IF(FLGSIT(I) == 1)THEN
          KK=KK+1
          IRES=GETRES(I,IBASE,NRES)
          IF(QBYGRP)THEN
             SITESET(KK)=GETRES(I,IGPBS,NGRP)
          ELSEIF(QBYRES)THEN
             SITESET(KK)=IRES
          ELSEIF(QBYSEG)THEN
             SITESET(KK)=GETSEG(IRES,NICTOT,NSEG)
          ELSE
             SITESET(KK)=I
          ENDIF
       ENDIF
    ENDDO
    RETURN
  END SUBROUTINE MSETINI

  SUBROUTINE ACVOLUM(MODE,ACCV,NPTS,DR,NSOLV,SOLV, &
       NMCPTS,NAT,SOLUTE,RS,RDMAX,ISEED)
    !
    ! For each of the NPTS DR-thick shells centered on each of the NSOLV
    ! solvent molecules calculate th eaccessible volume FRACTION accesible, ie NOT
    ! excluded by the NAT atoms with positions SOLUTE and squared radii RS
    ! which should include the probe radius
    ! MODE='INIT'  Initialize everything. NMCPTS is the total number of Monte
    !              Carlo points to use for the calculation in the whole sphere.
    ! MODE='ACCU'  Accumulate exclude volume data in EXV
    ! MODE='FINI'  Release local space allocations and convert to fractions
    !
    !  Lennart Nilsson, September 1998
    !
  use chm_kinds
  use consta
  use number
  use stream
  use memory
    implicit none
    !
    INTEGER NPTS,NMCPTS,NAT,NSOLV,ISEED
    real(chm_real) ACCV(NPTS),DR,SOLV(3,NSOLV)
    real(chm_real) SOLUTE(3,NAT),RS(NAT),RDMAX
    character(len=4) MODE
    !
    real(chm_real),allocatable,dimension(:,:),save :: probe
    INTEGER,allocatable,dimension(:),save :: IFREE,IEXCL
    integer I,NCONF,NPCHK,NATCHK
    real(chm_real) FPDR3N
    SAVE NCONF,NPCHK,NATCHK
    !
    IF(MODE == 'INIT')THEN
       call chmalloc('solana.src','ACVOLUM','PROBE',4,NMCPTS,crl=PROBE)
       call chmalloc('solana.src','ACVOLUM','IFREE',NPTS,intg=IFREE)
       call chmalloc('solana.src','ACVOLUM','IEXCL',NPTS,intg=IEXCL)
       ACCV(1:npts)=ZERO
       CALL GENPTS(ISEED,NMCPTS,PROBE,NPTS,DR)
       NCONF=0
       NPCHK=0
       NATCHK=0
       IF(PRNLEV > 10)THEN
          CALL PRNPTS(OUTU,NMCPTS,PROBE)
       ENDIF
    ELSEIF(MODE == 'FINI')THEN
       call chmdealloc('solana.src','ACVOLUM','PROBE',4,NMCPTS,crl=PROBE)
       call chmdealloc('solana.src','ACVOLUM','IFREE',NPTS,intg=IFREE)
       call chmdealloc('solana.src','ACVOLUM','IEXCL',NPTS,intg=IEXCL)
       FPDR3N=NCONF*FOUR*PI*DR**3
       IF(PRNLEV >= 2)THEN
          WRITE(OUTU,'(/A,I7,A,F7.1,A)' ) &
               'Accessible volume fractions from', NCONF, &
               ' conformations with',FLOAT(NPCHK)/NCONF,' MC points'
          IF(NPCHK > 0)THEN
             WRITE(OUTU,'(A,F7.1,A)') &
                  'and',FLOAT(NATCHK)/NPCHK,' solute atoms checked on average'
          ELSE
             WRITE(OUTU,'(A)') 'checked on average'
          ENDIF
          WRITE(OUTU,*)
          WRITE(OUTU,*) '    SHELL   ACC.VOL.FRAC.'
       ENDIF
       DO I=1,NPTS
          ACCV(I)=ACCV(I)/(FPDR3N*(I*I+I+THIRD))
          IF(PRNLEV >= 2) WRITE(OUTU,'(2F10.3)') I*DR,ACCV(I)
          ! ACCV appears in denominator of g(r)
          IF(ACCV(I) <= RSMALL) ACCV(I)=1.0
       ENDDO
    ELSEIF(MODE == 'ACCU')THEN
       CALL ACVOL1(ACCV,NPTS,DR,NSOLV,SOLV,NMCPTS,PROBE, &
            IFREE,IEXCL,NAT,SOLUTE, &
            RS,RDMAX,NCONF,NPCHK,NATCHK)
    ELSE
       CALL WRNDIE(-2,'<ACVOLM>','Unknown mode: '//MODE)
    ENDIF
    RETURN
  END SUBROUTINE ACVOLUM
  !
  SUBROUTINE ACVOL1(ACCV,NPTS,DR,NSOLV,SOLV,NMCPTS,PROBE, &
       IFREE,IEXCL,NAT,SOLUTE, &
       RS,RDMAX,NCONF,NPCHK,NATCHK)
  use chm_kinds
  use consta
  use number
    implicit none
    !
    ! Does the actual calculation
    !
    INTEGER NPTS,NMCPTS,NAT,NSOLV,IFREE(NPTS),IEXCL(NPTS)
    INTEGER NPCHK,NATCHK,NCONF
    real(chm_real) DR,ACCV(NPTS),PROBE(4,NMCPTS),SOLV(3,NSOLV)
    real(chm_real) SOLUTE(3,NAT),RS(NAT),RDMAX
    !

    INTEGER I,J,K,IRAD
    real(chm_real) XMIN,XMAX,YMIN,YMAX,ZMIN,ZMAX,RMIN
    real(chm_real) XP,YP,ZP,XI,YI,ZI,FPDR3
    LOGICAL QFREE
    !
    FPDR3=FOUR*PI*DR**3
    !
    ! Zero counters, find extent of solute (including maximal atom size)
    !
    XMIN=SOLUTE(1,1)
    XMAX=SOLUTE(1,1)
    YMIN=SOLUTE(2,1)
    YMAX=SOLUTE(2,1)
    ZMIN=SOLUTE(3,1)
    ZMAX=SOLUTE(3,1)
    DO I=2,NAT
       XMIN=MIN(XMIN,SOLUTE(1,I))
       XMAX=MAX(XMAX,SOLUTE(1,I))
       YMIN=MIN(YMIN,SOLUTE(2,I))
       YMAX=MAX(YMAX,SOLUTE(2,I))
       ZMIN=MIN(ZMIN,SOLUTE(3,I))
       ZMAX=MAX(ZMAX,SOLUTE(3,I))
    ENDDO
    XMIN=XMIN-RDMAX
    XMAX=XMAX+RDMAX
    YMIN=YMIN-RDMAX
    YMAX=YMAX+RDMAX
    ZMIN=ZMIN-RDMAX
    ZMAX=ZMAX+RDMAX
    DO I=1,NPTS
       IFREE(I)=0
       IEXCL(I)=0
    ENDDO
    !
    ! Main loop over solvent atoms
    DO I=1,NSOLV
       NCONF=NCONF+1
       XI=SOLV(1,I)
       YI=SOLV(2,I)
       ZI=SOLV(3,I)
       !
       ! Loop over Monte Carlo points
       DO J=1,NMCPTS
          QFREE=.TRUE.
          XP=PROBE(1,J)+XI
          YP=PROBE(2,J)+YI
          ZP=PROBE(3,J)+ZI
          !
          ! Direct exclusion in x-,y-,or z-direction?
          !
          IF    (XP < XMIN)THEN
          ELSEIF(XP > XMAX)THEN
          ELSEIF(YP < YMIN)THEN
          ELSEIF(YP > YMAX)THEN
          ELSEIF(ZP < ZMIN)THEN
          ELSEIF(ZP > ZMAX)THEN
          ELSE
             !
             ! OK, so we really have to do check each solute atom:
             NPCHK=NPCHK+1
             ! Set NATCHK here in case we fall thru the loop
             NATCHK=NATCHK+NAT
             DO K=1,NAT
                RMIN=(XP-SOLUTE(1,K))**2+(YP-SOLUTE(2,K))**2 &
                     +(ZP-SOLUTE(3,K))**2
                IF(RMIN <= RS(K))THEN
                   NATCHK=NATCHK+K-NAT
                   QFREE=.FALSE.
                   GOTO 99
                ENDIF
             ENDDO
99           CONTINUE
          ENDIF
          IRAD=PROBE(4,J)/DR+1
          IF(IRAD < 1 .OR. IRAD > NPTS)THEN
             CALL WRNDIE(-2,'<ACVOLM>','Impossible MC point')
          ELSE
             IF(QFREE)THEN
                IFREE(IRAD)=IFREE(IRAD)+1
             ELSE
                IEXCL(IRAD)=IEXCL(IRAD)+1
             ENDIF
          ENDIF
       ENDDO
       ! Sum up for this solvent molecule, using finite shell volume
       DO J=1,NPTS
          ACCV(J)=ACCV(J)+ &
               FPDR3*(J*J+J+THIRD)*IFREE(J)/(IEXCL(J)+IFREE(J))
       ENDDO
    ENDDO
    RETURN
  END SUBROUTINE ACVOL1

  SUBROUTINE GENPTS(ISEED,NMC,PROBE,NPTS,DR)
  use clcg_mod,only:random
  use chm_kinds
    !
  use consta
  use number
    implicit none
    !
    ! Generate NMC random points uniformly distributed in a sphere containing
    ! NPTS shells of thickness DR; however we only
    ! need to fill shells  from RDMAX and out to NPTS*DR
    !
    !
    INTEGER ISEED,NMC,NPTS
    real(chm_real) PROBE(4,NMC),DR,RDMAX

    !
    INTEGER I,J,NBIN
    real(chm_real) R1,R,THETA,PHI
    !
    NBIN=FLOAT(NMC)/NPTS**3
    R1=ZERO
    J=1
    DO I=1,NMC
       R=R1+DR*RANDOM(ISEED)
       THETA=RANDOM(ISEED)*PI
       PHI=RANDOM(ISEED)*TWOPI
       PROBE(1,I)=R*COS(PHI)*SIN(THETA)
       PROBE(2,I)=R*SIN(PHI)*SIN(THETA)
       PROBE(3,I)=R*COS(THETA)
       PROBE(4,I)=R
       ! To have the points uniformly distributed per volume in each shell:
       ! (and at least one point in each shell)
       IF(I >= NBIN)THEN
          R1=R1+DR
          J=J+1
          NBIN=NMC*(FLOAT(J)/NPTS)**3
       ENDIF
    ENDDO
    RETURN
  END SUBROUTINE GENPTS

  SUBROUTINE RADSET(NAT,FLGEXV,RPROBE,RS,RDMAX,VDWR,IAC,ITC,LWEIG)
    !
    ! Get vdw radii of the NAT atoms indexed by FLGEXV, add the probe
    ! radius and square
    !
  use chm_kinds
  use dimens_fcm
  use number
  use coord
    implicit none
    INTEGER NAT,IAC(*),ITC(*),FLGEXV(*)
    real(chm_real) RS(NAT),VDWR(*),RPROBE,RDMAX
    LOGICAL LWEIG
    !
    INTEGER I
    RDMAX=ZERO
    IF(LWEIG)THEN
       ! Use WMAIN 
       DO I=1,NAT
          RS(I)=(WMAIN(FLGEXV(I))+RPROBE)**2
          RDMAX=MAX(RDMAX,RS(I))
       ENDDO
    ELSE
       ! Use CHM vdW radii
       DO I=1,NAT
          RS(I)=(VDWR(ITC(IAC(FLGEXV(I))))+RPROBE)**2
          RDMAX=MAX(RDMAX,RS(I))
       ENDDO
    ENDIF
    RDMAX=SQRT(RDMAX)
    RETURN
  END SUBROUTINE RADSET

  SUBROUTINE PRNPTS(OUTU,NMCPTS,PROBE)
    !
    ! Print out the Monte Carlo points (mainly for debugging purposes)
    !
  use chm_kinds
    implicit none
    INTEGER OUTU,NMCPTS
    real(chm_real) PROBE(4,NMCPTS)
    !
    INTEGER I,J
    !
    IF(OUTU <= 0) RETURN
    WRITE(OUTU,'(A)') &
         '   Pt #         X         Y         Z         R', &
         '-----------------------------------------------'
    DO I=1,NMCPTS
       WRITE(OUTU,'(I7,4F10.3)') I,(PROBE(J,I),J=1,4)
    ENDDO
    RETURN
  END SUBROUTINE PRNPTS
  !SJS
  SUBROUTINE ROTVEC(NSOLV,ISTP,OX2,HX1,HX2,MOX2,VHHX,VHHY,VHHZ, &
       VOHX,VOHY,VOHZ,MAXTOT,NWAT)
  use chm_kinds
  use number
    !
    implicit none
    INTEGER I,NSOLV,ISTP,J
    INTEGER MAXTOT,NWAT,MOX2
    real(chm_real)    OX2(3,MOX2),HX1(3,MOX2),HX2(3,MOX2)
    real(chm_real) VHHX(MAXTOT,NWAT),VHHY(MAXTOT,NWAT), &
         VHHZ(MAXTOT,NWAT)
    real(chm_real) VOHX(MAXTOT,NWAT),VOHY(MAXTOT,NWAT), &
         VOHZ(MAXTOT,NWAT)
    DO I=1,NSOLV-2,3
       J=(I+2)/3
       VHHX(ISTP,J)=ZERO
       VHHY(ISTP,J)=ZERO
       VHHZ(ISTP,J)=ZERO
       VOHX(ISTP,J)=ZERO
       VOHY(ISTP,J)=ZERO
       VOHZ(ISTP,J)=ZERO
    ENDDO
    DO I=1,NSOLV-2,3
       J=(I+2)/3
       VOHX(ISTP,J) = ((OX2(1,I)-OX2(1,I+1))+(OX2(1,I)-OX2(1,I+2)))/TWO
       VOHY(ISTP,J) = ((OX2(2,I)-OX2(2,I+1))+(OX2(2,I)-OX2(2,I+2)))/TWO
       VOHZ(ISTP,J) = ((OX2(3,I)-OX2(3,I+1))+(OX2(3,I)-OX2(3,I+2)))/TWO
       !       VHHX(ISTP,J) = OX2(1,I+1)-OX2(1,I+2)
       !       VHHY(ISTP,J) = OX2(2,I+1)-OX2(2,I+2)
       !       VHHZ(ISTP,J) = OX2(3,I+1)-OX2(3,I+2)
       VHHX(ISTP,J) = (OX2(1,I+1)-OX2(1,I+2))/TWO
       VHHY(ISTP,J) = (OX2(2,I+1)-OX2(2,I+2))/TWO
       VHHZ(ISTP,J) = (OX2(3,I+1)-OX2(3,I+2))/TWO
    ENDDO
10  FORMAT(5F8.4)
    RETURN
  END SUBROUTINE ROTVEC
  !SJS
  !SJS
  SUBROUTINE ROTCOR(MAXTOT,NWAT,VHHX,VHHY,VHHZ,VOHX,VOHY,VOHZ, &
       NFRAMES,NTOT,TLOW,TUP,IROTCO,ROUT, &
       tqx,tq1x,tq2x,tcor,tcor1,tcor2,DT, &
       atcor,atcor1,atcor2)
    !---------------------------------------------------------------------
  use chm_kinds
  use dimens_fcm
  use consta
  use number
  use comand
  use stream
  use memory

    implicit none
    INTEGER IROTCO,JSOL,NWAT,ROUT
    INTEGER NTOT,NPRI2,IPN
    real(chm_real),allocatable,dimension(:),target :: space_rl
    real(chm_real),pointer,dimension(:) :: SFA,SFB,SFC,SFD,SFAH,SFBH
    integer :: MAXTOT
    INTEGER CRSCOD,VECCOD,LFA
    INTEGER I,NFRAMES,ISTP,IJ,J,K,ILOW,IUP
    real(chm_real) TQX(maxtot,3),TQ1X(maxtot,3),TQ2X(maxtot,3)
    real(chm_real) VHHX(NFRAMES,NWAT),VHHY(NFRAMES,NWAT), &
         VHHZ(NFRAMES,NWAT)
    real(chm_real) VOHX(NFRAMES,NWAT),VOHY(NFRAMES,NWAT), &
         VOHZ(NFRAMES,NWAT)
    LOGICAL QFFT,QLTC,QNNORM,QDIFF
    real(chm_real) TCOR(nframes),TCOR1(nframes),TCOR2(nframes)
    real(chm_real) ATCOR(nframes),ATCOR1(nframes),ATCOR2(nframes)
    real(chm_real) TAU1,TAU2,TAU3
    real(chm_real) STAU1,STAU2,STAU3,ATAU1,ATAU2,ATAU3
    real(chm_real) SD1,SD2,SD3
    real(chm_real) S2T1,S2T2,S2T3
    real(chm_real) TLOW,TUP,DT
    !
    !  LOOP OVER WATER MOLECULES
    !

    ATCOR (1:ntot)=ZERO
    ATCOR1(1:ntot)=ZERO
    ATCOR2(1:ntot)=ZERO

    DO JSOL=1,NWAT
       !
       DO ISTP=1,NFRAMES
          TQX(ISTP,1)=VHHX(ISTP,JSOL)
          TQX(ISTP,2)=VHHY(ISTP,JSOL)
          TQX(ISTP,3)=VHHZ(ISTP,JSOL)
          TQ1X(ISTP,1)=VOHX(ISTP,JSOL)
          TQ1X(ISTP,2)=VOHY(ISTP,JSOL)
          TQ1X(ISTP,3)=VOHZ(ISTP,JSOL)
       ENDDO
       CALL CROSSP(TQX,TQ1X,TQ2X,NFRAMES)
       CALL TQNORM(TQX,NFRAMES)
       CALL TQNORM(TQ1X,NFRAMES)
       CALL TQNORM(TQ2X,NFRAMES)
       !
       NPRI2=NTOT*4
       IPN=IRLP
       CRSCOD=1
       VECCOD=3
       !sjs
       ! use direct method to calculate corr coef
       !
       qfft=.false.
       qltc=.false.
       qdiff=.false.
       qnnorm=.true.
       !sjs
       LFA=NPRI2
       call chmalloc('solana.src','rotcor','space_rl',6*npri2,crl=space_rl)
       SFA => space_rl(1:npri2)
       SFB => space_rl(npri2+1 : 2*npri2)
       SFC => space_rl(2*npri2+1 : 3*npri2)
       SFD => space_rl(3*npri2+1 : 4*npri2)
       SFAH => space_rl(4*npri2+1 : 5*npri2)
       SFBH => space_rl(5*npri2+1 : 6*npri2)

       DO I=1,NTOT
          TCOR(I)=ZERO
          TCOR1(I)=ZERO
          TCOR2(I)=ZERO
       ENDDO
       !
       !mfc This is a stop-gap measure, proper IF logic needs to be
       !      worked out for the rotcor code.
       CALL CORFUN(TQX,TQX,MAXTOT, &
            SFA,SFB,SFC,SFD, &
            SFAH,SFBH,IPN, &
            CRSCOD,VECCOD,QFFT,QLTC,MAXTOT, &
            NTOT,QNNORM,TCOR,NPRI2,QDIFF,ZERO)
       !
       CALL CORFUN(TQ1X,TQ1X,MAXTOT, &
            SFA,SFB,SFC,SFD, &
            SFAH,SFBH,IPN, &
            CRSCOD,VECCOD,QFFT,QLTC,MAXTOT, &
            NTOT,QNNORM,TCOR1,NPRI2,QDIFF,ZERO)
       !
       CALL CORFUN(TQ2X,TQ2X,MAXTOT, &
            SFA,SFB,SFC,SFD, &
            SFAH,SFBH,IPN, &
            CRSCOD,VECCOD,QFFT,QLTC,MAXTOT, &
            NTOT,QNNORM,TCOR2,NPRI2,QDIFF,ZERO)
       call chmdealloc('solana.src','rotcor','space_rl',6*npri2,crl=space_rl)

       ATCOR(1:ntot)=ATCOR(1:ntot)+TCOR(1:ntot)
       ATCOR1(1:ntot)=ATCOR1(1:ntot)+TCOR1(1:ntot)
       ATCOR2(1:ntot)=ATCOR2(1:ntot)+TCOR2(1:ntot)
       !
       !  END OF DO LOOP OVER WATERS
       !
    ENDDO
    !
    !  CALCULATE AVERAGE 
    !
    ATCOR(1:ntot)=ATCOR(1:ntot)/NWAT
    ATCOR1(1:ntot)=ATCOR1(1:ntot)/NWAT
    ATCOR2(1:ntot)=ATCOR2(1:ntot)/NWAT

    IF(ROUT > 0.AND.PRNLEV >= 2 )THEN
       DO I=1,NTOT
          WRITE(ROUT,200)(I-1)*DT,ATCOR(I),ATCOR1(I),ATCOR2(I)
       ENDDO
    ENDIF
    !
    ! NOW CALCULATE CORR TIME from integral of correlation function/LNI
    !
    ILOW=MAX(1,INT(TLOW))
    IUP=MIN(NTOT,INT(TUP))
!    TAU1=0.0
!    TAU2=0.0
!    TAU3=0.0
    !
!    DO I=ILOW+1,IUP-1
!       TAU1=TAU1+ATCOR(I)
!       TAU2=TAU2+ATCOR1(I)
!       TAU3=TAU3+ATCOR2(I)
!    ENDDO
!    !
!    ! Simple normalization
!    TAU1=DT*(TAU1+.5*(ATCOR(ILOW)+ATCOR(IUP)))/ATCOR(1)
!    TAU2=DT*(TAU2+.5*(ATCOR1(ILOW)+ATCOR1(IUP)))/ATCOR1(1)
!    TAU3=DT*(TAU3+.5*(ATCOR2(ILOW)+ATCOR2(IUP)))/ATCOR2(1)
    CALL CORTIM(ATCOR,TAU1,iLOW,iUP,NTOT,DT)
    CALL CORTIM(ATCOR1,TAU2,iLOW,iUP,NTOT,DT)
    CALL CORTIM(ATCOR2,TAU3,iLOW,iUP,NTOT,DT)
    if (prnlev >= 2) WRITE(OUTU,111)'SOLA_RCOR>','lpr=',IPN,TAU1,TAU2,TAU3, &
         'Estimate (ps) assumes normalized C(t) decays to zero.'
111 FORMAT(1x,A10,20X,A4,1x,I1,5x,3(F10.3)/,A)
    !
200 FORMAT(3X,4F9.4)
    RETURN
  END SUBROUTINE ROTCOR
  !SJS
  SUBROUTINE CORTIM(AY,TAU,iTLOW,iTUP,NTOT,DT)
    !-----------------------------------------------------------------------
    use chm_kinds
    use number
    implicit none
    INTEGER NT
    INTEGER I,J
    INTEGER NTOT
    real(chm_real) AY(NTOT)
    real(chm_real) TLOW,TUP,DT
    real(chm_real) X,Y,SUMX,SUMY,SUMXY,SUMX2,SXX,SXY
    real(chm_real) SLOPE
    real(chm_real) TAU
    real(chm_real) XCUT
    integer itlow,itup       ! mc050702
    
    NT = 0
    SUMX = ZERO
    SUMY = ZERO
    SUMXY = ZERO
    SUMX2 = ZERO

! CHECK FOR NEGATIVE Y VALUES AND SET TO 0.0

    DO i =iTLOW+1,iTUP+1,1   ! mc050702
       ! I=NINT(T)           ! mc050702
       Y = AY(I)
       X = (I-1)*DT
       IF (Y <= ZERO) Y=TENM5
       Y = LOG(Y)
       NT = NT + 1
       SUMX = SUMX + X
       SUMY = SUMY + Y
       SUMXY = SUMXY + X*Y
       SUMX2 = SUMX2 + X*X
    ENDDO

    SXX = SUMX2
    SXY = SUMXY
    IF(SXX /= ZERO) THEN
       SLOPE = SXY/SXX
    ELSE
       SLOPE=ZERO
    ENDIF
!  CORRELATION COEFFICIENT - R
!    A = (NT * SUMXY - SUMX * SUMY)
!    C = SQRT ((NT*SUMX2 - SUMX*SUMX)*(NT*SUMY2 - SUMY*SUMY))
!    R = A/C
    IF(SLOPE /= ZERO)THEN
       TAU = MINONE/SLOPE
    ELSE
       TAU=ZERO
    ENDIF

    RETURN
  END SUBROUTINE CORTIM
  !SJS
  !SJS
  SUBROUTINE TQNORM(AX,NTOT)
  use chm_kinds
  use number
    implicit none
    INTEGER NTOT,J,I
    real(chm_real) AX(NTOT,3),QAVE
    DO I=1,NTOT
       QAVE=ZERO
       DO J=1,3
          QAVE=QAVE+AX(I,J)*AX(I,J)
       ENDDO
       IF(QAVE > ZERO)THEN
          QAVE=SQRT(QAVE)
          DO J=1,3
             AX(I,J)=AX(I,J)/QAVE
          ENDDO
       ENDIF
    ENDDO
    RETURN
  END SUBROUTINE TQNORM
  !SJS
  !SJS
  SUBROUTINE CROSSP(TQ,TQ1,TQ2,NTOT)
  use chm_kinds
    implicit none
    INTEGER I,NTOT
    real(chm_real) A,B,TQ(NTOT,3),TQ1(NTOT,3),TQ2(NTOT,3)
    DO I=1,NTOT
       TQ2(I,1)=TQ(I,2)*TQ1(I,3)-TQ(I,3)*TQ1(I,2)
       TQ2(I,2)=TQ(I,3)*TQ1(I,1)-TQ(I,1)*TQ1(I,3)
       TQ2(I,3)=TQ(I,1)*TQ1(I,2)-TQ(I,2)*TQ1(I,1)
    ENDDO
    RETURN
  end SUBROUTINE CROSSP
#endif 

  SUBROUTINE NULL_SA
    RETURN
  END SUBROUTINE NULL_SA

end module solana_mod

