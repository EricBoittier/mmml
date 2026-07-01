module emapmod
  use chm_kinds
  use,intrinsic :: iso_c_binding
  use stream, only:prnlev,outu
  implicit none

#if KEY_EMAP==1 /*emapmod*/
  !-----------------------------------------------------------------------
  !     Data structures for map objects and rigid domains
  !            by Xiongwu Wu, Nov. 2001
  !
  !  EMAP module parameters
  !
  !     EMRCUT    -- Cutoff densities
  !     EMRESO    -- Resolution of map objects
  !     EMICORE     -- Core type. 1--density, 2--Laplacian
  !     MXNCOMP   -- Maximum number of components
  !     NEMCOMP   -- Number of components
  !     IDCOMPS    -- Array of component rigid domain ID   
  !     PBCX,PBCY,PBCZ   -- Rectangular PBC sizes in x, y, z directions   
  !     EMSET      -- Flag to check parameter initialization     
  !     LCOMPFIX   -- Flag array of component rigid domain ID   
  !     LEMAPENG   -- Flag to calculate map constraints   
  !     LEMAPLD    -- Flag to perform rigid domain LD motion   
  !     LPBC       -- Flag to apply periodic condition   
  !
  INTEGER,PARAMETER :: MXNCOMP=10000
  real(chm_real)   EMRESO,EMDX,EMDY,EMDZ,EMAX,EMAY,EMAZ
  real(chm_real)   EMRCUT,ACORE,BCORE,CCORE
  real(chm_real)   PBCX,PBCY,PBCZ
  LOGICAL  EMSET,LCOMPFIX(MXNCOMP),LEMAPENG,LEMAPLD,LPBC
  INTEGER  EMICORE,NEMCOMP,IDCOMPS(MXNCOMP)

  !  CONSTANTS for MAP objects
  !     MXNEMP   - Maximum Number of map objects
  !
  !     NEMAP   - Number of map objects
  !  Arrays for MAP objects:  Properties of map i is stored at ARRAY(I)
  !     EMAPID(MXNEMP)   - Names of each emap
  !
  !     MEMAPX(MXNEMP)   - Starting grid on X-axis
  !     MEMAPY(MXNEMP)   - Starting grid on Y-axis
  !     MEMAPZ(MXNEMP)   - Starting grid on Z-axis
  !     LEMAPX(MXNEMP)   - Number of grids in X-axis direction
  !     LEMAPY(MXNEMP)   - Number of grids in Y-axis direction
  !     LEMAPZ(MXNEMP)   - Number of grids in Z-axis direction
  !     DEMAPX(MXNEMP)   - Grid size in X-axis direction
  !     DEMAPY(MXNEMP)   - Grid size in Y-axis direction
  !     DEMAPZ(MXNEMP)   - Grid size in Z-axis direction
  !     AEMAPX(MXNEMP)   - Reduce parameter in X-axis direction
  !     AEMAPY(MXNEMP)   - Reduce parameter in Y-axis direction
  !     AEMAPZ(MXNEMP)   - Reduce parameter in Z-axis direction
  !     CEMAPX(MXNEMP)   - Rotation CENTER of emap in X-axis direction
  !     CEMAPY(MXNEMP)   - Rotation CENTER of emap in Y-axis direction
  !     CEMAPZ(MXNEMP)   - Rotation CENTER of emap in Z-axis direction
  !     MNCOREX(MXNEMP)   - Starting core grid on X-axis
  !     MNCOREY(MXNEMP)   - Starting core grid on Y-axis
  !     MNCOREZ(MXNEMP)   - Starting core grid on Z-axis
  !     MXCOREX(MXNEMP)   - ENDing core grid on X-axis
  !     MXCOREY(MXNEMP)   - Ending core grid on Y-axis
  !     MXCOREZ(MXNEMP)   - Ending core grid on Z-axis
  !     RMXEMAP(MXNEMP)  - Maximum density
  !     RMNEMAP(MXNEMP)  - Minimum density
  !     AREMAP(MXNEMP)   - SUM of RHO
  !     RREMAP(MXNEMP)   - SUM of RHO*RHO
  !     ADEMAP(MXNEMP)   - SUM of DRHO
  !     DDEMAP(MXNEMP)   - SUM of DRHO*DRHO
  !     ACEMAP(MXNEMP)   - SUM of CORE
  !     CCEMAP(MXNEMP)   - SUM of CORE*CORE
  !     RCEMAP(MXNEMP)   - SUM of RHO*CORE
  !     DCEMAP(MXNEMP)   - SUM of DRHO*CORE
  !     EMAPGUID(MXNEMP) - Constraint constant of EMAP structures
  !     EMMASS(MXNEMP) - Masses of EMAP structures
  !     EMINER(MXNEMP) - Inertial of EMAP structures around COM
  !---mfc   !     RHOEMAP(MXNEMP)    - the density of each emap 
  !---mfc   !     DDREMAP(MXNEMP)    - the Laplacian density of each emap 
  !---mfc   !     COREEMAP(MXNEMP)   - the core label of each emap 
  !---mfc   !     XATEMAP(MXNEMP)    - X coordinates of structure atoms 
  !---mfc   !     YATEMAP(MXNEMP)    - Y coordinates of structure atoms 
  !---mfc   !     ZATEMAP(MXNEMP)    - Z coordinates of structure atoms 
  !---mfc   !     NCREMAP(MXNEMP)    - Number of CORE grids for each emap 
  !---mfc these are replaced by array of derived types empgrd and empcrd
  !
  !     NATEMAP(MXNEMP)    - Number of structure atoms for each emap 
  !
  !     RESOEMAP(MXNEMP)    - Resolution of the emaps 
  !
  integer,parameter :: MXNEMP=1000
  INTEGER NEMAP  
  character(len=80) EMAPID(MXNEMP)
  INTEGER MEMAPX(MXNEMP),MEMAPY(MXNEMP),MEMAPZ(MXNEMP)
  INTEGER MNCOREX(MXNEMP),MNCOREY(MXNEMP),MNCOREZ(MXNEMP)
  INTEGER MXCOREX(MXNEMP),MXCOREY(MXNEMP),MXCOREZ(MXNEMP)
  INTEGER LEMAPX(MXNEMP),LEMAPY(MXNEMP),LEMAPZ(MXNEMP)
  real(chm_real)  DEMAPX(MXNEMP),DEMAPY(MXNEMP),DEMAPZ(MXNEMP)
  real(chm_real)  AEMAPX(MXNEMP),AEMAPY(MXNEMP),AEMAPZ(MXNEMP)
  real(chm_real)  CEMAPX(MXNEMP),CEMAPY(MXNEMP),CEMAPZ(MXNEMP)
  real(chm_real)  RESOEMAP(MXNEMP),RMXEMAP(MXNEMP),RMNEMAP(MXNEMP)
  INTEGER NCREMAP(MXNEMP)
  real(chm_real)  AREMAP(MXNEMP),RREMAP(MXNEMP)
  real(chm_real)  ADEMAP(MXNEMP),DDEMAP(MXNEMP)
  real(chm_real)  ACEMAP(MXNEMP),CCEMAP(MXNEMP)
  real(chm_real)  RCEMAP(MXNEMP),DCEMAP(MXNEMP)
  real(chm_real)  EMAPGUID(MXNEMP)
  real(chm_real)  EMMASS(MXNEMP),EMINER(MXNEMP)
  real(chm_real)  TRNEMAP(3,MXNEMP),ROTEMAP(3,MXNEMP)
  real(chm_real)  LOWEMAP(3,MXNEMP),UPEMAP(3,MXNEMP)
  !  INTEGER RHOEMAP(MXNEMP),DDREMAP(MXNEMP),COREEMAP(MXNEMP)
  type emapgrid
     real(chm_real4),pointer,dimension(:) :: rho,ddrho
     integer,pointer,dimension(:) :: core
  end type emapgrid
  type emapcoord
     integer,pointer,dimension(:) :: iatom
     real(chm_real),pointer,dimension(:) :: x,y,z
  end type emapcoord
  type(emapgrid),dimension(mxnemp) :: empgrd
  type(emapcoord),dimension(mxnemp) :: empcrd

  type, bind(c) :: emap_geom
     integer(c_int32_t) :: nsx, nsy, nsz
     integer(c_int32_t) :: lx, ly, lz
     real(c_double) :: dx, dy, dz
  end type emap_geom

  INTEGER NATEMAP(MXNEMP)
  !  INTEGER XATEMAP(MXNEMP),YATEMAP(MXNEMP),ZATEMAP(MXNEMP)

  !  Variables for RIGID bodies
  !     MXNRIG   - Maximum Number of rigid bodies
  !
  !     EMRIGID(MXNEMP)   - Names of each rigid bodies
  !
  !     MXNRIG   - Maximum Number of rigid structures
  !     NEMRIG  - Number of rigid structures in the emap docking operation
  !     EMRIGID(MXNRIG) - Nmae of rigid structures
  !     IDEMRIG(MXNRIG) - EMAP indeces of rigid structures
  !     TEMRIG(3,MXNRIG) - Translation vectors of rigid structures
  !     REMRIG(9,MXNRIG) - Rotation matrix of rigid structures
  !     TEMRIG0(3,MXNRIG) - Storage for Translation vectors of rigid structures
  !     REMRIG0(9,MXNRIG) - Storage for Rotation matrix of rigid structures
  !     FRCRIG(3,MXNRIG) - Forces of rigid structures
  !     TRQRIG(3,MXNRIG) - Torques of rigid structures
  integer,parameter :: MXNRIG=10000
  INTEGER NEMRIG,IDEMRIG(MXNRIG),NRIGREST,IDRIGREST(100)
  character(len=80) EMRIGID(MXNRIG)
  real(chm_real)  TEMRIG0(3,MXNRIG),REMRIG0(9,MXNRIG)
  real(chm_real)  TEMRIG(3,MXNRIG),REMRIG(9,MXNRIG)
  real(chm_real)  FRCRIG(3,MXNRIG),TRQRIG(3,MXNRIG)

  !  Variables for binding database
  !     MDBRES   - Maximum Number of residue types
  !     NDBRES   - Number of residue types
  !     REMBIND  - Binding distance range
  !     DBRESN  -  Residue names
  !     PMATRIX  - Binding matrix
  !
  INTEGER  NDBRES
  integer,parameter :: MDBRES=30
  real(chm_real)   REMBIND,EMGUID
  real(chm_real)   PMATRIX(MDBRES,MDBRES)
  INTEGER   DBRESNO(MDBRES)
  CHARACTER(len=4) ::   DBRESN1(MDBRES),DBRESN3(MDBRES)

  !   Energies

  !     IEMPTRJ   - Frequency for rigid domain trajectory output
  !     UEMPTRJ   - UNIT for rigid domain trajectory output
  !
  !     EMAPEPC   - core interaction parameter
  !     EMAPEPS   - DeSolvation interaction parameter
  !     DDRAVG    - Average mass density
  real(chm_real)  EMAPEPC,EMAPEPS,EMAPEPE,EMDIELC,EMAPENG,EMAPENGM
  INTEGER IEMPTRJ,UEMPTRJ,NTEMAP,NREMAP

  !   Movement
  !     EMGAMMA  - Collision frequency for map objects
  !     EMTEMP   - Temperature of map objects
  real(chm_real)  EMGAMMA,EMTEMP

  !---------------------------------------------------------------------
contains

  subroutine emap_iniall()
  use number
    emset=.false.
    lemapld=.false.
    ndbres=0
    nrigrest=0
     EMRESO=FIFTN
     EMRCUT=ZERO
     EMDX=THREE
     EMDY=THREE
     EMDZ=THREE
     EMICORE=2
     ACORE=TWO
     BCORE=TWO
     CCORE=ONE
     NEMCOMP=0
     !  Field map parameters
     EMAX=TEN*TEN
     EMAY=TEN*TEN
     EMAZ=TEN*TEN
     EMDIELC=80.0D0
     EMAPEPC=0.14D0
     EMAPEPS=70.0D0
     EMAPEPE=330.0D0
     !  Docking parameters
     REMBIND=4.0D0
!  Map guiding parameters
     EMGUID=0.05D0
!  Map movtion parameters
     EMGAMMA=1.0D0
     EMTEMP=300.0D0
    return
  end subroutine emap_iniall
  
  !
  !     Subroutines to manipulate electonic density maps
  !                 By XIONGWU WU, NIH, Nov, 2001
  !                            Enhanced Dec, 2004
  !

  LOGICAL FUNCTION CHKEMPNM(NAME,NMLEN,ID)
    !-----------------------------------------------------------------------
    !     Check the existance of a map object and return a ID for it
    !
  use dimens_fcm
  use number
    !
    INTEGER NMLEN,ID,I
    CHARACTER(len=*) NAME
    !
    CHKEMPNM=.TRUE.
    DO I=1,NEMAP
       if(EMAPID(I) == NAME)THEN
          CHKEMPNM=.FALSE.
          ID=I
       ENDIF
    ENDDO
    IF(CHKEMPNM)THEN
       NEMAP=NEMAP+1
       ID=NEMAP
       EMAPID(ID)=NAME
    ENDIF
    RETURN
  END FUNCTION CHKEMPNM

  LOGICAL FUNCTION CHKRIGNM(NAME,NMLEN,ID)
    !-----------------------------------------------------------------------
    !     Check the existance of a rigid domain and return a ID for it
    !
  use dimens_fcm
  use number

    INTEGER NMLEN,ID,I
    CHARACTER(len=*) :: NAME
    !
    CHKRIGNM=.TRUE.
    DO I=1,NEMRIG
       if(EMRIGID(I) == NAME)THEN
          CHKRIGNM=.FALSE.
          ID=I
       ENDIF
    ENDDO
    IF(CHKRIGNM)THEN
       NEMRIG=NEMRIG+1
       ID=NEMRIG
       EMRIGID(ID)=NAME
    ENDIF
    RETURN
  END FUNCTION CHKRIGNM

  LOGICAL FUNCTION RMRIGNM(NAME,NMLEN,ID)
    !-----------------------------------------------------------------------
    !     Remove a rigid domain according to its name and ID
    !
  use dimens_fcm
  use number
  use stream

    INTEGER NMLEN,ID
    CHARACTER(len=*) :: NAME
    character(len=40) :: NAME1
    !
    RMRIGNM=.FALSE.
    !      WRITE(OUTU,'("REMOVE:NEMRIG,ID,NAME=",2I8,A20)')NEMRIG,ID,NAME
    IF(ID == NEMRIG.AND.EMRIGID(ID).EQ.NAME)THEN
       NEMRIG=NEMRIG-1
       !MH08: probably wrong here ???        CHKRIGNM=.TRUE.
       RMRIGNM=.TRUE.
    ELSE
       NAME1=NAME
       CALL WRNDIE(-4,'<RMRIG>','CANNOT REMOVE RIGID:'//NAME1)
    ENDIF
    RETURN
  END FUNCTION RMRIGNM

  SUBROUTINE PRINTRIG(IDRIG)
    !-----------------------------------------------------------------------
    !  Print our rigid domain information
    !                        
  use dimens_fcm
  use number
  use stream

    INTEGER IDRIG,I
    !
    IF(PRNLEV<3)RETURN
    WRITE(OUTU,10) IDRIG,EMRIGID(IDRIG),EMAPID(IDEMRIG(IDRIG))
    WRITE(OUTU,20) (TEMRIG(I,IDRIG),I=1,3)
    WRITE(OUTU,20) (REMRIG(I,IDRIG),I=1,9)
10  FORMAT(10X,'RIG ID: ',I3," RIG NAME: ",A20," MAP NAME: ",A20)
20  FORMAT(10X,3F10.4)
    RETURN
  END SUBROUTINE PRINTRIG

  SUBROUTINE EMAPOUT(NDATA,MX,MY,MZ,LX,LY,LZ,DX,DY,DZ, &
       RHO,DDRHO,CORE)
    !-----------------------------------------------------------------------
    !       Print density array
    !
  use dimens_fcm
  use number
  use exfunc
  use stream
    !
    INTEGER NDATA,MX,MY,MZ,LX,LY,LZ,CORE(*)
    INTEGER I,IX,IY,IZ,COREI
    real(chm_real) XI,YI,ZI,DX,DY,DZ
    real(chm_real4) RHO(*),DDRHO(*),RHOI,DDRHOI
    !
    IX=0
    IY=0
    IZ=0
    WRITE(OUTU,'("            INDEX      XI      YI      ZI", &
         & "       RHO          DDRHO          ICORE")')
    do i=1,ndata
       RHOI=RHO(I)
       DDRHOI=DDRHO(I)
       COREI=CORE(I)
       XI=(MX+IX)*DX
       YI=(MY+IY)*DY
       ZI=(MZ+IZ)*DZ
       WRITE(OUTU,10)I, XI,YI,ZI,RHOI,DDRHOI,COREI
       IX=IX+1
       IF(IX >= LX)THEN
          IX=0
          IY=IY+1
          IF(IY >= LY)THEN
             IY=0
             IZ=IZ+1
          END IF
       END IF
10     FORMAT('<EMAPOUT> ',I10,3F8.2,2E14.6,I6)
    enddo
    return
  end SUBROUTINE EMAPOUT

  SUBROUTINE RDSEGID(IUNIT,COMLYN,COMLEN)
    !-----------------------------------------------------------------------
    !     THIS ROUTINE READ CARD COORDINATE FILE OR PDB FILE AND
    !     CREAT A NEW SEGMENT WITHOUT SETUP BONDING TREES
    !     THIS PROVIDE A QUICK WAY TO BUILD MOLECULAE STRUCTURE FOR
    !     EMAP FITTING
    !        READ SEGID id_head UNIT unit 
    !
    !     Enhanced to allow segments with full CHARMM structures to be build 
    !     based on input PDB file.  Each chain as a segment.
    !        READ SEGID id_head UNIT unit BUILd 
    !
    !     To build Atoms with missing coordinates, SETUp must be used 
    !     to set up internal coordinates (IC) table:
    !        READ SEGID id_head UNIT unit BUILd SETUp
    !
  use memory
  use dimens_fcm
  use cnst_fcm, only: cnst_is_allocated, allocate_cnst
  use number
  use stream
  use psf
  use genpsf_m, only: genic
  use coord
  use ctitla
  use string
#ifdef KEY_RESIZE
  use resize
#endif  

    INTEGER IUNIT,COMLEN
    character(len=*) COMLYN 
    !
    LOGICAL LCARD,LPDB,LFREE,LBUILD
    !
    INTEGER NRESP,IRESP,IRES,ISEQ,IATOM,NAT,NATP,NSEGP
    real(chm_real)  XIN,YIN,ZIN,WIN,AMASSI
    character(len=10) CXIN,CYIN,CZIN,CWIN
    character(len=8) CHAINI,CHAINP,RESIN,ATOMIN,SID,SIDP,RID,RIDP
    character(len=8) PATF, PATL
    LOGICAL ERROR, LHETATM, LSETIC, LWARN
    LOGICAL EOF, QATOM
    INTEGER SLEN, IJ,I,J,ISTART,ISTOP,IDLEN,IDLEN1
    character(len=80) LYN,PDBLIN
    INTEGER MXLEN,NTITL,ICATOM
    character(len=8) NSEGID,RDELE
    integer,allocatable,dimension(:) :: istk
    character(len=86) fmt62
#ifdef KEY_RESIZE
    integer na0,nr0,ns0,ng0
    na0=natom; nr0=nres; ns0=nseg; ng0=ngrp
#endif

    write(fmt62,'(a,a)') "(' ** WARNING ** Error or EOF on input file.',", &
         "I5,' coordinates read.',I5,' expected.')"

    !
    CALL NXTWDA(COMLYN,COMLEN,NSEGID,4,IDLEN,1)
    IDLEN=IDLEN+1
    IF(IDLEN == 1) &
         CALL WRNDIE(0,'<RDSEGID>','INVALID SEGMENT NAME'//NSEGID)
    LBUILD=INDXA(COMLYN,COMLEN,'BUIL') > 0
    LHETATM=INDXA(COMLYN,COMLEN,'HETA') > 0
    IF(LBUILD.AND.IDLEN > 3)IDLEN=4
    LCARD=INDXA(COMLYN,COMLEN,'CARD') > 0
    LFREE=INDXA(COMLYN,COMLEN,'FREE') > 0
    LPDB=.NOT.(LCARD.OR.LFREE)
    LSETIC = INDXA(COMLYN, COMLEN, 'SETU')  >  0
    LWARN  = INDXA(COMLYN, COMLEN, 'WARN')  >  0
    !
    PATF = GTRMA(COMLYN, COMLEN, 'FIRS')
    IF (PATF  ==  ' ') PATF = 'DEFA'
    PATL = GTRMA(COMLYN, COMLEN, 'LAST')
    IF (PATL  ==  ' ') PATL = 'DEFA'
    !
    RDELE = GTRMA(COMLYN, COMLEN, 'DELE')
    IF (RDELE  ==  ' ') RDELE = 'HOH'
    !
    IF(IUNIT < 0 ) IUNIT=ISTRM
    IF(PRNLEV >= 2) WRITE(OUTU,430) IUNIT
430 FORMAT(10X,'Segment and COORDINATES BEING READ FROM UNIT',I3)
    IF(IUNIT < 0) THEN
       CALL WRNDIE(0,'<COORIO>','INVALID UNIT NUMBER')
       RETURN
    ENDIF
    !
    IF(IOLEV > 0 .AND. IUNIT /= ISTRM) THEN
       IF(LCARD) REWIND IUNIT
    ENDIF
    !
    EOF=.FALSE.
    !
    CALL TRYORO(IUNIT,'FORMATTED')
    !
    !  Start read and build new segment
    !
    NAT=NATOM
    NSEGP=NSEG
    NRESP=NRES
    CHAINP='____'
    RIDP='____'
    SIDP='____'
    ISTART=NRES+1
    ISTOP=NRES
    !  read in sequences for each segments
    IF(LBUILD.AND.LPDB)THEN
510    CONTINUE
       READ(IUNIT,'(A)',END=520,ERR=520) PDBLIN
       !ln...05-Jan-95, convert the string to upper case
       SLEN=LEN(PDBLIN)
       CALL CNVTUC(PDBLIN,SLEN)
       !ln...(1)
       IF (PDBLIN(1:6) == 'ATOM  '.OR.PDBLIN(1:6).EQ.'HETATM') THEN
          READ(PDBLIN, &
               '(6X,I5,1X,A4,A4,A2,A5,3X,3F8.3,6X,F6.2,6X,A4)') &
               ISEQ,ATOMIN,RESIN,CHAINI,RID,XIN,YIN,ZIN,WIN,SID
          IF (.NOT.LHETATM.AND.PDBLIN(1:6) == 'HETATM')GOTO 510
          IF (RESIN == RDELE)GOTO 510
          IF (CHAINI /= CHAINP .OR. SID /= SIDP) THEN
             !   if this is a new chain, patch the previous segment and current segment
             IF(ISTART < ISTOP)THEN
                NSEG=NSEG+1
#ifdef KEY_RESIZE
                if (nseg .gt. ns0) then
                   ns0=nseg+drsz0
                   call resize_psf('emapsubs.F90','RDSEGID','NSEG',ns0, .true.)
                endif
#endif        
                NRES=ISTOP
#ifdef KEY_RESIZE
                if (nres .gt. nr0) then
                   nr0=nres+drsz0
                   call resize_psf('emapsubs.F90','RDSEGID','NRES',nr0, .true.)
                endif
                CALL GENIC(ISTART,ISTOP,LWARN,LSETIC,PATF,PATL,.FALSE.)
#else
                call chmalloc('emapsubs.src','RDSEGID','Istk',MAXRES,intg=Istk)
                CALL GENIC(ISTART,ISTOP,LWARN,LSETIC,PATF,PATL,Istk,.FALSE.)
                call chmdealloc('emapsubs.src','RDSEGID','Istk',MAXRES,intg=Istk)
#endif
                SEGID(NSEG) = NSEGID
                IF(NSEG > 99)THEN
                   IDLEN1=IDLEN-2
                   WRITE(SEGID(NSEG)(IDLEN1:IDLEN),'(I3)')NSEG
                ELSE IF(NSEG > 9)THEN
                   IDLEN1=IDLEN-1
                   WRITE(SEGID(NSEG)(IDLEN1:IDLEN),'(I2)')NSEG
                ELSE
                   WRITE(SEGID(NSEG)(IDLEN:IDLEN),'(I1)')NSEG
                ENDIF
                ISTART=ISTOP+1
                CALL PSFSUM(OUTU)
             ENDIF
             CHAINP=CHAINI
             SIDP=SID
          ENDIF
          IF(RID /= RIDP)THEN
             !   if this is a residue
             RIDP=RID
             ISTOP=ISTOP+1
             !
             ! make RESIN left-justified
             !
             IJ=4
             CALL TRIMA(RESIN,IJ)
             !
             ! make RID left-justified
             !
             IJ=4
             CALL TRIMA(RID,IJ)
#ifdef KEY_RESIZE
             if (istop .gt. nr0) then
                nr0=nres+drsz0
                call resize_psf('emapsubs.F90','RDSEGID','NRES',nr0, .true.)
             endif
#endif
             RES(ISTOP)=RESIN
             RESID(ISTOP)=RID
          ENDIF
       ENDIF
       GOTO 510
520    REWIND(IUNIT)
       IF(ISTART < ISTOP)THEN
          NSEG=NSEG+1
#ifdef KEY_RESIZE
          if (nseg .gt. ns0) then
             ns0=nseg+drsz0
             call resize_psf('emapsubs.F90','RDSEGID','NSEG',ns0, .true.)
          endif
#endif        
          NRES=ISTOP
#ifdef KEY_RESIZE
          if (nres .gt. nr0) then
             nr0=nres+drsz0
             call resize_psf('emapsubs.F90','RDSEGID','NRES',nr0, .true.)
          endif
          CALL GENIC(ISTART,ISTOP,LWARN,LSETIC,PATF,PATL,.FALSE.)
#else
          call chmalloc('emapsubs.src','RDSEGID','Istk',MAXRES,intg=Istk)
          CALL GENIC(ISTART,ISTOP,LWARN,LSETIC,PATF,PATL,Istk,.FALSE.)
          call chmdealloc('emapsubs.src','RDSEGID','Istk',MAXRES,intg=Istk)
#endif
          SEGID(NSEG) = NSEGID
          IF(NSEG > 99)THEN
             IDLEN1=IDLEN-2
             WRITE(SEGID(NSEG)(IDLEN1:IDLEN),'(I3)')NSEG
          ELSE IF(NSEG > 9)THEN
             IDLEN1=IDLEN-1
             WRITE(SEGID(NSEG)(IDLEN1:IDLEN),'(I2)')NSEG
          ELSE
             WRITE(SEGID(NSEG)(IDLEN:IDLEN),'(I1)')NSEG
          ENDIF
          CALL PSFSUM(OUTU)
       ENDIF
    ENDIF
    IF (LPDB) THEN
       !
       ! read PDB title
       !
       NTITL=0
99963  CONTINUE
       READ(IUNIT,'(A)',END=61,ERR=61) PDBLIN
       !ln...05-Jan-95, convert the string to upper case
       SLEN=LEN(PDBLIN)
       CALL CNVTUC(PDBLIN,SLEN)
       !ln...(1)
       IF (PDBLIN(1:6) == 'REMARK') THEN
          NTITL=NTITL+1
          TITLEB(NTITL)=PDBLIN(8:)
          GOTO 99963
       ENDIF
       CALL WRTITL(TITLEB,NTITL,OUTU,+1)
       IATOM=9999999
    ELSE
       ! Read CARD file
       CALL RDTITL(TITLEB,NTITL,IUNIT,0)
       READ(IUNIT,30) IATOM
30     FORMAT(I5)
    ENDIF
    !
    RIDP=''
    ISTOP=NRESP
    !
    iloop: DO I=1,IATOM
       !
       IF (LFREE) THEN
          !
          ! FREE FIELD INPUT
          !
          CALL RDCMND(LYN,MXLEN,SLEN,IUNIT,EOF,.FALSE.,.FALSE.,' ')
          ISEQ=0
          IRES=0
          RESIN='    '
          ATOMIN='    '
          XIN=0.0
          YIN=0.0
          ZIN=0.0
          SID='    '
          RID='    '
          WIN=0.0
          CALL TRIME(LYN,SLEN)
          IF(SLEN > 0) ISEQ=NEXTI(LYN,SLEN)
          CALL TRIME(LYN,SLEN)
          IF(SLEN > 0) IRES=NEXTI(LYN,SLEN)
          CALL TRIME(LYN,SLEN)
          IF(SLEN > 0) RESIN=NEXTA4(LYN,SLEN)
          CALL TRIME(LYN,SLEN)
          IF(SLEN > 0) ATOMIN=NEXTA4(LYN,SLEN)
          CALL TRIME(LYN,SLEN)
          IF(SLEN > 0) XIN=NEXTF(LYN,SLEN)
          CALL TRIME(LYN,SLEN)
          IF(SLEN > 0) YIN=NEXTF(LYN,SLEN)
          CALL TRIME(LYN,SLEN)
          IF(SLEN > 0) ZIN=NEXTF(LYN,SLEN)
          CALL TRIME(LYN,SLEN)
          IF(SLEN > 0) SID=NEXTA4(LYN,SLEN)
          CALL TRIME(LYN,SLEN)
          IF(SLEN > 0) RID=NEXTA4(LYN,SLEN)
          CALL TRIME(LYN,SLEN)
          IF(SLEN > 0) WIN=NEXTF(LYN,SLEN)
          CALL TRIME(LYN,SLEN)
          IF(SLEN > 0) CALL XTRANE(LYN,SLEN,'CREAD')
       ELSE IF(LPDB)THEN
          !
          ! PDB format
          !
          QATOM=.FALSE.
          do while(.not. qatom)
             QATOM=.FALSE.
             IF (PDBLIN(1:3) == 'END') THEN
                IF(LBUILD)return
                IF(IATOM /= 99999999 .AND. WRNLEV >= 2)WRITE(OUTU,fmt62) I-1,IATOM
                exit iloop
             ELSE IF (PDBLIN(1:4) == 'ATOM') THEN
                QATOM=.TRUE.
             ELSE 
                IF (PDBLIN(1:4) == 'HETA'.AND.LHETATM) THEN
                   QATOM=.TRUE.
                ELSE
                   !
                   ! keep reading until reaching ATOM or END
                   !
                   READ(IUNIT,'(A)',END=61,ERR=61) PDBLIN
                   !ln...05-Jan-95, convert the string to upper case
                   SLEN=LEN(PDBLIN)
                   CALL CNVTUC(PDBLIN,SLEN)
                   !ln...(2)
                ENDIF
             ENDIF
          enddo
          !
          !               process ATOM line
          READ(PDBLIN, &
               '(6X,I5,1X,A4,A4,A2,A5,3X,3F8.3,6X,F6.2,6X,A4)') &
               ISEQ,ATOMIN,RESIN,CHAINI,RID,XIN,YIN,ZIN,WIN,SID
          !
          ! make ATOMIN left-justified
          !
          IJ=4
          CALL TRIMA(ATOMIN,IJ)
          !
          ! make RESIN left-justified
          !
          IJ=4
          CALL TRIMA(RESIN,IJ)
          !
          ! make RID left-justified
          !
          IJ=4
          CALL TRIMA(RID,IJ)
          !
          IF(LBUILD.AND.RESIN /= RDELE)THEN
             IF(RID /= RIDP)THEN
                !   if this is a new chain, patch the previous segment and current segment
                RIDP=RID
                ISTOP=ISTOP+1
                IF(RES(ISTOP) /= RESIN) &
                     CALL WRNDIE(0,'<RDSEGID>','Mismatched Residues:'//PDBLIN)
                NATP=NAT+1
                NAT=IBASE(ISTOP+1)
             ENDIF
             !   Assign the coordinate to named atom
             ICATOM=-1
             DO J=NATP,NAT
                IF (ATYPE(J) == ATOMIN) THEN
                   X(J)=XIN
                   Y(J)=YIN
                   Z(J)=ZIN
                   ICATOM=J
                ENDIF
             ENDDO
             IF(ICATOM < 0) &
                  CALL WRNDIE(0,'<RDSEGID>','Problem with PDB record:'//PDBLIN)
          ENDIF
          !
          ! read next PDB line
          !
          READ(IUNIT,'(A)',ERR=61,END=61) PDBLIN
          !ln...05-Jan-95, convert the string to upper case
          SLEN=LEN(PDBLIN)
          CALL CNVTUC(PDBLIN,SLEN)
          !ln...(3)
       ELSE
          !  READ CARD file
          READ(IUNIT,'(A)',ERR=61,END=61) PDBLIN
          SLEN=LEN(PDBLIN)
          CALL CNVTUC(PDBLIN,SLEN)
          READ(PDBLIN,40) ISEQ,IRES,RESIN,ATOMIN, &
               CXIN,CYIN,CZIN,SID,RID,CWIN
40        FORMAT(2I5,2(1X,A4),3A10,1X,A4,1X,A4,A10)
       ENDIF
       !
889    CONTINUE
       IF(LBUILD)cycle iloop
       !
       NATOM=NATOM+1
#ifdef KEY_RESIZE
       if (natom .gt. na0) then
          na0=na0+drsz0
          call resize_psf('emapsubs.F90','RDSEGID','NATOM',na0, .true.)
          call resize_coord('emapsubs.F90','RDSEGID',na0, .true.)
          call resize_coordc('emapsubs.F90','RDSEGID',na0, .true.)
          call resize_deriv('emapsubs.F90','RDSEGID',na0, .true.)
       endif
#endif        

       IF (LFREE .OR. LPDB) THEN
          X(NATOM)=XIN
          Y(NATOM)=YIN
          Z(NATOM)=ZIN
          WMAIN(NATOM)=WIN
       ELSE
          READ(CXIN,'(F10.5)',ERR=58) X(NATOM)
          READ(CYIN,'(F10.5)',ERR=58) Y(NATOM)
          READ(CZIN,'(F10.5)',ERR=58) Z(NATOM)
          READ(CWIN,'(F10.5)',ERR=58) WMAIN(NATOM)
          GOTO 59
58        CONTINUE
          CALL WRNDIE(1,'<CREAD>', &
               'Bad characters in coordinate field: Initialized')
          X(NATOM)=ANUM
          Y(NATOM)=ANUM
          Z(NATOM)=ANUM
          WMAIN(NATOM)=ZERO
59        CONTINUE
       ENDIF
       ATYPE(NATOM)=ATOMIN
       IF(ATOMIN(1:2) == 'FE')THEN
          AMASSI=55.6
       ELSE IF(ATOMIN(1:2) == 'CL')THEN
          AMASSI=35.5
       ELSE IF(ATOMIN(1:1) == 'C')THEN
          AMASSI=12.0
       ELSE IF(ATOMIN(1:1) == 'O')THEN
          AMASSI=16.0
       ELSE IF(ATOMIN(1:1) == 'N')THEN
          AMASSI=14.0
       ELSE IF(ATOMIN(1:1) == 'H')THEN
          AMASSI=1.0
       ELSE IF(ATOMIN(1:1) == 'P')THEN
          AMASSI=31.0
       ELSE IF(ATOMIN(1:1) == 'S')THEN
          AMASSI=32.0
       ELSE 
          AMASSI=1.0
       ENDIF
       AMASS(NATOM)=AMASSI
       CG(NATOM)=ZERO
       IAC(NATOM)=1    
       IBLO(NATOM)=0
       IMOVE(NATOM)=0
       RSCLF(NATOM)=1.0
#if KEY_WCA==1
       WCA(NATOM)=1.0      
#endif
       WMAIN(NATOM)=AMASSI
       IF(RID /= RIDP)THEN
          NRES=NRES+1
#ifdef KEY_RESIZE
          if (nres .gt. nr0) then
             nr0=nres+drsz0
             call resize_psf('emapsubs.F90','RDSEGID','NRES',nr0, .true.)
          endif
#endif    
          RIDP=RID
          RES(NRES)=RESIN
          RESID(NRES)=RID
          IBASE(NRES)=NATOM-1
          NGRP=NGRP+1
#ifdef KEY_RESIZE
          if (ngrp .gt. ng0) then
             ng0=ngrp+drsz0
             call resize_psf('emapsubs.F90','RDSEGID','NGRP',ng0, .true.)
          endif
#endif
          IMOVEG(NGRP)=0
          IGPBS(NGRP)=NATOM-1
          IGPTYP(NGRP)=0
       ENDIF
       !  End of loop over input lines
    enddo iloop

    if (cnst_is_allocated) call allocate_cnst(natom)
    
    GOTO 63
61  I=I-1
    IF(LBUILD)return
    IF(IATOM /= 99999999 .AND. WRNLEV >= 2) WRITE(OUTU,fmt62) I,IATOM
63  CONTINUE
    !
    !       Done reading coordinates.
    !   Define new segment
    NSEG=NSEG+1
#ifdef KEY_RESIZE
    if (nseg .gt. ns0) then
       ns0=nseg+drsz0
       call resize_psf('emapsubs.F90','RDSEGID','NSEG',ns0, .true.)
    endif
#endif        
    IBASE(NRES+1)=NATOM
    IGPBS(NGRP+1)=NATOM
    SEGID(NSEG)=NSEGID
    NICTOT(NSEG)=NRESP
    NICTOT(NSEG+1)=NRES
    !       write(OUTU,*)natom,nres,nseg,ngrp,DMIN,DMAX,DCUT
#ifdef KEY_RESIZE
    if (natom .lt. na0) then
       call resize_psf('emapsubs.F90','RDSEGID','NATOM',natom, .true.)
       call resize_coord('emapsubs.F90','RDSEGID',natom, .true.)
       call resize_coordc('emapsubs.F90','RDSEGID',natom, .true.)
       call resize_deriv('emapsubs.F90','RDSEGID',natom, .true.)
    endif
    if (nres .lt. nr0) &
         call resize_psf('emapsubs.F90','RDSEGID','NRES',nres, .true.)
    if (nseg .lt. ns0) &
         call resize_psf('emapsubs.F90','RDSEGID','NSEG',nseg, .true.)
    if (ngrp .lt. ng0) &
         call resize_psf('emapsubs.F90','RDSEGID','NGRP',ngrp, .true.)
#else
    IF(NATOM > MAXA .OR. &
         NSEG > MAXSEG .OR. &
         NRES > MAXRES .OR. &
         NGRP > MAXGRP ) THEN
       WRITE(OUTU,'("<RDSEGID> You need to increase skip or rcut")')
       CALL WRNDIE(-2,'<RDSEGID>','Size limit exceeded')
       RETURN
    ENDIF
#endif
    CALL PSFSUM(OUTU)
    RETURN
  END SUBROUTINE RDSEGID

  SUBROUTINE EMAPREST(comlyn,comlen,islct,jslct,lcomp,x,y,z,wmain, &
       xcomp,ycomp,zcomp,wcomp)
    !-----------------------------------------------------------------------
    !     This routine set EMAP restraint 
    !     Usage:
    !     CONS EMAP {force real atom-selection map-definition [MOVE] }
    !               {force real RIGId string                         }
    !               {SHOW                                            }    
    !               {RESEt|CLEAr                             }
    !     map-definition= {MAPId string     }
    !                     {FILEname string      }
    !                     {[DX real][DY real][DZ real][RESO real][COMP]}

    use chm_kinds
    use dimens_fcm
    use number
    use psf
    use select
    use stream
    use string

    implicit none

    character(len=*) COMLYn 
    INTEGER   COMLEn
    integer,intent(inout),dimension(natom) :: islct,jslct
    real(chm_real) x(*),y(*),z(*),wmain(*)
    real(chm_real) xcomp(*),ycomp(*),zcomp(*),wcomp(*)

    !-----------------------------------------------------------------------
    ! Local variables
    CHARACTER(len=4)   WRD
    CHARACTER(len=80)   NAME, FNAME,FMAP
    INTEGER I, IDEMP, IDRIG, IUNIT, LENGTH, FLEN,FMLEN
    REAL(CHM_REAL)  DEMPX,DEMPY,DEMPZ,RESO
    LOGICAL LFIX, LCOMP, NEWEMP, NEWRIG
    REAL(CHM_REAL)  DRBFF
    if_keyword: if(indxa(comlyn,comlen,'SHOW') .gt. 0)then
       IF(PRNLEV > 3)call prnemaprest()
       return

    elseif(indxa(comlyn,comlen,'RESE').gt.0.or. &
         indxa(comlyn,comlen,'CLEA').gt.0)then   if_keyword
       if(LEMAPENG)then
          if(prnlev.gt.2) WRITE(OUTU,100)nrigrest,' CONS EMAP restraints cleared '
100    format("<EMAPREST>", I4,A)
          do i=1,NRIGREST
             idrig = IDRIGREST(nrigrest-i+1)
             idemp = IDEMRIG(idrig)
             name=emrigid(idrig)
             if(INDEX(name,'restmap')>0)then
          if(prnlev.gt.2) WRITE(OUTU,110)'Remove rigid: ', idrig,trim(emrigid(idrig))
               CALL RMEMRIG(IDRIG)
          if(prnlev.gt.2) WRITE(OUTU,110)'Remove emap: ', idemp,trim(emapid(idemp))
               CALL RMEMAP(IDEMP)
             endif
          enddo
110    format("<EMAPREST>", A,I4," Name: ",A)
       endif
       NEMCOMP = NEMCOMP - NRIGREST
       NRIGREST = 0
       IF(NEMCOMP == 0 )THEN
         LEMAPENG=.false.
         LEMAPLD=.false.
       ENDIF
    else   if_keyword          ! get parameter values
      if(.not.LEMAPENG)then     
        NEMCOMP = 0
        NRIGREST = 0
        LEMAPENG  = .true.
        LEMAPLD=.false.
      endif
      NEMCOMP = NEMCOMP + 1
      NRIGREST=NRIGREST+1
      ! restraint constant, kcal/mol
      EMGUID = gtrmf(comlyn,comlen,'FORC',zero)
      LFIX=(INDXA(COMLYN,COMLEN,'MOVE') .LE.  0)
      LCOMPFIX(NEMCOMP)=LFIX
      LEMAPLD=LEMAPLD.OR..NOT.LFIX
      ! parameters to generate a map
      DEMPX=GTRMF(COMLYN,COMLEN,'DX',EMDX)
      DEMPY=GTRMF(COMLYN,COMLEN,'DY',EMDY)
      DEMPZ=GTRMF(COMLYN,COMLEN,'DZ',EMDZ)
      RESO=(DEMPX+DEMPY+DEMPZ)/2.0
      RESO=GTRMF(COMLYN,COMLEN,'RESO',RESO)
      CALL GTRMWA(COMLYN,COMLEN,'RIGI',4,NAME,80,LENGTH)
      IF(LENGTH > 0)THEN
        ! using rigid domain to define the EMAP restraint
          NEWRIG=CHKRIGNM(NAME,LENGTH,IDRIG)
          IF(NEWRIG)THEN
            NEMCOMP=NEMCOMP-1
            NRIGREST=NRIGREST-1
            CALL WRNDIE(0,'<EMAP>','NO rigid ID:'//NAME(1:LENGTH))
            CALL RMEMRIG(IDRIG)
            RETURN
          ENDIF
          IDEMP=IDEMRIG(IDRIG)
      ELSE
        ! select atoms for the EMAP restraint
        IF(LCOMP)THEN
          !
          ! Now find the atoms in comparison set for map generation.
          !
            CALL SELCTA(COMLYN,COMLEN,ISLCT,XCOMP,YCOMP,ZCOMP,WCOMP,.TRUE.)
       
        ELSE
          !
          ! Now find the atoms for map generation.
          !
           CALL SELCTA(COMLYN,COMLEN,ISLCT,X,Y,Z,WMAIN,.TRUE.)
             
        ENDIF
        ! restraint map
        CALL GTRMWA(COMLYN,COMLEN,'MAPI',4,NAME,80,LENGTH)
        IF(LENGTH > 0)THEN
          ! using existing map
          NEWEMP=CHKEMPNM(NAME,LENGTH,IDEMP)
          IF(NEWEMP)THEN
            NEMCOMP=NEMCOMP-1
            NRIGREST=NRIGREST-1
            CALL WRNDIE(0,'<EMAP>','NO EMAP:'//NAME(1:LENGTH))
            CALL RMEMAP(IDEMP)
            RETURN
          ENDIF
        ELSE
          ! no existing map
          WRITE(NAME,'("restmap",I1)')NRIGREST
          NEWEMP=CHKEMPNM(NAME,LEN(NAME),IDEMP)
          IF(.NOT.NEWEMP)THEN
            NEMCOMP=NEMCOMP-1
            NRIGREST=NRIGREST-1
            CALL WRNDIE(0,'<EMAPOPT>','ALREADY EXISTS EMAP NAMED:'//NAME)
            CALL RMEMAP(IDEMP)
            RETURN
          ENDIF
          IUNIT=GTRMI(COMLYN,COMLEN,'UNIT',-1)
          CALL GTRMWA(COMLYN,COMLEN,'FILE',4,FNAME,80,FLEN)
          CALL GTRMWA(COMLYN,COMLEN,'FORM',4,FMAP,80,FMLEN)
           IF(FMLEN.EQ.0)THEN
             FMAP=''
             IF(INDEX(fname,'.ccp4')>0.or.INDEX(fname,'.CCP4')>0)FMAP='CCP4'
             IF(INDEX(fname,'.map')>0.or.INDEX(fname,'.MAP')>0)FMAP='CCP4'
             IF(INDEX(fname,'.mrc')>0.or.INDEX(fname,'.MRC')>0)FMAP='MRC'
             IF(INDEX(fname,'.pdb')>0.or.INDEX(fname,'.PDB')>0)FMAP='PDB'
          ENDIF
          IF(IUNIT>0 .or. FLEN>0)THEN
            ! readin a map
            CALL RDEMAP(FNAME(1:FLEN),IUNIT,IDEMP,FMAP)
          ELSE
            MEMAPX(IDEMP)=GTRMI(COMLYN,COMLEN,'MX',-999999)
            MEMAPY(IDEMP)=GTRMI(COMLYN,COMLEN,'MY',-999999)
            MEMAPZ(IDEMP)=GTRMI(COMLYN,COMLEN,'MZ',-999999)
            LEMAPX(IDEMP)=GTRMI(COMLYN,COMLEN,'LX',-999999)
            LEMAPY(IDEMP)=GTRMI(COMLYN,COMLEN,'LY',-999999)
            LEMAPZ(IDEMP)=GTRMI(COMLYN,COMLEN,'LZ',-999999)
            ! define allowable region
            LOWEMAP(1,IDEMP)=GTRMF(COMLYN,COMLEN,'XLOW',-RBIG)
            LOWEMAP(2,IDEMP)=GTRMF(COMLYN,COMLEN,'YLOW',-RBIG)
            LOWEMAP(3,IDEMP)=GTRMF(COMLYN,COMLEN,'ZLOW',-RBIG)
            UPEMAP(1,IDEMP)=GTRMF(COMLYN,COMLEN,'XUP',RBIG)
            UPEMAP(2,IDEMP)=GTRMF(COMLYN,COMLEN,'YUP',RBIG)
            UPEMAP(3,IDEMP)=GTRMF(COMLYN,COMLEN,'ZUP',RBIG)
            TRNEMAP(1,IDEMP)=1.0D0
            TRNEMAP(2,IDEMP)=1.0D0
            TRNEMAP(3,IDEMP)=1.0D0
            ROTEMAP(1,IDEMP)=1.0D0
            ROTEMAP(2,IDEMP)=1.0D0
            ROTEMAP(3,IDEMP)=1.0D0
            IF(INDXA(COMLYN,COMLEN,'NOX')  >  0 )TRNEMAP(1,IDEMP)=0.0D0
            IF(INDXA(COMLYN,COMLEN,'NOY')  >  0 )TRNEMAP(2,IDEMP)=0.0D0
            IF(INDXA(COMLYN,COMLEN,'NOZ')  >  0 )TRNEMAP(3,IDEMP)=0.0D0
            IF(INDXA(COMLYN,COMLEN,'NORX')  >  0 )ROTEMAP(1,IDEMP)=0.0D0
            IF(INDXA(COMLYN,COMLEN,'NORY')  >  0 )ROTEMAP(2,IDEMP)=0.0D0
            IF(INDXA(COMLYN,COMLEN,'NORZ')  >  0 )ROTEMAP(3,IDEMP)=0.0D0
           
            IF(PRNLEV > 3)WRITE(OUTU,1310)NAME,IDEMP,(TRNEMAP(I,IDEMP),I=1,3),(ROTEMAP(I,IDEMP),I=1,3)
1310    FORMAT(" <EMAPREST> ",A10, &
             " ID:",I6," Trans. X,Y,Z: ",3F4.1," Rotat. X,Y,Z: ",3F4.1)
            IF(LCOMP)THEN
       !
       ! Now use selected atoms in comparison set for map generation.
       !
              CALL EMAPGEN(NATOM,XCOMP,YCOMP,ZCOMP,AMASS, &
                ISLCT,IDEMP,DEMPX,DEMPY,DEMPZ,RESO)
            ELSE
       !
       ! Now use selected atoms for map generation.
       !
             CALL EMAPGEN(NATOM,X,Y,Z,AMASS, &
                ISLCT,IDEMP,DEMPX,DEMPY,DEMPZ,RESO)
            ENDIF
          ENDIF
        ENDIF
        ! define rigid domain for the restraint map
        NEWRIG=CHKRIGNM(NAME,LENGTH,IDRIG)
        IF(.NOT.NEWRIG)THEN
            CALL WRNDIE(0,'<EMAP>', &
             'Overlapping existing rigid ID:'//NAME(1:LENGTH))
        ENDIF
        CALL EMAPASN(NATOM,X,Y,Z,AMASS,ISLCT,IDEMP,IDRIG)
      ENDIF
      IDCOMPS(NEMCOMP)=IDRIG
      IDRIGREST(NRIGREST)=IDRIG
      CALL EMAPGUIDPRM(IDRIG,IDEMP,AMASS,     &
            NATEMAP(IDEMP),EMPCRD(IDEMP)%IATOM)
      IF(LFIX)THEN
            IF(PRNLEV > 3)WRITE(OUTU,1410)NEMCOMP,TRIM(EMRIGID(IDRIG)),EMGUID
      ELSE
            IF(PRNLEV > 3)WRITE(OUTU,1420)NEMCOMP,TRIM(EMRIGID(IDRIG)),EMGUID
      ENDIF
1410      FORMAT(" <EMAPCONS> No. ",I4," map-restraint: ",A,     &
         " is FIXED   with FMAP=",F10.4," kcal/g")
1420      FORMAT(" <EMAPCONS> No. ",I4," map-restraint: ",A,     &
         " is MOVABLE with FMAP=",F10.4," kcal/g")
    endif    if_keyword

    return
  end subroutine emaprest

  SUBROUTINE PRNEMAPREST()
    !-----------------------------------------------------------------------
    !     This routine print out map restraint information

    INTEGER I
    INTEGER IDEMP,IDRIG
!
    DO I=1,NEMCOMP
        IDRIG=IDCOMPS(I)
        IDEMP=IDEMRIG(IDRIG)
        IF(LCOMPFIX(I))THEN
          WRITE(OUTU,1010)I,IDEMP,IDRIG, &
            TRIM(EMAPID(IDEMP)),TRIM(EMRIGID(IDRIG)),EMAPGUID(I)
        ELSE
          WRITE(OUTU,1020)I,IDEMP,IDRIG, &
            TRIM(EMAPID(IDEMP)),TRIM(EMRIGID(IDRIG)),EMAPGUID(I)
        ENDIF
    ENDDO
1010  FORMAT(" <EMAPCONS> NO,IDEMP,IDRIG: ",3I4," MAP: ",A, &
                  " RIG: ",A," Fwork: ",F8.4," FIXED")
1020  FORMAT(" <EMAPCONS> NO,IDEMP,IDRIG: ",3I4," MAP: ",A, &
                  " RIG: ",A," Fwork: ",F8.4," MOVABLE")
    RETURN
  END SUBROUTINE PRNEMAPREST


  SUBROUTINE MVCOREDT(NDATA,CORE,RHO)
    !-----------------------------------------------------------------------
    !     This routine move core data to rho array
    !     in CCP4 electronic map  format

    INTEGER NDATA
    integer,dimension(1:ndata) :: CORE
    real(chm_real4),dimension(1:ndata) :: RHO

    RHO=real(CORE,chm_real4)
    RETURN
  END SUBROUTINE MVCOREDT

  SUBROUTINE MVDDRDT(NDATA,DDR,RHO)
    !-----------------------------------------------------------------------
    !     This routine move ddr data to rho array
    !     in CCP4 electronic map  format
    !
    INTEGER NDATA
    real(chm_real4),dimension(1:ndata) :: RHO,DDR
    !
    RHO=DDR
    RETURN
  END SUBROUTINE MVDDRDT

  SUBROUTINE EMAPSCALE(IDEMP,S)
    !-----------------------------------------------------------------------
    !     This routeine scale the distribution properties of a map
    !
    INTEGER IDEMP,NDATA
    real(chm_real) S
    !
    NDATA=LEMAPX(IDEMP)*LEMAPY(IDEMP)*LEMAPZ(IDEMP)
    CALL EMAPSCALE1(NDATA,empgrd(IDEMP)%rho, &
         empgrd(IDEMP)%ddrho,S)
    RETURN
  END SUBROUTINE EMAPSCALE

  SUBROUTINE EMAPSCALE1(NEMP,RHO,DDR,S)
    !-----------------------------------------------------------------------
    integer NEMP,I
    real(chm_real) S
    real(chm_real4) RHO(*),DDR(*)
    !
    do i=1,NEMP
       RHO(I)=S*RHO(I)
       DDR(I)=S*DDR(I)
    ENDDO
    RETURN
  END SUBROUTINE EMAPSCALE1

       
      SUBROUTINE WRTCCP4(UNIT,IDEMP,RHO,ISMRC)
!_________________________________________________________________
!  write out CCP4 map information and its density array
!                    By Xiongwu Wu, wuxw@nhlbi.nih.gov
!_________________________________________________________________
  implicit none
      INTEGER IDEMP
      real(chm_real4) rho(*)
      INTEGER UNIT
      LOGICAL ISMRC
!
      INTEGER*4 LX,LY,LZ,MODE,MNX,MNY,MNZ,NX,NY,NZ
      real(chm_real4) DX,DY,DZ,XL,YL,ZL,ALPHA,BETA,GAMMA
      real(chm_real4) x0,y0,z0
      INTEGER*4 MAPC,MAPR,MAPS
      real(chm_real4) AMAX,AMIN,AMEAN,ARMS
      INTEGER*4 ISPG,NSYMBT,LSKFLG,NNOTE
      real(chm_real4) SKWMAT(9),SKWTRN(3),EXTRA(15),EXTRA25(25)
      CHARACTER(len=4) MAPLABLE,MACHST
      CHARACTER(len=80) NOTES(10)
      INTEGER*1 BDATAI
      INTEGER*2 IDATAI
      real(chm_real4) RDATAI
      COMPLEX(KIND=4) CDATAI
!
      INTEGER I,NDATA,FILLEN
!
      LX=LEMAPX(IDEMP)
      LY=LEMAPY(IDEMP)
      LZ=LEMAPZ(IDEMP)
      NDATA=LX*LY*LZ
      MODE=2
      MNX=MEMAPX(IDEMP)
      MNY=MEMAPY(IDEMP)
      MNZ=MEMAPZ(IDEMP)
      NX=LX
      NY=LY
      NZ=LZ
      DX=DEMAPX(IDEMP)
      DY=DEMAPY(IDEMP)
      DZ=DEMAPZ(IDEMP)
      XL=NX*DX
      YL=NY*DY
      ZL=NZ*DZ
      ALPHA=90.0D0
      BETA=90.0D0
      GAMMA=90.0D0
      MAPC=1
      MAPR=2
      MAPS=3
  !     RMXEMAP(MXNEMP)  - Maximum density
  !     RMNEMAP(MXNEMP)  - Minimum density
  !     AREMAP(MXNEMP)   - SUM of RHO
  !     RREMAP(MXNEMP)   - SUM of RHO*RHO
  !     ADEMAP(MXNEMP)   - SUM of DRHO
  !     DDEMAP(MXNEMP)   - SUM of DRHO*DRHO
      AMIN=RMNEMAP(IDEMP)
      AMAX=RMXEMAP(IDEMP)
      AMEAN=AREMAP(IDEMP)/NDATA
      ARMS=SQRT(RREMAP(IDEMP)/NDATA-AMEAN*AMEAN)
      ISPG=0
      NSYMBT=0
      if(ismrc)then
        EXTRA25=0.0D0
        X0=MNX*DX
        Y0=MNY*DY
        Z0=MNZ*DZ
        EXTRA25=0.0D0
      else
        LSKFLG=0
        SKWMAT=0.0d0
        SKWMAT(1)=1.0D0
        SKWMAT(5)=1.0D0
        SKWMAT(9)=1.0D0
        SKWTRN=0.0d0
        EXTRA=0.0D0
      endif
      MAPLABLE='EMAP '
      MACHST='ALL '
      NNOTE=3
      NOTES=""
      NOTES(1)=" This map is created with the emap module "
      NOTES(2)=" Report questions to Dr. Xiongwu Wu  "
      NOTES(3)="             Email: wuxw@nhlbi.nih.gov "
      WRITE(UNIT)LX,LY,LZ               !1,2,3
      WRITE(UNIT)MODE                   ! 4
      WRITE(UNIT)MNX,MNY,MNZ            ! 5,6,7
      WRITE(UNIT)NX,NY,NZ               ! 8,9,10
      WRITE(UNIT)XL,YL,ZL               ! 11,12,13
      WRITE(UNIT)ALPHA,BETA,GAMMA       ! 14,15,16
      WRITE(UNIT)MAPC,MAPR,MAPS         ! 17,18,19
      WRITE(UNIT)AMIN,AMAX,AMEAN        ! 20,21,22
      WRITE(UNIT)ISPG,NSYMBT            ! 23,24
      if(ismrc)then
        WRITE(UNIT)EXTRA25              ! 25-49
        WRITE(UNIT)X0,Y0,Z0             ! 50-52
      else
        WRITE(UNIT)LSKFLG               ! 25
        WRITE(UNIT)SKWMAT               ! 26-34
        WRITE(UNIT)SKWTRN               ! 35-37
        WRITE(UNIT)EXTRA                ! 38-52
      endif
      !INQUIRE(UNIT=UNIT, POS=I)
      WRITE(UNIT)MAPLABLE               ! 53
      WRITE(UNIT)MACHST                 ! 54
      WRITE(UNIT)ARMS                   ! 55
      WRITE(UNIT)NNOTE                  ! 56
      WRITE(UNIT)NOTES                  ! 57-256
! write data
      IF(MODE==0)THEN
        DO I=1,NDATA
          BDATAI=RHO(I)
          WRITE(UNIT)BDATAI
        ENDDO
      ELSE IF(MODE==1)THEN
        DO I=1,NDATA
          IDATAI=RHO(I)
          WRITE(UNIT)IDATAI
        ENDDO
      ELSE IF(MODE==2)THEN
        DO I=1,NDATA
          RDATAI=RHO(I)
          WRITE(UNIT)RDATAI
        ENDDO
      ELSE IF(MODE==3)THEN
        DO I=1,NDATA
          CDATAI=RHO(I)
          WRITE(UNIT)CDATAI
        ENDDO
      ELSE IF(MODE==4)THEN
        DO I=1,NDATA
          CDATAI=RHO(I)
          WRITE(UNIT)CDATAI
        ENDDO
      ELSE IF(MODE==5)THEN
        DO I=1,NDATA
          BDATAI=RHO(I)
          WRITE(UNIT)BDATAI
        ENDDO
      ENDIF
      CLOSE(UNIT)
! print map information
      IF(PRNLEV > 3)THEN
      WRITE(OUTU,1000)EMAPID(IDEMP), UNIT  !map name to be written
      WRITE(OUTU,1010)LX,LY,LZ               !1,2,3
      WRITE(OUTU,1020)MODE                   ! 4
      WRITE(OUTU,1030)MNX,MNY,MNZ            ! 5,6,7
      WRITE(OUTU,1040)NX,NY,NZ               ! 8,9,10
      WRITE(OUTU,1050)XL,YL,ZL               ! 11,12,13
      WRITE(OUTU,1060)ALPHA,BETA,GAMMA       ! 14,15,16
      WRITE(OUTU,1070)MAPC,MAPR,MAPS         ! 17,18,19
      WRITE(OUTU,1080)AMIN,AMAX,AMEAN,ARMS       ! 20,21,22,55
      WRITE(OUTU,1090)ISPG,NSYMBT     ! 23,24
      if(ismrc)then
        WRITE(OUTU,1102)EXTRA25                  ! 38-52
        WRITE(OUTU,1105)x0,y0,z0                  ! 38-52
      else
        WRITE(OUTU,1100)LSKFLG,NNOTE     ! 25,56
        WRITE(OUTU,1110)SKWMAT                 ! 26-34
        WRITE(OUTU,1120)SKWTRN                 ! 35-37
        WRITE(OUTU,1130)EXTRA                  ! 38-52
      endif
      WRITE(OUTU,1140)MAPLABLE               ! 53
      WRITE(OUTU,1150)MACHST                 ! 54
      WRITE(OUTU,1160)(I,NOTES(I),I=1,NNOTE)   ! 57-256
      WRITE(OUTU,1210)NDATA                  ! DATA NUMBER
      ENDIF
1000  FORMAT(" map object: ",A," is written to unit: ",I4)
1010  FORMAT(" LX, LY, LZ              = ",3I8)
1020  FORMAT(" MODE                    = ",I8)
1030  FORMAT(" MX, MY, MZ              = ",3I8)
1040  FORMAT(" NX, NY, NZ              = ",3I8)
1050  FORMAT(" XL, YL, ZL              = ",3F8.2)
1060  FORMAT(" ALPHA,BETA,GAMMA        = ",3F8.2)
1070  FORMAT(" MAPC, MAPR, MAPS        = ",3I8)
1080  FORMAT(" MIN,MAX,MEAN,STD        = ",4E12.4)
1090  FORMAT(" ISPG,NSYMBT= ",2I8)
1102  FORMAT(" EXTRA                   = ",5F8.2)
1105  FORMAT(" X0,Y0,Z0                = ",3F8.2)
1100  FORMAT(" LSKFLG,NNOTE= ",2I8)
1110  FORMAT(" SKWMAT                  = ",3F8.2)
1120  FORMAT(" SKWTRN                  = ",3F8.2)
1130  FORMAT(" EXTRA                   = ",5F8.2)
1140  FORMAT(" MAPLABLE                = ",A)
1150  FORMAT(" MACHST                  = ",A)
1160  FORMAT(" NOTES ",I2,": ",A80)
1210  FORMAT(" DATA POINT NUMBER       = ",I8)
      RETURN
      END SUBROUTINE WRTCCP4

      SUBROUTINE RDEMAP(FNAME,UNITMAP,IDEMP,FMAP)
!-----------------------------------------------------------------------
!  read in map information and its density array
!                    By Xiongwu Wu, wuxw@nhlbi.nih.gov
!                        
  use memory
  use number
    !
  implicit none
    INTEGER IDEMP,UNITMAP
    CHARACTER(len=*) FNAME
    CHARACTER(len=*) FMAP
    type(emap_geom) :: geom

!
      INTEGER I,FILLEN,UNIT
!
      UNIT=99
  ! remove any embedded double quotes from @param substitution
      FILLEN = LEN(FNAME)
      DO I=1,FILLEN
         IF (FNAME(I:I) == '"') THEN
            FNAME(I:) = FNAME(I+1:)
         ENDIF
      ENDDO
      FILLEN = LEN(FNAME)
!
      IF(FILLEN==0)THEN
         IF(UNITMAP>0)THEN
           UNIT=UNITMAP
           IF((INDEX(FMAP,'CCP4')>0) .or.(INDEX(FMAP,'MAP')>0))THEN
             CALL READCCP4(UNIT,IDEMP,.FALSE.)      
           ELSE IF(INDEX(FMAP,'MRC')>0)THEN
             CALL READCCP4(UNIT,IDEMP,.TRUE.)      
           ELSE IF(INDEX(FMAP,'PDB')>0)THEN
             CALL PDB2MAP(UNIT,IDEMP)   
             RETURN
           ELSE
             CALL WRNDIE(3,'<RDEMAP>', &
          'Error: Map format is not supported or not specified!')
           ENDIF
         ELSE
          ! map will be created from input coordinates based on rigid mask
           CALL WRNDIE(3,'<RDEMAP>', &
            'Error: Map file not opened or specified!')
           RETURN  
         ENDIF
      ELSE IF(INDEX(FNAME,'.pdb')>0.or.INDEX(FNAME,'.PDB')>0)THEN
        OPEN(UNIT,FILE=FNAME,ACCESS='SEQUENTIAL',STATUS='OLD')
        CALL PDB2MAP(UNIT,IDEMP)   
        RETURN
      ELSE IF(INDEX(FNAME,'.ccp4')>0.or.INDEX(FNAME,'.CCP4')>0 .or.  &
         INDEX(FNAME,'.map')>0.or.INDEX(FNAME,'.MAP')>0)THEN
 !
#if KEY_PATHSCALE==0
         OPEN(UNIT,FILE=FNAME,ACCESS='STREAM',FORM='UNFORMATTED',STATUS='OLD')
         CALL READCCP4(UNIT,IDEMP,.FALSE.)      
#else
         OPEN(UNIT,FILE=FNAME,ACCESS='DIRECT',FORM='UNFORMATTED',STATUS='OLD',RECL=1)
         CALL READCCP4(UNIT,IDEMP,.FALSE.)      
#endif
      ELSE IF((INDEX(FNAME,'.mrc')>0) .or.(INDEX(FNAME,'.MRC')>0) )THEN
 !
#if KEY_PATHSCALE==0
         OPEN(UNIT,FILE=FNAME,ACCESS='STREAM',FORM='UNFORMATTED',STATUS='OLD')
         CALL READCCP4(UNIT,IDEMP,.TRUE.)      
#else
         OPEN(UNIT,FILE=FNAME,ACCESS='DIRECT',FORM='UNFORMATTED',STATUS='OLD',RECL=1)
         CALL READCCP4(UNIT,IDEMP,.TRUE.)      
#endif
      ELSE
        CALL WRNDIE(3,'<RDEMAP>', &
          'Error: map format is not supported:'//fname)
        IF(PRNLEV>2)write(outu,*)' Constraint map format is not supported:',fname
        stop
      ENDIF
      RETURN
      END SUBROUTINE RDEMAP

      SUBROUTINE READCCP4(UNIT,IDEMP,ISMRC)
!_________________________________________________________________
!  read in CCP4 map information and its density array
!                    By Xiongwu Wu, wuxw@nhlbi.nih.gov
!_________________________________________________________________
  use memory
  use number
#if KEY_PARALLEL==1
  use parallel       
#endif
  use machutil,only:eclock
  implicit none
     INTEGER IDEMP
     LOGICAL ISMRC 
!
      INTEGER*4 LX,LY,LZ,MODE,MNX,MNY,MNZ,NX,NY,NZ
      real(chm_real4) XL,YL,ZL,ALPHA,BETA,GAMMA
      real(chm_real4) x0,y0,z0
      INTEGER*4 MAPC,MAPR,MAPS
      real(chm_real4) AMAX,AMIN,AMEAN,ARMS
      INTEGER*4 ISPG,NSYMBT,LSKFLG,NNOTE
      real(chm_real4) SKWMAT(9),SKWTRN(3),EXTRA(15),EXTRA25(25)
      CHARACTER(len=4) MAPLABLE,MACHST
      CHARACTER(len=80) NOTES(10)
      INTEGER*1 BDATAI
      INTEGER*2 IDATAI
      real(chm_real4) RDATAI
      COMPLEX(KIND=4) CDATAI
!
      INTEGER I,UNIT,FILLEN,NDATA,alloc_err
#if KEY_PARALLEL==1
    INTEGER*4 IARRAY(30)
    real(chm_real) RARRAY(30),TIMMER
#endif 

!
      !
#if KEY_PARALLEL==1
      IF(MYNOD.EQ.0) THEN
#endif 
      READ(UNIT)LX,LY,LZ               !1,2,3
      READ(UNIT)MODE                   ! 4
      READ(UNIT)MNX,MNY,MNZ            ! 5,6,7
      READ(UNIT)NX,NY,NZ               ! 8,9,10
      READ(UNIT)XL,YL,ZL               ! 11,12,13
      READ(UNIT)ALPHA,BETA,GAMMA       ! 14,15,16
      READ(UNIT)MAPC,MAPR,MAPS         ! 17,18,19
      READ(UNIT)AMIN,AMAX,AMEAN        ! 20,21,22
      READ(UNIT)ISPG,NSYMBT            ! 23,24
      if(ismrc)then
        READ(UNIT)EXTRA25              ! 25-49
        READ(UNIT)x0,y0,z0             ! 50-52
     else
        READ(UNIT)LSKFLG                 ! 25
        READ(UNIT)SKWMAT                 ! 26-34
        READ(UNIT)SKWTRN                 ! 35-37
        READ(UNIT)EXTRA                  ! 38-52
      endif
      READ(UNIT)MAPLABLE               ! 53
      READ(UNIT)MACHST                 ! 54
      READ(UNIT)ARMS                   ! 55
      READ(UNIT)NNOTE                  ! 56
      READ(UNIT)NOTES                  ! 57-256
#if KEY_PARALLEL==1
      ENDIF
      IF(NUMNOD.GT.1) THEN
        CALL PSYNC()
        TIMMER=ECLOCK()
        IARRAY(1)=LX
        IARRAY(2)=LY
        IARRAY(3)=LZ
        IARRAY(4)=MNX
        IARRAY(5)=MNY
        IARRAY(6)=MNZ
        IARRAY(7)=NX
        IARRAY(8)=NY
        IARRAY(9)=NZ
        IARRAY(10)=MODE
        RARRAY(1)=XL
        RARRAY(2)=YL
        RARRAY(3)=ZL
        CALL PSND4(IARRAY,10)
        CALL PSND8(RARRAY,3)
        LX=IARRAY(1)
        LY=IARRAY(2)
        LZ=IARRAY(3)
        MNX=IARRAY(4)
        MNY=IARRAY(5)
        MNZ=IARRAY(6)
        NX=IARRAY(7)
        NY=IARRAY(8)
        NZ=IARRAY(9)
        MODE=IARRAY(10)
        XL=RARRAY(1)
        YL=RARRAY(2)
        ZL=RARRAY(3)
        TMERI(TIMGCOMM)=TMERI(TIMGCOMM)+ECLOCK()-TIMMER
        TIMMER=ECLOCK()
      ENDIF
#endif 
! read data
      NDATA=LX*LY*LZ
      call chmalloc('emapsubs.src','RDEMAP','empgrd.RHO',NDATA, &
         cr4p=empgrd(idemp)%RHO)
      call chmalloc('emapsubs.src','RDEMAP','empgrd.CORE',NDATA, &
         intgp=empgrd(idemp)%CORE)
      call chmalloc('emapsubs.src','RDEMAP','empgrd.ddrho',NDATA, &
         cr4p=empgrd(idemp)%DDRho)

#if KEY_PARALLEL==1
      IF(MYNOD.EQ.0) THEN
#endif 
      IF(MODE==0)THEN
        DO I=1,NDATA
            READ(UNIT)BDATAI
            EMPGRD(IDEMP)%RHO(I)=BDATAI
        ENDDO
      ELSE IF(MODE==1)THEN
         DO I=1,NDATA
          READ(UNIT)IDATAI
          EMPGRD(IDEMP)%RHO(I)=IDATAI
        ENDDO
      ELSE IF(MODE==2)THEN
         DO I=1,NDATA
          READ(UNIT)RDATAI
          EMPGRD(IDEMP)%RHO(I)=RDATAI
        ENDDO
      ELSE IF(MODE==3)THEN
        DO I=1,NDATA
          READ(UNIT)CDATAI
          EMPGRD(IDEMP)%RHO(I)=CDATAI
        ENDDO
      ELSE IF(MODE==4)THEN
        DO I=1,NDATA
          READ(UNIT)CDATAI
          EMPGRD(IDEMP)%RHO(I)=CDATAI
        ENDDO
      ELSE IF(MODE==5)THEN
         DO I=1,NDATA
          READ(UNIT)BDATAI
          EMPGRD(IDEMP)%RHO(I)=BDATAI
        ENDDO
      ENDIF
#if KEY_PARALLEL==1
      ENDIF
      IF(NUMNOD.GT.1) THEN
        IF(MODE==1)THEN
          CALL PSND4(EMPGRD(IDEMP)%RHO,NDATA)
        ELSE IF(MODE==2)THEN
          CALL PSND4(EMPGRD(IDEMP)%RHO,NDATA)
        ELSE IF(MODE==3)THEN
          CALL PSNDC(EMPGRD(IDEMP)%RHO,NDATA)
        ELSE IF(MODE==4)THEN
          CALL PSND4r(EMPGRD(IDEMP)%RHO,NDATA)
        ELSE
          CALL WRNDIE(0,'<REACCP4>','Parallel version does not support this map mode!')
          RETURN
        ENDIF
      ENDIF
#endif 
      LEMAPX(IDEMP)=LX
      LEMAPY(IDEMP)=LY
      LEMAPZ(IDEMP)=LZ
      MEMAPX(IDEMP)=MNX
      MEMAPY(IDEMP)=MNY
      MEMAPZ(IDEMP)=MNZ
      !NEMAPX(IDEMP)=NX
      !NEMAPY(IDEMP)=NY
      !NEMAPZ(IDEMP)=NZ
      DEMAPX(IDEMP)=XL/NX
      DEMAPY(IDEMP)=YL/NY
      DEMAPZ(IDEMP)=ZL/NZ
!      MAPC=1
!      MAPR=2
!C      MAPS=3
!      ISPG=0
!      NSYMBT=0
!      LSKFLG=0
!      MAPLABLE='MAP '
!      MACHST='ALL '
!      NNOTE=3
!      NOTES(1)="          "
!     &    //" This map is created with the emap module of charmm "
!      NOTES(2)="          "
!     &    //" Report questions to Dr. Xiongwu Wu  "
!      NOTES(3)="          "
!     &    //"             Email: wuxw@nhlbi.nih.gov "
      CEMAPX(IDEMP)=(MNX+LX/2.0D0)*DEMAPX(IDEMP)
      CEMAPY(IDEMP)=(MNY+LY/2.0D0)*DEMAPY(IDEMP)
      CEMAPZ(IDEMP)=(MNZ+LZ/2.0D0)*DEMAPZ(IDEMP)
!     statistics
      CALL EMAPDDR(IDEMP)
      CALL EMAPCORE(IDEMP,EMRCUT,EMICORE)
      CALL EMAPSTAT(IDEMP)
! print map information
      IF(PRNLEV > 3 )THEN
      WRITE(OUTU,1000)EMAPID(IDEMP), UNIT  !map name to be written
      WRITE(OUTU,1010)LX,LY,LZ               !1,2,3
      WRITE(OUTU,1020)MODE                   ! 4
      WRITE(OUTU,1030)MNX,MNY,MNZ            ! 5,6,7
      WRITE(OUTU,1040)NX,NY,NZ               ! 8,9,10
      WRITE(OUTU,1050)XL,YL,ZL               ! 11,12,13
      WRITE(OUTU,1060)ALPHA,BETA,GAMMA       ! 14,15,16
      WRITE(OUTU,1070)MAPC,MAPR,MAPS         ! 17,18,19
      WRITE(OUTU,1080)AMIN,AMAX,AMEAN,ARMS       ! 20,21,22,55
      WRITE(OUTU,1090)ISPG,NSYMBT     ! 23,24
      if(ismrc)then
        WRITE(OUTU,1102)EXTRA25                  ! 38-52
        WRITE(OUTU,1105)x0,y0,z0                  ! 38-52
      else
        WRITE(OUTU,1100)LSKFLG,NNOTE     ! 25,56
        WRITE(OUTU,1110)SKWMAT                 ! 26-34
        WRITE(OUTU,1120)SKWTRN                 ! 35-37
        WRITE(OUTU,1130)EXTRA                  ! 38-52
      endif
      CALL ASCIIMASK(MAPLABLE)
      WRITE(OUTU,1140)MAPLABLE               ! 53
      CALL ASCIIMASK(MACHST)
      WRITE(OUTU,1150)MACHST                 ! 54
      DO I=1,NNOTE
        CALL ASCIIMASK(NOTES(I))
        IF(LEN(trim(NOTES(I)))>0)WRITE(OUTU,1160)I,NOTES(I)   ! 57-256
      ENDDO
      WRITE(OUTU,1210)NDATA                  ! DATA NUMBER
      WRITE(OUTU,1220)IDEMP                    ! DATA NUMBER
      ENDIF
1000  FORMAT(" ------------------EMAP ",A," INPUT from unit: ",I4)
1010  FORMAT(" LX, LY, LZ              = ",3I8)
1020  FORMAT(" MODE                    = ",I8)
1030  FORMAT(" MX, MY, MZ              = ",3I8)
1040  FORMAT(" NX, NY, NZ              = ",3I8)
1050  FORMAT(" XL, YL, ZL              = ",3F8.2)
1060  FORMAT(" ALPHA,BETA,GAMMA        = ",3F8.2)
1070  FORMAT(" MAPC, MAPR, MAPS        = ",3I8)
1080  FORMAT(" MIN,MAX,MEAN,STD        = ",4E12.4)
1090  FORMAT(" ISPG,NSYMBT= ",2I8)
1102  FORMAT(" EXTRA                   = ",5F8.2)
1105  FORMAT(" X0,Y0,Z0                = ",3F8.2)
1100  FORMAT(" LSKFLG,NNOTE= ",2I8)
1110  FORMAT(" SKWMAT                  = ",3F8.2)
1120  FORMAT(" SKWTRN                  = ",3F8.2)
1130  FORMAT(" EXTRA                   = ",5F8.2)
1140  FORMAT(" MAPLABLE                = ",A)
1150  FORMAT(" MACHST                  = ",A)
1160  FORMAT(" NOTES ",I2,": ",A80)
1210  FORMAT(" DATA POINT NUMBER       = ",I8)
1220  FORMAT(" ----------------------- END OF EMAP IMAGE ",I4, &
             "  -------------------------- ")
      CLOSE(UNIT)
      RETURN
      END SUBROUTINE READCCP4


      SUBROUTINE ASCIIMASK(STRING)
      CHARACTER*(*) STRING
      INTEGER I,J,K
      I=1
      DO WHILE(I.LT.LEN(STRING))
        J=ICHAR(STRING(I:I))
        IF(J<=0.or.J>127)THEN
          STRING(I:)=STRING(I+1:)
        ELSE
          I=I+1
        ENDIF
      ENDDO
      RETURN
      END SUBROUTINE ASCIIMASK
      
      SUBROUTINE PDB2MAP(UNIT,IDEMP)
!_________________________________________________________________
!  read in PDB coordinates to create a map object
!                    By Xiongwu Wu, wuxw@nhlbi.nih.gov
!_________________________________________________________________
!
  use memory
  use number
#if KEY_PARALLEL==1
  use parallel       
#endif
  use machutil,only:eclock
      INTEGER UNIT,IDEMP,OUTU
!
      CHARACTER*80 pdblin
      CHARACTER*4 CHAINI,CHAINP,RESIN,ATOMIN,SID,SIDP,RID,RIDP,ATOMNM
      real(chm_real) XIN,YIN,ZIN,AMASSI,RHOI,DG,WIN
      INTEGER NAT,IAT,ISEQ
!
      INTEGER alloc_err
      real(chm_real), allocatable, dimension(:)::tmpx,tmpy,tmpz,tmpm
      INTEGER, allocatable, dimension(:)::itmp
!
#if KEY_PARALLEL==1
      INTEGER*4 IARRAY(10)
      real(chm_real) :: TIMMER
#endif 
!
#if KEY_PARALLEL==1
      IF(MYNOD.EQ.0) THEN
#endif 
      NAT=0
99    READ(UNIT,'(A)',END=120,ERR=120) PDBLIN
      PDBLIN=TRIM(PDBLIN)
      IF(LEN_TRIM(PDBLIN)<44)GOTO 99
      IF (PDBLIN(1:6).NE.'ATOM  '.AND.PDBLIN(1:6).NE.'HETATM') GOTO 99
      NAT=NAT+1
      GOTO 99
120   CONTINUE
#if KEY_PARALLEL==1
      ENDIF
      IF(NUMNOD.GT.1) THEN
        CALL PSYNC()
        TIMMER=ECLOCK()
        IARRAY(1)=NAT
        CALL PSND4(IARRAY,1)
        NAT=IARRAY(1)
        TMERI(TIMGCOMM)=TMERI(TIMGCOMM)+ECLOCK()-TIMMER
        TIMMER=ECLOCK()
      ENDIF
#endif 
      call chmalloc('emapsubs.src','PDB2MAP','tmpx ',NAT,crl=tmpx)
      
      call chmalloc('emapsubs.src','PDB2MAP','tmpy ', &
      NAT,crl=tmpy)
      call chmalloc('emapsubs.src','PDB2MAP','tmpz ', &
      NAT,crl=tmpz)
      call chmalloc('emapsubs.src','PDB2MAP','tmpm ', &
      NAT,crl=tmpm)
      call chmalloc('emapsubs.src','PDB2MAP','itmp ', &
      NAT,intg=itmp)
#if KEY_PARALLEL==1
      IF(MYNOD.EQ.0) THEN
#endif 
      REWIND(UNIT)
      IAT=0
299   READ(UNIT,'(A)',END=320,ERR=320) PDBLIN
      PDBLIN=TRIM(PDBLIN)
      IF(LEN(PDBLIN)<44)GOTO 299
      IF (PDBLIN(1:6).NE.'ATOM  '.AND.PDBLIN(1:6).NE.'HETATM') GOTO 299
      READ(PDBLIN, &
          '(6X,I5,1X,A4,1X,A3,1X,A1,A5,3X,3F8.3)') &
               ISEQ,ATOMIN,RESIN,CHAINI,RID,XIN,YIN,ZIN     
!  determine atom mass
      ATOMNM=ADJUSTL(ATOMIN)
      IF(INDEX(ATOMNM,'CAL')==1)THEN
        AMASSI=40.0D0
      ELSE IF(INDEX(ATOMNM,'CL')==1)THEN
        AMASSI=35.5D0
      ELSE IF(INDEX(ATOMNM,'MG')==1)THEN
        AMASSI=24.3D0
      ELSE IF(INDEX(ATOMNM,'S')==1)THEN
        AMASSI=32.06D0
      ELSE IF(INDEX(ATOMNM,'O')==1)THEN
        AMASSI=16.0D0
      ELSE IF(INDEX(ATOMNM,'N')==1)THEN
        AMASSI=14.01D0
      ELSE IF(INDEX(ATOMNM,'P')==1)THEN
        AMASSI=31.0D0
      ELSE IF(INDEX(ATOMNM,'F')==1)THEN
        AMASSI=19.0D0
      ELSE IF(INDEX(ATOMNM,'C')==1)THEN
        AMASSI=12.01D0
      ELSE IF(INDEX(ATOMNM,'H')==1)THEN
        AMASSI=1.008D0
      ELSE 
        AMASSI=1.008D0
      ENDIF
      IAT=IAT+1
      itmp(IAT)=1
      tmpm(IAT)=AMASSI
      tmpx(IAT)=XIN
      tmpy(IAT)=YIN
      tmpz(IAT)=ZIN
      goto 299
320   CONTINUE
      CLOSE(UNIT)
#if KEY_PARALLEL==1
      ENDIF
      IF(NUMNOD.GT.1) THEN
        CALL PSND4(itmp,NAT)
        CALL PSND8(tmpm,NAT)
        CALL PSND8(tmpx,NAT)
        CALL PSND8(tmpy,NAT)
        CALL PSND8(tmpz,NAT)
      ENDIF
#endif 
!     create map object
      MEMAPX(IDEMP)=-999999
      MEMAPY(IDEMP)=-999999
      MEMAPZ(IDEMP)=-999999
      LEMAPX(IDEMP)=-999999
      LEMAPY(IDEMP)=-999999
      LEMAPZ(IDEMP)=-999999
      call EMAPGEN(NAT,TMPX,TMPY,TMPZ,TMPM,ITMP,IDEMP,EMDX,EMDY,EMDZ,EMRESO)
      
      call chmdealloc('emapsubs.src','PDB2MAP','itmp', &
      NAT,intg=itmp)            
      call chmdealloc('emapsubs.src','PDB2MAP','tmpm', &
      NAT,crl=tmpm)       
      call chmdealloc('emapsubs.src','PDB2MAP','tmpx', &
      NAT,crl=tmpx)     
      call chmdealloc('emapsubs.src','PDB2MAP','tmpy', &
      NAT,crl=tmpy)     
      call chmdealloc('emapsubs.src','PDB2MAP','tmpz', &
      NAT,crl=tmpz)                  
      RETURN
      END SUBROUTINE PDB2MAP

  SUBROUTINE EMAPMINUS(IDEMP,IDRIG)
    !-----------------------------------------------------------------------
    !     This routine substract a rigid domain from a map
    !
  use number

    integer IDEMP,IDEMP1,IDRIG
    !
    IDEMP1=IDEMRIG(IDRIG)
    CALL EMAPADDR(IDEMP,IDEMP1, &
         empgrd(IDEMP)%rho, &
         empgrd(IDEMP)%ddrho, &
         empgrd(IDEMP)%core, &
         empgrd(IDEMP1)%rho, &
         empgrd(IDEMP1)%ddrho, &
         empgrd(IDEMP1)%core, &
         TEMRIG(1,IDRIG),REMRIG(1,IDRIG),-ONE)
    RETURN
  END SUBROUTINE EMAPMINUS



  SUBROUTINE EMAPRESIZE(IDEMP,MX,MY,MZ,LX,LY,LZ,DX,DY,DZ)
    !-----------------------------------------------------------------------
    !   This routine Resize a map object to given boundary and grid
    !
  use memory
    implicit none

    INTEGER IDEMP,MX,MY,MZ,LX,LY,LZ,NDATA
    real(chm_real4),pointer,dimension(:) :: RHOEMPI,DDREMPI
    integer,pointer,dimension(:) :: COREEMPI
    real(chm_real) DX,DY,DZ
    real(chm_real) RHOCUT
    INTEGER ICORE
    !
    NDATA=LX*LY*LZ
    call chmalloc('emapsubs.src','EMAPRESIZE','RHOEMPI',NDATA,cr4p=RHOEMPI)
    call chmalloc('emapsubs.src','EMAPRESIZE','DDREMPI',NDATA,cr4p=DDREMPI)
    call chmalloc('emapsubs.src','EMAPRESIZE','COREEMPI',NDATA,intgp=COREEMPI)
    CALL EMAPCAST(MEMAPX(IDEMP),MEMAPY(IDEMP),MEMAPZ(IDEMP), &
         LEMAPX(IDEMP),LEMAPY(IDEMP),LEMAPZ(IDEMP), &
         DEMAPX(IDEMP),DEMAPY(IDEMP),DEMAPZ(IDEMP), &
         empgrd(IDEMP)%rho, &
         MX,MY,MZ,LX,LY,LZ,DX,DY,DZ,RHOEMPI)
    NDATA=LEMAPX(IDEMP)*LEMAPY(IDEMP)*LEMAPZ(IDEMP)
    call chmdealloc('emapsubs.src','EMAPRESIZE','empgrd(IDEMP)%rho',NDATA, &
         cr4p=empgrd(IDEMP)%rho)
    call chmdealloc('emapsubs.src','EMAPRESIZE','empgrd(IDEMP)%ddrho',NDATA, &
         cr4p=empgrd(IDEMP)%ddrho)
    call chmdealloc('emapsubs.src','EMAPRESIZE','empgrd%CORE',NDATA, &
         intgp=empgrd(IDEMP)%core)
    !--- swap pointers ---
    empgrd(IDEMP)%rho   => RHOEMPI
    empgrd(IDEMP)%ddrho => DDREMPI
    empgrd(IDEMP)%core  => COREEMPI
    MEMAPX(IDEMP)=MX
    MEMAPY(IDEMP)=MY
    MEMAPZ(IDEMP)=MZ
    LEMAPX(IDEMP)=LX
    LEMAPY(IDEMP)=LY
    LEMAPZ(IDEMP)=LZ
    DEMAPX(IDEMP)=DX
    DEMAPY(IDEMP)=DY
    DEMAPZ(IDEMP)=DZ
    RHOCUT=EMRCUT
    ICORE=EMICORE
    CALL EMAPDDR(IDEMP)
    CALL EMAPCORE(IDEMP,RHOCUT,ICORE)
    CALL EMAPSTAT(IDEMP)
    RETURN
  END SUBROUTINE EMAPRESIZE

  SUBROUTINE EMAPCAST(MX,MY,MZ,LX,LY,LZ,DX,DY,DZ,RHO, &
       MX1,MY1,MZ1,LX1,LY1,LZ1,DX1,DY1,DZ1,RHO1)
    !-----------------------------------------------------------------------
    !     This routine recalculate the density distribution at new
    !     boundary and grid
    !
  use number
    INTEGER MX,MY,MZ,LX,LY,LZ,IXYZ
    INTEGER MX1,MY1,MZ1,LX1,LY1,LZ1,IXYZ1
    INTEGER I,J,K,I1,J1,K1
    real(chm_real) DX,DY,DZ,DX1,DY1,DZ1
    real(chm_real4) RHO(*),RHO1(*)
    !
    DO I1=1,LX1
       I=NINT((I1+MX1-1)*DX1/DX)-MX+1
       DO J1=1,LY1
          J=NINT((J1+MY1-1)*DY1/DY)-MY+1
          DO K1=1,LZ1
             K=NINT((K1+MZ1-1)*DZ1/DZ)-MZ+1
             IXYZ=I+LX*(J-1+LY*(K-1))
             IXYZ1=I1+LX1*(J1-1+LY1*(K1-1))
             IF(I >= MX.AND.(I-MX) < LX.AND. &
                  J >= MY.AND.(J-MY) < LY.AND. &
                  K >= MZ.AND.(K-MZ) < LZ)THEN
                RHO1(IXYZ1)=RHO(IXYZ)
             ELSE
                RHO1(IXYZ1)=ZERO
             ENDIF
          END DO
       END DO
    END DO
    RETURN
  END SUBROUTINE EMAPCAST


  SUBROUTINE EMAPTRN(IDRIG,XDIR,YDIR,ZDIR)
    !-----------------------------------------------------------------------
    !     This routine  update the position of the rigid segment
    !
  use number
    !
    real(chm_real) XDIR,YDIR,ZDIR
    real(chm_real) T1,T2,T3
    INTEGER IDRIG,IDEMP
    !
    !
    TEMRIG(1,IDRIG)=TEMRIG(1,IDRIG)+XDIR
    TEMRIG(2,IDRIG)=TEMRIG(2,IDRIG)+YDIR
    TEMRIG(3,IDRIG)=TEMRIG(3,IDRIG)+ZDIR
    IF(LPBC)THEN
      IDEMP=IDEMRIG(IDRIG)
      T1=TEMRIG(1,IDRIG)+CEMAPX(IDEMP)
      T2=TEMRIG(2,IDRIG)+CEMAPY(IDEMP)
      T3=TEMRIG(3,IDRIG)+CEMAPZ(IDEMP)
      TEMRIG(1,IDRIG)=TEMRIG(1,IDRIG)-ANINT(T1/PBCX)*PBCX
      TEMRIG(2,IDRIG)=TEMRIG(2,IDRIG)-ANINT(T2/PBCY)*PBCY
      TEMRIG(3,IDRIG)=TEMRIG(3,IDRIG)-ANINT(T3/PBCZ)*PBCZ
    ENDIF
    RETURN
  END SUBROUTINE EMAPTRN

  SUBROUTINE EMAPTRNIN(IDRIG,XDIR,YDIR,ZDIR)
    !-----------------------------------------------------------------------
    !     This routine  move a rigid segment within a rectangular boundary
    !
  use number
  !use stream
    !
    real(chm_real) XDIR,YDIR,ZDIR
    real(chm_real) T1,T2,T3,DD
    INTEGER IDRIG,IDEMP
    !
    IDEMP=IDEMRIG(IDRIG)
    T1=TEMRIG(1,IDRIG)+CEMAPX(IDEMP)
    T2=TEMRIG(2,IDRIG)+CEMAPY(IDEMP)
    T3=TEMRIG(3,IDRIG)+CEMAPZ(IDEMP)
    !
    !write(OUTU,'(A,I4,8E10.6)')'EMAPTRNIN1: ',IDRIG,T1,T2,T3,XDIR,YDIR,ZDIR
    IF(XDIR>ZERO)THEN
      DD=UPEMAP(1,IDEMP)-T1
      IF(DD>ZERO)XDIR=DD-ABS(DD-XDIR)
    ELSE
      DD=LOWEMAP(1,IDEMP)-T1
      IF(DD<ZERO)XDIR=DD+ABS(DD-XDIR)
    ENDIF
    IF(YDIR>ZERO)THEN
      DD=UPEMAP(2,IDEMP)-T2
      IF(DD>ZERO)YDIR=DD-ABS(DD-YDIR)
    ELSE
      DD=LOWEMAP(2,IDEMP)-T2
      IF(DD<ZERO)YDIR=DD+ABS(DD-YDIR)
     ENDIF
    IF(ZDIR>ZERO)THEN
      DD=UPEMAP(3,IDEMP)-T3
      IF(DD>ZERO)ZDIR=DD-ABS(DD-ZDIR)
    ELSE
      DD=LOWEMAP(3,IDEMP)-T3
      IF(DD<ZERO)ZDIR=DD+ABS(DD-ZDIR)
    ENDIF
    !write(OUTU,'(A,I4,8E10.6)')'EMAPTRNIN2: ',IDRIG,XDIR,YDIR,ZDIR,DD
    !
    TEMRIG(1,IDRIG)=TEMRIG(1,IDRIG)+XDIR
    TEMRIG(2,IDRIG)=TEMRIG(2,IDRIG)+YDIR
    TEMRIG(3,IDRIG)=TEMRIG(3,IDRIG)+ZDIR
    IF(LPBC)THEN
      T1=TEMRIG(1,IDRIG)+CEMAPX(IDEMP)
      T2=TEMRIG(2,IDRIG)+CEMAPY(IDEMP)
      T3=TEMRIG(3,IDRIG)+CEMAPZ(IDEMP)
      TEMRIG(1,IDRIG)=TEMRIG(1,IDRIG)-ANINT(T1/PBCX)*PBCX
      TEMRIG(2,IDRIG)=TEMRIG(2,IDRIG)-ANINT(T2/PBCY)*PBCY
      TEMRIG(3,IDRIG)=TEMRIG(3,IDRIG)-ANINT(T3/PBCZ)*PBCZ
    ENDIF
    RETURN
  END SUBROUTINE EMAPTRNIN

  SUBROUTINE EMAPSAVE(IDRIG)
    !-----------------------------------------------------------------------
    !     This routine  save the position of the rigid segment
    !
    !
    INTEGER IDRIG
    !
    TEMRIG0(1,IDRIG)=TEMRIG(1,IDRIG)
    TEMRIG0(2,IDRIG)=TEMRIG(2,IDRIG)
    TEMRIG0(3,IDRIG)=TEMRIG(3,IDRIG)
    REMRIG0(1,IDRIG)=REMRIG(1,IDRIG)
    REMRIG0(2,IDRIG)=REMRIG(2,IDRIG)
    REMRIG0(3,IDRIG)=REMRIG(3,IDRIG)
    REMRIG0(4,IDRIG)=REMRIG(4,IDRIG)
    REMRIG0(5,IDRIG)=REMRIG(5,IDRIG)
    REMRIG0(6,IDRIG)=REMRIG(6,IDRIG)
    REMRIG0(7,IDRIG)=REMRIG(7,IDRIG)
    REMRIG0(8,IDRIG)=REMRIG(8,IDRIG)
    REMRIG0(9,IDRIG)=REMRIG(9,IDRIG)
    RETURN
  END SUBROUTINE EMAPSAVE

  SUBROUTINE EMAPRESTORE(IDRIG)
    !-----------------------------------------------------------------------
    !     This routine  restore the position of the rigid segment
    !
    !
    INTEGER IDRIG
    !
    TEMRIG(1,IDRIG)=TEMRIG0(1,IDRIG)
    TEMRIG(2,IDRIG)=TEMRIG0(2,IDRIG)
    TEMRIG(3,IDRIG)=TEMRIG0(3,IDRIG)
    REMRIG(1,IDRIG)=REMRIG0(1,IDRIG)
    REMRIG(2,IDRIG)=REMRIG0(2,IDRIG)
    REMRIG(3,IDRIG)=REMRIG0(3,IDRIG)
    REMRIG(4,IDRIG)=REMRIG0(4,IDRIG)
    REMRIG(5,IDRIG)=REMRIG0(5,IDRIG)
    REMRIG(6,IDRIG)=REMRIG0(6,IDRIG)
    REMRIG(7,IDRIG)=REMRIG0(7,IDRIG)
    REMRIG(8,IDRIG)=REMRIG0(8,IDRIG)
    REMRIG(9,IDRIG)=REMRIG0(9,IDRIG)
    RETURN
  END SUBROUTINE EMAPRESTORE

  SUBROUTINE EMAPTRAJ(IDRIG,NST,NSR,ENG,ENGM,INOUT)
    !-----------------------------------------------------------------------
    !   This routine  input/output one frame of a rigid domain trajectory
    !
    !
    INTEGER IDRIG,NST,NSR,INOUT
    real(chm_real) ENG,ENGM
    !
    IF(INOUT > 0)THEN
       READ(UEMPTRJ,1100,END=100)NST,NSR, &
            TEMRIG(1,IDRIG),TEMRIG(2,IDRIG),TEMRIG(3,IDRIG), &
            REMRIG(1,IDRIG),REMRIG(2,IDRIG),REMRIG(3,IDRIG), &
            REMRIG(4,IDRIG),REMRIG(5,IDRIG),REMRIG(6,IDRIG), &
            REMRIG(7,IDRIG),REMRIG(8,IDRIG),REMRIG(9,IDRIG), &
            ENG,ENGM
    ELSE
       WRITE(UEMPTRJ,1100)NST,NSR, &
            TEMRIG(1,IDRIG),TEMRIG(2,IDRIG),TEMRIG(3,IDRIG), &
            REMRIG(1,IDRIG),REMRIG(2,IDRIG),REMRIG(3,IDRIG), &
            REMRIG(4,IDRIG),REMRIG(5,IDRIG),REMRIG(6,IDRIG), &
            REMRIG(7,IDRIG),REMRIG(8,IDRIG),REMRIG(9,IDRIG), &
            ENG,ENGM
    ENDIF
1100 FORMAT(2I6,12F10.2,F12.4,1x,F12.4)
    RETURN
100 NST=-1   
    RETURN
  END SUBROUTINE EMAPTRAJ

  SUBROUTINE EMAPRIGINI(IDRIG)
    !-----------------------------------------------------------------------
    !   This routine  initiate a rigid segment
    !
  use number
    !
    INTEGER IDRIG,IDRIGJ
    TEMRIG(1,IDRIG)=ZERO
    TEMRIG(2,IDRIG)=ZERO
    TEMRIG(3,IDRIG)=ZERO
    REMRIG(1,IDRIG)=ONE
    REMRIG(2,IDRIG)=ZERO
    REMRIG(3,IDRIG)=ZERO
    REMRIG(4,IDRIG)=ZERO
    REMRIG(5,IDRIG)=ONE
    REMRIG(6,IDRIG)=ZERO
    REMRIG(7,IDRIG)=ZERO
    REMRIG(8,IDRIG)=ZERO
    REMRIG(9,IDRIG)=ONE
    RETURN
  END SUBROUTINE EMAPRIGINI

  SUBROUTINE EMAPRIGDUP(IDRIG,IDRIGJ)
    !-----------------------------------------------------------------------
    !     This routine  duplicate a rigid segment
    !
  use number
    !
    INTEGER IDRIG,IDRIGJ
    !
    IDEMRIG(IDRIGJ)=IDEMRIG(IDRIG)
    TEMRIG(1,IDRIGJ)=TEMRIG(1,IDRIG)
    TEMRIG(2,IDRIGJ)=TEMRIG(2,IDRIG)
    TEMRIG(3,IDRIGJ)=TEMRIG(3,IDRIG)
    REMRIG(1,IDRIGJ)=REMRIG(1,IDRIG)
    REMRIG(2,IDRIGJ)=REMRIG(2,IDRIG)
    REMRIG(3,IDRIGJ)=REMRIG(3,IDRIG)
    REMRIG(4,IDRIGJ)=REMRIG(4,IDRIG)
    REMRIG(5,IDRIGJ)=REMRIG(5,IDRIG)
    REMRIG(6,IDRIGJ)=REMRIG(6,IDRIG)
    REMRIG(7,IDRIGJ)=REMRIG(7,IDRIG)
    REMRIG(8,IDRIGJ)=REMRIG(8,IDRIG)
    REMRIG(9,IDRIGJ)=REMRIG(9,IDRIG)
    RETURN
  END SUBROUTINE EMAPRIGDUP

  SUBROUTINE EMAPRIGCPY(IDRIG,IDRIGJ)
    !-----------------------------------------------------------------------
    !     This routine  copy the position of the rigid segment
    !
    !
    INTEGER IDRIG,IDRIGJ
    !
    TEMRIG(1,IDRIGJ)=TEMRIG(1,IDRIG)
    TEMRIG(2,IDRIGJ)=TEMRIG(2,IDRIG)
    TEMRIG(3,IDRIGJ)=TEMRIG(3,IDRIG)
    REMRIG(1,IDRIGJ)=REMRIG(1,IDRIG)
    REMRIG(2,IDRIGJ)=REMRIG(2,IDRIG)
    REMRIG(3,IDRIGJ)=REMRIG(3,IDRIG)
    REMRIG(4,IDRIGJ)=REMRIG(4,IDRIG)
    REMRIG(5,IDRIGJ)=REMRIG(5,IDRIG)
    REMRIG(6,IDRIGJ)=REMRIG(6,IDRIG)
    REMRIG(7,IDRIGJ)=REMRIG(7,IDRIG)
    REMRIG(8,IDRIGJ)=REMRIG(8,IDRIG)
    REMRIG(9,IDRIGJ)=REMRIG(9,IDRIG)
    RETURN
  END SUBROUTINE EMAPRIGCPY

  SUBROUTINE EMAPRPBC(IDRIG,IDRIGJ)
    !-----------------------------------------------------------------------
    !   This routine find the transform to IDRIG related to 
    !       the position and orientation of IDRIGJ MAP
    !    RN=INV(MJ)*(M*(R-Rc)+Rc+T-T1-RJc)+RJc
    !    Compare:
    !    RN=MN*(R-Rc)+Rc+TN
    !    We have: MN=INV(MJ)*M 
    !             TN=INV(MJ)*(T+Rc-TJ-RJc)+RJc-Rc
    !
    !
    INTEGER IDRIG,IDRIGJ,IDEMP,IDEMPJ
    real(chm_real) DC1,DC2,DC3,T1,T2,T3
    !
    IDEMP=IDEMRIG(IDRIG)
    IDEMPJ=IDEMRIG(IDRIGJ)
    DC1=CEMAPX(IDEMP)-CEMAPX(IDEMPJ)
    DC2=CEMAPY(IDEMP)-CEMAPY(IDEMPJ)
    DC3=CEMAPZ(IDEMP)-CEMAPZ(IDEMPJ)
    T1=TEMRIG(1,IDRIG)-TEMRIG(1,IDRIGJ)+DC1
    T2=TEMRIG(2,IDRIG)-TEMRIG(2,IDRIGJ)+DC2
    T3=TEMRIG(3,IDRIG)-TEMRIG(3,IDRIGJ)+DC3
    TEMRIG(1,IDRIG)=TEMRIG(1,IDRIG)-ANINT(T1/PBCX)*PBCX
    TEMRIG(2,IDRIG)=TEMRIG(2,IDRIG)-ANINT(T2/PBCY)*PBCY
    TEMRIG(3,IDRIG)=TEMRIG(3,IDRIG)-ANINT(T3/PBCZ)*PBCZ
    RETURN
  END SUBROUTINE EMAPRPBC

  SUBROUTINE EMAPRTRN(IDRIG,IDRIGJ)
    !-----------------------------------------------------------------------
    !   This routine find the transform to IDRIG related to 
    !       the position and orientation of IDRIGJ MAP
    !    RN=INV(MJ)*(M*(R-Rc)+Rc+T-T1-RJc)+RJc
    !    Compare:
    !    RN=MN*(R-Rc)+Rc+TN
    !    We have: MN=INV(MJ)*M 
    !             TN=INV(MJ)*(T+Rc-TJ-RJc)+RJc-Rc
    !
    !
    INTEGER IDRIG,IDRIGJ,IDEMP,IDEMPJ
    real(chm_real) DC1,DC2,DC3,T1,T2,T3
    real(chm_real) R11,R21,R31,R12,R22,R32,R13,R23,R33
    real(chm_real) RJ11,RJ21,RJ31,RJ12,RJ22,RJ32,RJ13,RJ23,RJ33
    !
    IDEMP=IDEMRIG(IDRIG)
    IDEMPJ=IDEMRIG(IDRIGJ)
    DC1=CEMAPX(IDEMP)-CEMAPX(IDEMPJ)
    DC2=CEMAPY(IDEMP)-CEMAPY(IDEMPJ)
    DC3=CEMAPZ(IDEMP)-CEMAPZ(IDEMPJ)
    T1=TEMRIG(1,IDRIG)-TEMRIG(1,IDRIGJ)+DC1
    T2=TEMRIG(2,IDRIG)-TEMRIG(2,IDRIGJ)+DC2
    T3=TEMRIG(3,IDRIG)-TEMRIG(3,IDRIGJ)+DC3
    R11=REMRIG(1,IDRIG)
    R21=REMRIG(2,IDRIG)
    R31=REMRIG(3,IDRIG)
    R12=REMRIG(4,IDRIG)
    R22=REMRIG(5,IDRIG)
    R32=REMRIG(6,IDRIG)
    R13=REMRIG(7,IDRIG)
    R23=REMRIG(8,IDRIG)
    R33=REMRIG(9,IDRIG)
    RJ11=REMRIG(1,IDRIGJ)
    RJ21=REMRIG(2,IDRIGJ)
    RJ31=REMRIG(3,IDRIGJ)
    RJ12=REMRIG(4,IDRIGJ)
    RJ22=REMRIG(5,IDRIGJ)
    RJ32=REMRIG(6,IDRIGJ)
    RJ13=REMRIG(7,IDRIGJ)
    RJ23=REMRIG(8,IDRIGJ)
    RJ33=REMRIG(9,IDRIGJ)
    REMRIG(1,IDRIG)=RJ11*R11+RJ21*R21+RJ31*R31
    REMRIG(2,IDRIG)=RJ12*R11+RJ22*R21+RJ32*R31
    REMRIG(3,IDRIG)=RJ13*R11+RJ23*R21+RJ33*R31
    REMRIG(4,IDRIG)=RJ11*R12+RJ21*R22+RJ31*R32
    REMRIG(5,IDRIG)=RJ12*R12+RJ22*R22+RJ32*R32
    REMRIG(6,IDRIG)=RJ13*R12+RJ23*R22+RJ33*R32
    REMRIG(7,IDRIG)=RJ11*R13+RJ21*R23+RJ31*R33
    REMRIG(8,IDRIG)=RJ12*R13+RJ22*R23+RJ32*R33
    REMRIG(9,IDRIG)=RJ13*R13+RJ23*R23+RJ33*R33
    TEMRIG(1,IDRIG)=RJ11*T1+RJ21*T2+RJ31*T3-DC1
    TEMRIG(2,IDRIG)=RJ12*T1+RJ22*T2+RJ32*T3-DC2
    TEMRIG(3,IDRIG)=RJ13*T1+RJ23*T2+RJ33*T3-DC3
    RETURN
  END SUBROUTINE EMAPRTRN

  SUBROUTINE EMAPSWAP(IDRIG)
    !-----------------------------------------------------------------------
    !     This routine  swap the position of the rigid segment with its storage
    !
    !
    INTEGER IDRIG
    real(chm_real) temp
    !
    TEMP=TEMRIG0(1,IDRIG)
    TEMRIG0(1,IDRIG)=TEMRIG(1,IDRIG)
    TEMRIG(1,IDRIG)=TEMP
    TEMP=TEMRIG0(2,IDRIG)
    TEMRIG0(2,IDRIG)=TEMRIG(2,IDRIG)
    TEMRIG(2,IDRIG)=TEMP
    TEMRIG(3,IDRIG)=TEMRIG0(3,IDRIG)
    TEMRIG0(3,IDRIG)=TEMRIG(3,IDRIG)
    TEMRIG(3,IDRIG)=TEMRIG0(3,IDRIG)
    REMRIG(1,IDRIG)=REMRIG0(1,IDRIG)
    REMRIG0(1,IDRIG)=REMRIG(1,IDRIG)
    REMRIG(1,IDRIG)=REMRIG0(1,IDRIG)
    TEMP=REMRIG0(2,IDRIG)
    REMRIG0(2,IDRIG)=REMRIG(2,IDRIG)
    REMRIG(2,IDRIG)=TEMP
    TEMP=REMRIG0(3,IDRIG)
    REMRIG0(3,IDRIG)=REMRIG(3,IDRIG)
    REMRIG(3,IDRIG)=TEMP
    TEMP=REMRIG0(4,IDRIG)
    REMRIG0(4,IDRIG)=REMRIG(4,IDRIG)
    REMRIG(4,IDRIG)=TEMP
    TEMP=REMRIG0(5,IDRIG)
    REMRIG0(5,IDRIG)=REMRIG(5,IDRIG)
    REMRIG(5,IDRIG)=TEMP
    TEMP=REMRIG0(6,IDRIG)
    REMRIG0(6,IDRIG)=REMRIG(6,IDRIG)
    REMRIG(6,IDRIG)=TEMP
    TEMP=REMRIG0(7,IDRIG)
    REMRIG0(7,IDRIG)=REMRIG(7,IDRIG)
    REMRIG(7,IDRIG)=TEMP
    TEMP=REMRIG0(8,IDRIG)
    REMRIG0(8,IDRIG)=REMRIG(8,IDRIG)
    REMRIG(8,IDRIG)=TEMP
    TEMP=REMRIG0(9,IDRIG)
    REMRIG0(9,IDRIG)=REMRIG(9,IDRIG)
    REMRIG(9,IDRIG)=TEMP
    RETURN
  END SUBROUTINE EMAPSWAP

  SUBROUTINE EMAPROT(IDRIG,XDIR,YDIR,ZDIR,PHI)
    !-----------------------------------------------------------------------
    !     This routine  update the rotation matrix of the rigid segment
    !
    !
  use corsubs,only:fndu
    real(chm_real) U(3,3),R(3,3)
    real(chm_real) RN(3),PHI,XDIR,YDIR,ZDIR
    INTEGER I,J,IDRIG
    LOGICAL LOK
    !
    RN(1)=XDIR
    RN(2)=YDIR
    RN(3)=ZDIR
    CALL FNDU(U,RN,PHI,LOK)
    IF(.NOT.LOK) THEN
       CALL WRNDIE(0,'<EMAPROT>','PARSING ERROR')
       RETURN
    ENDIF
    DO I=1,3
       DO J=1,3
          R(J,I)=REMRIG(3*I+J-3,IDRIG)
       ENDDO
    ENDDO
    CALL MULNXN(REMRIG(1,IDRIG),U,R,3)
    RETURN
  END SUBROUTINE EMAPROT

  SUBROUTINE EMAPPROJ(NAT,ISLCT,NATEMP,CX,CY,CZ, &
       XATEMP,YATEMP,ZATEMP,XPROJ,YPROJ,ZPROJ,TR,U)
    !-----------------------------------------------------------------------
    !     This routine update the coordinates of a rigid body based on its
    !     traslation matrix and vector.
    !
    INTEGER NAT,NATEMP,ISLCT(*)
    real(chm_real) XATEMP(*),YATEMP(*),ZATEMP(*)
    real(chm_real) XPROJ(*),YPROJ(*),ZPROJ(*)
    real(chm_real) CX,CY,CZ,TR(3),U(3,3),RI(3)
    INTEGER I,K
    !
    K=0
    DO I=1,NAT
       IF(ISLCT(I) == 0)cycle
       K=K+1
       RI(1)=XATEMP(K)-CX
       RI(2)=YATEMP(K)-CY
       RI(3)=ZATEMP(K)-CZ
       XPROJ(I)=U(1,1)*RI(1)+U(1,2)*RI(2)+U(1,3)*RI(3)+TR(1)+CX
       YPROJ(I)=U(2,1)*RI(1)+U(2,2)*RI(2)+U(2,3)*RI(3)+TR(2)+CY
       ZPROJ(I)=U(3,1)*RI(1)+U(3,2)*RI(2)+U(3,3)*RI(3)+TR(3)+CZ
    enddo
    !  TR and U should not change
    !      CALL FINDRU(NAT,X,Y,Z,ISLCT,XATEMP,
    !     & YATEMP,ZATEMP,NATEMP,IDATOM,TR,U)
    RETURN
  END SUBROUTINE EMAPPROJ

  SUBROUTINE EMAPREMAP(IDRIG,TR,U,IDEMPN)
  !-----------------------------------------------------------------------
  !     This routine ASSIGN a rigid segment to a electronic density map
  !     EMAPID and calculate the position of the rigid segment related to
  !     the origin of EMAPID
  !
  use memory
  use number
  implicit none
  INTEGER IDEMP,IDRIG,IDEMPN
  REAL(chm_real) TR(3),U(3,3)
  INTEGER LX,LY,LZ
  REAL(chm_real) DX,DY,DZ,CX,CY,CZ,XL,YL,ZL
  INTEGER LXN,LYN,LZN,MXN,MYN,MZN,NDATA,NATI
  REAL(chm_real) CXN,CYN,CZN,XLM,YLM,ZLM,XLN,YLN,ZLN
  INTEGER, allocatable, dimension(:)::itmp
  !
  IDEMP=IDEMRIG(IDRIG)
  LX=LEMAPX(IDEMP)
  LY=LEMAPY(IDEMP)
  LZ=LEMAPZ(IDEMP)
  CX=CEMAPX(IDEMP)
  CY=CEMAPY(IDEMP)
  CZ=CEMAPZ(IDEMP)
  DX=DEMAPX(IDEMP)
  DY=DEMAPY(IDEMP)
  DZ=DEMAPZ(IDEMP)
  XL=LX*DX
  YL=LY*DY
  ZL=LZ*DZ
  XLN=ABS(U(1,1)*XL+U(1,2)*YL+U(1,3)*ZL)
  XLM=ABS(U(1,1)*XL-U(1,2)*YL+U(1,3)*ZL)
  IF(XLN<XLM)XLN=XLM
  XLM=ABS(U(1,1)*XL+U(1,2)*YL-U(1,3)*ZL)
  IF(XLN<XLM)XLN=XLM
  XLM=ABS(U(1,1)*XL-U(1,2)*YL-U(1,3)*ZL)
  IF(XLN<XLM)XLN=XLM
  YLN=ABS(U(2,1)*XL+U(2,2)*YL+U(2,3)*ZL)
  YLM=ABS(U(2,1)*XL-U(2,2)*YL+U(2,3)*ZL)
  IF(YLN<YLM)YLN=YLM
  YLM=ABS(U(2,1)*XL+U(2,2)*YL-U(2,3)*ZL)
  IF(YLN<YLM)YLN=YLM
  YLM=ABS(U(2,1)*XL-U(2,2)*YL-U(2,3)*ZL)
  IF(YLN<YLM)YLN=YLM
  ZLN=ABS(U(3,1)*XL+U(3,2)*YL+U(3,3)*ZL)
  ZLM=ABS(U(3,1)*XL-U(3,2)*YL+U(3,3)*ZL)
  IF(ZLN<ZLM)ZLN=ZLM
  ZLM=ABS(U(3,1)*XL+U(3,2)*YL-U(3,3)*ZL)
  IF(ZLN<ZLM)ZLN=ZLM
  ZLM=ABS(U(3,1)*XL-U(3,2)*YL-U(3,3)*ZL)
  IF(ZLN<ZLM)ZLN=ZLM
  LXN=INT(XLN/DX)+1
  LYN=INT(YLN/DY)+1
  LZN=INT(ZLN/DZ)+1
  CXN=CX+TR(1)
  CYN=CY+TR(2)
  CZN=CZ+TR(3)
  MXN=INT((CXN-XLN/TWO)/DX)
  MYN=INT((CYN-YLN/TWO)/DY)
  MZN=INT((CZN-ZLN/TWO)/DZ)
  LEMAPX(IDEMPN)=LXN
  LEMAPY(IDEMPN)=LYN
  LEMAPZ(IDEMPN)=LZN
  DEMAPX(IDEMPN)=DX
  DEMAPY(IDEMPN)=DY
  DEMAPZ(IDEMPN)=DZ
  MEMAPX(IDEMPN)=MXN
  MEMAPY(IDEMPN)=MYN
  MEMAPZ(IDEMPN)=MZN
  CEMAPX(IDEMPN)=CXN
  CEMAPY(IDEMPN)=CYN
  CEMAPZ(IDEMPN)=CZN
  CEMAPX(IDEMPN)=CXN
  NDATA=LXN*LYN*LZN
  call chmalloc('emapsubs.src','EMAPDUP','empgrd(IDEMPN)%rho',NDATA, &
       cr4p=empgrd(IDEMPN)%rho)
  call chmalloc('emapsubs.src','EMAPDUP','empgrd(IDEMPN)%ddrho',NDATA, &
       cr4p=empgrd(IDEMPN)%ddrho)
  call chmalloc('emapsubs.src','EMAPDUP','empgrd(IDEMPN)%core',NDATA, &
       intgp=empgrd(IDEMPN)%core)
!  initialize the new mapid
  CALL EMAPINIT(IDEMPN,ZERO)
!  add rigid to the map
  CALL EMAPADD(IDEMPN,IDRIG)
!     statistics
  CALL EMAPDDR(IDEMPN)
  CALL EMAPCORE(IDEMPN,EMRCUT,EMICORE)
  CALL EMAPSTAT(IDEMPN)
! project reference atoms
  NATI=NATEMAP(IDEMP)
  NATEMAP(IDEMPN)=NATI
  IF(NATI > 0)THEN
     call chmalloc('emapsubs.src','EMAPREMAP','itemp ',NATI,intg=itmp )
     call chmalloc('emapsubs.src','EMAPREMAP','empcrd(IDEMPN)%x ',NATI,crlp=empcrd(IDEMPN)%x )
     call chmalloc('emapsubs.src','EMAPREMAP','empcrd(IDEMPN)%y ',NATI,crlp=empcrd(IDEMPN)%y )
     call chmalloc('emapsubs.src','EMAPREMAP','empcrd(IDEMPN)%z ',NATI,crlp=empcrd(IDEMPN)%z )
     itmp=1
     CALL EMAPPROJ(NATI,ITMP,NATI, &
          CX,CY,CZ, &
          empcrd(IDEMP)%x,empcrd(IDEMP)%y,empcrd(IDEMP)%z, &
          empcrd(IDEMPN)%x,empcrd(IDEMPN)%y,empcrd(IDEMPN)%z, &
          TR,U)
      call chmdealloc('emapsubs.src','EMAPREMAP','itmp', &
      NATI,intg=itmp)            
  ENDIF
  ! copy move properties
  LOWEMAP(:,IDEMPN)=LOWEMAP(:,IDEMP)
  UPEMAP(:,IDEMPN)=UPEMAP(:,IDEMP)
  TRNEMAP(:,IDEMPN)=TRNEMAP(:,IDEMP)
  ROTEMAP(:,IDEMPN)=ROTEMAP(:,IDEMP)

  RETURN
  END SUBROUTINE EMAPREMAP

  SUBROUTINE EMAPSUM(IDEMP,NRIG,IDRIGS)
    !-----------------------------------------------------------------------
    !     This routine build the whole emap from that of all rigid segments
    !
    !
  use number
    real(chm_real) DTX,DTY,DTZ,RHOCUT
    INTEGER IRIG,IDRIG,IDEMP,ICORE,NDATA,NRIG,IDRIGS(*)
    !
    NDATA=LEMAPX(IDEMP)*LEMAPY(IDEMP)*LEMAPZ(IDEMP)
    CALL EMAPINIT(IDEMP,ZERO)
    Do IRIG=1,NRIG
       IDRIG=IDRIGS(IRIG)
       CALL EMAPADD(IDEMP,IDRIG)
    ENDDO
    RHOCUT=EMRCUT
    ICORE=EMICORE
    CALL EMAPDDR(IDEMP)
    CALL EMAPCORE(IDEMP,RHOCUT,ICORE)
    CALL EMAPSTAT(IDEMP)
    RETURN
  END SUBROUTINE EMAPSUM

  SUBROUTINE EMAPINIT(IDEMP,BASE)
    !-----------------------------------------------------------------------
    !     This routine initialize a map object
    !
    INTEGER IDEMP,NDATA
    real(chm_real) BASE
    !
    NDATA=LEMAPX(IDEMP)*LEMAPY(IDEMP)*LEMAPZ(IDEMP)
    CALL EMAPINIT1(NDATA,BASE,empgrd(IDEMP)%rho, &
         empgrd(IDEMP)%ddrho,empgrd(IDEMP)%core)
    RETURN
  END SUBROUTINE EMAPINIT

  SUBROUTINE EMAPINIT1(NEMP,BASE,RHO,DDR,CORE)
    !-----------------------------------------------------------------------
  use number
    integer NEMP,I,CORE(*)
    real(chm_real) BASE
    real(chm_real4) RHO(*),DDR(*)
    !
    do i=1,NEMP
       RHO(I)=BASE
       DDR(I)=ZERO
       CORE(I)=0
    ENDDO
    RETURN
  END SUBROUTINE EMAPINIT1

  SUBROUTINE RMEMRIG(IDRIG)
    !-----------------------------------------------------------------------
    !     This routeine remove the last rigid domain
    !
    !
    INTEGER IDRIG
    !
    IF(IDRIG == NEMRIG)THEN
       NEMRIG=NEMRIG-1
    ELSE
       CALL WRNDIE(0,'<EMAPDELET>', &
            'Cannot delete a rigid in the middle')
    ENDIF
    RETURN
  END SUBROUTINE RMEMRIG

  SUBROUTINE EMAPSTAT(IDEMP)
    !-----------------------------------------------------------------------
    !     This routeine calculate the statistics of a map
    !
    !
    INTEGER IDEMP,NDATA
    !
    NDATA=LEMAPX(IDEMP)*LEMAPY(IDEMP)*LEMAPZ(IDEMP)
    CALL EMAPSTAT1(NDATA,AREMAP(IDEMP),RREMAP(IDEMP), &
         ADEMAP(IDEMP),DDEMAP(IDEMP),ACEMAP(IDEMP),CCEMAP(IDEMP), &
         RCEMAP(IDEMP),DCEMAP(IDEMP),RMXEMAP(IDEMP),RMNEMAP(IDEMP), &
         NCREMAP(IDEMP),empgrd(IDEMP)%rho,empgrd(IDEMP)%ddrho,empgrd(IDEMP)%core)
    RETURN
  END SUBROUTINE EMAPSTAT

  SUBROUTINE EMAPSTAT1(NDATA,AR,RR,AD,DD,AC,CC,RC,DC, &
       RMAX,RMIN,NCORE,RHO,DDRHO,CORE)
    !-----------------------------------------------------------------------
    !
    !
  use number
    INTEGER I,NDATA,COREI,CORE(*),NCORE
    real(chm_real4) RHO(*),DDRHO(*),RHOI,DDRHOI
    real(chm_real) AR,RR,AD,DD,AC,CC,RC,DC,RMAX,RMIN
    !
    AR=ZERO
    RR=ZERO
    AD=ZERO
    DD=ZERO
    AC=ZERO
    CC=ZERO
    RC=ZERO
    DC=ZERO
    NCORE=0
    RMAX=-RBIG
    RMIN=RBIG
    do i=1,ndata
       RHOI=RHO(I)
       DDRHOI=DDRHO(I)
       COREI=CORE(I)
       AR=AR+RHOI
       RR=RR+RHOI*RHOI
       AD=AD+DDRHOI
       DD=DD+DDRHOI*DDRHOI
       AC=AC+COREI
       CC=CC+COREI*COREI
       RC=RC+RHOI*COREI
       DC=DC+DDRHOI*COREI
       IF(RMAX < RHOI)RMAX=RHOI
       IF(RMIN > RHOI)RMIN=RHOI
       IF(COREI > 0)NCORE=NCORE+1
    enddo
    return
  end SUBROUTINE EMAPSTAT1

  SUBROUTINE EMAPCOPY(ID1,ID2)
    !-----------------------------------------------------------------------
    !     This routeine copy the distribution property to other map
    !
    integer ID1,ID2,NDATA1,NDATA2
    !
    CALL EMAPCOPY1(ID1,ID2, &
         empgrd(ID1)%rho, empgrd(ID1)%ddrho, empgrd(ID1)%core, &
         empgrd(ID2)%rho, empgrd(ID2)%ddrho, empgrd(ID2)%core)
    CALL EMAPSTAT(ID2)
    return
  end SUBROUTINE EMAPCOPY

  SUBROUTINE EMAPCOPY1(IDEMP1,IDEMP2,RHO1,DDR1,CORE1, &
       RHO2,DDR2,CORE2)
    !-----------------------------------------------------------------------
    !     Correlation between Map1 and Map2 over  core of Map1
    !     Only the MAP1 CORE space are taken into account
    !     It is asymmetric: CMM2(M1,M2)<>CMM2(M2,M1)
    !
  use number
    INTEGER IDEMP1,IDEMP2,CORE1(*),CORE2(*),IXYZ1,IXYZ2
    real(chm_real4) RHO1(*),RHO2(*),DDR1(*),DDR2(*)
    real(chm_real) RXYZ1,RXYZ2,ARXYZ1,ARXYZ2,RRXYZ1,RRXYZ2,ARXYZ,RRXYZ
    INTEGER LX1,LY1,LZ1,LX2,LY2,LZ2,NDATA1,NDATA2
    INTEGER MX1,MY1,MZ1,MX2,MY2,MZ2
    INTEGER MXS1,MYS1,MZS1,MXS2,MYS2,MZS2,MXE2,MYE2,MZE2
    real(chm_real) DX1,DY1,DZ1,DX2,DY2,DZ2
    !
    MXS1=MEMAPX(IDEMP1)
    MYS1=MEMAPY(IDEMP1)
    MZS1=MEMAPZ(IDEMP1)
    MXS2=MEMAPX(IDEMP2)
    MYS2=MEMAPY(IDEMP2)
    MZS2=MEMAPZ(IDEMP2)
    LX1=LEMAPX(IDEMP1)
    LY1=LEMAPY(IDEMP1)
    LZ1=LEMAPZ(IDEMP1)
    LX2=LEMAPX(IDEMP2)
    LY2=LEMAPY(IDEMP2)
    LZ2=LEMAPZ(IDEMP2)
    DX1=DEMAPX(IDEMP1)
    DY1=DEMAPY(IDEMP1)
    DZ1=DEMAPZ(IDEMP1)
    DX2=DEMAPX(IDEMP2)
    DY2=DEMAPY(IDEMP2)
    DZ2=DEMAPZ(IDEMP2)
    MXE2=MXS2+LX2-1
    MYE2=MYS2+LY2-1
    MZE2=MZS2+LZ2-1
    NDATA1=LX1*LY1*LZ1
    NDATA2=LX2*LY2*LZ2
    loop10: DO MZ2=MZS2,MZE2
       MZ1=NINT(MZ2*DZ2/DZ1)-MZS1
       IF(MZ1 < 0)MZ1=0
       IF(MZ1 >= LZ1)MZ1=LZ1-1
       loop20: DO MY2=MYS2,MYE2
          MY1=NINT(MY2*DY2/DY1)-MYS1
          IF(MY1 < 0)MY1=0
          IF(MY1 >= LY1)MY1=LY1-1
          loop30:DO MX2=MXS2,MXE2
             MX1=NINT(MX2*DX2/DX1)-MXS1
             IF(MX1 < 0)MX1=0
             IF(MX1 >= LX1)MX1=LX1-1
             IXYZ1=MX1+1+LX1*(MY1+LY1*MZ1)
             IXYZ2=MX2-MXS2+1+LX2*(MY2-MYS2+LY2*(MZ2-MZS2))
             RHO2(IXYZ2)=RHO1(IXYZ1)
             DDR2(IXYZ2)=DDR1(IXYZ1)
             CORE2(IXYZ2)=CORE1(IXYZ1)
          enddo loop30
       enddo loop20
    enddo loop10
    RETURN
  END SUBROUTINE EMAPCOPY1

  SUBROUTINE EMAPCRMM(IDEMP,IDEMPS,IDRIG,LDDR,CORRT)
    !-----------------------------------------------------------------------
    !     Correlation between MAP1 and MAPS+RIGID based on RIGID space
    !
  use number
    INTEGER IDEMP,IDEMPS,IDEMP1,IDRIG,NXYZS
    real(chm_real) AR,RR
    real(chm_real) CORRT
    LOGICAL LDDR
    IDEMP1=IDEMRIG(IDRIG)
    IF(LDDR)THEN
       AR=ADEMAP(IDEMP1)
       RR=DDEMAP(IDEMP1)
       CALL EMAPCRMM1(IDEMP,IDEMPS,IDRIG,empgrd(IDEMP)%ddrho, &
            empgrd(IDEMPS)%ddrho,empgrd(IDEMP1)%ddrho, &
            TEMRIG(1,IDRIG),REMRIG(1,IDRIG),AR,RR,CORRT)
    ELSE
       AR=AREMAP(IDEMP1)
       RR=RREMAP(IDEMP1)
       CALL EMAPCRMM1(IDEMP,IDEMPS,IDRIG,empgrd(IDEMP)%rho, &
            empgrd(IDEMPS)%rho,empgrd(IDEMP1)%rho, &
            TEMRIG(1,IDRIG),REMRIG(1,IDRIG),AR,RR,CORRT)
    ENDIF
    RETURN
  END SUBROUTINE EMAPCRMM

  SUBROUTINE EMAPCRMM1(IDEMP1,IDEMPS,IDRIG,RHO1,RHOS, &
       RHO2,TR,U,AR2,RR2,CORRT)
    !-----------------------------------------------------------------------
  use number
    INTEGER IDEMP1,IDEMPS,IDEMP2,IDRIG,NXYZS
    real(chm_real4) RHO1(*),RHOS(*),RHO2(*)
    real(chm_real) RXYZ1,RXYZS,RXYZ2,ARXYZ1,ARXYZ2, &
         RRXYZ1,RRXYZ2,RRXYZS
    real(chm_real) AR2,RR2,CORRT,DR1,DR2,DDR1,DDR2,TR(3),U(3,3),RRXYZ
    INTEGER IXYZ1,IXYZ2,NXYZ,II
    INTEGER LX1,LY1,LZ1,LX2,LY2,LZ2
    INTEGER MX1,MY1,MZ1,MX2,MY2,MZ2
    INTEGER MXS1,MYS1,MZS1,MXS2,MYS2,MZS2
    INTEGER MXE1,MYE1,MZE1,MXE2,MYE2,MZE2
    real(chm_real) X1,Y1,Z1,X2,Y2,Z2
    real(chm_real) DX1,DY1,DZ1,DXS,DYS,DZS,DX2,DY2,DZ2
    real(chm_real) CX1,CY1,CZ1,CX2,CY2,CZ2,GRATIO1,GRATIOS
    IDEMP2=IDEMRIG(IDRIG)
    MXS1=MEMAPX(IDEMP1)
    MYS1=MEMAPY(IDEMP1)
    MZS1=MEMAPZ(IDEMP1)
    MXS2=MEMAPX(IDEMP2)
    MYS2=MEMAPY(IDEMP2)
    MZS2=MEMAPZ(IDEMP2)
    LX1=LEMAPX(IDEMP1)
    LY1=LEMAPY(IDEMP1)
    LZ1=LEMAPZ(IDEMP1)
    LX2=LEMAPX(IDEMP2)
    LY2=LEMAPY(IDEMP2)
    LZ2=LEMAPZ(IDEMP2)
    DX1=DEMAPX(IDEMP1)
    DX2=DEMAPX(IDEMP2)
    DY1=DEMAPY(IDEMP1)
    DY2=DEMAPY(IDEMP2)
    DZ1=DEMAPZ(IDEMP1)
    DZ2=DEMAPZ(IDEMP2)
    CX2=CEMAPX(IDEMP2)
    CY2=CEMAPY(IDEMP2)
    CZ2=CEMAPZ(IDEMP2)
    MXE2=MXS2+LX2-1
    MYE2=MYS2+LY2-1
    MZE2=MZS2+LZ2-1
    ARXYZ1=ZERO
    RRXYZ1=ZERO
    ARXYZ2=AR2
    RRXYZ2=RR2
    RRXYZ=ZERO
    DO MZ2=MZS2,MZE2
       DO MY2=MYS2,MYE2
          loop100: DO MX2=MXS2,MXE2
             IXYZ2=MX2-MXS2+1+LX2*(MY2-MYS2+LY2*(MZ2-MZS2))
             X2=MX2*DX2-CX2
             Y2=MY2*DY2-CY2
             Z2=MZ2*DZ2-CZ2
             x1=U(1,1)*X2+U(1,2)*Y2+U(1,3)*Z2+TR(1)+CX2
             Y1=U(2,1)*X2+U(2,2)*Y2+U(2,3)*Z2+TR(2)+CY2
             Z1=U(3,1)*X2+U(3,2)*Y2+U(3,3)*Z2+TR(3)+CZ2
             MX1=NINT(X1/DX1)-MXS1
             MY1=NINT(Y1/DY1)-MYS1
             MZ1=NINT(Z1/DZ1)-MZS1
             IF(MX1 < 0)cycle loop100
             IF(MY1 < 0)cycle loop100
             IF(MZ1 < 0)cycle loop100
             IF(MX1 >= LX1)MX1=LX1-1
             IF(MY1 >= LY1)MY1=LY1-1
             IF(MZ1 >= LZ1)MZ1=LZ1-1
             IXYZ1=MX1+1+LX1*(MY1+LY1*MZ1)
             RXYZ1=RHO1(IXYZ1)
             RXYZS=RHOS(IXYZ1)
             RXYZ2=RHO2(IXYZ2)
             RRXYZ=RRXYZ+RXYZ1*RXYZ2
             ARXYZ1=ARXYZ1+RXYZ1
             RRXYZ1=RRXYZ1+RXYZ1*RXYZ1
             ARXYZ2=ARXYZ2+RXYZS
             RRXYZ2=RRXYZ2+RXYZS*(RXYZ2+RXYZ2+RXYZS)
          enddo loop100
       ENDDO
    ENDDO
    NXYZ=LX2*LY2*LZ2
    DR1=RRXYZ1-ARXYZ1*ARXYZ1/NXYZ
    DR2=RRXYZ2-ARXYZ2*ARXYZ2/NXYZ
    CORRT=-TWO
    if(DR1 > RSMALL)THEN
       DDR1=SQRT(DR1)
       if(DR2 > RSMALL)THEN
          DDR2=SQRT(DR2)
          CORRT=(RRXYZ-ARXYZ1*ARXYZ2/NXYZ)/DDR1/DDR2
       ENDIF
    ENDIF
    RETURN
  END SUBROUTINE EMAPCRMM1

  function CWEIGHT(CREMP,CRRIG,AK) result(cweight_1)
    !-----------------------------------------------------------------------
    !     The core-weighting function
    !
  use number,only:rsmall
    INTEGER CREMP,CRRIG
    real(chm_real) AK,DC,DB,cweight_1
    DC=CRRIG*CRRIG
    DB=CREMP*CREMP
    !      DC=CRRIG-CREMP
    CWEIGHT_1=DC/(RSMALL+AK*DB+DC)
    !
    RETURN
  END function CWEIGHT



  SUBROUTINE EMAPCMM(IDEMP1,IDEMP2,LDDR,LCORE,CORRT)
    !-----------------------------------------------------------------------
    !     This routine calculate the Map-Map correlation
    !
  use number
    INTEGER IDEMP1,IDEMP2
    real(chm_real) AR,RR,AD,DD,CORRT
    LOGICAL LCORE,LDDR
    !            
    IF(LDDR)THEN
       IF(LCORE)THEN
          CALL EMAPCMM2(IDEMP1,IDEMP2, &
               empgrd(IDEMP1)%ddrho,empgrd(IDEMP1)%core, &
               empgrd(IDEMP2)%ddrho,empgrd(IDEMP2)%core, &
               CORRT)
       ELSE
          AD=ADEMAP(IDEMP1)
          DD=DDEMAP(IDEMP1)
          CALL EMAPCMM1(IDEMP1,IDEMP2,empgrd(IDEMP1)%ddrho, &
               empgrd(IDEMP2)%ddrho,AD,DD,CORRT)
       ENDIF
    ELSE
       IF(LCORE)THEN
          CALL EMAPCMM2(IDEMP1,IDEMP2, &
               empgrd(IDEMP1)%rho,empgrd(IDEMP1)%core, &
               empgrd(IDEMP2)%rho,empgrd(IDEMP2)%core, &
               CORRT)
       ELSE
          AR=AREMAP(IDEMP1)
          RR=RREMAP(IDEMP1)
          CALL EMAPCMM1(IDEMP1,IDEMP2,empgrd(IDEMP1)%rho, &
               empgrd(IDEMP2)%rho,AR,RR,CORRT)
       ENDIF
    ENDIF
    RETURN
  END SUBROUTINE EMAPCMM

  SUBROUTINE EMAPCMM1(IDEMP1,IDEMP2,RHO1,RHO2,AR1,RR1,CORRT)
    !-----------------------------------------------------------------------
    !     Correlation between Map1 and Map2
    !     Only the Map1 space are taken into account
    !     It is asymmetric: CMM1(M1,M2)<>CMM1(M2,M1)
    !
  use number
    INTEGER IDEMP1,IDEMP2
    real(chm_real4) RHO1(*),RHO2(*)
    real(chm_real) RXYZ1,RXYZ2,ARXYZ1,ARXYZ2,RRXYZ1,RRXYZ2,ARXYZ,RRXYZ
    real(chm_real) AR1,RR1,DR1,DR2,DDR1,DDR2,CORRT
    INTEGER MOXS,MOXE,MOYS,MOYE,MOZS,MOZE,ITEMP
    INTEGER MWXS,MWXE,MWYS,MWYE,MWZS,MWZE,IXYZ1,IXYZ2,NXYZ,NXYZT
    INTEGER LX1,LY1,LZ1,LX2,LY2,LZ2
    INTEGER MX1,MY1,MZ1,MX2,MY2,MZ2
    INTEGER MXS1,MYS1,MZS1,MXS2,MYS2,MZS2
    real(chm_real) DX1,DY1,DZ1,DX2,DY2,DZ2
    !             
    MXS1=MEMAPX(IDEMP1)
    MYS1=MEMAPY(IDEMP1)
    MZS1=MEMAPZ(IDEMP1)
    MXS2=MEMAPX(IDEMP2)
    MYS2=MEMAPY(IDEMP2)
    MZS2=MEMAPZ(IDEMP2)
    LX1=LEMAPX(IDEMP1)
    LY1=LEMAPY(IDEMP1)
    LZ1=LEMAPZ(IDEMP1)
    LX2=LEMAPX(IDEMP2)
    LY2=LEMAPY(IDEMP2)
    LZ2=LEMAPZ(IDEMP2)
    DX1=DEMAPX(IDEMP1)
    DX2=DEMAPX(IDEMP2)
    DY1=DEMAPY(IDEMP1)
    DY2=DEMAPY(IDEMP2)
    DZ1=DEMAPZ(IDEMP1)
    DZ2=DEMAPZ(IDEMP2)
    MOXS=MXS1
    MWXS=NINT(MXS2*DX2/DX1)
    IF(MOXS < MWXS)THEN
       ITEMP=MOXS
       MOXS=MWXS
       MWXS=ITEMP
    ENDIF
    MOXE=MXS1+LX1-1
    MWXE=NINT((MXS2+LX2-1)*DX2/DX1)
    IF(MOXE > MWXE)THEN
       ITEMP=MOXE
       MOXE=MWXE
       MWXE=ITEMP
    ENDIF
    MOYS=MYS1
    MWYS=NINT(MYS2*DY2/DY1)
    if(MOYS < MWYS)THEN
       ITEMP=MOYS
       MOYS=MWYS
       MWYS=ITEMP
    ENDIF
    MOYE=MYS1+LY1-1
    MWYE=NINT((MYS2+LY2-1)*DY2/DY1)
    IF(MOYE > MWYE)THEN
       ITEMP=MOYE
       MOYE=MWYE
       MWYE=ITEMP
    ENDIF
    MOZS=MZS1
    MWZS=NINT(MZS2*DZ2/DZ1)
    IF(MOZS < MWZS)THEN
       ITEMP=MOZS
       MOZS=MWZS
       MWZS=ITEMP
    ENDIF
    MOZE=MZS1+LZ1-1
    MWZE=NINT((MZS2+LZ2-1)*DZ2/DZ1)
    IF(MOZE > MWZE)THEN
       ITEMP=MOZE
       MOZE=MWZE
       MWZE=ITEMP
    ENDIF
    NXYZ=0
    ARXYZ=ZERO
    RRXYZ=ZERO
    ARXYZ1=ZERO
    RRXYZ1=ZERO
    ARXYZ2=ZERO
    RRXYZ2=ZERO
    DO MZ1=MOZS,MOZE
       MZ2=NINT(MZ1*DZ1/DZ2)
       DO MY1=MOYS,MOYE
          MY2=NINT(MY1*DY1/DY2)
          DO MX1=MOXS,MOXE
             MX2=NINT(MX1*DX1/DX2)
             IXYZ1=MX1-MXS1+1+LX1*(MY1-MYS1+LY1*(MZ1-MZS1))
             IXYZ2=MX2-MXS2+1+LX2*(MY2-MYS2+LY2*(MZ2-MZS2))
             NXYZ=NXYZ+1
             RXYZ1=RHO1(IXYZ1)
             RXYZ2=RHO2(IXYZ2)
             RRXYZ=RRXYZ+RXYZ1*RXYZ2
             ARXYZ1=ARXYZ1+RXYZ1
             RRXYZ1=RRXYZ1+RXYZ1*RXYZ1
             ARXYZ2=ARXYZ2+RXYZ2
             RRXYZ2=RRXYZ2+RXYZ2*RXYZ2
          ENDDO
       ENDDO
    ENDDO
    DR1=RRXYZ1-ARXYZ1*ARXYZ1/NXYZ
    DR2=RRXYZ2-ARXYZ2*ARXYZ2/NXYZ
    CORRT=-TWO
    if(DR1 > RSMALL)THEN
       DDR1=SQRT(DR1)
       if(DR2 > RSMALL)THEN
          DDR2=SQRT(DR2)
          CORRT=(RRXYZ-ARXYZ1*ARXYZ2/NXYZ)/DDR1/DDR2
       ENDIF
    ENDIF
    RETURN
  END SUBROUTINE EMAPCMM1

  SUBROUTINE EMAPCMM2(IDEMP1,IDEMP2,RHO1,CORE1,RHO2,CORE2, &
       CORRT)
    !-----------------------------------------------------------------------
    !     Correlation between Map1 and Map2 over  core of Map1
    !     Only the MAP1 CORE space are taken into account
    !     It is asymmetric: CMM2(M1,M2)<>CMM2(M2,M1)
    !
  use number
    INTEGER IDEMP1,IDEMP2,CORE1(*),CORE2(*),CORE1I,CORE2I
    real(chm_real4) RHO1(*),RHO2(*)
    real(chm_real) RXYZ1,RXYZ2,ARXYZ1,ARXYZ2,RRXYZ1,RRXYZ2,ARXYZ,RRXYZ
    real(chm_real) DR1,DR2,DDR1,DDR2,CORRT
    INTEGER MOXS,MOXE,MOYS,MOYE,MOZS,MOZE,ITEMP
    INTEGER MWXS,MWXE,MWYS,MWYE,MWZS,MWZE,IXYZ1,IXYZ2,NXYZ
    INTEGER LX1,LY1,LZ1,LX2,LY2,LZ2
    INTEGER MX1,MY1,MZ1,MX2,MY2,MZ2
    INTEGER MXS1,MYS1,MZS1,MXS2,MYS2,MZS2,MXE1,MYE1,MZE1
    real(chm_real) DX1,DY1,DZ1,DX2,DY2,DZ2
    real(chm_real) WEI,ACW
    !
    MXS1=MEMAPX(IDEMP1)
    MYS1=MEMAPY(IDEMP1)
    MZS1=MEMAPZ(IDEMP1)
    MXS2=MEMAPX(IDEMP2)
    MYS2=MEMAPY(IDEMP2)
    MZS2=MEMAPZ(IDEMP2)
    LX1=LEMAPX(IDEMP1)
    LY1=LEMAPY(IDEMP1)
    LZ1=LEMAPZ(IDEMP1)
    LX2=LEMAPX(IDEMP2)
    LY2=LEMAPY(IDEMP2)
    LZ2=LEMAPZ(IDEMP2)
    DX1=DEMAPX(IDEMP1)
    DX2=DEMAPX(IDEMP2)
    DY1=DEMAPY(IDEMP1)
    DY2=DEMAPY(IDEMP2)
    DZ1=DEMAPZ(IDEMP1)
    DZ2=DEMAPZ(IDEMP2)
    MXE1=MXS1+LX1-1
    MYE1=MYS1+LY1-1
    MZE1=MZS1+LZ1-1
    NXYZ=0
    ARXYZ=ZERO
    RRXYZ=ZERO
    ARXYZ1=ZERO
    RRXYZ1=ZERO
    ARXYZ2=ZERO
    RRXYZ2=ZERO
    ACW = ZERO
    loop10: DO MZ1=MZS1,MZE1
       MZ2=NINT(MZ1*DZ1/DZ2)-MZS2
       IF(MZ2 < 0)MZ2=0
       IF(MZ2 >= LZ2)MZ2=LZ2-1
       loop20: DO MY1=MYS1,MYE1
          MY2=NINT(MY1*DY1/DY2)-MYS2
          IF(MY2 < 0)MY2=0
          IF(MY2 >= LY2)MY2=LY2-1
          loop30: DO MX1=MXS1,MXE1
             IXYZ1=MX1-MXS1+1+LX1*(MY1-MYS1+LY1*(MZ1-MZS1))
             CORE1I=CORE1(IXYZ1)
             IF(CORE1I == 0) cycle loop30
             MX2=NINT(MX1*DX1/DX2)-MXS2
             IF(MX2 < 0)MX2=0
             IF(MX2 >= LX2)MX2=LX2-1
             IXYZ2=MX2+1+LX2*(MY2+LY2*MZ2)
             CORE2I=CORE2(IXYZ2)
             WEI=CWEIGHT(CORE2I,CORE1I,BCORE)
             ACW=ACW+WEI
             RXYZ1=RHO1(IXYZ1)
             RXYZ2=RHO2(IXYZ2)
             RRXYZ=RRXYZ+WEI*RXYZ1*RXYZ2
             ARXYZ1=ARXYZ1+WEI*RXYZ1
             RRXYZ1=RRXYZ1+WEI*RXYZ1*RXYZ1
             ARXYZ2=ARXYZ2+WEI*RXYZ2
             RRXYZ2=RRXYZ2+WEI*RXYZ2*RXYZ2
          enddo loop30
       enddo loop20
    enddo loop10
    CORRT=-TWO
    IF(ACW < RSMALL)RETURN
    DR1=RRXYZ1-ARXYZ1*ARXYZ1/ACW
    DR2=RRXYZ2-ARXYZ2*ARXYZ2/ACW
    if(DR1 > RSMALL)THEN
       DDR1=SQRT(DR1)
       if(DR2 > RSMALL)THEN
          DDR2=SQRT(DR2)
          CORRT=(RRXYZ-ARXYZ1*ARXYZ2/ACW)/DDR1/DDR2
       ENDIF
    ENDIF
    RETURN
  END SUBROUTINE EMAPCMM2


  SUBROUTINE EMAPCRM(IDEMP,IDRIG,LDDR,LCORE,CORRT)
    !-----------------------------------------------------------------------
    !     Correlation between RIGID DOMAIN and Map
    !
  use number
    INTEGER IDEMP,IDEMP1,IDRIG
    real(chm_real) AR,RR,AD,DD,CORRT
    LOGICAL LDDR,LCORE
    !
    IDEMP1=IDEMRIG(IDRIG)
    IF(LDDR)THEN
       IF(LCORE)THEN
          CALL EMAPCRM2(IDEMP,IDRIG,empgrd(IDEMP)%ddrho, &
               empgrd(IDEMP)%core,     empgrd(IDEMP1)%ddrho, &
               empgrd(IDEMP1)%core,TEMRIG(1,IDRIG),REMRIG(1,IDRIG), &
               CORRT)
       ELSE
          AD=ADEMAP(IDEMP1)
          DD=DDEMAP(IDEMP1)
          CALL EMAPCRM1(IDEMP,IDRIG,empgrd(IDEMP)%ddrho, &
               empgrd(IDEMP1)%ddrho, &
               TEMRIG(1,IDRIG),REMRIG(1,IDRIG),AD,DD,CORRT)
       ENDIF
    ELSE
       IF(LCORE)THEN
          CALL EMAPCRM2(IDEMP,IDRIG,empgrd(IDEMP)%rho, &
               empgrd(IDEMP)%core,  empgrd(IDEMP1)%rho, &
               empgrd(IDEMP1)%core,TEMRIG(1,IDRIG),REMRIG(1,IDRIG), &
               CORRT)
       ELSE
          AR=AREMAP(IDEMP1)
          RR=RREMAP(IDEMP1)
          CALL EMAPCRM1(IDEMP,IDRIG,empgrd(IDEMP)%rho, &
               empgrd(IDEMP1)%rho, &
               TEMRIG(1,IDRIG),REMRIG(1,IDRIG),AR,RR,CORRT)
       ENDIF
    ENDIF
    RETURN
  END SUBROUTINE EMAPCRM

  SUBROUTINE EMAPCRM2(IDEMP1,IDRIG,RHO1,CORE1,RHO2,CORE2, &
       TR,U,CORRO)
    !-----------------------------------------------------------------------
    !     Correlation between RIGID domain and Map over  core of RIGID DOMAIN
    !     Only the CORE space of RIGID DOMAIN are taken into account
    !
  use number
    INTEGER IDEMP1,IDEMP2,IDRIG,CORE1(*),CORE2(*)
    real(chm_real4) RHO1(*),RHO2(*)
    real(chm_real) RXYZ1,RXYZ2,ARXYZ1,ARXYZ2,RRXYZ1,RRXYZ2,CORR
    real(chm_real) CORRO,DR1,DR2,DDR1,DDR2,TR(3),U(3,3),AKW
    INTEGER IXYZ1,IXYZ2,NXYZ,II,CORE1I,CORE2I,NC2,NC12
    INTEGER LX1,LY1,LZ1,LX2,LY2,LZ2
    INTEGER MX1,MY1,MZ1,MX2,MY2,MZ2
    INTEGER MXS1,MYS1,MZS1,MXS2,MYS2,MZS2,MXE2,MYE2,MZE2
    real(chm_real) WEI,ACW
    real(chm_real) X1,Y1,Z1,X2,Y2,Z2
    real(chm_real) DX1,DY1,DZ1,DX2,DY2,DZ2
    real(chm_real) CX1,CY1,CZ1,CX2,CY2,CZ2
    !
    IDEMP2=IDEMRIG(IDRIG)
    MXS1=MEMAPX(IDEMP1)
    MYS1=MEMAPY(IDEMP1)
    MZS1=MEMAPZ(IDEMP1)
    MXS2=MEMAPX(IDEMP2)
    MYS2=MEMAPY(IDEMP2)
    MZS2=MEMAPZ(IDEMP2)
    LX1=LEMAPX(IDEMP1)
    LY1=LEMAPY(IDEMP1)
    LZ1=LEMAPZ(IDEMP1)
    LX2=LEMAPX(IDEMP2)
    LY2=LEMAPY(IDEMP2)
    LZ2=LEMAPZ(IDEMP2)
    DX1=DEMAPX(IDEMP1)
    DX2=DEMAPX(IDEMP2)
    DY1=DEMAPY(IDEMP1)
    DY2=DEMAPY(IDEMP2)
    DZ1=DEMAPZ(IDEMP1)
    DZ2=DEMAPZ(IDEMP2)
    CX2=CEMAPX(IDEMP2)
    CY2=CEMAPY(IDEMP2)
    CZ2=CEMAPZ(IDEMP2)
    MXE2=MXS2+LX2-1
    MYE2=MYS2+LY2-1
    MZE2=MZS2+LZ2-1
    ACW=ZERO
    CORR=ZERO
    ARXYZ1=ZERO
    RRXYZ1=ZERO
    ARXYZ2=ZERO
    RRXYZ2=ZERO
    WEI=ONE
    NC2=0
    NC12=0
    DO MZ2=MZS2,MZE2
       DO MY2=MYS2,MYE2
          loop100: DO MX2=MXS2,MXE2
             IXYZ2=MX2-MXS2+1+LX2*(MY2-MYS2+LY2*(MZ2-MZS2))
             CORE2I=CORE2(IXYZ2)
             RXYZ2=RHO2(IXYZ2)
             IF(CORE2I <= 0) cycle loop100
             NC2=NC2+1
             X2=MX2*DX2-CX2
             Y2=MY2*DY2-CY2
             Z2=MZ2*DZ2-CZ2
             x1=U(1,1)*X2+U(1,2)*Y2+U(1,3)*Z2+TR(1)+CX2
             Y1=U(2,1)*X2+U(2,2)*Y2+U(2,3)*Z2+TR(2)+CY2
             Z1=U(3,1)*X2+U(3,2)*Y2+U(3,3)*Z2+TR(3)+CZ2
             MX1=NINT(X1/DX1)-MXS1
             MY1=NINT(Y1/DY1)-MYS1
             MZ1=NINT(Z1/DZ1)-MZS1
             IF(MX1 < 0)  cycle loop100
             IF(MY1 < 0)  cycle loop100
             IF(MZ1 < 0)  cycle loop100
             IF(MX1 >= LX1)cycle loop100
             IF(MY1 >= LY1)cycle loop100
             IF(MZ1 >= LZ1)cycle loop100
             NC12=NC12+1
             IXYZ1=MX1+1+LX1*(MY1+LY1*MZ1)
             RXYZ1=RHO1(IXYZ1)
             CORE1I=CORE1(IXYZ1)
             WEI=CWEIGHT(CORE1I,CORE2I,BCORE)
             ACW=ACW+WEI
             CORR=CORR+WEI*RXYZ1*RXYZ2
             ARXYZ1=ARXYZ1+WEI*RXYZ1
             RRXYZ1=RRXYZ1+WEI*RXYZ1*RXYZ1
             ARXYZ2=ARXYZ2+WEI*RXYZ2
             RRXYZ2=RRXYZ2+WEI*RXYZ2*RXYZ2
          enddo loop100
       ENDDO
    ENDDO
    CORRO=-TWO
    IF(NC12*2 < NC2)RETURN
    IF(ACW < RSMALL)RETURN
    DR1=RRXYZ1-ARXYZ1*ARXYZ1/ACW
    DR2=RRXYZ2-ARXYZ2*ARXYZ2/ACW
    if(DR1 > RSMALL)THEN
       DDR1=SQRT(DR1)
       if(DR2 > RSMALL)THEN
          DDR2=SQRT(DR2)
          CORRO=(CORR-ARXYZ1*ARXYZ2/ACW)/DDR1/DDR2
       ENDIF
    ENDIF
    RETURN
  END SUBROUTINE EMAPCRM2

  SUBROUTINE EMAPCRM1(IDEMP1,IDRIG,RHO1,RHO2,TR,U,AR2,RR2,CORRT)
    !-----------------------------------------------------------------------
    !     Correlation between RIGID domain and Map
    !     Only the MAP space of RIGID is taken into account
    !
  use number
    INTEGER IDEMP1,IDEMP2,IDRIG
    real(chm_real4) RHO1(*),RHO2(*)
    real(chm_real) RXYZ1,RXYZ2,ARXYZ1,ARXYZ2,RRXYZ1,RRXYZ2,CORR
    real(chm_real) AR2,RR2,CORRT,DR1,DR2,DDR1,DDR2,TR(3),U(3,3),ACW
    INTEGER IXYZ1,IXYZ2,NXYZ,II,CORE1I,CORE2I,NXYZT,NC12
    INTEGER LX1,LY1,LZ1,LX2,LY2,LZ2,I,J
    INTEGER MX1,MY1,MZ1,MX2,MY2,MZ2
    INTEGER MXS1,MYS1,MZS1,MXS2,MYS2,MZS2,MXE2,MYE2,MZE2
    real(chm_real) X1,Y1,Z1,X2,Y2,Z2
    real(chm_real) DX1,DY1,DZ1,DX2,DY2,DZ2
    real(chm_real) CX1,CY1,CZ1,CX2,CY2,CZ2
    LOGICAL LCORE
    IDEMP2=IDEMRIG(IDRIG)
    MXS1=MEMAPX(IDEMP1)
    MYS1=MEMAPY(IDEMP1)
    MZS1=MEMAPZ(IDEMP1)
    LX1=LEMAPX(IDEMP1)
    LY1=LEMAPY(IDEMP1)
    LZ1=LEMAPZ(IDEMP1)
    DX1=DEMAPX(IDEMP1)
    DY1=DEMAPY(IDEMP1)
    DZ1=DEMAPZ(IDEMP1)
    MXS2=MEMAPX(IDEMP2)
    MYS2=MEMAPY(IDEMP2)
    MZS2=MEMAPZ(IDEMP2)
    LX2=LEMAPX(IDEMP2)
    LY2=LEMAPY(IDEMP2)
    LZ2=LEMAPZ(IDEMP2)
    DX2=DEMAPX(IDEMP2)
    DY2=DEMAPY(IDEMP2)
    DZ2=DEMAPZ(IDEMP2)
    CX2=CEMAPX(IDEMP2)
    CY2=CEMAPY(IDEMP2)
    CZ2=CEMAPZ(IDEMP2)
    MXE2=MXS2+LX2-1
    MYE2=MYS2+LY2-1
    MZE2=MZS2+LZ2-1
    ACW=0
    CORR=ZERO
    ARXYZ1=ZERO
    RRXYZ1=ZERO
    DO MZ2=MZS2,MZE2
       DO MY2=MYS2,MYE2
          loop100: DO MX2=MXS2,MXE2
             IXYZ2=MX2-MXS2+1+LX2*(MY2-MYS2+LY2*(MZ2-MZS2))
             RXYZ2=RHO2(IXYZ2)
             X2=MX2*DX2-CX2
             Y2=MY2*DY2-CY2
             Z2=MZ2*DZ2-CZ2
             x1=U(1,1)*X2+U(1,2)*Y2+U(1,3)*Z2+TR(1)+CX2
             Y1=U(2,1)*X2+U(2,2)*Y2+U(2,3)*Z2+TR(2)+CY2
             Z1=U(3,1)*X2+U(3,2)*Y2+U(3,3)*Z2+TR(3)+CZ2
             MX1=NINT(X1/DX1)-MXS1
             MY1=NINT(Y1/DY1)-MYS1
             MZ1=NINT(Z1/DZ1)-MZS1
             IF(MX1 < 0)  cycle loop100
             IF(MY1 < 0)  cycle loop100
             IF(MZ1 < 0)  cycle loop100
             IF(MX1 >= LX1)cycle loop100
             IF(MY1 >= LY1)cycle loop100
             IF(MZ1 >= LZ1)cycle loop100
             NC12=NC12+1
             IXYZ1=MX1+1+LX1*(MY1+LY1*MZ1)
             RXYZ1=RHO1(IXYZ1)
             ACW=ACW+ONE
             CORR=CORR+RXYZ1*RXYZ2
             ARXYZ1=ARXYZ1+RXYZ1
             RRXYZ1=RRXYZ1+RXYZ1*RXYZ1
          enddo loop100
       ENDDO
    ENDDO
    NXYZT=LX2*LY2*LZ2
    DR1=RRXYZ1-ARXYZ1*ARXYZ1/NXYZT
    DR2=RR2-AR2*AR2/NXYZT
    CORRT=-TWO
    if(DR1 > RSMALL)THEN
       DDR1=SQRT(DR1)
       if(DR2 > RSMALL)THEN
          DDR2=SQRT(DR2)
          CORRT=(CORR-ARXYZ1*AR2/NXYZT)/DDR1/DDR2
       ENDIF
    ENDIF
    RETURN
  END SUBROUTINE EMAPCRM1


  SUBROUTINE EMAPCRR(IDRIG1,IDRIG2,LDDR,LCORE,CORRT)
    !-----------------------------------------------------------------------
    !     Correlation between RIGID1 and RIGID2
    !
  use number
    INTEGER IDEMP1,IDEMP2,IDRIG1,IDRIG2
    real(chm_real) AR,RR,AD,DD,CORRT
    LOGICAL LDDR,LCORE
    !
    IDEMP1=IDEMRIG(IDRIG1)
    IDEMP2=IDEMRIG(IDRIG2)
    IF(LDDR)THEN
       IF(LCORE)THEN
          CALL EMAPCRR2(IDRIG1,IDRIG2, &
               empgrd(IDEMP1)%ddrho,empgrd(IDEMP1)%core, &
               empgrd(IDEMP2)%ddrho,empgrd(IDEMP2)%core, &
               TEMRIG(1,IDRIG1),REMRIG(1,IDRIG1), &
               TEMRIG(1,IDRIG2),REMRIG(1,IDRIG2), &
               CORRT)
       ELSE
          AD=ADEMAP(IDEMP1)
          DD=DDEMAP(IDEMP1)
          CALL EMAPCRR1(IDRIG1,IDRIG2, &
               empgrd(IDEMP1)%ddrho, &
               empgrd(IDEMP2)%ddrho, &
               TEMRIG(1,IDRIG1),REMRIG(1,IDRIG1), &
               TEMRIG(1,IDRIG2),REMRIG(1,IDRIG2), &
               AD,DD,CORRT)
       ENDIF
    ELSE
       IF(LCORE)THEN
          CALL EMAPCRR2(IDRIG1,IDRIG2, &
               empgrd(IDEMP1)%rho,empgrd(IDEMP1)%core, &
               empgrd(IDEMP2)%rho,empgrd(IDEMP2)%core, &
               TEMRIG(1,IDRIG1),REMRIG(1,IDRIG1), &
               TEMRIG(1,IDRIG2),REMRIG(1,IDRIG2), &
               CORRT)
       ELSE
          AR=AREMAP(IDEMP1)
          RR=RREMAP(IDEMP1)
          CALL EMAPCRR1(IDRIG1,IDRIG2, &
               empgrd(IDEMP1)%rho, &
               empgrd(IDEMP2)%rho, &
               TEMRIG(1,IDRIG1),REMRIG(1,IDRIG1), &
               TEMRIG(1,IDRIG2),REMRIG(1,IDRIG2), &
               AR,RR,CORRT)
       ENDIF
    ENDIF
    RETURN
  END SUBROUTINE EMAPCRR

  SUBROUTINE EMAPCRR1(IDRIG1,IDRIG2,RHO1,RHO2, &
       TR1,U1,TR2,U2,AR1,RR1,CORRT)
    !-----------------------------------------------------------------------
    !     Correlation between RIGID1 domain and RIGID2 domain
    !     Only map space of RIGID1 are taken into account
    !     It is asymmetric: CRR1(R1,R2)<>CRR1(R2,R1)
    !
  use number
    INTEGER IDEMP1,IDEMP2,IDRIG1,IDRIG2
    real(chm_real4) RHO1(*),RHO2(*)
    real(chm_real) AR1,RR1,RXYZ1,RXYZ2,ARXYZ2,RRXYZ2,CORR
    real(chm_real) CORRO,CORRT,DR1,DR2,DDR1,DDR2,GRATIO
    real(chm_real) TR(3),U(3,3),TR1(3),U1(3,3),TR2(3),U2(3,3), &
         AKW,WEI,ACW
    real(chm_real) X1,Y1,Z1,X2,Y2,Z2
    INTEGER I,J,IXYZ1,IXYZ2,NXYZ,NXYZT,CORE1I,CORE2I
    INTEGER LX1,LY1,LZ1,LX2,LY2,LZ2
    INTEGER MX1,MY1,MZ1,MX2,MY2,MZ2
    INTEGER MXS1,MYS1,MZS1,MXS2,MYS2,MZS2,MXE1,MYE1,MZE1
    real(chm_real) DX1,DY1,DZ1,DX2,DY2,DZ2
    real(chm_real) CX1,CY1,CZ1,CX2,CY2,CZ2
    LOGICAL OK
    IDEMP1=IDEMRIG(IDRIG1)
    IDEMP2=IDEMRIG(IDRIG2)
    MXS1=MEMAPX(IDEMP1)
    MYS1=MEMAPY(IDEMP1)
    MZS1=MEMAPZ(IDEMP1)
    MXS2=MEMAPX(IDEMP2)
    MYS2=MEMAPY(IDEMP2)
    MZS2=MEMAPZ(IDEMP2)
    LX1=LEMAPX(IDEMP1)
    LY1=LEMAPY(IDEMP1)
    LZ1=LEMAPZ(IDEMP1)
    LX2=LEMAPX(IDEMP2)
    LY2=LEMAPY(IDEMP2)
    LZ2=LEMAPZ(IDEMP2)
    DX1=DEMAPX(IDEMP1)
    DX2=DEMAPX(IDEMP2)
    DY1=DEMAPY(IDEMP1)
    DY2=DEMAPY(IDEMP2)
    DZ1=DEMAPZ(IDEMP1)
    DZ2=DEMAPZ(IDEMP2)
    CX1=CEMAPX(IDEMP1)
    CY1=CEMAPY(IDEMP1)
    CZ1=CEMAPZ(IDEMP1)
    CX2=CEMAPX(IDEMP2)
    CY2=CEMAPY(IDEMP2)
    CZ2=CEMAPZ(IDEMP2)
    MXE1=MXS1+LX1-1
    MYE1=MYS1+LY1-1
    MZE1=MZS1+LZ1-1
    CORR=ZERO
    !      NXYZ=0
    !      ACW=ZERO
    !      ARXYZ1=ZERO
    !      RRXYZ1=ZERO
    ARXYZ2=ZERO
    RRXYZ2=ZERO
    DO MZ1=MZS1,MZE1
       DO MY1=MYS1,MYE1
          loop100: DO MX1=MXS1,MXE1
             IXYZ1=MX1-MXS1+1+LX1*(MY1-MYS1+LY1*(MZ1-MZS1))
             X1=MX1*DX1-CX1
             Y1=MY1*DY1-CY1
             Z1=MZ1*DZ1-CZ1
             x2=U1(1,1)*X1+U1(1,2)*Y1+U1(1,3)*Z1+TR1(1)+CX1-TR2(1)-CX2
             Y2=U1(2,1)*X1+U1(2,2)*Y1+U1(2,3)*Z1+TR1(2)+CY1-TR2(2)-CY2
             Z2=U1(3,1)*X1+U1(3,2)*Y1+U1(3,3)*Z1+TR1(3)+CZ1-TR2(3)-CZ2
             x1=U2(1,1)*X2+U2(2,1)*Y2+U2(3,1)*Z2+CX2
             Y1=U2(1,2)*X2+U2(2,2)*Y2+U2(3,2)*Z2+CY2
             Z1=U2(1,3)*X2+U2(2,3)*Y2+U2(3,3)*Z2+CZ2
             MX2=NINT(X1/DX2)-MXS2
             MY2=NINT(Y1/DY2)-MYS2
             MZ2=NINT(Z1/DZ2)-MZS2
             IF(MX2 < 0.OR.MY2.LT.0.OR.MZ2.LT.0)cycle loop100
             IF(MX2 >= LX2.OR.MY2.GE.LY2.OR.MZ2.GE.LZ2)cycle loop100
             IXYZ2=MX2+1+LX2*(MY2+LY2*MZ2)
             RXYZ1=RHO1(IXYZ1)
             RXYZ2=RHO2(IXYZ2)
             CORR=CORR+RXYZ1*RXYZ2
             !            ARXYZ1=ARXYZ1+RXYZ1
             !            RRXYZ1=RRXYZ1+RXYZ1*RXYZ1
             ARXYZ2=ARXYZ2+RXYZ2
             RRXYZ2=RRXYZ2+RXYZ2*RXYZ2
             !            NXYZ=NXYZ+1
          enddo loop100
       ENDDO
    ENDDO
    NXYZT=LX1*LY1*LZ1
    DR1=RR1-AR1*AR1/NXYZT
    DR2=RRXYZ2-ARXYZ2*ARXYZ2/NXYZT
    CORRT=-TWO
    IF(DR1 > RSMALL)THEN
       DDR1=SQRT(DR1)
       IF(DR2 > RSMALL)THEN
          DDR2=SQRT(DR2)
          CORRT=(CORR-AR1*ARXYZ2/NXYZT)/DDR1/DDR2
       ENDIF
    ENDIF
    RETURN
  END SUBROUTINE EMAPCRR1

  SUBROUTINE EMAPCRR2(IDRIG1,IDRIG2,RHO1,CORE1,RHO2,CORE2, &
       TR1,U1,TR2,U2,CORRT)
    !-----------------------------------------------------------------------
    !     Correlation between RIGID1 domain and RIGID2 over core of RIGID2
    !     Only the core region of rigid1 is taken into account
    !     It is asymmetric: CRR2(R1,R2)<>CRR2(R2,R1)
    !
  use number
    INTEGER IDEMP1,IDEMP2,IDRIG1,IDRIG2,CORE1(*),CORE2(*)
    real(chm_real4) RHO1(*),RHO2(*)
    real(chm_real) RXYZ1,RXYZ2,ARXYZ1,ARXYZ2,RRXYZ1,RRXYZ2,CORR
    real(chm_real) CORRO,CORRT,DR1,DR2,DDR1,DDR2,GRATIO
    real(chm_real) TR(3),U(3,3),TR1(3),U1(3,3),TR2(3),U2(3,3)
    real(chm_real) X1,Y1,Z1,X2,Y2,Z2
    real(chm_real) WEI,ACW
    INTEGER IXYZ1,IXYZ2,NXYZ,NXYZT,II,CORE1I,CORE2I
    INTEGER LX1,LY1,LZ1,LX2,LY2,LZ2
    INTEGER MX1,MY1,MZ1,MX2,MY2,MZ2
    INTEGER MXS1,MYS1,MZS1,MXS2,MYS2,MZS2,MXE1,MYE1,MZE1
    real(chm_real) DX1,DY1,DZ1,DX2,DY2,DZ2
    real(chm_real) CX1,CY1,CZ1,CX2,CY2,CZ2
    LOGICAL OK
    IDEMP1=IDEMRIG(IDRIG1)
    IDEMP2=IDEMRIG(IDRIG2)
    MXS1=MEMAPX(IDEMP1)
    MYS1=MEMAPY(IDEMP1)
    MZS1=MEMAPZ(IDEMP1)
    MXS2=MEMAPX(IDEMP2)
    MYS2=MEMAPY(IDEMP2)
    MZS2=MEMAPZ(IDEMP2)
    LX1=LEMAPX(IDEMP1)
    LY1=LEMAPY(IDEMP1)
    LZ1=LEMAPZ(IDEMP1)
    LX2=LEMAPX(IDEMP2)
    LY2=LEMAPY(IDEMP2)
    LZ2=LEMAPZ(IDEMP2)
    DX1=DEMAPX(IDEMP1)
    DX2=DEMAPX(IDEMP2)
    DY1=DEMAPY(IDEMP1)
    DY2=DEMAPY(IDEMP2)
    DZ1=DEMAPZ(IDEMP1)
    DZ2=DEMAPZ(IDEMP2)
    CX1=CEMAPX(IDEMP1)
    CY1=CEMAPY(IDEMP1)
    CZ1=CEMAPZ(IDEMP1)
    CX2=CEMAPX(IDEMP2)
    CY2=CEMAPY(IDEMP2)
    CZ2=CEMAPZ(IDEMP2)
    MXE1=MXS1+LX1-1
    MYE1=MYS1+LY1-1
    MZE1=MZS1+LZ1-1
    !      CALL INVT33(U,U1,OK)
    CORR=ZERO
    ACW=ZERO
    ARXYZ1=ZERO
    RRXYZ1=ZERO
    ARXYZ2=ZERO
    RRXYZ2=ZERO
    DO MZ1=MZS1,MZE1
       DO MY1=MYS1,MYE1
          loop100: DO MX1=MXS1,MXE1
             IXYZ1=MX1-MXS1+1+LX1*(MY1-MYS1+LY1*(MZ1-MZS1))
             CORE1I=CORE1(IXYZ1)
             IF(CORE1I <= 0) cycle loop100
             X2=MX1*DX1-CX1
             Y2=MY1*DY1-CY1
             Z2=MZ1*DZ1-CZ1
             x1=U1(1,1)*X2+U1(1,2)*Y2+U1(1,3)*Z2+TR1(1)+CX1-TR2(1)-CX2
             Y1=U1(2,1)*X2+U1(2,2)*Y2+U1(2,3)*Z2+TR1(2)+CY1-TR2(2)-CY2
             Z1=U1(3,1)*X2+U1(3,2)*Y2+U1(3,3)*Z2+TR1(3)+CZ1-TR2(3)-CZ2
             x2=U2(1,1)*X1+U2(2,1)*Y1+U2(3,1)*Z1+CX1
             Y2=U2(1,2)*X1+U2(2,2)*Y1+U2(3,2)*Z1+CY1
             Z2=U2(1,3)*X1+U2(2,3)*Y1+U2(3,3)*Z1+CZ1
             MX2=NINT(X2/DX2)-MXS2
             MY2=NINT(Y2/DY2)-MYS2
             MZ2=NINT(Z2/DZ2)-MZS2
             IF(MX2 < 0)MX2=0
             IF(MY2 < 0)MY2=0
             IF(MZ2 < 0)MZ2=0
             IF(MX2 >= LX2)MX2=LX2-1
             IF(MY2 >= LY2)MY2=LY2-1
             IF(MZ2 >= LZ2)MZ2=LZ2-1
             IXYZ2=MX2+1+LX2*(MY2+LY2*MZ2)
             CORE2I=CORE2(IXYZ1)
             RXYZ1=RHO1(IXYZ1)
             RXYZ2=RHO2(IXYZ2)
             WEI=CWEIGHT(CORE2I,CORE1I,BCORE)
             ACW=ACW+WEI
             ARXYZ1=ARXYZ1+WEI*RXYZ1
             RRXYZ1=RRXYZ1+WEI*RXYZ1*RXYZ1
             ARXYZ2=ARXYZ2+WEI*RXYZ2
             RRXYZ2=RRXYZ2+WEI*RXYZ2*RXYZ2
             CORR=CORR+WEI*RXYZ1*RXYZ2
          enddo loop100
       ENDDO
    ENDDO
    DR1=RRXYZ1-ARXYZ1*ARXYZ1/ACW
    DR2=RRXYZ2-ARXYZ2*ARXYZ2/ACW
    CORRT=-TWO
    if(DR1 > RSMALL)THEN
       DDR1=SQRT(DR1)
       if(DR2 > RSMALL)THEN
          DDR2=SQRT(DR2)
          CORRT=(CORR-ARXYZ1*ARXYZ2/ACW)/DDR1/DDR2
       ENDIF
    ENDIF
    RETURN
  END SUBROUTINE EMAPCRR2

  SUBROUTINE EMAPCMMR(IDEMP,IDEMPS,IDRIG,LDDR,LCORE,CORRT,NCALC)
    !-----------------------------------------------------------------------
    !     Correlation between MAP1 and MAP2+RIGID
    !     Assume MAP1 and MAP2 have the same boundary and grid
    !     and RIGID has the same grid size as the maps.
    !     Correlation are based on MAP1 space
    !
    INTEGER IDEMP,IDEMPS,IDEMP1,IDRIG,NCALC
    real(chm_real) ACW,AR,RR,ARS,RRS,RRT
    real(chm_real) CORRO,CORRT,DR1,DR2,DDR1,DDR2,TR(3),U(3,3),RRXYZ
    LOGICAL LDDR,LCORE
    SAVE ACW,AR,RR,ARS,RRS,RRT
    IDEMP1=IDEMRIG(IDRIG)
    IF(LCORE)THEN
       IF(NCALC == 0)THEN
          IF(LDDR)THEN
             CALL EMAPCMMR20(IDEMP,IDEMPS,IDRIG,empgrd(IDEMP)%ddrho, &
                  empgrd(IDEMP)%core, empgrd(IDEMPS)%ddrho, &
                  empgrd(IDEMPS)%core,empgrd(IDEMP1)%ddrho, &
                  empgrd(IDEMP1)%core,TEMRIG(1,IDRIG),REMRIG(1,IDRIG), &
                  ACW,AR,RR,ARS,RRS,RRT,CORRT)
          ELSE
             CALL EMAPCMMR20(IDEMP,IDEMPS,IDRIG,empgrd(IDEMP)%rho, &
                  empgrd(IDEMP)%core,empgrd(IDEMPS)%rho, &
                  empgrd(IDEMPS)%core,empgrd(IDEMP1)%rho, &
                  empgrd(IDEMP1)%core,TEMRIG(1,IDRIG),REMRIG(1,IDRIG), &
                  ACW,AR,RR,ARS,RRS,RRT,CORRT)
          ENDIF
       ELSE
          IF(LDDR)THEN
             CALL EMAPCMMR21(IDEMP,IDEMPS,IDRIG,empgrd(IDEMP)%ddrho, &
                  empgrd(IDEMP)%core,              empgrd(IDEMPS)%ddrho, &
                  empgrd(IDEMPS)%core,             empgrd(IDEMP1)%ddrho, &
                  empgrd(IDEMP1)%core,TEMRIG(1,IDRIG),REMRIG(1,IDRIG), &
                  ACW,AR,RR,ARS,RRS,RRT,CORRT)
          ELSE
             CALL EMAPCMMR21(IDEMP,IDEMPS,IDRIG,empgrd(IDEMP)%rho, &
                  empgrd(IDEMP)%core,empgrd(IDEMPS)%rho, &
                  empgrd(IDEMPS)%core,empgrd(IDEMP1)%rho, &
                  empgrd(IDEMP1)%core,TEMRIG(1,IDRIG),REMRIG(1,IDRIG), &
                  ACW,AR,RR,ARS,RRS,RRT,CORRT)
          ENDIF
       ENDIF
    ELSE
       IF(NCALC == 0)THEN
          IF(LDDR)THEN
             AR=ADEMAP(IDEMP)
             RR=DDEMAP(IDEMP)
             ARS=ADEMAP(IDEMPS)
             RRS=DDEMAP(IDEMPS)
             CALL EMAPCMMR10(IDEMP,IDEMPS,IDRIG,empgrd(IDEMP)%ddrho, &
                  empgrd(IDEMPS)%ddrho,empgrd(IDEMP1)%ddrho, &
                  TEMRIG(1,IDRIG),REMRIG(1,IDRIG),AR,RR,ARS,RRS,RRT,CORRT)
          ELSE
             AR=AREMAP(IDEMP)
             RR=RREMAP(IDEMP)
             ARS=AREMAP(IDEMPS)
             RRS=RREMAP(IDEMPS)
             CALL EMAPCMMR10(IDEMP,IDEMPS,IDRIG,empgrd(IDEMP)%rho, &
                  empgrd(IDEMPS)%rho,empgrd(IDEMP1)%rho, &
                  TEMRIG(1,IDRIG),REMRIG(1,IDRIG),AR,RR,ARS,RRS,RRT,CORRT)
          ENDIF
       ELSE
          IF(LDDR)THEN
             AR=ADEMAP(IDEMP)
             RR=DDEMAP(IDEMP)
             ARS=ADEMAP(IDEMPS)
             RRS=DDEMAP(IDEMPS)
             CALL EMAPCMMR11(IDEMP,IDEMPS,IDRIG,empgrd(IDEMP)%ddrho, &
                  empgrd(IDEMPS)%ddrho,empgrd(IDEMP1)%ddrho, &
                  TEMRIG(1,IDRIG),REMRIG(1,IDRIG),AR,RR,ARS,RRS,RRT,CORRT)
          ELSE
             AR=AREMAP(IDEMP)
             RR=RREMAP(IDEMP)
             ARS=AREMAP(IDEMPS)
             RRS=RREMAP(IDEMPS)
             CALL EMAPCMMR11(IDEMP,IDEMPS,IDRIG,empgrd(IDEMP)%rho, &
                  empgrd(IDEMPS)%rho,empgrd(IDEMP1)%rho, &
                  TEMRIG(1,IDRIG),REMRIG(1,IDRIG),AR,RR,ARS,RRS,RRT,CORRT)
          ENDIF
       ENDIF
    ENDIF
    NCALC=NCALC+1
    RETURN
  END SUBROUTINE EMAPCMMR


  SUBROUTINE EMAPCMMR10(IDEMP1,IDEMPS,IDRIG,RHO1,RHOS, &
       RHO2,TR,U,AR1,RR1,ARS,RRS,RRT,CORRT)
    !-----------------------------------------------------------------------
    !     Correlation between M1 and MS+RIG with Unknown NXYZS and RRXYZS
    !
  use number
    INTEGER IDEMP1,IDEMPS,IDEMP2,IDRIG,NXYZS
    real(chm_real4) RHO1(*),RHOS(*),RHO2(*)
    real(chm_real) RXYZ1,RXYZS,RXYZ2,AR1,RR1,ARS,RRS,RRT,ARXYZ2,RRXYZ2
    real(chm_real) CORRO,CORRT,DR1,DR2,DDR1,DDR2,TR(3),U(3,3),RRXYZ
    INTEGER IXYZ1,IXYZ2,NXYZ,II
    INTEGER LX1,LY1,LZ1,LX2,LY2,LZ2
    INTEGER MX1,MY1,MZ1,MX2,MY2,MZ2
    INTEGER MXS1,MYS1,MZS1,MXS2,MYS2,MZS2
    INTEGER MXE1,MYE1,MZE1,MXE2,MYE2,MZE2
    real(chm_real) X1,Y1,Z1,X2,Y2,Z2
    real(chm_real) DX1,DY1,DZ1,DXS,DYS,DZS,DX2,DY2,DZ2
    real(chm_real) CX1,CY1,CZ1,CX2,CY2,CZ2,GRATIO1,GRATIOS
    IDEMP2=IDEMRIG(IDRIG)
    MXS1=MEMAPX(IDEMP1)
    MYS1=MEMAPY(IDEMP1)
    MZS1=MEMAPZ(IDEMP1)
    MXS2=MEMAPX(IDEMP2)
    MYS2=MEMAPY(IDEMP2)
    MZS2=MEMAPZ(IDEMP2)
    LX1=LEMAPX(IDEMP1)
    LY1=LEMAPY(IDEMP1)
    LZ1=LEMAPZ(IDEMP1)
    LX2=LEMAPX(IDEMP2)
    LY2=LEMAPY(IDEMP2)
    LZ2=LEMAPZ(IDEMP2)
    DX1=DEMAPX(IDEMP1)
    DX2=DEMAPX(IDEMP2)
    DY1=DEMAPY(IDEMP1)
    DY2=DEMAPY(IDEMP2)
    DZ1=DEMAPZ(IDEMP1)
    DZ2=DEMAPZ(IDEMP2)
    CX2=CEMAPX(IDEMP2)
    CY2=CEMAPY(IDEMP2)
    CZ2=CEMAPZ(IDEMP2)
    MXE1=MXS1+LX1-1
    MYE1=MYS1+LY1-1
    MZE1=MZS1+LZ1-1
    MXE2=MXS2+LX2-1
    MYE2=MYS2+LY2-1
    MZE2=MZS2+LZ2-1
    NXYZ=LX1*LY1*LZ1
    ARXYZ2=ARS
    RRXYZ2=RRS
    RRT=ZERO
    RRXYZ=RRT
    DO MZ1=MZS1,MZE1
       DO MY1=MYS1,MYE1
          loop100: DO MX1=MXS1,MXE1
             IXYZ1=MX1-MXS1+1+LX1*(MY1-MYS1+LY1*(MZ1-MZS1))
             RXYZ1=RHO1(IXYZ1)
             RXYZS=RHOS(IXYZ1)
             RRT=RRT+RXYZ1*RXYZS
             RRXYZ=RRXYZ+RXYZ1*RXYZS
             X2=MX1*DX1-CX2-TR(1)
             Y2=MY1*DY1-CY2-TR(2)
             Z2=MZ1*DZ1-CZ2-TR(3)
             X1=U(1,1)*X2+U(2,1)*Y2+U(3,1)*Z2+CX2
             Y1=U(1,2)*X2+U(2,2)*Y2+U(3,2)*Z2+CY2
             Z1=U(1,3)*X2+U(2,3)*Y2+U(3,3)*Z2+CZ2
             MX2=NINT(X1/DX2)-MXS2
             MY2=NINT(Y1/DY2)-MYS2
             MZ2=NINT(Z1/DZ2)-MZS2
             IF(MX2*(LX2-MX2-1) < 0) cycle loop100
             IF(MY2*(LY2-MY2-1) < 0) cycle loop100
             IF(MZ2*(LZ2-MZ2-1) < 0) cycle loop100
             IXYZ2=MX2+1+LX2*(MY2+LY2*MZ2)
             RXYZ2=RHO2(IXYZ2)
             RRXYZ=RRXYZ+RXYZ1*RXYZ2
             ARXYZ2=ARXYZ2+RXYZ2
             RRXYZ2=RRXYZ2+RXYZ2*(RXYZ2+RXYZS+RXYZS)
          enddo loop100
       ENDDO
    ENDDO
    DR1=RR1-AR1*AR1/NXYZ
    DR2=RRXYZ2-ARXYZ2*ARXYZ2/NXYZ
    CORRT=-TWO
    if(DR1 > RSMALL)THEN
       DDR1=SQRT(DR1)
       if(DR2 > RSMALL)THEN
          DDR2=SQRT(DR2)
          CORRT=(RRXYZ-AR1*ARXYZ2/NXYZ)/DDR1/DDR2
       ENDIF
    ENDIF
    RETURN
  END SUBROUTINE EMAPCMMR10


  SUBROUTINE EMAPCMMR11(IDEMP1,IDEMPS,IDRIG,RHO1,RHOS, &
       RHO2,TR,U,AR1,RR1,ARS,RRS,RRT,CORRT)
    !-----------------------------------------------------------------------
    !     Correlation between M1 and MS+RIG with Unknown NXYZS and RRXYZS
    !
  use number
    INTEGER IDEMP1,IDEMPS,IDEMP2,IDRIG,NXYZS
    real(chm_real4) RHO1(*),RHOS(*),RHO2(*)
    real(chm_real) RXYZ1,RXYZS,RXYZ2,AR1,RR1,ARS,RRS,RRT,ARXYZ2,RRXYZ2
    real(chm_real) CORRO,CORRT,DR1,DR2,DDR1,DDR2,TR(3),U(3,3),RRXYZ
    INTEGER IXYZ1,IXYZ2,NXYZ,II
    INTEGER LX1,LY1,LZ1,LX2,LY2,LZ2
    INTEGER MX1,MY1,MZ1,MX2,MY2,MZ2
    INTEGER MXS1,MYS1,MZS1,MXS2,MYS2,MZS2
    INTEGER MXE1,MYE1,MZE1,MXE2,MYE2,MZE2
    real(chm_real) X1,Y1,Z1,X2,Y2,Z2
    real(chm_real) DX1,DY1,DZ1,DXS,DYS,DZS,DX2,DY2,DZ2
    real(chm_real) CX1,CY1,CZ1,CX2,CY2,CZ2,GRATIO1,GRATIOS
    IDEMP2=IDEMRIG(IDRIG)
    MXS1=MEMAPX(IDEMP1)
    MYS1=MEMAPY(IDEMP1)
    MZS1=MEMAPZ(IDEMP1)
    MXS2=MEMAPX(IDEMP2)
    MYS2=MEMAPY(IDEMP2)
    MZS2=MEMAPZ(IDEMP2)
    LX1=LEMAPX(IDEMP1)
    LY1=LEMAPY(IDEMP1)
    LZ1=LEMAPZ(IDEMP1)
    LX2=LEMAPX(IDEMP2)
    LY2=LEMAPY(IDEMP2)
    LZ2=LEMAPZ(IDEMP2)
    DX1=DEMAPX(IDEMP1)
    DX2=DEMAPX(IDEMP2)
    DY1=DEMAPY(IDEMP1)
    DY2=DEMAPY(IDEMP2)
    DZ1=DEMAPZ(IDEMP1)
    DZ2=DEMAPZ(IDEMP2)
    CX2=CEMAPX(IDEMP2)
    CY2=CEMAPY(IDEMP2)
    CZ2=CEMAPZ(IDEMP2)
    MXE1=MXS1+LX1-1
    MYE1=MYS1+LY1-1
    MZE1=MZS1+LZ1-1
    MXE2=MXS2+LX2-1
    MYE2=MYS2+LY2-1
    MZE2=MZS2+LZ2-1
    NXYZ=LX1*LY1*LZ1
    ARXYZ2=ARS
    RRXYZ2=RRS
    RRXYZ=RRT
    DO MZ2=MZS2,MZE2
       DO MY2=MYS2,MYE2
          loop100: DO MX2=MXS2,MXE2
             IXYZ2=MX2-MXS2+1+LX2*(MY2-MYS2+LY2*(MZ2-MZS2))
             X2=MX2*DX2-CX2
             Y2=MY2*DY2-CY2
             Z2=MZ2*DZ2-CZ2
             x1=U(1,1)*X2+U(1,2)*Y2+U(1,3)*Z2+TR(1)+CX2
             Y1=U(2,1)*X2+U(2,2)*Y2+U(2,3)*Z2+TR(2)+CY2
             Z1=U(3,1)*X2+U(3,2)*Y2+U(3,3)*Z2+TR(3)+CZ2
             MX1=NINT(X1/DX1)-MXS1
             MY1=NINT(Y1/DY1)-MYS1
             MZ1=NINT(Z1/DZ1)-MZS1
             IF(MX1*(LX1-MX1-1) < 0) cycle loop100
             IF(MY1*(LY1-MY1-1) < 0) cycle loop100
             IF(MZ1*(LZ1-MZ1-1) < 0) cycle loop100
             IXYZ1=MX1+1+LX1*(MY1+LY1*MZ1)
             RXYZ1=RHO1(IXYZ1)
             RXYZS=RHOS(IXYZ1)
             RXYZ2=RHO2(IXYZ2)
             RRXYZ=RRXYZ+RXYZ1*RXYZ2
             ARXYZ2=ARXYZ2+RXYZ2
             RRXYZ2=RRXYZ2+RXYZ2*(RXYZ2+RXYZS+RXYZS)
          enddo loop100
       ENDDO
    ENDDO
    DR1=RR1-AR1*AR1/NXYZ
    DR2=RRXYZ2-ARXYZ2*ARXYZ2/NXYZ
    CORRT=-TWO
    if(DR1 > RSMALL)THEN
       DDR1=SQRT(DR1)
       if(DR2 > RSMALL)THEN
          DDR2=SQRT(DR2)
          CORRT=(RRXYZ-AR1*ARXYZ2/NXYZ)/DDR1/DDR2
       ENDIF
    ENDIF
    RETURN
  END SUBROUTINE EMAPCMMR11


  SUBROUTINE EMAPCMMR20(IDEMP1,IDEMPS,IDRIG,RHO1,CORE1, &
       RHOS,CORES,RHO2,CORE2,TR,U,ACWS,AR1,RR1,ARS,RRS,RRT,CORRT)
    !-----------------------------------------------------------------------
    !     Correlation between M1 and MS+RIG with Unknown NXYZS and RRXYZS
    !
  use number
    INTEGER IDEMP1,IDEMPS,IDEMP2,IDRIG,NXYZS
    INTEGER CORE1(*),CORES(*),CORE2(*),CORE1I,CORESI,CORE2I
    real(chm_real4) RHO1(*),RHOS(*),RHO2(*)
    real(chm_real) WEI,WEIS,ACW,ACWS
    real(chm_real) RXYZ1,RXYZS,RXYZ2,ARXYZ1,RRXYZ1,ARXYZ2,RRXYZ2
    real(chm_real) AR1,RR1,ARS,RRS,RRT,RRXYZ
    real(chm_real) CORRO,CORRT,DR1,DR2,DDR1,DDR2,TR(3),U(3,3)
    INTEGER IXYZ1,IXYZ2,NXYZ,II
    INTEGER LX1,LY1,LZ1,LX2,LY2,LZ2
    INTEGER MX1,MY1,MZ1,MX2,MY2,MZ2
    INTEGER MXS1,MYS1,MZS1,MXS2,MYS2,MZS2
    INTEGER MXE1,MYE1,MZE1,MXE2,MYE2,MZE2
    real(chm_real) X1,Y1,Z1,X2,Y2,Z2
    real(chm_real) DX1,DY1,DZ1,DXS,DYS,DZS,DX2,DY2,DZ2
    real(chm_real) CX1,CY1,CZ1,CX2,CY2,CZ2,GRATIO1,GRATIOS
    IDEMP2=IDEMRIG(IDRIG)
    MXS1=MEMAPX(IDEMP1)
    MYS1=MEMAPY(IDEMP1)
    MZS1=MEMAPZ(IDEMP1)
    MXS2=MEMAPX(IDEMP2)
    MYS2=MEMAPY(IDEMP2)
    MZS2=MEMAPZ(IDEMP2)
    LX1=LEMAPX(IDEMP1)
    LY1=LEMAPY(IDEMP1)
    LZ1=LEMAPZ(IDEMP1)
    LX2=LEMAPX(IDEMP2)
    LY2=LEMAPY(IDEMP2)
    LZ2=LEMAPZ(IDEMP2)
    DX1=DEMAPX(IDEMP1)
    DX2=DEMAPX(IDEMP2)
    DY1=DEMAPY(IDEMP1)
    DY2=DEMAPY(IDEMP2)
    DZ1=DEMAPZ(IDEMP1)
    DZ2=DEMAPZ(IDEMP2)
    CX2=CEMAPX(IDEMP2)
    CY2=CEMAPY(IDEMP2)
    CZ2=CEMAPZ(IDEMP2)
    MXE1=MXS1+LX1-1
    MYE1=MYS1+LY1-1
    MZE1=MZS1+LZ1-1
    MXE2=MXS2+LX2-1
    MYE2=MYS2+LY2-1
    MZE2=MZS2+LZ2-1
    NXYZ=LX1*LY1*LZ1
    ACWS=ZERO
    AR1=ZERO
    RR1=ZERO
    ARS=ZERO
    RRS=ZERO
    RRT=ZERO
    ACW=ZERO
    ARXYZ1=ZERO
    RRXYZ1=ZERO
    ARXYZ2=ZERO
    RRXYZ2=ZERO
    RRXYZ=ZERO
    DO MZ1=MZS1,MZE1
       DO MY1=MYS1,MYE1
          loop100: DO MX1=MXS1,MXE1
             IXYZ1=MX1-MXS1+1+LX1*(MY1-MYS1+LY1*(MZ1-MZS1))
             CORE1I=CORE1(IXYZ1)
             IF(CORE1I == 0) cycle loop100
             RXYZ1=RHO1(IXYZ1)
             RXYZS=RHOS(IXYZ1)
             CORESI=CORES(IXYZ1)
             WEIS=CWEIGHT(CORE1I,CORESI,BCORE)
             ACWS=ACWS+WEIS
             AR1=AR1+WEIS*RXYZ1
             RR1=RR1+WEIS*RXYZ1*RXYZ1
             ARS=ARS+WEIS*RXYZS
             RRS=RRS+WEIS*RXYZS*RXYZS
             RRT=RRT+RXYZ1*RXYZS*WEIS
             ACW=ACW+WEIS
             ARXYZ1=ARXYZ1+WEIS*RXYZ1
             RRXYZ1=RRXYZ1+WEIS*RXYZ1*RXYZ1
             ARXYZ2=ARXYZ2+WEIS*RXYZS
             RRXYZ2=RRXYZ2+WEIS*RXYZS*RXYZS
             RRXYZ=RRXYZ+WEIS*RXYZ1*RXYZS
             X2=MX1*DX1-CX2-TR(1)
             Y2=MY1*DY1-CY2-TR(2)
             Z2=MZ1*DZ1-CZ2-TR(3)
             X1=U(1,1)*X2+U(2,1)*Y2+U(3,1)*Z2+CX2
             Y1=U(1,2)*X2+U(2,2)*Y2+U(3,2)*Z2+CY2
             Z1=U(1,3)*X2+U(2,3)*Y2+U(3,3)*Z2+CZ2
             MX2=NINT(X1/DX2)-MXS2
             MY2=NINT(Y1/DY2)-MYS2
             MZ2=NINT(Z1/DZ2)-MZS2
             IF(MX2*(LX2-MX2-1) < 0) cycle loop100
             IF(MY2*(LY2-MY2-1) < 0) cycle loop100
             IF(MZ2*(LZ2-MZ2-1) < 0) cycle loop100
             IXYZ2=MX2+1+LX2*(MY2+LY2*MZ2)
             CORE2I=CORE2(IXYZ2)
             IF(CORE2I == 0) cycle loop100
             RXYZ2=RHO2(IXYZ2)+RXYZS
             CORE2I=CORE2I+CORESI
             WEI=CWEIGHT(CORE1I,CORE2I,BCORE)
             ACW=ACW+WEI-WEIS
             RRXYZ=RRXYZ+RXYZ1*(WEI*RXYZ2-WEIS*RXYZS)
             ARXYZ1=ARXYZ1+(WEI-WEIS)*RXYZ1
             RRXYZ1=RRXYZ1+(WEI-WEIS)*RXYZ1*RXYZ1
             ARXYZ2=ARXYZ2+WEI*RXYZ2-WEIS*RXYZS
             RRXYZ2=RRXYZ2+WEI*RXYZ2*RXYZ2-WEIS*RXYZS*RXYZS
          enddo loop100
       ENDDO
    ENDDO
    DR1=RRXYZ1-ARXYZ1*ARXYZ1/ACW
    DR2=RRXYZ2-ARXYZ2*ARXYZ2/ACW
    CORRT=-TWO
    if(DR1 > RSMALL)THEN
       DDR1=SQRT(DR1)
       if(DR2 > RSMALL)THEN
          DDR2=SQRT(DR2)
          CORRT=(RRXYZ-ARXYZ1*ARXYZ2/ACW)/DDR1/DDR2
       ENDIF
    ENDIF
    RETURN
  END SUBROUTINE EMAPCMMR20


  SUBROUTINE EMAPCMMR21(IDEMP1,IDEMPS,IDRIG,RHO1,CORE1, &
       RHOS,CORES,RHO2,CORE2,TR,U,ACWS,AR1,RR1,ARS,RRS,RRT,CORRT)
    !-----------------------------------------------------------------------
    !     Correlation between M1 and MS+RIG with Unknown NXYZS and RRXYZS
    !
  use number
    INTEGER IDEMP1,IDEMPS,IDEMP2,IDRIG,NXYZS
    INTEGER CORE1(*),CORES(*),CORE2(*),CORE1I,CORESI,CORE2I
    real(chm_real4) RHO1(*),RHOS(*),RHO2(*)
    real(chm_real) WEI,WEIS,ACW,ACWS
    real(chm_real) RXYZ1,RXYZS,RXYZ2,ARXYZ1,RRXYZ1,ARXYZ2,RRXYZ2
    real(chm_real) AR1,RR1,ARS,RRS,RRT,RRXYZ
    real(chm_real) CORRO,CORRT,DR1,DR2,DDR1,DDR2,TR(3),U(3,3)
    INTEGER IXYZ1,IXYZ2,NXYZ,II
    INTEGER LX1,LY1,LZ1,LX2,LY2,LZ2
    INTEGER MX1,MY1,MZ1,MX2,MY2,MZ2
    INTEGER MXS1,MYS1,MZS1,MXS2,MYS2,MZS2
    INTEGER MXE1,MYE1,MZE1,MXE2,MYE2,MZE2
    real(chm_real) X1,Y1,Z1,X2,Y2,Z2
    real(chm_real) DX1,DY1,DZ1,DXS,DYS,DZS,DX2,DY2,DZ2
    real(chm_real) CX1,CY1,CZ1,CX2,CY2,CZ2,GRATIO1,GRATIOS
    IDEMP2=IDEMRIG(IDRIG)
    MXS1=MEMAPX(IDEMP1)
    MYS1=MEMAPY(IDEMP1)
    MZS1=MEMAPZ(IDEMP1)
    MXS2=MEMAPX(IDEMP2)
    MYS2=MEMAPY(IDEMP2)
    MZS2=MEMAPZ(IDEMP2)
    LX1=LEMAPX(IDEMP1)
    LY1=LEMAPY(IDEMP1)
    LZ1=LEMAPZ(IDEMP1)
    LX2=LEMAPX(IDEMP2)
    LY2=LEMAPY(IDEMP2)
    LZ2=LEMAPZ(IDEMP2)
    DX1=DEMAPX(IDEMP1)
    DX2=DEMAPX(IDEMP2)
    DY1=DEMAPY(IDEMP1)
    DY2=DEMAPY(IDEMP2)
    DZ1=DEMAPZ(IDEMP1)
    DZ2=DEMAPZ(IDEMP2)
    CX2=CEMAPX(IDEMP2)
    CY2=CEMAPY(IDEMP2)
    CZ2=CEMAPZ(IDEMP2)
    MXE1=MXS1+LX1-1
    MYE1=MYS1+LY1-1
    MZE1=MZS1+LZ1-1
    MXE2=MXS2+LX2-1
    MYE2=MYS2+LY2-1
    MZE2=MZS2+LZ2-1
    NXYZ=LX1*LY1*LZ1
    ACW=ACWS
    ARXYZ1=AR1
    RRXYZ1=RR1
    ARXYZ2=ARS
    RRXYZ2=RRS
    RRXYZ=RRT
    DO MZ2=MZS2,MZE2
       DO MY2=MYS2,MYE2
          loop100: DO MX2=MXS2,MXE2
             IXYZ2=MX2-MXS2+1+LX2*(MY2-MYS2+LY2*(MZ2-MZS2))
             CORE2I=CORE2(IXYZ2)
             IF(CORE2I == 0) cycle loop100
             X2=MX2*DX2-CX2
             Y2=MY2*DY2-CY2
             Z2=MZ2*DZ2-CZ2
             x1=U(1,1)*X2+U(1,2)*Y2+U(1,3)*Z2+TR(1)+CX2
             Y1=U(2,1)*X2+U(2,2)*Y2+U(2,3)*Z2+TR(2)+CY2
             Z1=U(3,1)*X2+U(3,2)*Y2+U(3,3)*Z2+TR(3)+CZ2
             MX1=NINT(X1/DX1)-MXS1
             MY1=NINT(Y1/DY1)-MYS1
             MZ1=NINT(Z1/DZ1)-MZS1
             IF(MX1*(LX1-MX1-1) < 0) cycle loop100
             IF(MY1*(LY1-MY1-1) < 0) cycle loop100
             IF(MZ1*(LZ1-MZ1-1) < 0) cycle loop100
             IXYZ1=MX1+1+LX1*(MY1+LY1*MZ1)
             CORE1I=CORE1(IXYZ1)
             IF(CORE1I == 0) cycle loop100
             CORESI=CORES(IXYZ1)
             WEIS=CWEIGHT(CORE1I,CORESI,BCORE)
             CORE2I=CORE2I+CORESI
             WEI=CWEIGHT(CORE1I,CORE2I,BCORE)
             RXYZ1=RHO1(IXYZ1)
             RXYZS=RHOS(IXYZ1)
             RXYZ2=RHO2(IXYZ2)+RXYZS
             ACW=ACW+WEI-WEIS
             RRXYZ=RRXYZ+RXYZ1*(WEI*RXYZ2-WEIS*RXYZS)
             ARXYZ1=ARXYZ1+(WEI-WEIS)*RXYZ1
             RRXYZ1=RRXYZ1+(WEI-WEIS)*RXYZ1*RXYZ1
             ARXYZ2=ARXYZ2+WEI*RXYZ2-WEIS*RXYZS
             RRXYZ2=RRXYZ2+WEI*RXYZ2*RXYZ2-WEIS*RXYZS*RXYZS
          enddo loop100
       ENDDO
    ENDDO
    DR1=RRXYZ1-ARXYZ1*ARXYZ1/ACW
    DR2=RRXYZ2-ARXYZ2*ARXYZ2/ACW
    CORRT=-TWO
    if(DR1 > RSMALL)THEN
       DDR1=SQRT(DR1)
       if(DR2 > RSMALL)THEN
          DDR2=SQRT(DR2)
          CORRT=(RRXYZ-ARXYZ1*ARXYZ2/ACW)/DDR1/DDR2
       ENDIF
    ENDIF
    RETURN
  END SUBROUTINE EMAPCMMR21


  SUBROUTINE RMEMAP(IDEMP)
    !-----------------------------------------------------------------------
    !     This routeine remove the last map
    !
  use stream
  use memory
    !
    INTEGER IDEMP,NDATA,NATI
    !
    IF(IDEMP == NEMAP)THEN
       NATI=NATEMAP(IDEMP)
       IF(NATI > 0)THEN
          call chmdealloc('emapsubs.src','RMEMAP','empcrd(IDEMP)%x',NATI, &
               crlp=empcrd(IDEMP)%x)
          call chmdealloc('emapsubs.src','RMEMAP','empcrd(IDEMP)%y',NATI, &
               crlp=empcrd(IDEMP)%y)
          call chmdealloc('emapsubs.src','RMEMAP','empcrd(IDEMP)%z',NATI, &
               crlp=empcrd(IDEMP)%z)
          NATEMAP(IDEMP)=0
       ENDIF
       NDATA=LEMAPX(IDEMP)*LEMAPY(IDEMP)*LEMAPZ(IDEMP)
       call chmdealloc('emapsubs.src','RMEMAP','empgrd(IDEMP)%rho',NDATA, &
            cr4p=empgrd(IDEMP)%rho)
       call chmdealloc('emapsubs.src','RMEMAP','empgrd(IDEMP)%ddrho',NDATA, &
            cr4p=empgrd(IDEMP)%ddrho)
       call chmdealloc('emapsubs.src','RMEMAP','empgrd(IDEMP)%core',NDATA, &
            intgp=empgrd(IDEMP)%core)
       NEMAP=NEMAP-1
    ELSE
       IF(PRNLEV > 2) &
            WRITE(OUTU,1050)IDEMP,EMAPID(IDEMP),NEMAP,EMAPID(NEMAP)
1050   FORMAT("IDEMP=",I3," ID:",A20," NEMAP=",I3," ID:",A20)
       CALL WRNDIE(0,'<EMAPDELET>', &
            'Cannot delete an emap in the middle')
    ENDIF
    RETURN
  END SUBROUTINE RMEMAP


  SUBROUTINE FINDRU(NATOMA,XA,YA,ZA,ISLCT,NATOMB,XB,YB,ZB,TR,U)
    !-----------------------------------------------------------------------
    !     This routine find the translation vector and rotation matrix
    !     as respect to the reference atom set
    !
  use number
  use stream
  use corsubs,only:frotu
    !
    INTEGER NATOMB,NATOMA,ISLCT(*)
    real(chm_real) XA(*),YA(*),ZA(*),XB(*),YB(*),ZB(*)
    LOGICAL LPRINT,LNOROT,QEVW
    !
    !
    real(chm_real) TR(3),R(9),U(9),EVA(3),DEVA(3,3)
    real(chm_real) RN(3)
    real(chm_real) CMXA,CMYA,CMZA,CMXB,CMYB,CMZB,CMXC,CMYC,CMZC
    real(chm_real) XI,YI,ZI,XJ,YJ,ZJ
    real(chm_real) TMASS,RMST,RMSV
    INTEGER I,KA,KB
    !
    CMXA=0.0
    CMYA=0.0
    CMZA=0.0
    CMXB=0.0
    CMYB=0.0
    CMZB=0.0
    TMASS=0.0
    KB=0
    DO KA=1,NATOMA
       if(ISLCT(KA) /= 1) cycle
       KB=KB+1
       IF (XA(KA) /= ANUM .AND. XB(KB).NE.ANUM) THEN
          CMXA=CMXA+XA(KA)
          CMYA=CMYA+YA(KA)
          CMZA=CMZA+ZA(KA)
          CMXB=CMXB+XB(KB)
          CMYB=CMYB+YB(KB)
          CMZB=CMZB+ZB(KB)
          TMASS=TMASS+ONE
       ENDIF
    enddo
    IF(KB == 0)THEN
       DO I=1,9
          U(I)=0.0
       ENDDO
       DO I=1,3
          TR(I)=0.0
          U(3*(I-1)+I)=1.0
       ENDDO
       RETURN
    ELSE IF(KB /= NATOMB)THEN
       CALL WRNDIE(0,'<FINDRU>', &
            'Number of selected atom is different from'// &
            ' number of reference atoms')
    ENDIF
    CMXA=CMXA/TMASS
    CMYA=CMYA/TMASS
    CMZA=CMZA/TMASS
    CMXB=CMXB/TMASS
    CMYB=CMYB/TMASS
    CMZB=CMZB/TMASS
    !
    TR(1)=CMXA-CMXB
    TR(2)=CMYA-CMYB
    TR(3)=CMZA-CMZB
    !
    !       COMPUTE ROTATION MATRIX FROM LAGRANGIAN
    !
    DO I=1,9
       R(I)=0.0
    ENDDO
    KB=0
    DO KA=1,NATOMA
       IF(ISLCT(KA) /= 1) cycle
       KB=KB+1
       XI=XB(KB)-CMXB
       YI=YB(KB)-CMYB
       ZI=ZB(KB)-CMZB
       XJ=XA(KA)-CMXA
       YJ=YA(KA)-CMYA
       ZJ=ZA(KA)-CMZA
       R(1)=R(1)+XI*XJ
       R(2)=R(2)+XI*YJ
       R(3)=R(3)+XI*ZJ
       R(4)=R(4)+YI*XJ
       R(5)=R(5)+YI*YJ
       R(6)=R(6)+YI*ZJ
       R(7)=R(7)+ZI*XJ
       R(8)=R(8)+ZI*YJ
       R(9)=R(9)+ZI*ZJ
    enddo
    !
    CALL FROTU(R,EVA,DEVA,U,ZERO,QEVW,.FALSE.)
    !
    RETURN
  END SUBROUTINE FINDRU

  SUBROUTINE PRINTMAP(IDEMP)
    !-----------------------------------------------------------------------
    !  Print our map information and its density array
    !                        
  use stream
    INTEGER IDEMP,NDATA
    !
    IF(PRNLEV<3)RETURN
    WRITE(OUTU,10) IDEMP,EMAPID(IDEMP)
10  FORMAT(10X,'MAP ID: ',I3," MAP NAME: ",A)
    NDATA=LEMAPX(IDEMP)*LEMAPY(IDEMP)*LEMAPZ(IDEMP)
    CALL EMAPOUT(NDATA,MEMAPX(IDEMP),MEMAPY(IDEMP),MEMAPZ(IDEMP), &
         LEMAPX(IDEMP),LEMAPY(IDEMP),LEMAPZ(IDEMP), &
         DEMAPX(IDEMP),DEMAPY(IDEMP),DEMAPZ(IDEMP), &
         empgrd(IDEMP)%rho,empgrd(IDEMP)%ddrho,empgrd(IDEMP)%core)
    RETURN
  END SUBROUTINE PRINTMAP


  SUBROUTINE WRTEMAP(FNAME,UNITMAP,IDEMP,LCORE,LDDR,FMAP)
    !-----------------------------------------------------------------------
    !     This routine write a emap to a binary file
    !     in CCP4 electronic map  format
    !
  use memory
#if KEY_PARALLEL==1
  use parallel       
#endif
    !
    INTEGER UNITMAP,IDEMP
    CHARACTER*10 FMAP
    real(chm_real4),allocatable,dimension(:) :: RHOEMPI
    CHARACTER(len=*) FNAME
    LOGICAL LCORE,LDDR,ISMRC
    INTEGER I,NDATA,FLEN, UNIT
    type(emap_geom) :: geom

#if KEY_PARALLEL==1
      IF(MYNOD.NE.0)RETURN
#endif 
    UNIT=99
    ISMRC=.false.
    NDATA=LEMAPX(IDEMP)*LEMAPY(IDEMP)*LEMAPZ(IDEMP)
  ! remove any embedded double quotes from @param substitution
    FLEN = LEN(FNAME)
    DO I=1,FLEN
         IF (FNAME(I:I) == '"') THEN
            FNAME(I:) = FNAME(I+1:)
         ENDIF
    ENDDO
    FLEN = LEN(FNAME)
    IF(FLEN>0)THEN
 ! remove any embedded double quotes from @param substitution
      DO I=1,FLEN
         IF (FNAME(I:I) == '"') THEN
            FNAME(I:) = FNAME(I+1:)
         ENDIF
      ENDDO
!
      IF((INDEX(FNAME,'.ccp4')>0) .or.(INDEX(FNAME,'.CCP4')>0) .or. &
         (INDEX(FNAME,'.map')>0) .or.(INDEX(FNAME,'.MAP')>0))THEN
      ELSE IF((INDEX(FNAME,'.mrc')>0).or. (INDEX(FNAME,'.MRC')>0))THEN
        ISMRC=.TRUE.
      ELSE
        CALL WRNDIE(3,'<WRTEMAP>', &
          'Error: Map format is not supported or not specified!')
      ENDIF
#if KEY_PATHSCALE==0
      OPEN(UNIT,FILE=FNAME,ACCESS='STREAM',FORM='UNFORMATTED',STATUS='UNKNOWN')
#else
      OPEN(UNIT,FILE=FNAME,ACCESS='DIRECT',FORM='UNFORMATTED',STATUS='UNKNOWN',RECL=1)
#endif
    ELSE 
      IF(UNITMAP.LE.0)THEN
        IF(FLEN==0)THEN
          write(outu,*)' Map UNIT or filenmae must be specified!'
          CALL WRNDIE(3,'<WRTEMAP>', &
          'Error: Map file not opened or specified!')
        ENDIF
      ELSE
        UNIT=UNITMAP
        IF((INDEX(FMAP,'CCP4')>0) .or.(INDEX(FMAP,'MAP')>0))THEN
        ELSE IF(INDEX(FMAP,'MRC')>0)THEN
          ISMRC=.TRUE.
        ELSE
          CALL WRNDIE(3,'<WRTEMAP>', &
          'Error: Map format is not supported or not specified!')
        ENDIF
      ENDIF
    ENDIF
    IF(LCORE)THEN
       call chmalloc('emapsubs.src','WRTEMAP','RHOEMPI',NDATA,cr4=RHOEMPI)
         CALL MVCOREDT(NDATA,empgrd(IDEMP)%core,RHOEMPI)
         CALL WRTCCP4(UNIT,IDEMP,RHOEMPI,ISMRC)      
       call chmdealloc('emapsubs.src','WRTEMAP','RHOEMPI',NDATA,cr4=RHOEMPI)
    ELSE IF(LDDR)THEN
         CALL WRTCCP4(UNIT,IDEMP,EMPGRD(IDEMP)%DDRHO,ISMRC)      
    ELSE
         CALL WRTCCP4(UNIT,IDEMP,EMPGRD(IDEMP)%RHO,ISMRC) 
    ENDIF
    RETURN
  END SUBROUTINE WRTEMAP
#endif /* (emapmod)*/

end module emapmod

#if KEY_EMAP==1 /*emap_main*/


SUBROUTINE EMAPGEN(NAT,X,Y,Z,AMASS,ISLCT,IDEMP,DX,DY,DZ,RESO)
  !-----------------------------------------------------------------------
  !     This routine generate a electron density map from a set of atoms
  !
  use chm_kinds
  use dimens_fcm
  use number
  use exfunc
  use emapmod
  use memory
  implicit none
  !
  INTEGER NAT,ISLCT(*),IDEMP
  real(chm_real4),allocatable,dimension(:) :: RHOTMP1,RHOTMP2
  real(chm_real) X(*),Y(*),Z(*),AMASS(*),DX,DY,DZ,RESO
  real(chm_real) RHOCUT
  INTEGER I,NATI,NGRID,ICORE
  INTEGER MNX,MNY,MNZ,MXX,MXY,MXZ,IX,IY,IZ
  !
  MNX=100000
  MNY=100000
  MNZ=100000
  MXX=-100000
  MXY=-100000
  MXZ=-100000
  NATI=0
  DO I=1,NAT
     IF(ISLCT(I) == 1) THEN
        NATI=NATI+1
        IX=NINT(X(I)/DX)
        IY=NINT(Y(I)/DY)
        IZ=NINT(Z(I)/DZ)
        IF(MNX > IX)MNX=IX
        IF(MNY > IY)MNY=IY
        IF(MNZ > IZ)MNZ=IZ
        IF(MXX < IX)MXX=IX
        IF(MXY < IY)MXY=IY
        IF(MXZ < IZ)MXZ=IZ
     ENDIF
  ENDDO
  MNX=MNX-INT(TWO*RESO/DX)-1
  MNY=MNY-INT(TWO*RESO/DY)-1
  MNZ=MNZ-INT(TWO*RESO/DZ)-1
  MXX=MXX+INT(TWO*RESO/DX)+1
  MXY=MXY+INT(TWO*RESO/DY)+1
  MXZ=MXZ+INT(TWO*RESO/DZ)+1
  DEMAPX(IDEMP)=DX
  DEMAPY(IDEMP)=DY
  DEMAPZ(IDEMP)=DZ
  IF(MEMAPX(IDEMP)==-999999)MEMAPX(IDEMP)=MNX
  IF(MEMAPY(IDEMP)==-999999)MEMAPY(IDEMP)=MNY
  IF(MEMAPZ(IDEMP)==-999999)MEMAPZ(IDEMP)=MNZ
  IF(LEMAPX(IDEMP)==-999999)LEMAPX(IDEMP)=MXX-MNX+1
  IF(LEMAPY(IDEMP)==-999999)LEMAPY(IDEMP)=MXY-MNY+1
  IF(LEMAPZ(IDEMP)==-999999)LEMAPZ(IDEMP)=MXZ-MNZ+1
  MNX=MEMAPX(IDEMP)
  MNY=MEMAPY(IDEMP)
  MNZ=MEMAPZ(IDEMP)
  MXX=MNX+LEMAPX(IDEMP)-1
  MXY=MNY+LEMAPY(IDEMP)-1
  MXZ=MNZ+LEMAPZ(IDEMP)-1
  NGRID=(MXX-MNX+1)*(MXY-MNY+1)*(MXZ-MNZ+1)

  call chmalloc('emapsubs.src','EMAPGEN','empgrd.RHO',NGRID, &
       cr4p=empgrd(IDEMP)%rho)
  call chmalloc('emapsubs.src','EMAPGEN','empgrd.core',NGRID, &
       intgp=empgrd(IDEMP)%core)
  call chmalloc('emapsubs.src','EMAPGEN','empgre*idemp.ddrho', &
       NGRID,cr4p=empgrd(idemp)%ddrho)
  CALL EMAPINIT(IDEMP,ZERO)

  call chmalloc('emapsubs.src','EMAPGEN','RHOTMP1',NGRID,cr4=RHOTMP1)
  call chmalloc('emapsubs.src','EMAPGEN','RHOTMP2',NGRID,cr4=RHOTMP2)
  CALL XYZ2EMP(NAT,X,Y,Z,AMASS,ISLCT,MNX,MNY,MNZ, &
       MXX,MXY,MXZ,DX,DY,DZ, &
       empgrd(IDEMP)%rho,RHOTMP1,RHOTMP2,RESO)
  call chmdealloc('emapsubs.src','EMAPGEN','RHOTMP1',NGRID,cr4=RHOTMP1)
  call chmdealloc('emapsubs.src','EMAPGEN','RHOTMP2',NGRID,cr4=RHOTMP2)
  IF(PRNLEV > 2)THEN
     write(outu,1010)IDEMP
     WRITE(OUTU,1020)MEMAPX(IDEMP),MEMAPY(IDEMP),MEMAPZ(IDEMP)
     WRITE(OUTU,1030)LEMAPX(IDEMP),LEMAPY(IDEMP),LEMAPZ(IDEMP)
     WRITE(OUTU,1040)DEMAPX(IDEMP),DEMAPY(IDEMP),DEMAPZ(IDEMP)
  ENDIF
  RHOCUT=EMRCUT
  ICORE=EMICORE
  CALL EMAPDDR(IDEMP)
  CALL EMAPCORE(IDEMP,RHOCUT,ICORE)
  CALL EMAPSTAT(IDEMP)
!  Assign selected atoms as the reference atom set for the map 
  CALL EMAPREF(NAT,X,Y,Z,AMASS,ISLCT,IDEMP)
1010 FORMAT("Emap ",I3," is built from protein structure")
1020 FORMAT("MX,MY,MZ= ",3I6)
1030 FORMAT("LX,LY,LZ= ",3I6)
1040 FORMAT("DX,DY,DZ= ",3F10.4)
  RETURN
END SUBROUTINE EMAPGEN

SUBROUTINE XYZ2EMP(NAT,X,Y,Z,AMASS,ISLCT,MNX,MNY,MNZ, &
     MXX,MXY,MXZ,DX,DY,DZ,RHO,RHO1,RHO2,RESO)
  !-----------------------------------------------------------------------
  !     This routine to the generation of density from coordinates
  !
  use chm_kinds
  use dimens_fcm
  use number
  implicit none
  INTEGER NAT,NSEL,ISLCT(*)
  real(chm_real) X(*),Y(*),Z(*),AMASS(*)
  real(chm_real) DX,DY,DZ,RESO
  real(chm_real) TMASS,VARP,varmap,sigma,sigmap,bvalue,cvalue,kmsd
  real(chm_real4) RHO(*),RHO1(*),RHO2(*)
  real(chm_real) XI,YI,ZI,WEI,GX,GY,GZ,A,B,C,dsqu,DVAL
  INTEGER I,J,K,NATI,NGRID,IXYZ,IIXYZ,KXYZ,II,JJ,KK
  INTEGER MNX,MNY,MNZ,MXX,MXY,MXZ,NX,NY,NZ,KX,KY,KZ
  INTEGER MXS,MXE,MYS,MYE,MZS,MZE
  INTEGER X0,Y0,Z0,X1,Y1,Z1,NKC,NKER,XMIN,YMIN,ZMIN
  !
  NX=MXX-MNX+1
  NY=MXY-MNY+1
  NZ=MXZ-MNZ+1
  NGRID=NX*NY*NZ
  DO I=1,NGRID
     RHO(I)=ZERO
     RHO1(I)=ZERO
     RHO2(I)=ZERO
  END DO
  ! interpolate structure to protein map
  XMIN=MNX*DX
  YMIN=MNY*DY
  ZMIN=MNZ*DZ
  VARP=ZERO
  TMASS=ZERO
  NSEL=0
  loop100: DO I=1,NAT
     IF(ISLCT(I) /= 1) cycle loop100
     NSEL=NSEL+1
     XI=X(I)
     YI=Y(I)
     ZI=Z(I)
     WEI=AMASS(I)
     TMASS=TMASS+WEI
     GX=(XI-XMIN)/DX+ONE
     GY=(YI-YMIN)/DY+ONE
     GZ=(ZI-ZMIN)/DZ+ONE
     X0=INT(GX)
     Y0=INT(GY)
     Z0=INT(GZ)
     X1=X0+1
     Y1=Y0+1
     Z1=Z0+1
     IF(X0<1.OR.Y0<1.OR.Z0<1)cycle loop100
     IF(X1>NX.OR.Y1>NY.OR.Z1>NZ)cycle loop100
     A=X1-GX
     B=Y1-GY
     C=Z1-GZ
     IXYZ=X0+NX*(Y0-1+NY*(Z0-1))
     RHO1(IXYZ)=RHO1(IXYZ)+WEI*A*B*C
     VARP=VARP+WEI*A*B*C*((ONE-A)*(ONE-A)+ &
          (ONE-B)*(ONE-B)+(ONE-C)*(ONE-C))
     IXYZ=X0+NX*(Y0-1+NY*(Z1-1))
     RHO1(IXYZ)=RHO1(IXYZ)+WEI*A*B*(ONE-C)
     VARP=VARP+WEI*A*B*(ONE-C)*((ONE-A)*(ONE-A)+(ONE-B)*(ONE-B)+C*C)
     IXYZ=X0+NX*(Y1-1+NY*(Z0-1))
     RHO1(IXYZ)=RHO1(IXYZ)+WEI*A*(ONE-B)*C
     VARP=VARP+WEI*A*(ONE-B)*C*((ONE-A)*(ONE-A)+B*B+(ONE-C)*(ONE-C))
     IXYZ=X1+NX*(Y0-1+NY*(Z0-1))
     RHO1(IXYZ)=RHO1(IXYZ)+WEI*(ONE-A)*B*C
     VARP=VARP+WEI*(ONE-A)*B*C*(A*A+(ONE-B)*(ONE-B)+(ONE-C)*(ONE-C))
     IXYZ=X0+NX*(Y1-1+NY*(Z1-1))
     RHO1(IXYZ)=RHO1(IXYZ)+WEI*A*(ONE-B)*(ONE-C)
     VARP=VARP+WEI*A*(ONE-B)*(ONE-C)*((ONE-A)*(ONE-A)+B*B+C*C)
     IXYZ=X1+NX*(Y1-1+NY*(Z0-1))
     RHO1(IXYZ)=RHO1(IXYZ)+WEI*(ONE-A)*(ONE-B)*C
     VARP=VARP+WEI*(ONE-A)*(ONE-B)*C*(A*A+B*B+(ONE-C)*(ONE-C))
     IXYZ=X1+NX*(Y0-1+NY*(Z1-1))
     RHO1(IXYZ)=RHO1(IXYZ)+WEI*(ONE-A)*B*(ONE-C)
     VARP=VARP+WEI*(ONE-A)*B*(ONE-C)*(A*A+(ONE-B)*(ONE-B)+C*C)
     IXYZ=X1+NX*(Y1-1+NY*(Z1-1))
     RHO1(IXYZ)=RHO1(IXYZ)+WEI*(ONE-A)*(ONE-B)*(ONE-C)
     VARP=VARP+WEI*(ONE-A)*(ONE-B)*(ONE-C)*(A*A+B*B+C*C)
  enddo loop100
  IF(NSEL==0)RETURN
  VARP=VARP/TMASS
  !  Blur the kernel maps
  sigma = reso/(-2.0)
  kmsd = sigma**2/(DX*DY*DZ)**(TWO/THREE)
  varmap =kmsd - varp
  if (varmap  <  0) CALL WRNDIE(0,'<XYZ2EMAP>', &
       'Error: lattice smoothing exceeds kernel size')
  sigmap = sqrt(varmap/3.0)
  NKC = INT(3*sigmap)+2
  NKER=2*NKC-1
  bvalue = -ONE/(TWO*sigmap*sigmap)
  cvalue = NINE*sigmap*sigmap
  DO K=1,NKER
     DO J=1,NKER
        DO I=1,NKER
           IXYZ=I+NX*(J-1+NY*(K-1))
           dsqu=(I-NKC)*(I-NKC)+(J-NKC)*(J-NKC)+(K-NKC)*(K-NKC)
           if (dsqu < cvalue)RHO2(IXYZ) = exp(dsqu * bvalue)
        ENDDO
     ENDDO
  ENDDO
  !  Build up the EMAP
  DO K=1,NZ
     DO J=1,NY
        DO I=1,NX
           IXYZ=I+NX*(J-1+NY*(K-1))
           DVAL=RHO1(IXYZ)
           if (dval > RSMALL)THEN
              loop510: DO KZ=1,NKER
                 KK=K+KZ-NKC
                 IF(KK*(NZ-KK+1) <= 0)cycle loop510
                 loop520: DO KY=1,NKER
                    JJ=J+KY-NKC
                    IF(JJ*(NY-JJ+1) <= 0)cycle loop520
                    loop530: DO  KX=1,NKER
                       II=I+KX-NKC
                       IF(II*(NX-II+1) <= 0) cycle loop530
                       IIXYZ=II+NX*(JJ-1+NY*(KK-1))
                       KXYZ=KX+NX*(KY-1+NY*(KZ-1))
                       RHO(IIXYZ)=RHO(IIXYZ)+DVAL*RHO2(KXYZ)
                    enddo loop530
                 enddo loop520
              enddo loop510
           ENDIF
        ENDDO
     ENDDO
  ENDDO
  RETURN
END SUBROUTINE XYZ2EMP


SUBROUTINE EMAPDBIN(NF)
  !-----------------------------------------------------------------------
  !     This routine read matrix of residue binding prepensities
  !
  use chm_kinds
  use dimens_fcm
  use number
  use stream
  use exfunc
  use emapmod
  use psf
  use string
  implicit none
  INTEGER NF,LINELN,WORDLN
  integer,PARAMETER :: MAXLINE=200,MAXWD=20
  character(len=200) LINE
  character(len=20) WORD
  character(len=4) RESI1,RESI3
  INTEGER I,J,K,IRES,JRES
  INTEGER IDXROW,IDXCOL(50),NDBDATA
  LOGICAL QID,QBIND
  !MH08: moved to iniall ->      DATA NDBRES/0/
  !
  IF(NF < 0)THEN
     IF(NDBRES <= 0)CALL WRNDIE(0,'<EMAPINPDB>', &
          'Database matrix  must be load first by specifying DBUNIT ')
     RETURN
  ENDIF
  IRES=0
10 READ(NF,'(A)',END=100)LINE
  IF(LINE(1:1) == "#")GOTO 10
  LINELN=LEN(LINE)
  WORD=NEXT20(LINE,LINELN)
  WORDLN=LEN(WORD)
  IF(WORD == "DBEND")THEN
     GOTO 100
  ELSE IF(WORD == ' ')THEN
     IF(QID)THEN
        QID=.FALSE.
        NDBRES=IRES
        IF(NDBRES > MDBRES)CALL WRNDIE(0,'<EMAPINPDB>', &
             'Number of matrix residue EXCEEDS maximum (50)!')
     ENDIF
     IF(QBIND)THEN
        QBIND=.FALSE.
        IF(IRES < NDBDATA)CALL WRNDIE(0,'<EMAPINPDB>', &
             'Number of matrix residue is fewer than claimed')
     ENDIF
  ELSE IF(WORD == "RESID")THEN
     QID=.TRUE.
     IRES=0
  ELSE IF(WORD == "BIND")THEN
     QBIND=.TRUE.
     IRES=0
210  WORD=NEXT20(LINE,LINELN)
     IF(WORD == ' ')GOTO 230
     DO JRES=1,NDBRES
        IF(WORD == DBRESN1(JRES).OR. &
             WORD == DBRESN3(JRES))GOTO 220
     ENDDO
     CALL WRNDIE(0,'<EMAPINPDB>', &
          'Colume residue is not recognized: '//RESI3(1:J)//'!!')  
220  CONTINUE
     IRES=IRES+1
     IDXCOL(IRES)=DBRESNO(JRES)
     GOTO 210
230  CONTINUE
     NDBDATA=IRES
     IRES=0
  ELSE IF(QID)THEN
     IRES=IRES+1
     DBRESNO(IRES)=NEXTI(WORD,WORDLN)
     DBRESN1(IRES)=NEXTA4(LINE,LINELN)
     DBRESN3(IRES)=NEXTA4(LINE,LINELN)
  ELSE IF(QBIND)THEN
     IRES=IRES+1
     IF(IRES > NDBRES)CALL WRNDIE(0,'<EMAPINPDB>', &
          'Number of matrix residue is more than claimed')
     RESI3=NEXTA4(WORD,WORDLN)
     DO JRES=1,NDBRES
        IF(RESI3 == DBRESN1(JRES).OR. &
             RESI3 == DBRESN3(JRES))GOTO 310
     ENDDO
     CALL WRNDIE(0,'<EMAPINPDB>', &
          'Row residue is not recognized: '//RESI3(1:J)//'!!') 
310  CONTINUE
     IDXROW=DBRESNO(JRES)
     DO JRES=1,NDBDATA
        PMATRIX(IDXROW,IDXCOL(JRES))=NEXTF(LINE,LINELN)
     ENDDO
  ENDIF
  GOTO 10
100 CONTINUE
  CLOSE(NF)
  IF(PRNLEV > 5)THEN
     WRITE(OUTU,'(/"RESID    RESI1   RESI3")')
     DO I=1,NDBRES
        WRITE(OUTU,'(I6,4X,A1,4X,A4)')DBRESNO(I),DBRESN1(I),DBRESN3(I)
     ENDDO
     WRITE(OUTU,'(/"BIND",50(4X,I2))')(IDXCOL(I),I=1,NDBDATA)
     DO I=1,NDBDATA
        J=IDXCOL(I)
        WRITE(OUTU,'(I4,50F6.2)')J,(PMATRIX(IRES,J),IRES=1,NDBDATA)
     ENDDO
  ENDIF
  RETURN 
END SUBROUTINE EMAPDBIN

SUBROUTINE EMAPSCORE(SCORE,X,Y,Z,ISLCT1,ISLCT2, &
     IDXRES,LPRINT)
  !-----------------------------------------------------------------------
  !     This routine to the generation of density from coordinates
  !
  use chm_kinds
  use dimens_fcm
  use number
  use emapmod
  use psf
  implicit none
  INTEGER ISLCT1(*),ISLCT2(*),IDXRES(*)
  real(chm_real) X(*),Y(*),Z(*)
  real(chm_real) XI,YI,ZI,XIJ,YIJ,ZIJ,RIJ2,RBIND2,RMIN2,SCORE
  INTEGER I,J,IRES,JRES,IDX,IB1,IB2,JB1,JB2,IDIRES,IDJRES
  character(len=4) RESI
  LOGICAL LPRINT
  !
  RBIND2=REMBIND*REMBIND
  loop10: DO IRES=1,NRES
     RESI=RES(IRES)
     DO IDX=1,NDBRES
        IF(RESI == DBRESN3(IDX))THEN
           IDXRES(IRES)=DBRESNO(IDX)
           cycle loop10
        ENDIF
     ENDDO
     CALL WRNDIE(0,'<EMAPSCORE>', &
          'Residue '//resi//' does not exist in database matrix!')
  enddo loop10
  IB2=0
  loop100: DO IRES=1,NRES
     IB1=IB2+1
     IB2=IBASE(IRES+1)
     DO I=IB1,IB2
        IF(ISLCT1(I) == 1) exit loop100
     ENDDO
     cycle loop100

     IDIRES=IDXRES(IRES)
     JB2=0
     loop200: DO JRES=1,NRES
        JB1=JB2+1
        JB2=IBASE(JRES+1)
        DO J=JB1,JB2
           IF(ISLCT2(J) == 1) goto 220
        ENDDO
        cycle loop200
220     CONTINUE
        RMIN2=RBIG
        DO I=IB1,IB2
           XI=X(I)
           YI=Y(I)
           ZI=Z(I)
           DO J=JB1,JB2
              XIJ=X(J)-XI
              YIJ=Y(J)-YI
              ZIJ=Z(J)-ZI
              RIJ2=XIJ*XIJ+YIJ*YIJ+ZIJ*ZIJ
              IF(RIJ2 <= RMIN2)RMIN2=RIJ2
           ENDDO
        ENDDO
        RMIN2=RMIN2/RBIND2
        IDJRES=IDXRES(JRES)
        SCORE=SCORE+PMATRIX(IDIRES,IDJRES)*ONE/(ONE+RMIN2*RMIN2)
     enddo loop200
  enddo loop100
  RETURN
END SUBROUTINE EMAPSCORE

SUBROUTINE EMAPREF(NAT,X,Y,Z,AMASS,ISLCT,IDEMP)
  !-----------------------------------------------------------------------
  !   This routine assign a set of atoms as reference to a map
  !
  use chm_kinds
  use dimens_fcm
  use number
  use stream
  use exfunc
  use emapmod
  use memory
  implicit none
  !
  INTEGER NAT,ISLCT(*),IDEMP
  real(chm_real) X(*),Y(*),Z(*),AMASS(*),DX,DY,DZ,RESO
  real(chm_real) RHOCUT
  INTEGER I,NATI,NGRID,ICORE
  INTEGER MNX,MNY,MNZ,MXX,MXY,MXZ,IX,IY,IZ
  !
  NATI=NATEMAP(IDEMP)
  IF(NATI > 0)THEN
     call chmdealloc('emapsubs.src','EMAPREF', &
        'empcrd(IDEMP)%iatom ',NATI,intgp=empcrd(IDEMP)%iatom )
     call chmdealloc('emapsubs.src','EMAPREF','empcrd(IDEMP)%x ',NATI,crlp=empcrd(IDEMP)%x )
     call chmdealloc('emapsubs.src','EMAPREF','empcrd(IDEMP)%y ',NATI,crlp=empcrd(IDEMP)%y )
     call chmdealloc('emapsubs.src','EMAPREF','empcrd(IDEMP)%z ',NATI,crlp=empcrd(IDEMP)%z )
  ENDIF
  NATI=0
  DO I=1,NAT
     IF(ISLCT(I) == 1) THEN
        NATI=NATI+1
     ENDIF
  ENDDO
  call chmalloc('emapsubs.src','EMAPREF',  &
   'empcrd(IDEMP)%iatom ',NATI,intgp=empcrd(IDEMP)%iatom )
  call chmalloc('emapsubs.src','EMAPREF','empcrd(IDEMP)%x ',NATI,crlp=empcrd(IDEMP)%x )
  call chmalloc('emapsubs.src','EMAPREF','empcrd(IDEMP)%y ',NATI,crlp=empcrd(IDEMP)%y )
  call chmalloc('emapsubs.src','EMAPREF','empcrd(IDEMP)%z ',NATI,crlp=empcrd(IDEMP)%z )
  NATEMAP(IDEMP)=NATI
  CALL SETEMREF(NAT,X,Y,Z,AMASS,ISLCT,empcrd(IDEMP)%iatom, &
       EMMASS(IDEMP),EMINER(IDEMP),CEMAPX(IDEMP),CEMAPY(IDEMP),CEMAPZ(IDEMP), &
       empcrd(IDEMP)%x,empcrd(IDEMP)%y,empcrd(IDEMP)%z)
  RETURN
END SUBROUTINE EMAPREF

SUBROUTINE SETEMREF(NAT,X,Y,Z,AMASS,ISLCT,IDX,TMASS,TINER, &
     CX,CY,CZ,XR,YR,ZR)
  !-----------------------------------------------------------------------
  use chm_kinds
  use dimens_fcm
  use number
  implicit none
  INTEGER NAT,ISLCT(*),IDX(*)
  real(chm_real) X(*),Y(*),Z(*),XR(*),YR(*),ZR(*),AMASS(*)
  real(chm_real) CX,CY,CZ,TMASS,TINER
  real(chm_real) XI,YI,ZI,AMASSI
  INTEGER I,NATI
  !
  NATI=0
  CX=ZERO
  CY=ZERO
  CZ=ZERO
  TMASS=ZERO
  TINER=ZERO
  loop100: DO I=1,NAT
     IF(ISLCT(I) /= 1) cycle loop100
     NATI=NATI+1
     IDX(NATI)=I
     XI=X(I)
     YI=Y(I)
     ZI=Z(I)
     AMASSI=AMASS(I)
     XR(NATI)=XI
     YR(NATI)=YI
     ZR(NATI)=ZI
     CX=CX+AMASSI*XI
     CY=CY+AMASSI*YI
     CZ=CZ+AMASSI*ZI
     TMASS=TMASS+AMASSI
     TINER=TINER+AMASSI*(XI*XI+YI*YI+ZI*ZI)
  enddo loop100
  CX=CX/TMASS
  CY=CY/TMASS
  CZ=CZ/TMASS
  TINER=TINER-TMASS*(CX*CX+CY*CY+CZ*CZ)
  RETURN
END SUBROUTINE SETEMREF

SUBROUTINE EMAPASN(NAT,X,Y,Z,AMASS,ISLCT,IDEMP,IDRIG)
  !-----------------------------------------------------------------------
  !     This routine ASSIGN a rigid segment to a electronic density map
  !     EMAPID and calculate the position of the rigid segment related to
  !     the origin of EMAPID
  !
  use chm_kinds
  use dimens_fcm
  use number
  use exfunc
  use emapmod
  implicit none
  INTEGER NAT,NATI,ISLCT(*),IDEMP,IDRIG
  real(chm_real) X(*),Y(*),Z(*),AMASS(*)
  !
  IDEMRIG(IDRIG)=IDEMP
  !
  IF(NATEMAP(IDEMP) > 0)THEN
     !       FIND TRANSLATION VECTOR AND ROTATION MATRIX TO SUPERIMPOSE 
     !       THE EMAP ATOMS TO  THE RIGID STRUCTURE
     CALL FINDRU(NAT,X,Y,Z,ISLCT, &
          NATEMAP(IDEMP), &
          empcrd(IDEMP)%x, &
          empcrd(IDEMP)%y, &
          empcrd(IDEMP)%z, &
          TEMRIG(1,IDRIG),REMRIG(1,IDRIG))
  ELSE
     !       Assign selected atoms as the reference atom set for the map 
     CALL EMAPREF(NAT,X,Y,Z,AMASS,ISLCT,IDEMP)
     !       assign 0 TRANSLATION VECTOR AND unit ROTATION MATRIX 
     CALL EMAPRIGINI(IDRIG)
  ENDIF
  RETURN
END SUBROUTINE EMAPASN

SUBROUTINE EMAPDUP(ID1,ID2)
  !-----------------------------------------------------------------------
  !     This routeine duplicate a map
  !
  use chm_kinds
  use dimens_fcm
  use number
  use exfunc
  use emapmod
  use memory
  implicit none
  integer ID1,ID2,NDATA,NATI
  !
  NDATA=LEMAPX(ID1)*LEMAPY(ID1)*LEMAPZ(ID1)
  NATI=NATEMAP(ID1)
  NATEMAP(ID2)=NATI
  IF(NATI > 0)THEN
     call chmalloc('emapsubs.src','EMAPDUP','empcrd(ID2)%x ',NATI,crlp=empcrd(ID2)%x )
     call chmalloc('emapsubs.src','EMAPDUP','empcrd(ID2)%y ',NATI,crlp=empcrd(ID2)%y )
     call chmalloc('emapsubs.src','EMAPDUP','empcrd(ID2)%z ',NATI,crlp=empcrd(ID2)%z )
  ENDIF
  call chmalloc('emapsubs.src','EMAPDUP','empgrd(ID2)%rho',NDATA, &
       cr4p=empgrd(ID2)%rho)
  call chmalloc('emapsubs.src','EMAPDUP','empgrd(ID2)%ddrho',NDATA, &
       cr4p=empgrd(ID2)%ddrho)
  call chmalloc('emapsubs.src','EMAPDUP','empgrd(ID2)%core',NDATA, &
       intgp=empgrd(ID2)%core)
  CALL EMAPDUP1(ID1,ID2,NATI,NDATA, &
       empgrd(ID1)%rho, empgrd(ID1)%ddrho, empgrd(ID1)%core, &
       empgrd(ID2)%rho, empgrd(ID2)%ddrho, empgrd(ID2)%core, &
       empcrd(ID1)%x,empcrd(ID1)%y,empcrd(ID1)%z, &
       empcrd(ID2)%x,empcrd(ID2)%y,empcrd(ID2)%z)
  return
end SUBROUTINE EMAPDUP

SUBROUTINE EMAPDUP1(ID1,ID2,NATI,NDATA,RHO1,DDR1,CORE1, &
     RHO2,DDR2,CORE2,XAT1,YAT1,ZAT1,XAT2,YAT2,ZAT2)
  !-----------------------------------------------------------------------
  use chm_kinds
  use dimens_fcm
  use number
  use emapmod
  implicit none
  integer NATI,NDATA,ID1,ID2,I,CORE1(*),CORE2(*)
  real(chm_real4) RHO1(*),RHO2(*),DDR1(*),DDR2(*)
  real(chm_real) XAT1(*),YAT1(*),ZAT1(*),XAT2(*),YAT2(*),ZAT2(*)
  !
  MEMAPX(ID2)=MEMAPX(ID1)
  MEMAPY(ID2)=MEMAPY(ID1)
  MEMAPZ(ID2)=MEMAPZ(ID1)
  MNCOREX(ID2)=MNCOREX(ID1)
  MNCOREY(ID2)=MNCOREY(ID1)
  MNCOREZ(ID2)=MNCOREZ(ID1)
  MXCOREX(ID2)=MXCOREX(ID1)
  MXCOREY(ID2)=MXCOREY(ID1)
  MXCOREZ(ID2)=MXCOREZ(ID1)
  LEMAPX(ID2)=LEMAPX(ID1)
  LEMAPY(ID2)=LEMAPY(ID1)
  LEMAPZ(ID2)=LEMAPZ(ID1)
  DEMAPX(ID2)=DEMAPX(ID1)
  DEMAPY(ID2)=DEMAPY(ID1)
  DEMAPZ(ID2)=DEMAPZ(ID1)
  AEMAPX(ID2)=AEMAPX(ID1)
  AEMAPY(ID2)=AEMAPY(ID1)
  AEMAPZ(ID2)=AEMAPZ(ID1)
  CEMAPX(ID2)=CEMAPX(ID1)
  CEMAPY(ID2)=CEMAPY(ID1)
  CEMAPZ(ID2)=CEMAPZ(ID1)
  AREMAP(ID2)=AREMAP(ID1)
  RREMAP(ID2)=RREMAP(ID1)
  ADEMAP(ID2)=ADEMAP(ID1)
  DDEMAP(ID2)=DDEMAP(ID1)
  ACEMAP(ID2)=ACEMAP(ID1)
  CCEMAP(ID2)=CCEMAP(ID1)
  RCEMAP(ID2)=RCEMAP(ID1)
  DCEMAP(ID2)=DCEMAP(ID1)
  NCREMAP(ID2)=NCREMAP(ID1)
  LOWEMAP(:,ID2)=LOWEMAP(:,ID1)
  UPEMAP(:,ID2)=UPEMAP(:,ID1)
  TRNEMAP(:,ID2)=TRNEMAP(:,ID1)
  ROTEMAP(:,ID2)=ROTEMAP(:,ID1)
  do i=1,NDATA
     RHO2(I)=RHO1(I)
     DDR2(I)=DDR1(I)
     CORE2(I)=CORE1(I)
  ENDDO
  do i=1,NATI
     XAT2(I)=XAT1(I)
     YAT2(I)=YAT1(I)
     ZAT2(I)=ZAT1(I)
  ENDDO
  RETURN
END SUBROUTINE EMAPDUP1

SUBROUTINE EMAPREDUCE(IDEMP0,IDEMPS,IDEMP)
  !-----------------------------------------------------------------------
  !     This routine perform map reduction
  !
  use chm_kinds
  use dimens_fcm
  use number
  use emapmod
  implicit none
  integer IDEMP0,IDEMPS,IDEMP
  !
  IF(LEMAPX(IDEMP0) /= LEMAPX(IDEMP).OR. &
       LEMAPY(IDEMP0) /= LEMAPY(IDEMP).OR. &
       LEMAPZ(IDEMP0) /= LEMAPZ(IDEMP))CALL WRNDIE(0, &
       '<EMAPREDUCE>','The target MAP has different dimension!')
  CALL EMAPRED(IDEMP,IDEMPS,IDEMP,empgrd(IDEMP0)%rho, &
       empgrd(IDEMP0)%ddrho,empgrd(IDEMP0)%core, &
       empgrd(IDEMPS)%rho, &
       empgrd(IDEMPS)%ddrho, &
       empgrd(IDEMPS)%core, &
       empgrd(IDEMP)%rho, &
       empgrd(IDEMP)%ddrho, &
       empgrd(IDEMP)%core)
  CALL EMAPSTAT(IDEMP)
  RETURN
END SUBROUTINE EMAPREDUCE

SUBROUTINE EMAPRED(IDEMP0,IDEMP,IDEMPT,RHO0,DDR0,CORE0, &
     RHO,DDR,CORE,RHOT,DDRT,CORET)
  !-----------------------------------------------------------------------
  use chm_kinds
  use dimens_fcm
  use number
  use stream
  use emapmod
  implicit none
  INTEGER NDATA,CORE0(*),CORE(*),CORET(*),NRED
  INTEGER IDEMP0,IDEMP,IDEMPT
  INTEGER MTX,MTY,MTZ,LTX,LTY,LTZ,MX,MY,MZ,LX,LY,LZ
  INTEGER I,J,K,I0,J0,K0,IXYZ,IXYZ0
  real(chm_real) DTX,DTY,DTZ,DX,DY,DZ
  real(chm_real) RATIO,CI,CTI,ACW,ACR,RAVE
  real(chm_real4) RHO0(*),RHO(*),RHOT(*),DDR0(*),DDR(*),DDRT(*),RHOI
  !
  MTX=MEMAPX(IDEMP0)
  MTY=MEMAPY(IDEMP0)
  MTZ=MEMAPZ(IDEMP0)
  LTX=LEMAPX(IDEMP0)
  LTY=LEMAPY(IDEMP0)
  LTZ=LEMAPZ(IDEMP0)
  DTX=DEMAPX(IDEMP0)
  DTY=DEMAPY(IDEMP0)
  DTZ=DEMAPZ(IDEMP0)
  MX=MEMAPX(IDEMP)
  MY=MEMAPY(IDEMP)
  MZ=MEMAPZ(IDEMP)
  LX=LEMAPX(IDEMP)
  LY=LEMAPY(IDEMP)
  LZ=LEMAPZ(IDEMP)
  DX=DEMAPX(IDEMP)
  DY=DEMAPY(IDEMP)
  DZ=DEMAPZ(IDEMP)
  NDATA=LTX*LTY*LTZ
  ACW=ZERO
  ACR=ZERO
  DO I=1,NDATA
     !        CTI=CORET(I)
     CI=CORE(I)
     IF(CI > ZERO)THEN
        ACW=ACW+ONE
        !          RATIO=CTI/CI
        ACR=ACR+CI 
     ENDIF
  ENDDO
  RAVE=ONE
  IF(ACW > ZERO)RAVE=ACR/ACW
  NRED=0
  DO I0=0,LTX-1
     I=NINT((MTX+I0)*DTX/DX)-MX
     DO J0=0,LTY-1
        J=NINT((MTY+J0)*DTY/DY)-MY
        DO K0=0,LTZ-1
           K=NINT((MTZ+K0)*DTZ/DZ)-MZ
           IF(I >= 0.AND.I < LX.AND. &
                J >= 0.AND.J < LY.AND. &
                K >= 0.AND.K < LZ)THEN
              IXYZ=I+1+LX*(J+LY*K)
              CI=CORE(IXYZ)
           ELSE
              CI=ZERO
           ENDIF
           IXYZ0=I0+1+LTX*(J0+LTY*K0)
           CTI=CORE0(IXYZ0)
           RATIO=ONE
           IF(CI > RSMALL)THEN
              RATIO=ZERO
              NRED=NRED+1
           ENDIF
           RHOT(IXYZ0)=RHO0(IXYZ0)*RATIO
           DDRT(IXYZ0)=DDR0(IXYZ0)*RATIO
           CORET(IXYZ0)=CTI
        ENDDO
     ENDDO
  ENDDO
  MEMAPX(IDEMPT)=MTX
  MEMAPY(IDEMPT)=MTY
  MEMAPZ(IDEMPT)=MTZ
  LEMAPX(IDEMPT)=LTX
  LEMAPY(IDEMPT)=LTY
  LEMAPZ(IDEMPT)=LTZ
  DEMAPX(IDEMPT)=DTX
  DEMAPY(IDEMPT)=DTY
  DEMAPZ(IDEMPT)=DTZ
  WRITE(OUTU,'("NCORE,NRED,CAVE:",2I10,F10.5)')NINT(ACW),NRED,RAVE
  RETURN
END SUBROUTINE EMAPRED

SUBROUTINE EMAPADD(IDEMP,IDRIG)
  !-----------------------------------------------------------------------
  !     This routine add a rigid domain to a map
  !
  use chm_kinds
  use dimens_fcm
  use number
  use emapmod
  implicit none
  integer IDEMP,IDEMP1,IDRIG
  !
  IDEMP1=IDEMRIG(IDRIG)
  CALL EMAPADDR(IDEMP,IDEMP1, &
       empgrd(IDEMP)%rho, &
       empgrd(IDEMP)%ddrho, &
       empgrd(IDEMP)%core, &
       empgrd(IDEMP1)%rho, &
       empgrd(IDEMP1)%ddrho, &
       empgrd(IDEMP1)%core, &
       TEMRIG(1,IDRIG),REMRIG(1,IDRIG),ONE)
  RETURN
END SUBROUTINE EMAPADD

SUBROUTINE EMAPADDR(IDEMPT,IDEMP,RHOT,DDRT,CORET, &
     RHO,DDR,CORE,TR,U,ADDC)
  !-----------------------------------------------------------------------
  !  Rotat IDEMPT back to IDEMP and add the density of IDEMP 
  use chm_kinds
  use dimens_fcm
  use number
  use emapmod
  implicit none
  INTEGER I,J,K,II,JJ,KK,CORE(*),CORET(*),COREI,IADDC
  INTEGER IID,JJD,KKD,MMI,MMJ,MMK
  integer IDEMP,IDEMPT,MTX,MTY,MTZ,LTX,LTY,LTZ,LX,LY,LZ
  INTEGER IM,JM,KM,IDX,IDY,IDZ,IDXT,IXYZ
  integer MTXE,MTYE,MTZE,MXS,MYS,MZS,MXE,MYE,MZE
  real(chm_real) DTX,DTY,DTZ,DX,DY,DZ,CX,CY,CZ,ADDC
  real(chm_real) RI,RJ,RK,RI1,RJ1,RK1,TR(3),U(3,3)
  real(chm_real4) RHO(*),DDR(*),RHOT(*),DDRT(*),RHOI
  IADDC=NINT(ADDC)
  MTX=MEMAPX(IDEMPT)
  MTY=MEMAPY(IDEMPT)
  MTZ=MEMAPZ(IDEMPT)
  LTX=LEMAPX(IDEMPT)
  LTY=LEMAPY(IDEMPT)
  LTZ=LEMAPZ(IDEMPT)
  DTX=DEMAPX(IDEMPT)
  DTY=DEMAPY(IDEMPT)
  DTZ=DEMAPZ(IDEMPT)
  LX=LEMAPX(IDEMP)
  LY=LEMAPY(IDEMP)
  LZ=LEMAPZ(IDEMP)
  MTXE=MTX+LTX-1
  MTYE=MTY+LTY-1
  MTZE=MTZ+LTZ-1
  MXS=MEMAPX(IDEMP)
  MYS=MEMAPY(IDEMP)
  MZS=MEMAPZ(IDEMP)
  MXE=MXS+LX-1
  MYE=MYS+LY-1
  MZE=MZS+LZ-1
  DX=DEMAPX(IDEMP)
  DY=DEMAPY(IDEMP)
  DZ=DEMAPZ(IDEMP)
  CX=CEMAPX(IDEMP)
  CY=CEMAPY(IDEMP)
  CZ=CEMAPZ(IDEMP)
  DO K=MTZ,MTZE
     RK=K*DTZ-TR(3)-CZ
     DO J=MTY,MTYE
        RJ=J*DTY-TR(2)-CY
        loop100: DO I=MTX,MTXE
           IDX=I-MTX+1+LTX*(J-MTY+LTY*(K-MTZ))
           RI=I*DTX-TR(1)-CX
           II=NINT((U(1,1)*RI+U(2,1)*RJ+U(3,1)*RK+CX)/DX)-MXS
           IF(II*(LX-II-1) < 0)cycle loop100
           JJ=NINT((U(1,2)*RI+U(2,2)*RJ+U(3,2)*RK+CY)/DY)-MYS
           IF(JJ*(LY-JJ-1) < 0) cycle loop100
           KK=NINT((U(1,3)*RI+U(2,3)*RJ+U(3,3)*RK+CZ)/DZ)-MZS
           IF(KK*(LZ-KK-1) < 0) cycle loop100
           IXYZ=II+1+LX*(JJ+LY*KK)
           RHOT(IDX)=RHOT(IDX)+ADDC*RHO(IXYZ)
           DDRT(IDX)=DDRT(IDX)+ADDC*DDR(IXYZ)
           COREI=CORET(IDX)+IADDC*CORE(IXYZ)
           IF(COREI < 0)COREI=0
           CORET(IDX)=COREI
        ENDDO loop100
     ENDDO
  ENDDO
  RETURN
END SUBROUTINE EMAPADDR

SUBROUTINE EMAPCORE(IDEMP,RHOCUT,ICORE)
  !-----------------------------------------------------------------------
  !     This routine generate the core-index
  !     ICORE=1:  Core will be built according to density
  !     ICORE=2:  Core will be built according to Laplacian density
  !
  use chm_kinds
  use dimens_fcm
  use number
  use stream
  !
  use emapmod
  implicit none
  INTEGER IDEMP,LTX,LTY,LTZ,NCORE,NN,ICORE,NDATA
  real(chm_real) RHOCUT
  !
  LTX=LEMAPX(IDEMP)
  LTY=LEMAPY(IDEMP)
  LTZ=LEMAPZ(IDEMP)
  NDATA=LTX*LTY*LTZ
  CALL INITCORE(NDATA,empgrd(IDEMP)%core)
  RHOCUT=EMRCUT
  IF(ICORE == 1)THEN
     CALL FINDRCORE(1,1,1,ltx,lty,ltz,rhocut, &
          empgrd(IDEMP)%rho,empgrd(IDEMP)%core, &
          NCREMAP(IDEMP))
  ELSE IF(ICORE == 2)THEN
     CALL FINDDCORE(1,1,1,ltx,lty,ltz,rhocut,empgrd(IDEMP)%rho, &
          empgrd(IDEMP)%ddrho,empgrd(IDEMP)%core, &
          NCREMAP(IDEMP))
  ELSE
     CALL WRNDIE(0,'<EMAPCORE>','ICORE should be 1 or 2')
  ENDIF
  NCORE=NCREMAP(IDEMP)
  if(prnlev>3)write(outu,1010)NCREMAP(IDEMP),RHOCUT
1010 FORMAT(I8, " CORE points are found, RHOCUT=",F10.6)
  CALL BUILDCORE(ltx,lty,ltz,empgrd(IDEMP)%core)
  RETURN
END SUBROUTINE EMAPCORE

subroutine findrcore(i0,j0,k0,nx,ny,nz,rhocut, &
     rho,core,NCORE)
  !-----------------------------------------------------------------------
  !     define core=0 for outside grid with rho<rhocut
  use chm_kinds
  use number
  implicit none
  INTEGER nx,ny,nz,NDATA,NCORE,NCOREP,core(*)
  INTEGER I0,J0,K0,I,J,K,II,JJ,KK
  real(chm_real4) rho(*),RHOI
  real(chm_real) rhocut
  INTEGER IS,IE,JS,JE,KS,KE,IRHO
  !
  DO I=I0,NX
     DO J=J0,NY
        irho=I+(J-1)*nx
        CORE(IRHO)=0
        irho=I+(J-1+(nz-1)*ny)*nx
        CORE(IRHO)=0
     ENDDO
  ENDDO
  DO I=I0,NX
     DO K=K0,NZ
        irho=I+((K-1)*ny)*nx
        CORE(IRHO)=0
        irho=I+(NY-1+(K-1)*ny)*nx
        CORE(IRHO)=0
     ENDDO
  ENDDO
  DO J=J0,NY
     DO K=K0,NZ
        irho=1+(J-1+(K-1)*ny)*nx
        CORE(IRHO)=0
        irho=NX+(J-1+(K-1)*ny)*nx
        CORE(IRHO)=0
     ENDDO
  ENDDO
  NCORE=(NX-I0-1)*(NY-J0-1)*(NZ-K0-1)
100 NCOREP=NCORE
  DO I=I0,NX
     DO J=J0,NY
        DO K=K0,NZ
           irho=I+(J-1+(K-1)*ny)*nx
           IF(CORE(IRHO) == 0)THEN
              IF(I > I0)THEN
                 irho=I-1+(J-1+(K-1)*ny)*nx
                 if(CORE(IRHO) > 0.AND.RHO(IRHO) < RHOCUT)THEN
                    NCORE=NCORE-1
                    CORE(IRHO)=0
                 ENDIF
              ENDIF
              IF(I < NX)THEN
                 irho=I+1+(J-1+(K-1)*ny)*nx
                 if(CORE(IRHO) > 0.AND.RHO(IRHO) < RHOCUT)THEN
                    NCORE=NCORE-1
                    CORE(IRHO)=0
                 ENDIF
              ENDIF
              IF(J > J0)THEN
                 irho=I+(J-2+(K-1)*ny)*nx
                 if(CORE(IRHO) > 0.AND.RHO(IRHO) < RHOCUT)THEN
                    NCORE=NCORE-1
                    CORE(IRHO)=0
                 ENDIF
              ENDIF
              IF(J < NY)THEN
                 irho=I+(J+(K-1)*ny)*nx
                 if(CORE(IRHO) > 0.AND.RHO(IRHO) < RHOCUT)THEN
                    NCORE=NCORE-1
                    CORE(IRHO)=0
                 ENDIF
              ENDIF
              IF(K > K0)THEN
                 irho=I+(J-1+(K-2)*ny)*nx
                 if(CORE(IRHO) > 0.AND.RHO(IRHO) < RHOCUT)THEN
                    NCORE=NCORE-1
                    CORE(IRHO)=0
                 ENDIF
              ENDIF
              IF(K < NZ)THEN
                 irho=I+(J-1+(K)*ny)*nx
                 if(CORE(IRHO) > 0.AND.RHO(IRHO) < RHOCUT)THEN
                    NCORE=NCORE-1
                    CORE(IRHO)=0
                 ENDIF
              ENDIF
           ENDIF
        ENDDO
     ENDDO
  ENDDO
  IF(NCORE /= NCOREP)GOTO 100
  NDATA=NX*NY*NZ
  NCOREP=0
  DO I=1,NDATA
     RHOI=RHO(I)
     IF(CORE(I) > 0)THEN
        NCOREP=NCOREP+1
     ENDIF
  ENDDO
  IF(NCORE /= NCOREP) &
       CALL WRNDIE(0,'<FINDCORE>','CORE NUMBER ERROR')
  RETURN
END subroutine findrcore

subroutine EMAPRCUTC(i0,j0,k0,nx,ny,nz,rhocut, &
     rho,ddrho,core)
  !-----------------------------------------------------------------------
  !     Estimate rcut based on the densities where core=0
  use chm_kinds
  use number
  implicit none
  INTEGER nx,ny,nz,core(*)
  INTEGER I0,J0,K0,I,J,K
  real(chm_real4) rho(*),ddrho(*)
  real(chm_real) rhocut,ACRHO,ACM
  INTEGER IRHO
  !
  ACM=ZERO
  ACRHO=ZERO
  DO I=I0+1,NX-1
     DO J=J0+1,NY-1
        DO K=K0+1,NZ-1
           irho=I+(J-1+(K-1)*ny)*nx
           IF(CORE(IRHO) == 0)THEN
              ACRHO=ACRHO+RHO(IRHO)
              ACM=ACM+ONE
           ENDIF
        ENDDO
     ENDDO
  ENDDO
  IF(ACM > 0)THEN
     RHOCUT=ACRHO/ACM
  ELSE
     CALL WRNDIE(0,'<FINDRCUT>','CORE is not defined!')
  ENDIF
  RETURN
END subroutine EMAPRCUTC

subroutine EMAPDDR(IDEMP)
  !-----------------------------------------------------------------------
  !     Calculate Laplacian value
  use chm_kinds
  use number
  use emapmod
  implicit none
  INTEGER IDEMP
  !
  CALL EMAPDDR1(LEMAPX(IDEMP),LEMAPY(IDEMP),LEMAPZ(IDEMP), &
       empgrd(IDEMP)%rho,empgrd(IDEMP)%ddrho)
  RETURN
END subroutine EMAPDDR

subroutine EMAPDDR1(nx,ny,nz,rho,ddrho)
  !-----------------------------------------------------------------------
  !     Calculate Laplacian value
  !
  use chm_kinds
  use number
  implicit none
  INTEGER nx,ny,nz,NDATA,NCORE,NCOREP
  INTEGER I,J,K,II,JJ,KK
  real(chm_real4) rho(*),ddrho(*),RHOI,DDRI,DDR0
  real(chm_real) rhocut,ACR,RCR
  INTEGER IS,IE,JS,JE,KS,KE,IRHO
  !
  DDR0=ZERO
  DO I=1,NX
     DO J=1,NY
        irho=I+(J-1)*nx
        DDRHO(IRHO)=DDR0
        irho=I+(J-1+(nz-1)*ny)*nx
        DDRHO(IRHO)=DDR0
     ENDDO
  ENDDO
  DO I=1,NX
     DO K=1,NZ
        irho=I+((K-1)*ny)*nx
        DDRHO(IRHO)=DDR0
        irho=I+(NY-1+(K-1)*ny)*nx
        DDRHO(IRHO)=DDR0
     ENDDO
  ENDDO
  DO J=1,NY
     DO K=1,NZ
        irho=1+(J-1+(K-1)*ny)*nx
        DDRHO(IRHO)=DDR0
        irho=NX+(J-1+(K-1)*ny)*nx
        DDRHO(IRHO)=DDR0
     ENDDO
  ENDDO
  !  Calculate Laplace value
  DO I=2,NX-1
     DO J=2,NY-1
        DO K=2,NZ-1
           irho=I-1+(J-1+(K-1)*ny)*nx
           DDRI=RHO(IRHO)
           irho=I+1+(J-1+(K-1)*ny)*nx
           DDRI=DDRI+RHO(IRHO)
           irho=I+(J-2+(K-1)*ny)*nx
           DDRI=DDRI+RHO(IRHO)
           irho=I+(J+(K-1)*ny)*nx
           DDRI=DDRI+RHO(IRHO)
           irho=I+(J-1+(K-2)*ny)*nx
           DDRI=DDRI+RHO(IRHO)
           irho=I+(J-1+(K)*ny)*nx
           DDRI=DDRI+RHO(IRHO)
           irho=I+(J-1+(K-1)*ny)*nx
           DDRHO(IRHO)=DDRI-SIX*RHO(IRHO)
        ENDDO
     ENDDO
  ENDDO
  RETURN
END subroutine EMAPDDR1

subroutine finddcore(i0,j0,k0,nx,ny,nz,rhocut, &
     rho,ddrho,core,NCORE)
  !-----------------------------------------------------------------------
  !  define core=0 for outside grid with rho<rhocut and dd(rho)>0
  use chm_kinds
  use number
  implicit none
  integer nx,ny,nz,ndata,ncore,ncorep,core(*)
  integer i0,j0,k0,i,j,k,ii,jj,kk
  real(chm_real4) rho(*),ddrho(*),rhoi,ddri,ddr0
  real(chm_real) rhocut
  integer is,ie,js,je,ks,ke,irho
  logical notdone

  ddr0=zero
  do i=i0,nx
     do j=j0,ny
        irho=i+(j-1)*nx
        core(irho)=0
        irho=i+(j-1+(nz-1)*ny)*nx
        core(irho)=0
     enddo
  enddo
  do i=i0,nx
     do k=k0,nz
        irho=i+((k-1)*ny)*nx
        core(irho)=0
        irho=i+(ny-1+(k-1)*ny)*nx
        core(irho)=0
     enddo
  enddo
  do j=j0,ny
     do k=k0,nz
        irho=1+(j-1+(k-1)*ny)*nx
        core(irho)=0
        irho=nx+(j-1+(k-1)*ny)*nx
        core(irho)=0
     enddo
  enddo
  ncore=(nx-i0-1)*(ny-j0-1)*(nz-k0-1)
  !  calculate laplace value
  do i=i0+1,nx-1
     do j=j0+1,ny-1
        do k=k0+1,nz-1
           irho=i+(j-1+(k-1)*ny)*nx
           if(core(irho) == 0)ncore=ncore-1
        enddo
     enddo
  enddo

  !  Zero out core index of accessible spaces
  notdone=.true.
  !  IF(NCORE /= NCOREP)GOTO 100
  loop100: do while(notdone)
     NCOREP=NCORE
     do i=i0,nx
        do j=j0,ny
           do k=k0,nz
              irho=i+(j-1+(k-1)*ny)*nx
              if(core(irho) == 0)then
                 if(i > i0)then
                    irho=i-1+(j-1+(k-1)*ny)*nx
                    if(core(irho) > 0.and. &
                         (rho(irho) < rhocut.or.ddrho(irho) >= ddr0))then
                       ncore=ncore-1
                       core(irho)=0
                    endif
                 endif
                 if(i < nx)then
                    irho=i+1+(j-1+(k-1)*ny)*nx
                    if(core(irho) > 0.and. &
                         (rho(irho) < rhocut.or.ddrho(irho) >= ddr0))then
                       ncore=ncore-1
                       core(irho)=0
                    endif
                 endif
                 if(j > j0)then
                    irho=i+(j-2+(k-1)*ny)*nx
                    if(core(irho) > 0.and. &
                         (rho(irho) < rhocut.or.ddrho(irho) >= ddr0))then
                       ncore=ncore-1
                       core(irho)=0
                    endif
                 endif
                 if(j < ny)then
                    irho=i+(j+(k-1)*ny)*nx
                    if(core(irho) > 0.and. &
                         (rho(irho) < rhocut.or.ddrho(irho) >= ddr0))then
                       ncore=ncore-1
                       core(irho)=0
                    endif
                 endif
                 if(k > k0)then
                    irho=i+(j-1+(k-2)*ny)*nx
                    if(core(irho) > 0.and. &
                         (rho(irho) < rhocut.or.ddrho(irho) >= ddr0))then
                       ncore=ncore-1
                       core(irho)=0
                    endif
                 endif
                 if(k < nz)then
                    irho=i+(j-1+(k)*ny)*nx
                    if(core(irho) > 0.and. &
                         (rho(irho) < rhocut.or.ddrho(irho) >= ddr0))then
                       ncore=ncore-1
                       core(irho)=0
                    endif
                 endif
              endif
           enddo
        enddo
     enddo
     notdone = (NCORE /= NCOREP)
  enddo loop100

  ndata=nx*ny*nz
  ncorep=0
  do i=1,ndata
     rhoi=rho(i)
     if(core(i) > 0)then
        ncorep=ncorep+1
     endif
  enddo
  if(ncore /= ncorep) &
       CALL WRNDIE(0,'emapsubs.src<FINDCORE>','CORE NUMBER ERROR')
  return
end subroutine finddcore

subroutine buildcore(nx,ny,nz,core)
  !-----------------------------------------------------------------------
  use chm_kinds
  implicit none
  integer nx,ny,nz,core(*),corei,coremin
  integer i,j,k,ijk,ijk1,ijk2,ijk3,ijk4,ijk5,ijk6,nbuild
  !
  nbuild=1
  do while(nbuild > 0 )
     nbuild=0
     do i=2,nx-1
        do j=2,ny-1
           do k=2,nz-1
              ijk=i+(j-1+(k-1)*ny)*nx
              corei=core(ijk)
              if(corei > 0)then
                 ijk1=i-1+(j-1+(k-1)*ny)*nx
                 coremin=core(ijk1)
                 ijk2=i+1+(j-1+(k-1)*ny)*nx
                 if(coremin > core(ijk2))coremin=core(ijk2)
                 ijk3=i+(j-2+(k-1)*ny)*nx
                 if(coremin > core(ijk3))coremin=core(ijk3)
                 ijk4=i+(j+(k-1)*ny)*nx
                 if(coremin > core(ijk4))coremin=core(ijk4)
                 ijk5=i+(j-1+(k-2)*ny)*nx
                 if(coremin > core(ijk5))coremin=core(ijk5)
                 ijk6=i+(j-1+(k)*ny)*nx
                 if(coremin > core(ijk6))coremin=core(ijk6)
                 if(corei /= coremin+1)then
                    core(ijk)=coremin+1
                    nbuild=nbuild+1
                 endif
              endif
           enddo
        enddo
     enddo
  end do
  return
end subroutine buildcore

subroutine initcore(n,core)
  !-----------------------------------------------------------------------
  use chm_kinds
  implicit none
  integer i,n,core(*)
  core(1:n)=1
  return
end subroutine initcore

      SUBROUTINE EMAPFORCE(ENEMAP,NAT,X,Y,Z,AMASS,DX,DY,DZ)
!-----------------------------------------------------------------------
!
!   This routine calculate forces on reference atoms of give maps
!   The reference atoms are defined by: EMAP REFE mapid atom-selection
!     or by: EMAP GENE mapid atom -selection
!   Rigid domains are used to define the position of the map
!     Using the map at its original position: EMAP ASSIn mapid AS rigid
!     Using the map at selected atom position: EMAP ASSIn mapid AS rigid
!        EMAP ASSIn mapid AS rigid atom-selection
!   The active maps are defined by: EMAP FORCe rigid [rigid [...]]
!       
!
  use chm_kinds
  use number
#if KEY_PARALLEL==1
  use parallel       
#endif
  use machutil,only:eclock
  use emapmod
  implicit none
!
#if KEY_PARALLEL==1
    real(chm_real) GCARR(30),TIMMER
#endif 
      real(chm_real) ENEMAP,X(*),Y(*),Z(*),AMASS(*)
      real(chm_real) DX(*),DY(*),DZ(*)
      INTEGER NAT
      real(chm_real) ENRIG
      INTEGER I,IDRIG,IDEMP
!
      ENEMAP=ZERO
!  Loop over all active rigid domains
      DO I=1,NEMCOMP
        IDRIG=IDCOMPS(I)
        IDEMP=IDEMRIG(IDRIG)
        CALL RIGIDFORCE(ENRIG,NATEMAP(IDEMP),     &
          X,Y,Z,AMASS,DX,DY,DZ,     &
          TEMRIG(1,IDRIG),REMRIG(1,IDRIG),EMAPGUID(I), &
          FRCRIG(1,IDRIG),TRQRIG(1,IDRIG),     &
          MEMAPX(IDEMP),MEMAPY(IDEMP),MEMAPZ(IDEMP),     &
          LEMAPX(IDEMP),LEMAPY(IDEMP),LEMAPZ(IDEMP),     &
          CEMAPX(IDEMP),CEMAPY(IDEMP),CEMAPZ(IDEMP),     &
          DEMAPX(IDEMP),DEMAPY(IDEMP),DEMAPZ(IDEMP),     &
          EMPCRD(IDEMP)%IATOM,     &
          AREMAP(IDEMP),EMPGRD(IDEMP)%RHO)
        ENEMAP=ENEMAP+ENRIG
      ENDDO
#if KEY_PARALLEL==1
    CALL PSYNC()
    TIMMER=ECLOCK()
    CALL GCOMBR(FRCRIG,3*NEMRIG)  ! MH2023: extend the gcomb interface for fix
    CALL GCOMBR(TRQRIG,3*NEMRIG)
    TMERI(TIMGCOMM)=TMERI(TIMGCOMM)+ECLOCK()-TIMMER
    TIMMER=ECLOCK()
#endif 
      RETURN
      END SUBROUTINE EMAPFORCE

!
      SUBROUTINE RIGIDFORCE(ENRIG,NEMAT,X,Y,Z,AMASS,DX,DY,DZ, &
         TR,U,PGUID,FEM,TEM,  &
         MNX,MNY,MNZ,LX,LY,LZ,CX,CY,CZ,GX,GY,GZ,IDX,RAVG,RHO)
!-----------------------------------------------------------------------
!
!   This routine transform reference atoms to map frame and
!     calculate B-splie parameters and the energy and force
!     then transform the force back to simulation frame
!       
!
  use chm_kinds
  use dimens_fcm
  use number
  use stream 
  use consta
  use cnst_fcm
#if KEY_PARALLEL==1
  use parallel       
#endif
#if KEY_DOMDEC==1
  use domdec_common, only: q_domdec, homezone
#endif
  USE PMEUTIL
  implicit none
!
      real(chm_real) ENRIG,X(*),Y(*),Z(*),AMASS(*)
      real(chm_real) DX(*),DY(*),DZ(*)
      INTEGER NEMAT,MNX,MNY,MNZ,LX,LY,LZ,IDX(*)
      real(chm_real) TR(3),U(3,3),PGUID,FEM(3),TEM(3)
      real(chm_real4) RHO(*)
      real(chm_real) RAVG,RHO0,CX,CY,CZ,GX,GY,GZ
      real(chm_real) BS1(8),BS2(8),BS3(8)
      real(chm_real) DBS1(8),DBS2(8),DBS3(8)
      real(chm_real) XI,YI,ZI,XJ,YJ,ZJ
      real(chm_real) AMASSI,RHOI,F1,F2,F3,FX,FY,FZ
      real(chm_real) CFACT,CFACTX,CFACTY,CFACTZ,FR1,FR2,FR3,W
      real(chm_real) VAL0,VAL1,VAL2,VAL3,VAL0A,VAL1A,VAL2A,VAL3A
      INTEGER ID,IAT,I,J,K,M1,M2,M3,NSTA,NEND
      INTEGER ITH1,ITH2,ITH3,IPT1,IPT2,IPT3,BORDER
!
  ! Set loop limits
#if KEY_DOMDEC==1
  if (q_domdec) then
    NSTA=1
    NEND=NEMAT
  else
#endif
#if KEY_PARALLEL==1
      NSTA=(NEMAT*MYNOD)/NUMNOD+1
      NEND=(NEMAT*MYNODP)/NUMNOD
#else /**/
      NSTA=1
      NEND=NEMAT
#endif 
#if KEY_DOMDEC==1
  endif
#endif
      BORDER=6
      CFACT=PGUID
      CFACTX=CFACT/GX
      CFACTY=CFACT/GY
      CFACTZ=CFACT/GZ
      RHO0=RAVG/(LX*LY*LZ)
      ENRIG=ZERO
      FEM=ZERO
      TEM=ZERO
!  Calculate selected atoms
      atloop: DO ID=NSTA,NEND
        IAT=IDX(ID)
#if KEY_DOMDEC==1
    if (q_domdec) then
      if (.not. (homezone(iat) == 1)) cycle atloop
    endif
#endif
        AMASSI=AMASS(IAT)
        XI=X(IAT)-CX-TR(1)
        YI=Y(IAT)-CY-TR(2)
        ZI=Z(IAT)-CZ-TR(3)
! transform to map frame
        XJ=XI*U(1,1)+YI*U(2,1)+ZI*U(3,1)+CX
        YJ=XI*U(1,2)+YI*U(2,2)+ZI*U(3,2)+CY
        ZJ=XI*U(1,3)+YI*U(2,3)+ZI*U(3,3)+CZ
! calculate B-spline parameters
        FR1=XJ/GX
        FR2=YJ/GY
        FR3=ZJ/GZ
        M1=ANINT(FR1-HALF)
        W=FR1-M1
        call fill_bspline(w,BORDER,BS1,dBS1)
        M2=ANINT(FR2-HALF)
        W=FR2-M2
        call fill_bspline(w,BORDER,BS2,dBS2)
        M3=ANINT(FR3-HALF)
        W=FR3-M3
        call fill_bspline(w,BORDER,BS3,dBS3)
! Calculate B-spline interception
        F1 = ZERO 
        F2 = ZERO 
        F3 = ZERO 
        K = M3-MNZ+1 - BORDER/2 
        DO 100 ITH3 = 1,BORDER
           K=K+1
           IF(K*(LZ-K)<=0)GOTO 100
           IPT1=(K-1)*LY*LX
            VAL0A = AMASSI  * BS3(ITH3)
            VAL1A = AMASSI * CFACTX * BS3(ITH3)
            VAL2A = AMASSI * CFACTY * BS3(ITH3)
            VAL3A = AMASSI * CFACTZ * DBS3(ITH3)
!            VAL1A = AMASSI *  BS3(ITH3)
!            VAL2A = AMASSI *  BS3(ITH3)
!            VAL3A = AMASSI *  DBS3(ITH3)
!
            J = M2-MNY+1 - BORDER/2
!
            DO 200 ITH2 = 1,BORDER
               J=J+1
               IF(J*(LY-J)<=0)GOTO 200
               IPT2=IPT1+(J-1)*LX
!
               VAL0= VAL0A * BS2(ITH2)
               VAL1= VAL1A * BS2(ITH2)
               VAL2= VAL2A * DBS2(ITH2)
               VAL3= VAL3A * BS2(ITH2)

!
               I = M1-MNX+1 - BORDER/2
!
               DO 300 ITH1 = 1,BORDER
                  I=I+1
                  IF(I*(LX-I)<=0)GOTO 300
                  IPT3=IPT2+I
                  RHOI=RHO(IPT3)-RHO0
                  ENRIG = ENRIG + VAL0 * RHOI * BS1(ITH1)
!                                   force is negative of grad
                  F1 = F1 - VAL1 * RHOI * DBS1(ITH1)
                  F2 = F2 - VAL2 * RHOI * BS1(ITH1)
                  F3 = F3 - VAL3 * RHOI * BS1(ITH1)
!
300          CONTINUE
200        CONTINUE
100      CONTINUE
!
         FX = -(U(1,1)*F1+U(1,2)*F2+U(1,3)*F3)
         FY = -(U(2,1)*F1+U(2,2)*F2+U(2,3)*F3)
         FZ = -(U(3,1)*F1+U(3,2)*F2+U(3,3)*F3)
         DX(IAT) = DX(IAT)+FX
         DY(IAT) = DY(IAT)+FY
         DZ(IAT) = DZ(IAT)+FZ
         ! Calculate force and torque for map objects (negative of derivatives)
         FEM(1)=FEM(1)+FX
         FEM(2)=FEM(2)+FY
         FEM(3)=FEM(3)+FZ
         TEM(1)=TEM(1)+YI*FZ-ZI*FY
         TEM(2)=TEM(2)+ZI*FX-XI*FZ
         TEM(3)=TEM(3)+XI*FY-YI*FX
      ENDDO atloop
      ENRIG=CFACT*ENRIG
      RETURN
      END SUBROUTINE RIGIDFORCE

      SUBROUTINE EMAPGUIDPRM(IDRIG,IDEMP,AMASS,NAT,IDX)
!-----------------------------------------------------------------------
!
!   This routine calculate forces on reference atoms of give maps
!   The reference atoms are defined by: EMAP REFE mapid atom-selection
!     or by: EMAP GENE mapid atom -selection
!   Rigid domains are used to define the position of the map
!     Using the map at its original position: EMAP ASSIn mapid AS rigid
!     Using the map at selected atom position: EMAP ASSIn mapid AS rigid
!        EMAP ASSIn mapid AS rigid atom-selection
!   The active maps are defined by: EMAP FORCe rigid [rigid [...]]
!       
!
  use chm_kinds
  use dimens_fcm
  use number
  use stream 
  use consta
  use cnst_fcm
  use emapmod
  implicit none
!
      INTEGER IDRIG,IDEMP,NAT,IDX(*)
      real(chm_real) AMASS(*)
!
      INTEGER I,IAT,NDATA
      real(chm_real) TMASS,DELT
!
      TMASS=ZERO
!  Loop over all active rigid domains
      DO I=1,NAT
        IAT=IDX(I)
        TMASS=TMASS+AMASS(IAT)
      ENDDO
!  Statistics of map object
      CALL EMAPSTAT(IDEMP)
      NDATA=LEMAPX(IDEMP)*LEMAPY(IDEMP)*LEMAPZ(IDEMP)
      DELT=(RREMAP(IDEMP)-AREMAP(IDEMP)*AREMAP(IDEMP)/NDATA)/NDATA
      IF(DELT>ZERO)DELT=SQRT(DELT)
      EMAPGUID(NEMCOMP)=-EMGUID/DELT
      RETURN
      END SUBROUTINE EMAPGUIDPRM

      SUBROUTINE EMAPLD(DELT)
!-----------------------------------------------------------------------
!  moving rigid domain based on position Langevin motion
!
!     -dEmap/dr=f=gamma*m*v    v=f/(gamma*m)
!     -dEmap/da=T=gamma*I*w    w=T/(gamma*I)
!
!     v0=sqrt(3kT/m)   w0=sqrt(3kT/I)
!
!  Scale down based on the thermo motion
!     dr=dt*v/(1+sqrt(v/v0))
!     da=dt*w/(1+sqrt(w/w0))
!
!
  use chm_kinds
  use dimens_fcm
  use number
  use stream 
  use consta
  use cnst_fcm
  use emapmod
  implicit none
!
      INTEGER IDRIG,IDEMP
      INTEGER I
      real(chm_real) DELT,EK,GAMMA2
!
      real(chm_real)  fxi,fyi,fzi,txi,tyi,tzi
      real(chm_real)  amassi,aineri,ff,tt
      real(chm_real)  DRF,DGT
!
      !  map moving time step
      gamma2=emgamma*emgamma*timfac*timfac
      DO I=1,NEMCOMP
        IF(LCOMPFIX(I))CYCLE
        IDRIG=IDCOMPS(I)
        IDEMP=IDEMRIG(IDRIG)
        EK=3.0*KBOLTZ*EMTEMP*NATEMAP(IDEMP)
!     Magnitude of force and torque
        FXI=FRCRIG(1,IDRIG)
        FYI=FRCRIG(2,IDRIG)
        FZI=FRCRIG(3,IDRIG)
        TXI=TRQRIG(1,IDRIG)
        TYI=TRQRIG(2,IDRIG)
        TZI=TRQRIG(3,IDRIG)
        AMASSI=EMMASS(IDEMP)
        AINERI=EMINER(IDEMP)
        FF=FXI*FXI+FYI*FYI+FZI*FZI
        TT=TXI*TXI+TYI*TYI+TZI*TZI
!   Langevin velocities: f=gamma*p and torq=gamma*L
!        vrig=f/(EMGAMMA*TIMFAC*amassi)
!        wrig=t/(EMGAMMA*TIMFAC*aineri)
!   Constrol motion with temperature: Ek~ET(3kT/2)
!     Ec=Ek*ET/(Ek+ET) to mantain Ec<ET
!     vc=sqrt(3kTf^2/(mf^2+3kT(m*gamma)^2)     drf=vc/f
        DRF=delt/sqrt(amassi*(ff/ek+amassi*gamma2))
        DGT=delt*sqrt(tt/(aineri*(tt/ek+aineri*gamma2)))/DEGRAD
      ! translation
        if(ABS(DRF)>RSMALL)CALL EMAPTRN(IDRIG,FXI*DRF,FYI*DRF,FZI*DRF)
      ! rotation 
        if(ABS(DGT)>RSMALL)CALL EMAPROT(IDRIG,TXI,TYI,TZI,DGT)
      ENDDO
      return
      end subroutine EMAPLD


#endif /* (emap_main)*/

SUBROUTINE NULL_EMAP
  RETURN
END SUBROUTINE NULL_EMAP

