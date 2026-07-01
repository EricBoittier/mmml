!---------------------------------------------------------------------
!     DIMS - Dynamic Importance Sampling
!
!     Authors:
!               Juan Roberto Perilla <jrperillaj@jhu.edu>
!               Tom Woolf <twoolf@jhmi.edu>
!
!---------------------------------------------------------------------
module dims
  use chm_kinds
  use chm_types
  use dimens_fcm
  implicit none

#if KEY_DIMS==1
  !     QDIMS    -  DIMS ON/OFF
  !     IDIMS    -  DIMS Selection
  !     IBNM     -  IBNM Selection - deprecated - 
  !     ISND     -  Second atom Selection
  !     DIMSIDX  -  DIMS selection Index
  !     BNMIDX   -  BNM  selection Index
  !     IMIN     -  Array for best modes
  !     ASCALES  -
  !     DSCALE   -  Scaling factor
  !     X,Y,Z SCRATCH - Scratch arrays for coordinates
  !     X,Y,Z BNM     - Coordinates for BNM arrays
  !
  !     DORIENT  -  Logical for orient
  !     IORIENT  -  Orient every IORIENT
  !
  !     DIMTARG*A - Target arrays coords.
  !     D*DIMS    - DIMS Bias arrays
  !
  real(chm_real),allocatable,dimension(:) :: ascales
  integer,allocatable,dimension(:) :: dimsidx
  INTEGER IMIN
  real(chm_real)  DSCALE,DIMSMASS

  !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
  !
  !     Misc. DIMS Variables including Progress Score external call
  !
  !
  !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
  integer, allocatable :: dimactive(:,:) ! maxaim,3
  integer, allocatable :: IDIMS(:), dimdone(:) ! maxaim
  real(chm_real), allocatable :: DIMTARGXA(:), DIMTARGYA(:), DIMTARGZA(:), WDIMS(:)
  INTEGER NIDIMS,NMSKIP,NBIAS,IUDIMSNM,BSKIP
  LOGICAL DDIHE,QDIMS,DIMNM,MDISA,DDONE,DDFIX,NMPRINT,DIMNMON
  LOGICAL DYNDIMCUTOFF,DIMSON,DHARD,DCART,NMFIRST,UFSR,DHALT
  real(chm_real) DIMSTOL,QDPROD,DIMCUTOFF,DWIDTH

  real(chm_real),allocatable,dimension(:) :: XScratch, YScratch, ZScratch
  real(chm_real),allocatable,dimension(:) :: DXDIMS,DYDIMS,DZDIMS
  real(chm_real),allocatable,dimension(:) :: VXDIMS,VYDIMS,VZDIMS

  LOGICAL DORIENT
  INTEGER IORIENT

  !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
  !
  !     BNM Variables needed to compute NM along trajectory
  !
  !     SPECIAL FOR BLOCK HESSIAN METHOD FROM LI GUOHUI
  !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
  LOGICAL QCOVB,QCOVM,qserl,qpara,qssmd,qnurx
  INTEGER DD1,DD5,DDS,ddv2
  INTEGER NATBNM
  integer,allocatable,dimension(:) :: BNMIDX,nactv,nrissh,nriclh,isnd
  integer,allocatable,dimension(:) :: nstres,nndres,nstatm,nndatm
  integer,allocatable,dimension(:) :: idxres
  real(chm_real),allocatable,dimension(:) :: DDV,ddm,ddf,ddscr
  !     INTEGER ISLCT
  !     GNM
  logical qanm,qgnm,qganm,qquasi

  INTEGER SET1,SET2,NTMP,ICNTB
  LOGICAL QHALF1,QNCOVM1,QEXPERT,QHIGH,QGEN,QACTV,QGENR
  INTEGER NAT3
  INTEGER,allocatable,dimension(:) :: IDXATM,NSTATMRES
  INTEGER,allocatable,dimension(:) :: NNDATMRES
  real(chm_real),allocatable,dimension(:) :: ITREIGVI,ITREIG,hfinal
  integer NRISS,NRICL,IROW,ICOL,IDX2,IDX3,NMAXB
  !      LOGICAL QCOVB,QCOVM,qserl,qpara,qssmd,qnurx
  real(chm_real) CUTDIS,CUTVAL,tam,xcm,ycm

  !     Last-Mile TMD
#if KEY_TMD==1
  LOGICAL DTMD
#endif 

  !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
  !
  !     DIMS-Score Energies
  !
  !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

  INTEGER SAVER,CAVER
  real(chm_real),allocatable,dimension(:) :: DIMSSCORES
  real(chm_real) DIMSE0, DIMSEQ, DIMSED
  real(chm_real) DIMSSCR
  real(chm_real) LDIMSSCR

  !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
  !
  !     DIMS NM Self Avoidance and Combinatorial
  !
  !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
  INTEGER MTRAJ, NMUNIT, NCOMB,NBEST,NWIND
  INTEGER DSCUNIT
  INTEGER NMCOUNTER
  real(chm_real),allocatable,dimension(:) :: bestscores
  real(chm_real),allocatable,dimension(:) :: ddev,DISTDESC0,DISTDESC
  integer,allocatable,dimension(:) :: bestmodes,icmodes,icmin,nmvec,nmidx
  type(chm_iarray),allocatable,dimension(:) :: NMMATRIX

  !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
  !
  !
  !     Interatomic distance variables
  !
  !
  !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
  LOGICAL QASC
  INTEGER NICA, NICAKNN
  INTEGER NICAN, NICANKNN
  INTEGER CAKNN
  real(chm_real),allocatable,dimension(:) :: CADIST0,CADIST
  integer,allocatable,dimension(:) :: icasl,icaidx,ICAIDXN,caneigh
  real(chm_real) CAEPS

  !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
  !
  !
  !    Atom-distribution hash variables
  !
  !
  !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
  LOGICAL QATDIST
#else /**/
  logical, parameter :: QDIMS = .false.
#endif 
  character(len=*), parameter, private :: srcfile = 'dims.src'

contains

#if KEY_DIMS==1 /*if_dims*/
  subroutine dims_init
    qdims=.false.
    dimnm=.false.
    ddihe=.false.
    ufsr=.false.
    return
  end subroutine dims_init

  SUBROUTINE DIMSINIT(COMLYN,COMLEN)
    !---------------------------------------------------------------------
    !
    !     This routine initialize DIMS
    !
    !---------------------------------------------------------------------
    !
    !
  use number
  use psf
  use coord
  use stream
  use ctitla
  use memory
#if KEY_PARALLEL==1
  use parallel 
#endif
  use string

  character(len=*) COMLYN
  INTEGER COMLEN
  INTEGER I,J
  INTEGER DIHETEST
  character(len=*), parameter :: procname = 'DIMSINIT'
#if KEY_ANNLIB==1
  EXTERNAL CALLANN 
#endif
  QDIMS=.TRUE.
  MDISA=.TRUE.
  DIMSON=.FALSE.
  DCART=.FALSE.
  DDONE=.FALSE.
  MDISA=.FALSE.
  DIMNM=.TRUE.
  IORIENT=0
#if KEY_TMD==1
  DTMD=.FALSE.
#endif 
  NCOMB=1
  NMSKIP=-1
  SAVER=5
  MTRAJ=0
  NMCOUNTER=0
  NMUNIT=-1
  DSCUNIT=-1
  NBEST=10
  NWIND=1
  CAKNN=0
  DWIDTH=-1.0
  DIMCUTOFF=1.0
  DHARD=INDXA(COMLYN,COMLEN,'HARD') > 0
  DWIDTH=GTRMF(COMLYN,COMLEN,'DCAR',ONE)
  DCART=DWIDTH > -1.0
  DHALT=INDXA(COMLYN,COMLEN,'HALT') > 0
  IF(INDXA(COMLYN,COMLEN,'DBNM') == 0.AND.DCART) THEN
     DIMNM=.FALSE.
  ENDIF
  IORIENT=GTRMI(COMLYN,COMLEN,'ORIE',iorient)
  DDFIX=INDXA(COMLYN,COMLEN,'DFIX') > 0
  DSCUNIT=GTRMI(COMLYN,COMLEN,'DSUN',DSCUNIT)
  SAVER=GTRMI(COMLYN,COMLEN,'SCAV',saver)
  DIMSTOL=GTRMF(COMLYN,COMLEN,'DTOL',ONE)
  DIMCUTOFF=GTRMF(COMLYN,COMLEN,'COFF',ONE)
  DYNDIMCUTOFF=INDXA(COMLYN,COMLEN,'DCOF') > 0
  !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
  !
  !     BNM KEYWORDS
  !
  !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
  IF(DIMNM) THEN
#if KEY_PARALLEL==1
     !CALL WRNDIE(0,'DIMSNM>','NO parallel support')
#endif 
     NMFIRST=INDXA(COMLYN,COMLEN,'NMFI') > 0
     MTRAJ=GTRMI(COMLYN,COMLEN,'MTRA',MTRAJ)
     NCOMB=GTRMI(COMLYN,COMLEN,'COMB',NCOMB)
     NBEST=GTRMI(COMLYN,COMLEN,'NBES',NBEST)
     NWIND=GTRMI(COMLYN,COMLEN,'NWIN',NWIND)
     NWIND=NCOMB*NWIND
     NMPRINT=INDXA(COMLYN,COMLEN,'NMPR') > 0
     IUDIMSNM=GTRMI(COMLYN,COMLEN,'UNIT',1)
     NMUNIT=GTRMI(COMLYN,COMLEN,'NMUN',NMUNIT)
     NMSKIP=GTRMI(COMLYN,COMLEN,'SKIP',nmskip)
     DSCALE=GTRMF(COMLYN,COMLEN,'DSCA',ONE)
     NBIAS=GTRMI(COMLYN,COMLEN,'NBIA',1)
     BSKIP=GTRMI(COMLYN,COMLEN,'BSKI',NMSKIP)
     IF(NMSKIP <= 0) THEN
        NMSKIP=1
     ENDIF
     !     BIAS Skip
     IF(NBIAS < 3) THEN
        CALL WRNDIE(0,'<DIMSINIT>','NBIAS MUST BE AT LEAST EQUAL TO 3')
        NBIAS=3
     ENDIF
     IF(PRNLEV >= 3) THEN
        WRITE(OUTU,'(A30,I4,I4,I4)') &
             'DIMSINIT> NMSKIP BSKIP NBIAS :', NMSKIP &
             ,BSKIP,NBIAS
        WRITE(OUTU,'(A18,F10.3)') &
             'DIMSINIT> DSCALE: ',DSCALE
     ENDIF
     IF(NBIAS > NMSKIP) THEN
        IF (PRNLEV >= 0) THEN
           WRITE(OUTU,'(A57)')  &
                'DIMSINIT> NBIAS greater than NMSKIP. Setting NBIAS=NMSKIP'
        ENDIF
        NBIAS=NMSKIP
     ENDIF
  ENDIF
  !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
  !
  !     Misc. Warnings
  !
  !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
  IF(IORIENT <= 0) THEN
     IF(PRNLEV >= 3) THEN
        WRITE(OUTU, '(A56)')  &
             'DIMSNMINIT> ORIENT not specified. Turning off by default'
     ENDIF
     IORIENT=-1
  ENDIF
  IF (PRNLEV >= 3) WRITE(OUTU,'(A57,I4)') &
       'DIMSINIT> The DIMS score will be averaged over the first ', SAVER
  IF(NMPRINT) THEN
     IF (PRNLEV >= 3)   WRITE(OUTU, '(A123)') &
          'DIMSNMINIT> The unit you have selected must be open &
          for the dynamc cycle that is when DIMSNM computes/saves &
          the normal modes.'
  ENDIF
  IF(DYNDIMCUTOFF.AND.DHARD) THEN
     IF(PRNLEV >= 3)  WRITE(OUTU,'(A84)')  &
          'DIMSINIT> DynCutoff is not compatible with  &
          last mile DIMSHARD. Shutting off DynCutOff'
     DYNDIMCUTOFF=.FALSE.
  ENDIF

  call chmalloc(srcfile,procname,'dimactive',MAXAIM,3,intg=dimactive)
  call chmalloc(srcfile,procname,'IDIMS',MAXAIM,intg=IDIMS)
  call chmalloc(srcfile,procname,'dimdone',MAXAIM,intg=dimdone)
  call chmalloc(srcfile,procname,'DIMTARGXA',MAXAIM,crl=DIMTARGXA)
  call chmalloc(srcfile,procname,'DIMTARGYA',MAXAIM,crl=DIMTARGYA)
  call chmalloc(srcfile,procname,'DIMTARGZA',MAXAIM,crl=DIMTARGZA)
  call chmalloc(srcfile,procname,'WDIMS',MAXAIM,crl=WDIMS)

  QDPROD=1.0
  CALL SELDIMS(COMLYN,COMLEN)
  !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
  !
  !     Inter atomic DISTANCES AND ATOM-DIST HASH INIT
  !     Alternative metrics
  !
  !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
  !      UFSR=INDXA(COMLYN,COMLEN,'UFSR') > 0
#if KEY_ANNLIB==1
  CAKNN=GTRMI(COMLYN,COMLEN,'IATD',caknn)
  QASC=CAKNN > 0
  IF(.NOT.QASC) THEN
     CAKNN=GTRMI(COMLYN,COMLEN,'ATDI',caknn)
     QATDIST=CAKNN > 0
  ENDIF
  IF(QASC.OR.QATDIST) THEN
     CAEPS=0.1
     !     Data set
     call chmalloc('dims.src','DIMSINIT','ICASL',NATOM,intg=ICASL)
     CALL SELCTA(COMLYN,COMLEN,ICASL,X,Y,Z,WMAIN,.TRUE.)
     CALL COUNTSEL(ICASL,NICA,NATOM)
     IF(PRNLEV >= 4) THEN
        WRITE(OUTU,'(A36,I4)') &
             'DIMSINIT> GenCalpha data set atoms: ', NICA
     ENDIF
     call chmalloc('dims.src','DIMSINIT','ICAIDX',NICA,intg=ICAIDX)
     CALL BUILDDIMSID(ICASL,ICAIDX)
     !     Query set
     CALL SELCTA(COMLYN,COMLEN,ICASL,X,Y,Z,WMAIN,.TRUE.)
     CALL COUNTSEL(ICASL,NICAN,NATOM)
     IF(PRNLEV >= 4) THEN
        WRITE(OUTU,'(A37,I4)') &
             'DIMSINIT> GenCalpha query set atoms: ', NICAN
     ENDIF
     call chmalloc('dims.src','DIMSINIT','ICAIDXN',NICAN,intg=ICAIDXN)
     CALL BUILDDIMSID(ICASL,ICAIDXN)
     !
     !
     NICAKNN=NICA*CAKNN
     NICANKNN=NICAN*CAKNN
     call chmalloc('dims.src','DIMSINIT','CADIST0',NICANKNN,crl=CADIST0)
     call chmalloc('dims.src','DIMSINIT','CADIST',NICANKNN,crl=CADIST)
     call chmalloc('dims.src','DIMSINIT','CANEIGH',NICANKNN,intg=CANEIGH)
     CALL CALLANN(CAKNN,CAEPS,NICA,NICAN, &
          DIMTARGXA,DIMTARGYA,DIMTARGZA, &
          ICAIDX,ICAIDXN, &
          CADIST0,CANEIGH)
     IF(QATDIST) THEN
        call chmalloc('dims.src','DIMSINIT','DISTDESC0',NICAN*3,crl=DISTDESC0)
        call chmalloc('dims.src','DIMSINIT','DISTDESC',NICAN*3,crl=DISTDESC)
        CALL BUILDDESCRIPTOR(DIMTARGXA, DIMTARGYA, DIMTARGZA, &
             CANEIGH,ICAIDXN,DISTDESC0,CADIST,NICAN,CAKNN)
     ENDIF
  ENDIF
#endif /* ANNLIB*/
  !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
  IF(DSCUNIT /= -1) THEN
     !     Print whatever title is in TITLEA
     CALL WRTITL(TITLEA,NTITLA,DSCUNIT,0)
  ENDIF
  !     Compute DIMSMASS
  DIMSMASS=0.0
  DO J=1,NIDIMS
     I=DIMSIDX(J)
     DIMSMASS=DIMSMASS+AMASS(I)
  ENDDO
  IF(PRNLEV >= 3) WRITE(OUTU,'(A20,F10.3)')  &
       'DIMSINIT> DIMSMASS: ',DIMSMASS
  !     Allocate space for X,Y,Z Scratch
  call chmalloc('dims.src','DIMSINIT','XScratch',NATOM,crl=XScratch)
  call chmalloc('dims.src','DIMSINIT','YScratch',NATOM,crl=YScratch)
  call chmalloc('dims.src','DIMSINIT','ZScratch',NATOM,crl=ZScratch)
  call chmalloc('dims.src','DIMSINIT','DXDIMS',NATOM,crl=DXDIMS)
  call chmalloc('dims.src','DIMSINIT','DYDIMS',NATOM,crl=DYDIMS)
  call chmalloc('dims.src','DIMSINIT','DZDIMS',NATOM,crl=DZDIMS)
  call chmalloc('dims.src','DIMSINIT','VXDIMS',NATOM,crl=VXDIMS)
  call chmalloc('dims.src','DIMSINIT','VYDIMS',NATOM,crl=VYDIMS)
  call chmalloc('dims.src','DIMSINIT','VZDIMS',NATOM,crl=VZDIMS)
  call chmalloc('dims.src','DIMSINIT','DIMSSCORES',SAVER,crl=DIMSSCORES)
  DIMSE0=-1.0
  DIMSEQ=0.0
  DIMSED=0.0
  DIMSSCR=-1.0
  LDIMSSCR=0.0
  CAVER=0
  IF(DIMNM) THEN
     call chmalloc('dims.src','DIMSINIT','ASCALES',NBIAS+1,crl=ASCALES)
     IF(MTRAJ >= 1) THEN
        call chmalloc_chm_iptr('dims.src','DIMSINIT','NMMATRIX',MTRAJ,NMMATRIX)
        call chmalloc('dims.src','DIMSINIT','NMVEC',MTRAJ,intg=NMVEC)
     ENDIF
     call chmalloc('dims.src','DIMSINIT','BESTMODES',NBEST,intg=BESTMODES)
     call chmalloc('dims.src','DIMSINIT','ICMIN',NBEST,intg=ICMIN)
     call chmalloc('dims.src','DIMSINIT','ICMODES',NBEST,intg=ICMODES)
     call chmalloc('dims.src','DIMSINIT','BESTSCORES',NBEST,crl=BESTSCORES)
     call chmalloc('dims.src','DIMSINIT','NMIDX',NCOMB,intg=NMIDX)
     dxdims(1:natom) = zero
     dydims(1:natom) = zero
     dzdims(1:natom) = zero
     vxdims(1:natom) = zero
     vydims(1:natom) = zero
     vzdims(1:natom) = zero
     xscratch(1:natom) = zero
     yscratch(1:natom) = zero
     zscratch(1:natom) = zero
     BESTSCORES(1:NBEST) = zero
     BESTMODES(1:NBEST) = 0
     ICMODES(1:NBEST) = 0
     ICMIN(1:NBEST) = 0
     CALL BUILDSCALE(NBIAS+1,ASCALES,DSCALE)
     CALL DIMSNMINIT(COMLYN,COMLEN)
     DDIHE=.FALSE.
  ENDIF
  RETURN
  END subroutine dimsinit

  SUBROUTINE SELDIMS(COMLYN,COMLEN)
    !
    !     Subroutine for DIMS
    !                and BNM atoms selection
    !
  use number
  use memory
  use psf
  use coord
  use stream
  use select
    !
    character(len=*) COMLYN
    INTEGER COMLEN
    INTEGER K,J
    LOGICAL IERROR
    call chmalloc('dims.src','SELDIMS','ISND',NATOM,intg=ISND)
    CALL SELCTD(COMLYN,COMLEN,IDIMS,ISND,X,Y,Z,WMAIN,.TRUE.,IERROR)
    IF(IERROR) CALL WRNDIE(-5,'<SELDIMS>','ERROR IN SELECTION')
    NIDIMS=0
    NATBNM=0
    CALL COUNTSEL(IDIMS,NIDIMS,NATOM)
    CALL COUNTSEL(ISND,NATBNM,NATOM)
    IF(NIDIMS == 0.OR.NATBNM == 0) THEN
       CALL WRNDIE(0,'<DIMSINIT>','ZERO ATOMS SELECTED FOR DIMS')
       CALL WRNDIE(0,'<DIMSINIT>','ZERO ATOMS SELECTED FOR BNM')
    ELSE
       call chmalloc('dims.src','SELDIMS','DIMSIDX',NIDIMS,intg=DIMSIDX)
       call chmalloc('dims.src','SELDIMS','BNMIDX',NATBNM,intg=BNMIDX)
       CALL BUILDDIMSID(IDIMS,DIMSIDX)
       CALL BUILDDIMSID(ISND,BNMIDX)
       DO K=1,NIDIMS
          J=DIMSIDX(K)
          DIMACTIVE(J,1)=0
       ENDDO
    ENDIF
    IF(PRNLEV >= 3) THEN
       WRITE(OUTU,'(A37,I4)') &
            '<DIMSINIT>: ATOMS SELECTED FOR DIMS: ',NIDIMS
       WRITE(OUTU,'(A37,I4)') &
            '<DIMSINIT>: ATOMS SELECTED FOR  BNM: ',NATBNM
    ENDIF
    RETURN
  END SUBROUTINE SELDIMS

  SUBROUTINE COUNTSEL(IASEL,NSEL,NATOM)
    INTEGER IASEL(*)
    INTEGER NSEL,NATOM
    INTEGER I
    I=1
    NSEL=0
    DO I=1,NATOM
       IF(IASEL(I) == 1) THEN
          NSEL=NSEL+1
       ENDIF
    ENDDO
    RETURN
  END SUBROUTINE COUNTSEL

  SUBROUTINE BUILDDIMSID(IDIMS,DIMSIDX)
    !
  use psf

    !
    INTEGER IDIMS(*), DIMSIDX(*)
    INTEGER K,N
    N=1
    DO K=1,NATOM
       IF(IDIMS(K) == 1) THEN
          DIMSIDX(N)=K
          N=N+1
       ENDIF
    ENDDO
    RETURN
  END subroutine BUILDDIMSID

  SUBROUTINE DIMSBUILDIDX(IARRAY,IDX,NARRAY,NIDX)
!
    INTEGER IARRAY(*), NARRAY
    INTEGER IDX(*),NIDX
    INTEGER I
    NIDX=0
    DO I=1,NARRAY
       IF(IARRAY(I) == 1) THEN
          NIDX=NIDX+1
          IDX(NIDX)=I
       ENDIF
    ENDDO
    RETURN
  END SUBROUTINE DIMSBUILDIDX

  SUBROUTINE NEGIDIMS(IDIMS,ATFRST,ATLAST)
    !     Legacy
    !     Subroutine that do a logic not on the array IDIMS
    !     Thus if IDIMS(N)=0 => IDIMS(N)=1 and if IDIMS(N)=1 => IDIMS(N)=0
    INTEGER IDIMS(*)
    INTEGER ATFRST,ATLAST,J
    INTEGER DIMSSTATE, ESTATE
    !
    DIMSSTATE=1
    ESTATE=3
    !
    DO J=ATFRST,ATLAST
       IDIMS(J)=IEOR(IDIMS(J),DIMSSTATE)
    ENDDO
    RETURN
  END SUBROUTINE NEGIDIMS

  SUBROUTINE CLEARIDIMS(ATFRST,ATLAST,IDIMS)
    !     Clears idims'enable' flag
    INTEGER ATFRST,ATLAST,I
    INTEGER IDIMS(*)
    INTEGER DIMSSTATE, ESTATE
    !
    DIMSSTATE=1
    ESTATE=3
    !
    DO I=ATFRST,ATLAST
       IF (IDIMS(I) == ESTATE) IDIMS(I)=1
    ENDDO
    RETURN
  END SUBROUTINE CLEARIDIMS

  SUBROUTINE DIMSNMINIT(COMLYN,COMLEN)
    !
    ! DIMS-NM-INIT
    ! BNM Input parameters
    !
  use number
  use psf
  use memory
  use stream
  use chutil,only:atomid
  use string
#if KEY_PARALLEL==1
  use parallel  
#endif
    character(len=*) COMLYN
    INTEGER COMLEN
    !
    !     SPECIAL FOR BLOCK HESSIAN METHOD FROM LI GUOHUI
    INTEGER NDIM,NFREQ
    INTEGER NNMDS

    integer i,j,k,ipost,i1,j1,k1,II,ippp,dmd
    character(len=4) sidb,ridb,renb,acb
    character(len=4) sidold,ridold,renold,acold
    !     Local
    LOGICAL LNOMA

    !
    qanm=.false.
    qgnm=.false.
    qquasi=.false.
    qganm=.false.

    QSSMD=INDXA(COMLYN,COMLEN,'SSMD') > 0
    QNURX=INDXA(COMLYN,COMLEN,'NURX') > 0
    QGANM=INDXA(COMLYN,COMLEN,'GANM') > 0
    NFREQ=0
    NAT3=NATBNM*3
    NNMDS=NAT3 ! might want to change this later on

    LNOMA=.FALSE.
    DIMNMON=.TRUE.
    ! Allocate space for normal modes
#if KEY_PARALLEL==1
    call chmalloc('dims.src','DIMSNMINIT','DDV',1,crl=DDV)
    call chmalloc('dims.src','DIMSNMINIT','DDM',1,crl=DDM)
    call chmalloc('dims.src','DIMSNMINIT','DDF',1,crl=DDF)
    call chmalloc('dims.src','DIMSNMINIT','DDEV',1,crl=DDEV)
    call chmalloc('dims.src','DIMSNMINIT','DDSCR',1,crl=DDSCR)
#else /**/
    !if(qssmd.or.qnurx.or.qganm) then
    !   call chmalloc('dims.src','DIMSNMINIT','DDV',2,crl=DDV)
    !else 
    !   call chmalloc('dims.src','DIMSNMINIT','DDV',NAT3*NAT3,crl=DDV)
    !endif
    call chmalloc('dims.src','DIMSNMINIT','DDV',NAT3*NAT3,crl=DDV)
    call chmalloc('dims.src','DIMSNMINIT','DDM',NATBNM,crl=DDM)
    call chmalloc('dims.src','DIMSNMINIT','DDEV',NNMDS,crl=DDEV)
    DDEV(1:NNMDS)=zero
    call chmalloc('dims.src','DIMSNMINIT','DDF',NNMDS,crl=DDF)
    call chmalloc('dims.src','DIMSNMINIT','DDSCR',MAX(NAT3,NNMDS),crl=DDSCR)

    CALL FILDDM(DDM,AMASS,NATBNM,LNOMA)

#endif 
    ! PROCESS BLOCK HESSIAN METHOD FROM LI GUHOUI
    qserl=INDXA(COMLYN,COMLEN,'SERL') > 0
    qpara=INDXA(COMLYN,COMLEN,'PARA') > 0

    ! QEXPERT IS JUST FOR EXPERT !!!
    QEXPERT=INDXA(COMLYN,COMLEN,'EXPT') > 0
    QHIGH=INDXA(COMLYN,COMLEN,'HIGH') > 0
    QGEN=INDXA(COMLYN,COMLEN,'GENL') > 0
    QGENR=INDXA(COMLYN,COMLEN,'GENR') > 0
    QACTV=INDXA(COMLYN,COMLEN,'ACTV') > 0
    ! CHECK USER LEVEL
    ! THIS IS FOR EXPERT.
    IF(QEXPERT) THEN
       QHIGH=.FALSE.
       QGEN=.FALSE.
       QACTV=.FALSE.
       QGENR=.FALSE.
       call chmalloc('dims.src','DIMSNMINIT','NRISSH',2,intg=NRISSH)
       call chmalloc('dims.src','DIMSNMINIT','NRICLH',2,intg=NRICLH)
       call chmalloc('dims.src','DIMSNMINIT','NACTV',2,intg=NACTV)
       ! THIS IS FOR HIGH LEVEL USER.
    ELSE IF(QHIGH) THEN
       ! SO YOU CAN SET DIFFERENT RESI NUMBER FOR DIFFERENT SEG.
       QEXPERT=.FALSE.
       QGEN=.FALSE.
       QACTV=.FALSE.
       QGENR=.FALSE.
       call chmalloc('dims.src','DIMSNMINIT','NACTV',2,intg=NACTV)
       ! THIS IS FOR GENERAL USER.
    ELSE IF(QGEN) THEN
       QEXPERT=.FALSE.
       QHIGH=.FALSE.
       QACTV=.FALSE.
       QGENR=.FALSE.
       call chmalloc('dims.src','DIMSNMINIT','NRISSH',2,intg=NRISSH)
       call chmalloc('dims.src','DIMSNMINIT','NRICLH',2,intg=NRICLH)
       call chmalloc('dims.src','DIMSNMINIT','NACTV',2,intg=NACTV)
       NRISS=GTRMI(COMLYN,COMLEN,'SECD',0)
       ! O MEANS ALL RESIDUES IN THE SS WILL BE PUT INTO 1 BLOCK.
       NRICL=GTRMI(COMLYN,COMLEN,'COIL',0)
       ! 1 MEANS 1 RESIDUE IN THE COIL WILL BE PUT INTO 1 BLOCK.
    ELSE IF(QACTV) THEN
       ! THIS IS FOR ACTIVE METHOD TO SET BLOCK NUMBER IN EACH SEG.
       QGEN=.FALSE.
       QGENR=.FALSE.
       QEXPERT=.FALSE.
       QHIGH=.FALSE.
       call chmalloc('dims.src','DIMSNMINIT','NRISSH',2,intg=NRISSH)
       call chmalloc('dims.src','DIMSNMINIT','NRICLH',2,intg=NRICLH)
    ELSE
       QGENR=.TRUE.
       !         ntres=nres
       !         ntseg=nseg
       i=bnmidx(1)
       call atomid(i,sidb,ridb,renb,acb)
       sidold=sidb
       ridold=ridb
       renold=renb
       acold=acb
       j=1
       k=1
    ENDIF
    !     ENDOF USERLEVEL
    !
    set1=int(30*natom*8/1.0d6)+1
    call chmalloc('dims.src','DIMSNMINIT','IDXRES',NATOM,intg=IDXRES)
    call chmalloc('dims.src','DIMSNMINIT','IDXATM',NATOM,intg=IDXATM)
    call chmalloc('dims.src','DIMSNMINIT','ITREIG',6*NAT3,crl=ITREIG)
    RETURN
  END SUBROUTINE DIMSNMINIT

  SUBROUTINE DIMSNM()
    !     Calculate the normal modes 
    !
    use psf
    use number
    use coord
    use vibsub
    use bases_fcm

    implicit none

    !real(chm_real) :: X(:), Y(:), Z(:)
    !real(chm_real) AMASS(*)

    LOGICAL LNOMA,LRAISE
    LOGICAL LFINIT,LDSCF,LENTRO
    
    INTEGER NFREQ, NADD

    real(chm_real)  STEP
    !type(nonbonddatastructure) :: BNBND
    !type(imagedatastructure) :: BIMAG
    
    INTEGER NSAVDD1
    LOGICAL QREST
   
    LNOMA=.FALSE.
    LRAISE=.FALSE.
    LFINIT=.FALSE.
    LDSCF=.FALSE.
    LENTRO=.FALSE.
    QREST=.FALSE.

    NADD=0
    NSAVDD1=NAT3  
    NFREQ=NAT3

    STEP=PT005

    CALL NORMDS(X,Y,Z,NAT3,BNBND,BIMAG,NFREQ,LNOMA,AMASS, &
       DDV,DDM,DDF,DDEV,NADD,LRAISE, &
       LFINIT,STEP,LDSCF,LENTRO, &
       NSAVDD1,QREST) 

    RETURN
  END SUBROUTINE DIMSNM
    
  SUBROUTINE SCORENMMOVE(GAMMA,X,Y,Z,WMAIN,IMOD,ATFRST,ATLAST)
    !
    !     Wrapper for DIMSUFORCES
    !     Computes bias based on info from NMs
    !
    use psf
    use number
    INTEGER IMOD
    INTEGER ATFRST,ATLAST
    real(chm_real) GAMMA(*)
    real(chm_real) X(*), Y(*), Z(*), WMAIN(*)
    INTEGER NNMDS
    INTEGER GNBEST

    NNMDS=NAT3
    GNBEST=NBEST

   CALL DIMSUFORCES(DDV,NNMDS,DDF,GAMMA &
       ,DIMTARGXA, DIMTARGYA, DIMTARGZA,DORIENT &
       ,XScratch, YScratch, ZScratch &
       ,DSCALE &
       ,IDIMS,NIDIMS, DIMSIDX,DIMACTIVE, X, Y ,Z &
       ,DXDIMS, DYDIMS, DZDIMS,IMIN,WMAIN,NMUNIT &
       ,BESTMODES,BESTSCORES,GNBEST,NCOMB,ICMODES,ICMIN,NMIDX &
       ,ATFRST,ATLAST)
    NMCOUNTER=NMCOUNTER+NCOMB
    RETURN
  END SUBROUTINE SCORENMMOVE

  SUBROUTINE DIMSUFORCES(DDV,NNMDS,DDF,GAMMA &
       ,DIMTARGXA, DIMTARGYA, DIMTARGZA,DORIENT &
       ,XScratch, YScratch, ZScratch &
       ,DSCALE &
       ,IDIMS,NIDIMS, DIMSIDX,DIMACTIVE, X, Y ,Z &
       ,DXDIMS, DYDIMS, DZDIMS,IMIN,WMAIN,NMUNIT &
       ,BESTMODES,BESTSCORES,GNBEST,NCOMB,ICMODES,ICMIN,NMIDX &
       ,ATFRST,ATLAST)
    !
    !     Evaluates possible moves from NM and picks combination of NMs
    !     that _moves_ the structure "towards" the target.
    !
    use consta
    use stream
    use psf
    use vector
    use number
    use exfunc, only: dimsorderr

    INTEGER ATFRST,ATLAST
    INTEGER I,J,K,MODE,NIDIMS,IMIN,IPOS,NCOMB
    INTEGER PCOMB, NNMDS!, DIMSCOMBINATORIAL
    INTEGER DIMSIDX(*)
    INTEGER DIMACTIVE(MAXAIM,3)
    INTEGER IDIMS(*)
    INTEGER NMUNIT,NBEST,BESTC,CNM,CNMIN,GNBEST
    INTEGER BESTMODES(*),ICMODES(*),ICMIN(*),NMIDX(*)
    !LOGICAL DORIENT,UFSR,VALIDMODE, DIMSORDERR
    LOGICAL DORIENT,UFSR,VALIDMODE
    real(chm_real) BESTSCORES(*)
    real(chm_real) XScratch(*), YScratch(*), ZScratch(*)
    real(chm_real) DDV(NATOM*3,*)
    real(chm_real) GAMMA(*)
    real(chm_real) X(*), Y(*), Z(*), WMAIN(*)
    real(chm_real) DDF(*)
    real(chm_real) DIMTARGXA(*), DIMTARGYA(*), DIMTARGZA(*)
    real(chm_real) DXDIMS(*), DYDIMS(*), DZDIMS(*)
    real(chm_real) DTX, DTY, DTZ,DTNORM, DSCALE
    real(chm_real) FACT
    real(chm_real) SCORE1, SCORE0
    !EXTERNAL DIMSORDERR
    EXTERNAL DIMSEXCH5R
    !     --- Score0. Score without biasing
    !     --- SCORE IS RMSD by Default. Modify DIMProg... to implement
    !         new scoring methods.
    NBEST=GNBEST
    CALL DIMSProgressSCORE(X,Y,Z, &
         DIMTARGXA,DIMTARGYA,DIMTARGZA, &
         ATFRST,ATLAST,NATOM, &
         WMAIN,AMASS,SCORE0)
    IF(PRNLEV >= 3) WRITE(OUTU,'(A23,F10.3)') &
         'DIMS> Progress Score0: ',SCORE0
    IF(PRNLEV > 4) WRITE(OUTU,'(A20,F10.3)') &
         'DIMS> Scaling factor:', DSCALE
    !     -- Clear forces
    DXDIMS(1:NATOM)=zero
    DYDIMS(1:NATOM)=zero
    DZDIMS(1:NATOM)=zero
    BESTSCORES(1:NBEST)=zero
    BESTMODES(1:NBEST)=0
    ICMODES(1:NBEST)=0
    ICMIN(1:NBEST)=0
    IMIN=-1
    BESTC=0
    !     -- Now updates forces with NM vectors
    DO I=1,NNMDS
       !     -- Reject negative and zero frequencies
       IF(DDF(I) > 1.0) THEN
          !     -- First copy coordinates to scratch
          xscratch(1:natom) = x(1:natom)
          yscratch(1:natom) = y(1:natom)
          zscratch(1:natom) = z(1:natom)
          CALL DIMSNMBIAS(XScratch,YScratch,ZScratch,DDV &
               ,DIMSIDX,NIDIMS,DIMACTIVE,I, .TRUE.,DSCALE)
          !     --- Score for the Ith mode
          CALL DIMSProgressSCORE(XScratch,YScratch,ZScratch, &
               DIMTARGXA,DIMTARGYA,DIMTARGZA, &
               ATFRST,ATLAST,NATOM, &
               WMAIN,AMASS,SCORE1)
          BESTC=BESTC+1
          IF(BESTC <= NBEST) THEN
             BESTSCORES(BESTC)=SCORE1
             BESTMODES(BESTC)=I
             IF(BESTC == NBEST) THEN
                CALL SORT(NBEST,DIMSEXCH5R,DIMSORDERR, &
                     BESTSCORES,BESTMODES,0,0,0,0,0,2)
                !CALL SORT(NBEST,DIMSEXCH5R,ORDERR, &
                !    BESTSCORES,BESTMODES,0,0,0,0,0,2)
             ENDIF
          ELSE
             IF(SCORE1 < BESTSCORES(NBEST)) THEN
                CALL DIMSCOMPXL(BESTSCORES,SCORE1,1,NBEST,IPOS)
                CALL DIMSINSERTRF(BESTSCORES,SCORE1,IPOS,NBEST)
                CALL DIMSINSERTRI(BESTMODES,I,IPOS,NBEST)
             ENDIF
          ENDIF
          IF(PRNLEV >= 4) &
               WRITE(OUTU,'(A22,I4,F10.5)') &
               'DIMS> Progress Score: ', I, SCORE1
       ENDIF
    ENDDO
    IF(BESTC < NBEST) THEN
       IF(PRNLEV >= 3) WRITE(OUTU,'(A45,I4)')  &
            'DIMS> Warning! Not enough good modes, using: ',BESTC
       NBEST=BESTC
       CALL SORT(NBEST,DIMSEXCH5R,DIMSORDERR, &
            BESTSCORES,BESTMODES,0,0,0,0,0,2)
    ENDIF
    DO K=1,NBEST
       ICMODES(K)=1
       CALL DIMSBUILDIDX(ICMODES,NMIDX,NBEST,CNM)
       CALL MODEINMATRIX(VALIDMODE,CNM)
       IF(BESTSCORES(K) < SCORE0.AND.VALIDMODE &
            .AND.IMIN <= 0) THEN
          ICMIN(K)=1
          SCORE0=BESTSCORES(K)
          IMIN=1
          CNMIN=CNM
       ENDIF
       ICMODES(K)=0
    ENDDO
    !     END FOR I=1, NORMAL MODE LOOP
    !     -- Evaluate possible combinations of moves I>1
    DO I=2,NCOMB
       PCOMB=DIMSCOMBINATORIAL(NBEST,I)
       ICMODES(1:NBEST) = ZERO
       !     Initial Combination -> ICMODES
       DO J=1,I
          ICMODES(J)=1
       ENDDO
       IF(PRNLEV >= 4) WRITE(OUTU,'(A22,I4)') &
            'DIMSNM> Combinations: ',PCOMB
       DO J=2,PCOMB
          CALL DIMSBUILDIDX(ICMODES,NMIDX,NBEST,CNM)
          CALL MODEINMATRIX(VALIDMODE,CNM)
          IF(VALIDMODE) THEN
             XScratch(1:NATOM)=zero
             YScratch(1:NATOM)=zero
             ZScratch(1:NATOM)=zero
             DO K=1,CNM
                CALL DIMSNMBIAS(XScratch,YScratch,ZScratch,DDV &
                     ,DIMSIDX,NIDIMS,DIMACTIVE,BESTMODES(NMIDX(K)), .TRUE. &
                     ,DSCALE)
             ENDDO
             CALL DIMSNORMALL(XScratch,YScratch,ZScratch, &
                  NIDIMS,DIMSIDX,DSCALE)
             CALL ADDVEC(XScratch,X,XScratch,NATOM)
             CALL ADDVEC(YScratch,Y,YScratch,NATOM)
             CALL ADDVEC(ZScratch,Z,ZScratch,NATOM)
             CALL DIMSProgressSCORE(XScratch,YScratch,ZScratch, &
                  DIMTARGXA,DIMTARGYA,DIMTARGZA, &
                  ATFRST,ATLAST,NATOM, &
                  WMAIN,AMASS,SCORE1)
             IF(PRNLEV >= 5) THEN
                WRITE(OUTU,'(A22,F10.5)') &
                     'DIMS> Progress Score: ', SCORE1
                DO K=1,CNM
                   WRITE(OUTU,'(5X,I4)') BESTMODES(NMIDX(K))
                ENDDO
             ENDIF
             IF(SCORE1 < SCORE0) THEN
                SCORE0=SCORE1
                IMIN=1
                CNMIN=CNM
                ICMIN(1:NBEST) = ICMODES(1:NBEST)
             ENDIF
          ENDIF
          !     Next combination -> ICMODES
          CALL DIMSNXCBN(NBEST, I, ICMODES)
       ENDDO
    ENDDO
    IF(NMUNIT /= -1) THEN
       J=NCOMB
       DO I=1,NBEST
          IF(ICMIN(I) == 1) THEN
             WRITE(NMUNIT,'(5X,I4)') BESTMODES(I)
             J=J-1
          ENDIF
       ENDDO
       DO I=1,J
          WRITE(NMUNIT,'(5X,I4)') -1
       ENDDO
    ENDIF
    !     --- Copy accepted move to DxArrays
    IF(IMIN > 0) THEN
       IF(PRNLEV >= 2) THEN
          WRITE(OUTU,'(A21)') 'DIMS> Move accepted: '
          WRITE(OUTU,'(F10.5)') SCORE0
          IF(PRNLEV >= 6) THEN
             CALL DUMPVECTOR(OUTU,BESTMODES,NBEST)
             CALL DUMPVECTORF(OUTU,BESTSCORES,NBEST)
          ENDIF
       ENDIF
       DO K=1,NBEST
          IF(ICMIN(K) == 1) THEN
             IF(PRNLEV >= 3) THEN
                WRITE(OUTU,'(5X,F10.5,I4,I4)') &
                     DDF(BESTMODES(K)), BESTMODES(K), K
             ENDIF
             CALL DIMSNMBIAS(DXDIMS,DYDIMS,DZDIMS,DDV &
                  ,DIMSIDX,NIDIMS,DIMACTIVE,BESTMODES(K),.TRUE. &
                  ,DSCALE)
          ENDIF
       ENDDO
       CALL DIMSNORMALL(DXDIMS,DYDIMS,DZDIMS, &
            NIDIMS,DIMSIDX,DSCALE)
    ENDIF
    RETURN
  END SUBROUTINE DIMSUFORCES

  SUBROUTINE FREEDIMS(LEVEL)
    !     THIS SUBROUTINE FREES DIMS RESOURCES
    !
    !     LEVEL - 1 Inner loop
    !     LEVEL - 2 All
    use memory
    use psf

    INTEGER LEVEL,I
    INTEGER NNMDS

    character(len=*), parameter :: procname = 'FREEDIMS'

    !
    !
    NNMDS=NAT3
    IF(QDIMS) THEN
       IF(DIMNM) THEN
          !     CALL VCLOSE(IUDIMSNM,'KEEP',ERROR)
#if KEY_PARALLEL==1
          call chmdealloc('dims.src','FREEDIMS','DDV',1,crl=DDV)
          call chmdealloc('dims.src','FREEDIMS','DDM',1,crl=DDM)
          call chmdealloc('dims.src','FREEDIMS','DDF',1,crl=DDF)
          call chmdealloc('dims.src','FREEDIMS','DDEV',1,crl=DDEV)
          call chmdealloc('dims.src','FREEDIMS','DDSCR',1,crl=DDSCR)
#else /**/
          if(qssmd.or.qnurx.or.qganm) then
             call chmdealloc('dims.src','FREEDIMS','DDV',2,crl=DDV)
          endif
        call chmdealloc('dims.src','FREEDIMS','DDV',NAT3*NAT3,crl=DDV)  
          call chmdealloc('dims.src','FREEDIMS','DDM',NATBNM,crl=DDM)
        call chmdealloc('dims.src','FREEDIMS','DDEV',NNMDS,crl=DDEV)
        call chmdealloc('dims.src','FREEDIMS','DDF',NNMDS,crl=DDF)
        call chmdealloc('dims.src','FREEDIMS','DDSCR',MAX(NAT3,NNMDS),crl=DDSCR)

#endif 

          IF(QEXPERT) THEN
             call chmdealloc('dims.src','FREEDIMS','NACTV',2,intg=NACTV)
             call chmdealloc('dims.src','FREEDIMS','NRISSH',2,intg=NRISSH)
             call chmdealloc('dims.src','FREEDIMS','NRICLH',2,intg=NRICLH)
          ELSE if(QHIGH) then
             call chmdealloc('dims.src','FREEDIMS','NACTV',2,intg=NACTV)
          ELSE if(QGEN) then
             call chmdealloc('dims.src','FREEDIMS','NACTV',2,intg=NACTV)
             call chmdealloc('dims.src','FREEDIMS','NRISSH',2,intg=NRISSH)
             call chmdealloc('dims.src','FREEDIMS','NRICLH',2,intg=NRICLH)
          ELSE if(QACTV) then
             call chmdealloc('dims.src','FREEDIMS','NRISSH',2,intg=NRISSH)
             call chmdealloc('dims.src','FREEDIMS','NRICLH',2,intg=NRICLH)
          ENDIF
          call chmdealloc('dims.src','FREEDIMS','IDXRES',NATOM,intg=IDXRES)
          call chmdealloc('dims.src','FREEDIMS','IDXATM',NATOM,intg=IDXATM)
          call chmdealloc('dims.src','FREEDIMS','ITREIG',6*NAT3,crl=ITREIG)
       ENDIF
       IF(LEVEL == 2) THEN
          IF(DIMNM) THEN
             call chmdealloc('dims.src','FREEDIMS','DIMSIDX',NIDIMS,intg=DIMSIDX)
             call chmdealloc('dims.src','FREEDIMS','BNMIDX',NATBNM,INTG=BNMIDX)
             call chmdealloc('dims.src','FREEDIMS','ISND',NATOM,INTG=ISND)
             IF (MTRAJ >= 1) THEN
             call chmdealloc_chm_iptr('dims.src','FREEDIMS','NMMATRIX',MTRAJ,NMMATRIX)
             call chmdealloc('dims.src','FREEDIMS','NMVEC',MTRAJ,INTG=NMVEC)
             ENDIF
             call chmdealloc('dims.src','FREEDIMS','DIMSSCORES',SAVER,CRL=DIMSSCORES)
             call chmdealloc('dims.src','FREEDIMS','BESTMODES',NBEST,INTG=BESTMODES)
             call chmdealloc('dims.src','FREEDIMS','ICMODES',NBEST,INTG=ICMODES)
             call chmdealloc('dims.src','FREEDIMS','ICMIN',NBEST,INTG=ICMIN)
             call chmdealloc('dims.src','FREEDIMS','BESTSCORES',NBEST,CRL=BESTSCORES)
             call chmdealloc('dims.src','FREEDIMS','NMIDX',NCOMB,INTG=NMIDX)
             call chmdealloc('dims.src','FREEDIMS','ASCALES',NBIAS,CRL=ASCALES)
             call chmdealloc('dims.src','FREEDIMS','XScratch',NATOM,CRL=XScratch)
             call chmdealloc('dims.src','FREEDIMS','YScratch',NATOM,CRL=YScratch)
             call chmdealloc('dims.src','FREEDIMS','ZScratch',NATOM,CRL=ZScratch)
             call chmdealloc('dims.src','FREEDIMS','DXDIMS',NATOM,CRL=DXDIMS)
             call chmdealloc('dims.src','FREEDIMS','DYDIMS',NATOM,CRL=DYDIMS)
             call chmdealloc('dims.src','FREEDIMS','DZDIMS',NATOM,CRL=DZDIMS)
             call chmdealloc('dims.src','FREEDIMS','VXDIMS',NATOM,CRL=VXDIMS)
             call chmdealloc('dims.src','FREEDIMS','VYDIMS',NATOM,CRL=VYDIMS)
             call chmdealloc('dims.src','FREEDIMS','VZDIMS',NATOM,CRL=VZDIMS)
             DO I=1,MTRAJ
                call chmdealloc('dims.src','FREEDIMS','DZDIMS',NATOM,intg=nmmatrix(i)%a)
             ENDDO
          ENDIF
          IF(QASC.OR.QATDIST) THEN
             call chmdealloc('dims.src','FREEDIMS','ICASL',NATOM,INTG=ICASL)
             call chmdealloc('dims.src','FREEDIMS','ICAIDX',NICA,INTG=ICAIDX)
             call chmdealloc('dims.src','FREEDIMS','ICAIDXN',NICAN,INTG=ICAIDXN)
             call chmdealloc('dims.src','FREEDIMS','CADIST0',NICANKNN,CRL=CADIST0)
             call chmdealloc('dims.src','FREEDIMS','CADIST',NICANKNN,CRL=CADIST)
             call chmdealloc('dims.src','FREEDIMS','CANEIGH',NICANKNN,intg=CANEIGH)
          ENDIf
          IF(QATDIST) THEN
             call chmdealloc('dims.src','FREEDIMS','DISTDESC0',NICAN*3,CRL=DISTDESC0)
             call chmdealloc('dims.src','FREEDIMS','DISTDESC',NICAN*3,CRL=DISTDESC)
          ENDIF
          QDIMS=.FALSE.
       ENDIF
       call chmdealloc(srcfile,procname,'dimactive',MAXAIM,3,intg=dimactive)
       call chmdealloc(srcfile,procname,'IDIMS',MAXAIM,intg=IDIMS)
       call chmdealloc(srcfile,procname,'dimdone',MAXAIM,intg=dimdone)
       call chmdealloc(srcfile,procname,'DIMTARGXA',MAXAIM,crl=DIMTARGXA)
       call chmdealloc(srcfile,procname,'DIMTARGYA',MAXAIM,crl=DIMTARGYA)
       call chmdealloc(srcfile,procname,'DIMTARGZA',MAXAIM,crl=DIMTARGZA)
       call chmdealloc(srcfile,procname,'WDIMS',MAXAIM,crl=WDIMS)
    ENDIF
    RETURN
  END SUBROUTINE FREEDIMS
!
!
  SUBROUTINE DIMSProgressSCORE(XI,YI,ZI,XP,YP,ZP, &
       ATFRST,ATLAST,NATOM, &
       WMAIN,AMASS,SCORE)
    !
  use number
  use stream
  use memory
  use parallel
  use corsubs

    implicit none

    !
    !     S=SQRT(dX^2 + dY^2 + dZ^2)
    !
    real(chm_real) DX
    !     Usually current structure
    real(chm_real) XI(*),YI(*),ZI(*),WMAIN(*)
    !     Usually target
    real(chm_real) XP(*),YP(*),ZP(*)
    real(chm_real) AMASS(*)
    real(chm_real) SCORE, TMASS
    INTEGER I, J
    INTEGER ATFRST,ATLAST
    INTEGER NATOM
    integer, allocatable, dimension(:) :: iscr
    !     --- If ORIENT set, re-orient target
    !     --- Orientation only needed if using RMS or
    !         any other method that is orientation-dependent
    IF(DORIENT) THEN
#if KEY_PARALLEL==1
       CALL PSYNC()
#if KEY_SPACDEC==1
       call SPACBR(XP,NATOM,ICPUMAP)
       call SPACBR(YP,NATOM,ICPUMAP)
       call SPACBR(ZP,NATOM,ICPUMAP)
       call SPACBR(XI,NATOM,ICPUMAP)
       call SPACBR(YI,NATOM,ICPUMAP)
       call SPACBR(ZI,NATOM,ICPUMAP)
#else /**/
       CALL VDGBR(XP,YP,ZP,0)
       CALL VDGBR(XI,YI,ZI,0)
#endif 
#endif 
       call chmalloc('dims.src','DIMSProgressSCORE','ISCR',NATOM+NATOM,intg=ISCR)
       CALL ORINTC(NATOM,XP,YP,ZP,XI,YI,ZI, &
            AMASS,.TRUE.,.TRUE.,ISCR,IDIMS,.FALSE.,WMAIN, &
            .FALSE.,.FALSE.)
       call chmdealloc('dims.src','DIMSProgressSCORE','ISCR',NATOM+NATOM,INTG=iscr)
    ENDIF
#if KEY_ANNLIB==1
    IF(UFSR) THEN
       CALL UltraFastRecon(XI,YI,ZI,XP,YP,ZP, &
            NATOM,IDIMS, &
            AMASS,SCORE)
       SCORE=-1.0*SCORE
    ELSE IF(QASC) THEN
       CALL DIMSCALPHA(XI,YI,ZI,XP,YP,ZP, &
            NATOM, &
            IDIMS,NIDIMS,DIMSIDX, &
            WMAIN,AMASS,SCORE, &
            ICASL,ICAIDXN, &
            NICAN,CADIST0,CADIST, &
            CAKNN,CAEPS,HECANEIGH,NICANKNN)
    ELSE IF(QATDIST) THEN
       CALL DIMSATOMDIST(XI,YI,ZI,XP,YP,ZP, &
            NATOM, &
            IDIMS,NIDIMS,DIMSIDX, &
            WMAIN,AMASS,SCORE, &
            ICASL,ICAIDXN, &
            NICAN,CADIST0,CADIST, &
            CAKNN,CAEPS,CANEIGH,NICANKNN,  &
            DISTDESC,DISTDESC0)
    ELSE
#endif 
       CALL DIMSRMSD(XI,YI,ZI,XP,YP,ZP,ATFRST,ATLAST, &
            IDIMS,NIDIMS,DIMSIDX,WMAIN,AMASS,SCORE)
#if KEY_PARALLEL==1
       !     Sum for parallel ...
       CALL PSYNC()
       IF(PRNLEV >= 5)  &
            WRITE(OUTU,'(A22,I4,F10.4)')  &
            'DIMS> Score for node: ',MYNOD,SCORE
       CALL GCOMB(SCORE,1)
#endif 
       SCORE=SQRT(SCORE/DIMSMASS)
#if KEY_ANNLIB==1
    ENDIF 
#endif
    RETURN
  END SUBROUTINE DIMSProgressSCORE

  SUBROUTINE DIMSRMSD(XI,YI,ZI,XP,YP,ZP, &
       ATFRST,ATLAST, &
       IDIMS,NIDIMS,DIMSIDX, &
       WMAIN,AMASS,SCORE)
  use number
  use stream
    real(chm_real) XI(*),YI(*),ZI(*),WMAIN(*)
    real(chm_real) XP(*),YP(*),ZP(*)
    real(chm_real) AMASS(*)
    real(chm_real) SCORE, TMASS
    INTEGER NIDIMS, I, J, ATFRST,ATLAST
    INTEGER DIMSIDX(*),IDIMS(*)
    !
    !     -- Compute weighted RMS
    SCORE=ZERO
#if KEY_PARALLEL==1
    DO I=ATFRST,ATLAST 
#endif
#if KEY_PARALLEL==0
    DO J=1,NIDIMS      
#endif
#if KEY_PARALLEL==0
       I=DIMSIDX(J)    
#endif
#if KEY_PARALLEL==1
       IF(IDIMS(I) == 1) THEN 
#endif
          SCORE=SCORE+((XI(I)-XP(I))**2 +  &
               (YI(I)-YP(I))**2 + &
               (ZI(I)-ZP(I))**2)*AMASS(I)
#if KEY_PARALLEL==1
       ENDIF                  
#endif
    ENDDO
    RETURN
  END SUBROUTINE DIMSRMSD
!
  SUBROUTINE DIMSCALPHA(XI,YI,ZI,XP,YP,ZP, &
       NATOM, &
       IDIMS,NIDIMS,DIMSIDX, &
       WMAIN,AMASS,SCORE, &
       ICASL,ICAIDXN,NICAN,CADIST0,CADIST, &
       KNN,EPS,CANEIGH,NICANKNN)
  use number
  use stream
    implicit none
    real(chm_real) XI(*),YI(*),ZI(*),WMAIN(*)
    real(chm_real) XP(*),YP(*),ZP(*)
    real(chm_real) AMASS(*)
    real(chm_real) SCORE, TMASS
    real(chm_real) CADIST0(*),CADIST(*)
    INTEGER NIDIMS,NATOM
    INTEGER I,J,K,L
    INTEGER IAT
    INTEGER DIMSIDX(*),IDIMS(*),CANEIGH(*)
    INTEGER NICAN,NICANKNN
    INTEGER ICASL(*)
    INTEGER ICAIDXN(*)
    !     EXTERNAL CALLANN
    INTEGER KNN
    real(chm_real) EPS
    !     CALL CALLANN(KNN,EPS,NICA,XI,YI,ZI,ICAIDX,CADIST)
    SCORE=0.0
    DO I=1,NICAN
       J=ICAIDXN(I)
       DO L=1,KNN
          IAT=(I-1)*KNN + L
          K=CANEIGH(IAT)
          CADIST(IAT)=(XI(J)-XI(K))**2 &
               +(YI(J)-YI(K))**2 &
               +(ZI(J)-ZI(K))**2
          CADIST(IAT)=SQRT(CADIST(IAT))
          SCORE=SCORE+ &
               ABS(CADIST0(IAT)-CADIST(IAT))
       ENDDO
    ENDDO
    RETURN
  END SUBROUTINE DIMSCALPHA
!
  SUBROUTINE DIMSATOMDIST(XI,YI,ZI,XP,YP,ZP, &
       NATOM, &
       IDIMS,NIDIMS,DIMSIDX, &
       WMAIN,AMASS,SCORE, &
       ICASL,ICAIDXN,NICAN,CADIST0,CADIST, &
       KNN,EPS,CANEIGH,NICANKNN,DISTDESC,DISTDESC0)
  use number
  use stream
    real(chm_real) XI(*),YI(*),ZI(*),WMAIN(*)
    real(chm_real) XP(*),YP(*),ZP(*)
    real(chm_real) AMASS(*)
    real(chm_real) SCORE, TMASS
    real(chm_real) CADIST0(*),CADIST(*)
    INTEGER NIDIMS,NATOM
    INTEGER I,J,K,L
    INTEGER IAT
    INTEGER DIMSIDX(*),IDIMS(*),CANEIGH(*)
    INTEGER NICAN,NICANKNN
    INTEGER ICASL(*)
    INTEGER ICAIDXN(*)
    real(chm_real) DISTDESC(*),DISTDESC0(*)
    !     EXTERNAL CALLANN
    INTEGER KNN
    real(chm_real) EPS
    !     CALL CALLANN(KNN,EPS,NICA,XI,YI,ZI,ICAIDX,CADIST)
    SCORE=0.0
    CALL BUILDDESCRIPTOR(XI,YI,ZI, &
         CANEIGH,ICAIDXN, &
         DISTDESC,CADIST,NICAN,KNN)
    CALL UFSMANHATTAN(DISTDESC0,DISTDESC,NICAN*3,SCORE)
    RETURN
  END SUBROUTINE DIMSATOMDIST
  !
  SUBROUTINE BUILDDESCRIPTOR(X,Y,Z, &
       CANEIGH,ICAIDXN, &
       DESC,CADIST,NICAN,KNN)
    !     Build atomic descriptor for another progress metric
    real(chm_real) X(*),Y(*),Z(*)
!!! This is BUG (MH09) ::    real(chm_real) CADIST(*),CANEIGH(*)
    real(chm_real) CADIST(*)
    real(chm_real) DESC(*)
    real(chm_real) AVG
    INTEGER ICAIDXN(*),CANEIGH(*)
    INTEGER NICAN,KNN
    !     LOCAL
    INTEGER I,J,L,K,IAT
    real(chm_real) STDEV,SKEW,TDIST
    DO I=1,NICAN
       J=ICAIDXN(I)
       AVG=0.0
       DO L=1,KNN
          IAT=(I-1)*KNN + L
          K=CANEIGH(IAT)
          CADIST(IAT)=(X(J)-X(K))**2 &
               +(Y(J)-Y(K))**2 &
               +(Z(J)-Z(K))**2
          CADIST(IAT)=SQRT(CADIST(IAT))
          AVG=CADIST(IAT)+AVG
       ENDDO
       STDEV=0.0
       SKEW=0.0
       DO L=1,KNN
          IAT=(I-1)*KNN + L
          TDIST=(AVG-CADIST(IAT))**2
          STDEV=STDEV+TDIST
          SKEW=SKEW+TDIST*(AVG-CADIST(IAT))
       ENDDO
       STDEV=STDEV/(KNN-1)
       SKEW=SKEW/(KNN-1)
       DESC(I*3-2)=AVG
       DESC(I*3-1)=SQRT(STDEV)
       DESC(I*3)=SKEW**(1.0/3.0)
    ENDDO
    RETURN
  END SUBROUTINE BUILDDESCRIPTOR
!
  SUBROUTINE DIMSDONE(X,Y,Z,WMAIN,QPRINT,ISTEP,ATFRST,ATLAST)
!
!     Check if DIMS work is done. Controls program flow.
!
  use psf
  use stream
#if KEY_PARALLEL==1
  use parallel 
#endif
#if KEY_TMD==1
  use tmd
#endif 
    !
    INTEGER I, J, DIMCNT,ISTEP
    LOGICAL QPRINT
    real(chm_real) X(*), Y(*), Z(*), WMAIN(*)
    real(chm_real) SCORE0
    INTEGER ATFRST,ATLAST
    !
    DIMCNT=0
    DORIENT=(MOD(ISTEP,IORIENT) == 0.OR.ISTEP == 1).AND.IORIENT /= -1
    CALL DIMSProgressSCORE(X,Y,Z, &
         DIMTARGXA,DIMTARGYA,DIMTARGZA, &
         ATFRST,ATLAST,NATOM, &
         WMAIN,AMASS,SCORE0)
    DORIENT=.FALSE.
#if KEY_PARALLEL==1
    IF(MYNOD == 0) THEN 
#endif
       IF(SCORE0 < DIMCUTOFF) THEN
          IF(PRNLEV  >=  2.AND.DIMNMON) THEN
             WRITE(OUTU,'(A37,F10.4,F10.4)') &
                  'DIMSNM> Cutoff reached shutting down ', &
                  SCORE0, DIMCUTOFF
          ENDIF
          DIMNMON=.FALSE.
          IF(DIMNM) THEN
             IF(DCART) THEN
                CALL FREEDIMS(1)
                DIMNM=.FALSE.
                DIMCUTOFF=-1.0
             ELSE IF(DHARD) THEN
                CALL FREEDIMS(1)
                DIMNM=.FALSE.
                DHARD=.TRUE.
             ELSE IF(DHALT) THEN
                DDONE=.TRUE.
                IF(PRNLEV  >=  3)  &
                     WRITE(OUTU,'(A30,F10.4,F10.4)')  &
                     'DIMS> Reached cutoff halting. ', &
                     SCORE0,DIMCUTOFF
#if KEY_TMD==1
             ELSE IF(DTMD) THEN
                CALL FREEDIMS(1)
                DIMNM=.FALSE.
                QTMD=.TRUE.
                IF(PRNLEV  >=  2.AND.DIMNMON) THEN
                   WRITE(OUTU,'(A28)') 'DIMSNM> TMD is now enabled. '
                ENDIF
                ixtar(1:natom)=dimtargxa(1:natom)
                iytar(1:natom)=dimtargya(1:natom)
                iztar(1:natom)=dimtargza(1:natom)
#endif 
             ELSE
                DIMNM=.FALSE.
             ENDIF
          ELSE IF(DCART) THEN
             IF(DHALT) THEN
                DDONE=.TRUE.
                IF(PRNLEV  >=  3)  &
                     WRITE(OUTU,'(A30,F10.4,F10.4)') &
                     'DIMS> Reached cutoff halting. ', &
                     SCORE0,DIMCUTOFF
             ELSE IF(DHARD) THEN
                DCART=.FALSE.
                DHARD=.TRUE.
                IF(PRNLEV  >=  3)  &
                     WRITE(OUTU,'(A41)') &
                     'DIMS> Reached cutoff, switching to hard. '
             ELSE
                DCART=.FALSE.
                IF(PRNLEV  >=  3)  &
                     WRITE(OUTU,'(A39)')  &
                     'DIMS> Reached cuttof, turning off bias.'
             ENDIF
          ENDIF
       ENDIF
#if KEY_PARALLEL==1
    ENDIF 
#endif
#if KEY_PARALLEL==1
    CALL PSND4(DDONE,1) 
#endif
#if KEY_PARALLEL==1
    CALL PSND4(DCART,1) 
#endif
    RETURN
  END SUBROUTINE DIMSDONE

  SUBROUTINE UPDATEDIMSPOS(X,Y,Z,ATFRST,ATLAST)
  use deriv
  use number

    real(chm_real) X(*), Y(*), Z(*)
    INTEGER I,J
    INTEGER ATFRST,ATLAST
#if KEY_PARALLEL==1
    DO I=ATFRST,ATLAST 
#endif
#if KEY_PARALLEL==0
    DO J=1,NIDIMS      
#endif
#if KEY_PARALLEL==0
       I=DIMSIDX(J)    
#endif
#if KEY_PARALLEL==1
       IF(IDIMS(I) == 1) THEN 
#endif
          X(I)=X(I) + DXDIMS(I)
          Y(I)=Y(I) + DYDIMS(I)
          Z(I)=Z(I) + DZDIMS(I)
#if KEY_PARALLEL==1
       ENDIF 
#endif
    ENDDO
    RETURN
  END SUBROUTINE UPDATEDIMSPOS

  SUBROUTINE DIMSDYNFIX(IMOVE,FLAG)
  use deriv
  use number

    INTEGER IMOVE(*)
    INTEGER FLAG
    INTEGER I,J

    DO J=1,NIDIMS
       I=DIMSIDX(J)
       IMOVE(I)=FLAG
    ENDDO
    RETURN
  END SUBROUTINE DIMSDYNFIX

  SUBROUTINE DIMSWRITNM(X,Y,Z, &
       DXDIMS, DYDIMS, DZDIMS, &
       XScratch,YScratch,ZScratch, &
       IUDIMSNM, &
       NSAVC, DELTA, ISTEP, NSTEP,SKIP)
    use psf
    use ctitla
    use cvio
    use number
    INTEGER ISTEP, NSAVC, NSTEP
    INTEGER I,J,K,MODE
    INTEGER IUDIMSNM,IMIN
    INTEGER SKIP
    LOGICAL NMPRINT
    real(chm_real) X(*),Y(*),Z(*)
    real(chm_real) XScratch(*),YScratch(*),ZScratch(*)
    real(chm_real) DXDIMS(*), DYDIMS(*), DZDIMS(*)
    real(chm_real) DELTA
    IF(SKIP == 0) THEN
       DO J=1,NATOM
          XScratch(J)=X(J) + DXDIMS(J)
          YScratch(J)=Y(J) + DYDIMS(J)
          ZScratch(J)=Z(J) + DZDIMS(J)
       ENDDO
       CALL WRITCV(XScratch,YScratch,ZScratch, &
#if KEY_CHEQ==1
            (/ZERO/), .FALSE.,  & 
#endif
            NATOM, (/0/), NATOM, 1,  &
            ISTEP, NATOM*3, DELTA, NSAVC, NSTEP, &
            TITLEA, NTITLA, IUDIMSNM, &
            .FALSE.,.FALSE.,(/0/),.FALSE.,(/ZERO/))
    ELSE
       CALL WRITCV(X,Y,Z, &
#if KEY_CHEQ==1
            (/ZERO/), .FALSE.,  & 
#endif
            NATOM, (/0/), NATOM, 1,  &
            ISTEP, NATOM*3, DELTA, NSAVC, NSTEP, &
            TITLEA, NTITLA, IUDIMSNM, &
            .FALSE.,.FALSE.,(/0/),.FALSE.,(/ZERO/))
    ENDIF
    RETURN
  END SUBROUTINE DIMSWRITNM

  SUBROUTINE DIMSHARD(X,Y,Z,XT,YT,ZT, &
       ISTEP,NSTEP,NIDIMS,DIMSIDX, &
       DXDIMS,DYDIMS,DZDIMS,IDIMS, &
       ATFRST,ATLAST)
    !
    !     DIMS-Hard version
    !
  use deriv
  use number
#if KEY_PARALLEL==1
  use parallel 
#endif
    real(chm_real) X(*), Y(*), Z(*)
    real(chm_real) XT(*), YT(*), ZT(*)
    real(chm_real) DXDIMS(*), DYDIMS(*), DZDIMS(*)
    INTEGER I,J
    INTEGER ISTEP,NSTEP,NIDIMS
    INTEGER ATFRST,ATLAST
    INTEGER DIMSIDX(*),IDIMS(*)
    IF(ISTEP /= NSTEP) THEN
#if KEY_PARALLEL==1
       DO I=ATFRST,ATLAST 
#endif
#if KEY_PARALLEL==0
       DO J=1,NIDIMS      
#endif
#if KEY_PARALLEL==0
          I=DIMSIDX(J)    
#endif
#if KEY_PARALLEL==1
          IF(IDIMS(I) == 1) THEN 
#endif
             DXDIMS(I)=(XT(I)-X(I))/(NSTEP-ISTEP)
             DYDIMS(I)=(YT(I)-Y(I))/(NSTEP-ISTEP)
             DZDIMS(I)=(ZT(I)-Z(I))/(NSTEP-ISTEP)
#if KEY_PARALLEL==1
          ENDIF     
#endif
       ENDDO
    ENDIF
    RETURN
  END SUBROUTINE DIMSHARD
  !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
  !
  !     DIMS-Score Subroutines
  !
  !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

  SUBROUTINE UPDATEDIMSEQ(ENERGYVALUE)
    !     Unbiased Energy
    real(chm_real) ENERGYVALUE
    DIMSEQ=ENERGYVALUE
    RETURN
  END SUBROUTINE UPDATEDIMSEQ

  SUBROUTINE UPDATEDIMSED(ENERGYVALUE)
    !     Biased Energy
    implicit none
    real(chm_real) ENERGYVALUE
    DIMSED=ENERGYVALUE
    RETURN
  END SUBROUTINE UPDATEDIMSED

  SUBROUTINE DIMSSCORE(QPRINT)
  use stream
  use param_store, only: set_param

    LOGICAL QPRINT
    INTEGER I
    real(chm_real) TSCORE
    TSCORE=EXP(DIMSED-DIMSEQ)
    LDIMSSCR=LDIMSSCR+(DIMSED-DIMSEQ)
    IF(CAVER == SAVER) THEN
       DIMSE0=0.0
       DO I=1,SAVER
          DIMSE0=DIMSE0 + DIMSSCORES(I)
       ENDDO
       DIMSE0=TSCORE/SAVER
       DIMSE0=1/DIMSE0
       IF(PRNLEV >= 3) &
            WRITE(OUTU,'(A20,F10.4)') &
            'DIMS> DIMS Scaling: ', DIMSE0
       DIMSSCR=1.0
       DO I=1,SAVER
          DIMSSCR=DIMSSCR*DIMSE0*DIMSSCORES(I)
       ENDDO
       DIMSSCR=DIMSSCR* &
            (DIMSE0*TSCORE)
       CAVER=CAVER+1
    ELSEIF (CAVER > SAVER) THEN
       DIMSSCR=DIMSSCR* &
            (DIMSE0*TSCORE)
    ELSE
       CAVER=CAVER+1
       CALL DIMSUPDATESCORES(DIMSSCORES,CAVER,TSCORE)
    ENDIF
    IF(DSCUNIT /= -1) THEN
       WRITE(DSCUNIT,'(5X,E12.5,5X,E12.5)') &
            TSCORE,LDIMSSCR
    ENDIF
    IF(QPRINT) THEN
       WRITE(OUTU,'(A18,E12.5,F15.4,F15.4)')  &
            'DIMS> DIMS Score: ', DIMSSCR, DIMSED, DIMSEQ
       WRITE(OUTU,'(A21,E12.5)') &
            'DIMS> DIMS LogScore: ',LDIMSSCR
    ENDIF
    call set_param('DSCORE',DIMSSCR)
    call set_param('LDSCORE',LDIMSSCR)
    RETURN
  END SUBROUTINE DIMSSCORE

  SUBROUTINE DIMSUPDATESCORES(SCORES,INDEX,VALUE)
    implicit none
    real(chm_real) SCORES(*)
    real(chm_real) VALUE
    INTEGER INDEX
    SCORES(INDEX)=VALUE
    RETURN
  END SUBROUTINE DIMSUPDATESCORES

  !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
  !
  !     NM DIMS
  !     SELF Avoidance NM subroutines
  !
  !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

  SUBROUTINE READNM(NMVEC,NMMATRIX,MTRAJ,IUNIT,NTITLNM,TITLENM &
       ,NMUNIT)
  use chm_types
  use memory
  use stream
    INTEGER MTRAJ
    INTEGER IUNIT,NTITLNM,NMUNIT
    character(len=*) TITLENM(*)
    type(chm_iarray),dimension(:) :: NMMATRIX
    INTEGER NMVEC(*)
    INTEGER I,J,N,NLEN
    INTEGER IOS
    character(len=10) ASTK
    CALL RDTITL(TITLENM,NTITLNM,IUNIT,0)
    J=0
    NLEN=0
10  J=J+1
    DO WHILE(.TRUE.)
       READ(IUNIT,'(5X,I4)',ERR=100) N
       NLEN=NLEN+1
    ENDDO
100 call chmalloc('dims.src','READNM','NMMATRIX(J)',NLEN+1,intg=NMMATRIX(J)%a)
    NMVEC(J)=NLEN
    IF(PRNLEV >= 4)  &
         WRITE(OUTU,'(A8,I4)') 'READNM> ',NLEN
    IF(J < MTRAJ) THEN
       NLEN=0
       GOTO 10
    ENDIF
    REWIND(IUNIT)
    CALL RDTITL(TITLENM,NTITLNM,IUNIT,0)
    DO I=1,MTRAJ
       IF(PRNLEV >= 6) THEN
          WRITE(OUTU,'(A21,I4,I4)')  &
               '<READNM> Trajectory: ', I,NMVEC(I)
       ENDIF
       CALL READNMLINE(NMMATRIX(I)%a,NMVEC(I),IUNIT)
       READ(IUNIT,'(A10)') ASTK
    ENDDO
    IF(PRNLEV >= 6) THEN
       WRITE(OUTU,'(A26,I4,I4)') &
            '<READNM> NM-Trajectories: ', MTRAJ, J
       DO I=1,MTRAJ
          WRITE(OUTU,'(A21,I4,I4)') &
               '          Trajectory ', I, NMVEC(I)
          CALL DUMPVECTOR(OUTU,NMMATRIX(I)%a,NMVEC(I))
       ENDDO
    ENDIF
    RETURN
  END SUBROUTINE READNM
!
  SUBROUTINE READNMLINE(NMMATRIX,N,IUNIT)
  use stream
    INTEGER I,N,IUNIT,IOS
    INTEGER NMMATRIX(*)
    DO I=1,N
       READ(IUNIT,'(5X,I4)',ERR=100,IOSTAT=ios) NMMATRIX(I)
    ENDDO
    RETURN

100 IF(IOS == 11) THEN
       N=N-1
    ELSE
       IF(PRNLEV >= 0)  &
            WRITE(OUTU,'(A25,I4,I4,I4,I4)') &
            'DIMS> Error reading file: ', N,I, NMMATRIX(I), IOS
       CALL WRNDIE(0,'<DIMS>','Unable to read NM file, check format.')
    ENDIF
    !      IF(NMUNIT /= -1) CALL DUMPVECTOR(NMUNIT,NMMATRIX,N)
    !                      WRITE(NMUNIT,*) '', (NMMATRIX(I),I=1,N)
    !      READ(IUNIT,'(A40)') ASTK
    !      IF(NMUNIT /= -1) WRITE(NMUNIT, '(A40)') ASTK
    RETURN
  END SUBROUTINE READNMLINE
!

  SUBROUTINE DUMPNMMATRIX
    !
    !     Prints dims matrix. PRNLEV protected by invoking function
    !
    use stream
    INTEGER I,J,K,VC
    INTEGER IMODE,NMAX
    WRITE(OUTU,'(A32)')  '================================'
    WRITE(OUTU,'(A32)')  '======= VECTOR WATCH ==========='
    WRITE(OUTU,'(A32)')  '================================'
    DO I=1,MTRAJ
       NMAX=NMVEC(I)
       DO J=1,NMAX
          CALL GETMODEIJ(I,J,IMODE)
          WRITE(OUTU,'(I4)') IMODE
       ENDDO
    ENDDO
    RETURN
  END SUBROUTINE DUMPNMMATRIX
!
  SUBROUTINE GETMODEIJ(I,J,MODE)
    INTEGER I,J,MODE
    INTEGER NADDRESS
    mode = nmmatrix(i)%a(j)
    RETURN
  END SUBROUTINE GETMODEIJ

!
  SUBROUTINE MODEINMATRIX(VALIDMODE,CNM)
    use stream
    INTEGER I,J,K,VC,IMODE,LNMC,CNM
    LOGICAL VALIDMODE
    VALIDMODE=.TRUE.
    VC=0
    DO I=1,MTRAJ
       DO LNMC=MAX(0,NMCOUNTER-NWIND),NMCOUNTER+NWIND,NCOMB
          IF(LNMC <= NMVEC(I)) THEN
             VC=0
             IF(PRNLEV >= 6)  &
                  WRITE (OUTU,'(A7,I4,I4,I4)') &
                  'Modes: ',NMVEC(I),I,NCOMB
             DO K=LNMC+1,LNMC+NCOMB
                CALL GETMODEIJ(I,K,IMODE)
                IF(PRNLEV >= 5) WRITE (OUTU,'(A10,I4,I4,I4,I4)')  &
                     'IMODE6: ',IMODE,I,LNMC,K
                IF(IMODE == -1 .AND. CNM < NCOMB) THEN
                   VC=VC+1
                ELSE
                   DO J=1,CNM
                      IF(IMODE ==  bestmodes(nmidx(j))) THEN
                         VC=VC+1
                         IF(PRNLEV >= 4) WRITE(OUTU,'(A31,I4,I4,I4)') &
                              'SCORENMMOVE> Mode already used ' &
                              ,BESTMODES(J), K , VC
                      ENDIF
                   ENDDO
                ENDIF
             ENDDO
             IF(VC >= NCOMB) THEN
                VALIDMODE=.FALSE.
                IF(PRNLEV >= 2) WRITE(OUTU,'(A46,I4)') &
                     'SCORENMMOVE> Combination already used skipping', VC
                RETURN
             ENDIF
          ENDIF
       ENDDO
    ENDDO
    VALIDMODE=VC < NCOMB
    IF(.NOT.VALIDMODE.AND.PRNLEV >= 2) WRITE(OUTU,'(A46)') &
         'SCORENMMOVE> Combination already used skipping'
    RETURN
  END SUBROUTINE MODEINMATRIX
  !
  SUBROUTINE DUMPVECTOR(IUNIT,VECTOR,N)
    INTEGER VECTOR(*)
    INTEGER I,N
    INTEGER IUNIT
    DO I=1,N
       WRITE(IUNIT,'(5X,I4)') VECTOR(I)
    ENDDO
    RETURN
  END SUBROUTINE DUMPVECTOR

  SUBROUTINE DUMPVECTORF(IUNIT,VECTOR,N)
    real(chm_real) VECTOR(*)
    INTEGER I,N
    INTEGER IUNIT
    DO I=1,N
       WRITE(IUNIT,'(5X,F10.5)') VECTOR(I)
    ENDDO
    RETURN
  END SUBROUTINE DUMPVECTORF


  SUBROUTINE CLOSENMFILE
    use stream
    use ctitla
    INTEGER I,J
    IF(QDIMS) THEN
       IF(NMUNIT == -1) RETURN
       WRITE(NMUNIT,'(A10)') '##########'
       IF(PRNLEV >= 3) WRITE(OUTU,'(A33,I4)') &
            '<CLOSENMFILE> Normal modes used: ' &
            ,NMCOUNTER
       !      ENDFILE(NMUNIT)
    ENDIF
    RETURN
  END SUBROUTINE CLOSENMFILE

  SUBROUTINE DIMSNORMALL(DX,DY,DZ,NIDIMS,DIMSIDX,DSCALE)
    real(chm_real) DX(*),DY(*),DZ(*),DTNORM,DSCALE
    INTEGER I,J,NIDIMS,DIMSIDX(*)
    DO J=1,NIDIMS
       I=DIMSIDX(J)
       DTNORM=DX(I)**2+DY(I)**2+DZ(I)**2
       IF(DTNORM /= 0) THEN
          DTNORM=DSCALE/SQRT(DTNORM)
          DX(I)=DX(I)*DTNORM
          DY(I)=DY(I)*DTNORM
          DZ(I)=DZ(I)*DTNORM
       ENDIF
    ENDDO
    RETURN
  END SUBROUTINE DIMSNORMALL
  !CCCCCCCCCCCCCCCCCCC
  !C
  !C
  !C  UltraFastShape Recognition Algorithm
  !C  P. Ballester and W. Richards, Proc R. Soc. A
  !C  DOI:10.1098/rspa.2007.1823
  !C
  !C  Implemented by: Juan R. Perilla
  !C
  !CCCCCCCCCCCCCCCCCCC


  SUBROUTINE UltraFastRecon(XI,YI,ZI,XP,YP,ZP, &
       NATOM,ISEL, &
       AMASS,SCORE)
  use stream
    real(chm_real) XI(*),YI(*),ZI(*)
    real(chm_real) XP(*),YP(*),ZP(*)
    real(chm_real) AMASS(*)
    real(chm_real) SCORE, TMASS
    INTEGER I, J, NATOM
    INTEGER ISEL(*)
    !     UFS Internal DATA
    !     1. CTD - Centroid coordinates
    !     2. CST - Closest atom to CTD
    !     3. FCT - Farthest atom to CTD
    !     4. FTF - Farthest atom to FCT
    !     MOLDES* - Molecular descriptor for I and P
    !      real(chm_real) CTD(3),CST(3),FCT(3),FTF(3)
    real(chm_real) CENTERS(4,3)
    real(chm_real) MOLDESI(12),MOLDESP(12)
    !     FIRST FOR I
    CALL UFSCTD(XI,YI,ZI,AMASS,ISEL,NATOM,CENTERS,1)
    CALL UFSCF(XI,YI,ZI,ISEL,NATOM,CENTERS,1,2,3,.TRUE.,.TRUE.)
    CALL UFSCF(XI,YI,ZI,ISEL,NATOM,CENTERS,3,0,4,.FALSE.,.TRUE.)
    DO I=0,3
       CALL UFSATOMAVE(XI,YI,ZI,ISEL,CENTERS,I+1,MOLDESI(3*I+1),NATOM)
       CALL UFSATOMDIST(XI,YI,ZI,ISEL,CENTERS,I+1, &
            MOLDESI(3*I+1),MOLDESI(3*I+2),MOLDESI(3*I+3), NATOM)
    ENDDO
    IF(PRNLEV >= 6) THEN
       WRITE(OUTU,'(A28)')'UFSR> Molecular descriptors '
       WRITE(OUTU,'(A14)')'UFSR> MOLDESI '
       DO I=1,12
          WRITE(OUTU,'(F10.3)') MOLDESI(I)
       ENDDO
       WRITE(OUTU,'(A15)')'UFSR> CENTERS I'
       DO I=1,4
          WRITE(OUTU,'(F10.3,F10.3,F10.3)') &
               CENTERS(I,1), CENTERS(I,2), CENTERS(I,3)
       ENDDO
    ENDIF
    !     THEN FOR P
    CALL UFSCTD(XP,YP,ZP,AMASS,ISEL,NATOM,CENTERS,1)
    CALL UFSCF(XP,YP,ZP,ISEL,NATOM,CENTERS,1,2,3,.TRUE.,.TRUE.)
    CALL UFSCF(XP,YP,ZP,ISEL,NATOM,CENTERS,3,0,4,.FALSE.,.TRUE.)
    DO I=0,3
       CALL UFSATOMAVE(XP,YP,ZP,ISEL,CENTERS,I+1,MOLDESP(3*I+1),NATOM)
       CALL UFSATOMDIST(XP,YP,ZP,ISEL,CENTERS,I+1, &
            MOLDESP(3*I+1),MOLDESP(3*I+2),MOLDESP(3*I+3), NATOM)
    ENDDO
    CALL UFSMANHATTAN(MOLDESI,MOLDESP,12,SCORE)
    SCORE=1+SCORE
    SCORE=1/SCORE
    IF(PRNLEV >= 6) THEN
       WRITE(OUTU,'(A14)')'UFSR> MOLDES P'
       DO I=1,12
          WRITE(OUTU,'(F10.4)') MOLDESP(I)
       ENDDO
       WRITE(OUTU, '(A16)')'UFSR> CENTERS P '
       DO I=1,4
          WRITE(OUTU,'(F10.4,F10.4,F10.4)') &
               CENTERS(I,1), CENTERS(I,2), CENTERS(I,3)
       ENDDO
       WRITE(OUTU,'(A13,F10.4)') 'UFSR> Score: ',SCORE
    ENDIF
    RETURN
  END SUBROUTINE UltraFastRecon

  SUBROUTINE UFSCTD(X,Y,Z,AMASS,ISEL,NATOM,CENTERS,CTD)
    !     Computes structure centroid then it is stored in CENTERS(CTD)
    real(chm_real) X(*),Y(*),Z(*)
    real(chm_real) AMASS(*)
    real(chm_real) CENTERS(4,3)
    !      real(chm_real) CTD(3)
    real(chm_real) TMASS
    INTEGER ISEL(*)
    INTEGER CTD
    INTEGER NATOM
    INTEGER I
    CENTERS(CTD,1)=0.0d0
    CENTERS(CTD,2)=0.0d0
    CENTERS(CTD,3)=0.0d0
    TMASS=0.0d0
    DO I=1,NATOM
       IF(ISEL(I) == 1) THEN
          CENTERS(CTD,1)=CENTERS(CTD,1)+X(I)*AMASS(I)
          CENTERS(CTD,2)=CENTERS(CTD,2)+Y(I)*AMASS(I)
          CENTERS(CTD,3)=CENTERS(CTD,3)+Z(I)*AMASS(I)
          TMASS=TMASS+AMASS(I)
       ENDIF
    ENDDO
    DO I=1,3
       CENTERS(CTD,I)=CENTERS(CTD,I)/TMASS
    ENDDO
    RETURN
  END SUBROUTINE UFSCTD

  SUBROUTINE UFSCF(X,Y,Z,ISEL,NATOM,CENTERS,CTD,CST,FCT,MIN,MAX)
    !     UFSCF is a min/max routine.
    !     Min is stored in CENTERS(CST), Max is stored in CENTERS(FCT)
    !     MIN/MAX are logical variables that enable/disable min/max calculation
    real(chm_real) X(*),Y(*),Z(*)
    real(chm_real) CENTERS(4,3)
    !      real(chm_real) CTD(3),CST(3), FCT(3)
    real(chm_real) DISTANCE
    real(chm_real) BUFMIN, BUFMAX
    INTEGER ISEL(*)
    INTEGER NATOM
    INTEGER I
    INTEGER CTD,CST,FCT
    LOGICAL MIN, MAX
    !     CTD - Comparison
    !     CST - Closest
    !     FST - Farthest
    !     THE FOLLOWING TWO LINES ARE LIMITATIONS AND SHOULD BE
    !     CONSIDERED AS BUGGY :) - The coder
    BUFMAX=0
    BUFMIN=999999
    DO I=1,NATOM
       IF(ISEL(I) == 1) THEN
          DISTANCE=(CENTERS(CTD,1)-X(I))**2 + (CENTERS(CTD,2)-Y(I))**2  &
               + (CENTERS(CTD,3)-Z(I))**2
          DISTANCE=SQRT(DISTANCE)
          IF(DISTANCE <= BUFMIN.AND.MIN) THEN
             CENTERS(CST,1)=X(I)
             CENTERS(CST,2)=Y(I)
             CENTERS(CST,3)=Z(I)
             BUFMIN=DISTANCE
          ENDIF
          IF(DISTANCE >= BUFMAX.AND.MAX) THEN
             CENTERS(FCT,1)=X(I)
             CENTERS(FCT,2)=Y(I)
             CENTERS(FCT,3)=Z(I)
             BUFMAX=DISTANCE
          ENDIF
       ENDIF
    ENDDO
    RETURN
  END SUBROUTINE UFSCF

  SUBROUTINE UFSATOMAVE(X,Y,Z,ISEL,CENTERS,CID,AVDISTANCE,NATOM)
    !     Computes average distance to CENTERS(CID)
    real(chm_real) X(*),Y(*),Z(*)
    real(chm_real) CENTERS(4,3)
    real(chm_real) AVDISTANCE
    INTEGER ISEL(*)
    INTEGER I, NATOM, NTOT
    INTEGER CID
    AVDISTANCE=0.0d0
    NTOT=0
    DO I=1,NATOM
       IF(ISEL(I) == 1) THEN
          AVDISTANCE=AVDISTANCE+SQRT((X(I)-CENTERS(CID,1))**2  &
               + (Y(I)-CENTERS(CID,2))**2 + (Z(I)-CENTERS(CID,3))**2 )
          NTOT=NTOT+1
       ENDIF
    ENDDO
    AVDISTANCE=AVDISTANCE/NTOT
    RETURN
  END SUBROUTINE UFSATOMAVE

  SUBROUTINE UFSATOMDIST(X,Y,Z,ISEL,CENTERS,CID,AVDISTANCE,VARIANCE, &
       SKEWNESS, NATOM)
    !     Computes Variance and skewness for distribution from CENTERS(CID)
    real(chm_real) X(*),Y(*),Z(*)
    real(chm_real) CENTERS(4,3)
    real(chm_real) AVDISTANCE
    real(chm_real) TDISTANCE
    real(chm_real) TMP
    real(chm_real) VARIANCE
    real(chm_real) SKEWNESS
    INTEGER I,NATOM,NTOT
    INTEGER ISEL(*)
    INTEGER CID
    VARIANCE=0.0d0
    SKEWNESS=0.0d0
    NTOT=0
    DO I=1,NATOM
       IF(ISEL(I) == 1) THEN
          TDISTANCE=SQRT((X(I)-CENTERS(CID,1))**2  &
               + (Y(I)-CENTERS(CID,2))**2 + (Z(I)-CENTERS(CID,3))**2)
          TMP=TDISTANCE-AVDISTANCE
          TDISTANCE=TMP**2
          VARIANCE=VARIANCE+TDISTANCE
          TDISTANCE=TDISTANCE*TMP
          SKEWNESS=SKEWNESS+TDISTANCE
          NTOT=NTOT+1
       ENDIF
    ENDDO
    VARIANCE=SQRT(VARIANCE/(NTOT-1))
    SKEWNESS=(SKEWNESS/((NTOT-1)*VARIANCE**3))**(1./3.)
    RETURN
  END SUBROUTINE UFSATOMDIST

  SUBROUTINE UFSMANHATTAN(MOLDESI, MOLDESP,DIMENS,DISTANCE)
    real(chm_real) MOLDESI(*),MOLDESP(*)
    real(chm_real) DISTANCE
    INTEGER I,DIMENS
    DISTANCE=0.0d0
    DO I=1,DIMENS
       DISTANCE=DISTANCE+ABS(MOLDESI(I)-MOLDESP(I))
    ENDDO
    DISTANCE=DISTANCE/DIMENS
    RETURN
  END SUBROUTINE UFSMANHATTAN

  !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
  !
  !   Misc Algorithms.
  !   Different authors, if not specified by: Juan R. Perilla
  !
  !     Most are used in the combinatorial version of DIMS-nm
  !
  !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

  SUBROUTINE DIMSNXCBN(N, M, IC)
    !      ALGORITHM 452, COLLECTED ALGORITHMS FROM ACM.
    !      THIS WORK PUBLISHED IN COMMUNICATIONS OF THE ACM
    !      VOL. 16, NO. 8, August, 1973, P.485.
    INTEGER M, N, N1, J, NJ, J1, K1, K2, JP, I, I1
    !  EXPLANATION OF THE PARAMETERS IN THE CALLING SEQUENCE:
    !    N, THE TOTAL NUMBER OF OBJECTS.
    !    M, THE NUMBER OF OBJECTS TO BE TAKEN FROM N.
    !      IF M = 0, OR M>=N, EXIT WITH ARGUMENTS UNCHANGED.
    !    IC, AN INTEGER ARRAY.  IC CONTAINS AN N-DIMNSIONAL
    !      BINARY VECTOR WITH M ELEMENTS SET TO 1, REPRESENTING
    !      THE N OBJECTS IN A COMBINATION.
    !  THIS ALGORITHM IS PROGRAMMED IN ANSI STANDARD FORTRAN.
    INTEGER IC(N)
    !  CHECK ENDING PATTERN OF VECTOR.
    IF (M >= N .OR. M == 0 ) GO TO 140
    N1 = N - 1
    DO 10 J=1,N1
       NJ = N - J
       IF (IC(N) == IC(NJ)) GO TO 10
       J1 = J
       GO TO 20
10  CONTINUE
20  IF (MOD(M,2) == 1) GO TO 90
    !  FOR M EVEN.
    IF ( IC(N) == 1 ) GO TO 30
    K1 = N - J1
    K2 = K1 + 1
    GO TO 130
30  IF (MOD(J1,2) == 1) GO TO 40
    GO TO 120
    !  SCAN FROM RIGHT TO LEFT.
40  JP = ( N - J1 ) - 1
    DO 50 I=1,JP
       I1 = JP + 2 - I
       IF(IC(I1) == 0) GO TO 50
       IF(IC(I1-1) == 1) GO TO 70
       GO TO 80
50  CONTINUE
60  K1 = 1
    K2 = ( N + 1 ) - M
    GO TO 130
70  K1 = I1 - 1
    K2 = N - J1
    GO TO 130
80  K1 = I1 - 1
    K2 = ( N + 1 ) - J1
    GO TO 130
    !  FOR M ODD.
90  IF (IC(N) == 1) GO TO 110
    K2 = ( N - J1 ) - 1
    IF ( K2  ==  0 ) GO TO 60
    IF ( IC(K2+1)  ==  1 .AND. IC(K2)  ==  1 ) GO TO 100
    K1 = K2 + 1
    GO TO 130
100 K1 = N
    GO TO 130
110 IF ( MOD(J1,2)  ==  1 ) GO TO 120
    GO TO 40
120 K1 = N - J1
    K2 = MIN0 ( ( K1 + 2 ), N )
    !  COMPLEMENTING TWO BITS TO OBTAIN THE NEXT COMBINATION.
130 IC(K1) = 1 - IC(K1)
    IC(K2) = 1 - IC(K2)
140 RETURN
  END subroutine DIMSNXCBN

  SUBROUTINE DIMSCOMPXL(SL,X,IFROM,ITO,POS)
    !     SortedList->SL, Data->X
    real(chm_real)  SL(*),X
    INTEGER FROM,TO
    INTEGER MIDPOINT
    INTEGER IFROM,ITO
    INTEGER POS
    FROM=IFROM
    TO=ITO
10  IF(TO-FROM == 1) THEN
       IF(SL(FROM) > X) THEN
          POS=FROM
          RETURN
       ELSE
          POS=TO
          RETURN
       ENDIF
    ENDIF
    MIDPOINT=(FROM+TO)*0.5
    IF(SL(MIDPOINT) < X) THEN
       !     Recursive call :p
       !     CALL DIMSCOMPXL(SL,X,MIDPOINT,TO,POS)
       FROM=MIDPOINT
       GOTO 10
    ELSEIF(SL(MIDPOINT) > X) THEN
       !     Another recursive call aka GOTO :)
       !     CALL DIMSCOMPXL(SL,X,FROM,MIDPOINT,POS)
       TO=MIDPOINT
       GOTO 10
    ELSE
       POS=MIDPOINT
       RETURN
    ENDIF
    RETURN
  END SUBROUTINE DIMSCOMPXL

  SUBROUTINE DIMSINSERTRF(SL,X,POS,TOP)
    real(chm_real) SL(*)
    real(chm_real) X
    INTEGER POS,I,TOP
    DO I=0,TOP-(POS+1)
       SL(TOP-I)=SL(TOP-(I+1))
    ENDDO
    SL(POS)=X
    RETURN
  END SUBROUTINE DIMSINSERTRF

  SUBROUTINE DIMSINSERTRI(SL,X,POS,TOP)
    INTEGER SL(*)
    INTEGER X
    INTEGER POS,I,TOP
    DO I=0,TOP-(POS+1)
       SL(TOP-I)=SL(TOP-(I+1))
    ENDDO
    SL(POS)=X
    RETURN
  END SUBROUTINE DIMSINSERTRI


  !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
  !
  !
  !     Modified sort functions
  !
  !     Integer - Real support
  !
  !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

!----------------------------------------------------------------------

  INTEGER FUNCTION DIMSCOMBINATORIAL(N,L)
    INTEGER I,L,FACT,N
    FACT=1
    DO I=1,L
       FACT=FACT*(N-I)/(I+1)
    ENDDO
    DIMSCOMBINATORIAL=FACT
    RETURN
  END FUNCTION DIMSCOMBINATORIAL

  SUBROUTINE BUILDSCALE(NBIAS,ASCALES,DSCALE)
  use stream
  use ctitla
    INTEGER I,NBIAS,IMAX,IMIN
    real(chm_real) ASCALES(*),DSCALE,ARG
    IMAX=NBIAS*0.333
    IMIN=IMAX
    DO I=1,IMAX
       ARG=(24.0/NBIAS)*I-4.0
       ASCALES(I)=DSCALE*0.5*(1.0+TANH(ARG))
       ASCALES(NBIAS-I)=ASCALES(I)
       IF(PRNLEV >= 4) WRITE(OUTU,'(A26,I4,F10.3)') &
            'BUILDSCALE> DIMS Scaling: ', I, ASCALES(I)
       IF(PRNLEV >= 4) WRITE(OUTU,'(A26,I4,F10.3)')  &
            'BUILDSCALE> DIMS Scaling: ', &
            NBIAS-I, ASCALES(NBIAS-I)
    ENDDO
    IMAX=NBIAS-IMIN
    DO I=IMIN,IMAX
       ASCALES(I)=DSCALE
       IF(PRNLEV >= 4) WRITE(OUTU,'(A26,I4,F10.3)') &
            'BUILDSCALE> DIMS Scaling: ', I,ASCALES(I)
    ENDDO
    ASCALES(NBIAS)=ASCALES(1)
    RETURN
  END SUBROUTINE BUILDSCALE


  SUBROUTINE DIMSNMBIAS(XS,YS,ZS,DDV,DIMSIDX,NIDIMS,DIMACTIVE,NMODE &
       ,NORM,DSCALE)
  use consta
  use number
  use psf
  use stream
    real(chm_real) XS(*),YS(*),ZS(*)
    real(chm_real) DDV(NATOM*3,*)
    real(chm_real) DTX,DTY,DTZ,DTNORM,DSCALE
    INTEGER DIMSIDX(*),DIMACTIVE(MAXAIM,3)
    INTEGER NIDIMS,NMODE
    INTEGER J,K,MODE
    LOGICAL NORM
    !     -- Bias DIMS atoms
    DO K=1,NIDIMS
       J=DIMSIDX(K)
       MODE=3*(J-1)+1
       !         IF(DIMACTIVE(J,1) >= 3) THEN
       !     -- If inactive, don't bias
       !            DTX=ZERO
       !            DTY=ZERO
       !            DTZ=ZERO
       !         ELSE
       !     -- Otherwise bias with the information from NM
       DTX=DDV(MODE,NMODE)
       DTY=DDV(MODE+1,NMODE)
       DTZ=DDV(MODE+2,NMODE)
       !     -- Normalize normal modes
       IF(NORM) THEN
          DTNORM=DTX**2+DTY**2+DTZ**2
          IF(DTNORM /= 0) THEN
             DTNORM=DSCALE/SQRT(DTNORM)
             DTX=DTX*DTNORM
             DTY=DTY*DTNORM
             DTZ=DTZ*DTNORM
          ELSE
             IF(PRNLEV >= 5) THEN
                WRITE(OUTU,'(A22,I4,I4)')  &
                     'DIMSNM> ZERO-NORM NM: ',NMODE,J
             ENDIF
             DTX=ZERO
             DTY=ZERO
             DTZ=ZERO
          ENDIF
       ENDIF
       !         ENDIF
       !     --- Tentative move
       XS(J)=XS(J)+DTX
       YS(J)=YS(J)+DTY
       ZS(J)=ZS(J)+DTZ
    ENDDO
    RETURN
  END SUBROUTINE DIMSNMBIAS

  SUBROUTINE DIMSDLNGV(GAMMA,IG, &
       XCOMP,YCOMP,ZCOMP, XOLD, YOLD, ZOLD, &
       XTARG,YTARG,ZTARG, &
       XSCRATCH,YSCRATCH,ZSCRATCH, &
       DXDIMS,DYDIMS,DZDIMS, &
       !    &          DX,DY,DZ,
       XNEW,YNEW,ZNEW, &
       WMAIN,NATOM2,NATOM3,DWIDTH,QPRINT, &
       ATFRST,ATLAST,IDIMS)
    !
    ! This function routine generates 3*NATOM Gaussian
    ! random deviates of 0.0 mean and standard deviation RF.
    ! The algorithm from Box and Muller.
    !
    ! Vanilla by Bernard R. Brooks   January, 1988
    !
    !     DIMS Soft Ratcheting version by Juan R. Perilla 2007
    !
    !     Note: Code hasn't been optimized
    !
  use number
  use consta
  use deriv
  use psf
  use energym
  use fourdm
  use rndnum
  use clcg_mod, only: random
#if KEY_PARALLEL==1
  use parallel 
#endif
  use stream
    !
    real(chm_real) GAMMA(*)
    real(chm_real) XOLD(*),YOLD(*),ZOLD(*)
    real(chm_real) XCOMP(*),YCOMP(*),ZCOMP(*),XNEW(*),YNEW(*),ZNEW(*)
    real(chm_real) XScratch(*),YScratch(*),ZScratch(*)
    real(chm_real) XTarg(*),YTarg(*),ZTarg(*)
    real(chm_real) DXDIMS(*),DYDIMS(*),DZDIMS(*)
    real(chm_real) WMAIN(*)
    real(chm_real) DWIDTH
    INTEGER IDIMS(*)
    INTEGER ATFRST,ATLAST
    INTEGER IG, NATOM2, NATOM3
    LOGICAL QPRINT
    !
    real(chm_real)  A,B,PIS,RDX,RDY,RDZ,RD4
    real(chm_real)  FACT,ALPHA
    real(chm_real)  SCORE0,SCORE
    INTEGER I,K
    LOGICAL ACCEPT
    !
    !
    CALL DIMSProgressSCORE(XCOMP,YCOMP,ZCOMP, &
         XTarg, YTarg,ZTarg, &
         ATFRST,ATLAST,NATOM, &
         WMAIN,AMASS,SCORE0)
    PIS=PI
    K=0
    if(.not.qoldrng) then
       IG=1
#if KEY_PARALLEL==1
       IG=MYNODP          
#endif
    endif
    !     First update non-fixed-non-dims atoms
    DO I=ATFRST,ATLAST
#if KEY_PARASCAL==1
       IF(JPBLOCK(I) == MYNOD) THEN 
#endif
          IF(IMOVE(I) == 0.AND.IDIMS(I) == 0) THEN
             IF(K == 0) THEN
                A=GAMMA(I)*SQRT(MINTWO*LOG(RANDOM(IG)))
                B=TWO*PIS*RANDOM(IG)
                K=1
                RDX=A*COS(B)
                RDY=A*SIN(B)
                DX(I)=DX(I)+RDX
                DY(I)=DY(I)+RDY
                A=SQRT(MINTWO*LOG(RANDOM(IG)))
                B=TWO*PIS*RANDOM(IG)
                RDZ=GAMMA(I)*A*COS(B)
                DZ(I)=DZ(I)+RDZ
#if KEY_FOURD==1
                IF(DIM4) THEN
                   K=0
                   RD4=GAMMA(I)*A*SIN(B)
                   DFDIM(I)=DFDIM(I)+RD4
                ENDIF
#endif 
             ELSE
                K=0
                RDX=GAMMA(I)*A*SIN(B)
                DX(I)=DX(I)+RDX
                A=GAMMA(I)*SQRT(MINTWO*LOG(RANDOM(IG)))
                B=TWO*PIS*RANDOM(IG)
                RDY=A*COS(B)
                RDZ=A*SIN(B)
                DY(I)=DY(I)+RDY
                DZ(I)=DZ(I)+RDZ
             ENDIF
          ENDIF
#if KEY_PARASCAL==1
       ENDIF 
#endif
    ENDDO
    DO I=ATFRST,ATLAST
#if KEY_PARASCAL==1
       IF(JPBLOCK(I) == MYNOD) THEN 
#endif
          IF(IMOVE(I) == 0.AND.IDIMS(I) == 0) THEN
             FACT=GAMMA(I+NATOM)
             ALPHA=GAMMA(I+NATOM2)
             XNEW(I)=ALPHA*XOLD(I)-DXDIMS(I)*FACT
             YNEW(I)=ALPHA*YOLD(I)-DYDIMS(I)*FACT
             ZNEW(I)=ALPHA*ZOLD(I)-DZDIMS(I)*FACT
             XSCRATCH(I)=XCOMP(I)+XNEW(I)
             YSCRATCH(I)=YCOMP(I)+YNEW(I)
             ZSCRATCH(I)=ZCOMP(I)+ZNEW(I)
          ENDIF
#if KEY_PARASCAL==1
       ENDIF                  
#endif
    ENDDO
1013 DO I=ATFRST,ATLAST
#if KEY_PARASCAL==1
       IF(JPBLOCK(I) == MYNOD) THEN 
#endif
          IF(IMOVE(I) == 0.AND.IDIMS(I) == 1) THEN
             IF(K == 0) THEN
                A=GAMMA(I)*SQRT(MINTWO*LOG(RANDOM(IG)))
                B=TWO*PIS*RANDOM(IG)
                K=1
                RDX=A*COS(B)
                RDY=A*SIN(B)
                DXDIMS(I)=DX(I)+RDX
                DYDIMS(I)=DY(I)+RDY
                A=SQRT(MINTWO*LOG(RANDOM(IG)))
                B=TWO*PIS*RANDOM(IG)
                RDZ=GAMMA(I)*A*COS(B)
                DZDIMS(I)=DZ(I)+RDZ
#if KEY_FOURD==1
                IF(DIM4) THEN
                   K=0
                   RD4=GAMMA(I)*A*SIN(B)
                   DFDIM(I)=DFDIM(I)+RD4
                ENDIF
#endif 
             ELSE
                K=0
                RDX=GAMMA(I)*A*SIN(B)
                DXDIMS(I)=DX(I)+RDX
                A=GAMMA(I)*SQRT(MINTWO*LOG(RANDOM(IG)))
                B=TWO*PIS*RANDOM(IG)
                RDY=A*COS(B)
                RDZ=A*SIN(B)
                DYDIMS(I)=DY(I)+RDY
                DZDIMS(I)=DZ(I)+RDZ
             ENDIF
          ENDIF
#if KEY_PARASCAL==1
       ENDIF 
#endif
    ENDDO
    !
    !  Integrate ...
    !  We should skip later update of atoms positions, but given that we want to
    !  be able to add other algorithms as well. We limit to change the forces as
    !  in the original LANG implementation.
    !
    DO I=ATFRST,ATLAST
#if KEY_PARASCAL==1
       IF(JPBLOCK(I) == MYNOD) THEN 
#endif
          IF(IMOVE(I) == 0.AND.IDIMS(I) == 1) THEN
             FACT=GAMMA(I+NATOM)
             ALPHA=GAMMA(I+NATOM2)
             XNEW(I)=ALPHA*XOLD(I)-DXDIMS(I)*FACT
             YNEW(I)=ALPHA*YOLD(I)-DYDIMS(I)*FACT
             ZNEW(I)=ALPHA*ZOLD(I)-DZDIMS(I)*FACT
             XSCRATCH(I)=XCOMP(I)+XNEW(I)
             YSCRATCH(I)=YCOMP(I)+YNEW(I)
             ZSCRATCH(I)=ZCOMP(I)+ZNEW(I)
          ENDIF
#if KEY_PARASCAL==1
       ENDIF                  
#endif
    ENDDO
    CALL DIMSProgressSCORE(XScratch,YScratch,ZScratch, &
         XTarg, YTarg,ZTarg, &
         ATFRST,ATLAST,NATOM, &
         WMAIN,AMASS,SCORE)
#if KEY_PARALLEL==1
    IF(MYNOD == 0) THEN 
#endif
       IF(SCORE <= SCORE0) THEN
          ACCEPT=.TRUE.
       ELSE
          ACCEPT=(EXP(-((SCORE0-SCORE)/DWIDTH)**2)) > RANDOM(IG)
       ENDIF
#if KEY_PARALLEL==1
    ENDIF   
#endif
#if KEY_PARALLEL==1
    CALL PSND4(ACCEPT,1)
#endif 
    IF(.NOT.ACCEPT) THEN
#if KEY_PARALLEL==1
       IF(MYNOD == 0) THEN 
#endif
          IF(QPRINT) THEN
             WRITE(OUTU,'(A16F10.4F10.4)') &
                  'DIMS> REJECTED: ', SCORE0, SCORE
          ELSE IF(PRNLEV >= 4) THEN
             WRITE(OUTU,'(A16F10.4F10.4)') &
                  'DIMS> REJECTED: ', SCORE0, SCORE
          ENDIF
#if KEY_PARALLEL==1
       ENDIF 
#endif
       GOTO 1013
    ELSE
#if KEY_PARALLEL==1
       IF(MYNOD == 0) THEN 
#endif
          IF(QPRINT) THEN
             WRITE(OUTU,'(A16F10.4F10.4)') &
                  'DIMS> ACCEPTED: ', SCORE0, SCORE
          ELSE IF(PRNLEV >= 4) THEN
             WRITE(OUTU,'(A16F10.4F10.4)') &
                  'DIMS> ACCEPTED: ', SCORE0, SCORE
          ENDIF
#if KEY_PARALLEL==1
       ENDIF 
#endif
       DO I=ATFRST,ATLAST
#if KEY_PARASCAL==1
          IF(JPBLOCK(I) == MYNOD) THEN 
#endif
             IF(IMOVE(I) == 0.AND.IDIMS(I) == 1) THEN
                DX(I)=DXDIMS(I)
                DY(I)=DYDIMS(I)
                DZ(I)=DZDIMS(I)
             ENDIF
#if KEY_PARASCAL==1
          ENDIF 
#endif
       ENDDO
    ENDIF
    RETURN
  END SUBROUTINE DIMSDLNGV

  SUBROUTINE DIMSDLNGV_VV2(GAMMA,IG, &
       XCOMP,YCOMP,ZCOMP, XOLD, YOLD, ZOLD, &
       XTARG,YTARG,ZTARG, &
       XSCRATCH,YSCRATCH,ZSCRATCH, &
       DXDIMS,DYDIMS,DZDIMS, &
       !    &          DX,DY,DZ,
       XNEW,YNEW,ZNEW, &
       DELTA_, VX,VY,VZ, & 
       WMAIN,NATOM2,NATOM3,DWIDTH,QPRINT, &
       ATFRST,ATLAST,IDIMS)
    !
    ! This function routine generates 3*NATOM Gaussian
    ! random deviates of 0.0 mean and standard deviation RF.
    ! The algorithm from Box and Muller.
    !
    ! Vanilla by Bernard R. Brooks   January, 1988
    !
    !     DIMS Soft Ratcheting version by Juan R. Perilla 2007
    !     DIMS Soft Ratcheting version with VV2 by Samarjeet, January 2016
    !
    !     Note: Code hasn't been optimized
    !
  use number
  use consta
  use deriv
  use psf
  use energym
  use fourdm
  use rndnum
  use clcg_mod, only: random
#if KEY_PARALLEL==1
  use parallel 
#endif
  use stream
    !
    real(chm_real) GAMMA(*)
    real(chm_real) XOLD(*),YOLD(*),ZOLD(*)
    real(chm_real) XCOMP(*),YCOMP(*),ZCOMP(*),XNEW(*),YNEW(*),ZNEW(*)
    real(chm_real) XScratch(*),YScratch(*),ZScratch(*)
    real(chm_real) XTarg(*),YTarg(*),ZTarg(*)
    real(chm_real) DXDIMS(*),DYDIMS(*),DZDIMS(*), VX(*),VY(*),VZ(*)
    real(chm_real) WMAIN(*)
    real(chm_real) DWIDTH, DELTA_
    INTEGER IDIMS(*)
    INTEGER ATFRST,ATLAST
    INTEGER IG, NATOM2, NATOM3
    LOGICAL QPRINT
    !
    real(chm_real)  A,B,PIS,RDX,RDY,RDZ,RD4
    real(chm_real)  FACT,ALPHA
    real(chm_real)  SCORE0,SCORE
    INTEGER I,K
    LOGICAL ACCEPT
    !
    !
    CALL DIMSProgressSCORE(XCOMP,YCOMP,ZCOMP, &
         XTarg, YTarg,ZTarg, &
         ATFRST,ATLAST,NATOM, &
         WMAIN,AMASS,SCORE0)
    PIS=PI
    K=0
    if(.not.qoldrng) then
       IG=1
#if KEY_PARALLEL==1
       IG=MYNODP          
#endif
    endif
    !     First update non-fixed-non-dims atoms
    DO I=ATFRST,ATLAST
#if KEY_PARASCAL==1
       IF(JPBLOCK(I) == MYNOD) THEN 
#endif
          IF(IMOVE(I) == 0.AND.IDIMS(I) == 0) THEN
             IF(K == 0) THEN
                A=GAMMA(I)*SQRT(MINTWO*LOG(RANDOM(IG)))
                B=TWO*PIS*RANDOM(IG)
                K=1
                RDX=A*COS(B)
                RDY=A*SIN(B)
                DX(I)=DX(I)+RDX
                DY(I)=DY(I)+RDY
                A=SQRT(MINTWO*LOG(RANDOM(IG)))
                B=TWO*PIS*RANDOM(IG)
                RDZ=GAMMA(I)*A*COS(B)
                DZ(I)=DZ(I)+RDZ
#if KEY_FOURD==1
                IF(DIM4) THEN
                   K=0
                   RD4=GAMMA(I)*A*SIN(B)
                   DFDIM(I)=DFDIM(I)+RD4
                ENDIF
#endif 
             ELSE
                K=0
                RDX=GAMMA(I)*A*SIN(B)
                DX(I)=DX(I)+RDX
                A=GAMMA(I)*SQRT(MINTWO*LOG(RANDOM(IG)))
                B=TWO*PIS*RANDOM(IG)
                RDY=A*COS(B)
                RDZ=A*SIN(B)
                DY(I)=DY(I)+RDY
                DZ(I)=DZ(I)+RDZ
             ENDIF
          ENDIF
#if KEY_PARASCAL==1
       ENDIF 
#endif
    ENDDO
    DO I=ATFRST,ATLAST
#if KEY_PARASCAL==1
       IF(JPBLOCK(I) == MYNOD) THEN 
#endif
          IF(IMOVE(I) == 0.AND.IDIMS(I) == 0) THEN
             FACT=GAMMA(I+NATOM)
             ALPHA=GAMMA(I+NATOM2)
             XNEW(I)=ALPHA*XOLD(I)-DXDIMS(I)*FACT
             YNEW(I)=ALPHA*YOLD(I)-DYDIMS(I)*FACT
             ZNEW(I)=ALPHA*ZOLD(I)-DZDIMS(I)*FACT
             XSCRATCH(I)=XCOMP(I)+XNEW(I)
             YSCRATCH(I)=YCOMP(I)+YNEW(I)
             ZSCRATCH(I)=ZCOMP(I)+ZNEW(I)
          ENDIF
#if KEY_PARASCAL==1
       ENDIF                  
#endif
    ENDDO
1023 DO I=ATFRST,ATLAST
#if KEY_PARASCAL==1
       IF(JPBLOCK(I) == MYNOD) THEN 
#endif
          IF(IMOVE(I) == 0.AND.IDIMS(I) == 1) THEN
             IF(K == 0) THEN
                A=GAMMA(I)*SQRT(MINTWO*LOG(RANDOM(IG)))
                B=TWO*PIS*RANDOM(IG)
                K=1
                RDX=A*COS(B)
                RDY=A*SIN(B)
                DXDIMS(I)=DX(I)+RDX
                DYDIMS(I)=DY(I)+RDY
                A=SQRT(MINTWO*LOG(RANDOM(IG)))
                B=TWO*PIS*RANDOM(IG)
                RDZ=GAMMA(I)*A*COS(B)
                DZDIMS(I)=DZ(I)+RDZ
#if KEY_FOURD==1
                IF(DIM4) THEN
                   K=0
                   RD4=GAMMA(I)*A*SIN(B)
                   DFDIM(I)=DFDIM(I)+RD4
                ENDIF
#endif 
             ELSE
                K=0
                RDX=GAMMA(I)*A*SIN(B)
                DXDIMS(I)=DX(I)+RDX
                A=GAMMA(I)*SQRT(MINTWO*LOG(RANDOM(IG)))
                B=TWO*PIS*RANDOM(IG)
                RDY=A*COS(B)
                RDZ=A*SIN(B)
                DYDIMS(I)=DY(I)+RDY
                DZDIMS(I)=DZ(I)+RDZ
             ENDIF
          ENDIF
#if KEY_PARASCAL==1
       ENDIF 
#endif
    ENDDO
    !
    !  Integrate ...
    !  We should skip later update of atoms positions, but given that we want to
    !  be able to add other algorithms as well. We limit to change the forces as
    !  in the original LANG implementation.
    !
    DO I=ATFRST,ATLAST
#if KEY_PARASCAL==1
       IF(JPBLOCK(I) == MYNOD) THEN 
#endif
          IF(IMOVE(I) == 0.AND.IDIMS(I) == 1) THEN
             !FACT=GAMMA(I+NATOM)
             !ALPHA=GAMMA(I+NATOM2)
             !XNEW(I)=ALPHA*XOLD(I)-DXDIMS(I)*FACT
             !YNEW(I)=ALPHA*YOLD(I)-DYDIMS(I)*FACT
             !ZNEW(I)=ALPHA*ZOLD(I)-DZDIMS(I)*FACT
             !XSCRATCH(I)=XCOMP(I)+XNEW(I)
             !YSCRATCH(I)=YCOMP(I)+YNEW(I)
             !ZSCRATCH(I)=ZCOMP(I)+ZNEW(I)
             FACT=DELTA_/AMASS(I)
             VXDIMS(I)=VX(I) - FACT*DXDIMS(I)
             VYDIMS(I)=VX(I) - FACT*DYDIMS(I)
             VZDIMS(I)=VX(I) - FACT*DZDIMS(I)

             XSCRATCH(I)=XCOMP(I)+DELTA_ * VXDIMS(I)
             YSCRATCH(I)=YCOMP(I)+DELTA_ * VYDIMS(I)
             ZSCRATCH(I)=ZCOMP(I)+DELTA_ * VZDIMS(I)
          ENDIF
#if KEY_PARASCAL==1
       ENDIF                  
#endif
    ENDDO
    CALL DIMSProgressSCORE(XScratch,YScratch,ZScratch, &
         XTarg, YTarg,ZTarg, &
         ATFRST,ATLAST,NATOM, &
         WMAIN,AMASS,SCORE)
#if KEY_PARALLEL==1
    IF(MYNOD == 0) THEN 
#endif
       IF(SCORE <= SCORE0) THEN
          ACCEPT=.TRUE.
       ELSE
          ACCEPT=(EXP(-((SCORE0-SCORE)/DWIDTH)**2)) > RANDOM(IG)
       ENDIF
#if KEY_PARALLEL==1
    ENDIF   
#endif
#if KEY_PARALLEL==1
    CALL PSND4(ACCEPT,1)
#endif 
    IF(.NOT.ACCEPT) THEN
#if KEY_PARALLEL==1
       IF(MYNOD == 0) THEN 
#endif
          IF(QPRINT) THEN
             WRITE(OUTU,'(A16F10.4F10.4)') &
                  'DIMS> REJECTED: ', SCORE0, SCORE
          ELSE IF(PRNLEV >= 4) THEN
             WRITE(OUTU,'(A16F10.4F10.4)') &
                  'DIMS> REJECTED: ', SCORE0, SCORE
          ENDIF
#if KEY_PARALLEL==1
       ENDIF 
#endif
       GOTO 1023
    ELSE
#if KEY_PARALLEL==1
       IF(MYNOD == 0) THEN 
#endif
          IF(QPRINT) THEN
             WRITE(OUTU,'(A16F10.4F10.4)') &
                  'DIMS> ACCEPTED: ', SCORE0, SCORE
          ELSE IF(PRNLEV >= 4) THEN
             WRITE(OUTU,'(A16F10.4F10.4)') &
                  'DIMS> ACCEPTED: ', SCORE0, SCORE
          ENDIF
#if KEY_PARALLEL==1
       ENDIF 
#endif
       DO I=ATFRST,ATLAST
#if KEY_PARASCAL==1
          IF(JPBLOCK(I) == MYNOD) THEN 
#endif
             IF(IMOVE(I) == 0.AND.IDIMS(I) == 1) THEN
                DX(I)=DXDIMS(I)
                DY(I)=DYDIMS(I)
                DZ(I)=DZDIMS(I)
             ENDIF
#if KEY_PARASCAL==1
          ENDIF 
#endif
       ENDDO
    ENDIF
    RETURN
  END SUBROUTINE DIMSDLNGV_VV2

#else /* (if_dims)*/

  SUBROUTINE DIMSINIT(COMLYN,COMLEN)
    !---------------------------------------------------------------------
    !
    !     This routine initialize DIMS
    !
    !---------------------------------------------------------------------
    !
    !
    character(len=*) COMLYN
    INTEGER COMLEN
    CALL WRNDIE(0,'<DIMS>','DIMS code not compiled')
    return
  end SUBROUTINE DIMSINIT
#endif /* (if_dims)*/

end module dims

