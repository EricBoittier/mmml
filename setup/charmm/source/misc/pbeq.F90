module pbeq
  !     This module contains the data structure for the 
  !     Poisson-Boltzmann equation (PBEQ)  
  ! ---------------------------------------------------------
  !
  !  Note: Every variable and array needs to be commented here
  !         to bring this code up to CHARMM standards.
  !
  ! ---------------------------------------------------------
  ! QPBEQ   logical flag 
  !
  ! LSTPRP 
  use chm_kinds
  use vector, only: addctvr4,snrmvecr4
  ! QC: 11/17 Add EPERT
#if KEY_PERT == 1
  use pert,only: qpert
#endif /* pert */
  ! QC: 11/17 Done
  implicit none
#if KEY_PBEQ==1 /*pbeq*/
  private
  !routines
  !public :: pbeq0,pbforce,gsbp0,srdist
  !QC: export more
  public :: pbeq0,pbforce,gsbp0,srdist,m3,dm3,rpowerl2,cosmphi2,sinmphi2,&
       alpol2,dalpol2,lpol2,dlpol2,pbeq1,oldpbeq1,mayer,pbeq2,qgas1, &
       pbeq_iniall,smbp0

  !variables
  public :: qpbeq,qpbf,npbeq,qgsbp,lgrats,xgrbeg,xgrend,ygrbeg,ygrend,zgrbeg,zgrend,qchrg,&
            qsmbp,numsurf,nclxog,nclyog,nclzog,dcelog,tranxog,tranyog,tranzog,&
            xbcenog,ybcenog,zbcenog,cgthresh,cgmaxit,qsmbp_qc_grad,qcchrg
  !,mayer,boundpot,rforce,mayer_cd_tril,mayer_bp_all, &
  !          mayer_bp_int,mayer_memb,mayer_econe,mayer_cyln,mayer_mstep,mayer_reen,pbeq5, &
  !          mayer_bp_focus
  ! variables
  !public :: qpbeq,qpbf,npbeq,qgsbp,srdist,qgtot,qgaa,qgbb,qgab,qrectbox, &
  !      tranx,rbxmin,xbcen,dcel,rbxmax, &
  !      trany,rbymin,ybcen,rbymax,tranz,rbzmin,zbcen,rbzmax,qsphere,qprolate,minor,major, &
  !      rrxcen,rrycen,rrzcen,qlbox,qfocus, &
  !      nclx,ncly,nclz,qa,qb,ntprp,lstprp,pbrad,watr,ionr,may,mayx,mayy,mayz, &
  !      swin,iphip,tmemb,zmemb,nimgb,htmemb,hdens,posalp,negalp,qchrg,poffst,noffst, &
  !      droplet,xdroplet,ydroplet,zdroplet,qdtom,qdkap,lxmax,lymax,lzmax,lxmin,lymin,lzmin, &
  !      qbtom,qbkap,xcyln,ycyln,zcyln,rcyln,hcyln,qctom,qckap,xcone,ycone,zcone,ax,by,cz, &
  !      qectom,qeckap,ixmax,iymax,izmax,ixmin,iymin,izmin, &
  !      qkeep,qreen,qsmth,qbspl,qintbp,qzero,qpbc,qnonlinear,qpartlinear,qfkap,ipox,ipoy,ipoz, &
  !      mapt,epsp,maxits,dome,deps,ichc,epsw,kappa2,iphiw,vmemb,epsm,epsh,epsd,epsb,epsc,epsec, &
  !      qnpbc,qosor,rxnafx,rxnafy,rxnafz,egsbpa,iphix,ntra,lstra,ntrb,lstrb,&
  !      oldxnp,oldynp,oldznp,xnpol,ynpol,znpol,qmij,lstpx,lstpy,lstpz,lstpol,ntpol, &
  !      xscale,yscale,zscale,bnorm,cgscal,ncel3,mij,nmpol,lstpl,lstpm,oldnmp, &
  !      qmmij,tolsij,mmij,srcut,lgrats,xgrbeg,xgrend,ygrbeg,ygrend,zgrbeg,zgrend
  ! 
  !    QC: need to remove COEFX, only defined in SCC!
  !    DBFX,DBFY,DBFZ,IBFX,IBFY,IBFZ,NPFX,NPFY,NPFZ,RXNAFX,RXNAFY,RXNAFZ,MIJ,COEF,COEFX,&
  real(chm_real), allocatable, dimension (:) :: PBRAD,IPOX,IPOY,IPOZ,RXNFX,RXNFY,RXNFZ, &
       DBFX,DBFY,DBFZ,IBFX,IBFY,IBFZ,NPFX,NPFY,NPFZ,RXNAFX,RXNAFY,RXNAFZ,MIJ,COEF,&
       BNORM,MQ,MMIJ,RXNBFX,RXNBFY,RXNBFZ, &
       ISURFCRD,ISURFCHR ! JZ_UW12: SMBP

  real(chm_real4), allocatable, dimension (:) :: ICHC,IPHIW,MAY,MAYX,MAYY,MAYZ,IPHIP,IPHIX, &
       IPHIMMT,IPHIQMT,IPHIQMG  ! JZ_UW12: SMBP

  integer, allocatable, dimension (:) :: LSTPRP,LSTRA,LSTRB,LSTPOL,LSTPX,LSTPY,LSTPZ,&
       LSTPL,LSTPM, HPGDGR, HPGRAT !RJP heap pointers for phi average

  LOGICAL QPBEQ, QCHRG, QPBF, QKEEP, QREEN, &
       QCTOM, QCKAP, QDTOM, QDKAP, QBTOM, QBKAP, &
       QECTOM, QECKAP, &
       QBSPL, QINTBP, QZERO, QFOCUS, QOSOR, &
       QFMG, QSMTH, QOLDPB, QPBC, QNPBC, QLBOX, &
       QGSBP, QA, QB, QGTOT, QGAA, QGAB, QGBB,  &
       QMIJ, QPHIX, QFKAP, QNOSORT, &
       QRECTBOX, QSPHERE, QPROLATE, QMMIJ,  &
       QNONLINEAR,QPARTLINEAR,QUNDER, &
       QRFRECT,QRFSPHE, &
       LGRATS, & !RJP
       QSMBP,QSMBP_QC_GRAD,QREADPHIX,QSMBP_ALLOC_TOTP  ! JZ_UW12
       

  ! Integer
  INTEGER NCLX,   NCLY,   NCLZ,   NTPRP,  MAXITS, NPBEQ, &
       RNCLX,  RNCLY,  RNCLZ,  NIMGB, &
       NCEL3,  &
       MAPT,    &
       IXMAX,  IYMAX,  IZMAX,  IXMIN,  IYMIN,  IZMIN, &
       NCYC,   NPRE,   NPOST,  NGRIDpb, &
       NTPOL,  MAXNPOL,NTRA,   NTRB,  &
       XNPOL,  YNPOL,  ZNPOL,  OLDXNP, OLDYNP, OLDZNP, &
       NMPOL,  OLDNMP, & 
       NLIST, &
       NGRATS, GDMANY,  &  !RJP for grid-average atom selections
       SMBPIGU, NUMSURF, SPHEPTSALG, CGMAXIT, LNCELGL, & ! JZ_UW12
       QCCHRG, NCLXOG, NCLYOG, NCLZOG, NCEL3OG, SMBPMXIT !    

  ! Real
  real(chm_real) DCEL, TRANX, TRANY, TRANZ, XBCEN, YBCEN, ZBCEN, &
       RDCEL,RXBCEN,RYBCEN,RZBCEN, &
       EPSP, EPSW,  WATR,  IONR, &
       TEMP, CONC, KAPPA2, &
       EPSM, TMEMB, ZMEMB, VMEMB, HTMEMB, EPSH, &
       POSALP, NEGALP, HDENS, NOFFST, POFFST, &
       DROPLET, EPSD, XDROPLET, YDROPLET, ZDROPLET, &
       EPSB, LXMAX, LYMAX, LZMAX, LXMIN, LYMIN, LZMIN, &
       EPSC, XCYLN, YCYLN, ZCYLN, RCYLN, HCYLN, &
       EPSEC,XCONE,YCONE,ZCONE,AX,BY,CZ, &
       SWIN, &
       DOME, DEPS, EPB1, EPB2, STCO, &
       RRXCEN, RRYCEN, RRZCEN, &
       RBXMIN, RBXMAX, RBYMIN, RBYMAX, RBZMIN, RBZMAX, &
       XSCALE,  YSCALE,  ZSCALE, &
       EGSBPA, EGSBPB, CGSCAL, &
       SRDIST, MAJOR,  MINOR, &
       SRCUT, TOLSIJ, &
       XGRBEG,XGREND,YGRBEG,YGREND,ZGRBEG,ZGREND, & !RJP for grid av
       ESMBPA, ESMBPB, CGTHRESH, LTRANGL, LDCELGL, &                   ! JZ_UW12
       LXBCENGL, LYBCENGL, LZBCENGL, &                                 ! 
       DCELOG, XBCENOG, YBCENOG, ZBCENOG, TRANXOG, TRANYOG, TRANZOG, & !
       RINCX, RINCY, RINCZ, SMBPPTHR                                   !

  !  The following is related to QM/MM implementation of gsbp and pb 
  !  Xiao_QC_UW0609
  LOGICAL QPSC,QGAS1
#if KEY_SCCDFTB==1
  !
  !     COEFX        -----> heap index for COEFX(NTPOL)
  !        COEFX(NTPOL) -----> Q for MM atoms excluded QM/MM inter
  !
  real(chm_real), allocatable, dimension(:) ::  COEFX
  integer MXTPSC
  real(chm_real4),allocatable,dimension(:) ::IPHIWTB,IPHIPTB,IPHIWEX,IPHIPEX
  REAL(chm_real8) PSCTOL

#endif /*  SCCDFTB*/
#endif /* (pbeq)*/

contains

#if KEY_PBEQ==0 /*pbeq_main*/
  SUBROUTINE PBEQ0
    CALL WRNDIE(-1,'<CHARMM>','PBEQ code is not compiled.')
    return
  end SUBROUTINE PBEQ0
#else /* (pbeq_main)*/

  subroutine pbeq_iniall()
    qpbeq=.false.
    qchrg=.false.
    lgrats=.false.
    xgrbeg=0
    xgrend=0
    ygrbeg=0
    ygrend=0
    zgrbeg=0
    zgrend=0
    qsmbp = .false.
    qreadphix = .false.
    qsmbp_alloc_totp = .false.
    ncel3og = -1
    return
  end subroutine pbeq_iniall

  SUBROUTINE PBEQ0
    !-----------------------------------------------------------------------
    ! Poisson-Boltzmann Equation (PBEQ)
    !
    ! Authors:  Dmitrii Beglovd, Wonpil Im and Benoit Roux  (1998)
    !           Department of Chemistry, University of Montreal
    !
    !           Wonpil Im and Benoit Roux (2000)
    !           Department of Biochemistry and Structural Biology
    !           Weill Medical College of Cornell University
    !
    !     This subroutine is the main driver for all Poisson-Boltzmann
    !     calculations.
    !
    !     June, 2000: The code is fully revised.
    !                 GSBP added.
    !     June, 2002: running phi averages implemented --RJP
    !-----------------------------------------------------------------------
    !
    use chm_kinds
    use consta
    use dimens_fcm
    use number
    use chutil
    use comand
    use psf
    use select
    use stream
    use string
    use coord
    use corman_mod,only:corcom
    use deriv
    use timerm
    use memory
    use scalar_module, only: scalar
#if KEY_SCCDFTB==1
    use sccdftb
    use sccgsbp
    ! XIAO_QC_UW0609 fcm for scc/pb
    use sccpb
    use gamess_fcm, only: qmused_sccdftb
#endif 
    use machutil,only:wrttim
    use iapbs

    implicit none
#if KEY_SCCDFTB==1
    ! QC_UW04:
    real(chm_real) ECRAP
    ! QC_UW04:
#endif 


    !-----------------------------------------------------------------------
    !  Miscelaneous Local variables
    !  Local variable
    integer,allocatable,dimension(:) :: ISLCT,LSTPBI
    INTEGER       I,NTPBI,IMODE,NCEL
    CHARACTER(len=4) WRD
    LOGICAL       DONE,EOF,LUSED,OK,QPBINT
    real(chm_real)        EGSBP
    !
    IF(PRNLEV.GE.2) WRITE(OUTU,*)
    IF(PRNLEV.GE.2) WRITE(OUTU,100) &
         'Calculations with the Poisson-Boltzmann Equation '
    IF(PRNLEV.GE.2) WRITE(OUTU,*)
100 FORMAT(3X,A)

    !-----------------------------------------------------------------------
1000 CALL XTRANE(COMLYN,COMLEN,'PBEQ')

    OK    = .TRUE.
    LUSED = .FALSE.
    DONE  = .FALSE.
    EOF   = .FALSE.

    CALL RDCMND(COMLYN,MXCMSZ,COMLEN,ISTRM,EOF,.TRUE.,.TRUE., &
         '  PBEQ> ')

    IF(EOF)THEN
       CALL PPSTRM(OK)
       IF(.NOT.OK)  RETURN
    ENDIF
    CALL MISCOM(COMLYN,MXCMSZ,COMLEN,LUSED)
    IF(LUSED) GOTO 1000

    WRD  = '    '
    WRD=NEXTA4(COMLYN,COMLEN)
    IF(WRD.EQ.'    ') GOTO 1000

    !.......................................................................
    IF(WRD.EQ.'RESE')THEN
       !        -------------

       IF(QPBEQ)THEN
          IF(PRNLEV.GE.2) WRITE(OUTU,100)  &
               'PREVIOUS GRID SETUP HAS BEEN RESET '
          IF(PRNLEV.GE.2) WRITE(OUTU,100)
          call chmdealloc('pbeq.src','PBEQ0','ICHC',NCEL3,cr4=ICHC)
          call chmdealloc('pbeq.src','PBEQ0','IPHIW',NCEL3,cr4=IPHIW)
          call chmdealloc('pbeq.src','PBEQ0','MAY',NCEL3,cr4=MAY)
          call chmdealloc('pbeq.src','PBEQ0','MAYX',NCEL3,cr4=MAYX)
          call chmdealloc('pbeq.src','PBEQ0','MAYY',NCEL3,cr4=MAYY)
          call chmdealloc('pbeq.src','PBEQ0','MAYZ',NCEL3,cr4=MAYZ)
          call chmdealloc('pbeq.src','PBEQ0','LSTPRP',NATOM,intg=LSTPRP)
          call chmdealloc('pbeq.src','PBEQ0','PBRAD',NATOM,crl=PBRAD)
          call chmdealloc('pbeq.src','PBEQ0','IPOX',MAPT,crl=IPOX)
          call chmdealloc('pbeq.src','PBEQ0','IPOY',MAPT,crl=IPOY)
          call chmdealloc('pbeq.src','PBEQ0','IPOZ',MAPT,crl=IPOZ)
          QPBEQ    = .FALSE.
          QFKAP    = .FALSE.
       ELSE
          IF(PRNLEV.GE.2) WRITE(OUTU,100) 'Grid is not setup!'
       ENDIF

       IF(QPBF)THEN
          call chmdealloc('pbeq.src','PBEQ0','IPHIP',NCEL3,cr4=IPHIP)
          call chmdealloc('pbeq.src','PBEQ0','RXNFX',NATOM,crl=RXNFX)
          call chmdealloc('pbeq.src','PBEQ0','RXNFY',NATOM,crl=RXNFY)
          call chmdealloc('pbeq.src','PBEQ0','RXNFZ',NATOM,crl=RXNFZ)
          call chmdealloc('pbeq.src','PBEQ0','DBFX',NATOM,crl=DBFX)
          call chmdealloc('pbeq.src','PBEQ0','DBFY',NATOM,crl=DBFY)
          call chmdealloc('pbeq.src','PBEQ0','DBFZ',NATOM,crl=DBFZ)
          call chmdealloc('pbeq.src','PBEQ0','IBFX',NATOM,crl=IBFX)
          call chmdealloc('pbeq.src','PBEQ0','IBFY',NATOM,crl=IBFY)
          call chmdealloc('pbeq.src','PBEQ0','IBFZ',NATOM,crl=IBFZ)
          call chmdealloc('pbeq.src','PBEQ0','NPFX',NATOM,crl=NPFX)
          call chmdealloc('pbeq.src','PBEQ0','NPFY',NATOM,crl=NPFY)
          call chmdealloc('pbeq.src','PBEQ0','NPFZ',NATOM,crl=NPFZ)
          

#if KEY_SCCDFTB==1
          if (qmused_sccdftb) then ! XIAO_QC_UW0609
            call chmdealloc('pbeq.src','PBEQ0','IPHIWTB',NCEL3,cr4=IPHIWTB)
            ! for EPSW of mulligan charge
            call chmdealloc('pbeq.src','PBEQ0','IPHIPTB',NCEL3,cr4=IPHIPTB)
            ! for EPSP of mulligan charge
            call chmdealloc('pbeq.src','PBEQ0','IPHIWEX',NCEL3,cr4=IPHIWEX)
            ! for EPSW of excluded mm charge
            call chmdealloc('pbeq.src','PBEQ0','IPHIPEX',NCEL3,cr4=IPHIPEX)
            ! for EPSP of excluded mm charge
          end if
#endif 

          QPBF     = .FALSE.
       ENDIF

       IF(QGSBP) THEN
          call chmdealloc('pbeq.src','PBEQ0','IPHIP',NCEL3,cr4=IPHIP)
          IF(QGAB) call chmdealloc('pbeq.src','PBEQ0','IPHIX',NCEL3,cr4=IPHIX)
          call chmdealloc('pbeq.src','PBEQ0','LSTRA',NATOM,intg=LSTRA)
          call chmdealloc('pbeq.src','PBEQ0','LSTRB',NATOM,intg=LSTRB)
          call chmdealloc('pbeq.src','PBEQ0','RXNAFX',NATOM,crl=RXNAFX)
          call chmdealloc('pbeq.src','PBEQ0','RXNAFY',NATOM,crl=RXNAFY)
          call chmdealloc('pbeq.src','PBEQ0','RXNAFZ',NATOM,crl=RXNAFZ)
          call chmdealloc('pbeq.src','PBEQ0','RXNBFX',NATOM,crl=RXNBFX)
          call chmdealloc('pbeq.src','PBEQ0','RXNBFY',NATOM,crl=RXNBFY)
          call chmdealloc('pbeq.src','PBEQ0','RXNBFZ',NATOM,crl=RXNBFZ)
          call chmdealloc('pbeq.src','PBEQ0','MIJ',NTPOL*NTPOL,crl=MIJ)
          call chmdealloc('pbeq.src','PBEQ0','COEF',NTPOL,crl=COEF)
          call chmdealloc('pbeq.src','PBEQ0','BNORM',NTPOL,crl=BNORM)
          call chmdealloc('pbeq.src','PBEQ0','LSTPOL',NTPOL,intg=LSTPOL)

#if KEY_SCCDFTB==1
          if (qmused_sccdftb) then
            call chmdealloc('pbeq.src','PBEQ0','COEFX',NTPOL,crl=COEFX)
            call chmdealloc('pbeq.src','PBEQ0','MQ',NTPOL*NSCCRP,crl=MQ)
          else
            call chmdealloc('pbeq.src','PBEQ0','MQ',NTPOL,crl=MQ)
          end if
#else
          call chmdealloc('pbeq.src','PBEQ0','MQ',NTPOL,crl=MQ)
#endif 

          IF(QRECTBOX) THEN
             call chmdealloc('pbeq.src','PBEQ0','LSTPX',NTPOL,intg=LSTPX)
             call chmdealloc('pbeq.src','PBEQ0','LSTPY',NTPOL,intg=LSTPY)
             call chmdealloc('pbeq.src','PBEQ0','LSTPZ',NTPOL,intg=LSTPZ)
          ELSEIF(QSPHERE) THEN
             call chmdealloc('pbeq.src','PBEQ0','LSTPL',NTPOL,intg=LSTPL)
             call chmdealloc('pbeq.src','PBEQ0','LSTPM',NTPOL,intg=LSTPM)
          ELSEIF(QPROLATE) THEN
             call chmdealloc('pbeq.src','PBEQ0','LSTPL',NTPOL,intg=LSTPL)
             call chmdealloc('pbeq.src','PBEQ0','LSTPM',NTPOL,intg=LSTPM)
             call chmdealloc('pbeq.src','PBEQ0','MMIJ',NTPOL*NTPOL,crl=MMIJ)
          ELSEIF(QRFRECT) THEN
             call chmdealloc('pbeq.src','PBEQ0','LSTPX',NTPOL,intg=LSTPX)
             call chmdealloc('pbeq.src','PBEQ0','LSTPY',NTPOL,intg=LSTPY)
             call chmdealloc('pbeq.src','PBEQ0','LSTPZ',NTPOL,intg=LSTPZ)
             call chmdealloc('pbeq.src','PBEQ0','MMIJ',NTPOL*NTPOL,crl=MMIJ)
          ELSEIF(QRFSPHE) THEN
             call chmdealloc('pbeq.src','PBEQ0','LSTPL',NTPOL,intg=LSTPL)
             call chmdealloc('pbeq.src','PBEQ0','LSTPM',NTPOL,intg=LSTPM)
             call chmdealloc('pbeq.src','PBEQ0','MMIJ',NTPOL*NTPOL,crl=MMIJ)
          ENDIF
          EGSBPA=ZERO
          EGSBPB=ZERO
          QGSBP     = .FALSE.
          QMIJ      = .FALSE.
          QMMIJ     = .FALSE.
       ENDIF
        
!     SMBP (JZ_UW12)
      IF(QSMBP) THEN
          call chmdealloc('pbeq.src','PBEQ0','IPHIP',NCEL3,cr4=IPHIP)
          IF(QGAB) call chmdealloc('pbeq.src','PBEQ0','IPHIX',NCEL3,cr4=IPHIX)
          call chmdealloc('pbeq.src','PBEQ0','LSTRA',NATOM,intg=LSTRA)
          call chmdealloc('pbeq.src','PBEQ0','LSTRB',NATOM,intg=LSTRB)
          call chmdealloc('pbeq.src','PBEQ0','RXNAFX',NATOM,crl=RXNAFX)
          call chmdealloc('pbeq.src','PBEQ0','RXNAFY',NATOM,crl=RXNAFY)
          call chmdealloc('pbeq.src','PBEQ0','RXNAFZ',NATOM,crl=RXNAFZ)
          call chmdealloc('pbeq.src','PBEQ0','RXNBFX',NATOM,crl=RXNBFX)
          call chmdealloc('pbeq.src','PBEQ0','RXNBFY',NATOM,crl=RXNBFY)
          call chmdealloc('pbeq.src','PBEQ0','RXNBFZ',NATOM,crl=RXNBFZ)
          IF(QSMBP_ALLOC_TOTP) THEN
            call chmdealloc('pbeq.src','PBEQ0','IPHIMMT',NCEL3OG,cr4=IPHIMMT)
            call chmdealloc('pbeq.src','PBEQ0','IPHIQMT',NCEL3OG,cr4=IPHIQMT)
            call chmdealloc('pbeq.src','PBEQ0','IPHIQMG',NCEL3OG,cr4=IPHIQMG)
         ENDIF
         ESMBPA=ZERO
         ESMBPB=ZERO
         QSMBP=.FALSE.
         QSMBP_ALLOC_TOTP=.FALSE.
      ENDIF

#if KEY_APBS==1
    ELSEIF(WRD.EQ.'APBS')THEN
       WRITE(OUTU,100)
       WRITE(OUTU,100) 'PBEQ> Entering APBS module'
       call apbs(pbrad)
#endif 
       !.......................................................................
    ELSEIF(WRD.EQ.'SOLV')THEN
       !            -------------
       IF (TIMER.GT.1) &
            CALL WRTTIM('PBEQ solver times:')
       CALL PREP1
       IF (TIMER.GT.1 .AND. .NOT.QPBF) &
            CALL WRTTIM('Grid parameters preparation times:')
       !
       IF(QPBF)THEN
          !        CALL PBFORCE(NATOM,X,Y,Z,CG,EPB1,EPB2,DX,DY,DZ,NPBEQ,0,.true.)
          !    XIAO_QC_UW0609: ADD potential SCC-DFTB iteration here
          CALL PBFORCE(NATOM,X,Y,Z,CG,EPB1,EPB2,DX,DY,DZ,NPBEQ,0,.true. &
#if KEY_SCCDFTB==1
               ,.true.,ECRAP,.true.  &           
#endif
               )
       ELSEIF(QFMG) THEN
          IF(MAXITS.GT.0)THEN
             IF(PRNLEV.GE.2) WRITE(OUTU,100)
             IF(PRNLEV.GE.2) WRITE(OUTU,100)  &
                  'Begin full multigrid algorithm'
             CALL PBEQ2(NATOM,X,Y,Z,CG,.false.,.true.)
             !                                      QENVV  QPRIN
             IF (TIMER.GT.1) &
                  CALL WRTTIM('FMG iteration times:')
          ELSE
             IF(PRNLEV.GE.2) WRITE(OUTU,100) 'No iterations'
          ENDIF
          call DECOMP(NTPRP,LSTPRP,X,Y,Z,CG,WMAIN,NCLX,NCLY,NCLZ,DCEL, &
               IPHIW,TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN,HALF,QBSPL)
       ELSE
          IF(MAXITS.GT.0)THEN
             IF(PRNLEV.GE.2) WRITE(OUTU,100)
             IF(PRNLEV.GE.2) WRITE(OUTU,100) 'Begin iterative solution'
             IF(QOLDPB)THEN
                call OLDPBEQ1(MAXITS,DOME,DEPS,NCLX,NCLY,NCLZ,IPHIW,MAYX, &
                     MAYY,MAYZ,ICHC,MAY,TMEMB,QOSOR,QFOCUS,QPBC,QNPBC,.true.)
             ELSEIF(QNONLINEAR) THEN
                call PBEQ3(MAXITS,DOME,DEPS,TEMP,KAPPA2,NCLX,NCLY,NCLZ,DCEL,VMEMB, &
                     TMEMB,ZMEMB,TRANZ,ZBCEN,IPHIW,MAYX,MAYY,MAYZ,ICHC, &
                     MAY,QPBC,QNPBC,QUNDER)
             ELSEIF(QPARTLINEAR) THEN
                call PBEQ4(MAXITS,DOME,DEPS,TEMP,KAPPA2,NCLX,NCLY,NCLZ,DCEL,VMEMB, &
                     TMEMB,ZMEMB,TRANZ,ZBCEN,IPHIW,MAYX,MAYY,MAYZ,ICHC, &
                     MAY,QPBC,QNPBC,QUNDER)
             ELSE
                call PBEQ1(MAXITS,DOME,DEPS,EPSW,EPSP,EPSM,KAPPA2,NCLX,NCLY,NCLZ, &
                     IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN,IPHIW,MAYX,MAYY, &
                     MAYZ,ICHC,MAY,TMEMB,QOSOR,QFOCUS,QPBC,QNPBC,.true.)
             ENDIF
             IF (TIMER.GT.1) &
                  CALL WRTTIM('SOR iteration times:')
          ELSE
             IF(PRNLEV.GE.2) WRITE(OUTU,100) 'No iterations'
          ENDIF
          call DECOMP(NTPRP,LSTPRP,X,Y,Z,CG,WMAIN,NCLX,NCLY,NCLZ,DCEL, &
               IPHIW,TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN,HALF,QBSPL)
       ENDIF

       !.......................................................................
    ELSEIF(WRD.EQ.'GSBP')THEN
       !            -------------
#if KEY_GSBP==0
       CALL WRNDIE(-1,'<CHARMM>','GSBP code is not compiled.')
#else /**/
       QGSBP = .true.
       CALL PREP1  ! in pbeq.src
       CALL PREP2  ! in pbeq2.src
       !
       IF (TIMER.GT.1) CALL WRTTIM('GSBP times:')
       !     QC made a mistake
       CALL GSBP0(NATOM,X,Y,Z,CG,EGSBP,DX,DY,DZ,0,.true. &
#if KEY_SCCDFTB==1
            ,.FALSE.,ECRAP &      
#endif
            )
#endif 

       !.......................................................................
!     JZ_UW12: Implement SMBP 
      ELSEIF(WRD.EQ.'SMBP')THEN
       !            -------------
#if KEY_SMBP==0
        CALL WRNDIE(-1,'<CHARMM>','SMBP code is not compiled.')
#else /**/
        QSMBP = .true.
        QPHIX = .not.(INDXA(COMLYN, COMLEN, 'PHIX').GT.0)

        !---------------------------------------------------
        !Setup part: Calculate PHIX and setup grids 
        !(called by charmm_main.src)

        IF ((.not.QPHIX).and.PRNLEV.GE.5) THEN
          WRITE(OUTU,100) ''
          WRITE(OUTU,100) &
            '*******************************************'
          WRITE(OUTU,100) &
            '* PERFORMING SMBP SETUP: CALCULATING PHIX *'
          WRITE(OUTU,100) &
            '*******************************************'
          WRITE(OUTU,100) ''
        ENDIF

        CALL PREP1       ! in pbeq.src
        CALL PREP2_SMBP  ! in smbp.src

        IF (TIMER.GT.1) CALL WRTTIM('SMBP times:')
        CALL SMBP0_SETUP(NATOM,X,Y,Z,CG,DX,DY,DZ,0,.true.)

        QPHIX = .true.

        !---------------------------------------------------
#endif 

       !.......................................................................
       !  Get the Poisson-Boltzmann energy
    ELSEIF(WRD.EQ.'ENPB')THEN
       !            -------------
       IF(QPBEQ)THEN
          qpbint = ( INDXA(COMLYN,COMLEN,'INTE').GT.0 )
          if(qpbint)then
             ! allocate memory for vars local to PBEQ0 
             call chmalloc('pbeq.src','PBEQ0','ISLCT',NATOM,intg=ISLCT)
             call chmalloc('pbeq.src','PBEQ0','LSTPBI',NATOM,intg=LSTPBI)

             IMODE=0 !IMPLIES DEFAULT = ALL ATOMS SELECTED
             call SELRPN(COMLYN,COMLEN,ISLCT,NATOM,1,IMODE,.FALSE.,1,'',0, &
                  RESID,RES,IBASE,SEGID,NICTOT,NSEG,.TRUE.,X,Y,Z,.TRUE.,1,PBRAD)
             call MAKIND(NATOM,ISLCT,LSTPBI,NTPBI)
             IF(PRNLEV.GE.2) WRITE(OUTU,'(3X,A,I6,A)')  &
                  'Calculation with ',NTPBI,' atoms'
             call DECOMP(NTPBI,LSTPBI,X,Y,Z,CG,WMAIN,NCLX,NCLY,NCLZ,DCEL, &
                  IPHIW,TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN,ONE,QBSPL)

            ! free memory for vars local to PBEQ0
            call chmdealloc('pbeq.src','PBEQ0','ISLCT',NATOM,intg=ISLCT)
            call chmdealloc('pbeq.src','PBEQ0','LSTPBI',NATOM,intg=LSTPBI)
          else
             call DECOMP(NTPRP,LSTPRP,X,Y,Z,CG,WMAIN,NCLX,NCLY,NCLZ,DCEL, &
                  IPHIW,TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN,ONE,QBSPL)
          endif
          IF(PRNLEV.GE.2) WRITE(OUTU,100)  &
               'A factor of 1.0 is used in the energy'
          IF(PRNLEV.GE.2) WRITE(OUTU,100)
       ELSE
          IF(PRNLEV.GE.2) WRITE(OUTU,100) 'Grid is not setup!'
          IF(PRNLEV.GE.2) WRITE(OUTU,100)
       ENDIF

       !.......................................................................
       !  Get the density and number of counter ions
    ELSEIF(WRD.EQ.'COUN')THEN
       !            -------------
       IF(QPBEQ)THEN
          call COUNTERION(NCLX,NCLY,NCLZ,DCEL,IPHIW,MAY,VMEMB,TMEMB, &
               ZMEMB,TRANZ,ZBCEN,CONC,TEMP,QNONLINEAR,QPARTLINEAR)
       ELSE
          IF(PRNLEV.GE.2) WRITE(OUTU,100) 'Grid is not setup!'
          IF(PRNLEV.GE.2) WRITE(OUTU,100)
       ENDIF

       !.......................................................................
       !  Get the capacitive energy for membranes
    ELSEIF(WRD.EQ.'CAPA')THEN
       !            -------------
       IF(QPBEQ)THEN
          call CAPACI(NCLX,NCLY,NCLZ,DCEL,TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
               IPHIW,MAY,KAPPA2,EPSW,EPSM,VMEMB,TMEMB,ZMEMB)
       ELSE
          IF(PRNLEV.GE.2) WRITE(OUTU,100) 'Grid is not setup!'
          IF(PRNLEV.GE.2) WRITE(OUTU,100)
       ENDIF

       !.......................................................................
       !  Iterate on the Poisson-Boltzmann equation
    ELSEIF(WRD.EQ.'ITER')THEN
       !            -------------
       ! ITERATION Parameters (see prep1)
       MAXITS = GTRMI(COMLYN,COMLEN,'MAXI',2000)
       DEPS = GTRMF(COMLYN,COMLEN,'DEPS',0.000002*ONE)
       DOME = GTRMF(COMLYN,COMLEN,'LAMB',ONE)
       DOME = GTRMF(COMLYN,COMLEN,'DOME',DOME)
       ! PBEQ solver (see prep1)
       QOLDPB =INDXA(COMLYN, COMLEN, 'OLDP') .GT. 0
       QFMG= INDXA(COMLYN,COMLEN,'FMGR').GT.0
       IF(QFMG) THEN
          NCYC   = GTRMI(COMLYN,COMLEN,'NCYC',1)
          NPRE   = GTRMI(COMLYN,COMLEN,'NPRE',1)
          NPOST  = GTRMI(COMLYN,COMLEN,'NPOS',1)
       ENDIF
       QOSOR = INDXA(COMLYN, COMLEN, 'OSOR') .GT. 0
       QNONLINEAR = INDXA(COMLYN, COMLEN, 'NONL') .GT. 0
       QPARTLINEAR = INDXA(COMLYN, COMLEN, 'PART') .GT. 0
       QUNDER = INDXA(COMLYN, COMLEN, 'UNDE') .GT. 0
       !
       IF(QPBEQ)THEN
          IF(QFMG) THEN
             IF(MAXITS.GT.0)THEN
                IF(PRNLEV.GE.2) WRITE(OUTU,100)
                IF(PRNLEV.GE.2) WRITE(OUTU,100)  &
                     'Begin full multigrid algorithm'
                CALL PBEQ2(NATOM,X,Y,Z,CG,.false.,.true.)
                !                                         QENVV  QPRIN
                IF (TIMER.GT.1) &
                     CALL WRTTIM('FMG iteration times:')
             ELSE
                IF(PRNLEV.GE.2) WRITE(OUTU,100) 'No iterations'
             ENDIF
             call DECOMP(NTPRP,LSTPRP,X,Y,Z,CG,WMAIN,NCLX,NCLY,NCLZ,DCEL, &
                  IPHIW,TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN,HALF,QBSPL)
          ELSE
             IF(MAXITS.GT.0)THEN
                IF(PRNLEV.GE.2) WRITE(OUTU,100)
                IF(PRNLEV.GE.2) WRITE(OUTU,100) 'Begin iterative solution'
                IF(QOLDPB)THEN
                   call OLDPBEQ1(MAXITS,DOME,DEPS,NCLX,NCLY,NCLZ,IPHIW,MAYX, &
                        MAYY,MAYZ,ICHC,MAY,TMEMB,QOSOR,QFOCUS,QPBC,QNPBC,.true.)
                ELSEIF(QNONLINEAR) THEN
                   call PBEQ3(MAXITS,DOME,DEPS,TEMP,KAPPA2,NCLX,NCLY,NCLZ,DCEL,VMEMB, &
                        TMEMB,ZMEMB,TRANZ,ZBCEN,IPHIW,MAYX,MAYY,MAYZ,ICHC, &
                        MAY,QPBC,QNPBC,QUNDER)
                ELSEIF(QPARTLINEAR) THEN
                   call PBEQ4(MAXITS,DOME,DEPS,TEMP,KAPPA2,NCLX,NCLY,NCLZ,DCEL,VMEMB, &
                        TMEMB,ZMEMB,TRANZ,ZBCEN,IPHIW,MAYX,MAYY,MAYZ,ICHC, &
                        MAY,QPBC,QNPBC,QUNDER)
                ELSE
                   call PBEQ1(MAXITS,DOME,DEPS,EPSW,EPSP,EPSM,KAPPA2,NCLX,NCLY,NCLZ, &
                        IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN,IPHIW,MAYX,MAYY, &
                        MAYZ,ICHC,MAY,TMEMB,QOSOR,QFOCUS,QPBC,QNPBC,.true.)
                ENDIF
                IF (TIMER.GT.1) CALL WRTTIM('SOR iteration times:')
             ELSE
                IF(PRNLEV.GE.2) WRITE(OUTU,100) 'No iterations'
             ENDIF
             call DECOMP(NTPRP,LSTPRP,X,Y,Z,CG,WMAIN,NCLX,NCLY,NCLZ,DCEL, &
                  IPHIW,TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN,HALF,QBSPL)
          ENDIF
       ELSE
          IF(PRNLEV.GE.2) WRITE(OUTU,100)  &
               'Grid is not prepared, do prep first!'
          IF(PRNLEV.GE.2) WRITE(OUTU,100)
       ENDIF

       !.......................................................................
    ELSEIF(WRD.EQ.'WRIT')THEN
       !            -------------
       IF(QPBEQ)THEN
          CALL WRIGD0
       ELSE
          IF(PRNLEV.GE.2) WRITE(OUTU,100) 'Grid is not prepared, do prep first!'
          IF(PRNLEV.GE.2) WRITE(OUTU,100)
       ENDIF

       !.......................................................................
    ELSEIF(WRD.EQ.'PBAV')THEN   !RJP
       !            -------------
       IF(QPBEQ)THEN
          CALL PBAVGRID
       ELSE
          IF(PRNLEV.GE.2) WRITE(OUTU,100)  'Grid is not prepared, do prep first!'
          IF(PRNLEV.GE.2) WRITE(OUTU,100)
       ENDIF

       !.......................................................................
    ELSEIF(WRD.EQ.'READ')THEN
       !            -------------
       IF(INDXA(COMLYN,COMLEN,'CARD').GT.0)THEN
          CALL REAGD0
       ELSE
          CALL REAGD1
       ENDIF

       !.......................................................................
    ELSEIF(WRD.EQ.'COOR')THEN
       !            -------------
       CALL CORCOM(COMLYN,COMLEN)

       !.......................................................................
    ELSEIF(WRD.EQ.'SCAL')THEN
       !            -------------
       CALL SCALAR

       !.......................................................................
    ELSEIF(WRD.EQ.'HELP' .AND. PRNLEV.GE.2)THEN
       !            -------------
       WRITE(OUTU,100) 'List of commands:'
       WRITE(OUTU,100) 'SOLVE   - prepare grids and solve PB equation'
       WRITE(OUTU,100) 'ITERATE - iterate on the PB equation'
       WRITE(OUTU,100) 'ENPB    - compute the electrostatic PB energy'
       WRITE(OUTU,100) 'COUNterion  - compute the counter-ion distribution (Z)'
       WRITE(OUTU,100) 'CAPAcitance - compute the capacitance'
       WRITE(OUTU,100) 'WRITE   - write out the results'
       WRITE(OUTU,100) 'READ    - read the results (potential etc)'
       WRITE(OUTU,100) 'SCALAR  - call the scalar command '
       WRITE(OUTU,100) 'COOR    - call the coor command '
       WRITE(OUTU,100) 'RESET   - reset all grids and free memory'
       WRITE(OUTU,100) 'END     - exit the module'
       WRITE(OUTU,100)
    ELSEIF(WRD.EQ.'END ')THEN
       !            -------------
       DONE  = .TRUE.

       !.......................................................................
    ELSE
       !
       CALL WRNDIE(-1,'<PBEQ>','NOT A COMMAND:  '//WRD)

       DONE  = .TRUE.
    ENDIF

    !.......................................................................
    IF(DONE) RETURN
    GOTO 1000

  END SUBROUTINE PBEQ0
  !
  ! GSBP0 and PREP2 moved here from gsbp.src to avoid circular module dependency. LNI October 2009
  SUBROUTINE GSBP0(NATOM,X,Y,Z,CG,EGSBP,DX,DY,DZ,ICALL,QPRIN &
#if KEY_SCCDFTB==1
       !     QC_UW04: ADD potential SCC-DFTB iteration here
       ,QSCCTB,ESCCTB &
#endif 
       )
    
    !-----------------------------------------------------------------------
    !
#if KEY_GSBP==0
    CALL WRNDIE(-1,'<CHARMM>','GSBP code is not compiled.')
    RETURN
  END SUBROUTINE GSBP0
#else /**/
    !
    use chm_kinds
    use memory
    use number
    use stream
    use string
    use dimens_fcm
    use consta
    use comand
    use timerm
    use parallel
    use param_store, only: set_param
    use blockscc_fcm
    use ssbpm,only: qcavi,ntssbp,lstssbp,drmax2,rmax,acav,bcav
#if KEY_GCMC==1
    use gcmc   
#endif
#if KEY_SCCDFTB==1
    use sccdftb
    use sccgsbp
    !QC: UW0110: Also transfer molecular data
    use sccdftbsrc
    use gamess_fcm, only: qmused_sccdftb
#endif 
    use machutil,only:wrttim
    implicit none
    !
    real(chm_real),allocatable,dimension(:)   :: EMN,ALPX,ALPY,ALPZ,ADLPX,ADLPY,ADLPZ,AR,AC,AS
    real(chm_real),allocatable,dimension(:,:) :: AP,ADP

    INTEGER NATOM,ICALL
    real(chm_real)  X(*),Y(*),Z(*),DX(*),DY(*),DZ(*),CG(*)
    LOGICAL QPRIN
    !     QC_UW04: ADD potential SCC-DFTB iteration here
#if KEY_SCCDFTB==1
    LOGICAL QSCCTB
    real(chm_real)  ESCCTB  
    !     LOCAL
    !     real(chm_real),allocatable,dimension(:) :: COEFQM,MIJSCC
    ! QC: Avoid allocating MIJSCC in F95 conversion
    real(chm_real),allocatable,dimension(:) :: COEFQM
    real(chm_real),allocatable,dimension(:,:) :: COEFB
    real(chm_real)  ESCCTB2
    INTEGER I,II,J,N 
#endif 
    !     QC_UW04

    ! local
    real(chm_real)  EGSBP
    real(chm_real)  ECAVITY
    INTEGER LNCEL
    real(chm_real)  LTRAN,LDCEL,LXBCEN,LYBCEN,LZBCEN
    !
    !     HB Yu avoids returning for PARASCC: QC_UW0405
#if KEY_PARASCC==0
#if KEY_PARALLEL==1
    IF(MYNOD.GT.0) RETURN                      
#endif
#endif 
    !C      write(*,*)'GSBP0-begin>me,icall,ntpol,qlbox=',
    !C     $     mynod,icall,ntpol,qlbox
    !
    IF(ICALL.GT.0) GOTO 99
    !
    IF(QLBOX) THEN
       !     construct a large box to obtain boundary potentials of the original box.
       !     here, we just consider a cubic large box.
       LDCEL  = GTRMF(COMLYN,COMLEN,'LDCE',DCEL*FOUR)
       LNCEL  = GTRMI(COMLYN,COMLEN,'LNCE',33)
       LXBCEN = GTRMF(COMLYN,COMLEN,'LXBC',ZERO)
       LYBCEN = GTRMF(COMLYN,COMLEN,'LYBC',ZERO)
       LZBCEN = GTRMF(COMLYN,COMLEN,'LZBC',ZERO)

       IF ((LNCEL.GT.NCLX).OR.(LNCEL.GT.NCLY).OR.(LNCEL.GT.NCLZ)) &
            THEN
          CALL WRNDIE(-5,'<CHARMM>', &
               'LNCE too large, should be less than either NCLX or NCLY or NCLZ.')
       ENDIF

       IF(MOD(LNCEL,2).eq.0)THEN
          LNCEL=LNCEL+1
       ENDIF
       LTRAN=HALF*(LNCEL-1)*LDCEL

       QFOCUS=INDXA(COMLYN, COMLEN, 'FOCU') .GT. 0
       IF(QFOCUS) THEN
          IF(TRANX.GE.LTRAN .OR.TRANY.GE.LTRAN.OR.TRANZ.GE.LTRAN)THEN
             WRITE(OUTU,'(/,3X,A)') &
                  'GSBP WARNING: Large system should contain original one.'
             CALL WRNDIE(-5,'<GSBP>','FOCUSSING SELECTION ERROR')
          ENDIF
          WRITE(OUTU,'(/,3x,A,/,3x,A)') &
               'Following large box will be used to calculate boundary' , &
               'potentials of the original box using focussing'
       ELSE
          WRITE(OUTU,'(/,3x,A,/,3x,A)') &
               'Charge density on grid of the following large box will be', &
               'used to calculate boundary potentials of the original box'
       ENDIF
       WRITE(OUTU,101) &
            'Large Box in X from ',LXBCEN-LTRAN,' to ',LXBCEN+LTRAN
       WRITE(OUTU,101) &
            'Large Box in Y from ',LYBCEN-LTRAN,' to ',LYBCEN+LTRAN
       WRITE(OUTU,101) &
            'Large Box in Z from ',LZBCEN-LTRAN,' to ',LZBCEN+LTRAN
101    FORMAT(3X,A,F8.3,A,F8.3)
    ENDIF

    ! calculate the protein field and decompose electrosttic free energy
    CALL GDECOMP(NATOM,X,Y,Z,CG, &
         LNCEL,LTRAN,LDCEL,LXBCEN,LYBCEN,LZBCEN,QPRIN)
    IF(TIMER.GT.1.AND.ICALL.EQ.0) CALL WRTTIM('GDECOMP times:')

    ! Vmemb = 0 for the reaction field calculations using basis functions
    ! The influence of the transmembrane potential (VMEMB) is included
    ! in the (static) external field.
    IF(VMEMB.ne.ZERO) VMEMB=ZERO

    EGSBPB=ZERO
    call INITFORCE(NTRB,LSTRB,RXNBFX)
    call INITFORCE(NTRB,LSTRB,RXNBFY)
    call INITFORCE(NTRB,LSTRB,RXNBFZ)

    IF(NTPOL.EQ.0) GOTO 100
    ! set-up the generalized reaction field matrix MIJ using basis function
    IF(QRECTBOX) THEN
       IF(QLBOX) THEN
          CALL RECT_GSBP1(NATOM,X,Y,Z,CG, &
               LNCEL,LTRAN,LDCEL,LXBCEN,LYBCEN,LZBCEN,QPRIN)
       ELSE
          CALL RECT_GSBP2(NATOM,X,Y,Z,CG,QPRIN)
       ENDIF
    ELSEIF(QSPHERE) THEN
       IF(QLBOX) THEN
          CALL SPHE_GSBP1(NATOM,X,Y,Z,CG, &
               LNCEL,LTRAN,LDCEL,LXBCEN,LYBCEN,LZBCEN,QPRIN)
       ELSE
          CALL SPHE_GSBP2(NATOM,X,Y,Z,CG,QPRIN)
       ENDIF
    ELSEIF(QPROLATE) THEN
       IF(QLBOX) THEN
          CALL PROL_GSBP1(NATOM,X,Y,Z,CG, &
               LNCEL,LTRAN,LDCEL,LXBCEN,LYBCEN,LZBCEN,QPRIN)
       ELSE
          CALL PROL_GSBP2(NATOM,X,Y,Z,CG,QPRIN)
       ENDIF
    ELSEIF(QRFRECT) THEN
       IF(.not.(QMIJ.or.(QLBOX.and.QFOCUS))) THEN
          CALL WRNDIE(-5,'<GSBP>', &
               'LARGE BOX and FOCUSSING ARE NECESSARY')
       ELSE
          CALL RFRECT_GSBP1(NATOM,X,Y,Z,CG, &
               LNCEL,LTRAN,LDCEL,LXBCEN,LYBCEN,LZBCEN,QPRIN)
       ENDIF
    ELSEIF(QRFSPHE) THEN
       IF(.not.(QMIJ.or.(QLBOX.and.QFOCUS))) THEN
          CALL WRNDIE(-5,'<GSBP>', &
               'LARGE BOX and FOCUSSING ARE NECESSARY')
       ELSE
          CALL RFSPHE_GSBP1(NATOM,X,Y,Z,CG, &
               LNCEL,LTRAN,LDCEL,LXBCEN,LYBCEN,LZBCEN,QPRIN)
       ENDIF
    ENDIF

99  CONTINUE

    ! Calculate the reaction field energy and forces in region of interest
    IF(QRECTBOX) THEN
       call chmalloc('pbeq.src','GSBP0','EMN',NTPOL,crl=EMN)
       call chmalloc('pbeq.src','GSBP0','ALPX',XNPOL,crl=ALPX)
       call chmalloc('pbeq.src','GSBP0','ALPY',YNPOL,crl=ALPY)
       call chmalloc('pbeq.src','GSBP0','ALPZ',ZNPOL,crl=ALPZ)
       call chmalloc('pbeq.src','GSBP0','ADLPX',XNPOL,crl=ADLPX)
       call chmalloc('pbeq.src','GSBP0','ADLPY',YNPOL,crl=ADLPY)
       call chmalloc('pbeq.src','GSBP0','ADLPZ',ZNPOL,crl=ADLPZ)
       !     make a list of basis functions
       CALL RECT_STPOL(NTRB,LSTRB,X,Y,Z,CG,NTPOL, &
            XNPOL,YNPOL,ZNPOL,RRXCEN,RRYCEN,RRZCEN, &
            XSCALE,YSCALE,ZSCALE,MIJ,COEF,EMN, &
            LSTPX,LSTPY,LSTPZ, &
            ALPX,ALPY,ALPZ, &
            BNORM,LSTPOL,ICALL,NLIST,QNOSORT &
#if KEY_GCMC==1
            ,GCMCON  &  
#endif
#if KEY_SCCDFTB==1
            ,COEFX &    
#endif
            )
       IF(TIMER.GT.1.AND.ICALL.EQ.0) CALL WRTTIM('STPOL times:')
       CALL RECT_GSBP3(NTRB,LSTRB,X,Y,Z,CG,NTPOL,MAXNPOL, &
            RRXCEN,RRYCEN,RRZCEN,XSCALE,YSCALE,ZSCALE, &
            MIJ,COEF, &
            RXNBFX,RXNBFY,RXNBFZ, &
            LSTPX,LSTPY,LSTPZ,BNORM, &
            ALPX,ALPY,ALPZ, &
            ADLPX,ADLPY,ADLPZ, &
            XNPOL,YNPOL,ZNPOL, &
            MQ,LSTPOL,EGSBPB &
#if KEY_GCMC==1
            ,GCMCON  & 
#endif
            )
       IF (TIMER.GT.1.and.ICALL.eq.0) CALL WRTTIM('GSBP3 times:')
    ELSEIF(QSPHERE) THEN
       call chmalloc('pbeq.src','GSBP0','EMN',NTPOL,crl=EMN)
       call chmalloc('pbeq.src','GSBP0','AR',NMPOL,lbou=0,crl=AR)
       call chmalloc('pbeq.src','GSBP0','AC',NMPOL,lbou=0,crl=AC)
       call chmalloc('pbeq.src','GSBP0','AS',NMPOL,lbou=0,crl=AS)
       call chmalloc('pbeq.src','GSBP0','AP',NMPOL,NMPOL,lbou=0,lbou2=0,crl=AP)
       call chmalloc('pbeq.src','GSBP0','ADP',NMPOL,NMPOL,lbou=0,lbou2=0,crl=ADP)
       !     make a list of basis functions
       !     QC_UW06: we assume that the same set of basis functions
       !     will be constructed for the two states involved in the
       !     SCC-DFTB free energy perturbation calculations - which should
       !     be reasonable if the difference is small (to be improved later)
       !     The same assumption is made for sorting. 
       !     QC - UW07
       CALL SPHE_STPOL(NTRB,LSTRB,X,Y,Z,CG,NTPOL,SRDIST, &
            RRXCEN,RRYCEN,RRZCEN,MIJ,COEF,EMN, &
            AR,AC,AS,AP,NMPOL, &
            LSTPL,LSTPM,BNORM,LSTPOL, &
            ICALL,NLIST,QNOSORT &
#if KEY_SCCDFTB==1
            ,COEFX  & 
#endif
#if KEY_GCMC==1
            ,GCMCON  & 
#endif
            )

       IF(TIMER.GT.1.and.ICALL.eq.0) CALL WRTTIM('STPOL times:')
       CALL SPHE_GSBP3(NTRB,LSTRB,X,Y,Z,CG,NTPOL,MAXNPOL, &
            RRXCEN,RRYCEN,RRZCEN, &
            MIJ,COEF, &
            RXNBFX,RXNBFY,RXNBFZ, &
            LSTPL,LSTPM,BNORM, &
            AR,AC,AS,AP,ADP,NMPOL, &
            MQ,LSTPOL,EGSBPB &
#if KEY_GCMC==1
            ,GCMCON  & 
#endif
            )
       IF (TIMER.GT.1.and.ICALL.eq.0) CALL WRTTIM('GSBP3 times:')
    ELSEIF(QPROLATE) THEN
       call chmalloc('pbeq.src','GSBP0','EMN',NTPOL,crl=EMN)
       !     make a list of basis functions
       CALL PROL_STPOL(NTRB,LSTRB,X,Y,Z,CG,MAJOR,MINOR,NTPOL, &
            RRXCEN,RRYCEN,RRZCEN,MMIJ,COEF,EMN, &
            LSTPL,LSTPM,BNORM,LSTPOL, &
            ICALL,NLIST,QNOSORT &
            )
       IF(TIMER.GT.1.and.ICALL.eq.0) CALL WRTTIM('STPOL times:')
       CALL PROL_GSBP3(NTRB,LSTRB,X,Y,Z,CG,MAJOR,MINOR, &
            NTPOL,MAXNPOL,RRXCEN,RRYCEN,RRZCEN, &
            MMIJ,COEF, &
            RXNBFX,RXNBFY,RXNBFZ, &
            LSTPL,LSTPM,BNORM, &
            MQ,LSTPOL,EGSBPB &
            )
       IF (TIMER.GT.1.and.ICALL.eq.0) CALL WRTTIM('GSBP3 times:')
    ELSEIF(QRFRECT) THEN
       call chmalloc('pbeq.src','GSBP0','EMN',NTPOL,crl=EMN)
       call chmalloc('pbeq.src','GSBP0','ALPX',XNPOL,crl=ALPX)
       call chmalloc('pbeq.src','GSBP0','ALPY',YNPOL,crl=ALPY)
       call chmalloc('pbeq.src','GSBP0','ALPZ',ZNPOL,crl=ALPZ)
       call chmalloc('pbeq.src','GSBP0','ADLPX',XNPOL,crl=ADLPX)
       call chmalloc('pbeq.src','GSBP0','ADLPY',YNPOL,crl=ADLPY)
       call chmalloc('pbeq.src','GSBP0','ADLPZ',ZNPOL,crl=ADLPZ)
       !     make a list of basis functions
       CALL RECT_STPOL(NTRB,LSTRB,X,Y,Z,CG,NTPOL, &
            XNPOL,YNPOL,ZNPOL,RRXCEN,RRYCEN,RRZCEN, &
            XSCALE,YSCALE,ZSCALE,MMIJ,COEF,EMN, &
            LSTPX,LSTPY,LSTPZ, &
            ALPX,ALPY,ALPZ, &
            BNORM,LSTPOL,ICALL,NLIST,QNOSORT &
#if KEY_GCMC==1
            ,GCMCON  & 
#endif
#if KEY_SCCDFTB==1
            ,COEFX &   
#endif
            )
       IF(TIMER.GT.1.AND.ICALL.EQ.0) CALL WRTTIM('STPOL times:')
       CALL RECT_GSBP3(NTRB,LSTRB,X,Y,Z,CG,NTPOL,MAXNPOL, &
            RRXCEN,RRYCEN,RRZCEN,XSCALE,YSCALE,ZSCALE, &
            MMIJ,COEF, &
            RXNBFX,RXNBFY,RXNBFZ, &
            LSTPX,LSTPY,LSTPZ,BNORM, &
            ALPX,ALPY,ALPZ, &
            ADLPX,ADLPY,ADLPZ, &
            XNPOL,YNPOL,ZNPOL, &
            MQ,LSTPOL,EGSBPB &
#if KEY_GCMC==1
            ,GCMCON  & 
#endif
            )
       IF (TIMER.GT.1.and.ICALL.eq.0) CALL WRTTIM('GSBP3 times:')
    ELSEIF(QRFSPHE) THEN
       call chmalloc('pbeq.src','GSBP0','EMN',NTPOL,crl=EMN)
       call chmalloc('pbeq.src','GSBP0','AR',NMPOL,lbou=0,crl=AR)
       call chmalloc('pbeq.src','GSBP0','AC',NMPOL,lbou=0,crl=AC)
       call chmalloc('pbeq.src','GSBP0','AS',NMPOL,lbou=0,crl=AS)
       call chmalloc('pbeq.src','GSBP0','AP',NMPOL,NMPOL,lbou=0,lbou2=0,crl=AP)
       call chmalloc('pbeq.src','GSBP0','ADP',NMPOL,NMPOL,lbou=0,lbou2=0,crl=ADP)
       CALL SPHE_STPOL(NTRB,LSTRB,X,Y,Z,CG,NTPOL,SRDIST, &
            RRXCEN,RRYCEN,RRZCEN,MMIJ,COEF,EMN, &
            AR,AC,AS,AP,NMPOL, &
            LSTPL,LSTPM,BNORM,LSTPOL, &
            ICALL,NLIST,QNOSORT &
#if KEY_SCCDFTB==1
            ,COEFX   & 
#endif
#if KEY_GCMC==1
            ,GCMCON  & 
#endif
            )

       IF(TIMER.GT.1.and.ICALL.eq.0) CALL WRTTIM('STPOL times:')
       CALL SPHE_GSBP3(NTRB,LSTRB,X,Y,Z,CG,NTPOL,MAXNPOL, &
            RRXCEN,RRYCEN,RRZCEN, &
            MMIJ,COEF, &
            RXNBFX,RXNBFY,RXNBFZ, &
            LSTPL,LSTPM,BNORM, &
            AR,AC,AS,AP,ADP,NMPOL, &
            MQ,LSTPOL,EGSBPB &
#if KEY_GCMC==1
            ,GCMCON  & 
#endif
            )
       IF (TIMER.GT.1.and.ICALL.eq.0) CALL WRTTIM('GSBP3 times:')
    ENDIF

100 CONTINUE
    IF(QFOCUS) QFOCUS=.false.

    ! calculate the protein field energy and forces (NOTE: HALF -> ONE)
    IF(QGAB.or.QPHIX) THEN
       CALL RFORCE2(NTRB,LSTRB,X,Y,Z,CG, &
            NCLX,NCLY,NCLZ,DCEL,IPHIX, &
            TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN,ONE, &
            RXNAFX,RXNAFY,RXNAFZ,EGSBPA,QBSPL &
#if KEY_GCMC==1
            ,GCMCON  & 
#endif
            )
    ELSEIF(ICALL.EQ.0) THEN
       EGSBPA=ZERO
       call INITFORCE(NTRB,LSTRB,RXNAFX)
       call INITFORCE(NTRB,LSTRB,RXNAFY)
       call INITFORCE(NTRB,LSTRB,RXNAFZ)
    ENDIF

    ! return potential energy and forces
    CALL GSBP4(NTRB,LSTRB,EGSBPA,EGSBPB,EGSBP, &
         DX,DY,DZ,RXNAFX,RXNAFY,RXNAFZ, &
         RXNBFX,RXNBFY,RXNBFZ,QPRIN &
#if KEY_GCMC==1
         ,GCMCON     & 
#endif
         )
    call set_param('GSBE', EGSBP)

    IF(QCAVI) THEN
#if KEY_GCMC==1
       IF (QRECTBOX.OR.QRFRECT) THEN
          CALL REC_CAVP(ECAVITY,DX,DY,DZ,X,Y,Z,GCMCON,NTSSBP, &
               LSTSSBP,RBXMIN,RBXMAX,RBYMIN,RBYMAX, &
               RBZMIN,RBZMAX,DRMAX2)
       ELSEIF (QSPHERE) THEN
          CALL SPH_CAVP(ECAVITY,DX,DY,DZ,X,Y,Z,GCMCON,NTSSBP, &
               LSTSSBP,RRXCEN,RRYCEN,RRZCEN, &
               SRDIST,RMAX,ACAV,BCAV)
       ENDIF
#endif 
       EGSBP = EGSBP + ECAVITY
       IF(QPRIN) THEN
          WRITE(OUTU,'(3X,A,F13.5,A)') &
               'Non-polar cavity potential                       =', &
               ECAVITY,' [KCAL/MOL]'
       ENDIF
    ENDIF
    call set_param('GSBP', EGSBP)
    call set_param('GSBC', ECAVITY)

#if KEY_SCCDFTB==1

    if ( (.not. qmused_sccdftb) .or. (.not. qscctb) &
         .or. (nscctc .eq. 0)) goto 777

    !     ----------------------------------------------------------------
    !     QC_UW04: Before we release the stack, after the MM components
    !     are done, proceed to carry out the SCF cycles and evaluate
    !     the corresponding contributions to energy and force
    !     FIRST, evaluate necessary elements for SCF, the GAMMA and 
    !     OMEGA arrays, store those in a common block and transfer to
    !     SCC-DFTB. Need to improve memory manipulation later (O.K. for
    !     now because the QM region is typically small)
    !     QC_UW06: From this point, the SCC-GSBP related stuff should be
    !     state-dependent in the case of FEP calculations - either 
    !     dual topology or singe topology.
    !     ----------------------------------------------------------------
    !     State-independet quantities - MIJ,COEF
    ! LNI: NOTE WHEN CONVERTING TO F95 THAT MIJSCC,
    !      MIJ and MMIJ may be ALLOCATABLE ARRAYS; should 
    ! perhaps be pointers if the following operation is expensive
    !!    QC: Yes - avoid transfering pointers 
    !!            - simply use the correct array in the following calls
    !!    UW0110  
    IF (.NOT.(QSPHERE.OR.QRECTBOX.OR.QRFSPHE.OR.QRFRECT)) THEN
       !       write(*,*) "GSBP0: SPHERE/RECTBOX for SCC"
       !!      MIJSCC=MIJ
       !     ELSE IF (QRFSPHE.OR.QRFRECT) THEN
       !       write(*,*) "GSBP0: RFSPHE/RFRECT  for SCC"
       !!      MIJSCC=MMIJ
       !     ELSE 
       ! NOT YET IMPLEMENTED
       CALL WRNDIE(-5,'<GSBP>', &
            'OPTION NOT ALLOWED FOR SCC YET')

    ENDIF
    !     QC: Update the MM List for SCC - to include ONLY the INNER atoms
    !     Considering the use of GSBP, we ASSUME that the list does NOT
    !     need to be updated (state-independent)
    !  QC: 11/17 For PERT based on Xiya
    !  IF (.NOT.LSCCMM) CALL UPSCCMM(NTRB,LSTRB)
    IF ( &
#if KEY_PERT == 1 
         .NOT. QPERT .AND. &
#endif /* pert */
         .NOT.LSCCMM) THEN
       CALL UPSCCMM(NTRB,LSTRB)
#if KEY_PERT == 1
    ELSEIF (QPERT) THEN
       CALL UPSCCMM(NTRB,LSTRB)   ! Xiya:state dependent
#endif /* pert */
    ENDIF
    !  QC: 11/17 Done

    !    ========================= STATE A =============================
    if (qlamda) then
       !      >>>>>>>>>>>>>>>>>> restore state parameters <<<<<<<<<<<<<<<<<<<<
       nel=nela
       do n=1,nndim
          izp2(n)=izp2a(n)
       enddo
       TELEC=TELEC1
       SCFTOL=SCFTOL1
       MXITSCF=MXITSCF1
       LPRVEC=LPRVEC1
       ICHSCC=ICHSCC1
       nscctc=nscctc1
       do ii=1,nscctc1
          scctyp(ii)=scctyp1(ii)
          scczin(ii)=scczin1(ii)
          sccatm(ii)=sccatm1(ii)
       enddo
       do ii=1,nscctc1
          iqmlst(ii,1)=iqmlst1(ii,1)
       enddo
       !MG_UW1211: for spin-polarization get the right number of electrons
       lcolspin=lcolspin1
       if (lcolspin) then
         nelup=nelupa
         neldown=neldowna
       endif
       !MG_UW1211: bookkeeping of charges (globally stored at ltm/sccdftbsrc_ltm.src)
       if (lldep) then
         do i=1,3*nscctc
           ql(i)=qla(i)
         enddo
       else
         do i=1,nscctc
           qmat(i)=qmata(i)
         enddo
       endif
       if (lcolspin) then
         do i=1,3*nscctc
           qlup(i)=qlupa(i)
           qldown(i)=qldowna(i)
         enddo
       endif

       scal=1.0d0-scclamd
    endif ! qlamda
    !     >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    !       ALLOCATE SOME MEMORY
    !       Haibo Yu more memory for replica
    !
    call chmalloc('pbeq.src','GSBP0','COEFB',NTPOL,NSCCTC*NSCCRP,crl=COEFB)
    call chmalloc('pbeq.src','GSBP0','COEFQM',NTPOL*NSCCRP,crl=COEFQM)

    !       write(*,*) "GSBP0> Key arrays allocated",NSCCTC,NSCCRP

    !       IF (QRECTBOX.OR.QRFRECT) THEN
    IF (QRECTBOX) THEN
       CALL SCCRECT0(X,Y,Z,NTPOL,MAXNPOL,XNPOL,YNPOL,ZNPOL, &
            RRXCEN,RRYCEN,RRZCEN,XSCALE,YSCALE,ZSCALE, &
            MIJ,COEF,COEFX,MQ,COEFB, &
            LSTPX,LSTPY,LSTPZ, &
            ALPX,ALPY,ALPZ,BNORM, &
            LSTPOL,NCLX,NCLY,NCLZ,DCEL,IPHIX, &
            TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN,ONE,QBSPL)
    ELSEIF (QRFRECT) THEN
       CALL SCCRECT0(X,Y,Z,NTPOL,MAXNPOL,XNPOL,YNPOL,ZNPOL, &
            RRXCEN,RRYCEN,RRZCEN,XSCALE,YSCALE,ZSCALE, &
            MMIJ  ,COEF,COEFX,MQ,COEFB, &
            LSTPX,LSTPY,LSTPZ, &
            ALPX,ALPY,ALPZ,BNORM, &
            LSTPOL,NCLX,NCLY,NCLZ,DCEL,IPHIX, &
            TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN,ONE,QBSPL)
    ELSEIF (QSPHERE) THEN
       CALL SCCGSBP0(X,Y,Z,NTPOL,MAXNPOL,SRDIST,RRXCEN,RRYCEN,RRZCEN, &
            MIJ,COEF,COEFX,MQ,COEFB, &
            LSTPL,LSTPM,BNORM, &
            AR,AC,AS,AP,ADP,NMPOL, &
            LSTPOL,NCLX,NCLY,NCLZ,DCEL,IPHIX, &
            TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN,ONE,QBSPL)
    ELSEIF (QRFSPHE) THEN
       CALL SCCGSBP0(X,Y,Z,NTPOL,MAXNPOL,SRDIST,RRXCEN,RRYCEN,RRZCEN, &
            MMIJ  ,COEF,COEFX,MQ,COEFB, &
            LSTPL,LSTPM,BNORM, &
            AR,AC,AS,AP,ADP,NMPOL, &
            LSTPOL,NCLX,NCLY,NCLZ,DCEL,IPHIX, &
            TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN,ONE,QBSPL)
    ELSE
       !  NOT YET IMPLEMENTED
       CALL WRNDIE(-5,'<GSBP>','OPTION NOT ALLOWED FOR SCC YET')
    ENDIF

    !      Now call the scctbene to do the SCF
    QGSBPSCC=.TRUE.
    !      ENFORCE MULLIK - required in the SCC-GSBP FORCE CALCULATIONS
    IF (.NOT.LMULIK) LMULIK=.TRUE.
    !      XIAO_PHK_QC_UW0609 (SCCTBENE call)
    !      write(*,*) "GSBP0> Ready to call Scctbene?"
    CALL SCCTBENE(ESCCTB,X,Y,Z,DX,DY,DZ,NATOM,.true.) 
    QGSBPSCC=.FALSE.

    if (qlamda) then
      !MG_UW1211: bookkeeping of charges (globally stored at ltm/sccdftbsrc_ltm.src)
      if (lldep) then
        do i=1,3*nscctc
          qla(i)=ql(i)
        enddo
      else
        do i=1,nscctc
          qmata(i)=qmat(i)
        enddo
      endif
      if (lcolspin) then
        do i=1,3*nscctc
          qlupa(i)=qlup(i)
          qldowna(i)=qldown(i)
        enddo
      endif
    endif

    !      QC: Skip the following for Non-master? 
    !...##IF PARASCC
    !       IF (MYNOD.EQ.0) THEN
    !...##ENDIF
    !      Assemble QM-GSBP components of energy and force?
    !      IF (QRECTBOX.OR.QRFRECT) THEN
    IF (QRECTBOX) THEN
       CALL SCCRECT4(NTRB,LSTRB,X,Y,Z,DX,DY,DZ, &
            NTPOL,MAXNPOL,RRXCEN,RRYCEN,RRZCEN,XSCALE,YSCALE,ZSCALE,CG, &
            MIJ   ,COEF,COEFX,MQ,COEFQM, &
            LSTPX,LSTPY,LSTPZ,BNORM, &
            ALPX,ALPY,ALPZ, &
            ADLPX,ADLPY,ADLPZ,XNPOL,YNPOL,ZNPOL, &
            LSTPOL,NCLX,NCLY,NCLZ,DCEL,IPHIX, &
            TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN,ONE, &
            RXNAFX,RXNAFY,RXNAFZ, &
            RXNBFX,RXNBFY,RXNBFZ, QBSPL)
    ELSEIF (QRFRECT) THEN

       CALL SCCRECT4(NTRB,LSTRB,X,Y,Z,DX,DY,DZ, &
            NTPOL,MAXNPOL,RRXCEN,RRYCEN,RRZCEN,XSCALE,YSCALE,ZSCALE,CG, &
            MMIJ  ,COEF,COEFX,MQ,COEFQM, &
            LSTPX,LSTPY,LSTPZ,BNORM, &
            ALPX,ALPY,ALPZ, &
            ADLPX,ADLPY,ADLPZ,XNPOL,YNPOL,ZNPOL, &
            LSTPOL,NCLX,NCLY,NCLZ,DCEL,IPHIX, &
            TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN,ONE, &
            RXNAFX,RXNAFY,RXNAFZ, &
            RXNBFX,RXNBFY,RXNBFZ, QBSPL)
       !      ELSEIF (QSPHERE.OR.QRFSPHE) THEN
    ELSEIF (QSPHERE) THEN
       CALL SCCGSBP4(NTRB,LSTRB,X,Y,Z,DX,DY,DZ, &
            NTPOL,MAXNPOL,SRDIST,RRXCEN,RRYCEN,RRZCEN,CG, &
            MIJ   ,COEF,COEFX,MQ,COEFQM, &
            LSTPL,LSTPM,BNORM, &
            AR,AC,AS,AP,ADP,NMPOL, &
            LSTPOL,NCLX,NCLY,NCLZ,DCEL,IPHIX, &
            TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN,ONE, &
            RXNAFX,RXNAFY,RXNAFZ, &
            RXNBFX,RXNBFY,RXNBFZ,QBSPL)
    ELSEIF (QRFSPHE) THEN

       CALL SCCGSBP4(NTRB,LSTRB,X,Y,Z,DX,DY,DZ, &
            NTPOL,MAXNPOL,SRDIST,RRXCEN,RRYCEN,RRZCEN,CG, &
            MMIJ  ,COEF,COEFX,MQ,COEFQM, &
            LSTPL,LSTPM,BNORM, &
            AR,AC,AS,AP,ADP,NMPOL, &
            LSTPOL,NCLX,NCLY,NCLZ,DCEL,IPHIX, &
            TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN,ONE, &
            RXNAFX,RXNAFY,RXNAFZ, &
            RXNBFX,RXNBFY,RXNBFZ,QBSPL)
    ELSE
       ! NOT YET IMPLEMENTED
       CALL WRNDIE(-5,'<GSBP>', &
            'OPTION NOT ALLOWED FOR SCC YET')
    ENDIF
    !...##IF PARASCC
    !       ENDIF
    !...##ENDIF

    !      THAT IS ALL (QC_GSBP, Nov. 2003), FREE UP MEMORY
    call chmdealloc('pbeq.src','GSBP0','COEFB',NTPOL,NSCCTC*NSCCRP,crl=COEFB)
    call chmdealloc('pbeq.src','GSBP0','COEFQM',NTPOL*NSCCRP,crl=COEFQM)
    if (.not.qlamda) GOTO 777
    !    ========================= STATE B =============================
    !     >>>>>>>>>>>>>>>>>> restore state parameters <<<<<<<<<<<<<<<<<<<<
    nel=nelb
    do n=1,nndim
       izp2(n)=izp2b(n)
    enddo
    TELEC=TELEC2
    SCFTOL=SCFTOL2
    MXITSCF=MXITSCF2
    LPRVEC=LPRVEC2
    ICHSCC=ICHSCC2
    nscctc=nscctc2
    do ii=1,nscctc2
       scctyp(ii)=scctyp2(ii)
       scczin(ii)=scczin2(ii)
       sccatm(ii)=sccatm2(ii)
    enddo
    do ii=1,nscctc2
       iqmlst(ii,1)=iqmlst2(ii,1)
    enddo
    !MG_UW1211: for spin-polarization get the right number of electrons
    lcolspin=lcolspin2
    if (lcolspin) then
      nelup=nelupb
      neldown=neldownb
    endif
    !MG_UW1211: bookkeeping of charges (globally stored at ltm/sccdftbsrc_ltm.src)
    if (lldep) then
      do i=1,3*nscctc
        ql(i)=qlb(i)
      enddo
    else
      do i=1,nscctc
        qmat(i)=qmatb(i)
      enddo
    endif
    if (lcolspin) then
      do i=1,3*nscctc
        qlup(i)=qlupb(i)
        qldown(i)=qldownb(i)
      enddo
    endif

    scal=scclamd
    !     >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    !       ALLOCATE SOME MEMORY
    ! coefficients of basis functions
    call chmalloc('pbeq.src','GSBP0','COEFB',NTPOL,NSCCTC*NSCCRP,crl=COEFB)
    ! coefficients of basis functions QM
    call chmalloc('pbeq.src','GSBP0','COEFQM',NTPOL*NSCCRP,crl=COEFQM)

    !      IF (QRECTBOX.OR.QRFRECT) THEN
    IF (QRECTBOX) THEN
       CALL SCCRECT0(X,Y,Z,NTPOL,MAXNPOL,XNPOL,YNPOL,ZNPOL, &
            RRXCEN,RRYCEN,RRZCEN,XSCALE,YSCALE,ZSCALE, &
            MIJ   ,COEF,COEFX,MQ,COEFB, &
            LSTPX,LSTPY,LSTPZ, &
            ALPX,ALPY,ALPZ,BNORM, &
            LSTPOL,NCLX,NCLY,NCLZ,DCEL,IPHIX, & !Puja-bugfix
            TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN,ONE, QBSPL)
    ELSEIF (QRFRECT) THEN

       CALL SCCRECT0(X,Y,Z,NTPOL,MAXNPOL,XNPOL,YNPOL,ZNPOL, &
            RRXCEN,RRYCEN,RRZCEN,XSCALE,YSCALE,ZSCALE, &
            MMIJ  ,COEF,COEFX,MQ,COEFB, &
            LSTPX,LSTPY,LSTPZ, &
            ALPX,ALPY,ALPZ,BNORM, &
            LSTPOL,NCLY,NCLZ,DCEL,IPHIX, &
            TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN,ONE, QBSPL)
       !      ELSEIF (QSPHERE.OR.QRFSPHE) THEN
    ELSEIF (QSPHERE) THEN
       CALL SCCGSBP0(X,Y,Z,NTPOL,MAXNPOL,SRDIST,RRXCEN,RRYCEN,RRZCEN, &
            MIJ   ,COEF,COEFX,MQ,COEFB, &
            LSTPL,LSTPM,BNORM, &
            AR,AC,AS,AP,ADP,NMPOL, &
            LSTPOL,NCLX,NCLY,NCLZ,DCEL,IPHIX, &
            TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN,ONE, QBSPL)
    ELSEIF (QRFSPHE) THEN

       CALL SCCGSBP0(X,Y,Z,NTPOL,MAXNPOL,SRDIST,RRXCEN,RRYCEN,RRZCEN, &
            MMIJ  ,COEF,COEFX,MQ,COEFB, &
            LSTPL,LSTPM,BNORM, &
            AR,AC,AS,AP,ADP,NMPOL, &
            LSTPOL,NCLX,NCLY,NCLZ,DCEL,IPHIX, &
            TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN,ONE, QBSPL)
    ELSE
       !    NOT YET IMPLEMENTED
       CALL WRNDIE(-5,'<GSBP>', &
            'OPTION NOT ALLOWED FOR SCC YET')
    ENDIF

    !      Now call the scctbene to do the SCF
    QGSBPSCC=.TRUE.
    !      XIAO_PHK_QC_UW0609 ! c36a3 has ESCCTB, not ESCCTB2 ????
    CALL SCCTBENE(ESCCTB2,X,Y,Z,DX,DY,DZ,NATOM,.true.)
    QGSBPSCC=.FALSE.

    !MG_UW1211: bookkeeping of charges (globally stored at ltm/sccdftbsrc_ltm.src)
    if (lldep) then
      do i=1,3*nscctc
        qlb(i)=ql(i)
      enddo
    else
      do i=1,nscctc
        qmatb(i)=qmat(i)
      enddo
    endif
    if (lcolspin) then
      do i=1,3*nscctc
        qlupb(i)=qlup(i)
        qldownb(i)=qldown(i)
      enddo
    endif

    !      QC: Skip the following for Non-master? 
    !...##IF PARASCC
    !       IF (MYNOD.EQ.0) THEN
    !...##ENDIF
    !      Assemble QM-GSBP components of energy and force?
    !      IF (QRECTBOX.OR.QRFRECT) THEN
    IF (QRECTBOX) THEN
       CALL SCCRECT4(NTRB,LSTRB,X,Y,Z,DX,DY,DZ, &
            NTPOL,MAXNPOL,RRXCEN,RRYCEN,RRZCEN,XSCALE,YSCALE,ZSCALE,CG, &
            MIJ   ,COEF,COEFX,MQ,COEFQM, &
            LSTPX,LSTPY,LSTPZ,BNORM, &
            ALPX,ALPY,ALPZ, &
            ADLPX,ADLPY,ADLPZ,XNPOL,YNPOL,ZNPOL, &
            LSTPOL,NCLX,NCLY,NCLZ,DCEL,IPHIX, &
            TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN,ONE, &
            RXNAFX,RXNAFY,RXNAFZ, &
            RXNBFX,RXNBFY,RXNBFZ,QBSPL)
    ELSEIF (QRFRECT) THEN

       CALL SCCRECT4(NTRB,LSTRB,X,Y,Z,DX,DY,DZ, &
            NTPOL,MAXNPOL,RRXCEN,RRYCEN,RRZCEN,XSCALE,YSCALE,ZSCALE,CG, &
            MMIJ  ,COEF,COEFX,MQ,COEFQM, &
            LSTPX,LSTPY,LSTPZ,BNORM, &
            ALPX,ALPY,ALPZ, &
            ADLPX,ADLPY,ADLPZ,XNPOL,YNPOL,ZNPOL, &
            LSTPOL,NCLX,NCLY,NCLZ,DCEL,IPHIX, &
            TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN,ONE, &
            RXNAFX,RXNAFY,RXNAFZ, &
            RXNBFX,RXNBFY,RXNBFZ,QBSPL)
       !      ELSEIF (QSPHERE.OR.QRFSPHE) THEN
    ELSEIF (QSPHERE) THEN
       CALL SCCGSBP4(NTRB,LSTRB,X,Y,Z,DX,DY,DZ, &
            NTPOL,MAXNPOL,SRDIST,RRXCEN,RRYCEN,RRZCEN,CG, &
            MIJ   ,COEF,COEFX,MQ,COEFQM, &
            LSTPL,LSTPM,BNORM, &
            AR,AC,AS,AP,ADP,NMPOL, &
            LSTPOL,NCLX,NCLY,NCLZ,DCEL,IPHIX, &
            TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN,ONE, &
            RXNAFX,RXNAFY,RXNAFZ, &
            RXNBFX,RXNBFY,RXNBFZ,QBSPL)
    ELSEIF (QRFSPHE) THEN

       CALL SCCGSBP4(NTRB,LSTRB,X,Y,Z,DX,DY,DZ, &
            NTPOL,MAXNPOL,SRDIST,RRXCEN,RRYCEN,RRZCEN,CG, &
            MMIJ  ,COEF,COEFX,MQ,COEFQM, &
            LSTPL,LSTPM,BNORM, &
            AR,AC,AS,AP,ADP,NMPOL, &
            LSTPOL,NCLX,NCLY,NCLZ,DCEL,IPHIX, &
            TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN,ONE, &
            RXNAFX,RXNAFY,RXNAFZ, &
            RXNBFX,RXNBFY,RXNBFZ,QBSPL)
    ELSE
       !  NOT YET IMPLEMENTED
       CALL WRNDIE(-5,'<GSBP>', &
            'OPTION NOT ALLOWED FOR SCC YET')
    ENDIF

    !...##IF PARASCC
    !       ENDIF
    !...##ENDIF

    !      THAT IS ALL (QC_GSBP, Nov. 2003), FREE UP MEMORY
    call chmdealloc('pbeq.src','GSBP0','COEFB',NTPOL,NSCCTC*NSCCRP,crl=COEFB)
    call chmdealloc('pbeq.src','GSBP0','COEFQM',NTPOL*NSCCRP,crl=COEFQM)

    !     ---------------- Collect fep related quantities --------------
    !     forces should be scaled inside SCCTBENE and SCCGSBP4
    !     QC: UW_031205: flip the order here - dumb mistake!

    dvdlscc=       -         ESCCTB+        ESCCTB2
    ESCCTB = (1.0d0-scclamd)*ESCCTB+scclamd*ESCCTB2

    !     ==============================================================

777 CONTINUE 
#endif 
    IF(NTPOL == 0) RETURN
    IF(QRECTBOX.OR.QRFRECT) THEN
       call chmdealloc('pbeq.src','GSBP0','EMN',NTPOL,crl=EMN)
       call chmdealloc('pbeq.src','GSBP0','ALPX',XNPOL,crl=ALPX)
       call chmdealloc('pbeq.src','GSBP0','ALPY',YNPOL,crl=ALPY)
       call chmdealloc('pbeq.src','GSBP0','ALPZ',ZNPOL,crl=ALPZ)
       call chmdealloc('pbeq.src','GSBP0','ADLPX',XNPOL,crl=ADLPX)
       call chmdealloc('pbeq.src','GSBP0','ADLPY',YNPOL,crl=ADLPY)
       call chmdealloc('pbeq.src','GSBP0','ADLPZ',ZNPOL,crl=ADLPZ)
    ELSEIF(QSPHERE.OR.QRFSPHE) THEN
       call chmdealloc('pbeq.src','GSBP0','EMN',NTPOL,crl=EMN)
       call chmdealloc('pbeq.src','GSBP0','AR',NMPOL,crl=AR)
       call chmdealloc('pbeq.src','GSBP0','AC',NMPOL,crl=AC)
       call chmdealloc('pbeq.src','GSBP0','AS',NMPOL,crl=AS)
       call chmdealloc('pbeq.src','GSBP0','AP',NMPOL,NMPOL,crl=AP)
       call chmdealloc('pbeq.src','GSBP0','ADP',NMPOL,NMPOL,crl=ADP)
    ELSEIF(QPROLATE)THEN
       call chmdealloc('pbeq.src','GSBP0','EMN',NTPOL,crl=EMN)
    ENDIF
    !
    RETURN
  END SUBROUTINE GSBP0
  !
  SUBROUTINE PREP2
    !-----------------------------------------------------------------------
    !     INPUT PARAMETERS FOR GSBP
    !
    use chm_kinds
    use dimens_fcm
    use number
    use chutil
    use comand
    use psf
    use select
    use stream
    use string
    use coord
    use memory
    use ssbpm, only: qcavi,rmax,drmax2,acav,bcav,lstssbp,ntssbp
    use gcmc

#if KEY_SCCDFTB==1
    use sccdftb
    use sccgsbp
    use gamess_fcm, only: qmused_sccdftb
#endif 

    implicit none
    !-----------------------------------------------------------------
    ! Local variables
    real(chm_real),allocatable,dimension(:) :: OLDMIJ
    integer,allocatable,dimension(:) ::   OLDPX,OLDPY,OLDPZ,ISLCT
    INTEGER I,ERR
    real(chm_real)  QTOTA, QTOTB
    INTEGER OLDNTP,OLDBL,OLDBM
    !
    ! JZ_UW12: allocate GCMCON if not already!
#if KEY_GCMC==1
    if(.not.allocated(gcmcon)) call allocate_gcmc  
#endif
    !
    call chmalloc('pbeq.src','PREP2','ISLCT',NATOM,intg=ISLCT) 
    !
    !     Generalized Solvent Boundary Potential (GSBP)
    !     =============================================
    !     Region A = fixed region; Region B = region of interest

    QA=.FALSE.                ! for atom selection in region A
    QB=.FALSE.                ! for atom selection in region A
    !
    QGTOT = INDXA(COMLYN, COMLEN, 'GTOT').GT.0
    QGAA  = INDXA(COMLYN, COMLEN, 'G_OO').GT.0
    QGAB  = INDXA(COMLYN, COMLEN, 'G_IO').GT.0 .or. &
         INDXA(COMLYN, COMLEN, 'G_OI').GT.0
    QGBB  = INDXA(COMLYN, COMLEN, 'G_II').GT.0
    QNOSORT=INDXA(COMLYN, COMLEN, 'NOSO').GT.0
    ! basis charge scaling factor for the monopole basis function
    CGSCAL=GTRMF(COMLYN,COMLEN,'CGSC',ONE)
    ! get boundary potentials of basis charge distributions using
    ! a large box and the charge density on its grid  (it works with focussing)
    QLBOX =INDXA(COMLYN, COMLEN, 'LBOX') .GT. 0
    ! updating frequency for making an ordered list of basis functions
    ! during the MM calculations
    NLIST =GTRMI(COMLYN,COMLEN,'NLIS',1)
    ! choose one geometry
    QRECTBOX=INDXA(COMLYN, COMLEN, 'RECT').GT.0
    QSPHERE =INDXA(COMLYN, COMLEN, 'SPHE').GT.0
    QPROLATE=INDXA(COMLYN, COMLEN, 'PROL').GT.0
    QRFRECT =INDXA(COMLYN, COMLEN, 'RFRE').GT.0
    QRFSPHE =INDXA(COMLYN, COMLEN, 'RFSP').GT.0
    IF(.not.QRECTBOX.and..not.QSPHERE.and..not.QPROLATE.and. &
         .not.QRFRECT .and..not.QRFSPHE ) THEN
       WRITE(OUTU,'(/,3X,A)') 'PREP WARNING: choose one geometry'
       CALL WRNDIE(-1,'<PREP>','GSBP ERROR')
    ENDIF

    ! rectangular region of interest
    IF(QRECTBOX.or.QRFRECT) THEN
       RBXMIN = GTRMF(COMLYN,COMLEN,'XMIN',ZERO)
       RBXMAX = GTRMF(COMLYN,COMLEN,'XMAX',ZERO)
       RBYMIN = GTRMF(COMLYN,COMLEN,'YMIN',ZERO)
       RBYMAX = GTRMF(COMLYN,COMLEN,'YMAX',ZERO)
       RBZMIN = GTRMF(COMLYN,COMLEN,'ZMIN',ZERO)
       RBZMAX = GTRMF(COMLYN,COMLEN,'ZMAX',ZERO)
       IF(QRFRECT) THEN
          SRCUT  = GTRMF(COMLYN,COMLEN,'SRCU',ZERO)
          TOLSIJ = GTRMF(COMLYN,COMLEN,'TOLS',PT0001)
       ENDIF
       !     modify the region slightly to fit in grid
       IF(QPHIX.and.QMIJ) THEN
          NCLX=RNCLX
          NCLY=RNCLY
          NCLZ=RNCLZ
          DCEL=RDCEL
          XBCEN=RXBCEN
          YBCEN=RYBCEN
          ZBCEN=RZBCEN
          TRANX = HALF*(NCLX-1)*DCEL
          TRANY = HALF*(NCLY-1)*DCEL
          TRANZ = HALF*(NCLZ-1)*DCEL
       ENDIF
       RBXMIN = (INT((RBXMIN+TRANX)/DCEL)+HALF)*DCEL-TRANX
       RBXMAX = (INT((RBXMAX+TRANX)/DCEL)+HALF)*DCEL-TRANX
       RBYMIN = (INT((RBYMIN+TRANY)/DCEL)+HALF)*DCEL-TRANY
       RBYMAX = (INT((RBYMAX+TRANY)/DCEL)+HALF)*DCEL-TRANY
       RBZMIN = (INT((RBZMIN+TRANZ)/DCEL)+HALF)*DCEL-TRANZ
       RBZMAX = (INT((RBZMAX+TRANZ)/DCEL)+HALF)*DCEL-TRANZ
       RRXCEN = (RBXMAX+RBXMIN)/TWO
       RRYCEN = (RBYMAX+RBYMIN)/TWO
       RRZCEN = (RBZMAX+RBZMIN)/TWO
       XSCALE=TWO/(RBXMAX-RBXMIN)
       YSCALE=TWO/(RBYMAX-RBYMIN)
       ZSCALE=TWO/(RBZMAX-RBZMIN)
       !     number of Legendre Polynomials
       XNPOL = GTRMI(COMLYN,COMLEN,'XNPO',0)
       YNPOL = GTRMI(COMLYN,COMLEN,'YNPO',0)
       ZNPOL = GTRMI(COMLYN,COMLEN,'ZNPO',0)
       NTPOL = XNPOL*YNPOL*ZNPOL
       OLDNTP= OLDXNP*OLDYNP*OLDZNP
       IF(.not.QMIJ) THEN
          OLDXNP=0
          OLDYNP=0
          OLDZNP=0
          OLDNTP=0
       ENDIF
    ENDIF

    ! spherical region of interest
    IF(QSPHERE.or.QRFSPHE) THEN
       SRDIST = GTRMF(COMLYN,COMLEN,'SRDI',ZERO) ! a radius
       RRXCEN = GTRMF(COMLYN,COMLEN,'RRXC',ZERO) ! a sphere center
       RRYCEN = GTRMF(COMLYN,COMLEN,'RRYC',ZERO)
       RRZCEN = GTRMF(COMLYN,COMLEN,'RRZC',ZERO)
       IF(QRFSPHE) TOLSIJ = GTRMF(COMLYN,COMLEN,'TOLS',PT0001)
       !     number of multipoles
       NMPOL = GTRMI(COMLYN,COMLEN,'NMPO',0)
       NTPOL = NMPOL*NMPOL
       OLDNTP= OLDNMP*OLDNMP
       IF(.not.QMIJ) THEN
          OLDNMP=0
          OLDNTP=0
       ENDIF
    ENDIF

    ! prolate spheroidal region of interest
    IF(QPROLATE) THEN
       MAJOR  = GTRMF(COMLYN,COMLEN,'MAJO',ZERO) ! a major semi-axis
       MINOR  = GTRMF(COMLYN,COMLEN,'MINO',ZERO) ! a minor semi-axis
       RRXCEN = GTRMF(COMLYN,COMLEN,'RRXC',ZERO) ! a spheroidal center
       RRYCEN = GTRMF(COMLYN,COMLEN,'RRYC',ZERO)
       RRZCEN = GTRMF(COMLYN,COMLEN,'RRZC',ZERO)
       !     number of multipoles
       NMPOL = GTRMI(COMLYN,COMLEN,'NMPO',0)
       NTPOL = NMPOL*NMPOL
       OLDNTP= OLDNMP*OLDNMP
       IF(.not.QMIJ) THEN
          OLDNMP=0
          OLDNTP=0
       ENDIF
    ENDIF

    ! maximum number of basis functions for energy and force calculations
    ! (see subroutine GSBP3)
    MAXNPOL = GTRMI(COMLYN,COMLEN,'MAXN',NTPOL)

    ! rebuild MIJ(NTPOL,NTPOL) matrix using previous MIJ(OLDNTP,OLDNTP) matrix
    IF(QMIJ.AND.NTPOL.NE.OLDNTP) THEN
       call chmalloc('pbeq.src','PREP2','OLDMIJ',OLDNTP*OLDNTP,crl=OLDMIJ)
       CALL RPMIJ(OLDNTP,OLDMIJ,OLDNTP,MIJ)
       call chmdealloc('pbeq.src','PREP2','MIJ',OLDNTP*OLDNTP,crl=MIJ)
       call chmalloc('pbeq.src','PREP2','MIJ',NTPOL*NTPOL,crl=MIJ)
       call RPMIJ(NTPOL,MIJ,OLDNTP,OLDMIJ)
       call chmdealloc('pbeq.src','PREP2','OLDMIJ',OLDNTP*OLDNTP,crl=OLDMIJ)
       IF(QRECTBOX.or.QRFRECT) THEN
          call chmalloc('pbeq.src','PREP2','OLDPX',OLDNTP,intg=OLDPX)
          call chmalloc('pbeq.src','PREP2','OLDPY',OLDNTP,intg=OLDPY)
          call chmalloc('pbeq.src','PREP2','OLDPZ',OLDNTP,intg=OLDPZ)
          CALL RPPOL(OLDNTP,OLDPX,OLDNTP,LSTPX)
          CALL RPPOL(OLDNTP,OLDPY,OLDNTP,LSTPY)
          CALL RPPOL(OLDNTP,OLDPZ,OLDNTP,LSTPZ)
          call chmdealloc('pbeq.src','PREP2','LSTPX',OLDNTP,intg=LSTPX)
          call chmdealloc('pbeq.src','PREP2','LSTPY',OLDNTP,intg=LSTPY)
          call chmdealloc('pbeq.src','PREP2','LSTPZ',OLDNTP,intg=LSTPZ)
          call chmalloc('pbeq.src','PREP2','LSTPX',NTPOL,intg=LSTPX)
          call chmalloc('pbeq.src','PREP2','LSTPY',NTPOL,intg=LSTPY)
          call chmalloc('pbeq.src','PREP2','LSTPZ',NTPOL,intg=LSTPZ)
          CALL RPPOL(NTPOL,LSTPX,OLDNTP,OLDPX)
          CALL RPPOL(NTPOL,LSTPY,OLDNTP,OLDPY)
          CALL RPPOL(NTPOL,LSTPZ,OLDNTP,OLDPZ)
          call chmdealloc('pbeq.src','PREP2','OLDPX',OLDNTP,intg=OLDPX)
          call chmdealloc('pbeq.src','PREP2','OLDPY',OLDNTP,intg=OLDPY)
          call chmdealloc('pbeq.src','PREP2','OLDPZ',OLDNTP,intg=OLDPZ)
       ENDIF
    ENDIF

    ! allocate the grid and multipol parameters
    IF(.NOT.QPBEQ) THEN
       !     for EPSP
       IF(.not.QPHIX.or..not.QMIJ) call chmalloc('pbeq.src','PREP2','IPHIP',NCEL3,cr4=IPHIP)
       !     for static external field
       IF(.not.QPHIX.and.QGAB) call chmalloc('pbeq.src','PREP2','IPHIX',NCEL3,cr4=IPHIX)
       call chmalloc('pbeq.src','PREP2','LSTRA',NATOM,intg=LSTRA)
       call chmalloc('pbeq.src','PREP2','LSTRB',NATOM,intg=LSTRB)
       call chmalloc('pbeq.src','PREP2','RXNAFX',NATOM,crl=RXNAFX)
       call chmalloc('pbeq.src','PREP2','RXNAFY',NATOM,crl=RXNAFY)
       call chmalloc('pbeq.src','PREP2','RXNAFZ',NATOM,crl=RXNAFZ)
       call chmalloc('pbeq.src','PREP2','RXNBFX',NATOM,crl=RXNBFX)
       call chmalloc('pbeq.src','PREP2','RXNBFY',NATOM,crl=RXNBFY)
       call chmalloc('pbeq.src','PREP2','RXNBFZ',NATOM,crl=RXNBFZ)
       call chmalloc('pbeq.src','PREP2','BNORM',NTPOL,crl=BNORM)
       call chmalloc('pbeq.src','PREP2','COEF',NTPOL,crl=COEF)  ! coefficients of basis functions

#if KEY_SCCDFTB==1
       if (qmused_sccdftb) & ! coeffs of basis funcs for excluded MM
         call chmalloc('pbeq.src','PREP2','COEFX',NTPOL,crl=COEFX)

       !  Haibo Yu more memory for replica
       if((.not. qmused_sccdftb) .or. nsccrp == 0) nsccrp=1
       call chmalloc('pbeq.src','PREP2','MQ',NTPOL*NSCCRP,crl=MQ)  ! M(I,J)*Q(J)
#else
       call chmalloc('pbeq.src','PREP2','MQ',NTPOL,crl=MQ)   ! M(I,J)*Q(J)
#endif 

       call chmalloc('pbeq.src','PREP2','LSTPOL',NTPOL,intg=LSTPOL)  ! a list for basis functions
       IF(.not.QMIJ) call chmalloc('pbeq.src','PREP2','MIJ',NTPOL*NTPOL,crl=MIJ)
       IF(.not.QMIJ.and.QRECTBOX) THEN
          call chmalloc('pbeq.src','PREP2','LSTPX',NTPOL,intg=LSTPX)
          call chmalloc('pbeq.src','PREP2','LSTPY',NTPOL,intg=LSTPY)
          call chmalloc('pbeq.src','PREP2','LSTPZ',NTPOL,intg=LSTPZ)
       ELSEIF(QSPHERE) THEN
          call chmalloc('pbeq.src','PREP2','LSTPL',NTPOL,intg=LSTPL)
          call chmalloc('pbeq.src','PREP2','LSTPM',NTPOL,intg=LSTPM)
       ELSEIF(QPROLATE) THEN
          call chmalloc('pbeq.src','PREP2','LSTPL',NTPOL,intg=LSTPL)
          call chmalloc('pbeq.src','PREP2','LSTPM',NTPOL,intg=LSTPM)
          IF(.not.QMMIJ) &
               call chmalloc('pbeq.src','PREP2','MMIJ',NTPOL*NTPOL,crl=MMIJ) 
          !B^(-1) M B^(-1)
       ELSEIF(QRFRECT) THEN
          IF(.not.QMIJ) THEN
             call chmalloc('pbeq.src','PREP2','LSTPX',NTPOL,intg=LSTPX)
             call chmalloc('pbeq.src','PREP2','LSTPY',NTPOL,intg=LSTPY)
             call chmalloc('pbeq.src','PREP2','LSTPZ',NTPOL,intg=LSTPZ)
          ENDIF
          IF(.not.QMMIJ)  &
               call chmalloc('pbeq.src','PREP2','MMIJ',NTPOL*NTPOL,crl=MMIJ)
          !B^(-1) M B^(-1)
       ELSEIF(QRFSPHE) THEN
          call chmalloc('pbeq.src','PREP2','LSTPL',NTPOL,intg=LSTPL)
          call chmalloc('pbeq.src','PREP2','LSTPM',NTPOL,intg=LSTPM)
          IF(.not.QMMIJ) &
               call chmalloc('pbeq.src','PREP2','MMIJ',NTPOL*NTPOL,crl=MMIJ)
          !B^(-1) M B^(-1)
       ENDIF
    ENDIF

    ! change grid parameters if QPHIX is true.
    IF(QPHIX) THEN
       NCLX=RNCLX
       NCLY=RNCLY
       NCLZ=RNCLZ
       DCEL=RDCEL
       XBCEN=RXBCEN
       YBCEN=RYBCEN
       ZBCEN=RZBCEN
       TRANX = HALF*(NCLX-1)*DCEL
       TRANY = HALF*(NCLY-1)*DCEL
       TRANZ = HALF*(NCLZ-1)*DCEL
    ENDIF

    ! make lists of atoms in the fixed region (A) and the region of interest (B)
    IF(QRECTBOX) THEN
       CALL STRECTBOX(NTPRP,LSTPRP,X,Y,Z,CG, &
            NTRA,LSTRA,QTOTA,NTRB,LSTRB,QTOTB, &
            RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX &
#if KEY_GCMC==1
            ,GCMCON  & 
#endif
            )
    ELSEIF(QSPHERE) THEN
       CALL STSPHERE(NTPRP,LSTPRP,X,Y,Z,CG, &
            NTRA,LSTRA,QTOTA,NTRB,LSTRB,QTOTB, &
            SRDIST,RRXCEN,RRYCEN,RRZCEN &
#if KEY_GCMC==1
            ,GCMCON  & 
#endif
            )
    ELSEIF(QPROLATE) THEN
       CALL STPROLATE(NTPRP,LSTPRP,X,Y,Z,CG, &
            NTRA,LSTRA,QTOTA,NTRB,LSTRB,QTOTB, &
            MAJOR,MINOR,RRXCEN,RRYCEN,RRZCEN)
    ELSEIF(QRFRECT) THEN
       WRITE(OUTU,101)
       CALL SELCTA(COMLYN,COMLEN,ISLCT,X,Y,Z,WMAIN,.TRUE.)
       CALL MAKIND(NATOM,ISLCT,LSTRA,NTRA)
       QTOTA=ZERO
       DO I=1,NATOM
          IF(ISLCT(I).GT.0) QTOTA=QTOTA+CG(I)
       ENDDO
       CALL SELCTA(COMLYN,COMLEN,ISLCT,X,Y,Z,WMAIN,.TRUE.)
       CALL MAKIND(NATOM,ISLCT,LSTRB,NTRB)
       QTOTB=ZERO
       DO I=1,NATOM
          IF(ISLCT(I).GT.0) QTOTB=QTOTB+CG(I)
       ENDDO
    ELSEIF(QRFSPHE) THEN
       WRITE(OUTU,101)
       CALL SELCTA(COMLYN,COMLEN,ISLCT,X,Y,Z,WMAIN,.TRUE.)
       CALL MAKIND(NATOM,ISLCT,LSTRA,NTRA)
       QTOTA=ZERO
       DO I=1,NATOM
          IF(ISLCT(I).GT.0) QTOTA=QTOTA+CG(I)
       ENDDO
       CALL SELCTA(COMLYN,COMLEN,ISLCT,X,Y,Z,WMAIN,.TRUE.)
       CALL MAKIND(NATOM,ISLCT,LSTRB,NTRB)
       QTOTB=ZERO
       DO I=1,NATOM
          IF(ISLCT(I).GT.0) QTOTB=QTOTB+CG(I)
       ENDDO
    ENDIF
    ! select the atoms for printing forces
    ! in this case, the first selection for PBEQ solver should be issued
    WRITE(OUTU,101)
    WRITE(OUTU,101) &
         '======================================================='
    WRITE(OUTU,101) &
         '==== Generalized Solvent Boundary Potential (GSBP) ===='
    WRITE(OUTU,101) &
         '======================================================='
    WRITE(OUTU,101)
    IF(NTRB.EQ.0) THEN
       WRITE(OUTU,'(3X,2A)') &
            'PREP WARNING: There are no atoms inside region of interest'
       CALL WRNDIE(1,'<PREP>','GSBP WARNING')
    ENDIF
    WRITE(OUTU,'(3x,2A)') &
         'Summary of two regions : ','Outer [O] and Inner [I]'
    WRITE(OUTU,'(3X,A,I6,5X,A,F10.5)') &
         'Number of atoms [O] : ',NTRA,'Total charge [O] :',QTOTA
    WRITE(OUTU,'(3X,A,I6,5X,A,F10.5)') &
         'Number of atoms [I] : ',NTRB,'Total charge [I] :',QTOTB
    WRITE(OUTU,101)
    IF(QRECTBOX.or.QRFRECT) THEN
       IF(SRCUT.GT.0.0.and.QRFRECT) THEN
          WRITE(OUTU,'(3x,a,/,3x,a,f8.3,a,/,3x,a,3f8.3)') &
               'Spherical inner region : ', &
               ' radius                     (SRDIST) =',SRCUT,' [Angs]', &
               ' center       (RRXCEN,RRYCEN,RRZCEN) =',RRXCEN,RRYCEN,RRZCEN
       ELSE
          WRITE(OUTU,102) &
               'Inner region in X from ',RBXMIN,' to ',RBXMAX
          WRITE(OUTU,102) &
               'Inner region in Y from ',RBYMIN,' to ',RBYMAX
          WRITE(OUTU,102) &
               'Inner region in Z from ',RBZMIN,' to ',RBZMAX
       ENDIF
       WRITE(OUTU,101)
       IF(QMIJ) THEN
          IF(NTPOL.EQ.OLDNTP) THEN
             WRITE(OUTU,'(3x,a,i5,a)') &
                  'MIJ matrix was built using',OLDXNP*OLDYNP*OLDZNP, &
                  ' basis functions from'
             WRITE(OUTU,101) &
                  'Number of Legendre Polynomials in X (OLDXNP) = ',OLDXNP
             WRITE(OUTU,101) &
                  'Number of Legendre Polynomials in Y (OLDYNP) = ',OLDYNP
             WRITE(OUTU,101) &
                  'Number of Legendre Polynomials in Z (OLDZNP) = ',OLDZNP
          ELSE
             WRITE(OUTU,'(3x,2A)') &
                  'MIJ matrix will be rebuilt using the previous matrix ', &
                  'obtained from'
             WRITE(OUTU,101) &
                  'Number of Legendre Polynomials in X (OLDXNP) = ',OLDXNP
             WRITE(OUTU,101) &
                  'Number of Legendre Polynomials in Y (OLDYNP) = ',OLDYNP
             WRITE(OUTU,101) &
                  'Number of Legendre Polynomials in Z (OLDZNP) = ',OLDZNP
          ENDIF
       ELSE
          WRITE(OUTU,'(3x,a,i5,a)') &
               'MIJ matrix will be built using',NTPOL, &
               ' basis functions from'
          WRITE(OUTU,101) &
               'Number of Legendre Polynomials in X  (XNPOL) = ',XNPOL
          WRITE(OUTU,101) &
               'Number of Legendre Polynomials in Y  (YNPOL) = ',YNPOL
          WRITE(OUTU,101) &
               'Number of Legendre Polynomials in Z  (ZNPOL) = ',ZNPOL
       ENDIF
    ELSEIF(QSPHERE.or.QRFSPHE) THEN
       WRITE(OUTU,'(3x,a,/,3x,a,f8.3,a,/,3x,a,3f8.3)') &
            'Spherical inner region : ', &
            ' radius                     (SRDIST) =',SRDIST,' [Angs]', &
            ' center       (RRXCEN,RRYCEN,RRZCEN) =',RRXCEN,RRYCEN,RRZCEN
       WRITE(OUTU,101)
       IF(QMIJ) THEN
          IF(NTPOL.EQ.OLDNTP) THEN
             WRITE(OUTU,'(3x,A,2(i4,a))') &
                  'MIJ matrix was built using',OLDNMP*OLDNMP, &
                  ' basis functions of',OLDNMP,' mutipoles'
          ELSE
             WRITE(OUTU,'(3x,2A,/3x,2(i4,a))') &
                  'MIJ matrix will be rebuilt using the previous matrix ', &
                  'obtained from',OLDNMP*OLDNMP, &
                  ' basis functions of',OLDNMP,' mutipoles'
          ENDIF
       ELSE
          WRITE(OUTU,'(3x,a,2(i4,a))') &
               'MIJ matrix will be built using', &
               NTPOL,' basis functions of',NMPOL,' mutipoles'
       ENDIF
    ELSEIF(QPROLATE) THEN
       WRITE(OUTU,'(3x,a,/,2(3x,a,f8.3,a,/),3x,a,3f8.3)') &
            'Prolate spheroidal inner region : ', &
            ' major semi-axis on Z    (MAJOR) =',MAJOR,' [Angs]', &
            ' minor semi-axes on XY   (MINOR) =',MINOR,' [Angs]', &
            ' center   (RRXCEN,RRYCEN,RRZCEN) =',RRXCEN,RRYCEN,RRZCEN
       WRITE(OUTU,101)
       IF(QMIJ) THEN
          IF(NTPOL.EQ.OLDNTP) THEN
             WRITE(OUTU,'(3x,A,2(i4,a))') &
                  'MIJ matrix was built using',OLDNMP*OLDNMP, &
                  ' basis functions of',OLDNMP,' mutipoles'
          ELSE
             WRITE(OUTU,'(3x,2A,/3x,2(i4,a))') &
                  'MIJ matrix will be rebuilt using the previous matrix ', &
                  'obtained from',OLDNMP*OLDNMP, &
                  ' basis functions of',OLDNMP,' mutipoles'
          ENDIF
       ELSE
          WRITE(OUTU,'(3x,a,2(i4,a))') &
               'MIJ matrix will be built using', &
               NTPOL,' basis functions of',NMPOL,' mutipoles'
       ENDIF
    ENDIF
    !
    IF(CGSCAL.GT.ONE.and..not.QMIJ) &
         WRITE(OUTU,'(/,3x,A,F7.3,A)') 'Charge scaling factor', &
         CGSCAL,' will be used for the monopole basis function'
    WRITE(OUTU,'(/,3x,A,i4,A)') &
         'For energy and force calculations, ',MAXNPOL, &
         ' basis functions will be used'
    IF(.not.QNOSORT) THEN
       WRITE(OUTU,'(/,3x,A,I3,A,/,3x,A)') &
            'The ordered list of basis functions will be updated every', &
            NLIST,' step ','during the minimization or dynamics'
    ENDIF
    !
    !     Non-polar cavity potential setup
    QCAVI  = INDXA(COMLYN,COMLEN,'CAVI') .GT. 0
    IF(QCAVI) THEN
       WRITE(OUTU,'(/,3x,A)')  &
            'Non-polar cavity potential setup'
       RMAX = 2.8
       RMAX = SRDIST - GTRMF(COMLYN,COMLEN,'DRDI',RMAX)
       !where the cavity potential for RECT geometry should sit
       DRMAX2 = 3.5
       DRMAX2 = GTRMF(COMLYN,COMLEN,'DRCA',DRMAX2)
       !       Parameters for the cavity potential
       ACAV(1)= 0.56198800D0          
       ACAV(2)=-0.072798148D0
       ACAV(3)= 0.00426122036D0
       ACAV(5)=-1.6649500D0 + 8.391D0   ! fix the depth of the cavity potential for GCMC
       ACAV(4)=-0.0000925233817D0
       ACAV(6)= 0.0840D0
       ACAV(7)=15.39333D0
       ACAV(8)= 1.0D-12   !for the repulsive wall in potential
       ACAV(9)= 1.0D-04    !same
       BCAV(1)= 1.319978287D0
       BCAV(2)=-0.840953501D0
       BCAV(3)=-0.001602388122D0
       BCAV(4)=-8.392886499D0
       BCAV(5)=BCAV(2)+BCAV(4)
       BCAV(6)= 1.6D0
       BCAV(7)=-8.4751210228D0

       IF(QSPHERE)THEN
          WRITE(OUTU,'(/,3x,A)')  &
               'Spherical Cavity Potential'
       ELSEIF (QRECTBOX.OR.QRFRECT) THEN
          WRITE(OUTU,'(/,3x,A)')  &
               'Quartic Planar Cavity Potential'
       ELSE
          CALL WRNDIE(-5,'<GSBP>', &
               'Cavity potential not supported for the geometry')
       ENDIF
       ! MAXA replaced with NATOM during F95 conversion, LNI. 
       !        call chmalloc('pbeq.src','PREP2','ISLGC',NATOM,intg=ISLGC)
       !        CALL SELCTA(COMLYN,COMLEN,ISLGC,X,Y,Z,WMAIN,.TRUE.)
       CALL SELCTA(COMLYN,COMLEN,ISLCT,X,Y,Z,WMAIN,.TRUE.)
       call chmalloc('pbeq.src','PREP2','LSTSSBP',NATOM,intg=LSTSSBP)
#if KEY_GCMC==1
       !        CALL MAKIND(NATOM,ISLGC,LSTSSBP,NTSSBP)  
#endif
#if KEY_GCMC==1
       CALL MAKIND(NATOM,ISLCT,LSTSSBP,NTSSBP)  
#endif
       IF(NTSSBP.EQ.0) then
          CALL WRNDIE(-5,'<GSBP>','invalid atom selection for cavity potential')
       ELSE
          WRITE(OUTU,'(3x,A,i4,A,/)') &
               'activated for ',NTSSBP,' water oxygen atoms'
       ENDIF

    ENDIF
    !
    QPBEQ=.TRUE.
    call chmdealloc('pbeq.src','PREP2','ISLCT',NATOM,intg=ISLCT) 
    !
101 FORMAT(3X,A,I6,A)
102 FORMAT(3X,A,F8.3,A,F8.3)
    !
    RETURN
  END SUBROUTINE PREP2
  !
#endif 

  SUBROUTINE PREP1
    !-----------------------------------------------------------------------
    use chm_kinds
    use dimens_fcm
    use number
    use chutil
    use comand
    use psf
    use select
    use stream
    use string
    use coord
    use memory
#if KEY_SCCDFTB==1
    use sccpb      
    use gamess_fcm, only: qmused_sccdftb
#endif

    implicit none

    !-----------------------------------------------------------------
    real(chm_real),allocatable,dimension(:) :: FISTR
    real(chm_real4),allocatable,dimension(:) :: IPHIB
    INTEGER, ALLOCATABLE,DIMENSION(:) :: ISLCT,LISTR
    ! Local variables
    INTEGER IMODE, I, NCEL, NN
    INTEGER FNCEX, FNCEY, FNCEZ, FNCEL, FNCEL3, FMAPT
    real(chm_real)  FDCEL, FXBCEN, FYBCEN, FZBCEN,  &
         FTRANX, FTRANY, FTRANZ
    real(chm_real)  ONEPT4

    !mf, unit numbers for external grids
    INTEGER EPSU,EPSG

#if KEY_SCCDFTB==1
    integer urad            
#endif
    SAVE
    !
    call chmalloc('pbeq.src','PREP1','ISLCT',NATOM,intg=ISLCT) 
    !
    !     Atom Selection
    !     ==============

    IF(.NOT.QPBEQ) THEN
       call chmalloc('pbeq.src','PREP1','PBRAD',NATOM,crl=PBRAD)
       call chmalloc('pbeq.src','PREP1','LSTPRP',NATOM,intg=LSTPRP)
       PBRAD(1:NATOM)=WMAIN(1:NATOM) !save PB radii
!!!LNI     call SVPBRAD(NATOM,WMAIN,PBRAD)      
    ENDIF

    IMODE=0 !implies default = all atoms selected
    call SELRPN(COMLYN,COMLEN,ISLCT,NATOM,1,IMODE,.FALSE.,1,' ',0,RESID, &
         RES,IBASE,SEGID,NICTOT,NSEG,.TRUE.,X,Y,Z,.TRUE.,1,PBRAD)
    IF(IMODE.NE.0)THEN
       if(prnlev.ge.2) write(outu,'(/,3x,a)') &
            'PREP WARNING: Not all atoms selected, is this what you want?'
       !        CALL WRNDIE(-1,'<PREP>','ATOM SELECTION PARSING ERROR')
    ENDIF
    call MAKIND(NATOM,ISLCT,LSTPRP,NTPRP)
    IF(PRNLEV.GE.2) WRITE(OUTU,101)
    IF(PRNLEV.GE.2) WRITE(OUTU,101) 'Calculation with ',NTPRP,' atoms'
101 FORMAT(3X,A,I6,A)


    !     PBEQ solver selection
    !     =====================
    !     Default is SOR (Successive OverRelaxation) method in subroutine PBEQ1
    !     Old PBEQ solver (used in c26a2) is available in subroutine OLDPBEQ1.
    !     FMG (Full MultiGrid) method is available in subroutine PBEQ2.
    !     Non-linear PBEQ solver (based on SOR method) is available in subroutine
    !     PBEQ3
    !     Partially linearized PBEQ solver (based on SOR method) is available
    !     in subroutine PBEQ4

    ! OLD PBEQ solver
    QOLDPB =INDXA(COMLYN, COMLEN, 'OLDP') .GT. 0
    ! FMG method
    QFMG= INDXA(COMLYN,COMLEN,'FMGR').GT.0
    IF(QFMG) THEN
       ! number of cycles
       NCYC   = GTRMI(COMLYN,COMLEN,'NCYC',100)
       ! number of relaxation sweeps during PRE-smoothing
       NPRE   = GTRMI(COMLYN,COMLEN,'NPRE',2)
       ! number of relaxation sweeps during POST-smoothing
       NPOST  = GTRMI(COMLYN,COMLEN,'NPOS',2)
    ENDIF
    ! OSOR (Optimized Successive OverRelaxation) method
    QOSOR = INDXA(COMLYN, COMLEN, 'OSOR') .GT. 0
    ! Non-linear PBEQ solver
    QNONLINEAR = INDXA(COMLYN, COMLEN, 'NONL') .GT. 0
    ! Partially linearized PBEQ solver
    QPARTLINEAR = INDXA(COMLYN, COMLEN, 'PART') .GT. 0
    ! Under-relaxation for non-linear PBEQ solver
    QUNDER = INDXA(COMLYN, COMLEN, 'UNDE') .GT. 0

    IF(PRNLEV.GE.2) WRITE(OUTU,102)
    IF(QNONLINEAR.OR.QPARTLINEAR) THEN
       IF(Qunder) THEN
          IF(Qnonlinear.and.PRNLEV.GE.2) WRITE(OUTU,105) &
               'NON-LINEAR PBEQ SOLVER: ', &
               'Successive UnderRelaxation with fixed Lambda'
          IF(Qpartlinear.and.PRNLEV.GE.2) WRITE(OUTU,105) &
               'PARTIALLY LINEARIZED PBEQ SOLVER: ', &
               'Successive UnderRelaxation with fixed Lambda'
       ELSE
          IF(Qnonlinear.and.PRNLEV.GE.2) WRITE(OUTU,105) &
               'NON-LINEAR PBEQ SOLVER: ', &
               'Successive OverRelaxation (SOR) method'
          IF(Qpartlinear.and.PRNLEV.GE.2) WRITE(OUTU,105) &
               'PARTIALLY LINEARIZED PBEQ SOLVER: ', &
               'Successive OverRelaxation (SOR) method'
       ENDIF
    ELSE
       IF(QOSOR) THEN
          IF(PRNLEV.GE.2) WRITE(OUTU,105) &
               'LINEARIZED PBEQ SOLVER: ', &
               'Optimized Successive OverRelaxation (SOR)'
       ELSE
          IF(QOLDPB) THEN
             IF(PRNLEV.GE.2) WRITE(OUTU,105) &
                  'LINEARIZED PBEQ SOLVER: ', &
                  'Old Successive OverRelaxation (SOR)method used till c26a2'
          ELSEIF(QFMG) THEN
             IF(PRNLEV.GE.2) WRITE(OUTU,105) &
                  'LINEARIZED PBEQ SOLVER: ', &
                  'Full MultiGrid (FMG) method'
          ELSE
             IF(PRNLEV.GE.2) WRITE(OUTU,105) &
                  'LINEARIZED PBEQ SOLVER: ', &
                  'Successive OverRelaxation (SOR) method'
          ENDIF
       ENDIF
    ENDIF
105 FORMAT(3X,2A)

    !
    !     Stuff for Iteration
    !     ===================

    ! number of iterations
    MAXITS = GTRMI(COMLYN,COMLEN,'MAXI',2000)
    ! parameter of convergence
    DEPS = GTRMF(COMLYN,COMLEN,'DEPS',0.000002*ONE)
    ! initial mixing factor
    DOME = GTRMF(COMLYN,COMLEN,'LAMB',ONE)
    DOME = GTRMF(COMLYN,COMLEN,'DOME',DOME)
    ! Use current potential for initial guess (from last cycle)
    QKEEP= INDXA(COMLYN,COMLEN,'KEEP').GT.0       ! KEEPPHI
    ! Consider solvent accessible surface
    QREEN= INDXA(COMLYN,COMLEN,'REEN').GT.0

    IF(PRNLEV.GE.2) WRITE(OUTU,102)
    IF(PRNLEV.GE.2) WRITE(OUTU,102) 'ITERATION PARAMETERs'
    IF(PRNLEV.GE.2) WRITE(OUTU,104) &
         'Maximum # iterations          (MAXITS) =',MAXITS
    IF(PRNLEV.GE.2) WRITE(OUTU,'(3x,a,e8.3)') &
         'Tolerance of convergence        (DEPS) =',DEPS
    IF(PRNLEV.GE.2) WRITE(OUTU,103) &
         'Mixing factor            (LAMBDA,DOME) =',DOME
    IF(QKEEP) THEN
       IF(PRNLEV.GE.2) WRITE(OUTU,'(3x,A)') &
            'Previous potentials will be used for the initial guess'
    ENDIF

    !
    !     Charge Distribution Method
    !     ==========================
    !     The default is the trilinear method.
    !     The Cardinal B-spline method is invoked by BSPLINE keyword.

    QBSPL = INDXA(COMLYN, COMLEN, 'BSPL').GT.0
    IF(PRNLEV.GE.2) WRITE(OUTU,102)
    IF(QBSPL) THEN
       IF(PRNLEV.GE.2) WRITE(OUTU,102) &
            'CHARGE DISTRIBUTION METHOD: the Cardinal B-spline interpolation'
    ELSE
       IF(PRNLEV.GE.2) WRITE(OUTU,102) &
            'CHARGE DISTRIBUTION METHOD: the trilinear interpolation'
    ENDIF

    !
    !     Boundary Potentials Setting
    !     ============================
    !     The default is the use of potentials at boundary points due to
    !     the charges of the solute and the salt concentration using
    !     the Debye-Huckel approximation

    ! periodic boundary condition (3D) will be used for boundary potentials
    QPBC  = INDXA(COMLYN, COMLEN, 'PBC').GT.0
    ! periodic boundary condition will be surpressed in calculations with membrane
    QNPBC = INDXA(COMLYN, COMLEN, 'NPBC').GT.0
    ! use the image atoms in XY plane for boundary potentials
    NIMGB = GTRMI(COMLYN,COMLEN,'NIMG',0)
    ! zero boundary potentials
    QZERO =INDXA(COMLYN, COMLEN, 'ZERO') .GT. 0
    ! using previous potentials
    ! JZ_UW12: Modified for SMBP
    IF(.not.QGSBP .and. .not.QSMBP) QFOCUS=INDXA(COMLYN, COMLEN, 'FOCU') .GT. 0
    ! using bilinear interpolation from the Debye-Huckel approximation
    QINTBP=INDXA(COMLYN, COMLEN, 'INTB') .GT. 0

    IF(PRNLEV.GE.2) WRITE(OUTU,102)
    IF(PRNLEV.GE.2) WRITE(OUTU,102)  &
         'BOUNDARY POTENTIAL CALCULATION METHOD'
    IF(QPBC.and.prnlev.ge.2) WRITE(OUTU,102) &
         'Periodic boundary conditions in XYZ'
    IF(QNPBC.and.prnlev.ge.2) WRITE(OUTU,102) &
         'Periodic boundary conditions with membrane will be surpressed'
    IF(NIMGB.GT.0.and.prnlev.ge.2) WRITE(OUTU,'(3x,I5,A)') &
         (2*NIMGB+1)*(2*NIMGB+1)-1, &
         ' image cells in XY plane will be used for boundary potential'
    IF(QZERO.and.prnlev.ge.2) WRITE(OUTU,102) &
         'Zero potentials will be used at all boundary points'
    IF(QINTBP.and.prnlev.ge.2) WRITE(OUTU,'(3x,2A,/,3x,2A)') &
         'The Debye-Huckel approximation for ', &
         'half number of boundary points along 1d-axis and', &
         'potential of the rest will be interpolated from ', &
         'nearest grid points'
    IF(QFOCUS.and.prnlev.ge.2) WRITE(OUTU,102) &
         'Focussing method will be used for the boundary potentials'
    IF(.not.(QZERO.or.QINTBP.or.QFOCUS) .and. prnlev >= 2) WRITE(OUTU,102) &
         'The Debye-Huckel approximation for all boundary points'

    IF(.not.QPBEQ.AND.QFOCUS) THEN
       IF(PRNLEV.GE.2) WRITE(OUTU,102)
       IF(PRNLEV.GE.2) WRITE(OUTU,102) &
            'PREP WARNING: There is no previous run.'
       CALL WRNDIE(-5,'<PREP>','FOCUSSING SELECTION ERROR')
    ELSEIF((QPBC.and.QZERO).or.(QPBC.and.QINTBP).or. &
         (QPBC.and.QFOCUS).or.(QZERO.and.QINTBP).or. &
         (QZERO.and.QFOCUS).or.(QINTBP.and.QFOCUS)) THEN
       IF(PRNLEV.GE.2) WRITE(OUTU,102)
       IF(PRNLEV.GE.2) WRITE(OUTU,102) &
            'PREP WARNING: Choose just one boundary condition'
       CALL WRNDIE(-5,'<PREP>', &
            'BOUNDARY POTENTIAL METHOD SELECTION ERROR')
    ENDIF

    IF(.NOT.QFOCUS.AND..NOT.QZERO.AND..NOT.QINTBP.AND..NOT.QPBC)THEN
       IF(NTPRP.GT.100.and.prnlev.ge.2) WRITE(OUTU,102) &
            'PREP WARNING: INTBP is the faster method for boundary potential'
    ENDIF

    !
    !     Get value of the physical parameters
    !     ====================================

    IF(PRNLEV.GE.2) WRITE(OUTU,103)
    IF(PRNLEV.GE.2) WRITE(OUTU,102) 'PHYSICAL PARAMETERs'
    IF(QREEN)THEN
       ONEPT4 = 1.4
       WATR = GTRMF(COMLYN,COMLEN,'WATR',ONEPT4)
    ELSE
       WATR = GTRMF(COMLYN,COMLEN,'WATR',ZERO)
    ENDIF

    IF(PRNLEV.GE.2) WRITE(OUTU,103) &
         'Solvent probe radius          (WATR)   = ',WATR, ' [Angs]'
103 FORMAT(3X,A,F8.3,A,2X)
    IF( WATR.GT.0.0 .and. QREEN ) THEN
       IF(PRNLEV.GE.2) WRITE(OUTU,103) &
            'Calculation with reentrant (1.4 angs probe recommended)'
    ENDIF

    IONR = GTRMF(COMLYN,COMLEN,'IONR',ZERO)
    IF(PRNLEV.GE.2) WRITE(OUTU,103) &
         'Ion exclusion radius (Stern layer)     = ',IONR, ' [Angs]'

    EPSW = GTRMF(COMLYN,COMLEN,'EPSW',EIGHTY)
    IF(PRNLEV.GE.2) WRITE(OUTU,103) &
         'Solvent dielectric constant   (EPSW)   = ',EPSW

    EPSP = GTRMF(COMLYN,COMLEN,'EPSP',ONE)
    IF(PRNLEV.GE.2) WRITE(OUTU,103) &
         'Protein dielectric constant   (EPSP)   = ',EPSP

    CONC = GTRMF(COMLYN,COMLEN,'CONC',ZERO)
    IF(PRNLEV.GE.2) WRITE(OUTU,103) &
         'Salt concentration            (CONC)   = ', &
         CONC, ' [moles]/[liter]'

    TEMP = GTRMF(COMLYN,COMLEN,'TEMP',THRHUN)
    IF(PRNLEV.GE.2) WRITE(OUTU,103) &
         'Temperature                   (TEMP)   = ',TEMP,' [K]'

    !
    !     cat=avogadro*conc/10**27=6.02217*10**23*conc/10**27
    !     temperature 1/kBT =  beta= 167182.1551/temp in Angs/e**2
    !     beta=1/kT=167182.1551/temp in Angs/e**2
    !     kappa2 is EPSW*k**2 in 1/(Angs)**2
    !     kappa2 = 8*PI*cat*e**2*beta
    !

    !wi      KAPPA=SQRT(KAPPA2/EPSW)
    KAPPA2=2530.362733*CONC/TEMP

    IF(PRNLEV.GE.2) WRITE(OUTU,103) &
         'Debye-Huckel factor           (KAPPA2) = ', &
         KAPPA2/EPSW,' [1/Angs**2]'
    IF(KAPPA2.NE.0.0)THEN
       IF(PRNLEV.GE.2) WRITE(OUTU,103) &
            'Debye screening length                 = ', &
            SQRT(EPSW/KAPPA2),' [Angs]'
    ENDIF

    !
    !     Stuff for Membrane
    !     ==================

    !     thickness of membrane (TMEMB=0 means that no membrane is created)
    TMEMB = GTRMF(COMLYN,COMLEN,'TMEM',ZERO)
    TMEMB = GTRMF(COMLYN,COMLEN,'MEMB',TMEMB)

    !     thickness of headgroup region
    HTMEMB = GTRMF(COMLYN,COMLEN,'HTME',ZERO)

    !     membrane position (ZMEMB=0 means that the membrane is centered)
    ZMEMB = GTRMF(COMLYN,COMLEN,'ZMEM',ZERO)

    !     membrane dielectric constant
    EPSM = GTRMF(COMLYN,COMLEN,'EPSM',ONE)

    !     membrane headgroup dielectric constant
    EPSH = GTRMF(COMLYN,COMLEN,'EPSH',EPSM)

    !     potential difference across membrane (entered in [volts])
    VMEMB = GTRMF(COMLYN,COMLEN,'VMEM',ZERO)

    !     is there a charge distribution in the head group region?
    IF(INDXA(COMLYN, COMLEN, 'HCHA') .GT. 0) QCHRG = .TRUE.

    !     what is the spread of the negative/positive charge throughout the
    !     head group regions?
    NEGALP = GTRMF(COMLYN,COMLEN,'NEGA',ZERO)
    POSALP = GTRMF(COMLYN,COMLEN,'POSI',ZERO)

    !     and how are they offset from the center of the headgroup region
    NOFFST = GTRMF(COMLYN,COMLEN,'NOFF',ZERO)
    POFFST = GTRMF(COMLYN,COMLEN,'POFF',ZERO)

    !     how many headgroups per unit area (in A^2)
    HDENS = GTRMF(COMLYN,COMLEN,'DENS',ZERO)

    IF(TMEMB.GT.ZERO)THEN
       IF(PRNLEV.GE.2) WRITE(OUTU,103)
       IF(PRNLEV.GE.2) WRITE(OUTU,103) 'MEMBRANE'
       IF(PRNLEV.GE.2) WRITE(OUTU,103) &
            'Membrane thickness along Z    (TMEMB)  = ',TMEMB,' [Angs]'
       IF(PRNLEV.GE.2) WRITE(OUTU,103) &
            'Membrane position along Z     (ZMEMB)  = ',ZMEMB,' [Angs]'
       IF(PRNLEV.GE.2) WRITE(OUTU,103) &
            'Membrane dielectric constant  (EPSM)   = ',EPSM
       IF(PRNLEV.GE.2) WRITE(OUTU,'(3X,2(A,F8.3),A,2X)') &
            'The membrane goes from                 = ', &
            ZMEMB-TMEMB/2,' to ',ZMEMB+TMEMB/2,' [Angs]'
       if ( HTMEMB .gt. 0.0 ) then
          IF(PRNLEV.GE.2) WRITE(OUTU,103) &
               'Membrane headgroup thickness  (HTMEMB) = ',HTMEMB,' [Angs]'
          IF(PRNLEV.GE.2) WRITE(OUTU,103) &
               'Headgroup dielectric constant (EPSH)   = ',EPSH
          if (QCHRG) then
             IF(PRNLEV.GE.2) WRITE(OUTU,'(A)') &
                  'Model will include gaussian distribution of headgroup charges'
             IF(PRNLEV.GE.2) WRITE(OUTU,103) &
                  'Surface Density of headgroups (A^2/headgroup)   = ', &
                  HDENS
             HDENS = DCEL*DCEL/HDENS
             IF(PRNLEV.GE.2) WRITE(OUTU,103) &
                  'Negative/positive charge density in headgroup region  = ', &
                  HDENS
             IF(PRNLEV.GE.2) WRITE(OUTU,103) &
                  'Gaussian width of distribution for negative charge   = ', &
                  NEGALP
             IF(PRNLEV.GE.2) WRITE(OUTU,103) &
                  'Offset of distribution for negative charge   = ', &
                  NOFFST
             IF(PRNLEV.GE.2) WRITE(OUTU,103) &
                  'Gaussian width of distribution for positive charge  = ', &
                  POSALP
             IF(PRNLEV.GE.2) WRITE(OUTU,103) &
                  'Offset of distribution for positive charge   = ', &
                  POFFST
             POSALP = ONE/POSALP
             NEGALP = ONE/NEGALP
          endif
       endif
       IF(PRNLEV.GE.2) WRITE(OUTU,103) &
            'Transmembrane voltage along Z (VMEMB)  = ',VMEMB,' [Volts]'
    ENDIF
    !     convert to [unit charge]/[Angstroms] for the PBEQ1 subroutine
    VMEMB=VMEMB/14.40145905

    !
    !     Stuff for Droplet
    !     =================

    !     radius of spherical droplet
    DROPLET  = GTRMF(COMLYN,COMLEN,'DROP',ZERO)
    XDROPLET = GTRMF(COMLYN,COMLEN,'XDRO',ZERO)
    YDROPLET = GTRMF(COMLYN,COMLEN,'YDRO',ZERO)
    ZDROPLET = GTRMF(COMLYN,COMLEN,'ZDRO',ZERO)

    !     dielectric constant of spherical droplet
    EPSD = GTRMF(COMLYN,COMLEN,'EPSD',ONE)

    !     Do you want that the dielectric constant of the overlapped region
    !     with membrane is set to EPSM ?
    QDTOM = INDXA(COMLYN, COMLEN, 'DTOM') .GT. 0

    !     Do you want that the Debye-Huckel factor inside the droplet
    !     is set to KAPPA2 ?
    QDKAP = INDXA(COMLYN, COMLEN, 'DKAP') .GT. 0

    IF(DROPLET.GT.ZERO)THEN
       IF(PRNLEV.GE.2) WRITE(OUTU,103)
       IF(PRNLEV.GE.2) WRITE(OUTU,103) 'SPHERE'
       IF(PRNLEV.GE.2) WRITE(OUTU,103) &
            'Spherical droplet radius      (DROPLET)= ',DROPLET,' [Angs]'
       IF(PRNLEV.GE.2) WRITE(OUTU,103) &
            'Droplet dielectric constant   (EPSD)   = ',EPSD
       IF(PRNLEV.GE.2) WRITE(OUTU,'(3x,a,3f8.3)') &
            'Droplet center                         = ', &
            XDROPLET,YDROPLET,ZDROPLET
       IF(QDTOM.and.prnlev.ge.2) WRITE(OUTU,103) &
            'EPSD will be set to EPSM in membrane region'
       IF(QDKAP.and.prnlev.ge.2) WRITE(OUTU,103) &
            'Debye-Huckel factor inside the droplet is set to KAPPA2'
    ENDIF

    !
    !     Stuff for Orthorhombic BOX
    !     ==========================

    !     X, Y, and Z-dirctional dimensions and center of an orthorhombic box
    LXMAX = GTRMF(COMLYN,COMLEN,'LXMA',ZERO)
    LYMAX = GTRMF(COMLYN,COMLEN,'LYMA',ZERO)
    LZMAX = GTRMF(COMLYN,COMLEN,'LZMA',ZERO)
    LXMIN = GTRMF(COMLYN,COMLEN,'LXMI',ZERO)
    LYMIN = GTRMF(COMLYN,COMLEN,'LYMI',ZERO)
    LZMIN = GTRMF(COMLYN,COMLEN,'LZMI',ZERO)
    !     dielectric constant of box
    EPSB = GTRMF(COMLYN,COMLEN,'EPSB',ONE)

    !     Do you want that the dielectric constant of the overlapped region
    !     with membrane is set to EPSM ?
    QBTOM = INDXA(COMLYN, COMLEN, 'BTOM') .GT. 0

    !     Do you want that the Debye-Huckel factor inside the box
    !     is set to KAPPA2 ?
    QBKAP = INDXA(COMLYN, COMLEN, 'BKAP') .GT. 0

    IF(LXMAX.NE.LYMIN.and.prnlev.ge.2)THEN
       WRITE(OUTU,103)
       WRITE(OUTU,103) 'BOX'
       WRITE(OUTU,103) &
            'Box dielectric constant       (EPSB)   = ',EPSB
       WRITE(OUTU,102) &
            'Box in X from ',LXMIN,' to ',LXMAX
       WRITE(OUTU,102) &
            'Box in Y from ',LYMIN,' to ',LYMAX
       WRITE(OUTU,102) &
            'Box in Z from ',LZMIN,' to ',LZMAX
       IF(QBTOM) WRITE(OUTU,103) &
            'EPSB will be set to EPSM in membrane region'
       IF(QBKAP) WRITE(OUTU,103) &
            'Debye-Huckel factor inside the box is set to KAPPA2'
    ENDIF

    !
    !     Stuff for Cylinder
    !     ==================

    !     radius, height, and positions of cylinder
    RCYLN = GTRMF(COMLYN,COMLEN,'RCYL',ZERO)
    HCYLN = GTRMF(COMLYN,COMLEN,'HCYL',ZERO)
    XCYLN = GTRMF(COMLYN,COMLEN,'XCYL',ZERO)
    YCYLN = GTRMF(COMLYN,COMLEN,'YCYL',ZERO)
    ZCYLN = GTRMF(COMLYN,COMLEN,'ZCYL',ZERO)

    !     dielectric constant of cylinder
    EPSC = GTRMF(COMLYN,COMLEN,'EPSC',ONE)

    !     Do you want that the dielectric constant of the overlapped region
    !     with membrane is set to EPSM ?
    QCTOM = INDXA(COMLYN, COMLEN, 'CTOM') .GT. 0

    !     Do you want that the Debye-Huckel factor inside the cylinder
    !     is set to KAPPA2 ?
    QCKAP = INDXA(COMLYN, COMLEN, 'CKAP') .GT. 0

    IF(RCYLN.GT.ZERO.and.prnlev.ge.2)THEN
       WRITE(OUTU,103)
       WRITE(OUTU,103) 'CYLINDER'
       WRITE(OUTU,103) &
            'Cylinder height along Z       (HCYLN)  = ',HCYLN,' [Angs]'
       WRITE(OUTU,103) &
            'Cylinder radius in XY plane   (RCYLN)  = ',RCYLN,' [Angs]'
       WRITE(OUTU,103) &
            'Cylinder dielectric constant  (EPSC)   = ',EPSC
       WRITE(OUTU,'(3x,a,3f8.3)') &
            'Cylinder position                      = ', &
            XCYLN,YCYLN,ZCYLN
       IF(QCTOM) WRITE(OUTU,103) &
            'EPSC will be set to EPSM in membrane region'
       IF(QCKAP) WRITE(OUTU,103) &
            'Debye-Huckel factor inside the cylinder is set to KAPPA2'
    ENDIF

    !
    !     Stuff for Elliptical CONE
    !     =========================

    !     A (on X), B (on Y), C (on Z), and center of elliptical cone
    AX = GTRMF(COMLYN,COMLEN,'AX',ZERO)
    BY = GTRMF(COMLYN,COMLEN,'BY',ZERO)
    CZ = GTRMF(COMLYN,COMLEN,'CZ',ZERO)
    XCONE = GTRMF(COMLYN,COMLEN,'ZCON',ZERO)
    YCONE = GTRMF(COMLYN,COMLEN,'YCON',ZERO)
    ZCONE = GTRMF(COMLYN,COMLEN,'ZCON',ZERO)

    !     dielectric constant of cylinder
    EPSEC = GTRMF(COMLYN,COMLEN,'EPSE',ONE)

    !     Do you want that the dielectric constant of the overlapped region
    !     with membrane is set to EPSM ?
    QECTOM = INDXA(COMLYN, COMLEN, 'ECTO') .GT. 0

    !     Do you want that the Debye-Huckel factor inside the cylinder
    !     is set to KAPPA2 ?
    QECKAP = INDXA(COMLYN, COMLEN, 'ECKA') .GT. 0

    IF(AX.GT.ZERO.and.prnlev.ge.2)THEN
       WRITE(OUTU,103)
       WRITE(OUTU,103) 'ELLIPTICAL CONE'
       WRITE(OUTU,103) &
            'A on X-direction              (AX)     = ',AX,' [Angs]'
       WRITE(OUTU,103) &
            'B on Y-direction              (BY)     = ',BY,' [Angs]'
       WRITE(OUTU,103) &
            'C on Z-direction              (CZ)     = ',CZ,' [Angs]'
       WRITE(OUTU,103) &
            'Ellpitical cone epsilon       (EPSEC)  = ',EPSEC
       WRITE(OUTU,'(3x,a,3f8.3)') &
            'The center of elliptical cone          = ', &
            Xcone,Ycone,Zcone
       IF(QECTOM) WRITE(OUTU,103) &
            'EPSEC will be set to EPSM in membrane region'
       IF(QECKAP) WRITE(OUTU,103) &
            'Debye-Huckel factor inside the cone is set to KAPPA2'
    ENDIF

    !
    !     Set Up The Grid Parameters (if necessary)
    !     =========================================

    IF(.NOT.QPBEQ)THEN
       ! number of grid points
       NCEL  = GTRMI(COMLYN,COMLEN,'NCEL',5)     ! for cubic
       NCLX  = GTRMI(COMLYN,COMLEN,'NCLX',NCEL)   ! for non-cubic
       NCLY  = GTRMI(COMLYN,COMLEN,'NCLY',NCEL)
       NCLZ  = GTRMI(COMLYN,COMLEN,'NCLZ',NCEL)

       IF(QFMG) THEN
          IF((NCLx.ne.NCLy).or.(NCLy.ne.NCLz).or.(NCLz.ne.NCLx))THEN
             IF(PRNLEV.GE.2) WRITE(OUTU,'(3x,A,3I6)') &
                  'Number of grid points:',NCLX,NCLY,NCLZ
             CALL WRNDIE(-5,'<PREP>', &
                  'NUMBER OF GRID POINTS SELECTION ERRORS with FMG')
          ENDIF
          ! determine the maxinum grid number (ngridpb)
          ! 9 is a maxium variable (NG) in subroutine PBEQ2 (in pbeq2.src).
          DO I=1,9
             NN=2**I+1
             IF(NN.EQ.NCLX) THEN
                NGRIDpb=I
                GOTO 90
             ENDIF
          ENDDO
          IF(PRNLEV.GE.2) WRITE(OUTU,'(3x,A,3I6)') &
               'Number of grid points:',NCLX,NCLY,NCLZ
          CALL WRNDIE(-5,'<PREP>', &
               'NUMBER OF GRID POINTS SELECTION ERRORS with FMG')
90        CONTINUE
       ENDIF

       IF(MOD(NCLx,2).eq.0)THEN
          NCLx=NCLx+1
       ENDIF
       IF(MOD(NCLy,2).eq.0)THEN
          NCLy=NCLy+1
       ENDIF
       IF(MOD(NCLz,2).eq.0)THEN
          NCLz=NCLz+1
       ENDIF

       ! center of box
       XBCEN = GTRMF(COMLYN,COMLEN,'XBCE',ZERO)   ! the center of a box in X
       YBCEN = GTRMF(COMLYN,COMLEN,'YBCE',ZERO)
       ZBCEN = GTRMF(COMLYN,COMLEN,'ZBCE',ZERO)

       ! size of grid unit cell
       DCEL = GTRMF(COMLYN,COMLEN,'DCEL',PTONE)

       ! box size
       TRANX = HALF*(NCLX-1)*DCEL
       TRANY = HALF*(NCLY-1)*DCEL
       TRANZ = HALF*(NCLZ-1)*DCEL
       IF(PRNLEV.GE.2) WRITE(OUTU,102)
       IF(PRNLEV.GE.2) WRITE(OUTU,'(3x,A,3I6)') &
            'NUMBER OF GRID POINTS:',NCLX,NCLY,NCLZ
       IF(PRNLEV.GE.2) WRITE(OUTU,102) &
            'Box in X from ',XBCEN-TRANX,' to ',XBCEN+TRANX
       IF(PRNLEV.GE.2) WRITE(OUTU,102) &
            'Box in Y from ',YBCEN-TRANY,' to ',YBCEN+TRANY
       IF(PRNLEV.GE.2) WRITE(OUTU,102) &
            'Box in Z from ',ZBCEN-TRANZ,' to ',ZBCEN+TRANZ

       ! max number of points on the sphere surface
       MAPT   = GTRMI(COMLYN,COMLEN,'MAPT',10000)

       ! total number of grid points in 3d
       NCEL3=NCLX*NCLY*NCLZ

       ! JZ_UW12: For SMBP
       IF(QSMBP .and. QREADPHIX) THEN
          NCLXOG=RNCLX
          NCLYOG=RNCLY
          NCLZOG=RNCLZ
          NCEL3OG=RNCLX*RNCLY*RNCLZ
          DCELOG=RDCEL
          XBCENOG=RXBCEN
          YBCENOG=RYBCEN
          ZBCENOG=RZBCEN
          TRANXOG=HALF*(RNCLX-1)*RDCEL
          TRANYOG=HALF*(RNCLY-1)*RDCEL
          TRANZOG=HALF*(RNCLZ-1)*RDCEL
       ENDIF

       ! allocate the grid parameters
       call chmalloc('pbeq.src','PREP1','IPHIW',NCEL3,cr4=IPHIW)
       call chmalloc('pbeq.src','PREP1','ICHC',NCEL3,cr4=ICHC)
       IF(.not.QFKAP) call chmalloc('pbeq.src','PREP1','MAY',NCEL3,cr4=MAY)
       call chmalloc('pbeq.src','PREP1','MAYX',NCEL3,cr4=MAYX)
       call chmalloc('pbeq.src','PREP1','MAYY',NCEL3,cr4=MAYY)
       call chmalloc('pbeq.src','PREP1','MAYZ',NCEL3,cr4=MAYZ)
       call chmalloc('pbeq.src','PREP1','IPOX',MAPT,crl=IPOX)
       call chmalloc('pbeq.src','PREP1','IPOY',MAPT,crl=IPOY)
       call chmalloc('pbeq.src','PREP1','IPOZ',MAPT,crl=IPOZ)
       IF(QFKAP) THEN
          IF(PRNLEV.GE.2) WRITE(OUTU,102)
          IF(PRNLEV.GE.2) WRITE(OUTU,'(6x,2A)') &
               'Previous Debye screening factor map will be used ', &
               'for MAY array (no modification in MAYER)'
       ENDIF
    ENDIF
102 FORMAT(3X,A,F8.3,A,F8.3)
104 FORMAT(3X,A,I8)

    !
    !     New grid parameters (if necessary)
    !     ==================================
    !     To read new grid parameters from a input stream without RESET

    IF(QPBEQ) THEN
       FDCEL  = GTRMF(COMLYN,COMLEN,'DCEL',DCEL)
       FXBCEN = GTRMF(COMLYN,COMLEN,'XBCE',ZERO)
       FYBCEN = GTRMF(COMLYN,COMLEN,'YBCE',ZERO)
       FZBCEN = GTRMF(COMLYN,COMLEN,'ZBCE',ZERO)

       IF(INDX(COMLYN,COMLEN,'NCEL',4).GT.0  .OR. &
            INDX(COMLYN,COMLEN,'NCLX',4).GT.0) THEN
          FNCEL = GTRMI(COMLYN,COMLEN,'NCEL',NCEL)
          FNCEX = GTRMI(COMLYN,COMLEN,'NCLX',FNCEL)
          FNCEY = GTRMI(COMLYN,COMLEN,'NCLY',FNCEL)
          FNCEZ = GTRMI(COMLYN,COMLEN,'NCLZ',FNCEL)
          IF(MOD(FNCEX,2).eq.0)THEN
             FNCEX=FNCEX+1
          ENDIF
          IF(MOD(FNCEY,2).eq.0)THEN
             FNCEY=FNCEY+1
          ENDIF
          IF(MOD(FNCEZ,2).eq.0)THEN
             FNCEZ=FNCEZ+1
          ENDIF
       ELSE
          FNCEX = NCLX
          FNCEY = NCLY
          FNCEZ = NCLZ
       ENDIF

       !wi         IF(QFMG.and.(NCLX.NE.FNCEX)) THEN
       IF(QFMG) THEN
          IF((FNCEx.ne.FNCEy).or.(FNCEy.ne.FNCEz).or. &
               (FNCEz.ne.FNCEx))THEN
             IF(PRNLEV.GE.2) WRITE(OUTU,'(3x,A,3I6)') &
                  'Number of grid points:',FNCEX,FNCEY,FNCEZ
             CALL WRNDIE(-5,'<PREP>', &
                  'NUMBER OF GRID POINTS SELECTION ERRORS with FMG')
          ENDIF
          ! determine the maxinum grid number (ngridpb)
          ! 9 is a maxium variable (NG) in subroutine PBEQ2.
          DO I=1,9
             NN=2**I+1
             IF(NN.EQ.FNCEX) THEN
                NGRIDpb=I
                GOTO 91
             ENDIF
          ENDDO
          IF(PRNLEV.GE.2) WRITE(OUTU,'(3x,A,3I6)') &
               'Number of grid points:',FNCEX,FNCEY,FNCEZ
          CALL WRNDIE(-5,'<PREP>', &
               'NUMBER OF GRID POINTS SELECTION ERRORS with FMG')
91        CONTINUE
       ENDIF

       IF(INDX(COMLYN,COMLEN,'MAPT',4).GT.0) THEN
          FMAPT   = GTRMI(COMLYN,COMLEN,'MAPT',10000)
       ELSE
          FMAPT   = MAPT
       ENDIF

       IF(QFOCUS) THEN
          FTRANX=HALF*(FNCEX-1)*FDCEL
          FTRANY=HALF*(FNCEY-1)*FDCEL
          FTRANZ=HALF*(FNCEZ-1)*FDCEL
          IF(XBCEN-TRANX.GE.FXBCEN-FTRANX .OR. &
               XBCEN+TRANX.LE.FXBCEN+FTRANX .OR. &
               YBCEN-TRANY.GE.FYBCEN-FTRANY .OR. &
               YBCEN+TRANY.LE.FYBCEN+FTRANY .OR. &
               ZBCEN-TRANZ.GE.FZBCEN-FTRANZ .OR. &
               ZBCEN+TRANZ.LE.FZBCEN+FTRANZ) THEN
             IF(PRNLEV.GE.2) WRITE(OUTU,'(/,3X,2A)') &
                  'PREP WARNING: Focussed system should be ', &
                  'inside previous one.'
             CALL WRNDIE(-5,'<PREP>','FOCUSSING SELECTION ERROR')
          ENDIF
          FNCEL3 = 2*(FNCEX*FNCEY+FNCEY*FNCEZ+FNCEZ*FNCEX)
          call chmalloc('pbeq.src','PREP1','IPHIB',FNCEL3,cr4=IPHIB)
          ! construct boundary potentials
          CALL BOUNDPOT(NCLX,NCLY,NCLZ,DCEL, &
               TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
               FNCEX,FNCEY,FNCEZ,FDCEL,FXBCEN,FYBCEN,FZBCEN, &
               IPHIW,IPHIB)
       ENDIF

       ! reset and re-allocate the previous grid setup (if necessary)
       IF((NCLX.NE.FNCEX).OR.(NCLY.NE.FNCEY).OR.(NCLZ.NE.FNCEZ))THEN
          IF(PRNLEV.GE.2) WRITE(OUTU,'(/,3x,A)') &
               'Previous grid setup has been reset'
          call chmdealloc('pbeq.src','PREP1','IPHIW',NCEL3,cr4=IPHIW)
          call chmdealloc('pbeq.src','PREP1','ICHC',NCEL3,cr4=ICHC)
          call chmdealloc('pbeq.src','PREP1','MAY',NCEL3,cr4=MAY)
          call chmdealloc('pbeq.src','PREP1','MAYX',NCEL3,cr4=MAYX)
          call chmdealloc('pbeq.src','PREP1','MAYY',NCEL3,cr4=MAYY)
          call chmdealloc('pbeq.src','PREP1','MAYZ',NCEL3,cr4=MAYZ)
          NCEL3  = FNCEX*FNCEY*FNCEZ
          call chmalloc('pbeq.src','PREP1','IPHIW',NCEL3,cr4=IPHIW)
          call chmalloc('pbeq.src','PREP1','ICHC',NCEL3,cr4=ICHC)
          call chmalloc('pbeq.src','PREP1','MAY',NCEL3,cr4=MAY)
          call chmalloc('pbeq.src','PREP1','MAYX',NCEL3,cr4=MAYX)
          call chmalloc('pbeq.src','PREP1','MAYY',NCEL3,cr4=MAYY)
          call chmalloc('pbeq.src','PREP1','MAYZ',NCEL3,cr4=MAYZ)
       ENDIF
       IF(MAPT.NE.FMAPT)THEN
          call chmdealloc('pbeq.src','PREP1','IPOX',MAPT,crl=IPOX)
          call chmdealloc('pbeq.src','PREP1','IPOY',MAPT,crl=IPOY)
          call chmdealloc('pbeq.src','PREP1','IPOZ',MAPT,crl=IPOZ)
          call chmalloc('pbeq.src','PREP1','IPOX',FMAPT,crl=IPOX)
          call chmalloc('pbeq.src','PREP1','IPOY',FMAPT,crl=IPOY)
          call chmalloc('pbeq.src','PREP1','IPOZ',FMAPT,crl=IPOZ)
       ENDIF

       DCEL=FDCEL
       NCLX=FNCEX
       NCLY=FNCEY
       NCLZ=FNCEZ
       XBCEN=FXBCEN
       YBCEN=FYBCEN
       ZBCEN=FZBCEN
       TRANX = HALF*(NCLX-1)*DCEL
       TRANY = HALF*(NCLY-1)*DCEL
       TRANZ = HALF*(NCLZ-1)*DCEL
       IF(PRNLEV.GE.2) WRITE(OUTU,102)
       IF(PRNLEV.GE.2) WRITE(OUTU,'(3x,A,3I6)') &
            'NUMBER OF GRID POINTS:',NCLX,NCLY,NCLZ
       IF(PRNLEV.GE.2) WRITE(OUTU,102) &
            'Box in X from ',XBCEN-TRANX,' to ',XBCEN+TRANX
       IF(PRNLEV.GE.2) WRITE(OUTU,102) &
            'Box in Y from ',YBCEN-TRANY,' to ',YBCEN+TRANY
       IF(PRNLEV.GE.2) WRITE(OUTU,102) &
            'Box in Z from ',ZBCEN-TRANZ,' to ',ZBCEN+TRANZ
    ENDIF

    !
    !     Generalized Solvent Boundary Potential (GSBP) (see prep2 in gsbp.src)
    !     =============================================
    QA=.false.
    QB=.false.
    IF(QGSBP .or. QSMBP)THEN ! JZ_UW12: Modified for SMBP
       call chmdealloc('pbeq.src','PREP1','ISLCT',NATOM,intg=ISLCT)    
       RETURN
    ENDIF

    !
    !     Part for PB force, Dielectric Smoothing, and Nonpolar Surface Term
    !     ==================================================================

    QPBF   = INDXA(COMLYN,COMLEN,'FORC').GT.0
    IF(QPBF)THEN
       if (allocated(IPHIP)) then
          call chmrealloc('pbeq.src','PREP1','IPHIP',NCEL3,cr4=IPHIP)
          call chmrealloc('pbeq.src','PREP1','RXNFX',NATOM,crl=RXNFX)
          call chmrealloc('pbeq.src','PREP1','RXNFY',NATOM,crl=RXNFY)
          call chmrealloc('pbeq.src','PREP1','RXNFZ',NATOM,crl=RXNFZ)
          call chmrealloc('pbeq.src','PREP1','DBFX',NATOM,crl=DBFX)
          call chmrealloc('pbeq.src','PREP1','DBFY',NATOM,crl=DBFY)
          call chmrealloc('pbeq.src','PREP1','DBFZ',NATOM,crl=DBFZ)
          call chmrealloc('pbeq.src','PREP1','IBFX',NATOM,crl=IBFX)
          call chmrealloc('pbeq.src','PREP1','IBFY',NATOM,crl=IBFY)
          call chmrealloc('pbeq.src','PREP1','IBFZ',NATOM,crl=IBFZ)
          call chmrealloc('pbeq.src','PREP1','NPFX',NATOM,crl=NPFX)
          call chmrealloc('pbeq.src','PREP1','NPFY',NATOM,crl=NPFY)
          call chmrealloc('pbeq.src','PREP1','NPFZ',NATOM,crl=NPFZ)
       else
          call chmalloc('pbeq.src','PREP1','IPHIP',NCEL3,cr4=IPHIP)
          call chmalloc('pbeq.src','PREP1','RXNFX',NATOM,crl=RXNFX)
          call chmalloc('pbeq.src','PREP1','RXNFY',NATOM,crl=RXNFY)
          call chmalloc('pbeq.src','PREP1','RXNFZ',NATOM,crl=RXNFZ)
          call chmalloc('pbeq.src','PREP1','DBFX',NATOM,crl=DBFX)
          call chmalloc('pbeq.src','PREP1','DBFY',NATOM,crl=DBFY)
          call chmalloc('pbeq.src','PREP1','DBFZ',NATOM,crl=DBFZ)
          call chmalloc('pbeq.src','PREP1','IBFX',NATOM,crl=IBFX)
          call chmalloc('pbeq.src','PREP1','IBFY',NATOM,crl=IBFY)
          call chmalloc('pbeq.src','PREP1','IBFZ',NATOM,crl=IBFZ)
          call chmalloc('pbeq.src','PREP1','NPFX',NATOM,crl=NPFX)
          call chmalloc('pbeq.src','PREP1','NPFY',NATOM,crl=NPFY)
          call chmalloc('pbeq.src','PREP1','NPFZ',NATOM,crl=NPFZ)
       end if
    
#if KEY_SCCDFTB==1
       if (qmused_sccdftb) then
          if (allocated(iphiwtb)) then
             call chmrealloc('pbeq.src','PREP1','IPHIWTB',NCEL3,cr4=IPHIWTB)
             ! for EPSW of mulligan charge
             call chmrealloc('pbeq.src','PREP1','IPHIPTB',NCEL3,cr4=IPHIPTB)
             ! for EPSP of mulligan charge
             call chmrealloc('pbeq.src','PREP1','IPHIWEX',NCEL3,cr4=IPHIWEX)
             ! for EPSW of excluded mm charge
             call chmrealloc('pbeq.src','PREP1','IPHIPEX',NCEL3,cr4=IPHIPEX)
             ! for EPSP of excluded mm charge
          else
             call chmalloc('pbeq.src','PREP1','IPHIWTB',NCEL3,cr4=IPHIWTB)
             ! for EPSW of mulligan charge
             call chmalloc('pbeq.src','PREP1','IPHIPTB',NCEL3,cr4=IPHIPTB)
             ! for EPSP of mulligan charge
             call chmalloc('pbeq.src','PREP1','IPHIWEX',NCEL3,cr4=IPHIWEX)
             ! for EPSW of excluded mm charge
             call chmalloc('pbeq.src','PREP1','IPHIPEX',NCEL3,cr4=IPHIPEX)
             ! for EPSP of excluded mm charge
          end if
      
         ! XIAO_QC_UW0609  keyword for SCC/PB
         MXTPSC = GTRMI(COMLYN,COMLEN,'MXPS',5000)
         PSCTOL = GTRMF(COMLYN,COMLEN,'PSTL',1.0D-2)
         QGAS1 = INDXA(COMLYN,COMLEN,'IGAS').GT.0
         ! XIAO_QC_UW0609 add charge dependant radii
         QCHDRAD = INDXA(COMLYN,COMLEN,'CHDR').GT.0
         IF (QCHDRAD) THEN
            WRITE(OUTU,'(/,3X,A)') &
                 'SCC-PB: Charge dependant radii will be used.'
            ! XIAO_QC_UW0809 add charge dependant radii
            ! ADD unit number to read in radius.inp file
            URAD  = GTRMI (COMLYN,COMLEN,'URAD',-1)
            if(URAD.gt.0)then
               write(outu,'(a)') &
                    '   Charge dependant radii parameters will be'
               write(outu,'(a,i3)') &
                    '   read from unit ', URAD
            else
               write(outu,'(a)') '   please provide a positive unit for URAD'
               stop
            endif
            CALL chardradread(URAD)
         ENDIF
       end if
#endif 

       IF(QFOCUS) THEN
          IF(PRNLEV.GE.2) WRITE(OUTU,'(/,3X,A)') &
               'PREP WARNING: Focussing does not work with PB force.'
          CALL WRNDIE(-5,'<PREP>', &
               'BOUNDARY POTENTIAL METHOD SELECTION ERROR')
       ENDIF
    ENDIF

    IF(.not.QPBF) THEN
       QSMTH=INDXA(COMLYN,COMLEN,'SMOO').GT.0
       IF(QSMTH) THEN
          SWIN  = GTRMF(COMLYN,COMLEN,'SWIN',half)
          IF(PRNLEV.GE.2) WRITE(OUTU,103)
          IF(PRNLEV.GE.2) WRITE(OUTU,103) &
               'Dielectric Smoothing Function will be applied.'
          IF(PRNLEV.GE.2) WRITE(OUTU,'(3X,A,F5.3,A,2X)') &
               'Dielectric Smoothing Region   (2*SWIN) = +/-', &
               SWIN,' from atomic surface'
       ENDIF
    ELSE
       QSMTH=INDXA(COMLYN,COMLEN,'SMOO').GT.0
       IF(QSMTH) THEN
          SWIN = GTRMF(COMLYN,COMLEN,'SWIN',half)
          IF(SWIN.LE.ZERO) THEN
             IF(PRNLEV.GE.2) WRITE(OUTU,'(/,3X,A)') &
                  'PREP WARNING: SWIN must be greater than zero.'
             CALL WRNDIE(-5,'<PREP>', &
                  'SMOOTHING WINDOW SELECTION ERROR')
          ENDIF
          IF(PRNLEV.GE.2) WRITE(OUTU,103)
          IF(PRNLEV.GE.2) WRITE(OUTU,103) &
               'Dielectric Smoothing Function will be applied.'
          IF(PRNLEV.GE.2) WRITE(OUTU,'(3X,A,F5.3,A,2X)') &
               'Dielectric Smoothing Region   (2*SWIN) = +/-', &
               SWIN,' from atomic surface'
       ELSE
          IF(PRNLEV.GE.2) WRITE(OUTU,'(/,3X,A)') &
               'PREP WARNING: SWIN must be selected for Force.'
          CALL WRNDIE(-5,'<PREP>','SMOOTHING WINDOW SELECTION ERROR')
       ENDIF
       !
       NPBEQ = GTRMI(COMLYN,COMLEN,'NPBE',1)
       IF(PRNLEV.GE.2) WRITE(OUTU,'(3x,A,I3,A)') &
            'PBEQ will be carried out every',NPBEQ, &
            ' step during Minimization or Dynamics'
       !
       STCO = GTRMF(COMLYN,COMLEN,'STEN',ZERO)
       IF(STCO.GT.ZERO) THEN
          IF(PRNLEV.GE.2) WRITE(OUTU,103) &
               'Surface Tension Coefficient   (STEN)   = ',stco, &
               ' [KCAL/MOL/A^2]'
       ENDIF
       !
       IF(QPBEQ.AND..not.QSMTH) THEN
          IF(PRNLEV.GE.2) WRITE(OUTU,'(/,3X,A)') &
               'PREP WARNING: SWIN must be selected for Force.'
          CALL WRNDIE(-5,'<PREP>','SMOOTHING WINDOW SELECTION ERROR')
       ENDIF
       !
       QPBEQ  = .TRUE.
       call chmdealloc('pbeq.src','PREP1','ISLCT',NATOM,intg=ISLCT) 
       RETURN
       !
    ENDIF

    !mf, read dielectric grid from external file?
    EPSU  = GTRMI(COMLYN,COMLEN,'EPSU',-1)
    EPSG  = GTRMI(COMLYN,COMLEN,'EPSG',-1)

    ! =================================================================
    ! Construct all space-dependent functions (without calling PBFORCE)
    ! dielectric constant, Kappa, charge density and membrane.
    ! =================================================================
    IF(PRNLEV.GE.2) WRITE(OUTU,103)
    IF(PRNLEV.GE.2) WRITE(OUTU,103) &
         'Constructing all space-dependent functions'

    CALL MAYER(NTPRP,LSTPRP,EPSW,EPSP,KAPPA2, &
         X,Y,Z,CG,PBRAD,NCLX,NCLY,NCLZ,DCEL,WATR,IONR,MAY, &
         MAYX,MAYY,MAYZ,SWIN,ICHC,IPHIW, &
         !      -----                -----
         VMEMB,TMEMB,ZMEMB,EPSM,NIMGB, &
         HTMEMB,EPSH,HDENS,POSALP,NEGALP,QCHRG,POFFST,NOFFST, &
         DROPLET,EPSD,XDROPLET,YDROPLET,ZDROPLET,Qdtom,Qdkap, &
         EPSB,LXMAX,LYMAX,LZMAX,LXMIN,LYMIN,LZMIN,Qbtom,Qbkap, &
         EPSC,Xcyln,Ycyln,Zcyln,Rcyln,Hcyln,Qctom,Qckap, &
         EPSEC,Xcone,Ycone,Zcone,Ax,By,Cz,Qectom,Qeckap, &
         !mf, unit numbers for external dielectric grids
         EPSU,EPSG, &
         TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
         IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
         QKEEP,QREEN,QSMTH,QBSPL,QA,QB, &
         !                      ----- -----
         RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
         RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
         MAJOR,MINOR,QPROLATE, &
         QINTBP,QZERO,QFOCUS,IPHIB,QPBC,QNPBC, &
         !          ------ ----- ----- -----------
         QNONLINEAR,QPARTLINEAR,QFKAP, &
         !          ---------- -----------
         IPOX,IPOY,IPOZ,MAPT,NATOM)
    !
    QPBEQ  = .TRUE.
    !
    !
    if(qfocus) call chmdealloc('pbeq.src','PREP1','IPHIB',FNCEL3,cr4=IPHIB)
    call chmdealloc('pbeq.src','PREP1','ISLCT',NATOM,intg=ISLCT) 
    RETURN
  END SUBROUTINE PREP1


  SUBROUTINE STRECTBOX(NTPRP,LSTPRP,X,Y,Z,CG, &
       NTRA,LSTRA,QTOTA,NTRB,LSTRB,QTOTB, &
       RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX &
#if KEY_GCMC==1
       ,GCMCON  & 
#endif
       )
    !-----------------------------------------------------------------------
    !  Make the lists of the atoms that will be taken into account for
    !  the reaction field calculation inside a rectangular box.
    !  NTRA : Number of Atoms in fixed region (A)
    !  NTRB : Number of Atoms in region of interest (B)
    !
    use chm_kinds
    use number
    implicit none
    INTEGER  NTPRP,LSTPRP(*),NTRA,LSTRA(*),NTRB,LSTRB(*)
    real(chm_real)   X(*),Y(*),Z(*),CG(*)
    real(chm_real)   QTOTA,QTOTB
    real(chm_real)   RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX
#if KEY_GCMC==1
    LOGICAL  GCMCON(:) 
#endif
    ! Local variables
    INTEGER  I,J
    !
    NTRA=0
    NTRB=0
    QTOTA=ZERO
    QTOTB=ZERO
    DO I=1,NTPRP
       J=LSTPRP(I)
       IF( &
#if KEY_GCMC==1
            GCMCON(J) .AND.  & 
#endif
            (X(J).LT.RBXMIN.OR.X(J).GT.RBXMAX.OR. &
            Y(J).LT.RBYMIN.OR.Y(J).GT.RBYMAX.OR. &
            Z(J).LT.RBZMIN.OR.Z(J).GT.RBZMAX)) THEN
          QTOTA=QTOTA+CG(J)
          NTRA=NTRA+1
          LSTRA(NTRA)=J
       ELSE
          QTOTB=QTOTB+CG(J)
          NTRB=NTRB+1
          LSTRB(NTRB)=J
       ENDIF
    ENDDO

    RETURN
  END SUBROUTINE STRECTBOX

  SUBROUTINE STSPHERE(NTPRP,LSTPRP,X,Y,Z,CG, &
       NTRA,LSTRA,QTOTA,NTRB,LSTRB,QTOTB, &
       SRDIST,RRXCEN,RRYCEN,RRZCEN &
#if KEY_GCMC==1
       ,GCMCON  & 
#endif
       )
    !-----------------------------------------------------------------------
    !  Make the lists of the atoms that will be taken into account for
    !  the reaction field calculation inside a sphere.
    !  NTRA : Number of Atoms in fixed region (A)
    !  NTRB : Number of Atoms in region of interest (B)
    !
    use chm_kinds
    use number
    implicit none
    INTEGER  NTPRP,LSTPRP(*),NTRA,LSTRA(*),NTRB,LSTRB(*)
    real(chm_real)   X(*),Y(*),Z(*),CG(*)
    real(chm_real)   QTOTA,QTOTB
    real(chm_real)   SRDIST,RRXCEN,RRYCEN,RRZCEN
#if KEY_GCMC==1
    LOGICAL  GCMCON(:) 
#endif
    ! Local variables
    INTEGER  I,J
    real(chm_real)   XDIFF,YDIFF,ZDIFF,R2,SR2
    !
    NTRA=0
    NTRB=0
    QTOTA=ZERO
    QTOTB=ZERO
    SR2=SRDIST*SRDIST
    DO I=1,NTPRP
       J=LSTPRP(I)
#if KEY_GCMC==1
       IF (GCMCON(J)) THEN 
#endif
          XDIFF=X(J)-RRXCEN
          YDIFF=Y(J)-RRYCEN
          ZDIFF=Z(J)-RRZCEN
          R2=XDIFF*XDIFF+YDIFF*YDIFF+ZDIFF*ZDIFF
          IF(R2.GT.SR2) THEN
             QTOTA=QTOTA+CG(J)
             NTRA=NTRA+1
             LSTRA(NTRA)=J
          ELSE
             QTOTB=QTOTB+CG(J)
             NTRB=NTRB+1
             LSTRB(NTRB)=J
          ENDIF
#if KEY_GCMC==1
       ELSE
          QTOTB=QTOTB+CG(J)
          NTRB=NTRB+1
          LSTRB(NTRB)=J
       ENDIF
#endif 
    ENDDO

    RETURN
  END SUBROUTINE STSPHERE

  SUBROUTINE STPROLATE(NTPRP,LSTPRP,X,Y,Z,CG, &
       NTRA,LSTRA,QTOTA,NTRB,LSTRB,QTOTB, &
       MAJOR,MINOR,RRXCEN,RRYCEN,RRZCEN)
    !-----------------------------------------------------------------------
    !  Make the lists of the atoms that will be taken into account for
    !  the reaction field calculation inside a prolate spheroid.
    !  NTRA : Number of Atoms in fixed region (A)
    !  NTRB : Number of Atoms in region of interest (B)
    !
    use chm_kinds
    use number
    implicit none
    INTEGER  NTPRP,LSTPRP(*),NTRA,LSTRA(*),NTRB,LSTRB(*)
    real(chm_real)   X(*),Y(*),Z(*),CG(*)
    real(chm_real)   QTOTA,QTOTB
    real(chm_real)   MAJOR,MINOR,RRXCEN,RRYCEN,RRZCEN
    ! Local variables
    INTEGER  I,J
    real(chm_real)   XDIFF,YDIFF,ZDIFF,R1,R2
    !
    NTRA=0
    NTRB=0
    QTOTA=ZERO
    QTOTB=ZERO
    DO I=1,NTPRP
       J=LSTPRP(I)
       XDIFF=X(J)-RRXCEN
       YDIFF=Y(J)-RRYCEN
       ZDIFF=Z(J)-RRZCEN
       R1=(XDIFF*XDIFF+YDIFF*YDIFF)/(MINOR*MINOR)
       R2=ZDIFF*ZDIFF/(MAJOR*MAJOR)
       IF(R1+R2.GT.ONE) THEN
          QTOTA=QTOTA+CG(J)
          NTRA=NTRA+1
          LSTRA(NTRA)=J
       ELSE
          QTOTB=QTOTB+CG(J)
          NTRB=NTRB+1
          LSTRB(NTRB)=J
       ENDIF
    ENDDO

    RETURN
  END SUBROUTINE STPROLATE

  SUBROUTINE MAYER(NTPRP,LSTPRP,EPSW,EPSP,KAPPA2, &
       X,Y,Z,CG,PBRAD, &
       NCLX,NCLY,NCLZ,DCEL,RADW,RADI, &
       RMAY,RMAYX,RMAYY,RMAYZ,SWIN,CHC,PHI, &
       VMEMB,TMEMB,ZMEMB,EPSM,NIMGB, &
       HTMEMB,EPSH,HDENS,POSALP,NEGALP,QCHRG,POFFST,NOFFST, &
       DROPLET,EPSD,XDROPLET,YDROPLET,ZDROPLET,Qdtom,Qdkap, &
       EPSB,LXMAX,LYMAX,LZMAX,LXMIN,LYMIN,LZMIN,Qbtom,Qbkap, &
       EPSC,Xcyln,Ycyln,Zcyln,Rcyln,Hcyln,Qctom,Qckap, &
       EPSEC,Xcone,Ycone,Zcone,Ax,By,Cz,Qectom,Qeckap, &
       !mf, unit numbers for external dielectric grids
       EPSU, EPSG, &
       TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
       IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
       QKEEP,QREEN,QSMTH,QBSPL,QA,QB, &
       RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
       RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
       MAJOR,MINOR,QPROLATE, &
       QINTBP,QZERO,QFOCUS,PHIB,QPBC,QNPBC, &
       QNONLINEAR,QPARTLINEAR,QFKAP, &
       POX,POY,POZ,MAPT,NATOM) 
    !-----------------------------------------------------------------------
    !     Construction of all the 3d arrays necessary for performing the
    !     iterations on the Poisson-Boltzmann equation (epsilon, kappa,
    !     charge).This is really the central setup for any calculation.
    !     This is where the microscopic model is created and this is a key
    !     routine.
    !
    use chm_kinds
    use number
    use stream
    use consta
    use memory
    implicit none
    REAL(CHM_REAL4)   CHC(*),PHI(*),PHIB(*)
    REAL(CHM_REAL4)   RMAY(*),RMAYX(*),RMAYY(*),RMAYZ(*)
    real(chm_real)   X(*),Y(*),Z(*),CG(*),POX(*),POY(*),POZ(*)
    real(chm_real)   PBRAD(*)
    real(chm_real)   DCEL,TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN
    real(chm_real)   EPSW,EPSP,KAPPA2,RADW,RADI,SWIN
    real(chm_real)   VMEMB,ZMEMB,TMEMB,EPSM,HTMEMB,EPSH
    real(chm_real)   HDENS,POSALP,NEGALP,NOFFST,POFFST
    real(chm_real)   DROPLET,EPSD,XDROPLET,YDROPLET,ZDROPLET
    real(chm_real)   EPSB,LXMAX,LYMAX,LZMAX,LXMIN,LYMIN,LZMIN
    real(chm_real)   EPSC,XCYLN,YCYLN,ZCYLN,RCYLN,HCYLN
    real(chm_real)   EPSEC,Xcone,Ycone,Zcone,Ax,By,Cz
    real(chm_real)   RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX
    real(chm_real)   SRDIST,RRXCEN,RRYCEN,RRZCEN
    real(chm_real)   MAJOR,MINOR
    INTEGER  LSTPRP(*)
    INTEGER  NTPRP,NCLX,NCLY,NCLZ,MAPT,NIMGB
    INTEGER  IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN,NATOM
    !mf, unit numbers for external dielectric grids
    INTEGER  EPSU,EPSG
    LOGICAL  QKEEP,QREEN,QCHRG,QSMTH,QBSPL
    LOGICAL  QCTOM,QDTOM,QCKAP,QDKAP,QBTOM,QBKAP,QECTOM,QECKAP
    LOGICAL  QINTBP,QZERO,QFOCUS,QPBC,QNPBC
    LOGICAL  QA,QB,QRECTBOX,QSPHERE,QPROLATE
    LOGICAL  QNONLINEAR,QPARTLINEAR,QFKAP
    ! local
    real(chm_real)   XMAX,XMIN,YMAX,YMIN,ZMAX,ZMIN
    real(chm_real)   BISQ,KAPPA
    real(chm_real)   XC,YC,ZC,XI,YI,ZI,XJ,YJ,ZJ,XP,YP,ZP,XC1,YC1,ZC1
    real(chm_real)   DSQ,DSQ1,XSQ,YSQ,ZSQ,ZISQ,DCEL2,DCEL3
    integer  i,j,ig,iii,iiix,iix,iiy,iiz,il
    integer  ipo,ipx,ipy,ipz,ix,iy,iz,jg,jgj
    integer  jx1,jx2,jy1,jy2,jz1,jz2
    integer  k,kg,km,l,m,ncyz,nc3,nfil,nl,jl,ml
    integer  kl,ll
    integer  nfilz

    !
    !     Set kappa, ncyz, nc3, dcel2, dcel3, iseed
    !     =========================================

    ! write(*,*) "QC> Arrived at Mayer "

    kappa=sqrt(kappa2/EPSW)
    ncyz=ncly*nclz
    nc3=nclx*ncyz
    dcel2=dcel*dcel
    dcel3=dcel2*dcel

    !
    !     Check if some atoms lie outside the box
    !     =======================================

    xmax=-100000.
    ymax=-100000.
    zmax=-100000.
    xmin=100000.
    ymin=100000.
    zmin=100000.

    do il=1,ntprp
       i=lstprp(il)
       xi=x(i)
       yi=y(i)
       zi=z(i)
       bisq=pbrad(i)+radw+radi+swin
       if(xi+bisq.gt.xmax) xmax=xi+bisq
       if(yi+bisq.gt.ymax) ymax=yi+bisq
       if(zi+bisq.gt.zmax) zmax=zi+bisq
       if(xi-bisq.lt.xmin) xmin=xi-bisq
       if(yi-bisq.lt.ymin) ymin=yi-bisq
       if(zi-bisq.lt.zmin) zmin=zi-bisq
    enddo

    ! to make the SOR (in PBEQ1) method faster
    ! mf, modified to work with external dielectric grids
    IF (EPSU.LE.0 .AND. EPSG.LE.0) THEN
      IXMAX=int((xmax+tranx-xbcen)/dcel)+2
      IYMAX=int((ymax+trany-ybcen)/dcel)+2
      IZMAX=int((zmax+tranz-zbcen)/dcel)+2
      IXMIN=int((xmin+tranx-xbcen)/dcel)-1
      IYMIN=int((ymin+trany-ybcen)/dcel)-1
      IZMIN=int((zmin+tranz-zbcen)/dcel)-1
    ELSE
      IXMAX=NCLX-1
      IXMIN=2
      IYMAX=NCLY-1
      IYMIN=2
      IZMAX=NCLZ-1
      IZMIN=2
    ENDIF

    IF((QRECTBOX.or.QSPHERE.or.QPROLATE).and.(QA.and.QB))GOTO 108
    IF(XMAX.GT.XBCEN+TRANX.OR.XMIN.LT.XBCEN-TRANX)THEN
       IF(.not.QFOCUS.and..not.QA.and.QB)THEN
          IF(PRNLEV.GE.2) WRITE(OUTU,100)
          IF(PRNLEV.GE.2) WRITE(OUTU,100)  &
               'WARNING: X GRIDSIZE IS TOO SMALL'
          IF(PRNLEV.GE.2) WRITE(OUTU,100)  &
               'XMAX,GRIDXMAX=',XMAX,XBCEN+TRANX
          IF(PRNLEV.GE.2) WRITE(OUTU,100)  &
               'XMIN,GRIDXMIN=',XMIN,XBCEN-TRANX
          CALL WRNDIE(-1,'<MAYER>','GRID SIZE SELECTION ERRORS')
       ENDIF
    ENDIF
    IF(YMAX.GT.YBCEN+TRANY.OR.YMIN.LT.YBCEN-TRANY)THEN
       IF(.not.QFOCUS.and..not.QA.and.QB)THEN
          IF(PRNLEV.GE.2) WRITE(OUTU,100)
          IF(PRNLEV.GE.2) WRITE(OUTU,100)  &
               'WARNING: Y GRIDSIZE IS TOO SMALL'
          IF(PRNLEV.GE.2) WRITE(OUTU,100)  &
               'YMAX,GRIDYMAX=',YMAX,YBCEN+TRANY
          IF(PRNLEV.GE.2) WRITE(OUTU,100)  &
               'YMIN,GRIDYMIN=',YMIN,YBCEN-TRANY
          CALL WRNDIE(-1,'<MAYER>','GRID SIZE SELECTION ERRORS')
       ENDIF
    ENDIF
    IF(ZMAX.GT.ZBCEN+TRANZ.OR.ZMIN.LT.ZBCEN-TRANZ)THEN
       IF(.not.QFOCUS.and..not.QA.and.QB)THEN
          IF(PRNLEV.GE.2) WRITE(OUTU,100)
          IF(PRNLEV.GE.2) WRITE(OUTU,100)  &
               'WARNING: Z GRIDSIZE IS TOO SMALL'
          IF(PRNLEV.GE.2) WRITE(OUTU,100)  &
               'ZMAX,GRIDZMAX=',ZMAX,ZBCEN+TRANZ
          IF(PRNLEV.GE.2) WRITE(OUTU,100)  &
               'ZMIN,GRIDZMIN=',ZMIN,ZBCEN-TRANZ
          CALL WRNDIE(-1,'<MAYER>','GRID SIZE SELECTION ERRORS')
       ENDIF
    ENDIF
100 FORMAT(3X,A,2X,2F10.5)
108 CONTINUE

    ! write(*,*) "QC> Mayer: initialization 1",QKEEP
    !
    !     Initialize Potential at each Grid (if necessary)
    !     ================================================

    IF(.NOT.QKEEP)THEN
       do km=1,nc3
          phi(km)=0.0
       enddo
    ELSEIF(.NOT.QNONLINEAR.AND..NOT.QPARTLINEAR) THEN
       !     Boundary Potential
       IF(TMEMB.GT.0.0.and..not.QNPBC)THEN
          !     periodic boundary condition in XY
          do i=1,nclx
             do j=1,ncly
                phi((i-1)*ncyz + (j-1)*nclz + 1)=0     !xy plane at z=zmin
                phi((i-1)*ncyz + (j-1)*nclz + nclz)=0  !xy plane at z=zmax
             enddo
          enddo
       ELSE
          do i=1,nclx
             do j=1,ncly
                phi((i-1)*ncyz + (j-1)*nclz + 1)=0     !xy plane at z=zmin
                phi((i-1)*ncyz + (j-1)*nclz + nclz)=0  !xy plane at z=zmax
             enddo
          enddo
          do i=1,ncly
             do j=1,nclz
                phi(                (i-1)*nclz + j)=0  !yz plane at x=xmin
                phi((nclx-1)*ncyz + (i-1)*nclz + j)=0  !yz plane at x=xmax
             enddo
          enddo
          do i=1,nclx
             do j=1,nclz
                phi((i-1)*ncyz                 + j)=0  !xz plane at y=ymin
                phi((i-1)*ncyz + ncyz-nclz     + j)=0  !xz plane at y=ymax
             enddo
          enddo
       ENDIF
    ENDIF

    ! write(*,*) "QC> Mayer: initialization 2"
    !
    !     Initialize All the Space-dependant Functions
    !     ============================================
    !     rmay (Debye-Huckel factor) -> kappa2*dcel2
    !     rmayx,y,z (dielectric constant) -> EPSW (80.0)
    !     chc (charge density) -> 0.0

    do km=1,nc3
       rmayx(km)=EPSW
       rmayy(km)=EPSW
       rmayz(km)=EPSW
    enddo

    IF(.not.QFKAP) THEN
       do km=1,nc3
          rmay(km)=kappa2*dcel2
       enddo
    ENDIF

    IF(.not.QA.or..not.QB) THEN    ! for reaction field calculation
       do km=1,nc3                 ! in GSBP1 and GSBP2
          chc(km)=ZERO
       enddo
    ENDIF

    !
    !     MEMBRANE PART (if requested)
    !     ============================
    !

    IF(TMEMB.GT.0.0)THEN
       CALL MAYER_MEMB(NCLX,NCLY,NCLZ,DCEL,TRANX,TRANY,TRANZ, &
            ZBCEN,TMEMB,ZMEMB,EPSM,HTMEMB,EPSH, &
            HDENS,POSALP,NEGALP,QCHRG,POFFST,NOFFST, &
            CHC,RMAY,RMAYX,RMAYY,RMAYZ,QFKAP)
       ! for subroutine PBEQ1
       IXMIN=2
       IXMAX=NCLx-1
       IYMIN=2
       IYMAX=NCLy-1
       IZMIN=2
       IZMAX=NCLz-1
    ENDIF


    !
    !     DROPLET PART (if requested)
    !     ===========================

    IF(DROPLET.GT.0.0)THEN
       CALL MAYER_DROP(NCLX,NCLY,NCLZ,DCEL,TRANX,TRANY,TRANZ, &
            XBCEN,YBCEN,ZBCEN,KAPPA2,TMEMB,ZMEMB,EPSM,HTMEMB,EPSH, &
            DROPLET,EPSD,XDROPLET,YDROPLET,ZDROPLET,Qdtom,Qdkap, &
            IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
            CHC,RMAY,RMAYX,RMAYY,RMAYZ,QFKAP)
    ENDIF

    !
    !     CYLINDER PART (if requested)
    !     ============================

    IF(RCYLN.gt.0.0)THEN
       CALL MAYER_CYLN(NCLX,NCLY,NCLZ,DCEL,TRANX,TRANY,TRANZ, &
            XBCEN,YBCEN,ZBCEN,KAPPA2,TMEMB,ZMEMB,EPSM,HTMEMB,EPSH, &
            EPSC,Xcyln,Ycyln,Zcyln,Rcyln,Hcyln,Qctom,Qckap, &
            IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
            CHC,RMAY,RMAYX,RMAYY,RMAYZ,QFKAP)
    ENDIF

    !
    !     BOX PART (if requested)
    !     =======================

    IF(LXMAX.ne.LXMIN)THEN
       CALL MAYER_BOX(NCLX,NCLY,NCLZ,DCEL,TRANX,TRANY,TRANZ, &
            XBCEN,YBCEN,ZBCEN,KAPPA2,TMEMB,ZMEMB,EPSM,HTMEMB,EPSH, &
            EPSB,LXMAX,LYMAX,LZMAX,LXMIN,LYMIN,LZMIN,Qbtom,Qbkap, &
            IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
            CHC,RMAY,RMAYX,RMAYY,RMAYZ,QFKAP)
    ENDIF


    !
    !     Elliptical CONE PART (if requested)
    !     ===================================

    IF(Ax.gt.0.0)THEN
       CALL MAYER_ECONE(NCLX,NCLY,NCLZ,DCEL,TRANX,TRANY,TRANZ, &
            XBCEN,YBCEN,ZBCEN,KAPPA2,TMEMB,ZMEMB,EPSM,HTMEMB,EPSH, &
            EPSEC,Xcone,Ycone,Zcone,Ax,By,Cz,Qectom,Qeckap, &
            IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
            CHC,RMAY,RMAYX,RMAYY,RMAYZ,QFKAP)
    ENDIF
    ! write(*,*) "QC> Mayer: initialization 3"

    !     Set the potential at the edge of the box (BOUNDARY CONDITIONS)
    !     =============================================================
    !     The default is the use of potentials at boundary points due to
    !     the charges of the solute and the salt concentration.
    !     IF QINTBP is true, half number of grid points is used for
    !                        the Debye-Huckel approximation
    !     IF QZERO is true, zero potential is set (above)
    !     IF QFOCUS is true, previous potential is interpolated at boundary points
    !     IF NIMGb > 0 , boundary potential is built by using the nearest image
    !     IF QPBC is true, skip the boundary potential calculations

    IF(QKEEP.AND.(QNONLINEAR.OR.QPARTLINEAR)) GOTO 109
    IF(.NOT.QFOCUS.AND..NOT.QZERO.AND..NOT.QINTBP.AND..NOT.QPBC)THEN
       CALL MAYER_BP_ALL(NTPRP,LSTPRP,X,Y,Z,CG,EPSW,KAPPA2, &
            TMEMB,CHC,PHI,NCLX,NCLY,NCLZ,DCEL, &
            TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
            RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
            RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
            MAJOR,MINOR,QPROLATE, &
            NCLX,NCLY,NCLZ,DCEL,XBCEN,YBCEN,ZBCEN,NIMGB,QNPBC,QA,QB)
    ELSEIF(QINTBP) THEN
       CALL MAYER_BP_INT(NTPRP,LSTPRP,X,Y,Z,CG,EPSW,KAPPA2, &
            TMEMB,CHC,PHI,NCLX,NCLY,NCLZ,DCEL, &
            TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
            RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
            RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
            MAJOR,MINOR,QPROLATE, &
            NCLX,NCLY,NCLZ,DCEL,XBCEN,YBCEN,ZBCEN,NIMGB,QNPBC,QA,QB)
    ELSEIF(QFOCUS) THEN
       CALL MAYER_BPFOCUS(PHI,PHIB,NCLX,NCLY,NCLZ)
    ENDIF
109 CONTINUE

    !
    !     Charge Distribution
    !     ===================
    !     If QBSPL is true,
    !                use the 3rd-order B-splines interpolation (27 grid points)
    !     otherwise, use the trilinear interpolation ( 8 gird points)

    ! to exclude atoms in region of interest in the reaction field calculations
    IF(QA.AND.QB) GOTO 110
    IF(QBSPL) THEN
       CALL MAYER_CD_BSPL(NTPRP,LSTPRP,X,Y,Z,CG,CHC,KAPPA2, &
            NCLX,NCLY,NCLZ,DCEL,TRANX,TRANY,TRANZ, &
            XBCEN,YBCEN,ZBCEN, &
            RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
            RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
            MAJOR,MINOR,QPROLATE, &
            CHC,NCLX,NCLY,NCLZ,DCEL,XBCEN,YBCEN,ZBCEN, &
            QPBC,QA,QB)
    ELSE
       CALL MAYER_CD_TRIL(NTPRP,LSTPRP,X,Y,Z,CG,CHC,KAPPA2, &
            NCLX,NCLY,NCLZ,DCEL,TRANX,TRANY,TRANZ, &
            XBCEN,YBCEN,ZBCEN, &
            RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
            RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
            MAJOR,MINOR,QPROLATE, &
            CHC,NCLX,NCLY,NCLZ,DCEL,XBCEN,YBCEN,ZBCEN, &
            QPBC,QA,QB)
    ENDIF
110 CONTINUE

    ! write(*,*) "QC> Mayer: initialization 4"
    !
    !     Solute Volume Exclusion Functions (Mayer Function)
    !     ==================================================
    !     when the dielectric boundary is defined by the van der Waals surface
    !     M = 0 : inside solute
    !         1 : outside solute
    !
    !     when the extended dielectric boundary (smoothing) is used
    !     M = 0                       : inside solute
    !     M = smoothing function      : within the dielectric boundary
    !     M = 1                       : outside solute

    IF((EPSW.eq.EPSP).and.(KAPPA.eq.ZERO)) RETURN 
    !wi,bug      IF(EPSW.eq.EPSP) RETURN

    IF(.not.QSMTH) then
       CALL MAYER_MSTEP(NTPRP,LSTPRP,X,Y,Z,PBRAD,RADW,RADI,EPSP, &
            NCLX,NCLY,NCLZ,DCEL, &
            TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
            RMAY,RMAYX,RMAYY,RMAYZ,QREEN,QFKAP)
    ELSE
       IF(RADW.GT.0.or.RADI.GT.0)THEN
          if(prnlev.ge.2) write(outu,'(/,3x,2a)') &
               'MAYER WARNING: Dielectric Smoothing does not work ', &
               'with WATR or IONR.'
          CALL WRNDIE(-5,'<MAYER>','PROBE SOLVENT SELECTION ERRORS')
       ENDIF
       CALL MAYER_MSMOOTH(NTPRP,LSTPRP,X,Y,Z,PBRAD,SWIN, &
            EPSP,EPSW,EPSM,EPSH,KAPPA2,NCLX,NCLY,NCLZ,DCEL, &
            TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
            RMAY,RMAYX,RMAYY,RMAYZ,QFKAP)
    ENDIF

    ! write(*,*) "QC> Mayer: initialization 5"
    !
    !     PROBE RADIUS DEPENDENT PART
    !     ===========================
    !     works when radw >  0. .and. QREEN = true
    !     Then all grid points inside accessible surface
    !     are already set as a solute and it is now
    !     necessary to extend the dielectric
    !     till the molecular (contact+reentrance) surface

    IF( RADW.GT.0.0 .and. QREEN ) THEN
       CALL MAYER_REEN(NTPRP,LSTPRP,X,Y,Z,PBRAD,EPSP,RADW, &
            NCLX,NCLY,NCLZ,DCEL, &
            TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
            RMAY,RMAYX,RMAYY,RMAYZ,QFKAP, &
            POX,POY,POZ,MAPT)
    ENDIF

    !     Set the Membrane Potential
    !     ==========================
    !     Fix the initial membrane potential and the Boltzmann distribution
    !     for the salt concentration due to presence of offset potential VMEMB

    IF(VMEMB.ne.0.0)THEN
       CALL MAYER_VMEMB(NCLX,NCLY,NCLZ,DCEL,TRANX,TRANY,TRANZ, &
            ZBCEN,EPSW,KAPPA2,VMEMB,TMEMB,ZMEMB,EPSM, &
            CHC,PHI,RMAY,QFOCUS,QKEEP,QNONLINEAR,QPARTLINEAR)
    ENDIF

    !mf, read dielectric grid(s) from external file
    !    this will overwrite the previously generated grids
    IF (EPSU.GT.0) THEN
       IF (PRNLEV.GE.2) &
        WRITE(OUTU,*) '  Reading dielectric function from unit ',epsu 
       CALL MAYER_REPS(NCLX,NCLY,NCLZ,DCEL,TRANX,TRANY,TRANZ, &
                       XBCEN,YBCEN,ZBCEN,EPSU,RMAYX,RMAYY,RMAYZ)
    ENDIF
    IF (EPSG.GT.0) THEN
       IF (PRNLEV.GE.2) &
        WRITE(OUTU,*) '  Reading dielectric grid from unit ',epsg 
       CALL MAYER_REPSG(NCLX,NCLY,NCLZ,DCEL,TRANX,TRANY,TRANZ,  &
                        XBCEN,YBCEN,ZBCEN,EPSG,RMAYX,RMAYY,RMAYZ)
    ENDIF
      
    ! write(*,*) "QC> Mayer: Done"
    !
    RETURN
  END SUBROUTINE MAYER

  SUBROUTINE MAYER_DROP(NCLX,NCLY,NCLZ,DCEL,TRANX,TRANY,TRANZ, &
       XBCEN,YBCEN,ZBCEN,KAPPA2,TMEMB,ZMEMB,EPSM,HTMEMB,EPSH, &
       DROPLET,EPSD,XDROPLET,YDROPLET,ZDROPLET,Qdtom,Qdkap, &
       IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
       CHC,RMAY,RMAYX,RMAYY,RMAYZ,QFKAP)
    !----------------------------------------------------------------------
    !     Droplet Part
    !
    use chm_kinds
    use number
    use consta
    implicit none
    INTEGER  NCLx,NCLy,NCLz
    INTEGER  IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN
    REAL(CHM_REAL4)   CHC(*),RMAY(*),RMAYX(*),RMAYY(*),RMAYZ(*)
    real(chm_real)   DCEL,TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN
    real(chm_real)   TMEMB,ZMEMB,EPSM,HTMEMB,EPSH,KAPPA2
    real(chm_real)   DROPLET,EPSD,XDROPLET,YDROPLET,ZDROPLET
    LOGICAL  Qdtom,Qdkap,QFKAP
    ! local
    integer  ncyz,nfil,ix,iy,iz,jx1,jx2,jy1,jy2,jz1,jz2
    integer  k,l,m,ipx,ipy,ipz
    real(chm_real)   xi,yi,zi,bisq,xc,yc,zc,xsq,ysq,zsq,dsq,dsq1,dcel2
    real(chm_real)   xc1,yc1,zc1
    real(chm_real)   zmemb1,zmemb2,hmemb1,hmemb2

    ncyz=ncly*nclz
    dcel2=dcel*dcel
    zmemb1=tranz-zbcen-0.5*tmemb+zmemb-rsmall
    zmemb2=zmemb1+TMEMB+2.0*rsmall
    hmemb1=zmemb1+HTMEMB
    hmemb2=zmemb2-HTMEMB
    !
    xi=xdroplet+tranx-xbcen
    yi=ydroplet+trany-ybcen
    zi=zdroplet+tranz-zbcen
    bisq=droplet
    nfil=int(bisq/dcel)+2
    bisq=bisq*bisq+rsmall
    ix=int(xi/dcel)+1
    iy=int(yi/dcel)+1
    iz=int(zi/dcel)+1

    JX1=IX-NFIL+1
    IF(JX1.LT.1)JX1=1
    JX2=IX+NFIL
    IF(JX2.GT.NCLX)JX2=NCLX
    JY1=IY-NFIL+1
    IF(JY1.LT.1)JY1=1
    JY2=IY+NFIL
    IF(JY2.GT.NCLY)JY2=NCLY
    JZ1=IZ-NFIL+1
    IF(JZ1.LT.1)JZ1=1
    JZ2=IZ+NFIL
    IF(JZ2.GT.NCLZ)JZ2=NCLZ

    IF(JX2.GT.IXMAX) IXMAX=JX2+1
    IF(JY2.GT.IYMAX) IYMAX=JY2+1
    IF(JZ2.GT.IZMAX) IZMAX=JZ2+1
    IF(JX1.LT.IXMIN) IXMIN=JX1-1
    IF(JY1.LT.IYMIN) IYMIN=JY1-1
    IF(JZ1.LT.IZMIN) IZMIN=JZ1-1
    !
    do k=jx1,jx2
       ipx=(k-1)*ncyz
       xc=(k-1)*dcel
       do l=jy1,jy2
          ipy=(l-1)*nclz
          yc=(l-1)*dcel
          do m=jz1,jz2
             ipz=m+ipy+ipx
             zc=(m-1)*dcel
             xsq=(xc-xi)**2
             ysq=(yc-yi)**2
             zsq=(zc-zi)**2

             dsq=xsq+ysq+zsq
             IF(.not.QFKAP)THEN
                IF(dsq.le.bisq)THEN
                   rmay(ipz)=0.0 ! Debye screening factor zero
                   ! inside the dropplet
                   IF(QDKAP) rmay(ipz)=kappa2*dcel2
                ENDIF
             ENDIF

             IF(EPSD.ne.0.0) THEN
                xc1=xc+0.5*dcel
                dsq1=(xc1-xi)**2+ysq+zsq
                IF(dsq1.le.bisq)THEN
                   rmayx(ipz)=EPSD ! droplet dielectric constant
                ENDIF

                yc1=yc+0.5*dcel
                dsq1=(yc1-yi)**2+xsq+zsq
                IF(dsq1.le.bisq)THEN
                   rmayy(ipz)=EPSD ! droplet dielectric constant
                ENDIF

                zc1=zc+0.5*dcel
                dsq1=(zc1-zi)**2+ysq+xsq
                IF(dsq1.le.bisq)THEN
                   rmayz(ipz)=EPSD ! droplet dielectric constant
                ENDIF

                !     membrane region
                IF(tmemb.gt.zero.and.QDTOM) THEN
                   IF(zc.ge.zmemb1.and.zc.le.zmemb2) THEN
                      IF(zc.ge.hmemb1.and.zc.le.hmemb2) THEN
                         rmayx(ipz)=EPSM
                         rmayy(ipz)=EPSM
                      ELSE
                         rmayx(ipz)=EPSH
                         rmayy(ipz)=EPSH
                      ENDIF
                   ENDIF
                   IF (zc1.ge.zmemb1.and.zc1.le.zmemb2) THEN
                      IF (zc1.ge.hmemb1.and.zc1.le.hmemb2) THEN
                         rmayz(ipz)=EPSM
                      ELSE
                         rmayz(ipz)=EPSH
                      ENDIF
                   ENDIF
                ENDIF
             ENDIF ! EPSD.ne.0.0

          enddo
       enddo
    enddo
    !
    RETURN
  END SUBROUTINE MAYER_DROP

  SUBROUTINE MAYER_BOX(NCLx,NCLy,NCLz,DCEL,TRANX,TRANY,TRANZ, &
       XBCEN,YBCEN,ZBCEN,KAPPA2,TMEMB,ZMEMB,EPSM,HTMEMB,EPSH, &
       EPSB,LXMAX,LYMAX,LZMAX,LXMIN,LYMIN,LZMIN,Qbtom,Qbkap, &
       IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
       CHC,RMAY,RMAYX,RMAYY,RMAYZ,QFKAP)
    !-----------------------------------------------------------------------
    !     vacuum environment in region B.
    !
    use chm_kinds
    use number
    use consta
    implicit none
    INTEGER  NCLx,NCLy,NCLz
    INTEGER  IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN
    REAL(CHM_REAL4)   CHC(*),RMAY(*),RMAYX(*),RMAYY(*),RMAYZ(*)
    real(chm_real)   DCEL,TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN
    real(chm_real)   TMEMB,ZMEMB,EPSM,HTMEMB,EPSH,KAPPA2
    real(chm_real)   EPSB,LXMAX,LYMAX,LZMAX,LXMIN,LYMIN,LZMIN
    LOGICAL  Qbtom,Qbkap,QFKAP
    ! local
    INTEGER NCYZ,IG,JG,KG,IP0X,IP0Y,IP0
    INTEGER JX1,JX2,JY1,JY2,JZ1,JZ2
    real(chm_real)  xc,yc,zc,xc1,yc1,zc1,dcel2
    real(chm_real)  zmemb1,zmemb2,hmemb1,hmemb2
    real(chm_real)  lxmax1,lymax1,lzmax1,lxmin1,lymin1,lzmin1
    !
    ncyz=ncly*nclz
    dcel2=dcel*dcel
    zmemb1=tranz-zbcen-0.5*tmemb+zmemb-rsmall
    zmemb2=zmemb1+TMEMB+2.0*rsmall
    hmemb1=zmemb1+HTMEMB
    hmemb2=zmemb2-HTMEMB
    lxmax1=lxmax+tranx-xbcen+rsmall
    lxmin1=lxmin+tranx-xbcen-rsmall
    lymax1=lymax+trany-ybcen+rsmall
    lymin1=lymin+trany-ybcen-rsmall
    lzmax1=lzmax+tranz-zbcen+rsmall
    lzmin1=lzmin+tranz-zbcen-rsmall
    !
    JX1=INT(LXMIN1/DCEL)-1
    JX2=INT(LXMAX1/DCEL)+3
    JY1=INT(LYMIN1/DCEL)-1
    JY2=INT(LYMAX1/DCEL)+3
    JZ1=INT(LZMIN1/DCEL)-1
    JZ2=INT(LZMAX1/DCEL)+3

    IF(JX1.LT.1)JX1=1
    IF(JX2.GT.NCLX)JX2=NCLX
    IF(JY1.LT.1)JY1=1
    IF(JY2.GT.NCLY)JY2=NCLY
    IF(JZ1.LT.1)JZ1=1
    IF(JZ2.GT.NCLZ)JZ2=NCLZ

    IF(JX2.GT.IXMAX) IXMAX=JX2+1
    IF(JY2.GT.IYMAX) IYMAX=JY2+1
    IF(JZ2.GT.IZMAX) IZMAX=JZ2+1
    IF(JX1.LT.IXMIN) IXMIN=JX1-1
    IF(JY1.LT.IYMIN) IYMIN=JY1-1
    IF(JZ1.LT.IZMIN) IZMIN=JZ1-1

    ! For Debye-Huckel screening factors
    IF(.not.QFKAP)THEN
       ig_loop: DO IG=JX1,JX2
          XC=DCEL*(IG-1)
          IF(XC.LT.LXMIN1.OR.XC.GT.LXMAX1) cycle ig_loop
          IP0X=(IG-1)*NCyz
          jg_loop: DO JG=JY1,JY2
             YC=DCEL*(JG-1)
             IF(YC.LT.LYMIN1.OR.YC.GT.LYMAX1) cycle jg_loop
             IP0Y=(JG-1)*NCLz+IP0X
             kg_loop: DO KG=JZ1,JZ2
                ZC=DCEL*(KG-1)
                IF(ZC.LT.LZMIN1.OR.ZC.GT.LZMAX1) cycle kg_loop
                IP0 = IP0Y + KG

                rmay(ip0)=0.0 ! Debye screening factor zero
                ! inside the box
                IF(QBKAP) rmay(ip0)=kappa2*dcel2
             ENDDO kg_loop
          ENDDO jg_loop
       ENDDO ig_loop
    ENDIF

    ! For dielectric constants
    ! NOTE: dielectric constants are located at the middle point between grids
    IF(EPSB.eq.0.0) RETURN
    DO IG=JX1,JX2
       IP0X=(IG-1)*NCyz
       XC=DCEL*(IG-1)
       XC1=XC+DCEL*HALF
       DO JG=JY1,JY2
          IP0Y=(JG-1)*NCLz+IP0X
          YC=DCEL*(JG-1)
          YC1=YC+DCEL*HALF
          DO KG=JZ1,JZ2
             IP0 = IP0Y + KG
             ZC=DCEL*(KG-1)
             ZC1=ZC+DCEL*HALF

             if(xc1.ge.LXMIN1.and.xc1.le.LXMAX1.and. &
                  yc.ge.LYMIN1.and. yc.le.LYMAX1.and. &
                  zc.ge.LZMIN1.and. zc.le.LZMAX1     ) &
                  rmayx(ip0)=EPSB
             if( xc.ge.LXMIN1.and. xc.le.LXMAX1.and. &
                  yc1.ge.LYMIN1.and.yc1.le.LYMAX1.and. &
                  zc.ge.LZMIN1.and. zc.le.LZMAX1     ) &
                  rmayy(ip0)=EPSB
             if( xc.ge.LXMIN1.and. xc.le.LXMAX1.and. &
                  yc.ge.LYMIN1.and. yc.le.LYMAX1.and. &
                  zc1.ge.LZMIN1.and.zc1.le.LZMAX1     ) &
                  rmayz(ip0)=EPSB

             !     membrane region
             IF(tmemb.gt.zero.and.QBTOM) THEN
                IF(zc.ge.zmemb1.and.zc.le.zmemb2) THEN
                   IF(zc.ge.hmemb1.and.zc.le.hmemb2) THEN
                      rmayx(ip0)=EPSM
                      rmayy(ip0)=EPSM
                   ELSE
                      rmayx(ip0)=EPSH
                      rmayy(ip0)=EPSH
                   ENDIF
                ENDIF
                IF (zc1.ge.zmemb1.and.zc1.le.zmemb2) THEN
                   IF (zc1.ge.hmemb1.and.zc1.le.hmemb2) THEN
                      rmayz(ip0)=EPSM
                   ELSE
                      rmayz(ip0)=EPSH
                   ENDIF
                ENDIF
             ENDIF

          ENDDO
       ENDDO
    ENDDO
    !
    RETURN
  END SUBROUTINE MAYER_BOX

  SUBROUTINE MAYER_ECONE(NCLX,NCLY,NCLZ,DCEL,TRANX,TRANY,TRANZ, &
       XBCEN,YBCEN,ZBCEN,KAPPA2,TMEMB,ZMEMB,EPSM,HTMEMB,EPSH, &
       EPSEC,Xcone,Ycone,Zcone,Ax,By,Cz,Qectom,Qeckap, &
       IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
       CHC,RMAY,RMAYX,RMAYY,RMAYZ,QFKAP)
    !----------------------------------------------------------------------
    !     Droplet Part
    !
    use chm_kinds
    use number
    use consta
    implicit none
    INTEGER  NCLx,NCLy,NCLz
    INTEGER  IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN
    REAL(CHM_REAL4)   CHC(*),RMAY(*),RMAYX(*),RMAYY(*),RMAYZ(*)
    real(chm_real)   DCEL,TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN
    real(chm_real)   TMEMB,ZMEMB,EPSM,HTMEMB,EPSH,KAPPA2
    real(chm_real)   EPSEC,Xcone,Ycone,Zcone,Ax,By,Cz
    real(chm_real)   EPSC,Xcyln,Ycyln,Zcyln,Rcyln,Hcyln
    LOGICAL  Qectom,Qeckap,QFKAP
    ! local
    integer  ncyz,nfil,nfilz,ix,iy,iz,jx1,jx2,jy1,jy2,jz1,jz2
    integer  k,l,m,ipx,ipy,ipz
    real(chm_real)   xi,yi,zi,bisq,zisq,xc,yc,zc,xsq,ysq,zsq, &
         dsq,dsq1,dcel2
    real(chm_real)   xc1,yc1,zc1,ax2,by2,cz2
    real(chm_real)   zmemb1,zmemb2,hmemb1,hmemb2

    ncyz=ncly*nclz
    dcel2=dcel*dcel
    zmemb1=tranz-zbcen-0.5*tmemb+zmemb-rsmall
    zmemb2=zmemb1+TMEMB+2.0*rsmall
    hmemb1=zmemb1+HTMEMB
    hmemb2=zmemb2-HTMEMB
    !
    ax2=ax*ax
    by2=by*by
    cz2=cz*cz
    xi=xcone+tranx-xbcen
    yi=ycone+trany-ybcen
    zi=zcone+tranz-zbcen
    bisq=ax
    if(by.gt.ax) bisq=by
    nfil=int(bisq/dcel)+2
    ix=int(xi/dcel)+1
    iy=int(yi/dcel)+1
    iz=int(zi/dcel)+1
    nfilz=int(cz/dcel)+2
    zisq=cz2+rsmall

    JX1=IX-NFIL+1
    IF(JX1.LT.1)JX1=1
    JX2=IX+NFIL
    IF(JX2.GT.NCLX)JX2=NCLX
    JY1=IY-NFIL+1
    IF(JY1.LT.1)JY1=1
    JY2=IY+NFIL
    IF(JY2.GT.NCLY)JY2=NCLY
    JZ1=IZ-NFILz+1
    IF(JZ1.LT.1)JZ1=1
    JZ2=IZ+NFILz
    IF(JZ2.GT.NCLZ)JZ2=NCLZ

    IF(JX2.GT.IXMAX) IXMAX=JX2+1
    IF(JY2.GT.IYMAX) IYMAX=JY2+1
    IF(JZ2.GT.IZMAX) IZMAX=JZ2+1
    IF(JX1.LT.IXMIN) IXMIN=JX1-1
    IF(JY1.LT.IYMIN) IYMIN=JY1-1
    IF(JZ1.LT.IZMIN) IZMIN=JZ1-1
    !
    do k=jx1,jx2
       ipx=(k-1)*ncyz
       xc=(k-1)*dcel
       do l=jy1,jy2
          ipy=(l-1)*nclz
          yc=(l-1)*dcel
          do m=jz1,jz2
             ipz=m+ipy+ipx
             zc=(m-1)*dcel
             xsq=(xc-xi)*(xc-xi)/ax2
             ysq=(yc-yi)*(yc-yi)/by2
             zsq=(zc-zi)*(zc-zi)/cz2+rsmall

             dsq=xsq+ysq
             IF(.not.QFKAP)THEN
                IF(dsq.le.zsq.and.zsq.le.zisq)THEN
                   rmay(ipz)=0.0 ! Debye screening factor zero
                   ! inside the elliptical cone
                   IF(QECKAP) rmay(ipz)=kappa2*dcel2
                ENDIF
             ENDIF

             IF(EPSEC.ne.0.0)THEN
                xc1=xc+0.5*dcel
                dsq1=(xc1-xi)**2/ax2+ysq
                IF(dsq1.le.zsq.and.zsq.le.zisq)THEN
                   rmayx(ipz)=EPSEC ! dielectric constant of elliptical cone
                ENDIF

                yc1=yc+0.5*dcel
                dsq1=(yc1-yi)**2/by2+xsq
                IF(dsq1.le.zsq.and.zsq.le.zisq)THEN
                   rmayy(ipz)=EPSEC
                ENDIF

                zc1=zc+0.5*dcel
                dsq1=(zc1-zi)**2/cz2
                IF(dsq.le.zsq.and.dsq1.le.zisq)THEN
                   rmayz(ipz)=EPSEC
                ENDIF

                !     membrane region
                IF(tmemb.gt.zero.and.QECTOM) THEN
                   IF(zc.ge.zmemb1.and.zc.le.zmemb2) THEN
                      IF(zc.ge.hmemb1.and.zc.le.hmemb2) THEN
                         rmayx(ipz)=EPSM
                         rmayy(ipz)=EPSM
                      ELSE
                         rmayx(ipz)=EPSH
                         rmayy(ipz)=EPSH
                      ENDIF
                   ENDIF
                   IF (zc1.ge.zmemb1.and.zc1.le.zmemb2) THEN
                      IF (zc1.ge.hmemb1.and.zc1.le.hmemb2) THEN
                         rmayz(ipz)=EPSM
                      ELSE
                         rmayz(ipz)=EPSH
                      ENDIF
                   ENDIF
                ENDIF
             ENDIF ! EPSEC.ne.0.0

          enddo
       enddo
    enddo
    !
    RETURN
  END SUBROUTINE MAYER_ECONE

  SUBROUTINE MAYER_CYLN(NCLX,NCLY,NCLZ,DCEL,TRANX,TRANY,TRANZ, &
       XBCEN,YBCEN,ZBCEN,KAPPA2,TMEMB,ZMEMB,EPSM,HTMEMB,EPSH, &
       EPSC,Xcyln,Ycyln,Zcyln,Rcyln,Hcyln,Qctom,Qckap, &
       IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
       CHC,RMAY,RMAYX,RMAYY,RMAYZ,QFKAP)
    !----------------------------------------------------------------------
    !     Cylinder Part
    !
    use chm_kinds
    use number
    use consta
    implicit none
    INTEGER  NCLx,NCLy,NCLz
    INTEGER  IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN
    REAL(CHM_REAL4)   CHC(*),RMAY(*),RMAYX(*),RMAYY(*),RMAYZ(*)
    real(chm_real)   DCEL,TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN
    real(chm_real)   TMEMB,ZMEMB,EPSM,HTMEMB,EPSH,KAPPA2
    real(chm_real)   EPSC,Xcyln,Ycyln,Zcyln,Rcyln,Hcyln
    LOGICAL  Qctom,Qckap,QFKAP
    ! local
    integer  ncyz,nfil,nfilz,ix,iy,iz,jx1,jx2,jy1,jy2,jz1,jz2
    integer  k,l,m,ipx,ipy,ipz
    real(chm_real)   xi,yi,zi,bisq,zisq,xc,yc,zc,xsq,ysq,zsq, &
         dsq,dsq1,dcel2
    real(chm_real)   xc1,yc1,zc1
    real(chm_real)   zmemb1,zmemb2,hmemb1,hmemb2

    ncyz=ncly*nclz
    dcel2=dcel*dcel
    zmemb1=tranz-zbcen-0.5*tmemb+zmemb-rsmall
    zmemb2=zmemb1+TMEMB+2.0*rsmall
    hmemb1=zmemb1+HTMEMB
    hmemb2=zmemb2-HTMEMB
    !
    xi=xcyln+tranx-xbcen
    yi=ycyln+trany-ybcen
    zi=zcyln+tranz-zbcen
    bisq=rcyln
    nfil=int(bisq/dcel)+2
    bisq=bisq*bisq+rsmall
    ix=int(xi/dcel)+1
    iy=int(yi/dcel)+1
    iz=int(zi/dcel)+1
    nfilz=int(half*hcyln/dcel)+2
    zisq=hcyln*hcyln/four+rsmall

    JX1=IX-NFIL+1
    IF(JX1.LT.1)JX1=1
    JX2=IX+NFIL
    IF(JX2.GT.NCLX)JX2=NCLX
    JY1=IY-NFIL+1
    IF(JY1.LT.1)JY1=1
    JY2=IY+NFIL
    IF(JY2.GT.NCLY)JY2=NCLY
    JZ1=IZ-NFILz+1
    IF(JZ1.LT.1)JZ1=1
    JZ2=IZ+NFILz
    IF(JZ2.GT.NCLZ)JZ2=NCLZ

    IF(JX2.GT.IXMAX) IXMAX=JX2+1
    IF(JY2.GT.IYMAX) IYMAX=JY2+1
    IF(JZ2.GT.IZMAX) IZMAX=JZ2+1
    IF(JX1.LT.IXMIN) IXMIN=JX1-1
    IF(JY1.LT.IYMIN) IYMIN=JY1-1
    IF(JZ1.LT.IZMIN) IZMIN=JZ1-1
    !
    do k=jx1,jx2
       ipx=(k-1)*ncyz
       xc=(k-1)*dcel
       do l=jy1,jy2
          ipy=(l-1)*nclz
          yc=(l-1)*dcel
          do m=jz1,jz2
             ipz=m+ipy+ipx
             zc=(m-1)*dcel
             xsq=(xc-xi)*(xc-xi)
             ysq=(yc-yi)*(yc-yi)
             zsq=(zc-zi)*(zc-zi)

             dsq=xsq+ysq
             IF(.not.QFKAP)THEN
                IF(dsq.le.bisq.and.zsq.le.zisq)THEN
                   rmay(ipz)=0.0 ! Debye screening factor zero
                   ! inside the cylinder
                   IF(QCKAP) rmay(ipz)=kappa2*dcel2
                ENDIF
             ENDIF

             IF(EPSC.ne.0.0)THEN
                xc1=xc+0.5*dcel
                dsq1=(xc1-xi)**2+ysq
                IF(dsq1.le.bisq.and.zsq.le.zisq)THEN
                   rmayx(ipz)=EPSC ! dielectric constant of cylinder cavity
                ENDIF

                yc1=yc+0.5*dcel
                dsq1=(yc1-yi)**2+xsq
                IF(dsq1.le.bisq.and.zsq.le.zisq)THEN
                   rmayy(ipz)=EPSC
                ENDIF

                zc1=zc+0.5*dcel
                dsq1=(zc1-zi)**2
                IF(dsq.le.bisq.and.dsq1.le.zisq)THEN
                   rmayz(ipz)=EPSC
                ENDIF

                !     membrane region
                IF(tmemb.gt.zero.and.QCTOM) THEN
                   IF(zc.ge.zmemb1.and.zc.le.zmemb2) THEN
                      IF(zc.ge.hmemb1.and.zc.le.hmemb2) THEN
                         rmayx(ipz)=EPSM
                         rmayy(ipz)=EPSM
                      ELSE
                         rmayx(ipz)=EPSH
                         rmayy(ipz)=EPSH
                      ENDIF
                   ENDIF
                   IF (zc1.ge.zmemb1.and.zc1.le.zmemb2) THEN
                      IF (zc1.ge.hmemb1.and.zc1.le.hmemb2) THEN
                         rmayz(ipz)=EPSM
                      ELSE
                         rmayz(ipz)=EPSH
                      ENDIF
                   ENDIF
                ENDIF
             ENDIF ! EPSC.ne.0.0

          enddo
       enddo
    enddo
    !
    RETURN
  END SUBROUTINE MAYER_CYLN

  SUBROUTINE MAYER_REEN(NTPRP,LSTPRP,X,Y,Z,PBRAD,EPSP,RADW, &
       NCLX,NCLY,NCLZ,DCEL, &
       TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
       RMAY,RMAYX,RMAYY,RMAYZ,QFKAP, &
       POX,POY,POZ,MAPT)
    !------------------------------------------------------------------------
    !     Molecular (contact+reentrance) surface
    !
    use clcg_mod,only:clcginit
    use chm_kinds
    use memory
    use number
    use consta
    use parallel
    use rndnum
    implicit none
    INTEGER  NTPRP,LSTPRP(*)
    INTEGER  NCLX,NCLY,NCLZ,MAPT
    REAL(CHM_REAL4)   RMAY(*),RMAYX(*),RMAYY(*),RMAYZ(*)
    real(chm_real)   X(*),Y(*),Z(*),PBRAD(*),POX(*),POY(*),POZ(*)
    real(chm_real)   EPSP,RADW,DCEL,TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN
    LOGICAL  QFKAP
    ! local
    integer, allocatable, dimension (:) :: LISTR
    integer  iseed,il,jl,i,j,l,m,nl,ml,kl,ll,nfil,ix,iy,iz
    integer  jx1,jx2,jy1,jy2,jz1,jz2,k,ipx,ipy,ipz,ncyz
    real(chm_real), allocatable, dimension(:) :: FISTR
    real(chm_real)   wrsf,raneti,ranetj,xi,yi,zi,xj,yj,zj,aij,fistr0
    real(chm_real)   xp,yp,zp,bisq,xc,yc,zc,xsq,ysq,zsq,dsq, &
         xc1,yc1,zc1,dsq1

    ! 
    call chmalloc('pbeq.src','MAYER_REEN','LISTR',NTPRP,intg=LISTR)
    call chmalloc('pbeq.src','MAYER_REEN','FISTR',NTPRP,crl=FISTR)

    iseed=314159265
    ncyz=ncly*nclz
    if (.not.qoldrng) then             !yw 05-Aug-2008
       CALL CLCGINIT(ISEED)
       ISEED=1
    endif
    !
    ! generate points on the sphere surface and adjust parameter
    wrsf=0.
    do il=1,ntprp
       i=lstprp(il)
       raneti=pbrad(i)+radw
       if(wrsf.lt.raneti)wrsf=raneti
    enddo
    wrsf=MAPT/wrsf**2 + 0.000001
    do i=1,MAPT
       call genpoint(pox(i),poy(i),poz(i),iseed)
    enddo

    ! main loop by atoms
    do il=1,ntprp
       nl=0
       i=lstprp(il)
       xi=x(i)
       yi=y(i)
       zi=z(i)
       raneti=pbrad(i)+radw

       ! create the list of closest neighbors for the atom from the main loop
       ! according to their screening ability of the neighbors
       ! 0<Segpar<1
       ! Segpar=0  if the neighbor does not reduce the accessible surface of the atom
       !           and it means that it is not a neighbor at all.
       ! Segpar=1  if the neighbor buries the atom totally.
       do jl=1,ntprp
          if(il.eq.jl)goto 61
          j=lstprp(jl)
          xj=x(j)
          yj=y(j)
          zj=z(j)
          ranetj=pbrad(j)+radw
          aij=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj))
          fistr0=segpar(raneti,ranetj,aij)
          if(fistr0.eq.0.)goto 61
          if(fistr0.eq.1.)goto 60
          nl=nl+1
          fistr(nl)=fistr0
          listr(nl)=j
61        continue
       enddo

       ! sort neighbors according to their ability to
       ! reduce an accessible surface of the atom
       if(nl.gt.0)call qcksrt(nl,fistr,listr)
       ml=int(wrsf*raneti**2)
       if(ml.gt.MAPT)ml=MAPT
       ! loop over points on the sphere surface
       do jl=1,ml
          xp=xi+raneti*pox(jl)
          yp=yi+raneti*poy(jl)
          zp=zi+raneti*poz(jl)
          ! reject the points which are not on the accessible surface.
          if(nl.gt.0)then
             do kl=1,nl
                ll=listr(kl)
                aij=(xp-x(ll))**2+(yp-y(ll))**2+(zp-z(ll))**2
                if(aij.lt.(pbrad(ll)+radw)**2)goto 62
             enddo
          endif

          ! reset grid points inside a probe sphere around any
          ! of random accessible points as a dielectric media.
          bisq=radw
          nfil=int(bisq/dcel)+2
          xp=xp+tranx-xbcen
          yp=yp+trany-ybcen
          zp=zp+tranz-zbcen
          bisq=bisq*bisq + rsmall

          ix=int(xp/dcel)+1
          iy=int(yp/dcel)+1
          iz=int(zp/dcel)+1

          jx1=ix-nfil+1
          if(jx1.lt.1)jx1=1
          jx2=ix+nfil
          if(jx2.gt.nclx)jx2=nclx
          jy1=iy-nfil+1
          if(jy1.lt.1)jy1=1
          jy2=iy+nfil
          if(jy2.gt.ncly)jy2=ncly
          jz1=iz-nfil+1
          if(jz1.lt.1)jz1=1
          jz2=iz+nfil
          if(jz2.gt.nclz)jz2=nclz
          !
          do k=jx1,jx2
             ipx=(k-1)*ncyz
             xc =(k-1)*dcel
             xsq=(xc-xp)*(xc-xp)
             do l=jy1,jy2
                ipy=(l-1)*nclz
                yc =(l-1)*dcel
                ysq=(yc-yp)*(yc-yp)
                do m=jz1,jz2
                   ipz=m+ipy+ipx
                   zc=(m-1)*dcel
                   zsq=(zc-zp)*(zc-zp)
                   dsq=xsq+ysq+zsq

                   ! if RADI > 0 it never happens
                   IF(.not.QFKAP)THEN
                      if(rmay(ipz).lt.0.)then
                         if(dsq.le.bisq)then
                            rmay(ipz)=-rmay(ipz) ! bulk kappa restored
                         endif
                      endif
                   ENDIF

                   if(rmayx(ipz).lt.0.)then
                      xc1=xc+0.5*dcel
                      dsq1=(xc1-xp)**2+ysq+zsq
                      if(dsq1.le.bisq)then
                         rmayx(ipz)=-rmayx(ipz) ! bulk dielectric
                         ! constant restored
                      endif
                   endif

                   if(rmayy(ipz).lt.0.)then
                      yc1=yc+0.5*dcel
                      dsq1=(yc1-yp)**2+xsq+zsq
                      if(dsq1.le.bisq)then
                         rmayy(ipz)=-rmayy(ipz) ! bulk dielectric
                         ! constant restored
                      endif
                   endif

                   if(rmayz(ipz).lt.0.)then
                      zc1=zc+0.5*dcel
                      dsq1=(zc1-zp)**2+ysq+xsq
                      if(dsq1.le.bisq)then
                         rmayz(ipz)=-rmayz(ipz) ! bulk dielectric
                         ! constant restored
                      endif
                   endif

                enddo
             enddo
          enddo
62        continue
       enddo
60     continue
    enddo
    !
    do k=1,nclx
       ipx=(k-1)*ncyz
       do l=1,ncly
          ipy=(l-1)*nclz
          do m=1,nclz
             ipz=m+ipy+ipx
             IF(rmayz(ipz).lt.0.)rmayz(ipz)=epsp
             IF(rmayy(ipz).lt.0.)rmayy(ipz)=epsp
             IF(rmayx(ipz).lt.0.)rmayx(ipz)=epsp
             IF(rmay(ipz).lt.0.)rmay(ipz)=0.
          enddo
       enddo
    enddo
    !
    call chmdealloc('pbeq.src','MAYER_REEN','LISTR',NTPRP,intg=LISTR)
    call chmdealloc('pbeq.src','MAYER_REEN','FISTR',NTPRP,crl=FISTR)
    !
    RETURN
  END SUBROUTINE MAYER_REEN

  !mf, read dielectric distribution (epsx/epsy/epsz)  
  !    from external file (unit given in EPSU)
  !    format: 
  !       x y z epsx epsy epsz 
  !       ...
  !    number of lines has to match grid setup otherwise
  !
  SUBROUTINE MAYER_REPS(NCLX,NCLY,NCLZ,DCEL,TRANX,TRANY,TRANZ, &
                        XBCEN,YBCEN,ZBCEN,EPSU,RMAYX,RMAYY,RMAYZ)
    use chm_kinds
    use memory
    use number
    use consta
    implicit none

    INTEGER  NCLx,NCLy,NCLz
    INTEGER  EPSU
    REAL(CHM_REAL4)  RMAYX(*),RMAYY(*),RMAYZ(*)
    REAL(chm_real)   DCEL,TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN
    ! local
    integer ncyz,ipz,ix,iy,iz 
    integer iostatus

    REAL(chm_real)   xval,yval,zval,epsx,epsy,epsz,xi,yi,zi
    LOGICAL done

    done=.false. 

    ncyz=ncly*nclz

    DO WHILE (.not. done)
      READ(EPSU,2003,IOSTAT=iostatus) xval,yval,zval,epsx,epsy,epsz     
2003  format(3f10.5,3e16.7)
    
      IF (iostatus .eq. 0) THEN 
        xi=xval+tranx-xbcen
        ix=int(xi/dcel+0.5)+1
        if (ix.ge.1 .and. ix.le.nclx) THEN
           yi=yval+trany-ybcen
           iy=int(yi/dcel+0.5)+1
           if (iy.ge.1 .and. iy.le.ncly) THEN
              zi=zval+tranz-zbcen
              iz=int(zi/dcel+0.5)+1
              if (iz.ge.1 .and. iz.le.nclz) THEN
                 ipz=iz+(iy-1)*nclz+(ix-1)*ncyz
                 rmayx(ipz)=epsx
                 rmayy(ipz)=epsy
                 rmayz(ipz)=epsz  
              endif
           endif
        endif   
      ELSE
        done=.true.
      ENDIF
    enddo

    close(epsu)
    RETURN
  END SUBROUTINE MAYER_REPS

  !mf, read dielectric grid (eps)  
  !    from external file (unit given in EPSG)
  !    grid is then used to fill in PBEQ grid 
  !    format: 
  !       NX NY NZ
  !       XMIN YMIN ZMIN
  !       DELTAX DELTAY DELTAZ
  !       epsx epsy epsz
  !       ...
  !
  ! MAYER_REPSG reads the header and allocates temporary memory
  ! MAYER_REPSG_RD reads the actual data
  !
  SUBROUTINE MAYER_REPSG(NCLX,NCLY,NCLZ,DCEL,TRANX,TRANY,TRANZ, &
                         XBCEN,YBCEN,ZBCEN,EPSG,RMAYX,RMAYY,RMAYZ)
    use chm_kinds
    use memory
    use number
    use consta
    use stream
    implicit none

    INTEGER  NCLx,NCLy,NCLz
    INTEGER  EPSG
    REAL(chm_real4)  RMAYX(*),RMAYY(*),RMAYZ(*)
    REAL(chm_real)   DCEL,TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN

    REAL (chm_real), allocatable, dimension (:) :: egridx,egridy,egridz

    ! local
    integer nx,ny,nz
    real(chm_real) xmin,ymin,zmin,dx,dy,dz

    read(epsg,*) nx,ny,nz
    read(epsg,*) xmin,ymin,zmin
    read(epsg,*) dx,dy,dz

    IF (PRNLEV.GE.2) THEN
      write(OUTU,*) 'reading grid data:' 
      write(OUTU,*) 'dimensions : ',nx,ny,nz
      write(OUTU,*) 'min. coord.:',xmin,ymin,zmin 
      write(OUTU,*) 'spacing    :',dx,dy,dz
    ENDIF

    call chmalloc('pbeq.src','MAYER_REPSG','EGRIDX',NX*NY*NZ,crl=EGRIDX)
    call chmalloc('pbeq.src','MAYER_REPSG','EGRIDY',NX*NY*NZ,crl=EGRIDY)
    call chmalloc('pbeq.src','MAYER_REPSG','EGRIDZ',NX*NY*NZ,crl=EGRIDZ)

    CALL MAYER_REPSG_RD(EGRIDX,EGRIDY,EGRIDZ, &
                        nx,ny,nz,xmin,ymin,zmin,dx,dy,dz, &
                        NCLX,NCLY,NCLZ,DCEL,TRANX,TRANY,TRANZ, &
                        XBCEN,YBCEN,ZBCEN,RMAYX,RMAYY,RMAYZ,EPSG)

    call chmdealloc('pbeq.src','MAYER_REPSG','EGRIDX',NX*NY*NZ,crl=EGRIDX)
    call chmdealloc('pbeq.src','MAYER_REPSG','EGRIDY',NX*NY*NZ,crl=EGRIDY)
    call chmdealloc('pbeq.src','MAYER_REPSG','EGRIDZ',NX*NY*NZ,crl=EGRIDZ)

    close(epsg)

    RETURN
  END SUBROUTINE MAYER_REPSG


  SUBROUTINE MAYER_REPSG_RD(EGX,EGY,EGZ,NX,NY,NZ, &
           XMIN,YMIN,ZMIN,DX,DY,DZ, &
           NCLX,NCLY,NCLZ,DCEL,TRANX,TRANY,TRANZ, &
           XBCEN,YBCEN,ZBCEN, RMAYX,RMAYY,RMAYZ,EPSG)
    use chm_kinds
    use memory
    use number
    use consta
    implicit none

    INTEGER  NX,NY,NZ
    INTEGER  NCLx,NCLy,NCLz,EPSG
    real(chm_real4)  RMAYX(*),RMAYY(*),RMAYZ(*)
    real(chm_real)   xmin,ymin,zmin,dx,dy,dz 
    real(chm_real)   dcel,tranx,trany,tranz,xbcen,ybcen,zbcen
    real(chm_real)   EGX(*),EGY(*),EGZ(*)

    !local
    INTEGER ix,iy,iz,xi,yi,zi,inx
    INTEGER xnx,xny,xnz,ynx,yny,ynz,znx,zny,znz
    INTEGER xid,yid,zid,xid2,yid2,zid2
    INTEGER xi2,yi2,zi2
    REAL(chm_real) xval,yval,zval

    DO inx=1,nx*ny*nz
      READ(EPSG,*) EGX(inx),EGY(inx),EGZ(inx)
    ENDDO

    DO ix=1,NCLX
       xval=(ix-1)*dcel-tranx+xbcen
       xi=INT((xval-xmin)/dx+0.5)+1
       IF (xi.lt.1) xi=1
       IF (xi.gt.nx) xi=nx

       xi2=INT((xval-xmin-dx/2.0)/dx+0.5)+1
       IF (xi2.lt.1) xi2=1
       IF (xi2.gt.nx) xi2=nx

       xid=INT((xval-xmin+dcel/2.0)/dx+0.5)+1
       IF (xid.lt.1) xid=1
       IF (xid.gt.nx) xid=nx

       xid2=INT((xval-xmin+dcel/2.0-dx/2.0)/dx+0.5)+1
       IF (xid2.lt.1) xid2=1
       IF (xid2.gt.nx) xid2=nx

       DO iy=1,NCLY
          yval=(iy-1)*dcel-trany+ybcen
          yi=INT((yval-ymin)/dy+0.5)+1
          IF (yi.lt.1) yi=1
          IF (yi.gt.ny) yi=ny

          yi2=INT((yval-ymin-dy/2.0)/dy+0.5)+1
          IF (yi2.lt.1) yi2=1
          IF (yi2.gt.ny) yi2=ny

          yid=INT((yval-ymin+dcel/2.0)/dy+0.5)+1
          IF (yid.lt.1) yid=1
          IF (yid.gt.ny) yid=ny

          yid2=INT((yval-ymin+dcel/2.0-dy/2.0)/dy+0.5)+1
          IF (yid2.lt.1) yid2=1
          IF (yid2.gt.ny) yid2=ny

          DO iz=1,NCLZ
             zval=(iz-1)*dcel-tranz+zbcen
             zi=INT((zval-zmin)/dz+0.5)+1
             IF (zi.lt.1) zi=1
             IF (zi.gt.nz) zi=nz

             zi2=INT((zval-zmin-dz/2.0)/dz+0.5)+1
             IF (zi2.lt.1) zi2=1
             IF (zi2.gt.nz) zi2=nz

             zid=INT((zval-zmin+dcel/2.0)/dz+0.5)+1
             IF (zid.lt.1) zid=1
             IF (zid.gt.nz) zid=nz
             
             zid2=INT((zval-zmin+dcel/2.0-dz/2.0)/dz+0.5)+1
             IF (zid2.lt.1) zid2=1
             IF (zid2.gt.nz) zid2=nz
             
             inx=(ix-1)*ncly*nclz+(iy-1)*nclz+iz

             xnx=(xid2-1)*ny*nz+(yi-1)*nz+zi
             yny=(xi-1)*ny*nz+(yid2-1)*nz+zi
             znz=(xi-1)*ny*nz+(yi-1)*nz+zid2

             rmayx(inx)=egx(xnx)
             rmayy(inx)=egy(yny)
             rmayz(inx)=egz(znz)

          ENDDO
       ENDDO
    ENDDO
    RETURN
  END SUBROUTINE MAYER_REPSG_RD


  SUBROUTINE MAYER_VMEMB(NCLX,NCLY,NCLZ,DCEL,TRANX,TRANY,TRANZ, &
       ZBCEN,EPSW,KAPPA2,VMEMB,TMEMB,ZMEMB,EPSM, &
       CHC,PHI,RMAY,QFOCUS,QKEEP,QNONLINEAR,QPARTLINEAR)
    !------------------------------------------------------------------------
    !     Membrane Potential
    !
    use chm_kinds
    use number
    use consta
    implicit none
    INTEGER  NCLx,NCLy,NCLz
    REAL(CHM_REAL4)   CHC(*),PHI(*),RMAY(*)
    real(chm_real)   DCEL,TRANX,TRANY,TRANZ,ZBCEN
    real(chm_real)   EPSW,KAPPA2,VMEMB,TMEMB,ZMEMB,EPSM
    LOGICAL  QFOCUS,QKEEP,QNONLINEAR,QPARTLINEAR
    ! local
    integer  ncyz,kg,ig,iii,jg,ipo,iiz,iix,iiix,iiy
    real(chm_real)   zmemb1,zmemb2,kappa,afact,zc,phic
    !
    ncyz=ncly*nclz
    kappa=sqrt(kappa2/EPSW)
    zmemb1=tranz-zbcen-0.5*tmemb+zmemb-rsmall
    zmemb2=zmemb1+TMEMB+2.0*rsmall
    !
    IF(.not.QFOCUS.and..not.QKEEP) THEN
       AFACT = VMEMB/(2.0+(EPSW/EPSM)*KAPPA*TMEMB)
       do kg=1,nclz           !,ncel-1
          zc = (kg-1)*dcel
          if(zc.lt.zmemb1)then
             phic = afact*exp(kappa*(zc-zmemb1))
          elseif((zc.ge.zmemb1).and.(zc.le.zmemb2))then
             phic = afact*((epsw/epsm)*kappa*(zc-zmemb1)+1.0)
          elseif(zc.gt.zmemb2)then
             phic = vmemb-afact*exp(-kappa*(zc-zmemb2))
          endif
          do ig=1,nclx
             iii=(ig-1)*ncyz+kg
             do jg=1,ncly
                ipo=iii+(jg-1)*nclz
                phi(ipo)=phi(ipo)+phic
             enddo
          enddo
       enddo
    ENDIF

    IF(.not.QNONLINEAR.and..not.QPARTLINEAR) THEN
       ! construct the background charge density for transmembrane potential
       do iiz=1,nclz
          zc=(iiz-1)*dcel
          IF(ZC.GT.ZMEMB2)THEN
             do iix=1,nclx
                iiix=(iix-1)*ncyz
                do iiy=1,ncly
                   iii=iiix+(iiy-1)*nclz+iiz
                   chc(iii)=chc(iii)+rmay(iii)*VMEMB
                enddo
             enddo
          ENDIF
       enddo
    ENDIF
    !
    RETURN
  END SUBROUTINE MAYER_VMEMB

  SUBROUTINE MAYER_MEMB(NCLX,NCLY,NCLZ,DCEL,TRANX,TRANY,TRANZ, &
       ZBCEN,TMEMB,ZMEMB,EPSM,HTMEMB,EPSH, &
       HDENS,POSALP,NEGALP,QCHRG,POFFST,NOFFST, &
       CHC,RMAY,RMAYX,RMAYY,RMAYZ,QFKAP)
    !----------------------------------------------------------------------
    !     Membrane Part
    !
    !              <-tmemb->
    !    |       hmemb1 hmemb2
    !    |           |   |              |
    !    |           v   v              |
    !    |         |||||||||            |
    !    | phi=0   |||||||||  phi=VMEMB |
    !    |         |||||||||            |
    !    |         ^   ^   ^            |
    !    |         |   |   |            |
    !    |      zmemb1 |   zmemb2       |
    ! Vmemb1         zmemb              Vmemb2
    !
    !
    use chm_kinds
    use number
    use consta
    implicit none
    INTEGER  NCLx,NCLy,NCLz
    REAL(CHM_REAL4)   CHC(*),RMAY(*),RMAYX(*),RMAYY(*),RMAYZ(*)
    real(chm_real)   DCEL,TRANX,TRANY,TRANZ,ZBCEN
    real(chm_real)   TMEMB,ZMEMB,EPSM,HTMEMB,EPSH
    real(chm_real)   HDENS,POSALP,NEGALP,NOFFST,POFFST
    LOGICAL  QCHRG,QFKAP
    ! local
    integer  k,l,m,ipx,ipy,ipz,ncyz
    integer  iiz,iix,iiix,iiy,iii
    real(chm_real)   zmemb1,zmemb2,hmemb1,hmemb2,gpos,gneg,zcpos
    real(chm_real)   zc,zc1,s

    ncyz=ncly*nclz
    !wi      zmemb1=tranz+zbcen-0.5*tmemb+zmemb
    zmemb1=tranz-zbcen-0.5*tmemb+zmemb-rsmall
    zmemb2=zmemb1+TMEMB+2.0*rsmall
    hmemb1=zmemb1+HTMEMB
    hmemb2=zmemb2-HTMEMB

    !      write(6,'(a,4e20.15)') 'zmemb1, hmemb1, hmemb2, zmemb2',
    !     $     zmemb1, hmemb1, hmemb2, zmemb2

    do k=1,nclx
       ipx=(k-1)*ncyz
       do l=1,ncly
          ipy=(l-1)*nclz
          do m=1,nclz
             ipz=m+ipy+ipx
             zc=(m-1)*dcel

             IF(zc.ge.zmemb1.and.zc.le.zmemb2) THEN
                !     Debye screening factor zero inside the membrane
                IF(.not.QFKAP) rmay(ipz)=0.0
                !     membrane dielectric constant
                IF(zc.ge.hmemb1.and.zc.le.hmemb2) THEN
                   rmayx(ipz)=EPSM
                   rmayy(ipz)=EPSM
                ELSE
                   rmayx(ipz)=EPSH
                   rmayy(ipz)=EPSH
                ENDIF
             ENDIF
             zc1=zc+0.5*dcel
             IF(zc1.ge.zmemb1.and.zc1.le.zmemb2) THEN
                IF(zc1.ge.hmemb1.and.zc1.le.hmemb2) THEN
                   rmayz(ipz)=EPSM
                ELSE
                   rmayz(ipz)=EPSH
                ENDIF
             ENDIF
          enddo
       enddo
    enddo

    !***clbiii modification to add head-group layer
    if(qchrg) then
       do iiz=1,nclz
          zc=(iiz-1)*dcel
          do iix=1,nclx
             iiix=(iix-1)*ncyz
             do iiy=1,ncly
                iii=iiix+(iiy-1)*nclz+iiz

                zcpos = ( hmemb1 + zmemb1 ) / two

                s = posalp*( zc - zcpos - poffst )
                gpos = simp(s, posalp*dcel/two) * dcel

                s = negalp*( zc - zcpos - noffst )
                gneg = simp(s, negalp*dcel/two) * dcel

                zcpos = ( hmemb2 + zmemb2 ) / two

                s = posalp*( zc - zcpos - poffst )
                gpos = gpos + simp(s, posalp*dcel/two) * dcel

                s = negalp*( zc - zcpos - noffst )
                gneg = gneg + simp(s, negalp*dcel/two) * dcel

                gpos = hdens*gpos*posalp/sqrt(pi)
                gneg = -hdens*gneg*negalp/sqrt(pi)
                chc(iii) = chc(iii) + (gpos + gneg)
             enddo
          enddo
       enddo
    endif
    !***endof clbiii modification
    !
    RETURN
  END SUBROUTINE MAYER_MEMB

  SUBROUTINE MAYER_CD_BSPL(NTPRP,LSTPRP,X,Y,Z,CG,CHC,KAPPA2, &
       NCLX,NCLY,NCLZ,DCEL,TRANX,TRANY,TRANZ, &
       XBCEN,YBCEN,ZBCEN, &
       RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
       RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
       MAJOR,MINOR,QPROLATE, &
       OCHC,ONCLX,ONCLY,ONCLZ,ODCEL,OXBCEN,OYBCEN,OZBCEN, &
       QPBC,QA,QB)
    !----------------------------------------------------------------------
    !     the 3rd-order B-splines interpolation for charge distribution
    !
    use chm_kinds
    use number
    use consta
    implicit none
    INTEGER  NTPRP,LSTPRP(*)
    INTEGER  NCLX,NCLY,NCLZ,ONCLX,ONCLY,ONCLZ
    REAL(CHM_REAL4)   OCHC(*),CHC(*)
    real(chm_real)   X(*),Y(*),Z(*),CG(*),DCEL
    real(chm_real)   ODCEL,OXBCEN,OYBCEN,OZBCEN
    real(chm_real)   TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN,KAPPA2
    real(chm_real)   RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX
    real(chm_real)   SRDIST,RRXCEN,RRYCEN,RRZCEN
    real(chm_real)   MAJOR,MINOR
    LOGICAL  QPBC,QA,QB,QRECTBOX,QSPHERE,QPROLATE
    ! local
    integer  il,i,jx1,jx2,jy1,jy2,jz1,jz2,k,l,m,n,nc3
    integer  nfil,ix,iy,iz,ipx,ipy,ipz,ncyz,oncyz
    real(chm_real)   otranx,otrany,otranz
    real(chm_real)   chi,xi,yi,zi,xc,yc,zc,ai,bi,ci,fi,qtot
    real(chm_real)   r2,sr2,xdiff,ydiff,zdiff
    real(chm_real)   major2,minor2
    !
    ncyz=ncly*nclz
    oncyz=oncly*onclz
    otranx=half*(onclx-1)*odcel
    otrany=half*(oncly-1)*odcel
    otranz=half*(onclz-1)*odcel
    il_loop: do il=1,ntprp
       IF(QA.AND.QB) THEN
          ! to consider the boundary conditions of basis set charge distributions
          chi=ochc(il)*(odcel/two/twopi)
          if(chi.eq.zero) cycle il_loop
          l=int((il-1)/oncyz)+1
          m=int(mod((il-1),oncyz)/onclz)+1
          n=mod(mod((il-1),oncyz),onclz)+1
          xi=(l-1)*odcel-otranx+oxbcen+tranx-xbcen
          yi=(m-1)*odcel-otrany+oybcen+trany-ybcen
          zi=(n-1)*odcel-otranz+ozbcen+tranz-zbcen
       ELSE
          i=lstprp(il)
          chi=CG(i)
          IF(CHI.EQ.ZERO) cycle il_loop
          IF(QRECTBOX)THEN
             IF(QA.AND.(X(I).LT.RBXMIN.OR.X(I).GT.RBXMAX.OR. &
                  Y(I).LT.RBYMIN.OR.Y(I).GT.RBYMAX.OR. &
                  Z(I).LT.RBZMIN.OR.Z(I).GT.RBZMAX))  &
                  cycle il_loop
             IF(QB.AND.(X(I).GE.RBXMIN.AND.X(I).LE.RBXMAX.AND. &
                  Y(I).GE.RBYMIN.AND.Y(I).LE.RBYMAX.AND. &
                  Z(I).GE.RBZMIN.AND.Z(I).LE.RBZMAX))  &
                  cycle il_loop
          ELSEIF(QSPHERE)THEN
             XDIFF=X(I)-RRXCEN
             YDIFF=Y(I)-RRYCEN
             ZDIFF=Z(I)-RRZCEN
             SR2=SRDIST*SRDIST
             R2=XDIFF*XDIFF+YDIFF*YDIFF+ZDIFF*ZDIFF
             IF(QA.AND.R2.GT.SR2) cycle il_loop
             IF(QB.AND.R2.LE.SR2) cycle il_loop
          ELSEIF(QPROLATE)THEN
             XDIFF=X(I)-RRXCEN
             YDIFF=Y(I)-RRYCEN
             ZDIFF=Z(I)-RRZCEN
             MAJOR2=MAJOR*MAJOR
             MINOR2=MINOR*MINOR
             R2=(XDIFF*XDIFF+YDIFF*YDIFF)/MINOR2+ZDIFF*ZDIFF/MAJOR2
             IF(QA.AND.R2.GT.ONE) cycle il_loop
             IF(QB.AND.R2.LE.ONE) cycle il_loop
          ENDIF
          xi=x(i)+tranx-xbcen
          yi=y(i)+trany-ybcen
          zi=z(i)+tranz-zbcen
          if(xi.lt.zero .or. xi.gt.2*tranx .or.         & ! for a charge
               yi.lt.zero .or. yi.gt.2*trany .or.         & ! outside a box
               zi.lt.zero .or. zi.gt.2*tranz) cycle il_loop   ! in focussing
       ENDIF

       nfil = 1               ! to consider neighboring grids
       ! x-1 .. x .. x+1 .. x+2
       ix = int(xi/dcel)+1
       iy = int(yi/dcel)+1
       iz = int(zi/dcel)+1

       JX1=IX-NFIL
       IF(JX1.LT.1)JX1=1
       JX2=IX+NFIL+1
       IF(JX2.GT.NCLX)JX2=NCLX
       JY1=IY-NFIL
       IF(JY1.LT.1)JY1=1
       JY2=IY+NFIL+1
       IF(JY2.GT.NCLY)JY2=NCLY
       JZ1=IZ-NFIL
       IF(JZ1.LT.1)JZ1=1
       JZ2=IZ+NFIL+1
       IF(JZ2.GT.NCLZ)JZ2=NCLZ

       DO K=jx1,jx2
          IPX=(K-1)*NCyz
          XC=(K-1)*DCEL
          DO L=jy1,jy2
             IPY=(L-1)*nclz+IPX
             YC=(L-1)*DCEL
             DO M=jz1,jz2
                IPZ=M+IPY
                ZC=(M-1)*DCEL

                !
                !     If someone wants to use higher order B-splines, change 1.5 !!!
                !
                ai=1.5-(xi-xc)/dcel
                bi=1.5-(yi-yc)/dcel
                ci=1.5-(zi-zc)/dcel
                fi=M3(ai)*M3(bi)*M3(ci)

                chc(ipz)=chc(ipz)+fi*chi*TWO*TWOPI/DCEL

             enddo
          enddo
       enddo
    enddo il_loop
    !
    ! Construct homogeneous neturalizing background charge if kappa=0
    ! q(i,j,k) = 4 * pi * QTOT / (Lx*Ly*Lz) * dcel2 [e/angs] for finite-difference solver
    !
    IF(QPBC.AND.KAPPA2.EQ.0.0) THEN
       QTOT=ZERO
       IF(QA.AND.QB) THEN
          return
       ELSE
          do il=1,ntprp
             i=lstprp(il)
             QTOT=QTOT+CG(i)
          enddo
       ENDIF
       QTOT=QTOT/(NCLX*NCLY*NCLZ*DCEL*DCEL*DCEL)
       NC3=NCyz*NCLx
       DO il=1,nc3
          !wi            chc(il)=chc(il)-QTOT*TWO*TWOPI/DCEL
          chc(il)=chc(il)-QTOT*TWO*TWOPI*DCEL*DCEL
       ENDDO
    ENDIF
    !
    RETURN
  END SUBROUTINE MAYER_CD_BSPL

  SUBROUTINE MAYER_CD_TRIL(NTPRP,LSTPRP,X,Y,Z,CG,CHC,KAPPA2, &
       NCLX,NCLY,NCLZ,DCEL,TRANX,TRANY,TRANZ, &
       XBCEN,YBCEN,ZBCEN, &
       RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
       RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
       MAJOR,MINOR,QPROLATE, &
       OCHC,ONCLX,ONCLY,ONCLZ,ODCEL,OXBCEN,OYBCEN,OZBCEN, &
       QPBC,QA,QB)
    !----------------------------------------------------------------------
    !     the trilinear interpolation for charge distribution
    !
    use chm_kinds
    use number
    use consta
    implicit none
    INTEGER  NTPRP,LSTPRP(*)
    INTEGER  NCLX,NCLY,NCLZ,ONCLX,ONCLY,ONCLZ
    REAL(CHM_REAL4)   OCHC(*),CHC(*)
    real(chm_real)   X(*),Y(*),Z(*),CG(*),DCEL
    real(chm_real)   TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN,KAPPA2
    real(chm_real)   ODCEL,OXBCEN,OYBCEN,OZBCEN
    real(chm_real)   RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX
    real(chm_real)   SRDIST,RRXCEN,RRYCEN,RRZCEN
    real(chm_real)   MAJOR,MINOR
    LOGICAL  QPBC,QA,QB,QRECTBOX,QSPHERE,QPROLATE
    ! local
    integer  il,i,n1,n2,n3,in1,in2,in3,l,m,n,nc3
    integer  nfil,ix,iy,iz,ncyz,oncyz
    real(chm_real)   otranx,otrany,otranz
    real(chm_real)   chi,xi,yi,zi,ai,bi,ci,fi,qtot
    real(chm_real)   r2,sr2,xdiff,ydiff,zdiff
    real(chm_real)   major2,minor2
    !
    ncyz =ncly*nclz
    oncyz=oncly*onclz
    otranx=half*(onclx-1)*odcel
    otrany=half*(oncly-1)*odcel
    otranz=half*(onclz-1)*odcel
    il_loop: do il=1,ntprp
       IF(QA.AND.QB) THEN
          ! to consider the boundary conditions of basis set charge distributions
          chi=ochc(il)*(odcel/two/twopi)
          if(chi.eq.zero) cycle il_loop
          l=int((il-1)/oncyz)+1
          m=int(mod((il-1),oncyz)/onclz)+1
          n=mod(mod((il-1),oncyz),onclz)+1
          xi=(l-1)*odcel-otranx+oxbcen+tranx-xbcen
          yi=(m-1)*odcel-otrany+oybcen+trany-ybcen
          zi=(n-1)*odcel-otranz+ozbcen+tranz-zbcen
       ELSE
          i=lstprp(il)
          chi=CG(i)
          IF(CHI.EQ.ZERO) cycle il_loop
          IF(QRECTBOX)THEN
             IF(QA.AND.(X(I).LT.RBXMIN.OR.X(I).GT.RBXMAX.OR. &
                  Y(I).LT.RBYMIN.OR.Y(I).GT.RBYMAX.OR. &
                  Z(I).LT.RBZMIN.OR.Z(I).GT.RBZMAX))  &
                  cycle il_loop
             IF(QB.AND.(X(I).GE.RBXMIN.AND.X(I).LE.RBXMAX.AND. &
                  Y(I).GE.RBYMIN.AND.Y(I).LE.RBYMAX.AND. &
                  Z(I).GE.RBZMIN.AND.Z(I).LE.RBZMAX))  &
                  cycle il_loop
          ELSEIF(QSPHERE)THEN
             XDIFF=X(I)-RRXCEN
             YDIFF=Y(I)-RRYCEN
             ZDIFF=Z(I)-RRZCEN
             SR2=SRDIST*SRDIST
             R2=XDIFF*XDIFF+YDIFF*YDIFF+ZDIFF*ZDIFF
             IF(QA.AND.R2.GT.SR2) cycle il_loop
             IF(QB.AND.R2.LE.SR2) cycle il_loop
          ELSEIF(QPROLATE)THEN
             XDIFF=X(I)-RRXCEN
             YDIFF=Y(I)-RRYCEN
             ZDIFF=Z(I)-RRZCEN
             MAJOR2=MAJOR*MAJOR
             MINOR2=MINOR*MINOR
             R2=(XDIFF*XDIFF+YDIFF*YDIFF)/MINOR2+ZDIFF*ZDIFF/MAJOR2
             IF(QA.AND.R2.GT.ONE) cycle il_loop
             IF(QB.AND.R2.LE.ONE) cycle il_loop
          ENDIF
          xi=x(i)+tranx-xbcen
          yi=y(i)+trany-ybcen
          zi=z(i)+tranz-zbcen
          if(xi.lt.zero .or. xi.gt.2*tranx .or.         & ! for a charge
               yi.lt.zero .or. yi.gt.2*trany .or.         & ! outside a box
               zi.lt.zero .or. zi.gt.2*tranz) cycle il_loop   ! in focussing
       ENDIF

       ix=int(xi/dcel)+1
       iy=int(yi/dcel)+1
       iz=int(zi/dcel)+1

       ! Construct the charge distribution by 8 grid points adjacent to the atom
       do n1=1,2
          in1=ix+n1-1
          ai=abs(xi-(in1-1)*dcel)/dcel
          in1=(in1-1)*ncyz
          ai=1.0-ai
          !
          do n2=1,2
             in2=iy+n2-1
             bi=abs(yi-(in2-1)*dcel)/dcel
             in2=(in2-1)*nclz
             bi=ai*(1.0-bi)
             !
             do n3=1,2
                in3=iz+n3-1
                ci=abs(zi-(in3-1)*dcel)/dcel
                fi=bi*(1.0-ci)
                in3=in1+in2+in3

                chc(in3)=chc(in3)+(fi*chi)*TWO*TWOPI/DCEL

             enddo
          enddo
       enddo
    enddo il_loop
    !
    ! Construct homogeneous neturalizing background charge if kappa=0
    ! q(i,j,k) = 4 * pi * QTOT / (Lx*Ly*Lz) * dcel2 [e/angs] for finite-difference solver
    !
    IF(QPBC.AND.KAPPA2.EQ.0.0) THEN
       QTOT=ZERO
       IF(QA.AND.QB) THEN
          return
       ELSE
          do il=1,ntprp
             i=lstprp(il)
             QTOT=QTOT+CG(i)
          enddo
       ENDIF
       QTOT=QTOT/(NCLX*NCLY*NCLZ*DCEL*DCEL*DCEL)
       NC3=NCyz*NCLx
       DO il=1,nc3
          !wi            chc(il)=chc(il)-QTOT*TWO*TWOPI/DCEL
          chc(il)=chc(il)-QTOT*TWO*TWOPI*DCEL*DCEL
       ENDDO
    ENDIF
    !
    RETURN
  END SUBROUTINE MAYER_CD_TRIL

  SUBROUTINE MAYER_MSTEP(NTPRP,LSTPRP,X,Y,Z,PBRAD,RADW,RADI,EPSP, &
       NCLX,NCLY,NCLZ,DCEL, &
       TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
       RMAY,RMAYX,RMAYY,RMAYZ,QREEN,QFKAP)
    !----------------------------------------------------------------------
    !     when the dielectric boundary is defined by the van der Waals surface
    !     M = 0 : inside solute
    !         1 : outside solute
    !
    use chm_kinds
    use number
    use consta
    implicit none
    INTEGER  NTPRP,LSTPRP(*)
    INTEGER  NCLX,NCLY,NCLZ
    REAL(CHM_REAL4)   RMAY(*),RMAYX(*),RMAYY(*),RMAYZ(*)
    real(chm_real)   X(*),Y(*),Z(*),PBRAD(*)
    real(chm_real)   RADW,RADI,EPSP
    real(chm_real)   DCEL,TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN
    LOGICAL  QREEN,QFKAP
    ! local
    integer  i,il,k,l,m,ix,iy,iz,jx1,jx2,jy1,jy2,jz1,jz2
    integer  ipx,ipy,ipz
    integer  nfil,ncyz
    real(chm_real)   xi,yi,zi,xc,yc,zc,xc1,yc1,zc1,xsq,ysq,zsq, &
         dsq,dsq1
    real(chm_real)   sqr,sqrw,sqri
    !
    !wi      ncyz=nclx*ncly
    ncyz=ncly*nclz
    il_loop: do il=1,ntprp
       i=lstprp(il)
       sqr =pbrad(i)
       if(sqr.le.0.0) cycle il_loop
       xi=x(i)+tranx-xbcen
       yi=y(i)+trany-ybcen
       zi=z(i)+tranz-zbcen
       sqrw=sqr+radw
       sqri=sqr+radi
       nfil=int((sqrw+radi)/dcel)+2
       sqr =sqr *sqr
       sqrw=sqrw*sqrw
       sqri=sqri*sqri
       ix=int(xi/dcel)+1
       iy=int(yi/dcel)+1
       iz=int(zi/dcel)+1
       !
       JX1=IX-NFIL+1
       IF(JX1.LT.1)JX1=1
       JX2=IX+NFIL-1
       IF(JX2.GT.NCLX)JX2=NCLX
       JY1=IY-NFIL+1
       IF(JY1.LT.1)JY1=1
       JY2=IY+NFIL-1
       IF(JY2.GT.NCLY)JY2=NCLY
       JZ1=IZ-NFIL+1
       IF(JZ1.LT.1)JZ1=1
       JZ2=IZ+NFIL-1
       IF(JZ2.GT.NCLZ)JZ2=NCLZ
       !
       do k=jx1,jx2
          ipx=(k-1)*ncyz
          xc=(k-1)*dcel
          do l=jy1,jy2
             ipy=(l-1)*nclz
             yc=(l-1)*dcel
             do m=jz1,jz2
                ipz=m+ipy+ipx
                zc=(m-1)*dcel
                xsq=(xc-xi)*(xc-xi)
                ysq=(yc-yi)*(yc-yi)
                zsq=(zc-zi)*(zc-zi)

                IF(.not.QFKAP)THEN
                   IF(rmay(ipz).ne.0.)THEN
                      dsq=xsq+ysq+zsq
                      if(QREEN)then
                         if(RADI.gt.0.0) then
                            IF(dsq.le.sqri)THEN
                               ! Zero the Debye-Huckel factor inside the solute + Stern layer
                               rmay(ipz)=0.
                            ENDIF
                         else
                            IF(dsq.le.sqr)THEN
                               ! Zero the Debye-Huckel factor inside the solute
                               rmay(ipz)=0.
                            ELSEIF(dsq.gt.sqr.and.dsq.le.sqrw)THEN
                               IF(rmay(ipz).gt.0.)rmay(ipz)=-rmay(ipz)
                            ENDIF
                         endif
                      else
                         IF(dsq.le.sqri)THEN
                            ! Zero the Debye-Huckel factor inside the solute + Stern layer
                            rmay(ipz)=0.
                         ENDIF
                      endif
                   ENDIF
                ENDIF

                IF(rmayx(ipz).ne.EPSP)THEN
                   xc1=xc+0.5*dcel
                   dsq1=(xc1-xi)**2+ysq+zsq
                   if(QREEN)then
                      IF(dsq1.le.sqr)THEN
                         ! vaccum dielectric constant inside the solute
                         rmayx(ipz)=EPSP
                      ELSEIF(dsq1.gt.sqr.and.dsq1.le.sqrw)then
                         IF(rmayx(ipz).gt.0.)rmayx(ipz)=-rmayx(ipz)
                      ENDIF
                   else
                      IF(dsq1.le.sqrw)THEN
                         ! vaccum dielectric constant inside the solute
                         rmayx(ipz)=EPSP
                      ENDIF
                   endif
                ENDIF

                IF(rmayy(ipz).ne.EPSP)THEN
                   yc1=yc+0.5*dcel
                   dsq1=(yc1-yi)**2+xsq+zsq
                   if(QREEN)then
                      IF(dsq1.le.sqr)THEN
                         ! vaccum dielectric constant inside the solute
                         rmayy(ipz)=EPSP
                      ELSEIF(dsq1.gt.sqr.and.dsq1.le.sqrw)then
                         IF(rmayy(ipz).gt.0.)rmayy(ipz)=-rmayy(ipz)
                      ENDIF
                   else
                      IF(dsq1.le.sqrw)THEN
                         ! vaccum dielectric constant inside the solute
                         rmayy(ipz)=EPSP
                      ENDIF
                   endif
                ENDIF

                IF(rmayz(ipz).ne.EPSP)THEN
                   zc1=zc+0.5*dcel
                   dsq1=(zc1-zi)**2+ysq+xsq
                   if(QREEN)then
                      IF(dsq1.le.sqr)THEN
                         ! vaccum dielectric constant inside the solute
                         rmayz(ipz)=EPSP
                      ELSEIF(dsq1.gt.sqr.and.dsq1.le.sqrw)then
                         IF(rmayz(ipz).gt.0.)rmayz(ipz)=-rmayz(ipz)
                      ENDIF
                   else
                      IF(dsq1.le.sqrw)THEN
                         !  vaccum dielectric constant inside the solute
                         rmayz(ipz)=EPSP
                      ENDIF
                   endif
                ENDIF
                !
             enddo
          enddo
       enddo
    enddo il_loop
    !
    RETURN
  END SUBROUTINE MAYER_MSTEP

  SUBROUTINE MAYER_MSMOOTH(NTPRP,LSTPRP,X,Y,Z,PBRAD,SWIN, &
       EPSP,EPSW,EPSM,EPSH,KAPPA2,NCLX,NCLY,NCLZ,DCEL, &
       TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
       RMAY,RMAYX,RMAYY,RMAYZ,QFKAP)
    !----------------------------------------------------------------------
    !     when the extended dielectric boundary (smoothing) is used
    !     M = 0                       : inside solute
    !     M = smoothing function      : within the dielectric boundary
    !     M = 1                       : outside solute
    !
    use chm_kinds
    use number
    use consta
    implicit none
    INTEGER  NTPRP,LSTPRP(*)
    INTEGER  NCLX,NCLY,NCLZ
    REAL(CHM_REAL4)   RMAY(*),RMAYX(*),RMAYY(*),RMAYZ(*)
    real(chm_real)   X(*),Y(*),Z(*),PBRAD(*)
    real(chm_real)   SWIN,EPSP,EPSW,EPSM,EPSH,KAPPA2
    real(chm_real)   DCEL,TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN
    LOGICAL  QFKAP
    ! local
    integer  i,il,k,l,m,ix,iy,iz,jx1,jx2,jy1,jy2,jz1,jz2
    integer  ipx,ipy,ipz
    integer  nfil,ncyz
    real(chm_real)   xi,yi,zi,xc,yc,zc,xc1,yc1,zc1,xsq,ysq,zsq, &
         dsq,dcel2
    real(chm_real)   bisq,bps,bms,rij,dr
    !
    !wi      ncyz=nclx*ncly
    ncyz=ncly*nclz
    dcel2=dcel*dcel
    do il=1,ntprp
       i=lstprp(il)
       xi=x(i)+tranx-xbcen
       yi=y(i)+trany-ybcen
       zi=z(i)+tranz-zbcen
       bisq=pbrad(i)
       nfil=int((bisq+swin)/dcel)+2
       bps=bisq+swin
       bms=bisq-swin
       !wi         bisq=bisq*bisq      because epssf require not a square of
       !wi                             distance but a distance
       ix=int(xi/dcel)+1
       iy=int(yi/dcel)+1
       iz=int(zi/dcel)+1

       JX1=IX-NFIL+1
       IF(JX1.LT.1)JX1=1
       JX2=IX+NFIL
       IF(JX2.GT.NCLX)JX2=NCLX
       JY1=IY-NFIL+1
       IF(JY1.LT.1)JY1=1
       JY2=IY+NFIL
       IF(JY2.GT.NCLY)JY2=NCLY
       JZ1=IZ-NFIL+1
       IF(JZ1.LT.1)JZ1=1
       JZ2=IZ+NFIL
       IF(JZ2.GT.NCLZ)JZ2=NCLZ
       !
       DO K=jx1,jx2
          IPX=(K-1)*NCyz
          XC=(K-1)*DCEL
          !
          DO L=jy1,jy2
             IPY=(L-1)*NCLz
             YC=(L-1)*DCEL
             !
             DO M=jz1,jz2
                IPZ=M+IPY+IPX
                ZC=(M-1)*DCEL
                !
                xsq=(xc-xi)**2
                ysq=(yc-yi)**2
                zsq=(zc-zi)**2

                ! Debye-Huckel screening factor
                IF(.not.QFKAP)THEN
                   if(rmay(ipz).ne.zero) then
                      rij=sqrt(xsq+ysq+zsq)
                      if((rij.gt.bms).and.(rij.lt.bps)) then
                         dr=rij-bms
                         if(rmay(ipz).eq.kappa2*dcel2) then
                            rmay(ipz)=kappa2*dcel2*epssf(dr,swin)
                         else
                            rmay(ipz)=rmay(ipz)*epssf(dr,swin)
                         endif
                      elseif(rij.le.bms) then
                         rmay(ipz)=zero
                      endif
                   endif
                ENDIF

                ! X-COMPONENT
                if(rmayx(ipz).ne.epsp) then
                   rij=sqrt((xc+dcel*half-xi)**2+ysq+zsq)
                   !
                   if((rij.gt.bms).and.(rij.lt.bps)) then
                      dr=rij-bms
                      !
                      if(rmayx(ipz).eq.epsw) then
                         rmayx(ipz)=epsp+(epsw-epsp)*epssf(dr,swin)
                      elseif(rmayx(ipz).eq.epsm) then
                         rmayx(ipz)=epsp+(epsm-epsp)*epssf(dr,swin)
                      elseif(rmayx(ipz).eq.epsh) then
                         rmayx(ipz)=epsp+(epsh-epsp)*epssf(dr,swin)
                      else
                         rmayx(ipz)=epsp+(rmayx(ipz)-epsp)*epssf(dr,swin)
                      endif
                      !
                   elseif(rij.le.bms) then
                      rmayx(ipz)=epsp
                   endif
                   !
                endif

                ! Y-COMPONENT
                if(rmayy(ipz).ne.epsp)then
                   rij=sqrt(xsq+(yc+dcel*half-yi)**2+zsq)
                   !
                   if((rij.gt.bms).and.(rij.lt.bps)) then
                      dr=rij-bms
                      !
                      if(rmayy(ipz).eq.epsw) then
                         rmayy(ipz)=epsp+(epsw-epsp)*epssf(dr,swin)
                      elseif(rmayy(ipz).eq.epsm) then
                         rmayy(ipz)=epsp+(epsm-epsp)*epssf(dr,swin)
                      elseif(rmayy(ipz).eq.epsh) then
                         rmayy(ipz)=epsp+(epsh-epsp)*epssf(dr,swin)
                      else
                         rmayy(ipz)=epsp+(rmayy(ipz)-epsp)*epssf(dr,swin)
                      endif
                      !
                   elseif(rij.le.bms) then
                      rmayy(ipz)=epsp
                   endif
                   !
                endif

                ! Z-COMPONENT
                if(rmayz(ipz).ne.epsp)then
                   rij=sqrt(xsq+ysq+(zc+dcel*half-zi)**2)
                   !
                   if((rij.gt.bms).and.(rij.lt.bps)) then
                      dr=rij-bms
                      !
                      if(rmayz(ipz).eq.epsw) then
                         rmayz(ipz)=epsp+(epsw-epsp)*epssf(dr,swin)
                      elseif(rmayz(ipz).eq.epsm) then
                         rmayz(ipz)=epsp+(epsm-epsp)*epssf(dr,swin)
                      elseif(rmayz(ipz).eq.epsh) then
                         rmayz(ipz)=epsp+(epsh-epsp)*epssf(dr,swin)
                      else
                         rmayz(ipz)=epsp+(rmayz(ipz)-epsp)*epssf(dr,swin)
                      endif
                      !
                   elseif(rij.le.bms) then
                      rmayz(ipz)=epsp
                   endif
                   !
                endif

             enddo
          enddo
       enddo
    enddo
    !
    RETURN
  END SUBROUTINE MAYER_MSMOOTH

  SUBROUTINE MAYER_BP_ALL(NTPRP,LSTPRP,X,Y,Z,CG,EPSW,KAPPA2, &
       TMEMB,CHC,PHI,NCLX,NCLY,NCLZ,DCEL, &
       TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
       RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
       RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
       MAJOR,MINOR,QPROLATE, &
       ONCLX,ONCLY,ONCLZ,ODCEL,OXBCEN,OYBCEN,OZBCEN, &
       NIMGB,QNPBC,QA,QB)
    !----------------------------------------------------------------------
    !     The default is the use of potentials at boundary points due to
    !     the charges of the solute and the salt concentration.
    !
    use chm_kinds
    use number
    use consta
    implicit none
    INTEGER  NTPRP,LSTPRP(*)
    INTEGER  NCLX,NCLY,NCLZ,ONCLX,ONCLY,ONCLZ,NIMGB
    REAL(CHM_REAL4)   PHI(*),CHC(*)
    real(chm_real)   X(*),Y(*),Z(*),CG(*),DCEL,TMEMB
    real(chm_real)   ODCEL,OXBCEN,OYBCEN,OZBCEN
    real(chm_real)   TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN,EPSW,KAPPA2
    real(chm_real)   RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX
    real(chm_real)   SRDIST,RRXCEN,RRYCEN,RRZCEN
    real(chm_real)   MAJOR,MINOR
    LOGICAL  QNPBC,QA,QB,QRECTBOX,QSPHERE,QPROLATE
    ! local
    integer  i,il,ipo,ipox,ipoy,ig,jg,kg,iig,jjg,ncyz,oncyz
    integer  l,m,n
    real(chm_real)   otranx,otrany,otranz
    real(chm_real)   chi,xi,yi,zi,kappa,xii,yii,xj,yj,zj,dist1,dist2
    real(chm_real)   r2,sr2,xdiff,ydiff,zdiff
    real(chm_real)   major2,minor2
    real(chm_real)   xbox,ybox
    !
    kappa=sqrt(kappa2/EPSW)
    ncyz=ncly*nclz
    oncyz=oncly*onclz
    otranx=half*(onclx-1)*odcel
    otrany=half*(oncly-1)*odcel
    otranz=half*(onclz-1)*odcel
    il_loop: do il=1,ntprp
       IF(QA.AND.QB) THEN
          ! to consider the boundary conditions of basis set charge distributions
          chi=CHC(il)*(ODCEL/TWO/TWOPI)
          IF(CHI.EQ.ZERO) cycle il_loop
          l=int((il-1)/oncyz)+1
          m=int(mod((il-1),oncyz)/onclz)+1
          n=mod(mod((il-1),oncyz),onclz)+1
          xi=(l-1)*odcel-otranx+oxbcen+tranx-xbcen
          yi=(m-1)*odcel-otrany+oybcen+trany-ybcen
          zi=(n-1)*odcel-otranz+ozbcen+tranz-zbcen
       ELSE
          i=lstprp(il)
          chi=CG(i)
          IF(CHI.EQ.ZERO) cycle il_loop
          IF(QRECTBOX)THEN
             IF(QA.AND.(X(I).LT.RBXMIN.OR.X(I).GT.RBXMAX.OR. &
                  Y(I).LT.RBYMIN.OR.Y(I).GT.RBYMAX.OR. &
                  Z(I).LT.RBZMIN.OR.Z(I).GT.RBZMAX))  &
                  cycle il_loop
             IF(QB.AND.(X(I).GE.RBXMIN.AND.X(I).LE.RBXMAX.AND. &
                  Y(I).GE.RBYMIN.AND.Y(I).LE.RBYMAX.AND. &
                  Z(I).GE.RBZMIN.AND.Z(I).LE.RBZMAX))  &
                  cycle il_loop
          ELSEIF(QSPHERE)THEN
             XDIFF=X(I)-RRXCEN
             YDIFF=Y(I)-RRYCEN
             ZDIFF=Z(I)-RRZCEN
             SR2=SRDIST*SRDIST
             R2=XDIFF*XDIFF+YDIFF*YDIFF+ZDIFF*ZDIFF
             IF(QA.AND.R2.GT.SR2) cycle il_loop
             IF(QB.AND.R2.LE.SR2) cycle il_loop
          ELSEIF(QPROLATE)THEN
             XDIFF=X(I)-RRXCEN
             YDIFF=Y(I)-RRYCEN
             ZDIFF=Z(I)-RRZCEN
             MAJOR2=MAJOR*MAJOR
             MINOR2=MINOR*MINOR
             R2=(XDIFF*XDIFF+YDIFF*YDIFF)/MINOR2+ZDIFF*ZDIFF/MAJOR2
             IF(QA.AND.R2.GT.ONE) cycle il_loop
             IF(QB.AND.R2.LE.ONE) cycle il_loop
          ENDIF
          xi=x(i)+tranx-xbcen
          yi=y(i)+trany-ybcen
          zi=z(i)+tranz-zbcen
       ENDIF

       IF(TMEMB.EQ.ZERO.or.QNPBC)THEN

          ! yz plane
          ipox=(nclx-1)*ncyz
          xj=(nclx-1)*dcel-xi
          do jg=1,ncly
             ipoy=(jg-1)*nclz
             yj=(jg-1)*dcel-yi
             do kg=2,nclz-1
                zj=(kg-1)*dcel-zi

                ipo=ipoy+kg
                dist1=sqrt(xi*xi+yj*yj+zj*zj)
                if(kappa.eq.zero) then
                   phi(ipo)=phi(ipo)+CHI/EPSW/DIST1
                else
                   phi(ipo)=phi(ipo)+CHI*EXP(-KAPPA*DIST1)/EPSW/DIST1
                endif

                ipo=ipox+ipo
                dist2=sqrt(xj*xj+yj*yj+zj*zj)
                if(kappa.eq.zero) then
                   phi(ipo)=phi(ipo)+CHI/EPSW/DIST2
                else
                   phi(ipo)=phi(ipo)+CHI*EXP(-KAPPA*DIST2)/EPSW/DIST2
                endif

             enddo
          enddo

          ! xz plane
          yj=(ncly-1)*dcel-yi
          do ig=2,nclx-1
             ipox=(ig-1)*ncyz
             xj=(ig-1)*dcel-xi
             do kg=2,nclz-1
                zj=(kg-1)*dcel-zi

                ipo=ipox+kg
                dist1=sqrt(xj*xj+yi*yi+zj*zj)
                if(kappa.eq.zero) then
                   phi(ipo)=phi(ipo)+CHI/EPSW/DIST1
                else
                   phi(ipo)=phi(ipo)+CHI*EXP(-KAPPA*DIST1)/EPSW/DIST1
                endif

                ipo=ipo+ncyz-nclz
                dist2=sqrt(xj*xj+yj*yj+zj*zj)
                if(kappa.eq.zero) then
                   phi(ipo)=phi(ipo)+CHI/EPSW/DIST2
                else
                   phi(ipo)=phi(ipo)+CHI*EXP(-KAPPA*DIST2)/EPSW/DIST2
                endif

             enddo
          enddo
       ENDIF

       ! xy plane
       IF(NIMGB.gt.0.and..not.QNPBC) THEN

          ! with image atoms
          zj=(nclz-1)*dcel-zi
          Xbox=TWO*tranx+dcel   ! box size in X
          Ybox=TWO*trany+dcel   ! box size in Y
          do ig=1,nclx
             ipox=(ig-1)*ncyz
             do jg=1,ncly
                do iig=-NIMGB,NIMGB
                   do jjg=-NIMGB,NIMGB
                      xii=iig*Xbox+xi
                      yii=jjg*Ybox+yi
                      xj=(ig-1)*dcel-xii
                      yj=(jg-1)*dcel-yii

                      ipo=ipox+(jg-1)*nclz+1
                      dist1=sqrt(xj*xj+yj*yj+zi*zi)
                      if(kappa.eq.zero) then
                         phi(ipo)=phi(ipo)+CHI/EPSW/DIST1
                      else
                         phi(ipo)=phi(ipo)+ &
                              CHI*EXP(-KAPPA*DIST1)/EPSW/DIST1
                      endif

                      ipo=ipo+nclz-1
                      dist2=sqrt(xj*xj+yj*yj+zj*zj)
                      if(kappa.eq.zero) then
                         phi(ipo)=phi(ipo)+CHI/EPSW/DIST2
                      else
                         phi(ipo)=phi(ipo)+ &
                              CHI*EXP(-KAPPA*DIST2)/EPSW/DIST2
                      endif

                   enddo
                enddo
             enddo
          enddo
       ELSE

          ! without image atoms
          zj=(nclz-1)*dcel-zi
          do ig=1,nclx
             ipox=(ig-1)*ncyz
             xj=(ig-1)*dcel-xi
             do jg=1,ncly
                yj=(jg-1)*dcel-yi

                ipo=ipox+(jg-1)*nclz+1
                dist1=sqrt(xj*xj+yj*yj+zi*zi)
                if(kappa.eq.zero) then
                   phi(ipo)=phi(ipo)+CHI/EPSW/DIST1
                else
                   phi(ipo)=phi(ipo)+CHI*EXP(-KAPPA*DIST1)/EPSW/DIST1
                endif

                ipo=ipo+nclz-1
                dist2=sqrt(xj*xj+yj*yj+zj*zj)
                if(kappa.eq.zero) then
                   phi(ipo)=phi(ipo)+CHI/EPSW/DIST2
                else
                   phi(ipo)=phi(ipo)+CHI*EXP(-KAPPA*DIST2)/EPSW/DIST2
                endif

             enddo
          enddo
       ENDIF
    enddo il_loop
    !
    !     write(10,*) int((ipo-1)/ncyz)+1,
    !    $             int(mod((ipo-1),ncyz)/nclz)+1,
    !    $             mod(mod((ipo-1),ncyz),nclz)+1
    !
    RETURN
  END SUBROUTINE MAYER_BP_ALL

  SUBROUTINE MAYER_BP_INT(NTPRP,LSTPRP,X,Y,Z,CG,EPSW,KAPPA2, &
       TMEMB,CHC,PHI,NCLX,NCLY,NCLZ,DCEL, &
       TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
       RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
       RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
       MAJOR,MINOR,QPROLATE, &
       ONCLX,ONCLY,ONCLZ,ODCEL,OXBCEN,OYBCEN,OZBCEN, &
       NIMGB,QNPBC,QA,QB)
    !----------------------------------------------------------------------
    !     Half number of grid points is used for the Debye-Huckel approximation
    !
    use chm_kinds
    use number
    use consta
    implicit none
    INTEGER  NTPRP,LSTPRP(*)
    INTEGER  NCLX,NCLY,NCLZ,ONCLX,ONCLY,ONCLZ,NIMGB
    REAL(CHM_REAL4)   PHI(*),CHC(*)
    real(chm_real)   X(*),Y(*),Z(*),CG(*),DCEL,TMEMB
    real(chm_real)   ODCEL,OXBCEN,OYBCEN,OZBCEN
    real(chm_real)   TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN,EPSW,KAPPA2
    real(chm_real)   RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX
    real(chm_real)   SRDIST,RRXCEN,RRYCEN,RRZCEN
    real(chm_real)   MAJOR,MINOR
    LOGICAL  QNPBC,QA,QB,QRECTBOX,QSPHERE,QPROLATE
    ! local
    integer  i,il,ipo,ipox,ipoy,ig,jg,kg,iig,jjg,ncyz,oncyz
    integer  l,m,n
    real(chm_real)   otranx,otrany,otranz
    real(chm_real)   chi,xi,yi,zi,kappa,xii,yii,xj,yj,zj,dist1,dist2
    real(chm_real)   r2,sr2,xdiff,ydiff,zdiff
    real(chm_real)   major2,minor2
    real(chm_real)   xbox,ybox
    !
    kappa=sqrt(kappa2/EPSW)
    ncyz=ncly*nclz
    oncyz=oncly*onclz
    otranx=half*(onclx-1)*odcel
    otrany=half*(oncly-1)*odcel
    otranz=half*(onclz-1)*odcel
    il_loop: do il=1,ntprp
       IF(QA.AND.QB) THEN
          ! to consider the boundary conditions of basis set charge distributions
          chi=CHC(il)*(ODCEL/TWO/TWOPI)
          IF(CHI.EQ.ZERO) CYCLE IL_LOOP
          l=int((il-1)/oncyz)+1
          m=int(mod((il-1),oncyz)/onclz)+1
          n=mod(mod((il-1),oncyz),onclz)+1
          xi=(l-1)*odcel-otranx+oxbcen+tranx-xbcen
          yi=(m-1)*odcel-otrany+oybcen+trany-ybcen
          zi=(n-1)*odcel-otranz+ozbcen+tranz-zbcen
       ELSE
          i=lstprp(il)
          chi=CG(i)
          IF(CHI.EQ.ZERO) CYCLE IL_LOOP
          IF(QRECTBOX)THEN
             IF(QA.AND.(X(I).LT.RBXMIN.OR.X(I).GT.RBXMAX.OR. &
                  Y(I).LT.RBYMIN.OR.Y(I).GT.RBYMAX.OR. &
                  Z(I).LT.RBZMIN.OR.Z(I).GT.RBZMAX))  &
                  CYCLE IL_LOOP
             IF(QB.AND.(X(I).GE.RBXMIN.AND.X(I).LE.RBXMAX.AND. &
                  Y(I).GE.RBYMIN.AND.Y(I).LE.RBYMAX.AND. &
                  Z(I).GE.RBZMIN.AND.Z(I).LE.RBZMAX))  &
                  CYCLE IL_LOOP
          ELSEIF(QSPHERE)THEN
             XDIFF=X(I)-RRXCEN
             YDIFF=Y(I)-RRYCEN
             ZDIFF=Z(I)-RRZCEN
             SR2=SRDIST*SRDIST
             R2=XDIFF*XDIFF+YDIFF*YDIFF+ZDIFF*ZDIFF
             IF(QA.AND.R2.GT.SR2) CYCLE IL_LOOP
             IF(QB.AND.R2.LE.SR2) CYCLE IL_LOOP
          ELSEIF(QPROLATE)THEN
             XDIFF=X(I)-RRXCEN
             YDIFF=Y(I)-RRYCEN
             ZDIFF=Z(I)-RRZCEN
             MAJOR2=MAJOR*MAJOR
             MINOR2=MINOR*MINOR
             R2=(XDIFF*XDIFF+YDIFF*YDIFF)/MINOR2+ZDIFF*ZDIFF/MAJOR2
             IF(QA.AND.R2.GT.ONE) CYCLE IL_LOOP
             IF(QB.AND.R2.LE.ONE) CYCLE IL_LOOP
          ENDIF
          xi=x(i)+tranx-xbcen
          yi=y(i)+trany-ybcen
          zi=z(i)+tranz-zbcen
       ENDIF

       IF(TMEMB.EQ.ZERO.or.QNPBC)THEN

          ! yz plane
          ipox=(nclx-1)*ncyz
          xj=(nclx-1)*dcel-xi
          do jg=1,ncly,2
             ipoy=(jg-1)*nclz
             yj=(jg-1)*dcel-yi
             do kg=3,nclz-2,2
                zj=(kg-1)*dcel-zi

                ipo=ipoy+kg
                dist1=sqrt(xi*xi+yj*yj+zj*zj)
                if(kappa.eq.zero) then
                   phi(ipo)=phi(ipo)+CHI/EPSW/DIST1
                else
                   phi(ipo)=phi(ipo)+CHI*EXP(-KAPPA*DIST1)/EPSW/DIST1
                endif

                ipo=ipox+ipo
                dist2=sqrt(xj*xj+yj*yj+zj*zj)
                if(kappa.eq.zero) then
                   phi(ipo)=phi(ipo)+CHI/EPSW/DIST2
                else
                   phi(ipo)=phi(ipo)+CHI*EXP(-KAPPA*DIST2)/EPSW/DIST2
                endif

             enddo
          enddo

          ! xz plane
          yj=(ncly-1)*dcel-yi
          do ig=3,nclx-2,2
             ipox=(ig-1)*ncyz
             xj=(ig-1)*dcel-xi
             do kg=3,nclz-2,2
                zj=(kg-1)*dcel-zi

                ipo=ipox+kg
                dist1=sqrt(xj*xj+yi*yi+zj*zj)
                if(kappa.eq.zero) then
                   phi(ipo)=phi(ipo)+CHI/EPSW/DIST1
                else
                   phi(ipo)=phi(ipo)+CHI*EXP(-KAPPA*DIST1)/EPSW/DIST1
                endif

                ipo=ipo+ncyz-nclz
                dist2=sqrt(xj*xj+yj*yj+zj*zj)
                if(kappa.eq.zero) then
                   phi(ipo)=phi(ipo)+CHI/EPSW/DIST2
                else
                   phi(ipo)=phi(ipo)+CHI*EXP(-KAPPA*DIST2)/EPSW/DIST2
                endif

             enddo
          enddo
       ENDIF

       ! xy plane
       IF(NIMGB.gt.0.and..NOT.QNPBC) THEN

          ! with image atoms
          zj=(nclz-1)*dcel-zi
          Xbox=TWO*tranx+dcel   ! box size in X
          Ybox=TWO*trany+dcel   ! box size in Y
          do ig=1,nclx,2
             ipox=(ig-1)*ncyz
             do jg=1,ncly,2
                do iig=-NIMGB,NIMGB
                   do jjg=-NIMGB,NIMGB
                      xii=iig*Xbox+xi
                      yii=jjg*Ybox+yi
                      xj=(ig-1)*dcel-xii
                      yj=(jg-1)*dcel-yii

                      ipo=ipox+(jg-1)*nclz+1
                      dist1=sqrt(xj*xj+yj*yj+zi*zi)
                      if(kappa.eq.zero) then
                         phi(ipo)=phi(ipo)+CHI/EPSW/DIST1
                      else
                         phi(ipo)=phi(ipo)+ &
                              CHI*EXP(-KAPPA*DIST1)/EPSW/DIST1
                      endif

                      ipo=ipo+nclz-1
                      dist2=sqrt(xj*xj+yj*yj+zj*zj)
                      if(kappa.eq.zero) then
                         phi(ipo)=phi(ipo)+CHI/EPSW/DIST2
                      else
                         phi(ipo)=phi(ipo)+ &
                              CHI*EXP(-KAPPA*DIST2)/EPSW/DIST2
                      endif

                   enddo
                enddo
             enddo
          enddo
       ELSE

          ! without image atoms
          zj=(nclz-1)*dcel-zi
          do ig=1,nclx,2
             ipox=(ig-1)*ncyz
             xj=(ig-1)*dcel-xi
             do jg=1,ncly,2
                yj=(jg-1)*dcel-yi

                ipo=ipox+(jg-1)*nclz+1
                dist1=sqrt(xj*xj+yj*yj+zi*zi)
                if(kappa.eq.zero) then
                   phi(ipo)=phi(ipo)+CHI/EPSW/DIST1
                else
                   phi(ipo)=phi(ipo)+CHI*EXP(-KAPPA*DIST1)/EPSW/DIST1
                endif

                ipo=ipo+nclz-1
                dist2=sqrt(xj*xj+yj*yj+zj*zj)
                if(kappa.eq.zero) then
                   phi(ipo)=phi(ipo)+CHI/EPSW/DIST2
                else
                   phi(ipo)=phi(ipo)+CHI*EXP(-KAPPA*DIST2)/EPSW/DIST2
                endif

             enddo
          enddo
       ENDIF
    enddo il_loop

    ! bilinear interpolation

    ! xy plane
    ! elements in odd-numbered rows, interplating horizontally
    do ig=2,nclx-1,2
       ipox=(ig-1)*ncyz
       do jg=1,ncly,2
          ipo=ipox+(jg-1)*nclz+1
          phi(ipo)=(phi(ipo-ncyz)+phi(ipo+ncyz))/two
          ipo=ipo+nclz-1
          phi(ipo)=(phi(ipo-ncyz)+phi(ipo+ncyz))/two
       enddo
    enddo
    ! elements in even-numbered rows, interplating vertically
    do ig=1,nclx
       ipox=(ig-1)*ncyz
       do jg=2,ncly-1,2
          ipo=ipox+(jg-1)*nclz+1
          phi(ipo)=(phi(ipo-nclz)+phi(ipo+nclz))/two
          ipo=ipo+nclz-1
          phi(ipo)=(phi(ipo-nclz)+phi(ipo+nclz))/two
       enddo
    enddo

    IF(TMEMB.EQ.ZERO.or.QNPBC)THEN
       ! yz plane
       ! elements in odd-numbered rows, interplating horizontally
       ipox=(nclx-1)*ncyz
       do jg=2,ncly-1,2
          ipoy=(jg-1)*nclz
          do kg=3,nclz-2,2
             ipo=ipoy+kg
             phi(ipo)=(phi(ipo-nclz)+phi(ipo+nclz))/two
             ipo=ipox+ipo
             phi(ipo)=(phi(ipo-nclz)+phi(ipo+nclz))/two
          enddo
       enddo
       ! elements in even-numbered rows, interplating vertically
       do jg=1,ncly
          ipoy=(jg-1)*nclz
          do kg=2,nclz,2
             ipo=ipoy+kg
             phi(ipo)=(phi(ipo-1)+phi(ipo+1))/two
             ipo=ipox+ipo
             phi(ipo)=(phi(ipo-1)+phi(ipo+1))/two
          enddo
       enddo
       ! xz plane
       ! elements in odd-numbered rows, interplating horizontally
       do ig=2,nclx-1,2
          ipox=(ig-1)*ncyz
          do kg=3,nclz-2,2
             ipo=ipox+kg
             phi(ipo)=(phi(ipo-ncyz)+phi(ipo+ncyz))/two
             ipo=ipo+ncyz-nclz
             phi(ipo)=(phi(ipo-ncyz)+phi(ipo+ncyz))/two
          enddo
       enddo
       ! elements in even-numbered rows, interplating vertically
       do ig=2,nclx-1
          ipox=(ig-1)*ncyz
          do kg=2,nclz-1,2
             ipo=ipox+kg
             phi(ipo)=(phi(ipo-1)+phi(ipo+1))/two
             ipo=ipo+ncyz-nclz
             phi(ipo)=(phi(ipo-1)+phi(ipo+1))/two
          enddo
       enddo
    ENDIF
    !
    !     write(10,*) int((ipo-1)/ncyz)+1,
    !    $             int(mod((ipo-1),ncyz)/nclz)+1,
    !    $             mod(mod((ipo-1),ncyz),nclz)+1
    !
    RETURN
  END SUBROUTINE MAYER_BP_INT

  SUBROUTINE MAYER_BPFOCUS(PHI,PHIB,NCLX,NCLY,NCLZ)
    !----------------------------------------------------------------------
    !     Half number of grid points is used for the Debye-Huckel approximation
    !
    use chm_kinds
    use number
    implicit none
    INTEGER  NCLX,NCLY,NCLZ
    REAL(CHM_REAL4)   PHI(*),PHIB(*)
    ! local
    integer  ipo,ipox,ipoy,ig,jg,kg,cnt,ncyz
    !
    ncyz=ncly*nclz
    cnt=0

    ! yz plane
    do jg=1,ncly
       do kg=1,nclz
          ipo=(jg-1)*nclz+kg                 !yz plane at x=xmin
          cnt=cnt+1
          phi(ipo)=phib(cnt)
          ipo=(nclx-1)*ncyz+(jg-1)*nclz+kg   !yz plane at x=xmax
          cnt=cnt+1
          phi(ipo)=phib(cnt)
       enddo
    enddo

    ! xz plane
    do ig=2,nclx-1
       do kg=1,nclz
          ipo=(ig-1)*ncyz+kg                 !xz plane at y=ymin
          cnt=cnt+1
          phi(ipo)=phib(cnt)
          ipo=(ig-1)*ncyz+ncyz-nclz+kg       !xz plane at y=ymax
          cnt=cnt+1
          phi(ipo)=phib(cnt)
       enddo
    enddo

    ! xy plane
    do ig=2,nclx-1
       do jg=2,ncly-1
          ipo=(ig-1)*ncyz+(jg-1)*nclz+1      !xy plane at z=zmin
          cnt=cnt+1
          phi(ipo)=phib(cnt)
          ipo=(ig-1)*ncyz+(jg-1)*nclz+nclz   !xy plane at z=zmax
          cnt=cnt+1
          phi(ipo)=phib(cnt)
       enddo
    enddo
    !
    RETURN
  END SUBROUTINE MAYER_BPFOCUS

  SUBROUTINE BOUNDPOT(NCLX,NCLY,NCLZ,DCEL, &
       TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
       FNCEX,FNCEY,FNCEZ,FDCEL,FXBCEN,FYBCEN,FZBCEN, &
       PHI,PHIB)
    !----------------------------------------------------------------------
    !     Boundary potentials of the focussed system are obtained from
    !     the previous potentials using the trilinear interpolation method.
    !
    use chm_kinds
    use number
    implicit none
    !
    integer   nclx,ncly,nclz,fncex,fncey,fncez
    real(chm_real4)    phi(*),phib(*)
    real(chm_real)    dcel,tranx,trany,tranz,fdcel
    real(chm_real)    xbcen,ybcen,zbcen,fxbcen,fybcen,fzbcen
    ! local
    integer   ncyz,cnt,n1,in1,n2,in2,n3,in3
    Integer   ix,iy,iz,ig,jg,kg
    integer   jx,jy,jz,jn1,jn2,jn3
    real(chm_real)    xi,xj,yi,yj,zi,zj
    real(chm_real)    ftranx,ftrany,ftranz
    real(chm_real)    ai,bi,ci,aj,bj,cj,fi,fj
    !
    cnt=0
    ncyz=ncly*nclz
    ftranx=half*(fncex-1)*fdcel
    ftrany=half*(fncey-1)*fdcel
    ftranz=half*(fncez-1)*fdcel

    ! yz plane
    do jg=1,fncey
       do kg=1,fncez

          cnt=cnt+1           ! yz plane at x=xmin
          phib(cnt)=zero      ! ipo=(jg-1)*fncez+kg
          cnt=cnt+1           ! yz plane at x=xmax
          phib(cnt)=zero      ! ipx=(nclx-1)*ncyz+(jg-1)*nclz+kg

          ! find out the previous mapping points
          xi=zero         - ftranx + fxbcen + tranx - xbcen
          xj=2*ftranx     - ftranx + fxbcen + tranx - xbcen
          yi=(jg-1)*fdcel - ftrany + fybcen + trany - ybcen
          zi=(kg-1)*fdcel - ftranz + fzbcen + tranz - zbcen
          ix=int(xi/dcel)+1
          jx=int(xj/dcel)+1
          iy=int(yi/dcel)+1
          iz=int(zi/dcel)+1

          ! Previous potentials of 8 grid points around current boundary points
          ! are interpolated to boundary points.

          do n1=1,2
             in1=ix+n1-1
             jn1=jx+n1-1
             ai=one-abs(xi-(in1-1)*dcel)/dcel
             aj=one-abs(xj-(jn1-1)*dcel)/dcel
             in1=(in1-1)*ncyz
             jn1=(jn1-1)*ncyz

             do n2=1,2
                in2=iy+n2-1
                bi=one-abs(yi-(in2-1)*dcel)/dcel
                in2=(in2-1)*nclz

                do n3=1,2
                   in3=iz+n3-1
                   ci=one-abs(zi-(in3-1)*dcel)/dcel
                   fi=ai*bi*ci
                   fj=aj*bi*ci
                   jn3=jn1+in2+in3
                   in3=in1+in2+in3
                   phib(cnt-1)=phib(cnt-1)+fi*phi(in3)
                   phib(cnt)  =phib(cnt)  +fj*phi(jn3)

                enddo
             enddo
          enddo

       enddo
    enddo
    ! xz plane
    do ig=2,fncex-1
       do kg=1,fncez

          cnt=cnt+1           ! xz plane at y=ymin
          phib(cnt)=zero      ! ipo=(ig-1)*ncyz+kg
          cnt=cnt+1           ! xz plane at y=ymax
          phib(cnt)=zero      ! ipy=(ig-1)*ncyz+ncyz-nclz+kg

          ! find out the previous mapping points
          xi=(ig-1)*fdcel - ftranx + fxbcen + tranx - xbcen
          yi=zero         - ftrany + fybcen + trany - ybcen
          yj=2*ftrany     - ftrany + fybcen + trany - ybcen
          zi=(kg-1)*fdcel - ftranz + fzbcen + tranz - zbcen
          ix=int(xi/dcel)+1
          iy=int(yi/dcel)+1
          jy=int(yj/dcel)+1
          iz=int(zi/dcel)+1

          ! Previous potentials of 8 grid points around current boundary points
          ! are interpolated to boundary points.

          do n1=1,2
             in1=ix+n1-1
             ai=one-abs(xi-(in1-1)*dcel)/dcel
             in1=(in1-1)*ncyz

             do n2=1,2
                in2=iy+n2-1
                jn2=jy+n2-1
                bi=one-abs(yi-(in2-1)*dcel)/dcel
                bj=one-abs(yj-(jn2-1)*dcel)/dcel
                in2=(in2-1)*nclz
                jn2=(jn2-1)*nclz

                do n3=1,2
                   in3=iz+n3-1
                   ci=one-abs(zi-(in3-1)*dcel)/dcel
                   fi=ai*bi*ci
                   fj=ai*bj*ci
                   jn3=in1+jn2+in3
                   in3=in1+in2+in3

                   phib(cnt-1)=phib(cnt-1)+fi*phi(in3)
                   phib(cnt)  =phib(cnt)  +fj*phi(jn3)

                enddo
             enddo
          enddo

       enddo
    enddo

    ! xy plane  NOTE: currently, in focussing we don't consider image atoms.
    do ig=2,fncex-1
       do jg=2,fncey-1

          cnt=cnt+1                ! xy plane at z=zmin
          phib(cnt)=zero           ! ipo=(ig-1)*ncyz+(jg-1)*nclz+1
          cnt=cnt+1                ! xy plane at z=zmax
          phib(cnt)=zero           ! ipz=(ig-1)*ncyz+(jg-1)*nclz+nclz

          ! find out the previous mapping points
          xi=(ig-1)*fdcel - ftranx + fxbcen + tranx - xbcen
          yi=(jg-1)*fdcel - ftrany + fybcen + trany - ybcen
          zi=zero         - ftranz + fzbcen + tranz - zbcen
          zj=2*ftranz     - ftranz + fzbcen + tranz - zbcen
          ix=int(xi/dcel)+1
          iy=int(yi/dcel)+1
          iz=int(zi/dcel)+1
          jz=int(zj/dcel)+1

          ! Previous potentials of 8 grid points around current boundary points
          ! are interpolated to boundary points.

          do n1=1,2
             in1=ix+n1-1
             ai=one-abs(xi-(in1-1)*dcel)/dcel
             in1=(in1-1)*ncyz

             do n2=1,2
                in2=iy+n2-1
                bi=one-abs(yi-(in2-1)*dcel)/dcel
                in2=(in2-1)*nclz

                do n3=1,2
                   in3=iz+n3-1
                   jn3=jz+n3-1
                   ci=one-abs(zi-(in3-1)*dcel)/dcel
                   cj=one-abs(zj-(jn3-1)*dcel)/dcel
                   fi=ai*bi*ci
                   fj=ai*bi*cj
                   jn3=in1+in2+jn3
                   in3=in1+in2+in3

                   phib(cnt-1)=phib(cnt-1)+fi*phi(in3)
                   phib(cnt)  =phib(cnt)  +fj*phi(jn3)

                enddo
             enddo
          enddo

       enddo
    enddo

    !
    RETURN
  END SUBROUTINE BOUNDPOT

  SUBROUTINE CAPACI(NCLX,NCLY,NCLZ,DCEL, &
       TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
       PHI,FKAPPA2,KAPPA2,EPSW,EPSM,VMEMB,TMEMB,ZMEMB)
    !-----------------------------------------------------------------------
    ! This subroutine computes the net charge on the side I of membranes to obtain
    ! the capacitive contribution to the electrostatic free energy of solvation.
    !
    use chm_kinds
    use consta
    use number
    use stream
    !
    implicit none
    integer NCLX,NCLY,NCLZ
    real(chm_real)  dcel,tranx,trany,tranz,xbcen,ybcen,zbcen
    real(chm_real4)  phi(*), FKAPPA2(*)
    real(chm_real)  KAPPA2, EPSW, EPSM, VMEMB, TMEMB, ZMEMB
    !
    !   local
    integer ncyz,i,j,k,in3
    real(chm_real) dc2,xi,yi,zi,length,kappa
    real(chm_real) zmemb1, zmemb2, afact
    real(chm_real) qnet, qnet1, qnet2, enet, enelp
    real(chm_real) zfirst, zlast
    !
    if(prnlev.ge.2) write(outu,'(3x,a)')  &
         'Capacitance of the membrane system '
    !
    kappa=sqrt(kappa2/EPSW)
    ncyz = ncly*nclz
    dc2 = dcel*dcel
    length = (nclz-1)*dcel    ! along Z-axis
    enelp=zero
    qnet1 = zero
    qnet2 = zero
    zmemb1=-0.5*TMEMB+ZMEMB-zbcen
    zmemb2=zmemb1+TMEMB
    afact = vmemb/(2.0+(epsw/epsm)*kappa*Tmemb)

    qnet = - kappa2/(4*pi)*(Afact/kappa)
    zfirst = -tranz-zmemb1
    zlast = tranz-zmemb2

    if(zfirst.gt.0.0) zfirst = 0.0
    if(zlast.lt.0.0)  zlast = 0.0

    qnet1=-kappa2/(4*pi)*(Afact*exp(kappa*zfirst)/kappa)
    qnet2= kappa2/(4*pi)*(Afact*exp(-kappa*zlast)/kappa)
    qnet1=qnet1*length**2
    qnet2=qnet2*length**2
    !
    if(prnlev.ge.2) write(outu,101) &
         'Sigma of a slab            = ', &
         qnet,' [UNIT CHARGE]'
    if(prnlev.ge.2) write(outu,101) &
         'Total charge               = ', &
         qnet*length**2,' [e]'
    if(prnlev.ge.2) write(outu,'(3x,a,2f12.5,a)') &
         'initial charges            = ', &
         qnet1,qnet2,' [e]'
101 FORMAT(3X,A,F12.5,A,2X)
    !
    ! Calculate the capacitive energy contribution for membranes
    do k=1,nclz
       zi=(k-1)*dcel-tranz
       do i=1,nclx
          !wi            xi=(i-1)*dcel-tranx
          do j=1,ncly
             !wi               yi=(j-1)*dcel-trany
             in3=(i-1)*ncyz+(j-1)*nclz+k
             if(zi.le.zmemb1)then
                qnet1=qnet1-fkappa2(in3)*phi(in3)*dcel/(TWO*TWOPI)
             elseif(zi.ge.zmemb2)then
                qnet2=qnet2-fkappa2(in3)*(phi(in3)-vmemb) &
                     *dcel/(TWO*TWOPI)
             endif
          enddo
       enddo
       if(prnlev.gt.8)then
          write(OUTU,20) zi,fkappa2(in3)/dc2,qnet1,qnet2
20        format(10f10.5,2x,3e16.7)
       endif
    enddo
    ! total average charge
    qnet = half*(qnet2-qnet1)
    if(prnlev.ge.2) write(outu,'(3x,a,3f12.5,a)') &
         'Net external charge        = ', &
         qnet,qnet1,qnet2,' [e] '

    ! E_cap = - 0.5 * QNET * V
    if(prnlev.ge.2) write(outu,101) &
         'Capacitive energy          = ', &
         -HALF*QNET*VMEMB*CCELEC,' [KCAL/MOL] '

    ! CAPACITY = QNET/V
    if(prnlev.ge.2) write(outu,101) &
         'Capacitance                = ', &
         QNET/(VMEMB),' [Angstroms]'

    !     if(prnlev.ge.2) write(outu,'(a,e16.7)')
    !    $  'Capacity [Farad] ',1.602E-19*QNET/(VMEMB*14.40145905)

    RETURN
  END SUBROUTINE CAPACI

  SUBROUTINE genpoint(xp,yp,zp,iseed)
    !------------------------------------------------------------------------
    ! Generate random point on the sphere surface (used in MAYER)
    !
    use clcg_mod,only:random
    use chm_kinds
    implicit none
    real(chm_real) xp,yp,zp,rp
    integer iseed
    !
10  continue
    xp=2*random(iseed)-1.
    yp=2*random(iseed)-1.
    zp=2*random(iseed)-1.
    rp=xp**2+yp**2+zp**2
    if(rp.gt.1.)goto 10
    if(rp.eq.0.)goto 10
    rp=sqrt(rp)
    xp=xp/rp
    yp=yp/rp
    zp=zp/rp
    !
    return
  end SUBROUTINE genpoint

!!!LNI REMOVED AND REPLACED BY INLINE COPY. L. Nilsson, October 2009
  !SUBROUTINE SVPBRAD(NATOM,WMAIN,PBRAD)
  !  !-----------------------------------------------------------------------
  !  !     vdW Radii are stored.
  !  !
  !  use chm_kinds
  !  implicit none
  !  !
  !  real(chm_real)   WMAIN(*),PBRAD(*)
  !  INTEGER  NATOM,I
  !  !
  !  do i=1,natom
  !     pbrad(i)=wmain(i)
  !  enddo
  !  !
  !  RETURN
  !END SUBROUTINE SVPBRAD


  !--------------------------------------------------------
  ! PBEQ2.SRC insterted here November 2009, LNI
  !
  SUBROUTINE OLDPBEQ1(MAXITS,OMEGA,TOL,NCLX,NCLY,NCLZ,PHI, &
       EPSX,EPSY,EPSZ,CDEN,FKAPA,TMEMB, &
       QOSOR,QFOCUS,QPBC,QNPBC,QPRIN)
    !-----------------------------------------------------------------------
    !     Actual Poisson-Boltzmann iterator for discretized cubic lattice.
    !     It is assumed that all arrays have been prepared elsewhere.
    !
    !                        (in Z)
    !                        ip0+1=ip6
    !                               | ip3
    !                               | /
    !                               |/
    !                    ip1 ----- ip0 ----- ip2   (in X)
    !                              /|
    !                             / |
    !                           ip4 |
    !                    (in Y)    ip5=ip0-1
    !
    ! NOTE: CDEN and FKAPA already contained 4*pi/h and h*h, respectively.
    !
    use chm_kinds
    use memory
    use stream
    use consta
    use number
    implicit none
    REAL(CHM_REAL4)  PHI(*),EPSX(*),EPSY(*),EPSZ(*),CDEN(*),FKAPA(*)
    real(chm_real)  OMEGA,TOL,TMEMB
    INTEGER MAXITS,NCLX,NCLY,NCLZ
    LOGICAL QOSOR,QPRIN,QFOCUS,QPBC,QNPBC
    ! Local variables
    real(chm_real)  RESID,ANORM,E123,RJAC,Omega2
    INTEGER NCYZ,NC3,N
    INTEGER IG,JG,KG,IP0,IP0X,IP1X,IP2X,IP0Y,IP3Y,IP4Y,IP5Z,IP6Z
    INTEGER KSW
    !
    NCyz=NCLy*NCLz
    NC3=NCLx*NCyz

    ! conserve the initial mixing factor for successive calculations (?)
    OMEGA2=OMEGA

    IF(NCLx.eq.3 .and. NCLy.eq.3 .and. NCLz.eq.3) THEN
       RJAC=(COS(PI/NCLx)+COS(PI/NCLy)+COS(PI/NCLz))/THREE
       IF(PRNLEV.GT.10) WRITE(OUTU,'(3x,A,E15.5)') 'RJAC',RJAC
    ELSEIF(QOSOR) THEN
       call OPTRJAC(NCLX,NCLY,NCLZ,PHI,EPSX,EPSY,EPSZ,FKAPA,RJAC) 
       IF(PRNLEV.GT.10) WRITE(OUTU,'(3x,A,E15.5)') &
            'Optimal RJAC',RJAC
       IF(PRNLEV.GT.10) WRITE(OUTU,'(3x,A,E15.5)') 'RJAC', &
            (COS(PI/(NCLx-1))+COS(PI/(NCLy-1))+COS(PI/(NCLz-1)))/THREE
    ELSE
       RJAC=(COS(PI/(NCLx-1))+COS(PI/(NCLy-1))+COS(PI/(NCLz-1)))/THREE
       IF(PRNLEV.GT.10) WRITE(OUTU,'(3x,A,E15.5)') 'RJAC',RJAC
    ENDIF

    KSW=1

    IF(QPBC .AND. .NOT.QFOCUS  .AND. .NOT. QNPBC)THEN
       !     -------------------------------------------------
       ! periodic in XYZ

       DO N=1,MAXITS
          ANORM=ZERO

          ! x component
          DO IG=1,NCLx
             IP0X=(IG-1)*NCyz
             IP1X=-NCyz
             IF(IG.EQ.1) IP1X=NC3-NCyz    ! (NCLx-1)*NCyz
             IP2X=NCyz
             IF(IG.EQ.NCLx) IP2X=-(NC3-NCyz)

             ! y component
             DO JG=1,NCLy
                IP0Y=(JG-1)*NCLz+IP0X
                IP3Y=-NCLz
                IF(JG.EQ.1) IP3Y=NCyz-NCLz ! (NCLy-1)*NCLz
                IP4Y=NCLz
                IF(JG.EQ.NCLy) IP4Y=-(NCyz-NCLz)

                ! z component
                !wi               DO KG=1,NCLz
                DO KG=KSW,NCLZ-KSW+1,2
                   IP0 = IP0Y + KG
                   IP5Z=-1
                   IF(KG.EQ.1) IP5Z=NCLz-1
                   IP6Z=1
                   IF(KG.EQ.NCLz) IP6Z=-(NCLz-1)

                   E123=EPSX(IP0)+EPSY(IP0)+EPSZ(IP0)+ &
                        EPSX(IP0+IP1X)+EPSY(IP0+IP3Y)+EPSZ(IP0+IP5Z)+ &
                        FKAPA(IP0)

                   RESID=PHI(IP0+IP1X)*EPSX(IP0+IP1X)+ &
                        PHI(IP0+IP2X)*EPSX(IP0)+ &
                        PHI(IP0+IP3Y)*EPSY(IP0+IP3Y)+ &
                        PHI(IP0+IP4Y)*EPSY(IP0)+ &
                        PHI(IP0+IP5Z)*EPSZ(IP0+IP5Z)+ &
                        PHI(IP0+IP6Z)*EPSZ(IP0)- &
                        PHI(IP0)*E123+CDEN(IP0)

                   ! phi = phi_old + omega*(phi_new - phi_old)

                   ANORM=ANORM+ABS(RESID)
                   PHI(IP0)=PHI(IP0)+OMEGA*RESID/E123

                ENDDO
                KSW=3-KSW
             ENDDO
          ENDDO

          ANORM=ANORM/NC3

          IF(PRNLEV.GT.10) THEN
             IP0=NCyz*int(NCLx/2)+NCLz*int(NCLy/2)+NCLz/2+1
             WRITE(OUTU,'(3x,A,2X,I5,4E11.4)') &
                  'N,ANORM,DEPS,OMEGA,PHI',N,ANORM,TOL,OMEGA,PHI(IP0)
          ENDIF

          IF(N.EQ.1)THEN
             OMEGA=1./(1.-.5*RJAC**2)
          ELSE
             OMEGA=1./(1.-.25*RJAC**2*OMEGA)
          ENDIF

          IF((N.GT.1).AND.(ANORM.LT.TOL)) GOTO 19

       ENDDO

       !     ---------------------------------------------------------------
    ELSEIF(TMEMB.GT.ZERO  .AND. .NOT.QFOCUS  .AND. .NOT. QNPBC)THEN
       !     ---------------------------------------------------------------
       ! periodic in XY, Z fixed edge-value (in membrane)

       DO N=1,MAXITS
          ANORM=ZERO

          ! x component
          DO IG=1,NCLx
             IP0X=(IG-1)*NCyz
             IP1X=-NCyz
             IF(IG.EQ.1) IP1X=NC3-NCyz    ! (NCLx-1)*NCyz
             IP2X=NCyz
             IF(IG.EQ.NCLx) IP2X=-(NC3-NCyz)

             ! y component
             DO JG=1,NCLy
                IP0Y=(JG-1)*NCLz+IP0X
                IP3Y=-NCLz
                IF(JG.EQ.1) IP3Y=NCyz-NCLz ! (NCLy-1)*NCLz
                IP4Y=NCLz
                IF(JG.EQ.NCLy) IP4Y=-(NCyz-NCLz)

                ! z component
                DO KG=KSW+1,NCLZ-KSW,2
                   IP0 = IP0Y + KG

                   E123=EPSX(IP0)+EPSY(IP0)+EPSZ(IP0)+ &
                        EPSX(IP0+IP1X)+EPSY(IP0+IP3Y)+EPSZ(IP0-1)+ &
                        FKAPA(IP0)

                   RESID=PHI(IP0+IP1X)*EPSX(IP0+IP1X)+ &
                        PHI(IP0+IP2X)*EPSX(IP0)+ &
                        PHI(IP0+IP3Y)*EPSY(IP0+IP3Y)+ &
                        PHI(IP0+IP4Y)*EPSY(IP0)+ &
                        PHI(IP0-1)*EPSZ(IP0-1)+ &
                        PHI(IP0+1)*EPSZ(IP0)- &
                        PHI(IP0)*E123+CDEN(IP0)

                   ! phi = phi_old + omega*(phi_new - phi_old)

                   ANORM=ANORM+ABS(RESID)
                   PHI(IP0)=PHI(IP0)+OMEGA*RESID/E123

                ENDDO
                KSW=3-KSW
             ENDDO
          ENDDO

          ANORM=ANORM/NC3

          IF(PRNLEV.GT.10) THEN
             IP0=NCyz*int(NCLx/2)+NCLz*int(NCLy/2)+NCLz/2+1
             WRITE(OUTU,'(3x,A,2X,I5,4E11.4)') &
                  'N,ANORM,DEPS,OMEGA,PHI',N,ANORM,TOL,OMEGA,PHI(IP0)
          ENDIF

          IF(N.EQ.1)THEN
             OMEGA=1./(1.-.5*RJAC**2)
          ELSE
             OMEGA=1./(1.-.25*RJAC**2*OMEGA)
          ENDIF

          IF((N.GT.1).AND.(ANORM.LT.TOL)) GOTO 19

       ENDDO

       !     ----
    ELSE
       !     ----

       DO N=1,MAXITS
          ANORM=ZERO

          ! x component
          DO IG=2,NCLx-1
             IP0X=(IG-1)*NCyz

             ! y component
             DO JG=2,NCLy-1
                IP0Y=(JG-1)*NCLz+IP0X

                ! z component
                DO KG=KSW+1,NCLZ-KSW,2
                   IP0 = IP0Y + KG

                   E123=EPSX(IP0)+EPSY(IP0)+EPSZ(IP0)+ &
                        EPSX(IP0-NCyz)+EPSY(IP0-NCLz)+EPSZ(IP0-1)+ &
                        FKAPA(IP0)

                   RESID=PHI(IP0-NCyz)*EPSX(IP0-NCyz)+ &
                        PHI(IP0+NCyz)*EPSX(IP0)+ &
                        PHI(IP0-NCLz)*EPSY(IP0-NCLz)+ &
                        PHI(IP0+NCLz)*EPSY(IP0)+ &
                        PHI(IP0-1)*EPSZ(IP0-1)+ &
                        PHI(IP0+1)*EPSZ(IP0)- &
                        PHI(IP0)*E123+CDEN(IP0)

                   ! phi = phi_old + omega*(phi_new - phi_old)

                   ANORM=ANORM+ABS(RESID)
                   PHI(IP0)=PHI(IP0)+OMEGA*RESID/E123

                ENDDO
                KSW=3-KSW
             ENDDO
          ENDDO

          ANORM=ANORM/NC3

          !      write(6,*)' N: rjac, omega, anorm', n, rjac, omega, anorm
          IF(PRNLEV.GT.10) THEN
             IP0=NCyz*int(NCLx/2)+NCLz*int(NCLy/2)+NCLz/2+1
             WRITE(OUTU,'(3x,A,2X,I5,4E11.4)') &
                  'N,ANORM,DEPS,OMEGA,PHI',N,ANORM,TOL,OMEGA,PHI(IP0)
          ENDIF

          IF(N.EQ.1)THEN
             OMEGA=1./(1.-.5*RJAC**2)
          ELSE
             OMEGA=1./(1.-.25*RJAC**2*OMEGA)
          ENDIF

          IF((N.GT.1).AND.(ANORM.LT.TOL)) GOTO 19

       ENDDO

    ENDIF
    !     -----

    IF(PRNLEV.GE.2) WRITE(OUTU,'(3x,A,2X,2E13.6)') &
         'Calculation not converged, MAXITS is too small ',ANORM,TOL

19  CONTINUE

    OMEGA=OMEGA2

    IF(.not.QPRIN) RETURN
    IF(PRNLEV.GE.2) WRITE(OUTU,'(3X,A,I5)') 'Number of iterations: ',N
    !
    RETURN
  END SUBROUTINE OLDPBEQ1
  !
  SUBROUTINE PBEQ1(MAXITS,OMEGA,TOL,EPSW,EPSP,EPSM,KAPPA2, &
       NCLX,NCLY,NCLZ, &
       IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
       PHI,EPSX,EPSY,EPSZ,CDEN,FKAPA,TMEMB, &
       QOSOR,QFOCUS,QPBC,QNPBC,QPRIN)
    !-----------------------------------------------------------------------
    !     Actual Poisson-Boltzmann iterator for discretized cubic lattice.
    !     It is assumed that all arrays have been prepared elsewhere.
    !
    !                        (in Z)
    !                        ip0+1=ip6
    !                               | ip3
    !                               | /
    !                               |/
    !                    ip1 ----- ip0 ----- ip2   (in X)
    !                              /|
    !                             / |
    !                           ip4 |
    !                    (in Y)    ip5=ip0-1
    !
    ! NOTE: CDEN and FKAPA already contained 4*pi/h and h*h, respectively.
    !
    use chm_kinds
    use memory
    use stream
    use consta
    use number
    implicit none
    REAL(CHM_REAL4)  PHI(*),EPSX(*),EPSY(*),EPSZ(*),CDEN(*),FKAPA(*)
    real(chm_real)  OMEGA,TOL,TMEMB,EPSW,EPSP,EPSM,KAPPA2
    INTEGER IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN
    INTEGER MAXITS,NCLX,NCLY,NCLZ
    LOGICAL QOSOR,QPRIN,QFOCUS,QPBC,QNPBC
    ! Local variables
    real(chm_real)  RESID,ANORM,E123,RJAC,Omega2
    INTEGER NCYZ,NC3,N
    INTEGER IG,JG,KG,IP0,IP0X,IP1X,IP2X,IP0Y,IP3Y,IP4Y,IP5Z,IP6Z
    INTEGER KSW
    !
    NCyz=NCLy*NCLz
    NC3=NCLx*NCyz

    ! conserve the initial mixing factor for successive calculations (?)
    OMEGA2=OMEGA

    IF(NCLx.eq.3 .and. NCLy.eq.3 .and. NCLz.eq.3) THEN
       RJAC=(COS(PI/NCLx)+COS(PI/NCLy)+COS(PI/NCLz))/THREE
       IF(PRNLEV.GT.10) WRITE(OUTU,'(3x,A,E15.5)') 'RJAC',RJAC
    ELSEIF(QOSOR) THEN
       CALL OPTRJAC(NCLX,NCLY,NCLZ,PHI,EPSX,EPSY,EPSZ,FKAPA,RJAC)
       IF(PRNLEV.GT.10) WRITE(OUTU,'(3x,A,E15.5)') &
            'Optimal RJAC',RJAC
       IF(PRNLEV.GT.10) WRITE(OUTU,'(3x,A,E15.5)') 'RJAC', &
            (COS(PI/(NCLx-1))+COS(PI/(NCLy-1))+COS(PI/(NCLz-1)))/THREE
    ELSE
       RJAC=(COS(PI/(NCLx-1))+COS(PI/(NCLy-1))+COS(PI/(NCLz-1)))/THREE
       IF(PRNLEV.GT.10) WRITE(OUTU,'(3x,A,E15.5)') 'RJAC',RJAC
    ENDIF

    KSW=1

    IF(QPBC .AND. .NOT.QFOCUS  .AND. .NOT. QNPBC)THEN
       !     -------------------------------------------------
       ! periodic in XYZ

       DO N=1,MAXITS
          ANORM=ZERO

          IF(EPSW.eq.EPSP .and. EPSM.eq.EPSP .and. KAPPA2.eq.ZERO) THEN
             E123=SIX*EPSW
             DO IG=1,NCLx
                IP0X=(IG-1)*NCyz
                IP1X=-NCyz
                IF(IG.EQ.1) IP1X=NC3-NCyz ! (NCLx-1)*NCyz
                IP2X=NCyz
                IF(IG.EQ.NCLx) IP2X=-(NC3-NCyz)
                DO JG=1,NCLy
                   IP0Y=(JG-1)*NCLz+IP0X
                   IP3Y=-NCLz
                   IF(JG.EQ.1) IP3Y=NCyz-NCLz ! (NCLy-1)*NCLz
                   IP4Y=NCLz
                   IF(JG.EQ.NCLy) IP4Y=-(NCyz-NCLz)
                   !wi               DO KG=1,NCLz
                   DO KG=KSW,NCLZ-KSW+1,2
                      IP0 = IP0Y + KG
                      IP5Z=-1
                      IF(KG.EQ.1) IP5Z=NCLz-1
                      IP6Z=1
                      IF(KG.EQ.NCLz) IP6Z=-(NCLz-1)
                      RESID=(PHI(IP0+IP1X)+PHI(IP0+IP2X)+ &
                           PHI(IP0+IP3Y)+PHI(IP0+IP4Y)+ &
                           PHI(IP0+IP5Z)+PHI(IP0+IP6Z))*EPSW- &
                           PHI(IP0)*E123+CDEN(IP0)
                      ANORM=ANORM+ABS(RESID)
                      PHI(IP0)=PHI(IP0)+OMEGA*RESID/E123
                   ENDDO
                   KSW=3-KSW
                ENDDO
             ENDDO
          ELSE
             ! x component
             DO IG=1,NCLx
                IP0X=(IG-1)*NCyz
                IP1X=-NCyz
                IF(IG.EQ.1) IP1X=NC3-NCyz ! (NCLx-1)*NCyz
                IP2X=NCyz
                IF(IG.EQ.NCLx) IP2X=-(NC3-NCyz)
                ! y component
                DO JG=1,NCLy
                   IP0Y=(JG-1)*NCLz+IP0X
                   IP3Y=-NCLz
                   IF(JG.EQ.1) IP3Y=NCyz-NCLz ! (NCLy-1)*NCLz
                   IP4Y=NCLz
                   IF(JG.EQ.NCLy) IP4Y=-(NCyz-NCLz)
                   ! z component
                   !wi               DO KG=1,NCLz
                   DO KG=KSW,NCLZ-KSW+1,2
                      IP0 = IP0Y + KG
                      IP5Z=-1
                      IF(KG.EQ.1) IP5Z=NCLz-1
                      IP6Z=1
                      IF(KG.EQ.NCLz) IP6Z=-(NCLz-1)
                      E123=EPSX(IP0)+EPSY(IP0)+EPSZ(IP0)+ &
                           EPSX(IP0+IP1X)+EPSY(IP0+IP3Y)+EPSZ(IP0+IP5Z)+ &
                           FKAPA(IP0)
                      RESID=PHI(IP0+IP1X)*EPSX(IP0+IP1X)+ &
                           PHI(IP0+IP2X)*EPSX(IP0)+ &
                           PHI(IP0+IP3Y)*EPSY(IP0+IP3Y)+ &
                           PHI(IP0+IP4Y)*EPSY(IP0)+ &
                           PHI(IP0+IP5Z)*EPSZ(IP0+IP5Z)+ &
                           PHI(IP0+IP6Z)*EPSZ(IP0)- &
                           PHI(IP0)*E123+CDEN(IP0)

                      ! phi = phi_old + omega*(phi_new - phi_old)
                      ANORM=ANORM+ABS(RESID)
                      PHI(IP0)=PHI(IP0)+OMEGA*RESID/E123

                   ENDDO
                   KSW=3-KSW
                ENDDO
             ENDDO
          ENDIF

          ANORM=ANORM/NC3

          IF(PRNLEV.GT.10) THEN
             IP0=NCyz*int(NCLx/2)+NCLz*int(NCLy/2)+NCLz/2+1
             WRITE(OUTU,'(3x,A,2X,I5,4E11.4)') &
                  'N,ANORM,DEPS,OMEGA,PHI',N,ANORM,TOL,OMEGA,PHI(IP0)
          ENDIF

          IF(N.EQ.1)THEN
             OMEGA=1./(1.-.5*RJAC**2)
          ELSE
             OMEGA=1./(1.-.25*RJAC**2*OMEGA)
          ENDIF

          IF((N.GT.1).AND.(ANORM.LT.TOL)) GOTO 19

       ENDDO

       !     --------------------------------------------------------------
    ELSEIF(TMEMB.GT.ZERO .AND. .NOT.QFOCUS  .AND. .NOT. QNPBC)THEN
       !     --------------------------------------------------------------
       ! periodic in XY, Z fixed edge-value (in membrane)

       DO N=1,MAXITS
          ANORM=ZERO

          IF(EPSW.eq.EPSP .and. EPSM.eq.EPSP .and. KAPPA2.eq.ZERO) THEN
             E123=SIX*EPSW
             DO IG=1,NCLx
                IP0X=(IG-1)*NCyz
                IP1X=-NCyz
                IF(IG.EQ.1) IP1X=NC3-NCyz ! (NCLx-1)*NCyz
                IP2X=NCyz
                IF(IG.EQ.NCLx) IP2X=-(NC3-NCyz)
                DO JG=1,NCLy
                   IP0Y=(JG-1)*NCLz+IP0X
                   IP3Y=-NCLz
                   IF(JG.EQ.1) IP3Y=NCyz-NCLz ! (NCLy-1)*NCLz
                   IP4Y=NCLz
                   IF(JG.EQ.NCLy) IP4Y=-(NCyz-NCLz)
                   DO KG=KSW+1,NCLZ-KSW,2
                      IP0 = IP0Y + KG
                      RESID=(PHI(IP0+IP1X)+PHI(IP0+IP2X)+ &
                           PHI(IP0+IP3Y)+PHI(IP0+IP4Y)+ &
                           PHI(IP0-1)+PHI(IP0+1))*EPSW- &
                           PHI(IP0)*E123+CDEN(IP0)
                      ANORM=ANORM+ABS(RESID)
                      PHI(IP0)=PHI(IP0)+OMEGA*RESID/E123
                   ENDDO
                   KSW=3-KSW
                ENDDO
             ENDDO
          ELSE
             !  x component
             DO IG=1,NCLx
                IP0X=(IG-1)*NCyz
                IP1X=-NCyz
                IF(IG.EQ.1) IP1X=NC3-NCyz ! (NCLx-1)*NCyz
                IP2X=NCyz
                IF(IG.EQ.NCLx) IP2X=-(NC3-NCyz)
                !  y component
                DO JG=1,NCLy
                   IP0Y=(JG-1)*NCLz+IP0X
                   IP3Y=-NCLz
                   IF(JG.EQ.1) IP3Y=NCyz-NCLz ! (NCLy-1)*NCLz
                   IP4Y=NCLz
                   IF(JG.EQ.NCLy) IP4Y=-(NCyz-NCLz)
                   !  z component
                   DO KG=KSW+1,NCLZ-KSW,2
                      IP0 = IP0Y + KG
                      E123=EPSX(IP0)+EPSY(IP0)+EPSZ(IP0)+ &
                           EPSX(IP0+IP1X)+EPSY(IP0+IP3Y)+EPSZ(IP0-1)+ &
                           FKAPA(IP0)
                      RESID=PHI(IP0+IP1X)*EPSX(IP0+IP1X)+ &
                           PHI(IP0+IP2X)*EPSX(IP0)+ &
                           PHI(IP0+IP3Y)*EPSY(IP0+IP3Y)+ &
                           PHI(IP0+IP4Y)*EPSY(IP0)+ &
                           PHI(IP0-1)*EPSZ(IP0-1)+ &
                           PHI(IP0+1)*EPSZ(IP0)- &
                           PHI(IP0)*E123+CDEN(IP0)

                      ! phi = phi_old + omega*(phi_new - phi_old)
                      ANORM=ANORM+ABS(RESID)
                      PHI(IP0)=PHI(IP0)+OMEGA*RESID/E123
                   ENDDO
                   KSW=3-KSW
                ENDDO
             ENDDO
          ENDIF

          ANORM=ANORM/NC3

          IF(PRNLEV.GT.10) THEN
             IP0=NCyz*int(NCLx/2)+NCLz*int(NCLy/2)+NCLz/2+1
             WRITE(OUTU,'(3x,A,2X,I5,4E11.4)') &
                  'N,ANORM,DEPS,OMEGA,PHI',N,ANORM,TOL,OMEGA,PHI(IP0)
          ENDIF

          IF(N.EQ.1)THEN
             OMEGA=1./(1.-.5*RJAC**2)
          ELSE
             OMEGA=1./(1.-.25*RJAC**2*OMEGA)
          ENDIF

          IF((N.GT.1).AND.(ANORM.LT.TOL)) GOTO 19

       ENDDO

       !     ----
    ELSE
       !     ----

       IF(mod(IZMIN,2).ne.0) IZMIN=IZMIN-1
       IF(mod(IZMAX,2).eq.0) IZMAX=IZMAX+1
       IF(mod(IZMAX,2).ne.0) IZMAX=IZMAX+2

       ! for focussing
       !wi      IF(IXMIN.lt.2.or.QFOCUS) IXMIN=2
       !wi      IF(IXMAX.gt.(NCLx-1).or.QFOCUS) IXMAX=NCLx-1
       !wi      IF(IYMIN.lt.2.or.QFOCUS) IYMIN=2
       !wi      IF(IYMAX.gt.(NCLy-1).or.QFOCUS) IYMAX=NCLy-1
       !wi      IF(IZMIN.lt.2.or.QFOCUS) IZMIN=2
       !wi      IF(IZMAX.gt.(NCLz-1).or.QFOCUS) IZMAX=NCLz-1
       IF(IXMIN.lt.2) IXMIN=2
       IF(IXMAX.gt.NCLx-1) IXMAX=NCLx-1
       IF(IYMIN.lt.2) IYMIN=2
       IF(IYMAX.gt.NCLy-1) IYMAX=NCLy-1
       IF(IZMIN.lt.2) IZMIN=2
       IF(IZMAX.gt.NCLz-1) IZMAX=NCLz-1

       ! some constrictions for Red-black Gauss-Seidel relaxation
       IF(mod(IXMIN,2).ne.0) THEN
          IF(mod(IXMAX,2).eq.0 .and. IXMAX.ne.(NCLx-1)) IXMAX=IXMAX+1
          IF(mod(IXMAX,2).eq.0 .and. IXMAX.eq.(NCLx-1)) IXMAX=IXMAX-1
       ELSE
          IF(IXMIN.eq.2) THEN
             IF(mod(IXMAX,2).ne.0 .and. IXMAX.ne.(NCLx-1)) IXMAX=IXMAX+1
             IF(mod(IXMAX,2).ne.0 .and. IXMAX.eq.(NCLx-1)) IXMAX=IXMAX-1
          ELSE
             IXMIN=IXMIN-1
             IF(mod(IXMAX,2).eq.0 .and. IXMAX.ne.(NCLx-1)) IXMAX=IXMAX+1
             IF(mod(IXMAX,2).eq.0 .and. IXMAX.eq.(NCLx-1)) IXMAX=IXMAX-1
          ENDIF
       ENDIF
       IF(mod(IYMIN,2).ne.0) THEN
          IF(mod(IYMAX,2).eq.0 .and. IYMAX.ne.(NCLy-1)) IYMAX=IYMAX+1
          IF(mod(IYMAX,2).eq.0 .and. IYMAX.eq.(NCLy-1)) IYMAX=IYMAX-1
       ELSE
          IF(IYMIN.eq.2) THEN
             IF(mod(IYMAX,2).ne.0 .and. IYMAX.ne.(NCLy-1)) IYMAX=IYMAX+1
             IF(mod(IYMAX,2).ne.0 .and. IYMAX.eq.(NCLy-1)) IYMAX=IYMAX-1
          ELSE
             IYMIN=IYMIN-1
             IF(mod(IYMAX,2).eq.0 .and. IYMAX.ne.(NCLy-1)) IYMAX=IYMAX+1
             IF(mod(IYMAX,2).eq.0 .and. IYMAX.eq.(NCLy-1)) IYMAX=IYMAX-1
          ENDIF
       ENDIF

       ! for the vacuum environment
       IF(EPSW.eq.EPSP .and. KAPPA2.eq.ZERO) THEN
          IXMIN=NCLx
          IXMAX=NCLx-1
       ENDIF

       DO N=1,MAXITS
          ANORM=ZERO

          DO IG=2,IXMIN-1
             IP0X=(IG-1)*NCyz
             DO JG=2,NCLy-1
                IP0Y=(JG-1)*NCLz+IP0X
                DO KG=KSW+1,NCLZ-KSW,2
                   IP0 = IP0Y + KG
                   E123=SIX*EPSW+FKAPA(IP0)
                   RESID=(PHI(IP0-NCyz)+PHI(IP0+NCyz)+ &
                        PHI(IP0-NCLz)+PHI(IP0+NCLz)+ &
                        PHI(IP0-1)+PHI(IP0+1))*EPSW - &
                        PHI(IP0)*E123+CDEN(IP0)
                   ANORM=ANORM+ABS(RESID)
                   PHI(IP0)=PHI(IP0)+OMEGA*RESID/E123
                ENDDO
                KSW=3-KSW
             ENDDO
          ENDDO

          DO IG=IXMIN,IXMAX
             IP0X=(IG-1)*NCyz
             DO JG=2,IYMIN-1
                IP0Y=(JG-1)*NCLz+IP0X
                DO KG=KSW+1,NCLZ-KSW,2
                   IP0 = IP0Y + KG
                   E123=SIX*EPSW+FKAPA(IP0)
                   RESID=(PHI(IP0-NCyz)+PHI(IP0+NCyz)+ &
                        PHI(IP0-NCLz)+PHI(IP0+NCLz)+ &
                        PHI(IP0-1)+PHI(IP0+1))*EPSW - &
                        PHI(IP0)*E123
                   ANORM=ANORM+ABS(RESID)
                   PHI(IP0)=PHI(IP0)+OMEGA*RESID/E123
                ENDDO
                KSW=3-KSW
             ENDDO

             DO JG=IYMIN,IYMAX
                IP0Y=(JG-1)*NCLz+IP0X
                DO KG=KSW+1,IZMIN-1,2
                   IP0 = IP0Y + KG
                   E123=SIX*EPSW+FKAPA(IP0)
                   RESID=(PHI(IP0-NCyz)+PHI(IP0+NCyz)+ &
                        PHI(IP0-NCLz)+PHI(IP0+NCLz)+ &
                        PHI(IP0-1)+PHI(IP0+1))*EPSW - &
                        PHI(IP0)*E123
                   ANORM=ANORM+ABS(RESID)
                   PHI(IP0)=PHI(IP0)+OMEGA*RESID/E123
                ENDDO

                DO KG=IZMIN,IZMAX,2
                   IP0 = IP0Y + KG
                   E123=EPSX(IP0)+EPSY(IP0)+EPSZ(IP0)+ &
                        EPSX(IP0-NCyz)+EPSY(IP0-NCLz)+EPSZ(IP0-1)+ &
                        FKAPA(IP0)
                   RESID=PHI(IP0-NCyz)*EPSX(IP0-NCyz)+ &
                        PHI(IP0+NCyz)*EPSX(IP0)+ &
                        PHI(IP0-NCLz)*EPSY(IP0-NCLz)+ &
                        PHI(IP0+NCLz)*EPSY(IP0)+ &
                        PHI(IP0-1)*EPSZ(IP0-1)+ &
                        PHI(IP0+1)*EPSZ(IP0)- &
                        PHI(IP0)*E123+CDEN(IP0)
                   ! phi = phi_old + omega*(phi_new - phi_old)
                   ANORM=ANORM+ABS(RESID)
                   PHI(IP0)=PHI(IP0)+OMEGA*RESID/E123
                ENDDO

                DO KG=IZMAX+1,NCLZ-KSW,2
                   IP0 = IP0Y + KG
                   E123=SIX*EPSW+FKAPA(IP0)
                   RESID=(PHI(IP0-NCyz)+PHI(IP0+NCyz)+ &
                        PHI(IP0-NCLz)+PHI(IP0+NCLz)+ &
                        PHI(IP0-1)+PHI(IP0+1))*EPSW - &
                        PHI(IP0)*E123
                   ANORM=ANORM+ABS(RESID)
                   PHI(IP0)=PHI(IP0)+OMEGA*RESID/E123
                ENDDO
                IZMIN=IZMIN-KSW
                IZMAX=IZMAX+KSW
                KSW=3-KSW
                IZMIN=IZMIN+KSW
                IZMAX=IZMAX-KSW
             ENDDO

             DO JG=IYMAX+1,NCLy-1
                IP0Y=(JG-1)*NCLz+IP0X
                DO KG=KSW+1,NCLZ-KSW,2
                   IP0 = IP0Y + KG
                   E123=SIX*EPSW+FKAPA(IP0)
                   RESID=(PHI(IP0-NCyz)+PHI(IP0+NCyz)+ &
                        PHI(IP0-NCLz)+PHI(IP0+NCLz)+ &
                        PHI(IP0-1)+PHI(IP0+1))*EPSW - &
                        PHI(IP0)*E123
                   ANORM=ANORM+ABS(RESID)
                   PHI(IP0)=PHI(IP0)+OMEGA*RESID/E123
                ENDDO
                KSW=3-KSW
             ENDDO
          ENDDO

          DO IG=IXMAX+1,NCLx-1
             IP0X=(IG-1)*NCyz
             DO JG=2,NCLy-1
                IP0Y=(JG-1)*NCLz+IP0X
                DO KG=KSW+1,NCLZ-KSW,2
                   IP0 = IP0Y + KG
                   E123=SIX*EPSW+FKAPA(IP0)
                   RESID=(PHI(IP0-NCyz)+PHI(IP0+NCyz)+ &
                        PHI(IP0-NCLz)+PHI(IP0+NCLz)+ &
                        PHI(IP0-1)+PHI(IP0+1))*EPSW - &
                        PHI(IP0)*E123
                   ANORM=ANORM+ABS(RESID)
                   PHI(IP0)=PHI(IP0)+OMEGA*RESID/E123
                ENDDO
                KSW=3-KSW
             ENDDO
          ENDDO

          ANORM=ANORM/NC3

          !      write(6,*)' N: rjac, omega, anorm', n, rjac, omega, anorm
          IF(PRNLEV.GT.10) THEN
             IP0=NCyz*int(NCLx/2)+NCLz*int(NCLy/2)+NCLz/2+1
             WRITE(OUTU,'(3x,A,2X,I5,4E11.4)') &
                  'N,ANORM,DEPS,OMEGA,PHI',N,ANORM,TOL,OMEGA,PHI(IP0)
          ENDIF

          IF(N.EQ.1)THEN
             OMEGA=1./(1.-.5*RJAC**2)
          ELSE
             OMEGA=1./(1.-.25*RJAC**2*OMEGA)
          ENDIF

          IF((N.GT.1).AND.(ANORM.LT.TOL)) GOTO 19

       ENDDO

    ENDIF
    !     -----

    IF(PRNLEV.GE.2) WRITE(OUTU,'(3x,A,2X,2E13.6)') &
         'Calculation not converged, MAXITS is too small ',ANORM,TOL

19  CONTINUE

    OMEGA=OMEGA2

    IF(.not.QPRIN) RETURN
    IF(PRNLEV.GE.2) WRITE(OUTU,'(3X,A,I5)') 'Number of iterations: ',N
    !
    RETURN
  END SUBROUTINE PBEQ1
  !
  SUBROUTINE OPTRJAC(NCLX,NCLY,NCLZ,PHI,EPSX,EPSY,EPSZ, &
       FKAPA,RJAC)
    !-----------------------------------------------------------------------
    !     The optimal SOR parameter (RJAC) is obtained.
    !     Ref: A. Nicholls and B. Honig, J. Comput. Chem, 12(4),435-445 (1991)
    !
    use chm_kinds
    use stream
    use consta
    use number
    use memory
    implicit none
    REAL(CHM_REAL4)  PHI(*),EPSX(*),EPSY(*),EPSZ(*),FKAPA(*)

    INTEGER NCLX,NCLY,NCLZ

    ! Local variables
    real(chm_real4),allocatable, dimension(:) :: phi0
    real(chm_real)  RJAC
    INTEGER NCyz,N
    INTEGER IG,JG,KG,IP0,IP0X,IP0Y
    INTEGER KSW
    real(chm_real)  Sx0,Sy0,Sz0,Cx0,Cy0,Cz0,Cx1,Sx1,Cy1,Sy1,Cz1,Sz1
    real(chm_real)  Sx,Sy,Sz,Cx,Cy,Cz
    real(chm_real)  MOMENT(3),CM(3)
    real(chm_real)  TMPPHI0,FACTOR,NORMSQ,BOT,TOP,MSUM
    !
    NCyz=NCLy*NCLz
    call chmalloc('pbeq.src','OLDPBEQ1','PHI0',NCyz*NCLX,cr4=PHI0)

    !
    ! Fill PHI0 and PHI with the solution of the Laplace equation
    ! with maximum eigenvalue
    !

    ! boundary points
    IP0X=(NCLX-1)*NCYZ
    DO JG=1,NCLY
       IP0Y=(JG-1)*NCLZ
       DO KG=2,NCLZ-1
          IP0=IP0Y+KG
          PHI0(IP0)=0.0
          IP0=IP0X+IP0
          PHI0(IP0)=0.0
       ENDDO
    ENDDO
    DO IG=2,NCLX-1
       IP0X=(IG-1)*NCYZ
       DO KG=2,NCLZ-1
          IP0=IP0X+KG
          PHI0(IP0)=0.0
          IP0=IP0+NCYZ-NCLZ
          PHI0(IP0)=0.0
       ENDDO
    ENDDO
    DO IG=1,NCLX
       IP0X=(IG-1)*NCYZ
       DO JG=1,NCLY
          IP0=IP0X+(JG-1)*NCLZ+1
          PHI0(IP0)=0.0
          IP0=IP0+NCLZ-1
          PHI0(IP0)=0.0
       ENDDO
    ENDDO

    ! interior points
    Sx0=SIN(PI/(NCLx-1))
    Sy0=SIN(PI/(NCLy-1))
    Sz0=SIN(PI/(NCLz-1))
    Cx0=COS(PI/(NCLx-1))
    Cy0=COS(PI/(NCLy-1))
    Cz0=COS(PI/(NCLz-1))
    FACTOR=TWO*SQRT(TWO)
    NORMSQ=ZERO

    Cx1=Cx0
    Sx1=Sx0
    Sx=Sx0
    DO IG=2,NCLx-1
       IP0X=(IG-1)*NCyz
       Cy1=Cy0
       Sy1=Sx0
       Sy=Sy0
       DO JG=2,NCLy-1
          IP0Y=(JG-1)*NCLz+IP0X
          Cz1=Cz0
          Sz1=Sz0
          Sz=Sz0
          DO KG=2,NCLZ-1
             IP0 = IP0Y + KG

             TMPPHI0=FACTOR*SX*SY*SZ
             PHI0(IP0)=TMPPHI0
             PHI(IP0)=TMPPHI0
             NORMSQ=NORMSQ+TMPPHI0*TMPPHI0

             Sz=Sz1*Cz0+Cz1*Sz0
             Cz=Cz1*Cz0-Sz1*Sz0
             Sz1=Sz
             Cz1=Cz
          ENDDO
          Sy=Sy1*Cy0+Cy1*Sy0
          Cy=Cy1*Cy0-Sy1*Sy0
          Sy1=Sy
          Cy1=Cy
       ENDDO
       Sx=Sx1*Cx0+Cx1*Sx0
       Cx=Cx1*Cx0-Sx1*Sx0
       Sx1=Sx
       Cx1=Cx
    ENDDO

    !      DO IG=1,NCLx
    !         IP0X=(IG-1)*NCyz
    !         Sx=SIN(PI*(IG-1)/(NCLx-1))
    !         DO JG=1,NCLy
    !            IP0Y=(JG-1)*NCLz+IP0X
    !            Sy=SIN(PI*(JG-1)/(NCLy-1))
    !            DO KG=1,NCLZ
    !               IP0 = IP0Y + KG
    !               Sz=SIN(PI*(KG-1)/(NCLz-1))
    !
    !               PHI0(IP0)=FACTOR*SX*SY*SZ
    !               PHI(IP0)=PHI0(IP0)
    !               NORMSQ=NORMSQ+PHI0(IP0)*PHI0(IP0)
    !            ENDDO
    !         ENDDO
    !      ENDDO

    !
    ! Get 3 moments from Gauss-Seidel algorithm with ZERO CHARGEs
    !
    KSW=1
    DO N=1,3
       MOMENT(N)=ZERO

       DO IG=2,NCLx-1
          IP0X=(IG-1)*NCyz
          DO JG=2,NCLy-1
             IP0Y=(JG-1)*NCLz+IP0X
             DO KG=KSW+1,NCLZ-KSW,2
                IP0 = IP0Y + KG

                BOT=EPSX(IP0)+EPSY(IP0)+EPSZ(IP0)+ &
                     EPSX(IP0-NCyz)+EPSY(IP0-NCLz)+EPSZ(IP0-1)+ &
                     FKAPA(IP0)

                TOP=PHI(IP0-NCyz)*EPSX(IP0-NCyz)+ &
                     PHI(IP0+NCyz)*EPSX(IP0)+ &
                     PHI(IP0-NCLz)*EPSY(IP0-NCLz)+ &
                     PHI(IP0+NCLz)*EPSY(IP0)+ &
                     PHI(IP0-1)*EPSZ(IP0-1)+ &
                     PHI(IP0+1)*EPSZ(IP0)

                PHI(IP0)=TOP/BOT

             ENDDO
             KSW=3-KSW
          ENDDO
       ENDDO

       MSUM=ZERO
       DO IG=2,NCLx-1
          IP0X=(IG-1)*NCyz
          DO JG=2,NCLy-1
             IP0Y=(JG-1)*NCLz+IP0X
             DO KG=2,NCLZ-1
                IP0 = IP0Y + KG

                MSUM=MSUM+PHI(IP0)*PHI0(IP0)

             ENDDO
          ENDDO
       ENDDO

       MOMENT(N)=MSUM/NORMSQ

    ENDDO
    !
    ! Calculate the connected-moment
    !
    CM(1)=MOMENT(1)
    CM(2)=MOMENT(2)-CM(1)*MOMENT(1)
    CM(3)=MOMENT(3)-CM(1)*MOMENT(2)-TWO*CM(2)*MOMENT(1)

    !
    ! Get the optmal relaxation parameter
    !
    RJAC=CM(1)-CM(2)*CM(2)/CM(3)
    !
    ! Initialize PHI
    !
    DO IG=2,NCLx-1
       IP0X=(IG-1)*NCyz
       DO JG=2,NCLy-1
          IP0Y=(JG-1)*NCLz+IP0X
          DO KG=2,NCLZ-1
             IP0 = IP0Y + KG

             PHI(IP0)=0.0
             PHI0(IP0)=0.0

          ENDDO
       ENDDO
    ENDDO
    !
    call chmdealloc('pbeq.src','OLDPBEQ1','PHI0',NCyz*NCLX,cr4=PHI0)
    !
    RETURN
  END SUBROUTINE OPTRJAC
  !
  SUBROUTINE PBEQ2(NATOM,X,Y,Z,CG,QENVV,QPRIN)
    !-----------------------------------------------------------------------
    !     Full MultiGrid (FMG) Algorithm for PBEQ.
    !
    use chm_kinds
    use chm_types
    use number
    use memory
    use stream
    implicit none
    real(chm_real4),allocatable,dimension(:) :: IPHIB
    INTEGER NATOM
    real(chm_real)  X(*),Y(*),Z(*),CG(*)
    logical qenvv,qprin
    ! Local variables
    integer,parameter::  NG=9
    type(chm_ptr4) ::  ires(NG),irhs(NG), iphi(NG-1),icden(NG-1), &
         iepsx(NG-1),iepsy(NG-1),iepsz(NG-1),ikappa(NG-1)
    integer  i,icyc,ii,ipost,ipre,nf,nn,nn3,Q
    real(chm_real)   idcel,averes,anorm

    !
    !     GET GRID PARAMETERS FOR COARSE GRIDS
    !     ------------------------------------
    call chmalloc('pbeq.src','PBEQ2','IPHIB',1,cr4=IPHIB)
    i_loop: do i=ngridpb,1,-1
       nn=2**i+1
       nn3=nn**3
       ! allocate storage for ires,irhs,iphi,icden,iepsx,iepsy,iepsz,ikappa
       call chmalloc('pbeq.src','PBEQ2','IRES(I)',NN3,cr4p=IRES(I)%A)
       call chmalloc('pbeq.src','PBEQ2','IRHS(I)',NN3,cr4p=IRHS(I)%A)
       IF(I.EQ.NGRIDpb) cycle i_loop
       call chmalloc('pbeq.src','PBEQ2','IPHI(I)',NN3,cr4p=IPHI(I)%A)
       call chmalloc('pbeq.src','PBEQ2','ICDEN(I)',NN3,cr4p=ICDEN(I)%A)
       call chmalloc('pbeq.src','PBEQ2','IEPSX(I)',NN3,cr4p=IEPSX(I)%A)
       call chmalloc('pbeq.src','PBEQ2','IEPSY(I)',NN3,cr4p=IEPSY(I)%A)
       call chmalloc('pbeq.src','PBEQ2','IEPSZ(I)',NN3,cr4p=IEPSZ(I)%A)
       call chmalloc('pbeq.src','PBEQ2','IKAPPA(I)',NN3,cr4p=IKAPPA(I)%A)
       IDCEL    = DCEL*TWO**(NGRIDpb-I)

       ! get icden,iepsx,iepsy,iepsz,ikappa for coarse grids
       ! to call mayer, need to
       !                change DCEL (done) -> idcel
       !                add    QREEN,QSMTH to pbeq.f90  (done)
       !                skip   boundary potential calculation in MAYER (QZERO=.true.)
       ! for the force calculation, QENVV(?Vacuum ENVironment)
       IF(QENVV) THEN
          CALL MAYER(NTPRP,LSTPRP,ONE,ONE,ZERO, &
                                !                                        epsp=epsw kappa2
               X,Y,Z,CG,PBRAD,nn,nn,nn,idcel,WATR,IONR, &
               ikappa(i)%A,iepsx(i)%A,iepsy(i)%A, &
               iepsz(i)%A,SWIN,icden(i)%A,iphi(i)%A, &
               ZERO,TMEMB,ZMEMB,ONE,NIMGB, &
                                !                vmemb            epsm
               HTMEMB,ONE,HDENS,POSALP,NEGALP,QCHRG,POFFST,NOFFST, &
                                !                       epsh
               DROPLET,ONE,XDROPLET,YDROPLET,ZDROPLET,Qdtom,Qdkap, &
               ONE,LXMAX,LYMAX,LZMAX,LXMIN,LYMIN,LZMIN,Qbtom,Qbkap, &
                                !                epsb
               ONE,Xcyln,Ycyln,Zcyln,Rcyln,Hcyln,Qctom,Qckap, &
                                !                epsc
               EPSEC,Xcone,Ycone,Zcone,Ax,By,Cz,Qectom,Qeckap, &
                                !                epsec
               -1,-1,&
               TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
               Q,Q,Q,Q,Q,Q, &
                                !                IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN,
               QKEEP,.false.,.true.,QBSPL,QA,QB, &
                                !                       QREEN   QSMTH
               RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
               RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
               MAJOR,MINOR,QPROLATE, &
               .false.,.true.,.false.,IPHIB,.false.,QNPBC, &
                                !                QINTBP  QZERO  QFOCUS                QPBC
               QNONLINEAR,QPARTLINEAR,QFKAP, &
               IPOX,IPOY,IPOZ,MAPT,NATOM)
       ELSE
          CALL MAYER(NTPRP,LSTPRP,EPSW,EPSP,KAPPA2, &
               X,Y,Z,CG,PBRAD,nn,nn,nn,idcel,WATR,IONR, &
               ikappa(i)%A,iepsx(i)%A,iepsy(i)%A, &
               iepsz(i)%A,SWIN,icden(i)%A,iphi(i)%A, &
               VMEMB,TMEMB,ZMEMB,EPSM,NIMGB, &
               HTMEMB,EPSH,HDENS,POSALP,NEGALP,QCHRG,POFFST,NOFFST, &
               DROPLET,EPSD,XDROPLET,YDROPLET,ZDROPLET,Qdtom,Qdkap, &
               EPSB,LXMAX,LYMAX,LZMAX,LXMIN,LYMIN,LZMIN,Qbtom,Qbkap, &
               EPSC,Xcyln,Ycyln,Zcyln,Rcyln,Hcyln,Qctom,Qckap, &
               EPSEC,Xcone,Ycone,Zcone,Ax,By,Cz,Qectom,Qeckap, &
               -1,-1, &
               TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
               Q,Q,Q,Q,Q,Q, &
                                !                 IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN,
               QKEEP,QREEN,QSMTH,QBSPL,QA,QB, &
               RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
               RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
               MAJOR,MINOR,QPROLATE, &
               .false.,.true.,.false.,IPHIB,.false.,QNPBC, &
                                !                QINTBP  QZERO  QFOCUS                QPBC
               QNONLINEAR,QPARTLINEAR,QFKAP, &
               IPOX,IPOY,IPOZ,MAPT,NATOM)
       ENDIF

       ! INJect Boundary Potential from the finest grid to coarse grids
       if(i.eq.ngridpb-1) then
          IF(QENVV) THEN
             CALL INJBP(nn,IPHIP,iphi(i)%A)
          ELSE
             CALL INJBP(nn,IPHIW,iphi(i)%A)
          ENDIF
       else
          CALL INJBP(nn,iphi(i+1)%A,iphi(i)%A)
       endif

    enddo i_loop

    ! adjust ixmax,iymax,izmax,ixmin,iymin,and izmin
    IF(QENVV) THEN
       ixmin=2**ngridpb+1
       ixmax=ixmin-1
    ELSE
       call adjxyz(ngridpb,ixmax,iymax,izmax,ixmin,iymin,izmin, &
            epsw,epsp,kappa2)
    ENDIF

    ! get the initial solution on coarsest grid
    call slvsml(iphi(1)%A,icden(1)%A,ikappa(1)%A,iepsx(1)%A,iepsy(1)%A,iepsz(1)%A)

    !
    !     NESTED ITERATION
    !     ----------------

    nn=3

    iloop2: do i=2,ngridpb
       nn=2*nn-1
       ! interpolate from coarse grid to next finer grid
       ! set up r.h.s.
       if(i.eq.ngridpb) then
          if(QENVV) then
             call interpol(nn,IPHIP,iphi(i-1)%A)
          else
             call interpol(nn,IPHIW,iphi(i-1)%A)
          endif
          call cprho(nn,ICHC,irhs(i)%A)
       else
          call interpol(nn,iphi(i)%A,iphi(i-1)%A)
          call cprho(nn,icden(i)%A,irhs(i)%A)
       endif

       do icyc=1,NCYC                  !! V-cycle loop !!
          nf=nn
          do ii=i,2,-1                  ! (1) downward stoke of the V

             if(ii.eq.ngridpb) then
                do ipre=1,NPRE          !     pre-smoothing
                   if(QENVV) then
                      call relax1(nf,IPHIP,irhs(ii)%A, &
                           MAY,MAYX,MAYY,MAYZ, &
                           ONE,IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN)
                   else
                      call relax1(nf,IPHIW,irhs(ii)%A, &
                           MAY,MAYX,MAYY,MAYZ, &
                           EPSW,IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN)
                   endif
                enddo
                if(QENVV) then
                   call resid1(nf, &
                        ires(ii)%A,IPHIP,irhs(ii)%A, &
                        MAY,MAYX,MAYY,MAYZ, &
                        ONE,IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
                        anorm)
                else
                   call resid1(nf, &
                        ires(ii)%A,IPHIW,irhs(ii)%A, &
                        MAY,MAYX,MAYY,MAYZ, &
                        EPSW,IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
                        anorm)
                endif
             else
                do ipre=1,NPRE          !     pre-smoothing
                   call relax2(nf, &
                        iphi(ii)%A,irhs(ii)%A, &
                        ikappa(ii)%A,iepsx(ii)%A, &
                        iepsy(ii)%A,iepsz(ii)%A)
                enddo
                call resid2(nf, &
                     ires(ii)%A,iphi(ii)%A,irhs(ii)%A, &
                     ikappa(ii)%A,iepsx(ii)%A, &
                     iepsy(ii)%A,iepsz(ii)%A,averes)
                if(i.eq.ii) anorm=averes
             endif

             nf=nf/2+1
             !     the restriction of the residual
             !     is the next r.h.s
             call rstrct(nf,irhs(ii-1)%A,ires(ii)%A)

             !     Zero for initial guess
             !     in next relaxation
             call fill0(nf,iphi(ii-1)%A)

          enddo
          ! (2) bottom of V:
          !     solve on coarsest grid
          call slvsml(iphi(1)%A,irhs(1)%A,ikappa(1)%A, &
               iepsx(1)%A,iepsy(1)%A,iepsz(1)%A)

          nf=3

          do ii=2,i                     ! (3) upward stroke of V
             nf=2*nf-1

             if(ii.eq.ngridpb) then
                if(QENVV) then
                   call pbadd_int(nf,IPHIP,iphi(ii-1)%A,ires(ii)%A)
                else
                   call pbadd_int(nf,IPHIW,iphi(ii-1)%A,ires(ii)%A)
                endif
                do ipre=1,NPOST          !     post-smoothing
                   if(QENVV) then
                      call relax1(nf,IPHIP,irhs(ii)%A, &
                           MAY,MAYX,MAYY,MAYZ, &
                           ONE,IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN)
                   else
                      call relax1(nf,IPHIW,irhs(ii)%A, &
                           MAY,MAYX,MAYY,MAYZ, &
                           EPSW,IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN)
                   endif
                enddo
             else
                call pbadd_int(nf,iphi(ii)%A,iphi(ii-1)%A,ires(ii)%A)
                do ipre=1,NPOST          !     post-smoothing
                   call relax2(nf, &
                        iphi(ii)%A,irhs(ii)%A, &
                        ikappa(ii)%A,iepsx(ii)%A, &
                        iepsy(ii)%A,iepsz(ii)%A)
                enddo
             endif

          enddo

          if(QPRIN .and. &
               (anorm.le.deps .or. prnlev.gt.10)) then !! deps=0.2e-05
             write(outu,'(3x,2(a,i4),a,e12.5)') &
                  'NCEL',2**i+1,', number of CYCLEs:',icyc, &
                  '  and  RESID:',anorm
          endif
          if(anorm.le.deps) cycle iloop2

       enddo

       if(prnlev.ge.2) write(outu,'(3x,a,i4)') &
            'Calculation not converged, NCYCLE is too small for NCEL', &
            2**i+1
       if(prnlev.ge.2) write(outu,'(3x,2(a,e15.5))') &
            '--> RESID:',anorm,'  DEPS:',deps
    enddo iloop2
    !
    iloop3: DO I=1,NGRIDpb
       NN=2**I+1
       NN3=NN**3
       ! free  storage for ires,irhs,iphi,icden,iepsx,iepsy,iepsz,ikappa
       call chmdealloc('pbeq.src','PBEQ2','IRES(I)',NN3,cr4p=IRES(I)%A)
       call chmdealloc('pbeq.src','PBEQ2','IRHS(I)',NN3,cr4p=IRHS(I)%A)
       IF(I.EQ.NGRIDpb) cycle iloop3
       call chmdealloc('pbeq.src','PBEQ2','IPHI(I)',NN3,cr4p=IPHI(I)%A)
       call chmdealloc('pbeq.src','PBEQ2','ICDEN(I)',NN3,cr4p=ICDEN(I)%A)
       call chmdealloc('pbeq.src','PBEQ2','IEPSX(I)',NN3,cr4p=IEPSX(I)%A)
       call chmdealloc('pbeq.src','PBEQ2','IEPSY(I)',NN3,cr4p=IEPSY(I)%A)
       call chmdealloc('pbeq.src','PBEQ2','IEPSZ(I)',NN3,cr4p=IEPSZ(I)%A)
       call chmdealloc('pbeq.src','PBEQ2','IKAPPA(I)',NN3,cr4p=IKAPPA(I)%A)
    ENDDO iloop3
    !
    RETURN
  END SUBROUTINE PBEQ2
  !
  SUBROUTINE adjxyz(nmax,ixmax,iymax,izmax,ixmin,iymin,izmin, &
       epsw,epsp,kappa2)
    !------------------------------------------------------------------------
    !     Adjust ixmax,iymax,izmax,ixmin,iymin,and izmin
    !
    use chm_kinds
    implicit none
    integer  nmax,ixmax,iymax,izmax,ixmin,iymin,izmin
    real(chm_real)   epsw,epsp,kappa2
    !
    integer  n
    !
    n=2**nmax+1
    if(mod(izmin,2).ne.0) izmin=izmin-1
    if(mod(izmax,2).eq.0) izmax=izmax+1
    if(mod(izmax,2).ne.0) izmax=izmax+2

    ! for focussing
    if(ixmin.le.2) ixmin=2
    if(ixmax.ge.(n-1)) ixmax=n-1
    if(iymin.le.2) iymin=2
    if(iymax.ge.(n-1)) iymax=n-1
    if(izmin.le.2) izmin=2
    if(izmax.ge.(n-1)) izmax=n-1

    ! some restricitons for red-black gauss-seidel relaxation
    if(mod(ixmin,2).ne.0) then
       if(mod(ixmax,2).eq.0 .and. ixmax.ne.(n-1)) ixmax=ixmax+1
       if(mod(ixmax,2).eq.0 .and. ixmax.eq.(n-1)) ixmax=ixmax-1
    else
       if(ixmin.eq.2) then
          if(mod(ixmax,2).ne.0 .and. ixmax.ne.(n-1)) ixmax=ixmax+1
          if(mod(ixmax,2).ne.0 .and. ixmax.eq.(n-1)) ixmax=ixmax-1
       else
          ixmin=ixmin-1
          if(mod(ixmax,2).eq.0 .and. ixmax.ne.(n-1)) ixmax=ixmax+1
          if(mod(ixmax,2).eq.0 .and. ixmax.eq.(n-1)) ixmax=ixmax-1
       endif
    endif
    if(mod(iymin,2).ne.0) then
       if(mod(iymax,2).eq.0 .and. iymax.ne.(n-1)) iymax=iymax+1
       if(mod(iymax,2).eq.0 .and. iymax.eq.(n-1)) iymax=iymax-1
    else
       if(iymin.eq.2) then
          if(mod(iymax,2).ne.0 .and. iymax.ne.(n-1)) iymax=iymax+1
          if(mod(iymax,2).ne.0 .and. iymax.eq.(n-1)) iymax=iymax-1
       else
          iymin=iymin-1
          if(mod(iymax,2).eq.0 .and. iymax.ne.(n-1)) iymax=iymax+1
          if(mod(iymax,2).eq.0 .and. iymax.eq.(n-1)) iymax=iymax-1
       endif
    endif

    ! for the vacuum environment
    if(epsw.eq.epsp .and. kappa2.eq.0.0) then
       ixmin=n
       ixmax=n-1
    endif
    !
    RETURN
  END  SUBROUTINE adjxyz
  !
  SUBROUTINE pbadd_int(nf,uf,uc,res)
    !------------------------------------------------------------------------
    !     Does coarse-to-fine interpolation and adds result to UF.
    !
    use chm_kinds
    implicit none
    integer  nf
    real(chm_real4)   res(*),uc(*),uf(*)
    ! local
    integer  nf3,ip
    !
    call interpol(nf,res,uc)

    nf3=nf*nf*nf
    do ip=1,nf3
       uf(ip)=uf(ip)+res(ip)
    enddo
    !
    RETURN
  END  SUBROUTINE pbadd_int
  !
  SUBROUTINE rstrct(nc,uc,uf)
    !------------------------------------------------------------------------
    !     Half-weighting restriction  for RESID
    !     NC : the coarse-grid dimension
    !     UC : the coarse-grid solution  (NEXT R.H.S)
    !     UF : the fine-grid solution    (RESID)
    !
    use chm_kinds
    use number
    implicit none
    integer nc
    real(chm_real4)  uc(*),uf(*)
    ! local
    integer  nc2,nf,nf2,ip0,ip0x,ip0y
    integer  ig,jg,kg,ipncx,ipncy,ipnc

    !                        (in Z)
    !                              ip6 (1/12)
    !                               | ip3
    !                               | /
    !                               |/
    !                    ip1 ----- ip0 ----- ip2   (in X)
    !                              /|(1/2)
    !                             / |
    !                           ip4 |
    !                    (in Y)    ip5
    !

    nf=2*nc-1
    nc2=nc*nc
    nf2=nf*nf

    ! interior points
    do ig=2,nc-1
       ipncx=(ig-1)*nc2
       ip0x=2*(ig-1)*nf2
       do jg=2,nc-1
          ipncy=(jg-1)*nc+ipncx
          ip0y=2*(jg-1)*nf+ip0x
          do kg=2,nc-1
             ipnc= ipncy+kg
             ip0 = ip0y +2*kg-1
             uc(ipnc)=0.5*uf(ip0)+ &
                  1./12.*(uf(ip0-nf2)+uf(ip0+nf2)+uf(ip0-nf)+ &
                  uf(ip0+nf)+uf(ip0-1)+uf(ip0+1))
          enddo
       enddo
    enddo

    ! boundary points (always ZERO)
    !
    RETURN
  END SUBROUTINE rstrct
  !
  SUBROUTINE resid1(n,res,phi,rhs,fkapa,epsx,epsy,epsz,epsw, &
       ixmax,iymax,izmax,ixmin,iymin,izmin,averes)
    !------------------------------------------------------------------------
    !     Return value is minus the residual
    !
    use chm_kinds
    use number
    implicit none
    integer  n,ixmax,iymax,izmax,ixmin,iymin,izmin
    real(chm_real4)   res(*),phi(*),rhs(*), &
         epsx(*),epsy(*),epsz(*),fkapa(*)
    real(chm_real)   epsw
    ! local
    integer  n3,n2,ig,jg,kg,ip0,ip0x,ip0y
    real(chm_real)   bot,top
    real(chm_real)   sumres,averes
    !
    n2=n*n
    n3=n2*n

    sumres=0.0

    ! boundary points
    ip0x=(n-1)*n2
    do jg=1,n
       ip0y=(jg-1)*n
       do kg=2,n-1
          ip0=ip0y+kg         ! yz plane at x=xmin
          res(ip0)=0.0
          ip0=ip0x+ip0        ! yz plane at x=xmax
          res(ip0)=0.0
       enddo
    enddo
    do ig=2,n-1
       ip0x=(ig-1)*n2
       do kg=2,n-1
          ip0=ip0x+kg         ! xz plane at y=ymin
          res(ip0)=0.0
          ip0=ip0+n2-n        ! xz plane at y=ymax
          res(ip0)=0.0
       enddo
    enddo
    do ig=1,n
       ip0x=(ig-1)*n2
       do jg=1,n
          ip0=ip0x+(jg-1)*n+1 ! xy plane at z=zmax
          res(ip0)=0.0
          ip0=ip0+n-1         ! xy plane at z=zmin
          res(ip0)=0.0
       enddo
    enddo

    ! interior points
    do ig=2,ixmin-1
       ip0x=(ig-1)*n2
       do jg=2,n-1
          ip0y=(jg-1)*n+ip0x
          do kg=2,n-1
             ip0 = ip0y + kg
             bot=six*epsw+fkapa(ip0)
             top=(phi(ip0-n2)+phi(ip0+n2)+phi(ip0-n)+phi(ip0+n)+ &
                  phi(ip0-1)+phi(ip0+1))*epsw+rhs(ip0)
             res(ip0)=(top-phi(ip0)*bot)
             sumres=sumres+abs(res(ip0))
          enddo
       enddo
    enddo

    do ig=ixmin,ixmax
       ip0x=(ig-1)*n2
       do jg=2,iymin-1
          ip0y=(jg-1)*n+ip0x
          do kg=2,n-1
             ip0 = ip0y + kg
             bot=six*epsw+fkapa(ip0)
             top=(phi(ip0-n2)+phi(ip0+n2)+phi(ip0-n)+phi(ip0+n)+ &
                  phi(ip0-1)+phi(ip0+1))*epsw
             res(ip0)=(top-phi(ip0)*bot)
             sumres=sumres+abs(res(ip0))
          enddo
       enddo

       do jg=iymin,iymax
          ip0y=(jg-1)*n+ip0x
          do kg=2,izmin-1
             ip0 = ip0y + kg
             bot=six*epsw+fkapa(ip0)
             top=(phi(ip0-n2)+phi(ip0+n2)+phi(ip0-n)+phi(ip0+n)+ &
                  phi(ip0-1)+phi(ip0+1))*epsw
             res(ip0)=(top-phi(ip0)*bot)
             sumres=sumres+abs(res(ip0))
          enddo

          do kg=izmin,izmax
             ip0 = ip0y + kg
             bot=epsx(ip0)   +epsy(ip0)  +epsz(ip0)+ &
                  epsx(ip0-n2)+epsy(ip0-n)+epsz(ip0-1)+fkapa(ip0)
             top=phi(ip0-n2)*epsx(ip0-n2)+phi(ip0+n2)*epsx(ip0)+ &
                  phi(ip0-n) *epsy(ip0-n) +phi(ip0+n) *epsy(ip0)+ &
                  phi(ip0-1) *epsz(ip0-1) +phi(ip0+1) *epsz(ip0)+ &
                  rhs(ip0)
             res(ip0)=(top-phi(ip0)*bot)
             sumres=sumres+abs(res(ip0))
          enddo

          do kg=izmax+1,n-1
             ip0 = ip0y + kg
             bot=six*epsw+fkapa(ip0)
             top=(phi(ip0-n2)+phi(ip0+n2)+phi(ip0-n)+phi(ip0+n)+ &
                  phi(ip0-1)+phi(ip0+1))*epsw
             res(ip0)=(top-phi(ip0)*bot)
             sumres=sumres+abs(res(ip0))
          enddo
       enddo

       do jg=iymax+1,n-1
          ip0y=(jg-1)*n+ip0x
          do kg=2,n-1
             ip0 = ip0y + kg
             bot=six*epsw+fkapa(ip0)
             top=(phi(ip0-n2)+phi(ip0+n2)+phi(ip0-n)+phi(ip0+n)+ &
                  phi(ip0-1)+phi(ip0+1))*epsw
             res(ip0)=(top-phi(ip0)*bot)
             sumres=sumres+abs(res(ip0))
          enddo
       enddo
    enddo

    do ig=ixmax+1,n-1
       ip0x=(ig-1)*n2
       do jg=2,n-1
          ip0y=(jg-1)*n+ip0x
          do kg=2,n-1
             ip0 = ip0y + kg
             bot=six*epsw+fkapa(ip0)
             top=(phi(ip0-n2)+phi(ip0+n2)+phi(ip0-n)+phi(ip0+n)+ &
                  phi(ip0-1)+phi(ip0+1))*epsw
             phi(ip0)=top/bot
          enddo
       enddo
    enddo

    averes=sumres/n3
    !
    RETURN
  END SUBROUTINE resid1
  !
  SUBROUTINE resid2(n,res,phi,rhs,fkapa,epsx,epsy,epsz,averes)
    !------------------------------------------------------------------------
    !     Return value is minus the residual
    !
    use chm_kinds
    use number
    implicit none
    integer  n
    real(chm_real4)   res(*),phi(*),rhs(*), &
         epsx(*),epsy(*),epsz(*),fkapa(*)
    ! local
    integer  n3,n2,ig,jg,kg,ip0,ip0x,ip0y
    real(chm_real)   bot,top
    real(chm_real)   sumres,averes
    !
    n2=n*n
    n3=n2*n

    sumres=0.0

    ! boundary points
    ip0x=(n-1)*n2
    do jg=1,n
       ip0y=(jg-1)*n
       do kg=2,n-1
          ip0=ip0y+kg         ! yz plane at x=xmin
          res(ip0)=0.0
          ip0=ip0x+ip0        ! yz plane at x=xmax
          res(ip0)=0.0
       enddo
    enddo
    do ig=2,n-1
       ip0x=(ig-1)*n2
       do kg=2,n-1
          ip0=ip0x+kg         ! xz plane at y=ymin
          res(ip0)=0.0
          ip0=ip0+n2-n        ! xz plane at y=ymax
          res(ip0)=0.0
       enddo
    enddo
    do ig=1,n
       ip0x=(ig-1)*n2
       do jg=1,n
          ip0=ip0x+(jg-1)*n+1 ! xy plane at z=zmax
          res(ip0)=0.0
          ip0=ip0+n-1         ! xy plane at z=zmin
          res(ip0)=0.0
       enddo
    enddo

    ! interior points
    do ig=2,n-1
       ip0x=(ig-1)*n2
       do jg=2,n-1
          ip0y=(jg-1)*n+ip0x
          do kg=2,n-1
             ip0 = ip0y + kg
             bot=epsx(ip0)   +epsy(ip0)  +epsz(ip0)+ &
                  epsx(ip0-n2)+epsy(ip0-n)+epsz(ip0-1)+fkapa(ip0)
             top=phi(ip0-n2)*epsx(ip0-n2)+phi(ip0+n2)*epsx(ip0)+ &
                  phi(ip0-n) *epsy(ip0-n) +phi(ip0+n) *epsy(ip0)+ &
                  phi(ip0-1) *epsz(ip0-1) +phi(ip0+1) *epsz(ip0)+ &
                  rhs(ip0)
             !wi               res(ip0)=-(top-phi(ip0)*bot)
             res(ip0)=(top-phi(ip0)*bot)
             sumres=sumres+abs(res(ip0))
          enddo
       enddo
    enddo

    averes=sumres/n3
    !
    RETURN
  END SUBROUTINE resid2
  !
  SUBROUTINE relax1(n,phi,rhs,fkapa,epsx,epsy,epsz,epsw, &
       ixmax,iymax,izmax,ixmin,iymin,izmin)
    !------------------------------------------------------------------------
    !     Red-black Gauss-Seidel relaxation
    !     The current valus of the solution U is updated
    !     using the right-hand side function RHS (<- CDEN).
    !
    use chm_kinds
    use number
    implicit none
    integer  n,ixmax,iymax,izmax,ixmin,iymin,izmin
    real(chm_real4)   phi(*),rhs(*),epsx(*),epsy(*),epsz(*),fkapa(*)
    real(chm_real)   epsw
    ! local
    integer  n2,ip0,ip0x,ip0y,ig,jg,kg
    integer  ipass,ksw
    real(chm_real)   bot,top
    !
    !                        (in Z)
    !                              ip6
    !                               | ip3
    !                               | /
    !                               |/
    !                    ip1 ----- ip0 ----- ip2   (in X)
    !                              /|
    !                             / |
    !                           ip4 |
    !                    (in Y)    ip5
    !
    ! NOTE: RHS(CDEN) and FKAPA already contained 4*pi/h and h*h, respectively.
    !

    n2=n*n

    ksw=1
    do ipass=1,2

       do ig=2,ixmin-1
          ip0x=(ig-1)*n2
          do jg=2,n-1
             ip0y=(jg-1)*n+ip0x
             do kg=ksw+1,n-ksw,2
                ip0 = ip0y + kg
                bot=six*epsw+fkapa(ip0)
                top=(phi(ip0-n2)+phi(ip0+n2)+phi(ip0-n)+phi(ip0+n)+ &
                     phi(ip0-1)+phi(ip0+1))*epsw+rhs(ip0)
                phi(ip0)=top/bot
             enddo
             ksw=3-ksw
          enddo
       enddo

       do ig=ixmin,ixmax
          ip0x=(ig-1)*n2
          do jg=2,iymin-1
             ip0y=(jg-1)*n+ip0x
             do kg=ksw+1,n-ksw,2
                ip0 = ip0y + kg
                bot=six*epsw+fkapa(ip0)
                top=(phi(ip0-n2)+phi(ip0+n2)+phi(ip0-n)+phi(ip0+n)+ &
                     phi(ip0-1)+phi(ip0+1))*epsw
                phi(ip0)=top/bot
             enddo
             ksw=3-ksw
          enddo

          do jg=iymin,iymax
             ip0y=(jg-1)*n+ip0x
             do kg=ksw+1,izmin-1,2
                ip0 = ip0y + kg
                bot=six*epsw+fkapa(ip0)
                top=(phi(ip0-n2)+phi(ip0+n2)+phi(ip0-n)+phi(ip0+n)+ &
                     phi(ip0-1)+phi(ip0+1))*epsw
                phi(ip0)=top/bot
             enddo

             do kg=izmin,izmax,2
                ip0 = ip0y + kg
                bot=epsx(ip0)   +epsy(ip0)  +epsz(ip0)+ &
                     epsx(ip0-n2)+epsy(ip0-n)+epsz(ip0-1)+fkapa(ip0)
                top=phi(ip0-n2)*epsx(ip0-n2)+phi(ip0+n2)*epsx(ip0)+ &
                     phi(ip0-n) *epsy(ip0-n) +phi(ip0+n) *epsy(ip0)+ &
                     phi(ip0-1) *epsz(ip0-1) +phi(ip0+1) *epsz(ip0)+ &
                     rhs(ip0)
                phi(ip0)=top/bot
             enddo

             do kg=izmax+1,n-ksw,2
                ip0 = ip0y + kg
                bot=six*epsw+fkapa(ip0)
                top=(phi(ip0-n2)+phi(ip0+n2)+phi(ip0-n)+phi(ip0+n)+ &
                     phi(ip0-1)+phi(ip0+1))*epsw
                phi(ip0)=top/bot
             enddo
             izmin=izmin-ksw
             izmax=izmax+ksw
             ksw=3-ksw
             izmin=izmin+ksw
             izmax=izmax-ksw
          enddo

          do jg=iymax+1,n-1
             ip0y=(jg-1)*n+ip0x
             do kg=ksw+1,n-ksw,2
                ip0 = ip0y + kg
                bot=six*epsw+fkapa(ip0)
                top=(phi(ip0-n2)+phi(ip0+n2)+phi(ip0-n)+phi(ip0+n)+ &
                     phi(ip0-1)+phi(ip0+1))*epsw
                phi(ip0)=top/bot
             enddo
             ksw=3-ksw
          enddo
       enddo

       do ig=ixmax+1,n-1
          ip0x=(ig-1)*n2
          do jg=2,n-1
             ip0y=(jg-1)*n+ip0x
             do kg=ksw+1,n-ksw,2
                ip0 = ip0y + kg
                bot=six*epsw+fkapa(ip0)
                top=(phi(ip0-n2)+phi(ip0+n2)+phi(ip0-n)+phi(ip0+n)+ &
                     phi(ip0-1)+phi(ip0+1))*epsw
                phi(ip0)=top/bot
             enddo
             ksw=3-ksw
          enddo
       enddo

    enddo
    !
    RETURN
  END  SUBROUTINE relax1
  !
  SUBROUTINE relax2(n,phi,rhs,fkapa,epsx,epsy,epsz)
    !------------------------------------------------------------------------
    !     Red-black Gauss-Seidel relaxation
    !     The current valus of the solution U is updated
    !     using the right-hand side function RHS (<- CDEN).
    !
    use chm_kinds
    use number
    implicit none
    integer  n
    real(chm_real4)   phi(*),rhs(*),epsx(*),epsy(*),epsz(*),fkapa(*)
    ! local
    integer  n2,ip0,ip0x,ip0y,ig,jg,kg
    integer  ipass,jsw,ksw
    real(chm_real)   bot,top
    !
    !                        (in Z)
    !                              ip6
    !                               | ip3
    !                               | /
    !                               |/
    !                    ip1 ----- ip0 ----- ip2   (in X)
    !                              /|
    !                             / |
    !                           ip4 |
    !                    (in Y)    ip5
    !
    ! NOTE: RHS(CDEN) and FKAPA already contained 4*pi/h and h*h, respectively.
    !

    n2=n*n

    jsw=1
    do ipass=1,2

       do ig=2,n-1
          ip0x=(ig-1)*n2
          ksw=jsw
          do jg=2,n-1
             ip0y=(jg-1)*n+ip0x
             do kg=ksw+1,n-ksw,2
                ip0 = ip0y + kg
                bot=epsx(ip0)   +epsy(ip0)  +epsz(ip0)+ &
                     epsx(ip0-n2)+epsy(ip0-n)+epsz(ip0-1)+fkapa(ip0)
                top=phi(ip0-n2)*epsx(ip0-n2)+phi(ip0+n2)*epsx(ip0)+ &
                     phi(ip0-n) *epsy(ip0-n) +phi(ip0+n) *epsy(ip0)+ &
                     phi(ip0-1) *epsz(ip0-1) +phi(ip0+1) *epsz(ip0)+ &
                     rhs(ip0)
                phi(ip0)=top/bot
             enddo
             ksw=3-ksw
          enddo
          jsw=3-jsw
       enddo

    enddo
    !
    RETURN
  END SUBROUTINE relax2
  !
  SUBROUTINE cprho(n,ain,aout)
    !------------------------------------------------------------------------
    !
    use chm_kinds
    implicit none
    integer  n
    real(chm_real4)   ain(*),aout(*)
    ! local
    integer  i,n3
    !
    n3=n*n*n
    do i=1,n3
       aout(i)=ain(i)
    enddo
    !
    RETURN
  END SUBROUTINE cprho
  !
  SUBROUTINE interpol(nf,uf,uc)
    !------------------------------------------------------------------------
    !     Coarse-to-fine prolongation by trilinear interpolation.
    !     NF : the fine-grid dimension
    !     UF : the fine-grid solution
    !     UC : the coarse-grid solution
    !
    use chm_kinds
    use number
    implicit none
    integer  nf
    real(chm_real4)   uc(*),uf(*)
    ! local
    integer  nc,nc2,nf2,ig,jg,kg,ipncx,ipnfx,ipncy,ipnfy,ipnc,ipnf
    integer  ip0x,ip0y,ip0
    !

    nc=nf/2+1
    nc2=nc*nc
    nf2=nf*nf

    ! Injection from coarse to fine       Coarse Grid           Finest Grid
    !                                          ig                2*ig-1       in X
    do ig=2,nc-1
       ipncx=  (ig-1)*nc2
       ipnfx=2*(ig-1)*nf2
       do jg=2,nc-1
          ipncy=  (jg-1)*nc+ipncx
          ipnfy=2*(jg-1)*nf+ipnfx
          do kg=2,nc-1
             ipnc=ipncy+kg
             ipnf=ipnfy+2*kg-1
             uf(ipnf)=uc(ipnc)
          enddo
       enddo
    enddo

    ! only even-numbered layers along Z
    ! using 2 points in the upper and lower layers
    do ig=3,nf-2,2
       ip0x=(ig-1)*nf2
       do jg=3,nf-2,2
          ip0y=(jg-1)*nf+ip0x
          do kg=2,nf-1,2
             ip0=ip0y+kg
             uf(ip0)=0.5*(uf(ip0-1)+uf(ip0+1))
          enddo
       enddo
    enddo

    ! elements in odd-numbered rows, interplating horizontally
    do ig=2,nf-1,2
       ip0x=(ig-1)*nf2
       do jg=3,nf-2,2
          ip0y=(jg-1)*nf+ip0x
          do kg=2,nf-1
             ip0=ip0y+kg
             uf(ip0)=0.5*(uf(ip0-nf2)+uf(ip0+nf2))
          enddo
       enddo
    enddo

    ! elements in even-numbered rows, interplating vertically
    do ig=2,nf-1
       ip0x=(ig-1)*nf2
       do jg=2,nf-1,2
          ip0y=(jg-1)*nf+ip0x
          do kg=2,nf-1
             ip0=ip0y+kg
             uf(ip0)=0.5*(uf(ip0-nf)+uf(ip0+nf))
          enddo
       enddo
    enddo
    !
    RETURN
  END SUBROUTINE interpol
  !
  SUBROUTINE slvsml(PHI,CDEN,FKAPA,EPSX,EPSY,EPSZ)
    !------------------------------------------------------------------------
    !     Solution of the model problem on the coarsest grid
    !     NOTE: The result, phi(14), is different from the result of
    !           SOR (Successive OverRelaxation) method in subroutine PBEQ1
    !           because the OMEGA is not 1.
    !
    use chm_kinds
    use number
    implicit none
    real(chm_real4)   phi(*),cden(*),epsx(*),epsy(*),epsz(*),fkapa(*)
    ! local
    real(chm_real)   top,bot
    integer  i
    !
    !wi      call fill0(3,phi)

    bot=epsx(14)+epsy(14)+epsz(14)+epsx(5)+epsy(11)+epsz(13)+ &
         fkapa(14)
    top=phi(5) *epsx(5) +phi(23)*epsx(14)+ &
         phi(11)*epsy(11)+phi(17)*epsy(14)+ &
         phi(13)*epsz(13)+phi(15)*epsz(14)+cden(14)
    phi(14)=top/bot
    !      ENDIF
    !
    RETURN
  END SUBROUTINE slvsml
  !
  SUBROUTINE fill0(n,u)
    !------------------------------------------------------------------------
    !     Fill u(n**3) with zero
    !
    use chm_kinds
    implicit none
    real(chm_real4)   u(*)
    integer  n
    ! local
    integer  ip,n3
    !
    n3=n*n*n
    do ip=1,n3
       u(ip)=0.0
    enddo
    !
    RETURN
  END SUBROUTINE fill0
  !
  SUBROUTINE injbp(NC,UF,UC)
    !------------------------------------------------------------------------
    !     INJect Boundary Potential from the finest grid to coarse grids
    !
    use chm_kinds
    use number
    implicit none
    integer NC
    real(chm_real4)  UF(*),UC(*)
    ! local
    integer ig,jg,kg,ipnc,ipnf,nc2,nf,nf2

    !
    !     Coarse Grid           Finest Grid
    !         ig                2*ig-1       in X

    nf=2*nc-1
    nc2=nc*nc
    nf2=nf*nf

    do jg=1,nc
       do kg=1,nc
          ipnc=   (jg-1)*nc + kg                       ! yz plane at x=xmin
          ipnf= 2*(jg-1)*nf + 2*kg-1
          UC(ipnc)=UF(ipnf)
          ipnc= (nc-1)*nc2 +   (jg-1)*nc + kg          ! yz plane at x=xmax
          ipnf= (nf-1)*nf2 + 2*(jg-1)*nf + 2*kg-1
          UC(ipnc)=UF(ipnf)
       enddo
    enddo
    do ig=2,nc-1
       do kg=1,nc
          ipnc=   (ig-1)*nc2 + kg                      ! xz plane at y=ymin
          ipnf= 2*(ig-1)*nf2 + 2*kg-1
          UC(ipnc)=UF(ipnf)
          ipnc=   (ig-1)*nc2 + nc2 - nc + kg           ! xz plane at y=ymax
          ipnf= 2*(ig-1)*nf2 + nf2 - nf + 2*kg-1
          UC(ipnc)=UF(ipnf)
       enddo
    enddo
    do ig=2,nc-1
       do jg=2,nc-1
          ipnc=   (ig-1)*nc2 +   (jg-1)*nc + 1         ! xy plane at z=zmax
          ipnf= 2*(ig-1)*nf2 + 2*(jg-1)*nf + 1
          UC(ipnc)=UF(ipnf)
          ipnc=   (ig-1)*nc2 +   (jg-1)*nc + nc        ! xy plane at z=zmin
          ipnf= 2*(ig-1)*nf2 + 2*(jg-1)*nf + nf
          UC(ipnc)=UF(ipnf)
       enddo
    enddo
    !
    RETURN
  END SUBROUTINE injbp
  !
  SUBROUTINE PBEQ3(MAXITS,OMEGA,TOL,TEMP,KAPPA2, &
       NCLX,NCLY,NCLZ,DCEL,VMEMB,TMEMB,ZMEMB,TRANZ,ZBCEN, &
       PHI,EPSX,EPSY,EPSZ,CDEN,FKAPA, &
       QPBC,QNPBC,QUNDER)
    !-----------------------------------------------------------------------
    !     (Non-linear) Poisson-Boltzmann iterator for discretized cubic lattice.
    !     It is assumed that all arrays have been prepared elsewhere.
    !
    !                        (in Z)
    !                        ip0+1=ip6
    !                               | ip3
    !                               | /
    !                               |/
    !                    ip1 ----- ip0 ----- ip2   (in X)
    !                              /|
    !                             / |
    !                           ip4 |
    !                    (in Y)    ip5=ip0-1
    !
    ! NOTE: CDEN and FKAPA already contained 4*pi/h and h*h, respectively.
    !
    use chm_kinds
    use stream
    use consta
    use number
    implicit none
    REAL(CHM_REAL4)  PHI(*),EPSX(*),EPSY(*),EPSZ(*),CDEN(*),FKAPA(*)
    real(chm_real)  OMEGA,TOL,TEMP,TRANZ,ZBCEN,KAPPA2,DCEL
    real(chm_real)  VMEMB,TMEMB,ZMEMB
    INTEGER MAXITS,NCLX,NCLY,NCLZ
    LOGICAL QPBC,QNPBC,QUNDER

    ! Local variables
    real(chm_real)  FACTOR1,ZMEMB2,MKAPPA2,DCEL2,RJAC,RESID,ANORM
    real(chm_real)  MIONCD,FPROTCD,PHIF,RATIO,BOT,TOP,ZC
    REAL(CHM_REAL4)  PHI0,EPS1,EPS2,EPS3,EPS4,EPS5,EPS6
    INTEGER NCYZ,NC3,N
    INTEGER KSW
    INTEGER IG,JG,KG,IP0,IP0X,IP1X,IP2X,IP0Y,IP3Y,IP4Y,IP5Z,IP6Z
    !
    NCyz=NCLy*NCLz
    NC3=NCLx*NCyz
    DCEL2=DCEL*DCEL
    FACTOR1=ccelec/(kboltz*Temp) ! 1/(kcal/(mol*e))->1/(e/A)
    zmemb2=tranz-zbcen+0.5*tmemb+zmemb+rsmall
    MKAPPA2=KAPPA2*DCEL2
    !
    IF(NCLx.eq.3 .and. NCLy.eq.3 .and. NCLz.eq.3) THEN
       RJAC=(COS(PI/NCLx)+COS(PI/NCLy)+COS(PI/NCLz))/THREE
       IF(PRNLEV.GT.10) WRITE(OUTU,'(3x,A,E15.5)') 'RJAC',RJAC
    ELSE
       RJAC=(COS(PI/(NCLx-1))+COS(PI/(NCLy-1))+COS(PI/(NCLz-1)))/THREE
       IF(PRNLEV.GT.10) WRITE(OUTU,'(3x,A,E15.5)') 'RJAC',RJAC
    ENDIF

    KSW=1

    ! OUTPUT format
    IF(PRNLEV.GE.2) WRITE(OUTU,'(3x,a)') &
         'Non-linear PBEQ ITERATIONs for 1:1 charge-paired salt '

    IF(QPBC)THEN
       !     ------------
       ! periodic in XYZ

       DO N=1,MAXITS
          ANORM=ZERO

          ! x component
          DO IG=1,NCLx
             IP0X=(IG-1)*NCyz
             IP1X=-NCyz
             IF(IG.EQ.1) IP1X=NC3-NCyz    ! (NCLx-1)*NCyz
             IP2X=NCyz
             IF(IG.EQ.NCLx) IP2X=-(NC3-NCyz)
             ! y component
             DO JG=1,NCLy
                IP0Y=(JG-1)*NCLz+IP0X
                IP3Y=-NCLz
                IF(JG.EQ.1) IP3Y=NCyz-NCLz ! (NCLy-1)*NCLz
                IP4Y=NCLz
                IF(JG.EQ.NCLy) IP4Y=-(NCyz-NCLz)
                ! z component
                !wi               DO KG=1,NCLz
                DO KG=KSW,NCLZ-KSW+1,2
                   IP0 = IP0Y + KG
                   IP5Z=-1
                   IF(KG.EQ.1) IP5Z=NCLz-1
                   IP6Z=1
                   IF(KG.EQ.NCLz) IP6Z=-(NCLz-1)

                   PHI0=PHI(IP0)
                   EPS1=EPSX(IP0+IP1X)
                   EPS2=EPSX(IP0)
                   EPS3=EPSY(IP0+IP3Y)
                   EPS4=EPSY(IP0)
                   EPS5=EPSZ(IP0+IP5Z)
                   EPS6=EPSZ(IP0)

                   ! convert FKAPA array to ( kappa2 * nonlinearity ratio)
                   ! add the background charge (kappa2 * nonlinearity ratio * Vmemb) to CDEN
                   MIONCD=0.0
                   FPROTCD=CDEN(IP0)
                   IF(FKAPA(IP0).ne.0.0) THEN
                      PHIF=PHI0*FACTOR1
                      IF(VMEMB.NE.0.0)THEN
                         ZC=(KG-1)*DCEL
                         IF(ZC.GT.ZMEMB2) PHIF=(PHI0-VMEMB)*FACTOR1
                      ENDIF
                      RATIO=EXP(PHIF)
                      RATIO=(RATIO-1.0/RATIO)/2.0/PHIF
                      MIONCD=MKAPPA2*RATIO
                      IF(ZC.GT.ZMEMB2) FPROTCD=FPROTCD+MIONCD*VMEMB
                   ENDIF

                   BOT=EPS1+EPS2+EPS3+EPS4+EPS5+EPS6+MIONCD

                   TOP=PHI(IP0+IP1X)*EPS1+PHI(IP0+IP2X)*EPS2+ &
                        PHI(IP0+IP3Y)*EPS3+PHI(IP0+IP4Y)*EPS4+ &
                        PHI(IP0+IP5Z)*EPS5+PHI(IP0+IP6Z)*EPS6 + &
                        FPROTCD

                   !     phi = phi_old + omega*(phi_new - phi_old)
                   RESID=TOP-PHI0*BOT
                   ANORM=ANORM+ABS(RESID)
                   PHI(IP0)=PHI0+OMEGA*RESID/BOT
                ENDDO
                KSW=3-KSW
             ENDDO
          ENDDO

          ANORM=ANORM/NC3

          IF(PRNLEV.GT.6) THEN
             IP0=NCyz*int(NCLx/2)+NCLz*int(NCLy/2)+NCLz/2+1
             WRITE(OUTU,'(3x,A,2X,I5,4E11.4)') &
                  'N,ANORM,DEPS,OMEGA,PHI',N,ANORM,TOL,OMEGA,PHI(IP0)
          ENDIF

          IF(.not.Qunder) THEN
             IF(N.EQ.1)THEN
                OMEGA=1./(1.-.5*RJAC**2)
             ELSE
                OMEGA=1./(1.-.25*RJAC**2*OMEGA)
             ENDIF
          ENDIF

          IF((N.GT.1).AND.(ANORM.LT.TOL)) GOTO 19

       ENDDO

       !     -------------------------------------------
    ELSEIF(TMEMB.GT.ZERO .AND. .NOT. QNPBC)THEN
       !     -------------------------------------------
       ! periodic in XY, Z fixed edge-value (in membrane)

       DO N=1,MAXITS
          ANORM=ZERO

          !  x component
          DO IG=1,NCLx
             IP0X=(IG-1)*NCyz
             IP1X=-NCyz
             IF(IG.EQ.1) IP1X=NC3-NCyz    ! (NCLx-1)*NCyz
             IP2X=NCyz
             IF(IG.EQ.NCLx) IP2X=-(NC3-NCyz)
             !  y component
             DO JG=1,NCLy
                IP0Y=(JG-1)*NCLz+IP0X
                IP3Y=-NCLz
                IF(JG.EQ.1) IP3Y=NCyz-NCLz ! (NCLy-1)*NCLz
                IP4Y=NCLz
                IF(JG.EQ.NCLy) IP4Y=-(NCyz-NCLz)
                !  z component
                DO KG=KSW+1,NCLZ-KSW,2
                   IP0 = IP0Y + KG

                   PHI0=PHI(IP0)
                   EPS1=EPSX(IP0+IP1X)
                   EPS2=EPSX(IP0)
                   EPS3=EPSY(IP0+IP3Y)
                   EPS4=EPSY(IP0)
                   EPS5=EPSZ(IP0-1)
                   EPS6=EPSZ(IP0)

                   ! convert FKAPA array to ( kappa2 * nonlinearity ratio)
                   ! add the background charge (kappa2 * nonlinearity ratio * Vmemb) to CDEN
                   MIONCD=0.0
                   FPROTCD=CDEN(IP0)
                   IF(FKAPA(IP0).ne.0.0) THEN
                      PHIF=PHI0*FACTOR1
                      IF(VMEMB.NE.0.0)THEN
                         ZC=(KG-1)*DCEL
                         IF(ZC.GT.ZMEMB2) PHIF=(PHI0-VMEMB)*FACTOR1
                      ENDIF
                      RATIO=EXP(PHIF)
                      RATIO=(RATIO-1.0/RATIO)/2.0/PHIF
                      MIONCD=MKAPPA2*RATIO
                      IF(ZC.GT.ZMEMB2) FPROTCD=FPROTCD+MIONCD*VMEMB
                   ENDIF

                   BOT=EPS1+EPS2+EPS3+EPS4+EPS5+EPS6+MIONCD

                   TOP=PHI(IP0+IP1X)*EPS1+PHI(IP0+IP2X)*EPS2+ &
                        PHI(IP0+IP3Y)*EPS3+PHI(IP0+IP4Y)*EPS4+ &
                        PHI(IP0-1)*EPS5+PHI(IP0+1)*EPS6 + &
                        FPROTCD

                   !     phi = phi_old + omega*(phi_new - phi_old)
                   RESID=TOP-PHI0*BOT
                   ANORM=ANORM+ABS(RESID)
                   PHI(IP0)=PHI0+OMEGA*RESID/BOT
                ENDDO
                KSW=3-KSW
             ENDDO
          ENDDO

          ANORM=ANORM/NC3

          IF(PRNLEV.GT.6) THEN
             IP0=NCyz*int(NCLx/2)+NCLz*int(NCLy/2)+NCLz/2+1
             WRITE(OUTU,'(3x,A,2X,I5,4E11.4)') &
                  'N,ANORM,DEPS,OMEGA,PHI',N,ANORM,TOL,OMEGA,PHI(IP0)
          ENDIF

          IF(.not.QUNDER) THEN
             IF(N.EQ.1)THEN
                OMEGA=1./(1.-.5*RJAC**2)
             ELSE
                OMEGA=1./(1.-.25*RJAC**2*OMEGA)
             ENDIF
          ENDIF

          IF((N.GT.1).AND.(ANORM.LT.TOL)) GOTO 19

       ENDDO

       !     ----
    ELSE
       !     ----

       DO N=1,MAXITS
          ANORM=ZERO

          DO IG=2,NCLx-1
             IP0X=(IG-1)*NCyz
             DO JG=2,NCLy-1
                IP0Y=(JG-1)*NCLz+IP0X
                DO KG=KSW+1,NCLZ-KSW,2
                   IP0 = IP0Y + KG

                   PHI0=PHI(IP0)
                   EPS1=EPSX(IP0-NCyz)
                   EPS2=EPSX(IP0)
                   EPS3=EPSY(IP0-NCLz)
                   EPS4=EPSY(IP0)
                   EPS5=EPSZ(IP0-1)
                   EPS6=EPSZ(IP0)

                   ! convert FKAPA array to ( kappa2 * nonlinearity ratio)
                   ! add the background charge (kappa2 * nonlinearity ratio * Vmemb) to CDEN
                   MIONCD=0.0
                   FPROTCD=CDEN(IP0)
                   IF(FKAPA(IP0).ne.0.0) THEN
                      PHIF=PHI0*FACTOR1
                      IF(VMEMB.NE.0.0)THEN ! for focussing with vmemb
                         ZC=(KG-1)*DCEL
                         IF(ZC.GT.ZMEMB2) PHIF=(PHI0-VMEMB)*FACTOR1
                      ENDIF
                      RATIO=EXP(PHIF)
                      RATIO=(RATIO-1.0/RATIO)/2.0/PHIF
                      MIONCD=MKAPPA2*RATIO
                      IF(ZC.GT.ZMEMB2) FPROTCD=FPROTCD+MIONCD*VMEMB
                   ENDIF

                   BOT=EPS1+EPS2+EPS3+EPS4+EPS5+EPS6+MIONCD

                   TOP=PHI(IP0-NCyz)*EPS1+PHI(IP0+NCyz)*EPS2+ &
                        PHI(IP0-NCLz)*EPS3+PHI(IP0+NCLz)*EPS4+ &
                        PHI(IP0-1)*EPS5+PHI(IP0+1)*EPS6 + &
                        FPROTCD

                   ! phi = phi_old + omega*(phi_new - phi_old)
                   RESID=TOP-PHI0*BOT
                   ANORM=ANORM+ABS(RESID)
                   PHI(IP0)=PHI0+OMEGA*RESID/BOT
                ENDDO
                KSW=3-KSW
             ENDDO
          ENDDO

          ANORM=ANORM/NC3

          !      write(6,*)' N: rjac, omega, anorm', n, rjac, omega, anorm
          IF(PRNLEV.GT.6) THEN
             IP0=NCyz*int(NCLx/2)+NCLz*int(NCLy/2)+NCLz/2+1
             WRITE(OUTU,'(3x,A,2X,I5,4E11.4)') &
                  'N,ANORM,DEPS,OMEGA,PHI',N,ANORM,TOL,OMEGA,PHI(IP0)
          ENDIF

          IF(.not.QUNDER) THEN
             IF(N.EQ.1)THEN
                OMEGA=1./(1.-.5*RJAC**2)
             ELSE
                OMEGA=1./(1.-.25*RJAC**2*OMEGA)
             ENDIF
          ENDIF

          IF((N.GT.1).AND.(ANORM.LT.TOL)) GOTO 19

       ENDDO

    ENDIF
    !     -----

    IF(PRNLEV.GE.2) WRITE(OUTU,'(3x,A,2X,2E13.6)') &
         'Calculation not converged, MAXITS is too small ',ANORM,TOL

19  CONTINUE

    IF(PRNLEV.GE.2) WRITE(OUTU,'(3X,A,I5)') 'Number of iterations: ',N
    !
    RETURN
  END SUBROUTINE PBEQ3
  !
  SUBROUTINE PBEQ4(MAXITS,OMEGA,TOL,TEMP,KAPPA2, &
       NCLX,NCLY,NCLZ,DCEL,VMEMB,TMEMB,ZMEMB,TRANZ,ZBCEN, &
       PHI,EPSX,EPSY,EPSZ,CDEN,FKAPA, &
       QPBC,QNPBC,QUNDER)
    !-----------------------------------------------------------------------
    !     (Non-linear) Poisson-Boltzmann iterator for discretized cubic lattice.
    !     It is assumed that all arrays have been prepared elsewhere.
    !
    !                        (in Z)
    !                        ip0+1=ip6
    !                               | ip3
    !                               | /
    !                               |/
    !                    ip1 ----- ip0 ----- ip2   (in X)
    !                              /|
    !                             / |
    !                           ip4 |
    !                    (in Y)    ip5=ip0-1
    !
    ! NOTE: CDEN and FKAPA already contained 4*pi/h and h*h, respectively.
    !
    use chm_kinds
    use stream
    use consta
    use number
    implicit none
    REAL(CHM_REAL4)  PHI(*),EPSX(*),EPSY(*),EPSZ(*),CDEN(*),FKAPA(*)
    real(chm_real)  OMEGA,TOL,TEMP,TRANZ,ZBCEN,KAPPA2,DCEL
    real(chm_real)  VMEMB,TMEMB,ZMEMB
    INTEGER MAXITS,NCLX,NCLY,NCLZ
    LOGICAL QPBC,QNPBC,QUNDER

    ! Local variables
    real(chm_real)  FACTOR1,ZMEMB2,MKAPPA2,DCEL2,RJAC,RESID,ANORM
    real(chm_real)  MIONCD,FPROTCD,PHIF,RATIO,BOT,TOP,ZC
    REAL(CHM_REAL4)  PHI0,EPS1,EPS2,EPS3,EPS4,EPS5,EPS6
    INTEGER NCYZ,NC3,N
    INTEGER KSW
    INTEGER IG,JG,KG,IP0,IP0X,IP1X,IP2X,IP0Y,IP3Y,IP4Y,IP5Z,IP6Z
    !
    NCyz=NCLy*NCLz
    NC3=NCLx*NCyz
    DCEL2=DCEL*DCEL
    FACTOR1=ccelec/(kboltz*Temp) ! 1/(kcal/(mol*e))->1/(e/A)
    zmemb2=tranz-zbcen+0.5*tmemb+zmemb+rsmall
    MKAPPA2=KAPPA2*DCEL2
    !
    IF(NCLx.eq.3 .and. NCLy.eq.3 .and. NCLz.eq.3) THEN
       RJAC=(COS(PI/NCLx)+COS(PI/NCLy)+COS(PI/NCLz))/THREE
       IF(PRNLEV.GT.10) WRITE(OUTU,'(3x,A,E15.5)') 'RJAC',RJAC
    ELSE
       RJAC=(COS(PI/(NCLx-1))+COS(PI/(NCLy-1))+COS(PI/(NCLz-1)))/THREE
       IF(PRNLEV.GT.10) WRITE(OUTU,'(3x,A,E15.5)') 'RJAC',RJAC
    ENDIF

    KSW=1

    ! OUTPUT format
    IF(PRNLEV.GE.2) WRITE(OUTU,'(3x,a)') &
         'Non-linear PBEQ ITERATIONs for 1:1 charge-paired salt '

    IF(QPBC)THEN
       !     ------------
       ! periodic in XYZ

       DO N=1,MAXITS
          ANORM=ZERO

          ! x component
          DO IG=1,NCLx
             IP0X=(IG-1)*NCyz
             IP1X=-NCyz
             IF(IG.EQ.1) IP1X=NC3-NCyz    ! (NCLx-1)*NCyz
             IP2X=NCyz
             IF(IG.EQ.NCLx) IP2X=-(NC3-NCyz)
             ! y component
             DO JG=1,NCLy
                IP0Y=(JG-1)*NCLz+IP0X
                IP3Y=-NCLz
                IF(JG.EQ.1) IP3Y=NCyz-NCLz ! (NCLy-1)*NCLz
                IP4Y=NCLz
                IF(JG.EQ.NCLy) IP4Y=-(NCyz-NCLz)
                ! z component
                !wi               DO KG=1,NCLz
                DO KG=KSW,NCLZ-KSW+1,2
                   IP0 = IP0Y + KG
                   IP5Z=-1
                   IF(KG.EQ.1) IP5Z=NCLz-1
                   IP6Z=1
                   IF(KG.EQ.NCLz) IP6Z=-(NCLz-1)

                   PHI0=PHI(IP0)
                   EPS1=EPSX(IP0+IP1X)
                   EPS2=EPSX(IP0)
                   EPS3=EPSY(IP0+IP3Y)
                   EPS4=EPSY(IP0)
                   EPS5=EPSZ(IP0+IP5Z)
                   EPS6=EPSZ(IP0)

                   ! convert FKAPA array to ( kappa2 * nonlinearity ratio)
                   ! add the background charge (kappa2 * nonlinearity ratio * Vmemb) to CDEN
                   MIONCD=0.0
                   FPROTCD=CDEN(IP0)
                   IF(FKAPA(IP0).ne.0.0) THEN
                      PHIF=PHI0*FACTOR1
                      IF(VMEMB.NE.0.0)THEN
                         ZC=(KG-1)*DCEL
                         IF(ZC.GT.ZMEMB2) PHIF=(PHI0-VMEMB)*FACTOR1
                      ENDIF
                      IF(PHIF.GT.0.0) THEN
                         RATIO=(1.0+PHIF-EXP(-PHIF))/2.0/PHIF
                         MIONCD=MKAPPA2*RATIO
                      ELSE
                         RATIO=(EXP(PHIF)-1.0+PHIF)/2.0/PHIF
                         MIONCD=MKAPPA2*RATIO
                      ENDIF
                      IF(ZC.GT.ZMEMB2) FPROTCD=FPROTCD+MIONCD*VMEMB
                   ENDIF

                   BOT=EPS1+EPS2+EPS3+EPS4+EPS5+EPS6+MIONCD

                   TOP=PHI(IP0+IP1X)*EPS1+PHI(IP0+IP2X)*EPS2+ &
                        PHI(IP0+IP3Y)*EPS3+PHI(IP0+IP4Y)*EPS4+ &
                        PHI(IP0+IP5Z)*EPS5+PHI(IP0+IP6Z)*EPS6 + &
                        FPROTCD

                   !     phi = phi_old + omega*(phi_new - phi_old)
                   RESID=TOP-PHI0*BOT
                   ANORM=ANORM+ABS(RESID)
                   PHI(IP0)=PHI0+OMEGA*RESID/BOT
                ENDDO
                KSW=3-KSW
             ENDDO
          ENDDO

          ANORM=ANORM/NC3

          IF(PRNLEV.GT.6) THEN
             IP0=NCyz*int(NCLx/2)+NCLz*int(NCLy/2)+NCLz/2+1
             WRITE(OUTU,'(3x,A,2X,I5,4E11.4)') &
                  'N,ANORM,DEPS,OMEGA,PHI',N,ANORM,TOL,OMEGA,PHI(IP0)
          ENDIF

          IF(.not.Qunder) THEN
             IF(N.EQ.1)THEN
                OMEGA=1./(1.-.5*RJAC**2)
             ELSE
                OMEGA=1./(1.-.25*RJAC**2*OMEGA)
             ENDIF
          ENDIF

          IF((N.GT.1).AND.(ANORM.LT.TOL)) GOTO 19

       ENDDO

       !     -------------------------------------------
    ELSEIF(TMEMB.GT.ZERO .AND. .NOT. QNPBC)THEN
       !     -------------------------------------------
       ! periodic in XY, Z fixed edge-value (in membrane)

       DO N=1,MAXITS
          ANORM=ZERO

          !  x component
          DO IG=1,NCLx
             IP0X=(IG-1)*NCyz
             IP1X=-NCyz
             IF(IG.EQ.1) IP1X=NC3-NCyz    ! (NCLx-1)*NCyz
             IP2X=NCyz
             IF(IG.EQ.NCLx) IP2X=-(NC3-NCyz)
             !  y component
             DO JG=1,NCLy
                IP0Y=(JG-1)*NCLz+IP0X
                IP3Y=-NCLz
                IF(JG.EQ.1) IP3Y=NCyz-NCLz ! (NCLy-1)*NCLz
                IP4Y=NCLz
                IF(JG.EQ.NCLy) IP4Y=-(NCyz-NCLz)
                !  z component
                DO KG=KSW+1,NCLZ-KSW,2
                   IP0 = IP0Y + KG

                   PHI0=PHI(IP0)
                   EPS1=EPSX(IP0+IP1X)
                   EPS2=EPSX(IP0)
                   EPS3=EPSY(IP0+IP3Y)
                   EPS4=EPSY(IP0)
                   EPS5=EPSZ(IP0-1)
                   EPS6=EPSZ(IP0)

                   ! convert FKAPA array to ( kappa2 * nonlinearity ratio)
                   ! add the background charge (kappa2 * nonlinearity ratio * Vmemb) to CDEN
                   MIONCD=0.0
                   FPROTCD=CDEN(IP0)
                   IF(FKAPA(IP0).ne.0.0) THEN
                      PHIF=PHI0*FACTOR1
                      IF(VMEMB.NE.0.0)THEN
                         ZC=(KG-1)*DCEL
                         IF(ZC.GT.ZMEMB2) PHIF=(PHI0-VMEMB)*FACTOR1
                      ENDIF
                      IF(PHIF.GT.0.0) THEN
                         RATIO=(1.0+PHIF-EXP(-PHIF))/2.0/PHIF
                         MIONCD=MKAPPA2*RATIO
                      ELSE
                         RATIO=(EXP(PHIF)-1.0+PHIF)/2.0/PHIF
                         MIONCD=MKAPPA2*RATIO
                      ENDIF
                      IF(ZC.GT.ZMEMB2) FPROTCD=FPROTCD+MIONCD*VMEMB
                   ENDIF

                   BOT=EPS1+EPS2+EPS3+EPS4+EPS5+EPS6+MIONCD

                   TOP=PHI(IP0+IP1X)*EPS1+PHI(IP0+IP2X)*EPS2+ &
                        PHI(IP0+IP3Y)*EPS3+PHI(IP0+IP4Y)*EPS4+ &
                        PHI(IP0-1)*EPS5+PHI(IP0+1)*EPS6 + &
                        FPROTCD

                   !     phi = phi_old + omega*(phi_new - phi_old)
                   RESID=TOP-PHI0*BOT
                   ANORM=ANORM+ABS(RESID)
                   PHI(IP0)=PHI0+OMEGA*RESID/BOT
                ENDDO
                KSW=3-KSW
             ENDDO
          ENDDO

          ANORM=ANORM/NC3

          IF(PRNLEV.GT.6) THEN
             IP0=NCyz*int(NCLx/2)+NCLz*int(NCLy/2)+NCLz/2+1
             WRITE(OUTU,'(3x,A,2X,I5,4E11.4)') &
                  'N,ANORM,DEPS,OMEGA,PHI',N,ANORM,TOL,OMEGA,PHI(IP0)
          ENDIF

          IF(.not.QUNDER) THEN
             IF(N.EQ.1)THEN
                OMEGA=1./(1.-.5*RJAC**2)
             ELSE
                OMEGA=1./(1.-.25*RJAC**2*OMEGA)
             ENDIF
          ENDIF

          IF((N.GT.1).AND.(ANORM.LT.TOL)) GOTO 19

       ENDDO

       !     ----
    ELSE
       !     ----

       DO N=1,MAXITS
          ANORM=ZERO

          DO IG=2,NCLx-1
             IP0X=(IG-1)*NCyz
             DO JG=2,NCLy-1
                IP0Y=(JG-1)*NCLz+IP0X
                DO KG=KSW+1,NCLZ-KSW,2
                   IP0 = IP0Y + KG

                   PHI0=PHI(IP0)
                   EPS1=EPSX(IP0-NCyz)
                   EPS2=EPSX(IP0)
                   EPS3=EPSY(IP0-NCLz)
                   EPS4=EPSY(IP0)
                   EPS5=EPSZ(IP0-1)
                   EPS6=EPSZ(IP0)

                   ! convert FKAPA array to ( kappa2 * nonlinearity ratio)
                   ! add the background charge (kappa2 * nonlinearity ratio * Vmemb) to CDEN
                   MIONCD=0.0
                   FPROTCD=CDEN(IP0)
                   IF(FKAPA(IP0).ne.0.0) THEN
                      PHIF=PHI0*FACTOR1
                      IF(VMEMB.NE.0.0)THEN
                         ZC=(KG-1)*DCEL
                         IF(ZC.GT.ZMEMB2) PHIF=(PHI0-VMEMB)*FACTOR1
                      ENDIF
                      IF(PHIF.GT.0.0) THEN
                         RATIO=(1.0+PHIF-EXP(-PHIF))/2.0/PHIF
                         MIONCD=MKAPPA2*RATIO
                      ELSE
                         RATIO=(EXP(PHIF)-1.0+PHIF)/2.0/PHIF
                         MIONCD=MKAPPA2*RATIO
                      ENDIF
                      IF(ZC.GT.ZMEMB2) FPROTCD=FPROTCD+MIONCD*VMEMB
                   ENDIF

                   BOT=EPS1+EPS2+EPS3+EPS4+EPS5+EPS6+MIONCD

                   TOP=PHI(IP0-NCyz)*EPS1+PHI(IP0+NCyz)*EPS2+ &
                        PHI(IP0-NCLz)*EPS3+PHI(IP0+NCLz)*EPS4+ &
                        PHI(IP0-1)*EPS5+PHI(IP0+1)*EPS6 + &
                        FPROTCD

                   ! phi = phi_old + omega*(phi_new - phi_old)
                   RESID=TOP-PHI0*BOT
                   ANORM=ANORM+ABS(RESID)
                   PHI(IP0)=PHI0+OMEGA*RESID/BOT
                ENDDO
                KSW=3-KSW
             ENDDO
          ENDDO

          ANORM=ANORM/NC3

          !      write(6,*)' N: rjac, omega, anorm', n, rjac, omega, anorm
          IF(PRNLEV.GT.6) THEN
             IP0=NCyz*int(NCLx/2)+NCLz*int(NCLy/2)+NCLz/2+1
             WRITE(OUTU,'(3x,A,2X,I5,4E11.4)') &
                  'N,ANORM,DEPS,OMEGA,PHI',N,ANORM,TOL,OMEGA,PHI(IP0)
          ENDIF

          IF(.not.QUNDER) THEN
             IF(N.EQ.1)THEN
                OMEGA=1./(1.-.5*RJAC**2)
             ELSE
                OMEGA=1./(1.-.25*RJAC**2*OMEGA)
             ENDIF
          ENDIF

          IF((N.GT.1).AND.(ANORM.LT.TOL)) GOTO 19

       ENDDO

    ENDIF
    !     -----

    IF(PRNLEV.GE.2) WRITE(OUTU,'(3x,A,2X,2E13.6)') &
         'Calculation not converged, MAXITS is too small ',ANORM,TOL

19  CONTINUE

    IF(PRNLEV.GE.2) WRITE(OUTU,'(3X,A,I5)') 'Number of iterations: ',N
    !
    RETURN
  END SUBROUTINE PBEQ4
  !
  SUBROUTINE PBEQ5(MAXITS,OMEGA,TOL,EPSW,TMEMB,NCLX,NCLY,NCLZ, &
       PHI,CDEN,QFOCUS,QPBC,QNPBC)
    !-----------------------------------------------------------------------
    !     calculation in bulk solution for non-orthogonal basis set in GSBP.
    !
    use chm_kinds
    use memory
    use stream
    use consta
    use number
    implicit none
    REAL(CHM_REAL4)  PHI(*),CDEN(*)
    real(chm_real)  OMEGA,TOL,EPSW,TMEMB
    INTEGER MAXITS,NCLX,NCLY,NCLZ
    LOGICAL QFOCUS,QPBC,QNPBC
    ! Local variables
    real(chm_real)  RESID,ANORM,E123,RJAC,Omega2
    INTEGER PHI0
    INTEGER NCYZ,NC3,N
    INTEGER IG,JG,KG,IP0,IP0X,IP1X,IP2X,IP0Y,IP3Y,IP4Y,IP5Z,IP6Z
    INTEGER KSW
    !
    NCyz=NCLy*NCLz
    NC3=NCLx*NCyz

    ! conserve the initial mixing factor for successive calculations (?)
    OMEGA2=OMEGA

    IF(NCLx.eq.3 .and. NCLy.eq.3 .and. NCLz.eq.3) THEN
       RJAC=(COS(PI/NCLx)+COS(PI/NCLy)+COS(PI/NCLz))/THREE
       IF(PRNLEV.GT.10) WRITE(OUTU,'(3x,A,E15.5)') 'RJAC',RJAC
    ELSE
       RJAC=(COS(PI/(NCLx-1))+COS(PI/(NCLy-1))+COS(PI/(NCLz-1)))/THREE
       IF(PRNLEV.GT.10) WRITE(OUTU,'(3x,A,E15.5)') 'RJAC',RJAC
    ENDIF

    KSW=1

    IF(QPBC .AND. .NOT.QFOCUS  .AND. .NOT. QNPBC)THEN
       !     -------------------------------------------------
       ! periodic in XYZ

       DO N=1,MAXITS
          ANORM=ZERO

          E123=SIX*EPSW
          DO IG=1,NCLx
             IP0X=(IG-1)*NCyz
             IP1X=-NCyz
             IF(IG.EQ.1) IP1X=NC3-NCyz     ! (NCLx-1)*NCyz
             IP2X=NCyz
             IF(IG.EQ.NCLx) IP2X=-(NC3-NCyz)
             DO JG=1,NCLy
                IP0Y=(JG-1)*NCLz+IP0X
                IP3Y=-NCLz
                IF(JG.EQ.1) IP3Y=NCyz-NCLz ! (NCLy-1)*NCLz
                IP4Y=NCLz
                IF(JG.EQ.NCLy) IP4Y=-(NCyz-NCLz)
                !wi               DO KG=1,NCLz
                DO KG=KSW,NCLZ-KSW+1,2
                   IP0 = IP0Y + KG
                   IP5Z=-1
                   IF(KG.EQ.1) IP5Z=NCLz-1
                   IP6Z=1
                   IF(KG.EQ.NCLz) IP6Z=-(NCLz-1)
                   RESID=(PHI(IP0+IP1X)+PHI(IP0+IP2X)+ &
                        PHI(IP0+IP3Y)+PHI(IP0+IP4Y)+ &
                        PHI(IP0+IP5Z)+PHI(IP0+IP6Z))*EPSW- &
                        PHI(IP0)*E123+CDEN(IP0)
                   ANORM=ANORM+ABS(RESID)
                   PHI(IP0)=PHI(IP0)+OMEGA*RESID/E123
                ENDDO
                KSW=3-KSW
             ENDDO
          ENDDO

          ANORM=ANORM/NC3

          IF(PRNLEV.GT.10) THEN
             IP0=NCyz*int(NCLx/2)+NCLz*int(NCLy/2)+NCLz/2+1
             WRITE(OUTU,'(3x,A,2X,I5,4E11.4)') &
                  'N,ANORM,DEPS,OMEGA,PHI',N,ANORM,TOL,OMEGA,PHI(IP0)
          ENDIF

          IF(N.EQ.1)THEN
             OMEGA=1./(1.-.5*RJAC**2)
          ELSE
             OMEGA=1./(1.-.25*RJAC**2*OMEGA)
          ENDIF

          IF((N.GT.1).AND.(ANORM.LT.TOL)) GOTO 19

       ENDDO

       !     --------------------------------------------------------------
    ELSEIF(TMEMB.GT.ZERO .AND. .NOT.QFOCUS  .AND. .NOT. QNPBC)THEN
       !     --------------------------------------------------------------
       ! periodic in XY, Z fixed edge-value (in membrane)

       DO N=1,MAXITS
          ANORM=ZERO

          E123=SIX*EPSW
          DO IG=1,NCLx
             IP0X=(IG-1)*NCyz
             IP1X=-NCyz
             IF(IG.EQ.1) IP1X=NC3-NCyz     ! (NCLx-1)*NCyz
             IP2X=NCyz
             IF(IG.EQ.NCLx) IP2X=-(NC3-NCyz)
             DO JG=1,NCLy
                IP0Y=(JG-1)*NCLz+IP0X
                IP3Y=-NCLz
                IF(JG.EQ.1) IP3Y=NCyz-NCLz ! (NCLy-1)*NCLz
                IP4Y=NCLz
                IF(JG.EQ.NCLy) IP4Y=-(NCyz-NCLz)
                DO KG=KSW+1,NCLZ-KSW,2
                   IP0 = IP0Y + KG
                   RESID=(PHI(IP0+IP1X)+PHI(IP0+IP2X)+ &
                        PHI(IP0+IP3Y)+PHI(IP0+IP4Y)+ &
                        PHI(IP0-1)+PHI(IP0+1))*EPSW- &
                        PHI(IP0)*E123+CDEN(IP0)
                   ANORM=ANORM+ABS(RESID)
                   PHI(IP0)=PHI(IP0)+OMEGA*RESID/E123
                ENDDO
                KSW=3-KSW
             ENDDO
          ENDDO

          ANORM=ANORM/NC3

          IF(PRNLEV.GT.10) THEN
             IP0=NCyz*int(NCLx/2)+NCLz*int(NCLy/2)+NCLz/2+1
             WRITE(OUTU,'(3x,A,2X,I5,4E11.4)') &
                  'N,ANORM,DEPS,OMEGA,PHI',N,ANORM,TOL,OMEGA,PHI(IP0)
          ENDIF

          IF(N.EQ.1)THEN
             OMEGA=1./(1.-.5*RJAC**2)
          ELSE
             OMEGA=1./(1.-.25*RJAC**2*OMEGA)
          ENDIF

          IF((N.GT.1).AND.(ANORM.LT.TOL)) GOTO 19

       ENDDO

       !     ----
    ELSE
       !     ----

       DO N=1,MAXITS
          ANORM=ZERO

          DO IG=2,NCLx-1
             IP0X=(IG-1)*NCyz
             DO JG=2,NCLy-1
                IP0Y=(JG-1)*NCLz+IP0X
                DO KG=KSW+1,NCLZ-KSW,2
                   IP0 = IP0Y + KG
                   E123=SIX*EPSW
                   RESID=(PHI(IP0-NCyz)+PHI(IP0+NCyz)+ &
                        PHI(IP0-NCLz)+PHI(IP0+NCLz)+ &
                        PHI(IP0-1)+PHI(IP0+1))*EPSW - &
                        PHI(IP0)*E123+CDEN(IP0)
                   ANORM=ANORM+ABS(RESID)
                   PHI(IP0)=PHI(IP0)+OMEGA*RESID/E123
                ENDDO
                KSW=3-KSW
             ENDDO
          ENDDO

          ANORM=ANORM/NC3

          !      write(6,*)' N: rjac, omega, anorm', n, rjac, omega, anorm
          IF(PRNLEV.GT.10) THEN
             IP0=NCyz*int(NCLx/2)+NCLz*int(NCLy/2)+NCLz/2+1
             WRITE(OUTU,'(3x,A,2X,I5,4E11.4)') &
                  'N,ANORM,DEPS,OMEGA,PHI',N,ANORM,TOL,OMEGA,PHI(IP0)
          ENDIF

          IF(N.EQ.1)THEN
             OMEGA=1./(1.-.5*RJAC**2)
          ELSE
             OMEGA=1./(1.-.25*RJAC**2*OMEGA)
          ENDIF

          IF((N.GT.1).AND.(ANORM.LT.TOL)) GOTO 19

       ENDDO

    ENDIF
    !     -----

    IF(PRNLEV.GE.2) WRITE(OUTU,'(3x,A,2X,2E13.6)') &
         'Calculation not converged, MAXITS is too small ',ANORM,TOL

19  CONTINUE

    OMEGA=OMEGA2
    !
    RETURN
  END SUBROUTINE PBEQ5
  !
  SUBROUTINE DECOMP(NTPRP,LSTPRP,X,Y,Z,CG,WMAIN, &
       NCLX,NCLY,NCLZ,DCEL,PHI, &
       TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN,FACTOR,QBSPL)
    !-----------------------------------------------------------------------
    ! This subroutine computes the individual atomic contribution to the total
    ! electrostatic free energy of solvation.
    !
    use chm_kinds
    use consta
    use number
    use stream
    use param_store, only: set_param

    implicit none

    integer ntprp, lstprp(*)
    integer NCLX,NCLY,NCLZ
    real(chm_real)  x(*),y(*),z(*),cg(*),wmain(*)
    real(chm_real)  dcel,tranx,trany,tranz,xbcen,ybcen,zbcen
    real(chm_real4)  phi(*)
    real(chm_real)  FACTOR
    logical QBSPL
    ! local
    integer ncyz,il,i,ix,iy,iz,n1,in1,n2,in2,n3,in3
    real(chm_real)  chi,xi,yi,zi,ai,bi,ci,fi
    real(chm_real)  enet, enelp
    real(chm_real)  EREMAI !rjp
    integer EINTEG !rjp
    ! B-spline
    integer jx1,jx2,jy1,jy2,jz1,jz2,k,l,m,ipx,ipy,ipz,nfil
    real(chm_real)  xc,yc,zc
    !
    ncyz=ncly*nclz
    enelp=zero
    !
    !==============================================================================
    IF(QBSPL) THEN
       !
       ! Main loop by atoms
       !
       il_loop: do il=1,ntprp
          i=lstprp(il)
          chi=cg(i)
          wmain(i)=ZERO
          IF(CHI.EQ.0.0) cycle il_loop
          xi=x(i)+tranx-xbcen
          yi=y(i)+trany-ybcen
          zi=z(i)+tranz-zbcen
          if(xi.lt.zero .or. xi.gt.2*tranx .or. &
               yi.lt.zero .or. yi.gt.2*trany .or. &
               zi.lt.zero .or. zi.gt.2*tranz) cycle il_loop

          nfil=1
          ix=int(xi/dcel)+1
          iy=int(yi/dcel)+1
          iz=int(zi/dcel)+1
          jx1=ix-nfil
          if(jx1.lt.1)jx1=1
          jx2=ix+nfil+1
          if(jx2.gt.nclx)jx2=nclx
          jy1=iy-nfil
          if(jy1.lt.1)jy1=1
          jy2=iy+nfil+1
          if(jy2.gt.ncly)jy2=ncly
          jz1=iz-nfil
          if(jz1.lt.1)jz1=1
          jz2=iz+nfil+1
          if(jz2.gt.nclz)jz2=nclz

          DO K=jx1,jx2
             IPX=(K-1)*NCyz
             XC=(K-1)*DCEL
             !
             DO L=jy1,jy2
                IPY=(L-1)*NCLz
                YC=(L-1)*DCEL
                !
                DO M=jz1,jz2
                   IPZ=M+IPY+IPX
                   ZC=(M-1)*DCEL

                   ai=1.5-(xi-xc)/dcel
                   bi=1.5-(yi-yc)/dcel
                   ci=1.5-(zi-zc)/dcel
                   fi=M3(ai)*M3(bi)*M3(ci)

                   enet=FACTOR*fi*chi*phi(ipz)*CCELEC
                   enelp=enelp+enet
                   wmain(i)=wmain(i)+enet

                enddo
             enddo
          enddo
       enddo il_loop
       !
    ELSE
       !
       !     Main loop by atoms
       illoop2: do il=1,ntprp
          i=lstprp(il)
          chi=cg(i)
          wmain(i)=ZERO
          IF(CHI.EQ.0.0) cycle illoop2
          xi=x(i)+tranx-xbcen
          yi=y(i)+trany-ybcen
          zi=z(i)+tranz-zbcen
          if(xi.lt.zero .or. xi.gt.2*tranx .or. &
               yi.lt.zero .or. yi.gt.2*trany .or. &
               zi.lt.zero .or. zi.gt.2*tranz) cycle illoop2

          ix=int(xi/dcel)+1
          iy=int(yi/dcel)+1
          iz=int(zi/dcel)+1
          !     Atom charge distribution by 8 adjacent grid points
          do n1=1,2
             in1=ix+n1-1
             ai=abs(xi-(in1-1)*dcel)/dcel
             in1=(in1-1)*ncyz
             ai=1.-ai

             do n2=1,2
                in2=iy+n2-1
                bi=abs(yi-(in2-1)*dcel)/dcel
                in2=(in2-1)*nclz
                bi=ai*(1.-bi)

                do n3=1,2
                   in3=iz+n3-1
                   ci=abs(zi-(in3-1)*dcel)/dcel
                   fi=bi*(1.-ci)
                   in3=in1+in2+in3

                   enet=FACTOR*fi*chi*phi(in3)*CCELEC
                   enelp=enelp+enet
                   wmain(i)=wmain(i)+enet

                enddo
             enddo
          enddo
       enddo illoop2
       !
    ENDIF
    !==============================================================================
    !
    EINTEG = INT(ENELP)
    EREMAI = ENELP - EINTEG
    CALL set_param('ENPI',EINTEG)
    call set_param('ENPR',EREMAI)

    IF(PRNLEV.GE.2) WRITE(OUTU,'(3X,A)') &
         'The atomic contributions have been stored in WMAIN'

    IF(PRNLEV.GE.2) WRITE(OUTU,'(3X,A,F13.5)') &
         'Electrostatic energy [KCAL/MOL] = ',ENELP

    call set_param('ENPB',ENELP)
    RETURN
  END SUBROUTINE DECOMP
  !
  !     XIAO_QC_UW0609: ADD potential SCC-DFTB iteration here
  SUBROUTINE PBFORCE(NATOM,X,Y,Z,CG,ENER1,ENER2,DX,DY,DZ, &
       IPBEQ,ICALL,QPRIN &
#if KEY_SCCDFTB==1
       ,QSCCTB,ENER3,QGAS2 &        
#endif
       )
    !-----------------------------------------------------------------------
    !     This routine calcuates
    !          RXNF (Reaction Field Force),
    !          DBF (Dielectric Boundary Force),
    !          IBF (Ionic Boundary Force),
    !          NPF (Nonpolar Surface Force),
    !          Solvation Free Energy (EPB1), and
    !          Nonpolar Solvation Free Energy (EPB2).
    !     It is called in ENERGY.SRC and the returned values are
    !     first derivatives not Forces.
    !     The calculated forces are added to input values(DX,DY,DZ)
    !     in SUBROUTINE WAREA.
    !
    use chm_kinds
    use memory
    use number
    use stream
    use dimens_fcm
    use consta
    use timerm
    ! XIAO_QC_UW0609
#if KEY_SCCDFTB==1
    use sccdftb
    use sccpb
    use blockscc_fcm
    use gamess_fcm
#endif 
    use machutil,only:wrttim
    !
    implicit none
    !
    real(chm_real4),allocatable,dimension(:) :: IPHIB
    real(chm_real)  ENER1,ENER2
    real(chm_real)  X(*),Y(*),Z(*),CG(*),DX(*),DY(*),DZ(*)
    INTEGER NATOM,I,ICALL,IPBEQ
    LOGICAL QPRIN
    ! XIAO_QC_UW0609
#if KEY_SCCDFTB==1
    LOGICAL QSCCTB,QGAS2
    REAL(chm_real)  ENER3
    REAL(chm_real)  CG1(NATOM)
    REAL(chm_real)  RXNEX(NATOM),RXNEY(NATOM),RXNEZ(NATOM)
    !     REAL(chm_real)  ESCCTBPB,ESCCTBPBOLD,ESCCTBPBGAS
    INTEGER IATOM,IREP,ITER
#endif 

    !=====
    IF(MOD(ICALL,IPBEQ).EQ.0) THEN
       !=====

       call chmalloc('pbeq.src','PBFORCE','IPHIB',1,cr4=IPHIB)
       !-----------------------------------------------------------------------
       !     In vacuum calculations, vmemb, kappa2 (salt concentration) = 0.0
       !                             epsw, epsm, epsh, epsd, and epsc = 1.0
       !     However, to consider periodic conditions, TMEMB is not equal to zero.
       !-----------------------------------------------------------------------
       IF(QPRIN .and. prnlev >= 2) WRITE(OUTU,'(/,3x,A,f8.3)') &
            'Constructing all space-dependent functions: EPS =', epsp
       !
       CALL MAYER(NTPRP,LSTPRP,ONE,EPSP,ZERO, &
            !                                  epsw epsp kappa2
            X,Y,Z,CG,PBRAD,NCLX,NCLY,NCLZ,DCEL,WATR,IONR,MAY, &
            MAYX,MAYY,MAYZ,SWIN,ICHC,IPHIP, &
            !                        -----                 -----
            ZERO,TMEMB,ZMEMB,ONE,NIMGB, &
            !          vmemb            epsm
            HTMEMB,ONE,HDENS,POSALP,NEGALP,QCHRG,POFFST,NOFFST, &
            !                 epsh
            DROPLET,ONE,XDROPLET,YDROPLET,ZDROPLET,Qdtom,Qdkap, &
            !                 epsd
            ONE,LXMAX,LYMAX,LZMAX,LXMIN,LYMIN,LZMIN,Qbtom,Qbkap, &
            !          epsb
            ONE,Xcyln,Ycyln,Zcyln,Rcyln,Hcyln,Qctom,Qckap, &
            !          epsc
            ONE,Xcone,Ycone,Zcone,Ax,By,Cz,Qectom,Qeckap, &
            !          epsec
            -1, -1, &
            TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
            IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
            QKEEP,.false.,.true.,QBSPL,QA,QB, &
            !                 QREEN   QSMTH
            RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
            RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
            MAJOR,MINOR,QPROLATE, &
            QINTBP,QZERO,.false.,IPHIB,QPBC,QNPBC, &
            !          -----  -----  QFOCUS
            QNONLINEAR,QPARTLINEAR,QFKAP, &
            IPOX,IPOY,IPOZ,MAPT,NATOM)

       IF (TIMER.GT.1 .and. ICALL.eq.0) &
            CALL WRTTIM('Grid parameters preparation in vacuum times:')

       IF(QFMG) THEN
          CALL PBEQ2(NATOM,X,Y,Z,CG,.true.,QPRIN)
          IF (TIMER.GT.1 .and. ICALL.eq.0) &
               CALL WRTTIM('FMG iteration times:')
       ELSE
          IF(QOLDPB)THEN
             CALL OLDPBEQ1(MAXITS,DOME,DEPS,NCLX,NCLY,NCLZ, &
                  IPHIP,MAYX,MAYY,MAYZ,ICHC,MAY,TMEMB, &
                  .false.,.false.,QPBC,QNPBC,QPRIN)
             !                QOSOR   QFOCUS
          ELSE
             CALL PBEQ1(MAXITS,DOME,DEPS,ONE,EPSP,ONE,ZERO, &
                  !                                       ---         kappa2
                  NCLX,NCLY,NCLZ,IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
                  IPHIP,MAYX,MAYY,MAYZ,ICHC,MAY,TMEMB, &
                  .false.,.false.,QPBC,QNPBC,QPRIN)
             !                QOSOR   QFOCUS
          ENDIF
          IF (TIMER.GT.1 .and. ICALL.eq.0) &
               CALL WRTTIM('SOR iteration in vacuum times:')
       ENDIF

       !----------------------------------------------------------------
       !     In solution
       !----------------------------------------------------------------
       IF(QPRIN .and. prnlev >= 2) WRITE(OUTU,'(/,3x,A,f8.3)') &
            'Constructing all space-dependent functions: EPS =', epsw
       !
       CALL MAYER(NTPRP,LSTPRP,EPSW,EPSP,KAPPA2, &
            !                                   ----
            X,Y,Z,CG,PBRAD,NCLX,NCLY,NCLZ,DCEL,WATR,IONR,MAY, &
            MAYX,MAYY,MAYZ,SWIN,ICHC,IPHIW, &
            !          -----                 -----
            VMEMB,TMEMB,ZMEMB,EPSM,NIMGB, &
            HTMEMB,EPSH,HDENS,POSALP,NEGALP,QCHRG,POFFST,NOFFST, &
            DROPLET,EPSD,XDROPLET,YDROPLET,ZDROPLET,Qdtom,Qdkap, &
            EPSB,LXMAX,LYMAX,LZMAX,LXMIN,LYMIN,LZMIN,Qbtom,Qbkap, &
            EPSC,Xcyln,Ycyln,Zcyln,Rcyln,Hcyln,Qctom,Qckap, &
            EPSEC,Xcone,Ycone,Zcone,Ax,By,Cz,Qectom,Qeckap, &
            -1,-1,&
            TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
            IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
            QKEEP,.false.,.true.,QBSPL,QA,QB, &
            !                 QREEN   QSMTH
            RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
            RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
            MAJOR,MINOR,QPROLATE, &
            QINTBP,QZERO,.false.,IPHIB,QPBC,QNPBC, &
            !          ------,----- QFOCUS
            QNONLINEAR,QPARTLINEAR,QFKAP, &
            IPOX,IPOY,IPOZ,MAPT, NATOM)
       IF (TIMER.GT.1 .and. ICALL.eq.0) &
            CALL WRTTIM('Grid parameters preparation in solution times:')

       IF(QFMG) THEN
          CALL PBEQ2(NATOM,X,Y,Z,CG,.false.,QPRIN)
          IF (TIMER.GT.1 .and. ICALL.eq.0) &
               CALL WRTTIM('FMG iteration times:')
       ELSE
          IF(QOLDPB)THEN
             CALL OLDPBEQ1(MAXITS,DOME,DEPS,NCLX,NCLY,NCLZ, &
                  IPHIW,MAYX,MAYY,MAYZ, &
                  ICHC,MAY,TMEMB, &
                  QOSOR,.false.,QPBC,QNPBC,QPRIN)
             !                QOSOR   QFOCUS
          ELSE
             CALL PBEQ1(MAXITS,DOME,DEPS,EPSW,EPSP,EPSM,KAPPA2, &
                  NCLX,NCLY,NCLZ,IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
                  IPHIW,MAYX,MAYY,MAYZ, &
                  ICHC,MAY,TMEMB, &
                  QOSOR,.false.,QPBC,QNPBC,QPRIN)
             !                      QFOCUS
          ENDIF
          IF (TIMER.GT.1 .and. ICALL.eq.0) &
               CALL WRTTIM('SOR iteration in solution times:')
       ENDIF

       !----------------------------------------------------------------------------
       !     RFORCE: the electrostatic solvation free energy and reaction field forces
       !     BFORCE: the boundary forces: DBF, IBF
       !             the nonpolar solvation free energy and forces
       !----------------------------------------------------------------------------
       CALL RFORCE(NTPRP,LSTPRP,X,Y,Z,CG,NCLX,NCLY,NCLZ,DCEL, &
            IPHIW,IPHIP, &
            TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN,HALF, &
            RXNFX,RXNFY,RXNFZ,EPB1,QPRIN,QBSPL)

       IF (TIMER.GT.1 .and. ICALL.eq.0) &
            CALL WRTTIM('Reaction field forces times:')

       CALL BFORCE(NTPRP,LSTPRP,X,Y,Z,PBRAD,EPSW,EPSP, &
            NCLX,NCLY,NCLZ,DCEL, &
            IPHIW,MAYX,MAYY,MAYZ,SWIN, &
            TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
            DBFX,DBFY,DBFZ, &
            IBFX,IBFY,IBFZ,MAY,KAPPA2, &
            NPFX,NPFY,NPFZ,STCO,EPB2,QPRIN &
            !  XIAO_QC_UW0609: calculate non-polar contribution 
            !                  after qm radius obtained
#if KEY_SCCDFTB==1
            ,qscctb  &               
#endif
            )
       IF (TIMER.GT.1 .and. ICALL.eq.0) &
            CALL WRTTIM('Boundary forces times:')
#if KEY_SCCDFTB==1
       if ((.not. qscctb) .or. (.not. qmused_sccdftb)) goto 700
       !     ----------------------------------------------------------------

       DO IATOM=1,NATOM
          IF (IGMSEL(IATOM) .ne. 0) THEN
             CG1(IATOM)=ZERO
          ELSE
             CG1(IATOM)=CG(IATOM)
          ENDIF
       ENDDO

       !     WRITE(*,*) "PBFORCE> SCCPB_MM"

       CALL SCCPB_MM(NATOM,X,Y,Z,CG1,ICALL,.false., &!QPRIN,
            RXNEX,RXNEY,RXNEZ)
       !     WRITE(*,*) "PBFORCE> SCCPB_MM DONE"

       !     XIAO_QC_UW0609: DO SCC-DFTB calculation first in the gas phase to get
       !     the initial Mulligan
       !     charge and hence the MULLIK must be enforced, which is required in
       !     the SCC-PB Calculation

       IF (QGAS2) THEN
          QPBSCC=.FALSE.
          IF (.NOT.LMULIK) LMULIK=.TRUE.
          !      WRITE(*,*) "PBFORCE> SCC GAS PHASE..."
          CALL SCCTBENE(ESCCTBPB,X,Y,Z,DX,DY,DZ,NATOM,.false.)
          ESCCTBPBGAS=ESCCTBPB
       ENDIF

       ESCCTBPBOLD=ESCCTBPB

       DO ITER=1,MXTPSC

          DO IATOM=1,NATOM
             CG1(IATOM)=ZERO
          ENDDO

          DO IREP=1,NSCCRP
             DO IATOM=1,NSCCTC
                CG1(IQMLST(IATOM,IREP))=QMULI2(IATOM,IREP)
             ENDDO
          ENDDO

          IF (QCHDRAD) CALL updrad(pbrad)

          !      WRITE(*,*) "PBFORCE> SCCPB_QM"
          CALL SCCPB_QM(NATOM,X,Y,Z,CG1,ITER,.false.)
          !      WRITE(*,*) "PBFORCE> SCCPB_QM DONE"

          QPBSCC=.TRUE.
          IF (.NOT.LMULIK) LMULIK=.TRUE.
          CALL SCCTBENE(ESCCTBPB,X,Y,Z,DX,DY,DZ,NATOM,.false.)
          QPBSCC=.FALSE.

          IF (ABS(ESCCTBPBOLD-ESCCTBPB) .le. PSCTOL) goto 200
          ESCCTBPBOLD=ESCCTBPB
       ENDDO

200    CONTINUE

       ESCCTBPBOLD=ESCCTBPB

       QPBSCC=.TRUE.
       IF (.NOT.LMULIK) LMULIK=.TRUE.
       CALL SCCTBENE(ESCCTBPB,X,Y,Z,DX,DY,DZ,NATOM,.true.)
       QPBSCC=.FALSE.

       DO IREP=1,NSCCRP
          DO IATOM=1,NSCCTC
             CG1(IQMLST(IATOM,IREP))=QMULI2(IATOM,IREP)
          ENDDO
       ENDDO

       IF (QCHDRAD) CALL updrad(pbrad)
       CALL SCCPB_QM(NATOM,X,Y,Z,CG1,0,.false.)

       !     WRITE(*,*) "Additional force for SCC/PB...."

       !     Start additional force calculations
       !     I.Reaction field force due to QM reaction field on QM atoms
       !     II.Reaction field force due to MM reaction field on QM atoms
       CALL SCCPB4_QM(NATOM,X,Y,Z,CG,RXNFX,RXNFY, &
            RXNFZ,RXNEX,RXNEY,RXNEZ, &
            NCLX,NCLY,NCLZ,DCEL, &
            IPHIWTB,IPHIPTB,TRANX,TRANY,TRANZ, &
            XBCEN,YBCEN,ZBCEN,ONE,QPRIN,QBSPL) 

       !     III.Reaction field force due to QM reaction field on MM atoms

       CALL SCCPB4_MM(NATOM,X,Y,Z,CG,RXNFX,RXNFY, &
            RXNFZ,NCLX,NCLY,NCLZ,DCEL, &
            IPHIWTB,IPHIPTB,TRANX,TRANY,TRANZ, &
            XBCEN,YBCEN,ZBCEN,ONE,QPRIN,QBSPL)
       !
       CALL BFORCE2(NTPRP,LSTPRP,X,Y,Z,PBRAD,EPSW,EPSP, &
            NCLX,NCLY,NCLZ,DCEL, &
            IPHIWTB,MAYX,MAYY,MAYZ,SWIN, &
            TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
            DBFX,DBFY,DBFZ, &
            IBFX,IBFY,IBFZ,MAY,KAPPA2, &
            NPFX,NPFY,NPFZ,STCO,EPB2,QPRIN)

       CALL BFORCE3(NTPRP,LSTPRP,X,Y,Z,PBRAD,EPSW,EPSP, &
            NCLX,NCLY,NCLZ,DCEL, &
            IPHIWTB,IPHIWEX, &
            MAYX,MAYY,MAYZ,SWIN, &
            TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
            DBFX,DBFY,DBFZ, &
            IBFX,IBFY,IBFZ,MAY,KAPPA2)
       CALL BFORCE3(NTPRP,LSTPRP,X,Y,Z,PBRAD,EPSW,EPSP, &
            NCLX,NCLY,NCLZ,DCEL, &
            IPHIWEX,IPHIWTB, &
            MAYX,MAYY,MAYZ,SWIN, &
            TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
            DBFX,DBFY,DBFZ, &
            IBFX,IBFY,IBFZ,MAY,KAPPA2)
       IF (QPRIN) THEN
          if (prnlev >= 2) then
             WRITE(outu,'(/,3X,A,F13.5,A)') &
                  'The Quantum Mechanical Solvation Energy  = ', &
                  (ESCCTBPB-ESCCTBPBGAS),' [KCAL/MOL]'
             WRITE(outu,'(/,3X,A,F13.5,A)') &
                  'The Total Solvation Energy  = ', &
                  (ESCCTBPB-ESCCTBPBGAS+ener1+ener2),' [KCAL/MOL]'
          endif

          IF (PRNLEV .GE. 6) THEN
             WRITE(outu,'(/,2X,A)') 'Mullikan Charge After Solvation:'
             DO IREP=1,NSCCRP
                DO IATOM=1,NSCCTC
                   WRITE(OUTU,'(x,2(I5),f10.5)') IREP,IATOM, &
                        CG1(IQMLST(IATOM,IREP))
                ENDDO
             ENDDO
          ENDIF
       ENDIF

700    CONTINUE
#endif 

       call chmdealloc('pbeq.src','PBFORCE','IPHIB',1,cr4=IPHIB)
       !=====
    ENDIF
    !=====
    CALL WAREA(NATOM,DX,DY,DZ,RXNFX,RXNFY,RXNFZ, &
         DBFX,DBFY,DBFZ, &
         IBFX,IBFY,IBFZ, &
         NPFX,NPFY,NPFZ, &
         EPB1,EPB2,ENER1,ENER2 &
         ! XIAO_QC_UW0609
#if KEY_SCCDFTB==1
         ,ESCCTBPB,ENER3 &            
#endif
         )
    !

    RETURN
  END SUBROUTINE PBFORCE
  !
  SUBROUTINE WAREA(NATOM,DX,DY,DZ,RXNFX,RXNFY,RXNFZ, &
       DBFX,DBFY,DBFZ,IBFX,IBFY,IBFZ, &
       NPFX,NPFY,NPFZ,EPB1,EPB2,ENER1,ENER2 &
#if KEY_SCCDFTB==1
       ,ESCCTBPB,ENER3  &          
#endif
       )
    !-----------------------------------------------------------------------
    !     Working AREA
    !     Each force,in fact first derivative, is added to DX, DY, DZ.
    !
    use chm_kinds
    use stream
    implicit none
    !
    real(chm_real)   DX(*),DY(*),DZ(*)
    real(chm_real)   RXNFX(*),RXNFY(*),RXNFZ(*)
    real(chm_real)   DBFX(*),DBFY(*),DBFZ(*)
    real(chm_real)   IBFX(*),IBFY(*),IBFZ(*)
    real(chm_real)   NPFX(*),NPFY(*),NPFZ(*)
    real(chm_real)   EPB1,EPB2,ENER1,ENER2
    INTEGER  NATOM,I
#if KEY_SCCDFTB==1
    real(chm_real) ESCCTBPB,ENER3        
#endif
    !
    do i = 1, natom
       dx(i) = dx(i) - rxnfx(i) - dbfx(i) - ibfx(i) - npfx(i)
       dy(i) = dy(i) - rxnfy(i) - dbfy(i) - ibfy(i) - npfy(i)
       dz(i) = dz(i) - rxnfz(i) - dbfz(i) - ibfz(i) - npfz(i)
    enddo
    ENER1 = EPB1
    ENER2 = EPB2
#if KEY_SCCDFTB==1
    ENER3 = ESCCTBPB
#endif 
#if KEY_DEBUG==1
    if (prnlev .ge. 6) then
       do i = 1, natom
          write(outu,'(x,A,3f10.5)') 'RF',rxnfx(i),rxnfy(i),rxnfz(i)
          write(outu,'(x,A,3f10.5)') 'DF',dbfx(i),dbfy(i),dbfz(i)
          write(outu,'(x,A,3f10.5)') 'IF',ibfx(i),ibfy(i),ibfz(i)
          write(outu,'(x,A,3f10.5)') 'NF',npfx(i),npfy(i),npfz(i)
       enddo
    endif
#endif 
    !
    RETURN
  END SUBROUTINE WAREA
  !
  SUBROUTINE COUNTERION(NCLX,NCLY,NCLZ,DCEL,PHI,MCDEN, &
       VMEMB,TMEMB,ZMEMB,TRANZ,ZBCEN,CONC,TEMP, &
       Qnonlinear,Qpartlinear)
    !-----------------------------------------------------------------------
    ! This subroutine computes the number of count ions
    !
    use chm_kinds
    use stream
    use consta
    use number
    implicit none
    integer NCLX,NCLY,NCLZ
    real(chm_real)  dcel,tranz,zbcen,conc,temp
    real(chm_real)  vmemb,tmemb,zmemb
    real(chm_real4)  phi(*),mcden(*)
    logical Qnonlinear,Qpartlinear
    ! local
    real(chm_real)  factor1
    real(chm_real)  bulk_num,volume,nion_num,pion_num,bulk_rho
    real(chm_real)  nion_numzp,pion_numzp,nion_numzn,pion_numzn
    real(chm_real)  nion_nummb,pion_nummb,nfactor,pfactor,area
    real(chm_real)  dcel2,dcel3,zz,zc,zmemb2,phif
    real(chm_real)  liter

    integer ip0,nc3,ncyz,ig,jg,kg,iii
    !
    ncyz=ncly*nclz
    nc3=nclx*ncly*nclz
    dcel2=dcel*dcel
    dcel3=dcel2*dcel
    zmemb2=tranz-zbcen+0.5*tmemb+zmemb+rsmall
    liter=0.001D0

    ! conversion factoer from 1/(kcal/(mol*e)) to 1/(e/A)
    factor1 = ccelec  / ( kboltz * Temp )

    ! calculate ion accessible volume
    volume=0.0D0
    do ip0=1,nc3
       if(mcden(ip0).ne.0.0) volume=volume+dcel3
    enddo
    !
    ! bulk density and number of counter ions
    bulk_rho=conc*(avogadro/liter)*(angstrom**3) ! [unit charge]/A^3
    bulk_num=bulk_rho*volume

    ! deviation from bulk number of count ions
    pion_num=0.0D0
    nion_num=0.0D0
    pion_numzp=0.0D0
    nion_numzp=0.0D0
    pion_numzn=0.0D0
    nion_numzn=0.0D0
    pion_nummb=0.0D0
    nion_nummb=0.0D0
    do kg=1,nclz
       zz=(kg-1)*dcel-tranz+zbcen
       do ig=1,nclx
          iii=(ig-1)*ncyz+kg
          do jg=1,ncly
             ip0=iii+(jg-1)*nclz
             if(mcden(ip0).ne.0.0) then
                phif=phi(ip0)*factor1
                if(vmemb.ne.0.0)then
                   zc=(kg-1)*dcel
                   if(zc.gt.zmemb2) phif=phif-vmemb*factor1
                endif
                if(Qnonlinear) then
                   pfactor=bulk_rho*dcel3*exp(-phif)
                   nfactor=bulk_rho*dcel3*exp(+phif)
                elseif(Qpartlinear) then
                   if(phif.gt.0.0) then
                      pfactor=bulk_rho*dcel3*exp(-phif)
                      nfactor=bulk_rho*dcel3*(1.0 + phif)
                   else
                      pfactor=bulk_rho*dcel3*(1.0 - phif)
                      nfactor=bulk_rho*dcel3*exp(+phif)
                   endif
                else
                   pfactor=bulk_rho*dcel3*(1.0 - phif)
                   nfactor=bulk_rho*dcel3*(1.0 + phif)
                endif
                pion_num=pion_num+pfactor
                nion_num=nion_num+nfactor
                if(zz.ge.0.0) then
                   pion_numzp=pion_numzp+pfactor
                   nion_numzp=nion_numzp+nfactor
                else
                   pion_numzn=pion_numzn+pfactor
                   nion_numzn=nion_numzn+nfactor
                endif
             endif
          enddo
       enddo
    enddo
    !
    if(prnlev.ge.2) write(outu,'(3x,a)')
    if(prnlev.ge.2) write(outu,'(3x,a,f13.5,a)') &
         'Ion accessible volume                   :',volume, &
         ' [Angs**3]'
    if(prnlev.ge.2) write(outu,'(3x,a,f13.5,a)') &
         'Bulk density                            :',bulk_rho, &
         ' [unit charge]/[Angs**3]'
    if(prnlev.ge.2) write(outu,'(3x,a,f13.5,a)') &
         'Bulk number of counter ions             :',bulk_num, &
         ' [unit charge]'
    if(prnlev.ge.2) write(outu,'(3x,a,f13.5,a)') &
         'Number of positive ions                 :',pion_num, &
         ' [unit charge]'
    if(prnlev.ge.2) write(outu,'(3x,a,f13.5,a)') &
         'Number of positive ions (Z > 0)         :',pion_numzp, &
         ' [unit charge]'
    if(prnlev.ge.2) write(outu,'(3x,a,f13.5,a)') &
         'Number of positive ions (Z < 0)         :',pion_numzn, &
         ' [unit charge]'
    if(prnlev.ge.2) write(outu,'(3x,a,f13.5,a)') &
         'Number of negative ions                 :',nion_num, &
         ' [unit charge]'
    if(prnlev.ge.2) write(outu,'(3x,a,f13.5,a)') &
         'Number of negative ions (Z > 0)         :',nion_numzp, &
         ' [unit charge]'
    if(prnlev.ge.2) write(outu,'(3x,a,f13.5,a)') &
         'Number of negative ions (Z < 0)         :',nion_numzn, &
         ' [unit charge]'


    !  Counter Ion Distributions
    if(prnlev.ge.2) write(outu,'(3x,a)')
    if(prnlev.ge.2) write(outu,'(3x,2a)')  &
         'Integrated number of ions along Z : ', &
         '[unit charge]'
    if(prnlev.ge.2) write(outu,'(3x,a)')
    if(prnlev.ge.2) write(outu,'(11x,a,7x,a,7x,a,7x,a)') &
         'Z','AREA','+ ION','- ION'

    do kg=1,nclz
       zz=(kg-1)*dcel-tranz+zbcen
       pion_num=0.0D0
       nion_num=0.0D0
       area=0.0
       do ig=1,nclx
          iii=(ig-1)*ncyz+kg
          do jg=1,ncly
             ip0=iii+(jg-1)*nclz
             if(mcden(ip0).ne.0.0) then
                area=area+1.0
                phif=phi(ip0)*factor1
                if(vmemb.ne.0.0)then
                   zc=(kg-1)*dcel
                   if(zc.gt.zmemb2) phif=phif-vmemb*factor1
                endif
                if(Qnonlinear) then
                   pfactor=bulk_rho*dcel3*exp(-phif)
                   nfactor=bulk_rho*dcel3*exp(+phif)
                elseif(Qpartlinear) then
                   if(phif.gt.0.0) then
                      pfactor=bulk_rho*dcel3*exp(-phif)
                      nfactor=bulk_rho*dcel3*(1.0 + phif)
                   else
                      pfactor=bulk_rho*dcel3*(1.0 - phif)
                      nfactor=bulk_rho*dcel3*exp(+phif)
                   endif
                else
                   pfactor=bulk_rho*dcel3*(1.0 - phif)
                   nfactor=bulk_rho*dcel3*(1.0 + phif)
                endif
                pion_num=pion_num+pfactor
                nion_num=nion_num+nfactor
             endif
          enddo
       enddo
       if(prnlev.ge.2) write(outu,'(3x,2f10.3,2x,2(e10.4,2x))') &
            zz,area*dcel2,pion_num,nion_num
    enddo

    if(prnlev.ge.2) write(outu,'(3x,a)')
    if(prnlev.ge.2) write(outu,'(3x,2a)') &
         'Density Profile based on ion-accessible area along Z : ', &
         '[unit charge]/[A^3]'
    if(prnlev.ge.2) write(outu,'(3x,a)')
    if(prnlev.ge.2) write(outu,'(11x,a,7x,a,7x,a,7x,a)') &
         'Z','AREA','+ ION','- ION'

    do kg=1,nclz
       zz=(kg-1)*dcel-tranz+zbcen
       pion_num=0.0D0
       nion_num=0.0D0
       area=0.0
       do ig=1,nclx
          iii=(ig-1)*ncyz+kg
          do jg=1,ncly
             ip0=iii+(jg-1)*nclz
             if(mcden(ip0).ne.0.0) then
                area=area+1.0
                phif=phi(ip0)*factor1
                if(vmemb.ne.0.0)then
                   zc=(kg-1)*dcel
                   if(zc.gt.zmemb2) phif=phif-vmemb*factor1
                endif
                if(Qnonlinear) then
                   pfactor=bulk_rho*dcel3*exp(-phif)
                   nfactor=bulk_rho*dcel3*exp(+phif)
                elseif(Qpartlinear) then
                   if(phif.gt.0.0) then
                      pfactor=bulk_rho*dcel3*exp(-phif)
                      nfactor=bulk_rho*dcel3*(1.0 + phif)
                   else
                      pfactor=bulk_rho*dcel3*(1.0 - phif)
                      nfactor=bulk_rho*dcel3*exp(+phif)
                   endif
                else
                   pfactor=bulk_rho*dcel3*(1.0 - phif)
                   nfactor=bulk_rho*dcel3*(1.0 + phif)
                endif
                pion_num=pion_num+pfactor
                nion_num=nion_num+nfactor
             endif
          enddo
       enddo
       if(prnlev.ge.2) write(outu,'(3x,2f10.3,2x,2(e10.4,2x))') &
            zz,area*dcel2, &
            pion_num/(area*dcel3+rsmall),nion_num/(area*dcel3+rsmall)
    enddo

    if(prnlev.ge.2) write(outu,'(3x,a)')
    if(prnlev.ge.2) write(outu,'(3x,2a)') &
         'Density Profile based on system area along Z : ', &
         '[unit charge]/[A^3]'
    if(prnlev.ge.2) write(outu,'(3x,a)')
    if(prnlev.ge.2) write(outu,'(11x,a,7x,a,7x,a,7x,a)') &
         'Z','AREA','+ ION','- ION'

    area=nclx*ncly
    do kg=1,nclz
       zz=(kg-1)*dcel-tranz+zbcen
       pion_num=0.0D0
       nion_num=0.0D0
       do ig=1,nclx
          iii=(ig-1)*ncyz+kg
          do jg=1,ncly
             ip0=iii+(jg-1)*nclz
             if(mcden(ip0).ne.0.0) then
                phif=phi(ip0)*factor1
                if(vmemb.ne.0.0)then
                   zc=(kg-1)*dcel
                   if(zc.gt.zmemb2) phif=phif-vmemb*factor1
                endif
                if(Qnonlinear) then
                   pfactor=bulk_rho*dcel3*exp(-phif)
                   nfactor=bulk_rho*dcel3*exp(+phif)
                elseif(Qpartlinear) then
                   if(phif.gt.0.0) then
                      pfactor=bulk_rho*dcel3*exp(-phif)
                      nfactor=bulk_rho*dcel3*(1.0 + phif)
                   else
                      pfactor=bulk_rho*dcel3*(1.0 - phif)
                      nfactor=bulk_rho*dcel3*exp(+phif)
                   endif
                else
                   pfactor=bulk_rho*dcel3*(1.0 - phif)
                   nfactor=bulk_rho*dcel3*(1.0 + phif)
                endif
                pion_num=pion_num+pfactor
                nion_num=nion_num+nfactor
             endif
          enddo
       enddo
       if(prnlev.ge.2) write(outu,'(3x,2f10.3,2x,2(e10.4,2x))') &
            zz,area*dcel2, &
            pion_num/(area*dcel3+rsmall),nion_num/(area*dcel3+rsmall)
    enddo
    !
    RETURN
  END SUBROUTINE COUNTERION
  !
  SUBROUTINE RFORCE(NTPRP,LSTPRP,X,Y,Z,CG,NCLX,NCLY,NCLZ,DCEL, &
       PHI1,PHI2,TRANX,TRANY,TRANZ, &
       XBCEN,YBCEN,ZBCEN,FACTOR, &
       RXNFX,RXNFY,RXNFZ,ENSOLV,QPRIN,QBSPL)
    !-----------------------------------------------------------------------
    !     This subroutine computes the electrostatic force resulting from the
    !     reaction field generated by the solvent at each atom.
    !
    !     Reaction field forces - RXNF
    !
    use chm_kinds
    use consta
    use number
    use stream
    use dimens_fcm
    implicit none
    !
    integer ntprp, lstprp(*)
    real(chm_real)  x(*),y(*),z(*),cg(*)
    real(chm_real)  rxnfx(*),rxnfy(*),rxnfz(*)
    integer NCLX,NCLY,NCLZ
    real(chm_real)  dcel,tranx,trany,tranz,xbcen,ybcen,zbcen
    real(chm_real4)  phi1(*),phi2(*)
    real(chm_real)  FACTOR,ENSOLV
    logical QPRIN,QBSPL
    !   local
    integer ncyz,il,i,ix,iy,iz,n1,in1,n2,in2,n3,in3
    real(chm_real)  chi,xi,yi,zi,ai,bi,ci,fi
    real(chm_real)  enet1, enelp1, enet2, enelp2
    real(chm_real)  aisign,bisign,cisign,prefac
    real(chm_real)  xgrad,ygrad,zgrad
    ! B-spline
    integer jx1,jx2,jy1,jy2,jz1,jz2,k,l,m,ipx,ipy,ipz,nfil
    real(chm_real)  xc,yc,zc
    !
    ncyz=ncly*nclz
    enelp1=zero
    enelp2=zero
    !
    !
    !==============================================================================
    IF(QBSPL) THEN
       !
       !     Main loop by atoms
       !
       do il=1,ntprp
          i=lstprp(il)
          chi=cg(i)
          rxnfx(i)=zero
          rxnfy(i)=zero
          rxnfz(i)=zero
          IF(CHI.EQ.0.0) GOTO 490
          xi=x(i)+tranx-xbcen
          yi=y(i)+trany-ybcen
          zi=z(i)+tranz-zbcen
          nfil=1
          ix=int(xi/dcel)+1
          iy=int(yi/dcel)+1
          iz=int(zi/dcel)+1
          jx1=ix-nfil
          if(jx1.lt.1)jx1=1
          jx2=ix+nfil+1
          if(jx2.gt.nclx)jx2=nclx
          jy1=iy-nfil
          if(jy1.lt.1)jy1=1
          jy2=iy+nfil+1
          if(jy2.gt.ncly)jy2=ncly
          jz1=iz-nfil
          if(jz1.lt.1)jz1=1
          jz2=iz+nfil+1
          if(jz2.gt.nclz)jz2=nclz

          DO K=jx1,jx2
             IPX=(K-1)*NCyz
             XC=(K-1)*DCEL
             !
             DO L=jy1,jy2
                IPY=(L-1)*NCLz
                YC=(L-1)*DCEL
                !
                DO M=jz1,jz2
                   IPZ=M+IPY+IPX
                   ZC=(M-1)*DCEL

                   ai=1.5-(xi-xc)/dcel
                   bi=1.5-(yi-yc)/dcel
                   ci=1.5-(zi-zc)/dcel
                   fi=M3(ai)*M3(bi)*M3(ci)
                   !
                   !     Reaction Field Force  !!! Be careful of the sign (-1/dcel)
                   !
                   prefac=(phi1(ipz)-phi2(ipz))*CCELEC*chi/dcel
                   RXNFx(i)=RXNFx(i) + DM3(ai)*M3(bi)*M3(ci)*prefac
                   RXNFy(i)=RXNFy(i) + M3(ai)*DM3(bi)*M3(ci)*prefac
                   RXNFz(i)=RXNFz(i) + M3(ai)*M3(bi)*DM3(ci)*prefac
                   !
                   !     Electrostatic Energy
                   !
                   enet1=FACTOR*fi*chi*phi1(ipz)*CCELEC
                   enet2=FACTOR*fi*chi*phi2(ipz)*CCELEC
                   enelp1=enelp1+enet1
                   enelp2=enelp2+enet2

                enddo
             enddo
          enddo
490       continue
       enddo
       !
    ELSE
       !
       !
       !     Main loop by atoms
       !
       do il=1,ntprp
          i=lstprp(il)
          chi=cg(i)
          rxnfx(i)=zero
          rxnfy(i)=zero
          rxnfz(i)=zero
          IF(CHI.EQ.0.0) GOTO 500
          xi=x(i)+tranx-xbcen
          yi=y(i)+trany-ybcen
          zi=z(i)+tranz-zbcen
          ix=int(xi/dcel)+1
          iy=int(yi/dcel)+1
          iz=int(zi/dcel)+1
          !
          !     Atom charge distribution by 8 adjacent grid points
          !
          do n1=1,2
             in1=ix+n1-1
             ai=xi-(in1-1)*dcel
             aisign=sign(one,ai)
             ai=1.-abs(ai)/dcel
             in1=(in1-1)*ncyz

             do n2=1,2
                in2=iy+n2-1
                bi=yi-(in2-1)*dcel
                bisign=sign(one,bi)
                bi=1. - abs(bi)/dcel
                in2=(in2-1)*nclz

                do n3=1,2
                   in3=iz+n3-1
                   ci=zi-(in3-1)*dcel
                   cisign=sign(one,ci)
                   ci=1. - abs(ci)/dcel
                   fi=ai*bi*ci
                   in3=in1+in2+in3
                   !
                   prefac=(phi1(in3)-phi2(in3))*CCELEC*chi/dcel
                   !
                   if((ai.lt.(one-rsmall)).and.(ai.gt.rsmall)) then
                      RXNFx(i)=RXNFx(i)+aisign*bi*ci*prefac
                   endif
                   !
                   if((bi.lt.(one-rsmall)).and.(bi.gt.rsmall)) then
                      RXNFy(i)=RXNFy(i)+bisign*ai*ci*prefac
                   endif
                   !
                   if((ci.lt.(one-rsmall)).and.(ci.gt.rsmall)) then
                      RXNFz(i)=RXNFz(i)+cisign*ai*bi*prefac
                   endif
                   !
                   !     Electrostatic Energy
                   !
                   enet1=FACTOR*fi*chi*phi1(in3)*CCELEC
                   enet2=FACTOR*fi*chi*phi2(in3)*CCELEC
                   enelp1=enelp1+enet1
                   enelp2=enelp2+enet2

                enddo
             enddo
          enddo
500       continue
       enddo
       !
    ENDIF
    !==============================================================================
    !
    ENSOLV = ENELP1-ENELP2
    !
    IF(.not.QPRIN) RETURN
    !
    if(prnlev.ge.2) write(outu,'(/,3X,A,F13.5,A)') &
         'The Free Energy of Charging in Solvent  = ', &
         ENELP1,' [KCAL/MOL]'
    !
    if(prnlev.ge.2) write(outu,'(3X,A,F13.5,A)') &
         'The Free Energy of Charging in vacuum   = ', &
         ENELP2,' [KCAL/MOL]'
    !
    if(prnlev.ge.2) write(outu,'(3X,A,F13.5,A)') &
         'The Electrostatic Solvation Free Energy = ', &
         ENSOLV,' [KCAL/MOL]'
    !
    IF(PRNLEV.GT.6) THEN
       WRITE(outu,'(/,3x,A)') &
            'The Reaction Field Forces [KCAL/MOL/A] at Each Atom'
       Write(outu,'(3x,a)') '# ATOM       Fx        Fy        Fz'
       do il=1,ntprp
          i=lstprp(il)
          write(outu,'(3x,i5,4x,3f10.5)') i,rxnfx(i),rxnfy(i),rxnfz(i)
       enddo
    ENDIF
    !
    RETURN
  END SUBROUTINE RFORCE
  !
  SUBROUTINE RFORCE2(NTPRP,LSTPRP,X,Y,Z,CG,NCLX,NCLY,NCLZ,DCEL, &
       PHI1,TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN,FACTOR, &
       RXNFX,RXNFY,RXNFZ,ENSOLV,QBSPL &
#if KEY_GCMC==1
       ,GCMCON  & 
#endif
       )
    !-----------------------------------------------------------------------
    !     This subroutine computes the electrostatic force resulting from the
    !     reaction field generated by the solvent at each atom.
    !
    !     Reaction field forces - RXNF
    !
    use chm_kinds
    use consta
    use number
    use stream
    implicit none
    !
    integer ntprp, lstprp(*)
    real(chm_real)  x(*),y(*),z(*),cg(*)
    real(chm_real)  rxnfx(*),rxnfy(*),rxnfz(*)
    integer NCLX,NCLY,NCLZ
    real(chm_real)  dcel,tranx,trany,tranz,xbcen,ybcen,zbcen
    real(chm_real4)  phi1(*)
    real(chm_real)  FACTOR,ENSOLV
    logical QPRIN,QBSPL
#if KEY_GCMC==1
    LOGICAL GCMCON(:) 
#endif
    !   local
    integer ncyz,il,i,ix,iy,iz,n1,in1,n2,in2,n3,in3
    real(chm_real)  chi,xi,yi,zi,ai,bi,ci,fi
    real(chm_real)  enelp1
    real(chm_real)  aisign,bisign,cisign,prefac
    real(chm_real)  xgrad,ygrad,zgrad
    ! B-spline
    integer jx1,jx2,jy1,jy2,jz1,jz2,k,l,m,ipx,ipy,ipz,nfil
    real(chm_real)  xc,yc,zc
    !
    ncyz=ncly*nclz
    enelp1=zero
    !
    !
    !==============================================================================
    IF(QBSPL) THEN
       !
       !     Main loop by atoms
       !
       do il=1,ntprp
          i=lstprp(il)
#if KEY_GCMC==1
          IF (.NOT. GCMCON(I)) GOTO 490 
#endif
          chi=cg(i)
          rxnfx(i)=zero
          rxnfy(i)=zero
          rxnfz(i)=zero
          IF(CHI.EQ.0.0) GOTO 490
          xi=x(i)+tranx-xbcen
          yi=y(i)+trany-ybcen
          zi=z(i)+tranz-zbcen
          nfil=1
          ix=int(xi/dcel)+1
          iy=int(yi/dcel)+1
          iz=int(zi/dcel)+1
          jx1=ix-nfil
          if(jx1.lt.1)jx1=1
          jx2=ix+nfil+1
          if(jx2.gt.nclx)jx2=nclx
          jy1=iy-nfil
          if(jy1.lt.1)jy1=1
          jy2=iy+nfil+1
          if(jy2.gt.ncly)jy2=ncly
          jz1=iz-nfil
          if(jz1.lt.1)jz1=1
          jz2=iz+nfil+1
          if(jz2.gt.nclz)jz2=nclz

          DO K=jx1,jx2
             IPX=(K-1)*NCyz
             XC=(K-1)*DCEL
             !
             DO L=jy1,jy2
                IPY=(L-1)*NCLz
                YC=(L-1)*DCEL
                !
                DO M=jz1,jz2
                   IPZ=M+IPY+IPX
                   ZC=(M-1)*DCEL

                   ai=1.5-(xi-xc)/dcel
                   bi=1.5-(yi-yc)/dcel
                   ci=1.5-(zi-zc)/dcel
                   fi=M3(ai)*M3(bi)*M3(ci)
                   !
                   !     Reaction Field Force  !!! Be careful of the sign (-1/dcel)
                   !
                   prefac=phi1(ipz)*CCELEC*chi/dcel
                   RXNFx(i)=RXNFx(i) + DM3(ai)*M3(bi)*M3(ci)*prefac
                   RXNFy(i)=RXNFy(i) + M3(ai)*DM3(bi)*M3(ci)*prefac
                   RXNFz(i)=RXNFz(i) + M3(ai)*M3(bi)*DM3(ci)*prefac
                   !
                   !     Electrostatic Energy
                   !
                   enelp1=enelp1+FACTOR*fi*chi*phi1(ipz)*CCELEC

                enddo
             enddo
          enddo
490       continue
       enddo
       !
    ELSE
       !
       !
       !     Main loop by atoms
       !
       do il=1,ntprp
          i=lstprp(il)
#if KEY_GCMC==1
          IF (.NOT. GCMCON(I)) GOTO 500 
#endif
          chi=cg(i)
          rxnfx(i)=zero
          rxnfy(i)=zero
          rxnfz(i)=zero
          IF(CHI.EQ.0.0) GOTO 500
          xi=x(i)+tranx-xbcen
          yi=y(i)+trany-ybcen
          zi=z(i)+tranz-zbcen
          ix=int(xi/dcel)+1
          iy=int(yi/dcel)+1
          iz=int(zi/dcel)+1
          !
          !     Atom charge distribution by 8 adjacent grid points
          !
          do n1=1,2
             in1=ix+n1-1
             ai=xi-(in1-1)*dcel
             aisign=sign(one,ai)
             ai=1.-abs(ai)/dcel
             in1=(in1-1)*ncyz

             do n2=1,2
                in2=iy+n2-1
                bi=yi-(in2-1)*dcel
                bisign=sign(one,bi)
                bi=1. - abs(bi)/dcel
                in2=(in2-1)*nclz

                do n3=1,2
                   in3=iz+n3-1
                   ci=zi-(in3-1)*dcel
                   cisign=sign(one,ci)
                   ci=1. - abs(ci)/dcel
                   fi=ai*bi*ci
                   in3=in1+in2+in3
                   !
                   prefac=phi1(in3)*CCELEC*chi/dcel
                   !
                   if((ai.lt.(one-rsmall)).and.(ai.gt.rsmall)) then
                      RXNFx(i)=RXNFx(i)+aisign*bi*ci*prefac
                   endif
                   !
                   if((bi.lt.(one-rsmall)).and.(bi.gt.rsmall)) then
                      RXNFy(i)=RXNFy(i)+bisign*ai*ci*prefac
                   endif
                   !
                   if((ci.lt.(one-rsmall)).and.(ci.gt.rsmall)) then
                      RXNFz(i)=RXNFz(i)+cisign*ai*bi*prefac
                   endif
                   !
                   !     Electrostatic Energy
                   !
                   enelp1=enelp1+FACTOR*fi*chi*phi1(in3)*CCELEC

                enddo
             enddo
          enddo
500       continue
       enddo
       !
    ENDIF
    !==============================================================================
    !
    ENSOLV = ENELP1
    !
    RETURN
  END SUBROUTINE RFORCE2
  !
  SUBROUTINE BFORCE(NTPRP,LSTPRP,X,Y,Z,PBRAD,EPSW,EPSP, &
       NCLX,NCLY,NCLZ,DCEL,PHI1,RMAYX,RMAYY,RMAYZ,SWIN, &
       TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
       DBFX,DBFY,DBFZ,IBFX,IBFY,IBFZ, &
       RMAY,KAPPA2,NPFX,NPFY,NPFZ,STCO,Enp,QPRIN &
       ! XIAO_QC_UW0609
#if KEY_SCCDFTB==1
       ,QQSCCTB   &               
#endif
       )
    !-----------------------------------------------------------------------
    !     This subroutine computes electrostatic solvation forces resulting
    !     from the dielectric bounary and ionic boundary.
    !     And nonpolar solvation forces also are calculated, if necessary.
    !     Dielectric Boundary Force - DBF
    !     Ionic Boundary Force - IBF
    !     NonPolar solvation Force - NPF
    !     van der Waals surface

    use chm_kinds
    use consta
    use number
    use stream
    use gamess_fcm, only: qmused_sccdftb

    implicit none
    integer ntprp, lstprp(*)
    real(chm_real)  X(*),Y(*),Z(*),PBRAD(*)
    REAL(CHM_REAL4)  RMAYX(*),RMAYY(*),RMAYZ(*),RMAY(*)
    real(chm_real)  DBFX(*),DBFY(*),DBFZ(*)
    real(chm_real)  NPFX(*),NPFY(*),NPFZ(*)
    real(chm_real)  IBFX(*),IBFY(*),IBFZ(*)
    INTEGER NCLX,NCLY,NCLZ
    real(chm_real)  DCEL,TRANX,TRANY,TRANZ,EPSW,EPSP,STCO,KAPPA2
    real(chm_real)  XBCEN,YBCEN,ZBCEN
    REAL(CHM_REAL4)  PHI1(*)
    LOGICAL QPRIN
#if KEY_SCCDFTB==1
    logical QQSCCTB           
#endif
    !   local
    real(chm_real)    xc,yc,zc,xi,yi,zi,bisq
    INTEGER   IL,K,L,M,IPX,IPY,IPZ,NCyz,I,nfil
    INTEGER   ix,iy,iz,jx1,jx2,jy1,jy2,jz1,jz2
    !
    real(chm_real)    drp,drm,dxp,dxm,dyp,dym,dzp,dzm,dx,dy,dz
    real(chm_real)    rxp,rxm,ryp,rym,rzp,rzm,swin,bms,bps
    real(chm_real)    prefac1,prefac2,prefac3,prefac4
    real(chm_real)    Enp,depsx,depsy,depsz,moeps
    real(chm_real)    dcel2,rij,dr
    real(chm_real4)    kappab
    !
    !
    ENP=ZERO
    NCyz=NCLy*NCLz
    DCEL2=DCEL*DCEL
    KAPPAB=KAPPA2*DCEL2
    !
    !     Nonpolar Part - the surface of the solute for Solvation Free Energy
#if KEY_SCCDFTB==1
    if ((.not. QQSCCTB) .or. (.not. qmused_sccdftb)) then
#endif 

       IF(STCO.GT.ZERO) THEN
          do k=2,nclx-1
             ipx=(k-1)*ncyz
             do l=2,ncly-1
                ipy=(l-1)*nclz
                do m=2,nclz-1
                   ipz=ipx+ipy+m

                   depsx=(rmayx(ipz)-rmayx(ipz-ncyz))/(epsw-one)
                   depsy=(rmayy(ipz)-rmayy(ipz-nclz))/(epsw-one)
                   depsz=(rmayz(ipz)-rmayz(ipz-1))/(epsw-one)
                   moeps=sqrt(depsx**2+depsy**2+depsz**2)
                   Enp=Enp+stco*moeps*dcel2

                enddo
             enddo
          enddo
       ENDIF
#if KEY_SCCDFTB==1
    ENDIF
#endif 
    !
    DO IL=1,NTPRP
       I=LSTPRP(IL)
       XI=X(I)+TRANX-xbcen
       YI=Y(I)+TRANY-ybcen
       ZI=Z(I)+TRANZ-zbcen
       BISQ=PBRAD(I)
       NFIL=INT((BISQ+swin)/dcel)+2
       IX=INT(XI/DCEL)+1
       IY=INT(YI/DCEL)+1
       IZ=INT(ZI/DCEL)+1
       DBFX(I)=ZERO
       DBFY(I)=ZERO
       DBFZ(I)=ZERO
       IBFX(I)=ZERO
       IBFY(I)=ZERO
       IBFZ(I)=ZERO
       NPFX(I)=ZERO
       NPFY(I)=ZERO
       NPFZ(I)=ZERO
       bms=bisq-swin
       bps=bisq+swin
       !***
       if(pbrad(i).ne.ZERO) then
          !
          JX1=IX-NFIL
          IF(JX1.LT.1)JX1=1
          JX2=IX+NFIL+1
          IF(JX2.GT.NCLx)JX2=NCLx
          JY1=IY-NFIL
          IF(JY1.LT.1)JY1=1
          JY2=IY+NFIL+1
          IF(JY2.GT.NCLy)JY2=NCLy
          JZ1=IZ-NFIL
          IF(JZ1.LT.1)JZ1=1
          JZ2=IZ+NFIL+1
          IF(JZ2.GT.NCLz)JZ2=NCLz
          !
          DO K=jx1,jx2
             IPX=(K-1)*NCyz
             XC=(K-1)*DCEL
             !
             DO L=jy1,jy2
                IPY=(L-1)*NCLz
                YC=(L-1)*DCEL
                !
                DO M=jz1,jz2
                   IPZ=M+IPY+IPX
                   ZC=(M-1)*DCEL
                   !
                   depsx=rmayx(ipz)-rmayx(ipz-ncyz)
                   depsy=rmayy(ipz)-rmayy(ipz-nclz)
                   depsz=rmayz(ipz)-rmayz(ipz-1)
                   moeps=sqrt(depsx**2+depsy**2+depsz**2)
                   !
                   rxp=sqrt((xc+dcel/2.-xi)**2+(yc-yi)**2+(zc-zi)**2)
                   rxm=sqrt((xc-dcel/2.-xi)**2+(yc-yi)**2+(zc-zi)**2)
                   ryp=sqrt((xc-xi)**2+(yc+dcel/2.-yi)**2+(zc-zi)**2)
                   rym=sqrt((xc-xi)**2+(yc-dcel/2.-yi)**2+(zc-zi)**2)
                   rzp=sqrt((xc-xi)**2+(yc-yi)**2+(zc+dcel/2.-zi)**2)
                   rzm=sqrt((xc-xi)**2+(yc-yi)**2+(zc-dcel/2.-zi)**2)
                   !
                   dx=xc-xi
                   dy=yc-yi
                   dz=zc-zi
                   !
                   ! x-direction
                   drp=rxp-bms
                   drm=rxm-bms
                   dxp=xc+dcel/2.-xi
                   dxm=xc-dcel/2.-xi
                   !
                   !     (rxm.lt.bps) : to consider only atom j among other atoms in a molecule
                   !
                   if((rmayx(ipz-ncyz).ne.epsp).and. &
                        (rmayx(ipz-ncyz).ne.epsw).and.(rxm.lt.bps)) then
                      prefac1= depssf(drm,swin)*(rmayx(ipz-ncyz)-epsp) &
                           /rxm/epssf(drm,swin)
                      ! DBF part
                      prefac2= CCELEC/(eight*pi)*dcel*prefac1* &
                           phi1(ipz)*(phi1(ipz-ncyz)-phi1(ipz))
                      DBFx(i)=DBFx(i)+ prefac2*dxm
                      DBFy(i)=DBFy(i)+ prefac2*dy
                      DBFz(i)=DBFz(i)+ prefac2*dz
                      ! NPF part
#if KEY_SCCDFTB==1
                      IF (qmused_sccdftb .and. QQSCCTB) GOTO 301
#endif 
                      if((moeps.gt.1.0e-8).and.(STCO.GT.ZERO)) then
                         prefac3= stco*dcel2*depsx/moeps* &
                              prefac1/(epsw-epsp)
                         NPFx(i)=NPFx(i)- prefac3*dxm
                         NPFy(i)=NPFy(i)- prefac3*dy
                         NPFz(i)=NPFz(i)- prefac3*dz
                      endif
#if KEY_SCCDFTB==1
301                   CONTINUE            
#endif 
                   endif
                   !
                   if((rmayx(ipz).ne.epsp).and. &
                        (rmayx(ipz).ne.epsw).and.(rxp.lt.bps)) then
                      prefac1= depssf(drp,swin)*(rmayx(ipz)-epsp) &
                           /rxp/epssf(drp,swin)
                      ! DBF part
                      prefac2= CCELEC/(eight*pi)*dcel*prefac1* &
                           phi1(ipz)*(phi1(ipz+ncyz)-phi1(ipz))
                      DBFx(i)=DBFx(i)+ prefac2*dxp
                      DBFy(i)=DBFy(i)+ prefac2*dy
                      DBFz(i)=DBFz(i)+ prefac2*dz
                      ! NPF part
#if KEY_SCCDFTB==1
                      IF (qmused_sccdftb .and. QQSCCTB) GOTO 302
#endif 
                      if((moeps.gt.1.0e-8).and.(STCO.GT.ZERO)) then
                         prefac3= stco*dcel2*depsx/moeps* &
                              prefac1/(epsw-epsp)
                         NPFx(i)=NPFx(i)+ prefac3*dxp
                         NPFy(i)=NPFy(i)+ prefac3*dy
                         NPFz(i)=NPFz(i)+ prefac3*dz
                      endif
#if KEY_SCCDFTB==1
302                   CONTINUE
#endif 
                   endif
                   !
                   ! y-direction
                   drp=ryp-bms
                   drm=rym-bms
                   dyp=yc+dcel/2.-yi
                   dym=yc-dcel/2.-yi
                   !
                   if((rmayy(ipz-nclz).ne.epsp).and. &
                        (rmayy(ipz-nclz).ne.epsw).and.(rym.lt.bps)) then
                      prefac1= depssf(drm,swin)*(rmayy(ipz-nclz)-epsp) &
                           /rym/epssf(drm,swin)
                      ! DBF part
                      prefac2=  CCELEC/(eight*pi)*dcel*prefac1* &
                           phi1(ipz)*(phi1(ipz-nclz)-phi1(ipz))
                      DBFx(i)=DBFx(i)+ prefac2*dx
                      DBFy(i)=DBFy(i)+ prefac2*dym
                      DBFz(i)=DBFz(i)+ prefac2*dz
                      ! NPF part
#if KEY_SCCDFTB==1
                      IF (qmused_sccdftb .and. QQSCCTB) GOTO 303
#endif 
                      if((moeps.gt.1.0e-8).and.(STCO.GT.ZERO)) then
                         prefac3= stco*dcel2*depsy/moeps* &
                              prefac1/(epsw-epsp)
                         NPFx(i)=NPFx(i)- prefac3*dx
                         NPFy(i)=NPFy(i)- prefac3*dym
                         NPFz(i)=NPFz(i)- prefac3*dz
                      endif
#if KEY_SCCDFTB==1
303                   CONTINUE
#endif 
                   endif
                   !
                   if((rmayy(ipz).ne.epsp).and. &
                        (rmayy(ipz).ne.epsw).and.(ryp.lt.bps)) then
                      prefac1= depssf(drp,swin)*(rmayy(ipz)-epsp) &
                           /ryp/epssf(drp,swin)
                      ! DBF part
                      prefac2= CCELEC/(eight*pi)*dcel*prefac1* &
                           phi1(ipz)*(phi1(ipz+nclz)-phi1(ipz))
                      DBFx(i)=DBFx(i)+ prefac2*dx
                      DBFy(i)=DBFy(i)+ prefac2*dyp
                      DBFz(i)=DBFz(i)+ prefac2*dz
                      ! NPF part
#if KEY_SCCDFTB==1
                      IF (qmused_sccdftb .and. QQSCCTB) GOTO 304
#endif 
                      if((moeps.gt.1.0e-8).and.(STCO.GT.ZERO)) then
                         prefac3= stco*dcel2*depsy/moeps* &
                              prefac1/(epsw-epsp)
                         NPFx(i)=NPFx(i)+ prefac3*dx
                         NPFy(i)=NPFy(i)+ prefac3*dyp
                         NPFz(i)=NPFz(i)+ prefac3*dz
                      endif
#if KEY_SCCDFTB==1
304                   CONTINUE
#endif 
                   endif
                   !
                   ! z-direction
                   drp=rzp-bms
                   drm=rzm-bms
                   dzp=zc+dcel/2.-zi
                   dzm=zc-dcel/2.-zi
                   !
                   if((rmayz(ipz-1).ne.epsp).and. &
                        (rmayz(ipz-1).ne.epsw).and.(rzm.lt.bps)) then
                      prefac1= depssf(drm,swin)*(rmayz(ipz-1)-epsp) &
                           /rzm/epssf(drm,swin)
                      ! DBF part
                      prefac2= CCELEC/(eight*pi)*dcel*prefac1* &
                           phi1(ipz)*(phi1(ipz-1)-phi1(ipz))
                      DBFx(i)=DBFx(i)+ prefac2*dx
                      DBFy(i)=DBFy(i)+ prefac2*dy
                      DBFz(i)=DBFz(i)+ prefac2*dzm
                      ! NPF part
#if KEY_SCCDFTB==1
                      IF (qmused_sccdftb .and. QQSCCTB) GOTO 305
#endif 
                      if((moeps.gt.1.0e-8).and.(STCO.GT.ZERO)) then
                         prefac3= stco*dcel2*depsz/moeps* &
                              prefac1/(epsw-epsp)
                         NPFx(i)=NPFx(i)- prefac3*dx
                         NPFy(i)=NPFy(i)- prefac3*dy
                         NPFz(i)=NPFz(i)- prefac3*dzm
                      endif
#if KEY_SCCDFTB==1
305                   continue     
#endif
                   endif
                   !
                   if((rmayz(ipz).ne.epsp).and. &
                        (rmayz(ipz).ne.epsw).and.(rzp.lt.bps)) then
                      prefac1= depssf(drp,swin)*(rmayz(ipz)-epsp) &
                           /rzp/epssf(drp,swin)
                      ! DBF part
                      prefac2= CCELEC/(eight*pi)*dcel*prefac1* &
                           phi1(ipz)*(phi1(ipz+1)-phi1(ipz))
                      DBFx(i)=DBFx(i)+ prefac2*dx
                      DBFy(i)=DBFy(i)+ prefac2*dy
                      DBFz(i)=DBFz(i)+ prefac2*dzp
                      ! NPF part
#if KEY_SCCDFTB==1
                      if (qmused_sccdftb .and. qqscctb) goto 306     
#endif
                      if((moeps.gt.1.0e-8).and.(STCO.GT.ZERO)) then
                         prefac3= stco*dcel2*depsz/moeps* &
                              prefac1/(epsw-epsp)
                         NPFx(i)=NPFx(i)+ prefac3*dx
                         NPFy(i)=NPFy(i)+ prefac3*dy
                         NPFz(i)=NPFz(i)+ prefac3*dzp
                      endif
#if KEY_SCCDFTB==1
306                   continue                   
#endif
                   endif
                   !
                   ! IBF part
                   rij=sqrt((xc-xi)**2+(yc-yi)**2+(zc-zi)**2)
                   dr=rij-bms
                   !
                   if((rmay(ipz).lt.kappab).and. &
                        (rmay(ipz).gt.zero).and.(rij.lt.bps)) then
                      !wi,bug     $               (rmay(ipz).gt.zero).and.(dr.lt.bps)) then
                      prefac1= depssf(dr,swin)*rmay(ipz)/rij &
                           /epssf(dr,swin)
                      prefac4= CCELEC/(eight*pi)*phi1(ipz)**2*dcel &
                           *prefac1
                      IBFx(i)=IBFx(i)- prefac4*dx
                      IBFy(i)=IBFy(i)- prefac4*dy
                      IBFz(i)=IBFz(i)- prefac4*dz
                      !                     write(6,*) k,l,m,rmay(ipz),kappab,ibfx(i)
                   endif
                   !
                enddo
                !
             enddo
             !
          enddo
          !***
       endif
       !
    ENDDO
    !
    IF(.not.QPRIN) RETURN

#if KEY_SCCDFTB==1
    if (qmused_sccdftb .and. qqscctb) goto 307     
#endif
    !
    IF(STCO.GT.ZERO) THEN
       if(prnlev.ge.2) write(outu,'(/,3x,a,F13.5,a)') &
            'Molecular Surface = ', Enp/stco,' [A^2]'
       if(prnlev.ge.2) write(outu,'(3X,A,F13.5,A)') &
            'The Nonpolar Solvation Free Energy =',Enp,' [KCAL/MOL]'
    ENDIF
#if KEY_SCCDFTB==1
307 continue                   
#endif
    !
    IF(PRNLEV.GT.6) THEN
       if(prnlev.ge.2) write(outu,'(/,3x,A)') &
            'The Dielectric Boundary Forces [KCAL/MOL/A] at Each Atom'
       if(prnlev.ge.2) write(outu,'(3x,a)')   &
            '# ATOM       Fx        Fy        Fz'
       do il=1,ntprp
          i=lstprp(il)
          if(prnlev.ge.2) write(outu,'(3x,i5,4x,3f10.5)')  &
               i,dbfx(i),dbfy(i),dbfz(i)
       enddo

       IF(KAPPA2.GT.ZERO) THEN
          if(prnlev.ge.2) write(outu,'(/,3x,A)') &
               'The Ionic Boundary Forces [KCAL/MOL/A] at Each Atom'
          if(prnlev.ge.2) write(outu,'(3x,a)')   &
               '# ATOM       Fx        Fy        Fz'
          do il=1,ntprp
             i=lstprp(il)
             if(prnlev.ge.2) write(outu,'(3x,i5,4x,3f10.5)')  &
                  i,ibfx(i),ibfy(i),ibfz(i)
          enddo
       ENDIF

#if KEY_SCCDFTB==1
       if (qmused_sccdftb .and. qqscctb) goto 308     
#endif
       IF(STCO.GT.ZERO) THEN
          if(prnlev.ge.2) write(outu,'(/,3x,A)') &
               'The Nonpolar Surface Forces [KCAL/MOL/A] at Each Atom'
          if(prnlev.ge.2) write(outu,'(3x,a)')   &
               '# ATOM       Fx        Fy        Fz'
          do il=1,ntprp
             i=lstprp(il)
             if(prnlev.ge.2) write(outu,'(3x,i5,4x,3f10.5)')  &
                  i,npfx(i),npfy(i),npfz(i)
          enddo
       ENDIF
#if KEY_SCCDFTB==1
308    continue                   
#endif
       !
    ENDIF

    RETURN
  END SUBROUTINE BFORCE

  SUBROUTINE WRIGD0
    !-----------------------------------------------------------------------
    use chm_kinds
    use dimens_fcm
    use number
    use comand
    use consta
    use stream
    use string
    use memory
    use ctitla

    implicit none
    !
    REAL(CHM_REAL4) PRFTR
    INTEGER UNIT
    !
    UNIT    = GTRMI(COMLYN,COMLEN,'UNIT',OUTU)
    IF(PRNLEV.GE.2) WRITE(OUTU,'(3X,A,1X,I3)') 'Writing to unit ',UNIT

    PRFTR=1.

    IF(INDXA(COMLYN,COMLEN,'PHI ').GT.0)THEN
       IF(INDXA(COMLYN,COMLEN,'KCAL').GT.0)THEN
          PRFTR=CCELEC
          IF(PRNLEV.GE.2) WRITE(OUTU,100) &
               'Electrostatic potential in [KCAL/MOL]/[UNIT CHARGE]'
100       FORMAT(3X,A)
       ELSEIF (INDXA(COMLYN,COMLEN,'VOLT').GT.0)THEN
          PRFTR=14.40145905
          IF(PRNLEV.GE.2) WRITE(OUTU,100) &
               'Electrostatic potential in [VOLTS]'
       ELSE
          PRFTR=1.
          IF(PRNLEV.GE.2) WRITE(OUTU,100) &
               'Electrostatic potential in [UNIT CHARGE]/[ANGS]'
       ENDIF
       IF(INDXA(COMLYN,COMLEN,'CARD').GT.0)THEN
          CALL WRIGD1(UNIT,NCLX,NCLY,NCLZ,DCEL, &
               TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
               IPHIW,PRFTR)
       ELSE
          CALL SAVGD1(UNIT,NCLX,NCLY,NCLZ,DCEL,IPHIW,PRFTR, &
               XBCEN,YBCEN,ZBCEN,EPSW,EPSP,CONC,TMEMB,ZMEMB,EPSM)
       ENDIF

    ELSEIF(INDXA(COMLYN,COMLEN,'PHIX').GT.0)THEN
       IF(INDXA(COMLYN,COMLEN,'KCAL').GT.0)THEN
          PRFTR=CCELEC
          IF(PRNLEV.GE.2) WRITE(OUTU,100) &
               'Electrostatic potential in [KCAL/MOL]/[UNIT CHARGE]'
       ELSEIF (INDXA(COMLYN,COMLEN,'VOLT').GT.0)THEN
          PRFTR=14.40145905
          IF(PRNLEV.GE.2) WRITE(OUTU,100) &
               'Electrostatic potential in [VOLTS]'
       ELSE
          PRFTR=1.
          IF(PRNLEV.GE.2) WRITE(OUTU,100) &
               'Electrostatic potential in [UNIT CHARGE]/[ANGS]'
       ENDIF
       IF(INDXA(COMLYN,COMLEN,'CARD').GT.0)THEN
          !            CALL WRIGD1(UNIT,NCLX,NCLY,NCLZ,DCEL,
          !     $                  TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN,
          !     $                  IPHIX,PRFTR)
          CALL WRIGD3(UNIT,NCLX,NCLY,NCLZ,DCEL,IPHIX,PRFTR, &
               XBCEN,YBCEN,ZBCEN,EPSW,EPSP,CONC,TMEMB,ZMEMB,EPSM)
       ELSE
          CALL SAVGD1(UNIT,NCLX,NCLY,NCLZ,DCEL,IPHIX,PRFTR, &
               XBCEN,YBCEN,ZBCEN,EPSW,EPSP,CONC,TMEMB,ZMEMB,EPSM)
       ENDIF

    ELSEIF (INDXA(COMLYN,COMLEN,'FKAP').GT.0)THEN
       PRFTR=ONE/(DCEL*DCEL)
       IF(PRNLEV.GE.2) WRITE(OUTU,100)  &
            'Debye screening factor FKAPPA2 '
       IF(INDXA(COMLYN,COMLEN,'CARD').GT.0)THEN
          CALL WRIGD1(UNIT,NCLX,NCLY,NCLZ,DCEL, &
               TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
               MAY,PRFTR)
       ELSE
          PRFTR=1.    ! for the direct use of May array
          CALL SAVGD1(UNIT,NCLX,NCLY,NCLZ,DCEL,MAY,PRFTR, &
               XBCEN,YBCEN,ZBCEN,EPSW,EPSP,CONC,TMEMB,ZMEMB,EPSM)
       ENDIF

    ELSEIF (INDXA(COMLYN,COMLEN,'CHRG').GT.0)THEN
       PRFTR=DCEL/(TWO*TWOPI)
       IF(PRNLEV.GE.2) WRITE(OUTU,100) 'charges on the lattice '
       IF(INDXA(COMLYN,COMLEN,'CARD').GT.0)THEN
          CALL WRIGD1(UNIT,NCLX,NCLY,NCLZ,DCEL, &
               TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
               ICHC,PRFTR)
       ELSE
          CALL SAVGD1(UNIT,NCLX,NCLY,NCLZ,DCEL,ICHC,PRFTR, &
               XBCEN,YBCEN,ZBCEN,EPSW,EPSP,CONC,TMEMB,ZMEMB,EPSM)
       ENDIF

    ELSEIF (INDXA(COMLYN,COMLEN,'EPSX').GT.0)THEN
       IF(PRNLEV.GE.2) WRITE(OUTU,100) 'X set of dielectric function'
       IF(INDXA(COMLYN,COMLEN,'CARD').GT.0)THEN
          CALL WRIGD1(UNIT,NCLX,NCLY,NCLZ,DCEL, &
               TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
               MAYX,PRFTR)
       ELSE
          CALL SAVGD1(UNIT,NCLX,NCLY,NCLZ,DCEL,MAYX,PRFTR, &
               XBCEN,YBCEN,ZBCEN,EPSW,EPSP,CONC,TMEMB,ZMEMB,EPSM)
       ENDIF

    ELSEIF (INDXA(COMLYN,COMLEN,'EPSY').GT.0)THEN
       IF(PRNLEV.GE.2) WRITE(OUTU,100) 'Y set of dielectric function'
       IF(INDXA(COMLYN,COMLEN,'CARD').GT.0)THEN
          CALL WRIGD1(UNIT,NCLX,NCLY,NCLZ,DCEL, &
               TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
               MAYY,PRFTR)
       ELSE
          CALL SAVGD1(UNIT,NCLX,NCLY,NCLZ,DCEL,MAYY,PRFTR, &
               XBCEN,YBCEN,ZBCEN,EPSW,EPSP,CONC,TMEMB,ZMEMB,EPSM)
       ENDIF

    ELSEIF (INDXA(COMLYN,COMLEN,'EPSZ').GT.0)THEN
       IF(PRNLEV.GE.2) WRITE(OUTU,100) 'Z set of dielectric function'
       IF(INDXA(COMLYN,COMLEN,'CARD').GT.0)THEN
          CALL WRIGD1(UNIT,NCLX,NCLY,NCLZ,DCEL, &
               TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
               MAYZ,PRFTR)
       ELSE
          CALL SAVGD1(UNIT,NCLX,NCLY,NCLZ,DCEL,MAYZ,PRFTR, &
               XBCEN,YBCEN,ZBCEN,EPSW,EPSP,CONC,TMEMB,ZMEMB,EPSM)
       ENDIF

    ELSEIF (INDXA(COMLYN,COMLEN,'MIJ').GT.0)THEN
       IF(PRNLEV.GE.2) WRITE(OUTU,100) 'The MIJ matrix'
       IF(INDXA(COMLYN,COMLEN,'CARD').GT.0)THEN
          IF(QRECTBOX) THEN
             CALL WRIGD2(UNIT,XNPOL,YNPOL,ZNPOL,'RECTBOX ', &
                  LSTPX,LSTPY,LSTPZ,MIJ)
          ELSEIF(QSPHERE) THEN
             CALL WRIGD2(UNIT,NMPOL,0,0,'SPHERE  ', &
                  LSTPX,LSTPY,LSTPZ,MIJ)
          ELSEIF(QPROLATE) THEN
             CALL WRIGD2(UNIT,NMPOL,0,0,'PROLATE ', &
                  LSTPX,LSTPY,LSTPZ,MIJ)
          ELSEIF(QRFRECT) THEN
             CALL WRIGD2(UNIT,XNPOL,YNPOL,ZNPOL,'RECTBOX ', &
                  LSTPX,LSTPY,LSTPZ,MIJ)
          ELSEIF(QRFSPHE) THEN
             CALL WRIGD2(UNIT,NMPOL,0,0,'SPHERE  ', &
                  LSTPX,LSTPY,LSTPZ,MIJ)
          ENDIF
       ELSE
          IF(QRECTBOX) THEN
             CALL SAVGD2(UNIT,XNPOL,YNPOL,ZNPOL,'RECTBOX ', &
                  LSTPX,LSTPY,LSTPZ,MIJ)
          ELSEIF(QSPHERE) THEN
             CALL SAVGD2(UNIT,NMPOL,0,0,'SPHERE  ', &
                  LSTPX,LSTPY,LSTPZ,MIJ)
          ELSEIF(QPROLATE) THEN
             CALL SAVGD2(UNIT,NMPOL,0,0,'PROLATE ', &
                  LSTPX,LSTPY,LSTPZ,MIJ)
          ELSEIF(QRFRECT) THEN
             CALL SAVGD2(UNIT,XNPOL,YNPOL,ZNPOL,'RECTBOX ', &
                  LSTPX,LSTPY,LSTPZ,MIJ)
          ELSEIF(QRFSPHE) THEN
             CALL SAVGD2(UNIT,NMPOL,0,0,'SPHERE  ', &
                  LSTPX,LSTPY,LSTPZ,MIJ)
          ENDIF
       ENDIF

    ELSEIF (INDXA(COMLYN,COMLEN,'MMIJ').GT.0)THEN
       IF(PRNLEV.GE.2) WRITE(OUTU,100) 'The MMIJ matrix'
       IF(QPROLATE) THEN
          CALL SAVGD2(UNIT,NMPOL,0,0,'PROLATE ', &
               LSTPX,LSTPY,LSTPZ,MMIJ)
       ELSEIF(QRFRECT) THEN
          CALL SAVGD2(UNIT,XNPOL,YNPOL,ZNPOL,'RECTBOX ', &
               LSTPX,LSTPY,LSTPZ,MMIJ)
       ELSEIF(QRFSPHE) THEN
          CALL SAVGD2(UNIT,NMPOL,0,0,'SPHERE  ', &
               LSTPX,LSTPY,LSTPZ,MMIJ)
       ENDIF

    ELSEIF (INDXA(COMLYN,COMLEN,'TITL').GT.0)THEN
       CALL RDTITL(TITLEA,NTITLA,ISTRM,0)
       CALL WRTITL(TITLEA,NTITLA,UNIT,3)

    ELSEIF (INDXA(COMLYN,COMLEN,'HELP').GT.0)THEN
       IF(PRNLEV.GE.2) WRITE(OUTU,100) &
            'Syntax:     WRITE <quantity> [{CARD} <range>] '
       IF(PRNLEV.GE.2) WRITE(OUTU,100)  &
            'where <quantity> can be: '
       IF(PRNLEV.GE.2) WRITE(OUTU,100)  &
            'PHI     - electrostatic potential'
       IF(PRNLEV.GE.2) WRITE(OUTU,100)  &
            'PHIX    - static external field'
       IF(PRNLEV.GE.2) WRITE(OUTU,100)  &
            'FKAPPA2 - Debye screening factor'
       IF(PRNLEV.GE.2) WRITE(OUTU,100)  &
            'CHRG    - charges on the lattice'
       IF(PRNLEV.GE.2) WRITE(OUTU,100)  &
            'EPSX    - X sets of dielectric function'
       IF(PRNLEV.GE.2) WRITE(OUTU,100)  &
            'EPSY    - Y sets of dielectric function'
       IF(PRNLEV.GE.2) WRITE(OUTU,100)  &
            'EPSZ    - Z sets of dielectric function'
       IF(PRNLEV.GE.2) WRITE(OUTU,100) 'MIJ     - MIJ matrix'
       IF(PRNLEV.GE.2) WRITE(OUTU,100) 'MMIJ    - MMIJ matrix'
       IF(PRNLEV.GE.2) WRITE(OUTU,100)  &
            'TITLE   - formatted title line'
       IF(PRNLEV.GE.2) WRITE(OUTU,100) 'where <range> can be:'
       IF(PRNLEV.GE.2) WRITE(OUTU,100) 'XFIRST [REAL] XLAST [REAL] '
       IF(PRNLEV.GE.2) WRITE(OUTU,100) 'YFIRST [REAL] YLAST [REAL] '
       IF(PRNLEV.GE.2) WRITE(OUTU,100) 'ZFIRST [REAL] ZLAST [REAL] '
       IF(PRNLEV.GE.2) WRITE(OUTU,100)
    ELSE
       IF(PRNLEV.GE.2) WRITE(OUTU,100)  &
            'Set the quantity to write (try WRITE HELP)'

    ENDIF

    RETURN
  END SUBROUTINE WRIGD0
  !
  SUBROUTINE WRIGD1(UNIT,NCLX,NCLY,NCLZ,DCEL, &
       TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
       SMTHNG,PRFTR)
    !-----------------------------------------------------------------------
    use chm_kinds
    use dimens_fcm
    use number
    use comand
    use stream
    use string

    implicit none
    INTEGER UNIT,NCLX,NCLY,NCLZ
    real(chm_real)  DCEL,TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN
    REAL(CHM_REAL4)  SMTHNG(*), PRFTR
    !
    !     Local variables
    INTEGER IXS,IXF,IYS,IYF,IZS,IZF,I,J,K,L
    real(chm_real) XFIR,XLAS,YFIR,YLAS,ZFIR,ZLAS,XTEM,YTEM,ZTEM

    xfir   = GTRMF(COMLYN,COMLEN,'XFIR',ZERO)
    yfir   = GTRMF(COMLYN,COMLEN,'YFIR',ZERO)
    zfir   = GTRMF(COMLYN,COMLEN,'ZFIR',ZERO)
    xlas   = GTRMF(COMLYN,COMLEN,'XLAS',ONE*XFIR)
    ylas   = GTRMF(COMLYN,COMLEN,'YLAS',ONE*YFIR)
    zlas   = GTRMF(COMLYN,COMLEN,'ZLAS',ONE*ZFIR)

    xfir   = xfir+tranx-xbcen
    ixs=int(xfir/dcel)+1
    if(ixs.le.0)ixs=1
    if(ixs.gt.nclx)ixs=nclx

    yfir   = yfir+trany-ybcen
    iys=int(yfir/dcel)+1
    if(iys.le.0)iys=1
    if(iys.gt.ncly)iys=ncly

    zfir   = zfir+tranz-zbcen
    izs=int(zfir/dcel)+1
    if(izs.le.0)izs=1
    if(izs.gt.nclz)izs=nclz

    xlas   = xlas+tranx-xbcen
    ixf=int(xlas/dcel)+1
    if(ixf.le.0)ixf=1
    if(ixf.gt.nclx)ixf=nclx

    ylas   = ylas+trany-ybcen
    iyf=int(ylas/dcel)+1
    if(iyf.le.0)iyf=1
    if(iyf.gt.ncly)iyf=ncly

    zlas   = zlas+tranz-zbcen
    izf=int(zlas/dcel)+1
    if(izf.le.0)izf=1
    if(izf.gt.nclz)izf=nclz

    do i=ixs,ixf
       xtem=(i-1)*dcel-tranx+xbcen
       do j=iys,iyf
          ytem=(j-1)*dcel-trany+ybcen
          do k=izs,izf
             l=(i-1)*ncly*nclz+(j-1)*nclz+k
             ztem=(k-1)*dcel-tranz+zbcen
             write(UNIT,20) xtem,ytem,ztem,SMTHNG(l)*PRFTR
20           format(3f10.5,2x,e16.7)
          enddo
       enddo
    enddo

    RETURN
  END SUBROUTINE WRIGD1
  !
  SUBROUTINE WRIGD2(UNIT,POL1,POL2,POL3,SHAPE, &
       LSTP1,LSTP2,LSTP3,MIJ)
    !-----------------------------------------------------------------------
    use chm_kinds
    use stream
    implicit none
    real(chm_real)  MIJ(*)
    INTEGER UNIT,POL1,POL2,POL3
    INTEGER LSTP1(*),LSTP2(*),LSTP3(*)
    CHARACTER(len=8) SHAPE
    ! local
    INTEGER NTPOL,NTPOL2,I
    !
    IF(SHAPE.EQ.'RECTBOX ') THEN
       NTPOL =POL1*POL2*POL3
       NTPOL2=NTPOL*NTPOL
       IF(PRNLEV.GE.2) WRITE(OUTU,'(3x,2A)') &
            'FIRST RECORD IS: XNPOL,YNPOL,ZNPOL,LSTPX,LSTPY,LSTPZ,MIJ FOR ', &
            SHAPE
       WRITE(UNIT,'(a8)') SHAPE
       WRITE(UNIT,'(3i5)') POL1,POL2,POL3
       WRITE(UNIT,'(i5)') (LSTP1(I),I=1,NTPOL)
       WRITE(UNIT,'(i5)') (LSTP2(I),I=1,NTPOL)
       WRITE(UNIT,'(i5)') (LSTP3(I),I=1,NTPOL)
       WRITE(UNIT,'(e16.7)') (MIJ(I),I=1,NTPOL2)
    ELSE
       NTPOL =POL1*POL1
       NTPOL2=NTPOL*NTPOL
       IF(PRNLEV.GE.2) WRITE(OUTU,'(3x,2A)') &
            'FIRST RECORD IS: NMPOL,MIJ FOR ',SHAPE
       WRITE(UNIT,'(a8)') SHAPE
       WRITE(UNIT,'(i5)') POL1
       WRITE(UNIT,'(e16.7)') (MIJ(I),I=1,NTPOL2)
    ENDIF
    !
    RETURN
  END SUBROUTINE WRIGD2
  !
  SUBROUTINE WRIGD3(UNIT,NCLX,NCLY,NCLZ,DCEL,PHIX,PRFTR, &
       XBCEN,YBCEN,ZBCEN,EPSW,EPSP,CONC,TMEMB,ZMEMB,EPSM)
    !-----------------------------------------------------------------------
    use chm_kinds
    use dimens_fcm
    use stream
    implicit none
    INTEGER UNIT
    INTEGER NCLX,NCLY,NCLZ
    real(chm_real)  DCEL
    real(chm_real)  XBCEN,YBCEN,ZBCEN,EPSW,EPSP,CONC,TMEMB,ZMEMB,EPSM
    REAL(CHM_REAL4)  PHIX(*)
    REAL(CHM_REAL4)  PRFTR
    !
    !     Local variable
    INTEGER I

    IF(UNIT.GT.0)THEN
       IF(PRNLEV.GE.2) WRITE(OUTU,'(3x,A)') &
            'FIRST RECORD IS: NCLX,NCLY,NCLZ,DCEL,XBCEN,YBCEN,ZBCEN'
       IF(PRNLEV.GE.2) WRITE(OUTU,'(3x,A)') &
            '                 EPSW,EPSP,CONC,TMEMB,ZMEMB,EPSM'
       WRITE(UNIT, '(3i5)') NCLX,NCLY,NCLZ
       WRITE(UNIT, '(4e16.7)') DCEL,XBCEN,YBCEN,ZBCEN
       WRITE(UNIT, '(6e16.7)') EPSW,EPSP,CONC,TMEMB,ZMEMB,EPSM
       WRITE(UNIT, '(e16.7)')(PRFTR*PHIX(I),I=1,NCLX*NCLY*NCLZ)
    ENDIF
    !
    RETURN
  END SUBROUTINE WRIGD3
  !
  SUBROUTINE WRIGDPDB(UNIT,NCLX,NCLY,NCLZ,DCEL, &
       TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN,SMTHNG)
    !-----------------------------------------------------------------------
    use chm_kinds
    implicit none
    INTEGER UNIT,NCLX,NCLY,NCLZ
    real(chm_real)  DCEL,TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN
    REAL(CHM_REAL4)  SMTHNG(*)
    ! local
    integer  il,i,j,k,ncyz,cnt
    real(chm_real)   xi,yi,zi
    !
    cnt=0
    ncyz=ncly*nclz
    do il=1,nclx*ncyz
       if(smthng(il).ne.0.0) then
          cnt=cnt+1
          i=int((il-1)/ncyz)+1
          j=int(mod((il-1),ncyz)/nclz)+1
          k=mod(mod((il-1),ncyz),nclz)+1
          xi=dcel*(i-1)-tranx+xbcen
          yi=dcel*(j-1)-trany+ybcen
          zi=dcel*(k-1)-tranz+zbcen
          write(UNIT,101) &
               'ATOM',CNT,' POL ARG     1', &
               XI*10.,YI*10.,ZI*10.,SMTHNG(IL),'  1.00      LPOL'
       ENDIF
    enddo
101 format(a,2x,i5,1x,a,4x,3f8.3,f6.2,a)
    !
    RETURN
  END SUBROUTINE WRIGDPDB
  !
  SUBROUTINE TMPWRIGD(UNIT,NCLX,NCLY,NCLZ,DCEL, &
       TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
       XFIR0,YFIR0,ZFIR0,XLAS0,YLAS0,ZLAS0, &
       SMTHNG,PRFTR)
    !-----------------------------------------------------------------------
    use chm_kinds
    use dimens_fcm
    use number
    use comand
    use stream
    implicit none
    INTEGER UNIT,NCLX,NCLY,NCLZ
    real(chm_real)  DCEL,TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN
    REAL(CHM_REAL4)  SMTHNG(*), PRFTR
    !
    !     Local variables
    INTEGER IXS,IXF,IYS,IYF,IZS,IZF,I,J,K,L
    real(chm_real) XFIR0,XLAS0,YFIR0,YLAS0,ZFIR0,ZLAS0
    real(chm_real) XFIR,XLAS,YFIR,YLAS,ZFIR,ZLAS,XTEM,YTEM,ZTEM

    xfir   = xfir0+tranx-xbcen
    ixs=int(xfir/dcel)+1
    if(ixs.le.0)ixs=1
    if(ixs.gt.nclx)ixs=nclx

    yfir   = yfir0+trany-ybcen
    iys=int(yfir/dcel)+1
    if(iys.le.0)iys=1
    if(iys.gt.ncly)iys=ncly

    zfir   = zfir0+tranz-zbcen
    izs=int(zfir/dcel)+1
    if(izs.le.0)izs=1
    if(izs.gt.nclz)izs=nclz

    xlas   = xlas0+tranx-xbcen
    ixf=int(xlas/dcel)+1
    if(ixf.le.0)ixf=1
    if(ixf.gt.nclx)ixf=nclx

    ylas   = ylas0+trany-ybcen
    iyf=int(ylas/dcel)+1
    if(iyf.le.0)iyf=1
    if(iyf.gt.ncly)iyf=ncly

    zlas   = zlas0+tranz-zbcen
    izf=int(zlas/dcel)+1
    if(izf.le.0)izf=1
    if(izf.gt.nclz)izf=nclz

    !      write(6,'(3f10.5)') tranx,trany,tranz
    !      write(6,'(6f10.5)') xfir0,yfir0,zfir0,xlas0,ylas0,zlas0
    !      write(6,'(6f10.5)') xfir,yfir,zfir,xlas,ylas,zlas
    !      write(6,'(6i10)')   ixs,iys,izs,ixf,iyf,izf

    do i=ixs,ixf
       xtem=(i-1)*dcel-tranx+xbcen
       do j=iys,iyf
          ytem=(j-1)*dcel-trany+ybcen
          do k=izs,izf
             l=(i-1)*ncly*nclz+(j-1)*nclz+k
             ztem=(k-1)*dcel-tranz+zbcen
             write(UNIT,20) xtem,ytem,ztem,SMTHNG(l)*PRFTR
20           format(3f10.5,2x,e16.7)
          enddo
       enddo
    enddo

    RETURN
  END SUBROUTINE TMPWRIGD
  !
  SUBROUTINE SAVGD1(UNIT,NCLX,NCLY,NCLZ,DCEL,SMTHNG,PRFTR, &
       XBCEN,YBCEN,ZBCEN,EPSW,EPSP,CONC,TMEMB,ZMEMB,EPSM)
    !-----------------------------------------------------------------------
    use chm_kinds
    use dimens_fcm
    use stream
    implicit none
    INTEGER UNIT
    INTEGER NCLX,NCLY,NCLZ
    real(chm_real)  DCEL
    real(chm_real)  XBCEN,YBCEN,ZBCEN,EPSW,EPSP,CONC,TMEMB,ZMEMB,EPSM
    REAL(CHM_REAL4)  SMTHNG(*)
    REAL(CHM_REAL4)  PRFTR
    !
    !     Local variable
    INTEGER I

    IF(UNIT.GT.0)THEN
       IF(PRNLEV.GE.2) WRITE(OUTU,'(3x,A)') &
            'FIRST RECORD IS: NCLX,NCLY,NCLZ,DCEL,XBCEN,YBCEN,ZBCEN'
       IF(PRNLEV.GE.2) WRITE(OUTU,'(3x,A)') &
            '                 EPSW,EPSP,CONC,TMEMB,ZMEMB,EPSM'
       WRITE(UNIT) NCLX,NCLY,NCLZ,DCEL,XBCEN,YBCEN,ZBCEN
       WRITE(UNIT) EPSW,EPSP,CONC,TMEMB,ZMEMB,EPSM
       WRITE(UNIT)(PRFTR*SMTHNG(I),I=1,NCLX*NCLY*NCLZ)
    ENDIF
    !
    RETURN
  END SUBROUTINE SAVGD1
  !
  SUBROUTINE SAVGD2(UNIT,POL1,POL2,POL3,SHAPE, &
       LSTP1,LSTP2,LSTP3,MIJ)
    !-----------------------------------------------------------------------
    use chm_kinds
    use stream
    implicit none
    real(chm_real)  MIJ(*)
    INTEGER UNIT,POL1,POL2,POL3
    INTEGER LSTP1(*),LSTP2(*),LSTP3(*)
    CHARACTER(len=8) SHAPE
    ! local
    INTEGER NTPOL,NTPOL2,I
    !
    IF(SHAPE.EQ.'RECTBOX ') THEN
       NTPOL =POL1*POL2*POL3
       NTPOL2=NTPOL*NTPOL
       IF(PRNLEV.GE.2) WRITE(OUTU,'(3x,2A)') &
            'FIRST RECORD IS: XNPOL,YNPOL,ZNPOL,LSTPX,LSTPY,LSTPZ,MIJ FOR ', &
            SHAPE
       WRITE(UNIT) SHAPE
       WRITE(UNIT) POL1,POL2,POL3
       WRITE(UNIT) (LSTP1(I),I=1,NTPOL)
       WRITE(UNIT) (LSTP2(I),I=1,NTPOL)
       WRITE(UNIT) (LSTP3(I),I=1,NTPOL)
       WRITE(UNIT) (MIJ(I),I=1,NTPOL2)
    ELSE
       NTPOL =POL1*POL1
       NTPOL2=NTPOL*NTPOL
       IF(PRNLEV.GE.2) WRITE(OUTU,'(3x,2A)') &
            'FIRST RECORD IS: NMPOL,MIJ FOR ',SHAPE
       WRITE(UNIT) SHAPE
       WRITE(UNIT) POL1
       WRITE(UNIT) (MIJ(I),I=1,NTPOL2)
    ENDIF
    !
    RETURN
  END  SUBROUTINE SAVGD2
  !
  SUBROUTINE REAGD0
    !-----------------------------------------------------------------------
    use chm_kinds
    use dimens_fcm
    use number
    use comand
    use stream
    use string
    use memory
    use psf
    use coord
    use parallel,only: psnd4,psnd8

    implicit none
    !
    INTEGER UNIT
    INTEGER I
    CHARACTER(len=8) SHAPE
    real(chm_real)  REPSW,REPSP,RCONC,RTMEMB,RZMEMB,REPSM
    !
    INTEGER NTPOL2
    !
    UNIT    = GTRMI(COMLYN,COMLEN,'UNIT',OUTU)
    IF(PRNLEV.GE.2) WRITE(OUTU,'(3X,A,1X,I3)')  &
         'Reading from unit ',UNIT
    IF(UNIT.LE.0) GOTO 400
    !
    IF(INDXA(COMLYN,COMLEN,'MIJ').GT.0)THEN
       IF(IOLEV.GT.0)READ(UNIT,'(a8)') SHAPE
#if KEY_PARALLEL==1
       CALL PSNDC(SHAPE,1)                          
#endif
       IF(SHAPE.EQ.'RECTBOX ') THEN
          IF(IOLEV.GT.0)READ(UNIT,'(3i5)') OLDXNP,OLDYNP,OLDZNP
#if KEY_PARALLEL==1
          CALL PSND4(OLDXNP,1)
          CALL PSND4(OLDYNP,1)
          CALL PSND4(OLDZNP,1)
#endif 
          IF(PRNLEV.GT.2)THEN
             WRITE(OUTU,101) &
                  'Number of Legendre Polynomials in X  (XNPOL) = ',OLDXNP
             WRITE(OUTU,101) &
                  'Number of Legendre Polynomials in Y  (YNPOL) = ',OLDYNP
             WRITE(OUTU,101) &
                  'Number of Legendre Polynomials in Z  (ZNPOL) = ',OLDZNP
          ENDIF
          NTPOL =OLDXNP*OLDYNP*OLDZNP
          NTPOL2=NTPOL*NTPOL
          IF(.not.QGSBP) THEN
             call chmalloc('pbeq.src','REAGD0','MIJ',NTPOL2,crl=MIJ)
             call chmalloc('pbeq.src','REAGD0','LSTPX',NTPOL,intg=LSTPX)
             call chmalloc('pbeq.src','REAGD0','LSTPY',NTPOL,intg=LSTPY)
             call chmalloc('pbeq.src','REAGD0','LSTPZ',NTPOL,intg=LSTPZ)

          ENDIF
          CALL REAGD_INT0(UNIT,NTPOL,LSTPX)
          CALL REAGD_INT0(UNIT,NTPOL,LSTPY)
          CALL REAGD_INT0(UNIT,NTPOL,LSTPZ)
          CALL REAGD_R8_0(UNIT,NTPOL2,MIJ)
       ELSE
          IF(IOLEV.GT.0)READ(UNIT,'(i5)') OLDNMP
#if KEY_PARALLEL==1
          CALL PSND4(OLDNMP,1)                      
#endif
          IF(PRNLEV.GT.2)WRITE(OUTU,101) &
               'Number of Multipoles                 (NMPOL) = ',OLDNMP
          NTPOL = OLDNMP*OLDNMP
          NTPOL2= NTPOL*NTPOL
          IF(.not.QGSBP) THEN
             call chmalloc('pbeq.src','REAGD0','MIJ',NTPOL2,crl=MIJ)
          ENDIF
          CALL REAGD_R8_0(UNIT,NTPOL2,MIJ)
       ENDIF
       QMIJ=.true.
    ELSEIF (INDX(COMLYN,COMLEN,'PHIX',4).GT.0) THEN
       IF(IOLEV.GT.0)THEN
          READ(UNIT, '(3i5)') RNCLX,RNCLY,RNCLZ
          READ(UNIT, '(4e16.7)') RDCEL,RXBCEN,RYBCEN,RZBCEN
          READ(UNIT, '(6e16.7)') REPSW,REPSP,RCONC,RTMEMB,RZMEMB,REPSM
       ENDIF
#if KEY_PARALLEL==1
       CALL PSND4(RNCLX,1)
       CALL PSND4(RNCLY,1)
       CALL PSND4(RNCLZ,1)
       CALL PSND8(RDCEL,1) ! bug !? real was using psnd4 - also 3 below
       CALL PSND8(RXBCEN,1)
       CALL PSND8(RYBCEN,1)
       CALL PSND8(RZBCEN,1)
       CALL PSND8(REPSW,1)
       CALL PSND8(REPSP,1)
       CALL PSND8(RCONC,1)
       CALL PSND8(RTMEMB,1)
       CALL PSND8(RZMEMB,1)
       CALL PSND8(REPSM,1)
#endif 
       IF(PRNLEV.GT.2)THEN
          WRITE(OUTU,101)
          WRITE(OUTU,101) 'Number of grid points in X  (NCLX) = ',RNCLX
          WRITE(OUTU,101) 'Number of grid points in Y  (NCLY) = ',RNCLY
          WRITE(OUTU,101) 'Number of grid points in Z  (NCLZ) = ',RNCLZ
          WRITE(OUTU,102) 'Center of box in X          (XBCEN)= ',RXBCEN
          WRITE(OUTU,102) 'Center of box in Y          (YBCEN)= ',RYBCEN
          WRITE(OUTU,102) 'Center of box in Z          (ZBCEN)= ',RZBCEN
          WRITE(OUTU,102) 'Grid spacing                (DCEL) = ',RDCEL
          WRITE(OUTU,102)
          WRITE(OUTU,102) 'Solvent dielectric constant (EPSW) = ',REPSW
          WRITE(OUTU,102) 'Protein dielectric constant (EPSP) = ',REPSP
          WRITE(OUTU,102) 'Salt concentration          (CONC) = ',RCONC
          IF(RTMEMB.GT.ZERO)THEN
             WRITE(OUTU,102)
             WRITE(OUTU,102) 'Membrane thickness along Z  (TMEMB)= ',RTMEMB
             WRITE(OUTU,102) 'Membrane position along Z   (ZMEMB)= ',RZMEMB
             WRITE(OUTU,102) 'Membrane dielectric constant(EPSM) = ',REPSM
          ENDIF
       ENDIF
       TRANX = HALF*(RNCLX-1)*RDCEL
       TRANY = HALF*(RNCLY-1)*RDCEL
       TRANZ = HALF*(RNCLZ-1)*RDCEL
       IF(PRNLEV.GT.2)THEN
          WRITE(OUTU,102)
          WRITE(OUTU,102) 'Box in X from ',RXBCEN-TRANX,' to ',RXBCEN+TRANX
          WRITE(OUTU,102) 'Box in Y from ',RYBCEN-TRANY,' to ',RYBCEN+TRANY
          WRITE(OUTU,102) 'Box in Z from ',RZBCEN-TRANZ,' to ',RZBCEN+TRANZ
       ENDIF

       IF(INDXA(COMLYN,COMLEN,'PHIX').GT.0 )THEN
          ! static external potential
          IF(.not.QGSBP .or. .not.QSMBP) THEN ! JZ_UW12: Modified for SMBP
             NCEL3   =RNCLX*RNCLY*RNCLZ
             call chmalloc('pbeq.src','REAGD0','IPHIX',NCEL3,cr4=IPHIX)
          ELSEIF((NCLX.NE.RNCLX).OR.(NCLY.NE.RNCLY).OR.(NCLZ.NE.RNCLZ)) THEN
             IF(PRNLEV.GT.2) WRITE(OUTU,'(/,3x,A)') 'Previous grid setup has been reset'
             call chmdealloc('pbeq.src','REAGD0','IPHIX',NCEL3,cr4=IPHIX) 
             NCEL3   = RNCLX*RNCLY*RNCLZ
             call chmalloc('pbeq.src','REAGD0','IPHIX',NCEL3,cr4=IPHIX)
          ENDIF
          CALL REAGD_R4_0(UNIT,NCEL3,IPHIX)
          QPHIX=.true.
          QREADPHIX=.true. ! JZ_UW12
       ENDIF
    ELSE
       IF(PRNLEV.GT.2)WRITE(OUTU,'(A)') &
            'The code is under development'
    ENDIF
    !
101 FORMAT(3X,A,I6,A)
102 FORMAT(3X,A,F8.3,A,F8.3)
    !
400 CONTINUE
    RETURN
  END SUBROUTINE REAGD0
  !
  SUBROUTINE REAGD1
    !-----------------------------------------------------------------------
    use chm_kinds
    use dimens_fcm
    use number
    use comand
    use stream
    use string
    use memory
    use psf
    use coord
    use parallel,only: psnd4,psnd8
#if KEY_STRINGM==1 /*  VO stringm */
    use machio, only : ifreeu
    use multicom_aux
#endif
    !
    implicit none
    !
    INTEGER UNIT
    INTEGER I,RMAPT
    real(chm_real)  REPSW,REPSP,RCONC,RTMEMB,RZMEMB,REPSM
    CHARACTER(len=8) SHAPE
    !
    INTEGER NTPOL2
    !
#if KEY_STRINGM==1 /*  VO/XL to read from files open on each string node */
    integer :: oldiol
    logical :: qstr
    common /replicaio/ qstr ! needs to be global
#endif
!
    !
    UNIT    = GTRMI(COMLYN,COMLEN,'UNIT',OUTU)
    IF(UNIT.LE.0) GOTO 400
    IF(PRNLEV.GE.2) WRITE(OUTU,'(3X,A,1X,I3)')  &
         'Reading from unit ',UNIT
    !
#if KEY_STRINGM==1 /*  VO/XL stringm v */
    qstr=unit.le.size(ifreeu) ! protect against OOB on ifreeu
    if (qstr) qstr=((MOD(IFREEU(UNIT),8).eq.0).and.(IFREEU(UNIT).ne.0).and.(ME_LOCAL.eq.0))
    if (qstr) then
     oldiol=iolev
     iolev=1
    endif
#endif
    !
    IF(INDX(COMLYN,COMLEN,'PHI ',4).GT.0 .OR. &
         INDX(COMLYN,COMLEN,'FKAP',4).GT.0 .OR. &
         INDX(COMLYN,COMLEN,'PHIX',4).GT.0 )THEN
       IF(IOLEV.GT.0)THEN
          READ(UNIT) RNCLX,RNCLY,RNCLZ,RDCEL,RXBCEN,RYBCEN,RZBCEN
          READ(UNIT) REPSW,REPSP,RCONC,RTMEMB,RZMEMB,REPSM
       ENDIF
#if KEY_PARALLEL==1
       CALL PSND4(RNCLX,1)
       CALL PSND4(RNCLY,1)
       CALL PSND4(RNCLZ,1)
       CALL PSND8(RDCEL,1)
       CALL PSND8(RXBCEN,1)
       CALL PSND8(RYBCEN,1)
       CALL PSND8(RZBCEN,1)
       CALL PSND8(REPSW,1)
       CALL PSND8(REPSP,1)
       CALL PSND8(RCONC,1)
       CALL PSND8(RTMEMB,1)
       CALL PSND8(RZMEMB,1)
       CALL PSND8(REPSM,1)
#endif 
       IF(PRNLEV.GT.2)THEN
          WRITE(OUTU,101)
          WRITE(OUTU,101) 'Number of grid points in X  (NCLX) = ',RNCLX
          WRITE(OUTU,101) 'Number of grid points in Y  (NCLY) = ',RNCLY
          WRITE(OUTU,101) 'Number of grid points in Z  (NCLZ) = ',RNCLZ
          WRITE(OUTU,102) 'Center of box in X          (XBCEN)= ',RXBCEN
          WRITE(OUTU,102) 'Center of box in Y          (YBCEN)= ',RYBCEN
          WRITE(OUTU,102) 'Center of box in Z          (ZBCEN)= ',RZBCEN
          WRITE(OUTU,102) 'Grid spacing                (DCEL) = ',RDCEL
          WRITE(OUTU,102)
          WRITE(OUTU,102) 'Solvent dielectric constant (EPSW) = ',REPSW
          WRITE(OUTU,102) 'Protein dielectric constant (EPSP) = ',REPSP
          WRITE(OUTU,102) 'Salt concentration          (CONC) = ',RCONC
          IF(RTMEMB.GT.ZERO)THEN
             WRITE(OUTU,102)
             WRITE(OUTU,102) &
                  'Membrane thickness along Z  (TMEMB)= ',RTMEMB
             WRITE(OUTU,102) &
                  'Membrane position along Z   (ZMEMB)= ',RZMEMB
             WRITE(OUTU,102) &
                  'Membrane dielectric constant(EPSM) = ',REPSM
          ENDIF
       ENDIF
       TRANX = HALF*(RNCLX-1)*RDCEL
       TRANY = HALF*(RNCLY-1)*RDCEL
       TRANZ = HALF*(RNCLZ-1)*RDCEL
       IF(PRNLEV.GT.2)THEN
          WRITE(OUTU,102)
          WRITE(OUTU,102) &
               'Box in X from ',RXBCEN-TRANX,' to ',RXBCEN+TRANX
          WRITE(OUTU,102) &
               'Box in Y from ',RYBCEN-TRANY,' to ',RYBCEN+TRANY
          WRITE(OUTU,102) &
               'Box in Z from ',RZBCEN-TRANZ,' to ',RZBCEN+TRANZ
       ENDIF
101    FORMAT(3X,A,I6,A)
102    FORMAT(3X,A,F8.3,A,F8.3)

       IF(INDXA(COMLYN,COMLEN,'PHI ').GT.0)THEN
          IF(.not.QPBEQ) THEN
             call chmalloc('pbeq.src','REAGD1','PBRAD',NATOM,crl=PBRAD)
             call chmalloc('pbeq.src','REAGD1','LSTPRP',NATOM,intg=LSTPRP)
             PBRAD(1:NATOM)=WMAIN(1:NATOM)   ! save PB radii
!!!LNI               call SVPBRAD(NATOM,WMAIN,PBRAD)
             NCEL3=RNCLX*RNCLY*RNCLZ
             call chmalloc('pbeq.src','REAGD1','IPHIW',NCEL3,cr4=IPHIW)
             call chmalloc('pbeq.src','REAGD1','ICHC',NCEL3,cr4=ICHC)
             call chmalloc('pbeq.src','REAGD1','MAY',NCEL3,cr4=MAY)
             call chmalloc('pbeq.src','REAGD1','MAYX',NCEL3,cr4=MAYX)
             call chmalloc('pbeq.src','REAGD1','MAYY',NCEL3,cr4=MAYY)
             call chmalloc('pbeq.src','REAGD1','MAYZ',NCEL3,cr4=MAYZ)
             RMAPT  = 10000
             call chmalloc('pbeq.src','REAGD1','IPOX',RMAPT,crl=IPOX)
             call chmalloc('pbeq.src','REAGD1','IPOY',RMAPT,crl=IPOY)
             call chmalloc('pbeq.src','REAGD1','IPOZ',RMAPT,crl=IPOZ)
          ELSEIF((NCLX.NE.RNCLX).OR.(NCLY.NE.RNCLY).OR. &
               (NCLZ.NE.RNCLZ)) THEN
             IF(PRNLEV.GT.2)WRITE(OUTU,'(/,3x,A)') &
                  'Previous grid setup has been reset'
             call chmdealloc('pbeq.src','REAGD1','ICHC',NCEL3,cr4=ICHC)
             call chmdealloc('pbeq.src','REAGD1','IPHIW',NCEL3,cr4=IPHIW)
             call chmdealloc('pbeq.src','REAGD1','MAY',NCEL3,cr4=MAY)
             call chmdealloc('pbeq.src','REAGD1','MAYX',NCEL3,cr4=MAYX)
             call chmdealloc('pbeq.src','REAGD1','MAYY',NCEL3,cr4=MAYY)
             call chmdealloc('pbeq.src','REAGD1','MAYZ',NCEL3,cr4=MAYZ)
             NCEL3  = RNCLX*RNCLY*RNCLZ
             call chmalloc('pbeq.src','REAGD1','IPHIW',NCEL3,cr4=IPHIW)
             call chmalloc('pbeq.src','REAGD1','ICHC',NCEL3,cr4=ICHC)
             call chmalloc('pbeq.src','REAGD1','MAY',NCEL3,cr4=MAY)
             call chmalloc('pbeq.src','REAGD1','MAYX',NCEL3,cr4=MAYX)
             call chmalloc('pbeq.src','REAGD1','MAYY',NCEL3,cr4=MAYY)
             call chmalloc('pbeq.src','REAGD1','MAYZ',NCEL3,cr4=MAYZ)
          ENDIF
          DCEL=RDCEL
          NCLX=RNCLX
          NCLY=RNCLY
          NCLZ=RNCLZ
          XBCEN=RXBCEN
          YBCEN=RYBCEN
          ZBCEN=RZBCEN
          NCEL3=NCLX*NCLY*NCLZ
          CALL REAGD_R4(UNIT,NCEL3,IPHIW)
          QPBEQ=.true.
       ELSEIF(INDXA(COMLYN,COMLEN,'FKAP').GT.0 )THEN
          IF(.not.QPBEQ) THEN
             NCEL3  =RNCLX*RNCLY*RNCLZ
             call chmalloc('pbeq.src','REAGD1','MAY',NCEL3,cr4=MAY)
          ELSEIF((NCLX.NE.RNCLX).OR.(NCLY.NE.RNCLY).OR. &
               (NCLZ.NE.RNCLZ)) THEN
             IF(PRNLEV.GT.2)WRITE(OUTU,'(/,3x,A)') &
                  'Previous grid setup has been reset'
             call chmdealloc('pbeq.src','REAGD1','MAY',NCEL3,cr4=MAY)
             NCEL3  = RNCLX*RNCLY*RNCLZ
             call chmalloc('pbeq.src','REAGD1','MAY',NCEL3,cr4=MAY)
          ENDIF
          CALL REAGD_R4(UNIT,NCEL3,MAY)
          QFKAP=.true.
       ELSEIF(INDXA(COMLYN,COMLEN,'PHIX').GT.0 )THEN
          IF(.not.QGSBP .or. .not.QSMBP) THEN ! JZ_UW12: Modified for SMBP
             NCEL3   =RNCLX*RNCLY*RNCLZ
             call chmalloc('pbeq.src','REAGD1','IPHIX',NCEL3,cr4=IPHIX)
          ELSEIF((NCLX.NE.RNCLX).OR.(NCLY.NE.RNCLY).OR. &
               (NCLZ.NE.RNCLZ)) THEN
             IF(PRNLEV.GT.2)WRITE(OUTU,'(/,3x,A)') &
                  'Previous grid setup has been reset'
             call chmdealloc('pbeq.src','REAGD1','IPHIX',NCEL3,cr4=IPHIX)
             NCEL3   = RNCLX*RNCLY*RNCLZ
             call chmalloc('pbeq.src','REAGD1','IPHIX',NCEL3,cr4=IPHIX)
          ENDIF
          CALL REAGD_R4(UNIT,NCEL3,IPHIX)
          QPHIX=.true.
          QREADPHIX=.true. ! JZ_UW12
       ENDIF

    ELSEIF(INDXA(COMLYN,COMLEN,'MIJ').GT.0)THEN
       IF(IOLEV.GT.0)READ(UNIT) SHAPE
#if KEY_PARALLEL==1
       CALL PSNDC(SHAPE,1)                             
#endif
       IF(SHAPE.EQ.'RECTBOX ') THEN
          IF(IOLEV.GT.0)READ(UNIT) OLDXNP,OLDYNP,OLDZNP
#if KEY_PARALLEL==1
          CALL PSND4(OLDXNP,1)
          CALL PSND4(OLDYNP,1)
          CALL PSND4(OLDZNP,1)
#endif 
          IF(PRNLEV.GT.2)THEN
             WRITE(OUTU,101) &
                  'Number of Legendre Polynomials in X  (XNPOL) = ',OLDXNP
             WRITE(OUTU,101) &
                  'Number of Legendre Polynomials in Y  (YNPOL) = ',OLDYNP
             WRITE(OUTU,101) &
                  'Number of Legendre Polynomials in Z  (ZNPOL) = ',OLDZNP
          ENDIF
          NTPOL =OLDXNP*OLDYNP*OLDZNP
          NTPOL2=NTPOL*NTPOL
          IF(.not.QGSBP) THEN
             call chmalloc('pbeq.src','REAGD1','MIJ',NTPOL2,crl=MIJ)
             call chmalloc('pbeq.src','REAGD1','LSTPX',NTPOL,intg=LSTPX)
             call chmalloc('pbeq.src','REAGD1','LSTPY',NTPOL,intg=LSTPY)
             call chmalloc('pbeq.src','REAGD1','LSTPZ',NTPOL,intg=LSTPZ)
          ENDIF
          CALL REAGD_INT1(UNIT,NTPOL,LSTPX)
          CALL REAGD_INT1(UNIT,NTPOL,LSTPY)
          CALL REAGD_INT1(UNIT,NTPOL,LSTPZ)
          CALL REAGD_R8_1(UNIT,NTPOL2,MIJ)
       ELSE
          IF(IOLEV.GT.0)READ(UNIT) OLDNMP
#if KEY_PARALLEL==1
          CALL PSND4(OLDNMP,1)                      
#endif
          IF(PRNLEV.GT.2)WRITE(OUTU,101) &
               'Number of Multipoles                 (NMPOL) = ',OLDNMP
          NTPOL = OLDNMP*OLDNMP
          NTPOL2= NTPOL*NTPOL
          IF(.not.QGSBP) THEN
             call chmalloc('pbeq.src','REAGD1','MIJ',NTPOL2,crl=MIJ)
          ENDIF
          CALL REAGD_R8_1(UNIT,NTPOL2,MIJ)
       ENDIF
       QMIJ=.true.

    ELSEIF(INDXA(COMLYN,COMLEN,'MMIJ').GT.0)THEN
       IF(IOLEV.GT.0)READ(UNIT) SHAPE
#if KEY_PARALLEL==1
       CALL PSNDC(SHAPE,1)                        
#endif
       IF(SHAPE.EQ.'RECTBOX ') THEN
          IF(IOLEV.GT.0)READ(UNIT) OLDXNP,OLDYNP,OLDZNP
#if KEY_PARALLEL==1
          CALL PSND4(OLDXNP,1)
          CALL PSND4(OLDYNP,1)
          CALL PSND4(OLDZNP,1)
#endif 
          IF(PRNLEV.GT.2)THEN
             WRITE(OUTU,101) &
                  'Number of Legendre Polynomials in X  (XNPOL) = ',OLDXNP
             WRITE(OUTU,101) &
                  'Number of Legendre Polynomials in Y  (YNPOL) = ',OLDYNP
             WRITE(OUTU,101) &
                  'Number of Legendre Polynomials in Z  (ZNPOL) = ',OLDZNP
          ENDIF
          NTPOL =OLDXNP*OLDYNP*OLDZNP
          NTPOL2=NTPOL*NTPOL
          IF(.not.QGSBP) THEN
             call chmalloc('pbeq.src','REAGD1','MMIJ',NTPOL2,crl=MMIJ)
             call chmalloc('pbeq.src','REAGD1','LSTPX',NTPOL,intg=LSTPX)
             call chmalloc('pbeq.src','REAGD1','LSTPY',NTPOL,intg=LSTPY)
             call chmalloc('pbeq.src','REAGD1','LSTPZ',NTPOL,intg=LSTPZ)
          ENDIF
          CALL REAGD_INT1(UNIT,NTPOL,LSTPX)
          CALL REAGD_INT1(UNIT,NTPOL,LSTPY)
          CALL REAGD_INT1(UNIT,NTPOL,LSTPZ)
          CALL REAGD_R8_1(UNIT,NTPOL2,MMIJ)
       ELSE
          IF(IOLEV.GT.0)READ(UNIT) OLDNMP
#if KEY_PARALLEL==1
          CALL PSND4(OLDNMP,1)                     
#endif
          IF(PRNLEV.GT.2)WRITE(OUTU,101) &
               'Number of Multipoles                 (NMPOL) = ',OLDNMP
          NTPOL = OLDNMP*OLDNMP
          NTPOL2= NTPOL*NTPOL
          IF(.not.QGSBP) THEN
             call chmalloc('pbeq.src','REAGD1','MMIJ',NTPOL2,crl=MMIJ)
          ENDIF
          CALL REAGD_R8_1(UNIT,NTPOL2,MMIJ)
       ENDIF
       QMMIJ=.true.
       QMIJ =.true.

    ELSE
       IF(PRNLEV.GE.2) WRITE(OUTU,'(A)')  &
            'The code is under development'
    ENDIF
    !
#if KEY_STRINGM==1 /*  VO/XL  restore iolev */
  if (qstr) iolev=oldiol 
#endif
    !
400 CONTINUE
    RETURN
  END SUBROUTINE REAGD1
  !
  SUBROUTINE PBAVGRID
    !-----------------------------------------------------------------------
    ! sets up the calculation of averages over the grid
    !                                -RJP 1.20.02
    use chm_kinds
    use dimens_fcm
    use number
    use comand
    use consta
    use select
    use stream
    use string
    use memory
    use ctitla
    use psf
    use coord

    implicit none
    !
    integer,allocatable,dimension(:) :: ISLCT
    REAL(CHM_REAL4) PRFTR
    INTEGER :: DFLT,MODE,NUPDAT=0
    !
    PRFTR=1.
    ! LGRATS should be false in initialization (iniall)
    IF(INDXA(COMLYN,COMLEN,'PHI ').GT.0) THEN
100    FORMAT(3X,A)
       IF(INDXA(COMLYN,COMLEN,'ATOM').GT.0) THEN
          IF(PRNLEV.GE.2) WRITE(OUTU,100) &
               'Average electrostatic potential over atom-based grid'
          IF(INDXA(COMLYN,COMLEN,'KCAL').GT.0)THEN
             PRFTR=CCELEC
             IF(PRNLEV.GE.2) WRITE(OUTU,100) &
                  'in [KCAL/MOLE]/[UNIT CHARGE]'
          ELSEIF (INDXA(COMLYN,COMLEN,'VOLT').GT.0)THEN
             PRFTR=14.40145905
             IF(PRNLEV.GE.2) WRITE(OUTU,100) &
                  'in [VOLTS]'
          ELSE
             PRFTR=1.
             IF(PRNLEV.GE.2) WRITE(OUTU,100) &
                  'in [UNIT CHARGE]/[ANGS]'
          ENDIF
          IF(INDXA(COMLYN,COMLEN,'UPDA').GT.0) THEN
             IF(NUPDAT.EQ.0) THEN
                call chmalloc('pbeq.src','PBAVGRID','HPGRAT',NATOM,intg=HPGRAT)
                call chmalloc('pbeq.src','PBAVGRID','HPGDGR',NCEL3,intg=HPGDGR)
                NUPDAT = 1
             ENDIF
             DFLT = 0
             MODE = 0
             call chmalloc('pbeq.src','PBAVGRID','ISLCT',NATOM,intg=ISLCT)
             CALL SELRPN(COMLYN,COMLEN,ISLCT,NATOM,DFLT, &
                  MODE,.FALSE.,1,' ',0, RESID,RES,IBASE,SEGID,NICTOT, &
                  NSEG,.TRUE.,X,Y,Z,.TRUE.,1,WMAIN)
             !
             !     create the i and j atom lists for distance calculations
             !         
             CALL MKPBAVATAR(ISLCT,HPGRAT,NATOM,NGRATS)  !make selected atom array
             call chmdealloc('pbeq.src','PBAVGRID','ISLCT',NATOM,intg=ISLCT)
             CALL UPDTATMAVGRID(HPGDGR,GDMANY,HPGRAT, &
                  NGRATS,NCLX,NCLY,NCLZ,DCEL, &
                  TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
                  XGRBEG,XGREND,YGRBEG,YGREND,ZGRBEG,ZGREND)
             CALL PBPHIATMAVE(HPGDGR,GDMANY,IPHIW,PRFTR)
             LGRATS = .TRUE.
          ELSE IF(.NOT.LGRATS) THEN
             CALL WRNDIE(-4,'<PBAVGRID>', &
                  'ATOM-BASED GRID NOT SET UP: USE UPDATE SELE')
          ELSE
             CALL PBPHIATMAVE(HPGDGR,GDMANY,IPHIW,PRFTR)
          ENDIF
       ELSE !if not atom-based grid
          IF(PRNLEV.GE.2) WRITE(OUTU,100) &
               'Calculating average electrostatic potential over grid'
          IF(INDXA(COMLYN,COMLEN,'KCAL').GT.0)THEN
             PRFTR=CCELEC
             IF(PRNLEV.GE.2) WRITE(OUTU,100) &
                  'in [KCAL/MOLE]/[UNIT CHARGE]'
          ELSEIF (INDXA(COMLYN,COMLEN,'VOLT').GT.0)THEN
             PRFTR=14.40145905
             IF(PRNLEV.GE.2) WRITE(OUTU,100) &
                  'in [VOLTS]'
          ELSE
             PRFTR=1.
             IF(PRNLEV.GE.2) WRITE(OUTU,100) &
                  'in [UNIT CHARGE]/[ANGS]'
          ENDIF
          CALL PBPHIAVE(NCLX,NCLY,NCLZ,DCEL, &
               TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
               IPHIW,PRFTR, &
               XGRBEG,XGREND,YGRBEG,YGREND,ZGRBEG,ZGREND)
       ENDIF
    ENDIF
    RETURN
  END SUBROUTINE PBAVGRID
  !
  SUBROUTINE PBPHIATMAVE(GOODGR,GDMANY,SMTHNG,PRFTR)
    !-----------------------------------------------------------------------
    !  calculates the average phi value over a set of
    ! prespecified grid points   -RJP 1.20.02
    !
    use chm_kinds
    use stream
    use dimens_fcm
    use param_store, only: set_param

    implicit none
    !
    INTEGER GOODGR(*),GDMANY
    REAL(CHM_REAL4)  SMTHNG(*), PRFTR
    ! local variables
    INTEGER I,GRIDPT
    real(chm_real)  PHICEN, AVEPHI, AVEINC
    !
    !
    IF(GDMANY.EQ.0) THEN
       CALL WRNDIE(-5,'<PBPHIATMAVE>', &
            'NO GRIDPOINTS DEFINED FOR CALCULATION')
    ENDIF
    AVEPHI = 0
    DO I=1,GDMANY
       GRIDPT = GOODGR(I)
       PHICEN = SMTHNG(GRIDPT)- AVEPHI
       AVEINC = PHICEN/I
       AVEPHI = AVEPHI + AVEINC
    ENDDO
    AVEPHI = AVEPHI*PRFTR
    call set_param('AVPH',AVEPHI)
    IF(PRNLEV.GE.2) WRITE(OUTU,'(3x,A24,1x,I8,1x,A8,1x,F15.6)') &
         'AVG PHI CALCULATED OVER',GDMANY,'gridpts:',AVEPHI
    RETURN
  END SUBROUTINE PBPHIATMAVE
  !
  SUBROUTINE PBPHIAVE(NCLX,NCLY,NCLZ,DCEL, &
       TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
       SMTHNG,PRFTR, &
       XGRBEG,XGREND,YGRBEG,YGREND,ZGRBEG,ZGREND)
    !-----------------------------------------------------------------------
    !  calculates the average phi value over the grid.
    !  Made from WRIGD1   -RJP 1.20.01
    !
    use chm_kinds
    use stream
    use string
    use dimens_fcm
    use number
    use comand
    use param_store, only: set_param

    implicit none
    !
    INTEGER NCLX,NCLY,NCLZ
    real(chm_real)  DCEL,TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN
    REAL(CHM_REAL4)  SMTHNG(*), PRFTR
    real(chm_real) XGRBEG,XGREND,YGRBEG,YGREND,ZGRBEG,ZGREND
    !
    !     Local variables
    INTEGER IXS,IXF,IYS,IYF,IZS,IZF,I,J,K,L
    real(chm_real) XFIR,XLAS,YFIR,YLAS,ZFIR,ZLAS,XTEM,YTEM,ZTEM
    INTEGER NINCP1, NINCMT
    real(chm_real)  PHICEN, AVEPHI, AVEINC
    !
    xfir   = GTRMF(COMLYN,COMLEN,'XFIR',ONE*XGRBEG)
    yfir   = GTRMF(COMLYN,COMLEN,'YFIR',ONE*YGRBEG)
    zfir   = GTRMF(COMLYN,COMLEN,'ZFIR',ONE*ZGRBEG)
    xlas   = GTRMF(COMLYN,COMLEN,'XLAS',ONE*XGREND)
    ylas   = GTRMF(COMLYN,COMLEN,'YLAS',ONE*YGREND)
    zlas   = GTRMF(COMLYN,COMLEN,'ZLAS',ONE*ZGREND)
    XGRBEG = xfir
    YGRBEG = yfir
    ZGRBEG = zfir
    XGREND = xlas
    YGREND = ylas
    ZGREND = zlas
    !
    IF(((xfir.EQ.0).AND.(xlas.EQ.0)).OR.((yfir.EQ.0).and. &
         (ylas.EQ.0)).OR.((zfir.EQ.0).AND.(zlas.EQ.0))) THEN
       CALL WRNDIE(-5,'<PBPHIAVE>', &
            'GRID LIMITS NOT DEFINED: xfir, xlas, yfir, ylas...')
    ENDIF
    !
    xfir   = xfir+tranx-xbcen
    ixs=int(xfir/dcel)+1
    if(ixs.le.0)ixs=1
    if(ixs.gt.nclx)ixs=nclx

    yfir   = yfir+trany-ybcen
    iys=int(yfir/dcel)+1
    if(iys.le.0)iys=1
    if(iys.gt.ncly)iys=ncly

    zfir   = zfir+tranz-zbcen
    izs=int(zfir/dcel)+1
    if(izs.le.0)izs=1
    if(izs.gt.nclz)izs=nclz

    xlas   = xlas+tranx-xbcen
    ixf=int(xlas/dcel)+1
    if(ixf.le.0)ixf=1
    if(ixf.gt.nclx)ixf=nclx

    ylas   = ylas+trany-ybcen
    iyf=int(ylas/dcel)+1
    if(iyf.le.0)iyf=1
    if(iyf.gt.ncly)iyf=ncly

    zlas   = zlas+tranz-zbcen
    izf=int(zlas/dcel)+1
    if(izf.le.0)izf=1
    if(izf.gt.nclz)izf=nclz
    NINCMT = 0
    AVEPHI = 0
    do i=ixs,ixf
       xtem=(i-1)*dcel-tranx+xbcen
       do j=iys,iyf
          ytem=(j-1)*dcel-trany+ybcen
          do k=izs,izf
             l=(i-1)*ncly*nclz+(j-1)*nclz+k
             !         ztem=(k-1)*dcel-tranz+zbcen
             NINCP1 = (NINCMT+1)
             PHICEN = SMTHNG(l)- AVEPHI
             AVEINC = PHICEN/NINCP1
             AVEPHI = AVEPHI + AVEINC
             NINCMT = NINCMT + 1
          enddo
       enddo
    enddo
    AVEPHI = AVEPHI*PRFTR
    call set_param('AVPH',AVEPHI)
    IF(PRNLEV.GE.2) WRITE(OUTU,'(3x,A24,1x,I8,1x,A8,1x,F15.6)') &
         'AVG PHI CALCULATED OVER',NINCMT,'gridpts:',AVEPHI
    RETURN
  END SUBROUTINE PBPHIAVE
  !
  SUBROUTINE UPDTATMAVGRID(GOODGR,GDMANY,GRATSE,NGRATS, &
       NCLX,NCLY,NCLZ,DCEL, &
       TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
       XGRBEG,XGREND,YGRBEG,YGREND,ZGRBEG,ZGREND)
    !-----------------------------------------------------------------------
    !  updates the set of selected gridpoints used for the
    ! calculation of averages        --RJP 1.20.01
    use chm_kinds
    use stream
    use string
    use dimens_fcm
    use number
    use comand
    use param
    use psf
    use coord

    implicit none
    !
    INTEGER GOODGR(*),GRATSE(*)
    INTEGER NCLX,NCLY,NCLZ,GDMANY,NGRATS
    real(chm_real)  DCEL,TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN
    real(chm_real)  XGRBEG,XGREND,YGRBEG,YGREND,ZGRBEG,ZGREND
    !

    !     Local variables
    INTEGER IXS,IXF,IYS,IYF,IZS,IZF,I,J,K,L
    real(chm_real) XFIR,XLAS,YFIR,YLAS,ZFIR,ZLAS,XTEM,YTEM,ZTEM
    LOGICAL HIT
    INTEGER ATOMNB,COUNTI
    real(chm_real)  RSQUAR,RADIII,XDIFF,YDIFF,ZDIFF
    xfir   = GTRMF(COMLYN,COMLEN,'XFIR',ONE*XGRBEG)
    yfir   = GTRMF(COMLYN,COMLEN,'YFIR',ONE*YGRBEG)
    zfir   = GTRMF(COMLYN,COMLEN,'ZFIR',ONE*ZGRBEG)
    xlas   = GTRMF(COMLYN,COMLEN,'XLAS',ONE*XGREND)
    ylas   = GTRMF(COMLYN,COMLEN,'YLAS',ONE*YGREND)
    zlas   = GTRMF(COMLYN,COMLEN,'ZLAS',ONE*ZGREND)
    XGRBEG = xfir
    YGRBEG = yfir
    ZGRBEG = zfir
    XGREND = xlas
    YGREND = ylas
    ZGREND = zlas
    !
    IF(((xfir.EQ.0).AND.(xlas.EQ.0)).OR.((yfir.EQ.0).and. &
         (ylas.EQ.0)).OR.((zfir.EQ.0).AND.(zlas.EQ.0))) THEN
       CALL WRNDIE(-5,'<UPDTATMAVGRID>', &
            'GRID LIMITS NOT DEFINED: xfir, xlas, yfir, ylas...')
    ENDIF
    IF(NGRATS.EQ.0) THEN
       CALL WRNDIE(-5,'<UPDTATMAVGRID>', &
            'ATOM ARRAY EMPTY FOR ATOM-BASED AVERAGE CALC')
    ENDIF
    !
    xfir   = xfir+tranx-xbcen
    ixs=int(xfir/dcel)+1
    if(ixs.le.0)ixs=1
    if(ixs.gt.nclx)ixs=nclx

    yfir   = yfir+trany-ybcen
    iys=int(yfir/dcel)+1
    if(iys.le.0)iys=1
    if(iys.gt.ncly)iys=ncly

    zfir   = zfir+tranz-zbcen
    izs=int(zfir/dcel)+1
    if(izs.le.0)izs=1
    if(izs.gt.nclz)izs=nclz

    xlas   = xlas+tranx-xbcen
    ixf=int(xlas/dcel)+1
    if(ixf.le.0)ixf=1
    if(ixf.gt.nclx)ixf=nclx

    ylas   = ylas+trany-ybcen
    iyf=int(ylas/dcel)+1
    if(iyf.le.0)iyf=1
    if(iyf.gt.ncly)iyf=ncly

    zlas   = zlas+tranz-zbcen
    izf=int(zlas/dcel)+1
    if(izf.le.0)izf=1
    if(izf.gt.nclz)izf=nclz
    GDMANY = 0
    do i=ixs,ixf
       xtem=(i-1)*dcel-tranx+xbcen
       do j=iys,iyf
          ytem=(j-1)*dcel-trany+ybcen
          do k=izs,izf
             l=(i-1)*ncly*nclz+(j-1)*nclz+k
             ztem=(k-1)*dcel-tranz+zbcen
             HIT = .FALSE.
             COUNTI = 1
             DO WHILE((.NOT.HIT).AND.(COUNTI.LE.NGRATS))
                ATOMNB = GRATSE(COUNTI)
                RADIII = VDWR(ITC(IAC(ATOMNB)))
                RSQUAR = RADIII*RADIII
                XDIFF = X(ATOMNB)- XTEM
                XDIFF = XDIFF*XDIFF
                YDIFF = Y(ATOMNB)- YTEM
                YDIFF = YDIFF*YDIFF
                ZDIFF = Z(ATOMNB)- ZTEM
                ZDIFF = ZDIFF*ZDIFF
                IF ((XDIFF+YDIFF+ZDIFF).LE.(RSQUAR)) THEN
                   GDMANY = GDMANY + 1
                   GOODGR(GDMANY) = L
                   HIT = .TRUE.
                ENDIF
                COUNTI = COUNTI + 1
             ENDDO
          enddo
       enddo
    enddo
    IF(PRNLEV.GE.2) WRITE(OUTU,'(3x,A24,1x,I7,1x,A5,1x,I8,1x,A7)') &
         'UPDATED ATOM-BASED GRID:',NGRATS,'atoms',GDMANY,'gridpts'
    RETURN
  END SUBROUTINE UPDTATMAVGRID
  !
  SUBROUTINE MKPBAVATAR(ISLCT,GRATSE,NATOMX,NGRATS)
    ! creates the selected atom array for grid averages
    !                                 --RJP 1.20.01
    use chm_kinds
    use stream
    implicit none
    !
    INTEGER NGRATS,NATOMX,ISLCT(*),GRATSE(*)
    INTEGER I,COUNTI
    !
    COUNTI = 0
    DO I = 1,NATOMX
       IF(ISLCT(I).GT.0) THEN
          COUNTI = COUNTI + 1
       ENDIF
    ENDDO
    !
    IF(COUNTI.GT.0) THEN
       NGRATS = 0
       DO I = 1,NATOMX
          IF (ISLCT(I).GT.0) THEN
             NGRATS = NGRATS + 1
             GRATSE(NGRATS) = I
          ENDIF
       ENDDO
    ELSE
       IF(PRNLEV.GE.2) WRITE(OUTU,'(3x,A14,1x,I8,1x,A14)') &
            'USING PREVIOUS',NGRATS,'SELECTED ATOMS'
    ENDIF
    !
    RETURN
  END SUBROUTINE MKPBAVATAR
  !
  SUBROUTINE REAGD_R4(UNIT,DIM,SMTHNG)
    !----------------------------------------------------------------------
    !
    use chm_kinds
    use stream
    use parallel,only: psnd4,psnd8
    implicit none
    INTEGER I,DIM,UNIT
    REAL(CHM_REAL4)  SMTHNG(*)

    IF(IOLEV.GT.0)READ(UNIT)(SMTHNG(I),I=1,DIM)
#if KEY_PARALLEL==1
    CALL PSND4(SMTHNG,DIM)                      
#endif

    RETURN
  END SUBROUTINE REAGD_R4
  !
  SUBROUTINE REAGD_R4_0(UNIT,DIM,SMTHNG)
    !----------------------------------------------------------------------
    !
    use chm_kinds
    use stream
    use parallel,only: psnd4,psnd8
    implicit none
    INTEGER I,DIM,UNIT
    REAL(CHM_REAL4)  SMTHNG(*)

    IF(IOLEV.GT.0)READ(UNIT, '(e16.7)')(SMTHNG(I),I=1,DIM)
#if KEY_PARALLEL==1
    CALL PSND4(SMTHNG,DIM)                      
#endif

    RETURN
  END SUBROUTINE REAGD_R4_0

  SUBROUTINE REAGD_R8_0(UNIT,DIM,SMTHNG)
    !----------------------------------------------------------------------
    !
    use chm_kinds
    use stream
    use parallel,only: psnd8
    implicit none
    INTEGER I,DIM,UNIT
    real(chm_real)  SMTHNG(*)

    IF(IOLEV.GT.0)READ(UNIT,'(e16.7)')(SMTHNG(I),I=1,DIM)
#if KEY_PARALLEL==1
    CALL PSND8(SMTHNG,DIM)                      
#endif
    RETURN
  END SUBROUTINE REAGD_R8_0
  !
  SUBROUTINE REAGD_R8_1(UNIT,DIM,SMTHNG)
    !----------------------------------------------------------------------
    !
    use chm_kinds
    use stream
    use parallel,only: psnd8
    implicit none
    INTEGER I,DIM,UNIT
    real(chm_real)  SMTHNG(*)

    IF(IOLEV.GT.0)READ(UNIT)(SMTHNG(I),I=1,DIM)
#if KEY_PARALLEL==1
    CALL PSND8(SMTHNG,DIM)                     
#endif
    RETURN
  END SUBROUTINE REAGD_R8_1
  !
  SUBROUTINE REAGD_INT0(UNIT,DIM,SMTHNG)
    !----------------------------------------------------------------------
    !
    use chm_kinds
    use stream
    use parallel,only: psnd4
    implicit none
    INTEGER I,DIM,UNIT,SMTHNG(*)

    IF(IOLEV.GT.0)READ(UNIT,'(i5)')(SMTHNG(I),I=1,DIM)
#if KEY_PARALLEL==1
    CALL PSND4(SMTHNG,DIM)                     
#endif

    RETURN
  END SUBROUTINE REAGD_INT0
  !
  SUBROUTINE REAGD_INT1(UNIT,DIM,SMTHNG)
    !----------------------------------------------------------------------
    !
    use chm_kinds
    use stream
    use parallel,only: psnd4
    implicit none
    INTEGER I,DIM,UNIT,SMTHNG(*)

    IF(IOLEV.GT.0)READ(UNIT)(SMTHNG(I),I=1,DIM)
#if KEY_PARALLEL==1
    CALL PSND4(SMTHNG,DIM)                     
#endif

    RETURN
  END SUBROUTINE REAGD_INT1
  !
  FUNCTION segpar(raneti,ranetj,aij) result(segpar_rslt)
    !------------------------------------------------------------------------
    ! Segment part formula (used in MAYER)
    !
    use chm_kinds
    implicit none
    real(chm_real) :: segpar_rslt
    real(chm_real) raneti,ranetj,aij,v1,v2
    !
    v1=raneti+ranetj
    v2=abs(raneti-ranetj)
    if(aij.gt.v1)then
       segpar_rslt=0.
    elseif(aij.lt.v2)then
       if(raneti.gt.ranetj)then
          segpar_rslt=0.
       else
          segpar_rslt=1.
       endif
    else
       segpar_rslt=(ranetj**2-(raneti-aij)**2)/(4*aij*raneti)
    endif
    return
  end FUNCTION segpar
  !
  Function simp(x,h) result(simp_rslt)
    !------------------------------------------------------------------------
    !
    use chm_kinds
    use number
    implicit none
    real(chm_real) :: simp_rslt
    real(chm_real) x, h

    simp_rslt = ( exp(-(x-h)**2) + four*exp(-x**2) &
         + exp(-(x+h)**2) )/three
    !
    return
  end Function simp
  !
  Function epssf(r,h) result(epssf_rslt)
    !------------------------------------------------------------------------
    !
    use chm_kinds
    use number
    implicit none
    !
    real(chm_real)  r,h,epssf_rslt
    !
    epssf_rslt=(-one/four/h**3)*r**3 + (three/four/h**2)*r**2
    !
    return
  end FUNCTION epssf
  !
  Function depssf(r,h) result(depssf_rslt)
    !------------------------------------------------------------------------
    !
    use chm_kinds
    use number
    implicit none
    !
    real(chm_real)  r,h,depssf_rslt
    !
    depssf_rslt=three*(-one/four/h**3)*r**2 + two*(three/four/h**2)*r
    !
    return
  end function depssf
  !
  !
  Function M3(x) result(m3_rslt)
    !------------------------------------------------------------------------
    !
    use chm_kinds
    use number
    implicit none
    !
    real(chm_real)   M2m,M2,X,m3_rslt
    !
    !     N=2
    if((x.ge.zero).and.(x.le.two))  M2m = one - abs(x - one)
    if((x.lt.zero).or.(x.gt.two))   M2m = zero
    if((x.ge.one).and.(x.le.three)) M2  = one - abs(x - two)
    if((x.lt.one).or.(x.gt.three))  M2  = zero
    !
    !     N=3
    if((x.ge.zero).and.(x.le.three)) M3_rslt = x/two*M2m + (three-x)/two*M2
    if((x.lt.zero).or.(x.gt.three))  M3_rslt = zero
    !
    return
  end function m3
  !
  Function DM3(x) result(dm3_rslt)
    !------------------------------------------------------------------------
    !
    use chm_kinds
    use number
    implicit none
    !
    real(chm_real)   M2m,M2,X,dm3_rslt
    !
    !     N=2
    if((x.ge.zero).and.(x.le.two))  M2m = one - abs(x - one)
    if((x.lt.zero).or.(x.gt.two))   M2m = zero
    if((x.ge.one).and.(x.le.three)) M2  = one - abs(x - two)
    if((x.lt.one).or.(x.gt.three))  M2  = zero
    !
    DM3_rslt=M2m-M2
    !
#if KEY_ELSE==1
!!
#endif
!!!      SUBROUTINE NULL_PBEQ2
#if KEY_ENDIF==1
!!
#endif
    return
  end function dm3

  ! xxxxx
  ! XIAO_QC_UW0609
#if KEY_SCCDFTB==1
  SUBROUTINE BFORCE2(NTPRP,LSTPRP,X,Y,Z,PBRAD,EPSW,EPSP, &
       NCLX,NCLY,NCLZ,DCEL,PHI1,RMAYX,RMAYY,RMAYZ,SWIN, &
       TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
       DBFX,DBFY,DBFZ,IBFX,IBFY,IBFZ, &
                                !               RMAY,KAPPA2)
       RMAY,KAPPA2,NPFX,NPFY,NPFZ,STCO,Enp,QPRIN)
    !-----------------------------------------------------------------------
    !     This subroutine computes electrostatic solvation forces resulting
    !     from the dielectric bounary and ionic boundary.
    !     And nonpolar solvation forces also are calculated, if necessary.
    !     Dielectric Boundary Force - DBF
    !     NonPolar solvation Force - NPF
    !     van der Waals surface
    !     Xiao modifid to calculate DBF,NPF,IBF from bforce for scc/pb 
    !
    use consta
    use number
    use stream

    implicit none
    integer ntprp, lstprp(*)
    REAL(chm_real4)  RMAYX(*),RMAYY(*),RMAYZ(*),RMAY(*)
    REAL(chm_real)  X(*),Y(*),Z(*),PBRAD(*)
    REAL(chm_real)  DBFX(*),DBFY(*),DBFZ(*)
    REAL(chm_real)  NPFX(*),NPFY(*),NPFZ(*)
    REAL(chm_real)  IBFX(*),IBFY(*),IBFZ(*)
    REAL(chm_real)  DCEL,TRANX,TRANY,TRANZ,EPSW,EPSP,STCO,KAPPA2
    REAL(chm_real)  XBCEN,YBCEN,ZBCEN
    REAL(chm_real4)  PHI1(*)
    INTEGER NCLX,NCLY,NCLZ
    LOGICAL QPRIN
    !   local
    REAL(chm_real)    xc,yc,zc,xi,yi,zi,bisq
    INTEGER   IL,K,L,M,IPX,IPY,IPZ,NCyz,I,nfil
    INTEGER   ix,iy,iz,jx1,jx2,jy1,jy2,jz1,jz2
    !
    REAL(chm_real)    drp,drm,dxp,dxm,dyp,dym,dzp,dzm,dx,dy,dz
    REAL(chm_real)    rxp,rxm,ryp,rym,rzp,rzm,swin,bms,bps
    real(chm_real)    prefac1,prefac2,prefac3,prefac4
    real(chm_real)    Enp,depsx,depsy,depsz,moeps
    real(chm_real)    dcel2,rij,dr
    real(chm_real4)    kappab
    !
    !
    !     ENP=ZERO
    NCyz=NCLy*NCLz
    DCEL2=DCEL*DCEL
    KAPPAB=KAPPA2*DCEL2
    !
    !     Nonpolar Part - the surface of the solute for Solvation Free Energy

    IF(STCO.GT.ZERO) THEN
       do k=2,nclx-1
          ipx=(k-1)*ncyz
          do l=2,ncly-1
             ipy=(l-1)*nclz
             do m=2,nclz-1
                ipz=ipx+ipy+m

                depsx=(rmayx(ipz)-rmayx(ipz-ncyz))/(epsw-one)
                depsy=(rmayy(ipz)-rmayy(ipz-nclz))/(epsw-one)
                depsz=(rmayz(ipz)-rmayz(ipz-1))/(epsw-one)
                moeps=sqrt(depsx**2+depsy**2+depsz**2)
                Enp=Enp+stco*moeps*dcel2

             enddo
          enddo
       enddo
    ENDIF
    !
    DO IL=1,NTPRP
       I=LSTPRP(IL)
       XI=X(I)+TRANX-xbcen
       YI=Y(I)+TRANY-ybcen
       ZI=Z(I)+TRANZ-zbcen
       BISQ=PBRAD(I)
       NFIL=INT((BISQ+swin)/dcel)+2
       IX=INT(XI/DCEL)+1
       IY=INT(YI/DCEL)+1
       IZ=INT(ZI/DCEL)+1
       bms=bisq-swin
       bps=bisq+swin
       !***
       if(pbrad(i).ne.ZERO) then
          !
          JX1=IX-NFIL
          IF(JX1.LT.1)JX1=1
          JX2=IX+NFIL+1
          IF(JX2.GT.NCLx)JX2=NCLx
          JY1=IY-NFIL
          IF(JY1.LT.1)JY1=1
          JY2=IY+NFIL+1
          IF(JY2.GT.NCLy)JY2=NCLy
          JZ1=IZ-NFIL
          IF(JZ1.LT.1)JZ1=1
          JZ2=IZ+NFIL+1
          IF(JZ2.GT.NCLz)JZ2=NCLz
          !
          DO K=jx1,jx2
             IPX=(K-1)*NCyz
             XC=(K-1)*DCEL
             !
             DO L=jy1,jy2
                IPY=(L-1)*NCLz
                YC=(L-1)*DCEL
                !
                DO M=jz1,jz2
                   IPZ=M+IPY+IPX
                   ZC=(M-1)*DCEL
                   !
                   depsx=rmayx(ipz)-rmayx(ipz-ncyz)
                   depsy=rmayy(ipz)-rmayy(ipz-nclz)
                   depsz=rmayz(ipz)-rmayz(ipz-1)
                   moeps=sqrt(depsx**2+depsy**2+depsz**2)
                   !
                   rxp=sqrt((xc+dcel/2.-xi)**2+(yc-yi)**2+(zc-zi)**2)
                   rxm=sqrt((xc-dcel/2.-xi)**2+(yc-yi)**2+(zc-zi)**2)
                   ryp=sqrt((xc-xi)**2+(yc+dcel/2.-yi)**2+(zc-zi)**2)
                   rym=sqrt((xc-xi)**2+(yc-dcel/2.-yi)**2+(zc-zi)**2)
                   rzp=sqrt((xc-xi)**2+(yc-yi)**2+(zc+dcel/2.-zi)**2)
                   rzm=sqrt((xc-xi)**2+(yc-yi)**2+(zc-dcel/2.-zi)**2)
                   !
                   dx=xc-xi
                   dy=yc-yi
                   dz=zc-zi
                   !
                   ! x-direction
                   drp=rxp-bms
                   drm=rxm-bms
                   dxp=xc+dcel/2.-xi
                   dxm=xc-dcel/2.-xi
                   !
                   !     (rxm.lt.bps) : to consider only atom j among other atoms in a molecule
                   !
                   if((rmayx(ipz-ncyz).ne.epsp).and. &
                        (rmayx(ipz-ncyz).ne.epsw).and.(rxm.lt.bps)) then
                      prefac1= depssf(drm,swin)*(rmayx(ipz-ncyz)-epsp) &
                           /rxm/epssf(drm,swin)
                      ! DBF part
                      prefac2= CCELEC/(eight*pi)*dcel*prefac1* &
                           phi1(ipz)*(phi1(ipz-ncyz)-phi1(ipz))
                      DBFx(i)=DBFx(i)+ prefac2*dxm
                      DBFy(i)=DBFy(i)+ prefac2*dy
                      DBFz(i)=DBFz(i)+ prefac2*dz
                      ! NPF part
                      if((moeps.gt.1.0e-8).and.(STCO.GT.ZERO)) then
                         prefac3= stco*dcel2*depsx/moeps* &
                              prefac1/(epsw-epsp)
                         NPFx(i)=NPFx(i)- prefac3*dxm
                         NPFy(i)=NPFy(i)- prefac3*dy
                         NPFz(i)=NPFz(i)- prefac3*dz
                      endif
                   endif
                   !
                   if((rmayx(ipz).ne.epsp).and. &
                        (rmayx(ipz).ne.epsw).and.(rxp.lt.bps)) then
                      prefac1= depssf(drp,swin)*(rmayx(ipz)-epsp) &
                           /rxp/epssf(drp,swin)
                      ! DBF part
                      prefac2= CCELEC/(eight*pi)*dcel*prefac1* &
                           phi1(ipz)*(phi1(ipz+ncyz)-phi1(ipz))
                      DBFx(i)=DBFx(i)+ prefac2*dxp
                      DBFy(i)=DBFy(i)+ prefac2*dy
                      DBFz(i)=DBFz(i)+ prefac2*dz
                      ! NPF part
                      if((moeps.gt.1.0e-8).and.(STCO.GT.ZERO)) then
                         prefac3= stco*dcel2*depsx/moeps* &
                              prefac1/(epsw-epsp)
                         NPFx(i)=NPFx(i)+ prefac3*dxp
                         NPFy(i)=NPFy(i)+ prefac3*dy
                         NPFz(i)=NPFz(i)+ prefac3*dz
                      endif
                   endif
                   !
                   ! y-direction
                   drp=ryp-bms
                   drm=rym-bms
                   dyp=yc+dcel/2.-yi
                   dym=yc-dcel/2.-yi
                   !
                   if((rmayy(ipz-nclz).ne.epsp).and. &
                        (rmayy(ipz-nclz).ne.epsw).and.(rym.lt.bps)) then
                      prefac1= depssf(drm,swin)*(rmayy(ipz-nclz)-epsp) &
                           /rym/epssf(drm,swin)
                      ! DBF part
                      prefac2=  CCELEC/(eight*pi)*dcel*prefac1* &
                           phi1(ipz)*(phi1(ipz-nclz)-phi1(ipz))
                      DBFx(i)=DBFx(i)+ prefac2*dx
                      DBFy(i)=DBFy(i)+ prefac2*dym
                      DBFz(i)=DBFz(i)+ prefac2*dz
                      ! NPF part
                      if((moeps.gt.1.0e-8).and.(STCO.GT.ZERO)) then
                         prefac3= stco*dcel2*depsy/moeps* &
                              prefac1/(epsw-epsp)
                         NPFx(i)=NPFx(i)- prefac3*dx
                         NPFy(i)=NPFy(i)- prefac3*dym
                         NPFz(i)=NPFz(i)- prefac3*dz
                      endif
                   endif
                   !
                   if((rmayy(ipz).ne.epsp).and. &
                        (rmayy(ipz).ne.epsw).and.(ryp.lt.bps)) then
                      prefac1= depssf(drp,swin)*(rmayy(ipz)-epsp) &
                           /ryp/epssf(drp,swin)
                      ! DBF part
                      prefac2= CCELEC/(eight*pi)*dcel*prefac1* &
                           phi1(ipz)*(phi1(ipz+nclz)-phi1(ipz))
                      DBFx(i)=DBFx(i)+ prefac2*dx
                      DBFy(i)=DBFy(i)+ prefac2*dyp
                      DBFz(i)=DBFz(i)+ prefac2*dz
                      ! NPF part
                      if((moeps.gt.1.0e-8).and.(STCO.GT.ZERO)) then
                         prefac3= stco*dcel2*depsy/moeps* &
                              prefac1/(epsw-epsp)
                         NPFx(i)=NPFx(i)+ prefac3*dx
                         NPFy(i)=NPFy(i)+ prefac3*dyp
                         NPFz(i)=NPFz(i)+ prefac3*dz
                      endif
                   endif
                   !
                   ! z-direction
                   drp=rzp-bms
                   drm=rzm-bms
                   dzp=zc+dcel/2.-zi
                   dzm=zc-dcel/2.-zi
                   !
                   if((rmayz(ipz-1).ne.epsp).and. &
                        (rmayz(ipz-1).ne.epsw).and.(rzm.lt.bps)) then
                      prefac1= depssf(drm,swin)*(rmayz(ipz-1)-epsp) &
                           /rzm/epssf(drm,swin)
                      ! DBF part
                      prefac2= CCELEC/(eight*pi)*dcel*prefac1* &
                           phi1(ipz)*(phi1(ipz-1)-phi1(ipz))
                      DBFx(i)=DBFx(i)+ prefac2*dx
                      DBFy(i)=DBFy(i)+ prefac2*dy
                      DBFz(i)=DBFz(i)+ prefac2*dzm
                      ! NPF part
                      if((moeps.gt.1.0e-8).and.(STCO.GT.ZERO)) then
                         prefac3= stco*dcel2*depsz/moeps* &
                              prefac1/(epsw-epsp)
                         NPFx(i)=NPFx(i)- prefac3*dx
                         NPFy(i)=NPFy(i)- prefac3*dy
                         NPFz(i)=NPFz(i)- prefac3*dzm
                      endif
                   endif
                   !
                   if((rmayz(ipz).ne.epsp).and. &
                        (rmayz(ipz).ne.epsw).and.(rzp.lt.bps)) then
                      prefac1= depssf(drp,swin)*(rmayz(ipz)-epsp) &
                           /rzp/epssf(drp,swin)
                      ! DBF part
                      prefac2= CCELEC/(eight*pi)*dcel*prefac1* &
                           phi1(ipz)*(phi1(ipz+1)-phi1(ipz))
                      DBFx(i)=DBFx(i)+ prefac2*dx
                      DBFy(i)=DBFy(i)+ prefac2*dy
                      DBFz(i)=DBFz(i)+ prefac2*dzp
                      ! NPF part
                      if((moeps.gt.1.0e-8).and.(STCO.GT.ZERO)) then
                         prefac3= stco*dcel2*depsz/moeps* &
                              prefac1/(epsw-epsp)
                         NPFx(i)=NPFx(i)+ prefac3*dx
                         NPFy(i)=NPFy(i)+ prefac3*dy
                         NPFz(i)=NPFz(i)+ prefac3*dzp
                      endif
                   endif
                   !
                   ! IBF part
                   rij=sqrt((xc-xi)**2+(yc-yi)**2+(zc-zi)**2)
                   dr=rij-bms
                   !
                   if((rmay(ipz).lt.kappab).and. &
                        (rmay(ipz).gt.zero).and.(rij.lt.bps)) then
                      !cwi,bug     $               (rmay(ipz).gt.zero).and.(dr.lt.bps)) then
                      prefac1= depssf(dr,swin)*rmay(ipz)/rij &
                           /epssf(dr,swin)
                      prefac4= CCELEC/(eight*pi)*phi1(ipz)**2*dcel*prefac1
                      IBFx(i)=IBFx(i)- prefac4*dx
                      IBFy(i)=IBFy(i)- prefac4*dy
                      IBFz(i)=IBFz(i)- prefac4*dz
                      !                     write(6,*) k,l,m,rmay(ipz),kappab,ibfx(i)
                   endif
                   !
                enddo
                !
             enddo
             !
          enddo
          !***
       endif
       !
    ENDDO

    IF (.not. QPRIN) RETURN

    IF(STCO.GT.ZERO) THEN
       if(prnlev.ge.2) write(outu,'(/,3x,a,F13.5,a)') &
            'Molecular Surface = ', Enp/stco,' [A^2]'
       if(prnlev.ge.2) write(outu,'(3X,A,F13.5,A)') &
            'The Nonpolar Solvation Free Energy =',Enp,' [KCAL/MOL]'
    ENDIF

    IF(PRNLEV.GT.6) THEN  
       IF(STCO.GT.ZERO) THEN
          if(prnlev.ge.2) write(outu,'(/,3x,A)') &
               'The Nonpolar Surface Forces [KCAL/MOL/A] at Each Atom'
          if(prnlev.ge.2) write(outu,'(3x,a)')   &
               '# ATOM       Fx        Fy        Fz'
          do il=1,ntprp
             i=lstprp(il)
             if(prnlev.ge.2) write(outu,'(3x,i5,4x,3f10.5)') &
                  i,npfx(i),npfy(i),npfz(i)
          enddo
       ENDIF
    ENDIF
    !
    RETURN

  END SUBROUTINE BFORCE2
  !
  SUBROUTINE BFORCE3(NTPRP,LSTPRP,X,Y,Z,PBRAD,EPSW,EPSP, &
       NCLX,NCLY,NCLZ,DCEL,PHI1,PHI2,RMAYX,RMAYY,RMAYZ,SWIN, &
       TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
       DBFX,DBFY,DBFZ,IBFX,IBFY,IBFZ, &
       RMAY,KAPPA2)
    !    $           RMAY,KAPPA2,NPFX,NPFY,NPFZ,STCO,Enp,QPRIN)
    !-----------------------------------------------------------------------
    !     This subroutine computes electrostatic solvation forces resulting
    !     from the dielectric bounary and ionic boundary.
    !     And nonpolar solvation forces also are calculated, if necessary.
    !     Dielectric Boundary Force - DBF
    !     Ionic Boundary Force - IBF
    !     NonPolar solvation Force - NPF
    !     van der Waals surface
    !

    use consta
    use number
    use stream

    implicit none

    integer ntprp, lstprp(*)
    REAL(chm_real)  X(*),Y(*),Z(*),PBRAD(*)
    REAL(chm_real4)  RMAYX(*),RMAYY(*),RMAYZ(*),RMAY(*)
    REAL(chm_real)  DBFX(*),DBFY(*),DBFZ(*)
    REAL(chm_real)  IBFX(*),IBFY(*),IBFZ(*)
    INTEGER NCLX,NCLY,NCLZ
    REAL(chm_real)  DCEL,TRANX,TRANY,TRANZ,EPSW,EPSP,KAPPA2
    REAL(chm_real)  XBCEN,YBCEN,ZBCEN
    REAL(chm_real4)  PHI1(*),PHI2(*)
    !     LOGICAL QPRIN
    !   local
    REAL(chm_real)    xc,yc,zc,xi,yi,zi,bisq
    INTEGER   IL,K,L,M,IPX,IPY,IPZ,NCyz,I,nfil
    INTEGER   ix,iy,iz,jx1,jx2,jy1,jy2,jz1,jz2
    !
    REAL(chm_real)    drp,drm,dxp,dxm,dyp,dym,dzp,dzm,dx,dy,dz
    REAL(chm_real)    rxp,rxm,ryp,rym,rzp,rzm,swin,bms,bps
    real(chm_real)    prefac1,prefac2,prefac3,prefac4
    !     real(chm_real)    Enp,depsx,depsy,depsz,moeps
    real(chm_real)    dcel2,rij,dr
    real(chm_real4)    kappab
    !
    !
    NCyz=NCLy*NCLz
    DCEL2=DCEL*DCEL
    KAPPAB=KAPPA2*DCEL2
    !
    DO IL=1,NTPRP
       I=LSTPRP(IL)
       XI=X(I)+TRANX-xbcen
       YI=Y(I)+TRANY-ybcen
       ZI=Z(I)+TRANZ-zbcen
       BISQ=PBRAD(I)
       NFIL=INT((BISQ+swin)/dcel)+2
       IX=INT(XI/DCEL)+1
       IY=INT(YI/DCEL)+1
       IZ=INT(ZI/DCEL)+1
       bms=bisq-swin
       bps=bisq+swin
       !***
       if(pbrad(i).ne.ZERO) then
          !
          JX1=IX-NFIL
          IF(JX1.LT.1)JX1=1
          JX2=IX+NFIL+1
          IF(JX2.GT.NCLx)JX2=NCLx
          JY1=IY-NFIL
          IF(JY1.LT.1)JY1=1
          JY2=IY+NFIL+1
          IF(JY2.GT.NCLy)JY2=NCLy
          JZ1=IZ-NFIL
          IF(JZ1.LT.1)JZ1=1
          JZ2=IZ+NFIL+1
          IF(JZ2.GT.NCLz)JZ2=NCLz
          !
          DO K=jx1,jx2
             IPX=(K-1)*NCyz
             XC=(K-1)*DCEL
             !
             DO L=jy1,jy2
                IPY=(L-1)*NCLz
                YC=(L-1)*DCEL
                !
                DO M=jz1,jz2
                   IPZ=M+IPY+IPX
                   ZC=(M-1)*DCEL
                   !
                   rxp=sqrt((xc+dcel/2.-xi)**2+(yc-yi)**2+(zc-zi)**2)
                   rxm=sqrt((xc-dcel/2.-xi)**2+(yc-yi)**2+(zc-zi)**2)
                   ryp=sqrt((xc-xi)**2+(yc+dcel/2.-yi)**2+(zc-zi)**2)
                   rym=sqrt((xc-xi)**2+(yc-dcel/2.-yi)**2+(zc-zi)**2)
                   rzp=sqrt((xc-xi)**2+(yc-yi)**2+(zc+dcel/2.-zi)**2)
                   rzm=sqrt((xc-xi)**2+(yc-yi)**2+(zc-dcel/2.-zi)**2)
                   !
                   dx=xc-xi
                   dy=yc-yi
                   dz=zc-zi
                   !
                   ! x-direction
                   drp=rxp-bms
                   drm=rxm-bms
                   dxp=xc+dcel/2.-xi
                   dxm=xc-dcel/2.-xi
                   !
                   !     (rxm.lt.bps) : to consider only atom j among other atoms in a molecule
                   !
                   if((rmayx(ipz-ncyz).ne.epsp).and. &
                        (rmayx(ipz-ncyz).ne.epsw).and.(rxm.lt.bps)) then
                      prefac1= depssf(drm,swin)*(rmayx(ipz-ncyz)-epsp) &
                           /rxm/epssf(drm,swin)
                      ! DBF part
                      prefac2= CCELEC/(eight*pi)*dcel*prefac1* &
                           phi1(ipz)*(phi2(ipz-ncyz)-phi2(ipz))
                      DBFx(i)=DBFx(i)+ prefac2*dxm
                      DBFy(i)=DBFy(i)+ prefac2*dy
                      DBFz(i)=DBFz(i)+ prefac2*dz
                   endif
                   !
                   if((rmayx(ipz).ne.epsp).and. &
                        (rmayx(ipz).ne.epsw).and.(rxp.lt.bps)) then
                      prefac1= depssf(drp,swin)*(rmayx(ipz)-epsp) &
                           /rxp/epssf(drp,swin)
                      ! DBF part
                      prefac2= CCELEC/(eight*pi)*dcel*prefac1* &
                           phi1(ipz)*(phi2(ipz+ncyz)-phi2(ipz))
                      DBFx(i)=DBFx(i)+ prefac2*dxp
                      DBFy(i)=DBFy(i)+ prefac2*dy
                      DBFz(i)=DBFz(i)+ prefac2*dz
                   endif
                   !
                   ! y-direction
                   drp=ryp-bms
                   drm=rym-bms
                   dyp=yc+dcel/2.-yi
                   dym=yc-dcel/2.-yi
                   !
                   if((rmayy(ipz-nclz).ne.epsp).and. &
                        (rmayy(ipz-nclz).ne.epsw).and.(rym.lt.bps)) then
                      prefac1= depssf(drm,swin)*(rmayy(ipz-nclz)-epsp) &
                           /rym/epssf(drm,swin)
                      ! DBF part
                      prefac2=  CCELEC/(eight*pi)*dcel*prefac1* &
                           phi1(ipz)*(phi2(ipz-nclz)-phi2(ipz))
                      DBFx(i)=DBFx(i)+ prefac2*dx
                      DBFy(i)=DBFy(i)+ prefac2*dym
                      DBFz(i)=DBFz(i)+ prefac2*dz
                   endif
                   !
                   if((rmayy(ipz).ne.epsp).and. &
                        (rmayy(ipz).ne.epsw).and.(ryp.lt.bps)) then
                      prefac1= depssf(drp,swin)*(rmayy(ipz)-epsp) &
                           /ryp/epssf(drp,swin)
                      ! DBF part
                      prefac2= CCELEC/(eight*pi)*dcel*prefac1* &
                           phi1(ipz)*(phi2(ipz+nclz)-phi2(ipz))
                      DBFx(i)=DBFx(i)+ prefac2*dx
                      DBFy(i)=DBFy(i)+ prefac2*dyp
                      DBFz(i)=DBFz(i)+ prefac2*dz
                   endif
                   !
                   ! z-direction
                   drp=rzp-bms
                   drm=rzm-bms
                   dzp=zc+dcel/2.-zi
                   dzm=zc-dcel/2.-zi
                   !
                   if((rmayz(ipz-1).ne.epsp).and. &
                        (rmayz(ipz-1).ne.epsw).and.(rzm.lt.bps)) then
                      prefac1= depssf(drm,swin)*(rmayz(ipz-1)-epsp) &
                           /rzm/epssf(drm,swin)
                      ! DBF part
                      prefac2= CCELEC/(eight*pi)*dcel*prefac1* &
                           phi1(ipz)*(phi2(ipz-1)-phi2(ipz))
                      DBFx(i)=DBFx(i)+ prefac2*dx
                      DBFy(i)=DBFy(i)+ prefac2*dy
                      DBFz(i)=DBFz(i)+ prefac2*dzm
                   endif
                   !
                   if((rmayz(ipz).ne.epsp).and. &
                        (rmayz(ipz).ne.epsw).and.(rzp.lt.bps)) then
                      prefac1= depssf(drp,swin)*(rmayz(ipz)-epsp) &
                           /rzp/epssf(drp,swin)
                      ! DBF part
                      prefac2= CCELEC/(eight*pi)*dcel*prefac1* &
                           phi1(ipz)*(phi2(ipz+1)-phi2(ipz))
                      DBFx(i)=DBFx(i)+ prefac2*dx
                      DBFy(i)=DBFy(i)+ prefac2*dy
                      DBFz(i)=DBFz(i)+ prefac2*dzp
                   endif
                   !
                   ! IBF part
                   rij=sqrt((xc-xi)**2+(yc-yi)**2+(zc-zi)**2)
                   dr=rij-bms
                   !
                   if((rmay(ipz).lt.kappab).and. &
                        (rmay(ipz).gt.zero).and.(rij.lt.bps)) then
                      !cwi,bug     $               (rmay(ipz).gt.zero).and.(dr.lt.bps)) then
                      prefac1= depssf(dr,swin)*rmay(ipz)/rij &
                           /epssf(dr,swin)
                      prefac4= CCELEC/(eight*pi)*phi1(ipz)*phi2(ipz)*dcel &
                           *prefac1
                      IBFx(i)=IBFx(i)- prefac4*dx
                      IBFy(i)=IBFy(i)- prefac4*dy
                      IBFz(i)=IBFz(i)- prefac4*dz
                      !                     write(6,*) k,l,m,rmay(ipz),kappab,ibfx(i)
                   endif
                   !
                enddo
                !
             enddo
             !
          enddo
          !***
       endif
       !
    ENDDO

    RETURN

  END SUBROUTINE BFORCE3
#endif 
  !! gsbp.src and gsbp2.src inserted here to avoid circular dependencies. LNI, November 2009

  SUBROUTINE GSBP4(NTRB,LSTRB,EGSBPA,EGSBPB,EGSBP,DX,DY,DZ, &
       RXNAFX,RXNAFY,RXNAFZ,RXNBFX,RXNBFY,RXNBFZ,QPRIN &
#if KEY_GCMC==1
       ,GCMCON            & 
#endif
       )
    !-----------------------------------------------------------------------
    !     Generalized Solvent Boundary Potential working area
    !     Each first derivative (-force) is added to DX, DY, DZ.
    !
#if KEY_GSBP==0 /*gsbp*/
    CALL WRNDIE(-1,'<CHARMM>','GSBP code is not compiled.')
    RETURN
  END SUBROUTINE GSBP4
#else /* (gsbp)*/
    use chm_kinds
    use stream
#if KEY_GCMC==1
    use dimens_fcm
    use coord
#endif 
    !     QC_UW0405: We add a simple normalization for parallel run - which
    !     only would occur with PARAQCTMP
#if KEY_PARASCC==1
    use parallel  
#endif
    implicit none
    !
    real(chm_real)   DX(*),DY(*),DZ(*)
    real(chm_real)   RXNAFX(*),RXNAFY(*),RXNAFZ(*)
    real(chm_real)   RXNBFX(*),RXNBFY(*),RXNBFZ(*)
    real(chm_real)   EGSBPA,EGSBPB,EGSBP
    INTEGER  NATOM,NTRB,LSTRB(*)
    LOGICAL  QPRIN
#if KEY_GCMC==1
    LOGICAL  GCMCON(:)
#endif 
    !     QC_UW0405:
#if KEY_PARASCC==1
    real(chm_real)   SCALPARA 
#endif
    ! local
    INTEGER  I,J
    real(chm_real)   TMPFX,TMPFY,TMPFZ

#if KEY_PARASCC==1
    SCALPARA=1.d0/NUMNOD 
#endif
#if KEY_GCMC==1
    DO I = 1, NTRB
       J = LSTRB(I)
       IF (GCMCON(J)) THEN
#if KEY_PARASCC==1
          DX(J) = DX(J) - (RXNAFX(J) + RXNBFX(J))*SCALPARA
          DY(J) = DY(J) - (RXNAFY(J) + RXNBFY(J))*SCALPARA
          DZ(J) = DZ(J) - (RXNAFZ(J) + RXNBFZ(J))*SCALPARA
#else /**/
          DX(J) = DX(J) - RXNAFX(J) - RXNBFX(J)
          DY(J) = DY(J) - RXNAFY(J) - RXNBFY(J)
          DZ(J) = DZ(J) - RXNAFZ(J) - RXNBFZ(J)
#endif 
       ENDIF
    ENDDO
#endif 
    !
    EGSBP = EGSBPA + EGSBPB

    !
    IF(QPRIN) THEN
       WRITE(OUTU,'(A)')
       WRITE(OUTU,'(3X,A,F13.5,A)') &
            'Electrostatic interaction in solvent   [G_IO(80)]=', &
            EGSBPA,' [KCAL/MOL]'
       WRITE(OUTU,'(3X,A,/,3X,A,F13.5,A)') &
            'Reaction field free energy in inner region ', &
            'calculated from the basis functions        [G_II]=', &
            EGSBPB,' [KCAL/MOL]'
       WRITE(OUTU,'(3X,A,F13.5,A)') &
            'Electrostatic energy in region I  [G_IO(80)+G_II]=', &
            EGSBP,' [KCAL/MOL]'
    ENDIF
    !
#if KEY_PARASCC==1
    EGSBP = EGSBP*SCALPARA 
#endif
    RETURN
  END  SUBROUTINE GSBP4
  !
  SUBROUTINE INITFORCE(NTRB,LSTRB,FORCE)
    !-----------------------------------------------------------------------
    !     Initialize a FORCE
    !
    use chm_kinds
    use number
    implicit none
    !
    real(chm_real)   FORCE(*)
    INTEGER  NTRB,LSTRB(*)
    ! local
    INTEGER  I,J
    DO I = 1, NTRB
       J = LSTRB(I)
       FORCE(J)=ZERO
    ENDDO
    !
    RETURN
  END  SUBROUTINE INITFORCE
  !
  SUBROUTINE GDECOMP(NATOM,X,Y,Z,CG, &
       LNCEL,LTRAN,LDCEL,LXBCEN,LYBCEN,LZBCEN,QPRIN)
    !-----------------------------------------------------------------------
    !     The Decomposition of Electrostatic Solvation Free Energy in GSBP
    !
    use chm_kinds
    use memory
    use number
    use stream
    use dimens_fcm
    use consta
    use timerm
    implicit none
    !
    INTEGER NATOM
    real(chm_real)  X(*),Y(*),Z(*),CG(*)
    LOGICAL QPRIN
    INTEGER LNCEL
    real(chm_real)  LTRAN,LDCEL,LXBCEN,LYBCEN,LZBCEN
    ! local
    real(chm_real4),allocatable,dimension(:) :: ILCHC,IPHIB
    INTEGER NFIL,NFILZ,IXCEN,IYCEN,IZCEN,JX1,JX2,JY1,JY2,JZ1,JZ2
    INTEGER LNCEL3,BNCEL
    !
    IF(QGTOT.OR.QGAA.OR.QGBB.OR.QGAB) WRITE(OUTU,'(A)')
    !
    IF(QRECTBOX) THEN
       JX1=INT((TRANX+RBXMIN-XBCEN)/DCEL)-1
       JX2=INT((TRANX+RBXMAX-XBCEN)/DCEL)+3
       JY1=INT((TRANY+RBYMIN-YBCEN)/DCEL)-1
       JY2=INT((TRANY+RBYMAX-YBCEN)/DCEL)+3
       JZ1=INT((TRANZ+RBZMIN-ZBCEN)/DCEL)-1
       JZ2=INT((TRANZ+RBZMAX-ZBCEN)/DCEL)+3
    ELSEIF(QSPHERE) THEN
       NFIL=INT(SRDIST/DCEL)+2
       IXCEN=INT((RRXCEN+TRANX-XBCEN)/DCEL)+1
       IYCEN=INT((RRYCEN+TRANY-YBCEN)/DCEL)+1
       IZCEN=INT((RRZCEN+TRANZ-ZBCEN)/DCEL)+1
       JX1=IXCEN-NFIL
       JX2=IXCEN+NFIL+1
       JY1=IYCEN-NFIL
       JY2=IYCEN+NFIL+1
       JZ1=IZCEN-NFIL
       JZ2=IZCEN+NFIL+1
    ELSEIF(QPROLATE) THEN
       NFIL =INT(MINOR/DCEL)+2
       NFILZ=INT(MAJOR/DCEL)+2
       IXCEN=INT((RRXCEN+TRANX-XBCEN)/DCEL)+1
       IYCEN=INT((RRYCEN+TRANY-YBCEN)/DCEL)+1
       IZCEN=INT((RRZCEN+TRANZ-ZBCEN)/DCEL)+1
       JX1=IXCEN-NFIL
       JX2=IXCEN+NFIL+1
       JY1=IYCEN-NFIL
       JY2=IYCEN+NFIL+1
       JZ1=IZCEN-NFILZ
       JZ2=IZCEN+NFILZ+1
    ENDIF

    !     allocate charge densities (large box) and boundary potentials (original)
    IF(QLBOX.AND.QFOCUS) THEN
       LNCEL3=LNCEL*LNCEL*LNCEL
       BNCEL=2*(NCLX*NCLX+NCLY*NCLY+NCLZ*NCLZ)
       call chmalloc('pbeq.src','GDECOMP','ILCHC',LNCEL3,cr4=ILCHC)
       call chmalloc('pbeq.src','GDECOMP','IPHIB',BNCEL,cr4=IPHIB)
    ELSE
       call chmalloc('pbeq.src','GDECOMP','IPHIB',1,cr4=IPHIB)
    ENDIF

    !----------------------------------------------------------------------------
    !     In vacuum calculations, vmemb, kappa2 (salt concentration) = 0.0
    !                             epsw, epsm, epsh, epsd, and epsc = 1.0
    !     However, TMEMB is not equal to zero to consider periodic conditions.
    !----------------------------------------------------------------------------

    !
    !     DELTA G_TOT
    !

    IF(QGTOT) THEN
       QA=.false.
       QB=.false.

       ! In vacuum
       IF(QLBOX.AND.QFOCUS) THEN
          !     get the grid parameters in the lagre box
          CALL MAYER(NTPRP,LSTPRP,ONE,ONE,ZERO,X,Y,Z,CG, &
               PBRAD,LNCEL,LNCEL,LNCEL,LDCEL,WATR,IONR,MAY, &
               MAYX,MAYY,MAYZ,SWIN,ILCHC, &
               IPHIP,ZERO,TMEMB,ZMEMB,ONE,NIMGB, &
               HTMEMB,ONE,HDENS,POSALP,NEGALP,QCHRG,POFFST,NOFFST, &
               DROPLET,ONE,XDROPLET,YDROPLET,ZDROPLET,Qdtom,Qdkap, &
               ONE,LXMAX,LYMAX,LZMAX,LXMIN,LYMIN,LZMIN,Qbtom,Qbkap, &
               ONE,Xcyln,Ycyln,Zcyln,Rcyln,Hcyln,Qctom,Qckap, &
               ONE,Xcone,Ycone,Zcone,Ax,By,Cz,Qectom,Qeckap, &
               -1,-1,&
               LTRAN,LTRAN,LTRAN,LXBCEN,LYBCEN,LZBCEN, &
               IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
               QKEEP,QREEN,QSMTH,QBSPL,QA,QB, &
               RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
               RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
               MAJOR,MINOR,QPROLATE, &
               QINTBP,QZERO,.false.,IPHIB,QPBC,QNPBC, &
               QNONLINEAR,QPARTLINEAR,QFKAP, &
               IPOX,IPOY,IPOZ,MAPT, NATOM)
          IF(EPSP.ne.ONE) THEN
             IF(QRECTBOX) THEN
                CALL RECT_MAYER(LNCEL,LNCEL,LNCEL,LTRAN,LTRAN,LTRAN,LDCEL, &
                     LXBCEN,LYBCEN,LZBCEN, &
                     MAY,MAYX,MAYY,MAYZ, &
                     RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX)
             ELSEIF(QSPHERE) THEN
                CALL SPHE_MAYER(LNCEL,LNCEL,LNCEL,LTRAN,LTRAN,LTRAN,LDCEL, &
                     LXBCEN,LYBCEN,LZBCEN, &
                     MAY,MAYX,MAYY,MAYZ, &
                     RRXCEN,RRYCEN,RRZCEN,SRDIST)
             ELSEIF(QPROLATE) THEN
                CALL PROL_MAYER(LNCEL,LNCEL,LNCEL,LTRAN,LTRAN,LTRAN,LDCEL, &
                     LXBCEN,LYBCEN,LZBCEN, &
                     MAY,MAYX,MAYY,MAYZ, &
                     RRXCEN,RRYCEN,RRZCEN,MAJOR,MINOR)
             ENDIF
             IXMAX=LNCEL-1
             IYMAX=LNCEL-1
             IZMAX=LNCEL-1
             IXMIN=2
             IYMIN=2
             IZMIN=2
          ENDIF
          !     solve PBEQ in the large box
          CALL PBEQ1(MAXITS,DOME,DEPS,ONE,ONE,ONE,ZERO, &
               LNCEL,LNCEL,LNCEL,IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
               IPHIP,MAYX,MAYY,MAYZ, &
               ILCHC,MAY,TMEMB, &
               .false.,.false.,QPBC,QNPBC,.false.)
          !     construct boundary potentials from the large box
          CALL BOUNDPOT(LNCEL,LNCEL,LNCEL,LDCEL, &
               LTRAN,LTRAN,LTRAN,LXBCEN,LYBCEN,LZBCEN, &
               NCLX,NCLY,NCLZ,DCEL,XBCEN,YBCEN,ZBCEN, &
               IPHIP,IPHIB)

          CALL MAYER(NTPRP,LSTPRP,ONE,ONE,ZERO,X,Y,Z,CG, &
               PBRAD,NCLX,NCLY,NCLZ,DCEL,WATR,IONR,MAY, &
               MAYX,MAYY,MAYZ,SWIN,ICHC, &
               IPHIP,ZERO,TMEMB,ZMEMB,ONE,NIMGB, &
               HTMEMB,ONE,HDENS,POSALP,NEGALP,QCHRG,POFFST,NOFFST, &
               DROPLET,ONE,XDROPLET,YDROPLET,ZDROPLET,Qdtom,Qdkap, &
               ONE,LXMAX,LYMAX,LZMAX,LXMIN,LYMIN,LZMIN,Qbtom,Qbkap, &
               ONE,Xcyln,Ycyln,Zcyln,Rcyln,Hcyln,Qctom,Qckap, &
               ONE,Xcone,Ycone,Zcone,Ax,By,Cz,Qectom,Qeckap, &
               -1,-1,&
               TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
               IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
               QKEEP,QREEN,QSMTH,QBSPL,QA,QB, &
               RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
               RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
               MAJOR,MINOR,QPROLATE, &
               .false.,.false.,.true.,IPHIB,QPBC,QNPBC, &
               QNONLINEAR,QPARTLINEAR,QFKAP, &
               IPOX,IPOY,IPOZ,MAPT, NATOM)

       ELSE
          CALL MAYER(NTPRP,LSTPRP,ONE,ONE,ZERO,X,Y,Z,CG, &
               PBRAD,NCLX,NCLY,NCLZ,DCEL,WATR,IONR,MAY, &
               MAYX,MAYY,MAYZ,SWIN,ICHC, &
               IPHIP,ZERO,TMEMB,ZMEMB,ONE,NIMGB, &
               HTMEMB,ONE,HDENS,POSALP,NEGALP,QCHRG,POFFST,NOFFST, &
               DROPLET,ONE,XDROPLET,YDROPLET,ZDROPLET,Qdtom,Qdkap, &
               ONE,LXMAX,LYMAX,LZMAX,LXMIN,LYMIN,LZMIN,Qbtom,Qbkap, &
               ONE,Xcyln,Ycyln,Zcyln,Rcyln,Hcyln,Qctom,Qckap, &
               ONE,Xcone,Ycone,Zcone,Ax,By,Cz,Qectom,Qeckap, &
               -1,-1,&
               TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
               IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
               QKEEP,QREEN,QSMTH,QBSPL,QA,QB, &
               RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
               RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
               MAJOR,MINOR,QPROLATE, &
               QINTBP,QZERO,.false.,IPHIB,QPBC,QNPBC, &
               QNONLINEAR,QPARTLINEAR,QFKAP, &
               IPOX,IPOY,IPOZ,MAPT,NATOM)
       ENDIF

       IF(EPSP.ne.ONE) THEN
          ! vacuum environment in region B.
          IF(QRECTBOX) THEN
             CALL RECT_MAYER(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,DCEL, &
                  XBCEN,YBCEN,ZBCEN, &
                  MAY,MAYX,MAYY,MAYZ, &
                  RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX)
          ELSEIF(QSPHERE) THEN
             CALL SPHE_MAYER(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,DCEL, &
                  XBCEN,YBCEN,ZBCEN, &
                  MAY,MAYX,MAYY,MAYZ, &
                  RRXCEN,RRYCEN,RRZCEN,SRDIST)
          ELSEIF(QPROLATE) THEN
             CALL PROL_MAYER(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,DCEL, &
                  XBCEN,YBCEN,ZBCEN, &
                  MAY,MAYX,MAYY,MAYZ, &
                  RRXCEN,RRYCEN,RRZCEN,MAJOR,MINOR)
          ENDIF
       ENDIF
       !
       CALL PBEQ1(MAXITS,DOME,DEPS,ONE,ONE,ONE,ZERO, &
            NCLX,NCLY,NCLZ,IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
            IPHIP,MAYX,MAYY,MAYZ, &
            ICHC,MAY,TMEMB, &
            .false.,QFOCUS,QPBC,QNPBC,.false.)

       ! In solution
       IF(QLBOX.AND.QFOCUS) THEN
          !     get the grid parameters in the lagre box
          CALL MAYER(NTPRP,LSTPRP,EPSW,EPSP,KAPPA2,X,Y,Z,CG, &
               PBRAD,LNCEL,LNCEL,LNCEL,LDCEL,WATR,IONR,MAY, &
               MAYX,MAYY,MAYZ,SWIN,ILCHC, &
               IPHIW,VMEMB,TMEMB,ZMEMB,EPSM,NIMGB, &
               HTMEMB,EPSH,HDENS,POSALP,NEGALP,QCHRG,POFFST,NOFFST, &
               DROPLET,EPSD,XDROPLET,YDROPLET,ZDROPLET,Qdtom,Qdkap, &
               EPSB,LXMAX,LYMAX,LZMAX,LXMIN,LYMIN,LZMIN,Qbtom,Qbkap, &
               EPSC,Xcyln,Ycyln,Zcyln,Rcyln,Hcyln,Qctom,Qckap, &
               EPSEC,Xcone,Ycone,Zcone,Ax,By,Cz,Qectom,Qeckap, &
               -1,-1,&
               LTRAN,LTRAN,LTRAN,LXBCEN,LYBCEN,LZBCEN, &
               IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
               QKEEP,QREEN,QSMTH,QBSPL,QA,QB, &
               RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
               RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
               MAJOR,MINOR,QPROLATE, &
               QINTBP,QZERO,.false.,IPHIB,QPBC,QNPBC, &
               QNONLINEAR,QPARTLINEAR,QFKAP, &
               IPOX,IPOY,IPOZ,MAPT, NATOM) 
          IF(QRECTBOX) THEN
             CALL RECT_MAYER(LNCEL,LNCEL,LNCEL,LTRAN,LTRAN,LTRAN,LDCEL, &
                  LXBCEN,LYBCEN,LZBCEN, &
                  MAY,MAYX,MAYY,MAYZ, &
                  RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX)
          ELSEIF(QSPHERE) THEN
             CALL SPHE_MAYER(LNCEL,LNCEL,LNCEL,LTRAN,LTRAN,LTRAN,LDCEL, &
                  LXBCEN,LYBCEN,LZBCEN, &
                  MAY,MAYX,MAYY,MAYZ, &
                  RRXCEN,RRYCEN,RRZCEN,SRDIST)
          ELSEIF(QPROLATE) THEN
             CALL PROL_MAYER(LNCEL,LNCEL,LNCEL,LTRAN,LTRAN,LTRAN,LDCEL, &
                  LXBCEN,LYBCEN,LZBCEN, &
                  MAY,MAYX,MAYY,MAYZ, &
                  RRXCEN,RRYCEN,RRZCEN,MAJOR,MINOR)
          ENDIF
          IXMAX=LNCEL-1
          IYMAX=LNCEL-1
          IZMAX=LNCEL-1
          IXMIN=2
          IYMIN=2
          IZMIN=2
          !     solve PBEQ in the large box
          CALL PBEQ1(MAXITS,DOME,DEPS,EPSW,EPSP,EPSM,KAPPA2, &
               LNCEL,LNCEL,LNCEL,IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
               IPHIW,MAYX,MAYY,MAYZ, &
               ILCHC,MAY,TMEMB, &
               QOSOR,.false.,QPBC,QNPBC,.false.)

          !     construct boundary potentials from the large box
          CALL BOUNDPOT(LNCEL,LNCEL,LNCEL,LDCEL, &
               LTRAN,LTRAN,LTRAN,LXBCEN,LYBCEN,LZBCEN, &
               NCLX,NCLY,NCLZ,DCEL,XBCEN,YBCEN,ZBCEN, &
               IPHIW,IPHIB)

          CALL MAYER(NTPRP,LSTPRP,EPSW,EPSP,KAPPA2,X,Y,Z,CG, &
               PBRAD,NCLX,NCLY,NCLZ,DCEL,WATR,IONR,MAY, &
               MAYX,MAYY,MAYZ,SWIN,ICHC, &
               IPHIW,VMEMB,TMEMB,ZMEMB,EPSM,NIMGB, &
               HTMEMB,EPSH,HDENS,POSALP,NEGALP,QCHRG,POFFST,NOFFST, &
               DROPLET,EPSD,XDROPLET,YDROPLET,ZDROPLET,Qdtom,Qdkap, &
               EPSB,LXMAX,LYMAX,LZMAX,LXMIN,LYMIN,LZMIN,Qbtom,Qbkap, &
               EPSC,Xcyln,Ycyln,Zcyln,Rcyln,Hcyln,Qctom,Qckap, &
               EPSEC,Xcone,Ycone,Zcone,Ax,By,Cz,Qectom,Qeckap, &
               -1,-1,&
               TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
               IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
               QKEEP,QREEN,QSMTH,QBSPL,QA,QB, &
               RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
               RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
               MAJOR,MINOR,QPROLATE, &
               .false.,.false.,.true.,IPHIB,QPBC,QNPBC, &
               QNONLINEAR,QPARTLINEAR,QFKAP, &
               IPOX,IPOY,IPOZ,MAPT, NATOM)
       ELSE
          CALL MAYER(NTPRP,LSTPRP,EPSW,EPSP,KAPPA2,X,Y,Z,CG, &
               PBRAD,NCLX,NCLY,NCLZ,DCEL,WATR,IONR,MAY, &
               MAYX,MAYY,MAYZ,SWIN,ICHC, &
               IPHIW,VMEMB,TMEMB,ZMEMB,EPSM,NIMGB, &
               HTMEMB,EPSH,HDENS,POSALP,NEGALP,QCHRG,POFFST,NOFFST, &
               DROPLET,EPSD,XDROPLET,YDROPLET,ZDROPLET,Qdtom,Qdkap, &
               EPSB,LXMAX,LYMAX,LZMAX,LXMIN,LYMIN,LZMIN,Qbtom,Qbkap, &
               EPSC,Xcyln,Ycyln,Zcyln,Rcyln,Hcyln,Qctom,Qckap, &
               EPSEC,Xcone,Ycone,Zcone,Ax,By,Cz,Qectom,Qeckap, &
               -1,-1,&
               TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
               IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
               QKEEP,QREEN,QSMTH,QBSPL,QA,QB, &
               RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
               RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
               MAJOR,MINOR,QPROLATE, &
               QINTBP,QZERO,.false.,IPHIB,QPBC,QNPBC, &
               QNONLINEAR,QPARTLINEAR,QFKAP, &
               IPOX,IPOY,IPOZ,MAPT,NATOM)
       ENDIF

       ! vacuum environment in region B.
       IF(QRECTBOX) THEN
          CALL RECT_MAYER(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,DCEL, &
               XBCEN,YBCEN,ZBCEN, &
               MAY,MAYX,MAYY,MAYZ, &
               RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX)
       ELSEIF(QSPHERE) THEN
          CALL SPHE_MAYER(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,DCEL, &
               XBCEN,YBCEN,ZBCEN, &
               MAY,MAYX,MAYY,MAYZ, &
               RRXCEN,RRYCEN,RRZCEN,SRDIST)
       ELSEIF(QPROLATE) THEN
          CALL PROL_MAYER(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,DCEL, &
               XBCEN,YBCEN,ZBCEN, &
               MAY,MAYX,MAYY,MAYZ, &
               RRXCEN,RRYCEN,RRZCEN,MAJOR,MINOR)
       ENDIF

       ! redefine the maximum region outside which EPS is EPSW. (see subroutine MAYER)
       IF(JX2.GT.IXMAX) IXMAX=JX2+1
       IF(JY2.GT.IYMAX) IYMAX=JY2+1
       IF(JZ2.GT.IZMAX) IZMAX=JZ2+1
       IF(JX1.LT.IXMIN) IXMIN=JX1-1
       IF(JY1.LT.IYMIN) IYMIN=JY1-1
       IF(JZ1.LT.IZMIN) IZMIN=JZ1-1
       !
       CALL PBEQ1(MAXITS,DOME,DEPS,EPSW,EPSP,EPSM,KAPPA2, &
            NCLX,NCLY,NCLZ,IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
            IPHIW,MAYX,MAYY,MAYZ, &
            ICHC,MAY,TMEMB, &
            QOSOR,QFOCUS,QPBC,QNPBC,.false.)
       !
       CALL RFORCE(NTPRP,LSTPRP,X,Y,Z,CG,NCLX,NCLY,NCLZ,DCEL, &
            IPHIW,IPHIP, &
            TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN,HALF, &
            RXNAFX,RXNAFY,RXNAFZ,EGSBPA, &
            .false.,QBSPL)

       WRITE(OUTU,'(3X,A,F13.5,A)') &
            'Total electrostatic free energy           [G_TOT]=', &
            EGSBPA,' [KCAL/MOL]'

    ENDIF

    !
    !     DELTA G_AA
    !     NOTE: Vmemb = 0.0 for complex solution environment

    IF(QGAA) THEN
       QA=.false.
       QB=.true.

       ! In vacuum
       IF(QLBOX.AND.QFOCUS) THEN
          !     get the grid parameters in the lagre box
          CALL MAYER(NTPRP,LSTPRP,ONE,ONE,ZERO,X,Y,Z,CG, &
               PBRAD,LNCEL,LNCEL,LNCEL,LDCEL,WATR,IONR,MAY, &
               MAYX,MAYY,MAYZ,SWIN,ILCHC, &
               IPHIP,ZERO,TMEMB,ZMEMB,ONE,NIMGB, &
               HTMEMB,ONE,HDENS,POSALP,NEGALP,QCHRG,POFFST,NOFFST, &
               DROPLET,ONE,XDROPLET,YDROPLET,ZDROPLET,Qdtom,Qdkap, &
               ONE,LXMAX,LYMAX,LZMAX,LXMIN,LYMIN,LZMIN,Qbtom,Qbkap, &
               ONE,Xcyln,Ycyln,Zcyln,Rcyln,Hcyln,Qctom,Qckap, &
               ONE,Xcone,Ycone,Zcone,Ax,By,Cz,Qectom,Qeckap, &
               -1,-1,&
               LTRAN,LTRAN,LTRAN,LXBCEN,LYBCEN,LZBCEN, &
               IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
               QKEEP,QREEN,QSMTH,QBSPL,QA,QB, &
               RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
               RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
               MAJOR,MINOR,QPROLATE, &
               QINTBP,QZERO,.false.,IPHIB,QPBC,QNPBC, &
               QNONLINEAR,QPARTLINEAR,QFKAP, &
               IPOX,IPOY,IPOZ,MAPT, NATOM)
          IF(EPSP.ne.ONE) THEN
             IF(QRECTBOX) THEN
                CALL RECT_MAYER(LNCEL,LNCEL,LNCEL,LTRAN,LTRAN,LTRAN,LDCEL, &
                     LXBCEN,LYBCEN,LZBCEN, &
                     MAY,MAYX,MAYY,MAYZ, &
                     RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX)
             ELSEIF(QSPHERE) THEN
                CALL SPHE_MAYER(LNCEL,LNCEL,LNCEL,LTRAN,LTRAN,LTRAN,LDCEL, &
                     LXBCEN,LYBCEN,LZBCEN, &
                     MAY,MAYX,MAYY,MAYZ, &
                     RRXCEN,RRYCEN,RRZCEN,SRDIST)
             ELSEIF(QPROLATE) THEN
                CALL PROL_MAYER(LNCEL,LNCEL,LNCEL,LTRAN,LTRAN,LTRAN,LDCEL, &
                     LXBCEN,LYBCEN,LZBCEN, &
                     MAY,MAYX,MAYY,MAYZ, &
                     RRXCEN,RRYCEN,RRZCEN,MAJOR,MINOR)
             ENDIF
             IXMAX=LNCEL-1
             IYMAX=LNCEL-1
             IZMAX=LNCEL-1
             IXMIN=2
             IYMIN=2
             IZMIN=2
          ENDIF
          !     solve PBEQ in the large box
          CALL PBEQ1(MAXITS,DOME,DEPS,ONE,ONE,ONE,ZERO, &
               LNCEL,LNCEL,LNCEL,IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
               IPHIP,MAYX,MAYY,MAYZ, &
               ILCHC,MAY,TMEMB, &
               .false.,.false.,QPBC,QNPBC,.false.)
          !     construct boundary potentials from the large box
          CALL BOUNDPOT(LNCEL,LNCEL,LNCEL,LDCEL, &
               LTRAN,LTRAN,LTRAN,LXBCEN,LYBCEN,LZBCEN, &
               NCLX,NCLY,NCLZ,DCEL,XBCEN,YBCEN,ZBCEN, &
               IPHIP,IPHIB)
          CALL MAYER(NTPRP,LSTPRP,ONE,ONE,ZERO,X,Y,Z,CG, &
               PBRAD,NCLX,NCLY,NCLZ,DCEL,WATR,IONR,MAY, &
               MAYX,MAYY,MAYZ,SWIN,ICHC, &
               IPHIP,ZERO,TMEMB,ZMEMB,ONE,NIMGB, &
               HTMEMB,ONE,HDENS,POSALP,NEGALP,QCHRG,POFFST,NOFFST, &
               DROPLET,ONE,XDROPLET,YDROPLET,ZDROPLET,Qdtom,Qdkap, &
               ONE,LXMAX,LYMAX,LZMAX,LXMIN,LYMIN,LZMIN,Qbtom,Qbkap, &
               ONE,Xcyln,Ycyln,Zcyln,Rcyln,Hcyln,Qctom,Qckap, &
               ONE,Xcone,Ycone,Zcone,Ax,By,Cz,Qectom,Qeckap, &
               -1,-1,&
               TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
               IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
               QKEEP,QREEN,QSMTH,QBSPL,QA,QB, &
               RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
               RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
               MAJOR,MINOR,QPROLATE, &
               .false.,.false.,.true.,IPHIB,QPBC,QNPBC, &
               QNONLINEAR,QPARTLINEAR,QFKAP, &
               IPOX,IPOY,IPOZ,MAPT,NATOM)
       ELSE
          CALL MAYER(NTPRP,LSTPRP,ONE,ONE,ZERO,X,Y,Z,CG, &
               PBRAD,NCLX,NCLY,NCLZ,DCEL,WATR,IONR,MAY, &
               MAYX,MAYY,MAYZ,SWIN,ICHC, &
               IPHIP,ZERO,TMEMB,ZMEMB,ONE,NIMGB, &
               HTMEMB,ONE,HDENS,POSALP,NEGALP,QCHRG,POFFST,NOFFST, &
               DROPLET,ONE,XDROPLET,YDROPLET,ZDROPLET,Qdtom,Qdkap, &
               ONE,LXMAX,LYMAX,LZMAX,LXMIN,LYMIN,LZMIN,Qbtom,Qbkap, &
               ONE,Xcyln,Ycyln,Zcyln,Rcyln,Hcyln,Qctom,Qckap, &
               ONE,Xcone,Ycone,Zcone,Ax,By,Cz,Qectom,Qeckap, &
               -1,-1,&
               TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
               IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
               QKEEP,QREEN,QSMTH,QBSPL,QA,QB, &
               RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
               RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
               MAJOR,MINOR,QPROLATE, &
               QINTBP,QZERO,.false.,IPHIB,QPBC,QNPBC, &
               QNONLINEAR,QPARTLINEAR,QFKAP, &
               IPOX,IPOY,IPOZ,MAPT, NATOM)
       ENDIF
       !
       IF(EPSP.ne.ONE) THEN
          ! vacuum environment in region B.
          IF(QRECTBOX) THEN
             CALL RECT_MAYER(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,DCEL, &
                  XBCEN,YBCEN,ZBCEN, &
                  MAY,MAYX,MAYY,MAYZ, &
                  RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX)
          ELSEIF(QSPHERE) THEN
             CALL SPHE_MAYER(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,DCEL, &
                  XBCEN,YBCEN,ZBCEN, &
                  MAY,MAYX,MAYY,MAYZ, &
                  RRXCEN,RRYCEN,RRZCEN,SRDIST)
          ELSEIF(QPROLATE) THEN
             CALL PROL_MAYER(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,DCEL, &
                  XBCEN,YBCEN,ZBCEN, &
                  MAY,MAYX,MAYY,MAYZ, &
                  RRXCEN,RRYCEN,RRZCEN,MAJOR,MINOR)
          ENDIF
       ENDIF
       !
       CALL PBEQ1(MAXITS,DOME,DEPS,ONE,ONE,ONE,ZERO, &
            NCLX,NCLY,NCLZ,IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
            IPHIP,MAYX,MAYY,MAYZ, &
            ICHC,MAY,TMEMB, &
            .false.,QFOCUS,QPBC,QNPBC,.false.)

       ! In solution
       IF(QLBOX.AND.QFOCUS) THEN
          !     get the grid parameters in the lagre box
          CALL MAYER(NTPRP,LSTPRP,EPSW,EPSP,KAPPA2,X,Y,Z,CG, &
               PBRAD,LNCEL,LNCEL,LNCEL,LDCEL,WATR,IONR,MAY, &
               MAYX,MAYY,MAYZ,SWIN,ILCHC, &
               IPHIX,ZERO,TMEMB,ZMEMB,EPSM,NIMGB, &
               HTMEMB,EPSH,HDENS,POSALP,NEGALP,QCHRG,POFFST,NOFFST, &
               DROPLET,EPSD,XDROPLET,YDROPLET,ZDROPLET,Qdtom,Qdkap, &
               EPSB,LXMAX,LYMAX,LZMAX,LXMIN,LYMIN,LZMIN,Qbtom,Qbkap, &
               EPSC,Xcyln,Ycyln,Zcyln,Rcyln,Hcyln,Qctom,Qckap, &
               EPSEC,Xcone,Ycone,Zcone,Ax,By,Cz,Qectom,Qeckap, &
               -1,-1,&
               LTRAN,LTRAN,LTRAN,LXBCEN,LYBCEN,LZBCEN, &
               IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
               QKEEP,QREEN,QSMTH,QBSPL,QA,QB, &
               RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
               RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
               MAJOR,MINOR,QPROLATE, &
               QINTBP,QZERO,.false.,IPHIB,QPBC,QNPBC, &
               QNONLINEAR,QPARTLINEAR,QFKAP, &
               IPOX,IPOY,IPOZ,MAPT,NATOM)
          IF(QRECTBOX) THEN
             CALL RECT_MAYER(LNCEL,LNCEL,LNCEL,LTRAN,LTRAN,LTRAN,LDCEL, &
                  LXBCEN,LYBCEN,LZBCEN, &
                  MAY,MAYX,MAYY,MAYZ, &
                  RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX)
          ELSEIF(QSPHERE) THEN
             CALL SPHE_MAYER(LNCEL,LNCEL,LNCEL,LTRAN,LTRAN,LTRAN,LDCEL, &
                  LXBCEN,LYBCEN,LZBCEN, &
                  MAY,MAYX,MAYY,MAYZ, &
                  RRXCEN,RRYCEN,RRZCEN,SRDIST)
          ELSEIF(QPROLATE) THEN
             CALL PROL_MAYER(LNCEL,LNCEL,LNCEL,LTRAN,LTRAN,LTRAN,LDCEL, &
                  LXBCEN,LYBCEN,LZBCEN, &
                  MAY,MAYX,MAYY,MAYZ, &
                  RRXCEN,RRYCEN,RRZCEN,MAJOR,MINOR)
          ENDIF
          IXMAX=LNCEL-1
          IYMAX=LNCEL-1
          IZMAX=LNCEL-1
          IXMIN=2
          IYMIN=2
          IZMIN=2
          !     solve PBEQ in the large box
          CALL PBEQ1(MAXITS,DOME,DEPS,EPSW,EPSP,EPSM,KAPPA2, &
               LNCEL,LNCEL,LNCEL,IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
               IPHIX,MAYX,MAYY,MAYZ, &
               ILCHC,MAY,TMEMB, &
               QOSOR,.false.,QPBC,QNPBC,.false.)

          !     construct boundary potentials from the large box
          CALL BOUNDPOT(LNCEL,LNCEL,LNCEL,LDCEL, &
               LTRAN,LTRAN,LTRAN,LXBCEN,LYBCEN,LZBCEN, &
               NCLX,NCLY,NCLZ,DCEL,XBCEN,YBCEN,ZBCEN, &
               IPHIX,IPHIB)

          CALL MAYER(NTPRP,LSTPRP,EPSW,EPSP,KAPPA2,X,Y,Z,CG, &
               PBRAD,NCLX,NCLY,NCLZ,DCEL,WATR,IONR,MAY, &
               MAYX,MAYY,MAYZ,SWIN,ICHC, &
               IPHIX,ZERO,TMEMB,ZMEMB,EPSM,NIMGB, &
               HTMEMB,EPSH,HDENS,POSALP,NEGALP,QCHRG,POFFST,NOFFST, &
               DROPLET,EPSD,XDROPLET,YDROPLET,ZDROPLET,Qdtom,Qdkap, &
               EPSB,LXMAX,LYMAX,LZMAX,LXMIN,LYMIN,LZMIN,Qbtom,Qbkap, &
               EPSC,Xcyln,Ycyln,Zcyln,Rcyln,Hcyln,Qctom,Qckap, &
               EPSEC,Xcone,Ycone,Zcone,Ax,By,Cz,Qectom,Qeckap, &
               -1,-1,&
               TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
               IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
               QKEEP,QREEN,QSMTH,QBSPL,QA,QB, &
               RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
               RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
               MAJOR,MINOR,QPROLATE, &
               .false.,.false.,.true.,IPHIB,QPBC,QNPBC, &
               QNONLINEAR,QPARTLINEAR,QFKAP, &
               IPOX,IPOY,IPOZ,MAPT, NATOM)
       ELSE
          CALL MAYER(NTPRP,LSTPRP,EPSW,EPSP,KAPPA2,X,Y,Z,CG, &
               PBRAD,NCLX,NCLY,NCLZ,DCEL,WATR,IONR,MAY, &
               MAYX,MAYY,MAYZ,SWIN,ICHC, &
               IPHIX,ZERO,TMEMB,ZMEMB,EPSM,NIMGB, &
               HTMEMB,EPSH,HDENS,POSALP,NEGALP,QCHRG,POFFST,NOFFST, &
               DROPLET,EPSD,XDROPLET,YDROPLET,ZDROPLET,Qdtom,Qdkap, &
               EPSB,LXMAX,LYMAX,LZMAX,LXMIN,LYMIN,LZMIN,Qbtom,Qbkap, &
               EPSC,Xcyln,Ycyln,Zcyln,Rcyln,Hcyln,Qctom,Qckap, &
               EPSEC,Xcone,Ycone,Zcone,Ax,By,Cz,Qectom,Qeckap, &
               -1,-1,&
               TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
               IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
               QKEEP,QREEN,QSMTH,QBSPL,QA,QB, &
               RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
               RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
               MAJOR,MINOR,QPROLATE, &
               QINTBP,QZERO,.false.,IPHIB,QPBC,QNPBC, &
               QNONLINEAR,QPARTLINEAR,QFKAP, &
               IPOX,IPOY,IPOZ,MAPT, NATOM)
       ENDIF

       ! vacuum environment in region B.
       IF(QRECTBOX) THEN
          CALL RECT_MAYER(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,DCEL, &
               XBCEN,YBCEN,ZBCEN, &
               MAY,MAYX,MAYY,MAYZ, &
               RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX)
       ELSEIF(QSPHERE) THEN
          CALL SPHE_MAYER(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,DCEL, &
               XBCEN,YBCEN,ZBCEN, &
               MAY,MAYX,MAYY,MAYZ, &
               RRXCEN,RRYCEN,RRZCEN,SRDIST)
       ELSEIF(QPROLATE) THEN
          CALL PROL_MAYER(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,DCEL, &
               XBCEN,YBCEN,ZBCEN, &
               MAY,MAYX,MAYY,MAYZ, &
               RRXCEN,RRYCEN,RRZCEN,MAJOR,MINOR)
       ENDIF

       ! redefine the maximum region outside which EPS is EPSW. (see subroutine MAYER)
       IF(JX2.GT.IXMAX) IXMAX=JX2+1
       IF(JY2.GT.IYMAX) IYMAX=JY2+1
       IF(JZ2.GT.IZMAX) IZMAX=JZ2+1
       IF(JX1.LT.IXMIN) IXMIN=JX1-1
       IF(JY1.LT.IYMIN) IYMIN=JY1-1
       IF(JZ1.LT.IZMIN) IZMIN=JZ1-1
       !
       CALL PBEQ1(MAXITS,DOME,DEPS,EPSW,EPSP,EPSM,KAPPA2, &
            NCLX,NCLY,NCLZ,IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
            IPHIX,MAYX,MAYY,MAYZ, &
            ICHC,MAY,TMEMB, &
            QOSOR,QFOCUS,QPBC,QNPBC,.false.)
       !
       CALL RFORCE(NTRA,LSTRA,X,Y,Z,CG,NCLX,NCLY,NCLZ,DCEL, &
            IPHIX,IPHIP, &
            TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN,HALF, &
            RXNAFX,RXNAFY,RXNAFZ,EGSBPA, &
            .false.,QBSPL)

       WRITE(OUTU,'(3X,A,F13.5,A)') &
            'Reaction field free energy in outer region [G_OO]=', &
            EGSBPA,' [KCAL/MOL]'

    ENDIF

    !
    !     DELTA G_BB
    !     NOTE: Vmemb = 0.0 for complex solution environment
    !
    IF(QGBB) THEN
       QA=.true.
       QB=.false.

       ! In vacuum
       IF(QLBOX.AND.QFOCUS) THEN
          !     get the grid parameters in the lagre box
          CALL MAYER(NTPRP,LSTPRP,ONE,ONE,ZERO,X,Y,Z,CG, &
               PBRAD,LNCEL,LNCEL,LNCEL,LDCEL,WATR,IONR,MAY, &
               MAYX,MAYY,MAYZ,SWIN,ILCHC, &
               IPHIP,ZERO,TMEMB,ZMEMB,ONE,NIMGB, &
               HTMEMB,ONE,HDENS,POSALP,NEGALP,QCHRG,POFFST,NOFFST, &
               DROPLET,ONE,XDROPLET,YDROPLET,ZDROPLET,Qdtom,Qdkap, &
               ONE,LXMAX,LYMAX,LZMAX,LXMIN,LYMIN,LZMIN,Qbtom,Qbkap, &
               ONE,Xcyln,Ycyln,Zcyln,Rcyln,Hcyln,Qctom,Qckap, &
               ONE,Xcone,Ycone,Zcone,Ax,By,Cz,Qectom,Qeckap, &
               -1,-1,&
               LTRAN,LTRAN,LTRAN,LXBCEN,LYBCEN,LZBCEN, &
               IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
               QKEEP,QREEN,QSMTH,QBSPL,QA,QB, &
               RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
               RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
               MAJOR,MINOR,QPROLATE, &
               QINTBP,QZERO,.false.,IPHIB,QPBC,QNPBC, &
               QNONLINEAR,QPARTLINEAR,QFKAP, &
               IPOX,IPOY,IPOZ,MAPT,NATOM)
          IF(EPSP.ne.ONE) THEN
             IF(QRECTBOX) THEN
                CALL RECT_MAYER(LNCEL,LNCEL,LNCEL,LTRAN,LTRAN,LTRAN,LDCEL, &
                     LXBCEN,LYBCEN,LZBCEN, &
                     MAY,MAYX,MAYY,MAYZ, &
                     RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX)
             ELSEIF(QSPHERE) THEN
                CALL SPHE_MAYER(LNCEL,LNCEL,LNCEL,LTRAN,LTRAN,LTRAN,LDCEL, &
                     LXBCEN,LYBCEN,LZBCEN, &
                     MAY,MAYX,MAYY,MAYZ, &
                     RRXCEN,RRYCEN,RRZCEN,SRDIST)
             ELSEIF(QPROLATE) THEN
                CALL PROL_MAYER(LNCEL,LNCEL,LNCEL,LTRAN,LTRAN,LTRAN,LDCEL, &
                     LXBCEN,LYBCEN,LZBCEN, &
                     MAY,MAYX,MAYY,MAYZ, &
                     RRXCEN,RRYCEN,RRZCEN,MAJOR,MINOR)
             ENDIF
             IXMAX=LNCEL-1
             IYMAX=LNCEL-1
             IZMAX=LNCEL-1
             IXMIN=2
             IYMIN=2
             IZMIN=2
          ENDIF
          !     solve PBEQ in the large box
          CALL PBEQ1(MAXITS,DOME,DEPS,ONE,ONE,ONE,ZERO, &
               LNCEL,LNCEL,LNCEL,IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
               IPHIP,MAYX,MAYY,MAYZ, &
               ILCHC,MAY,TMEMB, &
               .false.,.false.,QPBC,QNPBC,.false.)

          !     construct boundary potentials from the large box
          CALL BOUNDPOT(LNCEL,LNCEL,LNCEL,LDCEL, &
               LTRAN,LTRAN,LTRAN,LXBCEN,LYBCEN,LZBCEN, &
               NCLX,NCLY,NCLZ,DCEL,XBCEN,YBCEN,ZBCEN, &
               IPHIP,IPHIB)
          CALL MAYER(NTPRP,LSTPRP,ONE,ONE,ZERO,X,Y,Z,CG, &
               PBRAD,NCLX,NCLY,NCLZ,DCEL,WATR,IONR,MAY, &
               MAYX,MAYY,MAYZ,SWIN,ICHC, &
               IPHIP,ZERO,TMEMB,ZMEMB,ONE,NIMGB, &
               HTMEMB,ONE,HDENS,POSALP,NEGALP,QCHRG,POFFST,NOFFST, &
               DROPLET,ONE,XDROPLET,YDROPLET,ZDROPLET,Qdtom,Qdkap, &
               ONE,LXMAX,LYMAX,LZMAX,LXMIN,LYMIN,LZMIN,Qbtom,Qbkap, &
               ONE,Xcyln,Ycyln,Zcyln,Rcyln,Hcyln,Qctom,Qckap, &
               ONE,Xcone,Ycone,Zcone,Ax,By,Cz,Qectom,Qeckap, &
               -1,-1,&
               TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
               IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
               QKEEP,QREEN,QSMTH,QBSPL,QA,QB, &
               RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
               RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
               MAJOR,MINOR,QPROLATE, &
               .false.,.false.,.true.,IPHIB,QPBC,QNPBC, &
               QNONLINEAR,QPARTLINEAR,QFKAP, &
               IPOX,IPOY,IPOZ,MAPT, NATOM)
       ELSE
          CALL MAYER(NTPRP,LSTPRP,ONE,ONE,ZERO,X,Y,Z,CG, &
               PBRAD,NCLX,NCLY,NCLZ,DCEL,WATR,IONR,MAY, &
               MAYX,MAYY,MAYZ,SWIN,ICHC, &
               IPHIP,ZERO,TMEMB,ZMEMB,ONE,NIMGB, &
               HTMEMB,ONE,HDENS,POSALP,NEGALP,QCHRG,POFFST,NOFFST, &
               DROPLET,ONE,XDROPLET,YDROPLET,ZDROPLET,Qdtom,Qdkap, &
               ONE,LXMAX,LYMAX,LZMAX,LXMIN,LYMIN,LZMIN,Qbtom,Qbkap, &
               ONE,Xcyln,Ycyln,Zcyln,Rcyln,Hcyln,Qctom,Qckap, &
               ONE,Xcone,Ycone,Zcone,Ax,By,Cz,Qectom,Qeckap, &
               -1,-1,&
               TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
               IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
               QKEEP,QREEN,QSMTH,QBSPL,QA,QB, &
               RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
               RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
               MAJOR,MINOR,QPROLATE, &
               QINTBP,QZERO,.false.,IPHIB,QPBC,QNPBC, &
               QNONLINEAR,QPARTLINEAR,QFKAP, &
               IPOX,IPOY,IPOZ,MAPT,NATOM)
       ENDIF

       IF(EPSP.ne.ONE) THEN
          ! vacuum environment in region B.
          IF(QRECTBOX) THEN
             CALL RECT_MAYER(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,DCEL, &
                  XBCEN,YBCEN,ZBCEN, &
                  MAY,MAYX,MAYY,MAYZ, &
                  RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX)
          ELSEIF(QSPHERE) THEN
             CALL SPHE_MAYER(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,DCEL, &
                  XBCEN,YBCEN,ZBCEN, &
                  MAY,MAYX,MAYY,MAYZ, &
                  RRXCEN,RRYCEN,RRZCEN,SRDIST)
          ELSEIF(QPROLATE) THEN
             CALL PROL_MAYER(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,DCEL, &
                  XBCEN,YBCEN,ZBCEN, &
                  MAY,MAYX,MAYY,MAYZ, &
                  RRXCEN,RRYCEN,RRZCEN,MAJOR,MINOR)
          ENDIF
       ENDIF
       !
       CALL PBEQ1(MAXITS,DOME,DEPS,ONE,ONE,ONE,ZERO, &
            NCLX,NCLY,NCLZ,IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
            IPHIP,MAYX,MAYY,MAYZ, &
            ICHC,MAY,TMEMB, &
            .false.,QFOCUS,QPBC,QNPBC,.false.)

       ! In solution
       IF(QLBOX.AND.QFOCUS) THEN
          !     get the grid parameters in the lagre box
          CALL MAYER(NTPRP,LSTPRP,EPSW,EPSP,KAPPA2,X,Y,Z,CG, &
               PBRAD,LNCEL,LNCEL,LNCEL,LDCEL,WATR,IONR,MAY, &
               MAYX,MAYY,MAYZ,SWIN,ILCHC, &
               IPHIW,ZERO,TMEMB,ZMEMB,EPSM,NIMGB, &
               HTMEMB,EPSH,HDENS,POSALP,NEGALP,QCHRG,POFFST,NOFFST, &
               DROPLET,EPSD,XDROPLET,YDROPLET,ZDROPLET,Qdtom,Qdkap, &
               EPSB,LXMAX,LYMAX,LZMAX,LXMIN,LYMIN,LZMIN,Qbtom,Qbkap, &
               EPSC,Xcyln,Ycyln,Zcyln,Rcyln,Hcyln,Qctom,Qckap, &
               EPSEC,Xcone,Ycone,Zcone,Ax,By,Cz,Qectom,Qeckap, &
               -1,-1,&
               LTRAN,LTRAN,LTRAN,LXBCEN,LYBCEN,LZBCEN, &
               IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
               QKEEP,QREEN,QSMTH,QBSPL,QA,QB, &
               RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
               RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
               MAJOR,MINOR,QPROLATE, &
               QINTBP,QZERO,.false.,IPHIB,QPBC,QNPBC, &
                                !              QINTBP QZERO
               QNONLINEAR,QPARTLINEAR,QFKAP, &
               IPOX,IPOY,IPOZ,MAPT,NATOM)
          IF(QRECTBOX) THEN
             CALL RECT_MAYER(LNCEL,LNCEL,LNCEL,LTRAN,LTRAN,LTRAN,LDCEL, &
                  LXBCEN,LYBCEN,LZBCEN, &
                  MAY,MAYX,MAYY,MAYZ, &
                  RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX)
          ELSEIF(QSPHERE) THEN
             CALL SPHE_MAYER(LNCEL,LNCEL,LNCEL,LTRAN,LTRAN,LTRAN,LDCEL, &
                  LXBCEN,LYBCEN,LZBCEN, &
                  MAY,MAYX,MAYY,MAYZ, &
                  RRXCEN,RRYCEN,RRZCEN,SRDIST)
          ELSEIF(QPROLATE) THEN
             CALL PROL_MAYER(LNCEL,LNCEL,LNCEL,LTRAN,LTRAN,LTRAN,LDCEL, &
                  LXBCEN,LYBCEN,LZBCEN, &
                  MAY,MAYX,MAYY,MAYZ, &
                  RRXCEN,RRYCEN,RRZCEN,MAJOR,MINOR)
          ENDIF
          IXMAX=LNCEL-1
          IYMAX=LNCEL-1
          IZMAX=LNCEL-1
          IXMIN=2
          IYMIN=2
          IZMIN=2
          !     solve PBEQ in the large box
          CALL PBEQ1(MAXITS,DOME,DEPS,EPSW,EPSP,EPSM,KAPPA2, &
               LNCEL,LNCEL,LNCEL,IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
               IPHIW,MAYX,MAYY,MAYZ, &
               ILCHC,MAY,TMEMB, &
               QOSOR,.false.,QPBC,QNPBC,.false.)

          !     construct boundary potentials from the large box
          CALL BOUNDPOT(LNCEL,LNCEL,LNCEL,LDCEL, &
               LTRAN,LTRAN,LTRAN,LXBCEN,LYBCEN,LZBCEN, &
               NCLX,NCLY,NCLZ,DCEL,XBCEN,YBCEN,ZBCEN, &
               IPHIW,IPHIB)
          CALL MAYER(NTPRP,LSTPRP,EPSW,EPSP,KAPPA2,X,Y,Z,CG, &
               PBRAD,NCLX,NCLY,NCLZ,DCEL,WATR,IONR,MAY, &
               MAYX,MAYY,MAYZ,SWIN,ICHC, &
               IPHIW,ZERO,TMEMB,ZMEMB,EPSM,NIMGB, &
               HTMEMB,EPSH,HDENS,POSALP,NEGALP,QCHRG,POFFST,NOFFST, &
               DROPLET,EPSD,XDROPLET,YDROPLET,ZDROPLET,Qdtom,Qdkap, &
               EPSB,LXMAX,LYMAX,LZMAX,LXMIN,LYMIN,LZMIN,Qbtom,Qbkap, &
               EPSC,Xcyln,Ycyln,Zcyln,Rcyln,Hcyln,Qctom,Qckap, &
               EPSEC,Xcone,Ycone,Zcone,Ax,By,Cz,Qectom,Qeckap, &
               -1,-1,&
               TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
               IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
               QKEEP,QREEN,QSMTH,QBSPL,QA,QB, &
               RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
               RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
               MAJOR,MINOR,QPROLATE, &
               .false.,.false.,.true.,IPHIB,QPBC,QNPBC, &
               QNONLINEAR,QPARTLINEAR,QFKAP, &
               IPOX,IPOY,IPOZ,MAPT,NATOM)
       ELSE
          CALL MAYER(NTPRP,LSTPRP,EPSW,EPSP,KAPPA2,X,Y,Z,CG, &
               PBRAD,NCLX,NCLY,NCLZ,DCEL,WATR,IONR,MAY, &
               MAYX,MAYY,MAYZ,SWIN,ICHC, &
               IPHIW,ZERO,TMEMB,ZMEMB,EPSM,NIMGB, &
               HTMEMB,EPSH,HDENS,POSALP,NEGALP,QCHRG,POFFST,NOFFST, &
               DROPLET,EPSD,XDROPLET,YDROPLET,ZDROPLET,Qdtom,Qdkap, &
               EPSB,LXMAX,LYMAX,LZMAX,LXMIN,LYMIN,LZMIN,Qbtom,Qbkap, &
               EPSC,Xcyln,Ycyln,Zcyln,Rcyln,Hcyln,Qctom,Qckap, &
               EPSEC,Xcone,Ycone,Zcone,Ax,By,Cz,Qectom,Qeckap, &
               -1,-1,&
               TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
               IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
               QKEEP,QREEN,QSMTH,QBSPL,QA,QB, &
               RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
               RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
               MAJOR,MINOR,QPROLATE, &
               QINTBP,QZERO,.false.,IPHIB,QPBC,QNPBC, &
               QNONLINEAR,QPARTLINEAR,QFKAP, &
               IPOX,IPOY,IPOZ,MAPT,NATOM)
       ENDIF

       ! vacuum environment in region B.
       IF(QRECTBOX) THEN
          CALL RECT_MAYER(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,DCEL, &
               XBCEN,YBCEN,ZBCEN, &
               MAY,MAYX,MAYY,MAYZ, &
               RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX)
       ELSEIF(QSPHERE) THEN
          CALL SPHE_MAYER(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,DCEL, &
               XBCEN,YBCEN,ZBCEN, &
               MAY,MAYX,MAYY,MAYZ, &
               RRXCEN,RRYCEN,RRZCEN,SRDIST)
       ELSEIF(QPROLATE) THEN
          CALL PROL_MAYER(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,DCEL, &
               XBCEN,YBCEN,ZBCEN, &
               MAY,MAYX,MAYY,MAYZ, &
               RRXCEN,RRYCEN,RRZCEN,MAJOR,MINOR)
       ENDIF

       ! redefine the maximum region outside which EPS is EPSW. (see subroutine MAYER)
       IF(JX2.GT.IXMAX) IXMAX=JX2+1
       IF(JY2.GT.IYMAX) IYMAX=JY2+1
       IF(JZ2.GT.IZMAX) IZMAX=JZ2+1
       IF(JX1.LT.IXMIN) IXMIN=JX1-1
       IF(JY1.LT.IYMIN) IYMIN=JY1-1
       IF(JZ1.LT.IZMIN) IZMIN=JZ1-1
       !
       CALL PBEQ1(MAXITS,DOME,DEPS,EPSW,EPSP,EPSM,KAPPA2, &
            NCLX,NCLY,NCLZ,IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
            IPHIW,MAYX,MAYY,MAYZ, &
            ICHC,MAY,TMEMB, &
            QOSOR,QFOCUS,QPBC,QNPBC,.false.)
       !
       CALL RFORCE(NTRB,LSTRB,X,Y,Z,CG,NCLX,NCLY,NCLZ,DCEL, &
            IPHIW,IPHIP, &
            TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN,HALF, &
            RXNAFX,RXNAFY,RXNAFZ,EGSBPA, &
            .false.,QBSPL)

       WRITE(OUTU,'(3X,A,F13.5,A)') &
            'Reaction field free energy in inner region [G_II]=', &
            EGSBPA,' [KCAL/MOL]'

    ENDIF

    !
    !     DELTA G_AB
    !     NOTE: Vmemb is considered here.
    !
    IF(QGAB) THEN
       QA=.false.
       QB=.true.

       ! In vacuum
       IF(QLBOX.AND.QFOCUS) THEN
          !     get the grid parameters in the lagre box
          CALL MAYER(NTPRP,LSTPRP,ONE,ONE,ZERO,X,Y,Z,CG, &
               PBRAD,LNCEL,LNCEL,LNCEL,LDCEL,WATR,IONR,MAY, &
               MAYX,MAYY,MAYZ,SWIN,ILCHC, &
               IPHIP,ZERO,TMEMB,ZMEMB,ONE,NIMGB, &
               HTMEMB,ONE,HDENS,POSALP,NEGALP,QCHRG,POFFST,NOFFST, &
               DROPLET,ONE,XDROPLET,YDROPLET,ZDROPLET,Qdtom,Qdkap, &
               ONE,LXMAX,LYMAX,LZMAX,LXMIN,LYMIN,LZMIN,Qbtom,Qbkap, &
               ONE,Xcyln,Ycyln,Zcyln,Rcyln,Hcyln,Qctom,Qckap, &
               ONE,Xcone,Ycone,Zcone,Ax,By,Cz,Qectom,Qeckap, &
               -1,-1,&
               LTRAN,LTRAN,LTRAN,LXBCEN,LYBCEN,LZBCEN, &
               IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
               QKEEP,QREEN,QSMTH,QBSPL,QA,QB, &
               RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
               RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
               MAJOR,MINOR,QPROLATE, &
               QINTBP,QZERO,.false.,IPHIB,QPBC,QNPBC, &
               QNONLINEAR,QPARTLINEAR,QFKAP, &
               IPOX,IPOY,IPOZ,MAPT,NATOM)
          IF(EPSP.ne.ONE) THEN
             IF(QRECTBOX) THEN
                CALL RECT_MAYER(LNCEL,LNCEL,LNCEL,LTRAN,LTRAN,LTRAN,LDCEL, &
                     LXBCEN,LYBCEN,LZBCEN, &
                     MAY,MAYX,MAYY,MAYZ, &
                     RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX)
             ELSEIF(QSPHERE) THEN
                CALL SPHE_MAYER(LNCEL,LNCEL,LNCEL,LTRAN,LTRAN,LTRAN,LDCEL, &
                     LXBCEN,LYBCEN,LZBCEN, &
                     MAY,MAYX,MAYY,MAYZ, &
                     RRXCEN,RRYCEN,RRZCEN,SRDIST)
             ELSEIF(QPROLATE) THEN
                CALL PROL_MAYER(LNCEL,LNCEL,LNCEL,LTRAN,LTRAN,LTRAN,LDCEL, &
                     LXBCEN,LYBCEN,LZBCEN, &
                     MAY,MAYX,MAYY,MAYZ, &
                     RRXCEN,RRYCEN,RRZCEN,MAJOR,MINOR)
             ENDIF
             IXMAX=LNCEL-1
             IYMAX=LNCEL-1
             IZMAX=LNCEL-1
             IXMIN=2
             IYMIN=2
             IZMIN=2
          ENDIF
          !     solve PBEQ in the large box
          CALL PBEQ1(MAXITS,DOME,DEPS,ONE,ONE,ONE,ZERO, &
               LNCEL,LNCEL,LNCEL,IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
               IPHIP,MAYX,MAYY,MAYZ, &
               ILCHC,MAY,TMEMB, &
               .false.,.false.,QPBC,QNPBC,.false.)

          !     construct boundary potentials from the large box
          CALL BOUNDPOT(LNCEL,LNCEL,LNCEL,LDCEL, &
               LTRAN,LTRAN,LTRAN,LXBCEN,LYBCEN,LZBCEN, &
               NCLX,NCLY,NCLZ,DCEL,XBCEN,YBCEN,ZBCEN, &
               IPHIP,IPHIB)
          CALL MAYER(NTPRP,LSTPRP,ONE,ONE,ZERO,X,Y,Z,CG, &
               PBRAD,NCLX,NCLY,NCLZ,DCEL,WATR,IONR,MAY, &
               MAYX,MAYY,MAYZ,SWIN,ICHC, &
               IPHIP,ZERO,TMEMB,ZMEMB,ONE,NIMGB, &
               HTMEMB,ONE,HDENS,POSALP,NEGALP,QCHRG,POFFST,NOFFST, &
               DROPLET,ONE,XDROPLET,YDROPLET,ZDROPLET,Qdtom,Qdkap, &
               ONE,LXMAX,LYMAX,LZMAX,LXMIN,LYMIN,LZMIN,Qbtom,Qbkap, &
               ONE,Xcyln,Ycyln,Zcyln,Rcyln,Hcyln,Qctom,Qckap, &
               ONE,Xcone,Ycone,Zcone,Ax,By,Cz,Qectom,Qeckap, &
               -1,-1,&
               TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
               IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
               QKEEP,QREEN,QSMTH,QBSPL,QA,QB, &
               RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
               RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
               MAJOR,MINOR,QPROLATE, &
               .false.,.false.,.true.,IPHIB,QPBC,QNPBC, &
               QNONLINEAR,QPARTLINEAR,QFKAP, &
               IPOX,IPOY,IPOZ,MAPT,NATOM)
       ELSE
          CALL MAYER(NTPRP,LSTPRP,ONE,ONE,ZERO,X,Y,Z,CG, &
               PBRAD,NCLX,NCLY,NCLZ,DCEL,WATR,IONR,MAY, &
               MAYX,MAYY,MAYZ,SWIN,ICHC, &
               IPHIP,ZERO,TMEMB,ZMEMB,ONE,NIMGB, &
               HTMEMB,ONE,HDENS,POSALP,NEGALP,QCHRG,POFFST,NOFFST, &
               DROPLET,ONE,XDROPLET,YDROPLET,ZDROPLET,Qdtom,Qdkap, &
               ONE,LXMAX,LYMAX,LZMAX,LXMIN,LYMIN,LZMIN,Qbtom,Qbkap, &
               ONE,Xcyln,Ycyln,Zcyln,Rcyln,Hcyln,Qctom,Qckap, &
               ONE,Xcone,Ycone,Zcone,Ax,By,Cz,Qectom,Qeckap, &
               -1,-1,&
               TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
               IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
               QKEEP,QREEN,QSMTH,QBSPL,QA,QB, &
               RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
               RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
               MAJOR,MINOR,QPROLATE, &
               QINTBP,QZERO,.false.,IPHIB,QPBC,QNPBC, &
               QNONLINEAR,QPARTLINEAR,QFKAP, &
               IPOX,IPOY,IPOZ,MAPT,NATOM)
       ENDIF

       IF(EPSP.ne.ONE) THEN
          ! vacuum environment in region B.
          IF(QRECTBOX) THEN
             CALL RECT_MAYER(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,DCEL, &
                  XBCEN,YBCEN,ZBCEN, &
                  MAY,MAYX,MAYY,MAYZ, &
                  RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX)
          ELSEIF(QSPHERE) THEN
             CALL SPHE_MAYER(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,DCEL, &
                  XBCEN,YBCEN,ZBCEN, &
                  MAY,MAYX,MAYY,MAYZ, &
                  RRXCEN,RRYCEN,RRZCEN,SRDIST)
          ELSEIF(QPROLATE) THEN
             CALL PROL_MAYER(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,DCEL, &
                  XBCEN,YBCEN,ZBCEN, &
                  MAY,MAYX,MAYY,MAYZ, &
                  RRXCEN,RRYCEN,RRZCEN,MAJOR,MINOR)
          ENDIF
       ENDIF
       !
       CALL PBEQ1(MAXITS,DOME,DEPS,ONE,ONE,ONE,ZERO, &
            NCLX,NCLY,NCLZ,IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
            IPHIP,MAYX,MAYY,MAYZ, &
            ICHC,MAY,TMEMB, &
            .false.,QFOCUS,QPBC,QNPBC,.false.)

       ! In solution
       IF(QLBOX.AND.QFOCUS) THEN
          !     get the grid parameters in the lagre box
          CALL MAYER(NTPRP,LSTPRP,EPSW,EPSP,KAPPA2,X,Y,Z,CG, &
               PBRAD,LNCEL,LNCEL,LNCEL,LDCEL,WATR,IONR,MAY, &
               MAYX,MAYY,MAYZ,SWIN,ILCHC, &
               IPHIX,VMEMB,TMEMB,ZMEMB,EPSM,NIMGB, &
               HTMEMB,EPSH,HDENS,POSALP,NEGALP,QCHRG,POFFST,NOFFST, &
               DROPLET,EPSD,XDROPLET,YDROPLET,ZDROPLET,Qdtom,Qdkap, &
               EPSB,LXMAX,LYMAX,LZMAX,LXMIN,LYMIN,LZMIN,Qbtom,Qbkap, &
               EPSC,Xcyln,Ycyln,Zcyln,Rcyln,Hcyln,Qctom,Qckap, &
               EPSEC,Xcone,Ycone,Zcone,Ax,By,Cz,Qectom,Qeckap, &
               -1,-1,&
               LTRAN,LTRAN,LTRAN,LXBCEN,LYBCEN,LZBCEN, &
               IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
               QKEEP,QREEN,QSMTH,QBSPL,QA,QB, &
               RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
               RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
               MAJOR,MINOR,QPROLATE, &
               QINTBP,QZERO,.false.,IPHIB,QPBC,QNPBC, &
                                !              QINTBP QZERO
               QNONLINEAR,QPARTLINEAR,QFKAP, &
               IPOX,IPOY,IPOZ,MAPT,NATOM)
          IF(QRECTBOX) THEN
             CALL RECT_MAYER(LNCEL,LNCEL,LNCEL,LTRAN,LTRAN,LTRAN,LDCEL, &
                  LXBCEN,LYBCEN,LZBCEN, &
                  MAY,MAYX,MAYY,MAYZ, &
                  RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX)
          ELSEIF(QSPHERE) THEN
             CALL SPHE_MAYER(LNCEL,LNCEL,LNCEL,LTRAN,LTRAN,LTRAN,LDCEL, &
                  LXBCEN,LYBCEN,LZBCEN, &
                  MAY,MAYX,MAYY,MAYZ, &
                  RRXCEN,RRYCEN,RRZCEN,SRDIST)
          ELSEIF(QPROLATE) THEN
             CALL PROL_MAYER(LNCEL,LNCEL,LNCEL,LTRAN,LTRAN,LTRAN,LDCEL, &
                  LXBCEN,LYBCEN,LZBCEN, &
                  MAY,MAYX,MAYY,MAYZ, &
                  RRXCEN,RRYCEN,RRZCEN,MAJOR,MINOR)
          ENDIF
          IXMAX=LNCEL-1
          IYMAX=LNCEL-1
          IZMAX=LNCEL-1
          IXMIN=2
          IYMIN=2
          IZMIN=2
          !     solve PBEQ in the large box
          CALL PBEQ1(MAXITS,DOME,DEPS,EPSW,EPSP,EPSM,KAPPA2, &
               LNCEL,LNCEL,LNCEL,IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
               IPHIX,MAYX,MAYY,MAYZ, &
               ILCHC,MAY,TMEMB, &
               .false.,.false.,QPBC,QNPBC,.false.)

          !     construct boundary potentials from the large box
          CALL BOUNDPOT(LNCEL,LNCEL,LNCEL,LDCEL, &
               LTRAN,LTRAN,LTRAN,LXBCEN,LYBCEN,LZBCEN, &
               NCLX,NCLY,NCLZ,DCEL,XBCEN,YBCEN,ZBCEN, &
               IPHIX,IPHIB)
          CALL MAYER(NTPRP,LSTPRP,EPSW,EPSP,KAPPA2,X,Y,Z,CG, &
               PBRAD,NCLX,NCLY,NCLZ,DCEL,WATR,IONR,MAY, &
               MAYX,MAYY,MAYZ,SWIN,ICHC, &
               IPHIX,VMEMB,TMEMB,ZMEMB,EPSM,NIMGB, &
               HTMEMB,EPSH,HDENS,POSALP,NEGALP,QCHRG,POFFST,NOFFST, &
               DROPLET,EPSD,XDROPLET,YDROPLET,ZDROPLET,Qdtom,Qdkap, &
               EPSB,LXMAX,LYMAX,LZMAX,LXMIN,LYMIN,LZMIN,Qbtom,Qbkap, &
               EPSC,Xcyln,Ycyln,Zcyln,Rcyln,Hcyln,Qctom,Qckap, &
               EPSEC,Xcone,Ycone,Zcone,Ax,By,Cz,Qectom,Qeckap, &
               -1,-1,&
               TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
               IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
               QKEEP,QREEN,QSMTH,QBSPL,QA,QB, &
               RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
               RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
               MAJOR,MINOR,QPROLATE, &
               .false.,.false.,.true.,IPHIB,QPBC,QNPBC, &
               QNONLINEAR,QPARTLINEAR,QFKAP, &
               IPOX,IPOY,IPOZ,MAPT, NATOM)
       ELSE
          CALL MAYER(NTPRP,LSTPRP,EPSW,EPSP,KAPPA2,X,Y,Z,CG, &
               PBRAD,NCLX,NCLY,NCLZ,DCEL,WATR,IONR,MAY, &
               MAYX,MAYY,MAYZ,SWIN,ICHC, &
               IPHIX,VMEMB,TMEMB,ZMEMB,EPSM,NIMGB, &
               HTMEMB,EPSH,HDENS,POSALP,NEGALP,QCHRG,POFFST,NOFFST, &
               DROPLET,EPSD,XDROPLET,YDROPLET,ZDROPLET,Qdtom,Qdkap, &
               EPSB,LXMAX,LYMAX,LZMAX,LXMIN,LYMIN,LZMIN,Qbtom,Qbkap, &
               EPSC,Xcyln,Ycyln,Zcyln,Rcyln,Hcyln,Qctom,Qckap, &
               EPSEC,Xcone,Ycone,Zcone,Ax,By,Cz,Qectom,Qeckap, &
               -1,-1,&
               TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
               IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
               QKEEP,QREEN,QSMTH,QBSPL,QA,QB, &
               RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
               RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
               MAJOR,MINOR,QPROLATE, &
               QINTBP,QZERO,.false.,IPHIB,QPBC,QNPBC, &
               QNONLINEAR,QPARTLINEAR,QFKAP, &
               IPOX,IPOY,IPOZ,MAPT,NATOM)
       ENDIF

       ! vacuum environment in region B.
       IF(QRECTBOX) THEN
          CALL RECT_MAYER(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,DCEL, &
               XBCEN,YBCEN,ZBCEN, &
               MAY,MAYX,MAYY,MAYZ, &
               RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX)
       ELSEIF(QSPHERE) THEN
          CALL SPHE_MAYER(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,DCEL, &
               XBCEN,YBCEN,ZBCEN, &
               MAY,MAYX,MAYY,MAYZ, &
               RRXCEN,RRYCEN,RRZCEN,SRDIST)
       ELSEIF(QPROLATE) THEN
          CALL PROL_MAYER(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,DCEL, &
               XBCEN,YBCEN,ZBCEN, &
               MAY,MAYX,MAYY,MAYZ, &
               RRXCEN,RRYCEN,RRZCEN,MAJOR,MINOR)
       ENDIF

       ! redefine the maximum region outside which EPS is EPSW. (see subroutine MAYER)
       IF(JX2.GT.IXMAX) IXMAX=JX2+1
       IF(JY2.GT.IYMAX) IYMAX=JY2+1
       IF(JZ2.GT.IZMAX) IZMAX=JZ2+1
       IF(JX1.LT.IXMIN) IXMIN=JX1-1
       IF(JY1.LT.IYMIN) IYMIN=JY1-1
       IF(JZ1.LT.IZMIN) IZMIN=JZ1-1
       !
       CALL PBEQ1(MAXITS,DOME,DEPS,EPSW,EPSP,EPSM,KAPPA2, &
            NCLX,NCLY,NCLZ,IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
            IPHIX,MAYX,MAYY,MAYZ, &
            ICHC,MAY,TMEMB, &
            QOSOR,QFOCUS,QPBC,QNPBC,.false.)

       !     get the electrostatic solvation free energy and reaction field forces
       !     NOTE: HALF -> ONE
       CALL RFORCE(NTRB,LSTRB,X,Y,Z,CG,NCLX,NCLY,NCLZ,DCEL, &
                                !                 ----      -----
            IPHIX,IPHIP, &
            TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN,ONE, &
                                !                                              ---
            RXNAFX,RXNAFY,RXNAFZ,EGSBPA, &
            .false.,QBSPL)

       WRITE(OUTU,'(3X,A,F13.5,A)') &
            'Electrostatic interaction free energy      [G_IO]=', &
            EGSBPA,' [KCAL/MOL]'

    ENDIF
    IF(QLBOX.AND.QFOCUS) THEN
       LNCEL3=LNCEL*LNCEL*LNCEL
       BNCEL=2*(NCLX*NCLX+NCLY*NCLY+NCLZ*NCLZ)
       call chmdealloc('pbeq.src','GDECOMP','ILCHC',LNCEL3,cr4=ILCHC)
       call chmdealloc('pbeq.src','GDECOMP','IPHIB',BNCEL,cr4=IPHIB)
    ELSE
       call chmdealloc('pbeq.src','GDECOMP','IPHIB',1,cr4=IPHIB)
    ENDIF
    QA=.false.
    QB=.false.
    !
    RETURN
  END SUBROUTINE GDECOMP
  !
  SUBROUTINE RECT_GSBP1(NATOM,X,Y,Z,CG, &
       LNCEL,LTRAN,LDCEL,LXBCEN,LYBCEN,LZBCEN,QPRIN)
    !-----------------------------------------------------------------------
    !     The reaction field energy and forces of the region of interest
    !     are calculated using the Legendre Polynomials
    !
    use chm_kinds
    use memory
    use number
    use stream
    use dimens_fcm
    use consta
    use timerm
    use machutil,only:wrttim
    implicit none
    !
    INTEGER NATOM
    real(chm_real)  X(*),Y(*),Z(*),CG(*)
    LOGICAL QPRIN
    INTEGER LNCEL
    real(chm_real)  LTRAN,LDCEL,LXBCEN,LYBCEN,LZBCEN
    ! local
    integer, allocatable, dimension(:) :: IIP0
    real(chm_real4),allocatable,dimension(:) :: ILCHC,IPHIB
    INTEGER NORDER,OLDNTP
    INTEGER JX1,JX2,JY1,JY2,JZ1,JZ2,NPROI,NPGD
    INTEGER LNCEL3,BNCEL

    ! save basis functions
    OLDNTP=OLDXNP*OLDYNP*OLDZNP
    CALL RECTPOL(XNPOL,YNPOL,ZNPOL,OLDNTP,QMIJ, &
         LSTPX,LSTPY,LSTPZ,LSTPOL)

    ! calculate the normalization constants of the basis functions
    CALL RECT_NORM(NTPOL,XSCALE,YSCALE,ZSCALE, &
         LSTPX,LSTPY,LSTPZ,BNORM)

    IF(QMIJ.AND.OLDNTP.EQ.NTPOL) RETURN

    ! find out the grid points inside region of interest
    JX1=INT((TRANX+RBXMIN-XBCEN)/DCEL)-1
    JX2=INT((TRANX+RBXMAX-XBCEN)/DCEL)+3
    JY1=INT((TRANY+RBYMIN-YBCEN)/DCEL)-1
    JY2=INT((TRANY+RBYMAX-YBCEN)/DCEL)+3
    JZ1=INT((TRANZ+RBZMIN-ZBCEN)/DCEL)-1
    JZ2=INT((TRANZ+RBZMAX-ZBCEN)/DCEL)+3
    NPROI=(JX2-JX1)*(JY2-JY1)*(JZ2-JZ1)
    call chmalloc('pbeq.src','RECT_GSBP1','IIP0',NPROI,intg=IIP0)
    CALL RECT_GPINRR(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,DCEL, &
         XBCEN,YBCEN,ZBCEN, &
         RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX, &
         JX1,JX2,JY1,JY2,JZ1,JZ2,IIP0,NPGD)

    ! skip the initialization of CHC array and
    !      the all charge distribution calculations in MAYER
    QA=.true.
    QB=.true.

    ! allocate charge densities in the large box and
    !          boundary potentials in the original box
    LNCEL3=LNCEL*LNCEL*LNCEL
    call chmalloc('pbeq.src','RECT_GSBP1','ILCHC',LNCEL3,cr4=ILCHC)
    BNCEL=1
    IF(QFOCUS)  BNCEL=2*(NCLX*NCLX+NCLY*NCLY+NCLZ*NCLZ)
    call chmalloc('pbeq.src','RECT_GSBP1','IPHIB',BNCEL,cr4=IPHIB)
    DO NORDER=OLDNTP+1,NTPOL
       IF(PRNLEV.GT.7.AND.NORDER.EQ.1) WRITE(OUTU,'(A)')
       IF(PRNLEV.GT.7) WRITE(OUTU,'(3x,a,i5,a)') &
            'Reaction field calculation for',NORDER,' basis function'

       ! replace ICHC (CDEN) array with the basis functions (Legendre Polynomials)
       ! (in original box)
       CALL RECT_CHC(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,DCEL, &
            XBCEN,YBCEN,ZBCEN, &
            NORDER,LSTPX,LSTPY,LSTPZ,BNORM, &
            ICHC,CGSCAL,IIP0,NPGD, &
            RRXCEN,RRYCEN,RRZCEN,XSCALE,YSCALE,ZSCALE)

       ! get charge distribution ILCHC in the large box from ICHC
       CALL INITGRID(LNCEL,LNCEL,LNCEL,ILCHC,ZERO)
       CALL MAYER_CD_TRIL(NCEL3,LSTPRP,X,Y,Z,CG,ILCHC, &
            KAPPA2, &
            LNCEL,LNCEL,LNCEL,LDCEL,LTRAN,LTRAN,LTRAN, &
            LXBCEN,LYBCEN,LZBCEN, &
            RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
            RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
            MAJOR,MINOR,QPROLATE, &
            ICHC,NCLX,NCLY,NCLZ,DCEL,XBCEN,YBCEN,ZBCEN, &
            QPBC,QA,QB)
       IF(TIMER.GT.1)CALL WRTTIM('Charge Distribution in large box:')

       !------------------------------------------------------------------------
       !     In vacuum (large box - focussing - original box)
       !------------------------------------------------------------------------

       IF(QFOCUS) THEN
          ! get the grid parameters in the lagre box
          CALL MAYER(NTPRP,LSTPRP,ONE,ONE,ZERO,X,Y,Z,CG, &
               PBRAD,LNCEL,LNCEL,LNCEL,LDCEL,WATR,IONR,MAY, &
               MAYX,MAYY,MAYZ,SWIN,ILCHC, &
               IPHIP,ZERO,TMEMB,ZMEMB,ONE,NIMGB, &
               HTMEMB,ONE,HDENS,POSALP,NEGALP,QCHRG,POFFST,NOFFST, &
               DROPLET,ONE,XDROPLET,YDROPLET,ZDROPLET,Qdtom,Qdkap, &
               ONE,LXMAX,LYMAX,LZMAX,LXMIN,LYMIN,LZMIN,Qbtom,Qbkap, &
               ONE,Xcyln,Ycyln,Zcyln,Rcyln,Hcyln,Qctom,Qckap, &
               ONE,Xcone,Ycone,Zcone,Ax,By,Cz,Qectom,Qeckap, &
               -1,-1,&
               LTRAN,LTRAN,LTRAN,LXBCEN,LYBCEN,LZBCEN, &
               IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
               QKEEP,QREEN,QSMTH,QBSPL,QA,QB, &
               RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
               RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
               MAJOR,MINOR,QPROLATE, &
               .false.,.true.,.false.,IPHIB,QPBC,QNPBC, &
                                !              QINTBP QZERO
               QNONLINEAR,QPARTLINEAR,QFKAP, &
               IPOX,IPOY,IPOZ,MAPT,NATOM)
          ! get boundary potentials in the large box: CG -> ICHC
          IF(.NOT.QINTBP.AND..NOT.QPBC.AND..NOT.QZERO)THEN
             CALL MAYER_BP_ALL(LNCEL3,LSTPRP,X,Y,Z,CG,ONE,ZERO, &
                  TMEMB,ILCHC,IPHIP,LNCEL,LNCEL,LNCEL,LDCEL, &
                  LTRAN,LTRAN,LTRAN,LXBCEN,LYBCEN,LZBCEN, &
                  RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
                  RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
                  MAJOR,MINOR,QPROLATE, &
                  LNCEL,LNCEL,LNCEL,LDCEL,LXBCEN,LYBCEN,LZBCEN, &
                  NIMGB,QNPBC,QA,QB)
          ELSEIF(QINTBP) THEN
             CALL MAYER_BP_INT(LNCEL3,LSTPRP,X,Y,Z,CG,ONE,ZERO, &
                  TMEMB,ILCHC,IPHIP,LNCEL,LNCEL,LNCEL,LDCEL, &
                  LTRAN,LTRAN,LTRAN,LXBCEN,LYBCEN,LZBCEN, &
                  RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
                  RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
                  MAJOR,MINOR,QPROLATE, &
                  LNCEL,LNCEL,LNCEL,LDCEL,LXBCEN,LYBCEN,LZBCEN, &
                  NIMGB,QNPBC,QA,QB)
          ENDIF
          IF(TIMER.GT.1) &
               CALL WRTTIM('Boundary potential in large box in vacuum:')

          IF(EPSP.ne.ONE) THEN
             ! vacuum environment in region B.
             CALL RECT_MAYER(LNCEL,LNCEL,LNCEL,LTRAN,LTRAN,LTRAN,LDCEL, &
                  LXBCEN,LYBCEN,LZBCEN, &
                  MAY,MAYX,MAYY,MAYZ, &
                  RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX)
             ! redefine the maximum region outside which EPS is ONE. (see subroutine MAYER)
             IXMAX=LNCEL-1
             IYMAX=LNCEL-1
             IZMAX=LNCEL-1
             IXMIN=2
             IYMIN=2
             IZMIN=2
          ENDIF

          ! solve PBEQ in the large box
          ! XIAO_PHK_QC_UW0609: consistent with PB reference 
          !        CALL PBEQ1(MAXITS,DOME,DEPS,ONE,EPSP,ONE,ZERO, &
          CALL PBEQ1(MAXITS,DOME,DEPS,ONE,ONE,ONE,ZERO, &
               LNCEL,LNCEL,LNCEL,IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
               IPHIP,MAYX,MAYY,MAYZ, &
               ILCHC,MAY,TMEMB, &
               .false.,.false.,QPBC,QNPBC,.false.)
          IF(TIMER.GT.1) &
               CALL WRTTIM('PBEQ solver in large box in vacuum:')

          ! construct boundary potentials from the large box
          CALL BOUNDPOT(LNCEL,LNCEL,LNCEL,LDCEL, &
               LTRAN,LTRAN,LTRAN,LXBCEN,LYBCEN,LZBCEN, &
               NCLX,NCLY,NCLZ,DCEL,XBCEN,YBCEN,ZBCEN, &
               IPHIP,IPHIB)

          ! get the grid parameters in the original box
          CALL MAYER(NTPRP,LSTPRP,ONE,ONE,ZERO,X,Y,Z,CG, &
               PBRAD,NCLX,NCLY,NCLZ,DCEL,WATR,IONR,MAY, &
               MAYX,MAYY,MAYZ,SWIN,ICHC, &
               IPHIP,ZERO,TMEMB,ZMEMB,ONE,NIMGB, &
               HTMEMB,ONE,HDENS,POSALP,NEGALP,QCHRG,POFFST,NOFFST, &
               DROPLET,ONE,XDROPLET,YDROPLET,ZDROPLET,Qdtom,Qdkap, &
               ONE,LXMAX,LYMAX,LZMAX,LXMIN,LYMIN,LZMIN,Qbtom,Qbkap, &
               ONE,Xcyln,Ycyln,Zcyln,Rcyln,Hcyln,Qctom,Qckap, &
               ONE,Xcone,Ycone,Zcone,Ax,By,Cz,Qectom,Qeckap, &
               -1,-1,&
               TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
               IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
               QKEEP,QREEN,QSMTH,QBSPL,QA,QB, &
               RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
               RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
               MAJOR,MINOR,QPROLATE, &
               .false.,.false.,.true.,IPHIB,QPBC,QNPBC, &
                                !              QINTBP QZERO
               QNONLINEAR,QPARTLINEAR,QFKAP, &
               IPOX,IPOY,IPOZ,MAPT,NATOM)
          IF(EPSP.ne.ONE) THEN
             ! vacuum environment in region B.
             CALL RECT_MAYER(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,DCEL, &
                  XBCEN,YBCEN,ZBCEN, &
                  MAY,MAYX,MAYY,MAYZ, &
                  RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX)
             ! Haibo GSBP (XIAO_QC_UW0609) for boundary
             ! redefine the maximum region outside which EPS is EPSW. 
             !    (see subroutine MAYER)
             IF(JX2.GT.IXMAX) IXMAX=JX2+1
             IF(JY2.GT.IYMAX) IYMAX=JY2+1
             IF(JZ2.GT.IZMAX) IZMAX=JZ2+1
             IF(JX1.LT.IXMIN) IXMIN=JX1-1
             IF(JY1.LT.IYMIN) IYMIN=JY1-1
             IF(JZ1.LT.IZMIN) IZMIN=JZ1-1
             ! Haibo GSBP

          ENDIF

          ! solve PBEQ in the original box
          ! XIAO_PHK_QC_UW0609: consistent with PB reference 
          !        CALL PBEQ1(MAXITS,DOME,DEPS,ONE,EPSP,ONE,ZERO, &
          CALL PBEQ1(MAXITS,DOME,DEPS,ONE,ONE,ONE,ZERO, &
               NCLX,NCLY,NCLZ,IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
               IPHIP,MAYX,MAYY,MAYZ, &
               ICHC,MAY,TMEMB, &
               .false.,QFOCUS,QPBC,QNPBC,.false.)
          IF(TIMER.GT.1) CALL WRTTIM('PBEQ solver in vacuum:')

       ELSE

          ! get the grid parameters in the original box
          CALL MAYER(NTPRP,LSTPRP,ONE,ONE,ZERO,X,Y,Z,CG, &
               PBRAD,NCLX,NCLY,NCLZ,DCEL,WATR,IONR,MAY, &
               MAYX,MAYY,MAYZ,SWIN,ICHC, &
               IPHIP,ZERO,TMEMB,ZMEMB,ONE,NIMGB, &
               HTMEMB,ONE,HDENS,POSALP,NEGALP,QCHRG,POFFST,NOFFST, &
               DROPLET,ONE,XDROPLET,YDROPLET,ZDROPLET,Qdtom,Qdkap, &
               ONE,LXMAX,LYMAX,LZMAX,LXMIN,LYMIN,LZMIN,Qbtom,Qbkap, &
               ONE,Xcyln,Ycyln,Zcyln,Rcyln,Hcyln,Qctom,Qckap, &
               ONE,Xcone,Ycone,Zcone,Ax,By,Cz,Qectom,Qeckap, &
               -1,-1,&
               TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
               IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
               QKEEP,QREEN,QSMTH,QBSPL,QA,QB, &
               RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
               RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
               MAJOR,MINOR,QPROLATE, &
               .false.,.true.,.false.,IPHIB,QPBC,QNPBC, &
                                !              QINTBP QZERO
               QNONLINEAR,QPARTLINEAR,QFKAP, &
               IPOX,IPOY,IPOZ,MAPT,NATOM)

          IF(.NOT.QINTBP.AND..NOT.QPBC.AND..NOT.QZERO)THEN
             CALL MAYER_BP_ALL(LNCEL3,LSTPRP,X,Y,Z,CG,ONE,ZERO, &
                  TMEMB,ILCHC,IPHIP,NCLX,NCLY,NCLZ,DCEL, &
                  TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
                  RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
                  RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
                  MAJOR,MINOR,QPROLATE, &
                  LNCEL,LNCEL,LNCEL,LDCEL,LXBCEN,LYBCEN,LZBCEN, &
                  NIMGB,QNPBC,QA,QB)
          ELSEIF(QINTBP) THEN
             CALL MAYER_BP_INT(LNCEL3,LSTPRP,X,Y,Z,CG,ONE,ZERO, &
                  TMEMB,ILCHC,IPHIP,NCLX,NCLY,NCLZ,DCEL, &
                  TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
                  RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
                  RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
                  MAJOR,MINOR,QPROLATE, &
                  LNCEL,LNCEL,LNCEL,LDCEL,LXBCEN,LYBCEN,LZBCEN, &
                  NIMGB,QNPBC,QA,QB)
          ENDIF
          IF(TIMER.GT.1) &
               CALL WRTTIM('Boundary potential with large box in vacuum:')

          IF(EPSP.ne.ONE) THEN
             ! vacuum environment in region B.
             CALL RECT_MAYER(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,DCEL, &
                  XBCEN,YBCEN,ZBCEN, &
                  MAY,MAYX,MAYY,MAYZ, &
                  RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX)
             ! Haibo GSBP (XIAO_QC_UW0609 for boundary)
             ! redefine the maximum region outside which EPS is EPSW. 
             !  (see subroutine MAYER)
             IF(JX2.GT.IXMAX) IXMAX=JX2+1
             IF(JY2.GT.IYMAX) IYMAX=JY2+1
             IF(JZ2.GT.IZMAX) IZMAX=JZ2+1
             IF(JX1.LT.IXMIN) IXMIN=JX1-1
             IF(JY1.LT.IYMIN) IYMIN=JY1-1
             IF(JZ1.LT.IZMIN) IZMIN=JZ1-1
             ! Haibo GSBP

          ENDIF

          ! solve PBEQ in the original box
          ! XIAO_PHK_QC_UW0609: consistent with PB reference 
          !        CALL PBEQ1(MAXITS,DOME,DEPS,ONE,EPSP,ONE,ZERO, &
          CALL PBEQ1(MAXITS,DOME,DEPS,ONE,ONE,ONE,ZERO, &
               NCLX,NCLY,NCLZ,IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
               IPHIP,MAYX,MAYY,MAYZ, &
               ICHC,MAY,TMEMB, &
               .false.,QFOCUS,QPBC,QNPBC,.false.)
          IF(TIMER.GT.1) CALL WRTTIM('PBEQ solver in vacuum:')

       ENDIF

       !------------------------------------------------------------------------
       !     In solution (large box - focussing - original box)
       !------------------------------------------------------------------------

       IF(QFOCUS) THEN
          ! get grid parameters in the large box
          CALL MAYER(NTPRP,LSTPRP,EPSW,EPSP,KAPPA2,X,Y,Z,CG, &
               PBRAD,LNCEL,LNCEL,LNCEL,LDCEL,WATR,IONR,MAY, &
               MAYX,MAYY,MAYZ,SWIN,ILCHC, &
               IPHIW,VMEMB,TMEMB,ZMEMB,EPSM,NIMGB, &
               HTMEMB,EPSH,HDENS,POSALP,NEGALP,QCHRG,POFFST,NOFFST, &
               DROPLET,EPSD,XDROPLET,YDROPLET,ZDROPLET,Qdtom,Qdkap, &
               EPSB,LXMAX,LYMAX,LZMAX,LXMIN,LYMIN,LZMIN,Qbtom,Qbkap, &
               EPSC,Xcyln,Ycyln,Zcyln,Rcyln,Hcyln,Qctom,Qckap, &
               EPSEC,Xcone,Ycone,Zcone,Ax,By,Cz,Qectom,Qeckap, &
               -1,-1,&
               LTRAN,LTRAN,LTRAN,LXBCEN,LYBCEN,LZBCEN, &
               IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
               QKEEP,QREEN,QSMTH,QBSPL,QA,QB, &
               RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
               RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
               MAJOR,MINOR,QPROLATE, &
               .false.,.true.,.false.,IPHIB,QPBC,QNPBC, &
                                !              QINTBP QZERO
               QNONLINEAR,QPARTLINEAR,QFKAP, &
               IPOX,IPOY,IPOZ,MAPT,NATOM)

          ! get boundary potentials in the large box: CG -> ICHC
          IF(.NOT.QINTBP.AND..NOT.QPBC.AND..NOT.QZERO)THEN
             CALL MAYER_BP_ALL(LNCEL3,LSTPRP,X,Y,Z,CG,EPSW,KAPPA2, &
                  TMEMB,ILCHC,IPHIW,LNCEL,LNCEL,LNCEL,LDCEL, &
                  LTRAN,LTRAN,LTRAN,LXBCEN,LYBCEN,LZBCEN, &
                  RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
                  RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
                  MAJOR,MINOR,QPROLATE, &
                  LNCEL,LNCEL,LNCEL,LDCEL,LXBCEN,LYBCEN,LZBCEN, &
                  NIMGB,QNPBC,QA,QB)
          ELSEIF(QINTBP) THEN
             CALL MAYER_BP_INT(LNCEL3,LSTPRP,X,Y,Z,CG,EPSW,KAPPA2, &
                  TMEMB,ILCHC,IPHIW,LNCEL,LNCEL,LNCEL,LDCEL, &
                  LTRAN,LTRAN,LTRAN,LXBCEN,LYBCEN,LZBCEN, &
                  RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
                  RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
                  MAJOR,MINOR,QPROLATE, &
                  LNCEL,LNCEL,LNCEL,LDCEL,LXBCEN,LYBCEN,LZBCEN, &
                  NIMGB,QNPBC,QA,QB)
          ENDIF
          IF(TIMER.GT.1) &
               CALL WRTTIM('Boundary potential in large box in solvent:')

          ! vacuum environment in region B.
          CALL RECT_MAYER(LNCEL,LNCEL,LNCEL,LTRAN,LTRAN,LTRAN,LDCEL, &
               LXBCEN,LYBCEN,LZBCEN, &
               MAY,MAYX,MAYY,MAYZ, &
               RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX)

          ! redefine the maximum region outside which EPS is EPSW. (see subroutine MAYER)
          IXMAX=LNCEL-1
          IYMAX=LNCEL-1
          IZMAX=LNCEL-1
          IXMIN=2
          IYMIN=2
          IZMIN=2

          ! solve PBEQ in the large box
          CALL PBEQ1(MAXITS,DOME,DEPS,EPSW,EPSP,EPSM,KAPPA2, &
               LNCEL,LNCEL,LNCEL,IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
               IPHIW,MAYX,MAYY,MAYZ, &
               ILCHC,MAY,TMEMB, &
               QOSOR,.false.,QPBC,QNPBC,.false.)
          IF(TIMER.GT.1) &
               CALL WRTTIM('PBEQ solver in large box in solvent:')

          ! construct boundary potentials from the large box
          CALL BOUNDPOT(LNCEL,LNCEL,LNCEL,LDCEL, &
               LTRAN,LTRAN,LTRAN,LXBCEN,LYBCEN,LZBCEN, &
               NCLX,NCLY,NCLZ,DCEL,XBCEN,YBCEN,ZBCEN, &
               IPHIW,IPHIB)

          ! get the grid parameters in the original box
          CALL MAYER(NTPRP,LSTPRP,EPSW,EPSP,KAPPA2,X,Y,Z,CG, &
               PBRAD,NCLX,NCLY,NCLZ,DCEL,WATR,IONR,MAY, &
               MAYX,MAYY,MAYZ,SWIN,ICHC, &
               IPHIW,VMEMB,TMEMB,ZMEMB,EPSM,NIMGB, &
               HTMEMB,EPSH,HDENS,POSALP,NEGALP,QCHRG,POFFST,NOFFST, &
               DROPLET,EPSD,XDROPLET,YDROPLET,ZDROPLET,Qdtom,Qdkap, &
               EPSB,LXMAX,LYMAX,LZMAX,LXMIN,LYMIN,LZMIN,Qbtom,Qbkap, &
               EPSC,Xcyln,Ycyln,Zcyln,Rcyln,Hcyln,Qctom,Qckap, &
               EPSEC,Xcone,Ycone,Zcone,Ax,By,Cz,Qectom,Qeckap, &
               -1,-1,&
               TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
               IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
               QKEEP,QREEN,QSMTH,QBSPL,QA,QB, &
               RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
               RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
               MAJOR,MINOR,QPROLATE, &
               .false.,.false.,.true.,IPHIB,QPBC,QNPBC, &
                                !              QINTBP QZERO
               QNONLINEAR,QPARTLINEAR,QFKAP, &
               IPOX,IPOY,IPOZ,MAPT,NATOM)

          ! vacuum environment in region B.
          CALL RECT_MAYER(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,DCEL, &
               XBCEN,YBCEN,ZBCEN, &
               MAY,MAYX,MAYY,MAYZ, &
               RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX)

          ! redefine the maximum region outside which EPS is EPSW. (see subroutine MAYER)
          IF(JX2.GT.IXMAX) IXMAX=JX2+1
          IF(JY2.GT.IYMAX) IYMAX=JY2+1
          IF(JZ2.GT.IZMAX) IZMAX=JZ2+1
          IF(JX1.LT.IXMIN) IXMIN=JX1-1
          IF(JY1.LT.IYMIN) IYMIN=JY1-1
          IF(JZ1.LT.IZMIN) IZMIN=JZ1-1

          ! solve PBEQ in the original box
          CALL PBEQ1(MAXITS,DOME,DEPS,EPSW,EPSP,EPSM,KAPPA2, &
               NCLX,NCLY,NCLZ,IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
               IPHIW,MAYX,MAYY,MAYZ, &
               ICHC,MAY,TMEMB, &
               QOSOR,QFOCUS,QPBC,QNPBC,.false.)
          IF(TIMER.GT.1) CALL WRTTIM('PBEQ solver in solvent:')

       ELSE

          ! get the grid parameters in the original box
          CALL MAYER(NTPRP,LSTPRP,EPSW,EPSP,KAPPA2,X,Y,Z,CG, &
               PBRAD,NCLX,NCLY,NCLZ,DCEL,WATR,IONR,MAY, &
               MAYX,MAYY,MAYZ,SWIN,ICHC, &
               IPHIW,VMEMB,TMEMB,ZMEMB,EPSM,NIMGB, &
               HTMEMB,EPSH,HDENS,POSALP,NEGALP,QCHRG,POFFST,NOFFST, &
               DROPLET,EPSD,XDROPLET,YDROPLET,ZDROPLET,Qdtom,Qdkap, &
               EPSB,LXMAX,LYMAX,LZMAX,LXMIN,LYMIN,LZMIN,Qbtom,Qbkap, &
               EPSC,Xcyln,Ycyln,Zcyln,Rcyln,Hcyln,Qctom,Qckap, &
               EPSEC,Xcone,Ycone,Zcone,Ax,By,Cz,Qectom,Qeckap, &
               -1,-1,&
               TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
               IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
               QKEEP,QREEN,QSMTH,QBSPL,QA,QB, &
               RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
               RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
               MAJOR,MINOR,QPROLATE, &
               .false.,.true.,.false.,IPHIB,QPBC,QNPBC, &
                                !              QINTBP QZERO
               QNONLINEAR,QPARTLINEAR,QFKAP, &
               IPOX,IPOY,IPOZ,MAPT,NATOM)

          IF(.NOT.QINTBP.AND..NOT.QPBC.AND..NOT.QZERO)THEN
             CALL MAYER_BP_ALL(LNCEL3,LSTPRP,X,Y,Z,CG,EPSW,KAPPA2, &
                  TMEMB,ILCHC,IPHIW,NCLX,NCLY,NCLZ,DCEL, &
                  TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
                  RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
                  RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
                  MAJOR,MINOR,QPROLATE, &
                  LNCEL,LNCEL,LNCEL,LDCEL,LXBCEN,LYBCEN,LZBCEN, &
                  NIMGB,QNPBC,QA,QB)
          ELSEIF(QINTBP) THEN
             CALL MAYER_BP_INT(LNCEL3,LSTPRP,X,Y,Z,CG,EPSW,KAPPA2, &
                  TMEMB,ILCHC,IPHIW,NCLX,NCLY,NCLZ,DCEL, &
                  TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
                  RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
                  RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
                  MAJOR,MINOR,QPROLATE, &
                  LNCEL,LNCEL,LNCEL,LDCEL,LXBCEN,LYBCEN,LZBCEN, &
                  NIMGB,QNPBC,QA,QB)
          ENDIF
          IF(TIMER.GT.1) &
               CALL WRTTIM('Boundary potential with large box in solvent:')

          ! vacuum environment in region B.
          CALL RECT_MAYER(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,DCEL, &
               XBCEN,YBCEN,ZBCEN, &
               MAY,MAYX,MAYY,MAYZ, &
               RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX)

          ! redefine the maximum region outside which EPS is EPSW. (see subroutine MAYER)
          IF(JX2.GT.IXMAX) IXMAX=JX2+1
          IF(JY2.GT.IYMAX) IYMAX=JY2+1
          IF(JZ2.GT.IZMAX) IZMAX=JZ2+1
          IF(JX1.LT.IXMIN) IXMIN=JX1-1
          IF(JY1.LT.IYMIN) IYMIN=JY1-1
          IF(JZ1.LT.IZMIN) IZMIN=JZ1-1

          ! solve PBEQ in the original box
          CALL PBEQ1(MAXITS,DOME,DEPS,EPSW,EPSP,EPSM,KAPPA2, &
               NCLX,NCLY,NCLZ,IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
               IPHIW,MAYX,MAYY,MAYZ, &
               ICHC,MAY,TMEMB, &
               QOSOR,QFOCUS,QPBC,QNPBC,.false.)
          IF(TIMER.GT.1) CALL WRTTIM('PBEQ solver in solvent:')

       ENDIF

       ! calculate the reaction field energy elements M(ij)
       CALL RECT_MIJ(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,NTPOL,NORDER, &
            LSTPX,LSTPY,LSTPZ,BNORM,DCEL, &
            XBCEN,YBCEN,ZBCEN, &
            RRXCEN,RRYCEN,RRZCEN,XSCALE,YSCALE,ZSCALE,IIP0,NPGD, &
            IPHIP,IPHIW,MIJ,CGSCAL)

    ENDDO
    call chmdealloc('pbeq.src','RECT_GSBP1','ILCHC',LNCEL3,cr4=ILCHC)
    call chmdealloc('pbeq.src','RECT_GSBP1','IIP0',NPROI,intg=IIP0)
    call chmdealloc('pbeq.src','RECT_GSBP1','IPHIB',BNCEL,cr4=IPHIB)
    QA=.false.
    QB=.false.
    !
    RETURN
  END SUBROUTINE RECT_GSBP1
  !
  SUBROUTINE RECT_GSBP2(NATOM,X,Y,Z,CG,QPRIN)
    !-----------------------------------------------------------------------
    !     The reaction field energy and forces of the region of interest
    !     are calculated using the Legendre Polynomials
    !
    use chm_kinds
    use memory
    use number
    use stream
    use dimens_fcm
    use consta
    use timerm
    use machutil,only:wrttim
    implicit none
    !
    INTEGER NATOM
    real(chm_real)  X(*),Y(*),Z(*),CG(*)
    LOGICAL QPRIN
    ! local
    integer,allocatable,dimension(:) :: IIP0
    real(chm_real4),allocatable,dimension(:) :: IPHIB
    INTEGER NORDER,OLDNTP
    INTEGER JX1,JX2,JY1,JY2,JZ1,JZ2,NPROI,NPGD

    ! save basis functions
    OLDNTP= OLDXNP*OLDYNP*OLDZNP
    CALL RECTPOL(XNPOL,YNPOL,ZNPOL,OLDNTP,QMIJ, &
         LSTPX,LSTPY,LSTPZ,LSTPOL)

    ! calculate the normalization constants of the basis functions
    CALL RECT_NORM(NTPOL,XSCALE,YSCALE,ZSCALE, &
         LSTPX,LSTPY,LSTPZ,BNORM)

    IF(QMIJ.AND.OLDNTP.EQ.NTPOL) RETURN

    call chmalloc('pbeq.src','RECT_GSBP2','IPHIB',1,cr4=IPHIB)

    ! find out the grid points inside region of interest
    JX1=INT((TRANX+RBXMIN-XBCEN)/DCEL)-1
    JX2=INT((TRANX+RBXMAX-XBCEN)/DCEL)+3
    JY1=INT((TRANY+RBYMIN-YBCEN)/DCEL)-1
    JY2=INT((TRANY+RBYMAX-YBCEN)/DCEL)+3
    JZ1=INT((TRANZ+RBZMIN-ZBCEN)/DCEL)-1
    JZ2=INT((TRANZ+RBZMAX-ZBCEN)/DCEL)+3
    NPROI=(JX2-JX1)*(JY2-JY1)*(JZ2-JZ1)
    call chmalloc('pbeq.src','RECT_GSBP2','IIP0',NPROI,intg=IIP0)


    CALL RECT_GPINRR(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,DCEL, &
         XBCEN,YBCEN,ZBCEN, &
         RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX, &
         JX1,JX2,JY1,JY2,JZ1,JZ2,IIP0,NPGD)

    ! skip the initialization of CHC array and
    !      the all charge distribution calculations in MAYER
    QA=.true.
    QB=.true.

    DO NORDER=OLDNTP+1,NTPOL

       IF(PRNLEV.GT.7.AND.NORDER.EQ.1) WRITE(OUTU,'(A)')
       IF(PRNLEV.GT.7) WRITE(OUTU,'(3x,a,i5,a)') &
            'Reaction field calculation for',NORDER,' basis function'

       ! replace ICHC (CDEN) array with the basis functions (Legendre Polynomials)
       CALL RECT_CHC(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,DCEL, &
            XBCEN,YBCEN,ZBCEN, &
            NORDER,LSTPX,LSTPY,LSTPZ,BNORM, &
            ICHC,CGSCAL,IIP0,NPGD, &
            RRXCEN,RRYCEN,RRZCEN,XSCALE,YSCALE,ZSCALE)

       !------------------------------------------------------------------------
       !     In vacuum
       !------------------------------------------------------------------------

       ! get the grid parameters
       CALL MAYER(NTPRP,LSTPRP,ONE,ONE,ZERO,X,Y,Z,CG, &
            PBRAD,NCLX,NCLY,NCLZ,DCEL,WATR,IONR,MAY, &
            MAYX,MAYY,MAYZ,SWIN,ICHC, &
            IPHIP,ZERO,TMEMB,ZMEMB,ONE,NIMGB, &
            HTMEMB,ONE,HDENS,POSALP,NEGALP,QCHRG,POFFST,NOFFST, &
            DROPLET,ONE,XDROPLET,YDROPLET,ZDROPLET,Qdtom,Qdkap, &
            ONE,LXMAX,LYMAX,LZMAX,LXMIN,LYMIN,LZMIN,Qbtom,Qbkap, &
            ONE,Xcyln,Ycyln,Zcyln,Rcyln,Hcyln,Qctom,Qckap, &
            ONE,Xcone,Ycone,Zcone,Ax,By,Cz,Qectom,Qeckap, &
               -1,-1,&
            TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
            IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
            QKEEP,QREEN,QSMTH,QBSPL,QA,QB, &
            RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
            RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
            MAJOR,MINOR,QPROLATE, &
            .false.,.true.,.false.,IPHIB,QPBC,QNPBC, &
                                !              QINTBP QZERO
            QNONLINEAR,QPARTLINEAR,QFKAP, &
            IPOX,IPOY,IPOZ,MAPT,NATOM)

       ! get boundary potentials : CG -> ICHC
       IF(.NOT.QINTBP.AND..NOT.QPBC.AND..NOT.QZERO)THEN
          CALL MAYER_BP_ALL(NCEL3,LSTPRP,X,Y,Z,CG,ONE,ZERO, &
               TMEMB,ICHC,IPHIP,NCLX,NCLY,NCLZ,DCEL, &
               TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
               RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
               RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
               MAJOR,MINOR,QPROLATE, &
               NCLX,NCLY,NCLZ,DCEL,XBCEN,YBCEN,ZBCEN, &
               NIMGB,QNPBC,QA,QB)
       ELSEIF(QINTBP) THEN
          CALL MAYER_BP_INT(NCEL3,LSTPRP,X,Y,Z,CG,ONE,ZERO, &
               TMEMB,ICHC,IPHIP,NCLX,NCLY,NCLZ,DCEL, &
               TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
               RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
               RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
               MAJOR,MINOR,QPROLATE, &
               NCLX,NCLY,NCLZ,DCEL,XBCEN,YBCEN,ZBCEN, &
               NIMGB,QNPBC,QA,QB)
       ENDIF
       IF(TIMER.GT.1) CALL WRTTIM('Boundary potential in vacuum:')

       IF(EPSP.ne.ONE) THEN
          ! vacuum environment in region B.
          CALL RECT_MAYER(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,DCEL, &
               XBCEN,YBCEN,ZBCEN, &
               MAY,MAYX,MAYY,MAYZ, &
               RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX)
          ! Haibo GSBP (XIAO_QC_UW0609 for boundary)
          ! redefine the maximum region outside which EPS is EPSW. 
          !(see subroutine MAYER)
          IF(JX2.GT.IXMAX) IXMAX=JX2+1
          IF(JY2.GT.IYMAX) IYMAX=JY2+1
          IF(JZ2.GT.IZMAX) IZMAX=JZ2+1
          IF(JX1.LT.IXMIN) IXMIN=JX1-1
          IF(JY1.LT.IYMIN) IYMIN=JY1-1
          IF(JZ1.LT.IZMIN) IZMIN=JZ1-1
          ! Haibo GSBP

       ENDIF

       ! solve PBEQ
       ! XIAO_PHK_QC_UW0609: consistent with PB reference 
       !        CALL PBEQ1(MAXITS,DOME,DEPS,ONE,EPSP,ONE,ZERO, &
       CALL PBEQ1(MAXITS,DOME,DEPS,ONE,ONE,ONE,ZERO, &
            NCLX,NCLY,NCLZ,IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
            IPHIP,MAYX,MAYY,MAYZ, &
            ICHC,MAY,TMEMB, &
            .false.,.false.,QPBC,QNPBC,.false.)
       IF(TIMER.GT.1) CALL WRTTIM('PBEQ solver in vacuum:')

       !------------------------------------------------------------------------
       !     In solution
       !------------------------------------------------------------------------

       ! get grid parameters
       CALL MAYER(NTPRP,LSTPRP,EPSW,EPSP,KAPPA2,X,Y,Z,CG, &
            PBRAD,NCLX,NCLY,NCLZ,DCEL,WATR,IONR,MAY, &
            MAYX,MAYY,MAYZ,SWIN,ICHC, &
            IPHIW,VMEMB,TMEMB,ZMEMB,EPSM,NIMGB, &
            HTMEMB,EPSH,HDENS,POSALP,NEGALP,QCHRG,POFFST,NOFFST, &
            DROPLET,EPSD,XDROPLET,YDROPLET,ZDROPLET,Qdtom,Qdkap, &
            EPSB,LXMAX,LYMAX,LZMAX,LXMIN,LYMIN,LZMIN,Qbtom,Qbkap, &
            EPSC,Xcyln,Ycyln,Zcyln,Rcyln,Hcyln,Qctom,Qckap, &
            EPSEC,Xcone,Ycone,Zcone,Ax,By,Cz,Qectom,Qeckap, &
               -1,-1,&
            TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
            IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
            QKEEP,QREEN,QSMTH,QBSPL,QA,QB, &
            RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
            RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
            MAJOR,MINOR,QPROLATE, &
            .false.,.true.,.false.,IPHIB,QPBC,QNPBC, &
                                !              QINTBP QZERO
            QNONLINEAR,QPARTLINEAR,QFKAP, &
            IPOX,IPOY,IPOZ,MAPT,NATOM)

       ! get boundary potentials : CG -> ICHC
       IF(.NOT.QINTBP.AND..NOT.QPBC.AND..NOT.QZERO)THEN
          CALL MAYER_BP_ALL(NCEL3,LSTPRP,X,Y,Z,CG,EPSW,KAPPA2, &
               TMEMB,ICHC,IPHIW,NCLX,NCLY,NCLZ,DCEL, &
               TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
               RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
               RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
               MAJOR,MINOR,QPROLATE, &
               NCLX,NCLY,NCLZ,DCEL,XBCEN,YBCEN,ZBCEN, &
               NIMGB,QNPBC,QA,QB)
       ELSEIF(QINTBP) THEN
          CALL MAYER_BP_INT(NCEL3,LSTPRP,X,Y,Z,CG,EPSW,KAPPA2, &
               TMEMB,ICHC,IPHIW,NCLX,NCLY,NCLZ,DCEL, &
               TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
               RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
               RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
               MAJOR,MINOR,QPROLATE, &
               NCLX,NCLY,NCLZ,DCEL,XBCEN,YBCEN,ZBCEN, &
               NIMGB,QNPBC,QA,QB)
       ENDIF
       IF(TIMER.GT.1) CALL WRTTIM('Boundary potential in solvent:')

       ! vacuum environment in region B.
       CALL RECT_MAYER(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,DCEL, &
            XBCEN,YBCEN,ZBCEN, &
            MAY,MAYX,MAYY,MAYZ, &
            RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX)

       ! redefine the maximum region outside which EPS is EPSW. (see subroutine MAYER)
       IF(JX2.GT.IXMAX) IXMAX=JX2+1
       IF(JY2.GT.IYMAX) IYMAX=JY2+1
       IF(JZ2.GT.IZMAX) IZMAX=JZ2+1
       IF(JX1.LT.IXMIN) IXMIN=JX1-1
       IF(JY1.LT.IYMIN) IYMIN=JY1-1
       IF(JZ1.LT.IZMIN) IZMIN=JZ1-1

       ! solve PBEQ
       CALL PBEQ1(MAXITS,DOME,DEPS,EPSW,EPSP,EPSM,KAPPA2, &
            NCLX,NCLY,NCLZ,IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
            IPHIW,MAYX,MAYY,MAYZ, &
            ICHC,MAY,TMEMB, &
            QOSOR,.false.,QPBC,QNPBC,.false.)
       IF(TIMER.GT.1) CALL WRTTIM('PBEQ solver in solvent:')

       ! calculate the reaction field energy elements M(ij)
       CALL RECT_MIJ(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,NTPOL,NORDER, &
            LSTPX,LSTPY,LSTPZ,BNORM,DCEL, &
            XBCEN,YBCEN,ZBCEN, &
            RRXCEN,RRYCEN,RRZCEN,XSCALE,YSCALE,ZSCALE,IIP0,NPGD, &
            IPHIP,IPHIW,MIJ,CGSCAL)

    ENDDO

    call chmdealloc('pbeq.src','RECT_GSBP2','IIP0',NPROI,intg=IIP0)
    QA=.false.
    QB=.false.
    !
    RETURN
  END SUBROUTINE RECT_GSBP2
  !
  SUBROUTINE SPHE_GSBP1(NATOM,X,Y,Z,CG, &
       LNCEL,LTRAN,LDCEL,LXBCEN,LYBCEN,LZBCEN,QPRIN)
    !-----------------------------------------------------------------------
    !     The reaction field energy and forces of the region of interest
    !     are calculated using the following basis functions :
    !     basis funcition (l,m) = radial function (l) * spherical harmonics (l,m)
    !
    use chm_kinds
    use memory
    use number
    use stream
    use dimens_fcm
    use consta
    use timerm
    use machutil,only:wrttim
    implicit none
    !

    INTEGER NATOM
    real(chm_real)  X(*),Y(*),Z(*),CG(*)
    LOGICAL QPRIN
    INTEGER LNCEL
    real(chm_real)  LTRAN,LDCEL,LXBCEN,LYBCEN,LZBCEN
    ! local
    integer,allocatable,dimension(:) :: IIP0
    real(chm_real),allocatable,dimension(:) :: RGD,CTHETA,CPHI,SPHI
    real(chm_real4),allocatable,dimension(:) :: ILCHC,IPHIB
    INTEGER NORDER,OLDNTP
    INTEGER NFIL,IXCEN,IYCEN,IZCEN,JX1,JX2,JY1,JY2,JZ1,JZ2
    INTEGER NPROI,NPGD
    INTEGER LNCEL3,BNCEL

    ! save basis functions
    OLDNTP= OLDNMP*OLDNMP
    CALL SPHEPOL(NMPOL,LSTPL,LSTPM,LSTPOL)

    ! calculate the normalization constants of the basis functions
    CALL SPHE_NORM(NMPOL,BNORM,SRDIST)

    IF(QMIJ.AND.OLDNTP.EQ.NTPOL) RETURN

    ! calculate the trigonometric functions of each grid points
    ! inside region of interest
    NFIL=INT(SRDIST/DCEL)+2
    IXCEN=INT((RRXCEN+TRANX-XBCEN)/DCEL)+1
    IYCEN=INT((RRYCEN+TRANY-YBCEN)/DCEL)+1
    IZCEN=INT((RRZCEN+TRANZ-ZBCEN)/DCEL)+1
    JX1=IXCEN-NFIL
    JX2=IXCEN+NFIL+1
    JY1=IYCEN-NFIL
    JY2=IYCEN+NFIL+1
    JZ1=IZCEN-NFIL
    JZ2=IZCEN+NFIL+1
    NPROI =(JX2-JX1)*(JY2-JY1)*(JZ2-JZ1)
    call chmalloc('pbeq.src','SPHE_GSBP1','IIP0',NPROI,intg=IIP0)
    call chmalloc('pbeq.src','SPHE_GSBP1','RGD',NPROI,crl=RGD)
    call chmalloc('pbeq.src','SPHE_GSBP1','CTHETA',NPROI,crl=CTHETA)
    call chmalloc('pbeq.src','SPHE_GSBP1','CPHI',NPROI,crl=CPHI)
    call chmalloc('pbeq.src','SPHE_GSBP1','SPHI',NPROI,crl=SPHI)
    CALL SPHE_TRIG(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,DCEL, &
         XBCEN,YBCEN,ZBCEN, &
         IIP0,CTHETA,CPHI,SPHI,RGD, &
         SRDIST,RRXCEN,RRYCEN,RRZCEN,JX1,JX2,JY1,JY2,JZ1,JZ2,NPGD)

    ! allocate charge densities in the large box and
    !          boundary potentials in the original box
    LNCEL3=LNCEL*LNCEL*LNCEL
    call chmalloc('pbeq.src','SPHE_GSBP1','ILCHC',LNCEL3,cr4=ILCHC)
    BNCEL=1  
    IF(QFOCUS) BNCEL=2*(NCLX*NCLX+NCLY*NCLY+NCLZ*NCLZ)
    call chmalloc('pbeq.src','SPHE_GSBP1','IPHIB',BNCEL,cr4=IPHIB)
    ! skip the initialization of CHC array and
    !      the all charge distribution calculations in MAYER
    QA=.true.
    QB=.true.

    DO NORDER=OLDNTP+1,NTPOL

       IF(PRNLEV.GT.7.AND.NORDER.EQ.1) WRITE(OUTU,'(A)')
       IF(PRNLEV.GT.7) WRITE(OUTU,'(3x,a,i5,a)') &
            'Reaction field calculation for',NORDER,' basis function'

       ! replace ICHC (CDEN) array with the basis functions (in original box)
       CALL SPHE_CHC(NCEL3,NPGD,DCEL,NORDER, &
            IIP0,CTHETA,CPHI,SPHI,RGD, &
            LSTPL,LSTPM,BNORM,ICHC,CGSCAL)

       ! get charge distribution ILCHC in the large box from ICHC
       CALL INITGRID(LNCEL,LNCEL,LNCEL,ILCHC,ZERO)
       CALL MAYER_CD_TRIL(NCEL3,LSTPRP,X,Y,Z,CG,ILCHC, &
            KAPPA2, &
            LNCEL,LNCEL,LNCEL,LDCEL,LTRAN,LTRAN,LTRAN, &
            LXBCEN,LYBCEN,LZBCEN, &
            RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
            RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
            MAJOR,MINOR,QPROLATE, &
            ICHC,NCLX,NCLY,NCLZ,DCEL,XBCEN,YBCEN,ZBCEN, &
            QPBC,QA,QB)
       IF(TIMER.GT.1)CALL WRTTIM('Charge Distribution in large box:')

       !------------------------------------------------------------------------
       !     In vacuum (large box - focussing - original box)
       !------------------------------------------------------------------------

       IF(QFOCUS) THEN
          ! get the grid parameters in the lagre box
          CALL MAYER(NTPRP,LSTPRP,ONE,ONE,ZERO,X,Y,Z,CG, &
               PBRAD,LNCEL,LNCEL,LNCEL,LDCEL,WATR,IONR,MAY, &
               MAYX,MAYY,MAYZ,SWIN,ILCHC, &
               IPHIP,ZERO,TMEMB,ZMEMB,ONE,NIMGB, &
               HTMEMB,ONE,HDENS,POSALP,NEGALP,QCHRG,POFFST,NOFFST, &
               DROPLET,ONE,XDROPLET,YDROPLET,ZDROPLET,Qdtom,Qdkap, &
               ONE,LXMAX,LYMAX,LZMAX,LXMIN,LYMIN,LZMIN,Qbtom,Qbkap, &
               ONE,Xcyln,Ycyln,Zcyln,Rcyln,Hcyln,Qctom,Qckap, &
               ONE,Xcone,Ycone,Zcone,Ax,By,Cz,Qectom,Qeckap, &
               -1,-1,&
               LTRAN,LTRAN,LTRAN,LXBCEN,LYBCEN,LZBCEN, &
               IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
               QKEEP,QREEN,QSMTH,QBSPL,QA,QB, &
               RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
               RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
               MAJOR,MINOR,QPROLATE, &
               .false.,.true.,.false.,IPHIB,QPBC,QNPBC, &
                                !              QINTBP QZERO
               QNONLINEAR,QPARTLINEAR,QFKAP, &
               IPOX,IPOY,IPOZ,MAPT,NATOM)

          ! get boundary potentials in the large box: CG -> ICHC
          IF(.NOT.QINTBP.AND..NOT.QPBC.AND..NOT.QZERO)THEN
             CALL MAYER_BP_ALL(LNCEL3,LSTPRP,X,Y,Z,CG,ONE,ZERO, &
                  TMEMB,ILCHC,IPHIP,LNCEL,LNCEL,LNCEL,LDCEL, &
                  LTRAN,LTRAN,LTRAN,LXBCEN,LYBCEN,LZBCEN, &
                  RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
                  RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
                  MAJOR,MINOR,QPROLATE, &
                  LNCEL,LNCEL,LNCEL,LDCEL,LXBCEN,LYBCEN,LZBCEN, &
                  NIMGB,QNPBC,QA,QB)
          ELSEIF(QINTBP) THEN
             CALL MAYER_BP_INT(LNCEL3,LSTPRP,X,Y,Z,CG,ONE,ZERO, &
                  TMEMB,ILCHC,IPHIP,LNCEL,LNCEL,LNCEL,LDCEL, &
                  LTRAN,LTRAN,LTRAN,LXBCEN,LYBCEN,LZBCEN, &
                  RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
                  RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
                  MAJOR,MINOR,QPROLATE, &
                  LNCEL,LNCEL,LNCEL,LDCEL,LXBCEN,LYBCEN,LZBCEN, &
                  NIMGB,QNPBC,QA,QB)
          ENDIF
          IF(TIMER.GT.1) &
               CALL WRTTIM('Boundary potential in large box in vacuum:')

          IF(EPSP.ne.ONE) THEN
             ! vacuum environment in region B.
             CALL SPHE_MAYER(LNCEL,LNCEL,LNCEL,LTRAN,LTRAN,LTRAN,LDCEL, &
                  LXBCEN,LYBCEN,LZBCEN, &
                  MAY,MAYX,MAYY,MAYZ, &
                  RRXCEN,RRYCEN,RRZCEN,SRDIST)
             ! redefine the maximum region outside which EPS is ONE. (see subroutine MAYER)
             IXMAX=LNCEL-1
             IYMAX=LNCEL-1
             IZMAX=LNCEL-1
             IXMIN=2
             IYMIN=2
             IZMIN=2
          ENDIF

          ! solve PBEQ in the large box
          ! XIAO_PHK_QC_UW0609: consistent with PB reference 
          !        CALL PBEQ1(MAXITS,DOME,DEPS,ONE,EPSP,ONE,ZERO, &
          CALL PBEQ1(MAXITS,DOME,DEPS,ONE,ONE,ONE,ZERO, &
               LNCEL,LNCEL,LNCEL,IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
               IPHIP,MAYX,MAYY,MAYZ, &
               ILCHC,MAY,TMEMB, &
               .false.,.false.,QPBC,QNPBC,.false.)
          IF(TIMER.GT.1) &
               CALL WRTTIM('PBEQ solver in large box in vacuum:')

          ! construct boundary potentials from the large box
          CALL BOUNDPOT(LNCEL,LNCEL,LNCEL,LDCEL, &
               LTRAN,LTRAN,LTRAN,LXBCEN,LYBCEN,LZBCEN, &
               NCLX,NCLY,NCLZ,DCEL,XBCEN,YBCEN,ZBCEN, &
               IPHIP,IPHIB)

          ! get the grid parameters in the original box
          CALL MAYER(NTPRP,LSTPRP,ONE,ONE,ZERO,X,Y,Z,CG, &
               PBRAD,NCLX,NCLY,NCLZ,DCEL,WATR,IONR,MAY, &
               MAYX,MAYY,MAYZ,SWIN,ICHC, &
               IPHIP,ZERO,TMEMB,ZMEMB,ONE,NIMGB, &
               HTMEMB,ONE,HDENS,POSALP,NEGALP,QCHRG,POFFST,NOFFST, &
               DROPLET,ONE,XDROPLET,YDROPLET,ZDROPLET,Qdtom,Qdkap, &
               ONE,LXMAX,LYMAX,LZMAX,LXMIN,LYMIN,LZMIN,Qbtom,Qbkap, &
               ONE,Xcyln,Ycyln,Zcyln,Rcyln,Hcyln,Qctom,Qckap, &
               ONE,Xcone,Ycone,Zcone,Ax,By,Cz,Qectom,Qeckap, &
               -1,-1,&
               TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
               IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
               QKEEP,QREEN,QSMTH,QBSPL,QA,QB, &
               RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
               RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
               MAJOR,MINOR,QPROLATE, &
               .false.,.false.,.true.,IPHIB,QPBC,QNPBC, &
                                !              QINTBP QZERO
               QNONLINEAR,QPARTLINEAR,QFKAP, &
               IPOX,IPOY,IPOZ,MAPT, NATOM)

          IF(EPSP.ne.ONE) THEN
             ! vacuum environment in region B.
             CALL SPHE_MAYER(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,DCEL, &
                  XBCEN,YBCEN,ZBCEN, &
                  MAY,MAYX,MAYY,MAYZ, &
                  RRXCEN,RRYCEN,RRZCEN,SRDIST)
             ! Haibo GSBP (XIAO_QC_UW0609 for boundary)
             ! redefine the maximum region outside which EPS is EPSW.
             ! (see subroutine MAYER)
             IF(JX2.GT.IXMAX) IXMAX=JX2+1
             IF(JY2.GT.IYMAX) IYMAX=JY2+1
             IF(JZ2.GT.IZMAX) IZMAX=JZ2+1
             IF(JX1.LT.IXMIN) IXMIN=JX1-1
             IF(JY1.LT.IYMIN) IYMIN=JY1-1
             IF(JZ1.LT.IZMIN) IZMIN=JZ1-1
             ! Haibo GSBP

          ENDIF

          ! solve PBEQ in the original box
          ! XIAO_PHK_QC_UW0609: consistent with PB reference 
          !        CALL PBEQ1(MAXITS,DOME,DEPS,ONE,EPSP,ONE,ZERO, &
          CALL PBEQ1(MAXITS,DOME,DEPS,ONE,ONE,ONE,ZERO, &
               NCLX,NCLY,NCLZ,IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
               IPHIP,MAYX,MAYY,MAYZ, &
               ICHC,MAY,TMEMB, &
               .false.,QFOCUS,QPBC,QNPBC,.false.)
          IF(TIMER.GT.1) CALL WRTTIM('PBEQ solver in vacuum:')

       ELSE

          ! get the grid parameters in the original box
          CALL MAYER(NTPRP,LSTPRP,ONE,ONE,ZERO,X,Y,Z,CG, &
               PBRAD,NCLX,NCLY,NCLZ,DCEL,WATR,IONR,MAY, &
               MAYX,MAYY,MAYZ,SWIN,ICHC, &
               IPHIP,ZERO,TMEMB,ZMEMB,ONE,NIMGB, &
               HTMEMB,ONE,HDENS,POSALP,NEGALP,QCHRG,POFFST,NOFFST, &
               DROPLET,ONE,XDROPLET,YDROPLET,ZDROPLET,Qdtom,Qdkap, &
               ONE,LXMAX,LYMAX,LZMAX,LXMIN,LYMIN,LZMIN,Qbtom,Qbkap, &
               ONE,Xcyln,Ycyln,Zcyln,Rcyln,Hcyln,Qctom,Qckap, &
               ONE,Xcone,Ycone,Zcone,Ax,By,Cz,Qectom,Qeckap, &
               -1,-1,&
               TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
               IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
               QKEEP,QREEN,QSMTH,QBSPL,QA,QB, &
               RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
               RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
               MAJOR,MINOR,QPROLATE, &
               .false.,.true.,.false.,IPHIB,QPBC,QNPBC, &
                                !              QINTBP QZERO
               QNONLINEAR,QPARTLINEAR,QFKAP, &
               IPOX,IPOY,IPOZ,MAPT, NATOM)

          IF(.NOT.QINTBP.AND..NOT.QPBC.AND..NOT.QZERO)THEN
             CALL MAYER_BP_ALL(LNCEL3,LSTPRP,X,Y,Z,CG,ONE,ZERO, &
                  TMEMB,ILCHC,IPHIP,NCLX,NCLY,NCLZ,DCEL, &
                  TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
                  RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
                  RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
                  MAJOR,MINOR,QPROLATE, &
                  LNCEL,LNCEL,LNCEL,LDCEL,LXBCEN,LYBCEN,LZBCEN, &
                  NIMGB,QNPBC,QA,QB)
          ELSEIF(QINTBP) THEN
             CALL MAYER_BP_INT(LNCEL3,LSTPRP,X,Y,Z,CG,ONE,ZERO, &
                  TMEMB,ILCHC,IPHIP,NCLX,NCLY,NCLZ,DCEL, &
                  TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
                  RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
                  RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
                  MAJOR,MINOR,QPROLATE, &
                  LNCEL,LNCEL,LNCEL,LDCEL,LXBCEN,LYBCEN,LZBCEN, &
                  NIMGB,QNPBC,QA,QB)
          ENDIF
          IF(TIMER.GT.1) &
               CALL WRTTIM('Boundary potential with large box in vacuum:')

          IF(EPSP.ne.ONE) THEN
             ! vacuum environment in region B.
             CALL SPHE_MAYER(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,DCEL, &
                  XBCEN,YBCEN,ZBCEN, &
                  MAY,MAYX,MAYY,MAYZ, &
                  RRXCEN,RRYCEN,RRZCEN,SRDIST)
             ! Haibo GSBP (XIAO_QC_UW0609 for boundary)
             ! redefine the maximum region outside which EPS is EPSW.
             ! (see subroutine MAYER)
             IF(JX2.GT.IXMAX) IXMAX=JX2+1
             IF(JY2.GT.IYMAX) IYMAX=JY2+1
             IF(JZ2.GT.IZMAX) IZMAX=JZ2+1
             IF(JX1.LT.IXMIN) IXMIN=JX1-1
             IF(JY1.LT.IYMIN) IYMIN=JY1-1
             IF(JZ1.LT.IZMIN) IZMIN=JZ1-1
             ! Haibo GSBP

          ENDIF

          ! solve PBEQ in the original box
          ! XIAO_PHK_QC_UW0609: consistent with PB reference 
          !        CALL PBEQ1(MAXITS,DOME,DEPS,ONE,EPSP,ONE,ZERO, &
          CALL PBEQ1(MAXITS,DOME,DEPS,ONE,ONE,ONE,ZERO, &
               NCLX,NCLY,NCLZ,IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
               IPHIP,MAYX,MAYY,MAYZ, &
               ICHC,MAY,TMEMB, &
               .false.,QFOCUS,QPBC,QNPBC,.false.)
          IF(TIMER.GT.1) CALL WRTTIM('PBEQ solver in vacuum:')

       ENDIF

       !------------------------------------------------------------------------
       !     In solution (large box - focussing - original box)
       !------------------------------------------------------------------------

       IF(QFOCUS) THEN
          ! get grid parameters in the large box
          CALL MAYER(NTPRP,LSTPRP,EPSW,EPSP,KAPPA2,X,Y,Z,CG, &
               PBRAD,LNCEL,LNCEL,LNCEL,LDCEL,WATR,IONR,MAY, &
               MAYX,MAYY,MAYZ,SWIN,ILCHC, &
               IPHIW,VMEMB,TMEMB,ZMEMB,EPSM,NIMGB, &
               HTMEMB,EPSH,HDENS,POSALP,NEGALP,QCHRG,POFFST,NOFFST, &
               DROPLET,EPSD,XDROPLET,YDROPLET,ZDROPLET,Qdtom,Qdkap, &
               EPSB,LXMAX,LYMAX,LZMAX,LXMIN,LYMIN,LZMIN,Qbtom,Qbkap, &
               EPSC,Xcyln,Ycyln,Zcyln,Rcyln,Hcyln,Qctom,Qckap, &
               EPSEC,Xcone,Ycone,Zcone,Ax,By,Cz,Qectom,Qeckap, &
               -1,-1,&
               LTRAN,LTRAN,LTRAN,LXBCEN,LYBCEN,LZBCEN, &
               IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
               QKEEP,QREEN,QSMTH,QBSPL,QA,QB, &
               RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
               RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
               MAJOR,MINOR,QPROLATE, &
               .false.,.true.,.false.,IPHIB,QPBC,QNPBC, &
                                !              QINTBP QZERO
               QNONLINEAR,QPARTLINEAR,QFKAP, &
               IPOX,IPOY,IPOZ,MAPT,NATOM)

          ! get boundary potentials in the large box: CG -> ICHC
          IF(.NOT.QINTBP.AND..NOT.QPBC.AND..NOT.QZERO)THEN
             CALL MAYER_BP_ALL(LNCEL3,LSTPRP,X,Y,Z,CG,EPSW,KAPPA2, &
                  TMEMB,ILCHC,IPHIW,LNCEL,LNCEL,LNCEL,LDCEL, &
                  LTRAN,LTRAN,LTRAN,LXBCEN,LYBCEN,LZBCEN, &
                  RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
                  RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
                  MAJOR,MINOR,QPROLATE, &
                  LNCEL,LNCEL,LNCEL,LDCEL,LXBCEN,LYBCEN,LZBCEN, &
                  NIMGB,QNPBC,QA,QB)
          ELSEIF(QINTBP) THEN
             CALL MAYER_BP_INT(LNCEL3,LSTPRP,X,Y,Z,CG,EPSW,KAPPA2, &
                  TMEMB,ILCHC,IPHIW,LNCEL,LNCEL,LNCEL,LDCEL, &
                  LTRAN,LTRAN,LTRAN,LXBCEN,LYBCEN,LZBCEN, &
                  RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
                  RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
                  MAJOR,MINOR,QPROLATE, &
                  LNCEL,LNCEL,LNCEL,LDCEL,LXBCEN,LYBCEN,LZBCEN, &
                  NIMGB,QNPBC,QA,QB)
          ENDIF
          IF(TIMER.GT.1) &
               CALL WRTTIM('Boundary potential in large box in solvent:')

          ! vacuum environment in region B.
          CALL SPHE_MAYER(LNCEL,LNCEL,LNCEL,LTRAN,LTRAN,LTRAN,LDCEL, &
               LXBCEN,LYBCEN,LZBCEN, &
               MAY,MAYX,MAYY,MAYZ, &
               RRXCEN,RRYCEN,RRZCEN,SRDIST)

          ! redefine the maximum region outside which EPS is EPSW. (see subroutine MAYER)
          IXMAX=LNCEL-1
          IYMAX=LNCEL-1
          IZMAX=LNCEL-1
          IXMIN=2
          IYMIN=2
          IZMIN=2

          ! solve PBEQ in the large box
          CALL PBEQ1(MAXITS,DOME,DEPS,EPSW,EPSP,EPSM,KAPPA2, &
               LNCEL,LNCEL,LNCEL,IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
               IPHIW,MAYX,MAYY,MAYZ, &
               ILCHC,MAY,TMEMB, &
               QOSOR,.false.,QPBC,QNPBC,.false.)
          IF(TIMER.GT.1) &
               CALL WRTTIM('PBEQ solver in large box in solvent:')

          ! construct boundary potentials from the large box
          CALL BOUNDPOT(LNCEL,LNCEL,LNCEL,LDCEL, &
               LTRAN,LTRAN,LTRAN,LXBCEN,LYBCEN,LZBCEN, &
               NCLX,NCLY,NCLZ,DCEL,XBCEN,YBCEN,ZBCEN, &
               IPHIW,IPHIB)

          ! get the grid parameters in the original box
          CALL MAYER(NTPRP,LSTPRP,EPSW,EPSP,KAPPA2,X,Y,Z,CG, &
               PBRAD,NCLX,NCLY,NCLZ,DCEL,WATR,IONR,MAY, &
               MAYX,MAYY,MAYZ,SWIN,ICHC, &
               IPHIW,VMEMB,TMEMB,ZMEMB,EPSM,NIMGB, &
               HTMEMB,EPSH,HDENS,POSALP,NEGALP,QCHRG,POFFST,NOFFST, &
               DROPLET,EPSD,XDROPLET,YDROPLET,ZDROPLET,Qdtom,Qdkap, &
               EPSB,LXMAX,LYMAX,LZMAX,LXMIN,LYMIN,LZMIN,Qbtom,Qbkap, &
               EPSC,Xcyln,Ycyln,Zcyln,Rcyln,Hcyln,Qctom,Qckap, &
               EPSEC,Xcone,Ycone,Zcone,Ax,By,Cz,Qectom,Qeckap, &
               -1,-1,&
               TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
               IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
               QKEEP,QREEN,QSMTH,QBSPL,QA,QB, &
               RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
               RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
               MAJOR,MINOR,QPROLATE, &
               .false.,.false.,.true.,IPHIB,QPBC,QNPBC, &
                                !              QINTBP QZERO
               QNONLINEAR,QPARTLINEAR,QFKAP, &
               IPOX,IPOY,IPOZ,MAPT, NATOM)

          ! vacuum environment in region B.
          CALL SPHE_MAYER(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,DCEL, &
               XBCEN,YBCEN,ZBCEN, &
               MAY,MAYX,MAYY,MAYZ, &
               RRXCEN,RRYCEN,RRZCEN,SRDIST)

          ! redefine the maximum region outside which EPS is EPSW. (see subroutine MAYER)
          IF(JX2.GT.IXMAX) IXMAX=JX2+1
          IF(JY2.GT.IYMAX) IYMAX=JY2+1
          IF(JZ2.GT.IZMAX) IZMAX=JZ2+1
          IF(JX1.LT.IXMIN) IXMIN=JX1-1
          IF(JY1.LT.IYMIN) IYMIN=JY1-1
          IF(JZ1.LT.IZMIN) IZMIN=JZ1-1

          ! solve PBEQ in the original box
          CALL PBEQ1(MAXITS,DOME,DEPS,EPSW,EPSP,EPSM,KAPPA2, &
               NCLX,NCLY,NCLZ,IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
               IPHIW,MAYX,MAYY,MAYZ, &
               ICHC,MAY,TMEMB, &
               QOSOR,QFOCUS,QPBC,QNPBC,.false.)
          IF(TIMER.GT.1) CALL WRTTIM('PBEQ solver in solvent:')

       ELSE

          ! get the grid parameters in the original box
          CALL MAYER(NTPRP,LSTPRP,EPSW,EPSP,KAPPA2,X,Y,Z,CG, &
               PBRAD,NCLX,NCLY,NCLZ,DCEL,WATR,IONR,MAY, &
               MAYX,MAYY,MAYZ,SWIN,ICHC, &
               IPHIW,VMEMB,TMEMB,ZMEMB,EPSM,NIMGB, &
               HTMEMB,EPSH,HDENS,POSALP,NEGALP,QCHRG,POFFST,NOFFST, &
               DROPLET,EPSD,XDROPLET,YDROPLET,ZDROPLET,Qdtom,Qdkap, &
               EPSB,LXMAX,LYMAX,LZMAX,LXMIN,LYMIN,LZMIN,Qbtom,Qbkap, &
               EPSC,Xcyln,Ycyln,Zcyln,Rcyln,Hcyln,Qctom,Qckap, &
               EPSEC,Xcone,Ycone,Zcone,Ax,By,Cz,Qectom,Qeckap, &
               -1,-1,&
               TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
               IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
               QKEEP,QREEN,QSMTH,QBSPL,QA,QB, &
               RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
               RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
               MAJOR,MINOR,QPROLATE, &
               .false.,.true.,.false.,IPHIB,QPBC,QNPBC, &
                                !              QINTBP QZERO
               QNONLINEAR,QPARTLINEAR,QFKAP, &
               IPOX,IPOY,IPOZ,MAPT,NATOM)

          IF(.NOT.QINTBP.AND..NOT.QPBC.AND..NOT.QZERO)THEN
             CALL MAYER_BP_ALL(LNCEL3,LSTPRP,X,Y,Z,CG,EPSW,KAPPA2, &
                  TMEMB,ILCHC,IPHIW,NCLX,NCLY,NCLZ,DCEL, &
                  TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
                  RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
                  RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
                  MAJOR,MINOR,QPROLATE, &
                  LNCEL,LNCEL,LNCEL,LDCEL,LXBCEN,LYBCEN,LZBCEN, &
                  NIMGB,QNPBC,QA,QB)
          ELSEIF(QINTBP) THEN
             CALL MAYER_BP_INT(LNCEL3,LSTPRP,X,Y,Z,CG,EPSW,KAPPA2, &
                  TMEMB,ILCHC,IPHIW,NCLX,NCLY,NCLZ,DCEL, &
                  TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
                  RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
                  RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
                  MAJOR,MINOR,QPROLATE, &
                  LNCEL,LNCEL,LNCEL,LDCEL,LXBCEN,LYBCEN,LZBCEN, &
                  NIMGB,QNPBC,QA,QB)
          ENDIF
          IF(TIMER.GT.1) &
               CALL WRTTIM('Boundary potential with large box in solvent:')

          ! vacuum environment in region B.
          CALL SPHE_MAYER(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,DCEL, &
               XBCEN,YBCEN,ZBCEN, &
               MAY,MAYX,MAYY,MAYZ, &
               RRXCEN,RRYCEN,RRZCEN,SRDIST)

          ! redefine the maximum region outside which EPS is EPSW. (see subroutine MAYER)
          IF(JX2.GT.IXMAX) IXMAX=JX2+1
          IF(JY2.GT.IYMAX) IYMAX=JY2+1
          IF(JZ2.GT.IZMAX) IZMAX=JZ2+1
          IF(JX1.LT.IXMIN) IXMIN=JX1-1
          IF(JY1.LT.IYMIN) IYMIN=JY1-1
          IF(JZ1.LT.IZMIN) IZMIN=JZ1-1

          ! solve PBEQ in the original box
          CALL PBEQ1(MAXITS,DOME,DEPS,EPSW,EPSP,EPSM,KAPPA2, &
               NCLX,NCLY,NCLZ,IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
               IPHIW,MAYX,MAYY,MAYZ, &
               ICHC,MAY,TMEMB, &
               QOSOR,QFOCUS,QPBC,QNPBC,.false.)
          IF(TIMER.GT.1) CALL WRTTIM('PBEQ solver in solvent:')

       ENDIF

       ! calculate the reaction field energy elements M(ij)
       CALL SPHE_MIJ(NPGD,DCEL,NTPOL,NORDER, &
            LSTPL,LSTPM,BNORM, &
            IIP0,CTHETA,CPHI,SPHI,RGD, &
            IPHIP,IPHIW,MIJ,CGSCAL)

    ENDDO
    call chmdealloc('pbeq.src','SPHE_GSBP1','IIP0',NPROI,intg=IIP0)
    call chmdealloc('pbeq.src','SPHE_GSBP1','RGD',NPROI,crl=RGD)
    call chmdealloc('pbeq.src','SPHE_GSBP1','CTHETA',NPROI,crl=CTHETA)
    call chmdealloc('pbeq.src','SPHE_GSBP1','CPHI',NPROI,crl=CPHI)
    call chmdealloc('pbeq.src','SPHE_GSBP1','SPHI',NPROI,crl=SPHI)
    call chmdealloc('pbeq.src','SPHE_GSBP1','ILCHC',LNCEL3,cr4=ILCHC)
    call chmdealloc('pbeq.src','SPHE_GSBP1','IPHIB',BNCEL,cr4=IPHIB)
    QA=.false.
    QB=.false.
    !
    RETURN
  END SUBROUTINE SPHE_GSBP1
  !
  
  SUBROUTINE SPHE_GSBP2(NATOM,X,Y,Z,CG,QPRIN)
    !-----------------------------------------------------------------------
    !     The reaction field energy and forces of the region of interest
    !     are calculated using the following basis functions :
    !     basis funcition (l,m) = radial function (l) * spherical harmonics (l,m)
    !
    use chm_kinds
    use memory
    use number
    use stream
    use dimens_fcm
    use consta
    use timerm
    use machutil,only:wrttim
    implicit none
    !
    INTEGER NATOM
    real(chm_real)  X(*),Y(*),Z(*),CG(*)
    LOGICAL QPRIN
    ! local
    integer,allocatable,dimension(:) :: IIP0
    real(chm_real),allocatable,dimension(:) :: RGD,CTHETA,CPHI,SPHI
    real(chm_real4),dimension(1) :: IPHIB !dummy
    INTEGER NORDER,OLDNTP
    INTEGER NFIL,IXCEN,IYCEN,IZCEN,JX1,JX2,JY1,JY2,JZ1,JZ2
    INTEGER NPROI,NPGD
    !
    ! save basis functions
    OLDNTP= OLDNMP*OLDNMP
    CALL SPHEPOL(NMPOL,LSTPL,LSTPM,LSTPOL)

    ! calculate the normalization constants of the basis functions
    CALL SPHE_NORM(NMPOL,BNORM,SRDIST)

    IF(QMIJ.AND.OLDNTP.EQ.NTPOL) RETURN

    ! calculate the trigonometric functions of each grid points
    ! inside region of interest
    NFIL=INT(SRDIST/DCEL)+2
    IXCEN=INT((RRXCEN+TRANX-XBCEN)/DCEL)+1
    IYCEN=INT((RRYCEN+TRANY-YBCEN)/DCEL)+1
    IZCEN=INT((RRZCEN+TRANZ-ZBCEN)/DCEL)+1
    JX1=IXCEN-NFIL
    JX2=IXCEN+NFIL+1
    JY1=IYCEN-NFIL
    JY2=IYCEN+NFIL+1
    JZ1=IZCEN-NFIL
    JZ2=IZCEN+NFIL+1
    NPROI =(JX2-JX1)*(JY2-JY1)*(JZ2-JZ1)
    call chmalloc('pbeq.src','SPHE_GSBP2','IIP0',NPROI,intg=IIP0)
    call chmalloc('pbeq.src','SPHE_GSBP2','RGD',NPROI,crl=RGD)
    call chmalloc('pbeq.src','SPHE_GSBP2','CTHETA',NPROI,crl=CTHETA)
    call chmalloc('pbeq.src','SPHE_GSBP2','CPHI',NPROI,crl=CPHI)
    call chmalloc('pbeq.src','SPHE_GSBP2','SPHI',NPROI,crl=SPHI)
    CALL SPHE_TRIG(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,DCEL, &
         XBCEN,YBCEN,ZBCEN, &
         IIP0,CTHETA,CPHI,SPHI,RGD, &
         SRDIST,RRXCEN,RRYCEN,RRZCEN,JX1,JX2,JY1,JY2,JZ1,JZ2,NPGD)

    ! skip the initialization of CHC array and
    !      the all charge distribution calculations in MAYER
    QA=.true.
    QB=.true.

    DO NORDER=OLDNTP+1,NTPOL

       IF(PRNLEV.GT.7.AND.NORDER.EQ.1) WRITE(OUTU,'(A)')
       IF(PRNLEV.GT.7) WRITE(OUTU,'(3x,a,i5,a)') &
            'Reaction field calculation for',NORDER,' basis function'

       ! replace ICHC (CDEN) array with the basis functions (spherical harmonics)
       CALL SPHE_CHC(NCEL3,NPGD,DCEL,NORDER, &
            IIP0,CTHETA,CPHI,SPHI,RGD, &
            LSTPL,LSTPM,BNORM,ICHC,CGSCAL)

       !------------------------------------------------------------------------
       !     In vacuum
       !------------------------------------------------------------------------

       ! get the grid parameters
       CALL MAYER(NTPRP,LSTPRP,ONE,ONE,ZERO,X,Y,Z,CG, &
            PBRAD,NCLX,NCLY,NCLZ,DCEL,WATR,IONR,MAY, &
            MAYX,MAYY,MAYZ,SWIN,ICHC, &
            IPHIP,ZERO,TMEMB,ZMEMB,ONE,NIMGB, &
            HTMEMB,ONE,HDENS,POSALP,NEGALP,QCHRG,POFFST,NOFFST, &
            DROPLET,ONE,XDROPLET,YDROPLET,ZDROPLET,Qdtom,Qdkap, &
            ONE,LXMAX,LYMAX,LZMAX,LXMIN,LYMIN,LZMIN,Qbtom,Qbkap, &
            ONE,Xcyln,Ycyln,Zcyln,Rcyln,Hcyln,Qctom,Qckap, &
            ONE,Xcone,Ycone,Zcone,Ax,By,Cz,Qectom,Qeckap, &
               -1,-1,&
            TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
            IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
            QKEEP,QREEN,QSMTH,QBSPL,QA,QB, &
            RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
            RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
            MAJOR,MINOR,QPROLATE, &
            .false.,.true.,.false.,IPHIB,QPBC,QNPBC, &
                                !              QINTBP QZERO
            QNONLINEAR,QPARTLINEAR,QFKAP, &
            IPOX,IPOY,IPOZ,MAPT, NATOM)

       ! get boundary potentials : CG -> ICHC
       IF(.NOT.QINTBP.AND..NOT.QPBC.AND..NOT.QZERO)THEN
          CALL MAYER_BP_ALL(NCEL3,LSTPRP,X,Y,Z,CG,ONE,ZERO, &
               TMEMB,ICHC,IPHIP,NCLX,NCLY,NCLZ,DCEL, &
               TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
               RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
               RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
               MAJOR,MINOR,QPROLATE, &
               NCLX,NCLY,NCLZ,DCEL,XBCEN,YBCEN,ZBCEN, &
               NIMGB,QNPBC,QA,QB)
       ELSEIF(QINTBP) THEN
          CALL MAYER_BP_INT(NCEL3,LSTPRP,X,Y,Z,CG,ONE,ZERO, &
               TMEMB,ICHC,IPHIP,NCLX,NCLY,NCLZ,DCEL, &
               TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
               RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
               RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
               MAJOR,MINOR,QPROLATE, &
               NCLX,NCLY,NCLZ,DCEL,XBCEN,YBCEN,ZBCEN, &
               NIMGB,QNPBC,QA,QB)
       ENDIF
       IF(TIMER.GT.1) CALL WRTTIM('Boundary potential in vacuum:')

       IF(EPSP.ne.ONE) THEN
          ! vacuum environment in region B.
          CALL SPHE_MAYER(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,DCEL, &
               XBCEN,YBCEN,ZBCEN, &
               MAY,MAYX,MAYY,MAYZ, &
               RRXCEN,RRYCEN,RRZCEN,SRDIST)
          ! Haibo GSBP (XIAO_QC_UW0609 for boundary)
          ! redefine the maximum region outside which EPS is EPSW.
          ! (see subroutine MAYER)
          IF(JX2.GT.IXMAX) IXMAX=JX2+1
          IF(JY2.GT.IYMAX) IYMAX=JY2+1
          IF(JZ2.GT.IZMAX) IZMAX=JZ2+1
          IF(JX1.LT.IXMIN) IXMIN=JX1-1
          IF(JY1.LT.IYMIN) IYMIN=JY1-1
          IF(JZ1.LT.IZMIN) IZMIN=JZ1-1
          ! Haibo GSBP

       ENDIF

       ! solve PBEQ
       ! XIAO_PHK_QC_UW0609: consistent with PB reference 
       !        CALL PBEQ1(MAXITS,DOME,DEPS,ONE,EPSP,ONE,ZERO,
       CALL PBEQ1(MAXITS,DOME,DEPS,ONE,ONE,ONE,ZERO, &
            NCLX,NCLY,NCLZ,IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
            IPHIP,MAYX,MAYY,MAYZ, &
            ICHC,MAY,TMEMB, &
            .false.,.false.,QPBC,QNPBC,.false.)
       IF(TIMER.GT.1) CALL WRTTIM('PBEQ solver in vacuum:')

       !------------------------------------------------------------------------
       !     In solution
       !------------------------------------------------------------------------

       ! get grid parameters
       CALL MAYER(NTPRP,LSTPRP,EPSW,EPSP,KAPPA2,X,Y,Z,CG, &
            PBRAD,NCLX,NCLY,NCLZ,DCEL,WATR,IONR,MAY, &
            MAYX,MAYY,MAYZ,SWIN,ICHC, &
            IPHIW,VMEMB,TMEMB,ZMEMB,EPSM,NIMGB, &
            HTMEMB,EPSH,HDENS,POSALP,NEGALP,QCHRG,POFFST,NOFFST, &
            DROPLET,EPSD,XDROPLET,YDROPLET,ZDROPLET,Qdtom,Qdkap, &
            EPSB,LXMAX,LYMAX,LZMAX,LXMIN,LYMIN,LZMIN,Qbtom,Qbkap, &
            EPSC,Xcyln,Ycyln,Zcyln,Rcyln,Hcyln,Qctom,Qckap, &
            EPSEC,Xcone,Ycone,Zcone,Ax,By,Cz,Qectom,Qeckap, &
               -1,-1,&
            TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
            IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
            QKEEP,QREEN,QSMTH,QBSPL,QA,QB, &
            RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
            RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
            MAJOR,MINOR,QPROLATE, &
            .false.,.true.,.false.,IPHIB,QPBC,QNPBC, &
                                !              QINTBP QZERO
            QNONLINEAR,QPARTLINEAR,QFKAP, &
            IPOX,IPOY,IPOZ,MAPT,NATOM)

       ! get boundary potentials : CG -> ICHC
       IF(.NOT.QINTBP.AND..NOT.QPBC.AND..NOT.QZERO)THEN
          CALL MAYER_BP_ALL(NCEL3,LSTPRP,X,Y,Z,CG,EPSW,KAPPA2, &
               TMEMB,ICHC,IPHIW,NCLX,NCLY,NCLZ,DCEL, &
               TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
               RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
               RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
               MAJOR,MINOR,QPROLATE, &
               NCLX,NCLY,NCLZ,DCEL,XBCEN,YBCEN,ZBCEN, &
               NIMGB,QNPBC,QA,QB)
       ELSEIF(QINTBP) THEN
          CALL MAYER_BP_INT(NCEL3,LSTPRP,X,Y,Z,CG,EPSW,KAPPA2, &
               TMEMB,ICHC,IPHIW,NCLX,NCLY,NCLZ,DCEL, &
               TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
               RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
               RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
               MAJOR,MINOR,QPROLATE, &
               NCLX,NCLY,NCLZ,DCEL,XBCEN,YBCEN,ZBCEN, &
               NIMGB,QNPBC,QA,QB)
       ENDIF
       IF(TIMER.GT.1) CALL WRTTIM('Boundary potential in solvent:')

       ! vacuum environment in region B.
       CALL SPHE_MAYER(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,DCEL, &
            XBCEN,YBCEN,ZBCEN, &
            MAY,MAYX,MAYY,MAYZ, &
            RRXCEN,RRYCEN,RRZCEN,SRDIST)

       ! redefine the maximum region outside which EPS is EPSW. (see subroutine MAYER)
       IF(JX2.GT.IXMAX) IXMAX=JX2+1
       IF(JY2.GT.IYMAX) IYMAX=JY2+1
       IF(JZ2.GT.IZMAX) IZMAX=JZ2+1
       IF(JX1.LT.IXMIN) IXMIN=JX1-1
       IF(JY1.LT.IYMIN) IYMIN=JY1-1
       IF(JZ1.LT.IZMIN) IZMIN=JZ1-1

       ! solve PBEQ
       CALL PBEQ1(MAXITS,DOME,DEPS,EPSW,EPSP,EPSM,KAPPA2, &
            NCLX,NCLY,NCLZ,IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
            IPHIW,MAYX,MAYY,MAYZ, &
            ICHC,MAY,TMEMB, &
            QOSOR,.false.,QPBC,QNPBC,.false.)
       IF(TIMER.GT.1) CALL WRTTIM('PBEQ solver in solvent:')

       ! calculate the reaction field energy elements M(ij)
       CALL SPHE_MIJ(NPGD,DCEL,NTPOL,NORDER, &
            LSTPL,LSTPM,BNORM, &
            IIP0,CTHETA,CPHI,SPHI,RGD, &
            IPHIP,IPHIW,MIJ,CGSCAL)

    ENDDO
    call chmdealloc('pbeq.src','SPHE_GSBP2','IIP0',NPROI,intg=IIP0)
    call chmdealloc('pbeq.src','SPHE_GSBP2','RGD',NPROI,crl=RGD)
    call chmdealloc('pbeq.src','SPHE_GSBP2','CTHETA',NPROI,crl=CTHETA)
    call chmdealloc('pbeq.src','SPHE_GSBP2','CPHI',NPROI,crl=CPHI)
    call chmdealloc('pbeq.src','SPHE_GSBP2','SPHI',NPROI,crl=SPHI) 
    QA=.false.
    QB=.false.
    !
    RETURN
  END SUBROUTINE SPHE_GSBP2
  !
  SUBROUTINE RECT_STPOL(NTRB,LSTRB,X,Y,Z,CGE,NTPOL, &
       !                                          CG->CGE, to avoid conflict
       !                                          with psf
       XNPOL,YNPOL,ZNPOL,RRXCEN,RRYCEN,RRZCEN, &
       XSCALE,YSCALE,ZSCALE,MIJ,COEF,EMN,LSTPX,LSTPY,LSTPZ, &
       ALPX,ALPY,ALPZ, &
       BNORM,LSTPOL,ICALL,NLIST,QNOSORT &
#if KEY_GCMC==1
       ,GCMCON  & 
#endif
#if KEY_SCCDFTB==1
       ,COEFX   & 
#endif
       )
    !-----------------------------------------------------------------------
    !     calculate the coefficients of the basis functions and
    !     make a list of the basis functions according to their contribution
    !     to the reaction field energy
    !
    use chm_kinds
    use stream
    use number
    use consta
#if KEY_SCCDFTB==1
    use sccdftb             !qmuli2 array
    use dimens_fcm              !need IGMSEL
    use gamess_fcm              !need IGMSEL
    use chutil,only: getres
    !  XIAO_PHK_QC_UW0609
    use psf
#endif 
    implicit none
#if KEY_SCCDFTB==1
    real(chm_real)  COEFX(*)
#endif 
    real(chm_real)  X(*),Y(*),Z(*),CGE(*),COEF(*),MIJ(*),EMN(*), &
         BNORM(*)
    real(chm_real)  ALPX(*),ALPY(*),ALPZ(*)
    real(chm_real)  RRXCEN,RRYCEN,RRZCEN
    real(chm_real)  XSCALE,YSCALE,ZSCALE
    INTEGER XNPOL,YNPOL,ZNPOL,ICALL,NLIST,NTPOL
    INTEGER NTRB,LSTRB(*)
    INTEGER LSTPX(*),LSTPY(*),LSTPZ(*),LSTPOL(*)
    LOGICAL QNOSORT
#if KEY_GCMC==1
    LOGICAL GCMCON(:) 
#endif
    ! local
    real(chm_real)  TMPEMN,TMPENN,EMIN,TMPCOEF,CCC
    real(chm_real)  XG,YG,ZG,NORM
    INTEGER I,J,IJ,II,JJ,NEWIJ,NORDER
    INTEGER XPOL,YPOL,ZPOL,POLMIN
    INTEGER TMPIPX,TMPIPY,TMPIPZ,TMPPOL
    ! QC_PS_UW_08
    INTEGER NSCCTMP
    ! XIAO_PHK_QC_UW0609
    ! PHK: DIV
    INTEGER K1, K2, IS,IQ
    REAL(chm_real) NQ,DQ,NE


    ! calculate the coefficients of the basis functions
#if KEY_SCCDFTB==1
    if (qmused_sccdftb) coefx(1:ntpol) = zero
#endif

    DO II=1,NTPOL
       COEF(II)=ZERO
    ENDDO
    !     QC_PS_UW04:
    NSCCTMP=0

    i_loop: DO I=1,NTRB
       J=LSTRB(I)
#if KEY_GCMC==1
       IF (.NOT. GCMCON(J)) cycle i_loop 
#endif
       CCC=CGE(J)
#if KEY_SCCDFTB==1
       if(qmused_sccdftb)then
         IF ((IGMSEL(J) .EQ. 1).or.(IGMSEL(J) .EQ. 2)) THEN
            NSCCTMP=NSCCTMP + 1
            CCC=QMULI2(NSCCTMP,1)
         ENDIF
       endif
#endif 

       IF(CCC.EQ.ZERO) cycle i_loop
       XG=X(J)-RRXCEN
       YG=Y(J)-RRYCEN
       ZG=Z(J)-RRZCEN

       CALL LPOL2(XNPOL,XG,XSCALE,ALPX)
       CALL LPOL2(YNPOL,YG,YSCALE,ALPY)
       CALL LPOL2(ZNPOL,ZG,ZSCALE,ALPZ)

       DO II=1,NTPOL
          XPOL=LSTPX(II)
          YPOL=LSTPY(II)
          ZPOL=LSTPZ(II)
          NORM=BNORM(II)
          COEF(II)=COEF(II)+CCC*NORM* &
               ALPX(XPOL+1)*ALPY(YPOL+1)*ALPZ(ZPOL+1)
       ENDDO

#if KEY_SCCDFTB==1
       !  PS: for sccgsbp need to calculated COEFX corresponding to
       !      excluded atoms
       ! this is only useful when SCCDFTB is specified in the run
       if(qmused_sccdftb)then
       IF(IGMSEL(J) .EQ. 5) THEN
          DO II=1,NTPOL
             XPOL=LSTPX(II)
             YPOL=LSTPY(II)
             ZPOL=LSTPZ(II)
             NORM=BNORM(II)
             COEFX(II)=COEFX(II)+CCC*NORM* &
                  ALPX(XPOL+1)*ALPY(YPOL+1)*ALPZ(ZPOL+1)
          ENDDO
       ENDIF
       ! PHK: DIV
       IF (QSCCDIV) THEN
          IF(IGMSEL(J).EQ.0) THEN
             ! loop over all atoms in group
             ! and find a div atom
             K1=GETRES(J,IGPBS,NGRP)
             IS=IGPBS(K1)+1
             IQ=IGPBS(K1+1)
             DQ=0
             NE=0
             DO K2=IS,IQ
                IF (IGMSEL(K2).EQ.5) THEN
                   DQ=DQ+CG(K2)
                   NE=NE+1
                ENDIF
             ENDDO
             IF (NE.gt.0) THEN
                DQ=-DQ/(IQ-IS+1-NE)
                DO II=1,NTPOL
                   XPOL=LSTPX(II)
                   YPOL=LSTPY(II)
                   ZPOL=LSTPZ(II)
                   NORM=BNORM(II)
                   COEFX(II)=COEFX(II)+DQ*NORM* &
                        ALPX(XPOL+1)*ALPY(YPOL+1)*ALPZ(ZPOL+1)               
                ENDDO
             ENDIF
          ENDIF
       ENDIF
       endif
#endif 

    END DO i_loop

    IF(MOD(ICALL,NLIST).NE.0.OR.QNOSORT) GOTO 100
    ! calculate the energy contribution of each basis function and
    ! find the smallest one and remove the function
    ! repeat it again
    DO II=1,NTPOL
       LSTPOL(II)=II
    ENDDO
    DO NORDER=NTPOL,1,-1
       DO I=1,NORDER
          II=LSTPOL(I)
          IJ=(II-1)*NTPOL+II
          EMN(I)=HALF*COEF(II)*COEF(II)*MIJ(IJ)*CCELEC
          DO J=1,I-1
             JJ=LSTPOL(J)
             IJ=(JJ-1)*NTPOL+II
             EMN(I)=EMN(I)+COEF(II)*COEF(JJ)*MIJ(IJ)*CCELEC
          ENDDO
          DO J=I+1,NORDER
             JJ=LSTPOL(J)
             IJ=(II-1)*NTPOL+JJ
             EMN(I)=EMN(I)+COEF(II)*COEF(JJ)*MIJ(IJ)*CCELEC
          ENDDO
       ENDDO
       EMIN=RBIG
       DO I=1,NORDER
          IF(abs(EMN(I)).lt.EMIN) THEN
             EMIN=abs(EMN(I))
             POLMIN=I
          ENDIF
       ENDDO
       TMPPOL=LSTPOL(POLMIN)
       DO I=POLMIN+1,NORDER
          LSTPOL(I-1)=LSTPOL(I)
       ENDDO
       LSTPOL(NORDER)=TMPPOL
    ENDDO
    !
100 CONTINUE
    !     QC_UW_06 
    !     IF(ICALL.GT.0.OR.PRNLEV.LT.6) RETURN
    IF(ICALL.GT.0.OR.PRNLEV.LT.6) GOTO 110
    !
    WRITE(OUTU,'(/,3X,A)') &
         'Coefficients of Basis Functions and Running Summations in G_II'
    WRITE(OUTU,'(3X,A)') &
         ' NORDER XPOL YPOL ZPOL      COEFF.       G_II(TOT)       G_II(N)'
    !
    DO I=1,NTPOL
       II=LSTPOL(I)
       IJ=(II-1)*NTPOL+II
       EMN(I)=HALF*COEF(II)*COEF(II)*MIJ(IJ)*CCELEC
       DO J=1,I-1
          JJ=LSTPOL(J)
          IJ=(II-1)*NTPOL+JJ
          IF(II.GT.JJ) IJ=(JJ-1)*NTPOL+II
          EMN(I)=EMN(I)+COEF(II)*COEF(JJ)*MIJ(IJ)*CCELEC
       ENDDO
    ENDDO

    TMPEMN=ZERO
    DO NORDER=1,NTPOL
       TMPEMN=TMPEMN+EMN(NORDER)
       II=LSTPOL(NORDER)
       XPOL=LSTPX(II)
       YPOL=LSTPY(II)
       ZPOL=LSTPZ(II)
       WRITE(OUTU,'(3x,4i5,2x,3E15.7)') &
            NORDER,XPOL,YPOL,ZPOL,COEF(II),TMPEMN,EMN(NORDER)
    ENDDO

110 CONTINUE

    !  QC_PS_UW08: now remove the QM atom contributions to COEF before
    !  printing and returning to main program
    !
#if KEY_SCCDFTB==1
    NSCCTMP=0
    ! Need to really run SCCDFTB or else there is no igmsel :-(
    if(qmused_sccdftb)then
    i_loop2: DO I=1,NTRB
       J=LSTRB(I)
       IF ((IGMSEL(J) .NE. 1).and.(IGMSEL(J) .NE. 2)) cycle i_loop2
       NSCCTMP=NSCCTMP + 1
       CCC=QMULI2(NSCCTMP,1)
       IF(CCC.EQ.ZERO) cycle i_loop2
       XG=X(J)-RRXCEN
       YG=Y(J)-RRYCEN
       ZG=Z(J)-RRZCEN

       CALL LPOL2(XNPOL,XG,XSCALE,ALPX)
       CALL LPOL2(YNPOL,YG,YSCALE,ALPY)
       CALL LPOL2(ZNPOL,ZG,ZSCALE,ALPZ)

       DO II=1,NTPOL
          XPOL=LSTPX(II)
          YPOL=LSTPY(II)
          ZPOL=LSTPZ(II)
          NORM=BNORM(II)
          COEF(II)=COEF(II)-CCC*NORM* &
               ALPX(XPOL+1)*ALPY(YPOL+1)*ALPZ(ZPOL+1)
       ENDDO
       !
    ENDDO i_loop2
    endif
#endif 
    RETURN
  END SUBROUTINE RECT_STPOL
  !
  SUBROUTINE SPHE_STPOL(NTRB,LSTRB,X,Y,Z,CGE,NTPOL,SRDIST, &
       RRXCEN,RRYCEN,RRZCEN,MIJ,COEF,EMN, &
       AR,AC,AS,AP,NMPOL, &
       LSTPL,LSTPM,BNORM,LSTPOL,ICALL,NLIST,QNOSORT &
#if KEY_SCCDFTB==1
       ,COEFX  & 
#endif
#if KEY_GCMC==1
       ,GCMCON  & 
#endif
       )

    !-----------------------------------------------------------------------
    !     calculate the coefficients of the basis functions and
    !     make a list of the basis functions according to their contribution
    !     to the reaction field energy
    !
    use chm_kinds
    use stream
    use number
    use consta
#if KEY_SCCDFTB==1
    use sccdftb             !qmuli2 array
    use dimens_fcm              !need IGMSEL
    use gamess_fcm              !need IGMSEL
    ! PHK: DIV
    use psf
    use chutil,only: getres
#endif 
    implicit none
#if KEY_SCCDFTB==1
    real(chm_real)  COEFX(*)
#endif 

    INTEGER NTRB,LSTRB(*),NTPOL,ICALL,NLIST,NMPOL
    INTEGER LSTPL(*),LSTPM(*),LSTPOL(*)
    real(chm_real)  X(*),Y(*),Z(*),CGE(*),COEF(*),MIJ(*),EMN(*), &
         BNORM(*)
    real(chm_real)  RRXCEN,RRYCEN,RRZCEN,SRDIST
    real(chm_real)  AR(0:NMPOL-1),AC(0:NMPOL-1),AS(0:NMPOL-1)
    real(chm_real)  AP(0:NMPOL-1,0:NMPOL-1)
    LOGICAL QNOSORT
#if KEY_GCMC==1
    LOGICAL GCMCON(:) 
#endif
    ! local
    real(chm_real)  NORM
    real(chm_real)  EMIN,TMPEMN
    real(chm_real)  CT,ST,CS,CP,SP,XDIFF,YDIFF,ZDIFF,R,R2,SRDIST2,CCC
    INTEGER II,JJ,L,N,M,I,J,IJ,KK
    INTEGER NORDER,POLMIN,TMPPOL
    INTEGER LMAX
    ! QC_PS_UW04:
    real(chm_real)  cff
    INTEGER NSCCTMP
    ! PHK: DIV
    INTEGER K1,K2,IS,IQ
    REAL(chm_real)  DQ,NQ,NE

    ! calculate the coefficients of the basis functions
    LMAX=NMPOL-1
    SRDIST2=(SRDIST+RSMALL)*(SRDIST+RSMALL)

#if KEY_SCCDFTB==1
    if (qmused_sccdftb) coefx(1:ntpol) = zero
#endif

    DO II=1,NTPOL
       COEF(II)=ZERO
    ENDDO
    !     QC_PS_UW04:
    !     Haibo Yu: need additional variable KK for the replica
    NSCCTMP=0
    KK=1

    i_loop3: DO I=1,NTRB
       J =LSTRB(I)
#if KEY_GCMC==1
       IF (.NOT. GCMCON(J)) cycle i_loop3 
#endif
       CCC=CGE(J)
#if KEY_SCCDFTB==1
       if(qmused_sccdftb)then
       IF ((IGMSEL(J) .EQ. 1).or.(IGMSEL(J) .EQ. 2)) THEN
          NSCCTMP=NSCCTMP + 1
          CCC=QMULI2(NSCCTMP,KK)
       ENDIF
       endif
       !     Haibo Yu deal with replica
       IF (NSCCTMP .EQ. NSCCTC) THEN
          NSCCTMP = 0
          KK = KK + 1
       ENDIF
#endif 

       IF(CCC.EQ.ZERO) cycle i_loop3
       XDIFF=X(J)-RRXCEN
       YDIFF=Y(J)-RRYCEN
       ZDIFF=Z(J)-RRZCEN
       R2=XDIFF*XDIFF+YDIFF*YDIFF+ZDIFF*ZDIFF
       ! The following line is commented out to avoid discontiuty
       ! when part of a molecule crosses the dielectric boundary. 
       ! The new cavity potential should be able to keep atoms from
       ! moving outside the simulation region. Y. Deng June/2007
       !         IF(R2.GT.SRDIST2) cycle loop99
       R=SQRT(R2)
       CT=ZDIFF/R
       ST=SQRT(ONE-(ZDIFF*ZDIFF)/R2)
       CP=XDIFF/R/ST
       SP=YDIFF/R/ST
       IF(R2.LT.RSMALL) THEN                               ! in the origin
          CT=ZERO
          CP=ZERO
          SP=ZERO
       ELSEIF(XDIFF.GT.-RSMALL.AND.XDIFF.LT.RSMALL.AND.     & ! in the Z-axis
            YDIFF.GT.-RSMALL.AND.YDIFF.LT.RSMALL) THEN
          CT=ONE
          IF(ZDIFF.LT.ZERO) CT=-ONE
          CP=ZERO
          SP=ZERO
       ELSEIF(ZDIFF.GT.-RSMALL.AND.ZDIFF.LT.RSMALL) THEN   ! in the XY plane
          CT=ZERO
          CP=XDIFF/R
          SP=YDIFF/R
       ENDIF

       CALL RPOWERL2(LMAX,R,AR)           !  fill AR  (r^l   ) array
       CALL COSMPHI2(LMAX,CP,AC)          !  fill AC  (cos.. ) array
       CALL SINMPHI2(LMAX,CP,SP,AS)       !  fill AS  (sin.. ) array
       CALL ALPOL2(LMAX,CT,AP)       !  fill AP  (P(lm) ) array

       DO II=1,NTPOL
          L=LSTPL(II)
          M=LSTPM(II)
          NORM=BNORM(II)
          IF(L.GE.0.AND.M.EQ.0) THEN
             COEF(II)=COEF(II)+CCC*NORM*AR(L)*AP(L,M)
          ELSEIF(L.GT.0.AND.M.GT.0) THEN
             COEF(II)=COEF(II)+CCC*NORM*AR(L)*AC(M)*AP(L,M)
          ELSEIF(L.GT.0.AND.M.LT.0) THEN
             M=-M
             COEF(II)=COEF(II)+CCC*NORM*AR(L)*AS(M)*AP(L,M)
          ENDIF
       ENDDO
#if KEY_SCCDFTB==1
       !  PS: for sccgsbp need to calculated COEFX corresponding to 
       !      excluded atoms
       if(qmused_sccdftb) then ! no igmsel() otherwise
       IF(IGMSEL(J) .EQ. 5) THEN
          DO II=1,NTPOL
             L=LSTPL(II)
             M=LSTPM(II)
             NORM=BNORM(II)
             IF(L.GE.0.AND.M.EQ.0) THEN
                COEFX(II)=COEFX(II)+CCC*NORM*AR(L)*AP(L,M)
             ELSEIF(L.GT.0.AND.M.GT.0) THEN
                COEFX(II)=COEFX(II)+CCC*NORM*AR(L)*AC(M)*AP(L,M)
             ELSEIF(L.GT.0.AND.M.LT.0) THEN
                M=-M
                COEFX(II)=COEFX(II)+CCC*NORM*AR(L)*AS(M)*AP(L,M)
             ENDIF
          ENDDO
       ENDIF
       ! PHK: DIV
       IF (QSCCDIV) THEN
          IF(IGMSEL(J).EQ.0) THEN
             ! loop over all atoms in group
             ! and find a div atom
             K1=GETRES(J,IGPBS,NGRP)
             IS=IGPBS(K1)+1
             IQ=IGPBS(K1+1)
             DQ=0.0d0
             NQ=0.0d0
             NE=0.0d0
             DO K2=IS,IQ
                IF (IGMSEL(K2).EQ.5) THEN
                   DQ=DQ+CG(K2)
                   NE=NE+1
                ENDIF
                IF (IGMSEL(K2).EQ.0) NQ=NQ+1
             ENDDO
             IF (NE.gt.0) THEN
                DQ=-DQ/NQ
                DO II=1,NTPOL
                   L=LSTPL(II)
                   M=LSTPM(II)
                   NORM=BNORM(II)
                   IF(L.GE.0.AND.M.EQ.0) THEN
                      COEFX(II)=COEFX(II)+ DQ*NORM*AR(L)*AP(L,M)
                   ELSEIF(L.GT.0.AND.M.GT.0) THEN
                      COEFX(II)=COEFX(II)+ DQ*NORM*AR(L)*AC(M)*AP(L,M)
                   ELSEIF(L.GT.0.AND.M.LT.0) THEN
                      M=-M
                      COEFX(II)=COEFX(II)+ DQ*NORM*AR(L)*AS(M)*AP(L,M)
                   ENDIF

                ENDDO
             ENDIF
          ENDIF
       ENDIF
       endif
#endif

    ENDDO i_loop3

    IF(MOD(ICALL,NLIST).NE.0.OR.QNOSORT) GOTO 100
    ! calculate the energy contribution of each basis function and
    ! find the smallest one and remove the function
    ! repeat it again
    DO II=1,NTPOL
       LSTPOL(II)=II
    ENDDO
    DO NORDER=NTPOL,1,-1
       DO I=1,NORDER
          II=LSTPOL(I)
          IJ=(II-1)*NTPOL+II
          EMN(I)=HALF*COEF(II)*COEF(II)*MIJ(IJ)*CCELEC
          DO J=1,I-1
             JJ=LSTPOL(J)
             IJ=(JJ-1)*NTPOL+II
             EMN(I)=EMN(I)+COEF(II)*COEF(JJ)*MIJ(IJ)*CCELEC
          ENDDO
          DO J=I+1,NORDER
             JJ=LSTPOL(J)
             IJ=(II-1)*NTPOL+JJ
             EMN(I)=EMN(I)+COEF(II)*COEF(JJ)*MIJ(IJ)*CCELEC
          ENDDO
       ENDDO
       EMIN=RBIG
       DO I=1,NORDER
          IF(abs(EMN(I)).lt.EMIN) THEN
             EMIN=abs(EMN(I))
             POLMIN=I
          ENDIF
       ENDDO
       TMPPOL=LSTPOL(POLMIN)
       DO I=POLMIN+1,NORDER
          LSTPOL(I-1)=LSTPOL(I)
       ENDDO
       LSTPOL(NORDER)=TMPPOL
    ENDDO
    !
100 CONTINUE
    !     QC_PS_UW04: 
    !     IF(ICALL.GT.0.OR.PRNLEV.LT.6) RETURN
    IF(ICALL.GT.0.OR.PRNLEV.LT.6) GOTO 110
    !
    WRITE(OUTU,'(/,3X,A)') &
         'Coefficients of Basis Functions and Running Summations in G_II'
    WRITE(OUTU,'(3X,A)') &
         ' NORDER   YL   YM        COEFF.       G_II(TOT)       G_II(N)'
    !
    DO I=1,NTPOL
       II=LSTPOL(I)
       IJ=(II-1)*NTPOL+II
       EMN(I)=HALF*COEF(II)*COEF(II)*MIJ(IJ)*CCELEC
       DO J=1,I-1
          JJ=LSTPOL(J)
          IJ=(II-1)*NTPOL+JJ
          IF(II.GT.JJ) IJ=(JJ-1)*NTPOL+II
          EMN(I)=EMN(I)+COEF(II)*COEF(JJ)*MIJ(IJ)*CCELEC
       ENDDO
    ENDDO

    TMPEMN=ZERO
    DO NORDER=1,NTPOL
       TMPEMN=TMPEMN+EMN(NORDER)
       II=LSTPOL(NORDER)
       L=LSTPL(II)
       M=LSTPM(II)
       WRITE(OUTU,'(3x,i5,2x,2i5,2x,3E15.7)') &
            NORDER,L,M,COEF(II),TMPEMN,EMN(NORDER)
    ENDDO
    !
110 CONTINUE
    ! QC_PS_UW04: now remove the QM atom contributions to COEF before 
    !  printing and returning to main program

#if KEY_SCCDFTB==1
    write(*,*) "QC> Chk qmused         in GSBP ",qmused
    write(*,*) "QC> Chk qmused_sccdftb in GSBP ",qmused_sccdftb
    if (qmused_sccdftb) then ! QC: 11/17 OK
    NSCCTMP=0
    KK=1
    i_loop4: DO I=1,NTRB
       J =LSTRB(I)
       ! if(qmused)then ! no igmsel() ?? !bug fixed by Xiya (QC: 11/17)
       IF ((IGMSEL(J) .NE. 1).and.(IGMSEL(J) .NE. 2)) cycle i_loop4
       ! endif (QC: 11/17)
       NSCCTMP=NSCCTMP + 1
       CCC=QMULI2(NSCCTMP,KK)
       ! Haibo Yu 
       IF (NSCCTMP .EQ. NSCCTC) THEN
          NSCCTMP = 0
          KK = KK + 1
       ENDIF
       ! Haibo Yu
       IF(CCC.EQ.ZERO) cycle i_loop4
       XDIFF=X(J)-RRXCEN
       YDIFF=Y(J)-RRYCEN
       ZDIFF=Z(J)-RRZCEN
       R2=XDIFF*XDIFF+YDIFF*YDIFF+ZDIFF*ZDIFF
       IF(R2.GT.SRDIST2) cycle i_loop4
       R=SQRT(R2)
       CT=ZDIFF/R
       ST=SQRT(ONE-(ZDIFF*ZDIFF)/R2)
       CP=XDIFF/R/ST
       SP=YDIFF/R/ST
       IF(R2.LT.RSMALL) THEN                      ! in the origin
          CT=ZERO
          CP=ZERO
          SP=ZERO
       ELSEIF(XDIFF.GT.-RSMALL.AND.XDIFF.LT.RSMALL.AND.  & ! in the Z-axis
            YDIFF.GT.-RSMALL.AND.YDIFF.LT.RSMALL) THEN
          CT=ONE
          IF(ZDIFF.LT.ZERO) CT=-ONE
          CP=ZERO
          SP=ZERO
       ELSEIF(ZDIFF.GT.-RSMALL.AND.ZDIFF.LT.RSMALL) THEN ! in the XY plane
          CT=ZERO
          CP=XDIFF/R
          SP=YDIFF/R
       ENDIF

       CALL RPOWERL2(LMAX,R,AR)           !  fill AR  (r^l   ) array
       CALL COSMPHI2(LMAX,CP,AC)          !  fill AC  (cos.. ) array
       CALL SINMPHI2(LMAX,CP,SP,AS)       !  fill AS  (sin.. ) array
       CALL ALPOL2(LMAX,CT,AP)       !  fill AP  (P(lm) ) array
       !  subtract off QM contributions
       DO II=1,NTPOL
          L=LSTPL(II)
          M=LSTPM(II)
          NORM=BNORM(II)
          IF(L.GE.0.AND.M.EQ.0) THEN
             COEF(II)=COEF(II)-CCC*NORM*AR(L)*AP(L,M)
          ELSEIF(L.GT.0.AND.M.GT.0) THEN
             COEF(II)=COEF(II)-CCC*NORM*AR(L)*AC(M)*AP(L,M)
          ELSEIF(L.GT.0.AND.M.LT.0) THEN
             M=-M
             COEF(II)=COEF(II)-CCC*NORM*AR(L)*AS(M)*AP(L,M)
          ENDIF
       ENDDO
    ENDDO i_loop4
    end if
#endif

    RETURN
  END SUBROUTINE SPHE_STPOL
  !
  SUBROUTINE RECT_GSBP3(NTRB,LSTRB,X,Y,Z,CG,NTPOL,MAXNPOL, &
       RRXCEN,RRYCEN,RRZCEN,XSCALE,YSCALE,ZSCALE, &
       MIJ,COEF, &
       RXNBFX,RXNBFY,RXNBFZ, &
       LSTPX,LSTPY,LSTPZ,BNORM, &
       ALPX,ALPY,ALPZ,ADLPX,ADLPY,ADLPZ, &
       XNPOL,YNPOL,ZNPOL, &
       MQ,LSTPOL,EGSBPB &
#if KEY_GCMC==1
       ,GCMCON  & 
#endif
       )
    !-----------------------------------------------------------------------
    !     calculate the coefficients of the basis functions and
    !               their derivatives
    !
    use chm_kinds
    use stream
    use number
    use consta
    implicit none
    real(chm_real)  X(*),Y(*),Z(*),CG(*),COEF(*),MIJ(*),BNORM(*),MQ(*)
    real(chm_real)  ALPX(*),ALPY(*),ALPZ(*),ADLPX(*),ADLPY(*),ADLPZ(*)
    real(chm_real)  RXNBFX(*),RXNBFY(*),RXNBFZ(*),EGSBPB
    real(chm_real)  RRXCEN,RRYCEN,RRZCEN
    real(chm_real)  XSCALE,YSCALE,ZSCALE
    INTEGER XNPOL,YNPOL,ZNPOL
    INTEGER NTRB,LSTRB(*),NTPOL,MAXNPOL
    INTEGER LSTPX(*),LSTPY(*),LSTPZ(*),LSTPOL(*)
#if KEY_GCMC==1
    LOGICAL GCMCON(:) 
#endif
    ! local
    real(chm_real)  DLPOL,XG,YG,ZG,DX,DY,DZ,NORM
    real(chm_real)  LPOLX,LPOLY,LPOLZ,CMIJ,CCC
    real(chm_real)  TMPFX,TMPFY,TMPFZ
    INTEGER I,J,II,JJ,IJ,LL,MM
    INTEGER XPI,YPI,ZPI

    ! construct MQ array to speed up the calculations
    DO I=1,MAXNPOL
       II=LSTPOL(I)
       MQ(II)=ZERO
       DO J=1,MAXNPOL
          JJ=LSTPOL(J)
          IJ=(II-1)*NTPOL+JJ
          IF(II.GT.JJ) IJ=(JJ-1)*NTPOL+II
          MQ(II)=MQ(II)+MIJ(IJ)*COEF(JJ)
       ENDDO
    ENDDO

    ! reaction field energy calculation
    EGSBPB=ZERO
    DO I=1,MAXNPOL
       II=LSTPOL(I)
       EGSBPB=EGSBPB+0.5d0*COEF(II)*MQ(II)
    ENDDO
    EGSBPB=EGSBPB*CCELEC

    ! QC_PS_UW_06  Initilization
    DO LL=1,NTRB
       MM=LSTRB(LL) 
       RXNBFX(MM)=ZERO
       RXNBFY(MM)=ZERO
       RXNBFZ(MM)=ZERO
    ENDDO

    ! reaction field force calculations
    ll_loop: DO LL=1,NTRB
       MM=LSTRB(LL)
#if KEY_GCMC==1
       IF (.NOT. GCMCON(MM)) cycle ll_loop 
#endif
       XG=X(MM)-RRXCEN
       YG=Y(MM)-RRYCEN
       ZG=Z(MM)-RRZCEN
       TMPFX=ZERO
       TMPFY=ZERO
       TMPFZ=ZERO
       CCC=CG(MM)*CCELEC
       IF(CCC.NE.ZERO)THEN
          CALL LPOL2(XNPOL,XG,XSCALE,ALPX)
          CALL LPOL2(YNPOL,YG,YSCALE,ALPY)
          CALL LPOL2(ZNPOL,ZG,ZSCALE,ALPZ)
          CALL DLPOL2(XNPOL,XG,XSCALE,ALPX,ADLPX)
          CALL DLPOL2(YNPOL,YG,YSCALE,ALPY,ADLPY)
          CALL DLPOL2(ZNPOL,ZG,ZSCALE,ALPZ,ADLPZ)
          DO I=1,MAXNPOL
             II=LSTPOL(I)
             XPI=LSTPX(II)
             YPI=LSTPY(II)
             ZPI=LSTPZ(II)
             NORM=BNORM(II)
             LPOLX=ALPX(XPI+1)
             LPOLY=ALPY(YPI+1)
             LPOLZ=ALPZ(ZPI+1)
             DX=NORM*ADLPX(XPI+1)*LPOLY*LPOLZ
             DY=NORM*LPOLX*ADLPY(YPI+1)*LPOLZ
             DZ=NORM*LPOLX*LPOLY*ADLPZ(ZPI+1)
             TMPFX=TMPFX-DX*MQ(II)
             TMPFY=TMPFY-DY*MQ(II)
             TMPFZ=TMPFZ-DZ*MQ(II)
          ENDDO
          RXNBFX(MM)=TMPFX*CCC
          RXNBFY(MM)=TMPFY*CCC
          RXNBFZ(MM)=TMPFZ*CCC
       ELSE
          RXNBFX(MM)=ZERO
          RXNBFY(MM)=ZERO
          RXNBFZ(MM)=ZERO
       ENDIF
    ENDDO ll_loop
    !
    RETURN
  END  SUBROUTINE RECT_GSBP3
  !
  SUBROUTINE SPHE_GSBP3(NTRB,LSTRB,X,Y,Z,CG,NTPOL,MAXNPOL, &
       RRXCEN,RRYCEN,RRZCEN, &
       MIJ,COEF, &
       RXNBFX,RXNBFY,RXNBFZ, &
       LSTPL,LSTPM,BNORM, &
       AR,AC,AS,AP,ADP,NMPOL, &
       MQ,LSTPOL,EGSBPB &
#if KEY_GCMC==1
       ,GCMCON  & 
#endif
       )
    !-----------------------------------------------------------------------
    !     calculate the electrostatic reaction field energy and forces
    !     in region of interest
    !
    use chm_kinds
    use stream
    use number
    use consta
    implicit none
    INTEGER NTRB,LSTRB(*)
    INTEGER LSTPL(*),LSTPM(*),LSTPOL(*),NTPOL,MAXNPOL,NMPOL
    real(chm_real)  X(*),Y(*),Z(*),CG(*),COEF(*),MIJ(*),BNORM(*),MQ(*)
    real(chm_real)  RXNBFX(*),RXNBFY(*),RXNBFZ(*),EGSBPB
    real(chm_real)  RRXCEN,RRYCEN,RRZCEN
    real(chm_real)  AR(0:NMPOL-1),AC(0:NMPOL-1),AS(0:NMPOL-1)
    real(chm_real)  AP(0:NMPOL-1,0:NMPOL-1),ADP(0:NMPOL-1,0:NMPOL-1)
#if KEY_GCMC==1
    LOGICAL GCMCON(:) 
#endif
    ! local
    INTEGER I,J,II,JJ,IJ,L,M,LL,MM,LMAX
    real(chm_real)  NORM
    real(chm_real)  CCC,CMIJ,RPL,CMP,SMP,APL
    real(chm_real)  SP,CP,ST,CT,R,R2,XDIFF,YDIFF,ZDIFF
    real(chm_real)  DX,DY,DZ,DR,DT,DP
    real(chm_real)  TMPFX,TMPFY,TMPFZ

    LMAX=NMPOL-1

    ! construct MQ array to speed up the calculations
    DO I=1,MAXNPOL
       II=LSTPOL(I)
       MQ(II)=ZERO
       DO J=1,MAXNPOL
          JJ=LSTPOL(J)
          IJ=(II-1)*NTPOL+JJ
          IF(II.GT.JJ) IJ=(JJ-1)*NTPOL+II
          MQ(II)=MQ(II)+MIJ(IJ)*COEF(JJ)
       ENDDO
    ENDDO

    ! reaction field energy calculation
    EGSBPB=ZERO
    DO I=1,MAXNPOL
       II=LSTPOL(I)
       EGSBPB=EGSBPB+0.5d0*COEF(II)*MQ(II)
    ENDDO
    EGSBPB=EGSBPB*CCELEC

    ! reaction field force calculations
    !     QC_UW031205: Darn it - not initialized?
    DO LL = 1,NTRB
       MM=LSTRB(LL)
       RXNBFX(MM)=ZERO
       RXNBFY(MM)=ZERO
       RXNBFZ(MM)=ZERO
    ENDDO
    !     QC_UW031205 DONE
    ll_loop: DO LL=1,NTRB
       MM=LSTRB(LL)
#if KEY_GCMC==1
       IF (.NOT. GCMCON(MM)) cycle ll_loop
#endif
       XDIFF=X(MM)-RRXCEN
       YDIFF=Y(MM)-RRYCEN
       ZDIFF=Z(MM)-RRZCEN
       TMPFX=ZERO
       TMPFY=ZERO
       TMPFZ=ZERO
       CCC=CG(MM)*CCELEC
       IF(CCC.NE.ZERO)THEN 
          R2=XDIFF*XDIFF+YDIFF*YDIFF+ZDIFF*ZDIFF
          R=SQRT(R2)
          CT=ZDIFF/R
          ST=SQRT(ONE-(ZDIFF*ZDIFF)/(R2))
          CP=XDIFF/R/ST
          SP=YDIFF/R/ST
          IF(R2.LT.RSMALL) THEN                               ! in the origin
             CT=ZERO
             ST=ZERO
             CP=ZERO
             SP=ZERO
          ELSEIF(XDIFF.GT.-RSMALL.AND.XDIFF.LT.RSMALL.AND.     & ! in the Z-axis
               YDIFF.GT.-RSMALL.AND.YDIFF.LT.RSMALL) THEN
             CT=ONE
             IF(ZDIFF.LT.ZERO) CT=-ONE
             ST=ZERO
             CP=ZERO
             SP=ZERO
          ELSEIF(ZDIFF.GT.-RSMALL.AND.ZDIFF.LT.RSMALL) THEN   ! in the XY plane
             CT=ZERO
             ST=ONE
             CP=XDIFF/R
             SP=YDIFF/R
          ENDIF

          CALL RPOWERL2(LMAX,R,AR)           !  fill AR  (r^l   ) array
          CALL COSMPHI2(LMAX,CP,AC)          !  fill AC  (cos.. ) array
          CALL SINMPHI2(LMAX,CP,SP,AS)       !  fill AS  (sin.. ) array
          CALL ALPOL2(LMAX,CT,AP)            !  fill AP  (P(lm) ) array
          CALL DALPOL2(LMAX,CT,AP,ADP)       !  fill ADP (DP(lm)) array

          DO I=1,MAXNPOL
             II=LSTPOL(I)
             L=LSTPL(II)
             M=LSTPM(II)
             NORM=BNORM(II)
             IF(M.EQ.0) THEN
                IF(L.EQ.0) THEN
                   DR=ZERO
                   DT=ZERO
                   DP=ZERO
                ELSE
                   RPL=ONE
                   IF(L.NE.0) RPL=AR(L-1)
                   DR= L*RPL*AP(L,M)
                   DT=-RPL*ADP(L,M)*ST
                   DP=ZERO
                ENDIF
             ELSEIF(M.GT.0) THEN
                RPL=ONE
                IF(L.NE.0) RPL=AR(L-1)
                CMP=AC(M)
                APL=AP(L,M)
                DR= L*RPL*CMP*APL
                DT=-RPL*CMP*ADP(L,M)*ST
                DP=-RPL*M*AS(M)*APL/ST
                IF(ST.EQ.ZERO) DP=ZERO
             ELSEIF(M.LT.0) THEN
                M=-M
                RPL=ONE
                IF(L.NE.0) RPL=AR(L-1)
                SMP=AS(M)
                APL=AP(L,M)
                DR= L*RPL*SMP*APL
                DT=-RPL*SMP*ADP(L,M)*ST
                DP= RPL*M*AC(M)*APL/ST
                IF(ST.EQ.ZERO) DP=ZERO
             ENDIF
             DX=NORM*(DR*ST*CP+DT*CT*CP-DP*SP)
             DY=NORM*(DR*ST*SP+DT*CT*SP+DP*CP)
             DZ=NORM*(DR*CT   -DT*ST         )
             TMPFX=TMPFX-DX*MQ(II)
             TMPFY=TMPFY-DY*MQ(II)
             TMPFZ=TMPFZ-DZ*MQ(II)
          ENDDO
          RXNBFX(MM)=TMPFX*CCC
          RXNBFY(MM)=TMPFY*CCC
          RXNBFZ(MM)=TMPFZ*CCC
       ELSE
          RXNBFX(MM)=ZERO
          RXNBFY(MM)=ZERO
          RXNBFZ(MM)=ZERO
       ENDIF
    ENDDO ll_loop
    !
    RETURN
  END SUBROUTINE SPHE_GSBP3
  !
  SUBROUTINE RECT_MIJ(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,NTPOL,NORDER, &
       LSTPX,LSTPY,LSTPZ,BNORM,DCEL, &
       XBCEN,YBCEN,ZBCEN, &
       RRXCEN,RRYCEN,RRZCEN,XSCALE,YSCALE,ZSCALE,IIP0,NPGD, &
       PHIP,PHIW,MIJ,CGSCAL)
    !-----------------------------------------------------------------------
    !     calculate the matrix M(ij)
    !
    use chm_kinds
    use number
    implicit none
    REAL(CHM_REAL4)  PHIP(*),PHIW(*)
    real(chm_real)  RRXCEN,RRYCEN,RRZCEN,CGSCAL,XBCEN,YBCEN,ZBCEN
    real(chm_real)  DCEL,TRANX,TRANY,TRANZ,XSCALE,YSCALE,ZSCALE
    real(chm_real)  MIJ(*),BNORM(*)
    INTEGER LSTPX(*),LSTPY(*),LSTPZ(*),NORDER,NTPOL
    INTEGER NCLx,NCLy,NCLz,NPGD,IIP0(*)
    ! local
    real(chm_real)  XG,YG,ZG,DCEL3,NORM
    real(chm_real)  XADD,YADD,ZADD
    INTEGER NC3,NCYZ,I,IG,JG,KG,IP0
    INTEGER XP,YP,ZP,N,IJ,JI
    !
    NCYZ  = NCLy*NCLz
    NC3   = NCYZ*NCLx
    DCEL3 = DCEL*DCEL*DCEL

    XADD = -TRANX+XBCEN-RRXCEN
    YADD = -TRANY+YBCEN-RRYCEN
    ZADD = -TRANZ+ZBCEN-RRZCEN
    !
    IF(NORDER.EQ.1) THEN
       N=1
       XP=LSTPX(N)
       YP=LSTPY(N)
       ZP=LSTPZ(N)
       NORM=BNORM(N)
       IJ=(N-1)*NTPOL+NORDER
       MIJ(IJ)=ZERO
       DO I=1,NPGD
          IP0=IIP0(I)
          IG=INT((IP0-1)/NCYZ)+1
          JG=INT(MOD((IP0-1),NCYZ)/NCLZ)+1
          KG=MOD(MOD((IP0-1),NCYZ),NCLZ)+1
          XG=DCEL*(IG-1)+XADD
          YG=DCEL*(JG-1)+YADD
          ZG=DCEL*(KG-1)+ZADD
          MIJ(IJ)=MIJ(IJ)+NORM*LPOL(XP,XG,XSCALE)* &
               LPOL(YP,YG,YSCALE)*LPOL(ZP,ZG,ZSCALE)* &
               DCEL3*(PHIW(IP0)-PHIP(IP0))*CGSCAL
       ENDDO
    ELSE
       DO N=1,NORDER
          XP=LSTPX(N)
          YP=LSTPY(N)
          ZP=LSTPZ(N)
          NORM=BNORM(N)
          IJ=(N-1)*NTPOL+NORDER
          MIJ(IJ)=ZERO
          DO I=1,NPGD
             IP0=IIP0(I)
             IG=INT((IP0-1)/NCYZ)+1
             JG=INT(MOD((IP0-1),NCYZ)/NCLZ)+1
             KG=MOD(MOD((IP0-1),NCYZ),NCLZ)+1
             XG=DCEL*(IG-1)+XADD
             YG=DCEL*(JG-1)+YADD
             ZG=DCEL*(KG-1)+ZADD
             MIJ(IJ)=MIJ(IJ)+NORM*LPOL(XP,XG,XSCALE)* &
                  LPOL(YP,YG,YSCALE)*LPOL(ZP,ZG,ZSCALE)* &
                  DCEL3*(PHIW(IP0)-PHIP(IP0))
          ENDDO
       ENDDO
    ENDIF
    !
    RETURN
  END SUBROUTINE RECT_MIJ
  !
  SUBROUTINE SPHE_MIJ(NPGD,DCEL,NTPOL,NORDER, &
       LSTPL,LSTPM,BNORM, &
       IIP0,CTHETA,CPHI,SPHI,RGD, &
       PHIP,PHIW,MIJ,CGSCAL)
    !-----------------------------------------------------------------------
    !     calculate the matrix M(ij)
    !
    use chm_kinds
    use number
    implicit none
    REAL(CHM_REAL4)  PHIP(*),PHIW(*)
    real(chm_real)  CTHETA(*),CPHI(*),SPHI(*),RGD(*),MIJ(*),BNORM(*)
    real(chm_real)  DCEL,CGSCAL
    INTEGER LSTPL(*),LSTPM(*),IIP0(*),NORDER,NTPOL,NPGD
    ! local
    real(chm_real)  DCEL3,NORM,CT,CP,SP,R
    real(chm_real)  FACTOR
    INTEGER I,L,M
    INTEGER N,IJ,IP0
    !
    DCEL3 = DCEL*DCEL*DCEL
    DO N=1,NORDER
       L=LSTPL(N)
       M=LSTPM(N)
       NORM=BNORM(N)
       IJ=(N-1)*NTPOL+NORDER
       MIJ(IJ)=ZERO

       IF(M.EQ.0) THEN
          IF(L.EQ.0) THEN
             FACTOR=ONE
             IF(NORDER.EQ.1) FACTOR=CGSCAL
             DO I=1,NPGD
                IP0=IIP0(I)
                MIJ(IJ)=MIJ(IJ)+ &
                     NORM*DCEL3*(PHIW(IP0)-PHIP(IP0))*FACTOR
             ENDDO
          ELSE
             DO I=1,NPGD
                IP0=IIP0(I)
                CT=CTHETA(I)
                R =RGD(I)
                MIJ(IJ)=MIJ(IJ)+ &
                     NORM*RPOWERL(L,R)*ALPOL(L,M,CT)* &
                     DCEL3*(PHIW(IP0)-PHIP(IP0))
             ENDDO
          ENDIF
       ELSEIF(M.GT.0) THEN
          DO I=1,NPGD
             IP0=IIP0(I)
             CT=CTHETA(I)
             CP=CPHI(I)
             SP=SPHI(I)
             R =RGD(I)
             MIJ(IJ)=MIJ(IJ)+ &
                  NORM*RPOWERL(L,R)*COSMPHI(M,CP)*ALPOL(L,M,CT)* &
                  DCEL3*(PHIW(IP0)-PHIP(IP0))
          ENDDO
       ELSEIF(M.LT.0) THEN
          M=-M
          DO I=1,NPGD
             IP0=IIP0(I)
             CT=CTHETA(I)
             CP=CPHI(I)
             SP=SPHI(I)
             R =RGD(I)
             MIJ(IJ)=MIJ(IJ)+ &
                  NORM*RPOWERL(L,R)*SINMPHI(M,CP,SP)*ALPOL(L,M,CT)* &
                  DCEL3*(PHIW(IP0)-PHIP(IP0))
          ENDDO
       ENDIF
       !         write(6,'(i5,e15.7)') N,MIJ(IJ)
    ENDDO
    !
    RETURN
  END  SUBROUTINE SPHE_MIJ
  !
  SUBROUTINE RECT_MAYER(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,DCEL, &
       XBCEN,YBCEN,ZBCEN, &
       FKAPA,EPSX,EPSY,EPSZ, &
       RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX)
    !-----------------------------------------------------------------------
    !     vacuum environment in region B.
    !
    use chm_kinds
    use number
    implicit none
    REAL(CHM_REAL4)  EPSX(*),EPSY(*),EPSZ(*),FKAPA(*)
    real(chm_real)  DCEL,TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN
    real(chm_real)  RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX
    INTEGER NCLx,NCLy,NCLz
    ! local
    real(chm_real)  XG,YG,ZG,XEPS,YEPS,ZEPS
    INTEGER NCYZ,IG,JG,KG,IP0X,IP0Y,IP0
    INTEGER JX1,JX2,JY1,JY2,JZ1,JZ2
    !
    NCYZ  = NCLy*NCLz
    !
    JX1=INT((TRANX+RBXMIN-XBCEN)/DCEL)-1
    JX2=INT((TRANX+RBXMAX-XBCEN)/DCEL)+3
    JY1=INT((TRANY+RBYMIN-YBCEN)/DCEL)-1
    JY2=INT((TRANY+RBYMAX-YBCEN)/DCEL)+3
    JZ1=INT((TRANZ+RBZMIN-ZBCEN)/DCEL)-1
    JZ2=INT((TRANZ+RBZMAX-ZBCEN)/DCEL)+3

    ! For Debye-Huckel screening factors
    ig_loop: DO IG=JX1,JX2
       XG=DCEL*(IG-1)-TRANX+XBCEN
       IF(XG.LT.RBXMIN-RSMALL.OR.XG.GT.RBXMAX+RSMALL) cycle ig_loop
       IP0X=(IG-1)*NCyz
       jg_loop: DO JG=JY1,JY2
          YG=DCEL*(JG-1)-TRANY+YBCEN
          IF(YG.LT.RBYMIN-RSMALL.OR.YG.GT.RBYMAX+RSMALL) cycle jg_loop
          IP0Y=(JG-1)*NCLz+IP0X
          kg_loop: DO KG=JZ1,JZ2
             ZG=DCEL*(KG-1)-TRANZ+ZBCEN
             IF(ZG < RBZMIN-RSMALL .OR. &
                  ZG > RBZMAX+RSMALL) cycle kg_loop
             IP0 = IP0Y + KG

             FKAPA(IP0)=ZERO

          ENDDO kg_loop
       ENDDO jg_loop
    ENDDO ig_loop

    ! For dielectric constants
    ! NOTE: dielectric constants are located at the middle point between grids
    DO IG=JX1,JX2
       IP0X=(IG-1)*NCyz
       XG=DCEL*(IG-1)-TRANX+XBCEN
       XEPS=DCEL*(IG-1)+DCEL*HALF-TRANX+XBCEN
       DO JG=JY1,JY2
          IP0Y=(JG-1)*NCLz+IP0X
          YG=DCEL*(JG-1)-TRANY+YBCEN
          YEPS=DCEL*(JG-1)+DCEL*HALF-TRANY+YBCEN
          DO KG=JZ1,JZ2
             IP0 = IP0Y + KG
             ZG=DCEL*(KG-1)-TRANZ+ZBCEN
             ZEPS=DCEL*(KG-1)+DCEL*HALF-TRANZ+ZBCEN

             IF(XEPS.GE.RBXMIN-RSMALL.AND.XEPS.LE.RBXMAX+RSMALL.AND. &
                  YG.GE.RBYMIN-RSMALL.AND.  YG.LE.RBYMAX+RSMALL.AND. &
                  ZG.GE.RBZMIN-RSMALL.AND.  ZG.LE.RBZMAX+RSMALL     ) &
                  EPSX(IP0)=ONE
             IF(  XG.GE.RBXMIN-RSMALL.AND.  XG.LE.RBXMAX+RSMALL.AND. &
                  YEPS.GE.RBYMIN-RSMALL.AND.YEPS.LE.RBYMAX+RSMALL.AND. &
                  ZG.GE.RBZMIN-RSMALL.AND.  ZG.LE.RBZMAX+RSMALL     ) &
                  EPSY(IP0)=ONE
             IF(  XG.GE.RBXMIN-RSMALL.AND.  XG.LE.RBXMAX+RSMALL.AND. &
                  YG.GE.RBYMIN-RSMALL.AND.  YG.LE.RBYMAX+RSMALL.AND. &
                  ZEPS.GE.RBZMIN-RSMALL.AND.ZEPS.LE.RBZMAX+RSMALL     ) &
                  EPSZ(IP0)=ONE

          ENDDO
       ENDDO
    ENDDO
    !
    RETURN
  END SUBROUTINE RECT_MAYER
  !
  SUBROUTINE SPHE_MAYER(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,DCEL, &
       XBCEN,YBCEN,ZBCEN, &
       FKAPA,EPSX,EPSY,EPSZ, &
       RRXCEN,RRYCEN,RRZCEN,SRDIST)
    !-----------------------------------------------------------------------
    !     vacuum environment in region B.
    !
    use chm_kinds
    use number
    implicit none
    REAL(CHM_REAL4)  EPSX(*),EPSY(*),EPSZ(*),FKAPA(*)
    real(chm_real)  DCEL,TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN
    real(chm_real)  RRXCEN,RRYCEN,RRZCEN,SRDIST
    INTEGER NCLx,NCLy,NCLz
    ! local
    real(chm_real)  XG,YG,ZG,XG2,YG2,ZG2,DSQ1,XEPS,YEPS,ZEPS,SR2
    INTEGER IG,JG,KG
    INTEGER IP0,IP0X,IP0Y,NCYZ
    INTEGER NFIL,IXCEN,IYCEN,IZCEN,JX1,JX2,JY1,JY2,JZ1,JZ2

    NFIL=INT(SRDIST/DCEL)+2
    IXCEN=INT((RRXCEN+TRANX-XBCEN)/DCEL)+1
    IYCEN=INT((RRYCEN+TRANY-YBCEN)/DCEL)+1
    IZCEN=INT((RRZCEN+TRANZ-ZBCEN)/DCEL)+1
    JX1=IXCEN-NFIL
    JX2=IXCEN+NFIL+1
    JY1=IYCEN-NFIL
    JY2=IYCEN+NFIL+1
    JZ1=IZCEN-NFIL
    JZ2=IZCEN+NFIL+1

    ! For dielectric constants and Debye-Huckel screening factors
    ! NOTE: dielectric constants are located at the middle point between grids
    NCYZ= NCLy*NCLz
    SR2=SRDIST*SRDIST+rsmall
    DO IG=JX1,JX2
       XG=DCEL*(IG-1)-TRANX+XBCEN-RRXCEN
       XG2=XG*XG
       IP0X=(IG-1)*NCyz
       DO JG=JY1,JY2
          YG=DCEL*(JG-1)-TRANY+YBCEN-RRYCEN
          YG2=YG*YG
          IP0Y=(JG-1)*NCLz+IP0X
          DO KG=JZ1,JZ2
             ZG=DCEL*(KG-1)-TRANZ+ZBCEN-RRZCEN
             ZG2=ZG*ZG
             IP0 = IP0Y + KG
             IF(FKAPA(IP0).ne.ZERO)THEN
                DSQ1=XG2+YG2+ZG2
                IF(DSQ1.LE.SR2) FKAPA(IP0)=ZERO
             ENDIF
             IF(EPSX(IP0).ne.ONE)THEN
                XEPS=XG+HALF*DCEL
                DSQ1=XEPS*XEPS+YG2+ZG2
                IF(DSQ1.LE.SR2) EPSX(IP0)=ONE
             ENDIF
             IF(EPSY(IP0).ne.ONE)THEN
                YEPS=YG+HALF*DCEL
                DSQ1=XG2+YEPS*YEPS+ZG2
                IF(DSQ1.LE.SR2) EPSY(IP0)=ONE
             ENDIF
             IF(EPSZ(IP0).ne.ONE)THEN
                ZEPS=ZG+HALF*DCEL
                DSQ1=XG2+YG2+ZEPS*ZEPS
                IF(DSQ1.LE.SR2) EPSZ(IP0)=ONE
             ENDIF
          ENDDO
       ENDDO
    ENDDO
    !
    RETURN
  END SUBROUTINE SPHE_MAYER
  !
  SUBROUTINE RECT_GPINRR(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,DCEL, &
       XBCEN,YBCEN,ZBCEN, &
       RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX, &
       JX1,JX2,JY1,JY2,JZ1,JZ2,IIP0,NPGD)
    !-----------------------------------------------------------------------
    !     find out the grid points inside region of interest and
    !     store in IIP0 array
    !
    use chm_kinds
    use number
    implicit none
    real(chm_real)  RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX
    real(chm_real)  DCEL,TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN
    INTEGER NCLx,NCLy,NCLz,NPGD,IIP0(*)
    INTEGER JX1,JX2,JY1,JY2,JZ1,JZ2
    ! local
    real(chm_real)  XG,YG,ZG
    INTEGER NCYZ,IG,JG,KG,IP0X,IP0Y,IP0
    !
    NPGD=0
    NCYZ =NCLy*NCLz
    ig_loop: DO IG=JX1,JX2
       XG=DCEL*(IG-1)-TRANX+XBCEN
       IF(XG.LT.RBXMIN.OR.XG.GT.RBXMAX) cycle ig_loop
       IP0X=(IG-1)*NCyz
       jg_loop: DO JG=JY1,JY2
          YG=DCEL*(JG-1)-TRANY+YBCEN
          IF(YG.LT.RBYMIN.OR.YG.GT.RBYMAX) cycle jg_loop
          IP0Y=(JG-1)*NCLz+IP0X
          kg_loop: DO KG=JZ1,JZ2
             ZG=DCEL*(KG-1)-TRANZ+ZBCEN
             IF(ZG.LT.RBZMIN.OR.ZG.GT.RBZMAX) cycle kg_loop
             IP0 = IP0Y + KG
             NPGD=NPGD+1
             IIP0(NPGD)=IP0
          ENDDO kg_loop
       ENDDO jg_loop
    ENDDO ig_loop
    !
    RETURN
  END SUBROUTINE RECT_GPINRR
  !
  SUBROUTINE SPHE_TRIG(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,DCEL, &
       XBCEN,YBCEN,ZBCEN, &
       IIP0,CTHETA,CPHI,SPHI,RGD,SRDIST, &
       RRXCEN,RRYCEN,RRZCEN,JX1,JX2,JY1,JY2,JZ1,JZ2,NPGD)
    !-----------------------------------------------------------------------
    !     calculate the trigonometric functions of each grid points
    !     inside region of interest
    !
    use chm_kinds
    use consta
    use number
    implicit none
    real(chm_real)  SRDIST,RRXCEN,RRYCEN,RRZCEN
    real(chm_real)  DCEL,TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN
    real(chm_real)  CTHETA(*),CPHI(*),SPHI(*),RGD(*)
    INTEGER NCLx,NCLy,NCLz,IIP0(*)
    INTEGER JX1,JX2,JY1,JY2,JZ1,JZ2,NPGD
    ! local
    real(chm_real)  XG,YG,ZG,XG2,YG2,ZG2,R,SR2,STHETA
    INTEGER NC3,NCYZ,IG,JG,KG,IP0X,IP0Y,IP0
    !
    NCYZ =NCLy*NCLz
    NC3  =NCYZ*NCLx
    SR2  =SRDIST*SRDIST+rsmall

    ! calculate the trigonometric functions of each grid points
    ! inside region of interest
    NPGD=0
    DO IG=JX1,JX2
       XG=DCEL*(IG-1)-TRANX+XBCEN-RRXCEN
       XG2=XG*XG
       IP0X=(IG-1)*NCyz
       DO JG=JY1,JY2
          YG=DCEL*(JG-1)-TRANY+YBCEN-RRYCEN
          YG2=YG*YG+XG2
          IP0Y=(JG-1)*NCLz+IP0X
          DO KG=JZ1,JZ2
             ZG=DCEL*(KG-1)-TRANZ+ZBCEN-RRZCEN
             ZG2=ZG*ZG+YG2
             IP0 = IP0Y + KG
             IF(ZG2.LE.SR2) THEN
                !     in the origin
                IF(ZG2.LT.RSMALL) THEN
                   NPGD=NPGD+1
                   IIP0(NPGD)=IP0
                   RGD(NPGD)=ZERO
                   CTHETA(NPGD)=ZERO
                   CPHI(NPGD)=ZERO
                   SPHI(NPGD)=ZERO
                   !     in the Z axis
                ELSEIF(XG.GT.-RSMALL.AND.XG.LT.RSMALL.AND. &
                     YG.GT.-RSMALL.AND.YG.LT.RSMALL) THEN
                   NPGD=NPGD+1
                   IIP0(NPGD)=IP0
                   RGD(NPGD)=abs(ZG)
                   CTHETA(NPGD)=ZG/abs(ZG)
                   CPHI(NPGD)=ZERO
                   SPHI(NPGD)=ZERO
                   !     in the XY plane
                ELSEIF(ZG.GT.-RSMALL.AND.ZG.LT.RSMALL) THEN
                   NPGD=NPGD+1
                   R=SQRT(ZG2)
                   IIP0(NPGD)=IP0
                   RGD(NPGD)=R
                   CTHETA(NPGD)=ZERO
                   CPHI(NPGD)=XG/R
                   SPHI(NPGD)=YG/R
                ELSE
                   NPGD=NPGD+1
                   R=SQRT(ZG2)
                   IIP0(NPGD)=IP0
                   RGD(NPGD)=R
                   CTHETA(NPGD)=ZG/R
                   STHETA=SQRT(ONE-ZG*ZG/ZG2)
                   CPHI(NPGD)=XG/R/STHETA
                   SPHI(NPGD)=YG/R/STHETA
                ENDIF
             ENDIF
          ENDDO
       ENDDO
    ENDDO
    !
    RETURN
  END SUBROUTINE SPHE_TRIG
  !
  SUBROUTINE RECT_CHC(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,DCEL, &
       XBCEN,YBCEN,ZBCEN, &
       NORDER,LSTPX,LSTPY,LSTPZ,BNORM, &
       CHC,CGSCAL,IIP0,NPGD, &
       RRXCEN,RRYCEN,RRZCEN,XSCALE,YSCALE,ZSCALE)
    !-----------------------------------------------------------------------
    !     store the basis functions (Legendre Polynomials) in ICHC (CDEN) array
    !     NOTE: CHC is a charge (not density) arrary and containes 4*pi/h
    !
    use chm_kinds
    use consta
    use number
    implicit none
    REAL(CHM_REAL4)  CHC(*)
    real(chm_real)  RRXCEN,RRYCEN,RRZCEN,BNORM(*)
    real(chm_real)  DCEL,TRANX,TRANY,TRANZ,XSCALE,YSCALE,ZSCALE,CGSCAL
    real(chm_real)  XBCEN,YBCEN,ZBCEN
    INTEGER NCLx,NCLy,NCLz,NPGD,IIP0(*)
    INTEGER LSTPX(*),LSTPY(*),LSTPZ(*),NORDER
    ! local
    real(chm_real)  XG,YG,ZG,DCEL3,NORM
    real(chm_real)  XADD,YADD,ZADD,FACTOR
    INTEGER NC3,NCYZ,I,IG,JG,KG,IP0,XPOL,YPOL,ZPOL
    !
    XPOL=LSTPX(NORDER)
    YPOL=LSTPY(NORDER)
    ZPOL=LSTPZ(NORDER)
    NORM=BNORM(NORDER)
    !
    NCYZ =NCLy*NCLz
    NC3  =NCYZ*NCLx
    DCEL3=DCEL*DCEL*DCEL

    ! initialize CHC array
    DO I=1,NC3
       CHC(I)=ZERO
    ENDDO

    XADD = -TRANX+XBCEN-RRXCEN
    YADD = -TRANY+YBCEN-RRYCEN
    ZADD = -TRANZ+ZBCEN-RRZCEN

    ! the Legendre polynomials are normalized inside region of interest
    IF(NORDER.EQ.1) THEN
       FACTOR=DCEL3*TWO*TWOPI/DCEL/CGSCAL
       DO I=1,NPGD
          IP0=IIP0(I)
          IG=INT((IP0-1)/NCYZ)+1
          JG=INT(MOD((IP0-1),NCYZ)/NCLZ)+1
          KG=MOD(MOD((IP0-1),NCYZ),NCLZ)+1
          XG=DCEL*(IG-1)+XADD
          YG=DCEL*(JG-1)+YADD
          ZG=DCEL*(KG-1)+ZADD
          CHC(IP0)=NORM*LPOL(XPOL,XG,XSCALE)*LPOL(YPOL,YG,YSCALE)* &
               LPOL(ZPOL,ZG,ZSCALE)*FACTOR
       ENDDO
    ELSE
       FACTOR=DCEL3*TWO*TWOPI/DCEL
       DO I=1,NPGD
          IP0=IIP0(I)
          IG=INT((IP0-1)/NCYZ)+1
          JG=INT(MOD((IP0-1),NCYZ)/NCLZ)+1
          KG=MOD(MOD((IP0-1),NCYZ),NCLZ)+1
          XG=DCEL*(IG-1)+XADD
          YG=DCEL*(JG-1)+YADD
          ZG=DCEL*(KG-1)+ZADD
          CHC(IP0)=NORM*LPOL(XPOL,XG,XSCALE)*LPOL(YPOL,YG,YSCALE)* &
               LPOL(ZPOL,ZG,ZSCALE)*FACTOR
       ENDDO
    ENDIF
    !
    !      DO I=1,NPGD
    !         IP0=IIP0(I)
    !         IG=INT((IP0-1)/NCYZ)+1
    !         JG=INT(MOD((IP0-1),NCYZ)/NCLZ)+1
    !         KG=MOD(MOD((IP0-1),NCYZ),NCLZ)+1
    !         XG=(DCEL*(IG-1)-TRANX-RRXCEN)*10.0
    !         YG=(DCEL*(JG-1)-TRANY-RRYCEN)*10.0
    !         ZG=(DCEL*(KG-1)-TRANZ-RRZCEN)*10.0
    !         IF(CHC(ip0).GT.RSMALL) THEN
    !            write(50+NORDER,101)
    !     $           'ATOM',I,' POL ARG     1',
    !     $           XG,YG,ZG,
    !     $           CHC(ip0),'  1.00      LPOL'
    !         ELSEIF(CHC(ip0).LT.-RSMALL) THEN
    !            write(50+NORDER,101)
    !     $           'ATOM',I,' POL GLU     2',
    !     $           XG,YG,ZG,
    !     $           CHC(ip0),' 10.00      LPOL'
    !         ELSE
    !            write(50+NORDER,101)
    !     $           'ATOM',I,' POL CYS     3',
    !     $           XG,YG,ZG,
    !     $           CHC(ip0),' 10.00      LPOL'
    !         ENDIF
    !      ENDDO
    ! 101  format(a,2x,i5,1x,a,4x,3f8.3,f6.2,a)
    !
    RETURN
  END SUBROUTINE RECT_CHC
  !
  SUBROUTINE SPHE_CHC(NCEL3,NPGD,DCEL,NORDER, &
       IIP0,CTHETA,CPHI,SPHI,RGD, &
       LSTPL,LSTPM,BNORM,CHC,CGSCAL)
    !-----------------------------------------------------------------------
    !     store the basis functions (shperical harmonics) in ICHC (CDEN) array
    !     NOTE: CHC is a charge (not density) arrary and containes 4*pi/h
    !
    use chm_kinds
    use consta
    use number
    implicit none
    REAL(CHM_REAL4)  CHC(*)
    real(chm_real)  CTHETA(*),CPHI(*),SPHI(*),RGD(*),BNORM(*), &
         CGSCAL,DCEL
    INTEGER NCEL3,NPGD,NORDER
    INTEGER IIP0(*),LSTPL(*),LSTPM(*)
    ! local
    real(chm_real)  DCEL3,NORM,R,CT,CP,SP
    INTEGER I,L,M,IP0
    !
    L   =LSTPL(NORDER)
    M   =LSTPM(NORDER)
    NORM=BNORM(NORDER)
    DCEL3=DCEL*DCEL*DCEL

    ! initialize CHC array
    DO I=1,NCEL3
       CHC(I)=ZERO
    ENDDO

    ! the basis functions are normalized inside region of interest
    IF(M.EQ.0) THEN
       IF(L.EQ.0) THEN
          DO I=1,NPGD
             IP0=IIP0(I)
             CHC(IP0)=NORM*DCEL3*TWO*TWOPI/DCEL/CGSCAL
          ENDDO
       ELSE
          DO I=1,NPGD
             IP0=IIP0(I)
             CT=CTHETA(I)
             R =RGD(I)
             CHC(IP0)=NORM*RPOWERL(L,R)*ALPOL(L,M,CT)* &
                  DCEL3*TWO*TWOPI/DCEL
          ENDDO
       ENDIF
    ELSEIF(M.GT.0) THEN
       DO I=1,NPGD
          IP0=IIP0(I)
          CT=CTHETA(I)
          CP=CPHI(I)
          R =RGD(I)
          CHC(IP0)=NORM*RPOWERL(L,R)*COSMPHI(M,CP)*ALPOL(L,M,CT)* &
               DCEL3*TWO*TWOPI/DCEL
       ENDDO
    ELSEIF(M.LT.0) THEN
       M=-M
       DO I=1,NPGD
          IP0=IIP0(I)
          CT=CTHETA(I)
          CP=CPHI(I)
          SP=SPHI(I)
          R =RGD(I)
          CHC(IP0)=NORM*RPOWERL(L,R)*SINMPHI(M,CP,SP)*ALPOL(L,M,CT)* &
               DCEL3*TWO*TWOPI/DCEL
       ENDDO
    ENDIF
    !
    !      DO I=1,NPGD
    !         IP0=IIP0(I)
    !         CT=CTHETA(I)
    !         CP=CPHI(I)
    !         SP=SPHI(I)
    !         R =RGD(I)*10.0
    !         IF(CHC(ip0).GT.RSMALL) THEN
    !            write(89+NORDER,101)
    !     $           'ATOM',I,' POL ARG     1',
    !     $           R*SQRT(1-CT*CT)*CP,R*SQRT(1-CT*CT)*SP,R*CT,
    !     $           CHC(ip0)*100.,'  1.00      LPOL'
    !         ELSEIF(CHC(ip0).LT.-RSMALL) THEN
    !            write(89+NORDER,101)
    !     $           'ATOM',I,' POL GLU     2',
    !     $           R*SQRT(1-CT*CT)*CP,R*SQRT(1-CT*CT)*SP,R*CT,
    !     $           CHC(ip0)*100.,' 10.00      LPOL'
    !         ELSE
    !            write(89+NORDER,101)
    !     $           'ATOM',I,' POL CYS     3',
    !     $           R*SQRT(1-CT*CT)*CP,R*SQRT(1-CT*CT)*SP,R*CT,
    !     $           CHC(ip0)*100.,' 10.00      LPOL'
    !         ENDIF
    !      ENDDO
    ! 101  format(a,2x,i5,1x,a,4x,3f8.3,f6.2,a)
    !
    RETURN
  END SUBROUTINE SPHE_CHC
  !
  SUBROUTINE RECTPOL(XNPOL,YNPOL,ZNPOL,OLDNTP,QMIJ, &
       LSTPX,LSTPY,LSTPZ,LSTPOL)
    !-----------------------------------------------------------------------
    !     store the basis functions
    !     in LSTPX, LSTPY, and LSTPZ array for Legendre Polynomials
    !
    use chm_kinds
    implicit none
    INTEGER XNPOL,YNPOL,ZNPOL,OLDNTP
    INTEGER LSTPX(*),LSTPY(*),LSTPZ(*),LSTPOL(*)
    LOGICAL QMIJ
    ! local
    INTEGER XPOL,YPOL,ZPOL,NORDER,N,NTPOL
    !
    NTPOL=XNPOL*YNPOL*ZNPOL
    IF(QMIJ.AND.OLDNTP.EQ.NTPOL) THEN
       DO NORDER=1,NTPOL
          LSTPOL(NORDER)=NORDER
       ENDDO
       RETURN
    ENDIF
    !
    NORDER=0
    IF(QMIJ) THEN
       DO NORDER=1,OLDNTP
          LSTPOL(NORDER)=NORDER
       ENDDO
       NORDER=OLDNTP
       DO XPOL=0,XNPOL-1
          DO YPOL=0,YNPOL-1
             zpol_loop: DO ZPOL=0,ZNPOL-1
                DO N=1,OLDNTP
                   IF(LSTPX(N).EQ.XPOL.AND.LSTPY(N).EQ.YPOL.AND. &
                        LSTPZ(N).EQ.ZPOL) cycle zpol_loop
                ENDDO
                NORDER=NORDER+1
                LSTPX(NORDER)=XPOL
                LSTPY(NORDER)=YPOL
                LSTPZ(NORDER)=ZPOL
                LSTPOL(NORDER)=NORDER
             ENDDO zpol_loop
          ENDDO
       ENDDO
    ELSE
       DO XPOL=0,XNPOL-1
          DO YPOL=0,YNPOL-1
             DO ZPOL=0,ZNPOL-1
                NORDER=NORDER+1
                LSTPX(NORDER)=XPOL
                LSTPY(NORDER)=YPOL
                LSTPZ(NORDER)=ZPOL
                LSTPOL(NORDER)=NORDER
             ENDDO
          ENDDO
       ENDDO
    ENDIF
    !
    RETURN
  END SUBROUTINE RECTPOL
  !
  SUBROUTINE SPHEPOL(NMPOL,LSTPL,LSTPM,LSTPOL)
    !-----------------------------------------------------------------------
    !     store the basis functions
    !     in LSTPL and LSTPM array for Spherical Harmonics
    !
    use chm_kinds
    implicit none
    INTEGER NMPOL,LSTPL(*),LSTPM(*),LSTPOL(*)
    ! local
    INTEGER L,M,NORDER

    ! always the same order in spherical harmonics
    NORDER=0
    DO L=0,NMPOL-1
       NORDER=NORDER+1
       LSTPOL(NORDER)=NORDER
       LSTPL(NORDER) =L
       LSTPM(NORDER) =0
       DO M=1,L
          NORDER=NORDER+1
          LSTPOL(NORDER)=NORDER
          LSTPL(NORDER) =L
          LSTPM(NORDER) =M
          NORDER=NORDER+1
          LSTPOL(NORDER)=NORDER
          LSTPL(NORDER) =L
          LSTPM(NORDER) =-M
       ENDDO
    ENDDO
    !
    RETURN
  END SUBROUTINE SPHEPOL
  !
  SUBROUTINE RECT_NORM(NTPOL,XSCALE,YSCALE,ZSCALE, &
       LSTPX,LSTPY,LSTPZ,BNORM)
    !-----------------------------------------------------------------------
    !     calculate the normalization constants of the basis functions
    !
    use chm_kinds
    use number
    use consta
    implicit none
    INTEGER NTPOL,LSTPX(*),LSTPY(*),LSTPZ(*)
    real(chm_real)  BNORM(*),XSCALE,YSCALE,ZSCALE
    ! local
    real(chm_real)  ILXYZ
    INTEGER N,LPOL,MPOL,NPOL

    ILXYZ=XSCALE*YSCALE*ZSCALE/EIGHT      ! inverse Lxyz
    DO N=1,NTPOL
       LPOL=LSTPX(N)*TWO
       MPOL=LSTPY(N)*TWO
       NPOL=LSTPZ(N)*TWO
       BNORM(N)=SQRT((LPOL+ONE)*(MPOL+ONE)*(NPOL+ONE)*ILXYZ)
    ENDDO
    !
    RETURN
  END SUBROUTINE RECT_NORM
  !
  SUBROUTINE SPHE_NORM(NMPOL,BNORM,SRDIST)
    !-----------------------------------------------------------------------
    !     calculate the normalization constants of the basis functions
    !
    use chm_kinds
    use number
    use consta
    implicit none
    INTEGER NMPOL
    real(chm_real)  BNORM(*),SRDIST
    ! local
    INTEGER L,M,NORDER
    real(chm_real)  SR2,SR3,LPART,RPART,UPFACTO,DNFACTO

    ! always the same order in spherical harmonics
    NORDER=1
    SR2=SRDIST*SRDIST
    SR3=SR2*SRDIST
    RPART=SR3
    LPART=THREE/TWO/TWOPI
    BNORM(NORDER)=SQRT(LPART/RPART)
    DO L=1,NMPOL-1
       LPART=(TWO*L+ONE)*(TWO*L+THREE)/(TWO*TWOPI)
       RPART=RPART*SR2
       NORDER=NORDER+1
       BNORM(NORDER)=SQRT(LPART/RPART)
       LPART=(TWO*L+ONE)*(TWO*L+THREE)/TWOPI       ! change for m > 0
       DO M=1,L
          NORDER=NORDER+1
          UPFACTO=FACTORI(L-M)
          DNFACTO=FACTORI(L+M)
          BNORM(NORDER)=SQRT(LPART*UPFACTO/(RPART*DNFACTO))
          NORDER=NORDER+1
          BNORM(NORDER)=BNORM(NORDER-1)
       ENDDO
    ENDDO
    !
    RETURN
  END  SUBROUTINE SPHE_NORM

  SUBROUTINE RPMIJ(NTPOL,MIJ,OLDNTP,OLDMIJ)
    !-----------------------------------------------------------------------
    !  replace OLDMIJ with MIJ
    !
    use chm_kinds
    use number
    implicit none
    INTEGER  NTPOL,OLDNTP
    real(chm_real)   MIJ(*),OLDMIJ(*)
    ! Local variables
    INTEGER  I,J,IJ,OLDIJ
    !
    DO I=1,NTPOL
       DO J=1,NTPOL
          IJ=(I-1)*NTPOL+J
          MIJ(IJ)=ZERO
       ENDDO
    ENDDO
    DO I=1,OLDNTP
       DO J=1,OLDNTP
          IJ=(I-1)*NTPOL+J
          OLDIJ=(I-1)*OLDNTP+J
          MIJ(IJ)=OLDMIJ(OLDIJ)
       ENDDO
    ENDDO
    !
    RETURN
  END SUBROUTINE RPMIJ
  SUBROUTINE RPPOL(NTPOL,LSTP,OLDNTP,OLDP)
    !-----------------------------------------------------------------------
    !  replace OLDP with LSTP
    !
    use chm_kinds
    use number
    implicit none
    INTEGER  NTPOL,OLDNTP
    INTEGER  LSTP(*),OLDP(*)
    ! Local variables
    INTEGER  I
    !
    DO I=1,NTPOL
       LSTP(I)=1000
    ENDDO
    DO I=1,OLDNTP
       LSTP(I)=OLDP(I)
    ENDDO
    !
    RETURN
  END SUBROUTINE RPPOL
  !
  FUNCTION DALPOL(L,M,X) result(dalpol_rslt)
    !------------------------------------------------------------------------
    !     Derivatives of Associate Legendre Polynomials (From Smythe's BOOK)
    !
    use chm_kinds
    use number
    implicit none
    real(chm_real) :: dalpol_rslt
    integer l,m
    real(chm_real)  x
    ! local
    real(chm_real)  fact
    !
    IF(X.EQ.ONE.OR.X.EQ.-ONE) THEN
       IF(M.EQ.0) THEN
          IF(X.EQ. ONE) DALPOL_RSLT=L*(L+ONE)/TWO
          IF(X.EQ.-ONE) DALPOL_RSLT=(-ONE)**(L+1)*L*(L+ONE)/TWO
       ELSE
          DALPOL_RSLT=ZERO
       ENDIF
       RETURN
    ENDIF

    IF(X.GT.ONE) THEN
       FACT=ONE/SQRT(X*X-ONE)
       DALPOL_RSLT=FACT*(ALPOL(L,M+1,X)+FACT*M*X*ALPOL(L,M,X))
    ELSE
       FACT=ONE/SQRT(ONE-X*X)
       DALPOL_RSLT=FACT*(ALPOL(L,M+1,X)-FACT*M*X*ALPOL(L,M,X))
    ENDIF
    !
    RETURN
  END FUNCTION DALPOL
  !
  FUNCTION ALPOL(L,M,X) result(alpol_rslt)
    !------------------------------------------------------------------------
    !     The Associate Legendre Polynomials
    !     (Numerical recipes in Fortran 77: 6.8. Spherical Harmonics)
    !
    use chm_kinds
    use number
    implicit none
    real(chm_real) :: alpol_rslt
    integer l,m
    real(chm_real)  x
    ! local
    integer i,ll
    real(chm_real)  fact,pll,pmm,pmmp1,somx2
    !
    if(m.gt.l) then
       alpol_rslt=zero
       return
    elseif(m.lt.0.or.x.lt.-one) then
       CALL WRNDIE(-5,'<ALPOL>','BAD ARGUMENTS IN ALPOL')
    endif

    ! compute pmm
    pmm=one
    if(m.gt.0) then
       if(x.gt.one) then
          somx2=sqrt((x-one)*(x+one))
       else
          somx2=sqrt((one-x)*(one+x))
       endif
       fact=one
       do i=1,m
          !wi            pmm=-pmm*fact*somx2   ! to remove (-1)^m in P(l,m)
          pmm=pmm*fact*somx2
          fact=fact+two
       enddo
    endif

    ! compute pmmp1
    if(l.eq.m) then
       alpol_rslt=pmm
    else
       pmmp1=x*(2*m+1)*pmm
       if(l.eq.m+1) then
          alpol_rslt=pmmp1
       else
          do ll=m+2,l
             pll=(x*(two*ll-1)*pmmp1-(ll+m-1)*pmm)/(ll-m)
             pmm=pmmp1
             pmmp1=pll
          enddo
          alpol_rslt=pll
       endif
    endif
    !
    RETURN
  END FUNCTION ALPOL
  !
  FUNCTION SINMPHI(M,C0,S0) result(sinmphi_rslt)
    !------------------------------------------------------------------------
    !     sin(M*phi) calculation (M > 0)
    !
    use chm_kinds
    use number
    implicit none
    real(chm_real) :: sinmphi_rslt
    !
    INTEGER   M,I
    real(chm_real)    C0,S0,S2,S3
    !
    IF(M.EQ.0) THEN
       SINMPHI_RSLT=ZERO
    ELSEIF(M.EQ.1) THEN
       SINMPHI_RSLT=S0
    ELSEIF(M.EQ.2) THEN
       SINMPHI_RSLT=TWO*C0*S0
    ELSEIF(M.EQ.3) THEN
       S2=TWO*C0*S0
       SINMPHI_RSLT=TWO*C0*S2-S0
    ELSEIF(M.GE.4) THEN
       S2=TWO*C0*S0
       S3=TWO*C0*S2-S0
       DO I=4,M
          SINMPHI_RSLT=TWO*C0*S3-S2
          S2=S3
          S3=SINMPHI_RSLT
       ENDDO
    ENDIF
    !
    RETURN
  END FUNCTION SINMPHI
  !
  FUNCTION COSMPHI(M,C0) result(cosmphi_rslt)
    !------------------------------------------------------------------------
    !     cos(M*phi) calculation (M > 0)
    !
    use chm_kinds
    use number
    implicit none
    real(chm_real) :: cosmphi_rslt
    !
    INTEGER   M,I
    real(chm_real)    C0,C2,C3
    !
    IF(M.EQ.0) THEN
       COSMPHI_RSLT=ONE
    ELSEIF(M.EQ.1) THEN
       COSMPHI_RSLT=C0
    ELSEIF(M.EQ.2) THEN
       COSMPHI_RSLT=TWO*C0*C0-ONE
    ELSEIF(M.EQ.3) THEN
       C2=TWO*C0*C0-ONE
       COSMPHI_RSLT=TWO*C0*C2-C0
    ELSEIF(M.GE.4) THEN
       C2=TWO*C0*C0-ONE
       C3=TWO*C0*C2-C0
       DO I=4,M
          COSMPHI_RSLT=TWO*C0*C3-C2
          C2=C3
          C3=COSMPHI_RSLT
       ENDDO
    ENDIF
    !
    RETURN
  END FUNCTION COSMPHI
  !
  FUNCTION RPOWERL(L,R) result(rpowerl_rslt)
    !------------------------------------------------------------------------
    !     R^L calculation (L >= 0)
    !
    use chm_kinds
    use number
    implicit none
    real(chm_real) :: rpowerl_rslt
    !
    INTEGER   L,I
    real(chm_real)    R
    !
    RPOWERL_RSLT=ONE
    DO I=1,L
       RPOWERL_RSLT=RPOWERL_RSLT*R
    ENDDO
    !
    RETURN
  END FUNCTION RPOWERL
  !
  FUNCTION FACTORI(N) result(factori_rslt)
    !------------------------------------------------------------------------
    !     N! calculation
    !
    use chm_kinds
    use number
    implicit none
    real(chm_real) :: factori_rslt
    !
    INTEGER   N,I
    !
    IF(N.EQ.0.or.N.EQ.1) THEN
       FACTORI_RSLT=ONE
    ELSE
       FACTORI_RSLT=ONE
       DO I=2,N
          FACTORI_RSLT=FACTORI_RSLT*I
       ENDDO
    ENDIF
    !
    RETURN
  END FUNCTION FACTORI
  !
  SUBROUTINE INITGRID(NCLX,NCLY,NCLZ,GRIDPAR,CONST)
    !-----------------------------------------------------------------------
    !     initialize a grid paramter
    !
    use chm_kinds
    implicit none
    INTEGER NCLX,NCLY,NCLZ
    REAL(CHM_REAL4)  GRIDPAR(*)
    real(chm_real)  CONST
    ! local
    INTEGER I,NC3
    !
    NC3=NCLx*NCLy*NCLz
    DO I=1,NC3
       GRIDPAR(I)=CONST
    ENDDO
    !
    RETURN
  END SUBROUTINE INITGRID
  !
  FUNCTION LPOL(NPOL,COOR,SCAL) result(lpol_rslt)
    !------------------------------------------------------------------------
    !     Legnedre Polynomials
    !
    use chm_kinds
    use number
    implicit none
    real(chm_real) :: lpol_rslt
    !
    INTEGER   NPOL,N
    real(chm_real)    X,COOR,SCAL
    real(chm_real)    LPOLNP,LPOLNPP
    real(chm_real)    X2,X3,X4,X5,X6,X7,X8,X9
    !
    X=SCAL*COOR
    !
    IF(NPOL.LE.4) THEN
       IF(NPOL.EQ.0) THEN
          LPOL_RSLT = ONE
       ELSEIF(NPOL.EQ.1) THEN
          LPOL_RSLT = X
       ELSEIF(NPOL.EQ.2) THEN
          LPOL_RSLT = HALF*(THREE*X*X-ONE)
       ELSEIF(NPOL.EQ.3) THEN
          LPOL_RSLT = HALF*(FIVE*X*X*X-THREE*X)
       ELSEIF(NPOL.EQ.4) THEN
          X2=X*X
          LPOL_RSLT = PT125*(35.*X2*X2-30.*X2+3.)
       ENDIF
    ELSEIF(NPOL.GT.9) THEN
       X2=X*X
       X3=X2*X
       X4=X2*X2
       X5=X3*X2
       X6=X4*X2
       X7=X5*X2
       LPOLNPP= (1./128.)*(6435.*X6*X2-12012.*X6+6930.*X4- &
            1260.*X2+35.)
       LPOLNP = (1./128.)*(12155.*X7*X2-25740.*X7+18018.*X5- &
            4620.*X3+315.*X)
       DO N=10,NPOL
          LPOL_RSLT=((2.*n-1)*X*LPOLNP-(N-1.)*LPOLNPP)/N
          LPOLNPP=LPOLNP
          LPOLNP=LPOL_RSLT
       ENDDO
    ELSE
       IF(NPOL.EQ.5) THEN
          X2=X*X
          X3=X2*X
          LPOL_RSLT = PT125*(63.*X3*X2-70.*X3+15.*X)
       ELSEIF(NPOL.EQ.6) THEN
          X2=X*X
          X4=X2*X2
          LPOL_RSLT = (1./16.)*(231.*X4*X2-315.*X4+105.*X2-5.)
       ELSEIF(NPOL.EQ.7) THEN
          X2=X*X
          X3=X2*X
          X5=X3*X2
          LPOL_RSLT = (1./16.)*(429.*X5*X2-693.*X5+315.*X3-35.*X)
       ELSEIF(NPOL.EQ.8) THEN
          X2=X*X
          X4=X2*X2
          X6=X4*X2
          LPOL_RSLT = (1./128.)*(6435.*X6*X2-12012.*X6+6930.*X4- &
               1260.*X2+35.)
       ELSEIF(NPOL.EQ.9) THEN
          X2=X*X
          X3=X2*X
          X5=X3*X2
          X7=X5*X2
          LPOL_RSLT = (1./128.)*(12155.*X7*X2-25740.*X7+18018.*X5- &
               4620.*X3+315.*X)
       ENDIF
    ENDIF
    !
    RETURN
  END FUNCTION LPOL
  !
  FUNCTION DLPOL(NPOL,COOR,SCAL) result(dlpol_rslt)
    !------------------------------------------------------------------------
    !     Derivatives of Legendre Polymonials
    !
    use chm_kinds
    use number
    implicit none
    real(chm_real) :: dlpol_rslt
    !
    INTEGER   NPOL,N
    real(chm_real)    X,COOR,SCAL,DLPOLNP
    real(chm_real)    X2,X3,X4,X5,X6,X7,X8,X9
    !
    X=SCAL*COOR
    !
    IF(NPOL.LE.4) THEN
       IF(NPOL.EQ.0) THEN
          DLPOL_RSLT = ZERO
       ELSEIF(NPOL.EQ.1) THEN
          DLPOL_RSLT = ONE
       ELSEIF(NPOL.EQ.2) THEN
          DLPOL_RSLT = THREE*X
       ELSEIF(NPOL.EQ.3) THEN
          DLPOL_RSLT = HALF*THREE*(FIVE*X*X-ONE)
       ELSEIF(NPOL.EQ.4) THEN
          DLPOL_RSLT = HALF*FIVE*(SEVEN*X*X*X-THREE*X)
       ENDIF
    ELSEIF(NPOL.GT.9) THEN
       X2=X*X
       X4=X2*X2
       X6=X4*X2
       DLPOLNP=(45./128.)*(2431.*X6*X2-4004.*X6+2002.*X4- &
            308.*X2+7.)
       DO N=10,NPOL
          DLPOL_RSLT=X*DLPOLNP+N*LPOL(N-1,COOR,SCAL)
          DLPOLNP=DLPOL_RSLT
       ENDDO
    ELSE
       IF(NPOL.EQ.5) THEN
          X2=X*X
          DLPOL_RSLT = PT125*15.*(21.*X2*X2-14.*X2+ONE)
       ELSEIF(NPOL.EQ.6) THEN
          X2=X*X
          X3=X2*X
          DLPOL_RSLT = PT125*21.0*(33.*X3*X2-30.*X3+5.*X)
       ELSEIF(NPOL.EQ.7) THEN
          X2=X*X
          X4=X2*X2
          DLPOL_RSLT = (7./16.)*(429.*X4*X2-495.*X4+135.*X2-5.)
       ELSEIF(NPOL.EQ.8) THEN
          X2=X*X
          X3=X2*X
          X5=X3*X2
          DLPOL_RSLT = (9./16.)*(715.*X5*X2-1001.*X5+385.*X3-35.*X)
       ELSEIF(NPOL.EQ.9) THEN
          X2=X*X
          X4=X2*X2
          X6=X4*X2
          DLPOL_RSLT = (45./128.)*(2431.*X6*X2-4004.*X6+2002.*X4- &
               308.*X2+7.)
       ENDIF
    ENDIF
    !
    DLPOL_RSLT=SCAL*DLPOL_RSLT
    RETURN
  END FUNCTION DLPOL

#if KEY_GCMC==1 /*gcmc0*/
  SUBROUTINE REC_CAVP(ECAVITY,DX,DY,DZ,X,Y,Z,GCMCON, &
       NTSSBP,LSTSSBP, &
       RBXMIN,RBXMAX,RBYMIN,RBYMAX, &
       RBZMIN,RBZMAX,DELTA)
    use chm_kinds
    use stream
    use dimens_fcm
    use number
    implicit none
    real(chm_real)  ECAVITY
    INTEGER I,J,NALIVE
    real(chm_real)  X(*),Y(*),Z(*),DX(*),DY(*),DZ(*)
    real(chm_real)  XARGX,XARGY,XARGZ,DPOTX,DPOTY,DPOTZ
    LOGICAL GCMCON(:)
    INTEGER NTSSBP,LSTSSBP(*)
    real(chm_real)  RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX
    real(chm_real)  XMIN,XMAX,YMIN,YMAX,ZMIN,ZMAX
    real(chm_real)  DELTA, DROFFX, DROFFY, DROFFZ
    real(chm_real), PARAMETER :: FORCE=0.2, P1=2.25

    ECAVITY = ZERO

    XMIN = RBXMIN + DELTA
    YMIN = RBYMIN + DELTA
    ZMIN = RBZMIN + DELTA
    XMAX = RBXMAX - DELTA
    YMAX = RBYMAX - DELTA
    ZMAX = RBZMAX - DELTA
    DO I = 1,NTSSBP
       J = LSTSSBP(I) 
       IF(GCMCON(J))THEN
          NALIVE = NALIVE + 1
          DPOTX = 0.0
          DPOTY = 0.0
          DPOTZ = 0.0
          IF (X(J).LE.XMIN) THEN
             DROFFX = X(J) - XMIN
             ECAVITY=ECAVITY+FORCE*DROFFX**2*(DROFFX**2-P1)
             DPOTX = FORCE*(FOUR*DROFFX**3-TWO*P1*DROFFX)
          ELSEIF (X(J).GE.XMAX) THEN
             DROFFX = X(J) - XMAX
             ECAVITY=ECAVITY+FORCE*DROFFX**2*(DROFFX**2-P1)
             DPOTX = FORCE*(FOUR*DROFFX**3-TWO*P1*DROFFX)
          ENDIF
          IF (Y(J).LE.YMIN) THEN
             DROFFY = Y(J) - YMIN
             ECAVITY=ECAVITY+FORCE*DROFFY**2*(DROFFY**2-P1)
             DPOTY = FORCE*(FOUR*DROFFY**3-TWO*P1*DROFFY)
          ELSEIF (Y(J).GE.YMAX) THEN
             DROFFY = Y(J) - YMAX
             ECAVITY=ECAVITY+FORCE*DROFFY**2*(DROFFY**2-P1)
             DPOTY = FORCE*(FOUR*DROFFY**3-TWO*P1*DROFFY)
          ENDIF
          IF (Z(J).LE.ZMIN) THEN
             DROFFZ = Z(J) - ZMIN
             ECAVITY=ECAVITY+FORCE*DROFFZ**2*(DROFFZ**2-P1)
             DPOTZ = FORCE*(FOUR*DROFFZ**3-TWO*P1*DROFFZ)
          ELSEIF (Z(J).GE.ZMAX) THEN
             DROFFZ = Z(J) - ZMAX
             ECAVITY=ECAVITY+FORCE*DROFFZ**2*(DROFFZ**2-P1)
             DPOTZ = FORCE*(FOUR*DROFFZ**3-TWO*P1*DROFFZ)
          ENDIF
          IF((X(J).GT.RBXMAX).OR.(X(J).LT.RBXMIN) &
               .OR.(Y(J).GT.RBYMAX).OR.(Y(J).LT.RBYMIN) &
               .OR.(Z(J).GT.RBZMAX).OR.(Z(J).LT.RBZMIN))THEN    !addition by Benoit Roux
             CALL WRNDIE(-5,'<GSBP>','CHARGE OUTSIDE INNER GSBP REGION')
          ENDIF

          DX(J) = DX(J) + DPOTX
          DY(J) = DY(J) + DPOTY
          DZ(J) = DZ(J) + DPOTZ
       ENDIF
    ENDDO

    RETURN
  END SUBROUTINE REC_CAVP

  !-------------------------------------------------------------------
  SUBROUTINE SPH_CAVP(ECAVITY,DX,DY,DZ,X,Y,Z,GCMCON, &
       NTSSBP,LSTSSBP,RRXCEN,RRYCEN,RRZCEN,        & !benoit
       SRDIST,RMAX,ACAV,BCAV)                                 !benoit
    use chm_kinds
    use stream
    use dimens_fcm
    use number
    use ssbpm, only: compot
    implicit none
    real(chm_real)  ECAVITY
    INTEGER I,J,NALIVE
    real(chm_real)  X(*),Y(*),Z(*),DX(*),DY(*),DZ(*)
    real(chm_real)  XX, YY, ZZ
    real(chm_real)  SHIFT,XARG,POT2,DPOT2,RADIST
    LOGICAL GCMCON(:)
    INTEGER NTSSBP,LSTSSBP(*)  !benoit
    real(chm_real)  RRXCEN, RRYCEN, RRZCEN
    real(chm_real)  SRDIST, RMAX, ACAV(*), BCAV(*)
    real(chm_real)  rr, REP, DREP
    real(chm_real), parameter ::  A=581923.367, B=595.014434, EPSI=0.1521

    ECAVITY = ZERO

    CALL GARA(RMAX,SHIFT,ACAV)

    DO I = 1,NTSSBP
       J = LSTSSBP(I) 
       IF(GCMCON(J))THEN
          NALIVE = NALIVE + 1
          XX = (X(J) - RRXCEN)
          YY = (Y(J) - RRYCEN)
          ZZ = (Z(J) - RRZCEN)
          RADIST = XX**2+YY**2+ZZ**2
          RADIST = SQRT(RADIST)
          XARG = -RMAX + RADIST
          CALL COMPOT(XARG,POT2,DPOT2,BCAV)
          ECAVITY=ECAVITY+POT2+SHIFT
          IF (RADIST .GT. RMAX+0.95) THEN
             XARG = SRDIST+1.7-RADIST
             DPOT2 = DPOT2 + 12.0*A/XARG**13 - 6.0*B/XARG**7
             ECAVITY=ECAVITY+A/XARG**12 - B/XARG**6 + EPSI
          ENDIF
          IF(RADIST.LT.SRDIST)THEN             !addition by Benoit Roux
             rr = (RADIST-SRDIST)
             REP =       ACAV(8)/(rr**2+ACAV(9))**6
             DREP=-12*rr*ACAV(8)/(rr**2+ACAV(9))**7
             ECAVITY = ECAVITY + REP
             DPOT2   = DPOT2   + DREP
          ELSE
             write(*,111) J, X(J), Y(J), Z(J), RADIST 
111          FORMAT(' "Atom: "', I5, 4(F8.3))
             write(*,112) RRXCEN, RRYCEN, RRZCEN 
112          FORMAT(' "Center: "',3(F12.7))
             CALL WRNDIE(-5,'<GSBP>','CHARGE OUTSIDE INNER GSBP REGION')
          ENDIF

          IF(RADIST.GT.RSMALL)THEN
             DX(J) = DX(J) + DPOT2*XX/RADIST
             DY(J) = DY(J) + DPOT2*YY/RADIST
             DZ(J) = DZ(J) + DPOT2*ZZ/RADIST
          ENDIF
       ENDIF
    ENDDO
    
    RETURN
  END SUBROUTINE SPH_CAVP

  SUBROUTINE GARA(XARG,SHIFT,ACAV)
    !-----------------------------------------------------------------------
    !     Calculation of the Function-axis shift of the approximated
    !     cavity potential
    use chm_kinds
    use stream
    implicit none
    real(chm_real) RAMA,XARG,SHIFT
    real(chm_real) XARG2,XARG3
    real(chm_real) ACAV(*)

    !a0   ACAV(5)=-1.6649500
    !a0   ACAV(5)=-1.6649500D0 + 8.391D0   
    !a0      ! fix the depth of the cavity potential for GCMC
    !a1   ACAV(1)=0.56198800
    !a2   ACAV(2)=-0.072798148
    !a3   ACAV(3)=0.00426122036
    !a4   ACAV(4)=-0.0000925233817
    !ac   ACAV(6)=0.0840
    !Xc   ACAV(7)=15.39333

    RAMA=XARG+2.6D0
    XARG2=XARG**2
    XARG3=XARG2*XARG
    IF(XARG.GT.ACAV(7))THEN
       SHIFT=ACAV(6) + 8.391D0
    ELSE
       SHIFT=ACAV(5)+ACAV(1)*XARG+ACAV(2)*XARG2+ACAV(3)*XARG3+ &
            ACAV(4)*XARG*XARG3
    ENDIF
    RETURN
  END SUBROUTINE GARA

  !      SUBROUTINE STGSBC0(NATOM,ISLCT,NTSSBP,LSTSSBP)
  !-----------------------------------------------------------------------
  !     Definition of the selection array LSTSSBP(NTSSBP)
  !      for the cavity potential of GSBP
  !     input NATOM ISLCT JSLCT
  !     output LSTSSBP,NTSSBP
  !     LSTSSBP- numbers of selected atoms
  !     NTSSBP- total number of selected atoms
  !
  !  use chm_kinds
  !  use stream
  !  use parallel
  !      implicit none
  ! Input variables
  !      INTEGER NATOM, ISLCT(*)
  ! Output variables
  !      INTEGER NTSSBP, LSTSSBP(*)
  ! Local variables
  !      INTEGER I
  !      NTSSBP=0
  !      DO I=1,NATOM
  !        IF(ISLCT(I).EQ.1)THEN
  !        NTSSBP=NTSSBP+1
  !        LSTSSBP(NTSSBP)=I
  !        ENDIF
  !      ENDDO
  !      RETURN
  !      END SUBROUTINE STGSBC0
#endif /* (gcmc0)*/
  ! pbeq.src inserted here

  SUBROUTINE PROL_GSBP1(NATOM,X,Y,Z,CG, &
       LNCEL,LTRAN,LDCEL,LXBCEN,LYBCEN,LZBCEN,QPRIN)
    !-----------------------------------------------------------------------
    !     The reaction field energy and forces of the region of interest
    !     are calculated using the following basis functions :
    !     basis funcition (l,m) = spheroidal harmonics (l,m)
    !
    use chm_kinds
    use memory
    use number
    use stream
    use dimens_fcm
    use consta
    use timerm
    use machutil,only:wrttim
    implicit none
    !
    INTEGER NATOM
    real(chm_real)  X(*),Y(*),Z(*),CG(*)
    LOGICAL QPRIN
    INTEGER LNCEL
    real(chm_real)  LTRAN,LDCEL,LXBCEN,LYBCEN,LZBCEN
    ! local
    real(chm_real),allocatable,dimension(:,:) :: SIJ,SIJEVEC
    real(chm_real),allocatable,dimension(:) :: SIJEVAL,ETA,ZETA,CPHI,SPHI
    integer,allocatable,dimension(:) :: IIP0
    real(chm_real4),allocatable,dimension(:) :: ILCHC,IPHIB
    INTEGER NORDER,OLDNTP
    INTEGER NFIL,NFILZ,IXCEN,IYCEN,IZCEN,JX1,JX2,JY1,JY2,JZ1,JZ2
    INTEGER NPROI,NPGD
    INTEGER LNCEL3,BNCEL

    ! save basis functions (the same as the spherical case)
    OLDNTP= OLDNMP*OLDNMP
    call SPHEPOL(NMPOL,LSTPL,LSTPM,LSTPOL)
    ! calculate the normalization constants of the basis functions
    call PROL_NORM(NMPOL,BNORM,MAJOR,MINOR)

    IF(QMMIJ.AND.OLDNTP.EQ.NTPOL) RETURN

    ! setup the overlap matrix (SIJ) and calculate its inverse
    call chmalloc('pbeq.src','PROL_GSBP1','SIJ',NTPOL,NTPOL,crl=SIJ)
    call chmalloc('pbeq.src','PROL_GSBP1','SIJEVEC',NTPOL,NTPOL,crl=SIJEVEC)
    call chmalloc('pbeq.src','PROL_GSBP1','SIJEVAL',NTPOL,crl=SIJEVAL)
    CALL PROL_SIJ(NTPOL,LSTPL,LSTPM,BNORM, &
         MAJOR,MINOR,TOLSIJ, &
         SIJ,SIJEVEC,SIJEVAL)
    IF(QMIJ.AND.OLDNTP.EQ.NTPOL) THEN

       CALL COMP_MMIJ(NTPOL,MMIJ,MIJ,SIJ)
       call chmdealloc('pbeq.src','PROL_GSBP1','SIJ',NTPOL,NTPOL,crl=SIJ)
       call chmdealloc('pbeq.src','PROL_GSBP1','SIJEVEC',NTPOL,NTPOL,crl=SIJEVEC)
       call chmdealloc('pbeq.src','PROL_GSBP1','SIJEVAL',NTPOL,crl=SIJEVAL)
       QA=.false.
       QB=.false.
       RETURN
    ENDIF
    ! calculate prolate spheroidal coordiantes of each grid points
    ! inside the inner region
    NFIL =INT(MINOR/DCEL)+2
    NFILZ=INT(MAJOR/DCEL)+2
    IXCEN=INT((RRXCEN+TRANX-XBCEN)/DCEL)+1
    IYCEN=INT((RRYCEN+TRANY-YBCEN)/DCEL)+1
    IZCEN=INT((RRZCEN+TRANZ-ZBCEN)/DCEL)+1
    JX1=IXCEN-NFIL
    JX2=IXCEN+NFIL+1
    JY1=IYCEN-NFIL
    JY2=IYCEN+NFIL+1
    JZ1=IZCEN-NFILZ
    JZ2=IZCEN+NFILZ+1
    NPROI =(JX2-JX1)*(JY2-JY1)*(JZ2-JZ1)
    call chmalloc('pbeq.src','PROL_GSBP1','IIP0',NPROI,intg=IIP0)
    call chmalloc('pbeq.src','PROL_GSBP1','ETA',NPROI,crl=ETA)
    call chmalloc('pbeq.src','PROL_GSBP1','ZETA',NPROI,crl=ZETA)
    call chmalloc('pbeq.src','PROL_GSBP1','CPHI',NPROI,crl=CPHI)
    call chmalloc('pbeq.src','PROL_GSBP1','SPHI',NPROI,crl=SPHI)
    CALL PROL_TRIG(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,DCEL, &
         XBCEN,YBCEN,ZBCEN, &
         NPGD,IIP0,ETA,ZETA,CPHI,SPHI, &
         MAJOR,MINOR,RRXCEN,RRYCEN,RRZCEN,JX1,JX2,JY1,JY2,JZ1,JZ2)

    ! skip the initialization of CHC array and
    !      the all charge distribution calculations in MAYER
    QA=.true.
    QB=.true.

    ! allocate charge densities in the large box and
    !          boundary potentials in the original box
    LNCEL3=LNCEL*LNCEL*LNCEL
    call chmalloc('pbeq.src','PROL_GSBP1','ILCHC',LNCEL3,cr4=ILCHC)
    BNCEL=1
    IF(QFOCUS)  BNCEL=2*(NCLX*NCLX+NCLY*NCLY+NCLZ*NCLZ)
    call chmalloc('pbeq.src','PROL_GSBP1','IPHIB',BNCEL,cr4=IPHIB)

    DO NORDER=OLDNTP+1,NTPOL

       IF(PRNLEV.GT.7.AND.NORDER.EQ.1) WRITE(OUTU,'(A)')
       IF(PRNLEV.GT.7) WRITE(OUTU,'(3x,a,i5,a)') &
            'Reaction field calculation for',NORDER,' basis function'

       ! replace ICHC (CDEN) array with the basis functions (in original box)
       CALL PROL_CHC(NORDER,NCEL3,DCEL,NPGD,IIP0, &
            ETA,ZETA,CPHI,SPHI,MAJOR,MINOR, &
            LSTPL,LSTPM,BNORM,ICHC,CGSCAL)

       ! get charge distribution ILCHC in the large box from ICHC
       CALL INITGRID(LNCEL,LNCEL,LNCEL,ILCHC,ZERO)
       CALL MAYER_CD_TRIL(NCEL3,LSTPRP,X,Y,Z,CG,ILCHC, &
            KAPPA2, &
            LNCEL,LNCEL,LNCEL,LDCEL,LTRAN,LTRAN,LTRAN, &
            LXBCEN,LYBCEN,LZBCEN, &
            RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
            RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
            MAJOR,MINOR,QPROLATE, &
            ICHC,NCLX,NCLY,NCLZ,DCEL,XBCEN,YBCEN,ZBCEN, &
            QPBC,QA,QB)
       IF(TIMER.GT.1)CALL WRTTIM('Charge Distribution in large box:')

       !------------------------------------------------------------------------
       !     In vacuum (large box - focussing - original box)
       !------------------------------------------------------------------------

       IF(QFOCUS) THEN
          ! get the grid parameters in the lagre box
          CALL MAYER(NTPRP,LSTPRP,ONE,ONE,ZERO,X,Y,Z,CG, &
               PBRAD,LNCEL,LNCEL,LNCEL,LDCEL,WATR,IONR,MAY, &
               MAYX,MAYY,MAYZ,SWIN,ILCHC, &
               IPHIP,ZERO,TMEMB,ZMEMB,ONE,NIMGB, &
               HTMEMB,ONE,HDENS,POSALP,NEGALP,QCHRG,POFFST,NOFFST, &
               DROPLET,ONE,XDROPLET,YDROPLET,ZDROPLET,Qdtom,Qdkap, &
               ONE,LXMAX,LYMAX,LZMAX,LXMIN,LYMIN,LZMIN,Qbtom,Qbkap, &
               ONE,Xcyln,Ycyln,Zcyln,Rcyln,Hcyln,Qctom,Qckap, &
               ONE,Xcone,Ycone,Zcone,Ax,By,Cz,Qectom,Qeckap, &
               -1,-1,&
               LTRAN,LTRAN,LTRAN,LXBCEN,LYBCEN,LZBCEN, &
               IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
               QKEEP,QREEN,QSMTH,QBSPL,QA,QB, &
               RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
               RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
               MAJOR,MINOR,QPROLATE, &
               .false.,.true.,.false.,IPHIB,QPBC,QNPBC, &
                                !              QINTBP QZERO
               QNONLINEAR,QPARTLINEAR,QFKAP, &
               IPOX,IPOY,IPOZ,MAPT, NATOM)

          ! get boundary potentials in the large box: CG -> ICHC
          IF(.NOT.QINTBP.AND..NOT.QPBC.AND..NOT.QZERO)THEN
             CALL MAYER_BP_ALL(LNCEL3,LSTPRP,X,Y,Z,CG,ONE,ZERO, &
                  TMEMB,ILCHC,IPHIP,LNCEL,LNCEL,LNCEL,LDCEL, &
                  LTRAN,LTRAN,LTRAN,LXBCEN,LYBCEN,LZBCEN, &
                  RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
                  RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
                  MAJOR,MINOR,QPROLATE, &
                  LNCEL,LNCEL,LNCEL,LDCEL,LXBCEN,LYBCEN,LZBCEN, &
                  NIMGB,QNPBC,QA,QB)
          ELSEIF(QINTBP) THEN
             CALL MAYER_BP_INT(LNCEL3,LSTPRP,X,Y,Z,CG,ONE,ZERO, &
                  TMEMB,ILCHC,IPHIP,LNCEL,LNCEL,LNCEL,LDCEL, &
                  LTRAN,LTRAN,LTRAN,LXBCEN,LYBCEN,LZBCEN, &
                  RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
                  RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
                  MAJOR,MINOR,QPROLATE, &
                  LNCEL,LNCEL,LNCEL,LDCEL,LXBCEN,LYBCEN,LZBCEN, &
                  NIMGB,QNPBC,QA,QB)
          ENDIF
          IF(TIMER.GT.1) &
               CALL WRTTIM('Boundary potential in large box in vacuum:')

          IF(EPSP.ne.ONE) THEN
             ! vacuum environment in region B.
             CALL PROL_MAYER(LNCEL,LNCEL,LNCEL,LTRAN,LTRAN,LTRAN,LDCEL, &
                  LXBCEN,LYBCEN,LZBCEN, &
                  MAY,MAYX,MAYY,MAYZ, &
                  RRXCEN,RRYCEN,RRZCEN,MAJOR,MINOR)

             ! redefine the maximum region outside which EPS is ONE. (see subroutine MAYER)
             IXMAX=LNCEL-1
             IYMAX=LNCEL-1
             IZMAX=LNCEL-1
             IXMIN=2
             IYMIN=2
             IZMIN=2
          ENDIF

          ! solve PBEQ in the large box
          CALL PBEQ1(MAXITS,DOME,DEPS,ONE,EPSP,ONE,ZERO, &
               LNCEL,LNCEL,LNCEL,IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
               IPHIP,MAYX,MAYY,MAYZ, &
               ILCHC,MAY,TMEMB, &
               .false.,.false.,QPBC,QNPBC,.false.)
          IF(TIMER.GT.1) &
               CALL WRTTIM('PBEQ solver in large box in vacuum:')

          ! construct boundary potentials from the large box
          CALL BOUNDPOT(LNCEL,LNCEL,LNCEL,LDCEL, &
               LTRAN,LTRAN,LTRAN,LXBCEN,LYBCEN,LZBCEN, &
               NCLX,NCLY,NCLZ,DCEL,XBCEN,YBCEN,ZBCEN, &
               IPHIP,IPHIB)

          ! get the grid parameters in the original box
          CALL MAYER(NTPRP,LSTPRP,ONE,ONE,ZERO,X,Y,Z,CG, &
               PBRAD,NCLX,NCLY,NCLZ,DCEL,WATR,IONR,MAY, &
               MAYX,MAYY,MAYZ,SWIN,ICHC, &
               IPHIP,ZERO,TMEMB,ZMEMB,ONE,NIMGB, &
               HTMEMB,ONE,HDENS,POSALP,NEGALP,QCHRG,POFFST,NOFFST, &
               DROPLET,ONE,XDROPLET,YDROPLET,ZDROPLET,Qdtom,Qdkap, &
               ONE,LXMAX,LYMAX,LZMAX,LXMIN,LYMIN,LZMIN,Qbtom,Qbkap, &
               ONE,Xcyln,Ycyln,Zcyln,Rcyln,Hcyln,Qctom,Qckap, &
               ONE,Xcone,Ycone,Zcone,Ax,By,Cz,Qectom,Qeckap, &
               -1,-1,&
               TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
               IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
               QKEEP,QREEN,QSMTH,QBSPL,QA,QB, &
               RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
               RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
               MAJOR,MINOR,QPROLATE, &
               .false.,.false.,.true.,IPHIB,QPBC,QNPBC, &
                                !              QINTBP QZERO
               QNONLINEAR,QPARTLINEAR,QFKAP, &
               IPOX,IPOY,IPOZ,MAPT, NATOM) 

          IF(EPSP.ne.ONE) THEN
             ! vacuum environment in region B.
             CALL PROL_MAYER(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,DCEL, &
                  XBCEN,YBCEN,ZBCEN, &
                  MAY,MAYX,MAYY,MAYZ, &
                  RRXCEN,RRYCEN,RRZCEN,MAJOR,MINOR)
          ENDIF

          ! solve PBEQ in the original box
          CALL PBEQ1(MAXITS,DOME,DEPS,ONE,EPSP,ONE,ZERO, &
               NCLX,NCLY,NCLZ,IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
               IPHIP,MAYX,MAYY,MAYZ, &
               ICHC,MAY,TMEMB, &
               .false.,QFOCUS,QPBC,QNPBC,.false.)
          IF(TIMER.GT.1) CALL WRTTIM('PBEQ solver in vacuum:')

          !      CALL TMPWRIGD(71,NCLX,NCLY,NCLZ,DCEL,TRANX,TRANY,TRANZ,
          !     $     XBCEN,YBCEN,ZBCEN,-100.D0,0.D0,0.D0,100.D0,0.D0,0.D0,
          !     $     IPHIP,1.0E0)

       ELSE

          ! get the grid parameters in the original box
          CALL MAYER(NTPRP,LSTPRP,ONE,ONE,ZERO,X,Y,Z,CG, &
               PBRAD,NCLX,NCLY,NCLZ,DCEL,WATR,IONR,MAY, &
               MAYX,MAYY,MAYZ,SWIN,ICHC, &
               IPHIP,ZERO,TMEMB,ZMEMB,ONE,NIMGB, &
               HTMEMB,ONE,HDENS,POSALP,NEGALP,QCHRG,POFFST,NOFFST, &
               DROPLET,ONE,XDROPLET,YDROPLET,ZDROPLET,Qdtom,Qdkap, &
               ONE,LXMAX,LYMAX,LZMAX,LXMIN,LYMIN,LZMIN,Qbtom,Qbkap, &
               ONE,Xcyln,Ycyln,Zcyln,Rcyln,Hcyln,Qctom,Qckap, &
               ONE,Xcone,Ycone,Zcone,Ax,By,Cz,Qectom,Qeckap, &
               -1,-1,&
               TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
               IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
               QKEEP,QREEN,QSMTH,QBSPL,QA,QB, &
               RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
               RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
               MAJOR,MINOR,QPROLATE, &
               .false.,.true.,.false.,IPHIB,QPBC,QNPBC, &
                                !              QINTBP QZERO
               QNONLINEAR,QPARTLINEAR,QFKAP, &
               IPOX,IPOY,IPOZ,MAPT, NATOM)

          IF(.NOT.QINTBP.AND..NOT.QPBC.AND..NOT.QZERO)THEN
             CALL MAYER_BP_ALL(LNCEL3,LSTPRP,X,Y,Z,CG,ONE,ZERO, &
                  TMEMB,ILCHC,IPHIP,NCLX,NCLY,NCLZ,DCEL, &
                  TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
                  RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
                  RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
                  MAJOR,MINOR,QPROLATE, &
                  LNCEL,LNCEL,LNCEL,LDCEL,LXBCEN,LYBCEN,LZBCEN, &
                  NIMGB,QNPBC,QA,QB)
          ELSEIF(QINTBP) THEN
             CALL MAYER_BP_INT(LNCEL3,LSTPRP,X,Y,Z,CG,ONE,ZERO, &
                  TMEMB,ILCHC,IPHIP,NCLX,NCLY,NCLZ,DCEL, &
                  TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
                  RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
                  RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
                  MAJOR,MINOR,QPROLATE, &
                  LNCEL,LNCEL,LNCEL,LDCEL,LXBCEN,LYBCEN,LZBCEN, &
                  NIMGB,QNPBC,QA,QB)
          ENDIF
          IF(TIMER.GT.1) &
               CALL WRTTIM('Boundary potential with large box in vacuum:')

          IF(EPSP.ne.ONE) THEN
             ! vacuum environment in region B.
             CALL PROL_MAYER(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,DCEL, &
                  XBCEN,YBCEN,ZBCEN, &
                  MAY,MAYX,MAYY,MAYZ, &
                  RRXCEN,RRYCEN,RRZCEN,MAJOR,MINOR)
          ENDIF

          ! solve PBEQ in the original box
          CALL PBEQ1(MAXITS,DOME,DEPS,ONE,EPSP,ONE,ZERO, &
               NCLX,NCLY,NCLZ,IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
               IPHIP,MAYX,MAYY,MAYZ, &
               ICHC,MAY,TMEMB, &
               .false.,QFOCUS,QPBC,QNPBC,.false.)
          IF(TIMER.GT.1) CALL WRTTIM('PBEQ solver in vacuum:')

       ENDIF

       !------------------------------------------------------------------------
       !     In solution (large box - focussing - original box)
       !------------------------------------------------------------------------

       IF(QFOCUS) THEN
          ! get grid parameters in the large box
          CALL MAYER(NTPRP,LSTPRP,EPSW,EPSP,KAPPA2,X,Y,Z,CG, &
               PBRAD,LNCEL,LNCEL,LNCEL,LDCEL,WATR,IONR,MAY, &
               MAYX,MAYY,MAYZ,SWIN,ILCHC, &
               IPHIW,VMEMB,TMEMB,ZMEMB,EPSM,NIMGB, &
               HTMEMB,EPSH,HDENS,POSALP,NEGALP,QCHRG,POFFST,NOFFST, &
               DROPLET,EPSD,XDROPLET,YDROPLET,ZDROPLET,Qdtom,Qdkap, &
               EPSB,LXMAX,LYMAX,LZMAX,LXMIN,LYMIN,LZMIN,Qbtom,Qbkap, &
               EPSC,Xcyln,Ycyln,Zcyln,Rcyln,Hcyln,Qctom,Qckap, &
               EPSEC,Xcone,Ycone,Zcone,Ax,By,Cz,Qectom,Qeckap, &
               -1,-1,&
               LTRAN,LTRAN,LTRAN,LXBCEN,LYBCEN,LZBCEN, &
               IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
               QKEEP,QREEN,QSMTH,QBSPL,QA,QB, &
               RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
               RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
               MAJOR,MINOR,QPROLATE, &
               .false.,.true.,.false.,IPHIB,QPBC,QNPBC, &
                                !              QINTBP QZERO
               QNONLINEAR,QPARTLINEAR,QFKAP, &
               IPOX,IPOY,IPOZ,MAPT, NATOM)

          ! get boundary potentials in the large box: CG -> ICHC
          IF(.NOT.QINTBP.AND..NOT.QPBC.AND..NOT.QZERO)THEN
             CALL MAYER_BP_ALL(LNCEL3,LSTPRP,X,Y,Z,CG,EPSW,KAPPA2, &
                  TMEMB,ILCHC,IPHIW,LNCEL,LNCEL,LNCEL,LDCEL, &
                  LTRAN,LTRAN,LTRAN,LXBCEN,LYBCEN,LZBCEN, &
                  RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
                  RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
                  MAJOR,MINOR,QPROLATE, &
                  LNCEL,LNCEL,LNCEL,LDCEL,LXBCEN,LYBCEN,LZBCEN, &
                  NIMGB,QNPBC,QA,QB)
          ELSEIF(QINTBP) THEN
             CALL MAYER_BP_INT(LNCEL3,LSTPRP,X,Y,Z,CG,EPSW,KAPPA2, &
                  TMEMB,ILCHC,IPHIW,LNCEL,LNCEL,LNCEL,LDCEL, &
                  LTRAN,LTRAN,LTRAN,LXBCEN,LYBCEN,LZBCEN, &
                  RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
                  RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
                  MAJOR,MINOR,QPROLATE, &
                  LNCEL,LNCEL,LNCEL,LDCEL,LXBCEN,LYBCEN,LZBCEN, &
                  NIMGB,QNPBC,QA,QB)
          ENDIF
          IF(TIMER.GT.1) &
               CALL WRTTIM('Boundary potential in large box in solvent:')

          ! vacuum environment in region B.
          CALL PROL_MAYER(LNCEL,LNCEL,LNCEL,LTRAN,LTRAN,LTRAN,LDCEL, &
               LXBCEN,LYBCEN,LZBCEN, &
               MAY,MAYX,MAYY,MAYZ, &
               RRXCEN,RRYCEN,RRZCEN,MAJOR,MINOR)

          ! redefine the maximum region outside which EPS is EPSW. (see subroutine MAYER)
          IXMAX=LNCEL-1
          IYMAX=LNCEL-1
          IZMAX=LNCEL-1
          IXMIN=2
          IYMIN=2
          IZMIN=2

          ! solve PBEQ in the large box
          CALL PBEQ1(MAXITS,DOME,DEPS,EPSW,EPSP,EPSM,KAPPA2, &
               LNCEL,LNCEL,LNCEL,IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
               IPHIW,MAYX,MAYY,MAYZ, &
               ILCHC,MAY,TMEMB, &
               QOSOR,.false.,QPBC,QNPBC,.false.)
          IF(TIMER.GT.1) &
               CALL WRTTIM('PBEQ solver in large box in solvent:')

          !      CALL TMPWRIGD(70,LNCEL,LNCEL,LNCEL,LDCEL,LTRAN,LTRAN,LTRAN,
          !     $     LXBCEN,LYBCEN,LZBCEN,-100.D0,0.D0,0.D0,100.D0,0.D0,0.D0,
          !     $     IPHIW,1.0E0)

          ! construct boundary potentials from the large box
          CALL BOUNDPOT(LNCEL,LNCEL,LNCEL,LDCEL, &
               LTRAN,LTRAN,LTRAN,LXBCEN,LYBCEN,LZBCEN, &
               NCLX,NCLY,NCLZ,DCEL,XBCEN,YBCEN,ZBCEN, &
               IPHIW,IPHIB)

          ! get the grid parameters in the original box
          CALL MAYER(NTPRP,LSTPRP,EPSW,EPSP,KAPPA2,X,Y,Z,CG, &
               PBRAD,NCLX,NCLY,NCLZ,DCEL,WATR,IONR,MAY, &
               MAYX,MAYY,MAYZ,SWIN,ICHC, &
               IPHIW,VMEMB,TMEMB,ZMEMB,EPSM,NIMGB, &
               HTMEMB,EPSH,HDENS,POSALP,NEGALP,QCHRG,POFFST,NOFFST, &
               DROPLET,EPSD,XDROPLET,YDROPLET,ZDROPLET,Qdtom,Qdkap, &
               EPSB,LXMAX,LYMAX,LZMAX,LXMIN,LYMIN,LZMIN,Qbtom,Qbkap, &
               EPSC,Xcyln,Ycyln,Zcyln,Rcyln,Hcyln,Qctom,Qckap, &
               EPSEC,Xcone,Ycone,Zcone,Ax,By,Cz,Qectom,Qeckap, &
               -1,-1,&
               TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
               IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
               QKEEP,QREEN,QSMTH,QBSPL,QA,QB, &
               RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
               RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
               MAJOR,MINOR,QPROLATE, &
               .false.,.false.,.true.,IPHIB,QPBC,QNPBC, &
                                !              QINTBP QZERO
               QNONLINEAR,QPARTLINEAR,QFKAP, &
               IPOX,IPOY,IPOZ,MAPT, NATOM)
          ! vacuum environment in region B.
          CALL PROL_MAYER(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,DCEL, &
               XBCEN,YBCEN,ZBCEN, &
               MAY,MAYX,MAYY,MAYZ, &
               RRXCEN,RRYCEN,RRZCEN,MAJOR,MINOR)

          ! redefine the maximum region outside which EPS is EPSW. (see subroutine MAYER)
          IF(JX2.GT.IXMAX) IXMAX=JX2+1
          IF(JY2.GT.IYMAX) IYMAX=JY2+1
          IF(JZ2.GT.IZMAX) IZMAX=JZ2+1
          IF(JX1.LT.IXMIN) IXMIN=JX1-1
          IF(JY1.LT.IYMIN) IYMIN=JY1-1
          IF(JZ1.LT.IZMIN) IZMIN=JZ1-1

          ! solve PBEQ in the original box
          CALL PBEQ1(MAXITS,DOME,DEPS,EPSW,EPSP,EPSM,KAPPA2, &
               NCLX,NCLY,NCLZ,IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
               IPHIW,MAYX,MAYY,MAYZ, &
               ICHC,MAY,TMEMB, &
               QOSOR,QFOCUS,QPBC,QNPBC,.false.)
          IF(TIMER.GT.1) CALL WRTTIM('PBEQ solver in solvent:')

          !      CALL TMPWRIGD(72,NCLX,NCLY,NCLZ,DCEL,TRANX,TRANY,TRANZ,
          !     $     XBCEN,YBCEN,ZBCEN,-100.D0,0.D0,0.D0,100.D0,0.D0,0.D0,
          !     $     IPHIW,1.0E0)

       ELSE

          ! get the grid parameters in the original box
          CALL MAYER(NTPRP,LSTPRP,EPSW,EPSP,KAPPA2,X,Y,Z,CG, &
               PBRAD,NCLX,NCLY,NCLZ,DCEL,WATR,IONR,MAY, &
               MAYX,MAYY,MAYZ,SWIN,ICHC, &
               IPHIW,VMEMB,TMEMB,ZMEMB,EPSM,NIMGB, &
               HTMEMB,EPSH,HDENS,POSALP,NEGALP,QCHRG,POFFST,NOFFST, &
               DROPLET,EPSD,XDROPLET,YDROPLET,ZDROPLET,Qdtom,Qdkap, &
               EPSB,LXMAX,LYMAX,LZMAX,LXMIN,LYMIN,LZMIN,Qbtom,Qbkap, &
               EPSC,Xcyln,Ycyln,Zcyln,Rcyln,Hcyln,Qctom,Qckap, &
               EPSEC,Xcone,Ycone,Zcone,Ax,By,Cz,Qectom,Qeckap, &
               -1,-1,&
               TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
               IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
               QKEEP,QREEN,QSMTH,QBSPL,QA,QB, &
               RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
               RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
               MAJOR,MINOR,QPROLATE, &
               .false.,.true.,.false.,IPHIB,QPBC,QNPBC, &
                                !              QINTBP QZERO
               QNONLINEAR,QPARTLINEAR,QFKAP, &
               IPOX,IPOY,IPOZ,MAPT, NATOM)

          IF(.NOT.QINTBP.AND..NOT.QPBC.AND..NOT.QZERO)THEN
             CALL MAYER_BP_ALL(LNCEL3,LSTPRP,X,Y,Z,CG,EPSW,KAPPA2, &
                  TMEMB,ILCHC,IPHIW,NCLX,NCLY,NCLZ,DCEL, &
                  TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
                  RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
                  RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
                  MAJOR,MINOR,QPROLATE, &
                  LNCEL,LNCEL,LNCEL,LDCEL,LXBCEN,LYBCEN,LZBCEN, &
                  NIMGB,QNPBC,QA,QB)
          ELSEIF(QINTBP) THEN
             CALL MAYER_BP_INT(LNCEL3,LSTPRP,X,Y,Z,CG,EPSW,KAPPA2, &
                  TMEMB,ILCHC,IPHIW,NCLX,NCLY,NCLZ,DCEL, &
                  TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
                  RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
                  RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
                  MAJOR,MINOR,QPROLATE, &
                  LNCEL,LNCEL,LNCEL,LDCEL,LXBCEN,LYBCEN,LZBCEN, &
                  NIMGB,QNPBC,QA,QB)
          ENDIF
          IF(TIMER.GT.1) &
               CALL WRTTIM('Boundary potential with large box in solvent:')

          ! vacuum environment in region B.
          CALL PROL_MAYER(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,DCEL, &
               XBCEN,YBCEN,ZBCEN, &
               MAY,MAYX,MAYY,MAYZ, &
               RRXCEN,RRYCEN,RRZCEN,MAJOR,MINOR)

          ! redefine the maximum region outside which EPS is EPSW. (see subroutine MAYER)
          IF(JX2.GT.IXMAX) IXMAX=JX2+1
          IF(JY2.GT.IYMAX) IYMAX=JY2+1
          IF(JZ2.GT.IZMAX) IZMAX=JZ2+1
          IF(JX1.LT.IXMIN) IXMIN=JX1-1
          IF(JY1.LT.IYMIN) IYMIN=JY1-1
          IF(JZ1.LT.IZMIN) IZMIN=JZ1-1

          ! solve PBEQ in the original box
          CALL PBEQ1(MAXITS,DOME,DEPS,EPSW,EPSP,EPSM,KAPPA2, &
               NCLX,NCLY,NCLZ,IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
               IPHIW,MAYX,MAYY,MAYZ, &
               ICHC,MAY,TMEMB, &
               QOSOR,QFOCUS,QPBC,QNPBC,.false.)
          IF(TIMER.GT.1) CALL WRTTIM('PBEQ solver in solvent:')

       ENDIF

       ! calculate the reaction field energy elements M(ij)
       CALL PROL_MIJ(NTPOL,NORDER,DCEL,NPGD,IIP0, &
            ETA,ZETA,CPHI,SPHI,MAJOR,MINOR, &
            LSTPL,LSTPM,BNORM, &
            IPHIP,IPHIW,MIJ,CGSCAL)

    ENDDO

    CALL COMP_MMIJ(NTPOL,MMIJ,MIJ,SIJ)
    call chmdealloc('pbeq.src','PROL_GSBP1','ILCHC',LNCEL3,cr4=ILCHC)
    call chmdealloc('pbeq.src','PROL_GSBP1','IPHIB',BNCEL,cr4=IPHIB)
    call chmdealloc('pbeq.src','PROL_GSBP1','IIP0',NPROI,intg=IIP0)
    call chmdealloc('pbeq.src','PROL_GSBP1','ETA',NPROI,crl=ETA)
    call chmdealloc('pbeq.src','PROL_GSBP1','ZETA',NPROI,crl=ZETA)
    call chmdealloc('pbeq.src','PROL_GSBP1','CPHI',NPROI,crl=CPHI)
    call chmdealloc('pbeq.src','PROL_GSBP1','SPHI',NPROI,crl=SPHI)
    call chmdealloc('pbeq.src','PROL_GSBP1','SIJ',NTPOL,NTPOL,crl=SIJ)
    call chmdealloc('pbeq.src','PROL_GSBP1','SIJEVEC',NTPOL,NTPOL,crl=SIJEVEC)
    call chmdealloc('pbeq.src','PROL_GSBP1','SIJEVAL',NTPOL,crl=SIJEVAL)
    QA=.false.
    QB=.false.
    !
    RETURN
  END SUBROUTINE PROL_GSBP1
  !
  SUBROUTINE PROL_GSBP2(NATOM,X,Y,Z,CG,QPRIN)
    !-----------------------------------------------------------------------
    !     The reaction field energy and forces of the region of interest (NTRB)
    !     are calculated using the following basis functions :
    !     basis funcition (l,m) = spheoridal harmonics (l,m)
    !
    use chm_kinds
    use memory
    use number
    use stream
    use dimens_fcm
    use consta
    use timerm
    use machutil,only:wrttim
    implicit none
    !
    INTEGER NATOM
    real(chm_real)  X(*),Y(*),Z(*),CG(*)
    LOGICAL QPRIN
    ! local
    real(chm_real),allocatable,dimension(:,:) :: SIJ,SIJEVEC
    real(chm_real),allocatable,dimension(:)   :: SIJEVAL,ETA,ZETA,CPHI,SPHI
    integer,allocatable,dimension(:) :: IIP0
    real(chm_real4)  :: IPHIB(1) !dummy
    INTEGER NORDER,OLDNTP
    INTEGER NFIL,NFILZ,IXCEN,IYCEN,IZCEN,JX1,JX2,JY1,JY2,JZ1,JZ2
    INTEGER NPROI,NPGD

    ! save basis functions (the same as the spherical case)
    OLDNTP= OLDNMP*OLDNMP
    CALL SPHEPOL(NMPOL,LSTPL,LSTPM,LSTPOL)

    ! calculate the normalization constants of the basis functions
    CALL PROL_NORM(NMPOL,BNORM,MAJOR,MINOR)

    IF(QMMIJ.AND.OLDNTP.EQ.NTPOL) RETURN

    ! setup the overlap matrix (SIJ) and calculate its inverse
    call chmalloc('pbeq.src','PROL_GSBP1','SIJ',NTPOL,NTPOL,crl=SIJ)
    call chmalloc('pbeq.src','PROL_GSBP1','SIJEVEC',NTPOL,NTPOL,crl=SIJEVEC)
    call chmalloc('pbeq.src','PROL_GSBP1','SIJEVAL',NTPOL,crl=SIJEVAL)
    CALL PROL_SIJ(NTPOL,LSTPL,LSTPM,BNORM, &
         MAJOR,MINOR,TOLSIJ, &
         SIJ,SIJEVEC,SIJEVAL)
    IF(QMIJ.AND.OLDNTP.EQ.NTPOL) THEN

       CALL COMP_MMIJ(NTPOL,MMIJ,MIJ,SIJ)
       call chmdealloc('pbeq.src','PROL_GSBP1','SIJ',NTPOL,NTPOL,crl=SIJ)
       call chmdealloc('pbeq.src','PROL_GSBP1','SIJEVEC',NTPOL,NTPOL,crl=SIJEVEC)
       call chmdealloc('pbeq.src','PROL_GSBP1','SIJEVAL',NTPOL,crl=SIJEVAL)
       QA=.false.
       QB=.false.
       RETURN
    ENDIF

    ! calculate prolate spheroidal coordiantes of each grid points
    ! inside the inner region
    NFIL =INT(MINOR/DCEL)+2
    NFILZ=INT(MAJOR/DCEL)+2
    IXCEN=INT((RRXCEN+TRANX-XBCEN)/DCEL)+1
    IYCEN=INT((RRYCEN+TRANY-YBCEN)/DCEL)+1
    IZCEN=INT((RRZCEN+TRANZ-ZBCEN)/DCEL)+1
    JX1=IXCEN-NFIL
    JX2=IXCEN+NFIL+1
    JY1=IYCEN-NFIL
    JY2=IYCEN+NFIL+1
    JZ1=IZCEN-NFILZ
    JZ2=IZCEN+NFILZ+1
    NPROI =(JX2-JX1)*(JY2-JY1)*(JZ2-JZ1)
    call chmalloc('pbeq.src','PROL_GSBP1','IIP0',NPROI,intg=IIP0)
    call chmalloc('pbeq.src','PROL_GSBP1','ETA',NPROI,crl=ETA)
    call chmalloc('pbeq.src','PROL_GSBP1','ZETA',NPROI,crl=ZETA)
    call chmalloc('pbeq.src','PROL_GSBP1','CPHI',NPROI,crl=CPHI)
    call chmalloc('pbeq.src','PROL_GSBP1','SPHI',NPROI,crl=SPHI)
    CALL PROL_TRIG(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,DCEL, &
         XBCEN,YBCEN,ZBCEN, &
         NPGD,IIP0,ETA,ZETA,CPHI,SPHI, &
         MAJOR,MINOR,RRXCEN,RRYCEN,RRZCEN,JX1,JX2,JY1,JY2,JZ1,JZ2)

    ! skip the initialization of CHC array and
    !      the all charge distribution calculations in MAYER
    QA=.true.
    QB=.true.

    DO NORDER=OLDNTP+1,NTPOL

       IF(PRNLEV.GT.7.AND.NORDER.EQ.1) WRITE(OUTU,'(A)')
       IF(PRNLEV.GT.7) WRITE(OUTU,'(3x,a,i5,a)') &
            'Reaction field calculation for',NORDER,' basis function'

       ! replace ICHC (CDEN) array with the basis functions (spheroidal harmonics)
       CALL PROL_CHC(NORDER,NCEL3,DCEL,NPGD,IIP0, &
            ETA,ZETA,CPHI,SPHI,MAJOR,MINOR, &
            LSTPL,LSTPM,BNORM,ICHC,CGSCAL)

       !------------------------------------------------------------------------
       !     In vacuum
       !------------------------------------------------------------------------

       ! get the grid parameters
       CALL MAYER(NTPRP,LSTPRP,ONE,ONE,ZERO,X,Y,Z,CG, &
            PBRAD,NCLX,NCLY,NCLZ,DCEL,WATR,IONR,MAY, &
            MAYX,MAYY,MAYZ,SWIN,ICHC, &
            IPHIP,ZERO,TMEMB,ZMEMB,ONE,NIMGB, &
            HTMEMB,ONE,HDENS,POSALP,NEGALP,QCHRG,POFFST,NOFFST, &
            DROPLET,ONE,XDROPLET,YDROPLET,ZDROPLET,Qdtom,Qdkap, &
            ONE,LXMAX,LYMAX,LZMAX,LXMIN,LYMIN,LZMIN,Qbtom,Qbkap, &
            ONE,Xcyln,Ycyln,Zcyln,Rcyln,Hcyln,Qctom,Qckap, &
            ONE,Xcone,Ycone,Zcone,Ax,By,Cz,Qectom,Qeckap, &
               -1,-1,&
            TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
            IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
            QKEEP,QREEN,QSMTH,QBSPL,QA,QB, &
            RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
            RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
            MAJOR,MINOR,QPROLATE, &
            .false.,.true.,.false.,IPHIB,QPBC,QNPBC, &
                                !              QINTBP QZERO
            QNONLINEAR,QPARTLINEAR,QFKAP, &
            IPOX,IPOY,IPOZ,MAPT, NATOM)
       ! get boundary potentials : CG -> ICHC
       IF(.NOT.QINTBP.AND..NOT.QPBC.AND..NOT.QZERO)THEN
          CALL MAYER_BP_ALL(NCEL3,LSTPRP,X,Y,Z,CG,ONE,ZERO, &
               TMEMB,ICHC,IPHIP,NCLX,NCLY,NCLZ,DCEL, &
               TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
               RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
               RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
               MAJOR,MINOR,QPROLATE, &
               NCLX,NCLY,NCLZ,DCEL,XBCEN,YBCEN,ZBCEN, &
               NIMGB,QNPBC,QA,QB)
       ELSEIF(QINTBP) THEN
          CALL MAYER_BP_INT(NCEL3,LSTPRP,X,Y,Z,CG,ONE,ZERO, &
               TMEMB,ICHC,IPHIP,NCLX,NCLY,NCLZ,DCEL, &
               TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
               RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
               RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
               MAJOR,MINOR,QPROLATE, &
               NCLX,NCLY,NCLZ,DCEL,XBCEN,YBCEN,ZBCEN, &
               NIMGB,QNPBC,QA,QB)
       ENDIF
       IF(TIMER.GT.1) CALL WRTTIM('Boundary potential in vacuum:')

       IF(EPSP.ne.ONE) THEN
          ! vacuum environment in region B.
          CALL PROL_MAYER(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,DCEL, &
               XBCEN,YBCEN,ZBCEN, &
               MAY,MAYX,MAYY,MAYZ, &
               RRXCEN,RRYCEN,RRZCEN,MAJOR,MINOR)
       ENDIF

       ! solve PBEQ
       CALL PBEQ1(MAXITS,DOME,DEPS,ONE,EPSP,ONE,ZERO, &
            NCLX,NCLY,NCLZ,IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
            IPHIP,MAYX,MAYY,MAYZ, &
            ICHC,MAY,TMEMB, &
            .false.,.false.,QPBC,QNPBC,.false.)
       IF(TIMER.GT.1) CALL WRTTIM('PBEQ solver in vacuum:')

       !------------------------------------------------------------------------
       !     In solution
       !------------------------------------------------------------------------

       ! get grid parameters
       CALL MAYER(NTPRP,LSTPRP,EPSW,EPSP,KAPPA2,X,Y,Z,CG, &
            PBRAD,NCLX,NCLY,NCLZ,DCEL,WATR,IONR,MAY, &
            MAYX,MAYY,MAYZ,SWIN,ICHC, &
            IPHIW,VMEMB,TMEMB,ZMEMB,EPSM,NIMGB, &
            HTMEMB,EPSH,HDENS,POSALP,NEGALP,QCHRG,POFFST,NOFFST, &
            DROPLET,EPSD,XDROPLET,YDROPLET,ZDROPLET,Qdtom,Qdkap, &
            EPSB,LXMAX,LYMAX,LZMAX,LXMIN,LYMIN,LZMIN,Qbtom,Qbkap, &
            EPSC,Xcyln,Ycyln,Zcyln,Rcyln,Hcyln,Qctom,Qckap, &
            EPSEC,Xcone,Ycone,Zcone,Ax,By,Cz,Qectom,Qeckap, &
               -1,-1,&
            TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
            IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
            QKEEP,QREEN,QSMTH,QBSPL,QA,QB, &
            RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
            RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
            MAJOR,MINOR,QPROLATE, &
            .false.,.true.,.false.,IPHIB,QPBC,QNPBC, &
                                !              QINTBP QZERO
            QNONLINEAR,QPARTLINEAR,QFKAP, &
            IPOX,IPOY,IPOZ,MAPT, NATOM)

       ! get boundary potentials : CG -> ICHC
       IF(.NOT.QINTBP.AND..NOT.QPBC.AND..NOT.QZERO)THEN
          CALL MAYER_BP_ALL(NCEL3,LSTPRP,X,Y,Z,CG,EPSW,KAPPA2, &
               TMEMB,ICHC,IPHIW,NCLX,NCLY,NCLZ,DCEL, &
               TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
               RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
               RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
               MAJOR,MINOR,QPROLATE, &
               NCLX,NCLY,NCLZ,DCEL,XBCEN,YBCEN,ZBCEN, &
               NIMGB,QNPBC,QA,QB)
       ELSEIF(QINTBP) THEN
          CALL MAYER_BP_INT(NCEL3,LSTPRP,X,Y,Z,CG,EPSW,KAPPA2, &
               TMEMB,ICHC,IPHIW,NCLX,NCLY,NCLZ,DCEL, &
               TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
               RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
               RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
               MAJOR,MINOR,QPROLATE, &
               NCLX,NCLY,NCLZ,DCEL,XBCEN,YBCEN,ZBCEN, &
               NIMGB,QNPBC,QA,QB)
       ENDIF
       IF(TIMER.GT.1) CALL WRTTIM('Boundary potential in solvent:')

       ! vacuum environment in region B.
       CALL PROL_MAYER(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,DCEL, &
            XBCEN,YBCEN,ZBCEN, &
            MAY,MAYX,MAYY,MAYZ, &
            RRXCEN,RRYCEN,RRZCEN,MAJOR,MINOR)

       ! redefine the maximum region outside which EPS is EPSW. (see subroutine MAYER)
       IF(JX2.GT.IXMAX) IXMAX=JX2+1
       IF(JY2.GT.IYMAX) IYMAX=JY2+1
       IF(JZ2.GT.IZMAX) IZMAX=JZ2+1
       IF(JX1.LT.IXMIN) IXMIN=JX1-1
       IF(JY1.LT.IYMIN) IYMIN=JY1-1
       IF(JZ1.LT.IZMIN) IZMIN=JZ1-1

       ! solve PBEQ
       CALL PBEQ1(MAXITS,DOME,DEPS,EPSW,EPSP,EPSM,KAPPA2, &
            NCLX,NCLY,NCLZ,IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
            IPHIW,MAYX,MAYY,MAYZ, &
            ICHC,MAY,TMEMB, &
            QOSOR,.false.,QPBC,QNPBC,.false.)
       IF(TIMER.GT.1) CALL WRTTIM('PBEQ solver in solvent:')

       !      CALL TMPWRIGD(73,NCLX,NCLY,NCLZ,DCEL,TRANX,TRANY,TRANZ,
       !     $     XBCEN,YBCEN,ZBCEN,-100.D0,0.D0,0.D0,100.D0,0.D0,0.D0,
       !     $     IPHIP,1.0E0)
       !      CALL TMPWRIGD(74,NCLX,NCLY,NCLZ,DCEL,TRANX,TRANY,TRANZ,
       !     $     XBCEN,YBCEN,ZBCEN,-100.D0,0.D0,0.D0,100.D0,0.D0,0.D0,
       !     $     IPHIW,1.0E0)

       ! calculate the reaction field energy elements M(ij)
       CALL PROL_MIJ(NTPOL,NORDER,DCEL,NPGD,IIP0, &
            ETA,ZETA,CPHI,SPHI,MAJOR,MINOR, &
            LSTPL,LSTPM,BNORM, &
            IPHIP,IPHIW,MIJ,CGSCAL)

    ENDDO

    CALL COMP_MMIJ(NTPOL,MMIJ,MIJ,SIJ)
    call chmdealloc('pbeq.src','PROL_GSBP1','IIP0',NPROI,intg=IIP0)
    call chmdealloc('pbeq.src','PROL_GSBP1','ETA',NPROI,crl=ETA)
    call chmdealloc('pbeq.src','PROL_GSBP1','ZETA',NPROI,crl=ZETA)
    call chmdealloc('pbeq.src','PROL_GSBP1','CPHI',NPROI,crl=CPHI)
    call chmdealloc('pbeq.src','PROL_GSBP1','SPHI',NPROI,crl=SPHI)
    call chmdealloc('pbeq.src','PROL_GSBP1','SIJ',NTPOL,NTPOL,crl=SIJ)
    call chmdealloc('pbeq.src','PROL_GSBP1','SIJEVEC',NTPOL,NTPOL,crl=SIJEVEC)
    call chmdealloc('pbeq.src','PROL_GSBP1','SIJEVAL',NTPOL,crl=SIJEVAL)
    QA=.false.
    QB=.false.
    !
    RETURN
  END SUBROUTINE PROL_GSBP2
  !
  SUBROUTINE RFSPHE_GSBP1(NATOM,X,Y,Z,CG, &
       LNCEL,LTRAN,LDCEL,LXBCEN,LYBCEN,LZBCEN,QPRIN)
    !-----------------------------------------------------------------------
    !     The reaction field energy and forces of the region of interest
    !     are calculated using the following basis functions :
    !     basis funcition (l,m) = radial function (l) * spherical harmonics (l,m)
    !
    use chm_kinds
    use memory
    use number
    use stream
    use dimens_fcm
    use consta
    use timerm
    use machutil,only:wrttim
    implicit none
    !
    INTEGER NATOM
    real(chm_real)  X(*),Y(*),Z(*),CG(*)
    LOGICAL QPRIN
    INTEGER LNCEL
    real(chm_real)  LTRAN,LDCEL,LXBCEN,LYBCEN,LZBCEN
    ! local
    integer,allocatable,dimension(:) :: IIP0
    real(chm_real),allocatable,dimension(:) :: RGD,CTHETA,CPHI,SPHI,SIJEVAL
    real(chm_real),allocatable,dimension(:,:) :: SIJ,SIJEVEC
    real(chm_real4),allocatable,dimension(:) :: ILCHC,IPHIB

    INTEGER NORDER,OLDNTP
    INTEGER NFIL,IXCEN,IYCEN,IZCEN,JX1,JX2,JY1,JY2,JZ1,JZ2
    INTEGER NPROI,NPGD
    INTEGER LNCEL3,BNCEL
    real(chm_real)  DCEL2

    ! save basis functions
    OLDNTP= OLDNMP*OLDNMP
    CALL SPHEPOL(NMPOL,LSTPL,LSTPM,LSTPOL)

    ! calculate the normalization constants of the basis functions
    CALL SPHE_NORM(NMPOL,BNORM,SRDIST)

    IF(QMMIJ.AND.OLDNTP.EQ.NTPOL) RETURN

    ! obtain all space functions (MAY,MAYX,MAYY,MAYZ)
    ! Array MAY will be used as the volume exculsion function for ions
    !
    ! NOTE: Unlike subroutine RFSPHE_GSBP2, when QFOCUS is true,
    !       we have to update the all space functions for each basis function

    IF(QFKAP) GOTO 11

    IF(KAPPA2.EQ.0.0) KAPPA2=ONE
    DCEL2=DCEL*DCEL
    CALL INITGRID(NCLx,NCLy,NCLz,MAY,KAPPA2*DCEL)
    CALL INITGRID(NCLx,NCLy,NCLz,MAYX,EPSW)
    CALL INITGRID(NCLx,NCLy,NCLz,MAYY,EPSW)
    CALL INITGRID(NCLx,NCLy,NCLz,MAYZ,EPSW)

    IF(Tmemb.gt.0.0) THEN
       CALL MAYER_MEMB(NCLX,NCLY,NCLZ,DCEL,TRANX,TRANY,TRANZ, &
            ZBCEN,TMEMB,ZMEMB,EPSM,HTMEMB,EPSH, &
            HDENS,POSALP,NEGALP,QCHRG,POFFST,NOFFST, &
            ICHC,MAY,MAYX,MAYY,MAYZ, &
            QFKAP)
    ENDIF

    IF(Ax.gt.0.0)THEN
       CALL MAYER_ECONE(NCLX,NCLY,NCLZ,DCEL,TRANX,TRANY,TRANZ, &
            XBCEN,YBCEN,ZBCEN,KAPPA2,TMEMB,ZMEMB,EPSM,HTMEMB,EPSH, &
            EPSEC,Xcone,Ycone,Zcone,Ax,By,Cz,Qectom,Qeckap, &
            IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
            ICHC,MAY,MAYX,MAYY,MAYZ, &
            QFKAP)
    ENDIF

    IF(RCYLN.gt.0.0)THEN
       CALL MAYER_CYLN(NCLX,NCLY,NCLZ,DCEL,TRANX,TRANY,TRANZ, &
            XBCEN,YBCEN,ZBCEN,KAPPA2,TMEMB,ZMEMB,EPSM,HTMEMB,EPSH, &
            EPSC,Xcyln,Ycyln,Zcyln,Rcyln,Hcyln,Qctom,Qckap, &
            IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
            ICHC,MAY,MAYX,MAYY,MAYZ, &
            QFKAP)
    ENDIF

    !      IF(LXMAX.ne.LXMIN)THEN
    !         CALL MAYER_BOX(NCLX,NCLY,NCLZ,DCEL,TRANX,TRANY,TRANZ,
    !     $        XBCEN,YBCEN,ZBCEN,KAPPA2,TMEMB,ZMEMB,EPSM,HTMEMB,EPSH,
    !     $        EPSB,LXMAX,LYMAX,LZMAX,LXMIN,LYMIN,LZMIN,Qbtom,Qbkap,
    !     $        IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN,
    !     $        ICHC,MAY,MAYX,MAYY,MAYZ,
    !     $        QFKAP)
    !      ENDIF

    ! here, we just use OUTER REGION array (NTRA, LSTRA) to construct
    ! the dielectric boundary.
    CALL MAYER_MSTEP(NTRA,LSTRA,X,Y,Z,PBRAD, &
         WATR,IONR,EPSP,NCLX,NCLY,NCLZ,DCEL, &
         TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
         MAY,MAYX,MAYY,MAYZ,QREEN,QFKAP)
    IF(QREEN) THEN
       CALL MAYER_REEN(NTRA,LSTRA,X,Y,Z,PBRAD,EPSP,WATR, &
            NCLX,NCLY,NCLZ,DCEL, &
            TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
            MAY,MAYX,MAYY,MAYZ,QFKAP, &
            IPOX,IPOY,IPOZ,MAPT)
    ENDIF

11  CONTINUE
    IF (TIMER.GT.1) &
         CALL WRTTIM('Volume exclusion function preparation times:')

    ! calculate the trigonometric functions of each grid points
    ! inside region of interest
    NFIL=INT(SRDIST/DCEL)+2
    IXCEN=INT((RRXCEN+TRANX-XBCEN)/DCEL)+1
    IYCEN=INT((RRYCEN+TRANY-YBCEN)/DCEL)+1
    IZCEN=INT((RRZCEN+TRANZ-ZBCEN)/DCEL)+1
    JX1=IXCEN-NFIL
    JX2=IXCEN+NFIL+1
    JY1=IYCEN-NFIL
    JY2=IYCEN+NFIL+1
    JZ1=IZCEN-NFIL
    JZ2=IZCEN+NFIL+1
    NPROI =(JX2-JX1)*(JY2-JY1)*(JZ2-JZ1) 
    call chmalloc('pbeq.src','RFSPHE_GSBP1','IIP0',NPROI,intg=IIP0)
    call chmalloc('pbeq.src','RFSPHE_GSBP1','RGD',NPROI,crl=RGD)
    call chmalloc('pbeq.src','RFSPHE_GSBP1','CTHETA',NPROI,crl=CTHETA)
    call chmalloc('pbeq.src','RFSPHE_GSBP1','CPHI',NPROI,crl=CPHI)
    call chmalloc('pbeq.src','RFSPHE_GSBP1','SPHI',NPROI,crl=SPHI)
    CALL RFSPHE_TRIG(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,DCEL, &
         XBCEN,YBCEN,ZBCEN,MAY, &
         IIP0,CTHETA,CPHI,SPHI,RGD, &
         SRDIST,RRXCEN,RRYCEN,RRZCEN,JX1,JX2,JY1,JY2,JZ1,JZ2,NPGD)

    IF(KAPPA2.EQ.ONE) THEN
       KAPPA2=ZERO
       CALL INITGRID(NCLx,NCLy,NCLz,MAY,ZERO)
    ELSE
       ! kappa = zero inside the inner region
       CALL RFSPHE_MAYER(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,DCEL, &
            XBCEN,YBCEN,ZBCEN,MAY, &
            RRXCEN,RRYCEN,RRZCEN,SRDIST)
    ENDIF

    IF(QFKAP) THEN
       QFKAP=.false.
       CALL INITGRID(NCLx,NCLy,NCLz,MAY,ZERO)
    ENDIF

    ! setup the overlap matrix (SIJ) and calculate its inverse
    call chmalloc('pbeq.src','RFSPHE_GSBP1','SIJ',NTPOL,NTPOL,crl=SIJ)
    call chmalloc('pbeq.src','RFSPHE_GSBP1','SIJEVEC',NTPOL,NTPOL,crl=SIJEVEC)
    call chmalloc('pbeq.src','RFSPHE_GSBP1','SIJEVAL',NTPOL,crl=SIJEVAL)
    CALL RFSPHE_SIJ(NPGD,DCEL,NTPOL,TOLSIJ, &
         LSTPL,LSTPM,BNORM, &
         CTHETA,CPHI,SPHI,RGD, &
         SIJ,SIJEVEC,SIJEVAL)
    IF(TIMER.GT.1) CALL WRTTIM('SIJ (overlap) matrix:')

    IF(QMIJ.AND.OLDNTP.EQ.NTPOL) THEN
       CALL COMP_MMIJ(NTPOL,MMIJ,MIJ,SIJ)
       call chmdealloc('pbeq.src','RFSPHE_GSBP1','IIP0',NPROI,intg=IIP0)
       call chmdealloc('pbeq.src','RFSPHE_GSBP1','RGD',NPROI,crl=RGD)
       call chmdealloc('pbeq.src','RFSPHE_GSBP1','CTHETA',NPROI,crl=CTHETA)
       call chmdealloc('pbeq.src','RFSPHE_GSBP1','CPHI',NPROI,crl=CPHI)
       call chmdealloc('pbeq.src','RFSPHE_GSBP1','SPHI',NPROI,crl=SPHI)

       call chmdealloc('pbeq.src','RFSPHE_GSBP1','SIJ',NTPOL,NTPOL,crl=SIJ)
       call chmdealloc('pbeq.src','RFSPHE_GSBP1','SIJEVEC',NTPOL,NTPOL,crl=SIJEVEC)
       call chmdealloc('pbeq.src','RFSPHE_GSBP1','SIJEVAL',NTPOL,crl=SIJEVAL)
       QA=.false.
       QB=.false.
       RETURN
    ENDIF

    ! allocate charge densities in the large box and
    !          boundary potentials in the original box
    LNCEL3=LNCEL*LNCEL*LNCEL
    BNCEL=2*(NCLX*NCLX+NCLY*NCLY+NCLZ*NCLZ)
    call chmalloc('pbeq.src','RFSPHE_GSBP1','ILCHC',LNCEL3,cr4=ILCHC)
    call chmalloc('pbeq.src','RFSPHE_GSBP1','IPHIB',BNCEL,cr4=IPHIB)

    ! skip the initialization of CHC array and
    !      the all charge distribution calculations in MAYER
    QA=.true.
    QB=.true.

    DO NORDER=OLDNTP+1,NTPOL

       IF(PRNLEV.GT.7.AND.NORDER.EQ.1) WRITE(OUTU,'(A)')
       IF(PRNLEV.GT.7) WRITE(OUTU,'(3x,a,i5,a)') &
            'Reaction field calculation for',NORDER,' basis function'

       ! replace ICHC (CDEN) array with the basis functions (in original box)
       CALL SPHE_CHC(NCEL3,NPGD,DCEL,NORDER, &
            IIP0,CTHETA,CPHI,SPHI,RGD, &
            LSTPL,LSTPM,BNORM,ICHC,CGSCAL)

       !         CALL TMPWRIGD(70,NCLX,NCLY,NCLZ,DCEL,TRANX,TRANY,TRANZ,
       !     $        XBCEN,YBCEN,ZBCEN,-200.D0,0.D0,0.D0,200.D0,0.D0,0.D0,
       !     $        ICHC,1.0E0)
       !         CALL TMPWRIGD(71,NCLX,NCLY,NCLZ,DCEL,TRANX,TRANY,TRANZ,
       !     $        XBCEN,YBCEN,ZBCEN,0.D0,-200.D0,0.D0,0.D0,200.D0,0.D0,
       !     $        ICHC,1.0E0)
       !         CALL TMPWRIGD(72,NCLX,NCLY,NCLZ,DCEL,TRANX,TRANY,TRANZ,
       !     $        XBCEN,YBCEN,ZBCEN,0.D0,0.D0,-200.D0,0.D0,0.D0,200.D0,
       !     $        ICHC,1.0E0)

       ! get charge distribution ILCHC in the large box from ICHC
       CALL INITGRID(LNCEL,LNCEL,LNCEL,ILCHC,ZERO)
       CALL MAYER_CD_TRIL(NCEL3,LSTPRP,X,Y,Z,CG,ILCHC, &
            KAPPA2, &
            LNCEL,LNCEL,LNCEL,LDCEL,LTRAN,LTRAN,LTRAN, &
            LXBCEN,LYBCEN,LZBCEN, &
            RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
            RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
            MAJOR,MINOR,QPROLATE, &
            ICHC,NCLX,NCLY,NCLZ,DCEL,XBCEN,YBCEN,ZBCEN, &
            QPBC,QA,QB)
       IF(TIMER.GT.1)CALL WRTTIM('Charge Distribution in large box:')

       !------------------------------------------------------------------------
       !     In solution without membrane ;
       !     KAPPA2 = ZERO
       !     VMEMB  = ZERO
       !     EPSM   = EPSW
       !     EPSH   = EPSW
       !     EPSD   = EPSW
       !     EPSC   = EPSW
       !------------------------------------------------------------------------

       ! initialize the potential array
       CALL INITGRID(NCLx,NCLy,NCLz,IPHIP,ZERO)

       ! get boundary potentials in the large box: CG -> ICHC
       IF(.NOT.QZERO) THEN
          CALL MAYER_BP_INT(LNCEL3,LSTPRP,X,Y,Z,CG,EPSW,ZERO, &
               TMEMB,ILCHC,IPHIP,LNCEL,LNCEL,LNCEL,LDCEL, &
               LTRAN,LTRAN,LTRAN,LXBCEN,LYBCEN,LZBCEN, &
               RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
               RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
               MAJOR,MINOR,QPROLATE, &
               LNCEL,LNCEL,LNCEL,LDCEL,LXBCEN,LYBCEN,LZBCEN, &
               NIMGB,QNPBC,QA,QB)
       ENDIF
       IF(TIMER.GT.1) &
            CALL WRTTIM('Boundary potential in solution (large box):')

       ! solve PBEQ in the large box
       CALL PBEQ5(MAXITS,DOME,DEPS,EPSW,TMEMB,LNCEL,LNCEL,LNCEL, &
            IPHIP,ILCHC,.false.,QPBC,QNPBC)
       !                                     qfocus
       IF(TIMER.GT.1) &
            CALL WRTTIM('PBEQ solver in solution (large box):')

       !      CALL TMPWRIGD(80,LNCEL,LNCEL,LNCEL,LDCEL,LTRAN,LTRAN,LTRAN,
       !     $     LXBCEN,LYBCEN,LZBCEN,-200.D0,0.D0,0.D0,200.D0,0.D0,0.D0,
       !     $     IPHIP,1.0E0)
       !      CALL TMPWRIGD(81,LNCEL,LNCEL,LNCEL,LDCEL,LTRAN,LTRAN,LTRAN,
       !     $     LXBCEN,LYBCEN,LZBCEN,0.D0,0.D0,-200.D0,0.D0,0.D0,200.D0,
       !     $     IPHIP,1.0E0)

       ! construct boundary potentials from the large box
       CALL BOUNDPOT(LNCEL,LNCEL,LNCEL,LDCEL, &
            LTRAN,LTRAN,LTRAN,LXBCEN,LYBCEN,LZBCEN, &
            NCLX,NCLY,NCLZ,DCEL,XBCEN,YBCEN,ZBCEN, &
            IPHIP,IPHIB)

       CALL MAYER_BPFOCUS(IPHIP,IPHIB,NCLX,NCLY,NCLZ)

       ! solve PBEQ in the original box
       CALL PBEQ5(MAXITS,DOME,DEPS,EPSW,TMEMB,NCLX,NCLY,NCLZ, &
            IPHIP,ICHC,QFOCUS,QPBC,QNPBC)
       IF(TIMER.GT.1) &
            CALL WRTTIM('PBEQ solver in solution (original box):')

       !      CALL TMPWRIGD(82,NCLX,NCLY,NCLZ,DCEL,TRANX,TRANY,TRANZ,
       !     $     XBCEN,YBCEN,ZBCEN,-200.D0,0.D0,0.D0,200.D0,0.D0,0.D0,
       !     $     IPHIP,1.0E0)
       !      CALL TMPWRIGD(83,NCLX,NCLY,NCLZ,DCEL,TRANX,TRANY,TRANZ,
       !     $     XBCEN,YBCEN,ZBCEN,0.D0,0.D0,-200.D0,0.D0,0.D0,200.D0,
       !     $     IPHIP,1.0E0)

       !------------------------------------------------------------------------
       !     In solution with membrane (large box - focussing - original box)
       !     VMEMB  = ZERO
       !     KAPPA2
       !     EPSM
       !     EPSH
       !     EPSD
       !     EPSC
       !------------------------------------------------------------------------

       ! get grid parameters in the large box
       CALL MAYER(NTRA,LSTRA,EPSW,EPSP,KAPPA2,X,Y,Z,CG, &
            PBRAD,LNCEL,LNCEL,LNCEL,LDCEL,WATR,IONR,MAY, &
            MAYX,MAYY,MAYZ,SWIN,ILCHC, &
            IPHIW,ZERO,TMEMB,ZMEMB,EPSM,NIMGB, &
            HTMEMB,EPSH,HDENS,POSALP,NEGALP,QCHRG,POFFST,NOFFST, &
            DROPLET,EPSD,XDROPLET,YDROPLET,ZDROPLET,Qdtom,Qdkap, &
            EPSB,LXMAX,LYMAX,LZMAX,LXMIN,LYMIN,LZMIN,Qbtom,Qbkap, &
            EPSC,Xcyln,Ycyln,Zcyln,Rcyln,Hcyln,Qctom,Qckap, &
            EPSEC,Xcone,Ycone,Zcone,Ax,By,Cz,Qectom,Qeckap, &
               -1,-1,&
            LTRAN,LTRAN,LTRAN,LXBCEN,LYBCEN,LZBCEN, &
            IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
            QKEEP,QREEN,QSMTH,QBSPL,QA,QB, &
            RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
            RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
            MAJOR,MINOR,QPROLATE, &
            .false.,.true.,.false.,IPHIB,QPBC,QNPBC, &
                                !              QINTBP QZERO
            QNONLINEAR,QPARTLINEAR,QFKAP, &
            IPOX,IPOY,IPOZ,MAPT, NATOM) 
       IF(TIMER.GT.1) &
            CALL WRTTIM('Space functions with membrane (large box):')

       ! kappa = zero inside the inner region
       IF(KAPPA2.ne.ZERO) THEN
          CALL RFSPHE_MAYER(LNCEL,LNCEL,LNCEL,LTRAN,LTRAN,LTRAN,LDCEL, &
               LXBCEN,LYBCEN,LZBCEN,MAY, &
               RRXCEN,RRYCEN,RRZCEN,SRDIST)
       ENDIF

       ! get boundary potentials in the large box: CG -> ICHC
       IF(.NOT.QZERO)THEN
          CALL MAYER_BP_INT(LNCEL3,LSTPRP,X,Y,Z,CG,EPSW,KAPPA2, &
               TMEMB,ILCHC,IPHIW,LNCEL,LNCEL,LNCEL,LDCEL, &
               LTRAN,LTRAN,LTRAN,LXBCEN,LYBCEN,LZBCEN, &
               RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
               RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
               MAJOR,MINOR,QPROLATE, &
               LNCEL,LNCEL,LNCEL,LDCEL,LXBCEN,LYBCEN,LZBCEN, &
               NIMGB,QNPBC,QA,QB)
       ENDIF
       IF(TIMER.GT.1) &
            CALL WRTTIM('Boundary potential with membrane (large box):')

       ! see subroutine MAYER
       IXMAX=LNCEL-1
       IYMAX=LNCEL-1
       IZMAX=LNCEL-1
       IXMIN=2
       IYMIN=2
       IZMIN=2

       ! solve PBEQ in the large box
       CALL PBEQ1(MAXITS,DOME,DEPS,EPSW,EPSP,EPSM,KAPPA2, &
            LNCEL,LNCEL,LNCEL,IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
            IPHIW,MAYX,MAYY,MAYZ, &
            ILCHC,MAY,TMEMB, &
            QOSOR,.false.,QPBC,QNPBC,.false.)
       IF(TIMER.GT.1) &
            CALL WRTTIM('PBEQ solver with membrane (large box):')

       !      CALL TMPWRIGD(74,LNCEL,LNCEL,LNCEL,LDCEL,LTRAN,LTRAN,LTRAN,
       !     $     LXBCEN,LYBCEN,LZBCEN,-200.D0,0.D0,0.D0,200.D0,0.D0,0.D0,
       !     $     MAY,1.0E0)
       !      CALL TMPWRIGD(75,LNCEL,LNCEL,LNCEL,LDCEL,LTRAN,LTRAN,LTRAN,
       !     $     LXBCEN,LYBCEN,LZBCEN,0.D0,0.D0,-200.D0,0.D0,0.D0,200.D0,
       !     $     MAY,1.0E0)
       !      CALL TMPWRIGD(76,LNCEL,LNCEL,LNCEL,LDCEL,LTRAN,LTRAN,LTRAN,
       !     $     LXBCEN,LYBCEN,LZBCEN,-200.D0,0.D0,0.D0,200.D0,0.D0,0.D0,
       !     $     MAYX,1.0E0)
       !      CALL TMPWRIGD(77,LNCEL,LNCEL,LNCEL,LDCEL,LTRAN,LTRAN,LTRAN,
       !     $     LXBCEN,LYBCEN,LZBCEN,0.D0,0.D0,-200.D0,0.D0,0.D0,200.D0,
       !     $     MAYZ,1.0E0)
       !      CALL TMPWRIGD(84,LNCEL,LNCEL,LNCEL,LDCEL,LTRAN,LTRAN,LTRAN,
       !     $     LXBCEN,LYBCEN,LZBCEN,-200.D0,0.D0,0.D0,200.D0,0.D0,0.D0,
       !     $     IPHIW,1.0E0)
       !      CALL TMPWRIGD(85,LNCEL,LNCEL,LNCEL,LDCEL,LTRAN,LTRAN,LTRAN,
       !     $     LXBCEN,LYBCEN,LZBCEN,0.D0,0.D0,-200.D0,0.D0,0.D0,200.D0,
       !     $     IPHIW,1.0E0)

       ! construct boundary potentials from the large box
       CALL BOUNDPOT(LNCEL,LNCEL,LNCEL,LDCEL, &
            LTRAN,LTRAN,LTRAN,LXBCEN,LYBCEN,LZBCEN, &
            NCLX,NCLY,NCLZ,DCEL,XBCEN,YBCEN,ZBCEN, &
            IPHIW,IPHIB)

       ! get the grid parameters in the original box
       CALL MAYER(NTRA,LSTRA,EPSW,EPSP,KAPPA2,X,Y,Z,CG, &
            PBRAD,NCLX,NCLY,NCLZ,DCEL,WATR,IONR,MAY, &
            MAYX,MAYY,MAYZ,SWIN,ICHC, &
            IPHIW,ZERO,TMEMB,ZMEMB,EPSM,NIMGB, &
            HTMEMB,EPSH,HDENS,POSALP,NEGALP,QCHRG,POFFST,NOFFST, &
            DROPLET,EPSD,XDROPLET,YDROPLET,ZDROPLET,Qdtom,Qdkap, &
            EPSB,LXMAX,LYMAX,LZMAX,LXMIN,LYMIN,LZMIN,Qbtom,Qbkap, &
            EPSC,Xcyln,Ycyln,Zcyln,Rcyln,Hcyln,Qctom,Qckap, &
            EPSEC,Xcone,Ycone,Zcone,Ax,By,Cz,Qectom,Qeckap, &
               -1,-1,&
            TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
            IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
            QKEEP,QREEN,QSMTH,QBSPL,QA,QB, &
            RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
            RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
            MAJOR,MINOR,QPROLATE, &
            .false.,.false.,.true.,IPHIB,QPBC,QNPBC, &
                                !              QINTBP QZERO
            QNONLINEAR,QPARTLINEAR,QFKAP, &
            IPOX,IPOY,IPOZ,MAPT,NATOM)

       IF(TIMER.GT.1) &
            CALL WRTTIM('Space functions with membrane (original box):')

       ! kappa = zero inside the inner region
       IF(KAPPA2.ne.ZERO) THEN
          CALL RFSPHE_MAYER(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,DCEL, &
               XBCEN,YBCEN,ZBCEN,MAY, &
               RRXCEN,RRYCEN,RRZCEN,SRDIST)
       ENDIF

       ! see subroutine MAYER
       IXMAX=NCLX-1
       IYMAX=NCLY-1
       IZMAX=NCLZ-1
       IXMIN=2
       IYMIN=2
       IZMIN=2

       ! solve PBEQ in the original box
       CALL PBEQ1(MAXITS,DOME,DEPS,EPSW,EPSP,EPSM,KAPPA2, &
            NCLX,NCLY,NCLZ,IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
            IPHIW,MAYX,MAYY,MAYZ, &
            ICHC,MAY,TMEMB, &
            QOSOR,QFOCUS,QPBC,QNPBC,.false.)
       IF(TIMER.GT.1) CALL WRTTIM('PBEQ solver with membrane:')

       !         CALL TMPWRIGD(94,NCLX,NCLY,NCLZ,DCEL,TRANX,TRANY,TRANZ,
       !     $        XBCEN,YBCEN,ZBCEN,-200.D0,0.D0,0.D0,200.D0,0.D0,0.D0,
       !     $        MAY,1.0E0)
       !         CALL TMPWRIGD(95,NCLX,NCLY,NCLZ,DCEL,TRANX,TRANY,TRANZ,
       !     $        XBCEN,YBCEN,ZBCEN,0.D0,0.D0,-200.D0,0.D0,0.D0,200.D0,
       !     $        MAY,1.0E0)
       !         CALL TMPWRIGD(96,NCLX,NCLY,NCLZ,DCEL,TRANX,TRANY,TRANZ,
       !     $        XBCEN,YBCEN,ZBCEN,-200.D0,0.D0,0.D0,200.D0,0.D0,0.D0,
       !     $        MAYX,1.0E0)
       !         CALL TMPWRIGD(97,NCLX,NCLY,NCLZ,DCEL,TRANX,TRANY,TRANZ,
       !     $        XBCEN,YBCEN,ZBCEN,0.D0,0.D0,-200.D0,0.D0,0.D0,200.D0,
       !     $        MAYZ,1.0E0)
       !         CALL TMPWRIGD(86,NCLX,NCLY,NCLZ,DCEL,TRANX,TRANY,TRANZ,
       !     $        XBCEN,YBCEN,ZBCEN,-200.D0,0.D0,0.D0,200.D0,0.D0,0.D0,
       !     $        IPHIW,1.0E0)
       !         CALL TMPWRIGD(87,NCLX,NCLY,NCLZ,DCEL,TRANX,TRANY,TRANZ,
       !     $        XBCEN,YBCEN,ZBCEN,0.D0,0.D0,-200.D0,0.D0,0.D0,200.D0,
       !     $        IPHIW,1.0E0)

       ! calculate the reaction field energy elements M(ij)
       CALL SPHE_MIJ(NPGD,DCEL,NTPOL,NORDER, &
            LSTPL,LSTPM,BNORM, &
            IIP0,CTHETA,CPHI,SPHI,RGD, &
            IPHIP,IPHIW,MIJ,CGSCAL)

    ENDDO

    CALL COMP_MMIJ(NTPOL,MMIJ,MIJ,SIJ)
    call chmdealloc('pbeq.src','RFSPHE_GSBP1','IIP0',NPROI,intg=IIP0)
    call chmdealloc('pbeq.src','RFSPHE_GSBP1','RGD',NPROI,crl=RGD)
    call chmdealloc('pbeq.src','RFSPHE_GSBP1','CTHETA',NPROI,crl=CTHETA)
    call chmdealloc('pbeq.src','RFSPHE_GSBP1','CPHI',NPROI,crl=CPHI)
    call chmdealloc('pbeq.src','RFSPHE_GSBP1','SPHI',NPROI,crl=SPHI)

    call chmdealloc('pbeq.src','RFSPHE_GSBP1','SIJ',NTPOL,NTPOL,crl=SIJ)
    call chmdealloc('pbeq.src','RFSPHE_GSBP1','SIJEVEC',NTPOL,NTPOL,crl=SIJEVEC)
    call chmdealloc('pbeq.src','RFSPHE_GSBP1','SIJEVAL',NTPOL,crl=SIJEVAL)
    call chmalloc('pbeq.src','RFSPHE_GSBP1','ILCHC',LNCEL3,cr4=ILCHC)
    call chmalloc('pbeq.src','RFSPHE_GSBP1','IPHIB',BNCEL,cr4=IPHIB)
    QA=.false.
    QB=.false.
    !
    RETURN
  END SUBROUTINE RFSPHE_GSBP1
  !
  SUBROUTINE RFRECT_GSBP1(NATOM,X,Y,Z,CG, &
       LNCEL,LTRAN,LDCEL,LXBCEN,LYBCEN,LZBCEN,QPRIN)
    !-----------------------------------------------------------------------
    !     The reaction field energy and forces of the region of interest
    !     are calculated using the Legendre Polynomials
    !
    use chm_kinds
    use memory
    use number
    use stream
    use dimens_fcm
    use consta
    use timerm
    use machutil,only:wrttim
    implicit none

    INTEGER NATOM
    real(chm_real)  X(*),Y(*),Z(*),CG(*)
    LOGICAL QPRIN
    INTEGER LNCEL
    real(chm_real)  LTRAN,LDCEL,LXBCEN,LYBCEN,LZBCEN
    ! local
    integer,allocatable,dimension(:) :: IIP0
    real(chm_real),allocatable,dimension(:) :: SIJEVAL
    real(chm_real),allocatable,dimension(:,:) :: SIJ,SIJEVEC
    real(chm_real4),allocatable,dimension(:) :: ILCHC,IPHIB
    INTEGER NORDER,OLDNTP
    INTEGER JX1,JX2,JY1,JY2,JZ1,JZ2,NPROI,NPGD
    INTEGER LNCEL3,BNCEL
    real(chm_real)  DCEL2

    ! save basis functions
    OLDNTP=OLDXNP*OLDYNP*OLDZNP
    CALL RECTPOL(XNPOL,YNPOL,ZNPOL,OLDNTP,QMIJ, &
         LSTPX,LSTPY,LSTPZ,LSTPOL)

    ! calculate the normalization constants of the basis functions
    CALL RECT_NORM(NTPOL,XSCALE,YSCALE,ZSCALE, &
         LSTPX,LSTPY,LSTPZ,BNORM)

    IF(QMMIJ.AND.OLDNTP.EQ.NTPOL) RETURN
    ! obtain all space functions (MAY,MAYX,MAYY,MAYZ)
    ! Array MAY will be used as the volume exculsion function for ions
    !

    IF(QFKAP) GOTO 11

    IF(KAPPA2.EQ.0.0) KAPPA2=ONE
    DCEL2=DCEL*DCEL
    CALL INITGRID(NCLx,NCLy,NCLz,MAY,KAPPA2*DCEL)
    CALL INITGRID(NCLx,NCLy,NCLz,MAYX,EPSW)
    CALL INITGRID(NCLx,NCLy,NCLz,MAYY,EPSW)
    CALL INITGRID(NCLx,NCLy,NCLz,MAYZ,EPSW)

    IF(Tmemb.gt.0.0) THEN
       CALL MAYER_MEMB(NCLX,NCLY,NCLZ,DCEL,TRANX,TRANY,TRANZ, &
            ZBCEN,TMEMB,ZMEMB,EPSM,HTMEMB,EPSH, &
            HDENS,POSALP,NEGALP,QCHRG,POFFST,NOFFST, &
            ICHC,MAY,MAYX,MAYY,MAYZ, &
            QFKAP)
    ENDIF

    IF(Ax.gt.0.0)THEN
       CALL MAYER_ECONE(NCLX,NCLY,NCLZ,DCEL,TRANX,TRANY,TRANZ, &
            XBCEN,YBCEN,ZBCEN,KAPPA2,TMEMB,ZMEMB,EPSM,HTMEMB,EPSH, &
            EPSEC,Xcone,Ycone,Zcone,Ax,By,Cz,Qectom,Qeckap, &
            IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
            ICHC,MAY,MAYX,MAYY,MAYZ, &
            QFKAP)
    ENDIF

    IF(RCYLN.gt.0.0)THEN
       CALL MAYER_CYLN(NCLX,NCLY,NCLZ,DCEL,TRANX,TRANY,TRANZ, &
            XBCEN,YBCEN,ZBCEN,KAPPA2,TMEMB,ZMEMB,EPSM,HTMEMB,EPSH, &
            EPSC,Xcyln,Ycyln,Zcyln,Rcyln,Hcyln,Qctom,Qckap, &
            IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
            ICHC,MAY,MAYX,MAYY,MAYZ, &
            QFKAP)
    ENDIF

    !      IF(LXMAX.ne.LXMIN)THEN
    !         CALL MAYER_BOX(NCLX,NCLY,NCLZ,DCEL,TRANX,TRANY,TRANZ,
    !     $        XBCEN,YBCEN,ZBCEN,KAPPA2,TMEMB,ZMEMB,EPSM,HTMEMB,EPSH,
    !     $        EPSB,LXMAX,LYMAX,LZMAX,LXMIN,LYMIN,LZMIN,Qbtom,Qbkap,
    !     $        IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN,
    !     $        ICHC,MAY,MAYX,MAYY,MAYZ,
    !     $        QFKAP)
    !      ENDIF

    ! here, we just use OUTER REGION array (NTRA, LSTRA) to construct
    ! the dielectric boundary.
    CALL MAYER_MSTEP(NTRA,LSTRA,X,Y,Z,PBRAD, &
         WATR,IONR,EPSP,NCLX,NCLY,NCLZ,DCEL, &
         TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
         MAY,MAYX,MAYY,MAYZ,QREEN,QFKAP)
    IF(QREEN) THEN
       CALL MAYER_REEN(NTRA,LSTRA,X,Y,Z,PBRAD,EPSP,WATR, &
            NCLX,NCLY,NCLZ,DCEL, &
            TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
            MAY,MAYX,MAYY,MAYZ,QFKAP, &
            IPOX,IPOY,IPOZ,MAPT)
    ENDIF

11  CONTINUE
    IF (TIMER.GT.1) &
         CALL WRTTIM('Volume exclusion function preparation times:')

    ! find out the grid points inside region of interest
    JX1=INT((TRANX+RBXMIN-XBCEN)/DCEL)-1
    JX2=INT((TRANX+RBXMAX-XBCEN)/DCEL)+3
    JY1=INT((TRANY+RBYMIN-YBCEN)/DCEL)-1
    JY2=INT((TRANY+RBYMAX-YBCEN)/DCEL)+3
    JZ1=INT((TRANZ+RBZMIN-ZBCEN)/DCEL)-1
    JZ2=INT((TRANZ+RBZMAX-ZBCEN)/DCEL)+3
    NPROI=(JX2-JX1)*(JY2-JY1)*(JZ2-JZ1)
    call chmalloc('pbeq.src','RFRECT_GSBP1','IIP0',NPROI,intg=IIP0)

    CALL RFRECT_GPINRR(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,DCEL, &
         XBCEN,YBCEN,ZBCEN,MAY, &
         RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,SRCUT, &
         JX1,JX2,JY1,JY2,JZ1,JZ2,IIP0,NPGD)

    ! re-adjust the kappa array
    IF(KAPPA2.EQ.ONE) THEN
       KAPPA2=ZERO
       CALL INITGRID(NCLx,NCLy,NCLz,MAY,ZERO)
    ELSE
       ! kappa = zero inside the inner region
       IF(SRCUT.GT.0.0) THEN
          CALL RFSPHE_MAYER(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,DCEL, &
               XBCEN,YBCEN,ZBCEN,MAY, &
               RRXCEN,RRYCEN,RRZCEN,SRCUT)
       ELSE
          CALL RFRECT_MAYER(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,DCEL, &
               XBCEN,YBCEN,ZBCEN,MAY, &
               RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX)
       ENDIF
    ENDIF

    IF(QFKAP) THEN
       QFKAP=.false.
       CALL INITGRID(NCLx,NCLy,NCLz,MAY,ZERO)
    ENDIF

    ! setup the overlap matrix (SIJ) and calculate its inverse
    call chmalloc('pbeq.src','RFRECT_GSBP1','SIJ',NTPOL,NTPOL,crl=SIJ)
    call chmalloc('pbeq.src','RFRECT_GSBP1','SIJEVEC',NTPOL,NTPOL,crl=SIJEVEC)
    call chmalloc('pbeq.src','RFRECT_GSBP1','SIJEVAL',NTPOL,crl=SIJEVAL)
    CALL RFRECT_SIJ(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,DCEL, &
         XBCEN,YBCEN,ZBCEN,TOLSIJ, &
         NTPOL,LSTPX,LSTPY,LSTPZ,BNORM, &
         IIP0,NPGD,RRXCEN,RRYCEN,RRZCEN,XSCALE,YSCALE,ZSCALE, &
         SIJ,SIJEVEC,SIJEVAL)
    IF(TIMER.GT.1) CALL WRTTIM('SIJ (overlap) matrix:')

    IF(QMIJ.AND.OLDNTP.EQ.NTPOL) THEN
       CALL COMP_MMIJ(NTPOL,MMIJ,MIJ,SIJ)
       call chmdealloc('pbeq.src','RFRECT_GSBP1','SIJ',NTPOL,NTPOL,crl=SIJ)
       call chmdealloc('pbeq.src','RFRECT_GSBP1','SIJEVEC',NTPOL,NTPOL,crl=SIJEVEC)
       call chmdealloc('pbeq.src','RFRECT_GSBP1','SIJEVAL',NTPOL,crl=SIJEVAL)
       call chmdealloc('pbeq.src','RFRECT_GSBP1','IIP0',NPROI,intg=IIP0)
       QA=.false.
       QB=.false.
       RETURN
    ENDIF

    ! allocate charge densities in the large box and
    !          boundary potentials in the original box
    LNCEL3=LNCEL*LNCEL*LNCEL
    BNCEL=2*(NCLX*NCLX+NCLY*NCLY+NCLZ*NCLZ)
    call chmalloc('pbeq.src','RFRECT_GSBP1','ILCHC',LNCEL3,cr4=ILCHC)
    call chmalloc('pbeq.src','RFRECT_GSBP1','IPHIB',BNCEL,cr4=IPHIB)
    ! skip the initialization of CHC array and
    !      the all charge distribution calculations in MAYER
    QA=.true.
    QB=.true.

    DO NORDER=OLDNTP+1,NTPOL

       IF(PRNLEV.GT.7.AND.NORDER.EQ.1) WRITE(OUTU,'(A)')
       IF(PRNLEV.GT.7) WRITE(OUTU,'(3x,a,i5,a)') &
            'Reaction field calculation for',NORDER,' basis function'

       ! replace ICHC (CDEN) array with the basis functions (in original box)
       CALL RECT_CHC(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,DCEL, &
            XBCEN,YBCEN,ZBCEN, &
            NORDER,LSTPX,LSTPY,LSTPZ,BNORM, &
            ICHC,CGSCAL,IIP0,NPGD, &
            RRXCEN,RRYCEN,RRZCEN,XSCALE,YSCALE,ZSCALE)

       !         CALL TMPWRIGD(70,NCLX,NCLY,NCLZ,DCEL,TRANX,TRANY,TRANZ,
       !     $        XBCEN,YBCEN,ZBCEN,-200.D0,0.D0,0.D0,200.D0,0.D0,0.D0,
       !     $        ICHC,1.0E0)
       !         CALL TMPWRIGD(71,NCLX,NCLY,NCLZ,DCEL,TRANX,TRANY,TRANZ,
       !     $        XBCEN,YBCEN,ZBCEN,0.D0,-200.D0,0.D0,0.D0,200.D0,0.D0,
       !     $        ICHC,1.0E0)
       !         CALL TMPWRIGD(72,NCLX,NCLY,NCLZ,DCEL,TRANX,TRANY,TRANZ,
       !     $        XBCEN,YBCEN,ZBCEN,0.D0,0.D0,-200.D0,0.D0,0.D0,200.D0,
       !     $        ICHC,1.0E0)

       ! get charge distribution ILCHC in the large box from ICHC
       CALL INITGRID(LNCEL,LNCEL,LNCEL,ILCHC,ZERO)
       CALL MAYER_CD_TRIL(NCEL3,LSTPRP,X,Y,Z,CG,ILCHC, &
            KAPPA2, &
            LNCEL,LNCEL,LNCEL,LDCEL,LTRAN,LTRAN,LTRAN, &
            LXBCEN,LYBCEN,LZBCEN, &
            RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
            RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
            MAJOR,MINOR,QPROLATE, &
            ICHC,NCLX,NCLY,NCLZ,DCEL,XBCEN,YBCEN,ZBCEN, &
            QPBC,QA,QB)
       IF(TIMER.GT.1)CALL WRTTIM('Charge Distribution in large box:')

       !------------------------------------------------------------------------
       !     In solution without membrane ;
       !     KAPPA2 = ZERO
       !     VMEMB  = ZERO
       !     EPSM   = EPSW
       !     EPSH   = EPSW
       !     EPSD   = EPSW
       !     EPSC   = EPSW
       !------------------------------------------------------------------------

       ! initialize the potential array
       CALL INITGRID(NCLx,NCLy,NCLz,IPHIP,ZERO)

       ! get boundary potentials in the large box: CG -> ICHC
       IF(.not.QZERO) THEN
          CALL MAYER_BP_INT(LNCEL3,LSTPRP,X,Y,Z,CG,EPSW,ZERO, &
               TMEMB,ILCHC,IPHIP,LNCEL,LNCEL,LNCEL,LDCEL, &
               LTRAN,LTRAN,LTRAN,LXBCEN,LYBCEN,LZBCEN, &
               RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
               RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
               MAJOR,MINOR,QPROLATE, &
               LNCEL,LNCEL,LNCEL,LDCEL,LXBCEN,LYBCEN,LZBCEN, &
               NIMGB,QNPBC,QA,QB)
       ENDIF
       IF(TIMER.GT.1) &
            CALL WRTTIM('Boundary potential in solution (large box):')

       ! solve PBEQ in the large box
       CALL PBEQ5(MAXITS,DOME,DEPS,EPSW,TMEMB,LNCEL,LNCEL,LNCEL, &
            IPHIP,ILCHC,.false.,QPBC,QNPBC)
       !                                     qfocus
       IF(TIMER.GT.1) &
            CALL WRTTIM('PBEQ solver in solution (large box):')

       !         CALL TMPWRIGD(80,LNCEL,LNCEL,LNCEL,LDCEL,LTRAN,LTRAN,LTRAN,
       !     $        LXBCEN,LYBCEN,LZBCEN,-200.D0,0.D0,0.D0,200.D0,0.D0,0.D0,
       !     $        IPHIP,1.0E0)
       !         CALL TMPWRIGD(81,LNCEL,LNCEL,LNCEL,LDCEL,LTRAN,LTRAN,LTRAN,
       !     $        LXBCEN,LYBCEN,LZBCEN,0.D0,0.D0,-200.D0,0.D0,0.D0,200.D0,
       !     $        IPHIP,1.0E0)

       ! construct boundary potentials from the large box
       CALL BOUNDPOT(LNCEL,LNCEL,LNCEL,LDCEL, &
            LTRAN,LTRAN,LTRAN,LXBCEN,LYBCEN,LZBCEN, &
            NCLX,NCLY,NCLZ,DCEL,XBCEN,YBCEN,ZBCEN, &
            IPHIP,IPHIB)

       CALL MAYER_BPFOCUS(IPHIP,IPHIB,NCLX,NCLY,NCLZ)

       ! solve PBEQ in the original box
       CALL PBEQ5(MAXITS,DOME,DEPS,EPSW,TMEMB,NCLX,NCLY,NCLZ, &
            IPHIP,ICHC,QFOCUS,QPBC,QNPBC)
       IF(TIMER.GT.1) &
            CALL WRTTIM('PBEQ solver in solution (original box):')

       !         CALL TMPWRIGD(82,NCLX,NCLY,NCLZ,DCEL,TRANX,TRANY,TRANZ,
       !     $        XBCEN,YBCEN,ZBCEN,-200.D0,0.D0,0.D0,200.D0,0.D0,0.D0,
       !     $        IPHIP,1.0E0)
       !         CALL TMPWRIGD(83,NCLX,NCLY,NCLZ,DCEL,TRANX,TRANY,TRANZ,
       !     $        XBCEN,YBCEN,ZBCEN,0.D0,0.D0,-200.D0,0.D0,0.D0,200.D0,
       !     $        IPHIP,1.0E0)

       !------------------------------------------------------------------------
       !     In solution with membrane (large box - focussing - original box)
       !     VMEMB  = ZERO
       !     KAPPA2
       !     EPSM
       !     EPSH
       !     EPSD
       !     EPSC
       !------------------------------------------------------------------------

       ! get grid parameters in the large box
       ! here, we just use OUTER REGION array (NTRA, LSTRA) to construct
       ! the dielectric boundary.
       CALL MAYER(NTRA,LSTRA,EPSW,EPSP,KAPPA2,X,Y,Z,CG, &
            PBRAD,LNCEL,LNCEL,LNCEL,LDCEL,WATR,IONR,MAY, &
            MAYX,MAYY,MAYZ,SWIN,ILCHC, &
            IPHIW,ZERO,TMEMB,ZMEMB,EPSM,NIMGB, &
            HTMEMB,EPSH,HDENS,POSALP,NEGALP,QCHRG,POFFST,NOFFST, &
            DROPLET,EPSD,XDROPLET,YDROPLET,ZDROPLET,Qdtom,Qdkap, &
            EPSB,LXMAX,LYMAX,LZMAX,LXMIN,LYMIN,LZMIN,Qbtom,Qbkap, &
            EPSC,Xcyln,Ycyln,Zcyln,Rcyln,Hcyln,Qctom,Qckap, &
            EPSEC,Xcone,Ycone,Zcone,Ax,By,Cz,Qectom,Qeckap, &
               -1,-1,&
            LTRAN,LTRAN,LTRAN,LXBCEN,LYBCEN,LZBCEN, &
            IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
            QKEEP,QREEN,QSMTH,QBSPL,QA,QB, &
            RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
            RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
            MAJOR,MINOR,QPROLATE, &
            .false.,.true.,.false.,IPHIB,QPBC,QNPBC, &
                                !              QINTBP QZERO
            QNONLINEAR,QPARTLINEAR,QFKAP, &
            IPOX,IPOY,IPOZ,MAPT,NATOM)
       IF(TIMER.GT.1) &
            CALL WRTTIM('Space functions with complex sol. (large box):')

       ! kappa = zero inside the inner region
       IF(KAPPA2.ne.ZERO) THEN
          IF(SRCUT.GT.0.0) THEN
             CALL RFSPHE_MAYER(LNCEL,LNCEL,LNCEL,LTRAN,LTRAN,LTRAN, &
                  LDCEL,LXBCEN,LYBCEN,LZBCEN,MAY, &
                  RRXCEN,RRYCEN,RRZCEN,SRCUT)
          ELSE
             CALL RFRECT_MAYER(LNCEL,LNCEL,LNCEL,LTRAN,LTRAN,LTRAN, &
                  LDCEL,LXBCEN,LYBCEN,LZBCEN,MAY, &
                  RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX)
          ENDIF
       ENDIF

       ! get boundary potentials in the large box: CG -> ICHC
       IF(.not.QZERO) THEN
          CALL MAYER_BP_INT(LNCEL3,LSTPRP,X,Y,Z,CG,EPSW,KAPPA2, &
               TMEMB,ILCHC,IPHIW,LNCEL,LNCEL,LNCEL,LDCEL, &
               LTRAN,LTRAN,LTRAN,LXBCEN,LYBCEN,LZBCEN, &
               RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
               RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
               MAJOR,MINOR,QPROLATE, &
               LNCEL,LNCEL,LNCEL,LDCEL,LXBCEN,LYBCEN,LZBCEN, &
               NIMGB,QNPBC,QA,QB)
       ENDIF
       IF(TIMER.GT.1) &
            CALL WRTTIM('Boundary potential with complex sol.(large box):')

       ! see subroutine MAYER
       IXMAX=LNCEL-1
       IYMAX=LNCEL-1
       IZMAX=LNCEL-1
       IXMIN=2
       IYMIN=2
       IZMIN=2

       ! solve PBEQ in the large box
       CALL PBEQ1(MAXITS,DOME,DEPS,EPSW,EPSP,EPSM,KAPPA2, &
            LNCEL,LNCEL,LNCEL,IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
            IPHIW,MAYX,MAYY,MAYZ, &
            ILCHC,MAY,TMEMB, &
            QOSOR,.false.,QPBC,QNPBC,.false.)
       IF(TIMER.GT.1) &
            CALL WRTTIM('PBEQ solver with complex sol. (large box):')

       !         CALL TMPWRIGD(84,LNCEL,LNCEL,LNCEL,LDCEL,LTRAN,LTRAN,LTRAN,
       !     $        LXBCEN,LYBCEN,LZBCEN,-200.D0,0.D0,0.D0,200.D0,0.D0,0.D0,
       !     $        IPHIW,1.0E0)
       !         CALL TMPWRIGD(85,LNCEL,LNCEL,LNCEL,LDCEL,LTRAN,LTRAN,LTRAN,
       !     $        LXBCEN,LYBCEN,LZBCEN,0.D0,0.D0,-200.D0,0.D0,0.D0,200.D0,
       !     $        IPHIW,1.0E0)
       !         CALL TMPWRIGD(90,LNCEL,LNCEL,LNCEL,LDCEL,LTRAN,LTRAN,LTRAN,
       !     $        LXBCEN,LYBCEN,LZBCEN,-200.D0,0.D0,0.D0,200.D0,0.D0,0.D0,
       !     $        MAY,1.0E0)
       !         CALL TMPWRIGD(91,LNCEL,LNCEL,LNCEL,LDCEL,LTRAN,LTRAN,LTRAN,
       !     $        LXBCEN,LYBCEN,LZBCEN,0.D0,0.D0,-200.D0,0.D0,0.D0,200.D0,
       !     $        MAY,1.0E0)
       !         CALL TMPWRIGD(92,LNCEL,LNCEL,LNCEL,LDCEL,LTRAN,LTRAN,LTRAN,
       !     $        LXBCEN,LYBCEN,LZBCEN,-200.D0,0.D0,0.D0,200.D0,0.D0,0.D0,
       !     $        MAYX,1.0E0)
       !         CALL TMPWRIGD(93,LNCEL,LNCEL,LNCEL,LDCEL,LTRAN,LTRAN,LTRAN,
       !     $        LXBCEN,LYBCEN,LZBCEN,0.D0,0.D0,-200.D0,0.D0,0.D0,200.D0,
       !     $        MAYZ,1.0E0)

       ! construct boundary potentials from the large box
       CALL BOUNDPOT(LNCEL,LNCEL,LNCEL,LDCEL, &
            LTRAN,LTRAN,LTRAN,LXBCEN,LYBCEN,LZBCEN, &
            NCLX,NCLY,NCLZ,DCEL,XBCEN,YBCEN,ZBCEN, &
            IPHIW,IPHIB)

       ! get the grid parameters in the original box
       CALL MAYER(NTRA,LSTRA,EPSW,EPSP,KAPPA2,X,Y,Z,CG, &
            PBRAD,NCLX,NCLY,NCLZ,DCEL,WATR,IONR,MAY, &
            MAYX,MAYY,MAYZ,SWIN,ICHC, &
            IPHIW,ZERO,TMEMB,ZMEMB,EPSM,NIMGB, &
            HTMEMB,EPSH,HDENS,POSALP,NEGALP,QCHRG,POFFST,NOFFST, &
            DROPLET,EPSD,XDROPLET,YDROPLET,ZDROPLET,Qdtom,Qdkap, &
            EPSB,LXMAX,LYMAX,LZMAX,LXMIN,LYMIN,LZMIN,Qbtom,Qbkap, &
            EPSC,Xcyln,Ycyln,Zcyln,Rcyln,Hcyln,Qctom,Qckap, &
            EPSEC,Xcone,Ycone,Zcone,Ax,By,Cz,Qectom,Qeckap, &
               -1,-1,&
            TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
            IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
            QKEEP,QREEN,QSMTH,QBSPL,QA,QB, &
            RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
            RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
            MAJOR,MINOR,QPROLATE, &
            .false.,.false.,.true.,IPHIB,QPBC,QNPBC, &
                                !              QINTBP QZERO
            QNONLINEAR,QPARTLINEAR,QFKAP, &
            IPOX,IPOY,IPOZ,MAPT,NATOM)

       IF(TIMER.GT.1) &
            CALL WRTTIM('Space functions with complex sol.(original box):')

       ! kappa = zero inside the inner region
       IF(KAPPA2.ne.ZERO) THEN
          IF(SRCUT.GT.0.0) THEN
             CALL RFSPHE_MAYER(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,DCEL, &
                  XBCEN,YBCEN,ZBCEN,MAY, &
                  RRXCEN,RRYCEN,RRZCEN,SRCUT)
          ELSE
             CALL RFRECT_MAYER(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,DCEL, &
                  XBCEN,YBCEN,ZBCEN,MAY, &
                  RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX)
          ENDIF
       ENDIF

       ! see subroutine MAYER
       IXMAX=NCLX-1
       IYMAX=NCLY-1
       IZMAX=NCLZ-1
       IXMIN=2
       IYMIN=2
       IZMIN=2

       ! solve PBEQ in the original box
       CALL PBEQ1(MAXITS,DOME,DEPS,EPSW,EPSP,EPSM,KAPPA2, &
            NCLX,NCLY,NCLZ,IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
            IPHIW,MAYX,MAYY,MAYZ, &
            ICHC,MAY,TMEMB, &
            QOSOR,QFOCUS,QPBC,QNPBC,.false.)
       IF(TIMER.GT.1) CALL WRTTIM('PBEQ solver with complex sol.:')

       !         CALL TMPWRIGD(86,NCLX,NCLY,NCLZ,DCEL,TRANX,TRANY,TRANZ,
       !     $        XBCEN,YBCEN,ZBCEN,-200.D0,0.D0,0.D0,200.D0,0.D0,0.D0,
       !     $        IPHIW,1.0E0)
       !         CALL TMPWRIGD(87,NCLX,NCLY,NCLZ,DCEL,TRANX,TRANY,TRANZ,
       !     $        XBCEN,YBCEN,ZBCEN,0.D0,0.D0,-200.D0,0.D0,0.D0,200.D0,
       !     $        IPHIW,1.0E0)
       !         CALL TMPWRIGD(94,NCLX,NCLY,NCLZ,DCEL,TRANX,TRANY,TRANZ,
       !     $        XBCEN,YBCEN,ZBCEN,-200.D0,0.D0,0.D0,200.D0,0.D0,0.D0,
       !     $        MAY,1.0E0)
       !         CALL TMPWRIGD(95,NCLX,NCLY,NCLZ,DCEL,TRANX,TRANY,TRANZ,
       !     $        XBCEN,YBCEN,ZBCEN,0.D0,0.D0,-200.D0,0.D0,0.D0,200.D0,
       !     $        MAY,1.0E0)
       !         CALL TMPWRIGD(96,NCLX,NCLY,NCLZ,DCEL,TRANX,TRANY,TRANZ,
       !     $        XBCEN,YBCEN,ZBCEN,-200.D0,0.D0,0.D0,200.D0,0.D0,0.D0,
       !     $        MAYX,1.0E0)
       !         CALL TMPWRIGD(97,NCLX,NCLY,NCLZ,DCEL,TRANX,TRANY,TRANZ,
       !     $        XBCEN,YBCEN,ZBCEN,0.D0,0.D0,-200.D0,0.D0,0.D0,200.D0,
       !     $        MAYZ,1.0E0)

       ! calculate the reaction field energy elements M(ij)
       CALL RECT_MIJ(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,NTPOL,NORDER, &
            LSTPX,LSTPY,LSTPZ,BNORM,DCEL, &
            XBCEN,YBCEN,ZBCEN, &
            RRXCEN,RRYCEN,RRZCEN,XSCALE,YSCALE,ZSCALE,IIP0,NPGD, &
            IPHIP,IPHIW,MIJ,CGSCAL)

    ENDDO

    CALL COMP_MMIJ(NTPOL,MMIJ,MIJ,SIJ)
    call chmdealloc('pbeq.src','RFRECT_GSBP1','SIJ',NTPOL,NTPOL,crl=SIJ)
    call chmdealloc('pbeq.src','RFRECT_GSBP1','SIJEVEC',NTPOL,NTPOL,crl=SIJEVEC)
    call chmdealloc('pbeq.src','RFRECT_GSBP1','SIJEVAL',NTPOL,crl=SIJEVAL)
    call chmalloc('pbeq.src','RFRECT_GSBP1','ILCHC',LNCEL3,cr4=ILCHC)
    call chmalloc('pbeq.src','RFRECT_GSBP1','IPHIB',BNCEL,cr4=IPHIB)
    call chmdealloc('pbeq.src','RFRECT_GSBP1','IIP0',NPROI,intg=IIP0)
    QA=.false.
    QB=.false.
    !
    RETURN
  END SUBROUTINE RFRECT_GSBP1
  !
  SUBROUTINE RFSPHE_TRIG(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,DCEL, &
       XBCEN,YBCEN,ZBCEN,VEXCL, &
       IIP0,CTHETA,CPHI,SPHI,RGD,SRDIST, &
       RRXCEN,RRYCEN,RRZCEN,JX1,JX2,JY1,JY2,JZ1,JZ2,NPGD)
    !-----------------------------------------------------------------------
    !     calculate the trigonometric functions of each grid points
    !     inside region of interest with the volume exclusion function
    !
    use chm_kinds
    use consta
    use number
    implicit none
    REAL(CHM_REAL4)  VEXCL(*)
    real(chm_real)  SRDIST,RRXCEN,RRYCEN,RRZCEN
    real(chm_real)  DCEL,TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN
    real(chm_real)  CTHETA(*),CPHI(*),SPHI(*),RGD(*)
    INTEGER NCLx,NCLy,NCLz,IIP0(*)
    INTEGER JX1,JX2,JY1,JY2,JZ1,JZ2,NPGD
    ! local
    INTEGER NC3,NCYZ,IG,JG,KG,IP0X,IP0Y,IP0
    real(chm_real)  XG,YG,ZG,XG2,YG2,ZG2,R,SR2,STHETA
    real(chm_real)  CT,CP,SP
    !
    NCYZ =NCLy*NCLz
    NC3  =NCYZ*NCLx
    SR2  =SRDIST*SRDIST

    ! calculate the trigonometric functions of each grid points
    ! inside region of interest
    NPGD=0
    DO IG=JX1,JX2
       XG=DCEL*(IG-1)-TRANX+XBCEN-RRXCEN
       XG2=XG*XG
       IP0X=(IG-1)*NCyz
       DO JG=JY1,JY2
          YG=DCEL*(JG-1)-TRANY+YBCEN-RRYCEN
          YG2=YG*YG+XG2
          IP0Y=(JG-1)*NCLz+IP0X
          DO KG=JZ1,JZ2
             ZG=DCEL*(KG-1)-TRANZ+ZBCEN-RRZCEN
             ZG2=ZG*ZG+YG2
             IP0 = IP0Y + KG
             IF(ZG2.LE.SR2+RSMALL.AND.VEXCL(IP0).GT.ZERO) THEN
                !     in the origin
                IF(ZG2.LT.RSMALL) THEN
                   NPGD=NPGD+1
                   IIP0(NPGD)=IP0
                   RGD(NPGD)=ZERO
                   CTHETA(NPGD)=ZERO
                   CPHI(NPGD)=ZERO
                   SPHI(NPGD)=ZERO
                   !     in the Z axis
                ELSEIF(XG.GT.-RSMALL.AND.XG.LT.RSMALL.AND. &
                     YG.GT.-RSMALL.AND.YG.LT.RSMALL) THEN
                   NPGD=NPGD+1
                   IIP0(NPGD)=IP0
                   RGD(NPGD)=abs(ZG)
                   CTHETA(NPGD)=ZG/abs(ZG)
                   CTHETA(NPGD)=ONE
                   CPHI(NPGD)=ZERO
                   SPHI(NPGD)=ZERO
                   !     in the XY plane
                ELSEIF(ZG.GT.-RSMALL.AND.ZG.LT.RSMALL) THEN
                   NPGD=NPGD+1
                   R=SQRT(ZG2)
                   IIP0(NPGD)=IP0
                   RGD(NPGD)=R
                   CTHETA(NPGD)=ZERO
                   CPHI(NPGD)=XG/R
                   SPHI(NPGD)=YG/R
                ELSE
                   NPGD=NPGD+1
                   R=SQRT(ZG2)
                   IIP0(NPGD)=IP0
                   RGD(NPGD)=R
                   CTHETA(NPGD)=ZG/R
                   STHETA=SQRT(ONE-ZG*ZG/ZG2)
                   CPHI(NPGD)=XG/R/STHETA
                   SPHI(NPGD)=YG/R/STHETA
                ENDIF
             ENDIF
          ENDDO
       ENDDO
    ENDDO
    !
    RETURN
  END SUBROUTINE RFSPHE_TRIG
  !
  SUBROUTINE RFRECT_GPINRR(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,DCEL, &
       XBCEN,YBCEN,ZBCEN,VEXCL, &
       RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,SRCUT, &
       JX1,JX2,JY1,JY2,JZ1,JZ2,IIP0,NPGD)
    !-----------------------------------------------------------------------
    !     find out the grid points inside region of interest and
    !     store in IIP0 array
    !
    use chm_kinds
    use number
    implicit none
    REAL(CHM_REAL4)  VEXCL(*)
    real(chm_real)  RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,SRCUT
    real(chm_real)  DCEL,TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN
    INTEGER NCLx,NCLy,NCLz,NPGD,IIP0(*)
    INTEGER JX1,JX2,JY1,JY2,JZ1,JZ2
    ! local
    real(chm_real)  XG,YG,ZG,RRXCEN,RRYCEN,RRZCEN,SR2,R2
    INTEGER NCYZ,IG,JG,KG,IP0X,IP0Y,IP0
    !
    RRXCEN = (RBXMAX+RBXMIN)/TWO
    RRYCEN = (RBYMAX+RBYMIN)/TWO
    RRZCEN = (RBZMAX+RBZMIN)/TWO
    SR2=SRCUT*SRCUT
    !
    NPGD=0
    NCYZ =NCLy*NCLz
    ig_loop: DO IG=JX1,JX2
       XG=DCEL*(IG-1)-TRANX+XBCEN
       IF(XG.LT.RBXMIN.OR.XG.GT.RBXMAX) cycle ig_loop
       IP0X=(IG-1)*NCyz
       jg_loop: DO JG=JY1,JY2
          YG=DCEL*(JG-1)-TRANY+YBCEN
          IF(YG.LT.RBYMIN.OR.YG.GT.RBYMAX) cycle jg_loop
          IP0Y=(JG-1)*NCLz+IP0X
          kg_loop: DO KG=JZ1,JZ2
             ZG=DCEL*(KG-1)-TRANZ+ZBCEN
             IF(ZG.LT.RBZMIN.OR.ZG.GT.RBZMAX) cycle kg_loop
             IP0 = IP0Y + KG

             IF(VEXCL(IP0).GT.ZERO) THEN
                IF(SRCUT.GT.0.0) THEN
                   XG=XG-RRXCEN
                   YG=YG-RRYCEN
                   ZG=ZG-RRZCEN
                   R2=XG*XG+YG*YG+ZG*ZG
                   IF(R2.LE.SR2) THEN
                      NPGD=NPGD+1
                      IIP0(NPGD)=IP0
                   ENDIF
                ELSE
                   NPGD=NPGD+1
                   IIP0(NPGD)=IP0
                ENDIF
             ENDIF

          ENDDO kg_loop
       ENDDO jg_loop
    ENDDO ig_loop
    !
    RETURN
  END SUBROUTINE RFRECT_GPINRR
  !
  SUBROUTINE RFSPHE_SIJ(NPGD,DCEL,NTPOL,TOLSIJ, &
       LSTPL,LSTPM,BNORM, &
       CTHETA,CPHI,SPHI,RGD, &
       SIJ,SIJEVEC,SIJEVAL)
    !-----------------------------------------------------------------------
    !     setup the overlap matrix (SIJ)
    !
    use chm_kinds
    use number
    use stream
    implicit none

    INTEGER LSTPL(*),LSTPM(*),NTPOL,NPGD
    real(chm_real)  CTHETA(*),CPHI(*),SPHI(*),RGD(*),BNORM(*)
    real(chm_real)  DCEL,TOLSIJ
    real(chm_real)  SIJ(NTPOL,NTPOL),SIJEVEC(NTPOL,NTPOL)
    real(chm_real)  SIJEVAL(NTPOL)
    ! local
    real(chm_real)  DCEL3,CT,CP,SP,R
    real(chm_real)  B(1000),NORM,S,TMPWORK(NTPOL)
    INTEGER I,J,K,PL,PM,M,NOFF
    !
    if(NTPOL.GT.1000) CALL WRNDIE(-3,'<RFSPHE_SIJ>', &
         'Dimension of B overflow.') 
    !
    DCEL3 = DCEL*DCEL*DCEL
    DO I=1,NTPOL
       DO J=1,NTPOL
          SIJ(J,I)=ZERO
       ENDDO
    ENDDO
    DO K=1,NPGD
       CT=CTHETA(K)
       CP=CPHI(K)
       SP=SPHI(K)
       R =RGD(K)
       DO I=1,NTPOL
          PL=LSTPL(I)
          PM=LSTPM(I)
          NORM=BNORM(I)
          IF(PM.GE.0) THEN
             b(i)=NORM*RPOWERL(PL,R)*COSMPHI(PM,CP)*ALPOL(PL,PM,CT)
          ELSEIF(PM.LT.0) THEN
             M=-PM
             b(i)=NORM*RPOWERL(PL,R)*SINMPHI(M,CP,SP)*ALPOL(PL,M,CT)
          ENDIF
       ENDDO
       DO I=1,NTPOL
          DO J=I,NTPOL
             SIJ(I,J)=SIJ(I,J)+b(i)*b(j)*DCEL3
          ENDDO
       ENDDO
    ENDDO

    DO I=1,NTPOL
       DO J=I,NTPOL
          S=SIJ(I,J)
          IF(S.LT.RSMALL.and.S.GT.-RSMALL) THEN
             SIJ(I,J)=ZERO
             S=SIJ(I,J)
          ENDIF
          SIJ(J,I)=S
       ENDDO
    ENDDO

    !
    !     Get the transformation matrix X=U*S^(-1/2)
    !
    J=0
    ! get the eigenvectors (SIJEVEC) and eigenvalues (SIJEVAL)
    ! of the overalp matrix
    CALL EIGRS(SIJ,NTPOL,11,SIJEVAL,SIJEVEC,NTPOL,J)
    IF (J .GT. 128) THEN
       J = J - 128
       WRITE (OUTU,'(A,I6,A)') &
            ' DIAGRS> Failed to converge on root number ', J, '.'
    ENDIF
    DO I=1,NTPOL
       write(outu,'(i5,f10.6)') I,SIJEVAL(I)
    ENDDO

    NOFF=0
    DO I=1,NTPOL
       IF(SIJEVAL(I).GT.TOLSIJ) THEN
          NOFF=I-1
          write(outu,*) 'NOFF =',NOFF,' TOLSIJ=',TOLSIJ
          GOTO 90
       ENDIF
    ENDDO
90  CONTINUE

    ! change the order of eigenvalues and column of eigenvectors
    DO I=1,NTPOL
       S=sqrt(SIJEVAL(NTPOL+1-I))
       DO J=1,NTPOL
          SIJ(J,I)=SIJEVEC(J,NTPOL+1-I)/ S
       ENDDO
    ENDDO

    ! construct X*X^T using SIJEVEC as a temporary array
    DO I=1,NTPOL
       DO J=1,NTPOL
          S=ZERO
          DO K=1,NTPOL-NOFF
             S=S+SIJ(I,K)*SIJ(J,K)
          ENDDO
          SIJEVEC(I,J)=S
       ENDDO
    ENDDO

    ! replace SIJEVEC with SIJ
    DO I=1,NTPOL
       DO J=1,NTPOL
          SIJ(I,J)=SIJEVEC(I,J)
       ENDDO
    ENDDO

    RETURN
  END SUBROUTINE RFSPHE_SIJ
  !
  SUBROUTINE RFRECT_SIJ(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,DCEL, &
       XBCEN,YBCEN,ZBCEN,TOLSIJ, &
       NTPOL,LSTPX,LSTPY,LSTPZ,BNORM, &
       IIP0,NPGD,RRXCEN,RRYCEN,RRZCEN,XSCALE,YSCALE,ZSCALE, &
       SIJ,SIJEVEC,SIJEVAL)
    !-----------------------------------------------------------------------
    !     setup the overlap matrix (SIJ)
    !
    use chm_kinds
    use consta
    use number
    use stream
    implicit none
    INTEGER NCLx,NCLy,NCLz,NPGD,IIP0(*)
    INTEGER LSTPX(*),LSTPY(*),LSTPZ(*),NTPOL
    real(chm_real)  RRXCEN,RRYCEN,RRZCEN,BNORM(*)
    real(chm_real)  DCEL,TRANX,TRANY,TRANZ,XSCALE,YSCALE,ZSCALE
    real(chm_real)  XBCEN,YBCEN,ZBCEN,TOLSIJ
    real(chm_real)  SIJ(NTPOL,NTPOL),SIJEVEC(NTPOL,NTPOL)
    real(chm_real)  SIJEVAL(NTPOL)
    ! local
    INTEGER I,J,K,NCYZ,NC3,IP0,IG,JG,KG,NOFF
    INTEGER XPOL,YPOL,ZPOL
    real(chm_real)  DCEL3,XG,YG,ZG,NORM,B(NTPOL)
    real(chm_real)  XADD,YADD,ZADD,S,TMPWORK(NTPOL)
    !
    DO I=1,NTPOL
       DO J=1,NTPOL
          SIJ(J,I)=ZERO
       ENDDO
    ENDDO
    NCYZ =NCLy*NCLz
    NC3  =NCYZ*NCLx
    DCEL3=DCEL*DCEL*DCEL

    XADD = -TRANX+XBCEN-RRXCEN
    YADD = -TRANY+YBCEN-RRYCEN
    ZADD = -TRANZ+ZBCEN-RRZCEN
    !
    DO K=1,NPGD
       IP0=IIP0(K)
       IG=INT((IP0-1)/NCYZ)+1
       JG=INT(MOD((IP0-1),NCYZ)/NCLZ)+1
       KG=MOD(MOD((IP0-1),NCYZ),NCLZ)+1
       XG=DCEL*(IG-1) + XADD
       YG=DCEL*(JG-1) + YADD
       ZG=DCEL*(KG-1) + ZADD
       DO I=1,NTPOL
          XPOL=LSTPX(I)
          YPOL=LSTPY(I)
          ZPOL=LSTPZ(I)
          NORM=BNORM(I)
          b(i)=NORM*LPOL(XPOL,XG,XSCALE)*LPOL(YPOL,YG,YSCALE)* &
               LPOL(ZPOL,ZG,ZSCALE)
       ENDDO
       DO I=1,NTPOL
          DO J=I,NTPOL
             SIJ(I,J)=SIJ(I,J)+b(i)*b(j)*DCEL3
          ENDDO
       ENDDO
    ENDDO

    DO I=1,NTPOL
       DO J=I,NTPOL
          S=SIJ(I,J)
          IF(S.LT.RSMALL.and.S.GT.-RSMALL) THEN
             SIJ(I,J)=ZERO
             S=ZERO
          ENDIF
          SIJ(J,I)=S
       ENDDO
    ENDDO

    !     Get the transformation matrix X=U*S^(-1/2)
    !
    J=0
    ! get the eigenvectors (SIJEVEC) and eigenvalues (SIJEVAL)
    ! of the overalp matrix
    CALL EIGRS(SIJ,NTPOL,11,SIJEVAL,SIJEVEC,NTPOL,J)
    IF (J .GT. 128) THEN
       J = J - 128
       WRITE (OUTU,'(A,I6,A)') &
            ' DIAGRS> Failed to converge on root number ', J, '.'
    ENDIF
    DO I=1,NTPOL
       write(outu,'(i5,f10.6)') I,SIJEVAL(I)
    ENDDO

    NOFF=0
    DO I=1,NTPOL
       IF(SIJEVAL(I).GT.TOLSIJ) THEN
          NOFF=I-1
          write(outu,*) 'NOFF =',NOFF,' TOLSIJ=',TOLSIJ
          GOTO 90
       ENDIF
    ENDDO
90  CONTINUE

    ! change the order of eigenvalues and column of eigenvectors
    DO I=1,NTPOL
       S=sqrt(SIJEVAL(NTPOL+1-I))
       DO J=1,NTPOL
          SIJ(J,I)=SIJEVEC(J,NTPOL+1-I)/ S
       ENDDO
    ENDDO

    ! construct X*X^T using SIJEVEC as a temporary array
    DO I=1,NTPOL
       DO J=1,NTPOL
          S=ZERO
          DO K=1,NTPOL-NOFF
             S=S+SIJ(I,K)*SIJ(J,K)
          ENDDO
          SIJEVEC(I,J)=S
       ENDDO
    ENDDO

    ! replace SIJEVEC with SIJ
    DO I=1,NTPOL
       DO J=1,NTPOL
          SIJ(I,J)=SIJEVEC(I,J)
       ENDDO
    ENDDO

    RETURN
  END SUBROUTINE RFRECT_SIJ
  !
  SUBROUTINE COMP_MMIJ(NTPOL,MMIJ,MIJ,XXIJ)
    !------------------------------------------------------------------------
    !     MMIJ=XXIJ*MIJ*XIJ
    !
    use chm_kinds
    use number
    implicit none
    INTEGER NTPOL
    real(chm_real)  MMIJ(*),MIJ(*),XXIJ(NTPOL,NTPOL)
    !
    INTEGER I,J,IJ,JI,M,N,MN
    real(chm_real)  SUM,XIM
    !
    DO I=1,NTPOL
       DO J=1,NTPOL
          IF(I.GT.J) THEN
             IJ=(I-1)*NTPOL+J
             JI=(J-1)*NTPOL+I
             MIJ(IJ)=MIJ(JI)
          ENDIF
       ENDDO
    ENDDO

    DO M=1,NTPOL
       DO N=1,NTPOL
          MN=(M-1)*NTPOL+N
          SUM=ZERO
          DO I=1,NTPOL
             XIM=XXIJ(I,M)
             DO J=1,NTPOL
                IJ=(I-1)*NTPOL+J
                SUM=SUM+XIM*MIJ(IJ)*XXIJ(J,N)
             ENDDO
          ENDDO
          MMIJ(MN)=SUM
       ENDDO
    ENDDO

    RETURN
  END SUBROUTINE COMP_MMIJ
  !
  SUBROUTINE RFSPHE_MAYER(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,DCEL, &
       XBCEN,YBCEN,ZBCEN,FKAPA, &
       RRXCEN,RRYCEN,RRZCEN,SRDIST)
    !-----------------------------------------------------------------------
    !     kappa = 0 inside the inner region
    !
    use chm_kinds
    use number
    implicit none
    REAL(CHM_REAL4)  FKAPA(*)
    real(chm_real)  DCEL,TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN
    real(chm_real)  RRXCEN,RRYCEN,RRZCEN,SRDIST
    INTEGER NCLx,NCLy,NCLz
    ! local
    real(chm_real)  XG,YG,ZG,XG2,YG2,ZG2,DSQ1,SR2
    INTEGER IG,JG,KG
    INTEGER IP0,IP0X,IP0Y,NCYZ
    INTEGER NFIL,IXCEN,IYCEN,IZCEN,JX1,JX2,JY1,JY2,JZ1,JZ2

    NFIL=INT(SRDIST/DCEL)+2
    IXCEN=INT((RRXCEN+TRANX-XBCEN)/DCEL)+1
    IYCEN=INT((RRYCEN+TRANY-YBCEN)/DCEL)+1
    IZCEN=INT((RRZCEN+TRANZ-ZBCEN)/DCEL)+1
    JX1=IXCEN-NFIL
    JX2=IXCEN+NFIL+1
    JY1=IYCEN-NFIL
    JY2=IYCEN+NFIL+1
    JZ1=IZCEN-NFIL
    JZ2=IZCEN+NFIL+1

    ! For dielectric constants and Debye-Huckel screening factors
    ! NOTE: dielectric constants are located at the middle point between grids
    NCYZ= NCLy*NCLz
    SR2=SRDIST*SRDIST+rsmall
    DO IG=JX1,JX2
       XG=DCEL*(IG-1)-TRANX+XBCEN-RRXCEN
       XG2=XG*XG
       IP0X=(IG-1)*NCyz
       DO JG=JY1,JY2
          YG=DCEL*(JG-1)-TRANY+YBCEN-RRYCEN
          YG2=YG*YG
          IP0Y=(JG-1)*NCLz+IP0X
          DO KG=JZ1,JZ2
             ZG=DCEL*(KG-1)-TRANZ+ZBCEN-RRZCEN
             ZG2=ZG*ZG
             IP0 = IP0Y + KG
             IF(FKAPA(IP0).ne.ZERO)THEN
                DSQ1=XG2+YG2+ZG2
                IF(DSQ1.LE.SR2) FKAPA(IP0)=ZERO
             ENDIF
          ENDDO
       ENDDO
    ENDDO
    !
    RETURN
  END SUBROUTINE RFSPHE_MAYER
  !
  SUBROUTINE RFRECT_MAYER(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,DCEL, &
       XBCEN,YBCEN,ZBCEN,FKAPA, &
       RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX)
    !-----------------------------------------------------------------------
    !     kappa = 0 inside the inner region
    !
    use chm_kinds
    use number
    implicit none
    REAL(CHM_REAL4)  FKAPA(*)
    real(chm_real)  DCEL,TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN
    real(chm_real)  RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX
    INTEGER NCLx,NCLy,NCLz
    ! local
    real(chm_real)  XG,YG,ZG,XEPS,YEPS,ZEPS
    INTEGER NCYZ,IG,JG,KG,IP0X,IP0Y,IP0
    INTEGER JX1,JX2,JY1,JY2,JZ1,JZ2
    !
    NCYZ  = NCLy*NCLz
    !
    JX1=INT((TRANX+RBXMIN-XBCEN)/DCEL)-1
    JX2=INT((TRANX+RBXMAX-XBCEN)/DCEL)+3
    JY1=INT((TRANY+RBYMIN-YBCEN)/DCEL)-1
    JY2=INT((TRANY+RBYMAX-YBCEN)/DCEL)+3
    JZ1=INT((TRANZ+RBZMIN-ZBCEN)/DCEL)-1
    JZ2=INT((TRANZ+RBZMAX-ZBCEN)/DCEL)+3

    ! For Debye-Huckel screening factors
    ig_loop: DO IG=JX1,JX2
       XG=DCEL*(IG-1)-TRANX+XBCEN
       IF(XG.LT.RBXMIN-RSMALL.OR.XG.GT.RBXMAX+RSMALL) cycle ig_loop
       IP0X=(IG-1)*NCyz
       jg_loop: DO JG=JY1,JY2
          YG=DCEL*(JG-1)-TRANY+YBCEN
          IF(YG.LT.RBYMIN-RSMALL.OR.YG.GT.RBYMAX+RSMALL) cycle jg_loop
          IP0Y=(JG-1)*NCLz+IP0X
          kg_loop: DO KG=JZ1,JZ2
             ZG=DCEL*(KG-1)-TRANZ+ZBCEN
             IF(ZG < RBZMIN-RSMALL .or.  &
                  ZG > RBZMAX+RSMALL) cycle kg_loop
             IP0 = IP0Y + KG

             FKAPA(IP0)=ZERO

          ENDDO kg_loop
       ENDDO jg_loop
    ENDDO ig_loop
    !
    RETURN
  END SUBROUTINE RFRECT_MAYER
  !
  SUBROUTINE PROL_SIJ(NTPOL,LSTPL,LSTPM,BNORM, &
       MAJOR,MINOR,TOLSIJ, &
       SIJ,SIJEVEC,SIJEVAL)
    !-----------------------------------------------------------------------
    !     setup the overlap matrix (SIJ) in a prolate spheroidal reaction inner
    !
    use chm_kinds
    use number
    use consta
    use stream
    implicit none
    INTEGER NTPOL,LSTPL(*),LSTPM(*)
    real(chm_real)  BNORM(*),MAJOR,MINOR,TOLSIJ
    real(chm_real)  SIJ(NTPOL,NTPOL),SIJEVEC(NTPOL,NTPOL)
    real(chm_real)  SIJEVAL(NTPOL)
    ! local
    INTEGER I,J,K,PL1,PM1,PL2,PM2,NOFF
    !LNI: TMPWORK is an automatic array that will be allocated on the system stack
    real(chm_real)  S,NORM1,NORM2,ETAPART,ZETAPART,FOCI,ETAMAX,TMPWORK(NTPOL) 

    FOCI=SQRT(MAJOR*MAJOR-MINOR*MINOR)
    ETAMAX=MAJOR/FOCI
    DO I=1,NTPOL
       DO J=1,NTPOL
          SIJ(J,I)=ZERO
       ENDDO
    ENDDO
    DO I=1,NTPOL
       SIJ(I,I)=ONE
    ENDDO
    DO I=1,NTPOL
       PL1=LSTPL(I)
       PM1=LSTPM(I)
       NORM1=BNORM(I)
       DO J=I,NTPOL
          PL2=LSTPL(J)
          PM2=LSTPM(J)
          NORM2=BNORM(J)
          IF(PM1.EQ.PM2.AND.PL2-2.EQ.PL1) THEN
             IF(PM1.LT.0) THEN
                PM1=-PM1
                PM2=-PM2
             ENDIF
             ETAPART=(ONE-ETAMAX*ETAMAX)* &
                  (ALPOL(PL1,PM1,ETAMAX)*DALPOL(PL2,PM2,ETAMAX)- &
                  ALPOL(PL2,PM2,ETAMAX)*DALPOL(PL1,PM1,ETAMAX))/ &
                  ((PL2-TWO)*(PL2-ONE)-PL2*(PL2+ONE))
             ZETAPART=-TWO*FACTORI(PL2+PM2)/FACTORI(PL2-PM2-2)/ &
                  ((TWO*PL2+ONE)*(TWO*PL2-ONE)*(TWO*PL2-THREE))
             IF(PM2.EQ.0) THEN
                SIJ(I,J)=TWOPI*NORM1*NORM2*FOCI**(2*PL2+1)* &
                     ETAPART*ZETAPART
             ELSE
                SIJ(I,J)=PI*NORM1*NORM2*FOCI**(2*PL2+1)* &
                     ETAPART*ZETAPART
             ENDIF
             SIJ(J,I)=SIJ(I,J)
          ENDIF
       ENDDO
    ENDDO

    !
    !     Get the transformation matrix X=U*S^(-1/2)
    !
    J=0
    ! get the eigenvectors (SIJEVEC) and eigenvalues (SIJEVAL) of the overalp matrix
    CALL EIGRS(SIJ,NTPOL,11,SIJEVAL,SIJEVEC,NTPOL,J)
    IF (J .GT. 128) THEN
       J = J - 128
       WRITE (OUTU,'(A,I6,A)') &
            ' DIAGRS> Failed to converge on root number ', J, '.'
    ENDIF
    DO I=1,NTPOL
       write(outu,'(i5,f10.6)') I,SIJEVAL(I)
    ENDDO

    NOFF=0
    DO I=1,NTPOL
       IF(SIJEVAL(I).GT.TOLSIJ) THEN
          NOFF=I-1
          write(outu,*) 'NOFF =',NOFF,' TOLSIJ=',TOLSIJ
          GOTO 90
       ENDIF
    ENDDO
90  CONTINUE

    ! change the order of eigenvalues and column of eigenvectors
    DO I=1,NTPOL
       S=sqrt(SIJEVAL(NTPOL+1-I))
       DO J=1,NTPOL
          SIJ(J,I)=SIJEVEC(J,NTPOL+1-I)/ S
       ENDDO
    ENDDO

    ! construct X*X^T using SIJEVEC as a temporary array
    DO I=1,NTPOL
       DO J=1,NTPOL
          S=ZERO
          DO K=1,NTPOL-NOFF
             S=S+SIJ(I,K)*SIJ(J,K)
          ENDDO
          SIJEVEC(I,J)=S
       ENDDO
    ENDDO

    ! replace SIJEVEC with SIJ
    DO I=1,NTPOL
       DO J=1,NTPOL
          SIJ(I,J)=SIJEVEC(I,J)
       ENDDO
    ENDDO

    RETURN
  END SUBROUTINE PROL_SIJ
  !
  SUBROUTINE PROL_STPOL(NTRB,LSTRB,X,Y,Z,CG,MAJOR,MINOR,NTPOL, &
       RRXCEN,RRYCEN,RRZCEN,MMIJ,COEF,EMN, &
       LSTPL,LSTPM,BNORM,LSTPOL,ICALL,NLIST,QNOSORT)
    !-----------------------------------------------------------------------
    !     calculate the coefficients of the basis functions and
    !     make a list of the basis functions according to their contribution
    !     to the reaction field energy
    !
    use chm_kinds
    use number
    use consta
    use stream
    implicit none
    real(chm_real)  X(*),Y(*),Z(*),CG(*),COEF(*),MMIJ(*),EMN(*), &
         BNORM(*)
    real(chm_real)  RRXCEN,RRYCEN,RRZCEN,MAJOR,MINOR
    INTEGER NTRB,LSTRB(*),NTPOL,ICALL,NLIST
    INTEGER LSTPL(*),LSTPM(*),LSTPOL(*)
    LOGICAL QNOSORT
    ! local
    real(chm_real)  NORM,EGD,ZGD,CP,SP
    real(chm_real)  XDIFF,YDIFF,ZDIFF,XY2,ZF1,ZF2,R1,R2,R,FOCI
    real(chm_real)  MAJOR2,MINOR2
    real(chm_real)  EMIN,TMPEMN,CCC
    INTEGER II,JJ,L,N,M,I,J,IJ
    INTEGER NORDER,POLMIN,TMPPOL

    ! calculate the coefficients of the basis functions
    FOCI=SQRT(MAJOR*MAJOR-MINOR*MINOR)
    MAJOR2=MAJOR*MAJOR
    MINOR2=MINOR*MINOR
    DO II=1,NTPOL
       COEF(II)=ZERO
    ENDDO
    ntrb_loop: DO I=1,NTRB
       J =LSTRB(I)
       CCC=CG(J)
       IF(CCC.EQ.ZERO) cycle ntrb_loop
       XDIFF=X(J)-RRXCEN
       YDIFF=Y(J)-RRYCEN
       ZDIFF=Z(J)-RRZCEN
       XY2=XDIFF*XDIFF+YDIFF*YDIFF
       R=XY2/MINOR2+ZDIFF*ZDIFF/MAJOR2
       IF(R.GT.ONE) cycle ntrb_loop
       ZF1=ZDIFF+FOCI
       ZF2=ZDIFF-FOCI
       R1=SQRT(XY2+ZF1*ZF1)
       R2=SQRT(XY2+ZF2*ZF2)
       EGD=(R1+R2)/(TWO*FOCI)
       ZGD=(R1-R2)/(TWO*FOCI)
       R=SQRT((EGD**2-ONE)*(ONE-ZGD**2))
       CP=XDIFF/R/FOCI
       SP=YDIFF/R/FOCI
       IF(XDIFF.GT.-RSMALL.AND.XDIFF.LT.RSMALL.AND.     & ! in the Z-axis
            YDIFF.GT.-RSMALL.AND.YDIFF.LT.RSMALL) THEN
          CP=ZERO
          SP=ZERO
       ENDIF
       DO II=1,NTPOL
          L=LSTPL(II)
          M=LSTPM(II)
          NORM=BNORM(II)
          IF(M.EQ.0) THEN
             COEF(II)=COEF(II)+CCC*NORM*FOCI**L* &
                  ALPOL(L,M,EGD)*ALPOL(L,M,ZGD)
          ELSEIF(M.GT.0) THEN
             COEF(II)=COEF(II)+CCC*NORM*FOCI**L* &
                  ALPOL(L,M,EGD)*ALPOL(L,M,ZGD)*COSMPHI(M,CP)
          ELSEIF(M.LT.0) THEN
             M=-M
             COEF(II)=COEF(II)+CCC*NORM*FOCI**L* &
                  ALPOL(L,M,EGD)*ALPOL(L,M,ZGD)*SINMPHI(M,CP,SP)
          ENDIF
       ENDDO
    ENDDO ntrb_loop

    IF(MOD(ICALL,NLIST).NE.0.OR.QNOSORT) GOTO 100
    ! calculate the energy contribution of each basis function and
    ! find the smallest one and remove the function
    ! repeat it again
    DO II=1,NTPOL
       LSTPOL(II)=II
    ENDDO
    DO NORDER=NTPOL,1,-1
       DO I=1,NORDER
          II=LSTPOL(I)
          IJ=(II-1)*NTPOL+II
          EMN(I)=HALF*COEF(II)*COEF(II)*MMIJ(IJ)*CCELEC
          DO J=1,I-1
             JJ=LSTPOL(J)
             IJ=(JJ-1)*NTPOL+II
             EMN(I)=EMN(I)+COEF(II)*COEF(JJ)*MMIJ(IJ)*CCELEC
          ENDDO
          DO J=I+1,NORDER
             JJ=LSTPOL(J)
             IJ=(II-1)*NTPOL+JJ
             EMN(I)=EMN(I)+COEF(II)*COEF(JJ)*MMIJ(IJ)*CCELEC
          ENDDO
       ENDDO
       EMIN=RBIG
       DO I=1,NORDER
          IF(abs(EMN(I)).lt.EMIN) THEN
             EMIN=abs(EMN(I))
             POLMIN=I
          ENDIF
       ENDDO
       TMPPOL=LSTPOL(POLMIN)
       DO I=POLMIN+1,NORDER
          LSTPOL(I-1)=LSTPOL(I)
       ENDDO
       LSTPOL(NORDER)=TMPPOL
    ENDDO
    !
100 CONTINUE
    IF(ICALL.GT.0.OR.PRNLEV.LT.6) RETURN
    !
    WRITE(OUTU,'(/,3X,A)') &
         'Coefficients of Basis Functions and Running Summations in G_II'
    WRITE(OUTU,'(3X,A)') &
         ' NORDER   YL   YM        COEFF.       G_II(TOT)       G_II(N)'
    !
    DO I=1,NTPOL
       II=LSTPOL(I)
       IJ=(II-1)*NTPOL+II
       EMN(I)=HALF*COEF(II)*COEF(II)*MMIJ(IJ)*CCELEC
       DO J=1,I-1
          JJ=LSTPOL(J)
          IJ=(II-1)*NTPOL+JJ
          IF(II.GT.JJ) IJ=(JJ-1)*NTPOL+II
          EMN(I)=EMN(I)+COEF(II)*COEF(JJ)*MMIJ(IJ)*CCELEC
       ENDDO
    ENDDO

    TMPEMN=ZERO
    DO NORDER=1,NTPOL
       TMPEMN=TMPEMN+EMN(NORDER)
       II=LSTPOL(NORDER)
       L=LSTPL(II)
       M=LSTPM(II)
       write(OUTU,'(3x,i5,2x,2i5,2x,3E15.7)') &
            NORDER,L,M,COEF(II),TMPEMN,EMN(NORDER)
    ENDDO
    !
    RETURN
  END SUBROUTINE PROL_STPOL
  !
  SUBROUTINE PROL_GSBP3(NTRB,LSTRB,X,Y,Z,CG,MAJOR,MINOR, &
       NTPOL,MAXNPOL,RRXCEN,RRYCEN,RRZCEN, &
       MMIJ,COEF, &
       RXNBFX,RXNBFY,RXNBFZ, &
       LSTPL,LSTPM,BNORM, &
       MQ,LSTPOL,EGSBPB)
    !-----------------------------------------------------------------------
    !     calculate the electrostatic reaction field energy and forces
    !     in region of interest
    !     NOTE: when a charge is on the foci,
    !           the current code give zero forces (incorrect one).
    !     But, the situation represents a set of measure zero.
    !
    use chm_kinds
    use stream
    use number
    use consta
    implicit none
    real(chm_real)  X(*),Y(*),Z(*),CG(*),COEF(*),MMIJ(*), &
         BNORM(*),MQ(*)
    real(chm_real)  RXNBFX(*),RXNBFY(*),RXNBFZ(*),EGSBPB
    real(chm_real)  MAJOR,MINOR,RRXCEN,RRYCEN,RRZCEN
    INTEGER NTRB,LSTRB(*)
    INTEGER LSTPL(*),LSTPM(*),LSTPOL(*),NTPOL,MAXNPOL
    ! local
    INTEGER I,J,II,JJ,IJ,L,M,LL,MM
    real(chm_real)  NORM
    real(chm_real)  CCC,CMMIJ,CMPHI,SMPHI
    real(chm_real)  XDIFF,YDIFF,ZDIFF,XY2,ZF1,ZF2,R1,R2,R
    real(chm_real)  FOCI,FOCIL,EGD,ZGD,EGD2,ZGD2,CP,SP
    real(chm_real)  IHETA,IHETA2,IHZETA,IHZETA2,IHPHI
    real(chm_real)  EPOL,ZPOL,DETA,DZETA,DPHI,DX,DY,DZ
    real(chm_real)  TMPFX,TMPFY,TMPFZ

    ! construct MQ array to speed up the calcaulations
    DO I=1,MAXNPOL
       II=LSTPOL(I)
       MQ(II)=ZERO
       DO J=1,MAXNPOL
          JJ=LSTPOL(J)
          IJ=(II-1)*NTPOL+JJ
          IF(II.GT.JJ) IJ=(JJ-1)*NTPOL+II
          MQ(II)=MQ(II)+MMIJ(IJ)*COEF(JJ)
       ENDDO
    ENDDO

    ! reaction field energy calculation
    EGSBPB=ZERO
    DO I=1,MAXNPOL
       II=LSTPOL(I)
       EGSBPB=EGSBPB+0.5d0*COEF(II)*MQ(II)
    ENDDO
    EGSBPB=EGSBPB*CCELEC

    ! reaction field force calculations
    FOCI=SQRT(MAJOR*MAJOR-MINOR*MINOR)
    ll_loop: DO LL=1,NTRB
       MM=LSTRB(LL)
       XDIFF=X(MM)-RRXCEN
       YDIFF=Y(MM)-RRYCEN
       ZDIFF=Z(MM)-RRZCEN
       TMPFX=ZERO
       TMPFY=ZERO
       TMPFZ=ZERO
       CCC=CG(MM)*CCELEC
       IF(CCC.EQ.ZERO) cycle ll_loop
       XY2=XDIFF*XDIFF+YDIFF*YDIFF
       ZF1=ZDIFF+FOCI
       ZF2=ZDIFF-FOCI
       R1=SQRT(XY2+ZF1*ZF1)
       R2=SQRT(XY2+ZF2*ZF2)
       EGD=(R1+R2)/(TWO*FOCI)
       ZGD=(R1-R2)/(TWO*FOCI)
       EGD2=EGD*EGD
       ZGD2=ZGD*ZGD
       R=SQRT((EGD2-ONE)*(ONE-ZGD2))
       CP=XDIFF/R/FOCI
       SP=YDIFF/R/FOCI
       IHETA=SQRT((EGD2-ONE)/(EGD2-ZGD2))/FOCI           ! inverse Heta
       IHZETA=SQRT((ONE-ZGD2)/(EGD2-ZGD2))/FOCI
       IHPHI=ONE/SQRT((EGD2-ONE)*(ONE-ZGD2))/FOCI
       IF(XDIFF.GT.-RSMALL.AND.XDIFF.LT.RSMALL.AND.       & ! in the Z-axis
            YDIFF.GT.-RSMALL.AND.YDIFF.LT.RSMALL) THEN
          CP=ZERO
          SP=ZERO
          IHPHI=ZERO
          IF(EGD.GT.ONE-RSMALL.AND.EGD.LT.ONE+RSMALL) IHETA=ZERO
          IF(ZGD.GT.ONE-RSMALL.AND.ZGD.LT.ONE+RSMALL) IHZETA=ZERO
          IF(ZGD.GT.-ONE-RSMALL.AND.ZGD.LT.-ONE+RSMALL) IHZETA=ZERO
       ENDIF
       IHETA2=IHETA*IHETA
       IHZETA2=IHZETA*IHZETA
       DO I=1,MAXNPOL
          II=LSTPOL(I)
          L=LSTPL(II)
          M=LSTPM(II)
          NORM=BNORM(II)
          IF(M.GE.0) THEN
             EPOL =ALPOL(L,M,EGD)
             ZPOL =ALPOL(L,M,ZGD)
             CMPHI=COSMPHI(M,CP)
             DETA =DALPOL(L,M,EGD)*ZPOL*CMPHI
             DZETA=EPOL*DALPOL(L,M,ZGD)*CMPHI
             DPHI=-EPOL*ZPOL*M*SINMPHI(M,CP,SP)
          ELSEIF(M.LT.0) THEN
             M=-M
             EPOL =ALPOL(L,M,EGD)
             ZPOL =ALPOL(L,M,ZGD)
             SMPHI=SINMPHI(M,CP,SP)
             DETA =DALPOL(L,M,EGD)*ZPOL*SMPHI
             DZETA=EPOL*DALPOL(L,M,ZGD)*SMPHI
             DPHI =EPOL*ZPOL*M*COSMPHI(M,CP)
          ENDIF
          FOCIL=FOCI**L
          DX=NORM*FOCIL* &
               ((DETA*EGD-DZETA*ZGD)*FOCI*CP*IHETA*IHZETA- &
               DPHI*SP*IHPHI)
          DY=NORM*FOCIL* &
               ((DETA*EGD-DZETA*ZGD)*FOCI*SP*IHETA*IHZETA+ &
               DPHI*CP*IHPHI)
          DZ=NORM*FOCIL*FOCI* &
               (DETA*ZGD*IHETA2+DZETA*EGD*IHZETA2)
          TMPFX=TMPFX-DX*MQ(II)
          TMPFY=TMPFY-DY*MQ(II)
          TMPFZ=TMPFZ-DZ*MQ(II)
       ENDDO
       RXNBFX(MM)=TMPFX*CCC
       RXNBFY(MM)=TMPFY*CCC
       RXNBFZ(MM)=TMPFZ*CCC
    ENDDO ll_loop
    !
    RETURN
  END SUBROUTINE PROL_GSBP3
  !
  SUBROUTINE PROL_MIJ(NTPOL,NORDER,DCEL,NPGD,IIP0, &
       ETA,ZETA,CPHI,SPHI,MAJOR,MINOR, &
       LSTPL,LSTPM,BNORM, &
       PHIP,PHIW,MIJ,CGSCAL)
    !-----------------------------------------------------------------------
    !     calculate the matrix M(ij)
    !
    use chm_kinds
    use number
    implicit none
    REAL(CHM_REAL4)  PHIP(*),PHIW(*)
    real(chm_real)  ETA(*),ZETA(*),CPHI(*),SPHI(*),BNORM(*),MIJ(*)
    real(chm_real)  MAJOR,MINOR,CGSCAL,DCEL
    INTEGER NORDER,NTPOL,NPGD
    INTEGER LSTPL(*),LSTPM(*),IIP0(*)
    ! local
    real(chm_real)  DCEL3,NORM,EGD,ZGD,CP,SP,FOCI
    real(chm_real)  FACTOR
    INTEGER I,L,M,N,IJ,IP0
    !
    FOCI=SQRT(MAJOR*MAJOR-MINOR*MINOR)
    DCEL3 = DCEL*DCEL*DCEL
    DO N=1,NORDER
       L=LSTPL(N)
       M=LSTPM(N)
       NORM=BNORM(N)
       IJ=(N-1)*NTPOL+NORDER
       MIJ(IJ)=ZERO

       IF(M.EQ.0) THEN
          IF(L.EQ.0) THEN
             FACTOR=ONE
             IF(NORDER.EQ.1) FACTOR=CGSCAL
             DO I=1,NPGD
                IP0=IIP0(I)
                MIJ(IJ)=MIJ(IJ)+ &
                     NORM*DCEL3*(PHIW(IP0)-PHIP(IP0))*FACTOR
             ENDDO
          ELSE
             DO I=1,NPGD
                IP0=IIP0(I)
                EGD=ETA(I)
                ZGD=ZETA(I)
                MIJ(IJ)=MIJ(IJ)+ &
                     NORM*FOCI**L*ALPOL(L,M,EGD)*ALPOL(L,M,ZGD)* &
                     DCEL3*(PHIW(IP0)-PHIP(IP0))
             ENDDO
          ENDIF
       ELSEIF(M.GT.0) THEN
          DO I=1,NPGD
             IP0=IIP0(I)
             EGD=ETA(I)
             ZGD=ZETA(I)
             CP=CPHI(I)
             MIJ(IJ)=MIJ(IJ)+ &
                  NORM*FOCI**L*ALPOL(L,M,EGD)*ALPOL(L,M,ZGD)* &
                  COSMPHI(M,CP)*DCEL3*(PHIW(IP0)-PHIP(IP0))
          ENDDO
       ELSEIF(M.LT.0) THEN
          M=-M
          DO I=1,NPGD
             IP0=IIP0(I)
             EGD=ETA(I)
             ZGD=ZETA(I)
             CP=CPHI(I)
             SP=SPHI(I)
             MIJ(IJ)=MIJ(IJ)+ &
                  NORM*FOCI**L*ALPOL(L,M,EGD)*ALPOL(L,M,ZGD)* &
                  SINMPHI(M,CP,SP)*DCEL3*(PHIW(IP0)-PHIP(IP0))
          ENDDO
       ENDIF
    ENDDO
    !
    RETURN
  END SUBROUTINE PROL_MIJ
  !
  SUBROUTINE PROL_MAYER(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,DCEL, &
       XBCEN,YBCEN,ZBCEN, &
       FKAPA,EPSX,EPSY,EPSZ, &
       RRXCEN,RRYCEN,RRZCEN,MAJOR,MINOR)
    !-----------------------------------------------------------------------
    !     vacuum environment in region B.
    !
    use chm_kinds
    use number
    implicit none
    REAL(CHM_REAL4)  EPSX(*),EPSY(*),EPSZ(*),FKAPA(*)
    real(chm_real)  DCEL,TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN
    real(chm_real)  RRXCEN,RRYCEN,RRZCEN,MAJOR,MINOR
    INTEGER NCLx,NCLy,NCLz
    ! local
    real(chm_real)  XG,YG,ZG,XG2,YG2,ZG2,DSQ1,XEPS,YEPS,ZEPS,SR2
    real(chm_real)  MAJOR2,MINOR2
    INTEGER IG,JG,KG
    INTEGER IP0,IP0X,IP0Y,NCYZ
    INTEGER NFIL,NFILZ,IXCEN,IYCEN,IZCEN,JX1,JX2,JY1,JY2,JZ1,JZ2

    NFIL =INT(MINOR/DCEL)+2
    NFILZ=INT(MAJOR/DCEL)+2
    IXCEN=INT((RRXCEN+TRANX-XBCEN)/DCEL)+1
    IYCEN=INT((RRYCEN+TRANY-YBCEN)/DCEL)+1
    IZCEN=INT((RRZCEN+TRANZ-ZBCEN)/DCEL)+1
    JX1=IXCEN-NFIL
    JX2=IXCEN+NFIL+1
    JY1=IYCEN-NFIL
    JY2=IYCEN+NFIL+1
    JZ1=IZCEN-NFILZ
    JZ2=IZCEN+NFILZ+1

    ! For dielectric constants and Debye-Huckel screening factors
    ! NOTE: dielectric constants are located at the middle point between grids
    MAJOR2=MAJOR*MAJOR
    MINOR2=MINOR*MINOR
    NCYZ= NCLy*NCLz
    DO IG=JX1,JX2
       XG=DCEL*(IG-1)-TRANX+XBCEN-RRXCEN
       XG2=XG*XG
       IP0X=(IG-1)*NCyz
       DO JG=JY1,JY2
          YG=DCEL*(JG-1)-TRANY+YBCEN-RRYCEN
          YG2=YG*YG
          IP0Y=(JG-1)*NCLz+IP0X
          DO KG=JZ1,JZ2
             ZG=DCEL*(KG-1)-TRANZ+ZBCEN-RRZCEN
             ZG2=ZG*ZG
             IP0 = IP0Y + KG
             IF(FKAPA(IP0).ne.ZERO)THEN
                DSQ1=(XG2+YG2)/MINOR2+ZG2/MAJOR2
                IF(DSQ1.LE.ONE+RSMALL) FKAPA(IP0)=ZERO
             ENDIF
             IF(EPSX(IP0).ne.ONE)THEN
                XEPS=XG+HALF*DCEL
                DSQ1=(XEPS*XEPS+YG2)/MINOR2+ZG2/MAJOR2
                IF(DSQ1.LE.ONE+RSMALL) EPSX(IP0)=ONE
             ENDIF
             IF(EPSY(IP0).ne.ONE)THEN
                YEPS=YG+HALF*DCEL
                DSQ1=(XG2+YEPS*YEPS)/MINOR2+ZG2/MAJOR2
                IF(DSQ1.LE.ONE+RSMALL) EPSY(IP0)=ONE
             ENDIF
             IF(EPSZ(IP0).ne.ONE)THEN
                ZEPS=ZG+HALF*DCEL
                DSQ1=(XG2+YG2)/MINOR2+ZEPS*ZEPS/MAJOR2
                IF(DSQ1.LE.ONE+RSMALL) EPSZ(IP0)=ONE
             ENDIF
          ENDDO
       ENDDO
    ENDDO
    !
    RETURN
  END SUBROUTINE PROL_MAYER
  !
  SUBROUTINE PROL_TRIG(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,DCEL, &
       XBCEN,YBCEN,ZBCEN, &
       NPGD,IIP0,ETA,ZETA,CPHI,SPHI,MAJOR,MINOR, &
       RRXCEN,RRYCEN,RRZCEN,JX1,JX2,JY1,JY2,JZ1,JZ2)
    !-----------------------------------------------------------------------
    !     calculate the trigonometric functions of each grid points
    !     inside region of interest
    !
    use chm_kinds
    use consta
    use number
    implicit none
    real(chm_real)  MAJOR,MINOR,RRXCEN,RRYCEN,RRZCEN
    real(chm_real)  DCEL,TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN
    real(chm_real)  ETA(*),ZETA(*),CPHI(*),SPHI(*)
    INTEGER NCLx,NCLy,NCLz,IIP0(*)
    INTEGER JX1,JX2,JY1,JY2,JZ1,JZ2,NPGD
    ! local
    real(chm_real)  XG,YG,ZG,XG2,YG2,ZG2
    real(chm_real)  MAJOR2,MINOR2,R,R1,R2,FOCI,EGD,ZGD
    INTEGER NC3,NCYZ,IG,JG,KG,IP0X,IP0Y,IP0
    !
    NCYZ=NCLy*NCLz
    NC3 =NCYZ*NCLx
    FOCI=SQRT(MAJOR*MAJOR-MINOR*MINOR)
    MAJOR2=MAJOR*MAJOR
    MINOR2=MINOR*MINOR

    ! calculate the trigonometric functions of each grid points
    ! inside region of interest
    NPGD=0
    DO IG=JX1,JX2
       XG=DCEL*(IG-1)-TRANX+XBCEN-RRXCEN
       XG2=XG*XG
       IP0X=(IG-1)*NCyz
       DO JG=JY1,JY2
          YG=DCEL*(JG-1)-TRANY+YBCEN-RRYCEN
          YG2=YG*YG
          IP0Y=(JG-1)*NCLz+IP0X
          DO KG=JZ1,JZ2
             ZG=DCEL*(KG-1)-TRANZ+ZBCEN-RRZCEN
             ZG2=ZG*ZG
             IP0 = IP0Y + KG
             IF((XG2+YG2)/MINOR2+ZG2/MAJOR2.LE.ONE+RSMALL) THEN
                !     in the origin
                IF(XG2+YG2+ZG2.LT.RSMALL) THEN
                   NPGD=NPGD+1
                   IIP0(NPGD)=IP0
                   ETA(NPGD)=ONE
                   ZETA(NPGD)=ZERO
                   CPHI(NPGD)=ZERO
                   SPHI(NPGD)=ZERO
                   !     in the Z axis
                ELSEIF(XG.GT.-RSMALL.AND.XG.LT.RSMALL.AND. &
                     YG.GT.-RSMALL.AND.YG.LT.RSMALL) THEN
                   NPGD=NPGD+1
                   IIP0(NPGD)=IP0
                   R1=ABS(ZG+FOCI)
                   R2=ABS(ZG-FOCI)
                   ETA(NPGD)=(R1+R2)/(TWO*FOCI)
                   ZETA(NPGD)=(R1-R2)/(TWO*FOCI)
                   CPHI(NPGD)=ZERO
                   SPHI(NPGD)=ZERO
                ELSE
                   NPGD=NPGD+1
                   IIP0(NPGD)=IP0
                   R1=SQRT(XG2+YG2+(ZG+FOCI)*(ZG+FOCI))
                   R2=SQRT(XG2+YG2+(ZG-FOCI)*(ZG-FOCI))
                   EGD=(R1+R2)/(TWO*FOCI)
                   ZGD=(R1-R2)/(TWO*FOCI)
                   ETA(NPGD)=EGD
                   ZETA(NPGD)=ZGD
                   R=SQRT((EGD**2-ONE)*(ONE-ZGD**2))
                   CPHI(NPGD)=XG/R/FOCI
                   SPHI(NPGD)=YG/R/FOCI
                ENDIF
             ENDIF
          ENDDO
       ENDDO
    ENDDO
    !
    RETURN
  END SUBROUTINE PROL_TRIG
  !
  SUBROUTINE PROL_CHC(NORDER,NCEL3,DCEL,NPGD,IIP0, &
       ETA,ZETA,CPHI,SPHI,MAJOR,MINOR, &
       LSTPL,LSTPM,BNORM,CHC,CGSCAL)
    !-----------------------------------------------------------------------
    !     store the basis functions (shperical harmonics) in ICHC (CDEN) array
    !     NOTE: CHC is a charge (not density) arrary and containes 4*pi/h
    !
    use chm_kinds
    use consta
    use number
    implicit none
    REAL(CHM_REAL4)  CHC(*)
    real(chm_real)  ETA(*),ZETA(*),CPHI(*),SPHI(*),BNORM(*)
    real(chm_real)  MAJOR,MINOR,CGSCAL,DCEL
    INTEGER NPGD,NORDER,NCEL3
    INTEGER IIP0(*),LSTPL(*),LSTPM(*)
    ! local
    real(chm_real)  DCEL3,NORM,EGD,ZGD,CP,SP,FOCI
    INTEGER I,L,M,IP0
    !      real(chm_real)  R,XG,YG,ZG
    !
    L   =LSTPL(NORDER)
    M   =LSTPM(NORDER)
    NORM=BNORM(NORDER)
    DCEL3=DCEL*DCEL*DCEL
    FOCI=SQRT(MAJOR*MAJOR-MINOR*MINOR)

    ! initialize CHC array
    DO I=1,NCEL3
       CHC(I)=0.0
    ENDDO

    ! the basis functions are normalized inside region of interest
    IF(M.EQ.0) THEN
       IF(L.EQ.0) THEN
          DO I=1,NPGD
             IP0=IIP0(I)
             CHC(IP0)=NORM*DCEL3*TWO*TWOPI/DCEL/CGSCAL
          ENDDO
       ELSE
          DO I=1,NPGD
             IP0=IIP0(I)
             EGD=ETA(I)
             ZGD=ZETA(I)
             CHC(IP0)=NORM*FOCI**L*ALPOL(L,M,EGD)*ALPOL(L,M,ZGD)* &
                  DCEL3*TWO*TWOPI/DCEL
          ENDDO
       ENDIF
    ELSEIF(M.GT.0) THEN
       DO I=1,NPGD
          IP0=IIP0(I)
          EGD=ETA(I)
          ZGD=ZETA(I)
          CP=CPHI(I)
          CHC(IP0)=NORM*FOCI**L*ALPOL(L,M,EGD)*ALPOL(L,M,ZGD)* &
               COSMPHI(M,CP)*DCEL3*TWO*TWOPI/DCEL
       ENDDO
    ELSEIF(M.LT.0) THEN
       M=-M
       DO I=1,NPGD
          IP0=IIP0(I)
          EGD=ETA(I)
          ZGD=ZETA(I)
          CP=CPHI(I)
          SP=SPHI(I)
          CHC(IP0)=NORM*FOCI**L*ALPOL(L,M,EGD)*ALPOL(L,M,ZGD)* &
               SINMPHI(M,CP,SP)*DCEL3*TWO*TWOPI/DCEL
       ENDDO
    ENDIF
    !
    !      DO I=1,NPGD
    !         IP0=IIP0(I)
    !         EGD=ETA(I)
    !         ZGD=ZETA(I)
    !         CP=CPHI(I)
    !         SP=SPHI(I)
    !         R=SQRT((EGD**2-ONE)*(ONE-ZGD**2))
    !         XG=FOCI*R*CP*10.0
    !         YG=FOCI*R*SP*10.0
    !         ZG=FOCI*EGD*ZGD*10.0
    !         IF(CHC(ip0).GT.RSMALL) THEN
    !            write(70+NORDER,101)
    !     $           'ATOM',I,' POL ARG     1',
    !     $           XG,YG,ZG,
    !     $           CHC(ip0)*100.,'  1.00      LPOL'
    !         ELSEIF(CHC(ip0).LT.-RSMALL) THEN
    !            write(70+NORDER,101)
    !     $           'ATOM',I,' POL GLU     2',
    !     $           XG,YG,ZG,
    !     $           CHC(ip0)*100.,' 10.00      LPOL'
    !         ELSE
    !            write(70+NORDER,101)
    !     $           'ATOM',I,' POL CYS     3',
    !     $           XG,YG,ZG,
    !     $           CHC(ip0)*100.,' 10.00      LPOL'
    !         ENDIF
    !      ENDDO
    ! 101  format(a,2x,i5,1x,a,4x,3f8.3,f6.2,a)
    !
    RETURN
  END SUBROUTINE PROL_CHC
  !
  SUBROUTINE PROL_NORM(NMPOL,BNORM,MAJOR,MINOR)
    !-----------------------------------------------------------------------
    !     calculate the normalization constants of the basis functions
    !     in a prolate inner region
    !
    use chm_kinds
    use number
    use consta
    implicit none
    INTEGER NMPOL
    real(chm_real)  BNORM(*),MAJOR,MINOR
    ! local
    INTEGER L,M,NORDER
    real(chm_real)  FOCI,ETAMAX,TMPETA1,TMPETA2
    real(chm_real)  ETAPART1,ETAPART2,ZETAPART1,ZETAPART2
    real(chm_real)  FACTO1,FACTO2,FACTO3,FACTO4,NLM2
    !
#if KEY_G95==1
    procedure (), pointer :: func
#endif
    !
    ! always the same order in spherical harmonics
    FOCI=SQRT(MAJOR*MAJOR-MINOR*MINOR)
    ETAMAX=MAJOR/FOCI
    TMPETA1=ETA1(0,0,ETAMAX)    ! it is necessary to make QSIMP work
    TMPETA2=ETA2(0,0,ETAMAX)
    NORDER=0
    DO L=0,NMPOL-1
       NORDER=NORDER+1
#if KEY_G95==1
       func=>eta1
       CALL QSIMP(func,L,0,ONE,ETAMAX,ETAPART1)
       func=>eta2
       CALL QSIMP(func,L,0,ONE,ETAMAX,ETAPART2)
#else
       CALL QSIMP(ETA1,L,0,ONE,ETAMAX,ETAPART1)
       CALL QSIMP(ETA2,L,0,ONE,ETAMAX,ETAPART2)
#endif
       ZETAPART1=TWO/(TWO*L+ONE)
       ZETAPART2=TWO/(TWO*L+ONE)**2* &
            (ONE/(TWO*L+THREE)*(L+2)*(L+1)+ONE/(TWO*L-ONE)*L*(L-1))
       NLM2=TWOPI*FOCI**(2*L+3)* &
            (ETAPART1*ZETAPART1+ETAPART2*ZETAPART2)
       BNORM(NORDER)=SQRT(ONE/NLM2)
       DO M=1,L
          NORDER=NORDER+1
#if KEY_G95==1
          func=>eta1
          CALL QSIMP(func,L,M,ONE,ETAMAX,ETAPART1)
          func=>eta2
          CALL QSIMP(func,L,M,ONE,ETAMAX,ETAPART2)
#else
          CALL QSIMP(ETA1,L,M,ONE,ETAMAX,ETAPART1)
          CALL QSIMP(ETA2,L,M,ONE,ETAMAX,ETAPART2)
#endif
          FACTO1=FACTORI(L+M)
          FACTO2=FACTORI(L-M)
          FACTO3=FACTORI(L+M+2)
          ZETAPART1=TWO/(TWO*L+ONE)*FACTO1/FACTO2
          IF(M.LE.L-2) THEN
             FACTO4=FACTORI(L-M-2)
             ZETAPART2=TWO/(TWO*L+ONE)**2* &
                  (ONE/(TWO*L+THREE)*FACTO3/FACTO2+ &
                  ONE/(TWO*L-ONE)*FACTO1/FACTO4)
          ELSE
             ZETAPART2=TWO/(TWO*L+ONE)**2* &
                  (ONE/(TWO*L+THREE)*FACTO3/FACTO2)
          ENDIF
          NLM2=PI*FOCI**(2*L+3)* &
               (ETAPART1*ZETAPART1+ETAPART2*ZETAPART2)
          BNORM(NORDER)=SQRT(ONE/NLM2)
          NORDER=NORDER+1
          BNORM(NORDER)=BNORM(NORDER-1)
       ENDDO
    ENDDO
    !
    RETURN
  END SUBROUTINE PROL_NORM
  !
  SUBROUTINE QSIMP(FUNC,L,M,A,B,S)
    !------------------------------------------------------------------------
    !     Returns as S the integral of the function FUNC from a and b
    !     using Simpson's rule
    !
    use chm_kinds
    use number
    use consta
    implicit none
    INTEGER L,M
    real(chm_real)  A,B,S
    real(chm_real), PARAMETER :: EPS=1.D-10
    INTEGER, parameter :: JMAX=20
#if KEY_G95==0
    real(chm_real) :: FUNC
    EXTERNAL FUNC
#else
    procedure (), pointer :: func
#endif
    ! local
    INTEGER J
    real(chm_real)  OS,OST,ST
    !
    S=ZERO
    OST=-1.D30
    OS=OST
    DO J=1,JMAX
       CALL TRAPZD(FUNC,L,M,A,B,ST,J)
       S=(4.D0*ST-OST)/3.D0
       IF(J.GT.5) THEN
          IF(ABS(S-OS).LT.EPS*ABS(OS).OR. &
               (S.LT.RSMALL.AND.OS.LT.RSMALL)) RETURN
       ENDIF
       OS=S
       OST=ST
    ENDDO
    !
    CALL WRNDIE(-1,'<QSIMP>','TOO MANY STEPS IN QSIMP')
  END SUBROUTINE QSIMP
  !
  SUBROUTINE TRAPZD(FUNC,L,M,A,B,S,N)
    !------------------------------------------------------------------------
    !     This routine computes the nth stage of refinement of an extended
    !     trapezoidal rule.
    !
    use chm_kinds
    IMPLICIT NONE
    INTEGER N,L,M
    real(chm_real)  A,B,S,FUNC
    EXTERNAL FUNC
    ! local
    INTEGER IT,J
    real(chm_real)  DX,SUM,X
    !
    IF(N.EQ.1) THEN
       S=.5D0*(B-A)*(FUNC(L,M,A)+FUNC(L,M,B))
    ELSE
       IT=2**(N-2)
       DX=(B-A)/IT
       X=A+.5D0*DX
       SUM=0.D0
       DO J=1,IT
          SUM=SUM+FUNC(L,M,X)
          X=X+DX
       ENDDO
       S=.5D0*(S+DX*SUM)
    ENDIF
    !
    RETURN
  END SUBROUTINE TRAPZD
  !
  FUNCTION ETA1(L,M,X) result(eta1_rslt)
    !------------------------------------------------------------------------
    !
    use chm_kinds
    use number
    implicit none
    real(chm_real) :: eta1_rslt
    integer l,m
    real(chm_real)  x,tmpalpol

    TMPALPOL=ALPOL(L,M,X)
    ETA1_rslt=TMPALPOL*TMPALPOL*(X*X-one)

    return
  end FUNCTION ETA1
  !
  FUNCTION ETA2(L,M,X) result(eta2_rslt)
    !------------------------------------------------------------------------
    !
    use chm_kinds
    implicit none
    real(chm_real) :: eta2_rslt
    integer l,m
    real(chm_real)  x,tmpalpol

    TMPALPOL=ALPOL(L,M,X)
    ETA2_rslt=TMPALPOL*TMPALPOL

    return
  end FUNCTION ETA2
  !
  FUNCTION SINCOS(L,M,X) result(sincos_rslt)
    !------------------------------------------------------------------------
    !     (cos x)**l*(sin x)**m
    !
    use chm_kinds
    IMPLICIT NONE
    real(chm_real) :: sincos_rslt
    INTEGER L,M
    real(chm_real)  X,TMPSIN,TMPCOS
    !
    TMPSIN=SIN(X)
    TMPCOS=COS(X)
    SINCOS_rslt=TMPCOS**L*TMPSIN**M
    !
    RETURN
  END FUNCTION SINCOS
  !
  SUBROUTINE COSMPHI2(M,C0,AC)
    !------------------------------------------------------------------------
    !     cos(M*phi) calculation (M > 0)
    !
    use chm_kinds
    implicit none
    INTEGER   M,I
    real(chm_real)    C0,C2,C3,AC(0:M)
    !!LNI, F95 conversion. Added check on M.
    !!      IF(M < 4) WRITE(*,*) 'COSMPHI2: M=',M
    AC(0)=1.0
    IF(M > 0) AC(1)=C0
    IF(M > 1) AC(2)=2.0*C0*C0-1.D0
    IF(M > 2) AC(3)=(2.0*AC(2)-1.0)*C0
    DO I=4,M
       AC(I)=2.D0*C0*AC(I-1)-AC(I-2)
    ENDDO
    !
    RETURN
  END SUBROUTINE COSMPHI2
  !
  SUBROUTINE SINMPHI2(M,C0,S0,AS)
    !------------------------------------------------------------------------
    !     sin(M*phi) calculation (M > 0)
    !
    use chm_kinds
    implicit none
    INTEGER   M,I
    real(chm_real)    C0,S0,S2,S3,AS(0:M)
    !
    !!LNI, F95 conversion. Added check on M.
    !!      IF(M < 4) WRITE(*,*) 'SINMPHI2: M=',M
    AS(0)=0.0
    IF(M > 0) AS(1)=S0
    IF(M > 1) AS(2)=2.0*C0*S0
    IF(M > 2) AS(3)=2.0*C0*AS(2)-S0
    DO I=4,M
       AS(I)=2.0*C0*AS(I-1)-AS(I-2)
    ENDDO
    !
    RETURN
  END SUBROUTINE SINMPHI2
  !
  SUBROUTINE DLPOL2(NPOL,COOR,SCAL,LP,DLP)
    !------------------------------------------------------------------------
    !     Legendre Polynomials
    !
    use chm_kinds
    implicit none
    INTEGER NPOL
    real(chm_real)  COOR,SCAL,LP(NPOL),DLP(NPOL-1)
    ! local
    INTEGER N
    real(chm_real)  X
    !
    X=COOR*SCAL
    DLP(1)= 0.0D0
    DLP(2)= 1.0D0
    DLP(3)= 3.0D0*X
    DO N=3,NPOL-1
       DLP(N+1)=X*DLP(N)+N*LP(N)
    ENDDO
    DO N=1,NPOL
       DLP(N)=DLP(N)*SCAL
    ENDDO
    !
    RETURN
  END SUBROUTINE DLPOL2
  !
  SUBROUTINE LPOL2(NPOL,COOR,SCAL,LP)
    !------------------------------------------------------------------------
    !     Legendre Polynomials
    !
    use chm_kinds
    implicit none
    INTEGER NPOL
    real(chm_real)  COOR,SCAL,LP(NPOL)
    ! local
    INTEGER N
    real(chm_real)  X
    !
    X=COOR*SCAL
    LP(1)= 1.0D0
    LP(2)= X
    LP(3)= 0.5D0*(3.0D0*X*X-1.0D0)
    DO N=3,NPOL-1
       LP(N+1)=((2.*N-1)*X*LP(N)-(N-1.)*LP(N-1))/N
    ENDDO
    !
    RETURN
  END SUBROUTINE LPOL2
  !
  SUBROUTINE DALPOL2(LMAX,X,AP,ADP)
    !------------------------------------------------------------------------
    !     Derivatives of Associate Legendre Polynomials (From Smythe's BOOK)
    !     This is different from FUNCTION DALPOL because we don't
    !     consider x > 1 here.
    !
    use chm_kinds
    use number
    implicit none
    integer lmax
    real(chm_real)  x,ap(0:lmax,0:lmax),adp(0:lmax,0:lmax)
    ! local
    integer l,m,mmax
    real(chm_real)  alpol,fact
    !
    IF(ABS(X).EQ.ONE) THEN
       DO L=0,LMAX
          ADP(L,0)=ZERO
          IF(X.EQ.ONE) ADP(L,0)=L*(L+1)/TWO
          IF(X.EQ.MINONE) ADP(L,0)=(MINONE)**(L+1)*L*(L+1)/TWO
          DO M=1,L
             ADP(L,M)=ZERO
          ENDDO
       ENDDO
       RETURN
    ENDIF

    IF(X.GT.ONE) THEN
       FACT=ONE/SQRT(X*X-ONE)
       DO L=0,LMAX
          DO M=0,L-1
             ADP(L,M)=FACT*(AP(L,M+1)+FACT*M*X*AP(L,M))
          ENDDO
          ADP(L,L)=FACT*FACT*M*X*AP(L,M)
       ENDDO
    ELSE
       FACT=ONE/SQRT(ONE-X*X)
       DO L=0,LMAX
          DO M=0,L-1
             ADP(L,M)=FACT*(AP(L,M+1)-FACT*M*X*AP(L,M))
          ENDDO
          ADP(L,L)=-FACT*FACT*M*X*AP(L,M)
       ENDDO
    ENDIF
    !
    RETURN
  END SUBROUTINE DALPOL2
  !
  SUBROUTINE ALPOL2(LMAX,X,AP)
    !------------------------------------------------------------------------
    !     The Associate Legendre Polynomials
    !     (Numerical recipes in Fortran 77: 6.8. Spherical Harmonics)
    !     This is different from FUNCTION ALPOL because we don't
    !     consider x > 1 here.
    !
    use chm_kinds
    use number
    implicit none
    integer lmax
    real(chm_real)  x,ap(0:lmax,0:lmax)
    ! local
    integer i,ll,l,m,mmax
    real(chm_real)  fact,pll,pmm,pmmp1,somx2
    !
    mmax=lmax

    ! compute p(m,m)
    AP(0,0)=one
    somx2=sqrt((one-x)*(one+x))
    fact=one
    do m=1,mmax
       AP(m,m)=AP(m-1,m-1)*fact*somx2
       fact=fact+two
    enddo

    ! compute p(m+1,m)
    !!LNI change during F95 conversion; original code fails simple bounds check
    !      do m=0,mmax
    do m=0,mmax-1
       AP(m+1,m)=x*(2*m+1)*AP(m,m)
    enddo

    ! compute p(l,m)
    do l=2,lmax
       do m=0,l-2
          AP(l,m)=(x*(2*l-1)*AP(l-1,m)-(l+m-1)*AP(l-2,m))/(l-m)
       enddo
    enddo
    !
    RETURN
  END SUBROUTINE ALPOL2
  !
  SUBROUTINE RPOWERL2(L,R,AR)
    !------------------------------------------------------------------------
    !     R^L calculation (L >= 0)
    !
    use chm_kinds
    implicit none
    INTEGER   L,I
    real(chm_real)    R,AR(0:L)
    !
    AR(0)=1.D0
    DO I=1,L
       AR(I)=AR(I-1)*R
    ENDDO
    RETURN
  END SUBROUTINE RPOWERL2

#endif /* (gsbp)*/

  !-----------------------------------------------------------------------
  !QC: attach pbeqscc.src below to avoid variable transfers
  !CHARMM Element source/misc/pbeqscc.src $Revision: 1.27 $
#if KEY_IF==1 || KEY_PBEQ==1
  
#endif
#if KEY_SCCDFTB==1
  !
  !      XIAO_QC_UW0609: Subroutines needed for computing SCC-DFTB in the presence
  !      of the pb contribution.
  !      The "difficulty" is that the PB contribution has to be included
  !      in the SCF procedure. With SCC-DFTB, this is very straightforward
  !      since the QM/PB interactions can be described with the Mulliken
  !      charges on the QM atoms, as in SCC-DFTB/MM interactions.

  SUBROUTINE SCCPB_MM(NATOM,X,Y,Z,CG,ICALL,QPRIN, &
       RXNEX,RXNEY,RXNEZ)
    !
    use chm_kinds
    !  use heapm (QC: corrected 12/09)
    !  use stackm (QC: corrected 12/09)
    !(QC: corrected 12/09)
    use memory
    use number
    use stream
    use dimens_fcm
    use consta
    use timerm
    use sccdftb
    use sccpb
    use machutil,only:wrttim
    !
    implicit none
    !
    !-----------------------------------------------------------------------------

    INTEGER NATOM,ICALL
    REAL(CHM_REAL)  X(*),Y(*),Z(*),CG(*)
    real(chm_real)  rxnex(*),rxney(*),rxnez(*)
    LOGICAL QPRIN
    !
    !!    XIAO ZHU: CALCULATE the charge independent reaction field, 
    !                which is induced by CHARMM charges

    !         integer,         allocatable, dimension (:) :: LISTR
    !         real(chm_real),  allocatable, dimension (:) :: FISTR
    real(chm_real4), allocatable, dimension (:) :: IPHIB

    !         call chmalloc('pbeqscc.src','SCCPB_MM','LISTR',NATOM,intg=LISTR)
    !         call chmalloc('pbeqscc.src','SCCPB_MM','FISTR',NATOM,crl=FISTR)
    call chmalloc('pbeqscc.src','SCCPB_MM','IPHIB',    1,cr4=IPHIB)

    !         write(*,*) "SCCPB_MM> key arrays allocated, calling mayer..."

    !     -----------------------------------------------------------------
    !     In vacuum
    !     -----------------------------------------------------------------
    CALL MAYER(NTPRP,LSTPRP,ONE,EPSP,ZERO, &
         !                                  epsw epsp kappa2
         X,Y,Z,CG,PBRAD,NCLX,NCLY,NCLZ,DCEL,WATR,IONR,MAY, &
         MAYX,MAYY,MAYZ,SWIN,ICHC, &
         IPHIPEX, &
         !                                 -----                 -----
         ZERO,TMEMB,ZMEMB,ONE,NIMGB, &
         !          vmemb            epsm
         HTMEMB,ONE,HDENS,POSALP,NEGALP,QCHRG,POFFST,NOFFST, &
         !                 epsh
         DROPLET,ONE,XDROPLET,YDROPLET,ZDROPLET,Qdtom,Qdkap, &
         !                 epsd
         ONE,LXMAX,LYMAX,LZMAX,LXMIN,LYMIN,LZMIN,Qbtom,Qbkap, &
         !          epsb
         ONE,Xcyln,Ycyln,Zcyln,Rcyln,Hcyln,Qctom,Qckap, &
         !          epsc
         ONE,Xcone,Ycone,Zcone,Ax,By,Cz,Qectom,Qeckap, &
         !          epsec
               -1,-1,&
         TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
         IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
         QKEEP,qreen,qsmth,QBSPL,QA,QB, &
         !          .true.,qreen,qsmth,QBSPL,QA,QB,
         !                QREEN QSMTH
         RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
         RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
         MAJOR,MINOR,QPROLATE, &
         QINTBP,QZERO,.false.,IPHIB,QPBC,QNPBC, &
         !          -----  -----  QFOCUS
         QNONLINEAR,QPARTLINEAR,QFKAP, &
         IPOX,IPOY,IPOZ,MAPT, &
         NATOM) ! QC: 12/09 following new pbeq
    !          LISTR,FISTR)

    !         write(*,*) "SCCPB_MM> returned from vacuum mayer "

    IF (TIMER.GT.1 .and. ICALL.eq.0) &
         CALL WRTTIM('Grid parameters preparation in vacuum times:')

    IF(QFMG) THEN
       CALL PBEQ2(NATOM,X,Y,Z,CG,.true.,QPRIN)
       IF (TIMER.GT.1 .and. ICALL.eq.0) &
            CALL WRTTIM('FMG iteration times:')
    ELSE
       IF(QOLDPB)THEN
          CALL OLDPBEQ1(MAXITS,DOME,DEPS,NCLX,NCLY,NCLZ, &
               IPHIPEX,MAYX,MAYY,MAYZ, &
               !                     -----
               ICHC,MAY,TMEMB, &
               .false.,.false.,QPBC,QNPBC,QPRIN)
          !                QOSOR   QFOCUS
       ELSE
          CALL PBEQ1(MAXITS,DOME,DEPS,ONE,EPSP,ONE,ZERO, &
               !                                       ---         kappa2
               NCLX,NCLY,NCLZ,IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
               IPHIPEX,MAYX,MAYY,MAYZ, &
               !                     -----
               ICHC,MAY,TMEMB, &
               .false.,.false.,QPBC,QNPBC,QPRIN)
          !                QOSOR   QFOCUS
       ENDIF
       IF (TIMER.GT.1 .and. ICALL.eq.0) &
            CALL WRTTIM('SOR iteration in vacuum times:')
    ENDIF
    !----------------------------------------------------------------------------

    !----------------------------------------------------------------------------
    !     In solution
    !----------------------------------------------------------------------------
    IF(QPRIN) WRITE(OUTU,'(/,3x,A,f8.3)') &
         'Constructing all space-dependent functions:1 EPS =', epsw

    !
    !     write(*,*) "SCCPB_MM> calling solution mayer "
    CALL MAYER(NTPRP,LSTPRP,EPSW,EPSP,KAPPA2, &
         !                                   ----
         X,Y,Z,CG,PBRAD,NCLX,NCLY,NCLZ,DCEL,WATR,IONR,MAY, &
         MAYX,MAYY,MAYZ,SWIN,ICHC, &
         IPHIWEX, &
         !                                           -----                 -----
         VMEMB,TMEMB,ZMEMB,EPSM,NIMGB, &
         HTMEMB,EPSH,HDENS,POSALP,NEGALP,QCHRG,POFFST,NOFFST, &
         DROPLET,EPSD,XDROPLET,YDROPLET,ZDROPLET,Qdtom,Qdkap, &
         EPSB,LXMAX,LYMAX,LZMAX,LXMIN,LYMIN,LZMIN,Qbtom,Qbkap, &
         EPSC,Xcyln,Ycyln,Zcyln,Rcyln,Hcyln,Qctom,Qckap, &
         EPSEC,Xcone,Ycone,Zcone,Ax,By,Cz,Qectom,Qeckap, &
               -1,-1,&
         TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
         IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
         QKEEP,qreen,qsmth,QBSPL,QA,QB, &
         !          .true.,qreen,qsmth,QBSPL,QA,QB,
         !                QREEN QSMTH
         RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
         RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
         MAJOR,MINOR,QPROLATE, &
         QINTBP,QZERO,.false.,IPHIB,QPBC,QNPBC, &
         !          ------,----- QFOCUS
         QNONLINEAR,QPARTLINEAR,QFKAP, &
         IPOX,IPOY,IPOZ,MAPT, &
         NATOM) ! QC: 12/09 following new pbeq
    !          LISTR,FISTR)

    !        write(*,*) "done solution mayer ......"

    IF (TIMER.GT.1 .and. ICALL.eq.0) &
         CALL WRTTIM('Grid parameters preparation in solution times:')

    IF(QFMG) THEN
       CALL PBEQ2(NATOM,X,Y,Z,CG,.false.,QPRIN)
       IF (TIMER.GT.1 .and. ICALL.eq.0) &
            CALL WRTTIM('FMG iteration times:')
    ELSE
       IF(QOLDPB)THEN
          CALL OLDPBEQ1(MAXITS,DOME,DEPS,NCLX,NCLY,NCLZ, &
               IPHIWEX,MAYX,MAYY,MAYZ, &
               !                     -----
               ICHC,MAY,TMEMB, &
               QOSOR,.false.,QPBC,QNPBC,QPRIN)
          !                QOSOR   QFOCUS
       ELSE
          CALL PBEQ1(MAXITS,DOME,DEPS,EPSW,EPSP,EPSM,KAPPA2, &
               !                                       ----
               NCLX,NCLY,NCLZ,IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
               IPHIWEX,MAYX,MAYY,MAYZ, &
               !                     -----
               ICHC,MAY,TMEMB, &
               QOSOR,.false.,QPBC,QNPBC,QPRIN)
          !                      QFOCUS
       ENDIF
       IF (TIMER.GT.1 .and. ICALL.eq.0) &
            CALL WRTTIM('SOR iteration in solution times:')
    ENDIF
    !     QC: left here? 12/09
    !         call chmdealloc('pbeqscc.src','SCCPB_MM','LISTR',NATOM,intg=LISTR)
    !         call chmdealloc('pbeqscc.src','SCCPB_MM','FISTR',NATOM,crl=FISTR)
    call chmdealloc('pbeqscc.src','SCCPB_MM','IPHIB',    1,cr4=IPHIB)
    !----------------------------------------------------------------------------

    !     write(*,*) "SCCPB_MM> calling refref"
    CALL RERFEF(X,Y,Z,NCLX,NCLY,NCLZ,DCEL, &
         IPHIWEX,IPHIPEX,TRANX,TRANY,TRANZ, &
         XBCEN,YBCEN,ZBCEN,RF_MM,RXNEX,RXNEY,RXNEZ,ONE,QBSPL)

    !     XIAO ZHU: RF_MM stores the reaction field induced by CHARMM charges, which is charge
    !     independent.    

    RETURN

  END SUBROUTINE SCCPB_MM

  SUBROUTINE SCCPB_QM(NATOM,X,Y,Z,CG,ICALL,QPRIN)
    !
    use chm_kinds
    use memory !(QC: corrected 12/09)
    use number
    use stream
    use dimens_fcm
    use consta
    use timerm
    use sccdftb
    use sccpb
    use machutil,only:wrttim
    !
    implicit none
    !---------------------------------------------------------------------
    !
    INTEGER NATOM,ICALL
    REAL(CHM_REAL)  X(*),Y(*),Z(*),CG(*)
    LOGICAL QPRIN

    !
    !     logical qk
    !
    !         integer,         allocatable, dimension (:) :: LISTR
    !         real(chm_real),  allocatable, dimension (:) :: FISTR
    real(chm_real4), allocatable, dimension (:) :: IPHIB

    !         call chmalloc('pbeqscc.src','SCCPB_MM','LISTR',NATOM,intg=LISTR)
    !         call chmalloc('pbeqscc.src','SCCPB_MM','FISTR',NATOM,crl=FISTR)
    call chmalloc('pbeqscc.src','SCCPB_MM','IPHIB',    1,cr4=IPHIB)
    !     QKEEP=.true.
    !     if (icall .eq. one) then 
    !      QK=.false.  
    !     else
    !      qk=qkeep  
    !      qk=.true.  
    !     endif
    !     print *,'keep',qk
    !     -----------------------------------------------------------------
    !     In vacuum
    !     -----------------------------------------------------------------
    CALL MAYER(NTPRP,LSTPRP,ONE,EPSP,ZERO, &
         !                                  epsw epsp kappa2
         X,Y,Z,CG,PBRAD,NCLX,NCLY,NCLZ,DCEL,WATR,IONR,MAY, &
         MAYX,MAYY,MAYZ,SWIN,ICHC, &
         IPHIPTB, &
         !                                           -----                 -----
         ZERO,TMEMB,ZMEMB,ONE,NIMGB, &
         !          vmemb            epsm
         HTMEMB,ONE,HDENS,POSALP,NEGALP,QCHRG,POFFST,NOFFST, &
         !                 epsh
         DROPLET,ONE,XDROPLET,YDROPLET,ZDROPLET,Qdtom,Qdkap, &
         !                 epsd
         ONE,LXMAX,LYMAX,LZMAX,LXMIN,LYMIN,LZMIN,Qbtom,Qbkap, &
         !          epsb
         ONE,Xcyln,Ycyln,Zcyln,Rcyln,Hcyln,Qctom,Qckap, &
         !          epsc
         ONE,Xcone,Ycone,Zcone,Ax,By,Cz,Qectom,Qeckap, &
         !          epsec
               -1,-1,&
         TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
         IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
         qkeep,qreen,qsmth,QBSPL,QA,QB, &
         !          .true.,qreen,qsmth,QBSPL,QA,QB,
         !          QKEEP QREEN   QSMTH
         RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
         RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
         MAJOR,MINOR,QPROLATE, &
         QINTBP,QZERO,.false.,IPHIB,QPBC,QNPBC, &
         !          -----  -----  QFOCUS
         QNONLINEAR,QPARTLINEAR,QFKAP, &
         IPOX,IPOY,IPOZ,MAPT, &
         NATOM) !QC 12/09 following new pbeq
    !          LISTR,FISTR)

    IF (TIMER.GT.1 .and. ICALL.eq.0) &
         CALL WRTTIM('Grid parameters preparation in vacuum times:')

    IF(QFMG) THEN
       CALL PBEQ2(NATOM,X,Y,Z,CG,.true.,QPRIN)
       IF (TIMER.GT.1 .and. ICALL.eq.0) &
            CALL WRTTIM('FMG iteration times:')
    ELSE
       IF(QOLDPB)THEN
          CALL OLDPBEQ1(MAXITS,DOME,DEPS,NCLX,NCLY,NCLZ, &
               IPHIPTB,MAYX,MAYY,MAYZ, &
                                !                     -----
               ICHC,MAY,TMEMB, &
               .false.,.false.,QPBC,QNPBC,QPRIN)
          !                QOSOR   QFOCUS
       ELSE
          CALL PBEQ1(MAXITS,DOME,DEPS,ONE,EPSP,ONE,ZERO, &
                                !                                       ---         kappa2
               NCLX,NCLY,NCLZ,IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
               IPHIPTB,MAYX,MAYY,MAYZ, &
                                !                     -----
               ICHC,MAY,TMEMB, &
               .false.,.false.,QPBC,QNPBC,QPRIN)
          !                QOSOR   QFOCUS
       ENDIF
       IF (TIMER.GT.1 .and. ICALL.eq.0) &
            CALL WRTTIM('SOR iteration in vacuum times:')
    ENDIF
    !----------------------------------------------------------------------------
    !----------------------------------------------------------------------------
    !     In solution
    !----------------------------------------------------------------------------
    IF(QPRIN) WRITE(OUTU,'(/,3x,A,f8.3)') &
         'Constructing all space-dependent functions:2 EPS =', epsw

    !
    CALL MAYER(NTPRP,LSTPRP,EPSW,EPSP,KAPPA2, &
         !                                   ----
         X,Y,Z,CG,PBRAD,NCLX,NCLY,NCLZ,DCEL,WATR,IONR,MAY, &
         MAYX,MAYY,MAYZ,SWIN,ICHC, &
         IPHIWTB, &
         !                                           -----                 -----
         VMEMB,TMEMB,ZMEMB,EPSM,NIMGB, &
         HTMEMB,EPSH,HDENS,POSALP,NEGALP,QCHRG,POFFST,NOFFST, &
         DROPLET,EPSD,XDROPLET,YDROPLET,ZDROPLET,Qdtom,Qdkap, &
         EPSB,LXMAX,LYMAX,LZMAX,LXMIN,LYMIN,LZMIN,Qbtom,Qbkap, &
         EPSC,Xcyln,Ycyln,Zcyln,Rcyln,Hcyln,Qctom,Qckap, &
         EPSEC,Xcone,Ycone,Zcone,Ax,By,Cz,Qectom,Qeckap, &
               -1,-1,&
         TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
         IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
         qkeep,qreen,qsmth,QBSPL,QA,QB, &
         !          .true.,qreen,qsmth,QBSPL,QA,QB,
         !          QKEEP QREEN   QSMTH
         RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
         RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
         MAJOR,MINOR,QPROLATE, &
         QINTBP,QZERO,.false.,IPHIB,QPBC,QNPBC, &
         !          ------,----- QFOCUS
         QNONLINEAR,QPARTLINEAR,QFKAP, &
         IPOX,IPOY,IPOZ,MAPT, &
         NATOM) !QC 12/09 following new pbeq
    !          LISTR,FISTR)

    IF (TIMER.GT.1 .and. ICALL.eq.0) &
         CALL WRTTIM('Grid parameters preparation in solution times:')

    IF(QFMG) THEN
       CALL PBEQ2(NATOM,X,Y,Z,CG,.false.,QPRIN)
       IF (TIMER.GT.1 .and. ICALL.eq.0) &
            CALL WRTTIM('FMG iteration times:')
    ELSE
       IF(QOLDPB)THEN
          CALL OLDPBEQ1(MAXITS,DOME,DEPS,NCLX,NCLY,NCLZ, &
               IPHIWTB,MAYX,MAYY,MAYZ, &
                                !                     -----
               ICHC,MAY,TMEMB, &
               QOSOR,.false.,QPBC,QNPBC,QPRIN)
          !                QOSOR   QFOCUS
       ELSE
          CALL PBEQ1(MAXITS,DOME,DEPS,EPSW,EPSP,EPSM,KAPPA2, &
                                !                                       ----
               NCLX,NCLY,NCLZ,IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
               IPHIWTB,MAYX,MAYY,MAYZ, &
                                !                     ----- 
               ICHC,MAY,TMEMB, &
               QOSOR,.false.,QPBC,QNPBC,QPRIN)
          !                      QFOCUS
       ENDIF
       IF (TIMER.GT.1 .and. ICALL.eq.0) &
            CALL WRTTIM('SOR iteration in solution times:')
    ENDIF
    !     QC: left here? 12/09
    !         call chmdealloc('pbeqscc.src','SCCPB_MM','LISTR',NATOM,intg=LISTR)
    !         call chmdealloc('pbeqscc.src','SCCPB_MM','FISTR',NATOM,crl=FISTR)
    call chmdealloc('pbeqscc.src','SCCPB_MM','IPHIB',    1,cr4=IPHIB)
    !----------------------------------------------------------------------------
    CALL RERF(X,Y,Z,NCLX,NCLY,NCLZ,DCEL, &
         IPHIWTB,IPHIPTB,TRANX,TRANY,TRANZ, &
         XBCEN,YBCEN,ZBCEN,RF_QM,ONE,QBSPL)

    RETURN
  END SUBROUTINE SCCPB_QM

  SUBROUTINE RERF(X,Y,Z,NCLX,NCLY,NCLZ,DCEL, &
       PHI1,PHI2,TRANX,TRANY,TRANZ, &
       XBCEN,YBCEN,ZBCEN,RF,FACTOR,QBSPL)

    ! XIAO_QC_UW0809
    use chm_kinds
    use number
    use sccdftb
    use sccpb

    implicit none

    REAL(CHM_REAL)  X(*),Y(*),Z(*)
    REAL(CHM_REAL)  RF(*)
    REAL(CHM_REAL)  DCEL,TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN
    REAL(CHM_REAL)  PHIW,PHIV,DELPHI
    REAL(CHM_REAL)  FACTOR       
    REAL(chm_real4)  PHI1(*),PHI2(*)
    INTEGER NCLX,NCLY,NCLZ
    LOGICAL QBSPL
    !   local
    integer irep,iatom
    integer ncyz,jq,ix,iy,iz,n1,in1,n2,in2,n3,in3
    real(chm_real)  chi,xi,yi,zi,ai,bi,ci,fi
    REAL(CHM_REAL)  aisign,bisign,cisign,prefac
    REAL(CHM_REAL)  xgrad,ygrad,zgrad
    ! B-spline
    integer jx1,jx2,jy1,jy2,jz1,jz2,k,l,m,ipx,ipy,ipz,nfil
    real(chm_real)  xc,yc,zc !,M3,DM3
    !
    ncyz=ncly*nclz
    !
    !
    !==============================================================================
    IF (QBSPL) THEN
       !
       !     main loop by quantum atoms
       !      

       DO IREP=1,NSCCRP
          DO IATOM=1,NSCCTC
             rf((IREP-1)*NSCCTC+IATOM)=ZERO

             JQ=IQMLST(IATOM,IREP)
             xi=x(jq)+tranx-xbcen
             yi=y(jq)+trany-ybcen
             zi=z(jq)+tranz-zbcen
             nfil=1
             ix=int(xi/dcel)+1
             iy=int(yi/dcel)+1
             iz=int(zi/dcel)+1
             jx1=ix-nfil
             if(jx1.lt.1)jx1=1
             jx2=ix+nfil+1
             if(jx2.gt.nclx)jx2=nclx
             jy1=iy-nfil
             if(jy1.lt.1)jy1=1
             jy2=iy+nfil+1
             if(jy2.gt.ncly)jy2=ncly
             jz1=iz-nfil
             if(jz1.lt.1)jz1=1
             jz2=iz+nfil+1
             if(jz2.gt.nclz)jz2=nclz
             DO K=jx1,jx2
                IPX=(K-1)*NCyz
                XC=(K-1)*DCEL
                !
                DO L=jy1,jy2
                   IPY=(L-1)*NCLz
                   YC=(L-1)*DCEL
                   !
                   DO M=jz1,jz2
                      IPZ=M+IPY+IPX
                      ZC=(M-1)*DCEL

                      ai=1.5-(xi-xc)/dcel
                      bi=1.5-(yi-yc)/dcel
                      ci=1.5-(zi-zc)/dcel
                      fi=M3(ai)*M3(bi)*M3(ci)
                      !
                      !     Electrostatic "Energy"
                      !     NOTE: no CCELEC to be consistent with SCC
                      !
                      phiw=FACTOR*fi*phi1(ipz)
                      phiv=FACTOR*fi*phi2(ipz)

                      delphi=phiw-phiv
                      ! XIAO_QC_UW0809
                      rf((IREP-1)*NSCCTC+IATOM)=rf((IREP-1)*NSCCTC+IATOM)+delphi
                      !
                      !                 enet1=FACTOR*fi*chi*phi1(ipz)*CCELEC
                      !                 enet2=FACTOR*fi*chi*phi2(ipz)*CCELEC
                      !                 enelp1=enelp1+enet1
                      !                 enelp2=enelp2+enet2

                   enddo !M 
                enddo  !L
             enddo  !K

          ENDDO !enddo nscctc
       ENDDO !enddo nsccrp
    ELSE
       !
       !     main loop by quantum atoms
       !      
       DO IREP=1,NSCCRP
          DO IATOM=1,NSCCTC
             rf((IREP-1)*NSCCTC+IATOM)=ZERO

             JQ=IQMLST(IATOM,IREP)
             xi=x(jq)+tranx-xbcen
             yi=y(jq)+trany-ybcen
             zi=z(jq)+tranz-zbcen
             ix=int(xi/dcel)+1
             iy=int(yi/dcel)+1
             iz=int(zi/dcel)+1

             !     
             !     Atom charge distribution by 8 adjacent grid points
             !     
             do n1=1,2
                in1=ix+n1-1
                ai=xi-(in1-1)*dcel
                aisign=sign(one,ai)
                ai=1.-abs(ai)/dcel
                in1=(in1-1)*ncyz

                do n2=1,2
                   in2=iy+n2-1
                   bi=yi-(in2-1)*dcel
                   bisign=sign(one,bi)
                   bi=1. - abs(bi)/dcel
                   in2=(in2-1)*nclz

                   do n3=1,2
                      in3=iz+n3-1
                      ci=zi-(in3-1)*dcel
                      cisign=sign(one,ci)
                      ci=1. - abs(ci)/dcel
                      fi=ai*bi*ci
                      in3=in1+in2+in3
                      !     
                      !     Electrostatic "Energy" 
                      !     NOTE: no CCELEC to be consistent with SCC
                      phiw=FACTOR*fi*phi1(in3)
                      phiv=FACTOR*fi*phi2(in3)
                      delphi=phiw-phiv
                      rf((IREP-1)*NSCCTC+IATOM)=rf((IREP-1)*NSCCTC+IATOM)+delphi
                      !    
                      !                 enet1=FACTOR*fi*chi*phi1(in3)*CCELEC
                      !                 enelp1=enelp1+enet1

                   enddo !n3
                enddo ! n2
             enddo !n1

          ENDDO
       ENDDO
    ENDIF

    RETURN
  END SUBROUTINE RERF

  SUBROUTINE RERFEF(X,Y,Z,NCLX,NCLY,NCLZ,DCEL, &
       PHI1,PHI2,TRANX,TRANY,TRANZ, &
       XBCEN,YBCEN,ZBCEN,RF,EX,EY,EZ,FACTOR,QBSPL)

    use chm_kinds
    use consta
    use number
    use stream
    use dimens_fcm
    use sccdftb
    use sccpb

    implicit none

    REAL(CHM_REAL)  X(*),Y(*),Z(*)
    REAL(CHM_REAL)  EX(*),EY(*),EZ(*)
    REAL(CHM_REAL)  RF(*)
    REAL(CHM_REAL)  DCEL,TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN
    REAL(CHM_REAL)  PHIW,PHIV,DELPHI
    REAL(CHM_REAL)  FACTOR       
    REAL(chm_real4)  PHI1(*),PHI2(*)
    INTEGER NCLX,NCLY,NCLZ
    LOGICAL QBSPL
    !   local
    integer irep,iatom
    integer ncyz,jq,ix,iy,iz,n1,in1,n2,in2,n3,in3
    real(chm_real)  chi,xi,yi,zi,ai,bi,ci,fi
    REAL(CHM_REAL)  aisign,bisign,cisign,prefac
    REAL(CHM_REAL)  xgrad,ygrad,zgrad
    ! B-spline
    integer jx1,jx2,jy1,jy2,jz1,jz2,k,l,m,ipx,ipy,ipz,nfil
    real(chm_real)  xc,yc,zc!,M3,DM3
    !
    ncyz=ncly*nclz
    !
    !
    !==============================================================================
    IF (QBSPL) THEN
       !
       !     main loop by quantum atoms
       !      
       DO IREP=1,NSCCRP
          DO IATOM=1,NSCCTC
             rf((IREP-1)*NSCCTC+IATOM)=ZERO
             Ex((irep-1)*nscctc+iatom)=zero
             Ey((irep-1)*nscctc+iatom)=zero
             Ez((irep-1)*nscctc+iatom)=zero
             JQ=IQMLST(IATOM,IREP)
             xi=x(jq)+tranx-xbcen
             yi=y(jq)+trany-ybcen
             zi=z(jq)+tranz-zbcen
             nfil=1
             ix=int(xi/dcel)+1
             iy=int(yi/dcel)+1
             iz=int(zi/dcel)+1
             jx1=ix-nfil
             if(jx1.lt.1)jx1=1
             jx2=ix+nfil+1
             if(jx2.gt.nclx)jx2=nclx
             jy1=iy-nfil
             if(jy1.lt.1)jy1=1
             jy2=iy+nfil+1
             if(jy2.gt.ncly)jy2=ncly
             jz1=iz-nfil
             if(jz1.lt.1)jz1=1
             jz2=iz+nfil+1
             if(jz2.gt.nclz)jz2=nclz
             DO K=jx1,jx2
                IPX=(K-1)*NCyz
                XC=(K-1)*DCEL
                !
                DO L=jy1,jy2
                   IPY=(L-1)*NCLz
                   YC=(L-1)*DCEL
                   !
                   DO M=jz1,jz2
                      IPZ=M+IPY+IPX
                      ZC=(M-1)*DCEL

                      ai=1.5-(xi-xc)/dcel
                      bi=1.5-(yi-yc)/dcel
                      ci=1.5-(zi-zc)/dcel
                      fi=M3(ai)*M3(bi)*M3(ci)
                      !
                      prefac=(phi1(ipz)-phi2(ipz))*CCELEC/dcel
                      Ex((irep-1)*nscctc+iatom)= &
                           Ex((irep-1)*nscctc+iatom) +  &
                           DM3(ai)*M3(bi)*M3(ci)*prefac
                      Ey((irep-1)*nscctc+iatom)= &
                           Ey((irep-1)*nscctc+iatom) +  &
                           M3(ai)*DM3(bi)*M3(ci)*prefac
                      Ez((irep-1)*nscctc+iatom)= &
                           Ez((irep-1)*nscctc+iatom) +  &
                           M3(ai)*M3(bi)*DM3(ci)*prefac
                      !
                      !     Electrostatic "Energy"
                      !     NOTE: no CCELEC to be consistent with SCC
                      phiw=FACTOR*fi*phi1(ipz)
                      phiv=FACTOR*fi*phi2(ipz)
                      delphi=phiw-phiv
                      rf((IREP-1)*NSCCTC+IATOM)=rf((IREP-1)*NSCCTC+IATOM)+delphi
                      !                 enet1=FACTOR*fi*chi*phi1(ipz)*CCELEC
                      !                 enet2=FACTOR*fi*chi*phi2(ipz)*CCELEC
                      !                 enelp1=enelp1+enet1
                      !                 enelp2=enelp2+enet2

                   enddo !M 
                enddo  !L
             enddo  !K

          ENDDO !enddo nscctc
       ENDDO !enddo nsccrp
    ELSE
       !
       !     main loop by quantum atoms
       !      
       DO IREP=1,NSCCRP
          DO IATOM=1,NSCCTC
             rf((IREP-1)*NSCCTC+IATOM)=ZERO
             Ex((irep-1)*nscctc+iatom)=zero
             Ey((irep-1)*nscctc+iatom)=zero
             Ez((irep-1)*nscctc+iatom)=zero
             JQ=IQMLST(IATOM,IREP)
             xi=x(jq)+tranx-xbcen
             yi=y(jq)+trany-ybcen
             zi=z(jq)+tranz-zbcen
             ix=int(xi/dcel)+1
             iy=int(yi/dcel)+1
             iz=int(zi/dcel)+1

             !     
             !     Atom charge distribution by 8 adjacent grid points
             !     
             do n1=1,2
                in1=ix+n1-1
                ai=xi-(in1-1)*dcel
                aisign=sign(one,ai)
                ai=1.-abs(ai)/dcel
                in1=(in1-1)*ncyz

                do n2=1,2
                   in2=iy+n2-1
                   bi=yi-(in2-1)*dcel
                   bisign=sign(one,bi)
                   bi=1. - abs(bi)/dcel
                   in2=(in2-1)*nclz

                   do n3=1,2
                      in3=iz+n3-1
                      ci=zi-(in3-1)*dcel
                      cisign=sign(one,ci)
                      ci=1. - abs(ci)/dcel
                      fi=ai*bi*ci
                      in3=in1+in2+in3
                      !
                      prefac=(phi1(in3)-phi2(in3))*CCELEC*factor/dcel
                      !
                      if((ai.lt.(one-rsmall)).and.(ai.gt.rsmall)) then
                         Ex((IREP-1)*NSCCTC+IATOM)= &
                              Ex((IREP-1)*NSCCTC+IATOM)+aisign*bi*ci*prefac
                      endif
                      !     
                      if((bi.lt.(one-rsmall)).and.(bi.gt.rsmall)) then
                         Ey((IREP-1)*NSCCTC+IATOM)= &
                              Ey((IREP-1)*NSCCTC+IATOM)+bisign*ai*ci*prefac
                      endif
                      !     
                      if((ci.lt.(one-rsmall)).and.(ci.gt.rsmall)) then
                         Ez((IREP-1)*NSCCTC+IATOM)= &
                              Ez((IREP-1)*NSCCTC+IATOM)+cisign*ai*bi*prefac
                      endif
                      !
                      !     
                      !     Electrostatic "Energy" 
                      !     NOTE: no CCELEC to be consistent with SCC
                      phiw=FACTOR*fi*phi1(in3)
                      phiv=FACTOR*fi*phi2(in3)
                      delphi=phiw-phiv
                      rf((IREP-1)*NSCCTC+IATOM)=rf((IREP-1)*NSCCTC+IATOM)+delphi
                      !     
                      !                 enet1=FACTOR*fi*chi*phi1(in3)*CCELEC
                      !                 enelp1=enelp1+enet1

                   enddo !n3
                enddo ! n2
             enddo !n1

          ENDDO
       ENDDO

    ENDIF

    RETURN
  END SUBROUTINE RERFEF

  SUBROUTINE SCCPB4_QM(NATOM,X,Y,Z,CG,DX,DY,DZ, &
       ex,ey,ez,NCLX,NCLY,NCLZ,DCEL, &
       PHI1,PHI2,TRANX,TRANY,TRANZ, &
       XBCEN,YBCEN,ZBCEN,FACTOR,QPRIN,QBSPL)

    use chm_kinds
    use number
    use stream
    use dimens_fcm
    use consta
    use sccdftb
    use sccpb
    use blockscc_fcm
    !
    implicit none

    REAL(CHM_REAL)  X(*),Y(*),Z(*),CG(*),DX(*),DY(*),DZ(*)
    REAL(CHM_REAL)  eX(*),eY(*),eZ(*)
    REAL(CHM_REAL)  DCEL,TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN
    REAL(CHM_REAL)  FACTOR       
    REAL(chm_real4)  PHI1(*),PHI2(*)
    real(chm_real)  RXNFX(natom),RXNFY(natom),RXNFZ(natom)
    INTEGER NCLX,NCLY,NCLZ
    integer natom
    LOGICAL QBSPL,QPRIN
    !   local
    integer irep,iatom
    integer ncyz,jq,ix,iy,iz,n1,in1,n2,in2,n3,in3
    real(chm_real)  chi,xi,yi,zi,ai,bi,ci,fi
    REAL(CHM_REAL)  aisign,bisign,cisign,prefac
    REAL(CHM_REAL)  xgrad,ygrad,zgrad
    ! B-spline
    integer jx1,jx2,jy1,jy2,jz1,jz2,k,l,m,ipx,ipy,ipz,nfil
    real(chm_real)  xc,yc,zc!,M3,DM3

    !      FOR FEP - scale the forces otherwise not.
    if(qlamda.eqv..false.) scal=1.0d0

    !
    ncyz=ncly*nclz
    !
    !
    !==============================================================================
    IF (QBSPL) THEN
       !
       !     main loop by quantum atoms
       !      
       DO IREP=1,NSCCRP
          DO IATOM=1,NSCCTC
             chi=QMULI2(IATOM,IREP)
             RXNFX((IREP-1)*NSCCTC+IATOM)=ZERO
             RXNFY((IREP-1)*NSCCTC+IATOM)=ZERO
             RXNFZ((IREP-1)*NSCCTC+IATOM)=ZERO
             JQ=IQMLST(IATOM,IREP)
             xi=x(jq)+tranx-xbcen
             yi=y(jq)+trany-ybcen
             zi=z(jq)+tranz-zbcen
             nfil=1
             ix=int(xi/dcel)+1
             iy=int(yi/dcel)+1
             iz=int(zi/dcel)+1
             jx1=ix-nfil
             if(jx1.lt.1)jx1=1
             jx2=ix+nfil+1
             if(jx2.gt.nclx)jx2=nclx
             jy1=iy-nfil
             if(jy1.lt.1)jy1=1
             jy2=iy+nfil+1
             if(jy2.gt.ncly)jy2=ncly
             jz1=iz-nfil
             if(jz1.lt.1)jz1=1
             jz2=iz+nfil+1
             if(jz2.gt.nclz)jz2=nclz
             DO K=jx1,jx2
                IPX=(K-1)*NCyz
                XC=(K-1)*DCEL
                !
                DO L=jy1,jy2
                   IPY=(L-1)*NCLz
                   YC=(L-1)*DCEL
                   !
                   DO M=jz1,jz2
                      IPZ=M+IPY+IPX
                      ZC=(M-1)*DCEL

                      ai=1.5-(xi-xc)/dcel
                      bi=1.5-(yi-yc)/dcel
                      ci=1.5-(zi-zc)/dcel
                      fi=M3(ai)*M3(bi)*M3(ci)
                      !
                      !     Reaction Field Force  !!! Be careful of the sign (-1/dcel)
                      !           
                      prefac=factor*(phi1(ipz)-phi2(ipz))*CCELEC*chi/dcel
                      RXNFx((irep-1)*nscctc+iatom)= &
                           RXNFx((irep-1)*nscctc+iatom) + &
                           DM3(ai)*M3(bi)*M3(ci)*prefac
                      RXNFy((irep-1)*nscctc+iatom)= &
                           RXNFy((irep-1)*nscctc+iatom) + &
                           M3(ai)*DM3(bi)*M3(ci)*prefac
                      RXNFz((irep-1)*nscctc+iatom)= &
                           RXNFz((irep-1)*nscctc+iatom) + &
                           M3(ai)*M3(bi)*DM3(ci)*prefac
                      !
                   enddo !M 
                enddo  !L
             enddo  !K

          ENDDO !enddo nscctc
       ENDDO !enddo nsccrp
    ELSE
       !
       !     main loop by quantum atoms
       !      
       DO IREP=1,NSCCRP
          DO IATOM=1,NSCCTC
             chi=QMULI2(IATOM,IREP)
             RXNFX((IREP-1)*NSCCTC+IATOM)=ZERO
             RXNFY((IREP-1)*NSCCTC+IATOM)=ZERO
             RXNFZ((IREP-1)*NSCCTC+IATOM)=ZERO
             JQ=IQMLST(IATOM,IREP)
             xi=x(jq)+tranx-xbcen
             yi=y(jq)+trany-ybcen
             zi=z(jq)+tranz-zbcen
             ix=int(xi/dcel)+1
             iy=int(yi/dcel)+1
             iz=int(zi/dcel)+1

             !     
             !     Atom charge distribution by 8 adjacent grid points
             !     
             do n1=1,2
                in1=ix+n1-1
                ai=xi-(in1-1)*dcel
                aisign=sign(one,ai)
                ai=1.-abs(ai)/dcel
                in1=(in1-1)*ncyz

                do n2=1,2
                   in2=iy+n2-1
                   bi=yi-(in2-1)*dcel
                   bisign=sign(one,bi)
                   bi=1. - abs(bi)/dcel
                   in2=(in2-1)*nclz

                   do n3=1,2
                      in3=iz+n3-1
                      ci=zi-(in3-1)*dcel
                      cisign=sign(one,ci)
                      ci=1. - abs(ci)/dcel
                      fi=ai*bi*ci
                      in3=in1+in2+in3
                      !              
                      prefac=(phi1(in3)-phi2(in3))*CCELEC*factor*chi/dcel
                      !        
                      if((ai.lt.(one-rsmall)).and.(ai.gt.rsmall)) then
                         RXNFx((IREP-1)*NSCCTC+IATOM)= &
                              RXNFx((IREP-1)*NSCCTC+IATOM)+aisign*bi*ci*prefac
                      endif
                      !     
                      if((bi.lt.(one-rsmall)).and.(bi.gt.rsmall)) then
                         RXNFy((IREP-1)*NSCCTC+IATOM)= &
                              RXNFy((IREP-1)*NSCCTC+IATOM)+bisign*ai*ci*prefac
                      endif
                      !     
                      if((ci.lt.(one-rsmall)).and.(ci.gt.rsmall)) then
                         RXNFz((IREP-1)*NSCCTC+IATOM)= &
                              RXNFz((IREP-1)*NSCCTC+IATOM)+cisign*ai*bi*prefac
                      endif
                      !

                   enddo !n3
                enddo ! n2
             enddo !n1

          ENDDO
       ENDDO

    ENDIF

    !     Update DX,DY,DZ - DO NOT FORGET TO SCALE THE FORCES WHEN FEP
    DO IREP=1,NSCCRP
       DO IATOM=1,NSCCTC
          chi=QMULI2(IATOM,IREP)
          DX(IQMLST(IATOM,IREP))=DX(IQMLST(IATOM,IREP))+ &
               RXNFX((IREP-1)*NSCCTC+IATOM)*scal/DBLE(NSCCRP)+ &
               eX((IREP-1)*NSCCTC+IATOM)*chi*scal/DBLE(NSCCRP)
          DY(IQMLST(IATOM,IREP))=DY(IQMLST(IATOM,IREP))+ &
               RXNFY((IREP-1)*NSCCTC+IATOM)*scal/DBLE(NSCCRP)+ &
               eY((IREP-1)*NSCCTC+IATOM)*chi*scal/DBLE(NSCCRP)
          DZ(IQMLST(IATOM,IREP))=DZ(IQMLST(IATOM,IREP))+ &
               RXNFZ((IREP-1)*NSCCTC+IATOM)*scal/DBLE(NSCCRP)+ &
               eZ((IREP-1)*NSCCTC+IATOM)*chi*scal/DBLE(NSCCRP)
          IF(PRNLEV.GT.6) THEN
             WRITE(OUTU,'(/,2X,A)') 'Reaction Field Force: PHI(QM)/QM' 
             WRITE(OUTU,'(2I5,3F14.6)') IATOM,IREP &
                  ,RXNFX((IREP-1)*NSCCTC+IATOM)*scal/DBLE(NSCCRP)  &
                  ,RXNFY((IREP-1)*NSCCTC+IATOM)*scal/DBLE(NSCCRP)  &
                  ,RXNFZ((IREP-1)*NSCCTC+IATOM)*scal/DBLE(NSCCRP) 
             WRITE(OUTU,'(/,2X,A)') 'Reaction Field Force: PHI(QM)/MM' 
             WRITE(OUTU,'(2I5,3F14.6)') IATOM,IREP &
                  ,eX((IREP-1)*NSCCTC+IATOM)*scal*chi/DBLE(NSCCRP) &
                  ,eY((IREP-1)*NSCCTC+IATOM)*scal*chi/DBLE(NSCCRP) &
                  ,eZ((IREP-1)*NSCCTC+IATOM)*scal*chi/DBLE(NSCCRP)
          ENDIF
       ENDDO
    ENDDO


    RETURN
  END SUBROUTINE SCCPB4_QM

  SUBROUTINE SCCPB4_MM(NATOM,X,Y,Z,CG,DX,DY,DZ,NCLX,NCLY,NCLZ,DCEL, &
       PHI1,PHI2,TRANX,TRANY,TRANZ, &
       XBCEN,YBCEN,ZBCEN,FACTOR,QPRIN,QBSPL)

    use chm_kinds
    use number
    use stream
    use dimens_fcm
    use consta
    use gamess_fcm
    use sccdftb
    use sccpb
    use blockscc_fcm
    !
    implicit none
    REAL(CHM_REAL)  X(*),Y(*),Z(*),CG(*),DX(*),DY(*),DZ(*)
    REAL(CHM_REAL)  DCEL,TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN
    REAL(CHM_REAL)  FACTOR       
    REAL(chm_real4)  PHI1(*),PHI2(*)
    real(chm_real)  RXNFX(natom),RXNFY(natom),RXNFZ(natom)
    integer natom
    INTEGER NCLX,NCLY,NCLZ
    LOGICAL QBSPL,QPRIN
    !   local
    integer i
    integer ncyz,jq,ix,iy,iz,n1,in1,n2,in2,n3,in3
    real(chm_real)  chi,xi,yi,zi,ai,bi,ci,fi
    REAL(CHM_REAL)  aisign,bisign,cisign,prefac
    REAL(CHM_REAL)  xgrad,ygrad,zgrad
    ! B-spline
    integer jx1,jx2,jy1,jy2,jz1,jz2,k,l,m,ipx,ipy,ipz,nfil
    real(chm_real)  xc,yc,zc!,M3,DM3

    !      FOR FEP - scale the forces otherwise not.
    if(qlamda.eqv..false.) scal=1.0d0

    !
    ncyz=ncly*nclz
    !
    !
    ! QC: 11/17 have to be careful here -
    ! do we use qmused or qmused_sccdftb?
    write(*,*) "SCCPB4_MM>",qmused,qmused_sccdftb

    !==============================================================================
    IF (QBSPL) THEN
       !
       !     main loop by mm atoms
       !      
       DO I = 1, NPTC
          JQ=IMMLST(I)
          chi=cg(jq)
          RXNFX(I)=ZERO
          RXNFY(I)=ZERO
          RXNFZ(I)=ZERO
         ! if(qmused)then
         ! QC: 11/17 test qmused_sccdftb vs. qmused
          if(qmused_sccdftb)then
          IF ((chi .eq. zero) .or. (IGMSEL(JQ) .eq. 5 )) cycle
          endif
          xi=x(jq)+tranx-xbcen
          yi=y(jq)+trany-ybcen
          zi=z(jq)+tranz-zbcen
          nfil=1
          ix=int(xi/dcel)+1
          iy=int(yi/dcel)+1
          iz=int(zi/dcel)+1
          jx1=ix-nfil
          if(jx1.lt.1)jx1=1
          jx2=ix+nfil+1
          if(jx2.gt.nclx)jx2=nclx
          jy1=iy-nfil
          if(jy1.lt.1)jy1=1
          jy2=iy+nfil+1
          if(jy2.gt.ncly)jy2=ncly
          jz1=iz-nfil
          if(jz1.lt.1)jz1=1
          jz2=iz+nfil+1
          if(jz2.gt.nclz)jz2=nclz
          DO K=jx1,jx2
             IPX=(K-1)*NCyz
             XC=(K-1)*DCEL
             !
             DO L=jy1,jy2
                IPY=(L-1)*NCLz
                YC=(L-1)*DCEL
                !
                DO M=jz1,jz2
                   IPZ=M+IPY+IPX
                   ZC=(M-1)*DCEL

                   ai=1.5-(xi-xc)/dcel
                   bi=1.5-(yi-yc)/dcel
                   ci=1.5-(zi-zc)/dcel
                   fi=M3(ai)*M3(bi)*M3(ci)
                   !
                   !     Reaction Field Force  !!! Be careful of the sign (-1/dcel)
                   !           
                   prefac=factor*(phi1(ipz)-phi2(ipz))*CCELEC*chi/dcel
                   RXNFx(i)=RXNFx(i) + DM3(ai)*M3(bi)*M3(ci)*prefac
                   RXNFy(i)=RXNFy(i) + M3(ai)*DM3(bi)*M3(ci)*prefac
                   RXNFz(i)=RXNFz(i) + M3(ai)*M3(bi)*DM3(ci)*prefac
                   !
                enddo !M 
             enddo  !L
          enddo  !K

       ENDDO
    ELSE
       !
       !     main loop by mm atoms
       !      
       DO I = 1, NPTC
          JQ=IMMLST(I)
          chi=cg(jq)
          RXNFX(I)=ZERO
          RXNFY(I)=ZERO
          RXNFZ(I)=ZERO
          if(qmused)then
          IF ((chi .eq. zero) .or. (IGMSEL(JQ) .eq. 5 )) cycle
          endif
          xi=x(jq)+tranx-xbcen
          yi=y(jq)+trany-ybcen
          zi=z(jq)+tranz-zbcen
          ix=int(xi/dcel)+1
          iy=int(yi/dcel)+1
          iz=int(zi/dcel)+1

          !     
          !     Atom charge distribution by 8 adjacent grid points
          !     
          do n1=1,2
             in1=ix+n1-1
             ai=xi-(in1-1)*dcel
             aisign=sign(one,ai)
             ai=1.-abs(ai)/dcel
             in1=(in1-1)*ncyz

             do n2=1,2
                in2=iy+n2-1
                bi=yi-(in2-1)*dcel
                bisign=sign(one,bi)
                bi=1. - abs(bi)/dcel
                in2=(in2-1)*nclz

                do n3=1,2
                   in3=iz+n3-1
                   ci=zi-(in3-1)*dcel
                   cisign=sign(one,ci)
                   ci=1. - abs(ci)/dcel
                   fi=ai*bi*ci
                   in3=in1+in2+in3

                   prefac=FACTOR*(phi1(in3)-phi2(in3))*CCELEC*chi/dcel
                   !
                   if((ai.lt.(one-rsmall)).and.(ai.gt.rsmall)) then
                      RXNFx(i)=RXNFx(i)+aisign*bi*ci*prefac
                   endif
                   !
                   if((bi.lt.(one-rsmall)).and.(bi.gt.rsmall)) then
                      RXNFy(i)=RXNFy(i)+bisign*ai*ci*prefac
                   endif
                   !
                   if((ci.lt.(one-rsmall)).and.(ci.gt.rsmall)) then
                      RXNFz(i)=RXNFz(i)+cisign*ai*bi*prefac
                   endif
                   !     
                enddo !n3
             enddo ! n2
          enddo !n1

       ENDDO

    ENDIF

    !     Update DX,DY,DZ - DO NOT FORGET TO SCALE THE FORCES WHEN FEP
    DO I=1,NPTC
       DX(IMMLST(I))=DX(IMMLST(I))+RXNFX(I)*scal/DBLE(NSCCRP)
       DY(IMMLST(I))=DY(IMMLST(I))+RXNFY(I)*scal/DBLE(NSCCRP)
       DZ(IMMLST(I))=DZ(IMMLST(I))+RXNFZ(I)*scal/DBLE(NSCCRP)
       IF(PRNLEV.GT.6) THEN
          WRITE(OUTU,'(I5,3F14.6)') I &
               ,RXNFX(I)*scal/DBLE(NSCCRP)  &
               ,RXNFY(I)*scal/DBLE(NSCCRP)  &
               ,RXNFZ(I)*scal/DBLE(NSCCRP)  
       ENDIF
    ENDDO

    RETURN
  END SUBROUTINE SCCPB4_MM


  ! XIAO_QC_UW0809
  ! ADD unit number to read in radius.inp file
  subroutine chardradread(fortnum)
    use chm_kinds
    use sccpb

    integer fortnum
    integer itype,ntype  

    !     open(unit=54,file="radius.inp",status="unknown")
    !     rewind 54
    read (fortnum,*) ntype,fmode

    do itype = 1, ntype
       if (fmode.eq.1) then
          read(fortnum,*) COEFA(itype),COEFB(itype) 
       else if (fmode.eq.2) then
          read(fortnum,*) COEFA(itype),COEFB(itype),COEFC(itype), &
               COEFD(itype)
       endif
    enddo

    return
  end subroutine chardradread


  subroutine updrad(prad)
    use chm_kinds
    use sccdftb
    use sccpb

    REAL(CHM_REAL)  prad(*)
    real(chm_real)  chc,vrad
    integer irep,iatom,itdex,itype

    do irep=1,nsccrp
       do iatom=1,nscctc
          chc=QMULI2(IATOM,IREP)
          itdex=IQMLST(IATOM,IREP)
          itype=scctyp(iatom)
          if (fmode.eq.1) then
             vrad=COEFA(itype)*chc+coefb(itype)
             prad(itdex)=vrad
          else if (fmode.eq.2) then
             vrad=COEFA(itype)*chc**3+coefb(itype)*chc**2+coefc(itype)*chc &
                  +coefd(itype)
             prad(itdex)=vrad
          endif
          !       write(*,*) sccatm(iatom),itype,chc,vrad
       enddo
    enddo

    return 
  end subroutine updrad

#endif /* SCCDFTB*/
#if KEY_ENDIF==1
   /*PBEQ*/
#endif

!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
  !---------------------------------------------------------------------
  ! JZ_UW12: Add SMBP 
  ! CHARMM Element source/misc/smbp.src
  SUBROUTINE SMBP0(NATOM,X,Y,Z,CG,ESMBP,DX,DY,DZ,ICALL,QPRIN &
#if KEY_SCCDFTB==1 || KEY_QCHEM==1 || KEY_G09==1
                  ,EQM &
#endif 
                  )       
!-----------------------------------------------------------------------
!
! Main routine for the SMBP, called from ENERGY 
! Calculates phi_rf (MM, EX and QM) as well as phi_tot (MM, QM) 
! and corresponding energies
!
! JZ_UW12 adapted from subroutine GSBP0
!
!-----------------------------------------------------------------------
!
      use chm_kinds
      use memory
      use number
      use stream
      use string
      use dimens_fcm
      use consta
      use comand
      use timerm
      use parallel
#if KEY_GCMC==1
      use gcmc   
#endif
#if KEY_SCCDFTB==1
      use sccdftb
      use sccgsbp
      !QC: UW0110: Also transfer molecular data
      use sccdftbsrc
#endif 
      use machutil,only:wrttim
      use gamess_fcm
      use param_store, only: set_param

      implicit none

      real(chm_real) X(*),Y(*),Z(*),DX(*),DY(*),DZ(*),CG(*)
      real(chm_real) ESMBP,EQM
      INTEGER NATOM,ICALL
      LOGICAL QPRIN
! local
      real(chm_real) ECAVITY
      real(chm_real) LTRAN,LDCEL,LXBCEN,LYBCEN,LZBCEN
      real(chm_real) EDUMMY
      real(chm_real),  allocatable, dimension (:) :: IQMGX,IQMGY,IQMGZ,IRDUMMY,CGSAVE 
      real(chm_real),  allocatable, dimension (:) :: RXNEXFX,RXNEXFY,RXNEXFZ 
      real(chm_real4), allocatable, dimension (:) :: IPHIQM,IPHIQMOLD,IPHIQMDIFF
      real(chm_real4), allocatable, dimension (:) :: IPHITOTSAVE,IPHIMM,IPHIEX 
      integer, allocatable, dimension (:) :: IQMLSTT,IQMLSTTINV
      integer, allocatable, dimension (:) :: LSTREX,LSTREXTMP 
      integer, allocatable, dimension (:) :: LSTRAINV 
      INTEGER LNCEL,NQM,NTREX
      INTEGER I,II,J,N 
      LOGICAL QDOSCC,QDOQM
! parameters for surface charges
      INTEGER MAXPTS
      real(chm_real) INICHRG
      PARAMETER (MAXPTS=1500)   ! Max. number of surface charges == 500
      PARAMETER (INICHRG=0.0E0) ! Initial surface charges are zero
! debug
      real(chm_real) EDBG
      INTEGER I1,DBGPRNT
      PARAMETER (DBGPRNT=1)

!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
#if KEY_SMBP==0
      CALL WRNDIE(-1,'<SMBP0>','SMBP code is not compiled.')
      RETURN 
      END SUBROUTINE SMBP0
#else /**/
!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

!     For the release version, Q-Chem is currently (June 2012) not supported 
!     as changes would have to be made to the Q-Chem code in order to function
!     properly with the SMBP. These might be incorporated into a future
!     version of Q-Chem, but for now skip out when a Q-Chem calculation is tried
!#if KEY_QCHEM==1
!      CALL WRNDIE(-5,'<SMBP0>','SMBP/Q-Chem is currently not supported')
!#endif 

!
!     Check first if PHIX was setup
      IF (NCEL3OG .eq. -1) CALL WRNDIE(-5,'<SMBP0>','PHIX not setup')


!     Check for GCMC allocation
#if KEY_GCMC==1
      IF (.not. allocated(gcmcon)) CALL WRNDIE(-5,'<SMBP0>','GCMCON not allocated') 
#endif

      QDOSCC = .false.
      QDOQM  = .false.

#if KEY_SCCDFTB==1
      IF (NSCCTC .ge. 1) QDOSCC = .true.
#endif 

#if KEY_QCHEM==1 || KEY_G09==1
      IF (NGAMES .ge. 1) QDOQM = .true.
#endif 

!     Some initial allocations
      call chmalloc('pbeq.src','SMBP0','LSTREXTMP',NATOM,intg=LSTREXTMP)
      call chmalloc('pbeq.src','SMBP0','LSTRAINV',NATOM,intg=LSTRAINV)
      call chmalloc('pbeq.src','SMBP0','CGSAVE',NATOM,crl=CGSAVE)

!     Revive LBOX variables from global replacement ones
      LNCEL = LNCELGL
      LTRAN = LTRANGL
      LDCEL = LDCELGL
      LXBCEN = LXBCENGL
      LYBCEN = LYBCENGL
      LZBCEN = LZBCENGL

!     
!     A. Initialize energies and forces, and "total" potentials 
!        Note: RXNAFX, ... are never used, as phi_s^o contrib
!              is added to inner (B) contrib (within PHIMMT) 
!

      ESMBPA=ZERO
      ESMBPB=ZERO
      CALL INITFORCE(NTRB,LSTRB,RXNAFX)
      CALL INITFORCE(NTRB,LSTRB,RXNAFY)
      CALL INITFORCE(NTRB,LSTRB,RXNAFZ)

      CALL INITFORCE(NTRB,LSTRB,RXNBFX)
      CALL INITFORCE(NTRB,LSTRB,RXNBFY)
      CALL INITFORCE(NTRB,LSTRB,RXNBFZ)

!     Initialize both QM and MM total potentials with phi^o_s
!     Need two QM total potentials, for the energy and gradient, respectively
!     where the latter is initialized further below
      CALL DCOPYR4(NCEL3OG,IPHIX,1,IPHIMMT,1) 
      CALL DCOPYR4(NCEL3OG,IPHIX,1,IPHIQMT,1) 

!     
!     B. Calculate reaction-field contributions
!

!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!     Step 1: Calculate phi_rf^MM and inner MM energy
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc


!     Calculate the MM reaction-field potentials
      IF (.not.QSPHERE .and. .not.QRECTBOX) THEN
        CALL WRNDIE(-5,'<SMBP0>', &
                    'SMBP currently implemented for SPHERE & RECT only')
      ENDIF 

      IF (PRNLEV .ge. 5) THEN 
        WRITE(OUTU,100) '###############################'
        WRITE(OUTU,100) '#    Calculating phi_rf^MM    #'
        WRITE(OUTU,100) '###############################'
      ENDIF


!     1.1 Calculate MM RF potential (MM)
      IF (PRNLEV .ge. 5) WRITE(OUTU,100) 'Calculating RF potential (MM)'

!     Create 'inverse' version of LSTRA for zeroing out the charges
!     in the A-part
      CALL FILLI4(LSTRAINV,NATOM,0)
      DO I = 1, NTRA
        LSTRAINV(LSTRA(I)) = 1
      ENDDO 
 
!     Create index for excluded atoms
      CALL FILLI4(LSTREXTMP,NATOM,0)
      NTREX = 0
      IF (QDOSCC .or. QDOQM) THEN
        DO I = 1, NATOM
          IF (IGMSEL(I) .eq. 5) THEN 
            NTREX = NTREX + 1
            LSTREXTMP(NTREX) = I
          ENDIF
        ENDDO
        call chmalloc('pbeq.src','SMBP0','LSTREX',NATOM,intg=LSTREX)
        CALL DCOPYI4(NATOM,LSTREXTMP,1,LSTREX,1)
      ENDIF

#if KEY_GAMESS==1
      CALL DCOPY(int8(NATOM),CG,1_8,CGSAVE,1_8)
#else
      CALL DCOPY(NATOM,CG,1,CGSAVE,1)
#endif
      IF (QDOSCC .or. QDOQM) THEN
        DO I = 1, NATOM
          IF ((IGMSEL(I) .ne. 0 .AND. IGMSEL(I) .ne. 5) &
              .OR. LSTRAINV(I) .eq. 1) CG(I) = ZERO
        ENDDO 
      ELSE 
        DO I = 1, NATOM
          IF (LSTRAINV(I) .eq. 1) CG(I) = ZERO
        ENDDO 
      ENDIF

!     Calculate MM potential
      call chmalloc('pbeq.src','SMBP0','IPHIMM',NCEL3,cr4=IPHIMM)
      IF(QLBOX) THEN
        CALL SPRC_SMBP1(NATOM,X,Y,Z,CG,LNCEL,LTRAN,LDCEL,  &
                        LXBCEN,LYBCEN,LZBCEN,QPRIN,IPHIMM, &
                        .TRUE.,.TRUE.)
      ELSE
        CALL SPRC_SMBP2(NATOM,X,Y,Z,CG,QPRIN,IPHIMM, &
                        .TRUE.,.TRUE.)
      ENDIF
#if KEY_GAMESS==1
      CALL DCOPY(int8(NATOM),CGSAVE,1_8,CG,1_8) ! Get actual charges back
#else
      CALL DCOPY(NATOM,CGSAVE,1,CG,1) ! Get actual charges back
#endif


!     1.2 Calculate MM RF potential (Excluded MM)
      call chmalloc('pbeq.src','SMBP0','IPHIEX',NCEL3,cr4=IPHIEX)
      IF (QDOSCC .or. QDOQM) THEN
        IF (PRNLEV .ge. 5) WRITE(OUTU,100) 'Calculating RF potential (Ex. MM)'
      
#if KEY_GAMESS==1
        CALL DCOPY(int8(NATOM),CG,1_8,CGSAVE,1_8)
#else
        CALL DCOPY(NATOM,CG,1,CGSAVE,1)
#endif
        DO I = 1, NATOM
          IF (IGMSEL(I) .ne. 5) CG(I) = ZERO
        ENDDO  

!       Calculate MM potential
        IF(QLBOX) THEN
          CALL SPRC_SMBP1(NATOM,X,Y,Z,CG,LNCEL,LTRAN,LDCEL, &
                          LXBCEN,LYBCEN,LZBCEN,QPRIN,IPHIEX, &
                          .TRUE.,.TRUE.)
        ELSE
          CALL SPRC_SMBP2(NATOM,X,Y,Z,CG,QPRIN,IPHIEX, &
                          .TRUE.,.TRUE.)
        ENDIF
#if KEY_GAMESS==1
        CALL DCOPY(int8(NATOM),CGSAVE,1_8,CG,1_8) ! Get actual charges back
#else
        CALL DCOPY(NATOM,CGSAVE,1,CG,1) ! Get actual charges back
#endif

      ELSE

        CALL FILLR41(IPHIEX,NCEL3,0.0D0)

      ENDIF

!     Debug printout of PHIMM and PHIEX
      IF (DBGPRNT .ge. 1) THEN
        EDBG = ZERO
        call chmalloc('pbeq.src','SMBP0','IRDUMMY',NATOM,crl=IRDUMMY)
        CALL RFORCE2(NTRB,LSTRB,X,Y,Z,CG,              & 
            NCLX,NCLY,NCLZ,DCEL,IPHIMM,                &
            TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN,ONE,   &
            IRDUMMY,IRDUMMY,IRDUMMY,EDBG,QBSPL         &
#if KEY_GCMC==1
           ,GCMCON & 
#endif
            )
        IF (PRNLEV .ge. 5) THEN
          WRITE(OUTU,100) ''
          WRITE(OUTU,101) 'E_RF (PHIMM) = ', EDBG
        ENDIF

        CALL RFORCE2(NTRB,LSTRB,X,Y,Z,CG,               & 
             NCLX,NCLY,NCLZ,DCEL,IPHIEX,                 &
             TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN,ONE,    &
             IRDUMMY,IRDUMMY,IRDUMMY,EDBG,QBSPL          &
#if KEY_GCMC==1
            ,GCMCON & 
#endif
            )
        IF (PRNLEV .ge. 5) THEN
          WRITE(OUTU,101) 'E_RF (PHIEX) = ', EDBG
        ENDIF
        call chmdealloc('pbeq.src','SMBP0','IRDUMMY',NATOM,crl=IRDUMMY)
      ENDIF

!
!     1.3 Now that both MM RF potentials (MM and Ex. MM) have been calculated,
!     assemble phi^MM_tot and add MM contributions to phi^QM_tot
!     Also calculate energies for outer part (ESMBPA) and MM inner part 
!     (ESMBPB(MM))
!
!     Outer part: phi_o^s energy + gradient 
      call chmalloc('pbeq.src','SMBP0','IRDUMMY',NATOM,crl=IRDUMMY)
      CALL RFORCE2(NTRB,LSTRB,X,Y,Z,CG, &
           NCLXOG,NCLYOG,NCLZOG,DCELOG,IPHIX, &
           TRANXOG,TRANYOG,TRANZOG,XBCENOG,YBCENOG,ZBCENOG,ONE, &
           IRDUMMY,IRDUMMY,IRDUMMY,ESMBPA,QBSPL &
#if KEY_GCMC==1
          ,GCMCON & 
#endif
           )

!     Inner MM part: energy
!     NOTE: The gradients are calculated in Step 3 below (need phi_rf^QM)
      CALL ADD_SMALL_LRG_GRID(IPHIMMT,IPHIMM, &
                              NCEL3OG,NCEL3,DCELOG,DCEL, &
                              NCLXOG,NCLYOG,NCLZOG,NCLX,NCLY,NCLZ, &
                              HALF) 

      CALL RFORCE2(NTRB,LSTRB,X,Y,Z,CG, &
           NCLXOG,NCLYOG,NCLZOG,DCELOG,IPHIMMT, &
           TRANXOG,TRANYOG,TRANZOG,XBCENOG,YBCENOG,ZBCENOG,ONE, &
           IRDUMMY,IRDUMMY,IRDUMMY,ESMBPB,QBSPL &
#if KEY_GCMC==1
          ,GCMCON & 
#endif
           )
      call chmdealloc('pbeq.src','SMBP0','IRDUMMY',NATOM,crl=IRDUMMY)
      ESMBPB = ESMBPB - ESMBPA


!     Finally add MM contributions to PHI_QM (energy and gradient)
      IF (QDOSCC .or. QDOQM) THEN
        CALL ADD_SMALL_LRG_GRID(IPHIQMT,IPHIMM, &
                                NCEL3OG,NCEL3,DCELOG,DCEL, &
                                NCLXOG,NCLYOG,NCLZOG,NCLX,NCLY,NCLZ, &
                                ONE) 
        CALL ADD_SMALL_LRG_GRID(IPHIQMT,IPHIEX, &
                                NCEL3OG,NCEL3,DCELOG,DCEL, &
                                NCLXOG,NCLYOG,NCLZOG,NCLX,NCLY,NCLZ, &
                                MINONE) 
        CALL DCOPYR4(NCEL3OG,IPHIQMT,1,IPHIQMG,1) 
      ENDIF
      call chmdealloc('pbeq.src','SMBP0','IPHIEX',NCEL3,cr4=IPHIEX)

!     Allocate and initialize QM rf potential     
      call chmalloc('pbeq.src','SMBP0','IPHIQM',NCEL3,cr4=IPHIQM)
      CALL FILLR41(IPHIQM,NCEL3,0.0D0)

     
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!     Step 2: Calculate phi_rf^QM and QM energy & gradient
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      
      IF (QDOSCC .or. QDOQM) THEN
        IF (PRNLEV .ge. 5) THEN 
          WRITE(OUTU,100) ''
          WRITE(OUTU,100) '###############################'
          WRITE(OUTU,100) '#    Calculating phi_rf^QM    #'
          WRITE(OUTU,100) '###############################'
        ENDIF
      
!       Determine number of QM-atoms
        NQM = NGAMES
#if KEY_SCCDFTB==1
        NQM = NSCCTC
#if KEY_QCHEM==1 
        if(qmused_qchem) then 
          NQM = NGAMES
        endif 
#endif 
#endif 
        IF (NQM .eq. 0) CALL WRNDIE(-5,'<SMBP0>','NQM == 0')

!       Create indices for QM-Atoms
        call chmalloc('pbeq.src','SMBP0','IQMLSTT',NQM,intg=IQMLSTT)
        call chmalloc('pbeq.src','SMBP0','IQMLSTTINV',NATOM,intg=IQMLSTTINV)
        CALL SMBP_QM_LST(NATOM,NQM,IQMLSTT,IQMLSTTINV)

!       2.1 Calculate QM energy contribution using SCRF
        call chmalloc('pbeq.src','SMBP0','ISURFCRD',MAXPTS,crl=ISURFCRD)
        call chmalloc('pbeq.src','SMBP0','ISURFCHR',MAXPTS,crl=ISURFCHR)
        call chmalloc('pbeq.src','SMBP0','IPHIQMOLD',NCEL3,cr4=IPHIQMOLD)
        call chmalloc('pbeq.src','SMBP0','IPHIQMDIFF',NCEL3,cr4=IPHIQMDIFF)
        call chmalloc('pbeq.src','SMBP0','IPHITOTSAVE',NCEL3OG,cr4=IPHITOTSAVE)

        CALL SPRC_SMBP3(EQM,NATOM,X,Y,Z,CG,NQM, &
                        LNCEL,LTRAN,LDCEL,LXBCEN,LYBCEN,LZBCEN,  &
                        QPRIN,QDOSCC,QDOQM, &
                        IQMLSTT,IQMLSTTINV, &
                        IPHIQMT,IPHITOTSAVE, &
                        IPHIQM,IPHIQMOLD, &
                        IPHIQMDIFF, &
                        ISURFCRD,ISURFCHR, &
                        MAXPTS,INICHRG)

        call chmdealloc('pbeq.src','SMBP0','IPHIQMOLD',NCEL3,cr4=IPHIQMOLD)
        call chmdealloc('pbeq.src','SMBP0','IPHIQMDIFF',NCEL3,cr4=IPHIQMDIFF)
        call chmdealloc('pbeq.src','SMBP0','IPHITOTSAVE',NCEL3OG,cr4=IPHITOTSAVE)


!       2.2 Now go for the QM gradient contributions
!           and store them in QMGX, ... arrays for now
!           (to be added to total gradient below)

!       complete total QM gradient potential
        CALL ADD_SMALL_LRG_GRID(IPHIQMG,IPHIQM, &
                                NCEL3OG,NCEL3,DCELOG,DCEL, &
                                NCLXOG,NCLYOG,NCLZOG,NCLX,NCLY,NCLZ, &
                                ONE) 

        call chmalloc('pbeq.src','SMBP0','IQMGX',NTRB,crl=IQMGX)
        call chmalloc('pbeq.src','SMBP0','IQMGY',NTRB,crl=IQMGY)
        call chmalloc('pbeq.src','SMBP0','IQMGZ',NTRB,crl=IQMGZ)
        CALL SPRC_SMBP4(NATOM,X,Y,Z,DX,DY,DZ,CG,NQM, &
                        LNCEL,LTRAN,LDCEL,LXBCEN,LYBCEN,LZBCEN, &
                        QPRIN,QDOSCC,QDOQM, &
                        IQMLSTT,IQMLSTTINV, &
                        IPHIQMG,ISURFCRD,ISURFCHR, &
                        IQMGX,IQMGY,IQMGZ, &
                        MAXPTS,INICHRG)


        call chmdealloc('pbeq.src','SMBP0','ISURFCRD',MAXPTS,crl=ISURFCRD)
        call chmdealloc('pbeq.src','SMBP0','ISURFCHR',MAXPTS,crl=ISURFCHR)

      ENDIF ! QDOSCC || QDOQM

!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!     Step 3: Calculate inner MM gradient
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       
!     Inner MM part: gradient
!     we need phi_o^s + 1.0 * phi_mm, so add another half of phi_mm
      CALL ADD_SMALL_LRG_GRID(IPHIMMT,IPHIMM, &
                              NCEL3OG,NCEL3,DCELOG,DCEL, &
                              NCLXOG,NCLYOG,NCLZOG,NCLX,NCLY,NCLZ, &
                              HALF) 
      call chmdealloc('pbeq.src','SMBP0','IPHIMM',NCEL3,cr4=IPHIMM)

!     Now add phi_qm to create total MM gradient potential
      CALL ADD_SMALL_LRG_GRID(IPHIMMT,IPHIQM, &
                              NCEL3OG,NCEL3,DCELOG,DCEL, &
                              NCLXOG,NCLYOG,NCLZOG,NCLX,NCLY,NCLZ, &
                              ONE) 

!     and calculate the first part of the MM gradient:
!     (phi_s^o + phi_rf^MM + phi_rf^QM) * d/dx rho_mm
      CALL RFORCE2(NTRB,LSTRB,X,Y,Z,CG, &
           NCLXOG,NCLYOG,NCLZOG,DCELOG,IPHIMMT, &
           TRANXOG,TRANYOG,TRANZOG,XBCENOG,YBCENOG,ZBCENOG,ONE, &
           RXNBFX,RXNBFY,RXNBFZ,EDUMMY,QBSPL &
#if KEY_GCMC==1
          ,GCMCON & 
#endif
           )

!     and now the second part, which accounts for excluded atoms:
!     phi_rf^QM * d/dx rho_ex
      IF (NTREX .gt. 0) THEN
        call chmalloc('pbeq.src','SMBP0','RXNEXFX',NATOM,crl=RXNEXFX)
        call chmalloc('pbeq.src','SMBP0','RXNEXFY',NATOM,crl=RXNEXFY)
        call chmalloc('pbeq.src','SMBP0','RXNEXFZ',NATOM,crl=RXNEXFZ)
        CALL INITFORCE(NTREX,LSTREX,RXNEXFX)
        CALL INITFORCE(NTREX,LSTREX,RXNEXFY)
        CALL INITFORCE(NTREX,LSTREX,RXNEXFZ)
       
        CALL RFORCE2(NTREX,LSTREX,X,Y,Z,CG, &
             NCLX,NCLY,NCLZ,DCEL,IPHIQM, &
             TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN,ONE, &
             RXNEXFX,RXNEXFY,RXNEXFZ,EDUMMY,QBSPL &
#if KEY_GCMC==1
            ,GCMCON & 
#endif
             )
       
!       calculate final MM gradient: part1 - part2
        CALL SMBP_SUM_GRAD(NTREX,LSTREX,MINONE,1, &
                           RXNBFX,RXNBFY,RXNBFZ, &
                           RXNEXFX,RXNEXFY,RXNEXFZ &
#if KEY_GCMC==1
                          ,GCMCON & 
#endif
                           )
       
        call chmdealloc('pbeq.src','SMBP0','RXNEXFX',NATOM,crl=RXNEXFX)
        call chmdealloc('pbeq.src','SMBP0','RXNEXFY',NATOM,crl=RXNEXFY)
        call chmdealloc('pbeq.src','SMBP0','RXNEXFZ',NATOM,crl=RXNEXFZ)
      ENDIF

      IF (QDOSCC .or. QDOQM) call chmdealloc('pbeq.src','SMBP0','LSTREX',NATOM,intg=LSTREX)
      call chmdealloc('pbeq.src','SMBP0','IPHIQM',NCEL3,cr4=IPHIQM)

!     finally add QM gradient contributions to MM gradient
      IF (QDOQM .or. QDOSCC) THEN

        CALL SMBP_SUM_GRAD(NTRB,LSTRB,ONE,2, &
                           RXNBFX,RXNBFY,RXNBFZ, &
                           IQMGX,IQMGY,IQMGZ &
#if KEY_GCMC==1
                          ,GCMCON & 
#endif
                           ) 

        call chmdealloc('pbeq.src','SMBP0','IQMGX',NTRB,crl=IQMGX)
        call chmdealloc('pbeq.src','SMBP0','IQMGY',NTRB,crl=IQMGY)
        call chmdealloc('pbeq.src','SMBP0','IQMGZ',NTRB,crl=IQMGZ)
        call chmdealloc('pbeq.src','SMBP0','IQMLSTT',NQM,intg=IQMLSTT)
        call chmdealloc('pbeq.src','SMBP0','IQMLSTTINV',NATOM,intg=IQMLSTTINV)
      ENDIF

!
! return potential energy and forces

333   CONTINUE


      CALL GSBP4(NTRB,LSTRB,ESMBPA,ESMBPB,ESMBP, &
           DX,DY,DZ,RXNAFX,RXNAFY,RXNAFZ, &
           RXNBFX,RXNBFY,RXNBFZ,QPRIN &
#if KEY_GCMC==1
           ,GCMCON &   
#endif
           )

      call set_param('SMBE', ESMBP)


      call chmdealloc('pbeq.src','SMBP0','LSTREXTMP',NATOM,intg=LSTREXTMP)
      call chmdealloc('pbeq.src','SMBP0','LSTRAINV',NATOM,intg=LSTRAINV)
      call chmdealloc('pbeq.src','SMBP0','CGSAVE',NATOM,crl=CGSAVE)

!
100   FORMAT(3X,A)
101   FORMAT(3X,A,F20.6)
      RETURN
      END SUBROUTINE SMBP0
#endif 

!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

      SUBROUTINE SMBP0_SETUP(NATOM,X,Y,Z,CG,DX,DY,DZ,ICALL,QPRIN)
!
!-----------------------------------------------------------------------
!
! Setup routine for the SMBP 
! Basically calculates phi_s^o (PHIX) and determines large box params
!
! JZ_UW12 adapted from subroutine GSBP0
!
!-----------------------------------------------------------------------
!
!
      use chm_kinds
      use memory
      use number
      use stream
      use string
      use dimens_fcm
      use consta
      use comand
      use timerm
      use parallel
      use machutil,only:wrttim
#if KEY_GCMC==1
      use gcmc   
#endif
#if KEY_SCCDFTB==1
      use sccdftb
      use sccgsbp
      !QC: UW0110: Also transfer molecular data
      use sccdftbsrc
#endif 
      implicit none

!
      real(chm_real) X(*),Y(*),Z(*),DX(*),DY(*),DZ(*),CG(*)
      INTEGER NATOM,ICALL
      LOGICAL QLBOX_PARA_ONLY,QPRIN
! local
      real(chm_real) ESMBP,LTRAN,LDCEL,LXBCEN,LYBCEN,LZBCEN
      INTEGER I,II,J,N 
      INTEGER LNCEL

#if KEY_SMBP==0
      CALL WRNDIE(-1,'<SMBP0_SETUP>','SMBP code is not compiled.')
      RETURN
      END SUBROUTINE SMBP0_SETUP
#else /**/
!
!     Calculate static potential of outer region phi_o^s
!
      IF(QLBOX) THEN
!     construct a large box to obtain boundary potentials of the original box.
!     here, we just consider a cubic large box.
         LDCEL  = GTRMF(COMLYN,COMLEN,'LDCE',DCEL*FOUR)
         LNCEL  = GTRMI(COMLYN,COMLEN,'LNCE',33)
         LXBCEN = GTRMF(COMLYN,COMLEN,'LXBC',ZERO)
         LYBCEN = GTRMF(COMLYN,COMLEN,'LYBC',ZERO)
         LZBCEN = GTRMF(COMLYN,COMLEN,'LZBC',ZERO)

         IF ((LNCEL.GT.NCLX).OR.(LNCEL.GT.NCLY).OR.(LNCEL.GT.NCLZ)) THEN
           CALL WRNDIE(-5,'<SMBP0_SETUP>',& 
           'LNCE too large, should be less than either NCLX or NCLY or NCLZ.')
         ENDIF

         IF(MOD(LNCEL,2).eq.0)THEN
            LNCEL=LNCEL+1
         ENDIF
         LTRAN=HALF*(LNCEL-1)*LDCEL

         QFOCUS=INDXA(COMLYN, COMLEN, 'FOCU') .GT. 0
         IF(QFOCUS) THEN
            IF(TRANX.GE.LTRAN .OR.TRANY.GE.LTRAN.OR.TRANZ.GE.LTRAN)THEN
               WRITE(OUTU,'(/,3X,A)') &
              'SMBP WARNING: Large system should contain original one.'
               CALL WRNDIE(-5,'<SMBP0_SETUP>', &
                           'FOCUSSING SELECTION ERROR')
            ENDIF
            WRITE(OUTU,'(/,3x,A,/,3x,A)') &
            'Following large box will be used to calculate boundary' , &
            'potentials of the original box using focussing'
         ELSE
            WRITE(OUTU,'(/,3x,A,/,3x,A)') &
            'Charge density on grid of the following large box will be', &
            'used to calculate boundary potentials of the original box'
         ENDIF
         WRITE(OUTU,101) &
              'Large Box in X from ',LXBCEN-LTRAN,' to ',LXBCEN+LTRAN
         WRITE(OUTU,101) &
              'Large Box in Y from ',LYBCEN-LTRAN,' to ',LYBCEN+LTRAN
         WRITE(OUTU,101) &
              'Large Box in Z from ',LZBCEN-LTRAN,' to ',LZBCEN+LTRAN
 101     FORMAT(3X,A,F8.3,A,F8.3)
! save the LBOX variables globally in the replacement variables
! LNCELGL etc., so that we can use them outside PBEQ
         LNCELGL = LNCEL
         LTRANGL = LTRAN
         LDCELGL = LDCEL
         LXBCENGL = LXBCEN
         LYBCENGL = LYBCEN
         LZBCENGL = LZBCEN
      ENDIF

! calculate the protein field and decompose electrostatic free energy
! Note: Whether we do the actual calculation is determined by QGAB <-- QPHIX 
      CALL GDECOMP(NATOM,X,Y,Z,CG, &
           LNCEL,LTRAN,LDCEL,LXBCEN,LYBCEN,LZBCEN,QPRIN)
      IF(TIMER.GT.1.AND.ICALL.EQ.0) CALL WRTTIM('GDECOMP times:')

! save current grid parameters as 'outer grid' params if this is a PHIX run
      IF (.not.QPHIX) THEN
        NCLXOG=NCLX
        NCLYOG=NCLY
        NCLZOG=NCLZ
        NCEL3OG=NCEL3
        DCELOG=DCEL
        XBCENOG=XBCEN
        YBCENOG=YBCEN
        ZBCENOG=ZBCEN
        TRANXOG=HALF*(NCLX-1)*DCEL
        TRANYOG=HALF*(NCLY-1)*DCEL
        TRANZOG=HALF*(NCLZ-1)*DCEL
      ENDIF

! calculate the protein field energy and forces (NOTE: HALF -> ONE)
      ESMBPA=ZERO
      ESMBPB=ZERO
      CALL INITFORCE(NTRB,LSTRB,RXNAFX)
      CALL INITFORCE(NTRB,LSTRB,RXNAFY)
      CALL INITFORCE(NTRB,LSTRB,RXNAFZ)

      CALL INITFORCE(NTRB,LSTRB,RXNBFX)
      CALL INITFORCE(NTRB,LSTRB,RXNBFY)
      CALL INITFORCE(NTRB,LSTRB,RXNBFZ)

      IF (QPHIX) THEN
           CALL RFORCE2(NTRB,LSTRB,X,Y,Z,CG, &
                NCLXOG,NCLYOG,NCLZOG,DCELOG,IPHIX, &
                TRANXOG,TRANYOG,TRANZOG,XBCENOG,YBCENOG,ZBCENOG,ONE, &
                RXNAFX,RXNAFY,RXNAFZ,ESMBPA,QBSPL &
#if KEY_GCMC==1
               ,GCMCON & 
#endif
                )
      ELSE
           CALL RFORCE2(NTRB,LSTRB,X,Y,Z,CG, &
                NCLX,NCLY,NCLZ,DCEL,IPHIX, &
                TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN,ONE, &
                RXNAFX,RXNAFY,RXNAFZ,ESMBPA,QBSPL &
#if KEY_GCMC==1
               ,GCMCON & 
#endif
                )
      ENDIF

! return potential energy and forces

      CALL GSBP4(NTRB,LSTRB,ESMBPA,ESMBPB,ESMBP, &
           DX,DY,DZ,RXNAFX,RXNAFY,RXNAFZ, &
           RXNBFX,RXNBFY,RXNBFZ,QPRIN &
#if KEY_GCMC==1
           ,GCMCON &   
#endif
           )

!
      RETURN
      END SUBROUTINE SMBP0_SETUP

!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

      SUBROUTINE PREP2_SMBP
!-----------------------------------------------------------------------
!     INPUT PARAMETERS FOR SMBP
!
!
      use chm_kinds
      use memory
      use number
      use stream
      use string
      use dimens_fcm
      use consta
      use comand
      use psf
      use coord
      use parallel
      use ssbpm,only: qcavi
#if KEY_GCMC==1
      use gcmc   
#endif
!-----------------------------------------------------------------
! Local variables
      real(chm_real) QTOTA, QTOTB
      INTEGER I
      INTEGER ISLGC,ERR
      INTEGER OLDNTP,OLDMIJ,OLDPX,OLDPY,OLDPZ,OLDBL,OLDBM
!
!
#if KEY_GCMC==1
      if(.not.allocated(gcmcon)) call allocate_gcmc  
#endif
!
!     Solvent Macromolecule Boundary Potential (SMBP)
!     =============================================
!     Region A = fixed region; Region B = region of interest

      QA=.false.                ! for atom selection in region A
      QB=.false.                ! for atom selection in region B
      QGAB=.not.QPHIX           ! true when we need to calculate PHIX
                                ! (done in setup part)               
!
!      QGTOT = INDXA(COMLYN, COMLEN, 'GTOT').GT.0
!      QGAA  = INDXA(COMLYN, COMLEN, 'G_OO').GT.0
!      QGAB  = INDXA(COMLYN, COMLEN, 'G_IO').GT.0 .or.
!     $        INDXA(COMLYN, COMLEN, 'G_OI').GT.0
!      QGBB  = INDXA(COMLYN, COMLEN, 'G_II').GT.0
!      QNOSORT=INDXA(COMLYN, COMLEN, 'NOSO').GT.0

! get boundary potentials of basis charge distributions using
! a large box and the charge density on its grid  (it works with focussing)
      QLBOX =INDXA(COMLYN, COMLEN, 'LBOX') .GT. 0
! choose one geometry
      QRECTBOX=INDXA(COMLYN, COMLEN, 'RECT').GT.0
      QSPHERE =INDXA(COMLYN, COMLEN, 'SPHE').GT.0
      QPROLATE=INDXA(COMLYN, COMLEN, 'PROL').GT.0
      QRFRECT =INDXA(COMLYN, COMLEN, 'RFRE').GT.0
      QRFSPHE =INDXA(COMLYN, COMLEN, 'RFSP').GT.0
      IF(.not.QSPHERE .and. .not.QRECTBOX) THEN
         WRITE(OUTU,'(/,3X,A)') & 
        'PREP WARNING: Currently only SPHERE and RECTBOX implemented'
         CALL WRNDIE(-5,'<PREP2_SMBP>','SMBP ERROR')
      ENDIF

! rectangular region of interest
      IF(QRECTBOX) THEN
         RBXMIN = GTRMF(COMLYN,COMLEN,'XMIN',ZERO)
         RBXMAX = GTRMF(COMLYN,COMLEN,'XMAX',ZERO)
         RBYMIN = GTRMF(COMLYN,COMLEN,'YMIN',ZERO)
         RBYMAX = GTRMF(COMLYN,COMLEN,'YMAX',ZERO)
         RBZMIN = GTRMF(COMLYN,COMLEN,'ZMIN',ZERO)
         RBZMAX = GTRMF(COMLYN,COMLEN,'ZMAX',ZERO)
         RINCX  = GTRMF(COMLYN,COMLEN,'INCX',ONE)
         RINCY  = GTRMF(COMLYN,COMLEN,'INCY',ONE)
         RINCZ  = GTRMF(COMLYN,COMLEN,'INCZ',ONE)
         RBXMIN = (INT((RBXMIN+TRANX)/DCEL)+HALF)*DCEL-TRANX
         RBXMAX = (INT((RBXMAX+TRANX)/DCEL)+HALF)*DCEL-TRANX
         RBYMIN = (INT((RBYMIN+TRANY)/DCEL)+HALF)*DCEL-TRANY
         RBYMAX = (INT((RBYMAX+TRANY)/DCEL)+HALF)*DCEL-TRANY
         RBZMIN = (INT((RBZMIN+TRANZ)/DCEL)+HALF)*DCEL-TRANZ
         RBZMAX = (INT((RBZMAX+TRANZ)/DCEL)+HALF)*DCEL-TRANZ
         RRXCEN = (RBXMAX+RBXMIN)/TWO
         RRYCEN = (RBYMAX+RBYMIN)/TWO
         RRZCEN = (RBZMAX+RBZMIN)/TWO
         XSCALE=TWO/(RBXMAX-RBXMIN)
         YSCALE=TWO/(RBYMAX-RBYMIN)
         ZSCALE=TWO/(RBZMAX-RBZMIN)
      ENDIF

! spherical region of interest
      IF(QSPHERE) THEN
         SRDIST = GTRMF(COMLYN,COMLEN,'SRDI',ZERO) ! a radius
         RRXCEN = GTRMF(COMLYN,COMLEN,'RRXC',ZERO) ! a sphere center
         RRYCEN = GTRMF(COMLYN,COMLEN,'RRYC',ZERO)
         RRZCEN = GTRMF(COMLYN,COMLEN,'RRZC',ZERO)
      ENDIF

! see which iguess we want for the inital QM charges
      SMBPIGU = GTRMI(COMLYN,COMLEN,'IGUE',1)

! read in number of surface charges on sphere and algorithm
! that is used to calculate them
      NUMSURF = GTRMI(COMLYN,COMLEN,'NSPT',90)  
      SPHEPTSALG = GTRMI(COMLYN,COMLEN,'SPAL',2)  

! threshold and maxit for the Conjugate Gradient optimizing the
! surface charges
      CGTHRESH = GTRMF(COMLYN,COMLEN,'CGTH',TENM6)
      CGMAXIT  = GTRMI(COMLYN,COMLEN,'CGMX',2000)

! controls for the SCRF
      SMBPPTHR = GTRMF(COMLYN,COMLEN,'SCTH',PT0005) ! PT0005 = 5 * 10^-4
      SMBPMXIT = GTRMI(COMLYN,COMLEN,'SCMX',50)

! whether we use MDC or Mulliken charges from Q-Chem
! 1 = MDC; 2 = Mulliken
      QCCHRG   = GTRMI(COMLYN,COMLEN,'QCCH',1)


! allocate the grid and multipol parameters
      IF(.NOT.QPBEQ) THEN
!       for EPSP
        call chmalloc('pbeq.src','PREP2_SMBP','IPHIP',NCEL3,cr4=IPHIP)
!       for static external field
        IF(.not.QPHIX) call chmalloc('pbeq.src','PBEQ0','IPHIX',NCEL3,cr4=IPHIX)
        call chmalloc('pbeq.src','PREP2_SMBP','LSTRA',NATOM,intg=LSTRA)
        call chmalloc('pbeq.src','PREP2_SMBP','LSTRB',NATOM,intg=LSTRB)
        call chmalloc('pbeq.src','PREP2_SMBP','RXNAFX',NATOM,crl=RXNAFX)
        call chmalloc('pbeq.src','PREP2_SMBP','RXNAFY',NATOM,crl=RXNAFY)
        call chmalloc('pbeq.src','PREP2_SMBP','RXNAFZ',NATOM,crl=RXNAFZ)
        call chmalloc('pbeq.src','PREP2_SMBP','RXNBFX',NATOM,crl=RXNBFX)
        call chmalloc('pbeq.src','PREP2_SMBP','RXNBFY',NATOM,crl=RXNBFY)
        call chmalloc('pbeq.src','PREP2_SMBP','RXNBFZ',NATOM,crl=RXNBFZ)
!       total MM and QM potentials
        IF (QPHIX) THEN
          call chmalloc('pbeq.src','PREP2_SMBP','IPHIMMT',NCEL3OG,cr4=IPHIMMT)
          call chmalloc('pbeq.src','PREP2_SMBP','IPHIQMT',NCEL3OG,cr4=IPHIQMT)
          call chmalloc('pbeq.src','PREP2_SMBP','IPHIQMG',NCEL3OG,cr4=IPHIQMG)
          QSMBP_ALLOC_TOTP = .true. 
        ENDIF
      ENDIF

! make lists of atoms in the fixed region (A) and the region of interest (B)
      IF(QRECTBOX) THEN
         CALL STRECTBOX(NTPRP,LSTPRP,X,Y,Z,CG, &
              NTRA,LSTRA,QTOTA,NTRB,LSTRB,QTOTB, &
              RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX &
#if KEY_GCMC==1
             ,GCMCON & 
#endif
              )
      ELSEIF(QSPHERE) THEN
         CALL STSPHERE(NTPRP,LSTPRP,X,Y,Z,CG, &
              NTRA,LSTRA,QTOTA,NTRB,LSTRB,QTOTB, &
              SRDIST,RRXCEN,RRYCEN,RRZCEN &
#if KEY_GCMC==1
             ,GCMCON & 
#endif
              )
      ENDIF
! select the atoms for printing forces
! in this case, the first selection for PBEQ solver should be issued
      WRITE(OUTU,101)
      WRITE(OUTU,101) &
           '========================================================='
      WRITE(OUTU,101) &
           '==== Solvent Macromolecule Boundary Potential (SMBP) ===='
      WRITE(OUTU,101) &
           '========================================================='
      WRITE(OUTU,101) 
      IF(NTRB.EQ.0) THEN
         WRITE(OUTU,'(3X,2A)') &
            'PREP WARNING: There are no atoms inside region of interest'
         CALL WRNDIE(1,'<PREP2_SMBP>','SMBP WARNING')
      ENDIF
      WRITE(OUTU,'(3x,2A)') &
           'Summary of two regions : ','Outer [O] and Inner [I]'
      WRITE(OUTU,'(3X,A,I6,5X,A,F10.5)') &
           'Number of atoms [O] : ',NTRA,'Total charge [O] :',QTOTA
      WRITE(OUTU,'(3X,A,I6,5X,A,F10.5)') &
           'Number of atoms [I] : ',NTRB,'Total charge [I] :',QTOTB
      WRITE(OUTU,101)
      IF(QRECTBOX) THEN
         WRITE(OUTU,102) &
              'Inner region in X from ',RBXMIN,' to ',RBXMAX
         WRITE(OUTU,102) &
              'Inner region in Y from ',RBYMIN,' to ',RBYMAX
         WRITE(OUTU,102) &
              'Inner region in Z from ',RBZMIN,' to ',RBZMAX
         WRITE(OUTU,101)
      ELSEIF(QSPHERE) THEN
         WRITE(OUTU,'(3x,a,/,3x,a,f8.3,a,/,3x,a,3f8.3)') &
           'Spherical inner region : ', &
           ' radius                     (SRDIST) =',SRDIST,' [Angs]', &
           ' center       (RRXCEN,RRYCEN,RRZCEN) =',RRXCEN,RRYCEN,RRZCEN
         WRITE(OUTU,101)
      ENDIF
!
!
!     Non-polar cavity potential setup
      QCAVI  = INDXA(COMLYN,COMLEN,'CAVI') .GT. 0
      IF(QCAVI) CALL WRNDIE(-5, '<PREP2_SMBP>', 'CAVI NYI for SMBP')
!
      QPBEQ=.TRUE.
!
 101  FORMAT(3X,A,I6,A)
 102  FORMAT(3X,A,F8.3,A,F8.3)
!
      RETURN
      END SUBROUTINE PREP2_SMBP

!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      SUBROUTINE SPRC_SMBP1(NATOM,X,Y,Z,CG, &
                            LNCEL,LTRAN,LDCEL,LXBCEN,LYBCEN,LZBCEN, &
                            QPRIN,PHIRF,QVAC,QSOL)
!-----------------------------------------------------------------------
!     The MM reaction field potential of the region of interest
!     is calculated using focussing (adapted from SPHE_GSBP1) 
!
!
      use chm_kinds
      use memory
      use number
      use stream
      use dimens_fcm
      use consta
      use timerm
      use machutil,only:wrttim
      implicit none

      real(chm_real)  X(*),Y(*),Z(*),CG(*)
      real(chm_real)  LTRAN,LDCEL,LXBCEN,LYBCEN,LZBCEN
      real(chm_real4) PHIRF(*)
      INTEGER NATOM
      INTEGER LNCEL
      LOGICAL QPRIN,QVAC,QSOL
! local
      real(chm_real4),allocatable,dimension(:) :: ILCHC,IPHIB
      INTEGER NFIL,IXCEN,IYCEN,IZCEN,JX1,JX2,JY1,JY2,JZ1,JZ2
      INTEGER LNCEL3,BNCEL

      QA=.true.
      QB=.false.

      IF (QRECTBOX) THEN
        JX1=INT((TRANX+RBXMIN-XBCEN)/DCEL)-1
        JX2=INT((TRANX+RBXMAX-XBCEN)/DCEL)+3
        JY1=INT((TRANY+RBYMIN-YBCEN)/DCEL)-1
        JY2=INT((TRANY+RBYMAX-YBCEN)/DCEL)+3
        JZ1=INT((TRANZ+RBZMIN-ZBCEN)/DCEL)-1
        JZ2=INT((TRANZ+RBZMAX-ZBCEN)/DCEL)+3
      ELSEIF (QSPHERE) THEN
        NFIL=INT(SRDIST/DCEL)+2
        IXCEN=INT((RRXCEN+TRANX-XBCEN)/DCEL)+1
        IYCEN=INT((RRYCEN+TRANY-YBCEN)/DCEL)+1
        IZCEN=INT((RRZCEN+TRANZ-ZBCEN)/DCEL)+1
        JX1=IXCEN-NFIL
        JX2=IXCEN+NFIL+1
        JY1=IYCEN-NFIL
        JY2=IYCEN+NFIL+1
        JZ1=IZCEN-NFIL
        JZ2=IZCEN+NFIL+1
      ENDIF

! allocate charge densities in the large box and
!          boundary potentials in the original box
      LNCEL3=LNCEL*LNCEL*LNCEL
      call chmalloc('pbeq.src','SPRC_SMBP1','ILCHC',LNCEL3,cr4=ILCHC)
      BNCEL=1
      IF(QFOCUS) BNCEL=2*(NCLX*NCLX+NCLY*NCLY+NCLZ*NCLZ)
      call chmalloc('pbeq.src','SPRC_SMBP1','IPHIB',BNCEL,cr4=IPHIB)


!------------------------------------------------------------------------
!     In vacuum (large box - focussing - original box)
!------------------------------------------------------------------------

      IF (QVAC) THEN
      IF (PRNLEV .GE. 100) &
        WRITE(OUTU, '(A)') '   Calculating El. Potential in Vacuum'

      IF(QFOCUS) THEN
! get the grid parameters in the lagre box
        CALL MAYER(NTPRP,LSTPRP,ONE,ONE,ZERO,X,Y,Z,CG, &
             PBRAD,LNCEL,LNCEL,LNCEL,LDCEL,WATR,IONR,MAY, &
             MAYX,MAYY,MAYZ,SWIN,ILCHC, &
             IPHIP,ZERO,TMEMB,ZMEMB,ONE,NIMGB, &
             HTMEMB,ONE,HDENS,POSALP,NEGALP,QCHRG,POFFST,NOFFST, &
             DROPLET,ONE,XDROPLET,YDROPLET,ZDROPLET,Qdtom,Qdkap, &
             ONE,LXMAX,LYMAX,LZMAX,LXMIN,LYMIN,LZMIN,Qbtom,Qbkap, &
             ONE,Xcyln,Ycyln,Zcyln,Rcyln,Hcyln,Qctom,Qckap, &
             ONE,Xcone,Ycone,Zcone,Ax,By,Cz,Qectom,Qeckap, &
             -1,-1,&
             LTRAN,LTRAN,LTRAN,LXBCEN,LYBCEN,LZBCEN, &
             IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
             QKEEP,QREEN,QSMTH,QBSPL,QA,QB, &
             RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
             RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
             MAJOR,MINOR,QPROLATE, &
             .false.,.true.,.false.,IPHIB,QPBC,QNPBC, &
                              !              QINTBP QZERO
             QNONLINEAR,QPARTLINEAR,QFKAP, &
             IPOX,IPOY,IPOZ,MAPT,NATOM)
        ! get boundary potentials in the large box: CG -> ICHC
        IF(.NOT.QINTBP.AND..NOT.QPBC.AND..NOT.QZERO)THEN
           CALL MAYER_BP_ALL(NTPRP,LSTPRP,X,Y,Z,CG,ONE,ZERO, &
                TMEMB,ILCHC,IPHIP,LNCEL,LNCEL,LNCEL,LDCEL, &
                LTRAN,LTRAN,LTRAN,LXBCEN,LYBCEN,LZBCEN, &
                RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
                RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
                MAJOR,MINOR,QPROLATE, &
                LNCEL,LNCEL,LNCEL,LDCEL,LXBCEN,LYBCEN,LZBCEN, &
                NIMGB,QNPBC,QA,QB)
        ELSEIF(QINTBP) THEN
           CALL MAYER_BP_INT(NTPRP,LSTPRP,X,Y,Z,CG,ONE,ZERO, &
                TMEMB,ILCHC,IPHIP,LNCEL,LNCEL,LNCEL,LDCEL, &
                LTRAN,LTRAN,LTRAN,LXBCEN,LYBCEN,LZBCEN, &
                RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
                RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
                MAJOR,MINOR,QPROLATE, &
                LNCEL,LNCEL,LNCEL,LDCEL,LXBCEN,LYBCEN,LZBCEN, &
                NIMGB,QNPBC,QA,QB)
        ENDIF

        IF(TIMER.GT.1) &
             CALL WRTTIM('Boundary potential in large box in vacuum:')

        IF(EPSP.ne.ONE) THEN
! vacuum environment in region B.
          IF (QRECTBOX) THEN
            CALL RECT_MAYER(LNCEL,LNCEL,LNCEL,LTRAN,LTRAN,LTRAN,LDCEL, &
                 LXBCEN,LYBCEN,LZBCEN, &
                 MAY,MAYX,MAYY,MAYZ, &
                 RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX)
          ELSEIF (QSPHERE) THEN
            CALL SPHE_MAYER(LNCEL,LNCEL,LNCEL,LTRAN,LTRAN,LTRAN,LDCEL, &
                 LXBCEN,LYBCEN,LZBCEN, &
                 MAY,MAYX,MAYY,MAYZ, &
                 RRXCEN,RRYCEN,RRZCEN,SRDIST)
          ENDIF
! redefine the maximum region outside which EPS is ONE. (see subroutine MAYER)
          IXMAX=LNCEL-1
          IYMAX=LNCEL-1
          IZMAX=LNCEL-1
          IXMIN=2
          IYMIN=2
          IZMIN=2
        ENDIF

        ! solve PBEQ in the large box
        ! XIAO_PHK_QC_UW0609: consistent with PB reference 
        !        CALL PBEQ1(MAXITS,DOME,DEPS,ONE,EPSP,ONE,ZERO, &
        CALL PBEQ1(MAXITS,DOME,DEPS,ONE,ONE,ONE,ZERO, &
             LNCEL,LNCEL,LNCEL,IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
             IPHIP,MAYX,MAYY,MAYZ, &
             ILCHC,MAY,TMEMB, &
             .false.,.false.,QPBC,QNPBC,.false.)
        IF(TIMER.GT.1) &
             CALL WRTTIM('PBEQ solver in large box in vacuum:')

        ! construct boundary potentials from the large box
        CALL BOUNDPOT(LNCEL,LNCEL,LNCEL,LDCEL, &
             LTRAN,LTRAN,LTRAN,LXBCEN,LYBCEN,LZBCEN, &
             NCLX,NCLY,NCLZ,DCEL,XBCEN,YBCEN,ZBCEN, &
             IPHIP,IPHIB)

        ! get the grid parameters in the original box
        CALL MAYER(NTPRP,LSTPRP,ONE,ONE,ZERO,X,Y,Z,CG, &
             PBRAD,NCLX,NCLY,NCLZ,DCEL,WATR,IONR,MAY, &
             MAYX,MAYY,MAYZ,SWIN,ICHC, &
             IPHIP,ZERO,TMEMB,ZMEMB,ONE,NIMGB, &
             HTMEMB,ONE,HDENS,POSALP,NEGALP,QCHRG,POFFST,NOFFST, &
             DROPLET,ONE,XDROPLET,YDROPLET,ZDROPLET,Qdtom,Qdkap, &
             ONE,LXMAX,LYMAX,LZMAX,LXMIN,LYMIN,LZMIN,Qbtom,Qbkap, &
             ONE,Xcyln,Ycyln,Zcyln,Rcyln,Hcyln,Qctom,Qckap, &
             ONE,Xcone,Ycone,Zcone,Ax,By,Cz,Qectom,Qeckap, &
             -1,-1,&
             TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
             IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
             QKEEP,QREEN,QSMTH,QBSPL,QA,QB, &
             RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
             RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
             MAJOR,MINOR,QPROLATE, &
             .false.,.false.,.true.,IPHIB,QPBC,QNPBC, &
                              !              QINTBP QZERO
             QNONLINEAR,QPARTLINEAR,QFKAP, &
             IPOX,IPOY,IPOZ,MAPT, NATOM)

        IF(EPSP.ne.ONE) THEN
! vacuum environment in region B.
          IF (QRECTBOX) THEN
            CALL RECT_MAYER(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,DCEL, &
                 XBCEN,YBCEN,ZBCEN, &
                 MAY,MAYX,MAYY,MAYZ, &
                 RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX) 
          ELSEIF (QSPHERE) THEN
            CALL SPHE_MAYER(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,DCEL, &
                 XBCEN,YBCEN,ZBCEN, &
                 MAY,MAYX,MAYY,MAYZ, &
                 RRXCEN,RRYCEN,RRZCEN,SRDIST)
          ENDIF
! Haibo GSBP (XIAO_QC_UW0609 for boundary)
! redefine the maximum region outside which EPS is EPSW. (see subroutine MAYER)
          IF(JX2.GT.IXMAX) IXMAX=JX2+1
          IF(JY2.GT.IYMAX) IYMAX=JY2+1
          IF(JZ2.GT.IZMAX) IZMAX=JZ2+1
          IF(JX1.LT.IXMIN) IXMIN=JX1-1
          IF(JY1.LT.IYMIN) IYMIN=JY1-1
          IF(JZ1.LT.IZMIN) IZMIN=JZ1-1
! Haibo GSBP
        ENDIF

        ! solve PBEQ in the original box
        ! XIAO_PHK_QC_UW0609: consistent with PB reference 
        !        CALL PBEQ1(MAXITS,DOME,DEPS,ONE,EPSP,ONE,ZERO, &
        CALL PBEQ1(MAXITS,DOME,DEPS,ONE,ONE,ONE,ZERO, &
             NCLX,NCLY,NCLZ,IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
             IPHIP,MAYX,MAYY,MAYZ, &
             ICHC,MAY,TMEMB, &
             .false.,QFOCUS,QPBC,QNPBC,.false.)
        IF(TIMER.GT.1) CALL WRTTIM('PBEQ solver in vacuum:')

      ELSE

! get the grid parameters in the original box
        CALL MAYER(NTPRP,LSTPRP,ONE,ONE,ZERO,X,Y,Z,CG, &
             PBRAD,NCLX,NCLY,NCLZ,DCEL,WATR,IONR,MAY, &
             MAYX,MAYY,MAYZ,SWIN,ICHC, &
             IPHIP,ZERO,TMEMB,ZMEMB,ONE,NIMGB, &
             HTMEMB,ONE,HDENS,POSALP,NEGALP,QCHRG,POFFST,NOFFST, &
             DROPLET,ONE,XDROPLET,YDROPLET,ZDROPLET,Qdtom,Qdkap, &
             ONE,LXMAX,LYMAX,LZMAX,LXMIN,LYMIN,LZMIN,Qbtom,Qbkap, &
             ONE,Xcyln,Ycyln,Zcyln,Rcyln,Hcyln,Qctom,Qckap, &
             ONE,Xcone,Ycone,Zcone,Ax,By,Cz,Qectom,Qeckap, &
             -1,-1,&
             TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
             IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
             QKEEP,QREEN,QSMTH,QBSPL,QA,QB, &
             RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
             RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
             MAJOR,MINOR,QPROLATE, &
             .false.,.true.,.false.,IPHIB,QPBC,QNPBC, &
                              !              QINTBP QZERO
             QNONLINEAR,QPARTLINEAR,QFKAP, &
             IPOX,IPOY,IPOZ,MAPT, NATOM)

        IF(.NOT.QINTBP.AND..NOT.QPBC.AND..NOT.QZERO)THEN
           CALL MAYER_BP_ALL(NTPRP,LSTPRP,X,Y,Z,CG,ONE,ZERO, &
                TMEMB,ILCHC,IPHIP,NCLX,NCLY,NCLZ,DCEL, &
                TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
                RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
                RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
                MAJOR,MINOR,QPROLATE, &
                LNCEL,LNCEL,LNCEL,LDCEL,LXBCEN,LYBCEN,LZBCEN, &
                NIMGB,QNPBC,QA,QB)
        ELSEIF(QINTBP) THEN
           CALL MAYER_BP_INT(NTPRP,LSTPRP,X,Y,Z,CG,ONE,ZERO, &
                TMEMB,ILCHC,IPHIP,NCLX,NCLY,NCLZ,DCEL, &
                TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
                RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
                RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
                MAJOR,MINOR,QPROLATE, &
                LNCEL,LNCEL,LNCEL,LDCEL,LXBCEN,LYBCEN,LZBCEN, &
                NIMGB,QNPBC,QA,QB)
        ENDIF

        IF(TIMER.GT.1) &
        CALL WRTTIM('Boundary potential with large box in vacuum:')

        IF(EPSP.ne.ONE) THEN
! vacuum environment in region B.
          IF (QRECTBOX) THEN
            CALL RECT_MAYER(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,DCEL, &
                 XBCEN,YBCEN,ZBCEN, &
                 MAY,MAYX,MAYY,MAYZ, &
                 RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX)
          ELSEIF (QSPHERE) THEN
            CALL SPHE_MAYER(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,DCEL, &
                 XBCEN,YBCEN,ZBCEN, &
                 MAY,MAYX,MAYY,MAYZ, &
                 RRXCEN,RRYCEN,RRZCEN,SRDIST)
          ENDIF
! Haibo GSBP (XIAO_QC_UW0609 for boundary)
! redefine the maximum region outside which EPS is EPSW. (see subroutine MAYER)
          IF(JX2.GT.IXMAX) IXMAX=JX2+1
          IF(JY2.GT.IYMAX) IYMAX=JY2+1
          IF(JZ2.GT.IZMAX) IZMAX=JZ2+1
          IF(JX1.LT.IXMIN) IXMIN=JX1-1
          IF(JY1.LT.IYMIN) IYMIN=JY1-1
          IF(JZ1.LT.IZMIN) IZMIN=JZ1-1
! Haibo GSBP
        ENDIF

        ! solve PBEQ in the original box
        ! XIAO_PHK_QC_UW0609: consistent with PB reference 
        !        CALL PBEQ1(MAXITS,DOME,DEPS,ONE,EPSP,ONE,ZERO, &
        CALL PBEQ1(MAXITS,DOME,DEPS,ONE,ONE,ONE,ZERO, &
             NCLX,NCLY,NCLZ,IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
             IPHIP,MAYX,MAYY,MAYZ, &
             ICHC,MAY,TMEMB, &
             .false.,QFOCUS,QPBC,QNPBC,.false.)
        IF(TIMER.GT.1) CALL WRTTIM('PBEQ solver in vacuum:')

      ENDIF
      ENDIF ! QVAC
!------------------------------------------------------------------------
!     In solution (large box - focussing - original box)
!------------------------------------------------------------------------

      IF (QSOL) THEN
      IF (PRNLEV .GE. 100) THEN
        WRITE(OUTU, '(A)') ' '
        WRITE(OUTU, '(A)') '   Calculating El. Potential in Solution'
      ENDIF

      IF(QFOCUS) THEN
! get grid parameters in the large box
        CALL MAYER(NTPRP,LSTPRP,EPSW,EPSP,KAPPA2,X,Y,Z,CG, &
             PBRAD,LNCEL,LNCEL,LNCEL,LDCEL,WATR,IONR,MAY, &
             MAYX,MAYY,MAYZ,SWIN,ILCHC, &
             IPHIW,VMEMB,TMEMB,ZMEMB,EPSM,NIMGB, &
             HTMEMB,EPSH,HDENS,POSALP,NEGALP,QCHRG,POFFST,NOFFST, &
             DROPLET,EPSD,XDROPLET,YDROPLET,ZDROPLET,Qdtom,Qdkap, &
             EPSB,LXMAX,LYMAX,LZMAX,LXMIN,LYMIN,LZMIN,Qbtom,Qbkap, &
             EPSC,Xcyln,Ycyln,Zcyln,Rcyln,Hcyln,Qctom,Qckap, &
             EPSEC,Xcone,Ycone,Zcone,Ax,By,Cz,Qectom,Qeckap, &
             -1,-1,&
             LTRAN,LTRAN,LTRAN,LXBCEN,LYBCEN,LZBCEN, &
             IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
             QKEEP,QREEN,QSMTH,QBSPL,QA,QB, &
             RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
             RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
             MAJOR,MINOR,QPROLATE, &
             .false.,.true.,.false.,IPHIB,QPBC,QNPBC, &
                              !              QINTBP QZERO
             QNONLINEAR,QPARTLINEAR,QFKAP, &
             IPOX,IPOY,IPOZ,MAPT,NATOM)

! get boundary potentials in the large box: CG -> ICHC
        IF(.NOT.QINTBP.AND..NOT.QPBC.AND..NOT.QZERO)THEN
           CALL MAYER_BP_ALL(NTPRP,LSTPRP,X,Y,Z,CG,EPSW,KAPPA2, &
                TMEMB,ILCHC,IPHIW,LNCEL,LNCEL,LNCEL,LDCEL, &
                LTRAN,LTRAN,LTRAN,LXBCEN,LYBCEN,LZBCEN, &
                RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
                RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
                MAJOR,MINOR,QPROLATE, &
                LNCEL,LNCEL,LNCEL,LDCEL,LXBCEN,LYBCEN,LZBCEN, &
                NIMGB,QNPBC,QA,QB)
        ELSEIF(QINTBP) THEN
           CALL MAYER_BP_INT(NTPRP,LSTPRP,X,Y,Z,CG,EPSW,KAPPA2, &
                TMEMB,ILCHC,IPHIW,LNCEL,LNCEL,LNCEL,LDCEL, &
                LTRAN,LTRAN,LTRAN,LXBCEN,LYBCEN,LZBCEN, &
                RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
                RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
                MAJOR,MINOR,QPROLATE, &
                LNCEL,LNCEL,LNCEL,LDCEL,LXBCEN,LYBCEN,LZBCEN, &
                NIMGB,QNPBC,QA,QB)
        ENDIF
        IF(TIMER.GT.1) &
             CALL WRTTIM('Boundary potential in large box in solvent:')

! vacuum environment in region B.
        IF (QRECTBOX) THEN
          CALL RECT_MAYER(LNCEL,LNCEL,LNCEL,LTRAN,LTRAN,LTRAN,LDCEL, &
               LXBCEN,LYBCEN,LZBCEN, &
               MAY,MAYX,MAYY,MAYZ, &
               RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX)
        ELSEIF (QSPHERE) THEN
          CALL SPHE_MAYER(LNCEL,LNCEL,LNCEL,LTRAN,LTRAN,LTRAN,LDCEL, &
               LXBCEN,LYBCEN,LZBCEN, &
               MAY,MAYX,MAYY,MAYZ, &
               RRXCEN,RRYCEN,RRZCEN,SRDIST)
        ENDIF

! redefine the maximum region outside which EPS is EPSW. (see subroutine MAYER)
        IXMAX=LNCEL-1
        IYMAX=LNCEL-1
        IZMAX=LNCEL-1
        IXMIN=2
        IYMIN=2
        IZMIN=2

! solve PBEQ in the large box
        CALL PBEQ1(MAXITS,DOME,DEPS,EPSW,EPSP,EPSM,KAPPA2, &
             LNCEL,LNCEL,LNCEL,IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
             IPHIW,MAYX,MAYY,MAYZ, &
             ILCHC,MAY,TMEMB, &
             QOSOR,.false.,QPBC,QNPBC,.false.)
        IF(TIMER.GT.1) &
             CALL WRTTIM('PBEQ solver in large box in solvent:')

        ! construct boundary potentials from the large box
        CALL BOUNDPOT(LNCEL,LNCEL,LNCEL,LDCEL, &
             LTRAN,LTRAN,LTRAN,LXBCEN,LYBCEN,LZBCEN, &
             NCLX,NCLY,NCLZ,DCEL,XBCEN,YBCEN,ZBCEN, &
             IPHIW,IPHIB)

        ! get the grid parameters in the original box
        CALL MAYER(NTPRP,LSTPRP,EPSW,EPSP,KAPPA2,X,Y,Z,CG, &
             PBRAD,NCLX,NCLY,NCLZ,DCEL,WATR,IONR,MAY, &
             MAYX,MAYY,MAYZ,SWIN,ICHC, &
             IPHIW,VMEMB,TMEMB,ZMEMB,EPSM,NIMGB, &
             HTMEMB,EPSH,HDENS,POSALP,NEGALP,QCHRG,POFFST,NOFFST, &
             DROPLET,EPSD,XDROPLET,YDROPLET,ZDROPLET,Qdtom,Qdkap, &
             EPSB,LXMAX,LYMAX,LZMAX,LXMIN,LYMIN,LZMIN,Qbtom,Qbkap, &
             EPSC,Xcyln,Ycyln,Zcyln,Rcyln,Hcyln,Qctom,Qckap, &
             EPSEC,Xcone,Ycone,Zcone,Ax,By,Cz,Qectom,Qeckap, &
             -1,-1,&
             TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
             IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
             QKEEP,QREEN,QSMTH,QBSPL,QA,QB, &
             RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
             RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
             MAJOR,MINOR,QPROLATE, &
             .false.,.false.,.true.,IPHIB,QPBC,QNPBC, &
                              !              QINTBP QZERO
             QNONLINEAR,QPARTLINEAR,QFKAP, &
             IPOX,IPOY,IPOZ,MAPT, NATOM)

! vacuum environment in region B.
        IF (QRECTBOX) THEN
          CALL RECT_MAYER(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,DCEL, &
               XBCEN,YBCEN,ZBCEN, &
               MAY,MAYX,MAYY,MAYZ, &
               RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX)
        ELSEIF (QSPHERE) THEN
          CALL SPHE_MAYER(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,DCEL, &
               XBCEN,YBCEN,ZBCEN, &
               MAY,MAYX,MAYY,MAYZ, &
               RRXCEN,RRYCEN,RRZCEN,SRDIST)
        ENDIF

! redefine the maximum region outside which EPS is EPSW. (see subroutine MAYER)
        IF(JX2.GT.IXMAX) IXMAX=JX2+1
        IF(JY2.GT.IYMAX) IYMAX=JY2+1
        IF(JZ2.GT.IZMAX) IZMAX=JZ2+1
        IF(JX1.LT.IXMIN) IXMIN=JX1-1
        IF(JY1.LT.IYMIN) IYMIN=JY1-1
        IF(JZ1.LT.IZMIN) IZMIN=JZ1-1

! solve PBEQ in the original box
        CALL PBEQ1(MAXITS,DOME,DEPS,EPSW,EPSP,EPSM,KAPPA2, &
             NCLX,NCLY,NCLZ,IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
             IPHIW,MAYX,MAYY,MAYZ, &
             ICHC,MAY,TMEMB, &
             QOSOR,QFOCUS,QPBC,QNPBC,.false.)
        IF(TIMER.GT.1) CALL WRTTIM('PBEQ solver in solvent:')

      ELSE

! get the grid parameters in the original box
        CALL MAYER(NTPRP,LSTPRP,EPSW,EPSP,KAPPA2,X,Y,Z,CG, &
             PBRAD,NCLX,NCLY,NCLZ,DCEL,WATR,IONR,MAY, &
             MAYX,MAYY,MAYZ,SWIN,ICHC, &
             IPHIW,VMEMB,TMEMB,ZMEMB,EPSM,NIMGB, &
             HTMEMB,EPSH,HDENS,POSALP,NEGALP,QCHRG,POFFST,NOFFST, &
             DROPLET,EPSD,XDROPLET,YDROPLET,ZDROPLET,Qdtom,Qdkap, &
             EPSB,LXMAX,LYMAX,LZMAX,LXMIN,LYMIN,LZMIN,Qbtom,Qbkap, &
             EPSC,Xcyln,Ycyln,Zcyln,Rcyln,Hcyln,Qctom,Qckap, &
             EPSEC,Xcone,Ycone,Zcone,Ax,By,Cz,Qectom,Qeckap, &
             -1,-1,&
             TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
             IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
             QKEEP,QREEN,QSMTH,QBSPL,QA,QB, &
             RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
             RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
             MAJOR,MINOR,QPROLATE, &
             .false.,.true.,.false.,IPHIB,QPBC,QNPBC, &
                              !              QINTBP QZERO
             QNONLINEAR,QPARTLINEAR,QFKAP, &
             IPOX,IPOY,IPOZ,MAPT,NATOM)

        IF(.NOT.QINTBP.AND..NOT.QPBC.AND..NOT.QZERO)THEN
           CALL MAYER_BP_ALL(NTPRP,LSTPRP,X,Y,Z,CG,EPSW,KAPPA2, &
                TMEMB,ILCHC,IPHIW,NCLX,NCLY,NCLZ,DCEL, &
                TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
                RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
                RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
                MAJOR,MINOR,QPROLATE, &
                LNCEL,LNCEL,LNCEL,LDCEL,LXBCEN,LYBCEN,LZBCEN, &
                NIMGB,QNPBC,QA,QB)
        ELSEIF(QINTBP) THEN
           CALL MAYER_BP_INT(NTPRP,LSTPRP,X,Y,Z,CG,EPSW,KAPPA2, &
                TMEMB,ILCHC,IPHIW,NCLX,NCLY,NCLZ,DCEL, &
                TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
                RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
                RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
                MAJOR,MINOR,QPROLATE, &
                LNCEL,LNCEL,LNCEL,LDCEL,LXBCEN,LYBCEN,LZBCEN, &
                NIMGB,QNPBC,QA,QB)
        ENDIF
        IF(TIMER.GT.1) &
               CALL WRTTIM('Boundary potential with large box in solvent:')

! vacuum environment in region B.
        IF (QRECTBOX) THEN
          CALL RECT_MAYER(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,DCEL, &
               XBCEN,YBCEN,ZBCEN, &
               MAY,MAYX,MAYY,MAYZ, &
               RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX)
        ELSEIF (QSPHERE) THEN
          CALL SPHE_MAYER(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,DCEL, &
               XBCEN,YBCEN,ZBCEN, &
               MAY,MAYX,MAYY,MAYZ, &
               RRXCEN,RRYCEN,RRZCEN,SRDIST)
        ENDIF

! redefine the maximum region outside which EPS is EPSW. (see subroutine MAYER)
        IF(JX2.GT.IXMAX) IXMAX=JX2+1
        IF(JY2.GT.IYMAX) IYMAX=JY2+1
        IF(JZ2.GT.IZMAX) IZMAX=JZ2+1
        IF(JX1.LT.IXMIN) IXMIN=JX1-1
        IF(JY1.LT.IYMIN) IYMIN=JY1-1
        IF(JZ1.LT.IZMIN) IZMIN=JZ1-1

        CALL PBEQ1(MAXITS,DOME,DEPS,EPSW,EPSP,EPSM,KAPPA2, &
             NCLX,NCLY,NCLZ,IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
             IPHIW,MAYX,MAYY,MAYZ, &
             ICHC,MAY,TMEMB, &
             QOSOR,QFOCUS,QPBC,QNPBC,.false.)
        IF(TIMER.GT.1) CALL WRTTIM('PBEQ solver in solvent:')

      ENDIF
      ENDIF ! QSOL 
!------------------------------------------------------------------------
! Calculate the reaction field potential and store it in PHIRF
!------------------------------------------------------------------------

      IF (PRNLEV .GE. 100) THEN
        WRITE(OUTU, '(A)') ' '
        WRITE(OUTU, '(A)') '   Calculating RF Potential'
        WRITE(OUTU, '(A)') ' '
        WRITE(OUTU, '(A)') ' '
      ENDIF

      IF (QVAC .and. QSOL) THEN
        CALL DCOPYR4(NCEL3,IPHIW,1,PHIRF,1) 
        CALL ADDCTVR4(PHIRF,IPHIP,NCEL3,MINONE)
      ELSEIF (QVAC) THEN
        CALL DCOPYR4(NCEL3,IPHIP,1,PHIRF,1) 
      ELSEIF (QSOL) THEN
        CALL DCOPYR4(NCEL3,IPHIW,1,PHIRF,1) 
      ENDIF

      call chmdealloc('pbeq.src','SPRC_SMBP1','ILCHC',LNCEL3,cr4=ILCHC)
      QA=.false.
      QB=.false.
!
      RETURN
      END SUBROUTINE SPRC_SMBP1

!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

      SUBROUTINE SPRC_SMBP2(NATOM,X,Y,Z,CG,QPRIN,PHIRF,QVAC,QSOL)
!-----------------------------------------------------------------------
!     The MM reaction field potential of the region of interest
!     is calculated (adapted from SPHE_GSBP2) 
!
      use chm_kinds
      use memory
      use number
      use stream
      use dimens_fcm
      use consta
      use timerm
      use machutil,only:wrttim
      implicit none
!
      real(chm_real)  X(*),Y(*),Z(*),CG(*)
      real(chm_real4) PHIRF(*)
      INTEGER NATOM
      LOGICAL QPRIN,QVAC,QSOL
! local
      real(chm_real4),dimension(1) :: IPHIB !dummy
      INTEGER NFIL,IXCEN,IYCEN,IZCEN,JX1,JX2,JY1,JY2,JZ1,JZ2

      QA=.true.
      QB=.false.

      IF (QRECTBOX) THEN
        JX1=INT((TRANX+RBXMIN-XBCEN)/DCEL)-1
        JX2=INT((TRANX+RBXMAX-XBCEN)/DCEL)+3
        JY1=INT((TRANY+RBYMIN-YBCEN)/DCEL)-1
        JY2=INT((TRANY+RBYMAX-YBCEN)/DCEL)+3
        JZ1=INT((TRANZ+RBZMIN-ZBCEN)/DCEL)-1
        JZ2=INT((TRANZ+RBZMAX-ZBCEN)/DCEL)+3
      ELSEIF (QSPHERE) THEN
        NFIL=INT(SRDIST/DCEL)+2
        IXCEN=INT((RRXCEN+TRANX-XBCEN)/DCEL)+1
        IYCEN=INT((RRYCEN+TRANY-YBCEN)/DCEL)+1
        IZCEN=INT((RRZCEN+TRANZ-ZBCEN)/DCEL)+1
        JX1=IXCEN-NFIL
        JX2=IXCEN+NFIL+1
        JY1=IYCEN-NFIL
        JY2=IYCEN+NFIL+1
        JZ1=IZCEN-NFIL
        JZ2=IZCEN+NFIL+1
      ENDIF

!------------------------------------------------------------------------
!     In vacuum
!------------------------------------------------------------------------

      IF (QVAC) THEN
      IF (PRNLEV .GE. 100) & 
        WRITE(OUTU, '(A)') '   Calculating El. Potential in Vacuum'

! get the grid parameters
      CALL MAYER(NTPRP,LSTPRP,ONE,ONE,ZERO,X,Y,Z,CG, &
           PBRAD,NCLX,NCLY,NCLZ,DCEL,WATR,IONR,MAY, &
           MAYX,MAYY,MAYZ,SWIN,ICHC, &
           IPHIP,ZERO,TMEMB,ZMEMB,ONE,NIMGB, &
           HTMEMB,ONE,HDENS,POSALP,NEGALP,QCHRG,POFFST,NOFFST, &
           DROPLET,ONE,XDROPLET,YDROPLET,ZDROPLET,Qdtom,Qdkap, &
           ONE,LXMAX,LYMAX,LZMAX,LXMIN,LYMIN,LZMIN,Qbtom,Qbkap, &
           ONE,Xcyln,Ycyln,Zcyln,Rcyln,Hcyln,Qctom,Qckap, &
           ONE,Xcone,Ycone,Zcone,Ax,By,Cz,Qectom,Qeckap, &
              -1,-1,&
           TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
           IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
           QKEEP,QREEN,QSMTH,QBSPL,QA,QB, &
           RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
           RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
           MAJOR,MINOR,QPROLATE, &
           .false.,.true.,.false.,IPHIB,QPBC,QNPBC, &
                               !              QINTBP QZERO
           QNONLINEAR,QPARTLINEAR,QFKAP, &
           IPOX,IPOY,IPOZ,MAPT, NATOM)
     
      ! get boundary potentials : CG -> ICHC
      IF(.NOT.QINTBP.AND..NOT.QPBC.AND..NOT.QZERO)THEN
         CALL MAYER_BP_ALL(NTPRP,LSTPRP,X,Y,Z,CG,ONE,ZERO, &
              TMEMB,ICHC,IPHIP,NCLX,NCLY,NCLZ,DCEL, &
              TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
              RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
              RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
              MAJOR,MINOR,QPROLATE, &
              NCLX,NCLY,NCLZ,DCEL,XBCEN,YBCEN,ZBCEN, &
              NIMGB,QNPBC,QA,QB)
      ELSEIF(QINTBP) THEN
         CALL MAYER_BP_INT(NTPRP,LSTPRP,X,Y,Z,CG,ONE,ZERO, &
              TMEMB,ICHC,IPHIP,NCLX,NCLY,NCLZ,DCEL, &
              TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
              RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
              RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
              MAJOR,MINOR,QPROLATE, &
              NCLX,NCLY,NCLZ,DCEL,XBCEN,YBCEN,ZBCEN, &
              NIMGB,QNPBC,QA,QB)
      ENDIF
      IF(TIMER.GT.1) CALL WRTTIM('Boundary potential in vacuum:')
     
      IF(EPSP.ne.ONE) THEN
! vacuum environment in region B.
        IF (QRECTBOX) THEN
          CALL RECT_MAYER(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,DCEL, &
               XBCEN,YBCEN,ZBCEN, &
               MAY,MAYX,MAYY,MAYZ, &
               RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX)
        ELSEIF (QSPHERE) THEN
          CALL SPHE_MAYER(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,DCEL, &
               XBCEN,YBCEN,ZBCEN, &
               MAY,MAYX,MAYY,MAYZ, &
               RRXCEN,RRYCEN,RRZCEN,SRDIST)
        ENDIF
! Haibo GSBP (XIAO_QC_UW0609 for boundary)
! redefine the maximum region outside which EPS is EPSW. (see subroutine MAYER)
        IF(JX2.GT.IXMAX) IXMAX=JX2+1
        IF(JY2.GT.IYMAX) IYMAX=JY2+1
        IF(JZ2.GT.IZMAX) IZMAX=JZ2+1
        IF(JX1.LT.IXMIN) IXMIN=JX1-1
        IF(JY1.LT.IYMIN) IYMIN=JY1-1
        IF(JZ1.LT.IZMIN) IZMIN=JZ1-1
! Haibo GSBP
      ENDIF

! solve PBEQ
      ! solve PBEQ
      ! XIAO_PHK_QC_UW0609: consistent with PB reference 
      !        CALL PBEQ1(MAXITS,DOME,DEPS,ONE,EPSP,ONE,ZERO,
      CALL PBEQ1(MAXITS,DOME,DEPS,ONE,ONE,ONE,ZERO, &
           NCLX,NCLY,NCLZ,IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
           IPHIP,MAYX,MAYY,MAYZ, &
           ICHC,MAY,TMEMB, &
           .false.,.false.,QPBC,QNPBC,.false.)
      IF(TIMER.GT.1) CALL WRTTIM('PBEQ solver in vacuum:')
    
      ENDIF ! QVAC

!------------------------------------------------------------------------
!     In solution
!------------------------------------------------------------------------

      IF (QSOL) THEN
      IF (PRNLEV .GE. 100) THEN
        WRITE(OUTU, '(A)') ' ' 
        WRITE(OUTU, '(A)') '   Calculating El. Potential in Solution'
      ENDIF

! get grid parameters
      CALL MAYER(NTPRP,LSTPRP,EPSW,EPSP,KAPPA2,X,Y,Z,CG, &
           PBRAD,NCLX,NCLY,NCLZ,DCEL,WATR,IONR,MAY, &
           MAYX,MAYY,MAYZ,SWIN,ICHC, &
           IPHIW,VMEMB,TMEMB,ZMEMB,EPSM,NIMGB, &
           HTMEMB,EPSH,HDENS,POSALP,NEGALP,QCHRG,POFFST,NOFFST, &
           DROPLET,EPSD,XDROPLET,YDROPLET,ZDROPLET,Qdtom,Qdkap, &
           EPSB,LXMAX,LYMAX,LZMAX,LXMIN,LYMIN,LZMIN,Qbtom,Qbkap, &
           EPSC,Xcyln,Ycyln,Zcyln,Rcyln,Hcyln,Qctom,Qckap, &
           EPSEC,Xcone,Ycone,Zcone,Ax,By,Cz,Qectom,Qeckap, &
              -1,-1,&
           TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
           IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
           QKEEP,QREEN,QSMTH,QBSPL,QA,QB, &
           RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
           RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
           MAJOR,MINOR,QPROLATE, &
           .false.,.true.,.false.,IPHIB,QPBC,QNPBC, &
                               !              QINTBP QZERO
           QNONLINEAR,QPARTLINEAR,QFKAP, &
           IPOX,IPOY,IPOZ,MAPT,NATOM)

! get boundary potentials : CG -> ICHC
      IF(.NOT.QINTBP.AND..NOT.QPBC.AND..NOT.QZERO)THEN
         CALL MAYER_BP_ALL(NTPRP,LSTPRP,X,Y,Z,CG,EPSW,KAPPA2, &
              TMEMB,ICHC,IPHIW,NCLX,NCLY,NCLZ,DCEL, &
              TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
              RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
              RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
              MAJOR,MINOR,QPROLATE, &
              NCLX,NCLY,NCLZ,DCEL,XBCEN,YBCEN,ZBCEN, &
              NIMGB,QNPBC,QA,QB)
      ELSEIF(QINTBP) THEN
         CALL MAYER_BP_INT(NTPRP,LSTPRP,X,Y,Z,CG,EPSW,KAPPA2, &
              TMEMB,ICHC,IPHIW,NCLX,NCLY,NCLZ,DCEL, &
              TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN, &
              RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX,QRECTBOX, &
              RRXCEN,RRYCEN,RRZCEN,SRDIST,QSPHERE, &
              MAJOR,MINOR,QPROLATE, &
              NCLX,NCLY,NCLZ,DCEL,XBCEN,YBCEN,ZBCEN, &
              NIMGB,QNPBC,QA,QB)
      ENDIF
      IF(TIMER.GT.1) CALL WRTTIM('Boundary potential in solvent:')

! vacuum environment in region B.
      IF (QRECTBOX) THEN
         CALL RECT_MAYER(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,DCEL, &
              XBCEN,YBCEN,ZBCEN, &
              MAY,MAYX,MAYY,MAYZ, &
              RBXMIN,RBXMAX,RBYMIN,RBYMAX,RBZMIN,RBZMAX)
      ELSEIF (QSPHERE) THEN
        CALL SPHE_MAYER(NCLx,NCLy,NCLz,TRANX,TRANY,TRANZ,DCEL, &
             XBCEN,YBCEN,ZBCEN, &
             MAY,MAYX,MAYY,MAYZ, &
             RRXCEN,RRYCEN,RRZCEN,SRDIST)
      ENDIF
    

! redefine the maximum region outside which EPS is EPSW. (see subroutine MAYER)
      IF(JX2.GT.IXMAX) IXMAX=JX2+1
      IF(JY2.GT.IYMAX) IYMAX=JY2+1
      IF(JZ2.GT.IZMAX) IZMAX=JZ2+1
      IF(JX1.LT.IXMIN) IXMIN=JX1-1
      IF(JY1.LT.IYMIN) IYMIN=JY1-1
      IF(JZ1.LT.IZMIN) IZMIN=JZ1-1

! solve PBEQ
      CALL PBEQ1(MAXITS,DOME,DEPS,EPSW,EPSP,EPSM,KAPPA2, &
           NCLX,NCLY,NCLZ,IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN, &
           IPHIW,MAYX,MAYY,MAYZ, &
           ICHC,MAY,TMEMB, &
           QOSOR,.false.,QPBC,QNPBC,.false.)
      IF(TIMER.GT.1) CALL WRTTIM('PBEQ solver in solvent:')

      ENDIF ! QSOL 

!------------------------------------------------------------------------
! Calculate the reaction field potential and store it in PHIRF
!------------------------------------------------------------------------

      IF (PRNLEV .GE. 100) THEN 
        WRITE(OUTU, '(A)') ' ' 
        WRITE(OUTU, '(A)') '   Calculating RF Potential'
        WRITE(OUTU, '(A)') ' ' 
        WRITE(OUTU, '(A)') ' ' 
      ENDIF

      IF (QVAC .and. QSOL) THEN
        CALL DCOPYR4(NCEL3,IPHIW,1,PHIRF,1) 
        CALL ADDCTVR4(PHIRF,IPHIP,NCEL3,MINONE)
      ELSEIF (QVAC) THEN
        CALL DCOPYR4(NCEL3,IPHIP,1,PHIRF,1) 
      ELSEIF (QSOL) THEN
        CALL DCOPYR4(NCEL3,IPHIW,1,PHIRF,1) 
      ENDIF
        

      QA=.false.
      QB=.false.
!
      RETURN
      END SUBROUTINE SPRC_SMBP2

!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

      SUBROUTINE SPRC_SMBP3(EQM,NATOM,X,Y,Z,CG,NQM, &
                            LNCEL,LTRAN,LDCEL,LXBCEN,LYBCEN,LZBCEN, &
                            QPRIN,QDOSCC,QDOQM,QMLST,QMLSTINV, &
                            PHITOT,PHITOTSAVE,PHIRF,PHIRFOLD,PHIRFDIFF, &
                            SURFCRD,SURFCHR,MAXPTS,INICHRG)
!-----------------------------------------------------------------------
!     The QM reaction field potential of the region of interest
!     is calculated via an SCRF procedure 
!     Note: Surface charges must be allocated on entry
!
      use chm_kinds
      use memory
      use number
      use stream
      use dimens_fcm
      use consta
      use timerm
      use gamess_fcm
#if KEY_SCCDFTB==1
      use sccdftb 
#endif
      use machutil,only:wrttim
#if KEY_GCMC==1
      use gcmc 
#endif
      implicit none
!
      real(chm_real)  EQM
      real(chm_real)  INICHRG
      real(chm_real)  X(*),Y(*),Z(*),CG(*)
      real(chm_real)  SURFCRD(MAXPTS),SURFCHR(MAXPTS)
      real(chm_real)  LTRAN,LDCEL,LXBCEN,LYBCEN,LZBCEN
      real(chm_real4)  PHITOT(NCEL3OG)
      real(chm_real4)  PHITOTSAVE(NCEL3OG)
      real(chm_real4)  PHIRF(NCEL3)
      real(chm_real4)  PHIRFOLD(NCEL3)
      real(chm_real4)  PHIRFDIFF(NCEL3)
      INTEGER NATOM,NQM
      INTEGER LNCEL
      INTEGER MAXPTS
      INTEGER QMLST(NQM)
      INTEGER QMLSTINV(NATOM)
      LOGICAL QPRIN,QDOSCC,QDOQM
! local
      real(chm_real),allocatable,dimension(:) :: QMCHR
      real(chm_real),allocatable,dimension(:) :: CGSAVE

      real(chm_real)  PHIDIFF,PHITHRSH
      INTEGER NFIL,IXCEN,IYCEN,IZCEN,JX1,JX2,JY1,JY2,JZ1,JZ2
      INTEGER I,NUM,NUMIT,MXIT
      LOGICAL QSWITCHGUESS
! JZ debug
      real(chm_real)  EDBG
      LOGICAL DBGCALLED

      QSWITCHGUESS = .false. 
      DBGCALLED = .false.

      PHITHRSH = SMBPPTHR
      MXIT     = SMBPMXIT

      call chmalloc('pbeq.src','SPRC_SMBP3','QMCHR',NQM,crl=QMCHR)
      call chmalloc('pbeq.src','SPRC_SMBP3','CGSAVE',NATOM,crl=CGSAVE)

!     Save Current PHIQMTOT
      CALL DCOPYR4(NCEL3OG,PHITOT,1,PHITOTSAVE,1)

!     Restart point here
  666 CONTINUE

!   
!     1.1 Construct the initial phi_rf^QM with an initial guess for the
!        QM charges
!
      IF (PRNLEV .ge. 5) THEN
        WRITE(OUTU,100)''
        WRITE(OUTU,100)'Calculating initial QM RF potential'
        WRITE(OUTU,100)'-----------------------------------'
        WRITE(OUTU,100)''
      ENDIF
      CALL SPRC_QMRF_IGU(NATOM,X,Y,Z,NQM,QMLST,PHIRF,CG, &
                         LNCEL,LTRAN,LDCEL,LXBCEN,LYBCEN,LZBCEN, &
                         QPRIN,QDOSCC,QDOQM,QSWITCHGUESS)


!
!     1.2 Determine positions of surface charges and
!         initialize vector with charges (SURFCHR)
!
      IF (QRECTBOX) THEN
        IF (PRNLEV .ge. 5) THEN
          WRITE(OUTU,100)''
          WRITE(OUTU,100)'Determining surface charge pos. on cuboid'
          WRITE(OUTU,100)'-----------------------------------------'
          WRITE(OUTU,100)''
        ENDIF
        CALL RECT_SURFACE_CHRG(SURFCRD,SURFCHR,INICHRG)
      ELSEIF (QSPHERE) THEN
        IF (PRNLEV .ge. 5) THEN
          WRITE(OUTU,100)''
          WRITE(OUTU,100)'Determining surface charge pos. on sphere'
          WRITE(OUTU,100)'-----------------------------------------'
          WRITE(OUTU,100)''
        ENDIF
        CALL SPHE_SURFACE_CHRG(SURFCRD,SURFCHR,INICHRG)
      ENDIF
      IF (NUMSURF .gt. MAXPTS/3) THEN
        CALL WRNDIE(-5,'<SMBP>','Too many surface charges!')
      ENDIF


      IF (PRNLEV .ge. 5) THEN
        WRITE(OUTU,100)''
        WRITE(OUTU,100)'Starting SCRF-Calculation'
        WRITE(OUTU,100)'-------------------------'
      ENDIF
!     2. Start SCRF loop 
      PHIDIFF  = 1.0E0
      NUMIT    = 1
10    CONTINUE

      IF (PRNLEV .ge. 5) THEN
        WRITE(OUTU,100)''
        WRITE(OUTU,101)'### Iteration ', NUMIT, ' ###'
      ENDIF
!   
!     2.1 Assemble phi_tot^QM 
!

      CALL DCOPYR4(NCEL3,PHIRF,1,PHIRFOLD,1) 
      CALL DCOPYR4(NCEL3OG,PHITOTSAVE,1,PHITOT,1)
      CALL ADD_SMALL_LRG_GRID(PHITOT,PHIRF, &
                              NCEL3OG,NCEL3,DCELOG,DCEL, &
                              NCLXOG,NCLYOG,NCLZOG,NCLX,NCLY,NCLZ, &
                              HALF) 

!   
!     2.2 Project phi_tot^QM onto surface charges
!
      IF (PRNLEV .ge. 5) THEN 
        WRITE(OUTU,100) '### Projecting QM-Potential onto surface charges ###'
      ENDIF
      CALL SPRC_PROJ(NATOM,X,Y,Z,CG,PHITOT,SURFCRD,SURFCHR, &
                     NQM,QMLST,QPRIN,QDOSCC,QDOQM,QBSPL)

!
!     2.3 Calculate QM WF in the field of the surface charges AND
!         the inner-region MM charges and get QM charges
!
      IF (PRNLEV .ge. 5) THEN 
        WRITE(OUTU,100) '### Calculating QM-Wavefunction ###'
      ENDIF
      CALL SMBP_CALC_QM(EQM,NATOM,X,Y,Z,CG, &
                        SURFCRD,SURFCHR, &
                        NQM,QMLST,QMCHR,QPRIN,QDOSCC,QDOQM)

!
!     2.4 Calculate new phi_rf^QM based on new QM charges 
!
#if KEY_GAMESS==1
      CALL DCOPY(int8(NATOM),CG,1_8,CGSAVE,1_8)
#else
      CALL DCOPY(NATOM,CG,1,CGSAVE,1)
#endif
      CALL FILLR8(CG,NATOM,ZERO)
      DO I = 1, NQM
        CG(QMLST(I)) = QMCHR(I)
      ENDDO          
     
      CALL FILLR41(PHIRF,NCEL3,0.0D0)
      IF(QLBOX) THEN
        CALL SPRC_SMBP1(NATOM,X,Y,Z,CG,LNCEL,LTRAN,LDCEL, &
                        LXBCEN,LYBCEN,LZBCEN,QPRIN,PHIRF, &
                        .TRUE.,.TRUE.)
      ELSE 
        CALL SPRC_SMBP2(NATOM,X,Y,Z,CG,QPRIN,PHIRF,.TRUE.,.TRUE.)      
      ENDIF
#if KEY_GAMESS==1
      CALL DCOPY(int8(NATOM),CGSAVE,1_8,CG,1_8) ! Get actual charges back
#else
      CALL DCOPY(NATOM,CGSAVE,1,CG,1) ! Get actual charges back
#endif

!
!     2.5. Check phi_rf^QM for convergence
!
      DO I = 1, NCEL3
        PHIRFDIFF(I) = PHIRF(I) - PHIRFOLD(I)
      ENDDO
      CALL SNRMVECR4(PHIRFDIFF,PHIDIFF,NCEL3) 
      IF (PRNLEV .ge. 5) WRITE(OUTU,102) 'Delta Phi = ', PHIDIFF

!     Try restarting with zero charge guess if PHIDIFF gets too large
      IF (PHIDIFF .gt. THOSND .and. .not. QSWITCHGUESS) THEN 
       CALL WRNDIE(0,'<SMBP>', &
          'WARNING: PHIDIFF > 1000. Restarting with zero charge guess!')
       QSWITCHGUESS = .true.
       GOTO 666
      ENDIF

      NUMIT = NUMIT + 1
      IF (NUMIT .gt. MXIT) THEN
        CALL WRNDIE(-4,'<SMBP>','SCRF did not converge!')
        GOTO 20
      ENDIF

!     End of SCRF loop
      IF (PHIDIFF .ge. PHITHRSH) GOTO 10

20    CONTINUE
      IF (PRNLEV .ge. 5) WRITE(OUTU,103) 'Final QM Energy = ', EQM

      IF (QSWITCHGUESS) QSWITCHGUESS = .false.
!
      call chmdealloc('pbeq.src','SPRC_SMBP3','QMCHR',NQM,crl=QMCHR)
      call chmdealloc('pbeq.src','SPRC_SMBP3','CGSAVE',NATOM,crl=CGSAVE)
!
100   FORMAT(3X,A)
101   FORMAT(3X,A,I4,A)
102   FORMAT(3X,A,E10.2)
103   FORMAT(3X,A,F16.2)
      RETURN
      END SUBROUTINE SPRC_SMBP3

!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

      SUBROUTINE SPRC_SMBP4(NATOM,X,Y,Z,DX,DY,DZ,CG,NQM, &
                            LNCEL,LTRAN,LDCEL,LXBCEN,LYBCEN,LZBCEN, &
                            QPRIN,QDOSCC,QDOQM,QMLST,QMLSTINV, &
                            PHITOTG,SURFCRD,SURFCHR, &
                            QMGX,QMGY,QMGZ,MAXPTS,INICHRG)
!-----------------------------------------------------------------------
!     The QM gradient contributions are calculated and stored 
!     in QMGX,QMGY,QMGZ
!     Note: PHITOTG is assumed to be complete on entry
!     Note: Surface charges must be allocated on entry
!
      use chm_kinds
      use memory
      use number
      use stream
      use dimens_fcm
      use consta
      use timerm
      use gamess_fcm
#if KEY_SCCDFTB==1
      use sccdftb 
#endif
      use machutil,only:wrttim
#if KEY_GCMC==1
      use gcmc 
#endif
      implicit none
!
      real(chm_real)  INICHRG
      real(chm_real)  X(*),Y(*),Z(*),DX(*),DY(*),DZ(*),CG(*)
      real(chm_real)  SURFCRD(MAXPTS),SURFCHR(MAXPTS)
      real(chm_real)  LTRAN,LDCEL,LXBCEN,LYBCEN,LZBCEN
      real(chm_real)  QMGX(*),QMGY(*),QMGZ(*)
      real(chm_real4)  PHITOTG(NCEL3OG)
      INTEGER NATOM,NQM
      INTEGER LNCEL
      INTEGER MAXPTS
      INTEGER QMLST(NQM)
      INTEGER QMLSTINV(NATOM)
      LOGICAL QPRIN,QDOSCC,QDOQM
! local
      real(chm_real),allocatable,dimension(:) :: QMCHR
      real(chm_real),allocatable,dimension(:) :: CGSAVE

      INTEGER NFIL,IXCEN,IYCEN,IZCEN,JX1,JX2,JY1,JY2,JZ1,JZ2
      INTEGER I,NUM,NUMIT
! JZ debug
      real(chm_real)  EDBG

 
      call chmalloc('pbeq.src','SPRC_SMBP4','QMCHR',NQM,crl=QMCHR)
      call chmalloc('pbeq.src','SPRC_SMBP4','CGSAVE',NATOM,crl=CGSAVE)

!
!     1.1 Project PHITOTG onto a set of gradient surface charges
!         Use the same # and positions of the energy surface charges
!
      IF (PRNLEV .ge. 5) THEN
        WRITE(OUTU,100)''
        WRITE(OUTU,100)'Projecting grad. pot. onto surf. charges'
        WRITE(OUTU,100)'-----------------------------------------'
        WRITE(OUTU,100)''
      ENDIF
      CALL FILLR8(SURFCHR,NUMSURF,INICHRG)
      CALL SPRC_PROJ(NATOM,X,Y,Z,CG,PHITOTG,SURFCRD,SURFCHR, &
                     NQM,QMLST,QPRIN,QDOSCC,QDOQM,QBSPL)

!
!     1.2 Calculate QM gradient in the field of the surface charges AND
!         the inner-region MM charges and get QM charges and
!         store QM gradient in QMGX,... arrays
!
      IF (PRNLEV .ge. 5) THEN
        WRITE(OUTU,100)'Calculating QM gradient'
        WRITE(OUTU,100)''
      ENDIF

#if KEY_SCCDFTB==1
      if (qmused_sccdftb) &
        SMBP_GRAD = .TRUE.   ! For Qm == SCC, sets maxiter to one within eglcao
#endif
      
#if KEY_QCHEM==1 || KEY_G09==1
      QSMBP_QC_GRAD = .TRUE.  ! For QM == Q-Chem, makes Q-Chem stop after first iteration
#endif
      
      ! (last density is read in)
      CALL SMBP_CALC_QM_GRAD(NTRB,LSTRB,NATOM,X,Y,Z,DX,DY,DZ,CG, &
                             SURFCRD,SURFCHR,NUMSURF, &
                             QMGX,QMGY,QMGZ, &
                             NQM,QMLST,QMCHR,QPRIN,QDOSCC,QDOQM)
      
#if KEY_SCCDFTB==1
      if (qmused_sccdftb) &
        SMBP_GRAD = .FALSE. 
#endif
      
#if KEY_QCHEM==1 || KEY_G09==1
      QSMBP_QC_GRAD = .FALSE. 
#endif

!
      call chmdealloc('pbeq.src','SPRC_SMBP4','QMCHR',NQM,crl=QMCHR)
      call chmdealloc('pbeq.src','SPRC_SMBP4','CGSAVE',NATOM,crl=CGSAVE)
!
100   FORMAT(3X,A)
101   FORMAT(3X,A,I4,A)
      RETURN
      END SUBROUTINE SPRC_SMBP4

!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

      SUBROUTINE SMBP_SUM_GRAD(NTR,LSTR,FACTOR,METHOD, &
                               AGRADX,AGRADY,AGRADZ, &
                               BGRADX,BGRADY,BGRADZ &
#if KEY_GCMC==1
                              ,GCMCON & 
#endif
                               )
!-----------------------------------------------------------------------
!     Sum gradient contributions 
!     
!
      use chm_kinds
      use memory
      use number
      use dimens_fcm
      implicit none
!
      real(chm_real)  AGRADX(*),AGRADY(*),AGRADZ(*)
      real(chm_real)  BGRADX(*),BGRADY(*),BGRADZ(*)
      real(chm_real)  FACTOR
      INTEGER LSTR(*)
      INTEGER NTR,METHOD
#if KEY_GCMC==1
      LOGICAL  GCMCON(:) 
#endif
! local
      INTEGER I,J

 
      IF (METHOD .eq. 1) THEN
        DO I = 1, NTR
          J = LSTR(I)
#if KEY_GCMC==1
          IF (GCMCON(J)) THEN 
#endif
            AGRADX(J) = AGRADX(J) + FACTOR*BGRADX(J)
            AGRADY(J) = AGRADY(J) + FACTOR*BGRADY(J)
            AGRADZ(J) = AGRADZ(J) + FACTOR*BGRADZ(J)
#if KEY_GCMC==1
          ENDIF 
#endif
        ENDDO        
      ELSEIF (METHOD .eq. 2) THEN
        DO I = 1, NTR
          J = LSTR(I)
#if KEY_GCMC==1
          IF (GCMCON(J)) THEN 
#endif
            AGRADX(J) = AGRADX(J) + FACTOR*BGRADX(I)
            AGRADY(J) = AGRADY(J) + FACTOR*BGRADY(I)
            AGRADZ(J) = AGRADZ(J) + FACTOR*BGRADZ(I)
#if KEY_GCMC==1
          ENDIF 
#endif
        ENDDO        
      ELSE
        CALL WRNDIE(-5,'<SMBP_SUM_GRAD>','WRONG METHOD ON INPUT')
      ENDIF

      RETURN
      END SUBROUTINE SMBP_SUM_GRAD


!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

      SUBROUTINE SMBP_QM_LST(NATOM,NQM,QMLST,QMLSTINV)
!-----------------------------------------------------------------------
!     Create index list for QM-atoms 
!     
!
      use chm_kinds
      use memory
      use number
      use dimens_fcm
      use gamess_fcm
      implicit none
!
      INTEGER QMLST(*),QMLSTINV(*) 
      INTEGER NATOM,NQM
! local
      INTEGER I,NUM

      NUM = 1
      DO I = 1, NATOM
        IF (IGMSEL(I) .eq. 1 .OR. IGMSEL(I) .eq. 2) THEN
          QMLST(NUM)  = I
          QMLSTINV(I) = NUM
          NUM = NUM + 1
        ELSE
          QMLSTINV(I) = -1
        ENDIF
      ENDDO

      IF (NUM-1 .ne. NQM) CALL WRNDIE(-5,'<SMBP_QM_LST>','NUM != NQM')

      RETURN
      END SUBROUTINE SMBP_QM_LST


!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

      SUBROUTINE SPRC_QMRF_IGU(NATOM,X,Y,Z,NQM,QMLST,PHI,CG, &
                               LNCEL,LTRAN,LDCEL,LXBCEN,LYBCEN,LZBCEN, &
                               QPRIN,QDOSCC,QDOQM,QSWITCHGUESS)
!-----------------------------------------------------------------------
!     Provide an initial guess for the QM charges and construct 
!     initial phi_rf^QM
!
      use chm_kinds
      use memory
      use number
      use stream
      use dimens_fcm
      use consta
      use timerm
      use gamess_fcm
#if KEY_SCCDFTB==1
      use sccdftb 
#endif
#if KEY_GCMC==1
      use gcmc    
#endif
      use machutil,only:wrttim
      implicit none
!
      real(chm_real4)  PHI(*)
      real(chm_real)  X(*),Y(*),Z(*),CG(*)
      real(chm_real)  LTRAN,LDCEL,LXBCEN,LYBCEN,LZBCEN
      INTEGER QMLST(*)
      INTEGER LNCEL
      INTEGER NATOM,NQM
      LOGICAL QPRIN,QDOSCC,QDOQM,QSWITCHGUESS
! local
      real(chm_real),allocatable,dimension(:) :: QMCHR
      real(chm_real),allocatable,dimension(:) :: CGSAVE
      INTEGER I,NUMQM
      LOGICAL QPREVCHR,LEXIST
! debug
      real(chm_real),allocatable, dimension (:) :: IRDUMMY
      real(chm_real) EDBG
      INTEGER DBGPRNT
      PARAMETER (DBGPRNT=1)

      QPREVCHR = .true.

      call chmalloc('pbeq.src','SPRC_QMRF_IGU','QMCHR',NQM,crl=QMCHR)
      call chmalloc('pbeq.src','SPRC_QMRF_IGU','CGSAVE',NATOM,crl=CGSAVE)

#if KEY_SCCDFTB==1 || KEY_QCHEM==1 || KEY_G09==1
      CALL DCOPY(NATOM,CG,1,CGSAVE,1)

#if KEY_SCCDFTB==1
      if (qmused_sccdftb) then
        IF (.not. LMULIK) CALL WRNDIE(-5,'<SMBP_QMRF_IGU>', &
                             'MULL switch needed for SMBP/SCC')
        IF (.not. SCC_CALLED) QPREVCHR = .false.
      end if
#endif 

!     Read charges from Q-chem if already there...
#if KEY_QCHEM==1
      IF (SMBPIGU .eq. 1) THEN
        IF (QCCHRG .eq. 1) THEN
          INQUIRE(FILE='mdc_charges.dat',EXIST=LEXIST)
          IF (LEXIST) THEN
            IF (PRNLEV.ge.10) write(OUTU,100)'Reading MDC charges'
            OPEN (UNIT=11, FILE='mdc_charges.dat', status='old')
            REWIND(11)
            DO I=1, NQM
              READ(11,'(3X,F19.16)')QMCHR(I)
            ENDDO
          ELSE
            QPREVCHR = .false.
          ENDIF
        ELSEIF (QCCHRG .eq. 2) THEN
          INQUIRE(FILE='charges.dat',EXIST=LEXIST)
          IF (LEXIST) THEN
            IF (PRNLEV.ge.10) write(OUTU,100)'Reading charges from Q-Chem'
            OPEN (UNIT=11, FILE='charges.dat', status='old')
            REWIND(11)
            DO I=1, NQM
              READ(11,'(3X,F19.16)')QMCHR(I)
            ENDDO
          ELSE
            QPREVCHR = .false.
          ENDIF
        ELSE
          CALL WRNDIE(-5,'<SMBP_CALC_QM>','QCCH must be 1 or 2')
        ENDIF
      ENDIF
#endif 

!     Read charges from Gaussian if already there...
#if KEY_G09==1
      IF (SMBPIGU .eq. 1) THEN
        IF (QCCHRG .eq. 1) THEN
          INQUIRE(FILE='esp_charges.dat',EXIST=LEXIST)
          IF (LEXIST) THEN
            IF (PRNLEV.ge.10) write(OUTU,100)'Reading ESP charges'
            OPEN (UNIT=11, FILE='esp_charges.dat', status='old')
            REWIND(11)
            DO I=1, NQM
              READ(11,'(E16.9)')QMCHR(I)
            ENDDO
          ELSE
            QPREVCHR = .false.
          ENDIF
        ELSEIF (QCCHRG .eq. 2) THEN
          INQUIRE(FILE='charges.dat',EXIST=LEXIST)
          IF (LEXIST) THEN
            IF (PRNLEV.ge.10) write(OUTU,100)'Reading Mulliken charges'
            OPEN (UNIT=11, FILE='charges.dat', status='old')
            REWIND(11)
            DO I=1, NQM
              READ(11,'(E16.9)')QMCHR(I)
            ENDDO
          ELSE
            QPREVCHR = .false.
          ENDIF
        ELSE
          CALL WRNDIE(-5,'<SMBP_CALC_QM>','QCCH must be 1 or 2')
        ENDIF
      ENDIF
#endif 

!     --------------------------------------------------------------------------
!     Create the guess charges
!     --------------------------------------------------------------------------
!     Note: If QSWITCHGUESS is true, always use zero-charge guess (SMBPIGU = 2)
!     Default iguess from previous SCC/Q-Chem run
      IF (.not. QSWITCHGUESS .and. (QPREVCHR .and. SMBPIGU .eq. 1)) THEN
        NUMQM = 1
        DO I = 1, NATOM
          IF (IGMSEL(I) .eq. 1 .OR. IGMSEL(I) .eq. 2) THEN
            IF (QDOSCC) THEN     
#if KEY_SCCDFTB==1
              if (qmused_sccdftb) then
                CG(I) = QMULIK(NUMQM)
                NUMQM = NUMQM + 1
              end if
#endif 
            ELSEIF(QDOQM) THEN
#if KEY_QCHEM==1 || KEY_G09==1
              CG(I) = QMCHR(NUMQM)
              NUMQM = NUMQM + 1
#endif 
            ELSE
              CALL WRNDIE(-5,'<SPHERE_QMRF_IGU>', &
                          'Either QDOSCC or QDOQM must be true')
            ENDIF
          ELSE 
            CG(I) = ZERO
          ENDIF
        ENDDO
!     Set all charges to zero
      ELSEIF (QSWITCHGUESS .or. (.not.QPREVCHR .or. SMBPIGU .eq. 2)) THEN 
        IF (.not. QSWITCHGUESS .and. SMBPIGU .eq. 1) THEN
          CALL WRNDIE(0,'<SPHERE_QMRF_IGU>', &
              'No previous QM charges found. Using zero charge guess!')
        ENDIF
        DO I = 1, NATOM
          CG(I) = ZERO
        ENDDO
      ELSE 
        CALL WRNDIE(-5,'<SPHERE_QMRF_IGU>', &
                   'Unsupported SMBPIGU option')
      ENDIF

!     --------------------------------------------------------------------------
!     Now calculate the initial QM RF potential based on the charges
!     defined above
!     --------------------------------------------------------------------------
      IF(QLBOX) THEN
        CALL SPRC_SMBP1(NATOM,X,Y,Z,CG,LNCEL,LTRAN,LDCEL, &
                        LXBCEN,LYBCEN,LZBCEN,QPRIN,PHI,.TRUE.,.TRUE.)
      ELSE 
        CALL SPRC_SMBP2(NATOM,X,Y,Z,CG,QPRIN,PHI,.TRUE.,.TRUE.)
      ENDIF

!     Debug printout of initial QM RF potential
      IF (DBGPRNT .ge. 1) THEN
        call chmalloc('pbeq.src','SPRC_QMRF_IGU','IRDUMMY',NATOM,crl=IRDUMMY)
        CALL RFORCE2(NQM,QMLST,X,Y,Z,CG, &
             NCLX,NCLY,NCLZ,DCEL,PHI, &
             TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN,ONE, &
             IRDUMMY,IRDUMMY,IRDUMMY,EDBG,QBSPL &
#if KEY_GCMC==1
            ,GCMCON & 
#endif
             )
        IF (PRNLEV .ge. 5) THEN
          WRITE(OUTU,101) 'E_RF (PHI_QM_RF_IGUESS) = ', EDBG
        ENDIF
        call chmdealloc('pbeq.src','SPRC_QMRF_IGU','IRDUMMY',NATOM,crl=IRDUMMY)
      ENDIF

      CALL DCOPY(NATOM,CGSAVE,1,CG,1)
#else /**/
      CALL WRNDIE(-5,'<SPHERE_QMRF_IGU>', &
                  'QM method != SCCDFTB || QCHEM || G09 NYI!')
#endif 


      call chmdealloc('pbeq.src','SPRC_QMRF_IGU','QMCHR',NQM,crl=QMCHR)
      call chmdealloc('pbeq.src','SPRC_QMRF_IGU','CGSAVE',NATOM,crl=CGSAVE)

100   FORMAT(3X,A)
101   FORMAT(3X,A,F20.6)
      RETURN
      END SUBROUTINE SPRC_QMRF_IGU

!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      SUBROUTINE RECT_SURFACE_CHRG(SURFCRD,SURFCHR,INICHRG)
!-----------------------------------------------------------------------
!     Provide positions of surface charges on cuboid surface 
!
      use chm_kinds
      use memory
      use number
      use stream
      use dimens_fcm
      use consta
      use comand
      use timerm
      use machutil,only:wrttim
      implicit none
!
      real(chm_real)  SURFCRD(*),SURFCHR(*)
      real(chm_real)  INICHRG
! local
      real(chm_real)  XS,YS,ZS,X0,Y0,Z0,XE,YE,ZE,LX,LY,LZ
      INTEGER QUOTX,QUOTY,QUOTZ 
      INTEGER I,J,K,N,NX,NY,NZ
      INTEGER NTMP
      LOGICAL QOUTPUT_SURF_CHRG_COORDS

      QOUTPUT_SURF_CHRG_COORDS = (PRNLEV .ge. 10)

!     Box boundaries 
      X0 = RBXMIN
      XE = RBXMAX 
      Y0 = RBYMIN
      YE = RBYMAX 
      Z0 = RBZMIN
      ZE = RBZMAX 
      LX = ABS(RBXMAX-RBXMIN)
      LY = ABS(RBYMAX-RBYMIN)
      LZ = ABS(RBZMAX-RBZMIN)

!     Warn user if he chooses different increments or if increments
!     don't fit into the grid dimensions
      IF ((ABS(RINCX - RINCY) .gt. TENM8) .or. &
          (ABS(RINCX - RINCZ) .gt. TENM8)) THEN 
        CALL WRNDIE(0,'<SMBP>', &
                    'Not all increments are the same! Are you sure?')
      ENDIF

      QUOTX = INT(LX/RINCX + 0.001)    ! Add small value for numerical stability
      QUOTY = INT(LY/RINCY + 0.001)    ! Only increments that fit into boxlength
      QUOTZ = INT(LZ/RINCZ + 0.001)    ! are considered reasonable anyway
 
      IF ((QUOTX*RINCX - LX) .gt. TENM8 .or. &
          (QUOTY*RINCY - LY) .gt. TENM8 .or. &
          (QUOTZ*RINCZ - LZ) .gt. TENM8) THEN
        CALL WRNDIE(-4,'<SMBP>', &
                    'Some increments do not fit into boxlength!')
      ENDIF

!     Calculate total number of surface charges from given increments
      NUMSURF = (QUOTX + 1) * (QUOTY + 1) * (QUOTZ + 1) &
              - (QUOTX - 1) * (QUOTY - 1) * (QUOTZ - 1)

!     Create grid of surface charges
      NX = QUOTX + 1 
      NY = QUOTY + 1
      NZ = QUOTZ + 1
      ZS = Z0
      N  = 1
      DO I = 1, NZ
        XS = X0
        IF (I .eq. 1 .or. I .eq. NZ) THEN
          DO J = 1, NX
            YS = Y0
            DO K = 1, NY
              SURFCRD(N)   = XS
              SURFCRD(N+1) = YS
              SURFCRD(N+2) = ZS
              YS = YS + RINCY
              N = N + 3
            ENDDO
            XS = XS + RINCX
          ENDDO
        ELSE
          DO J = 1, NX
            YS = Y0
            IF (J .eq. 1 .or. J .eq. NX) THEN
              DO K = 1, NY
                SURFCRD(N)   = XS
                SURFCRD(N+1) = YS
                SURFCRD(N+2) = ZS
                YS = YS + RINCY
                N = N + 3
              ENDDO
            ELSE
              SURFCRD(N)   = XS
              SURFCRD(N+1) = Y0
              SURFCRD(N+2) = ZS
              N = N + 3
              SURFCRD(N)   = XS
              SURFCRD(N+1) = YE
              SURFCRD(N+2) = ZS
              N = N + 3
            ENDIF
            XS = XS + RINCX
          ENDDO
        ENDIF  
        ZS = ZS + RINCZ
      ENDDO

!     Finally initialize charges-vector
      DO I = 1, NUMSURF
        SURFCHR(I) = INICHRG
      ENDDO

!     Printout of the surface charge coordinates
      IF (QOUTPUT_SURF_CHRG_COORDS) THEN
        NTMP = 1
        DO I = 1, NUMSURF
          XS = SURFCRD(NTMP)
          YS = SURFCRD(NTMP+1)
          ZS = SURFCRD(NTMP+2)
          WRITE(OUTU,'(3X,A4,F8.4,F8.4,F8.4)')'SURF', XS, YS, ZS
          NTMP = NTMP + 3
        ENDDO
      ENDIF

      IF (PRNLEV .ge. 5) THEN
        WRITE(OUTU,102)'Total number of surface charges = ', NUMSURF
      ENDIF

100   FORMAT(3X,A)
101   FORMAT(3X,A,F20.6)
102   FORMAT(3X,A,I6)
      RETURN
      END SUBROUTINE RECT_SURFACE_CHRG


!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      SUBROUTINE SPHE_SURFACE_CHRG(SURFCRD,SURFCHR,INICHRG)
!-----------------------------------------------------------------------
!     Provide positions of surface charges on spherical surface 
!
      use chm_kinds
      use memory
      use number
      use stream
      use dimens_fcm
      use consta
      use comand
      use timerm
      use machutil,only:wrttim
      implicit none
!
      real(chm_real)  SURFCRD(*),SURFCHR(*)
      real(chm_real)  INICHRG
! local
      real(chm_real),allocatable,dimension(:) :: ATHETA 
      real(chm_real),allocatable,dimension(:) :: ANPT
      real(chm_real)  PHI,PHI_OLD,PHI_INC,THETA,STHET,GOLDR,INC
      real(chm_real)  A,B,P,H,KR
      real(chm_real)  XS,YS,ZS,R
      INTEGER I,J,N,K,MI,NUM,NTOT
      INTEGER NTMP
      LOGICAL QOUTPUT_SURF_CHRG_COORDS
      INTEGER MAXK
      PARAMETER (MAXK=100)

      call chmalloc('pbeq.src','SPHE_SURFACE_CHRG','ATHETA',MAXK,crl=ATHETA)
      call chmalloc('pbeq.src','SPHE_SURFACE_CHRG','ANPT',MAXK,crl=ANPT)

      QOUTPUT_SURF_CHRG_COORDS = (PRNLEV .ge. 10)

!     Radius of sphere R == SRDIST
      R = SRDIST  

!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      IF (SPHEPTSALG .eq. 1) THEN
!       Uniform distribution on equally theta-distant sphere-slices
!       Note that actual NUMSURF will be determined by algorithm!
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

        IF (PRNLEV .ge. 5) THEN
          WRITE(OUTU,100) &
            'Creating uniform distribution on equally dist. circles'
        ENDIF

        N = NUMSURF
        IF (N .LT. 2) THEN
          CALL WRNDIE(-5,'<SPHE_SURFACE_CHRG>', 'NSPT must be at least 2')
        ENDIF


!       Sphere-slice with PHI == const. is divided in K segments
!       Calculate K from desired number of points first
        IF (N .EQ. 2) THEN
          K = 1
        ELSE
          K = NINT(SQRT(((N-2)*PI)/FOUR))
        ENDIF
        IF (K .GT. MAXK) &
          CALL WRNDIE(-5,'<SPHE_SURFACE_CHRG>','NSPT is too large')

!       Now calculate theta values for the equally theta-distant circles
!       parallel to z; two polar points are extra
        INC = PI/K;
        DO I = 1, K-1
          ATHETA(I) = -HALF*PI + I*INC
        ENDDO

!       Calulate number of points for each theta-circle
!       M_i = 2*K*cos(theta_i)
        NTOT = 2
        DO I = 1, K-1
          ANPT(I) = NINT(TWO*K*COS(ATHETA(I)))
          NTOT = NTOT + ANPT(I)
        ENDDO
        IF (PRNLEV .ge. 5) THEN
          WRITE(OUTU,102) 'Actual number of surface charges = ', NTOT
        ENDIF
        IF (NTOT .NE. N) THEN
          CALL WRNDIE(0,'<SPHE_SURFACE_CHRG>', &
                      'Calculated #(Surface Chrg) differs from input')
        ENDIF

        NUMSURF = NTOT

!       Calculate cartesian coordinates of the points
        SURFCRD(1) = ZERO
        SURFCRD(2) = ZERO
        SURFCRD(3) = R

        NUM = 4
        DO I = 1, K-1
          MI    = ANPT(I)
          THETA = ATHETA(I) + HALF*PI
          STHET = SIN(THETA)
          INC   = TWOPI/MI

          ZS    = R*COS(THETA)
          DO J = 1, MI
            PHI = J*INC

            XS  = R*STHET*COS(PHI)
            YS  = R*STHET*SIN(PHI)
            
            SURFCRD(NUM)   = XS
            SURFCRD(NUM+1) = YS
            SURFCRD(NUM+2) = ZS
            NUM = NUM + 3
          ENDDO
        ENDDO

        SURFCRD(NUM)   = ZERO
        SURFCRD(NUM+1) = ZERO
        SURFCRD(NUM+2) = -R
        NUM = NUM + 3

        IF (NTOT .NE. NUM/3) THEN
          CALL WRNDIE(-5,'<SPHE_SURFACE_CHRG>', &
                      'Bugs in SPHE_SURFACE_CHRG')
        ENDIF


!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      ELSEIF (SPHEPTSALG .eq. 2) THEN
!       Spiral point method
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        IF (PRNLEV .ge. 5) THEN
          WRITE(OUTU,100) 'Using Spiral-points method'
        ENDIF

        IF (NUMSURF .lt. 2 .OR. NUMSURF .eq. 3) THEN
          CALL WRNDIE(-5,'<SPHE_SURFACE_CHRG>' , & 
                      'NSPT must be either 2 or > 3')
        ENDIF

!       Initialize
        N = NUMSURF
        P = HALF
        A = ONE - TWO*P/(N-THREE)
        B = P*(N+ONE)/(N-THREE)
        GOLDR   = (ONE + SQRT(FIVE))/TWO
        PHI_OLD = ZERO
        PHI_INC = TWOPI/GOLDR 

!       Calculate cartesian coordinates of the points
        SURFCRD(1) = ZERO
        SURFCRD(2) = ZERO
        SURFCRD(3) = -R

        NUM = 4
        DO I = 2, N-1
          KR = A*I + B
          H = MINONE + TWO*(KR-1)/(N-1)
          THETA = ACOS(H)
          STHET = SIN(THETA)          

!         The REAL*8 modulo function is not problematic here
!         because for the considered # surface charges PHI/TWOPI
!         should not come close enough to 1 to cause numerical errors
!         (tested PHI/TWOPI for 1 -- 200 surface charges: 
!         deviation from 1 was always larger than 0.001)
          PHI = PHI_OLD + PHI_INC
          PHI = MOD(PHI,TWOPI)
 
          XS = R * STHET * COS(PHI)
          YS = R * STHET * SIN(PHI)
          ZS = R * H

          SURFCRD(NUM)   = XS
          SURFCRD(NUM+1) = YS
          SURFCRD(NUM+2) = ZS
          NUM = NUM + 3
          PHI_OLD = PHI
        ENDDO

        SURFCRD(NUM)   = ZERO
        SURFCRD(NUM+1) = ZERO
        SURFCRD(NUM+2) = R
        NUM = NUM + 3

        IF (NUMSURF .NE. NUM/3) THEN
          CALL WRNDIE(-5,'<SPHE_SURFACE_CHRG>', &
                      'Bugs in SPHE_SURFACE_CHRG')
        ENDIF

      ENDIF ! SPHEPTSALG
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

!     Finally initialize charges-vector
      DO I = 1, NUMSURF
        SURFCHR(I) = INICHRG
      ENDDO

!     Printout of the surface charge coordinates
      IF (QOUTPUT_SURF_CHRG_COORDS) THEN
        NTMP = 1
        DO I = 1, NUMSURF
          XS = SURFCRD(NTMP)
          YS = SURFCRD(NTMP+1)
          ZS = SURFCRD(NTMP+2)
          WRITE(OUTU,'(3X,A4,F8.4,F8.4,F8.4)')'SURF', XS, YS, ZS
          NTMP = NTMP + 3
        ENDDO
      ENDIF

      IF (PRNLEV .ge. 5) THEN
        WRITE(OUTU,102)'Total number of surface charges = ', NUMSURF
      ENDIF

      call chmdealloc('pbeq.src','SPHE_SURFACE_CHRG','ATHETA',MAXK,crl=ATHETA)
      call chmdealloc('pbeq.src','SPHE_SURFACE_CHRG','ANPT',MAXK,crl=ANPT)

100   FORMAT(3X,A)
101   FORMAT(3X,A,F20.6)
102   FORMAT(3X,A,I6)
      RETURN
      END SUBROUTINE SPHE_SURFACE_CHRG


!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      SUBROUTINE ADD_SMALL_LRG_GRID(PHIL,PHIS,NCEL3L,NCEL3S,DCELL,DCELS, &
                                    NXL,NYL,NZL,NXS,NYS,NZS,FACTORX)
!-----------------------------------------------------------------------
!     Add values from small rectangular grid to large one
!
      use chm_kinds
      use memory
      use number
      use stream
      use dimens_fcm
      use consta
      use comand
      implicit none
!
      real(chm_real)  DCELL,DCELS,FACTORX  
      real(chm_real4) PHIL(*),PHIS(*) 
      INTEGER NCEL3L,NCEL3S
      INTEGER NXL,NYL,NZL,NXS,NYS,NZS
! local
      real(chm_real)  FACTOR
      INTEGER NXDIFF,NYDIFF,NZDIFF,NYZL
      INTEGER COUNTL,COUNTS,IXL,IYL,IZL

      FACTOR = FACTORX

!     Simple if grid dimensions are actually the same
      IF (NXL .eq. NXS .and. NYL .eq. NYS .and. NZL .eq. NZS .and. &
          NCEL3L .eq. NCEL3S) THEN
        CALL ADDCTVR4(PHIL,PHIS,NCEL3L,FACTOR)
        RETURN
      ENDIF

!     Terminate if spacing is not the same
      IF (DCELL .ne. DCELS) THEN 
        CALL WRNDIE(-5,'<ADD_SMALL_LRG_GRID>', &
                    'Spacing of small and large grids must be equal')
      ENDIF

!     Determine #points on each +/- side not used by the small grid,
!     for each x, y, z direction (always an even number)
      NXDIFF = HALF*(NXL - NXS)
      NYDIFF = HALF*(NYL - NYS)
      NZDIFF = HALF*(NZL - NZS)
      NYZL   = NXL*NYL

!     Loop over grid points of large grid
      COUNTL = 1
      COUNTS = 1
      DO IXL=1,NXL      
        IF (IXL .le. NXDIFF .or. IXL .gt. (NXDIFF + NXS)) THEN
          COUNTL = COUNTL + NYZL
          CYCLE
        ENDIF

        DO IYL=1,NYL
          IF (IYL .le. NYDIFF .or. IYL .gt. (NYDIFF + NYS)) THEN
            COUNTL = COUNTL + NZL
            CYCLE
          ENDIF

          DO IZL=1,NZL
            IF (IZL .le. NZDIFF .or. IZL .gt. (NZDIFF + NZS)) THEN
              COUNTL = COUNTL + 1
              CYCLE
            ENDIF
           
            PHIL(COUNTL) = PHIL(COUNTL) + FACTOR*PHIS(COUNTS)
            COUNTL = COUNTL + 1
            COUNTS = COUNTS + 1

          ENDDO
        ENDDO
      ENDDO

      RETURN
      END SUBROUTINE ADD_SMALL_LRG_GRID

!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
#endif /*  SMBP*/
!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

#endif /*  (pbeq_main)*/

end module pbeq

