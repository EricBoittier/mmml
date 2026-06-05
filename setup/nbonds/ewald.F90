module ewald

  use chm_kinds
  use dimens_fcm
  use ewald_1m
  use memory
  use number

  implicit none

  !   TOTK         - Total number of K vectors
  !   PKXV         (in ewald_1m.src)- pointer to integer array KXV(MAXKV)
  !   PKYV         (in ewald_1m.src)- pointer to integer array KYV(MAXKV)
  !   PKZV         (in ewald_1m.src)- pointer to integer array KZV(MAXKV)
  !   PKVEC        (in ewald_1m.src)- pointer to KVEC vectors for the particular geometry
  !   MAXKV        (in ewald_1m.src) - Allocated dimension of KVEC, KXV, KYV, KZV
  !   OTOTK        - previous TOTK value
  !   OKMAX        - previous KMAX value
  !   OKSQMAX      - previous KMAXSQ value
  !   OKMAXX       - previous KMAXX value
  !   OKMAXY       - previous KMAXY value
  !   OKMAXZ       - previous KMAXZ value
  !   ERFMOD       - Mode for calculating erfc:
  !      -1 - use lookup table (fast but uses memory) cubic spline
  !       0 - use lookup table linear interpolation (fastest)
  !       1 - use Abramowitz and Stegun (commonly used)
  !       2 - Exact iterative method (slow)
  !       3 - Low precision exact iterative
  !       4 - use Chebyshev (better than A&S but a bit slower)
  !   OLEWLD       - previous x direction box length
  !   OMEWLD       - previous y direction box length
  !   ONEWLD       - previous z direction box length
  !   OKAPPA       - old kappa value
  !   EWXMAX       - Maximum distance for erfc lookup table
  !   QSETUPK     (in ewald_1m.src) - Flag indicating that the k-space needs to be set up
  !   EWVIRIAL(9) (in ewald_1m.src) - reciprocal space contribution to pressure tensor
  !
  INTEGER,save :: KMAX,KSQMAX,KMAXX,KMAXY,KMAXZ
  INTEGER,save :: TOTK
  integer,save :: OTOTK,OKMAX,OKSQMAX,OKMAXX,OKMAXY,OKMAXZ
  real(chm_real),save :: OLEWLD,OMEWLD,ONEWLD,OKAPPA,EWXMAX
#if KEY_FASTEW==1
  ! Cache storage for FAST EWald version of REWALD; Scott Brozell TSRI
  INTEGER,PARAMETER :: INTCRECSZ=1     ! record size of the integer cache
  INTEGER,PARAMETER :: REALCRECSZ=5    ! record size of the real(chm_real) cache
#endif
  !

  !----------------------------------------------------------------------------
contains

  !----------------------------------------------------------------------------
  ! Parse ewald options
  !----------------------------------------------------------------------------
  subroutine parse_ewald(comlyn, comlen)
    use psf,only:natom
    use image
    use string,only:gtrmi,gtrmf,indxa
    use stream
#if KEY_PHMD==1
    use phmd, only: qphmd
    use gbsw, only: qgbsw
    use gbmv, only: qgbmv
#endif /* phmd */
    use pme_module,only: qpme,qfinit, &
         reqcor,rewcut,pmesh_setup,pmesh_clear
    use pmeutil,only: nfft1,nfft2,nfft3,forder
    use erfcd_mod,only:ewrdel,deallocate_ewldt,allocate_ewldt,ferfct
    use inbnd,only:ctofnb
#if KEY_DOMDEC==1
    use domdec_common,only:q_split
    use domdec_dr_common,only:nrecip
    use colfft_util,only:ny_box,nz_box
#else /* domdec */
#if KEY_PARALLEL==1 && KEY_COLFFT==1
    use colfft_util,only:ny_box,nz_box
#endif
#endif /* domdec */
#if KEY_PARALLEL==1
    use parallel,only:numnod
#endif
#if KEY_LJPME==1
  ! DPME ACS/FCP 2/2016
  use pmeutil, only: qljpme, dkappa, dkappa2, dkappa4, dkappa6,&
                     dorder, dfftx, dffty, dfftz, dewvirial,&
                     dopcut, r6_inv_cut, dopcut_sp, r6_inv_cut_sp,&
                     qcpme, qmax, qmaxx, qmaxy, qmaxz, dmax, dmaxx, dmaxy, dmaxz
#endif

    implicit none
    ! Input
    character(len=*) comlyn
    integer comlen
    ! Variables
    INTEGER NFFT1N,NFFT2N,NFFT3N,FORDEN
    INTEGER DFFT1N,DFFT2N,DFFT3N,DORDEN
    real(chm_real) DKAPN ! LJPME temporaries
    logical ljpmeold
    LOGICAL QPMEOLD,CHANGE,QFINOLD, LJCHANGE
    real(chm_real)  OXMAX,OKAPV
    integer i

    OKAPV=KAPPA
    KAPPA=GTRMF(COMLYN,COMLEN,'KAPP',KAPPA)
    KMAX=GTRMI(COMLYN,COMLEN,'KMAX',KMAX)
    !                     added by Scott Feller 5/24/95, NIH
    KMAXX=GTRMI(COMLYN,COMLEN,'KMXX',KMAX)
    KMAXY=GTRMI(COMLYN,COMLEN,'KMXY',KMAX)
    KMAXZ=GTRMI(COMLYN,COMLEN,'KMXZ',KMAX)
    KSQMAX=GTRMI(COMLYN,COMLEN,'KSQM',KSQMAX)
    !
    !  parse the particle mesh ewald commands and do PME setup
    QPMEOLD=QPME
    IF(INDXA(COMLYN,COMLEN,'NOPM') > 0) QPME=.FALSE.
    IF(INDXA(COMLYN,COMLEN,'PMEW') > 0) QPME=.TRUE.
    IF(INDXA(COMLYN,COMLEN,'PME ') > 0) QPME=.TRUE.

#if KEY_COLFFT==1 && KEY_PHMD==1
    if (qpme .and. qphmd .and. (.not. (qgbmv .or. qgbsw))) &
        call wrndie(-5, '<parse_ewald>', &
            'PHMD with not compatible with COLFFT PME')
#endif /* colfft and phmd */

    CHANGE=(QPME.NEQV.QPMEOLD)
    IF(QPME) THEN
       IF(.NOT.QPMEOLD) THEN
          !              set default values when invoking PME
          NFFT1=2**KMAX
          NFFT2=2**KMAX
          NFFT3=2**KMAX
          FORDER=4
          QFINIT=.FALSE.
          REWCUT=999.0
          REQCOR=ZERO            ! ONE brb mod 24-JUL-2003
       ENDIF
       NFFT1N=GTRMI(COMLYN,COMLEN,'FFTX',NFFT1)
       CHANGE=CHANGE.OR.(NFFT1N /= NFFT1)
       NFFT2N=GTRMI(COMLYN,COMLEN,'FFTY',NFFT2)
       CHANGE=CHANGE.OR.(NFFT2N /= NFFT2)
       NFFT3N=GTRMI(COMLYN,COMLEN,'FFTZ',NFFT3)
       CHANGE=CHANGE.OR.(NFFT3N /= NFFT3)
       FORDEN=GTRMI(COMLYN,COMLEN,'ORDE',FORDER)
       CHANGE=CHANGE.OR.(FORDEN /= FORDER)
       QFINOLD=QFINIT
       IF(INDXA(COMLYN,COMLEN,'FINI') > 0) QFINIT=.TRUE.
       IF(INDXA(COMLYN,COMLEN,'NOFI') > 0) QFINIT=.FALSE.
       REWCUT=GTRMF(COMLYN,COMLEN,'CUTEW',REWCUT)
       REQCOR=GTRMF(COMLYN,COMLEN,'QCOR',REQCOR)
       IF((QFINOLD.NEQV.QFINIT) .AND. (PRNLEV > 3)) THEN
          IF(QFINIT.and.prnlev >= 2)  WRITE(OUTU,86) 'finite'
          IF(QFINOLD.and.prnlev >= 2) WRITE(OUTU,86) 'periodic'
86        FORMAT(' GTNBCT>  Switching PME method to ',A,' method.')
       ENDIF
#if KEY_COLFFT==1
#if KEY_PARALLEL==1
#if KEY_DOMDEC==1
       if (q_split) then
          if (ny_box*nz_box /= nrecip) change = .true.
       else
#endif
          if (ny_box*nz_box /= numnod) change = .true.
#if KEY_DOMDEC==1
       endif
#endif
#endif
#endif

    ENDIF

#if KEY_LJPME==1
    ! DPME ACS/FCP 2/2016
    LJPMEOLD=QLJPME
    IF(INDXA(COMLYN,COMLEN,'DPME') > 0) QLJPME=.TRUE.
    IF(INDXA(COMLYN,COMLEN,'LJPM') > 0) QLJPME=.TRUE.
    IF(INDXA(COMLYN,COMLEN,'NODP') > 0) QLJPME=.FALSE.
    LJCHANGE=(QLJPME .neqv. LJPMEOLD)
    IF (INDXA(COMLYN,COMLEN, 'CPME') > 0) QCPME=.TRUE.
    IF(.NOT.QPMEOLD) THEN
       !              set default values when invoking PME
       DFFTX=2**KMAX
       DFFTY=2**KMAX
       DFFTZ=2**KMAX
       DORDER=4
    ENDIF
    DKAPN=GTRMF(COMLYN,COMLEN,'DKAP',DKAPPA)
    LJCHANGE=LJCHANGE.OR.(DKAPN /= DKAPPA)
    DORDEN=GTRMI(COMLYN,COMLEN,'DORD',DORDER)
    LJCHANGE=LJCHANGE.OR.(DORDEN /= DORDER)
    DFFT1N=GTRMI(COMLYN,COMLEN,'DFTX',DFFTX)
    LJCHANGE=LJCHANGE.OR.(DFFT1N /= DFFTX)
    DFFT2N=GTRMI(COMLYN,COMLEN,'DFTY',DFFTY)
    LJCHANGE=LJCHANGE.OR.(DFFT2N /= DFFTY)
    DFFT3N=GTRMI(COMLYN,COMLEN,'DFTZ',DFFTZ)
    LJCHANGE=LJCHANGE.OR.(DFFT3N /= DFFTZ)
    IF(LJCHANGE.AND.QLJPME) THEN
       DORDER=DORDEN
       DKAPPA=DKAPN
       DKAPPA2=DKAPPA*DKAPPA
       DKAPPA4=DKAPPA2*DKAPPA2
       DKAPPA6=DKAPPA4*DKAPPA2
       DFFTX=DFFT1N
       DFFTY=DFFT2N
       DFFTZ=DFFT3N
       R6_INV_CUT = 1d0/CTOFNB**6
       DOPCUT = 1d0-(1d0 + DKAPPA**2*CTOFNB**2 + 0.5d0*DKAPPA**4*CTOFNB**4)*EXP(-DKAPPA**2*CTOFNB**2)
       DOPCUT_SP = REAL(DOPCUT, chm_real4)
       R6_INV_CUT_SP = REAL(R6_INV_CUT, chm_real4)
    ENDIF
    ! Compressed PME number of vectors in each dimension
    IF(.NOT.QPMEOLD) THEN
       QMAX=KMAX
       DMAX=KMAX
    ENDIF
    if(QCPME) then
        QMAX=GTRMI(COMLYN,COMLEN,'QMAX',QMAX)
        QMAXX=GTRMI(COMLYN,COMLEN,'QMXX',QMAX)
        QMAXY=GTRMI(COMLYN,COMLEN,'QMXY',QMAX)
        QMAXZ=GTRMI(COMLYN,COMLEN,'QMXZ',QMAX)
        DMAX=GTRMI(COMLYN,COMLEN,'DMAX',DMAX)
        DMAXX=GTRMI(COMLYN,COMLEN,'DMXX',DMAX)
        DMAXY=GTRMI(COMLYN,COMLEN,'DMXY',DMAX)
        DMAXZ=GTRMI(COMLYN,COMLEN,'DMXZ',DMAX)
    endif
#else
    ! LJPME not compiled in this case
    if((INDXA(COMLYN,COMLEN,'DPME') > 0) .or. (INDXA(COMLYN,COMLEN,'LJPM') > 0)) then
       CALL WRNDIE(-5,'<NBONDS>', &
            'LJPME requested but this version has not been compiled with the LJPME option')
    endif
    if(INDXA(COMLYN,COMLEN,'CPME') > 0) then
       CALL WRNDIE(-5,'<NBONDS>', &
            'cPME requested but this version has not been compiled with the LJPME option, which is required')
    endif
#endif /* end LJPME */



    !
    !        free space from last parse of PME options.
    IF(CHANGE.AND.QPMEOLD) CALL PMESH_CLEAR


    !
    IF(CHANGE.AND.QPME) THEN
       !           allocate new space for PME calculation
       NFFT1= NFFT1N
       NFFT2= NFFT2N
       NFFT3= NFFT3N
       FORDER=FORDEN
       !
       !  Check to see that the grid counts have prime factors of only 2,3,5!
       !
       DO WHILE(MOD(NFFT1N,2) == 0 .AND. NFFT1N /= 0)
          NFFT1N=NFFT1N/2
       ENDDO
       DO WHILE(MOD(NFFT1N,3) == 0 .AND. NFFT1N /= 0)
          NFFT1N=NFFT1N/3
       ENDDO
       DO WHILE(MOD(NFFT1N,5) == 0 .AND. NFFT1N /= 0)
          NFFT1N=NFFT1N/5
       ENDDO
       IF(NFFT1N /= 1) CALL WRNDIE(-3,'<GTNBCT>', &
            'The PME FFTX value has bad prime factors')
       !
       DO WHILE(MOD(NFFT2N,2) == 0 .AND. NFFT2N /= 0)
          NFFT2N=NFFT2N/2
       ENDDO
       DO WHILE(MOD(NFFT2N,3) == 0 .AND. NFFT2N /= 0)
          NFFT2N=NFFT2N/3
       ENDDO
       DO WHILE(MOD(NFFT2N,5) == 0 .AND. NFFT2N /= 0)
          NFFT2N=NFFT2N/5
       ENDDO
       IF(NFFT2N /= 1) CALL WRNDIE(-3,'<GTNBCT>', &
            'The PME FFTY value has bad prime factors')
       !
       DO WHILE(MOD(NFFT3N,2) == 0 .AND. NFFT3N /= 0)
          NFFT3N=NFFT3N/2
       ENDDO
       DO WHILE(MOD(NFFT3N,3) == 0 .AND. NFFT3N /= 0)
          NFFT3N=NFFT3N/3
       ENDDO
       DO WHILE(MOD(NFFT3N,5) == 0 .AND. NFFT3N /= 0)
          NFFT3N=NFFT3N/5
       ENDDO
       IF(NFFT3N /= 1) CALL WRNDIE(-3,'<GTNBCT>', &
            'The PME FFTZ value has bad prime factors')
       !
       CALL PMESH_SETUP(NATOM,XNSYMM)
    ENDIF
    !
    ! Parse the ERFMOD by number
    ERFMOD=GTRMI(COMLYN,COMLEN,'ERFM',ERFMOD)
    ! Parse the ERFMOD by name
    IF(INDXA(COMLYN,COMLEN,'ABRO') > 0) ERFMOD=1
    IF(INDXA(COMLYN,COMLEN,'EXAC') > 0) ERFMOD=2
    IF(INDXA(COMLYN,COMLEN,'LOWP') > 0) ERFMOD=3
    IF(INDXA(COMLYN,COMLEN,'CHEB') > 0) ERFMOD=4
    IF(INDXA(COMLYN,COMLEN,'INTE') > 0) ERFMOD=0
    IF(INDXA(COMLYN,COMLEN,'SPLI') > 0) ERFMOD=-1
    !
    IF(ERFMOD <= 0) THEN
       ! setup Ewald lookup table
       IF(EWNPTS > 0) THEN
          OXMAX=EWXMAX
          CHANGE=.FALSE.
       ELSE
          OXMAX=ZERO
          CHANGE=.TRUE.
       ENDIF
       IF(CHANGE.OR. OKAPV /= KAPPA) THEN
          EWXMAX=KAPPA*(CTOFNB+0.5)
          IF(EWXMAX > FIVE) EWXMAX=FIVE ! limit erfc table values to 1.0e-12
       ENDIF
       EWXMAX=GTRMF(COMLYN,COMLEN,'EWMA',EWXMAX)
       IF(EWXMAX /= OXMAX) CHANGE=.TRUE.
       !
       I=EWNPTS
       IF(EWNPTS == 0) EWNPTS=10000
       EWNPTS=GTRMI(COMLYN,COMLEN,'EWNP',EWNPTS)
       IF(EWNPTS < 10) EWNPTS=10
       IF(I /= EWNPTS) THEN
          CHANGE=.TRUE.
          IF(I > 0) call deallocate_ewldt()
          call allocate_ewldt()
       ENDIF
       IF(CHANGE) THEN
          !              (re)fill the erfc table
          EWRDEL=(EWNPTS-FIVE)/EWXMAX
          IF(PRNLEV > 3) WRITE(OUTU,88) EWNPTS,EWXMAX
88        FORMAT(' Fill Ewald table: Number of points=',I10, &
               ' EWXmax=',F12.6)
          CALL FERFCT()
       ENDIF
    ELSE
       IF(EWNPTS > 0) THEN
          call deallocate_ewldt()
          EWNPTS=0
       ENDIF
    ENDIF

    return
  end subroutine parse_ewald


#if KEY_FEWMFC==1 /*mfc_fast_ewald*/
#if KEY_FEWSB==0 /*fewsb*/
#if KEY_CHEQ==1 /*cheq*/
  !--------------------------------------------------------------------------------
  !             REWALD95   CHEQ version
  !--------------------------------------------------------------------------------
  subroutine rewald95_cheq(enb,eel,natomx,jnbl,inbl,lelecx,lvdwx, &
       iacnb,nitcc2,lowtp, &
       dx,dy,dz,x,y,z,cgx, &
       maxpairs,lused)

    use erfcd_mod
    use nb_module
#if KEY_CHEQ==1
    use cheq,only:qcg,qpartbin,   &
#endif
#if KEY_CHEQ==1
         DCH,SUMDCH,DDCH
#endif

    use stream
    use number
    use param
    use inbnd
    use consta
#if KEY_BLOCK==1
    use exfunc
    use block_fcm
#endif

    integer, intent(in) :: natomx,maxpairs
    real(chm_real) ::  enb,eel
    integer,intent(in)  :: jnbl(*),inbl(*)
    logical,intent(in)  :: lelecx,lvdwx
    integer,intent(in)  :: iacnb(natomx),nitcc2,lowtp(*)
    real(chm_real),intent(in),dimension(natomx) ::  x,y,z,cgx
    real(chm_real),intent(inout),dimension(natomx) :: dx,dy,dz
    logical,intent(out) :: lused

    REAL(KIND=CHM_REAL) DCHI
    REAL(KIND=CHM_REAL) E14FCQ

    !---namkh 01/20/04

    integer         :: ivect,kvect
    real(chm_real)  :: ca,cb,cc,ch,ene,enn
    real(chm_real)  :: tf,tx,ty,tz,dtx,dty,dtz
    real(chm_real)  :: s2,tr2,tr6,fsw,dfsw,dfrs
    real(chm_real)  :: cgf,cgt,crxi,cryi,crzi,crxj,cryj,crzj
    integer         :: itemp,i,j,jj,npr,iaci
    real(chm_real)  :: c2onnb,c2ofnb,rul3,rul12,rijl,riju
    real(chm_real)  :: e14fm1,e14f,rs,r1s,erfc2,erfcx,drfc
    real(chm_real)  :: xval
    real(chm_real)  :: rem,val0,val1,val2,d1,d2
    integer         :: ixval
    logical         :: elecfg

    !-- av_080628
    !-- Block code added.H Kamberaj, November 2007
    !--
#if KEY_BLOCK==1
    real(kind=chm_real)  :: COEF
    integer              :: IBL,JBL,KKB
#endif /*  close BLOCK*/

    !---    Translation Table of Some Variable Names
    !---    DTX     - tally in the pair loop of the x force displacements
    !---    DTY     - tally in the pair loop of the y force displacements
    !---    DTZ     - tally in the pair loop of the z force displacements
    !---    ENE     - update to the electrostatic energy
    !---    ENN     - update to the van der Waals energy
    !---    E14F    - modified electrostatic scale factor for 1-4 interactions
    !---    FSW     - van der Waals switch function
    !---    LVSHFT  - compute van der Waals energy via shifting?
    !---    NPR     - number of pairs
    !---    RS      - delta r
    !---    R1S     - one over delta r
    !---    S2      - delta r**2
    !---    TF      - force displacement factor
    !---    TR2     - one over delta r**2
    !---    TX      - delta x
    !---    TY      - delta y
    !---    TZ      - delta z

    !---Temps put in to take operations out of loops
    real(kind=chm_real) ekappa
    real(kind=chm_real)    xforce,yforce,zforce  ! x,y,z force displacements
    integer   cachednpr             ! the number of cached records
    !---Cache storage for FAST EWald version
    real(chm_real),allocatable,dimension(:,:) :: realcache
    real(chm_real),allocatable,dimension(:) :: rinv,rsq,rss,xvalv
    integer, allocatable,dimension(:,:)         :: intcache
    integer,parameter :: j_member=1,k_member=2,i_member=3
    integer,parameter :: s2_member=1,tf_member=2,tx_member=3, &
         ty_member=4,tz_member=5, e14f_member=6, e14fcq_member=7

#if KEY_LBMASSV==1
    !      include 'massv.include'
    integer(kind=4) :: len4
    integer(kind=8) :: len8
#elif KEY_INTELMKL==1
    integer(kind=4) :: len4
#endif

    call chmalloc("ewald.src","rewald95_cheq","realcache",7,maxpairs,crl=realcache)
    call chmalloc("ewald.src","rewald95_cheq","rinv" ,maxpairs,crl=rinv)
    call chmalloc("ewald.src","rewald95_cheq","rsq"  ,maxpairs,crl=rsq)
    call chmalloc("ewald.src","rewald95_cheq","rss"  ,maxpairs,crl=rss)
    call chmalloc("ewald.src","rewald95_cheq","xvalv",maxpairs,crl=xvalv)
    call chmalloc("ewald.src","rewald95_cheq","intcache",3,maxpairs,intg=intcache)


    !---------- Sanity check -------------------------------------
    if(.not. allocated(ccnba))then
       ! How we got here without vdw table filled, who knows?
       call wrndie(-4,"rewald95<rewald.src>", &
            "CCNBA not allocated")
    endif

    !
    !---Begin.
    !---    check to see if we should be here...
    LUSED = .FALSE.
    IF(ERFMOD >= 0) RETURN
    IF(LGROUP) RETURN
    IF(LVFSWT) RETURN
    LUSED = .TRUE.

    E14FM1 = E14FAC - ONE
    E14FCQ = 0.0

    C2OFNB = CTOFNB*CTOFNB
    IF (.NOT.(LSHFT.AND.LVSHFT)) THEN
       C2ONNB = CTONNB*CTONNB
       IF (CTOFNB > CTONNB) THEN
          RUL3 = ONE/(C2OFNB-C2ONNB)**3
          RUL12 = TWELVE*RUL3
          !---Precompute values needed for the VDW switch arithemetic transformations
          !---Note that when CTOFNB  <  CTONNB and when CTOFNB  ==  CTONNB
          !---the value of C2ONNB below insures a zero length switch region.
       ELSE
          C2ONNB = C2OFNB+RSMALL
          RUL3 = ONE/(C2OFNB-C2ONNB)**3
          RUL12 = ZERO
       ENDIF
    ENDIF
    !
    ENB = ZERO
    EEL = ZERO
    ELECFG = (LELECX.AND.(EPS /= 0.0))
    IF (ELECFG) THEN
       CGF = CCELEC/EPS
    ELSE
       CGF = ZERO
    ENDIF
    IF(.NOT.(LVDW.OR.ELECFG)) RETURN

    !======================================================================
    !
    !     DO VDW AND ELECTROSTATIC TERMS AS REQUESTED
    !
    !======================================================================
    ! Precompute values needed for Ewald erfc argument calculation
    EKAPPA = EWRDEL*KAPPA
    ITEMP = 0
    IF( .not. LVDWX) CALL WRNDIE(-4,'<REWALD95> fast-95 code', &
         'LVDWX must be .true.')
    IF(LVSHFT) CALL WRNDIE(-4,'<REWALD95> fast-95 code', &
         'VSHFT must be .false.')

    !---------------------------------------------------------------
    ! VDW switch.  Case 2 of 3.  (LVDWX .and. .not. LVSHFT)
    !---------------------------------------------------------------
    DO I = 1,NATOMX
#if KEY_IMCUBES==1
       IF(LBYCBIM) ITEMP = INBL(I+NATOMX)
#endif
       NPR = INBL(I) - ITEMP
       IF(NPR > 0) THEN
          if(npr > maxpairs)  &
               call wrndie(-4,'<REWALD95> fast-95 code', &
               ' number of pairs exceeds maxpairs')
          DCHI = ZERO
          IACI = IACNB(I)
          CRXI = X(I)
          CRYI = Y(I)
          CRZI = Z(I)
          DTX = ZERO
          DTY = ZERO
          DTZ = ZERO
          !---prologue loop: compute and conditionally cache delta r**2
          cachednpr = 0
          DO JJ = 1,NPR
             KVECT = JNBL(ITEMP+JJ)
             J = ABS(KVECT)
             !
             !---namkh 01/20/04
             !---This also possible to be screwed up, but hope not.
             !
             TX = CRXI-X(J)
             TY = CRYI-Y(J)
             TZ = CRZI-Z(J)
             !MFC is this necessary?  S2 = MAX(RSMALL,TX*TX+TY*TY+TZ*TZ)
             s2 = tx*tx+ty*ty+tz*tz
             if (s2 < c2ofnb) then
                cachednpr = cachednpr + 1
                intcache(j_member,cachednpr) = j
                intcache(k_member,cachednpr) = kvect

                realcache(s2_member,cachednpr) = s2
                realcache(tx_member,cachednpr) = tx
                realcache(ty_member,cachednpr) = ty
                realcache(tz_member,cachednpr) = tz
                rsq(cachednpr)=s2
             endif
          enddo
          do jj=1,cachednpr
             j=intcache(j_member,jj)
             ivect = lowtp(max(iacnb(j),iaci))+iacnb(j)+iaci
             e14f = zero
             e14fcq = zero
             if(intcache(k_member,jj) < 0) then
                e14f = e14fm1
                e14fcq = -1.0
                ivect = ivect+nitcc2
             endif
             intcache(i_member,jj) = ivect
             realcache(e14f_member,jj)=e14f
             realcache(e14fcq_member,jj)=e14fcq
          enddo

          !---- Try a vector operation to get the inverse sqrt -----------------
#if KEY_LBMASSV==1
          len4=cachednpr
          call vrsqrt(rinv,rsq,len4)
          !              call vrec(rss,rinv,len4)
#elif KEY_INTELMKL==1
          len4=cachednpr
          call vdinvsqrt(len4, rsq, rinv)
#else /**/
          rinv(1:cachednpr)=one/sqrt(rsq(1:cachednpr))
#endif
          rss(1:cachednpr) = rsq(1:cachednpr)*rinv(1:cachednpr)
          xvalv(1:cachednpr)= ekappa*rss(1:cachednpr)
          !---------------------------------------------------------------------

          !---electrostatic loop: compute and cache direct Ewald sum
          do jj = 1,cachednpr
             kvect = intcache(k_member,jj)
             j     = intcache(j_member,jj)
             r1s = rinv(jj)

             !---call ERFCD( RS, KAPPA, ERFCX, DRFC, ERFCT, -1 )
             !----1 means lookup table method using cubic spline interpolation

             !------ Inlined erfc calculation for speed (from ERFCD).
             xval = xvalv(jj)
             ixval = xval+half
             rem = xval-ixval
             ixval=ixval+2
             ixval = min(ixval,ewnpts-1)
             val0 = ewldt(ixval-1)
             val1 = ewldt(ixval)
             val2 = ewldt(ixval+1)
             d1 = (val0-val2)*half
             d2 = (val1+val1-val0-val2)*rem
             erfcx = val1-(d1+half*d2)*rem
             drfc = (d1+d2)*ekappa
             !------ end of erfc inlined code.

             e14f  =realcache(e14f_member,  jj)
             e14fcq=realcache(e14fcq_member,jj)
             ch = cgf
             if( (qpartbin(i) /= 0).and.(qpartbin(j).ne.0)) then
                ene = ch*(erfcx + e14fcq)*r1s
             else
                ene = ch*(erfcx + e14f)*r1s
             endif

             if ( (cgx(i) /= zero).and.(cgx(j).ne.zero) ) then
                dchi = dchi + ene*cgx(j)
                dch(j) = dch(j) + ene*cgx(i)
             else                         ! for tip4pfq type cases
                dchi = dchi + zero
             endif                        ! for tip4pfq type cases

             ene = ene*cgx(i)*cgx(j)
             ch = ch*cgx(i)*cgx(j)

             ! Block code: H Kamberaj November 2007
#if KEY_BLOCK==1 /*(block)*/
             IF (QBLOCK) THEN
                ibl = iblckp(i)
                jbl = iblckp(j)
                if (jbl  <  ibl) then
                   kkb = ibl
                   ibl = jbl
                   jbl = kkb
                endif
                kkb = ibl + jbl*(jbl-1)/2
                coef = blcoee(kkb)
                ENE = ENE * coef
                CH  = CH * coef
             ENDIF
#endif /*    (block)*/

             eel = eel+ene
             realcache(tf_member,jj) = -(ch*drfc + ene)
          enddo

          !--------------- VdW Loop --------------------------------------------
          !     VDW loop: compute van der Waals interaction and update forces
          do jj = 1,cachednpr
             s2 = realcache(s2_member,jj)
             kvect = intcache(k_member,jj)
             j = intcache(j_member,jj)
             ivect = intcache(i_member,jj)
             tr2=rinv(jj)*rinv(jj)
             tr6 = tr2*tr2*tr2
             ca = ccnba(ivect)*tr6

             !av_080628        enn = (ca-ccnbb(ivect))*tr6
             cb = ccnbb(ivect)
             ! Block code: H Kamberaj November 2007
#if KEY_BLOCK==1 /*(block)*/
             IF (QBLOCK) THEN
                ibl = iblckp(i)
                jbl = iblckp(j)
                if (jbl  <  ibl) then
                   kkb = ibl
                   ibl = jbl
                   jbl = kkb
                endif
                kkb = ibl + jbl*(jbl-1)/2
                coef = blcoev(kkb)
                CA = CA * coef
                CB = CB * coef
             ENDIF
#endif /*    (block)*/

             enn = (ca-cb)*tr6
             !av_080628
             tf = -six*(ca*tr6+enn)

             ! to produce pipelined code the s2 > c2onnb if was transformed into
             ! these min and max statements.

             rijl = min(zero,c2onnb-s2)
             riju = c2ofnb-max(s2,c2onnb)

             ! when s2 is greater than c2onnb then rijl = c2onnb - s2 and
             ! riju = c2ofnb - s2 and the switch function is calculated as in
             ! the untransformed source code.
             ! when s2 is not greater than c2onnb then rijl = zero
             ! and riju = c2ofnb - c2onnb;

             fsw = riju*riju*(riju-three*rijl)*rul3

             ! fsw will then equal one within the error of finite precision floating
             ! point arithmetic;

             tf = s2*enn*rijl*riju*rul12+tf*fsw
             enn = enn*fsw

             ! consequently, tf will then equal zero + tf * one and enn will then
             ! equal enn * one.  in other words, tf and enn will remain un-switched
             ! within the error of finite precision floating point arithmetic.

             enb = enb+enn
             tf = (realcache(tf_member,jj) + tf) * tr2
             !---- update forces
             xforce = realcache(tx_member,jj) * tf
             yforce = realcache(ty_member,jj) * tf
             zforce = realcache(tz_member,jj) * tf
             dtx = dtx+xforce
             dty = dty+yforce
             dtz = dtz+zforce
             dx(j) = dx(j)-xforce
             dy(j) = dy(j)-yforce
             dz(j) = dz(j)-zforce
          enddo

          !              restore the i-th component of the force
          dx(i) = dx(i)+dtx
          dy(i) = dy(i)+dty
          dz(i) = dz(i)+dtz
          dch(i) = dch(i) + dchi
       endif
       itemp = inbl(i)
    enddo

    return
  end subroutine rewald95_cheq
#endif /* (cheq)*/
  !--------------------------------------------------------------------------
  !             REWALD95
  !               No CHEQ
  !--------------------------------------------------------------------------
  subroutine rewald95(enb,eel,natomx,jnbl,inbl,lelecx,lvdwx, &
       iacnb,nitcc2,lowtp, &
       dx,dy,dz,x,y,z,cgx, &
       maxpairs,lused)

    use erfcd_mod
    use nb_module
#if KEY_CHEQ==1
    use cheq,only:qcg,qpartbin,   &
#endif
#if KEY_CHEQ==1
         DCH,SUMDCH,DDCH
#endif

    use stream
    use number
    use param
    use inbnd
    use consta
#if KEY_BLOCK==1
    use exfunc
    use block_fcm
#endif

    integer, intent(in) :: natomx,maxpairs
    real(chm_real) ::  enb,eel
    integer,intent(in)  :: jnbl(*),inbl(*)
    logical,intent(in)  :: lelecx,lvdwx
    integer,intent(in)  :: iacnb(natomx),nitcc2,lowtp(*)
    real(chm_real),intent(in),dimension(natomx) ::  x,y,z,cgx
    real(chm_real),intent(inout),dimension(natomx) :: dx,dy,dz
    logical,intent(out) :: lused

    integer         :: ivect,kvect
    real(chm_real)  :: ca,cb,cc,ch,ene,enn
    real(chm_real)  :: tf,tx,ty,tz,dtx,dty,dtz
    real(chm_real)  :: s2,tr2,tr6,fsw,dfsw,dfrs
    real(chm_real)  :: cgf,cgt,crxi,cryi,crzi,crxj,cryj,crzj
    integer         :: itemp,i,j,jj,npr,iaci
    real(chm_real)  :: c2onnb,c2ofnb,rul3,rul12,rijl,riju
    real(chm_real)  :: e14fm1,e14f,rs,r1s,erfc2,erfcx,drfc
    real(chm_real)  :: xval
    real(chm_real)  :: rem,val0,val1,val2,d1,d2
    integer         :: ixval
    logical         :: elecfg
#if KEY_CHEQ==1
    real(chm_real) dchi,e14fcq
#endif

    !-----------------------------------------------------------------------
    !---    Calculate nonbonded interaction energies and forces.
    !---    ENB     - calculated vdw interaction energy
    !---    EEL     - calculated electrostatic interaction energy
    !---    NATOMX  - number of atoms
    !---    JNBL    - nonbond atom pair list for index j
    !---    INBL    - nonbond atom pair list for index i
    !---    LELECX  - compute electrostatic energy?
    !---    LVDWX   - compute van der Waals energy?
    !---    CCNBA, CCNBB, CCNBC, CCNBD -
    !---    IACNB   -
    !---    NITCC2  -
    !---    LOWTP   -
    !---    DX, DY, DZ - atom forces as Cartesian coordinates
    !---    X, Y, Z - atom positions as Cartesian coordinates
    !---    CGX     - charges
    !---    ERFCT   - ERFc lookup table
    !---    QFLUc
    !---    FQCFOR  -
    !---    INTCACHE  - in the FAST EWald version cache storage for integers
    !---    REALCACHE - in the FAST EWald version cache storage for real(kind=chm_real)s
    !---    LUSED   - was anything calculated ?
    !---------------------------------------------------------------------
    !---    This is the fast scalar version of the nonbonded energy terms.
    !---    Electrostatic interactions herein are calculated via the
    !---    direct part of the Ewald summation.
    !---    Van der Waals interactions are switched only
    !---    electrostatic interactions are switched with erfc for ewald
    !----------------------------------------------------------------------
    !---    Translation Table of Some Variable Names
    !---    DTX     - tally in the pair loop of the x force displacements
    !---    DTY     - tally in the pair loop of the y force displacements
    !---    DTZ     - tally in the pair loop of the z force displacements
    !---    ENE     - update to the electrostatic energy
    !---    ENN     - update to the van der Waals energy
    !---    E14F    - modified electrostatic scale factor for 1-4 interactions
    !---    FSW     - van der Waals switch function
    !---    LVSHFT  - compute van der Waals energy via shifting?
    !---    NPR     - number of pairs
    !---    RS      - delta r
    !---    R1S     - one over delta r
    !---    S2      - delta r**2
    !---    TF      - force displacement factor
    !---    TR2     - one over delta r**2
    !---    TX      - delta x
    !---    TY      - delta y
    !---    TZ      - delta z

    !-- av_080628
    !-- Block code added.H Kamberaj, November 2007
    !--
#if KEY_BLOCK==1
    real(kind=chm_real)  :: COEF
    integer              :: IBL,JBL,KKB
#endif /*  close BLOCK*/

    !---Temps put in to take operations out of loops
    real(kind=chm_real) ekappa
    real(kind=chm_real)    xforce,yforce,zforce  ! x,y,z force displacements
    integer   cachednpr             ! the number of cached records
    !---Cache storage for FAST EWald version
    real(chm_real),allocatable,dimension(:,:) :: realcache
    real(chm_real),allocatable,dimension(:) :: rinv,rsq,rss,xvalv
    integer, allocatable,dimension(:,:)         :: intcache
    integer,parameter :: j_member=1,k_member=2,i_member=3
    integer,parameter :: s2_member=1,tf_member=2,tx_member=3, &
         ty_member=4,tz_member=5, e14f_member=6, e14fcq_member=7

#if KEY_LBMASSV==1
    !      include 'massv.include'
    integer(kind=4) :: len4
    integer(kind=8) :: len8
#elif KEY_INTELMKL==1
    integer(kind=4) :: len4
#endif


    !---------- Sanity check -------------------------------------
    if(.not. allocated(ccnba))then
       ! How we got here without vdw table filled, who knows?
       call wrndie(-4,"rewald95<rewald.src>", &
            "CCNBA not allocated")
    endif

    allocate(realcache(7,maxpairs))
    allocate(rinv(maxpairs),rsq(maxpairs),rss(maxpairs), &
         xvalv(maxpairs))
    allocate(intcache(3,maxpairs))

    !---Begin.
    !---    check to see if we should be here...
    LUSED = .FALSE.
    IF(ERFMOD >= 0) RETURN
    IF(LGROUP) RETURN
    IF(LVFSWT) RETURN
    LUSED = .TRUE.

    E14FM1 = E14FAC - ONE

    C2OFNB = CTOFNB*CTOFNB
    IF (.NOT.(LSHFT.AND.LVSHFT)) THEN
       C2ONNB = CTONNB*CTONNB
       IF (CTOFNB > CTONNB) THEN
          RUL3 = ONE/(C2OFNB-C2ONNB)**3
          RUL12 = TWELVE*RUL3
          !---Precompute values needed for the VDW switch arithemetic transformations
          !---Note that when CTOFNB  <  CTONNB and when CTOFNB  ==  CTONNB
          !---the value of C2ONNB below insures a zero length switch region.
       else
          c2onnb = c2ofnb+rsmall
          rul3 = one/(c2ofnb-c2onnb)**3
          rul12 = zero
       endif
    endif

    enb = zero
    eel = zero
    elecfg = (lelecx.and.(eps /= 0.0))
    if (elecfg) then
       cgf = ccelec/eps
    else
       cgf = zero
    endif
    if(.not.(lvdw.or.elecfg)) return

    !======================================================================
    !
    !     DO VDW AND ELECTROSTATIC TERMS AS REQUESTED
    !
    !======================================================================
    ! Precompute values needed for Ewald erfc argument calculation
    ekappa = ewrdel*kappa
    itemp = 0
    if( .not. lvdwx) call wrndie(-4,'<REWALD95> fast-95 code', &
         'LVDWX must be .true.')
    if(lvshft) call wrndie(-4,'<REWALD95> fast-95 code', &
         'VSHFT must be .false.')

    !---------------------------------------------------------------
    ! VDW switch.  Case 2 of 3.  (LVDWX .and. .not. LVSHFT)
    !---------------------------------------------------------------
    do i = 1,natomx
#if KEY_IMCUBES==1
       if(lbycbim) itemp = inbl(i+natomx)
#endif
       npr = inbl(i) - itemp
       if(npr > 0) then
          if(npr > maxpairs)  &
               call wrndie(-4,'<rewald95> fast-95 code', &
               ' number of pairs exceeds maxpairs')
          cgt = cgx(i)*cgf
          iaci = iacnb(i)
          crxi = x(i)
          cryi = y(i)
          crzi = z(i)
          dtx = zero
          dty = zero
          dtz = zero
          !---prologue loop: compute and conditionally cache delta r**2
          cachednpr = 0
          do jj = 1,npr
             kvect = jnbl(itemp+jj)
             j = abs(kvect)
             tx = crxi-x(j)
             ty = cryi-y(j)
             tz = crzi-z(j)
             s2 = tx*tx+ty*ty+tz*tz
             if (s2 < c2ofnb) then
                cachednpr = cachednpr + 1
                intcache(j_member,cachednpr) = j
                intcache(k_member,cachednpr) = kvect

                realcache(s2_member,cachednpr) = s2
                realcache(tx_member,cachednpr) = tx
                realcache(ty_member,cachednpr) = ty
                realcache(tz_member,cachednpr) = tz
                rsq(cachednpr)=s2
             endif
          enddo
          do jj=1,cachednpr
             j=intcache(j_member,jj)
             ivect = lowtp(max(iacnb(j),iaci))+iacnb(j)+iaci
             e14f = zero
             if(intcache(k_member,jj) < 0) then
                e14f = e14fm1
                ivect = ivect+nitcc2
             endif
             intcache(i_member,jj) = ivect
             realcache(e14f_member,jj)=e14f
          enddo

          !---- Try a vector operation to get the inverse sqrt -----------------
#if KEY_LBMASSV==1
          len4=cachednpr
          call vrsqrt(rinv,rsq,len4)
          !              call vrec(rss,rinv,len4)
#elif KEY_INTELMKL==1
          len4=cachednpr
          call vdinvsqrt(len4, rsq, rinv)
#else /**/
          rinv(1:cachednpr)=one/sqrt(rsq(1:cachednpr))
#endif
          rss(1:cachednpr) = rsq(1:cachednpr)*rinv(1:cachednpr)
          xvalv(1:cachednpr)= ekappa*rss(1:cachednpr)
          !---------------------------------------------------------------------

          !---electrostatic loop: compute and cache direct Ewald sum
#if KEY_CHEQ==1 /*cheq*/
          testcheq: if(qcg) then
             do jj = 1,cachednpr
                kvect = intcache(k_member,jj)
                j     = intcache(j_member,jj)
                r1s = rinv(jj)

                !---call ERFCD( RS, KAPPA, ERFCX, DRFC, ERFCT, -1 )
                !----1 means lookup table method using cubic spline interpolation
                !------ Inlined erfc calculation for speed (from ERFCD).
                xval = xvalv(jj)
                ixval = xval+half
                rem = xval-ixval
                ixval=ixval+2
                ixval = min(ixval,ewnpts-1)
                val0 = ewldt(ixval-1)
                val1 = ewldt(ixval)
                val2 = ewldt(ixval+1)
                d1 = (val0-val2)*half
                d2 = (val1+val1-val0-val2)*rem
                erfcx = val1-(d1+half*d2)*rem
                drfc = (d1+d2)*ekappa
                !------ end of erfc inlined code.

                e14f  =realcache(e14f_member,  jj)
                e14fcq=realcache(e14fcq_member,jj)
                ch = cgf
                if( (qpartbin(i) /= 0).and.(qpartbin(j).ne.0)) then
                   ene = ch*(erfcx + e14fcq)*r1s
                else
                   ene = ch*(erfcx + e14f)*r1s
                endif

                if ( (cgx(i) /= zero).and.(cgx(j) /= zero) ) then
                   dchi = dchi + ene*cgx(j)
                   dch(j) = dch(j) + ene*cgx(i)
                else        ! for tip4pfq type cases
                   dchi = dchi + zero
                endif       ! for tip4pfq type cases

                ene = ene*cgx(i)*cgx(j)
                ch = ch*cgx(i)*cgx(j)
                eel = eel+ene
                realcache(tf_member,jj) = -(ch*drfc + ene)
             enddo
          else testcheq
#endif /* (cheq)*/
             !---electrostatic loop: compute and cache direct Ewald sum
             do jj = 1,cachednpr
                kvect = intcache(k_member,jj)
                j     = intcache(j_member,jj)
                r1s = rinv(jj)
                !---call ERFCD( RS, KAPPA, ERFCX, DRFC, ERFCT, -1 )
                !----1 means lookup table method using cubic spline interpolation
                !------ Inlined erfc calculation for speed (from ERFCD).
                xval = xvalv(jj)
                ixval = xval+half
                rem = xval-ixval
                ixval=ixval+2
                ixval = min(ixval,ewnpts-1)
                val0 = ewldt(ixval-1)
                val1 = ewldt(ixval)
                val2 = ewldt(ixval+1)
                d1 = (val0-val2)*half
                d2 = (val1+val1-val0-val2)*rem
                erfcx = val1-(d1+half*d2)*rem
                drfc = (d1+d2)*ekappa
                !------ end of erfc inlined code.
                e14f  =realcache(e14f_member,  jj)
                ch = cgt*cgx(j)
                ene = ch*(erfcx + e14f)*r1s

                ! Block code: H Kamberaj November 2007
#if KEY_BLOCK==1 /*(block)*/
                IF (QBLOCK) THEN
                   ibl = iblckp(i)
                   jbl = iblckp(j)
                   if (jbl  <  ibl) then
                      kkb = ibl
                      ibl = jbl
                      jbl = kkb
                   endif
                   kkb = ibl + jbl*(jbl-1)/2
                   coef = blcoee(kkb)
                   ENE = ENE * coef
                   CH  = CH * coef
                ENDIF
#endif /*    (block)*/

                eel = eel+ene
                realcache(tf_member,jj) = -(ch*drfc + ene)
             enddo
#if KEY_CHEQ==1
          endif testcheq
#endif
          !--------------- VdW Loop --------------------------------------------
          !     VDW loop: compute van der Waals interaction and update forces
          do jj = 1,cachednpr
             s2 = realcache(s2_member,jj)
             kvect = intcache(k_member,jj)
             j = intcache(j_member,jj)
             ivect = intcache(i_member,jj)
             !                  tr2 = one/s2
             tr2=rinv(jj)*rinv(jj)
             tr6 = tr2*tr2*tr2
             ca = ccnba(ivect)*tr6

             !av_080628        enn = (ca-ccnbb(ivect))*tr6
             cb = ccnbb(ivect)
             ! Block code: H Kamberaj November 2007
#if KEY_BLOCK==1 /*(block)*/
             IF (QBLOCK) THEN
                ibl = iblckp(i)
                jbl = iblckp(j)
                if (jbl  <  ibl) then
                   kkb = ibl
                   ibl = jbl
                   jbl = kkb
                endif
                kkb = ibl + jbl*(jbl-1)/2
                coef = blcoev(kkb)
                CA = CA * coef
                CB = CB * coef
             ENDIF
#endif /*    (block)*/

             enn = (ca-cb)*tr6
             !av_080628

             tf = -six*(ca*tr6+enn)

             ! to produce pipelined code the s2 > c2onnb if was transformed into
             ! these min and max statements.

             rijl = min(zero,c2onnb-s2)
             riju = c2ofnb-max(s2,c2onnb)

             ! when s2 is greater than c2onnb then rijl = c2onnb - s2 and
             ! riju = c2ofnb - s2 and the switch function is calculated as in
             ! the untransformed source code.
             ! when s2 is not greater than c2onnb then rijl = zero
             ! and riju = c2ofnb - c2onnb;

             fsw = riju*riju*(riju-three*rijl)*rul3

             ! fsw will then equal one within the error of finite precision floating
             ! point arithmetic;

             tf = s2*enn*rijl*riju*rul12+tf*fsw
             enn = enn*fsw

             ! consequently, tf will then equal zero + tf * one and enn will then
             ! equal enn * one.  in other words, tf and enn will remain un-switched
             ! within the error of finite precision floating point arithmetic.

             enb = enb+enn
             tf = (realcache(tf_member,jj) + tf) * tr2
             !--- update forces
             xforce = realcache(tx_member,jj) * tf
             yforce = realcache(ty_member,jj) * tf
             zforce = realcache(tz_member,jj) * tf
             dtx = dtx+xforce
             dty = dty+yforce
             dtz = dtz+zforce
             dx(j) = dx(j)-xforce
             dy(j) = dy(j)-yforce
             dz(j) = dz(j)-zforce
          enddo

          !---- restore the i-th component of the force
          dx(i) = dx(i)+dtx
          dy(i) = dy(i)+dty
          dz(i) = dz(i)+dtz
#if KEY_CHEQ==1
          if (qcg) dch(i) = dch(i) + dchi
#endif
       endif
       itemp = inbl(i)
    enddo

    return
  end subroutine rewald95
  !   End of modified c33a2 code for speed testing......MFC
  !======================================================================================

#else /* (fewsb)*/
  !sb Stefan Boresch May 2007
  !
  !     Contains modified subroutine rewald95 for ##IMCUBES code
  !     compile with IMCUBES FEWMFC pref.dat keywords and *without* CHEQ
  !
  SUBROUTINE REWALD95_sb(ENB,EEL,NATOMX,JNBL,INBL,LELECX,LVDWX, &
       IACNB,NITCC2,LOWTP, &
       DX,DY,DZ,X,Y,Z,CGX, &
       maxpairs,LUSED)
    !
    !sb   Stefan Boresch, May 2007
    !     An attempt to optimize Mike Crowley's rewald95 routine
    !     Requires Intel ifort 9.1 and optionally the Intel MKL library
    !
    !     Compiler options were:
    !
    !     ifort -O3 -no-prec-div -tpp7 -132 -xW -w95 -cm -align all $(ENDIAN) $(I8DUM1)
    !
    !     In my tests I find the following speedups relative to
    !     Bernie Brook's EXPANDed (fast on) routines (testcase was the
    !     jac1000, 200 steps (instead of 1000) only
    !
    !     machine description                        speedup
    !     -----------------------------------------------------
    !     P4 (3GHz)                                   1.17    (with MKL)
    !     P4 (3.4GHz, EM64T, 64bit)                   1.06    (no MKL)
    !     Opteron (2.4GHz, 64bit)                     0.98    (no MKL)
    !     Core2 (2.4GHz, 64bit)                       1.18    (no MKL)
    !
    !     On the EM64T machine the MKL version of the code is slightly
    !     slower; the MKL version segfaulted for unknown reasons on the
    !     Opteron and Core2
    !
    !     For a significantly larger system with truncated octahedral b.c.,
    !     the best speedup seen is 1.08.
    !
    !     Some general comments:
    !
    !     1) I avoid the use of ALLOCATE (it does cost a few percent). In this
    !     quick and dirty implementation I use static arrays instead. In my
    !     opinion, the correct way of doing things would be calling GETMAXNPR
    !     after a list update only, and doing the allocation there,
    !     or the result of GETMAXNPR should be passed to rewald95 as an argument
    !     and the automatic arrays of FORTRAN95 should be used ...
    !
    !     2) My experience on Intel P4 indicates that it is often better to
    !     use several one dimensional arrays rather than two-dimensional
    !     arrays such as real_cache and int_cache. Rationale: While the 2d
    !     arrays have the best cache/memory locality, they are often
    !     accessed with a nonunit stride. And on Intel Netburst/Core2 this
    !     prevents (efficient) 'vectorization'. Thus, real_cache and int_cache
    !     are dropped and several aux. 1d arrays are used (see also Aart Bik,
    !     'The Software Vectorization Handbook', pp204)
    !
    !     3) My design goal was getting as many 'LOOP WAS VECTORIZED' diagnostics
    !     as possible. Most likely, not all of these really improve performance
    !     (see comments in the code) When it was clear that a loop could not
    !     be vectorized, then I tried unrolling it. Again, the performance benefit
    !     is not fully clear.
    !
    !sb
    !
    !-----------------------------------------------------------------------
    !---    Calculate nonbonded interaction energies and forces.
    !
    !---    ENB     - calculated vdw interaction energy
    !---    EEL     - calculated electrostatic interaction energy
    !---    NATOMX  - number of atoms
    !---    JNBL    - nonbond atom pair list for index j
    !---    INBL    - nonbond atom pair list for index i
    !---    LELECX  - compute electrostatic energy?
    !---    LVDWX   - compute van der Waals energy?
    !---    CCNBA, CCNBB, CCNBC, CCNBD -
    !---    IACNB   -
    !---    NITCC2  -
    !---    LOWTP   -
    !---    DX, DY, DZ - atom forces as Cartesian coordinates
    !---    X, Y, Z - atom positions as Cartesian coordinates
    !---    CGX     - charges
    !---    ERFCT   - ERFc lookup table
    !---    QFLUc   -
    !---    FQCFOR  -
    !---    INTCACHE  - in the FAST EWald version cache storage for integers
    !---    REALCACHE - in the FAST EWald version cache storage for real(kind=chm_real)s
    !---    LUSED   - was anything calculated ?
    !
    !---------------------------------------------------------------------
    !---    This is the fast scalar version of the nonbonded energy terms.
    !---    Electrostatic interactions herein are calculated via the
    !---    direct part of the Ewald summation.
    !---    Van der Waals interactions are switched only
    !---    electrostatic interactions are switched with erfc for ewald
    !
    !-----------------------------------------------------------------------

    use erfcd_mod
    use nb_module

    use stream
    use number
    use param
    use inbnd
    use consta
#if KEY_BLOCK==1
    use exfunc
    use block_fcm
#endif

    integer,intent(in) :: natomx, maxpairs
    integer,intent(in) :: jnbl(*),inbl(*)
    logical,intent(in) :: lelecx,lvdwx
    integer,intent(in) :: iacnb(*),nitcc2,lowtp(*)
    real(chm_real),intent(in),   dimension(natomx) ::  x,y,z,cgx
    real(chm_real),intent(inout),dimension(natomx) ::  dx,dy,dz

    real(chm_real),intent(out) ::  enb,eel
    logical,intent(out) :: lused

    !---    DTX     - tally in the pair loop of the x force displacements
    !---    DTY     - tally in the pair loop of the y force displacements
    !---    DTZ     - tally in the pair loop of the z force displacements
    !---    ENE     - update to the electrostatic energy
    !---    ENN     - update to the van der Waals energy
    !---    E14F    - modified electrostatic
    !---              scale factor for 1-4 interactions
    !---    FSW     - van der Waals switch function
    !---    LVSHFT  - compute van der Waals energy via shifting?
    !---    NPR     - number of pairs
    !---    RS      - delta r
    !---    R1S     - one over delta r
    !---    S2      - delta r**2
    !---    TF      - force displacement factor
    !---    TR2     - one over delta r**2
    !---    TX      - delta x
    !---    TY      - delta y
    !---    TZ      - delta z

    !---Temps put in to take operations out of loops
    real(chm_real) :: ekappa
    !---Declarations for loop fission and arithmetic transformations
    real(chm_real) :: xforce,yforce,zforce  ! x,y,z force displacements
    !---Cache storage for FAST EWald version
    !sb
    !     as discussed above, one-dimens. arrays are used and allocated
    !     statically
    !
    integer :: ivect,kvect,itemp,i,j,jj,npr,iaci,ixval
    logical :: elecfg
    integer :: cachednpr             ! the number of cached records
    integer,dimension(maxpairs) :: imember,jmember, &
         member14,auxiac
    integer :: iacj,ixval2,ixval3,ixval4,j1,j2,j3
    real(chm_real) :: ca,cb,cc,ch,ene,enn,tf,tx,ty,tz,dtx,dty,dtz, &
         s2,tr2,tr6,fsw,dfsw,dfrs
    real(chm_real)  :: cgf,cgt,crxi,cryi,crzi,crxj,cryj,crzj
    real(chm_real)  :: c2onnb,c2ofnb,rul3,rul12,rijl,riju,e14fm1,e14f, &
         rs,r1s,erfc2,erfcx,drfc,xval
    real(chm_real),dimension(maxpairs) :: rinv,rsq,rss,xvalv
    real(chm_real),dimension(maxpairs) :: delx,dely,delz
    real(chm_real),dimension(maxpairs) :: e14aux,forceaux,jcharge
    real(chm_real),dimension(maxpairs) :: lj1aux,lj2aux,mask2
    real(chm_real),dimension(maxpairs) ::  REM,VAL0,VAL1,VAL2
    real(chm_real),dimension(maxpairs) :: mask
    real(chm_real) :: aux,xval2,xval3,xval4,d1,d2,erfc

    !-- av_080628
    !-- Block code added.H Kamberaj, November 2007
    !--
#if KEY_BLOCK==1
    real(kind=chm_real)  :: COEF
    integer              :: IBL,JBL,KKB
#endif /*  close BLOCK*/

    !---------- Sanity check -------------------------------------
    if(.not. allocated(ccnba))then
       ! How we got here without vdw table filled, who knows?
       call wrndie(-4,"rewald95<rewald.src>", &
            "CCNBA not allocated")
    endif
    !---Begin.
    !---    check to see if we should be here...
    lused = .false.
    if(erfmod >= 0) return
    if(lgroup) return
    if(lvfswt) return
    lused = .true.
    !
    e14fm1 = e14fac - one


    !
    c2ofnb = ctofnb*ctofnb
    if (.not.(lshft.and.lvshft)) then
       c2onnb = ctonnb*ctonnb
       if (ctofnb > ctonnb) then
          rul3 = one/(c2ofnb-c2onnb)**3
          rul12 = twelve*rul3
          !---Precompute values needed for the VDW switch arithemetic
          !---transformations
          !---Note that when CTOFNB  <  CTONNB and when CTOFNB  ==  CTONNB
          !---the value of C2ONNB below insures a zero length switch region.
       else
          c2onnb = c2ofnb+rsmall
          rul3 = one/(c2ofnb-c2onnb)**3
          rul12 = zero
       endif
    endif

    enb = zero
    eel = zero
    elecfg = (lelecx .and. (eps /= zero))
    if (elecfg) then
       cgf = ccelec/eps
    else
       cgf = zero
    endif
    if( .not. (lvdw .or. elecfg) ) return

    !======================================================================
    !
    !     DO VDW AND ELECTROSTATIC TERMS AS REQUESTED
    !
    !======================================================================
    ! Precompute values needed for Ewald erfc argument calculation
    ekappa = ewrdel*kappa
    itemp = 0
    IF( .not. LVDWX) CALL WRNDIE(-4,'<REWALD95> fast-95 code', &
         'LVDWX must be .true.')
    IF(LVSHFT) CALL WRNDIE(-4,'<REWALD95> fast-95 code', &
         'LVDWX must be .false.')

    !---------------------------------------------------------------
    ! VDW switch.  Case 2 of 3.  (LVDWX .and. .not. LVSHFT)
    !---------------------------------xs------------------------------
    natom_loop: do i = 1,natomx
#if KEY_IMCUBES==1
       if(lbycbim) itemp = inbl(i+natomx)
#endif
       npr = inbl(i) - itemp
       do_pairs: if(npr > 0) then
          cgt = cgx(i)*cgf

          iaci = iacnb(i)
          crxi = x(i)
          cryi = y(i)
          crzi = z(i)
          dtx = zero
          dty = zero
          dtz = zero

          !--- prologue loop: compute and conditionally cache delta r**2
          !    unvectorizable loop

          !sb   I would still think that this could be accelerated, but naive
          !     unrolling attempts definitively *slow* down the code

          cachednpr = 0
          do jj = 1,npr
             kvect = jnbl(itemp+jj)
             j = abs(kvect)
             tx = crxi-x(j)
             ty = cryi-y(j)
             tz = crzi-z(j)

             s2 = tx*tx+ty*ty+tz*tz

             if (s2 < c2ofnb) then
                cachednpr = cachednpr + 1

                jmember(cachednpr) = kvect ! note: this needs to be
                ! postprocessed
                auxiac(cachednpr)=iacnb(j)
                delx(cachednpr)=tx
                dely(cachednpr)=ty
                delz(cachednpr)=tz
                rsq(cachednpr)=s2

             endif
          enddo

          !     sort out 1-4 pairs
          !     whether the effort to get the next few lines to vectorize is
          !     worth is is not fully clear
          !
          !     the ugly contortions that follow are a consequence of the inability
          !     of the compiler (the architecture) to 'vectorize' a loop that contains
          !     32bit and 64bit data types (e.g., integer*4 and real(chm_real))
          !
          !DEC$ vector always
          do jj=1,cachednpr
             member14(jj)=0
             if (jmember(jj) < 0) member14(jj)=1
          enddo
          !DEC$ vector always
          do jj=1,cachednpr
             mask2(jj)=zero
             if (jmember(jj) < 0) mask2(jj)=one
          enddo
          !sb   vectorization stalls on direct conversion integer*4 -> real(chm_real);
          !     thus, this round about thing
          !            mask2(1:cachednpr)=mask(1:cachednpr)

          jmember(1:cachednpr)=abs(jmember(1:cachednpr))

          !     preparations for elec
          do jj=1,cachednpr
             e14aux(jj)=e14fm1*mask2(jj)
          enddo
          !DEC$ vector always
          do jj=1,cachednpr
             jcharge(jj)=cgx(jmember(jj))
          enddo

          !     preparations for vdw

          !     While this version gets rid of the j-dependence of Mike's code,
          !     it still cannot be vectorized; hence one might consider keeping
          !     Mike's code and save one aux. array ...
          do jj=1,cachednpr
             iacj=auxiac(jj)
             ivect = lowtp(max(iacj,iaci))+iacj+iaci
             imember(jj) = ivect
          enddo
          !DEC$ vector always
          do jj=1,cachednpr
             if (member14(jj) > 0) imember(jj)=imember(jj)+nitcc2
          enddo

          !DEC$ vector always
          do jj=1,cachednpr
             lj1aux(jj)=ccnba(imember(jj))
             lj2aux(jj)=ccnbb(imember(jj))
          enddo

          !---- Try a vector operation to get the inverse sqrt -----------------

          !sb   this works slightly better than the three f95 array statements
          !     in Mike's code, probably because we only have one loop
          !     On 32bit P4, however, calling MKL is even better ...
          !            do jj=1,cachednpr
          !               aux=sqrt(rsq(jj))
          !               xvalv(jj)=ekappa*aux
          !               rss(jj)=aux
          !               rinv(jj)=one/aux
          !            enddo

          !     So, ... use MKL instead. HOWEVER, On 64bit platforms tested, MKL
          !     did either not work or was SLOWER than the above loop!!

          call vdinvsqrt(cachednpr,rsq,rinv)
          call vdinv(cachednpr,rinv,rss)
          xvalv(1:cachednpr)= ekappa*rss(1:cachednpr)

          !---------------------------------------------------------------------

          !           ---electrostatic loop: compute and cache direct Ewald sum

          !sb
          !     This was actually the most interesting loop to vectorize
          !     The loop from Mike's code did not vectorize because in the
          !     table lookup there is (2) the usual integer / real(chm_real) mixup ...
          !     and (2) the lookup itself can't be vectorized (no known stride
          !     between elements. Thus, the loop was split into two parts:

          !     1) fill aux. arrays which can be used to compute erfc and its
          !     derivative with elementary operations. This loop can't be
          !     vectorized, but we attempt manual unrolling
          !     [performance benefit of unrolling is not clear, but splitting the
          !      Ewald calc. in the two parts, one of which is vectorizable, is
          !      definitely beneficial]

          elec_loop1: do jj = 1,(cachednpr/4)*4,4
             xval  = xvalv(jj)
             xval2 = xvalv(jj+1)
             xval3 = xvalv(jj+2)
             xval4 = xvalv(jj+3)
             ixval  = xval +half
             ixval2 = xval2+half
             ixval3 = xval3+half
             ixval4 = xval4+half
             rem(jj)   = xval -ixval
             rem(jj+1) = xval2-ixval2
             rem(jj+2) = xval3-ixval3
             rem(jj+3) = xval4-ixval4
             ixval =ixval +2
             ixval2=ixval2+2
             ixval3=ixval3+2
             ixval4=ixval4+2
             ixval  = min(ixval ,ewnpts-1)
             ixval2 = min(ixval2,ewnpts-1)
             ixval3 = min(ixval3,ewnpts-1)
             ixval4 = min(ixval4,ewnpts-1)
             val0(jj) = ewldt(ixval-1)
             val1(jj) = ewldt(ixval)
             val2(jj) = ewldt(ixval+1)
             val0(jj+1) = ewldt(ixval2-1)
             val1(jj+1) = ewldt(ixval2)
             val2(jj+1) = ewldt(ixval2+1)
             val0(jj+2) = ewldt(ixval3-1)
             val1(jj+2) = ewldt(ixval3)
             val2(jj+2) = ewldt(ixval3+1)
             val0(jj+3) = ewldt(ixval4-1)
             val1(jj+3) = ewldt(ixval4)
             val2(jj+3) = ewldt(ixval4+1)
          enddo elec_loop1
          !sb   cleanup!!
          do jj = (cachednpr/4)*4+1,cachednpr
             xval = xvalv(jj)
             ixval = xval+half
             rem(jj) = xval-ixval
             ixval=ixval+2
             ixval = min(ixval,ewnpts-1)
             val0(jj) = ewldt(ixval-1)
             val1(jj) = ewldt(ixval)
             val2(jj) = ewldt(ixval+1)
          enddo

          !     2) calculate rest of erfc(d) and
          !     calculate real space ewald energy and force
          !     this loop vectorizes cleanly

          do jj = 1,cachednpr
             d1 = (val0(jj)-val2(jj))*half
             d2 = (val1(jj)+val1(jj)-val0(jj)-val2(jj))*rem(jj)
             erfc = val1(jj)-(d1+half*d2)*rem(jj)
             erfcx = (d1+d2)*ekappa
             !
             r1s = rinv(jj)
             e14f  = e14aux(jj)
             ch = cgt*jcharge(jj)
             ! Block code: H Kamberaj November 2007
#if KEY_BLOCK==1 /*(block)*/
             IF (QBLOCK) THEN
                ibl = iblckp(i)
                jbl = iblckp(j)
                if (jbl  <  ibl) then
                   kkb = ibl
                   ibl = jbl
                   jbl = kkb
                endif
                kkb = ibl + jbl*(jbl-1)/2
                coef = blcoee(kkb)
                ch  = ch * coef
             ENDIF
#endif /*    (block)*/
             ene = ch*(erfc + e14f)*r1s
             eel = eel+ene
             forceaux(jj) = -(ch*erfcx + ene)
          enddo



          !--------------- VdW Loop --------------------------------------------
          !     VDW loop: compute van der Waals interaction and update forces

          !sb
          !     provided aux. force arrays are used, this loop vectorizes
          !     cleanly. One could play with splitting it into shorter pieces,
          !     might be particularly interesting for Core2
          !

          vdw_loop1: do jj = 1,cachednpr
             s2 = rsq(jj)
             tr2=rinv(jj)*rinv(jj)
             tr6 = tr2*tr2*tr2
             ca = lj1aux(jj)*tr6
             enn = (ca-lj2aux(jj))*tr6

             ! Block code: H Kamberaj November 2007
#if KEY_BLOCK==1 /*(block)*/
             IF (QBLOCK) THEN
                ibl = iblckp(i)
                jbl = iblckp(j)
                if (jbl  <  ibl) then
                   kkb = ibl
                   ibl = jbl
                   jbl = kkb
                endif
                kkb = ibl + jbl*(jbl-1)/2
                coef = blcoev(kkb)
                ca  = ca * coef
                enn = enn * coef
             ENDIF
#endif /*    (block)*/

             tf = -six*(ca*tr6+enn)

             ! to produce pipelined code the s2 > c2onnb if was transformed into
             ! these min and max statements.

             rijl = min(zero,c2onnb-s2)
             riju = c2ofnb-max(s2,c2onnb)

             ! when s2 is greater than c2onnb then rijl = c2onnb - s2 and
             ! riju = c2ofnb - s2 and the switch function is calculated as in
             ! the untransformed source code.
             ! when s2 is not greater than c2onnb then rijl = zero
             ! and riju = c2ofnb - c2onnb;

             fsw = riju*riju*(riju-three*rijl)*rul3

             ! fsw will then equal one within the error of finite precision floating
             ! point arithmetic;

             tf = s2*enn*rijl*riju*rul12+tf*fsw
             enn = enn*fsw

             ! consequently, tf will then equal zero + tf * one and enn will then
             ! equal enn * one.  in other words, tf and enn will remain un-switched
             ! within the error of finite precision floating point arithmetic.

             enb = enb+enn
             tf = (forceaux(jj) + tf) * tr2

             !              ---                update forces
             !sb make this with auxiliary arrays -- reuse del[xyz]!
             !     the use of auxiliary arrays (and reusing an existing array)
             !     is actually important for performance
             !

             delx(jj) = delx(jj) * tf
             dely(jj) = dely(jj) * tf
             delz(jj) = delz(jj) * tf
             dtx = dtx+delx(jj)
             dty = dty+dely(jj)
             dtz = dtz+delz(jj)
          enddo vdw_loop1

          !     restore the j-th components of the force
          !     this loop is not vectorizable (no scatter), so I have unrolled it
          !     Performance gain is marginal
          !
          vdw_loop2: do jj=1,(cachednpr/4)*4,4
             j =jmember(jj)
             j1=jmember(jj+1)
             j2=jmember(jj+2)
             j3=jmember(jj+3)
             dx(j) =dx(j) -delx(jj)
             dx(j1)=dx(j1)-delx(jj+1)
             dx(j2)=dx(j2)-delx(jj+2)
             dx(j3)=dx(j3)-delx(jj+3)
             dy(j) =dy(j) -dely(jj)
             dy(j1)=dy(j1)-dely(jj+1)
             dy(j2)=dy(j2)-dely(jj+2)
             dy(j3)=dy(j3)-dely(jj+3)
             dz(j) =dz(j) -delz(jj)
             dz(j1)=dz(j1)-delz(jj+1)
             dz(j2)=dz(j2)-delz(jj+2)
             dz(j3)=dz(j3)-delz(jj+3)
          enddo vdw_loop2
          !     cleanup
          do jj=(cachednpr/4)*4+1,cachednpr
             j=jmember(jj)
             dx(j)=dx(j)-delx(jj)
             dy(j)=dy(j)-dely(jj)
             dz(j)=dz(j)-delz(jj)
          enddo

          !     restore the i-th component of the force
          dx(i) = dx(i)+dtx
          dy(i) = dy(i)+dty
          dz(i) = dz(i)+dtz
       endif do_pairs
       itemp = inbl(i)
    enddo natom_loop
    !
    return
#if KEY_MKLLIB==0
  contains
    subroutine vdinvsqrt(n,asq,ainvsqrt)
      integer n
      real(chm_real)::asq(n),ainvsqrt(n)

      ainvsqrt = one/sqrt(asq)
      return
    end subroutine vdinvsqrt

    subroutine vdinv(n,a,ainv)
      integer n
      real(chm_real)::a(n),ainv(n)

      ainv = one/a
      return
    end subroutine vdinv
#endif

  end subroutine rewald95_sb
  !   End of modified c33a2 code for speed testing......MFC
#endif /* (fewsb)*/


#else /* (mfc_fast_ewald)*/

  !======================================================================================
  !---Original source from c33a1 tree
  !----------------------------------------------------------------------------
  !             REWALD
  !----------------------------------------------------------------------------
  subroutine rewald(enb,eel,natomx,jnbl,inbl,lelecx,lvdwx, &
       iacnb,nitcc2,lowtp, &
       dx,dy,dz,x,y,z,cgx, &
#if KEY_FLUCQ==1
       qfluc,fqcfor,    &
#endif
#if KEY_FASTEW==1
       intcache,icsz,realcache,rcsz,    &
#endif
       lused)
    !
    !-----------------------------------------------------------------------
    !---    Calculate nonbonded interaction energies and forces.
    !
    !---    ENB     - calculated vdw interaction energy
    !---    EEL     - calculated electrostatic interaction energy
    !---    NATOMX  - number of atoms
    !---    JNBL    - nonbond atom pair list for index j
    !---    INBL    - nonbond atom pair list for index i
    !---    LELECX  - compute electrostatic energy?
    !---    LVDWX   - compute van der Waals energy?
    !---    CCNBA, CCNBB, CCNBC, CCNBD -
    !---    IACNB   -
    !---    NITCC2  -
    !---    LOWTP   -
    !---    DX, DY, DZ - atom forces as Cartesian coordinates
    !---    X, Y, Z - atom positions as Cartesian coordinates
    !---    CGX     - charges
    !---    EWLDT   - ERFc lookup table
    !---    QFLUc   -
    !---    FQCFOR  -
    !---    INTCACHE  - in the FAST EWald version cache storage for integers
    !---    REALCACHE - in the FAST EWald version cache storage for real(kind=chm_real)s
    !--- ab ICSZ,RCSZ - first dimension of above
    !---    LUSED   - was anything calculated ?
    !
    !---------------------------------------------------------------------
    !---    This is the fast scalar version of the nonbonded energy terms.
    !---    Electrostatic interactions herein are calculated via the
    !---    direct part of the Ewald summation.
    !---    Van der Waals interactions can be shifted or switched, but
    !---    electrostatic interactions are truncated.
    !
    !---    October 15, 1991  Roland H. Stote
    !---    21-Oct-91, Roland Stote
    !---               ATOM/VATOM nonbond cutoff options are supported.
    !---    2001?      Norm T SGI
    !---               FASTEW; SGI performance optimizations.
    !---    January 3, 2002   Scott R. Brozell TSRI
    !---               FASTEW; more SGI performance optimizations; these have
    !---               been shown to be beneficial for Intel 32 bit (IA32).
    !---    August 2002   Sandeep A. Patel TSRI
    !---               CHEQ added to non-FASTEW.
    !---    November 12, 2003   Scott R. Brozell TSRI
    !---               CHEQ added to FASTEW.
    !
    !...ab.HybH.See below for details. A.Blondel.
    !...If this implemantation breaks optimization, duplicate and rename REWALDB
    !...Can be tested easyly by compiling without "BLOCK" in pref.dat.
    !...For the FASTEW part, can imagine to move all the pairs involved in
    !...reactant or product in an extra loop for efficiency...
    !-----------------------------------------------------------------------

    use erfcd_mod
#if KEY_CHEQ==1
    use cheq,only: qcg,qpartbin,DCH,SUMDCH,DDCH
#endif
    use nb_module,only:ccnba,ccnbb,ccnbc,ccnbd
    use new_timer,only:timer_start,timer_stop,T_2extra

#if KEY_BLOCK==1
    use block_fcm
    use trunk,only:idx,ndx,ptable,maxpai,r02l,ndxm,maxtab
#endif
    use param
    use inbnd,only: &
#if KEY_IMCUBES==1
         lbycbim, &
#endif
         lgroup,lshft,ctofnb,e14fac,lvfswt, &
         lvshft,lvdw,ctonnb,eps
    use consta,only:ccelec
#if KEY_DRUDE==1
    use psf
#endif
#if KEY_GAMESS==1 || KEY_GAMESSUK==1 || KEY_MNDO97==1 || KEY_SQUANTM==1 || \
    KEY_QCHEM==1 || KEY_QTURBO==1 || KEY_G09==1
    use gamess_fcm
#endif
#if KEY_PBOUND==1
    use pbound
#endif

    real(kind=chm_real)  enb,eel
    integer natomx,jnbl(*),inbl(*)
    logical lelecx,lvdwx
    integer iacnb(*),nitcc2,lowtp(*)
    real(kind=chm_real)  x(*),y(*),z(*),dx(*),dy(*),dz(*)
    real(kind=chm_real)  cgx(*)
#if KEY_FLUCQ==1
    logical qfluc
    real(kind=chm_real)  fqcfor(*)
#endif

#if KEY_BLOCK==1
    logical qout
    integer ibl,jbl,kk,idxm,pt,id
    real(chm_real) r02,czz,accnb,bccnb,l6thr,l6thp,dl112r,dl112p
    real(chm_real) tfe,fr,fp,dfr,dfp,dnnr,dner,dnnp,dnep
#endif
    !.ab.
    !av_080628
    ! Block code added:H Kamberaj November 2007
    real(kind=chm_real)  cd
#if KEY_BLOCK==1
    REAL(kind=chm_real)    :: COEF
    INTEGER                :: KKB
#endif /*  close BLOCK*/
    !av_080628
#if KEY_FASTEW==1
    !---Cache storage for FAST EWald version
    !.ab.      integer intcache(intcrecsz,*)
    !.ab.      real(kind=chm_real)  realcache(realcrecsz,*)
    integer icsz,rcsz
    integer intcache(icsz,*)
    real(kind=chm_real)  realcache(rcsz,*)
    !.ab.
#endif
    logical lused

#if KEY_CHEQ==1
    REAL(KIND=CHM_REAL) DCHI
    REAL(KIND=CHM_REAL) E14FCQ
#endif

    !
    integer ivect,kvect
    real(kind=chm_real)  ca,cb,cc,ch,ene,enn
    real(kind=chm_real)  tf,tx,ty,tz,dtx,dty,dtz
    real(kind=chm_real)  s2,tr2,tr6,fsw,dfsw,dfrs
    !
    real(kind=chm_real)  cgf,cgt,crxi,cryi,crzi,crxj,cryj,crzj
    integer itemp,i,j,jj,npr,iaci,i0
    real(kind=chm_real)  c2onnb,c2ofnb,rul3,rul12,rijl,riju
    real(kind=chm_real)  e14fm1,e14f,rs,r1s,erfc2,erfcx,drfc
    real(kind=chm_real)  xval,rem,val0,val1,val2,d1,d2
    integer ixval
    logical elecfg
    !
    !---    Translation Table of Some Variable Names
    !
    !---    DTX     - tally in the pair loop of the x force displacements
    !---    DTY     - tally in the pair loop of the y force displacements
    !---    DTZ     - tally in the pair loop of the z force displacements
    !---    ENE     - update to the electrostatic energy
    !---    ENN     - update to the van der Waals energy
    !---    E14F    - modified electrostatic scale factor for 1-4 interactions
    !---    FSW     - van der Waals switch function
    !---    LVSHFT  - compute van der Waals energy via shifting?
    !---    NPR     - number of pairs
    !---    RS      - delta r
    !---    R1S     - one over delta r
    !---    S2      - delta r**2
    !---    TF      - force displacement factor
    !---    TR2     - one over delta r**2
    !---    TX      - delta x
    !---    TY      - delta y
    !---    TZ      - delta z
    !
#if KEY_FASTEW==1
    !---***** NEW Code Norm T SGI
    !---Temps put in to take operations out of loops
    real(kind=chm_real) ekappa
    !---Declarations for loop fissioning and arithemetic transformations
    real(kind=chm_real)    xforce,yforce,zforce  ! x,y,z force displacements
    integer   cachednpr             ! the number of cached records
    !---Record structures for the integer and real(kind=chm_real) caches
    !---integer cache
    integer, parameter :: k_member=1
    !---real(kind=chm_real) cache
    integer,parameter :: s2_member=1, tf_member = 2, tx_member = 3,  &
         ty_member = 4, tz_member = 5
    integer, parameter :: kb_member=1, r02_member=6, ene_member=7 !.ab.
    real(kind=chm_real) enbtmp
#endif /*  FASTEW*/
#if KEY_PBOUND==1
    real(chm_real) corr
#endif


    !---------- Sanity check -------------------------------------
    if(.not. allocated(ccnba))then
       ! How we got here without vdw table filled, who knows?
       call wrndie(-4,"rewald<rewald.src>", &
            "CCNBA not allocated")
    endif

    !
    !---Begin.


    !---    check to see if we should be here...
    LUSED = .FALSE.
    IF(ERFMOD >= 0) RETURN
    IF(LGROUP) RETURN
    IF(LVFSWT) RETURN
    LUSED = .TRUE.
    !
    E14FM1 = E14FAC - ONE

#if KEY_CHEQ==1
    E14FCQ = zero
#endif

    !
    C2OFNB = CTOFNB*CTOFNB
    IF (.NOT.(LSHFT.AND.LVSHFT)) THEN
       C2ONNB = CTONNB*CTONNB
       IF (CTOFNB > CTONNB) THEN
          RUL3 = ONE/(C2OFNB-C2ONNB)**3
          RUL12 = TWELVE*RUL3
#if KEY_FASTEW==1
          !---Precompute values needed for the VDW switch arithemetic transformations
          !---Note that when CTOFNB  <  CTONNB and when CTOFNB  ==  CTONNB
          !---the value of C2ONNB below insures a zero length switch region.
       ELSE
          C2ONNB = C2OFNB+RSMALL
          RUL3 = ONE/(C2OFNB-C2ONNB)**3
          RUL12 = ZERO
#endif
       ENDIF
    ENDIF
    !
    ENB = ZERO
    EEL = ZERO
    ELECFG = (LELECX.AND.(EPS /= 0.0))
    IF (ELECFG) THEN
       CGF = CCELEC/EPS
    ELSE
       CGF = ZERO
    ENDIF
    IF(.NOT.(LVDW.OR.ELECFG)) RETURN
    !
    !.ab.Coefs, tables... for HYBH.
#if KEY_BLOCK==1
    if (qhybh) then
       dnnr=zero
       dner=zero
       dnnp=zero
       dnep=zero
       fr=(three*hybhlb-four)*(hybhlb-one)/four
       dfr=(six*hybhlb-seven)/four
       fp=(three*hybhlb+one)*hybhlb/four
       dfp=(six*hybhlb+one)/four
       if (hybhlb == zero) then
          l6thr=zero
          l6thp=one
          dl112r=zero
          dl112p=minone/twelve
       else if (hybhlb == one) then
          l6thr=one
          l6thp=zero
          dl112r=one/twelve
          dl112p=zero
       else
          l6thr=hybhlb**sixth
          l6thp=(one-hybhlb)**sixth
          dl112r=one/hybhlb/twelve
          dl112p=minone/(one-hybhlb)/twelve
       endif
       !.ab. Make table for Indeces
       if (ptable < natomx) then
          idxm=0
          do i=1,natomx
             do j=i-1,1,-1
                if ((iacnb(j) == iacnb(i)).and.(cgx(i).eq.cgx(j))) &
                     idx(i)=idx(j)
             enddo
             if (idx(i) == -1) then
                idxm=idxm+1
                idx(i)=idxm
             endif
          enddo
          !            write(outu,'(i5,a20)') idxm,' Atom types for HYBH'
          if ((idxm+1)*idxm > maxpai) call wrndie(-6,'<rewald>', &
               'Too many atom types, Increase MAXPAI.')
          ptable=natomx
       endif
    endif
#endif
    !.ab.Note about HYBH scheme. I choose to exclude block 2-3 interactions
    !....in real space and exclusions (?) calculations because atoms from
    !....these blocks can be close and thus could create an end-point
    !....catastrophy. It is not consistant whith the way Ewald sum are
    !....calculated, but this should not matter since the end points are
    !....correct, and the "alchimical" path does not matter.
    !.ab.
    !
    !---    DO VDW AND ELECTROSTATIC TERMS AS REQUESTED
    !
#if KEY_FASTEW==0 /*fast_ewald*/
    !---This is the original, ie, non-optimized for SGI source
    ITEMP = 0
    DO I = 1,NATOMX
#if KEY_IMCUBES==1
       IF(LBYCBIM)ITEMP = INBL(I+NATOMX)
#endif
       NPR = INBL(I) - ITEMP
       IF(NPR > 0) THEN
#if KEY_CHEQ==1
          DCHI=ZERO
#else /**/
          CGT = CGX(I)*CGF
#endif
          !.ab.HybH.Do block in loop (outer loop should not hurt).
#if KEY_BLOCK==1
          if (qhybh) ibl=iblckp(i)
#endif
          iaci = iacnb(i)
          crxi = x(i)
          cryi = y(i)
          crzi = z(i)
          dtx = zero
          dty = zero
          dtz = zero
          !
          !---pair loop
          pair_loop: do jj = 1,npr
             kvect = jnbl(itemp+jj)
             j = abs(kvect)
             !---namkh 01/20/04
#if KEY_GAMESS==1 || KEY_GAMESSUK==1 || KEY_MNDO97==1 || KEY_SQUANTM==1 || \
    KEY_QCHEM==1 || KEY_QTURBO==1 || KEY_G09==1
             if(qmused)then
             if((abs(igmsel(i)) == 1 .or. abs(igmsel(i)) == 2) .and. &
                  (abs(igmsel(j)) == 1 .or. abs(igmsel(j)) == 2) &
                  .and. qgmrem) cycle pair_loop
             endif
#endif
             crxj = x(j)
             cryj = y(j)
             crzj = z(j)

             tx = crxi-crxj
             ty = cryi-cryj
             tz = crzi-crzj
#if KEY_PBOUND==1
             If(qBoun) then
                If(qCUBoun.or.qTOBoun) then
                   tx = BOXINV * tx
                   ty = BOYINV * ty
                   tz = BOZINV * tz
                   tx = tx - nint(tx)
                   ty = ty - nint(ty)
                   tz = tz - nint(tz)
                   If (qTOBoun) Then
                      CORR = HALF * AINT ( R75 * (ABS(TX) + &
                           ABS(TY) + &
                           ABS(TZ)))
                      TX = TX - SIGN( CORR,  TX  )
                      TY = TY - SIGN( CORR,  TY  )
                      TZ = TZ - SIGN( CORR,  TZ  )
                   Endif
                   TX = XSIZE * TX
                   TY = YSIZE * TY
                   TZ = ZSIZE * TZ
                Else
                   Call PBMove(TX, TY, TZ)
                Endif
             Endif
#endif
             s2 = max(rsmall,tx*tx+ty*ty+tz*tz)
             if (s2 < c2ofnb) then

                !.ab.
#if KEY_BLOCK==1
                if (qhybh) then
                   jbl=iblckp(j)
                   kk=max(ibl,jbl)
                   kk=kk*(kk-1)/2+min(ibl,jbl)
                   if (kk /= 1) then
                      if (kk == 5) then
                         goto 30
                      else
                         pt=max(idx(i),idx(j))
                         pt=pt*(pt-1)/2+min(idx(i),idx(j))
                         id=ndx(pt)
                         r02=r02l(id)
                         !.ab. R02L(1)=-1. is none atributed signal.
                         !    Give an index and calculate R02.
                         if (r02 < 0.) then
                            if ( (idx(i) == -1 ).or.(idx(j) == -1 ) ) then
                               call wrndie(-5,'<REWALD>','HYBH: IDX table misinitiated.')
                            endif
                            ndxm=ndxm+1
                            if(ndxm > maxtab)call wrndie(-5,'<rewaldb>', &
                                 'increase maxtab(trunk.f90).')
                            ndx(pt)=ndxm
                            czz=cgt*cgx(j)
                            ivect=iacnb(j)+iaci
                            accnb=ccnba(ivect)
                            bccnb=ccnbb(ivect)
                            call getr02(r02l(ndxm),czz,accnb,bccnb, &
                                 idxm,ndxm,i,j)
                            r02=r02l(ndxm)
                         endif
                         if (kk < 4) then
                            r02=r02*l6thr
                         else
                            r02=r02*l6thp
                         endif
                         qout=.true.
                         if (s2 < r02) then
                            s2=r02
                            qout=.false.
                         endif
                      endif
                   endif
                   !.KK=1, do nothing.
                endif
#endif
                !.ab.

                ivect = lowtp(max(iacnb(j),iaci))+iacnb(j)+iaci
#if KEY_CHEQ==1
                ch=cgf
#else /**/
                ch=cgt*cgx(j)
#endif

                !av_080628 Block code: H Kamberaj November 2007
#if KEY_BLOCK==1 /*(block)*/
                IF (QBLOCK) THEN
                   ibl = iblckp(i)
                   jbl = iblckp(j)
                   if (jbl  <  ibl) then
                      kkb = ibl
                      ibl = jbl
                      jbl = kkb
                   endif
                   kkb = ibl + jbl*(jbl-1)/2
                   coef = blcoee(kkb)
                   CH = CH*coef
                ENDIF
#endif /*    (block)*/
                !av_080628
                e14f = zero
#if KEY_CHEQ==1
                e14fcq = zero
#endif
                if(kvect < 0) then
                   e14f = e14fm1
#if KEY_CHEQ==1
                   e14fcq = -1.0
#endif
                   ivect = ivect+nitcc2
                endif

                tr2 = one/s2
                tr6 = tr2*tr2*tr2
                ca = ccnba(ivect)*tr6*tr6
                cb = ccnbb(ivect)*tr6
                !av_080628
                ! Block code: H Kamberaj November 2007
#if KEY_BLOCK==1 /*(block)*/
                IF (QBLOCK) THEN
                   CA=CA*coef
                   CB=CB*coef
                ENDIF
#endif /*    (block)*/
                !av_080628
                !---    Calculate van der Waals and Electrostatic energies
                !
                if(lvdwx) then
                   if(lvshft) then
                      cc = s2*s2*s2*ccnbc(ivect)
                      !av_080628                        enn = ca-cb-cc+ccnbd(ivect)
                      CD = CCNBD(IVECT)
                      !
                      ! Block code: H Kamberaj November 2007
#if KEY_BLOCK==1 /*(block)*/
                      IF (QBLOCK) THEN
                         CC=CC*coef
                         CD=CD*coef
                      ENDIF
#endif /*    (block)*/

                      ENN = CA-CB-CC+CD
                      !av_080628
                      tf = six*(cb-ca-ca-cc)*tr2
                   else if(s2 > c2onnb) then
                      !---                   van der waals switch function
                      rijl = c2onnb-s2
                      riju = c2ofnb-s2
                      fsw = riju*riju*(riju-three*rijl)*rul3
                      dfsw = rijl*riju*rul12
                      enn = (ca-cb)*fsw
                      tf = (ca-cb)*dfsw-six*tr2*(ca+ca-cb)*fsw
                   else
                      enn = (ca-cb)
                      tf = -six*tr2*(ca+ca-cb)
                   endif
                else
                   enn = zero
                   tf = zero
                endif

                !---ERROR FUNCTION CALCULATION FOR REAL SPACE PART OF EWALD
                !------      EWALD R-space sum
                rs = sqrt(s2)
                r1s = one/rs
                !---call ERFCD( RS, KAPPA, ERFCX, DRFC, EWLDT, -1 )
                !----1 means lookup table method using cubic spline interpolation
                !---Inline erfc calculation for speed.
                xval = rs*kappa*ewrdel
                ixval = xval+half
                rem = xval-ixval
                ixval=ixval+2
                ixval = min(ixval,ewnpts-1)
                val0 = ewldt(ixval-1)
                val1 = ewldt(ixval)
                val2 = ewldt(ixval+1)
                d1 = (val0-val2)*half
                d2 = (val1+val1-val0-val2)*rem
                erfcx = val1-(d1+half*d2)*rem
                drfc = (d1+d2)*ewrdel*kappa
                !---end of erfc inlined code.
                erfc2 = erfcx + e14f

#if KEY_CHEQ==1
                if(qcg)then
                   if ( (qpartbin(i) /= 0) .and. &
                        (qpartbin(j) /= 0) ) &
                        erfc2 = erfcx + e14fcq
                endif
#endif

                ene = ch*erfc2*r1s
                dfrs = (ch*drfc + ene)*tr2 &
#if KEY_CHEQ==1
                     *cgx(i)*cgx(j)
#endif
#if KEY_CHEQ==0
                ;
#endif
                !.ab.QHYBH code. TF = dvdW/|r|.dr
#if KEY_BLOCK==1
                if (qhybh) then
                   if (kk == 1) then
                      tf=tf-dfrs
                   else if (kk < 4) then
                      dner=dner+ene*dfr
                      dnnr=dnnr+enn*dfr
                      if (qout) then
                         tf=tf-dfrs
                         tf=tf*fr
                      else
                         !.
                         dner=dner-fr*dfrs*r02*dl112r
                         dnnr=dnnr+fr*tf*r02*dl112r
                         tf=zero
                      endif
                      ene=ene*fr
                      enn=enn*fr
                   else
                      dnep=dnep+ene*dfp
                      dnnp=dnnp+enn*dfp
                      if (qout) then
                         tf=tf-dfrs
                         tf=tf*fp
                      else
                         dnep=dnep-fp*dfrs*r02*dl112p
                         dnnp=dnnp+fp*tf*r02*dl112p
                         tf=zero
                      endif
                      ene=ene*fp
                      enn=enn*fp
                   endif
                else
                   tf=tf-dfrs
                endif
#else /**/
                tf = tf-dfrs
#endif
                !.ab.

                dtx = dtx+tf*tx
                dty = dty+tf*ty
                dtz = dtz+tf*tz
                dx(j) = dx(j)-tf*tx
                dy(j) = dy(j)-tf*ty
                dz(j) = dz(j)-tf*tz

#if KEY_CHEQ==1
                if (qcg) then
                   if ( (cgx(i) /= zero).and.(cgx(j) /= zero) ) then
                      dchi=dchi+ene*cgx(j)
                      dch(j)=dch(j)+ene*cgx(i)
                   else                          !for tip4pfq type cases
                      dchi=dchi + 0.0
                   endif                         !for tip4pfq type cases
                endif
                ene=ene*cgx(j)*cgx(i)
#endif

#if KEY_FLUCQ==1
                if (qfluc) then
                   fqcfor(i) = fqcfor(i)+ene
                   fqcfor(j) = fqcfor(j)+ene
                endif
#endif
                ENB = ENB+ENN
                EEL = EEL+ENE

             ENDIF
          ENDDO pair_loop
          !---          restore the i-th component of the force
          DX(I) = DX(I)+DTX
          DY(I) = DY(I)+DTY
          DZ(I) = DZ(I)+DTZ
#if KEY_CHEQ==1
          IF (QCG) DCH(I)=DCH(I)+DCHI
#endif
       ENDIF
       ITEMP = INBL(I)
    ENDDO
    !.ab.Save.
#if KEY_BLOCK==1
    if (qhybh) then
       call sumhyb(ihybh,dnnr,dnnp)
       call sumhyb(ihybh+1,dner,dnep)
    endif
#endif
    !.ab.
    !
#else /* (fast_ewald)*/
    !---
    !---  ***** NEW Code Norm T SGI
    !---  Invert if(lvdwx), the VDW toggle
    !---  Invert if(LVSHFT), the VDW shift control
    !---  These inversions result in three cases: VDW shift, VDW switch,
    !---  and No VDW.
    !---  Reorder and remove unneeded operations for better
    !---  register usage and pipeline flow.
    !---
    !---  ***** Further performance improvements; Scott Brozell TSRI
    !---  To produce pipelined loops the SGI compiler needs more help.
    !---  For each of the three cases the inner, pair loop from 1 to the
    !---  number of pairs is fissioned into three loops:
    !---  prologue, electrostatic, and VDW.
    !---  The prologue loop computes delta r**2 and conditionally caches
    !---  it, delta x, delta y, delta z, and the JNBL index.
    !---  The electrostatic loop computes and caches the direct Ewald sum.
    !---  The VDW loop computes the van der Waals interaction and updates
    !---  the forces using the previously cached data.
    !---  These changes are similar to those made in AMBER's short_ene.
    !---  Additional arithmetic transformations produced efficiency
    !---  enhancements, and some were necessary to pipeline the
    !---  electrostatic and VDW loops.  For example, one over delta r**2
    !---  has been factored out of the electrostatic and van der Waals
    !---  computations and multiplied back in during the force updating.
    !---
    !---  Precompute values needed for Ewald erfc argument calculation
    EKAPPA = EWRDEL*KAPPA

    itemp = 0
    if(lvdwx) then
       lvshft_blk: if(lvshft) then
          !---  VDW shift.  Case 1 of 3.  (LVDWX .and. LVSHFT)
          do i = 1,natomx
#if KEY_IMCUBES==1
             if(lbycbim) itemp = inbl(i+natomx)
#endif
             npr = inbl(i) - itemp
             if(npr > 0) then
#if KEY_CHEQ==1
                DCHI = ZERO
#else /**/
                CGT = CGX(I)*CGF
#endif
                !.ab.HybH.Do block in loop (outer loop should not hurt).
#if KEY_BLOCK==1
                if (qhybh) ibl=iblckp(i)
#endif
                iaci = iacnb(i)
                crxi = x(i)
                cryi = y(i)
                crzi = z(i)
                dtx = zero
                dty = zero
                dtz = zero
                !--- prologue loop: compute and conditionally cache delta r**2
                !.ab.HybH.Modify prologue loop.
                cachednpr = 0
                do jj = 1,npr
                   kvect = jnbl(itemp+jj)
                   j = abs(kvect)

                   !---- namkh 01/20/04
                   !---- This also possible to be screwed up, but hope not.
#if KEY_GAMESS==1 || KEY_GAMESSUK==1 || KEY_MNDO97==1 || KEY_SQUANTM==1 || \
    KEY_QCHEM==1 || KEY_QTURBO==1 || KEY_G09==1 /*gamess*/
                   if(qmused)then
                   if((abs(igmsel(i)) == 1.or.abs(igmsel(i)).eq.2) .and. &
                        (abs(igmsel(j)) == 1.or.abs(igmsel(j)).eq.2)) &
                        goto 200
                   endif
#endif /* (gamess)*/

                   tx = crxi-x(j)
                   ty = cryi-y(j)
                   tz = crzi-z(j)
#if KEY_PBOUND==1
                   If(qBoun) then
                      If(qCUBoun.or.qTOBoun) then
                         tx = BOXINV * tx
                         ty = BOYINV * ty
                         tz = BOZINV * tz
                         tx = tx - nint(tx)
                         ty = ty - nint(ty)
                         tz = tz - nint(tz)
                         If (qTOBoun) Then
                            CORR = HALF * AINT ( R75 * (ABS(TX) + &
                                 ABS(TY) + &
                                 ABS(TZ)))
                            TX = TX - SIGN( CORR,  TX  )
                            TY = TY - SIGN( CORR,  TY  )
                            TZ = TZ - SIGN( CORR,  TZ  )
                         Endif
                         TX = XSIZE * TX
                         TY = YSIZE * TY
                         TZ = ZSIZE * TZ
                      Else
                         Call PBMove(TX, TY, TZ)
                      Endif
                   Endif
#endif
                   s2 = max(rsmall,tx*tx+ty*ty+tz*tz)
                   if (s2 < c2ofnb) then
                      !.ab.
#if KEY_BLOCK==1
                      if(.not.qhybh) then
#endif
                         cachednpr = cachednpr + 1
                         intcache(k_member,cachednpr) = kvect
                         realcache(s2_member,cachednpr) = s2
                         realcache(tx_member,cachednpr) = tx
                         realcache(ty_member,cachednpr) = ty
                         realcache(tz_member,cachednpr) = tz
                         !.ab.
#if KEY_BLOCK==1
                      else
                         jbl=iblckp(j)
                         kk=max(ibl,jbl)
                         kk=kk*(kk-1)/2+min(ibl,jbl)
                         if (kk /= 1) then
                            if (kk /= 5) then
                               pt=max(idx(i),idx(j))
                               pt=pt*(pt-1)/2+min(idx(i),idx(j))
                               id=ndx(pt)
                               r02=r02l(id)
                               if (r02 < 0.) then
                                  if ( (idx(i) == -1 ).or.(idx(j) == -1 ) )  then
                                     call wrndie(-5,'<REWALD>', &
                                         'HYBH: IDX table misinitiated.')
                                  endif
                                  ndxm=ndxm+1
                                  if(ndxm > maxtab)call wrndie(-5, &
                                       '<rewaldb>','increase maxtab(trunk.f90).')
                                  ndx(pt)=ndxm
                                  czz=cgt*cgx(j)
                                  ivect=iacnb(j)+iaci
                                  accnb=ccnba(ivect)
                                  bccnb=ccnbb(ivect)
                                  call getr02(r02l(ndxm),czz,accnb,bccnb, &
                                       idxm,ndxm,i,j)
                                  r02=r02l(ndxm)
                               endif
                               if (kk < 4) then
                                  r02=r02*l6thr
                               else
                                  r02=r02*l6thp
                               endif
                               qout=.true.
                               if (s2 < r02) then
                                  qout=.false.
                               endif
                            endif
                            !.kk=5, do not keep...
                         endif
                         !.KK=1, do nothing.
                         !...
                         !.KK=5, do not keep...
                         if(kk /= 5)then
                            cachednpr = cachednpr + 1
                            intcache(k_member,cachednpr) = kvect
                            if(qout) then
                               intcache(kb_member,cachednpr) = kk
                            else
                               intcache(kb_member,cachednpr) = -kk
                            endif
                            realcache(s2_member,cachednpr) = s2
                            realcache(tx_member,cachednpr) = tx
                            realcache(ty_member,cachednpr) = ty
                            realcache(tz_member,cachednpr) = tz
                            realcache(r02_member,cachednpr) = r02
                         endif
                      endif
#endif
                      !.ab.
                   endif
200                continue
                enddo
                !---electrostatic loop: compute and cache direct ewald sum
                do jj = 1,cachednpr
                   s2 = realcache(s2_member,jj)
                   kvect = intcache(k_member,jj)
                   j = abs(kvect)
                   !                 ewald real space sum using the error function
                   rs = sqrt(s2)
                   r1s = one/rs
                   !
                   !---call ERFCD( RS, KAPPA, ERFCX, DRFC, EWLDT, -1 )
                   !----1 means lookup table method using cubic spline interpolation
                   !     !---Inlined erfc calculation for speed (from ERFCD).
                   xval = rs*ekappa
                   ixval = xval+half
                   rem = xval-ixval
                   ixval=ixval+2
                   ixval = min(ixval,ewnpts-1)
                   val0 = ewldt(ixval-1)
                   val1 = ewldt(ixval)
                   val2 = ewldt(ixval+1)
                   d1 = (val0-val2)*half
                   d2 = (val1+val1-val0-val2)*rem
                   erfcx = val1-(d1+half*d2)*rem
                   drfc = (d1+d2)*ekappa
                   !      !---end of erfc inlined code.
                   !
                   e14f = zero
#if KEY_CHEQ==1
                   e14fcq = zero
#endif
                   if(kvect < 0) then
                      e14f = e14fm1
#if KEY_CHEQ==1
                      e14fcq = -1.0
#endif
                   endif
#if KEY_CHEQ==1 /*cheq*/
                   ch = cgf
                   if(qcg)then

                      if ( (qpartbin(i) /= 0)  &
                           .and. (qpartbin(j) /= 0)) then
                         ene = ch*(erfcx + e14fcq)*r1s
                      else
                         ene = ch*(erfcx + e14f)*r1s
                      endif

                      if ( (cgx(i) /= zero).and.(cgx(j).ne.zero) ) then
                         dchi = dchi + ene*cgx(j)
                         dch(j) = dch(j) + ene*cgx(i)
                      else                          ! for tip4pfq type cases
                         dchi = dchi + zero
                      endif                         ! for tip4pfq type cases

                   else
                      ene = ch*(erfcx + e14f)*r1s
                   endif

                   ene = ene*cgx(i)*cgx(j)
                   ch = ch*cgx(i)*cgx(j)
#else /*  (cheq)*/
                   ch = cgt*cgx(j)
                   ene = ch*(erfcx + e14f)*r1s
#endif /* (cheq)*/

                   !av_080628
                   ! Block code: H Kamberaj November 2007
                   !
#if KEY_BLOCK==1 /*(block)*/
                   IF (QBLOCK) THEN
                      ibl = iblckp(i)
                      jbl = iblckp(j)
                      if (jbl  <  ibl) then
                         kkb = ibl
                         ibl = jbl
                         jbl = kkb
                      endif
                      kkb = ibl + jbl*(jbl-1)/2
                      coef = blcoee(kkb)
                      ch = ch * coef
                      ene = ene * coef
                   ENDIF
#endif /*    (block)*/
                   !av_080628
                   eel = eel+ene
                   realcache(tf_member,jj) = -(ch*drfc + ene)
                   !.ab.
#if KEY_BLOCK==1
                   if(QHYBH) realcache(ene_member,jj) = ene
#endif
                   !.ab.
#if KEY_FLUCQ==1
                   if (qfluc) then
                      fqcfor(i) = fqcfor(i)+ene
                      fqcfor(j) = fqcfor(j)+ene
                   endif
#endif
                enddo

                !.ab.Note: I don't see why the elec loop is repeated 3 times....
#if KEY_BLOCK==1
                if(.not.qhybh) then
#endif
                   !--- vdw loop: compute van der waals interaction and update forces
                   do jj = 1,cachednpr
                      s2 = realcache(s2_member,jj)
                      kvect = intcache(k_member,jj)
                      j = abs(kvect)
                      ivect = lowtp(max(iacnb(j),iaci))+iacnb(j)+iaci
                      if(kvect < 0) then
                         ivect = ivect+nitcc2
                      endif
                      tr2 = one/s2
                      tr6 = tr2*tr2*tr2
                      ca = ccnba(ivect)*tr6
                      !av_080628                  enn = (ca-ccnbb(ivect))*tr6
                      cb = ccnbb(ivect)
                      cc = s2*s2*s2*ccnbc(ivect)
                      cd = ccnbd(ivect)
                      !
                      ! Block code: H Kamberaj November 2007
                      !
#if KEY_BLOCK==1 /*(block)*/
                      IF (QBLOCK) THEN
                         ibl = iblckp(i)
                         jbl = iblckp(j)
                         if (jbl  <  ibl) then
                            kkb = ibl
                            ibl = jbl
                            jbl = kkb
                         endif
                         kkb = ibl + jbl*(jbl-1)/2
                         coef = blcoev(kkb)
                         ca = ca * coef
                         cb = cb * coef
                         cc = cc * coef
                         cd = cd * coef
                      ENDIF
#endif /*    (block)*/
                      !
                      enn = (ca-cb)*tr6
                      !av_080628
                      !yw_080813                  cc = s2*s2*s2*ccnbc(ivect)
                      !---                van der waals shift
                      tf = -six*(ca*tr6+enn+cc)
                      !av_080628                  enn = enn+ccnbd(ivect)-cc
                      enn = enn+cd-cc
                      enb = enb+enn
                      tf = (realcache(tf_member,jj) + tf) * tr2
                      !---                update forces
                      xforce = realcache(tx_member,jj) * tf
                      yforce = realcache(ty_member,jj) * tf
                      zforce = realcache(tz_member,jj) * tf
                      dtx = dtx+xforce
                      dty = dty+yforce
                      dtz = dtz+zforce
                      dx(j) = dx(j)-xforce
                      dy(j) = dy(j)-yforce
                      dz(j) = dz(j)-zforce
                   enddo
                   !.ab.HybH code.QHYBH=.TRUE.
#if KEY_BLOCK==1
                else
                   do jj = 1,cachednpr
                      s2 = abs(realcache(s2_member,jj))
                      kvect = intcache(k_member,jj)
                      j = abs(kvect)
                      ivect = lowtp(max(iacnb(j),iaci))+iacnb(j)+iaci
                      if(kvect < 0) then
                         ivect = ivect+nitcc2
                      endif
                      tr2 = one/s2
                      tr6 = tr2*tr2*tr2
                      ca = ccnba(ivect)*tr6
                      enn = (ca-ccnbb(ivect))*tr6
                      cc = s2*s2*s2*ccnbc(ivect)
                      !---                van der waals shift
                      !.ab. tf * tr2 !
                      tf = -six*(ca*tr6+enn+cc) * tr2
                      enn = enn+ccnbd(ivect)-cc
                      enb = enb+enn
                      !.ab.
                      !                  tf = (realcache(tf_member,jj) + tf) * tr2
                      !.ab.QHYBH code. TF = dvdW/|r|.dr
                      kk=intcache(kb_member,jj)
                      qout=.true.
                      if(kk < 0) then
                         qout=.false.
                         kk=-kk
                      endif
                      tfe=realcache(tf_member,jj) * tr2
                      if (kk == 1) then
                         tf=tf + tfe
                      else if (kk < 4) then
                         ene=realcache(ene_member,jj)
                         dner=dner+ene*dfr
                         dnnr=dnnr+enn*dfr
                         if (qout) then
                            tf=tf+tfe
                            tf=tf*fr
                         else
                            r02=realcache(r02_member,jj)
                            dner=dner+fr*tfe*r02*dl112r
                            dnnr=dnnr+fr*tf *r02*dl112r
                            tf=zero
                         endif
                         ene=ene*fr
                         enn=enn*fr
                      else
                         ene=realcache(ene_member,jj)
                         dnep=dnep+ene*dfp
                         dnnp=dnnp+enn*dfp
                         if (qout) then
                            tf=tf+tfe
                            tf=tf*fp
                         else
                            dnep=dnep+fp*tfe*r02*dl112p
                            dnnp=dnnp+fp*tf *r02*dl112p
                            tf=zero
                         endif
                         ene=ene*fp
                         enn=enn*fp
                      endif
                      !.ab.

                      !---                update forces
                      xforce = realcache(tx_member,jj) * tf
                      yforce = realcache(ty_member,jj) * tf
                      zforce = realcache(tz_member,jj) * tf
                      dtx = dtx+xforce
                      dty = dty+yforce
                      dtz = dtz+zforce
                      dx(j) = dx(j)-xforce
                      dy(j) = dy(j)-yforce
                      dz(j) = dz(j)-zforce
                   enddo
                ENDIF
#endif
                !.ab.
                !
                !---             restore the i-th component of the force
                dx(i) = dx(i)+dtx
                dy(i) = dy(i)+dty
                dz(i) = dz(i)+dtz
#if KEY_CHEQ==1
                if (qcg) dch(i) = dch(i) + dchi
#endif
             endif
             itemp = inbl(i)
          enddo
       else          ! end of case 1 vdw shift
          !
          !----------------------------------------------------------------------
          !---            VDW switch.
          !----------------------------------------------------------------------
          !---            Case 2 of 3.  (LVDWX .and. .not. LVSHFT)
          !
          DO I0 = 1,NATOMX
             i=i0
#if KEY_IMCUBES==1
             IF(LBYCBIM) ITEMP = INBL(I+NATOMX)
#endif
             NPR = INBL(I) - ITEMP
             IF(NPR > 0) THEN
#if KEY_CHEQ==1
                if(qcg) DCHI = ZERO
#else /**/
                CGT = CGX(I)*CGF
#endif
                !.ab.HybH.Do block in loop (outer loop should not hurt).
#if KEY_BLOCK==1
                if (qhybh) ibl=iblckp(i)
#endif
                IACI = IACNB(I)
                CRXI = X(I)
                CRYI = Y(I)
                CRZI = Z(I)
                DTX = ZERO
                DTY = ZERO
                DTZ = ZERO
                !---prologue loop: compute and conditionally cache delta r**2
                CACHEDNPR = 0
                call timer_start(T_2extra)
                   DO JJ = 1,NPR
                      KVECT = JNBL(ITEMP+JJ)
                      J = ABS(KVECT)
                      !
                      !---namkh 01/20/04
                      !---This also possible to be screwed up, but hope not.
#if KEY_GAMESS==1 || KEY_GAMESSUK==1 || KEY_MNDO97==1 || KEY_SQUANTM==1 || \
    KEY_QCHEM==1 || KEY_QTURBO==1 || KEY_G09==1
                      if(qmused)then
                      IF((ABS(IGMSEL(I)) == 1.OR.ABS(IGMSEL(I)).EQ.2) .AND. &
                           (ABS(IGMSEL(J)) == 1.OR.ABS(IGMSEL(J)).EQ.2)) &
                           GOTO 320
                      endif
#endif
                      !
                      TX = CRXI-X(J)
                      TY = CRYI-Y(J)
                      TZ = CRZI-Z(J)
#if KEY_PBOUND==1
                      If(qBoun) then
                         If(qCUBoun.or.qTOBoun) then
                            tx = BOXINV * tx
                            ty = BOYINV * ty
                            tz = BOZINV * tz
                            tx = tx - nint(tx)
                            ty = ty - nint(ty)
                            tz = tz - nint(tz)
                            If (qTOBoun) Then
                               CORR = HALF * AINT ( R75 * (ABS(TX) + &
                                    ABS(TY) + &
                                    ABS(TZ)))
                               TX = TX - SIGN( CORR,  TX  )
                               TY = TY - SIGN( CORR,  TY  )
                               TZ = TZ - SIGN( CORR,  TZ  )
                            Endif
                            TX = XSIZE * TX
                            TY = YSIZE * TY
                            TZ = ZSIZE * TZ
                         Else
                            Call PBMove(TX, TY, TZ)
                         Endif
                      Endif
#endif
                      S2 = MAX(RSMALL,TX*TX+TY*TY+TZ*TZ)
                      IF (S2 < C2OFNB) THEN
                         !.ab. hybh code here ....
#if KEY_BLOCK==1
                         if(.not.qhybh) then
#endif
                            cachednpr = cachednpr + 1
                            INTCACHE(K_MEMBER,cachednpr) = KVECT
                            REALCACHE(S2_MEMBER,cachednpr) = S2
                            REALCACHE(TX_MEMBER,cachednpr) = TX
                            REALCACHE(TY_MEMBER,cachednpr) = TY
                            REALCACHE(TZ_MEMBER,cachednpr) = TZ
                            !.ab.
#if KEY_BLOCK==1
                         else
                            jbl=iblckp(j)
                            kk=max(ibl,jbl)
                            kk=kk*(kk-1)/2+min(ibl,jbl)
                            if (kk /= 1) then
                               if (kk /= 5) then
                                  pt=max(idx(i),idx(j))
                                  pt=pt*(pt-1)/2+min(idx(i),idx(j))
                                  id=ndx(pt)
                                  r02=r02l(id)
                                  if (r02 < 0.) then
                                     if ( (idx(i) == -1 ).or.(idx(j) == -1 ) ) then
                                        call wrndie(-5,'<REWALD>', &
                                             'HYBH: IDX table misinitiated.')
                                     endif
                                     ndxm=ndxm+1
                                     if(ndxm > maxtab)call wrndie(-5, &
                                          '<REWALDB>','Increase MAXTAB(trunk.f90).')
                                     ndx(pt)=ndxm
                                     czz=cgt*cgx(j)
                                     ivect=iacnb(j)+iaci
                                     accnb=ccnba(ivect)
                                     bccnb=ccnbb(ivect)
                                     call getr02(r02l(ndxm),czz,accnb,bccnb, &
                                          idxm,ndxm,i,j)
                                     r02=r02l(ndxm)
                                  endif
                                  if (kk < 4) then
                                     r02=r02*l6thr
                                  else
                                     r02=r02*l6thp
                                  endif
                                  qout=.true.
                                  if (s2 < r02) then
                                     qout=.false.
                                  endif
                               endif
                            endif
                            !.KK=1, do nothing.
                            !...
                            !.KK=5, do not keep...
                            if(kk /= 5)then
                               cachednpr = cachednpr + 1
                               intcache(k_member,cachednpr) = kvect
                               if(qout) then
                                  intcache(kb_member,cachednpr) = kk
                               else
                                  intcache(kb_member,cachednpr) = -kk
                               endif
                               realcache(s2_member,cachednpr) = s2
                               realcache(tx_member,cachednpr) = tx
                               realcache(ty_member,cachednpr) = ty
                               realcache(tz_member,cachednpr) = tz
                               realcache(r02_member,cachednpr) = r02
                            endif
                         endif
#endif
                         !.ab.
                      ENDIF
320                   CONTINUE
                   ENDDO
                call timer_stop(T_2extra)
                !---electrostatic loop: compute and cache direct Ewald sum
                DO JJ = 1,cachednpr
                   S2 = REALCACHE(S2_MEMBER,JJ)
                   KVECT = INTCACHE(K_MEMBER,JJ)
                   J = ABS(KVECT)
                   !---                Ewald real space sum using the error function
                   R1S = one/SQRT(S2)
                   RS = R1S*s2
                   !---call ERFCD( RS, KAPPA, ERFCX, DRFC, EWLDT, -1 )
                   !----1 means lookup table method using cubic spline interpolation
                   !      !---Inlined erfc calculation for speed (from ERFCD).
                   XVAL = RS*EKAPPA
                   IXVAL = XVAL+HALF
                   REM = XVAL-IXVAL
                   IXVAL=IXVAL+2
                   IXVAL = MIN(IXVAL,EWNPTS-1)
                   VAL0 = EWLDT(IXVAL-1)
                   VAL1 = EWLDT(IXVAL)
                   VAL2 = EWLDT(IXVAL+1)
                   D1 = (VAL0-VAL2)*HALF
                   D2 = (VAL1+VAL1-VAL0-VAL2)*REM
                   ERFCX = VAL1-(D1+HALF*D2)*REM
                   DRFC = (D1+D2)*EKAPPA
                   !      !---end of erfc inlined code.
                   !
                   E14F = ZERO
#if KEY_CHEQ==1
                   if(qcg) E14FCQ = ZERO
#endif
                   IF(KVECT < 0) THEN
                      E14F = E14FM1
#if KEY_CHEQ==1
                      if(qcg) E14FCQ = -1.0
#endif
                   ENDIF
#if KEY_CHEQ==1
                   CH = CGF
                   if(qcg) then
                      IF( (QPARTBIN(I) /= 0).and.(QPARTBIN(J)/=0)) THEN
                         ENE = CH*(ERFCX + E14FCQ)*R1S
                      ELSE
                         ENE = CH*(ERFCX + E14F)*R1S
                      ENDIF
                   else
                      ENE = CH*(ERFCX + E14F)*R1S
                   endif
                   IF (QCG) THEN
                      IF ( (CGX(I) /= ZERO).AND.(CGX(J)/=ZERO) ) THEN
                         DCHI = DCHI + ENE*CGX(J)
                         DCH(J) = DCH(J) + ENE*CGX(I)
                      ELSE                          ! For TIP4PFQ type cases
                         DCHI = DCHI + ZERO
                      ENDIF                         ! For TIP4PFQ type cases
                   ENDIF
                   ENE = ENE*CGX(I)*CGX(J)
                   CH = CH*CGX(I)*CGX(J)
#else /*            */
                   CH = CGT*CGX(J)
                   ENE = CH*(ERFCX + E14F)*R1S
#endif

                   !av_080628
                   ! Block code: H Kamberaj November 2007
                   !
#if KEY_BLOCK==1 /*(block)*/
                   IF (QBLOCK) THEN
                      ibl = iblckp(i)
                      jbl = iblckp(j)
                      if (jbl  <  ibl) then
                         kkb = ibl
                         ibl = jbl
                         jbl = kkb
                      endif
                      kkb = ibl + jbl*(jbl-1)/2
                      coef = blcoee(kkb)
                      ENE = ENE * coef
                      ch = ch * coef
                   ENDIF
#endif /*    (block)*/
                   !av_080628

                   EEL = EEL+ENE
                   REALCACHE(TF_MEMBER,JJ) = -(CH*DRFC + ENE)
                   !.ab.
#if KEY_BLOCK==1
                   if(qhybh) realcache(ene_member,jj) = ene
#endif
                   !.ab.
#if KEY_FLUCQ==1
                   IF (QFLUC) THEN
                      FQCFOR(I) = FQCFOR(I)+ENE
                      FQCFOR(J) = FQCFOR(J)+ENE
                   ENDIF
#endif
                ENDDO
                !---VDW loop: compute van der Waals interaction and update forces
                !.ab.
#if KEY_BLOCK==1
                if(.not.qhybh) then
#endif
                   enbtmp=zero
                   DO JJ = 1,CACHEDNPR
                      S2 = REALCACHE(S2_MEMBER,JJ)
                      KVECT = INTCACHE(K_MEMBER,JJ)
                      J = ABS(KVECT)
                      IVECT = LOWTP(MAX(IACNB(J),IACI))+IACNB(J)+IACI
                      IF(KVECT < 0) THEN
                         IVECT = IVECT+NITCC2
                      ENDIF
                      TR2 = ONE/S2
                      TR6 = TR2*TR2*TR2
                      CA = CCNBA(IVECT)*TR6
                      !av_080628                  ENN = (CA-CCNBB(IVECT))*TR6
                      ! Block code: H Kamberaj November 2007
                      !
                      CB=CCNBB(IVECT)                   !yw_080813
#if KEY_BLOCK==1 /*(block)*/
                      IF (QBLOCK) THEN
                         ibl = iblckp(i)
                         jbl = iblckp(j)
                         if (jbl  <  ibl) then
                            kkb = ibl
                            ibl = jbl
                            jbl = kkb
                         endif
                         kkb = ibl + jbl*(jbl-1)/2
                         coef = blcoev(kkb)
                         ca = ca*coef
                         cb = cb*coef
                      ENDIF
#endif /*    (block)*/
                      !
                      ENN = (CA-CB)*TR6
                      !av_080628
                      TF = -SIX*(CA*TR6+ENN)
                      !---                ELSE IF(S2 > C2ONNB) THEN
                      !---                van der Waals switch function
                      !---                   RIJL = C2ONNB-S2
                      !---                   RIJU = C2OFNB-S2
                      !---To produce pipelined code the S2 > C2ONNB if was transformed into
                      !---these min and max statements.
                      RIJL = min(ZERO,C2ONNB-S2)
                      RIJU = C2OFNB-max(S2,C2ONNB)
                      !---When S2 is greater than C2ONNB then RIJL = C2ONNB - S2 and
                      !---RIJU = C2OFNB - S2 and the switch function is calculated as in
                      !---the untransformed source code.
                      !---When S2 is not greater than C2ONNB then RIJL = ZERO
                      !---and RIJU = C2OFNB - C2ONNB;
                      FSW = RIJU*RIJU*(RIJU-THREE*RIJL)*RUL3
                      !---FSW will then equal ONE within the error of finite precision floating
                      !---point arithmetic;
                      TF = S2*ENN*RIJL*RIJU*RUL12+TF*FSW
                      !mc...060712                  ENN = ENN*FSW
                      !---consequently, TF will then equal ZERO + TF * ONE and ENN will then
                      !---equal ENN * ONE.  In other words, TF and ENN will remain un-switched
                      !---within the error of finite precision floating point arithmetic.
                      ENBtmp = ENBtmp+ENN*fsw
                      TF = (REALCACHE(TF_MEMBER,JJ) + TF) * TR2
                      !---                update forces
                      XFORCE = REALCACHE(TX_MEMBER,JJ) * TF
                      YFORCE = REALCACHE(TY_MEMBER,JJ) * TF
                      ZFORCE = REALCACHE(TZ_MEMBER,JJ) * TF
                      DTX = DTX+XFORCE
                      DTY = DTY+YFORCE
                      DTZ = DTZ+ZFORCE
                      DX(J) = DX(J)-XFORCE
                      DY(J) = DY(J)-YFORCE
                      DZ(J) = DZ(J)-ZFORCE
                   ENDDO
                   enb=enb + enbtmp
                   !.ab.HybH code.QHYBH=.TRUE.
#if KEY_BLOCK==1
                else
                   enbtmp=zero
                   do jj = 1,cachednpr
                      s2 = realcache(s2_member,jj)
                      kvect = intcache(k_member,jj)
                      j = abs(kvect)
                      ivect = lowtp(max(iacnb(j),iaci))+iacnb(j)+iaci
                      if(kvect < 0) then
                         ivect = ivect+nitcc2
                      endif
                      tr2 = one/s2
                      tr6 = tr2*tr2*tr2
                      ca = ccnba(ivect)*tr6
                      enn = (ca-ccnbb(ivect))*tr6
                      tf = -six*(ca*tr6+enn)
                      !.ab.Remove comments about switch...See above.
                      rijl = min(zero,c2onnb-s2)
                      riju = c2ofnb-max(s2,c2onnb)
                      fsw = riju*riju*(riju-three*rijl)*rul3
                      !.ab TF * TR2
                      tf = s2*enn*rijl*riju*rul12+tf*fsw * tr2
                      enbtmp = enbtmp+enn*fsw
                      !.ab.
                      !                  TF = (REALCACHE(TF_MEMBER,JJ) + TF) * TR2
                      !.ab.QHYBH code. TF = dvdW/|r|.dr
                      kk=intcache(kb_member,jj)
                      qout=.true.
                      if(kk < 0) then
                         qout=.false.
                         kk=-kk
                      endif
                      tfe=realcache(tf_member,jj) * tr2
                      if (kk == 1) then
                         tf=tf + tfe
                      else if (kk < 4) then
                         ene=realcache(ene_member,jj)
                         dner=dner+ene*dfr
                         dnnr=dnnr+enn*dfr*fsw
                         if (qout) then
                            tf=tf+tfe
                            tf=tf*fr
                         else
                            r02=realcache(r02_member,jj)
                            dner=dner+fr*tfe*r02*dl112r
                            dnnr=dnnr+fr*tf *r02*dl112r
                            tf=zero
                         endif
                         ene=ene*fr
                         enn=enn*fr*fsw
                      else
                         ene=realcache(ene_member,jj)
                         dnep=dnep+ene*dfp
                         dnnp=dnnp+enn*dfp*fsw
                         if (qout) then
                            tf=tf+tfe
                            tf=tf*fp
                         else
                            dnep=dnep+fp*tfe*r02*dl112p
                            dnnp=dnnp+fp*tf *r02*dl112p
                            tf=zero
                         endif
                         ene=ene*fp
                         enn=enn*fp*fsw
                      endif
                      !.ab.

                      !---                update forces
                      xforce = realcache(tx_member,jj) * tf
                      yforce = realcache(ty_member,jj) * tf
                      zforce = realcache(tz_member,jj) * tf
                      dtx = dtx+xforce
                      dty = dty+yforce
                      dtz = dtz+zforce
                      dx(j) = dx(j)-xforce
                      dy(j) = dy(j)-yforce
                      dz(j) = dz(j)-zforce
                   enddo
                   enb=enb + enbtmp
                endif
#endif
                !.ab.
                !
                !---             restore the i-th component of the force
                DX(I) = DX(I)+DTX
                DY(I) = DY(I)+DTY
                DZ(I) = DZ(I)+DTZ
#if KEY_CHEQ==1
                IF (QCG) DCH(I) = DCH(I) + DCHI
#endif
             ENDIF
             ITEMP = INBL(I)
          ENDDO
       ENDIF lvshft_blk
    ELSE             ! End of case 2 VDW Switch (if(lvdwx) then... ?)
       !
       !----------------------------------------------------------------------
       !---                  No VDW.
       !----------------------------------------------------------------------
       !---              Case 3 of 3.  (.not. LVDWX)
       !
       !.ab.HybH. does not make sens without vdw (+I am tried...)
#if KEY_BLOCK==1
       if(qhybh) call wrndie(-5,'<REWALD>',                     &
#endif
#if KEY_BLOCK==1
            'HYBH without vdw is not implemented, see code.')
#endif
       DO I = 1,NATOMX
#if KEY_IMCUBES==1
          IF(LBYCBIM) ITEMP = INBL(I+NATOMX)
#endif
          NPR = INBL(I) - ITEMP
          IF(NPR > 0) THEN
#if KEY_CHEQ==1
             DCHI = ZERO
#else /**/
             CGT = CGX(I)*CGF
#endif
             IACI = IACNB(I)
             CRXI = X(I)
             CRYI = Y(I)
             CRZI = Z(I)
             DTX = ZERO
             DTY = ZERO
             DTZ = ZERO
             !---prologue loop: compute and conditionally cache delta r**2
             cachednpr = 0
             DO JJ = 1,NPR
                KVECT = JNBL(ITEMP+JJ)
                J = ABS(KVECT)
                !
                !---namkh 01/20/04
                !---This also possible to be screwed up, but hope not.
#if KEY_GAMESS==1 || KEY_GAMESSUK==1 || KEY_MNDO97==1 || KEY_SQUANTM==1 || \
    KEY_QCHEM==1 || KEY_QTURBO==1 || KEY_G09==1
                if(qmused)then
                IF((ABS(IGMSEL(I)) == 1.OR.ABS(IGMSEL(I)).EQ.2) .AND. &
                     (ABS(IGMSEL(J)) == 1.OR.ABS(IGMSEL(J)).EQ.2)) &
                     GOTO 400
                endif
#endif
                !
                TX = CRXI-X(J)
                TY = CRYI-Y(J)
                TZ = CRZI-Z(J)
#if KEY_PBOUND==1
                If(qBoun) then
                   If(qCUBoun.or.qTOBoun) then
                      tx = BOXINV * tx
                      ty = BOYINV * ty
                      tz = BOZINV * tz
                      tx = tx - nint(tx)
                      ty = ty - nint(ty)
                      tz = tz - nint(tz)
                      If (qTOBoun) Then
                         CORR = HALF * AINT ( R75 * (ABS(TX) + &
                              ABS(TY) + &
                              ABS(TZ)))
                         TX = TX - SIGN( CORR,  TX  )
                         TY = TY - SIGN( CORR,  TY  )
                         TZ = TZ - SIGN( CORR,  TZ  )
                      Endif
                      TX = XSIZE * TX
                      TY = YSIZE * TY
                      TZ = ZSIZE * TZ
                   Else
                      Call PBMove(TX, TY, TZ)
                   Endif
                Endif
#endif
                S2 = MAX(RSMALL,TX*TX+TY*TY+TZ*TZ)
                IF (S2 < C2OFNB) THEN
                   cachednpr = cachednpr + 1
                   INTCACHE(K_MEMBER,cachednpr) = KVECT
                   REALCACHE(S2_MEMBER,cachednpr) = S2
                   REALCACHE(TX_MEMBER,cachednpr) = TX
                   REALCACHE(TY_MEMBER,cachednpr) = TY
                   REALCACHE(TZ_MEMBER,cachednpr) = TZ
                ENDIF
400             CONTINUE
             ENDDO
             !---electrostatic loop: compute direct Ewald sum and update forces
             !---VDW loop: merged here because No van der Waals interaction
             DO JJ = 1,cachednpr
                S2 = REALCACHE(S2_MEMBER,JJ)
                KVECT = INTCACHE(K_MEMBER,JJ)
                J = ABS(KVECT)
                !---                Ewald real space sum using the error function
                R1S = ONE/SQRT(S2)
                RS = S2*R1S
                !
                !---call ERFCD( RS, KAPPA, ERFCX, DRFC, EWLDT, -1 )
                !----1 means lookup table method using cubic spline interpolation
                !      !---Inlined erfc calculation for speed (from ERFCD).
                XVAL = RS*EKAPPA
                IXVAL = XVAL+HALF
                REM = XVAL-IXVAL
                IXVAL=IXVAL+2
                IXVAL = MIN(IXVAL,EWNPTS-1)
                VAL0 = EWLDT(IXVAL-1)
                VAL1 = EWLDT(IXVAL)
                VAL2 = EWLDT(IXVAL+1)
                D1 = (VAL0-VAL2)*HALF
                D2 = (VAL1+VAL1-VAL0-VAL2)*REM
                ERFCX = VAL1-(D1+HALF*D2)*REM
                DRFC = (D1+D2)*EKAPPA
                !      !--- end of erfc inlined code.
                !
                E14F = ZERO
#if KEY_CHEQ==1
                E14FCQ = ZERO
#endif
                IF(KVECT < 0) THEN
                   E14F = E14FM1
#if KEY_CHEQ==1
                   E14FCQ = -1.0
#endif
                ENDIF
#if KEY_CHEQ==1
                CH = CGF
                if(qcg) then
                   IF ( (QPARTBIN(I) /= 0).and.(QPARTBIN(J).ne.0) ) THEN
                      ENE = CH*(ERFCX + E14FCQ)*R1S
                   ELSE
                      ENE = CH*(ERFCX + E14F)*R1S
                   ENDIF
                else
                   ENE = CH*(ERFCX + E14F)*R1S
                endif
                IF (QCG) THEN
                   IF ( (CGX(I) /= ZERO).AND.(CGX(J).NE.ZERO) ) THEN
                      DCHI = DCHI + ENE*CGX(J)
                      DCH(J) = DCH(J) + ENE*CGX(I)
                   ELSE                          ! For TIP4PFQ type cases
                      DCHI = DCHI + ZERO
                   ENDIF                         ! For TIP4PFQ type cases
                ENDIF
                ENE = ENE*CGX(I)*CGX(J)
                CH = CH*CGX(I)*CGX(J)
#else /*            */
                CH = CGT*CGX(J)
                ENE = CH*(ERFCX + E14F)*R1S
#endif

                !av_080628
                ! Block code: H Kamberaj November 2007
                !
#if KEY_BLOCK==1 /*(block)*/
                IF (QBLOCK) THEN
                   ibl = iblckp(i)
                   jbl = iblckp(j)
                   if (jbl  <  ibl) then
                      kkb = ibl
                      ibl = jbl
                      jbl = kkb
                   endif
                   kkb = ibl + jbl*(jbl-1)/2
                   coef = blcoee(kkb)
                   ENE = ENE * coef
                   CH = CH * coef
                ENDIF
#endif /*    (block) */
                !av_080628

                EEL = EEL+ENE
                TR2 = R1S*R1S
                TF = -(CH*DRFC + ENE) * TR2
#if KEY_FLUCQ==1
                IF (QFLUC) THEN
                   FQCFOR(I) = FQCFOR(I)+ENE
                   FQCFOR(J) = FQCFOR(J)+ENE
                ENDIF
#endif
                !---                 update forces
                XFORCE = REALCACHE(TX_MEMBER,JJ) * TF
                YFORCE = REALCACHE(TY_MEMBER,JJ) * TF
                ZFORCE = REALCACHE(TZ_MEMBER,JJ) * TF
                DTX = DTX+XFORCE
                DTY = DTY+YFORCE
                DTZ = DTZ+ZFORCE
                DX(J) = DX(J)-XFORCE
                DY(J) = DY(J)-YFORCE
                DZ(J) = DZ(J)-ZFORCE
             ENDDO
             !
             !---              restore the i-th component of the force
             DX(I) = DX(I)+DTX
             DY(I) = DY(I)+DTY
             DZ(I) = DZ(I)+DTZ
#if KEY_CHEQ==1
             IF (QCG) DCH(I) = DCH(I) + DCHI
#endif
          ENDIF
          ITEMP = INBL(I)
       ENDDO
    ENDIF      !------------- End of Case 3 no switch or shift ------------
    !.ab.Save HybH dH/dlamda
#if KEY_BLOCK==1
    if (qhybh) then
       call sumhyb(ihybh,dnnr,dnnp)
       call sumhyb(ihybh+1,dner,dnep)
    endif
#endif
    !.ab.
#endif /* (fast_ewald)*/
    !
    RETURN
  END subroutine rewald

#endif /* (mfc_fast_ewald)*/


  SUBROUTINE EWLDEX(ELEC,NATOM,IBLO14,INB14,NNB14,CG, &
       CTONNB,CTOFNB,EPS,DX,DY,DZ,X,Y,Z,QECONTX,ECONTX &
#if KEY_FLUCQ==1
       ,QFLUC,FQCFOR    &
#endif
       )
    !
    !---     Does Non-bond exclusion corrections for Ewald and PME.
    !
    !---     By Stephen H. Fleischman 8 Oct., 1991
    !
    !.ab.-------------------------------------------------------------------
    !     AB's Block version. Helec = Helec((1-L)*ch(r)+L*ch(p))
    !     This is to make kspace faster since dHewald/dL can be
    !     calculated in one cycle.
    !-----------------------------------------------------------------------

    use erfcd_mod
#if KEY_CHEQ==1
    use cheq,only:qcg,   &
#endif
#if KEY_CHEQ==1
         DCH,SUMDCH,DDCH
#endif

    use consta
    use econtmod
    use parallel
    use stream
    !--- namkh 01/20/04
#if KEY_GAMESS==1 || KEY_GAMESSUK==1 || KEY_MNDO97==1 || KEY_SQUANTM==1 || \
    KEY_QCHEM==1 || KEY_QTURBO==1 || KEY_G09==1
    use gamess_fcm
#endif
    !
#if KEY_BLOCK==1
    use block_fcm
#endif
    use chutil,only:atomid

    !--- input/output
    REAL(KIND=CHM_REAL) ELEC
    INTEGER NATOM
    INTEGER IBLO14(*),NNB14
    INTEGER INB14(*)
    REAL(KIND=CHM_REAL) CG(*)
    REAL(KIND=CHM_REAL) CTONNB, CTOFNB, EPS
    REAL(KIND=CHM_REAL) DX(*),DY(*),DZ(*)
    REAL(KIND=CHM_REAL) X(*),Y(*),Z(*)
    LOGICAL QECONTX
    REAL(KIND=CHM_REAL) ECONTX(*)
#if KEY_FLUCQ==1
    LOGICAL QFLUC
    REAL(KIND=CHM_REAL) FQCFOR(*)
#endif
    !
#if KEY_CHEQ==1
    REAL(KIND=CHM_REAL) DCHI,HIJ
#endif
    !
    !--- local
    INTEGER JND
    REAL(KIND=CHM_REAL) CGIL,CGJL
    REAL(KIND=CHM_REAL) CGF
    REAL(KIND=CHM_REAL) C2OFNB
    INTEGER I, J
    REAL(KIND=CHM_REAL) REWALD
    REAL(KIND=CHM_REAL) DF, R2, S
    REAL(KIND=CHM_REAL) DXI, DYI, DZI, DELTX, DELTY, DELTZ
    INTEGER NXI,NXIMAX
    !---     for ewald summation
    REAL(KIND=CHM_REAL) RS,R1S,ERF2,ERFCX,DRFC
    REAL(KIND=CHM_REAL) XVAL,REM,VAL0,VAL1,VAL2,D1,D2
    INTEGER IXVAL
    INTEGER II,KK
    LOGICAL QAFIRST
    CHARACTER(len=8) SIDDNI,RIDDNI,RESDNI,ACDNI,SIDDNJ,RIDDNJ,RESDNJ,ACDNJ
    !
    REAL(KIND=CHM_REAL), PARAMETER :: RSQPI2 = TWO*5.6418958354775628695D-1
    !
    !.ab.
#if KEY_BLOCK==1
    real(chm_real) dxr,dxp,tmpe(3),chf(3)
    integer ibl,jbl
#endif
    !.ab.
    !av_080628 Block code: H Kamberaj November 2007
#if KEY_BLOCK==1
    REAL(kind=chm_real)   :: COEF
    INTEGER               :: KKB
#endif
    integer istart,iend,iadd
    !av_080628

    !--- begin
#if KEY_CHEQ==1
    !--- If QCG=.TRUE. ewald correction will be done in EWLDEXQ
    !---      IF (QCG) RETURN
#endif
    QAFIRST = .TRUE.
    IF(NNB14 <= 0) RETURN
    CGF = CCELEC/EPS
    C2OFNB = CTOFNB*CTOFNB

    !.ab..Initialyse block and energy array.
#if KEY_BLOCK==1
    if (qhybh) then
       if(nblock  /=  3) call wrndie(-3,'<EWLDEX>', &
            'Must use three blocks for this option.')
       dxr=zero
       dxp=zero
       chf(1)=one
       chf(2)=one-hybhlb
       chf(3)=hybhlb
    endif
#endif
    !.ab.
    !
    !---     main loop over atom I

    istart = mynodp
    iend = natom
    iadd = numnod

    do i = istart,iend,iadd

       DXI = ZERO
       DYI = ZERO
       DZI = ZERO
#if KEY_CHEQ==1
       DCHI= ZERO
#endif

       !.ab..Initialyse block and energy array.
#if KEY_BLOCK==1
       if (qhybh) then
          ibl=iblckp(i)
          !.ab.See above: Note about HYBH scheme.
          !....It is not clear what should be done... Exclustion of 2-3 block
          !....interactions or not. I guess 2-3 terms should be small anyway....
          tmpe(1)=chf(ibl)
          if (ibl == 1) then
             tmpe(2)=chf(2)
             tmpe(3)=chf(3)
          else
             tmpe(2)=.0
             tmpe(3)=.0
             tmpe(ibl)=chf(ibl)*chf(ibl)
          endif
       endif
#endif
       !.ab.

       CGIL = CG(I)*CGF
       IF (I > 1) THEN
          NXI = IBLO14(I-1)+1
       ELSE
          NXI = 1
       ENDIF
       NXIMAX = IBLO14(I)
       !---     do exclusions and 1,4 interactions
       DO J = NXI,NXIMAX
          IF(INB14(J) > 0) THEN
             JND = INB14(J)
             !
             !--- namkh 01/20/04
#if KEY_GAMESS==1 || KEY_GAMESSUK==1 || KEY_MNDO97==1 || KEY_SQUANTM==1 || \
    KEY_QCHEM==1 || KEY_QTURBO==1 || KEY_G09==1
             if(qmused)then
             IF((ABS(IGMSEL(I)) == 1.OR.ABS(IGMSEL(I)).EQ.2) .AND. &
                  (ABS(IGMSEL(JND)) == 1.OR.ABS(IGMSEL(JND)).EQ.2)) &
                  GOTO 100
             endif
#endif
             !
             !---        non-bond exclusion
             !.ab.
#if KEY_BLOCK==1
             if (qhybh) jbl=iblckp(jnd)
#endif
             CGJL = CG(JND)
             DELTX = X(I)-X(JND)
             DELTY = Y(I)-Y(JND)
             DELTZ = Z(I)-Z(JND)
             S = DELTX**2+DELTY**2+DELTZ**2
             RS = SQRT(S)
             IF(S <= RSMALL) THEN
                REWALD = -CGIL*CGJL*KAPPA*RSQPI2

                !av_080628 Block code: H Kamberaj November 2007
#if KEY_BLOCK==1 /*(block)*/
                IF (QBLOCK) THEN
                   ibl = iblckp(i)
                   jbl = iblckp(jnd)
                   if (jbl  <  ibl) then
                      kkb = ibl
                      ibl = jbl
                      jbl = kkb
                   endif
                   kkb = ibl + jbl*(jbl-1)/2
                   coef = blcoee(kkb)
                   REWALD = REWALD * coef
                ENDIF
#endif /*    (block)*/
                !av_080628

                !.ab.do.later              ELEC = ELEC+REWALD
#if KEY_CHEQ==1
                IF (QCG) THEN
                   HIJ=-CGF*KAPPA*RSQPI2
                ENDIF
#endif
                DF = ZERO
             ELSE
                R2 = ONE/S
                R1S = ONE/RS
                CALL ERFCD(RS,KAPPA,ERFCX,DRFC,ERFMOD)
                ERF2 = ONE-ERFCX
                REWALD = -CGIL*CGJL*ERF2*R1S

                !av_080628 Block code: H Kamberaj November 2007
                DF = CGIL*CGJL*(-DRFC*R1S + ERF2*R2)*R1S
#if KEY_BLOCK==1 /*(block)*/
                IF (QBLOCK) THEN
                   ibl = iblckp(i)
                   jbl = iblckp(jnd)
                   if (jbl  <  ibl) then
                      kkb = ibl
                      ibl = jbl
                      jbl = kkb
                   endif
                   kkb = ibl + jbl*(jbl-1)/2
                   coef = blcoee(kkb)
                   REWALD = REWALD * coef
                   DF = DF * coef
                ENDIF
#endif /*   (block)*/
                !av_080628

                !.ab.do.later              ELEC = ELEC+REWALD
#if KEY_CHEQ==1
                IF (QCG) THEN
                   HIJ=-CGF*ERF2*R1S
                ENDIF
#endif
                !av_080628              DF = CGIL*CGJL*(-DRFC*R1S + ERF2*R2)*R1S
             ENDIF

             !.ab.DF ok. now accumulate for dH/dL.
#if KEY_BLOCK==1
             if(qhybh)then
                if (ibl == 1) then
                   if (jbl == 2) dxr=dxr-rewald
                   if (jbl == 3) dxp=dxp+rewald
                else if (ibl == 2) then
                   if (jbl == 1) dxr=dxr-rewald
                   if (jbl == 2) dxr=dxr-two*tmpe(1)*rewald
                   !     IF (JBL == 3) write(6,*) '2-3 block excl',I,JND,REWALD
                else
                   if (jbl == 1) dxp=dxp+rewald
                   !     IF (JBL == 2) write(6,*) '2-3 block excl',I,JND,REWALD
                   if (jbl == 3) dxp=dxp+two*tmpe(1)*rewald
                endif
                rewald=rewald*tmpe(jbl)
                df=df*tmpe(jbl)
             endif
#endif
             elec = elec+rewald
             !.ab.
             !
             !CC##IF analys (analys)
             IF(QATERM) THEN
                IF(QANBND) THEN
                   KK = ANSLCT(I)+ANSLCT(JND)
                   IF(KK == 2 .OR. (KK >= 1 .AND. .NOT.QAONLY)) THEN
                      IF(QAUNIT < 0) THEN
                         II = OUTU
                      ELSE
                         II = QAUNIT
                      ENDIF
                      !---
                      IF(PRNLEV >= 5) THEN
                         IF(QAFIRST) THEN
                            IF(QLONGL) THEN
                               WRITE(II,243)
                            ELSE
                               WRITE(II,244)
                            ENDIF
                            QAFIRST = .FALSE.
                         ENDIF
                         CALL ATOMID(I,SIDDNI,RIDDNI,RESDNI,ACDNI)
                         CALL ATOMID(JND,SIDDNJ,RIDDNJ,RESDNJ,ACDNJ)
                         IF(QLONGL) THEN
                            WRITE(II,245) J,I,SIDDNI(1:idleng), &
                                 RIDDNI(1:idleng),RESDNI(1:idleng), &
                                 ACDNI(1:idleng), &
                                 JND,SIDDNJ(1:idleng),RIDDNJ(1:idleng), &
                                 RESDNJ(1:idleng),ACDNJ(1:idleng), &
                                 RS,REWALD,DF,CG(I),CG(JND)
                         ELSE
                            WRITE(II,246) J,I,SIDDNI(1:idleng), &
                                 RIDDNI(1:idleng),RESDNI(1:idleng), &
                                 ACDNI(1:idleng), &
                                 JND,SIDDNJ(1:idleng),RIDDNJ(1:idleng), &
                                 RESDNJ(1:idleng),ACDNJ(1:idleng), &
                                 RS,REWALD,DF,CG(I),CG(JND)
                         ENDIF
                      ENDIF
                   ENDIF
                ENDIF
             ENDIF
             IF(QECONTX) THEN
                S = HALF*REWALD
                ECONTX(I) = ECONTX(I)+S
                ECONTX(JND) = ECONTX(JND)+S
             ENDIF
             !CC##ENDIF (analys)
             !
             DELTX = DELTX*DF
             DELTY = DELTY*DF
             DELTZ = DELTZ*DF
             !
             DX(JND) = DX(JND)-DELTX
             DY(JND) = DY(JND)-DELTY
             DZ(JND) = DZ(JND)-DELTZ
#if KEY_FLUCQ==1
             IF (QFLUC) THEN
                FQCFOR(JND) = FQCFOR(JND)+REWALD
                FQCFOR(I) = FQCFOR(I)+REWALD
             ENDIF
#endif
             DXI = DXI + DELTX
             DYI = DYI + DELTY
             DZI = DZI + DELTZ
#if KEY_CHEQ==1
             IF(QCG) THEN
                IF ( (CG(I) /= 0.0).AND.(CG(JND).NE.0.0) ) THEN
                   DCH(JND)= DCH(JND) + HIJ*CG(I)
                   DCH(I)  = DCH(I)   + HIJ*CG(JND)
                ENDIF
             ENDIF
#endif
          ENDIF
100       CONTINUE
       ENDDO
       !
       DX(I) = DX(I) + DXI
       DY(I) = DY(I) + DYI
       DZ(I) = DZ(I) + DZI
    ENDDO

    !.ab.Wrap up HYBH.(test 46=EWEXCL)
#if KEY_BLOCK==1
    IF(QHYBH) IHYBH=46
#endif
#if KEY_BLOCK==1
    IF(QHYBH) CALL SUMHYB(IHYBH,DXR,DXP)
#endif
    !      write(6,*) 'Exit EWLDEXB, ELEC=',ELEC
    !.ab.
    !
    !--- Format statements
243 FORMAT('ANAL: EWLDEX:    Index        Atom-I', &
         '                   Atom-J          ', &
         '        Dist     ', &
         '      EELEC         Force          Charges')
244 FORMAT('ANAL: EWLDEX:    Index        Atom-I', &
         '                   Atom-J          ',/ &
         '        Dist      ', &
         '      EELEC         Force          Charges')
245 FORMAT('ANAL: EWLDEX>',I10,I5,4(1X,A),I5,4(1X,A),5F15.6)
246 FORMAT('ANAL: EWLDEX>',I10,I5,4(1X,A),I5,4(1X,A),/5F15.6)
    !
    RETURN
  END SUBROUTINE EWLDEX

#if KEY_LJPME==1
  subroutine ljpme_dir(c6a, c6b, r2, e, df, NBA, NBB, add_shift)
    use pmeutil, only: dopcut, r6_inv_cut, dkappa2, dkappa4, dkappa6
    real(chm_real), intent(in) :: r2, c6a, c6b, NBA, NBB
    real(chm_real), intent(in out) :: e, df
    logical, intent(in) :: add_shift
    !
    integer i
    real(chm_real) :: pair_f(3)
    real(chm_real) :: e6, f6, EXPTERM, taco(3), r4, r6, r8
    real(chm_real) :: kr2, kr4, kr6, eterm, dterm
    real(chm_real) :: c6_add, c6_mult
    real(chm_real) :: e_add, f_add, e_mult, f_mult, sig6
    !
    ! adj multiplicative part
    r4 = r2 * r2
    r6 = r4 * r2
    r8 = r6 * r2
    !
    kr2 = dkappa2 * r2
    kr4 = dkappa4 * r4
    kr6 = dkappa6 * r6
    EXPTERM = EXP(-kr2)
    eterm = 1d0 + kr2 + 5d-1 * kr4
    e6 = (1d0 - EXPTERM * eterm) * c6a * c6b / r6
    dterm = eterm + kr6/6d0
    f6 = (-1d0 + EXPTERM * dterm) * c6a * c6b * 6d0 / r8

    e = e + e6
    df = df + f6
    ! potential/force shift
    !
    if(add_shift) then
        ! C6 additive
        sig6 = NBA * NBA * NBA
        c6_add = sig6 * NBB * (2.0d0 - sig6*r6_inv_cut) ! Add the r^12 shift in too
        e_add = c6_add * r6_inv_cut
        ! C6 multi
        c6_mult = c6a * c6b
        !
        e_mult = -c6_mult * DOPCUT * r6_inv_cut
        e = e + e_add + e_mult ! constant energy shift term, ensures continuity of potential
    endif
  endsubroutine ljpme_dir

  SUBROUTINE LJPMEEX(natom,IBLO14,INB14,NNB14,CTONNB,CTOFNB,DX,DY,DZ,X,Y,Z)
    !
    !---     Does Non-bond exclusion corrections for LJ-PME.
    !---     Andy Simmonett (03/18)
    !
    !-----------------------------------------------------------------------

    use consta
    use econtmod
    use parallel
    use stream
    !
    use energym, only: ETERM, LJEXC
    use pmeutil, only: DKAPPA, QLJPME
    use psf, only: ljc6, IAC
    use param, only:  ITC, EFF, VDWR
    use chutil,only:atomid

    !--- input/output
    INTEGER NATOM
    INTEGER IBLO14(*),NNB14
    INTEGER INB14(*)
    REAL(KIND=CHM_REAL) CTONNB, CTOFNB
    REAL(KIND=CHM_REAL) DX(*),DY(*),DZ(*)
    REAL(KIND=CHM_REAL) X(*),Y(*),Z(*)
    !
    !--- local
    INTEGER JND
    REAL(KIND=CHM_REAL) C2OFNB
    INTEGER I, J
    REAL(KIND=CHM_REAL) DF, R2, S
    REAL(KIND=CHM_REAL) DELTX, DELTY, DELTZ
    INTEGER NXI,NXIMAX
    !---     for ewald summation
    REAL(KIND=CHM_REAL) RS,R1S, ENBPR, c6i, c6j
    INTEGER II,KK, atom_i, atom_j, maxij, minij
    !
    integer istart,iend,iadd, ptr

    IF(NNB14 <= 0) RETURN
    C2OFNB = CTOFNB*CTOFNB

    istart = mynodp
    iend = natom
    iadd = numnod

    do i = istart,iend,iadd
       atom_i = ITC(IAC(i))
       c6i = ljc6(i)
       IF (I > 1) THEN
          NXI = IBLO14(I-1)+1
       ELSE
          NXI = 1
       ENDIF
       NXIMAX = IBLO14(I)
       !---     do exclusions and 1,4 interactions
       DO J = NXI,NXIMAX
          IF(INB14(J) > 0) THEN
             JND = INB14(J)
             atom_j = ITC(IAC(jnd))
             c6j = ljc6(jnd)
             DELTX = X(I)-X(JND)
             DELTY = Y(I)-Y(JND)
             DELTZ = Z(I)-Z(JND)
             R2 = DELTX**2+DELTY**2+DELTZ**2
             IF(R2 > RSMALL) THEN
                ! This is an exclusion so we need to remove the reciprocal space term
                ! but we don't need to add the potential shift.  The forces will be
                ! accumulated into the right place within this function
                DF=0d0
                minij = min(atom_i, atom_j)
                maxij = max(atom_i, atom_j)
                ptr = maxij*(maxij+1)/2 + minij
                call ljpme_dir(c6i, c6j, r2, eterm(LJEXC), df, eff(ptr), vdwr(ptr), .false.)
                DELTX = DELTX*DF
                DELTY = DELTY*DF
                DELTZ = DELTZ*DF
                DX(I) = DX(I)+DELTX
                DY(I) = DY(I)+DELTY
                DZ(I) = DZ(I)+DELTZ
                DX(JND) = DX(JND)-DELTX
                DY(JND) = DY(JND)-DELTY
                DZ(JND) = DZ(JND)-DELTZ
             ENDIF
          ENDIF
100       CONTINUE
       ENDDO
    ENDDO
    RETURN
  END SUBROUTINE LJPMEEX
#endif /*LJPME*/

#if KEY_CHEQ==1 /*cheq_ewldex*/

  subroutine ewldexq(elec,natom,cg, &
       ctonnb,ctofnb,eps,dx,dy,dz,x,y,z,qecontx,econtx,ewldtq)
    !---     vectorized non-bonding routine
    !
    !---     features:
    !---     does connected corrections for ewald.
    !

    use erfcd_mod
#if KEY_CHEQ==1
    use cheq,only: qcg,molt,molbl,molptr,   &
#endif
#if KEY_CHEQ==1
         DCH,SUMDCH,DDCH
#endif

    use consta,only:ccelec
    use econtmod
    use parallel
    use stream
    !--- namkh 01/20/04
#if KEY_GAMESS==1 || KEY_GAMESSUK==1 || KEY_MNDO97==1 || KEY_SQUANTM==1 || \
    KEY_QCHEM==1 || KEY_QTURBO==1 || KEY_G09==1
    use gamess_fcm
#endif
#if KEY_BLOCK==1
    use block_fcm
#endif
    use chutil,only:atomid

    !--- input/output
    real(kind=chm_real) elec
    integer natom
    real(kind=chm_real) cg(*)
    real(kind=chm_real) ctonnb, ctofnb, eps
    real(kind=chm_real) dx(*),dy(*),dz(*)
    real(kind=chm_real) x(*),y(*),z(*)
    logical qecontx
    real(kind=chm_real) econtx(*),ewldtq(*)

    real(kind=chm_real) dchi,hij


    !--- av_080628 Block code: H Kamberaj November 2007
#if KEY_BLOCK==1
    REAL(kind=chm_real)    :: COEF
    INTEGER                :: IBL,JBL,KKB
#endif /*  av_080628 close BLOCK*/

    !--- local
    integer jnd
    real(kind=chm_real) cgil,cgjl
    real(kind=chm_real) cgf
    real(kind=chm_real) c2ofnb
    integer i, j
    real(kind=chm_real) rewald
    real(kind=chm_real) df, r2, s
    real(kind=chm_real) dxi, dyi, dzi, deltx, delty, deltz
    integer nxi,nximax
    !---     for ewald summation
    real(kind=chm_real) rs,r1s,erf2,erfcx,derfc
    real(kind=chm_real) xval,rem,val0,val1,val2,d1,d2
    integer ixval
    integer ii,kk
    logical qafirst
    character(len=8) siddni,riddni,resdni,acdni,siddnj,riddnj,resdnj,acdnj
    integer atfrst,atlast,atadd

    !--- begin
    !--- if qcg=.true. ewald correction will be done in ewldexq
    if (.not.qcg) return
    qafirst=.true.
    cgf=ccelec*half/eps
    c2ofnb=ctofnb*ctofnb

    !-----   main loop over atom i ----------------------
    atfrst = mynodp
    atlast = natom
    atadd = numnod

    do i = atfrst,atlast,atadd
       dxi = zero
       dyi = zero
       dzi = zero
       dchi=zero
       cgil=cg(i)*cgf
       nxi=molptr(molt(i))
       nximax=molptr(molt(i)+1)-1
       DO J = NXI,NXIMAX
          JND = MOLBL(J)

          !--- namkh 01/20/04
#if KEY_GAMESS==1 || KEY_GAMESSUK==1 || KEY_MNDO97==1 || KEY_SQUANTM==1 || \
    KEY_QCHEM==1 || KEY_QTURBO==1 || KEY_G09==1
          if(qmused)then
          if((abs(igmsel(i)) == 1.or.abs(igmsel(i)).eq.2) .and. &
               (abs(igmsel(jnd)) == 1.or.abs(igmsel(jnd)).eq.2)) &
               goto 100
          endif
#endif

          if (i /= jnd) then
             cgjl = cg(jnd)
             deltx=x(i)-x(jnd)
             delty=y(i)-y(jnd)
             deltz=z(i)-z(jnd)
             s=max(rsmall,deltx**2+delty**2+deltz**2)
             r2=one/s
             !---     ---          ewald r-space sum
             rs=sqrt(s)
             r1s=one/rs
             !
             !--- inline erfc calculation for speed (from erfcd).
             xval=rs*kappa*ewrdel
             ixval=xval+half
             rem=xval-ixval
             ixval=ixval+2
             ixval=min(ixval,ewnpts-1)
             val0=ewldtq(ixval-1)
             val1=ewldtq(ixval)
             val2=ewldtq(ixval+1)
             d1=(val0-val2)*half
             d2=(val1+val1-val0-val2)*rem
             erfcx=val1-(d1+half*d2)*rem
             derfc=(d1+d2)*ewrdel*kappa
             !--- end of erfc inlined code.
             !
             erf2=one-erfcx
             rewald= -cgil*cgjl*erf2*r1s

             !av_080628
             ! Block code: H Kamberaj January 2007
#if KEY_BLOCK==1 /*(block)*/
             IF (QBLOCK) THEN
                ibl = iblckp(i)
                jbl = iblckp(jnd)
                if (jbl  <  ibl) then
                   kkb = ibl
                   ibl = jbl
                   jbl = kkb
                endif
                kkb = ibl + jbl*(jbl-1)/2
                coef = blcoee(kkb)
                REWALD = REWALD * coef
                DF = DF * coef
             ENDIF
#endif /*    (block)*/
             !av_080628

             elec=elec+rewald
             hij=-cgf*erf2*r1s
             df=cgil*cgjl*(-derfc*r1s + erf2*r2)*r1s
             !
             !cc##if analys (analys)
             if(qaterm) then
                if(qanbnd) then
                   kk=anslct(i)+anslct(jnd)
                   if(kk == 2 .or. (kk >= 1 .and. .not.qaonly)) then
                      if(qaunit < 0) then
                         ii=outu
                      else
                         ii=qaunit
                      endif
                      !
                      if(prnlev >= 5) then
                         if(qafirst) then
                            if(qlongl) then
                               write(ii,243)
                            else
                               write(ii,244)
                            endif
                            qafirst=.false.
                         endif
                         call atomid(i,siddni,riddni,resdni,acdni)
                         call atomid(jnd,siddnj,riddnj,resdnj,acdnj)
                         if(qlongl) then
                            write(ii,245) j,i, &
                                 siddni(1:idleng),riddni(1:idleng), &
                                 resdni(1:idleng),acdni(1:idleng),jnd, &
                                 siddnj(1:idleng),riddnj(1:idleng), &
                                 resdnj(1:idleng),acdnj(1:idleng), &
                                 rs,rewald,df,cg(i),cg(jnd)
                         else
                            write(ii,246) j,i, &
                                 siddni(1:idleng),riddni(1:idleng), &
                                 resdni(1:idleng),acdni(1:idleng),jnd, &
                                 siddnj(1:idleng),riddnj(1:idleng), &
                                 resdnj(1:idleng),acdnj(1:idleng), &
                                 rs,rewald,df,cg(i),cg(jnd)
                         endif
                      endif
                   endif
                endif
             endif
             if(qecontx) then
                s=half*rewald
                econtx(i)=econtx(i)+s
                econtx(jnd)=econtx(jnd)+s
             endif
             !cc##endif (analys)
             !
             deltx=deltx*df
             delty=delty*df
             deltz=deltz*df
             !
             dx(jnd) = dx(jnd)-deltx
             dy(jnd) = dy(jnd)-delty
             dz(jnd) = dz(jnd)-deltz
             dxi = dxi + deltx
             dyi = dyi + delty
             dzi = dzi + deltz
#if KEY_CHEQ==1
             if ( (cg(i) /= 0.0).and.(cg(jnd).ne.0.0)) then
#endif
                dch(jnd)=dch(jnd)+hij*cg(i)
                dchi=dchi+hij*cg(jnd)
#if KEY_CHEQ==1
             else
                dchi=dchi+0.0
             endif
#endif
          endif
100       continue
       enddo

       dx(i) = dx(i) + dxi
       dy(i) = dy(i) + dyi
       dz(i) = dz(i) + dzi
       dch(i)=dch(i)+dchi
    enddo

    !--- format statements
243 FORMAT('ANAL: EWLDEXQ:    Index        Atom-I', &
         '                   Atom-J          ', &
         '        Dist     ', &
         '      EELEC         Force          Charges')
244 FORMAT('ANAL: EWLDEXQ:    Index        Atom-I', &
         '                   Atom-J          ',/ &
         '        Dist      ', &
         '      EELEC         Force          Charges')
245 FORMAT('ANAL: EWLDEXQ>',I10,I5,4(1X,A),I5,4(1X,A),5F15.6)
246 FORMAT('ANAL: EWLDEXQ>',I10,I5,4(1X,A),I5,4(1X,A),/5F15.6)


    !---      dchi=zero
    !---      do i =1,natom
    !---        dchi=dchi+cg(i)*dch(i)
    !---        write(6,*)"i,cg,dch=",i,cg(i),dch(i)
    !---      enddo
    !---      dchi=0.5*dchi
    !---      write(6,*)"#$#$FROM EWLDEXQ> elec,dch*cg/2=",elec,dchi
    return
  end subroutine ewldexq
#endif /* (cheq_ewldex)*/

  !CHARMM Element source/nbonds/ewaldf.src 1.1
  SUBROUTINE KSPACE(EKSUM,LESELF,EQCOR,EUTIL, &
       QEKSUM,QESELF,QEQCOR,QEUTIL, &
       X,Y,Z,DX,DY,DZ,NATOM,CG,CGTOT &
#if KEY_LJPME==1
       ,ljc6 &
#endif
#if KEY_FLUCQ==1
       ,QFLUC,FQCFOR    &
#endif
       )
    !
    use fast
    use stream
#if KEY_TSM==1
    use tsms_mod
#endif
    use parallel
#if KEY_QUANTUM==1
    use quantm
#endif
#if KEY_MNDO97==1
    use mndo97
    use qm1_info, only : qm_main_r,mm_main_r
    use qmmmewald_module, only : getgrdq
#endif
#if KEY_SQUANTM==1 || KEY_QTURBO==1 || KEY_G09==1
    use squantm
#endif
#if KEY_GAMESS==1 || KEY_GAMESSUK==1 && KEY_SQUANTM==0
    use mndo97
#endif
    use pme_module,only:pme,qpme
#if KEY_PHMD==1 /*phmd_outer*/
    use phmd,only:QPHMD  ! PME-CPHMD -- Y Huang 2017
#endif

    character(len=9) :: file = "ewald.src"
    character(len=9) :: routine = "kspace"
    real(chm_real)  EKSUM,LESELF,EQCOR,EUTIL
    LOGICAL QEKSUM,QESELF,QEQCOR,QEUTIL
    real(chm_real)  X(*),Y(*),Z(*),DX(*),DY(*),DZ(*)
    real(chm_real)  CG(*),CGTOT
#if KEY_LJPME==1
    real(chm_real), intent(in) :: ljc6(*)
#endif
    INTEGER NATOM
#if KEY_FLUCQ==1
    LOGICAL QFLUC
    real(chm_real) FQCFOR(*)
#endif
    !
#if KEY_MNDO97==1
    real(chm_real) KHARGE
    logical :: q_qmmm_pme_do
#endif
#if KEY_GAMESS==1 || KEY_GAMESSUK==1 || KEY_QCHEM==1 || KEY_QTURBO==1 || KEY_G09==1
#if KEY_SQUANTM==0 && KEY_MNDO97==0
    real(chm_real) KHARGE
#endif
#endif
    !
    real(chm_real) CGTOTT
    ! end
    !
    !
    real(chm_real) BOXL(3)
    LOGICAL QDONE
    !
    ! begin
    !
    IF (NATOM <= 0) RETURN
    !
    ! namkh 08/08/04
    ! QM/MM-Ewald : Apply QM charge into total MM charge
#if KEY_QUANTUM==1 || KEY_MNDO97==1 || KEY_SQUANTM==1 || KEY_GAMESS==1 || \
    KEY_GAMESSUK==1 || KEY_QTURBO==1 || KEY_G09==1 /*qmewald*/
#if KEY_QUANTUM==1
    CGTOTT = CGTOT
    IF(LQMEWD) CGTOT = CHAGTOT
#elif KEY_MNDO97==1
    KHARGE = qm_main_r%qmcharge   ! REAL(IN2(65))
    CGTOTT = CGTOT
    IF(LQMEWD) CGTOT = CGMM+KHARGE
    !!if(lqmewd .and. mm_main_r%PMEwald) then
    !!   q_qmmm_pme_do=.true.  !
    !!else
       q_qmmm_pme_do=.false.
    !!end if
#elif KEY_SQUANTM==1
    CGTOTT = CGTOT
    IF(LQMEWD) CGTOT = CGMM+QMCHARGE(1)
#endif
    !
    ! now take care of gradient from MNDO
    ! virial portion: already added into EWVIRIAL.
#if KEY_SQUANTM==1
    IF(LQMEWD) CALL GETGRDQ(NATOM,DX,DY,DZ,QMFEP)
#elif KEY_QUANTUM==1 || KEY_GAMESSUK==1
#if KEY_GAMESS==0
    IF(LQMEWD) CALL GETGRDQ(NATOM,DX,DY,DZ,PDXYZ)
#endif
#if KEY_QTURBO==1
    CALL WRNDIE(-4,'<>', 'Ewald not implemented for QTURBO')
#endif
#elif KEY_MNDO97==1
    IF(LQMEWD) CALL GETGRDQ(NATOM,DX,DY,DZ)
#endif
#endif /*  (qmewald)*/
    !
    IF(.NOT.QPME) THEN
       CALL SETUPK1(QDONE,BOXL)
       IF(.NOT.QDONE) THEN
          IF(MAXKV < TOTK) THEN
             IF(MAXKV > 0) THEN
                call chmdealloc(file,routine,"pkvec",maxkv,crl=pkvec)
                call chmdealloc(file,routine,"pkxv", maxkv,intg=pkxv)
                call chmdealloc(file,routine,"pkyv", maxkv,intg=pkyv)
                call chmdealloc(file,routine,"pkzv", maxkv,intg=pkzv)
             ENDIF
             MAXKV=TOTK
             call chmalloc(file,routine,"pkvec",maxkv,crl=pkvec)
             call chmalloc(file,routine,"pkxv", maxkv,intg=pkxv)
             call chmalloc(file,routine,"pkyv", maxkv,intg=pkyv)
             call chmalloc(file,routine,"pkzv", maxkv,intg=pkzv)
          ENDIF
          CALL SETUPK2(BOXL,PKVEC,PKXV,PKYV,PKZV)
       ENDIF
    ENDIF
    !
    ! Call scalar/parallel version
    IF(QPME) THEN
#if KEY_FLUCQ==1
     IF (QFLUC) CALL WRNDIE(-4,'<KSPACE>', &
          'No FlucQ implementation for PME')
#endif
       IF(PRNLEV > 6) WRITE(OUTU,125) 'PME'
       CALL PME(EKSUM,LESELF,EQCOR,EUTIL, &
            QEKSUM,QESELF,QEQCOR,QEUTIL, &
            X,Y,Z,DX,DY,DZ,NATOM,CG,CGTOT, &
#if KEY_LJPME
            ljc6, &
#endif
            ewvirial,kappa &
#if KEY_PHMD==1
          , QPHMD & ! PME-CPHMD -- Y Huang 2017
#endif /* phmd */
#if KEY_MNDO97==1
           ,q_qmmm_pme_do    &
#endif
           ,.false.  & !DCM
            )
    ELSE
       IF(PRNLEV > 6) WRITE(OUTU,125) 'KSPACEP'
       CALL KSPACEP(EKSUM,LESELF,QEKSUM,QESELF, &
            X,Y,Z,DX,DY,DZ,NATOM,CG,BOXL &
#if KEY_FLUCQ==1
            ,QFLUC,FQCFOR     &
#endif
            )
    ENDIF
    !
    ! namkh 08/08/04
    ! QM/MM-Ewald : Recover original CGTOT
#if KEY_QUANTUM==1 || KEY_MNDO97==1 || KEY_SQUANTM==1
    IF(LQMEWD) CGTOT = CGTOTT
#endif
    !
#if KEY_DEBUG==1
    write(OUTU,456) ewvirial
456 format(' EWVIRIAL:',9F12.6)
#endif
    !
    RETURN
    !
125 FORMAT(' KSPACE: Using routine ',A,' for energy calculation.')
    !
  END SUBROUTINE KSPACE

#if KEY_MNDO97==1 /*qmmm-pme*/
  SUBROUTINE KSPACE_qmmm_prep(EKSUM,LESELF,EQCOR,EUTIL, &
                              QEKSUM,QESELF,QEQCOR,QEUTIL, &
                              X,Y,Z,DXYZ,NATOM,CG,CGTOT &
#if KEY_FLUCQ==1
                             ,QFLUC,FQCFOR    &
#endif
                             )
    !
    ! This routine only compute Kspace terms for the PME part, not regular Ewald sum part.
    !
    use fast
    use stream
#if KEY_TSM==1
    use tsms_mod
#endif
    use parallel
#if KEY_QUANTUM==1
    use quantm
#endif
#if KEY_MNDO97==1
    use mndo97
    use qm1_info, only : qm_main_r,mm_main_r
    use qmmmewald_module, only : qmmm_ewald_r,getgrdq
#endif
!!#if KEY_SQUANTM==1 || KEY_QCHEM==1 || KEY_QTURBO==1 || KEY_G09==1
!!    use squantm
!!#endif
!!#if KEY_GAMESS==1 || KEY_GAMESSUK==1 && KEY_SQUANTM==0
!!    use mndo97
!!#endif
    use pme_module,only:pme,qpme

    character(len=9) :: file = "ewald.src"
    character(len=9) :: routine = "kspace"
    real(chm_real)  EKSUM,LESELF,EQCOR,EUTIL
    LOGICAL QEKSUM,QESELF,QEQCOR,QEUTIL
    real(chm_real)  X(*),Y(*),Z(*),DXYZ(3,natom)
    real(chm_real)  CG(*),CGTOT
    real(chm_real),allocatable,dimension(:) :: dx,dy,dz
    INTEGER NATOM
#if KEY_FLUCQ==1
    LOGICAL QFLUC
    real(chm_real) FQCFOR(*)
#endif
    !
#if KEY_MNDO97==1
    integer :: i
    real(chm_real) KHARGE
    logical :: q_qmmm_pme_do
#endif
    !
    real(chm_real) CGTOTT
    ! end
    !
    !
    real(chm_real) BOXL(3)
    LOGICAL QDONE
    !
    ! begin
    !
    IF (NATOM <= 0) RETURN
    allocate(dx(natom),dy(natom),dz(natom))
    !
    ! namkh 08/08/04
    ! QM/MM-Ewald : Apply QM charge into total MM charge
#if KEY_QUANTUM==1 || KEY_MNDO97==1 || KEY_SQUANTM==1 || KEY_GAMESS==1 || \
    KEY_GAMESSUK==1 || KEY_QCHEM==1 || KEY_QTURBO==1 || KEY_G09==1 /*qmewald*/
#if KEY_MNDO97==1
    KHARGE = qm_main_r%qmcharge
    CGTOTT = CGTOT
    IF(LQMEWD) CGTOT = CGMM+KHARGE
    if(lqmewd .and. mm_main_r%PMEwald) then
       q_qmmm_pme_do=.true.  !
       ! initialization here: d_ewald_mm contains all forces from PME routines.
       do i=1,natom
          dx(i) = zero  ! qmmm_ewald_r%d_ewald_mm(1:3,i)=zero
          dy(i) = zero
          dz(i) = zero
       end do
    else
       q_qmmm_pme_do=.false.
    end if
#endif
!!    !
!!    ! now take care of gradient from MNDO
!!    ! virial portion: already added into EWVIRIAL.
!!#if KEY_MNDO97==1
!!    IF(LQMEWD) CALL GETGRDQ(NATOM,DX,DY,DZ)
!!#endif
#endif   /*qmewald*/
    !
    ! Call scalar/parallel version
    IF(QPME) THEN
#if KEY_FLUCQ==1
       IF (QFLUC) CALL WRNDIE(-4,'<KSPACE>','No FlucQ implementation for PME')
#endif
       IF(PRNLEV > 6) WRITE(OUTU,125) 'PME'
       CALL PME(EKSUM,LESELF,EQCOR,EUTIL, &
            QEKSUM,QESELF,QEQCOR,QEUTIL, &
            X,Y,Z,DX,DY,DZ,NATOM,CG,CGTOT, &
#if KEY_LJPME == 1
          (/ 0.0_chm_real /), &
#endif
            ewvirial,kappa &
#if KEY_PHMD==1
          , .false. &
#endif /* phmd */
#if KEY_MNDO97==1
          , q_qmmm_pme_do  &
#endif
          , .false. & ! qdcm
           )
    ENDIF

    ! copy back and deallocate memory
    do i=1,natom
       dxyz(1,i) = dx(i)
       dxyz(2,i) = dy(i)
       dxyz(3,i) = dz(i)
    end do
    deallocate(dx,dy,dz)
    !
    ! namkh 08/08/04
    ! QM/MM-Ewald : Recover original CGTOT
#if KEY_QUANTUM==1 || KEY_MNDO97==1 || KEY_SQUANTM==1
    IF(LQMEWD) CGTOT = CGTOTT
#endif
    !
#if KEY_DEBUG==1
    write(OUTU,456) ewvirial
456 format(' EWVIRIAL:',9F12.6)
#endif
    !
    RETURN
    !
125 FORMAT(' KSPACE_qmmm_prep: Using routine ',A,' for energy calculation.')
    !
  END SUBROUTINE KSPACE_qmmm_prep


  SUBROUTINE KSPACE_qmmm_get_force(DX,DY,DZ,ewvirialax,NATOM)
  !
#if KEY_MNDO97==1
  use mndo97
  use qmmmewald_module, only : getgrdq
  use ewald_1m, only: lewald,EWVIRIAL2
#endif

  integer        :: natom
  real(chm_real) :: DX(*),DY(*),DZ(*)

#if KEY_MNDO97==1
  real(chm_real) :: ewvirialax(9)
  !ewvirialax(1:9) = ewvirialax(1:9)+ewvirial2(1:9)
  if(LQMEWD) call getgrdq(NATOM,DX,DY,DZ)
#endif

  return
  END SUBROUTINE KSPACE_qmmm_get_force
  !
#endif /*qmmm-pme*/

  SUBROUTINE SETUPK1(QDONE,BOXL)
    !
    !    Setup wave-vector arrays for K-space ewald summation.
    !    Authors:
    !           Stephen H. Fleischman
    !           Roland Stote
    !           11/91
    !
    use consta,only:twopi
    use stream,only:outu
    use image,only:xtlabc
    use pbound
    use param_store, only: set_param

    implicit none

    LOGICAL QDONE
    real(chm_real)  BOXL(3)
    !
    !    *******************************************************************
    !    ** ROUTINE TO SET UP THE WAVE-VECTORS FOR THE EWALD SUM.         **
    !    **                                                               **
    !    *******************************************************************
    !
    real(chm_real), PARAMETER :: TOL = 1.0D-04
    !
    INTEGER     KSQ, KX, KY, KZ, KSY, KSZ
    real(chm_real)      B, RKX, RKY, RKZ
    !    *******************************************************************
    !
    !
    ! Calculate box lengths
    if (qboun) then
#if KEY_PBOUND==1
       boxl(1) = xsize
       boxl(2) = ysize
       boxl(3) = zsize
#endif
    else
       BOXL(1)=XTLABC(1)
       BOXL(2)=XTLABC(3)
       BOXL(3)=XTLABC(6)
    endif
    IF(BOXL(1) <= ZERO.OR.BOXL(2).LE.ZERO.OR.BOXL(3).LE.ZERO) &
         CALL WRNDIE(-5,'<SETUPK>', &
         'One or more of the box sides was less than or equal to zero')
    ! Check to see if the box is orthorhombic (die otherwise)
    if (qboun &
#if KEY_PBOUND==1
          .and. (qTOBoun .or. qRDBoun .or. qRHBoun) &
#endif
          ) then
       CALL WRNDIE(-5,'<SETUPK>', &
            'The periodic box is non orthorhombic (EWALD wont work)')
    else
       B=ABS(XTLABC(2))+ABS(XTLABC(4))+ABS(XTLABC(5))
       IF(B > RSMALL) CALL WRNDIE(-5,'<SETUPK>', &
            'The periodic box is non orthorhombic (EWALD wont work)')
    endif
    !
    ! Do we have to do this?
    IF(QSETUPK) THEN
       QDONE =OKMAX == KMAX     .AND. &
            OKAPPA == KAPPA   .AND. &
            OKSQMAX == KSQMAX .AND. &
            OLEWLD == BOXL(1) .AND. &
            OMEWLD == BOXL(2) .AND. &
            ONEWLD == BOXL(3) .AND. &
            OKMAXX == KMAXX   .AND. &
            OKMAXY == KMAXY   .AND. &
            OKMAXZ == KMAXZ
       IF(QDONE) RETURN
    ELSE
       QSETUPK = .TRUE.
       QDONE=.FALSE.
    ENDIF
    !
    B = ONE/FOUR/KAPPA/KAPPA
    TOTK = 0
    DO KX = 0, KMAXX
       RKX = TWOPI*KX/BOXL(1)
       IF(KX == 0) THEN
          KSY = 0
       ELSE
          KSY = -KMAXY
       ENDIF
       DO KY = KSY, KMAXY
          RKY = TWOPI*KY/BOXL(2)
          IF(KX == 0.AND.KY.EQ.0) THEN
             KSZ = 1
          ELSE
             KSZ = -KMAXZ
          ENDIF
          DO KZ = KSZ, KMAXZ
             RKZ = TWOPI*KZ/BOXL(3)
             KSQ = KX*KX + KY*KY + KZ*KZ
             IF (KSQ <= KSQMAX.AND. KSQ /= 0) TOTK = TOTK + 1
          ENDDO
       ENDDO
    ENDDO
    !
    OKMAX = KMAX
    OKMAXX = KMAXX
    OKMAXY = KMAXY
    OKMAXZ = KMAXZ
    OKAPPA = KAPPA
    OKSQMAX = KSQMAX
    OLEWLD = BOXL(1)
    OMEWLD = BOXL(2)
    ONEWLD = BOXL(3)
    IF(TOTK /= OTOTK) THEN
       IF(WRNLEV >= 2) WRITE(OUTU,'(A,I10)') &
            'Setup Kspace: Number of wave-vectors = ',TOTK
       OTOTK = TOTK
       CALL set_param('TOTK',TOTK)
    ENDIF
    !
    RETURN
  END SUBROUTINE SETUPK1



  SUBROUTINE SETUPK2(BOXL,KVEC,KXV,KYV,KZV)
    !
    !    Setup wave-vector arrays for K-space ewald summation.
    !    Authors:
    !           Stephen H. Fleischman
    !           Roland Stote
    !           11/91
    !
    !    Code Modified to setup wave-vector arrays for K-space Ewald summation
    !    for any crystal shape
    !    Author:
    !    Hiqmet Kamberaj
    !    November 2007
    !
    use consta,only:twopi
    use stream,only:outu,prnlev
    use image,only:xtlabc
    use pbound
    use prssre

    real(chm_real) BOXL(3)
    real(chm_real) KVEC(*)
    INTEGER KXV(*),KYV(*),KZV(*)
    !
    !    *******************************************************************
    !    ** ROUTINE TO SET UP THE WAVE-VECTORS FOR THE EWALD SUM.         **
    !    **                                                               **
    !    *******************************************************************
    !
    !      real(chm_real)        TOL
    !      PARAMETER     (TOL = 1.0D-04)
    !      real(chm_real)        VFACT
    !
    !      INTEGER     KSQ, KX, KY, KZ, KSY, KSZ
    !      INTEGER     IPT
    !      real(chm_real)      B, RKX, RKY, RKZ, RKSQ
    !    *******************************************************************
    !
    ! Local variables
    real(chm_real), PARAMETER :: TOL = 1.0D-04
    real(chm_real)      VFACT
    INTEGER     KSQ, KSQT
    INTEGER     KX2,KX, KY, KZ, KSY, KSZ
    INTEGER     IPT
    real(chm_real)      B, RKX, RKY, RKZ, RKSQ
    real(chm_real)      RKXT,RKYT,RKZT,VOL
    real(chm_real)      XTLINV(6),RECIP(3,3)
    LOGICAL     OK
    !
    B = ONE/FOUR/KAPPA/KAPPA
    if (qboun) then
#if KEY_PBOUND==1
       call pbound_getvol(vol)
#endif
    else
       ! Calculate the volume (H Kamberaj, November 2007)
       CALL GETVOL(VOL)
    endif

    IF (PRNLEV  >  6) THEN
       WRITE(OUTU,*) VOL
    ENDIF

    if (qboun) then
#if KEY_PBOUND==1
       ! APH: Assumes orthorhombic box
       xtlinv(1) = boxinv
       xtlinv(2) = zero
       xtlinv(3) = boyinv
       xtlinv(4) = zero
       xtlinv(5) = zero
       xtlinv(6) = bozinv
       ok = .true.
#endif
    else
       CALL INVT33S(XTLINV,XTLABC,OK)
    endif
    IF (.NOT. OK) &
         CALL WRNDIE(-4,'<SETUPK2>','ERROR CALC INV BOX')

    !
    RECIP(1,1) = TWOPI*XTLINV(1)
    RECIP(2,2) = TWOPI*XTLINV(3)
    RECIP(3,3) = TWOPI*XTLINV(6)
    RECIP(2,1) = TWOPI*XTLINV(2)
    RECIP(1,2) = RECIP(2,1)
    RECIP(3,1) = TWOPI*XTLINV(4)
    RECIP(1,3) = RECIP(3,1)
    RECIP(3,2) = TWOPI*XTLINV(5)
    RECIP(2,3) = RECIP(3,2)

    ! Setup wave-vector array (Last modified H Kamberaj, November 2007)
    VFACT = TWOPI/VOL
    IPT = 0
    DO KX = 0, KMAXX
       KX2 = KX*KX
       IF(KX == 0) THEN
          KSY = 0
       ELSE
          KSY = -KMAXY
       ENDIF
       DO KY = KSY, KMAXY
          RKXT = RECIP(1,1)*KX + RECIP(2,1)*KY
          RKYT = RECIP(1,2)*KX + RECIP(2,2)*KY
          RKZT = RECIP(1,3)*KX + RECIP(2,3)*KY
          KSQT = KX2 + KY*KY
          IF(KX == 0.AND.KY.EQ.0) THEN
             KSZ = 1
          ELSE
             KSZ = -KMAXZ
          ENDIF
          DO KZ = KSZ, KMAXZ
             RKX = RKXT + RECIP(3,1)*KZ
             RKY = RKYT + RECIP(3,2)*KZ
             RKZ = RKZT + RECIP(3,3)*KZ
             KSQ = KSQT + KZ*KZ
             IF (KSQ <= KSQMAX.AND. KSQ /= 0) THEN
                IPT = IPT + 1
                RKSQ = RKX*RKX + RKY*RKY + RKZ*RKZ
                KVEC(IPT) = VFACT*EXP((-B*RKSQ))/RKSQ
                KXV(IPT) = KX
                KYV(IPT) = KY
                KZV(IPT) = KZ
             ENDIF
          ENDDO
       ENDDO
    ENDDO
    !
    IF(IPT /= TOTK) CALL WRNDIE(-4,'<SETUPK2>','Bad TOTK count')
    !
    RETURN
  END SUBROUTINE SETUPK2

  SUBROUTINE KSPACEP(EKSUM,LESELF,QEKSUM,QESELF, &
       X,Y,Z,DX,DY,DZ,NATOM,CG,BOXL &
#if KEY_FLUCQ==1
       ,QFLUC,FQCFOR     &
#endif
       )
#if KEY_TSM==1
    use tsms_mod
    use tsmh
#endif
    use parallel

    character(len=9) :: file = "ewald.src"
    character(len=9) :: routine = "kspacep"
    !-----------------------------------------------------------------------
    !     This routine calculates non bonded interaction energies and
    !     forces via the Ewald Summation
    !     EKSUM  - electrostatic energy from kspace sum
    !     NATOM  - number of atoms
    !
    !     This is the scalar version that has been desiged to work on
    !     distributed memory parallel computers.
    !                    Scott Feller and Bernie Brooks, NIH, 6/95
    !-----------------------------------------------------------------------
    real(chm_real)  EKSUM,LESELF
    LOGICAL QEKSUM,QESELF
    real(chm_real)  X(*),Y(*),Z(*),DX(*),DY(*),DZ(*)
    real(chm_real)  CG(*)
    INTEGER NATOM
    real(chm_real)  BOXL(3)
#if KEY_FLUCQ==1
    LOGICAL QFLUC
    real(chm_real) FQCFOR(*)
#endif
    !
#if KEY_TSM==1
    real(chm_real) ESELFP,ESELFR,EELPP,EELRP
#endif
    !
    INTEGER ATFRST,ATLAST,NAT
    !
#if KEY_PARALLEL==1 /*paramain*/
#if KEY_PARAFULL==1 /*parfmain*/
#if KEY_PARASCAL==1 /*parstest*/
#error  'Illegal parallel compile options'
#endif /* (parstest)*/
    ATFRST=1+IPARPT(MYNOD)
    ATLAST=IPARPT(MYNODP)
#elif KEY_PARASCAL==1 || KEY_SPACDEC==1 /*parfmain*/
    ATFRST=1
    ATLAST=NATOM
#else /* (parfmain)*/
#error  'Illegal parallel compile options'
#endif /* (parfmain)*/
#else /* (paramain)*/
    ATFRST=1
    ATLAST=NATOM
#endif /* (paramain)*/
    !
    NAT=ATLAST-ATFRST+1

    !
    CALL KSPACEP2(EKSUM,LESELF,QEKSUM,QESELF,X,Y,Z,DX,DY,DZ,NATOM,CG, &
         KMAXX,KMAXY,KMAXZ,NAT,ATFRST,TOTK, &
         BOXL, &
         PKVEC,PKXV,PKYV,PKZV &
#if KEY_TSM==1
         ,QTSM,ESELFR,ESELFP,EELRP,EELPP,LAMBDA     &
#endif
#if KEY_TSM==1
         ,REACLS,PRODLS                             &
#endif
#if KEY_FLUCQ==1
         ,QFLUC,FQCFOR                              &
#endif
         )
    !
#if KEY_TSM==1
    ! It is assumed that tsme has already been called and that the
    ! vprtxx variables have already been initialized.
    IF (QTSM) THEN
       IF(QESELF) THEN
          !          subtract the self term
          VPRTTR = VPRTTR - ESELFR
          VPRTNR = VPRTNR - ESELFR
          VPRTER = VPRTER - ESELFR
          VPRTTP = VPRTTP - ESELFP
          VPRTNP = VPRTNP - ESELFP
          VPRTEP = VPRTEP - ESELFP
       ENDIF
       VPRTTR = VPRTTR + EELRP
       VPRTNR = VPRTNR + EELRP
       VPRTER = VPRTER + EELRP
       VPRTTP = VPRTTP + EELPP
       VPRTNP = VPRTNP + EELPP
       VPRTEP = VPRTEP + EELPP
    ENDIF
#endif
    !
    RETURN
  END SUBROUTINE KSPACEP

  SUBROUTINE KSPACEP2(EKSUM,LESELF,QEKSUM,QESELF, &
       X,Y,Z,DX,DY,DZ,NATOM,CG, &
       KMX,KMY,KMZ,NAT,ATFRST,TTK, &
       BOXL,KVEC,KXV,KYV,KZV &
#if KEY_TSM==1
       ,LTSM,ESELFR,ESELFP,EELR,EELP         &
#endif
#if KEY_TSM==1
       ,LAMBDA,REACLS,PRODLS                 &
#endif
#if KEY_FLUCQ==1
       ,QFLUC,FQCFOR                         &
#endif
       )
    !-----------------------------------------------------------------------
    !     This routine calculates non bonded interaction energies and
    !     forces via the Ewald Summation
    !     EKSUM  - electrostatic energy from kspace sum
    !     NATOM  - number of atoms
    !
    !                ATOM/VATOM nonbond cutoff options are supported.
    !     This routine is a combination of SHF's old subroutines KSPACE, KTABLE,
    !     and KSPACEF2.  It has been rearranged so as to work on distributed
    !     memory parallel computers.  Scott Feller, NIH, 6/95
    !
    !     The code is modified for any box shape;
    !     The BLOCK is added.
    !     Author:
    !     Hiqmet Kamberaj
    !     November2007
    !
    !-----------------------------------------------------------------------

    ! QM/MM-Ewald
#if KEY_QUANTUM==1
    use quantm,only: LQMEWD, QSETUPKQ, QCGSC
#endif
#if KEY_MNDO97==1
    use mndo97,only: lqmewd
#endif
#if KEY_GAMESS==1 || KEY_GAMESSUK==1
#if KEY_SQUANTM==0
    use mndo97,only: lqmewd
#endif
#endif
#if KEY_SQUANTM==1 || KEY_QTURBO==1 || KEY_G09==1
    use squantm,only: lqmewd,ewmode
#endif

#if KEY_CHEQ==1
    use cheq,only:qcg,   &
#endif
#if KEY_CHEQ==1
         DCH,SUMDCH,DDCH
#endif

    use exfunc
#if KEY_BLOCK==1
    use block_fcm,only:qblock,iblckp,nblock,blcoee
#endif
    use image,only:xtlabc
    use pbound
    use consta,only:twopi,ccelec,pi
    use parallel
    use machutil,only:eclock

    character(len=9) :: file = "ewald.src"
    character(len=9) :: routine = "kspacep2"
#if KEY_CHEQ==1
    real(chm_real) HIJ
#endif
    real(chm_real)  EKSUM,LESELF
    LOGICAL QEKSUM,QESELF
    real(chm_real)  X(*),Y(*),Z(*),DX(*),DY(*),DZ(*)
    INTEGER NATOM
    real(chm_real)  CG(*)
    INTEGER KMX,KMY,KMZ,NAT,ATFRST,TTK
    real(chm_real) BOXL(3),KVEC(*)
    INTEGER KXV(*),KYV(*),KZV(*)
#if KEY_TSM==1
    LOGICAL LTSM
    real(chm_real) ESELFR,ESELFP,EELR,EELP,LAMBDA
    INTEGER REACLS(*),PRODLS(*)
#endif
#if KEY_FLUCQ==1
    LOGICAL QFLUC
    real(chm_real) FQCFOR(*)
#endif
    !
    ! H Kamberaj: block code added, Nov 2007
#if KEY_BLOCK==1
    real(chm_real) COEF
    INTEGER IBL, JBL, KK,JB,IB
    real(chm_real) BSUML(2,NBLOCK),BSUM(2,TTK,NBLOCK)
    real(chm_real) BSUMCOS(NBLOCK),BSUMSIN(NBLOCK)
#endif /*  block*/
    !
    ! LOCAL VARIABLES
    !
    real(chm_real)  EEL,ESLF,RPISQRT
    real(chm_real)  CRECIP(3,3),PRECIP(3,3),RECIP(3,3)
    real(chm_real)  KR1,KR2,KR3
    real(chm_real)  XTLINV(6)
    real(chm_real)  SUML(6)
    INTEGER I,J,II
    INTEGER KX,KY,KZ
    real(chm_real)  KXR,KYR,KZR
    real(chm_real)  CCFKX,CCFKY,CCFKZ,FD,TIMME1
    real(chm_real)  TPL,TPM,TPN,SUMSIN,SUMCOS
    real(chm_real)  KTG1,KTG2
    real(chm_real)  KLEN,EWPR,EWEN,EN
    LOGICAL OK
    !
#if KEY_TSM==1
    real(chm_real) SUMCOSR,SUMSINR,SUMCOSP,SUMSINP
    real(chm_real) SUMSINE,SUMCOSE,SUMSINRE,SUMCOSRE,SUMSINPE,SUMCOSPE
#endif
    !
!    real(chm_real),PARAMETER  :: CCONST=FOUR*TWOPI*CCELEC
    real(chm_real)  :: CCONST
    !
    real(chm_real),allocatable,dimension(:,:) :: &
         KTABXC,KTABXS,KTABYC,KTABYS,KTABZC,KTABZS,sum
    real(chm_real),allocatable,dimension(:) :: CT,ST

    CCONST=FOUR*TWOPI*CCELEC

    call chmalloc(file,routine,"ktabxc",NAT,KMX+1  ,lbou2=0,crl=ktabxc)
    call chmalloc(file,routine,"ktabxs",NAT,KMX+1  ,lbou2=0,crl=ktabxs)
    call chmalloc(file,routine,"ktabyc",NAT,2*KMY+1,lbou2=-kmy,crl=ktabyc)
    call chmalloc(file,routine,"ktabys",NAT,2*KMY+1,lbou2=-kmy,crl=ktabys)
    call chmalloc(file,routine,"ktabzc",NAT,2*KMZ+1,lbou2=-kmz,crl=ktabzc)
    call chmalloc(file,routine,"ktabzs",NAT,2*KMZ+1,lbou2=-kmz,crl=ktabzs)
    call chmalloc(file,routine,"sum",   2,ttk,         crl=sum)
    call chmalloc(file,routine,"ct",nat,crl=ct)
    call chmalloc(file,routine,"st",nat,crl=st)

    RPISQRT=ONE/SQRT(PI)
    !
    ! Set some zeros
#if KEY_PARALLEL==1
    TIMME1 = ECLOCK()
#endif
    EEL=ZERO
    ESLF=ZERO
    !
#if KEY_QUANTUM==1 || KEY_MNDO97==1 || KEY_SQUANTM==1 || KEY_GAMESS==1 || \
    KEY_GAMESSUK==1 || KEY_QTURBO==1 || KEY_G09==1
    IF (.NOT.LQMEWD) THEN
#endif
       DO I = 1, 9
          EWVIRIAL(I) = ZERO
       END DO
#if KEY_QUANTUM==1 || KEY_MNDO97==1 || KEY_SQUANTM==1 || KEY_GAMESS==1 || \
    KEY_GAMESSUK==1 || KEY_QTURBO==1 || KEY_G09==1
    END IF
#endif
    !
    !
    ! Calculate the volume and inverse of box cell
    ! (H Kamberaj, November 2007)
    !
    if (qboun) then
#if KEY_PBOUND==1
       ! APH: Assumes orthorhombic box
       xtlinv(1) = boxinv
       xtlinv(2) = zero
       xtlinv(3) = boyinv
       xtlinv(4) = zero
       xtlinv(5) = zero
       xtlinv(6) = bozinv
       ok = .true.
#endif
    else
       CALL INVT33S(XTLINV,XTLABC,OK)
    endif
    !
    IF (.NOT. OK) THEN
       CALL WRNDIE(-4,'<KSPACEP2>','ERROR CALC INV BOX')
    ENDIF
    !

    RECIP(1,1) = XTLINV(1)
    RECIP(2,2) = XTLINV(3)
    RECIP(3,3) = XTLINV(6)
    RECIP(2,1) = XTLINV(2)
    RECIP(1,2) = RECIP(2,1)
    RECIP(3,1) = XTLINV(4)
    RECIP(1,3) = RECIP(3,1)
    RECIP(3,2) = XTLINV(5)
    RECIP(2,3) = RECIP(3,2)


    PRECIP(1,1) = TWOPI*XTLINV(1)
    PRECIP(2,2) = TWOPI*XTLINV(3)
    PRECIP(3,3) = TWOPI*XTLINV(6)
    PRECIP(2,1) = TWOPI*XTLINV(2)
    PRECIP(1,2) = PRECIP(2,1)
    PRECIP(3,1) = TWOPI*XTLINV(4)
    PRECIP(1,3) = PRECIP(3,1)
    PRECIP(3,2) = TWOPI*XTLINV(5)
    PRECIP(2,3) = PRECIP(3,2)

    CRECIP(1,1) = CCONST*XTLINV(1)
    CRECIP(2,2) = CCONST*XTLINV(3)
    CRECIP(3,3) = CCONST*XTLINV(6)
    CRECIP(2,1) = CCONST*XTLINV(2)
    CRECIP(1,2) = CRECIP(2,1)
    CRECIP(3,1) = CCONST*XTLINV(4)
    CRECIP(1,3) = CRECIP(3,1)
    CRECIP(3,2) = CCONST*XTLINV(5)
    CRECIP(2,3) = CRECIP(3,2)
    !

#if KEY_TSM==1
    ESELFR=ZERO
    ESELFP=ZERO
#endif
    ! SAPATEL
#if KEY_CHEQ==1
    HIJ = TWO*KAPPA*RPISQRT*CCELEC
#endif
#if KEY_CHEQ==1
    IF(.not.QESELF) HIJ = ZERO
#endif
    ! SAPATEL
    !
    DO I = 1,NAT
       II=I+ATFRST-1
       !     construct ktable for each atom
       KTABXC(I,0) = ONE
       KTABXS(I,0) = ZERO
       !
       KTABYC(I,0) = ONE
       KTABYS(I,0) = ZERO
       !
       KTABZC(I,0) = ONE
       KTABZS(I,0) = ZERO
       !

       ! H Kamberaj (Jan. 2007) modified for any box shape
       KR1 = PRECIP(1,1)*X(II) + &
            PRECIP(2,1)*Y(II) + &
            PRECIP(3,1)*Z(II)

       KR2 = PRECIP(1,2)*X(II) + &
            PRECIP(2,2)*Y(II) + &
            PRECIP(3,2)*Z(II)

       KR3 = PRECIP(1,3)*X(II) + &
            PRECIP(2,3)*Y(II) + &
            PRECIP(3,3)*Z(II)


       KTABXC(I,1) = COS(KR1)
       KTABXS(I,1) = SIN(KR1)
       KTABYC(I,1) = COS(KR2)
       KTABYS(I,1) = SIN(KR2)
       KTABYC(I,-1) =  KTABYC(I,1)
       KTABYS(I,-1) = -KTABYS(I,1)
       KTABZC(I,1) = COS(KR3)
       KTABZS(I,1) = SIN(KR3)
       KTABZC(I,-1) =  KTABZC(I,1)
       KTABZS(I,-1) = -KTABZS(I,1)

       DO J = 2, KMX
          KTABXC(I,J)=KTABXC(I,1)*KTABXC(I,J-1) &
               -KTABXS(I,1)*KTABXS(I,J-1)
          KTABXS(I,J)=KTABXS(I,1)*KTABXC(I,J-1) &
               +KTABXC(I,1)*KTABXS(I,J-1)
       ENDDO
       DO J = 2, KMY
          KTABYC(I,J)=KTABYC(I,1)*KTABYC(I,J-1) &
               -KTABYS(I,1)*KTABYS(I,J-1)
          KTABYS(I,J)=KTABYS(I,1)*KTABYC(I,J-1) &
               +KTABYC(I,1)*KTABYS(I,J-1)
          KTABYC(I,-J)= KTABYC(I,J)
          KTABYS(I,-J)=-KTABYS(I,J)
       ENDDO
       DO J = 2, KMZ
          KTABZC(I,J)=KTABZC(I,1)*KTABZC(I,J-1) &
               -KTABZS(I,1)*KTABZS(I,J-1)
          KTABZS(I,J)=KTABZS(I,1)*KTABZC(I,J-1) &
               +KTABZC(I,1)*KTABZS(I,J-1)
          KTABZC(I,-J)= KTABZC(I,J)
          KTABZS(I,-J)=-KTABZS(I,J)
       ENDDO
       EN = CG(II)*CG(II)
       !
#if KEY_BLOCK==1
       IF (QBLOCK) THEN
          IBL=iblckp(II)
          KK=IBL+IBL*(IBL-1)/2
          COEF=BLCOEE(KK)
          EN=EN*COEF
       ENDIF
#endif /*  close BLOCK*/
       !
#if KEY_TSM==1
       IF(.NOT.LTSM) THEN
          ESLF = ESLF + EN
       ELSE
          IF(REACLS(II) == 1) THEN
             ESELFR = ESELFR + EN
          ELSEIF(PRODLS(II) == 1) THEN
             ESELFP = ESELFP + EN
          ELSE
             ESLF = ESLF + EN
          ENDIF
       ENDIF
#else /**/
       ESLF = ESLF + EN
#endif
       ! SAPATEL
#if KEY_CHEQ==1
       IF(QCG) THEN
          EN = HIJ*CG(II)
          !
#if KEY_BLOCK==1
          IF (QBLOCK) EN = EN * COEF
#endif /*  CLOSE BLOCK*/
          !
#if KEY_CHEQ==1
          DCH(II) = DCH(II) - EN
#endif
       ENDIF
       !
#endif
       ! SAPATEL
#if KEY_FLUCQ==1
       !
       IF (QFLUC.AND.QESELF) THEN
          EN = TWO*KAPPA*RPISQRT*CCELEC*CG(II)
          !
#if KEY_BLOCK==1
          IF (QBLOCK) EN=EN*COEF
#endif /*  close BLOCK*/
          !
          FQCFOR(II)=FQCFOR(II)-EN
       ENDIF
       !
#endif /* close FLUCQ*/
       !
    ENDDO
    !
    IF(QESELF) THEN
       LESELF =  -ESLF*KAPPA*RPISQRT*CCELEC
#if KEY_TSM==1
       IF(LTSM) THEN
          LESELF = -ESLF*KAPPA*RPISQRT*CCELEC
          ESELFR = -ESELFR*KAPPA*RPISQRT*CCELEC
          ESELFP = -ESELFP*KAPPA*RPISQRT*CCELEC
          !           assume linear lambda scaling
          LESELF = LESELF + (ONE-LAMBDA)*ESELFR + LAMBDA*ESELFP
       ENDIF
#endif
    ENDIF
    !
    IF(.NOT.QEKSUM) then
       call chmdealloc(file,routine,"ktabxc",NAT,KMX+1  ,crl=ktabxc)
       call chmdealloc(file,routine,"ktabxs",NAT,KMX+1  ,crl=ktabxs)
       call chmdealloc(file,routine,"ktabyc",NAT,2*KMY+1,crl=ktabyc)
       call chmdealloc(file,routine,"ktabys",NAT,2*KMY+1,crl=ktabys)
       call chmdealloc(file,routine,"ktabzc",NAT,2*KMZ+1,crl=ktabzc)
       call chmdealloc(file,routine,"ktabzs",NAT,2*KMZ+1,crl=ktabzs)
       call chmdealloc(file,routine,"sum",   2,ttk,         crl=sum)
       call chmdealloc(file,routine,"ct",nat,crl=ct)
       call chmdealloc(file,routine,"st",nat,crl=st)
       RETURN
    endif
    !
    !     Now do kspace summation for each atom
    !
#if KEY_PARALLEL==1 /*many_sum*/
    IF(NUMNOD > 0 &
#if KEY_TSM==1
         .AND. .NOT.LTSM &
#endif
         )then
       !
       ! Parallel version
       DO J = 1,TOTK
          !
          KX = KXV(J)
          KY = KYV(J)
          KZ = KZV(J)
          !
#if KEY_BLOCK==1
          IF (QBLOCK) THEN
             DO IB=1,NBLOCK
                BSUMCOS(IB) = ZERO
                BSUMSIN(IB) = ZERO
             ENDDO
          ELSE
             SUMSIN = ZERO
             SUMCOS = ZERO
          ENDIF
#else /**/
          SUMSIN = ZERO
          SUMCOS = ZERO
#endif /* close BLOCK*/
          !
          DO I = 1,NAT
             II=I+ATFRST-1
             KTG1 = KTABZC(I,KZ)*KTABYC(I,KY)-KTABYS(I,KY)*KTABZS(I,KZ)
             KTG2 = KTABZC(I,KZ)*KTABYS(I,KY)+KTABYC(I,KY)*KTABZS(I,KZ)
             CT(I) = CG(II)*(KTABXC(I,KX)*KTG1 - KTABXS(I,KX)*KTG2)
             ST(I) = CG(II)*(KTABXS(I,KX)*KTG1 + KTABXC(I,KX)*KTG2)
             !
#if KEY_BLOCK==1
             IF (QBLOCK) THEN
                IBL=iblckp(II)
                BSUMCOS(IBL) = BSUMCOS(IBL) + CT(I)
                BSUMSIN(IBL) = BSUMSIN(IBL) + ST(I)
             ELSE
                SUMCOS = SUMCOS + CT(I)
                SUMSIN = SUMSIN + ST(I)
             ENDIF
#else /**/
             SUMCOS = SUMCOS + CT(I)
             SUMSIN = SUMSIN + ST(I)
#endif /* close BLOCK*/
             !
          ENDDO
          !
#if KEY_BLOCK==1
          IF (QBLOCK) THEN
             DO IB=1, NBLOCK
                BSUM(1,J,IB) = BSUMCOS(IB)
                BSUM(2,J,IB) = BSUMSIN(IB)
             ENDDO
          ELSE
             SUM(1,J) = SUMSIN
             SUM(2,J) = SUMCOS
          ENDIF
#else /**/
          SUM(1,J) = SUMSIN
          SUM(2,J) = SUMCOS
#endif /*  close BLOCK*/
          !
       ENDDO
       !
       TIMME1 = ECLOCK()
       !
#if KEY_BLOCK==1
       IF (QBLOCK) THEN
          CALL GCOMBr(BSUM,2*TOTK*NBLOCK)  ! MH2023: extend the gcomb interface for fix
       ELSE
          CALL GCOMBr(SUM,2*TOTK)
       ENDIF
#else /**/
       CALL GCOMBr(SUM,2*TOTK)
#endif /* close BLOCK*/
       !
       TMERI(TIMGCOMM)=TMERI(TIMGCOMM)+ECLOCK()-TIMME1
       TIMME1 = ECLOCK()
       !
       DO J = 1,TOTK
          KX = KXV(J)
          KY = KYV(J)
          KZ = KZV(J)
          !
          ! H Kamberaj (Jan. 2007): Modified for any box shape
          KR1 = CRECIP(1,1)*KX + CRECIP(2,1)*KY + CRECIP(3,1)*KZ
          KR2 = CRECIP(1,2)*KX + CRECIP(2,2)*KY + CRECIP(3,2)*KZ
          KR3 = CRECIP(1,3)*KX + CRECIP(2,3)*KY + CRECIP(3,3)*KZ

          CCFKX =  KR1*KVEC(J)
          CCFKY =  KR2*KVEC(J)
          CCFKZ =  KR3*KVEC(J)

          DO I = 1, NAT
             II=I+ATFRST-1
             KTG1 = KTABZC(I,KZ)*KTABYC(I,KY)-KTABYS(I,KY)*KTABZS(I,KZ)
             KTG2 = KTABZC(I,KZ)*KTABYS(I,KY)+KTABYC(I,KY)*KTABZS(I,KZ)
             CT(I) = CG(II)*(KTABXC(I,KX)*KTG1 - KTABXS(I,KX)*KTG2)
             ST(I) = CG(II)*(KTABXS(I,KX)*KTG1 + KTABXC(I,KX)*KTG2)
             !
#if KEY_BLOCK==1
             IF (QBLOCK) THEN
                FD = ZERO
                IB=iblckp(II)
                DO JB=1, NBLOCK
                   JBL = IB
                   JBL = JB
                   IF (JBL  <  IBL) THEN
                      KK=IBL
                      IBL=JBL
                      JBL=KK
                   ENDIF
                   KK=IBL+JBL*(JBL-1)/2
                   COEF=BLCOEE(KK)
                   FD = FD + COEF*( &
                        BSUM(1,J,JBL)*CT(I) - &
                        BSUM(2,J,JBL)*ST(I) )
                ENDDO

             ELSE
                FD = SUM(1,J)*CT(I) - SUM(2,J)*ST(I)
             ENDIF
#else /**/
             FD = SUM(1,J)*CT(I) - SUM(2,J)*ST(I)
#endif /*   close BLOCK*/
             !
             DX(II) = DX(II) + CCFKX*FD
             DY(II) = DY(II) + CCFKY*FD
             DZ(II) = DZ(II) + CCFKZ*FD
             ! SAPATEL
             !
#if KEY_CHEQ==1
             !
             IF(QCG.and.CG(II) /= 0) THEN
                !
#if KEY_BLOCK==1
                IF (QBLOCK) THEN
                   IB=iblckp(II)
                   DO JB=1, NBLOCK
                      IBL = IB
                      JBL = JB
                      IF (JBL  <  IBL) THEN
                         KK=IBL
                         IBL=JBL
                         JBL=KK
                      ENDIF
                      KK=IBL+JBL*(JBL-1)/2
                      COEF=BLCOEE(KK)
                      DCH(II) = DCH(II) + COEF*FOUR*CCELEC* &
                           KVEC(J)*( &
                           BSUM(2,J,JBL)*CT(I) + &
                           BSUM(1,J,JBL)*ST(I) ) / CG(II)
                   ENDDO

                ELSE
                   DCH(II) = DCH(II) + FOUR * CCELEC * &
                        KVEC(J)*(SUM(2,J)*CT(I)+SUM(1,J)*ST(I)) / CG(II)
                ENDIF
#else /*    */
                DCH(II) = DCH(II) + FOUR * CCELEC * &
                     KVEC(J)*(SUM(2,J)*CT(I) + SUM(1,J)*ST(I)) / CG(II)
#endif /* close BLOCK*/
                !
             ENDIF
             !
#endif /* close CHEQ*/
             !
             ! SAPATEL
#if KEY_FLUCQ==1
             !
             IF (QFLUC.AND.CG(II) /= ZERO) THEN
                !
#if KEY_BLOCK==1
                IF (QBLOCK) THEN
                   IB=iblckp(II)
                   DO JB=1, NBLOCK
                      IBL = IB
                      JBL = JB
                      IF (JBL  <  IBL) THEN
                         KK=IBL
                         IBL=JBL
                         JBL=KK
                      ENDIF
                      KK=IBL+JBL*(JBL-1)/2
                      COEF=BLCOEE(KK)
                      FQCFOR(II)=FQCFOR(II)+ &
                           COEF*FOUR* &
                           KVEC(J)*( &
                           BSUM(2,J,JBL)*CT(I) + &
                           BSUM(1,J,JBL)*ST(I) ) / CG(II)
                   ENDDO

                ELSE
                   FQCFOR(II)=FQCFOR(II)+ &
                        FOUR*KVEC(J)*( &
                        SUM(2,J)*CT(I)+ &
                        SUM(1,J)*ST(I) ) / CG(II)
                ENDIF
#else /**/
                FQCFOR(II)=FQCFOR(II) + &
                     FOUR*KVEC(J)*( &
                     SUM(2,J)*CT(I) + &
                     SUM(1,J)*ST(I) ) / CG(II)
#endif /* close BLOCK*/
                !
             ENDIF
             !
#endif /* close FLUCQ*/
             !
          ENDDO
          IF(MYNOD == 0) THEN
             !
#if KEY_BLOCK==1
             IF (QBLOCK) THEN
                DO IB=1, NBLOCK
                   DO JB=1, NBLOCK
                      IBL = IB
                      JBL = JB
                      IF (JBL  <  IBL) THEN
                         KK=IBL
                         IBL=JBL
                         JBL=KK
                      ENDIF
                      KK=IBL+JBL*(JBL-1)/2
                      COEF=BLCOEE(KK)
                      EEL=EEL+COEF*KVEC(J)*( &
                           BSUM(2,J,IB)*BSUM(2,J,JB) + &
                           BSUM(1,J,IB)*BSUM(1,J,JB))
                   ENDDO
                ENDDO
             ELSE
                EEL = EEL + KVEC(J)*(SUM(2,J)**2+SUM(1,J)**2)
             ENDIF
#else /**/
             EEL = EEL + KVEC(J)*(SUM(2,J)**2 + SUM(1,J)**2)
#endif /*   close BLOCK*/
             !
             !  Calculate the reciprocal space portion of the pressure tensor
             !  Code modified for any box shape (H Kamberaj, January 2007)
             KXR = RECIP(1,1)*KX + RECIP(2,1)*KY + RECIP(3,1)*KZ
             KYR = RECIP(1,2)*KX + RECIP(2,2)*KY + RECIP(3,2)*KZ
             KZR = RECIP(1,3)*KX + RECIP(2,3)*KY + RECIP(3,3)*KZ
             KLEN = KXR**2 + KYR**2 + KZR**2
             EWPR = TWO/KLEN + TWO*((PI/KAPPA)**2)
             !
#if KEY_BLOCK==1
             IF (QBLOCK) THEN
                EWEN = ZERO
                DO IB=1, NBLOCK
                   DO JB=1, NBLOCK
                      IBL = IB
                      JBL = JB
                      IF (JBL  <  IBL) THEN
                         KK=IBL
                         IBL=JBL
                         JBL=KK
                      ENDIF
                      KK=IBL+JBL*(JBL-1)/2
                      COEF=BLCOEE(KK)
                      EWEN=EWEN+COEF*KVEC(J)*( &
                           BSUM(2,J,IB)*BSUM(2,J,JB) + &
                           BSUM(1,J,IB)*BSUM(1,J,JB))
                   ENDDO
                ENDDO
                EWEN=EWEN*TWO*CCELEC
             ELSE
                EWEN = TWO*CCELEC*KVEC(J)*(SUM(2,J)**2 + SUM(1,J)**2)
             ENDIF
#else /**/
             EWEN = TWO*CCELEC*KVEC(J)*(SUM(2,J)**2 + SUM(1,J)**2)
#endif /* close BLOCK*/
             !
             EWVIRIAL(1)=EWVIRIAL(1)+EWEN*(ONE-EWPR*KXR*KXR)    ! xx
             EWVIRIAL(2)=EWVIRIAL(2)-EWEN*EWPR*KXR*KYR          ! xy
             EWVIRIAL(3)=EWVIRIAL(3)-EWEN*EWPR*KXR*KZR          ! xz
             EWVIRIAL(4)=EWVIRIAL(4)-EWEN*EWPR*KXR*KYR          ! yx
             EWVIRIAL(5)=EWVIRIAL(5)+EWEN*(ONE-EWPR*KYR*KYR)    ! yy
             EWVIRIAL(6)=EWVIRIAL(6)-EWEN*EWPR*KYR*KZR          ! yz
             EWVIRIAL(7)=EWVIRIAL(7)-EWEN*EWPR*KXR*KZR          ! zx
             EWVIRIAL(8)=EWVIRIAL(8)-EWEN*EWPR*KYR*KZR          ! zy
             EWVIRIAL(9)=EWVIRIAL(9)+EWEN*(ONE-EWPR*KZR*KZR)    ! zz
          ENDIF
          !
       ENDDO
       !
    ELSE
#endif /* (many_sum)*/
       !
       DO J = 1,TOTK
          !
          KX = KXV(J)
          KY = KYV(J)
          KZ = KZV(J)
          SUMSIN = ZERO
          SUMCOS = ZERO
#if KEY_TSM==1
          SUMCOSR = ZERO
          SUMSINR = ZERO
          SUMCOSP = ZERO
          SUMSINP = ZERO
#endif /* close TSM*/
          !
#if KEY_BLOCK==1
          IF (QBLOCK) THEN
             DO IB=1,NBLOCK
                BSUMCOS(IB) = ZERO
                BSUMSIN(IB) = ZERO
             ENDDO
          ENDIF
#endif /* close BLOCK*/
          !
          DO I = 1,NAT
             II=I+ATFRST-1
             KTG1 = KTABZC(I,KZ)*KTABYC(I,KY)-KTABYS(I,KY)*KTABZS(I,KZ)
             KTG2 = KTABZC(I,KZ)*KTABYS(I,KY)+KTABYC(I,KY)*KTABZS(I,KZ)
             CT(I) = CG(II)*(KTABXC(I,KX)*KTG1 - KTABXS(I,KX)*KTG2)
             ST(I) = CG(II)*(KTABXS(I,KX)*KTG1 + KTABXC(I,KX)*KTG2)
             !
#if KEY_TSM==1
             IF(.NOT.LTSM) THEN
#endif
                !
#if KEY_BLOCK==1
                IF (QBLOCK) THEN
                   IBL=iblckp(II)
                   BSUMCOS(IBL) = BSUMCOS(IBL) + CT(I)
                   BSUMSIN(IBL) = BSUMSIN(IBL) + ST(I)
                ELSE
                   SUMCOS = SUMCOS + CT(I)
                   SUMSIN = SUMSIN + ST(I)
                ENDIF
#else /**/
                SUMCOS = SUMCOS + CT(I)
                SUMSIN = SUMSIN + ST(I)
#endif /* close BLOCK*/
                !
#if KEY_TSM==1
             ELSE
                IF(REACLS(I) == 1) THEN
                   SUMCOSR = SUMCOSR + CT(I)
                   SUMSINR = SUMSINR + ST(I)
                ELSE IF(PRODLS(I) == 1) THEN
                   SUMCOSP = SUMCOSP + CT(I)
                   SUMSINP = SUMSINP + ST(I)
                ELSE
                   SUMCOS = SUMCOS + CT(I)
                   SUMSIN = SUMSIN + ST(I)
                ENDIF
             ENDIF
#endif /*   close TSM*/
          ENDDO
          !
#if KEY_PARALLEL==1 /*sum_sum*/
          IF(NUMNOD > 1) THEN
             !
#if KEY_BLOCK==1
             IF (QBLOCK) THEN
                DO IB=1,NBLOCK
                   BSUML(1,IB)=BSUMCOS(IB)
                   BSUML(2,IB)=BSUMSIN(IB)
                ENDDO
                II = 2
             ELSE
                II=2
                SUML(1)=SUMSIN
                SUML(2)=SUMCOS
             ENDIF
#else /**/
             II=2
             SUML(1)=SUMSIN
             SUML(2)=SUMCOS
#endif /* close BLOCK*/
             !
#if KEY_TSM==1
             IF(LTSM) THEN
                II=6
                SUML(3)=SUMSINR
                SUML(4)=SUMCOSR
                SUML(5)=SUMSINP
                SUML(6)=SUMCOSP
             ENDIF
#endif
             TMERI(TIMEXTE)=TMERI(TIMEXTE)-ECLOCK()+TIMME1
             TIMME1 = ECLOCK()
             !
#if KEY_BLOCK==1
             IF (QBLOCK) THEN
                CALL GCOMBr(BSUML,II*NBLOCK)  ! MH2023: extend the gcomb interface for fix
             ELSE
                CALL GCOMB(SUML,II)
             ENDIF
#else /**/
             CALL GCOMB(SUML,II)
#endif /*  close BLOCK*/
             !
             TMERI(TIMGCOMM)=TMERI(TIMGCOMM)+ECLOCK()-TIMME1
             !
#if KEY_BLOCK==1
             IF (QBLOCK) THEN
                DO IB=1, NBLOCK
                   BSUMCOS(IB)=BSUML(1,IB)
                   BSUMSIN(IB)=BSUML(2,IB)
                ENDDO
             ELSE
                SUMSIN=SUML(1)
                SUMCOS=SUML(2)
             ENDIF
#else /**/
             SUMSIN=SUML(1)
             SUMCOS=SUML(2)
#endif /* close BLOCK*/
             !
#if KEY_TSM==1
             IF(LTSM) THEN
                SUMSINR=SUML(3)
                SUMCOSR=SUML(4)
                SUMSINP=SUML(5)
                SUMCOSP=SUML(6)
             ENDIF
#endif
          ENDIF
#endif /* (sum_sum)*/
          !
#if KEY_TSM==1
          IF(LTSM) THEN
             !           calculate hybrid summation terms
             SUMSINE = SUMSIN + (ONE-LAMBDA)*SUMSINR + LAMBDA*SUMSINP
             SUMCOSE = SUMCOS + (ONE-LAMBDA)*SUMCOSR + LAMBDA*SUMCOSP
             SUMSINRE = (ONE-LAMBDA)*(SUMSIN +SUMSINR)
             SUMCOSRE = (ONE-LAMBDA)*(SUMCOS +SUMCOSR)
             SUMSINPE = LAMBDA*(SUMSIN +SUMSINP)
             SUMCOSPE = LAMBDA*(SUMCOS +SUMCOSP)
          ENDIF
#endif /*  close TSM*/
          !
          ! H Kamberaj (January 2007): Modified for any box shape
          !
          KR1 = CRECIP(1,1)*KX + CRECIP(2,1)*KY + CRECIP(3,1)*KZ
          KR2 = CRECIP(1,2)*KX + CRECIP(2,2)*KY + CRECIP(3,2)*KZ
          KR3 = CRECIP(1,3)*KX + CRECIP(2,3)*KY + CRECIP(3,3)*KZ

          CCFKX =  KR1*KVEC(J)
          CCFKY =  KR2*KVEC(J)
          CCFKZ =  KR3*KVEC(J)

          DO I = 1, NAT
             II=I+ATFRST-1
#if KEY_TSM==1
             IF(.NOT.LTSM) THEN
#endif
                !
#if KEY_BLOCK==1
                IF (QBLOCK) THEN
                   FD = ZERO
                   IB=iblckp(II)
                   DO JB=1, NBLOCK
                      IBL = IB
                      JBL = JB
                      IF (JBL  <  IBL) THEN
                         KK=IBL
                         IBL=JBL
                         JBL=KK
                      ENDIF
                      KK=IBL+JBL*(JBL-1)/2
                      COEF=BLCOEE(KK)
                      FD = FD + COEF*( &
                           BSUMSIN(JB)*CT(I) - &
                           BSUMCOS(JB)*ST(I) )
                   ENDDO

                ELSE
                   FD = SUMSIN*CT(I) - SUMCOS*ST(I)
                ENDIF
#else /**/
                FD = SUMSIN*CT(I) - SUMCOS*ST(I)
#endif /* close BLOCK*/
                !
#if KEY_TSM==1
             ELSE
                IF(REACLS(I) == 1) THEN
                   FD = SUMSINRE*CT(I) - SUMCOSRE*ST(I)
                ELSEIF(PRODLS(I) == 1) THEN
                   FD = SUMSINPE*CT(I) - SUMCOSPE*ST(I)
                ELSE
                   FD = SUMSINE*CT(I) - SUMCOSE*ST(I)
                ENDIF
             ENDIF
#endif /*   close TSM*/
             DX(II) = DX(II) + CCFKX*FD
             DY(II) = DY(II) + CCFKY*FD
             DZ(II) = DZ(II) + CCFKZ*FD
             ! SAPATEL
#if KEY_CHEQ==1
             IF(QCG.and.CG(II) /= 0) THEN
                !
#if KEY_BLOCK==1
                IF (QBLOCK) THEN
                   IB=iblckp(II)
                   DO JB=1, NBLOCK
                      IBL = IB
                      JBL = JB
                      IF (JBL  <  IBL) THEN
                         KK=IBL
                         IBL=JBL
                         JBL=KK
                      ENDIF
                      KK=IBL+JBL*(JBL-1)/2
                      COEF=BLCOEE(KK)
                      DCH(II) = DCH(II) + &
                           COEF*FOUR*CCELEC*KVEC(J)*( &
                           BSUMCOS(JB)*CT(I) + &
                           BSUMSIN(JB)*ST(I) ) / CG(II)
                   ENDDO

                ELSE
                   DCH(II) = DCH(II) + FOUR * CCELEC * &
                        KVEC(J)*(SUMCOS*CT(I) + SUMSIN*ST(I)) / CG(II)
                ENDIF
#else /**/
                DCH(II) = DCH(II) + FOUR * CCELEC * &
                     KVEC(J)*(SUMCOS*CT(I) + SUMSIN*ST(I)) / CG(II)
#endif /*  close BLOCK*/
                !
             ENDIF
#endif /*    close CHEQ*/
             !
             ! SAPATEL
          ENDDO
          !
#if KEY_FLUCQ==1
          IF (QFLUC) THEN
             DO I = 1, NAT
                II=I+ATFRST-1
#if KEY_BLOCK==1
                IF (QBLOCK) THEN
                   IB=iblckp(II)
                   DO JB=1, NBLOCK
                      IBL = IB
                      JBL = JB
                      IF (JBL  <  IBL) THEN
                         KK=IBL
                         IBL=JBL
                         JBL=KK
                      ENDIF
                      KK=IBL+JBL*(JBL-1)/2
                      COEF=BLCOEE(KK)
                      IF (CG(II) /= ZERO) FQCFOR(II)=FQCFOR(II)+ &
                           COEF*FOUR*KVEC(J)*(BSUMCOS(JB)*CT(I)+ &
                           BSUMSIN(JB)*ST(I))/CG(II)

                   ENDDO
                ELSE
                   IF (CG(II) /= ZERO) FQCFOR(II)=FQCFOR(II)+ &
                        FOUR*KVEC(J)*(SUMCOS*CT(I)+SUMSIN*ST(I)) / CG(II)
                ENDIF
#else /**/
                IF (CG(II)  /=  ZERO) FQCFOR(II)=FQCFOR(II) + &
                     FOUR*KVEC(J)*(SUMCOS*CT(I) + SUMSIN*ST(I)) / CG(II)
#endif /*   close BLOCK*/

             ENDDO
          ENDIF
#endif /*   close FLUCQ*/
          !
#if KEY_PARALLEL==1
          IF(MYNOD == 0) THEN
#endif
             !
#if KEY_TSM==1
             IF(LTSM) THEN
                !             These are the terms that go into the TSM trajectory file.
                EELR = EELR + KVEC(J)*( &
                     (SUMCOSR*SUMCOSR + TWO*SUMCOSR*SUMCOS) + &
                     (SUMSINR*SUMSINR + TWO*SUMSINR*SUMSIN))
                EELP = EELP + KVEC(J)*( &
                     (SUMCOSP*SUMCOSP + TWO*SUMCOSP*SUMCOS) + &
                     (SUMSINP*SUMSINP + TWO*SUMSINP*SUMSIN))
             ENDIF
#endif /*  close TSM*/
             !
#if KEY_BLOCK==1
             IF (QBLOCK) THEN
                DO IB=1, NBLOCK
                   DO JB=1, NBLOCK
                      IBL = IB
                      JBL = JB
                      IF (JBL  <  IBL) THEN
                         KK=IBL
                         IBL=JBL
                         JBL=KK
                      ENDIF
                      KK=IBL+JBL*(JBL-1)/2
                      COEF=BLCOEE(KK)
                      EEL=EEL+COEF*KVEC(J)*( &
                           BSUMCOS(IB)*BSUMCOS(JB) + &
                           BSUMSIN(IB)*BSUMSIN(JB) )
                   ENDDO
                ENDDO
             ELSE
                EEL = EEL + KVEC(J)*(SUMCOS**2 + SUMSIN**2)
             ENDIF
#else /**/
             EEL = EEL + KVEC(J)*(SUMCOS**2 + SUMSIN**2)
#endif /*  close BLOCK*/
             !
             !  Calculate the reciprocal space portion of the pressure tensor
             !  Code modified for any box shape (H Kamberaj, January 2007)
             !
             KXR = RECIP(1,1)*KX + RECIP(2,1)*KY + RECIP(3,1)*KZ
             KYR = RECIP(1,2)*KX + RECIP(2,2)*KY + RECIP(3,2)*KZ
             KZR = RECIP(1,3)*KX + RECIP(2,3)*KY + RECIP(3,3)*KZ
             KLEN = KXR**2 + KYR**2 + KZR**2
             EWPR = TWO/KLEN + TWO*((PI/KAPPA)**2)

#if KEY_BLOCK==1
             IF (QBLOCK) THEN
                EWEN = ZERO
                DO IB=1, NBLOCK
                   DO JB=1, NBLOCK
                      IBL = IB
                      JBL = JB
                      IF (JBL  <  IBL) THEN
                         KK=IBL
                         IBL=JBL
                         JBL=KK
                      ENDIF
                      KK=IBL+JBL*(JBL-1)/2
                      COEF=BLCOEE(KK)
                      EWEN=EWEN+COEF*KVEC(J)*( &
                           BSUMCOS(IB)*BSUMCOS(JB) + &
                           BSUMSIN(IB)*BSUMSIN(JB))
                   ENDDO
                ENDDO
                EWEN=EWEN*TWO*CCELEC
             ELSE
                EWEN = KVEC(J)*(SUMCOS**2+SUMSIN**2)*TWO*CCELEC
             ENDIF
#else /**/
             EWEN = KVEC(J)*(SUMCOS**2 + SUMSIN**2)*TWO*CCELEC
#endif /*   close BLOCK*/
             !
             EWVIRIAL(1)=EWVIRIAL(1)+EWEN*(ONE-EWPR*KXR*KXR)    ! xx
             EWVIRIAL(2)=EWVIRIAL(2)-EWEN*EWPR*KXR*KYR          ! xy
             EWVIRIAL(3)=EWVIRIAL(3)-EWEN*EWPR*KXR*KZR          ! xz
             EWVIRIAL(4)=EWVIRIAL(4)-EWEN*EWPR*KXR*KYR          ! yx
             EWVIRIAL(5)=EWVIRIAL(5)+EWEN*(ONE-EWPR*KYR*KYR)    ! yy
             EWVIRIAL(6)=EWVIRIAL(6)-EWEN*EWPR*KYR*KZR          ! yz
             EWVIRIAL(7)=EWVIRIAL(7)-EWEN*EWPR*KXR*KZR          ! zx
             EWVIRIAL(8)=EWVIRIAL(8)-EWEN*EWPR*KYR*KZR          ! zy
             EWVIRIAL(9)=EWVIRIAL(9)+EWEN*(ONE-EWPR*KZR*KZR)    ! zz
#if KEY_PARALLEL==1
          ENDIF
#endif
          !
       ENDDO
       !
#if KEY_PARALLEL==1 /*many_sum2*/
    ENDIF
#endif /* (many_sum2)*/
    !
    EKSUM = TWO*EEL*CCELEC
    !
#if KEY_TSM==1
    IF(LTSM) THEN
       !        assume linear lambda scaling
       EELR = TWO*CCELEC*EELR
       EELP = TWO*CCELEC*EELP
       !        returned the correct hybrid value.
       EKSUM = EKSUM + (ONE-LAMBDA)*EELR + LAMBDA*EELP
    ENDIF
#endif
    !
    call chmdealloc(file,routine,"ktabxc",NAT,KMX+1  ,crl=ktabxc)
    call chmdealloc(file,routine,"ktabxs",NAT,KMX+1  ,crl=ktabxs)
    call chmdealloc(file,routine,"ktabyc",NAT,2*KMY+1,crl=ktabyc)
    call chmdealloc(file,routine,"ktabys",NAT,2*KMY+1,crl=ktabys)
    call chmdealloc(file,routine,"ktabzc",NAT,2*KMZ+1,crl=ktabzc)
    call chmdealloc(file,routine,"ktabzs",NAT,2*KMZ+1,crl=ktabzs)
    call chmdealloc(file,routine,"sum",   2,ttk,         crl=sum)
    call chmdealloc(file,routine,"ct",nat,crl=ct)
    call chmdealloc(file,routine,"st",nat,crl=st)
    RETURN
  END subroutine KSPACEP2

end module ewald
