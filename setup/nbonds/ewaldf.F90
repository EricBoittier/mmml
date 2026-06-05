#if KEY_NOTINEWMOD==1
! This file needs to be checked for validity. MFC
! BRB will check it out before we decide to either convert it or to remove it.

SUBROUTINE KSPACE(EKSUM,LESELF,EQCOR,EUTIL, &
     QEKSUM,QESELF,QEQCOR,QEUTIL, &
     X,Y,Z,DX,DY,DZ,NATOM,CG,CGTOT &
#if KEY_FLUCQ==1
     ,QFLUC,FQCFOR    & 
#endif
     )
  !
  !--
  !
  use chm_kinds
  use dimens_fcm
  use consta
  use number
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
#endif
#if KEY_GAMESSUK==1 || KEY_G09==1
#if KEY_SQUANTM==0
  use mndo97     
#endif
#endif 

#if KEY_SQUANTM==1
  use squantm    
#endif
  use pme_module,only:pme,qpme
  use ewald_1m,only:pkvec,pkxv,pkyv,pkzv,ewald_kv_clear,maxkv
#if KEY_PHMD==1
  use phmd,only:QPHMD ! PME-CPHMD -- Y Huang 2017
#endif /* phmd */

  implicit none
  !
  real(chm_real)  EKSUM,LESELF,EQCOR,EUTIL
  LOGICAL QEKSUM,QESELF,QEQCOR,QEUTIL
  real(chm_real)  X(*),Y(*),Z(*),DX(*),DY(*),DZ(*)
  real(chm_real)  CG(*),CGTOT
  INTEGER NATOM
#if KEY_FLUCQ==1
  LOGICAL QFLUC
  real(chm_real) FQCFOR(*)
#endif 
  !
#if KEY_MNDO97==1
  real(chm_real) KHARGE     
#endif
#if KEY_GAMESSUK==1 || KEY_G09==1
#if KEY_SQUANTM==0
  real(chm_real) KHARGE    
#endif
#endif 
  !
  real(chm_real) CGTOTT
  ! end
  !
  !
  real(chm_real) BOXL(3)
  LOGICAL QDONE,qlocal_dummy
  !
  ! begin
  !
  IF (NATOM <= 0) RETURN
  !
  ! namkh 08/08/04
  ! QM/MM-Ewald : Apply QM charge into total MM charge
#if KEY_QUANTUM==1 || KEY_MNDO97==1 || KEY_SQUANTM==1 || KEY_GAMESSUK==1 || KEY_G09==1 /*qmewald*/
#if KEY_QUANTUM==1
  CGTOTT = CGTOT
  IF(LQMEWD) CGTOT = CHAGTOT
#elif KEY_MNDO97==1
  KHARGE = REAL(IN2(65))
  CGTOTT = CGTOT
  IF(LQMEWD) CGTOT = CGMM+KHARGE
#elif KEY_SQUANTM==1
  CGTOTT = CGTOT
  IF(LQMEWD) CGTOT = CGMM+QMCHARGE(1)
#endif 
  !
  ! now take care of gradient from MNDO
  ! virial portion: already added into EWVIRIAL.
#if KEY_SQUANTM==1
  qlocal_dummy=QMFEP    
#endif
#if KEY_SQUANTM==0
  qlocal_dummy=.FALSE.  
#endif
  IF(LQMEWD) CALL GETGRDQ(NATOM,DX,DY,DZ,qlocal_dummy)
#endif /*  (qmewald)*/
  !
  IF(.NOT.QPME) THEN
     CALL SETUPK1(QDONE,BOXL)
     IF(.NOT.QDONE) THEN
        IF(MAXKV < TOTK) THEN
           IF(MAXKV > 0) call ewald_kv_clear
           MAXKV=TOTK
           call ewald_kv_allocate
        ENDIF
        CALL SETUPK2(BOXL)
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
#if KEY_LJPME == 1
          (/ 0.0_chm_real /), &
#endif
          ewvirial, kappa &
#if KEY_PHMD==1
        , QPHMD & ! PME-PHMD -- Y Huang 2017
#endif
#if KEY_MNDO97==1
        , .false. &
#endif
        , .false. & !DCM
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

SUBROUTINE SETUPK1(QDONE,BOXL)
  !
  !    Setup wave-vector arrays for K-space ewald summation.
  !    Authors:
  !           Stephen H. Fleischman
  !           Roland Stote
  !           11/91
  use consta
  use inbnd
  use stream
  use image
  use pbound,only:qboun,qTOBoun,qRDBoun,qRHBoun,xsize,ysize,zsize
  use dimens_fcm
  use number
  use chm_kinds
  use param_store, only: set_param

  implicit none
  !
  LOGICAL QDONE
  real(chm_real)  BOXL(3)
  !
  !
  !
  !    *******************************************************************
  !    ** ROUTINE TO SET UP THE WAVE-VECTORS FOR THE EWALD SUM.         **
  !    **                                                               **
  !    *******************************************************************
  !
  real(chm_real)        TOL
  PARAMETER     (TOL = 1.0D-04)
  !
  INTEGER     KSQ, KX, KY, KZ, KSY, KSZ
  real(chm_real)      B, RKX, RKY, RKZ
  !    *******************************************************************
  !
  !
  ! Calculate box lengths
  if (qboun) then
     boxl(1) = xsize
     boxl(2) = ysize
     boxl(3) = zsize
  else
     BOXL(1)=XTLABC(1)
     BOXL(2)=XTLABC(3)
     BOXL(3)=XTLABC(6)
  endif
  IF(BOXL(1) <= ZERO.OR.BOXL(2).LE.ZERO.OR.BOXL(3).LE.ZERO) &
       CALL WRNDIE(-5,'<SETUPK>', &
       'One or more of the box sides was less than or equal to zero')
  ! Check to see if the box is orthorhombic (die otherwise)
  if (qboun .and. (qTOBoun .or. qRDBoun .or. qRHBoun)) then
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



SUBROUTINE SETUPK2(BOXL)
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
  use chm_kinds
  use dimens_fcm
  use number
  use consta
  use inbnd
  use stream
  use image
  use pbound,only:qboun,boxinv,boyinv,bozinv,pbound_getvol
  use ewald_1m,only:kvec=>pkvec,kxv=>pkxv,kyv=>pkyv,kzv=>pkzv
  use prssre
  implicit none
  !
  real(chm_real) BOXL(3)
  !
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
  real(chm_real)      TOL
  PARAMETER   (TOL = 1.0D-04)
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

  IF (PRNLEV  >  6) THEN
     WRITE(OUTU,*) VOL
  ENDIF

  if (qboun) then
     ! APH: Assumes orthorhombic box
     call pbound_getvol(vol)
     xtlinv(1) = boxinv
     xtlinv(2) = zero
     xtlinv(3) = boyinv
     xtlinv(4) = zero
     xtlinv(5) = zero
     xtlinv(6) = bozinv
     ok = .true.
  else
     ! Calculate the volume (H Kamberaj, November 2007)
     CALL GETVOL(VOL)
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
#if KEY_PARALLEL==1
  use parallel     
#endif
  use dimens_fcm
  use number
  use inbnd
  use chm_kinds
  implicit none
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
     BOXL, &
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
#if KEY_BLOCK==1
  use block_fcm         
#endif
  use image
  use pbound,only:qboun,boxinv,boyinv,bozinv
  use inbnd
  use machdep
  use number
  use consta
#if KEY_PARALLEL==1
  use parallel          
#endif
  use dimens_fcm
  !
  ! QM/MM-Ewald
  use quantm,only: lqmewd,qsetupkq,qcgsc
  use mndo97,only: lqmewd
#if KEY_GAMESSUK==1 || KEY_G09==1
#if KEY_SQUANTM==0
  use mndo97,only: lqmewd             
#endif
#endif 
#if KEY_SQUANTM==1
  use squantm,only: lqmewd,ewmode     
#endif

#if KEY_CHEQ==1
  use cheq,only:qcg,   &                  
#endif
#if KEY_CHEQ==1
     DCH,SUMDCH,DDCH                
#endif

  use chm_kinds
  use ewald_1m,only:kvec=>pkvec,kxv=>pkxv,kyv=>pkyv,kzv=>pkzv
  implicit none

#if KEY_CHEQ==1
  real(chm_real) HIJ    
#endif
  real(chm_real)  EKSUM,LESELF
  LOGICAL QEKSUM,QESELF
  real(chm_real)  X(*),Y(*),Z(*),DX(*),DY(*),DZ(*)
  INTEGER NATOM
  real(chm_real)  CG(*)
  INTEGER KMX,KMY,KMZ,NAT,ATFRST,TTK

  !mfc      real(chm_real) KTABXC(NAT,0:KMX),KTABXS(NAT,0:KMX)
  !mfc      real(chm_real) KTABYC(NAT,-KMY:KMY),KTABYS(NAT,-KMY:KMY)
  !mfc      real(chm_real) KTABZC(NAT,-KMZ:KMZ),KTABZS(NAT,-KMZ:KMZ)
  !mfc      real(chm_real) CT(NAT),ST(NAT),SUM(2,TTK)
  real(chm_real),allocatable,dimension(:,:) :: &
       ktabxc,ktabxs,ktabyc,ktabys,ktabzc,ktabzs,sum
  real(chm_real),allocatable,dimension(:) :: ct,st

  real(chm_real) BOXL(3)

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
  !
  ! LOCAL VARIABLES
  !
  real(chm_real)  EEL,ESLF,RPISQRT,CCONST
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
  PARAMETER  (CCONST=FOUR*TWOPI*CCELEC)


  call chmalloc('ewaldf.src','KSPACEP','KTABXC',NAT,  KMX+1,lbou2=0,   crl=KTABXC)
  call chmalloc('ewaldf.src','KSPACEP','KTABXS',NAT,  KMX+1,lbou2=0,   crl=KTABXS)
  call chmalloc('ewaldf.src','KSPACEP','KTABYC',NAT,2*KMY+1,lbou2=-kmy,crl=KTABYC)
  call chmalloc('ewaldf.src','KSPACEP','KTABYS',NAT,2*KMY+1,lbou2=-kmy,crl=KTABYS)
  call chmalloc('ewaldf.src','KSPACEP','KTABZC',NAT,2*KMZ+1,lbou2=-kmz,crl=KTABZC)
  call chmalloc('ewaldf.src','KSPACEP','KTABZS',NAT,2*KMZ+1,lbou2=-kmz,crl=KTABZS)
  call chmalloc('ewaldf.src','KSPACEP','SUM',2,TTK,crl=SUM)
  call chmalloc('ewaldf.src','KSPACEP','CT',NAT,crl=CT)
  call chmalloc('ewaldf.src','KSPACEP','ST',NAT,crl=ST)

  RPISQRT=ONE/SQRT(PI)
  !
  ! Set some zeros
#if KEY_PARALLEL==1
  TIMME1 = ECLOCK()        
#endif
  EEL=ZERO
  ESLF=ZERO
  !
#if KEY_QUANTUM==1 || KEY_MNDO97==1 || KEY_SQUANTM==1 || KEY_GAMESSUK==1 || KEY_G09==1
  IF (.NOT.LQMEWD) THEN    
#endif
     DO I = 1, 9
        EWVIRIAL(I) = ZERO
     END DO
#if KEY_QUANTUM==1 || KEY_MNDO97==1 || KEY_SQUANTM==1 || KEY_GAMESSUK==1 || KEY_G09==1
  END IF                   
#endif
  !
  !
  ! Calculate the volume and inverse of box cell
  ! (H Kamberaj, November 2007)
  !
 if (qboun) then
     ! APH: Assumes orthorhombic box
     xtlinv(1) = boxinv
     xtlinv(2) = zero
     xtlinv(3) = boyinv
     xtlinv(4) = zero
     xtlinv(5) = zero
     xtlinv(6) = bozinv
     ok = .true.
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
#if KEY_CHEQ==1
  HIJ = TWO*KAPPA*RPISQRT*CCELEC 
#endif
#if KEY_CHEQ==1
  IF(.not.QESELF) HIJ = ZERO     
#endif

  mainloop: DO I = 1,NAT
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
        IBL = IBLCKP(II)
        KK=IBL+IBL*(IBL-1)/2
        COEF = BLCOEP(KK)
        EN=EN*COEF
     ENDIF
#endif /*  close BLOCK*/
     !
#if KEY_TSM==1
     IF(.NOT.LTSM) THEN              
#endif
        ESLF = ESLF + EN
#if KEY_TSM==1
     ELSE
        IF(REACLS(II) == 1) THEN
           ESELFR = ESELFR + EN
        ELSEIF(PRODLS(II) == 1) THEN
           ESELFP = ESELFP + EN
        ELSE
           ESLF = ESLF + EN
        ENDIF
     ENDIF
#endif 

#if KEY_CHEQ==1 /*cheq1*/
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
#endif /* (cheq1)*/

#if KEY_FLUCQ==1
     IF (QFLUC.AND.QESELF) THEN
        EN = TWO*KAPPA*RPISQRT*CCELEC*CG(II)
#if KEY_BLOCK==1
        IF (QBLOCK) EN=EN*COEF    
#endif
        FQCFOR(II)=FQCFOR(II)-EN
     ENDIF
#endif /* close FLUCQ*/

  ENDDO mainloop

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
     call chmdealloc('ewaldf.src','KSPACEP','KTABXC',NAT,  KMX+1,lbou2=0,   crl=KTABXC)
     call chmdealloc('ewaldf.src','KSPACEP','KTABXS',NAT,  KMX+1,lbou2=0,   crl=KTABXS)
     call chmdealloc('ewaldf.src','KSPACEP','KTABYC',NAT,2*KMY+1,lbou2=-kmy,crl=KTABYC)
     call chmdealloc('ewaldf.src','KSPACEP','KTABYS',NAT,2*KMY+1,lbou2=-kmy,crl=KTABYS)
     call chmdealloc('ewaldf.src','KSPACEP','KTABZC',NAT,2*KMZ+1,lbou2=-kmz,crl=KTABZC)
     call chmdealloc('ewaldf.src','KSPACEP','KTABZS',NAT,2*KMZ+1,lbou2=-kmz,crl=KTABZS)
     call chmdealloc('ewaldf.src','KSPACEP','SUM',2,TTK,crl=SUM)
     call chmdealloc('ewaldf.src','KSPACEP','CT',NAT,crl=CT)
     call chmdealloc('ewaldf.src','KSPACEP','ST',NAT,crl=ST)

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
       ) then
     !
     ! Parallel version
     jloop: DO J = 1,TOTK
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
        iloop: DO I = 1,NAT
           II=I+ATFRST-1
           KTG1 = KTABZC(I,KZ)*KTABYC(I,KY)-KTABYS(I,KY)*KTABZS(I,KZ)
           KTG2 = KTABZC(I,KZ)*KTABYS(I,KY)+KTABYC(I,KY)*KTABZS(I,KZ)
           CT(I) = CG(II)*(KTABXC(I,KX)*KTG1 - KTABXS(I,KX)*KTG2)
           ST(I) = CG(II)*(KTABXS(I,KX)*KTG1 + KTABXC(I,KX)*KTG2)
           !
#if KEY_BLOCK==1
           IF (QBLOCK) THEN
              IBL = IBLCKP(II)
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
        ENDDO iloop
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
     ENDDO jloop

     TIMME1 = ECLOCK()

#if KEY_BLOCK==1
     IF (QBLOCK) THEN
        CALL GCOMB(BSUM,2*TOTK*NBLOCK)
     ELSE
        CALL GCOMB(SUM,2*TOTK)
     ENDIF
#else /**/
     CALL GCOMB(SUM,2*TOTK)
#endif /* close BLOCK*/

     TMERI(TIMGCOMM)=TMERI(TIMGCOMM)+ECLOCK()-TIMME1
     TIMME1 = ECLOCK()

     jloop2: DO J = 1,TOTK
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
              IB = IBLCKP(II)
              DO JB=1, NBLOCK
                 JBL = IB
                 JBL = JB
                 IF (JBL  <  IBL) THEN
                    KK=IBL
                    IBL=JBL
                    JBL=KK
                 ENDIF
                 KK=IBL+JBL*(JBL-1)/2
                 COEF = BLCOEP(KK)
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
                 IB = IBLCKP(II)
                 DO JB=1, NBLOCK
                    IBL = IB
                    JBL = JB
                    IF (JBL  <  IBL) THEN
                       KK=IBL
                       IBL=JBL
                       JBL=KK
                    ENDIF
                    KK=IBL+JBL*(JBL-1)/2
                    COEF = BLCOEP(KK)
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

#endif /* close CHEQ*/

#if KEY_FLUCQ==1
           IF (QFLUC.AND.CG(II) /= ZERO) THEN
#if KEY_BLOCK==1
              IF (QBLOCK) THEN
                 IB = IBLCKP(II)
                 DO JB=1, NBLOCK
                    IBL = IB
                    JBL = JB
                    IF (JBL  <  IBL) THEN
                       KK=IBL
                       IBL=JBL
                       JBL=KK
                    ENDIF
                    KK=IBL+JBL*(JBL-1)/2
                    COEF = BLCOEP(KK)
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
           ENDIF
#endif /* close FLUCQ*/
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
                    COEF = BLCOEP(KK)
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
                    COEF = BLCOEP(KK)
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
     ENDDO jloop2
     !
  ELSE
#endif /* (many_sum)*/
     !
     jloop3: DO J = 1,TOTK
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
                 IBL = IBLCKP(II)
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
              CALL GCOMB(BSUML,II*NBLOCK)
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
                 IB = IBLCKP(II)
                 DO JB=1, NBLOCK
                    IBL = IB
                    JBL = JB
                    IF (JBL  <  IBL) THEN
                       KK=IBL
                       IBL=JBL
                       JBL=KK
                    ENDIF
                    KK=IBL+JBL*(JBL-1)/2
                    COEF = BLCOEP(KK)
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
                 IB = IBLCKP(II)
                 DO JB=1, NBLOCK
                    IBL = IB
                    JBL = JB
                    IF (JBL  <  IBL) THEN
                       KK=IBL
                       IBL=JBL
                       JBL=KK
                    ENDIF
                    KK=IBL+JBL*(JBL-1)/2
                    COEF = BLCOEP(KK)
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
           ENDIF
#endif /*    close CHEQ*/
        ENDDO
#if KEY_FLUCQ==1
        IF (QFLUC) THEN
           DO I = 1, NAT
              II=I+ATFRST-1
#if KEY_BLOCK==1
              IF (QBLOCK) THEN
                 IB = IBLCKP(II)
                 DO JB=1, NBLOCK
                    IBL = IB
                    JBL = JB
                    IF (JBL  <  IBL) THEN
                       KK=IBL
                       IBL=JBL
                       JBL=KK
                    ENDIF
                    KK=IBL+JBL*(JBL-1)/2
                    COEF = BLCOEP(KK)
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

#if KEY_PARALLEL==1
        IF(MYNOD == 0) THEN
#endif 
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
                    COEF = BLCOEP(KK)
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
                    COEF = BLCOEP(KK)
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
     ENDDO jloop3
     !
#if KEY_PARALLEL==1 /*many_sum2*/
  ENDIF
#endif /* (many_sum2)*/
  EKSUM = TWO*EEL*CCELEC
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
  call chmdealloc('ewaldf.src','KSPACEP','KTABXC',NAT,  KMX+1,lbou2=0,   crl=KTABXC)
  call chmdealloc('ewaldf.src','KSPACEP','KTABXS',NAT,  KMX+1,lbou2=0,   crl=KTABXS)
  call chmdealloc('ewaldf.src','KSPACEP','KTABYC',NAT,2*KMY+1,lbou2=-kmy,crl=KTABYC)
  call chmdealloc('ewaldf.src','KSPACEP','KTABYS',NAT,2*KMY+1,lbou2=-kmy,crl=KTABYS)
  call chmdealloc('ewaldf.src','KSPACEP','KTABZC',NAT,2*KMZ+1,lbou2=-kmz,crl=KTABZC)
  call chmdealloc('ewaldf.src','KSPACEP','KTABZS',NAT,2*KMZ+1,lbou2=-kmz,crl=KTABZS)
  call chmdealloc('ewaldf.src','KSPACEP','SUM',2,TTK,crl=SUM)
  call chmdealloc('ewaldf.src','KSPACEP','CT',NAT,crl=CT)
  call chmdealloc('ewaldf.src','KSPACEP','ST',NAT,crl=ST)

  RETURN
END SUBROUTINE KSPACEP2
#endif /* NOTINEWMOD */
subroutine ewaldf_stub()
 return
end subroutine ewaldf_stub

