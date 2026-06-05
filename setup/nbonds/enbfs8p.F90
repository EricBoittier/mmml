module enbfs8pm
  use chm_kinds

  implicit none

#if KEY_FASTENBFS8==1 /*Fast_ENBFS8*/
  INTEGER,PARAMETER :: INTCRECSZ=1  ! record size of the integer cache
  INTEGER,PARAMETER :: REALCRECSZ=8 ! record size of the real(chm_real) cache

contains

Logical Function ENBFS8P(ENB,EEL,LELECX,LVDWX,IFRSTA,NATOMX, &
     CGX,JNBL,INBL,CCNBA,CCNBB,CCNBC,CCNBD, &
     IACNB,NITCC2,LOWTP, &
#if KEY_BLOCK==1
     IBLOCK,BLCOE,BLCOV,BLCOVR,BLCOVA,  & 
#endif
#if KEY_CHEQ==1
     ETA,FQINV,                         & 
#endif
#if KEY_FLUCQ==1
     QFLUC,FQCFOR,                      & 
#endif
     INTCACHE,REALCACHE, &
     LUSED,QRXNF)  ! GRF -- Wei Chen 2015
  !----------------------------------------------------------------------
  !     This is the performance optimized ENBFS8 targeted for 
  !     software pipelining architectures.  It applies very similar code
  !     tuning to that of the FASTEW version of subroutine REWALD.
  !     ENBFS8 is the fast scalar version of the nonboned energy terms
  !     for {constant dielectric} {electrostatic shifting} {vdW shifting}
  !         {distance dielectric} {electrostatic switch  } {vdW switch  }
  !     ENBFS8P implements selected combinations of these nonbond energy
  !     options and returns true if the cases it handles are being addressed;
  !     otherwise it returns false indicating that the vanilla ENBFS8 code
  !     should be used.  This design is intended to support incremental
  !     implementation of additional cases.
  !
  !     March, 2004   Roberto Gomperts SGI and Scott Brozell TSRI
  !-----------------------------------------------------------------------

#if KEY_CHEQ==1
  use cheq, only: qcg  
#endif
  use consta
  use dimens_fcm
  use number
  use coord
  use deriv
  use param
  use inbnd
#if KEY_BLOCK==1
  use block_fcm  
#endif
  use lambdam
#if KEY_PBOUND==1
  use pbound     
#endif
#if KEY_MTS==1
  use tbmts      
#endif
  use galgor
  ! namkh 01/20/04
#if KEY_GAMESS==1 || KEY_GAMESSUK==1 || KEY_MNDO97==1 || KEY_SQUANTM==1 || KEY_QCHEM==1 || KEY_QTURBO==1 || KEY_G09==1
  use gamess_fcm
#endif 

  implicit none

  real(chm_real)  ENB,EEL
  LOGICAL LELECX,LVDWX,QRXNF
  INTEGER IFRSTA,NATOMX,JNBL(*),INBL(*)
  real(chm_real)  CGX(*)
  real(chm_real)  CCNBA(*),CCNBB(*),CCNBC(*),CCNBD(*)
  INTEGER IACNB(*),NITCC2,LOWTP(*)
#if KEY_FLUCQ==1
  LOGICAL QFLUC
  real(chm_real) FQCFOR(*)
#endif 
  LOGICAL LUSED

#if KEY_BLOCK==1
  INTEGER IBL,JBL,KK, IBLOCK(*)
  real(chm_real)  BLCOE(*),BLCOV(*),BLCOVR(*),BLCOVA(*)
#if KEY_DOCK==1
  INTEGER KDOC
  real(chm_real)  DOCFI, DOCFJ
#endif 
!ldm
  real(chm_real) FALPHA
  INTEGER ILDM, JLDM
! LDM
#endif /*  BLOCK*/
#if KEY_PBOUND==1 /*pbound*/
  real(chm_real) CORR
#endif /*     (pbound)*/
#if KEY_MTS==1 /* MTS*/
  real(chm_real)  RR1,RR2,RR3
  real(chm_real)  SWF,SWFE
#endif 
#if KEY_CHEQ==1
  real(chm_real) HIJ,ETA(NATOMX,*)
  LOGICAL FQINV
#endif 
  !
#if KEY_BLOCK==1
  real(chm_real) ENEORG                                /*ldm*/
#endif
  INTEGER IVECT,JVECT,KVECT
  real(chm_real) CA,CC,CH,ENE,ENN
  real(chm_real) TF,TX,TY,TZ,DTX,DTY,DTZ
  real(chm_real) TFELEC,TFVDW
  real(chm_real) S2,TR2,TR6,FSW,DFSW,FSH
  real(chm_real) EADD,ON3,ON6,ONOFF2,OFF3,OFF4,OFF5,OFF6,R1,R3,DENOM, &
       ACOEF,BCOEF,CCOEF,DCOEF,COVER3,DOVER5,CONST,ENEVDW
  real(chm_real) RECOF3,RECOF6,OFDIF3,OFDIF6,ONOFF3,ONOFF6, &
       CR6,CR12,RJUNK3,RJUNK6,MIN2OF
  !
  real(chm_real) C2ONNB,C2OFNB,CTROF2,C4ROF2,RUL3,RUL12,RIJL,RIJU
  real(chm_real) CGF,CGT,CRXI,CRYI,CRZI
  INTEGER ITEMP,I,J,NPR,IACI
  LOGICAL ELECFG,LOUTER,RSHFT,RSWIT,CSHIFT,CFSWIT,CSHFT,CSWIT, &
       SWITCH,LVSW,LVSH,LVFSW,RSHIFT,RFSWIT
  INTEGER First

#if KEY_SOFTVDW==1
  INTEGER ifi,ifi1
  real(chm_real) rfimin,rdiff,s1,ediff,tfvdwn,enno
  real(chm_real) alfa, ct2, rc2,x1,emin,emax,beta,cho,rc2o,ecut
  logical qalter,qvdw
#endif 
  !
  ! Cache storage for this FAST version
  INTEGER INTCACHE(INTCRECSZ,*)
  real(chm_real)  REALCACHE(REALCRECSZ,*)
  INTEGER CACHEDNPR
  ! integer cache
  ! real(chm_real) cache
  integer,PARAMETER :: J_MEMBER = 1,S2_MEMBER = 1 ,CH_MEMBER = 2 ,TX_MEMBER = 3 &
       ,TY_MEMBER = 4 ,TZ_MEMBER = 5 ,LO_MEMBER = 6 ,CA_MEMBER = 7 ,CB_MEMBER = 8 
  !
  Logical SkipIt, DoHere
  !
  Integer JJ
  real(chm_real) DL, CB, CT
  !
  #if KEY_NBIPS==1 /*nbips_comm*/
  !WXW Long range potential using isotropic periodic sum (IPS)
        LOGICAL LEIPSX,LVIPSX
        real(chm_real) U1,U2,U4,U6R,U12R
        real(chm_real) PE,PVC,PVA,DPE,DPVC,DPVA
        real(chm_real) ENEP,ENEVC,ENEVA,ENBC
  #endif /*   (nbips_comm)*/
    logical DVIPS,DBEXP,DBEXPIPS
    real(chm_real) SIG,AKR,EXPAR,CKR,EXPCR,AKRC,EXPARC,CKRC,EXPCRC
  !===========================================
  !
  ! Check if this routine cannot be used because of active PREFs.
  ! (These prefs either introduce code complexity that prohibits
  ! software pipelining or merely have not yet been supported.)
  !
  SkipIt = .False.
#if KEY_PBOUND==1
  SkipIt = SkipIt .or. qBoun
#endif 
#if KEY_SOFTVDW==1
  SkipIt = SkipIt .or. qgamin
#endif 
#if KEY_MTS==1

  SkipIt = SkipIt .or. SLFG
#endif 
#if KEY_BLOCK==1
  ! This also excludes DOCK, LDM
  SkipIt = SkipIt .or. QBLOCK
#endif 
#if KEY_FLUCQ==1
  SkipIt = SkipIt .or. QFLUC
#endif 
#if KEY_CHEQ==1
  SkipIt = SkipIt .or. QCG
#endif 
  !
  SkipIt = SkipIt .or. (Lfma .and. (.not. lvdwx)) 
  !
  ENBFS8P = .not. SkipIt
  If(SkipIt) RETURN
  !
  ! OK, begin as in the original subroutine ENBFS8.
  !
  IF(QRXNF)THEN ! GRF -- Wei Chen 2015
    RC  = 1 / ctofnb
    RC3 = 1 / (ctofnb*ctofnb*ctofnb)
  ENDIF
  !
  First = IFRSTA
#if KEY_GENETIC==1
  If(qGA_Ener) then
     First = Int(ENB)
  endif
#endif 
  LUSED=.TRUE.
  ENB=ZERO
  EEL=ZERO
  ELECFG=(LELECX.AND.(EPS /= ZERO))
  IF (.NOT.(LVDWX.OR.ELECFG)) RETURN
  CGF=ZERO
  IF (ELECFG) CGF=CCELEC/EPS
  !
  ! Set flags for electrostatics options (6 options supported)
  RSHIFT= .NOT.LCONS .AND.      LSHFT .AND.      LFSWT .AND. ELECFG
  RFSWIT= .NOT.LCONS .AND. .NOT.LSHFT .AND.      LFSWT .AND. ELECFG
  RSHFT = .NOT.LCONS .AND.      LSHFT .AND. .NOT.LFSWT .AND. ELECFG
  RSWIT = .NOT.LCONS .AND. .NOT.LSHFT .AND. .NOT.LFSWT .AND. ELECFG
  CSHIFT=      LCONS .AND.      LSHFT .AND.      LFSWT .AND. ELECFG
  CFSWIT=      LCONS .AND. .NOT.LSHFT .AND.      LFSWT .AND. ELECFG
  CSHFT =      LCONS .AND.      LSHFT .AND. .NOT.LFSWT .AND. ELECFG
  CSWIT =      LCONS .AND. .NOT.LSHFT .AND. .NOT.LFSWT .AND. ELECFG
  !
#if KEY_NBIPS==1
  IF(LEIPS)THEN
    RSHIFT=.FALSE.
    RFSWIT=.FALSE.
    RSHFT=.FALSE.
    RSWIT=.FALSE.
    CSHIFT=.FALSE.
    CFSWIT=.FALSE.
    CSHFT=.FALSE.
    CSWIT=.FALSE.
  ENDIF
#endif
  IF(RSHIFT .OR. RFSWIT) THEN
     LUSED=.FALSE.
     RETURN
  ENDIF
  !
  LVFSW=      LVFSWT                   .AND. LVDWX
  LVSH = .NOT.LVFSWT .AND.      LVSHFT .AND. LVDWX
  LVSW = .NOT.LVFSWT .AND. .NOT.LVSHFT .AND. LVDWX
  SWITCH= RSWIT.OR.CSWIT.OR.LVSW
  !
  IF(LDBEXP)THEN
    LVFSW=.FALSE.
    LVSH=.FALSE.
    LVSW=.FALSE.
    SWITCH=.FALSE.
  ENDIF
#if KEY_NBIPS==1
  IF(LVIPS)THEN
    LVFSW=.FALSE.
    LVSH=.FALSE.
    LVSW=.FALSE.
    SWITCH=.FALSE.
  ENDIF
  DBEXP=LDBEXP.AND..NOT.LVIPS
  DBEXPIPS=LDBEXP.AND.LVIPS
  DVIPS=LVIPS.AND..NOT.LDBEXP
  #endif
  #if KEY_DEBUG==1
  write(outu,'(9(a,l4))') ' RFSWIT=',RFSWIT,' RSHFT=',RSHFT, &
       ' CSHIFT=',CSHIFT,' CFSWIT=',CFSWIT,' CSHFT=',CSHFT, &
       ' CSWIT=',CSWIT
  write(outu,*) ' LVFSW=',LVFSW,' LVSH=',LVSH,' LVSW=',LVSW
#endif 
  ! Check if the nonbond energy cases handled here are being requested.
  !
  ! Cases handled here:
  !
  !         If(LVSW .and. RSHFT) then
  !           ElseIf(LVSW .and. CSHFT) then
  !           ElseIf(LVSW .and. RSWIT) then
  !           ElseIf(LVSW .and. CSWIT) then
  !           ElseIf(LVFSW .and. CFSWIT) then
  !           ElseIf(LVFSW .and. CSHIFT) then
  !           ElseIf(LVIPS .and. LEIPS) then
  !           ElseIf(LDBEXP .and. LEIPS) then
  !           ElseIf(LDBEXP .LVIPS.and. LEIPS) then
  !         Endif
  !
  ENBFS8P = (LVSW .and. RSHFT) .or. (LVSW .and. CSHFT) .or. &
       (LVSW .and. RSWIT) .or. (LVSW .and. CSWIT) .or. &
       (LVFSW .and. CFSWIT) .or. (LVFSW .and. CSHIFT) .or. &
       (LVIPS .and. LEIPS) .or. (LDBEXP .and. LEIPS)  .or. &
       (LDBEXP .and. LVIPS .and. LEIPS)  
  IF(LEGROM.OR.LVGROM) ENBFS8P=.FALSE.


  If(.not.ENBFS8P) RETURN
  !
  ! OK, begin calculation using performance optimized code.
  ! 
#if KEY_DEBUG==1
  write(outu,*) ' Using FASTENBFS8 Non-Bonded (Vacuum) Routine'  
#endif /*   */
  !
  C2OFNB=CTOFNB*CTOFNB
  C2ONNB=CTONNB*CTONNB
  CTROF2=-ONE/C2OFNB
  C4ROF2=FOUR*CTROF2
  IF (CSHIFT) MIN2OF = MINTWO/CTOFNB
  !
  !      IF (SWITCH) THEN
  IF (CTOFNB > CTONNB) THEN
     RUL3=ONE/(C2OFNB-C2ONNB)**3
     RUL12=TWELVE*RUL3
  ENDIF
  !      ENDIF
  IF (CFSWIT) THEN
     !       force-based cdie switching coeffs
     IF(CTONNB  <  CTOFNB) THEN
        ONOFF2 = C2ONNB*C2OFNB
        ON3    = C2ONNB*CTONNB
        OFF3   = C2OFNB*CTOFNB
        OFF4   = C2OFNB*C2OFNB
        OFF5   = OFF3*C2OFNB
        DENOM  = ONE/(C2OFNB-C2ONNB)**3
        EADD   = (ONOFF2*(CTOFNB-CTONNB)-(OFF5-ON3*C2ONNB)/FIVE)* &
             EIGHT*DENOM
        ACOEF  = OFF4*(C2OFNB-THREE*C2ONNB)*DENOM
        BCOEF  = SIX*ONOFF2*DENOM
        COVER3 = -(C2ONNB+C2OFNB)*DENOM
        CCOEF  = THREE*COVER3
        DCOEF  = TWO*DENOM
        DOVER5 = DCOEF/FIVE
        CONST  = BCOEF*CTOFNB-ACOEF/CTOFNB+COVER3*OFF3+DOVER5*OFF5
     ELSE
        EADD  = -ONE/CTOFNB
     END IF
  ENDIF
  IF (LVFSW) THEN
     OFF3 = C2OFNB*CTOFNB
     OFF6 = OFF3*OFF3
     RECOF6 = ONE/OFF6
     IF(CTONNB  <  CTOFNB) THEN
        ON3 = C2ONNB*CTONNB
        ON6 = ON3*ON3
        RECOF3 = ONE/OFF3
        OFDIF6 = OFF6/(OFF6 - ON6)
        OFDIF3 = OFF3/(OFF3 - ON3)
        ONOFF6 = RECOF6/ON6
        ONOFF3 = RECOF3/ON3
     ELSE
        ONOFF6 = RECOF6*RECOF6
        ONOFF3 = RECOF6
     END IF
  END IF

  !
  !     DO VDW AND ELECTROSTATIC TERMS AS REQUESTED
  !
  IF (First > 1) THEN
     ITEMP=INBL(First-1)
  ELSE
     ITEMP=0
  ENDIF
  DO I=First,NATOMX
#if KEY_IMCUBES==1
     IF (lbycbim) THEN
        ITEMP=INBL(I+NATOMX)
     ENDIF
#endif 
     NPR=INBL(I)-ITEMP
     IF (NPR > 0) THEN
        IACI=IACNB(I)
        CGT=CGF*CGX(I)
        CRXI=X(I)
        CRYI=Y(I)
        CRZI=Z(I)
        DTX=ZERO
        DTY=ZERO
        DTZ=ZERO
        !
        ! Prologue loop: Gather r**2, jvect pointers, Delta x, y and z
        !                and cutoff mask. If possible avoid IF statements
        !
        ! Among others the following trick may be used:
        !
        !    If Louter == T, DL == 1
        !              == F, DL == 0
        !    Change if(louter) then
        !              Ene = A
        !           else
        !              Ene = B
        !           endif
        !    Into: Ene = DL*A + (1-DL)*B
        !
        CACHEDNPR=1
        DO J = 1,NPR
           KVECT=JNBL(ITEMP+J)
           JVECT=ABS(KVECT)
           !
           ! namkh 01/20/04
           ! When you using this part, be carefule. I might screw this part up.
           ! But, hopefully not.
#if KEY_GAMESS==1 || KEY_GAMESSUK==1 || KEY_MNDO97==1 || KEY_SQUANTM==1 || KEY_QCHEM==1 || KEY_QTURBO==1 || KEY_G09==1
           if(qmused) then
           IF((ABS(IGMSEL(I)) == 1.OR.ABS(IGMSEL(I)).EQ.2) .AND. &
                (ABS(IGMSEL(JVECT)) == 1.OR.ABS(IGMSEL(JVECT)).EQ.2) &
                .AND.QGMREM) GOTO 100
           endif
#endif 
           !
           TX=CRXI-X(JVECT)
           TY=CRYI-Y(JVECT)
           TZ=CRZI-Z(JVECT)
           S2=MAX(RSMALL,TX*TX+TY*TY+TZ*TZ)
           IF (S2 < C2OFNB) THEN
              IVECT=LOWTP(MAX(IACNB(JVECT),IACI))+IACNB(JVECT)+IACI
              CH=CGT*CGX(JVECT)
              IF(KVECT < 0) THEN
                 CH=CH*E14FAC
                 IVECT=IVECT+NITCC2
                 ! IPS skip 1-4 calculation
                 IF(LVIPS.AND.LEIPS)CYCLE
                 ! only EIPS makes 1-4 interaction zero
                 IF(DBEXP.AND.LEIPS)CH=ZERO
              ENDIF
              INTCACHE(J_MEMBER,CACHEDNPR) = JVECT

              REALCACHE(S2_MEMBER,CACHEDNPR) = S2
              REALCACHE(CH_MEMBER,CACHEDNPR) = CH
              REALCACHE(TX_MEMBER,CACHEDNPR) = TX
              REALCACHE(TY_MEMBER,CACHEDNPR) = TY
              REALCACHE(TZ_MEMBER,CACHEDNPR) = TZ
              !
              REALCACHE(LO_MEMBER,CACHEDNPR) = ZERO
              IF(S2 > C2ONNB) REALCACHE(LO_MEMBER,CACHEDNPR) = ONE
              REALCACHE(CA_MEMBER,CACHEDNPR) = CCNBA(IVECT)
              REALCACHE(CB_MEMBER,CACHEDNPR) = CCNBB(IVECT)
              CACHEDNPR = CACHEDNPR + 1
           ENDIF
100     ENDDO
#if KEY_DEBUG==1
        write(outu,'(1(a,l8))') 'i_M1=',i_M1
        write(outu,'(1(a,l8))') 'i_M2=',i_M2
        write(outu,'(1(a,l8))') 'i_M3=',i_M3
        write(outu,'(1(a,i8))') 'l_M1=',l_M1
        write(outu,'(1(a,i8))') 'l_M2=',l_M2
        write(outu,'(1(a,i8))') 'l_M3=',l_M3
        write(outu,'(1(a,i8))') 'CACHEDNPR=',CACHEDNPR
#endif 
        CACHEDNPR = CACHEDNPR - 1
        !
        If(LVSW .AND. QRXNF) then ! GRF -- Wei Chen 2015

           DO JJ=1,CACHEDNPR
              S2 = REALCACHE(S2_MEMBER,JJ)
              CH = REALCACHE(CH_MEMBER,JJ)
              TX = REALCACHE(TX_MEMBER,JJ)
              TY = REALCACHE(TY_MEMBER,JJ)
              TZ = REALCACHE(TZ_MEMBER,JJ)
              DL = REALCACHE(LO_MEMBER,JJ)
              !
              TFELEC=ZERO
              TFVDW=ZERO
              !
              TR2=ONE/S2
              TR6=TR2*TR2*TR2
              !
              CA=REALCACHE(CA_MEMBER,JJ)*TR6*TR6
              CB=REALCACHE(CB_MEMBER,JJ)
              !
              !     Van der Waals
              !
              RIJL=C2ONNB-S2
              RIJU=C2OFNB-S2
              FSW=RIJU*RIJU*(RIJU-THREE*RIJL)*RUL3
              DFSW=RIJL*RIJU*RUL12
              ENEVDW = CA-CB*TR6
              TF = DL*FSW+(ONE-DL)
              ENN=ENEVDW*TF
              TFVDW = DL*ENEVDW*DFSW-SIX*TR2*(ENN+CA*TF)
              !
              !     Electrostatic
              !
              R1 = ONE/SQRT(S2)
              ENE = CH*(R1 + (HALF*RFCON*RC3*S2) - ((ONE+HALF*RFCON)*RC))
              TFELEC = CH*(RFCON*RC3-R1*TR2)
              !
              ENB=ENB+ENN
              EEL=EEL+ENE
              TF=TFELEC+TFVDW
              !
              !------ End  energies
              !
              TX=TX*TF
              TY=TY*TF
              TZ=TZ*TF
              DTX=DTX+TX
              DTY=DTY+TY
              DTZ=DTZ+TZ
              REALCACHE(TX_MEMBER,JJ)=TX
              REALCACHE(TY_MEMBER,JJ)=TY
              REALCACHE(TZ_MEMBER,JJ)=TZ
           ENDDO

        !
        ElseIf(LVSW .and. CSWIT) then

           DO JJ=1,CACHEDNPR
              S2 = REALCACHE(S2_MEMBER,JJ)
              CH = REALCACHE(CH_MEMBER,JJ)
              TX = REALCACHE(TX_MEMBER,JJ)
              TY = REALCACHE(TY_MEMBER,JJ)
              TZ = REALCACHE(TZ_MEMBER,JJ)
              DL = REALCACHE(LO_MEMBER,JJ)
              !
              TFELEC=ZERO
              TFVDW=ZERO
              !      
              TR2=ONE/S2
              TR6=TR2*TR2*TR2
              !
              CA=REALCACHE(CA_MEMBER,JJ)*TR6*TR6
              CB=REALCACHE(CB_MEMBER,JJ)
              !
              !     Van der Waals
              !
              RIJL=C2ONNB-S2
              RIJU=C2OFNB-S2
              FSW=RIJU*RIJU*(RIJU-THREE*RIJL)*RUL3
              DFSW=RIJL*RIJU*RUL12
              ENEVDW = CA-CB*TR6
              TF = DL*FSW+(ONE-DL)
              ENN=ENEVDW*TF
              TFVDW = DL*ENEVDW*DFSW-SIX*TR2*(ENN+CA*TF)
              !
              !     Electrostatic
              !
              R1 = ONE/SQRT(S2)
              CH=CH*R1
              ENE= CH*TF
              TFELEC= -ENE*TR2+CH*DFSW*DL 
              !
              ENB=ENB+ENN
              EEL=EEL+ENE
              TF=TFELEC+TFVDW
              !
              !------ End  energies
              !
              TX=TX*TF
              TY=TY*TF
              TZ=TZ*TF
              DTX=DTX+TX
              DTY=DTY+TY
              DTZ=DTZ+TZ
              REALCACHE(TX_MEMBER,JJ)=TX
              REALCACHE(TY_MEMBER,JJ)=TY
              REALCACHE(TZ_MEMBER,JJ)=TZ
           ENDDO
           !
        ElseIf(LVSW .and. RSWIT) then

           DO JJ=1,CACHEDNPR
              S2 = REALCACHE(S2_MEMBER,JJ)
              CH = REALCACHE(CH_MEMBER,JJ)
              TX = REALCACHE(TX_MEMBER,JJ)
              TY = REALCACHE(TY_MEMBER,JJ)
              TZ = REALCACHE(TZ_MEMBER,JJ)
              DL = REALCACHE(LO_MEMBER,JJ)
              !
              TFELEC=ZERO
              TFVDW=ZERO
              !      
              TR2=ONE/S2
              TR6=TR2*TR2*TR2
              !
              CA=REALCACHE(CA_MEMBER,JJ)*TR6*TR6
              CB=REALCACHE(CB_MEMBER,JJ)
              !
              !     Van der Waals
              !
              RIJL=C2ONNB-S2
              RIJU=C2OFNB-S2
              FSW=RIJU*RIJU*(RIJU-THREE*RIJL)*RUL3
              DFSW=RIJL*RIJU*RUL12
              ENEVDW = CA-CB*TR6
              TF = DL*FSW+(ONE-DL)
              ENN=ENEVDW*TF
              TFVDW = DL*ENEVDW*DFSW-SIX*TR2*(ENN+CA*TF)
              !
              !     Electrostatic
              !
              CH=CH*TR2
              ENE=CH*TF
              TFELEC= -TWO*ENE*TR2+CH*DFSW*DL
              !
              ENB=ENB+ENN
              EEL=EEL+ENE
              TF=TFELEC+TFVDW
              !
              !------ End  energies
              !
              TX=TX*TF
              TY=TY*TF
              TZ=TZ*TF
              DTX=DTX+TX
              DTY=DTY+TY
              DTZ=DTZ+TZ
              REALCACHE(TX_MEMBER,JJ)=TX
              REALCACHE(TY_MEMBER,JJ)=TY
              REALCACHE(TZ_MEMBER,JJ)=TZ
           ENDDO
           !
        ElseIf(LVSW .and. CSHFT) then

           DO JJ=1,CACHEDNPR
              S2 = REALCACHE(S2_MEMBER,JJ)
              CH = REALCACHE(CH_MEMBER,JJ)
              TX = REALCACHE(TX_MEMBER,JJ)
              TY = REALCACHE(TY_MEMBER,JJ)
              TZ = REALCACHE(TZ_MEMBER,JJ)
              DL = REALCACHE(LO_MEMBER,JJ)
              !
              TFELEC=ZERO
              TFVDW=ZERO
              !
              TR2=ONE/S2
              TR6=TR2*TR2*TR2
              !
              CA=REALCACHE(CA_MEMBER,JJ)*TR6*TR6
              CB=REALCACHE(CB_MEMBER,JJ)
              !
              !     Van der Waals
              !
              RIJL=C2ONNB-S2
              RIJU=C2OFNB-S2
              FSW=RIJU*RIJU*(RIJU-THREE*RIJL)*RUL3
              DFSW=RIJL*RIJU*RUL12
              ENEVDW = CA-CB*TR6
              TF = DL*FSW+(ONE-DL)
              ENN=ENEVDW*TF
              TFVDW = DL*ENEVDW*DFSW-SIX*TR2*(ENN+CA*TF)
              !
              !     Electrostatic
              !
              R1 = ONE/SQRT(S2)
              FSH=ONE+S2*CTROF2
              CH=CH*R1*FSH
              ENE=CH*FSH
              TFELEC= -ENE*TR2+C4ROF2*CH

              ENB=ENB+ENN
              EEL=EEL+ENE
              TF=TFELEC+TFVDW
              !
              !------ End  energies
              !
              TX=TX*TF
              TY=TY*TF
              TZ=TZ*TF
              DTX=DTX+TX
              DTY=DTY+TY
              DTZ=DTZ+TZ
              REALCACHE(TX_MEMBER,JJ)=TX
              REALCACHE(TY_MEMBER,JJ)=TY
              REALCACHE(TZ_MEMBER,JJ)=TZ
           ENDDO
           !
        ElseIf(LVSW .and. RSHFT) then

           DO JJ=1,CACHEDNPR
              S2 = REALCACHE(S2_MEMBER,JJ)
              CH = REALCACHE(CH_MEMBER,JJ)
              TX = REALCACHE(TX_MEMBER,JJ)
              TY = REALCACHE(TY_MEMBER,JJ)
              TZ = REALCACHE(TZ_MEMBER,JJ)
              DL = REALCACHE(LO_MEMBER,JJ)
              !
              TFELEC=ZERO
              TFVDW=ZERO
              !      
              TR2=ONE/S2
              TR6=TR2*TR2*TR2
              !
              CA=REALCACHE(CA_MEMBER,JJ)*TR6*TR6
              CB=REALCACHE(CB_MEMBER,JJ)
              !
              !     Van der Waals
              !
              RIJL=C2ONNB-S2
              RIJU=C2OFNB-S2
              FSW=RIJU*RIJU*(RIJU-THREE*RIJL)*RUL3
              DFSW=RIJL*RIJU*RUL12
              ENEVDW = CA-CB*TR6
              TF = DL*FSW+(ONE-DL)
              ENN=ENEVDW*TF
              TFVDW = DL*ENEVDW*DFSW-SIX*TR2*(ENN+CA*TF) 
              !
              !     Electrostatic
              !
              FSH=ONE+S2*CTROF2
              CH=CH*TR2*FSH
              ENE=CH*FSH
              TFELEC= -TWO*ENE*TR2+C4ROF2*CH
              !
              ENB=ENB+ENN
              EEL=EEL+ENE
              TF=TFELEC+TFVDW
              !
              !------ End  energies
              !
              TX=TX*TF
              TY=TY*TF
              TZ=TZ*TF
              DTX=DTX+TX
              DTY=DTY+TY
              DTZ=DTZ+TZ
              REALCACHE(TX_MEMBER,JJ)=TX
              REALCACHE(TY_MEMBER,JJ)=TY
              REALCACHE(TZ_MEMBER,JJ)=TZ
           ENDDO
           !
        ElseIf(LVFSW .and. CFSWIT) then

           DO JJ=1,CACHEDNPR
              S2 = REALCACHE(S2_MEMBER,JJ)
              CH = REALCACHE(CH_MEMBER,JJ)
              TX = REALCACHE(TX_MEMBER,JJ)
              TY = REALCACHE(TY_MEMBER,JJ)
              TZ = REALCACHE(TZ_MEMBER,JJ)
              DL = REALCACHE(LO_MEMBER,JJ)
              !
              TFELEC=ZERO
              TFVDW=ZERO
              !      
              TR2=ONE/S2
              TR6=TR2*TR2*TR2
              !
              CB=REALCACHE(CB_MEMBER,JJ)
              CT=REALCACHE(CA_MEMBER,JJ)
              !
              !     Van der Waals
              !
              R1 = ONE/SQRT(S2)
              R3 = R1*TR2
              RJUNK6 = TR6-RECOF6
              RJUNK3 = R3-RECOF3
              CR12 = CT*OFDIF6*RJUNK6
              CR6  = CB*OFDIF3*RJUNK3
              CA=CT*TR6*TR6
              ENEVDW = CA-CB*TR6
              ENN = DL*(CR12*RJUNK6 - CR6*RJUNK3) + &
                   (ONE-DL)*(ENEVDW+CB*ONOFF3-CT*ONOFF6)
              TFVDW = DL*(TR2*(SIX*CR6*R3 - TWELVE*CR12*TR6)) + &
                   (ONE-DL)*(MINSIX*TR2*(ENEVDW+CA))
              !
              !     Electrostatic
              !
              ENE = DL* &
                   (CH*(R1*(ACOEF-S2*(BCOEF+S2*(COVER3+DOVER5*S2)))+CONST))+ &
                   (ONE-DL)*(CH*(R1+EADD))
              TFELEC = DL*(-CH*R1*(ACOEF*TR2+BCOEF+S2*(CCOEF+DCOEF*S2)))+ &
                   (ONE-DL)*(- CH*R1*TR2)
              !
              ENB=ENB+ENN
              EEL=EEL+ENE
              TF=TFELEC+TFVDW
              !
              !------ End  energies
              !
              TX=TX*TF
              TY=TY*TF
              TZ=TZ*TF
              DTX=DTX+TX
              DTY=DTY+TY
              DTZ=DTZ+TZ
              REALCACHE(TX_MEMBER,JJ)=TX
              REALCACHE(TY_MEMBER,JJ)=TY
              REALCACHE(TZ_MEMBER,JJ)=TZ
           ENDDO
           !
        ElseIf(LVFSW .and. CSHIFT) then

           DO JJ=1,CACHEDNPR
              S2 = REALCACHE(S2_MEMBER,JJ)
              CH = REALCACHE(CH_MEMBER,JJ)
              TX = REALCACHE(TX_MEMBER,JJ)
              TY = REALCACHE(TY_MEMBER,JJ)
              TZ = REALCACHE(TZ_MEMBER,JJ)
              DL = REALCACHE(LO_MEMBER,JJ)

              TFELEC=ZERO
              TFVDW=ZERO
              !      
              TR2=ONE/S2
              TR6=TR2*TR2*TR2
              !
              CB=REALCACHE(CB_MEMBER,JJ)
              CT=REALCACHE(CA_MEMBER,JJ)
              !
              !     Van der Waals
              !
              R1 = ONE/SQRT(S2)
              R3 = R1*TR2
              RJUNK6 = TR6-RECOF6
              RJUNK3 = R3-RECOF3
              CR12 = CT*OFDIF6*RJUNK6
              CR6  = CB*OFDIF3*RJUNK3
              CA=CT*TR6*TR6
              ENEVDW = CA-CB*TR6
              ENN = DL*(CR12*RJUNK6 - CR6*RJUNK3) + &
                   (ONE-DL)*(ENEVDW+CB*ONOFF3-CT*ONOFF6)
              TFVDW = DL*(TR2*(SIX*CR6*R3 - TWELVE*CR12*TR6)) + &
                   (ONE-DL)*(MINSIX*TR2*(ENEVDW+CA))
              !
              !     Electrostatic
              !
              CH=CH*R1
              ENE=CH*(ONE + S2*(MIN2OF*R1-CTROF2))
              TFELEC= - CH*(CTROF2 + TR2)
              !
              ENB=ENB+ENN
              EEL=EEL+ENE
              TF=TFELEC+TFVDW
              !
              !------ End  energies
              !
              TX=TX*TF
              TY=TY*TF
              TZ=TZ*TF
              DTX=DTX+TX
              DTY=DTY+TY
              DTZ=DTZ+TZ
              REALCACHE(TX_MEMBER,JJ)=TX
              REALCACHE(TY_MEMBER,JJ)=TY
              REALCACHE(TZ_MEMBER,JJ)=TZ
           ENDDO
           !
        !
#if KEY_NBIPS==1 /*nbips_elec*/
           !WXW Electrostatic potential using isotropic periodic sum (IPS)
            ELSE IF (DVIPS.AND.LEIPS) THEN
              DO JJ=1,CACHEDNPR
                S2 = REALCACHE(S2_MEMBER,JJ)
                CH = REALCACHE(CH_MEMBER,JJ)
                TX = REALCACHE(TX_MEMBER,JJ)
                TY = REALCACHE(TY_MEMBER,JJ)
                TZ = REALCACHE(TZ_MEMBER,JJ)
                DL = REALCACHE(LO_MEMBER,JJ)
                TFELEC=ZERO
                TFVDW=ZERO
                !      
                TR2=ONE/S2
                TR6=TR2*TR2*TR2
                !
                CB=REALCACHE(CB_MEMBER,JJ)
                CT=REALCACHE(CA_MEMBER,JJ)
                !
                !     Van der Waals
                !
                U2=S2/RIPS2
                U4=U2*U2
                U6R=ONE/U4/U2
                U12R=U6R*U6R
                !  L-J r6 term
                !   etr6=1/r6+a0+r2*(a1+r2*(a2+r2*(a3+r2*(a4+r4*(a5+a6*r4)))))
                !   detr6/dr*r1=-6/r6+r2*(d1+r2*(d2+r2*(d3+r2*(d4+r4*(d5+d6*r4)))))
                !
                PVC=U6R+AIPSVC(0)+U2*(AIPSVC(1)+U2*(AIPSVC(2)+U2*(AIPSVC(3) &
                     +U2*(AIPSVC(4)+U4*(AIPSVC(5)+U4*AIPSVC(6))))))-PIPSVCC
                DPVC=-SIX*U6R+U2*(BIPSVC(1)+U2*(BIPSVC(2)+U2*(BIPSVC(3) &
                     +U2*(BIPSVC(4)+U4*(BIPSVC(5)+U4*BIPSVC(6))))))
                !  L-J r12 term 
                !   etr12=1/r12+a0+r2*(a1+r2*(a2+r2*(a3+r4*(a4+r4*(a5+a6*r4)))))
                !   detr12/dr*r1=-6/r6+r2*(d1+r2*(d2+r2*(d3+r4*(d4+r4*(d5+d6*r4)))))
                !
                PVA=U12R+AIPSVA(0)+U2*(AIPSVA(1)+U2*(AIPSVA(2)+U2*(AIPSVA(3) &
                     +U4*(AIPSVA(4)+U4*(AIPSVA(5)+U4*AIPSVA(6))))))-PIPSVAC
                DPVA=-TWELVE*U12R+U2*(BIPSVA(1)+U2*(BIPSVA(2)+U2*(BIPSVA(3) &
                     +U4*(BIPSVA(4)+U4*(BIPSVA(5)+U4*BIPSVA(6))))))
                ENEVA =CT*RIPS12R
                ENEVC =-CB(IVECT)*RIPS6R
                ENN=ENEVA*PVA+ENEVC*PVC
                TFVDW=(ENEVA*DPVA+ENEVC*DPVC)*TR2
                !  Electrostatic IPS
                !   etr1=1/r+r2*(a1+r2*(a2+r2*(a3+r2*(a4+r2*(a5+a6*r2)))))
                !   detr1/dr*r=-1/r+r2*(d1+r2*(d2+r2*(d3+r2*(d4+r2*(d5+d6*r2)))))
                ! 
                U1=SQRT(U2)
                PE=ONE/U1+U2*(AIPSE(1)+U2*(AIPSE(2)+U2*(AIPSE(3) &
                 +U2*(AIPSE(4)+U2*(AIPSE(5)+U2*AIPSE(6))))))
                DPE=-ONE/U1+U2*(BIPSE(1)+U2*(BIPSE(2)+U2*(BIPSE(3) &
                 +U2*(BIPSE(4)+U2*(BIPSE(5)+U2*BIPSE(6))))))
                ENEP=CH*RIPSR
                ENE=ENEP*(PE-PIPSEC)
                TFELEC=ENEP*DPE*TR2
               !
                ENB=ENB+ENN
                EEL=EEL+ENE
                TF=TFELEC+TFVDW
                !
                !
                !------ End  energies
                !
                TX=TX*TF
                TY=TY*TF
                TZ=TZ*TF
                DTX=DTX+TX
                DTY=DTY+TY
                DTZ=DTZ+TZ
                REALCACHE(TX_MEMBER,JJ)=TX
                REALCACHE(TY_MEMBER,JJ)=TY
                REALCACHE(TZ_MEMBER,JJ)=TZ
              ENDDO
            ElseIf(DBEXP .and. LEIPS) then

              DO JJ=1,CACHEDNPR
               S2 = REALCACHE(S2_MEMBER,JJ)
               CH = REALCACHE(CH_MEMBER,JJ)
               TX = REALCACHE(TX_MEMBER,JJ)
               TY = REALCACHE(TY_MEMBER,JJ)
               TZ = REALCACHE(TZ_MEMBER,JJ)
               DL = REALCACHE(LO_MEMBER,JJ)
 
               TFELEC=ZERO
               TFVDW=ZERO
               !      
               TR2=ONE/S2
               TR6=TR2*TR2*TR2
               !
               CB=REALCACHE(CB_MEMBER,JJ)
               CT=REALCACHE(CA_MEMBER,JJ)
               !
               !     Van der Waals
               !
               SIG=CB
               IF(SIG>RSMALL)THEN
                EMIN=SIG*SIG/CT/FOUR
                SIG=(HALF*SIG/EMIN)**SIXTH
                R1 = SQRT(TR2)
                AKR=AIPSVA(0)/R1/SIG
                EXPAR=EXP(-AKR)
                ! energy
                ENEVA=AIPSVA(1)*EXPAR
                ! force 
                DPVA=BIPSVA(1)*EXPAR/SIG
                !
                CKR=AIPSVC(0)/R1/SIG
                EXPCR=EXP(-CKR)
                ! energy
                ENEVC=AIPSVC(1)*EXPCR
                ! force
                DPVC=BIPSVC(1)*EXPCR/SIG
                ENN=EMIN*(ENEVA-ENEVC)
                TFVDW=EMIN*(DPVA-DPVC)*R1
               ELSE 
                ENN=ZERO 
                TFVDW=ZERO 
               ENDIF
           !  Electrostatic IPS
           !   etr1=1/r+r2*(a1+r2*(a2+r2*(a3+r2*(a4+r2*(a5+a6*r2)))))
           !   detr1/dr*r=-1/r+r2*(d1+r2*(d2+r2*(d3+r2*(d4+r2*(d5+d6*r2)))))
           ! 
               U1=RIPS1R/R1
               U2=U1*U1
               PE=ONE/U1+U2*(AIPSE(1)+U2*(AIPSE(2)+U2*(AIPSE(3) &
                +U2*(AIPSE(4)+U2*(AIPSE(5)+U2*AIPSE(6))))))
               DPE=-ONE/U1+U2*(BIPSE(1)+U2*(BIPSE(2)+U2*(BIPSE(3) &
                +U2*(BIPSE(4)+U2*(BIPSE(5)+U2*BIPSE(6))))))
               ENEP=CH*RIPSR
               ENE=ENEP*(PE-PIPSEC)
               TFELEC=ENEP*DPE*TR2
       !
               ENB=ENB+ENN
               EEL=EEL+ENE
               TF=TFELEC+TFVDW
               !
               !------ End  energies
               !
               TX=TX*TF
               TY=TY*TF
               TZ=TZ*TF
               DTX=DTX+TX
               DTY=DTY+TY
               DTZ=DTZ+TZ
               REALCACHE(TX_MEMBER,JJ)=TX
               REALCACHE(TY_MEMBER,JJ)=TY
               REALCACHE(TZ_MEMBER,JJ)=TZ
              ENDDO
            !
            ElseIf(DBEXPIPS .and. LEIPS) then

              DO JJ=1,CACHEDNPR
                 S2 = REALCACHE(S2_MEMBER,JJ)
                 CH = REALCACHE(CH_MEMBER,JJ)
                 TX = REALCACHE(TX_MEMBER,JJ)
                 TY = REALCACHE(TY_MEMBER,JJ)
                 TZ = REALCACHE(TZ_MEMBER,JJ)
                 DL = REALCACHE(LO_MEMBER,JJ)
   
                 TFELEC=ZERO
                 TFVDW=ZERO
                 !      
                 TR2=ONE/S2
                 TR6=TR2*TR2*TR2
                 !
                 CB=REALCACHE(CB_MEMBER,JJ)
                 CT=REALCACHE(CA_MEMBER,JJ)
                 !
                 !     Van der Waals
                 !
                 R1 = ONE/SQRT(S2)
                 SIG=CB
                 IF(SIG>RSMALL)THEN
                  EMIN=SIG*SIG/CT/FOUR
                  SIG=(HALF*SIG/EMIN)**SIXTH
                  IF(SRIPS*R1*SIG>ONE)THEN
                    AKR=AIPSVA(0)/R1/SIG
                   EXPAR=EXP(-AKR)
                   !AKRC=AIPSVA(0)*RIPS/SIG
                   !EXPARC=EXP(-AKRC)
                   ! energy
                   !ENEVA=AIPSVA(1)*(EXPAR+EXPARC*(EXPARC/EXPAR+(TWELVE+SIX*EXPARC/AKRC)/AKRC/AKRC))
                   ENEVA=AIPSVA(1)*(EXPAR+AIPSVA(4)/EXPAR+AIPSVA(5))
                   ! force 
                   !DPVA=BIPSVA(1)*(EXPAR-EXPARC*EXPARC/EXPAR)/SIG
                   DPVA=BIPSVA(1)*(EXPAR-AIPSVA(4)/EXPAR)/SIG
                   !
                   CKR=AIPSVC(0)/R1/SIG
                   EXPCR=EXP(-CKR)
                   !CKRC=AIPSVC(0)*RIPS/SIG
                   !EXPCRC=EXP(-CKRC)
                   ! energy
                   !ENEVC=AIPSVC(1)*(EXPCR+EXPCRC*(EXPCRC/EXPCR+(TWELVE+SIX*EXPCRC/CKRC)/CKRC/CKRC))
                   ENEVC=AIPSVC(1)*(EXPCR+AIPSVC(4)/EXPCR+AIPSVC(5))
                   ! force
                   !DPVC=BIPSVC(1)*(EXPCR-EXPCRC*EXPCRC/EXPCR)/SIG
                   DPVC=BIPSVC(1)*(EXPCR-AIPSVC(4)/EXPCR)/SIG
                   ENN=EMIN*(ENEVA-ENEVC)
                   TFVDW=EMIN*(DPVA-DPVC)*R1
                  ELSE 
                   ENN=ZERO 
                   TFVDW=ZERO 
                  ENDIF
                ENDIF
                !  Electrostatic IPS
             !   etr1=1/r+r2*(a1+r2*(a2+r2*(a3+r2*(a4+r2*(a5+a6*r2)))))
             !   detr1/dr*r=-1/r+r2*(d1+r2*(d2+r2*(d3+r2*(d4+r2*(d5+d6*r2)))))
             ! 
                 U1=RIPS1R/R1
                 U2=U1*U1
                 PE=ONE/U1+U2*(AIPSE(1)+U2*(AIPSE(2)+U2*(AIPSE(3) &
                  +U2*(AIPSE(4)+U2*(AIPSE(5)+U2*AIPSE(6))))))
                 DPE=-ONE/U1+U2*(BIPSE(1)+U2*(BIPSE(2)+U2*(BIPSE(3) &
                  +U2*(BIPSE(4)+U2*(BIPSE(5)+U2*BIPSE(6))))))
                 ENEP=CH*RIPSR
                 ENE=ENEP*(PE-PIPSEC)
                 TFELEC=ENEP*DPE*TR2
         !
                 ENB=ENB+ENN
                 EEL=EEL+ENE
                 TF=TFELEC+TFVDW
                 !
                 !------ End  energies
                 !
                 TX=TX*TF
                 TY=TY*TF
                 TZ=TZ*TF
                 DTX=DTX+TX
                 DTY=DTY+TY
                 DTZ=DTZ+TZ
                 REALCACHE(TX_MEMBER,JJ)=TX
                 REALCACHE(TY_MEMBER,JJ)=TY
                 REALCACHE(TZ_MEMBER,JJ)=TZ
              ENDDO
              !
#endif /*  (nbips_elec)*/
 
        Else    ! The other cases: Should not be here.
           !
           CALL WRNDIE(-5,'<ENBFS8P>','Should not be here')
           !
        Endif ! LVFSW
        !
        !        Epilogue: Scatter the Forces
        !
        DO JJ=1,CACHEDNPR
           JVECT = INTCACHE(J_MEMBER,JJ)
           TX = REALCACHE(TX_MEMBER,JJ)
           TY = REALCACHE(TY_MEMBER,JJ)
           TZ = REALCACHE(TZ_MEMBER,JJ)
           !
           DX(JVECT)=DX(JVECT)-TX
           DY(JVECT)=DY(JVECT)-TY
           DZ(JVECT)=DZ(JVECT)-TZ
        ENDDO
        !
        !
        !     RESTORE i-TH COMPONENT OF FORCE IN THE ARRAY

        DX(I)=DX(I)+DTX
        DY(I)=DY(I)+DTY
        DZ(I)=DZ(I)+DTZ

     ENDIF  ! IF (NPR > 0)
     ITEMP=INBL(I)
  ENDDO
#if KEY_DEBUG==1
  write(outu,'(2(a,f9.4))') '752ENB=',ENB,'EEL=',EEL
#endif 
  !
  return
end Function ENBFS8P

#endif /* (Fast_ENBFS8)*/
end module enbfs8pm

