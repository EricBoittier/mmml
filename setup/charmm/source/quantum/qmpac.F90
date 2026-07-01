#if KEY_QUANTUM==1 /*qmpac_main*/
SUBROUTINE MNAINTGS (X,K)
  !
  use chm_kinds
  use number, only : one
  use quantm,only : A_int
  implicit none
  real(chm_real) X
  INTEGER K
  !***********************************************************************
  !
  !    AINTGS FORMS THE "A" INTEGRALS FOR THE OVERLAP CALCULATION.
  !
  !***********************************************************************
  !
  real(chm_real)  C,rinv_X
  INTEGER I
  !
  C=EXP(-X)
  rinv_X = one/X
  A_int(1)=C*rinv_X       ! /X
  do I=1,K
     A_int(I+1)=(A_int(I)*I+C)*rinv_X     ! /X
  end do
  RETURN
  !
END SUBROUTINE MNAINTGS

FUNCTION AABABC(IOCCA1,IOCCB1,IOCCA2,IOCCB2,NMOS) result(aababc_rtn)
  !
  use chm_kinds
  use sizes
  use number,only : zero
  use quantm,only : NMECI,XY_meci,OCCA_meci
  implicit none
  INTEGER NMOS
  INTEGER IOCCA1(NMOS), IOCCB1(NMOS), IOCCA2(NMOS), IOCCB2(NMOS)
  !***********************************************************************
  !*
  !* AABABC EVALUATES THE C.I. MATRIX ELEMENT FOR TWO MICROSTATES DIFFERING
  !*       BY BETA ELECTRON. THAT IS, ONE MICROSTATE HAS A BETA ELECTRON
  !*       IN PSI(I) WHICH, IN THE OTHER MICROSTATE IS IN PSI(J)
  !*
  !***********************************************************************
  real(chm_real) aababc_rtn
  !
  INTEGER IJ,I,J,K
  real(chm_real)  SUM
  !
  do I=1,NMOS
     IF(IOCCA1(I).NE.IOCCA2(I)) EXIT
  end do
  IJ=IOCCB1(I)
  do J=I+1,NMOS
     IF(IOCCA1(J).NE.IOCCA2(J)) EXIT
     IJ=IJ+IOCCA1(J)+IOCCB1(J)
  end do
  SUM=zero
  do K=1,NMOS
     SUM=SUM+(XY_meci(I,J,K,K)-XY_meci(I,K,J,K))*(IOCCA1(K)-OCCA_meci(K))+XY_meci(I,J,K,K)*(IOCCB1(K)-OCCA_meci(K))
  end do
  AABABC_rtn=SUM*((-1)**(IJ-(IJ/2)*2))
  RETURN
END FUNCTION AABABC

FUNCTION AABACD(IOCCA1, IOCCB1, IOCCA2, IOCCB2, NMOS) result(aabacd_rtn)
  !
  use chm_kinds
  use sizes
  use quantm,only : NMECI,XY_meci
  implicit none
  INTEGER NMOS
  INTEGER IOCCA1(NMOS), IOCCB1(NMOS), IOCCA2(NMOS), IOCCB2(NMOS)
  !***********************************************************************
  !*
  !* AABACD EVALUATES THE C.I. MATRIX ELEMENT FOR TWO MICROSTATES DIFFERING
  !*       BY TWO ALPHA MOS. ONE MICROSTATE HAS ALPHA ELECTRONS IN
  !*       M.O.S PSI(I) AND PSI(J) FOR WHICH THE OTHER MICROSTATE HAS
  !*       ELECTRONS IN PSI(K) AND PSI(L)
  !*
  !***********************************************************************
  real(chm_real) aabacd_rtn
  !
  INTEGER IJ,I,J,K,L
  !
  IJ=0
  do I=1,NMOS
     IF(IOCCA1(I) .LT. IOCCA2(I)) EXIT
  end do
  do J=I+1,NMOS
     IF(IOCCA1(J) .LT. IOCCA2(J)) EXIT
     IJ=IJ+IOCCA2(J)+IOCCB2(J)
  end do
  do K=1,NMOS
     IF(IOCCA1(K) .GT. IOCCA2(K)) EXIT
  end do
  do L=K+1,NMOS
     IF(IOCCA1(L) .GT. IOCCA2(L)) EXIT
     IJ=IJ+IOCCA1(L)+IOCCB1(L)
  end do
  IJ=IJ+IOCCB2(I)+IOCCB1(K)
  AABACD_RTN=(XY_meci(I,K,J,L)-XY_meci(I,L,K,J))*((-1)**(IJ-(IJ/2)*2))
  RETURN
END FUNCTION AABACD

FUNCTION AABBCD(IOCCA1, IOCCB1, IOCCA2, IOCCB2, NMOS) result(aabbcd_rtn)
  !
  use chm_kinds
  use sizes
  use quantm,only : NMECI,XY_meci,ISPQR_meci,IS_meci,ILOOP_meci,JLOOP_meci
  implicit none
  INTEGER NMOS
  INTEGER IOCCA1(NMOS), IOCCB1(NMOS), IOCCA2(NMOS), IOCCB2(NMOS)
  !***********************************************************************
  !*
  !* AABBCD EVALUATES THE C.I. MATRIX ELEMENT FOR TWO MICROSTATES DIFFERING
  !*       BY TWO SETS OF M.O.S. ONE MICROSTATE HAS AN ALPHA ELECTRON
  !*       IN PSI(I) AND A BETA ELECTRON IN PSI(K) FOR WHICH THE OTHER
  !*       MICROSTATE HAS AN ALPHA ELECTRON IN PSI(J) AND A BETA ELECTRON
  !*       IN PSI(L)
  !*
  !***********************************************************************
  real(chm_real) aabbcd_rtn
  !
  INTEGER IJ,I,J,K,L,M,IJMIN,IJMAX,KLMIN,KLMAX
  real(chm_real) XR
  !
  do I=1,NMOS
     IF(IOCCA1(I) .NE. IOCCA2(I)) EXIT
  end do
  do J=I+1,NMOS
     IF(IOCCA1(J) .NE. IOCCA2(J)) EXIT
  end do
  do K=1,NMOS
     IF(IOCCB1(K) .NE. IOCCB2(K)) EXIT
  end do
  do L=K+1,NMOS
     IF(IOCCB1(L) .NE. IOCCB2(L)) EXIT
  end do
  IF( I.EQ.K .AND. J.EQ.L .AND. IOCCA1(I).NE.IOCCB1(I)) THEN
     ISPQR_meci(ILOOP_meci,IS_meci)=JLOOP_meci
     IS_meci=IS_meci+1
  ENDIF
  IF(IOCCA1(I) .LT. IOCCA2(I)) THEN
     M=I
     I=J
     J=M
  ENDIF
  IF(IOCCB1(K) .LT. IOCCB2(K)) THEN
     M=K
     K=L
     L=M
  ENDIF
  XR=XY_meci(I,J,K,L)
  !#      WRITE(6,'(4I5,F12.6)')I,J,K,L,XR
  !
  !   NOW UNTANGLE THE MICROSTATES
  !
  IJ=1
  IF( I.GT.K .AND. J.GT.L .OR. I.LE.K .AND. J.LE.L) IJ=0
  M=J
  J=K
  K=M
  IF( I.GT.J ) IJ=IJ+IOCCA1(J)+IOCCB1(I)
  IF( K.GT.L ) IJ=IJ+IOCCA2(L)+IOCCB2(K)
  IF(I.NE.J)THEN
     IJMAX=MAX(I,J)
     IJMIN=MIN(I,J)
     do M=IJMIN,IJMAX
        IJ=IJ+IOCCB1(M)+IOCCA1(M)
     end do
  ENDIF
  IF(K.NE.L) THEN
     KLMIN=MIN(K,L)
     KLMAX=MAX(K,L)
     do M=KLMIN,KLMAX
        IJ=IJ+IOCCB2(M)+IOCCA2(M)
     end do
  ENDIF
  !
  !   IJ IN THE PERMUTATION NUMBER, .EQUIV. -1 IF IJ IS ODD.
  !
  AABBCD_rtn=XR*((-1)**(IJ-(IJ/2)*2))
  RETURN
END FUNCTION AABBCD

FUNCTION BABBBC(IOCCA1, IOCCB1, IOCCA2, IOCCB2, NMOS) result(babbbc_rtn)
  !
  use chm_kinds
  use number
  use sizes
  use quantm,only : NMECI,XY_meci,OCCA_meci
  implicit none
  INTEGER NMOS
  INTEGER IOCCA1(NMOS), IOCCB1(NMOS), IOCCA2(NMOS), IOCCB2(NMOS)
  !***********************************************************************
  !*
  !* BABBBC EVALUATES THE C.I. MATRIX ELEMENT FOR TWO MICROSTATES DIFFERING
  !*       BY ONE BETA ELECTRON. THAT IS, ONE MICROSTATE HAS A BETA
  !*       ELECTRON IN PSI(I) AND THE OTHER MICROSTATE HAS AN ELECTRON IN
  !*       PSI(J).
  !***********************************************************************
  real(chm_real) babbbc_rtn
  !
  INTEGER IJ,I,J,K
  real(chm_real)  SUM
  !
  do I=1,NMOS
     IF(IOCCB1(I).NE.IOCCB2(I)) EXIT
  end do
  IJ=0
  do J=I+1,NMOS
     IF(IOCCB1(J).NE.IOCCB2(J)) EXIT
     IJ=IJ+IOCCA1(J)+IOCCB1(J)
  end do
  IJ=IJ+IOCCA1(J)
  !
  !   THE UNPAIRED M.O.S ARE I AND J
  SUM=ZERO
  do K=1,NMOS
     SUM=SUM+ (XY_meci(I,J,K,K)-XY_meci(I,K,J,K))*(IOCCB1(K)-OCCA_meci(K))+XY_meci(I,J,K,K)*(IOCCA1(K)-OCCA_meci(K))
  end do
  BABBBC_rtn=SUM*((-1)**(IJ-(IJ/2)*2))
  RETURN
END FUNCTION BABBBC

FUNCTION BABBCD(IOCCA1, IOCCB1, IOCCA2, IOCCB2, NMOS) result(babbcd_rtn)
  !
  use chm_kinds
  use number
  use sizes
  use quantm,only : NMECI,XY_meci
  implicit none
  INTEGER NMOS
  INTEGER IOCCA1(NMOS), IOCCB1(NMOS), IOCCA2(NMOS), IOCCB2(NMOS)
  !***********************************************************************
  !*
  !* BABBCD EVALUATES THE C.I. MATRIX ELEMENT FOR TWO MICROSTATES DIFFERING
  !*       BY TWO BETA MOS. ONE MICROSTATE HAS BETA ELECTRONS IN
  !*       M.O.S PSI(I) AND PSI(J) FOR WHICH THE OTHER MICROSTATE HAS
  !*       ELECTRONS IN PSI(K) AND PSI(L)
  !*
  !***********************************************************************
  real(chm_real) babbcd_rtn
  !
  INTEGER IJ,I,J,K,L
  real(chm_real) AONE
  !
  IJ=0
  do I=1,NMOS
     IF(IOCCB1(I) .LT. IOCCB2(I)) EXIT
  end do
  do J=I+1,NMOS
     IF(IOCCB1(J) .LT. IOCCB2(J)) EXIT
     IJ=IJ+IOCCA2(J)+IOCCB2(J)
  end do
  IJ=IJ+IOCCA2(J)
  do K=1,NMOS
     IF(IOCCB1(K) .GT. IOCCB2(K)) EXIT
  end do
  do L=K+1,NMOS
     IF(IOCCB1(L) .GT. IOCCB2(L)) EXIT
     IJ=IJ+IOCCA1(L)+IOCCB1(L)
  end do
  IJ=IJ+IOCCA1(L)
  IF((IJ/2)*2.EQ.IJ) THEN
     AONE=ONE
  ELSE
     AONE=-ONE
  ENDIF
  BABBCD_rtn=(XY_meci(I,K,J,L)-XY_meci(I,L,J,K))*AONE
  RETURN
END FUNCTION BABBCD

SUBROUTINE MNANALYT(PSUM,PALPHA,PBETA,COORD,NAT,JJA,JJD,IIA,IID,NORB,ENG,KREP,ISTORE,IREAD)
  !
  use chm_kinds
  use number
  use sizes
  use am1parm,only:alpa=>alfa, &
       CORE,BETAS,BETAP,BETAD,ZS,ZP,ZD,natorb, &
       FN1,FN2,FN3,VS,VP,VD
  use consta
  use quantm, only: KEYWRD,F03_anal,ALP3_anal,BETA3_anal,NZTYPE_anal, &
       DS_ant,DG_ant,DR_ant,TDX_ant,TDY_ant,TDZ_ant, &
       G_ant,TX_ant,TY_ant, TZ_ant
  implicit none
  real(chm_real) COORD(3,*),ENG(3), PSUM(*), PALPHA(*), PBETA(*)
  INTEGER NAT(*)
  !************************************************************************
  !*                                                                      *
  !*         calculation of analytical derivatives                        *
  !*                                                                      *
  !************************************************************************
  !
  ! COMMON BLOCKS 'OWNED' BY REST OF PROGRAM.
  !
  ! ON RETURN, ENG HOLDS ANALYTICAL DERIVATIVES
  !
  integer :: IX
  real(chm_real) ISTORE(22,*)
  real(chm_real) EAA(3),EAB(3),ENUC(3), BI(4), BJ(4)
  !
  INTEGER IA,JA,KA,I22,ID,LA,JD,IG,II,KG,IJ,LG,JJ,KK,KL,LK,NI,LL, &
       MK,NJ,ML,NK,NL,MN,IS,NORB,KREP,I,J,K,L,M,N, &
       ISTART,JSTART,IIA,JJA,IID,JJD,IOL,ISP,I1,IREAD,J1,NBOND
  real(chm_real)  TERMK,TERML,AA,BB,DD,RR,DR1, &
       TERMAA,TERMAB,ANAM1,TERMNC, &
       RIJ,RRIJ,C1,C2,F3,ALPHA,R0,R2,DEL1,DEL2,DEL3
  real(chm_real) :: temp0,temp1,temp2,temp3
  real(chm_real), parameter :: A0=BOHRR, &
       RA0=one/BOHRR, RA02=one/(A0*A0) 
  !
  logical, save :: FIRST=.TRUE.
  logical, save :: AM1, MINDO3
  ! 
  IF( FIRST ) THEN
     AM1=(INDEX(KEYWRD,'AM1').NE.0).OR.(INDEX(KEYWRD,'PM3').NE.0)
     MINDO3=(INDEX(KEYWRD,'MINDO').NE.0)
     FIRST=.FALSE.
  ENDIF
  !
  JD=JJD-JJA+1
  JA=1
  ID=IID-IIA+1+JD
  IA=JD+1
  EAA(1:3) =ZERO
  EAB(1:3) =ZERO
  ENUC(1:3)=ZERO
  ENG(1:3) =ZERO

  I=2
  NI=NAT(I)
  ISTART=NZTYPE_anal(NI)*4-3
  J=1
  NJ=NAT(J)
  JSTART=NZTYPE_anal(NJ)*4-3
  R2=(COORD(1,I)-COORD(1,J))**2+(COORD(2,I)-COORD(2,J))**2 +(COORD(3,I)-COORD(3,J))**2
  RIJ=SQRT(R2)
  RRIJ=one/RIJ
  R0=RIJ*RA0     ! /A0
  RR=R2*RA02     ! /(A0*A0)
  lp150: DO IX=1,3
     DEL1=COORD(IX,I)-COORD(IX,J)
     TERMAA=ZERO
     TERMAB=ZERO
     ISP=0
     IOL=0
     !   THE FIRST DERIVATIVES OF OVERLAP INTEGRALS
     lp30: DO K=IA,ID
        KA=K-IA
        KG=ISTART+KA
        lp30a: DO L=JA,JD
           LA=L-JA
           LG=JSTART+LA
           IOL=IOL+1
           DS_ant(IOL)=ZERO
           IF(KA.EQ.0.AND.LA.EQ.0) THEN        ! (S/S) TERM
              IF(ABS(DEL1).LE.TENM6) cycle lp30a     ! 1.0D-6
              IS=1
           ELSEIF(KA.EQ.0.AND.LA.GT.0) THEN    ! (S/P) TERM
              IS=3
              if(IX.ne.LA) then
                 IF(ABS(DEL1).LE.TENM6) cycle lp30a     ! 1.0D-6
                 IS=2
                 DEL2=COORD(LA,I)-COORD(LA,J)
              end if
           ELSEIF(KA.GT.0.AND.LA.EQ.0) THEN    ! (P/S) TERM
              IS=5
              if(IX.ne.KA) then
                 IF(ABS(DEL1).LE.TENM6) cycle lp30a     ! 1.0D-6
                 IS=4
                 DEL2=COORD(KA,I)-COORD(KA,J)
              end if
           ELSE                                ! (P/P) TERM
              IF(KA.EQ.LA) THEN                   ! P/P
                 IS=9
                 if(IX.ne.KA) then
                    IF(ABS(DEL1).LE.TENM6) cycle lp30a  ! 1.0D-6
                    !                                !  P'/P'
                    IS=8
                    DEL2=COORD(KA,I)-COORD(KA,J)
                 end if
              ELSEIF(IX.NE.KA.AND.IX.NE.LA) THEN  ! P'/P"
                 IF(ABS(DEL1).LE.TENM6) IS=7          ! 1.0D-6
                 DEL2=COORD(KA,I)-COORD(KA,J)
                 DEL3=COORD(LA,I)-COORD(LA,J)
              ELSE                                ! P/P' OR P'/P
                 DEL2=COORD(KA+LA-IX,I)-COORD(KA+LA-IX,J)
                 IS=6
              ENDIF
           ENDIF
           !
           !        CALCULATE OVERLAP DERIVATIVES, STORE RESULTS IN DS
           !
           CALL MNDERS(KG,LG,RR,DEL1,DEL2,DEL3,IS,IOL)
        enddo lp30a
     enddo lp30
     IF(IX.EQ.1) G_ant(1:22) = ISTORE(1:22,IREAD)
     !
     IF(.NOT.MINDO3) CALL MNDELRI(DG_ant,NI,NJ,R0,DEL1)
     CALL MNDELMOL(COORD,I,J,NI,NJ,IA,ID,JA,JD,IX,RIJ,DEL1,ISP)
     !
     !   THE FIRST DERIVATIVE OF NUCLEAR REPULSION TERM
     IF(MINDO3)THEN
        II=MAX(NI,NJ)
        NBOND=(II*(II-1))/2+NI+NJ-II
        ALPHA=0
        IF(NBOND.LT.154)THEN
           ALPHA=ALP3_anal(NBOND)
        ELSE
           IF(NATORB(NI).EQ.0)ALPHA=ALPA(NI)
           IF(NATORB(NJ).EQ.0)ALPHA=ALPHA+ALPA(NJ)
        ENDIF
        C2=(7.1995D0/F03_anal(NI)+7.1995D0/F03_anal(NJ))**2
        C1=DEL1*RRIJ*CORE(NI)*CORE(NJ)*14.399D0         ! /RIJ
        IF(NBOND.EQ.22.OR.NBOND.EQ.29)THEN
           TERMNC=-C1*ALPHA*(RRIJ**2-RIJ*(RIJ**2+C2)**(-ONEPT5)+RRIJ-ONE/SQRT(RIJ**2+C2))*EXP(-RIJ) &
                -C1*RIJ*(RIJ**2+C2)**(-ONEPT5)    ! one/RIJ = rrij
        ELSEIF(RIJ.LT.ONE.AND.ALPHA.NE.ZERO)THEN
           TERMNC=ZERO
        ELSE
           TERMNC=-C1*((RRIJ)**2-RIJ*(RIJ**2+C2)**(-ONEPT5)+ALPHA*RRIJ-ALPHA/SQRT(RIJ**2+C2))*EXP(-ALPHA*RIJ) &
                -C1*RIJ*(RIJ**2+C2)**(-ONEPT5)
        ENDIF
        DR1=DEL1*RRIJ*14.399D0*RIJ*(RIJ**2+C2)**(-ONEPT5)
     ELSE
        !
        !      CORE-CORE TERMS, MNDO AND AM1
        !
        !
        !  SPECIAL TREATMENT FOR N-H AND O-H TERMS
        !
        IF(RIJ.LT.ONE.AND.NATORB(NI)*NATORB(NJ).EQ.0)THEN
           TERMNC=ZERO
           GOTO 50
        ENDIF
        IF(NI.EQ.1.AND.(NJ.EQ.7.OR.NJ.EQ.8)) THEN
           temp1=EXP(-ALPA(1)*RIJ)
           temp2=EXP(-ALPA(NJ)*RIJ)
           F3=ONE+temp1+RIJ*temp2
           DD=DG_ant(1)*F3-G_ant(1)*(DEL1*RRIJ)*(ALPA(1)*temp1+(ALPA(NJ)*RIJ-ONE)*temp2)
        ELSEIF((NI.EQ.7.OR.NI.EQ.8).AND.NJ.EQ.1) THEN
           temp1=EXP(-ALPA(1)*RIJ)
           temp2=EXP(-ALPA(NI)*RIJ)
           F3=ONE+temp1+RIJ*temp2
           DD=DG_ant(1)*F3-G_ant(1)*(DEL1*RRIJ)*(ALPA(1)*temp1+(ALPA(NI)*RIJ-ONE)*temp2)
        ELSE
           temp1=EXP(-ALPA(NI)*RIJ)
           temp2=EXP(-ALPA(NJ)*RIJ)
           F3=ONE+temp1+temp2
           DD=DG_ant(1)*F3-G_ant(1)*(DEL1*RRIJ)*(ALPA(NI)*temp1+ALPA(NJ)*temp2)
        ENDIF
        TERMNC=CORE(NI)*CORE(NJ)*DD
     ENDIF
     !
     !   ****   START OF THE AM1 SPECIFIC DERIVATIVE CODE   ***
     !
     !      ANALYT=-A*(1/(R*R)+TWO*B*(R-C)/R)*EXP(-B*(R-C)**2)
     !
     !    ANALYTICAL DERIVATIVES
     !
     IF( AM1 )THEN
        ANAM1=ZERO
        DO IG=1,10
           IF(ABS(FN1(NI,IG)).GT.ZERO) &
                ANAM1=ANAM1+FN1(NI,IG)*(RRIJ**2+TWO*FN2(NI,IG)*(RIJ-FN3(NI,IG))*RRIJ)* &
                EXP(MAX(-THIRTY,-FN2(NI,IG)*(RIJ-FN3(NI,IG))**2))
           IF(ABS(FN1(NJ,IG)).GT.ZERO) &
                ANAM1=ANAM1+FN1(NJ,IG)*(RRIJ**2+TWO*FN2(NJ,IG)*(RIJ-FN3(NJ,IG))*RRIJ)* &
                EXP(MAX(-THIRTY,-FN2(NJ,IG)*(RIJ-FN3(NJ,IG))**2))
        enddo
        ANAM1=ANAM1*CORE(NI)*CORE(NJ)
        TERMNC=TERMNC-ANAM1*DEL1*RRIJ
     ENDIF
     !
     !   ****   END OF THE AM1 SPECIFIC DERIVATIVE CODE   ***
     !
50   CONTINUE
     !
     !   COMBINE TOGETHER THE OVERLAP DERIVATIVE PARTS
     !
     IF(MINDO3)THEN
        II=MAX(NI,NJ)
        NBOND=(II*(II-1))/2+NI+NJ-II
        IF(NBOND <= 153)then
           BI(1)=BETA3_anal(NBOND)*VS(NI)*TWO
           BI(2)=BETA3_anal(NBOND)*VP(NI)*TWO
           BI(3)=BI(2)
           BI(4)=BI(2)
           BJ(1)=BETA3_anal(NBOND)*VS(NJ)*TWO
           BJ(2)=BETA3_anal(NBOND)*VP(NJ)*TWO
           BJ(3)=BJ(2)
           BJ(4)=BJ(2)
        endif
     ELSE
        BI(1)=BETAS(NI)
        BI(2)=BETAP(NI)
        BI(3)=BI(2)
        BI(4)=BI(2)
        BJ(1)=BETAS(NJ)
        BJ(2)=BETAP(NJ)
        BJ(3)=BJ(2)
        BJ(4)=BJ(2)
     ENDIF
     !
     !       CODE COMMON TO MINDO/3, MNDO, AND AM1
     !
     IOL=0
     kloop: DO K=IA,ID
        lloop: DO L=JA,JD
           LK=L+K*(K-1)/2
           TERMK=BI(K-IA+1)
           TERML=BJ(L-JA+1)
           IOL=IOL+1
           TERMAB=TERMAB+(TERMK+TERML)*PSUM(LK)*DS_ant(IOL)
        enddo lloop
     enddo kloop
     IF(MINDO3)THEN
        !
        !        FIRST, CORE-ELECTRON ATTRACTION DERIVATIVES (MINDO/3)
        !
        !          ATOM CORE I AFFECTING A.O.S ON J
        DO M=JA,JD
           MN=(M*(M+1))/2
           TERMAB=TERMAB+CORE(NI)*PSUM(MN)*DR1
        enddo
        !          ATOM CORE J AFFECTING A.O.S ON I
        DO M=IA,ID
           MN=(M*(M+1))/2
           TERMAB=TERMAB+CORE(NJ)*PSUM(MN)*DR1
        enddo
        !
        !   NOW FOR COULOMB AND EXCHANGE TERMS (MINDO/3)
        !
        lp100: DO I1=IA,ID
           II=(I1*(I1+1))/2
           lp100a: DO  J1=JA,JD
              JJ=(J1*(J1+1))/2
              IJ=J1+(I1*(I1-1))/2
              !
              !           COULOMB TERM
              !
              TERMAA=TERMAA-PSUM(II)*DR1*PSUM(JJ)
              !
              !           EXCHANGE TERM
              !
              TERMAA=TERMAA+(PALPHA(IJ)*PALPHA(IJ)+PBETA(IJ)*PBETA(IJ))*DR1
           enddo lp100a
        enddo lp100
     ELSE
        !
        !        FIRST, CORE-ELECTRON ATTRACTION DERIVATIVES (MNDO AND AM1)
        !
        !          ATOM CORE I AFFECTING A.O.S ON J
        ISP=0
        lp110: DO M=JA,JD
           BB=ONE
           lp110a: DO N=M,JD
              MN=M+N*(N-1)/2
              ISP=ISP+1
              TERMAB=TERMAB-BB*CORE(NI)*PSUM(MN)*DR_ant(ISP)
              BB=TWO
           enddo lp110a
        enddo lp110
        !          ATOM CORE J AFFECTING A.O.S ON I
        K=MAX(JD-JA+1,1)
        K=(K*(K+1))/2
        ISP=-K+1
        lp120: DO M=IA,ID
           BB=ONE
           lp120a: DO N=M,ID
              MN=M+N*(N-1)/2
              ISP=ISP+K
              TERMAB=TERMAB-BB*CORE(NJ)*PSUM(MN)*DR_ant(ISP)
              BB=TWO
           enddo lp120a
        enddo lp120
        ISP=0
        !
        !   NOW FOR COULOMB AND EXCHANGE TERMS (MNDO AND AM1)
        !
        lp140: DO K=IA,ID
           AA=ONE
           KK=(K*(K-1))/2
           lp140a: DO L=K,ID
              LL=(L*(L-1))/2
              lp130: DO M=JA,JD
                 BB=ONE
                 lp130a: DO N=M,JD
                    ISP=ISP+1
                    KL=K+LL
                    MN=M+N*(N-1)/2
                    !
                    !    COULOMB TERM
                    !
                    TERMAA=TERMAA+AA*BB*PSUM(KL)*PSUM(MN)*DR_ant(ISP)
                    MK=M+KK
                    NK=N+KK
                    ML=M+LL
                    NL=N+LL
                    !
                    !    EXCHANGE TERM
                    !
                    TERMAA=TERMAA-HALF*AA*BB*( PALPHA(MK)*PALPHA(NL)+PALPHA(NK)*PALPHA(ML)+PBETA(MK)*PBETA(NL) &
                         +PBETA(NK)*PBETA(ML) )*DR_ant(ISP)
                    BB=TWO
                 enddo lp130a
              enddo lp130
              AA=TWO
           enddo lp140a
        enddo lp140
        !           END OF MNDO AND AM1 SPECIFIC CODE
     ENDIF
     EAA(IX)=EAA(IX)+TERMAA
     EAB(IX)=EAB(IX)+TERMAB
     ENUC(IX)=ENUC(IX)+TERMNC
  enddo lp150
  !#            WRITE(6,*)EAA,EAB,ENUC,NAT(1),NAT(2)

  ENG(1:3) = -TWO*EV_TO_KCAL*(EAA(1:3)+EAB(1:3)+ENUC(1:3))      ! 23.061D0

  RETURN
END subroutine mnanalyt


SUBROUTINE MNBFN(X,BF)
  !
  use chm_kinds
  use number
  implicit none
  real(chm_real) X,BF(13)
  !**********************************************************************
  !
  !     BINTGS FORMS THE "B" INTEGRALS FOR THE OVERLAP CALCULATION.
  !
  !**********************************************************************
  real(chm_real) FACT(17)
  !
  INTEGER IO,LAST,I,K,M
  real(chm_real)  EXPMX,XF,ABSX,EXPX,Y
  !
  DATA FACT/1.D0,2.D0,6.D0,24.D0,120.D0,720.D0,5040.D0,40320.D0, &
       362880.D0,3628800.D0,39916800.D0,479001600.D0,6227020800.D0, &
       8.71782912E10,1.307674368E12,2.092278989E13,3.556874281E14/

  K    = 12
  IO   = 0
  ABSX = ABS(X)

  if(ABSX.GT.THREE) then
     EXPX=EXP(X)
     EXPMX=ONE/EXPX
     BF(1)=(EXPX-EXPMX)/X
     do i=1,k
        BF(I+1)=(I*BF(I)+(-ONE)**I*EXPX-EXPMX)/X
     end do
  else
     if(ABSX.LE.two) then
        IF (ABSX.LE.ONE) then
           IF (ABSX.LE.HALF) then
              IF (ABSX.LE.TENM6) then       ! 1.D-6
                 do i=IO,K
                    BF(I+1)=(2*MOD(I+1,2))/(I+ONE)
                 end do

                 return
              else
                 last=6
              end if
           else
              last=7
           end if
        else
           last=12
        end if
     else
        last=15
     end if

     do i=IO,K
        y=zero
        do m=IO,LAST
           XF=ONE
           IF(M.NE.0) XF=FACT(M)
           Y=Y+(-X)**M*(2*MOD(M+I+1,2))/(XF*(M+I+1))
        end do
        BF(I+1)=Y
     end do
  end if

  RETURN
  !
END SUBROUTINE MNBFN

SUBROUTINE MNBINTGS (X,K)
  !
  use chm_kinds
  use number
  use quantm,only : B_int
  implicit none
  real(chm_real) X
  INTEGER K
  real(chm_real) FACT(17)
  !**********************************************************************
  !
  !     BINTGS FORMS THE "B" INTEGRALS FOR THE OVERLAP CALCULATION.
  !
  !**********************************************************************
  !
  INTEGER IO,LAST,I,M
  real(chm_real)  EXPMX,XF,ABSX,EXPX,Y
  !
  DATA FACT/1.D0,2.D0,6.D0,24.D0,120.D0,720.D0,5040.D0,40320.D0, &
       362880.D0,3628800.D0,39916800.D0,479001600.D0,6227020800.D0, &
       8.71782912E10,1.307674368E12,2.092278989E13,3.556874281E14/
  IO=0
  ABSX = ABS(X)
  if(ABSX.GT.THREE) then
     goto 40
  else
     if(ABSX.LE.TWO) then
        if(ABSX.LE.ONE) then
           if(ABSX.LE.HALF) then 
              if(ABSX.LE.1.D-6) then
                 do I=IO,K
                    B_int(I+1)=(2*MOD(I+1,2))/(I+ONE)
                 end do

                 return
              else
                 last=6
              end if
           else
              if(K.LE.5) goto 40
              last=7
           end if
        else
           if(K.le.7) goto 40
           last=12
        end if
     else
        if(K.LE.10) goto 40
        last=15
     end if

     do I=IO,K
        Y=ZERO
        do M=IO,LAST
           XF=ONE
           IF(M.NE.0) XF=FACT(M)
           Y=Y+(-X)**M*(2*MOD(M+I+1,2))/(XF*(M+I+1))
        end do
        B_int(I+1)=Y
     end do

     return
  end if

40 EXPX=EXP(X)
  EXPMX=ONE/EXPX
  B_int(1)=(EXPX-EXPMX)/X
  do I=1,K
     B_int(I+1)=(I*B_int(I)+(-ONE)**I*EXPX-EXPMX)/X
  end do

  RETURN
END SUBROUTINE MNBINTGS

SUBROUTINE CNVG(PNEW, P, P1,NORBS, NITER, PL, UHF)
  !
  use chm_kinds
  use number
  implicit none
  real(chm_real) P1(*), P(*), PNEW(*)
  LOGICAL EXTRAP
  !***********************************************************************
  !
  !  CNVG IS A TWO-POINT INTERPOLATION ROUTINE FOR SPEEDING CONVERGENCE
  !       OF THE DENSITY MATRIX.
  !
  ! ON OUTPUT P      = NEW DENSITY MATRIX
  !           P1     = DIAGONAL OF OLD DENSITY MATRIX
  !           PL     = LARGEST DIFFERENCE BETWEEN OLD AND NEW DENSITY
  !                    MATRICES
  !***********************************************************************
  !
  INTEGER NITER,NORBS,IE,II,I,J,K
  real(chm_real)  SA,FACA,FACB,PL,DAMP,A,FAC
  !
  LOGICAL UHF
  real(chm_real)  RHFUHF
  real(chm_real) tempx
  !
  PL=ZERO
  FACA=ZERO
  !
#if KEY_SINGLE==1
  DAMP=1.E10
#else /**/
  DAMP=1.0D10
#endif 
  ! debug 
  IF(UHF) THEN
     RHFUHF=ONE
  ELSE
     RHFUHF=TWO
  END IF
  ! end
  IF(NITER.GT.3) DAMP=PT05
  FACB  =ZERO
  FAC   =ZERO
  II    =MOD(NITER,3)
  EXTRAP=II.NE.0
  K=0
  do I=1,NORBS
     K =K+I
     A =PNEW(K)
     SA=ABS(A-P(K))
     IF(SA.GT.PL) PL=SA
     if(.not.EXTRAP) then
        FACA=FACA+SA**2
        FACB=FACB+(A-TWO*P(K)+P1(I))**2
     end if
     P1(I)=P(K)
     P(K) =A
  end do
  !
  if( FACB.gt.ZERO .and. FACA.LT.(HUNDRD*FACB) ) FAC=SQRT(FACA/FACB)
  IE=0
  do I=1,NORBS
     II=I-1
     do J=1,II
        IE      =IE+1
        A       =PNEW(IE)
        P(IE)   =A+FAC*(A-P(IE))
        PNEW(IE)=P(IE)
     end do
     IE=IE+1
     IF(ABS(P(IE)-P1(I)) .GT. DAMP) THEN
        P(IE)=P1(I)+SIGN(DAMP,P(IE)-P1(I))
     ELSE
        P(IE)=P(IE)+FAC*(P(IE)-P1(I))
     ENDIF
     P(IE)=MIN(TWO,MAX(P(IE),ZERO))
     !
     PNEW(IE)=P(IE)
  end do
  !
  RETURN
END SUBROUTINE CNVG

SUBROUTINE MNCOE(X1,Y1,Z1,X2,Y2,Z2,PQ1,PQ2,C,R)
  !
  use chm_kinds
  use number
  implicit none
  INTEGER PQ1,PQ2,PQ,CO
  real(chm_real) C(75)
  !
  INTEGER I
  real(chm_real) CA,CB,SA,SB,XY,C2A,C2B,S2A,S2B,R,X1,X2,Y1,Y2,Z1,Z2

  !
  XY=(X2-X1)**2+(Y2-Y1)**2
  R =SQRT(XY+(Z2-Z1)**2)
  XY=SQRT(XY)
  IF (XY.LT.TENM10) then     ! 1.D-10
     if(Z2.lt.Z1) then  
        CA=-ONE
        CB=-ONE
        SA= ZERO
        SB= ZERO
     else if(Z2.eq.Z1) then 
        CA=ZERO
        CB=ZERO
        SA=ZERO
        SB=ZERO
     else if(Z2.gt.Z1) then 
        CA=ONE
        CB=ONE
        SA=ZERO
        SB=ZERO
     end if
  else
     CA=(X2-X1)/XY
     CB=(Z2-Z1)/R
     SA=(Y2-Y1)/XY
     SB=XY/R
  end if

  CO=0
  C(1:75) = zero
  if(PQ1.GT.PQ2) then
     PQ=PQ1
  else
     PQ=PQ2
  end if
  C(37)=ONE
  if(PQ.ge.2) then 
     C(56)= CA*CB
     C(41)= CA*SB
     C(26)=-SA
     C(53)=-SB
     C(38)= CB
     C(23)= ZERO
     C(50)= SA*CB
     C(35)= SA*SB
     C(20)= CA
     if(PQ.ge.3) then 
        C2A  = TWO*CA*CA-ONE
        C2B  = TWO*CB*CB-ONE
        S2A  = TWO*SA*CA
        S2B  = TWO*SB*CB
        C(75)= C2A*CB*CB+HALF*C2A*SB*SB
        C(60)= HALF*C2A*S2B
        C(45)= 0.8660254037841D0*C2A*SB*SB
        C(30)=-S2A*SB
        C(15)=-S2A*CB
        C(72)=-HALF*CA*S2B
        C(57)= CA*C2B
        C(42)= 0.8660254037841D0*CA*S2B
        C(27)=-SA*CB
        C(12)= SA*SB
        C(69)= 0.5773502691894D0*SB*SB*1.5D0
        C(54)=-0.8660254037841D0*S2B
        C(39)= CB*CB-HALF*SB*SB
        C(66)=-HALF*SA*S2B
        C(51)= SA*C2B
        C(36)= 0.8660254037841D0*SA*S2B
        C(21)= CA*CB
        C(6) =-CA*SB
        C(63)= S2A*CB*CB+HALF*S2A*SB*SB
        C(48)= HALF*S2A*S2B
        C(33)= 0.8660254037841D0*S2A*SB*SB
        C(18)= C2A*SB
        C(3) = C2A*CB
     end if
  end if
  RETURN
END SUBROUTINE MNCOE

SUBROUTINE MNDELMOL(COORD,I,J,NI,NJ,IA,ID,JA,JD,IX,RIJ,TOMB,ISP)
  !
  use chm_kinds
  use quantm, only : DS_ant,DG_ant,DR_ant, &
       TDX_ant,TDY_ant,TDZ_ant, &
       G_ant, TX_ant,TY_ant,TZ_ant
  implicit none
  real(chm_real) COORD(3,25)
  !
  INTEGER IA,IB,JA,JB,ID,JD,KK,NI,LL,NJ,MM,NN,IX,I,J,K,L,M,N,ISP
  real(chm_real)  TOMB,TEMP1,TEMP2,RIJ
  !
  IF(NI.GT.1.OR.NJ.GT.1) CALL MNROTAT(COORD,I,J,IX,RIJ,TOMB,2)
  IB=MAX(IA,ID)
  JB=MAX(JA,JD)
  lp10: DO K=IA,IB
     KK=K-IA
     lp10a: DO L=K,IB
        LL=L-IA
        lp10b: DO M=JA,JB
           MM=M-JA
           lp10c: DO N=M,JB
              NN=N-JA
              ISP=ISP+1
              IF(NN.EQ.0)THEN
                 IF(LL.EQ.0) THEN         ! (SS/SS)
                    DR_ant(ISP)=DG_ant(1)
                 ELSEIF(KK.EQ.0) THEN     ! (SP/SS)
                    DR_ant(ISP)=DG_ant(2)*TX_ant(LL)+G_ant(2)*TDX_ant(LL)
                 ELSE                     ! (PP/SS)
                    DR_ant(ISP)=DG_ant(3)*TX_ant(KK)*TX_ant(LL)+G_ant(3)* &
                         (TDX_ant(KK)*TX_ant(LL)+TX_ant(KK)*TDX_ant(LL)) &
                         +DG_ant(4)*(TY_ant(KK)*TY_ant(LL)+TZ_ant(KK)* &
                         TZ_ant(LL))+G_ant(4)*(TDY_ant(KK)*TY_ant(LL) &
                         +TY_ant(KK)*TDY_ant(LL)+TDZ_ant(KK)*TZ_ant(LL) &
                         +TZ_ant(KK)*TDZ_ant(LL))
                 ENDIF
              ELSEIF(MM.EQ.0) THEN
                 IF(LL.EQ.0) THEN         ! (SS/SP)
                    DR_ant(ISP)=DG_ant(5)*TX_ant(NN)+G_ant(5)*TDX_ant(NN)
                 ELSEIF(KK.EQ.0) THEN     ! (SP/SP)
                    DR_ant(ISP)=DG_ant(6)*TX_ant(LL)*TX_ant(NN) &
                         +G_ant(6)*(TDX_ant(LL)*TX_ant(NN)+TX_ant(LL)*TDX_ant(NN)) &
                         +DG_ant(7)*(TY_ant(LL)*TY_ant(NN)+TZ_ant(LL)*TZ_ant(NN)) &
                         +G_ant(7)*(TDY_ant(LL)*TY_ant(NN) &
                         +TY_ant(LL)*TDY_ant(NN)+TDZ_ant(LL)*TZ_ant(NN) &
                         +TZ_ant(LL)*TDZ_ant(NN))
                 ELSE                     ! (PP/SP)
                    DR_ant(ISP)=DG_ant(8)*TX_ant(KK)*TX_ant(LL)*TX_ant(NN) &
                         +G_ant(8)*(TDX_ant(KK)*TX_ant(LL)*TX_ant(NN) &
                         +TX_ant(KK)*TDX_ant(LL)*TX_ant(NN) &
                         +TX_ant(KK)*TX_ant(LL)*TDX_ant(NN)) &
                         +DG_ant(9)*(TY_ant(KK)*TY_ant(LL) &
                         +TZ_ant(KK)*TZ_ant(LL))*TX_ant(NN) &
                         +G_ant(9)*((TDY_ant(KK)*TY_ant(LL) &
                         +TY_ant(KK)*TDY_ant(LL) &
                         +TDZ_ant(KK)*TZ_ant(LL) &
                         +TZ_ant(KK)*TDZ_ant(LL))*TX_ant(NN) &
                         +(TY_ant(KK)*TY_ant(LL) &
                         +TZ_ant(KK)*TZ_ant(LL))*TDX_ant(NN)) &
                         +DG_ant(10)*(TX_ant(KK)*(TY_ant(LL)*TY_ant(NN) &
                         +TZ_ant(LL)*TZ_ant(NN)) &
                         +TX_ant(LL)*(TY_ant(KK)*TY_ant(NN) &
                         +TZ_ant(KK)*TZ_ant(NN))) &
                         +G_ant(10)*(TDX_ant(KK)*(TY_ant(LL)*TY_ant(NN) &
                         +TZ_ant(LL)*TZ_ant(NN)) &
                         +TDX_ant(LL)*(TY_ant(KK)*TY_ant(NN) &
                         +TZ_ant(KK)*TZ_ant(NN)) &
                         +TX_ant(KK)*(TDY_ant(LL)*TY_ant(NN) &
                         +TY_ant(LL)*TDY_ant(NN) &
                         +TDZ_ant(LL)*TZ_ant(NN) &
                         +TZ_ant(LL)*TDZ_ant(NN)) &
                         +TX_ant(LL)*(TDY_ant(KK)*TY_ant(NN) &
                         +TY_ant(KK)*TDY_ant(NN) &
                         +TDZ_ant(KK)*TZ_ant(NN)+TZ_ant(KK)*TDZ_ant(NN)))
                 ENDIF
              ELSEIF(LL.EQ.0) THEN         ! (SS/PP)
                 DR_ant(ISP)=DG_ant(11)*TX_ant(MM)*TX_ant(NN) &
                      +G_ant(11)*(TDX_ant(MM)*TX_ant(NN)+TX_ant(MM)*TDX_ant(NN)) &
                      +DG_ant(12)*(TY_ant(MM)*TY_ant(NN)+TZ_ant(MM)*TZ_ant(NN)) &
                      +G_ant(12)*(TDY_ant(MM)*TY_ant(NN) &
                      +TY_ant(MM)*TDY_ant(NN)+TDZ_ant(MM)*TZ_ant(NN) &
                      +TZ_ant(MM)*TDZ_ant(NN))
              ELSEIF(KK.EQ.0) THEN         ! (SP/PP)
                 DR_ant(ISP)=DG_ant(13)*TX_ant(LL)*TX_ant(MM)*TX_ant(NN) &
                      +G_ant(13)*(TDX_ant(LL)*TX_ant(MM)*TX_ant(NN) &
                      +TX_ant(LL)*TDX_ant(MM)*TX_ant(NN) &
                      +TX_ant(LL)*TX_ant(MM)*TDX_ant(NN)) &
                      +DG_ant(14)*TX_ant(LL)*(TY_ant(MM)*TY_ant(NN) &
                      +TZ_ant(MM)*TZ_ant(NN)) &
                      +G_ant(14)*(TDX_ant(LL)*(TY_ant(MM)*TY_ant(NN)+TZ_ant(MM)*TZ_ant(NN)) &
                      +TX_ant(LL)*(TDY_ant(MM)*TY_ant(NN)+TY_ant(MM)*TDY_ant(NN) &
                      +TDZ_ant(MM)*TZ_ant(NN)+TZ_ant(MM)*TDZ_ant(NN))) &
                      +DG_ant(15)*(TY_ant(LL)*(TY_ant(MM)*TX_ant(NN)+TY_ant(NN)*TX_ant(MM)) &
                      +TZ_ant(LL)*(TZ_ant(MM)*TX_ant(NN)+TZ_ant(NN)*TX_ant(MM))) &
                      +G_ant(15)*(TDY_ant(LL)*(TY_ant(MM)*TX_ant(NN)+TY_ant(NN)*TX_ant(MM)) &
                      +TDZ_ant(LL)*(TZ_ant(MM)*TX_ant(NN)+TZ_ant(NN)*TX_ant(MM)) &
                      +TY_ant(LL)*(TDY_ant(MM)*TX_ant(NN)+TY_ant(MM)*TDX_ant(NN) &
                      +TDY_ant(NN)*TX_ant(MM)+TY_ant(NN)*TDX_ant(MM)) &
                      +TZ_ant(LL)*(TDZ_ant(MM)*TX_ant(NN)+TZ_ant(MM)*TDX_ant(NN) &
                      +TDZ_ant(NN)*TX_ant(MM)+TZ_ant(NN)*TDX_ant(MM)))
              ELSE                         ! (PP/PP)
                 DR_ant(ISP)=DG_ant(16)*TX_ant(KK)*TX_ant(LL)*TX_ant(MM)*TX_ant(NN) &
                      +G_ant(16)*(TDX_ant(KK)*TX_ant(LL)*TX_ant(MM)*TX_ant(NN) &
                      +TX_ant(KK)*TDX_ant(LL)*TX_ant(MM)*TX_ant(NN) &
                      +TX_ant(KK)*TX_ant(LL)*TDX_ant(MM)*TX_ant(NN) &
                      +TX_ant(KK)*TX_ant(LL)*TX_ant(MM)*TDX_ant(NN)) &
                      +DG_ant(17)*(TY_ant(KK)*TY_ant(LL)+TZ_ant(KK)*TZ_ant(LL))*TX_ant(MM)*TX_ant(NN) &
                      +G_ant(17)*((TDY_ant(KK)*TY_ant(LL)+TY_ant(KK)*TDY_ant(LL) &
                      +TDZ_ant(KK)*TZ_ant(LL)+TZ_ant(KK)*TDZ_ant(LL))*TX_ant(MM)*TX_ant(NN) &
                      +(TY_ant(KK)*TY_ant(LL)+TZ_ant(KK)*TZ_ant(LL)) &
                      *(TDX_ant(MM)*TX_ant(NN)+TX_ant(MM)*TDX_ant(NN))) &
                      +DG_ant(18)*TX_ant(KK)*TX_ant(LL)*(TY_ant(MM)*TY_ant(NN)+TZ_ant(MM)*TZ_ant(NN)) &
                      +G_ant(18)*((TDX_ant(KK)*TX_ant(LL)+TX_ant(KK)*TDX_ant(LL)) &
                      *(TY_ant(MM)*TY_ant(NN)+TZ_ant(MM)*TZ_ant(NN)) &
                      +TX_ant(KK)*TX_ant(LL)*(TDY_ant(MM)*TY_ant(NN)+TY_ant(MM)*TDY_ant(NN) &
                      +TDZ_ant(MM)*TZ_ant(NN)+TZ_ant(MM)*TDZ_ant(NN)))
                 DR_ant(ISP)=DR_ant(ISP) &
                      +DG_ant(19)*(TY_ant(KK)*TY_ant(LL)*TY_ant(MM)*TY_ant(NN) &
                      +TZ_ant(KK)*TZ_ant(LL)*TZ_ant(MM)*TZ_ant(NN)) &
                      +G_ant(19)*(TDY_ant(KK)*TY_ant(LL)*TY_ant(MM)*TY_ant(NN) &
                      +TY_ant(KK)*TDY_ant(LL)*TY_ant(MM)*TY_ant(NN) &
                      +TY_ant(KK)*TY_ant(LL)*TDY_ant(MM)*TY_ant(NN) &
                      +TY_ant(KK)*TY_ant(LL)*TY_ant(MM)*TDY_ant(NN) &
                      +TDZ_ant(KK)*TZ_ant(LL)*TZ_ant(MM)*TZ_ant(NN) &
                      +TZ_ant(KK)*TDZ_ant(LL)*TZ_ant(MM)*TZ_ant(NN) &
                      +TZ_ant(KK)*TZ_ant(LL)*TDZ_ant(MM)*TZ_ant(NN) &
                      +TZ_ant(KK)*TZ_ant(LL)*TZ_ant(MM)*TDZ_ant(NN)) &
                      +DG_ant(20)*(TX_ant(KK)*(TX_ant(MM)*(TY_ant(LL) &
                      *TY_ant(NN)+TZ_ant(LL)*TZ_ant(NN)) &
                      +TX_ant(NN)*(TY_ant(LL)*TY_ant(MM)+TZ_ant(LL)*TZ_ant(MM))) &
                      +TX_ant(LL)*(TX_ant(MM)*(TY_ant(KK)*TY_ant(NN)+TZ_ant(KK)*TZ_ant(NN)) &
                      +TX_ant(NN)*(TY_ant(KK)*TY_ant(MM)+TZ_ant(KK)*TZ_ant(MM))))

                 !      TO AVOID COMPILER DIFFICULTIES THIS IS DIVIDED
                 TEMP1= TDX_ant(KK)*(TX_ant(MM)*(TY_ant(LL)*TY_ant(NN)+TZ_ant(LL)*TZ_ant(NN)) &
                      +TX_ant(NN)*(TY_ant(LL)*TY_ant(MM)+TZ_ant(LL)*TZ_ant(MM))) &
                      +TDX_ant(LL)*(TX_ant(MM)*(TY_ant(KK)*TY_ant(NN)+TZ_ant(KK)*TZ_ant(NN)) &
                      +TX_ant(NN)*(TY_ant(KK)*TY_ant(MM)+TZ_ant(KK)*TZ_ant(MM))) &
                      +TX_ant(KK)*(TDX_ant(MM)*(TY_ant(LL)*TY_ant(NN)+TZ_ant(LL)*TZ_ant(NN)) &
                      +TDX_ant(NN)*(TY_ant(LL)*TY_ant(MM)+TZ_ant(LL)*TZ_ant(MM))) &
                      +TX_ant(LL)*(TDX_ant(MM)*(TY_ant(KK)*TY_ant(NN)+TZ_ant(KK)*TZ_ant(NN)) &
                      +TDX_ant(NN)*(TY_ant(KK)*TY_ant(MM)+TZ_ant(KK)*TZ_ant(MM)))
                 TEMP2=TX_ant(KK)*(TX_ant(MM)*(TDY_ant(LL)*TY_ant(NN)+TY_ant(LL)*TDY_ant(NN) &
                      +TDZ_ant(LL)*TZ_ant(NN)+TZ_ant(LL)*TDZ_ant(NN)) &
                      +TX_ant(NN)*(TDY_ant(LL)*TY_ant(MM)+TY_ant(LL)*TDY_ant(MM) &
                      +TDZ_ant(LL)*TZ_ant(MM)+TZ_ant(LL)*TDZ_ant(MM))) &
                      +TX_ant(LL)*(TX_ant(MM)*(TDY_ant(KK)*TY_ant(NN)+TY_ant(KK)*TDY_ant(NN) &
                      +TDZ_ant(KK)*TZ_ant(NN)+TZ_ant(KK)*TDZ_ant(NN)) &
                      +TX_ant(NN)*(TDY_ant(KK)*TY_ant(MM)+TY_ant(KK)*TDY_ant(MM) &
                      +TDZ_ant(KK)*TZ_ant(MM)+TZ_ant(KK)*TDZ_ant(MM)))
                 DR_ant(ISP)=DR_ant(ISP)+G_ant(20)*(TEMP1+TEMP2)
                 DR_ant(ISP)=DR_ant(ISP) &
                      +DG_ant(21)*(TY_ant(KK)*TY_ant(LL)*TZ_ant(MM)*TZ_ant(NN) &
                      +TZ_ant(KK)*TZ_ant(LL)*TY_ant(MM)*TY_ant(NN)) &
                      +G_ant(21)*(TDY_ant(KK)*TY_ant(LL)*TZ_ant(MM)*TZ_ant(NN) &
                      +TY_ant(KK)*TDY_ant(LL)*TZ_ant(MM)*TZ_ant(NN) &
                      +TY_ant(KK)*TY_ant(LL)*TDZ_ant(MM)*TZ_ant(NN) &
                      +TY_ant(KK)*TY_ant(LL)*TZ_ant(MM)*TDZ_ant(NN) &
                      +TDZ_ant(KK)*TZ_ant(LL)*TY_ant(MM)*TY_ant(NN) &
                      +TZ_ant(KK)*TDZ_ant(LL)*TY_ant(MM)*TY_ant(NN) &
                      +TZ_ant(KK)*TZ_ant(LL)*TDY_ant(MM)*TY_ant(NN) &
                      +TZ_ant(KK)*TZ_ant(LL)*TY_ant(MM)*TDY_ant(NN))
                 DR_ant(ISP)=DR_ant(ISP) &
                      +DG_ant(22)*(TY_ant(KK)*TZ_ant(LL)+TZ_ant(KK)*TY_ant(LL)) &
                      *(TY_ant(MM)*TZ_ant(NN)+TZ_ant(MM)*TY_ant(NN)) &
                      +G_ant(22)*((TDY_ant(KK)*TZ_ant(LL)+TY_ant(KK)*TDZ_ant(LL) &
                      +TDZ_ant(KK)*TY_ant(LL)+TZ_ant(KK)*TDY_ant(LL)) &
                      *(TY_ant(MM)*TZ_ant(NN)+TZ_ant(MM)*TY_ant(NN)) &
                      +(TY_ant(KK)*TZ_ant(LL)+TZ_ant(KK)*TY_ant(LL)) &
                      *(TDY_ant(MM)*TZ_ant(NN)+TY_ant(MM)*TDZ_ant(NN) &
                      +TDZ_ant(MM)*TY_ant(NN)+TZ_ant(MM)*TDY_ant(NN)))
              END IF
           enddo lp10c
        enddo lp10b
     enddo lp10a
  enddo lp10
  RETURN
END SUBROUTINE MNDELMOL

SUBROUTINE MNDELRI(DG,NI,NJ,RR,DEL1)
  !
  use am1parm,only: DD,QQ,BDD,natorb
  use chm_kinds
  use number
  use consta
  use quantm, only : KEYWRD
  implicit none
  real(chm_real) DG(22)
  !***********************************************************************
  !                                                                      *
  !    on input ni = atomic number of first atom                         *
  !             nj = atomic number of second atom                        *
  !             rr = interatomic distance in bohrs                       *
  !                                                                      *
  !***********************************************************************
  !
  INTEGER NI,NJ
  real(chm_real) DA,DB,EE,QA,QB,DXQXZ,DZQXX,QXXDZ,QXZDX,DZQZZ,QZZDZ, &
       RR,DXDX,TERM,DZDZ,EQXX,QXXE,EQZZ,QZZE,ADD,ADE, &
       AED,AEE,ADQ,AQD,AEQ,AQE,AQQ,DZE,EDZ, &
       QXXQXX,QXXQYY,QXYQXY,QXXQZZ,QXZQXZ,QZZQXX,QZZQZZ,DEL1
  !
  real(chm_real),parameter :: RTWO= half, &       ! 0.5D0
       RFOUR=PT25, &       ! 0.25D0
       REIGH = PT125, &    ! 0.125D0
       RSIXT = 0.0625D0, & !
       RTHIR2= 0.03125D0, &
       A0=BOHRR
  logical, save :: FIRST=.TRUE.
  logical, save :: MINDO3
  !
  IF(FIRST)THEN
     FIRST=.TRUE.
     MINDO3=(INDEX(KEYWRD,'MINDO').NE.0)
  ENDIF

  TERM=(AU_TO_EV*DEL1)/(RR*A0*A0)    ! 27.21D0
  !
  DA=DD(NI)
  DB=DD(NJ)
  QA=QQ(NI)
  QB=QQ(NJ)
  !   HYDROGEN-HYDROGEN
  AEE   = RFOUR*(ONE/BDD(NI,1)+ONE/BDD(NJ,1))**2
  EE    =-RR/(SQRT(RR**2+AEE))**3
  DG(1) =TERM*EE
  IF(NATORB(NI).LE.2.AND.NATORB(NJ).LE.2) RETURN
  if(NATORB(NI).gt.2) then        ! IF(NATORB(NI).LE.2) GO TO 10
     !   HEAVY ATOM-HYDROGEN
     ADE= RFOUR*(ONE/BDD(NI,2)+ONE/BDD(NJ,1))**2
     AQE= RFOUR*(ONE/BDD(NI,3)+ONE/BDD(NJ,1))**2
     DZE   = (RR+DA)/(SQRT((RR+DA)**2+ADE))**3-(RR-DA)/(SQRT((RR-DA)**2+ADE))**3
     QZZE  =-(RR+TWO*QA)/(SQRT((RR+TWO*QA)**2+AQE))**3-(RR-TWO*QA)/(SQRT((RR-TWO*QA)**2+AQE))**3 &
          +(TWO*RR)/(SQRT(RR**2+AQE))**3
     QXXE  =-(TWO*RR)/(SQRT(RR**2+FOUR*QA**2+AQE))**3+(TWO*RR)/(SQRT(RR**2+AQE))**3
     DG(2)=-(TERM*DZE)*RTWO           ! /TWO
     DG(3)=TERM*(EE+QZZE*RFOUR)       ! /FOUR 
     DG(4)=TERM*(EE+QXXE*RFOUR)       ! /FOUR
     IF(NATORB(NJ).LE.2) RETURN
  end if
  !   HYDROGEN-HEAVY ATOM
10 AED= RFOUR*(ONE/BDD(NI,1)+ONE/BDD(NJ,2))**2
  AEQ= RFOUR*(ONE/BDD(NI,1)+ONE/BDD(NJ,3))**2
  EDZ   = (RR-DB)/(SQRT((RR-DB)**2+AED))**3-(RR+DB)/(SQRT((RR+DB)**2+AED))**3 
  EQZZ  =-(RR-TWO*QB)/(SQRT((RR-TWO*QB)**2+AEQ))**3-(RR+TWO*QB)/(SQRT((RR+TWO*QB)**2+AEQ))**3 &
       +(TWO*RR)/(SQRT(RR**2+AEQ))**3
  EQXX  =-(TWO*RR)/(SQRT(RR**2+FOUR*QB**2+AEQ))**3+(TWO*RR)/(SQRT(RR**2+AEQ))**3
  DG(5)=-(TERM*EDZ)*RTWO        ! /TWO
  DG(11)=TERM*(EE+EQZZ*RFOUR)   ! /FOUR 
  DG(12)=TERM*(EE+EQXX*RFOUR)   ! /FOUR
  IF(NATORB(NI).LE.2) RETURN
  !   HEAVY ATOM-HEAVY ATOM
  ADD= RFOUR*(ONE/BDD(NI,2)+ONE/BDD(NJ,2))**2
  ADQ= RFOUR*(ONE/BDD(NI,2)+ONE/BDD(NJ,3))**2
  AQD= RFOUR*(ONE/BDD(NI,3)+ONE/BDD(NJ,2))**2
  AQQ= RFOUR*(ONE/BDD(NI,3)+ONE/BDD(NJ,3))**2
  DXDX  =-(TWO*RR)/(SQRT(RR**2+(DA-DB)**2+ADD))**3+(TWO*RR)/(SQRT(RR**2+(DA+DB)**2+ADD))**3
  DZDZ  =-(RR+DA-DB)/(SQRT((RR+DA-DB)**2+ADD))**3 -(RR-DA+DB)/(SQRT((RR-DA+DB)**2+ADD))**3 &
       +(RR-DA-DB)/(SQRT((RR-DA-DB)**2+ADD))**3 +(RR+DA+DB)/(SQRT((RR+DA+DB)**2+ADD))**3
  DZQXX = TWO*(RR+DA)/(SQRT((RR+DA)**2+FOUR*QB**2+ADQ))**3 -TWO*(RR-DA)/(SQRT((RR-DA)**2+FOUR*QB**2+ADQ))**3 &
       -TWO*(RR+DA)/(SQRT((RR+DA)**2+ADQ))**3 +TWO*(RR-DA)/(SQRT((RR-DA)**2+ADQ))**3
  QXXDZ = TWO*(RR-DB)/(SQRT((RR-DB)**2+FOUR*QA**2+AQD))**3 -TWO*(RR+DB)/(SQRT((RR+DB)**2+FOUR*QA**2+AQD))**3 &
       -TWO*(RR-DB)/(SQRT((RR-DB)**2+AQD))**3 +TWO*(RR+DB)/(SQRT((RR+DB)**2+AQD))**3
  DZQZZ = (RR+DA-TWO*QB)/(SQRT((RR+DA-TWO*QB)**2+ADQ))**3 -(RR-DA-TWO*QB)/(SQRT((RR-DA-TWO*QB)**2+ADQ))**3 &
       +(RR+DA+TWO*QB)/(SQRT((RR+DA+TWO*QB)**2+ADQ))**3 -(RR-DA+TWO*QB)/(SQRT((RR-DA+TWO*QB)**2+ADQ))**3 &
       +TWO*(RR-DA)/(SQRT((RR-DA)**2+ADQ))**3 -TWO*(RR+DA)/(SQRT((RR+DA)**2+ADQ))**3
  QZZDZ = (RR+TWO*QA-DB)/(SQRT((RR+TWO*QA-DB)**2+AQD))**3 -(RR+TWO*QA+DB)/(SQRT((RR+TWO*QA+DB)**2+AQD))**3 &
       +(RR-TWO*QA-DB)/(SQRT((RR-TWO*QA-DB)**2+AQD))**3 -(RR-TWO*QA+DB)/(SQRT((RR-TWO*QA+DB)**2+AQD))**3 &
       -TWO*(RR-DB)/(SQRT((RR-DB)**2+AQD))**3 +TWO*(RR+DB)/(SQRT((RR+DB)**2+AQD))**3
  QXXQXX=-(TWO*RR)/(SQRT(RR**2+FOUR*(QA-QB)**2+AQQ))**3 -(TWO*RR)/(SQRT(RR**2+FOUR*(QA+QB)**2+AQQ))**3 &
       +(FOUR*RR)/(SQRT(RR**2+FOUR*QA**2+AQQ))**3 +(FOUR*RR)/(SQRT(RR**2+FOUR*QB**2+AQQ))**3 &
       -(FOUR*RR)/(SQRT(RR**2+AQQ))**3
  QXXQYY=-(FOUR*RR)/(SQRT(RR**2+FOUR*QA**2+FOUR*QB**2+AQQ))**3 +(FOUR*RR)/(SQRT(RR**2+FOUR*QA**2+AQQ))**3 &
       +(FOUR*RR)/(SQRT(RR**2+FOUR*QB**2+AQQ))**3 -(FOUR*RR)/(SQRT(RR**2+AQQ))**3
  QXXQZZ=-TWO*(RR-TWO*QB)/(SQRT((RR-TWO*QB)**2+FOUR*QA**2+AQQ))**3 &
       -TWO*(RR+TWO*QB)/(SQRT((RR+TWO*QB)**2+FOUR*QA**2+AQQ))**3 &
       +TWO*(RR-TWO*QB)/(SQRT((RR-TWO*QB)**2+AQQ))**3 +TWO*(RR+TWO*QB)/(SQRT((RR+TWO*QB)**2+AQQ))**3 &
       +(FOUR*RR)/(SQRT(RR**2+FOUR*QA**2+AQQ))**3 -(FOUR*RR)/(SQRT(RR**2+AQQ))**3
  QZZQXX=-TWO*(RR+TWO*QA)/(SQRT((RR+TWO*QA)**2+FOUR*QB**2+AQQ))**3 &
       -TWO*(RR-TWO*QA)/(SQRT((RR-TWO*QA)**2+FOUR*QB**2+AQQ))**3 & 
       +TWO*(RR+TWO*QA)/(SQRT((RR+TWO*QA)**2+AQQ))**3 +TWO*(RR-TWO*QA)/(SQRT((RR-TWO*QA)**2+AQQ))**3 &
       +(FOUR*RR)/(SQRT(RR**2+FOUR*QB**2+AQQ))**3 -(FOUR*RR)/(SQRT(RR**2+AQQ))**3
  QZZQZZ=-(RR+TWO*QA-TWO*QB)/(SQRT((RR+TWO*QA-TWO*QB)**2+AQQ))**3 &
       -(RR+TWO*QA+TWO*QB)/(SQRT((RR+TWO*QA+TWO*QB)**2+AQQ))**3 &
       -(RR-TWO*QA-TWO*QB)/(SQRT((RR-TWO*QA-TWO*QB)**2+AQQ))**3 &
       -(RR-TWO*QA+TWO*QB)/(SQRT((RR-TWO*QA+TWO*QB)**2+AQQ))**3 &
       +TWO*(RR-TWO*QA)/(SQRT((RR-TWO*QA)**2+AQQ))**3 +TWO*(RR+TWO*QA)/(SQRT((RR+TWO*QA)**2+AQQ))**3 &
       +TWO*(RR-TWO*QB)/(SQRT((RR-TWO*QB)**2+AQQ))**3 +TWO*(RR+TWO*QB)/(SQRT((RR+TWO*QB)**2+AQQ))**3 &
       -(FOUR*RR)/(SQRT(RR**2+AQQ))**3
  DXQXZ = TWO*(RR-QB)/(SQRT((RR-QB)**2+(DA-QB)**2+ADQ))**3 -TWO*(RR+QB)/(SQRT((RR+QB)**2+(DA-QB)**2+ADQ))**3 &
       -TWO*(RR-QB)/(SQRT((RR-QB)**2+(DA+QB)**2+ADQ))**3 +TWO*(RR+QB)/(SQRT((RR+QB)**2+(DA+QB)**2+ADQ))**3
  QXZDX = TWO*(RR+QA)/(SQRT((RR+QA)**2+(QA-DB)**2+AQD))**3 -TWO*(RR-QA)/(SQRT((RR-QA)**2+(QA-DB)**2+AQD))**3 &
       -TWO*(RR+QA)/(SQRT((RR+QA)**2+(QA+DB)**2+AQD))**3 +TWO*(RR-QA)/(SQRT((RR-QA)**2+(QA+DB)**2+AQD))**3
  QXYQXY=-(FOUR*RR)/(SQRT(RR**2+TWO*(QA-QB)**2+AQQ))**3 -(FOUR*RR)/(SQRT(RR**2+TWO*(QA+QB)**2+AQQ))**3 &
       +(EIGHT*RR)/(SQRT(RR**2+TWO*(QA**2+QB**2)+AQQ))**3
  QXZQXZ=-TWO*(RR+QA-QB)/(SQRT((RR+QA-QB)**2+(QA-QB)**2+AQQ))**3 &
       +TWO*(RR+QA+QB)/(SQRT((RR+QA+QB)**2+(QA-QB)**2+AQQ))**3 &
       +TWO*(RR-QA-QB)/(SQRT((RR-QA-QB)**2+(QA-QB)**2+AQQ))**3 &
       -TWO*(RR-QA+QB)/(SQRT((RR-QA+QB)**2+(QA-QB)**2+AQQ))**3 &
       +TWO*(RR+QA-QB)/(SQRT((RR+QA-QB)**2+(QA+QB)**2+AQQ))**3 &
       -TWO*(RR+QA+QB)/(SQRT((RR+QA+QB)**2+(QA+QB)**2+AQQ))**3 &
       -TWO*(RR-QA-QB)/(SQRT((RR-QA-QB)**2+(QA+QB)**2+AQQ))**3 &
       +TWO*(RR-QA+QB)/(SQRT((RR-QA+QB)**2+(QA+QB)**2+AQQ))**3
  DG(6) =(TERM*DZDZ)*RFOUR                               ! /FOUR
  DG(7) =(TERM*DXDX)*RFOUR
  DG(8) =-TERM*(EDZ*RTWO+QZZDZ*REIGH)                    ! /TWO, /EIGHT 
  DG(9) =-TERM*(EDZ*RTWO+QXXDZ*REIGH)
  DG(10)=-(TERM*QXZDX)*REIGH
  DG(13)=-TERM*(DZE*RTWO+DZQZZ*REIGH)
  DG(14)=-TERM*(DZE*RTWO+DZQXX*REIGH)
  DG(15)=-(TERM*DXQXZ)*REIGH
  DG(16)= TERM*(EE+EQZZ*RFOUR+QZZE*RFOUR+QZZQZZ*RSIXT)   ! /16.0D0
  DG(17)= TERM*(EE+EQZZ*RFOUR+QXXE*RFOUR+QXXQZZ*RSIXT)
  DG(18)= TERM*(EE+EQXX*RFOUR+QZZE*RFOUR+QZZQXX*RSIXT)
  DG(19)= TERM*(EE+EQXX*RFOUR+QXXE*RFOUR+QXXQXX*RSIXT)
  DG(20)=(TERM*QXZQXZ)*RSIXT
  DG(21)= TERM*(EE+EQXX*RFOUR+QXXE*RFOUR+QXXQYY*RSIXT)
  DG(22)= TERM*(QXXQXX-QXXQYY)*RTHIR2                    ! /32.0D0

  RETURN
END SUBROUTINE MNDELRI

SUBROUTINE CALDENS( C,MDIM, NORBS,NDUBL, NSINGL, FRACT, P)
  !
  use chm_kinds
  use number
  use quantm, only : IFRAC
  implicit none
  INTEGER MDIM
  real(chm_real) P(*), C(MDIM,*)
  !***********************************************************************
  !
  !  DENSIT COMPUTES THE DENSITY MATRIX GIVEN THE EIGENVECTOR MATRIX, AND
  !          INFORMATION ABOUT THE M.O. OCCUPANCY.
  !
  !  INPUT:  C     = SQUARE EIGENVECTOR MATRIX, C IS OF SIZE MDIM BY MDIM
  !                  AND THE EIGENVECTORS ARE STORED IN THE TOP LEFT-HAND
  !                  CORNER.
  !          NORBS = NUMBER OF ORBITALS
  !          NDUBL = NUMBER OF DOUBLY-OCCUPIED M.O.S ( =0 IF UHF)
  !          NSINGL= NUMBER OF SINGLY OR FRACTIONALLY OCCUPIED M.O.S.
  !
  !   ON EXIT: P   = DENSITY MATRIX
  !
  !   MOPAC routine DENSIT
  !   Change to CALDENS in order to resolve a name conflect with DENSIT
  !      in SOLANA of CHARMM 22. 02.15.91 YDW
  !
  !***********************************************************************
  !
  INTEGER NORBS,NORBS2,NL1,NL2,NU1,NU2,I,J,K,L,NSINGL,NDUBL
  real(chm_real)  CONST,SUM1,SUM2,FRAC,SIGN,FRACT
  !
  ! SET UP LIMITS FOR SUMS
  !  NL1 = BEGINING OF ONE ELECTRON SUM
  !  NU1 = END OF SAME
  !  NL2 = BEGINING OF TWO ELECTRON SUM
  !  NU2 = END OF SAME
  !
  NORBS2=NORBS/2
  NSINGL=MAX(NDUBL,NSINGL)

  IF((IFRAC .LE. 0) .OR. (NSINGL .GT. NORBS2)) THEN
     !
     !    TAKE POSITRON EQUIVALENT
     !
     SIGN=-ONE
     IF(FRACT .GT. zero) THEN
        FRAC=TWO-FRACT
     ELSE
        FRAC=TWO
     END IF
     IF(NDUBL.EQ.0)THEN
        CONST=ONE
        NL2=2
        NU2=0
        NL1=NSINGL+1
        NU1=NORBS
     ELSE
        CONST=TWO
        NL2=NSINGL+1
        NU2=NORBS
        NL1=NDUBL+1
        NU1=NSINGL
     ENDIF
  ELSE
     !
     !    TAKE ELECTRON EQUIVALENT
     !
50   CONTINUE
     SIGN=ONE
     FRAC=FRACT
     CONST=ZERO
     NL2=1
     NU2=NDUBL
     NL1=NDUBL+1
     NU1=NSINGL
  ENDIF
  L=0
  IF((IFRAC .LE. 0) .OR. (NSINGL .GT. NORBS2)) THEN
     DO I=1,NORBS
        do J=1,I
           L=L+1
           SUM2=ZERO
           SUM1=ZERO
           do K=NL2,NU2
              SUM2=SUM2+C(I,K)*C(J,K)
           end do
           SUM2=SUM2*TWO
           do K=NL1,NU1
              SUM1=SUM1+C(I,K)*C(J,K)
           end do
           P(L)=(SUM2+SUM1*FRAC)*SIGN
        end do
        P(L)=CONST+P(L)
     END DO
  ELSE
     DO I=1,NORBS
        DO J=1,I
           L=L+1
           SUM2=ZERO
           SUM1=ZERO
           DO K=NL2,NU2
              SUM2=SUM2+C(I,K)*C(J,K)
           END DO

           SUM2=SUM2*TWO

           IF(IFRAC.NE.0) THEN
              DO K=NL1,NU1
                 SUM1=SUM1+C(I,K)*C(J,K)
              END DO

              !                 P(L)=(SUM2+SUM1*FRAC)*SIGN
              P(L)= SUM2+SUM1*FRAC
           ELSE
              P(L)=SUM2
           END IF
        END DO
     END DO
  END IF
  RETURN
END SUBROUTINE CALDENS

SUBROUTINE MNDERS(M,N,RR,DEL1,DEL2,DEL3,IS,IOL)
  !
  !************************************************************************
  !*                                                                      *
  !*    on input m    = index of first atomic orbital                     *
  !*             n    = index of second atomic orbital                    *
  !*             rr   = square if interatomic diatance (in bohr)          *
  !*             del1 = catersian distance in derivative direction        *
  !*             del2 = cartesian distance in m a.o.'s direction          *
  !*             del3 = cartesian distance in n a.o.'s direction          *
  !*             is   = indicates type of a.o.-a.o. interaction           *
  !*                  = 1 s/s, 2 s/p', 3 s/p, 4 p'/s, 5 p/s, 6 p/p',      *
  !*                    7 p'/p", 8 p'p', 9 p/p                            *
  !*             iol  = index for storing derivatives in ds               *
  !*                                                                      *
  !************************************************************************
  use chm_kinds
  use number
  use consta
  use quantm, only : C_tempm,Z_tempm,DS_ant
  implicit none
  real(chm_real) SS(6,6)
  !
  INTEGER IS,I,J,M,N,IOL
  real(chm_real)  RR,ADB,AMB,ABN,APB,ADR,DEL1,DEL2,DEL3
  real(chm_real),parameter :: THRTY5=35.0D0, &
       A0=BOHRR,      &
       A02=(A0**2),   &
       A03=(A0**3),   &
       A04=(A0**4),   &
       rA02=one/(A0**2)
  real(chm_real) :: temp1,temp2,temp_1(6),temp_2(6)

  !
  do I=1,6
     temp1= Z_tempm(M,I)
     do J=1,6
        SS(I,J)=ZERO
        APB=temp1*Z_tempm(N,J)
        AMB=temp1+Z_tempm(N,J)
        ADB=APB/AMB
        ADR=MIN(ADB*RR,THRTY5)
        select case (IS)
        case (:0)                ! just in case, IS it out of range between 1 and 9
           ABN=-TWO*ADB*DEL1*rA02                                   ! /(A0**2)
        case (1)
           ABN=-TWO*ADB*DEL1*rA02
        case (2)
           ABN=-FOUR*(ADB**2)*DEL1*DEL2/(SQRT(Z_tempm(N,J))*A03)         ! (A0**3)
        case (3)
           ABN=(TWO*ADB/(SQRT(Z_tempm(N,J))*A0))*(ONE-TWO*ADB*(DEL1**2)*rA02)
        case (4)
           ABN=FOUR*(ADB**2)*DEL1*DEL2/(SQRT(Z_tempm(M,I))*A03)
        case (5)
           ABN=-(TWO*ADB/(SQRT(Z_tempm(M,I))*A0))*(ONE-TWO*ADB*(DEL1**2)*rA02)
        case (6)
           ABN=-(FOUR*(ADB**2)*DEL2/(SQRT(APB)*A02))*(ONE-TWO*ADB*(DEL1**2)*rA02)   ! (A0**2)
        case (7)
           ABN=EIGHT*(ADB**3)*DEL1*DEL2*DEL3/(SQRT(APB)*A04)                        ! (A0**4)
        case (8)
           ABN=-(EIGHT*(ADB**2)*DEL1/(SQRT(APB)*A02))*(HALF-ADB*(DEL2**2)*rA02)
        case (9)
           ABN=-(EIGHT*(ADB**2)*DEL1/(SQRT(APB)*A02))*(ONEPT5-ADB*(DEL1**2)*rA02)
        case (10:)               ! just in case, IS it out of range between 1 and 9
           ABN=-TWO*ADB*DEL1*rA02
        end select
        SS(I,J)=SQRT((TWO*SQRT(APB)/AMB)**3)*EXP(-ADR)*ABN
     end do
  end do

  temp_1(1:6)=C_tempm(M,1:6)
  temp_2(1:6)=C_tempm(N,1:6)
  do J=1,6
     do I=1,6
        DS_ant(IOL)=DS_ant(IOL)+SS(I,J)*temp_1(I)*temp_2(J)
     end do
  end do
  RETURN
END SUBROUTINE MNDERS

SUBROUTINE MNDHC (P,PA,PB,XI,NAT1,IF,IM,IL,JF,JM,JL,DENER,UHF)
  ! 
  ! This routine was substantially modified by DCC (mainly variable
  ! name changes).
  !
  use chm_kinds
  use number
  use dimens_fcm  
  !
  use sizes
  use quantm
  use am1parm, only : USS,UPP,UDD
  implicit none
  EXTERNAL MNHELECT
  real(chm_real)   MNHELECT
  !
  real(chm_real) P(*), PA(*), PB(*)
  real(chm_real) XI(3,*)
  INTEGER NFIRST(2),NMID(2),NLST(2),NAT1(*)
  !***********************************************************************
  !
  !  DHC CALCULATES THE ENERGY CONTRIBUTIONS FROM THOSE PAIRS OF ATOMS
  !         THAT HAVE BEEN MOVED BY SUBROUTINE DERIV.
  !
  !***********************************************************************  
  !
  LOGICAL UHF
  real(chm_real)  H(171), SHMAT(9,9), F(171), &
       WJ(100), E1B(10), E2A(10), WK(100), W(100)
  !
  INTEGER IA,IB,JA,IC,JB,JC,IF,JF,II,JJ,IL,IM,JL,JM,NI,NJ, &
       KR,JT,LINEAR,I,J,I1,I2,J1
  real(chm_real)  EE,WLIM,ENUCLR,DENER,W100P1
  !
  WLIM=ZERO
  !
  NFIRST(1)=1
  NMID(1)=IM-IF+1
  NLST(1)=IL-IF+1
  NFIRST(2)=NLST(1)+1
  NMID(2)=NFIRST(2)+JM-JF
  NLST(2)=NFIRST(2)+JL-JF
  LINEAR=(NLST(2)*(NLST(2)+1))/2
  F(1:LINEAR) =zero
  H(1:LINEAR) =zero
  do I=1,2
     NI=NAT1(I)
     J=NFIRST(I)
     H((J*(J+1))/2)     =USS(NI)
     H(((J+1)*(J+2))/2) =UPP(NI)
     H(((J+2)*(J+3))/2) =UPP(NI)
     H(((J+3)*(J+4))/2) =UPP(NI)
     H(((J+4)*(J+5))/2) =UDD(NI)
     H(((J+5)*(J+6))/2) =UDD(NI)
     H(((J+6)*(J+7))/2) =UDD(NI)
     H(((J+7)*(J+8))/2) =UDD(NI)
     H(((J+8)*(J+9))/2) =UDD(NI)
     H(((J+9)*(J+10))/2)=UDD(NI)
  end do
  F(1:LINEAR)=H(1:LINEAR)
  JA=NFIRST(2)
  JB=NLST(2)
  JC=NMID(2)
  IA=NFIRST(1)
  IB=NLST(1)
  IC=NMID(1)
  JT=JB*(JB+1)/2
  J=2
  I=1
  NJ=NAT1(2)
  NI=NAT1(1)
  !
  CALL MNH1ELEC(NI,NJ,XI(1,1),XI(1,2),SHMAT)
  !
  !      IF(NAT1(1).EQ.0.OR.NAT1(2).EQ.0) THEN
  !         K=(JB*(JB+1))/2
  !         H(1:K)=ZERO
  !      ELSE
  J1=0
  do J=JA,JB
     JJ=J*(J-1)/2
     J1=J1+1
     I1=0
     do I=IA,IB
        JJ=JJ+1
        I1=I1+1
        H(JJ)=SHMAT(I1,J1)
        F(JJ)=SHMAT(I1,J1)
     end do
  end do
  !      ENDIF
  KR=1
  W100P1=100.1D0
  CALL MNROTATE (NJ,NI,XI(1,2),XI(1,1),W(KR),KR,E2A,E1B,ENUCLR,W100P1,[ZERO],0)
  IF(WJ(1).LT.WLIM) then
     WK(1:KR-1) = ZERO
  ENDIF
  !
  !    * ENUCLR IS SUMMED OVER CORE-CORE REPULSION INTEGRALS.
  !
  I2=0
  do I1=IA,IC
     II=I1*(I1-1)/2+IA-1
     do J1=IA,I1
        II=II+1
        I2=I2+1
        H(II)=H(II)+E1B(I2)
        F(II)=F(II)+E1B(I2)
     end do
  end do
  do I1=IC+1,IB
     II=(I1*(I1+1))/2
     F(II)=F(II)+E1B(1)
     H(II)=H(II)+E1B(1)
  end do
  I2=0
  do I1=JA,JC
     II=I1*(I1-1)/2+JA-1
     do J1=JA,I1
        II=II+1
        I2=I2+1
        H(II)=H(II)+E2A(I2)
        F(II)=F(II)+E2A(I2)
     end do
  end do
  do I1=JC+1,JB
     II=(I1*(I1+1))/2
     F(II)=F(II)+E2A(1)
     H(II)=H(II)+E2A(1)
  end do
  CALL FOCK2D(F,P,PA,W, WJ, WK,2,NFIRST,NMID,NLST)
  EE=MNHELECT(NLST(2),PA,H,F,.FALSE.)
  IF( UHF ) THEN
     F(1:LINEAR)=H(1:LINEAR)
     CALL FOCK2D(F,P,PB,W, WJ, WK,2,NFIRST,NMID,NLST)
     EE=EE+MNHELECT(NLST(2),PB,H,F,.FALSE.)
  ELSE
     EE=EE*TWO
  ENDIF
  DENER=ZERO
  !       
  IF (QEQTRM(QQEL)) DENER=EE                ! If include QM/QM electronic energy.
  IF (QEQTRM(QQCC)) DENER=DENER+ENUCLR      ! If include QM/QM core-core repulsion energy.
  !
  RETURN
END SUBROUTINE MNDHC

SUBROUTINE DIAG(FAO,VECTOR,NOCC,EIG,MDIM,N,FMO)
  !
  use chm_kinds
  use number
  use sizes
  implicit none
  INTEGER N,MDIM
  real(chm_real) FAO(N,N),VECTOR(MDIM,N),EIG(N),WS(MAXORB),FMO(N,N)

  !***********************************************************************
  !
  !   "FAST" DIAGONALISATION PROCEDURE.
  !
  !    ON INPUT FAO CONTAINS THE MATRIX TO BE DIAGONALISED IN SQUARE
  !             PACKED FORM.
  !             VECTOR  CONTAINS THE OLD EIGENVECTORS ON INPUT, THE NEW
  !             VECTORS ON EXITING.
  !             NOCC = NUMBER OF OCCUPIED MOLECULAR ORBITALS.
  !             EIG  = EIGENVALUES FROM AN EXACT DIAGONALISATION
  !             MDIM = DECLARED SIZE OF MATRIX "C".
  !             N = NUMBER OF ATOMIC ORBITALS IN BASIS SET
  !
  !  DIAG IS A PSEUDO-DIAGONALISATION PROCEDURE, IN THAT THE VECTORS THAT
  !       ARE GENERATED BY IT ARE MORE NEARLY ABLE TO BLOCK-DIAGONALISE
  !       THE FOCK MATRIX OVER MOLECULAR ORBITALS THAN THE STARTING
  !       VECTORS. IT MUST BE CONSIDERED PSEUDO FOR SEVERAL REASONS:
  !       (A) IT DOES NOT GENERATE EIGENVECTORS - THE SECULAR DETERMINANT
  !           IS NOT DIAGONALISED, ONLY THE OCCUPIED-VIRTUAL INTERSECTION.
  !       (B) MANY SMALL ELEMENTS IN THE SEC.DET. ARE IGNORED AS BEING TOO
  !           SMALL COMPARED WITH THE LARGEST ELEMENT.
  !       (C) WHEN ELEMENTS ARE ELIMINATED BY ROTATION, THE REST OF THE
  !           SEC. DET. IS ASSUMED NOT TO CHANGE, I.E. ELEMENTS CREATED
  !           ARE IGNORED.
  !       (D) THE ROTATION REQUIRED TO ELIMINATE THOSE ELEMENTS CONSIDERED
  !           SIGNIFICANT IS APPROXIMATED TO USING THE EIGENVALUES OF THE
  !           EXACT DIAGONALISATION THROUGHOUT THE REST OF THE ITERATIVE
  !           PROCEDURE.
  !
  !  (NOTE:- IN AN ITERATIVE PROCEDURE ALL THE APPROXIMATIONS PRESENT IN
  !          DIAG BECOME VALID AT SELF-CONSISTENCY, SELF-CONSISTENCY IS
  !          NOT SLOWED DOWN BY USE OF THESE APPROXIMATIONS)
  !
  !    REFERENCE:
  !             "FAST SEMIEMPIRICAL CALCULATIONS",
  !             STEWART. J.J.P., CSASZAR, P., PULAY, P., J. COMP. CHEM.,
  !             3, 227, (1982)
  !
  !***********************************************************************
  !
  INTEGER IJ,NOCC,LUMO,I,J,K,M
  real(chm_real)  BETA,TINY,A,B,C,D,E,ETA,EPS,ALPHA,SUM
  !
  real(chm_real), save :: AONE=one
  logical, save        :: FIRST=.TRUE.
  !
  IF (FIRST) THEN
     FIRST=.FALSE.
     CALL EPSETA(EPS,ETA)
     AONE=ONE+100.D0*EPS
  ENDIF
  !             FMO  IS A WORK-SPACE OF SIZE (N-NOCC)*NOCC, IT WILL HOLD
  !                  THE FOCK MOLECULAR ORBITAL INTERACTION MATRIX.
  !
  !  FIRST, CONSTRUCT THAT PART OF A SECULAR DETERMINANT OVER MOLECULAR
  !  ORBITALS WHICH CONNECTS THE OCCUPIED AND VIRTUAL SETS.
  !
  !***********************************************************************
  !
  !
  !   FOR VECTOR MACHINES, REMOVE THE FOLLOWING LINE.
  !
  TINY=ZERO
  LUMO=NOCC+1
  IJ=0
  DO I=LUMO,N
     DO J=1,N
        SUM=ZERO
        DO K=1,N
           SUM=SUM+FAO(J,K)*VECTOR(K,I)
        END DO
        WS(J)=SUM
     END DO
     DO J=1,NOCC
        SUM=ZERO
        DO K=1,N
           SUM=SUM+WS(K)*VECTOR(K,J)
        END DO
        !
        !   FOR VECTOR MACHINES, REMOVE THE FOLLOWING LINE.
        !
        TINY=MAX(TINY,ABS(SUM))
        FMO(J,I)=SUM
     END DO
  END DO
  TINY=0.04D0*TINY
  !***********************************************************************
  !
  !   NOW DO A CRUDE 2 BY 2 ROTATION TO "ELIMINATE" SIGNIFICANT ELEMENTS
  !
  !***********************************************************************
  IJ=0
  DO I=LUMO,N
     DO J=1,NOCC
        IJ=IJ+1
        !
        !   FOR VECTOR MACHINES, REMOVE THE FOLLOWING LINE.
        !
        if(ABS(FMO(J,I)).ge.TINY) then
           !
           !      BEGIN 2 X 2 ROTATIONS
           !
           C=FMO(J,I)
           D=EIG(J)-EIG(I)
           ! namkh
           !              E=SIGN(SQRT(FOUR*C*C+D*D),D)
           !              ALPHA=SQRT(HALF*(AONE+D/E))
           !              BETA=-SIGN(SQRT(HALF*(AONE-D/E)),C)
           !            
           E      = (C/D)**2
           ALPHA  = ONE-HALF*E
           BETA   = -SIGN(SQRT(E-E*E*PT25),C)
           !
           !      ROTATION OF PSEUDO-EIGENVECTORS
           !
           DO M=1,N
              A=VECTOR(M,J)
              B=VECTOR(M,I)
              VECTOR(M,J)=ALPHA*A +BETA*B
              VECTOR(M,I)=ALPHA*B -BETA*A
           END DO
        end if
     END DO
  END DO
  RETURN
END SUBROUTINE DIAG

FUNCTION DIAGI(IALPHA,IBETA,EIGA,XY,NMOS)  result(diagi_rtn)
  !
  use chm_kinds
  use number
  use sizes
  use quantm,only : NMECI
  implicit none
  INTEGER NMOS
  real(chm_real) XY(NMECI,NMECI,NMECI,NMECI), EIGA(NMECI)
  real(chm_real) :: diagi_rtn
  INTEGER IALPHA(NMOS), IBETA(NMOS)
  !***********************************************************************
  !
  !  CALCULATES THE ENERGY OF A MICROSTATE DEFINED BY IALPHA AND IBETA
  !
  !***********************************************************************
  !
  INTEGER I,J
  real(chm_real)  X,temp1,temp2
  !
  X=ZERO
  do I=1,NMOS
     IF (IALPHA(I).NE.0)THEN
        X=X+EIGA(I)
        do J=1,NMOS
           temp1=XY(I,I,J,J)
           temp2=XY(I,J,I,J)
           X=X + ((temp1-temp2)*IALPHA(J)*HALF + temp1*IBETA(J))
        end do
     ENDIF
  end do
  do I=1,NMOS
     IF (IBETA(I).NE.0) THEN
        X=X+EIGA(I)
        do J=1,I
           X=X+(XY(I,I,J,J)-XY(I,J,I,J))*IBETA(J)
        end do
     ENDIF
  end do
  DIAGI_rtn=X
  RETURN
END FUNCTION DIAGI

SUBROUTINE MNDIAT(NI,NJ,XI,XJ,DI)
  !
  !***********************************************************************
  !
  !   diat calculates the di-atomic overlap integrals between atoms
  !        centered at xi and xj.
  !
  !   on input ni  = atomic number of the first atom.
  !            nj  = atomic number of the second atom.
  !            xi  = cartesian coordinates of the first atom.
  !            xj  = cartesian coordinates of the second atom.
  !
  !  on output di  = diatomic overlap, in a 9 * 9 matrix. layout of
  !                  atomic orbitals in di is
  !                  1   2   3   4   5            6     7       8     9
  !                  s   px  py  pz  d(x**2-y**2) d(xz) d(z**2) d(yz)d(xy)
  !
  !   limitations:  in this formulation, ni and nj must be less than 107
  !         exponents are assumed to be present in common block expont.
  !
  !***********************************************************************
  use chm_kinds
  use number
  use quantm, only : KEYWRD
  use am1parm, only : ZS,ZP,ZD
  implicit none
  INTEGER A,PQ2,B,PQ1,AA,BB
  !
  real(chm_real)  RAA,RBB
  !
  INTEGER IA,IB,II,JJ,NI,NJ,KMIN,LMIN,KMAX,LMAX,NEWK,NK1,I,J,K,L,ISS,JSS,KSS
  real(chm_real)  MNSS,R,X1,X2,Y1,Y2,Z1,Z2
  EXTERNAL MNSS
  !
  LOGICAL FIRST, ANALYT
  real(chm_real) DI(9,9),S(3,3,3),UL1(3),UL2(3),C(3,5,5) &
       ,XI(3),XJ(3), SLIN(27) &
       ,C1(3,5), C2(3,5), C3(3,5), C4(3,5), C5(3,5) &
       ,S1(3,3), S2(3,3), S3(3,3)
  INTEGER NPQ(0:100), IVAL(3,5)
  EQUIVALENCE(SLIN(1),S(1,1,1))
  EQUIVALENCE (C1(1,1),C(1,1,1)), (C2(1,1),C(1,1,2)), &
       (C3(1,1),C(1,1,3)), (C4(1,1),C(1,1,4)), &
       (C5(1,1),C(1,1,5)), (S1(1,1),S(1,1,1)), &
       (S2(1,1),S(1,1,2)), (S3(1,1),S(1,1,3))
  DATA NPQ/1, &
       1,0,2,2,2,2,2,2,2,0, &
       0,3,3,3,3,3,3,0,0,4, &
       4,4,4,4,4,4,4,4,4,4, &
       4,4,4,4,4,0,5,5,5,5, &
       5,5,5,5,5,5,5,5,5,5, &
       5,5,5,5,32*6,14*0/
  DATA IVAL/1,0,9,1,3,8,1,4,7,1,2,6,0,0,5/
  DATA FIRST /.TRUE./

  ANALYT=(INDEX(KEYWRD,'ANALYT').NE.0)
  !
  X1=XI(1)
  X2=XJ(1)
  Y1=XI(2)
  Y2=XJ(2)
  Z1=XI(3)
  Z2=XJ(3)
  PQ1=NPQ(NI)
  PQ2=NPQ(NJ)
  DI(1:9,1:9)=zero
  CALL MNCOE(X1,Y1,Z1,X2,Y2,Z2,PQ1,PQ2,C,R)
  IF(PQ1.EQ.0.OR.PQ2.EQ.0.OR.R.GE.TEN) return
  IF(R.LT.0.001d0) RETURN
  IA=MIN(PQ1,3)
  IB=MIN(PQ2,3)
  A=IA-1
  B=IB-1
  IF(ANALYT)THEN
     CALL MNGOVER(NI,NJ,XI,XJ,R,DI)
     RETURN
  ENDIF
  IF(NI.LT.18.AND.NJ.LT.18) THEN
     CALL MNDIAT2(NI,ZS(NI),ZP(NI),R,NJ,ZS(NJ),ZP(NJ),S)
  ELSE
     UL1(1)=ZS(NI)
     UL2(1)=ZS(NJ)
     UL1(2)=ZP(NI)
     UL2(2)=ZP(NJ)
     UL1(3)=MAX(ZD(NI),THREE*PTONE)
     UL2(3)=MAX(ZD(NJ),THREE*PTONE)
     SLIN(1:27)=ZERO
     NEWK=MIN(A,B)
     NK1=NEWK+1
     loopI1: do I=1,IA
        ISS=I
        IB=B+1
        loopJ1: do J=1,IB
           JSS=J
           loopK1: do K=1,NK1
              IF(K.GT.I.OR.K.GT.J) cycle loopK1
              KSS=K
              S(I,J,K)=MNSS(PQ1,PQ2,ISS,JSS,KSS,UL1(I),UL2(J),R,FIRST)
           end do loopK1
        end do loopJ1
     end do loopI1
  ENDIF
  do I=1,IA
     KMIN=4-I
     KMAX=2+I
     do J=1,IB
        IF(J.EQ.2)THEN
!!!               AA=-1
!!!               BB=1
           RAA=-ONE
           RBB= ONE
        ELSE
!!!               AA=1
           RAA=ONE
           IF(J.EQ.3) THEN
!!!                  BB=-1
              RBB=-ONE
           ELSE
!!!                  BB=1
              RBB=ONE
           ENDIF
        ENDIF
        LMIN=4-J
        LMAX=2+J
        do K=KMIN,KMAX
           do L=LMIN,LMAX
              II=IVAL(I,K)
              JJ=IVAL(J,L)
              DI(II,JJ)= S1(I,J)*C3(I,K)*C3(J,L)*RAA+(C4(I,K)*C4(J,L)+C2(I,K)*C2(J,L))*RBB*S2(I,J) &
                   +(C5(I,K)*C5(J,L)+C1(I,K)*C1(J,L))*S3(I,J)
           end do
        end do
     end do
  end do
  !#      WRITE(6,*)' OVERLAP FROM DIAT2'
  !#      DO I=1,4
  !#         WRITE(6,'(4F15.10)')(DI(J,I),J=1,4)
  !#      END DO
  RETURN
END SUBROUTINE MNDIAT

SUBROUTINE EPSETA(EPS,ETA)
  !-----------------------------------------------------------------------
  !
  !     COMPUTE AND RETURN ETA, THE SMALLEST REPRESENTABLE NUMBER,
  !     AND EPS IS THE SMALLEST NUMBER FOR WHICH 1+EPS.NE.1.
  !
  use chm_kinds
  use number
  implicit none
  real(chm_real) ETA,EPS
  real(chm_real),parameter :: rTWO=one/two
  !
  ETA = ONE

  do while ((ETA*rTWO).ne.ZERO)
     if(ETA.LT.1.D-38) EXIT
     ETA = ETA * rTWO
  end do
  EPS = one
  do while ((ONE+(EPS*rTWO)).ne.ONE)
     if(EPS.LT.1.D-17) EXIT
     EPS = EPS * rTWO
  end do
  RETURN
END SUBROUTINE EPSETA

SUBROUTINE MNDIAT2(NA,ESA,EPA,R12,NB,ESB,EPB,S)
  !
  use chm_kinds
  use number
  use consta
  use quantm, only : A_int,B_int,SA_int,SB_int,ISP_int,IPS_int
  implicit none
  INTEGER NA,NB
  real(chm_real) ESA,EPA,ESB,EPB,R12
  real(chm_real) S(3,3,3)
  !***********************************************************************
  !
  ! OVERLP CALCULATES OVERLAPS BETWEEN ATOMIC ORBITALS FOR PAIRS OF ATOMS
  !        IT CAN HANDLE THE ORBITALS 1S, 2S, 3S, 2P, AND 3P.
  !
  !***********************************************************************
  INTEGER INMB(0:17),III(78)
  !
  INTEGER II,JMIN,JMAX,I,J,K,NBOND
  real(chm_real)  D,RT3,E,W,RAB,RAB2
  real(chm_real):: temp1,temp2,temp3,temp4,temp5,temp6,temp7,temp8, rQR_THREE, SROOT3
  real(chm_real), parameter :: rTHREE=one/THREE, &
       PT0625=0.0625D00, &
       PR0312=0.03125D00, &
       RPT750=one/7.5D00, &
       R480  =one/480.D00
  !
  DATA INMB/1,1,0,2,2,3,4,5,6,7,0,8,8,8,9,10,11,12/
  !     NUMBERING CORRESPONDS TO BOND TYPE MATRIX GIVEN ABOVE
  !      THE CODE IS
  !
  !     III=1      FIRST - FIRST  ROW ELEMENTS
  !        =2      FIRST - SECOND
  !        =3      FIRST - THIRD
  !        =4      SECOND - SECOND
  !        =5      SECOND - THIRD
  !        =6      THIRD - THIRD
  DATA III/1,2,4,   2,4,4,   2,4,4,4,   2,4,4,4,4, &
       2,4,4,4,4,4,   2,4,4,4,4,4,4,   3,5,5,5,5,5,5,6, &
       3,5,5,5,5,5,5,6,6,   3,5,5,5,5,5,5,6,6,6,   3,5,5,5,5,5,5,6,6,6,6 &
       , 3,5,5,5,5,5,5,6,6,6,6,6/


  ! intialize
  rQR_THREE=one/SQRT(THREE)
  SROOT3 = SQRT(THREE)

  !
  !      ASSIGNS BOND NUMBER
  !
  JMAX=MAX0(INMB(NA),INMB(NB))
  JMIN=MIN0(INMB(NA),INMB(NB))
  NBOND=(JMAX*(JMAX-1))/2+JMIN
  II=III(NBOND)
  S(1:3,1:3,1:3)=zero
  RAB=R12/BOHRR
  RAB2 =RAB*RAB
  !
  !     ------------------------------------------------------------------
  ! *** THE ORDERING OF THE ELEMENTS WITHIN S IS
  ! *** S(1,1,1)=(S(B)/S(A))
  ! *** S(1,2,1)=(P-SIGMA(B)/S(A))
  ! *** S(2,1,1)=(S(B)/P-SIGMA(A))
  ! *** S(2,2,1)=(P-SIGMA(B)/P-SIGMA(A))
  ! *** S(2,2,2)=(P-PI(B)/P-PI(A))
  !     ------------------------------------------------------------------
  select case (II)
  case (:0)
     ! *** Just in case out of range, use default (1)
     CALL MNSET (ESA,ESB,NA,NB,RAB,NBOND,II)
     S(1,1,1)=PT25*SQRT((SA_int*SB_int*RAB2)**3)*(A_int(3)*B_int(1)-B_int(3)*A_int(1))
  case (1)
     ! *** FIRST ROW - FIRST ROW OVERLAPS
     CALL MNSET (ESA,ESB,NA,NB,RAB,NBOND,II)
     S(1,1,1)=PT25*SQRT((SA_int*SB_int*RAB2)**3)*(A_int(3)*B_int(1)-B_int(3)*A_int(1)) ! 0.25D00
  case (2)
     ! *** FIRST ROW - SECOND ROW OVERLAPS
     CALL MNSET (ESA,ESB,NA,NB,RAB,NBOND,II)
     W=SQRT((SA_int**3)*(SB_int**5))*(RAB**4)*PT125     ! 0.125D00
     S(1,1,1) = rQR_THREE      ! SQRT(ONE/THREE)
     S(1,1,1)=W*S(1,1,1)*(A_int(4)*B_int(1)-B_int(4)*A_int(1)+A_int(3)*B_int(2)-B_int(3)*A_int(2))
     IF (NA.GT.1) CALL MNSET (EPA,ESB,NA,NB,RAB,NBOND,II)
     IF (NB.GT.1) CALL MNSET (ESA,EPB,NA,NB,RAB,NBOND,II)
     W=SQRT((SA_int**3)*(SB_int**5))*(RAB**4)*PT125     ! 0.125D00
     S(ISP_int,IPS_int,1)=W*(A_int(3)*B_int(1)-B_int(3)*A_int(1)+A_int(4)*B_int(2)-B_int(4)*A_int(2))
  case (3)
     ! *** FIRST ROW - THIRD ROW OVERLAPS
     CALL MNSET (ESA,ESB,NA,NB,RAB,NBOND,II)
     W=SQRT((SA_int**3)*(SB_int**7)*RPT750)*(RAB**5)*PT0625      ! /7.5D00, 0.0625D00
     S(1,1,1)=W*(A_int(5)*B_int(1)-B_int(5)*A_int(1)+two*(A_int(4)*B_int(2)-B_int(4)*A_int(2)))*rQR_THREE      ! /SROOT3
     IF (NA.GT.1) CALL MNSET (EPA,ESB,NA,NB,RAB,NBOND,II)
     IF (NB.GT.1) CALL MNSET (ESA,EPB,NA,NB,RAB,NBOND,II)
     W=SQRT((SA_int**3)*(SB_int**7)*RPT750)*(RAB**5)*PT0625
     S(ISP_int,IPS_int,1)=W*(A_int(4)*(B_int(1)+B_int(3))-B_int(4)*(A_int(1)+A_int(3)) &
          +B_int(2)*(A_int(3)+A_int(5))-A_int(2)*(B_int(3)+B_int(5)))
  case (4)
     ! *** SECOND ROW - SECOND ROW OVERLAPS
     CALL MNSET (ESA,ESB,NA,NB,RAB,NBOND,II)
     W  =SQRT((SA_int*SB_int)**5)*(RAB**5)*PT0625     ! 0.0625D00
     RT3=rQR_THREE      ! ONE/SQRT(THREE)
     S(1,1,1)=W*(A_int(5)*B_int(1)+B_int(5)*A_int(1)-TWO*A_int(3)*B_int(3))*rTHREE  ! /THREE
     CALL MNSET (ESA,EPB,NA,NB,RAB,NBOND,II)
     IF (NA.GT.NB) CALL MNSET (EPA,ESB,NA,NB,RAB,NBOND,II)
     W=SQRT((SA_int*SB_int)**5)*(RAB**5)*PT0625
     D=A_int(4)*(B_int(1)-B_int(3))-A_int(2)*(B_int(3)-B_int(5))
     E=B_int(4)*(A_int(1)-A_int(3))-B_int(2)*(A_int(3)-A_int(5))
     S(ISP_int,IPS_int,1)=W*RT3*(D+E)
     CALL MNSET (EPA,ESB,NA,NB,RAB,NBOND,II)
     IF (NA.GT.NB) CALL MNSET (ESA,EPB,NA,NB,RAB,NBOND,II)
     W=SQRT((SA_int*SB_int)**5)*(RAB**5)*PT0625
     D=A_int(4)*(B_int(1)-B_int(3))-A_int(2)*(B_int(3)-B_int(5))
     E=B_int(4)*(A_int(1)-A_int(3))-B_int(2)*(A_int(3)-A_int(5))
     S(IPS_int,ISP_int,1)=-W*RT3*(E-D)
     CALL MNSET (EPA,EPB,NA,NB,RAB,NBOND,II)
     W=SQRT((SA_int*SB_int)**5)*(RAB**5)*PT0625
     S(2,2,1)=-W*(B_int(3)*(A_int(5)+A_int(1))-A_int(3)*(B_int(5)+B_int(1)))
     S(2,2,2)=half*W*(A_int(5)*(B_int(1)-B_int(3))-B_int(5)*(A_int(1)-A_int(3)) &
          -A_int(3)*B_int(1)+B_int(3)*A_int(1))
  case (5)
     ! *** SECOND ROW - THIRD ROW OVERLAPS
     CALL MNSET (ESA,ESB,NA,NB,RAB,NBOND,II)
     W=SQRT((SA_int**5)*(SB_int**7)*RPT750)*(RAB**6)*PR0312       ! 0.03125D00
     RT3 = rQR_THREE      ! ONE / SQRT(THREE)
     S(1,1,1)=W*(A_int(6)*B_int(1)+A_int(5)*B_int(2)-two*(A_int(4)*B_int(3)+A_int(3)*B_int(4)) &
          +A_int(2)*B_int(5)+A_int(1)*B_int(6))*rTHREE        ! /THREE
     CALL MNSET (ESA,EPB,NA,NB,RAB,NBOND,II)
     IF (NA.GT.NB) CALL MNSET (EPA,ESB,NA,NB,RAB,NBOND,II)
     W=SQRT((SA_int**5)*(SB_int**7)*RPT750)*(RAB**6)*PR0312
     S(ISP_int,IPS_int,1)=W*RT3*(A_int(6)*B_int(2)+A_int(5)*B_int(1)-two*(A_int(4)*B_int(4)+A_int(3)*B_int(3)) &
          +A_int(2)*B_int(6)+A_int(1)*B_int(5))
     CALL MNSET (EPA,ESB,NA,NB,RAB,NBOND,II)
     IF (NA.GT.NB) CALL MNSET (ESA,EPB,NA,NB,RAB,NBOND,II)
     W=SQRT((SA_int**5)*(SB_int**7)*RPT750)*(RAB**6)*PR0312
     S(IPS_int,ISP_int,1)=-W*RT3*(A_int(5)*(two*B_int(3)-B_int(1))-B_int(5)*(two*A_int(3)-A_int(1)) &
          -A_int(2)*(B_int(6)-two*B_int(4))+B_int(2)*(A_int(6)-two*A_int(4)))
     CALL MNSET (EPA,EPB,NA,NB,RAB,NBOND,II)
     W=SQRT((SA_int**5)*(SB_int**7)*RPT750)*(RAB**6)*PR0312
     S(2,2,1)=-W*(B_int(4)*(A_int(1)+A_int(5))-A_int(4)*(B_int(1)+B_int(5)) &
          +B_int(3)*(A_int(2)+A_int(6))-A_int(3)*(B_int(2)+B_int(6)))
     S(2,2,2)=half*W*(A_int(6)*(B_int(1)-B_int(3))-B_int(6)*(A_int(1)-A_int(3)) &
          +A_int(5)*(B_int(2)-B_int(4))-B_int(5)*(A_int(2)-A_int(4)) &
          -A_int(4)*B_int(1)+B_int(4)*A_int(1)-A_int(3)*B_int(2)+B_int(3)*A_int(2)) 
  case (6)
     ! *** THIRD ROW - THIRD ROW OVERLAPS
     CALL MNSET (ESA,ESB,NA,NB,RAB,NBOND,II)
     W=SQRT((SA_int*SB_int*RAB*RAB)**7)*R480           ! /480.D00
     RT3 = rQR_THREE      ! ONE / SQRT(THREE)
     S(1,1,1)=W*(A_int(7)*B_int(1)-THREE*(A_int(5)*B_int(3)-A_int(3)*B_int(5))-A_int(1)*B_int(7))*rTHREE        ! /THREE
     CALL MNSET (ESA,EPB,NA,NB,RAB,NBOND,II)
     IF (NA.GT.NB) CALL MNSET (EPA,ESB,NA,NB,RAB,NBOND,II)
     W=SQRT((SA_int*SB_int*RAB*RAB)**7)*R480
     D=A_int(6)*(B_int(1)-B_int(3))-TWO*A_int(4)*(B_int(3)-B_int(5))+A_int(2)*(B_int(5)-B_int(7))
     E=B_int(6)*(A_int(1)-A_int(3))-TWO*B_int(4)*(A_int(3)-A_int(5))+B_int(2)*(A_int(5)-A_int(7))
     S(ISP_int,IPS_int,1)=W*RT3*(D-E)
     CALL MNSET (EPA,ESB,NA,NB,RAB,NBOND,II)
     IF (NA.GT.NB) CALL MNSET (ESA,EPB,NA,NB,RAB,NBOND,II)
     W=SQRT((SA_int*SB_int*RAB*RAB)**7)*R480
     D=A_int(6)*(B_int(1)-B_int(3))-TWO*A_int(4)*(B_int(3)-B_int(5))+A_int(2)*(B_int(5)-B_int(7))
     E=B_int(6)*(A_int(1)-A_int(3))-TWO*B_int(4)*(A_int(3)-A_int(5))+B_int(2)*(A_int(5)-A_int(7))
     S(IPS_int,ISP_int,1)=-W*RT3*(-D-E)
     CALL MNSET (EPA,EPB,NA,NB,RAB,NBOND,II)
     W=SQRT((SA_int*SB_int*RAB*RAB)**7)*R480
     S(2,2,1)=-W*(A_int(3)*(B_int(7)+two*B_int(3))-A_int(5)*(B_int(1)+two*B_int(5)) &
          -B_int(5)*A_int(1)+A_int(7)*B_int(3))
     S(2,2,2)=half*W*(A_int(7)*(B_int(1)-B_int(3))+B_int(7)*(A_int(1)-A_int(3)) &
          +A_int(5)*(B_int(5)-B_int(3)-B_int(1))+B_int(5)*(A_int(5)-A_int(3)-A_int(1)) &
          +two*A_int(3)*B_int(3))
  case (7:)
     ! *** Just in case out of range, use default (1)
     CALL MNSET (ESA,ESB,NA,NB,RAB,NBOND,II)
     S(1,1,1)=PT25*SQRT((SA_int*SB_int*RAB2)**3)*(A_int(3)*B_int(1)-B_int(3)*A_int(1))
  end select
  !
  RETURN
  !
END SUBROUTINE MNDIAT2

FUNCTION DOT1(X,Y,N) result(dot1_rtn)
  !
  use chm_kinds
  implicit none
  INTEGER N
  real(chm_real) X(*), Y(*),dot1_rtn
  !************************************************************************
  !*
  !*   DOT FORMS THE SCALAR PRODUCT OF TWO VECTORS.
  !*
  !*   ON INPUT     X   =    FIRST VECTOR, OF LENGTH N.
  !*                Y   =    SECOND VECTOR, OF LENGTH N.
  !*
  !*   ON RETURN    DOT =    DOT PRODUCT OF X AND Y.
  !*
  !************************************************************************
  INTEGER I
  real(chm_real):: rtemp
  !
  rtemp = 0.0D0
  do I=1,N
     rtemp = rtemp + X(I)*Y(I)
  end do
  dot1_rtn=rtemp
  RETURN
END FUNCTION DOT1

SUBROUTINE MNFOCK2(F, PTOT, P, W, WJ, WK, NUMAT, NFIRST, NMIDLE, NLAST)
  !
  use chm_kinds
  use number
  use sizes
#if KEY_PARALLEL==1
  use parallel     
#endif
  use quantm, only : KEYWRD
  implicit none
  real(chm_real) F(*), PTOT(*), WJ(*), WK(*)
  INTEGER NFIRST(*), NMIDLE(*), NLAST(*)
  real(chm_real) P(*), W(*)
  ! for parallel run
  !***********************************************************************
  !
  ! FOCK2 FORMS THE TWO-ELECTRON TWO-CENTER REPULSION PART OF THE FOCK
  ! MATRIX
  ! ON INPUT  PTOT = TOTAL DENSITY MATRIX.
  !           P    = ALPHA OR BETA DENSITY MATRIX.
  !           W    = TWO-ELECTRON INTEGRAL MATRIX.
  !
  !  ON OUTPUT F   = PARTIAL FOCK MATRIX
  !***********************************************************************
  real(chm_real) SPPOP(NUMATM), DPOP(NUMATM)
  INTEGER IFACT(MAXORB),I1FACT(MAXORB)
  !
  INTEGER NUMAT,IA,IB,JA,IC,JB,KA,JC,KB,KC,II,IJ,IK,JJ,IL,JK,JL, &
       KK,KL,IONE,I,J,K,L,IMINUS,I2,J2
  real(chm_real)  AA,BB,SS,DREP,A,SUM
  !
  ! go parallel
  INTEGER ATFRST,ATLAST,NODNQM,NII,NJJ,KKJ(10),ntoterm,nterm
  DATA KKJ/1,3,6,10,15,21,28,36,45,55/
  !
  !   SET UP ARRAY OF (I*(I-1))/2
  !
  do I=1,MAXORB
     IFACT(I) =(I*(I-1))/2
     I1FACT(I)=IFACT(I)+I
  end do
  IONE=1
  KK=0
  ! go parallel
  nterm   = 0
  ntoterm = 0
  do i  = 2,numat
     ntoterm=ntoterm+(i-1)
  end do
  !
  ! go parallel
#if KEY_PARALLEL==1 /*paramain*/
  nodnqm = ntoterm / numnod
#if KEY_PARAFULL==1 /*parfmain*/
#if KEY_PARASCAL==1 /*parstest*/
#error  'Illegal parallel compile options'
#endif /* (parstest)*/
  ATFRST=MYNOD*NODNQM + 1
  ATLAST=(MYNOD+1)*NODNQM
  IF(MYNOD.EQ.(NUMNOD-1)) ATLAST=ntoterm
#elif KEY_PARASCAL==1 || KEY_SPACDEC==1 /*parfmain*/
  ATFRST=1
  ATLAST=ntoterm
#else /* (parfmain)*/
#error  'Illegal parallel compile options'
#endif /* (parfmain)*/
  !  DTM use PI parallel
  IF (QMPI) THEN
     ATFRST=1
     ATLAST=ntoterm
  ENDIF
#else /* (paramain)*/
  ATFRST=1
  ATLAST=ntoterm
#endif /* (paramain)*/
  !
  !  I'VE FORGOTTEN WHAT THESE LINES WERE IN FOR, BUT THEY CAUSED
  !  SEVERE BUGS IN ITER, AT TIMES
  !#      NORBS=NLAST(NUMAT)
  !#      LINEA1=(NORBS*(NORBS+1))/2 + 1
  !#      P(LINEA1)=ZERO
  !
  ! go parallel
  loopII: Do II=1,NUMAT
     IA=NFIRST(II)
     IB=NLAST(II)
     IC=NMIDLE(II)
     NII=NMIDLE(II)-NFIRST(II)+1
     SUM=ZERO
     do I=IA,IC
        SUM=SUM+PTOT(I1FACT(I))
     end do
     SPPOP(II)=SUM
     SUM=ZERO
     do I=IC+1,IB
        SUM=SUM+PTOT(I1FACT(I))
     end do
     DPOP(II)=SUM
     IMINUS=II-IONE
     loopJJ: do JJ=1,IMINUS
        JA=NFIRST(JJ)
        JB=NLAST(JJ)
        JC=NMIDLE(JJ)
        NJJ=NMIDLE(JJ)-NFIRST(JJ)+1
        ! go parallel
#if KEY_PARALLEL==1
        !  DTM PI
        IF (.NOT. QMPI) THEN 
           nterm = nterm + 1
           if(nterm.lt.atfrst) then
              kk = kk + KKJ(NII)*KKJ(NJJ)
              cycle loopJJ                   ! goto 150 
           else if(nterm.gt.atlast) then
              EXIT loopII                    ! goto 170
           end if
        ENDIF
#endif 
        DREP=W(KK+1)
        loopI: do I=IA,IC
           KA=IFACT(I)
           loopJ: do J=IA,I
              KB=IFACT(J)
              IJ=KA+J
              AA=TWO
              IF (I.EQ.J) AA=ONE
              loopK: do K=JA,JC
                 KC=IFACT(K)
                 IK=KA+K
                 JK=KB+K
                 loopL: do L=JA,K
                    IL=KA+L
                    JL=KB+L
                    KL=KC+L
                    BB=TWO
                    IF (K.EQ.L) BB=ONE
                    KK=KK+1
                    A=W(KK)
                    !
                    !     A  IS THE REPULSION INTEGRAL (I,J/K,L) WHERE ORBITALS I AND J ARE
                    !     ON ATOM II, AND ORBITALS K AND L ARE ON ATOM JJ.
                    !     AA AND BB ARE CORRECTION FACTORS SINCE
                    !     (I,J/K,L)=(J,I/K,L)=(I,J/L,K)=(J,I/L,K)
                    !     IJ IS THE LOCATION OF THE MATRIX ELEMENTS BETWEEN ATOMIC ORBITALS
                    !     I AND J.  SIMILARLY FOR IK ETC.
                    !
                    ! THIS FORMS THE TWO-ELECTRON TWO-CENTER REPULSION PART OF THE FOCK
                    ! MATRIX.  THE CODE HERE IS HARD TO FOLLOW, AND IMPOSSIBLE TO MODIFY!,
                    ! BUT IT WORKS,
                    F(IJ)=F(IJ)+BB*A*PTOT(KL)
                    F(KL)=F(KL)+AA*A*PTOT(IJ)
                    A=A*AA*BB*PT25             ! 0.25D0
                    F(IK)=F(IK)-A*P(JL)
                    F(IL)=F(IL)-A*P(JK)
                    F(JK)=F(JK)-A*P(IL)
                    F(JL)=F(JL)-A*P(IK)
                 end do loopL
              end do loopK
           end do loopJ
        end do loopI
        !
        !   D-ORBITAL CORRECTION
        !
        do I=IC+1,IB
           KA=IFACT(I)
           do J=JA,JB
              IJ=KA+J
              !
              !   ATOM J (S, P, AND D (IF PRESENT)) EXCHANGE WITH ATOM I (D ONLY)
              !
              F(IJ)=F(IJ)-HALF*DREP*P(IJ)
           end do
        end do
        do I=IA,IC
           KA=IFACT(I)
           do J=JC+1,JB
              IJ=KA+J
              !
              !    ATOM J (D(IF PRESENT)) EXCHANGE WITH ATOM I (S AND P ONLY)
              F(IJ)=F(IJ)-HALF*DREP*P(IJ)
           end do
        end do
        !
        !                      THE COULOMB REPULSION TERMS.
        !
        !     FIRST, ATOM J (S, P AND D SHELLS) BEING REPELLED BY ATOM I(DSHELL)
        do J=JA,JB
           J2=I1FACT(J)
           F(J2)=F(J2)+DREP*DPOP(II)
        end do
        !
        !     ATOM J (D SHELL) BEING REPELLED BY ATOM I (S AND P SHELLS)
        do J=JC+1,JB
           J2=I1FACT(J)
           F(J2)=F(J2)+DREP*SPPOP(II)
        end do
        !
        !     ATOM I (S, P AND D SHELLS) BEING REPELLED BY ATOM J (D SHELL)
        do I=IA,IB
           I2=I1FACT(I)
           F(I2)=F(I2)+DREP*DPOP(JJ)
        end do
        !
        !    ATOM I (D SHELL) BEING REPELLED BY ATOM J (S AND P SHELLS)
        do I=IC+1,IB
           I2=I1FACT(I)
           F(I2)=F(I2)+DREP*SPPOP(JJ)
        end do
     end do loopJJ
  End do loopII
  !
  RETURN
END SUBROUTINE MNFOCK2

SUBROUTINE FOCK2D(F,PTOT, P, W, WJ, WK, NUMAT, NFIRST, NMIDLE, NLAST)
  !
  use chm_kinds
  use number
  use quantm, only : KEYWRD
  implicit none
  real(chm_real) F(*), PTOT(*), WJ(*), WK(*)
  INTEGER NFIRST(*), NMIDLE(*), NLAST(*)
  real(chm_real) P(*), W(*)
  !***********************************************************************
  !
  ! FOCK2 FORMS THE TWO-ELECTRON TWO-CENTER REPULSION PART OF THE FOCK
  ! MATRIX
  ! ON INPUT  PTOT = TOTAL DENSITY MATRIX.
  !           P    = ALPHA OR BETA DENSITY MATRIX.
  !           W    = TWO-ELECTRON INTEGRAL MATRIX.
  !
  !  ON OUTPUT F   = PARTIAL FOCK MATRIX
  !***********************************************************************
  real(chm_real) SPPOP(2), DPOP(2)
  INTEGER IFACT(11), I1FACT(11)
  !
  INTEGER NORBS,NUMAT,IA,IB,JA,IC,JB,KA,JC,KB,KC,LINEA1,II,IJ,IK, &
       JJ,IL,JK,JL,KK,KL,IONE,I,J,K,L,IMINUS,I2,J2
  real(chm_real)  AA,BB,SS,DREP,A,SUM
  !
  !   SET UP ARRAY OF (I*(I-1))/2
  !
  do I=1,11
     IFACT(I)=(I*(I-1))/2
     I1FACT(I)=IFACT(I)+I
  end do
  IONE=1
  KK=0
  NORBS=NLAST(NUMAT)
  LINEA1=(NORBS*(NORBS+1))/2 + 1
  P(LINEA1)=ZERO
  loopII: Do II=1,NUMAT
     IA=NFIRST(II)
     IB=NLAST(II)
     IC=NMIDLE(II)
     SUM=ZERO
     do I=IA,IC
        SUM=SUM+PTOT(I1FACT(I))
     end do
     SPPOP(II)=SUM
     SUM=ZERO
     do I=IC+1,IB
        SUM=SUM+PTOT(I1FACT(I))
     end do
     DPOP(II)=SUM
     IMINUS=II-IONE
     loopJJ: do JJ=1,IMINUS
        JA=NFIRST(JJ)
        JB=NLAST(JJ)
        JC=NMIDLE(JJ)
        DREP=WJ(KK+1)
        DREP=W(KK+1)
        do I=IA,IC
           KA=IFACT(I)
           do J=IA,I
              KB=IFACT(J)
              IJ=KA+J
              AA=TWO
              IF (I.EQ.J) AA=ONE
              do K=JA,JC
                 KC=IFACT(K)
                 IK=KA+K
                 JK=KB+K
                 do L=JA,K
                    IL=KA+L
                    JL=KB+L
                    KL=KC+L
                    BB=TWO
                    IF (K.EQ.L) BB=ONE
                    KK=KK+1
                    A=W(KK)
                    !
                    !     A  IS THE REPULSION INTEGRAL (I,J/K,L) WHERE ORBITALS I AND J ARE
                    !     ON ATOM II, AND ORBITALS K AND L ARE ON ATOM JJ.
                    !     AA AND BB ARE CORRECTION FACTORS SINCE
                    !     (I,J/K,L)=(J,I/K,L)=(I,J/L,K)=(J,I/L,K)
                    !     IJ IS THE LOCATION OF THE MATRIX ELEMENTS BETWEEN ATOMIC ORBITALS
                    !     I AND J.  SIMILARLY FOR IK ETC.
                    !
                    ! THIS FORMS THE TWO-ELECTRON TWO-CENTER REPULSION PART OF THE FOCK
                    ! MATRIX.  THE CODE HERE IS HARD TO FOLLOW, AND IMPOSSIBLE TO MODIFY!,
                    ! BUT IT WORKS,
                    F(IJ)=F(IJ)+BB*A*PTOT(KL)
                    F(KL)=F(KL)+AA*A*PTOT(IJ)
                    A=A*AA*BB*PT25            ! 0.25D0
                    F(IK)=F(IK)-A*P(JL)
                    F(IL)=F(IL)-A*P(JK)
                    F(JK)=F(JK)-A*P(IL)
                    F(JL)=F(JL)-A*P(IK)
                 end do
              end do
           end do
        end do
        !
        !   D-ORBITAL CORRECTION
        !
        do I=IC+1,IB
           KA=IFACT(I)
           do J=JA,JB
              IJ=KA+J
              !
              !   ATOM J (S, P, AND D (IF PRESENT)) EXCHANGE WITH ATOM I (D ONLY)
              !
              F(IJ)=F(IJ)-HALF*DREP*P(IJ)
           end do
        end do
        do I=IA,IC
           KA=IFACT(I)
           do J=JC+1,JB
              IJ=KA+J
              !
              !    ATOM J (D(IF PRESENT)) EXCHANGE WITH ATOM I (S AND P ONLY)
              !
              F(IJ)=F(IJ)-HALF*DREP*P(IJ)
           end do
        end do
        !
        !                      THE COULOMB REPULSION TERMS.
        !
        !     FIRST, ATOM J (S, P AND D SHELLS) BEING REPELLED BY ATOM I(DSHELL)
        !
        do J=JA,JB
           J2=I1FACT(J)
           F(J2)=F(J2)+DREP*DPOP(II)
        end do
        !
        !     ATOM J (D SHELL) BEING REPELLED BY ATOM I (S AND P SHELLS)
        !
        do J=JC+1,JB
           J2=I1FACT(J)
           F(J2)=F(J2)+DREP*SPPOP(II)
        end do
        !
        !     ATOM I (S, P AND D SHELLS) BEING REPELLED BY ATOM J (D SHELL)
        !
        do I=IA,IB
           I2=I1FACT(I)
           F(I2)=F(I2)+DREP*DPOP(JJ)
        end do
        !
        !    ATOM I (D SHELL) BEING REPELLED BY ATOM J (S AND P SHELLS)
        !
        do I=IC+1,IB
           I2=I1FACT(I)
           F(I2)=F(I2)+DREP*SPPOP(JJ)
        end do
     end do loopJJ
  End do loopII
  !
  RETURN
END SUBROUTINE FOCK2D

SUBROUTINE MNGOVER(NI,NJ,XI,XJ,R,SG)
  !***********************************************************************
  !                                                                      *
  !   gover calculates the overlap integrals using a gaussian expansion  *
  !         sto-6g by r.f. stewart, j. chem. phys., 52 431-438, 1970     *
  !                                                                      *
  !         on input   ni   =  atomic number of first atom               *
  !                    nj   =  atomic number of second atom              *
  !                    r    =  interatomic distance in angstroms         *
  !         on exit    s    =  9x9 array of overlaps, in order s,px,py,  *
  !                            pz                                        *
  !                                                                      *
  !***********************************************************************
  !
  use chm_kinds
  use number
  use sizes
  use consta
  use quantm, only : NZTYPE_anal,C_tempm,Z_tempm
  use am1parm,only : NGAUSS,NATORB
  implicit none
  real(chm_real) S(6,6), XI(3), XJ(3), SG(9,9)
  !
  INTEGER KA,KB,NI,NJ,IS,I,J,K,L,IFA,IFB,ILA,ILB,NAT,NBT
  real(chm_real)  SS,TOMB,ADB,R,AMB,ABN,APB
  !
  !    FIND START AND END OF GAUSSIAN
  !
  IFA=NZTYPE_anal(NI)*4-3
  IF(C_tempm(IFA+1,1).NE.ZERO)THEN
     ILA=IFA+3
  ELSE
     ILA=IFA
  ENDIF
  IFB=NZTYPE_anal(NJ)*4-3
  IF(C_tempm(IFB+1,1).NE.ZERO)THEN
     ILB=IFB+3
  ELSE
     ILB=IFB
  ENDIF
  !
  !  CONVERT R INTO AU
  !
  R=R/BOHRR
  R = R**2
  KA=0
  Do I=IFA,ILA
     KA=KA+1
     NAT=KA-1
     KB=0
     do J=IFB,ILB
        KB=KB+1
        NBT=KB-1
        !
        !           decine is it an S-S, S-P, P-S, or P-P overlap 
        IF(NAT.GT.0.AND.NBT.GT.0) THEN          ! P-P
           IS=4
           TOMB=(XI(NAT)-XJ(NAT))*(XI(NBT)-XJ(NBT))*3.5711928576D0
        ELSEIF(NAT.GT.0) THEN                   ! P-S
           IS=3
           TOMB=(XI(NAT)-XJ(NAT))*1.88976D0
        ELSEIF(NBT.GT.0) THEN                   ! S-P
           IS=2
           TOMB=(XI(NBT)-XJ(NBT))*1.88976D0
        ELSE                                    ! S-S
           IS=1
        ENDIF
        do K=1,NGAUSS
           do L=1,NGAUSS
              S(K,L)=ZERO
              AMB=Z_tempm(I,K)+Z_tempm(J,L)
              APB=Z_tempm(I,K)*Z_tempm(J,L)
              ADB=APB/AMB
              !
              !           CHECK OF OVERLAP IS NON-ZERO BEFORE STARTING
              !
              IF((ADB*R).LT.90.D0) THEN
                 ABN=ONE
                 select case (IS)        ! only 1, 2, 3, 4
                 case (1)
                    EXIT
                 case (2)
                    ABN=TWO*TOMB*Z_tempm(I,K)*SQRT(Z_tempm(J,L))/AMB
                 case (3)
                    ABN=-TWO*TOMB*Z_tempm(J,L)*SQRT(Z_tempm(I,K))/AMB
                 case (4)
                    ABN=-ADB*TOMB
                    IF(NAT.EQ.NBT) ABN=ABN+HALF
                    ABN=FOUR*ABN*SQRT(APB)/AMB
                 end select
                 S(K,L)=SQRT((TWO*SQRT(APB)/AMB)**3)*EXP(-ADB*R)*ABN
              ENDIF
           end do
        end do
        SG(KA,KB)=ZERO
        do K=1,NGAUSS
           do L=1,NGAUSS
              SG(KA,KB)=SG(KA,KB)+S(K,L)*C_tempm(I,K)*C_tempm(J,L)
           end do
        end do
     end do  ! J=IFB,ILB
  End do     ! I=IFA,ILA
  RETURN
END SUBROUTINE MNGOVER

SUBROUTINE MNH1ELEC(NI,NJ,XI,XJ,SMAT)
  !
  use chm_kinds
  use number
  use am1parm, only : NATORB,BETAS,BETAP,BETAD,VS,VP,VD
  use quantm, only : KEYWRD
  implicit none
  INTEGER NI,NJ
  real(chm_real) XI(3),XJ(3),SMAT(9,9), BI(9), BJ(9)
  !***********************************************************************
  !
  !  H1ELEC FORMS THE ONE-ELECTRON MATRIX BETWEEN TWO ATOMS.
  !
  !   ON INPUT    NI   = ATOMIC NO. OF FIRST ATOM.
  !               NJ   = ATOMIC NO. OF SECOND ATOM.
  !               XI   = COORDINATES OF FIRST ATOM.
  !               XJ   = COORDINATES OF SECOND ATOM.
  !
  !   ON OUTPUT   SMAT = MATRIX OF ONE-ELECTRON INTERACTIONS.
  !
  !***********************************************************************
  real(chm_real) SBITS(9,9), XJUC(3)
  INTEGER LIMS(3,2)
  LOGICAL FIRST
  !
  INTEGER I,J,NORBI,NORBJ, NINI,NJNJ
  real(chm_real) SS
  !
  DATA FIRST/.TRUE./

  !      IF(NI.EQ.0.OR.NJ.EQ.0)THEN
  !         IF(SQRT((XI(1)-XJ(1))**2+(XI(2)-XJ(2))**2+(XI(3)-XJ(3))**2) .GT.1.8D0)THEN
  !            SMAT(1:9,1:9)=ZERO
  !            RETURN
  !         ENDIF
  !      ENDIF
  !
  !   USE STANDARD CARBON PARAMETERS TO CALCULATE THE OVERLAP INTEGRAL FOR QM GHO-BOUNDARY ATOMS...JG
  !
  NINI = NI
  NJNJ = NJ
  IF(NINI.GT.90) NINI = 6
  IF(NJNJ.GT.90) NJNJ = 6
  CALL MNDIAT(NINI,NJNJ,XI,XJ,SMAT)
  !
  BI(1)=BETAS(NI)*HALF
  BI(2)=BETAP(NI)*HALF
  BI(3:4)=BI(2)
  BI(5)=BETAD(NI)*HALF
  BI(6:9)=BI(5)
  BJ(1)=BETAS(NJ)*HALF
  BJ(2)=BETAP(NJ)*HALF
  BJ(3:4)=BJ(2)
  BJ(5)=BETAD(NJ)*HALF
  BJ(6:9)=BJ(5)
  NORBI=NATORB(NI)
  NORBJ=NATORB(NJ)
  IF(NORBI.EQ.9.OR.NORBJ.EQ.9) THEN
     !
     !    IN THE CALCULATION OF THE ONE-ELECTRON TERMS THE GEOMETRIC MEAN
     !    OF THE TWO BETA VALUES IS BEING USED IF ONE OF THE ATOMS
     !    CONTAINS D-ORBITALS.
     do J=1,NORBJ
        SMAT(1:NORBI,J)=-TWO*SMAT(1:NORBI,J)*SQRT(BI(1:NORBI)*BJ(J))
     end do
  ELSE
     do J=1,NORBJ
        SMAT(1:NORBI,J)=SMAT(1:NORBI,J)*(BI(1:NORBI)+BJ(J))
     end do
  ENDIF
  RETURN
END SUBROUTINE MNH1ELEC

FUNCTION MNHELECT(N,P,H,F,QESCF)
  !
  use chm_kinds
  use dimens_fcm
  use consta, only : BOHRR,EV_TO_KCAL,AU_TO_EV
  use number
#if KEY_PARALLEL==1
  use parallel  
#endif
  !
  ! namkh 08/08/04
  ! QM/MM-Ewald
  use quantm
  implicit none
  INTEGER N
  real(chm_real) P(*), H(*), F(*), mnhelect
  !
  !
  !***********************************************************************
  !
  !    SUBROUTINE CALCULATES THE ELECTRONIC ENERGY OF THE SYSTEM IN EV.
  !
  !    ON ENTRY N = NUMBER OF ATOMIC ORBITALS.
  !             P = DENSITY MATRIX, PACKED, LOWER TRIANGLE.
  !             H = ONE-ELECTRON MATRIX, PACKED, LOWER TRIANGLE.
  !             F = TWO-ELECTRON MATRIX, PACKED, LOWER TRIANGLE.
  !    ON EXIT
  !        HELECT = ELECTRONIC ENERGY.
  !
  !    NO ARGUMENTS ARE CHANGED.
  !
  !***********************************************************************
  !
  ! namkh 08/08/04
  ! QM/MM-Ewald
  INTEGER IA,IB,IC,I1,I2
  real(chm_real)  ETEMP, A0C, EVC
  LOGICAL QESCF
  !
  INTEGER JJ,I,J,K,NMAX
  real(chm_real)  ED,EE
  !
  ! go parallel
  INTEGER ATFRST,ATLAST,NODELE
  INTEGER ATFRS2,ATLAS2,NODEL2
  real(chm_real)  EMPOTF, RFLAMB
  !
  !
  NMAX=(N*(N+1))/2
  ED=ZERO
  EE=ZERO
  !
#if KEY_PARALLEL==1 /*paramain*/
  NODELE = N     / NUMNOD
  NODEL2 = NMAX  / NUMNOD
#if KEY_PARAFULL==1 /*parfmain*/
#if KEY_PARASCAL==1 /*parstest*/
#error  'Illegal parallel compile options'
#endif /* (parstest)*/
  IF(QESCF) THEN
     ATFRST=MYNOD*NODELE + 1
     ATLAST=(MYNOD+1)*NODELE
     !
     ATFRS2=MYNOD*NODEL2 + 1
     ATLAS2=(MYNOD+1)*NODEL2
     !
     IF(MYNOD.EQ.(NUMNOD-1)) THEN
        ATLAST=N
        ATLAS2=NMAX 
     END IF
  ELSE
     ATFRST=1
     ATLAST=N
     !
     ATFRS2=1
     ATLAS2=NMAX
  END IF
#elif KEY_PARASCAL==1 || KEY_SPACDEC==1 /*parfmain*/
  ATFRST=1
  ATLAST=N     
  !
  ATFRS2=1
  ATLAS2=NMAX 
#else /* (parfmain)*/
#error  'Illegal parallel compile options'
#endif /* (parfmain)*/
  !   DTM PI
  IF (QMPI) THEN
     ATFRST=1
     ATLAST=N
     !
     ATFRS2=1
     ATLAS2=NMAX
  ENDIF
#else /* (paramain)*/
  ATFRST=1
  ATLAST=N     
  !
  ATFRS2=1
  ATLAS2=NMAX 
#endif /* (paramain)*/
  !
  do I=ATFRS2,ATLAS2      ! I=1,NMAX
     EE=EE+P(I)*(H(I)+F(I))
  end do
  do I=ATFRST,ATLAST      ! I=1,N
     K=(I*(I+1))/2
     ED=ED+P(K)*(H(K)+F(K))
  end do
  EE=EE-HALF*ED
  !
  !
  ! namkh 08/08/04
  ! QM/MM-Ewald
  !     Until now, the energy for Ewald sum only added half of the term
  !     from MM atoms, and half of them from QM atoms.
  !     QM atoms should contribute only half, but MM atoms are not.
  !     Correct the EMPOT contribution and FOCK matrix back.
  !
  IF(LQMEWD.AND.QESCF) THEN
     ETEMP = ZERO
     A0C   = BOHRR       ! 0.529167D0
     EVC   = AU_TO_EV    ! 27.21D0

     RFLAMB = ONE
     IF(QMPERT) RFLAMB=RLAMBF

     DO I = 1, NATQM
        IA= N1ST(I)
        IB= NLAST(I)

        EMPOTF = RFLAMB*EMPOT(I)

        !CC            IF(QMPERT) THEN
        !CC               EMPOTF = RLAMBF*EMPOT(I)
        !CC            ELSE
        !CC               EMPOTF = EMPOT(I)
        !CC            END IF

        DO I1 = IA, IB
           I2 = I1*(I1-1)/2+I1
           ETEMP=ETEMP-EMPOTF*P(I2)*A0C*EVC
           ! previous before QMPERT implementation
           !              ETEMP=ETEMP-EMPOT(I)*P(I2)*A0C*EVC
        END DO
     END DO
     EE = EE + HALF*ETEMP
  END IF
  !
  MNHELECT=EE
  !
  RETURN
END FUNCTION MNHELECT

SUBROUTINE HQRII(A,N,M,E,V)
  !
  use chm_kinds
  use number
  use sizes
  use stream
  implicit none
  INTEGER N,M
  real(chm_real) A(*), E(N), V(N,M)
  !*************************************************************
  !*
  !* HQRII IS A DIAGONALISATION ROUTINE, WRITTEN BY YOSHITAKA BEPPU OF
  !*       NAGOYA UNIVERSITY, JAPAN.
  !*       FOR DETAILS SEE 'COMPUTERS & CHEMISTRY' VOL.6 1982. PAGE 000.
  !*
  !* ON INPUT    A       = MATRIX TO BE DIAGONALISED
  !*             N       = SIZE OF MATRIX TO BE DIAGONALISED.
  !*             M       = NUMBER OF EIGENVECTORS NEEDED.
  !*             E       = ARRAY OF SIZE AT LEAST N
  !*             V       = ARRAY OF SIZE AT LEAST NMAX*M
  !*
  !* ON OUTPUT   E       = EIGENVALUES
  !*             V       = EIGENVECTORS IN ARRAY OF SIZE NMAX*M
  !*
  !************************************************************************
  real(chm_real) W(5,MAXPAR)
  !
  INTEGER IG,II,KK,LL,IORD,IM1,IP1,KP1,NM1,NM2,I,J,K,L,ITERE
  real(chm_real) EPS1,EPS2,EPS3,EE,FF,RA,FN,RN,VN,WW,SEPS,SINV,SUMM, &
       GERSCH,C,H,R,S,T,U,Z,DEL,SORTER,EPS,SUM
  !
  IF(N.LE.1 .OR. M .LE.1 .OR. M .GT. N) THEN
     IF(N.EQ.1 .AND. M.EQ.1) THEN
        E(1)=A(1)
        V(1,1)=ONE
        RETURN
     ENDIF
     IF(PRNLEV.GE.2) WRITE(6,'(////10X,''IN HQRII, N ='',I4,'' M ='',I4)') N,M
     CALL WRNDIE(-5,'<HQRII>','INVALID PARAMETERS.')
  ENDIF
  !*
  !* EPS3 AND EPS ARE MACHINE-PRECISION DEPENDENT
  !*
  EPS3=1.D-30
  LL=(N*(N+1))/2+1
  EPS=1.D-8
  IORD=-1
  NM1=N-1
  if(N.ne.2) then
     NM2=N-2
     !     HOUSEHOLDER TRANSFORMATION
     do K=1,NM2
        KP1=K+1
        W(2,K)=A((K*(K+1))/2)
        SUM=zero
        do J=KP1,N
           W(2,J)=A((J*(J-1))/2+K)
           SUM=W(2,J)**2+SUM
        end do
        S=SIGN(SQRT(SUM),W(2,KP1))
        W(1,K)=-S
        W(2,KP1)=W(2,KP1)+S
        A(K+(KP1*(KP1-1))/2)=W(2,KP1)
        H=W(2,KP1)*S
        if(ABS(H).GE.1.D-35) then
           !#             IF(H.EQ.ZERO) cylcle
           SUMM=ZERO
           do I=KP1,N
              SUM=ZERO
              do J=KP1,I
                 SUM=SUM+A(J+(I*(I-1))/2)*W(2,J)
              end do
              if(I.lt.N) then
                 IP1=I+1
                 do J=IP1,N
                    SUM=SUM+A(I+(J*(J-1))/2)*W(2,J)
                 end do
              end if
              W(1,I)=SUM/H
              SUMM=W(1,I)*W(2,I)+SUMM
           end do
           U=SUMM*HALF/H
           do J=KP1,N
              W(1,J)=W(2,J)*U-W(1,J)
              do I=KP1,J
                 A(I+(J*(J-1))/2)=W(1,I)*W(2,J)+W(1,J)*W(2,I)+A(I+(J*(J-1))/2)
              end do
           end do
        end if
        A((K*(K+1))/2)=H
     end do
  end if

  W(2,NM1)=A((NM1*(NM1+1))/2)
  W(2,N)  =A((N*(N+1))/2)
  W(1,NM1)=A(NM1+(N*(N-1))/2)
  W(1,N)  =ZERO
  GERSCH  =ABS(W(2,1))+ABS(W(1,1))
  do I=1,NM1
     GERSCH=MAX(ABS(W(2,I+1))+ABS(W(1,I))+ABS(W(1,I+1)),GERSCH)
  end do
  DEL=EPS*GERSCH
  do I=1,N
     W(3,I)=W(1,I)
     E(I)  =W(2,I)
     V(I,M)=E(I)
  end do

  if(DEL.NE.ZERO) then
     !        QR-METHOD WITH ORIGIN SHIFT
     K=N

     loopOUT: DO
110     L=K
        do while (ABS(W(3,L-1)).LT.DEL)
           L=L-1
           if(L.LE.1) EXIT
        end do
        if(L.ne.K) then
           WW=(E(K-1)+E(K))*HALF
           R =E(K)-WW
           Z =SIGN(SQRT(W(3,K-1)**2+R*R),R)+WW
           EE=E(L)-Z
           E(L)=EE
           FF=W(3,L)
           R =SQRT(EE*EE+FF*FF)
           J =L

           R     =R+1.D-15
           C     =E(J)/R
           S     =W(3,J)/R
           WW    =E(J+1)-Z
           E(J)  =(FF*C+WW*S)*S+EE+Z
           E(J+1)=C*WW-S*FF
           J     =J+1

           loopIN: do
              R     =SQRT(E(J)**2+W(3,J)**2)
              W(3,J-1)=S*R
              EE    =E(J)*C
              FF    =W(3,J)*C
              R     =R+1.D-15
              C     =E(J)/R
              S     =W(3,J)/R
              WW    =E(J+1)-Z
              E(J)  =(FF*C+WW*S)*S+EE+Z
              E(J+1)=C*WW-S*FF
              J     =J+1
              if(L.GE.K) EXIT
           end do loopIN     !

           W(3,K-1)=E(K)*S
           E(K)=E(K)*C+Z
           GOTO 110        ! 
        end if
        K=K-1
        IF(K.LE.1) EXIT     ! IF(K.GT.1) GOTO 110
     End do loopOUT        !
     !*    *    *    *    *    *    *    *    *    *    *    *    *
     !*
     !*   AT THIS POINT THE ARRAY 'E' CONTAINS THE UN-ORDERED EIGENVALUES
     !*
     !*    *    *    *    *    *    *    *    *    *    *    *    *
     !     STRAIGHT SELECTION SORT OF EIGENVALUES
     if(IORD.LT.0) then
        SORTER=-ONE
     else
        SORTER=ONE
     end if
     J=N

     do
        L=1
        II=1
        LL=1
        do I=2,J
           if((E(I)-E(L))*SORTER .GT. ZERO) then
              II=I
              LL=L
           else
              L=I
           end if
        end do
        if(II.ne.LL) then
           WW=E(LL)
           E(LL)=E(II)
           E(II)=WW
        end if
        J=II-1
        IF(J.LT.2) EXIT
     end do
  end if      ! DEL.NE.ZERO 

  IF(M.EQ.0) RETURN

  !***************
  !*  ORDERING OF EIGENVALUES COMPLETE.
  !***************
  !      INVERSE-ITERATION FOR EIGENVECTORS
  FN  =N
  EPS1=1.D-5
  SEPS=SQRT(EPS)
  EPS2=0.05D0
  RN  =ZERO
  RA  =EPS*0.6180339887485D0    ! 0.618... IS THE FIBONACCI NUMBER (-1+SQRT(5))/2.
  ! 
  IG=1
  LoopII: Do I=1,M
     IM1=I-1
     do J=1,N
        W(3,J)=ZERO
        W(4,J)=W(1,J)
        W(5,J)=V(J,M)-E(I)
        RN=RN+RA
        IF(RN.GE.EPS) RN=RN-EPS
        V(J,I)=RN
     end do
     do J=1,NM1
        if(ABS(W(5,J)).lt.ABS(W(1,J))) then
           W(2,J)=-W(5,J)/W(1,J)
           W(5,J)=W(1,J)
           T     =W(5,J+1)
           W(5,J+1)=W(4,J)
           W(4,J) =T
           W(3,J) =W(4,J+1)
           IF(W(3,J).EQ.ZERO) W(3,J)=DEL
           W(4,J+1)=ZERO
        else
           IF(W(5,J).EQ.ZERO) W(5,J)=DEL
           W(2,J)=-W(1,J)/W(5,J)
        end if
        W(4,J+1)=W(3,J)*W(2,J)+W(4,J+1)
        W(5,J+1)=W(4,J)*W(2,J)+W(5,J+1)
     end do
     IF(ABS(W(5,N)) .LT. EPS3) W(5,N)=DEL
     LoopITERE: do ITERE=1,5
        if(ITERE.ne.1) then
           do J=1,NM1
              if(W(3,J).ne.ZERO) then
                 T=V(J,I)
                 V(J,I)=V(J+1,I)
                 V(J+1,I)=T
              end if
              V(J+1,I)=V(J,I)*W(2,J)+V(J+1,I)
           end do
        end if
        V(N,I)  = V(N,I)/W(5,N)
        V(NM1,I)=(V(NM1,I)-V(N,I)*W(4,NM1))/W(5,NM1)
        VN      = MAX(ABS(V(N,I)),ABS(V(NM1,I)),TENM20)
        if(N.ne.2) then
           K=NM2
           do
              V(K,I)=(V(K,I)-V(K+1,I)*W(4,K)-V(K+2,I)*W(3,K))/W(5,K)
              VN=MAX(ABS(V(K,I)),VN,TENM20)
              K=K-1
              if(K.lt.1) EXIT
           end do
        end if
        S=EPS1/VN
        V(1:N,I)=V(1:N,I)*S
        IF(ITERE.GT.1 .AND. VN.GT.1) EXIT LoopITERE
     end do LoopITERE   ! ITERE=1,5

     !     TRANSFORMATION OF EIGENVECTORS
     if(N.ne.2) then
        do J=1,NM2
           K=N-J-1
           if(A((K*(K+1))/2).NE.ZERO) then
              KP1=K+1
              SUM=ZERO
              do KK=KP1,N
                 SUM=SUM+A(K+(KK*(KK-1))/2)*V(KK,I)
              end do
              S=-SUM/A((K*(K+1))/2)
              do KK=KP1,N
                 V(KK,I)=A(K+(KK*(KK-1))/2)*S+V(KK,I)
              end do
           end if
        end do
     end if
     do J=IG,I
        IF(ABS(E(J)-E(I)) .LT. EPS2) GOTO 380
     end do
     J=I
380  IG=J
     if(IG.ne.I) then
        !     RE-ORTHOGONALISATION
        do K=IG,IM1
           SUM=ZERO
           do J=1,N
              SUM=V(J,K)*V(J,I)+SUM
           end do
           S=-SUM
           V(1:N,I)=V(1:N,K)*S+V(1:N,I)
        end do
     end if
     !     NORMALISATION
     SUM=1.D-24
     do J=1,N
        SUM=SUM+V(J,I)**2
     end do
     SINV=ONE/SQRT(SUM)
     V(1:N,I)=V(1:N,I)*SINV
  End do LoopII

  RETURN
END SUBROUTINE HQRII

SUBROUTINE IJKL(I1, I2, J1, J2, ELEM, A1, MDIM)
  !
  use chm_kinds
  use number
  use sizes
  use memory
  use quantm,only : NMECI,XY_meci,WJ_anal,WK_anal
  implicit none
  INTEGER I1,I2,J1,J2,MDIM
  real(chm_real) ELEM,A1(MDIM,NMECI)
  !************************************************************************
  !*
  !*   IJKL FILLS THE TWO-ELECTRON MATRIX XY WITH REPULSION INTEGRALS.
  !*        XY(I,J,K,L) IS THE REPULSION BETWEEN ONE ELECTRON IN
  !*        M.O.S I AND J AND AN ELECTRON IN M.O.S K AND L.
  !*        <I1(1),J1(1)/I2(2),J2(2)>
  !************************************************************************
  !
  real(chm_real) X,Y,Z
  real(chm_real) SPCG
  EXTERNAL SPCG

  ! allocate memory for WJ_anal and WK_anal
  if(N2ELEC .gt. size(WJ_anal) .or. N2ELEC .gt. size(WK_anal)) then
     if(allocated(WJ_anal)) call chmdealloc('qmpac.src','IJKL','WJ_anal',size(WJ_anal),crl=WJ_anal)
     if(allocated(WK_anal)) call chmdealloc('qmpac.src','IJKL','WK_anal',size(WK_anal),crl=WK_anal)
  end if
  if(.not.allocated(WJ_anal)) call chmalloc('qmpac.src','IJKL','WJ_anal',N2ELEC+50,crl=WJ_anal)
  if(.not.allocated(WK_anal)) call chmalloc('qmpac.src','IJKL','WK_anal',N2ELEC+50,crl=WK_anal)
  !
  !
  IF (XY_meci(I1,J1,I2,J2).EQ.HUNDRD) THEN
     X= SPCG(A1(1,I1),A1(1,J1),A1(1,I2),A1(1,J2),WJ_anal,WJ_anal)
     XY_meci(I1,J1,I2,J2)=X
     XY_meci(I1,J1,J2,I2)=X
     XY_meci(J1,I1,I2,J2)=X
     XY_meci(J1,I1,J2,I2)=X
     XY_meci(I2,J2,I1,J1)=X
     XY_meci(I2,J2,J1,I1)=X
     XY_meci(J2,I2,I1,J1)=X
     XY_meci(J2,I2,J1,I1)=X
  ENDIF
  IF (XY_meci(I1,I2,J1,J2).EQ.HUNDRD) THEN
     Z= SPCG(A1(1,I1),A1(1,I2),A1(1,J1),A1(1,J2),WJ_anal,WJ_anal)
     XY_meci(I1,I2,J1,J2)=Z
     XY_meci(I1,I2,J2,J1)=Z
     XY_meci(I2,I1,J1,J2)=Z
     XY_meci(I2,I1,J2,J1)=Z
     XY_meci(J1,J2,I1,I2)=Z
     XY_meci(J1,J2,I2,I1)=Z
     XY_meci(J2,J1,I1,I2)=Z
     XY_meci(J2,J1,I2,I1)=Z
  ENDIF
  IF (XY_meci(I1,J2,I2,J1).EQ.HUNDRD) THEN
     Y= SPCG(A1(1,I1),A1(1,J2),A1(1,I2),A1(1,J1),WJ_anal,WJ_anal)
     XY_meci(I1,J2,I2,J1)=Y
     XY_meci(I1,J2,J1,I2)=Y
     XY_meci(J2,I1,I2,J1)=Y
     XY_meci(J2,I1,J1,I2)=Y
     XY_meci(I2,J1,I1,J2)=Y
     XY_meci(I2,J1,J2,I1)=Y
     XY_meci(J1,I2,I1,J2)=Y
     XY_meci(J1,I2,J2,I1)=Y
  ENDIF
  X=XY_meci(I1,J1,I2,J2)
  Y=XY_meci(I1,J2,J1,I2)
  ELEM=X-Y
  RETURN
END SUBROUTINE IJKL

SUBROUTINE INTERP(N,NP,NQ,MODE,E,FP,CP,VEC,FOCK,P,H,VECL)
  !
  use chm_kinds
  use number
  use consta, only : EV_TO_KCAL
  use sizes
  use stream
#if KEY_PARALLEL==1
  use parallel   
#endif
  use quantm, only : KEYWRD,NPNTS_fit,XLOW_fit,XHIGH_fit, &
       XMIN_fit,EMIN_fit,DEMIN_fit,X_fit,F_fit,DF_fit
  implicit none
  INTEGER N,NP,NQ,MODE
  real(chm_real) E
  real(chm_real) FP(MPACK), CP(N,N)
  real(chm_real) VEC(N,N), FOCK(N,N), P(N,N), H(N*N), VECL(N*N)
  !
  !
  !**********************************************************************
  !*
  !* INTERP: AN INTERPOLATION PROCEDURE FOR FORCING SCF CONVERGANCE
  !*         ORIGINAL THEORY AND FORTRAN WRITTEN BY R.N. CAMP AND
  !*         H.F. KING, J. CHEM. PHYS. 75, 268 (1981)
  !**********************************************************************
  !*
  !* ON INPUT N     = NUMBER OF ORBITALS
  !*          NP    = NUMBER OF FILLED LEVELS
  !*          NQ    = NUMBER OF EMPTY LEVELS
  !*          MODE  = 1, DO NOT RESET.
  !*          E     = ENERGY
  !*          FP    = FOCK MATRIX, AS LOWER HALF TRIANGLE, PACKED
  !*          CP    = EIGENVECTORS OF FOCK MATRIX OF ITERATION -1
  !*                  AS PACKED ARRAY OF N*N COEFFICIENTS
  !*
  !* ON OUTPUT CP   = BEST GUESSED SET OF EIGENVECTORS
  !*           MODE = 2 OR 3 - USED BY CALLING PROGRAM
  !**********************************************************************
  real(chm_real) THETA(MAXORB)
  INTEGER IA(MAXORB)
  !
  INTEGER MINPQ,MAXPQ,II,IJ,IK,IL,NP1,NP2,I,J,K,IPOINT,I1,J1,K1,K2
  real(chm_real) :: CK,SK,EOLD,DEDX,ROLD,RMIN,XOLD,ENOW,RNOW,XNOW,DUM,DEOLD,DENOW
  real(chm_real) :: FF=0.9D0, RAAMAX=1.5708D0
  !
  logical, save :: FIRST=.TRUE.
  logical, save :: DEBUG1,DEBUG2


  IF(FIRST)THEN
     DEBUG2=(INDEX(KEYWRD,'INTERP').NE.0)
     DEBUG1=(INDEX(KEYWRD,'DEBUG').NE.0.AND.DEBUG2)
     FIRST=.FALSE.
  ENDIF
  !
  DO I=1,MAXORB
     IA(I)=(I*I-I)/2
  END DO
  !
  !     RADMAX=MAXIMUM ROTATION ANGLE (RADIANS).  1.5708 = 90 DEGREES.
  !         FF=FACTOR FOR CONVERGENCE TEST FOR 1D SEARCH.
  !
  MINPQ=MIN0(NP,NQ)
  MAXPQ=MAX0(NP,NQ)
  NP1=NP+1
  NP2=MAX0(1,NP/2)

  If(MODE.ne.2) then
     !
     !     (MODE=1 OR 3 ENTRY)
     !     TRANSFORM FOCK MATRIX TO CURRENT MO BASIS.
     !     ONLY THE OFF DIAGONAL OCC-VIRT BLOCK IS COMPUTED.
     !     STORE IN FOCK ARRAY
     !
     II=0
     do I=1,N
        I1=I+1
        do J=1,NQ
           DUM=ZERO
           do K=1,I
              DUM=DUM+FP(II+K)*CP(K,J+NP)
           end do
           if(I.ne.N) then
              IK=II+I+I
              do K=I1,N
                 DUM=DUM+FP(IK)*CP(K,J+NP)
                 IK=IK+K
              end do
           end if
           P(I,J)=DUM
        end do
        II=II+I
     end do
     do I=1,NP
        do J=1,NQ
           DUM=ZERO
           do K=1,N
              DUM=DUM+CP(K,I)*P(K,J)
           end do
           FOCK(I,J)=DUM
        end do
     end do
     if(MODE.ne.3) then
        !
        !     CURRENT POINT BECOMES OLD POINT (MODE=1 ENTRY)
        !
        VEC(1:N,1:N)=CP(1:N,1:N)
        EOLD=E
        XOLD=ONE
        MODE=2
        RETURN
     end if
     !
     !     (MODE=3 ENTRY)
     !     FOCK CORRESPONDS TO CURRENT POINT IN CORRESPONDING REPRESENTATION.
     !     VEC DOES NOT HOLD CURRENT VECTORS. VEC SET IN LAST MODE=2 ENTRY.
     !
     NPNTS_fit=NPNTS_fit+1
     IF(DEBUG2.AND.PRNLEV.GE.2) WRITE(6,'(''   INTERPOLATED ENERGY:'',F13.6)') E*EV_TO_KCAL
     IPOINT=NPNTS_fit
  Else        ! MODE.ne.2
     !
     !    (MODE=2 ENTRY) CALCULATE THETA, AND U, V, W MATRICES.
     !                   U ROTATES CURRENT INTO OLD MO.
     !                   V ROTATES CURRENT INTO CORRESPONDING CURRENT MO.
     !                   W ROTATES OLD INTO CORRESPONDING OLD MO.
     !
     J1=1
     do I=1,N
        IF(I.EQ.NP1) J1=NP1
        do J=J1,N
           P(I,J)=ZERO
           do K=1,N
              P(I,J)=P(I,J)+CP(K,I)*VEC(K,J)
           end do
        end do
     end do
     !
     !     U = CP(DAGGER)*VEC IS NOW IN P ARRAY.
     !     VEC IS NOW AVAILABLE FOR TEMPORARY STORAGE.
     !
     IJ=0
     do I=1,NP
        do J=1,I
           IJ=IJ+1
           H(IJ)=ZERO
           do K=NP1,N
              H(IJ)=H(IJ)+P(I,K)*P(J,K)
           end do
        end do
     end do
     CALL HQRII(H,NP,NP,THETA,VECL)
     do I=NP,1,-1
        IL=I*NP-NP
        do J=NP,1,-1
           VEC(J,I)=VECL(J+IL)
        end do
     end do
     do I=1,NP2
        DUM=THETA(NP1-I)
        THETA(NP1-I)=THETA(I)
        THETA(I)=DUM
        do J=1,NP
           DUM=VEC(J,NP1-I)
           VEC(J,NP1-I)=VEC(J,I)
           VEC(J,I)=DUM
        end do
     end do
     do I=1,MINPQ
        THETA(I)=MAX(THETA(I),ZERO)
        THETA(I)=MIN(THETA(I),ONE)
        THETA(I)=ASIN(SQRT(THETA(I)))
     end do
     !
     !     THETA MATRIX HAS NOW BEEN CALCULATED, ALSO UNITARY VP MATRIX
     !     HAS BEEN CALCULATED AND STORED IN FIRST NP COLUMNS OF VEC MATRIX.
     !     NOW COMPUTE WQ
     !
     do I=1,NQ
        do J=1,MINPQ
           VEC(I,NP+J)=ZERO
           do K=1,NP
              VEC(I,NP+J)=VEC(I,NP+J)+P(K,NP+I)*VEC(K,J)
           end do
        end do
     end do
     CALL SCHMIT(VEC(1,NP1),NQ,N)
     !
     !     UNITARY WQ MATRIX NOW IN LAST NQ COLUMNS OF VEC MATRIX.
     !     TRANSPOSE NP BY NP BLOCK OF U STORED IN P
     !
     do I=1,NP
        do J=1,I
           DUM=P(I,J)
           P(I,J)=P(J,I)
           P(J,I)=DUM
        end do
     end do
     !
     !     CALCULATE WP MATRIX AND HOLD IN FIRST NP COLUMNS OF P
     !
     do I=1,NP
        H(1:NP)=P(I,1:NP)
        do J=1,NP
           P(I,J)=ZERO
           do K=1,NP
              P(I,J)=P(I,J)+H(K)*VEC(K,J)
           end do
        end do
     end do
     CALL SCHMIB(P,NP,N)
     !
     !     CALCULATE VQ MATRIX AND HOLD IN LAST NQ COLUMNS OF P MATRIX.
     !
     do I=1,NQ
        H(1:NQ)=P(NP+I,NP+1:NP+NQ)
        do J=NP1,N
           P(I,J)=ZERO
           do K=1,NQ
              P(I,J)=P(I,J)+H(K)*VEC(K,J)
           end do
        end do
     end do
     CALL SCHMIB(P(1,NP1),NQ,N)
     !
     !     CALCULATE (DE/DX) AT OLD POINT
     !
     DEDX=ZERO
     do I=1,NP
        do J=1,NQ
           DUM=ZERO
           do K=1,MINPQ
              DUM=DUM+THETA(K)*P(I,K)*VEC(J,NP+K)
           end do
           DEDX=DEDX+DUM*FOCK(I,J)
        end do
     end do
     !
     !     STORE OLD POINT INFORMATION FOR SPLINE FIT
     !
     DEOLD    =-four*DEDX
     X_fit(2) = XOLD
     F_fit(2) = EOLD
     DF_fit(2)= DEOLD
     !
     !     MOVE VP OUT OF VEC ARRAY INTO FIRST NP COLUMNS OF P MATRIX.
     !
     P(1:NP,1:NP)=VEC(1:NP,1:NP)
     K1=0
     K2=NP
     do J=1,N
        IF(J.EQ.NP1) K1=NP
        IF(J.EQ.NP1) K2=NQ
        do I=1,N
           DUM=ZERO
           do K=1,K2
              DUM=DUM+CP(I,K1+K)*P(K,J)
           end do
           VEC(I,J)=DUM
        end do
     end do
     !
     !     CORRESPONDING CURRENT MO VECTORS NOW HELD IN VEC.
     !     COMPUTE VEC(DAGGER)*FP*VEC
     !     STORE OFF-DIAGONAL BLOCK IN FOCK ARRAY.
     !
     II=0
     do I=1,N
        I1=I+1
        do J=1,NQ
           DUM=ZERO
           do K=1,I
              DUM=DUM+FP(II+K)*VEC(K,J+NP)
           end do
           if(I.ne.N) then
              IK=II+I+I
              do K=I1,N
                 DUM=DUM+FP(IK)*VEC(K,J+NP)
                 IK=IK+K
              end do
           end if
           P(I,J)=DUM
        end do
        II=II+I
     end do
     do I=1,NP
        do J=1,NQ
           DUM=ZERO
           do K=1,N
              DUM=DUM+VEC(K,I)*P(K,J)
           end do
           FOCK(I,J)=DUM
        end do
     end do
     !
     !     SET LIMITS ON RANGE OF 1-D SEARCH
     !
     NPNTS_fit=2
     IPOINT=1
     XNOW=ZERO
     XHIGH_fit=RAAMAX/THETA(1)
     XLOW_fit=-HALF*XHIGH_fit

  End if     ! MODE.ne.2
  !
  !     CALCULATE (DE/DX) AT CURRENT POINT AND
  !     STORE INFORMATION FOR SPLINE FIT
  !     ***** JUMP POINT FOR MODE=3 ENTRY *****
  !
  DEDX=ZERO
  do K=1,MINPQ
     DEDX=DEDX+THETA(K)*FOCK(K,K)
  end do
  DENOW=-four*DEDX
  ENOW=E
  !
  !     PERFORM 1-D SEARCH AND DETERMINE EXIT MODE.
  !
  X_fit(IPOINT)=XNOW
  F_fit(IPOINT)=ENOW
  DF_fit(IPOINT)=DENOW
  CALL SPLINE
  if((EOLD-ENOW).GT.FF*(EOLD-EMIN_fit).OR.IPOINT.GT.10) then
     continue
  else
     !
     !     (MODE=3 EXIT) RECOMPUTE CP VECTORS AT PREDICTED MINIMUM.
     !
     XNOW=XMIN_fit
     do K=1,MINPQ
        CK=COS(XNOW*THETA(K))
        SK=SIN(XNOW*THETA(K))
        IF(DEBUG2.AND.PRNLEV.GE.2) WRITE(6,'('' ROTATION ANGLE:'',F12.4)') SK*57.29578
        do I=1,N
           CP(I,K)   =CK*VEC(I,K)-SK*VEC(I,NP+K)
           CP(I,NP+K)=SK*VEC(I,K)+CK*VEC(I,NP+K)
        end do
     end do
     MODE=3
     RETURN
  end if
  !
  !     (MODE=2 EXIT) CURRENT VECTORS GIVE SATISFACTORY ENERGY IMPROVEMENT
  !     CURRENT POINT BECOMES OLD POINT FOR THE NEXT 1-D SEARCH.
  !
  if(MODE.ne.2) then
     VEC(1:N,1:N)=CP(1:N,1:N)
     MODE=2
  end if
  ROLD=XOLD*THETA(1)*57.29578D0
  RNOW=XNOW*THETA(1)*57.29578D0
  RMIN=XMIN_fit*THETA(1)*57.29578D0
  IF(DEBUG2.AND.PRNLEV.GE.2) WRITE(6,600) XOLD,EOLD*EV_TO_KCAL,DEOLD,ROLD,XNOW,ENOW*EV_TO_KCAL, &
       DENOW,RNOW,XMIN_fit,EMIN_fit*EV_TO_KCAL,DEMIN_fit,RMIN
  IF(PRNLEV.GE.2) write (6,*) ' +++ Flag_1'
  EOLD=ENOW
  IF(PRNLEV.GE.2) write (6,*) ' +++ Flag_2'
  IF(NPNTS_fit.LE.200) RETURN
  IF(PRNLEV.GE.2) THEN
     write (6,*) ' +++ Flag_3'
     WRITE(6,610)
     do K=1,NPNTS_fit
        WRITE(6,620) K,X_fit(K),F_fit(K),DF_fit(K)
     end do
     WRITE(6,630)
     write (6,*) ' +++ Flag_4'
  END IF
  RETURN

600 FORMAT( &
       /14X,3H X ,10X,6H F(X) ,9X,7H DF/DX ,21H   ROTATION (DEGREES), &
       /10H      OLD ,F10.5,3F15.10, &
       /10H  CURRENT ,F10.5,3F15.10, &
       /10H PREDICTED,F10.5,3F15.10/)
610 FORMAT(3H  K,10H     X(K) ,15H       F(K)    ,10H     DF(K))
620 FORMAT(I3,F10.5,2F15.10)
630 FORMAT(10X)
END SUBROUTINE INTERP

SUBROUTINE MAMULT(A,B,C,N,ONE)
  !
  use chm_kinds
  implicit none
  INTEGER N
  real(chm_real) A(*),B(*),C(*)
  real(chm_real) ONE
  !************************************************************************
  !*
  !*   MAMULT MULTIPLIES A BY B AND PUTS THE RESULT IN C
  !*
  !************************************************************************
  !
  INTEGER II,JJ,KK,I,J,K,L
  real(chm_real)  SUM
  !
  L=0
  do I=1,N
     II=((I-1)*I)/2
     do J=1,I
        JJ=((J-1)*J)/2
        L=L+1
        SUM=0.0D0
        do K=1,J
           SUM=SUM+A(II+K)*B(JJ+K)
        end do
        do K=J+1,I
           SUM=SUM+A(II+K)*B(((K-1)*K)/2+J)
        end do
        do K=I+1,N
           KK=(K*(K-1))/2
           SUM=SUM+A(KK+I)*B(KK+J)
        end do
        C(L)=SUM+ONE*C(L)
     end do
  end do
  RETURN
END SUBROUTINE MAMULT

SUBROUTINE MULT(C,S,VECS,N)
  !
  use chm_kinds
  implicit none
  INTEGER N
  real(chm_real) C(N,*), S(N,*), VECS(N,*)
  !***********************************************************************
  !*
  !*   MULT IS USED IN THE MULLIKEN ANALYSIS ONLY. IT PERFORMS THE
  !*        OPERATION:-
  !*                                   VECS=BACK-TRANSFORMED EIGENVECTORS
  !*        VECS  =  C*S               C   =UN-BACK-TRANSFORMED VECTORS
  !*                                   S   =1/SQRT(OVERLAP MATRIX)
  !*
  !***********************************************************************
  !
  INTEGER I,J,K
  real(chm_real) SUM
  !
  do I=1,N
     do J=1,N
        SUM=0.0D0
        do K=1,N
           SUM=SUM+C(K,I)*S(J,K)
        end do
        VECS(J,I)=SUM
     end do
  end do
  RETURN
END SUBROUTINE MULT

SUBROUTINE OSINV (A,N,D)
  !
  use chm_kinds
  use sizes
  implicit none
  INTEGER N
  real(chm_real) A(*),D
  !************************************************************************
  !*
  !*    OSINV INVERTS A GENERAL SQUARE MATRIX OF ORDER UP TO MAXORB. SEE
  !*          DIMENSION STATEMENTS BELOW.
  !*
  !*   ON INPUT       A = GENERAL SQUARE MATRIX STORED LINEARLY.
  !*                  N = DIMENSION OF MATRIX A.
  !*                  D = VARIABLE, NOT DEFINED ON INPUT.
  !*
  !*   ON OUTPUT      A = INVERSE OF ORIGINAL A.
  !*                  D = DETERMINANT OF ORIGINAL A, UNLESS A WAS SINGULAR,
  !*                      IN WHICH CASE D = 0.0
  !*
  !************************************************************************
  INTEGER L(MAXORB), M(MAXORB)
  !
  INTEGER IJ,JI,IK,KI,JK,KJ,KK,NK,JP,JQ,JR,IZ,I,J,K
  real(chm_real)  BIGA,HOLO
  !
  !     IF THE VALUE OF TOL GIVEN HERE IS UNSUITABLE, IT CAN BE CHANGED.
  real(chm_real), parameter :: TOL=1.D-8
  !
  !
  D=1.0D0
  NK=-N
  loopKK: do K=1,N
     NK=NK+N
     L(K)=K
     M(K)=K
     KK=NK+K
     BIGA=A(KK)
     do J=K,N
        IZ=N*(J-1)
        do I=K,N
           IJ=IZ+I
           if( ABS(BIGA).lt.ABS(A(IJ)) ) then
              BIGA=A(IJ)
              L(K)=I
              M(K)=J
           end if
        end do
     end do
     J=L(K)
     if(J.gt.K) then
        KI=K-N
        do I=1,N
           KI=KI+N
           HOLO=-A(KI)
           JI=KI-K+J
           A(KI)=A(JI)
           A(JI)=HOLO
        end do
     end if
     I=M(K)
     if(I.gt.K) then
        JP=N*(I-1)
        do J=1,N
           JK=NK+J
           JI=JP+J
           HOLO=-A(JK)
           A(JK)=A(JI)
           A(JI)=HOLO
        end do
     end if
     if(ABS(BIGA).lt.TOL) then
        D=0.0D0
        RETURN
     end if
     do I=1,N
        if(I.eq.K) cycle
        IK=NK+I
        A(IK)=A(IK)/(-BIGA)
     end do
     loopII: do I=1,N
        IK=NK+I
        IJ=I-N
        loopJJ: do J=1,N
           IJ=IJ+N
           if(I.eq.K .or. J.eq.K) cycle loopJJ
           KJ=IJ-I+K
           A(IJ)=A(IK)*A(KJ)+A(IJ)
        end do loopJJ
     end do loopII
     KJ=K-N
     do J=1,N
        KJ=KJ+N
        if(J.eq.K) cycle
        A(KJ)=A(KJ)/BIGA
     end do
     D=D*BIGA
     A(KK)=1.0D0/BIGA
  end do loopKK          ! K=1,N
  K=N

  Do
     K=K-1
     if(K.le.0) EXIT
     I=L(K)
     if(I.gt.K) then
        JQ=N*(K-1)
        JR=N*(I-1)
        do J=1,N
           JK=JQ+J
           HOLO=A(JK)
           JI=JR+J
           A(JK)=-A(JI)
           A(JI)=HOLO
        end do
     end if
     J=M(K)
     if(J.gt.K) then
        KI=K-N
        do I=1,N
           KI=KI+N
           HOLO=A(KI)
           JI=KI+J-K
           A(KI)=-A(JI)
           A(JI)=HOLO
        end do
     end if
  End do

  RETURN
  !
END SUBROUTINE OSINV

SUBROUTINE PERM(IPERM,NELS,NMOS,MAXMOS,NPERMS)
  !
  use chm_kinds
  use stream
  implicit none
  INTEGER NELS,NMOS,MAXMOS,NPERMS
  INTEGER IPERM(MAXMOS,60), IADD(20), NEL(20)
  !************************************************************************
  !*
  !*  PERM PERMUTES NELS ENTITIES AMONG NMOS LOCATIONS. THE ENTITIES AND
  !*       LOCATIONS ARE EACH INDISTINGUISHABLE. THE PAULI EXCLUSION
  !*       PRINCIPLE IS FOLLOWED. THE NUMBER OF STATES PRODUCED IS GIVEN
  !*       BY NMOS!/(NELS!*(NMOS-NELS)!).
  !* ON INPUT: NELS  = NUMBER OF INDISTINGUISHABLE ENTITIES
  !*           NMOS  = NUMBER OF INDISTINGUISHABLE LOCATIONS
  !*
  !* ON OUTPUT IPERM = ARRAY OF PERMUTATIONS, A 0 INDICATES NO ENTITY,
  !*                   A 1 INDICATES AN ENTITY.
  !*           NPERM = NUMBER OF PERMUTATIONS.
  !*
  !************************************************************************
  !
  INTEGER I10,I11,I12,I,J,I1,I2,I3,I4,I5,I6,I7,I8,I9
  !
  IF(NELS.GT.NMOS)THEN
     IF(PRNLEV.GE.2) WRITE(6,'('' NUMBER OF PARTICLES,'',I3,'' GREATER THAN NO. '',''OF STATES,'',I3)')NELS,NMOS
     NPERMS=0
     RETURN
  ENDIF
  NPERMS=1
  NEL(1:20)=1000
  NEL(1:NELS)=1

  loopI12: do I12=1-12+NELS,NMOS,NEL(12)
     IADD(12)=I12
     loopI11: do I11=I12+1,NMOS,NEL(11)
        IADD(11)=I11
        loopI10: do I10=I11+1,NMOS,NEL(10)
           IADD(10)=I10
           loopI9: do I9=I10+1,NMOS,NEL(9)
              IADD(9)=I9
              loopI8: do I8=I9+1,NMOS,NEL(8)
                 IADD(8)=I8
                 loopI7: do I7=I8+1,NMOS,NEL(7)
                    IADD(7)=I7
                    loopI6: do I6=I7+1,NMOS,NEL(6)
                       IADD(6)=I6
                       loopI5: do I5=I6+1,NMOS,NEL(5)
                          IADD(5)=I5
                          loopI4: do I4=I5+1,NMOS,NEL(4)
                             IADD(4)=I4
                             loopI3: do I3=I4+1,NMOS,NEL(3)
                                IADD(3)=I3
                                loopI2: do I2=I3+1,NMOS,NEL(2)
                                   IADD(2)=I2
                                   loopI1: do I1=I2+1,NMOS,NEL(1)
                                      IADD(1)=I1
                                      IPERM(1:NMOS,NPERMS)=0
                                      IPERM(IADD(1:NELS),NPERMS)=1
                                      NPERMS=NPERMS+1
                                      IF(NPERMS.GT.61)THEN
                                         IF(PRNLEV.GE.2) WRITE(6,'('' NUMBER OF PERMUTATIONS TOO GREAT, LIMIT 60'')')
                                         EXIT loopI12
                                      ENDIF
                                   end do loopI1
                                end do loopI2
                             end do loopI3
                          end do loopI4
                       end do loopI5
                    end do loopI6
                 end do loopI7
              end do loopI8
           end do loopI9
        end do loopI10
     end do loopI11
  end do loopI12
  NPERMS=NPERMS-1
  RETURN
END SUBROUTINE PERM

SUBROUTINE PULAY(F,P,N,FPPF,FOCK,EMAT,LFOCK,NFOCK,MSIZE,START,PL)
  !
  use chm_kinds
  use number
  use stream
  use quantm, only : KEYWRD
  implicit none
  !
  INTEGER MSIZE,N,LFOCK,NFOCK
  real(chm_real) F(*), P(*), FPPF(*), FOCK(*), PL
  LOGICAL START
  !************************************************************************
  !*
  !*   PULAY USES DR. PETER PULAY'S METHOD FOR CONVERGENCE.
  !*         A MATHEMATICAL DESCRIPTION CAN BE FOUND IN
  !*         "P. PULAY, J. COMP. CHEM. 3, 556 (1982).
  !*
  !* ARGUMENTS:-
  !*         ON INPUT F      = FOCK MATRIX, PACKED, LOWER HALF TRIANGLE.
  !*                  P      = DENSITY MATRIX, PACKED, LOWER HALF TRIANGLE.
  !*                  N      = NUMBER OF ORBITALS.
  !*                  FPPF   = WORKSTORE OF SIZE MSIZE, CONTENTS WILL BE
  !*                           OVERWRITTEN.
  !*                  FOCK   =      "       "              "         "
  !*                  EMAT   = WORKSTORE OF AT LEAST 15**2 ELEMENTS.
  !*                  START  = LOGICAL, = TRUE TO START PULAY.
  !*                  PL     = UNDEFINED ELEMENT.
  !*      ON OUTPUT   F      = "BEST" FOCK MATRIX, = LINEAR COMBINATION
  !*                           OF KNOWN FOCK MATRICES.

  !*                  PL     = MEASURE OF NON-SELF-CONSISTENCY
  !*                         = [F*P] = F*P - P*F.
  !*
  !************************************************************************
  real(chm_real) EMAT(20,20), EVEC(1000), COEFFS(20)
  !
  real(chm_real) DOT1
  EXTERNAL DOT1
  INTEGER II,IL,NFOCK1,I,J,L,LBASE
  real(chm_real)  CONST,D,SUM
  !
  logical, save :: FIRST=.TRUE.
  logical, save :: DEBUG1
  integer, save :: MAXLIM,MFOCK,LINEAR
  !
  IF(FIRST) THEN
     FIRST=.FALSE.
     MAXLIM=6
     DEBUG1=(INDEX(KEYWRD,'DEBUGPULAY') .NE.0)
  ENDIF
  IF(START) THEN
     LINEAR=(N*(N+1))/2
     MFOCK=MSIZE/LINEAR
     IF(MFOCK.GT.MAXLIM)MFOCK=MAXLIM
     IF(DEBUG1.AND.PRNLEV.GE.2) WRITE(6,'('' MAXIMUM SIZE:'',I5)') MFOCK
     NFOCK=1
     LFOCK=1
     START=.FALSE.
  ELSE
     IF(NFOCK.LT.MFOCK) NFOCK=NFOCK+1
     IF(LFOCK.NE.MFOCK)THEN
        LFOCK=LFOCK+1
     ELSE
        LFOCK=1
     ENDIF
  ENDIF
  LBASE=(LFOCK-1)*LINEAR
  ! 
  ! FIRST, STORE FOCK MATRIX FOR FUTURE REFERENCE.
  ! 
  FOCK(LFOCK:((LINEAR-1)*MFOCK+LFOCK))=F(1:LINEAR)
  ! 
  ! NOW FORM /FOCK*DENSITY-DENSITY*FOCK/, AND STORE THIS IN FPPF
  ! 
  CALL MAMULT(P,F,FPPF(LBASE+1),N,ZERO)
  CALL MAMULT(F,P,FPPF(LBASE+1),N,MINONE)
  ! 
  ! FPPF NOW CONTAINS THE RESULT OF FP - PF.
  ! 
  NFOCK1=NFOCK+1
  do I=1,NFOCK
     EMAT(NFOCK1,I)=-ONE
     EMAT(I,NFOCK1)=-ONE
     EMAT(LFOCK,I) = DOT1(FPPF((I-1)*LINEAR+1),FPPF(LBASE+1),LINEAR)
     EMAT(I,LFOCK) = EMAT(LFOCK,I)
  end do
  PL=EMAT(LFOCK,LFOCK)/LINEAR

  EMAT(NFOCK1,NFOCK1)=ZERO

  CONST=ONE/EMAT(LFOCK,LFOCK)
  EMAT(1:NFOCK,1:NFOCK)=EMAT(1:NFOCK,1:NFOCK)*CONST
  IF(DEBUG1.AND.PRNLEV.GE.2) THEN
     WRITE(6,'('' EMAT'')')
     do I=1,NFOCK1
        WRITE(6,'(6E13.6)')(EMAT(J,I),J=1,NFOCK1)
     end do
  ENDIF
  L=0
  do I=1,NFOCK1
     EVEC(L+1:L+NFOCK1)=EMAT(I,1:NFOCK1)
     L=L+NFOCK1
  end do
  CONST=ONE/CONST
  EMAT(1:NFOCK,1:NFOCK)=EMAT(1:NFOCK,1:NFOCK)*CONST

  !*********************************************************************
  !*   THE MATRIX EMAT SHOULD HAVE FORM
  !*
  !*      |<E(1)*E(1)>  <E(1)*E(2)> ...   -1.0|
  !*      |<E(2)*E(1)>  <E(2)*E(2)> ...   -1.0|
  !*      |<E(3)*E(1)>  <E(3)*E(2)> ...   -1.0|
  !*      |<E(4)*E(1)>  <E(4)*E(2)> ...   -1.0|
  !*      |     .            .      ...     . |
  !*      |   -1.0         -1.0     ...    0. |
  !*
  !*   WHERE <E(I)*E(J)> IS THE SCALAR PRODUCT OF [F*P] FOR ITERATION I
  !*   TIMES [F*P] FOR ITERATION J.
  !*
  !*********************************************************************
  CALL OSINV(EVEC,NFOCK1,D)
  IF(ABS(D).LT.1.D-6)THEN
     START=.TRUE.
     RETURN
  ENDIF
  IF(NFOCK.LT.2) RETURN
  IL=NFOCK*NFOCK1
  COEFFS(1:NFOCK)=-EVEC(1+IL:NFOCK+IL)
  IF(DEBUG1.AND.PRNLEV.GE.2) THEN
     WRITE(6,'('' EVEC'')')
     WRITE(6,'(6F12.6)')(COEFFS(I),I=1,NFOCK)
     WRITE(6,'(''    LAGRANGIAN MULTIPLIER (ERROR) ='',F13.6)') EVEC(NFOCK1*NFOCK1) 
  ENDIF
  do I=1,LINEAR
     SUM=ZERO
     L=0
     II=(I-1)*MFOCK
     do J=1,NFOCK
        SUM=SUM+COEFFS(J)*FOCK(J+II)
     end do
     F(I)=SUM
  end do
  RETURN
END SUBROUTINE PULAY

SUBROUTINE MNREPP(NI,NJ,RIJ,RI,CORE_local)
  !
  use chm_kinds
  use number
  use consta
  use am1parm, only : NATORB,CORE,DD,QQ,bdd      ! bdd(*,1)=AM(0:100)
  ! bdd(*,2)=AD(0:100)
  ! bdd(*,3)=AQ(0:100) 
  implicit none
  INTEGER NI,NJ
  real(chm_real) RI(22),CORE_local(4,2),RIJ
  !***********************************************************************
  !
  !  REPP CALCULATES THE TWO-ELECTRON REPULSION INTEGRALS AND THE
  !       NUCLEAR ATTRACTION INTEGRALS.
  !
  !     ON INPUT RIJ     = INTERATOMIC DISTANCE
  !              NI      = ATOM NUMBER OF FIRST ATOM
  !              NJ      = ATOM NUMBER OF SECOND ATOM
  !    (REF)     ADD     = ARRAY OF GAMMA, OR TWO-ELECTRON ONE-CENTER,
  !                        INTEGRALS.
  !    (REF)     TORE    = ARRAY OF NUCLEAR CHARGES OF THE ELEMENTS
  !    (REF)     DD      = ARRAY OF DIPOLE CHARGE SEPARATIONS
  !    (REF)     QQ      = ARRAY OF QUADRUPOLE CHARGE SEPARATIONS
  !
  !     THE COMMON BLOCKS ARE INITIALISED IN BLOCK-DATA, AND NEVER CHANGED
  !
  !    ON OUTPUT RI      = ARRAY OF TWO-ELECTRON REPULSION INTEGRALS
  !              CORE    = 4 X 2 ARRAY OF ELECTRON-CORE ATTRACTION
  !                        INTEGRALS
  !
  !***********************************************************************
  !
  INTEGER I
  real(chm_real) DA,DB,EE,FD,QA,OD,QB,DXQXZ,DZQXX,QXXDZ,QXZDX,TD, &
       DZQZZ,QZZDZ,DXDX,DZDZ,EQXX,QXXE,EQZZ,QZZE, &
       ADE,AED,AEE,ADQ,AQD,AEQ,AQE,AQQ,DZE,EDZ, &
       QXXQXX,QXXQYY,QXYQXY,QXXQZZ,QXZQXZ,QZZQXX,QZZQZZ
  !
  real(chm_real) :: R
  real(chm_real),parameter :: PP=TWO, P2=FOUR, P3=EIGHT, P4=16.0D00, &
       RPP= 0.50D0, RP2= 0.25D0, RP3= 0.125D0, RP4= 0.0625D0

  !
  ! *** THIS ROUTINE COMPUTES THE TWO-CENTRE REPULSION INTEGRALS AND THE
  ! *** NUCLEAR ATTRACTION INTEGRALS.
  ! *** THE TWO-CENTRE REPULSION INTEGRALS (OVER LOCAL COORDINATES) ARE
  ! *** STORED AS FOLLOWS (WHERE P-SIGMA = O,  AND P-PI = P AND P* )
  !     (SS/SS)=1,   (SO/SS)=2,   (OO/SS)=3,   (PP/SS)=4,   (SS/OS)=5,
  !     (SO/SO)=6,   (SP/SP)=7,   (OO/SO)=8,   (PP/SO)=9,   (PO/SP)=10,
  !     (SS/OO)=11,  (SS/PP)=12,  (SO/OO)=13,  (SO/PP)=14,  (SP/OP)=15,
  !     (OO/OO)=16,  (PP/OO)=17,  (OO/PP)=18,  (PP/PP)=19,  (PO/PO)=20,
  !     (PP/P*P*)=21,   (P*P/P*P)=22.
  ! *** THE STORAGE OF THE NUCLEAR ATTRACTION INTEGRALS  CORE(KL/IJ) IS
  !     (SS/)=1,   (SO/)=2,   (OO/)=3,   (PP/)=4
  !     WHERE IJ=1 IF THE ORBITALS CENTRED ON ATOM I,  =2 IF ON ATOM J.
  ! *** NI AND NJ ARE THE ATOMIC NUMBERS OF THE TWO ELEMENTS.
  !
  R=RIJ/BOHRR
  RI(1:22)           =ZERO
  CORE_local(1:4,1:2)=ZERO
  !
  !     ATOMIC UNITS ARE USED IN THE CALCULATION
  !     DEFINE CHARGE SEPARATIONS.
  !
  DA = DD(NI)
  DB = DD(NJ)
  QA = QQ(NI)
  QB = QQ(NJ)
  TD = TWO
  OD = ONE
  FD = FOUR
  !
  !     HYDROGEN - HYDROGEN
  !
  AEE  =PT25*(OD/bdd(NI,1)+OD/bdd(NJ,1))**2         ! 0.25D00
  EE   =OD/SQRT(R**2+AEE)
  RI(1)=EE*AU_TO_EV        ! 27.21D00
  CORE_local(1,1)=CORE(NJ)*RI(1)
  CORE_local(1,2)=CORE(NI)*RI(1)
  IF (NATORB(NI).LT.3.AND.NATORB(NJ).LT.3) RETURN

  if(NATORB(NI).ge.3) then
     !
     !     HEAVY ATOM - HYDROGEN
     !
     ADE=PT25*(OD/bdd(NI,2)+OD/bdd(NJ,1))**2
     AQE=PT25*(OD/bdd(NI,3)+OD/bdd(NJ,1))**2
     DZE=-OD/SQRT((R+DA)**2+ADE)+OD/SQRT((R-DA)**2+ADE)
     QZZE=OD/SQRT((R-TD*QA)**2+AQE)-TD/SQRT(R**2+AQE)+OD/SQRT((R+TD*QA)**2+AQE)
     QXXE=TD/SQRT(R**2+FD*QA**2+AQE)-TD/SQRT(R**2+AQE)
     !
     DZE =RPP*DZE      ! /PP
     QXXE=RP2*QXXE     ! /P2
     QZZE=RP2*QZZE     ! /P2
     ! 
     RI(2)=-DZE
     RI(3)=EE+QZZE
     RI(4)=EE+QXXE
     IF (NATORB(NJ).LT.3) GO TO 40
  end if
  !
  !     HYDROGEN - HEAVY ATOM
  !
  AED=PT25*(OD/bdd(NI,1)+OD/bdd(NJ,2))**2
  AEQ=PT25*(OD/bdd(NI,1)+OD/bdd(NJ,3))**2
  EDZ=-OD/SQRT((R-DB)**2+AED)+OD/SQRT((R+DB)**2+AED)
  EQZZ=OD/SQRT((R-TD*QB)**2+AEQ)-TD/SQRT(R**2+AEQ)+OD/SQRT((R+TD*QB)**2+AEQ)
  EQXX=TD/SQRT(R**2+FD*QB**2+AEQ)-TD/SQRT(R**2+AEQ)
  !
  EDZ =RPP*EDZ
  EQXX=RP2*EQXX
  EQZZ=RP2*EQZZ
  ! 
  RI(5)=-EDZ
  RI(11)=EE+EQZZ
  RI(12)=EE+EQXX

  If(NATORB(NI).ge.3) then
     !
     !     HEAVY ATOM - HEAVY ATOM
     !     CAUTION. ADD REPLACES ADD(1,1) IN /MULTIP/ AND MUST BE RESET.
     !
     bdd(1,1)=PT25*(OD/bdd(NI,2)+OD/bdd(NJ,2))**2
     ADQ     =PT25*(OD/bdd(NI,2)+OD/bdd(NJ,3))**2
     AQD     =PT25*(OD/bdd(NI,3)+OD/bdd(NJ,2))**2
     AQQ     =PT25*(OD/bdd(NI,3)+OD/bdd(NJ,3))**2
     DXDX=TD/SQRT(R**2+(DA-DB)**2+bdd(1,1))-TD/SQRT(R**2+(DA+DB)**2+bdd(1,1))
     DZDZ=OD/SQRT((R+DA-DB)**2+bdd(1,1))+ OD/SQRT((R-DA+DB)**2+bdd(1,1)) &
          - OD/SQRT((R-DA-DB)**2+bdd(1,1))-OD/SQRT((R+DA+DB)**2+bdd(1,1))

     DZQXX=-TD/SQRT((R+DA)**2+FD*QB**2+ADQ) + TD/SQRT((R-DA)**2+FD*QB**2+ADQ) &
          + TD/SQRT((R+DA)**2+ADQ) - TD/SQRT((R-DA)**2+ADQ)

     QXXDZ=-TD/SQRT((R-DB)**2+FD*QA**2+AQD) + TD/SQRT((R+DB)**2+FD*QA**2+AQD) &
          + TD/SQRT((R-DB)**2+AQD)-TD/SQRT((R+DB)**2+AQD)

     DZQZZ=-OD/SQRT((R+DA-TD*QB)**2+ADQ) + OD/SQRT((R-DA-TD*QB)**2+ADQ) &
          - OD/SQRT((R+DA+TD*QB)**2+ADQ) + OD/SQRT((R-DA+TD*QB)**2+ADQ) &
          - TD/SQRT((R-DA)**2+ADQ)+TD/SQRT((R+DA)**2+ADQ)

     QZZDZ=-OD/SQRT((R+TD*QA-DB)**2+AQD) + OD/SQRT((R+TD*QA+DB)**2+AQD) &
          - OD/SQRT((R-TD*QA-DB)**2+AQD) + OD/SQRT((R-2.D00*QA+DB)**2+AQD) &
          + TD/SQRT((R-DB)**2+AQD) - TD/SQRT((R+DB)**2+AQD)

     QXXQXX=TD/SQRT(R**2+FD*(QA-QB)**2+AQQ) + TD/SQRT(R**2+FD*(QA+QB)**2+AQQ) &
          - FD/SQRT(R**2+FD*QA**2+AQQ) - FD/SQRT(R**2+FD*QB**2+AQQ)+FD/SQRT(R**2+AQQ)

     QXXQYY=FD/SQRT(R**2+FD*QA**2+FD*QB**2+AQQ) - FD/SQRT(R**2+FD*QA**2+AQQ) &
          - FD/SQRT(R**2+FD*QB**2+AQQ)+FD/SQRT(R**2+AQQ)

     QXXQZZ=TD/SQRT((R-TD*QB)**2+FD*QA**2+AQQ) + TD/SQRT((R+TD*QB)**2+FD*QA**2+AQQ) &
          - TD/SQRT((R-TD*QB)**2+AQQ) - TD/SQRT((R+TD*QB)**2+AQQ) &
          - FD/SQRT(R**2+FD*QA**2+AQQ) + FD/SQRT(R**2+AQQ)

     QZZQXX=TD/SQRT((R+TD*QA)**2+FD*QB**2+AQQ) + TD/SQRT((R-TD*QA)**2+FD*QB**2+AQQ) &
          - TD/SQRT((R+TD*QA)**2+AQQ)-TD/SQRT((R-TD*QA)**2+AQQ) - FD/SQRT(R**2+FD*QB**2+AQQ)+FD/SQRT(R**2+AQQ)

     QZZQZZ=OD/SQRT((R+TD*QA-TD*QB)**2+AQQ) + OD/SQRT((R+TD*QA+TD*QB)**2+AQQ) &
          + OD/SQRT((R-TD*QA-TD*QB)**2+AQQ) + OD/SQRT((R-TD*QA+TD*QB)**2+AQQ) &
          - TD/SQRT((R-TD*QA)**2+AQQ)-TD/SQRT((R+TD*QA)**2+AQQ) &
          - TD/SQRT((R-TD*QB)**2+AQQ)-TD/SQRT((R+TD*QB)**2+AQQ) &
          + FD/SQRT(R**2+AQQ)

     DXQXZ=-TD/SQRT((R-QB)**2+(DA-QB)**2+ADQ) + TD/SQRT((R+QB)**2+(DA-QB)**2+ADQ) &
          + TD/SQRT((R-QB)**2+(DA+QB)**2+ADQ) - TD/SQRT((R+QB)**2+(DA+QB)**2+ADQ)

     QXZDX=-TD/SQRT((R+QA)**2+(QA-DB)**2+AQD) + TD/SQRT((R-QA)**2+(QA-DB)**2+AQD) &
          + TD/SQRT((R+QA)**2+(QA+DB)**2+AQD) - TD/SQRT((R-QA)**2+(QA+DB)**2+AQD)

     QXYQXY=FD/SQRT(R**2+TD*(QA-QB)**2+AQQ) + FD/SQRT(R**2+TD*(QA+QB)**2+AQQ) &
          - EIGHT/SQRT(R**2+TD*(QA**2+QB**2)+AQQ)

     QXZQXZ=TD/SQRT((R+QA-QB)**2+(QA-QB)**2+AQQ) - TD/SQRT((R+QA+QB)**2+(QA-QB)**2+AQQ) &
          - TD/SQRT((R-QA-QB)**2+(QA-QB)**2+AQQ) + TD/SQRT((R-QA+QB)**2+(QA-QB)**2+AQQ) &
          - TD/SQRT((R+QA-QB)**2+(QA+QB)**2+AQQ) + TD/SQRT((R+QA+QB)**2+(QA+QB)**2+AQQ) &
          + TD/SQRT((R-QA-QB)**2+(QA+QB)**2+AQQ) - TD/SQRT((R-QA+QB)**2+(QA+QB)**2+AQQ)
     !
     DXDX   =RP2*DXDX
     DZDZ   =RP2*DZDZ
     DZQXX  =RP3*DZQXX
     QXXDZ  =RP3*QXXDZ
     DZQZZ  =RP3*DZQZZ
     QZZDZ  =RP3*QZZDZ
     DXQXZ  =RP3*DXQXZ
     QXZDX  =RP3*QXZDX
     QXXQXX =RP4*QXXQXX
     QXXQYY =RP4*QXXQYY
     QXXQZZ =RP4*QXXQZZ
     QZZQXX =RP4*QZZQXX
     QZZQZZ =RP4*QZZQZZ
     QXZQXZ =RP4*QXZQXZ
     QXYQXY =RP4*QXYQXY
     ! 
     RI(6) = DZDZ
     RI(7) = DXDX
     RI(8) =-EDZ-QZZDZ
     RI(9) =-EDZ-QXXDZ
     RI(10)=-QXZDX
     RI(13)=-DZE-DZQZZ
     RI(14)=-DZE-DZQXX
     RI(15)=-DXQXZ
     RI(16)= EE+EQZZ+QZZE+QZZQZZ
     RI(17)= EE+EQZZ+QXXE+QXXQZZ
     RI(18)= EE+EQXX+QZZE+QZZQXX
     RI(19)= EE+EQXX+QXXE+QXXQXX
     RI(20)= QXZQXZ
     RI(21)= EE+EQXX+QXXE+QXXQYY
     RI(22)= HALF*(QXXQXX-QXXQYY)
     bdd(1,1)=bdd(1,2)

  end if        ! (NATORB(NI).ge.3)

40 CONTINUE
  !
  !     CONVERT INTO EV.
  !
  RI(2:22)=RI(2:22)*AU_TO_EV      ! 27.21D00
  !
  !     CALCULATE CORE-ELECTRON ATTRACTIONS.
  !
  CORE_local(2,1)=CORE(NJ)*RI(2)
  CORE_local(3,1)=CORE(NJ)*RI(3)
  CORE_local(4,1)=CORE(NJ)*RI(4)
  CORE_local(2,2)=CORE(NI)*RI(5)
  CORE_local(3,2)=CORE(NI)*RI(11)
  CORE_local(4,2)=CORE(NI)*RI(12)
  RETURN
  !
END SUBROUTINE MNREPP

SUBROUTINE MNROTAT(COORD,I,J,IX,RIJ,DEL1,IDX)
  !
  use chm_kinds
  use number
  use quantm, only : DS_ant,DG_ant,DR_ant, &
       TDX_ant,TDY_ant,TDZ_ant,  &
       G_ant, TX_ant, TY_ant, TZ_ant
  implicit none
  INTEGER I,J,IX,IDX
  real(chm_real) RIJ,DEL1
  !
  real(chm_real) COORD(3,25)
  !
  INTEGER IJK
  real(chm_real):: XD,YD,ZD,TERM,RXY,RZX,RYZ,r_RIJ,temp1,temp2,temp3
  !
  XD=COORD(1,I)-COORD(1,J)
  YD=COORD(2,I)-COORD(2,J)
  ZD=COORD(3,I)-COORD(3,J)
  RXY=SQRT(XD*XD+YD*YD)
  RYZ=SQRT(YD*YD+ZD*ZD)
  RZX=SQRT(ZD*ZD+XD*XD)
  r_RIJ=one/RIJ
  TX_ant(1:3) =ZERO
  TY_ant(1:3) =ZERO
  TZ_ant(1:3) =ZERO
  TDX_ant(1:3)=ZERO
  TDY_ant(1:3)=ZERO
  TDZ_ant(1:3)=ZERO
  IF(RXY.LT.1.0D-4) THEN
     !   MOLECULAR Z AXIS IS PARALLEL TO DIATOMIC Z AXIS
     if(ZD.LT.ZERO) then
        TX_ant(3)=-ONE
     else
        TX_ant(3)=ONE
     end if
     TY_ant(2)=ONE
     TZ_ant(1)=TX_ant(3)
     IF(IDX.EQ.1) RETURN
     if(IX.EQ.1) then
        TDX_ant(1)= r_RIJ     ! ONE/RIJ
        TDZ_ant(3)=-r_RIJ     ! ONE/RIJ
     else if(IX.EQ.2) then
        TDX_ant(2)= r_RIJ     ! ONE/RIJ
        TDY_ant(3)=-TX_ant(3)*r_RIJ
     end if
  ELSEIF(RYZ.LT.1.0D-4) THEN
     !   MOLECULAR X AXIS IS PARALLEL TO DIATOMIC Z AXIS
     if(XD.LT.ZERO) then
        TX_ant(1)=-ONE
     else
        TX_ant(1)=ONE
     end if
     TY_ant(2)=TX_ant(1)
     TZ_ant(3)=ONE
     IF(IDX.EQ.1) RETURN
     if(IX.EQ.2) then
        TDX_ant(2)= r_RIJ     ! ONE/RIJ
        TDY_ant(1)=-r_RIJ
     else if(IX.EQ.3) then
        TDX_ant(3)= r_RIJ
        TDZ_ant(1)=-TX_ant(1)*r_RIJ   ! /RIJ
     end if
  ELSEIF(RZX.LT.1.0D-4) THEN
     !   MOLECULAR Y AXIS IS PARALLEL TO DIATOMIC Z AXIS
     if(YD.LT.ZERO) then
        TX_ant(2)=-ONE
     else
        TX_ant(2)=ONE
     end if
     TY_ant(1)=-TX_ant(2)
     TZ_ant(3)=ONE
     IF(IDX.EQ.1) RETURN
     if(IX.EQ.1) then
        TDX_ant(1)= r_RIJ     ! ONE/RIJ
        TDY_ant(2)= r_RIJ
     else if(IX.EQ.3) then
        TDX_ant(3)= r_RIJ
        TDZ_ant(2)=-TX_ant(2)*r_RIJ   ! /RIJ
     end if
  ELSE
     TX_ant(1)=XD*r_RIJ   ! /RIJ
     TX_ant(2)=YD*r_RIJ 
     TX_ant(3)=ZD*r_RIJ 
     TZ_ant(3)=RXY*r_RIJ 
     temp1    =one/TZ_ant(3)
     temp2    =one/TZ_ant(3)**2
     TY_ant(1)=-TX_ant(2)*SIGN(+ONE,TX_ant(1))*temp1      ! /TZ_ant(3)
     TY_ant(2)= ABS(TX_ant(1)*temp1)
     TY_ant(3)= ZERO
     TZ_ant(1)=-TX_ant(1)*TX_ant(3)*temp1  ! /TZ_ant(3)
     TZ_ant(2)=-TX_ant(2)*TX_ant(3)*temp1  ! /TZ_ant(3)
     IF(IDX.EQ.1) RETURN
     TERM=DEL1/(RIJ*RIJ)
     IF(IX.EQ.1)THEN
        TDX_ant(1)= r_RIJ-TX_ant(1)*TERM    ! r_RIJ
        TDX_ant(2)=-TX_ant(2)*TERM
        TDX_ant(3)=-TX_ant(3)*TERM
        TDZ_ant(3)= TX_ant(1)/RXY-TZ_ant(3)*TERM
     ELSEIF(IX.EQ.2) THEN
        TDX_ant(1)=-TX_ant(1)*TERM
        TDX_ant(2)= r_RIJ-TX_ant(2)*TERM
        TDX_ant(3)=-TX_ant(3)*TERM
        TDZ_ant(3)= TX_ant(2)/RXY-TZ_ant(3)*TERM
     ELSEIF(IX.EQ.3)THEN
        TDX_ant(1)=-TX_ant(1)*TERM
        TDX_ant(2)=-TX_ant(2)*TERM
        TDX_ant(3)= r_RIJ-TX_ant(3)*TERM
        TDZ_ant(3)=-TZ_ant(3)*TERM
     ENDIF
     TDY_ant(1)=-TDX_ant(2)*temp1+TX_ant(2)*TDZ_ant(3)*temp2     ! /TZ_ant(3) .and. /TZ_ant(3)**2
     TDY_ant(2)=TDX_ant(1)*temp1-TX_ant(1)*TDZ_ant(3)*temp2
     if(TX_ant(1).LT.ZERO) then
        TDY_ant(1)=-TDY_ant(1)
        TDY_ant(2)=-TDY_ant(2)
     end if
     TDY_ant(3)=ZERO
     TDZ_ant(1)=-TX_ant(3)*TDX_ant(1)*temp1-TX_ant(1)*TDX_ant(3)*temp1+TX_ant(1)*TX_ant(3)*TDZ_ant(3)*temp2
     TDZ_ant(2)=-TX_ant(3)*TDX_ant(2)*temp1-TX_ant(2)*TDX_ant(3)*temp1+TX_ant(2)*TX_ant(3)*TDZ_ant(3)*temp2
  ENDIF
  RETURN
END SUBROUTINE MNROTAT

SUBROUTINE MNROTATE (NI,NJ,XI,XJ,W,KR,E1B,E2A,ENUC,CUTOFF, &
     ISTORE,IREAD)
  !
  use am1parm,only: alp=>alfa,NATORB
  use chm_kinds
  use number
  use quantm, only : F03_anal,KEYWRD, CSS1,CSP1,CPPS1,CPPP1,CSS2,CSP2,CPPS2,CPPP2, &
       X_rotdm,Y_rotdm,Z_rotdm
  use am1parm, only : CORE,FN1,FN2,FN3
  implicit none
  INTEGER NI,NJ,KR,IREAD
  real(chm_real)  ENUC,CUTOFF
  real(chm_real) XI(3),XJ(3),W(100),E1B(10),E2A(10)
  !***********************************************************************
  !
  !   ROTATE CALCULATES THE TWO-PARTICLE INTERACTIONS.
  !
  !   ON INPUT  NI     = ATOMIC NUMBER OF FIRST ATOM.
  !             NJ     = ATOMIC NUMBER OF SECOND ATOM.
  !             XI     = COORDINATE OF FIRST ATOM.
  !             XJ     = COORDINATE OF SECOND ATOM.
  !
  ! ON OUTPUT W      = ARRAY OF TWO-ELECTRON REPULSION INTEGRALS.
  !           E1B,E2A= ARRAY OF ELECTRON-NUCLEAR ATTRACTION INTEGRALS,
  !                    E1B = ELECTRON ON ATOM NI ATTRACTING NUCLEUS OF NJ.
  !           ENUC   = NUCLEAR-NUCLEAR REPULSION TERM.
  !
  !***********************************************************************
  real(chm_real) X(3),Y(3),Z(3),RI(22),CORE_local(4,2)
  real(chm_real) ISTORE(22,*)
  !
  INTEGER IB,JB,IG,II,IJ,JJ,KI,KK,LL,NT,I,J,K,L
  real(chm_real)  A,GAM,RIJ,SCALE
  !
  LOGICAL SI,SK, AM1, ANALYT
  !
  ! *** THIS ROUTINE COMPUTES THE REPULSION AND NUCLEAR ATTRACTION
  !     INTEGRALS OVER MOLECULAR-FRAME COORDINATES.  THE INTEGRALS OVER
  !     LOCAL FRAME COORDINATES ARE EVALUATED BY SUBROUTINE REPP AND
  !     STORED AS FOLLOWS (WHERE P-SIGMA = O,   AND P-PI = P AND P* )
  !     IN RI
  !     (SS/SS)=1,   (SO/SS)=2,   (OO/SS)=3,   (PP/SS)=4,   (SS/OS)=5,
  !     (SO/SO)=6,   (SP/SP)=7,   (OO/SO)=8,   (PP/SO)=9,   (PO/SP)=10,
  !     (SS/OO)=11,  (SS/PP)=12,  (SO/OO)=13,  (SO/PP)=14,  (SP/OP)=15,
  !     (OO/OO)=16,  (PP/OO)=17,  (OO/PP)=18,  (PP/PP)=19,  (PO/PO)=20,
  !     (PP/P*P*)=21,   (P*P/P*P)=22.
  !
  X(1:3)=XI(1:3)-XJ(1:3)
  RIJ=X(1)*X(1)+X(2)*X(2)+X(3)*X(3)
  IF(RIJ.LT.1.D-10) THEN
     E1B(1:10)=ZERO
     E2A(1:10)=ZERO
     W(KR)=ZERO
     ENUC=ZERO
     RETURN
  ENDIF
  ANALYT=(INDEX(KEYWRD,'ANALYT') .NE. 0)
  AM1= (INDEX(KEYWRD,'AM1') .NE. 0) .OR. (INDEX(KEYWRD,'PM3') .NE. 0)
  RIJ=MIN(SQRT(RIJ),CUTOFF)
  CALL MNREPP(NI,NJ,RIJ,RI,CORE_local)

  ! copy CORE_local to CSS1,CSP1,CPPS1,CPPP1,CSS2,CSP2,CPPS2,CPPP2
  CSS1 =CORE_local(1,1)
  CSP1 =CORE_local(2,1)
  CPPS1=CORE_local(3,1)
  CPPP1=CORE_local(4,1)
  CSS2 =CORE_local(1,2)
  CSP2 =CORE_local(2,2)
  CPPS2=CORE_local(3,2)
  CPPP2=CORE_local(4,2)

  IF (ANALYT) ISTORE(1:22,IREAD) = RI(1:22)
  GAM=RI(1)
  !
  ! *** THE REPULSION INTEGRALS OVER MOLECULAR FRAME (W) ARE STORED IN THE
  !     ORDER IN WHICH THEY WILL LATER BE USED.  IE.  (I,J/K,L) WHERE
  !     J.LE.I  AND  L.LE.K     AND L VARIES MOST RAPIDLY AND I LEAST
  !     RAPIDLY.  (ANTI-NORMAL COMPUTER STORAGE)
  !
  A=ONE/RIJ
  X(1:3)=X(1:3)*A
  Z(3)=ZERO
  IF(ABS(X(3)).GT.0.99999999D0) THEN
     X(3)=SIGN(ONE,X(3))
     Y(1)=ZERO
     Y(2)=ONE
     Y(3)=ZERO
     Z(1)=ONE
     Z(2)=ZERO
  Else
     Z(3)=SQRT(ONE-X(3)**2)
     A=ONE/Z(3)
     Y(1)=-A*X(2)*SIGN(ONE,X(1))
     Y(2)=ABS(A*X(1))
     Y(3)=ZERO
     Z(1)=-A*X(1)*X(3)
     Z(2)=-A*X(2)*X(3)
  End if
  IB=MIN(NATORB(NI),4)
  JB=MIN(NATORB(NJ),4)
  KI=0
  lp270: DO I=1,IB
     SI=I.EQ.1
     II=I-1
     lp270a: DO J=1,I
        JJ=J-1
        IJ=0
        IF (JJ.EQ.0) IJ=-1
        IF (SI) IJ=+1
        lp270b: DO K=1,JB
           KK=K-1
           SK=KK.GT.0
           lp270c: DO L=1,K
              KI=KI+1
              if(.not.SK) then
                 !
                 ! *** INTEGRAL (I,J/K,L) IS OF THE TYPE (I,J/S,S)
                 !
                 if(IJ.gt.0) then        ! (SS/SS)
                    W(KI)=RI(1)
                    cycle lp270c
                 else if(IJ.lt.0) then   ! (PS/SS)
                    W(KI)=RI(2)*X(II)
                    cycle lp270c
                 else if(IJ.eq.0) then   ! (PP/SS)
                    W(KI)=RI(3)*X(II)*X(JJ)+RI(4)*(Y(II)*Y(JJ)+Z(II)*Z(JJ))
                    cycle lp270c
                 end if
              else
                 LL=L-1
                 if(LL.le.0) then
                    !
                    ! *** INTEGRAL (I,J/K,L) IS OF THE TYPE (I,J/P,S)
                    !
                    if(IJ.gt.0) then          ! (SS/PS)
                       W(KI)=RI(5)*X(KK)
                       cycle lp270c
                    else if(IJ.lt.0) then     ! (PS/PS)
                       W(KI)=RI(6)*X(II)*X(KK)+RI(7)*(Y(II)*Y(KK)+Z(II)*Z(KK))
                       cycle lp270c
                    else if(IJ.eq.0) then     ! (PP/PS)
                       W(KI)=X(KK)*(RI(8)*X(II)*X(JJ)+RI(9)*(Y(II)*Y(JJ)+Z(II)*Z(JJ))) &
                            +RI(10)*(X(II)*(Y(JJ)*Y(KK)+Z(JJ)*Z(KK))+X(JJ)*(Y(II)*Y(KK)+Z(II)*Z(KK)))
                       cycle lp270c
                    end if
                 else
                    !
                    ! *** INTEGRAL (I,J/K,L) IS OF THE TYPE (I,J/P,P)
                    !
                    if(ij.gt.0) then            ! (SS/PP)
                       W(KI)=RI(11)*X(KK)*X(LL)+RI(12)*(Y(KK)*Y(LL)+Z(KK)*Z(LL))
                       cycle lp270c
                    else if(ij.lt.0) then       ! (PS/PP)
                       W(KI)=X(II)*(RI(13)*X(KK)*X(LL)+ RI(14)*(Y(KK)*Y(LL)+Z(KK)*Z(LL))) &
                            +RI(15)*(Y(II)*(Y(KK)*X(LL)+Y(LL)*X(KK))+Z(II)*(Z(KK)*X(LL)+Z(LL)*X(KK)))
                       cycle lp270c
                    else if(ij.eq.0) then       ! (PP/PP)
                       W(KI)=(RI(16)*X(II)*X(JJ)+RI(17)*(Y(II)*Y(JJ)+Z(II)*Z(JJ)))*X(KK)*X(LL) &
                            +RI(18)*X(II)*X(JJ)*(Y(KK)*Y(LL)+Z(KK)*Z(LL)) &
                            +RI(19)*(Y(II)*Y(JJ)*Y(KK)*Y(LL)+Z(II)*Z(JJ)*Z(KK)*Z(LL)) &
                            +RI(20)*( X(II)*(X(KK)*(Y(JJ)*Y(LL)+Z(JJ)*Z(LL))+X(LL)*(Y(JJ)*Y(KK)+Z(JJ)*Z(KK))) &
                            +X(JJ)*(X(KK)*(Y(II)*Y(LL)+Z(II)*Z(LL))+X(LL)*(Y(II)*Y(KK)+Z(II)*Z(KK))) ) &
                            +RI(21)*(Y(II)*Y(JJ)*Z(KK)*Z(LL)+Z(II)*Z(JJ)*Y(KK)*Y(LL)) &
                            +RI(22)*(Y(II)*Z(JJ)+Z(II)*Y(JJ))*(Y(KK)*Z(LL)+Z(KK)*Y(LL))
                       cycle lp270c
                    end if
                 end if     ! LL.le.0
              end if        !  .not.SK
           enddo lp270c
        enddo lp270b
     enddo lp270a
  enddo lp270
  !
  !
  ! copying X(1:3) to X1,X2,X3, and same for Y(1:3) and Z(1:3)
  X_rotdm(1:3)=X(1:3)
  Y_rotdm(1:3)=Y(1:3)
  Z_rotdm(1:3)=Z(1:3)
  E1B(1)=-CSS1
  IF(NI.GT.1) THEN
     E1B(2) = -CSP1 *X_rotdm(1)
     E1B(3) = -CPPS1*X_rotdm(1)**2-CPPP1*(Y_rotdm(1)**2+Z_rotdm(1)**2)
     E1B(4) = -CSP1 *X_rotdm(2)
     E1B(5) = -CPPS1*X_rotdm(1)*X_rotdm(2)-CPPP1*(Y_rotdm(1)*Y_rotdm(2)+Z_rotdm(1)*Z_rotdm(2))
     E1B(6) = -CPPS1*X_rotdm(2)*X_rotdm(2)-CPPP1*(Y_rotdm(2)*Y_rotdm(2)+Z_rotdm(2)*Z_rotdm(2))
     E1B(7) = -CSP1 *X_rotdm(3)
     E1B(8) = -CPPS1*X_rotdm(1)*X_rotdm(3)-CPPP1*(Y_rotdm(1)*Y_rotdm(3)+Z_rotdm(1)*Z_rotdm(3))
     E1B(9) = -CPPS1*X_rotdm(2)*X_rotdm(3)-CPPP1*(Y_rotdm(2)*Y_rotdm(3)+Z_rotdm(2)*Z_rotdm(3))
     E1B(10)= -CPPS1*X_rotdm(3)*X_rotdm(3)-CPPP1*(Y_rotdm(3)*Y_rotdm(3)+Z_rotdm(3)*Z_rotdm(3))
  ENDIF
  E2A(1)=-CSS2
  IF(NJ.GT.1) THEN
     E2A(2) = -CSP2 *X_rotdm(1)
     E2A(3) = -CPPS2*X_rotdm(1)**2-CPPP2*(Y_rotdm(1)**2+Z_rotdm(1)**2)
     E2A(4) = -CSP2 *X_rotdm(2)
     E2A(5) = -CPPS2*X_rotdm(1)*X_rotdm(2)-CPPP2*(Y_rotdm(1)*Y_rotdm(2)+Z_rotdm(1)*Z_rotdm(2))
     E2A(6) = -CPPS2*X_rotdm(2)*X_rotdm(2)-CPPP2*(Y_rotdm(2)*Y_rotdm(2)+Z_rotdm(2)*Z_rotdm(2))
     E2A(7) = -CSP2 *X_rotdm(3)
     E2A(8) = -CPPS2*X_rotdm(1)*X_rotdm(3)-CPPP2*(Y_rotdm(1)*Y_rotdm(3)+Z_rotdm(1)*Z_rotdm(3))
     E2A(9) = -CPPS2*X_rotdm(2)*X_rotdm(3)-CPPP2*(Y_rotdm(2)*Y_rotdm(3)+Z_rotdm(2)*Z_rotdm(3))
     E2A(10)= -CPPS2*X_rotdm(3)*X_rotdm(3)-CPPP2*(Y_rotdm(3)*Y_rotdm(3)+Z_rotdm(3)*Z_rotdm(3))
  ENDIF
  IF(ABS(CORE(NI)).GT.20.AND.ABS(CORE(NJ)).GT.20) THEN
     ! SPARKLE-SPARKLE INTERACTION
     ENUC=ZERO
     RETURN
  ELSEIF (RIJ.LT.ONE.AND.NATORB(NI)*NATORB(NJ).EQ.0) THEN
     ENUC=ZERO
     RETURN
  ENDIF
  SCALE = EXP(-ALP(NI)*RIJ)+EXP(-ALP(NJ)*RIJ)
  NT=NI+NJ
  IF(NT.EQ.7.OR.NT.EQ.8.OR.NT.EQ.9) THEN
     IF (NI.EQ.7.OR.NI.EQ.8 .AND. NJ.LT.2) SCALE = SCALE + (RIJ-ONE)*EXP(-ALP(NI)*RIJ)
     IF (NJ.EQ.7.OR.NJ.EQ.8 .AND. NI.LT.2) SCALE = SCALE + (RIJ-ONE)*EXP(-ALP(NJ)*RIJ)
  ENDIF
  ENUC = CORE(NI)*CORE(NJ)*GAM
  SCALE= ABS(SCALE*ENUC)
  IF( AM1 )THEN
     do IG=1,10
        IF(ABS(FN1(NI,IG)).GT.ZERO) SCALE=SCALE +CORE(NI)*CORE(NJ)/RIJ*FN1(NI,IG)* &
             EXP(MAX(-THIRTY,-FN2(NI,IG)*(RIJ-FN3(NI,IG))**2))
        IF(ABS(FN1(NJ,IG)).GT.ZERO) SCALE=SCALE +CORE(NI)*CORE(NJ)/RIJ*FN1(NJ,IG)* &
             EXP(MAX(-THIRTY,-FN2(NJ,IG)*(RIJ-FN3(NJ,IG))**2))
     end do
  ENDIF
  ENUC=ENUC+SCALE
  !
  ! *** NOW ROTATE THE NUCLEAR ATTRACTION INTEGRALS.
  ! *** THE STORAGE OF THE NUCLEAR ATTRACTION INTEGRALS  CORE_local(KL/IJ) IS
  !     (SS/)=1,   (SO/)=2,   (OO/)=3,   (PP/)=4
  !
  !   DEBUG PRINTING
  !
  KR=KR+KI
  RETURN
END SUBROUTINE MNROTATE

SUBROUTINE RSPQM(A,N,MATZ,W,Z)
  !
  use chm_kinds
  use number
  use sizes
  implicit none
  INTEGER N,MATZ
  real(chm_real) A(*),  W(N), Z(N,N)
  !*******************************************************************
  !*
  !*   EISPACK DIAGONALIZATION ROUTINES: TO FIND THE EIGENVALUES AND
  !*           EIGENVECTORS (IF DESIRED) OF A REAL SYMMETRIC PACKED MATRIX.
  !* ON INPUT-      N  IS THE ORDER OF THE MATRIX  A,
  !*                A  CONTAINS THE LOWER TRIANGLE OF THE REAL SYMMETRIC
  !*                   PACKED MATRIX STORED ROW-WISE,
  !*             MATZ  IS AN INTEGER VARIABLE SET EQUAL TO ZERO IF ONLY
  !*                   EIGENVALUES ARE DESIRED,  OTHERWISE IT IS SET TO
  !*                   ANY NON-ZERO INTEGER FOR BOTH EIGENVALUES AND
  !*                   EIGENVECTORS.
  !* ON OUTPUT-     W  CONTAINS THE EIGENVALUES IN ASCENDING ORDER,
  !*                Z  CONTAINS THE EIGENVECTORS IF MATZ IS NOT ZERO,
  !*
  !*******************************************************************
  !* THIS SUBROUTINE WAS CHOSEN AS BEING THE MOST RELIABLE. (JJPS)
  !     QUESTIONS AND COMMENTS SHOULD BE DIRECTED TO B. S. GARBOW,
  !     APPLIED MATHEMATICS DIVISION, ARGONNE NATIONAL LABORATORY
  !
  !     ------------------------------------------------------------------
  !
  real(chm_real) FV1(MAXORB),FV2(MAXORB)
  INTEGER NM,NV,IERR,I,J
  !
  logical, save        :: FIRST=.TRUE.
  real(chm_real), save :: eta,eps
  !
  IF (FIRST) THEN
     FIRST=.FALSE.
     CALL EPSETA(EPS,ETA)
  ENDIF
  NV=(MAXORB*(MAXORB+1))/2
  NM=N
  if(NV .lt. (N*(N+1))/2) then
     IERR = 20 * N
  else
     CALL TRED3QM(N,NV,A,W,FV1,FV2)
     if(MATZ .eq. 0) then
        !     ********** FIND EIGENVALUES ONLY **********
        CALL  TQLRAT(N,W,FV2,IERR,EPS)
     else
        !     ********** FIND BOTH EIGENVALUES AND EIGENVECTORS **********
        do I = 1, N
           Z(1:N,I)=ZERO
           Z(I,I)  =ONE
        end do
        CALL MNTQL2(NM,N,W,FV1,Z,IERR,EPS)
        if(ierr.eq.0) CALL  TRBAK3QM(NM,N,NV,A,N,Z)
     end if
  end if
  !     ********** LAST CARD OF RSP **********
  RETURN
END SUBROUTINE RSPQM

SUBROUTINE SCHMIB(U,N,NDIM)
  !
  use chm_kinds
  use number
  use sizes
  implicit none
  !
  !     SAME AS SCHMIDT BUT WORKS FROM RIGHT TO LEFT.
  !
  INTEGER N,NDIM
  real(chm_real) U(NDIM,NDIM)
  !
  INTEGER NPASS,II,I,J,K,K1,N1
  real(chm_real)  DOT,SCALE
  !
  real(chm_real),parameter:: SMALL=0.01D0
  !
  N1=N+1
  II=0
  loopKK: do K=1,N
     K1=K-1
     !
     !     NORMALIZE KTH COLUMN VECTOR
     !
     DOT = ZERO
     DO I=1,N
        DOT=DOT+U(I,N1-K)*U(I,N1-K)
     END DO
     if(DOT.EQ.ZERO) then
        II=II+1              ! REPLACE LINEARLY DEPENDENT KTH 
        U(II,N1-K)=ONE       ! VECTOR BY A UNIT VECTOR.
     else
        SCALE=ONE/SQRT(DOT)
        U(1:N,N1-K)=SCALE*U(1:N,N1-K)
     end if
     loopout: Do                            ! outer while
        IF(K1.EQ.0) cycle loopKK   
        NPASS=0
        !
        !     PROJECT OUT K-1 PREVIOUS ORTHONORMAL VECTORS FROM KTH VECTOR
        !
        loopin: do                         ! inner while
           continue
           NPASS=NPASS+1
           DO J=1,K1
              DOT=ZERO
              DO I=1,N
                 DOT=DOT+U(I,N1-J)*U(I,N1-K)
              END DO
              U(1:N,N1-K)=U(1:N,N1-K)-DOT*U(1:N,N1-J)
           END DO
           !
           !     SECOND NORMALIZATION (AFTER PROJECTION)
           !     IF KTH VECTOR IS SMALL BUT NOT ZERO THEN NORMALIZE
           !     AND PROJECT AGAIN TO CONTROL ROUND-OFF ERRORS.
           !
           DOT=ZERO
           DO I=1,N
              DOT=DOT+U(I,N1-K)*U(I,N1-K)
           END DO
           if(DOT.EQ.ZERO) then
              II=II+1                 ! REPLACE LINEARLY DEPENDENT KTH
              U(II,N1-K)=ONE          ! VECTOR BY A UNIT VECTOR.
              EXIT     ! exit inner while
           else  
              if(DOT.LT.SMALL.AND.NPASS.GT.2) then
                 II=II+1              ! REPLACE LINEARLY DEPENDENT KTH
                 U(II,N1-K)=ONE       ! VECTOR BY A UNIT VECTOR.
                 EXIT  ! exit inner while
              else
                 SCALE=ONE/SQRT(DOT)
                 U(1:N,N1-K)=SCALE*U(1:N,N1-K)
                 if(DOT.LT.SMALL) then
                    continue       ! loop while (inner) 
                 else
                    cycle loopKK   ! exit for continuing do-loop
                 end if
              end if
           end if  ! DOT.EQ.ZERO
        end do loopin    ! inner while
     End do loopout      ! outer while
  end do  loopKK ! K=1,N
  !
  RETURN
END SUBROUTINE SCHMIB

SUBROUTINE SCHMIT(U,N,NDIM)
  !
  use chm_kinds
  use number
  use sizes
  implicit none
  !
  INTEGER N,NDIM
  real(chm_real) U(NDIM,NDIM)
  !
  INTEGER NPASS,II,I,J,K,K1,N1
  real(chm_real)  DOT,SCALE,UTMP
  !
  real(chm_real),parameter :: SMALL=0.01D0
  !
  II=0
  loopKK: do K=1,N
     K1=K-1
     !
     !     NORMALIZE KTH COLUMN VECTOR
     !
     DOT = ZERO
     DO I=1,N
        DOT=DOT+U(I,K)*U(I,K)
     END DO
     if(DOT.EQ.ZERO) then
        II=II+1                     ! REPLACE LINEARLY DEPENDENT KTH VECTOR
        U(II,K)=ONE                 ! BY A UNIT VECTOR.
     else
        SCALE=ONE/SQRT(DOT)
        U(1:N,K)=SCALE*U(1:N,K)
     end if
     loopout: Do
        IF(K1.EQ.0) cycle loopKK
        NPASS=0
        !
        !     PROJECT OUT K-1 PREVIOUS ORTHONORMAL VECTORS FROM KTH VECTOR
        !
        loopin: do
           NPASS=NPASS+1
           DO J=1,K1
              DOT=ZERO
              DO I=1,N
                 DOT=DOT+U(I,J)*U(I,K)
              END DO
              U(1:N,K)=U(1:N,K)-DOT*U(1:N,J)
           END DO
           !
           !     SECOND NORMALIZATION (AFTER PROJECTION)
           !     IF KTH VECTOR IS SMALL BUT NOT ZERO THEN NORMALIZE
           !     AND PROJECT AGAIN TO CONTROL ROUND-OFF ERRORS.
           !
           DOT=ZERO
           DO I=1,N
              DOT=DOT+U(I,K)*U(I,K)
           END DO
           if(DOT.eq.ZERO) then
              II=II+1                     ! REPLACE LINEARLY DEPENDENT KTH VECTOR
              U(II,K)=ONE                 ! BY A UNIT VECTOR.
              EXIT    ! exit inner-while
           else
              if(DOT.LT.SMALL.AND.NPASS.GT.2) then
                 II=II+1                     ! REPLACE LINEARLY DEPENDENT KTH VECTOR
                 U(II,K)=ONE                 ! BY A UNIT VECTOR.
                 EXIT    ! exit inner-while
              else
                 SCALE=ONE/SQRT(DOT)
                 U(1:N,K)= SCALE*U(1:N,K)
                 if(DOT.LT.SMALL) then
                    continue        ! loop while (inner)
                 else
                    cycle loopKK    ! exit for continuing do-loop
                 end if
              end if
           end if
        end do loopin      ! inner-while
     End do loopout        ! outer-while
  End do loopKK
  !
  RETURN
END SUBROUTINE SCHMIT

SUBROUTINE MNSET (S1,S2,NA,NB,RAB,NBOND,II)
  !
  use chm_kinds
  use number
  use quantm, only : SA_int,SB_int,ISP_int,IPS_int
  implicit none
  real(chm_real) S1,S2,RAB
  INTEGER NA,NB,NBOND,II
  !
  INTEGER J,JCALL
  real(chm_real)  BETA,ALPHA
  !
  !***********************************************************************
  !
  !     SET IS PART OF THE OVERLAP CALCULATION, CALLED BY OVERLP.
  !         IT CALLS AINTGS AND BINTGS
  !
  !***********************************************************************
  if(NA.GT.NB) then
     ISP_int=2
     IPS_int=1 
     SA_int=S2
     SB_int=S1
  else
     ISP_int=1
     IPS_int=2
     SA_int=S1
     SB_int=S2
  end if
  J=II+2
  IF (II.GT.3) J=J-1
  ALPHA=HALF*RAB*(SA_int+SB_int)
  BETA =HALF*RAB*(SB_int-SA_int)
  JCALL=J-1
  CALL MNAINTGS (ALPHA,JCALL)
  CALL MNBINTGS (BETA,JCALL)
  RETURN
  !
END SUBROUTINE MNSET

SUBROUTINE MNSETUPG
  !
  use chm_kinds
  use sizes
  use number,only : zero
  use stream
  use am1parm, only : ZS,ZP,ZD
  use quantm, only : MTYPE_anal,C_tempm,Z_tempm
  implicit none
  !
  real(chm_real), save:: ALLC(6,5,2)=zero, ALLZ(6,5,2)=zero
  !
  INTEGER IA,IB,NI,I,J,K,L,NQN
  real(chm_real)  XI
  !
  !     SET-UP THE STEWART'S STO-6G EXPANSIONS
  !                                            1S
  ALLZ(1,1,1) =2.310303149D01
  ALLZ(2,1,1) =4.235915534D00
  ALLZ(3,1,1) =1.185056519D00
  ALLZ(4,1,1) =4.070988982D-01
  ALLZ(5,1,1) =1.580884151D-01
  ALLZ(6,1,1) =6.510953954D-02
  !
  ALLC(1,1,1) =9.163596280D-03
  ALLC(2,1,1) =4.936149294D-02
  ALLC(3,1,1) =1.685383049D-01
  ALLC(4,1,1) =3.705627997D-01
  ALLC(5,1,1) =4.164915298D-01
  ALLC(6,1,1) =1.303340841D-01
  !                                      2S
  ALLZ(1,2,1) =2.768496241D01
  ALLZ(2,2,1) =5.077140627D00
  ALLZ(3,2,1) =1.426786050D00
  ALLZ(4,2,1) =2.040335729D-01
  ALLZ(5,2,1) =9.260298399D-02
  ALLZ(6,2,1) =4.416183978D-02
  !
  ALLC(1,2,1) =-4.151277819D-03
  ALLC(2,2,1) =-2.067024148D-02
  ALLC(3,2,1) =-5.150303337D-02
  ALLC(4,2,1) =3.346271174D-01
  ALLC(5,2,1) =5.621061301D-01
  ALLC(6,2,1) =1.712994697D-01
  !                                     2P
  ALLZ(1,2,2) =5.868285913D00
  ALLZ(2,2,2) =1.530329631D00
  ALLZ(3,2,2) =5.475665231D-01
  ALLZ(4,2,2) =2.288932733D-01
  ALLZ(5,2,2) =1.046655969D-01
  ALLZ(6,2,2) =4.948220127D-02
  !
  ALLC(1,2,2) =7.924233646D-03
  ALLC(2,2,2) =5.144104825D-02
  ALLC(3,2,2) =1.898400060D-01
  ALLC(4,2,2) =4.049863191D-01
  ALLC(5,2,2) =4.012362861D-01
  ALLC(6,2,2) =1.051855189D-01
  !                                      3S
  ALLZ(1,3,1) =3.273031938D00
  ALLZ(2,3,1) =9.200611311D-01
  ALLZ(3,3,1) =3.593349765D-01
  ALLZ(4,3,1) =8.636686991D-02
  ALLZ(5,3,1) =4.797373812D-02
  ALLZ(6,3,1) =2.724741144D-02
  ALLC(1,3,1) =-6.775596947D-03
  ALLC(2,3,1) =-5.639325779D-02
  ALLC(3,3,1) =-1.587856086D-01
  ALLC(4,3,1) =5.534527651D-01
  ALLC(5,3,1) =5.015351020D-01
  ALLC(6,3,1) =7.223633674D-02
  !                                     3P
  ALLZ(1,3,2) =5.077973607D00
  ALLZ(2,3,2) =1.340786940D00
  ALLZ(3,3,2) =2.248434849D-01
  ALLZ(4,3,2) =1.131741848D-01
  ALLZ(5,3,2) =6.076408893D-02
  ALLZ(6,3,2) =3.315424265D-02
  ALLC(1,3,2) =-3.329929840D-03
  ALLC(2,3,2) =-1.419488340D-02
  ALLC(3,3,2) =1.639395770D-01
  ALLC(4,3,2) =4.485358256D-01
  ALLC(5,3,2) =3.908813050D-01
  ALLC(6,3,2) =7.411456232D-02
  !                                     4S
  ALLZ(1,4,1) = 1.365346d+00
  ALLZ(2,4,1) = 4.393213d-01
  ALLZ(3,4,1) = 1.877069d-01
  ALLZ(4,4,1) = 9.360270d-02
  ALLZ(5,4,1) = 5.052263d-02
  ALLZ(6,4,1) = 2.809354d-02
  ALLC(1,4,1) = 3.775056d-03
  ALLC(2,4,1) =-5.585965d-02
  ALLC(3,4,1) =-3.192946d-01
  ALLC(4,4,1) =-2.764780d-02
  ALLC(5,4,1) = 9.049199d-01
  ALLC(6,4,1) = 3.406258d-01
  !                                   4P
  ALLC(1,4,2) =-7.052075d-03
  ALLC(2,4,2) =-5.259505d-02
  ALLC(3,4,2) =-3.773450d-02
  ALLC(4,4,2) = 3.874773d-01
  ALLC(5,4,2) = 5.791672d-01
  ALLC(6,4,2) = 1.221817d-01
  ALLZ(1,4,2) = 1.365346d+00
  ALLZ(2,4,2) = 4.393213d-01
  ALLZ(3,4,2) = 1.877069d-01
  ALLZ(4,4,2) = 9.360270d-02
  ALLZ(5,4,2) = 5.052263d-02
  ALLZ(6,4,2) = 2.809354d-02
  !                                     5S
  ALLZ(1,5,1) = 7.701420258D-01
  ALLZ(2,5,1) = 2.756268915D-01
  ALLZ(3,5,1) = 1.301847480D-01
  ALLZ(4,5,1) = 6.953441940D-02
  ALLZ(5,5,1) = 4.002545502D-02
  ALLZ(6,5,1) = 2.348388309D-02
  ALLC(1,5,1) = 1.267447151D-02
  ALLC(2,5,1) = 3.266734789D-03
  ALLC(3,5,1) =-4.307553999D-01
  ALLC(4,5,1) =-3.231998963D-01
  ALLC(5,5,1) = 1.104322879D+00
  ALLC(6,5,1) = 4.368498703D-01
  !                                      5P
  ALLZ(1,5,2) = 7.701420258D-01
  ALLZ(2,5,2) = 2.756268915D-01
  ALLZ(3,5,2) = 1.301847480D-01
  ALLZ(4,5,2) = 6.953441940D-02
  ALLZ(5,5,2) = 4.002545502D-02
  ALLZ(6,5,2) = 2.348388309D-02
  ALLC(1,5,2) =-1.105673292D-03
  ALLC(2,5,2) =-6.243132446D-02
  ALLC(3,5,2) =-1.628476766D-01
  ALLC(4,5,2) = 3.210328714D-01
  ALLC(5,5,2) = 6.964579592D-01
  ALLC(6,5,2) = 1.493146125D-01

  do I=1,10
     NI=MTYPE_anal(I)
     XI=ZS(NI)
     IA=I*4-3
     IB=IA+3
     IF(NI.LT.2) THEN
        NQN=1
     ELSEIF(NI.LT.10)THEN
        NQN=2
     ELSEIF(NI.LT.18)THEN
        NQN=3
     ELSEIF(NI.LT.36)THEN
        NQN=4
     ELSEIF(NI.LT.54)THEN
        NQN=5
        !
        !       SETUP FOR QM LINK ATOMS...JG 5/17/97
        !
     ELSEIF(NI.GT.90) THEN
        NQN=2
     ELSE
        CALL WRNDIE(-5,'<MNSETUPG>','No Gaussians avaiable.')
        return
     ENDIF
     do K=IA,IB
        L=1
        IF(K.GT.IA) L=2
        IF(K.GT.IA) XI=ZP(NI)
        C_tempm(K,1:6)=ALLC(1:6,NQN,L)
        Z_tempm(K,1:6)=ALLZ(1:6,NQN,L)*XI**2
     end do
  end do
  !
  RETURN
END SUBROUTINE MNSETUPG

SUBROUTINE SPLINE
  !
  use chm_kinds
  use number
  use sizes
  use quantm, only : NPNTS_fit,XLOW_fit,XHIGH_fit,XMIN_fit, &
       EMIN_fit,DEMIN_fit,X_fit,F_fit,DF_fit
  implicit none
  LOGICAL SKIP1,SKIP2
  !
  !     FIT F_fit(X) BY A CUBIC SPLINE GIVEN VALUES OF THE FUNCTION
  !     AND ITS FIRST DERIVATIVE AT N PNTS.
  !     SUBROUTINE RETURNS VALUES OF XMIN_fit,FMIN, AND DFMIN
  !     AND MAY REORDER THE DATA.
  !     CALLING PROGRAM SUPPLIES ALL OTHER VALUES IN THE
  !     COMMON BLOCK.
  !     XLOW_fit AND XHIGH_fit SET LIMITS ON THE INTERVAL WITHIN WHICH
  !     TO SEARCH.  SUBROUTINE MAY FURTHER REDUCE THIS INTERVAL.
  !
  INTEGER K,N1
  real(chm_real)  BB,FM,XSTOP,DX,XK,AC3,STEP,A,B,C,R,XSTART,DUM,X1,X2,temp1
  !
  real(chm_real) :: CLOSE=1.0D-8, BIG=500.0D0, HUGE=1.0D+10, &
       USTEP=1.0D0, DSTEP=2.0D0
  !
  !     SUBROUTINE ASSUMES THAT THE FIRST N-1 DATA PNTS HAVE BEEN
  !     PREVIOUSLY ORDERED,  X_fit(I).LT.X_fit(I+1) FOR I=1,2,...,N-2
  !     NOW MOVE NTH POINT TO ITS PROPER PLACE.
  !
  XMIN_fit =X_fit(NPNTS_fit)
  EMIN_fit =F_fit(NPNTS_fit)
  DEMIN_fit=DF_fit(NPNTS_fit)
  N1       =NPNTS_fit-1
  K        =N1
  do
     IF(X_fit(K).LT.XMIN_fit) EXIT
     X_fit(K+1) =X_fit(K)
     F_fit(K+1) =F_fit(K)
     DF_fit(K+1)=DF_fit(K)
     K          =K-1
     if(K.le.0) EXIT
  end do

  X_fit(K+1) =XMIN_fit
  F_fit(K+1) =EMIN_fit
  DF_fit(K+1)=DEMIN_fit
  !
  !     DEFINE THE INTERVAL WITHIN WHICH WE TRUST THE SPLINE FIT.
  !     USTEP =  UP HILL STEP SIZE FACTOR
  !     DSTEP = DOWN HILL STEP SIZE FACTOR
  !
  if(DF_fit(1).GT.0.0D0) then
     STEP=DSTEP
  else                ! if(DF_fit(1).LE.0.0D0) then
     STEP=USTEP
  end if
  XSTART=X_fit(1)-STEP*(X_fit(2)-X_fit(1))
  XSTART=MAX(XSTART,XLOW_fit)
  if(DF_fit(NPNTS_fit).GT.0.0D0) then
     STEP=USTEP
  else                ! if(DF_fit(NPNTS_fit).LE.0.0D0) then
     STEP=DSTEP
  end if
  XSTOP=X_fit(NPNTS_fit)+STEP*(X_fit(NPNTS_fit)-X_fit(N1))
  XSTOP=MIN(XSTOP,XHIGH_fit)
  !
  !     SEARCH FOR MINIMUM
  !
  loopKK: do K=1,N1
     SKIP1=K.NE.1
     SKIP2=K.NE.N1
     if(F_fit(K).lt.EMIN_fit) then
        XMIN_fit =X_fit(K)
        EMIN_fit =F_fit(K)
        DEMIN_fit=DF_fit(K)
     end if
     DX=X_fit(K+1)-X_fit(K)
     !
     !     SKIP INTERVAL IF PNTS ARE TOO CLOSE TOGETHER
     !
     IF(DX.LE.CLOSE) cycle loopKK  ! GO TO 110
     X1=zero
     X2=DX
     IF(K.EQ.1 ) X1=XSTART-X_fit(1)
     IF(K.EQ.N1) X2=XSTOP -X_fit(N1)
     !
     !     (A,B,C)=COEF OF (CUBIC,QUADRATIC,LINEAR) TERMS
     !
     temp1=one/DX
     DUM=(F_fit(K+1)-F_fit(K))*temp1   ! /DX
     A  =(DF_fit(K)+DF_fit(K+1)-DUM-DUM)*temp1*temp1  ! /(DX*DX)
     B  =(DUM+DUM+DUM-DF_fit(K)-DF_fit(K)-DF_fit(K+1))*temp1  !/DX
     C  =DF_fit(K)
     !
     !     XK = X-X_fit(K) AT THE MINIMUM WITHIN THE KTH SUBINTERVAL
     !     TEST FOR PATHOLOGICAL CASES.
     !
     BB = B*B
     AC3=(A+A+A)*C
     IF(BB.LT.AC3) GO TO 90
     if(B.gt.0.0D0) then
        if(BB.GT.BIG*ABS(AC3)) then
           R=AC3/BB                      ! CUBIC IS DOMINATED BY QUADRATIC TERM
           XK=-(((0.039063D0*R+0.0625D0)*R+0.125D0)*R+HALF)*C/B
        else   
           XK=(-B+SQRT(BB-AC3))/(A+A+A)  ! WELL BEHAVED CUBIC
        end if
     else
        if(ABS(B).GT.HUGE*ABS(A)) then
           GO TO 90
        else
           XK=(-B+SQRT(BB-AC3))/(A+A+A)     ! WELL BEHAVED CUBIC
        end if
     end if
     IF(XK.LT.X1.OR.XK.GT.X2) GO TO 90
80   FM=((A*XK+B)*XK+C)*XK+F_fit(K)
     if(FM.le.EMIN_fit) then
        XMIN_fit  =XK+X_fit(K)
        EMIN_fit  =FM
        DEMIN_fit =((A+A+A)*XK+B+B)*XK+C
     end if
     !
     !     EXTRAPOLATE TO END OF INTERVAL IF K=1 AND/OR K=N1
     !
90   if(.not.SKIP1) then
        SKIP1=.TRUE.
        XK=X1
        GO TO 80
     end if
     if(SKIP2) then
        cycle loopKK 
     else
        SKIP2=.TRUE.
        XK=X2
        GO TO 80
     end if
  End do loopKK

  RETURN
END SUBROUTINE SPLINE

SUBROUTINE SQUAR2(A,N,B)
  !
  use chm_kinds
  use sizes
  implicit none
  INTEGER N
  real(chm_real) A(N,N), B(N*N)
  !
  INTEGER I,J,L
  !
  !************************************************************************
  !*
  !*   SQUARE TAKES A PACKED LOWER-TRIANGULAR MATRIX IN ARRAY A
  !*   AND RE-ARRANGES IT INTO A PACKED SQUARE MATRIX IN THE
  !*   SAME SPACE.  A AND B SHOULD BE THE SAME IN THE ARE EQUIVALENCED.
  !*
  !************************************************************************
  !
  !  FIRST FILL ONE HALF OF THE SQUARE MATRIX.
  !
  L=(N*(N+1))/2+1
  do I=N,1,-1
     do J=I,1,-1
        L=L-1
        A(J,I)=B(L)
     end do
  end do
  !
  !  NOW FILL OTHER HALF OF THE SQUARE MATRIX.  IF NOT NEEDED THEN DO
  !  A RETURN AT THIS POINT
  !
  do I=1,N
     do j=I+1,N
        A(J,I)=A(I,J)
     end do
  end do
END SUBROUTINE SQUAR2

SUBROUTINE SQUARE(A,N)
  !
  use chm_kinds
  implicit none
  INTEGER N
  real(chm_real) A(N,N)
  CALL SQUAR2(A,N,A)
  RETURN
END SUBROUTINE SQUARE

FUNCTION MNSS(NA,NB,LA,LB,M,UC,UD,R1,FIRST)
  !
  use chm_kinds
  use number
  use consta
  implicit none
  INTEGER NA,NB,LA,LB,M
  real(chm_real) UC,UD,R1,mnss
  LOGICAL FIRST
  INTEGER A,PP,B,Q
  real(chm_real) FA(14),BI(13,13),AF(20),BF(20)
  real(chm_real) :: AFF(3,3,3)=zero
  !
  INTEGER LAM1,LBM1,IA,IB,IC,ID,IX,JX,NANB,NANB1,I,J,N,I1,K1,K2,K3,K4,K5,K6
  real(chm_real)  AB,BA,SUM1,SA,UA,ER,UB,EX,P,R,X, &
       PART1,PART2,PART3,PART4,PART5,PART6,QUO,SUM
  !
  DATA FA/1.D0      ,1.D0       ,2.D0        ,6.D0         ,24.D0    , &
       120.D0    ,720.D0     ,5040.D0     ,40320.D0     ,362880.D0, &
       3628800.D0,39916800.D0,479001600.D0,6227020800.D0/
  R =R1
  UA=UC
  UB=UD
  R =R/BOHRR
  ER=R
  IF(FIRST) THEN
     FIRST=.FALSE.
     BI(1:13,1)=one
     do I=1,13
        BI(I,I)=ONE
     end do
     do I=1,12
        BI(I+1,2:I)=BI(I,2:I)+BI(I,1:I-1)
     end do
     AFF(1,1,1)= ONE
     AFF(2,1,1)= ONE
     AFF(2,2,1)= ONE
     AFF(3,1,1)= ONEPT5
     AFF(3,2,1)= 1.73205D0
     AFF(3,3,1)= 1.224745D0
     AFF(3,1,3)=-HALF
  ENDIF
  P    =(UA+UB)*ER*HALF
  BA   =(UA-UB)*ER*HALF
  EX   =EXP(BA)
  QUO  =1/P
  AF(1)=QUO*EXP(-P)
  NANB =NA+NB
  do N=1,19
     AF(N+1)=N*QUO*AF(N)+AF(1)
  end do
  NANB1=NANB+1
  CALL MNBFN(BA,BF)
  SUM =ZERO
  LAM1=LA-M+1
  LBM1=LB-M+1
  lp50: DO I=1,LAM1,2
     A =NA+I-LA
     IC=LA+2-I-M
     lp50a: DO J=1,LBM1,2
        B   =NB+J-LB
        ID  =LB-J-M+2
        SUM1=ZERO
        IA  =A+1
        IB  =B+1
        AB  =A+B-1
        lp40: DO K1=1,IA
           PART1=BI(IA,K1)
           lp40a: DO K2=1,IB
              PART2=PART1*BI(IB,K2)
              lp40b: DO K3=1,IC
                 PART3=PART2*BI(IC,K3)
                 lp40c: DO K4=1,ID
                    PART4=PART3*BI(ID,K4)
                    lp40d: DO K5=1,M
                       PART5=PART4*BI(M,K5)
                       Q=AB-K1-K2+K3+K4+2*K5
                       lp40e: DO K6=1,M
                          PART6=PART5*BI(M,K6)
                          PP=K1+K2+K3+K4+2*K6-5
                          JX=M+K2+K4+K5+K6-5
                          IX=JX/2
                          SUM1=SUM1+PART6*(IX*2-JX+HALF)*AF(Q)*BF(PP)
                       enddo lp40e
                    enddo lp40d
                 enddo lp40c
              enddo lp40b
           enddo lp40a
        enddo lp40
        SUM=SUM+SUM1*AFF(LA,M,I)*AFF(LB,M,J)*TWO
     enddo lp50a
  enddo lp50
  X=R**(NA+NB+1)*UA**NA*UB**NB
  SA=SUM*X*SQRT(UA*UB/(FA(NA+NA+1)*FA(NB+NB+1))*((LA+LA-1)*(LB+LB-1)))/(TWO**M)
  MNSS=SA
  RETURN
END FUNCTION MNSS

SUBROUTINE SWAP(C,N,MDIM,NOCC,IFILL)
  !
  use chm_kinds
  use number
  use sizes
  implicit none
  INTEGER N,MDIM,NOCC,IFILL
  real(chm_real) C(MDIM,MDIM), PSI(MAXORB), STDPSI(MAXORB)
  !
  INTEGER I,JFILL
  real(chm_real)  X,SUMMAX,SUM
  !
  !******************************************************************
  !
  !        SWAP ENSURES THAT A NAMED MOLECULAR ORBITAL IFILL IS FILLED
  ! ON INPUT
  !          C = EIGENVECTORS IN A MDIM*MDIM MATRIX
  !          N = NUMBER OF ORBITALS
  !          NOCC = NUMBER OF OCCUPIED ORBITALS
  !          IFILL = FILLED ORBITAL
  !******************************************************************
  !
  !     WE NOW DEFINE THE FILLED ORBITAL
  !
  If(IFILL.le.0) then
     IFILL=-IFILL
     do I=1,N
        STDPSI(I)=C(I,IFILL)
        PSI(I)=C(I,IFILL)
     end do
     RETURN
  End if
  !
  !     FIRST FIND THE LOCATION OF IFILL
  !
  SUM=ZERO
  do I=1,N
     SUM=SUM+PSI(I)*C(I,IFILL)
  end do
  if(ABS(SUM).le.0.7071D0) then
     !
     !     IFILL HAS MOVED!
     !
     SUMMAX=ZERO
     do IFILL=1,N
        SUM=ZERO
        do I=1,N
           SUM=SUM+STDPSI(I)*C(I,IFILL)
        end do
        SUM=ABS(SUM)
        if(SUM.GT.SUMMAX) then
           JFILL=IFILL
           SUMMAX=SUM
        end if
        IF(SUM.GT.0.7071D0) GOTO 90
     end do
     do IFILL=1,N
        SUM=ZERO
        do I=1,N
           SUM=SUM+PSI(I)*C(I,IFILL)
        end do
        SUM=ABS(SUM)
        if(SUM.GT.SUMMAX) then
           JFILL=IFILL
           SUMMAX=SUM
        end if
        IF(SUM.GT.0.7071D0) GOTO 90
     end do
     IFILL=JFILL
  end if      ! ABS(SUM).le.0.7071D0
90 CONTINUE
  !
  !    STORE THE NEW VECTOR IN PSI
  !
  PSI(1:N)=C(1:N,IFILL)
  !
  !    NOW CHECK TO SEE IF IFILL IS FILLED
  !
  IF(IFILL.LE.NOCC) RETURN
  !
  !    ITS EMPTY, SO SWAP IT WITH THE HIGHEST FILLED
  !
  do I=1,N
     X=C(I,NOCC)
     C(I,NOCC)=C(I,IFILL)
     C(I,IFILL)=X
  end do
  RETURN
END SUBROUTINE SWAP

SUBROUTINE MNTQL2(NM,N,D,E,Z,IERR,EPS)
  !
  !               ===== PROCESSED BY AUGMENT, VERSION 4N =====
  !     APPROVED FOR VAX 11/780 ON MAY 6,1980.  J.D.NEECE
  !               ----- LOCAL VARIABLES -----
  !               ----- GLOBAL VARIABLES -----
  use chm_kinds
  use number
  implicit none
  INTEGER NM,N,IERR
  real(chm_real) D(N), E(N), Z(NM,N)
  real(chm_real) EPS
  !               ----- SUPPORTING PACKAGE FUNCTIONS -----
  !               ===== TRANSLATED PROGRAM =====
  !
  !
  !     THIS SUBROUTINE IS A TRANSLATION OF THE ALGOL PROCEDURE TQL2,
  !     NUM. MATH. 11, 293-306(1968) BY BOWDLER, MARTIN, REINSCH, AND
  !     WILKINSON.
  !     HANDBOOK FOR AUTO. COMP., VOL.II-LINEAR ALGEBRA, 227-240(1971).
  !
  !     THIS SUBROUTINE FINDS THE EIGENVALUES AND EIGENVECTORS
  !     OF A SYMMETRIC TRIDIAGONAL MATRIX BY THE QL METHOD.
  !     THE EIGENVECTORS OF A FULL SYMMETRIC MATRIX CAN ALSO
  !     BE FOUND IF  TRED2  HAS BEEN USED TO REDUCE THIS
  !     FULL MATRIX TO TRIDIAGONAL FORM.
  !
  !     ON INPUT-
  !
  !        NM MUST BE SET TO THE ROW DIMENSION OF TWO-DIMENSIONAL
  !          ARRAY PARAMETERS AS DECLARED IN THE CALLING PROGRAM
  !          DIMENSION STATEMENT,
  !
  !        N IS THE ORDER OF THE MATRIX,
  !
  !        D CONTAINS THE DIAGONAL ELEMENTS OF THE INPUT MATRIX,
  !
  !        E CONTAINS THE SUBDIAGONAL ELEMENTS OF THE INPUT MATRIX
  !          IN ITS LAST N-1 POSITIONS.  E(1) IS ARBITRARY,
  !
  !        Z CONTAINS THE TRANSFORMATION MATRIX PRODUCED IN THE
  !          REDUCTION BY  TRED2, IF PERFORMED.  IF THE EIGENVECTORS
  !          OF THE TRIDIAGONAL MATRIX ARE DESIRED, Z MUST CONTAIN
  !          THE IDENTITY MATRIX.
  !
  !      ON OUTPUT-
  !
  !        D CONTAINS THE EIGENVALUES IN ASCENDING ORDER.  IF AN
  !          ERROR EXIT IS MADE, THE EIGENVALUES ARE CORRECT BUT
  !          UNORDERED FOR INDICES 1,2,...,IERR-1,
  !
  !        E HAS BEEN DESTROYED,
  !
  !        Z CONTAINS ORTHONORMAL EIGENVECTORS OF THE SYMMETRIC
  !          TRIDIAGONAL (OR FULL) MATRIX.  IF AN ERROR EXIT IS MADE,
  !          Z CONTAINS THE EIGENVECTORS ASSOCIATED WITH THE STORED
  !          EIGENVALUES,
  !
  !        IERR IS SET TO
  !          ZERO       FOR NORMAL RETURN,
  !          J          IF THE J-TH EIGENVALUE HAS NOT BEEN
  !                     DETERMINED AFTER 30 ITERATIONS.
  !
  !     QUESTIONS AND COMMENTS SHOULD BE DIRECTED TO B. S. GARBOW,
  !     APPLIED MATHEMATICS DIVISION, ARGONNE NATIONAL LABORATORY
  !
  !     ------------------------------------------------------------------
  !
  INTEGER II,I,J,K,L,M,MML,L1
  real(chm_real)  B,C,F,G,H,P,R,S
  !
  IERR = 0
  IF (N.EQ.1) return
  !
  do I=2,N
     E(I-1) = E(I)
  end do
  !
  F   =ZERO
  B   =ZERO
  E(N)=ZERO
  !
  loopLL: do L=1,N
     J = 0
     H =EPS*(ABS (D(L))+ABS (E(L)))
     IF(B.LT.H) B=H
     !     ********** LOOK FOR SMALL SUB-DIAGONAL ELEMENT **********
     do M=L,N
        IF(ABS(E(M)).LE.B)  EXIT ! GO TO 30
        !     ********** E(N) IS ALWAYS ZERO, SO THERE IS NO EXIT
        !                THROUGH THE BOTTOM OF THE LOOP **********
     end do
     !
     if(M.ne.L) then
        loopL2: do
           if(J.EQ.30) then
              !     ********** SET ERROR -- NO CONVERGENCE TO AN
              !                EIGENVALUE AFTER 30 ITERATIONS **********
              IERR = L
              return
           end if

           J =J+1
           !     ********** FORM SHIFT **********
           L1=L+1
           G =D(L)
           P   =(D(L1)-G)/(TWO*E(L))
           R   =SQRT(P*P+ONE)
           D(L)=E(L)/(P+SIGN (R,P))
           H   =G-D(L)
           !
           D(L1:N)=D(L1:N)-H
           F =F+H
           !     ********** QL TRANSFORMATION **********
           P  =D(M)
           C  =ONE
           S  =ZERO
           MML=M-L
           !     ********** FOR I=M-1 STEP -1 UNTIL L DO -- **********
           loopII: do II = 1,MML
              I = M - II
              G = C * E(I)
              H = C * P
              if(ABS(P).LT.ABS(E(I))) then
                 C=P/E(I)
                 R=SQRT(C*C+ONE)
                 E(I+1)=S*E(I)*R
                 S=ONE/R
                 C=C*S
              else
                 C=E(I)/P
                 R=SQRT(C*C+ONE)
                 E(I+1)=S*P*R
                 S=C/R
                 C=ONE/R
              end if
              P      = C*D(I)-S*G
              D(I+1) = H+S*(C*G+S*D(I))
              !     ********** FORM VECTOR **********
              do K =1,N
                 H        = Z(K,I+1)
                 Z(K,I+1) = S*Z(K,I) + C*H
                 Z(K,I)   = C*Z(K,I) - S*H
              end do
           end do loopII
           !
           E(L) = S*P
           D(L) = C*P
           if(ABS(E(L)).le.B) EXIT
        end do loopL2
     end if        ! M.ne.L 
     D(L) = D(L) + F
  End do loopLL
  !     ********** ORDER EIGENVALUES AND EIGENVECTORS **********
  loopI2: do II=2,N
     I = II - 1
     K = I
     P = D(I)
     !
     loopJJ: do J = II,N
        IF(D(J).GE.P) cycle loopJJ 
        K = J
        P = D(J)
     end do loopJJ
     !
     IF(K.EQ.I) cycle loopI2 
     D(K) = D(I)
     D(I) = P
     !
     do J=1,N
        P      = Z(J,I)
        Z(J,I) = Z(J,K)
        Z(J,K) = P
     end do
  end do loopI2

  RETURN
  !     ********** LAST CARD OF TQL2 **********
END SUBROUTINE MNTQL2

SUBROUTINE TQLRAT(N,D,E2,IERR,EPS)
  !
  !               ===== PROCESSED BY AUGMENT, VERSION 4N =====
  !     APPROVED FOR VAX 11/780 ON MAY 6,1980.  J.D.NEECE
  !               ----- LOCAL VARIABLES -----
  !               ----- GLOBAL VARIABLES -----
  use chm_kinds
  use number
  implicit none
  INTEGER N,IERR
  real(chm_real) D(N), E2(N)
  real(chm_real) EPS
  !               ----- SUPPORTING PACKAGE FUNCTIONS -----
  !               ===== TRANSLATED PROGRAM =====
  !
  !
  !     THIS SUBROUTINE IS A TRANSLATION OF THE ALGOL PROCEDURE TQLRAT,
  !     ALGORITHM 464, COMM. ACM 16, 689(1973) BY REINSCH.
  !
  !     THIS SUBROUTINE FINDS THE EIGENVALUES OF A SYMMETRIC
  !     TRIDIAGONAL MATRIX BY THE RATIONAL QL METHOD.
  !
  !     ON INPUT-
  !
  !        N IS THE ORDER OF THE MATRIX,
  !
  !        D CONTAINS THE DIAGONAL ELEMENTS OF THE INPUT MATRIX,
  !
  !        E2 CONTAINS THE SQUARES OF THE SUBDIAGONAL ELEMENTS OF THE
  !          INPUT MATRIX IN ITS LAST N-1 POSITIONS.  E2(1) IS ARBITRARY.
  !
  !      ON OUTPUT-
  !
  !        D CONTAINS THE EIGENVALUES IN ASCENDING ORDER.  IF AN
  !          ERROR EXIT IS MADE, THE EIGENVALUES ARE CORRECT AND
  !          ORDERED FOR INDICES 1,2,...IERR-1, BUT MAY NOT BE
  !          THE SMALLEST EIGENVALUES,
  !
  !        E2 HAS BEEN DESTROYED,
  !
  !        IERR IS SET TO
  !          ZERO       FOR NORMAL RETURN,
  !          J          IF THE J-TH EIGENVALUE HAS NOT BEEN
  !                     DETERMINED AFTER 30 ITERATIONS.
  !
  !     QUESTIONS AND COMMENTS SHOULD BE DIRECTED TO B. S. GARBOW,
  !     APPLIED MATHEMATICS DIVISION, ARGONNE NATIONAL LABORATORY
  !
  !     ------------------------------------------------------------------
  !
  INTEGER II,I,J,L,M,MML,L1
  real(chm_real)  B,C,F,G,H,P,R,S
  !
  IERR = 0
  IF (N.EQ.1) return
  !
  do I=2,N
     E2(I-1) = E2(I)
  end do
  !
  F    =ZERO
  B    =ZERO
  E2(N)=ZERO
  !
  loopLL: do L=1,N
     J = 0
     H=EPS*(ABS(D(L))+SQRT(E2(L)))
     if(B.le.H) then
        B=H
        C=B*B
     end if
     !     ********** LOOK FOR SMALL SQUARED SUB-DIAGONAL ELEMENT **********
     loopMM: do M=L,N
        IF (E2(M).LE.C) exit loopMM
        !     ********** E2(N) IS ALWAYS ZERO, SO THERE IS NO EXIT
        !                THROUGH THE BOTTOM OF THE LOOP **********
     end do loopMM
     !
     if(M.ne.L) then
        loopLLL: do
           if (J.EQ.30) then
              !     ********** SET ERROR -- NO CONVERGENCE TO AN
              !                EIGENVALUE AFTER 30 ITERATIONS **********
              IERR = L
              RETURN
           end if
           J  =J+1
           !     ********** FORM SHIFT **********
           L1  =L+1
           S   =SQRT(E2(L))
           G   =D(L)
           P   =(D(L1)-G)/(TWO*S)
           R   =SQRT(P*P+ONE)
           D(L)=S/(P+SIGN (R,P))
           H   =G-D(L)
           !
           D(L1:N) = D(L1:N)-H
           F       = F+H
           !     ********** RATIONAL QL TRANSFORMATION **********
           G   =D(M)
           IF(G.EQ.ZERO) G=B
           H   =G
           S   =ZERO
           MML =M-L
           !     ********** FOR I=M-1 STEP -1 UNTIL L DO -- **********
           do II=1,MML
              I = M-II
              P = G*H
              R = P+E2(I)
              E2(I+1)=S*R
              S = E2(I)/R
              D(I+1) = H+S*(H+D(I))
              G = D(I)-E2(I)/G
              IF(G.EQ.ZERO) G=B
              H = G*P/R
           end do
           !
           E2(L)= S*G
           D(L) = H
           !     ********** GUARD AGAINST UNDERFLOW IN CONVERGENCE TEST **********
           IF(H.EQ.ZERO .or. ABS(E2(L)).LE.ABS(C/H)) EXIT
           E2(L) = H*E2(L)
           if(E2(L).eq.ZERO) EXIT
        end do loopLLL
     end if    ! M.ne.L

     P = D(L) + F
     !     ********** ORDER EIGENVALUES **********
     if(L.ne.1) then
        !     ********** FOR I=L STEP -1 UNTIL 2 DO -- **********
        loopII: do II=2,L
           I   = L+2-II
           if(P.GE.D(I-1)) then
              D(I) = P   ! GO TO 110
              cycle loopLL
           end if
           D(I)= D(I-1)
        end do loopII
     end if
     I   =1
     D(I)=P
  end do loopLL
  !
  RETURN
  !     ********** LAST CARD OF TQLRAT **********
END SUBROUTINE TQLRAT

SUBROUTINE TRBAK3QM(NM,N,NV,A,M,Z)
  !
  !               ===== PROCESSED BY AUGMENT, VERSION 4N =====
  !     APPROVED FOR VAX 11/780 ON MAY 6,1980.  J.D.NEECE
  !               ----- LOCAL VARIABLES -----
  !               ----- GLOBAL VARIABLES -----
  use chm_kinds
  use number
  implicit none
  INTEGER NM,N,NV,M
  real(chm_real) A(NV), Z(NM,M)
  !               ===== TRANSLATED PROGRAM =====
  !
  !
  !     THIS SUBROUTINE IS A TRANSLATION OF THE ALGOL PROCEDURE TRBAK3,
  !     NUM. MATH. 11, 181-195(1968) BY MARTIN, REINSCH, AND WILKINSON.
  !     HANDBOOK FOR AUTO. COMP., VOL.II-LINEAR ALGEBRA, 212-226(1971).
  !
  !     THIS SUBROUTINE FORMS THE EIGENVECTORS OF A REAL SYMMETRIC
  !     MATRIX BY BACK TRANSFORMING THOSE OF THE CORRESPONDING
  !     SYMMETRIC TRIDIAGONAL MATRIX DETERMINED BY  TRED3.
  !
  !     ON INPUT-
  !
  !        NM MUST BE SET TO THE ROW DIMENSION OF TWO-DIMENSIONAL
  !          ARRAY PARAMETERS AS DECLARED IN THE CALLING PROGRAM
  !          DIMENSION STATEMENT,
  !
  !        N IS THE ORDER OF THE MATRIX,
  !
  !        NV MUST BE SET TO THE DIMENSION OF THE ARRAY PARAMETER A
  !          AS DECLARED IN THE CALLING PROGRAM DIMENSION STATEMENT,
  !
  !        A CONTAINS INFORMATION ABOUT THE ORTHOGONAL TRANSFORMATIONS
  !          USED IN THE REDUCTION BY  TRED3  IN ITS FIRST
  !          N*(N+1)/2 POSITIONS,
  !
  !        M IS THE NUMBER OF EIGENVECTORS TO BE BACK TRANSFORMED,
  !
  !        Z CONTAINS THE EIGENVECTORS TO BE BACK TRANSFORMED
  !          IN ITS FIRST M COLUMNS.
  !
  !     ON OUTPUT-
  !
  !        Z CONTAINS THE TRANSFORMED EIGENVECTORS
  !          IN ITS FIRST M COLUMNS.
  !
  !     NOTE THAT TRBAK3 PRESERVES VECTOR EUCLIDEAN NORMS.
  !
  !     QUESTIONS AND COMMENTS SHOULD BE DIRECTED TO B. S. GARBOW,
  !     APPLIED MATHEMATICS DIVISION, ARGONNE NATIONAL LABORATORY
  !
  !     ------------------------------------------------------------------
  !
  INTEGER IK,IZ,I,J,K,L
  real(chm_real)  H,S
  !
  IF (M.EQ.0 .or. N.EQ.1) return
  !
  loopII: do I=2,N
     L = I - 1
     IZ = (I * L) / 2
     IK = IZ + I
     H = A(IK)
     IF(H.EQ.ZERO) cycle loopII
     !
     loopJJ: do J=1,M
        S  =ZERO
        IK =IZ
        do K=1,L
           IK=IK+1
           S =S+A(IK)*Z(K,J)
        end do
        !     ********** DOUBLE DIVISION AVOIDS POSSIBLE UNDERFLOW **********
        S =(S/H)/H
        IK= IZ
        do K=1,L
           IK=IK+1
           Z(K,J)=Z(K,J)-S*A(IK)
        end do
     end do loopJJ
  end do loopII

  Return
  !     ********** LAST CARD OF TRBAK3 **********
END SUBROUTINE TRBAK3QM

SUBROUTINE TRED3QM(N,NV,A,D,E,E2)
  !
  !               ===== PROCESSED BY AUGMENT, VERSION 4N =====
  !     APPROVED FOR VAX 11/780 ON MAY 6,1980.  J.D.NEECE
  !               ----- LOCAL VARIABLES -----
  !               ----- GLOBAL VARIABLES -----
  use chm_kinds
  use number
  implicit none
  INTEGER NV,N
  real(chm_real) A(NV), D(N), E(N), E2(N)
  !               ----- SUPPORTING PACKAGE FUNCTIONS -----
  !               ===== TRANSLATED PROGRAM =====
  !
  !
  !     THIS SUBROUTINE IS A TRANSLATION OF THE ALGOL PROCEDURE TRED3,
  !     NUM. MATH. 11, 181-195(1968) BY MARTIN, REINSCH, AND WILKINSON.
  !     HANDBOOK FOR AUTO. COMP., VOL.II-LINEAR ALGEBRA, 212-226(1971).
  !
  !     THIS SUBROUTINE REDUCES A REAL SYMMETRIC MATRIX, STORED AS
  !     A ONE-DIMENSIONAL ARRAY, TO A SYMMETRIC TRIDIAGONAL MATRIX
  !     USING ORTHOGONAL SIMILARITY TRANSFORMATIONS.
  !
  !     ON INPUT-
  !
  !        N IS THE ORDER OF THE MATRIX,
  !
  !        NV MUST BE SET TO THE DIMENSION OF THE ARRAY PARAMETER A
  !          AS DECLARED IN THE CALLING PROGRAM DIMENSION STATEMENT,
  !
  !        A CONTAINS THE LOWER TRIANGLE OF THE REAL SYMMETRIC
  !          INPUT MATRIX, STORED ROW-WISE AS A ONE-DIMENSIONAL
  !          ARRAY, IN ITS FIRST N*(N+1)/2 POSITIONS.
  !
  !     ON OUTPUT-
  !
  !        A CONTAINS INFORMATION ABOUT THE ORTHOGONAL
  !          TRANSFORMATIONS USED IN THE REDUCTION,
  !
  !        D CONTAINS THE DIAGONAL ELEMENTS OF THE TRIDIAGONAL MATRIX,
  !
  !        E CONTAINS THE SUBDIAGONAL ELEMENTS OF THE TRIDIAGONAL
  !          MATRIX IN ITS LAST N-1 POSITIONS.  E(1) IS SET TO ZERO,
  !
  !        E2 CONTAINS THE SQUARES OF THE CORRESPONDING ELEMENTS OF E.
  !          E2 MAY COINCIDE WITH E IF THE SQUARES ARE NOT NEEDED.
  !
  !     QUESTIONS AND COMMENTS SHOULD BE DIRECTED TO B. S. GARBOW,
  !     APPLIED MATHEMATICS DIVISION, ARGONNE NATIONAL LABORATORY
  !
  !     ------------------------------------------------------------------
  !
  INTEGER II,JK,IZ,I,J,K,L
  real(chm_real)  HH,F,G,H,SCALE,temp1
  !
  !     ********** FOR I=N STEP -1 UNTIL 1 DO -- **********
  loopII: do II = 1, N
     I = N + 1 - II
     L = I - 1
     IZ = ( I * L ) / 2
     H=ZERO
     SCALE=ZERO
     do K=1,L
        IZ  =IZ+1
        D(K)=A(IZ)
        SCALE=SCALE+ABS(D(K))
     end do
     !
     if(SCALE.EQ.ZERO) then
        E(I) =ZERO
        E2(I)=ZERO
     else
        temp1=one/SCALE
        do K=1,L
           D(K)=D(K)*temp1  ! /SCALE
           H   =H+D(K)*D(K)
        end do
        !
        E2(I)= SCALE*SCALE*H
        F    = D(L)
        G    =-SIGN(SQRT(H),F)
        E(I) = SCALE*G
        H    = H-F*G
        D(L) = F-G
        A(IZ)= SCALE*D(L)
        if(L.ne.1) then
           F=ZERO
           !
           loopJJ: do J=1,L
              G  = ZERO
              JK =(J*(J-1))/2
              !     ********** FORM ELEMENT OF A*U **********
              K = 0
              do
                 K  =K+1
                 JK =JK+1
                 G  =G+A(JK)*D(K)
                 if(K.GE.J) EXIT
              end do
              if(K.ne.L) then
                 do
                    JK =JK+K
                    K  =K+1
                    G  =G+A(JK)*D(K)
                    if(K.GE.L) EXIT
                 end do
              end if
              !     ********** FORM ELEMENT OF P **********
              E(J)=G/H
              F   =F+E(J)*D(J)
           end do loopJJ
           !
           HH = F / (H + H)
           JK = 0
           !     ********** FORM REDUCED A **********
           do J=1,L
              F = D(J)
              G = E(J)-HH*F
              E(J)= G
              do K=1,J
                 JK   =JK+1
                 A(JK)=A(JK)-F*E(K)-G*D(K)
              end do
           end do
        end if  ! L.ne.1
     end if    ! SCALE.EQ.ZERO
     !
     D(I)   =A(IZ+1)
     A(IZ+1)=SCALE*SQRT(H)
  end do loopII
  !
  !     ********** LAST CARD OF TRED3 **********
  return
end SUBROUTINE TRED3QM


#endif /* (qmpac_main)*/
SUBROUTINE NULL_QP
  RETURN
END SUBROUTINE NULL_QP

