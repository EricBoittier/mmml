#if KEY_CFF==1
SUBROUTINE REWALD_CFF(ENB,EEL,NATOMX,JNBL,INBL,LELECX,LVDWX, &
     CCNBA,CCNBB,CCNBC,CCNBD,IACNB,NITCC2,LOWTP, &
     DX,DY,DZ,X,Y,Z,CGX,LUSED)
  !
  !-----------------------------------------------------------------------
  !     This routine calculates non bonded interaction energies and
  !     forces via the Ewald Summation
  !     ENBX    - vdw energy
  !     EELX    - electrostatic energy
  !     NATOMX  - number of atoms
  !     NGRPX   - number of groups
  !     JNBX    - nonbond atom pair list
  !     INBLOX  - pointers into JNBX for each atom
  !     LELECX  - compute electrostatic energy?
  !     LVDWX   - compute van der waal energy?
  !
  !     21-Oct-91, Roland Stote
  !                ATOM/VATOM nonbond cutoff options are supported.
  !----------------------------------------------------------------------
  !     This is the fast scalar version of the nonboned energy terms
  !     uses the Ewald summation method for calculating the electrostatic
  !     interaction potential.
  !     Van der Waals interactions can be shift of switched, but
  !     electrostatic interactions are truncated.
  !
  !     October 15, 1991   Roland H. Stote
  !-----------------------------------------------------------------------
  use ewald_1m,only: kappa,erfmod,ewnpts
  use erfcd_mod,only: ewrdel, ewldt
  use chm_kinds
  use dimens_fcm
  use number
  !
  use param
  use inbnd
  use consta
  !C  use fast
  !
  implicit none
  real(chm_real)  ENB,EEL
  INTEGER NATOMX,JNBL(*),INBL(*)
  LOGICAL LELECX,LVDWX
  real(chm_real)  CCNBA(*),CCNBB(*),CCNBC(*),CCNBD(*)
  INTEGER IACNB(*),NITCC2,LOWTP(*)
  real(chm_real)  X(*),Y(*),Z(*),DX(*),DY(*),DZ(*)
  real(chm_real)  CGX(*)
  LOGICAL LUSED
  !
  INTEGER IVECT,KVECT
  real(chm_real) CA,CB,CC,CH,ENE,ENN
  real(chm_real) TF,TX,TY,TZ,DTX,DTY,DTZ
  real(chm_real) S2,KRIJ,DFRS
  real(chm_real) TR2,TR6,FSW,DFSW,SS,TRS2
  ! JLS Added TRS2 = sqrt(TR2)      ^^
  !
  real(chm_real) CGF,CGT,CRXI,CRYI,CRZI,CRXJ,CRYJ,CRZJ
  INTEGER ITEMP,I,J,JJ,NPR,MAXCU2,IACI
  real(chm_real) C2ONNB,C2OFNB,CTROF2,C4ROF2,RUL3,RUL12,RIJL,RIJU
  real(chm_real) E14FM1,E14F,RS,R1S,ERFC2,ERFCX,DERFC
  real(chm_real) XVAL,REM,VAL0,VAL1,VAL2,D1,D2
  INTEGER IXVAL
  LOGICAL ELECFG
  !
  !MFC ERROR MAXCU2 used before it is defined
  !    Setting to 0 to avoid compiling warnings and
  !    unpredictable behavior
  maxcu2=0
  !MFC
  !
  ! Begin.
  !     check to see if we should be here...
  LUSED=.FALSE.
  IF(ERFMOD >= 0) RETURN
  IF(LGROUP) RETURN
  IF(LVFSWT) RETURN
  LUSED=.TRUE.
  !
  E14FM1 = E14FAC - ONE
  !
  C2OFNB=CTOFNB*CTOFNB
  CTROF2=-ONE/C2OFNB
  C4ROF2=FOUR*CTROF2
  IF (.NOT.(LSHFT.AND.LVSHFT)) THEN
     C2ONNB=CTONNB*CTONNB
     IF (CTOFNB > CTONNB) THEN
        RUL3=ONE/(C2OFNB-C2ONNB)**3
        RUL12=TWELVE*RUL3
     ENDIF
  ENDIF
  !
  ! SET SOME ZEROS
  ENB=ZERO
  EEL=ZERO
  ELECFG=(LELECX.AND.(EPS /= 0.0))
  IF (ELECFG) THEN
     CGF=CCELEC/EPS
  ELSE
     CGF=ZERO
  ENDIF
  IF(.NOT.(LVDW.OR.ELECFG)) RETURN
  !
  !     DO VDW AND ELECTROSTATIC TERMS AS REQUESTED
  !
  ITEMP=0
  loop60: DO I=1,NATOMX
#if KEY_IMCUBES==1
     if(lbycbim)then
           ITEMP=INBL(I+NATOMX)
     endif 
#endif 
     CGT = CGX(I)*CGF
     NPR=INBL(I) - ITEMP
     IF(NPR == 0) then
        ITEMP=INBL(I)
        cycle loop60
     endif
     IACI=IACNB(I)
     CRXI=X(I)
     CRYI=Y(I)
     CRZI=Z(I)
     DTX=ZERO
     DTY=ZERO
     DTZ=ZERO
     !
     loop30: DO JJ=1,NPR
        KVECT=JNBL(ITEMP+JJ)
        J=ABS(KVECT)
        CRXJ=X(J)
        CRYJ=Y(J)
        CRZJ=Z(J)
        !
        TX= CRXI-CRXJ
        TY= CRYI-CRYJ
        TZ= CRZI-CRZJ
        S2=MAX(RSMALL,TX*TX+TY*TY+TZ*TZ)
        IF (S2 <= C2OFNB) THEN

           !      van der Waals switch function
           IF (.NOT.(LVSHFT)) THEN
              IF (C2OFNB > C2ONNB) THEN
                 SS=MAX(S2,C2ONNB)
                 RIJL=C2ONNB-SS
                 RIJU=C2OFNB-SS
                 FSW=RIJU*RIJU*(RIJU-THREE*RIJL)*RUL3
                 DFSW=RIJL*RIJU*RUL12
              ELSE
                 FSW=ONE
                 DFSW=ZERO
              ENDIF
           ENDIF
           !
           IVECT=LOWTP(MAX(IACNB(J),IACI))+IACNB(J)+IACI
           CH=CGT*CGX(J)
           E14F = ZERO
           IF(KVECT < 0) THEN
              E14F = E14FM1
              IVECT=IVECT+MAXCU2
           ENDIF
           !
           TR2=ONE/S2
           TR6=TR2*TR2*TR2
           ! JLS  Changed 1/r^12 term to 1/r^9:
           TRS2 = SQRT(TR2)
           CA=CCNBA(IVECT)*TR6*TR2*TRS2
           !           CA=CCNBA(IVECT)*TR6*TR6
           CB=CCNBB(IVECT)*TR6
           !
           !     Calculate van der Waals and Electrostatic energies
           !
           IF (LVDWX) THEN
              IF (LVSHFT) THEN
                 CC=S2*S2*S2*CCNBC(IVECT)
                 ENN=CA-CB-CC+CCNBD(IVECT)
                 TF=THREE*(CB+CB-CA-CA-CA-CC)*TR2
              ELSE
                 ENN=(CA-CB)*FSW
                 TF=(CA-CB)*DFSW-THREE*TR2*(CA+CA+CA-CB-CB)*FSW
              ENDIF
           ELSE
              ENN = ZERO
              TF = ZERO
           ENDIF
           !
           !
           ! ERROR FUNCTION CALCULATION FOR REAL SPACE PART OF EWALD
           ! ---      EWALD R-space sum
           RS=SQRT(S2)
           R1S=ONE/RS
           !
           ! Inline erfc calculation for speed (from ERFCD).
           XVAL=RS*KAPPA*EWRDEL
           IXVAL=XVAL+HALF
           REM=XVAL-IXVAL
           IXVAL=IXVAL+2
           IXVAL=MIN(IXVAL,EWNPTS-1)
           VAL0=EwldT(IXVAL-1)
           VAL1=EwldT(IXVAL)
           VAL2=EwldT(IXVAL+1)
           D1=(VAL0-VAL2)*HALF
           D2=(VAL1+VAL1-VAL0-VAL2)*REM
           ERFCX=VAL1-(D1+HALF*D2)*REM
           DERFC=(D1+D2)*EWRDEL*KAPPA
           !C end of erfc inlined code.
           !
           ERFC2=ERFCX + E14F
           ENE =CH*ERFC2*R1S
           DFRS=(CH*DERFC + ENE)*TR2
           !
           TF=TF-DFRS
           !
           DTX=DTX+TF*TX
           DTY=DTY+TF*TY
           DTZ=DTZ+TF*TZ
           DX(J)=DX(J)-TF*TX
           DY(J)=DY(J)-TF*TY
           DZ(J)=DZ(J)-TF*TZ
           !
           ENB=ENB+ENN
           EEL=EEL+ENE
           !
        ENDIF
     enddo loop30
     !                        restore i-th component of force in the array
     DX(I)=DX(I)+DTX
     DY(I)=DY(I)+DTY
     DZ(I)=DZ(I)+DTZ
     ITEMP=INBL(I)
  enddo loop60
  RETURN
end SUBROUTINE REWALD_CFF
#endif 

SUBROUTINE NULL_EWCFF
END SUBROUTINE NULL_EWCFF

