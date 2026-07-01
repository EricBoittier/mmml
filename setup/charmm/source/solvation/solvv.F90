#if KEY_RISM==0 /*rism_main*/

SUBROUTINE NULL_SOLVV
  RETURN
end SUBROUTINE NULL_SOLVV

#else /* (rism_main)*/

SUBROUTINE  SOLVV(NSITV,NPVV,INTV,INTVV,X,Y,Z,SEGID,A,B,C, &
     USR,USR2,PHIK,GR,XVVK,CSR,TSR,CSK,TSK,CS2R,W,WI,PMFR, &
     CLOS,KBT,RHO,ADIE,AXTR,RXTR1,RXTR2,SHIFT, &
     SW,SWI,SWF,DSW,NSW,IUNCSR,IUNGR,NDMAX,NCYCLE, &
     NPRINT,TOL,RMIX,QINIT,LQUSR,LQGR,LQWR,LQTSR,LQBDR)

  !     This subroutine sets up the site-site potential and the long
  !     range phik bond.  It includes loops on the 4
  !     switching parameters
  !     on the potential sw(1)
  !     on the sigma     sw(2)
  !     on the charges   sw(3)
  !     on the bridge    sw(4)

  use chm_kinds
  use number
  use rism
  use fft
  use stream
  implicit none
  !     Structural and Site-site potential parameters
  real(chm_real) X(*),Y(*),Z(*)
  CHARACTER(len=*) SEGID(*)
  real(chm_real) A(*),B(*),C(*)

  !     Inter and intramolecular matrix and index pointer arrays
  real(chm_real) USR(DVECT,*),USR2(DVECT,*),PHIK(DVECT,*),GR(DVECT,*) &
       ,XVVK(DVECT,*),CSR(DVECT,*),TSR(DVECT,*),CSK(DVECT,*), &
       TSK(DVECT,*),CS2R(DVECT,*),W(DVECT,*),WI(DVECT,*), &
       PMFR(DVECT,*)

  INTEGER NSITV,NPVV,INTV(DSITV,DSITV), &
       INTVV(DSITV,DSITV)

  !     Physical control parameters
  CHARACTER(len=3) CLOS
  real(chm_real) KBT,RHO(*),ADIE,AXTR(*),RXTR1(*),RXTR2(*)
  real(chm_real) SHIFT(DVECT,*)

  !     Cycle control parameters, switches mixing and tolerance
  INTEGER IUNCSR,IUNGR,NSW(*),NPRINT,NCYCLE
  real(chm_real) SW(*),SWI(*),SWF(*),DSW(*),RMIX,TOL
  LOGICAL QINIT,CONVRG,LQUSR,LQGR,LQWR,LQTSR,LQBDR

  !     Local Variables
  real(chm_real) DMAX
  INTEGER NDMAX
  INTEGER IK,I,J,ISW1,ISW2,ISW3,ISW4,ICYCLE,IP,IR

  !     Construct the constant matrices for CYCLE3 (10)
  !     ------------------------------------------

  !     Make the intramolecular correlation matrix with density rho
  CALL MKOMEG(X,Y,Z,SEGID,NSITV,INTV,DSITV,W,WI,'INVRS')
  DO IK=NFIRST,NPOINT
     DO I=1,NSITV
        DO J=1,I
           W(IK,INTV(I,J))=W(IK,INTV(I,J))*RHO(J)
           WI(IK,INTV(I,J))=WI(IK,INTV(I,J))/RHO(J)
        ENDDO
     ENDDO
  ENDDO


  !     Main Switching Loop (20)
  !     ===================

  DO ISW1=1,NSW(1)
     SW(1)=SWI(1)+(ISW1-1)*DSW(1)

     DO ISW2=1,NSW(2)
        SW(2)=SWI(2)+(ISW2-1)*DSW(2)

        !     Make the scaled short range potential
        IF(LQUSR)THEN
           DO IP=1,NPVV
              DO IR=NFIRST,NPOINT
                 USR2(IR,IP)=SW(1)*USR(IR,IP)
              ENDDO
           ENDDO
        ELSE
           CALL MKUSR(NPVV,A,B,SW(1),SW(2),KBT,USR2)
        ENDIF

        DO ISW3=1,NSW(3)
           SW(3)=SWI(3)+(ISW3-1)*DSW(3)

           !     Make the Coulomb phi bond in k-space
           CALL MKPHIK(NPVV,C,ADIE,SW(1),SW(3),KBT,PHIK)

           DO ISW4=1,NSW(4)
              SW(4)=SWI(4)+(ISW4-1)*DSW(4)

              IF(CLOS.EQ.'XTR')THEN
                 !     Set up the "experimental bridge function" and the shift function
                 CALL SETXTR(SW(1),SW(4),GR,PMFR,SHIFT,RXTR1,RXTR2,AXTR,NPVV)
              ENDIF

              WRITE(OUTU,98) (SW(I),I=1,4)
98            FORMAT(1X,'switch =',4F9.4)


              !     Call the elementary cycle
              !     -------------------------
              IF(QINIT)THEN
                 WRITE(OUTU,99)
99               FORMAT(' The iteration cycle is initialized')
                 CALL INIT0(NPOINT,NPVV,CSR)

                 CALL SOLVV2(NPVV,NSITV,INTV,INTVV,USR2,PHIK,W,WI, &
                      ICYCLE,2,NPRINT,CLOS,RHO,TOL, &
                      ZERO,CONVRG,NDMAX,DMAX,GR,PMFR,SHIFT, &
                      CSR,TSR,CSK,TSK,CS2R)

                 QINIT=.FALSE.
              ENDIF
              CALL SOLVV2(NPVV,NSITV,INTV,INTVV,USR2,PHIK,W,WI, &
                   ICYCLE,NCYCLE,NPRINT,CLOS,RHO,TOL, &
                   RMIX,CONVRG,NDMAX,DMAX,GR,PMFR,SHIFT, &
                   CSR,TSR,CSK,TSK,CS2R)


              IF(IUNCSR.NE.0)THEN
                 !     append to the save file unit=iuncsr
                 CALL WRTDAT(IUNCSR,CSR,NPOINT,NPVV,SW)
              ELSEIF(IUNGR.NE.0)THEN
                 !     append to the save file unit=iungr
                 CALL MKGR(CLOS,GR,TSR,USR2,NPVV,PMFR,SHIFT)
                 WRITE(OUTU,102)
                 CALL WRTDAT(IUNGR,GR,NPOINT,NPVV,SW)
              ENDIF

              IF(CONVRG)THEN
                 WRITE(OUTU,100) ICYCLE,DMAX
100              FORMAT(6X,'convergence reached at cycle ',I8,' dmax=',F10.5)
              ELSE
                 WRITE(OUTU,101) DMAX
101              FORMAT(/,6X,'* Warning *', &
                      ' loop ended before tolerance was satisfied, dmax',F10.5)
                 ! If convergence was not achieved return
                 !
                 GOTO 1000
              ENDIF
              ! end of switching loop
              !
           ENDDO
        ENDDO
     ENDDO
  ENDDO

1000 CONTINUE

  !     Make the solvent-solvent suceptibility Xvvk
  !     -------------------------------------------
  CALL MKXVVK(NSITV,INTV,INTVV,W,TSK,CSK,RHO,XVVK)

  !     Make the radial distribution function
  !     -------------------------------------
  IF((CLOS.NE.'XTR').OR.LQGR)THEN
     CALL MKGR(CLOS,GR,TSR,USR2,NPVV,PMFR,SHIFT)
     WRITE(OUTU,102)
102  FORMAT(6X,'The radial distribution function is generated')
  ENDIF

  !     Make the potential of mean force and put it in usr(1) = usr(1)
  !     --------------------------------------------------------------
  IF(LQTSR)THEN
     DO IP=1,NPVV
        DO IR=NFIRST,NPOINT
           USR(IR,IP)=TSR(IR,IP)
        ENDDO
     ENDDO
     WRITE(OUTU,103)
103  FORMAT(6X,'The cavity potential h(r)-c(r) is stored in us(r)')
  ELSEIF(LQWR)THEN
     CALL MKWR(CLOS,KBT,TSR,USR2,NPVV,PMFR,SHIFT)
     DO IP=1,NPVV
        DO IR=NFIRST,NPOINT
           USR(IR,IP)=USR2(IR,IP)
        ENDDO
     ENDDO
     WRITE(OUTU,104)
104  FORMAT(6X,'The Potential of Mean Force is stored in us(r)')
  ELSEIF(LQBDR.AND.(CLOS.EQ.'XTR'))THEN
     DO IP=1,NPVV
        DO IR=NFIRST,NPOINT
           IF(GR(IR,IP).LT.1.D-36)THEN
              USR(IR,IP)=1.0D12
           ELSE
              USR(IR,IP)= USR2(IR,IP)+ &
                   (PMFR(IR,IP)-USR2(IR,IP)+TSR(IR,IP))*SHIFT(IR,IP)
           ENDIF
        ENDDO
     ENDDO
     WRITE(OUTU,105)
105  FORMAT(6X,'The bridge function bd(r) is stored in us(r)')
  ENDIF
  WRITE(OUTU,*)

  RETURN
END SUBROUTINE SOLVV

!====================================================================

SUBROUTINE SOLVV2(NPVV,NSITV,INTV,INTVV,USR,PHIK,W,WI, &
     ICYCLE,NCYCLE,NPRINT,CLOS,RHO,TOL, &
     RMIX,CONVRG,NDMAX,DMAX,GR,PMFR,SHIFT, &
     CSR,TSR,CSK,TSK,CS2R)

  !     This subroutine performs the elementary cycle "ncycle" times
  !     or until the "tol" convergence is reached

  use chm_kinds
  use number
  use rism
  use fft
  use stream
  implicit none

  !     Pointer index variables
  INTEGER NPVV,NSITV,INTV(DSITV,DSITV),INTVV(DSITV,DSITV)

  !     Site-site potential and intramolecular matrix
  real(chm_real) USR(DVECT,*),PHIK(DVECT,*),W(DVECT,*),WI(DVECT,*)

  !     Control parameters
  INTEGER ICYCLE,NCYCLE,NPRINT
  CHARACTER(len=3) CLOS
  real(chm_real) RHO(*),TOL,RMIX
  LOGICAL CONVRG
  INTEGER NDMAX
  real(chm_real)  DMAX
  !     for the XTR closure
  real(chm_real) GR(DVECT,*),SHIFT(DVECT,*),PMFR(DVECT,*)

  !     Intra and intermolecular correlation matrix and pointer arrays
  real(chm_real) CSR(DVECT,*),TSR(DVECT,*),CSK(DVECT,*), &
       TSK(DVECT,*),CS2R(DVECT,*)

  !
  !     Local variable: matrix used for inversion in VV calculation
  real(chm_real) RM(DSITE,DSITE),RMI(DSITE,DSITE)
  real(chm_real) DMAX2,DIFF

  INTEGER IP,IK,I,J,IR,IPRINT
  WRITE(OUTU,*)


  !     Elementary cycle to solve the integral equations  (10)
  !     ================================================
  loop10:DO ICYCLE=1,NCYCLE


     !     Fourier transform the csr to get csk  (20)
     !     ------------------------------------
     IF(QLOG)THEN
        DO IP=1,NPVV
           CALL LOGFFT(CSR(1,IP),CSK(1,IP),TP32)
        ENDDO
     ELSE
        DO IP=1,NPVV
           CALL LINFFT(CSR(1,IP),CSK(1,IP),DVR)
        ENDDO
     ENDIF


     !     Calculate the tsk from the csk  (30)
     !     ------------------------------
     DO IK=NFIRST,NPOINT
        DO I=1,NSITV
           DO J=1,I
              RM(I,J)=WI(IK,INTV(I,J))- &
                   (CSK(IK,INTVV(I,J))+PHIK(IK,INTVV(I,J)))
              RM(J,I)=RM(I,J)
           ENDDO
        ENDDO
        CALL INVRS(RM,RMI,NSITV)
        DO I=1,NSITV
           DO J=1,I
              TSK(IK,INTVV(I,J))= &
                   (RMI(I,J)-W(IK,INTV(I,J)))/(RHO(I)*RHO(J)) &
                   -CSK(IK,INTVV(I,J))
           ENDDO
        ENDDO
     ENDDO

     !     Fourier transform the tsk to get the tsr (40)
     !     ----------------------------------------
     IF(QLOG)THEN
        DO IP=1,NPVV
           CALL LOGFFT(TSK(1,IP),TSR(1,IP),TPM32)
        ENDDO
     ELSE
        DO IP=1,NPVV
           CALL LINFFT(TSK(1,IP),TSR(1,IP),DVK)
        ENDDO
     ENDIF

     !     Apply the closure to get a new CS2r  (50)
     !     -----------------------------------
     CALL CLOSURE(CLOS,USR,CS2R,TSR,NPVV,NPOINT,SHIFT,PMFR,GR)

     !     Comparison and mixing of CS2r and CSr (60)
     !     -------------------------------------
     DMAX=ZERO
     DO IR=1,NPOINT
        DO IP=1,NPVV
           DIFF=ABS(CS2R(IR,IP)-CSR(IR,IP))
           IF(DIFF.GT.DMAX) DMAX=DIFF
           CSR(IR,IP)=(ONE-RMIX)*CS2R(IR,IP)+RMIX*CSR(IR,IP)
        ENDDO
     ENDDO
     IF(MOD(ICYCLE,NPRINT).EQ.0)THEN
        WRITE(OUTU,100) ICYCLE,DMAX
100     FORMAT(6X,'cycle',I5,'  dmax',F10.5)
        IPRINT=0
     ENDIF
     CONVRG=DMAX.LT.TOL
     ! exit the cycle loop when tol is reached
     !
     IF(CONVRG) GOTO 1000
     IF(ICYCLE.EQ.1) DMAX2=DMAX
     !
     !     Check for divergence of the iterations
     IF(MOD(ICYCLE,NDMAX).EQ.0)THEN
        IF(DMAX.GT.DMAX2)THEN
           RMIX=RMIX**0.8
           WRITE(OUTU,101) ICYCLE,DMAX2,DMAX,RMIX
101        FORMAT(6X,'cycle',I5,' dmax ',F10.5,' increase to ',F10.5,/, &
                6X,'mixing factor increased to ',F10.5)
        ENDIF
        DMAX2=DMAX
     ENDIF
     ! Continue the icycle loop
     !
  enddo loop10
  ICYCLE=NCYCLE

1000 CONTINUE
  RETURN

END SUBROUTINE SOLVV2


!============================================================================

SUBROUTINE CLOSURE(CLOS,USR,CS2R,TSR,NPAIR,NPOINT,SHIFT,PMFR,GR)
  use chm_kinds
  use number
  use rism
  implicit none
  CHARACTER(len=3) CLOS
  real(chm_real) USR(DVECT,*),CS2R(DVECT,*),TSR(DVECT,*), &
       SHIFT(DVECT,*),PMFR(DVECT,*),GR(DVECT,*)
  INTEGER NPAIR,NPOINT
  real(chm_real) BDR

  INTEGER IP,IR

  IF(CLOS.EQ.'HNC')THEN
     !     ---------------------
     DO IP=1,NPAIR
        DO IR=1,NPOINT
           CS2R(IR,IP)=EXP(-USR(IR,IP)+TSR(IR,IP))-TSR(IR,IP)-ONE
        ENDDO
     ENDDO

  ELSEIF(CLOS.EQ.'XTR')THEN
     !     -------------------------
     DO IP=1,NPAIR
        DO IR=1,NPOINT
           IF(GR(IR,IP).LT.RPRECI)THEN
              BDR=RBIG
           ELSE
              BDR=(PMFR(IR,IP)-USR(IR,IP)+TSR(IR,IP))*SHIFT(IR,IP)
           ENDIF
           CS2R(IR,IP)=EXP(-USR(IR,IP)+TSR(IR,IP)-BDR) &
                -TSR(IR,IP)-ONE
        ENDDO
     ENDDO

  ELSEIF(CLOS.EQ.'PY ')THEN
     !     -------------------------
     DO IP=1,NPAIR
        DO IR=1,NPOINT
           CS2R(IR,IP)=EXP(-USR(IR,IP))*(ONE+TSR(IR,IP)) &
                -TSR(IR,IP)-ONE
        ENDDO
     ENDDO

  ELSEIF(CLOS.EQ.'PY2')THEN
     !     -------------------------
     DO IP=1,NPAIR
        DO IR=1,NPOINT
           CS2R(IR,IP)=EXP(-USR(IR,IP))* &
                (ONE+TSR(IR,IP)+HALF*TSR(IR,IP)**2) &
                -TSR(IR,IP)-ONE
        ENDDO
     ENDDO
  ENDIF
  RETURN
END SUBROUTINE CLOSURE
!============================================================================

SUBROUTINE CLOSUREUV(CLOS,USR,CS2R,TSR,NPAIR,NPOINT)
  !     This subroutine solves the closure equation for the solute-solvent
  !     distribution function
  use chm_kinds
  use number
  use rism
  implicit none
  CHARACTER(len=3) CLOS
  real(chm_real) USR(DVECT,*),CS2R(DVECT,*),TSR(DVECT,*)
  INTEGER NPAIR,NPOINT

  INTEGER IP,IR

  IF(CLOS.EQ.'HNC')THEN
     !     ---------------------
     DO IP=1,NPAIR
        DO IR=1,NPOINT
           CS2R(IR,IP)=EXP(-USR(IR,IP)+TSR(IR,IP))-TSR(IR,IP)-ONE
        ENDDO
     ENDDO

  ELSEIF(CLOS.EQ.'XTR')THEN
     !     -------------------------
     CALL WRNDIE(-3,'<CLOSUREUV>','XTR CLOSURE IMPLEMENTED ONLY FOR'// &
          'PURE SOLVENT CALCULATIONS')

  ELSEIF(CLOS.EQ.'PY ')THEN
     !     -------------------------
     DO IP=1,NPAIR
        DO IR=1,NPOINT
           CS2R(IR,IP)=EXP(-USR(IR,IP))*(ONE+TSR(IR,IP)) &
                -TSR(IR,IP)-ONE
        ENDDO
     ENDDO

  ELSEIF(CLOS.EQ.'PY2')THEN
     !     -------------------------
     DO IP=1,NPAIR
        DO IR=1,NPOINT
           CS2R(IR,IP)=EXP(-USR(IR,IP))* &
                (ONE+TSR(IR,IP)+HALF*TSR(IR,IP)**2) &
                -TSR(IR,IP)-ONE
        ENDDO
     ENDDO
     !
  ENDIF
  RETURN
END SUBROUTINE CLOSUREUV

!================================================================

SUBROUTINE MKGR(CLOS,GR,TSR,USR,NPAIR,PMFR,SHIFT)
  !     This subroutine makes the radial distribution function g(r)
  !     for the different pairs using the appropriate closure.

  use chm_kinds
  use number
  use rism
  use fft
  implicit none

  !     Closure
  CHARACTER(len=3) CLOS

  !     Intermolecular functions
  real(chm_real) USR(DVECT,*),GR(DVECT,*),TSR(DVECT,*), &
       PMFR(DVECT,*),SHIFT(DVECT,*)
  real(chm_real) BDR
  INTEGER NPAIR

  INTEGER IP,IR

  IF(CLOS.EQ.'HNC')THEN
     !     ---------------------
     DO IP=1,NPAIR
        DO IR=1,NPOINT
           GR(IR,IP)=EXP(-USR(IR,IP)+TSR(IR,IP))
        ENDDO
     ENDDO

  ELSEIF(CLOS.EQ.'PY ')THEN
     !     -------------------------
     DO IP=1,NPAIR
        DO IR=1,NPOINT
           GR(IR,IP)=EXP(-USR(IR,IP))*(ONE+TSR(IR,IP))
        ENDDO
     ENDDO

  ELSEIF(CLOS.EQ.'PY2')THEN
     !     -------------------------
     DO IP=1,NPAIR
        DO IR=1,NPOINT
           GR(IR,IP)=EXP(-USR(IR,IP))* &
                (ONE+TSR(IR,IP)+HALF*TSR(IR,IP)**2)
        ENDDO
     ENDDO

  ELSEIF(CLOS.EQ.'XTR')THEN
     !     ---------------------
     DO IP=1,NPAIR
        DO IR=1,NPOINT
           IF(GR(IR,IP).LT.RPRECI)THEN
              BDR=RBIG
           ELSE
              BDR=(PMFR(IR,IP)-USR(IR,IP)+TSR(IR,IP))*SHIFT(IR,IP)
           ENDIF
           GR(IR,IP)=EXP(-USR(IR,IP)+TSR(IR,IP)-BDR)
        ENDDO
     ENDDO

  ENDIF
  RETURN
END SUBROUTINE MKGR

!================================================================

SUBROUTINE MKGRUV(CLOS,GR,TSR,USR,NPAIR)
  !     This subroutine makes the radial distribution function g(r)
  !     for the different solute-solvent pairs using the appropriate closure.

  use chm_kinds
  use number
  use rism
  use fft
  implicit none

  !     Closure
  CHARACTER(len=3) CLOS

  !     Intermolecular functions
  real(chm_real) USR(DVECT,*),GR(DVECT,*),TSR(DVECT,*)
  INTEGER NPAIR

  INTEGER IP,IR

  IF(CLOS.EQ.'HNC')THEN
     !     ---------------------
     DO IP=1,NPAIR
        DO IR=1,NPOINT
           GR(IR,IP)=EXP(-USR(IR,IP)+TSR(IR,IP))
        ENDDO
     ENDDO

  ELSEIF(CLOS.EQ.'PY ')THEN
     !     -------------------------
     DO IP=1,NPAIR
        DO IR=1,NPOINT
           GR(IR,IP)=EXP(-USR(IR,IP))*(ONE+TSR(IR,IP))
        ENDDO
     ENDDO

  ELSEIF(CLOS.EQ.'PY2')THEN
     !     -------------------------
     DO IP=1,NPAIR
        DO IR=1,NPOINT
           GR(IR,IP)=EXP(-USR(IR,IP))* &
                (ONE+TSR(IR,IP)+HALF*TSR(IR,IP)**2)
        ENDDO
     ENDDO

  ELSEIF(CLOS.EQ.'XTR')THEN
     !     ---------------------
     CALL WRNDIE(-3,'<MKGRUV>','CLOSURE XTR IMPLEMENTED ONLY FOR '// &
          'PURE SOLVENT CALCULATION')
  ENDIF
  RETURN
END SUBROUTINE MKGRUV

!=============================================================================

SUBROUTINE MKXVVK(NSITV,INTV,INTVV,W,TSK,CSK,RHO,XVVK)
  !     Make the solvent-solvent suceptibility for infinite dilution
  !     calculations
  use chm_kinds
  use rism
  use fft
  implicit none
  INTEGER NSITV,INTV(DSITV,DSITV),INTVV(DSITV,DSITV)
  real(chm_real) W(DVECT,*),TSK(DVECT,*),CSK(DVECT,*),RHO(*), &
       XVVK(DVECT,*)

  INTEGER IK,I,J

  DO IK=NFIRST,NPOINT
     DO I=1,NSITV
        DO J=1,I
           XVVK(IK,INTV(I,J))=W(IK,INTV(I,J))+ &
                RHO(I)*(TSK(IK,INTVV(I,J))+CSK(IK,INTVV(I,J)))*RHO(J)
        ENDDO
     ENDDO
  ENDDO
  RETURN
END SUBROUTINE MKXVVK

!=============================================================================

SUBROUTINE MKPHIK(NPAIR,C,ADIE,SW1,SW3,KBT,PHIK)
  !     This subroutine builds the phi Coulomb bond in k-space
  use chm_kinds
  use number
  use rism
  use fft
  use consta
  implicit none
  INTEGER NPAIR
  real(chm_real) C(*),ADIE,SW1,SW3,KBT,PHIK(DVECT,*)
  real(chm_real) FACT,RKM2

  INTEGER IK,IP

  FACT=-FOUR*PI*ADIE*SW1*SW3*SW3/KBT
  DO IK=NFIRST,NPOINT
     ! This is the fourier transform 1/k**2
     !
     RKM2=FACT/RK(IK)**2
     DO IP=1,NPAIR
        PHIK(IK,IP)=C(IP)*RKM2
     ENDDO
  ENDDO
  RETURN
END SUBROUTINE MKPHIK

!=============================================================================

SUBROUTINE MKPHIR(NPAIR,C,ADIE,SW1,SW3,KBT,PHIR)
  !     THIS SUBROUTINE BUILDS THE PHI COULOMB BOND IN R-SPACE
  use chm_kinds
  use rism
  use fft
  implicit none
  INTEGER NPAIR
  real(chm_real) C(*),ADIE,SW1,SW3,KBT,PHIR(DVECT,*)
  real(chm_real) FACT,RM1

  INTEGER IR,IP

  FACT=-ADIE*SW1*SW3*SW3/KBT
  DO IR=NFIRST,NPOINT
     RM1=FACT/R(IR)
     DO IP=1,NPAIR
        PHIR(IR,IP)=C(IP)*RM1
     ENDDO
  ENDDO

  RETURN
END SUBROUTINE MKPHIR


!=============================================================================

SUBROUTINE MKUSR(NPAIR,A,B,SW1,SW2,KBT,USR)
  !     THIS SUBROUTINE BUILDS THE SHORT RANGE POTENTIAL USR
  use chm_kinds
  use rism
  use fft
  implicit none
  INTEGER NPAIR
  real(chm_real) A(*),B(*),SW1,SW2,KBT,USR(DVECT,*)
  real(chm_real) RM6,RM12
  LOGICAL OVRFLW
  real(chm_real), PARAMETER :: RRBIG=1.0E38,BIGLOG=85.0
  INTEGER IR,IP

  DO IR=NFIRST,NPOINT
     RM6=(SW2/R(IR))**6
     ! This is the repulsive term 1/r**12
     !
     RM12=RM6*RM6
     DO IP=1,NPAIR
        OVRFLW=(LOG(A(IP))+LOG(RM12)) .GT. BIGLOG
        IF(OVRFLW)THEN
           USR(IR,IP)=RRBIG
        ELSE
           USR(IR,IP)=SW1*(A(IP)*RM12+B(IP)*RM6)/KBT
        ENDIF
     ENDDO
  ENDDO

  RETURN
END SUBROUTINE MKUSR

!================================================================

SUBROUTINE MKWR(CLOS,KBT,TSR,USR,NPAIR,PMFR,SHIFT)
  !     This subroutine makes W(r) for the different pairs
  !     using the appropriate closure and put it in usr

  use chm_kinds
  use rism
  use fft
  implicit none

  !     Closure
  CHARACTER(len=3) CLOS

  !     Intermolecular functions
  INTEGER NPAIR
  real(chm_real) KBT,USR(DVECT,*),TSR(DVECT,*), &
       SHIFT(DVECT,*),PMFR(DVECT,*),BDR

  INTEGER IP,IR

  IF(CLOS.EQ.'HNC')THEN
     !     ---------------------
     DO IP=1,NPAIR
        DO IR=1,NPOINT
           USR(IR,IP)=KBT*(USR(IR,IP)-TSR(IR,IP))
        ENDDO
     ENDDO

  ELSEIF(CLOS.EQ.'XTR')THEN
     !     -------------------------
     DO IP=1,NPAIR
        DO IR=1,NPOINT
           BDR=(PMFR(IR,IP)-USR(IR,IP)+TSR(IR,IP))*SHIFT(IR,IP)
           USR(IR,IP)=USR(IR,IP)-TSR(IR,IP)+BDR
        ENDDO
     ENDDO

  ENDIF

  RETURN
END SUBROUTINE MKWR
!================================================================

SUBROUTINE MKWRUV(CLOS,KBT,TSR,USR,NPAIR)
  !     This subroutine makes W(r) for the different solute-solvent pairs
  !     using the appropriate closure and put it in usr

  use chm_kinds
  use rism
  use fft
  implicit none

  !     Closure
  CHARACTER(len=3) CLOS

  !     Intermolecular functions
  INTEGER NPAIR
  real(chm_real) KBT,USR(DVECT,*),TSR(DVECT,*)

  INTEGER IP,IR

  IF(CLOS.EQ.'HNC')THEN
     !     ---------------------
     DO IP=1,NPAIR
        DO IR=1,NPOINT
           USR(IR,IP)=KBT*(USR(IR,IP)-TSR(IR,IP))
        ENDDO
     ENDDO
     !
  ENDIF

  RETURN
END SUBROUTINE MKWRUV

!===========================================================================
SUBROUTINE SETXTR(SW1,SW4,GR,PMFR,SHIFT,RXTR1,RXTR2,AXTR,NPAIR)
  !     This subroutine sets up the "bridge" functions
  use chm_kinds
  use number
  use rism
  use fft
  implicit none
  INTEGER NPAIR
  real(chm_real) SW1,SW4,GR(DVECT,*),PMFR(DVECT,*), &
       SHIFT(DVECT,*),RXTR1(*),RXTR2(*),AXTR(*)
  real(chm_real) RXTR3,RXTR4
  INTEGER IP,IR

  DO IP=1,NPAIR
     RXTR3=RXTR2(IP)-RXTR1(IP)
     DO IR=NFIRST,NPOINT
        IF(R(IR).LE.RXTR2(IP))THEN
           IF(R(IR).LE.RXTR1(IP))THEN
              SHIFT(IR,IP)=AXTR(IP)*SW4
           ELSE
              RXTR4=R(IR)-RXTR1(IP)
              SHIFT(IR,IP)=AXTR(IP)*SW4*(TWO*(RXTR4/RXTR3)**3 &
                   -THREE*(RXTR4/RXTR3)**2+ONE)
           ENDIF
        ELSE
           SHIFT(IR,IP)=ZERO
        ENDIF
     ENDDO
  ENDDO

  DO IP=1,NPAIR
     DO IR=NFIRST,NPOINT
        IF(GR(IR,IP).EQ.ZERO)THEN
           PMFR(IR,IP)=RBIG
        ELSE
           PMFR(IR,IP)=-SW1*LOG(GR(IR,IP))
        ENDIF
     ENDDO
  ENDDO

  RETURN
END SUBROUTINE SETXTR
!-----------------------------------------------------------------------
SUBROUTINE INIT0(NPOINT,NPAIR,CSR)
  !     initialization of the function cs(r)
  use chm_kinds
  use number
  use rism
  implicit none
  real(chm_real) CSR(DVECT,*)
  INTEGER NPOINT,NPAIR

  INTEGER IP,IR
  DO IP=1,NPAIR
     DO IR=1,NPOINT
        CSR(IR,IP)=ZERO
     ENDDO
  ENDDO
  RETURN
END SUBROUTINE INIT0

#endif /* (rism_main)*/

