#if KEY_RISM==0 /*rism_main*/

SUBROUTINE NULL_SOLUU
  RETURN
end SUBROUTINE NULL_SOLUU


#else /* (rism_main)*/

SUBROUTINE SOLUU(NSITV,INTV,XVVK,RHO,CDIE,CDIE2, &
     NSITA,NPAV,XA,YA,ZA,SEGIDA, &
     INTA,INTAV,CSAK,PHIAK,WA, &
     NSITB,NPBV,XB,YB,ZB,SEGIDB, &
     INTB,INTBV,CSBK,PHIBK,WB, &
     NPAB,INTAB,A,B,C, &
     USR,USR2,PHIK,GR,CSR, &
     TSR,CSK,TSK, &
     CS2R,WCXCW,PHIABK,PHIABR, &
     CLOS,KBT,SW,SWI,SWF,DSW,NSW,IUNCSR,IUNGR, &
     NDMAX,NCYCLE,NPRINT,TOL,RMIX,QINIT,LQUSR,LQWR, &
     LQTSR,LQBDR,LQCAV)

  !     This subroutine sets up the site-site potential and the long
  !     range phik bond.  It includes loops on
  !     the 4 switching parameters
  !     on the potential sw(1)
  !     on the sigma     sw(2)
  !     on the charges   sw(3)

  use chm_kinds
  use number
  use rism
  use fft
  use stream
  implicit none

  INTEGER NSITV,INTV(DSITV,DSITV),NSITA,NPAV,NSITB,NPBV,NPAB
  INTEGER INTA(DSITU,DSITU),INTAV(DSITU,DSITV),INTB(DSITU,DSITU), &
       INTBV(DSITU,DSITU),INTAB(DSITU,DSITU)
  real(chm_real) XVVK(DVECT,*),RHO(*),CDIE,CDIE2
  real(chm_real) CSAK(DVECT,*),PHIAK(DVECT,*),WA(DVECT,*)
  real(chm_real) XA(*),YA(*),ZA(*),XB(*),YB(*),ZB(*)
  real(chm_real) CSBK(DVECT,*),PHIBK(DVECT,*),WB(DVECT,*)
  real(chm_real) A(*),B(*),C(*),USR(DVECT,*),USR2(DVECT,*),PHIK(DVECT,*), &
       GR(DVECT,*),CSR(DVECT,*),TSR(DVECT,*), &
       CSK(DVECT,*),TSK(DVECT,*),CS2R(DVECT,*), &
       WCXCW(DVECT,*),PHIABK(*),PHIABR(*)
  CHARACTER(len=*) SEGIDA(*),SEGIDB(*)

  !     Physical control parameters
  CHARACTER(len=3) CLOS
  real(chm_real) KBT

  !     Cycle control parameters, switches mixing and tolerance
  INTEGER IUNCSR,IUNGR,NSW(*),NPRINT,NDMAX,NCYCLE
  real(chm_real) SW(*),SWI(*),SWF(*),DSW(*),RMIX,TOL
  LOGICAL QINIT,CONVRG,LQUSR,LQWR,LQTSR,LQBDR,LQCAV

  !     Local Variables
  real(chm_real) DMAX
  real(chm_real) CAB
  INTEGER I,J,ISW1,ISW2,IP,IR,ISW3,ICYCLE

  !     Construct the constant matrices for CYCLE3 (10)
  !     ------------------------------------------
  CALL MKOMEG(XA,YA,ZA,SEGIDA,NSITA,INTA,DSITU,WA,WA,'NOINV')
  CALL MKOMEG(XB,YB,ZB,SEGIDB,NSITB,INTB,DSITU,WB,WB,'NOINV')

  !     Make the modified the cavity part W(a)*C(av)*X(vv)*C(vb)*W(b)
  CALL MKWCXCW(NSITV,INTV,XVVK, &
       NSITA,INTA,INTAV,WA,CSAK,PHIAK, &
       NSITB,INTB,INTBV,WB,CSBK,PHIBK, &
       INTAB,WCXCW)

  !     Calculate the total Coulomb interaction between the 2 solutes
  CAB=ZERO
  DO I=1,NSITA
     DO J=1,NSITB
        CAB=CAB+C(INTAB(I,J))
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
           DO IP=1,NPAB
              DO IR=NFIRST,NPOINT
                 USR2(IR,IP)=SW(1)*USR(IR,IP)
              ENDDO
           ENDDO
        ELSE
           CALL MKUSR(NPAB,A,B,SW(1),SW(2),KBT,USR2)
        ENDIF

        DO ISW3=1,NSW(3)
           SW(3)=SWI(3)+(ISW3-1)*DSW(3)

           !     Make the Coulomb phi bond in k-space
           CALL MKPHIK(NPAB,C,ONE,SW(1),SW(3),KBT,PHIK)
           CALL MKPHIK(1,CAB,ONE,SW(1),SW(3),KBT,PHIABK)
           CALL MKPHIR(1,CAB,ONE,SW(1),SW(3),KBT,PHIABR)
           IF(CAB.NE.ZERO)THEN
              PHIABK(nfirst:npoint)=PHIABK(nfirst:npoint)/CDIE
              PHIABR(nfirst:npoint)=PHIABR(nfirst:npoint)/CDIE2
           ENDIF

           WRITE(OUTU,98) (SW(I),I=1,4)
98         FORMAT(1X,'switch =',4F9.4)

           !     Call the elementary cycle
           !     -------------------------
           IF(QINIT)THEN
              WRITE(OUTU,99)
99            FORMAT(' The iteration cycle is initialized')
              CALL INIT0(NPOINT,NPAB,CSR)
              CALL SOLUU2(NSITA,WA,INTA,NSITB,WB,INTB, &
                   NPAB,INTAB,WCXCW,CAB,PHIABK,PHIABR, &
                   USR2,PHIK,CSR,TSR,CSK,TSK,CS2R, &
                   ICYCLE,2,NPRINT,CLOS,TOL,ZERO,CONVRG,NDMAX,DMAX)
              QINIT=.FALSE.
           ENDIF
           CALL SOLUU2(NSITA,WA,INTA,NSITB,WB,INTB, &
                NPAB,INTAB,WCXCW,CAB,PHIABK,PHIABR, &
                USR2,PHIK,CSR,TSR,CSK,TSK,CS2R, &
                ICYCLE,NCYCLE,NPRINT,CLOS,TOL,RMIX,CONVRG,NDMAX,DMAX)


           !     Make the radial distribution function
           !     -------------------------------------
           CALL MKGRUV(CLOS,GR,TSR,USR2,NPAB)
           WRITE(OUTU,102)
102        FORMAT(6X,'The radial distribution function is generated')

           IF(IUNCSR.NE.0)THEN
              !     append to the save file unit=iuncsr
              CALL WRTDAT(IUNCSR,CSR,NPOINT,NPAB,SW)
           ELSEIF(IUNGR.NE.0)THEN
              !     append to the save file unit=iungr
              CALL WRTDAT(IUNGR,GR,NPOINT,NPAB,SW)
           ENDIF

           IF(CONVRG)THEN
              WRITE(OUTU,100) ICYCLE,DMAX
100           FORMAT(6X,'convergence reached at cycle ',I8,' dmax=',F10.5)
           ELSE
              WRITE(OUTU,101) DMAX
101           FORMAT(/,6X,'* Warning *', &
                   ' loop ended before tolerance was satisfied, dmax',F10.5)
              ! return if not converge
              GOTO 1000
           ENDIF
           ! end of the switching loop
        ENDDO
     ENDDO
  ENDDO

1000 CONTINUE



  !     Store output functions in us(r)
  !     -------------------------------
  IF(LQTSR)THEN
     DO IP=1,NPAB
        DO IR=NFIRST,NPOINT
           USR(IR,IP)=TSR(IR,IP)
        ENDDO
     ENDDO
     WRITE(OUTU,103)
103  FORMAT(6X,'The cavity potential h(r)-cs(r) is stored in us(r)')
  ELSEIF(LQWR)THEN
     CALL MKWRUV(CLOS,KBT,TSR,USR2,NPAB)
     DO IP=1,NPAB
        DO IR=NFIRST,NPOINT
           USR(IR,IP)=USR2(IR,IP)
        ENDDO
     ENDDO
     WRITE(OUTU,104)
104  FORMAT(6X,'The Potential of Mean Force is stored in us(r)')
  ELSEIF(LQCAV)THEN
     DO IP=1,NPAB
        DO IR=NFIRST,NPOINT
           USR(IR,IP)=-KBT*(TSR(IR,IP)-PHIABR(IR))
        ENDDO
     ENDDO
     WRITE(OUTU,105)
105  FORMAT(6X,'The short range cavity potential is stored in us(r)')

  ENDIF

  RETURN
END SUBROUTINE SOLUU

!===================================================================

SUBROUTINE SOLUU2(NSITA,WA,INTA,NSITB,WB,INTB, &
     NPAB,INTAB,WCXCW,CAB,PHIABK,PHIABR, &
     USR,PHIK,CSR,TSR,CSK,TSK,CS2R, &
     ICYCLE,NCYCLE,NPRINT,CLOS,TOL,RMIX,CONVRG,NDMAX,DMAX)

  !     This subroutine performs the elementary cycle "ncycle" times
  !     or until the "tol" convergence is reached

  use chm_kinds
  use number
  use rism
  use fft
  use stream
  implicit none

  INTEGER NSITA,INTA(DSITU,DSITU),NSITB,INTB(DSITU,DSITU)
  real(chm_real) WA(DVECT,*),WB(DVECT,*)
  INTEGER NPAB,INTAB(DSITU,DSITV)
  real(chm_real) WCXCW(DVECT,*),CAB,PHIABK(*),PHIABR(*),USR(DVECT,*), &
       PHIK(DVECT,*),CSR(DVECT,*),TSR(DVECT,*),CSK(DVECT,*), &
       TSK(DVECT,*),CS2R(DVECT,*)

  !     Control parameters
  INTEGER ICYCLE,NCYCLE,NPRINT
  CHARACTER(len=3) CLOS
  real(chm_real) TOL,RMIX
  LOGICAL CONVRG
  real(chm_real)  DMAX
  INTEGER NDMAX
  real(chm_real) DMAX2,DIFF
  INTEGER IP,IK,I,J,I2,J2,IR,IPRINT

  !     Elementary cycle to solve the integral equations  (10)
  !     ================================================
  loop10:DO ICYCLE=1,NCYCLE

     !     Fourier transform the csr to get csk  (20)
     !     ------------------------------------
     IF(QLOG)THEN
        DO IP=1,NPAB
           CALL LOGFFT(CSR(1,IP),CSK(1,IP),TP32)
        ENDDO
     ELSE
        DO IP=1,NPAB
           CALL LINFFT(CSR(1,IP),CSK(1,IP),DVR)
        ENDDO
     ENDIF


     !     Calculate the tsk from the csk  (30)
     !     ------------------------------
     DO IK=NFIRST,NPOINT
        DO I=1,NSITA
           DO J=1,NSITB
              TSK(IK,INTAB(I,J))=WCXCW(IK,INTAB(I,J))-CSK(IK,INTAB(I,J)) &
                   -PHIABK(IK)
              DO I2=1,NSITA
                 DO J2=1,NSITB
                    TSK(IK,INTAB(I,J))=TSK(IK,INTAB(I,J))+ &
                         WA(IK,INTA(I,I2))* &
                         (CSK(IK,INTAB(I2,J2))+PHIK(IK,INTAB(I2,J2)))* &
                         WB(IK,INTB(J2,J))
                 ENDDO
              ENDDO
           ENDDO
        ENDDO
     ENDDO

     !     Fourier transform the tsk to get the tsr (40)
     !     ----------------------------------------
     IF(QLOG)THEN
        DO IP=1,NPAB
           CALL LOGFFT(TSK(1,IP),TSR(1,IP),TPM32)
        ENDDO
     ELSE
        DO IP=1,NPAB
           CALL LINFFT(TSK(1,IP),TSR(1,IP),DVK)
        ENDDO
     ENDIF

     IF(CAB.NE.ZERO)THEN
        DO IP=1,NPAB
           DO IR=NFIRST,NPOINT
              TSR(IR,IP)=TSR(IR,IP)+PHIABR(IR)
           ENDDO
        ENDDO
     ENDIF


     !     Apply the closure to get a new CS2r  (50)
     !     -----------------------------------
     CALL CLOSUREUV(CLOS,USR,CS2R,TSR,NPAB,NPOINT)

     !     Comparison and mixing of CS2r and CSr (60)
     !     -------------------------------------
     DMAX=ZERO
     DO IR=1,NPOINT
        DO IP=1,NPAB
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
     ! Exit the cycle loop when tol is reached
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
     ! icycle loop
     !
  enddo loop10
  ! total number of cycles executed
  ICYCLE=NCYCLE

1000 CONTINUE
  RETURN

END SUBROUTINE SOLUU2

!=============================================================================

SUBROUTINE MKWCXCW(NSITV,INTV,XVVK, &
     NSITA,INTA,INTAV,WA,CSAK,PHIAK, &
     NSITB,INTB,INTBV,WB,CSBK,PHIBK, &
     INTAB,WCXCW)
  !     This subroutine makes the solute-solute cavity part used in
  !     U(a)U(b) calculations.
  !     WCXCW = W(a)*C(a,v)*X(v,v)*C(v,b)*W(b)
  !
  use chm_kinds
  use number
  use rism
  use fft
  implicit none

  INTEGER NSITV,INTV(DSITV,DSITV)
  INTEGER NSITA,INTA(DSITU,DSITU),INTAV(DSITU,DSITV), &
       NSITB,INTB(DSITU,DSITU),INTBV(DSITU,DSITV)
  INTEGER NPAB,INTAB(DSITU,DSITU)
  real(chm_real)  XVVK(DVECT,*)
  real(chm_real)  WA(DVECT,*),CSAK(DVECT,*),PHIAK(DVECT,*), &
       WB(DVECT,*),CSBK(DVECT,*),PHIBK(DVECT,*)
  real(chm_real)  WCXCW(DVECT,*)

  INTEGER IK,I,J,I1,I2,I3,I4

  DO IK=NFIRST,NPOINT
     DO I=1,NSITA
        DO J=1,NSITB
           WCXCW(IK,INTAB(I,J))=ZERO
           DO I1=1,NSITA
              DO I2=1,NSITV
                 DO I3=1,NSITV
                    DO I4=1,NSITB
                       WCXCW(IK,INTAB(I,J))=WCXCW(IK,INTAB(I,J))+ &
                            WA(IK,INTA(I,I1))* &
                            ((CSAK(IK,INTAV(I1,I2))+PHIAK(IK,INTAV(I1,I2)))* &
                            XVVK(IK,INTV(I2,I3))* &
                            (CSBK(IK,INTBV(I4,I3))+PHIBK(IK,INTBV(I4,I3))))* &
                            WB(IK,INTB(I4,J))
                    ENDDO
                 ENDDO
              ENDDO
           ENDDO
        ENDDO
     ENDDO
  ENDDO
  !
  RETURN
END SUBROUTINE MKWCXCW

#endif /* (rism_main)*/


