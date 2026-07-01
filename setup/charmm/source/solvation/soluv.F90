#if KEY_RISM==0 /*rism_main*/

SUBROUTINE NULL_SOLUV
  RETURN
end SUBROUTINE NULL_SOLUV

#else /* (rism_main)*/

SUBROUTINE SOLUV(NSITV,INTV,XVVK,XVV2K1,XVV2K2,RHO, &
     NSITU,NPUV,X,Y,Z,SEGID,INTU,INTUV,A,B,C, &
     USR,USR2,PHIK,GR,CSR,TSR,CSK,TSK,CS2R,WU, &
     CLOS,KBT,SW,SWI,SWF,DSW,NSW,IUNCSR,IUNGR,NDMAX,NCYCLE, &
     NPRINT,TOL,RMIX,QINIT,LQUSR,LQWR,LQTSR,LQBDR)

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

  INTEGER NSITV,INTV(DSITV,DSITV)
  real(chm_real) XVVK(DVECT,*),XVV2K1(DVECT,*),XVV2K2(DVECT,*),RHO(*)

  !     Structural and Site-site potential parameters
  INTEGER NSITU,NPUV
  real(chm_real) X(*),Y(*),Z(*)
  CHARACTER(len=*) SEGID(*)
  INTEGER INTU(DSITU,DSITU),INTUV(DSITU,DSITV)
  real(chm_real) A(*),B(*),C(*)

  !     Inter and intramolecular matrix and index pointer arrays
  real(chm_real) USR(DVECT,*),USR2(DVECT,*),PHIK(DVECT,*),GR(DVECT,*), &
       CSR(DVECT,*),TSR(DVECT,*),CSK(DVECT,*),TSK(DVECT,*), &
       CS2R(DVECT,*),WU(DVECT,*)

  !     Physical control parameters
  CHARACTER(len=3) CLOS
  real(chm_real) KBT

  !     Cycle control parameters, switches mixing and tolerance
  INTEGER IUNCSR,IUNGR,NSW(*),NPRINT,NCYCLE
  real(chm_real) SW(*),SWI(*),SWF(*),DSW(*),RMIX,TOL
  LOGICAL QINIT,CONVRG,LQUSR,LQWR,LQTSR,LQBDR

  !     Local Variables
  real(chm_real) DMAX
  INTEGER NDMAX

  INTEGER IK,I,J,ISW1,ISW2,ISW3,IP,IR,ICYCLE

  !     Construct the constant matrices for CYCLE3 (10)
  !     ------------------------------------------
  CALL MKOMEG(X,Y,Z,SEGID,NSITU,INTU,DSITU,WU,WU,'NOINV')

  !     Make the modified solvent-solvent susceptibility matrices
  DO IK=NFIRST,NPOINT
     DO I=1,NSITV
        DO J=1,I
           XVV2K1(IK,INTV(I,J))=XVVK(IK,INTV(I,J))/RHO(I)
           XVV2K2(IK,INTV(I,J))=XVVK(IK,INTV(I,J))/RHO(J)
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
           DO IP=1,NPUV
              DO IR=NFIRST,NPOINT
                 USR2(IR,IP)=SW(1)*USR(IR,IP)
              ENDDO
           ENDDO
        ELSE
           CALL MKUSR(NPUV,A,B,SW(1),SW(2),KBT,USR2)
        ENDIF

        DO ISW3=1,NSW(3)
           SW(3)=SWI(3)+(ISW3-1)*DSW(3)

           !     Make the Coulomb phik bond in k-space
           CALL MKPHIK(NPUV,C,ONE,SW(1),SW(3),KBT,PHIK)

           WRITE(OUTU,98) (SW(I),I=1,4)
98         FORMAT(1X,'switch =',4F9.4)

           !     Call the elementary cycle
           !     -------------------------
           IF(QINIT)THEN
              WRITE(OUTU,99)
99            FORMAT(' The iteration cycle is initialized')
              CALL INIT0(NPOINT,NPUV,CSR)
              CALL SOLUV2(NSITU,INTU,NPUV,INTUV,NSITV,INTV,XVV2K1,XVV2K2, &
                   WU,USR2,PHIK,CSR,TSR,GR,CSK,TSK,CS2R, &
                   ICYCLE,2,1,CLOS,TOL,ZERO,CONVRG,NDMAX,DMAX)

              QINIT=.FALSE.
           ENDIF
           CALL SOLUV2(NSITU,INTU,NPUV,INTUV,NSITV,INTV,XVV2K1,XVV2K2, &
                WU,USR2,PHIK,CSR,TSR,GR,CSK,TSK,CS2R, &
                ICYCLE,NCYCLE,NPRINT,CLOS,TOL,RMIX,CONVRG,NDMAX,DMAX)

           !     Make the radial distribution function
           !     -------------------------------------

           CALL MKGRUV(CLOS,GR,TSR,USR2,NPUV)
           WRITE(OUTU,102)
102        FORMAT(6X,'The radial distribution function is generated')

           IF(IUNCSR.NE.0)THEN
              !     append to the save file unit=iuncsr
              CALL WRTDAT(IUNCSR,CSR,NPOINT,NPUV,SW)
           ELSEIF(IUNGR.NE.0)THEN
              !     append to the save file unit=iungr
              CALL WRTDAT(IUNGR,GR,NPOINT,NPUV,SW)
           ENDIF

           IF(CONVRG)THEN
              WRITE(OUTU,100) ICYCLE,DMAX
100           FORMAT(6X,'convergence reached at cycle ',I8,' dmax=',F10.5)
           ELSE
              WRITE(OUTU,101) DMAX
101           FORMAT(/,6X,'* Warning *', &
                   ' loop ended before tolerance was satisfied, dmax',F10.5)
              ! return if not converge
              !
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
     DO IP=1,NPUV
        DO IR=NFIRST,NPOINT
           USR(IR,IP)=TSR(IR,IP)
        ENDDO
     ENDDO
     WRITE(OUTU,103)
103  FORMAT(6X,'The cavity potential h(r)-c(r) is stored in us(r)')
  ELSEIF(LQWR)THEN
     CALL MKWRUV(CLOS,KBT,TSR,USR2,NPUV)
     DO IP=1,NPUV
        DO IR=NFIRST,NPOINT
           USR(IR,IP)=USR2(IR,IP)
        ENDDO
     ENDDO
     WRITE(OUTU,104)
104  FORMAT(6X,'The Potential of Mean Force is stored in us(r)')
  ENDIF
  WRITE(OUTU,*)

  RETURN
END SUBROUTINE SOLUV

!====================================================================

SUBROUTINE SOLUV2(NSITU,INTU,NPUV,INTUV,NSITV,INTV,XVV2K1,XVV2K2, &
     WU,USR,PHIK,CSR,TSR,GR,CSK,TSK,CS2R, &
     ICYCLE,NCYCLE,NPRINT,CLOS,TOL,RMIX,CONVRG,NDMAX,DMAX)

  !     This subroutine performs the elementary cycle "ncycle" times
  !     or until the "tol" convergence is reached

  use chm_kinds
  use number
  use rism
  use fft
  use stream
  implicit none

  INTEGER NSITU,INTU(DSITU,DSITU),NPUV,INTUV(DSITU,DSITV), &
       NSITV,INTV(DSITV,DSITV)
  real(chm_real) XVV2K1(DVECT,*),XVV2K2(DVECT,*),WU(DVECT,*),USR(DVECT,*), &
       PHIK(DVECT,*),CSR(DVECT,*),TSR(DVECT,*),GR(DVECT,*), &
       CSK(DVECT,*),TSK(DVECT,*),CS2R(DVECT,*)

  !     Control parameters
  INTEGER ICYCLE,NCYCLE,NPRINT
  CHARACTER(len=3) CLOS
  real(chm_real) TOL,RMIX
  LOGICAL CONVRG
  real(chm_real)  DMAX,DMAX2,DIFF
  INTEGER NDMAX

  INTEGER IP,IK,I,J,I2,J2,IR,IPRINT

  !     Elementary cycle to solve the integral equations  (10)
  !     ================================================
  loop10:DO ICYCLE=1,NCYCLE


     !     Fourier transform the csr to get csk  (20)
     !     ------------------------------------
     IF(QLOG)THEN
        DO IP=1,NPUV
           CALL LOGFFT(CSR(1,IP),CSK(1,IP),TP32)
        ENDDO
     ELSE
        DO IP=1,NPUV
           CALL LINFFT(CSR(1,IP),CSK(1,IP),DVR)
        ENDDO
     ENDIF


     !     Calculate the tsk from the csk  (30)
     !     ------------------------------
     DO IK=NFIRST,NPOINT
        DO I=1,NSITU
           DO J=1,NSITV
              TSK(IK,INTUV(I,J))=-CSK(IK,INTUV(I,J))
              DO I2=1,NSITU
                 DO J2=1,J
                    TSK(IK,INTUV(I,J))=TSK(IK,INTUV(I,J))+ &
                         WU(IK,INTU(I,I2))* &
                         (CSK(IK,INTUV(I2,J2))+PHIK(IK,INTUV(I2,J2)))* &
                         XVV2K1(IK,INTV(J2,J))
                 ENDDO
                 DO J2=J+1,NSITV
                    TSK(IK,INTUV(I,J))=TSK(IK,INTUV(I,J))+ &
                         WU(IK,INTU(I,I2))* &
                         (CSK(IK,INTUV(I2,J2))+PHIK(IK,INTUV(I2,J2)))* &
                         XVV2K2(IK,INTV(J2,J))
                 ENDDO
              ENDDO
           ENDDO
        ENDDO
     ENDDO

     !     Fourier transform the tsk to get the tsr (40)
     !     ----------------------------------------
     IF(QLOG)THEN
        DO IP=1,NPUV
           CALL LOGFFT(TSK(1,IP),TSR(1,IP),TPM32)
        ENDDO
     ELSE
        DO IP=1,NPUV
           CALL LINFFT(TSK(1,IP),TSR(1,IP),DVK)
        ENDDO
     ENDIF


     !     Apply the closure to get a new CS2r  (50)
     !     -----------------------------------
     CALL CLOSUREUV(CLOS,USR,CS2R,TSR,NPUV,NPOINT)

     !     Comparison and mixing of CS2r and CSr (60)
     !     -------------------------------------
     DMAX=ZERO
     DO IR=1,NPOINT
        DO IP=1,NPUV
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
  enddo loop10
  ! total number of cycles executed
  ICYCLE=NCYCLE
1000 CONTINUE
  RETURN
END SUBROUTINE SOLUV2

#endif /* (rism_main)*/

