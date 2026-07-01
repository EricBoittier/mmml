#if KEY_RISM==0 /*rism_main*/
SUBROUTINE NULL_DERIV
  RETURN
end SUBROUTINE NULL_DERIV

#else /*  (rism_main)*/

SUBROUTINE DERIV0(COMLYN,COMLEN)
  !-----------------------------------------------------------------------
  !     This subroutine parses the control command for the iteration and
  !     calls the elementary cycle for the iteration procedure that
  !     calculates the solvent density response to a solute

  use stream
  use struc
  use distri
  use rism_control
  use fft
  use dimens_fcm
  use string
  use memory
  use number
  !
  use rism
  use chm_kinds
  implicit none

  !     Command parser
  real(chm_real),allocatable,dimension(:,:) :: WU
  real(chm_real),allocatable,dimension(:,:) :: XVV2K
  real(chm_real),allocatable,dimension(:,:) :: HWIHK
  real(chm_real),allocatable,dimension(:,:) :: DCSK
  real(chm_real),allocatable,dimension(:,:) :: DTSR
  real(chm_real),allocatable,dimension(:,:) :: DTSK
  real(chm_real),allocatable,dimension(:,:) :: DCS2R
  CHARACTER COMLYN*(*)
  INTEGER   COMLEN
  ! CHARMM declarations
  !

  !     Local variables
  !     ---------------

  !     Cycle iteration control variables
  real(chm_real)  TOL,RMIX
  INTEGER NCYCLE,NPRINT,IK,ICYCLE
  LOGICAL CONVRG

  !     Solute label
  INTEGER UA,UB,IUA,IUB,UUAB,IOFF,I,J

  !     Local Heap management
  !!      INTEGER IOFF1,IOFF2

  !     Parsing the command
  !     ===================

  !     Control parameters
  NCYCLE=GTRMI(COMLYN,COMLEN,'NCYCLE',1)
  NPRINT=GTRMI(COMLYN,COMLEN,'NPRINT',NCYCLE)
  RMIX=GTRMF(COMLYN,COMLEN,'RMIX',THIRD)
  TOL=GTRMF(COMLYN,COMLEN,'TOL',PT01)

  !     Solution to solve:
  !     -----------------

  IUA=GTRMI(COMLYN,COMLEN,'SOLU',1)
  IOFF=PUV(IUA)
  UA=1+DSITV+DSITU*(IUA-1)
  call chmalloc('deriv.src','DERIV0','WU',DVECT,NPU(IUA),crl=WU)
  call chmalloc('deriv.src','DERIV0','XVV2K',DVECT,NPV,crl=XVV2K)
  call chmalloc('deriv.src','DERIV0','HWIHK',DVECT,NPVV,crl=HWIHK)
  call chmalloc('deriv.src','DERIV0','DCSK',DVECT,NPVV,crl=DCSK)
  call chmalloc('deriv.src','DERIV0','DTSR',DVECT,NPVV,crl=DTSR)
  call chmalloc('deriv.src','DERIV0','DTSK',DVECT,NPVV,crl=DTSK)
  call chmalloc('deriv.src','DERIV0','DCS2R',DVECT,NPVV,crl=DCS2R)


  !     Closures:
  IF(CHECQUE(COMLYN,'HNC'))THEN
     CLOS='HNC'
  ELSEIF(CHECQUE(COMLYN,'PY'))THEN
     CLOS='PY '
  ELSEIF(CHECQUE(COMLYN,'PY2'))THEN
     CLOS='PY2'
  ENDIF


  !     Write out all the options in use
  WRITE(OUTU,100) CLOS,NCYCLE,NPRINT,RMIX,TOL

100 FORMAT(/,' SOLVENT RESPONSE DERIVATIVE',/,1X, &
       'closure:    ',A3,5X,'ncycle',I9,5X,'nprint',I9,/,1X, &
       'rmix',F11.2,5X,'tol',F12.5)

  WRITE(OUTU,103) IUA
103 FORMAT(1X,'solute   #',I5)

  IF(CHECQUE(COMLYN,'INIT'))THEN
     WRITE(OUTU,99)
99   FORMAT(' The derivative cycle is initialized')
     !
     IPDCSR(:,NPRVV*(IUA-1)+1:NPRVV*(IUA-1)+NPVV) = zero        ! APH: potential bug here
     !
  ENDIF

  CALL DERIV1(NSITV,NPVV,INTV,INTVV, &
       IPGR,IPXVVK,XVV2K,KBT,RHO, &
       HWIHK,IPDGR(1,NPRVV*(IUA-1)+1),IPDCSR(1,NPRVV*(IUA-1)+1), &
       DCSK,DTSR,DTSK,DCS2R, &
       NSITU(IUA),INTU(1,1,IUA),INTUV(1,1,IUA),NPUV, &
       WU,IPCSK(1,IOFF),IPPHIK(1,IOFF), &
       X(UA),Y(UA),Z(UA),SEGMID(UA), &
       NCYCLE,NPRINT,TOL,RMIX,CONVRG,CLOS)


  call chmdealloc('deriv.src','DERIV0','WU',DVECT,NPU(IUA),crl=WU)
  call chmdealloc('deriv.src','DERIV0','XVV2K',DVECT,NPV,crl=XVV2K)
  call chmdealloc('deriv.src','DERIV0','HWIHK',DVECT,NPVV,crl=HWIHK)
  call chmdealloc('deriv.src','DERIV0','DCSK',DVECT,NPVV,crl=DCSK)
  call chmdealloc('deriv.src','DERIV0','DTSR',DVECT,NPVV,crl=DTSR)
  call chmdealloc('deriv.src','DERIV0','DTSK',DVECT,NPVV,crl=DTSK)
  call chmdealloc('deriv.src','DERIV0','DCS2R',DVECT,NPVV,crl=DCS2R)

  RETURN
END SUBROUTINE DERIV0

SUBROUTINE DERIV1(NSITV,NPVV,INTV,INTVV, &
     GR,XVVK,XVV2K,KBT,RHO, &
     HWIHK,DGR,DCSR,DCSK,DTSR,DTSK,DCS2R, &
     NSITU,INTU,INTUV,NPUV,WU,CSK,PHIK,X,Y,Z,SEGID, &
     NCYCLE,NPRINT,TOL,RMIX,CONVRG,CLOS)
  !-----------------------------------------------------------------------
  use fft
  use stream
  use number
  use rism
  use chm_kinds
  implicit none

  !     Solvent-solvent distribution
  INTEGER NSITV,NPVV,INTV(DSITV,DSITV),INTVV(DSITV,DSITV)
  real(chm_real) GR(DVECT,*),XVVK(DVECT,*),XVV2K(DVECT,*),KBT,RHO(*)
  real(chm_real) HWIHK(DVECT,*),DGR(DVECT,*),DCSR(DVECT,*),DCSK(DVECT,*), &
       DTSR(DVECT,*),DTSK(DVECT,*),DCS2R(DVECT,*)

  !     Solute-solvent distribution
  INTEGER NSITU,INTU(DSITU,DSITU),INTUV(DSITU,DSITV),NPUV
  real(chm_real) WU(DVECT,*),CSK(DVECT,*),PHIK(DVECT,*)
  real(chm_real) X(*),Y(*),Z(*)
  CHARACTER*(*) SEGID(*)

  !     Control variables
  CHARACTER CLOS*3
  INTEGER NCYCLE,NPRINT
  real(chm_real) TOL,RMIX
  real(chm_real) DMAX
  LOGICAL CONVRG

  INTEGER IK,I,J,I1,I2,I3,I4,ICYCLE

  !     Make the modified  solvent-solvent susceptibility matrix
  DO IK=NFIRST,NPOINT
     DO I=1,NSITV
        DO J=1,I
           XVV2K(IK,INTV(I,J))=XVVK(IK,INTV(I,J))/RHO(J)
        ENDDO
     ENDDO
  ENDDO

  !     Make the solute intramolecular distribution
  CALL MKOMEG(X,Y,Z,SEGID,NSITU,INTU,DSITU,WU,WU,'NOINV')

  !     Make the constant term
  !     Hvu * inv(Wu) * Huv = 1/rho Xvv Cvu Wu inv(Wu) Wu Cuv Xvv/rho
  !     notice:  the Coulomb phik bond in k-space is still in phik()
  DO IK=NFIRST,NPOINT
     DO I=1,NSITV
        DO J=1,I
           HWIHK(IK,INTVV(I,J))=ZERO
           DO I1=1,NSITV
              DO I2=1,NSITU
                 DO I3=1,NSITU
                    DO I4=1,NSITV
                       HWIHK(IK,INTVV(I,J))=HWIHK(IK,INTVV(I,J)) + &
                            XVV2K(IK,INTV(I1,I)) * &
                            (PHIK(IK,INTUV(I2,I1)) + CSK(IK,INTUV(I2,I1))) * &
                            WU(IK,INTU(I2,I3)) * &
                            (PHIK(IK,INTUV(I3,I4)) + CSK(IK,INTUV(I3,I4))) &
                            * XVV2K(IK,INTV(I4,J))
                    ENDDO
                 ENDDO
              ENDDO
           ENDDO
        ENDDO
     ENDDO
  ENDDO

  !     Call the elementary cycle
  !     -------------------------
  CALL DERIV2(NSITV,NPVV,INTV,INTVV,HWIHK,XVV2K, &
       DCSR,DCSK,DTSR,DTSK,DCS2R,GR,TOL, &
       RMIX,ICYCLE,NCYCLE,NPRINT,DMAX,CONVRG,CLOS)

  !     Make the radial distribution function
  !     -------------------------------------
  CALL MKDGR(DGR,GR,DTSR,NPVV,CLOS)

  IF(CONVRG)THEN
     WRITE(OUTU,100) ICYCLE,DMAX
100  FORMAT(6X,'convergence reached at cycle ',I8,' dmax=',F10.5)
  ELSE
     WRITE(OUTU,101) DMAX
101  FORMAT(/,6X,'* Warning *', &
          ' loop ended before tolerance was satisfied, dmax',F10.5)
     RETURN  !RETURN IF NOT CONVERGED
  ENDIF

  RETURN
END SUBROUTINE DERIV1

SUBROUTINE DERIV2(NSITV,NPVV,INTV,INTVV,HWIHK,XVV2K, &
     DCSR,DCSK,DTSR,DTSK,DCS2R,GR,TOL, &
     RMIX,ICYCLE,NCYCLE,NPRINT,DMAX,CONVRG,CLOS)
  !-----------------------------------------------------------------------
  !     This subroutine performs the elementary cycle "ncycle" times
  !     or until the "tol" convergence is reached

  use fft
  use stream
  use number
  use rism
  use chm_kinds
  implicit none

  INTEGER NSITV,NPVV,INTV(DSITV,DSITV),INTVV(DSITV,DSITV)
  real(chm_real) HWIHK(DVECT,*),XVV2K(DVECT,*),DCSR(DVECT,*), &
       DCSK(DVECT,*), DTSR(DVECT,*),DTSK(DVECT,*), &
       DCS2R(DVECT,*),GR(DVECT,*)

  !     Control parameters
  INTEGER ICYCLE,NCYCLE,NPRINT
  CHARACTER CLOS*3
  real(chm_real) TOL,RMIX
  LOGICAL CONVRG
  real(chm_real)  DMAX

  real(chm_real) DIFF
  INTEGER IP,IK,I,J,I2,J2,IR,IPRINT

  !     Elementary cycle to solve the integral equations  (10)
  !     ================================================
  loop10:DO ICYCLE=1,NCYCLE


     !     Fourier transform the csr to get dcsk  (20)
     !     -------------------------------------
     IF(QLOG)THEN
        DO IP=1,NPVV
           CALL LOGFFT(DCSR(1,IP),DCSK(1,IP),TP32)
        ENDDO
     ELSE
        DO IP=1,NPVV
           CALL LINFFT(DCSR(1,IP),DCSK(1,IP),DVR)
        ENDDO
     ENDIF


     !     Calculate the dtsk from the dcsk  (30)
     !     --------------------------------
     DO IK=NFIRST,NPOINT
        DO I=1,NSITV
           DO J=1,I
              DTSK(IK,INTVV(I,J))=HWIHK(IK,INTVV(I,J))-DCSK(IK,INTVV(I,J))
              DO I2=1,NSITV
                 DO J2=1,NSITV
                    DTSK(IK,INTVV(I,J))=DTSK(IK,INTVV(I,J))+ &
                         XVV2K(IK,INTV(I2,I))* &
                         DCSK(IK,INTVV(I2,J2))* &
                         XVV2K(IK,INTV(J2,J))
                 ENDDO
              ENDDO
           ENDDO
        ENDDO
     ENDDO

     !     Fourier transform the dtsk to get the dtsr (40)
     !     ------------------------------------------
     IF(QLOG)THEN
        DO IP=1,NPVV
           CALL LOGFFT(DTSK(1,IP),DTSR(1,IP),TPM32)
        ENDDO
     ELSE
        DO IP=1,NPVV
           CALL LINFFT(DTSK(1,IP),DTSR(1,IP),DVK)
        ENDDO
     ENDIF


     !     Apply the closure to get a new CS2r  (50)
     !     -----------------------------------
     CALL DCLOS(DCS2R,DTSR,GR,NPVV,CLOS)

     !     Comparison and mixing of CS2r and CSr (60)
     !     -------------------------------------
     DMAX=ZERO
     DO IR=1,NPOINT
        DO IP=1,NPVV
           DIFF=ABS(DCS2R(IR,IP)-DCSR(IR,IP))
           IF(DIFF.GT.DMAX) DMAX=DIFF
           DCSR(IR,IP)=(ONE-RMIX)*DCS2R(IR,IP)+RMIX*DCSR(IR,IP)
        ENDDO
     ENDDO
     IF(MOD(ICYCLE,NPRINT).EQ.0)THEN
        WRITE(OUTU,100) ICYCLE,DMAX
        IPRINT=0
     ENDIF
100  FORMAT(6X,'cycle',I5,'  dmax',F10.5)
     !
     CONVRG=DMAX.LT.TOL
     ! exit the cycle loop when tol is reached
     IF(CONVRG) GOTO 1000
     ! continue the icycle loop
  enddo loop10
  ! set icycle equal to the total number of cycles executed
  ICYCLE=NCYCLE

1000 CONTINUE
  RETURN

END SUBROUTINE DERIV2

SUBROUTINE DCLOS(DCSR,DTSR,GR,NPAIR,CLOS)
  !-----------------------------------------------------------------------
  use fft
  use rism
  use chm_kinds
  implicit none
  CHARACTER CLOS*3
  real(chm_real) DCSR(DVECT,*),DTSR(DVECT,*),GR(DVECT,*)
  INTEGER NPAIR

  INTEGER IPRINT,IP,IR

  IF(CLOS.EQ.'HNC')THEN
     DO IP=1,NPAIR
        DO IR=NFIRST,NPOINT
           DCSR(IR,IP)=GR(IR,IP)*DTSR(IR,IP)-DTSR(IR,IP)
        ENDDO
     ENDDO
  ENDIF
  !
  RETURN
END SUBROUTINE DCLOS

SUBROUTINE MKDGR(DGR,GR,DTSR,NPAIR,CLOS)
  !-----------------------------------------------------------------------
  use fft
  use rism
  use chm_kinds
  implicit none
  CHARACTER CLOS*3
  real(chm_real) DGR(DVECT,*),GR(DVECT,*),DTSR(DVECT,*)
  INTEGER NPAIR

  INTEGER IP,IR

  IF(CLOS.EQ.'HNC')THEN
     DO IP=1,NPAIR
        DO IR=NFIRST,NPOINT
           DGR(IR,IP)=GR(IR,IP)*DTSR(IR,IP)
        ENDDO
     ENDDO
  ENDIF
  RETURN
END SUBROUTINE MKDGR

#endif /* (rism_main)*/

