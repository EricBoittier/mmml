#if KEY_RISM==0 /*rism_main*/
SUBROUTINE NULL_IN
  RETURN
end SUBROUTINE NULL_IN

#else /*  (rism_main)*/
SUBROUTINE INITFFT(COMLYN,COMLEN)
  !-----------------------------------------------------------------------
  !     This subroutine initializes the necessary arrays for the FFT
  !     used to calculate the convolution equations
  use chm_kinds
  use number
  use string
  use rism
  use fft
  use stream
  use consta
  implicit none
  CHARACTER COMLYN*(*)
  INTEGER   COMLEN
  !
  INTEGER,PARAMETER :: NTERM=20

  INTEGER I,M
  real(chm_real)  DT,TI,RI,THETA,DELTA


  !---------------------------------------------------------------------------
  !     GENERAL FFT INITIALIZATION SETUP
  !     --------------------------------

  QLOG=CHECQUE(COMLYN,'LOGFFT').OR.QLOG
  QLOG=.NOT.CHECQUE(COMLYN,'FFT')
  NPOINT=GTRMI(COMLYN,COMLEN,'NPOINT',NPOINT)
  IF(NPOINT.GT.DVECT) THEN
     CALL WRNDIE(-3,'<FFTINIT>','NPOINT bigger than '// &
          'the maximum array size DVECT in rism.f90')
  ENDIF
  DR=GTRMF(COMLYN,COMLEN,'DR',DR)
  RMIN=GTRMF(COMLYN,COMLEN,'RMIN',RMIN)
  ! 
  !     2**MPWR2 = 2*NPOINT (Used by the IMSLD routine FFT2C in LOGFFT)
  !     We use 2*NPOINT in the FFT because NPOINT zeros are appended to
  !     stabilize the convolution.
  !
  !...##IF SUN64
  ! As NPOINT is INTERGER*8, the compiler thinks that 2.0*NPOINT might
  ! not fit in REAL*4, so the result is real(chm_real). The change from ALOG to
  ! DLOG should not make any difference, as MPWR2 is INTEGER. - BC
  MPWR2=INT(RSMALL+LOG(TWO*NPOINT)/LOG(TWO))
  !...##ELSE
  !      MPWR2=INT(RSMALL+ALOG(2.0*NPOINT)/ALOG(2.0))
  !...##ENDIF
  !--------------------------------------------------------------------
  !     LOG SCALE FFT INITIALIZATION
  !     ----------------------------

  IF(QLOG)THEN
     !     first point to put in the r() and rk() vectors
     NFIRST=1
     !
     !     (2*PI)**3/2
     TP32=(TWO*PI)**(ONEPT5)
     !
     !     (2*PI)**-3/2
     TPM32=ONE/TP32
     !
     !     there are 2*npoint in the Fourier transform
     !
     DT=PI/(NPOINT*DR)
     !
     !     General loop to initialize all the necessary arrays
     !
     loop1:DO I=1,NPOINT
        !     Generate the scale in r(i) and r(i)**3/2
        !
        !     First the real space vector
        R(I)=EXP(RMIN+DR*(I-1))
        !
        !     Then the fourier space vector
        RK(I)=R(I)
        RK32(I)=EXP(DR*(I-1)*ONEPT5)
        RKM32(I)=EXP(-DR*(I-1)*ONEPT5)

        !     Generate the complex function  M(t)  -->  CM(i)
        TI=(I-1)*DT
        DELTA=ATAN(TWO*TI/(TWO*NTERM+ONE))
        RI=SQRT((NTERM+HALF)**2+TI**2)
        THETA=-ATAN(TANH(PI*TI*HALF))
        DO M=1,NTERM
           THETA=THETA+ATAN(TWO*TI/(TWO*M-ONE))
        enddo
        THETA=THETA &
             +TI*(ONE-LOG(RI)) &
             -NTERM*DELTA &
             +SIN(DELTA)/(TWELVE*RI) &
             -SIN(THREE*DELTA)/(THR6TY*RI**3) &
             +SIN(FIVE*DELTA)/(ONE8TY*FIVE*RI**5) &
             +TWO*RMIN*TI
        !     final COMPLEX M(t) --> CM(t) for storage:
        CM(I)=EXP(DCMPLX(ZERO,ONE)*THETA)
        !
        !     Combine all constant factors with M(T)
        !
        CM(I)=(DR*DT/PI)*CM(I)
        !     Calculate the truncation error TRNC
        !     for the integral from  r=-infinity to r=rmin
        TRNC(I)=ONE/(DR*DCMPLX(ONEPT5,TI))
     enddo loop1


     !---------------------------------------------------------------------------
     !     ORDINARY SCALE FFT INITIALIZATION
     !     ---------------------------------

  ELSE
     !
     !     Define the first point to put in the R() and RK() vectors
     !
     NFIRST=2
     !     Calculate r-space and k-space increments and volume elements
     DVR=DR**3

     !     NPOINT zeros are appended at the end to stabilize the convolution
     !     Thus, the total number of points used in FFT is 2*NPOINT
     !
     DK=PI/(NPOINT*DR)
     DVK=(ONE/(2*NPOINT*DR))**3
     DO I=1,NPOINT
        R(I)=DR*(I-1)
        RK(I)=DK*(I-1)
     enddo
  ENDIF

  RETURN
END SUBROUTINE INITFFT


!========================================================================

SUBROUTINE LINFFT(F1,F2,DV)
  !-----------------------------------------------------------------------
  !     This subroutine performs the 3-d Fourier transform of a
  !     radially symmetric function F1(N) and puts the answer in F2(N)
  !     the subroutine appends N zeros at the end of the vector
  !     to make the convolution transform more accurate.
  !
  !     IF NN is the # of points used for the FFT, IMSLD requires
  !     arrays with sizes:
  !                                      Example: NN=512
  !                                      ---------------
  !
  !     real(chm_real) WK(6*NN/2+150)                     | 1686    |
  !     real(chm_real) F1(NN),F2(NN),ST(NN/2+1),CT(NN/2+1)| 512,257 |
  !     complex*16 CWK(NN/2+1)                    | 257     |
  !     integer IWK(M2PWR2)                       | 9       |
  !     with 2**M2PWR2=NN if NN is power of 2, otherwise
  !     M2PWR2=6*NN/2+150
  !     Here we pass the maximum possible size 6*DVECT+150
  !
  !     These arrays are needed by the IMSLD routine; for more
  !     information consult the IMSLD library manual
  !
  use chm_kinds
  use number
  use rism
  use fft
  implicit none
  real(chm_real) DV
  INTEGER I
  integer,PARAMETER :: NN=2*DVECT
  real(chm_real) F1(NN),F2(NN)

  real(chm_real) A(NN),ST(NN/2+1),CT(NN/2+1),WK(6*DVECT+150)
  INTEGER IWK(6*DVECT+150)
  COMPLEX*16 CWK(NN/2+1)
  !
  !     Make the radial function to transform
  DO I=2,NPOINT
     A(I)=(I-1)*F1(I)
     A(I+NPOINT)=ZERO
  enddo
  !     Call the IMSL subroutine to perform the fourier transform
  !
  CALL FFTSC(A,2*NPOINT,ST,CT,IWK,WK,CWK)

  DO I=2,NPOINT
     F2(I) = DV*(TWO*NPOINT)*ST(I)/(I-1)
  enddo

  RETURN
END SUBROUTINE LINFFT

!=========================================================================

SUBROUTINE LOGFFT(F1,F2,FACT)
  !-----------------------------------------------------------------------
  !     This subroutine performs Fourier transform on a log scale.
  !     Reference: Rossky P. J. and H. L. Friedman, JCP 72, 5694 (1980).

  use chm_kinds
  use number
  use rism
  use fft
  use stream
  implicit none
  real(chm_real) F1(2*DVECT),F2(2*DVECT)
  ! Conversion factor TP32 (forward transform) or TPM32 (backward transform)
  real(chm_real) FACT

  !     The size needed for the work array IWK used in the IMSLD routine
  !     is MPWR2+1, with 2**MPWR2=NN and NN the # of points in the FFT.
  !     Here we pass the maximum possible size 2*DVECT
  !
  INTEGER IWK(2*DVECT)

  INTEGER I

  !     Calculate the Fourier transform of the first function
  !     on the scale   rk(i)=exp{rmin+dr*(i-1)}
  !     and          rk32(i)=exp{rmin+dr*(i-1)}**(3/2)

  DO I=1,NPOINT
     A1(I)=RK32(I)*F1(I)
     !
     !     append NPOINT zeros to stabilize the convolution
     !
     A1(I+NPOINT)=ZERO
  enddo
  !
  !     Count only half of the the first dr interval and half of the last
  !
  A1(1)=HALF*A1(1)
  A1(NPOINT)=HALF*A1(NPOINT)

  !     Calculate the FFT of the data points stored in the column A1
  !     Use the IMSL subroutine
  !
  CALL FFT2C(A1,MPWR2,IWK)

  !     Calculate the second Fourier transform
  DO I=1,NPOINT
     A1(I)=(A1(I)+F1(1)*TRNC(I))*CM(I)
     A1(I+NPOINT)=ZERO
  enddo
  !     Count only half of the first dt and half of the last dt
  !
  A1(1)=HALF*A1(1)
  A1(NPOINT)=HALF*A1(NPOINT)

  !     Calculate the FFT of the data points stored in the column A1
  !     Use the IMSL subroutine
  !
  CALL FFT2C(A1,MPWR2,IWK)
  !
  !     multiply by (2*pi)**3/2  = TP32  --> forward transform
  !     multiply by (2*pi)**-3/2 = TPM32 --> inverse transform
  DO I=1,NPOINT
     F2(I)=FACT*RKM32(I)*DBLE(A1(I))
  enddo
  RETURN
END SUBROUTINE LOGFFT
#endif /* (rism_main)*/

