module fft
  use chm_kinds
  use dimens_fcm

#if KEY_RISM==1 /*fft_fcm*/
  use rism
  !-----------------------------------------------------------------------
  !             Fast Fourier Transform common block 
  !-----------------------------------------------------------------------
  !     VARIABLE     PURPOSE
  !
  !     NFIRST        The first point in the arrays r(i) and rk(i)
  !     NPOINT        The number of points to be used in the FFT 
  !                   (default=512)
  !     MPWR2         Used in the IMSLD routine for logfft
  !     RMIN          Used in the logfft (default=5.12)
  !     DR            spacing for the r-space points (default=0.02)
  !     DVR           volume element in r-space = DR**3
  !     DK            spacing for the k-space points
  !     DVK           volume element in k-space 
  !     R(DVECT)      array for the r-space vector
  !     RK(DVECT)     array for the k-space vector
  !     TP32          (2*PI)**3/2   - multiplies the forward FFT
  !     TPM32         (2*PI)**-3/2  - multiplies the inverse FFT
  !
  !     RK32(DVECT)       \
  !     RKM32(DVECT) 
  !     TRNC(DVECT)
  !                     used in the logfft calculation
  !     CM(DVECT)    
  !     A1(2*DVECT)
  !     WORK(5*2*DVECT/2) /
  !
  !     QLOG          .TRUE. if the log scale fft is to be used (default)
  !     
  INTEGER NFIRST,NPOINT,MPWR2
  real(chm_real) RMIN,DR,DVR,DK,DVK,R(dvect),RK(dvect),TP32,TPM32, &
                 RK32(dvect),RKM32(dvect)
  COMPLEX(chm_cmpx),dimension(dvect) :: CM,TRNC
  COMPLEX(chm_cmpx) A1(2*DVECT)
  LOGICAL QLOG

#endif /* (fft_fcm)     RISM*/

end module fft

