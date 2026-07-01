SUBROUTINE CORFUN(TQ,TQ2,MXNTOT,FA,FB,FC,FD,FAH,FBH, &
     CORCOD,CRSCOD,VECCOD, &
     QFFT,QLTC,NTOT,NCORR,QNNORM,TCORR,NTOT2,QDIFF, &
     XNORM)
  !
  !     THIS ROUTINE TAKES THE TIME SERIES AND CALCULATES
  !     THE CORRELATION FUNCTION.
  !
  !     OPTIONS :
  !     QFFT       - TRUE - USE FFT TO CALCULATE C.F. (DEFAULT)
  !                - FALSE- USE DIRECT MULTIPLICATION
  !
  !     QLTC       - TRUE - USE LONG TAIL CORRECTION
  !                - FALSE- NO LONG TAIL CORRECTION  (DEFAULT)
  !
  !     CORCOD     - 1 - <F(0).F(T)>  (DEFAULT)
  !                - 2 - <3*(F(0).F(T))**2 - 1>
  !
  !     CRSCOD     - 1 - AUTO CORRELATION
  !                - 2 - CROSS CORRELATION
  !
  !     QNNORM     - TRUE - CORRELATION FUNCTION IS NOT NORMALIZED
  !                - FALSE -CORRELATION FUNCTION IS NORMALIZED (DEFAULT)
  !
  !     QDIFF      - Correlate time series differences, not product.
  !     XNORM .ne. 0 - multiply all points in final correlation function by XNORM
  !
  !     Author: S. Swaminathan
  !     Modifications by David Perahia and T. Ichiye
  !
  use chm_kinds
  use consta
  use number
  use param_store, only: set_param

  implicit none

  INTEGER MXNTOT
  real(chm_real) TQ(MXNTOT,*),TQ2(MXNTOT,*)
  real(chm_real) FA(*),FB(*),FC(*),FD(*),FAH(*),FBH(*),XNORM
  INTEGER CORCOD,CRSCOD,VECCOD
  LOGICAL QFFT,QLTC
  INTEGER NTOT,NCORR
  LOGICAL QNNORM
  real(chm_real) TCORR(*)
  INTEGER NTOT2
  LOGICAL QDIFF
  !
  real(chm_real) FBA1,FBA2
  real(chm_real) QH(3)
  !sb      real(chm_real) SQRT2,FQA2,DNORM,FAAT1,FAAT2,FAA1,FAA2,FHH,QHH,HALF3
  real(chm_real) SQRT2,FQA2,DNORM,FAAT1,FAAT2,FAA1,FAA2,FHH,QHH
  INTEGER NTOT1,NTOTP1,M,I,IMVC,MVC,MV2,M1,NMAX,N
  !
  SQRT2 = SQRT(TWO)
  !sb      HALF3=THREE*HALF
  NTOT1=NTOT
  NTOTP1=NTOT+1
  !
  !*****SELECT FFT OR DIRECT (QFFT)
  !
  IF(QDIFF) THEN
     !        process-diff-by-direct-manipulation
     DO M=1,NCORR
        FC(M)=ZERO
     ENDDO
     DO M=1,NTOT1
        M1=M-1
        NMAX=NCORR+M1
        IF(NMAX.GT.NTOT1) NMAX=NTOT1
        DO MVC=1,VECCOD
           QH(MVC)=TQ(M,MVC)
        ENDDO
        DO N=M,NMAX
           QHH=zero
           DO MVC=1,VECCOD
              QHH=QHH+(QH(MVC)-TQ2(N,MVC))**2
           ENDDO
           FC(N-M1)=FC(N-M1)+QHH
        ENDDO
     ENDDO
     !
     DO M=1,NCORR
        FC(M)=FC(M)/(NTOTP1-M)
     ENDDO
     !
  ELSE IF(QFFT) THEN
     IF(CORCOD.EQ.1) THEN
        !           process-p1-correlation-function-by-fft
        !
        !*****CALCULATE THE CORRELATION FUNCTION USING FOURIER TRANSFORMS
        !
        FQA2=zero
        FAAT1=zero
        FAAT2=zero
        DO M=1,NTOT2
           FC(M)=zero
           FD(M)=zero
        ENDDO
        !
        ! MVC IS A CRAZY COUNTER FOR HANDLING VECTORS.
        ! IF VECCOD = 1, FA = TQ(T)
        ! IF VECCOD = 3, WHEN MVC=1: FA = X(T), FB = Y(T) SO THE FFT IS
        ! FOR THE COMPLEX VECTOR X+iY; WHEN MVC=3: FA = Z(T), FB = 0.
        ! ONLY 2 CALLS TO FFT INSTEAD OF 3
        !
        IMVC=2
        DO MVC=1,VECCOD,IMVC
           FAA1=zero
           FAA2=zero
           FBA1=zero
           FBA2=zero
           !
           !  FOR P1
           !  FAA, FBA ARE SUMS OF FA & FB, RESPECTIVELY
           !  FAAT = SUM OF Q**2, (IF VECCOD=3, Q IS NORM OF VECTOR)
           !     CALCULATED ONLY FOR X-COR
           !
           DO M=1,NTOT1
              FA(M)=TQ(M,MVC)
           ENDDO
           IF(MVC.LT.VECCOD) THEN
              DO M=1,NTOT1
                 !sb                     FB(M)=TQ(M,MVC)
                 FB(M)=TQ(M,MVC+1)
              ENDDO
           ELSE
              DO M=1,NTOT1
                 FB(M)=zero
              ENDDO
           ENDIF
           IF(QLTC) THEN
              DO M=1,NTOT1
                 FAA1=FAA1+FA(M)
              ENDDO
              IF(MVC.LT.VECCOD) THEN
                 DO M=1,NTOT1
                    FBA1=FBA1+FB(M)
                 ENDDO
              ENDIF
           ENDIF
           IF(CRSCOD.EQ.2) THEN
              DO M=1,NTOT1
                 FAAT1=FAAT1+FA(M)*FA(M)
              ENDDO
              IF(QLTC) FAAT1=FAAT1-FAA1*FAA1/NTOT1
              IF(MVC.LT.VECCOD) THEN
                 DO M=1,NTOT1
                    FAAT1=FAAT1+FB(M)*FB(M)
                 ENDDO
                 IF(QLTC) FAAT1=FAAT1-FBA1*FBA1/NTOT1
              ENDIF
           ENDIF
           !
           DO M=NTOTP1,NTOT2
              FA(M)=zero
              FB(M)=zero
           ENDDO
           !
           CALL correl_FFT(FA,FB,NTOT2,NTOT2,NTOT2,1)
           !
           IF(CRSCOD.EQ.2) THEN
              DO M=1,NTOT2
                 FAH(M)=FA(M)
                 FBH(M)=FB(M)
              ENDDO
              !
              !     FOR P1
              !     FAA, FBA ARE SUMS OF FA & FB, RESPECTIVELY
              !     FAAT = SUM OF Q**2, (IF VECCOD=3, Q IS NORM OF VECTOR)
              !          CALCULATED ONLY FOR X-COR
              !
              DO M=1,NTOT1
                 FA(M)=TQ2(M,MVC)
              ENDDO
              IF(MVC.LT.VECCOD) THEN
                 DO M=1,NTOT1
                    !sb                        FB(M)=TQ2(M,MVC)
                    FB(M)=TQ2(M,MVC+1)
                 ENDDO
              ELSE
                 DO M=1,NTOT1
                    FB(M)=zero
                 ENDDO
              ENDIF
              IF(QLTC) THEN
                 DO M=1,NTOT1
                    FAA2=FAA2+FA(M)
                 ENDDO
                 IF(MVC.LT.VECCOD) THEN
                    DO M=1,NTOT1
                       FBA2=FBA2+FB(M)
                    ENDDO
                 ENDIF
              ENDIF
              IF (CRSCOD.EQ.2) THEN
                 DO M=1,NTOT1
                    FAAT2=FAAT2+FA(M)*FA(M)
                 ENDDO
                 IF(QLTC) FAAT2=FAAT2-FAA2*FAA2/NTOT1
                 IF(MVC.LT.VECCOD) THEN
                    DO M=1,NTOT1
                       FAAT2=FAAT2+FB(M)*FB(M)
                    ENDDO
                    IF(QLTC) FAAT2=FAAT2-FBA2*FBA2/NTOT1
                 ENDIF
              ENDIF
              !           
              DO M=NTOTP1,NTOT2
                 FA(M)=zero
                 FB(M)=zero
              ENDDO
              !
              CALL correl_FFT(FA,FB,NTOT2,NTOT2,NTOT2,1)
              !
           ENDIF
           !
           IF(CRSCOD.EQ.1) THEN
              !                 for auto-cor, fc = <q**2>
              DO M=1,NTOT2
                 FC(M)=FC(M)+FA(M)*FA(M)+FB(M)*FB(M)
              ENDDO
              IF(QLTC) FQA2=FQA2+FAA1*FAA1+FBA1*FBA1
           ELSE IF(CRSCOD.EQ.2) THEN
              !                 for x-cor, fc = <qa.qb>
              DO M=1,NTOT2
                 FC(M)=FC(M)+FA(M)*FAH(M)+FB(M)*FBH(M)
                 FD(M)=FD(M)+FAH(M)*FB(M)-FA(M)*FBH(M)
              ENDDO
              IF(QLTC) FQA2=FQA2+FAA1*FAA2+FBA1*FBA2
           ENDIF
        ENDDO
        !
        !*****TAKE INVERSE FOURIER TRANSFORM
        CALL correl_FFT(FC,FD,NTOT2,NTOT2,NTOT2,-1)
        !
        !           select long tail correction (qltc)
        IF(QLTC) THEN
           FQA2=(FQA2/NTOT1)/NTOT1
           !C               FQA2=FQA2/(NTOT1*NTOT1)
        ELSE
           FQA2=zero
        ENDIF
        !
        ! lni, mar04, allow for reporting of initial amplitude, and xternal normalization factor
        !           select normalization
        !            IF(QNNORM) THEN
        !               DNORM=ONE
        !            ELSE
        IF(CRSCOD.EQ.1) THEN
           DNORM=ONE/((FC(1)/NTOT2)/NTOT1-FQA2)
           !C                  DNORM=ONE/(FC(1)/(NTOT2*NTOT1)-FQA2)
        ELSE IF(CRSCOD.EQ.2) THEN
           DNORM=NTOT1/SQRT(FAAT1*FAAT2)
        ENDIF
        !            ENDIF
        call set_param('CFNORM',DNORM)
        IF(QNNORM) DNORM=ONE
        IF(XNORM .NE. ZERO) DNORM=XNORM
        DO M=1,NTOT1
           FC(M)=((FC(M)/NTOT2)/(NTOTP1-M)-FQA2)*DNORM
           !C               FC(M)=(FC(M)/(NTOT2*(NTOTP1-M))-FQA2)*DNORM
        ENDDO
        !
     ELSE IF(CORCOD.EQ.2) THEN
        !           process-p2-correlation-function-by-fft
        !
        !*****CALCULATE THE CORRELATION FUNCTION USING FOURIER TRANSFORMS
        !
        ! IF P2, MUST BE A VECTOR!
        IF(VECCOD.NE.3) CALL WRNDIE(-2,'<CORFUN>', &
             'P2 BY FFT MUST BE FOR A VECTOR')
        !
        FQA2=zero
        DO M=1,NTOT2
           FC(M)=zero
           FD(M)=zero
        ENDDO
        !
        ! MVC IS A CRAZY COUNTER FOR HANDLING VECTORS.
        ! IF VECCOD = 1, FA = TQ(T)
        ! IF VECCOD = 3, WHEN MVC=1: FA = X(T), FB = Y(T) SO THE FFT IS
        ! FOR THE COMPLEX VECTOR X+iY; WHEN MVC=3: FA = Z(T), FB = 0.
        ! ONLY 2 CALLS TO FFT INSTEAD OF 3
        !
        IMVC=1
        DO MVC=1,VECCOD,IMVC
           FAA1=zero
           FAA2=zero
           FBA1=zero
           FBA2=zero
           !
           ! FOR P2
           ! MVC=1, FA=X*X, FB=sqrt(2)*X*Y; MVC=2, FA=Y*Y, FB=sqrt(2)*Y*Z;
           ! MVC=3, FA=Z*Z, FB=sqrt(2)*Z*X
           MV2=MVC+1
           IF(MVC.EQ.3) MV2=MV2-3
           DO M=1,NTOT1
              FA(M)=TQ(M,MVC)*TQ(M,MVC)
              FB(M)=SQRT2*TQ(M,MVC)*TQ(M,MV2)
           ENDDO
           !
           DO M=NTOTP1,NTOT2
              FA(M)=zero
              FB(M)=zero
           ENDDO
           !
           CALL correl_FFT(FA,FB,NTOT2,NTOT2,NTOT2,1)
           !
           IF(CRSCOD.EQ.2) THEN
              DO M=1,NTOT2
                 FAH(M)=FA(M)
                 FBH(M)=FB(M)
              ENDDO
           ENDIF
           !
           IF(CRSCOD.EQ.2) THEN
              !
              MV2=MVC+1
              IF(MVC.EQ.3) MV2=MV2-3
              DO M=1,NTOT1
                 FA(M)=TQ2(M,MVC)*TQ2(M,MVC)
                 FB(M)=SQRT2*TQ2(M,MVC)*TQ2(M,MV2)
              ENDDO
              !                 
              DO M=NTOTP1,NTOT2
                 FA(M)=zero
                 FB(M)=zero
              ENDDO
              !
              CALL correl_FFT(FA,FB,NTOT2,NTOT2,NTOT2,1)
              !
           ENDIF
           !
           IF(CRSCOD.EQ.1) THEN
              !                 for auto-cor, fc = <q**2>
              DO M=1,NTOT2
                 FC(M)=FC(M)+FA(M)*FA(M)+FB(M)*FB(M)
              ENDDO
              IF(QLTC) FQA2=FQA2+FAA1*FAA1+FBA1*FBA1
           ELSE IF(CRSCOD.EQ.2) THEN
              !                 for x-cor, fc = <qa.qb>
              DO M=1,NTOT2
                 FC(M)=FC(M)+FA(M)*FAH(M)+FB(M)*FBH(M)
                 FD(M)=FD(M)+FAH(M)*FB(M)-FA(M)*FBH(M)
              ENDDO
              IF(QLTC) FQA2=FQA2+FAA1*FAA2+FBA1*FBA2
           ENDIF
        ENDDO
        !
        !*****take inverse fourier transform
        CALL correl_FFT(FC,FD,NTOT2,NTOT2,NTOT2,-1)
        !
        DO M=1,NTOT1
           !C               FC(M)=HALF3*FC(M)/(NTOT2*(NTOTP1-M))-HALF
           !sb               FC(M)=((HALF3*FC(M))/NTOT2)/(NTOTP1-M)-HALF
           FC(M)=((onept5*FC(M))/NTOT2)/(NTOTP1-M)-HALF
        ENDDO
     ENDIF
     !
  ELSE
     DO M=1,NTOT1
        FC(M)=zero
     ENDDO
     IF(CORCOD.EQ.1) THEN
        !           process-p1-by-direct-manipulation
        FQA2=zero
        FAAT1=zero
        FAAT2=zero
        DO MVC=1,VECCOD
           FAA1=zero
           FAA2=zero
           !
           DO M=1,NTOT1
              FA(M)=TQ(M,MVC)
              FAH(M)=TQ2(M,MVC)
           ENDDO
           !
           DO M=1,NTOT1
              M1=M-1
              NMAX=NCORR+M1
              IF(NMAX.GT.NTOT1) NMAX=NTOT1
              FHH=FA(M)
              DO N=M,NMAX
                 FC(N-M1)=FC(N-M1)+FHH*FAH(N)
              ENDDO
           ENDDO
           !
           !     for normalization and ltc:  faa = sum of fa,  fqa2 is ltc;
           !     for x-cor, faat1 = ntot * ms delta atom a
           IF(CRSCOD.EQ.1) THEN
              IF(QLTC) THEN
                 DO M=1,NTOT1
                    FAA1=FAA1+FA(M)
                 ENDDO
                 FQA2=FQA2+FAA1*FAA1
              ENDIF
           ELSE IF(CRSCOD.EQ.2) THEN
              DO M=1,NTOT1
                 FAAT1=FAAT1+FA(M)*FA(M)
                 FAAT2=FAAT2+FAH(M)*FAH(M)
                 FAA1=FAA1+FA(M)
                 FAA2=FAA2+FAH(M)
              ENDDO
              IF(QLTC) THEN
                 FAAT1=FAAT1-FAA1*FAA1/NTOT1
                 FAAT2=FAAT2-FAA2*FAA2/NTOT1
                 FQA2=FQA2+FAA1*FAA2
              ENDIF
           ENDIF
        ENDDO
        !
        !     select long tail correction (qltc)
        IF(QLTC) THEN
           FQA2=(FQA2/NTOT1)/NTOT1
        ELSE
           FQA2=zero
        ENDIF
        ! lni, mar04, allow for reporting of initial amplitude, and xternal normalization factor
        !     select normalization
        !            IF(QNNORM) THEN
        !               DNORM=ONE
        !            ELSE
        IF(CRSCOD.EQ.1) THEN
           DNORM=ONE/(FC(1)/NTOT1-FQA2)
        ELSE IF(CRSCOD.EQ.2) THEN
           DNORM=NTOT1/SQRT(FAAT1*FAAT2)
        ENDIF
        !            ENDIF
        call set_param('CFNORM',DNORM)
        IF(QNNORM) DNORM=ONE
        IF(XNORM .NE. ZERO) DNORM=XNORM
        DO M=1,NCORR
           FC(M)=(FC(M)/(NTOTP1-M)-FQA2)*DNORM
        ENDDO
     ELSE IF(CORCOD.EQ.2) THEN
        !           process-p2-by-direct-manipulation
        DO M=1,NTOT1
           M1=M-1
           NMAX=NCORR+M1
           IF (NMAX.GT.NTOT1) NMAX=NTOT1
           DO MVC=1,VECCOD
              QH(MVC)=TQ(M,MVC)
           ENDDO
           DO N=M,NMAX
              QHH=zero
              DO MVC=1,VECCOD
                 QHH=QHH+QH(MVC)*TQ2(N,MVC)
              ENDDO
              FC(N-M1)=FC(N-M1)+QHH*QHH
           ENDDO
        ENDDO
        !
        DO M=1,NCORR
           !sb               FC(M)=HALF3*FC(M)/(NTOTP1-M)-HALF
           FC(M)=onept5*FC(M)/(NTOTP1-M)-HALF
        ENDDO
     ENDIF
  ENDIF
  !
  DO I=1,NCORR
     TCORR(I)=FC(I)
  ENDDO
  RETURN
  !
end SUBROUTINE CORFUN

