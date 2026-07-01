#if KEY_OLDDYN==1
SUBROUTINE DLNGVV(RFD,NATOM,RFT,DX,DY,DZ,ISEED)
  !
  ! This function routine generates a Gaussian random
  ! deviate of 0.0 mean and standard deviation RF.
  ! The algorithm is sum of uniform deviates algorithm
  ! which may be found in Abramowitz and Stegun,
  ! "Handbook of Mathematical Functions", pg 952.
  !
  ! Charles Brooks III, 3-JULY-1983
  !
  use clcg_mod,only:random
  use chm_kinds
  use number
  use consta
  use exfunc
  implicit none
  !
  real(chm_real) RFT(*)
  !
  real(chm_real)  RFD(*)
  real(chm_real)  DX(*),DY(*),DZ(*)
  real(chm_real)  RFDT
  INTEGER NATOM,ISEED
  real(chm_real)  A,B
  INTEGER I,J,N
  IF (NATOM <= 0) RETURN
  N=2*((NATOM*3+1)/2)
  DO I=1,N,2
     A=SQRT(MINTWO*LOG(MAX(RANDOM(ISEED),RSMALL)))
     B=TWOPI*RANDOM(ISEED)
     RFT(I) = A*COS(B)
     RFT(I+1) = A*SIN(B)
  ENDDO
  !
  DO I=1,NATOM
     J= INT((ONEPT5*I-ONE)*TWO)
     RFDT = MAX(RFD(I),RSMALL)
     DX(I)=DX(I)-RFDT*RFT(J)
     J=J+1
     DY(I)=DY(I)-RFDT*RFT(J)
     J=J+1
     DZ(I)=DZ(I)-RFDT*RFT(J)
  ENDDO
  RETURN
END SUBROUTINE DLNGVV

SUBROUTINE LNGFILV(GAMMA,RFD,TBATH,DELTA,RBUF, &
     SXREF,SYREF,SZREF,X,Y,Z &
#if KEY_ACE==1
     ,QRADI,BSARR,LLBACE          & 
#endif
     )
  !
  !     This routine fills the GAMMA and RFD arrays for Langevin dynamics.
  !     For more information see routine DYNLNG
  !
  !     Charles Brooks III and Axel Brunger, 3-JULY-1983
  !     ------------------------------------------------
  !
  ! input/output
  use chm_kinds
  use dimens_fcm
  use psf
  use number
  use cnst_fcm
  use consta
  use stream
  implicit none
  real(chm_real) GAMMA(*), RFD(*), TBATH, DELTA, RBUF
  real(chm_real) SXREF, SYREF, SZREF
  real(chm_real)  X(*), Y(*), Z(*)
#if KEY_ACE==1
  real(chm_real) QRADI(*),BSARR(*)       
#endif
#if KEY_ACE==1
  LOGICAL LLBACE                 
#endif
  ! local
  INTEGER I, ILANG, ISTRT
  LOGICAL QYES
  real(chm_real)  ROXY, RBUF2, DDX, DDY, DDZ
#if KEY_ACE==1
  real(chm_real)  FBETAL, SOLFAC         
#endif
  ! Save the local variable
  SAVE ISTRT
  DATA ISTRT/0/
  ! begin
  RBUF2=RBUF*RBUF
  ILANG=0
  DO I=1,NATOM
     GAMMA(I)=ZERO
     RFD(I)=ZERO
     IF(ABS(FBETA(I)) > RSMALL.AND.IMOVE(I) == 0) THEN
        IF(RBUF2 < RSMALL) THEN
           QYES=.TRUE.
        ELSE
           DDX=X(I)-SXREF
           DDY=Y(I)-SYREF
           DDZ=Z(I)-SZREF
           ROXY=DDX*DDX+DDY*DDY+DDZ*DDZ
           QYES=(ROXY > RBUF2)
        ENDIF
        IF(QYES) THEN
#if KEY_ACE==1
           IF (LLBACE) THEN
              !                 scale FBETA with (Ri/bi)^2, where Ri=vdW Radius, and
              !                 bi=Born solvation radius of atom i (see ACE potential)
              SOLFAC=QRADI(IAC(I))/BSARR(I)
              FBETAL=SOLFAC*SOLFAC*FBETA(I)
              GAMMA(I)=TIMFAC*FBETAL*DELTA/TWO
              RFD(I)=TWO*AMASS(I)*TIMFAC*FBETAL*KBOLTZ*TBATH/DELTA
           ELSE
#endif 
              GAMMA(I)=TIMFAC*FBETA(I)*DELTA/TWO
              RFD(I)=TWO*AMASS(I)*TIMFAC*FBETA(I)*KBOLTZ*TBATH/DELTA
#if KEY_ACE==1
           ENDIF                                   
#endif
           IF(RFD(I) <= RSMALL) THEN
              RFD(I)=ZERO
           ELSE
              RFD(I)=SQRT(RFD(I))
           ENDIF
           ILANG=ILANG+1
        ENDIF
     ENDIF
  ENDDO
  !
  IF(ILANG > 0 .AND. ISTRT /= 1) THEN
     ISTRT=1
     IF(PRNLEV >= 2) WRITE(OUTU,9000) TBATH, DELTA, ILANG, RBUF, &
          SXREF, SYREF, SZREF
9000 FORMAT(' LNGFIL: TBATH = ',F12.6,'  DELTA =',F12.6,/, &
          ' LNGFIL: Langevin dynamics setup for ',I7,' atoms',/, &
          ' LNGFIL: RBUF =',F12.6,/, &
          ' LNGFIL: SXREF =',F12.6,' SYREF =',F12.6,' SZREF =',F12.6)
#if KEY_ACE==1
     IF (LLBACE) WRITE(OUTU,9001)                          
#endif
#if KEY_ACE==1
9001 FORMAT(' LNGFIL: Use (Rvdw/Bsolv)^2 to scale FBETA')  
#endif
  ENDIF
  !
  return
end SUBROUTINE LNGFILV

#endif 
SUBROUTINE NULL_GV
  RETURN
END SUBROUTINE NULL_GV

