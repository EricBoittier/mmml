SUBROUTINE SPECTR(TRES,FA,FAH,NTOT,NSKIP,DELTA,QFOLD, &
     QRAMP,QSWIT,NPRI2)
  !
  !     This routine computes the spectral density from a correlation
  !     function.
  !
  !      Bernard R. Brooks  -  12-Feb-1985
  !
  use chm_kinds
  use consta
  implicit none

  real(chm_real) TRES(*)
  real(chm_real) FA(*),FAH(*)
  INTEGER NTOT,NSKIP
  real(chm_real) DELTA
  LOGICAL QFOLD,QRAMP,QSWIT
  INTEGER NPRI2
  !
  real(chm_real) FSKIP
  INTEGER I,NPI
  !
  FSKIP=NSKIP*DELTA
  NPI=NTOT
  !
  DO I=1,NPRI2
     FA(I)=0.0
     FAH(I)=0.0
  ENDDO
  IF(QRAMP) THEN
     DO I=1,NPI
        FA(I)=(TRES(I)*2.0*(NPI+1-I))/NPI
     ENDDO
  ELSE
     DO I=1,NPI
        FA(I)=TRES(I)
     ENDDO
  ENDIF
  !
  IF(QSWIT) THEN
     DO I=1,NPI
        FA(I)=FA(I)*(1.0+COS((PI*(I-1))/NPI))
     ENDDO
  ENDIF
  !
  IF(QFOLD) THEN
     IF(NPRI2.GE.NPI) THEN
        DO I=2,NPI
           FA(NPRI2+2-I)=FA(I)
        ENDDO
     ELSE
        CALL WRNDIE(0,'<SPECTR>', &
             'Cannot use FOLDing option without at least 2xN points')
     ENDIF
  ENDIF
  !
  CALL correl_FFT(FA,FAH,NPRI2,NPRI2,NPRI2,1)
  DELTA=1.0/(NPRI2*FSKIP*SPEEDL)
  NSKIP=1
  TRES(1)=FA(1)
  DO I=2,NPI
     TRES(I)=FA(NPRI2+2-I)
  ENDDO
  !
  return
end SUBROUTINE SPECTR

