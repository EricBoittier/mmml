#if KEY_ADUMB==1 /*adumb*/
!
! Routines to evaluate MD runs generated with adaptive umbrella sampling
!
! C. Bartels, 1996
!
! =======================================================================
SUBROUTINE UMGETP(MMF,MEM,MCA,MFL)
  ! 
  ! Calculates and writes series with weighting factors for each structure
  ! of the trajectory
  ! INPUT:  File with umbrella coordinates of each structure
  ! OUTPUT: File with weighting factors
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use consta
  use number
  use umb
  use stream
  implicit none
  real(chm_real)  MMF(*),MEM(*),MCA(*)
  INTEGER MFL(*)
  !
  INTEGER I,DIM,IND,IC,STEP,CNT,NMF,IS,J
  real(chm_real)  RT,RT2,C(MAXUMP),F,TFACT,E,AVEMF
  real(chm_real)  TF,TALPH
  !
  !      IF((TEUM /= TUM).AND.((NUMBR > 0).OR.(IEUM == 0))) THEN
  IF((TEUM /= TUM).AND.(IEUM == 0).AND.PRNLEV > 2) THEN
     WRITE(OUTU,*)  &
          'Energy ubrella must be active to change temperature.'
     !        WRITE(OUTU,*) 'Use instead: UMBR INIT TEMP ',TEUM
     RETURN 
  ENDIF
  RT=KBOLTZ*TUM
  RT2=KBOLTZ*TEUM
  ! calculate weighting factor for each bin
  ! from sum of counts (MEM) and potential of mean force (MMF)
  ! make sure that the important weighting factors are within machine precision
  AVEMF=0.0
  NMF=0
  DO I=1,INSIUM
     IF(MEM(I) > 0) THEN 
        IF(TEUM /= TUM) THEN
           ! calculate index j of bin in energy umbrella dimension
           J=I
           IF(IEUM < NUMBR)  J=MOD(J,INSTUM(IEUM+1))
           IF(IEUM > 1)      J=J/INSTUM(IEUM)
           E=EMINUM+ &
                ((FLOAT(J)-0.5)/NBUM(IEUM)-0.25)*2.0*(EMAXUM-EMINUM)
           TFACT=E*(1.0/RT-1.0/RT2)
        ELSE
           TFACT=0.0
        END IF
        TFACT=TFACT-LOG(MEM(I))-MMF(I)/RT
        IF(NMF == 0) THEN 
           AVEMF=TFACT
        ELSE 
           AVEMF=MAX(AVEMF,TFACT)
        END IF
        NMF=NMF+1
     END IF
  ENDDO
  AVEMF=AVEMF-250
  DO I=1,INSIUM
     MCA(I)=0.0
     IF(MEM(I) > 0) MCA(I)=-MMF(I)/RT-LOG(MEM(I))-AVEMF
  ENDDO
  !
  IF(IOLEV < 0) RETURN
  !
  ! go through file
  CNT=0
20 READ(RCUNUM,*,END=100,ERR=90) STEP,(C(I),I=1,NUMBR,1)
  ! get weigthing factor appropriate for the umbralla coordinates 
  CNT=CNT+1
  IS=(STEP-1)/FUPUM+1
  F=0.0
  TF=0.0
  TALPH=0.0
  E=EMINUM+(C(IEUM)-0.25)*((EMAXUM-EMINUM)*2.0)
  IF(MOD(STEP,FUPUM) > NEQUM.AND.MFL(IS) == 0) THEN
     IND=0
     DO DIM=1,NUMBR
        IC=MIN(MAX(INT(C(DIM)*NBUM(DIM)),0),NBUM(DIM)-1)
        IND=IND+IC*INSTUM(DIM)
     ENDDO
     TF=1.0
     TALPH=-MMF(IND+1)/RT-LOG(MEM(IND+1))+E/RT
     IF(TEUM /= TUM) THEN
        TFACT=E*(1.0/RT-1.0/RT2)
     ELSE
        TFACT=0.0
     END IF
     TFACT=MCA(IND+1)+TFACT
     IF(TFACT > -250) F=EXP(TFACT)
  END IF
  IF(WPUNUM > 0) THEN
     WRITE(WPUNUM,400) F
400  FORMAT(E20.12E4)
  ENDIF
  IF(WTUNUM > 0) THEN
     WRITE(WTUNUM,410) TF,TALPH,E
410  FORMAT(F5.1,1X,F20.5,1X,F20.5)
  ENDIF
  GOTO 20
90 WRITE(OUTU,*) '*** Error in reading umbrella coordinate file'
100 WRITE(OUTU,*) CNT,' umbrella coordinates converted'
  !
  return
end SUBROUTINE UMGETP

#endif /* (adumb)*/

SUBROUTINE NULL_UMG
  RETURN
END SUBROUTINE NULL_UMG

