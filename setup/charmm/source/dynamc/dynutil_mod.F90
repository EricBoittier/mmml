module dynutil

contains

SUBROUTINE ASSVEL(TEMNEW,X,Y,Z,VX,VY,VZ, &
     AMASS,ISEED,IASVEL,IGVOPT,NATOM,IMOVE &
#if KEY_TSM == 1
     , BACKLS & ! for TSM; optional
#endif
     , QFHDGB, VS_DHDGB, SAMASS, TOTALS) ! for DHDGB; optional, no need for cpp

  !     PERFORMS VELOCITY ASSIGNMENTS FOR A DYNAMICS RUN.
  !
  !     Authors: S. Swaminathan
  !              Bernie Brooks

  use chm_kinds
  use dimens_fcm
  use number
  use rndnum
  use consta
  use stream
#if KEY_TSM==1
  use tsms_mod
#endif 
  use parallel
#if KEY_DHDGB==1
!AP/MF
  use gbmv
#endif
  implicit none

  real(chm_real) TEMNEW
  real(chm_real) X(*),Y(*),Z(*)
  real(chm_real) VX(*),VY(*),VZ(*)
  real(chm_real) AMASS(*)
  INTEGER ISEED,IASVEL,IGVOPT,NATOM
  INTEGER IMOVE(*)

#if KEY_TSM == 1
  integer, optional, intent(in) :: backls(*)
#endif

  !AP/MF -- for DHDGB; optional so no need for cpp guards
  LOGICAL,optional,intent(in)::QFHDGB
  REAL(chm_real),optional,intent(out)::VS_DHDGB(*)
  REAL(chm_real),optional,intent(in)::SAMASS
  INTEGER,optional,intent(in)::TOTALS
 
  INTEGER I,NUMNODSAV
  real(chm_real) BOLTZ

  real(chm_real) SD,VEL

  IF(IASVEL == 0) THEN
     IF(PRNLEV >= 2) WRITE(OUTU,22)
22   FORMAT(' NOTE: IASVEL is zero when assignment is requested.', &
          ' Velocities unchanged.')
     RETURN
  ENDIF
  IF(PRNLEV >= 2) WRITE(OUTU,5) IASVEL,TEMNEW,(rngseeds(i),i=1,nrand)
5 FORMAT( &
         /10X,'GAUSSIAN OPTION                  IS ',I12, &
         /10X,'VELOCITIES ASSIGNED AT TEMPERATURE =',F12.4, &
         /10X,'SEED FOR RANDOM NUMBER GENERATOR IS:', &
         /10X,'SEEDS>',8(1X,I0))
  BOLTZ=TEMNEW*KBOLTZ

#if KEY_PARALLEL==1
  IF(QVSEND)THEN
     NUMNODSAV=NUMNOD
     NUMNOD=1
  ENDIF
#endif 
  !
#if KEY_TSM==1
  IF(.NOT.QTSM) THEN
#endif 
     DO I=1,NATOM
        IF(IMOVE(I) == 0) THEN
           SD=BOLTZ/AMASS(I)
           SD=SQRT(SD)
           CALL GAUSSI(ZERO,SD,VEL,ISEED,IASVEL)
           VX(I)=VEL           
           CALL GAUSSI(ZERO,SD,VEL,ISEED,IASVEL)
           VY(I)=VEL
           CALL GAUSSI(ZERO,SD,VEL,ISEED,IASVEL)
           VZ(I)=VEL
        ELSE
           VX(I)=ZERO
           VY(I)=ZERO
           VZ(I)=ZERO
        ENDIF
     ENDDO
#if KEY_DHDGB==1
!AP/MF
  if (present(qfhdgb) .and. present(vs_dhdgb) .and. present(samass) &
                      .and. present(totals)) then
     IF (QFHDGB) THEN
         DO I=1,TOTALS/2
            SD=BOLTZ/SAMASS
            SD=SQRT(SD)
            CALL GAUSSI(ZERO,SD,VEL,ISEED,IASVEL)
            VS_DHDGB(I)=VEL
            IF (I .GT. NDT_FHDGB) THEN
               VS_DHDGB(I)=0.D0
            ENDIF
         ENDDO
         DO I=1,TOTALS/2
            SD=BOLTZ/SAMASS
            SD=SQRT(SD)
            CALL GAUSSI(ZERO,SD,VEL,ISEED,IASVEL)
            VS_DHDGB(I+TOTALS/2)=VEL
            IF (I .GT. NDT_FHDGB) THEN           
               VS_DHDGB(I+TOTALS/2)=0.D0
            ENDIF
         ENDDO
     ELSE
         DO I=1,TOTALS
            VS_DHDGB(I)=ZERO
         ENDDO
     ENDIF
   end if
#endif
#if KEY_TSM==1
  ELSE if (present(backls)) then
     DO I=1,NATOM
        IF(IMOVE(I) == 0.AND.BACKLS(I).EQ.0) THEN
           SD=BOLTZ/AMASS(I)
           SD=SQRT(SD)
           CALL GAUSSI(ZERO,SD,VEL,ISEED,IASVEL)
           VX(I)=VEL
           CALL GAUSSI(ZERO,SD,VEL,ISEED,IASVEL)
           VY(I)=VEL
           CALL GAUSSI(ZERO,SD,VEL,ISEED,IASVEL)
           VZ(I)=VEL
        ELSE
           VX(I)=ZERO
           VY(I)=ZERO
           VZ(I)=ZERO
        ENDIF
     ENDDO
  ENDIF
#endif 
  !
#if KEY_PARALLEL==1
  IF(QVSEND)THEN
     NUMNOD=NUMNODSAV
     CALL PSND8(VX,NATOM)
     CALL PSND8(VY,NATOM)
     CALL PSND8(VZ,NATOM)
  ENDIF
  !
#endif 
  !
#if KEY_TSM==1
  IF (QTSM.AND.PIGSET) THEN
     CALL PIGCVSET(X,Y,Z)
     CALL BACK0(VX,VY,VZ)
  ENDIF
#endif 

  IGVOPT=2
  RETURN
END SUBROUTINE ASSVEL

end module dynutil
