module smd
  !CHARMM Element source/rxncor/smd.src $Revision: 1.15 $

#if KEY_SMD==1 /*smd*/
contains

  SUBROUTINE SMDINI(PDIR,NRXNCR,TREELO,DELVAL,DL0,SMDDEL,BASALO, &
       BASAHI,BASBLO,BASBHI,QFLIP,PUMB)
    !
    !       Initialize variables for SMD based on RXNCOR with TPS.
    !       ARD and Jie Hu 06-06-30
    !
    use chm_kinds
    use number
    use rxncom,only: ddperi
    implicit none
    !      
    INTEGER PDIR, NRXNCR, TREELO(:), QFLIP(:)
    real(chm_real)  DL0(:), SMDDEL(:), PUMB(:), DELVAL(5,*)
    real(chm_real)  BASALO(:), BASAHI(:), BASBLO(:), BASBHI(:)
    !
    INTEGER IR
    !

    DO IR = 1, NRXNCR
       !         Check if each order parameter is being steered
       IF (ABS(SMDDEL(IR))  >  RSMALL) THEN
          !           Set the target value to the current value
          DL0(3*(IR-1)+1) = DELVAL(1,TREELO(IR))
       ENDIF
    ENDDO

    !       Determine sign of SMDDEL
    CALL SMDSGN(PDIR,NRXNCR,SMDDEL,DL0,BASALO,BASAHI,BASBLO,BASBHI, &
         QFLIP,PUMB)

    DO IR = 1, NRXNCR
       IF (ABS(SMDDEL(IR))  >  RSMALL) THEN
          DL0(3*(IR-1)+1) =  &
               DDPERI(DELVAL(1,TREELO(IR))+SMDDEL(IR),PUMB(IR))
       ENDIF
    ENDDO

    RETURN
  END SUBROUTINE SMDINI

  FUNCTION HALFPE(ALO,AHI,PUMB)
    !
    !       Determine the midpoint of a basin for SMD based on RXNCOR. 
    !       Allows basins based on periodic order parameters.
    !       ARD and Jie Hu 06-06-30
    !
    use chm_kinds
    use number
    use rxncom,only: ddperi

    implicit none
    real(chm_real) ALO, AHI, PUMB, DD, halfpe
    DD = AHI - ALO
    DD = DDPERI(DD,PUMB)
    HALFPE = ALO + HALF * DD
    HALFPE = DDPERI(HALFPE,PUMB)
    RETURN
  END FUNCTION HALFPE

  SUBROUTINE SMDADD(IR,DELTA,DELTA0,PUMB,SMDDEL,BASALO,BASAHI, &
       BASBLO,BASBHI,QFLIP)
    !
    !       Advance the target order parameter value for SMD based on RXNCOR.
    !       ARD and Jie Hu 06-06-30
    !
    use chm_kinds
    use number
    use rxncom,only: ddperi
    implicit none
    INTEGER IR, QFLIP(:)
    real(chm_real)  DELTA(3), DELTA0(3,*), PUMB(:)
    real(chm_real)  SMDDEL(:), BASALO(:), BASAHI(:), BASBLO(:),  &
         BASBHI(:)
    !
    real(chm_real)  DD, TMPDIR, R
    LOGICAL QINB
    !
    !       Check if we should advance the umbrella for SMD
    DD = DELTA(1) - DELTA0(1,IR)
    DD = DDPERI(DD,PUMB(IR))

    !        write(*,*)'smdadd>ir=',ir
    !        write(*,*)'smdadd>pumb(ir)=',pumb(ir)
    !        write(*,*)'smdadd>basahi(ir)=',basahi(ir)
    !        write(*,*)'smdadd>basalo(ir)=',basalo(ir)

    TMPDIR = DDPERI(BASAHI(IR)-BASALO(IR),PUMB(IR))

    !       Most of the rest of the routine handles times when the system
    !       is already in the endpoint basins. 

    R = QFLIP(IR)*SMDDEL(IR)*TMPDIR

    !       Check if in basin B
    IF (R  >  RSMALL) THEN
       QINB = QFFNC1(DELTA0(1,IR),BASBLO(IR),BASBHI(IR),PUMB(IR), &
            TMPDIR,SMDDEL(IR))
       IF ((.NOT. QINB) .AND. (DD*SMDDEL(IR)  >  ZERO)) &
            DELTA0(1,IR) = DELTA0(1,IR) + SMDDEL(IR)

       !       Check if in basin A
    ELSE IF (R  <  -RSMALL) THEN
       QINB = QFFNC1(DELTA0(1,IR),BASALO(IR),BASAHI(IR),PUMB(IR), &
            TMPDIR,SMDDEL(IR))
       IF ((.NOT. QINB) .AND. (DD*SMDDEL(IR)  >  ZERO)) &
            DELTA0(1,IR) = DELTA0(1,IR) + SMDDEL(IR)
    ENDIF

    !       Adjust resulting target order parameter value for periodicity
    DELTA0(1,IR) = DDPERI(DELTA0(1,IR),PUMB(IR))
    RETURN
  END SUBROUTINE SMDADD

  LOGICAL FUNCTION QFFNC1(D0,BLO,BHI,P,TD,SD)
    !
    !       Check if in a basin and within SD of its center.
    !
    use chm_kinds
    use number
    use rxncom,only: ddperi

    implicit none
    real(chm_real) D0,BLO,BHI,P,TD,SD
    QFFNC1 = (DDPERI(D0-BLO,P) * TD  >  ZERO) .AND. &
         (DDPERI(D0-BHI,P) * TD  <  ZERO) .AND. &
         (ABS(DDPERI(D0-HALFPE(BLO,BHI,P),P))  <  ABS(SD))
    RETURN
  END FUNCTION QFFNC1

  LOGICAL FUNCTION QFFNC2(D0,BLO,BHI,P,TD,SD,SIGN)
    !
    !       Check if in basin and outside the center.
    !
    use chm_kinds
    use number
    use rxncom,only: ddperi

    implicit none
    INTEGER SIGN
    real(chm_real) D0,BLO,BHI,P,TD,SD
    QFFNC2 = (DDPERI(D0-BLO,P) * TD  >  ZERO) .AND. &
         (DDPERI(D0-BHI,P) * TD  <  ZERO) .AND. &
         (DDPERI(D0-HALFPE(BLO,BHI,P),P) * TD * SIGN  >  ZERO)
    RETURN
  END FUNCTION QFFNC2

  SUBROUTINE SMDSGN(PDIR,NRXNCR,SMDDEL,DL0PTR,BASALO, &
       BASAHI,BASBLO,BASBHI,QFLIP,PUMB)
    !
    !       ARD and Jie Hu 06-06-30
    !       Set the sign of SMDDEL for TPS.
    !
    use chm_kinds
    use number
    use rxncom,only: ddperi

    implicit none
    INTEGER PDIR, NRXNCR, QFLIP(:)
    real(chm_real)  SMDDEL(:), DL0PTR(3,*), BASALO(:), BASAHI(:)
    real(chm_real)  BASBLO(:), BASBHI(:), PUMB(:)
    real(chm_real)  TMPDIR
    !
    INTEGER IR
    !
    DO IR = 1, NRXNCR

       QFLIP(IR) = 1
       TMPDIR = DDPERI(BASAHI(IR)-BASALO(IR),PUMB(IR))
       TMPDIR = TMPDIR / ABS(TMPDIR)
       SMDDEL(IR) = -FLOAT(PDIR)*ABS(SMDDEL(IR))*TMPDIR

       IF ((PDIR==-1).AND.QFFNC2(DL0PTR(1,IR),BASBLO(IR),BASBHI(IR), &
            PUMB(IR),TMPDIR,SMDDEL(IR),1) &
            .OR.(PDIR== 1).AND.QFFNC2(DL0PTR(1,IR),BASALO(IR),BASAHI(IR), &
            PUMB(IR),TMPDIR,SMDDEL(IR),-1)) THEN
          SMDDEL(IR) = -SMDDEL(IR)
          QFLIP(IR) = -1
       ENDIF
    ENDDO
    RETURN
  END SUBROUTINE SMDSGN

#endif /* (smd)*/

end module smd

