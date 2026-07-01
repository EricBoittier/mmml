SUBROUTINE DIEWRN(LEV)
  !-----------------------------------------------------------------------
  !     THIS ROUTINE CALLS DIE IF LEV IS  <=  THAN BOMLEV
  !
  !      BRB
  !
  use chm_kinds
  use stream
  use timerm
  use machutil,only:die
  implicit none
  INTEGER LEV
  !
  IF(LEV < BOMMIN) BOMMIN=LEV
  IF(WRNLEV >= LEV) WRITE(OUTU,20) LEV,BOMLEV
20 FORMAT(' *** LEVEL',I3,' WARNING *** BOMLEV IS',I5)
  IF(LEV <= BOMLEV) THEN
     WRITE(OUTU,22)
     CALL DIE
  ENDIF
22 FORMAT(' BOMLEV HAS BEEN SATISFIED. TERMINATING.')
  RETURN
END SUBROUTINE DIEWRN

SUBROUTINE WRNDIE(LEV,SCALL,MESSAG)
  !-----------------------------------------------------------------------
  !     THIS ROUTINE PRINTS A WARNING (IF LEV <= WRNLEV) AND CALLS DIE
  !     IF (LEV <= BOMLEV).
  !     LEV - LEVEL OF WARNING (FROM +5 MINOR TO -5 SEVERE)
  !     SCALL - 8-CHAR REPRESENTATION OF CALLING ROUTINE
  !     E.G. "<CHARMM>"
  !     MESSAGE - HOLLERITH STRING WITH WARNING MESSAGE.
  !
  use chm_kinds
  use stream
  use timerm
  !      -- mikem ->
  use machutil,only:die
  !      <- mikem --
  implicit none
  character(len=*) MESSAG,SCALL
  INTEGER LEV
  !
  IF(LEV < BOMMIN) BOMMIN=LEV
  IF(LEV <= BOMLEV .OR. LEV.LE.WRNLEV) THEN
     WRITE(OUTU,22) LEV,SCALL,MESSAG
22   FORMAT(/'      ***** LEVEL',I3,' WARNING FROM ',A,' *****',/, &
          '      ***** ',A,/, &
          '      ******************************************')
     !
     IF (LEV > BOMLEV) THEN
        IF(WRNLEV >= LEV) THEN
           ! BIOVIA Code Start : Bug fix
           WRITE(OUTU,31,ERR=10) BOMLEV,WRNLEV
           ! BIOVIA Code End
31         FORMAT('      BOMLEV (',I3,') IS NOT REACHED. WRNLEV IS', &
                I3/)
        ENDIF
     ELSE
        ! BIOVIA Code Start : Bug fix
        WRITE(OUTU,32,ERR=10) BOMLEV,WRNLEV
        ! BIOVIA Code End
32      FORMAT('      BOMLEV (',I3,') IS REACHED - TERMINATING.', &
             ' WRNLEV IS',I3)
        CALL DIE
     ENDIF
  ENDIF
  !
! BIOVIA Code Start : Bug fix
10  RETURN
! BIOVIA Code End
END SUBROUTINE WRNDIE

!  Modified by SRD 1/27/91.
!
FUNCTION EXPC(X) result(val)
  !-----------------------------------------------------------------------
  !     Returns exponential of X or ZERO if X is greater than OVFLOW.
  !-----------------------------------------------------------------------
  use chm_kinds
  use number
  use stream
  implicit none
  real(chm_real) X,val
  !
  !     LOCAL
  real(chm_real) OVFLOW
  !     the bigest double precision exponential argument is
  !     approximately 709.
  PARAMETER (OVFLOW=709.0)
  !
  IF (X > OVFLOW) THEN
     val=ZERO
     IF(PRNLEV >= 2) WRITE(OUTU,10) X,OVFLOW
  ELSE
     val=EXP(X)
  ENDIF
10 FORMAT(' EXPC> WARNING: return ZERO due to exponent overflow ', &
       E10.3,' over ',E10.3)
  RETURN
END FUNCTION EXPC

