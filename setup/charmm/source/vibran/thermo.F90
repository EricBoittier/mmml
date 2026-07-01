!CHARMM Element source/vibran/thermo.src $Revision: 1.9 $

  SUBROUTINE THERMV(IFRQ,JFRQ,FREQ,TEMP,STEP,LPRINT,FCUT,NIGNORE)
    !
    ! ROUTINE TO COMPUTE THERMODYNAMIC FUNCTIONS FOR NORMAL MODES
    ! BERNARD R. BROOKS   5/19/83
    !
    ! Modifications by L. Nilsson, October 2002
    ! STEP <0: COMPUTE VALUES ONLY AT T=TEMP
    ! LPRINT=.TRUE. PRINT OUT VALUES
    ! NIGNORE number of frequencies < 1.0 that have been ignored
    !
  use chm_kinds
  use consta
  use stream
  use energym
  use param_store, only: set_param

    implicit none

    INTEGER IFRQ,JFRQ,NIGNORE
    real(chm_real) FREQ(*)
    real(chm_real) TEMP,STEP,FCUT
    LOGICAL LPRINT

    real(chm_real) CONST
    real(chm_real) FTOT,FCTOT,STOT,HTOT,CTOT,ZTOT,ETOT,UTOT
    real(chm_real) F,FC,S,H,C,Z,U,Q
    INTEGER NTEMP,ITEMP,I,N1

    ! conversion from CM-1 to degrees Kelvin
    ! CONST = ( h * c / Kb )
    CONST=(6.6262 * 10.0 * SPEEDL ) / 1.38062

    IF(PRNLEV.GE.2 .AND. LPRINT) WRITE(OUTU,23) IFRQ,JFRQ
23  FORMAT(' Modes',I4,' to ',I4,' will be used.')
    IF(PRNLEV.GE.2 .AND. LPRINT) WRITE(OUTU,24)
24  FORMAT(' F - Free energy'/' FC- Classical free energy'/ &
         ' S - Entropy'/' H - Enthalpy'/ &
         ' C - Heat capacity (constant volume)'/ &
         ' Z - Zero point correction energy' // &
         4X,'Temp',11X,'F',10X,'FC',11X,'S',11X,'H',11X,'C',11X,'Z')

    NIGNORE=0
    DO I=IFRQ,JFRQ
       IF(FREQ(I).LT.FCUT) THEN
          NIGNORE=NIGNORE+1
          IF(PRNLEV.GE.6) WRITE(OUTU,44) I,FREQ(I)
44        FORMAT(' WARNING: Mode',I4,' has a frequency of ',F10.2, &
               '. It will be ignored.')
       ENDIF
    ENDDO
    IF(LPRINT.AND.PRNLEV.GE.2.AND.NIGNORE.GT.0) &
         WRITE(OUTU,'(/I10,A,G9.3,A/)')  &
         NIGNORE,' frequencies <',FCUT,' have been ignored.'

    IF(STEP.GT.0.0)THEN
       N1=0
       NTEMP=((TEMP/STEP)+0.5)
    ELSE
       N1=1
       NTEMP=1
    ENDIF
    DO ITEMP=N1,NTEMP
       IF(STEP.GT.0.0) TEMP=ITEMP*STEP

       FTOT=0.0
       FCTOT=0.0
       STOT=0.0
       HTOT=0.0
       CTOT=0.0
       ZTOT=0.0
       UTOT=1.0

       DO I=IFRQ,JFRQ
          IF(FREQ(I).GE.FCUT)THEN
             IF(TEMP.GE.0.1) THEN
                U=CONST*FREQ(I)/TEMP
                Q=1.0/(1.0-EXP(-U))
                F=LOG(1.0/Q)
                FC=LOG(U)
                Z=U/2

                ! This test is to prevent overflows
                IF(U.GT.40.0) U=40.0
                H=U/(EXP(U)-1.0)
                S=H-F
                C=U*U*EXP(U)/(EXP(U)-1.0)**2
             ELSE
                U=9999.0
                Q=1.0
                F=0.0
                FC=0.0
                H=0.0
                S=0.0
                C=0.0
                Z=0.0
             ENDIF

             FTOT=FTOT+F
             FCTOT=FCTOT+FC
             STOT=STOT+S
             HTOT=HTOT+H
             CTOT=CTOT+C
             ZTOT=ZTOT+Z
             IF(UTOT.LT.1.0D30) UTOT=UTOT*U
             !
             !     WRITE(2,83) I,FREQ(I),U,Q,F,S,H,C
             ! 83  FORMAT(' I=',I4,' FREQ=',F10.2,' U=',F12.6,' Q=',F12.6/
             !    1   15X,' F=',F12.6,' S=',F12.6,' H=',F12.6,' CV=',F12.6)
             !
          ENDIF
       ENDDO
       !
       ! FTOT - FREE ENERGY (LESS ZERO POINT CORRECTION)
       ! STOT - ENTROPY
       ! HTOT - ENTHALPY (LESS ZERO POINT CORRECTION)
       ! CTOT - HEAT CAPACITY
       ! ZTOT - ZERO POINT CORRECTION ENERGY
       ! UTOT - PRODUCT OF PARTION FUNCTIONS.
       !
       FTOT=FTOT*KBOLTZ*TEMP
       FCTOT=FCTOT*KBOLTZ*TEMP
       STOT=STOT*KBOLTZ
       HTOT=HTOT*KBOLTZ*TEMP
       CTOT=CTOT*KBOLTZ
       ZTOT=ZTOT*KBOLTZ*TEMP

       IF(PRNLEV.GE.2.AND.LPRINT) THEN
          !     WRITE(OUTU,90) TEMP
          ! 90  FORMAT(' TEMP= ',F12.1)
          WRITE(OUTU,92)TEMP, FTOT,FCTOT,STOT,HTOT,CTOT,ZTOT
92        FORMAT(F8.1,6F12.4)
       ENDIF

    ENDDO

    ETOT=FCTOT+EPROP(EPOT)
    IF(PRNLEV.GE.2.AND.LPRINT) WRITE(OUTU,94) ETOT
94  FORMAT(/' Total classical harmonic free energy (ETOT=FC+E) is: ' &
         ,F12.4)

    call set_param('FTOT',FTOT)
    call set_param('FCTO',FCTOT)
    call set_param('STOT',STOT)
    call set_param('HTOT',HTOT)
    call set_param('CTOT',CTOT)
    call set_param('ZTOT',ZTOT)
    call set_param('ETOT',ETOT)

    RETURN
  END SUBROUTINE THERMV

