module tpvv2
  use chm_kinds
  use dimens_fcm
  implicit none

contains

#if KEY_DYNVV2==1 /*dynvv2_main*/
  SUBROUTINE TPCONTROL(COMLYN,COMLEN)
    !-----------------------------------------------------------------------
    !     Author: G. Lamoureux (2003) Edited by Ed Harder (2004)
    !
    !     Temperature and pressure control for DYNA VV2
    !
    !     TPCONTROL is a parser for some of the functionalities of the
    !     DYNA VV2 command:
    !     - multiple thermostats
    !     - SCF treatment of the Drude oscillators (if any)
    !     - separate cooling of the Drude oscillators (if any)
    !     - constant pressure for isotropic and anisotropic systems
    !     - damping of the motion of the center of mass
    !
    !-----------------------------------------------------------------------
    use exfunc
    use number
    !
    use psf
    use stream
    use string
    use nose_mod
    use reawri, only: iupten, iptfrq ! P tensor unit, frequency
    use consta ! for TIMFAC,ROOMT
#if KEY_BLOCK==1
    use block_fcm, only : blasgn  
#endif
    use dynvv2, only: ident33,T_Drude
    character(len=*) COMLYN
    INTEGER       COMLEN
    !

    !     Locals
    INTEGER     I, J, itmp
    integer,allocatable,dimension(:) :: ITEMP
    CHARACTER(len=4) WRD
    real(chm_real)      TMIN,TMAX
    INTEGER     oldWRNLEV
    INTEGER     NATOMinT(MAXNOS),TATOMinT
    INTEGER     NDRUDinT(MAXNOS)


    !     OFF switch
    IF (INDXA(COMLYN,COMLEN,'OFF') > 0) THEN
       QTPCON = .FALSE.
       QTCON = .FALSE.
       QPCON = .FALSE.

       QNOSE = .FALSE.
       DO I=1,NOBL
          NDGN(I)=0
       ENDDO
       !        Free allocated memory
       IF (QNOSP) THEN
          call chmdealloc('tpcontrol.src','TPCONTROL','INLCKP',NATOM,intg=INLCKP)
          QNOSP = .FALSE.
       ENDIF

       IF (PRNLEV > 2) WRITE(OUTU,'(A)') ' TPCONTROL> OFF'
       RETURN
       QQRELT = .FALSE.
       DO I=1,MAXNOS
          QRELT(I) = .FALSE.
       ENDDO
       IABST = 0
       QNHCH = .FALSE.
    ENDIF


    !     Number of thermostats
    NOBL = GTRMI(COMLYN,COMLEN,'NTHE',1)
    IF (NOBL > MAXNOS-1) THEN
       CALL WRNDIE(-3,'<TPCONTROL>','Too many thermostats.')
       RETURN
    ELSEIF (NOBL < 1) THEN
       CALL WRNDIE(-3,'<TPCONTROL>','Needs at least one thermostat.')
       RETURN
    ENDIF


    !     Turn temperature control ON
    IF (.NOT.QTPCON) THEN
       QTPCON = .TRUE.

       QNOSE = .FALSE.
       !        Allocate memory
       call chmalloc('tpcontrol.src','TPCONTROL','INLCKP',NATOM,intg=INLCKP)
       QNOSP = .TRUE.
       IF (PRNLEV > 2) WRITE(OUTU,'(A)')' TPCONTROL> ON'
    ENDIF


    !
    !     Initialization
    !

    !     Thermostats
    QTCON = .TRUE.
    INLCKP(1:NATOM) = 0
    DO I = 1,MAXNOS
       RTMPR(I) = ROOMT
       SQM(I) = ZERO
       NHTAU(I) = -ONE
    ENDDO

    !     Relative/absolute thermostatting
    QQRELT = .FALSE.
    DO I=1,MAXNOS
       QRELT(I) = .FALSE.
    ENDDO
    IABST = 0
    CMGAMMA = ZERO
    QCMLANGEVIN = .FALSE.

    !     Langevin thermostatting
    NHGAMMA = ZERO
    QNHLANGEVIN = .FALSE.

    !     SCF dipoles
    TOLSCF = TENM5
    MAXSCF = 50

    !     Nose-Hoover chains
    QNHCH = .FALSE.
    DO I=1,MAXNOS
       NHCH_M(I) = 1
       NHCH_B(I) = ZERO
       NHCH_Q(I) = ZERO
    ENDDO

    !     Constant pressure
    QPCON = .FALSE.
    QDSCALE_CP = .TRUE.
    W_CP = ZERO !RBIG
    TAU_CP = -ONE
    P_CP = ZERO

    ! Unit number for pressure tensor
    IUPTEN = GTRMI(COMLYN,COMLEN,'IUPT',-1)
    IPTFRQ = GTRMI(COMLYN,COMLEN,'IPTF',-1)
    IF(PRNLEV >= 2.AND.IUPTEN>0) THEN
      IF (IPTFRQ>0) THEN
        WRITE(OUTU,'(8X,A,I4,A,I8,A)') &
          'Writing P tensor to unit ',IUPTEN, &
          ' every ',IPTFRQ,' steps.'
      ELSE
        CALL WRNDIE(-3,'<TPCONTROL>', &
          'IUPTEN set w/o specifying IPTFRQ')
      ENDIF
    ENDIF

    !     Multi-timestep code for the thermostats
    !     (NC or NSTEps)
    IF (NDRUDE == 0) THEN
       YSNC = GTRMI(COMLYN,COMLEN,'NC',1)
       IF (YSNC == 1) YSNC = GTRMI(COMLYN,COMLEN,'NSTE',1)
    ELSE
       YSNC = GTRMI(COMLYN,COMLEN,'NC',20)
       IF (YSNC == 20) YSNC = GTRMI(COMLYN,COMLEN,'NSTE',20)
    ENDIF
    IF (YSNC <= 0) YSNC = 1

    !     For langevin thermostat on the nuclei
    !     (anything goes:  NHGAMma)
    NHGAMMA = GTRMF(COMLYN,COMLEN,'NHGAM',ZERO)
    NHGAMMAD = GTRMF(COMLYN,COMLEN,'NHGAMD',ZERO) 

    IF (NHGAMMA /= ZERO) THEN
       NHGAMMA = NHGAMMA*TIMFAC ! convert to 1/AKMA
       QNHLANGEVIN = .TRUE.
    ENDIF

    IF (NHGAMMAD /= ZERO) THEN
       NHGAMMAD = NHGAMMAD*TIMFAC ! convert to 1/AKMA
       YSNC = 1
    ENDIF

    !     For the damping of the center-of-mass motion code
    !     (anything goes: CMDAmping, CMFBeta, SINK)
    CMGAMMA = GTRMF(COMLYN,COMLEN,'CMDAM',ZERO)
    IF (CMGAMMA == ZERO) CMGAMMA = GTRMF(COMLYN,COMLEN,'CMFB',ZERO)
    IF (CMGAMMA == ZERO) CMGAMMA = GTRMF(COMLYN,COMLEN,'SINK',ZERO)

    IF (CMGAMMA /= ZERO) THEN
       CMGAMMA = CMGAMMA*TIMFAC ! convert to 1/AKMA
       !        Create an additional "thermostat"
       NOBL = NOBL + 1
       RTMPR(NOBL) = ZERO
       IABST = NOBL           ! it's an "absolute" thermostat
       QCMLANGEVIN = .TRUE.
       IF (PRNLEV >= 2) THEN
       ENDIF
       !        Make all other thermostats "relative"
       QQRELT = .TRUE.
       DO I = 1,NOBL-1
          QRELT(I) = .TRUE.
       ENDDO
    ENDIF



    !
    !     Read the thermostats or the barostat
    !     The barostat should be read after all the thermostats.
    !
    WRD = NEXTA4(COMLYN,COMLEN)

111 CONTINUE
    IF (WRD == 'THER') THEN

       I = NEXTI(COMLYN,COMLEN)
       IF (I < 1 .OR. I > NOBL) THEN
          CALL WRNDIE(-3,'<TPCONTROL>','Invalid thermostat number.')
       ENDIF

       !        Default options
       QNHLANG(I) = .FALSE.
       SQM(I) = ZERO
       IF(QNHLANGEVIN)THEN
       NHTAU(I) = MEGA
       ELSE 
       NHTAU(I) = -ONE
       ENDIF
       RTMPR(I) = ROOMT

       !        Read options
112    CONTINUE
       WRD = NEXTA4(COMLYN,COMLEN)
       IF (WRD == 'SELE') THEN
          !           Assign selection
          CALL JOINWD(COMLYN,MXCMSZ,COMLEN,WRD,4)
#if KEY_BLOCK==1
          call chmalloc('tpcontrol.src','TPCONTROL','ITEMP',NATOM,intg=ITEMP)
          oldWRNLEV = WRNLEV
          WRNLEV = 0
          CALL BLASGN(COMLYN,COMLEN,ITEMP,NATOM,INLCKP,I)
          WRNLEV = oldWRNLEV
          call chmdealloc('tpcontrol.src','TPCONTROL','ITEMP',NATOM,intg=itemp)
#endif 
       ELSEIF (WRD == 'QREF') THEN
          SQM(I) = NEXTF(COMLYN,COMLEN)
       ELSEIF (WRD == 'TAU ') THEN
         IF(QNHLANGEVIN) THEN 
          CALL WRNDIE(-3,'<TPCONTROL>','Langevin thermostat already there!')
         ELSE  
          NHTAU(I) = NEXTF(COMLYN,COMLEN)
         ENDIF
       ELSEIF (WRD == 'TREF') THEN
          RTMPR(I) = NEXTF(COMLYN,COMLEN)
          IF (RTMPR(I) == ZERO) THEN
             TOLSCF = GTRMF(COMLYN,COMLEN,'TOLS',TENM5)
             IF (TOLSCF <= ZERO) TOLSCF = TENM5
             MAXSCF = GTRMI(COMLYN,COMLEN,'MAXS',50)
             IF (MAXSCF <= 0) MAXSCF = 50
          ENDIF
       ELSEIF (WRD == 'LANG') THEN
          QNHLANG(I) = .TRUE.
       ELSEIF (WRD == 'THER' .OR. WRD.EQ.'BARO') THEN
          GOTO 111            ! next thermostat (or barostat)
       ELSEIF (WRD == '    ') THEN
          GOTO 119            ! parsing finished
       ELSE
          CALL WRNDIE(-3,'<TPCONTROL>','Thermostat syntax error')
       ENDIF
       GOTO 112               ! next option

    ELSEIF (WRD == 'BARO') THEN

       QPCON = .TRUE.
       QZONLY = .FALSE.
       QPCONFULL = .FALSE.
       !        Initialize
       CALL IDENT33(RX1_CP)
       CALL IDENT33(RX2_CP)
       CALL IDENT33(RV1_CP)
       CALL IDENT33(RV2_CP)
       !        Default options
       W_CP = ZERO
       TAU_CP = -ONE
       P_CP = ONE
       TI_CP = -1

       TMAX = ZERO
       TMIN = RBIG
       DO I = 1,NOBL
          IF (I == IABST) GOTO 23
          IF (RTMPR(I) > TMAX) TMAX = RTMPR(I)
          IF (RTMPR(I) < TMIN) TMIN = RTMPR(I)
23        CONTINUE
       ENDDO
       T_Drude = TMIN
       IF (TMIN > HALF*TMAX) THEN
          QDSCALE_CP = .TRUE.
       ENDIF

       !        Read options
113    CONTINUE
       WRD = NEXTA4(COMLYN,COMLEN)
       IF (WRD == 'WREF') THEN
          W_CP = NEXTF(COMLYN,COMLEN)
       ELSEIF (WRD == 'BTAU') THEN
          TAU_CP = NEXTF(COMLYN,COMLEN)
       ELSEIF (WRD == 'ZONL') THEN
          QZONLY = .TRUE.
       ELSEIF (WRD == 'FULL') THEN
          QPCONFULL = .TRUE.
       ELSEIF (WRD == 'PREF') THEN
          P_CP = NEXTF(COMLYN,COMLEN)
       ELSEIF (WRD == 'BTHE') THEN
          TI_CP = NEXTI(COMLYN,COMLEN)
       ELSEIF (WRD == 'DSCY') THEN ! DSCALE YES
          QDSCALE_CP = .TRUE.
       ELSEIF (WRD == 'DSCN') THEN ! DSCALE NO
          QDSCALE_CP = .FALSE.
       ELSEIF (WRD == '    ') THEN
          GOTO 119            ! parsing finished
       ELSE
          CALL WRNDIE(-3,'<TPCONTROL>','Barostat syntax error')
       ENDIF
       GOTO 113

    ENDIF
119 CONTINUE


    DO I = 1,NOBL
       NATOMinT(I) = 0
       NDRUDinT(I) = 0
    ENDDO
    DO I = 1,NATOM
       ITMP = INLCKP(I)
       NATOMinT(ITMP) = NATOMinT(ITMP)+1
       IF (ISDRUDE(I)) NDRUDinT(ITMP) = NDRUDinT(ITMP)+1
    ENDDO
    TATOMinT = 0
    DO I = 1,NOBL
       TATOMint = TATOMinT + NATOMinT(I)
    ENDDO


    !
    !     Complete printout
    !
    IF(PRNLEV >= 3) THEN

       DO I = 1,NOBL
          WRITE(OUTU,'(A,I2)') 'Thermostat ',I
          IF (RTMPR(I) > ZERO) THEN
             WRITE(OUTU,'(4X,A,F12.5,A)') &
                  'Reference temperature = ',RTMPR(I),' K'
           IF(QNHLANGEVIN) THEN
           ELSE 
             IF (NHTAU(I) > ZERO) THEN
                WRITE(OUTU,'(4X,A,F12.5,A)') &
                     'Tau (timescale)       = ',NHTAU(I),' ps'
             ELSE
                WRITE(OUTU,'(4X,A,F12.5,A)') &
                     'Inertia parameter     = ',SQM(I),' kcal*AKMA**2'
             ENDIF
            ENDIF
             IF (QNHLANG(I)) THEN
                WRITE(OUTU,'(4X,A)') &
                     'Langevin forces will be added.'
                IF((NDRUDE > 0).AND.(I==2)) NHGAMMA = NHGAMMAD 
                WRITE(OUTU,'(4X,A,F12.5,A)') &
                     'Friction coefficient = ', NHGAMMA/TIMFAC, &
                     ' 1/ps'  
             ENDIF
          ELSE
             IF (QNHLANG(I)) THEN
                IF (NHGAMMA > ZERO) THEN
                   WRITE(OUTU,'(4X,A,F12.5,A)') &
                        'Friction coefficient  = ',NHGAMMA/TIMFAC, &
                        ' 1/ps'
                ENDIF
             ELSE IF(QCMLANGEVIN .AND. (I .eq. IABST) ) THEN
                WRITE(OUTU,'(4X,A)') &
                           'Langevin center of mass damping will be performed.'
                WRITE(OUTU,'(4X,A,F12.5,A)') &
                           'Friction coefficient  = ',CMGAMMA/TIMFAC, &
                           ' 1/ps'
             ELSE 
                WRITE(OUTU,'(4X,A)') &
                     'Energy minimization (SCF dipoles)'
                WRITE(OUTU,'(4X,A,F12.8,A,/,4X,A,I3)') &
                     'Tolerance on the gradient    = ',TOLSCF, &
                     ' kcal/A', &
                     'Maximum number of iterations = ',MAXSCF
             ENDIF
          ENDIF

          IF (QNHCH) THEN
             IF (NHCH_M(I) > 1) THEN
                WRITE(OUTU,'(4X,A,I2,A)') &
                     'Nose-Hoover chain:        ',NHCH_M(I), &
                     ' thermostats'
             ENDIF
             IF (NHCH_B(I) /= ZERO) THEN
                WRITE(OUTU,'(4X,A,F10.5,A)') &
                     'Branka non-eq. correction ',NHCH_B(I)
             ENDIF
          ENDIF

          IF (QRELT(I)) THEN
             WRITE(OUTU,'(4X,A)') 'Relative to the center of mass'
          ENDIF

          WRITE(OUTU,'(4X,A,I7)') &
               'Number of atoms       = ',NATOMinT(I)
       ENDDO

       IF (YSNC > 1) THEN
          WRITE(OUTU,'(A,A)') 'Multi-timestep integration', &
               ' of the thermostats'
          WRITE(OUTU,'(4X,A,I6)') &
               'NC (number of steps)  = ',YSNC
       ENDIF

       IF (QPCON) THEN
          WRITE(OUTU,'(A)') 'Barostat'
          WRITE(OUTU,'(4X,A,F12.5,A)') &
               'Reference pressure    = ',P_CP,' atm'
          IF (TAU_CP > ZERO) THEN
             WRITE(OUTU,'(4X,A,F12.5,A)') &
                  'Tau (timescale)       = ',TAU_CP,' ps'
          ELSE
             WRITE(OUTU,'(4X,A,F12.5,A)') &
                  'Inertia parameter     = ',W_CP,' kcal*AKMA**2'
          ENDIF
          IF (TI_CP > 0) THEN
             WRITE(OUTU,'(4X,A,I2)') &
                  'Coupled to thermostat ',TI_CP
          ENDIF
          IF (QZONLY) THEN
             WRITE(OUTU,'(4X,A)') 'Z-scaling only'
          ENDIF
          IF (QPCONFULL) THEN
             WRITE(OUTU,'(4X,A)') &
                  'Fully flexible scaling (Parrinello-Rahman)'
          ENDIF
          IF (QDSCALE_CP) THEN
             WRITE(OUTU,'(4X,A)') 'Dipole scaling: YES'
          ELSE
             WRITE(OUTU,'(4X,A)') 'Dipole scaling: NO'
          ENDIF
       ENDIF

    ENDIF


    !
    !     Warnings
    !
    DO I = 1,NOBL
       IF (NHTAU(I) < ZERO .AND. SQM(I) == ZERO) THEN
          IF (I /= IABST) THEN
             CALL WRNDIE(-3,'<TPCONTROL>', &
                  'Inertia parameter for thermostat is zero')
          ENDIF
       ENDIF
    ENDDO
    IF (QPCON .AND. TAU_CP < ZERO .AND. W_CP == ZERO) THEN
       CALL WRNDIE(-3,'<TPCONTROL>', &
            'Inertia parameter for barostat is zero')
    ENDIF
    IF (QZONLY .AND. QPCONFULL) THEN
       CALL WRNDIE(-3,'<TPCONTROL>', &
            'Cannot request both ZONLY and FULL.')
    ENDIF

    IF (TATOMinT /= NATOM) THEN
       CALL WRNDIE(-3,'<TPCONTROL>', &
            'Some particles are not coupled to a thermostat.')
    ENDIF

    DO I = 1,NOBL
       IF (NDRUDinT(I) > 0 .AND. NDRUDinT(I) /= NATOMinT(I)) THEN
          CALL WRNDIE(-3,'<TPCONTROL>', &
               'Atoms and Drudes coupled to the same thermostat.')
       ENDIF
       !        Die if many thermostats are assigned to Drude oscillators,
       !        but if some are asking for an SCF treatment, while others
       !        are asking for an extended dynamics treatment.
       IF (NDRUDinT(I) > 0 .AND. RTMPR(I) < TENM8) THEN
          DO J = 1,NOBL
             IF (NDRUDinT(J) > 0 .AND. RTMPR(J) >= TENM8) THEN
                CALL WRNDIE(-3,'<TPCONTROL>', &
                     'Non-uniform treatment of Drude oscillators')
             ENDIF
          ENDDO
       ENDIF
    ENDDO
    return
  end SUBROUTINE TPCONTROL
#else /* (dynvv2_main)*/
  SUBROUTINE NULL_TPCONT
    RETURN
  END SUBROUTINE NULL_TPCONT
#endif /* (dynvv2_main)*/

end module tpvv2

