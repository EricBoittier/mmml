module pull_mod
  ! Torque code from Ioan Andricioaei & Jeff Wereszczynski:
  ! PNAS (2006) 103, 16200-16205
  ! Implemented with small changes in c36a5 L Nilsson Feb 2011


  use chm_kinds
  use dimens_fcm
  implicit none
  !=======================================================================
  ! pull.fcm - stores pointers and timestep information for PULL,EPULL
  !            routines. Lennart Nilsson, March 1996

  INTEGER :: NPULL,NTORQ
  logical :: QTAX
  integer,allocatable,dimension(:) :: IPU, VARY, TTO
  real(chm_real),allocatable,dimension(:) :: AMPL, PERIOD, SAMP
  real(chm_real),allocatable,dimension(:) :: XDIR, YDIR, ZDIR
  real(chm_real),allocatable,dimension(:) :: TAMPL, TPER
  real(chm_real),allocatable,dimension(:) :: TAXISX,TAXISY,TAXISZ
  real(chm_real),allocatable,dimension(:) :: TAXISCX,TAXISCY,TAXISCZ
  real(chm_real) :: PTIME

  real(chm_real),parameter :: EQ=1.6021892D-19, NAVG=6.022045D23, KCAL=4184D0
  real(chm_real),parameter :: ANGST=1D-10, PICO=1D-12
  ! Conversion between AKMA force units and pN or electron charge * Volts/m
  real(chm_real),parameter :: AKMPN=1D0/(KCAL/NAVG/ANGST/PICO)
  real(chm_real),parameter :: AKMEVM=AKMPN*EQ/PICO

contains
  subroutine epull_init()
    npull=0
    return
  end subroutine epull_init

  SUBROUTINE EPULL(EP,DX,DY,DZ,X,Y,Z)
    !-----------------------------------------------------------------------
    ! This routine calculates the externally applied forces (and energies)
    ! as specified in the PULL command.
    ! Lennart Nilsson, Karolinska institutet, March 1996
    !
    use psf
    implicit none
    real(chm_real) EP,DX(*),DY(*),DZ(*),X(*),Y(*),Z(*)

    IF(NPULL .GT. 0) CALL EPULL2(EP,NATOM,DX,DY,DZ,X,Y,Z)
    IF(NTORQ .GT. 0) CALL ETORQ2(EP,NATOM,DX,DY,DZ,X,Y,Z)
    RETURN
  END SUBROUTINE EPULL

  SUBROUTINE EPULL2(EP,NATOM,DX,DY,DZ,X,Y,Z)
    !-----------------------------------------------------------------------
    use number
    use consta
    use stream
    use reawri
    use contrl
    implicit none
    INTEGER NATOM
    real(chm_real)  EP,DX(NATOM),DY(NATOM),DZ(NATOM)
    real(chm_real)  X(NATOM),Y(NATOM),Z(NATOM)
    real(chm_real)  AMPST,FRAC

    INTEGER I,J
    real(chm_real)  T,XF,YF,ZF

    EP=ZERO
    T=PTIME*TWOPI
    DO I=1,NPULL
       J=IPU(I)
       IF(J.GT.NATOM)THEN
          CALL WRNDIE(-4,'<EPULL>','Pulling on non existing atom')
       ENDIF
       IF(PERIOD(I) .GT. ZERO)THEN
          XF=XDIR(I)*AMPL(I)*COS(T/PERIOD(I))
          YF=YDIR(I)*AMPL(I)*COS(T/PERIOD(I))
          ZF=ZDIR(I)*AMPL(I)*COS(T/PERIOD(I))
       ELSE IF(VARY(I) .EQ. 0)THEN
          XF=XDIR(I)*AMPL(I)
          YF=YDIR(I)*AMPL(I)
          ZF=ZDIR(I)*AMPL(I)
       ELSE
          ! switching force linearly between SAMP and AMP
          IF(DYNAMQ)THEN
             AMPST=SAMP(I)+(AMPL(I)-SAMP(I))*(MDSTEP+1)/NSTEP
             XF=XDIR(I)*AMPST
             YF=YDIR(I)*AMPST
             ZF=ZDIR(I)*AMPST
          ELSE
             ! linear switching has to be invoked with dynamics
             CALL WRNDIE(0,'<PULL>', &
                  'Linearly time-varying switching has to be invoked')
          ENDIF
       ENDIF
       DX(J)=DX(J)-XF
       DY(J)=DY(J)-YF
       DZ(J)=DZ(J)-ZF
       ! Calculate energy from this force using a plane thru the origin
       ! as reference position, and previous coordinates (should perhaps
       ! have been the updated coordinates or so, but..)
       EP=EP-(XF*X(J)+YF*Y(J)+ZF*Z(J))
    ENDDO
    RETURN
  END SUBROUTINE EPULL2

  SUBROUTINE PULL_SETUP(COMLYN,COMLEN)
    !-----------------------------------------------------------------------
    ! Parse the PULL command:
    !
    ! PULL { FORCe  <real> } XDIR <real> YDIR <real> ZDIR <real> [PERIod <real>]
    !      { EFIEld <real> }
    !                        [SWITch <int>] [SFORce <real>]
    !                        [WEIGht] [atom-selection]
    !      { TORQue <real> } [PERIod <real>] 
    !                        [AXIS | XDIR <real> YDIR <real> ZDIR <real> -
    !                                XORI <real> YORI <real> ZORI <real>]
    !      { OFF       }
    !      { LIST      }
    !
    !  A force will be applied in the specified direction on the selected atoms
    !  either as a constant: FORCe <value> specified in picoNewtons (pN)
    !  or oscillating in time: FORCe*COS(TWOPI*TIME/PERIod), FORCe <pN> PERIod <ps>
    !     time is counted from the start of the dynamcis run.
    !  or forces due to an electrical EFIEld (V/m) (possibly also oscillating).
    !  Partial charges are then taken from the psf and used to calculate the force.
    !  If WEIGht is specified the forces are multiplied by the wmain array.
    !
    !  The invocation of a non-zero SWITch value will result in a linearly
    !  time-varying force. This command must be used in conjunction with a
    !  dynamics routine (using leap-frog integrator). The force is switched
    !  linearly between SFORce <pN> (starting force) and FORCe <pN> (end force)
    !  over the course of the simulation.
    !
    !  TORQue <pN/A> either uses the axis defined by a prior COOR AXIS command
    !     (or any COOR command which sets the corman axis) or
    !     an axis has to be specified with the *DIR and *ORI keywords.
    !  Each invocation of this command adds a set of forces to the previously
    !  defined set.
    !  PULL OFF turns off all these forces, and PULL LIST prints a listing of
    !  all forces defined so far; this could possibly be a long list....
    !  Note that the direction of the forces obbtained
    !  with the COOR FORCE command is opposite that of the forces given here.
    !
    !  Lennart Nilsson, Karolinska institutet, March 1996
    !
    use coord
    use corsubs
    use memory
    use number
    use psf
    use select
    use stream
    use string
    implicit none
    INTEGER COMLEN
    CHARACTER(len=*) COMLYN

    LOGICAL QWEIG 
    INTEGER I,N
    integer,allocatable,dimension(:) :: ISLCT
    real(chm_real) FORC,PER,V(6),EFI,TOR
    INTEGER SWIT
    real(chm_real) SFOR
    INTEGER WDLEN
    integer,parameter :: WDMAX = 4
    CHARACTER(len=WDMAX) WD

    ! What to do?
    WD=CURRA4(COMLYN,COMLEN)
    IF(WD .EQ. 'OFF ')THEN
       ! Reset pulling forces - if there are any defined
       IF(NPULL .GT. 0)THEN
          call chmdealloc('epull.src','PULL_SETUP','IPU',NPULL,intg=IPU)
          call chmdealloc('epull.src','PULL_SETUP','AMPL',NPULL,crl=AMPL)
          call chmdealloc('epull.src','PULL_SETUP','PERIOD',NPULL,crl=PERIOD)
          call chmdealloc('epull.src','PULL_SETUP','XDIR',NPULL,crl=XDIR)
          call chmdealloc('epull.src','PULL_SETUP','YDIR',NPULL,crl=YDIR)
          call chmdealloc('epull.src','PULL_SETUP','ZDIR',NPULL,crl=ZDIR)
          call chmdealloc('epull.src','PULL_SETUP','SAMP',NPULL,crl=SAMP)
          call chmdealloc('epull.src','PULL_SETUP','VARY',NPULL,intg=VARY)
       ENDIF
       NPULL=0
       ! Reset torques - if there are any defined
       IF(NTORQ .GT. 0)THEN
          call chmdealloc('epull.src','TORQ_SETUP','TTO',NTORQ,intg=TTO)
          call chmdealloc('epull.src','TORQ_SETUP','TAMPL',NTORQ,crl=TAMPL)
          call chmdealloc('epull.src','TORQ_SETUP','TPER',NTORQ,crl=TPER)
          call chmdealloc('epull.src','TORQ_SETUP','TAXISX',NTORQ,crl=TAXISX)
          call chmdealloc('epull.src','TORQ_SETUP','TAXISY',NTORQ,crl=TAXISY)
          call chmdealloc('epull.src','TORQ_SETUP','TAXISZ',NTORQ,crl=TAXISZ)
          call chmdealloc('epull.src','TORQ_SETUP','TAXISCX',NTORQ,crl=TAXISCX)
          call chmdealloc('epull.src','TORQ_SETUP','TAXISCY',NTORQ,crl=TAXISCY)
          call chmdealloc('epull.src','TORQ_SETUP','TAXISCZ',NTORQ,crl=TAXISCZ)
       ENDIF
       NTORQ=0
       COMLEN=0
       RETURN
    ELSEIF (WD .EQ. 'LIST')THEN
       ! Produce listing
       CALL PULIST()
       CALL TORQLIST()
       COMLEN=0
       RETURN
    ENDIF
    ! What atoms are we using
    call chmalloc('epull.src','PULL_SETUP','ISLCT',NATOM,intg=ISLCT)
    CALL SELCTA(COMLYN,COMLEN,ISLCT,X,Y,Z,WMAIN,.TRUE.)
    N=NSELCT(NATOM,ISLCT)
    IF(N .LE. 0)THEN
       ! Nothing to be done now, clean up and return
       IF(PRNLEV .GT. 2)THEN
          WRITE(OUTU,90) NPULL+NTORQ
90        FORMAT(/' PULL> No pulling force added. Total now',I8/)
       ENDIF
       call chmdealloc('epull.src','PULL_SETUP','ISLCT',NATOM,intg=ISLCT)
       RETURN
    ENDIF
    ! Options common to FORCe, EFIEld, TORQue
    QWEIG= (INDXA(COMLYN,COMLEN,'WEIG').GT.0)
    PER=GTRMF(COMLYN,COMLEN,'PERI',MINONE)
    V(1)=GTRMF(COMLYN,COMLEN,'XDIR',ZERO)
    V(2)=GTRMF(COMLYN,COMLEN,'YDIR',ZERO)
    V(3)=GTRMF(COMLYN,COMLEN,'ZDIR',ZERO)
    ! What kind of pulling force? 
    FORC=GTRMF(COMLYN,COMLEN,'FORC',ZERO)
    EFI=GTRMF(COMLYN,COMLEN,'EFIE',ZERO)
    TOR=GTRMF(COMLYN,COMLEN,'TORQ',ZERO)

    SWIT = ZERO
    SFOR = ZERO
    V(4:6) = ZERO
    IF(FORC > ZERO)THEN
       IF(TOR > ZERO .OR. EFI > ZERO) THEN
          CALL WRNDIE(-2,'<PULL_SETUP>',  &
               'Only one of FORC, EFIE, TORQ may be specified')
       ENDIF
       SWIT=GTRMF(COMLYN,COMLEN,'SWIT',ZERO)
       SFOR=GTRMF(COMLYN,COMLEN,'SFOR',ZERO)
       IF(SWIT.NE.0)THEN
          if (prnlev > 2) write(OUTU,95) SFOR,FORC
95        FORMAT(/' PULL> Linear switching between ', &
               F10.3,' and ',F10.3,' pN'/)
       ENDIF
    ELSEIF( EFI > ZERO)THEN
       IF(TOR > ZERO .OR. FORC > ZERO) THEN
          CALL WRNDIE(-2,'<PULL_SETUP>',  &
               'Only one of FORC, EFIE, TORQ may be specified')
       ENDIF
    ELSEIF( TOR > ZERO)THEN
       IF(EFI > ZERO .OR. FORC > ZERO) THEN
          CALL WRNDIE(-2,'<PULL_SETUP>',  &
               'Only one of FORC, EFIE, TORQ may be specified')
       ENDIF
       QTAX=INDXA(COMLYN,COMLEN,'AXIS') > 0
       IF(QTAX)THEN
          IF(.NOT. QAXISC)THEN
             CALL WRNDIE(-3,'<PULL_SETUP>','No previous axis defined')
          ENDIF
       ELSE
          V(4)=GTRMF(COMLYN,COMLEN,'XORI',ZERO)
          V(5)=GTRMF(COMLYN,COMLEN,'YORI',ZERO)
          V(6)=GTRMF(COMLYN,COMLEN,'ZORI',ZERO)
       ENDIF
    ELSE
       CALL WRNDIE(-2,'<PULL_SETUP>','No action specified')
    ENDIF
    IF( (V(1) == ZERO .AND. V(2) == ZERO .AND. V(3) == ZERO) &
         .AND. (.NOT. QTAX .OR. (TOR == ZERO)) ) THEN
       CALL WRNDIE(-2,'<PULL_SETUP>','No direction specified')
    ENDIF

    IF(TOR > 0)THEN
       IF(NTORQ .GT. 0)THEN
          !
          ! Have to make room for new data
          !
          call chmrealloc('epull.src','PULL_SETUP','TTO',N+NTORQ,intg=TTO)
          call chmrealloc('epull.src','PULL_SETUP','TAMPL',N+NTORQ,crl=TAMPL)
          call chmrealloc('epull.src','PULL_SETUP','TPER',N+NTORQ,crl=TPER)
          call chmrealloc('epull.src','PULL_SETUP','TAXISX',N+NTORQ,crl=TAXISX)
          call chmrealloc('epull.src','PULL_SETUP','TAXISY',N+NTORQ,crl=TAXISY)
          call chmrealloc('epull.src','PULL_SETUP','TAXISZ',N+NTORQ,crl=TAXISZ)
          call chmrealloc('epull.src','PULL_SETUP','TAXISCX',N+NTORQ,crl=TAXISCX)
          call chmrealloc('epull.src','PULL_SETUP','TAXISCY',N+NTORQ,crl=TAXISCY)
          call chmrealloc('epull.src','PULL_SETUP','TAXISCZ',N+NTORQ,crl=TAXISCZ)
       ELSE
          call chmalloc('epull.src','PULL_SETUP','TTO',N,intg=TTO)
          call chmalloc('epull.src','PULL_SETUP','TAMPL',N,crl=TAMPL)
          call chmalloc('epull.src','PULL_SETUP','TPER',N,crl=TPER)
          call chmalloc('epull.src','PULL_SETUP','TAXISX',N,crl=TAXISX)
          call chmalloc('epull.src','PULL_SETUP','TAXISY',N,crl=TAXISY)
          call chmalloc('epull.src','PULL_SETUP','TAXISZ',N,crl=TAXISZ)
          call chmalloc('epull.src','PULL_SETUP','TAXISCX',N,crl=TAXISCX)
          call chmalloc('epull.src','PULL_SETUP','TAXISCY',N,crl=TAXISCY)
          call chmalloc('epull.src','PULL_SETUP','TAXISCZ',N,crl=TAXISCZ)
       ENDIF

       CALL TORQ2(N,NATOM,TOR,PER,V,QWEIG,ISLCT,WMAIN)

       IF(PRNLEV .GT. 2)THEN
          WRITE(OUTU,100) N, N+NTORQ
100       FORMAT(/' TORQ>',I8,' Torque(s) added. Total now',I8/)
       ENDIF
       NTORQ=N+NTORQ
    ELSE

       IF(NPULL .GT. 0)THEN
          ! Have to make room for new data
          call chmrealloc('epull.src','PULL_SETUP','IPU',N+NPULL,intg=IPU)
          call chmrealloc('epull.src','PULL_SETUP','AMPL',N+NPULL,crl=AMPL)
          call chmrealloc('epull.src','PULL_SETUP','PERIOD',N+NPULL,crl=PERIOD)
          call chmrealloc('epull.src','PULL_SETUP','XDIR',N+NPULL,crl=XDIR)
          call chmrealloc('epull.src','PULL_SETUP','YDIR',N+NPULL,crl=YDIR)
          call chmrealloc('epull.src','PULL_SETUP','ZDIR',N+NPULL,crl=ZDIR)
          call chmrealloc('epull.src','PULL_SETUP','SAMP',N+NPULL,crl=SAMP)
          call chmrealloc('epull.src','PULL_SETUP','VARY',N+NPULL,intg=VARY)
       ELSE
          call chmalloc('epull.src','PULL_SETUP','IPU',N,intg=IPU)
          call chmalloc('epull.src','PULL_SETUP','AMPL',N,crl=AMPL)
          call chmalloc('epull.src','PULL_SETUP','PERIOD',N,crl=PERIOD)
          call chmalloc('epull.src','PULL_SETUP','XDIR',N,crl=XDIR)
          call chmalloc('epull.src','PULL_SETUP','YDIR',N,crl=YDIR)
          call chmalloc('epull.src','PULL_SETUP','ZDIR',N,crl=ZDIR)
          call chmalloc('epull.src','PULL_SETUP','SAMP',N,crl=SAMP)
          call chmalloc('epull.src','PULL_SETUP','VARY',N,intg=VARY)
       ENDIF

       CALL PULL2(N,NATOM,FORC,PER,V,EFI,QWEIG, &
            SFOR,SWIT,ISLCT,WMAIN,CG)

       IF(PRNLEV .GT. 2)THEN
          WRITE(OUTU,101) N, N+NPULL
101       FORMAT(/' PULL>',I8,' pulling force(s) added. Total now',I8/)
       ENDIF
       NPULL=N+NPULL
    ENDIF
    call chmdealloc('epull.src','PULL_SETUP','ISLCT',NATOM,intg=ISLCT)
    RETURN
  END SUBROUTINE PULL_SETUP

  SUBROUTINE PULL2(N,NATOM,FORC,PER,V,EFI,QWEIG, &
       SFOR,SWIT,ISLCT,WMAIN,CG)
    !-----------------------------------------------------------------------
    use number
    use vector
    implicit none

    INTEGER N,NATOM,ISLCT(NATOM)
    LOGICAL QWEIG
    real(chm_real) FORC,PER,EFI,V(6)
    real(chm_real) WMAIN(NATOM), CG(NATOM)
    INTEGER SWIT
    real(chm_real) SFOR

    INTEGER I,J,JS,JS1
    real(chm_real) XX

    CALL NORMALL(V,3)
    IF(FORC.GT.ZERO)THEN
       ! Use this value
       XX=FORC*AKMPN
    ELSE IF(EFI .NE. ZERO)THEN
       XX=EFI*AKMEVM
    ELSE IF(SFOR .EQ. ZERO)THEN
       ! we didn't specify force at all
       CALL WRNDIE(-3,'<PULL>','No force specified')
       XX=ZERO
    ENDIF
    JS=0
    DO I=1+NPULL,N+NPULL
       XDIR(I)=V(1)
       YDIR(I)=V(2)
       ZDIR(I)=V(3)
       PERIOD(I)=PER
       AMPL(I)=XX
       IF(SFOR.GT.ZERO)THEN
          SAMP(I)=SFOR*AKMPN
       ELSE
          SAMP(I)=ZERO
       ENDIF
       VARY(I)=SWIT
       ! Find atom index of first selected atom
       JS1 = JS + 1
       DO J=JS1,NATOM
          IF(ISLCT(J)== 1)THEN
             JS=J
             ISLCT(J)=0
             EXIT
          ENDIF
       ENDDO
       IPU(I)=JS
       IF(QWEIG) AMPL(I)=AMPL(I)*WMAIN(JS)
       IF(EFI .NE. ZERO) AMPL(I)=AMPL(I)*CG(JS)
       ! Keep amplitudes positive and make directions point the way the
       ! atom will move
       IF(AMPL(I) .LT. ZERO)THEN
          XDIR(I)=-XDIR(I)
          YDIR(I)=-YDIR(I)
          ZDIR(I)=-ZDIR(I)
          AMPL(I)=-AMPL(I)
       ENDIF
    ENDDO
    RETURN
  END SUBROUTINE PULL2

  SUBROUTINE PULIST()
    !-----------------------------------------------------------------------
    ! List currently effective pulling forces to OUTU
    !
    use stream
    use chutil,only:atomid
    implicit none

    INTEGER I,J
    CHARACTER(len=8) ATNAM,RID,SID,REN

    IF(IOLEV .LE. 0 .OR. PRNLEV .LE. 1) RETURN
    IF(NPULL.LE.0)THEN
       WRITE(OUTU,'(/,A,/)') ' PULL> No external forces defined.'
       RETURN
    ENDIF
    WRITE(OUTU,800) NPULL
800 FORMAT(' PULL> Currently these',I6, &
         ' external forces are defined',/16X, &
         'Atom      Ampl [AKMA]       [pN]   Swit  Start[pN]   Per [ps]', &
         '  Direction (X,Y,Z)')
    DO I=1,NPULL
       J=IPU(I)
       CALL ATOMID(J,SID,RID,REN,ATNAM)
       WRITE(OUTU,900) J, &
            SID(1:idleng),RID(1:idleng),REN(1:idleng),ATNAM(1:idleng), &
            AMPL(I),AMPL(I)/AKMPN,VARY(I),SAMP(I)/AKMPN,PERIOD(I), &
            XDIR(I),YDIR(I),ZDIR(I)
900    FORMAT(1X,I5,4(1X,A),1X,F10.4,1X,F10.3,4(1X),I2,2X,F10.3,1X, &
            F10.3,3(1X,F5.2))
    ENDDO
    WRITE(OUTU,'(//)')
    RETURN
  END SUBROUTINE PULIST

  ! Torque code from Ioan Andricioaei & Jeff Wereszczynski
  ! 

  SUBROUTINE ETORQ2(EP,NATOM,DX,DY,DZ,X,Y,Z)
    !-----------------------------------------------------------------------
    use number
    use consta
    use stream
    use reawri
    use contrl
    !  use corman_mod
    INTEGER NATOM
    REAL(chm_real)  EP,DX(NATOM),DY(NATOM),DZ(NATOM)
    REAL(chm_real)  X(NATOM),Y(NATOM),Z(NATOM)
    REAL(chm_real)  YD,ZD,TIME,RADIUS,XD
    REAL(chm_real)  DIFX,DIFY,DIFZ,LENG
    REAL(chm_real)  CX,CY,CZ,DOTX,DOTY,DOTZ
    REAL(chm_real)  ADOTX,ADOTY,ADOTZ,VX,VY,VZ,TAXISR2
    INTEGER APPLY
    !
    INTEGER I,J
    REAL(chm_real)  T,XF,YF,ZF
    !
    EP=ZERO
    T=PTIME*TWOPI
    DO I=1,NTORQ
       APPLY=1
       J=TTO(I)
       IF(J.GT.NATOM)THEN
          CALL WRNDIE(-4,'<ETORQ>','Torque on non existing atom')
       ENDIF
       DIFX=TAXISCX(I)-X(J)
       DIFY=TAXISCY(I)-Y(J)
       DIFZ=TAXISCZ(I)-Z(J)
       TAXISR2= TAXISX(I)**2+TAXISY(I)**2+TAXISZ(I)**2
       ADOTX=-1*DIFX*TAXISX(I)/TAXISR2
       ADOTY=-1*DIFY*TAXISY(I)/TAXISR2
       ADOTZ=-1*DIFZ*TAXISZ(I)/TAXISR2
       CX=TAXISCX(I)+ADOTX*TAXISX(I)
       CY=TAXISCY(I)+ADOTY*TAXISY(I)
       CZ=TAXISCZ(I)+ADOTZ*TAXISZ(I)
       VX=X(J)-CX
       VY=Y(J)-CY
       VZ=Z(J)-CZ
       RADIUS=SQRT(VX**2+VY**2+VZ**2)
       !         IF(RADIUS.LT.1) THEN
       !           RADIUS=1
       !         END
       XD=-VY*TAXISZ(I)+VZ*TAXISY(I)
       YD=-VZ*TAXISX(I)+VX*TAXISZ(I)
       ZD=-VX*TAXISY(I)+VY*TAXISX(I)
       LENG=SQRT(XD**2+YD**2+ZD**2)
       XD=XD/LENG
       YD=YD/LENG
       ZD=ZD/LENG
       IF(TPER(I) .GT. ZERO)THEN
          XF=XD*TAMPL(I)*COS(T/TPER(I))/RADIUS
          YF=YD*TAMPL(I)*COS(T/TPER(I))/RADIUS
          ZF=ZD*TAMPL(I)*COS(T/TPER(I))/RADIUS
       ELSE
          XF=XD*TAMPL(I)/RADIUS
          YF=YD*TAMPL(I)/RADIUS
          ZF=ZD*TAMPL(I)/RADIUS
       ENDIF
       DX(J)=DX(J)-XF
       DY(J)=DY(J)-YF
       DZ(J)=DZ(J)-ZF
       EP=EP-(XF*X(J)+YF*Y(J)+ZF*Z(J))
    ENDDO
    RETURN
  END SUBROUTINE ETORQ2

  SUBROUTINE TORQ2(N,NATOM,TOR,PER,V,QWEIG,ISLCT,WMAIN)

    !-----------------------------------------------------------------------
    use corsubs
    use number
    implicit none
    !
    INTEGER N,NATOM,ISLCT(NATOM)
    LOGICAL QWEIG
    REAL(chm_real) TOR,PER,V(6)
    REAL(chm_real) WMAIN(NATOM),CG(NATOM)
    !
    INTEGER I,J,JS,JS1
    REAL(chm_real) XX,X0,X1,Y0,Y1,Z0,Z1
    !
    XX=TOR*AKMPN
    JS=0
    IF(QTAX)THEN
       X0=AXISCX
       X1=AXISX
       Y0=AXISCY
       Y1=AXISY
       Z0=AXISCZ
       Z1=AXISZ
    ELSE
       X0=V(4)
       X1=V(1)
       Y0=V(5)
       Y1=V(2)
       Z0=V(6)
       Z1=V(3)
    ENDIF
    DO I=1+NTORQ,N+NTORQ
       TPER(I)=PER
       TAMPL(I)=XX
       TAXISX(I)=X1
       TAXISY(I)=Y1
       TAXISZ(I)=Z1
       TAXISCX(I)=X0
       TAXISCY(I)=Y0
       TAXISCZ(I)=Z0

       ! Find atom index of first selected atom
       JS1 = JS + 1
       DO J=JS1,NATOM
          IF(ISLCT(J)== 1)THEN
             JS=J
             ISLCT(J)=0
             EXIT
          ENDIF
       ENDDO
       TTO(I)=JS
       IF(QWEIG) TAMPL(I)=TAMPL(I)*WMAIN(JS)
    ENDDO
    RETURN
  END SUBROUTINE TORQ2

  SUBROUTINE TORQLIST()
    !-----------------------------------------------------------------------
    ! List currently effective torques to OUTU
    !
    use stream
    use chutil,only:atomid
    implicit none
    INTEGER I,J
    CHARACTER*4 ATNAM,RID,SID,REN
    !
    IF(IOLEV .LE. 0 .OR. PRNLEV .LE. 1)RETURN
    IF(NTORQ.LE.0)THEN
       WRITE(OUTU,'(/,A,/)') ' TORQ> No external forces defined.'
       RETURN
    ENDIF
    WRITE(OUTU,801) NTORQ
801 FORMAT(' TORQ> Currently these',I6,' torques are defined', &
         /16X,'Atom      Ampl [AKMA]       [pN]   Per [ps]', &
         '  Direction (X,Y,Z)')
    DO I=1,NTORQ
       J=TTO(I)
       CALL ATOMID(J,SID,RID,REN,ATNAM)
       WRITE(OUTU,901) J,SID,RID,REN,ATNAM,TAMPL(I),TAMPL(I)/AKMPN, &
            TPER(I),TAXISX(I),TAXISY(I),TAXISZ(I)
901    FORMAT(1X,I5,4(1X,A4),1X,F10.4,1X,F10.3,1X,F10.3,3(1X,F5.2))
    ENDDO
    WRITE(OUTU,'(//)')
    RETURN
  END SUBROUTINE TORQLIST


end module pull_mod

