module vibcom
  implicit none
contains

  SUBROUTINE RALEIG(ISTRT,ISTOP,X,Y,Z,NAT3, &
       BNBND,BIMAG,DDV,DQUOT,DDF,DDSCR,LSAVE, &
       DDM)
    !
    ! SETS UP THE SECOND DERIVATIVE MATRIX AND
    ! COMPUTES RALEIGH QUOTIENTS FOR SELECTED MODES
    !
    ! By Bernard R. Brooks   10-MAR-1984
    !
  use chm_kinds
  use chm_types
  use dimens_fcm
  use number
  use deriv
  use consta
  use energym
  use parallel,only: gcomb
  use stream
  use memory

    implicit none

    INTEGER ISTRT,ISTOP,NAT3
    type(nonbondDataStructure) BNBND
    type(imageDataStructure) BIMAG
    real(chm_real) :: X(:), Y(:), Z(:)
    real(chm_real) DDV(NAT3,*),DQUOT(*),DDF(*),DDSCR(*),DDM(*)
    LOGICAL LSAVE

    real(chm_real),allocatable,dimension(:) :: DD1
    INTEGER NATOM,N6,I,K
    real(chm_real) QUOT,FREQ

    NATOM=NAT3/3

    N6=(NAT3*(NAT3+1))/2
    call chmalloc('vibcom.src','RALEIG','DD1',N6,crl=DD1)
    DD1(1:N6)=0.0

    CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,1,NAT3,DD1)
#if KEY_PARALLEL==1
    CALL VDGBR(DX,DY,DZ,1)
    CALL GCOMB(DD1,N6)
#endif 

    IF(PRNLEV.GT.2) THEN
       CALL PRINTE(OUTU, EPROP, ETERM, 'VIBR', 'ENR', .TRUE., &
            1, ZERO, ZERO, .TRUE.)
    ENDIF

    CALL RAISE(.FALSE.,.FALSE.,NAT3,NATOM,.FALSE.,0,DD1,DDM, &
         0,0,0,0,0,0,0,.FALSE.,.TRUE.)

    DO I=ISTRT,ISTOP
       CALL RALEG2(DDV(1,I),DDSCR,NATOM,DD1)
       QUOT=0.0
       DO K=1,NAT3
          QUOT=QUOT+ DDV(K,I)*DDSCR(K)
       ENDDO
       FREQ=CNVFRQ*SQRT(ABS(QUOT))
       IF(QUOT.LT.0.0) FREQ=-FREQ

       IF(PRNLEV.GT.2) WRITE(OUTU,22) I,QUOT,FREQ
22     FORMAT(' MODE=',I4,' QUOTIENT=',F12.6,' FREQUENCY=',F12.4)

       IF(LSAVE) THEN
          DQUOT(I)=QUOT
          DDF(I)=FREQ
       ENDIF
    ENDDO
    call chmdealloc('vibcom.src','RALEIG','DD1',N6,crl=DD1)

    RETURN
  END SUBROUTINE RALEIG

  SUBROUTINE REMATM(NFREQ,NAT3,NATNEW,ISLCT, &
       DDV,DDSCR,AMASS,AMASSN)

  use chm_kinds
    implicit none
    INTEGER NFREQ,NAT3,NATNEW
    INTEGER ISLCT(*)
    !CCCCCCCCCCC DDV(NAT3,NFREQ)
    real(chm_real) DDV(*),DDSCR(*)
    real(chm_real) AMASS(*),AMASSN(*)

    INTEGER NATOM,NSEL,I,ISEL,IATOM,IPT,JPT,IFREQ
    !
    !CC      WRITE(6,44) 'NFREQ,NAT3',NFREQ,NAT3
    !CC      WRITE(6,44) 'ISLCT',(ISLCT(I),I=1,20)
    !CC  44  FORMAT(A/20I5)
    !CC      WRITE(6,45) 'DDV',(DDV(I),I=1,60)
    !CC  45  FORMAT(A/(3F12.6))
    !CC      WRITE(6,45) 'AMASS',(AMASS(I),I=1,20)
    !
    NATOM=NAT3/3
    NSEL=0
    DO I=1,NATOM
       IF(ISLCT(I).EQ.0) THEN
          NSEL=NSEL+1
          AMASSN(NSEL)=AMASS(I)
       ELSE
          AMASSN(NSEL)=AMASSN(NSEL)+AMASS(I)
       ENDIF
    ENDDO
    NATNEW=NSEL*3
    !
    !CC      WRITE(6,44) 'NSEL',NSEL
    !CC      WRITE(6,45) 'AMASSN',(AMASSN(I),I=1,20)
    !
    DO IFREQ=1,NFREQ
       ISEL=0
       JPT=(IFREQ-1)*NAT3-2
       IPT=-2
       DO IATOM=1,NATOM
          JPT=JPT+3
          IF(ISLCT(IATOM).EQ.0) THEN
             ISEL=ISEL+1
             IPT=IPT+3
             DDSCR(IPT)=DDV(JPT)*SQRT(AMASS(IATOM)/AMASSN(ISEL))
             DDSCR(IPT+1)=DDV(JPT+1)*SQRT(AMASS(IATOM)/AMASSN(ISEL))
             DDSCR(IPT+2)=DDV(JPT+2)*SQRT(AMASS(IATOM)/AMASSN(ISEL))
          ELSE
             DDSCR(IPT)  =DDSCR(IPT)+DDV(JPT) &
                  *SQRT(AMASS(IATOM)/AMASSN(ISEL))
             DDSCR(IPT+1)=DDSCR(IPT+1)+DDV(JPT+1) &
                  *SQRT(AMASS(IATOM)/AMASSN(ISEL))
             DDSCR(IPT+2)=DDSCR(IPT+2)+DDV(JPT+2) &
                  *SQRT(AMASS(IATOM)/AMASSN(ISEL))
          ENDIF

       ENDDO

       IPT=(IFREQ-1)*NATNEW
       DO I=1,NATNEW
          DDV(IPT+I)=DDSCR(I)
       ENDDO
    ENDDO
    !
    !CC      WRITE(6,45) 'DDVF',(DDV(I),I=1,60)
    !
    RETURN
  END SUBROUTINE REMATM

  SUBROUTINE WRTSCD(X,Y,Z,XNEW,YNEW,ZNEW,NAT3,DDM,LCARD,ISLCT, &
       BNBND,BIMAG,NFREQ,LNOMA,AMASS,LFINIT,STEP,TOL, &
       IUNSCD,LRAISE)
    !
    ! WRITES OUT THE SECOND DERIVATIVE MATRIX IN COMPRESSED FORM
    !
    ! By Bernard R. Brooks    1982
    !
    use chm_kinds
    use chm_types
    use dimens_fcm
    use number
    use ctitla
    use deriv
    use exfunc
    use energym
    use memory
    use parallel,only: gcomb
    use stream
    implicit none

    INTEGER NAT3,NFREQ,IUNSCD
    type(nonbondDataStructure) BNBND
    type(imageDataStructure) BIMAG
    real(chm_real) :: X(:), Y(:), Z(:), XNEW(:), YNEW(:), ZNEW(:)
    real(chm_real) AMASS(*)
    real(chm_real) STEP,TOL

    INTEGER ICNTRL(20)

    real(chm_real) DDM(*)
    LOGICAL LNOMA,LRAISE,LFINIT,LCARD
    LOGICAL QMASWT
    INTEGER ISLCT(*)

    INTEGER NATOM
    real(chm_real),allocatable,dimension(:) :: DDS,DD1
    INTEGER NIJ,I,K,PRLEV

    CHARACTER(len=4) :: HDRS
    DATA HDRS/'SECD'/

    NATOM=NAT3/3
    QMASWT=.NOT.LNOMA
    PRLEV=PRNLEV
    IF(IUNSCD.NE.OUTU) PRLEV=IOLEV

    NIJ=(NAT3*(NAT3+1))/2
    call chmalloc('vibio.src','WRTSCD','DD1',NIJ,crl=DD1)
    DD1(1:NIJ)=0.0

    CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,1,NAT3,DD1)
#if KEY_PARALLEL==1
    CALL VDGBR(DX,DY,DZ,1)
    CALL GCOMB(DD1,NIJ)
#endif 

    IF(PRNLEV.GE.2) THEN
       CALL PRINTE(OUTU, EPROP, ETERM, 'VIBR', 'ENR', .TRUE., &
            1, ZERO, STEP, .TRUE.)

       WRITE(OUTU,205) IUNSCD
205    FORMAT(' WRITING THE SECOND DERIVATIVE MATRIX TO UNIT',I5)
    ENDIF

    IF(LCARD) THEN
       ! write out complete matrix
       call chmalloc('vibio.src','WRTSCD','DDS',NAT3*6,crl=DDS)
       CALL RAISE(LRAISE,LRAISE,NAT3,NATOM,QMASWT,AMASS, &
            DD1,DDM,DDS,X,Y,Z,0,0,0,.FALSE.,QMASWT)

       IF(LFINIT) THEN
          CALL GENSD(IUNSCD,PRLEV,NAT3,X,Y,Z,XNEW,YNEW,ZNEW,DD1, &
               DDS,STEP,TOL,ISLCT)
       ELSE
          IF(PRLEV.LT.0) RETURN
          WRITE(IUNSCD,725) NATOM
725       FORMAT(I5)
          WRITE(IUNSCD,726) EPROP(EPOT)
726       FORMAT(3F20.10)
          WRITE(IUNSCD,726) (DX(K),DY(K),DZ(K),K=1,NATOM)
          WRITE(IUNSCD,727) (DD1(K),K=1,NIJ)
          WRITE(IUNSCD,726) (X(K),Y(K),Z(K),K=1,NATOM)
727       FORMAT(F20.10)
       ENDIF
       call chmdealloc('vibio.src','WRTSCD','DDS',NAT3*6,crl=DDS)
    ELSE
       ! write out binary compressed matrix
       ICNTRL(1)=IABS(NFREQ)
       ICNTRL(2)=NAT3
       ICNTRL(3)=NATOM
       ICNTRL(4:20)=0

       IF(IOLEV.LT.0) RETURN
       WRITE(IUNSCD) HDRS,ICNTRL
       CALL WRTITL(TITLEA,NTITLA,IUNSCD,-1)
       WRITE(IUNSCD) (DD1(I),I=1,NIJ)
       WRITE(IUNSCD) (AMASS(I),I=1,NATOM)
    ENDIF
    call chmdealloc('vibio.src','WRTSCD','DD1',NIJ,crl=DD1)

    RETURN
  END SUBROUTINE WRTSCD

  SUBROUTINE GENSD(IUNIT,PRLEV,NAT3,X,Y,Z,XNEW,YNEW,ZNEW,DD1,DER, &
       STEP,TOL,ISLCT)
    !
    ! ROUTINE TO GENERATE A SECOND DERIVATIVE BY FINITE DIFFERENCES.
    !
    ! By Bernard R. Brooks    1982
    !
    use chm_kinds
    use dimens_fcm
    use deriv
    use energym
    use eutil, only: gete
    use memory
    use param_store, only: set_param

    implicit none

    INTEGER IUNIT,PRLEV,NAT3
    real(chm_real) CR(3),DER(NAT3),DD1(*)
    real(chm_real) :: X(:), Y(:), Z(:), XNEW(:), YNEW(:), ZNEW(:)
    real(chm_real) STEP,TOL
    INTEGER ISLCT(*)

    real(chm_real),allocatable,dimension(:,:) :: DD
    real(chm_real),allocatable,dimension(:) :: DXFD,DXAD
    real(chm_real) VALMIN,DEL,DEL2,DIF,VAL,TRACE
    INTEGER NATOM,IPT,IAT,IAT3,I1,JAT,J1,I,J,NOK
    CHARACTER(len=2) :: SNAME(3)
    DATA SNAME/' X',' Y',' Z'/

    VALMIN=1.0
    DEL=STEP
    DEL2=DEL+DEL
    NATOM=NAT3/3

    call chmalloc('vibio.src','GENSD','DD',NAT3,NAT3,crl=DD)
    call chmalloc('vibio.src','GENSD','DXFD',NAT3,crl=DXFD)
    call chmalloc('vibio.src','GENSD','DXAD',NAT3,crl=DXAD)

    ! save current forces
    IPT=0
    DO I=1,NATOM
       XNEW(I)=X(I)
       YNEW(I)=Y(I)
       ZNEW(I)=Z(I)
       IF(ISLCT(I).GT.0) THEN
          DXAD(IPT+1)=DX(I)
          DXAD(IPT+2)=DY(I)
          DXAD(IPT+3)=DZ(I)
       ENDIF
       IPT=IPT+3
    ENDDO

    DO IAT=1,NATOM
       IF(ISLCT(IAT).GT.0) THEN
          CR(1)=X(IAT)
          CR(2)=Y(IAT)
          CR(3)=Z(IAT)
          IAT3=(IAT-1)*3
          DO I=1,3
             IAT3=IAT3+1

             CR(I)=CR(I)+DEL
             XNEW(IAT)=CR(1)
             YNEW(IAT)=CR(2)
             ZNEW(IAT)=CR(3)

             CALL GETE(XNEW,YNEW,ZNEW,X,Y,Z,0)
             DXFD(IAT3)=EPROP(EPOT)

             DO J=1,NATOM
                IF(ISLCT(J).GT.0) THEN
                   IPT=J*3-2
                   DER(IPT)=DX(J)
                   DER(IPT+1)=DY(J)
                   DER(IPT+2)=DZ(J)
                ENDIF
             ENDDO

             CR(I)=CR(I)-2.0*DEL
             XNEW(IAT)=CR(1)
             YNEW(IAT)=CR(2)
             ZNEW(IAT)=CR(3)

             CALL GETE(XNEW,YNEW,ZNEW,X,Y,Z,0)

             DXFD(IAT3)=(DXFD(IAT3)-EPROP(EPOT))/DEL2
             DO J=1,NATOM
                IF(ISLCT(J).GT.0) THEN
                   IPT=J*3-2
                   DD(IPT,IAT3)=(DER(IPT)-DX(J))/DEL2
                   DD(IPT+1,IAT3)=(DER(IPT+1)-DY(J))/DEL2
                   DD(IPT+2,IAT3)=(DER(IPT+2)-DZ(J))/DEL2
                ENDIF
             ENDDO
             CR(I)=CR(I)+DEL
             XNEW(IAT)=X(IAT)
             YNEW(IAT)=Y(IAT)
             ZNEW(IAT)=Z(IAT)

          ENDDO
       ENDIF
    ENDDO

    IF(PRLEV.GE.1) WRITE(IUNIT,23) TOL
23  FORMAT(' THE FOLLOWING FIRST DERIVATIVE ELEMENTS DIFFER BY', &
         ' MORE THAN TOL=',F12.6)
    NOK=0
    DO I=1,NAT3
       IAT=(I+2)/3
       IF(ISLCT(IAT).GT.0) THEN
          I1=I-3*IAT+3
          DIF=DXAD(I)-DXFD(I)
          VAL=ABS(DXAD(I))
          IF(VAL.LT.VALMIN) VAL=VALMIN
          IF(ABS(DIF/VAL).GE.TOL) THEN
             IF(PRLEV.GE.1) THEN
                WRITE(IUNIT,28) IAT,SNAME(I1),DXAD(I),DXFD(I),DIF
             ENDIF
28           FORMAT(I4,A2,3F16.8)
          ELSE
             NOK=NOK+1
          ENDIF
       ENDIF
    ENDDO
    IF(PRLEV.GE.1) WRITE(IUNIT,36) NOK

    IF(PRLEV.GE.1) WRITE(IUNIT,24) TOL
24  FORMAT(' THE FOLLOWING SECOND DERIVATIVE ELEMENTS DIFFER BY', &
         ' MORE THAN TOL=',F12.6)
    NOK=0
    DO I=1,NAT3
       IAT=(I+2)/3
       IF(ISLCT(IAT).GT.0) THEN
          I1=I-3*IAT+3
          DO J=I,NAT3
             JAT=(J+2)/3
             IF(ISLCT(JAT).GT.0) THEN
                J1=J-3*JAT+3
                IPT=((2*NAT3-I)*(I-1))/2+J
                DIF=DD1(IPT)-0.5*(DD(I,J)+DD(J,I))
                VAL=ABS(DD1(IPT))
                IF(VAL.LT.VALMIN) VAL=VALMIN
                IF(ABS(DIF/VAL).GE.TOL) THEN
                   IF(PRLEV.GE.1) WRITE(IUNIT,29) IAT,SNAME(I1), &
                        JAT,SNAME(J1), &
                        DD1(IPT),DD(I,J),DD(J,I),DIF,IPT
29                 FORMAT(I4,A2,' -',I3,A2,4F16.8,I10)
                ELSE
                   NOK=NOK+1
                ENDIF
             ENDIF
          ENDDO
       ENDIF
    ENDDO

    call chmdealloc('vibio.src','GENSD','DXAD',NAT3,crl=DXAD)
    call chmdealloc('vibio.src','GENSD','DXFD',NAT3,crl=DXFD)
    call chmdealloc('vibio.src','GENSD','DD',NAT3,NAT3,crl=DD)

    IF(PRLEV.GE.1) WRITE(IUNIT,36) NOK
36  FORMAT(' A TOTAL OF',I6,' ELEMENTS WERE WITHIN THE TOLERANCE')

    ! compute trace of the Hessian for selected atoms.
    TRACE=0.0
    DO I=1,NAT3
       IAT=(I+2)/3
       IF(ISLCT(IAT).GT.0) THEN
          IPT=((2*NAT3-I)*(I-1))/2+I
          TRACE=TRACE+DD1(IPT)
       ENDIF
    ENDDO

    IF(PRLEV.GE.1) WRITE(IUNIT,37) TRACE
37  FORMAT(' THE TRACE OF THE HESSIAN FOR ALL SELECTED ATOMS IS', &
         F16.6)
    call set_param('TRAC',TRACE)

    RETURN
  END SUBROUTINE GENSD

  SUBROUTINE GENSD2(NAT3,X,Y,Z,XREF,YREF,ZREF,DD1,DXF,DYF,DZF,STEP, &
       LDSCF, &
       NSAVDD1,QREST)  ! JZ_UW12
    !
    ! Routine to generate a second derivative by finite differences.
    !
    ! For finite derivatives in DIAG, Benoit Roux 1992
    !
  use chm_kinds
  use dimens_fcm
  use number
  use deriv
  use energym
  use eutil
  use stream
  use psf
  use gamess_fcm, only:qreadhess,qsavehess  /* JZ_UW12*/
#if KEY_SCCDFTB==1
  use sccdftb     /* JZ_UW12*/
#endif

    implicit none

    INTEGER NAT3
    real(chm_real) :: X(:), Y(:), Z(:), XREF(:), YREF(:), ZREF(:)
    real(chm_real) DD1(*)
    real(chm_real) DXF(*),DYF(*),DZF(*)
    real(chm_real) STEP
    LOGICAL LDSCF
    INTEGER NSAVDD1 ! JZ_UW12 
    LOGICAL QREST   ! JZ_UW12

    ! Local variables
    real(chm_real) STEPR
    INTEGER NATOML,IAT,JAT,I,J,IPT,NPT
    CHARACTER(len=3) :: XYZ(3)
    INTEGER ENERCOUNT,NUMCYC,LASTCYC,DUMMY ! JZ_UW12
    LOGICAL READDD1,SAVEDD1 ! JZ_UW12
    XYZ(1) = ' X '
    XYZ(2) = ' Y '
    XYZ(3) = ' Z '

! JZ_UW12: For finite differences restart
      READDD1 = .FALSE.
      SAVEDD1 = .FALSE.
! qreadhess,qsavehess can only be true if qmused_qchem is on => no extra protection needed
      IF (QREADHESS) READDD1 = .TRUE.
      IF (QSAVEHESS) SAVEDD1 = .TRUE.
#if KEY_SCCDFTB==1
      IF (QSCCREADHESS) READDD1 = .TRUE.
      IF (QSCCSAVEHESS) SAVEDD1 = .TRUE.
#endif 

    IF(PRNLEV >= 2) WRITE(OUTU,100)  &
         'SECOND DERIVATIVE CALCULATED BY FINITE DIFFERENCES'
100 FORMAT(1X,A)
    IF(QDRUDE) THEN
       IF(PRNLEV >= 2) WRITE(OUTU,'(A,F10.7)')  &
            ' Numerical increment STEP = ',STEP
       IF(LDSCF) THEN
          IF(PRNLEV >= 2) WRITE(OUTU,'(A,A)') &
               ' Drudes are self-consistently relaxed'
       ELSE
          IF(PRNLEV >= 2) WRITE(OUTU,*) &
               ' Drudes are considered as regular particles'
       ENDIF
    ENDIF

    STEPR=HALF/(STEP+STEP)
    NATOML=NAT3/3

    ! Initialize the second derivative array
    CALL GENIPT(NAT3,NATOML,3,NATOM,3,NPT)
    DO IPT=1,NPT
       DD1(IPT) = ZERO
    ENDDO

    ! JZ_UW12: Read DD1 if requested
      IF (READDD1) THEN
        IF(PRNLEV >= 2) WRITE(OUTU,100) 'READING DD1'
        OPEN(UNIT=99,FILE='dd1.dat',STATUS='UNKNOWN')
        REWIND(99)

        DO IPT=1,NPT
          READ(99,52)DD1(IPT)
        ENDDO
        CLOSE(99)

        ! Save the coordinates as the reference state
        DO IAT=1,NATOML
           XREF(IAT) = X(IAT)
           YREF(IAT) = Y(IAT)
           ZREF(IAT) = Z(IAT)
        ENDDO

        ! Recompute the energy just to get the correct forces.
        CALL GETE(X,Y,Z,XREF,YREF,ZREF,0)

        ! Leave here
        RETURN
      ENDIF

    ! Save the coordinates as the reference state
    ENERCOUNT = 0 ! JZ_UW12
    DO IAT=1,NATOML
       XREF(IAT) = X(IAT)
       YREF(IAT) = Y(IAT)
       ZREF(IAT) = Z(IAT)
       IF (.NOT.(LDSCF.AND.ISDRUDE(IAT))) ENERCOUNT = ENERCOUNT + 1
    ENDDO

    ! JZ_UW12: Print number of energy evaluations
    IF(PRNLEV >= 2) THEN
      WRITE(OUTU,102) 'NUMBER OF ENERGY EVALUATIONS: ', 6*ENERCOUNT + 1
      WRITE(OUTU,102) 'TOTAL NUMBER OF CYCLES:       ', 3*ENERCOUNT
    ENDIF
102 FORMAT(1X,A,I8)

    ! JZ_UW12: If this is a restart run, read cycle where
    !          to start
    IF (QREST) THEN
      OPEN(UNIT=99,FILE='dd1_save.dat',STATUS='UNKNOWN')
      REWIND(99)
      READ(99,51)LASTCYC
      CLOSE(99)
    ENDIF

    NUMCYC = 0 ! JZ_UW12
    DO IAT=1,NATOML
       ! Exclude Drude particles from numerical differentiation (if LDSCF==.TRUE.)
       IF(.NOT.(LDSCF.AND.ISDRUDE(IAT))) THEN
          DO I=1,3

             ! JZ_UW12: Logic for restart
             IF (QREST) THEN
               IF (NUMCYC .LT. LASTCYC) THEN
                 NUMCYC = NUMCYC + 1
                 CYCLE
               ELSEIF (NUMCYC .EQ. LASTCYC) THEN
                 IF(PRNLEV >= 2) &
                   WRITE(OUTU,103) 'RETRIEVING DD1       (LAST CYCLE ', LASTCYC, ')' 
                 OPEN(UNIT=99,FILE='dd1_save.dat',STATUS='UNKNOWN')
                 REWIND(99)
                 READ(99,51)DUMMY
                 DO IPT=1,NPT
                   READ(99,52)DD1(IPT)
                 ENDDO
                 CLOSE(99)
               ENDIF
             ENDIF

             ! Get and store the forward values of the first derivatives
             ! I = 1,2,3 gives X,Y,Z finite derivatives
             IF(I.EQ.1) THEN
                X(IAT)=XREF(IAT)+STEP
             ELSE IF(I.EQ.2) THEN
                Y(IAT)=YREF(IAT)+STEP
             ELSE
                Z(IAT)=ZREF(IAT)+STEP
             ENDIF

             ! Relax Drudes self-consistently
             IF(LDSCF) CALL DRELAX()
             CALL GETE(X,Y,Z,XREF,YREF,ZREF,0)

             DO JAT=1,NATOML
                DXF(JAT) = DX(JAT)
                DYF(JAT) = DY(JAT)
                DZF(JAT) = DZ(JAT)
             ENDDO

             ! Get and store the backward values of the first derivatives
             ! I = 1,2,3 gives X,Y,Z finite derivatives
             IF(I.EQ.1)THEN
                X(IAT)=XREF(IAT)-STEP
             ELSE IF(I.EQ.2)THEN
                Y(IAT)=YREF(IAT)-STEP
             ELSE
                Z(IAT)=ZREF(IAT)-STEP
             ENDIF

             ! Relax Drudes self-consistently
             IF(LDSCF) CALL DRELAX()
             CALL GETE(X,Y,Z,XREF,YREF,ZREF,0)

             ! Symmetrize and store the finite second derivative
             DO JAT=1,NATOM
                CALL GENIPT(NAT3,IAT,I,JAT,1,IPT)
                DD1(IPT) = DD1(IPT) + (DXF(JAT)-DX(JAT))*STEPR
                CALL GENIPT(NAT3,IAT,I,JAT,2,IPT)
                DD1(IPT) = DD1(IPT) + (DYF(JAT)-DY(JAT))*STEPR
                CALL GENIPT(NAT3,IAT,I,JAT,3,IPT)
                DD1(IPT) = DD1(IPT) + (DZF(JAT)-DZ(JAT))*STEPR
             ENDDO

             ! Restore the positions
             X(IAT)=XREF(IAT)
             Y(IAT)=YREF(IAT)
             Z(IAT)=ZREF(IAT)

             ! JZ_UW12: Save DD1 at specific intervals for restart
             NUMCYC = NUMCYC + 1
             IF (MOD(NUMCYC,NSAVDD1).EQ.0 .AND. NUMCYC.NE.3*ENERCOUNT) THEN
               IF(PRNLEV >= 2) &
                 WRITE(OUTU,103) 'INTERMEDIATELY SAVING DD1 (CYCLE ', NUMCYC, ')'
               OPEN(UNIT=99,FILE='dd1_save.dat',STATUS='UNKNOWN')
               REWIND(99)

               WRITE(99,51)NUMCYC
               DO IPT=1,NPT
                 WRITE(99,52)DD1(IPT)
               ENDDO
               ENDFILE(99)
               CLOSE(99)
             ENDIF

          ENDDO
       ENDIF
    ENDDO

    ! Fix the diagonal terms
    DO IAT=1,NATOM
       DO I=1,3
          CALL GENIPT(NAT3,IAT,I,IAT,I,IPT)
          DD1(IPT) = TWO*DD1(IPT)
       ENDDO
    ENDDO

    IF(PRNLEV.GE.9)THEN
       DO IPT=1,NPT
          WRITE(OUTU,101) IPT, DD1(IPT)
101       FORMAT(1X,I5,F20.10)
       ENDDO
    ENDIF

    ! Recompute the energy just to get the correct forces.
    CALL GETE(X,Y,Z,XREF,YREF,ZREF,0)

    ! JZ_UW12: Save DD1 if requested
    IF (SAVEDD1) THEN
      IF(PRNLEV >= 2) WRITE(OUTU,100) 'SAVING DD1'
      OPEN(UNIT=99,FILE='dd1.dat',STATUS='UNKNOWN')
      REWIND(99)

      DO IPT=1,NPT
        WRITE(99,52)DD1(IPT)
      ENDDO
      ENDFILE(99)
      CLOSE(99)
    ENDIF

 51 FORMAT(I10)
 52 FORMAT(F25.18)
103 FORMAT(1X,A,I5,A)

    RETURN
  END SUBROUTINE GENSD2

  SUBROUTINE GENIPT(NAT3,IAT,I,JAT,J,IPT)
    ! Returns the upper triangular matrix pointer for second derivative
  use chm_kinds
    implicit none
    INTEGER NAT3, IAT, I, JAT, J, IPT
    INTEGER I2, J2

    I2=(IAT-1)*3+I
    J2=(JAT-1)*3+J

    IF(I2.LE.J2) THEN
       IPT=((2*NAT3-I2)*(I2-1))/2+J2
    ELSE
       IPT=((2*NAT3-J2)*(J2-1))/2+I2
    ENDIF
    RETURN
  END SUBROUTINE GENIPT

  SUBROUTINE TESTSD(COMLYN,COMLEN,NATOM,X,Y,Z,WMAIN,DDA,DDF, &
       XNEW,YNEW,ZNEW,DXF,DYF,DZF, &
       islct, jslct)
    !-----------------------------------------------------------------------
    !     ROUTINE TO GENERATE A SECOND DERIVATIVE BY FINITE DIFFERENCES.
    !
    !     For finite derivatives in DIAG, Benoit Roux 1992
    !
    !     Modifications and incorporation in TEST by Arnaud Blondel 1994
    !
    use chm_kinds
    use dimens_fcm
    use number
    use deriv
    use bases_fcm
    use energym
    use eutil
    use select
    use parallel,only: gcomb
    use stream
    use string
    implicit none

    CHARACTER(len=*) COMLYN
    INTEGER COMLEN,NATOM,NAT3,IUNIT
    INTEGER ISLCT(*),JSLCT(*)
    real(chm_real) :: X(:), Y(:), Z(:), WMAIN(*), XNEW(:), YNEW(:), ZNEW(:)
    real(chm_real) :: DDA(:), DDF(*)
    real(chm_real) DXF(*),DYF(*),DZF(*)
    real(chm_real) STEP, TOL1
    !
    ! Local variables
    real(chm_real) STEP2, DIF, VAL, VALMIN
    INTEGER IAT,JAT,I,J,IPT,NPT
    INTEGER NOK, NOT, JD
    logical :: qprint, err
    CHARACTER(len=3) XYZ(3)
    XYZ(1) = ' X '
    XYZ(2) = ' Y '
    XYZ(3) = ' Z '
    VALMIN = 1.0
    !
    STEP=GTRMF(COMLYN,COMLEN,'STEP',PT005)
    TOL1=GTRMF(COMLYN,COMLEN,'TOL',PT0001)
    IUNIT=GTRMI(COMLYN,COMLEN,'UNIT',OUTU)
    CALL SELCTD(COMLYN,COMLEN,ISLCT,JSLCT, &
         X,Y,Z,WMAIN,.TRUE.,ERR)
    IF(ERR) RETURN

    QPRINT=(IUNIT.EQ.OUTU .AND. PRNLEV.GE.2) .OR. &
         (IUNIT.NE.OUTU .AND. IOLEV.GT.0)
    !
    IF(QPRINT) WRITE(IUNIT,100) &
         'SECOND DERIVATIVE CALCULATED BY FINITE DIFFERENCES'
100 FORMAT(1X,A)
    STEP2=STEP+STEP
    NAT3=NATOM*3

    ! Initialize the second derivative array
    CALL GENIPT(NAT3,NATOM,3,NATOM,3,NPT)
    DO IPT=1,NPT
       DDF(IPT) = ZERO
    ENDDO
    !
    DO IPT=1,NPT
       DDA(IPT) = ZERO
    ENDDO

    ! Get analytic second derivatives [and update nb-list].
    CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,0,NAT3,DDA)

#if KEY_PARALLEL==1
    CALL GCOMB(DDA,NPT)  
#endif

    !...##IF PARALLEL
    !        CALL VDGBR(DX,DY,DZ,1)
    !...##ENDIF

    ! Initialize the coordinates
    DO IAT=1,NATOM
       XNEW(IAT) = X(IAT)
       YNEW(IAT) = Y(IAT)
       ZNEW(IAT) = Z(IAT)
    ENDDO

    DO IAT=1,NATOM
       if(islct(iat) == 1) then
          DO I=1,3
             !     Get and store the forward values of the first derivatives
             !     I = 1,2,3 gives X,Y,Z finite derivatives
             IF(I.EQ.1)THEN
                XNEW(IAT)=X(IAT)+STEP
             ELSEIF(I.EQ.2)THEN
                YNEW(IAT)=Y(IAT)+STEP
             ELSEIF(I.EQ.3)THEN
                ZNEW(IAT)=Z(IAT)+STEP
             ENDIF

             CALL GETE(XNEW,YNEW,ZNEW,X,Y,Z,0)

             DO JAT=1,NATOM
                if(jslct(jat) == 1) then
                   DXF(JAT) = DX(JAT)
                   DYF(JAT) = DY(JAT)
                   DZF(JAT) = DZ(JAT)
                endif
             ENDDO

             !     Get and store the backward values of the first derivatives
             !     I = 1,2,3 gives X,Y,Z finite derivatives
             IF(I.EQ.1)THEN
                XNEW(IAT)=X(IAT)-STEP
             ELSEIF(I.EQ.2)THEN
                YNEW(IAT)=Y(IAT)-STEP
             ELSEIF(I.EQ.3)THEN
                ZNEW(IAT)=Z(IAT)-STEP
             ENDIF

             CALL GETE(XNEW,YNEW,ZNEW,X,Y,Z,0)

             !     Symmetrize and store the finite second derivative
             DO JAT=1,NATOM
                if(jslct(jat) == 1) then

                   CALL GENIPT(NAT3,IAT,I,JAT,1,IPT)
                   DDF(IPT) = DDF(IPT) + 0.5*(DXF(JAT)-DX(JAT))/STEP2

                   CALL GENIPT(NAT3,IAT,I,JAT,2,IPT)
                   DDF(IPT) = DDF(IPT) + 0.5*(DYF(JAT)-DY(JAT))/STEP2

                   CALL GENIPT(NAT3,IAT,I,JAT,3,IPT)
                   DDF(IPT) = DDF(IPT) + 0.5*(DZF(JAT)-DZ(JAT))/STEP2
                endif
             ENDDO

             !     Restore the positions
             XNEW(IAT)=X(IAT)
             YNEW(IAT)=Y(IAT)
             ZNEW(IAT)=Z(IAT)
          ENDDO
       endif
    ENDDO

    !     Fix the diagonal terms
    DO IAT=1,NATOM
       if(islct(iat) == 1) then
          DO I=1,3
             CALL GENIPT(NAT3,IAT,I,IAT,I,IPT)
             DDF(IPT) = 2*DDF(IPT)
          ENDDO
       endif
    ENDDO

    ! Print out of the differences.
    IF(QPRINT) WRITE(IUNIT,23) STEP, TOL1
23  FORMAT(' TESTSD: Parameters: STEP=',F10.5, &
         /,' TESTSD: The following second derivatives', &
         ' differ by more than TOL=',F12.6,//, &
         ' #ATOM1 DIM #ATOM2 DIM  POINTER',6X,'ANALYTIC', &
         5X,'FINITE-DIFF',7X,'DEVIATION')
    NOK=0
    NOT=0
    DO IAT=1,NATOM
       if(islct(iat) == 1) then
          DO JAT=IAT,NATOM
             if(jslct(jat) == 1) then
                DO I=1,3
                   JD=1
                   IF (JAT.EQ.IAT) JD=I
                   DO J=JD,3
                      NOT=NOT+1
                      CALL GENIPT(NAT3,IAT,I,JAT,J,IPT)
                      DIF=DDA(IPT)-DDF(IPT)
                      VAL=ABS(DDA(IPT))
                      IF(VAL.LT.VALMIN) VAL=VALMIN
                      IF(ABS(DIF/VAL).GE.TOL1) THEN
                         IF(QPRINT) WRITE(IUNIT,28)  &
                              IAT,XYZ(I),JAT,XYZ(J),IPT, &
                              DDA(IPT),DDF(IPT),DIF
28                       FORMAT(3X,I4,2X,A2,3X,I4,2X,A2,1X,I6,3F16.8)
                      ELSE
                         NOK=NOK+1
                      ENDIF
                   ENDDO
                ENDDO
             endif
          ENDDO
       endif
    ENDDO

    IF(QPRINT) WRITE(IUNIT,36) NOK, NOT
36  FORMAT(/ &
         ' TESTSD: A total of',I6,' elements out of ',I6, &
         ' were within the tolerance')
    RETURN
  END SUBROUTINE TESTSD

  SUBROUTINE DRELAX()
    ! Purpose:
    ! Optimize coordinates of Drude particles to minimize potential energy.
    ! Coordinates of real atoms (non-Drude) are fixed.
    ! Energy minimization employs ABNR algorithm.
    ! Authors:
    !    Guillaume Lamoureux (2000-2001)
    !    Victor Anisimov, 2004
    !
  use abnerm,only:abner

  use chm_kinds
  use dimens_fcm
  use exfunc
  use number

  use coord
  use psf
  use memory
  use stream
  use string
    implicit none

    ! Local variables
    INTEGER IA
    CHARACTER(len=80) :: COMLYN2
    INTEGER COMLEN2
    INTEGER SAVEPR
    integer,allocatable,dimension(:) :: SAVEMV

    IF(NDRUDE.EQ.0) RETURN

    ! Only Drude particles are allowed to move
    call chmalloc('vibcom.src','DRELAX','SAVEMV',NATOM,intg=SAVEMV)
    SAVEMV(1:NATOM)=IMOVE(1:NATOM)
    DO IA=1,NATOM
       IF(ISDRUDE(IA)) THEN
          ! Make mobile the Drude particle
          IMOVE(IA)=0
       ELSE
          ! Fix the Atom
          IMOVE(IA)=1
       ENDIF
    ENDDO

    ! Save Print Level
    SAVEPR=PRNLEV
    PRNLEV=0

    ! Minimize energy
    COMLYN2='STEP 0.0001 NSTEPS 1000'
    COMLEN2=STRLNG(COMLYN2)
    CALL ABNER(COMLYN2,COMLEN2)
    !*      comlyn2 = 'ABNR STEP 0.0001 NSTEPS 1000'
    !*      comlen2 = strlen(comlyn2)
    !*      call MINMIZ(comlyn2,comlen2)

    ! Restore Print Level
    PRNLEV=SAVEPR

    IMOVE(1:NATOM)=SAVEMV(1:NATOM)

    call chmdealloc('vibcom.src','DRELAX','SAVEMV',NATOM,intg=SAVEMV)
    RETURN
  END SUBROUTINE DRELAX

  ! QC: UW_06
  SUBROUTINE GENSD3(NATOM,NVAR,IMOVE,X,Y,Z,XREF,YREF,ZREF,DD1, &
       DXF,DYF,DZF,STEP, &
       NSAVDD1,QREST) ! JZ_UW12
    !
    ! Routine to generate a second derivative by finite differences.
    !
    ! For finite derivatives in Redu, Q. Cui, 1999.
    ! With FROZEN ATOMS.
    !
  use chm_kinds
  use dimens_fcm
  use number
  use deriv
  use energym
  use eutil
  use stream
  use gamess_fcm,only:qreadhess,qsavehess  /* JZ_UW12*/
#if KEY_SCCDFTB==1
  use sccdftb     /* JZ_UW12*/
#endif

    implicit none

    INTEGER NATOM,NVAR,IMOVE(*)
    real(chm_real) :: X(:), Y(:), Z(:), XREF(:), YREF(:), ZREF(:)
    real(chm_real) DD1(*)
    real(chm_real) DXF(*),DYF(*),DZF(*)
    real(chm_real) STEP
    INTEGER NSAVDD1 ! JZ_UW12
    LOGICAL QREST   ! JZ_UW12

    ! Local variables
    real(chm_real) STEPR
    INTEGER NAT3,IAT,JAT,I,J,IPT,NPT
    INTEGER IATMVE, JATMVE
    CHARACTER(len=3) :: XYZ(3)
    INTEGER ENERCOUNT,NUMCYC,LASTCYC,DUMMY ! JZ_UW12
    LOGICAL READDD1,SAVEDD1 ! JZ_UW12
    XYZ(1) = ' X '
    XYZ(2) = ' Y '
    XYZ(3) = ' Z '

    ! JZ_UW12: Finite differences restart
    READDD1 = .FALSE.
    SAVEDD1 = .FALSE.
! qreadhess,qsavehess can only be true if qmused_qchem is on => no extra protection needed
    IF (QREADHESS) READDD1 = .TRUE.
    IF (QSAVEHESS) SAVEDD1 = .TRUE.
#if KEY_SCCDFTB==1
    IF (QSCCREADHESS) READDD1 = .TRUE.
    IF (QSCCSAVEHESS) SAVEDD1 = .TRUE.
#endif 

    IF(PRNLEV >= 2) WRITE(OUTU,100)  &
         'SECOND DERIVATIVE CALCULATED BY FINITE DIFFERENCES'
100 FORMAT(1X,A)
    STEPR=HALF/(STEP+STEP)
    NAT3=NATOM*3

    ! Initialize the second derivative array
    CALL GENIPT(NVAR,NVAR/3,3,NVAR/3,3,NPT)
    DO IPT=1,NPT
       DD1(IPT) = ZERO
    ENDDO

    ! JZ_UW12: Read DD1 if requested
    IF (READDD1) THEN
      IF(PRNLEV >= 2) WRITE(OUTU,100) 'READING DD1'
      OPEN(UNIT=99,FILE='dd1.dat',STATUS='UNKNOWN')
      REWIND(99)

      DO IPT=1,NPT
        READ(99,52)DD1(IPT)
      ENDDO
      CLOSE(99)

    ! Save the coordinates as the reference state
      DO IAT=1,NATOM
         XREF(IAT) = X(IAT)
         YREF(IAT) = Y(IAT)
         ZREF(IAT) = Z(IAT)
      ENDDO

    ! Recompute the energy just to get the correct forces.
      CALL GETE(X,Y,Z,XREF,YREF,ZREF,0)

    ! Leave here
      RETURN
    ENDIF

    ! Save the coordinates as the reference state
    ENERCOUNT = 0 ! JZ_UW12
    DO IAT=1,NATOM
       XREF(IAT) = X(IAT)
       YREF(IAT) = Y(IAT)
       ZREF(IAT) = Z(IAT)
       IF (IMOVE(IAT).EQ.0) ENERCOUNT = ENERCOUNT + 1
    ENDDO

    ! JZ_UW12: Print number of energy evaluations
    IF(PRNLEV >= 2) THEN
      WRITE(OUTU,102) 'NUMBER OF ENERGY EVALUATIONS: ', 6*ENERCOUNT + 1
      WRITE(OUTU,102) 'TOTAL NUMBER OF CYCLES:       ', 3*ENERCOUNT
    ENDIF
102 FORMAT(1X,A,I8)

    IATMVE=0

    ! JZ_UW12: If this is a restart run, read cycle where
    !          to start
    IF (QREST) THEN
      OPEN(UNIT=99,FILE='dd1_save.dat',STATUS='UNKNOWN')
      REWIND(99)
      READ(99,51)LASTCYC
      CLOSE(99)
    ENDIF

    NUMCYC = 0 ! JZ_UW12

    iat_loop: DO IAT=1,NATOM

       IF (IMOVE(IAT).NE.0) cycle iat_loop
       IATMVE=IATMVE + 1
       DO I=1,3

       ! JZ_UW12: Logic for restart
          IF (QREST) THEN
            IF (NUMCYC .LT. LASTCYC) THEN
              NUMCYC = NUMCYC + 1
              CYCLE
            ELSEIF (NUMCYC .EQ. LASTCYC) THEN
              IF(PRNLEV >= 2) &
                WRITE(OUTU,103) 'RETRIEVING DD1       (LAST CYCLE ', LASTCYC, ')'
              OPEN(UNIT=99,FILE='dd1_save.dat',STATUS='UNKNOWN')
              REWIND(99)
              READ(99,51)DUMMY
              DO IPT=1,NPT
                READ(99,52)DD1(IPT)
              ENDDO
              CLOSE(99)
            ENDIF
          ENDIF

          ! Get and store the forward values of the first derivatives
          ! I = 1,2,3 gives X,Y,Z finite derivatives
          IF(I.EQ.1) THEN
             X(IAT)=XREF(IAT)+STEP
          ELSE IF(I.EQ.2) THEN
             Y(IAT)=YREF(IAT)+STEP
          ELSE
             Z(IAT)=ZREF(IAT)+STEP
          ENDIF
          CALL GETE(X,Y,Z,XREF,YREF,ZREF,0)

          DO JAT=1,NATOM
             DXF(JAT) = DX(JAT)
             DYF(JAT) = DY(JAT)
             DZF(JAT) = DZ(JAT)
          ENDDO

          ! Get and store the backward values of the first derivatives
          ! I = 1,2,3 gives X,Y,Z finite derivatives
          IF(I.EQ.1)THEN
             X(IAT)=XREF(IAT)-STEP
          ELSE IF(I.EQ.2)THEN
             Y(IAT)=YREF(IAT)-STEP
          ELSE
             Z(IAT)=ZREF(IAT)-STEP
          ENDIF

          CALL GETE(X,Y,Z,XREF,YREF,ZREF,0)

          ! Symmetrize and store the finite second derivative
          ! QC: fix for case with frozen atom
          JATMVE=0
          jat_loop: DO JAT=1,NATOM
             IF (IMOVE(JAT).NE.0) cycle jat_loop
             JATMVE=JATMVE + 1
             ! CALL GENIPT(NAT3,IAT,I,JAT,1,IPT)
             CALL GENIPT(NVAR,IATMVE,I,JATMVE,1,IPT)
             DD1(IPT) = DD1(IPT) + (DXF(JAT)-DX(JAT))*STEPR
             ! CALL GENIPT(NAT3,IAT,I,JAT,2,IPT)
             CALL GENIPT(NVAR,IATMVE,I,JATMVE,2,IPT)
             DD1(IPT) = DD1(IPT) + (DYF(JAT)-DY(JAT))*STEPR
             ! CALL GENIPT(NAT3,IAT,I,JAT,3,IPT)
             CALL GENIPT(NVAR,IATMVE,I,JATMVE,3,IPT)
             DD1(IPT) = DD1(IPT) + (DZF(JAT)-DZ(JAT))*STEPR
          ENDDO jat_loop

          ! Restore the positions
          X(IAT)=XREF(IAT)
          Y(IAT)=YREF(IAT)
          Z(IAT)=ZREF(IAT)

          ! JZ_UW12: Save DD1 at specific intervals for restart
          NUMCYC = NUMCYC + 1
          IF (MOD(NUMCYC,NSAVDD1).EQ.0 .AND. NUMCYC.NE.3*ENERCOUNT) THEN
            IF(PRNLEV >= 2) &
              WRITE(OUTU,103) 'INTERMEDIATELY SAVING DD1 (CYCLE ', NUMCYC, ')'
            OPEN(UNIT=99,FILE='dd1_save.dat',STATUS='UNKNOWN')
            REWIND(99)

            WRITE(99,51)NUMCYC
            DO IPT=1,NPT
              WRITE(99,52)DD1(IPT)
            ENDDO
            ENDFILE(99)
            CLOSE(99)
          ENDIF

       ENDDO
    ENDDO iat_loop

    ! Fix the diagonal terms
    IATMVE=0
    DO IAT=1,NATOM
       IF (IMOVE(IAT).EQ.0) THEN
          IATMVE=IATMVE + 1
          DO I=1,3
             ! CALL GENIPT(NAT3,IAT,I,IAT,I,IPT)
             CALL GENIPT(NVAR,IATMVE,I,IATMVE,I,IPT)
             DD1(IPT) = TWO*DD1(IPT)
          ENDDO
       ENDIF
    ENDDO

    IF(PRNLEV.GE.9)THEN
       DO IPT=1,NPT
          WRITE(OUTU,101) IPT, DD1(IPT)
101       FORMAT(1X,I5,F20.10)
       ENDDO
    ENDIF

    ! Recompute the energy just to get the correct forces.
    CALL GETE(X,Y,Z,XREF,YREF,ZREF,0)

    ! JZ_UW12: Save DD1 if requested
    IF (SAVEDD1) THEN
      IF(PRNLEV >= 2) WRITE(OUTU,100) 'SAVING DD1'
      OPEN(UNIT=99,FILE='dd1.dat',STATUS='UNKNOWN')
      REWIND(99)

      DO IPT=1,NPT
        WRITE(99,52)DD1(IPT)
      ENDDO
      ENDFILE(99)
      CLOSE(99)
    ENDIF

 51 FORMAT(I10)
 52 FORMAT(F25.18)
103 FORMAT(1X,A,I5,A)

    RETURN
  END SUBROUTINE GENSD3

  SUBROUTINE GENSD4(NATOM,X,Y,Z,XREF,YREF,ZREF,DD1, &
       DXF,DYF,DZF,STEP,INBCMP,JNBCMP)
    !
    ! Routine to generate a second derivative by finite differences.
    !
    ! For finite derivatives in Redu, Q. Cui, 1999.
    ! With compact hessian option.
    !
  use chm_kinds
  use dimens_fcm
  use number
  use deriv
  use energym
  use stream
    implicit none

    INTEGER NATOM
    real(chm_real) X(*),Y(*),Z(*),XREF(*),YREF(*),ZREF(*)
    real(chm_real) DD1(*)
    real(chm_real) DXF(*),DYF(*),DZF(*)
    real(chm_real) STEP
    INTEGER INBCMP(*),JNBCMP(*)

    WRITE(*,*) "NOT YET IMPLEMENTED"
    ! QC_UW_06 DONE
    RETURN
  END SUBROUTINE GENSD4

end module vibcom

