!CHARMM Element source/vibran/raise.src 1.1
  SUBROUTINE RAISE(QTRANS,QROT,NVAR,NATOM,QMASS,AMASS,DDF,DDM, &
       DDTROT,X,Y,Z,DX,DY,DZ,QCHECK,QMASWT)
    !
    ! Modify the second derivative matrix so that the rotational
    ! and translational modes have large eigenvalues.
    ! Also can be used to mass weight the entire matrix.
    !
    use chm_kinds
    use number
    use vector
    implicit none

    INTEGER   NATOM, NVAR
    LOGICAL   QCHECK, QMASS, QROT, QTRANS
    real(chm_real)    AMASS(*)
    real(chm_real)    DDF(*), DDM(*), DDTROT(NVAR,*), DX(*), DY(*), DZ(*)
    real(chm_real)    X(*), Y(*), Z(*)
    LOGICAL QMASWT

    INTEGER   I, IPT, J, K, NMODE, NTROT, II, JJ
    LOGICAL   QT(3)
    real(chm_real)    AMASST, HFREQ, Q, TF(3), XCM, YCM, ZCM, DDMM
    real(chm_real)    TOL1, TOL2

    ! Perform overall mass weighting if requested
    IF(QMASWT) THEN
       IPT=0
       DO I=1,NATOM
          DO II=1,3
             DDMM=DDM(I)*DDM(I)
             DO JJ=II,3
                IPT=IPT+1
                DDF(IPT)=DDF(IPT)*DDMM
             ENDDO
             DO J=I+1,NATOM
                DDMM=DDM(I)*DDM(J)
                DO JJ=1,3
                   IPT=IPT+1
                   DDF(IPT)=DDF(IPT)*DDMM
                ENDDO
             ENDDO
          ENDDO
       ENDDO
    ENDIF
    !
    ! Determine which modes need to be changed and calculate them.
    !
    TOL1=RSMALL
    TOL2=PT0001
    NMODE = 0

    IF(QTRANS) THEN
       DO I = 1,3
          QT(I) = .TRUE.
       ENDDO
       IF(QCHECK) THEN
          DO I = 1,3
             TF(I) = ZERO
          ENDDO
          DO I = 1,NATOM
             TF(1) = TF(1) + DX(I)
             TF(2) = TF(2) + DY(I)
             TF(3) = TF(3) + DZ(I)
          ENDDO
          DO I = 1,3
             IF (ABS(TF(I)) .GT. TOL2) QT(I) = .FALSE.
          ENDDO
       ENDIF

       DO I = 1,3
          IF (QT(I)) THEN
             NMODE = NMODE + 1
             DO J = 1,NVAR
                DDTROT(J,NMODE) = ZERO
             ENDDO
             IPT = 0
             DO J = 1,NATOM
                IF(I.EQ.1) THEN
                   DDTROT(IPT+1,NMODE) = ONE / DDM(J)
                ELSE IF(I.EQ.2) THEN
                   DDTROT(IPT+2,NMODE) = ONE / DDM(J)
                ELSE
                   DDTROT(IPT+3,NMODE) = ONE / DDM(J)
                ENDIF
                IPT = IPT + 3
             ENDDO
          ENDIF
       ENDDO
    ENDIF

    IF (QROT) THEN
       XCM    = ZERO
       YCM    = ZERO
       ZCM    = ZERO
       AMASST = ZERO
       IF(QMASS) THEN
          DO I = 1,NATOM
             XCM = XCM + X(I) * AMASS(I)
             YCM = YCM + Y(I) * AMASS(I)
             ZCM = ZCM + Z(I) * AMASS(I)
             AMASST = AMASST + AMASS(I)
          ENDDO
       ELSE
          DO I = 1,NATOM
             XCM = XCM + X(I)
             YCM = YCM + Y(I)
             ZCM = ZCM + Z(I)
             AMASST = AMASST + ONE
          ENDDO
       ENDIF
       XCM = XCM/AMASST
       YCM = YCM/AMASST
       ZCM = ZCM/AMASST

       DO I = 1,3
          NMODE = NMODE + 1
          DO J = 1,NVAR
             DDTROT(J,NMODE) = ZERO
          ENDDO
          IPT = 0
          DO J = 1,NATOM
             IF(I.EQ.1) THEN
                DDTROT(IPT+3,NMODE) =  (Y(J)-YCM) / DDM(J)
                DDTROT(IPT+2,NMODE) = -(Z(J)-ZCM) / DDM(J)
             ELSE IF (I.EQ.2) THEN
                DDTROT(IPT+1,NMODE) =  (Z(J)-ZCM) / DDM(J)
                DDTROT(IPT+3,NMODE) = -(X(J)-XCM) / DDM(J)
             ELSE
                DDTROT(IPT+2,NMODE) =  (X(J)-XCM) / DDM(J)
                DDTROT(IPT+1,NMODE) = -(Y(J)-YCM) / DDM(J)
             ENDIF
             IPT = IPT + 3
          ENDDO
       ENDDO
    ENDIF
    !
    ! Check modes and orthogonalise them.
    !
    NTROT = 0
    DO I = 1,NMODE
       DO J = 1,NTROT
          CALL ORTHOG(DDTROT(1,I),DDTROT(1,J),NVAR)
       ENDDO
       CALL DOTPR(DDTROT(1,I),DDTROT(1,I),NVAR,Q)
       IF (Q .GT. TOL1) THEN
          NTROT = NTROT + 1
          CALL NORMALL(DDTROT(1,I),NVAR)
          IF(I.NE.NTROT)ddtrot(1:nvar,ntrot) = DDTROT(1:nvar,I)
       ENDIF
    ENDDO
    !
    ! Remove modes from second derivative matrix.
    !
    IF(NTROT.GT.0) THEN
       IF(QMASS) THEN
          HFREQ = FIFHUN
       ELSE
          HFREQ = FTHSND
       ENDIF
       IPT = 0
       DO I = 1,NVAR
          DO J = I,NVAR
             IPT = IPT+1
             DO K=1,NTROT
                DDF(IPT)=DDF(IPT)+HFREQ*DDTROT(I,K)*DDTROT(J,K)
             ENDDO
          ENDDO
       ENDDO
    ENDIF

    RETURN
  END SUBROUTINE RAISE

  SUBROUTINE FILDDM(DDM,AMASS,NATOM,LNOMA)
    !
    ! THIS ROUTINE FILLS THE DDM ARRAY
    !
    ! By Bernard R. Brooks   1982
    !
  use chm_kinds
  use number
    implicit none
    INTEGER NATOM
    real(chm_real) DDM(*)
    real(chm_real) AMASS(*)
    LOGICAL LNOMA
    INTEGER I

    IF (LNOMA) THEN
       DO I=1,NATOM
          DDM(I)=ONE
       ENDDO
    ELSE
       DO I=1,NATOM
          ! Avoid division by zero for zero-mass Drude particles: Victor Anisimov, 2004
          IF(AMASS(I).GT.ZERO) THEN
             DDM(I)=ONE/SQRT(AMASS(I))
          ELSE
             DDM(I)=ZERO
          ENDIF
       ENDDO
    ENDIF

    RETURN
  END SUBROUTINE FILDDM

  SUBROUTINE RAISE2(DDF,DDM,NATOM,IMOVE,QMASS,NPT)
    !
    ! QC_UW_06: MASS WEIGHT THE HESSIAN AND RAISE THE RX PATH DIRECTION.
    !
    ! JZ_UW12: Added NPT argument
    !
  use chm_kinds
  use number
  use dimens_fcm
  use deriv
    implicit none

    INTEGER   NATOM,NPT,IMOVE(*)
    real(chm_real)    DDF(*), DDM(*)
    LOGICAL   QMASS

    INTEGER   I, J, IPT
    INTEGER   II,JJ   
    real(chm_real)    DDMM

    ! MASS-WEIGHTING IF NECC.
    IF (QMASS) THEN
       IPT=0   
       i_loop: DO I=1,NATOM
          IF (IMOVE(I).NE.0) cycle i_loop
          DO II=1,3
             DDMM=DDM(I)*DDM(I)
             DO JJ=II,3 
                IPT=IPT+1
                DDF(IPT)=DDF(IPT)*DDMM
             ENDDO
             j_loop: DO J=I+1,NATOM
                IF(IMOVE(J).NE.0) cycle j_loop
                DDMM=DDM(I)*DDM(J)
                DO JJ=1,3
                   IPT=IPT+1
                   DDF(IPT)=DDF(IPT)*DDMM
                ENDDO
             ENDDO j_loop
          ENDDO
       ENDDO   i_loop
    ENDIF

    ! JZ_UW12: Check if everything is consistent
    IF (IPT .NE. NPT) THEN
      CALL WRNDIE(-5,'<RAISE2>', &
           'Bug in RAISE2! Reduce compiler opt. level for raise.src')
    ENDIF

    RETURN
  END SUBROUTINE RAISE2
  ! QC_UW_06 Done

  SUBROUTINE RALEG2(V,W,NATOM,DD1)
    !
    ! THIS ROUTINE COMPUTES THE RALEIGH QUOTIENT FOR A SINGLE VECTOR
    !
    ! BERNARD R. BROOKS    10-MAR-1984
    !
  use chm_kinds
    implicit none
    INTEGER NATOM
    real(chm_real) V(*),W(*),DD1(*)

    ! local storage
    INTEGER NAT3,IPT,I,J

    NAT3=NATOM*3
    DO I=1,NAT3
       W(I)=0.0
    ENDDO

    IPT=0
    DO I=1,NAT3
       IPT=IPT+1
       W(I)=W(I)+DD1(IPT)*V(I)
       DO J=I+1,NAT3
          IPT=IPT+1
          W(I)=W(I)+DD1(IPT)*V(J)
          W(J)=W(J)+DD1(IPT)*V(I)
       ENDDO
    ENDDO

    RETURN
  END SUBROUTINE RALEG2

  SUBROUTINE RAISE_BEFOREMW(QTRANS,QROT,NVAR,NATOM,AMASS,DDF, &
             DDTROT,X,Y,Z,DX,DY,DZ,QCHECK)
!        Modify the second derivative matrix so that the rotational
!     and translational modes have large eigenvalues.
!
!     Added by An Ghysels, July 2010
!
!     The Hessian is NOT mass-weighted yet. Therefore the modification
!     becomes more tricky: use mass-weighted trans/rot vectors and
!     un-mass-weight again, such that mass-weighting of this
!     modified Cartesian Hessian, will indeed lead to a (mass-weighted)
!     Hessian with raised trans/rot modes.
!     The routine calculates H_raise:
!
!       Hmw       = M^(-0.5)  H  M^(-0.5)
!       Hmw_raise = Hmw + HFREQ * Dmw Dmw^T
!       H_raise   = H + HFREQ * M^0.5 Dmw Dmw^T M^0.5
!
!     where mw refers to mass-weighted quantities, M refers to mass
!     matrix, H refers to Hessians, HFREQ is the high frequency to
!     which trans/rot modes are raised, D refers to the trans/rot
!     vectors. Note: Dmw vectors should be orthonormalized.
!
!     QTRANS  --  whether translation should be raised
!     QROT  --  whether rotation should be raised
!     DDF  --  second derivative matrix (Hessian)
!     AMASS  --  vector with atom masses
!     X,Y,Z  --  atom positions
!     DX,DY,DZ  --  if different from zero, can be used to flag
!                   the translational modes that should not be considered
!     QCHECK  --  whether to check which trans modes should be considered
!     DDTROT  --  will contain mass-weighted trans/rot modes
!##INCLUDE '~/charmm_fcm/impnon.fcm'
!##INCLUDE '~/charmm_fcm/number.fcm'
  use chm_kinds
  use number
  use vector
  implicit none

  ! passed
  INTEGER   NATOM, NVAR
  LOGICAL   QCHECK, QROT, QTRANS
  REAL(chm_real)    AMASS(*)
  REAL(chm_real)    DDF(*), DDTROT(NVAR,*), DX(*), DY(*), DZ(*)
  REAL(chm_real)    X(*), Y(*), Z(*)
  ! local
  INTEGER   I, IPT, J, K, NMODE, NTROT, II, JJ, I2, J2
  LOGICAL   QT(3)
  REAL(chm_real)    DDM(NATOM)
  REAL(chm_real)    AMASST, HFREQ, Q, TF(3), XCM, YCM, ZCM, DDMM
  REAL(chm_real)    TOL1, TOL2
  !
  ! Calculate root mean squares of masses.
  !
  DO I=1,NATOM
      DDM(I)=AMASS(I)**(0.5)
  ENDDO
  !
  ! Determine which modes need to be changed and calculate them.
  ! These modes are mass-weighted with DDM.
  !
  TOL1=RSMALL
  TOL2=PT0001
  NMODE = 0

  IF(QTRANS) THEN
     DO I = 1,3
        QT(I) = .TRUE.
     ENDDO
     IF(QCHECK) THEN
        DO I = 1,3
           TF(I) = ZERO
        ENDDO
        DO I = 1,NATOM
           TF(1) = TF(1) + DX(I)
           TF(2) = TF(2) + DY(I)
           TF(3) = TF(3) + DZ(I)
        ENDDO
        DO I = 1,3
           IF (ABS(TF(I)) .GT. TOL2) QT(I) = .FALSE.
        ENDDO
     ENDIF

     DO I = 1,3
        IF (QT(I)) THEN
           NMODE = NMODE + 1
           DO J = 1,NVAR
              DDTROT(J,NMODE) = ZERO
           ENDDO
           IPT = 0
           DO J = 1,NATOM
              IF(I.EQ.1) THEN
                 DDTROT(IPT+1,NMODE) = ONE * DDM(J)
              ELSE IF(I.EQ.2) THEN
                 DDTROT(IPT+2,NMODE) = ONE * DDM(J)
              ELSE
                 DDTROT(IPT+3,NMODE) = ONE * DDM(J)
              ENDIF
              IPT = IPT + 3
           ENDDO
        ENDIF
     ENDDO
  ENDIF

  IF (QROT) THEN
     ! first get center of mass
     XCM    = ZERO
     YCM    = ZERO
     ZCM    = ZERO
     AMASST = ZERO
     DO I = 1,NATOM
        XCM = XCM + X(I) * AMASS(I)
        YCM = YCM + Y(I) * AMASS(I)
        ZCM = ZCM + Z(I) * AMASS(I)
        AMASST = AMASST + AMASS(I)
     ENDDO
     XCM = XCM/AMASST
     YCM = YCM/AMASST
     ZCM = ZCM/AMASST

     DO I = 1,3
        NMODE = NMODE + 1
        DO J = 1,NVAR
           DDTROT(J,NMODE) = ZERO
        ENDDO
        IPT = 0
        DO J = 1,NATOM
           IF(I.EQ.1) THEN
              DDTROT(IPT+3,NMODE) =  (Y(J)-YCM) * DDM(J)
              DDTROT(IPT+2,NMODE) = -(Z(J)-ZCM) * DDM(J)
           ELSE IF (I.EQ.2) THEN
              DDTROT(IPT+1,NMODE) =  (Z(J)-ZCM) * DDM(J)
              DDTROT(IPT+3,NMODE) = -(X(J)-XCM) * DDM(J)
           ELSE
              DDTROT(IPT+2,NMODE) =  (X(J)-XCM) * DDM(J)
              DDTROT(IPT+1,NMODE) = -(Y(J)-YCM) * DDM(J)
           ENDIF
           IPT = IPT + 3
        ENDDO
     ENDDO
  ENDIF
  !
  ! DDTROT now contains the mass-weighted trans/rot modes.
  ! Check modes and orthogonalise them and normalize them.
  !
  NTROT = 0
  DO I = 1,NMODE
     DO J = 1,NTROT
        CALL ORTHOG(DDTROT(1,I),DDTROT(1,J),NVAR)
     ENDDO
     CALL DOTPR(DDTROT(1,I),DDTROT(1,I),NVAR,Q)
     IF (Q .GT. TOL1) THEN
        NTROT = NTROT + 1
        CALL NORMALL(DDTROT(1,I),NVAR)
        IF(I.NE.NTROT)ddtrot(1:nvar,ntrot) = DDTROT(1:nvar,I)
     ENDIF
  ENDDO
  !
  ! Remove modes from second derivative matrix.
  !
  IF(NTROT.GT.0) THEN
     HFREQ = FTHSND
     HFREQ = FIFHUN
     IPT = 0
     I2=0
     J2=0
     DO I = 1,NATOM
        DO II=1,3
           I2=3*(I-1)+II
           DO J=I,NATOM
              DO JJ=1,3
                 J2=3*(J-1)+JJ
                 IF(J2.GE.I2) THEN
                   IPT=IPT+1
                    DO K=1,NTROT
                       DDF(IPT)=DDF(IPT)+HFREQ*DDTROT(I2,K) &
                      *DDTROT(J2,K) *DDM(I)*DDM(J) 
                    ENDDO
                 ENDIF
              ENDDO
           ENDDO
        ENDDO
     ENDDO
  ENDIF

  RETURN
  END SUBROUTINE RAISE_BEFOREMW
