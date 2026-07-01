#if KEY_MOLVIB==1
SUBROUTINE GMAT(NIC0,NIC,NQ,NAT,NAT3,IPRNT,NUMAT,JCONT, &
     NQM,ICTYP,IATI,IATJ,IATK,IATL, &
     U1,U2,W1,AMASS,B,G,X,IO)
  !
  !   Calculation of B and G matrices; G=B*[M-1]*BT
  !   B is found in BMAT in primitive internal coords and is then
  !   transformed by symmetrizing matrices U1 and U2 if necessary,
  !   as determined by the value of NUMAT.
  !   Finally G is created in the target symmetrized coordinates S.
  !   NIC0 - no. of primitive IC's
  !   NIC - number of IC's after transormation by U1
  !   NQ  - number of vibrational degrees of freedom
  !         This should be the number of IC's after all transformations
  !   NQM - physical first dimension of arrays in calling
  !           subroutine
  !   NAT - number of atoms, NAT3=NAT*3
  !
  use chm_kinds
  implicit none
  INTEGER NIC0,NIC,NQ,NAT,NAT3,NQM,IPRNT,NUMAT,IO
  INTEGER ICTYP(NQM),IATI(NQM),IATJ(NQM),IATK(NQM),IATL(NQM)
  real(chm_real)  U1(NQM,NQM),U2(NQM,NQM),W1(NQM,NQM),AMASS(NAT3)
  real(chm_real)  B(NQM,NQM),G(NQM,NQM),X(NAT3)
  !
  CHARACTER(len=4) JCONT
  real(chm_real) S
  INTEGER I,J,K,NINT
  !
  ! ... Calculate B matrix
  !
  CALL BMATR(B,X,NIC0,NQM,NAT,NAT3,ICTYP,IATI,IATJ,IATK,IATL, &
       IPRNT,IO)

  IF(IPRNT.GE.3) THEN
     WRITE(IO,*)
     WRITE(IO,*) ' B matrix'
     WRITE(IO,*)
     DO I=1,NIC0
        WRITE(IO,1000) I,(B(I,J),J=1,NAT3)
     ENDDO
  ENDIF
1000 FORMAT(1X,I3,8F9.5,12(/4X,8F9.5))
1001 FORMAT(1X,I3,8F9.6,12(/4X,8F9.6))
  !
  !     TRANSFORM B
  ! ... from dependent to independent internal coordinates
  ! ... S=U*R,
  ! ... U=U1 for NUMAT=1, U=U2*U1 for NUMAT=2
  !
  !     TRANSFORM B -> U1 * B
  !
  IF(NUMAT.GE.1) THEN
     DO I=1,NIC
        DO J=1,NAT3
           S=0.0
           DO K=1,NIC0
              S = S + U1(I,K)*B(K,J)
           ENDDO
           W1(I,J)=S
        ENDDO
     ENDDO
     !
     DO I=1,NIC
        DO J=1,NAT3
           B(I,J)=W1(I,J)
        ENDDO
     ENDDO
     !
     IF(IPRNT.GT.2) THEN
        WRITE(IO,*)
        WRITE(IO,*) ' The U1*B matrix '
        WRITE(IO,*)
        DO I=1,NIC
           WRITE(IO,1001) I,(W1(I,J),J=1,NAT3)
        ENDDO
     ENDIF
  ENDIF
  !
  !     TRANSFORM B = U1*B -> U2 * B
  !
  IF(NUMAT.GE.2 .AND. JCONT.NE.'G   ') THEN
     DO I=1,NQ
        DO J=1,NAT3
           S=0.0
           DO K=1,NIC
              S = S + U2(I,K)*B(K,J)
           ENDDO
           W1(I,J)=S
        ENDDO
     ENDDO
     !
     DO I=1,NQ
        DO J=1,NAT3
           B(I,J)=W1(I,J)
        ENDDO
     ENDDO
     !
     IF(IPRNT.GT.2) THEN
        WRITE(IO,*)
        WRITE(IO,*) ' The U2*U1*B matrix '
        WRITE(IO,*)
        DO I=1,NQ
           WRITE(IO,1001) I,(W1(I,J),J=1,NAT3)
        ENDDO
     ENDIF
  ENDIF
  !
  ! ... Test printout of masses
  !
  IF(IPRNT.GT.2) THEN
     WRITE(IO,*)
     WRITE(IO,*) ' Masses in atomic units '
     WRITE(IO,*)
     WRITE(IO,999) (I,AMASS(I),I=1,NAT3)
  ENDIF
999 FORMAT(1X,5(I2,2X,F8.4,2X),20(/1X,5(I2,2X,F8.4,2X)))
  !
  ! ... Create G matrix in target S coordinates
  !
  IF(NUMAT.LE.0) NINT=NIC0
  IF(NUMAT.GE.1) NINT=NIC
  IF(NUMAT.GE.2 .AND. JCONT.NE.'G   ') NINT=NQ
  DO I=1,NINT
     DO J=1,NINT
        S=0.0
        DO K=1,NAT3
           S = S + B(I,K)*B(J,K)/AMASS(K)
        ENDDO
        G(I,J)=S
     ENDDO
  ENDDO
  !
  IF(IPRNT.GE.2) THEN
     WRITE(IO,*)
     WRITE(IO,*) ' G matrix'
     WRITE(IO,*)
     DO I=1,NINT
        WRITE(IO,1000) I,(G(I,J),J=1,NINT)
     ENDDO
  ENDIF
  !
  RETURN
END SUBROUTINE GMAT

SUBROUTINE GMATC(NIC0,NIC,NQ,NAT,NAT3,IPRNT,NUMAT,JCONT, &
     IZMOL,IZMOL3,ICTYP,IATI,IATJ,IATK,IATL, &
     NQM,B,G,U1,U2,W1,X,AMASS,PRINCI,TOMM,IO)
  !
  !   Calculation of B and G matrices for crystal vibrations.
  !   Formalism is the same as for molecular vibrations, i.e.
  !   B is found in BMATR for primitive internal coords
  !   and external coordinates. A transformation by matrix U1
  !   transforms the IC's to an independent set, leaving the EC's
  !   unchanged. Because of the operations performed in MOLORT
  !   and the normalization chosen in BMATC,
  !   the EC's used here are such that the
  !   corresponding part of G is a unit matrix and there are no
  !   couplings with the IC's. Thus B is really a sort of C matrix
  !   (in the formalism of K.Kuczera, probably never to be published),
  !   i.e it gives a one-to-one
  !   correspondence between the molecular coordinate set (EC's+IC's)
  !   and the cartesian coordinates X.
  !   The standard formula is used to evaluate G:
  !          G=B*[M-1]*BT
  !   for the EC's this is just a test that everything is all right:
  !   check printout. Generally, it is not necessary for G to be a
  !   unit matrix in the EC part : the GF diagoanlization will work
  !   anyway.
  !
  !   If crystal symmetry is to be employed in analyzing vibrations,
  !   the matrix U2 may be used to transform G to coordinates
  !   forming the bases of irreducible representations of the space
  !   group.
  !
  !   IZMOL - no. of molecules in unit cell; IZMOL3=3*IZMOL
  !   PRINCI - principal components of moment of inertia of all molecules
  !   TOMM  - total masses of molecules
  !   NIC0 - no. of primitive MC's = EC's + primitive IC's
  !   NIC - number of vibrational degrees of freedom =
  !         EC's + independent IC's = NAT3
  !   NQ  - number of vibrational degrees of freedom (=NIC here)
  !   NQM - physical first dimension of arrays in calling
  !           subroutine
  !   NAT - total number of atoms in unit cell, NAT3=NAT*3
  !
  use chm_kinds
  implicit none
  INTEGER IO
  INTEGER NIC0,NIC,NQ,NAT,NAT3,IPRNT,NUMAT,IZMOL,IZMOL3,NQM
  INTEGER ICTYP(NQM),IATI(NQM),IATJ(NQM),IATK(NQM),IATL(NQM)
  real(chm_real) B(NQM,NQM),G(NQM,NQM),U1(NQM,NQM),U2(NQM,NQM)
  real(chm_real) W1(NQM,NQM),X(NAT3),AMASS(NAT3)
  real(chm_real) PRINCI(IZMOL3),TOMM(IZMOL)
  !
  INTEGER I,J,K,NINT
  CHARACTER(len=4) JCONT
  real(chm_real) S
  !
  ! ... Calculate B matrix
  !
  CALL BMATC(B,X,NIC0,NQM,NAT,NAT3,ICTYP,IATI,IATJ,IATK,IATL, &
       IPRNT,PRINCI,TOMM,IZMOL,IZMOL3,AMASS,IO)
  !
  IF(IPRNT.GE.3) THEN
     WRITE(IO,*)
     WRITE(IO,*) ' B matrix'
     WRITE(IO,*)
     DO I=1,NIC0
        WRITE(IO,1000) I,(B(I,J),J=1,NAT3)
     ENDDO
  ENDIF
1000 FORMAT(1X,I3,8F9.5,12(/4X,8F9.5))
1001 FORMAT(1X,I3,8F9.6,12(/4X,8F9.6))
  !
  !     TRANSFORM B
  ! ... from dependent to independent internal coordinates
  ! ... S=U*R,
  ! ... U=U1 for NUMAT=1, U=U2*U1 for NUMAT=2
  !
  !     TRANSFORM B -> U1 * B
  !
  IF(NUMAT.GE.1) THEN
     DO I=1,NIC
        DO J=1,NAT3
           S=0.0
           DO K=1,NIC0
              S = S + U1(I,K)*B(K,J)
           ENDDO
           W1(I,J)=S
        ENDDO
     ENDDO
     !
     DO I=1,NIC
        DO J=1,NAT3
           B(I,J)=W1(I,J)
        ENDDO
     ENDDO
     !
     IF(IPRNT.GT.2) THEN
        WRITE(IO,*)
        WRITE(IO,*) ' The U1*B matrix '
        WRITE(IO,*)
        DO I=1,NIC
           WRITE(IO,1001) I,(W1(I,J),J=1,NAT3)
        ENDDO
     ENDIF
  ENDIF
  !
  !     TRANSFORM B = U1*B -> U2*U1*B
  !
  IF(NUMAT.GE.2 .AND. JCONT.NE.'G   ') THEN
     DO I=1,NQ
        DO J=1,NAT3
           S=0.0
           DO K=1,NIC
              S = S + U2(I,K)*B(K,J)
           ENDDO
           W1(I,J)=S
        ENDDO
     ENDDO
     !
     DO I=1,NQ
        DO J=1,NAT3
           B(I,J)=W1(I,J)
        ENDDO
     ENDDO
     !
     IF(IPRNT.GT.2) THEN
        WRITE(IO,*)
        WRITE(IO,*) ' The U2*U1*B matrix '
        WRITE(IO,*)
        DO I=1,NQ
           WRITE(IO,1001) I,(W1(I,J),J=1,NAT3)
        ENDDO
     ENDIF
  ENDIF
  !
  ! ... Test printout of masses
  !
  IF(IPRNT.GT.2) THEN
     WRITE(IO,*)
     WRITE(IO,*) ' Masses in atomic units '
     WRITE(IO,*)
     WRITE(IO,999) (I,AMASS(I),I=1,NAT3)
  ENDIF
999 FORMAT(1X,5(I2,2X,F8.4,2X),20(/1X,5(I2,2X,F8.4,2X)))
  !
  ! ... Create G matrix in target S coordinates
  !
  IF(NUMAT.LE.0) NINT=NIC0
  IF(NUMAT.GE.1) NINT=NIC
  DO I=1,NINT
     DO J=1,NINT
        S=0.0
        DO K=1,NAT3
           S = S + B(I,K)*B(J,K)/AMASS(K)
        ENDDO
        G(I,J)=S
     ENDDO
  ENDDO
  !
  IF(IPRNT.GE.2) THEN
     WRITE(IO,*)
     WRITE(IO,*) ' G matrix'
     WRITE(IO,*)
     DO I=1,NINT
        WRITE(IO,1000) I,(G(I,J),J=1,NINT)
     ENDDO
  ENDIF
  !
  RETURN
END SUBROUTINE GMATC

SUBROUTINE BMATR(B,X,NIC,NICT,NAT,NAT3,ICTYP,ICATI,ICATJ, &
     ICATK,ICATL,IPRNT,IO)
  !
  !  Calculate B matrix from s-vector formulas from
  !  E.B.Wilson, J.C.Decius, P.C.Cross "Molecular vibrations",
  !  Dover,1980 (corrected 1955 McGraw Hill edition; Dihedrals
  !  are with corrected sign here).
  !
  !  NIC  - actual total number of internal coordinates
  !  NICT - physical dimension of B in calling subroutine
  !  NAT  - total no. of atoms, NAT3=3*NAT
  !  B - B matrix; (NIC,NAT3)
  !  X(NAT3) - cartesian coordinates, (x(NAT),Y(NAT),z(NAT))
  !  ICTYP - type of IC
  !          = 1 for bond stretch
  !          = 2 for angle bend
  !          = 3 for Wilson oopl bend
  !          = 4 for torsion
  ! ICATI,ICATJ,ICATK,ICATL - numbers of atoms I,J,K,L in the IC
  !
  use chm_kinds
  implicit none
  INTEGER NIC,NICT,NAT,NAT3,IPRNT,IO
  real(chm_real) B(NICT,NICT)
  real(chm_real) X(NAT3)
  REAL(CHM_REAL4) :: EJI(3),EJK(3),EKL(3),EPS=0.000001_chm_real,RJI,RJK,RKL,DOT,DOS,F1,F2
  REAL(CHM_REAL4) ELI(3),ELJ(3),ELK(3),RLI,RLJ,RLK,SINT,SING,COSG
  REAL(CHM_REAL4) VP1(3),VP2(3),VP3(3)
  INTEGER ICTYP(NIC),ICATI(NIC),ICATJ(NIC),ICATK(NIC)
  INTEGER ICATL(NIC)
  INTEGER NI,NJ,NK,NL,N,M,IPT,JPT,KPT,LPT
  INTEGER I,J
  !
  !...Set B to zero
  !
  DO I=1,NICT
     DO J=1,NICT
        B(I,J)=0.0
     ENDDO
  ENDDO
  !
  ! ... start main loop over IC's - find rows of B
  !
  DO N=1,NIC
     NI=ICATI(N)
     NJ=ICATJ(N)
     NK=ICATK(N)
     NL=ICATL(N)
     !
     ! ... Section for bond stretch between atoms I and J
     !
     IF(ICTYP(N).EQ.1) THEN
        RJI=0.
        DO M=1,3
           IPT = (M-1)*NAT + NI
           JPT = (M-1)*NAT + NJ
           EJI(M) = X(IPT) - X(JPT)
           RJI = RJI + EJI(M)*EJI(M)
        ENDDO
        IF(RJI.LT.EPS) GOTO 2000
        RJI = SQRT(RJI)
        DO M=1,3
           IPT = (M-1)*NAT + NI
           JPT = (M-1)*NAT + NJ
           B(N,IPT) =  EJI(M)/RJI
           B(N,JPT) = -B(N,IPT)
        ENDDO
        GOTO 1000
     ENDIF
     !
     ! ... Section for angle bend I-J-K
     !
     IF(ICTYP(N).EQ.2) THEN
        RJI=0.0
        RJK=0.0
        DO M=1,3
           IPT = (M-1)*NAT + NI
           JPT = (M-1)*NAT + NJ
           KPT = (M-1)*NAT + NK
           EJK(M) = X(KPT) - X(JPT)
           EJI(M) = X(IPT) - X(JPT)
           RJK = RJK + EJK(M)*EJK(M)
           RJI = RJI + EJI(M)*EJI(M)
        ENDDO
        IF(RJK.LT.EPS .OR. RJI.LT.EPS) GOTO 2000
        RJI=SQRT(RJI)
        RJK=SQRT(RJK)
        DOT=0.0
        DO M=1,3
           EJK(M) = EJK(M)/RJK
           EJI(M) = EJI(M)/RJI
           DOT = DOT + EJK(M)*EJI(M)
        ENDDO
        SINT=SQRT(1.0-DOT*DOT)
        IF(SINT.LT.EPS) GOTO 2100
        DO M=1,3
           IPT = (M-1)*NAT + NI
           JPT = (M-1)*NAT + NJ
           KPT = (M-1)*NAT + NK
           B(N,IPT) = (DOT*EJI(M) - EJK(M))/(RJI*SINT)
           B(N,KPT) = (DOT*EJK(M) - EJI(M))/(RJK*SINT)
           B(N,JPT) = -B(N,IPT)-B(N,KPT)
        ENDDO
        GOTO 1000
     ENDIF
     !
     ! ... Section for Wilson out-of-plane bend. The coordinate
     ! ... is the angle gamma between the I-L bond and the J-L-K plane
     ! ... (defined as 90-(angle of plane normal and bond).
     ! ... L is the central atom
     ! ... Version supports gamma.ne.0
     !
     IF(ICTYP(N).EQ.3) THEN
        RLI=0.0
        RLJ=0.0
        RLK=0.0
        DO M=1,3
           IPT = (M-1)*NAT + NI
           JPT = (M-1)*NAT + NJ
           KPT = (M-1)*NAT + NK
           LPT = (M-1)*NAT + NL
           ELI(M) = X(IPT) - X(LPT)
           ELJ(M) = X(JPT) - X(LPT)
           ELK(M) = X(KPT) - X(LPT)
           RLI = RLI + ELI(M)*ELI(M)
           RLJ = RLJ + ELJ(M)*ELJ(M)
           RLK = RLK + ELK(M)*ELK(M)
        ENDDO
        IF(RLI.LT.EPS .OR. RLJ.LT.EPS .OR. RLK.LT.EPS) GOTO 2000
        RLI=SQRT(RLI)
        RLJ=SQRT(RLJ)
        RLK=SQRT(RLK)
        DOT=0.0
        DO M=1,3
           ELI(M) = ELI(M)/RLI
           ELJ(M) = ELJ(M)/RLJ
           ELK(M) = ELK(M)/RLK
           DOT = DOT + ELJ(M)*ELK(M)
        ENDDO
        SINT=SQRT(1.0-DOT*DOT)
        IF(SINT.LT.EPS) GOTO 2100
        CALL CROSPR(VP1,ELJ,ELK)
        CALL CROSPR(VP2,ELK,ELI)
        CALL CROSPR(VP3,ELI,ELJ)
        SING=0.0
        DO M=1,3
           SING = SING + VP1(M)*ELI(M)
        ENDDO
        SING = SING/SINT
        COSG = -(1.0-SING*SING)
        IF(ABS(COSG).LT.EPS) GOTO 2100
        F1 = COSG*SINT
        F2 = SING/(COSG*SINT*SINT)
        DO M=1,3
           IPT = (M-1)*NAT + NI
           JPT = (M-1)*NAT + NJ
           KPT = (M-1)*NAT + NK
           LPT = (M-1)*NAT + NL
           B(N,IPT) = (VP1(M)/F1 - ELI(M)*SING/COSG)/RLI
           B(N,JPT) = (VP2(M)/F1 - F2*(ELJ(M) - DOT*ELK(M)))/RLJ
           B(N,KPT) = (VP3(M)/F1 - F2*(ELK(M) - DOT*ELJ(M)))/RLK
           B(N,LPT) = -B(N,IPT)-B(N,JPT)-B(N,KPT)
        ENDDO
        GOTO 1000
     ENDIF
     !
     ! ... Section for dihedral angle torsion. Angle between planes
     ! ... I-J-K and J-K-L changes
     !
     IF(ICTYP(N).EQ.4) THEN
        RJI=0.0
        RJK=0.0
        RLK=0.0
        DO M=1,3
           IPT = (M-1)*NAT + NI
           JPT = (M-1)*NAT + NJ
           KPT = (M-1)*NAT + NK
           LPT = (M-1)*NAT + NL
           EJI(M) = X(IPT) - X(JPT)
           EJK(M) = X(KPT) - X(JPT)
           ELK(M) = X(KPT) - X(LPT)
           RJI = RJI + EJI(M)*EJI(M)
           RJK = RJK + EJK(M)*EJK(M)
           RLK = RLK + ELK(M)*ELK(M)
        ENDDO
        IF(RJI.LT.EPS .OR. RJK.LT.EPS .OR. RLK.LT.EPS) GOTO 2000
        RJI=SQRT(RJI)
        RJK=SQRT(RJK)
        RLK=SQRT(RLK)
        DO M=1,3
           EJI(M) = EJI(M)/RJI
           EJK(M) = EJK(M)/RJK
           ELK(M) = ELK(M)/RLK
        ENDDO
        DOT=0.0
        DOS=0.0
        DO M=1,3
           DOT = DOT + EJI(M)*EJK(M)
           DOS = DOS + EJK(M)*ELK(M)
        ENDDO
        SINT=SQRT(1.0-DOT*DOT)
        SING=SQRT(1.0-DOS*DOS)
        IF(SINT.LT.EPS .OR. SING.LT.EPS) GOTO 2100
        CALL CROSPR(VP1,EJI,EJK)
        CALL CROSPR(VP2,ELK,EJK)
        F1 = (RJK - RJI*DOT)/(RJK*RJI*SINT)
        F2 = (RJK - RLK*DOS)/(RJK*RLK*SING)
        DO M=1,3
           IPT = (M-1)*NAT + NI
           JPT = (M-1)*NAT + NJ
           KPT = (M-1)*NAT + NK
           LPT = (M-1)*NAT + NL
           B(N,IPT) =  VP1(M)/(RJI*SINT*SINT)
           B(N,JPT) = -F1*VP1(M)/SINT - DOS*VP2(M)/(RJK*SING*SING)
           B(N,KPT) = -F2*VP2(M)/SING - DOT*VP1(M)/(RJK*SINT*SINT)
           B(N,LPT) =  VP2(M)/(RLK*SING*SING)
        ENDDO
        GOTO 1000
     ENDIF
     !
     ! ... Section for linear triatomic bending coordinates I-J-K:
     ! ... ICTYP=5 - in-plane; ICTYP=6 - out-of-plane
     !
     IF(ICTYP(N).EQ.5 .OR. ICTYP(N).EQ.6) THEN
        !
        ! ... First calculate a unit vector along the bond (EJI(M))
        ! ... and the two distances RJI,RJK
        RJK=0.
        RJI=0.
        DO M=1,3
           IPT=(M-1)*NAT + NI
           JPT=(M-1)*NAT + NJ
           KPT=(M-1)*NAT + NK
           EJI(M)=X(IPT)-X(JPT)
           EJK(M)=X(KPT)-X(JPT)
           RJI=RJI+EJI(M)**2
           RJK=RJK+EJK(M)**2
        ENDDO
        IF(RJI.LT.EPS .OR. RJK.LT.EPS) GOTO 2000
        RJI=SQRT(RJI)
        RJK=SQRT(RJK)
        DO M=1,3
           EJI(M)=EJI(M)/RJI
        ENDDO
        !
        ! ... If only three atom nos. are provided, find the unit
        ! ... vectors perpendicular to bond by trial (and error)
        ! ... First the y axis unit vector is orthogonalized to EJI,
        ! ... if zero results, the z axis is tried.
        ! ... In this case calling VP1(M) in-plane and VP2(M) oopl
        ! ... is somewhat arbitrary, but that's life.
        IF(NL.LE.0) THEN
           ! ... Try y unit vector
           VP2(1) = -EJI(1)*EJI(2)
           VP2(2) = 1.0 - EJI(2)**2
           VP2(3) = -EJI(3)*EJI(2)
           RKL=0.0
           DO M=1,3
              RKL=RKL+VP2(M)**2
           ENDDO
           ! ... Try z unit vector
           IF(RKL.LT.EPS) THEN
              VP2(1) = -EJI(1)*EJI(3)
              VP2(2) = -EJI(2)*EJI(3)
              VP2(3) =  1.0 - EJI(3)**2
              RKL=0.0
              DO M=1,3
                 RKL=RKL+VP2(M)**2
              ENDDO
           ENDIF
           IF(RKL.LT.EPS) GOTO 2000
           RKL=SQRT(RKL)
           DO M=1,3
              VP2(M)=VP2(M)/RKL
           ENDDO
           ! ...Find second vector by vector product with bond
           CALL CROSPR(VP1,VP2,EJI)
        ENDIF
        !              ! NL=0
        !
        ! ... If fourth atom is provided, IJL defines the plane for
        ! ... "in-plane".
        IF(NL.GT.0) THEN
           RKL=0.0
           DO M=1,3
              KPT=(M-1)*NAT+NK
              LPT=(M-1)*NAT+NL
              EKL(M)=X(LPT)-X(KPT)
              RKL=RKL+EKL(M)**2
           ENDDO
           IF(RKL.LT.EPS) GOTO 2000
           RKL=SQRT(RKL)
           DO M=1,3
              EKL(M)=EKL(M)/RKL
           ENDDO
           CALL CROSPR(VP2,EJI,EKL)
           RKL=0.0
           DO M=1,3
              RKL=RKL+VP2(M)**2
           ENDDO
           RKL=SQRT(RKL)
           DO M=1,3
              VP2(M)=VP2(M)/RKL
           ENDDO
           CALL CROSPR(VP1,VP2,EJI)
        ENDIF
        !              ! N>0
        !
        ! ... Finally: evaluate B matrix elements
        RJI=1.0/RJI
        RJK=1.0/RJK
        !
        ! ... In-plane bend
        IF(ICTYP(N).EQ.5) THEN
           DO M=1,3
              IPT = (M-1)*NAT + NI
              JPT = (M-1)*NAT + NJ
              KPT = (M-1)*NAT + NK
              B(N,IPT) = RJI*VP1(M)
              B(N,JPT) = -(RJI+RJK)*VP1(M)
              B(N,KPT) = RJK*VP1(M)
           ENDDO
        ENDIF
        !
        ! ... Out-of-plane bend
        IF(ICTYP(N).EQ.6) THEN
           DO M=1,3
              IPT = (M-1)*NAT + NI
              JPT = (M-1)*NAT + NJ
              KPT = (M-1)*NAT + NK
              B(N,IPT) = RJI*VP2(M)
              B(N,JPT) = -(RJI+RJK)*VP2(M)
              B(N,KPT) = RJK*VP2(M)
           ENDDO
        ENDIF
        !
        GOTO 1000
     ENDIF
     !           ! ICTYP=5,6
     !
     ! ... Check for undefined IC types
     !
     GOTO 2200
     !
1000 CONTINUE
  ENDDO
  IF(IPRNT.GT.4) THEN
     WRITE(IO,*)
     WRITE(IO,*) ' BMATR : the B matrix has been constructed for ', &
          'the following internal coordinates'
     WRITE(IO,*)
     WRITE(IO,8000) (ICTYP(N),ICATI(N),ICATJ(N),ICATK(N), &
          ICATL(N),N=1,NIC)
     WRITE(IO,*)
     WRITE(IO,*) ' The B matrix inside BMATR'
     WRITE(IO,*)
     DO I=1,NICT
        WRITE(IO,1001) I,(B(I,J),J=1,NAT3)
     ENDDO
  ENDIF
8000 FORMAT(1X,'ICTYP=',I1,2X,'ICATI=',I2,2X,'ICATJ=',I2, &
       2X,'ICATK=',I2,2X,'ICATL=',I2)
1001 FORMAT(1X,I3,8F9.6,12(/4X,8F9.6))
  RETURN
  !
  ! ... End of calculations. Error processing follows
  !
2000 WRITE(*,*) 'ERROR in IC : TYPE,I,J,K,L=',ICTYP(N),ICATI(N), &
       ICATJ(N),ICATK(N),ICATl(N)
  WRITE(*,*) 'CHECK FOR IDENTICAL ATOMS'
  RETURN
2100 WRITE(*,*) 'ERROR in IC : TYPE,I,J,K,L=',ICTYP(N),ICATI(N), &
       ICATJ(N),ICATK(N),ICATl(N)
  WRITE(*,*) 'CHECK ANGLES FOR 0,90,180 deg'
2200 WRITE(*,*) 'ERROR in IC : TYPE,I,J,K,L=',ICTYP(N),ICATI(N), &
       ICATJ(N),ICATK(N),ICATl(N)
  WRITE(*,*) 'CHECK IC TYPE FIELD'
  RETURN
END SUBROUTINE BMATR

SUBROUTINE CROSPR(U,V,W)
  !
  !  Calculate vector product U = V x W
  !
  use chm_kinds
  implicit none
  REAL(CHM_REAL4) U(3),V(3),W(3)
  !
  U(1) = V(2)*W(3) - V(3)*W(2)
  U(2) = V(3)*W(1) - V(1)*W(3)
  U(3) = V(1)*W(2) - V(2)*W(1)
  !
  RETURN
END SUBROUTINE CROSPR

SUBROUTINE BMATC(B,X,NIC,NICT,NAT,NAT3,ICTYP,ICATI,ICATJ, &
     ICATK,ICATL,IPRNT,PRINCI,TOMM,IZMOL,IZMOL3,AMASS,IO)
  !
  !  Version of BMAT for crystal vibrations. Differs from BMAT in that
  !  B elements for external coordinates may be calculated here.
  !  The internal coordinate parts are identical:
  !  Calculate B matrix from s-vector formulas from
  !  E.B.Wilson, J.C.Decius, P.C.Cross "Molecular vibrations",
  !  Dover,1980 (corrected 1955 McGraw Hill edition; Dihedrals
  !  are with corrected sign here).
  !
  !  IZMOL - no. of molecules in unit cell (CRYS)
  !  PRINCI - array of molecular principal components of momemts
  !         of inertia  (see MOLORT)
  !  TOMM   - array of molecular masses (see MOLORT)
  !
  !  NIC  - actual total number of (primitive) molecular coordinates
  !  NICT - physical dimension of B in calling subroutine
  !  NAT  - total no. of atoms, NAT3=3*NAT
  !         in 'CRYS' this is the total number of atoms in unit cell
  !  AMASS(NAT3) - atomic masses in amu, ordering as in X
  !  B - B matrix; (NIC,NAT3)
  !  X(NAT3) - cartesian coordinates, (x(NAT),Y(NAT),z(NAT))
  !  ICTYP - type of IC
  !          = 1 for bond stretch
  !          = 2 for angle bend
  !          = 3 for Wilson oopl bend
  !          = 4 for torsion
  !          = 11,12,13 for x,y,z translations, respectively
  !          = 14,15,16 for x,y,z rotations, respectively
  ! ICATI,ICATJ,ICATK,ICATL - numbers of atoms I,J,K,L in the IC
  ! ICATI - no. of molecule in unit cell for external coords (CRYS)
  !
  use chm_kinds
  implicit none
  INTEGER NIC,NICT,NAT,NAT3,IPRNT,IZMOL,IZMOL3,IO
  real(chm_real) B(NICT,NICT),PRINCI(IZMOL3)
  real(chm_real) X(NAT3),AMASS(NAT3),TOMM(IZMOL)
  REAL(CHM_REAL4) :: EJI(3),EJK(3),EKL(3),EPS=0.000001_chm_real,RJI,RJK,RKL,DOT,DOS,F1,F2
  REAL(CHM_REAL4) ELI(3),ELJ(3),ELK(3),RLI,RLJ,RLK,SINT,SING,COSG
  REAL(CHM_REAL4) VP1(3),VP2(3),VP3(3),S
  INTEGER ICTYP(NIC),ICATI(NIC),ICATJ(NIC),ICATK(NIC)
  INTEGER ICATL(NIC)
  INTEGER NI,NJ,NK,NL,N,M,IPT,JPT,KPT,LPT
  INTEGER I,J,K,II,JJ,IJK,I1,I2,IJK1,IJK2,NAT0
  !
  DO I=1,NICT
     DO J=1,NICT
        B(I,J)=0.0
     ENDDO
  ENDDO
  !
  ! ... NAT0 gives the no. of atoms in one molecule
  NAT0=NAT/IZMOL
  !
  ! ... start main loop over IC's - find rows of B
  !
  DO N=1,NIC
     NI=ICATI(N)
     NJ=ICATJ(N)
     NK=ICATK(N)
     NL=ICATL(N)
     !
     ! ... Section for bond stretch between atoms I and J
     !
     IF(ICTYP(N).EQ.1) THEN
        RJI=0.
        DO M=1,3
           IPT = (M-1)*NAT + NI
           JPT = (M-1)*NAT + NJ
           EJI(M) = X(IPT) - X(JPT)
           RJI = RJI + EJI(M)*EJI(M)
        ENDDO
        IF(RJI.LT.EPS) GOTO 2000
        RJI = SQRT(RJI)
        DO M=1,3
           IPT = (M-1)*NAT + NI
           JPT = (M-1)*NAT + NJ
           B(N,IPT) =  EJI(M)/RJI
           B(N,JPT) = -B(N,IPT)
        ENDDO
        GOTO 1000
     ENDIF
     !
     ! ... Section for angle bend I-J-K
     !
     IF(ICTYP(N).EQ.2) THEN
        RJI=0.0
        RJK=0.0
        DO M=1,3
           IPT = (M-1)*NAT + NI
           JPT = (M-1)*NAT + NJ
           KPT = (M-1)*NAT + NK
           EJK(M) = X(KPT) - X(JPT)
           EJI(M) = X(IPT) - X(JPT)
           RJK = RJK + EJK(M)*EJK(M)
           RJI = RJI + EJI(M)*EJI(M)
        ENDDO
        IF(RJK.LT.EPS .OR. RJI.LT.EPS) GOTO 2000
        RJI=SQRT(RJI)
        RJK=SQRT(RJK)
        DOT=0.0
        DO M=1,3
           EJK(M) = EJK(M)/RJK
           EJI(M) = EJI(M)/RJI
           DOT = DOT + EJK(M)*EJI(M)
        ENDDO
        SINT=SQRT(1.0-DOT*DOT)
        IF(SINT.LT.EPS) GOTO 2100
        DO M=1,3
           IPT = (M-1)*NAT + NI
           JPT = (M-1)*NAT + NJ
           KPT = (M-1)*NAT + NK
           B(N,IPT) = (DOT*EJI(M) - EJK(M))/(RJI*SINT)
           B(N,KPT) = (DOT*EJK(M) - EJI(M))/(RJK*SINT)
           B(N,JPT) = -B(N,IPT)-B(N,KPT)
        ENDDO
        GOTO 1000
     ENDIF
     !
     ! ... Section for Wilson out-of-plane bend. The coordinate
     ! ... is the angle gamma between the I-L bond and the J-L-K plane
     ! ... (defined as 90-(angle of plane normal and bond).
     ! ... L is the central atom
     ! ... Version supports gamma.ne.0
     !
     IF(ICTYP(N).EQ.3) THEN
        RLI=0.0
        RLJ=0.0
        RLK=0.0
        DO M=1,3
           IPT = (M-1)*NAT + NI
           JPT = (M-1)*NAT + NJ
           KPT = (M-1)*NAT + NK
           LPT = (M-1)*NAT + NL
           ELI(M) = X(IPT) - X(LPT)
           ELJ(M) = X(JPT) - X(LPT)
           ELK(M) = X(KPT) - X(LPT)
           RLI = RLI + ELI(M)*ELI(M)
           RLJ = RLJ + ELJ(M)*ELJ(M)
           RLK = RLK + ELK(M)*ELK(M)
        ENDDO
        IF(RLI.LT.EPS .OR. RLJ.LT.EPS .OR. RLK.LT.EPS) GOTO 2000
        RLI=SQRT(RLI)
        RLJ=SQRT(RLJ)
        RLK=SQRT(RLK)
        DOT=0.0
        DO M=1,3
           ELI(M) = ELI(M)/RLI
           ELJ(M) = ELJ(M)/RLJ
           ELK(M) = ELK(M)/RLK
           DOT = DOT + ELJ(M)*ELK(M)
        ENDDO
        SINT=SQRT(1.0-DOT*DOT)
        IF(SINT.LT.EPS) GOTO 2100
        CALL CROSPR(VP1,ELJ,ELK)
        CALL CROSPR(VP2,ELK,ELI)
        CALL CROSPR(VP3,ELI,ELJ)
        SING=0.0
        DO M=1,3
           SING = SING + VP1(M)*ELI(M)
        ENDDO
        SING = SING/SINT
        COSG = -(1.0-SING*SING)
        IF(ABS(COSG).LT.EPS) GOTO 2100
        F1 = COSG*SINT
        F2 = SING/(COSG*SINT*SINT)
        DO M=1,3
           IPT = (M-1)*NAT + NI
           JPT = (M-1)*NAT + NJ
           KPT = (M-1)*NAT + NK
           LPT = (M-1)*NAT + NL
           B(N,IPT) = (VP1(M)/F1 - ELI(M)*SING/COSG)/RLI
           B(N,JPT) = (VP2(M)/F1 - F2*(ELJ(M) - DOT*ELK(M)))/RLJ
           B(N,KPT) = (VP3(M)/F1 - F2*(ELK(M) - DOT*ELJ(M)))/RLK
           B(N,LPT) = -B(N,IPT)-B(N,JPT)-B(N,KPT)
        ENDDO
        GOTO 1000
     ENDIF
     !
     ! ... Section for dihedral angle torsion. Angle between planes
     ! ... I-J-K and J-K-L changes
     !
     IF(ICTYP(N).EQ.4) THEN
        RJI=0.0
        RJK=0.0
        RLK=0.0
        DO M=1,3
           IPT = (M-1)*NAT + NI
           JPT = (M-1)*NAT + NJ
           KPT = (M-1)*NAT + NK
           LPT = (M-1)*NAT + NL
           EJI(M) = X(IPT) - X(JPT)
           EJK(M) = X(KPT) - X(JPT)
           ELK(M) = X(KPT) - X(LPT)
           RJI = RJI + EJI(M)*EJI(M)
           RJK = RJK + EJK(M)*EJK(M)
           RLK = RLK + ELK(M)*ELK(M)
        ENDDO
        IF(RJI.LT.EPS .OR. RJK.LT.EPS .OR. RLK.LT.EPS) GOTO 2000
        RJI=SQRT(RJI)
        RJK=SQRT(RJK)
        RLK=SQRT(RLK)
        DO M=1,3
           EJI(M) = EJI(M)/RJI
           EJK(M) = EJK(M)/RJK
           ELK(M) = ELK(M)/RLK
        ENDDO
        DOT=0.0
        DOS=0.0
        DO M=1,3
           DOT = DOT + EJI(M)*EJK(M)
           DOS = DOS + EJK(M)*ELK(M)
        ENDDO
        SINT=SQRT(1.0-DOT*DOT)
        SING=SQRT(1.0-DOS*DOS)
        IF(SINT.LT.EPS .OR. SING.LT.EPS) GOTO 2100
        CALL CROSPR(VP1,EJI,EJK)
        CALL CROSPR(VP2,ELK,EJK)
        F1 = (RJK - RJI*DOT)/(RJK*RJI*SINT)
        F2 = (RJK - RLK*DOS)/(RJK*RLK*SING)
        DO M=1,3
           IPT = (M-1)*NAT + NI
           JPT = (M-1)*NAT + NJ
           KPT = (M-1)*NAT + NK
           LPT = (M-1)*NAT + NL
           B(N,IPT) =  VP1(M)/(RJI*SINT*SINT)
           B(N,JPT) = -F1*VP1(M)/SINT - DOS*VP2(M)/(RJK*SING*SING)
           B(N,KPT) = -F2*VP2(M)/SING - DOT*VP1(M)/(RJK*SINT*SINT)
           B(N,LPT) =  VP2(M)/(RLK*SING*SING)
        ENDDO
        GOTO 1000
     ENDIF
     !
     ! ... Section for linear triatomic bending coordinates I-J-K:
     ! ... ICTYP=5 - in-plane; ICTYP=6 - out-of-plane
     !
     IF(ICTYP(N).EQ.5 .OR. ICTYP(N).EQ.6) THEN
        !
        ! ... First calculate a unit vector along the bond (EJI(M))
        ! ... and the two distances RJI,RJK
        RJK=0.
        RJI=0.
        DO M=1,3
           IPT=(M-1)*NAT + NI
           JPT=(M-1)*NAT + NJ
           KPT=(M-1)*NAT + NK
           EJI(M)=X(IPT)-X(JPT)
           EJK(M)=X(KPT)-X(JPT)
           RJI=RJI+EJI(M)**2
           RJK=RJK+EJK(M)**2
        ENDDO
        IF(RJI.LT.EPS .OR. RJK.LT.EPS) GOTO 2000
        RJI=SQRT(RJI)
        RJK=SQRT(RJK)
        DO M=1,3
           EJI(M)=EJI(M)/RJI
        ENDDO
        !
        ! ... If only three atom nos. are provided, find the unit
        ! ... vectors perpendicular to bond by trial (and error)
        ! ... First the y axis unit vector is orthogonalized to EJI,
        ! ... if zero results, the z axis is tried.
        ! ... In this case calling VP1(M) in-plane and VP2(M) oopl
        ! ... is somewhat arbitrary, but that's life.
        IF(NL.LE.0) THEN
           ! ... Try y unit vector
           VP2(1) = -EJI(1)*EJI(2)
           VP2(2) = 1.0 - EJI(2)**2
           VP2(3) = -EJI(3)*EJI(2)
           RKL=0.0
           DO M=1,3
              RKL=RKL+VP2(M)**2
           ENDDO
           ! ... Try z unit vector
           IF(RKL.LT.EPS) THEN
              VP2(1) = -EJI(1)*EJI(3)
              VP2(2) = -EJI(2)*EJI(3)
              VP2(3) =  1.0 - EJI(3)**2
              RKL=0.0
              DO M=1,3
                 RKL=RKL+VP2(M)**2
              ENDDO
           ENDIF
           IF(RKL.LT.EPS) GOTO 2000
           RKL=SQRT(RKL)
           DO M=1,3
              VP2(M)=VP2(M)/RKL
           ENDDO
           ! ...Find second vector by vector product with bond
           CALL CROSPR(VP1,VP2,EJI)
        ENDIF
        !             ! NL=0
        !
        ! ... If fourth atom is provided, IJL defines the plane for
        ! ... "in-plane".
        IF(NL.GT.0) THEN
           RKL=0.0
           DO M=1,3
              KPT=(M-1)*NAT+NK
              LPT=(M-1)*NAT+NL
              EKL(M)=X(LPT)-X(KPT)
              RKL=RKL+EKL(M)**2
           ENDDO
           IF(RKL.LT.EPS) GOTO 2000
           RKL=SQRT(RKL)
           DO M=1,3
              EKL(M)=EKL(M)/RKL
           ENDDO
           CALL CROSPR(VP2,EJI,EKL)
           RKL=0.0
           DO M=1,3
              RKL=RKL+VP2(M)**2
           ENDDO
           RKL=SQRT(RKL)
           DO M=1,3
              VP2(M)=VP2(M)/RKL
           ENDDO
           CALL CROSPR(VP1,VP2,EJI)
        ENDIF
        !              ! N>0
        !
        ! ... Finally: evaluate B matrix elements
        RJI=1.0/RJI
        RJK=1.0/RJK
        !
        ! ... In-plane bend
        IF(ICTYP(N).EQ.5) THEN
           DO M=1,3
              IPT = (M-1)*NAT + NI
              JPT = (M-1)*NAT + NJ
              KPT = (M-1)*NAT + NK
              B(N,IPT) = RJI*VP1(M)
              B(N,JPT) = -(RJI+RJK)*VP1(M)
              B(N,KPT) = RJK*VP1(M)
           ENDDO
        ENDIF
        !
        ! ... Out-of-plane bend
        IF(ICTYP(N).EQ.6) THEN
           DO M=1,3
              IPT = (M-1)*NAT + NI
              JPT = (M-1)*NAT + NJ
              KPT = (M-1)*NAT + NK
              B(N,IPT) = RJI*VP2(M)
              B(N,JPT) = -(RJI+RJK)*VP2(M)
              B(N,KPT) = RJK*VP2(M)
           ENDDO
        ENDIF
        !
        GOTO 1000
     ENDIF
     !            ! ICTYP=5,6
     !
     ! ... Section for molecular translations
     ! ... ICTYP=11,12,13 - x,y,z translations, respectively
     !
     IF(ICTYP(N).GE.11 .AND. ICTYP(N).LE.13) THEN
        IF(ICTYP(N).EQ.11) II=0
        IF(ICTYP(N).EQ.12) II=NAT
        IF(ICTYP(N).EQ.13) II=2*NAT
        JJ=(NI-1)*NAT0
        DO K=1,NAT0
           IJK = II+JJ+K
           B(N,IJK) = AMASS(IJK)/SQRT(TOMM(NI))
        ENDDO
        GOTO 1000
     ENDIF
     !
     ! ... Section for molecular rotations
     ! ... ICTYP=14,15,16 - x,y,z rotations, respectively
     !
     IF(ICTYP(N).GE.14 .AND. ICTYP(N).LE.16) THEN
        IF(ICTYP(N).EQ.14) THEN
           I1=NAT
           I2=2*NAT
           II=3*NI-2
        ENDIF
        IF(ICTYP(N).EQ.15) THEN
           I1=2*NAT
           I2=0
           II=3*NI-1
        ENDIF
        IF(ICTYP(N).EQ.16) THEN
           I1=0
           I2=NAT
           II=3*NI
        ENDIF
        JJ=(NI-1)*NAT0
        DO K=1,NAT0
           IJK1 = I1+JJ+K
           IJK2 = I2+JJ+K
           S=SQRT(PRINCI(II))
           B(N,IJK1) = -(AMASS(IJK1)*X(IJK2))/S
           B(N,IJK2) =  (AMASS(IJK2)*X(IJK1))/S
        ENDDO
        GOTO 1000
     ENDIF
     !
     ! ... Check for undefined IC types
     !
     GOTO 2200
     !
1000 CONTINUE
  ENDDO
  IF(IPRNT.GT.4) THEN
     WRITE(IO,*)
     WRITE(IO,*) ' BMAT : the B matrix has been constructed for ', &
          'the following internal coordinates'
     WRITE(IO,*)
     WRITE(IO,8000) (ICTYP(N),ICATI(N),ICATJ(N),ICATK(N), &
          ICATL(N),N=1,NIC)
  ENDIF
8000 FORMAT(1X,'ICTYP=',I2,2X,'ICATI=',I2,2X,'ICATJ=',I2, &
       2X,'ICATK=',I2,2X,'ICATL=',I2)
  RETURN
  !
  ! ... End of calculations. Error processing follows
  !
2000 WRITE(*,*) 'ERROR in IC : TYPE,I,J,K,L=',ICTYP(N),ICATI(N), &
       ICATJ(N),ICATK(N),ICATl(N)
  WRITE(*,*) 'CHECK FOR IDENTICAL ATOMS'
  RETURN
2100 WRITE(*,*) 'ERROR in IC : TYPE,I,J,K,L=',ICTYP(N),ICATI(N), &
       ICATJ(N),ICATK(N),ICATl(N)
  WRITE(*,*) 'CHECK ANGLES FOR 0,90,180 deg'
2200 WRITE(*,*) 'ERROR in IC : TYPE,I,J,K,L=',ICTYP(N),ICATI(N), &
       ICATJ(N),ICATK(N),ICATl(N)
  WRITE(*,*) 'CHECK IC TYPE FIELD'
  !
  RETURN
END SUBROUTINE BMATC
#else /**/
SUBROUTINE NULL_GM
  RETURN
END SUBROUTINE NULL_GM
#endif /*  MOLVIB*/

