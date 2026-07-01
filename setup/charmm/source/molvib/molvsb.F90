#if KEY_MOLVIB==1
SUBROUTINE CRYSTF(NQ,IPRNT,NQM,FX,FS,B,LS,W1,W2,V1,INDX,IO)
  !
  !   Perform the necessary transformations on the F matrix for
  !   crystal calculations in molecular coordinates.
  !   1. On X coordinate level: transform FX from input coordinates
  !      to the oriented set given by MOLORT (W2 matrix).
  !          FX -> W2*FX*W2T (W2 is orthogonal)
  !
  !   2. Cartesian -> Molecular coord transformation
  !         FS = [B-1]T*FX*[B-1]
  !      where the B matrix generated in GMATC has already been
  !      transformed by U1 (and U2, if it was supplied).
  !
  !   On input W2 should contain the coordinate transformation
  !   which orients all individual molecules in unit cell (MOLORT)
  !   and B should contain the transformation from the oriented
  !   cartesian coordinates to the molecular coordinates (GMATC,BMATC).
  !
  !   NQ -  number of vibrational degrees of freedom
  !
  use chm_kinds
  implicit none
  INTEGER NQ,IPRNT,NQM,IO
  real(chm_real) FX(NQM,NQM),FS(NQM,NQM),B(NQM,NQM),LS(NQM,NQM)
  real(chm_real) W1(NQM,NQM),W2(NQM,NQM),V1(NQM)
  INTEGER INDX(NQM)
  !
  INTEGER I,J,K,L
  real(chm_real) S
  !
  WRITE(IO,*)
  WRITE(IO,*) '   CRYSTF : transforming F to molecular coords'
  WRITE(IO,*)
  !
  ! ... Transform FX to oriented cartesian coordinates
  !
  DO I=1,NQ
     DO J=1,NQ
        S=0.0D0
        DO K=1,NQ
           DO L=1,NQ
              S = S + W2(I,K)*FX(K,L)*W2(J,L)
           ENDDO
        ENDDO
        W1(I,J)=S
     ENDDO
  ENDDO
  !
  ! ... test
  IF(IPRNT.GT.2) THEN
     WRITE(IO,*)
     WRITE(IO,*) ' The FX matrix in oriented coordinates'
     WRITE(IO,*)
     DO I=1,NQ
        WRITE(IO,1006) I,(W1(I,J),J=1,NQ)
     ENDDO
  ENDIF
  !
  ! ... Invert the B matrix! (returned in LS, B is destroyed)
  !
  !      CALL MATNV(B,LS,V1,NQ,NQM)
  CALL MATNVN(B,LS,V1,INDX,NQ,NQM)
  !
  ! ... test
  IF(IPRNT.GT.3) THEN
     WRITE(IO,*)
     WRITE(IO,*) ' The inverse of the B matrix'
     WRITE(IO,*)
     DO I=1,NQ
        WRITE(IO,1006) I,(LS(I,J),J=1,NQ)
     ENDDO
  ENDIF
  !
  ! ... Transform FX to molecular coordinates
  !
  DO I=1,NQ
     DO J=1,NQ
        S=0.0D0
        DO K=1,NQ
           DO L=1,NQ
              S = S + LS(K,I)*W1(K,L)*LS(L,J)
           ENDDO
        ENDDO
        FS(I,J)=S
     ENDDO
  ENDDO
  !
  ! ... test
  IF(IPRNT.GT.2) THEN
     WRITE(IO,*)
     WRITE(IO,*) ' The FS matrix : molecular coordinates'
     WRITE(IO,*)
     DO I=1,NQ
        WRITE(IO,1006) I,(FS(I,J),J=1,NQ)
     ENDDO
  ENDIF
1006 FORMAT(1X,I3,8F9.3,12(/4X,8F9.3))
  !
  RETURN
END SUBROUTINE CRYSTF

SUBROUTINE FRGET(NIC0,NIC,NQ,IPRNT,NUMAT,ICANO, &
     NQM,U1,U2,W1,W2,FS,FX,IOUT)
  !
  !     Transform F matrix from independent internal coords S
  !     to  primitive internal coordinates R
  !          W1 = UT*FS*U
  !     where U=U1 for NUMAT=1, U=U2*U1 for NUMAT=2
  !
  !     The ICANO variable is used for forcing printout of FR
  !     in the preliminary step of canonic force field evaluation
  !
  !   FX -  force constant matrix in R coords (NIC,NIC)
  !   FS -  force constant matrix in S coords (NQ,NQ)
  !   dimensions:
  !   NIC0 - number of primitive internal coordinates R
  !   NIC - number of IC's S1=U1*R
  !   NQ  - no. of S2=U2*S1 (=number of vibrational degrees of freedom)
  !
  use chm_kinds
  implicit none
  real(chm_real) S
  INTEGER  I,J,K,N1
  !
  INTEGER NQM,NIC0,NIC,NQ,IPRNT,NUMAT,ICANO,IOUT
  real(chm_real) U1(NQM,NQM),U2(NQM,NQM),W1(NQM,NQM),W2(NQM,NQM)
  real(chm_real) FS(NQM,NQM),FX(NQM,NQM)
  !
  WRITE(IOUT,*)
  WRITE(IOUT,*) '     Calculating FR = UT*FS*U '
  WRITE(IOUT,*)
  !
  DO I=1,NQM
     DO J=1,NQM
        W1(I,J)=0.0
        W2(I,J)=0.0
        FX(I,J)=0.0
     ENDDO
  ENDDO
  !
  ! ... Determine total transformation
  !
  N1=NIC
  DO I=1,NIC
     DO J=1,NIC0
        W1(I,J)=U1(I,J)
     ENDDO
  ENDDO
  !
  IF(NUMAT.EQ.2) THEN
     N1=NQ
     DO I=1,NQ
        DO J=1,NIC0
           S=0.0
           DO K=1,NIC
              S = S + U2(I,K)*U1(K,J)
           ENDDO
           W1(I,J)=S
        ENDDO
     ENDDO
  ENDIF
  !
  ! ... Transform FS to FR
  !
  DO I=1,N1
     DO J=1,NIC0
        S=0.0
        DO K=1,N1
           S = S + FS(I,K)*W1(K,J)
        ENDDO
        W2(I,J)=S
     ENDDO
  ENDDO
  !
  DO I=1,NIC0
     DO J=1,NIC0
        S=0.0
        DO K=1,N1
           S = S + W1(K,I)*W2(K,J)
        ENDDO
        FX(I,J)=S
     ENDDO
  ENDDO
  !
  IF(IPRNT.GE.2) THEN
     WRITE(IOUT,*)
     WRITE(IOUT,*) ' The matrix FR=U1T*FS*U1 '
     WRITE(IOUT,*)
     DO I=1,NIC
        WRITE(IOUT,1006) I,(FX(I,J),J=1,NIC)
     ENDDO
  ENDIF
  !
  IF(ICANO.NE.0) THEN
     WRITE(IOUT,*)
     WRITE(IOUT,*) ' The FR=U1T*FS*U1 matrix', &
          ' in Schachtschneider format'
     WRITE(IOUT,*)
     CALL WRMAT(FX,NIC0,NIC0,NQM,IOUT,IPRNT)
  ENDIF
  !
1006 FORMAT(1X,I3,8F9.3,12(/4X,8F9.3))
  RETURN
END SUBROUTINE FRGET

SUBROUTINE FSGL(NAT,NAT3,NQ,IPRNT,NQM,LX,FS,DD,IO)
  !
  !  Calculate the force constant matrix FS in independent internal
  !  coordinates from the internal coordinate eigenvectors LS and
  !  the eigenvalues DD :
  !                       FS = {[LS-1]T}*DD*[LS-1]
  !
  !  B - the B natrix in the indep. S coords, (=U*B); (NQ,NAT3)
  !  G  = B*[M-1]*BT - the Wilson G matrix in S coords; (NQ,NQ)
  !  LS - vibrational eigenvectors in independent internal coords S (NQ,NQ)
  !  LX - here only : contains inverse of LS (calculated in routine PED)
  !  DD - vector containing vibrational eigenvalues
  !  AMASS - vector of atomic masses in amu ; (NAT)
  !  dimensions:
  !  NAT - number of atoms,  NAT3= 3*NAT
  !  NQ  - number of vibrational degrees of freedom
  !
  use chm_kinds
  implicit none
  INTEGER NAT,NAT3,NQ,IPRNT,NQM,IO
  real(chm_real) LX(NQM,NQM),FS(NQM,NQM),DD(NQM)
  !
  INTEGER I,J,K
  real(chm_real) S
  !
  ! ... Calculate FS
  !
  DO I=1,NQ
     DO J=1,NQ
        S=0.0
        DO K=1,NQ
           S = S + LX(K,I)*DD(K)*LX(K,J)
        ENDDO
        FS(I,J)=S
     ENDDO
  ENDDO
  !
  IF(IPRNT.GE.4) THEN
     WRITE(IO,*)
     WRITE(IO,*) ' The FS matrix'
     WRITE(IO,*)
     DO I=1,NQ
        WRITE(IO,1001) I,(FS(I,J),J=1,NQ)
     ENDDO
  ENDIF
  !
1001 FORMAT(1X,I3,8F9.3,12(/4X,8F9.3))
  !
  RETURN
END SUBROUTINE FSGL

SUBROUTINE FTRAK(NIC0,NIC,NQ,IPRNT,NUMAT,NQM,W1,U1,U2,FS,LX, &
     V1,W2,INDX,IO)
  !
  !   Transform F matrix from canonic ff form FR* to indep IC's S
  !
  !   On input arrays contain
  !   FS - canonic ff FR*
  !   LX - G matrix eigenvectors in R coords (NIC0,NIC0)
  !   dimensions:
  !   NIC0 - number of primitive internal coordinates R
  !   NQ  - number of vibrational degrees of freedom
  !
  !   On output the FS contains the force field in indep coords S=U*R
  !                    where U=U1 for NUMAT=1 and U=U2*U1 for NUMAT=2,
  !
  use chm_kinds
  implicit none
  INTEGER NIC0,NIC,NQ,IPRNT,NUMAT,NQM,IO
  real(chm_real) W1(NQM,NQM),FS(NQM,NQM),LX(NQM,NQM),V1(NQM)
  real(chm_real) U1(NQM,NQM),U2(NQM,NQM),W2(NQM,NQM)
  INTEGER INDX(NQM)
  !
  INTEGER I,J,K,KK,NRED
  real(chm_real) S,EPS
  LOGICAL QORT
  !
  QORT=.TRUE.
  EPS=1.0D-5
  !
  WRITE(IO,*)
  WRITE(IO,*) '     Calculating FS from FR*'
  WRITE(IO,*)
  !
  ! ... Generate transformation matrix in W1:
  ! ...   first NQ rows contain matrix U1 (or U2*U1 for NUMAT=2)
  ! ...   last NIC0-NQ rows contain G eigenvectors of zero eigenvalue
  !
  DO I=1,NIC0
     DO J=1,NIC0
        W1(I,J)=0.0
     ENDDO
  ENDDO
  !
  NRED=NIC0-NQ
  !
  IF(NUMAT.EQ.1) THEN
     DO I=1,NIC
        DO J=1,NIC0
           W1(I,J)=U1(I,J)
        ENDDO
     ENDDO
  ENDIF
  !
  IF(NUMAT.EQ.2) THEN
     DO I=1,NQ
        DO J=1,NIC0
           S=0.0
           DO K=1,NIC
              S = S + U2(I,K)*U1(K,J)
           ENDDO
           W1(I,J)=S
        ENDDO
     ENDDO
  ENDIF
  !
  DO I=1,NRED
     DO J=1,NIC0
        KK=NQ+I
        W1(KK,J)=LX(J,I)
     ENDDO
  ENDDO
  !
  ! ... Check if W1 is orthogonal
  !
  DO I=1,NIC0
     DO J=1,NIC0
        S=0.0
        DO K=1,NIC0
           S=S+W1(I,K)*W1(J,K)
        ENDDO
        IF(I.EQ.J) THEN
           IF(ABS(1.0D0-S).GT.EPS) QORT=.FALSE.
        ELSE
           IF(ABS(S).GT.EPS) QORT=.FALSE.
        ENDIF
     ENDDO
  ENDDO
  !
  ! ... Transform F : FS = U*FR*UT for orthogonal matrices
  ! ...                  = [U-1]T *FR*[U-1] otherwise
  !
  IF(QORT) THEN
     WRITE(IO,*) ' The transformation matrix is orthogonal'
     WRITE(IO,*)
     DO I=1,NIC0
        DO J=1,NIC0
           S=0.0
           DO K=1,NIC0
              S = S + FS(I,K)*W1(J,K)
           ENDDO
           LX(I,J)=S
        ENDDO
     ENDDO
     !
     DO I=1,NIC0
        DO J=1,NIC0
           S=0.0
           DO K=1,NIC0
              S = S + W1(I,K)*LX(K,J)
           ENDDO
           FS(I,J)=S
        ENDDO
     ENDDO
  ENDIF
  !
  IF(.NOT.QORT) THEN
     WRITE(IO,*) ' The transformation matrix is not orthogonal'
     WRITE(IO,*)
     DO I=1,NIC0
        DO J=1,NIC0
           W2(I,J)=W1(I,J)
        ENDDO
     ENDDO
     !
     !        CALL MATNV(W2,W1,V1,NIC0,NQM)
     CALL MATNVN(W2,W1,V1,INDX,NIC0,NQM)
     !
     DO I=1,NIC0
        DO J=1,NIC0
           S=0.0
           DO K=1,NIC0
              S = S + FS(I,K)*W1(K,J)
           ENDDO
           LX(I,J)=S
        ENDDO
     ENDDO
     !
     DO I=1,NIC0
        DO J=1,NIC0
           S=0.0
           DO K=1,NIC0
              S = S + W1(K,I)*LX(K,J)
           ENDDO
           FS(I,J)=S
        ENDDO
     ENDDO
  ENDIF
  !
  IF(IPRNT.GE.2) THEN
     WRITE(IO,*)
     WRITE(IO,*) ' The transformed matrix FS'
     WRITE(IO,*)
     DO I=1,NQ
        WRITE(IO,1006) I,(FS(I,J),J=1,NQ)
     ENDDO
  ENDIF
  !
1006 FORMAT(1X,I3,8F9.3,12(/4X,8F9.3))
  RETURN
END SUBROUTINE FTRAK

SUBROUTINE MOLORT(W,V,X,AM,NAT,NAT3,NQM,IZMOL,IZMOL3, &
     IPRNT,PRINCI,TOMM,MNAT,IO)
  !
  !   The atomic cartesian coordinates X are transformed so that
  !   for each molecule in the unit cell the origin lies at the CM
  !   and the axes are aligned with the principal axes of the moment
  !   of inertia.
  !
  !   NAT - total no. of atoms in unit cell; NAT3=3*NAT
  !   IZMOL - no. of molecules in unit cell; IZMOL3=3*IZMOL
  !   X(NAT3) - cartesian coordinates: X=[X1...Xnat;Y1...Ynat;Z1...Znat]
  !   AM(NAT3) - masses in amu; ordered as X
  !   MNAT(IZMOL) - array of numbers of atoms in the molecules
  !                 comprising unit cell/aggregate
  !   TOMM(IZMOL) - array of molecular masses TOMM=[M1,...,Mizmol]
  !   PRINCI(IZMOL3) - array of molecular principal components of
  !      moment of inertia,(IA <= IB <= IC):
  !      PRINCI=[IA1,IB1,IC1,...,IAizmol,IBizmol,ICizmol], [amu*A**2]
  !   W - on output this contains the coordinate transformation
  !       matrix (rotational part): Xnew=W*Xold
  !
  use chm_kinds
  implicit none
  INTEGER NAT,NAT3,NQM,IZMOL,IZMOL3,IPRNT,IO
  real(chm_real) PRINCI(IZMOL3),TOMM(IZMOL)
  real(chm_real) W(NQM,NQM),V(NAT3),X(NAT3),AM(NAT3),PRINC(3),TOM
  real(chm_real) XCM,YCM,ZCM,AI(3,3),TEST(3,3),R(3),S
  INTEGER MNAT(IZMOL)
  INTEGER I,J,K,M,N,NAT0,NAT2,I3,JK,JK1,JK2
  !
  WRITE(IO,*)
  WRITE(IO,*)  '   MOLORT : orienting molecules '
  WRITE(IO,*)
  !
  NAT2=2*NAT
  DO I=1,NAT3
     DO J=1,NAT3
        W(I,J)=0.0D0
     ENDDO
  ENDDO
  !
  ! ... Loop over molecules : find CM and moment of inertia
  ! ... and transform coordinates
  !
  K=-MNAT(1)
  DO I=1,IZMOL
     !
     NAT0=MNAT(I)
     K=K+NAT0
     I3=3*I
     !
     ! ... find CM coordinates and shift X; also get total mass TOM
     XCM=0.0D0
     YCM=0.0D0
     ZCM=0.0D0
     TOM=0.0D0
     !
     IF(IPRNT.GE.4) THEN
        WRITE(IO,*)
        WRITE(IO,*) '  Test : starting coordinates, molecule',I
        DO J=1,NAT0
           WRITE(IO,950) J,X(J+K),X(J+K+NAT),X(J+K+NAT2),AM(J+K)
        ENDDO
     ENDIF
     !
     DO J=1,NAT0
        XCM = XCM + X(J+K)*AM(J+K)
        YCM = YCM + X(J+K+NAT)*AM(J+K)
        ZCM = ZCM + X(J+K+NAT2)*AM(J+K)
        TOM = TOM + AM(J+K)
     ENDDO
     XCM = XCM/TOM
     YCM = YCM/TOM
     ZCM = ZCM/TOM
     DO J=1,NAT0
        X(J+K)      = X(J+K)      - XCM
        X(J+K+NAT)  = X(J+K+NAT)  - YCM
        X(J+K+NAT2) = X(J+K+NAT2) - ZCM
     ENDDO
     !
     IF(IPRNT.GE.4) THEN
        WRITE(IO,*)
        WRITE(IO,*) '  Test : CM coordinates, molecule',I
        WRITE(IO,950) I,XCM,YCM,ZCM
        WRITE(IO,*)
        WRITE(IO,*) '  Test : shifted coordinates, molecule',I
        DO J=1,NAT0
           WRITE(IO,950) I,X(J+K),X(J+K+NAT),X(J+K+NAT2)
        ENDDO
     ENDIF
     !
     DO M=1,3
        DO N=1,3
           AI(M,N)=0.0D0
        ENDDO
     ENDDO
     !
     ! ... calculate inertia tensor components AI
     DO J=1,NAT0
        JK=J+K
        JK1=JK+NAT
        JK2=JK1+NAT
        AI(1,1)=AI(1,1) + (X(JK1)**2 + X(JK2)**2)*AM(JK)
        AI(1,2)=AI(1,2) - X(JK)*X(JK1)*AM(JK)
        AI(1,3)=AI(1,3) - X(JK)*X(JK2)*AM(JK)
        AI(2,2)=AI(2,2) + (X(JK)**2 + X(JK2)**2)*AM(JK)
        AI(2,3)=AI(2,3) - X(JK1)*X(JK2)*AM(JK)
        AI(3,3)=AI(3,3) + (X(JK)**2 + X(JK1)**2)*AM(JK)
     ENDDO
     AI(2,1)=AI(1,2)
     AI(3,1)=AI(1,3)
     AI(3,2)=AI(2,3)
     !
     DO M=1,3
        DO N=1,3
           TEST(M,N)=AI(M,N)
        ENDDO
     ENDDO
     !
     ! ... diagonalize the moment of inertia tensor; eigenvectors in TEST
     CALL HSHLDR(TEST,PRINC,R,3,3)
     !
     ! ... Set up the coordinate transformation matrix
     DO J=1,NAT0
        JK=J+K
        JK1=JK+NAT
        JK2=JK1+NAT
        W(JK,JK)   = TEST(1,1)
        W(JK,JK1)  = TEST(2,1)
        W(JK,JK2)  = TEST(3,1)
        W(JK1,JK)  = TEST(1,2)
        W(JK1,JK1) = TEST(2,2)
        W(JK1,JK2) = TEST(3,2)
        W(JK2,JK)  = TEST(1,3)
        W(JK2,JK1) = TEST(2,3)
        W(JK2,JK2) = TEST(3,3)
     ENDDO
     !
     ! ... test output
     IF(IPRNT.GT.2) THEN
        WRITE(IO,*)
        WRITE(IO,*) ' MOLORT: molecule no.',I
        WRITE(IO,*) '  Total mass =',TOM,' # of atoms =',NAT0
        WRITE(IO,905)
        DO M=1,3
           WRITE(IO,910) M,PRINC(M),(AI(M,J),J=1,3),(TEST(J,M),J=1,3)
        ENDDO
        WRITE(IO,*)
     ENDIF
     !
     TOMM(I)      = TOM
     PRINCI(I3-2) = PRINC(1)
     PRINCI(I3-1) = PRINC(2)
     PRINCI(I3)   = PRINC(3)
     !
  ENDDO
  !            ! I=1,IZMOL
  !
905 FORMAT(1X,' I : principal values, tensor elements and', &
       ' eigenvectors (in rows)')
910 FORMAT(1X,I3,F10.4,4X,3F10.4,4X,3F10.4)
  !
  ! ... Transform cartesian coordinates
  !
  DO I=1,NAT3
     S=0.0D0
     DO J=1,NAT3
        S = S + W(I,J)*X(J)
     ENDDO
     V(I)=S
  ENDDO
  DO I=1,NAT3
     X(I)=V(I)
  ENDDO
  !
  ! ... Some more testing and printouts
  !
  IF(IPRNT.GT.2) THEN
     WRITE(IO,*)
     WRITE(IO,*) '  MOLORT: the transformed coordinates'
     WRITE(IO,*) '   #         X         Y         Z'
     WRITE(IO,*)
     DO I=1,NAT
        WRITE(IO,950) I,X(I),X(I+NAT),X(I+NAT2),AM(I)
     ENDDO
950  FORMAT(1X,I3,4X,4F10.5)
     !
     ! ... Recalculate inertia tensor components AI in new coords
     K=-MNAT(1)
     DO I=1,IZMOL
        NAT0=MNAT(I)
        K=K+NAT0
        DO M=1,3
           DO N=1,3
              AI(M,N)=0.0D0
           ENDDO
        ENDDO
        DO J=1,NAT0
           JK=J+K
           JK1=JK+NAT
           JK2=JK1+NAT
           AI(1,1)=AI(1,1) + (X(JK1)**2 + X(JK2)**2)*AM(JK)
           AI(1,2)=AI(1,2) - X(JK)*X(JK1)*AM(JK)
           AI(1,3)=AI(1,3) - X(JK)*X(JK2)*AM(JK)
           AI(2,2)=AI(2,2) + (X(JK)**2 + X(JK2)**2)*AM(JK)
           AI(2,3)=AI(2,3) - X(JK1)*X(JK2)*AM(JK)
           AI(3,3)=AI(3,3) + (X(JK)**2 + X(JK1)**2)*AM(JK)
        ENDDO
        AI(2,1)=AI(1,2)
        AI(3,1)=AI(1,3)
        AI(3,2)=AI(2,3)
        !
        WRITE(IO,*) ' MOLORT : test of X transformation '
        WRITE(IO,*) '   I tensor for molecule no.', I
        DO M=1,3
           WRITE(IO,920) M,(AI(M,N),N=1,3)
        ENDDO
        WRITE(IO,*)
     ENDDO
     !              ! (I=1,IZMOL)
  ENDIF
  !            ! (IPRNT.GT.2)
  !
920 FORMAT(1X,I3,4X,3F12.6)
  RETURN
END SUBROUTINE MOLORT

SUBROUTINE ORTHOP(U,V,NULL,NSTRT,NQ,NQM,IO)
  !
  !   MOLVIB ORTHO renamed to ORTHOP: name conflict with PATH module
  !   Gram-Schmidt procedure .
  !   The U matrix contains a set of orthonormal vectors
  !   in rows 1,...,NULL (usually, this is a null space basis);
  !   and test vectors in rows NULL+1,NULL+NSTRT.
  !   The procedure will generate a set of NULL+NSTRT orthonormal
  !   vectors (if provided set is independent).
  !   On output, the vectors in the rows NULL+1,...,NULL+NSTRT
  !   will form a basis of the vibrational space.
  !
  !   U - matrix whose rows are the coordinates of the
  !       unit vectors of the :
  !       null space basis (I=1,NULL)
  !       and indep. coords (I=NULL+1,NQ)
  !   V - work vector
  !
  use chm_kinds
  implicit none
  INTEGER NULL,NSTRT,NQ,NQM,IO
  real(chm_real) U(NQM,NQM),V(NQM),S
  !
  INTEGER I,J,N,II,NN,NT
  !
  ! ... Main loop starts here, at each pass, the existing orthonormal
  ! ... basis is in rows 1,...,NN of U, while the II=NN+1 row is
  ! ... processed
  !
  NN=NULL-1
  DO N=1,NSTRT
     NN=NN+1
     II=NN+1
     !
     ! ... Transfer next start vector into work vector V
     !
     DO J=1,NQ
        V(J)=U(II,J)
     ENDDO
     !
     ! ... Orthnormalize to existing vectors
     !
     DO I=1,NN
        ! ... scalar product
        S=0.0
        DO J=1,NQ
           S=S+V(J)*U(I,J)
        ENDDO
        ! ... orthogonalization
        DO J=1,NQ
           V(J)=V(J)-S*U(I,J)
        ENDDO
        ! ... normalization
        S=0.0
        DO J=1,NQ
           S=S+V(J)*V(J)
        ENDDO
        S=SQRT(S)
        IF(S.LT.0.00001) THEN
           WRITE(IO,*)
           WRITE(IO,*) ' ORTHO : warning, test vector dependent, N= ',N
           WRITE(IO,*)
        ELSE
           DO J=1,NQ
              V(J)=V(J)/S
           ENDDO
        ENDIF
     ENDDO
     !              ! I=1,NN
     !
     ! ... add new vector to basis, replacing the starting vector
     !
     DO J=1,NQ
        U(II,J)=V(J)
     ENDDO
     !
  ENDDO
  !            ! N=1,NSTRT
  !
  ! ... Finished : output section
  !
  NT=NULL+NSTRT
  WRITE(IO,*)
  WRITE(IO,*) ' The orthonormal coordinates '
  WRITE(IO,*) '                 null space: '
  WRITE(IO,*)
  DO I=1,NULL
     WRITE(IO,1000) I,(U(I,J),J=1,NQ)
  ENDDO
  WRITE(IO,*)
  WRITE(IO,*) '    independent coordinates: '
  WRITE(IO,*)
  DO I=NULL+1,NT
     WRITE(IO,1000) I,(U(I,J),J=1,NQ)
  ENDDO
1000 FORMAT(1X,I3,8F9.6,12(/4X,8F9.6))
  !
  RETURN
END SUBROUTINE ORTHOP

SUBROUTINE ORTNUL(U,W,T,V,NULL,NULT,NQ,NQM,IO)
  !
  !     The rows of matrix U contain some basis vectors for the
  !     null space, I=1,...,NULL, determined from literature,
  !     analytically or from 'G   ' option level=1 calculations.
  !     Here the remaining independent vectors in the null space
  !     J=NULL+1,NULT will be determined, so that they are orthogonal to
  !     the initial set.
  !
  !   U - matrix, here only NULT rows and NQ columns used
  !   W - matrix with G eigenvectors stored in columns (NQxNQ)
  !
  use chm_kinds
  implicit none
  INTEGER NULL,NULT,NQ,NQM,IO
  real(chm_real) U(NQM,NQM),W(NQM,NQM),V(NQM),T(NQM,NQM)
  real(chm_real) AMAX,S
  INTEGER I,J,K,IZERO,NZ,IMAX
  !
  ! ... First calculate scalar products of all G eigenvectors
  ! ... having null eigenvalues with the basis vectors in U
  !
  DO I=1,NULT
     DO J=1,NULL
        S=0.0
        DO K=1,NQ
           S = S + W(K,I)*U(J,K)
        ENDDO
        T(I,J)=S
     ENDDO
  ENDDO
  !
  ! ... Next eliminate contributions from the existing basis vectors
  !
  DO I=1,NULT
     DO J=1,NULL
        DO K=1,NQ
           W(K,I) = W(K,I) - T(I,J)*U(J,K)
        ENDDO
     ENDDO
  ENDDO
  !
  ! ... Find the vectors of largest norm and put them into the
  ! ... first NZ columns of W
  !
  IZERO=0
  DO I=1,NULT
     S=0.0
     DO K=1,NQ
        S = S + W(K,I)**2
     ENDDO
     IF(S.LT. 1.0E-10) THEN
        S=0.0
        IZERO=IZERO+1
     ENDIF
     V(I)=SQRT(S)
  ENDDO

  WRITE(IO,*)
  WRITE(IO,*) ' ORTNUL: Number of zero norm vectors = ',IZERO
  WRITE(IO,*)
  IF(IZERO.GE.NULT) THEN
     WRITE(IO,*) ' No independent vectors left'
     RETURN
  ENDIF
  !
  DO J=1,NULT
     AMAX=-1.0
     DO I=J,NULT
        IF(V(I).GT.AMAX) THEN
           IMAX=I
           AMAX=V(I)
        ENDIF
     ENDDO
     !
     S=V(J)
     V(J)=V(IMAX)
     V(IMAX)=S
     DO K=1,NQ
        S=W(K,J)
        W(K,J)=W(K,IMAX)
        W(K,IMAX)=S
     ENDDO
  ENDDO
  !
  ! ... The selected vectors are now added to the existing basis in U
  ! ... and normalized
  !

  NZ=NULT-NULL
  DO I=1,NZ
     DO K=1,NQ
        U(NULL+I,K) = W(K,I)/V(I)
     ENDDO
  ENDDO
  !
  ! ... output section
  !
  WRITE(IO,*)
  WRITE(IO,*) ' The orthonormal basis of the null space '
  WRITE(IO,*)
  DO I=1,NULT
     WRITE(IO,1000) I,(U(I,J),J=1,NQ)
  ENDDO
1000 FORMAT(1X,I3,8F9.6,12(/4X,8F9.6))
  !
  RETURN
END SUBROUTINE ORTNUL

SUBROUTINE STEPUP(NUMAT,NQ,NIC,NIC0,IPRNT,NAT,NAT3, &
     IFSTEP,ISTCOR,STPSIZ,IFFMAT,IFLMAT, &
     NQM,LX,X,V7,IO)
  !
  !    Control routine for the STEP option
  !
  !    IFSTEP - selects type of step, ISTCOR - coordinate number
  !         =1 step along LX eigenvector #ISTCOR
  !         =2 step along LS eigenvector #ISTCOR
  !         =3 step along IC #ISTCOR
  !    IFFMAT,IFLMAT - define the starting point of the calculation
  !    IFFMAT=0 and IFLMAT=
  !                        1 - LX matrix has been read in
  !                        2 - LS matrix has been read in
  !    IFLMAT=0 and IFFMAT=
  !                        1 - FX matrix has been read in
  !                        2 - FS matrix has been read in
  !
  !    NUMAT,NQ,NIC,NIC0,NAT,NAT3 - as in molvib.f
  !
  !    (N.B. Some options may not be implemented immediately)
  !----------------------------------------------------------------
  !                            K.Kuczera, 08-Feb-1990             |
  !----------------------------------------------------------------
  !
  !
  use chm_kinds
  implicit none
  INTEGER NUMAT,NQ,NIC,NIC0,IPRNT,NAT,NAT3,NQM
  INTEGER IFSTEP,ISTCOR,IFFMAT,IFLMAT,IO
  real(chm_real) LX(NQM,NQM),X(NAT3),V7(NAT3)
  !
  real(chm_real) S,STPSIZ
  INTEGER I,J
  !
  DO J=1,NAT3
     V7(J)=X(J)
  ENDDO
  !-----------------------------------------------------------------
  !
  ! ... IFSTEP=1 branch
  !
  IF(IFSTEP.EQ.1) THEN
     IF(IFFMAT.LE.0 .AND. IFLMAT.EQ.1) THEN
        WRITE(IO,*)
        WRITE(IO,*) ' STEPUP : the LX matrix has been provided'
        WRITE(IO,*)
        !
        ! ... Print header
        !
        IF(IPRNT.GE.2) THEN
           WRITE(IO,*)
           WRITE(IO,*) ' STEPUP : stepping along eigenvector', &
                ISTCOR
           WRITE(IO,*)
           WRITE(IO,*)  ' LX(I,ISTCOR),I=1,NAT3 '
           WRITE(IO,1005) (LX(I,ISTCOR),I=1,NAT3)
        ENDIF
        !
        ! ... Check eigenvector normalization
        !
        S=0.0D0
        DO J=1,NAT3
           S=S+LX(J,ISTCOR)**2
        ENDDO
        IF(S.LT.1.0D-6) THEN
           WRITE(IO,*) ' **** Error in STEPUP, norm too small'
           CALL WRNDIE(-4,'<STEPUP>',' SO LONG ')
        ENDIF
        S=SQRT(S)
        DO J=1,NAT3
           LX(J,ISTCOR)=LX(J,ISTCOR)/S
        ENDDO
        !
        ! ... Transform
        !
        DO J=1,NAT3
           X(J) = X(J) + STPSIZ*LX(J,ISTCOR)
        ENDDO
        !
     ENDIF
     !              !   (IFFMAT.LE.0 .AND. IFLMAT.EQ.1)
     !
     IF(IFFMAT.GT.0 .OR. IFLMAT.NE.1) THEN
        WRITE(IO,*)
        WRITE(IO,*) ' STEPUP : option not implemented '
        WRITE(IO,*) '       See you later, Al'
     ENDIF
     !
  ENDIF
  !            ! IFSTEP.EQ.1
  !-----------------------------------------------------------------
  !
  ! ... IFSTEP=2 branch
  !
  IF(IFSTEP.EQ.2) THEN
     WRITE(IO,*)
     WRITE(IO,*) ' STEPUP : option not implemented '
     WRITE(IO,*) '       See you later, Al'
  ENDIF
  !            ! IFSTEP.EQ.2
  !------------------------------------------------------------------
  !
  ! ... IFSTEP=3 branch
  !
  IF(IFSTEP.EQ.3) THEN
     WRITE(IO,*)
     WRITE(IO,*) ' STEPUP : option not implemented '
     WRITE(IO,*) '       See you later, Al'
  ENDIF
  !            ! IFSTEP.EQ.3
  !
  !------------------------------------------------------------------
  !
  ! ... Final section : ouput distorted coordinates
  !
  WRITE(IO,*)
  WRITE(IO,*) '    The new cartesian coordinates '
  WRITE(IO,*)
  WRITE(IO,*) '   #        X          Y          Z'
  WRITE(IO,*)
  DO J=1,NAT
     WRITE(IO,1006) J,X(J),X(J+NAT),X(J+2*NAT)
  ENDDO
  !
  IF(IPRNT.GE.2) THEN
     WRITE(IO,*) ' Testing of tranformation'
     DO J=1,NAT3
        V7(J)=S*(V7(J)-X(J))/STPSIZ
     ENDDO
     WRITE(IO,1005) (V7(I),I=1,NAT3)
     WRITE(IO,*)
  ENDIF
  !
1005 FORMAT(14X,10F9.4,12(/14X,10F9.4))
1006 FORMAT(1X,I3,2X,3F11.5)
  RETURN
END SUBROUTINE STEPUP

SUBROUTINE KANONR(NIC0,NQ,IPRNT,NUMAT,NQM,W1,FX,FS,LX,IO)
  !
  !   Transform F matrix (in primitive internal coordinates R)
  !   to canonic form FR* using eigenvectors of G matrix with non-zero
  !   eigenvalues :
  !                     FR* = WT*W*F*WT*W
  !
  !   On input arrays contain
  !   W1 - G matrix eigenvectors in R coords (NIC0,NIC0)
  !   FS -  force constant matrix in R coords (NIC0,NIC0)
  !   dimensions:
  !   NIC0 - number of primitive internal coordinates R
  !   NQ  - number of vibrational degrees of freedom
  !
  !   On output the FS contains the FR* force field,
  !                 FX the projection operator matrix WT*W
  !                 LX the G eigenvectors in primitive IC's R
  !
  use chm_kinds
  implicit none
  INTEGER NIC0,NQ,IPRNT,NUMAT,NQM,IO
  real(chm_real) W1(NQM,NQM),FX(NQM,NQM),FS(NQM,NQM),LX(NQM,NQM)
  !
  INTEGER I,J,K,KK,NRED
  real(chm_real) S
  !
  !
  WRITE(IO,*)
  WRITE(IO,*) '     Calculating FR* = WT*W*FR*WT*W'
  WRITE(IO,*)
  !
  ! ... Generate projection operator on the vibrational space
  ! ...  WT*W
  !
  DO I=1,NIC0
     DO J=1,NIC0
        LX(I,J)=W1(I,J)
     ENDDO
  ENDDO
  !
  NRED=NIC0-NQ
  WRITE(IO,*) ' There are',NRED,'   null coordinates'
  WRITE(IO,*)
  DO I=1,NIC0
     DO J=1,NIC0
        S=0.0
        DO K=1,NQ
           KK=K+NRED
           S = S + W1(I,KK)*W1(J,KK)
        ENDDO
        FX(I,J)=S
     ENDDO
  ENDDO
  !
  ! ... Perform projection operation on F
  !
  DO I=1,NIC0
     DO J=1,NIC0
        S=0.0
        DO K=1,NIC0
           S = S + FX(I,K)*FS(K,J)
        ENDDO
        W1(I,J)=S
     ENDDO
  ENDDO
  !
  DO I=1,NIC0
     DO J=1,NIC0
        S=0.0
        DO K=1,NIC0
           S = S + W1(I,K)*FX(K,J)
        ENDDO
        FS(I,J)=S
     ENDDO
  ENDDO
  !
  IF(IPRNT.GE.2) THEN
     WRITE(IO,*)
     WRITE(IO,*) ' The matrix FR*=WT*W*FR*WT*W'
     WRITE(IO,*)
     DO I=1,NIC0
        WRITE(IO,1006) I,(FS(I,J),J=1,NIC0)
     ENDDO
  ENDIF
  !
1006 FORMAT(1X,I3,8F9.3,12(/4X,8F9.3))
  RETURN
END SUBROUTINE KANONR

SUBROUTINE PED0(NAT,NAT3,NQ,NIC,NIC0,IPRNT, &
     ISTER,NUMAT,CUTPED, &
     NQM,B,LX,LS,W2,DD,V1,WPED,IPED,INDX,NBLMAX,IO)
  !
  !   MOLVIB PED routine renamed to PED0: name conflict with VIBRAN module
  !   Calculate PED matrix from vibrational eigenvectors
  !   given in independent coordinates
  !   The contribution of internal coordinate J to the energy
  !   of mode K is
  !      PED(K,J)=LS(J,K)*[LS-1](K,J)
  !
  !   B - vibrational B matrix; (NIC,NAT3)
  !   U - on input : S=U*R gives transformation to indep.
  !       internal coords S;  (NQ,NIC)
  !   LX -  on input: eigenvectors in cartesian coords; (NAT3,NQ)
  !   LS -  work array for eigenvector transformations
  !   DD - vector of frequencies (NQ)
  !   W - work vector
  !   dimensions:
  !   NAT - number of atoms, NAT3=NAT*3
  !   NIC - number of internal coordinates
  !   NQ  - number of vibrational degrees of freedom
  !
  !   ISTER - selects options
  !           =0 - start from cartesian eigenvectors
  !           =1 - start form indep. internal eigenvectors
  !   parameters for reduced PED:
  !   CUTPED - minimum contribution which will be printed out
  !   NPED   - max number of contributions to be printed
  !
  !   Note: The cartesian eigenvectors are destroyed here,
  !          overwritten by inverse of LS
  !
  use chm_kinds
  implicit none
  INTEGER NQM,NAT,NAT3,NQ,NIC,NIC0,IPRNT,ISTER,NUMAT
  INTEGER NBLMAX,IO
  real(chm_real) B(NQM,NQM),LX(NQM,NQM),LS(NQM,NQM),W2(NQM,NQM)
  real(chm_real) V1(NQM),DD(NQM)
  !
  INTEGER I,J,K,N,KPED,NPED
  real(chm_real) WPED(NQM),CUTPED,S
  INTEGER IPED(NQM),INDX(NQM)
  !
  !
  NPED = NQ
  !
  IF(ISTER.NE.0) GOTO 150
  !
  ! ... eigenvectors LS are obtained from cartesian
  ! ... eigenvectors LX
  ! ... LS = U*B*LX , U=U1 for NUMAT=1, U=U2*U1 for NUMAT=2
  !
  !
  DO I=1,NQ
     DO J=1,NQ
        S=0.0
        DO K=1,NAT3
           S = S + B(I,K)*LX(K,J)
        ENDDO
        LS(I,J)=S
     ENDDO
  ENDDO
  !
  ! ... Section for ISTER=1 follows :
  ! ... LS already contains eigenvectors in independent internal
  ! ... coordinates
  ! ... make copy of LS in W2
  !
150 CONTINUE
  DO I=1,NQ
     DO J=1,NQ
        W2(I,J)=LS(I,J)
     ENDDO
  ENDDO
  !
  IF(IPRNT.GT.1) THEN
     WRITE(IO,*)
     WRITE(IO,*) ' LS matrix '
     WRITE(IO,*)
     DO I=1,NQ
        WRITE(IO,1006) I,(LS(I,J),J=1,NQ)
     ENDDO
  ENDIF
1006 FORMAT(1X,I3,8F9.6,12(/4X,8F9.6))
  !
  ! ... Now put inverse of indep. IC eigenvector matrix into LX
  !
  !      CALL MATNV(W2,LX,V1,NQ,NQM)
  CALL MATNVN(W2,LX,V1,INDX,NQ,NQM)
  !
  ! ... Finally put  PED into W2 array : eigenvector matrix is in LS,
  ! ... inverse matrix in LX
  !
  IF(IPRNT.GE.3) THEN
     WRITE(IO,*)
     WRITE(IO,*) 'The inverse of the LS matrix'
     WRITE(IO,*)
     DO I=1,NQ
        WRITE(IO,1006) I,(LX(I,J),J=1,NQ)
     ENDDO
  ENDIF
  DO I=1,NQ
     DO J=1,NQ
        W2(I,J)=LS(J,I)*LX(I,J)
     ENDDO
  ENDDO
  !
  ! ... Output section
  !
  IF(IPRNT.GE.1) THEN
     WRITE(IO,*)
     WRITE(IO,*) ' FULL POTENTIAL ENERGY DISTRIBUTION MATRIX'
     WRITE(IO,*)
     DO I=1,NQ
        WRITE(IO,1000) I,DD(I),(W2(I,J),J=1,NQ)
     ENDDO
1000 FORMAT(1X,I3,F10.1,10F5.2,12(/14X,10F5.2))
     !
     WRITE(IO,*)
     WRITE(IO,*) ' REDUCED POTENTIAL ENERGY DISTRIBUTION MATRIX [%]'
     WRITE(IO,*)
     !
     DO I=1,NQ
        DO N=1,NPED
           IPED(N)=0
           WPED(N)=0.0
        ENDDO
        KPED=0
        DO N=1,NQ
           IF(ABS(W2(I,N)).GT.CUTPED) THEN
              KPED=KPED+1
              IF(KPED.LE.NPED) THEN
                 WPED(KPED)=100.0*W2(I,N)
                 IPED(KPED)=N
              ELSE
                 WRITE(IO,*) 'Warning: too many terms for mode ',I
              ENDIF
           ENDIF
        ENDDO
        WRITE(IO,1005) I,DD(I),(IPED(J),WPED(J),J=1,KPED)
     ENDDO
  ENDIF
1005 FORMAT(1X,I3,F10.1,5(3X,I3,F5.0),4(/14X,5(3X,I3,F5.0)))
  !
  RETURN
END SUBROUTINE PED0

SUBROUTINE PED3(W,DD,PPED,ZPED,NQ,NQM,IFSYMM,IFEXPF, &
     EXPFRQ,CFREQ, &
     MAXSYM,NGMAX,NBLMAX,LGRUP,IGRUP,KSYMB, &
     NGRUP,NSYMB,SYMB,SPED,SBLOCK,IPTA,IPTB, &
     IBLOCK,NBLOCK,QSEL,CUTPED,IO)
  !
  !    Performs a more detaled analysis of the Potential Energy
  !    Distribution matrix (stored in W array)
  !    Called when the 'PED ',NGRUP card is given in input deck
  !    This card is followed by two lists:
  !    - list of coord groups in the form
  !    n,ci1,ci2,...,cin  (free format, gives total no of coords
  !      in group, followed by the numbers of those coords)
  !    - list of individual coordinates and their symbols
  !    (c(i),s(i),i=1,...,4) - individual coords and their symbols
  !      4(I4,6X,A8,2X) 0-4 in line, negative value ends file.
  !
  !     Differences from PED2: performs symmetry blocking
  !
  !    Two types of coordinates are defined :
  !    a) indvidual - a symbol is specified for each of those
  !    b) groups    - a symbol is given only for the first coord
  !                   of group, rest set to this in code
  !    NQ - number of vibrational degrees of freedom
  !    NQM - array dimension in calling subroutine
  !    DD - vibrational frequencies in cm-1
  !    W - PED matrix (calculated as W2 in ped.f)
  !    IPTB - pointer array for symmetry blocking returned by SYMSOR
  !
  !
  use chm_kinds
  implicit none
  INTEGER MAXSYM,NGMAX,NBLMAX,IO
  INTEGER LGRUP(NGMAX),IGRUP(NGMAX,MAXSYM),KSYMB(MAXSYM)
  INTEGER NGRUP,NSYMB
  CHARACTER(len=*) SYMB(MAXSYM)
  CHARACTER(len=*) SPED(NBLMAX)
  CHARACTER(len=*) SBLOCK(NBLMAX)
  INTEGER IPTA(NBLMAX),IBLOCK(NBLMAX),NBLOCK
  INTEGER IPTB(NBLMAX)
  LOGICAL QSEL(NBLMAX)
  real(chm_real) CUTPED
  !
  !
  INTEGER I,J,K,L,IB,JB,KS,NPED,IHI,IBLI,ILO,IPB
  INTEGER NQ,NQM,IFSYMM,IFEXPF
  real(chm_real) PPED(NQ),ZPED(NQ),W(NQM,NQM),DD(NQ),EXPFRQ(NQ)
  real(chm_real) SSQ,SSQT,CFREQ(NQ)
  !
  ! ... First sort eigenvalues and PED rows in symmetry blocks
  !
  DO I=1,NQ
     IPTB(I)=I
  ENDDO
  IF(IFSYMM.GT.0) CALL SYMSOR(W,DD,NQ,NQM,IBLOCK,NBLOCK,IPTB)
  !
  ! ... Now sort the IC contributions to each mode and output
  !
  WRITE(IO,*)
  WRITE(IO,*) '   Symbolic PED matrix [%] (sorted)'
  WRITE(IO,*)
  IB=0
  JB=1
  DO I=1,NQ
     IPB=IPTB(I)
     !
     ! ... Print symmetry block header
     !
     IF(IFSYMM.GT.0) THEN
        IB=IB+1
        IF(IB.EQ.1) THEN
           WRITE(IO,*)
           WRITE(IO,*) ' Symmetry block ',SBLOCK(JB)
           WRITE(IO,*)
        ENDIF
        IF(IB.EQ.IBLOCK(JB)) THEN
           IB=0
           JB=JB+1
        ENDIF
     ENDIF
     !
     DO K=1,NSYMB
        PPED(K)=0.0D0
     ENDDO
     !
     DO J=1,NQ
        KS=KSYMB(J)
        PPED(KS)=PPED(KS)+W(IPB,J)
     ENDDO
     !
     NPED=0
     DO K=1,NSYMB
        IF(ABS(PPED(K)).GT.CUTPED) THEN
           NPED=NPED+1
           ZPED(NPED)=PPED(K)*100.0
           SPED(NPED)=SYMB(K)
        ENDIF
     ENDDO
     !
     CALL BRUSOR(ZPED,IPTA,QSEL,NPED)
     WRITE(IO,900) I,DD(IPB), &
          (SPED(IPTA(L)),ZPED(IPTA(L)),L=1,NPED)
     !
  ENDDO
  !            ! I=1,NQ
  !
900 FORMAT(1X,I3,1X,F8.1,4(3X,A8,F5.0),12(/13X,4(3X,A8,F5.0)))
  !
  ! ... Section for rms deviation from reference frequencies
  ! ... CFREQ contains the symmetry sorted frequencies, DD - only acsending order
  !
  SSQT=0.0D0
  IF(IFEXPF.GT.0) THEN
     DO I=1,NQ
        IPB=IPTB(I)
        CFREQ(I)=DD(IPB)
     ENDDO
     IHI=0
     DO I=1,NBLOCK
        IBLI=IBLOCK(I)
        ILO=IHI+1
        IHI=IHI+IBLI
        IF(IFSYMM.GT.0) THEN
           WRITE(IO,*)
           WRITE(IO,*) ' Symmetry block ',SBLOCK(I)
           WRITE(IO,*)
        ENDIF
        !
        CALL DIFFRQ(CFREQ,EXPFRQ,NQ,ILO,IHI,SSQ,IO)
        SSQT=SSQT+SSQ
        !
     ENDDO
  ENDIF
  !
  IF(NBLOCK.GT.1) THEN
     WRITE(IO,*)
     WRITE(IO,*) ' Total sum of squares and rms deviation:'
     WRITE(IO,910) SSQT,SQRT(SSQT/NQ)
  ENDIF
910 FORMAT(11X,2F12.2)
  !
  RETURN
END SUBROUTINE PED3
#else /**/
SUBROUTINE NULL_MS
  RETURN
END SUBROUTINE NULL_MS
#endif /*  MOLVIB*/

