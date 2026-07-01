!CHARMM Element source/util/matrix.src 1.1
      SUBROUTINE DETM33(U, DET)
!
!     DET is the determinant of the double precision 3x3 matrix U.
!
  use chm_kinds
      implicit none
!
      real(chm_real)  DET, U(3,3)
!
      DET = U(1,1) * (U(2,2)*U(3,3) - U(2,3)*U(3,2)) - &
            U(1,2) * (U(2,1)*U(3,3) - U(3,1)*U(2,3)) + &
            U(1,3) * (U(2,1)*U(3,2) - U(3,1)*U(2,2))
!
      RETURN
#if KEY_PNM==1 /*called by the PNM code*/
      END
!
      SUBROUTINE DIAGRS(OPTION, NDIM, MATRIX, EVAL, EVEC)
!
!     Diagonalise a real symmetric matrix. Matrix is destroyed
!     during the diagonalisation. The matrix can be stored in
!     the full square form or in upper or lower triangle form.
!
!     OPTION    The storage form of the matrix.
!     NDIM      The dimension of the matrix.
!     MATRIX    The second derivative matrix.
!     EVAL      The eigenvalues.
!     EVEC      The eigenvectors.
!
  use chm_kinds
  use stream
  use allocation, only: chmalloc
  use deallocation, only: chmdealloc
      implicit none
! . Passed variables.
      CHARACTER(len=*) OPTION
      INTEGER   NDIM
      real(chm_real)    MATRIX(*), EVAL(NDIM), EVEC(NDIM,NDIM)
! . Local variables.
      real(chm_real),allocatable,dimension(:) :: WORK1
      real(chm_real),allocatable,dimension(:) :: WORK2
      real(chm_real),allocatable,dimension(:) :: WORK3
      real(chm_real),allocatable,dimension(:) :: WORK4
      real(chm_real),allocatable,dimension(:) :: WORK5
      real(chm_real),allocatable,dimension(:) :: WORK6
      real(chm_real),allocatable,dimension(:) :: WORK7
      real(chm_real),allocatable,dimension(:) :: WORK8
      INTEGER   IER, JOBN, WDIM
! . Pointers.
! . Do some initialisation.
! . Upper triangle form.
      IF (OPTION  ==  'UPPER') THEN
         WDIM  = NDIM + 1
         call chmalloc('matrix.src','DIAGRS','WORK1',WDIM,crl=WORK1)
         call chmalloc('matrix.src','DIAGRS','WORK2',WDIM,crl=WORK2)
         call chmalloc('matrix.src','DIAGRS','WORK3',WDIM,crl=WORK3)
         call chmalloc('matrix.src','DIAGRS','WORK4',WDIM,crl=WORK4)
         call chmalloc('matrix.src','DIAGRS','WORK5',WDIM,crl=WORK5)
         call chmalloc('matrix.src','DIAGRS','WORK6',WDIM,crl=WORK6)
         call chmalloc('matrix.src','DIAGRS','WORK7',WDIM,crl=WORK7)
         call chmalloc('matrix.src','DIAGRS','WORK8',WDIM,crl=WORK8)
         CALL DIAGQ(NDIM, NDIM, MATRIX, EVEC, WORK1, &
                    WORK2, WORK3, WORK4, WORK5, WORK6, &
                    WORK7, WORK8, 0)
! . Copy eigenvalues to the eigenvalue array.
         eval(1:ndim) = WORK5(1:NDIM)

! free memory
         call chmdealloc('matrix.src', 'DIAGRS', 'WORK1', WDIM, crl=WORK1)
         call chmdealloc('matrix.src', 'DIAGRS', 'WORK2', WDIM, crl=WORK2)
         call chmdealloc('matrix.src', 'DIAGRS', 'WORK3', WDIM, crl=WORK3)
         call chmdealloc('matrix.src', 'DIAGRS', 'WORK4', WDIM, crl=WORK4)
         call chmdealloc('matrix.src', 'DIAGRS', 'WORK5', WDIM, crl=WORK5)
         call chmdealloc('matrix.src', 'DIAGRS', 'WORK6', WDIM, crl=WORK6)
         call chmdealloc('matrix.src', 'DIAGRS', 'WORK7', WDIM, crl=WORK7)
         call chmdealloc('matrix.src', 'DIAGRS', 'WORK8', WDIM, crl=WORK8)

! . Lower triangle or full square form.
      ELSE
         IF (OPTION  ==  'LOWER') THEN
            JOBN = 1
         ELSE IF (OPTION  ==  'FULL') THEN
            JOBN = 11
         ENDIF
         IER   = 0
         CALL EIGRS(MATRIX, NDIM, JOBN, EVAL, EVEC, NDIM, IER)
         IF (IER  >  128) THEN
            IER = IER - 128
            IF(WRNLEV >= 2) WRITE (OUTU,'(A,I6,A)') &
            ' DIAGRS> Failed to converge on root number ', IER, '.'
         ENDIF
      ENDIF
!
      RETURN
#endif /* used in PNM code */
      END
!
      SUBROUTINE EIGRS(A,N,JOBN,D,Z,IZ,IER)
!                                  SPECIFICATIONS FOR ARGUMENTS
  use chm_kinds
  use number
      implicit none
      INTEGER            N,JOBN,IZ,IER
      real(chm_real)   A(*),D(*),Z(IZ,*)
!                                  SPECIFICATIONS FOR LOCAL VARIABLES
      real(chm_real) :: WK(N)  ! Automatic array, will be on system stack, but should be OK/LNI
      INTEGER            IJOB,IR,JR,IJ,JI,NP1
      INTEGER            JER,NA,ND,IIZ,IBEG,IL,KK,LK,I,J,K,L
      real(chm_real)   ANORM,ASUM,PI,SUMZ,SUMR,AN,S,RDELP
!CC      DATA               RDELP/2.775557562D-17/
!                                  INITIALIZE ERROR PARAMETERS
!                                  FIRST EXECUTABLE STATEMENT
      RDELP=RPRECI
      IER = 0
      JER = 0
      IF (JOBN < 10) GO TO 15
!                                  CONVERT TO SYMMETRIC STORAGE MODE
      K = 1
      JI = N-1
      IJ = 1
      DO J=1,N
         DO I=1,J
            A(K) = A(IJ)
            IJ = IJ+1
            K = K+1
         ENDDO
         IJ = IJ + JI
         JI = JI - 1
      ENDDO
   15 IJOB = MOD(JOBN,10)
      IF (IJOB >= 0.AND.IJOB <= 3) GO TO 20
!                                  WARNING ERROR - IJOB IS NOT IN THE
!                                    RANGE
      IER = 66
      IJOB = 1
      GO TO 25
   20 IF (IJOB == 0) GO TO 35
   25 IF (IZ >= N) GO TO 30
!                                  WARNING ERROR - IZ IS LESS THAN N
!                                    EIGENVECTORS CAN NOT BE COMPUTED,
!                                    IJOB SET TO ZERO
      IER = 67
      IJOB = 0
   30 IF (IJOB == 3) GO TO 75
   35 NA = (N*(N+1))/2
      IF (IJOB /= 2) GO TO 45
      DO 40 I=1,NA
         WK(I) = A(I)
   40 CONTINUE
!                                  SAVE INPUT A IF IJOB = 2
   45 ND = 1
      IF (IJOB == 2) ND = NA+1
!                                  REDUCE A TO SYMMETRIC TRIDIAGONAL
!                                    FORM
!mu...25-Jul-93 Remove the second WK(ND) in the call to EHOUSS
!mu   CALL EHOUSS(A,N,D,WK(ND),WK(ND))
      CALL EHOUSS(A,N,D,WK(ND))
      IIZ = 1
      IF (IJOB == 0) GO TO 60
      IIZ = IZ
!                                  SET Z TO THE IDENTITY MATRIX
      DO 55 I=1,N
         DO 50 J=1,N
            Z(I,J) = ZERO
   50    CONTINUE
         Z(I,I) = ONE
   55 CONTINUE
!                                  COMPUTE EIGENVALUES AND EIGENVECTORS
   60 CALL EQRT2S(D,WK(ND),N,Z,IIZ,JER)
      IF (IJOB == 0) GO TO 9000
      IF (JER > 128) GO TO 65
!                                  BACK TRANSFORM EIGENVECTORS
      CALL EHOBKS(A,N,1,N,Z,IZ)
   65 IF (IJOB <= 1) GO TO 9000
!                                  MOVE INPUT MATRIX BACK TO A
      DO 70 I=1,NA
         A(I) = WK(I)
   70 CONTINUE
      WK(1) = THOSND
      IF (JER /= 0) GO TO 9000
!                                  COMPUTE 1 - NORM OF A
   75 ANORM = ZERO
      IBEG = 1
      DO 85 I=1,N
         ASUM = ZERO
         IL = IBEG
         KK = 1
         DO 80 L=1,N
            ASUM = ASUM+ABS(A(IL))
            IF (L >= I) KK = L
            IL = IL+KK
   80    CONTINUE
         ANORM = MAX(ANORM,ASUM)
         IBEG = IBEG+I
   85 CONTINUE
      IF (ANORM == ZERO) ANORM = ONE
!                                  COMPUTE PERFORMANCE INDEX
      PI = ZERO
      DO 100 I=1,N
         IBEG = 1
         S = ZERO
         SUMZ = ZERO
         DO 95 L=1,N
            LK = IBEG
            KK = 1
            SUMZ = SUMZ+ABS(Z(L,I))
            SUMR = -D(I)*Z(L,I)
            DO 90 K=1,N
               SUMR = SUMR+A(LK)*Z(K,I)
               IF (K >= L) KK = K
               LK = LK+KK
   90       CONTINUE
            S = S+ABS(SUMR)
            IBEG = IBEG+L
   95    CONTINUE
         IF (SUMZ == ZERO) GO TO 100
         PI = MAX(PI,S/SUMZ)
  100 CONTINUE
      AN = N
      PI = PI/(ANORM*TEN*AN*RDELP)
      WK(1) = PI
      IF (JOBN < 10) GO TO 9000
!                                  CONVERT BACK TO FULL STORAGE MODE
      NP1 = N+1
      IJ = (N-1)*NP1 + 2
      K = (N*(NP1))/2
      DO 110 JR=1,N
         J = NP1-JR
         DO 105 IR=1,J
            IJ = IJ-1
            A(IJ) = A(K)
            K = K-1
  105    CONTINUE
         IJ = IJ-JR
  110 CONTINUE
      JI = 0
      K = N-1
      DO 120 I=1,N
         IJ = I-N
         DO 115 J=1,I
            IJ = IJ+N
            JI = JI+1
            A(IJ) = A(JI)
  115    CONTINUE
         JI = JI + K
         K = K-1
  120 CONTINUE
 9000 CONTINUE
!MJF  IF (IER /= 0) CALL UERTST(IER,6HEIGRS )
      IF (JER == 0) GO TO 9005
      IER = JER
!MJF  CALL UERTST(IER,6HEIGRS )
 9005 RETURN
      END SUBROUTINE EIGRS
!
      SUBROUTINE EHOBKS(A,N,M1,M2,Z,IZ)
!
  use chm_kinds
      implicit none
      INTEGER N,M1,M2,IZ
      real(chm_real)  A(*),Z(IZ,*)
!
      INTEGER I,J,K,L,IA
      real(chm_real)   H,S
!                                  FIRST EXECUTABLE STATEMENT
      IF (N  ==  1) GO TO 30
      DO 25 I=2,N
         L = I-1
         IA = (I*L)/2
         H = A(IA+I)
         IF (H == 0.D0) GO TO 25
!                                  DERIVES EIGENVECTORS M1 TO M2 OF
!                                  THE ORIGINAL MATRIX FROM EIGENVECTORS
!                                  M1 TO M2 OF THE SYMMETRIC
!                                  TRIDIAGONAL MATRIX
         DO 20 J = M1,M2
            S = 0.0D0
            DO 10 K = 1,L
               S = S+A(IA+K)*Z(K,J)
   10       CONTINUE
            S = S/H
            DO 15 K=1,L
               Z(K,J) = Z(K,J)-S*A(IA+K)
   15       CONTINUE
   20    CONTINUE
   25 CONTINUE
   30 RETURN
      END SUBROUTINE EHOBKS
!
      SUBROUTINE EHOUSS(A,N,D,E)
!-----------------------------------------------------------------------
!     called by EIGRS
!mu...25-Jul-93 M.E. Karpen.  The original E2 vector is the square
!     of the E vector.  The EIGRS routine passed the same vector (WK)
!     for both E & E2, which caused conflicting, machine-dependent results. 
!     Since EIGRS uses only the E vector, and is currently the only routine
!     that calls EHOUSS, E2 has been eliminated to prevent this ambiguity.
!
  use chm_kinds
      implicit none
      INTEGER  N
      real(chm_real)   A(*),D(N),E(N)
!
      INTEGER  JK1,JP1,NP1,I,J,K,L,NBEG,II,IK,JK,NK,NN
      real(chm_real)   ZERO,H,SCALE,F,G,HH
      DATA     ZERO/0.0D0/
!                                  FIRST EXECUTABLE STATEMENT
      NP1 = N+1
      NN = (N*NP1)/2-1
      NBEG = NN+1-N
      DO 70 II = 1,N
         I = NP1-II
         L = I-1
         H = ZERO
         SCALE = ZERO
         IF (L  <  1) GO TO 10
!                                  SCALE ROW (ALGOL TOL THEN NOT NEEDED)
         NK = NN
         DO K = 1,L
            SCALE = SCALE+ABS(A(NK))
            NK = NK-1
         ENDDO
         IF (SCALE  /=  ZERO) GO TO 15
   10    E(I) = ZERO
!mu...25-Jul-93 remove E2 usage (NOT used)
!mu         E2(I) = ZERO
         GO TO 65
   15    NK = NN
         DO 20 K = 1,L
            A(NK) = A(NK)/SCALE
            H = H+A(NK)*A(NK)
            NK = NK-1
   20    CONTINUE
!mu         E2(I) = SCALE*SCALE*H
         F = A(NN)
         G = -SIGN(SQRT(H),F)
         E(I) = SCALE*G
         H = H-F*G
         A(NN) = F-G
         IF (L  ==  1) GO TO 55
         F = ZERO
         JK1 = 1
         DO 40 J = 1,L
            G = ZERO
            IK = NBEG+1
            JK = JK1
!                                  FORM ELEMENT OF A*U
            DO 25 K = 1,J
               G = G+A(JK)*A(IK)
               JK = JK+1
               IK = IK+1
   25       CONTINUE
            JP1 = J+1
            IF (L  <  JP1) GO TO 35
            JK = JK+J-1
            DO 30 K = JP1,L
               G = G+A(JK)*A(IK)
               JK = JK+K
               IK = IK+1
   30       CONTINUE
!                                  FORM ELEMENT OF P
   35       E(J) = G/H
            F = F+E(J)*A(NBEG+J)
            JK1 = JK1+J
   40    CONTINUE
         HH = F/(H+H)
!                                  FORM REDUCED A
         JK = 1
         DO 50 J = 1,L
            F = A(NBEG+J)
            G = E(J)-HH*F
            E(J) = G
            DO 45 K = 1,J
               A(JK) = A(JK)-F*E(K)-G*A(NBEG+K)
               JK = JK+1
   45       CONTINUE
   50    CONTINUE
   55    DO 60 K = 1,L
            A(NBEG+K) = SCALE*A(NBEG+K)
   60    CONTINUE
   65    D(I) = A(NBEG+I)
         A(NBEG+I) = H*SCALE*SCALE
         NBEG = NBEG-I+1
         NN = NN-I
   70 CONTINUE
      RETURN
      END SUBROUTINE EHOUSS
!
      SUBROUTINE EQRT2S(D,E,N,Z,IZ,IER)
!
  use chm_kinds
  use number
      implicit none
      INTEGER  N,IZ,IER
      real(chm_real)   D(*),E(*),Z(IZ,*)
!
      INTEGER  IP1,MM1,I,J,K,L,M,L1,II,MM1PL
      real(chm_real)   B,C,F,G,H,P,R,S,RDELP
!CC      DATA               RDELP/2.775557562D-17/
!                                  MOVE THE LAST N-1 ELEMENTS
!                                  OF E INTO THE FIRST N-1 LOCATIONS
!                                  FIRST EXECUTABLE STATEMENT
      RDELP=RPRECI
      IER  = 0
      IF (N  ==  1) GO TO 9005
      DO I=2,N
         E(I-1) = E(I)
      ENDDO
      E(N) = ZERO
      B = ZERO
      F = ZERO
      DO  60  L=1,N
         J = 0
         H = RDELP*(ABS(D(L))+ABS(E(L)))
         IF (B < H) B = H
!                                  LOOK FOR SMALL SUB-DIAGONAL ELEMENT
         DO 10  M=L,N
            K=M
            IF (ABS(E(K))  <=  B) GO TO 15
   10    CONTINUE
   15    M = K
         IF (M == L) GO TO 55
   20    IF (J  ==  30) GO TO 85
         J = J+1
         L1 = L+1
         G = D(L)
         P = (D(L1)-G)/(E(L)+E(L))
         R = ABS(P)
         IF (RDELP*ABS(P)  <  1.0D0) R = SQRT(P*P+ONE)
         D(L) = E(L)/(P+SIGN(R,P))
         H = G-D(L)
         DO 25 I = L1,N
            D(I) = D(I)-H
   25    CONTINUE
         F = F+H
!                                  QL TRANSFORMATION
         P = D(M)
         C = ONE
         S = ZERO
         MM1 = M-1
         MM1PL = MM1+L
         IF (L > MM1) GO TO 50
         DO 45 II=L,MM1
            I = MM1PL-II
            G = C*E(I)
            H = C*P
            IF (ABS(P) < ABS(E(I))) GO TO 30
            C = E(I)/P
            R = SQRT(C*C+ONE)
            E(I+1) = S*P*R
            S = C/R
            C = ONE/R
            GO TO 35
   30       C = P/E(I)
            R = SQRT(C*C+ONE)
            E(I+1) = S*E(I)*R
            S = ONE/R
            C = C*S
   35       P = C*D(I)-S*G
            D(I+1) = H+S*(C*G+S*D(I))
            IF (IZ  <  N) GO TO 45
!                                  FORM VECTOR
            DO 40 K=1,N
               H = Z(K,I+1)
               Z(K,I+1) = S*Z(K,I)+C*H
               Z(K,I) = C*Z(K,I)-S*H
   40       CONTINUE
   45    CONTINUE
   50    E(L) = S*P
         D(L) = C*P
         IF (ABS(E(L))  > B) GO TO 20
   55    D(L) = D(L) + F
   60 CONTINUE
!                                  ORDER EIGENVALUES AND EIGENVECTORS
      DO  80  I=1,N
         K = I
         P = D(I)
         IP1 = I+1
         IF (IP1 > N) GO TO 70
         DO 65  J=IP1,N
            IF (D(J)  >=  P) GO TO 65
            K = J
            P = D(J)
   65    CONTINUE
   70    IF (K == I) GO TO 80
         D(K) = D(I)
         D(I) = P
         IF (IZ  <  N) GO TO 80
         DO 75 J = 1,N
            P = Z(J,I)
            Z(J,I) = Z(J,K)
            Z(J,K) = P
   75    CONTINUE
   80 CONTINUE
      GO TO 9005
   85 IER = 128+L
!MJF  CALL UERTST(IER,6HEQRT2S)
 9005 RETURN
      END SUBROUTINE EQRT2S
!
      SUBROUTINE INVT33(V, U, OK)
!
!     Find the inverse of a double precision 3x3 matrix.
!
  use chm_kinds
      implicit none
!
      INTEGER I, J
      LOGICAL OK
      real(chm_real)  DET, U(3,3), V(3,3)
!
      real(chm_real)  TOL
      PARAMETER (TOL = 1.0D-10)
!
      OK = .TRUE.
!
      V(1,1) =   U(2,2)*U(3,3) - U(2,3)*U(3,2)
      V(2,1) =   U(2,3)*U(3,1) - U(2,1)*U(3,3)
      V(3,1) =   U(2,1)*U(3,2) - U(2,2)*U(3,1)
      V(1,2) =   U(1,3)*U(3,2) - U(1,2)*U(3,3)
      V(2,2) =   U(1,1)*U(3,3) - U(1,3)*U(3,1)
      V(3,2) =   U(1,2)*U(3,1) - U(1,1)*U(3,2)
      V(1,3) =   U(1,2)*U(2,3) - U(1,3)*U(2,2)
      V(2,3) =   U(2,1)*U(1,3) - U(1,1)*U(2,3)
      V(3,3) =   U(1,1)*U(2,2) - U(2,1)*U(1,2)
!
      DET = U(1,1)*V(1,1) + U(1,2)*V(2,1) + U(1,3)*V(3,1)
      IF (ABS(DET)  <=  TOL) THEN
         OK = .FALSE.
      ELSE
         DO 20 I=1,3
            DO 10 J=1,3
               V(J,I) = V(J,I) / DET
   10       CONTINUE
   20    CONTINUE
      ENDIF
      RETURN
      END SUBROUTINE INVT33
!
      SUBROUTINE INVT33S(V, U, OK)
!
!     Find the inverse of a double precision 3x3 symmetric matrix.
!     stored as a lower triangle.
!     Invers matrix is also stored as a lower triangle.
!
  use chm_kinds
      implicit none
!
      INTEGER I
      LOGICAL OK
      real(chm_real)  DET, U(6), V(6)
!
      real(chm_real)  TOL
      PARAMETER (TOL = 1.0D-10)
!
      V(1) =  U(3)*U(6) - U(5)*U(5)
      V(2) =  U(5)*U(4) - U(2)*U(6)
      V(3) =  U(1)*U(6) - U(4)*U(4)
      V(4) =  U(2)*U(5) - U(3)*U(4)
      V(5) =  U(2)*U(4) - U(1)*U(5)
      V(6) =  U(1)*U(3) - U(2)*U(2)
!
      DET = U(1)*V(1) + U(2)*V(2) + U(4)*V(4)
      IF (ABS(DET)  <=  TOL) THEN
         OK = .FALSE.
      ELSE
         OK = .TRUE.
         DO 10 I=1,6
            V(I) = V(I) / DET
   10    CONTINUE
      ENDIF
      RETURN
      END SUBROUTINE INVT33S
!
      SUBROUTINE MULMXN(RESULT, A, NA1, NA2, B, NB1, NB2)
!
!     Multiply two matrices A and B to give RESULT. The dimensions of
!     A are NA1 x NA2 and those of B are NB1 x NB2. RESULT has dimension
!     NA1 x NB2. All arrays are double precision.
!
  use chm_kinds
  use stream
      implicit none
!
      INTEGER I, J, NA1, NA2, NB1, NB2
      real(chm_real)  A(NA1,NA2), B(NB1,NB2), RESULT(NA1,NB2)
!
      INTEGER K
      real(chm_real)  ZERO
      PARAMETER (ZERO = 0.D0)
!
      IF (NA2  ==  NB1) THEN
         DO 30 I = 1,NB2
            DO 20 J = 1,NA1
               RESULT(J,I) = ZERO
               DO 10 K = 1,NA2
                  RESULT(J,I) = RESULT(J,I) + A(J,K) * B(K,I)
   10          CONTINUE
   20       CONTINUE
   30    CONTINUE
      ELSE
         IF(WRNLEV >= 2) WRITE (OUTU,'(A)') &
                              ' MULMXN> Invalid matrix dimensions.'
      ENDIF
      RETURN
      END SUBROUTINE MULMXN
!
      SUBROUTINE MULNXN(RESULT, A, B, N)
!
!     Multiply the NxN matrices A and B to give RESULT. The matrices
!     are assumed to be stored column first and are double precision.
!
  use chm_kinds
      implicit none
!
      INTEGER I, J, K, N
      real(chm_real)  A(N,N), B(N,N), RESULT(N,N)
!
      real(chm_real)  ZERO
      PARAMETER (ZERO = 0.D0)
!
      DO 30 I = 1,N
         DO 20 J = 1,N
            RESULT(J,I) = ZERO
            DO 10 K = 1,N
               RESULT(J,I) = RESULT(J,I) + A(J,K) * B(K,I)
   10       CONTINUE
   20    CONTINUE
   30 CONTINUE
      RETURN
      END SUBROUTINE MULNXN

      SUBROUTINE MULNXNFL(RESULT, A, B, N)
!
!     Multiply the NxN matrices A and B to give RESULT. The matrices
!     are assumed to be stored column first and are double precision.
!     B is symmetric matrix stored as lower triangle.
!
  use chm_kinds
      implicit none
!
      INTEGER I, J, K, N
      real(chm_real)  A(N,N), B(*), RESULT(N,N)
!
      real(chm_real)  ZERO
      PARAMETER (ZERO = 0.D0)
      INTEGER INDEX
      INDEX(K,I) = MAX(K,I)*(MAX(K,I)-1)/2 + MIN(K,I)
!
      DO 30 I = 1,N
         DO 20 J = 1,N
            RESULT(J,I) = ZERO
            DO 10 K = 1,N
               RESULT(J,I) = RESULT(J,I) + A(J,K) * B(INDEX(K,I))
   10       CONTINUE
   20    CONTINUE
   30 CONTINUE
      RETURN
      END SUBROUTINE MULNXNFL

      SUBROUTINE MULNXNLF(RESULT, A, B, N)
!
!     Multiply the NxN matrices A and B to give RESULT. The matrices
!     are assumed to be stored column first and are double precision.
!     A is symmetric matrix stored as lower triangle.
!
  use chm_kinds
      implicit none
!
      INTEGER I, J, K, N
      real(chm_real)  A(*), B(N,N), RESULT(N,N)
!
      real(chm_real)  ZERO
      PARAMETER (ZERO = 0.D0)
      INTEGER INDEX
      INDEX(K,I) = MAX(K,I)*(MAX(K,I)-1)/2 + MIN(K,I)
!
      DO 30 I = 1,N
         DO 20 J = 1,N
            RESULT(J,I) = ZERO
            DO 10 K = 1,N
               RESULT(J,I) = RESULT(J,I) + A(INDEX(J,K)) * B(K,I)
   10       CONTINUE
   20    CONTINUE
   30 CONTINUE
      RETURN
      END  SUBROUTINE MULNXNLF

      SUBROUTINE MULNXNLL(RESULT, A, B, N)
!
!     Multiply the NxN matrices A and B to give RESULT. The matrices
!     are assumed to be stored column first and are double precision.
!     A & B are symmetric matrices stored as lower triangle.
!
  use chm_kinds
      implicit none
!
      INTEGER I, J, K, N
      real(chm_real)  A(*), B(*), RESULT(N,N)
!
      real(chm_real)  ZERO
      PARAMETER (ZERO = 0.D0)
      INTEGER INDEX
      INDEX(K,I) = MAX(K,I)*(MAX(K,I)-1)/2 + MIN(K,I)
!
      DO 30 I = 1,N
         DO 20 J = 1,N
            RESULT(J,I) = ZERO
            DO 10 K = 1,N
               RESULT(J,I) = RESULT(J,I) + A(INDEX(J,K)) * B(INDEX(K,I))
   10       CONTINUE
   20    CONTINUE
   30 CONTINUE
      RETURN
      END SUBROUTINE MULNXNLL

      SUBROUTINE MULNXNU(RESULT, A, B, N)
!
!     Multiply the NxN matrices A and B to give RESULT. The matrices
!     are assumed to be stored column first and are double precision.
!     RESULT is symmetric matrix stored as upper triangle.
!
  use chm_kinds
      implicit none
!
      INTEGER N
      real(chm_real)  A(N,N), B(N,N), RESULT(*)
!
      INTEGER I, J, K,IPT
      real(chm_real)  R
!
      IPT=0
      DO I = 1,N
         DO J = I,N
            IPT=IPT+1
            R = 0.0
            DO K = 1,N
               R = R + A(J,K) * B(K,I)
            ENDDO
            RESULT(IPT) = R
         ENDDO
      ENDDO
      RETURN
      END SUBROUTINE MULNXNU

      SUBROUTINE MULNXNFU(RESULT, A, B, N)
!
!     Multiply the NxN matrices A and B to give RESULT. The matrices
!     are assumed to be stored column first and are double precision.
!     B is symmetric matrix stored as upper triangle.
!
  use chm_kinds
      implicit none
!
      INTEGER N
      real(chm_real)  A(N,N), B(*), RESULT(N,N)
!
      INTEGER I, J, K
      real(chm_real)  R
!
      INTEGER INDEX
      INDEX(K,I) = (MIN(K,I)-2*N)*(1-MIN(K,I))/2 + MAX(K,I)
!
      DO I = 1,N
         DO J = 1,N
            R = 0.0
            DO K = 1,N
               R = R + A(J,K) * B(INDEX(K,I))
            ENDDO
            RESULT(J,I) = R
         ENDDO
      ENDDO
      RETURN
      END SUBROUTINE MULNXNFU

      SUBROUTINE MULNXNUF(RESULT, A, B, N)
!
!     Multiply the NxN matrices A and B to give RESULT. The matrices
!     are assumed to be stored column first and are double precision.
!     A is symmetric matrix stored as upper triangle.
!
  use chm_kinds
      implicit none
!
      INTEGER N
      real(chm_real)  A(*), B(N,N), RESULT(N,N)
!
      INTEGER I, J, K
      real(chm_real)  R
!
      INTEGER INDEX
      INDEX(K,I) = (MIN(K,I)-2*N)*(1-MIN(K,I))/2 + MAX(K,I)
!
      DO I = 1,N
         DO J = 1,N
            R = 0.0
            DO K = 1,N
               R = R + A(INDEX(J,K)) * B(K,I)
            ENDDO
            RESULT(J,I) = R
         ENDDO
      ENDDO
      RETURN
      END SUBROUTINE MULNXNUF

      SUBROUTINE MULNXNUU(RESULT, A, B, N)
!
!     Multiply the NxN matrices A and B to give RESULT. The matrices
!     are assumed to be stored column first and are double precision.
!     A & B are symmetric matrices stored as upper triangle.
!
  use chm_kinds
      implicit none
!
      INTEGER N
      real(chm_real)  A(*), B(*), RESULT(N,N)
!
      INTEGER I, J, K
      real(chm_real)  R
!
      INTEGER INDEX
      INDEX(K,I) = (MIN(K,I)-2*N)*(1-MIN(K,I))/2 + MAX(K,I)
!
      DO I = 1,N
         DO J = 1,N
            R = 0.0
            DO K = 1,N
               R = R + A(INDEX(J,K)) * B(INDEX(K,I))
            ENDDO
            RESULT(J,I) = R
         ENDDO
      ENDDO
      RETURN
      END SUBROUTINE MULNXNUU

!
      SUBROUTINE MULTTRM(A,B,C)
  !
  ! This routine multiplies two rotation translation matrices
  !
  !     C = B * A
  !
  ! Do a copy at the end to allow duplicate use of arrays - BRB
  !
      use number
      implicit none

      real(chm_real) A(3,4),B(3,4),C(3,4)
  !
      INTEGER I,J,K
      real(chm_real) TEMP(3,4)
  !
      DO I=1,4
         DO J=1,3
            TEMP(J,I)=ZERO
            DO K=1,3
               TEMP(J,I)=TEMP(J,I)+A(J,K)*B(K,I)
            ENDDO
         ENDDO
      ENDDO
      DO J=1,3
         TEMP(J,4)=TEMP(J,4)+A(J,4)
      ENDDO
  !
      DO I=1,4
         DO J=1,3
            C(J,I)=TEMP(J,I)
         ENDDO
      ENDDO
  !
      RETURN
      END SUBROUTINE MULTTRM
!
      SUBROUTINE MULTTRMR(A,B,C)
  !
  ! This routine multiplies two rotation translation matrices
  !
  !     C = A * B
  !
  ! Do a copy at the end to allow duplicate use of arrays - BRB
  !
      use number
      implicit none

      real(chm_real) A(3,4),B(3,4),C(3,4)
  !
      INTEGER I,J,K
      real(chm_real) TEMP(3,4)
  !
      DO I=1,3
         DO J=1,3
            TEMP(J,I)=ZERO
            DO K=1,3
               TEMP(J,I)=TEMP(J,I)+A(J,K)*B(K,I)
            ENDDO
         ENDDO
      ENDDO

      DO J=1,3
         TEMP(J,4)=ZERO
         DO K=1,3
            TEMP(J,4)=TEMP(J,4)+B(K,J)*A(K,4)
         ENDDO
      ENDDO
      DO J=1,3
         TEMP(J,4)=TEMP(J,4)+B(J,4)
      ENDDO
  !
      DO I=1,4
         DO J=1,3
            C(J,I)=TEMP(J,I)
         ENDDO
      ENDDO
  !
      RETURN
      END SUBROUTINE MULTTRMR
!
      SUBROUTINE TRANSPS(RESULT, A, N1, N2)
!
!     Transpose a double precision N1xN2 matrix, A, into RESULT.
!
  use chm_kinds
      implicit none
!
      INTEGER I, J, N1, N2
      real(chm_real)  A(N1,N2), RESULT(N2,N1)
!
      DO 20 I = 1,N1
         DO 10 J = 1,N2
            RESULT(J,I) = A(I,J)
   10    CONTINUE
   20 CONTINUE
      RETURN
      END SUBROUTINE TRANSPS
!
      SUBROUTINE EIGCH(A,N,JOBN,D,Z,IZ,WK,IER)
      use chm_kinds
      implicit none
      INTEGER            N,JOBN,IZ,IER
      real(chm_real)             A(*),D(N),Z(*),WK(*)
      INTEGER            JER,K,I,NE,NTAU,NA,NI,NI2,IM1,J,IIZ,NZ,IIZ1, &
                         IJOB,JR,IR,IJ,JI,NP1, &
                         JZ,JZI,L,M,II,IL,KK,LZ,MZ,LK,KZ
      real(chm_real)     ANORM,ASUM,PI,SUMZ,SUMR,SUMI,S,TEN,RDELP, &
                         ZERO,ONE,THOUS,AN,SIGNA
      DATA               RDELP/2.775557562D-17/
      DATA               ZERO,ONE/0.0D0,1.0D0/,TEN/10.0D0/,THOUS/1000.0D0/
      IER = 0
      JER = 0
      IF (JOBN < 10) GO TO 15
      JR = N + N - 2
      IJ = 2
      K = 2
      DO J=1,N
         DO I=1,J
            A(K-1) = A(IJ-1)
            A(K) = -A(IJ)
            K = K+2
            IJ = IJ + 2
         ENDDO
         IJ = IJ + JR
         JR = JR - 2
      ENDDO
   15 IJOB = MOD(JOBN,10)
      IF (IJOB >= 0.AND.IJOB <= 3) GO TO 20
      IER = 66
      IJOB = 1
      GO TO 25
   20 IF (IJOB == 0) GO TO 45
   25 IF (IZ >= N) GO TO 30
      IER = 67
      IJOB = 0
   30 K = 2
      DO 40 I=1,N
         IF (A(K) == ZERO) GO TO 35
         A(K) = ZERO
         IER = 68
   35    K = K+I+I+2
   40 CONTINUE
      IF (IJOB == 3) GO TO 110
   45 NE = 1
      NTAU = NE+N
      NA = NTAU+N+N
      NI = (N*(N+1))/2
      NI2 = NI+NI
      IF (IJOB /= 2) GO TO 55
      K = NA
      DO 50 I=1,NI2
         WK(K) = A(I)
         K = K+1
   50 CONTINUE
   55 IF (NI < 2) GO TO 70
      IM1 = 1
      DO 65 I=2,NI
         K = IM1+I
         PI = A(K)
         DO 60 J=1,IM1
            A(K) = A(K-1)
            K = K-1
   60    CONTINUE
         A(I) = PI
         IM1 = I
   65 CONTINUE
   70 CALL EHOUSH (A(1),A(NI+1),N,D,WK(NE),WK(NTAU))
      IIZ = 1
      IF (IJOB /= 0) IIZ = IZ+IZ
      IF (IIZ == 1) GO TO 85
      NZ = (IZ+IZ)*N
      DO 75 I=1,NZ
         Z(I) = ZERO
   75 CONTINUE
      K = 1
      IIZ1 = IIZ+1
      DO 80 I=1,N
         Z(K) = ONE
         K = K+IIZ1
   80 CONTINUE
   85 CALL EQRT2S (D,WK(NE),N,Z(1),IIZ,JER)
      IF (IJOB == 0) GO TO 9000
      CALL EHBCKH (A(1),A(NI+1),WK(NTAU),N,Z(1),Z(IZ+1),IIZ)
      JZ = 0
      DO 100 J=1,N
         JZI = JZ+IZ
         DO 90 I=1,N
            K = JZI+I
            WK(I) = Z(K)
   90    CONTINUE
         K = JZ+N
         L = K+N-1
         M = N
         DO 95 I=1,N
            Z(L) = Z(K)
            Z(L+1) = WK(M)
            K = K-1
            L = L-2
            M = M-1
   95    CONTINUE
         JZ = JZ+IZ+IZ
  100 CONTINUE
      IF (IJOB /= 2) GO TO 9000
      K = NA
      DO 105 I=1,NI2
         A(I) = WK(K)
         K = K+1
  105 CONTINUE
      WK(1) = THOUS
      IF (JER /= 0) GO TO 9000
  110 ANORM = ZERO
      II = 1
      DO 120 I=1,N
         ASUM = ZERO
         IL = II
         KK = 2
         DO 115 L=1,N
            ASUM = ASUM+ABS(cmplx(A(IL),A(IL+1),chm_cmpx))
            IF (L >= I) KK = L+L
            IL = IL+KK
  115    CONTINUE
         ANORM = MAX(ANORM,ASUM)
         II = II+I+I
  120 CONTINUE
      IF (ANORM == ZERO) ANORM = ONE
      PI = ZERO
      DO 135 I=1,N
         II = 1
         S = ZERO
         SUMZ = ZERO
         LZ = IZ*(I-1)*2+1
         MZ = LZ
         DO 130 L=1,N
            LK = II
            KK = 2
            SUMZ = SUMZ+ABS(cmplx(Z(LZ),Z(LZ+1),chm_cmpx))
            SUMR = -D(I)*Z(LZ)
            SUMI = -D(I)*Z(LZ+1)
            KZ = MZ
            DO 125 K=1,N
               SIGNA = ONE
               IF (K > L) SIGNA = -ONE
               SUMR = SUMR+A(LK)*Z(KZ)-SIGNA*A(LK+1)*Z(KZ+1)
               SUMI = SUMI+A(LK)*Z(KZ+1)+SIGNA*A(LK+1)*Z(KZ)
               IF (K >= L) KK = K+K
               LK = LK+KK
               KZ = KZ+2
  125       CONTINUE
            S = S+ABS(cmplx(SUMR,SUMI,chm_cmpx))
            LZ = LZ+2
            II = II+L+L
  130    CONTINUE
         IF (SUMZ == ZERO) GO TO 135
         PI = MAX(PI,S/SUMZ)
  135 CONTINUE
      AN = N
      PI = PI/(ANORM*TEN*AN*RDELP)
      WK(1) = PI
      IF (JOBN < 10) GO TO 9000
      NP1 = N + 1
      IJ = (N-1) * NP1
      IJ = IJ + IJ + 2
      K = N * NP1
      DO 145 JR=1,N
         J = N+1-JR
         DO 140 IR=1,J
            A(IJ-1) = A(K-1)
            A(IJ) = -A(K)
            K = K-2
            IJ = IJ - 2
  140    CONTINUE
         IJ = IJ - JR - JR
  145 CONTINUE
      JR = N + N
      II = 2
      JI = 2
      DO 155 I=1,N
         IJ = II
         DO 150 J=1,I
            A(IJ-1) = A(JI-1)
            A(IJ) = -A(JI)
            JI = JI+2
            IJ = IJ+JR
  150    CONTINUE
         JI = JI + JR - I - I
         II = II + 2
  155 CONTINUE
 9000 CONTINUE
!MJF      IF (IER /= 0) CALL UERTST (IER,6HEIGCH )
      IF (JER == 0) GO TO 9005
      IER = JER
!MJF      CALL UERTST (IER,6HEIGCH )
 9005 RETURN
      END SUBROUTINE EIGCH
!
      SUBROUTINE EHBCKH(AR,AI,TAU,N,ZR,ZI,IZ)
  use chm_kinds
      implicit none
      INTEGER            N,IZ
      real(chm_real)             AR(*),AI(*),TAU(2,*),ZR(IZ,*),ZI(IZ,*)
      INTEGER            J,K,NR,L,NRM1,INX1,INX2,K1
      real(chm_real)             DELTA,ZERO,ALPHA1,ALPHA2
      DATA               ZERO/0.0D0/
      DO J=1,N
         DO K=1,N
            ZI(J,K)=-ZR(J,K)*TAU(2,J)
            ZR(J,K)=ZR(J,K)*TAU(1,J)
         ENDDO
      ENDDO
      IF (N  <=  2) GO TO 30
      DO 25 L=3,N
         NR=N-L+2
         NRM1=NR-1
         INX1=(NR*(NRM1))/2+NR
         INX2=INX1-1
         IF (AI(INX1)  ==  ZERO) GO TO 25
         DELTA=AI(INX1)* SQRT(AR(INX2)**2+AI(INX2)**2)
         DO 20 J=1,N
            ALPHA1=ZERO
            ALPHA2=ZERO
            DO 10 K=NR,N
               K1=(K*(K-1))/2+NRM1
               ALPHA1=ALPHA1+AR(K1)*ZR(K,J)+AI(K1)*ZI(K,J)
               ALPHA2=ALPHA2-AI(K1)*ZR(K,J)+AR(K1)*ZI(K,J)
   10       CONTINUE
            ALPHA1=ALPHA1/DELTA
            ALPHA2=ALPHA2/DELTA
            DO 15 K=NR,N
               K1=(K*(K-1))/2+NRM1
               ZR(K,J)=ZR(K,J)-AR(K1)*ALPHA1+AI(K1)*ALPHA2
               ZI(K,J)=ZI(K,J)-AR(K1)*ALPHA2-AI(K1)*ALPHA1
   15       CONTINUE
   20    CONTINUE
   25 CONTINUE
   30 RETURN
      END SUBROUTINE EHBCKH
!
      SUBROUTINE EHOUSH(AR,AI,N,D,E,TAU)
      use chm_kinds
      implicit none
      INTEGER            N
      real(chm_real)             AR(*),AI(*),D(*),E(*),TAU(2,*)
      INTEGER            NM1,NN,I,NR,NRM1,L,INDX,J,JJ,INX1,INX2,JP1,KK, &
                         IX,IM1
      real(chm_real)   RHO,TOLER,ZERO,ONE,T1,T2,TESTBB,VR,ROOT,DELTA, &
                         RATIO,RDELP,Q1,Q2,X1,X2,TT1,TT2,BB
      DATA               ZERO/0.0D0/,ONE/1.0D0/
      DATA               RDELP/2.775557562D-17/
      NM1=N-1
      TOLER=ZERO
      NN=(N*(N+1))/2
      DO I=1,NN
         T1=ABS(AR(I))
         T2=ABS(AI(I))
         IF(T2 > T1) T1=T2
         IF (T1 > TOLER) TOLER=T1
      ENDDO
      TESTBB=RDELP*TOLER
      IF (N <= 2) GO TO 65
      DO 60 NR=2,NM1
         NRM1=NR-1
         VR=ZERO
         TAU(1,NR)=ZERO
         TAU(2,NR)=ZERO
         TAU(2,1)=ZERO
         DO 10 L=NR,N
            INDX=(L*(L-1))/2+NRM1
            VR=AR(INDX)**2+AI(INDX)**2+VR
   10    CONTINUE
         INDX=(NR*NRM1)/2+NRM1
         IF ((TESTBB)**2  >=  VR) GO TO 60
         ROOT = ABS(cmplx(AR(INDX),AI(INDX),chm_cmpx))*SQRT(VR)
         IF(ROOT /= ZERO) GO TO 15
         AR(INDX)=SQRT(VR)
         DELTA=VR
         TAU(1,1)=-AR(INDX)
         GO TO 20
   15    DELTA=VR+ROOT
         RATIO=VR/ROOT
         TAU(1,1)=-RATIO*AR(INDX)
         TAU(2,1)= RATIO*AI(INDX)
         AR(INDX)=(RATIO+ONE)*AR(INDX)
         AI(INDX)=(RATIO+ONE)*AI(INDX)
   20    DO 35 J=NR,N
            JJ=(J*(J-1))/2
            INDX=JJ+NRM1
            TAU(1,J)=AR(INDX)/DELTA
            TAU(2,J)=AI(INDX)/DELTA
            D(J)=ZERO
            E(J)=ZERO
            DO 25 L=NR,J
               INX1=(L*(L-1))/2+NRM1
               INX2=JJ+L
               D(J)= D(J)+AR(INX2)*AR(INX1)-AI(INX2)*AI(INX1)
               E(J)= E(J)+AR(INX2)*AI(INX1)+AI(INX2)*AR(INX1)
   25       CONTINUE
            JP1=J+1
            IF (JP1  >  N) GO TO 40
            DO 30 L=JP1,N
               KK=(L*(L-1))/2
               INX1=KK+NRM1
               INX2=KK+J
               D(J)=D(J)+AR(INX2)*AR(INX1)+AI(INX2)*AI(INX1)
               E(J)=E(J)+AR(INX2)*AI(INX1)-AI(INX2)*AR(INX1)
   30       CONTINUE
   35    CONTINUE
   40    RHO=ZERO
         DO 45 L=NR,N
            RHO=RHO+D(L)*TAU(1,L)+E(L)*TAU(2,L)
   45    CONTINUE
         IX=(NRM1*(NR-2))/2
         DO 55 I=NR,N
            IX=IX+I-1
            INX2=IX+NRM1
            DO 50 J=NR,I
               INX1=IX+J
               X1=TAU(1,I)*D(J)+TAU(2,I)*E(J)
               X2=TAU(2,I)*D(J)-TAU(1,I)*E(J)
               Q1=D(I)-RHO*AR(INX2)
               Q2=E(I)-RHO*AI(INX2)
               T1=Q1*TAU(1,J)+Q2*TAU(2,J)
               T2=Q2*TAU(1,J)-Q1*TAU(2,J)
               AR(INX1)=AR(INX1)-X1-T1
               AI(INX1)=AI(INX1)-X2-T2
   50       CONTINUE
   55    CONTINUE
         TAU(1,NR)=TAU(1,1)
         TAU(2,NR)=TAU(2,1)
   60 CONTINUE
   65 INDX=0
      DO 70 I=1,N
         INDX=INDX+I
         D(I)=AR(INDX)
   70 CONTINUE
      TAU(1,1)=ONE
      TAU(2,1)=ZERO
      E(1)=ZERO
      IF (N  ==  1) GO TO 85
      INDX=(N*NM1)/2+NM1
      TAU(1,N)=AR(INDX)
      TAU(2,N)=-AI(INDX)
      INDX=1
      DO 80 I=2,N
         INDX=INDX+I
         IM1=I-1
         BB= SQRT(TAU(1,I)**2+TAU(2,I)**2)
         E(I)=BB
         AI(INDX)=BB
         IF (TESTBB  <  BB) GO TO 75
         TAU(1,I)=ONE
         TAU(2,I)=ZERO
         BB=ONE
   75    TT1=TAU(1,I)*TAU(1,IM1)-TAU(2,I)*TAU(2,IM1)
         TT2=TAU(1,I)*TAU(2,IM1)+TAU(2,I)*TAU(1,IM1)
         TAU(1,I)=TT1/BB
         TAU(2,I)=TT2/BB
   80 CONTINUE
   85 RETURN
      END SUBROUTINE EHOUSH
!
      subroutine ltof(full,lowt,ndim,n)
!
!     copies lower triangle to full matrix
!
  use chm_kinds
      implicit none
      integer ndim,n,i,j,ij
      real(chm_real) full(ndim,n),lowt(*)
!
      ij=0
      do i=1,n
        do j=1,i
          ij=ij+1
          full(i,j)=lowt(ij)
          full(j,i)=lowt(ij)
        enddo
      enddo
      return
      END subroutine ltof
!
      subroutine utof(full,upt,ndim,n)
!
!     copies upper triangle to full matrix
!
  use chm_kinds
      implicit none
      integer ndim,n,i,j,ij
      real(chm_real) full(ndim,n),upt(*)
!
      ij=0
      do i=1,n
        do j=i,n
          ij=ij+1
          full(i,j)=upt(ij)
          full(j,i)=upt(ij)
        enddo
      enddo
      return
      END subroutine utof
!
      subroutine ltou(upt,lowt,n)
!
!     copies lower triangle to upper triangle matrix
!
  use chm_kinds
      implicit none
      integer n
      real(chm_real) upt(*),lowt(*)
      integer i,j,iju,ijl
      do i=1,n
        do j=1,i
!         ijl=max(i,j)*(max(i,j)-1)/2 + min(i,j)
!         iju=(min(i,j)-1)*(2*n-min(i,j))/2 + max(i,j)
          ijl=i*(i-1)/2 + j
          iju=(j-1)*(2*n-j)/2 + i
          upt(iju)=lowt(ijl)
        enddo
      enddo
      return
      END subroutine ltou
!
      subroutine addltou(upt,lowt,n)
!
!     adds lower triangle to upper triangle matrix
!
  use chm_kinds
      implicit none
      integer n
      real(chm_real) upt(*),lowt(*)
      integer i,j,iju,ijl
      do i=1,n
        do j=1,i
!         ijl=max(i,j)*(max(i,j)-1)/2 + min(i,j)
!         iju=(min(i,j)-1)*(2*n-min(i,j))/2 + max(i,j)
          ijl=i*(i-1)/2 + j
          iju=(j-1)*(2*n-j)/2 + i
          upt(iju)=upt(iju)+lowt(ijl)
        enddo
      enddo
      return
      END subroutine addltou

! JZ_UW12: Stuff from old CHARMM (needed for SMBP)
      SUBROUTINE OLDCHM_MATVEC(RESULT, A, B, N)
!
!     Multiply the NxN matrix A and vector B to give RESULT. The matrices
!     are assumed to be stored column first and are double precision.
!     A is symmetric matrix stored as upper triangle.
!
      use chm_kinds
      implicit none

      INTEGER N
      real(chm_real)  A(N,N), B(N), RESULT(N)

      INTEGER I, J, K
      real(chm_real)  R
      DO J = 1,N
         R = 0.0
         DO K = 1,N
            R = R + A(J,K) * B(K)
         ENDDO
         RESULT(J) = R
      ENDDO
      RETURN
      END SUBROUTINE OLDCHM_MATVEC



