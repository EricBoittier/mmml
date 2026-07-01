! 
! **********************************************************************
!
!  This is a collection of subroutines designated to solve the real
!  general symmetric eigenvalue problem with or without eigenvectors.
!  The routines have been taken from different freeware FORTRAN
!  libraries and optimized by hand (or eye ?! ;-)). Most of the
!  optimizations have been done with respect to stride minimization
!  for the innermost loops of the subroutines. Problems with
!  bugs, roaches and other lifestock please report to
!
!  Dirk Porezag   porezag@physik.tu-chemnitz.de
!
!  or to your nearest pest control agency (I doubt they will help).
!  Have fun !!
!
!  Copyright for this file by Dirk Porezag
!  Washington, DC, Janurary 8th, 1995 
!
! **********************************************************************
! 
!     SUBROUTINE EWEVGE 
!     ================= 
! 
! ********************************************************************** 
! 
!  Evevge calculates eigenvalues and eigenvectors of the general 
!  symmetric eigenvalue problem.
! 
!  Method:  *  A*C = E*S*C 
!           *  Choleski decomposition  S = R'*R
!           *  A*C = E*R'*R*C  ->  INV(R')*A*C = E*R*C
!           *  Transformation Y = R*C  ->  C = INV(R)*Y
!           *  Solve INV(R')*A*INV(R)*Y = E*Y  (Householder + IQL) 
!           *  Back transformation C = INV(R)*Y
!           *  Sorting of eigenvalues and eigenvectors 
! 
!     Parameters: 
! 
!       NA      (I) :  Dimension of A 
!       NB      (I) :  Dimension of B 
!       N       (I) :  Dimension of Problem  
!       A       (I) :  Matrix A (lower triangle)
!               (O) :  Eigenvector matrix  
!       B       (I) :  Matrix B (lower triangle)
!               (O) :  R where B = R'*R (upper triangle) 
!       EW      (O) :  Eigenvalues 
!       H       (-) :  Auxiliary vector 
!       IEV     (I) :  0: No eigenvectors  
!       IORD    (I) :  1: Descending order of eigenvalues 
!                     -1: Ascending order of eigenvalues
!                      otherwise: no sorting 
!       IER     (O) :  Error indication  
!                      0: No error 
!                      K: (K <= N)  B is not positive definite
!                      K: (K > N) Convergence failure for eigenvalue
!                                 (K-N), (K-N-1) eigenvalues are correct
! 
! ********************************************************************** 
!
      SUBROUTINE EWEVGE (NA,NB,N,A,B,EW,H,IEV,IORD,IER) 
        IMPLICIT REAL*8 (A-H,O-Z)
        DIMENSION  A(NA,N),B(NB,N),EW(N),H(N)
!
        IER = 0 
        EPS = 0.0
        CALL CHOLES(N,B,NB,IER) 
        IF (IER .NE. 0) RETURN 
        CALL MATRAF(N,A,NA,B,NB,H) 
        CALL TRIDIA(NA,N,EW,H,A,IEV) 
        CALL IQLDIA(NA,N,EW,H,A,IEV,IER) 
        IF (IER .GT. 0) IER = IER+N 
        IF (IER .NE. 0)  RETURN 
        IF (IEV .NE. 0) CALL BACKTR(N,N,B,NB,A,NA,A,NA,H) 
        II = 0 
        IF (IEV .NE. 0) II = 1 
        CALL SORTVC(NA,N,N,EW,A,IORD,II,H)
        RETURN 
      END
!
#if KEY_DFTBMKL==1
************************************************************************
!     Eigenvalue-solver for IBM RS6000/SP2 machines with esslp2 library
!
!     Parameters:
!
!       NA      (I) :  Dimension of A
!       NB      (I) :  Dimension of B
!       N       (I) :  Dimension of Problem
!       A       (I) :  Matrix A
!               (O) :  Eigenvector matrix
!       B       (I) :  Matrix B
!       EW      (O) :  Eigenvalues
!       H       (-) :  Dummy vector
!       AUX     (-) :  Auxiliary vector
!       IEV     (I) :  0: No eigenvectors
!       IER     (O) :  Dummy variable
!
! **********************************************************************
      SUBROUTINE EWEVGE_DSYGV (NA,NB,N,A,B,EW,H,IEV,IORD,IER)
      IMPLICIT NONE
      !IMPLICIT REAL*8 (A-H,O-Z)
      REAL*8 A,B,EW,H,AUX
      INTEGER NA, NB, N
      INTEGER IORD, IER, IEV
      DIMENSION  A(NA,N),B(NB,N),EW(N),H(N),AUX(3*NA)
      CHARACTER*1 JOBZ

!
! INFO FROM RCS
!
        character*65 rcs_author
        character*65 rcs_date
        character*65 rcs_source
        character*65 rcs_revision
        rcs_author='$Author: ckoe,koenig $'
        rcs_date='$Date: 2002/05/18 20:49:40 $'
        rcs_source='$RCSfile: ewevge.lapack.f,v $'
        rcs_revision='$Revision: 2.0 $'
!
! END RCS INFO
!
      IF (IEV.EQ.0) THEN
         JOBZ='N'
      ELSE
         JOBZ='V'
      ENDIF

      CALL DSYGV(IEV,JOBZ,'L',N,A,NA,B,NB,EW,AUX,3*NA,IER)

      RETURN
      END

      SUBROUTINE EWEVGE_DSYGVD (NA,NB,N,A,B,EW,H,IEV,IORD,IER)
      IMPLICIT NONE
!      INCLUDE 'maxima.inc'
      INTEGER NA, NB, N
      INTEGER IORD, IER, IEV
      REAL*8 A,B,EW,H
      DIMENSION  A(NA,N),B(NB,N),EW(N),H(N)
      REAL*8 WORK,IW
      DIMENSION  WORK( 1 + 6*NA + 2 * NA* NA )
      DIMENSION  IW( 3 + 5 * NA )
      CHARACTER*1 JOBZ

      IF (IEV.EQ.0) THEN
         JOBZ='N'
      ELSE
         JOBZ='V'
      ENDIF

      CALL DSYGVD(1,JOBZ,'L',N,A,NA,B,NB,EW,
     *     WORK,1 + 6*NA + 2 * NA*NA,IW,3 + 5 * NA,IER)

!      CALL DSYGVD(IEV,'V','L',N,A,NA,B,NB,EW,AUX,3*NA,IER)

      RETURN
      END
#endif
!
! ******************************************************************
!
!     SUBROUTINE CHOLES
!     =================
!
! ******************************************************************
!
!  Choles calculates the Choleski decomposition B = R' * R of B
!  into an upper triangle matrix R for the symmetric positive
!  definite Matrix B. The elements of the main diagonal are
!  stored inverted.
!
!     Parameters:
!
!       N       (I) :  Dimension of problem 
!       B       (I) :  Matrix B (lower triangle)
!               (O) :  Matrix R (upper triangle), inverted main diagonal
!       NB      (I) :  Dimension of B 
!       ICHO    (I) :  ICHO - 1 is the dimension of the submatrix that
!                      is available as Choleski decomposition ( < 1 = 1)
!               (O) :  Row number where decomposition failed (0 if success)
!
! ******************************************************************
!
      SUBROUTINE CHOLES (N,B,NB,ICHO)
        IMPLICIT REAL*8 (A-H,O-Z)
        DIMENSION  B(NB,N)
!
        IF (ICHO .GT. N)  GOTO 200
        IF (ICHO .LT. 1)  ICHO = 1
        DO 80 I = ICHO,N
          I1 = I - 1
          DO 70 J = I,N
            S = B(J,I)
            DO 20 K = 1,I1
              S = S - B(K,I) * B(K,J)
   20       CONTINUE
            IF (I .NE .J) GOTO 40
            IF (S .LE. 0.0) GOTO 100
            S = 1.0 / SQRT(S)
            D = S
            GOTO 60
   40       S = S * D
   60       B(I,J) = S
   70     CONTINUE
   80   CONTINUE
        ICHO = 0
        GOTO 200
  100   ICHO = I
  200   RETURN
      END
!
! ******************************************************************
!
!     SUBROUTINE MATRAF
!     =================
!
! ******************************************************************
!
!  Matraf calculates out of the symmetric matrix A and the 
!  upper triangular matrix R the product INV(R') * A * INV(R), 
!  where the main diagonal of R is given inverted.
!
!     Parameters:
!
!       N       (I) :  Dimension of problem
!       A       (I) :  Matrix A (lower triangle)
!               (O) :  Transformed matrix (lower triangle)
!       NA      (I) :  Dimension of A
!       B       (I) :  Matrix R (upper triangle), inverted main diagonal
!       NB      (I) :  Dimension of B
!
! *********************************************************************
!
      SUBROUTINE MATRAF (N,A,NA,B,NB,H)
        IMPLICIT REAL*8 (A-H,O-Z)
        DIMENSION  A(NA,N),B(NB,N),H(N)
!
!  FILL MATRIX 
!
          DO 20 I = 1,N
            DO 10 J = I+1,N
               A(I,J) = A(J,I)
   10       CONTINUE
   20     CONTINUE
!
!  CALCULATION OF A = INV(R') * A
!
          DO 60 I = 1,N
            I1 = I-1
            D = B(I,I)
            DO 50 J = 1,N
              S = A(I,J)
              DO 30 K = 1,I1
                S = S - B(K,I) * A(K,J)
   30         CONTINUE
              A(I,J) = S * D
   50       CONTINUE  
   60     CONTINUE
!
!  CALCULATION OF A = A * INV(R) (USE BUFFER FOR STRIDE OPTIMIZATION)
!
          DO 160 I = 1,N
            I1 = I-1
            D = B(I,I)
            DO 110 J = I,N
              H(J) = A(J,I)
  110       CONTINUE              
            DO 130 K = 1,I1
              S = B(K,I)
              DO 120 J = I,N
                H(J) = H(J) - S * A(J,K)
  120         CONTINUE
  130       CONTINUE
            DO 140 J = I,N
              A(J,I) = H(J) * D
  140       CONTINUE              
  160     CONTINUE
          RETURN
        END
!
! ******************************************************************
!
!     SUBROUTINE TRIDIA
!     =================
!
! ******************************************************************
!
!  Tridiagonalization of a given symmetric matrix A using Householder
!
!     Parameters:
!
!       NM      (I) :  Dimension of A 
!       N       (I) :  Dimension of problem
!       D       (O) :  Diagonal of tridiagonal matrix
!       E       (O) :  Subdiagonal of tridiagonal matrix (E(1) = 0.0)
!       A       (I) :  Matrix A (lower triangle)
!               (O) :  Transformation Matrix
!       IEV     (I) :  0: No eigenvectors
!
! ******************************************************************
!
      SUBROUTINE TRIDIA (NM,N,D,E,A,IEV)
        IMPLICIT REAL*8 (A-H,O-Z)
        DIMENSION  A(NM,N),D(N),E(N)
!
        DO 100 I = 1,N
          D(I) = A(N,I)
  100   CONTINUE
        IF (N .EQ. 1) GOTO 510
!
!  FOR I = N STEP -1 UNTIL 2 DO
!
        DO 300 II = 2,N
          I = N + 2 - II
          L = I - 1
          H = 0.0
          SCALE = 0.0
          IF (L .LT. 2) GOTO 130
!
!  SCALE ROW
!
          DO 120 K = 1,L
            SCALE = SCALE + ABS(D(K))
  120     CONTINUE
!
          IF (SCALE .NE. 0.0) GOTO 140
  130     E(I) = D(L)
          DO 135 J = 1,L
            D(J) = A(L,J)
            A(I,J) = 0.0
            A(J,I) = 0.0
  135     CONTINUE
          GOTO 290
!
  140     DO 150 K = 1,L
            D(K) = D(K) / SCALE
            H = H + D(K) * D(K)
  150     CONTINUE 
          F = D(L)
          G = -SIGN(SQRT(H),F)
          E(I) = SCALE * G
          H = H - F * G
          D(L) = F - G
!
!  FORM A * U
!
          DO 170 J = 1,L
            E(J) = 0.0
  170     CONTINUE
          DO 240 J = 1,L
            F = D(J)
            A(J,I) = F
            G = E(J) + A(J,J) * F
            JP1 = J + 1
            DO 200 K = JP1,L
              G = G + A(K,J) * D(K)
              E(K) = E(K) + A(K,J) * F
  200       CONTINUE 
            E(J) = G
  240     CONTINUE
!
!  FORM P
!
          F = 0.0
          DO 245 J = 1,L
            E(J) = E(J) / H
            F = F + E(J) * D(J)
  245     CONTINUE
          HH = F / (H + H)
!
!  FORM Q
!
          DO 250 J = 1,L
            E(J) = E(J) - HH * D(J)
  250     CONTINUE
!
!  FORM REDUCED A
!
          DO 280 J = 1,L
            F = D(J)
            G = E(J)
            DO 260 K = J,L
              A(K,J) = A(K,J) - F * E(K) - G * D(K)
  260       CONTINUE
            D(J) = A(L,J)
            A(I,J) = 0.0 
  280     CONTINUE 
!
!  DONE WITH THIS TRANSFORMATION
! 
  290     D(I) = H
  300   CONTINUE 
!
!  ACCUMULATION OF TRANSFORMATION MATRICES
!
        IF (IEV .EQ. 0) GOTO 600
        DO 500 I = 2,N
          L = I - 1
          A(N,L) = A(L,L)
          A(L,L) = 1.0
          H = D(I)
          IF (H .EQ. 0.0) GOTO 380
          DO 330 K = 1,L
            D(K) = A(K,I) / H
  330     CONTINUE
          DO 360 J = 1,L
            G = 0.0
            DO 340 K = 1,L
              G = G + A(K,I) * A(K,J)
  340       CONTINUE
            DO 350 K = 1,L
              A(K,J) = A(K,J) - G * D(K)
  350       CONTINUE
  360     CONTINUE
!        
  380     DO 400 K = 1,L
            A(K,I) = 0.0
  400     CONTINUE
  500   CONTINUE
  510   DO 520 I = 1,N
         D(I) = A(N,I)
         A(N,I) = 0.0
  520   CONTINUE
        GOTO 700
!
!  DEAL WITH EIGENVALUES ONLY
!
  600   DO 610 I = 1,N
          D(I) = A(I,I)
  610   CONTINUE
!
  700   A(N,N) = 1.0
        E(1) = 0.0
        RETURN
      END
!
! ******************************************************************
!
!     SUBROUTINE IQLDIA
!     =================
!
! ******************************************************************
!
!  Iqldia calculates eigenvalues and eigenvectors of a tridiagonal
!  matrix using the QL algorithm with implicit shifting.
!
!     Parameters:
!
!       NM      (I) :  Dimension of Z
!       N       (I) :  Dimension of the problem 
!       D       (I) :  Diagonal of tridiagonal matrix
!               (O) :  Eigenvalues
!       E       (I) :  Subdiagonal of tridiagonal matrix
!       Z       (I) :  Transformation matrix 
!               (O) :  Eigenvectors according to Z
!       IEV     (I) :  0: No eigenvectors
!       IER     (O) :  Error indication
!                      0: no error
!                      K: Convergence failure for the eigenvalue 
!                         number k, k-1 eigenvalues are correct
!
! **********************************************************************
!
      SUBROUTINE IQLDIA (NM,N,D,E,Z,IEV,IER)
        IMPLICIT REAL*8 (A-H,O-Z)
        DIMENSION  D(N),E(N),Z(NM,N)
!
        IER = 0
        IF (N .EQ. 1) RETURN
!
!  GET MACHINE EPSILON AND BIG
!
        EPS = 1.0E-2
   10   IF ((1.0 + EPS) .EQ. 1.0) GOTO 20
        EPS = 0.5 * EPS
        GOTO 10
   20   EPS = 2.0 * EPS
        EPSS = SQRT(EPS)
        EPS4 = EPS * 1.0E-4
        BIG = 1.0/EPS4
!
        ANORM = 0.0  
        R = 0.0
        DO 30 I = 2, N
          S = E(I)
          E(I-1) = S
          S = ABS(S)
          P = ABS(D(I-1)) + R + S
          IF (P .GT. ANORM) ANORM = P
          R = S
   30   CONTINUE
        P = ABS(D(N)) + R
        IF (P .GT. ANORM) ANORM = P
        E(N) = 0.0
        DO 250 L = 1, N
          J = 0
!
!  LOOK FOR SMALL SUBDIAGONAL ELEMENT 
! 
   50     DO 60 M = L, N-1
            DD = ABS(D(M)) + ABS(D(M+1))
            IF (ABS(E(M)) .LE. (EPS * DD)) GOTO 70
            IF (ABS(E(M)) .LE. (EPS4 * ANORM)) GOTO 70
   60     CONTINUE
          M = N
   70     P = D(L)
          MM1 = M - 1
          IF (M .EQ. L) GOTO 250
          IF (J .EQ. 30) GOTO 900
          J = J + 1
!
!  FORM SHIFT. THIS IS A SLIGHTLY ADVANCED FORM OF SHIFTING MAKING
!  THE ROUTINE ABOUT 20 PERCENT FASTER THAN THE USUAL STUFF.
!
          G = (D(L+1) - P) / (2.0 * E(L))
          R = SQRT (G * G + 1.0)
          S = P - E(L) / (G + SIGN (R, G))
          IF (M .EQ. L+1) GOTO 120
          T = S
          R = MAX(ABS(S),(ANORM / N))
          DO 100 I = 1, 6
            PSI = D(M) - T
            PSJ = -1.0
            DO 90 KK = L, MM1
              K = L + MM1 - KK
              IF (ABS(PSI) .GE. (EPS * ABS(E(K)))) GOTO 80
              PSI = BIG
              PSJ = BIG * BIG
              GOTO 90
   80         P = E(K) / PSI
              PSI = D(K) - T - P * E(K)
              PSJ = P * P * PSJ - 1.0
   90       CONTINUE
            IF (ABS(PSJ) .LE. EPS4) GOTO 120
            P = PSI / PSJ
            C = P
            IF (ABS(P) .GT. (0.5 * R)) C = SIGN(R,P)
            T = T - C
            IF (ABS(P) .LE. (EPSS * R)) GOTO 110
  100     CONTINUE
          GOTO 120
  110     S = T
  120     G = D(M) - S
          S = 1.0
          C = 1.0
          P = 0.0
          MML = M - L
!
!  FOR I = M - 1 STEP -1 UNTIL L DO 
!
          DO 200 II = 1, MML
            I = M - II
            F = S * E(I)
            B = C * E(I)
!
!  SAFE CALCULATION OF SQRT(G * G + F * F) AND SIMILAR STUFF
!
            IF (ABS(F) .LT. ABS(G)) GOTO 150
            C = G / F
            R = SQRT(1.0 + C * C)
            E(I+1) = F * R
            S = 1.0 / R
            C = C * S
            GOTO 160
  150       S = F / G
            R = SQRT (1.0 + S * S)
            E(I+1) = G * R
            C = 1.0 / R
            S = S * C
  160       G = D(I+1) - P
            R = (D(I) - G) * S + 2.0 * C * B
            P = S * R
            D(I+1) = G + P
            G = C * R - B
            IF (IEV .EQ. 0) GOTO 200
!
!  FORM VECTOR
!
            DO 180 K = 1,N
              F = Z(K,I+1)
              B = Z(K,I)
              Z(K,I+1) = S * B + C * F
              Z(K,I)   = C * B - S * F
  180       CONTINUE
  200     CONTINUE
          D(L) = D(L) - P
          E(L) = G
          E(M) = 0.0
          GOTO 50
  250   CONTINUE
        RETURN
  900   IER = L
        RETURN
      END
!
! ******************************************************************
!
!  This is another version of Iqldia using a less sophisticated 
!  shifting algorithm. It is much simpler but 20 percent slower.
!
! ******************************************************************
!
!     SUBROUTINE IQLDIA (NM,N,D,E,Z,IEV,IER)
!       IMPLICIT REAL*8 (A-H,O-Z)
!       DIMENSION  D(N),E(N),Z(NM,N)
!
!       IER = 0
!       IF (N .EQ. 1) RETURN
!       DO 10 I = 2, N
!         E(I-1) = E(I)
!  10   CONTINUE
!       E(N) = 0.0
!       DO 250 L = 1, N
!         ITER = 0
!
!  LOOK FOR SMALL SUBDIAGONAL ELEMENT 
! 
! 100     DO 110 M = L, N-1
!           DD = ABS(D(M)) + ABS(D(M+1))
!           IF ((ABS(E(M)) + DD) .EQ. DD) GOTO 120
! 110     CONTINUE
!         M = N
! 120     IF (M .EQ. L) GOTO 250
!         IF (ITER .EQ. 30) GOTO 900
!         ITER = ITER + 1
!
!  FORM SHIFT 
!
!         G = (D(L+1) - D(L)) / (2.0 * E(L))
!         R = SQRT (G * G + 1.0)
!         G = D(M) - D(L) + E(L) / (G + SIGN(R,G))
!         S = 1.0
!         C = 1.0
!         P = 0.0
!
!  FOR I = M - 1 STEP -1 UNTIL L DO 
!
!         DO 200 II = 1, M-L
!           I = M - II
!           F = S * E(I)
!           B = C * E(I)
!
!  SAFE CALCULATION OF SQRT(G * G + F * F) AND SIMILAR STUFF
!
!           IF (ABS(F) .LT. ABS(G)) GOTO 150
!           C = G / F
!           R = SQRT(1.0 + C * C)
!           E(I+1) = F * R
!           S = 1.0 / R
!           C = C * S
!           GOTO 160
! 150       S = F / G
!           R = SQRT (1.0 + S * S)
!           E(I+1) = G * R
!           C = 1.0 / R
!           S = S * C
! 160       G = D(I+1) - P
!           R = (D(I) - G) * S + 2.0 * C * B
!           P = S * R
!           D(I+1) = G + P
!           G = C * R - B
!           IF (IEV .EQ. 0) GOTO 200
!
!  FORM VECTOR
!
!           DO 180 K = 1, N
!             F = Z(K,I+1)
!             Z(K,I+1) = S * Z(K,I) + C * F
!             Z(K,I) =   C * Z(K,I) - S * F
! 180       CONTINUE
! 200     CONTINUE
!         D(L) = D(L) - P
!         E(L) = G
!         E(M) = 0.0
!         GOTO 100
! 250   CONTINUE
!       RETURN
! 900   IER = L
!       RETURN
!     END
!
! ******************************************************************
!
!     SUBROUTINE BACKTR
!     =================
!
! ******************************************************************
!
!  Backtr solves the system R * X = Y (R upper triangular matrix),
!  where the main diagonal of R is given inverted.
!
!     Parameters:
!       N       (I) :  Dimension of problem 
!       M       (I) :  Number of columns in X and Y
!       R       (I) :  Matrix R (upper triangle)
!       NR      (I) :  Dimension of R
!       X       (O) :  Matrix X (solution of system)
!       NX      (I) :  Dimension of X 
!       Y       (I) :  Matrix Y (right side)
!       NY      (I) :  Dimension of Y
!       H       (I) :  Auxiliary vector
!
! **********************************************************************
!
      SUBROUTINE BACKTR (N,M,R,NR,X,NX,Y,NY,H)
        IMPLICIT REAL*8 (A-H,O-Z)
        DIMENSION  R(NR,N),X(NX,M),Y(NY,M),H(N)
!
!  CALCULATION OF X = INV(R) * Y 
!
        DO 40 II = 1,N
          I = N + 1 - II
          I1 = I + 1
          D = R(I,I)
          DO 10 J= I,N
            H(J)= R(I,J)
   10     CONTINUE
          DO 30 J = 1,M
            S = Y(I,J)
            DO 20 K = I1,N
              S = S - H(K) * X(K,J)
   20       CONTINUE
            X(I,J) = S * D
   30     CONTINUE
   40   CONTINUE
        RETURN
      END
!
! ******************************************************************
!
!     SUBROUTINE SORTVC
!     =================
!
! ******************************************************************
!
!  Sortvc sorts D and (if required) E and the columns of Q.
!
!     Prameters:
!
!       NM      (I) :  Dimension of Q
!       N       (I) :  Dimension of problem (size of one vector in Q)
!       NQ      (I) :  Number of elements in D (or columns in Q)
!       D       (I) :  Vector to sort
!               (O) :  Sorted vector 
!       Q       (I) :  Matrix to sort (vectors in columns)
!               (O) :  Sorted matrix (vectors in columns)
!       M       (I) :  1: Descending order in D
!                     -1: Ascending order in D
!                      otherwise: no sorting
!       IEV     (I) :  0: No sorting of Q and E
!                      1: Sorting of Q, no sorting of E
!                      2: Sorting of Q and E
!       E       (I) :  Additional Vector to sort
!               (O) :  Sorted additional vector
!
! **********************************************************************
!
      SUBROUTINE SORTVC (NM,N,NQ,D,Q,M,IEV,E)
        IMPLICIT REAL*8 (A-H,O-Z)
        LOGICAL    LMIN,LMAX
        DIMENSION  D(NQ),E(NQ),Q(NM,NQ)
!
        IF (NQ .LT. 2) RETURN
        LMAX = (M .EQ.  1)
        LMIN = (M .EQ. -1)
        IF (.NOT. (LMAX .OR. LMIN)) RETURN
        DO 40 KK = 2,NQ
          K = KK - 1
          J = K
          H = D(K)
!
!  FIND EXTREMUM
!
          DO 10 I = KK,NQ
            S = D(I)
            IF (LMIN .AND. (S .GE. H)) GOTO 10
            IF (LMAX .AND. (S .LE. H)) GOTO 10
            J = I
            H = S
   10     CONTINUE
          IF (J .EQ. K) GOTO 40
!
!  SORT D
!
          D(J) = D(K)
          D(K) = H
          IF (IEV .EQ. 0) GOTO 40
!
!  SORT Q
!
          DO 20 I = 1,N
            H = Q(I,K)
            Q(I,K) = Q(I,J)
            Q(I,J) = H
   20     CONTINUE
          IF (IEV .LT. 2) GOTO 40
!
!  SORT E
!
          H    = E(K)
          E(K) = E(J)
          E(J) = H
   40   CONTINUE
        RETURN
      END

