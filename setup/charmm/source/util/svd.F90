SUBROUTINE SVD (NM, M, N, A, W, MATU, U, MATV, V, IERR, RV1)
  !-----------------------------------------------------------------------
  use chm_kinds
  use number
  implicit none
  !***BEGIN PROLOGUE  SVD
  !***SUBSIDIARY
  !***PURPOSE  Perform the singular value decomposition of a rectangular
  !            matrix.
  !***LIBRARY   SLATEC
  !***TYPE      SINGLE PRECISION (SVD-S)
  !***AUTHOR  (UNKNOWN)
  !***DESCRIPTION
  !
  !     This subroutine is a translation of the ALGOL procedure SVD,
  !     NUM. MATH. 14, 403-420(1970) by Golub and Reinsch.
  !     HANDBOOK FOR AUTO. COMP., VOL II-LINEAR ALGEBRA, 134-151(1971).
  !
  !     This subroutine determines the singular value decomposition
  !          T
  !     A=USV  of a REAL M by N rectangular matrix.  Householder
  !     bidiagonalization and a variant of the QR algorithm are used.
  !
  !     On Input
  !
  !        NM must be set to the row dimension of the two-dimensional
  !          array parameters, A, U and V, as declared in the calling
  !          program dimension statement.  NM is an INTEGER variable.
  !          Note that NM must be at least as large as the maximum
  !          of M and N.
  !
  !        M is the number of rows of A and U.
  !
  !        N is the number of columns of A and U and the order of V.
  !
  !        A contains the rectangular input matrix to be decomposed.  A is
  !          a two-dimensional REAL array, dimensioned A(NM,N).
  !
  !        MATU should be set to .TRUE. if the U matrix in the
  !          decomposition is desired, and to .FALSE. otherwise.
  !          MATU is a LOGICAL variable.
  !
  !        MATV should be set to .TRUE. if the V matrix in the
  !          decomposition is desired, and to .FALSE. otherwise.
  !          MATV is a LOGICAL variable.
  !
  !     On Output
  !
  !        A is unaltered (unless overwritten by U or V).
  !
  !        W contains the N (non-negative) singular values of A (the
  !          diagonal elements of S).  They are unordered.  If an
  !          error exit is made, the singular values should be correct
  !          for indices IERR+1, IERR+2, ..., N.  W is a one-dimensional
  !          REAL array, dimensioned W(N).
  !
  !        U contains the matrix U (orthogonal column vectors) of the
  !          decomposition if MATU has been set to .TRUE.  Otherwise,
  !          U is used as a temporary array.  U may coincide with A.
  !          If an error exit is made, the columns of U corresponding
  !          to indices of correct singular values should be correct.
  !          U is a two-dimensional REAL array, dimensioned U(NM,N).
  !
  !        V contains the matrix V (orthogonal) of the decomposition if
  !          MATV has been set to .TRUE.  Otherwise, V is not referenced.
  !          V may also coincide with A if U does not.  If an error
  !          exit is made, the columns of V corresponding to indices of
  !          correct singular values should be correct.  V is a two-
  !          dimensional REAL array, dimensioned V(NM,N).
  !
  !        IERR is an INTEGER flag set to
  !          Zero       for normal return,
  !          K          if the K-th singular value has not been
  !                     determined after 30 iterations.
  !
  !        RV1 is a one-dimensional REAL array used for temporary storage,
  !          dimensioned RV1(N).
  !
  !     CALLS PYTHAG(A,B) for sqrt(A**2 + B**2).
  !
  !     Questions and comments should be directed to B. S. Garbow,
  !     APPLIED MATHEMATICS DIVISION, ARGONNE NATIONAL LABORATORY
  !     ------------------------------------------------------------------
  !
  !***SEE ALSO  EISDOC
  !***ROUTINES CALLED  PYTHAG
  !***REVISION HISTORY  (YYMMDD)
  !   811101  DATE WRITTEN
  !   890531  Changed all specific intrinsics to generic.  (WRB)
  !   890831  Modified array declarations.  (WRB)
  !   891214  Prologue converted to Version 4.0 format.  (BAB)
  !   900402  Added TYPE section.  (WRB)
  !***END PROLOGUE  SVD
  !
  INTEGER I,J,K,L,M,N,II,I1,KK,K1,LL,L1,MN,NM,ITS,IERR
  real(chm_real) A(NM,*),W(*),U(NM,*),V(NM,*),RV1(*)
  real(chm_real) C,F,G,H,S,X,Y,Z,SCALE,S1
  real(chm_real) PYTHAG1
  LOGICAL MATU,MATV
  !
  !***FIRST EXECUTABLE STATEMENT  SVD
  IERR = 0
  !
  U(1:m,1:n) = A(1:m,1:n)
  !     .......... HOUSEHOLDER REDUCTION TO BIDIAGONAL FORM ..........
  G = 0.0E0
  SCALE = 0.0E0
  S1 = 0.0E0
  !
  loop300:DO I = 1, N
     L = I + 1
     RV1(I) = SCALE * G
     G = 0.0E0
     S = 0.0E0
     SCALE = 0.0E0
     IF (I  >  M) GO TO 210
     !
     DO K = I, M
        SCALE = SCALE + ABS(U(K,I))
     enddo
     !
     IF (SCALE  ==  0.0E0) GO TO 210
     !
     DO K = I, M
        U(K,I) = U(K,I) / SCALE
        S = S + U(K,I)**2
     enddo
     !
     F = U(I,I)
     G = -SIGN(SQRT(S),F)
     H = F * G - S
     U(I,I) = F - G
     IF (I  ==  N) GO TO 190
     !
     DO J = L, N
        S = 0.0E0
        !
        DO K = I, M
           S = S + U(K,I) * U(K,J)
        enddo
        !
        F = S / H
        !
        DO K = I, M
           U(K,J) = U(K,J) + F * U(K,I)
        enddo
     enddo
     !
190  DO K = I, M
        U(K,I) = SCALE * U(K,I)
     enddo
     !
210  W(I) = SCALE * G
     G = 0.0E0
     S = 0.0E0
     SCALE = 0.0E0
     IF (I  >  M .OR. I  ==  N) GO TO 290
     !
     DO K = L, N
        SCALE = SCALE + ABS(U(I,K))
     enddo
     !
     IF (SCALE  ==  0.0E0) GO TO 290
     !
     DO K = L, N
        U(I,K) = U(I,K) / SCALE
        S = S + U(I,K)**2
     enddo
     !
     F = U(I,L)
     G = -SIGN(SQRT(S),F)
     H = F * G - S
     U(I,L) = F - G
     !
     DO K = L, N
        RV1(K) = U(I,K) / H
     enddo
     !
     IF (I  /=  M)then ! GO TO 270
        !
        DO J = L, M
           S = 0.0E0
           !
           DO K = L, N
              S = S + U(J,K) * U(I,K)
           enddo
           !
           DO K = L, N
              U(J,K) = U(J,K) + S * RV1(K)
           enddo
        enddo
        !
     endif

     U(I,l:n) = SCALE * U(I,l:n)
     !
290  S1 = MAX(S1,ABS(W(I))+ABS(RV1(I)))
  enddo loop300
  !     .......... ACCUMULATION OF RIGHT-HAND TRANSFORMATIONS ..........
  IF ( MATV)then  !  GO TO 410
     !     .......... FOR I=N STEP -1 UNTIL 1 DO -- ..........
     loop400: DO II = 1, N
        I = N + 1 - II
        IF (I  /=  N)then ! GO TO 390
           IF (G  /=  0.0E0)then  ! GO TO 360
              !
              DO J = L, N
                 !     .......... DOUBLE DIVISION AVOIDS POSSIBLE UNDERFLOW ..........
                 V(J,I) = (U(I,J) / U(I,L)) / G
              enddo
              !
              loop350:DO J = L, N
                 S = 0.0E0
                 !
                 DO K = L, N
                    S = S + U(I,K) * V(K,J)
                 enddo
                 !
                 DO K = L, N
                    V(K,J) = V(K,J) + S * V(K,I)
                 enddo
              enddo loop350
              !
           endif
           DO  J = L, N
              V(I,J) = 0.0E0
              V(J,I) = 0.0E0
           enddo
        endif
        !
        V(I,I) = 1.0E0
        G = RV1(I)
        L = I
     enddo loop400
  endif
  !     .......... ACCUMULATION OF LEFT-HAND TRANSFORMATIONS ..........
  IF ( MATU)then    ! GO TO 510
     !     ..........FOR I=MIN(M,N) STEP -1 UNTIL 1 DO -- ..........
     MN = N
     IF (M  <  N) MN = M
     !
     loop500:DO II = 1, MN
        I = MN + 1 - II
        L = I + 1
        G = W(I)
        IF (I  /=  N)then ! GO TO 430
           !
           DO J = L, N
              U(I,J) = 0.0E0
           enddo
        endif
        !
        IF (G  /=  0.0E0)then   ! GO TO 475
           IF (I  /=  MN)then    ! GO TO 460
              !
              DO J = L, N
                 S = 0.0E0
                 !
                 DO K = L, M
                    S = S + U(K,I) * U(K,J)
                 enddo
                 !     .......... DOUBLE DIVISION AVOIDS POSSIBLE UNDERFLOW ..........
                 F = (S / U(I,I)) / G
                 !
                 U(i:m,J) = U(i:m,J) + F * U(i:m,I)
              enddo
              !
           endif
           U(i:m,I) = U(i:m,I) / G
           U(I,I) = U(I,I) + one
           cycle loop500
        endif
        U(i:m,I) = zero
        U(I,I) = U(I,I) + one
     enddo loop500
  endif
  !     .......... DIAGONALIZATION OF THE BIDIAGONAL FORM ..........
  !     .......... FOR K=N STEP -1 UNTIL 1 DO -- ..........
  loop700: DO KK = 1, N
     K1 = N - KK
     K = K1 + 1
     ITS = 0
     !     .......... TEST FOR SPLITTING.
     !                FOR L=K STEP -1 UNTIL 1 DO -- ..........
520  loop530:DO LL = 1, K
        L1 = K - LL
        L = L1 + 1
        IF (S1 + ABS(RV1(L))  ==  S1) GO TO 565
        !     .......... RV1(1) IS ALWAYS ZERO, SO THERE IS NO EXIT
        !                THROUGH THE BOTTOM OF THE LOOP ..........
        IF (S1 + ABS(W(L1))  ==  S1) exit loop530
     enddo loop530
     !     .......... CANCELLATION OF RV1(L) IF L GREATER THAN 1 ..........
     C = 0.0E0
     S = 1.0E0
     !
     loop560: DO I = L, K
        F = S * RV1(I)
        RV1(I) = C * RV1(I)
        IF (S1 + ABS(F)  ==  S1) exit loop560
        G = W(I)
        H = PYTHAG1(F,G)
        W(I) = H
        C = G / H
        S = -F / H
        IF (MATU)then  ! GO TO 560
           !
           DO J = 1, M
              Y = U(J,L1)
              Z = U(J,I)
              U(J,L1) = Y * C + Z * S
              U(J,I) = -Y * S + Z * C
           enddo
        endif
     enddo loop560
     !     .......... TEST FOR CONVERGENCE ..........
565  Z = W(K)
     if650: IF (L  /=  K)then   ! GO TO 650
        !     .......... SHIFT FROM BOTTOM 2 BY 2 MINOR ..........
        IF (ITS  ==  30) GO TO 1000
        ITS = ITS + 1
        X = W(L)
        Y = W(K1)
        G = RV1(K1)
        H = RV1(K)
        F = 0.5E0 * (((G + Z) / H) * ((G - Z) / Y) + Y / H - H / Y)
        G = PYTHAG1(F,ONE)
        F = X - (Z / X) * Z + (H / X) * (Y / (F + SIGN(G,F)) - H)
        !     .......... NEXT QR TRANSFORMATION ..........
        C = 1.0E0
        S = 1.0E0
        !
        loop600: DO I1 = L, K1
           I = I1 + 1
           G = RV1(I)
           Y = W(I)
           H = S * G
           G = C * G
           Z = PYTHAG1(F,H)
           RV1(I1) = Z
           C = F / Z
           S = H / Z
           F = X * C + G * S
           G = -X * S + G * C
           H = Y * S
           Y = Y * C
           IF (MATV)then   ! GO TO 575
              !
              DO J = 1, N
                 X = V(J,I1)
                 Z = V(J,I)
                 V(J,I1) = X * C + Z * S
                 V(J,I) = -X * S + Z * C
              enddo
           endif
           !
           Z = PYTHAG1(F,H)
           W(I1) = Z
           !     .......... ROTATION CAN BE ARBITRARY IF Z IS ZERO ..........
           IF (Z  /=  zero) then   ! GO TO 580
              C = F / Z
              S = H / Z
           endif
           F = C * G + S * Y
           X = -S * G + C * Y
           IF (.NOT. MATU) cycle loop600
           !
           DO J = 1, M
              Y = U(J,I1)
              Z = U(J,I)
              U(J,I1) = Y * C + Z * S
              U(J,I) = -Y * S + Z * C
           enddo
           !
        enddo loop600
        !
        RV1(L) = 0.0E0
        RV1(K) = F
        W(K) = X
        GO TO 520
     endif if650
     !     .......... CONVERGENCE ..........
     IF (Z  < zero)then   ! GO TO 700
        !     .......... W(K) IS MADE NON-NEGATIVE ..........
        W(K) = -Z
        IF (MATV)then   ! GO TO 700
           !
           V(1:n,K) = -V(1:n,K)
           !
        endif
     endif
  enddo loop700
  !
  GO TO 1001
  !     .......... SET ERROR -- NO CONVERGENCE TO A
  !                SINGULAR VALUE AFTER 30 ITERATIONS ..........
1000 IERR = K
1001 RETURN
END SUBROUTINE SVD

FUNCTION PYTHAG1 (A, B) result(pythagg)
  use chm_kinds
  use number
  implicit none
  !***BEGIN PROLOGUE  PYTHAG
  !***SUBSIDIARY
  !***PURPOSE  Compute the complex square root of a complex number without
  !            destructive overflow or underflow.
  !***LIBRARY   SLATEC
  !***TYPE      SINGLE PRECISION (PYTHAG-S)
  !***AUTHOR  (UNKNOWN)
  !***DESCRIPTION
  !
  !     Finds sqrt(A**2+B**2) without overflow or destructive underflow
  !
  !***SEE ALSO  EISDOC
  !***ROUTINES CALLED  (NONE)
  !***REVISION HISTORY  (YYMMDD)
  !   811101  DATE WRITTEN
  !   890531  Changed all specific intrinsics to generic.  (WRB)
  !   891214  Prologue converted to Version 4.0 format.  (BAB)
  !   900402  Added TYPE section.  (WRB)
  !***END PROLOGUE  PYTHAG
  real(chm_real) A,B,pythagg
  !
  real(chm_real) P,Q,R,S,T
  !***FIRST EXECUTABLE STATEMENT  PYTHAG
  P = MAX(ABS(A),ABS(B))
  Q = MIN(ABS(A),ABS(B))
  IF (Q  ==  0.0E0) then

  endif
  do while(.true.)
     R = (Q/P)**2
     T = 4.0E0 + R
     IF (T  ==  4.0E0) exit
     S = R/T
     P = P + 2.0E0*P*S
     Q = Q*S
  enddo
  PYTHAGg = P
  RETURN
END FUNCTION PYTHAG1

SUBROUTINE svdbsb(u,w,v,m,n,mp,np,b,x)
  use chm_kinds
  use number
  implicit none
  INTEGER m,mp,n,np,NMAX
  real(chm_real) b(mp),u(mp,np),v(mp,np),w(np),x(np)
  PARAMETER (NMAX=500)
  INTEGER i,j,jj
  real(chm_real) s,tmp(NMAX)
  do i=1,n
     s=zero
     if(w(i) /= zero)then
        do j=1,m
           s=s+u(j,i)*b(j)
        enddo
        s=s/w(i)
     endif
     tmp(i)=s
  enddo

  do j=1,n
     s=zero
     do jj=1,n
        s=s+v(j,jj)*tmp(jj)
     enddo
     x(j)=s
  enddo
  return
END SUBROUTINE svdbsb

