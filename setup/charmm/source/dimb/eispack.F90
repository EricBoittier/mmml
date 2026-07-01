FUNCTION EPSLN (X) result(epsln_1)
  !
  use chm_kinds
  implicit none
  real(chm_real) X, epsln_1
  !
  !     ESTIMATE UNIT ROUNDOFF IN QUANTITIES OF SIZE X.
  !
  real(chm_real) A,B,C,EPS
  !
  !     THIS PROGRAM SHOULD FUNCTION PROPERLY ON ALL SYSTEMS
  !     SATISFYING THE FOLLOWING TWO ASSUMPTIONS,
  !        1.  THE BASE USED IN REPRESENTING FLOATING POINT
  !            NUMBERS IS NOT A POWER OF THREE.
  !        2.  THE QUANTITY  A  IN STATEMENT 10 IS REPRESENTED TO
  !            THE ACCURACY USED IN FLOATING POINT VARIABLES
  !            THAT ARE STORED IN MEMORY.
  !     THE STATEMENT NUMBER 10 AND THE GO TO 10 ARE INTENDED TO
  !     FORCE OPTIMIZING COMPILERS TO GENERATE CODE SATISFYING
  !     ASSUMPTION 2.
  !     UNDER THESE ASSUMPTIONS, IT SHOULD BE TRUE THAT,
  !            A  IS NOT EXACTLY EQUAL TO FOUR-THIRDS,
  !            B  HAS A ZERO FOR ITS LAST BIT OR DIGIT,
  !            C  IS NOT EXACTLY EQUAL TO ONE,
  !            EPS  MEASURES THE SEPARATION OF 1.0 FROM
  !                 THE NEXT LARGER FLOATING POINT NUMBER.
  !     THE DEVELOPERS OF EISPACK WOULD APPRECIATE BEING INFORMED
  !     ABOUT ANY SYSTEMS WHERE THESE ASSUMPTIONS DO NOT HOLD.
  !
  !     THIS VERSION DATED 4/6/83.
  !
  A = 4.0D0/3.0D0
  EPS = 0.0D0
  do while(EPS  ==  0.0D0)
     B = A - 1.0D0
     C = B + B + B
     EPS = DABS(C-1.0D0)
  enddo
  EPSLN_1 = EPS*DABS(X)
  RETURN
END FUNCTION EPSLN

FUNCTION PYTHAG3(A,B) result(pythag_1)
  !
  use chm_kinds
  use number,only:zero
  implicit none
  real(chm_real) A,B,pythag_1
  !
  !     FINDS DSQRT(A**2+B**2) WITHOUT OVERFLOW OR DESTRUCTIVE UNDERFLOW
  !
  real(chm_real) P,R,S,T,U
  P = DMAX1(DABS(A),DABS(B))
  IF (P  /=  zero) then
     R = (DMIN1(DABS(A),DABS(B))/P)**2
     do while(.true.)
        T = 4.0D0 + R
        IF (T  ==  4.0D0) exit
        S = R/T
        U = 1.0D0 + 2.0D0*S
        P = U*P
        R = (S/U)**2 * R
     enddo
  endif
  PYTHAG_1 = P
  RETURN
END FUNCTION PYTHAG3

SUBROUTINE RATQR(N,EPS1,D,E,E2,M,W,IND,BD,FIND_SMALL,IDEF,IERR)
  !-----------------------------------------------------------------------
  use chm_kinds
  use number
  implicit none
  INTEGER I,J,K,M,N,II,JJ,K1,IDEF,IERR,JDEF
  real(chm_real) D(N),E(N),E2(N),W(N),BD(N)
  real(chm_real) F,P,Q,R,S,EP,QP,ERR,TOT,EPS1,DELTA,EPSLN
  INTEGER IND(N)
  LOGICAL FIND_SMALL
  !
  !     THIS SUBROUTINE IS A TRANSLATION OF THE ALGOL PROCEDURE RATQR,
  !     NUM. MATH. 11, 264-272(1968) BY REINSCH AND BAUER.
  !     HANDBOOK FOR AUTO. COMP., VOL.II-LINEAR ALGEBRA, 257-265(1971).
  !
  !     THIS SUBROUTINE FINDS THE ALGEBRAICALLY SMALLEST OR LARGEST
  !     EIGENVALUES OF A SYMMETRIC TRIDIAGONAL MATRIX BY THE
  !     RATIONAL QR METHOD WITH NEWTON CORRECTIONS.
  !
  !     ON INPUT
  !
  !        N IS THE ORDER OF THE MATRIX.
  !
  !        EPS1 IS A THEORETICAL ABSOLUTE ERROR TOLERANCE FOR THE
  !          COMPUTED EIGENVALUES.  IF THE INPUT EPS1 IS NON-POSITIVE,
  !          OR INDEED SMALLER THAN ITS DEFAULT VALUE, IT IS RESET
  !          AT EACH ITERATION TO THE RESPECTIVE DEFAULT VALUE,
  !          NAMELY, THE PRODUCT OF THE RELATIVE MACHINE PRECISION
  !          AND THE MAGNITUDE OF THE CURRENT EIGENVALUE ITERATE.
  !          THE THEORETICAL ABSOLUTE ERROR IN THE K-TH EIGENVALUE
  !          IS USUALLY NOT GREATER THAN K TIMES EPS1.
  !
  !        D CONTAINS THE DIAGONAL ELEMENTS OF THE INPUT MATRIX.
  !
  !        E CONTAINS THE SUBDIAGONAL ELEMENTS OF THE INPUT MATRIX
  !          IN ITS LAST N-1 POSITIONS.  E(1) IS ARBITRARY.
  !
  !        E2 CONTAINS THE SQUARES OF THE CORRESPONDING ELEMENTS OF E.
  !          E2(1) IS ARBITRARY.
  !
  !        M IS THE NUMBER OF EIGENVALUES TO BE FOUND.
  !
  !        IDEF SHOULD BE SET TO 1 IF THE INPUT MATRIX IS KNOWN TO BE
  !          POSITIVE DEFINITE, TO -1 IF THE INPUT MATRIX IS KNOWN TO
  !          BE NEGATIVE DEFINITE, AND TO 0 OTHERWISE.
  !
  !        FIND_SMALL SHOULD BE SET TO .TRUE. IF THE SMALLEST EIGENVALUES
  !          ARE TO BE FOUND, AND TO .FALSE. IF THE LARGEST EIGENVALUES
  !          ARE TO BE FOUND.
  !
  !     ON OUTPUT
  !
  !        EPS1 IS UNALTERED UNLESS IT HAS BEEN RESET TO ITS
  !          (LAST) DEFAULT VALUE.
  !
  !        D AND E ARE UNALTERED (UNLESS W OVERWRITES D).
  !
  !        ELEMENTS OF E2, CORRESPONDING TO ELEMENTS OF E REGARDED
  !          AS NEGLIGIBLE, HAVE BEEN REPLACED BY ZERO CAUSING THE
  !          MATRIX TO SPLIT INTO A DIRECT SUM OF SUBMATRICES.
  !          E2(1) IS SET TO 0.0D0 IF THE SMALLEST EIGENVALUES HAVE BEEN
  !          FOUND, AND TO 2.0D0 IF THE LARGEST EIGENVALUES HAVE BEEN
  !          FOUND.  E2 IS OTHERWISE UNALTERED (UNLESS OVERWRITTEN BY BD).
  !
  !        W CONTAINS THE M ALGEBRAICALLY SMALLEST EIGENVALUES IN
  !          ASCENDING ORDER, OR THE M LARGEST EIGENVALUES IN
  !          DESCENDING ORDER.  IF AN ERROR EXIT IS MADE BECAUSE OF
  !          AN INCORRECT SPECIFICATION OF IDEF, NO EIGENVALUES
  !          ARE FOUND.  IF THE NEWTON ITERATES FOR A PARTICULAR
  !          EIGENVALUE ARE NOT MONOTONE, THE BEST ESTIMATE OBTAINED
  !          IS RETURNED AND IERR IS SET.  W MAY COINCIDE WITH D.
  !
  !        IND CONTAINS IN ITS FIRST M POSITIONS THE SUBMATRIX INDICES
  !          ASSOCIATED WITH THE CORRESPONDING EIGENVALUES IN W --
  !          1 FOR EIGENVALUES BELONGING TO THE FIRST SUBMATRIX FROM
  !          THE TOP, 2 FOR THOSE BELONGING TO THE SECOND SUBMATRIX, ETC..
  !
  !        BD CONTAINS REFINED BOUNDS FOR THE THEORETICAL ERRORS OF THE
  !          CORRESPONDING EIGENVALUES IN W.  THESE BOUNDS ARE USUALLY
  !          WITHIN THE TOLERANCE SPECIFIED BY EPS1.  BD MAY COINCIDE
  !          WITH E2.
  !
  !        IERR IS SET TO
  !          ZERO       FOR NORMAL RETURN,
  !          6*N+1      IF  IDEF  IS SET TO 1 AND  FIND_SMALL  TO .TRUE.
  !                     WHEN THE MATRIX IS NOT POSITIVE DEFINITE, OR
  !                     IF  IDEF  IS SET TO -1 AND  FIND_SMALL  TO .FALSE.
  !                     WHEN THE MATRIX IS NOT NEGATIVE DEFINITE,
  !          5*N+K      IF SUCCESSIVE ITERATES TO THE K-TH EIGENVALUE
  !                     ARE NOT MONOTONE INCREASING, WHERE K REFERS
  !                     TO THE LAST SUCH OCCURRENCE.
  !
  !     NOTE THAT SUBROUTINE TRIDIB IS GENERALLY FASTER AND MORE
  !     ACCURATE THAN RATQR IF THE EIGENVALUES ARE CLUSTERED.
  !
  !     QUESTIONS AND COMMENTS SHOULD BE DIRECTED TO BURTON S. GARBOW,
  !     MATHEMATICS AND COMPUTER SCIENCE DIV, ARGONNE NATIONAL LABORATORY
  !
  !     THIS VERSION DATED AUGUST 1983.
  !
  !     ------------------------------------------------------------------
  !
  IERR = 0
  JDEF = IDEF
  !     .......... COPY D ARRAY INTO W ..........
  W(1:n) = D(1:n)
  !
  IF (.not. FIND_SMALL) then
     J = 1
     GO TO 400
  endif
40 ERR = 0.0D0
  S = 0.0D0
  !     .......... LOOK FOR SMALL SUB-DIAGONAL ENTRIES AND DEFINE
  !                INITIAL SHIFT FROM LOWER GERSCHGORIN BOUND.
  !                COPY E2 ARRAY INTO BD ..........
  TOT = W(1)
  Q = 0.0D0
  J = 0
  !
  loop100: DO I = 1, N
     P = Q
     IF (I  /=  1) then !GO TO 60
        IF (P  >  EPSLN(DABS(D(I)) + DABS(D(I-1)))) GO TO 80
     endif
     E2(I) = 0.0D0
80   BD(I) = E2(I)
     !     .......... COUNT ALSO IF ELEMENT OF E2 HAS UNDERFLOWED ..........
     IF (E2(I)  ==  0.0D0) J = J + 1
     IND(I) = J
     Q = 0.0D0
     IF (I  /=  N) Q = DABS(E(I+1))
     TOT = DMIN1(W(I)-P-Q,TOT)
  enddo loop100
  !
  IF (JDEF  /=  1 .or. TOT  >  zero) then  !GO TO 140
     W(1:n) = W(1:n) - TOT
  else
     TOT = 0.0D0
  endif
  !
  loop360: DO K = 1, M
     !     .......... NEXT QR TRANSFORMATION ..........
180  TOT = TOT + S
     DELTA = W(N) - S
     I = N
     F = DABS(EPSLN(TOT))
     IF (EPS1  <  F) EPS1 = F
     IF (DELTA  <=  EPS1) then
        IF (DELTA  <  (-EPS1)) then
           IERR = 6 * N + 1
           return
        endif
        GO TO 300
        !     .......... REPLACE SMALL SUB-DIAGONAL SQUARES BY ZERO
        !                TO REDUCE THE INCIDENCE OF UNDERFLOWS ..........
     endif
     IF (K  /=  N) then !GO TO 210
        K1 = K + 1
        DO J = K1, N
           IF (BD(J)  <=  (EPSLN(W(J)+W(J-1))) ** 2) BD(J) = zero
        enddo
     endif
     F = BD(N) / DELTA
     QP = DELTA + F
     P = 1.0D0
     IF (K  /=  N) then   !GO TO 260
        K1 = N - K
        !     .......... FOR I=N-1 STEP -1 UNTIL K DO -- ..........
        loop240: DO II = 1, K1
           I = N - II
           Q = W(I) - S - F
           R = Q / QP
           P = P * R + 1.0D0
           EP = F * R
           W(I+1) = QP + EP
           DELTA = Q - EP
           IF (DELTA  <=  EPS1)then  ! GO TO 220
              IF (DELTA  <  (-EPS1)) then
                 IERR = 6 * N + 1
                 return
              endif
              GO TO 300
           endif
           F = BD(I) / Q
           QP = DELTA + F
           BD(I+1) = QP * EP
        enddo loop240
        !
     endif
     W(K) = QP
     S = QP / P
     IF (TOT + S  >  TOT) GO TO 180
     !     .......... SET ERROR -- IRREGULAR END OF ITERATION.
     !                DEFLATE MINIMUM DIAGONAL ELEMENT ..........
     IERR = 5 * N + K
     S = 0.0D0
     DELTA = QP
     !
     DO J = K, N
        IF (W(J)  >  DELTA) cycle
        I = J
        DELTA = W(J)
     enddo
     !     .......... CONVERGENCE ..........
300  IF (I  <  N) BD(I+1) = BD(I) * F / QP
     II = IND(I)
     IF (I  /=  K)then   ! GO TO 340
        K1 = I - K
        !     .......... FOR J=I-1 STEP -1 UNTIL K DO -- ..........
        DO JJ = 1, K1
           J = I - JJ
           W(J+1) = W(J) - S
           BD(J+1) = BD(J)
           IND(J+1) = IND(J)
        enddo
        !
     endif
     W(K) = TOT
     ERR = ERR + DABS(DELTA)
     BD(K) = ERR
     IND(K) = II
  enddo loop360
  !
  IF (FIND_SMALL) return    !GO TO 1001
  F = BD(1)
  E2(1) = 2.0D0
  BD(1) = F
  J = 2
  !     .......... NEGATE ELEMENTS OF W FOR LARGEST VALUES ..........
400 continue
  W(1:n) = -W(1:n)
  !
  JDEF = -JDEF
  if(j == 1) goto 40        ! GO TO (40,1001), J

  RETURN
END SUBROUTINE RATQR

SUBROUTINE TINVIT(NM,N,D,E,E2,M,W,IND,Z, &
     IERR,RV1,RV2,RV3,RV4,RV6)
  !-----------------------------------------------------------------------
  use chm_kinds
  use number
  implicit none
  INTEGER I,J,M,N,P,Q,R,S,II,IP,JJ,NM,ITS,TAG,IERR,GROUP
  real(chm_real) D(N),E(N),E2(N),W(M),Z(NM,M), &
       RV1(N),RV2(N),RV3(N),RV4(N),RV6(N)
  real(chm_real) U,V,UK,XU,X0,X1,EPS2,EPS3,EPS4,NORM,ORDER,EPSLN, &
       PYTHAG3
  INTEGER IND(M)
  !
  !     THIS SUBROUTINE IS A TRANSLATION OF THE INVERSE ITERATION TECH-
  !     NIQUE IN THE ALGOL PROCEDURE TRISTURM BY PETERS AND WILKINSON.
  !     HANDBOOK FOR AUTO. COMP., VOL.II-LINEAR ALGEBRA, 418-439(1971).
  !
  !     THIS SUBROUTINE FINDS THOSE EIGENVECTORS OF A TRIDIAGONAL
  !     SYMMETRIC MATRIX CORRESPONDING TO SPECIFIED EIGENVALUES,
  !     USING INVERSE ITERATION.
  !
  !     ON INPUT
  !
  !        NM MUST BE SET TO THE ROW DIMENSION OF TWO-DIMENSIONAL
  !          ARRAY PARAMETERS AS DECLARED IN THE CALLING PROGRAM
  !          DIMENSION STATEMENT.
  !
  !        N IS THE ORDER OF THE MATRIX.
  !
  !        D CONTAINS THE DIAGONAL ELEMENTS OF THE INPUT MATRIX.
  !
  !        E CONTAINS THE SUBDIAGONAL ELEMENTS OF THE INPUT MATRIX
  !          IN ITS LAST N-1 POSITIONS.  E(1) IS ARBITRARY.
  !
  !        E2 CONTAINS THE SQUARES OF THE CORRESPONDING ELEMENTS OF E,
  !          WITH ZEROS CORRESPONDING TO NEGLIGIBLE ELEMENTS OF E.
  !          E(I) IS CONSIDERED NEGLIGIBLE IF IT IS NOT LARGER THAN
  !          THE PRODUCT OF THE RELATIVE MACHINE PRECISION AND THE SUM
  !          OF THE MAGNITUDES OF D(I) AND D(I-1).  E2(1) MUST CONTAIN
  !          0.0D0 IF THE EIGENVALUES ARE IN ASCENDING ORDER, OR 2.0D0
  !          IF THE EIGENVALUES ARE IN DESCENDING ORDER.  IF  BISECT,
  !          TRIDIB, OR  IMTQLV  HAS BEEN USED TO FIND THE EIGENVALUES,
  !          THEIR OUTPUT E2 ARRAY IS EXACTLY WHAT IS EXPECTED HERE.
  !
  !        M IS THE NUMBER OF SPECIFIED EIGENVALUES.
  !
  !        W CONTAINS THE M EIGENVALUES IN ASCENDING OR DESCENDING ORDER.
  !
  !        IND CONTAINS IN ITS FIRST M POSITIONS THE SUBMATRIX INDICES
  !          ASSOCIATED WITH THE CORRESPONDING EIGENVALUES IN W --
  !          1 FOR EIGENVALUES BELONGING TO THE FIRST SUBMATRIX FROM
  !          THE TOP, 2 FOR THOSE BELONGING TO THE SECOND SUBMATRIX, ETC.
  !
  !     ON OUTPUT
  !
  !        ALL INPUT ARRAYS ARE UNALTERED.
  !
  !        Z CONTAINS THE ASSOCIATED SET OF ORTHONORMAL EIGENVECTORS.
  !          ANY VECTOR WHICH FAILS TO CONVERGE IS SET TO ZERO.
  !
  !        IERR IS SET TO
  !          ZERO       FOR NORMAL RETURN,
  !          -R         IF THE EIGENVECTOR CORRESPONDING TO THE R-TH
  !                     EIGENVALUE FAILS TO CONVERGE IN 5 ITERATIONS.
  !
  !        RV1, RV2, RV3, RV4, AND RV6 ARE TEMPORARY STORAGE ARRAYS.
  !
  !     CALLS PYTHAG FOR  DSQRT(A*A + B*B) .
  !
  !     QUESTIONS AND COMMENTS SHOULD BE DIRECTED TO BURTON S. GARBOW,
  !     MATHEMATICS AND COMPUTER SCIENCE DIV, ARGONNE NATIONAL LABORATORY
  !
  !     THIS VERSION DATED AUGUST 1983.
  !
  !     ------------------------------------------------------------------
  !
  IERR = 0
  IF (M  ==  0) return
  TAG = 0
  ORDER = 1.0D0 - E2(1)
  Q = 0
  !     .......... ESTABLISH AND PROCESS NEXT SUBMATRIX ..........
100 P = Q + 1
  !
  DO Q = P, N
     IF (Q  ==  N) exit
     IF (E2(Q+1)  ==  0.0D0) exit
  enddo
  !     .......... FIND VECTORS BY INVERSE ITERATION ..........
  TAG = TAG + 1
  S = 0
  !
  loop920: DO R = 1, M
     IF (IND(R)  /=  TAG) cycle loop920
     ITS = 1
     X1 = W(R)
     IF (S  /=  0) GO TO 510
     !     .......... CHECK FOR ISOLATED ROOT ..........
     XU = 1.0D0
     IF (P  ==  Q) then
        RV6(P) = 1.0D0
        GO TO 870
     endif
     NORM = DABS(D(P))
     IP = P + 1
     !
     DO I = IP, Q
        NORM = DMAX1(NORM, DABS(D(I))+DABS(E(I)))
     enddo
     !     .......... EPS2 IS THE CRITERION FOR GROUPING,
     !                EPS3 REPLACES ZERO PIVOTS AND EQUAL
     !                ROOTS ARE MODIFIED BY EPS3,
     !                EPS4 IS TAKEN VERY SMALL TO AVOID OVERFLOW ..........
     EPS2 = 1.0D-3 * NORM
     EPS3 = EPSLN(NORM)
     UK = Q - P + 1
     EPS4 = UK * EPS3
     UK = EPS4 / DSQRT(UK)
     S = P
505  GROUP = 0
     GO TO 520
     !     .......... LOOK FOR CLOSE OR COINCIDENT ROOTS ..........
510  IF (DABS(X1-X0)  >=  EPS2) GO TO 505
     GROUP = GROUP + 1
     IF (ORDER * (X1 - X0)  <=  0.0D0) X1 = X0 + ORDER * EPS3
     !     .......... ELIMINATION WITH INTERCHANGES AND
     !                INITIALIZATION OF VECTOR ..........
520  V = 0.0D0
     !
     loop580: DO I = P, Q
        RV6(I) = UK
        IF (I  /=  P) then !GO TO 560
           IF (DABS(E(I))  >=  DABS(U))then ! GO TO 540
              !     .......... WARNING -- A DIVIDE CHECK MAY OCCUR HERE IF
              !                E2 ARRAY HAS NOT BEEN SPECIFIED CORRECTLY ..........
              XU = U / E(I)
              RV4(I) = XU
              RV1(I-1) = E(I)
              RV2(I-1) = D(I) - X1
              RV3(I-1) = 0.0D0
              IF (I  /=  Q) RV3(I-1) = E(I+1)
              U = V - XU * RV2(I-1)
              V = -XU * RV3(I-1)
              cycle loop580
           endif
           XU = E(I) / U
           RV4(I) = XU
           RV1(I-1) = U
           RV2(I-1) = V
           RV3(I-1) = 0.0D0
        endif
        U = D(I) - X1 - XU * V
        IF (I  /=  Q) V = E(I+1)
     enddo loop580
     !
     IF (U  ==  0.0D0) U = EPS3
     RV1(Q) = U
     RV2(Q) = 0.0D0
     RV3(Q) = 0.0D0
     !     .......... BACK SUBSTITUTION
     !                FOR I=Q STEP -1 UNTIL P DO -- ..........
600  DO II = P, Q
        I = P + Q - II
        RV6(I) = (RV6(I) - U * RV2(I) - V * RV3(I)) / RV1(I)
        V = U
        U = RV6(I)
     enddo
     !     .......... ORTHOGONALIZE WITH RESPECT TO PREVIOUS
     !                MEMBERS OF GROUP ..........
     IF (GROUP  ==  0) GO TO 700
     J = R
     !
     j=j-1
     loop680:DO JJ = 1, GROUP
        do while(IND(J)  /=  TAG)
           J = J - 1
        enddo
        XU = 0.0D0
        !
        DO I = P, Q
           XU = XU + RV6(I) * Z(I,J)
        enddo
        !
        DO I = P, Q
           RV6(I) = RV6(I) - XU * Z(I,J)
        enddo
     enddo loop680
     !
700  NORM = 0.0D0
     !
     DO I = P, Q
        NORM = NORM + DABS(RV6(I))
     enddo
     !
     IF (NORM  <  one)then   ! GO TO 840
        !     .......... FORWARD SUBSTITUTION ..........
        IF (ITS  /=  5) then  ! GO TO 830
           IF (NORM  ==  zero)then
              RV6(S) = EPS4
              S = S + 1
              IF (S  >  Q) S = P
           else
              XU = EPS4 / NORM
              RV6(p:q) = RV6(p:q) * XU
           endif
           !     .......... ELIMINATION OPERATIONS ON NEXT VECTOR
           !                ITERATE ..........

           DO I = IP, Q
              U = RV6(I)
              !     .......... IF RV1(I-1)  ==  E(I), A ROW INTERCHANGE
              !                WAS PERFORMED EARLIER IN THE
              !                TRIANGULARIZATION PROCESS ..........
              IF (RV1(I-1)  ==  E(I)) then
                 U = RV6(I-1)
                 RV6(I-1) = RV6(I)
              endif
              RV6(I) = U - RV4(I) * RV6(I-1)
           enddo
           !
           ITS = ITS + 1
           GO TO 600
           !     .......... SET ERROR -- NON-CONVERGED EIGENVECTOR ..........
        endif
        IERR = -R
        XU = 0.0D0
     else
     !     .......... NORMALIZE SO THAT SUM OF SQUARES IS
     !                1 AND EXPAND TO FULL ORDER ..........
        U = zero
     !
        DO I = P, Q
           U = PYTHAG3(U,RV6(I))
        enddo
        !
        XU = one / U
     endif
        !
870  Z(1:n,R) = zero
        !
     Z(p:q,R) = RV6(p:q) * XU
     !
     X0 = X1
  enddo loop920
  !
  IF (Q  <  N) GO TO 100
  RETURN
END SUBROUTINE TINVIT

