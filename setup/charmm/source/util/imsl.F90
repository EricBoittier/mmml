      SUBROUTINE ZXMIN(FUNCT,N,NSIG,MAXFN,IOPT,X,H,G,F,W,IER)
!-----------------------------------------------------------------------
!   IMSL ROUTINE NAME   - ZXMIN 
!   LATEST REVISION     - JUNE 1, 1981
!   PURPOSE             - MINIMUM OF A FUNCTION OF N VARIABLES USING
!                           A QUASI-NEWTON METHOD 
!   ARGUMENTS    FUNCT  - A USER SUPPLIED ROUTINE WHICH CALCULATES 
!                           THE FUNCTION F FOR GIVEN PARAMETER VALUES 
!                           X(1),X(2),...,X(N). 
!                           THE CALLING SEQUENCE HAS THE FOLLOWING FORM 
!                           CALL FUNCT(N,X,F) 
!                           WHERE X IS A VECTOR OF LENGTH N.
!                           FUNCT MUST APPEAR IN AN EXTERNAL STATEMENT
!                           IN THE CALLING PROGRAM. FUNCT MUST NOT
!                           ALTER THE VALUES OF X(I),I=1,...,N OR N.
!                N      - THE NUMBER OF PARAMETERS (I.E., THE LENGTH
!                           OF X) (INPUT) 
!                NSIG   - CONVERGENCE CRITERION. (INPUT). THE NUMBER
!                           OF DIGITS OF ACCURACY REQUIRED IN THE 
!                           PARAMETER ESTIMATES.
!                           THIS CONVERGENCE CONDITION IS SATISIFIED IF 
!                           ON TWO SUCCESSIVE ITERATIONS, THE PARAMETER 
!                           ESTIMATES (I.E.,X(I), I=1,...,N) AGREE, 
!                           COMPONENT BY COMPONENT, TO NSIG DIGITS. 
!                MAXFN  - MAXIMUM NUMBER OF FUNCTION EVALUATIONS (I.E., 
!                           CALLS TO ROUTINE FUNCT) ALLOWED. (INPUT) 
!                IOPT   - OPTIONS SELECTOR. (INPUT) 
!                         IOPT = 0 CAUSES ZXMIN TO INITIALIZE THE 
!                           HESSIAN MATRIX H TO THE IDENTITY MATRIX.
!                         IOPT = 1 INDICATES THAT H HAS BEEN INITIALIZED
!                           BY THE USER TO A POSITIVE DEFINITE MATRIX.
!                         IOPT = 2 CAUSES ZXMIN TO COMPUTE THE DIAGONAL 
!                           VALUES OF THE HESSIAN MATRIX AND SET H TO 
!                           A DIAGONAL MATRIX CONTAINING THESE VALUES.
!                         IOPT = 3 CAUSES ZXMIN TO COMPUTE AN ESTIMATE
!                           OF THE HESSIAN IN H.
!                X      - VECTOR OF LENGTH N CONTAINING PARAMETER 
!                           VALUES. 
!                         ON INPUT, X MUST CONTAIN THE INITIAL
!                           PARAMETER ESTIMATES.
!                         ON OUTPUT, X CONTAINS THE FINAL PARAMETER 
!                           ESTIMATES AS DETERMINED BY ZXMIN. 
!                H      - VECTOR OF LENGTH N*(N+1)/2 CONTAINING AN
!                           ESTIMATE OF THE HESSIAN MATRIX
!                           D**2F/(DX(I)DX(J)), I,J=1,...,N.
!                           H IS STORED IN SYMMETRIC STORAGE MODE.
!                         ON INPUT, IF IOPT = 0, 2, OR 3 ZXMIN INITIA-
!                           LIZES H. AN INITIAL SETTING OF H BY THE 
!                           USER IS INDICATED BY IOPT=1.
!                           H MUST BE POSITIVE DEFINITE. IF IT IS NOT,
!                           A TERMINAL ERROR OCCURS.
!                         ON OUTPUT, H CONTAINS AN ESTIMATE OF THE
!                           HESSIAN AT THE FINAL PARAMETER ESTIMATES
!                           (I.E., AT X(1),X(2),...,X(N)) 
!                G      - A VECTOR OF LENGTH N CONTAINING AN ESTIMATE 
!                           OF THE GRADIENT DF/DX(I),I=1,...,N AT THE 
!                           FINAL PARAMETER ESTIMATES. (OUTPUT) 
!                F      - A SCALAR CONTAINING THE VALUE OF THE FUNCTION 
!                           AT THE FINAL PARAMETER ESTIMATES. (OUTPUT)
!                W      - A VECTOR OF LENGTH 3*N USED AS WORKING SPACE. 
!                         ON OUTPUT, WORK(I), CONTAINS FOR
!                           I = 1, THE NORM OF THE GRADIENT (I.E.,
!                             SQRT(G(1)**2+G(2)**2+...+G(N)**2))
!                           I = 2, THE NUMBER OF FUNCTION EVALUATIONS 
!                             PERFORMED.
!                           I = 3, AN ESTIMATE OF THE NUMBER OF 
!                             SIGNIFICANT DIGITS IN THE FINAL 
!                             PARAMETER ESTIMATES.
!                IER    - ERROR PARAMETER (OUTPUT)
!                         TERMINAL ERROR
!                           IER = 129 IMPLIES THAT THE INITIAL HESSIAN
!                             USED BY ZXMIN IS NOT POSITIVE DEFINITE, 
!                             EVEN AFTER ADDING A MULTIPLE OF THE 
!                             IDENTITY TO MAKE ALL DIAGONAL ELEMENTS
!                             POSITIVE. 
!                           IER = 130 IMPLIES THAT THE ITERATION WAS
!                             TERMINATED DUE TO ROUNDING ERRORS 
!                             BECOMING DOMINANT. THE PARAMETER
!                             ESTIMATES HAVE NOT BEEN DETERMINED TO 
!                             NSIG DIGITS.
!                           IER = 131 IMPLIES THAT THE ITERATION WAS
!                             TERMINATED BECAUSE MAXFN WAS EXCEEDED.
!   REQD. IMSL ROUTINES - UERTST,UGETIO,ZXMJN 
!-----------------------------------------------------------------------
!                                  SPECIFICATIONS FOR ARGUMENTS 
  use chm_kinds
      implicit none
      EXTERNAL FUNCT
      INTEGER  N,NSIG,MAXFN,IOPT,IER
      real(chm_real)   X(N),G(N),H(*),F,W(*)
!                                  SPECIFICATIONS FOR LOCAL VARIABLES 
      INTEGER  IG,IGG,IS,IDIFF,IR,IJ,I,J,NM1,JJ,JP1,L,KJ,K,  &
               IFN,LINK,ITN,II,IM1,JNT,NP1,JB,NJ
      real(chm_real)   REPS,AX,ZERO,ONE,HALF,SEVEN,FIVE,TWELVE,TEN,HH, &
               EPS,HJJ,V,DF,RELX,GS0,DIFF,AEPS,ALPHA,FF,TOT, &
               F1,F2,Z,GYS,DGS,SIG,ZZ,GNRM,P1,HHH,GHH,H2,F11,  &
               F12,F21,F22,HMAX,HMIN
      DATA     REPS/.710542735760100E-14/,AX/0.1/
      DATA     ZERO/0.0/,ONE/1.0/,HALF/0.5/, &
               SEVEN/7.0/,FIVE/5.0/,TWELVE/12.0/,  &
               TEN/10.0/,P1/0.1/
!                                  INITIALIZATION 
!                                  FIRST EXECUTABLE STATEMENT 
      IER = 0 
      HH = SQRT(REPS) 
      H2 = SQRT(HH) 
      EPS = TEN**(-NSIG)
      IG = N
      IGG = N+N 
      IS = IGG
      IDIFF = 1 
      IR = N
      W(1) = -ONE 
      W(2) = ZERO 
      W(3) = ZERO 
!                                  EVALUATE FUNCTION AT STARTING POINT
      DO 5 I=1,N
         G(I) = X(I)
    5 CONTINUE
      CALL FUNCT(N,G,F) 
      IFN = 1 
      IF (IOPT == 1) GO TO 50 
!                                  SET OFF-DIAGONAL ELEMENTS OF H TO 0.0
      IF (N == 1) GO TO 20
      IJ = 2
      DO 15 I=2,N 
         DO 10 J=2,I
            H(IJ) = ZERO
            IJ = IJ+1 
   10    CONTINUE 
         IJ = IJ+1
   15 CONTINUE
   20 IF (IOPT /= 0) GO TO 30 
!                                  SET DIAGONAL ELEMENTS OF H TO ONE
      IJ = 0
      DO 25 I=1,N 
         IJ = IJ+I
         H(IJ) = ONE
   25 CONTINUE
      GO TO 95
!                                  GET DIAGONAL ELEMENTS OF HESSIAN 
   30 IM1 = 1 
      NM1 = 1 
      NP1 = N+1 
      DO 35 I=2,NP1 
         HHH = H2*MAX(ABS(X(IM1)),AX) 
         G(IM1) = X(IM1)+HHH
         CALL FUNCT(N,G,F2) 
         G(IM1) = X(IM1)-HHH
         CALL FUNCT(N,G,FF) 
         H(NM1) = (FF-F+F2-F)/(HHH*HHH) 
         G(IM1) = X(IM1)
         IM1 = I
         NM1 = I+NM1
   35 CONTINUE
      IFN = IFN+N+N 
      IF (IOPT /= 3 .OR. N == 1) GO TO 50 
!                                  GET THE REST OF THE HESSIAN
      JJ = 1
      II = 2
      DO 45 I=2,N 
         GHH = H2*MAX(ABS(X(I)),AX) 
         DO 40 J=1,JJ 
            HHH = H2*MAX(ABS(X(J)),AX)
            G(I) = X(I)+GHH 
            G(J) = X(J)+HHH 
            CALL FUNCT(N,G,F22) 
            G(I) = X(I)-GHH 
            CALL FUNCT(N,G,F12) 
            G(J) = X(J)-HHH 
            CALL FUNCT(N,G,F11) 
            G(I) = X(I)+GHH 
            CALL FUNCT(N,G,F21) 
            H(II) = (F22-F21-F12+F11)/(4.*HHH*GHH)
            G(J) = X(J) 
            II = II+1 
   40    CONTINUE 
         G(I) = X(I)
         JJ = JJ+1
         II = II+1
   45 CONTINUE
      IFN = IFN+((N*N-N)*2) 
!                                  ADD MULTIPLE OF IDENTITY TO
!                                  MAKE DIAGONAL ELEMENTS POSITIVE
   50 HMIN = H(1) 
      HMAX = H(1) 
      NM1 = 1 
      DO 55 I=1,N 
         HMIN = MIN(HMIN,H(NM1))
         HMAX = MAX(HMAX,H(NM1))
         NM1 = NM1+I+1
   55 CONTINUE
      HMIN = MAX(0.01*(ABS(HMAX)+ABS(HMIN))-HMIN,ZERO) 
      NM1 = 1 
      DO 60 I=1,N 
         H(NM1) = H(NM1)+HMIN 
         NM1 = NM1+I+1
   60 CONTINUE
!                                  FACTOR H TO L*D*L-TRANSPOSE
      IR = N
      IF (N > 1) GO TO 65
      IF (H(1) > ZERO) GO TO 95
      H(1) = ZERO 
      IR = 0
      GO TO 90
   65 NM1 = N-1 
      JJ = 0
      DO 85 J=1,N 
         JP1 = J+1
         JJ = JJ+J
         HJJ = H(JJ)
         IF (HJJ > ZERO) GO TO 70
         H(JJ) = ZERO 
         IR = IR-1
         GO TO 85 
   70    IF (J == N) GO TO 85 
         IJ = JJ
         L = 0
         DO 80 I=JP1,N
            L = L+1 
            IJ = IJ+I-1 
            V = H(IJ)/HJJ 
            KJ = IJ 
            DO 75 K=I,N 
               H(KJ+L) = H(KJ+L)-H(KJ)*V
               KJ = KJ+K
   75       CONTINUE
            H(IJ) = V 
   80    CONTINUE 
   85 CONTINUE
   90 IF (IR == N) GO TO 95 
      IER = 129 
      GO TO 9000
   95 ITN = 0 
      DF = -ONE 
!                                  EVALUATE GRADIENT W(IG+I),I=1,...,N
  100 LINK = 1
      GO TO 280 
  105 CONTINUE
!                                  BEGIN ITERATION LOOP 
      IF (IFN >= MAXFN) GO TO 240 
      ITN = ITN+1 
      DO 110 I=1,N
         W(I) = -W(IG+I)
  110 CONTINUE
!                                  DETERMINE SEARCH DIRECTION W 
!                                    BY SOLVING H*W = -G WHERE
!                                    H = L*D*L-TRANSPOSE
      IF (IR < N) GO TO 140
!                                  N  ==  1 
      G(1) = W(1) 
      IF (N > 1) GO TO 115 
      W(1) = W(1)/H(1)
      GO TO 140 
!                                  N  >  1 
  115 II = 1
!                                  SOLVE L*W = -G 
      DO 125 I=2,N
         IJ = II
         II = II+I
         V = W(I) 
         IM1 = I-1
         DO 120 J=1,IM1 
            IJ = IJ+1 
            V = V-H(IJ)*W(J)
  120    CONTINUE 
         G(I) = V 
         W(I) = V 
  125 CONTINUE
!                                  SOLVE (D*LT)*Z = W WHERE 
!                                  LT = L-TRANSPOSE 
      W(N) = W(N)/H(II) 
      JJ = II 
      NM1 = N-1 
      DO 135 NJ=1,NM1 
!                                  J = N-1,N-2,...,1
         J = N-NJ 
         JP1 = J+1
         JJ = JJ-JP1
         V = W(J)/H(JJ) 
         IJ = JJ
         DO 130 I=JP1,N 
            IJ = IJ+I-1 
            V = V-H(IJ)*W(I)
  130    CONTINUE 
         W(J) = V 
  135 CONTINUE
!                                  DETERMINE STEP LENGTH ALPHA
  140 RELX = ZERO 
      GS0 = ZERO
      DO 145 I=1,N
         W(IS+I) = W(I) 
         DIFF = ABS(W(I))/MAX(ABS(X(I)),AX) 
         RELX = MAX(RELX,DIFF)
         GS0 = GS0+W(IG+I)*W(I) 
  145 CONTINUE
      IF (RELX == ZERO) GO TO 245 
      AEPS = EPS/RELX 
      IER = 130 
      IF (GS0 >= ZERO) GO TO 245
      IF (DF == ZERO) GO TO 245 
      IER = 0 
      ALPHA = (-DF-DF)/GS0
      IF (ALPHA <= ZERO) ALPHA = ONE
      ALPHA = MIN(ALPHA,ONE)
      IF (IDIFF == 2) ALPHA = MAX(P1,ALPHA) 
      FF = F
      TOT = ZERO
      JNT = 0 
!                                  SEARCH ALONG  X+ALPHA*W
  150 IF (IFN >= MAXFN) GO TO 240 
      DO 155 I=1,N
         W(I) = X(I)+ALPHA*W(IS+I)
  155 CONTINUE
      CALL FUNCT(N,W,F1)
      IFN = IFN+1 
      IF (F1 >= F) GO TO 180
      F2 = F
      TOT = TOT+ALPHA 
  160 IER = 0 
      F = F1
      DO 165 I=1,N
         X(I) = W(I)
  165 CONTINUE
      IF(JNT == 1) GOTO 200
      IF(JNT > 1) GOTO 205
      IF (IFN >= MAXFN) GO TO 240 
      DO 175 I=1,N
         W(I) = X(I)+ALPHA*W(IS+I)
  175 CONTINUE
      CALL FUNCT(N,W,F1)
      IFN = IFN+1 
      IF (F1 >= F) GO TO 205
      IF (F1+F2 >= F+F .AND. SEVEN*F1+FIVE*F2 > TWELVE*F) JNT = 2
      TOT = TOT+ALPHA 
      ALPHA = ALPHA+ALPHA 
      GO TO 160 
  180 CONTINUE
      IF (F == FF .AND. IDIFF.EQ.2 .AND. RELX > EPS) IER = 130 
      IF (ALPHA < AEPS) GO TO 245
      IF (IFN >= MAXFN) GO TO 240 
      ALPHA = HALF*ALPHA
      DO 185 I=1,N
         W(I) = X(I)+ALPHA*W(IS+I)
  185 CONTINUE
      CALL FUNCT(N,W,F2)
      IFN = IFN+1 
      IF (F2 >= F) GO TO 195
      TOT = TOT+ALPHA 
      IER = 0 
      F = F2
      DO 190 I=1,N
         X(I) = W(I)
  190 CONTINUE
      GO TO 200 
  195 Z = P1
      IF (F1+F > F2+F2) Z = ONE+HALF*(F-F1)/(F+F1-F2-F2) 
      Z = MAX(P1,Z) 
      ALPHA = Z*ALPHA 
      JNT = 1 
      GO TO 150 
  200 IF (TOT < AEPS) GO TO 245
  205 ALPHA = TOT 
!                                  SAVE OLD GRADIENT
      DO 210 I=1,N
         W(I) = W(IG+I) 
  210 CONTINUE
!                                  EVALUATE GRADIENT W(IG+I), I=1,...,N 
      LINK = 2
      GO TO 280 
  215 IF (IFN >= MAXFN) GO TO 240 
      GYS = ZERO
      DO 220 I=1,N
         GYS = GYS+W(IG+I)*W(IS+I)
         W(IGG+I) = W(I)
  220 CONTINUE
      DF = FF-F 
      DGS = GYS-GS0 
      IF (DGS <= ZERO) GO TO 105
      IF (DGS+ALPHA*GS0 > ZERO) GO TO 230
!                                  UPDATE HESSIAN H USING 
!                                    COMPLEMENTARY DFP FORMULA
      SIG = ONE/GS0 
      IR = -IR
      CALL ZXMJN(H,N,W,SIG,G,IR,0,ZERO) 
      DO 225 I=1,N
         G(I) = W(IG+I)-W(IGG+I)
  225 CONTINUE
      SIG = ONE/(ALPHA*DGS) 
      IR = -IR
      CALL ZXMJN(H,N,G,SIG,W,IR,0,ZERO) 
      GO TO 105 
!                                  UPDATE HESSIAN USING 
!                                    DFP FORMULA
  230 ZZ = ALPHA/(DGS-ALPHA*GS0)
      SIG = -ZZ 
      CALL ZXMJN(H,N,W,SIG,G,IR,0,REPS) 
      Z = DGS*ZZ-ONE
      DO 235 I=1,N
         G(I) = W(IG+I)+Z*W(IGG+I)
  235 CONTINUE
      SIG = ONE/(ZZ*DGS*DGS)
      CALL ZXMJN(H,N,G,SIG,W,IR,0,ZERO) 
      GO TO 105 
  240 IER = 131 
!                                  MAXFN FUNCTION EVALUATIONS 
      GO TO 250 
  245 IF (IDIFF == 2) GO TO 250 
!                                  CHANGE TO CENTRAL DIFFERENCES
      IDIFF = 2 
      GO TO 100 
  250 IF (IER /= 0) GO TO 255 
      IF (RELX <= EPS) GO TO 255
      GO TO 100 
!                                  MOVE GRADIENT TO G AND RETURN
  255 GNRM = ZERO 
      DO 260 I=1,N
         G(I) = W(IG+I) 
         GNRM = GNRM+G(I)*G(I)
  260 CONTINUE
      GNRM = SQRT(GNRM) 
      W(1) = GNRM 
      W(2) = IFN
      W(3) = -LOG10(MAX(REPS,RELX))
!                                  COMPUTE H = L*D*L-TRANSPOSE
      IF (N == 1) GO TO 9000
      NP1 = N+1 
      NM1 = N-1 
      JJ = (N*(NP1))/2
      DO 275 JB=1,NM1 
         JP1 = NP1-JB 
         JJ = JJ-JP1
         HJJ = H(JJ)
         IJ = JJ
         L = 0
         DO 270 I=JP1,N 
            L = L+1 
            IJ = IJ+I-1 
            V = H(IJ)*HJJ 
            KJ = IJ 
            DO 265 K=I,N
               H(KJ+L) = H(KJ+L)+H(KJ)*V
               KJ = KJ+K
  265       CONTINUE
            H(IJ) = V 
  270    CONTINUE 
         HJJ = H(JJ)
  275 CONTINUE
      GO TO 9000
!                                  EVALUATE GRADIENT
  280 IF (IDIFF == 2) GO TO 290 
!                                  FORWARD DIFFERENCES
!                                    GRADIENT = W(IG+I), I=1,...,N
      DO 285 I=1,N
         Z = HH*MAX(ABS(X(I)),AX) 
         ZZ = X(I)
         X(I) = ZZ+Z
         CALL FUNCT(N,X,F1) 
         W(IG+I) = (F1-F)/Z 
         X(I) = ZZ
  285 CONTINUE
      IFN = IFN+N 
                                !      GO TO (105, 215), LINK
      select case (link)
         case (1)
            goto 105
         case (2)
            goto 215
      end select

!                                  CENTRAL DIFFERENCES
!                                    GRADIENT = W(IG+I), I=1,...,N
  290 DO 295 I=1,N
         Z = HH*MAX(ABS(X(I)),AX) 
         ZZ = X(I)
         X(I) = ZZ+Z
         CALL FUNCT(N,X,F1) 
         X(I) = ZZ-Z
         CALL FUNCT(N,X,F2) 
         W(IG+I) = (F1-F2)/(Z+Z)
         X(I) = ZZ
  295 CONTINUE
      IFN = IFN+N+N 
                                !     GO TO (105, 215), LINK
      select case (link)
         case (1)
            goto 105
         case (2)
            goto 215
      end select

 9000 CONTINUE
!
! Let caller worry about IER. L.Nilsson, June 2002
!      IF (IER /= 0) CALL UERTST(IER,'ZXMIN') 
      RETURN
      END 

      SUBROUTINE ZXMJN(A,N,Z,SIG,W,IR,MK,EPS) 
!-----------------------------------------------------------------------
!   IMSL ROUTINE NAME   - ZXMJN 
!   LATEST REVISION     - NOVEMBER 1, 1984
!   PURPOSE             - NUCLEUS CALLED BY IMSL ROUTINES ZXMIN AND 
!                           ZXMWD 
!-----------------------------------------------------------------------
!                                  SPECIFICATIONS FOR ARGUMENTS 
  use chm_kinds      
      implicit none
      INTEGER N,IR,MK
      real(chm_real)  A(*),Z(N),SIG,W(N),EPS 
!                                  SPECIFICATIONS FOR LOCAL VARIABLES 
      INTEGER J,JJ,IJ,JP1,I,II,MM
      real(chm_real)  ZERO,ONE,FOUR,TI,V,TIM,AL,R,B,GM,Y,RINF,SQRINF 
      DATA    ZERO/0.0/,ONE/1.0/,FOUR/4.0/ 
      DATA    RINF/.319334449525555E+20/
!                                  UPDATE FACTORS GIVEN IN A
!                                    SIG*Z*Z-TRANSPOSE IS ADDED 
!                                  FIRST EXECUTABLE STATEMENT 
      SQRINF = SQRT(RINF) 
      IF (N > 1) GO TO 5 
!                                  N  ==  1 
      A(1) = A(1) + SIG*Z(1)*Z(1) 
      IR = 1
      IF (A(1) > ZERO) GO TO 9005
      A(1) = ZERO 
      IR = 0
      GO TO 9005
!                                  N  >  1 
    5 IF (SIG > ZERO) GO TO 65 
      IF (SIG == ZERO .OR. IR.EQ.0) GO TO 9005
      TI = ONE/SIG
      JJ = 0
      IF (MK == 0) GO TO 15 
!                                  L*W = Z ON INPUT 
      DO 10 J = 1, N
        JJ = JJ + J 
        IF (A(JJ) /= ZERO) TI = TI + (W(J)*W(J))/A(JJ)
   10 CONTINUE
      GO TO 40
!                                  SOLVE L*W = Z
   15 DO 20 J = 1, N
        W(J) = Z(J) 
   20 CONTINUE
      DO 35 J = 1, N
        JJ = JJ + J 
        V = W(J)
        IF (A(JJ) > ZERO) GO TO 25 
        W(J) = ZERO 
        GO TO 35
   25   TI = TI + (V*V)/A(JJ) 
        IF (J == N) GO TO 35
        IJ = JJ 
        JP1 = J + 1 
        DO 30 I = JP1, N
          IJ = IJ + I - 1 
          W(I) = W(I) - V*A(IJ) 
   30   CONTINUE
   35 CONTINUE
!                                  SET TI, TIM AND W
   40 IF (IR <= 0) GO TO 45 
      IF (TI > ZERO) GO TO 50
      IF(MK > 1) GOTO 55
      GOTO 65
!
   45 TI = ZERO 
      IR = -IR - 1
      GO TO 55
   50 TI = EPS/SIG
      IF (EPS == ZERO) IR = IR - 1
   55 TIM = TI
      II = JJ 
      I = N 
      DO 60 J = 1, N
        IF (A(II) /= ZERO) TIM = TI - (W(I)*W(I))/A(II) 
        W(I) = TI 
        TI = TIM
        II = II - I 
        I = I - 1 
   60 CONTINUE
      MM = 1
      GO TO 70
   65 MM = 0
      TIM = ONE/SIG 
   70 JJ = 0
!                                  UPDATE A 
      DO 120 J = 1, N 
        JJ = JJ + J 
        IJ = JJ 
        JP1 = J + 1 
!                                  UPDATE A(J,J)
        V = Z(J)
        IF (A(JJ) > ZERO) GO TO 95 
!                                  A(J,J)  ==  ZERO 
        IF (IR > 0 .OR. SIG < ZERO .OR. V == ZERO) GO TO 90 
        IR = 1 - IR 
        IF (V >= SQRINF) GO TO 75 
        A(JJ) = (V*V)/TIM 
        GO TO 80
   75   A(JJ) = RINF/TIM
   80   IF (J == N) GO TO 9005
        DO 85 I = JP1, N
          IJ = IJ + I - 1 
          A(IJ) = Z(I)/V
   85   CONTINUE
        GO TO 9005
   90   TI = TIM
        GO TO 120 
!                                  A(J,J)  >  ZERO 
   95   AL = V/A(JJ)
        TI = W(J) 
        IF (MM == 0) TI = TIM + V*AL
        R = TI/TIM
        A(JJ) = R*A(JJ) 
        IF (R == ZERO) GO TO 125
        IF (J == N) GO TO 125 
!                                  UPDATE REMAINDER OF COLUMN J 
        B = AL/TI 
        IF (R > FOUR) GO TO 105
        DO 100 I = JP1, N 
          IJ = IJ + I - 1 
          Z(I) = Z(I) - V*A(IJ) 
          A(IJ) = A(IJ) + B*Z(I)
  100   CONTINUE
        GO TO 115 
  105   GM = TIM/TI 
        DO 110 I = JP1, N 
          IJ = IJ + I - 1 
          Y = A(IJ) 
          A(IJ) = B*Z(I) + Y*GM 
          Z(I) = Z(I) - V*Y 
  110   CONTINUE
  115   TIM = TI
  120 CONTINUE
  125 IF (IR < 0) IR = -IR 
 9005 CONTINUE
      RETURN
      END 

      SUBROUTINE UERTST(IER,NAME)
!-----------------------------------------------------------------------
!   IMSL ROUTINE NAME   - UERTST
!   LATEST REVISION     - JUNE 1, 1982
!   PURPOSE             - PRINT A MESSAGE REFLECTING AN ERROR CONDITION 
!   ARGUMENTS    IER    - ERROR PARAMETER. (INPUT)
!                           IER = I+J WHERE 
!                             I = 128 IMPLIES TERMINAL ERROR MESSAGE, 
!                             I =  64 IMPLIES WARNING WITH FIX MESSAGE, 
!                             I =  32 IMPLIES WARNING MESSAGE.
!                             J = ERROR CODE RELEVANT TO CALLING
!                                 ROUTINE.
!                NAME   - A CHARACTER STRING OF LENGTH SIX PROVIDING
!                           THE NAME OF THE CALLING ROUTINE. (INPUT)
!   REQD. IMSL ROUTINES - USPKD
!-----------------------------------------------------------------------
  use stream
!                                  SPECIFICATIONS FOR ARGUMENTS 
      INTEGER IER
      character(len=*) NAME
!                                  SPECIFICATIONS FOR LOCAL VARIABLES 
      character(len=6) NAMEQ,NAMSET
      character(len=1) IEQ
      INTEGER I,IEQDF,IOUNIT,LEVEL,LEVOLD 
      DATA    NAMSET/'UERSET'/
      DATA    NAMEQ /'      '/ 
      DATA    LEVEL/4/,IEQDF/0/,IEQ/'='/ 
!
!                                  FIRST EXECUTABLE STATEMENT 
!                                  GET OUTPUT UNIT NUMBER 
      IOUNIT=OUTU
!                                  CHECK IER
      IF (IER > 999) GO TO 25
      IF (IER < -32) GO TO 55
      IF (IER <= 128) GO TO 5 
      IF (LEVEL < 1) GO TO 30
!                                  PRINT TERMINAL MESSAGE 
      IF (IEQDF == 1 .AND. WRNLEV >= 2) WRITE(IOUNIT,35) &
                                              IER,NAMEQ,IEQ,NAME
      IF (IEQDF == 0 .AND. WRNLEV >= 2) WRITE(IOUNIT,35) IER,NAME
      GO TO 30
    5 IF (IER <= 64) GO TO 10 
      IF (LEVEL < 2) GO TO 30
!                                  PRINT WARNING WITH FIX MESSAGE 
      IF (IEQDF == 1 .AND. WRNLEV >= 2) WRITE(IOUNIT,40) &
                                              IER,NAMEQ,IEQ,NAME
      IF (IEQDF == 0 .AND. WRNLEV >= 2) WRITE(IOUNIT,40) IER,NAME
      GO TO 30
   10 IF (IER <= 32) GO TO 15 
!                                  PRINT WARNING MESSAGE
      IF (LEVEL < 3) GO TO 30
      IF (IEQDF == 1 .AND. WRNLEV >= 2) WRITE(IOUNIT,45) &
                                              IER,NAMEQ,IEQ,NAME
      IF (IEQDF == 0 .AND. WRNLEV >= 2) WRITE(IOUNIT,45) IER,NAME
      GO TO 30
   15 CONTINUE
!                                  CHECK FOR UERSET CALL
      DO 20 I=1,6 
         IF (NAME /= NAMSET) GO TO 25 
   20 CONTINUE
      LEVOLD = LEVEL
      LEVEL = IER 
      IER = LEVOLD
      IF (LEVEL < 0) LEVEL = 4 
      IF (LEVEL > 4) LEVEL = 4 
      GO TO 30
   25 CONTINUE
      IF (LEVEL < 4) GO TO 30
!                                  PRINT NON-DEFINED MESSAGE
      IF (IEQDF == 1 .AND. WRNLEV >= 2) WRITE(IOUNIT,50) &
                                              IER,NAMEQ,IEQ,NAME
      IF (IEQDF == 0 .AND. WRNLEV >= 2) WRITE(IOUNIT,50) IER,NAME
   30 IEQDF = 0 
      RETURN
   35 FORMAT(' *** TERMINAL ERROR          (IER = ',I3,  &
             ') FROM IMSL ROUTINE ',A6,A1,A6)
   40 FORMAT(' *** WARNING WITH FIX ERROR  (IER = ',I3, &
             ') FROM IMSL ROUTINE ',A6,A1,A6)
   45 FORMAT(' *** WARNING ERROR           (IER = ',I3, &
             ') FROM IMSL ROUTINE ',A6,A1,A6)
   50 FORMAT(' *** UNDEFINED ERROR         (IER = ',I5,  &
             ') FROM IMSL ROUTINE ',A6,A1,A6)
! 
!                                  SAVE P FOR P = R CASE
!                                    P IS THE PAGE NAME
!                                    R IS THE ROUTINE NAME
   55 IEQDF = 1 
      NAMEQ = NAME
      RETURN
      END

      SUBROUTINE FFTCC(A,N,IWK,WK)
!-----------------------------------------------------------------------
      use chm_kinds
      implicit none
      INTEGER            N,IWK(*)
      real(chm_real)             WK(*)
      complex(chm_cmpx) :: A(N)
!                                  SPECIFICATIONS FOR LOCAL VARIABLES
      INTEGER            I,IAM,IAP,IBM,IBP,IC,ICC,ICF,ICK,ID,IDM1,II, &
                         IJA,IKB,IKT,ILL,IM,IRD,ISF,ISK,ISP,ISS,ITA,ITB, &
                         J,JA,JF,JJ,JK,K,K0,K1,K2,K3,KA,KB,KD2,KF,KH,KN, &
                         KT,KTP,L,L1,M,MM,MM1,MP
      real(chm_real)     CM,SM,C1,C2,C3,S1,S2,S3,C30,RAD,A0,A1,A4,B4, &
                         A2,A3,B0,B1,B2,B3,ZERO,HALF,ONE,TWO,Z0(2), &
                         Z1(2),Z2(2),Z3(2),Z4(2)
      complex(chm_cmpx) :: ZA0,ZA1,ZA2,ZA3,ZA4,AK2
      EQUIVALENCE        (ZA0,Z0(1)),(ZA1,Z1(1)),(ZA2,Z2(1)), &
                         (ZA3,Z3(1)),(A0,Z0(1)),(B0,Z0(2)),(A1,Z1(1)), &
                         (B1,Z1(2)),(A2,Z2(1)),(B2,Z2(2)),(A3,Z3(1)), &
                         (B3,Z3(2)),(ZA4,Z4(1)),(Z4(1),A4),(Z4(2),B4)
      DATA               RAD/6.283185307179586D0/, &
                         C30/.8660254037844386D0/
      DATA               ZERO,HALF,ONE,TWO/0.0D0,0.5D0,1.0D0,2.0D0/
!                                  FIRST EXECUTABLE STATEMENT
      IF (N  ==  1) GO TO 9005
      K = N
      M = 0
      J = 2
      JJ = 4
      JF = 0
!                                  DETERMINE THE SQUARE FACTORS OF N
      IWK(1) = 1
    5 I = K/JJ
      IF (I*JJ  /=  K) GO TO 10
      M = M+1
      IWK(M+1) = J
      K = I
      GO TO 5
   10 J = J + 2
      IF (J  ==  4) J = 3
      JJ = J * J
      IF (JJ  <=  K) GO TO 5
      KT = M
!                                  DETERMINE THE REMAINING FACTORS OF N
      J = 2
   15 I = K / J
      IF (I*J  /=  K) GO TO 20
      M = M + 1
      IWK(M+1) = J
      K = I
      GO TO 15
   20 J = J + 1
      IF (J  ==  3) GO TO 15
      J = J + 1
      IF(J <= K) GO TO 15
      K = IWK(M+1)
      IF (IWK(KT+1)  >  IWK(M+1)) K = IWK(KT+1)
      IF(KT <= 0) GO TO 30
      KTP = KT + 2
      DO 25  I = 1,KT
         J = KTP - I
         M = M+1
         IWK(M+1) = IWK(J)
   25 CONTINUE
   30 MP = M+1
      IC = MP+1
      ID = IC+MP
      ILL = ID+MP
      IRD = ILL+MP+1
      ICC = IRD+MP
      ISS = ICC+MP
      ICK = ISS+MP
      ISK = ICK+K
      ICF = ISK+K
      ISF = ICF+K
      IAP = ISF+K
      KD2 = (K-1) / 2 + 1
      IBP = IAP + KD2
      IAM = IBP + KD2
      IBM = IAM + KD2
      MM1 = M-1
      I = 1
   35 L = MP - I
      J = IC - I
      IWK(ILL+L) = 0
      IF ((IWK(J-1) + IWK(J))  ==  4) IWK(ILL+L) = 1
      IF (IWK(ILL+L)  ==  0) GO TO 40
      I = I + 1
      L = L - 1
      IWK(ILL+L) = 0
   40 I = I + 1
      IF(I <= MM1) GO TO 35
      IWK(ILL+1) = 0
      IWK(ILL+MP) = 0
      IWK(IC) = 1
      IWK(ID) = N
      DO 45  J = 1,M
         K = IWK(J+1)
         IWK(IC+J) = IWK(IC+J-1) * K
         IWK(ID+J) = IWK(ID+J-1) / K
         WK(IRD+J) = RAD/IWK(IC+J)
         C1 = RAD/K
         IF (K  <=  2) GO TO 45
         WK(ICC+J) = COS(C1)
         WK(ISS+J) = SIN(C1)
   45 CONTINUE
      MM = M
      IF (IWK(ILL+M)  ==  1) MM = M - 1
      IF (MM  <=  1) GO TO 50
      SM = IWK(IC+MM-2) * WK(IRD+M)
      CM = COS(SM)
      SM = SIN(SM)
   50 KB = 0
      KN = N
      JJ = 0
      I = 1
      C1 = ONE
      S1 = ZERO
      L1 = 1
   55 IF (IWK(ILL+I+1)  ==  1) GO TO 60
      KF = IWK(I+1)
      GO TO 65
   60 KF = 4
      I = I+1
   65 ISP = IWK(ID+I)
      IF (L1  ==  1) GO TO 70
      S1 = JJ * WK(IRD+I)
      C1 = COS(S1)
      S1 = SIN(S1)
!                                  FACTORS OF 2, 3, AND 4 ARE
!                                  HANDLED SEPARATELY.
   70 IF (KF  >  4) GO TO 140
                                ! GO TO (75,75,90,115), KF
      select case ( kf)
         case(1:2)    !75
            K0 = KB + ISP
            K2 = K0 + ISP
            IF (L1  ==  1) then
               K0 = K0 - 1
               do while ( k0 >= kb)
                                !       IF (K0  <  KB) GO TO 190
                  K2 = K2 - 1
                  AK2 = A(K2+1)
                  A(K2+1) = A(K0+1)-AK2
                  A(K0+1) = A(K0+1)+AK2
                  K0 = K0 - 1
               enddo
               goto 190
            endif
            K0 = K0 - 1 
                                ! IF (K0  <  KB) GO TO 190
            do while(k0 >= kb)
               K2 = K2 - 1
               ZA4 = A(K2+1)
               A0 = A4*C1-B4*S1
               B0 = A4*S1+B4*C1
               A(K2+1) = A(K0+1)-ZA0
               A(K0+1) = A(K0+1)+ZA0
               K0 = K0 - 1
            enddo
            goto 190
         

! 80         K0 = K0 - 1
!            IF (K0  <  KB) GO TO 190
!            K2 = K2 - 1
!            ZA4 = A(K2+1)
!            A0 = A4*C1-B4*S1
!            B0 = A4*S1+B4*C1
!            A(K2+1) = A(K0+1)-ZA0
!            A(K0+1) = A(K0+1)+ZA0
!            GO TO 80
!   85 K0 = K0 - 1
!      IF (K0  <  KB) GO TO 190
!      K2 = K2 - 1
!      AK2 = A(K2+1)
!      A(K2+1) = A(K0+1)-AK2
!      A(K0+1) = A(K0+1)+AK2
!      GO TO 85


                                ! 90 IF (L1  ==  1) GO TO 95
         case(3)
            IF (L1 /= 1) then    !90
               C2 = C1 * C1 - S1 * S1
               S2 = TWO * C1 * S1
            endif
            JA = KB + ISP - 1   !95
            KA = JA + KB
            IKB = KB+1
            IJA = JA+1
            DO  II = IKB,IJA
               K0 = KA - II + 1
               K1 = K0 + ISP
               K2 = K1 + ISP
               ZA0 = A(K0+1)
               IF (L1 /= 1) then  !GO TO 100
                  ZA4 = A(K1+1)
                  A1 = A4*C1-B4*S1
                  B1 = A4*S1+B4*C1
                  ZA4 = A(K2+1)
                  A2 = A4*C2-B4*S2
                  B2 = A4*S2+B4*C2
                                ! GO TO 105
               else
                  ZA1 = A(K1+1) ! 100
                  ZA2 = A(K2+1)
               endif
               A(K0+1) = cmplx(A0+A1+A2,B0+B1+B2,chm_cmpx)  !105
               A0 = -HALF * (A1+A2) + A0
               A1 = (A1-A2) * C30
               B0 = -HALF * (B1+B2) + B0
               B1 = (B1-B2) * C30
               A(K1+1) = cmplx(A0-B1,B0+A1,chm_cmpx)
               A(K2+1) = cmplx(A0+B1,B0-A1,chm_cmpx)
            enddo
            GO TO 190

         case(4)
           ! IF (L1  ==  1) GO TO 120    !115
            IF (L1 /= 1) then            !115
               C2 = C1 * C1 - S1 * S1
               S2 = TWO * C1 * S1
               C3 = C1 * C2 - S1 * S2
               S3 = S1 * C2 + C1 * S2
            endif
            JA = KB + ISP - 1   !120
            KA = JA + KB
            IKB = KB+1
            IJA = JA+1
            DO II = IKB,IJA
               K0 = KA - II + 1
               K1 = K0 + ISP
               K2 = K1 + ISP
               K3 = K2 + ISP
               ZA0 = A(K0+1)
                 !               IF (L1  ==  1) GO TO 125
               IF (L1 /= 1) then
                  ZA4 = A(K1+1)
                  A1 = A4*C1-B4*S1
                  B1 = A4*S1+B4*C1
                  ZA4 = A(K2+1)
                  A2 = A4*C2-B4*S2
                  B2 = A4*S2+B4*C2
                  ZA4 = A(K3+1)
                  A3 = A4*C3-B4*S3
                  B3 = A4*S3+B4*C3
               else             ! GO TO 130
                  ZA1 = A(K1+1) ! 125
                  ZA2 = A(K2+1)
                  ZA3 = A(K3+1)
               endif
               A(K0+1) = cmplx(A0+A2+A1+A3,B0+B2+B1+B3,chm_cmpx) ! 130
               A(K1+1) = cmplx(A0+A2-A1-A3,B0+B2-B1-B3,chm_cmpx)
               A(K2+1) = cmplx(A0-A2-B1+B3,B0-B2+A1-A3,chm_cmpx)
               A(K3+1) = cmplx(A0-A2+B1-B3,B0-B2-A1+A3,chm_cmpx)
            enddo
            GO TO 190
      end select
  140 JK = KF - 1
      KH = JK/2
      K3 = IWK(ID+I-1)
      K0 = KB + ISP
      IF (L1  ==  1) GO TO 150
      K = JK - 1
      WK(ICF+1) = C1
      WK(ISF+1) = S1
      DO 145 J = 1,K
         WK(ICF+J+1) = WK(ICF+J) * C1 - WK(ISF+J) * S1
         WK(ISF+J+1) = WK(ICF+J) * S1 + WK(ISF+J) * C1
  145 CONTINUE
  150 IF (KF  ==  JF) GO TO 160
      C2 = WK(ICC+I)
      WK(ICK+1) = C2
      WK(ICK+JK) = C2
      S2 = WK(ISS+I)
      WK(ISK+1) = S2
      WK(ISK+JK) = -S2
      DO 155 J = 1,KH
         K = JK - J
         WK(ICK+K) = WK(ICK+J) * C2 - WK(ISK+J) * S2
         WK(ICK+J+1) = WK(ICK+K)
         WK(ISK+J+1) = WK(ICK+J) * S2 + WK(ISK+J) * C2
         WK(ISK+K) = -WK(ISK+J+1)
  155 CONTINUE
  160 K0 = K0 - 1
      K1 = K0
      K2 = K0 + K3
      ZA0 = A(K0+1)
      A3 = A0
      B3 = B0
      DO 175 J = 1,KH
         K1 = K1 + ISP
         K2 = K2 - ISP
         IF (L1  ==  1) GO TO 165
         K = KF - J
         ZA4 = A(K1+1)
         A1 = A4*WK(ICF+J)-B4*WK(ISF+J)
         B1 = A4*WK(ISF+J)+B4*WK(ICF+J)
         ZA4 = A(K2+1)
         A2 = A4*WK(ICF+K)-B4*WK(ISF+K)
         B2 = A4*WK(ISF+K)+B4*WK(ICF+K)
         GO TO 170
  165    ZA1 = A(K1+1)
         ZA2 = A(K2+1)
  170    WK(IAP+J) = A1 + A2
         WK(IAM+J) = A1 - A2
         WK(IBP+J) = B1 + B2
         WK(IBM+J) = B1 - B2
         A3 = A1 + A2 + A3
         B3 = B1 + B2 + B3
  175 CONTINUE
      A(K0+1) = cmplx(A3,B3,chm_cmpx)
      K1 = K0
      K2 = K0 + K3
      DO 185 J = 1,KH
         K1 = K1 + ISP
         K2 = K2 - ISP
         JK = J
         A1 = A0
         B1 = B0
         A2 = ZERO
         B2 = ZERO
         DO 180  K = 1,KH
            A1 = A1 + WK(IAP+K) * WK(ICK+JK)
            A2 = A2 + WK(IAM+K) * WK(ISK+JK)
            B1 = B1 + WK(IBP+K) * WK(ICK+JK)
            B2 = B2 + WK(IBM+K) * WK(ISK+JK)
            JK = JK + J
            IF (JK  >=  KF) JK = JK - KF
  180    CONTINUE
         A(K1+1) = cmplx(A1-B2,B1+A2,chm_cmpx)
         A(K2+1) = cmplx(A1+B2,B1-A2,chm_cmpx)
  185 CONTINUE
      IF (K0  >  KB) GO TO 160
      JF = KF
  190 IF ( I  >=  MM ) GO TO 195
      I = I + 1
      GO TO 55
  195 I = MM
      L1 = 0
      KB = IWK(ID+I-1) + KB
      IF (KB  >=  KN) GO TO 215
  200 JJ = IWK(IC+I-2) + JJ
      IF (JJ  <  IWK(IC+I-1)) GO TO 205
      I = I - 1
      JJ = JJ - IWK(IC+I)
      GO TO 200
  205 IF (I  /=  MM) GO TO 210
      C2 = C1
      C1 = CM * C1 - SM * S1
      S1 = SM * C2 + CM * S1
      GO TO 70
  210 IF (IWK(ILL+I)  ==  1) I = I + 1
      GO TO 55
  215 I = 1
      JA = KT - 1
      KA = JA + 1
      IF(JA < 1) GO TO 225
      DO 220  II = 1,JA
         J = KA - II
         IWK(J+1) = IWK(J+1) - 1
         I = IWK(J+1) + I
  220 CONTINUE
!                                  THE RESULT IS NOW PERMUTED TO
!                                  NORMAL ORDER.
  225 IF (KT  <=  0) GO TO 270
      J = 1
      I = 0
      KB = 0
  230 K2 = IWK(ID+J) + KB
      K3 = K2
      JJ = IWK(IC+J-1)
      JK = JJ
      K0 = KB + JJ
      ISP = IWK(IC+J) - JJ
  235 K = K0 + JJ
  240 ZA4 = A(K0+1)
      A(K0+1) = A(K2+1)
      A(K2+1) = ZA4
      K0 = K0 + 1
      K2 = K2 + 1
      IF (K0  <  K) GO TO 240
      K0 = K0 + ISP
      K2 = K2 + ISP
      IF (K0  <  K3) GO TO 235
      IF (K0  >=  K3 + ISP) GO TO 245
      K0 = K0 - IWK(ID+J) + JJ
      GO TO 235
  245 K3 = IWK(ID+J) + K3
      IF (K3 - KB  >=  IWK(ID+J-1)) GO TO 250
      K2 = K3 + JK
      JK = JK + JJ
      K0 = K3 - IWK(ID+J) + JK
      GO TO 235
  250 IF (J  >=  KT) GO TO 260
      K = IWK(J+1) + I
      J = J + 1
  255 I = I + 1
      IWK(ILL+I) = J
      IF (I  <  K) GO TO 255
      GO TO 230
  260 KB = K3
      IF (I  <=  0) GO TO 265
      J = IWK(ILL+I)
      I = I - 1
      GO TO 230
  265 IF (KB  >=  N) GO TO 270
      J = 1
      GO TO 230
  270 JK = IWK(IC+KT)
      ISP = IWK(ID+KT)
      M = M - KT
      KB = ISP/JK-2
      IF (KT  >=  M-1 ) GO TO 9005
      ITA = ILL+KB+1
      ITB = ITA+JK
      IDM1 = ID-1
      IKT = KT+1
      IM = M+1
      DO 275 J = IKT,IM
         IWK(IDM1+J) = IWK(IDM1+J)/JK
  275 CONTINUE
      JJ = 0
      DO 290 J = 1,KB
         K = KT
  280    JJ = IWK(ID+K+1) + JJ
         IF (JJ  <  IWK(ID+K)) GO TO 285
         JJ = JJ - IWK(ID+K)
         K = K + 1
         GO TO 280
  285    IWK(ILL+J) = JJ
         IF (JJ  ==  J) IWK(ILL+J) = -J
  290 CONTINUE
!                                  DETERMINE THE PERMUTATION CYCLES
!                                  OF LENGTH GREATER THAN OR EQUAL
!                                  TO TWO.
      DO 300  J = 1,KB
         IF (IWK(ILL+J)  <=  0) GO TO 300
         K2 = J
  295    K2 = IABS(IWK(ILL+K2))
         IF (K2  ==  J) GO TO 300
         IWK(ILL+K2) = -IWK(ILL+K2)
         GO TO 295
  300 CONTINUE
!                                  REORDER A FOLLOWING THE
!                                  PERMUTATION CYCLES
      I = 0
      J = 0
      KB = 0
      KN = N
  305 J = J + 1
      IF (IWK(ILL+J)  <  0) GO TO 305
      K = IWK(ILL+J)
      K0 = JK * K + KB
  310 ZA4 = A(K0+I+1)
      WK(ITA+I) = A4
      WK(ITB+I) = B4
      I = I + 1
      IF (I  <  JK) GO TO 310
      I = 0
  315 K = -IWK(ILL+K)
      JJ = K0
      K0 = JK * K + KB
  320 A(JJ+I+1) = A(K0+I+1)
      I = I + 1
      IF (I  <  JK) GO TO 320
      I = 0
      IF (K  /=  J) GO TO 315
  325 A(K0+I+1) = cmplx(WK(ITA+I),WK(ITB+I),chm_cmpx)
      I = I + 1
      IF (I  <  JK) GO TO 325
      I = 0
      IF (J  <  K2) GO TO 305
      J = 0
      KB = KB + ISP
      IF (KB  <  KN) GO TO 305
 9005 RETURN
      END

      SUBROUTINE FFTRC(A,N,X,IWK,WK)
!-----------------------------------------------------------------------
      use chm_kinds
      implicit none
      INTEGER            N,IWK(*)
      real(chm_real)             A(N),WK(*)
      complex(chm_cmpx) :: X(*)
!                                  SPECIFICATIONS FOR LOCAL VARIABLES
      INTEGER            ND2P1,ND2,I,MTWO,M,IMAX,ND4,NP2,K,NMK,J
      real(chm_real)     RPI,ZERO,ONE,HALF,THETA,TP,G(2),B(2),Z(2),AI, &
                         AR
      complex(chm_cmpx) :: XIMAG,ALPH,BETA,GAM,S1,ZD
      EQUIVALENCE        (GAM,G(1)),(ALPH,B(1)),(Z(1),AR),(Z(2),AI), &
                         (ZD,Z(1))
      DATA               ZERO/0.0D0/,HALF/0.5D0/,ONE/1.0D0/,IMAX/24/
      DATA               RPI/3.141592653589793D0/
!                                  FIRST EXECUTABLE STATEMENT
      IF (N  /=  2) GO TO 5
!                                  N EQUAL TO 2
      ZD = cmplx(A(1),A(2),chm_cmpx)
      THETA = AR
      TP = AI
      X(2) = cmplx(THETA-TP,ZERO,chm_cmpx)
      X(1) = cmplx(THETA+TP,ZERO,chm_cmpx)
      GO TO 9005
    5 CONTINUE
!                                  N GREATER THAN 2
      ND2 = N/2
      ND2P1 = ND2+1
!                                  MOVE A TO X
      J = 1
      DO 6 I=1,ND2
         X(I) = cmplx(A(J),A(J+1),chm_cmpx)
         J = J+2
    6 CONTINUE
!                                  COMPUTE THE CENTER COEFFICIENT
      GAM = cmplx(ZERO,ZERO,chm_cmpx)
      DO 10 I=1,ND2
         GAM = GAM + X(I)
   10 CONTINUE
      TP = G(1)-G(2)
      GAM = cmplx(TP,ZERO,chm_cmpx)
!                                  DETERMINE THE SMALLEST M SUCH THAT
!                                  N IS LESS THAN OR EQUAL TO 2**M
      MTWO = 2
      M = 1
      DO 15 I=1,IMAX
         IF (ND2  <=  MTWO) GO TO 20
         MTWO = MTWO+MTWO
         M = M+1
   15 CONTINUE
   20 IF (ND2  ==  MTWO) GO TO 25
!                                  N IS NOT A POWER OF TWO, CALL FFTCC
      CALL FFTCC (X,ND2,IWK,WK)
      GO TO 30
!                                  N IS A POWER OF TWO, CALL FFT2C
   25 CALL FFT2C (X,M,IWK)
   30 ALPH = X(1)
      X(1) = B(1) + B(2)
      ND4 = (ND2+1)/2
      IF (ND4  <  2) GO TO 40
      NP2 = ND2 + 2
      THETA = RPI/ND2
      TP = THETA
      XIMAG = cmplx(ZERO,ONE,chm_cmpx)
!                                  DECOMPOSE THE COMPLEX VECTOR X
!                                  INTO THE COMPONENTS OF THE TRANSFORM
!                                  OF THE INPUT DATA.
      DO 35 K = 2,ND4
         NMK = NP2 - K
         S1 = conjg(X(NMK))
         ALPH = X(K) + S1
         BETA = XIMAG*(S1-X(K))
         S1 = cmplx(COS(THETA),SIN(THETA),chm_cmpx)
         X(K) = (ALPH+BETA*S1)*HALF
         X(NMK) = conjg(ALPH-BETA*S1)*HALF
         THETA = THETA + TP
   35 CONTINUE
   40 CONTINUE
      X(ND2P1) = GAM
 9005 RETURN
      END

      SUBROUTINE FFTSC(A,N,ST,CT,IWK,WK,CWK)
!-----------------------------------------------------------------------
      use chm_kinds
      use number
      implicit none
      INTEGER            N,IWK(*)
      real(chm_real)             A(N),ST(*),CT(*),WK(*)
      complex(chm_cmpx) :: CWK(*)
!                                  SPECIFICATIONS FOR LOCAL VARIABLES
      INTEGER            I,ND2M1
!                                  FIRST EXECUTABLE STATEMENT
      ND2M1 = N/2+1
!                                  CALL FFTRC TO COMPUTE THE FOURIER
!                                    COEFFICIENTS
      CALL FFTRC (A,N,CWK,IWK,WK)
!                                  THE COSINE TRANSFORM COEFFICIENT IS
!                                    THE REAL PART OF THE FOURIER
!                                    COEFFICIENT AND THE SINE TRANSFORM
!                                    COEFFICIENT IS THE IMAGINARY PART
      DO 5 I=1,ND2M1
         CT(I) = TWO*DBLE(CWK(I))
         ST(I) = TWO*aimag(CWK(I))
    5 CONTINUE
      RETURN
      END

      SUBROUTINE FFT2C(A,M,IWK)
!-----------------------------------------------------------------------
  use chm_kinds
      implicit none
      INTEGER            M,IWK(*)
      complex(chm_cmpx) :: A(*)
!                                  SPECIFICATIONS FOR LOCAL VARIABLES
      INTEGER            I,ISP,J,JJ,JSP,K,K0,K1,K2,K3,KB,KN,MK,MM,MP,N, &
                         N4,N8,N2,LM,NN,JK
      real(chm_real)     RAD,C1,C2,C3,S1,S2,S3,CK,SK,SQ,A0,A1,A2,A3, &
                         B0,B1,B2,B3,TWOPI,TEMP, &
                         ZERO,ONE,Z0(2),Z1(2),Z2(2),Z3(2)
      complex(chm_cmpx) :: ZA0,ZA1,ZA2,ZA3,AK2
      EQUIVALENCE        (ZA0,Z0(1)),(ZA1,Z1(1)),(ZA2,Z2(1)), &
                         (ZA3,Z3(1)),(A0,Z0(1)),(B0,Z0(2)),(A1,Z1(1)), &
                         (B1,Z1(2)),(A2,Z2(1)),(B2,Z2(2)),(A3,Z3(1)), &
                         (B3,Z3(2))
      DATA               SQ/.7071067811865475D0/, &
                         SK/.3826834323650898D0/, &
                         CK/.9238795325112868D0/, &
                         TWOPI/6.283185307179586D0/
      DATA               ZERO/0.0D0/,ONE/1.0D0/
!                                  SQ=SQRT2/2,SK=SIN(PI/8),CK=COS(PI/8)
!                                  TWOPI=2*PI
!                                  FIRST EXECUTABLE STATEMENT
      MP = M+1
      N = 2**M
      IWK(1) = 1
      MM = (M/2)*2
      KN = N+1
!                                  INITIALIZE WORK VECTOR
      DO 5  I=2,MP
         IWK(I) = IWK(I-1)+IWK(I-1)
    5 CONTINUE
      RAD = TWOPI/N
      MK = M - 4
      KB = 1
      IF (MM  ==  M) GO TO 15
      K2 = KN
      K0 = IWK(MM+1) + KB
   10 K2 = K2 - 1
      K0 = K0 - 1
      AK2 = A(K2)
      A(K2) = A(K0) - AK2
      A(K0) = A(K0) + AK2
      IF (K0  >  KB) GO TO 10
   15 C1 = ONE
      S1 = ZERO
      JJ = 0
      K = MM - 1
      J = 4
      IF (K  >=  1) GO TO 30
      GO TO 70
   20 IF (IWK(J)  >  JJ) GO TO 25
      JJ = JJ - IWK(J)
      J = J-1
      IF (IWK(J)  >  JJ) GO TO 25
      JJ = JJ - IWK(J)
      J = J - 1
      K = K + 2
      GO TO 20
   25 JJ = IWK(J) + JJ
      J = 4
   30 ISP = IWK(K)
      IF (JJ  ==  0) GO TO 40
!                                  RESET TRIGONOMETRIC PARAMETERS
      C2 = JJ * ISP * RAD
      C1 = COS(C2)
      S1 = SIN(C2)
   35 C2 = C1 * C1 - S1 * S1
      S2 = C1 * (S1 + S1)
      C3 = C2 * C1 - S2 * S1
      S3 = C2 * S1 + S2 * C1
   40 JSP = ISP + KB
!                                  DETERMINE FOURIER COEFFICIENTS
!                                    IN GROUPS OF 4
      DO 50 I=1,ISP
         K0 = JSP - I
         K1 = K0 + ISP
         K2 = K1 + ISP
         K3 = K2 + ISP
         ZA0 = A(K0)
         ZA1 = A(K1)
         ZA2 = A(K2)
         ZA3 = A(K3)
         IF (S1  ==  ZERO) GO TO 45
         TEMP = A1
         A1 = A1 * C1 - B1 * S1
         B1 = TEMP * S1 + B1 * C1
         TEMP = A2
         A2 = A2 * C2 - B2 * S2
         B2 = TEMP * S2 + B2 * C2
         TEMP = A3
         A3 = A3 * C3 - B3 * S3
         B3 = TEMP * S3 + B3 * C3
   45    TEMP = A0 + A2
         A2 = A0 - A2
         A0 = TEMP
         TEMP = A1 + A3
         A3 = A1 - A3
         A1 = TEMP
         TEMP = B0 + B2
         B2 = B0 - B2
         B0 = TEMP
         TEMP = B1 + B3
         B3 = B1 - B3
         B1 = TEMP
         A(K0) = cmplx(A0+A1,B0+B1,chm_cmpx)
         A(K1) = cmplx(A0-A1,B0-B1,chm_cmpx)
         A(K2) = cmplx(A2-B3,B2+A3,chm_cmpx)
         A(K3) = cmplx(A2+B3,B2-A3,chm_cmpx)
   50 CONTINUE
      IF (K  <=  1) GO TO 55
      K = K - 2
      GO TO 30
   55 KB = K3 + ISP
!                                  CHECK FOR COMPLETION OF FINAL
!                                    ITERATION
      IF (KN  <=  KB) GO TO 70
      IF (J  /=  1) GO TO 60
      K = 3
      J = MK
      GO TO 20
   60 J = J - 1
      C2 = C1
      IF (J  /=  2) GO TO 65
      C1 = C1 * CK + S1 * SK
      S1 = S1 * CK - C2 * SK
      GO TO 35
   65 C1 = (C1 - S1) * SQ
      S1 = (C2 + S1) * SQ
      GO TO 35
   70 CONTINUE
!                                  PERMUTE THE COMPLEX VECTOR IN
!                                    REVERSE BINARY ORDER TO NORMAL
!                                    ORDER
      IF(M  <=  1) GO TO 9005
      MP = M+1
      JJ = 1
!                                  INITIALIZE WORK VECTOR
      IWK(1) = 1
      DO 75  I = 2,MP
         IWK(I) = IWK(I-1) * 2
   75 CONTINUE
      N4 = IWK(MP-2)
      IF (M  >  2) N8 = IWK(MP-3)
      N2 = IWK(MP-1)
      LM = N2
      NN = IWK(MP)+1
      MP = MP-4
!                                  DETERMINE INDICES AND SWITCH A
      J = 2
   80 JK = JJ + N2
      AK2 = A(J)
      A(J) = A(JK)
      A(JK) = AK2
      J = J+1
      IF (JJ  >  N4) GO TO 85
      JJ = JJ + N4
      GO TO 105
   85 JJ = JJ - N4
      IF (JJ  >  N8) GO TO 90
      JJ = JJ + N8
      GO TO 105
   90 JJ = JJ - N8
      K = MP
   95 IF (IWK(K)  >=  JJ) GO TO 100
      JJ = JJ - IWK(K)
      K = K - 1
      GO TO 95
  100 JJ = IWK(K) + JJ
  105 IF (JJ  <=  J) GO TO 110
      K = NN - J
      JK = NN - JJ
      AK2 = A(J)
      A(J) = A(JJ)
      A(JJ) = AK2
      AK2 = A(K)
      A(K) = A(JK)
      A(JK) = AK2
  110 J = J + 1
!                                  CYCLE REPEATED UNTIL LIMITING NUMBER
!                                    OF CHANGES IS ACHIEVED
      IF (J  <=  LM) GO TO 80
!
 9005 RETURN
      END

      SUBROUTINE LINV1F(A,N,IA,AINV,IDGT,WKAREA,IER)
!-----------------------------------------------------------------------
  use chm_kinds
      implicit none
      integer n,ia,idgt,ier
      integer i,j
      real(chm_real)             A(IA,N),AINV(IA,N),WKAREA(*),ZERO,ONE
      DATA               ZERO/0.0D0/,ONE/1.0D0/
!                                  FIRST EXECUTABLE STATEMENT
      IER=0
      DO 10 I=1,N
         DO 5 J=1,N
            AINV(I,J) = ZERO
    5    CONTINUE
         AINV(I,I) = ONE
   10 CONTINUE
      CALL LEQT1F (A,N,N,IA,AINV,IDGT,WKAREA,IER)
      IF (IER  ==  0) GO TO 9005
!      CALL UERTST (IER,6HLINV1F)  B.ROUX
 9005 RETURN
      END

      SUBROUTINE LEQT1F(A,M,N,IA,B,IDGT,WKAREA,IER)
!-----------------------------------------------------------------------
  use chm_kinds
      implicit none
      integer n,m,ia,idgt,ier
      integer j
      real(chm_real) A(IA,*),B(IA,*),WKAREA(*)
      real(chm_real) D1,D2,WA
!                                  INITIALIZE IER
!                                  FIRST EXECUTABLE STATEMENT
      IER=0
!                                  DECOMPOSE A
      CALL LUDATN (A,IA,N,A,IA,IDGT,D1,D2,WKAREA,WKAREA,WA,IER)
      IF (IER  >  128) GO TO 9000
!                                  CALL ROUTINE LUELMN (FORWARD AND
!                                  BACKWARD SUBSTITUTIONS)
      DO 10 J=1,M
         CALL LUELMN (A,IA,N,B(1,J),WKAREA,B(1,J))
   10 CONTINUE
      IF (IER  ==  0) GO TO 9005
 9000 CONTINUE
!      CALL UERTST (IER,6HLEQT1F)   B.ROUX
 9005 RETURN
      END

      SUBROUTINE LUELMN(A,IA,N,B,APVT,X)
!-----------------------------------------------------------------------
  use chm_kinds
  use number
      implicit none
!
      integer ia,n
      integer i,iw,ip,im1,j,ib,ip1

      real(chm_real) A(IA,*),B(*),APVT(*),X(*),SUM
!
!                                  FIRST EXECUTABLE STATEMENT
!                                  SOLVE LY = B FOR Y
      DO I=1,N
         X(I) = B(I)
      ENDDO
      IW = 0
      DO I=1,N
         IP = APVT(I)
         SUM = X(IP)
         X(IP) = X(I)
         IF (IW  >  0) THEN
           IM1 = I-1
           DO J=IW,IM1
              SUM = SUM-A(I,J)*X(J)
           ENDDO
         ELSE
           IF (SUM  /=  ZERO) IW = I
         ENDIF
         X(I) = SUM
      ENDDO
!                                  SOLVE UX = Y FOR X
      DO IB=1,N
         I = N+1-IB
         IP1 = I+1
         SUM = X(I)
         DO J=IP1,N
            SUM = SUM-A(I,J)*X(J)
         ENDDO
         X(I) = SUM/A(I,I)
      ENDDO
      RETURN
      END

      SUBROUTINE LUDATN(A,IA,N,LU,ILU,IDGT,D1,D2,APVT,EQUIL,WA,IER)
!-----------------------------------------------------------------------
  use chm_kinds
      implicit none
      integer ia,n,ilu,idgt,ier
      integer i,j,jm1,im1,k,imax,jp1
      real(chm_real) A(IA,*),LU(ILU,*),APVT(*),EQUIL(*)
      real(chm_real)             D1,D2,WA,ZERO,ONE,FOUR,SIXTN,SIXTH, &
                         RN,WREL,BIGA,BIG,P,SUM,AI,WI,T,TEST,Q
      DATA               ZERO,ONE,FOUR,SIXTN,SIXTH/0.D0,1.D0,4.D0, &
                         16.D0,.0625D0/
!                                  FIRST EXECUTABLE STATEMENT
!                                  INITIALIZATION
      IER = 0
      RN = N
      WREL = ZERO
      D1 = ONE
      D2 = ZERO
      BIGA = ZERO
      DO 10 I=1,N
         BIG = ZERO
         DO 5 J=1,N
            P = A(I,J)
            LU(I,J) = P
            P = ABS(P)
            IF (P  >  BIG) BIG = P
    5    CONTINUE
         IF (BIG  >  BIGA) BIGA = BIG
         IF (BIG  ==  ZERO) GO TO 110
         EQUIL(I) = ONE/BIG
   10 CONTINUE
      DO 105 J=1,N
         JM1 = J-1
         IF (JM1  <  1) GO TO 40
!                                  COMPUTE U(I,J), I=1,...,J-1
         DO 35 I=1,JM1
            SUM = LU(I,J)
            IM1 = I-1
            IF (IDGT  ==  0) GO TO 25
!                                  WITH ACCURACY TEST
            AI = ABS(SUM)
            WI = ZERO
            IF (IM1  <  1) GO TO 20
            DO 15 K=1,IM1
               T = LU(I,K)*LU(K,J)
               SUM = SUM-T
               WI = WI+ABS(T)
   15       CONTINUE
            LU(I,J) = SUM
   20       WI = WI+ABS(SUM)
            IF (AI  ==  ZERO) AI = BIGA
            TEST = WI/AI
            IF (TEST  >  WREL) WREL = TEST
            GO TO 35
!                                  WITHOUT ACCURACY
   25       IF (IM1  <  1) GO TO 35
            DO 30 K=1,IM1
               SUM = SUM-LU(I,K)*LU(K,J)
   30       CONTINUE
            LU(I,J) = SUM
   35    CONTINUE
   40    P = ZERO
!                                  COMPUTE U(J,J) AND L(I,J), I=J+1,...,
         DO 70 I=J,N
            SUM = LU(I,J)
            IF (IDGT  ==  0) GO TO 55
!                                  WITH ACCURACY TEST
            AI = ABS(SUM)
            WI = ZERO
            IF (JM1  <  1) GO TO 50
            DO 45 K=1,JM1
               T = LU(I,K)*LU(K,J)
               SUM = SUM-T
               WI = WI+ABS(T)
   45       CONTINUE
            LU(I,J) = SUM
   50       WI = WI+ABS(SUM)
            IF (AI  ==  ZERO) AI = BIGA
            TEST = WI/AI
            IF (TEST  >  WREL) WREL = TEST
            GO TO 65
!                                  WITHOUT ACCURACY TEST
   55       IF (JM1  <  1) GO TO 65
            DO 60 K=1,JM1
               SUM = SUM-LU(I,K)*LU(K,J)
   60       CONTINUE
            LU(I,J) = SUM
   65       Q = EQUIL(I)*ABS(SUM)
            IF (P  >=  Q) GO TO 70
            P = Q
            IMAX = I
   70    CONTINUE
!                                  TEST FOR ALGORITHMIC SINGULARITY
         IF (RN+P  ==  RN) GO TO 110
         IF (J  ==  IMAX) GO TO 80
!                                  INTERCHANGE ROWS J AND IMAX
         D1 = -D1
         DO 75 K=1,N
            P = LU(IMAX,K)
            LU(IMAX,K) = LU(J,K)
            LU(J,K) = P
   75    CONTINUE
         EQUIL(IMAX) = EQUIL(J)
   80    APVT(J) = IMAX
         D1 = D1*LU(J,J)
   85    IF (ABS(D1)  <=  ONE) GO TO 90
         D1 = D1*SIXTH
         D2 = D2+FOUR
         GO TO 85
   90    IF (ABS(D1)  >=  SIXTH) GO TO 95
         D1 = D1*SIXTN
         D2 = D2-FOUR
         GO TO 90
   95    CONTINUE
         JP1 = J+1
         IF (JP1  >  N) GO TO 105
!                                  DIVIDE BY PIVOT ELEMENT U(J,J)
         P = LU(J,J)
         DO 100 I=JP1,N
            LU(I,J) = LU(I,J)/P
  100    CONTINUE
  105 CONTINUE
!                                  PERFORM ACCURACY TEST
      IF (IDGT  ==  0) GO TO 9005
      P = 3*N+3
      WA = P*WREL
      IF (WA+10.D0**(-IDGT)  /=  WA) GO TO 9005
      IER = 34
      GO TO 9000
!                                  ALGORITHMIC SINGULARITY
  110 IER = 129
      D1 = ZERO
      D2 = ZERO
 9000 CONTINUE
!                                  PRINT ERROR
!      CALL UERTST(IER,6HLUDATN) B.ROUX
 9005 CONTINUE
      RETURN
      END

      SUBROUTINE LINV3F(A,B,IJOB,N,IA,D1,D2,WKAREA,IER)
!-----------------------------------------------------------------------
!     IMSL ROUTINE NAME   - LINV3F
!     PURPOSE             - IN PLACE INVERSE, EQUATION SOLUTION, AND/OR 
!                           DETERMINANT EVALUATION - FULL STORAGE MODE
! 
!     REQD. IMSL ROUTINES - LUDATN,LUELMN,UERTST,UGETIO 
!
!     4-Feb-2003, for LN corman2.src modification
!                 from UT-CDC Single precision version 9.2
!-----------------------------------------------------------------------
! 
  use chm_kinds
  use number
      implicit none
!
      INTEGER IJOB,N,IA,IER,NM1,I,J,K,L,M,II,JP
      real(chm_real)  A(IA,*),B(*),WKAREA(*),C1,C2,D1,D2,WA,SUM,C
!                                  FIRST EXECUTABLE STATEMENT 
!                                  LU DECOMPOSITION OF A
      CALL LUDATN (A,IA,N,A,IA,0,C1,C2,WKAREA,WKAREA,WA,IER)
      IF (D1  <  ZERO .AND. IJOB  >=  1 .AND. IJOB .LT. 4) GO TO 5 
      D1 = C1 
      D2 = C2 
    5 IF (IER  >=  128) GO TO 60
      IF (IJOB  <=  0 .OR. IJOB  >  4) GO TO 55
!                                  SOLVE AX = B 
      IF (IJOB  ==  2 .OR. IJOB .EQ. 3) CALL LUELMN (A,IA,N,B,WKAREA,B)
      IF (IJOB  /=  1 .AND. IJOB .NE. 3) GO TO 9005 
!                                  MATRIX INVERSION 
      A(N,N) = ONE/A(N,N) 
      NM1 = N-1 
      IF (NM1  <  1) GO TO 9005
      DO 40 II=1,NM1
         L = N-II 
         M = L+1
         DO 15 I=M,N
            SUM = ZERO
            DO 10 K=M,N 
               SUM = SUM-A(I,K)*A(K,L)
   10       CONTINUE
            WKAREA(N+I) = SUM 
   15    CONTINUE 
         DO 20 I=M,N
            A(I,L) = WKAREA(N+I)
   20    CONTINUE 
         DO 30 J=L,N
            SUM = ZERO
            IF (J  ==  L) SUM = ONE 
            DO 25 K=M,N 
               SUM = SUM-A(L,K)*A(K,J)
   25       CONTINUE
            WKAREA(N+J) = SUM/A(L,L)
   30    CONTINUE 
         DO 35 J=L,N
            A(L,J) = WKAREA(N+J)
   35    CONTINUE 
   40 CONTINUE
!                                  PERMUTE COLUMNS OF A INVERSE 
      DO 50 I=1,N 
         J = N-I+1
         JP = WKAREA(J) 
         IF (J  ==  JP) GO TO 50
         DO 45 K=1,N
            C = A(K,JP) 
            A(K,JP) = A(K,J)
            A(K,J) = C
   45    CONTINUE 
   50 CONTINUE
      GO TO 9005
   55 CONTINUE
!                                  WARNING WITH FIX - IJOB WAS SET
!                                  INCORRECTLY
      IER = 65
      GO TO 9000
!                                  TERMINAL ERROR - MATRIX A IS 
!                                  ALGORITHMICALLY SINGULAR 
   60 IER = 130 
 9000 CONTINUE
!      CALL UERTST(IER,6HLINV3F) 
 9005 RETURN
      END

