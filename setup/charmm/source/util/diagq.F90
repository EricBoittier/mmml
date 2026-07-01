SUBROUTINE DIAGQ(NX,NFRQX,DD,VEC,A,B,P,W,EV,TA,TB,Y,NADD)
  !
  !   THIS ROUTINE IS A CONGLOMERATION OF GIVEN, HOUSEC, AND EIGEN
  !   WHERE THE BEST FEATURES OF EACH WERE KEPT AND SEVERAL OTHER
  !   MODIFICATIONS HAVE BEEN MADE TO INCREASE EFFICIENCY AND ACCURACY.
  !
  !   By Bernard R. Brooks   1981
  !
  !   NX      - ORDER OF MATRIX
  !   NFRQX   - NUMBER OF ROOTS DESIRED
  !   DD      - SECOND DERIVATIVE MATRIX IN UPPER TRIANGULAR FORM
  !   VEC     - EIGENVECTORS RETURNED (NX,NFRQX)
  !   EV      - EIGENVALUES RETURNED (NX)
  !   A,B,P,W,TA,TB,Y - ALL SCRATCH VECTORS (NX+1)
  !   NADD    - NUMBER OF LOWEST ROOTS TO SKIP
  !
  !
  use chm_kinds
  use number
  !
  use stream
  implicit none
  INTEGER NX,NFRQX,NADD
  real(chm_real) DD(*),VEC(*)
  real(chm_real) A(NX),B(NX),P(NX),W(NX),EV(NX),TA(NX),TB(NX),Y(NX)
  !
  real(chm_real) ETA,THETA,DEL1,DELTA,SMALL,DELBIG,THETA1,TOLER,ETAR
  real(chm_real) RPOWER,RPOW1,RAND1,FACTOR,ANORM,U,ANORMR
  real(chm_real) SUM1,BX,S,SGN,TEMP,XKAP,EXPR,ALIMIT, &
       ROOTL,ROOTX,TRIAL,F0
  real(chm_real) AROOT,ELIM1,ELIM2,T,EPR,XNORM,XNORM1,EVDIFF
  INTEGER N,NEV,NEVADD,NTOT,I,IPT,J,IJ,NN,MI,MI1,JI,JI2,II
  INTEGER ML,ML1,L,M,K,MJ,MJ1,NOMTCH,NOM,IA,ITER
  INTEGER J1,MK,MK1,KK
  !
  !CC
  real(chm_real) anumx
  anumx=zero
  do i=1,nx
     A(I)=anumx
     B(I)=anumx
     P(I)=anumx
     W(I)=anumx
     EV(I)=anumx
     TA(I)=anumx
     TB(I)=anumx
     Y(I)=anumx
  enddo
  !CC
  !
  ETA=RPRECI
  THETA=RBIGST
  !
  N=NX
  NEV=NFRQX
  NEVADD=NEV+NADD
  IA=0
  !
  DEL1=ETA/100.0
  DELTA=ETA**2*100.0
  SMALL=ETA**2/100.0
  DELBIG=THETA*DELTA/1000.0
  THETA1=1000.0/THETA
  TOLER=100.0*ETA
  ETAR=1.0/ETA
  RPOWER=8388608.0
  RPOW1=RPOWER*0.50
  RAND1=RPOWER-3.0
  !
  ! Find largest element.
  FACTOR=ZERO
  NTOT=(N*(N+1))/2
  DO I=1,NTOT
     FACTOR=MAX(FACTOR,ABS(DD(I)))
  ENDDO
  !
  ! Check for zero matrix.
  IF(FACTOR <= THETA1) THEN
     IF(WRNLEV >= 2) WRITE(OUTU,811)
811  FORMAT(' WARNING FROM <DIAGQ>. Zero matrix passed.', &
          ' Identity matrix returned.')
     DO I=1,NEV
        EV(I)=ZERO
        IPT=(I-1)*N
        DO J=1,N
           IPT=IPT+1
           VEC(IPT)=ZERO
           IF(I+NADD == J) VEC(IPT)=ONE
        ENDDO
     ENDDO
     RETURN
  ENDIF
  !
  ! Compute norm of matrix
  FACTOR=ONE/FACTOR
  IJ=0
  ANORM=ZERO
  DO I=1,N
     DO J=I,N
        IJ=IJ+1
        U=(DD(IJ)*FACTOR)**2
        IF(I == J) U=U*HALF
        ANORM=ANORM+U
     ENDDO
  ENDDO
  !
  ! Scale the matrix
  ANORM=SQRT(ANORM+ANORM)/FACTOR
  ANORMR=ONE/ANORM
  DO I=1,NTOT
     DD(I)=DD(I)*ANORMR
  ENDDO
  !
  NN=N-1
  MI=0
  MI1=N-1
  !
  ! Perform trigiagonalization
  DO I=1,NN
     SUM1=ZERO
     B(I)=ZERO
     JI=I+1
     IPT=MI+I
     A(I)=DD(IPT)
     IPT=IPT+1
     BX=DD(IPT)
     JI2=JI+1
     DO J=JI2,N
        IPT=IPT+1
        SUM1=SUM1+DD(IPT)*DD(IPT)
     ENDDO
     IF(SUM1 < SMALL) THEN
        B(I)=BX
        DD(MI+JI)=ZERO
     ELSE
        S=SQRT(SUM1+BX**2)
        SGN=SIGN(ONE,BX)
        TEMP=ABS(BX)
        W(JI)=SQRT(HALF*(ONE+(TEMP/S)))
        IPT=MI+JI
        DD(IPT)=W(JI)
        II=I+2
        IF(II <= N) THEN
           TEMP=SGN/(TWO*W(JI)*S)
           DO J=II,N
              IPT=IPT+1
              W(J)=TEMP * DD(IPT)
              DD(IPT)=W(J)
           ENDDO
        ENDIF
        B(I)=-SGN*S
        !
        DO J=JI,N
           P(J)=ZERO
        ENDDO
        ML=MI + MI1
        ML1=MI1-1
        DO L=JI,N
           IPT=ML+L
           DO M=L,N
              BX=DD(IPT)
              P(L)=P(L)+BX*W(M)
              IF(L /= M) P(M)=P(M)+BX*W(L)
              IPT=IPT+1
           ENDDO
           ML=ML +ML1
           ML1=ML1-1
        ENDDO
        !
        !
        XKAP=ZERO
        DO K=JI,N
           XKAP=XKAP+W(K)*P(K)
        ENDDO
        DO L=JI,N
           P(L)=P(L)-XKAP*W(L)
        ENDDO
        MJ=MI+MI1
        MJ1=MI1-1
        DO J=JI,N
           DO K=J,N
              EXPR=(P(J)*W(K))+(P(K)*W(J))
              DD(MJ+K)=DD(MJ+K)-EXPR-EXPR
           ENDDO
           MJ=MJ+MJ1
           MJ1=MJ1-1
        ENDDO
     ENDIF
     MI=MI+MI1
     MI1=MI1-1
  ENDDO
  !
  ! Begin sturm bisection method.
  !
  A(N)=DD(MI+N)
  B(N)=ZERO
  !
  ALIMIT=ONE
  DO I=1,N
     W(I)=B(I)
     B(I)=B(I)*B(I)
  ENDDO
  DO I=1,NEVADD
     EV(I)=ALIMIT
  ENDDO
  ROOTL=-ALIMIT
  !
  DO I=1,NEVADD
     ROOTX=ALIMIT
     DO J=I,NEVADD
        ROOTX=MIN(ROOTX,EV(J))
     ENDDO
     EV(I)=ROOTX
     !
130  CONTINUE
     TRIAL=(ROOTL+EV(I))*HALF
     EVDIFF=ABS(ROOTL-EV(I))
     IF(EVDIFF < THETA1) GOTO 200
     IF(EVDIFF*ETAR < ABS(TRIAL)) GOTO 200
     NOMTCH=N
     J=1
150  CONTINUE
     F0=A(J)-TRIAL
160  CONTINUE
     IF(ABS(F0) < THETA1) GOTO 170
     IF(F0 >= ZERO) NOMTCH=NOMTCH-1
     J=J+1
     IF(J > N) GOTO 180
     F0=A(J)-TRIAL-B(J-1)/F0
     GOTO 160
170  CONTINUE
     J=J+2
     NOMTCH=NOMTCH-1
     IF(J <= N) GOTO 150
180  CONTINUE
     IF(NOMTCH >= I) GOTO 190
     ROOTL=TRIAL
     GOTO 130
190  CONTINUE
     EV(I)=TRIAL
     NOM=MIN0(NEVADD,NOMTCH)
     EV(NOM)=TRIAL
     GOTO 130
200  CONTINUE
  ENDDO
  !
  ! Finished computing requested eigenvalues
  DO I=1,NEV
     EV(I)=EV(I+NADD)
  ENDDO
  !
  ! Compute eigenvectors (backtransformation)
  DO I=1,NEV
     AROOT=EV(I)
     DO J=1,N
        Y(J)=ONE
     ENDDO
     IA=IA+1
     IF(I == 1) THEN
        IA=0
     ELSE
        IF(ABS(EV(I-1)-AROOT) >= TOLER) IA=0
     ENDIF
     ELIM1=A(1)-AROOT
     ELIM2=W(1)
     DO J=1,NN
        IF(ABS(ELIM1) <= ABS(W(J))) THEN
           TA(J)=W(J)
           TB(J)=A(J+1)-AROOT
           P(J)=W(J+1)
           TEMP=ONE
           IF(ABS(W(J)) > THETA1) TEMP=ELIM1/W(J)
           ELIM1=ELIM2-TEMP*TB(J)
           ELIM2=-TEMP*W(J+1)
        ELSE
           TA(J)=ELIM1
           TB(J)=ELIM2
           P(J)=ZERO
           TEMP=W(J)/ELIM1
           ELIM1=A(J+1)-AROOT-TEMP*ELIM2
           ELIM2=W(J+1)
        ENDIF
        B(J)=TEMP
     ENDDO
     !
     TA(N)=ELIM1
     TB(N)=ZERO
     P(N)=ZERO
     P(NN)=ZERO
     ITER=1
     IF(IA /= 0) GOTO 460
     !
320  L=N+1
     DO J=1,N
        L=L-1
330     CONTINUE
        IF(L == N) THEN
           ELIM1=Y(L)
        ELSE IF(L == N-1) THEN
           ELIM1=Y(L)-Y(L+1)*TB(L)
        ELSE
           ELIM1=Y(L)-Y(L+1)*TB(L)-Y(L+2)*P(L)
        ENDIF
        !
        ! Overflow check
        IF(ABS(ELIM1) > DELBIG) THEN
           DO K=1,N
              Y(K)=Y(K)/DELBIG
           ENDDO
           GOTO 330
        ENDIF
        TEMP=TA(L)
        IF(ABS(TEMP) < DELTA) TEMP=DELTA
        Y(L)=ELIM1/TEMP
     ENDDO
     !
     IF(ITER == 2) GOTO 500
     ITER=ITER+1
     !
420  CONTINUE
     ELIM1=Y(1)
     DO J=1,NN
        IF(TA(J) == W(J)) THEN
           Y(J)=Y(J+1)
           ELIM1=ELIM1-Y(J+1)*B(J)
        ELSE
           Y(J)=ELIM1
           ELIM1=Y(J+1)-ELIM1*B(J)
        ENDIF
     ENDDO
     Y(N)=ELIM1
     GOTO 320
     !
460  CONTINUE
     DO J=1,N
        RAND1=MOD(4099.0*RAND1,RPOWER)
        Y(J)=RAND1/RPOW1-ONE
     ENDDO
     GOTO 320
     !
     ! Orthog to previous
500  IF(IA == 0) GOTO 550
     DO J1=1,IA
        K=I-J1
        TEMP=ZERO
        IPT=(K-1)*N
        DO J=1,N
           IPT=IPT+1
           TEMP=TEMP+Y(J)*VEC(IPT)
        ENDDO
        IPT=(K-1)*N
        DO J=1,N
           IPT=IPT+1
           Y(J)=Y(J)-TEMP*VEC(IPT)
        ENDDO
     ENDDO
550  CONTINUE
     IF(ITER == 1) GOTO 420
     !
     ! Normalize
560  CONTINUE
     ELIM1=ZERO
     DO J=1,N
        ELIM1=MAX(ELIM1,ABS(Y(J)))
     ENDDO
     TEMP=ZERO
     DO J=1,N
        ELIM2=Y(J)/ELIM1
        TEMP=TEMP+ELIM2*ELIM2
     ENDDO
     TEMP=ONE/(SQRT(TEMP)*ELIM1)
     DO J=1,N
        Y(J)=Y(J)*TEMP
        IF(ABS(Y(J)) < DEL1) Y(J)=ZERO
     ENDDO
     IPT=(I-1)*N
     DO J=1,N
        IPT=IPT+1
        VEC(IPT)=Y(J)
     ENDDO
  ENDDO
  !
  DO I=1,NEV
     IPT=(I-1)*N
     DO J=1,N
        IPT=IPT+1
        Y(J)=VEC(IPT)
     ENDDO
     !
     L=N-2
     MK=(N*(N-1))/2-3
     MK1=3
     !
     DO J=1,L
        T=ZERO
        K=N-J-1
        M=K+1
        DO KK=M,N
           T=T+DD(MK+KK)*Y(KK)
        ENDDO
        DO KK=M,N
           EPR=T*DD(MK+KK)
           Y(KK)=Y(KK)-EPR-EPR
        ENDDO
        MK=MK-MK1
        MK1=MK1+1
     ENDDO
     !
     T=ZERO
     DO J=1,N
        T=T+Y(J)*Y(J)
     ENDDO
     XNORM=SQRT(T)
     XNORM1=ONE/XNORM
     DO J=1,N
        Y(J)=Y(J)*XNORM1
     ENDDO
     !
     IPT=(I-1)*N
     DO J=1,N
        IPT=IPT+1
        VEC(IPT)=Y(J)
     ENDDO
  ENDDO
  !
  DO I=1,N
     EV(I)=EV(I)*ANORM
  ENDDO
  !
  RETURN
END SUBROUTINE DIAGQ

SUBROUTINE LSSOLV(R,IDTP,Y,W,N,X,A,INDEX,M)
  !
  !     THIS ROUTINE SOLVES A LEAST SQUARES PROBLEM WITH A (NSOL) ORDER
  !     POLYNOMIAL FIT.
  !            Y(I)= X(1) + X(2)*R(I) + X(3)*R(I)**2 + ...
  !
  !     By Bernard R. Brooks    1983
  !
  !     R(N) - DATA POINTS
  !     IDTP(N) - DERIVATIVE OF DATA EVALUATION
  !                 0 - FUNCTION VALUE
  !                 1 - FIRST DERIVATIVE
  !                 2 - SECOND DERIVATIVE
  !     Y(N) - FUNCTION VALUES
  !     W(N) - WEIGHTING OF POINTS
  !     N    - NUMBER OF DATA POINTS
  !     X(M) - SOLUTION COEFFICIENTS
  !     A(M,M+1) - SCRATCH MATRIX
  !     M    - ORDER OF THE SOLUTION
  !
  use chm_kinds
  use stream
  use machutil,only:die
  implicit none
  !      IMPLICIT real(chm_real) (A-H,O-Z)
  !
  INTEGER N,M
  real(chm_real) R(N),Y(N),W(N),X(M),A(M,M+1)
  INTEGER INDEX(M,2),IDTP(N)
  !
  real(chm_real) DETERM,AMAX,T,SWAP,PIVOT,RJ,RK
  INTEGER MM,K,J,JX,KX,I,IROW,ICOLUM,L
  !
  MM=M+1
  DO K=1,M
     KX=M-K
     DO J=1,M
        JX=M-J
        A(J,K)=0.0
        DO I=1,N
           IF(JX < IDTP(I)) THEN
              RJ=0.0
           ELSE IF(JX == IDTP(I) .AND. JX <= 1) THEN
              RJ=1.0
           ELSE IF(JX == IDTP(I) .AND. JX >= 2) THEN
              RJ=2.0
           ELSE IF(IDTP(I) == 0) THEN
              RJ=R(I)**JX
           ELSE IF(IDTP(I) == 1) THEN
              RJ=JX*R(I)**(JX-1)
           ELSE IF(IDTP(I) == 2) THEN
              RJ=JX*(JX-1)*R(I)**(JX-2)
           ELSE
              CALL DIE
           ENDIF
           !
           IF(KX < IDTP(I)) THEN
              RK=0.0
           ELSE IF(KX == IDTP(I) .AND. KX <= 1) THEN
              RK=1.0
           ELSE IF(KX == IDTP(I) .AND. KX >= 2) THEN
              RK=2.0
           ELSE IF(IDTP(I) == 0) THEN
              RK=R(I)**KX
           ELSE IF(IDTP(I) == 1) THEN
              RK=KX*R(I)**(KX-1)
           ELSE IF(IDTP(I) == 2) THEN
              RK=KX*(KX-1)*R(I)**(KX-2)
           ELSE
              CALL DIE
           ENDIF
           !
           A(J,K)=W(I)*RJ*RK+A(J,K)
        ENDDO
     ENDDO
  ENDDO
  !
  DO J=1,M
     JX=M-J
     A(J,MM)=0.0
     DO I=1,N
        IF(JX < IDTP(I)) THEN
           RJ=0.0
        ELSE IF(JX == IDTP(I) .AND. JX <= 1) THEN
           RJ=1.0
        ELSE IF(JX == IDTP(I) .AND. JX >= 2) THEN
           RJ=2.0
        ELSE IF(IDTP(I) == 0) THEN
           RJ=R(I)**JX
        ELSE IF(IDTP(I) == 1) THEN
           RJ=JX*R(I)**(JX-1)
        ELSE IF(IDTP(I) == 2) THEN
           RJ=JX*(JX-1)*R(I)**(JX-2)
        ELSE
           CALL DIE
        ENDIF
        !
        A(J,MM)=W(I)*RJ*Y(I)+A(J,MM)
     ENDDO
  ENDDO
  !
  !      IF(PRNLEV >= 2)
  !         DO J=1,3
  !            WRITE(OUTU,65) (A(J,K),K=1,MM)
  !  65        FORMAT(7F12.6)
  !         ENDDO
  !      ENDIF
  !
  !     INITIALIZATION
  !
  MM=M+1
  DETERM=1.0
  DO J=1,M
     INDEX(J,1)=0
  ENDDO
  !
  ! SEARCH FOR PIVOT ELEMENT
  !
  I=0
40 AMAX=-1.0
  !
  DO J=1,M
     IF(INDEX(J,1) == 0) THEN
        DO K=1,M
           IF(INDEX(K,1) == 0) THEN
              T=ABS(A(J,K))
              IF(T > AMAX) THEN
                 IROW=J
                 ICOLUM=K
                 AMAX=T
              ENDIF
           ENDIF
        ENDDO
     ENDIF
  ENDDO
  IF(AMAX <= 0.0) GOTO 720
  INDEX(ICOLUM,1)=IROW
  !
  ! INTERCHANGE ROWS TO PUT PIVOT ELEMENT ON DIAGONAL
  !
  IF(IROW /= ICOLUM) THEN
     DETERM=-DETERM
     DO L=1,MM
        SWAP=A(IROW,L)
        A(IROW,L)=A(ICOLUM,L)
        A(ICOLUM,L)=SWAP
     ENDDO
     I=I+1
     INDEX(I,2)=ICOLUM
  ENDIF
  !
  PIVOT=A(ICOLUM,ICOLUM)
  DETERM=PIVOT*DETERM
  !
  ! DIVIDE PIVOT ROW BY PIVOT ELEMENT
  !
  PIVOT=1.0/PIVOT
  DO L=1,MM
     A(ICOLUM,L)=A(ICOLUM,L)*PIVOT
  ENDDO
  !
  ! REDUCE NON-PIVOT ROWS
  !
  DO K=1,M
     IF(K /= ICOLUM) THEN
        T=A(K,ICOLUM)
        DO L=1,MM
           A(K,L)=A(K,L)-A(ICOLUM,L)*T
        ENDDO
        A(K,ICOLUM)=0.0
     ENDIF
  ENDDO
  GOTO 40
  !
  ! INTERCHANGE COLUMS
  !
630 ICOLUM=INDEX(I,2)
  IROW=INDEX(ICOLUM,1)
  DO K=1,M
     SWAP=A(K,IROW)
     A(K,IROW)=A(K,ICOLUM)
     A(K,ICOLUM)=SWAP
  ENDDO
  !
  I=I-1
720 IF(I /= 0) GOTO 630
  !
  !      DO J=1,3
  !         IF(PRNLEV >= 2) WRITE(OUTU,65) (A(J,K),K=1,MM)
  !      ENDDO
  !
  DO I=1,M
     X(I)=A(I,MM)
     !        IF(PRNLEV >= 2) WRITE(OUTU,66) (X(J),J=1,M)
     !  66    FORMAT(/7F12.6)
  ENDDO
  !
END SUBROUTINE LSSOLV

