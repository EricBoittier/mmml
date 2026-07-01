#if KEY_MC==1
SUBROUTINE MCDCMP(A,N,NP,INDX,D)
  !
  !       LU decomposition routine from Numerical Recipes (Press, 1986)
  !
  !       modified to be implicit none
  !
  !       Aaron R. Dinner
  !
  use chm_kinds
  implicit none
  INTEGER N, NP
  INTEGER INDX(N)
  real(chm_real) A(NP,NP), D
  !
  INTEGER NMAX, IMAX, I, J, K
  real(chm_real) TINY, AAMAX, SUM, DUM
  !       NMAX is the largest expected N
  PARAMETER (NMAX=10, TINY=1.0E-20)
  real(chm_real) VV(NMAX)

  D = 1.0
  DO I = 1, N
     AAMAX = 0.0
     DO J = 1, N
        IF (ABS(A(I,J)).GT.AAMAX) AAMAX = ABS(A(I,J))
     ENDDO
     IF (AAMAX .EQ. 0.0)  &
          CALL WRNDIE (-2, '<MCDCMP>', 'SINGULAR MATRIX')
     VV(I)=1.0/AAMAX
  ENDDO

  DO J = 1, N
     DO I = 1, J-1
        SUM = A(I,J)
        DO K = 1, I-1
           SUM = SUM - A(I,K)*A(K,J)
        ENDDO
        A(I,J) = SUM
     ENDDO
     AAMAX = 0.0
     DO I = J, N
        SUM = A(I,J)
        DO K = 1, J-1
           SUM = SUM - A(I,K)*A(K,J)
        ENDDO
        A(I,J) = SUM
        DUM=VV(I)*ABS(SUM)
        IF (DUM.GE.AAMAX) THEN
           IMAX = I
           AAMAX = DUM
        ENDIF
     ENDDO
     IF (J .NE. IMAX) THEN
        DO K = 1, N
           DUM = A(IMAX,K)
           A(IMAX,K) = A(J,K)
           A(J,K) = DUM
        ENDDO
        D = -D
        VV(IMAX) = VV(J)
     ENDIF
     INDX(J) = IMAX
     IF (A(J,J) .EQ. 0) A(J,J)=TINY
     IF (J .NE. N) THEN
        DUM = 1.0/A(J,J)
        DO I = J+1, N
           A(I,J) = A(I,J)*DUM
        ENDDO
     ENDIF
  ENDDO

  RETURN
END SUBROUTINE MCDCMP

SUBROUTINE MCBKSB(A,N,NP,INDX,B)
  !
  !       Solves N linear equations using an 
  !       LU decomposition of a matrix from Numerical Recipes (Press, 1986).
  !
  !       modified to be implicit none
  !
  !       Aaron R. Dinner
  !
  use chm_kinds
  implicit none
  INTEGER N, NP
  INTEGER INDX(N)
  real(chm_real) A(NP,NP), B(N)
  !
  INTEGER II, I, LL, J
  real(chm_real) SUM

  II = 0
  DO I = 1, N
     LL = INDX(I)
     SUM = B(LL)
     B(LL) = B(I)
     IF (II .NE. 0) THEN
        DO J = II, I-1
           SUM = SUM - A(I,J)*B(J)
        ENDDO
     ELSE IF (SUM .NE. 0.0) THEN
        II = I
     ENDIF
     B(I) = SUM
  ENDDO
  DO I = N, 1, -1
     SUM = B(I)
     DO J = I+1, N
        SUM = SUM - A(I,J)*B(J)
     ENDDO
     B(I) = SUM/A(I,I)
  ENDDO
  return
end SUBROUTINE MCBKSB
#endif 

SUBROUTINE NULL_NUMREC
  RETURN
END SUBROUTINE NULL_NUMREC

