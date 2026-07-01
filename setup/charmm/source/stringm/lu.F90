! ********************************************************************************!
! Preprocessed by GNU CPP !
! ********************************************************************************!
!*******************************************************
!* LU decomposition routines *
!* *
!* F90 version by J-P Moreau, Paris *
!* --------------------------------------------------- *
!* Reference: *
!* *
!* Numerical Recipes By W.H. Press, B. P. Flannery, *
!* S.A. Teukolsky and W.T. Vetterling, Cambridge *
!* University Press, 1986 [BIBLI 08]. *
!* *
!*******************************************************
      MODULE LU
#if (KEY_STRINGM==1) /*  automatically protect all code */
!
      use chm_kinds
      implicit none
!
      CONTAINS
!
! ***************************************************************
! * Given an N x N matrix A, this routine replaces it by the LU *
! * decomposition of a rowwise permutation of itself. A and N *
! * are input. INDX is an output vector which records the row *
! * permutation effected by the partial pivoting; D is output *
! * as -1 or 1, depending on whether the number of row inter- *
! * changes was even or odd, respectively. This routine is used *
! * in combination with LUBKSB to solve linear equations or to *
! * invert a matrix. Return code is 1, if matrix is singular. *
! ***************************************************************
      Subroutine LUDCMP(A,N,INDX,D,CODE)
      integer :: n,i,j,k,imax
      real(chm_real), PARAMETER :: TINY=1.5D-16
      real(chm_real) :: AMAX,DUM, SUM, A(N,N),VV(N)
      integer :: CODE, D, INDX(N)
      D=1; CODE=0
      DO I=1,N
        AMAX=0.d0
        DO J=1,N
          IF (ABS(A(I,J)).GT.AMAX) AMAX=ABS(A(I,J))
        END DO ! j loop
        IF(AMAX.LT.TINY) THEN
          CODE = 1
          RETURN
        END IF
        VV(I) = 1.d0 / AMAX
      END DO ! i loop
      DO J=1,N
        DO I=1,J-1
          SUM = A(I,J)
          DO K=1,I-1
            SUM = SUM - A(I,K)*A(K,J)
          END DO ! k loop
          A(I,J) = SUM
        END DO ! i loop
        AMAX = 0.d0
        DO I=J,N
          SUM = A(I,J)
          DO K=1,J-1
            SUM = SUM - A(I,K)*A(K,J)
          END DO ! k loop
          A(I,J) = SUM
          DUM = VV(I)*ABS(SUM)
          IF(DUM.GE.AMAX) THEN
            IMAX = I
            AMAX = DUM
          END IF
        END DO ! i loop

        IF(J.NE.IMAX) THEN
          DO K=1,N
            DUM = A(IMAX,K)
            A(IMAX,K) = A(J,K)
            A(J,K) = DUM
          END DO ! k loop
          D = -D
          VV(IMAX) = VV(J)
        END IF

        INDX(J) = IMAX
        IF(ABS(A(J,J)) .lt. TINY) A(J,J) = TINY

        IF(J.NE.N) THEN
          DUM = 1.d0 / A(J,J)
          DO I=J+1,N
            A(I,J) = A(I,J)*DUM
          END DO ! i loop
        END IF
      END DO ! j loop

      RETURN
      END subroutine LUDCMP
!
! ******************************************************************
! * Solves the set of N linear equations A . X = B. Here A is *
! * input, not as the matrix A but rather as its LU decomposition, *
! * determined by the routine LUDCMP. INDX is input as the permuta-*
! * tion vector returned by LUDCMP. B is input as the right-hand *
! * side vector B, and returns with the solution vector X. A, N and*
! * INDX are not modified by this routine and can be used for suc- *
! * cessive calls with different right-hand sides. This routine is *
! * also efficient for plain matrix inversion. *
! ******************************************************************
      Subroutine LUBKSB(A,N,INDX,B)
      integer :: n, i, j, ii, ll
      real(chm_real) SUM, A(N,N),B(N)
      integer INDX(N)

      II = 0

      DO I=1,N
        LL = INDX(I)
        SUM = B(LL)
        B(LL) = B(I)
        IF(II.NE.0) THEN
          DO J=II,I-1
            SUM = SUM - A(I,J)*B(J)
          END DO ! j loop
        ELSE IF(SUM.NE.0.d0) THEN
          II = I
        END IF
        B(I) = SUM
      END DO ! i loop

      DO I=N,1,-1
        SUM = B(I)
        IF(I .lt. N) THEN
          DO J=I+1,N
            SUM = SUM - A(I,J)*B(J)
          END DO ! j loop
        END IF
        B(I) = SUM / A(I,I)
      END DO ! i loop
!
      END subroutine LUBKSB
!
!***** DRIVER ROUTINE FOR MATRIX INVERSION USING LU DECOMPOSITION *****
!*****
      subroutine inv_lu(A, B, n, err)
! this routine assumes that the matrix is nonsingular
      use stream
      integer, intent(out) :: err
      integer, intent(in) :: n
      real(chm_real), intent(in) :: A(n,n)
      real(chm_real), intent(out) :: B(n,n)
! locals
      integer :: i, perm(n), flag
      real(chm_real) :: Z(n,n)
!
      character(len=len("INV_LU>") ),parameter::whoami="INV_LU>";!macro
! begin
! save A
      B=A
! compute LUDCMP
      call LUDCMP(B, n, perm, flag, err)
! check err
      if (err.eq.1) then
! error
       call wrndie(0,whoami,trim(' MATRIX IS SINGULAR. ABORTING.'))
       return
      else
       Z=0d0
! now 'apply' inverse of A to identity columnwise, and store answers as A inverse
       do i=1,n
        Z(i,i)=1
        call LUBKSB(B, n, perm, Z(1,i))
       enddo
      endif
      B=Z
      end subroutine inv_lu
!*******************************************************************
#endif /* automatically protect all code */
      END MODULE LU
