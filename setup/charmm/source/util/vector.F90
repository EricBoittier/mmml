module vector
  implicit none

  ! real(chm_real) functions
  !    DOTVEC - Dotproduct of two vectors
  !    LENVEC - Inner product of a vector
  !    SUMVEC - Get the sum of a vector of numbers

contains

!***********************************************************************
!     ARRAY UTILITIES
!***********************************************************************
SUBROUTINE DOTPR(V,W,N,CX)
  !-----------------------------------------------------------------------
  !     COMPUTES THE INNER PRODUCT OF TWO VECTORS
  !
  use chm_kinds
  implicit none
  INTEGER N,I
  real(chm_real) V(N),W(N),C,CX
  !
  C=0.0
  DO I=1,N
     C=C+V(I)*W(I)
  enddo
  CX=C
  RETURN
END SUBROUTINE DOTPR

SUBROUTINE ADDCTV(V,W,N,CX)
  !-----------------------------------------------------------------------
  !     ADDS A SCALAR TIMES VECTOR W TO VECTOR V.
  !
  use chm_kinds
  implicit none
  INTEGER N,I
  real(chm_real) V(N),W(N),C,CX
  !
  C=CX
  V(1:n)=V(1:n)+W(1:n)*C
  RETURN
END SUBROUTINE ADDCTV

SUBROUTINE ADDCTVR4(V,W,N,CX)
  !-----------------------------------------------------------------------
  !     ADDS A SCALAR TIMES VECTOR W TO VECTOR V.
  !
  use chm_kinds
  implicit none
  INTEGER N,I
  real(chm_real4) V(N),W(N)
  real(chm_real)  C,CX
  !
  C=CX
  V(1:n)=V(1:n)+W(1:n)*C
  RETURN
END SUBROUTINE ADDCTVR4

SUBROUTINE ORTHOG(V,W,N)
  !-----------------------------------------------------------------------
  !     ORTHOGONALIZES VECTOR V TO W WHEN W IS NORMALIZED
  !
  use chm_kinds
  implicit none
  INTEGER N
  real(chm_real) V(N),W(N),C
  !
  CALL DOTPR(V,W,N,C)
  C=-C
  CALL ADDCTV(V,W,N,C)
  RETURN
END SUBROUTINE ORTHOG

SUBROUTINE NORMALL(V,N)
  !-----------------------------------------------------------------------
  !     NORMALIZES VECTOR V OF LENGTH N
  !
  use chm_kinds
  use stream
  use number
  implicit none
  INTEGER N,I
  real(chm_real) V(N),C
  !
  C=0.0
  DO I=1,N
     C=C+V(I)*V(I)
  enddo
  IF(C < 1.0D-12) THEN
     IF(WRNLEV >= 2) WRITE(OUTU,25) C
     !RCZ    CALL DIE
     V(1:n)=zero
     RETURN
  ENDIF
25 FORMAT(' **** WARNING **** TRYING TO NORMALIZE A ZERO VECTOR', &
       ' NORM=',E12.4/' IT WILL BE ZEROED.')
  !
  C=1.0/SQRT(C)
  V(1:n)=V(1:n)*C
  !
  RETURN
END SUBROUTINE NORMALL

SUBROUTINE SCALR8(A,N,SFACT)
  !-----------------------------------------------------------------------
  !     The real(chm_real) array, A, is scaled here by SFACT.
  !
  use chm_kinds
  implicit none
  !
  INTEGER I,N
  real(chm_real) A(N),SFACT
  !
  A(1:n) = SFACT * A(1:n)
  !
  RETURN
END SUBROUTINE SCALR8

SUBROUTINE CROSS3(A,B,C)
  !-----------------------------------------------------------------------
  !     Compute the cross-product/vector product/ of the vectors A&B with
  !     the result in C: C=AxB; all vectors are of dimension 3.
  !
  use chm_kinds
  implicit none
  real(chm_real) A(3),B(3),C(3)
  !
  C(1)=A(2)*B(3)-A(3)*B(2)
  C(2)=A(3)*B(1)-A(1)*B(3)
  C(3)=A(1)*B(2)-A(2)*B(1)
  RETURN
END SUBROUTINE CROSS3
!
LOGICAL FUNCTION EVEN(N)
  !
  INTEGER N
  !
  EVEN=.FALSE.
  IF(MOD(N,2)  ==  0) EVEN=.TRUE.
  RETURN
END FUNCTION EVEN

!***********************************************************************
!     VECTOR UTILITIES
!***********************************************************************
SUBROUTINE ADDVEC(A,B,C,N)
  !-----------------------------------------------------------------------
  !     Compute the sum, C, of the double precision vectors
  !     A and B.
  !
  use chm_kinds
  implicit none
  INTEGER I,N
  real(chm_real)  A(N),B(N),C(N)
  !
  C(1:n) = A(1:n) + B(1:n)
  RETURN
END SUBROUTINE ADDVEC

FUNCTION DOTVEC(A,B,N) result(dot)
  !-----------------------------------------------------------------------
  !     Compute the double precision dot product of two arrays.
  !
  use chm_kinds
  implicit none
  INTEGER I,N
  real(chm_real)  A(N),B(N),SUM,dot
  !
  SUM = 0.0
  DO I = 1,N
     SUM = SUM + A(I) * B(I)
  enddo
  DOT = SUM
  RETURN
END FUNCTION DOTVEC

FUNCTION SUMVEC(A,N) result(sumv)
  !-----------------------------------------------------------------------
  !     Compute the double precision sum of an array.
  !
  use chm_kinds
  implicit none
  INTEGER I,N
  real(chm_real)  A(N),SUM,sumv
  !
  SUM = 0.0
  DO I = 1,N
     SUM = SUM + A(I)
  enddo
  SUMV = SUM
  RETURN
END FUNCTION SUMVEC

FUNCTION LENVEC(A,N) result(len)
  !-----------------------------------------------------------------------
  !     Compute the length of a double precision vector.
  !
  use chm_kinds
  implicit none
  INTEGER I,N
  real(chm_real)  A(N),SUM,len
  !
  SUM = 0.0
  DO I = 1,N
     SUM = SUM + A(I) * A(I)
  enddo
  LEN = SQRT(SUM)
  RETURN
END FUNCTION LENVEC

#if KEY_CADPAC==0 /*cadpac_subvec*/
SUBROUTINE SUBVEC(A,B,C,N)
  !-----------------------------------------------------------------------
  !     Compute the difference, C, of the double precision vectors
  !     A and B.
  !
  use chm_kinds
  implicit none
  INTEGER I,N
  real(chm_real)  A(N),B(N),C(N)
  !

  C(1:n) = A(1:n) - B(1:n)
  RETURN
END SUBROUTINE SUBVEC
#endif /*  (cadpac_subvec)*/

! JZ_UW12: Stuff from old CHARMM (needed for SMBP)
      SUBROUTINE SNRMVEC(A,NRM,N)
!-----------------------------------------------------------------------
!     Compute the vector magnitude norm of the double precision vector
!     A.
!
      use chm_kinds
      implicit none
      INTEGER N
      real(chm_real) A(N),NRM
!
      INTEGER I
      real(chm_real) C
!
      C = 0.0
      DO 10 I = 1,N
        C = C + A(I)*A(I)
 10   CONTINUE
      C = SQRT(C)
      NRM = C
      RETURN
      END SUBROUTINE SNRMVEC

      SUBROUTINE SNRMVECR4(A,NRM,N)
!-----------------------------------------------------------------------
!     Compute the vector magnitude norm of the double precision vector
!     A.
!
      use chm_kinds
      implicit none
      INTEGER N
      real(chm_real4) A(N)
      real(chm_real)  NRM
!
      INTEGER I
      real(chm_real)  C
!
      C = 0.0
      DO 10 I = 1,N
        C = C + A(I)*A(I)
 10   CONTINUE
      C = SQRT(C)
      NRM = C
      RETURN
      END SUBROUTINE SNRMVECR4

end module vector

