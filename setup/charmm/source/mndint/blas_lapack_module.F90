#if KEY_MNDO97==1
!=======================================================================!
!============ BLAS routines ============================================!
!=======================================================================|
!  subroutine dgemm_mn(TRANSA,TRANSB,M,N,K,ALPHA,A,LDA,B,LDB,BETA,C,LDC)
!
!  use chm_kinds
!  use number
! 
!  implicit none
!
!  character(LEN=1):: TRANSA, TRANSB
!  integer ::         M, N, K, LDA, LDB, LDC
!  real(chm_real)::   ALPHA, BETA
!  real(chm_real)::   A( LDA, * ), B( LDB, * ), C( LDC, * )
!  !
!  !  Purpose
!  !  DGEMM  performs one of the matrix-matrix operations
!  !
!  !         C := alpha*op( A )*op( B ) + beta*C, 
!  !         where  op( X ) is one of op( X ) = X   or   op( X ) = X',
!  !
!  !  alpha and beta are scalars, and A, B and C are matrices, with op( A )
!  !  an m by k matrix,  op( B )  a  k by n matrix and  C an m by n matrix.
!  !
!  !  Parameters
!  !  TRANSA - CHARACTER*1.
!  !           On entry, TRANSA specifies the form of op( A ) to be used in
!  !           the matrix multiplication as follows:
!  !           TRANSA = 'N' or 'n',  op( A ) = A.
!  !           TRANSA = 'T' or 't',  op( A ) = A'.
!  !           TRANSA = 'C' or 'c',  op( A ) = A'.
!  !
!  !           Unchanged on exit.
!  !
!  !  TRANSB - CHARACTER*1.
!  !           On entry, TRANSB specifies the form of op( B ) to be used in
!  !           the matrix multiplication as follows:
!  !           TRANSB = 'N' or 'n',  op( B ) = B.
!  !           TRANSB = 'T' or 't',  op( B ) = B'.
!  !           TRANSB = 'C' or 'c',  op( B ) = B'.
!  !
!  !           Unchanged on exit.
!  !
!  !  M      - INTEGER.
!  !           On entry,  M  specifies  the number  of rows  of the  matrix
!  !           op( A )  and of the  matrix  C.  M  must  be at least  zero.
!  !           Unchanged on exit.
!  !
!  !  N      - INTEGER.
!  !           On entry,  N  specifies the number  of columns of the matrix
!  !           op( B ) and the number of columns of the matrix C. N must be
!  !           at least zero.
!  !           Unchanged on exit.
!  !
!  !  K      - INTEGER.
!  !           On entry,  K  specifies  the number of columns of the matrix
!  !           op( A ) and the number of rows of the matrix op( B ). K must
!  !           be at least  zero.
!  !           Unchanged on exit.
!  !
!  !  ALPHA  - DOUBLE PRECISION.
!  !           On entry, ALPHA specifies the scalar alpha.
!  !           Unchanged on exit.
!  !
!  !  A      - DOUBLE PRECISION array of DIMENSION ( LDA, ka ), where ka is
!  !           k  when  TRANSA = 'N' or 'n',  and is  m  otherwise.
!  !           Before entry with  TRANSA = 'N' or 'n',  the leading  m by k
!  !           part of the array  A  must contain the matrix  A,  otherwise
!  !           the leading  k by m  part of the array  A  must contain  the
!  !           matrix A.
!  !           Unchanged on exit.
!  !
!  !  LDA    - INTEGER.
!  !           On entry, LDA specifies the first dimension of A as declared
!  !           in the calling (sub) program. When  TRANSA = 'N' or 'n' then
!  !           LDA must be at least  max( 1, m ), otherwise  LDA must be at
!  !           least  max( 1, k ).
!  !           Unchanged on exit.
!  !
!  !  B      - DOUBLE PRECISION array of DIMENSION ( LDB, kb ), where kb is
!  !           n  when  TRANSB = 'N' or 'n',  and is  k  otherwise.
!  !           Before entry with  TRANSB = 'N' or 'n',  the leading  k by n
!  !           part of the array  B  must contain the matrix  B,  otherwise
!  !           the leading  n by k  part of the array  B  must contain  the
!  !           matrix B.
!  !           Unchanged on exit.
!  !
!  !  LDB    - INTEGER.
!  !  LDB    - INTEGER.
!  !           On entry, LDB specifies the first dimension of B as declared
!  !           in the calling (sub) program. When  TRANSB = 'N' or 'n' then
!  !           LDB must be at least  max( 1, k ), otherwise  LDB must be at
!  !           least  max( 1, n ).
!  !           Unchanged on exit.
!  !
!  !  BETA   - DOUBLE PRECISION.
!  !           On entry,  BETA  specifies the scalar  beta.  When  BETA  is
!  !           supplied as zero then C need not be set on input.
!  !           Unchanged on exit.
!  !
!  !  C      - DOUBLE PRECISION array of DIMENSION ( LDC, n ).
!  !           Before entry, the leading  m by n  part of the array  C must
!  !           contain the matrix  C,  except when  beta  is zero, in which
!  !           case C need not be set on entry.
!  !           On exit, the array  C  is overwritten by the  m by n  matrix
!  !           ( alpha*op( A )*op( B ) + beta*C ).
!  !
!  !  LDC    - INTEGER.
!  !           On entry, LDC specifies the first dimension of C as declared
!  !           in  the  calling  (sub)  program.   LDC  must  be  at  least
!  !           max( 1, m ).
!  !           Unchanged on exit.
!  !
!  !
!  !  Level 3 Blas routine.
!  !
!  !  -- Written on 8-February-1989.
!  !     Jack Dongarra, Argonne National Laboratory.
!  !     Iain Duff, AERE Harwell.
!  !     Jeremy Du Croz, Numerical Algorithms Group Ltd.
!  !     Sven Hammarling, Numerical Algorithms Group Ltd.
!  !
!  !
!  ! local variables
!  logical :: NOTA, NOTB
!  integer :: I, INFO, J, L, NCOLA, NROWA, NROWB
!  real(chm_real):: temp
!
!  logical :: LSAME_mn
!  ! external functions:  LSAME_mn,XERBLA_mn
!  ! intrinsic functions: MAX
!
!  ! Set NOTA and NOTB as true if A and B respectively are not
!  ! transposed and set NROWA, NCOLA and NROWB as the number of rows
!  ! and columns of A  and the number of rows of B respectively.
!  NOTA  = LSAME_mn( TRANSA, 'N' )
!  NOTB  = LSAME_mn( TRANSB, 'N' )
!  IF( NOTA )THEN
!     NROWA = M
!     NCOLA = K
!  ELSE
!     NROWA = K
!     NCOLA = M
!  END IF
!  IF( NOTB )THEN
!     NROWB = K
!  ELSE
!     NROWB = N
!  END IF
!  ! Test the input parameters.
!  INFO = 0
!  if(      ( .NOT.NOTA                 ).AND. &
!           ( .NOT.LSAME_mn( TRANSA, 'C' ) ).AND. &
!           ( .NOT.LSAME_mn( TRANSA, 'T' ) )      ) then
!     INFO = 1
!  else if( ( .NOT.NOTB                 ).AND. &
!           ( .NOT.LSAME_mn( TRANSB, 'C' ) ).AND. &
!           ( .NOT.LSAME_mn( TRANSB, 'T' ) )      ) then
!     INFO = 2
!  else if( M  .LT.0               ) then
!     INFO = 3
!  else if( N  .LT.0               ) then
!     INFO = 4
!  else if( K  .LT.0               ) then
!     INFO = 5
!  else if( LDA.LT.MAX( 1, NROWA ) ) then
!     INFO = 8
!  else if( LDB.LT.MAX( 1, NROWB ) ) then
!     INFO = 10
!  else if( LDC.LT.MAX( 1, M     ) ) then
!     INFO = 13
!  end if
!  if( INFO.ne.0 ) then
!     CALL XERBLA_mn( 'dgemm ', INFO )
!     return
!  end if
!  ! Quick return if possible.
!  if( ( M.EQ.0 ).OR.( N.EQ.0 ).OR.  &
!      ( ( ( ALPHA.EQ.ZERO ).OR.( K.EQ.0 ) ).AND.( BETA.EQ.ONE ) ) ) return
!  ! And if  alpha.eq.zero.
!  if( ALPHA.eq.ZERO ) then
!     if( BETA.eq.ZERO )then
!        C(1:M,1:N) = ZERO
!     else
!        C(1:M,1:N) = BETA*C(1:M,1:N)
!     end if
!     return
!  end if
!  ! Start the operations: bring if/then/else/end if out of the do-loops.
!  if( NOTB ) then
!     if( NOTA ) then
!        ! Form  C := alpha*A*B + beta*C.
!        if(BETA.eq.ZERO) then
!           if(alpha.eq.one) then
!              do J = 1, N
!                 c(1:m,j) = b(1,j)*a(1:m,1)
!                 do L = 2, K
!                    C(1:M,J) = C(1:M,J) + b(l,j)*A(1:M,L)
!                 end do
!              end do
!           else
!              do J = 1, N
!                 temp = alpha*b(1,j)
!                 c(1:m,j) = temp*a(1:m,1)
!                 do L = 2, K
!                    temp = ALPHA*B(L,J)
!                    C(1:M,J) = C(1:M,J) + temp*A(1:M,L)
!                 end do
!              end do
!           end if
!        else
!           if(alpha.eq.one) then
!              do J = 1, N
!                 C(1:M,J) = BETA*C(1:M,J)+b(1,j)*a(1:m,1)
!                 do L = 2, K
!                    C(1:M,J) = C(1:M,J) + b(l,j)*A(1:M,L)
!                 end do
!              end do
!           else
!              do J = 1, N
!                 temp = alpha*b(1,j)
!                 C(1:M,J) = BETA*C(1:M,J)+temp*a(1:m,1)
!                 do L = 2, K
!                    temp = ALPHA*B(L,J)
!                    C(1:M,J) = C(1:M,J) + temp*A(1:M,L)
!                 end do
!              end do
!           end if
!        end if
!     else
!        ! Form  C := alpha*A'*B + beta*C
!        if(BETA.eq.ZERO) then
!           if(alpha.eq.one) then
!              do J = 1, N
!                 do I = 1, M
!                    C(I,J) = DOT_PRODUCT(A(1:K,I),B(1:K,J))
!                 end do
!              end do
!           else
!              do J = 1, N
!                 do I = 1, M
!                    C(I,J) = ALPHA*DOT_PRODUCT(A(1:K,I),B(1:K,J))
!                 end do
!              end do
!           end if
!        else
!           if(alpha.eq.one) then
!              do J = 1, N
!                 do I = 1, M
!                    C(I,J) = DOT_PRODUCT(A(1:K,I),B(1:K,J)) + BETA*C(I,J)
!                 end do
!              end do
!           else
!              do J = 1, N
!                 do I = 1, M
!                    C(I,J) = ALPHA*DOT_PRODUCT(A(1:K,I),B(1:K,J)) + BETA*C(I,J)
!                 end do
!              end do
!           end if
!        end if
!     end if
!  else
!     if( NOTA ) then
!        ! Form  C := alpha*A*B' + beta*C
!        if(BETA.eq.ZERO) then
!           if(alpha.eq.one) then
!              ! for l=1
!              do j=1,n
!                 c(1:m,j)=b(j,1)*a(1:m,1)
!              end do
!              ! for l>2
!              do l=2,k
!                 do j=1,n
!                    c(1:m,j)=c(1:m,j)+b(j,l)*a(1:m,l)
!                 end do
!              end do
!           else
!              ! for l=1
!              do j=1,n
!                 c(1:m,j)=alpha*b(j,1)*a(1:m,1)
!              end do                 
!              ! for l>2
!              do l=2,k
!                 do j=1,n
!                    c(1:m,j)=c(1:m,j)+alpha*b(j,l)*a(1:m,l)
!                 end do
!              end do
!           end if
!        else
!           if(alpha.eq.one) then
!              ! for l=1
!              do j=1,n
!                 c(1:m,j)=beta*c(1:m,j)+b(j,1)*a(1:m,1)
!              end do
!              ! for l>2
!              do l=2,k
!                 do j=1,n
!                    c(1:m,j)=c(1:m,j)+b(j,l)*a(1:m,l)
!                 end do
!              end do
!           else
!              ! for l=1
!              do j=1,n
!                 c(1:m,j)=beta*c(1:m,j)+alpha*b(j,1)*a(1:m,1)
!              end do
!              ! for l>2
!              do l=2,k
!                 do j=1,n
!                    c(1:m,j)=c(i:m,j)+alpha*b(j,l)*a(1:m,l)
!                 end do
!              end do
!           end if
!        end if
!     else
!        ! Form  C := alpha*A'*B' + beta*C
!        if(BETA.eq.ZERO) then
!           if(alpha.eq.one) then
!              do J = 1, N
!                 do I = 1, M
!                    C(I,J) = DOT_PRODUCT(A(1:K,I),B(J,1:K))
!                 end do
!              end do
!           else
!              do J = 1, N
!                 do I = 1, M
!                    C(I,J) = ALPHA*DOT_PRODUCT(A(1:K,I),B(J,1:K))
!                 end do
!              end do
!           end if
!        else
!           if(alpha.eq.one) then
!              do J = 1, N
!                 do I = 1, M
!                    C(I,J) = DOT_PRODUCT(A(1:K,I),B(J,1:K)) + BETA*C(I,J)
!                 end do
!              end do
!           else
!              do J = 1, N
!                 do I = 1, M
!                    C(I,J) = ALPHA*DOT_PRODUCT(A(1:K,I),B(J,1:K)) + BETA*C(I,J)
!                 end do
!              end do
!           end if
!        end if
!     end if
!  end if
!
!  return
!  end subroutine dgemm_mn
!  !******************************************************************


  subroutine dgemv_mn(TRANS,M,N,ALPHA,A,LDA,X,INCX,BETA,Y,INCY)
  !
  use chm_kinds
  implicit none
  !
  real(chm_real) :: ALPHA,BETA,A(LDA,*),X(*),Y(*)
  integer :: INCX, INCY, LDA, M, N
  character(len=1) :: TRANS
  !
  !  Purpose
  !  DGEMV  performs one of the matrix-vector operations
  !     y := alpha*A*x + beta*y,   or   y := alpha*A'*x + beta*y,
  !  where alpha and beta are scalars, x and y are vectors and A is an m by n
  !  matrix.
  !
  !  Parameters
  !  TRANS  - CHARACTER*1.
  !           On entry, TRANS specifies the operation to be performed as
  !           follows:
  !              TRANS = 'N' or 'n'   y := alpha*A*x + beta*y.
  !              TRANS = 'T' or 't'   y := alpha*A'*x + beta*y.
  !              TRANS = 'C' or 'c'   y := alpha*A'*x + beta*y.
  !           Unchanged on exit.
  !  M      - INTEGER.
  !           On entry, M specifies the number of rows of the matrix A.
  !           M must be at least zero.
  !           Unchanged on exit.
  !  N      - INTEGER.
  !           On entry, N specifies the number of columns of the matrix A.
  !           N must be at least zero.
  !           Unchanged on exit.
  !  ALPHA  - DOUBLE PRECISION.
  !           On entry, ALPHA specifies the scalar alpha.
  !           Unchanged on exit.
  !  A      - DOUBLE PRECISION array of DIMENSION ( LDA, n ).
  !           Before entry, the leading m by n part of the array A must
  !           contain the matrix of coefficients.
  !           Unchanged on exit.
  !  LDA    - INTEGER.
  !           On entry, LDA specifies the first dimension of A as declared
  !           in the calling (sub) program. LDA must be at least
  !           max( 1, m ).
  !           Unchanged on exit.
  !  X      - DOUBLE PRECISION array of DIMENSION at least
  !           ( 1 + ( n - 1 )*abs( INCX ) ) when TRANS = 'N' or 'n'
  !           and at least
  !           ( 1 + ( m - 1 )*abs( INCX ) ) otherwise.
  !           Before entry, the incremented array X must contain the
  !           vector x.
  !           Unchanged on exit.
  !  INCX   - INTEGER.
  !           On entry, INCX specifies the increment for the elements of
  !           X. INCX must not be zero.
  !           Unchanged on exit.
  !  BETA   - DOUBLE PRECISION.
  !           On entry, BETA specifies the scalar beta. When BETA is
  !           supplied as zero then Y need not be set on input.
  !           Unchanged on exit.
  !  Y      - DOUBLE PRECISION array of DIMENSION at least
  !           ( 1 + ( m - 1 )*abs( INCY ) ) when TRANS = 'N' or 'n'
  !           and at least
  !           ( 1 + ( n - 1 )*abs( INCY ) ) otherwise.
  !           Before entry with BETA non-zero, the incremented array Y
  !           must contain the vector y. On exit, Y is overwritten by the
  !           updated vector y.
  !  INCY   - INTEGER.
  !           On entry, INCY specifies the increment for the elements of
  !           Y. INCY must not be zero.
  !           Unchanged on exit.
  !
  !  Level 2 Blas routine.
  !
  !  -- Written on 22-October-1986.
  !     Jack Dongarra, Argonne National Lab.
  !     Jeremy Du Croz, Nag Central Office.
  !     Sven Hammarling, Nag Central Office.
  !     Richard Hanson, Sandia National Labs.
  !
  ! local variables
  logical :: q_trans
  integer :: I, INFO, IX, IY, J, JX, JY, KX, KY, LENX, LENY
  real(chm_real):: TEMP
  real(chm_real),parameter :: zero=0.0d+0, one=1.0d+0

  ! external functions: LSAME_mn,XERBLA_mn
  ! intrinsic functions: MAX
  logical :: LSAME_mn
  real(chm_real):: ddot_mn ! external function


  info = 0
  if ( .not.LSAME_mn( TRANS, 'N' ) .and.       &
       .not.LSAME_mn( TRANS, 'T' ) .and.       &
       .not.LSAME_mn( TRANS, 'C' )      ) then
     info = 1
  else if( m.lt.0 ) then
     info = 2
  else if( n.lt.0 ) then
     info = 3
  else if( LDA.lt.MAX(1,m) ) then
     info = 6
  else if( incx.eq.0 ) then
     info = 8
  else if( incy.eq.0 ) then
     info = 11
  end if
  if( info.ne.0 ) then
     call xerbla_mn( 'dgemv ', info )
     return
  end if

  ! Quick return if possible.
  if( (m.eq.0) .or. (n.eq.0) .or.    &
      ((alpha.eq.zero).and.(beta.eq.one)) ) return

  q_trans = lsame_mn( trans, 'N' )

  ! Set LENX and LENY, the lengths of the vectors x and y, and set up the start
  ! points in  X  and  Y.
  if( q_trans ) then   ! lsame_mn( trans, 'N' )
     lenx = n
     leny = m
  else
     lenx = m
     leny = n
  end if
  if( incx.gt.0 ) then
     kx = 1
  else
     kx = 1 - (lenx-1)*incx
  end if
  if( incy.gt.0 ) then
     ky = 1
  else
     ky = 1 - (leny-1)*incy
  end if

  ! Start the operations. In this version the elements of A are
  ! accessed sequentially with one pass through A.
  !
  ! First form  y := beta*y.
  if( beta.ne.one ) then
     if( incy.eq.1 ) then
        if( beta.eq.zero ) then
           y(1:leny) = zero
        else
           y(1:leny) = beta*y(1:leny)
        end if
     else 
        iy = ky
        if( beta.eq.zero ) then
           do i = 1, leny
              y(iy) = zero
              iy    = iy + incy
           end do
        else
           do i = 1, leny
              y(iy) = beta*y(iy)
              iy    = iy + incy
           end do
        end if
     end if
  end if

  ! quick return
  if( alpha.eq.zero ) return

  if( q_trans ) then  ! lsame_mn( trans, 'N' )
     ! Form  y := alpha*A*x + y.
     jx = kx
     if( incy.eq.1 ) then
        do j = 1, n
           if( x(jx).ne.zero ) then
              temp = alpha*x(jx)
              y(1:m)=y(1:m) + beta*a(1:m,j)
           end if
           jx = jx + incx
        end do
     else
        do j = 1, n
           if( x(jx).ne.zero ) then
              temp = alpha*x(jx)
              iy   = ky
              do i = 1, m
                 y(iy) = y(iy) + temp*a(i,j)
                 iy    = iy    + incy
              end do
           end if
           jx = jx + incx
        end do
     end if
  else
     ! Form  y := alpha*A'*x + y.
     jy = ky
     if( incx.eq.1 ) then
        do j = 1, n
           y(jy)= y(jy) + alpha*ddot_mn(m,a(1:m,j),1,x(1:m),1)  ! DOT_PRODUCT(a(1:m,j),x(1:m))
           jy   = jy    + incy
        end do
     else
        do j = 1, n
           temp = zero
           ix   = kx
           do i = 1, m
              temp = temp + a(i,j)*x(ix)
              ix   = ix   + incx
           end do
           y(jy) = y(jy) + alpha*temp
           jy    = jy    + incy
        end do
     end if
  end if
  return
  end subroutine dgemv_mn
  !******************************************************************


  subroutine dger_mn(M,N,ALPHA,X,INCX,Y,INCY,A,LDA)
  !
  use chm_kinds
  implicit none
  integer :: INCX, INCY, LDA, M, N
  real(chm_real):: ALPHA,A(LDA,*),X(*),Y(*)
  !
  !  Purpose
  !  DGER   performs the rank 1 operation
  !     A := alpha*x*y' + A,
  !  where alpha is a scalar, x is an m element vector, y is an n element
  !  vector and A is an m by n matrix.
  !
  !  Parameters
  !  M      - INTEGER.
  !           On entry, M specifies the number of rows of the matrix A.
  !           M must be at least zero.
  !           Unchanged on exit.
  !  N      - INTEGER.
  !           On entry, N specifies the number of columns of the matrix A.
  !           N must be at least zero.
  !           Unchanged on exit.
  !  ALPHA  - DOUBLE PRECISION.
  !           On entry, ALPHA specifies the scalar alpha.
  !           Unchanged on exit.
  !  X      - DOUBLE PRECISION array of dimension at least
  !           ( 1 + ( m - 1 )*abs( INCX ) ).
  !           Before entry, the incremented array X must contain the m
  !           element vector x.
  !           Unchanged on exit.
  !  INCX   - INTEGER.
  !           On entry, INCX specifies the increment for the elements of
  !           X. INCX must not be zero.
  !           Unchanged on exit.
  !  Y      - DOUBLE PRECISION array of dimension at least
  !           ( 1 + ( n - 1 )*abs( INCY ) ).
  !           Before entry, the incremented array Y must contain the n
  !           element vector y.
  !           Unchanged on exit.
  !  INCY   - INTEGER.
  !           On entry, INCY specifies the increment for the elements of
  !           Y. INCY must not be zero.
  !           Unchanged on exit.
  !  A      - DOUBLE PRECISION array of DIMENSION ( LDA, n ).
  !           Before entry, the leading m by n part of the array A must
  !           contain the matrix of coefficients. On exit, A is
  !           overwritten by the updated matrix.
  !  LDA    - INTEGER.
  !           On entry, LDA specifies the first dimension of A as declared
  !           in the calling (sub) program. LDA must be at least
  !           max( 1, m ).
  !           Unchanged on exit.
  !
  !  Level 2 Blas routine.
  !
  !  -- Written on 22-October-1986.
  !     Jack Dongarra, Argonne National Lab.
  !     Jeremy Du Croz, Nag Central Office.
  !     Sven Hammarling, Nag Central Office.
  !     Richard Hanson, Sandia National Labs.
  !
  ! local variables
  integer :: I, INFO, IX, J, JY, KX
  real(chm_real):: TEMP
  real(chm_real),parameter :: zero=0.0d+0

  ! external functions: XERBLA
  ! intrinsic functions: MAX

  info = 0
  if ( m.lt.0 ) then
     info = 1
  else if( n.lt.0 ) then
     info = 2
  else if( incx.eq.0 ) then
     info = 5
  else if( incy.eq.0 ) then
     info = 7
  else if( LDA.lt.MAX(1,m) ) then
     info = 9
  end if
  if( info.ne.0 ) then
     call xerbla_mn( 'dger  ', info )
     return
  end if

  ! Quick return if possible.
  if( (m.eq.0).or.(n.eq.0).or.(alpha.eq.zero) ) return

  ! Start the operations. In this version the elements of A are
  ! accessed sequentially with one pass through A.
  if( incy.gt.0 ) then
     jy = 1
  else
     jy = 1 - (n-1)*incy
  end if
  if( incx.eq.1 ) then
     do j = 1, n
        if( y(jy).ne.zero ) then
           temp     = alpha*y(jy)
           a(1:m,j) = a(1:m,j) + x(1:m)*temp
        end if
        jy = jy + incy
     end do
  else
     if( incx.gt.0 ) then
        kx = 1
     else
        kx = 1 - (m-1)*incx
     end if
     do j = 1, n
        if( y(jy).ne.zero ) then
           temp = alpha*y(jy)
           ix   = kx
           do i = 1, m
              a(i,j) = a(i,j) + x(ix)*temp
              ix     = ix + incx
           end do
        end if
        jy = jy + incy
     end do
  end if
  return
  end subroutine dger_mn
  !******************************************************************
 

  subroutine dspr_mn( UPLO, N, ALPHA, X, INCX, AP )
  use chm_kinds
  implicit none

  real(chm_real):: alpha,ap(*),x(*)
  integer       :: incx,n
  character(len=1):: uplo
  !
  !  Purpose
  !  DSPR    performs the symmetric rank 1 operation
  !
  !     A := alpha*x*x' + A,
  !
  !  where alpha is a real scalar, x is an n element vector and A is an
  !  n by n symmetric matrix, supplied in packed form.
  !
  !  Parameters
  !  UPLO   - CHARACTER*1.
  !           On entry, UPLO specifies whether the upper or lower
  !           triangular part of the matrix A is supplied in the packed
  !           array AP as follows:
  !              UPLO = 'U' or 'u'   The upper triangular part of A is
  !                                  supplied in AP.
  !
  !              UPLO = 'L' or 'l'   The lower triangular part of A is
  !                                  supplied in AP.
  !           Unchanged on exit.
  !  N      - INTEGER.
  !           On entry, N specifies the order of the matrix A.
  !           N must be at least zero.
  !           Unchanged on exit.
  !  ALPHA  - DOUBLE PRECISION.
  !           On entry, ALPHA specifies the scalar alpha.
  !           Unchanged on exit.
  !  X      - DOUBLE PRECISION array of dimension at least
  !           ( 1 + ( n - 1 )*abs( INCX ) ).
  !           Before entry, the incremented array X must contain the n
  !           element vector x.
  !           Unchanged on exit.
  !  INCX   - INTEGER.
  !           On entry, INCX specifies the increment for the elements of
  !           X. INCX must not be zero.
  !           Unchanged on exit.
  !  AP     - DOUBLE PRECISION array of DIMENSION at least
  !           ( ( n*( n + 1 ) )/2 ).
  !           Before entry with  UPLO = 'U' or 'u', the array AP must
  !           contain the upper triangular part of the symmetric matrix
  !           packed sequentially, column by column, so that AP( 1 )
  !           contains a( 1, 1 ), AP( 2 ) and AP( 3 ) contain a( 1, 2 )
  !           and a( 2, 2 ) respectively, and so on. On exit, the array
  !           AP is overwritten by the upper triangular part of the
  !           updated matrix.
  !           Before entry with UPLO = 'L' or 'l', the array AP must
  !           contain the lower triangular part of the symmetric matrix
  !           packed sequentially, column by column, so that AP( 1 )
  !           contains a( 1, 1 ), AP( 2 ) and AP( 3 ) contain a( 2, 1 )
  !           and a( 3, 1 ) respectively, and so on. On exit, the array
  !           AP is overwritten by the lower triangular part of the
  !           updated matrix.
  !
  !  Level 2 Blas routine.
  !
  !  -- Written on 22-October-1986.
  !     Jack Dongarra, Argonne National Lab.
  !     Jeremy Du Croz, Nag Central Office.
  !     Sven Hammarling, Nag Central Office.
  !     Richard Hanson, Sandia National Labs.
  !
  !
  real(chm_real),parameter:: zero=0.0D+0 
  real(chm_real) ::  temp
  integer        ::  i, info, ix, j, jx, k, kk, kx
  logical :: lsame_mn
  ! external routines: lsame, xerbla
  !
  ! Test the input parameters.
  info = 0
  if     ( .not.lsame_mn( uplo, 'U' ) .and. .not.lsame_mn( uplo, 'L' )      ) then
     info = 1
  else if( n.lt.0 ) then
     info = 2
  else if( incx.eq.0 ) then
     info = 5
  end if

  if( info.ne.0 ) then
     call xerbla_mn( 'dspr  ', info )
     return
  end if

  ! Quick return if possible.
  if( ( n.eq.0 ).or.( alpha.eq.zero ) ) return

  ! Set the start point in X if the increment is not unity.
  if( incx.le.0 ) then
     kx = 1 - ( n - 1 )*incx
  else if( incx.ne.1 ) then
     kx = 1
  end if

  ! Start the operations. In this version the elements of the array AP
  ! are accessed sequentially with one pass through AP.
  kk = 1
  if( lsame_mn( uplo, 'U' ) ) then
     ! Form  A  when upper triangle is stored in AP.
     if( incx.eq.1 ) then
        do j = 1, n
           if( x( j ).ne.zero ) then
              temp = alpha*x( j )
              ap(kk:kk+j-1)=ap(kk:kk+j-1)+x(1:j)*temp
           end if
           kk = kk + j
        end do
     else
        jx = kx
        do j = 1, n
           if( x( jx ).ne.zero ) then
              temp = alpha*x( jx )
              ix   = kx
              do k = kk, kk + j - 1
                 ap( k ) = ap( k ) + x( ix )*temp
                 ix      = ix      + incx
              end do
           end if
           jx = jx + incx
           kk = kk + j
        end do
     end if
  else
     ! Form  A  when lower triangle is stored in AP.
     if( incx.eq.1 ) then
        do j = 1, n
           if( x( j ).ne.zero ) then
              temp = alpha*x( j )
              ap(kk:kk+n-j)=ap(kk:kk+n-j)+x(j:n)*temp
           end if
           kk = kk + n - j + 1
        end do
     else
        jx = kx
        do j = 1, n
           if( x( jx ).ne.zero ) then
              temp = alpha*x( jx )
              ix   = jx
              do k = kk, kk + n - j
                 ap(k) = ap(k) + x(ix)*temp
                 ix    = ix    + incx
              end do
           end if
           jx = jx + incx
           kk = kk + n - j + 1
        end do
     end if
  end if
  !
  RETURN
  end subroutine dspr_mn
  !******************************************************************


  subroutine dswap_mn(n,dx,incx,dy,incy)
  !
  ! interchanges two vectors.
  ! uses unrolled loops for increments equal one.
  ! jack dongarra, linpack, 3/11/78.
  !
  use chm_kinds
  implicit none
  real(chm_real) :: dx(*),dy(*),dtemp
  integer :: i,incx,incy,ix,iy,m,n

  if(n.le.0) return
  if(incx.eq.1 .and. incy.eq.1) then
     ! code for both increments equal to 1
     m = mod(n,3)
     if(m.ne.0) then
        do i = 1,m
           dtemp = dx(i)
           dx(i) = dy(i)
           dy(i) = dtemp
        end do
        if( n .lt. 3 ) return
     end if
     do i = m+1,n,3
       dtemp   = dx(i)
       dx(i)   = dy(i)
       dy(i)   = dtemp

       dtemp   = dx(i+1)
       dx(i+1) = dy(i+1)
       dy(i+1) = dtemp

       dtemp   = dx(i+2)
       dx(i+2) = dy(i +2)
       dy(i+2) = dtemp
     end do
  else
     ! code for unequal increments or equal increments not equal to 1
     ix = 1
     iy = 1
     if(incx.lt.0) ix = (-n+1)*incx + 1
     if(incy.lt.0) iy = (-n+1)*incy + 1
     do i = 1,n
        dtemp  = dx(ix)
        dx(ix) = dy(iy)
        dy(iy) = dtemp
        ix     = ix + incx
        iy     = iy + incy
     end do
  end if
  return
  end subroutine dswap_mn
  !******************************************************************


  subroutine dcopy_mn(n,dx,incx,dy,incy)
  !
  ! copies a vector, x, to a vector, y.
  ! uses unrolled loops for increments equal to one.
  ! jack dongarra, linpack, 3/11/78.
  !
  use chm_kinds
  implicit none
  real(chm_real) :: dx(*),dy(*)
  integer :: i,incx,incy,ix,iy,m,n

  if(n.le.0) return
  if(incx.eq.1 .and. incy.eq.1) then
     ! code for both increments equal to 1
     m = mod(n,7)
     if(m.ne.0) then
        dy(1:m) = dx(1:m)
        if( n .lt. 7 ) return
     end if
     do i = m+1,n,7
        dy(i)   = dx(i)
        dy(i+1) = dx(i+1)
        dy(i+2) = dx(i+2)
        dy(i+3) = dx(i+3)
        dy(i+4) = dx(i+4)
        dy(i+5) = dx(i+5)
        dy(i+6) = dx(i+6)
     end do
  else
     ! code for unequal increments or equal increments not equal to 1
     ix = 1
     iy = 1
     if(incx.lt.0) ix = (-n+1)*incx + 1
     if(incy.lt.0) iy = (-n+1)*incy + 1
     do i = 1,n
        dy(iy) = dx(ix)
        ix = ix + incx
        iy = iy + incy
     end do
  end if
  return
  end subroutine dcopy_mn
  !******************************************************************


  subroutine daxpy_mn(n,da,dx,incx,dy,incy)
  !
  ! constant times a vector plus a vector.
  ! uses unrolled loops for increments equal to one.
  ! jack dongarra, linpack, 3/11/78.
  !
  use chm_kinds
  implicit none
  real(chm_real) :: dx(*),dy(*),da
  integer :: i,incx,incy,ix,iy,m,mp1,n

  if(n.le.0)return
  if (da .eq. 0.0d0) return
  if (incx.eq.1 .and. incy.eq.1) then
     ! code for both increments equal to 1
     m = mod(n,4)
     if(m.ne.0) then
        dy(1:m) = dy(1:m) + da*dx(1:m)
        if( n .lt. 4 ) return
     end if
     do i = m+1,n,4
        dy(i)   = dy(i)   + da*dx(i)
        dy(i+1) = dy(i+1) + da*dx(i+1)
        dy(i+2) = dy(i+2) + da*dx(i+2)
        dy(i+3) = dy(i+3) + da*dx(i+3)
     end do
  else
     ! code for unequal increments or equal increments not equal to 1
     ix = 1
     iy = 1
     if(incx.lt.0) ix = (-n+1)*incx + 1
     if(incy.lt.0) iy = (-n+1)*incy + 1
     do i = 1,n
        dy(iy) = dy(iy) + da*dx(ix)
        ix = ix + incx
        iy = iy + incy
     end do
  end if
  return
  end subroutine daxpy_mn
  !******************************************************************


  subroutine drot_mn(n,dx,incx,dy,incy,c,s)
  !
  ! applies a plane rotation.
  ! jack dongarra, linpack, 3/11/78.
  !
  use chm_kinds
  implicit none
  real(chm_real):: dx(*),dy(*),dtemp,c,s
  integer i,incx,incy,ix,iy,n

  if(n.le.0)return
  if(incx.eq.1 .and. incy.eq.1) then
     ! code for both increments equal to 1
     do i = 1,n
        dtemp = c*dx(i) + s*dy(i)
        dy(i) = c*dy(i) - s*dx(i)
        dx(i) = dtemp
     end do
  else
     ! code for unequal increments or equal increments not equal to 1
     ix = 1
     iy = 1
     if(incx.lt.0) ix = (-n+1)*incx + 1
     if(incy.lt.0) iy = (-n+1)*incy + 1
     do i = 1,n
        dtemp  = c*dx(ix) + s*dy(iy)
        dy(iy) = c*dy(iy) - s*dx(ix)
        dx(ix) = dtemp
        ix     = ix + incx
        iy     = iy + incy
     end do
  end if
  return
  end subroutine drot_mn
  !******************************************************************
 

  subroutine dscal_mn(n,da,dx,incx)
  !
  ! scales a vector by a constant.
  ! uses unrolled loops for increment equal to one.
  ! jack dongarra, linpack, 3/11/78.
  ! modified 3/93 to return if incx .le. 0.
  !
  use chm_kinds
  implicit none
  real(chm_real):: da,dx(*)
  integer :: i,incx,m,n,nincx

  if( n.le.0 .or. incx.le.0 )return
  if(incx.eq.1) then
     ! code for increment equal to 1
     m = mod(n,5)
     if( m .ne. 0 ) then
         dx(1:m) = da*dx(1:m)
         if( n .lt. 5 ) return
     end if
     do i = m+1,n,5
        dx(i)   = da*dx(i)
        dx(i+1) = da*dx(i+1)
        dx(i+2) = da*dx(i+2)
        dx(i+3) = da*dx(i+3)
        dx(i+4) = da*dx(i+4) 
     end do
  else
     ! code for increment not equal to 1
     nincx = n*incx
     do i = 1,nincx,incx
        dx(i) = da*dx(i)
     end do 
  end if
  return
  end subroutine dscal_mn
  !******************************************************************


  subroutine xerbla_mn( SRNAME, INFO )
  !
  !  -- LAPACK auxiliary routine (preliminary version) --
  !     Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
  !     Courant Institute, Argonne National Lab, and Rice University
  !     February 29, 1992
  !
  use chm_kinds
  implicit none
  character(LEN=6):: SRNAME
  integer::          INFO
  !
  !  Purpose
  !  XERBLA  is an error handler for the LAPACK routines.
  !  It is called by an LAPACK routine if an input parameter has an
  !  invalid value.  A message is printed and execution stops.
  !
  !  Installers may consider modifying the STOP statement in order to
  !  call system-specific exception-handling facilities.
  !
  !  Arguments
  !  SRNAME  (input) CHARACTER*6
  !          The name of the routine which called XERBLA.
  !  INFO    (input) INTEGER
  !          The position of the invalid parameter in the parameter list
  !          of the calling routine.

  !WRITE( *, FMT = 9999 )SRNAME, INFO
  WRITE( 6, FMT = 9999 )SRNAME, INFO

  return
 9999 FORMAT( ' ** On entry to ', A6, ' parameter number ', I2, ' had ','anillegal value' )
  end subroutine xerbla_mn
  !******************************************************************


  function ddot_mn(n,dx,incx,dy,incy) result(ddtot_return)
  !
  ! forms the dot product of two vectors.
  ! uses unrolled loops for increments equal to one.
  ! jack dongarra, linpack, 3/11/78.
  !
  use chm_kinds
  use number, only : zero, one
  implicit none
  real(chm_real):: dx(*),dy(*),dtemp
  real(chm_real):: ddtot_return
  integer i,incx,incy,ix,iy,m,n

  !ddot_mn  = zero
  ddtot_return = zero
  dtemp = zero
  if(n.le.0) return
  if(incx.eq.1 .and. incy.eq.1) then
     ! code for both increments equal to 1
     m = mod(n,5)
     if(m.ne.0) then
        dtemp=dtemp+DOT_PRODUCT(dx(1:m),dy(1:m))
        if( n .lt. 5 ) then
            !ddot_mn = dtemp
            ddtot_return = dtemp
            return
        end if
     end if
     do i = m+1,n,5
        dtemp = dtemp + dx(i  )*dy(i  ) + dx(i+1)*dy(i+1) +  dx(i+2)*dy(i+2) + &
                        dx(i+3)*dy(i+3) + dx(i+4)*dy(i+4)
     end do
  else
     ! code for unequal increments or equal increments not equal to 1
     ix = 1
     iy = 1
     if(incx.lt.0) ix = (-n+1)*incx + 1
     if(incy.lt.0) iy = (-n+1)*incy + 1
     do i = 1,n
        dtemp = dtemp + dx(ix)*dy(iy)
        ix    = ix + incx
        iy    = iy + incy
     end do
  end if
  !ddot_mn = dtemp
  ddtot_return = dtemp
  return
  end function ddot_mn
  !******************************************************************


  function ddot2d_mn(n,dx,dx2,incx,dy,dy2,incy,q_plusmin) result(ddtot_return)
  !
  ! Forms the two dot products of four vectors, based on ddot_mn routine.
  ! dot_product(dx(1:n),dy(1:n)) +/- dot_product(dx2(1:n),dy2(1:n))
  !
  use chm_kinds
  use number, only : zero, one
  implicit none
  real(chm_real):: dx(*),dx2(*),dy(*),dy2(*),dtemp
  logical       :: q_plusmin
  real(chm_real):: ddtot_return
  integer i,incx,incy,ix,iy,m,n

  ddtot_return = zero
  dtemp = zero
  if(n.le.0) return
  if(incx.eq.1 .and. incy.eq.1) then
     ! code for both increments equal to 1
     m = mod(n,5)
     if(q_plusmin) then
        ! Plus case: dot_product(dx(1:n),dy(1:n)) + dot_product(dx2(1:n),dy2(1:n))
        if(m.ne.0) then
           dtemp=dtemp+DOT_PRODUCT(dx(1:m),dy(1:m))+DOT_PRODUCT(dx2(1:m),dy2(1:m))
           if( n .lt. 5 ) then
               ddtot_return = dtemp
               return
           end if
        end if
        do i = m+1,n,5
           dtemp = dtemp +(dx(i  )*dy(i  )   + dx(i+1)*dy(i+1)   +  dx(i+2)*dy(i+2)   + &
                           dx(i+3)*dy(i+3)   + dx(i+4)*dy(i+4))                         &
                         +(dx2(i  )*dy2(i  ) + dx2(i+1)*dy2(i+1) +  dx2(i+2)*dy2(i+2) + &
                           dx2(i+3)*dy2(i+3) + dx2(i+4)*dy2(i+4))
        end do
     else
        ! Minus case: dot_product(dx(1:n),dy(1:n)) - dot_product(dx2(1:n),dy2(1:n))
        if(m.ne.0) then
           dtemp=dtemp+DOT_PRODUCT(dx(1:m),dy(1:m))-DOT_PRODUCT(dx2(1:m),dy2(1:m))
           if( n .lt. 5 ) then
               ddtot_return = dtemp
               return
           end if
        end if
        do i = m+1,n,5
           dtemp = dtemp +(dx(i  )*dy(i  )   + dx(i+1)*dy(i+1)   +  dx(i+2)*dy(i+2)   + &
                           dx(i+3)*dy(i+3)   + dx(i+4)*dy(i+4))                         &
                         -(dx2(i  )*dy2(i  ) + dx2(i+1)*dy2(i+1) +  dx2(i+2)*dy2(i+2) + &
                           dx2(i+3)*dy2(i+3) + dx2(i+4)*dy2(i+4))
        end do
     end if
  else
     ! code for unequal increments or equal increments not equal to 1
     ix = 1
     iy = 1
     if(incx.lt.0) ix = (-n+1)*incx + 1
     if(incy.lt.0) iy = (-n+1)*incy + 1
     if(q_plusmin) then
        ! Plus case: dot_product(dx(1:n),dy(1:n)) + dot_product(dx2(1:n),dy2(1:n))
        do i = 1,n
           dtemp = dtemp + dx(ix)*dy(iy) + dx2(ix)*dy2(iy)
           ix    = ix + incx
           iy    = iy + incy
        end do
     else
        ! Minus case: dot_product(dx(1:n),dy(1:n)) - dot_product(dx2(1:n),dy2(1:n))
        do i = 1,n
           dtemp = dtemp + dx(ix)*dy(iy) - dx2(ix)*dy2(iy)
           ix    = ix + incx
           iy    = iy + incy
        end do
     end if
  end if
  ddtot_return = dtemp
  return
  end function ddot2d_mn
  !******************************************************************


  function ddot1d_mn(n,dx,incx,dy,dy2,incy,q_plusmin) result(ddtot_return)
  !
  ! Forms the two dot products of four vectors, based on ddot_mn routine.
  ! dot_product(dx(1:n),dy(1:n)) +/- dot_product(dx(1:n),dy2(1:n))
  !
  use chm_kinds
  use number, only : zero, one
  implicit none
  real(chm_real):: dx(*),dy(*),dy2(*),dtemp
  logical       :: q_plusmin
  real(chm_real):: ddtot_return
  integer i,incx,incy,ix,iy,m,n

  ddtot_return = zero
  dtemp = zero
  if(n.le.0) return
  if(incx.eq.1 .and. incy.eq.1) then
     ! code for both increments equal to 1
     m = mod(n,5)
     if(q_plusmin) then
        ! Plus case: dot_product(dx(1:n),dy(1:n)) + dot_product(dx(1:n),dy2(1:n))
        if(m.ne.0) then
           dtemp=dtemp+DOT_PRODUCT(dx(1:m),dy(1:m))+DOT_PRODUCT(dx(1:m),dy2(1:m))
           if( n .lt. 5 ) then
               ddtot_return = dtemp
               return
           end if
        end if
        do i = m+1,n,5
           dtemp = dtemp +(dx(i  )*(dy(i  )+dy2(i  )) + dx(i+1)*(dy(i+1)+dy2(i+1)) + &
                           dx(i+2)*(dy(i+2)+dy2(i+2)) + dx(i+3)*(dy(i+3)+dy2(i+3)) + &
                           dx(i+4)*(dy(i+4)+dy2(i+4)))
        end do
     else
        ! Minus case: dot_product(dx(1:n),dy(1:n)) - dot_product(dx(1:n),dy2(1:n))
        if(m.ne.0) then
           dtemp=dtemp+DOT_PRODUCT(dx(1:m),dy(1:m))-DOT_PRODUCT(dx(1:m),dy2(1:m))
           if( n .lt. 5 ) then
               ddtot_return = dtemp
               return
           end if
        end if
        do i = m+1,n,5
           dtemp = dtemp +(dx(i  )*(dy(i  )-dy2(i  )) + dx(i+1)*(dy(i+1)-dy2(i+1)) + &
                           dx(i+2)*(dy(i+2)-dy2(i+2)) + dx(i+3)*(dy(i+3)-dy2(i+3)) + &
                           dx(i+4)*(dy(i+4)-dy2(i+4)))
        end do
     end if
  else
     ! code for unequal increments or equal increments not equal to 1
     ix = 1
     iy = 1
     if(incx.lt.0) ix = (-n+1)*incx + 1
     if(incy.lt.0) iy = (-n+1)*incy + 1
     if(q_plusmin) then
        ! Plus case: dot_product(dx(1:n),dy(1:n)) + dot_product(dx(1:n),dy2(1:n))
        do i = 1,n
           dtemp = dtemp + dx(ix)*(dy(iy) + dy2(iy))
           ix    = ix + incx
           iy    = iy + incy
        end do
     else
        ! Minus case: dot_product(dx(1:n),dy(1:n)) - dot_product(dx(1:n),dy2(1:n))
        do i = 1,n
           dtemp = dtemp + dx(ix)*(dy(iy) - dy2(iy))
           ix    = ix + incx
           iy    = iy + incy
        end do
     end if
  end if
  ddtot_return = dtemp
  return
  end function ddot1d_mn
  !******************************************************************


  function dnrm2_mn ( n, dx, incx) result(dnrm2_return)
  !
  ! euclidean norm of the n-vector stored in dx() with storage increment
  ! incx.
  ! if    n .le. 0 return with result = 0.
  ! if n .ge. 1 then incx must be .ge. 1
  !
  ! DNRM2_mn := sqrt( x'*x )
  use chm_kinds
  use number, only : zero, one
  implicit none
  integer :: incx, ix, n, icnt
  real(chm_real):: dx(*), sum, xmax, absxi
  real(chm_real):: dnrm2_return

  ! first check.
  if(n .le. 0 .or. incx.le.0) then
     !dnrm2_mn  = zero
     dnrm2_return = zero
  else if(n.eq.1) then
     !dnrm2_mn  = dabs(dx(1))
     dnrm2_return = dabs(dx(1))
  else
     xmax  = zero
     sum   = one
     do ix = 1, 1+(n-1)*incx, incx
        if(dx(ix).ne.zero) then
           absxi = dabs(dx(ix))
           if(xmax.lt.absxi) then
              sum = one + sum*(xmax/absxi)**2
              xmax= absxi
           else
              sum = sum + (absxi/xmax)**2
           end if
        end if
      end do
      !dnrm2_mn = xmax * dSQRT(sum)
      dnrm2_return = xmax * dSQRT(sum)
  end if
  return
  end function dnrm2_mn
  !******************************************************************


  integer function idamax_mn(n,dx,incx)
  !
  ! finds the index of element having max. absolute value.
  ! jack dongarra, linpack, 3/11/78.
  ! modified 3/93 to return if incx .le. 0.
  !
  use chm_kinds
  implicit none
  real(chm_real):: dx(*),dmax
  integer i,incx,ix,n

  idamax_mn = 0
  if( n.lt.1 .or. incx.le.0 ) return
  idamax_mn = 1
  if(n.eq.1) return
  if(incx.eq.1) then
     !        code for increment equal to 1
     dmax = dabs(dx(1))
     do i = 2,n
        if(dabs(dx(i)).gt.dmax) then
           idamax_mn = i
           dmax   = dabs(dx(i))
        end if
     end do
  else
     ! code for increment not equal to 1
     dmax = dabs(dx(1))
     ix   = 1 + incx
     do i = 2,n
        if(dabs(dx(ix)).gt.dmax) then
           idamax_mn = i
           dmax   = dabs(dx(ix))
        end if
        ix = ix + incx
     end do
  end if
  return
  end function idamax_mn
  !******************************************************************


  logical function lsame_mn( CA, CB )
  !
  !  -- LAPACK auxiliary routine (version 1.1) --
  !     Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
  !     Courant Institute, Argonne National Lab, and Rice University
  !     February 29, 1992
  !
  use chm_kinds
  implicit none
  character(len=1):: CA, CB
  !
  !  Purpose
  !  LSAME_mn returns .TRUE. if CA is the same letter as CB regardless of case.
  !
  !  Arguments
  !  CA      (input) CHARACTER*1
  !  CB      (input) CHARACTER*1
  !          CA and CB specify the single characters to be compared.
  !
  integer ::         INTA, INTB, ZCODE
  !
  !     Test if the characters are equal
  LSAME_mn = CA.EQ.CB
  IF( LSAME_mn )  RETURN

  !     Now test for equivalence if both characters are alphabetic.
  ZCODE = ICHAR( 'Z' )
  !
  !     Use 'Z' rather than 'A' so that ASCII can be detected on Prime
  !     machines, on which ICHAR returns a value with bit 8 set.
  !     ICHAR('A') on Prime machines returns 193 which is the same as
  !     ICHAR('A') on an EBCDIC machine.

  INTA = ICHAR( CA )
  INTB = ICHAR( CB )

  IF( ZCODE.EQ.90 .OR. ZCODE.EQ.122 ) THEN
  !        ASCII is assumed - ZCODE is the ASCII code of either lower or
  !        upper case 'Z'.
     IF( INTA.GE.97 .AND. INTA.LE.122 ) INTA = INTA - 32
     IF( INTB.GE.97 .AND. INTB.LE.122 ) INTB = INTB - 32
  ELSE IF( ZCODE.EQ.233 .OR. ZCODE.EQ.169 ) THEN
  !        EBCDIC is assumed - ZCODE is the EBCDIC code of either lower or
  !        upper case 'Z'.
     IF( INTA.GE.129 .AND. INTA.LE.137 .OR.  &
         INTA.GE.145 .AND. INTA.LE.153 .OR.  &
         INTA.GE.162 .AND. INTA.LE.169 ) INTA = INTA + 64
     IF( INTB.GE.129 .AND. INTB.LE.137 .OR.  &
         INTB.GE.145 .AND. INTB.LE.153 .OR.  &
         INTB.GE.162 .AND. INTB.LE.169 ) INTB = INTB + 64
  ELSE IF( ZCODE.EQ.218 .OR. ZCODE.EQ.250 ) THEN
  !        ASCII is assumed, on Prime machines - ZCODE is the ASCII code
  !        plus 128 of either lower or upper case 'Z'.
     IF( INTA.GE.225 .AND. INTA.LE.250 ) INTA = INTA - 32
     IF( INTB.GE.225 .AND. INTB.LE.250 ) INTB = INTB - 32
  END IF
  LSAME_mn = INTA.EQ.INTB
  return
  end function lsame_mn
  !******************************************************************
!=======================================================================!
!============ END BLAS routines ========================================!
!=======================================================================|


!=======================================================================!
!============ LAPACK routines ==========================================!
!=======================================================================|

  subroutine dspsv_mn( UPLO, N, NRHS, AP, IPIV, B, LDB, INFO )
  !
  !  -- LAPACK driver routine (version 1.1) --
  !     Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
  !     Courant Institute, Argonne National Lab, and Rice University
  !     March 31, 1993
  !
  use chm_kinds
  implicit none
  character::      UPLO
  integer  ::      INFO, LDB, N, NRHS, IPIV(*)
  real(chm_real):: AP(*), B(LDB,*)
  !
  !  Purpose
  !  DSPSV computes the solution to a real system of linear equations
  !     A * X = B,
  !  where A is an N-by-N symmetric matrix stored in packed format and X
  !  and B are N-by-NRHS matrices.
  !
  !  The diagonal pivoting method is used to factor A as
  !     A = U * D * U**T,  if UPLO = 'U', or
  !     A = L * D * L**T,  if UPLO = 'L',
  !  where U (or L) is a product of permutation and unit upper (lower)
  !  triangular matrices, D is symmetric and block diagonal with 1-by-1
  !  and 2-by-2 diagonal blocks.  The factored form of A is then used to
  !  solve the system of equations A * X = B.
  !
  !  Arguments
  !  UPLO    (input) CHARACTER*1
  !          = 'U':  Upper triangle of A is stored;
  !          = 'L':  Lower triangle of A is stored.
  !  N       (input) INTEGER
  !          The number of linear equations, i.e., the order of the
  !          matrix A.  N >= 0.
  !  NRHS    (input) INTEGER
  !          The number of right hand sides, i.e., the number of columns
  !          of the matrix B.  NRHS >= 0.
  !  AP      (input/output) DOUBLE PRECISION array, dimension (N*(N+1)/2)
  !          On entry, the upper or lower triangle of the symmetric matrix
  !          A, packed columnwise in a linear array.  The j-th column of A
  !          is stored in the array AP as follows:
  !          if UPLO = 'U', AP(i + (j-1)*j/2) = A(i,j) for 1<=i<=j;
  !          if UPLO = 'L', AP(i + (j-1)*(2n-j)/2) = A(i,j) for j<=i<=n.
  !          See below for further details.
  !          On exit, the block diagonal matrix D and the multipliers used
  !          to obtain the factor U or L from the factorization
  !          A = U*D*U**T or A = L*D*L**T as computed by DSPTRF, stored as
  !          a packed triangular matrix in the same storage format as A.
  !  IPIV    (output) INTEGER array, dimension (N)
  !          Details of the interchanges and the block structure of D, as
  !          determined by DSPTRF.  If IPIV(k) > 0, then rows and columns
  !          k and IPIV(k) were interchanged, and D(k,k) is a 1-by-1
  !          diagonal block.  If UPLO = 'U' and IPIV(k) = IPIV(k-1) < 0,
  !          then rows and columns k-1 and -IPIV(k) were interchanged and
  !          D(k-1:k,k-1:k) is a 2-by-2 diagonal block.  If UPLO = 'L' and
  !          IPIV(k) = IPIV(k+1) < 0, then rows and columns k+1 and
  !          -IPIV(k) were interchanged and D(k:k+1,k:k+1) is a 2-by-2
  !          diagonal block.
  !  B       (input/output) DOUBLE PRECISION array, dimension (LDB,NRHS)
  !          On entry, the N-by-NRHS right hand side matrix B.
  !          On exit, if INFO = 0, the N-by-NRHS solution matrix X.
  !  LDB     (input) INTEGER
  !          The leading dimension of the array B.  LDB >= max(1,N).
  !  INFO    (output) INTEGER
  !          = 0:  successful exit
  !          < 0:  if INFO = -i, the i-th argument had an illegal value
  !          > 0:  if INFO = i, D(i,i) is exactly zero.  The factorization
  !                has been completed, but the block diagonal matrix D is
  !                exactly singular, so the solution could not be
  !                computed.
  !
  !  Further Details
  !  The packed storage scheme is illustrated by the following example
  !  when N = 4, UPLO = 'U':
  !
  !  Two-dimensional storage of the symmetric matrix A:
  !     a11 a12 a13 a14
  !         a22 a23 a24
  !             a33 a34     (aij = aji)
  !                 a44
  !
  !  Packed storage of the upper triangle of A:
  !  AP = [ a11, a12, a22, a13, a23, a33, a14, a24, a34, a44 ]
  !
  ! external routines: LSAME_mn,DSPTRF, DSPTRS, XERBLA
  ! intrinsic function: MAX
  logical :: LSAME_mn

  !     Test the input parameters.
  info = 0
  if( .not.lsame_mn(UPLO,'U') .and. .not.lsame_mn(UPLO,'L') ) then
     info = -1
  else if( n.lt.0 ) then
     info = -2
  else if( nrhs.lt.0 ) then
     info = -3
  else if( ldb.lt.MAX(1,n) ) then
     info = -7
  end if
  if( info.ne.0 ) then
     call xerbla_mn( 'dspsv ', -info )
     return
  end if

  ! Compute the factorization A = U*D*U' or A = L*D*L'.
  call dsptrf_mn(UPLO,n,AP,IPIV,info)

  ! Solve the system A*X = B, overwriting B with X.
  if(info.eq.0) call dsptrs_mn(UPLO,n,nrhs,AP,IPIV,B,LDB,info)

  return
  end subroutine dspsv_mn


  subroutine dsptrf_mn( UPLO, N, AP, IPIV, INFO )
  !
  !  -- LAPACK routine (version 1.1) --
  !     Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
  !     Courant Institute, Argonne National Lab, and Rice University
  !     March 31, 1993
  !
  use chm_kinds
  implicit none
  character ::     UPLO
  integer   ::     INFO,N,IPIV(*)
  real(chm_real):: AP(*)
  !
  !  Purpose
  !  DSPTRF computes the factorization of a real symmetric matrix A stored
  !  in packed format using the Bunch-Kaufman diagonal pivoting method:
  !     A = U*D*U**T  or  A = L*D*L**T
  !  where U (or L) is a product of permutation and unit upper (lower)
  !  triangular matrices, and D is symmetric and block diagonal with
  !  1-by-1 and 2-by-2 diagonal blocks.
  !
  !  Arguments
  !  UPLO    (input) CHARACTER*1
  !          = 'U':  Upper triangle of A is stored;
  !          = 'L':  Lower triangle of A is stored.
  !  N       (input) INTEGER
  !          The order of the matrix A.  N >= 0.
  !  AP      (input/output) DOUBLE PRECISION array, dimension (N*(N+1)/2)
  !          On entry, the upper or lower triangle of the symmetric matrix
  !          A, packed columnwise in a linear array.  The j-th column of A
  !          is stored in the array AP as follows:
  !          if UPLO = 'U', AP(i + (j-1)*j/2) = A(i,j) for 1<=i<=j;
  !          if UPLO = 'L', AP(i + (j-1)*(2n-j)/2) = A(i,j) for j<=i<=n.
  !          On exit, the block diagonal matrix D and the multipliers used
  !          to obtain the factor U or L, stored as a packed triangular
  !          matrix overwriting A (see below for further details).
  !  IPIV    (output) INTEGER array, dimension (N)
  !          Details of the interchanges and the block structure of D.
  !          If IPIV(k) > 0, then rows and columns k and IPIV(k) were
  !          interchanged and D(k,k) is a 1-by-1 diagonal block.
  !          If UPLO = 'U' and IPIV(k) = IPIV(k-1) < 0, then rows and
  !          columns k-1 and -IPIV(k) were interchanged and D(k-1:k,k-1:k)
  !          is a 2-by-2 diagonal block.  If UPLO = 'L' and IPIV(k) =
  !          IPIV(k+1) < 0, then rows and columns k+1 and -IPIV(k) were
  !          interchanged and D(k:k+1,k:k+1) is a 2-by-2 diagonal block.
  !  INFO    (output) INTEGER
  !          = 0: successful exit
  !          < 0: if INFO = -i, the i-th argument had an illegal value
  !          > 0: if INFO = i, D(i,i) is exactly zero.  The factorization
  !               has been completed, but the block diagonal matrix D is
  !               exactly singular, and division by zero will occur if it
  !               is used to solve a system of equations.
  !
  !  Further Details
  !  If UPLO = 'U', then A = U*D*U', where
  !     U = P(n)*U(n)* ... *P(k)U(k)* ...,
  !  i.e., U is a product of terms P(k)*U(k), where k decreases from n to
  !  1 in steps of 1 or 2, and D is a block diagonal matrix with 1-by-1
  !  and 2-by-2 diagonal blocks D(k).  P(k) is a permutation matrix as
  !  defined by IPIV(k), and U(k) is a unit upper triangular matrix, such
  !  that if the diagonal block D(k) is of order s (s = 1 or 2), then
  !             (   I    v    0   )   k-s
  !     U(k) =  (   0    I    0   )   s
  !             (   0    0    I   )   n-k
  !                k-s   s   n-k
  !  If s = 1, D(k) overwrites A(k,k), and v overwrites A(1:k-1,k).
  !  If s = 2, the upper triangle of D(k) overwrites A(k-1,k-1), A(k-1,k),
  !  and A(k,k), and v overwrites A(1:k-2,k-1:k).
  !
  !  If UPLO = 'L', then A = L*D*L', where
  !     L = P(1)*L(1)* ... *P(k)*L(k)* ...,
  !  i.e., L is a product of terms P(k)*L(k), where k increases from 1 to
  !  n in steps of 1 or 2, and D is a block diagonal matrix with 1-by-1
  !  and 2-by-2 diagonal blocks D(k).  P(k) is a permutation matrix as
  !  defined by IPIV(k), and L(k) is a unit lower triangular matrix, such
  !  that if the diagonal block D(k) is of order s (s = 1 or 2), then
  !             (   I    0     0   )  k-1
  !     L(k) =  (   0    I     0   )  s
  !             (   0    v     I   )  n-k-s+1
  !                k-1   s  n-k-s+1
  !  If s = 1, D(k) overwrites A(k,k), and v overwrites A(k+1:n,k).
  !  If s = 2, the lower triangle of D(k) overwrites A(k,k), A(k+1,k),
  !  and A(k+1,k+1), and v overwrites A(k+2:n,k:k+1).
  !
  ! external functions : LSAME,IDAMAX,DLAEV2, DROT, DSCAL, DSPR, DSWAP, XERBLA
  ! intrinsic functions: ABS, MAX, SQRT

  ! local variables
  logical :: LSAME_mn
  integer :: idamax_mn  ! external function
  logical ::       UPPER
  integer ::       IMAX, J, JMAX, K, KC, KK, KNC, KP, KPC, KSTEP, KX, NPP
  real(chm_real):: ABSAKK, ALPHA, C, COLMAX, R1, R2, ROWMAX, S, T
 
  real(chm_real),parameter :: zero=0.0D+0, one=1.0D+0, eight=8.0D+0, sevten=17.0D+0


  info = 0
  upper = lsame_mn( UPLO, 'U' )
  if( .not.upper .and. .not.lsame_mn( UPLO, 'L' ) ) then
     info = -1
  else if( n.lt.0 ) then
     info = -2
  end if
  if( info.ne.0 ) then
     call xerbla_mn( 'dsptrf', -info )
     return
  end if

  ! Initialize ALPHA for use in choosing pivot block size.
  alpha = (one+SQRT(sevten))/eight

  if(upper) then
     ! Factorize A as U*D*U' using the upper triangle of A
     ! K is the main loop index, decreasing from N to 1 in steps of 1 or 2
     k  = n
     kc =(n-1)*n/2+1
     loop1ST: do
       knc = kc
       if( k.lt.1 ) exit loop1ST  ! If K < 1, exit from loop

       ! init
       kstep = 1

       ! Determine rows and columns to be interchanged and whether
       ! a 1-by-1 or 2-by-2 pivot block will be used
       absakk = abs(ap(kc+k-1))

       ! IMAX is the row-index of the largest off-diagonal element in
       ! column K, and COLMAX is its absolute value
       if( k.gt.1 ) then
          imax   = idamax_mn(k-1,ap(kc),1)
          colmax = abs(ap(kc+imax-1))
       else
          colmax = zero
       end if

       if( max(absakk,colmax) .eq. zero ) then
          ! Column K is zero: set INFO and continue
          if( info.eq.0 ) info = k
          kp = k
       else
          if( absakk.ge.alpha*colmax ) then
             ! no interchange, use 1-by-1 pivot block
             kp = k
          else
             ! jmax is the column-index of the largest off-diagonal element in row
             ! imax, and rowmax is its absolute value.
             rowmax= zero
             jmax  = imax
             kx    = imax*(imax+1)/2 + imax
             do j = imax+1, k
                if(abs(ap(kx)).gt.rowmax) then
                   rowmax = abs(ap(kx))
                   jmax = j
                end if
                kx = kx+j
             end do
             kpc = (imax-1)*imax/2 + 1
             if(imax.gt.1) then
                jmax   = idamax_mn(imax-1,ap(kpc),1)
                rowmax = MAX(rowmax,abs(ap(kpc+jmax-1)))
             end if

             if(absakk.ge.alpha*colmax*(colmax/rowmax)) then
                ! no interchange, use 1-by-1 pivot block
                kp = k
             else if(abs(ap(kpc+imax-1)).ge.alpha*rowmax) then
                ! interchange rows and columns K and IMAX, use 1-by-1 pivot block
                kp = imax
             else
                ! interchange rows and columns K-1 and IMAX, use 2-by-2 pivot block
                kp    = imax
                kstep = 2
             end if
          end if

          kk = k - kstep + 1
          if( kstep.eq.2 ) knc = knc - k + 1
          if( kp.ne.kk ) then
             ! Interchange rows and columns KK and KP in the leading submatrix A(1:k,1:k)
             call dswap_mn(kp-1,ap(knc),1,ap(kpc),1)
             kx = kpc + kp - 1
             do j = kp+1,kk-1
                kx          = kx + j - 1
                t           = ap(knc+j-1)
                ap(knc+j-1) = ap(kx)
                ap(kx)      = t
             end do
             t            = ap(knc+kk-1)
             ap(knc+kk-1) = ap(kpc+kp-1)
             ap(kpc+kp-1) = t
             if( kstep.eq.2 ) then
                t           = ap(kc+k-2)
                ap(kc+k-2)  = ap(kc+kp-1)
                ap(kc+kp-1) = T
             end if
          end if

          ! Update the leading submatrix
          if( kstep.eq.1 ) then
             ! 1-by-1 pivot block D(k): column k now holds W(k) = U(k)*D(k),
             !                          where U(k) is the k-th column of U
             ! Perform a rank-1 update of A(1:k-1,1:k-1) as 
             !                          A := A - U(k)*D(k)*U(k)' = A - W(k)*1/D(k)*W(k)'
             r1 = one/ap(kc+k-1)
             call dspr_mn(UPLO,k-1,-r1,ap(kc),1,ap)
             ! Store U(k) in column k
             call dscal_mn(k-1,r1,ap(kc),1)
          else
             ! 2-by-2 pivot block D(k): columns k and k-1 now hold
             !                          ( W(k-1) W(k) ) = ( U(k-1) U(k) )*D(k)
             !                          where U(k) and U(k-1) are the k-th and (k-1)-th columns of U
             ! Perform a rank-2 update of A(1:k-2,1:k-2) as
             !                         A := A - ( U(k-1) U(k) )*D(k)*( U(k-1) U(k) )'
             !                            = A - ( W(k-1) W(k) )*inv(D(k))*( W(k-1) W(k) )'
             !                         Convert this to two rank-1 updates by using 
             !                         the eigen-decomposition of D(k)
             call dlaev2_mn(ap(kc-1),ap(kc+k-2),ap(kc+k-1),r1,r2,c,s)
             r1 = one/r1
             r2 = one/r2
             call drot_mn(k-2,ap(knc),1,ap(kc),1,c,s)
             call dspr_mn(UPLO,k-2,-r1,ap(knc),1,ap)
             call dspr_mn(UPLO,k-2,-r2,ap(kc ),1,ap)
             ! Store U(k) and U(k-1) in columns k and k-1
             call dscal_mn(k-2,r1,ap(knc),1)
             call dscal_mn(k-2,r2,ap(kc ),1)
             call drot_mn(k-2,ap(knc),1,ap(kc),1,c,-s)
          end if
       end if  ! (MAX(absakk,colmax) .eq. zero)

       ! Store details of the interchanges in IPIV
       if( kstep.eq.1 ) then
          ipiv(k)   = kp
       else
          ipiv(k)   =-kp
          ipiv(k-1) =-kp
       end if

       ! Decrease K and return to the start of the main loop
       k = k - kstep
       kc = knc - k
     end do loop1ST
  else
     ! Factorize A as L*D*L' using the lower triangle of A
     ! K is the main loop index, increasing from 1 to N in steps of 1 or 2
     k   = 1
     kc  = 1
     npp = n*(n+1)/2
     loop2ND: do
        knc = kc
        if( k.gt.n ) exit loop2ND  ! if K > N, exit from loop

        ! init
        kstep = 1

        ! Determine rows and columns to be interchanged and whether
        ! a 1-by-1 or 2-by-2 pivot block will be used
        absakk = abs(ap(kc))

        ! IMAX is the row-index of the largest off-diagonal element in
        ! column K, and COLMAX is its absolute value
        if( k.lt.n ) then
           imax   = k + idamax_mn(n-k,ap(kc+1),1)
           colmax = abs(ap(kc+imax-k))
        else
           colmax = zero
        end if

        if( MAX(absakk,colmax).eq.zero ) then
           ! Column K is zero: set INFO and continue
           if( info.eq.0 ) info = k
           kp = k
        else
           if(absakk.ge.alpha*colmax) then
              ! no interchange, use 1-by-1 pivot block
              kp = k
           else
              ! JMAX is the column-index of the largest off-diagonal
              ! element in row IMAX, and ROWMAX is its absolute value
              rowmax = zero
              kx = kc + imax - k
              do j = k, imax - 1
                 if(abs(ap(kx)).gt.rowmax) then
                    rowmax = abs(ap(kx))
                    jmax = j
                 end if
                 kx = kx + n - j
              end do
              kpc = npp - (n-imax+1)*(n-imax+2)/2 + 1
              if( imax.lt.n ) then
                 jmax = imax + idamax_mn(n-imax,ap(kpc+1),1)
                 rowmax = MAX(rowmax,abs(ap(kpc+jmax-imax)))
              end if

              if(absakk.ge.alpha*colmax*(colmax/rowmax)) then
                 !  no interchange, use 1-by-1 pivot block
                 kp = k
              else if(abs(ap(kpc)).ge.alpha*rowmax) then
                 !  interchange rows and columns K and IMAX, use 1-by-1 pivot block
                 kp = imax
              else
                 !  interchange rows and columns K+1 and IMAX, use 2-by-2 pivot block
                 kp = imax
                 kstep = 2
              end if
           end if

           kk = k + kstep - 1
           if( kstep.eq.2 ) knc = knc + n - k + 1
           if( kp.ne.kk ) then
              ! Interchange rows and columns KK and KP in the trailing submatrix A(k:n,k:n)
              if( kp.lt.n ) call dswap_mn(n-kp,ap(knc+kp-kk+1),1,ap(kpc+1),1)
              kx = knc + kp - kk
              do j = kk+1, kp-1
                 kx          = kx+n-j+1
                 t           = ap(knc+j-kk)
                 ap(knc+j-kk)= ap(kx)
                 ap(kx)      = t
              end do
              t       = ap(knc)
              ap(knc) = ap(kpc)
              ap(kpc) = t
              if( kstep.eq.2) then
                 t          = ap(kc+1)
                 ap(kc+1)   = ap(kc+kp-k)
                 ap(kc+kp-k)= t
              end if
           end if

           ! Update the trailing submatrix
           if( kstep.eq.1 ) then
              !  1-by-1 pivot block D(k): column k now holds W(k) = L(k)*D(k),
              !                           where L(k) is the k-th column of L
              if( k.lt.n ) then
                 ! Perform a rank-1 update of A(k+1:n,k+1:n) as
                 ! A := A - L(k)*D(k)*L(k)' = A - W(k)*(1/D(k))*W(k)'
                 r1 = one/ap(kc)
                 call dspr_mn(UPLO,n-k,-r1,ap(kc+1),1,ap(kc+n-k+1))
                 ! Store L(k) in column K
                 call dscal_mn(n-k,r1,ap(kc+1),1)
              end if
           else
              ! 2-by-2 pivot block D(k): columns K and K+1 now hold
              ! ( W(k) W(k+1) ) = ( L(k) L(k+1) )*D(k)
              ! where L(k) and L(k+1) are the k-th and (k+1)-th columns  of L
              if( k.lt.n-1 ) then
                 ! Perform a rank-2 update of A(k+2:n,k+2:n) as
                 ! A := A - ( L(k) L(k+1) )*D(k)*( L(k) L(k+1) )'
                 !    = A - ( W(k) W(k+1) )*inv(D(k))*( W(k) W(k+1) )'
                 ! Convert this to two rank-1 updates by using the eigen-decomposition of D(k)
                 call dlaev2_mn(ap(kc),ap(kc+1),ap(knc),r1,r2,c,s)
                 r1 = one/r1
                 r2 = one/r2
                 call drot_mn(n-k-1,ap(kc+2),1,ap(knc+1),1,c,s)
                 call dspr_mn(UPLO,n-k-1,-r1,ap(kc +2),1,ap(knc+n-k))
                 call dspr_mn(UPLO,n-k-1,-r2,ap(knc+1),1,ap(knc+n-k))
                 ! Store L(k) and L(k+1) in columns k and k+1
                 call dscal_mn(n-k-1,r1,ap(kc +2),1)
                 call dscal_mn(n-k-1,r2,ap(knc+1),1)
                 call drot_mn(n-k-1,ap(kc+2),1,ap(knc+1),1,c,-s)
              end if
           end if
        end if  ! ( MAX(absakk,colmax).eq.zero )

        ! Store details of the interchanges in IPIV
        if( kstep.eq.1 ) then
           ipiv(k)  = kp
        else
           ipiv(k)  =-kp
           ipiv(k+1)=-kp
        end if

        ! Increase K and return to the start of the main loop
        k  = k + kstep
        kc = knc + n - k + 2
     end do loop2ND
  end if
  return
  end subroutine dsptrf_mn


  subroutine dsptrs_mn( UPLO, N, NRHS, AP, IPIV, B, LDB, INFO )
  !
  !  -- LAPACK routine (version 1.1) --
  !     Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
  !     Courant Institute, Argonne National Lab, and Rice University
  !     March 31, 1993
  !
  use chm_kinds
  implicit none
  character ::  UPLO
  integer   ::  INFO,LDB,N,NRHS,IPIV( * )
  real(chm_real):: AP( * ), B( LDB, * )
  !
  !  Purpose
  !  DSPTRS solves a system of linear equations A*X = B with a real
  !  symmetric matrix A stored in packed format using the factorization
  !  A = U*D*U**T or A = L*D*L**T computed by DSPTRF.
  !
  !  Arguments
  !  UPLO    (input) CHARACTER*1
  !          Specifies whether the details of the factorization are stored
  !          as an upper or lower triangular matrix.
  !          = 'U':  Upper triangular, form is A = U*D*U**T;
  !          = 'L':  Lower triangular, form is A = L*D*L**T.
  !  N       (input) INTEGER
  !          The order of the matrix A.  N >= 0.
  !  NRHS    (input) INTEGER
  !          The number of right hand sides, i.e., the number of columns
  !          of the matrix B.  NRHS >= 0.
  !  AP      (input) DOUBLE PRECISION array, dimension (N*(N+1)/2)
  !          The block diagonal matrix D and the multipliers used to
  !          obtain the factor U or L as computed by DSPTRF, stored as a
  !          packed triangular matrix.
  !  IPIV    (input) INTEGER array, dimension (N)
  !          Details of the interchanges and the block structure of D
  !          as determined by DSPTRF.
  !  B       (input/output) DOUBLE PRECISION array, dimension (LDB,NRHS)
  !          On entry, the right hand side matrix B.
  !          On exit, the solution matrix X.
  !  LDB     (input) INTEGER
  !          The leading dimension of the array B.  LDB >= max(1,N).
  !  INFO    (output) INTEGER
  !          = 0:  successful exit
  !          < 0: if INFO = -i, the i-th argument had an illegal value
  !
  ! local variables
  logical :: upper
  integer :: j,k,kc,kp
  real(chm_real):: AK,AKM1,AKM1K,BK,BKM1,DENOM,r_akm1k,r_denom

  real(chm_real),parameter :: one=1.0d+0
  !
  ! external functions: LSAME,DGEMV,DGER,DSCAL,DSWAP,XERBLA
  ! intrinsic functions: MAX
  logical :: LSAME_mn


  info = 0
  upper = lsame_mn( UPLO, 'U' )
  if( .not.upper .and. .not.lsame_mn( UPLO, 'L' ) ) then
     info = -1
  else if( n.lt.0 ) then
     info = -2
  else if( nrhs.lt.0 ) then
     info = -3
  else if( LDB.lt.MAX(1,n) ) then
     info = -7
  end if
  if( info.ne.0 ) then
     call xerbla_mn( 'dsptrs', -info )
     return
  end if

  ! Quick return if possible
  if( n.eq.0 .or. nrhs.eq.0 ) return

  if( upper ) then
     ! Solve A*X = B, where A = U*D*U'.
     !        First solve U*D*X = B, overwriting B with X.
     !        K is the main loop index, decreasing from N to 1 in steps of
     !        1 or 2, depending on the size of the diagonal blocks.
     k  = n
     kc = n*(n+1)/2 + 1
     loop1ST: do
        if( k.lt.1 ) exit loop1ST  ! if K < 1, exit from loop.

        kc = kc - k
        if( ipiv(k).gt.0 ) then
           ! 1 x 1 diagonal block: Interchange rows K and IPIV(K).
           kp = ipiv(k)
           if(kp.ne.k) call dswap_mn(nrhs,b(k,1),LDB,b(kp,1),LDB)

           ! Multiply by inv(U(K)), where U(K) is the transformation
           ! stored in column K of A.
           call dger_mn(k-1,nrhs,-one,ap(kc),1,b(k,1),LDB,b(1,1),LDB)

           ! Multiply by the inverse of the diagonal block.
           call dscal_mn(nrhs,one/ap(kc+k-1),b(k,1),LDB)
           k = k - 1
        else
           ! 2 x 2 diagonal block: Interchange rows K-1 and -IPIV(K).
           kp = -ipiv(k)
           if(kp.ne.(k-1)) call dswap_mn(nrhs,b(k-1,1),LDB,b(kp,1),LDB)

           ! Multiply by inv(U(K)), where U(K) is the transformation
           ! stored in columns K-1 and K of A.
           call dger_mn(k-2,nrhs,-one,ap(kc)      ,1,b(k  ,1),LDB,b(1,1),LDB)
           call dger_mn(k-2,nrhs,-one,ap(kc-(k-1)),1,b(k-1,1),LDB,b(1,1),LDB)

           ! Multiply by the inverse of the diagonal block.
           akm1k  = ap(kc+k-2)
           r_akm1k= one/akm1k
           akm1   = ap(kc-1)*r_akm1k  ! / AKM1K
           ak     = ap(kc+k-1)*r_akm1k  ! / AKM1K
           denom  = akm1*ak - one
           r_denom= one/denom
           do j = 1,nrhs
              bkm1    = b(k-1,j)*r_akm1k         ! / AKM1K
              bk      = b(k,j)*r_akm1k           ! / AKM1K
              b(k-1,j)= (ak*bkm1 - bk)*r_denom   ! / DENOM
              B(k,j)  = (akm1*bk - bkm1)*r_denom ! / DENOM
           end do
           kc = kc - k + 1
           k  = k - 2
        end if
     end do loop1ST

     ! Next solve U'*X = B, overwriting B with X.
     ! K is the main loop index, increasing from 1 to N in steps of 1 or 2, 
     ! depending on the size of the diagonal blocks.
     K = 1
     KC = 1
     loop2ND: do
        if( k.gt.n ) exit loop2ND   ! if K > N, exit from loop.

        if( ipiv(k).gt.0 ) then
           ! 1 x 1 diagonal block
           ! Multiply by inv(U'(K)), where U(K) is the transformation stored in column K of A.
           call dgemv_mn('Transpose',k-1,nrhs,-one,b,LDB,ap(kc),1,one,b(k,1),LDB)

           ! Interchange rows K and IPIV(K).
           kp = ipiv(k)
           if(kp.ne.k) call dswap_mn(nrhs,b(k,1),LDB,b(kp,1),LDB)
           kc = kc + k
           k  = k + 1
        else
           ! 2 x 2 diagonal block
           ! Multiply by inv(U'(K+1)), where U(K+1) is the transformation stored in columns K and K+1 of A.
           call dgemv_mn('Transpose',k-1,nrhs,-one,b,LDB,ap(kc  ),1,one,b(k  ,1),LDB)
           call dgemv_mn('Transpose',k-1,nrhs,-one,b,LDB,ap(kc+k),1,one,b(k+1,1),LDB)

           ! Interchange rows K and -IPIV(K).
           kp = -ipiv(k)
           if(kp.ne.k) call dswap_mn(nrhs,b(k,1),LDB,b(kp,1),LDB)
           kc = kc + 2*k + 1
           k  = k + 2
        end if
     end do loop2ND
  else
     ! Solve A*X = B, where A = L*D*L'.
     !    First solve L*D*X = B, overwriting B with X.
     !    K is the main loop index, increasing from 1 to N in steps of 1 or 2, 
     !    depending on the size of the diagonal blocks.
     k  = 1
     kc = 1
     loop3RD: do
        if( k.gt.n ) exit loop3RD   ! if K > N, exit from loop.

        if( ipiv(k).gt.0 ) then
           ! 1 x 1 diagonal block: Interchange rows K and IPIV(K).
           kp = ipiv(k)
           if(kp.ne.k) call dswap_mn(nrhs,b(k,1),LDB,b(kp,1),LDB)

           ! Multiply by inv(L(K)), where L(K) is the transformation stored in column K of A.
           if(k.lt.n) call dger_mn(n-k,nrhs,-one,ap(kc+1),1,b(k,1),LDB,b(k+1,1),LDB)

           ! Multiply by the inverse of the diagonal block.
           call dscal_mn(nrhs,one/ap(kc),b(k,1),LDB)
           kc = kc + n - k + 1
           k  = k + 1
        else
           ! 2 x 2 diagonal block: Interchange rows K+1 and -IPIV(K).
           kp = -ipiv(k)
           if(kp.ne.(k+1)) call dswap_mn(nrhs,b(k+1,1),LDB,b(kp,1),LDB)

           ! Multiply by inv(L(K)), where L(K) is the transformation stored in columns K and K+1 of A.
           if( k.lt.(n-1) ) then
              call dger_mn(n-k-1,nrhs,-one,ap(kc+2)    ,1,b(k  ,1),LDB,b(k+2,1),LDB)
              call dger_mn(n-k-1,nrhs,-one,ap(kc+n-k+2),1,b(k+1,1),LDB,b(k+2,1),LDB)
           end if

           ! Multiply by the inverse of the diagonal block.
           akm1k  = ap(kc+1)
           r_akm1k= one/akm1k
           akm1   = ap(kc)*r_akm1k        ! / AKM1K
           ak     = ap(kc+n-k+1)*r_akm1k  ! / AKM1K
           denom  = akm1*ak - one
           r_denom= one/denom
           do j = 1, nrhs
              bkm1    = b(k,j)*r_akm1k            ! / AKM1K
              bk      = b(k+1,j)*r_akm1k          ! / AKM1K
              b(k,j)  = (ak*bkm1 - bk)*r_denom    ! / DENOM
              b(k+1,j)= (akm1*bk - bkm1)*r_denom  ! / DENOM
           end do
           kc = kc + 2*(n-k) + 1
           k  = k + 2
        end if
     end do loop3RD

     ! Next solve L'*X = B, overwriting B with X.
     !     K is the main loop index, decreasing from N to 1 in steps of
     !     1 or 2, depending on the size of the diagonal blocks.
     k  = n
     kc = n*(n+1)/2 + 1
     loop4RD: do
        if( k.lt.1 ) exit loop4RD  ! if K < 1, exit from loop.

        kc = kc - ( n-k+1 )
        if( ipiv(k).gt.0 ) then
           ! 1 x 1 diagonal block
           ! Multiply by inv(L'(K)), where L(K) is the transformation stored in column K of A.
           if(k.lt.n) call dgemv_mn('Transpose',n-k,nrhs,-one,b(k+1,1),LDB,ap(kc+1),1,one,b(k,1),LDB)

           ! Interchange rows K and IPIV(K).
           kp = ipiv(k)
           if(kp.ne.k) call dswap_mn(nrhs,b(k,1),LDB,b(kp,1),LDB)
           k = k - 1
        else
           ! 2 x 2 diagonal block
           ! Multiply by inv(L'(K-1)), where L(K-1) is the transformation stored in columns K-1 and K of A.
           if( k.lt.n ) then
              call dgemv_mn('Transpose',n-k,nrhs,-one,b(k+1,1),LDB,ap(kc+1)    ,1,one,b(k  ,1),LDB)
              call dgemv_mn('Transpose',n-k,nrhs,-one,b(k+1,1),LDB,ap(kc-(n-k)),1,one,b(k-1,1),LDB)
           end if

           ! Interchange rows K and -IPIV(K).
           kp = -ipiv(k)
           if(kp.ne.k) call dswap_mn(nrhs,b(k,1),LDB,b(kp,1),LDB)
           kc = kc - (n-k+2)
           k  = k - 2
        end if
     end do loop4RD
  end if   ! ( upper )
  return
  end subroutine dsptrs_mn


  subroutine dlaev2_mn(A,B,C,RT1,RT2,CS1,SN1)
  !
  !  -- LAPACK auxiliary routine (version 2.0) --
  !     Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
  !     Courant Institute, Argonne National Lab, and Rice University
  !     October 31, 1992
  !
  use chm_kinds
  implicit none
  real(chm_real):: A, B, C, CS1, RT1, RT2, SN1
  !
  !  Purpose
  !  DLAEV2 computes the eigendecomposition of a 2-by-2 symmetric matrix
  !     [  A   B  ]
  !     [  B   C  ].
  !  On return, RT1 is the eigenvalue of larger absolute value, RT2 is the
  !  eigenvalue of smaller absolute value, and (CS1,SN1) is the unit right
  !  eigenvector for RT1, giving the decomposition
  !
  !     [ CS1  SN1 ] [  A   B  ] [ CS1 -SN1 ]  =  [ RT1  0  ]
  !     [-SN1  CS1 ] [  B   C  ] [ SN1  CS1 ]     [  0  RT2 ].
  !
  !  Arguments
  !  A       (input) DOUBLE PRECISION
  !          The (1,1) element of the 2-by-2 matrix.
  !  B       (input) DOUBLE PRECISION
  !          The (1,2) element and the conjugate of the (2,1) element of
  !          the 2-by-2 matrix.
  !  C       (input) DOUBLE PRECISION
  !          The (2,2) element of the 2-by-2 matrix.
  !  RT1     (output) DOUBLE PRECISION
  !          The eigenvalue of larger absolute value.
  !  RT2     (output) DOUBLE PRECISION
  !          The eigenvalue of smaller absolute value.
  !  CS1     (output) DOUBLE PRECISION
  !  SN1     (output) DOUBLE PRECISION
  !          The vector (CS1, SN1) is a unit right eigenvector for RT1.
  !
  !  Further Details
  !  RT1 is accurate to a few ulps barring over/underflow.
  !
  !  RT2 may be inaccurate if there is massive cancellation in the
  !  determinant A*C-B*B; higher precision or correctly rounded or
  !  correctly truncated arithmetic would be needed to compute RT2
  !  accurately in all cases.
  !
  !  CS1 and SN1 are accurate to a few ulps barring over/underflow.
  !
  !  Overflow is possible only if RT1 is within a factor of 5 of overflow.
  !  Underflow is harmless if the input data is 0 or exceeds
  !     underflow_threshold / macheps.
  !
  ! intrinsic functions: ABS, SQRT
  
  ! local variables
  integer :: SGN1,SGN2
  real(chm_real):: AB,ACMN,ACMX,ACS,ADF,CS,CT,DF,RT,SM,TB,TN

  real(chm_real),parameter :: zero=0.0d0,half=0.5d0,one=1.0d0,two=2.0d0

  ! Compute the eigenvalues
  sm  = a + c
  df  = a - c
  adf = abs(df)
  tb  = b + b
  ab  = abs(tb)
  if(abs(a).gt.abs(c)) then
     acmx = a
     acmn = c
  else
     acmx = c
     acmn = a
  end if
  if( adf.gt.ab ) then
     rt = adf*SQRT( one+(ab/adf)**2 )
  else if( adf.lt.ab ) then
     rt = ab*SQRT( one+(adf/ab)**2 )
  else
     ! Includes case AB=ADF=0
     rt = ab*sqrt(two)
  end if
  if( sm.lt.zero ) then
     rt1 = half*(sm-rt)
     sgn1 = -1
     ! Order of execution important:
     ! To get fully accurate smaller eigenvalue,
     ! next line needs to be executed in higher precision.
     rt2 = (acmx/rt1)*acmn - (b/rt1)*b
  else if( sm.gt.zero ) then
     rt1 = half*(sm+rt)
     sgn1 = 1
     ! Order of execution important:
     ! To get fully accurate smaller eigenvalue,
     ! next line needs to be executed in higher precision.
     rt2 = (acmx/rt1)*acmn - (b/rt1)*b
  else
     ! Includes case RT1 = RT2 = 0
     rt1  = half*rt
     rt2  =-half*rt
     sgn1 = 1
  end if

  ! Compute the eigenvector
  if( df.ge.zero ) then
     cs = df + rt
     sgn2 = 1
  else
     cs = df - rt
     sgn2 = -1
  end if
  acs = abs(cs)
  if( acs.gt.ab ) then
     ct  =-tb/cs
     sn1 = one/SQRT(one+ct*ct)
     cs1 = ct*sn1
  else
     if( ab.eq.zero ) then
        cs1 = one
        sn1 = zero
     else
        tn  =-cs/tb
        cs1 = one/SQRT(one+tn*tn)
        sn1 = tn*cs1
     end if
  end if
  if( sgn1.eq.sgn2 ) then
     tn  = cs1
     cs1 =-sn1
     sn1 = tn
  end if
  return
  end subroutine dlaev2_mn
  !******************************************************************
!=======================================================================!
!============ END LAPACK routines ======================================!
!=======================================================================|


#else
  subroutine blas_dummy

  return
  end subroutine blas_dummy
#endif
