!
#if KEY_MNDO97==1 /*mndo97*/

!  subroutine linear(A,B,N,LM2,LM4)
!  !
!  ! store symmetrix square matrix A(LM2,LM2) as linear array B(LM4).
!  ! B(LM4)  corresponds to the lower triangle in row-wise order.
!  ! Elements up to and including the N-th row are copied.
!  ! A and B may occupy the same core space.
!  !
!
!  use chm_kinds
!  implicit none
!
!  integer :: N,LM2,LM4
!  real(chm_real):: A(LM2,LM2),B(LM4)
!
!  integer :: I,J,IJ
!
!  ij = 0
!  do I=1,N
!     b(ij+1:ij+i) = A(1:i,i)
!     ij           = ij+i
!  end do
!  return
!  end


  integer FUNCTION ISTRT_CHECK(mstop,NCNT)
  !
  ! for parallel jobs.
  ! compute mstart and mstop for each node.
  !
  use chm_kinds
#if KEY_PARALLEL==1
  use parallel 
#endif

  implicit none
  integer:: mstop,ncnt
  integer:: mstart,ncnt2,icnt
  !
#if KEY_PARALLEL==1 /*paramain*/
  if(QMPI) then                  ! protect for MPI-pi calc.
     mstart = 1
     mstop  = ncnt
  else
#if KEY_PARAFULL==1 /*parfmain*/
#if KEY_PARASCAL==1 /*parstest*/
#error 'Illegal parallel compile options'
#endif /* (parstest)*/

     if(ncnt.ge.numnod) then 
        ncnt2 = ncnt/NUMNOD
        mstart =  mynod*ncnt2 + 1
        mstop  = (mynod+1)*ncnt2
        if(MYNOD.EQ.(NUMNOD-1)) mstop =ncnt
     else
        ! ncnt < numnod. only do until ncnt = numnod. (each node has a single job to do.)
        if(mynod+1 .le. ncnt) then
           mstart = mynod + 1
           mstop  = mstart
        else
           ! this part has a potential to cause segmentation fault.
           ! mynod+1 > ncnt.. meaning the number of nodes (mynod) is larger tha ncnt.
     ! show warning message
     write(6,*)'ISTRT_CHECK> Severe warning that numnod > NCNT. Shoule check the code.'
           mstart = 0
           mstop  = 0
        end if
     end if

#elif KEY_PARASCAL==1 || KEY_SPACDEC==1 /*parfmain*/
     mstart = 1
     mstop  = ncnt

#else /* (parfmain)*/
#error 'Illegal parallel compile options'
#endif /* (parfmain)*/
  end if                         ! protect for MPI-pi calc.

#else /*   (paramain)*/

  mstart = 1
  mstop  = ncnt

#endif /*  (paramain)*/

  ISTRT_CHECK = mstart
  RETURN
  END function ISTRT_CHECK


  subroutine upper_triangle(A,B,N,n1,n2)
  !
  ! STORE UPPER TRIANGLE MATRIX A(LM4) AS SYMMETRIC SQUARE MATRIX
  ! B(LM2,LM2). A AND B MAY OCCUPY THE SAME CORE SPACE.
  ! N ROWS AND COLUMNS ARE COPIED.
  !
  ! Just fill in the upper triangular part of B(LM2,LM2)
  !
  use chm_kinds

  implicit none
  integer :: N,n2,n1
  real(chm_real):: A(n2),B(n1,n1)

  ! local variables
  integer :: ID,I,J

  id = n2
  do j=n,1,-1
     do i=j,1,-1
        B(i,j) = A(id)
        id     = id-1
     end do
     ! for symmetric matrix, lower triangle.
     !do i=n,j+1,-1
     !   b(i,j)=b(j,i)
     !end do
  end do
  !do i=2,n
  !   do j=1,i-1
  !      b(i,j)=b(j,i)
  !   end do
  !end do

  return
  end subroutine upper_triangle


  subroutine lower_triangle(A,B,N,n1,n2)
  !
  ! STORE LOWER TRIANGLE MATRIX A(LM4) AS SYMMETRIC SQUARE MATRIX 
  ! B(LM2,LM2). A AND B MAY OCCUPY THE SAME CORE SPACE.
  ! N ROWS AND COLUMNS ARE COPIED.
  !
  ! Just, fill in the lower triangluar part of B(LM2,LM2)
  !
  use chm_kinds

  implicit none
  integer :: N,n2,n1
  real(chm_real):: A(n2),B(n1,n1)

  ! local variables
  integer :: ID,I,J

  id = n2
  do j=n,1,-1
     do i=j,1,-1
        B(j,i) = A(id)
        id     = id-1
     end do
  end do

  return
  end subroutine lower_triangle


  subroutine square(A,B,N,n1,n2,q_do_parallel)
  ! 
  ! STORE LOWER TRIANGLE MATRIX A(LM4) AS SYMMETRIC SQUARE MATRIX
  ! B(LM2,LM2). A AND B MAY OCCUPY THE SAME CORE SPACE.
  ! N ROWS AND COLUMNS ARE COPIED.
  ! 
  use chm_kinds
#if KEY_PARALLEL==1
  use parallel,only : mynod,numnod
#endif

  implicit none
  integer :: N,n2,n1
  real(chm_real):: A(n2),B(n1,n1)
  logical :: q_do_parallel

  ! local variables
  integer :: ID,I,J
  real(chm_real):: bb
#if KEY_PARALLEL==1
  integer :: mstart,mstop
#endif

  ! regular MPI-only case.
  id = n2
  do j=n,1,-1
     do i=j,1,-1
        B(i,j) = A(id)
        id     = id-1
     end do
     ! for symmetric matrix, lower triangle.
     !do i=n,j+1,-1
     !   b(i,j)=b(j,i)
     !end do
  end do

#if KEY_PARALLEL==1
  if(q_do_parallel) then
     ! only execute this when using parallel
     ! this part (mstart and mstop) is synchronized with the fast_diag subroutine.
     ! (We only need b(1:n,mstart:mstop) portion.
     mstart = n*(mynod)/numnod+1
     mstop  = n*(mynod+1)/numnod
     do i=mstart,mstop
        do j=i+1,n
           bb     = b(i,j)
           b(j,i) = bb ! b(i,j)
        end do
     end do
  else
#endif
     ! q_do_parallel=.false., do a regular square work.
     do i=2,n
        do j=1,i-1
           bb    =b(j,i)
           b(i,j)=bb ! b(j,i)
        end do
     end do
#if KEY_PARALLEL==1
  end if
#endif
  return
  end subroutine square

  subroutine square2(A,B,C,D,N,n1,n2  &
#if KEY_PARALLEL==1
                    ,q_ij_pair  &
#endif
                    )
  ! 
  ! when the dimensions of the two matrices are the same.
  ! STORE LOWER TRIANGLE MATRIX A(LM4) AS SYMMETRIC SQUARE MATRIX
  ! B(LM2,LM2). A AND B MAY OCCUPY THE SAME CORE SPACE.
  ! N ROWS AND COLUMNS ARE COPIED.
  !
  use chm_kinds

  implicit none
  integer :: N,n2,n1
  real(chm_real):: A(n2),B(n1,n1),C(n2),D(n1,n1)
#if KEY_PARALLEL==1
  logical :: q_ij_pair(n)
#endif

  ! local variables
  integer :: ID,I,J
  real(chm_real):: bb,dd

  id = n2
  do j=n,1,-1
     do i=j,1,-1
        B(i,j) = A(id)
        D(i,j) = C(id)
        id     = id-1
     end do
     ! for symmetric matrix, lower triangle.
     !do i=n,j+1,-1
     !   b(i,j)=b(j,i)
     !   d(i,j)=d(j,i)
     !end do
  end do
  do i=1,n
#if KEY_PARALLEL==1
     if(q_ij_pair(i)) then
        ! only copy necessary information.
#endif
        do j=i+1,n
           bb = b(i,j)
           dd = d(i,j)
           b(j,i) = bb ! b(i,j)
           d(j,i) = dd ! d(i,j)
        end do
#if KEY_PARALLEL==1
     end if
#endif
  end do

  return
  end subroutine square2


  subroutine square_transpose(A,n1)
  !
  ! copy the lower triangle to the upper triangle to complete the square matrix.
  ! 

  use chm_kinds

  implicit none
  integer :: n1
  real(chm_real):: A(n1,n1)

  integer :: i,j

  do i=2,n1
     do j=1,i-1
        A(j,i) = A(i,j)
     end do
  end do
  return
  end subroutine square_transpose


! move to as a contained subroutine in hcorep
!
!  subroutine w2mat(IP,JP,WW,W,LM6,LIMIJ,LIMKL)
!  ! 
!  ! store two-center two-electron integrals in a square matrix.
!  !
!  use chm_kinds
!
!  implicit none
!  integer       :: IP,JP,LM6,LIMIJ,LIMKL
!  real(chm_real):: W(LM6,LM6),WW(LIMKL,LIMIJ)
!
!  ! local variable
!  integer :: KL,ipa,jpa
!
!  ipa    = ip-1
!  jpa    = jp-1
!  do kl=1,LIMKL
!     W(ipa+1:ipa+LIMIJ,jpa+kl) = WW(kl,1:LIMIJ)
!  end do
!  return
!  end subroutine w2mat



  subroutine zzero(nr,nc,a,n1)
  ! 
  ! initialization of the matrix A(LM2,NC)
  ! rows 1 ... NR and columns 1 ...NC are set to zero.
  !
  use chm_kinds
  use number, only : zero

  implicit none
 
  integer        :: nr,nc,n1
  real(chm_real) :: a(n1,nc)

  a(1:nr,1:nc)=zero

  return
  end subroutine zzero

!moved to as a contained subroutine in qm_ewald_calc_ktable.
!  subroutine vdcos(n,x,y)
!  !
!  ! Vectorize Cosine routine
!  !
!  use chm_kinds
!
!  implicit none
!  ! 
!  integer      :: n
!  real(chm_real)     :: x(n), y(n)
!
!  y(1:n) = cos(x(1:n))
!  return
!  end subroutine Vdcos
!--------------------------------------------------------------

  subroutine set_spline_lookup(inpnt,x_gap,r_val,y_val,coef,y_shift)
     !
     ! copied from RXNBF
     !
     use chm_kinds
     use number,only: zero,half,two,four,six
     implicit none

     integer :: inpnt
     real(chm_real) :: x_gap,r_val(inpnt),y_val(inpnt),coef(4,inpnt),y_shift
     integer :: i, j, ncount
     real(chm_real) :: x,y
     real(chm_real),pointer :: s_val(:)=>Null()

     allocate(s_val(inpnt))
     !
     ! initializing spline coefficients.
     y_shift = y_val(1)
     do i = 1,inpnt
        y_val(I) = y_val(I) - y_shift
     end do
     call spline_1d_lookup(r_val,y_val,s_val,x_gap,inpnt)

     !
     ! set spline coefficients.
     do i = 1,inpnt-1
        coef(1,i) = (s_val(i+1)-s_val(i))/(six*x_gap)
        coef(2,i) = half*s_val(i)
        coef(3,i) = (y_val(i+1)-y_val(i))/x_gap - (two*x_gap*s_val(i)+x_gap*s_val(i+1))/six
        coef(4,i) = y_val(i)
     end do

     deallocate(s_val)
     return
     !
     contains
        subroutine spline_1d_lookup(X,Y,S,x_gap,N)
        !
        ! copied from SETSPLN
        !
        INTEGER :: N
        real(chm_real) :: x(*),y(*),s(*),x_gap
        !
        real(chm_real) :: DX1,DX2,DY1,DY2,DXN1,DXN2
        real(chm_real),pointer :: A(:,:)=>Null()
        integer        :: i,j, NM2, NM1

        allocate(A(4,N))
        A(1:4,1:N) =zero
        NM2 = N-2
        NM1 = N-1
        DX1 = X(2)-X(1)
        DY1 =(Y(2)-Y(1))/x_gap*six
        Do I = 1,NM2
           DY2    =(Y(I+2)-Y(I+1))/DX1*six
           A(1,I) = DX1
           A(2,I) = four*DX1
           A(3,I) = DX1
           A(4,I) = DY2 - DY1
           DY1    = DY2
        End do

        do I = 2,NM2
           A(2,I) = A(2,I)-A(1,I)/A(2,I-1)*A(3,I-1)
           A(4,I) = A(4,I)-A(1,I)/A(2,I-1)*A(4,I-1)
        end do
        !
        A(4,NM2) = A(4,NM2)/A(2,NM2)
        do I = 2,NM2
           J      = NM1 - I
           A(4,J) =(A(4,J) - A(3,J) * A(4,J+1))/A(2,J)
        end do
        !
        do I = 1,NM2
           S(I+1) = A(4,I)
        end do

        S(1) = zero
        S(N) = zero
        deallocate(A)
        RETURN
        end subroutine spline_1d_lookup
  end subroutine set_spline_lookup

#else /* (mndo97)*/
  subroutine util_dummy_mnd
  implicit none
  return
  end subroutine util_dummy_mnd
#endif /* (mndo97)*/
