#define PASTE(a) a
#define ADD_PREC(name) PASTE(name)PASTE(_)COLFFT_PREC

! this code relies on the C Preprocessor
! # define COLFFT_PREC to be either
! PS => SINGLE and set SINGLEP <= 1
! PD => DOUBLE and set DOUBLEP <= 1

! ##EXP P0 sp dp .when. EXP (exp_precision)
! ##PASS1 PS sp
! ##PASS2 PD dp
! ##EXEND

  ! *
  ! * The kernel that fills the charge grid
  ! *
  subroutine ADD_PREC(spread_charge_kernel)(natomlist, atomlist, grid_atom, forder, &
       fr1, fr2, fr3, fr1_orig, fr2_orig, fr3_orig, xsize, xysize, &
       charge, theta1, theta2, theta3, qlen, q)
    implicit none
    ! Input / Output
    integer, intent(in) :: natomlist, atomlist(:), grid_atom(:), forder
    integer, intent(in) :: fr1(:), fr2(:), fr3(:)
    integer, intent(in) :: fr1_orig, fr2_orig, fr3_orig
    integer, intent(in) :: xsize, xysize
    integer, intent(in) :: qlen
#if DOUBLEP==1
    real(chm_real), intent(in) :: charge(:)
    real(chm_real), intent(in),dimension(:,:) :: theta1, theta2, theta3
    real(chm_real), intent(inout) :: q(0:qlen-1)
#elif SINGLEP==1
    real(chm_real4), intent(in) :: charge(:)
    real(chm_real4), intent(in),dimension(:,:) :: theta1, theta2, theta3
    real(chm_real4), intent(inout) :: q(0:qlen-1)
#endif 
    ! Parameters
#if DOUBLEP==1
    real(chm_real), parameter :: zero = 0.0_chm_real
#else /**/
    real(chm_real4), parameter :: zero = 0.0_chm_real4
#endif 
    ! Variables
#if DOUBLEP==1
    real(chm_real) chargen, theta3v, prod, val
#elif SINGLEP==1
    real(chm_real4) chargen, theta3v, prod, val
#endif 
    integer fr1i, fr2i, fr3i
    integer ith1, ith2, ith3
    integer ind_add1, ind_add2
    integer ii, i, n, ind

    ind_add1 = xsize - forder
    ind_add2 = xysize - forder*xsize

    do ii = 1,natomlist
       i = atomlist(ii)
       n = grid_atom(i)

       fr1i = fr1(i)
       fr2i = fr2(i)
       fr3i = fr3(i)

       chargen = charge(n)

!       if (chargen == zero) cycle

       ind = (fr1i - fr1_orig) + (fr2i - fr2_orig)*xsize + (fr3i - fr3_orig)*xysize

       select case (forder)
       case (4)
          do ith3 = 1,4
             theta3v = theta3(ith3,i)*chargen
             do ith2 = 1,4
                prod = theta2(ith2,i)*theta3v
!dec$ ivdep
                do ith1 = 1,4
                   val = theta1(ith1,i)*prod
                   q(ind) = q(ind) + val
                   ind = ind + 1
                enddo
                ind = ind + ind_add1
             enddo
             ind = ind + ind_add2
          enddo
       case default
          do ith3 = 1,forder
             theta3v = theta3(ith3,i)*chargen
             do ith2 = 1,forder
                prod = theta2(ith2,i)*theta3v
                do ith1 = 1,forder
                   val = theta1(ith1,i)*prod
                   q(ind) = q(ind) + val
                   ind = ind + 1
                enddo
                ind = ind + ind_add1
             enddo
             ind = ind + ind_add2
          enddo
       end select
    enddo

    return
  end subroutine ADD_PREC(spread_charge_kernel)

! MSLDPME ->
#if KEY_BLOCK==1
  subroutine ADD_PREC(spread_charge_kernel_block)(natomlist, atomlist, grid_atom, forder, &
       fr1, fr2, fr3, fr1_orig, fr2_orig, fr3_orig, xsize, xysize, &
       charge, theta1, theta2, theta3, qlen, q)
    use lambdam,only:bixlam
    use block_ltm,only:iblckp
    implicit none
    ! Input / Output
    integer, intent(in) :: natomlist, atomlist(:), grid_atom(:), forder
    integer, intent(in) :: fr1(:), fr2(:), fr3(:)
    integer, intent(in) :: fr1_orig, fr2_orig, fr3_orig
    integer, intent(in) :: xsize, xysize
    integer, intent(in) :: qlen
#if DOUBLEP==1
    real(chm_real), intent(in) :: charge(:)
    real(chm_real), intent(in),dimension(:,:) :: theta1, theta2, theta3
    real(chm_real), intent(inout) :: q(0:qlen-1)
#elif SINGLEP==1
    real(chm_real4), intent(in) :: charge(:)
    real(chm_real4), intent(in),dimension(:,:) :: theta1, theta2, theta3
    real(chm_real4), intent(inout) :: q(0:qlen-1)
#endif 
    ! Parameters
#if DOUBLEP==1
    real(chm_real), parameter :: zero = 0.0_chm_real
#else /**/
    real(chm_real4), parameter :: zero = 0.0_chm_real4
#endif 
    ! Variables
#if DOUBLEP==1
    real(chm_real) chargen, theta3v, prod, val
#elif SINGLEP==1
    real(chm_real4) chargen, theta3v, prod, val
#endif 
    integer fr1i, fr2i, fr3i
    integer ith1, ith2, ith3
    integer ind_add1, ind_add2
    integer ii, i, n, ind

    ind_add1 = xsize - forder
    ind_add2 = xysize - forder*xsize

    do ii = 1,natomlist
       i = atomlist(ii)
       n = grid_atom(i)

       fr1i = fr1(i)
       fr2i = fr2(i)
       fr3i = fr3(i)

       chargen = bixlam(iblckp(n)) * charge(n)

!       if (chargen == zero) cycle

       ind = (fr1i - fr1_orig) + (fr2i - fr2_orig)*xsize + (fr3i - fr3_orig)*xysize

       select case (forder)
       case (4)
          do ith3 = 1,4
             theta3v = theta3(ith3,i)*chargen
             do ith2 = 1,4
                prod = theta2(ith2,i)*theta3v
!dec$ ivdep
                do ith1 = 1,4
                   val = theta1(ith1,i)*prod
                   q(ind) = q(ind) + val
                   ind = ind + 1
                enddo
                ind = ind + ind_add1
             enddo
             ind = ind + ind_add2
          enddo
       case default
          do ith3 = 1,forder
             theta3v = theta3(ith3,i)*chargen
             do ith2 = 1,forder
                prod = theta2(ith2,i)*theta3v
                do ith1 = 1,forder
                   val = theta1(ith1,i)*prod
                   q(ind) = q(ind) + val
                   ind = ind + 1
                enddo
                ind = ind + ind_add1
             enddo
             ind = ind + ind_add2
          enddo
       end select
    enddo

    return
  end subroutine ADD_PREC(spread_charge_kernel_block)
#endif
! <- MSLDPME

#define ISORTHO O0
#define ORTHOG 0
#include "colfft_kernel_ortho.inc"
#undef ORTHOG
#undef ISORTHO

#define ISORTHO O1
#define ORTHOG 1
#include "colfft_kernel_ortho.inc"
#undef ORTHOG
#undef ISORTHO

  subroutine ADD_PREC(gather_force_kernel_fortran)(forder, istart, iend, grid_atom, &
       fr1, fr2, fr3, &
       fr2_orig, fr3_orig, &
       xsize, ysize, &
       recip, &
       charge, &
       theta1, theta2, theta3, &
       dtheta1, dtheta2, dtheta3, &
       qlen, q, &
       forcex, forcey, forcez)
    implicit none
    ! Input / Output
    integer, intent(in) :: forder, istart, iend, grid_atom(:)
    integer, intent(in) :: fr1(:), fr2(:), fr3(:)
    integer, intent(in) :: fr2_orig, fr3_orig
    integer, intent(in) :: xsize, ysize
    integer, intent(in) :: qlen
#if SINGLEP==1
    real(chm_real4), intent(in) :: recip(3,3)
    real(chm_real4), intent(in) :: charge(:)
    real(chm_real4), intent(in) :: theta1(:,:), theta2(:,:), theta3(:,:)
    real(chm_real4), intent(in) :: dtheta1(:,:), dtheta2(:,:), dtheta3(:,:)
    real(chm_real4), intent(in) :: q(0:qlen-1)
#else /**/
    real(chm_real), intent(in) :: recip(3,3)
    real(chm_real), intent(in) :: charge(:)
    real(chm_real), intent(in) :: theta1(:,:), theta2(:,:), theta3(:,:)
    real(chm_real), intent(in) :: dtheta1(:,:), dtheta2(:,:), dtheta3(:,:)
    real(chm_real), intent(in) :: q(0:qlen-1)
#endif 
    real(chm_real), intent(inout) :: forcex(:), forcey(:), forcez(:)
    ! Parameters
#if SINGLEP==1
    real(chm_real4), parameter :: zero = 0.0_chm_real4
#else /**/
    real(chm_real), parameter :: zero = 0.0_chm_real
#endif 
    ! Variables
#if SINGLEP==1
    real(chm_real4) chargen
    real(chm_real4) f1, f2, f3, q_dtheta1_sum, q_theta1_sum
    real(chm_real4) theta3_val, dtheta3_val, qval
    real(chm_real4) dforcex, dforcey, dforcez
#else /**/
    real(chm_real) chargen
    real(chm_real) f1, f2, f3, q_dtheta1_sum, q_theta1_sum
    real(chm_real) theta3_val, dtheta3_val, qval
    real(chm_real) dforcex, dforcey, dforcez
#endif 
    integer ind_add1, ind_add2, xysize
    integer fr1i, fr2i, fr3i
    integer i, n, ind, ith3, ith2, ith1

    xysize = xsize*ysize
    ind_add1 = xsize - forder
    ind_add2 = xysize - forder*xsize

    do i=istart,iend
       n = grid_atom(i)
       chargen = charge(n)

!       if (chargen /= zero) then
          fr1i = fr1(i)
          fr2i = fr2(i)
          fr3i = fr3(i)
          
          f1 = zero
          f2 = zero
          f3 = zero
          ind = fr1i + (fr2i - fr2_orig)*xsize + (fr3i - fr3_orig)*xysize

          select case (forder)
          case (4)
             do ith3 = 1,4
                theta3_val = theta3(ith3,i)
                dtheta3_val = dtheta3(ith3,i)
                do ith2 = 1,4
                   q_dtheta1_sum = zero
                   q_theta1_sum = zero
                   do ith1 = 1,4
                      qval = q(ind)
                      ind = ind + 1
                      q_dtheta1_sum = q_dtheta1_sum - qval*dtheta1(ith1,i)
                      q_theta1_sum = q_theta1_sum - qval*theta1(ith1,i)
                   end do
                   f1 = f1 + q_dtheta1_sum*theta2(ith2,i)*theta3_val
                   f2 = f2 + q_theta1_sum*dtheta2(ith2,i)*theta3_val
                   f3 = f3 + q_theta1_sum*theta2(ith2,i)*dtheta3_val
                   ind = ind + ind_add1
                end do
                ind = ind + ind_add2
             end do
          case default
             do ith3 = 1,forder
                theta3_val = theta3(ith3,i)
                dtheta3_val = dtheta3(ith3,i)
                do ith2 = 1,forder
                   q_dtheta1_sum = zero
                   q_theta1_sum = zero
                   do ith1 = 1,forder
                      qval = q(ind)
                      ind = ind + 1
                      q_dtheta1_sum = q_dtheta1_sum - qval*dtheta1(ith1,i)
                      q_theta1_sum = q_theta1_sum - qval*theta1(ith1,i)
                   end do
                   f1 = f1 + q_dtheta1_sum*theta2(ith2,i)*theta3_val
                   f2 = f2 + q_theta1_sum*dtheta2(ith2,i)*theta3_val
                   f3 = f3 + q_theta1_sum*theta2(ith2,i)*dtheta3_val
                   ind = ind + ind_add1
                end do
                ind = ind + ind_add2
             end do
          end select
          dforcex = chargen*(recip(1,1)*f1+recip(1,2)*f2+recip(1,3)*f3)
          dforcey = chargen*(recip(2,1)*f1+recip(2,2)*f2+recip(2,3)*f3)
          dforcez = chargen*(recip(3,1)*f1+recip(3,2)*f2+recip(3,3)*f3)
          forcex(n) = forcex(n) - dforcex
          forcey(n) = forcey(n) - dforcey
          forcez(n) = forcez(n) - dforcez
!       endif
    end do


    return
  end subroutine ADD_PREC(gather_force_kernel_fortran)

! MSLDPME ->
#if KEY_BLOCK==1
  subroutine ADD_PREC(gather_force_kernel_block_fortran)(forder, istart, iend, grid_atom, &
       fr1, fr2, fr3, &
       fr2_orig, fr3_orig, &
       xsize, ysize, &
       recip, &
       charge, &
       theta1, theta2, theta3, &
       dtheta1, dtheta2, dtheta3, &
       qlen, q, &
       forcex, forcey, forcez)
    use lambdam,only:bixlam,biflam
    use block_ltm,only:iblckp
    use consta,only:ccelec
    implicit none
    ! Input / Output
    integer, intent(in) :: forder, istart, iend, grid_atom(:)
    integer, intent(in) :: fr1(:), fr2(:), fr3(:)
    integer, intent(in) :: fr2_orig, fr3_orig
    integer, intent(in) :: xsize, ysize
    integer, intent(in) :: qlen
#if SINGLEP==1
    real(chm_real4), intent(in) :: recip(3,3)
    real(chm_real4), intent(in) :: charge(:)
    real(chm_real4), intent(in) :: theta1(:,:), theta2(:,:), theta3(:,:)
    real(chm_real4), intent(in) :: dtheta1(:,:), dtheta2(:,:), dtheta3(:,:)
    real(chm_real4), intent(in) :: q(0:qlen-1)
#else /**/
    real(chm_real), intent(in) :: recip(3,3)
    real(chm_real), intent(in) :: charge(:)
    real(chm_real), intent(in) :: theta1(:,:), theta2(:,:), theta3(:,:)
    real(chm_real), intent(in) :: dtheta1(:,:), dtheta2(:,:), dtheta3(:,:)
    real(chm_real), intent(in) :: q(0:qlen-1)
#endif 
    real(chm_real), intent(inout) :: forcex(:), forcey(:), forcez(:)
    ! Parameters
#if SINGLEP==1
    real(chm_real4), parameter :: zero = 0.0_chm_real4
#else /**/
    real(chm_real), parameter :: zero = 0.0_chm_real
#endif 
    ! Variables
#if SINGLEP==1
    real(chm_real4) chargen
    real(chm_real4) f1, f2, f3, flambda, q_dtheta1_sum, q_theta1_sum
    real(chm_real4) theta3_val, dtheta3_val, qval
    real(chm_real4) dforcex, dforcey, dforcez, dforcelambda
#else /**/
    real(chm_real) chargen
    real(chm_real) f1, f2, f3, flambda, q_dtheta1_sum, q_theta1_sum
    real(chm_real) theta3_val, dtheta3_val, qval
    real(chm_real) dforcex, dforcey, dforcez, dforcelambda
#endif 
    integer ind_add1, ind_add2, xysize
    integer fr1i, fr2i, fr3i
    integer i, n, ind, ith3, ith2, ith1

    xysize = xsize*ysize
    ind_add1 = xsize - forder
    ind_add2 = xysize - forder*xsize

    do i=istart,iend
       n = grid_atom(i)
       chargen = bixlam(iblckp(n)) * charge(n)

!       if (chargen /= zero) then
          fr1i = fr1(i)
          fr2i = fr2(i)
          fr3i = fr3(i)
          
          f1 = zero
          f2 = zero
          f3 = zero
          flambda = zero
          ind = fr1i + (fr2i - fr2_orig)*xsize + (fr3i - fr3_orig)*xysize

          select case (forder)
          case (4)
             do ith3 = 1,4
                theta3_val = theta3(ith3,i)
                dtheta3_val = dtheta3(ith3,i)
                do ith2 = 1,4
                   q_dtheta1_sum = zero
                   q_theta1_sum = zero
                   do ith1 = 1,4
                      qval = q(ind)
                      ind = ind + 1
                      q_dtheta1_sum = q_dtheta1_sum - qval*dtheta1(ith1,i)
                      q_theta1_sum = q_theta1_sum - qval*theta1(ith1,i)
                   end do
                   f1 = f1 + q_dtheta1_sum*theta2(ith2,i)*theta3_val
                   f2 = f2 + q_theta1_sum*dtheta2(ith2,i)*theta3_val
                   f3 = f3 + q_theta1_sum*theta2(ith2,i)*dtheta3_val
                   flambda = flambda + q_theta1_sum*theta2(ith2,i)*theta3_val
                   ind = ind + ind_add1
                end do
                ind = ind + ind_add2
             end do
          case default
             do ith3 = 1,forder
                theta3_val = theta3(ith3,i)
                dtheta3_val = dtheta3(ith3,i)
                do ith2 = 1,forder
                   q_dtheta1_sum = zero
                   q_theta1_sum = zero
                   do ith1 = 1,forder
                      qval = q(ind)
                      ind = ind + 1
                      q_dtheta1_sum = q_dtheta1_sum - qval*dtheta1(ith1,i)
                      q_theta1_sum = q_theta1_sum - qval*theta1(ith1,i)
                   end do
                   f1 = f1 + q_dtheta1_sum*theta2(ith2,i)*theta3_val
                   f2 = f2 + q_theta1_sum*dtheta2(ith2,i)*theta3_val
                   f3 = f3 + q_theta1_sum*theta2(ith2,i)*dtheta3_val
                   flambda = flambda + q_theta1_sum*theta2(ith2,i)*theta3_val
                   ind = ind + ind_add1
                end do
                ind = ind + ind_add2
             end do
          end select
          
          dforcex = chargen*(recip(1,1)*f1+recip(1,2)*f2+recip(1,3)*f3)
          dforcey = chargen*(recip(2,1)*f1+recip(2,2)*f2+recip(2,3)*f3)
          dforcez = chargen*(recip(3,1)*f1+recip(3,2)*f2+recip(3,3)*f3)

          forcex(n) = forcex(n) - dforcex
          forcey(n) = forcey(n) - dforcey
          forcez(n) = forcez(n) - dforcez

          if (iblckp(n) > 1) then
            dforcelambda = ccelec * charge(n) * flambda
!$omp atomic
            biflam(iblckp(n)) = biflam(iblckp(n)) - dforcelambda
!$omp end atomic
          endif
!       endif
    end do


    return
  end subroutine ADD_PREC(gather_force_kernel_block_fortran)
#endif
! <-MSLDPME

  subroutine ADD_PREC(fill_bspline_kernel_fortran)(jstart, jend, grid_atom, &
       x, y, z, charge, recip, &
       forder, ydim, zdim, &
       nfft1, nfft2, nfft3, fr1, fr2, fr3, &
       theta1, theta2, theta3, dtheta1, dtheta2, dtheta3, &
       grid2tx_lo, grid2ty_lo, grid2tz_lo, &
       grid2tx_hi, grid2ty_hi, grid2tz_hi,&  
       grid2tx, grid2ty, grid2tz, &
       nthread, &
       natom_thread, &
       thread_id_list)
    use colfft_util,only:coord_to_grid
    implicit none
    ! Input / Output
    integer, intent(in) :: jstart, jend, grid_atom(:)
#if SINGLEP==1
    real(chm_real4), intent(in) :: x(:), y(:), z(:), charge(:)
#else /**/
    real(chm_real), intent(in) :: x(:), y(:), z(:), charge(:)
#endif 
#if SINGLEP==1
    real(chm_real4), intent(in) :: recip(3,3)
#else /**/
    real(chm_real), intent(in) :: recip(3,3)
#endif 
    integer, intent(in) :: forder, ydim, zdim
    integer, intent(in) :: nfft1, nfft2, nfft3
    integer, intent(out) :: fr1(:), fr2(:), fr3(:)
#if SINGLEP==1
    real(chm_real4), intent(out) :: theta1(:,:), theta2(:,:), theta3(:,:)
    real(chm_real4), intent(out) :: dtheta1(:,:), dtheta2(:,:), dtheta3(:,:)
#else /**/
    real(chm_real), intent(out) :: theta1(:,:), theta2(:,:), theta3(:,:)
    real(chm_real), intent(out) :: dtheta1(:,:), dtheta2(:,:), dtheta3(:,:)
#endif 
    integer, intent(in) :: grid2tx_lo, grid2ty_lo, grid2tz_lo
    integer, intent(in) :: grid2tx_hi, grid2ty_hi, grid2tz_hi
    integer, intent(in) :: grid2tx(0:grid2tx_hi-grid2tx_lo)
    integer, intent(in) :: grid2ty(0:grid2ty_hi-grid2ty_lo)
    integer, intent(in) :: grid2tz(0:grid2tz_hi-grid2tz_lo)
    integer, intent(in) :: nthread
    integer, intent(inout) :: natom_thread(0:nthread-1)
    integer, intent(inout) :: thread_id_list(:)
    ! Parameters
#if SINGLEP==1
    real(chm_real4), parameter :: zero = 0.0_chm_real4, third = 1.0_chm_real4/3.0_chm_real4, &
         half = 0.5_chm_real4, one = 1.0_chm_real4, two = 2.0_chm_real4, three = 2.0_chm_real4
#else /**/
    real(chm_real), parameter :: zero = 0.0_chm_real, third = 1.0_chm_real/3.0_chm_real, &
         half = 0.5_chm_real, one = 1.0_chm_real, two = 2.0_chm_real, three = 2.0_chm_real
#endif 
    ! Functions
#ifdef _OPENMP
    integer omp_get_thread_num
#endif 
    ! Variables
#if SINGLEP==1
    real(chm_real4) w
    real(chm_real4) fr1v, fr2v, fr3v
    real(chm_real4) w1, w2, w3
#else /**/
    real(chm_real) w
    real(chm_real) fr1v, fr2v, fr3v
    real(chm_real) w1, w2, w3
#endif 
    integer fr1i, fr2i, fr3i
    integer dfr2, dfr3
    integer thread_id
    integer j, i

    do j=jstart,jend
       i = grid_atom(j)

       call coord_to_grid(x(i), y(i), z(i), recip, fr1v, fr2v, fr3v)
!!$
!!$       w = x(i)*recip(1,1) + y(i)*recip(2,1) + z(i)*recip(3,1) + two
!!$       fr1v = nfft1*(w - (anint(w) - half))
!!$
!!$       w = x(i)*recip(1,2) + y(i)*recip(2,2) + z(i)*recip(3,2) + two
!!$       fr2v = nfft2*(w - (anint(w) - half))
!!$
!!$       w = x(i)*recip(1,3) + y(i)*recip(2,3) + z(i)*recip(3,3) + two
!!$       fr3v = nfft3*(w - (anint(w) - half))

       fr1i = int(fr1v)
       fr2i = int(fr2v)
       fr3i = int(fr3v)
       ! fr1i = 0...nfft1-1
       ! fr2i = 0...nfft2-1
       ! fr3i = 0...nfft3-1

       w1 = fr1v - fr1i
       w2 = fr2v - fr2i
       w3 = fr3v - fr3i

       ! Apply periodic boundaries
       dfr2 = ((fr2i + forder-1)/ydim)*ydim
       dfr3 = ((fr3i + forder-1)/zdim)*zdim
       fr2i = fr2i - dfr2
       fr3i = fr3i - dfr3
       ! For periodic systems:
       ! fr1i = 0...nfft1-1
       ! fr2i = -(forder-1)...nfft2-forder
       ! fr3i = -(forder-1)...nfft3-forder
       ! For non-periodic systems:
       ! fr1i = 0...nfft1-1
       ! fr2i = 0...nfft2-1
       ! fr3i = 0...nfft3-1

       fr1(j) = fr1i
       fr2(j) = fr2i
       fr3(j) = fr3i

       thread_id = grid2tx(fr1i-grid2tx_lo) + grid2ty(fr2i-grid2ty_lo) + grid2tz(fr3i-grid2tz_lo)
       thread_id_list(j) = thread_id
       natom_thread(thread_id) = natom_thread(thread_id) + 1

       call ADD_PREC(calc_bspline)(forder, w1, theta1(1:forder,j), dtheta1(1:forder,j))
       call ADD_PREC(calc_bspline)(forder, w2, theta2(1:forder,j), dtheta2(1:forder,j))
       call ADD_PREC(calc_bspline)(forder, w3, theta3(1:forder,j), dtheta3(1:forder,j))
    enddo

    return
  end subroutine ADD_PREC(fill_bspline_kernel_fortran)

  subroutine ADD_PREC(calc_bspline)(forder, w, array, darray)
    implicit none
    ! Input / Output
    integer, intent(in) :: forder
#if SINGLEP==1
    real(chm_real4), intent(in) :: w
    real(chm_real4), intent(out) :: array(forder)
    real(chm_real4), intent(out) :: darray(forder)
#else /**/
    real(chm_real), intent(in) :: w
    real(chm_real), intent(out) :: array(forder)
    real(chm_real), intent(out) :: darray(forder)
#endif 
    ! Parameters
#if SINGLEP==1
    real(chm_real4), parameter :: zero = 0.0_chm_real4, third = 1.0_chm_real4/3.0_chm_real4,&
         half = 0.5_chm_real4, one = 1.0_chm_real4, two = 2.0_chm_real4, three = 3.0_chm_real4
#else /**/
    real(chm_real), parameter :: zero = 0.0_chm_real, third = 1.0_chm_real/3.0_chm_real,&
         half = 0.5_chm_real, one = 1.0_chm_real, two = 2.0_chm_real, three = 3.0_chm_real
#endif 
    ! Variables
#if SINGLEP==1
    real(chm_real4) div
#else /**/
    real(chm_real) div
#endif 
    integer k, j

    if (forder == 4) then
       array(4) = zero
       array(2) = w
       array(1) = one - w

       !--- compute standard b-spline recursion
       array(3) = half*w*array(2)
       array(2) = half*((w+one)*array(1) + (two-w)*array(2))
       array(1) = half*(one-w)*array(1)
       
       !--- perform standard b-spline differentiation
       darray(1) = -array(1)
       darray(2) = array(1) - array(2)
       darray(3) = array(2) - array(3)
       darray(4) = array(3) - array(4)
    
       !--- one more recursion
       array(4) = third*w*array(3)
       array(3) = third*((w+one)*array(2) + (three-w)*array(3))
       array(2) = third*((w+two)*array(1) + (two-w)*array(2))
       array(1) = third*(one-w)*array(1)
    else
       array(forder) = zero
       array(2) = w
       array(1) = one - w

       do k = 3,forder-1
          div = one / (k-1)
          array(k) = div*w*array(k-1)
          do j = 1,k-2
             array(k-j) = div*((w+j)*array(k-j-1) + (k-j-w)*array(k-j))
          enddo
          array(1) = div*(one-w)*array(1)
       enddo
       
       !--- perform standard b-spline differentiation
       darray(1) = -array(1)
       do j = 2,forder
          darray(j) = array(j-1) - array(j)
       enddo
       
       !--- one more recursion
       div = one / (forder-1)
       array(forder) = div*w*array(forder-1)
       do j = 1,forder-2
          array(forder-j) = div*((w+j)*array(forder-j-1) + (forder-j-w)*array(forder-j))
       enddo
    
       array(1) = div*(one-w)*array(1)
    endif

    return
  end subroutine ADD_PREC(calc_bspline)

! ##ENDEX (exp_precision)
