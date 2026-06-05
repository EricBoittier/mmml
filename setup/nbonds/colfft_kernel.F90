module colfft_kernel
  ! some of the use statements are here to satisfy dependency scanning for
  ! the *.h files that do not get scanned
  use chm_kinds
  use colfft_util
  use consta
#if KEY_BLOCK == 1
  use block_ltm
  use lambdam
#endif
#if KEY_PARALLEL == 1
  use parallel
#endif
  
  implicit none
  
  private

  ! ps, sp => single precision
  ! pd, dp => double precision
  ! o0 => nonorthogonal
  ! o1 => orthogonal
  
  public :: &
       spread_charge_kernel4_fortran, &
       scalar_sumrc_kernel_o0_ps, &
       scalar_sumrc_kernel_o1_ps, &
       scalar_sumrc_kernel_o0_pd, &
       scalar_sumrc_kernel_o1_pd, &
       scalar_sumrc_kernel_orig_o0_ps, &
       scalar_sumrc_kernel_orig_o1_ps, &
       scalar_sumrc_kernel_orig_o0_pd, &
       scalar_sumrc_kernel_orig_o1_pd, &
       spread_charge_kernel_ps, &
       spread_charge_kernel_pd, &
       gather_force_kernel_fortran_ps, &
       gather_force_kernel_fortran_pd, &
       fill_bspline_kernel_fortran_ps, &
       fill_bspline_kernel_fortran_pd
  
! MSLDPME ->
#if KEY_BLOCK==1
  public :: &
       spread_charge_kernel_block_ps, &
       spread_charge_kernel_block_pd, &
       gather_force_kernel_block_fortran_ps, &
       gather_force_kernel_block_fortran_pd
#endif

contains

  ! *
  ! * atomlist(1:natomlist) = list of atoms that this thread fills to the charge grid
  ! * atomlist(1:natomlist) = list of indices between 1...ngrid_atom
  ! * grid_atom(1:ngrid_atom) = global atom index
  ! *
  subroutine spread_charge_kernel4_fortran(natomlist, atomlist, grid_atom, fr1, fr2, fr3, &
       fr1_orig, fr2_orig, fr3_orig, xsize, xysize, ind_add1, ind_add2, &
       charge, theta1, theta2, theta3, qlen, q)
    implicit none
    ! Input / Output
    integer, intent(in) :: natomlist, atomlist(:), grid_atom(:)
    integer, intent(in) :: fr1(:), fr2(:), fr3(:)
    integer, intent(in) :: fr1_orig, fr2_orig, fr3_orig
    integer, intent(in) :: xsize, xysize, ind_add1, ind_add2
    real(chm_real4), intent(in) :: charge(:), theta1(:,:), theta2(:,:), theta3(:,:)
    integer, intent(in) :: qlen
    real(chm_real4), intent(inout) :: q(0:qlen-1)
    ! Variables
    real(chm_real4) chargev, theta3v, prod, val
    integer ith1, ith2, ith3
    integer fr1i, fr2i, fr3i
    integer j, i, k, ind

!$omp do
    do j=1,natomlist
       i = atomlist(j)

       fr1i = fr1(i)
       fr2i = fr2(i)
       fr3i = fr3(i)

       chargev = charge(grid_atom(i))

       ind = (fr1i - fr1_orig) + (fr2i - fr2_orig)*xsize + (fr3i - fr3_orig)*xysize

       do ith3 = 1,4
          theta3v = theta3(ith3,i)*chargev
          do ith2 = 1,4
             prod = theta2(ith2,i)*theta3v
!dir$ simd assert
             do ith1 = 1,4
                val = theta1(ith1,i)*prod
                q(ind) = q(ind) + val
                ind = ind + 1
             enddo
             ind = ind + ind_add1
          enddo
          ind = ind + ind_add2
       enddo

    enddo
!$omp end do

    return
  end subroutine spread_charge_kernel4_fortran

#define COLFFT_PREC PS
#define SINGLEP 1
#include "colfft_kernel_precision.inc"
#undef SINGLEP
#undef COLFFT_PREC

#define COLFFT_PREC PD
#define DOUBLEP 1
#include "colfft_kernel_precision.inc"
#undef DOUBLEP
#undef COLFFT_PREC

end module colfft_kernel
