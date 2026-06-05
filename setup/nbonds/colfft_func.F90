module colfft_func

  ! *
  ! * Column FFT subcroutines & functions
  ! *
  ! some of these use statements are here to satisfy dependency scanning
  ! for the .h file
  use chm_kinds
  use colfft_kernel
  use colfft_types
  use colfft_util
  use consta
  use domdec_common
#if KEY_DOMDEC == 1
  use domdec_util_gpu_mod
#endif
#if KEY_BLOCK == 1
  use block_ltm
  use lambdam,only:iqldm_pme
#endif
  use memory
  use nblist_types
  use parallel
  use stream
  
  implicit none
  
  private

  real(chm_real) :: last_energy = 0.0_chm_real, last_virial(6)

#if KEY_DOMDEC==1
  real(chm_real), allocatable, dimension(:) :: q_check_pd
  real(chm_real4), allocatable, dimension(:) :: q_check_ps
  real(chm_real), allocatable, dimension(:) :: dx_orig, dy_orig, dz_orig
#endif 

  ! Public subroutines
  public grad_sumrc_ps, grad_sumrc_pd
  public scalar_sumrc_ps, scalar_sumrc_pd
  public reduce_charge_grid_ps, reduce_charge_grid_pd
  public spread_charge_grid_ps, spread_charge_grid_pd
  public fill_bspline_ps, fill_bspline_pd
#if KEY_LJPME==1
  public scalar_sumrc_lj_ps, scalar_sumrc_lj_pd
#endif
#if KEY_DOMDEC==1
  public test_fill_bspline_ps, test_fill_bspline_pd
  public test_spread_charge_grid_ps, test_spread_charge_grid_pd
  public test1_scalar_sumrc_ps, test1_scalar_sumrc_pd
  public test2_scalar_sumrc_ps, test2_scalar_sumrc_pd
  public test_replicate_grid_borders_ps, test_replicate_grid_borders_pd
  public test1_grad_sumrc
  public test2_grad_sumrc_ps, test2_grad_sumrc_pd
#endif 

  interface
     subroutine spread_charge_kernel4u_sse(natomlist_in, atomlist, grid_atom, &
          fr1, fr2, fr3, &
          fr1_orig_in, fr2_orig_in, fr3_orig_in, &
          xsize_in, xysize_in, &
          charge, &
          theta1, theta2, theta3, q) bind (C)
       use, intrinsic :: iso_c_binding
       import
       integer(c_int), intent(in) :: natomlist_in, atomlist, grid_atom
       integer(c_int), intent(in) :: fr1, fr2, fr3
       integer(c_int), intent(in) :: fr1_orig_in, fr2_orig_in, fr3_orig_in
       integer(c_int), intent(in) :: xsize_in, xysize_in
       real(c_float), intent(in) :: charge
       real(c_float), intent(in) :: theta1, theta2, theta3
       real(c_float), intent(inout) :: q
     end subroutine spread_charge_kernel4u_sse

     subroutine gather_force_kernel4u_sse(istart_in, iend_in, grid_atom, &
          fr1, fr2, fr3, &
          fr2_orig_in, fr3_orig_in, &
          xsize_in, xysize_in, &
          recip, &
          charge, &
          theta1, theta2, theta3, &
          dtheta1, dtheta2, dtheta3, &
          q, &
          forcex, forcey, forcez) bind(C)
       use, intrinsic :: iso_c_binding
       import
       integer(c_int), intent(in) :: istart_in, iend_in, grid_atom
       integer(c_int), intent(in) :: fr1, fr2, fr3
       integer(c_int), intent(in) :: fr2_orig_in, fr3_orig_in
       integer(c_int), intent(in) :: xsize_in, xysize_in
       real(c_float), intent(in) :: recip
       real(c_float), intent(in) :: charge
       real(c_float), intent(in) :: theta1, theta2, theta3
       real(c_float), intent(in) :: dtheta1, dtheta2, dtheta3
       real(c_float), intent(in) :: q
       real(c_double), intent(inout) :: forcex, forcey, forcez
     end subroutine gather_force_kernel4u_sse

     subroutine fill_bspline4_kernel_sse_ps(jstart, jend, grid_atom, x, y, z, charge, recip_in, &
          ydim_in, zdim_in, &
          nfft1_in, nfft2_in, nfft3_in, fr1, fr2, fr3, &
          theta1, theta2, theta3, dtheta1, dtheta2, dtheta3, &
          grid2tx, grid2ty, grid2tz, grid2tx_lo, grid2ty_lo, grid2tz_lo, &
          natom_thread, thread_id_list) bind(C)
       use, intrinsic :: iso_c_binding
       import
       integer(c_int), intent(in) :: jstart, jend, grid_atom
       real(c_float), intent(in) :: x, y, z, charge, recip_in
       integer(c_int), intent(in) :: ydim_in, zdim_in
       integer(c_int), intent(in) :: nfft1_in, nfft2_in, nfft3_in
       integer(c_int), intent(out) :: fr1, fr2, fr3
       real(c_float), intent(out) :: theta1, theta2, theta3
       real(c_float), intent(out) :: dtheta1, dtheta2, dtheta3
       integer(c_int), intent(in) :: grid2tx, grid2ty, grid2tz
       integer(c_int), intent(in) :: grid2tx_lo, grid2ty_lo, grid2tz_lo
       integer(c_int), intent(out) :: natom_thread, thread_id_list
     end subroutine fill_bspline4_kernel_sse_ps

     subroutine fill_bspline4_kernel_sse_pd(jstart, jend, grid_atom, x, y, z, charge, recip_in, &
          ydim_in, zdim_in, &
          nfft1_in, nfft2_in, nfft3_in, fr1, fr2, fr3, &
          theta1, theta2, theta3, dtheta1, dtheta2, dtheta3, &
          grid2tx, grid2ty, grid2tz, grid2tx_lo, grid2ty_lo, grid2tz_lo, &
          natom_thread, thread_id_list) bind(C)
       use, intrinsic :: iso_c_binding
       import
       integer(c_int), intent(in) :: jstart, jend, grid_atom
       real(c_double), intent(in) :: x, y, z, charge, recip_in
       integer(c_int), intent(in) :: ydim_in, zdim_in
       integer(c_int), intent(in) :: nfft1_in, nfft2_in, nfft3_in
       integer(c_int), intent(out) :: fr1, fr2, fr3
       real(c_double), intent(out) :: theta1, theta2, theta3
       real(c_double), intent(out) :: dtheta1, dtheta2, dtheta3
       integer(c_int), intent(in) :: grid2tx, grid2ty, grid2tz
       integer(c_int), intent(in) :: grid2tx_lo, grid2ty_lo, grid2tz_lo
       integer(c_int), intent(out) :: natom_thread, thread_id_list
     end subroutine fill_bspline4_kernel_sse_pd
  end interface

contains

#define COLFFT_PREC PS
#define SINGLEP 1
#include "colfft_func.inc"
#undef SINGLEP
#undef COLFFT_PREC

#define COLFFT_PREC PD
#define DOUBLEP 1
#include "colfft_func.inc"
#undef DOUBLEP
#undef COLFFT_PREC

  ! *
  ! * Builds atomlist_thread() -structure
  ! *
  subroutine build_atomlist_thread_safe(nthread, n_grid_atom, natom_thread, thread_id_list, &
       atomlist_thread)
    use memory
    use nblist_types,only:intarray_t
    implicit none
    ! Input / Output
    integer, intent(in) :: nthread, n_grid_atom
    type(intarray_t), intent(inout) :: natom_thread(0:nthread)
    integer, intent(in) :: thread_id_list(:)
    type(intarray_t), intent(inout) :: atomlist_thread(0:nthread-1)
    ! Variables
    integer thread_id, tid, cur, prev
    integer i, jstart, jend

    ! natom_thread(thread_id, tid) = number of atoms thread "tid" found
    !                                that belong to thread "thread_id"

    do tid=0,nthread-1
       ! Exclusive cumulative sum
       prev = natom_thread(0)%array(tid)
       natom_thread(0)%array(tid) = 0
       do i=1,nthread
          cur = natom_thread(i)%array(tid)
          natom_thread(i)%array(tid) = natom_thread(i-1)%array(tid) + prev
          prev = cur
       enddo
    enddo

    ! atomlist_thread(tid)%array(:) = atomlist for thread "tid"

    ! Building atom list for thread 0:
    ! natom_thread(0,0) = position where thread 0 starts writing
    ! natom_thread(0,1) = position where thread 1 starts writing
    ! natom_thread(0,2) = position where thread 2 starts writing
    ! ...
    ! natom_thread(0,nthread) = number of atoms for thread 0

    ! Building atom list for thread 1:
    ! natom_thread(1,0) = position where thread 0 starts writing
    ! natom_thread(1,1) = position where thread 1 starts writing
    ! natom_thread(1,2) = position where thread 2 starts writing

    do tid=0,nthread-1
       if (natom_thread(nthread)%array(tid) > size(atomlist_thread(tid)%array)) then
          call chmdealloc('colfft_func.src','build_atomlist_thread','atomlist_thread%array',&
               size(atomlist_thread(tid)%array),intg=atomlist_thread(tid)%array)
          call chmalloc('colfft_func.src','build_atomlist_thread','atomlist_thread%array',&
               ceiling(real(natom_thread(nthread)%array(tid))*1.5),intg=atomlist_thread(tid)%array)
       endif
    enddo

    do tid=0,nthread-1
       jstart = tid*n_grid_atom/nthread + 1
       jend = (tid+1)*n_grid_atom/nthread
       do i=jstart,jend
          thread_id = thread_id_list(i)
          natom_thread(tid)%array(thread_id) = natom_thread(tid)%array(thread_id) + 1
          atomlist_thread(thread_id)%array(natom_thread(tid)%array(thread_id)) = i
       enddo
    enddo

    ! Now:
    ! atomlist_thread(tid)%array(:) = atoms that thread "tid" puts into charge grid

    return
  end subroutine build_atomlist_thread_safe

  ! *
  ! * Builds atomlist_thread() -structure
  ! *
  subroutine build_atomlist_thread(nthread, tid, jstart, jend, natom_thread, thread_id_list, &
       atomlist_thread)
    use memory
    use nblist_types,only:intarray_t
    implicit none
    ! Input / Output
    integer, intent(in) :: nthread, tid, jstart, jend
    type(intarray_t), intent(inout) :: natom_thread(0:nthread)
    integer, intent(in) :: thread_id_list(:)
    type(intarray_t), intent(inout) :: atomlist_thread(0:nthread-1)
    ! Variables
    integer thread_id, cur, prev
    integer i

    ! natom_thread(tid)%array(thread_id) = number of atoms thread "tid" found
    !                                      that belong to thread "thread_id"

    ! Exclusive cumulative sum
!$omp barrier
    prev = natom_thread(0)%array(tid)
    natom_thread(0)%array(tid) = 0
    do i=1,nthread
       cur = natom_thread(i)%array(tid)
       natom_thread(i)%array(tid) = natom_thread(i-1)%array(tid) + prev
       prev = cur
    enddo

    ! atomlist_thread(tid)%array(:) = atomlist for thread "tid"

    ! Building atom list for thread 0:
    ! natom_thread(0,0) = position where thread 0 starts writing
    ! natom_thread(0,1) = position where thread 1 starts writing
    ! natom_thread(0,2) = position where thread 2 starts writing
    ! ...
    ! natom_thread(0,nthread) = number of atoms for thread 0

    ! Building atom list for thread 1:
    ! natom_thread(1,0) = position where thread 0 starts writing
    ! natom_thread(1,1) = position where thread 1 starts writing
    ! natom_thread(1,2) = position where thread 2 starts writing

    if (natom_thread(nthread)%array(tid) > size(atomlist_thread(tid)%array)) then
       call chmdealloc('colfft_func.src','build_atomlist_thread','atomlist_thread%array',&
            size(atomlist_thread(tid)%array),intg=atomlist_thread(tid)%array)
       call chmalloc('colfft_func.src','build_atomlist_thread','atomlist_thread%array',&
            ceiling(real(natom_thread(nthread)%array(tid))*1.5),intg=atomlist_thread(tid)%array)
    endif

!$omp barrier
    do i=jstart,jend
       thread_id = thread_id_list(i)
       natom_thread(tid)%array(thread_id) = natom_thread(tid)%array(thread_id) + 1
       atomlist_thread(thread_id)%array(natom_thread(tid)%array(thread_id)) = i
    enddo

    ! Now:
    ! atomlist_thread(tid)%array(:) = atoms that thread "tid" puts into charge grid

    return
  end subroutine build_atomlist_thread

  !----------------------------------------------------------------------------
  !----------------------------------------------------------------------------
  ! Testing routines
  !----------------------------------------------------------------------------
  !----------------------------------------------------------------------------

#if KEY_DOMDEC==1 /*domdec*/
  ! *
  ! * First part of test for grad_sumrc_ps, saves (dx, dy, dz) into
  ! * (dx_orig, dy_orig, dz_orig)
  ! *
  subroutine test1_grad_sumrc(natom, dx, dy, dz)
    use memory,only:chmalloc
    implicit none
    ! Input / Output
    integer, intent(in) :: natom
    real(chm_real), intent(in) :: dx(:), dy(:), dz(:)

    call chmalloc('colfft_func.src','test1_grad_sumrc_ps','dx_orig',natom,crl=dx_orig)
    call chmalloc('colfft_func.src','test1_grad_sumrc_ps','dy_orig',natom,crl=dy_orig)
    call chmalloc('colfft_func.src','test1_grad_sumrc_ps','dz_orig',natom,crl=dz_orig)

    dx_orig(1:natom) = dx(1:natom)
    dy_orig(1:natom) = dy(1:natom)
    dz_orig(1:natom) = dz(1:natom)

    return
  end subroutine test1_grad_sumrc
#endif /* (domdec)*/

end module colfft_func
