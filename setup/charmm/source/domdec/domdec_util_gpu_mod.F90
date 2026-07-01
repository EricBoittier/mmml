#if __PGI == 0
#define isnan4 isnan
#endif /* __PGI */

module domdec_util_gpu_mod
#if KEY_DOMDEC_GPU==1 /*domdec_util_gpu*/
  use, intrinsic :: iso_c_binding
  use domdec_local_types,only:xyzq_sp_t
  use domdec_bonded_types,only:list14_t
  implicit none

  private

  ! Public CUDA subroutines
  public sort_xyzq_on_gpu, wait_sort_xyzq_on_gpu, build_neighborlist_on_gpu
  public clear_force_virial_energy_gpu, reduce_force_virial_gpu, copy_force_virial_energy_from_gpu,&
       copy_vdwparam_to_gpu, copy_home_xyzq_to_gpu, copy_import_xyzq_to_gpu, copy_xyzq_to_gpu, &
       set_direct_ncoord_gpu, copy_vdwtype_to_gpu, copy_vdwparam14_to_gpu, &
       range_start, range_stop, copy_14_list_to_gpu, calc_14_force_gpu
  public wait_force_virial_energy_from_gpu
  public copy_blocktype_to_gpu, copy_blockparam_to_gpu, copy_bixlam_to_gpu, read_biflam_from_gpu, &
       copy_14_block_pos_to_gpu, copy_isitemld_to_gpu, copy_biflam_to_gpu, generate_rn_gpu
  
  ! Public Fortran subroutines
  public combine_force_from_gpu

  interface

     subroutine build_neighborlist_on_gpu(whichlist, zonelist_atom, cutnb, boxx, boxy, boxz) &
          bind(C,name="build_neighborlist_on_gpu")
       import
       integer(c_int), intent(in) :: whichlist, zonelist_atom(*)
       real(c_double), intent(in) :: cutnb, boxx, boxy, boxz
     end subroutine build_neighborlist_on_gpu
     
     subroutine sort_xyzq_on_gpu(whichlist, zonelist_atom, h_xyzq, h_loc2glo_ind) &
          bind(C,name="sort_xyzq_on_gpu")
       import
       integer(c_int), intent(in) :: whichlist, zonelist_atom(*)
       type(xyzq_sp_t), intent(inout) :: h_xyzq(*)
       integer(c_int), intent(inout) :: h_loc2glo_ind(*)       
     end subroutine sort_xyzq_on_gpu

     subroutine wait_sort_xyzq_on_gpu(whichlist) bind(C,name="wait_sort_xyzq_on_gpu")
       import
       integer(c_int), intent(in) :: whichlist       
     end subroutine wait_sort_xyzq_on_gpu

     subroutine clear_force_virial_energy_gpu(q_calc_energy, q_calc_virial) &
          bind(C,name="clear_force_virial_energy_gpu")
       import
       logical(c_int), intent(in) :: q_calc_energy, q_calc_virial
     end subroutine clear_force_virial_energy_gpu

     subroutine reduce_force_virial_gpu() bind(C,name="reduce_force_virial_gpu")
     end subroutine reduce_force_virial_gpu

     subroutine copy_force_virial_energy_from_gpu() bind(C,name="copy_force_virial_energy_from_gpu")
     end subroutine copy_force_virial_energy_from_gpu

     subroutine get_force_pointer(force_cptr) &
          bind(C,name="get_force_pointer")
       import
       type(c_ptr), intent(out) :: force_cptr
     end subroutine get_force_pointer

     subroutine get_force_stride(stride) bind(C,name="get_force_stride")
       import
       integer, intent(out) :: stride
     end subroutine get_force_stride

     subroutine get_force_type(ftype) bind(C,name="get_force_type")
       import
       integer, intent(out) :: ftype
     end subroutine get_force_type
     
     subroutine wait_force_virial_energy_from_gpu() bind(C,name="wait_force_virial_energy_from_gpu")
     end subroutine wait_force_virial_energy_from_gpu

     subroutine copy_vdwparam_to_gpu(h_nvdwparam, h_vdwparam) bind(C,name="copy_vdwparam_to_gpu")
       import
       integer(c_int), intent(in) :: h_nvdwparam
       real(c_float), intent(in) :: h_vdwparam(*)
     end subroutine copy_vdwparam_to_gpu

     subroutine copy_vdwparam14_to_gpu(h_nvdwparam, h_vdwparam) &
          bind(C,name="copy_vdwparam14_to_gpu")
       import
       integer(c_int), intent(in) :: h_nvdwparam
       real(c_float), intent(in) :: h_vdwparam(*)
     end subroutine copy_vdwparam14_to_gpu

     subroutine copy_14_list_to_gpu(nin14list, nex14list, h_in14list, h_ex14list) &
          bind(C,name="copy_14_list_to_gpu")
       import
       integer(c_int), intent(in) :: nin14list, nex14list
       type(list14_t), intent(in) :: h_in14list(*), h_ex14list(*)
     end subroutine copy_14_list_to_gpu

     subroutine calc_14_force_gpu(q_calc_energy, q_calc_virial, boxx, boxy, boxz) &
          bind(C,name="calc_14_force_gpu")
       import
       logical(c_int), intent(in) :: q_calc_energy, q_calc_virial
       real(c_double), intent(in) :: boxx, boxy, boxz
     end subroutine calc_14_force_gpu
     
     subroutine copy_home_xyzq_to_gpu(h_xyzq) bind(C,name="copy_home_xyzq_to_gpu")
       import
       type(xyzq_sp_t), intent(in) :: h_xyzq(*)
     end subroutine copy_home_xyzq_to_gpu

     subroutine copy_import_xyzq_to_gpu(h_xyzq) bind(C,name="copy_import_xyzq_to_gpu")
       import
       type(xyzq_sp_t), intent(in) :: h_xyzq(*)
     end subroutine copy_import_xyzq_to_gpu

     subroutine copy_xyzq_to_gpu(ncoord, h_x, h_y, h_z, h_q) bind(C,name="copy_xyzq_to_gpu")
       import
       integer, intent(in) :: ncoord
       real(c_double), intent(in) :: h_x(*), h_y(*), h_z(*), h_q(*)
     end subroutine copy_xyzq_to_gpu

     subroutine set_direct_ncoord_gpu(h_ncoord_home, h_ncoord_import) &
          bind(C,name="set_direct_ncoord_gpu")
       import
       integer(c_int), intent(in) :: h_ncoord_home, h_ncoord_import
     end subroutine set_direct_ncoord_gpu

     subroutine copy_vdwtype_to_gpu(h_vdwtype) bind(C,name="copy_vdwtype_to_gpu")
       import
       integer(c_int), intent(in) :: h_vdwtype(*)
     end subroutine copy_vdwtype_to_gpu

     subroutine range_start_padded(range_name) bind(C,name="range_start")
       import
       character(c_char), intent(in) :: range_name(*)
     end subroutine range_start_padded

     subroutine range_stop() bind(C,name="range_stop")
     end subroutine range_stop

     subroutine copy_blocktype_to_gpu(ncoord, h_blocktype) bind(C,name="copy_blocktype_to_gpu")
       import
       integer(c_int), intent(in) :: ncoord, h_blocktype(*)
     end subroutine copy_blocktype_to_gpu

     subroutine copy_blockparam_to_gpu(h_blockparam) bind(C,name="copy_blockparam_to_gpu")
       import
       real(c_float), intent(in) :: h_blockparam(*)
     end subroutine copy_blockparam_to_gpu

     subroutine copy_bixlam_to_gpu(h_bixlam) bind(C,name="copy_bixlam_to_gpu")
       import
       real(c_float), intent(in) :: h_bixlam(*)
     end subroutine copy_bixlam_to_gpu

     subroutine copy_biflam_to_gpu(h_biflam, h_biflam2) bind(C,name="copy_biflam_to_gpu")
       import
       real(c_double), intent(inout) :: h_biflam(*), h_biflam2(*)
     end subroutine copy_biflam_to_gpu

     subroutine copy_isitemld_to_gpu(h_isitemld) bind(C,name="copy_isitemld_to_gpu")
       import
       integer(c_int), intent(in) :: h_isitemld(*)
     end subroutine copy_isitemld_to_gpu

     subroutine read_biflam_from_gpu(h_biflam, h_biflam2) bind(C,name="read_biflam_from_gpu")
       import
       real(c_double), intent(inout) :: h_biflam(*), h_biflam2(*)
     end subroutine read_biflam_from_gpu

     subroutine copy_14_block_pos_to_gpu(in14tbl_block_pos, ex14tbl_block_pos) &
          bind(C,name="copy_14_block_pos_to_gpu")
       import
       integer(c_int), intent(in) :: in14tbl_block_pos(*), ex14tbl_block_pos(*)
     end subroutine copy_14_block_pos_to_gpu

     subroutine generate_rn_gpu(natom, gaussian_rn) bind(C,name="generate_rn_gpu")
       import
       integer(c_int), intent(in) :: natom
       real(c_float), intent(out) :: gaussian_rn(*)
     end subroutine generate_rn_gpu
     
  end interface

contains
  
  ! *
  ! * Combines forces from GPU into CPU forces
  ! * NOTE: This assumes forces are already copied to CPU
  ! *
  subroutine combine_force_from_gpu(forcex, forcey, forcez, nloc2glo_ind, loc2glo_ind)
    use chm_kinds
    use domdec_common,only:q_test
    implicit none
    ! Input / Output
    real(chm_real), intent(inout) :: forcex(*), forcey(*), forcez(*)
    integer, intent(in) :: nloc2glo_ind
    integer, intent(in), optional :: loc2glo_ind(*)
    ! Variables
    type(c_ptr) force_cptr
    real(chm_real), pointer, dimension(:) :: forcep_dp
    real(chm_real4), pointer, dimension(:) :: forcep_sp
    integer stride, stride2
    ! forcep_type: 0 = double precision, 1 = single precision
    integer forcep_type
    integer i, ind

    call get_force_pointer(force_cptr)
    call get_force_stride(stride)
    stride2 = 2*stride

    call get_force_type(forcep_type)

    if (forcep_type == 0) then
       call c_f_pointer(force_cptr, forcep_dp, [stride*3])
    else
       call c_f_pointer(force_cptr, forcep_sp, [stride*3])
    endif

    if (q_test) then
       if (forcep_type == 0) then
          call test_combine_dp_force_from_gpu(nloc2glo_ind, stride, forcep_dp)
       else
          call test_combine_sp_force_from_gpu(nloc2glo_ind, stride, forcep_sp)
       endif
    endif

    call range_start('Add GPU forces to CPU forces')
    if (forcep_type == 0) then
       if (present(loc2glo_ind)) then
!$omp parallel do schedule(static) private(i, ind)
          do i=1,nloc2glo_ind
             ind = loc2glo_ind(i)+1
             forcex(ind) = forcex(ind) + forcep_dp(i)
             forcey(ind) = forcey(ind) + forcep_dp(i+stride)
             forcez(ind) = forcez(ind) + forcep_dp(i+stride2)
          enddo
!$omp end parallel do
       else
!$omp parallel do schedule(static) private(i, ind)
          do i=1,nloc2glo_ind
             ind = i
             forcex(ind) = forcex(ind) + forcep_dp(i)
             forcey(ind) = forcey(ind) + forcep_dp(i+stride)
             forcez(ind) = forcez(ind) + forcep_dp(i+stride2)
          enddo
!$omp end parallel do
       endif
       nullify(forcep_dp)
    else
       if (present(loc2glo_ind)) then
!$omp parallel do schedule(static) private(i, ind)
          do i=1,nloc2glo_ind
             ind = loc2glo_ind(i)+1
             forcex(ind) = forcex(ind) + forcep_sp(i)
             forcey(ind) = forcey(ind) + forcep_sp(i+stride)
             forcez(ind) = forcez(ind) + forcep_sp(i+stride2)
          enddo
!$omp end parallel do
       else
!$omp parallel do schedule(static) private(i, ind)
          do i=1,nloc2glo_ind
             ind = i
             forcex(ind) = forcex(ind) + forcep_sp(i)
             forcey(ind) = forcey(ind) + forcep_sp(i+stride)
             forcez(ind) = forcez(ind) + forcep_sp(i+stride2)
          enddo
!$omp end parallel do
       endif
       nullify(forcep_sp)
    endif
    call range_stop()

    return
  end subroutine combine_force_from_gpu

  ! *
  ! * Test for combine_force_from_gpu sanity
  ! *
  subroutine test_combine_dp_force_from_gpu(ncoord, stride, forcep)
    use chm_kinds
    use stream,only:outu,prnlev
    implicit none
    ! Input
    integer, intent(in) :: ncoord, stride
    real(chm_real), pointer, dimension(:), intent(in) :: forcep
    ! Variables
    real(chm_real) max_force
    integer i, max_ind

    max_force = 0.0_chm_real
    max_ind = 0
    do i=1,ncoord
       if (isnan(forcep(i)) .or. isnan(forcep(i+stride)) .or. isnan(forcep(i+stride*2))) then
          max_ind = i
          max_force = 1000.0_chm_real
          exit
       endif
       if (max_force < abs(forcep(i))) then
          max_ind = i
          max_force = abs(forcep(i))
       endif
       if (max_force < abs(forcep(i+stride))) then
          max_ind = i
          max_force = abs(forcep(i+stride))
       endif
       if (max_force < abs(forcep(i+stride*2))) then
          max_ind = i
          max_force = abs(forcep(i+stride*2))
       endif
    enddo

    if (max_force > 200.0_chm_real) then
       write (outu,'(a,f20.3,i8)') 'max_force, max_ind=',max_force, max_ind
       write (outu,'(a,3f15.3)') 'force=',&
            forcep(max_ind), forcep(max_ind+stride), forcep(max_ind+stride*2)
       call wrndie(-5,'<domdec_util_gpu_mod>',&
            'test_combine_force_from_gpu: Force greater than 200.0')
    endif

    if (prnlev > 2) write (outu,'(a,f10.3)') 'test_combine_force_from_gpu OK, max_force=',max_force

    return
  end subroutine test_combine_dp_force_from_gpu

  ! *
  ! * Test for combine_force_from_gpu sanity
  ! *
  subroutine test_combine_sp_force_from_gpu(ncoord, stride, forcep)
    use chm_kinds
    use stream,only:outu,prnlev
    implicit none
    ! Input
    integer, intent(in) :: ncoord, stride
    real(chm_real4), pointer, dimension(:), intent(in) :: forcep
    ! Variables
    real(chm_real) max_force
    integer i, max_ind

    max_force = 0.0_chm_real
    max_ind = 0
    do i=1,ncoord
       if (isnan4(forcep(i)) .or. isnan4(forcep(i+stride)) .or. isnan4(forcep(i+stride*2))) then
          max_ind = i
          max_force = 1000.0_chm_real
          exit
       endif
       if (max_force < abs(forcep(i))) then
          max_ind = i
          max_force = abs(forcep(i))
       endif
       if (max_force < abs(forcep(i+stride))) then
          max_ind = i
          max_force = abs(forcep(i+stride))
       endif
       if (max_force < abs(forcep(i+stride*2))) then
          max_ind = i
          max_force = abs(forcep(i+stride*2))
       endif
    enddo

    if (max_force > 200.0_chm_real) then
       write (outu,'(a,f20.3,i8)') 'max_force, max_ind=',max_force, max_ind
       write (outu,'(a,3f15.3)') 'force=',&
            forcep(max_ind), forcep(max_ind+stride), forcep(max_ind+stride*2)
       call wrndie(-5,'<domdec_util_gpu_mod>',&
            'test_combine_force_from_gpu: Force greater than 200.0')
    endif

    if (prnlev > 2) write (outu,'(a,f10.3)') 'test_combine_force_from_gpu OK, max_force=',max_force

    return
  end subroutine test_combine_sp_force_from_gpu

#ifdef __PGI
  logical function isnan(a) result(f)
    use chm_kinds, only: chm_real
    implicit none
    real(chm_real),intent(in) :: a
    f = a /= a
    return
  end function isnan

  logical function isnan4(a) result(f)
    use chm_kinds, only: chm_real4
    implicit none
    real(chm_real4),intent(in) :: a
    f = a /= a
    return
  end function isnan4
#endif /* __PGI */

  subroutine range_start(range_name)
    use, intrinsic :: iso_c_binding, only: c_null_char
    character(len=*), intent(in) :: range_name
    call range_start_padded(range_name // c_null_char)
  end subroutine range_start

#endif /* (domdec_util_gpu)*/
end module domdec_util_gpu_mod
