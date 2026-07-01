module nblist_util

  !
  ! Utility functions for neighborlist builders
  !

#if KEY_DOMDEC==1 /*domdec*/
  use chm_kinds
  use dimens_fcm
  implicit none
  private

  ! Public subroutines
  public check_excl, pack_grouppos, pack_groups, &
       pack_vdwtype, &
       init_array, uninit_array, init_arrayp, uninit_arrayp, &
       reduce_buffer, concat_buffer, test_concat_buffer
  public cumsum_exclusive, cumsum_inclusive, test_cumsum
  public bucket_sort, test_bucket_sort
  public reduce_intarray
  
  interface bucket_sort
     module procedure bucket_sort_integer
  end interface

  interface pack_vdwtype
     module procedure pack_vdwtype_group
     module procedure pack_vdwtype_atom
  end interface

  interface concat_buffer
     module procedure concat_buffer_intarray
     module procedure concat_buffer_intarray2
  end interface

  interface init_array
     module procedure init_array_byte
     module procedure init_array_integer
     module procedure init_array_double_1d
     module procedure init_array_double_2d
     module procedure init_array_single_1d
     module procedure init_array_single_2d
     module procedure init_array_xyzq_sp
     module procedure init_array_xyzq_dp
     module procedure init_array_intarray
     module procedure init_array_intarray_2d
     module procedure init_array_intarray2
     module procedure init_array_cr4array
     module procedure init_array_crlarray
  end interface

  ! Pointer initialization put here to avoid compiler bug in gcc 4.7 (or so)
  interface init_arrayp
     module procedure init_array_integerp
  end interface

  interface uninit_array
     module procedure uninit_array_xyzq_sp
     module procedure uninit_array_integer
     module procedure uninit_array_intarray
     module procedure uninit_array_intarray_2d
     module procedure uninit_array_intarray2
     module procedure uninit_array_cr4array
     module procedure uninit_array_crlarray
  end interface

  interface uninit_arrayp
     module procedure uninit_array_integerp
  end interface

#if KEY_DOMDEC_GPU==1 /*domdec_gpu*/

  public init_array_gpu, uninit_array_gpu, alloc_gpu, dealloc_gpu, realloc_gpu

  ! *
  ! * init_array_gpu allocates pinned memory on CPU
  ! *
  interface init_array_gpu
     module procedure init_array_gpu_xyzq_sp
     module procedure init_array_gpu_integer_1d
     module procedure init_array_gpu_integer_2d
     module procedure init_array_gpu_intarray2
     module procedure init_array_gpu_double_1d
     module procedure init_array_gpu_single_1d
  end interface

  ! *
  ! * uninit_array_gpu deallocates pinned memory on CPU
  ! *
  interface uninit_array_gpu
     module procedure uninit_array_gpu_intarray2
  end interface

  interface alloc_gpu
     module procedure alloc_gpu_ientry
     module procedure alloc_gpu_tile_excl
     module procedure alloc_gpu_integer_1d
     module procedure alloc_gpu_integer_2d
     module procedure alloc_gpu_bondlist
     module procedure alloc_gpu_anglelist
     module procedure alloc_gpu_dihelist
     module procedure alloc_gpu_list14
  end interface

  interface dealloc_gpu
     module procedure dealloc_gpu_ientry
     module procedure dealloc_gpu_tile_excl
     module procedure dealloc_gpu_integer_1d
     module procedure dealloc_gpu_integer_2d
     module procedure dealloc_gpu_xyzq_sp
     module procedure dealloc_gpu_double_1d
     module procedure dealloc_gpu_single_1d
     module procedure dealloc_gpu_bondlist
     module procedure dealloc_gpu_anglelist
     module procedure dealloc_gpu_dihelist
     module procedure dealloc_gpu_list14
  end interface

  interface realloc_gpu
     module procedure realloc_gpu_integer_2d
  end interface

#endif /* (domdec_gpu)*/

  ! Block sum for cumulative summation routines
  integer, parameter :: MAX_NTHREAD = 256
  integer blockval(MAX_NTHREAD)

contains

  !------------------------------------------------------------------------------------
  !------------------------------------------------------------------------------------
  !------------------------------------------------------------------------------------
#if KEY_DOMDEC_GPU==1 /*domdec_gpu*/
  ! *
  ! * Initializes (allocates) and re-allocates coordinate array
  ! *
  subroutine init_array_gpu_xyzq_sp(array, narray)
    use memory
    use iso_c_binding,only:c_ptr, c_f_pointer, c_loc
    use domdec_local_types,only:xyzq_sp_t
    use domdec_gpu_mod,only:alloc_gpu_pinned_float4, dealloc_gpu_pinned_float4
    implicit none
    ! Input
    type(xyzq_sp_t), intent(inout), pointer, dimension(:) :: array
    integer, intent(in) :: narray
    ! Variables
    integer narray_new
    type(c_ptr) cptr

    if (associated(array)) then
       if (size(array) < narray) then
          cptr = c_loc(array(lbound(array,1)))
          call dealloc_gpu_pinned_float4(cptr)
          nullify(array)
       endif
    endif

    if (.not.associated(array)) then
       narray_new = int(narray*1.2)
       call alloc_gpu_pinned_float4(cptr, narray_new)
       call c_f_pointer(cptr, array, [narray_new])
    endif

    return
  end subroutine init_array_gpu_xyzq_sp

  ! *
  ! * Allocates 2d integer structure in pinned memory
  ! *
  subroutine alloc_gpu_integer_2d(array_inout, narray1, narray2)
    use iso_c_binding,only:c_ptr, c_f_pointer
    use domdec_gpu_mod,only:alloc_gpu_pinned_int
    implicit none
    ! Input / Output
    integer, intent(inout), pointer, dimension(:,:) :: array_inout
    integer, intent(in) :: narray1, narray2
    ! Variables
    type(c_ptr) cptr

    if (associated(array_inout)) then
       call wrndie(-5,'<nblist_util>','alloc_gpu_integer_2d: memory location already associated')
    else
       call alloc_gpu_pinned_int(cptr, narray1*narray2)
       call c_f_pointer(cptr, array_inout, [narray1, narray2])
    endif

    return
  end subroutine alloc_gpu_integer_2d

  ! *
  ! * Deallocates 2d integer structure in pinned memory
  ! *
  subroutine dealloc_gpu_integer_2d(array_inout)
    use iso_c_binding,only:c_ptr, c_loc
    use domdec_gpu_mod,only:dealloc_gpu_pinned_int
    implicit none
    ! Input / Output
    integer, intent(inout), pointer, dimension(:,:) :: array_inout
    ! Variables
    type(c_ptr) cptr

    if (.not.associated(array_inout)) then
       call wrndie(-5,'<nblist_util>','dealloc_gpu_integer_2d: memory location not associated')
    else
       cptr = c_loc(array_inout(lbound(array_inout,1),lbound(array_inout,2)))
       call dealloc_gpu_pinned_int(cptr)
       nullify(array_inout)
    endif

    return
  end subroutine dealloc_gpu_integer_2d

  ! *
  ! * Reallocates 2d integer structure in pinned memory
  ! *
  subroutine realloc_gpu_integer_2d(array_inout, narray1, narray2)
    use memory,only:chmalloc,chmdealloc
    implicit none
    ! Input / Output
    integer, intent(inout), pointer, dimension(:,:) :: array_inout
    integer, intent(in) :: narray1, narray2
    ! Variables
    integer, allocatable, dimension(:,:) :: array_tmp
    integer narray1_old, narray2_old
    integer narray1_copy, narray2_copy

    narray1_old = size(array_inout,1)
    narray2_old = size(array_inout,2)
    narray1_copy = min(narray1_old, narray1)
    narray2_copy = min(narray2_old, narray2)

    ! Allocate temporary buffer
    call chmalloc('nblist_util.src','realloc_gpu_integer_2d','array_tmp',&
         narray1_copy,narray2_copy,intg=array_tmp)

    ! Copy to temporary buffer
    array_tmp(1:narray1_copy,1:narray2_copy) = array_inout(1:narray1_copy,1:narray2_copy)

    ! Deallocate old buffer
    call dealloc_gpu(array_inout)

    ! Allocate new buffer
    call alloc_gpu(array_inout, narray1, narray2)

    ! Copy to new buffer
    array_inout(1:narray1_copy,1:narray2_copy) = array_tmp(1:narray1_copy,1:narray2_copy)

    ! Deallocate temporary buffer
    call chmdealloc('nblist_util.src','realloc_gpu_integer_2d','array_tmp',&
         narray1_copy,narray2_copy,intg=array_tmp)

    return
  end subroutine realloc_gpu_integer_2d

  ! *
  ! * Allocates ientry_t structure in pinned memory
  ! *
  subroutine alloc_gpu_ientry(array_inout, narray)
    use nblist_types,only:ientry_t
    use iso_c_binding,only:c_ptr, c_f_pointer
    use domdec_gpu_mod,only:alloc_gpu_pinned_ientry
    implicit none
    ! Input / Output
    type(ientry_t), intent(inout), pointer, dimension(:) :: array_inout
    integer, intent(in) :: narray
    ! Variables
    type(c_ptr) cptr

    if (associated(array_inout)) then
       call wrndie(-5,'<nblist_util>','alloc_gpu_ientry: memory location already associated')
    else
       call alloc_gpu_pinned_ientry(cptr, narray)
       call c_f_pointer(cptr, array_inout, [narray])
    endif

    return
  end subroutine alloc_gpu_ientry

  ! *
  ! * Deallocates ientry_t structure in pinned memory
  ! *
  subroutine dealloc_gpu_ientry(array_inout)
    use nblist_types,only:ientry_t
    use iso_c_binding,only:c_ptr, c_loc
    use domdec_gpu_mod,only:dealloc_gpu_pinned_ientry
    implicit none
    ! Input / Output
    type(ientry_t), intent(inout), pointer, dimension(:) :: array_inout
    ! Variables
    type(c_ptr) cptr

    if (.not.associated(array_inout)) then
       call wrndie(-5,'<nblist_util>','dealloc_gpu_ientry: memory location not associated')
    else
       cptr = c_loc(array_inout(lbound(array_inout,1)))
       call dealloc_gpu_pinned_ientry(cptr)
       nullify(array_inout)
    endif

    return
  end subroutine dealloc_gpu_ientry

  ! *
  ! * Allocates tile_excl_t structure in pinned memory
  ! *
  subroutine alloc_gpu_tile_excl(array_inout, narray)
    use nblist_types,only:tile_excl_t
    use iso_c_binding,only:c_ptr, c_f_pointer
    use domdec_gpu_mod,only:alloc_gpu_pinned_tile_excl
    implicit none
    ! Input / Output
    type(tile_excl_t), intent(inout), pointer, dimension(:) :: array_inout
    integer, intent(in) :: narray
    ! Variables
    type(c_ptr) cptr

    if (associated(array_inout)) then
       call wrndie(-5,'<nblist_util>','alloc_gpu_tile_excl: memory location already associated')
    else
       call alloc_gpu_pinned_tile_excl(cptr, narray)
       call c_f_pointer(cptr, array_inout, [narray])
    endif

    return
  end subroutine alloc_gpu_tile_excl

  ! *
  ! * Deallocates tile_excl_t structure in pinned memory
  ! *
  subroutine dealloc_gpu_tile_excl(array_inout)
    use nblist_types,only:tile_excl_t
    use iso_c_binding,only:c_ptr, c_loc
    use domdec_gpu_mod,only:dealloc_gpu_pinned_tile_excl
    implicit none
    ! Input / Output
    type(tile_excl_t), intent(inout), pointer, dimension(:) :: array_inout
    ! Variables
    type(c_ptr) cptr

    if (.not.associated(array_inout)) then
       call wrndie(-5,'<nblist_util>','dealloc_gpu_tile_excl: memory location not associated')
    else
       cptr = c_loc(array_inout(lbound(array_inout,1)))
       call dealloc_gpu_pinned_tile_excl(cptr)
       nullify(array_inout)
    endif

    return
  end subroutine dealloc_gpu_tile_excl

  ! *
  ! * Deallocates xyzq_sp_t structure in pinned memory
  ! *
  subroutine dealloc_gpu_xyzq_sp(array_inout)
    use domdec_local_types,only:xyzq_sp_t
    use iso_c_binding,only:c_ptr, c_loc
    use domdec_gpu_mod,only:dealloc_gpu_pinned_float4
    implicit none
    ! Input / Output
    type(xyzq_sp_t), intent(inout), pointer, dimension(:) :: array_inout
    ! Variables
    type(c_ptr) cptr

    if (.not.associated(array_inout)) then
       call wrndie(-5,'<nblist_util>','dealloc_gpu_xyzq_sp_t: memory location not associated')
    else
       cptr = c_loc(array_inout(lbound(array_inout,1)))
       call dealloc_gpu_pinned_float4(cptr)
       nullify(array_inout)
    endif

    return
  end subroutine dealloc_gpu_xyzq_sp

  ! *
  ! * Allocates ientry_t structure in pinned memory
  ! *
  subroutine alloc_gpu_integer_1d(array_inout, narray)
    use iso_c_binding,only:c_ptr, c_f_pointer
    use domdec_gpu_mod,only:alloc_gpu_pinned_int
    implicit none
    ! Input / Output
    integer, intent(inout), pointer, dimension(:) :: array_inout
    integer, intent(in) :: narray
    ! Variables
    type(c_ptr) cptr

    if (associated(array_inout)) then
       call wrndie(-5,'<nblist_util>','alloc_gpu_integer_1d: memory location already associated')
    else
       call alloc_gpu_pinned_int(cptr, narray)
       call c_f_pointer(cptr, array_inout, [narray])
    endif

    return
  end subroutine alloc_gpu_integer_1d

  ! *
  ! * Deallocates integer array in pinned memory
  ! *
  subroutine dealloc_gpu_integer_1d(array_inout)
    use iso_c_binding,only:c_ptr, c_loc
    use domdec_gpu_mod,only:dealloc_gpu_pinned_int
    implicit none
    ! Input / Output
    integer, intent(inout), pointer, dimension(:) :: array_inout
    ! Variables
    type(c_ptr) cptr

    if (.not.associated(array_inout)) then
       call wrndie(-5,'<nblist_util>','dealloc_gpu_integer_1d: memory location not associated')
    else
       cptr = c_loc(array_inout(lbound(array_inout,1)))
       call dealloc_gpu_pinned_int(cptr)
       nullify(array_inout)
    endif

    return
  end subroutine dealloc_gpu_integer_1d

  ! *
  ! * Deallocates double array in pinned memory
  ! *
  subroutine dealloc_gpu_double_1d(array_inout)
    use iso_c_binding,only:c_ptr, c_loc
    use domdec_gpu_mod,only:dealloc_gpu_pinned_double
    implicit none
    ! Input / Output
    real(chm_real), intent(inout), pointer, dimension(:) :: array_inout
    ! Variables
    type(c_ptr) cptr

    if (.not.associated(array_inout)) then
       call wrndie(-5,'<nblist_util>','dealloc_gpu_double_1d: memory location not associated')
    else
       cptr = c_loc(array_inout(lbound(array_inout,1)))
       call dealloc_gpu_pinned_double(cptr)
       nullify(array_inout)
    endif

    return
  end subroutine dealloc_gpu_double_1d

  ! *
  ! * Deallocates double array in pinned memory
  ! *
  subroutine dealloc_gpu_single_1d(array_inout)
    use iso_c_binding,only:c_ptr, c_loc
    use domdec_gpu_mod,only:dealloc_gpu_pinned_float
    implicit none
    ! Input / Output
    real(chm_real4), intent(inout), pointer, dimension(:) :: array_inout
    ! Variables
    type(c_ptr) cptr

    if (.not.associated(array_inout)) then
       call wrndie(-5,'<nblist_util>','dealloc_gpu_single_1d: memory location not associated')
    else
       cptr = c_loc(array_inout(lbound(array_inout,1)))
       call dealloc_gpu_pinned_float(cptr)
       nullify(array_inout)
    endif

    return
  end subroutine dealloc_gpu_single_1d

  ! #############################################################################################
  ! #############################################################################################
  ! #############################################################################################

  ! *
  ! * Allocates bondlist_t structure in pinned memory
  ! *
  subroutine alloc_gpu_bondlist(array_inout, narray)
    use domdec_bonded_types,only:bondlist_t, sizeof_bondlist_t
    use iso_c_binding,only:c_ptr, c_f_pointer
    use domdec_gpu_mod,only:alloc_gpu_pinned_char
    implicit none
    ! Input / Output
    type(bondlist_t), intent(inout), pointer, dimension(:) :: array_inout
    integer, intent(in) :: narray
    ! Variables
    type(c_ptr) cptr

    if (associated(array_inout)) then
       call wrndie(-5,'<nblist_util>','alloc_gpu_bondlist: memory location already associated')
    else
       call alloc_gpu_pinned_char(cptr, narray*sizeof_bondlist_t)
       call c_f_pointer(cptr, array_inout, [narray])
    endif

    return
  end subroutine alloc_gpu_bondlist

  ! *
  ! * Deallocates bondlist structure in pinned memory
  ! *
  subroutine dealloc_gpu_bondlist(array_inout)
    use domdec_bonded_types,only:bondlist_t
    use iso_c_binding,only:c_ptr, c_loc
    use domdec_gpu_mod,only:dealloc_gpu_pinned_char
    implicit none
    ! Input / Output
    type(bondlist_t), intent(inout), pointer, dimension(:) :: array_inout
    ! Variables
    type(c_ptr) cptr

    if (.not.associated(array_inout)) then
       call wrndie(-5,'<nblist_util>','dealloc_gpu_bondlist: memory location not associated')
    else
       cptr = c_loc(array_inout(lbound(array_inout,1)))
       call dealloc_gpu_pinned_char(cptr)
       nullify(array_inout)
    endif

    return
  end subroutine dealloc_gpu_bondlist

  ! *
  ! * Allocates anglelist_t structure in pinned memory
  ! *
  subroutine alloc_gpu_anglelist(array_inout, narray)
    use domdec_bonded_types,only:anglelist_t, sizeof_anglelist_t
    use iso_c_binding,only:c_ptr, c_f_pointer
    use domdec_gpu_mod,only:alloc_gpu_pinned_char
    implicit none
    ! Input / Output
    type(anglelist_t), intent(inout), pointer, dimension(:) :: array_inout
    integer, intent(in) :: narray
    ! Variables
    type(c_ptr) cptr

    if (associated(array_inout)) then
       call wrndie(-5,'<nblist_util>','alloc_gpu_anglelist: memory location already associated')
    else
       call alloc_gpu_pinned_char(cptr, narray*sizeof_anglelist_t)
       call c_f_pointer(cptr, array_inout, [narray])
    endif

    return
  end subroutine alloc_gpu_anglelist

  ! *
  ! * Deallocates anglelist structure in pinned memory
  ! *
  subroutine dealloc_gpu_anglelist(array_inout)
    use domdec_bonded_types,only:anglelist_t
    use iso_c_binding,only:c_ptr, c_loc
    use domdec_gpu_mod,only:dealloc_gpu_pinned_char
    implicit none
    ! Input / Output
    type(anglelist_t), intent(inout), pointer, dimension(:) :: array_inout
    ! Variables
    type(c_ptr) cptr

    if (.not.associated(array_inout)) then
       call wrndie(-5,'<nblist_util>','dealloc_gpu_anglelist: memory location not associated')
    else
       cptr = c_loc(array_inout(lbound(array_inout,1)))
       call dealloc_gpu_pinned_char(cptr)
       nullify(array_inout)
    endif

    return
  end subroutine dealloc_gpu_anglelist

  ! *
  ! * Allocates dihelist_t structure in pinned memory
  ! *
  subroutine alloc_gpu_dihelist(array_inout, narray)
    use domdec_bonded_types,only:dihelist_t, sizeof_dihelist_t
    use iso_c_binding,only:c_ptr, c_f_pointer
    use domdec_gpu_mod,only:alloc_gpu_pinned_char
    implicit none
    ! Input / Output
    type(dihelist_t), intent(inout), pointer, dimension(:) :: array_inout
    integer, intent(in) :: narray
    ! Variables
    type(c_ptr) cptr

    if (associated(array_inout)) then
       call wrndie(-5,'<nblist_util>','alloc_gpu_dihelist: memory location already associated')
    else
       call alloc_gpu_pinned_char(cptr, narray*sizeof_dihelist_t)
       call c_f_pointer(cptr, array_inout, [narray])
    endif

    return
  end subroutine alloc_gpu_dihelist

  ! *
  ! * Deallocates dihelist structure in pinned memory
  ! *
  subroutine dealloc_gpu_dihelist(array_inout)
    use domdec_bonded_types,only:dihelist_t
    use iso_c_binding,only:c_ptr, c_loc
    use domdec_gpu_mod,only:dealloc_gpu_pinned_char
    implicit none
    ! Input / Output
    type(dihelist_t), intent(inout), pointer, dimension(:) :: array_inout
    ! Variables
    type(c_ptr) cptr

    if (.not.associated(array_inout)) then
       call wrndie(-5,'<nblist_util>','dealloc_gpu_dihelist: memory location not associated')
    else
       cptr = c_loc(array_inout(lbound(array_inout,1)))
       call dealloc_gpu_pinned_char(cptr)
       nullify(array_inout)
    endif

    return
  end subroutine dealloc_gpu_dihelist

  ! *
  ! * Allocates list14_t structure in pinned memory
  ! *
  subroutine alloc_gpu_list14(array_inout, narray)
    use domdec_bonded_types,only:list14_t, sizeof_list14_t
    use iso_c_binding,only:c_ptr, c_f_pointer
    use domdec_gpu_mod,only:alloc_gpu_pinned_char
    implicit none
    ! Input / Output
    type(list14_t), intent(inout), pointer, dimension(:) :: array_inout
    integer, intent(in) :: narray
    ! Variables
    type(c_ptr) cptr

    if (associated(array_inout)) then
       call wrndie(-5,'<nblist_util>','alloc_gpu_list14: memory location already associated')
    else
       call alloc_gpu_pinned_char(cptr, narray*sizeof_list14_t)
       call c_f_pointer(cptr, array_inout, [narray])
    endif

    return
  end subroutine alloc_gpu_list14

  ! *
  ! * Deallocates list14 structure in pinned memory
  ! *
  subroutine dealloc_gpu_list14(array_inout)
    use domdec_bonded_types,only:list14_t
    use iso_c_binding,only:c_ptr, c_loc
    use domdec_gpu_mod,only:dealloc_gpu_pinned_char
    implicit none
    ! Input / Output
    type(list14_t), intent(inout), pointer, dimension(:) :: array_inout
    ! Variables
    type(c_ptr) cptr

    if (.not.associated(array_inout)) then
       call wrndie(-5,'<nblist_util>','dealloc_gpu_list14: memory location not associated')
    else
       cptr = c_loc(array_inout(lbound(array_inout,1)))
       call dealloc_gpu_pinned_char(cptr)
       nullify(array_inout)
    endif

    return
  end subroutine dealloc_gpu_list14


  ! #############################################################################################
  ! #############################################################################################
  ! #############################################################################################

  ! *
  ! * Initializes (allocates) and re-allocates integer array in pinned memory.
  ! * Allocates *extraf_in (default 1.2) amount of extra space
  ! *
  subroutine init_array_gpu_integer_1d(array, narray, extraf_in)
    use memory
    use iso_c_binding,only:c_ptr, c_f_pointer, c_loc
    use domdec_gpu_mod,only:alloc_gpu_pinned_int, dealloc_gpu_pinned_int
    implicit none
    ! Input
    integer, intent(inout), pointer, dimension(:) :: array
    integer, intent(in) :: narray
    real(chm_real), intent(in), optional :: extraf_in
    ! Variables
    real(chm_real) extraf
    integer narray_new
    type(c_ptr) cptr

    if (present(extraf_in)) then
       extraf = extraf_in
    else
       extraf = 1.2
    endif

    if (associated(array)) then
       if (size(array) < narray) then
          cptr = c_loc(array(lbound(array,1)))
          call dealloc_gpu_pinned_int(cptr)
          nullify(array)
       endif
    endif

    if (.not.associated(array)) then
       narray_new = int(narray*extraf)
       call alloc_gpu_pinned_int(cptr, narray_new)
       call c_f_pointer(cptr, array, [narray_new])
    endif

    return
  end subroutine init_array_gpu_integer_1d

  ! *
  ! * Initializes (allocates) and re-allocates integer array in pinned memory.
  ! * Allocates *extraf_in (default 1.2) amount of extra space
  ! *
  subroutine init_array_gpu_integer_2d(array, narray1, narray2, extraf_in)
    use memory
    use iso_c_binding,only:c_ptr, c_f_pointer, c_loc
    use domdec_gpu_mod,only:alloc_gpu_pinned_int, dealloc_gpu_pinned_int
    implicit none
    ! Input
    integer, intent(inout), pointer, dimension(:,:) :: array
    integer, intent(in) :: narray1, narray2
    real(chm_real), intent(in), optional :: extraf_in
    ! Variables
    real(chm_real) extraf
    integer narray2_new
    type(c_ptr) cptr

    if (present(extraf_in)) then
       extraf = extraf_in
    else
       extraf = 1.2
    endif

    if (associated(array)) then
       if (size(array,2) < narray2) then
          cptr = c_loc(array(lbound(array,1),lbound(array,2)))
          call dealloc_gpu_pinned_int(cptr)
          nullify(array)
       endif
    endif

    if (.not.associated(array)) then
       narray2_new = int(narray2*extraf)
       call alloc_gpu_pinned_int(cptr, narray1*narray2_new)
       call c_f_pointer(cptr, array, [narray1, narray2_new])
    endif

    return
  end subroutine init_array_gpu_integer_2d

  ! *
  ! * Initializes (allocates) and re-allocates integer array in pinned memory
  ! *
  subroutine init_array_gpu_intarray2(array_inout, narray1, narray2, narray3, narray3_new_in)
    use memory
    use iso_c_binding,only:c_ptr, c_f_pointer, c_loc
    use domdec_gpu_mod,only:alloc_gpu_pinned_int, dealloc_gpu_pinned_int
    use nblist_types,only:intarray2p_t
    implicit none
    ! Input
    type(intarray2p_t), intent(inout), allocatable, dimension(:) :: array_inout
    integer, intent(in) :: narray1, narray2, narray3
    integer, intent(in), optional :: narray3_new_in
    ! Variables
    integer narray3_new, i
    type(c_ptr) cptr


    if (allocated(array_inout) .and. size(array_inout) < narray1) then
       deallocate(array_inout)
    endif

    if (.not.allocated(array_inout)) then
       allocate(array_inout(0:narray1-1))
       do i=0,narray1-1
          nullify(array_inout(i)%array)
       enddo
    endif

    if (present(narray3_new_in)) then
       narray3_new = narray3_new_in
    else
       narray3_new = int(narray3*1.1)
    endif

    do i=0,narray1-1
       if (associated(array_inout(i)%array) .and. &
            (size(array_inout(i)%array,1) < narray2 .or. size(array_inout(i)%array,2) < narray3)) then
          cptr = c_loc(array_inout(i)%array(lbound(array_inout(i)%array,1),&
               lbound(array_inout(i)%array,2)))
          call dealloc_gpu_pinned_int(cptr)
          nullify(array_inout(i)%array)
       endif
       
       if (.not.associated(array_inout(i)%array)) then
          call alloc_gpu_pinned_int(cptr, narray2*narray3_new)
          call c_f_pointer(cptr, array_inout(i)%array, [narray2, narray3_new])
       endif
    enddo

    return
  end subroutine init_array_gpu_intarray2

  ! *
  ! * Initializes (allocates) and re-allocates double array in pinned memory.
  ! * Allocates *extraf_in (default 1.2) amount of extra space
  ! *
  subroutine init_array_gpu_double_1d(array, narray, extraf_in)
    use memory
    use iso_c_binding,only:c_ptr, c_f_pointer, c_loc
    use domdec_gpu_mod,only:alloc_gpu_pinned_double, dealloc_gpu_pinned_double
    implicit none
    ! Input
    real(chm_real), intent(inout), pointer, dimension(:) :: array
    integer, intent(in) :: narray
    real(chm_real), intent(in), optional :: extraf_in
    ! Variables
    real(chm_real) extraf
    integer narray_new
    type(c_ptr) cptr

    if (present(extraf_in)) then
       extraf = extraf_in
    else
       extraf = 1.2
    endif

    if (associated(array)) then
       if (size(array) < narray) then
          cptr = c_loc(array(lbound(array,1)))
          call dealloc_gpu_pinned_double(cptr)
          nullify(array)
       endif
    endif

    if (.not.associated(array)) then
       narray_new = int(narray*extraf)
       call alloc_gpu_pinned_double(cptr, narray_new)
       call c_f_pointer(cptr, array, [narray_new])
    endif

    return
  end subroutine init_array_gpu_double_1d

  ! *
  ! * Initializes (allocates) and re-allocates single array in pinned memory.
  ! * Allocates *extraf_in (default 1.2) amount of extra space
  ! *
  subroutine init_array_gpu_single_1d(array, narray, extraf_in)
    use memory
    use iso_c_binding,only:c_ptr, c_f_pointer, c_loc
    use domdec_gpu_mod,only:alloc_gpu_pinned_float, dealloc_gpu_pinned_float
    implicit none
    ! Input
    real(chm_real4), intent(inout), pointer, dimension(:) :: array
    integer, intent(in) :: narray
    real(chm_real), intent(in), optional :: extraf_in
    ! Variables
    real(chm_real) extraf
    integer narray_new
    type(c_ptr) cptr

    if (present(extraf_in)) then
       extraf = extraf_in
    else
       extraf = 1.2
    endif

    if (associated(array)) then
       if (size(array) < narray) then
          cptr = c_loc(array(lbound(array,1)))
          call dealloc_gpu_pinned_float(cptr)
          nullify(array)
       endif
    endif

    if (.not.associated(array)) then
       narray_new = int(narray*extraf)
       call alloc_gpu_pinned_float(cptr, narray_new)
       call c_f_pointer(cptr, array, [narray_new])
    endif

    return
  end subroutine init_array_gpu_single_1d

  ! *
  ! * Uninitializes (deallocates) integer array in pinned memory
  ! *
  subroutine uninit_array_gpu_intarray2(array_inout)
    use memory
    use iso_c_binding,only:c_ptr, c_f_pointer, c_loc
    use domdec_gpu_mod,only:dealloc_gpu_pinned_int
    use nblist_types,only:intarray2p_t
    implicit none
    ! Input
    type(intarray2p_t), intent(inout), allocatable, dimension(:) :: array_inout
    ! Variables
    integer i
    type(c_ptr) cptr

    if (allocated(array_inout)) then
       do i=0,size(array_inout)-1
          if (associated(array_inout(i)%array)) then
             cptr = c_loc(array_inout(i)%array(lbound(array_inout(i)%array,1),&
                  lbound(array_inout(i)%array,2)))
             call dealloc_gpu_pinned_int(cptr)
             nullify(array_inout(i)%array)
          endif
       enddo
       deallocate(array_inout)
    endif

    return
  end subroutine uninit_array_gpu_intarray2

#endif /* (domdec_gpu)*/

  ! *
  ! * Sorts list of integers key_array(1:n) using bucket sort.
  ! * The new ordering is given in order_array(1:n)
  ! *
  subroutine bucket_sort_integer(n, key_array, order_array, bucket, fsize, min_key_in, max_key_in)
    use domdec_common,only:nthread
    use memory
    implicit none
    ! Input / Output
    integer, intent(in) :: n
    integer, intent(in) :: key_array(:)
    integer, intent(out) :: order_array(:)
    integer, allocatable, dimension(:,:), intent(inout) :: bucket
    real(chm_real), intent(in) :: fsize
    integer, optional, intent(in) :: min_key_in, max_key_in
    ! Functions
#ifdef _OPENMP
    integer omp_get_thread_num
#endif
    ! Variables
    integer nbucket
    integer key, min_key, max_key
    integer ibucket, val, tmp, pos
    integer tid
    integer i, j

    ! Find minimum and maximum key value, if needed
    if (present(min_key_in) .and. present(max_key_in)) then
       min_key = min_key_in
       max_key = max_key_in
    else
       min_key = 100000000
       max_key = 0
!$omp parallel do schedule(static) private(i, key) reduction(min:min_key) reduction(max:max_key)
       do i=1,n
          key = key_array(i)
          min_key = min(key, min_key)
          max_key = max(key, max_key)
       enddo
!$omp end parallel do
    endif

    nbucket = max_key - min_key + 1

    ! Allocate & Re-allocate bucket(1:nbucket,0:nthread)
    if (allocated(bucket)) then
       if (size(bucket,1) < nbucket .or. size(bucket,2) < nthread+1) then
          call chmdealloc('nblist_util.src','bucket_sort_integer','bucket',&
               size(bucket,1),size(bucket,2),intg=bucket)
       endif
    endif

    if (.not.allocated(bucket)) then
       call chmalloc('nblist_util.src','bucket_sort_integer','bucket',&
            int(nbucket*max(1.0_chm_real,fsize)),nthread+1,lbou2=0,intg=bucket)
    endif

!$omp parallel private(tid)
#ifdef _OPENMP
    tid = omp_get_thread_num()
#else
    tid = 0
#endif
    bucket(1:nbucket, tid) = 0
!$omp do private(i, ibucket) schedule(static)
    do i=1,n
       ibucket = key_array(i) - min_key + 1
       bucket(ibucket,tid) = bucket(ibucket,tid) + 1
    enddo
!$omp end do
!$omp end parallel

    ! bucket(1,tid) = number of entries thread "tid" found for bucket 1
    ! bucket(2,tid) = number of entries thread "tid" found for bucket 2
    ! ...

!$omp parallel do schedule(static) private(i, val, j, tmp)
    do i=1,nbucket
       ! Do exclusive cumulative sum on bucket(i,0:nthread)
       val = bucket(i,0)
       bucket(i,0) = 0
       do j=1,nthread
          tmp = bucket(i,j)
          bucket(i,j) = val + bucket(i,j-1)
          val = tmp
       enddo
    enddo
!$omp end parallel do

    ! For tid = 0...nthread-1
    ! bucket(1,tid) = position thread "tid" has in bucket 1
    ! bucket(2,tid) = position thread "tid" has in bucket 2
    ! ...

    ! bucket(1,nthread) = number of entries in bucket 1
    ! bucket(2,nthread) = number of entries in bucket 2
    ! ...

    call cumsum_exclusive(bucket(:,nthread), nbucket)

    ! bucket(1,nthread) = starting position for bucket 1
    ! bucket(2,nthread) = starting position for bucket 2
    ! ...

!$omp parallel private(tid)
#ifdef _OPENMP
    tid = omp_get_thread_num()
#else
    tid = 0
#endif    
!$omp do schedule(static) private(i, ibucket, pos)
    do i=1,n
       ibucket = key_array(i) - min_key + 1
       bucket(ibucket,tid) = bucket(ibucket,tid) + 1
       pos = bucket(ibucket,nthread) + bucket(ibucket,tid)
       order_array(pos) = i
    enddo
!$omp end do
!$omp end parallel

    return
  end subroutine bucket_sort_integer

  ! *
  ! * Tests bucket_sort -routine
  ! *
  subroutine test_bucket_sort()
    use stream,only:outu,prnlev
    use memory
    implicit none
    ! Parameters
    integer, parameter :: N=10000
    ! Variables
    real(chm_real) r
    integer, allocatable, dimension(:) :: keys, order, seed
    integer, allocatable, dimension(:,:) :: bucket
    integer seed_len, i

    call chmalloc('nblist_util.src','test_bucket_sort','keys',N,intg=keys)
    call chmalloc('nblist_util.src','test_bucket_sort','order',N,intg=order)

    call random_seed(size=seed_len)
    call chmalloc('nblist_util.src','test_bucket_sort','seed',seed_len,intg=seed)
    seed(1:seed_len) = 87342392
    call random_seed(put=seed)

    do i=1,N
       call random_number(r)
       keys(i) = int(r*400)
    enddo

    call bucket_sort_integer(N, keys, order, bucket, 1.0_chm_real)
    call test_order()

    do i=1,N
       call random_number(r)
       keys(i) = int(r*400)
    enddo

    call bucket_sort_integer(N, keys, order, bucket, 1.0_chm_real, minval(keys), maxval(keys))
    call test_order()

    call chmdealloc('nblist_util.src','test_bucket_sort','bucket',&
         size(bucket,1),size(bucket,2),intg=bucket)

    call chmdealloc('nblist_util.src','test_bucket_sort','seed',seed_len,intg=seed)
    call chmdealloc('nblist_util.src','test_bucket_sort','keys',N,intg=keys)
    call chmdealloc('nblist_util.src','test_bucket_sort','order',N,intg=order)

    if (prnlev > 2) write (outu,'(a)') 'test_bucket_sort OK'

    return
  contains

    subroutine test_order()
      implicit none
      integer i

      do i=1,N-1
         if (keys(order(i+1)) < keys(order(i))) then
            call wrndie(-5,'<nblist_util>','test_bucket_sort FAILED')
         endif
      enddo

      return
    end subroutine test_order

  end subroutine test_bucket_sort

  ! *
  ! * Reduces (sums) buffers:
  ! *
  ! * buffer(1      )%array(1:buffer_len) = 
  ! *
  ! * buffer(1      )%array(1:buffer_len) +
  ! * buffer(2      )%array(1:buffer_len) +
  ! * ...
  ! * buffer(nbuffer)%array(1:buffer_len)
  ! *
  subroutine reduce_buffer(nbuffer, buffer_len, buffer)
    use nblist_types,only:intarray_t
    implicit none
    ! Input / Output
    integer, intent(in) :: nbuffer, buffer_len
    type(intarray_t), intent(inout) :: buffer(1:nbuffer)
    ! Variables
    integer i, j

!$omp parallel do private(i, j)
    do i=1,buffer_len
       do j=2,nbuffer
          buffer(1)%array(i) = buffer(1)%array(i) + buffer(j)%array(i)
       enddo
    enddo
!$omp end parallel do

  end subroutine reduce_buffer

  ! *
  ! * Concatenate integer buffer
  ! *
  subroutine concat_buffer_intarray(nbuffer, buffer_len, buffer)
    use nblist_types,only:intarray_t
    implicit none
    ! Input / Output
    integer, intent(in) :: nbuffer
    integer, intent(inout) :: buffer_len(1:nbuffer)
    type(intarray_t), intent(inout) :: buffer(1:nbuffer)
    ! Variables
    integer i

    ! Calculate starting positions
    call cumsum_inclusive(buffer_len, nbuffer)

!$comp parallel do
    do i=2,nbuffer
       buffer(1)%array(buffer_len(i-1)+1:buffer_len(i)) = &
            buffer(i)%array(1:buffer_len(i)-buffer_len(i-1))
    enddo
!$comp end parallel do

    return
  end subroutine concat_buffer_intarray

  ! *
  ! * Concatenate integer buffer
  ! *
  subroutine concat_buffer_intarray2(nbuffer, buffer_len, buffer)
    use nblist_types,only:intarray2_t
    implicit none
    ! Input / Output
    integer, intent(in) :: nbuffer
    integer, intent(inout) :: buffer_len(1:nbuffer)
    type(intarray2_t), intent(inout) :: buffer(1:nbuffer)
    ! Variables
    integer i

    ! Calculate starting positions
    call cumsum_inclusive(buffer_len, nbuffer)

!$omp parallel do private(i)
    do i=2,nbuffer
       buffer(1)%array(:,buffer_len(i-1)+1:buffer_len(i)) = &
            buffer(i)%array(:,1:buffer_len(i)-buffer_len(i-1))
    enddo
!$omp end parallel do

    return
  end subroutine concat_buffer_intarray2

  ! *
  ! * Tests concat_buffer -subroutines
  ! *
  subroutine test_concat_buffer()
    use stream,only:prnlev, outu
    use nblist_types,only:intarray_t, intarray2_t
    use memory
    implicit none
    ! Variables
    type(intarray_t), allocatable, dimension(:) :: buffer
    type(intarray2_t), allocatable, dimension(:) :: buffer2
    integer, allocatable, dimension(:) :: buffer_len
    integer nbuffer
    integer i, j

    nbuffer = 16
    call chmalloc('nblist_util.src','test_concat_buffer','buffer_len',nbuffer,intg=buffer_len)

    !-------------------------------------------------------------------------------

    call init_array(buffer, nbuffer, nbuffer*100, nbuffer*100)
    
    do i=0,nbuffer-1
       do j=1,100
          buffer(i)%array(j) = i*100 + j
       enddo
    enddo

    buffer_len(1:nbuffer) = 100

    call concat_buffer(nbuffer, buffer_len, buffer)

    do j=1,nbuffer*100
       if (buffer(0)%array(j) /= j) then
          write (outu,'(a,2i8)') 'j,buffer(0)%array(j)=',j,buffer(0)%array(j)
          call wrndie(-5,'<nblist_util>','test_concat_buffer FAILED')
       endif
    enddo

    call uninit_array(buffer)

    !-------------------------------------------------------------------------------

    call init_array(buffer2, nbuffer, nbuffer*100, nbuffer*100)
    
    do i=0,nbuffer-1
       do j=1,100
          buffer2(i)%array(1:2,j) = (/ -(i*100 + j), i*100 + j /)
       enddo
    enddo

    buffer_len(1:nbuffer) = 100

    call concat_buffer(nbuffer, buffer_len, buffer2)

    do j=1,nbuffer*100
       if (buffer2(0)%array(1,j) /= -j .or. buffer2(0)%array(2,j) /= j) then
          write (outu,'(a,3i8)') 'j,buffer2(0)%array(j)=',j,buffer2(0)%array(1:2,j)
          call wrndie(-5,'<nblist_util>','test_concat_buffer FAILED')
       endif
    enddo

    call uninit_array(buffer2)

    !-------------------------------------------------------------------------------

    call chmdealloc('nblist_util.src','test_concat_buffer','buffer_len',nbuffer,intg=buffer_len)

    if (prnlev > 2) write (outu,'(a)') 'test_concat_buffer OK'

    return
  end subroutine test_concat_buffer

  ! *
  ! * Cumulative exclusive sum (wrapper)
  ! *
  subroutine cumsum_exclusive(array, array_len, array_out)
    implicit none
    ! Input / Output
    integer, intent(inout) :: array(:)
    integer, intent(in) :: array_len
    integer, intent(out), optional :: array_out(:)
    ! Variables
    integer i

    if (array_len == 0) return

    if (present(array_out)) then
!$omp parallel do schedule(static) private(i)
       do i=1,array_len
          array_out(i) = array(i)
       enddo
!$omp end parallel do
       call cumsum_inplace_exclusive(array_out, array_len)
    else
       call cumsum_inplace_exclusive(array, array_len)
    endif

    return
  end subroutine cumsum_exclusive

  ! *
  ! * Cumulative inclusive sum (wrapper)
  ! *
  subroutine cumsum_inclusive(array, array_len, array_out)
    implicit none
    ! Input / Output
    integer, intent(inout) :: array(:)
    integer, intent(in) :: array_len
    integer, intent(out), optional :: array_out(:)
    ! Variables
    integer i

    if (array_len == 0) return

    if (present(array_out)) then
!$omp parallel do schedule(static) private(i)
       do i=1,array_len
          array_out(i) = array(i)
       enddo
!$omp end parallel do
       call cumsum_inplace_inclusive(array_out, array_len)
    else
       call cumsum_inplace_inclusive(array, array_len)
    endif

    return
  end subroutine cumsum_inclusive

  ! *
  ! * Calculates exclusive cumulative sum
  ! *
  subroutine cumsum_inplace_exclusive(array, array_len)
    use domdec_common,only:nthread
    implicit none
    ! Input / Output
    integer, intent(inout) :: array(:)
    integer, intent(in) :: array_len
    ! Functions
    integer omp_get_thread_num
    ! Variables
    integer i, istart, iend, tid, val

    if (nthread == 1 .or. array_len <= nthread*20) then
       ! For small arrays, we just do a simple single thread sum:
       call cumsum_inplace_exclusive_block(array, 1, array_len, val)
    else
       if (nthread > MAX_NTHREAD) then
          call wrndie(-5,'<nblist_util>','cumsum_inplace_exclusive: MAX_NTHREAD exceeded')
       endif
!$omp parallel private(tid, istart, iend, val)
#ifdef _OPENMP
       tid = omp_get_thread_num()
#else
       tid = 0
#endif
       istart = tid*array_len/nthread + 1
       iend = (tid+1)*array_len/nthread
       call cumsum_inplace_exclusive_block(array, istart, iend, val)
       blockval(tid+1) = val + array(iend)
!$omp barrier
!$omp single
       call cumsum_inplace_exclusive_block(blockval, 1, nthread, val)
!$omp end single
       array(istart:iend) = array(istart:iend) + blockval(tid+1)
!$omp end parallel
    endif
    
    return
  end subroutine cumsum_inplace_exclusive

  ! *
  ! * Cumulative inplace exclusive sum for a block
  ! *
  subroutine cumsum_inplace_exclusive_block(array, istart, iend, val)
    implicit none
    ! Input / Output
    integer, intent(inout) :: array(:)
    integer, intent(in) :: istart, iend
    integer, intent(out) :: val
    ! Variables
    integer i, array_i

    val = array(istart)
    array(istart) = 0
    do i=istart+1,iend
       array_i = array(i)
       array(i) = val + array(i-1)
       val = array_i
    enddo

    return
  end subroutine cumsum_inplace_exclusive_block

  ! *
  ! * Calculates inclusive cumulative sum
  ! *
  subroutine cumsum_inplace_inclusive(array, array_len)
    use domdec_common,only:nthread
    implicit none
    ! Input / Output
    integer, intent(inout) :: array(:)
    integer, intent(in) :: array_len
    ! Functions
    integer omp_get_thread_num
    ! Variables
    integer i, istart, iend, tid, dummy

    if (nthread == 1 .or. array_len <= nthread*20) then
       ! For small arrays, we just do a simple single thread sum:
       call cumsum_inplace_inclusive_block(array, 1, array_len)
    else
       if (nthread > MAX_NTHREAD) then
          call wrndie(-5,'<nblist_util>','cumsum_inplace_inclusive: MAX_NTHREAD exceeded')
       endif
!$omp parallel private(tid, istart, iend)
#ifdef _OPENMP
       tid = omp_get_thread_num()
#else
       tid = 0
#endif
       istart = tid*array_len/nthread + 1
       iend = (tid+1)*array_len/nthread
       call cumsum_inplace_inclusive_block(array, istart, iend)
       blockval(tid+1) = array(iend)
!$omp barrier
!$omp single
       call cumsum_inplace_exclusive_block(blockval, 1, nthread, dummy)
!$omp end single
       array(istart:iend) = array(istart:iend) + blockval(tid+1)
!$omp end parallel
    endif
    
    return
  end subroutine cumsum_inplace_inclusive

  ! *
  ! * Cumulative inplace inclusive sum for a block
  ! *
  subroutine cumsum_inplace_inclusive_block(array, istart, iend)
    implicit none
    ! Input / Output
    integer, intent(inout) :: array(:)
    integer, intent(in) :: istart, iend
    ! Variables
    integer i

    do i=istart+1,iend
       array(i) = array(i) + array(i-1)
    enddo

    return
  end subroutine cumsum_inplace_inclusive_block

  ! *
  ! * Tests cumulative summation subroutines
  ! *
  subroutine test_cumsum()
    use stream,only:prnlev, outu
    use domdec_common,only:nthread
    use memory
    implicit none
    ! Variables
    integer, allocatable, dimension(:) :: a, a_ex, a_in, b
    integer i, len, dummy

    len = nthread*100 + 5
    call chmalloc('nblist_util.src','test_cumsum','a',len,intg=a)
    call chmalloc('nblist_util.src','test_cumsum','b',len,intg=b)
    call chmalloc('nblist_util.src','test_cumsum','a_ex',len,intg=a_ex)
    call chmalloc('nblist_util.src','test_cumsum','a_in',len,intg=a_in)

    ! Simple [1, 2, 3, 4, ...] test sequence => analytical answer given by the arithmetic series sum
    do i=1,len
       a_ex(i) = (i-1)*i/2
       a_in(i) = i*(i+1)/2
    enddo    

    ! In-place:
    do i=1,len
       a(i) = i
    enddo
    call cumsum_inplace_exclusive_block(a, 1, len, dummy)
    call test_cumsum_compare(a, a_ex, len)

    do i=1,len
       a(i) = i
    enddo
    call cumsum_inplace_inclusive_block(a, 1, len)
    call test_cumsum_compare(a, a_in, len)

    do i=1,len
       a(i) = i
    enddo
    call cumsum_exclusive(a, len)
    call test_cumsum_compare(a, a_ex, len)

    do i=1,len
       a(i) = i
    enddo
    call cumsum_inclusive(a, len)
    call test_cumsum_compare(a, a_in, len)

    do i=1,nthread*20+1
       a(i) = i
    enddo
    call cumsum_exclusive(a, nthread*20+1)
    call test_cumsum_compare(a, a_ex, nthread*20+1)

    do i=1,nthread*20+1
       a(i) = i
    enddo
    call cumsum_inclusive(a, nthread*20+1)
    call test_cumsum_compare(a, a_in, nthread*20+1)

    ! Out-place:
    do i=1,len
       a(i) = i
    enddo

    call cumsum_exclusive(a, len, b)
    call test_cumsum_compare(b, a_ex, len)

    call cumsum_inclusive(a, len, b)
    call test_cumsum_compare(b, a_in, len)

    call chmdealloc('nblist_util.src','test_cumsum','a',len,intg=a)
    call chmdealloc('nblist_util.src','test_cumsum','b',len,intg=b)
    call chmdealloc('nblist_util.src','test_cumsum','a_ex',len,intg=a_ex)
    call chmdealloc('nblist_util.src','test_cumsum','a_in',len,intg=a_in)

    if (prnlev > 2) write (outu,'(a)') 'test_cumsum OK'

    return
  end subroutine test_cumsum

  subroutine test_cumsum_compare(a, b, len)
    use stream,only:outu
    implicit none
    integer, intent(in) :: a(:), b(:), len
    integer i, j
    
    do i=1,len
       if (a(i) /= b(i)) then
          do j=1,len
             write (outu,'(3i10)') j, a(j), b(j)
          enddo
          call wrndie(-5,'<nblist_util>','test_cumsum FAILED')
       endif
    enddo
    
    return
  end subroutine test_cumsum_compare

  !------------------------------------------------------------------------------------
  !------------------------------------------------------------------------------------
  !------------------------------------------------------------------------------------

  ! *
  ! * Given the size of arrays ntbl(1:n, 1:m), reduces tbl(1:n, m)%array() into
  ! * tbl(i,1)%array(1:ntbl(i,1)), i=1...n
  ! *
  subroutine reduce_intarray(n, m, ntbl, tbl)
    use nblist_types,only:intarray_t
    implicit none
    ! Input / Output
    integer, intent(in) :: n, m
    integer, intent(inout) :: ntbl(:,:)
    type(intarray_t), intent(inout) :: tbl(:,:)
    ! Variables
    integer i, j

!$omp parallel private(i, j)
    do j=1,n
!$omp single
       do i=2,m
          ntbl(j,i) = ntbl(j,i) + ntbl(j,i-1)
       enddo
       if (ntbl(j,m) > size(tbl(j,1)%array)) then
          call wrndie(-5,'<nblist_util>','Error using reduce_intarray')
       endif
!$omp end single
       ! Concatenate tbl(j,2:m) to tbl(j,0)
       if (ntbl(j,m) > 0) then
!$omp do schedule(static)
          do i=2,m
             tbl(j,1)%array(ntbl(j,i-1)+1:ntbl(j,i)) = tbl(j,i)%array(1:ntbl(j,i)-ntbl(j,i-1))
          enddo
!$omp end do
       endif
       ntbl(j,1) = ntbl(j,m)
    enddo
!$omp end parallel


    return
  end subroutine reduce_intarray

  !------------------------------------------------------------------------------------
  !------------------------------------------------------------------------------------
  !------------------------------------------------------------------------------------

  ! *
  ! * Initializes (allocates) and re-allocates intarray_t array with 10% extra or
  ! * the amount defined in narray2_new_in
  ! * NOTE: dimensions are array(0:narray-1,narray2_new)
  ! *
  subroutine init_array_intarray(array_inout, narray1, narray2, narray2_new_in)
    use nblist_types,only:intarray_t
    use memory
    implicit none
    ! Input
    type(intarray_t), intent(inout), allocatable, dimension(:) :: array_inout
    integer, intent(in) :: narray1, narray2
    integer, intent(in), optional :: narray2_new_in
    ! Variables
    integer narray2_new, i

    if (allocated(array_inout)) then
       if (size(array_inout) < narray1) then
          do i=0,size(array_inout)-1
             if (allocated(array_inout(i)%array)) then             
                call chmdealloc('nblist_util.src','init_array_intarray','array_inout(i)%array',&
                     size(array_inout(i)%array),intg=array_inout(i)%array)
             endif
          enddo
          deallocate(array_inout)
       endif
    endif

    if (.not.allocated(array_inout)) then
       allocate(array_inout(0:narray1-1))
    endif

    if (present(narray2_new_in)) then
       narray2_new = narray2_new_in
    else
       narray2_new = int(narray2*1.1)
    endif

    do i=0,narray1-1
       if (allocated(array_inout(i)%array)) then
          if (size(array_inout(i)%array) < narray2) then
             call chmdealloc('nblist_util.src','init_array_intarray','array_inout(i)%array',&
                  size(array_inout(i)%array),intg=array_inout(i)%array)
          endif
       endif
       if (.not.allocated(array_inout(i)%array)) then
          call chmalloc('nblist_util.src','init_array_intarray','array_inout(i)%array',&
               narray2_new,intg=array_inout(i)%array)
       endif
    enddo

    return
  end subroutine init_array_intarray

  ! *
  ! * Initializes (allocates) and re-allocates intarray_t array with 10% extra or
  ! * the amount defined in narray3_new_in
  ! * NOTE: dimensions are array(lb1:ub1,lb2:ub2)%array(1:narray_new)
  ! *
  subroutine init_array_intarray_2d(array_inout, lb1, ub1, lb2, ub2, narray, narray_new_in)
    use nblist_types,only:intarray_t
    use memory
    implicit none
    ! Input
    type(intarray_t), intent(inout), allocatable, dimension(:,:) :: array_inout
    integer, intent(in) :: lb1, ub1, lb2, ub2, narray
    integer, intent(in), optional :: narray_new_in
    ! Variables
    integer size1, size2
    integer narray_new, i, j

    size1 = ub1-lb1+1
    size2 = ub2-lb2+1

    if (allocated(array_inout)) then
       if (size(array_inout,1) < size1 .or. size(array_inout,2) < size2 .or. &
            lbound(array_inout,1) /= lb1 .or. ubound(array_inout,1) /= ub1 .or. &
            lbound(array_inout,2) /= lb2 .or. ubound(array_inout,2) /= ub2) then
          do j=lbound(array_inout,2),ubound(array_inout,2)
             do i=lbound(array_inout,1),ubound(array_inout,1)
                if (allocated(array_inout(i,j)%array)) then
                   call chmdealloc('nblist_util.src','init_array_intarray_2d',&
                        'array_inout(i,j)%array',size(array_inout(i,j)%array),&
                        intg=array_inout(i,j)%array)
                endif
             enddo
          enddo
          deallocate(array_inout)
       endif
    endif

    if (.not.allocated(array_inout)) then
       allocate(array_inout(lb1:ub1, lb2:ub2))
    endif

    if (present(narray_new_in)) then
       narray_new = narray_new_in
    else
       narray_new = int(narray*1.1)
    endif

    do j=lb2,ub2
       do i=lb1,ub1
          if (allocated(array_inout(i,j)%array) .and. size(array_inout(i,j)%array) < narray) then
             call chmdealloc('nblist_util.src','init_array_intarray_2d','array_inout(i,j)%array',&
                  size(array_inout(i,j)%array),intg=array_inout(i,j)%array)
          endif
          if (.not.allocated(array_inout(i,j)%array)) then
             call chmalloc('nblist_util.src','init_array_intarray','array_inout(i,j)%array',&
                  narray_new,intg=array_inout(i,j)%array)
          endif
       enddo
    enddo

    return
  end subroutine init_array_intarray_2d

  ! *
  ! * Initializes (allocates) and re-allocates intarray_t array with 10% extra or
  ! * the amount defined in narray3_new_in
  ! * NOTE: dimensions are array_inout(0:narray1-1)%array(narray2, narray3)
  ! *
  subroutine init_array_intarray2(array_inout, narray1, narray2, narray3, narray3_new_in)
    use nblist_types,only:intarray2_t
    use memory
    implicit none
    ! Input
    type(intarray2_t), intent(inout), allocatable, dimension(:) :: array_inout
    integer, intent(in) :: narray1, narray2, narray3
    integer, intent(in), optional :: narray3_new_in
    ! Variables
    integer narray3_new, i

    if (allocated(array_inout)) then
       if (size(array_inout) < narray1) then
          do i=0,size(array_inout)-1
             if (allocated(array_inout(i)%array)) then
                call chmdealloc('nblist_util.src','init_array_intarray2','array_inout(i)%array',&
                     size(array_inout(i)%array,1),size(array_inout(i)%array,2),&
                     intg=array_inout(i)%array)
             endif
          enddo
          deallocate(array_inout)
       endif
    endif

    if (.not.allocated(array_inout)) then
       allocate(array_inout(0:narray1-1))
    endif

    if (present(narray3_new_in)) then
       narray3_new = narray3_new_in
    else
       narray3_new = int(narray3*1.1)
    endif
       
    do i=0,narray1-1
       if (allocated(array_inout(i)%array) .and. &
            (size(array_inout(i)%array,1) < narray2 .or. &
            size(array_inout(i)%array,2) < narray3)) then
          call chmdealloc('nblist_util.src','init_array_intarray','array_inout(i)%array',&
               size(array_inout(i)%array,1),size(array_inout(i)%array,2),intg=array_inout(i)%array)
       endif
       if (.not.allocated(array_inout(i)%array)) then
          call chmalloc('nblist_util.src','init_array_intarray','array_inout(i)%array',&
               narray2,narray3_new,intg=array_inout(i)%array)
       endif
    enddo

    return
  end subroutine init_array_intarray2

  ! *
  ! * Initializes (allocates) and re-allocates cr4array_t array with 10% extra or
  ! * the amount defined in narray2_new_in
  ! * NOTE: dimensions are array(0:narray-1,narray2_new)
  ! *
  subroutine init_array_cr4array(array_inout, narray1, narray2, narray2_new_in)
    use nblist_types,only:cr4array_t
    use memory
    implicit none
    ! Input
    type(cr4array_t), intent(inout), allocatable, dimension(:) :: array_inout
    integer, intent(in) :: narray1, narray2
    integer, intent(in), optional :: narray2_new_in
    ! Variables
    integer narray2_new, i

    if (allocated(array_inout)) then
       if (size(array_inout) < narray1) then
          do i=0,size(array_inout)-1
             if (allocated(array_inout(i)%array)) then             
                call chmdealloc('nblist_util.src','init_array_cr4array','array_inout(i)%array',&
                     size(array_inout(i)%array),cr4=array_inout(i)%array)
             endif
          enddo
          deallocate(array_inout)
       endif
    endif

    if (.not.allocated(array_inout)) then
       allocate(array_inout(0:narray1-1))
    endif

    if (present(narray2_new_in)) then
       narray2_new = narray2_new_in
    else
       narray2_new = int(narray2*1.1)
    endif

    do i=0,narray1-1
       if (allocated(array_inout(i)%array) .and. size(array_inout(i)%array) < narray2) then
          call chmdealloc('nblist_util.src','init_array_cr4array','array_inout(i)%array',&
               size(array_inout(i)%array),cr4=array_inout(i)%array)
       endif
       if (.not.allocated(array_inout(i)%array)) then
          call chmalloc('nblist_util.src','init_array_cr4array','array_inout(i)%array',&
               narray2_new,cr4=array_inout(i)%array)
       endif
    enddo

    return
  end subroutine init_array_cr4array

  ! *
  ! * Initializes (allocates) and re-allocates cr4array_t array with 10% extra or
  ! * the amount defined in narray2_new_in
  ! * NOTE: dimensions are array(0:narray-1,narray2_new)
  ! *
  subroutine init_array_crlarray(array_inout, narray1, narray2, narray2_new_in, q_reallocated)
    use domdec_local_types,only:crlarray_t
    use memory
    implicit none
    ! Input
    type(crlarray_t), intent(inout), allocatable, dimension(:) :: array_inout
    integer, intent(in) :: narray1, narray2
    integer, intent(in), optional :: narray2_new_in
    logical, intent(out), optional :: q_reallocated
    ! Variables
    integer narray2_new, i

    if (present(q_reallocated)) q_reallocated = .false.

    if (allocated(array_inout)) then
       if (size(array_inout) < narray1) then
          do i=0,size(array_inout)-1
             if (allocated(array_inout(i)%array)) then             
                call chmdealloc('nblist_util.src','init_array_crlarray','array_inout(i)%array',&
                     size(array_inout(i)%array),crl=array_inout(i)%array)
             endif
          enddo
          deallocate(array_inout)
       endif
    endif

    if (.not.allocated(array_inout)) then
       allocate(array_inout(0:narray1-1))
    endif

    if (present(narray2_new_in)) then
       narray2_new = narray2_new_in
    else
       narray2_new = int(narray2*1.1)
    endif

    do i=0,narray1-1
       if (allocated(array_inout(i)%array) .and. size(array_inout(i)%array) < narray2) then
          call chmdealloc('nblist_util.src','init_array_crlarray','array_inout(i)%array',&
               size(array_inout(i)%array),crl=array_inout(i)%array)
       endif
       if (.not.allocated(array_inout(i)%array)) then
          call chmalloc('nblist_util.src','init_array_crlarray','array_inout(i)%array',&
               narray2_new,crl=array_inout(i)%array)
          if (present(q_reallocated)) q_reallocated = .true.
       endif
    enddo

    return
  end subroutine init_array_crlarray

  ! *
  ! * Uninitializes (deallocates) integer array
  ! *
  subroutine uninit_array_integer(array_inout)
    use memory,only:chmdealloc
    implicit none
    ! Input
    integer, intent(inout), allocatable, dimension(:) :: array_inout
    ! Variables
    integer i

    if (allocated(array_inout)) then
       call chmdealloc('nblist_util.src','uninit_array_integer','array_inout',&
            size(array_inout),intg=array_inout)
    endif

    return
  end subroutine uninit_array_integer

  ! *
  ! * Uninitializes (deallocates) integer array pointer
  ! *
  subroutine uninit_array_integerp(array_inout)
    use memory,only:chmdealloc
    implicit none
    ! Input
    integer, intent(inout), pointer, dimension(:) :: array_inout
    ! Variables
    integer i

    if (associated(array_inout)) then
       call chmdealloc('nblist_util.src','uninit_array_integerp','array_inout',&
            size(array_inout),intgp=array_inout)
    endif

    return
  end subroutine uninit_array_integerp

  ! *
  ! * Uninitializes (deallocates) intarray_t array
  ! *
  subroutine uninit_array_intarray(array_inout)
    use nblist_types,only:intarray_t
    use memory,only:chmdealloc
    implicit none
    ! Input
    type(intarray_t), intent(inout), allocatable, dimension(:) :: array_inout
    ! Variables
    integer i

    if (allocated(array_inout)) then
       do i=0,size(array_inout)-1
          if (allocated(array_inout(i)%array)) then
             call chmdealloc('nblist_util.src','uninit_array_intarray','array_inout(i)%array',&
                  size(array_inout(i)%array),intg=array_inout(i)%array)
          endif
       enddo
       deallocate(array_inout)
    endif

    return
  end subroutine uninit_array_intarray


  ! *
  ! * Uninitializes (deallocates) intarray_t array
  ! *
  subroutine uninit_array_intarray_2d(array_inout)
    use nblist_types,only:intarray_t
    use memory,only:chmdealloc
    implicit none
    ! Input
    type(intarray_t), intent(inout), allocatable, dimension(:,:) :: array_inout
    ! Variables
    integer i, j

    if (allocated(array_inout)) then
       do j=lbound(array_inout,2),ubound(array_inout,2)
          do i=lbound(array_inout,1),ubound(array_inout,1)
             if (allocated(array_inout(i,j)%array)) then
                call chmdealloc('nblist_util.src','uninit_array_intarray_2d',&
                     'array_inout(i,j)%array',&
                     size(array_inout(i,j)%array),intg=array_inout(i,j)%array)
             endif
          enddo
       enddo
       deallocate(array_inout)
    endif

    return
  end subroutine uninit_array_intarray_2d

  ! *
  ! * Uninitializes (deallocates) intarray2_t array
  ! *
  subroutine uninit_array_intarray2(array_inout)
    use nblist_types,only:intarray2_t
    use memory,only:chmdealloc
    implicit none
    ! Input
    type(intarray2_t), intent(inout), allocatable, dimension(:) :: array_inout
    ! Variables
    integer i

    if (allocated(array_inout)) then
       do i=0,size(array_inout)-1
          if (allocated(array_inout(i)%array)) then
             call chmdealloc('nblist_util.src','uninit_array_intarray','array_inout(i)%array',&
                  size(array_inout(i)%array,1),size(array_inout(i)%array,2),intg=array_inout(i)%array)
          endif
       enddo
       deallocate(array_inout)
    endif

    return
  end subroutine uninit_array_intarray2

  ! *
  ! * Uninitializes (deallocates) cr4array_t array
  ! *
  subroutine uninit_array_cr4array(array_inout)
    use nblist_types,only:cr4array_t
    use memory,only:chmdealloc
    implicit none
    ! Input
    type(cr4array_t), intent(inout), allocatable, dimension(:) :: array_inout
    ! Variables
    integer i

    if (allocated(array_inout)) then
       do i=0,size(array_inout)-1
          if (allocated(array_inout(i)%array)) then
             call chmdealloc('nblist_util.src','uninit_array_intarray','array_inout(i)%array',&
                  size(array_inout(i)%array),cr4=array_inout(i)%array)
          endif
       enddo
       deallocate(array_inout)
    endif

    return
  end subroutine uninit_array_cr4array

  ! *
  ! * Uninitializes (deallocates) crlarray_t array
  ! *
  subroutine uninit_array_crlarray(array_inout)
    use domdec_local_types,only:crlarray_t
    use memory,only:chmdealloc
    implicit none
    ! Input
    type(crlarray_t), intent(inout), allocatable, dimension(:) :: array_inout
    ! Variables
    integer i

    if (allocated(array_inout)) then
       do i=0,size(array_inout)-1
          if (allocated(array_inout(i)%array)) then
             call chmdealloc('nblist_util.src','uninit_array_intarray','array_inout(i)%array',&
                  size(array_inout(i)%array),crl=array_inout(i)%array)
          endif
       enddo
       deallocate(array_inout)
    endif

    return
  end subroutine uninit_array_crlarray

  ! *
  ! * Uninitializes (deallocates) xyzq_sp_t array
  ! *
  subroutine uninit_array_xyzq_sp(array)
    use domdec_local_types,only:xyzq_sp_t
    use memory
    implicit none
    ! Input
    type(xyzq_sp_t), pointer, dimension(:) :: array

    if (associated(array)) then
       deallocate(array)
    endif

    return
  end subroutine uninit_array_xyzq_sp

  ! *
  ! * Initializes (allocates) and re-allocates integer array with 10% extra
  ! *
  subroutine init_array_byte(array, narray)
    use memory
    implicit none
    ! Input
    integer(kind=int_byte), allocatable, dimension(:) :: array
    integer, intent(in) :: narray
    ! Variables
    integer narray_new

    if (allocated(array)) then
       if (size(array) < narray) then
          call chmdealloc('nblist_util.src','init_array','array',size(array),iby=array)
       endif
    endif

    if (.not.allocated(array)) then
       narray_new = int(narray*1.1) + 16
       call chmalloc('nblist_util.src','init_array','array',narray_new,iby=array)
      endif

    return
  end subroutine init_array_byte

  ! *
  ! * Initializes (allocates) and re-allocates integer array with 10% extra OR
  ! * the exact amount narray_new_in
  ! *
  subroutine init_array_integer(array, narray, narray_new_in)
    use memory
    implicit none
    ! Input
    integer, allocatable, dimension(:) :: array
    integer, intent(in) :: narray
    integer, intent(in), optional :: narray_new_in
    ! Variables
    integer narray_new

    if (allocated(array)) then
       if (size(array) < narray) then
          call chmdealloc('nblist_util.src','init_array','array',size(array),intg=array)
       endif
    endif

    if (.not.allocated(array)) then
       if (present(narray_new_in)) then
          narray_new = narray_new_in
       else
          narray_new = int(narray*1.1) + 16
       endif
       call chmalloc('nblist_util.src','init_array','array',narray_new,intg=array)
      endif

    return
  end subroutine init_array_integer

  ! *
  ! * Initializes (allocates) and re-allocates integer array with 10% extra OR
  ! * the exact amount narray_new_in
  ! *
  subroutine init_array_integerp(array, narray, narray_new_in)
    use memory
    implicit none
    ! Input
    integer, pointer, dimension(:) :: array
    integer, intent(in) :: narray
    integer, intent(in), optional :: narray_new_in
    ! Variables
    integer narray_new

    if (associated(array)) then
       if (size(array) < narray) then
          call chmdealloc('nblist_util.src','init_array_integerp','array',size(array),intgp=array)
       endif
    endif

    if (.not.associated(array)) then
       if (present(narray_new_in)) then
          narray_new = narray_new_in
       else
          narray_new = int(narray*1.1) + 16
       endif
       call chmalloc('nblist_util.src','init_array_integerp','array',narray_new,intgp=array)
      endif

    return
  end subroutine init_array_integerp

  ! *
  ! * Initializes (allocates) and re-allocates double precision array with 10% extra
  ! *
  subroutine init_array_double_1d(array, narray)
    use memory
    implicit none
    ! Input
    real(chm_real), allocatable, dimension(:) :: array
    integer, intent(in) :: narray
    ! Variables
    integer narray_new

    if (allocated(array)) then
       if (size(array) < narray) then
          call chmdealloc('nblist_util.src','init_array','array',size(array),crl=array)
       endif
    endif

    if (.not.allocated(array)) then
       narray_new = int(narray*1.1) + 8
       call chmalloc('nblist_util.src','init_array','array',narray_new,crl=array)
      endif

    return
  end subroutine init_array_double_1d

  ! *
  ! * Initializes (allocates) and re-allocates xyzq_sp_t array with 20% extra
  ! *
  subroutine init_array_xyzq_sp(array, narray)
    use domdec_local_types,only:xyzq_sp_t
    use memory
    implicit none
    ! Input
    type(xyzq_sp_t), pointer, dimension(:) :: array
    integer, intent(in) :: narray
    ! Variables
    integer narray_new

    if (associated(array)) then
       if (size(array) < narray) then
          deallocate(array)
       endif
    endif

    if (.not.associated(array)) then
       narray_new = int(narray*1.2)
       allocate(array(narray_new))
    endif

    return
  end subroutine init_array_xyzq_sp

  ! *
  ! * Initializes (allocates) and re-allocates xyzq_dp_t array with 20% extra
  ! *
  subroutine init_array_xyzq_dp(array, narray)
    use domdec_local_types,only:xyzq_dp_t
    use memory
    implicit none
    ! Input
    type(xyzq_dp_t), pointer, dimension(:) :: array
    integer, intent(in) :: narray
    ! Variables
    integer narray_new

    if (associated(array)) then
       if (size(array) < narray) then
          deallocate(array)
       endif
    endif

    if (.not.associated(array)) then
       narray_new = int(narray*1.2)
       allocate(array(narray_new))
    endif

    return
  end subroutine init_array_xyzq_dp

  ! *
  ! * Initializes (allocates) and re-allocates 2d double precision array with 10% extra
  ! * in narray2 direction
  ! *
  subroutine init_array_double_2d(array, narray1, narray2)
    use memory
    implicit none
    ! Input
    real(chm_real), allocatable, dimension(:,:) :: array
    integer, intent(in) :: narray1, narray2
    ! Variables
    integer narray2_new

    if (allocated(array)) then
       if (size(array,1) < narray1 .or. size(array,2) < narray2) then
          call chmdealloc('nblist_util.src','init_array','array',&
               size(array,1),size(array,2),crl=array)
       endif
    endif

    if (.not.allocated(array)) then
       narray2_new = int(narray2*1.1) + 8
       call chmalloc('nblist_util.src','init_array','array',narray1,narray2_new,crl=array)
      endif

    return
  end subroutine init_array_double_2d

  ! *
  ! * Initializes (allocates) and re-allocates single precision array with 10% extra
  ! *
  subroutine init_array_single_1d(array, narray)
    use memory
    implicit none
    ! Input
    real(chm_real4), allocatable, dimension(:) :: array
    integer, intent(in) :: narray
    ! Variables
    integer narray_new

    if (allocated(array)) then
       if (size(array) < narray) then
          call chmdealloc('nblist_util.src','init_array','array',size(array),cr4=array)
       endif
    endif

    if (.not.allocated(array)) then       
       narray_new = int(narray*1.1) + 16
       call chmalloc('nblist_util.src','init_array','array',narray_new,cr4=array)
      endif

    return
  end subroutine init_array_single_1d

  ! *
  ! * Initializes (allocates) and re-allocates single precision array with 10% extra
  ! *
  subroutine init_array_single_2d(array, narray1, narray2)
    use memory
    implicit none
    ! Input
    real(chm_real4), allocatable, dimension(:,:) :: array
    integer, intent(in) :: narray1, narray2
    ! Variables
    integer narray1_new, narray2_new

    if (allocated(array)) then
       if (size(array,1) < narray1 .or. size(array,2) < narray2) then
          call chmdealloc('nblist_util.src','init_array','array',&
               size(array,1),size(array,2),cr4=array)
       endif
    endif

    if (.not.allocated(array)) then       
       narray1_new = int(narray1*1.1) + 16
       narray2_new = int(narray2*1.1) + 16
       call chmalloc('nblist_util.src','init_array','array',narray1_new,narray2_new,cr4=array)
      endif

    return
  end subroutine init_array_single_2d

  ! *
  ! * Packs vdwtype
  ! *
  subroutine pack_vdwtype_group(ngroupl, groupl, group, &
       vdwtype, vdwtype_pack, group_order)
    use groupxfast,only:group_out
    implicit none
    ! Input / Output
    integer, intent(in) :: ngroupl, groupl(*), group(*), vdwtype(*)
    integer, intent(out) :: vdwtype_pack(*)
    integer, intent(in), optional :: group_order(*)
    ! Variables
    integer i, j, ig, is, iq

    j = 0
    if (present(group_order)) then
       do ig=1,ngroupl
          ! i is the unpacked group index
          i = groupl(group_order(ig))
          call pack_kernel()
       enddo
    else
       do ig=1,ngroupl
          ! i is the unpacked group index
          i = groupl(ig)
          call pack_kernel()
       enddo
    endif

    return

  contains
    subroutine pack_kernel
      implicit none
      call group_out(group(i), is, iq)
      vdwtype_pack(j+1:j+iq-is+1) = vdwtype(is:iq)
      j = j + iq-is+1
      return
    end subroutine pack_kernel

  end subroutine pack_vdwtype_group

  ! *
  ! * Packs vdwtype to the sorted array
  ! * q_convert_to_c = .true. when we need to convert to c-style array (starting with 0)
  ! *
  subroutine pack_vdwtype_atom(natoml, vdwtype, vdwtype_pack, loc2glo_ind, q_convert_to_c)
    implicit none
    ! Input / Output
    integer, intent(in) :: natoml
    integer, intent(in) :: vdwtype(*)
    integer, intent(out) :: vdwtype_pack(*)
    integer, intent(in) :: loc2glo_ind(*)
    logical, intent(in) :: q_convert_to_c
    ! Variables
    integer i

    if (q_convert_to_c) then
!$omp parallel do private(i) schedule(static)
       do i=1,natoml
          vdwtype_pack(i) = vdwtype(loc2glo_ind(i)+1) - 1
       enddo
!$omp end parallel do
    else
!$omp parallel do private(i) schedule(static)
       do i=1,natoml
          vdwtype_pack(i) = vdwtype(loc2glo_ind(i)+1)
       enddo
!$omp end parallel do
    endif
    
    return
  end subroutine pack_vdwtype_atom

  ! *
  ! * Packs groups
  ! *
  subroutine pack_groups(ngroupl, groupl, group, group_pack, group_order)
    use groupxfast,only:group_in, group_out
    implicit none
    ! Input / Output
    integer, intent(in) :: ngroupl, groupl(*), group(*)
    integer, intent(out) :: group_pack(*)
    integer, intent(in), optional :: group_order(*)
    ! Variables
    integer i, j, ig

    j = 0
    if (present(group_order)) then
       do ig=1,ngroupl
          ! i is the unpacked group index
          i = groupl(group_order(ig))
          call pack_kernel()
       enddo
    else
       do ig=1,ngroupl
          ! i is the unpacked group index
          i = groupl(ig)
          call pack_kernel()
       enddo
    endif
    return

  contains
    subroutine pack_kernel
      implicit none
      integer is, iq, itype
      ! Unpacked group is defined in group(i)
      call group_out(group(i), is, iq, itype)
      ! Packed group is written to group_pack(ig)
      group_pack(ig) = group_in(j+1, j+iq-is+1, itype)
      j = j + iq-is+1
      return
    end subroutine pack_kernel
   
  end subroutine pack_groups

  ! *
  ! * Packs grouppos_pack and groupbox_pack
  ! * Adds: grouppos = groupcenter + groupsh*box_size
  ! *
  subroutine pack_grouppos(ngroup, groupl, groupcenter, groupsh, groupbox, &
       grouppos_pack, groupbox_pack)
    use domdec_common,only:boxx, boxy, boxz
    implicit none
    ! Input / Output
    integer, intent(in) :: ngroup, groupl(*)
    real(chm_real), intent(in) :: groupcenter(3,*), groupsh(3,*), groupbox(3,*)
    real(chm_real), intent(out) :: grouppos_pack(3,*), groupbox_pack(3,*)
    ! Variables
    integer ig, i

    do ig=1,ngroup
       i = groupl(ig)
       grouppos_pack(1,ig) = groupcenter(1,i) + groupsh(1,i)*boxx
       grouppos_pack(2,ig) = groupcenter(2,i) + groupsh(2,i)*boxy
       grouppos_pack(3,ig) = groupcenter(3,i) + groupsh(3,i)*boxz
       groupbox_pack(1:3,ig) = groupbox(1:3,i)
    enddo

    return
  end subroutine pack_grouppos

  ! *
  ! * Check for 1-4 exclusion between atoms i and j:
  ! * ix14 = 0 => exclusion
  ! * ix14 < 0 => 1-4 pair
  ! * ix14 > 0 => regular interaction
  ! *
  subroutine check_excl(i,j,inb14,iblo14,ix14)
    implicit none
    ! Input / Output
    integer, intent(in) :: i, j, inb14(*), iblo14(*)
    integer, intent(out) :: ix14
    ! Variables
    integer ii, jj, nxi, nximax, inbx

    if (i < j) then
       ii = i
       jj = j
    else
       ii = j
       jj = i
    endif

    if (ii > 1) then
       nxi=iblo14(ii-1)+1
    else
       nxi=1
    endif
    nximax=iblo14(ii)
    inbx=iabs(inb14(nxi))
    do while(nxi <= nximax .and. jj > inbx)
       nxi=nxi+1
       inbx=iabs(inb14(nxi))
    enddo
    if(nxi > nximax) then
       ix14=jj
    else if(jj == inb14(nxi)) then  ! exclusion found
       ix14=0
    else
       if(jj == inbx) then
          ix14=-jj    ! it's a 1-4 pair
       else
          ix14=jj
       endif
    endif

    return
  end subroutine check_excl

#endif /* (domdec)*/

end module nblist_util

