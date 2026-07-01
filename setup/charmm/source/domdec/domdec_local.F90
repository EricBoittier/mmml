module domdec_local
#if KEY_DOMDEC==1
  use chm_kinds
  use dimens_fcm
  use domdec_local_types
  use nblist_types,only:dpsp_t
  implicit none
  private

  !-------------------------------------------
  ! Domdec local coordinate and force arrays
  !-------------------------------------------
  ! Local arrays. These arrays are packed in terms of not being sparse (e.g. there are no empty
  ! spaces). Also, the order of atoms can be different in the local arrays, the mapping to the
  ! global CHARMM atom indices is determined by loc2glo_ind and glo2loc_ind defined below

  ! Local coordinates + charge (x, y, z, q)
  type(xyzq_dpsp_t) :: xyzq_loc

  integer, pointer, dimension(:) :: vdwtype_loc

  ! This flag tells if (xyzq_loc, vdwtype_loc) were allocated on the pinned GPU memory or not
  logical :: q_xyzq_loc_gpu_alloc = .false.
  logical :: q_vdwtype_loc_gpu_alloc = .false.

  ! Local force array, one for each CPU thread
  type(crlarray_t), allocatable, dimension(:) :: force_loc
  ! Logical flag that is true if force_loc contains data
  logical :: q_force_loc_used = .false.
  
  ! Temporary buffer for buffered version of unpack_reduce_zero_force()
  type(crlarray_t), allocatable, dimension(:) :: buf

  ! Indexing that maps atom indices local -> global
  ! NOTE: loc2glo_ind -values start at 0
  integer, allocatable, dimension(:) :: loc2glo_ind

  ! Indexing that maps atom indices global -> local
  ! NOTE: glo2loc_ind -values start at 0
  integer, allocatable, dimension(:) :: glo2loc_ind

  ! Local coordinate shift (-1.0, 0.0, or 1.0)
  real(chm_real4), allocatable, dimension(:,:) :: xyz_shift

  ! Shifts for coordinates and forces
  type(dpsp_t) :: scoordtab                  ! size 3*27
  real(chm_real) sforce(3*27)

#if KEY_DOMDEC_GPU==1 || KEY_DOMDEC==1
  real(chm_real), pointer, dimension(:) :: xyz_loc
  real(chm_real4), pointer, dimension(:) :: mass_loc
  integer, allocatable, dimension(:) :: coordpos

  ! Temporary xyzq -array
  type(xyzq_sp_t), pointer, dimension(:) :: xyzq_tmp

  ! Temporary local -> global index mapping
  integer, allocatable, dimension(:) :: loc2glo_ind_tmp
#endif

  interface pack_q
     module procedure pack_q_sp
     module procedure pack_q_dp
  end interface

  interface pack_xyzq
     module procedure pack_xyzq_sp
     module procedure pack_xyzq_nodisp_sp
     module procedure pack_xyzq_dp
  end interface

  interface pack_xyz
     module procedure pack_xyz_sp
     module procedure pack_xyz_dp
  end interface

  interface unpack_add_force
     module procedure unpack_add_force_cpu
#if KEY_DOMDEC_GPU==1
     module procedure unpack_add_force_gpu
#endif
  end interface

!!$  interface write_xyz_to_file
!!$     module procedure write_xyz_to_file_ps
!!$     module procedure write_xyz_to_file_pd
!!$  end interface

#if KEY_DOMDEC_GPU==1
  interface pack_xyzq_loc2glo
     module procedure pack_xyzq_loc2glo_sp
  end interface
#endif

  ! Public variables
  public xyzq_loc, force_loc, q_force_loc_used, loc2glo_ind, glo2loc_ind, scoordtab, sforce
  public vdwtype_loc

  ! Public subroutines
  public unpack_forces, build_local_coord
  public update_local_coord, update_local_home_coord, update_local_import_coord
  public uninit_local_coord
  public pack_xyz, pack_xyzq, unpack_add_force, build_local_vdwtype
  public uninit_local_vdwtype
#if KEY_DOMDEC_GPU==1
  public combine_gpu_forces_to_global_forces
  public uninit_pinned_memory
#endif

#endif

contains

#if KEY_DOMDEC==1

  ! *
  ! * Unpacks and reduces forces from force_loc to (forcex, forcey, forcez)
  ! *
  subroutine unpack_forces(forcex, forcey, forcez)
    use domdec_common,only:natoml_tot, q_gpu
#if KEY_DOMDEC_GPU==1
    use domdec_util_gpu_mod,only:range_start, range_stop
#endif
    implicit none
    ! Input / Output
    real(chm_real), intent(inout) :: forcex(*), forcey(*), forcez(*)

    if (q_force_loc_used) then
       ! Reduce force from threads (force_loc) to global force (forcex, forcey, forcez)
#if KEY_DOMDEC_GPU==1
       if (q_gpu) call range_start('unpack_reduce_force')
#endif
       call unpack_reduce_zero_force(natoml_tot, loc2glo_ind, forcex, forcey, forcez, force_loc)
#if KEY_DOMDEC_GPU==1
       if (q_gpu) call range_stop()
#endif
       q_force_loc_used = .false.
    endif
    
    return
  end subroutine unpack_forces

#if KEY_DOMDEC_GPU==1
  ! *
  ! * Combines GPU forces to global force array
  ! * NOTE: This assumes that we have waited until the GPU-CPU copy has finished using
  ! *       "wait_force_virial_energy_from_gpu"
  ! *
  subroutine combine_gpu_forces_to_global_forces(forcex, forcey, forcez)
    use domdec_common,only:q_gpu, natoml_tot
    use domdec_util_gpu_mod,only:range_start, range_stop, combine_force_from_gpu
    implicit none
    ! Input / Output
    real(chm_real), intent(inout) :: forcex(*), forcey(*), forcez(*)

    ! Combine forces from GPU to (forcex, forcey, forcez)
    if (q_gpu) then
       ! Combine forces from gpu to global force (forcex, forcey, forcez)
       call range_start('combine_force_from_gpu')
       call combine_force_from_gpu(forcex, forcey, forcez, natoml_tot, loc2glo_ind)
       call range_stop()
    endif
    
    return
  end subroutine combine_gpu_forces_to_global_forces
#endif

  ! *
  ! * Updates all local coordinates
  ! * NOTE: Called before energy calculation
  ! *
  subroutine update_local_coord(x, y, z)
    use number,only:zero
    use domdec_common,only:q_single, &
#if KEY_DOMDEC_GPU==1
         q_gpu, &
#endif
         natoml_tot
#if KEY_DOMDEC_GPU==1
    use domdec_util_gpu_mod,only:range_start, range_stop, copy_home_xyzq_to_gpu, &
         copy_import_xyzq_to_gpu
#endif
    implicit none
    ! Input
    real(chm_real), intent(in) :: x(*), y(*), z(*)

    ! Fill in scoordtab
    call calc_scoordtab()

#if KEY_DOMDEC_GPU==1
    if (q_gpu) then
       ! Pack coordinates to xyzq_loc
       call range_start('pack: xyzq_loc')
       call pack_xyz(1, natoml_tot, x, y, z, xyzq_loc%sp, loc2glo_ind, xyz_shift)
       call range_stop()

       ! Copy coordinates to GPU
       call copy_home_xyzq_to_gpu(xyzq_loc%sp)
       call copy_import_xyzq_to_gpu(xyzq_loc%sp)

    else
#endif
       ! Pack and recenter coordinates
       ! NOTE: array xyzq_loc is already initialized in nblist_builder
       
       ! Pack (x, y, z) to xyzq_loc and displace by xyz_shift:
       ! xyzq_loc = (x,y,z) + xyz_shift*box_size
       if (q_single) then
          call pack_xyz(1, natoml_tot, x, y, z, xyzq_loc%sp, loc2glo_ind, xyz_shift)
       else
          call pack_xyz(1, natoml_tot, x, y, z, xyzq_loc%dp, loc2glo_ind, xyz_shift)
       endif

       !-----------------------------------------------------------------------
       !### APH 10/24/14 Force zeroing moved to happen at unpack_reduce_forces
       !-----------------------------------------------------------------------
       !call zero_force_loc()
#if KEY_DOMDEC_GPU==1
    endif
#endif

    sforce = zero

    return
  end subroutine update_local_coord

  ! *
  ! * Updates local coordinates of the home zone
  ! * NOTE: Called before energy calculation
  ! *
  subroutine update_local_home_coord(x, y, z)
    use number,only:zero
    use domdec_common,only:q_single, &
#if KEY_DOMDEC_GPU==1
         q_gpu, &
#endif
         natoml
#if KEY_DOMDEC_GPU==1
    use domdec_util_gpu_mod,only:range_start, range_stop, copy_home_xyzq_to_gpu
    use domdec_common,only:boxx, boxy, boxz
#endif
    implicit none
    ! Input
    real(chm_real), intent(inout) :: x(*), y(*), z(*)

    ! Fill in scoordtab
    call calc_scoordtab()

#if KEY_DOMDEC_GPU==1
    if (q_gpu) then
       ! Pack coordinates to xyzq_loc
       ! NOTE: this is needed for the CPU calculations of bonded interactions and 
       !       1-4 exclusions/interactions
       call range_start('pack: xyzq_loc (home)')
       call pack_xyz(1, natoml, x, y, z, xyzq_loc%sp, loc2glo_ind, xyz_shift)
       call range_stop()
       ! Copy coordinates to GPU
       call copy_home_xyzq_to_gpu(xyzq_loc%sp)
    else
#endif
       ! Pack and recenter coordinates
       ! NOTE: array xyzq_loc is already initialized in nblist_builder
       
       ! Pack (x, y, z) to xyzq_loc and displace by xyz_shift:
       ! xyzq_loc = (x,y,z) + xyz_shift
       if (q_single) then
          call pack_xyz(1, natoml, x, y, z, xyzq_loc%sp, loc2glo_ind, xyz_shift)
       else
          call pack_xyz(1, natoml, x, y, z, xyzq_loc%dp, loc2glo_ind, xyz_shift)
       endif

       !-----------------------------------------------------------------------
       !### APH 10/24/14 Force zeroing moved to happen at unpack_reduce_forces
       !-----------------------------------------------------------------------
       !call zero_force_loc()
#if KEY_DOMDEC_GPU==1
    endif
#endif

    sforce = zero

    return
  end subroutine update_local_home_coord

  ! *
  ! * Updates local coordinates of the import zone
  ! * NOTE: Called before energy calculation
  ! *
  subroutine update_local_import_coord(x, y, z)
    use number,only:zero
    use domdec_common,only:q_single, &
#if KEY_DOMDEC_GPU==1
         q_gpu, &
#endif
         natoml, natoml_tot
#if KEY_DOMDEC_GPU==1
    use domdec_util_gpu_mod,only:range_start, range_stop, copy_import_xyzq_to_gpu
#endif
    implicit none
    ! Input
    real(chm_real), intent(in) :: x(*), y(*), z(*)

#if KEY_DOMDEC_GPU==1
    if (q_gpu) then
       ! Pack coordinates to xyzq_loc
       call range_start('pack: xyzq_loc (import)')
       call pack_xyz(natoml+1, natoml_tot, x, y, z, xyzq_loc%sp, loc2glo_ind, xyz_shift)
       call range_stop()

       ! Copy coordinates to GPU
       call copy_import_xyzq_to_gpu(xyzq_loc%sp)

    else
#endif
       ! Pack and recenter coordinates
       ! NOTE: array xyzq_loc is already initialized in nblist_builder
       
       ! Pack (x, y, z) to xyzq_loc and displace by xyz_shift:
       ! xyzq_loc = (x,y,z) + xyz_shift*box_size
       if (q_single) then
          call pack_xyz(natoml+1, natoml_tot, x, y, z, xyzq_loc%sp, loc2glo_ind, xyz_shift)
       else
          call pack_xyz(natoml+1, natoml_tot, x, y, z, xyzq_loc%dp, loc2glo_ind, xyz_shift)
       endif
#if KEY_DOMDEC_GPU==1
    endif
#endif

    return
  end subroutine update_local_import_coord

  ! *
  ! * Builds vdwtype_loc
  ! *
  subroutine build_local_vdwtype()
    use fast,only:iacnb
    use nblist_util,only:pack_vdwtype
    use domdec_common,only:q_gpu, natoml_tot
#if KEY_DOMDEC_GPU==1
    use domdec_util_gpu_mod,only:copy_vdwtype_to_gpu, range_start, range_stop
#endif
    implicit none

#if KEY_DOMDEC_GPU==1
    if (q_gpu) call range_start('init_local_vdwtype')
#endif
    call init_local_vdwtype()
#if KEY_DOMDEC_GPU==1
    if (q_gpu) call range_stop()
#endif

#if KEY_DOMDEC_GPU==1
    if (q_gpu) then
       ! Pack and send vdwtype to GPU
       call range_start('pack_vdwtype')
       call pack_vdwtype(natoml_tot, iacnb, vdwtype_loc, loc2glo_ind, .true.)
       call range_stop
       call range_start('copy_vdwtype_to_gpu')
       call copy_vdwtype_to_gpu(vdwtype_loc)
       call range_stop()
    else
#endif
       ! Pack vdwtype
       call pack_vdwtype(natoml_tot, iacnb, vdwtype_loc, loc2glo_ind, .false.)
#if KEY_DOMDEC_GPU==1
    endif
#endif

    return
  end subroutine build_local_vdwtype

  ! *
  ! * Builds local coordinate and mapping arrays:
  ! * loc2glo_ind, glo2loc_ind, xyz_shift, xyzq_loc
  ! * NOTE: Called before list builder
  ! *
  subroutine build_local_coord(x, y, z, charge &
#if KEY_LJPME==1
    ,c6coefs &
#endif
    )
    use number,only:zero, one
    use nblist_util,only:init_array
    use domdec_common,only:q_single, natoml_tot, zonelist, q_sort_groups, &
#if KEY_DOMDEC_GPU==1
         q_gpu, zonelist_atom, min_xyz, max_xyz, gpu_code_version, &
#endif
         groupl
    use groupxfast,only:groupsh, group
    use consta,only:ccelec
    use inbnd,only:eps
#if KEY_DOMDEC_GPU==1
    use domdec_util_gpu_mod,only:range_start, range_stop, copy_home_xyzq_to_gpu, &
         copy_import_xyzq_to_gpu, sort_xyzq_on_gpu, wait_sort_xyzq_on_gpu
    use nblist_tilex_sort,only:sort_tilex, copy_cell_start_to_gpu
#if KEY_BLOCK==1
    use lambdam,only:qmld
    use domdec_block,only:set_blocktype_to_gpu
#endif
#endif
    use psf,only:natom
    implicit none
    ! Input
    real(chm_real), intent(in) :: x(*), y(*), z(*), charge(*)
#if KEY_LJPME==1
    real(chm_real), intent(in) :: c6coefs(*)
#endif
    
    ! Initialize local coordinate and mapping arrays
    call init_local_coord()

    ! Fill in scoordtab
    call calc_scoordtab()

#if KEY_DOMDEC_GPU==1
    if (q_gpu) then

       if (gpu_code_version == 2) then
          ! Pack coordinates into xyzq_tmp, displace
          ! NOTE: Scaling removed!
          call range_start('pack_xyzq_loc2glo')
          call pack_xyzq_loc2glo(zonelist, groupl, group, &
               x, y, z, charge,&
#if KEY_LJPME==1
               c6coefs, &
#endif
               groupsh, xyzq_loc%sp, loc2glo_ind, &
               sqrt(one/eps), min_xyz, max_xyz)
          call range_stop()

          ! Sort coordinates
          call sort_xyzq_on_gpu(1, zonelist_atom, xyzq_loc%sp, loc2glo_ind)
          call sort_xyzq_on_gpu(2, zonelist_atom, xyzq_loc%sp, loc2glo_ind)

          ! Wait for sorting to finish
          call wait_sort_xyzq_on_gpu(1)
          call wait_sort_xyzq_on_gpu(2)
          
          ! Build global => local mapping
          call build_glo2loc(natoml_tot, loc2glo_ind, glo2loc_ind)

       else
          ! Pack coordinates into xyzq_tmp, displace
          ! NOTE: Scaling removed!
          call range_start('pack_xyzq_loc2glo')
          call pack_xyzq_loc2glo(zonelist, groupl, group, &
               x, y, z, charge, &
#if KEY_LJPME==1
               c6coefs, &
#endif
               groupsh, xyzq_tmp, loc2glo_ind_tmp, &
               sqrt(one/eps), min_xyz, max_xyz)
          call range_stop()

          call range_start('sort_tilex')
          ! Sort atoms, builds loc2glo_ind, glo2loc_ind, and xyzq_loc
          call sort_tilex(zonelist_atom, xyzq_tmp, xyzq_loc%sp, &
               loc2glo_ind_tmp, loc2glo_ind, natoml_tot, natom)
          call range_stop()
       
          ! Zeros glo2loc. This is important for the build_excl_topology which can tell if atom
          ! is on the node based on glo2loc_ind
          call zero_glo2loc(natom, glo2loc_ind)
          
          ! Build global => local mapping
          call build_glo2loc(natoml_tot, loc2glo_ind, glo2loc_ind)
          
          ! Copy coordinates to GPU
          call copy_home_xyzq_to_gpu(xyzq_loc%sp)
          call copy_import_xyzq_to_gpu(xyzq_loc%sp)
          
          ! Copy cell_start to GPU
          call copy_cell_start_to_gpu()
       endif

       ! Build xyz_shift
       ! NOTE: xyz_shift is used for mapping: xyzq_loc = (x,y,z) + xyz_shift*box_size
       call range_start('build_xyz_shift')
       call build_xyz_shift(zonelist(8), groupl, group, groupsh, glo2loc_ind, xyz_shift)
       call range_stop()

#if KEY_BLOCK==1
       if (qmld) then
          ! Setup blocktype for GPU
          call set_blocktype_to_gpu(natoml_tot, loc2glo_ind)
       endif
#endif
       
    else
#endif
       if (q_sort_groups) call wrndie(-5,'<domdec_local>','sort_groups not implemented yet')

       ! Build local => global mapping
       call pack_loc2glo(zonelist(8), groupl, group, loc2glo_ind)
       ! Build global => local mapping
       call build_glo2loc(natoml_tot, loc2glo_ind, glo2loc_ind)

       ! Build xyz_shift
       ! NOTE: xyz_shift is used for mapping: xyzq_loc = (x, y, z) + xyz_shift*box_size
       call build_xyz_shift(zonelist(8), groupl, group, groupsh, glo2loc_ind, xyz_shift)
       ! Pack (x, y, z, q) to xyzq_loc and displace by xyz_shift:
       ! xyzq_loc = (x,y,z) + xyz_shift*box_size
       if (q_single) then
          call pack_xyzq(natoml_tot, x, y, z, charge,&
#if KEY_LJPME==1                   
               c6coefs, &
#endif
               xyzq_loc%sp, loc2glo_ind, &
               xyz_shift, sqrt(ccelec/eps))
       else
          call pack_xyzq(natoml_tot, x, y, z, charge,&
#if KEY_LJPME==1                   
               c6coefs, &
#endif
               xyzq_loc%dp, loc2glo_ind, &
               xyz_shift, sqrt(ccelec/eps))
       endif

       !-----------------------------------------------------------------------
       !### APH 10/24/14 Force zeroing moved to happen at unpack_reduce_forces
       !-----------------------------------------------------------------------
       !call zero_force_loc()
#if KEY_DOMDEC_GPU==1
    endif
#endif

    sforce = zero

    return
  end subroutine build_local_coord

  ! *
  ! * Sets glo2loc_ind(1:natom) = -1
  ! * 
  subroutine zero_glo2loc(natom, glo2loc_ind)
    implicit none
    ! Input / Output
    integer, intent(in) :: natom
    integer, intent(out) :: glo2loc_ind(:)
    ! Variables
    integer i

!$omp parallel do schedule(static) private(i)
    do i=1,natom
       glo2loc_ind(i) = -1
    enddo
!$omp end parallel do

    return
  end subroutine zero_glo2loc

  ! *
  ! * Sets force_loc to zero
  ! *
  subroutine zero_force_loc()
    use number,only:zero
    use domdec_common,only:natoml_tot
    implicit none
#ifdef _OPENMP
    integer omp_get_thread_num
#endif
    integer tid

!$omp parallel private(tid)
#ifdef _OPENMP
    tid = omp_get_thread_num()
#else
    tid = 0
#endif
    call zero_force_loc_kernel(natoml_tot*3, force_loc(tid)%array)
!$omp end parallel

    return
  end subroutine zero_force_loc

  subroutine zero_force_loc_kernel(narray, array)
    use number,only:zero
    implicit none
    ! Input / Output
    integer, intent(in) :: narray
    real(chm_real), intent(inout) :: array(:)
    ! Variables
    integer i

    do i=1,narray
       array(i) = zero
    enddo
    
    return
  end subroutine zero_force_loc_kernel

  ! *
  ! * Fill in scoordtab
  ! *
  subroutine calc_scoordtab()
    use memory,only:chmalloc
    use image,only:xucell
    implicit none
    ! Variables
    integer i, j, k, ii
    
    if (.not.allocated(scoordtab%dp)) then
       call chmalloc('domdec_local.src','calc_scoordtab','scoordtab%dp',&
            3*27,crl=scoordtab%dp)
       call chmalloc('domdec_local.src','calc_scoordtab','scoordtab%sp',&
            3*27,cr4=scoordtab%sp)
    endif

    ii = 1
    do k=-1,1
       do j=-1,1
          do i=-1,1
             scoordtab%dp(ii)   = i*xucell(1)
             scoordtab%dp(ii+1) = j*xucell(2)
             scoordtab%dp(ii+2) = k*xucell(3)
             scoordtab%sp(ii)   = real(scoordtab%dp(ii), kind=chm_real4)
             scoordtab%sp(ii+1) = real(scoordtab%dp(ii+1), kind=chm_real4)
             scoordtab%sp(ii+2) = real(scoordtab%dp(ii+2), kind=chm_real4)
             ii = ii + 3
          enddo
       enddo
    enddo

    return
  end subroutine calc_scoordtab

  ! *
  ! * Initialize local coordinate and mapping arrays:
  ! * loc2glo_ind, glo2loc_ind, xyz_shift, xyzq_loc
  ! *
  subroutine init_local_coord()
    use memory,only:chmalloc
    use nblist_util,only:&
#if KEY_DOMDEC_GPU==1
         init_array_gpu, &
#endif
         init_array, uninit_array
    use domdec_common,only:natoml, natoml_tot, nthread, &
#if KEY_DOMDEC_GPU==1
         q_gpu, &
#endif
         q_single
#if KEY_DOMDEC_GPU==1
    use domdec_util_gpu_mod,only:set_direct_ncoord_gpu, range_start, range_stop
#endif
    use psf,only:natom
    implicit none
    logical q_reallocated

    ! Initialize (allocate & re-allocate) arrays
    call init_array(loc2glo_ind, natoml_tot)
    call init_array(glo2loc_ind, natom)
    call init_array(xyz_shift, 3, natoml_tot)

    ! Initialize & zero force array
    call init_array(force_loc, nthread, natoml_tot*3, natoml_tot*3, q_reallocated)
    if (q_reallocated) call zero_force_loc()
    q_force_loc_used = .false.
    
#if KEY_DOMDEC_GPU==1
    if (q_gpu) then
       ! Set the system size
       call range_start('set_direct_ncoord_gpu')
       call set_direct_ncoord_gpu(natoml, natoml_tot-natoml)
       call range_stop()
       ! Initialize (allocate & re-allocate) arrays
       call init_array(loc2glo_ind_tmp, natoml_tot)
       call init_array(xyzq_tmp, natoml_tot)
       ! Allocate pinned arrays for CPU -> GPU transfers
       if (.not.q_xyzq_loc_gpu_alloc) then
          ! xyzq_loc%sp was allocated without pinning => deallocate and allocate with pinning
          call uninit_array(xyzq_loc%sp)
       endif
       call init_array_gpu(xyzq_loc%sp, natoml_tot)
       q_xyzq_loc_gpu_alloc = .true.
    else
#endif
       ! Initialize coordinate arrays
       call init_array(xyzq_loc%dp, natoml_tot)
       if (q_single) then
#if KEY_DOMDEC_GPU==1
          if (associated(xyzq_loc%sp) .and. q_xyzq_loc_gpu_alloc) then
             call wrndie(-5,'<domdec_local>', &
                  'Pinned memory for xyzq_loc%sp not deallocated properly')
          endif
#endif
          call init_array(xyzq_loc%sp, natoml_tot)
       endif
       q_xyzq_loc_gpu_alloc = .false.
#if KEY_DOMDEC_GPU==1
    endif
#endif
    
    return
  end subroutine init_local_coord


#if KEY_DOMDEC_GPU==1
  ! *
  ! * Deallocates all pinned memory: xyzq_loc%sp,  vdwtype_loc, mass_loc
  ! *
  subroutine uninit_pinned_memory()
    use nblist_util,only:dealloc_gpu
    implicit none

    if (associated(xyzq_loc%sp) .and. q_xyzq_loc_gpu_alloc) then
       call dealloc_gpu(xyzq_loc%sp)
    endif
    q_xyzq_loc_gpu_alloc = .false.

    if (associated(vdwtype_loc) .and. q_vdwtype_loc_gpu_alloc) then
       call dealloc_gpu(vdwtype_loc)
    endif
    q_vdwtype_loc_gpu_alloc = .false.

    if (associated(mass_loc)) then
       call dealloc_gpu(mass_loc)
    endif

    if (associated(xyz_loc)) then
       call dealloc_gpu(xyz_loc)
    endif

    return
  end subroutine uninit_pinned_memory
#endif

  ! *
  ! * Initialize vdwtype_loc
  ! *
  subroutine init_local_vdwtype()
    use nblist_util,only:&
#if KEY_DOMDEC_GPU==1
         init_array_gpu, dealloc_gpu, &
#endif
         init_arrayp, uninit_arrayp
    use domdec_common,only:q_gpu, q_single, natoml_tot
    implicit none

#if KEY_DOMDEC_GPU==1
    if (q_gpu) then
       ! Allocate pinned arrays for CPU -> GPU transfers
       if (.not.q_vdwtype_loc_gpu_alloc) then
          ! vdwtype_loc was allocated without pinning => deallocate and allocate with pinning
          call uninit_arrayp(vdwtype_loc)
       endif
       call init_array_gpu(vdwtype_loc, natoml_tot+1, 1.4_chm_real)
       q_vdwtype_loc_gpu_alloc = .true.
    else
#endif
       ! Allocate memory for vdwtype_loc
#if KEY_DOMDEC_GPU==1
       if (associated(vdwtype_loc) .and. q_vdwtype_loc_gpu_alloc) then
          call wrndie(-5,'<domdec_local>',&
               'Pinned memory for vdwtype_loc not deallocated properly')
       endif
#endif
       call init_arrayp(vdwtype_loc, natoml_tot, natoml_tot)
       q_vdwtype_loc_gpu_alloc = .false.
#if KEY_DOMDEC_GPU==1
    endif
#endif

    return
  end subroutine init_local_vdwtype

  ! *
  ! * Uninitialize local coordinate and mapping arrays:
  ! * loc2glo_ind, glo2loc_ind, xyz_shift, xyzq_loc
  ! *
  subroutine uninit_local_coord()
    use memory,only:chmdealloc
    use nblist_util,only:uninit_array
#if KEY_DOMDEC_GPU==1
    use nblist_util,only:dealloc_gpu
#endif
    implicit none

    call uninit_array(force_loc)
    call uninit_array(buf)

    if (allocated(loc2glo_ind)) then
       call chmdealloc('domdec_local.src','uninit_local_coord','loc2glo_ind',&
            size(loc2glo_ind),intg=loc2glo_ind)
    endif

    if (allocated(glo2loc_ind)) then
       call chmdealloc('domdec_local.src','uninit_local_coord','glo2loc_ind',&
            size(glo2loc_ind),intg=glo2loc_ind)
    endif

#if KEY_DOMDEC_GPU==1
    if (allocated(loc2glo_ind_tmp)) then
       call chmdealloc('domdec_local.src','uninit_local_coord','loc2glo_ind_tmp',&
            size(loc2glo_ind_tmp),intg=loc2glo_ind_tmp)
    endif

    if (associated(xyzq_tmp)) then
       deallocate(xyzq_tmp)
    endif

#endif

    ! xyz_shift
    if (allocated(xyz_shift)) then
       call chmdealloc('domdec_local.src','uninit_local_coord','xyz_shift',&
            size(xyz_shift,1),size(xyz_shift,2),cr4=xyz_shift)
    endif

    ! xyzq_loc%dp
    if (associated(xyzq_loc%dp)) then
       deallocate(xyzq_loc%dp)
    endif

    ! xyzq_loc%sp
    if (associated(xyzq_loc%sp)) then
#if KEY_DOMDEC_GPU==1
       if (q_xyzq_loc_gpu_alloc) then
          call dealloc_gpu(xyzq_loc%sp)
       else
#endif
          call uninit_array(xyzq_loc%sp)
#if KEY_DOMDEC_GPU==1
       endif
#endif
    endif

    if (allocated(scoordtab%dp)) then
       call chmdealloc('domdec_local.src','uninit_local_coord','scoordtab%dp',&
            3*27,crl=scoordtab%dp)
       call chmdealloc('domdec_local.src','uninit_local_coord','scoordtab%sp',&
            3*27,cr4=scoordtab%sp)
    endif

    return
  end subroutine uninit_local_coord

  ! *
  ! * Uninitialize vdwtype_loc
  ! *
  subroutine uninit_local_vdwtype()
    use nblist_util,only:uninit_arrayp
#if KEY_DOMDEC_GPU==1
    use nblist_util,only:dealloc_gpu
#endif
    implicit none

    ! vdwtype_loc
    if (associated(vdwtype_loc)) then
#if KEY_DOMDEC_GPU==1
       if (q_vdwtype_loc_gpu_alloc) then
          call dealloc_gpu(vdwtype_loc)
       else
#endif
          call uninit_arrayp(vdwtype_loc)
#if KEY_DOMDEC_GPU==1
       endif
#endif
    endif

    return
  end subroutine uninit_local_vdwtype

  ! *
  ! * Builds global => local mapping index
  ! *
  subroutine build_glo2loc(natoml, loc2glo_ind, glo2loc_ind)
    implicit none
    ! Input / Output
    integer, intent(in) :: natoml
    integer, intent(in) :: loc2glo_ind(:)
    integer, intent(out) :: glo2loc_ind(:)
    ! Variables
    integer i

!$omp parallel do schedule(static) private(i)
    do i=1,natoml
       glo2loc_ind(loc2glo_ind(i)+1) = i-1
    enddo
!$omp end parallel do

    return
  end subroutine build_glo2loc

  ! *
  ! * Builds simple linear loc2glo (no sorting)
  ! *
  subroutine pack_loc2glo(ngroupl, groupl, group, loc2glo_ind)
#ifdef _OPENMP
    use nblist_util,only:cumsum_exclusive
#endif
    use groupxfast,only:group_out
    implicit none
    ! Input / Output
    integer, intent(in) :: ngroupl, groupl(*), group(*)
    integer, intent(out) :: loc2glo_ind(:)
    ! Variables
    integer i, j, ig, is, iq
    integer k

#ifdef _OPENMP
    call alloc_realloc_coordpos(ngroupl)

!$omp parallel do schedule(static) private(ig, i, is, iq)
    do ig=1,ngroupl
       ! i is the unpacked group index
       i = groupl(ig)
       call group_out(group(i), is, iq)
       coordpos(ig) = iq-is+1
    enddo
!$omp end parallel do

    ! Calculate exclusive cumulative sum on coordpos
    call cumsum_exclusive(coordpos, ngroupl)

!$omp parallel do schedule(static) private(ig, i, is, iq, j, k)
    do ig=1,ngroupl
       ! i is the unpacked group index
       i = groupl(ig)
       call group_out(group(i), is, iq)
       ! Set loc2glo
       j = coordpos(ig) + 1
       do k=is,iq
          loc2glo_ind(j) = k-1
          j = j + 1
       enddo
    enddo
!$omp end parallel do

#else
    j = 1
    do ig=1,ngroupl
       ! i is the unpacked group index
       i = groupl(ig)
       call group_out(group(i), is, iq)
       ! Set packed coordinates
       do k=is,iq
          loc2glo_ind(j) = k-1
          j = j + 1
       enddo
    enddo
#endif

    return
  end subroutine pack_loc2glo

  ! *
  ! * Build xyz_shift from groupsh(1:3,:)
  ! *
  subroutine build_xyz_shift(ngroupl, groupl, group, groupsh, glo2loc, xyz_shift)
    use groupxfast,only:group_out
    use domdec_local_types,only:xyzq_dp_t
    implicit none
    ! Input / Output
    integer, intent(in) :: ngroupl, groupl(:), group(:)
    real(chm_real), intent(in) :: groupsh(3,*)
    integer, intent(in) :: glo2loc(:)
    real(chm_real4), intent(out) :: xyz_shift(3,*)
    ! Variables
    integer i, k, ig, is, iq, ind

!!$    j = 1
!!$    do ig=1,ngroupl
!!$       ! i is the unpacked group index
!!$       i = groupl(ig)
!!$       call group_out(group(i), is, iq)
!!$       ! Set packed coordinates
!!$       do k=is,iq
!!$          xyz_shift(1:3,j) = groupsh(1:3,i)
!!$          j = j + 1
!!$       enddo
!!$    enddo

!$omp parallel do schedule(static) private(ig, i, is, iq, k, ind)
    do ig=1,ngroupl
       ! i is the unpacked group index
       i = groupl(ig)
       call group_out(group(i), is, iq)
       ! Set packed coordinates
       do k=is,iq
          ind = glo2loc(k)+1
          xyz_shift(1:3,ind) = groupsh(1:3,i)
       enddo
    enddo
!$omp end parallel do

    return
  end subroutine build_xyz_shift

  ! *
  ! * Packs coordinates (x, y, z) using loc2glo
  ! * Optionally displaces the coordinates by xyz_shift(3,:)*box_size
  ! * NOTE: Does NOT pack charge (q)
  ! *
  subroutine pack_xyz_sp(istart, iend, x, y, z, xyzq, loc2glo, xyz_shift)
    use domdec_common,only:boxx, boxy, boxz
    use domdec_local_types,only:xyzq_sp_t
    implicit none
    ! Input / Output
    integer, intent(in) :: istart, iend
    real(chm_real), intent(in) :: x(*), y(*), z(*)
    type(xyzq_sp_t), intent(out) :: xyzq(:)
    integer, intent(in) :: loc2glo(:)
    real(chm_real4), intent(in), optional :: xyz_shift(3,*)
    ! Variables
    integer i, ind

    if (present(xyz_shift)) then
!$omp parallel do schedule(static) private(i, ind)
       do i=istart,iend
          ind = loc2glo(i)+1
          xyzq(i)%x = real(x(ind) + xyz_shift(1,i)*boxx,kind=chm_real4)
          xyzq(i)%y = real(y(ind) + xyz_shift(2,i)*boxy,kind=chm_real4)
          xyzq(i)%z = real(z(ind) + xyz_shift(3,i)*boxz,kind=chm_real4)
       enddo
!$omp end parallel do
    else
!$omp parallel do schedule(static) private(i, ind)
       do i=istart,iend
          ind = loc2glo(i)+1
          xyzq(i)%x = real(x(ind),kind=chm_real4)
          xyzq(i)%y = real(y(ind),kind=chm_real4)
          xyzq(i)%z = real(z(ind),kind=chm_real4)
       enddo
!$omp end parallel do
    endif

    return
  end subroutine pack_xyz_sp

  ! *
  ! * Packs coordinates (x, y, z) using loc2glo
  ! * Optionally displaces the coordinates by xyz_shift(3,:)
  ! * NOTE: Does NOT pack charge (q)
  ! *
  subroutine pack_xyz_dp(istart, iend, x, y, z, xyzq, loc2glo, xyz_shift)
    use domdec_common,only:boxx, boxy, boxz
    use domdec_local_types,only:xyzq_dp_t
    implicit none
    ! Input / Output
    integer, intent(in) :: istart, iend
    real(chm_real), intent(in) :: x(*), y(*), z(*)
    type(xyzq_dp_t), intent(out) :: xyzq(:)
    integer, intent(in) :: loc2glo(:)
    real(chm_real4), intent(in), optional :: xyz_shift(3,*)
    ! Variables
    integer i, ind

    if (present(xyz_shift)) then
!$omp parallel do schedule(static) private(i, ind)
       do i=istart,iend
          ind = loc2glo(i)+1
          xyzq(i)%x = x(ind) + xyz_shift(1,i)*boxx
          xyzq(i)%y = y(ind) + xyz_shift(2,i)*boxy
          xyzq(i)%z = z(ind) + xyz_shift(3,i)*boxz
       enddo
!$omp end parallel do
    else
!$omp parallel do schedule(static) private(i, ind)
       do i=istart,iend
          ind = loc2glo(i)+1
          xyzq(i)%x = x(ind)
          xyzq(i)%y = y(ind)
          xyzq(i)%z = z(ind)
       enddo
!$omp end parallel do
    endif

    return
  end subroutine pack_xyz_dp

  ! *
  ! * Packs coordinates and charge (x, y, z, q) using loc2glo_ind.
  ! * Optionally multiplies the charges by qscale
  ! *
  subroutine pack_xyzq_nodisp_sp(natoml, x, y, z, charge, &
#if KEY_LJPME==1                   
                                 c6coefs, &
#endif
                                 xyzq, loc2glo_ind, qscale)
    use domdec_local_types,only:xyzq_sp_t
    implicit none
    ! Input / Output
    integer, intent(in) :: natoml
    real(chm_real), intent(in) :: x(*), y(*), z(*), charge(*)
#if KEY_LJPME==1                   
    real(chm_real), intent(in) :: c6coefs(*)
#endif
    type(xyzq_sp_t), intent(out) :: xyzq(:)
    integer, intent(in) :: loc2glo_ind(:)
    real(chm_real), intent(in), optional :: qscale
    ! Variables
    integer i, j

    if (present(qscale)) then
!$omp parallel do schedule(static) private(i, j)
       do i=1,natoml
          j = loc2glo_ind(i)+1
          xyzq(i)%x = real(x(j),kind=chm_real4)
          xyzq(i)%y = real(y(j),kind=chm_real4)
          xyzq(i)%z = real(z(j),kind=chm_real4)
          xyzq(i)%q = real(charge(j)*qscale,kind=chm_real4)
#if KEY_LJPME==1
          xyzq(i)%c6 = real(c6coefs(j),kind=chm_real4)
#endif
       enddo
!$omp end parallel do
    else
!$omp parallel do schedule(static) private(i, j)
       do i=1,natoml
          j = loc2glo_ind(i)+1
          xyzq(i)%x = real(x(j),kind=chm_real4)
          xyzq(i)%y = real(y(j),kind=chm_real4)
          xyzq(i)%z = real(z(j),kind=chm_real4)
          xyzq(i)%q = real(charge(j),kind=chm_real4)
#if KEY_LJPME==1
          xyzq(i)%c6 = real(c6coefs(j),kind=chm_real4)
#endif
       enddo
!$omp end parallel do
    endif

    return
  end subroutine pack_xyzq_nodisp_sp

  ! *
  ! * Packs coordinates and charge (x, y, z, q) using loc2glo_ind.
  ! * Optionally multiplies the charges by qscale
  ! *
  subroutine pack_xyzq_sp(natoml, x, y, z, charge,&
#if KEY_LJPME==1                   
                          c6coefs, &
#endif
                          xyzq, loc2glo_ind, xyz_shift, qscale)
    use domdec_local_types,only:xyzq_sp_t
    use domdec_common,only:boxx, boxy, boxz
    implicit none
    ! Input / Output
    integer, intent(in) :: natoml
    real(chm_real), intent(in) :: x(*), y(*), z(*), charge(*)
#if KEY_LJPME==1                   
    real(chm_real), intent(in) :: c6coefs(*)
#endif
    type(xyzq_sp_t), intent(out) :: xyzq(:)
    integer, intent(in) :: loc2glo_ind(:)
    real(chm_real4), intent(in) :: xyz_shift(3,*)
    real(chm_real), intent(in), optional :: qscale
    ! Variables
    integer i, j

    if (present(qscale)) then
!$omp parallel do schedule(static) private(i, j)
       do i=1,natoml
          j = loc2glo_ind(i)+1
          xyzq(i)%x = real(x(j) + xyz_shift(1,i)*boxx,kind=chm_real4)
          xyzq(i)%y = real(y(j) + xyz_shift(2,i)*boxy,kind=chm_real4)
          xyzq(i)%z = real(z(j) + xyz_shift(3,i)*boxz,kind=chm_real4)
          xyzq(i)%q = real(charge(j)*qscale,kind=chm_real4)
#if KEY_LJPME==1
          xyzq(i)%c6 = real(c6coefs(j),kind=chm_real4)
#endif
       enddo
!$omp end parallel do
    else
!$omp parallel do schedule(static) private(i, j)
       do i=1,natoml
          j = loc2glo_ind(i)+1
          xyzq(i)%x = real(x(j) + xyz_shift(1,i)*boxx,kind=chm_real4)
          xyzq(i)%y = real(y(j) + xyz_shift(2,i)*boxy,kind=chm_real4)
          xyzq(i)%z = real(z(j) + xyz_shift(3,i)*boxz,kind=chm_real4)
          xyzq(i)%q = real(charge(j),kind=chm_real4)
#if KEY_LJPME==1
          xyzq(i)%c6 = real(c6coefs(j),kind=chm_real4)
#endif
       enddo
!$omp end parallel do
    endif

    return
  end subroutine pack_xyzq_sp

  ! *
  ! * Packs coordinates and charge (x, y, z, q) using loc2glo_ind.
  ! * Optionally multiplies the charges by qscale
  ! *
  subroutine pack_xyzq_dp(natoml, x, y, z, charge,&
#if KEY_LJPME==1                   
                          c6coefs, &
#endif
                          xyzq, loc2glo_ind, xyz_shift, qscale)
    use domdec_common,only:boxx, boxy, boxz
    use domdec_local_types,only:xyzq_dp_t
    implicit none
    ! Input / Output
    integer, intent(in) :: natoml
    real(chm_real), intent(in) :: x(*), y(*), z(*), charge(*)
#if KEY_LJPME==1                   
    real(chm_real), intent(in) :: c6coefs(*)
#endif
    type(xyzq_dp_t), intent(out) :: xyzq(:)
    integer, intent(in) :: loc2glo_ind(:)
    real(chm_real4), intent(in) :: xyz_shift(3,*)
    real(chm_real), intent(in), optional :: qscale
    ! Variables
    integer i, j

    if (present(qscale)) then
!$omp parallel do schedule(static) private(i, j)
       do i=1,natoml
          j = loc2glo_ind(i)+1
          xyzq(i)%x = x(j) + xyz_shift(1,i)*boxx
          xyzq(i)%y = y(j) + xyz_shift(2,i)*boxy
          xyzq(i)%z = z(j) + xyz_shift(3,i)*boxz
          xyzq(i)%q = charge(j)*qscale
#if KEY_LJPME==1
          xyzq(i)%c6 = c6coefs(j)
#endif
       enddo
!$omp end parallel do
    else
!$omp parallel do schedule(static) private(i, j)
       do i=1,natoml
          j = loc2glo_ind(i)+1
          xyzq(i)%x = x(j) + xyz_shift(1,i)*boxx
          xyzq(i)%y = y(j) + xyz_shift(2,i)*boxy
          xyzq(i)%z = z(j) + xyz_shift(3,i)*boxz
          xyzq(i)%q = charge(j)
#if KEY_LJPME==1
          xyzq(i)%c6 = c6coefs(j)
#endif
       enddo
!$omp end parallel do
    endif

    return
  end subroutine pack_xyzq_dp

  ! *
  ! * Packs charge using loc2glo_ind.
  ! * Optionally multiplies the charges by qscale
  ! *
  subroutine pack_q_sp(natoml, charge, xyzq, loc2glo_ind, qscale)
    use domdec_local_types,only:xyzq_sp_t
    implicit none
    ! Input / Output
    integer, intent(in) :: natoml
    real(chm_real), intent(in) :: charge(*)
    type(xyzq_sp_t), intent(out) :: xyzq(:)
    integer, intent(in) :: loc2glo_ind(:)
    real(chm_real), intent(in), optional :: qscale
    ! Variables
    integer i, j

    if (present(qscale)) then
!$omp parallel do schedule(static) private(i, j)
       do i=1,natoml
          j = loc2glo_ind(i)+1
          xyzq(i)%q = real(charge(j)*qscale,kind=chm_real4)
       enddo
!$omp end parallel do
    else
!$omp parallel do schedule(static) private(i, j)
       do i=1,natoml
          j = loc2glo_ind(i)+1
          xyzq(i)%q = real(charge(j),kind=chm_real4)
       enddo
!$omp end parallel do
    endif

    return
  end subroutine pack_q_sp

  ! *
  ! * Packs charge using loc2glo_ind.
  ! * Optionally multiplies the charges by qscale
  ! *
  subroutine pack_q_dp(natoml, charge, xyzq, loc2glo_ind, qscale)
    use domdec_local_types,only:xyzq_dp_t
    implicit none
    ! Input / Output
    integer, intent(in) :: natoml
    real(chm_real), intent(in) :: charge(*)
    type(xyzq_dp_t), intent(out) :: xyzq(:)
    integer, intent(in) :: loc2glo_ind(:)
    real(chm_real), intent(in), optional :: qscale
    ! Variables
    integer i, j

    if (present(qscale)) then
!$omp parallel do schedule(static) private(i, j)
       do i=1,natoml
          j = loc2glo_ind(i)+1
          xyzq(i)%q = charge(j)*qscale
       enddo
!$omp end parallel do
    else
!$omp parallel do schedule(static) private(i, j)
       do i=1,natoml
          j = loc2glo_ind(i)+1
          xyzq(i)%q = charge(j)
       enddo
!$omp end parallel do
    endif

    return
  end subroutine pack_q_dp

  ! *
  ! * Unpacks and adds forces
  ! *
  subroutine unpack_add_force_cpu(ngroupl, groupl, group, &
       forcex, forcey, forcez, &
       force, group_order)
    use groupxfast,only:group_out
    implicit none
    ! Input / Output
    integer, intent(in) :: ngroupl, groupl(*), group(*)
    real(chm_real), intent(in) :: force(*)
    real(chm_real), intent(out) :: forcex(*), forcey(*), forcez(*)
    integer, intent(in), optional :: group_order(*)
    ! Variables
    integer i, j, ig, is, iq

    j = 1
    if (present(group_order)) then
       do ig=1,ngroupl
          ! i is the unpacked group index
          i = groupl(group_order(ig))
          call unpack_kernel()
       enddo
    else
       do ig=1,ngroupl
          ! i is the unpacked group index
          i = groupl(ig)
          call unpack_kernel()
       enddo
    endif
    return

  contains
    subroutine unpack_kernel
      implicit none
      integer k

      call group_out(group(i), is, iq)
      ! Set packed coordinates
      do k=is,iq
         forcex(k) = forcex(k) + force(j)
         forcey(k) = forcey(k) + force(j+1)
         forcez(k) = forcez(k) + force(j+2)
         j = j + 3
      enddo
      return
    end subroutine unpack_kernel

  end subroutine unpack_add_force_cpu

  ! *
  ! * Unpacks, reduces (for OpenMP), adds force to (forcex, forcey, forcez) and zeros original forces
  ! *
  subroutine unpack_reduce_zero_force(natoml, loc2glo_ind, &
       forcex, forcey, forcez, force)
    use domdec_local_types,only:crlarray_t
    use number,only:zero
#ifdef _OPENMP
    use nblist_util,only:init_array
    use domdec_common,only:cpu_vendor, CPU_AMD
    use domdec_common,only:nthread
#endif
    implicit none
    ! Input / Output
    integer, intent(in) :: natoml
    integer, intent(in) :: loc2glo_ind(:)
    type(crlarray_t), intent(inout) :: force(0:*)
    real(chm_real), intent(out) :: forcex(*), forcey(*), forcez(*)
    ! Variables
    integer i, ii, k
#ifdef _OPENMP
    integer j
    real(chm_real) forcex_sum, forcey_sum, forcez_sum
    integer omp_get_thread_num
    integer tid, istart, iend, bufend, kk, jj
    integer, parameter :: bufsize=2048
#endif

#ifdef _OPENMP
    if (cpu_vendor == CPU_AMD) then
       ! Use buffered version for AMD CPUs (improved performance on Titan)
       call init_array(buf, nthread, bufsize*3, bufsize*3)
!$omp parallel private(tid, istart, iend, i, ii, bufend, kk, j, k, jj)
       tid = omp_get_thread_num()
       istart = natoml*tid/nthread + 1
       iend = natoml*(tid+1)/nthread
       do i=istart,iend,bufsize
          ii = i*3 - 2
          bufend = min(bufsize, iend-i+1)
          do kk=0,(bufend-1)*3,3
             buf(tid)%array(kk+1) = force(tid)%array(ii+kk)
             buf(tid)%array(kk+2) = force(tid)%array(ii+kk+1)
             buf(tid)%array(kk+3) = force(tid)%array(ii+kk+2)
             force(tid)%array(ii+kk)   = zero
             force(tid)%array(ii+kk+1) = zero
             force(tid)%array(ii+kk+2) = zero
          enddo
          do j=1,nthread-1
             jj = mod(j + tid, nthread)
             do kk=0,(bufend-1)*3,3
                buf(tid)%array(kk+1) = buf(tid)%array(kk+1) + force(jj)%array(ii+kk)
                buf(tid)%array(kk+2) = buf(tid)%array(kk+2) + force(jj)%array(ii+kk+1)
                buf(tid)%array(kk+3) = buf(tid)%array(kk+3) + force(jj)%array(ii+kk+2)
                force(jj)%array(ii+kk)   = zero
                force(jj)%array(ii+kk+1) = zero
                force(jj)%array(ii+kk+2) = zero
             enddo
          enddo
          do kk=0,bufend-1
             k = loc2glo_ind(i+kk)+1
             forcex(k) = forcex(k) + buf(tid)%array(kk*3+1)
             forcey(k) = forcey(k) + buf(tid)%array(kk*3+2)
             forcez(k) = forcez(k) + buf(tid)%array(kk*3+3)
          enddo
       enddo
!$omp end parallel
    else
!$omp parallel do schedule(static) private(i, ii, k, j, forcex_sum, forcey_sum, forcez_sum)
       do i=1,natoml
          ii = i*3 - 2
          forcex_sum = zero
          forcey_sum = zero
          forcez_sum = zero
          do j=0,nthread-1
             forcex_sum = forcex_sum + force(j)%array(ii)
             forcey_sum = forcey_sum + force(j)%array(ii+1)
             forcez_sum = forcez_sum + force(j)%array(ii+2)
             force(j)%array(ii) = zero
             force(j)%array(ii+1) = zero
             force(j)%array(ii+2) = zero
          enddo
          k = loc2glo_ind(i)+1
          forcex(k) = forcex(k) + forcex_sum
          forcey(k) = forcey(k) + forcey_sum
          forcez(k) = forcez(k) + forcez_sum
       enddo
!$omp end parallel do
    endif
#else
    do i=1,natoml
       ii = i*3 - 2
       k = loc2glo_ind(i)+1
       forcex(k) = forcex(k) + force(0)%array(ii)
       forcey(k) = forcey(k) + force(0)%array(ii+1)
       forcez(k) = forcez(k) + force(0)%array(ii+2)
       force(0)%array(ii) = zero
       force(0)%array(ii+1) = zero
       force(0)%array(ii+2) = zero
    enddo
#endif

    return
  end subroutine unpack_reduce_zero_force

  ! -------------------------------------------------------------------------------------
  ! Following subroutines for domdec_gpu only
  ! -------------------------------------------------------------------------------------
#if KEY_DOMDEC_GPU==1
  ! *
  ! * Packs coordinates (x, y, z, q), creating xyzq and loc2glo_ind
  ! * Optionally, finds the minimum (x, y, z) for each zone
  ! *
  subroutine pack_xyzq_loc2glo_sp(zonelist, groupl, group, x, y, z, charge,&
#if KEY_LJPME==1
                                  c6coefs, &
#endif
                                  groupsh, xyzq, loc2glo_ind, qscale, &
       min_xyz, max_xyz)
    use number,only:half
    use inbnd,only:cutnb
    use nblist_util,only:cumsum_exclusive
    use domdec_local_types,only:xyzq_sp_t
    use groupxfast,only:group_out
    use domdec_common,only:boxx, boxy, boxz
    use domdec_dlb,only:get_zone_corners
    implicit none
    ! Input / Output
    integer, intent(in) :: zonelist(8), groupl(*), group(*)
    real(chm_real), intent(in) :: x(*), y(*), z(*), charge(*), groupsh(3,*)
#if KEY_LJPME==1                   
    real(chm_real), intent(in) :: c6coefs(*)
#endif
    type(xyzq_sp_t), intent(out) :: xyzq(:)
    integer, intent(out) :: loc2glo_ind(:)
    real(chm_real), intent(in) :: qscale
    real(chm_real4), intent(out), optional :: min_xyz(3,8), max_xyz(3,8)
    ! Variables
    logical q_find_min_max    
    real(chm_real4) xj, yj, zj
    real(chm_real4) min_x, min_y, min_z, max_x, max_y, max_z
    real(chm_real) x0, y0, z0, x1, y1, z1
    integer i, j, k, ig, is, iq, ngroupl
    integer izone, ig_start, ig_end

    q_find_min_max = .false.
    if (present(min_xyz) .and. present(max_xyz)) then
       q_find_min_max = .true.
    elseif (present(min_xyz) .or. present(max_xyz)) then
       call wrndie(-5,'<domdec_local>',&
            'pack_xyzq_loc2glo_sp: both min_xyz and max_xyz must be present if one of them is')
    endif

    ngroupl = zonelist(8)

#ifdef _OPENMP
    call alloc_realloc_coordpos(ngroupl)

!$omp parallel do schedule(static) private(ig, i, is, iq)
    do ig=1,ngroupl
       ! i is the unpacked group index
       i = groupl(ig)
       call group_out(group(i), is, iq)
       coordpos(ig) = iq-is+1
    enddo
!$omp end parallel do

    ! Calculate prefix sum on coordpos(:)
    call cumsum_exclusive(coordpos, ngroupl)

    if (q_find_min_max) then
!$omp parallel private(izone, ig_start, ig_end)
       do izone=1,8
          if (izone == 1) then
             ig_start = 1
          else
             ig_start = zonelist(izone-1) + 1
          endif
          ig_end = zonelist(izone)
          min_x = 1.0e20_chm_real4
          min_y = 1.0e20_chm_real4
          min_z = 1.0e20_chm_real4
          max_x = -1.0e20_chm_real4
          max_y = -1.0e20_chm_real4
          max_z = -1.0e20_chm_real4
          if (ig_end >= ig_start) then
!$omp barrier
!$omp do schedule(static) private(i, is, iq, j, k, xj, yj, zj) &
!$omp&                    reduction(min:min_x, min_y, min_z) &
!$omp&                    reduction(max:max_x, max_y, max_z)
             do ig=ig_start,ig_end
                ! i is the unpacked group index
                i = groupl(ig)
                call group_out(group(i), is, iq)
                ! Set packed coordinates
                j = coordpos(ig) + 1
                do k=is,iq
                   xj = x(k) + groupsh(1,i)*boxx
                   yj = y(k) + groupsh(2,i)*boxy
                   zj = z(k) + groupsh(3,i)*boxz
                   xyzq(j)%x = xj
                   xyzq(j)%y = yj
                   xyzq(j)%z = zj
                   xyzq(j)%q = charge(k)*qscale
#if KEY_LJPME==1
                   xyzq(j)%c6 = c6coefs(k)
#endif
                   min_x = min(min_x, xj)
                   min_y = min(min_y, yj)
                   min_z = min(min_z, zj)
                   max_x = max(max_x, xj)
                   max_y = max(max_y, yj)
                   max_z = max(max_z, zj)
                   loc2glo_ind(j) = k-1
                   j = j + 1
                enddo
             enddo
!$omp end do
          endif
!$omp single
          call get_zone_corners(izone, cutnb, x0, y0, z0, x1, y1, z1)
          x0 = x0 - boxx*half
          x1 = x1 - boxx*half
          y0 = y0 - boxy*half
          y1 = y1 - boxy*half
          z0 = z0 - boxz*half
          z1 = z1 - boxz*half
          min_x = min(min_x, x0)
          min_y = min(min_y, y0)
          min_z = min(min_z, z0)
          max_x = max(max_x, x1)
          max_y = max(max_y, y1)
          max_z = max(max_z, z1)
          min_xyz(1:3,izone) = (/ min_x, min_y, min_z /)
          max_xyz(1:3,izone) = (/ max_x, max_y, max_z /)
!$omp end single
       enddo
!$omp end parallel
    else
!$omp parallel do schedule(static) private(ig, i, is, iq, j, k)
       do ig=1,ngroupl
          ! i is the unpacked group index
          i = groupl(ig)
          call group_out(group(i), is, iq)
          ! Set packed coordinates
          j = coordpos(ig) + 1
          do k=is,iq
             xyzq(j)%x = x(k) + groupsh(1,i)*boxx
             xyzq(j)%y = y(k) + groupsh(2,i)*boxy
             xyzq(j)%z = z(k) + groupsh(3,i)*boxz
             xyzq(j)%q = charge(k)*qscale
#if KEY_LJPME==1
             xyzq(j)%c6 = c6coefs(k)
#endif
             loc2glo_ind(j) = k-1
             j = j + 1
          enddo
       enddo
!$omp end parallel do
    endif

#else
    if (q_find_min_max) then
       call wrndie(-5,'<domdec_local>',&
            'pack_xyzq_loc2glo_sp: find_min_max not implemented when no OPENMP is used')
    endif
    j = 1
    do ig=1,ngroupl
       ! i is the unpacked group index
       i = groupl(ig)
       call group_out(group(i), is, iq)
       ! Set packed coordinates
       do k=is,iq
          xyzq(j)%x = x(k) + groupsh(1,i)*boxx
          xyzq(j)%y = y(k) + groupsh(2,i)*boxy
          xyzq(j)%z = z(k) + groupsh(3,i)*boxz
          xyzq(j)%q = charge(k)*qscale
#if KEY_LJPME==1
          xyzq(j)%c6 = c6coefs(k)
#endif
          loc2glo_ind(j) = k-1
          j = j + 1
       enddo
    enddo
#endif

    return
  end subroutine pack_xyzq_loc2glo_sp

  ! *
  ! * Unpacks and adds forces
  ! *
  subroutine unpack_add_force_gpu(natoml, atom_index, &
       forcex, forcey, forcez, force)
    implicit none
    ! Input / Output
    integer, intent(in) :: natoml
    integer, intent(in) :: atom_index(*)
    real(chm_real), intent(in) :: force(*)
    real(chm_real), intent(out) :: forcex(*), forcey(*), forcez(*)
    ! Variables
    integer i, j, k

    j = 1
    do i=1,natoml
       k = atom_index(i)
       forcex(k) = forcex(k) + force(j)
       forcey(k) = forcey(k) + force(j+1)
       forcez(k) = forcez(k) + force(j+2)
       j = j + 3
    enddo

    return
  end subroutine unpack_add_force_gpu

#endif
  ! *
  ! * Allocate & reallocate coordpos(:) -array
  ! *
  subroutine alloc_realloc_coordpos(n)
    use memory
    implicit none
    ! Input
    integer, intent(in) :: n

    if (allocated(coordpos) .and. size(coordpos) < n+1) then
       call chmdealloc('domdec_local.src','alloc_realloc_coordpos','coordpos',&
            size(coordpos),intg=coordpos)
    endif

    if (.not.allocated(coordpos)) then
       call chmalloc('domdec_local.src','alloc_realloc_coordpos','coordpos',&
            int(n*1.2),intg=coordpos)
    endif

    return
  end subroutine alloc_realloc_coordpos

  ! *
  ! * Deallocate coordpos(:) -array
  ! *
  subroutine dealloc_coordpos()
    use memory,only:chmdealloc
    implicit none

    if (allocated(coordpos)) then
       call chmdealloc('nblist_util.src','dealloc_coordpos','coordpos',&
            size(coordpos),intg=coordpos)
    endif

    return
  end subroutine dealloc_coordpos

#endif

end module domdec_local
