module domdec_r2d_comm

  ! *
  ! * Domain decomposition reciprocal-to-direct and reciprocal-to-reciprocal communication
  ! * NOTE: Only reciprocal nodes call subroutines in this module
  ! *

#if KEY_DOMDEC==1 /*domdec_main*/
#if KEY_PARALLEL==1 /*parallel*/
!#if KEY_CMPI==0 /*not_cmpi*/
  use chm_kinds
  use dimens_fcm
  use nblist_types,only:intarray_t, intarray2_t
  use domdec_local_types
  implicit none
  private

  ! Communication buffers
  integer(int_byte), pointer, dimension(:) :: commbuffer2
  integer, allocatable, dimension(:) :: commbuffer2size
  integer, allocatable, dimension(:) :: commbuffer2pos

  integer, allocatable, dimension(:) :: nrecv, nsend
  integer, allocatable, dimension(:) :: recip_atoml, recip_natoml
  integer, allocatable, dimension(:) :: grid_atom_pos

  integer, allocatable, dimension(:) :: cons_atom_pos

  ! Reciprocal force arrays.
  ! NOTE: These are used only when direct nodes are also reciprocal nodes
  !       (q_split = .false AND ndirect > 1)
  real(chm_real), allocatable, dimension(:) :: recipforcex, recipforcey, recipforcez

  ! Used for determining the nodes to which atom belongs
  integer, allocatable, dimension(:,:) :: filter_atom_y, filter_atom_z

  type atomlist_th_t
     ! atomlist(inode)%array(1:n_atomlist(inode)) = list of atom indices for node inode
     type(intarray_t), allocatable, dimension(:) :: atomlist
     ! n_atomlist(inode) = number of atoms for node inode
     integer, allocatable, dimension(:) :: n_atomlist
  end type atomlist_th_t

  ! Each thread has its own list:
  ! atomlist_th(0:nthread-1)%atomlist(0:nrecip)%array(1:)
  !
  ! Note atomlist(0) is used to store the dummy data, only needs array of size 1
  !
  ! atomlist_th(tid)%atomlist(inode)%array(:) = list of atoms thread tid found for node inode
  type(atomlist_th_t), target, allocatable, dimension(:) :: atomlist_th
  ! atomlist_p => atomlist_th(0)%atomlist, atomlist_p(0:nrecip)%array(:)
  ! atomlist_p(inode)%array(:) = list of atoms for node inode
  type(intarray_t), pointer, dimension(:) :: atomlist_p
  ! grid_atom_p => atomlist_th(0)%atomlist(mynod_split+1)%array
  ! atomlist_th(0)%atomlist(mynod_split+1)%array(:) = list of atoms this node has
  integer, pointer, dimension(:) :: grid_atom_p
  integer n_grid_atom

  ! n_atomlist_p => atomlist_th(0)%n_atomlist
  ! n_atomlist_p(0:nrecip)
  ! n_atomlist_p(inode) = number of atoms for node inode
  integer, pointer, dimension(:) :: n_atomlist_p

  ! n_conslist_th(0:nthread-1) = Number of cons atoms each thread found
  integer, allocatable, dimension(:) :: n_conslist_th
  ! conslist_th(0:nthread-1)%array(1:ncons) = Index of cons atoms each thread found
  type(intarray_t), allocatable, dimension(:) :: conslist_th
  ! Total number of cons atoms this node found
  integer n_cons_tot

  ! Location of this reciprocal node in the grid 0...ny_box-1, 0...nz_box-1
  !integer iy_box, iz_box

#if KEY_DOMDEC_GPU==1
  type(xyzq_sp_t), pointer, dimension(:) :: xyzq
#endif

  ! Public subroutines
  public comm_coord_among_recip, comm_force_among_recip, zero_recip_force, &
       reduce_recip_forces, recv_coord_from_direct, send_results_to_direct, &
       uninit_r2d_comm

  ! Public variables
  public n_grid_atom, grid_atom_p, recipforcex, recipforcey, recipforcez


contains

  ! *
  ! * Communicates atom coordinates among reciprocal nodes
  ! *
  subroutine comm_coord_among_recip(natom, x, y, z, cg)
#if KEY_DOMDEC_GPU==1
    use domdec_common,only:q_gpu, gpu_code_version
    use domdec_util_gpu_mod,only:range_start, range_stop, copy_xyzq_to_gpu
#endif
    use domdec_common,only:q_split
    use domdec_dr_common,only:nrecip

#if KEY_BLOCK==1
    use block_ltm, only: iblckp
    use lambdam, only: bixlam
#endif

    implicit none

    ! Input / Output
    integer, intent(in) :: natom
    real(chm_real), intent(inout) :: x(*), y(*), z(*), cg(*)

    if (.not.q_split) then
       if (nrecip > 1) then
          call alloc_realloc_recipforce(natom)
#if KEY_DOMDEC_GPU==1
          if (q_gpu) call range_start('build_atomlist_nosplit')
#endif
          call build_atomlist_nosplit(x, y, z)
#if KEY_DOMDEC_GPU==1
          if (q_gpu) call range_stop()
#endif
          call comm_coord_among_recip_split(natom, x, y, z)
       else
          call filter_grid_atom_single(natom)
       endif
    else
       call comm_coord_among_recip_split(natom, x, y, z)
    endif

#if KEY_DOMDEC_GPU==1
    if (q_gpu .and. gpu_code_version == 2 .and. q_split) then
       call copy_xyzq_to_gpu(natom, x, y, z, cg)
    end if
#endif

    return
  end subroutine comm_coord_among_recip

  ! *
  ! * Filters grid atoms for a single node case
  ! *
  subroutine filter_grid_atom_single(natom)
    use domdec_common,only:atoml
    implicit none
    ! Input
    integer, intent(in) :: natom
    ! Variables
    integer i

    ! Deallocate atomlist_th
    ! NOTE: this also nulls grid_atom_p
    call dealloc_atomlist_th()

    n_grid_atom = natom
    ! Set grid_atom_p to point to atoml
    grid_atom_p => atoml

    n_cons_tot = 0

!!$    ! In the case of single reciprocal node, set all atoms to be grid_atoms
!!$    n_grid_atom = natom
!!$!$omp parallel do
!!$    do i=1,n_grid_atom
!!$       grid_atom_p(i) = i
!!$    enddo
!!$!$omp end parallel do

    return
  end subroutine filter_grid_atom_single

  ! *
  ! * Builds atomlist() -array when direct/reciprocal split is OFF
  ! *
  subroutine build_atomlist_nosplit(x, y, z)
    use colfft_util,only:coord_to_grid
    use domdec_common,only:boxx, boxy, boxz, natoml, atoml, q_use_single, &
         nthread, divide_thread_work, q_test, q_cons, ncons
    use nblist_util,only:concat_buffer, reduce_buffer
    use pmeutil,only:nfft1, nfft2, nfft3, forder
    use inbnd,only:ctofnb
#if KEY_DOMDEC_GPU==1
    use domdec_common,only:q_gpu
    use domdec_util_gpu_mod,only:range_start, range_stop
#endif
    use domdec_dr_common,only:nrecip, mynod_split, ncomm, cons_node_split
    use parallel,only:mynod
    implicit none
    ! Input / Output
    real(chm_real), intent(in) :: x(*), y(*), z(*)
#ifdef _OPENMP
    ! Functions
    integer omp_get_thread_num
#endif
    ! Variables
    logical q_filter_all, q_check_cons
    real(chm_real) forder_x, forder_y, forder_z
    real(chm_real) recip(3,3)
    real(chm_real4) recip_sp(3,3)
    real(chm_real) fr2, fr3
    real(chm_real4) fr2_sp, fr3_sp
    integer ierror
    integer nnode, node(4)
    integer i, j, l
    integer istart, iend
    integer tid

    ncomm = 0

    ! Determines if q_cons are checked or not
    q_check_cons = ((ncons > 0) .and. (mynod_split /= cons_node_split))

!    ! Allocate / Reallocate grid_atom
!    call alloc_realloc_grid_atom(natoml)

    ! As long as the real-space cut-off is smaller than the charge interpolation range,
    ! we can assume that the nodes have all the atoms to the "right" of their homezones.
    !
    ! If this is not the case, we should also communicate extra atoms from the right.

    ! Calculate the extent of the charge spreading "forder". +1 added for safety
    forder_x = real(nfft1*(forder+1))/boxx
    forder_y = real(nfft2*(forder+1))/boxy
    forder_z = real(nfft3*(forder+1))/boxz

    q_filter_all = .true.
    if (forder_x > ctofnb .or. forder_y > ctofnb .or. forder_z > ctofnb) then
       q_filter_all = .true.
    endif

    ! Allocate & re-allocate memory (atomlist_th, n_conslist_th, conslist_th)
    call alloc_realloc(natoml, nthread, nrecip, ncons)

    ! Setup (filter_atom_y, filter_atom_z)
    call setup_filter_atom_yz()

    ! Calculate reciprocal vectors
    call setup_recip(recip)

    ! Go through the atom list and see if any of the atoms is in the neighboring nodes'
    ! charge grid space
    if (q_filter_all) then
       if (q_use_single()) then
          recip_sp = recip
#if KEY_DOMDEC_GPU==1
          if (q_gpu) call range_start('kernel')
#endif
!$omp parallel private(tid, istart, iend)
#ifdef _OPENMP
          tid = omp_get_thread_num()
#else
          tid = 0
#endif
          atomlist_th(tid)%n_atomlist(0:nrecip) = 0
          n_conslist_th(tid) = 0
          call divide_thread_work(natoml, istart, iend)
          call build_atomlist_kernel_sp(x, y, z, istart, iend, atoml, &
               recip_sp, q_check_cons, q_cons, nrecip, &
               atomlist_th(tid)%n_atomlist, atomlist_th(tid)%atomlist, &
               n_conslist_th(tid), conslist_th(tid)%array)
!$omp end parallel
#if KEY_DOMDEC_GPU==1
          if (q_gpu) call range_stop()
#endif
       else

!$omp parallel private(tid, istart, iend)
#ifdef _OPENMP
          tid = omp_get_thread_num()
#else
          tid = 0
#endif
          atomlist_th(tid)%n_atomlist(0:nrecip) = 0
          n_conslist_th(tid) = 0
          call divide_thread_work(natoml, istart, iend)
          call build_atomlist_kernel_dp(x, y, z, istart, iend, atoml, &
               recip, q_check_cons, q_cons, nrecip, &
               atomlist_th(tid)%n_atomlist, atomlist_th(tid)%atomlist, &
               n_conslist_th(tid), conslist_th(tid)%array)
!$omp end parallel
!          call wrndie(-5,'<domdec_r2d_comm>',&
!               'build_atomlist_nosplit: This part not implemented yet!')
       endif
    else
       call wrndie(-5,'<domdec_r2d_comm>','build_atomlist_nosplit: This part not implemented yet!')
    endif

    call reduce_atomlist_th(nthread, nrecip)
    n_grid_atom = n_atomlist_p(mynod_split+1)

    call reduce_conslist_th(nthread)
    n_cons_tot = n_conslist_th(0)

    if (q_test) then
       call test_atomlist(x, y, z, q_use_single())
    endif

    return

  contains
    subroutine alloc_realloc(natoml, nthread, nrecip, ncons)
      use nblist_util,only:init_array
      implicit none
      ! Input
      integer, intent(in) :: natoml, nthread, nrecip, ncons

      ! atomlist_th
      call alloc_realloc_atomlist_th(nthread, nrecip, natoml*4, .false.)

      ! n_conslist_th(0:nthread-1)
      call alloc_realloc_integer_buffer(n_conslist_th, nthread, 0)

      ! conslist_th(0:nthread-1)%array(1:ncons)
      call init_array(conslist_th, nthread, ncons, max(ncons,1))

      return
    end subroutine alloc_realloc
  end subroutine build_atomlist_nosplit

  ! *
  ! * Reduces atomlist_th(0:nthread-1) into atomlist_th(0)
  ! *
  subroutine reduce_atomlist_th(nthread, nrecip)
    implicit none
    ! Input
    integer, intent(in) :: nthread, nrecip
    ! Variables
    integer i, j, pos1, pos2

    do j=1,nrecip
       ! Perform inplace inclusive cumulative sum on atomlist_th(0:nthread-1)%n_atomlist(j)
       do i=1,nthread-1
          atomlist_th(i)%n_atomlist(j) = atomlist_th(i)%n_atomlist(j) + &
               atomlist_th(i-1)%n_atomlist(j)
       enddo
    enddo

    ! Concatenate buffers to atomlist_th(0)%atomlist(1:nrecip)%array()
    do j=1,nrecip
       do i=1,nthread-1
          pos1 = atomlist_th(i-1)%n_atomlist(j)
          pos2 = atomlist_th(i)%n_atomlist(j)
          atomlist_th(0)%atomlist(j)%array(pos1+1:pos2) = &
               atomlist_th(i)%atomlist(j)%array(1:pos2-pos1)
       enddo
    enddo

    ! Set thread 0 to have the total number of atoms
    do j=1,nrecip
       atomlist_th(0)%n_atomlist(j) = atomlist_th(nthread-1)%n_atomlist(j)
    enddo

    return
  end subroutine reduce_atomlist_th

  ! *
  ! * Reduces conslist_th(0:nthread-1) into conslist_th(0)
  ! *
  subroutine reduce_conslist_th(nthread)
    implicit none
    ! Input
    integer, intent(in) :: nthread
    ! Variables
    integer i, pos1, pos2

    ! Perform inplace inclusive cumulative sum
    do i=1,nthread-1
       n_conslist_th(i) = n_conslist_th(i) + n_conslist_th(i-1)
    enddo

    ! Concatenate buffers
    do i=1,nthread-1
       pos1 = n_conslist_th(i-1)
       pos2 = n_conslist_th(i)
       conslist_th(0)%array(pos1+1:pos2) = conslist_th(i)%array(1:pos2-pos1)
    enddo

    ! Set thread 0 to have the total number of atoms
    n_conslist_th(0) = n_conslist_th(nthread-1)

    return
  end subroutine reduce_conslist_th

  ! *
  ! *
  ! *
  subroutine build_atomlist_kernel_sp(x, y, z, istart, iend, atoml, &
       recip, q_check_cons, q_cons, nrecip, n_atomlist, atomlist, &
       n_conslist, conslist)
    use domdec_dr_common,only:cons_node_split
    use colfft_util,only:ny_box
    use pmeutil,only:nfft2,nfft3
    implicit none
    ! Input / Output
    real(chm_real), intent(in) :: x(*), y(*), z(*)
    integer, intent(in) :: istart, iend, atoml(:)
    real(chm_real4), intent(in) :: recip(3,3)
    logical, intent(in) :: q_check_cons
    logical, intent(in) :: q_cons(:)
    integer, intent(in) :: nrecip
    integer, intent(inout) :: n_atomlist(0:nrecip+1)
    type(intarray_t), intent(inout) :: atomlist(0:nrecip+1)
    integer, intent(inout) :: n_conslist
    integer, intent(inout) :: conslist(:)
    ! Parameters
    real(chm_real4), parameter :: two_sp = 2.0_chm_real4, half_sp = 0.5_chm_real4
    ! Variables
    integer i, j
    real(chm_real4) xj, yj, zj, w2, w3
    integer iy, iz
    integer node(4)
    integer valy1, valy2, valz1, valz2
    integer ind_valy2, ind_valz2
    logical q_send_to_cons
    integer l

    if (q_check_cons) then

       do i=istart,iend
          j = atoml(i)
          ! Calculate charge grid coordinates

          xj = x(j)
          yj = y(j)
          zj = z(j)

          w2 = xj*recip(1,2) + yj*recip(2,2) + zj*recip(3,2) + two_sp
          w3 = xj*recip(1,3) + yj*recip(2,3) + zj*recip(3,3) + two_sp

          iy = int(nfft2*(w2 - (anint(w2) - half_sp)))
          iz = int(nfft3*(w3 - (anint(w3) - half_sp)))

          ! Check the recip nodes to which this atom belongs to (max 4)

          valy1 = filter_atom_y(1,iy)   ! Must be > 0
          valy2 = filter_atom_y(2,iy)   ! Can be = 0
          valz1 = filter_atom_z(1,iz)   ! Must be > 0
          valz2 = filter_atom_z(2,iz)   ! Can be = 0

          ! Returns:
          ! 0 if valy2 == 0
          ! 1 if valy2 > 0
          ind_valy2 = xor(ishft(valy2-1,-31),1)
          ind_valz2 = xor(ishft(valz2-1,-31),1)

          valz1 = valz1 - ny_box
          valz2 = valz2 - ny_box

          node(1) = valy1 + valz1
          node(2) = (valy2 + valz1)*ind_valy2
          node(3) = (valy1 + valz2)*ind_valz2
          node(4) = (valy2 + valz2)*ind_valy2*ind_valz2

          ! Put the atom in the correct recip node
          q_send_to_cons = q_cons(j)

          do l=1,4
             n_atomlist(node(l)) = n_atomlist(node(l)) + 1
             n_atomlist(0) = 1
             atomlist(node(l))%array(n_atomlist(node(l))) = j
             q_send_to_cons = q_send_to_cons .and. (node(l) /= cons_node_split+1)
          enddo

          if (q_send_to_cons) then
             ! This atom is communicated strictly as a "cons atom"
             n_conslist = n_conslist + 1
             conslist(n_conslist) = j
          endif

       enddo
    else

       do i=istart,iend
          j = atoml(i)
          ! Calculate charge grid coordinates

          xj = x(j)
          yj = y(j)
          zj = z(j)

          w2 = xj*recip(1,2) + yj*recip(2,2) + zj*recip(3,2) + two_sp
          w3 = xj*recip(1,3) + yj*recip(2,3) + zj*recip(3,3) + two_sp

          iy = int(nfft2*(w2 - (anint(w2) - half_sp)))
          iz = int(nfft3*(w3 - (anint(w3) - half_sp)))

          ! Check the recip nodes to which this atom belongs to (max 4)

          valy1 = filter_atom_y(1,iy)   ! Must be > 0
          valy2 = filter_atom_y(2,iy)   ! Can be = 0
          valz1 = filter_atom_z(1,iz)   ! Must be > 0
          valz2 = filter_atom_z(2,iz)   ! Can be = 0

          ! Returns:
          ! 0 if valy2 == 0
          ! 1 if valy2 > 0
          ind_valy2 = xor(ishft(valy2-1,-31),1)
          ind_valz2 = xor(ishft(valz2-1,-31),1)

          valz1 = valz1 - ny_box
          valz2 = valz2 - ny_box

          node(1) = valy1 + valz1
          node(2) = (valy2 + valz1)*ind_valy2
          node(3) = (valy1 + valz2)*ind_valz2
          node(4) = (valy2 + valz2)*ind_valy2*ind_valz2

          do l=1,4
             n_atomlist(node(l)) = n_atomlist(node(l)) + 1
             n_atomlist(0) = 1
             atomlist(node(l))%array(n_atomlist(node(l))) = j
          enddo

       enddo

    endif

    return
  end subroutine build_atomlist_kernel_sp

!!$  ! *
!!$  ! *
!!$  ! *
!!$  subroutine OLDbuild_atomlist_kernel_sp(x, y, z, istart, iend, atoml, &
!!$       recip, q_check_cons, q_cons, nrecip, n_atomlist, atomlist)
!!$    use colfft_util,only:coord_to_grid, ny_box
!!$    !use colfft_util,only:ny_box
!!$    implicit none
!!$    ! Input / Output
!!$    real(chm_real), intent(in) :: x(*), y(*), z(*)
!!$    integer, intent(in) :: istart, iend, atoml(:)
!!$    real(chm_real4), intent(in) :: recip(3,3)
!!$    logical, intent(in) :: q_check_cons, q_cons(:)
!!$    integer, intent(in) :: nrecip
!!$    integer, intent(inout) :: n_atomlist(0:nrecip+1)
!!$    type(intarray_t), intent(inout) :: atomlist(0:nrecip+1)
!!$    ! Variables
!!$    real(chm_real4) fr2, fr3
!!$    integer node(4)
!!$    integer i, j
!!$
!!$    do i=istart,iend
!!$       j = atoml(i)
!!$       ! Calculate charge grid coordinates
!!$       call coord_to_grid(real(x(j),kind=chm_real4), real(y(j),kind=chm_real4), &
!!$            real(z(j),kind=chm_real4), recip, fr2, fr3)
!!$       ! Check the recip nodes to which this atom belongs to (max 4)
!!$       call get_recip_node_ind(int(fr2), int(fr3), ny_box, node)
!!$       ! Put the atom in the correct recip node
!!$       call decide_recip_node(j, node, q_check_cons, q_cons, nrecip, n_atomlist, atomlist)
!!$    enddo
!!$
!!$    return
!!$  end subroutine OLDbuild_atomlist_kernel_sp

  subroutine build_atomlist_kernel_dp(x, y, z, istart, iend, atoml, &
       recip, q_check_cons, q_cons, nrecip, n_atomlist, atomlist, &
       n_conslist, conslist)
    use colfft_util,only:coord_to_grid, ny_box
!!$    use pmeutil,only:nfft2, nfft3
!!$    use number,only:half, two
    implicit none
    ! Input / Output
    real(chm_real), intent(in) :: x(*), y(*), z(*)
    integer, intent(in) :: istart, iend, atoml(:)
    real(chm_real), intent(in) :: recip(3,3)
    logical, intent(in) :: q_check_cons
    logical, intent(in) :: q_cons(:)
    integer, intent(in) :: nrecip
    integer, intent(inout) :: n_atomlist(0:nrecip+1)
    type(intarray_t), intent(inout) :: atomlist(0:nrecip+1)
    integer, intent(inout) :: n_conslist
    integer, intent(inout) :: conslist(:)
    ! Variables
    real(chm_real) fr2, fr3 !, w2, w3
    integer node(4)
    integer i, j

    do i=istart,iend
       j = atoml(i)
       ! Calculate charge grid coordinates
       call coord_to_grid(x(j), y(j), z(j), recip, fr2, fr3)
!!$       w2 = x(j)*recip(1,2) + y(j)*recip(2,2) + z(j)*recip(3,2) + two
!!$       w3 = x(j)*recip(1,3) + y(j)*recip(2,3) + z(j)*recip(3,3) + two
!!$       fr2 = nfft2*(w2 - (anint(w2) - half))
!!$       fr3 = nfft3*(w3 - (anint(w3) - half))
       ! Check the recip nodes to which this atom belongs to (max 4)
       call get_recip_node_ind(int(fr2), int(fr3), ny_box, node)
       ! Put the atom in the correct recip node
       call decide_recip_node(j, node, q_check_cons, q_cons, nrecip, n_atomlist, atomlist, &
            n_conslist, conslist)
    enddo

    return
  end subroutine build_atomlist_kernel_dp

  ! *
  ! * Tests atomlist_p(1:nrecip)%array(1:n_atomlist_p)
  ! *
  subroutine test_atomlist(x, y, z, q_single)
    use stream,only:outu
    use colfft_util,only:YZ_X_PARTITION, get_spatial_limits, filter_atom, ny_box, nz_box, &
         coord_to_grid
    use pmeutil,only:nfft2, nfft3, forder
    use domdec_dr_common,only:nrecip, mynod_split
    use number,only:two
    implicit none
    ! Input
    real(chm_real), intent(in) :: x(*), y(*), z(*)
    logical, intent(in) :: q_single
    ! Variables
    real(chm_real) recip(3,3), fr2, fr3
    real(chm_real4) recip_sp(3,3), fr2_sp, fr3_sp
    integer xgridmin, xgridmax, ygridmin, ygridmax, zgridmin, zgridmax
    integer ydim, zdim
    integer iy, iz
    integer k, i, j

    if (n_grid_atom /= n_atomlist_p(mynod_split+1)) then
       call wrndie(-5,'<domdec_r2d_comm>','test_atomlist: n_grid_atom setup incorrectly (1)')
    endif

    ! Calculate reciprocal vectors
    call setup_recip(recip)
    recip_sp = recip

    do k=1,nrecip
       call get_spatial_limits(YZ_X_PARTITION,xgridmin,xgridmax, &
            ygridmin,ygridmax,zgridmin,zgridmax,k-1)
       if (ny_box == 1 .or. ygridmin >= forder-1) then
          ydim = nfft2*4
       else
          ydim = nfft2
       endif
       if (nz_box == 1 .or. zgridmin >= forder-1) then
          zdim = nfft3*4
       else
          zdim = nfft3
       endif
       do i=1,n_atomlist_p(k)
          j = atomlist_p(k)%array(i)
          if (q_single) then
             call coord_to_grid(real(x(j),kind=chm_real4), real(y(j),kind=chm_real4), &
                  real(z(j),kind=chm_real4), recip_sp, fr2_sp, fr3_sp)
             iy = int(fr2_sp)
             iz = int(fr3_sp)
          else
             call coord_to_grid(x(j), y(j), z(j), recip, fr2, fr3)
             iy = int(fr2)
             iz = int(fr3)
          endif
          if (.not.filter_atom(iy, forder, ygridmin, ygridmax, ydim) .or. &
               .not.filter_atom(iz, forder, zgridmin, zgridmax, zdim)) then
             write (outu,'(a,i3,2i6,2i4)') 'k,i,j,iy,iz=',k,i,j,iy,iz
             write (outu,'(a,2i4)') 'ygridmin, ygridmax=',ygridmin, ygridmax
             write (outu,'(a,2i4)') 'zgridmin, zgridmax=',zgridmin, zgridmax
             write (outu,'(a,2f12.6)') 'y(j),z(j)=',y(j),z(j)
             call wrndie(-5,'<domdec_r2d_comm>','test_atomlist: atomlist setup incorrectly (2)')
          endif
       enddo
    enddo

    if (mynod_split == 0) write (outu,'(a)') 'test_atomlist OK'

    return
  end subroutine test_atomlist

  ! *
  ! * Writes all recip nodes into ind(1:4) to which the atom with grid coordinates (iy, iz)
  ! * belongs to.
  ! * NOTE: Uses (filter_atom_y, filter_atom_z)
  ! *
  subroutine get_recip_node_ind(iy, iz, ny_box, ind)
    implicit none
    ! Input / Output
    integer, intent(in) :: iy, iz, ny_box
    integer, intent(out) :: ind(4)
    ! Variables
    integer valy1, valy2, valz1, valz2
    integer ind_valy2, ind_valz2

    valy1 = filter_atom_y(1,iy)   ! Must be > 0
    valy2 = filter_atom_y(2,iy)   ! Can be = 0
    valz1 = filter_atom_z(1,iz)   ! Must be > 0
    valz2 = filter_atom_z(2,iz)   ! Can be = 0

    ! Returns:
    ! 0 if valy2 == 0
    ! 1 if valy2 > 0
    ind_valy2 = xor(ishft(valy2-1,-31),1)
    ind_valz2 = xor(ishft(valz2-1,-31),1)

!    ind_valy2 = (valy2 /= 0)       ! = 1 if (val2y /= 0)
!    ind_valz2 = (valz2 /= 0)       ! = 1 if (val2z /= 0)

    valz1 = valz1 - ny_box
    valz2 = valz2 - ny_box

    ind(1) = valy1 + valz1
    ind(2) = (valy2 + valz1)*ind_valy2
    ind(3) = (valy1 + valz2)*ind_valz2
    ind(4) = (valy2 + valz2)*ind_valy2*ind_valz2

    return
  end subroutine get_recip_node_ind

  ! *
  ! * Decides the recip node atom i belongs to.
  ! *
  subroutine decide_recip_node(i, node, q_check_cons, q_cons, nrecip, n_atomlist, atomlist, &
       n_conslist, conslist)
    use domdec_dr_common,only:mynod_split, cons_node_split
    implicit none
    ! Input
    integer, intent(in) :: i, node(4)
    logical, intent(in) :: q_check_cons
    logical, intent(in) :: q_cons(:)
    integer, intent(in) :: nrecip
    integer, intent(inout) :: n_atomlist(0:nrecip)
    type(intarray_t), intent(inout) :: atomlist(0:nrecip)
    integer, intent(inout) :: n_conslist
    integer, intent(inout) :: conslist(:)
    ! Variables
    logical q_send_to_cons
    integer l

    if (q_check_cons) then
       q_send_to_cons = q_cons(i)
    else
       q_send_to_cons = .false.
    endif

    do l=1,4
       n_atomlist(node(l)) = n_atomlist(node(l)) + 1
       n_atomlist(0) = 1
       atomlist(node(l))%array(n_atomlist(node(l))) = i
       q_send_to_cons = q_send_to_cons .and. (node(l) /= cons_node_split+1)
    enddo

    if (q_send_to_cons) then
       ! This atom is communicated strictly as a "cons atom"
       n_conslist = n_conslist + 1
       conslist(n_conslist) = i
    endif

    return
  end subroutine decide_recip_node

  ! *
  ! * Communicates coordinates among reciprocal nodes
  ! *
  subroutine comm_coord_among_recip_split(natom, x, y, z)
    use pack_mod,only:pack_int_byte, pack_double_byte, unpack_int_byte, unpack_double_byte
    use memory
    use stream,only:outu, prnlev
    use parallel,only:mpi_integer_size, mpi_real8_size
    use mpi,only:mpi_integer, mpi_byte, mpi_success, mpi_real8
    use new_timer,only:timer_start, timer_stop, T_r2r
    use domdec_common,only:nthread, q_test
    use domdec_dr_common,only:comm_recip, nrecip, commbuffersize, commbufferpos, commbuffer, &
         mynod_split, ncomm, alloc_realloc_commbuffer, cons_node_split
#if KEY_DOMDEC_GPU==1
    use domdec_common,only:q_gpu
    use domdec_util_gpu_mod,only:range_start, range_stop
#endif
    implicit none
    ! Input / Output
    integer, intent(in) :: natom
    real(chm_real), intent(inout) :: x(*), y(*), z(*)
    ! Variables
    real(chm_real) xyz(3)
    integer ierror
    integer coord_size, nrecvtot
    integer commbuffersizetot, commbuffer2sizetot
    integer i, j, k, jabs, n_cons_atom

    call timer_start(T_r2r)

    coord_size = mpi_integer_size + 3*mpi_real8_size

    ! Allocate & reallocate memory (commbufferpos, commbuffersize, commbuffer2pos, commbuffer2size,
    ! nrecv, grid_atom_pos, nsend, cons_atom_pos)
    call alloc_realloc(natom, ncomm)

    ! Make nsend(1:nrecip)
    ! n_atom_list_p(i-i) = number of atoms to be sent to recip node i
    ! NOTE: self node is set to zero
    nsend(1:nrecip) = n_atomlist_p(1:nrecip)
    nsend(cons_node_split+1) = nsend(cons_node_split+1) + n_cons_tot
    nsend(mynod_split+1) = 0

    ! Do all-to-all within recip nodes
    ! Send the number of atoms to all recip nodes
    call mpi_alltoall(nsend, 1, mpi_integer, nrecv, 1, mpi_integer, &
         comm_recip, ierror)
    if (ierror /= mpi_success) then
       call wrndie(-5,'<domdec_r2d_comm>','mpi_alltoall failed')
    endif

#if KEY_DOMDEC_GPU==1
     if (q_gpu) call range_stop()
#endif

    nrecvtot = sum(nrecv(1:nrecip))
    ! nrecvtot = total number of atoms this recip node will receive
    ! nrecv(1:nrecip) = number of atoms that this node is going to receive from each recip node

    ! Calculate displacements
    commbuffersize(1:nrecip) = nsend(1:nrecip)*coord_size
    commbuffer2size(1:nrecip) = nrecv(1:nrecip)*coord_size
    commbufferpos(1) = 0
    commbuffer2pos(1) = 0
    do i=2,nrecip
       commbufferpos(i) = commbufferpos(i-1) + commbuffersize(i-1)
       commbuffer2pos(i) = commbuffer2pos(i-1) + commbuffer2size(i-1)
    enddo

    ! Allocate / Reallocate commbuffer as needed
    commbuffersizetot = sum(commbuffersize(1:nrecip))
    call alloc_realloc_commbuffer(commbuffer, commbuffersizetot)

#if KEY_DOMDEC_GPU==1
     if (q_gpu) call range_start('pack send data')
#endif

    ! Pack in send data to commbuffer
    commbuffersize(1:nrecip) = 0
    do k=1,nrecip
       if (k /= mynod_split+1) then
          do i=1,n_atomlist_p(k)
             j = atomlist_p(k)%array(i)
             xyz(1:3) = (/ x(j), y(j), z(j) /)
             call pack_int_byte(j, 1, commbuffer(commbufferpos(k)+1), commbuffersize(k))
!!$             call mpi_pack(j, 1, mpi_integer, commbuffer(commbufferpos(k)+1), &
!!$                  size(commbuffer), commbuffersize(k), comm_recip, ierror)
!!$             if (ierror /= mpi_success) call wrndie(-5, &
!!$                  '<domdec_r2d_comm>','Error in mpi_pack in comm_coord_among_recip_split')
             call pack_double_byte(xyz(1), 3, commbuffer(commbufferpos(k)+1), commbuffersize(k))
!!$             call mpi_pack(xyz, 3, mpi_real8, commbuffer(commbufferpos(k)+1), &
!!$                  size(commbuffer), commbuffersize(k), comm_recip, ierror)
!!$             if (ierror /= mpi_success) call wrndie(-5, &
!!$                  '<domdec_r2d_comm>','Error in mpi_pack in comm_coord_among_recip_split')
          enddo
          if (k == cons_node_split+1) then
             do i=1,n_cons_tot
                j = conslist_th(0)%array(i)
                xyz(1:3) = (/ x(j), y(j), z(j) /)
                call pack_int_byte(-j, 1, commbuffer(commbufferpos(k)+1), commbuffersize(k))
!!$                call mpi_pack(-j, 1, mpi_integer, commbuffer(commbufferpos(k)+1), &
!!$                     size(commbuffer), commbuffersize(k), comm_recip, ierror)
!!$                if (ierror /= mpi_success) call wrndie(-5, &
!!$                     '<domdec_r2d_comm>','Error in mpi_pack in comm_coord_among_recip_split')
                call pack_double_byte(xyz(1), 3, commbuffer(commbufferpos(k)+1), commbuffersize(k))
!!$                call mpi_pack(xyz, 3, mpi_real8, commbuffer(commbufferpos(k)+1), &
!!$                     size(commbuffer), commbuffersize(k), comm_recip, ierror)
!!$                if (ierror /= mpi_success) call wrndie(-5, &
!!$                     '<domdec_r2d_comm>','Error in mpi_pack in comm_coord_among_recip_split')
             enddo
          endif
       endif
    enddo

#if KEY_DOMDEC_GPU==1
     if (q_gpu) call range_stop()
#endif

    ! Check commbuffersize
    do i=1,nrecip
       if (commbuffersize(i) /= nsend(i)*coord_size) then
          write (outu,'(a,2i8)') 'commbuffersize(i), nsend(i)*coord_size=',&
               commbuffersize(i),nsend(i)*coord_size
          call wrndie(-5,'<domdec_r2d_comm>','Invalid commbuffersize (1)')
       endif
    enddo
    if (sum(commbuffersize(1:nrecip)) > size(commbuffer)) then
       call wrndie(-5,'<domdec_r2d_comm>','commbuffer exceeded')
    endif

    ! Allocate / Reallocate commbuffer2 as needed
    call alloc_realloc_commbuffer(commbuffer2, nrecvtot*coord_size)

#if KEY_DOMDEC_GPU==1
     if (q_gpu) call range_start('mpi_alltoallv')
#endif

    ! Perform all-to-all communication for coordinates
    call mpi_alltoallv(commbuffer, commbuffersize, commbufferpos, mpi_byte, &
         commbuffer2, commbuffer2size, commbuffer2pos, mpi_byte, comm_recip, ierror)
    if (ierror /= mpi_success) then
       call wrndie(-5,'<domdec_r2d_comm>','mpi_alltoallv failed')
    endif

#if KEY_DOMDEC_GPU==1
     if (q_gpu) call range_stop()
#endif

    commbuffer2sizetot = sum(commbuffer2size(1:nrecip))

    ! Check commbuffer2size
    do i=1,nrecip
       if (commbuffer2size(i) /= nrecv(i)*coord_size) then
          call wrndie(-5,'<domdec_r2d_comm>','Invalid commbuffer2size')
       endif
    enddo
    if (size(commbuffer2) < commbuffer2sizetot) then
       write (outu,'(a,2i8)') 'size(commbuffer2),commbuffer2sizetot=',&
            size(commbuffer2),commbuffer2sizetot
       write (outu,'(a,i8,i4)') 'nrecvtot,coord_size=',nrecvtot,coord_size
       write (outu,'(a,20i6)') 'nrecv=',nrecv(1:nrecip)
       write (outu,'(a,20i6)') 'commbuffer2size=',commbuffer2size(1:nrecip)
       call wrndie(-5,'<domdec_r2d_comm>','commbuffer2 exceeded')
    endif

    ! Re-size grid_atom as needed
    ! NOTE: If realloc is needed, we need to copy the original atomlist
    !       over to the new one at this point
    call alloc_realloc_atomlist_th(nthread, nrecip, n_grid_atom + nrecvtot, .true.)

#if KEY_DOMDEC_GPU==1
    if (q_gpu) call range_start('unpack coord')
#endif

    ! Unpack coordinates
    ! For each recip node, save n_grid_atom starting positions to grid_atom_pos
    n_cons_atom = 0
    do i=1,nrecip
       k = 0
       grid_atom_pos(i) = n_grid_atom
       cons_atom_pos(i) = n_cons_atom
       do while (k < commbuffer2size(i))
          call unpack_int_byte(commbuffer2(commbuffer2pos(i)+1), k, j, 1)
!!$          call mpi_unpack(commbuffer2(commbuffer2pos(i)+1), size(commbuffer2), k, &
!!$               j, 1, mpi_integer, comm_recip, ierror)
!!$          if (ierror /= mpi_success) call wrndie(-5, &
!!$               '<domdec_r2d_comm>','Error in mpi_unpack in comm_coord_among_recip_split')
          jabs = iabs(j)
          call unpack_double_byte(commbuffer2(commbuffer2pos(i)+1), k, xyz(1), 3)
!!$          call mpi_unpack(commbuffer2(commbuffer2pos(i)+1), size(commbuffer2), k, &
!!$               xyz, 3, mpi_real8, comm_recip, ierror)
!!$          if (ierror /= mpi_success) call wrndie(-5, &
!!$               '<domdec_r2d_comm>','Error in mpi_unpack in comm_coord_among_recip_split')
          x(jabs) = xyz(1)
          y(jabs) = xyz(2)
          z(jabs) = xyz(3)
          if (j > 0) then
             n_grid_atom = n_grid_atom + 1
             grid_atom_p(n_grid_atom) = jabs
          else
             ! NOTE: Only cons node can go here
             n_cons_atom = n_cons_atom + 1
             conslist_th(0)%array(n_cons_atom) = jabs
          endif
       enddo
    enddo
    grid_atom_pos(nrecip+1) = n_grid_atom
    cons_atom_pos(nrecip+1) = n_cons_atom
    n_atomlist_p(mynod_split+1) = n_grid_atom

#if KEY_DOMDEC_GPU==1
     if (q_gpu) call range_stop()
#endif

    if (q_test) then
       call test_recip_coord(natom, x, y, z)
    endif

    call timer_stop(T_r2r)

    return

  contains
    subroutine alloc_realloc(natom, ncomm)
      use memory
      implicit none
      ! Input
      integer, intent(in) :: natom, ncomm

      ! commbufferpos
      call alloc_realloc_integer_buffer(commbufferpos, max(nrecip,ncomm))

      ! commbuffersize
      call alloc_realloc_integer_buffer(commbuffersize, max(nrecip,ncomm))

      ! commbuffer2pos
      call alloc_realloc_integer_buffer(commbuffer2pos, max(nrecip,ncomm))

      ! commbuffer2size
      call alloc_realloc_integer_buffer(commbuffer2size, max(nrecip,ncomm))

      ! nrecv
      call alloc_realloc_integer_buffer(nrecv, nrecip)

      ! nsend
      call alloc_realloc_integer_buffer(nsend, nrecip)

      ! grid_atom_pos
      call alloc_realloc_integer_buffer(grid_atom_pos, nrecip+1)

      ! cons_atom_pos
      call alloc_realloc_integer_buffer(cons_atom_pos, nrecip+1)

      return
    end subroutine alloc_realloc
  end subroutine comm_coord_among_recip_split

  ! *
  ! * Checks that the recip node the correct atom coordinates
  ! *
  subroutine test_recip_coord(natom, x, y, z)
    use domdec_common,only:q_single
    use domdec_dr_common,only:mynod_split
    use stream,only:outu
    use memory
    use pmeutil,only:nfft2, nfft3, forder
    use colfft_util,only:get_spatial_limits, YZ_X_PARTITION, ny_box, nz_box, filter_atom, &
         coord_to_grid
    implicit none
    ! Input / Output
    integer, intent(in) :: natom
    real(chm_real), intent(in) :: x(*), y(*), z(*)
    ! Variables
    logical, allocatable, dimension(:) :: has_atom
    real(chm_real) recip(3,3), fr2, fr3
    real(chm_real4) recip_sp(3,3), fr2_sp, fr3_sp
    integer xgridmin, xgridmax, ygridmin, ygridmax, zgridmin, zgridmax
    integer ydim, zdim
    integer iy, iz
    integer i, j


    ! Calculate reciprocal vectors
    call setup_recip(recip)
    recip_sp = recip


    call chmalloc('domdec_r2d_comm.src','test_recip_coord','has_atom',&
         natom,log=has_atom)
    has_atom(1:natom) = .false.

    call get_spatial_limits(YZ_X_PARTITION,xgridmin,xgridmax,ygridmin,ygridmax,&
         zgridmin,zgridmax,mynod_split)

    if (ny_box == 1 .or. ygridmin >= forder-1) then
       ydim = nfft2*4
    else
       ydim = nfft2
    endif

    if (nz_box == 1 .or. zgridmin >= forder-1) then
       zdim = nfft3*4
    else
       zdim = nfft3
    endif

    do i=1,n_grid_atom
       j = grid_atom_p(i)
       if (j < 1 .or. j > natom) then
          write (outu,'(a,2i12)') 'i,j=',i,j
          call wrndie(-5,'<domdec_r2d_comm>','test_recip_coord: grid_atom setup incorrectly')
       endif
       has_atom(j) = .true.
       if (q_single) then
          call coord_to_grid(real(x(j),kind=chm_real4), real(y(j),kind=chm_real4), &
               real(z(j),kind=chm_real4), recip_sp, fr2_sp, fr3_sp)
          iy = int(fr2_sp)
          iz = int(fr3_sp)
       else
          call coord_to_grid(x(j), y(j), z(j), recip, fr2, fr3)
          iy = int(fr2)
          iz = int(fr3)
       endif
       if (.not.filter_atom(iy, forder, ygridmin, ygridmax, ydim) .or. &
            .not.filter_atom(iz, forder, zgridmin, zgridmax, zdim)) then
          write (outu,'(a,i2,4i6)') 'mynod_split,i,j,iy,iz=',mynod_split,i,j,iy,iz
          call wrndie(-5,'<domdec_r2d_comm>','test_recip_coord: atom in incorrect recip node')
       endif
    enddo

    call chmdealloc('domdec_r2d_comm.src','test_recip_coord','has_atom',&
         natom,log=has_atom)

    if (mynod_split == 0) write (outu,'(a)') 'test_recip_coord OK'

    return
  end subroutine test_recip_coord


  ! *
  ! * Setups (filter_atom_y, filter_atom_z)
  ! *
  subroutine setup_filter_atom_yz()
    use pmeutil,only:nfft2,nfft3,forder
    use colfft_util,only:ny_box,nz_box
    use memory
    implicit none
    ! Variables
    logical q_redo_filter_atom_y, q_redo_filter_atom_z
    integer, allocatable, dimension(:) :: ygridmin, ygridmax, zgridmin, zgridmax
    integer, allocatable, dimension(:) :: ydim, zdim
    integer, save :: ny_box_save = 0, nz_box_save = 0, forder_save = 0
    integer i

    q_redo_filter_atom_y = (forder /= forder_save)

    if (allocated(filter_atom_y)) then
       if (size(filter_atom_y,2) /= nfft2) then
          call chmdealloc('domdec_r2d_comm.src','setup_filter_atom_yz','filter_atom_y',&
               2,size(filter_atom_y),intg=filter_atom_y)
       endif
    endif

    if (.not.allocated(filter_atom_y)) then
       call chmalloc('domdec_r2d_comm.src','setup_filter_atom_yz','filter_atom_y',&
            2,nfft2,lbou2=0,intg=filter_atom_y)
       q_redo_filter_atom_y = .true.
    endif

    if (ny_box_save /= ny_box) then
       q_redo_filter_atom_y = .true.
    endif

    if (q_redo_filter_atom_y) then
       !-------------------------------------------------------------------------
       call chmalloc('domdec_r2d_comm.src','setup_filter_atom_yz','ygridmin',&
            ny_box,intg=ygridmin)
       call chmalloc('domdec_r2d_comm.src','setup_filter_atom_yz','ygridmax',&
            ny_box,intg=ygridmax)
       call chmalloc('domdec_r2d_comm.src','setup_filter_atom_yz','ydim',&
            ny_box,intg=ydim)
       !-------------------------------------------------------------------------
       call setup_gridminmax(forder, ny_box, 0, nfft2, ygridmin, ygridmax, ydim)
       call prepare_filter_atom_yz(ny_box, nfft2, ygridmin, ygridmax, ydim, filter_atom_y)
       ny_box_save = ny_box
       !-------------------------------------------------------------------------
       call chmdealloc('domdec_r2d_comm.src','setup_filter_atom_yz','ygridmin',&
            size(ygridmin),intg=ygridmin)
       call chmdealloc('domdec_r2d_comm.src','setup_filter_atom_yz','ygridmax',&
            size(ygridmax),intg=ygridmax)
       call chmdealloc('domdec_r2d_comm.src','setup_filter_atom_yz','ydim',&
            size(ydim),intg=ydim)
       !-------------------------------------------------------------------------
    endif

    ! ---------------------------------------------------------------------------------

    q_redo_filter_atom_z = (forder /= forder_save)

    if (allocated(filter_atom_z)) then
       if (size(filter_atom_z,2) /= nfft3) then
          call chmdealloc('domdec_r2d_comm.src','setup_filter_atom_yz','filter_atom_z',&
               2,size(filter_atom_z),intg=filter_atom_z)
       endif
    endif

    if (.not.allocated(filter_atom_z)) then
       call chmalloc('domdec_r2d_comm.src','setup_filter_atom_yz','filter_atom_z',&
            2,nfft3,lbou2=0,intg=filter_atom_z)
       q_redo_filter_atom_z = .true.
    endif

    if (nz_box_save /= nz_box) then
       q_redo_filter_atom_z = .true.
    endif

    if (q_redo_filter_atom_z) then
       !-------------------------------------------------------------------------
       call chmalloc('domdec_r2d_comm.src','setup_filter_atom_yz','zgridmin',&
            nz_box,intg=zgridmin)
       call chmalloc('domdec_r2d_comm.src','setup_filter_atom_yz','zgridmax',&
            nz_box,intg=zgridmax)
       call chmalloc('domdec_r2d_comm.src','setup_filter_atom_yz','zdim',&
            nz_box,intg=zdim)
       !-------------------------------------------------------------------------
       call setup_gridminmax(forder, nz_box, ny_box, nfft3, zgridmin, zgridmax, zdim)
       call prepare_filter_atom_yz(nz_box, nfft3, zgridmin, zgridmax, zdim, filter_atom_z)
       ! post-process filter_atom_z
       do i=0,nfft3-1
          filter_atom_z(1,i) = (filter_atom_z(1,i))*ny_box
          if (filter_atom_z(2,i) > 0) then
             filter_atom_z(2,i) = (filter_atom_z(2,i))*ny_box
          endif
       enddo
       nz_box_save = nz_box
       !-------------------------------------------------------------------------
       call chmdealloc('domdec_r2d_comm.src','setup_filter_atom_yz','zgridmin',&
            size(zgridmin),intg=zgridmin)
       call chmdealloc('domdec_r2d_comm.src','setup_filter_atom_yz','zgridmax',&
            size(zgridmax),intg=zgridmax)
       call chmdealloc('domdec_r2d_comm.src','setup_filter_atom_yz','zdim',&
            size(zdim),intg=zdim)
       !-------------------------------------------------------------------------
    endif

    forder_save = forder

    return
  end subroutine setup_filter_atom_yz

  ! *
  ! * Sets up (ygridmin, ygridmax, ydim, zgridmin, zgridmax, zdim) arrays
  ! *
  subroutine setup_gridminmax(forder, nyz_box, ny_box, nfftyz, yzgridmin, yzgridmax, yzdim)
    use colfft_util,only:YZ_X_PARTITION, get_spatial_limits
    implicit none
    ! Input / Output
    integer, intent(in) :: forder, nyz_box, ny_box, nfftyz
    integer, intent(inout) :: yzgridmin(:), yzgridmax(:), yzdim(:)
    ! Variables
    integer xgridmin_dummy, xgridmax_dummy
    integer ygridmin_dummy, ygridmax_dummy
    integer zgridmin_dummy, zgridmax_dummy
    integer i

    ! Get grid limits in y/z-direction
    if (ny_box == 0) then
       do i=1,nyz_box
          call get_spatial_limits(YZ_X_PARTITION,xgridmin_dummy,xgridmax_dummy, &
               yzgridmin(i),yzgridmax(i),zgridmin_dummy,zgridmax_dummy,i-1)
       enddo
    else
       do i=1,nyz_box
          call get_spatial_limits(YZ_X_PARTITION,xgridmin_dummy,xgridmax_dummy, &
               ygridmin_dummy,ygridmax_dummy,yzgridmin(i),yzgridmax(i),(i-1)*ny_box)
       enddo
    endif

    ! Set the location of this reciprocal node
    !iy_box = mod(mynod_split, ny_box)
    !iz_box = mynod_split/ny_box

    ! Set ydim and zdim for each grid division
    ! NOTE: *4 makes sure no periodic boundary is used
    !
    ! Logic:
    ! 1. If only a single box => no periodic boundary condition
    ! 2. If grid minimum does not overlap with spline
    !    (when more than single box, for inner boxes)
    !    => no periodic boundary condition
    ! 3. Otherwise => use periodic boundary condition
    !
    do i=1,nyz_box
       if (nyz_box == 1 .or. yzgridmin(i) >= forder-1) then
          yzdim(i) = nfftyz*4
       else
          yzdim(i) = nfftyz
       endif
    enddo

    return
  end subroutine setup_gridminmax

  ! *
  ! * Prepares (filter_atom_y, filter_atom_z)
  ! * Note: Sizes are filter_atom_y(1:2,0:nfft2-1) and filter_atom_z(1:2,0:nfft3-1)
  ! *
  subroutine prepare_filter_atom_yz(nyz_box, nfftyz, yzgridmin, yzgridmax, yzdim, filter_atom_yz)
    use colfft_util,only:filter_atom
    use pmeutil,only:forder
    implicit none
    ! Input / Output
    integer, intent(in) :: nyz_box, nfftyz, yzgridmin(:), yzgridmax(:), yzdim(:)
    integer, intent(out) :: filter_atom_yz(1:2,0:nfftyz-1)
    ! Variables
    integer iyz, j, n

    do iyz=0,nfftyz-1
       filter_atom_yz(1:2, iyz) = 0
       n = 0
       do j=1,nyz_box
          if (filter_atom(iyz, forder, yzgridmin(j), yzgridmax(j), yzdim(j))) then

             n = n + 1
             filter_atom_yz(n, iyz) = j
          endif
       enddo
       if (n == 0 .or. n > 2) then
          write(6,'(a,/,a,i4,/,a,2i6,/,a,i6)') &
               "********** prepare_filter_atom_yz() problem", &
               " n=",n,"   nfftyz,iyz=",nfftyz,iyz, &
               "   nyz_box=",nyz_box

          call wrndie(-5,'<domdec_r2d_comm>',&
               'prepare_filter_atom_yz: Invalid number of nodes found')
       endif
    enddo

    return
  end subroutine prepare_filter_atom_yz

!!$  ! *
!!$  ! * Converts grid indices (iy, iz) into reciprocal node index ind
!!$  ! * NOTE: iy = 0...nfft2-1
!!$  ! *       iz = 0...nfft3-1
!!$  ! * NOTE: Only atoms that are to the left of the destination are sent
!!$  ! *       <=> only check recip nodes that are to the right of this node
!!$  ! *
!!$  subroutine get_recip_node_ind_left(iy, iz, ydim, zdim, ind, nind)
!!$    use pmeutil,only:nfft2,nfft3,forder
!!$    use colfft_util,only:ny_box,nz_box,filter_atom
!!$    use number
!!$    implicit none
!!$    ! Input / Output
!!$    integer, intent(in) :: iy, iz, ydim(*), zdim(*)
!!$    integer, intent(out) :: ind(4), nind
!!$    ! Variables
!!$    integer j, k, nindz, nindy, indz(2), indy(2)
!!$
!!$    nindy = 0
!!$    do k=0,ny_box-1
!!$       j = mod(k + iy_box, ny_box) + 1
!!$       if (filter_atom(iy, forder, ygridmin(j), ygridmax(j), ydim(j))) then
!!$          nindy = nindy + 1
!!$          indy(nindy) = j
!!$       endif
!!$    enddo
!!$
!!$    nindz = 0
!!$    do k=0,nz_box-1
!!$       j = mod(k + iz_box, nz_box) + 1
!!$       if (filter_atom(iz, forder, zgridmin(j), zgridmax(j), zdim(j))) then
!!$          nindz = nindz + 1
!!$          indz(nindz) = j
!!$       endif
!!$    enddo
!!$
!!$    if (nindy*nindz == 0) then
!!$       call wrndie(-5,'<domdec_r2d_comm>',&
!!$            'Error in setup, atom must belong to at least one node')
!!$    endif
!!$
!!$    if (nindy*nindz > 4) then
!!$       call wrndie(-5,'<domdec_2r_comm>',&
!!$            'Error in setup, atom belongs to more than 4 reciprocal nodes')
!!$    endif
!!$
!!$    nind = 0
!!$    do k=1,nindz
!!$       do j=1,nindy
!!$          nind = nind + 1
!!$          ind(nind) = indy(j) + (indz(k)-1)*ny_box
!!$       enddo
!!$    enddo
!!$
!!$    return
!!$  end subroutine get_recip_node_ind_left

  ! *
  ! * Receives coordinates from direct nodes
  ! *
  subroutine recv_coord_from_direct(natom, x, y, z, q_stop_recip)
    use pack_mod,only:unpack_double_byte, unpack_int_byte, unpack_xyz_atom
    use memory
    use image,only:xtlabc
    use number,only:zero
    use stream,only:outu, prnlev
    use domdec_common,only:ndirect, nthread, q_test, q_use_single, ncons, q_cons
    use mpi,only:mpi_integer, mpi_byte, mpi_success, mpi_statuses_ignore
    use parallel,only:mpi_integer_size, mpi_real8_size
    use new_timer,only:timer_start, timer_stpstrt, timer_stop, T_d2r, T_d2r_recv_size, T_d2r_recv,&
         T_d2r_unpack
    use reawri,only:qcnstp
    use domdec_dr_common,only:q_recip_node, get_recip_node, ncomm, commnode, reqbuffer, &
         commbuffersize, comm_direct_recip, commbufferpos, commbuffer, COORDBUF, &
         COORDBUFLEN, nreqbuffer, nrecip, mynod_split, alloc_realloc_commbuffer, &
         STOP_BIT, ATOML_BIT, cons_node_split
#if KEY_DOMDEC_GPU==1
    use domdec_common,only:q_gpu
    use domdec_util_gpu_mod,only:range_start, range_stop
#endif
    implicit none
    ! Input / Output
    integer, intent(in) :: natom
    real(chm_real), intent(inout) :: x(*), y(*), z(*)
    logical, intent(out) :: q_stop_recip
    ! Variables
    logical q_box_size_set, q_check_cons, q_recv_atoml
    real(chm_real) xyz(6), recip(3,3)
    real(chm_real4) recip_sp(3,3)
    integer commbuffersizetot
    integer max_recip_natom, ntot_recip_natom, header
    integer ierror
    integer i, j, k, pos

    ! Little sanity check to start with
    if (.not.q_recip_node) then
       call wrndie(-5,'<domdec_r2d_comm>',&
            'recv_coord_from_direct can only be called by a reciprocal node')
    endif

    ! Count the number of direct nodes this recip node communicates with
    ncomm = 0
    do i=0,ndirect-1
       if (get_recip_node(i) == mynod_split + ndirect) then
          ncomm = ncomm + 1
       endif
    enddo

    if (ncomm <= 0) then
       call wrndie(-5,'<domdec_r2d_comm>','Recip node has no direct nodes!')
    endif

    ! Allocate & reallocate memory (reqbuffer, commnode,
    !    commbuffersize, commbufferpos, recip_atoml, recip_natoml,
    !    n_conslist_th, conslist_th)
    call alloc_realloc(natom, ndirect, ncomm, nthread, ncons)

    ! Assign commnode(1:ncomm)
    ncomm = 0
    do i=0,ndirect-1
       if (get_recip_node(i) == mynod_split + ndirect) then
          ncomm = ncomm + 1
          commnode(ncomm) = i
       endif
    enddo

    ! Setup recip(3,3) -matrix
    call setup_recip(recip)

#if KEY_DOMDEC_GPU==1
     if (q_gpu) call range_start('Receive commbuffersize')
#endif

    call timer_start(T_d2r)

    q_box_size_set = .false.

    ! In sync

    call timer_start(T_d2r_recv_size)
    ! Receive commbuffersize
    do i=1,ncomm
       call mpi_irecv(commbuffersize(i), 1, mpi_integer, commnode(i), COORDBUFLEN, &
            comm_direct_recip, reqbuffer(i), ierror)
    enddo

#if KEY_DOMDEC_GPU==1
    if (q_gpu) call range_start('mpi_waitall')
#endif
    call mpi_waitall(ncomm, reqbuffer, mpi_statuses_ignore, ierror)
    if (ierror /= mpi_success) then
       call wrndie(-5,'<domdec_r2d_comm>','Error in mpi_waitall in comm_coord_to_recip')
    endif
#if KEY_DOMDEC_GPU==1
     if (q_gpu) call range_stop()
#endif

    call timer_stpstrt(T_d2r_recv_size, T_d2r_recv)

    ! Out of sync

#if KEY_DOMDEC_GPU==1
     if (q_gpu) call range_stop()
#endif

    ! Calculate commbuffer positions
    commbufferpos(1) = 1
    do i=2,ncomm
       commbufferpos(i) = commbufferpos(i-1) + commbuffersize(i-1)
    enddo

    ! Total number of bytes to receive
    commbuffersizetot = sum(commbuffersize(1:ncomm))

    ! Allocate / Reallocate commbuffer as needed
    call alloc_realloc_commbuffer(commbuffer, commbuffersizetot)

#if KEY_DOMDEC_GPU==1
     if (q_gpu) call range_start('Receive commbuffer')
#endif

    ! Receive commbuffer
    nreqbuffer = 0
    do i=1,ncomm
       if (commbuffersize(i) > 0) then
          nreqbuffer = nreqbuffer + 1
          call mpi_irecv(commbuffer(commbufferpos(i)), commbuffersize(i), mpi_byte, &
               commnode(i), COORDBUF, comm_direct_recip, reqbuffer(nreqbuffer), ierror)
       endif
    enddo
#if KEY_DOMDEC_GPU==1
    if (q_gpu) call range_start('mpi_waitall')
#endif
    call mpi_waitall(nreqbuffer, reqbuffer, mpi_statuses_ignore, ierror)
    if (ierror /= mpi_success) then
       call wrndie(-5,'<domdec_r2d_comm>','error in mpi_waitall commbuffer')
    endif
#if KEY_DOMDEC_GPU==1
    if (q_gpu) call range_stop()
#endif

    ! Out of sync

    call timer_stpstrt(T_d2r_recv, T_d2r_unpack)

#if KEY_DOMDEC_GPU==1
    if (q_gpu) call range_stop()
#endif

    ! Now commbuffer(:) contains all the data received from direct nodes
    q_stop_recip = .false.
    q_recv_atoml = .false.
    do i=1,ncomm
       k = 0
       if (commbuffersize(i) > 0) then
          ! For constant pressure, unpack new box sizes
          if (qcnstp) then
             call unpack_double_byte(commbuffer(commbufferpos(i)), k, xyz(1), 6)
!!$             call mpi_unpack(commbuffer(commbufferpos(i)), size(commbuffer), k, &
!!$                  xyz, 6, mpi_real8, comm_recip, ierror)
!!$             if (ierror /= mpi_success) call wrndie(-5, &
!!$                  '<domdec_r2d_comm>','Error in mpi_unpack in recv_coord_from_direct')
             if (xyz(1) < zero) then
                q_stop_recip = .true.
                cycle
             endif
             if (.not.q_box_size_set) then
                xtlabc(1:6) = xyz(1:6)
                call setup_recip(recip)
                q_box_size_set = .true.
             endif
          endif

          ! Unpack number of atoms
          call unpack_int_byte(commbuffer(commbufferpos(i)), k, header, 1)
!!$          call mpi_unpack(commbuffer(commbufferpos(i)), size(commbuffer), k, &
!!$               header, 1, mpi_integer, comm_recip, ierror)
!!$          if (ierror /= mpi_success) call wrndie(-5, &
!!$               '<domdec_r2d_comm>','Error in mpi_unpack in recv_coord_from_direct')
          if (iand(header, STOP_BIT) /= 0) then
             recip_natoml(i) = 0
             q_stop_recip = .true.
             cycle
          endif

          if (iand(header, ATOML_BIT) /= 0) then
             q_recv_atoml = .true.
          else
             if (recip_natoml(i) /= ishft(header, -2)) then
                call wrndie(-5,'<domdec_r2d_comm>',&
                     'Number of atoms received does not agree with number of atoms in recip_atoml')
             endif
          endif

          recip_natoml(i) = ishft(header, -2)

       endif
    enddo

    if (.not.q_stop_recip) then

       ntot_recip_natom = sum(recip_natoml(1:ncomm))

       ! Now recip_natoml(1:ncomm) = number of atoms we received from each direct node

       ! Re-size recip_atoml if needed
       ! max_recip_natom = upper bound for the number of atoms we received
       max_recip_natom = ntot_recip_natom
       ! Assuming we receive all constraint atoms
       if (ncons > 0) max_recip_natom = max_recip_natom + ncons
       if (size(recip_atoml) < max_recip_natom) then
          if (prnlev > 5) write (outu,'(a,i3,a,i8,a,i8)') 'recip node ',mynod_split, &
               ' resize recip_atoml from ',size(recip_atoml),' to ',max_recip_natom
          call chmdealloc('domdec_r2d_comm.src','recv_coord_from_direct','recip_atoml',&
               size(recip_atoml),intg=recip_atoml)
          call chmalloc('domdec_r2d_comm.src','recv_coord_from_direct','recip_atoml',&
               int(1.3*max_recip_natom),intg=recip_atoml)
       endif

       ! Allocate / Reallocate atomlist as needed
       ! NOTE: +1 here for making sure we have enough space for the case "nrecip > 1"
       !       (see "call unpack_xz" below)
       call alloc_realloc_atomlist_th(nthread, nrecip, max_recip_natom+1, .false.)

#if KEY_DOMDEC_GPU==1
       if (q_gpu) call range_start('Unpack coordinates')
#endif

       ! Unpack coordinates to recip_atoml(1:ntot_recip_natom)
       pos = 1
       do i=1,ncomm
          k = 0
          if (commbuffersize(i) > 0) then
             ! For constant pressure, unpack new box sizes
             if (qcnstp) then
                k = k + 6*mpi_real8_size
             endif

             ! Unpack number of atoms
             k = k + mpi_integer_size

             ! Unpack atom indices
             if (q_recv_atoml) then
                call unpack_int_byte(commbuffer(commbufferpos(i)), k, recip_atoml(pos), &
                     recip_natoml(i))
             endif
#if KEY_DOMDEC_GPU==1
             if (q_gpu) call range_start('loop')
#endif
             if (nrecip > 1) then
                call unpack_xyz_atom(recip_natoml(i), recip_atoml(pos:), &
                     commbuffer(commbufferpos(i)), k, x, y, z, &
                     atomlist_th(0)%atomlist(0)%array(pos+1:))
             else
                call unpack_xyz_atom(recip_natoml(i), recip_atoml(pos:), &
                     commbuffer(commbufferpos(i)), k, x, y, z, &
                     atomlist_th(0)%atomlist(1)%array(pos:))
             endif
             pos = pos + recip_natoml(i)
#if KEY_DOMDEC_GPU==1
             if (q_gpu) call range_stop()
#endif
          endif
       enddo

#if KEY_DOMDEC_GPU==1
       if (q_gpu) call range_stop()
#endif

       if (pos-1 /= ntot_recip_natom) then
          call wrndie(-5,'<domdec_r2d_comm>','Invalid pos')
       endif

       if (pos-1 > size(recip_atoml)) then
          call wrndie(-5,'<domdec_r2d_comm>','pos exceeds buffer size')
       endif

       ! Setup (filter_atom_y, filter_atom_z)
       if (nrecip > 1) then
          call setup_filter_atom_yz()
       endif

#if KEY_DOMDEC_GPU==1
       if (q_gpu) call range_start('build_atomlist_kernel_dp')
#endif

       n_cons_tot = 0
       if (nrecip > 1) then
          ! Determines if q_cons are checked or not
          q_check_cons = ((ncons > 0) .and. (mynod_split /= cons_node_split))
          n_atomlist_p(1:nrecip) = 0
          n_conslist_th(0) = 0
          !------------------------------
          ! Assign atoms to recip nodes
          !------------------------------
          if (q_use_single()) then
             recip_sp = recip
             call build_atomlist_kernel_sp(x, y, z, 1, ntot_recip_natom, &
                  atomlist_th(0)%atomlist(0)%array(2:ntot_recip_natom+1), &
                  recip_sp, q_check_cons, q_cons, &
                  nrecip, atomlist_th(0)%n_atomlist, atomlist_th(0)%atomlist, &
                  n_conslist_th(0), conslist_th(0)%array)
          else
             call build_atomlist_kernel_dp(x, y, z, 1, ntot_recip_natom, &
                  atomlist_th(0)%atomlist(0)%array(2:ntot_recip_natom+1), &
                  recip, q_check_cons, q_cons, &
                  nrecip, atomlist_th(0)%n_atomlist, atomlist_th(0)%atomlist, &
                  n_conslist_th(0), conslist_th(0)%array)
          endif
          n_cons_tot = n_conslist_th(0)
       else
          if (ntot_recip_natom /= natom) then
             call wrndie(-5,'<domdec_r2d_comm>','Error: ntot_recip_natom /= natom')
          endif
          atomlist_th(0)%n_atomlist(1) = ntot_recip_natom
       endif

#if KEY_DOMDEC_GPU==1
       if (q_gpu) call range_stop()
#endif

       n_grid_atom = n_atomlist_p(mynod_split+1)

       if (q_test) then
          call test_atomlist(x, y, z, q_use_single())
       endif

    endif

    call timer_stop(T_d2r_unpack)
    call timer_stop(T_d2r)

    return
  contains

    subroutine alloc_realloc(natom, ndirect, ncomm, nthread, ncons)
      use memory
      use nblist_util,only:init_array
      use mpi,only:mpi_status_size
      implicit none
      ! Input
      integer, intent(in) :: natom, ndirect, ncomm, nthread, ncons

      ! reqbuffer
      call alloc_realloc_integer_buffer(reqbuffer, max(ndirect,ncomm))

      ! commnode
      call alloc_realloc_integer_buffer(commnode, ncomm)

      ! commbuffersize
      call alloc_realloc_integer_buffer(commbuffersize, max(nrecip,ncomm))

      ! commbufferpos
      call alloc_realloc_integer_buffer(commbufferpos, max(nrecip,ncomm))

      ! recip_atoml
      call alloc_realloc_integer_buffer(recip_atoml, max(10,int(natom/nrecip)))

      ! recip_natoml
      call alloc_realloc_integer_buffer(recip_natoml, ncomm)

      ! n_conslist_th(0:nthread-1)
      call alloc_realloc_integer_buffer(n_conslist_th, nthread, 0)

      ! conslist_th(0:nthread-1)%array(1:ncons)
      call init_array(conslist_th, nthread, ncons, max(1,ncons))

      return
    end subroutine alloc_realloc

  end subroutine recv_coord_from_direct

  ! *
  ! * Communicate forces among recip nodes
  ! *
  subroutine comm_force_among_recip(dx, dy, dz)
    use pack_mod,only:pack_double_byte, unpack_double_byte
    use mpi,only:mpi_real8, mpi_byte, mpi_success, mpi_sum
    use parallel,only:mpi_integer_size, mpi_real8_size
    use domdec_dr_common,only:comm_recip, commbuffersize, commbufferpos, commbuffer, mynod_split, &
         nrecip, cons_node_split
#if KEY_DOMDEC_GPU==1
    use psf,only:natom
    use domdec_common,only:q_gpu, gpu_code_version
    use domdec_util_gpu_mod,only:range_start, range_stop, reduce_force_virial_gpu, &
         copy_force_virial_energy_from_gpu, wait_force_virial_energy_from_gpu, &
         combine_force_from_gpu
#endif
    implicit none
    ! Input / Output
    real(chm_real), intent(inout) :: dx(*), dy(*), dz(*)
    ! Variables
    real(chm_real) xyz(3)
    integer ierror
    integer i, j, k, l, jabs

#if KEY_DOMDEC_GPU==1
     if (q_gpu .and. gpu_code_version==2) then
        ! Reduce forces on GPU
        call reduce_force_virial_gpu()
        ! Copy forces from GPU to CPU
        call copy_force_virial_energy_from_gpu()
        ! Wait for copying to finish
        call wait_force_virial_energy_from_gpu()
        ! Combine forces from GPU
        ! NOTE: This only works for a single reciprocal node
        call combine_force_from_gpu(dx, dy, dz, natom)
     endif
#endif

    ! coord: received nrecvtot atoms from other recip nodes
    ! force: send nrecvtot forces to other recip nodes
    ! force: receive nsendtot forces

    ! Calculate commbuffersize and commbufferpos
    commbuffersize(1:nrecip) = nsend(1:nrecip)*3*mpi_real8_size
    commbufferpos(1) = 0
    do i=2,nrecip
       commbufferpos(i) = commbufferpos(i-1) + commbuffersize(i-1)
    enddo

#if KEY_DOMDEC_GPU==1
    if (q_gpu) call range_start('Pack forces')
#endif

    ! Pack forces into commbuffer2:
    ! forces in grid_atom(grid_atom_pos(i)+1:grid_atom_pos(i+1)) are sent to recip node i-1
    k = 0
    do i=1,nrecip
       commbuffer2size(i) = k
       commbuffer2pos(i) = k
       do j=grid_atom_pos(i)+1,grid_atom_pos(i+1)
          l = grid_atom_p(j)
          ! atom l force is sent to recip node i
          xyz(1:3) = (/ dx(l), dy(l), dz(l) /)
          call pack_double_byte(xyz(1), 3, commbuffer2(1), k)
       enddo
       ! NOTE: Only cons node does the loop below
       do j=cons_atom_pos(i)+1,cons_atom_pos(i+1)
          l = conslist_th(0)%array(j)
          ! atom l force is sent to recip node i
          xyz(1:3) = (/ dx(l), dy(l), dz(l) /)
          call pack_double_byte(xyz(1), 3, commbuffer2(1), k)
       enddo
       ! Number of bytes to be sent to recip node i-1
       commbuffer2size(i) = k - commbuffer2size(i)
    enddo

#if KEY_DOMDEC_GPU==1
    if (q_gpu) call range_stop()
#endif

#if KEY_DOMDEC_GPU==1
    if (q_gpu) call range_start('mpi_alltoallv')
#endif

    ! Communicate forces among recip nodes
    ! send buffer = commbuffer2
    ! recv buffer = commbuffer
    ! NOTE: after this call, all recip nodes have all forces
    call mpi_alltoallv(commbuffer2, commbuffer2size, commbuffer2pos, mpi_byte, &
         commbuffer, commbuffersize, commbufferpos, mpi_byte, comm_recip, ierror)

#if KEY_DOMDEC_GPU==1
    if (q_gpu) call range_stop()
#endif

    ! Unpack forces from commbuffer to (dx, dy, dz)
    commbuffersize(1:nrecip) = 0

#if KEY_DOMDEC_GPU==1
    if (q_gpu) call range_start('Unpack force')
#endif

    ! Unpack forces from other recip nodes and add to global force array
    do k=1,nrecip
       if (k /= mynod_split+1) then
          do i=1,n_atomlist_p(k)
             j = atomlist_p(k)%array(i)
             call unpack_double_byte(commbuffer(commbufferpos(k)+1), commbuffersize(k), &
                  xyz(1), 3)
             dx(j) = dx(j) + xyz(1)
             dy(j) = dy(j) + xyz(2)
             dz(j) = dz(j) + xyz(3)
          enddo
          if (k == cons_node_split+1) then
             do i=1,n_cons_tot
                j = conslist_th(0)%array(i)
                call unpack_double_byte(commbuffer(commbufferpos(k)+1), commbuffersize(k), &
                     xyz(1), 3)
                dx(j) = dx(j) + xyz(1)
                dy(j) = dy(j) + xyz(2)
                dz(j) = dz(j) + xyz(3)
             enddo
          endif
       endif
    enddo

#if KEY_DOMDEC_GPU==1
    if (q_gpu) call range_stop()
#endif

    ! Check commbuffersize
    do i=1,nrecip
       if (commbuffersize(i) /= nsend(i)*3*mpi_real8_size) then
          call wrndie(-5,'<domdec_r2d_comm>','Invalid commbuffersize (2)')
       endif
    enddo

    return
  end subroutine comm_force_among_recip

  ! *
  ! * Sets the recip forces in (forcex, forcey, forcez) to zero
  ! *
  subroutine zero_recip_force(forcex, forcey, forcez)
    use number,only:zero
    use domdec_dr_common,only:nrecip,mynod_split
    implicit none
    ! Input / Output
    real(chm_real), intent(out) :: forcex(*), forcey(*), forcez(*)
    ! Variables
    integer i, j, n

    do i=1,nrecip
       do j=1,n_atomlist_p(i)
          n = atomlist_p(i)%array(j)
          forcex(n) = zero
          forcey(n) = zero
          forcez(n) = zero
       enddo
       do j=cons_atom_pos(i)+1,cons_atom_pos(i+1)
          n = conslist_th(0)%array(j)
          forcex(n) = zero
          forcey(n) = zero
          forcez(n) = zero
       enddo
    enddo

    return
  end subroutine zero_recip_force

  ! *
  ! * Adds the recip forces to the homezone I atoms
  ! * (forcex, forcey, forcez) += (recipforcex, recipforcey, recipforcez)
  ! *
  subroutine reduce_recip_forces(recipforcex, recipforcey, recipforcez, forcex, forcey, forcez)
    use domdec_common,only:natoml, atoml
    implicit none
    ! Input / Output
    real(chm_real), intent(in) :: recipforcex(*), recipforcey(*), recipforcez(*)
    real(chm_real), intent(inout) :: forcex(*), forcey(*), forcez(*)
    ! Variables
    integer i, j

!$omp parallel do private(i, j)
    do i=1,natoml
       j = atoml(i)
       forcex(j) = forcex(j) + recipforcex(j)
       forcey(j) = forcey(j) + recipforcey(j)
       forcez(j) = forcez(j) + recipforcez(j)
    enddo
!$omp end parallel do

    return
  end subroutine reduce_recip_forces

  ! *
  ! * Send forces, energies, and virial to direct nodes
  ! *
  subroutine send_results_to_direct(forcex, forcey, forcez, nauxdata, auxdata)
    use number,only:zero
    use pack_mod,only:pack_xyz_atom_byte, pack_double_byte, pack_double_zeros
    use mpi,only:mpi_success, mpi_byte, mpi_statuses_ignore
    use domdec_dr_common,only:ncomm, commbuffersize, commbufferpos, commbuffer, &
         nreqbuffer, reqbuffer, commnode, FORCEBUF, comm_direct_recip, mynod_split
#if KEY_DOMDEC_GPU==1
    use domdec_common,only:q_gpu
    use domdec_util_gpu_mod,only:range_start, range_stop
#endif
    implicit none
    ! Input / Output
    real(chm_real), intent(in) :: forcex(*), forcey(*), forcez(*)
    integer, intent(in) :: nauxdata
    real(chm_real), intent(in) :: auxdata(:)
    ! Variables
    real(chm_real) xyz(3)
    integer ierror
    integer i, j, k, m, pos

    ! #### RECIP ####

#if KEY_DOMDEC_GPU==1
    if (q_gpu) call range_start('Pack recip forces')
#endif

    ! Pack forces into commbuffer
    ! recip_natoml(i) = number of atoms that are sent to direct node commnode(i)
    pos = 1
    k = 0
    do i=1,ncomm
       commbuffersize(i) = k
       commbufferpos(i) = k + 1
       call pack_xyz_atom_byte(recip_natoml(i), recip_atoml(pos:), commbuffer(1), k, &
            forcex, forcey, forcez)
       pos = pos + recip_natoml(i)
       ! Only send the auxdata to the first direct node to avoid double counting
       if (i == 1) then
          call pack_double_byte(auxdata(1), nauxdata, commbuffer(1), k)
       else
          call pack_double_zeros(nauxdata, commbuffer(1), k)
       endif
       commbuffersize(i) = k - commbuffersize(i)
    enddo

#if KEY_DOMDEC_GPU==1
    if (q_gpu) call range_stop()
#endif

    ! Send force data to direct nodes
    nreqbuffer = 0
    do i=1,ncomm
       if (commbuffersize(i) > 0) then
          nreqbuffer = nreqbuffer + 1
          call mpi_isend(commbuffer(commbufferpos(i)), commbuffersize(i), &
               mpi_byte, commnode(i), FORCEBUF, comm_direct_recip, &
               reqbuffer(nreqbuffer), ierror)
          if (ierror /= mpi_success) then
             call wrndie(-5,'<domdec_r2d_comm>','Error in mpi_isend buffer')
          endif
       endif
    enddo

    call mpi_waitall(nreqbuffer, reqbuffer, mpi_statuses_ignore, ierror)
    if (ierror /= mpi_success) then
       call wrndie(-5,'<domdec_r2d_comm>','Error in mpi_waitall in send_results_to_direct')
    endif

    return
  end subroutine send_results_to_direct

  ! *
  ! * If needed, allocates (recipforcex, recipforcey, recipforcez)
  ! * If needed, reallocates
  ! *
  subroutine alloc_realloc_recipforce(natom)
    use memory
    implicit none
    ! Input / Output
    integer, intent(in) :: natom

    if (allocated(recipforcex)) then
       if (size(recipforcex) < natom) then
          call chmdealloc('domdec_r2d_comm.src','alloc_realloc_recipforce','recipforcex',&
               size(recipforcex),crl=recipforcex)
          call chmdealloc('domdec_r2d_comm.src','alloc_realloc_recipforce','recipforcey',&
               size(recipforcey),crl=recipforcey)
          call chmdealloc('domdec_r2d_comm.src','alloc_realloc_recipforce','recipforcez',&
               size(recipforcez),crl=recipforcez)
       endif
    endif

    if (.not.allocated(recipforcex)) then
       call chmalloc('domdec_r2d_comm.src','alloc_realloc_recipforce','recipforcex',&
            natom,crl=recipforcex)
       call chmalloc('domdec_r2d_comm.src','alloc_realloc_recipforce','recipforcey',&
            natom,crl=recipforcey)
       call chmalloc('domdec_r2d_comm.src','alloc_realloc_recipforce','recipforcez',&
            natom,crl=recipforcez)
    endif

    return
  end subroutine alloc_realloc_recipforce

  ! *
  ! * Sets up recip(3,3) -matrix
  ! *
  subroutine setup_recip(recip)
    use image,only:xtlabc
    implicit none
    ! Output
    real(chm_real), intent(out) :: recip(3,3)
    ! Variables
    real(chm_real) xtlinv(6)
    logical ok

    call invt33s(xtlinv, xtlabc, ok)
    recip(1,1) = xtlinv(1)
    recip(2,2) = xtlinv(3)
    recip(3,3) = xtlinv(6)
    recip(1,2) = xtlinv(2)
    recip(2,1) = xtlinv(2)
    recip(1,3) = xtlinv(4)
    recip(3,1) = xtlinv(4)
    recip(2,3) = xtlinv(5)
    recip(3,2) = xtlinv(5)

    return
  end subroutine setup_recip

  ! *
  ! * Allocates / Reallocates atomlist_th as needed with 50% extra space
  ! * NOTE: maximum size is restricted by natom+1
  ! * if q_copy = .true. we copy old grid_atom_p -list over
  ! *
  subroutine alloc_realloc_atomlist_th(nthread, nrecip, req_size_in, q_copy)
    use memory
    use psf,only:natom
    use nblist_util,only:init_array, uninit_array
    use domdec_dr_common,only:mynod_split
    implicit none
    ! Input / Output
    integer, intent(in) :: nthread, nrecip, req_size_in
    logical, intent(in) :: q_copy
    ! Variables
    type(intarray_t), allocatable, dimension(:) :: atomlist_copy
    integer req_size, new_size, old_size, this_size
    integer len, i, j

    req_size = req_size_in
    new_size = int(req_size*1.5)

    req_size = min(req_size, natom+1)
    new_size = min(new_size, natom+1)

    nullify(atomlist_p)
    nullify(grid_atom_p)
    nullify(n_atomlist_p)

    if (allocated(atomlist_th)) then
       if (size(atomlist_th) < nthread) then
          if (q_copy) then
             call wrndie(-5,'<domdec_r2d_comm>',&
                  'alloc_realloc_atomlist_th: q_copy not supported when allocating entire struture')
          endif
          ! Different number of threads, need to reallocate the the entire structure
          do i=0,size(atomlist_th)-1
             call uninit_array(atomlist_th(i)%atomlist)
             call chmdealloc('domdec_r2d_comm.src','alloc_realloc_atomlist_th',&
                  'atomlist_th(i)%n_atomlist',size(atomlist_th(i)%n_atomlist),&
                  intg=atomlist_th(i)%n_atomlist)
             enddo
          deallocate(atomlist_th)
       endif
    endif

    if (.not.allocated(atomlist_th)) then
       allocate(atomlist_th(0:nthread-1))
    endif

    do i=0,nthread-1
       if (q_copy .and. i == 0) then
          old_size = 0
          do j=0,nrecip
             old_size = max(old_size, size(atomlist_th(0)%atomlist(j)%array))
          enddo
          if (req_size > old_size) then
             call init_array(atomlist_copy, nrecip+1, old_size, old_size)
             do j=0,nrecip
                this_size = size(atomlist_th(0)%atomlist(j)%array)
                atomlist_copy(j)%array(1:this_size) = &
                     atomlist_th(0)%atomlist(j)%array(1:this_size)
             enddo
          endif
       endif

       call init_array(atomlist_th(i)%atomlist, nrecip+1, req_size, new_size)
       if (allocated(atomlist_th(i)%n_atomlist)) then
          if (size(atomlist_th(i)%n_atomlist) < nrecip+1) then
             call chmdealloc('domdec_r2d_comm.src','alloc_realloc_atomlist_th',&
                  'atomlist_th(i)%n_atomlist',size(atomlist_th(i)%n_atomlist),&
                  intg=atomlist_th(i)%n_atomlist)
          endif
       endif
       if (.not.allocated(atomlist_th(i)%n_atomlist)) then
          ! +32 is added here to avoid cache trashing within threaded code
          call chmalloc('domdec_r2d_comm.src','alloc_realloc_atomlist_th',&
               'atomlist_th(i)%n_atomlist',nrecip+132,lbou=0,&
               intg=atomlist_th(i)%n_atomlist)
       endif

       if (allocated(atomlist_copy)) then
          do j=0,nrecip
             this_size = size(atomlist_copy(j)%array)
             atomlist_th(0)%atomlist(j)%array(1:this_size) = &
                  atomlist_copy(j)%array(1:this_size)
          enddo
          call uninit_array(atomlist_copy)
       endif

    enddo

    atomlist_p => atomlist_th(0)%atomlist
    grid_atom_p => atomlist_th(0)%atomlist(mynod_split+1)%array
    n_atomlist_p => atomlist_th(0)%n_atomlist

    return
  end subroutine alloc_realloc_atomlist_th

  ! *
  ! * Allocates / Reallocates integer buffer as needed
  ! *
  subroutine alloc_realloc_integer_buffer(buffer, req_size, lowbound_in)
    use memory
    implicit none
    ! Input / Output
    integer, allocatable, dimension(:), intent(inout) :: buffer
    integer, intent(in) :: req_size
    integer, intent(in), optional :: lowbound_in
    ! Variables
    integer lowbound

    if (allocated(buffer)) then
       if (size(buffer) < req_size) then
          call chmdealloc('domdec_r2d_comm.src','alloc_realloc_integer_buffer','buffer',&
               size(buffer),intg=buffer)
       endif
    endif

    if (.not.allocated(buffer)) then
       if (present(lowbound_in)) then
          lowbound = lowbound_in
       else
          lowbound = 1
       endif
       call chmalloc('domdec_r2d_comm.src','alloc_realloc_integer_buffer','buffer',&
            req_size,lbou=lowbound,intg=buffer)
    endif

    return
  end subroutine alloc_realloc_integer_buffer

  ! *
  ! * De-allocates memory for this module
  ! *
  subroutine uninit_r2d_comm()
    use memory,only:chmdealloc
    use nblist_util,only:uninit_array
    implicit none
    integer i

    ! nrecv
    if (allocated(nrecv)) then
       call chmdealloc('domdec_r2d_comm.src','deallocate_r2d_comm','nrecv',&
            size(nrecv),intg=nrecv)
    endif

    ! nsend
    if (allocated(nsend)) then
       call chmdealloc('domdec_r2d_comm.src','deallocate_r2d_comm','nsend',&
            size(nsend),intg=nsend)
    endif

    ! grid_atom_pos
    if (allocated(grid_atom_pos)) then
       call chmdealloc('domdec_r2d_comm.src','deallocate_r2d_comm','grid_atom_pos',&
            size(grid_atom_pos),intg=grid_atom_pos)
    endif

    ! recip_atoml
    if (allocated(recip_atoml)) then
       call chmdealloc('domdec_r2d_comm.src','deallocate_r2d_comm','recip_atoml',&
            size(recip_atoml),intg=recip_atoml)
    endif

    ! recip_natoml
    if (allocated(recip_natoml)) then
       call chmdealloc('domdec_r2d_comm.src','deallocate_r2d_comm','recip_natoml',&
            size(recip_natoml),intg=recip_natoml)
    endif

    ! commbuffer2pos
    if (allocated(commbuffer2pos)) then
       call chmdealloc('domdec_r2d_comm.src','deallocate_r2d_comm','commbuffer2pos',&
            size(commbuffer2pos),intg=commbuffer2pos)
    endif

    ! commbuffer2size
    if (allocated(commbuffer2size)) then
       call chmdealloc('domdec_r2d_comm.src','deallocate_r2d_comm','commbuffer2size',&
            size(commbuffer2size),intg=commbuffer2size)
    endif

    ! commbuffer2
    if (associated(commbuffer2)) then
       call chmdealloc('domdec_r2d_comm.src','deallocate_r2d_comm','commbuffer2',&
            size(commbuffer2),mibyp=commbuffer2)
    endif

    ! filter_atom_y
    if (allocated(filter_atom_y)) then
       call chmdealloc('domdec_r2d_comm.src','deallocate_r2d_comm','filter_atom_y',&
            2,size(filter_atom_y),intg=filter_atom_y)
    endif

    ! filter_atom_z
    if (allocated(filter_atom_z)) then
       call chmdealloc('domdec_r2d_comm.src','deallocate_r2d_comm','filter_atom_z',&
            2,size(filter_atom_z),intg=filter_atom_z)
    endif

    ! atomlist_th
    call dealloc_atomlist_th()

    ! (recipforcex, recipforcey, recipforcez)
    if (allocated(recipforcex)) then
       call chmdealloc('domdec_r2d_comm.src','deallocate_r2d_comm','recipforcex',&
            size(recipforcex),crl=recipforcex)
       call chmdealloc('domdec_r2d_comm.src','deallocate_r2d_comm','recipforcey',&
            size(recipforcey),crl=recipforcey)
       call chmdealloc('domdec_r2d_comm.src','deallocate_r2d_comm','recipforcez',&
            size(recipforcez),crl=recipforcez)
    endif

    ! cons_atom_pos
    if (allocated(cons_atom_pos)) then
       call chmdealloc('domdec_r2d_comm.src','deallocate_r2d_comm','cons_atom_pos',&
            size(cons_atom_pos),intg=cons_atom_pos)
    endif

    ! n_conslist_th
    if (allocated(n_conslist_th)) then
       call chmdealloc('domdec_r2d_comm.src','deallocate_r2d_comm','n_conslist_th',&
            size(n_conslist_th),intg=n_conslist_th)
    endif

    ! conslist_th
    call uninit_array(conslist_th)

    return
  end subroutine uninit_r2d_comm

  ! *
  ! * Deallocates atomlist_th and nulls the pointers
  ! *
  subroutine dealloc_atomlist_th()
    use nblist_util,only:uninit_array
    implicit none
    integer i

    if (allocated(atomlist_th)) then
       do i=0,size(atomlist_th)-1
          call uninit_array(atomlist_th(i)%atomlist)
       enddo
       deallocate(atomlist_th)
    endif
    nullify(atomlist_p)
    nullify(grid_atom_p)
    nullify(n_atomlist_p)

    return
  end subroutine dealloc_atomlist_th

!#endif /* (not_cmpi)*/
#endif /* (parallel)*/
#endif /* (domdec_main)*/
end module domdec_r2d_comm
