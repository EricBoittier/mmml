module domdec_common

  ! *
  ! * Domain decomposition common data and subroutines
  ! * NOTE: All data and subroutines here are public
  ! *

  use chm_kinds
  use dimens_fcm
#if KEY_DOMDEC==1
  use mpi
#endif 
  use number
  implicit none
  public

  ! .true. for constraint node (=recip root node), .false. for others
  logical q_cons_node

  ! Total list of constraints/restraints
  ! q_cons(1:natom) = .true. for atoms that are involved in constraints / restraints
  logical, allocatable, dimension(:) :: q_cons
  ! Total number of constraints
  integer :: ncons = 0
  ! List of atoms in q_cons(1:natom)
  integer, allocatable, dimension(:) :: conslist

  ! Number of threads per MPI node
  integer :: nthread = 1

  ! SIMD version:
  integer, parameter :: SIMD_NONE = 0, SIMD_SSE = 1, SIMD_AVX = 2, SIMD_AVX_FMA = 3, &
       SIMD_AVX2 = 4, SIMD_AVX2_FMA = 5
  integer :: simd_version = SIMD_NONE

  ! CPU Vendor:
  integer, parameter :: CPU_UNKNOWN = 0, CPU_INTEL = 1, CPU_AMD = 2
  integer :: cpu_vendor = CPU_UNKNOWN

#if KEY_DOMDEC==1 /*domdec_main*/
  ! Logical flag determining if GPU code is used
  logical :: q_gpu = .false.

  ! .true. if energy/virial etc. output is required
  ! NOTE: this is used especially in the GPU code to determine if
  !       energy and virial should be computed along with the force
  !       computation
  logical :: q_print_output = .false.

  ! 1 = GPU: direct non-bonded forces, CPU: rest
  ! 2 = GPU: direct+reciprocal non-bonded forces, CPU: rest
  integer :: gpu_code_version = 1

  ! Logical flag determining if the simulation box is orthorhombic or not
  logical q_ortho

  ! Logical flag determining if Domain Decomposition is turned on or off
  logical :: q_domdec = .false.

  ! Spatial group sorts
  logical :: q_sort_groups = .false.

  ! Precision model
  logical :: q_single = .false.

  ! True if domdec needs to be re-started
  logical :: q_system_changed = .true.

  ! Flag that tells if iblo14 and inb14 have changed
  logical :: q_inb_changed = .true.  

  ! If set, runs unit testing version of the DOMDEC/DOMDEC_GPU engine
  logical :: q_test = .false.

  ! Number of communication boxes
  integer nx_comm, ny_comm, nz_comm

  ! Wallclock time for evaluating the non-bonded energies
  real(chm_real) energy_time

  ! min and max of atom coordinates for each zone
  real(chm_real4) min_xyz(3,8), max_xyz(3,8)

  ! Points per Angstrom in the lookup tables. Default 200
  integer :: ppang = 200

  ! q_split = .true. => direct/recip split is on
  logical :: q_split = .false.

  ! Orthogonal box sizes
  real(chm_real) boxx, boxy, boxz
  real(chm_real) hboxx, hboxy, hboxz
  real(chm_real) invx, invy, invz
  ! Box displacement vectors
  real(chm_real) xdisp(3), ydisp(2), zdisp
  ! Lattice vectors
  real(chm_real) box_a(3), box_b(3), box_c(3)
  ! Number of boxes in each coordinate direction and total number of boxes
  ! (ndirect = nx*ny*nz)
  integer nx, ny, nz, ndirect
  ! Index of home box for this node
  integer homeix, homeiy, homeiz
  ! Size of the boxes
  real(chm_real) frx, fry, frz
  ! Index of the CPU node for each box, size (nx, ny, nz)
  integer, allocatable, dimension(:,:,:) :: nodeind

  ! Gives the home zone of this atom
  ! I,FZ,FY,EX,FX,EZ,EY,C = 1,...8. 0 if not on this node
  integer, allocatable, dimension(:) :: homezone
  
  integer neighind(-1:1,-1:1,-1:1), neighlist(26)
  integer nneigh

  ! Zone particle lists
  ! zone I: 1:zonelist(1)
  ! zone FZ: zonelist(1)+1:zonelist(2)
  ! etc...
  integer zonelist(8)        ! End index for groups for each zone
  integer zonelist_atom(8)   ! End index for atoms for each zone
  integer, allocatable, dimension(:) :: groupl  ! Group list
  integer natoml             ! Number of atoms in the homebox
  integer natoml_tot         ! Total number of atoms that this node has
  integer, target, allocatable, dimension(:) :: atoml   ! Atom list

#endif /* (domdec_main)*/

contains

#if KEY_DOMDEC==1 /*domdec_main*/

  subroutine domdec_system_changed()
    implicit none

    q_system_changed = .true.

    return
  end subroutine domdec_system_changed

  ! *
  ! * Tests coordinates. DOMDEC version of TSTCRD -routine
  ! *
  subroutine testcoord_domdec(ok, x, y, z)
    implicit none
    ! Input / Output
    logical, intent(out) :: ok
    real(chm_real), intent(in) :: x(*), y(*), z(*)
    ! Variables
    real(chm_real), parameter :: range = 9990.0_chm_real
    integer i, j

    ok = .true.

!$omp parallel do schedule(static) private(j, i) reduction(.and.:ok)
    do j=1,natoml
       i = atoml(j)
       ok=(x(i).gt.-range .and. x(i).lt.range) .and. &
            (y(i).gt.-range .and. y(i).lt.range) .and. &
            (z(i).gt.-range .and. z(i).lt.range)
    enddo
!$omp end parallel do

    if(.not.ok) then
       call wrndie(-1,'<testcoord_domdec>', &
            'Some atom coordinates undefined or out of range')
    endif

    return
  end subroutine testcoord_domdec

  !
  ! Returns the node index for box (ix, iy, iz)
  !
  integer function nodeindfunc(ix,iy,iz)
    implicit none
    ! Input
    integer ix, iy, iz
    ! Variables
    integer i, ixt, iyt, izt
    
    ixt = ix
    iyt = iy
    izt = iz
    do while (ixt > nx)
       ixt = ixt - nx
    enddo
    do while (ixt <= 0) 
       ixt = ixt + nx
    enddo
    do while (iyt > ny)
       iyt = iyt - ny
    enddo
    do while (iyt <= 0) 
       iyt = iyt + ny
    enddo
    do while (izt > nz)
       izt = izt - nz
    enddo
    do while (izt <= 0) 
       izt = izt + nz
    enddo
    nodeindfunc = nodeind(ixt,iyt,izt)
    
    return
  end function nodeindfunc

  ! *
  ! * Calculates the minimum number of 
  ! *
  subroutine calc_min_nx_ny_nz(min_nx, min_ny, min_nz)
    use number,only:two
    use inbnd,only:cutnb
    use groupxfast,only:maxgrp_rad
    implicit none
    ! Input / Output
    integer, intent(out) :: min_nx, min_ny, min_nz
    ! Variables
    real(chm_real) cutgrp

    ! maxgrp_rad was calculated when make_groups was called
    cutgrp = cutnb + two*maxgrp_rad

    min_nx = ceiling(boxx/(boxx-two*cutgrp))
    min_ny = ceiling(boxy/(boxy-two*cutgrp))
    min_nz = ceiling(boxz/(boxz-two*cutgrp))

    return
  end subroutine calc_min_nx_ny_nz

    ! *
  ! * Sets box sizes from xucell
  ! *
  subroutine set_box()
    use number,only:one,half,zero
    use image,only:xucell
    implicit none
    boxx = xucell(1)
    boxy = xucell(2)
    boxz = xucell(3)
    invx = one/boxx
    invy = one/boxy
    invz = one/boxz
    hboxx = half*boxx
    hboxy = half*boxy
    hboxz = half*boxz
    if (q_ortho) then
       box_a(1) = boxx
       box_a(2) = zero
       box_a(3) = zero
       box_b(1) = zero
       box_b(2) = boxy
       box_b(3) = zero
       box_c(1) = zero
       box_c(2) = zero
       box_c(3) = boxz
    endif
    return
  end subroutine set_box

#endif /* (domdec_main)*/

  ! *
  ! * Divides n iterations evenly on threads
  ! *
  subroutine divide_thread_work(n, istart, iend)
    implicit none
    ! Input / Output
    integer, intent(in) :: n
    integer, intent(out) :: istart, iend
#ifdef _OPENMP
    ! Functions
    integer omp_get_thread_num
#endif 
#if KEY_DOMDEC==1
    ! Variables
    integer tid
#endif 

#if KEY_DOMDEC==1 /*domdec*/
#ifdef _OPENMP
    tid = omp_get_thread_num()
#else /**/
    tid = 0
#endif 
    istart = tid*n/nthread + 1
    iend = (tid+1)*n/nthread
#else /* (domdec)*/
    istart = 1
    iend = n
#endif /* (domdec)*/

    return
  end subroutine divide_thread_work

  ! *
  ! * Returns .true. if single precision is used, .false. otherwise
  ! *
  logical function q_use_single()
    implicit none

#if KEY_DOMDEC==1
    q_use_single = q_domdec .and. q_single
#else
    q_use_single = .false.
#endif

    return
  end function q_use_single

end module domdec_common

