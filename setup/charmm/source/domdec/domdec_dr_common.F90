module domdec_dr_common

  ! *
  ! * Domain decomposition direct-reciprocal common setup routines
  ! * NOTE: Only direct nodes call subroutines in this module
  ! *

#if KEY_DOMDEC==1 /*domdec_main*/
#if KEY_PARALLEL==1 /*parallel*/
!#if KEY_CMPI==0 /*not_cmpi*/
  use chm_kinds
  use dimens_fcm
  use mpi,only:MPI_STATUS_SIZE
  implicit none
  private

  ! Parameters
  ! Constraint node after split
  integer, parameter :: cons_node_split = 0
  ! Constraint node before split
  integer cons_node

  ! q_interleave = .true. => interleave direct and recip nodes
  logical :: q_interleave = .false.

  ! Bits for Direct-Recip communication
  integer, parameter :: STOP_BIT = 1, ATOML_BIT = 2

  ! Communication tags for mpi:
  integer, parameter :: COORDBUF=1, COORDBUFLEN=2, FORCEBUF=4

  ! q_recip_node  = .true. => this node is a reciprocal node
  ! q_direct_node = .true. => this node is a direct node
  logical :: q_direct_node = .true., q_recip_node = .false.

  ! Number of reciprocal nodes
  integer :: nrecip = 0

  ! MPI Commmunicators
  integer comm_recip, comm_direct, comm_direct_recip

  ! Defines direct and reciprocal node mapping:
  ! direct_nodes(1:ndirect) = node ranks for direct nodes 1:ndirect
  ! recip_nodes(1:nrecip)   = node ranks for reciprocal nodes 1:nrecip
  integer, allocatable, dimension(:) :: direct_nodes, recip_nodes

  ! Save variables
  integer comm_charmm_save, numnod_save, mynod_save

  ! Mynod (rank) after start_direct_recip is called
  ! direct nodes: mynod_split = 0...ndirect-1
  ! recip nodes:  mynod_split = 0...nrecip-1
  integer mynod_split

  integer(int_byte), pointer, dimension(:) :: commbuffer
  integer, allocatable, dimension(:) :: commbuffersize, reqbuffer
  integer, allocatable, dimension(:) :: commbufferpos, commnode

  integer ncomm, nreqbuffer

  ! .true. if "ndir" -option was set in domdec_com
  logical ndir_set


  ! Public subroutines
  public init_direct_recip, &
       start_split_direct_recip, stop_split_direct_recip, &
       merge_direct_recip, determine_direct_split, &
       copy_to_recip, alloc_realloc_commbuffer, uninit_dr_common

  ! Public functions
  public get_recip_node

  ! Public variables
  public COORDBUF, COORDBUFLEN, FORCEBUF, q_direct_node, q_recip_node, &
       nrecip, mynod_save, &
       comm_recip, comm_direct, comm_direct_recip, direct_nodes, recip_nodes, mynod_split, &
       commbuffer, commbuffersize, reqbuffer, commbufferpos, commnode, ncomm, &
       nreqbuffer, ndir_set
  public cons_node_split, cons_node
  public STOP_BIT, ATOML_BIT

contains

  ! *
  ! * Initialize direct/recip split
  ! *
  subroutine init_direct_recip(qpme, q_split_set, q_split_new)
    use domdec_common,only:nx, ny, nz, ndirect, set_box, calc_min_nx_ny_nz, q_cons_node, q_split
    use parallel,only:numnod, mynod
#if KEY_DOMDEC_GPU==1
    use domdec_common,only:q_gpu, gpu_code_version
#endif
    implicit none
    ! Input
    logical, intent(in) :: qpme, q_split_set, q_split_new
    ! Variables
    integer min_nx, min_ny, min_nz

    nreqbuffer = 0

    ! Remove old split if it exists
    call merge_direct_recip()

    ! No NDIR set, guess ndirect/nrecip split and set nx, ny, nz
    if (.not.ndir_set) then
       ! Get the minimum number of nodes for each direction
       call determine_direct_recip_split(numnod, qpme, q_split_set, q_split_new, &
            ndirect, nrecip)
    else
       ndirect = nx*ny*nz
       nrecip = numnod - ndirect
    endif

    if (ndirect < 1) then
       call wrndie(-5,'<domdec_dr_common>','NDIR values must be positive integers')
    elseif (ndirect > numnod) then
       call wrndie(-5,'<domdec_dr_common>','Number of requested sub-boxes exceeds number of cores')
    endif

    if (q_split_set .and. .not.q_split_new .and. ndirect /= numnod) then
       call wrndie(-5,'<domdec_dr_common>',&
            'When SPLIt is OFF, you must use all nodes as direct nodes')
    endif

    if (q_split_set .and. q_split_new .and. nrecip == 0) then
       call wrndie(-5,'<domdec_dr_common>','When SPLIt is ON, you must have nodes left for recip')
    endif

    if (q_split_set) then
       ! SPLIt is defined in the command line
       if (q_split_new) then
          q_split = .true.
          nrecip = numnod - ndirect
       else
          q_split = .false.
          nrecip = ndirect
       endif
    else
       if (numnod == ndirect .and. qpme) then
          ! PME: Direct nodes are also reciprocal nodes <=> no split
          q_split = .false.
          nrecip = ndirect
       elseif (qpme) then
          ! PME: Direct / Reciprocal node split
          q_split = .true.
          nrecip = numnod - ndirect
       else
          ! No PME <=> no split
          q_split = .false.
          nrecip = 0
       endif
    endif

    ! Sanity checks
    if (nrecip > ndirect) then
       call wrndie(-5,'<domdec_dr_common>',&
            'Number of recip nodes larger than number of direct nodes')
    endif

#if KEY_DOMDEC_GPU==1
    if (q_gpu) then
       if (ndirect /= numnod .and. gpu_code_version==1) then
          call wrndie(-5,'<domdec_dr_common>','DOMDEC_GPU ver 1.0 requires #direct nodes = #mpi nodes')
       endif
       if (numnod > 1 .and. gpu_code_version==2) then
          if (qpme) then
             if (ndirect /= numnod-1) then
                call wrndie(-5,'<domdec_dr_common>','DOMDEC_GPU ver 2.0 with PME requires #recip nodes = 1')
             endif
          else
             if (ndirect /= numnod .or. nrecip /= 0) then
                call wrndie(-5,'<domdec_dr_common>','DOMDEC_GPU ver 2.0 without PME requires #recip nodes = 0')
             endif
          endif
       endif
    endif
#endif

    q_cons_node = .false.

    ! Split into direct/reciprocal nodes
    call split_direct_recip()

    return
  end subroutine init_direct_recip

  ! *
  ! * Determine ndirect/nrecip split
  ! *
  subroutine determine_direct_recip_split(numnod, qpme, q_split_set, q_split_new, &
       ndirect_out, nrecip_out)
    use memory
    use domdec_common,only:q_gpu
#if KEY_DOMDEC_GPU==1
    use domdec_common,only:gpu_code_version
#endif
    implicit none
    ! Input / Output
    integer, intent(in) :: numnod
    logical, intent(in) :: qpme, q_split_set, q_split_new
    integer, intent(out) :: ndirect_out, nrecip_out
    ! Variables
    integer nfactor
    integer, allocatable, dimension(:) :: factor

    call chmalloc('domdec_dr_common.src','determine_direct_recip_split','factor',&
         max(10,numnod/2),intg=factor)

    nrecip_out = 0

    if ((q_split_set .and. .not.q_split_new) .or. .not.qpme .or. numnod==1) then
       ! SPLIt is OFF => Set nrecip = 0 for now
       ! No PME => nrecip = 0
       ! Numnod ==1 => nrecip = 0
       nrecip_out = 0
    else
       ! Set nrecip to be approx 1/4 of total number of nodes
       ! We look for a close-by non-prime number
       nrecip_out = max(1,numnod / 4)
       if (nrecip_out > 8) then
          call prime_factors(nrecip_out, factor, nfactor)
          do while (nfactor == 2 .and. nrecip_out > 1)
             nrecip_out = max(1,nrecip_out - 1)
             call prime_factors(nrecip_out, factor, nfactor)
          enddo
       endif
#if KEY_DOMDEC_GPU==1
       if (q_gpu) then
          if (gpu_code_version==2) then
             nrecip_out = 1
          else
             ! GPU on and code version /= 2 => nrecip = 0
             nrecip_out = 0
          endif
       endif
#endif
     endif

    call chmdealloc('domdec_dr_common.src','determine_direct_recip_split','factor',&
         max(10,numnod/2),intg=factor)

    ndirect_out = numnod - nrecip_out

    return
  end subroutine determine_direct_recip_split

  ! *
  ! * Splits direct into (nx, ny, nz)
  ! *
  subroutine determine_direct_split(min_nx, min_ny, min_nz, ndirect, nx, ny, nz)
    use stream,only:outu, prnlev
    use number,only:two
    use inbnd,only:cutnb
    use domdec_common,only:boxx, boxy, boxz
    use groupxfast,only:maxgrp_rad
    use memory
    implicit none
    ! Input / Output
    integer, intent(in) :: min_nx, min_ny, min_nz, ndirect
    integer, intent(out) :: nx, ny, nz
    ! Variables
    real(chm_real) cutgrp
    integer n(3), min_n(3)
    integer nfactor
    integer, allocatable, dimension(:) :: factor

    cutgrp = cutnb + two*maxgrp_rad

    min_n(1:3) = (/min_nx, min_ny, min_nz/)

    call chmalloc('domdec_dr_common.src','determine_direct_split','factor',&
         max(10,ndirect/2),intg=factor)

    call prime_factors(ndirect, factor, nfactor)

    n(1:3) = 1
    if (nfactor < 3) then
       n(1:nfactor) = factor(1:nfactor)
       if (ndirect > 10) then
          write (outu,'(a,3i5)') 'Calculated NDIR=',n(1:3)
          call wrndie(-5,'<domdec_dr_common>',&
               'Number of direct nodes is a large prime, try setting NDIR manually')
       endif
    else
       call find_best_n(nfactor, factor, min_n, boxx, boxy, boxz, cutgrp, n)
       if (product(n(1:3)) == 0) then
          if (prnlev > 2) write (outu,'(a,i4,a)') 'Must have 1 or greater than ',min_n(1),' sub-boxes in X direction'
          if (prnlev > 2) write (outu,'(a,i4,a)') 'Must have 1 or greater than ',min_n(2),' sub-boxes in Y direction'
          if (prnlev > 2) write (outu,'(a,i4,a)') 'Must have 1 or greater than ',min_n(3),' sub-boxes in Z direction'
          call wrndie(-5,'<domdec_dr_common>',&
               'determine_direct_split: Unable to find appropriate sub-box counts, see above')
       endif
    endif

    call chmdealloc('domdec_dr_common.src','determine_direct_split','factor',&
         max(10,ndirect/2),intg=factor)

    if (product(n(1:3)) /= ndirect) then
       write (outu,'(a,3i4)') 'n(1:3)=',n(1:3)
       call wrndie(-5,'<domdec_dr_common>',&
            'determine_direct_split: Fatal error setting nx,ny,nz')
    endif

    nx = n(1)
    ny = n(2)
    nz = n(3)

    return
  end subroutine determine_direct_split

  ! *
  ! * Loop through all the possible choices:
  ! * Choose the one that fits the limitations (n == 1 .or. n >= min_n)
  ! *
  subroutine find_best_n(nfactor, factor, min_n, boxx, boxy, boxz, rcut, n)
    use number,only:zero
    use consta,only:pi
    use domdec_common,only:q_split
#if KEY_DOMDEC_GPU==1
    use domdec_common,only:q_gpu
#endif
    use pmeutil,only:nfft1, nfft2
    use colfft_util,only:calc_column_num
    implicit none
    ! Input / Output
    integer, intent(in) :: nfactor, factor(nfactor), min_n(3)
    real(chm_real), intent(in) :: boxx, boxy, boxz, rcut
    integer, intent(out) :: n(3)
    ! Variables
    real(chm_real) vol, fac, min_fac
    integer i, j, k, ii, jj, kk, jstart
    integer best_n(3), nt(3)
    integer ny_recip, nz_recip

    vol = boxx*boxy*boxz

    if (vol == zero .or. rcut == zero) then
       call wrndie(-5,'<domdec_dr_common>','find_best_n: volume of box or radial cut-off are zero!')
    endif

    ny_recip = 0
    nz_recip = 0
    if (q_split .and. nrecip > 0) then
       call calc_column_num(nrecip, nfft1, nfft2, ny_recip, nz_recip)
    endif

    min_fac = 1000000000.0
    best_n(1:3) = 0
    ! Loop through all splits
    do i=1,nfactor-2
       if (i == 1) then
          jstart = 0
       else
          jstart = 1
       endif
       do j=jstart,nfactor-1-i
          k = nfactor-i-j
          nt(1) = product(factor(1:i))
          if (j == 0) then
             nt(2) = nt(1)
          else
             nt(2) = product(factor(i+1:i+j))
          endif
          nt(3) = product(factor(i+j+1:i+j+k))
          ! Loop through all orderings of nt(1:3) (there are six of them)
          do ii=1,3
             do jj=1,3
                if (jj == ii) cycle
                do kk=1,3
                   if (kk == ii .or. kk == jj) cycle
                   n(1) = nt(ii)
                   n(2) = nt(jj)
                   n(3) = nt(kk)
                   if (n(1) > 1 .and. n(1) < min_n(1)) cycle
                   if (n(2) > 1 .and. n(2) < min_n(2)) cycle
                   if (n(3) > 1 .and. n(3) < min_n(3)) cycle
#if KEY_DOMDEC_GPU==1
                   if (q_gpu) then
                      ! For DOMDEC_GPU, choose the configuration that minimizes FFT transpose
                      ! communication cost
                      fac = zero
                      if (n(1) > 1) fac = fac + real(2*n(1),chm_real)/boxx
                      if (n(2) > 1) fac = fac + real(n(2),chm_real)/boxy
                      if (n(3) > 1) fac = fac + real(n(3),chm_real)/boxz
                   else
#endif
                      ! For DOMDEC, either:
                      ! 1) choose the configuration that minimizes the direct-recip and
                      !    recip-recip communication
                      !
                      ! OR
                      !
                      ! 2) choose the configuration that gives the most square box
                      ! (measured as the deviation from boxx/(nx*V^(1/3)) == 1
                      if (ny_recip == n(2) .and. nz_recip == n(3)) then
                         fac = 0.0
                      else
                         fac = 0.25*pi*rcut*rcut*(boxx/n(1) + boxy/n(2) + boxz/n(3)) + &
                              rcut*(n(1)/boxx + n(2)/boxy + n(3)/boxz)
                      endif
#if KEY_DOMDEC_GPU==1
                   endif
#endif
                   if (fac < min_fac) then
                      min_fac = fac
                      best_n(1:3) = n(1:3)
                   endif
                enddo
             enddo
          enddo
       enddo
    enddo

    n(1:3) = best_n(1:3)
    return
  end subroutine find_best_n

  ! *
  ! * Returns the prime factors of number n into factor(1:nfactor)
  ! *
  subroutine prime_factors(n, factor, nfactor)
    implicit none
    ! Input / Output
    integer n, nfactor, factor(*)
    ! Variables
    integer i, nt, f

    nfactor = 1
    factor(1) = 1

    nt = n
    f = 2

    do while (nt > 1)
       if (mod(nt,f) == 0) then
          nfactor = nfactor + 1
          factor(nfactor) = f
          nt = nt/f
       else
          f = f + 1
       endif
    enddo

    return
  end subroutine prime_factors

  ! *
  ! * Undo split direct/recip
  ! *
  subroutine merge_direct_recip()
    use mpi
    use domdec_common,only:ndirect, q_cons_node, q_split
    use parallel_groups,only:free_comm_group
    use memory,only:chmdealloc
    use parallel,only:numnod
    implicit none
    ! Variables
    integer ierror

    if (q_split) then
       call mpi_barrier(comm_charmm_save, ierror)
       if (ierror /= mpi_success) then
          call wrndie(-5,'<domdec_dr_common>','error in mpi_barrier')
       endif
       ! Free comm_direct and comm_recip communicators
       if (q_direct_node) then
          call free_comm_group(comm_direct)
       else
          call free_comm_group(comm_recip)
       endif
    endif

!    ! When no split, all nodes are direct nodes
!    comm_charmm = comm_charmm_save
!    numnod = numnod_save
!    mynod = mynod_save

    if (allocated(direct_nodes)) then
       call chmdealloc('domdec_dr_common.src','merge_direct_recip','direct_nodes',&
            size(direct_nodes),intg=direct_nodes)
    endif

    if (allocated(recip_nodes)) then
       call chmdealloc('domdec_dr_common.src','merge_direct_recip','recip_nodes',&
            size(recip_nodes),intg=recip_nodes)
    endif

    ndirect = numnod
    nrecip = 0
    q_direct_node = .true.
    q_recip_node = .false.
    q_cons_node = .false.
    q_split = .false.

    return
  end subroutine merge_direct_recip

  ! *
  ! * Split nodes to direct and recip nodes
  ! * NOTE: ndirect, nrecip, and q_split are already set
  ! *
  subroutine split_direct_recip()
    use parallel
    use mpi
    use stream
    use memory
    use parallel_groups,only:split_comm_group, create_comm_group
    use domdec_common,only:ndirect, q_cons_node, q_split
    implicit none
    ! Variables
    integer ierror
    logical print_split
    integer, save :: ndirect_prev = -1
    integer tmp

    if(ndirect < 1) then
       call wrndie(-5, '<domdec_dr_common>','Not enough nodes for nrecip requested')
    endif

    print_split = .false.

    if (ndirect_prev == -1) then
       ! First call, save comm_charmm, numnod_save, mynod_save
       comm_charmm_save = comm_charmm
       numnod_save = numnod
       mynod_save = mynod
       print_split = .true.
    else
       if (ndirect_prev /= ndirect) print_split = .true.
    endif

    call chmalloc('domdec_dr_common.src','split_direct_recip','direct_nodes',&
         ndirect,intg=direct_nodes)
    call chmalloc('domdec_dr_common.src','split_direct_recip','recip_nodes',&
         max(1,nrecip),intg=recip_nodes)

    if (q_split) then
       if (q_interleave) then
          if (print_split .and. prnlev >= 2) then
             write(outu,'(a,2i4)') '<domdec_dr_common> Interleaving direct/recip: ',&
                  ndirect,nrecip
          endif
          call interleave_direct_recip(ndirect, nrecip, direct_nodes, recip_nodes)
       else
          if (print_split .and. prnlev >= 2) then
             write(outu,'(a,2i4)') '<domdec_dr_common> Splitting into direct/recip: ',&
                  ndirect,nrecip
          endif
          call pile_direct_recip(ndirect, nrecip, direct_nodes, recip_nodes)
       endif
    else
       if (print_split .and. prnlev >= 2) then
          write (outu,'(a)') '<domdec_dr_common> No direct/recip split, using all nodes on both'
       endif
       call pile_nodes(ndirect, direct_nodes)
       call pile_nodes(nrecip, recip_nodes)
    endif

    call set_q_direct_recip(ndirect, nrecip, direct_nodes, recip_nodes, &
         q_direct_node, q_recip_node)

    if (.not.q_split) then
       ! No split
       mynod_split = mynod
       comm_direct = comm_charmm
       comm_direct_recip = comm_charmm
       comm_recip = comm_charmm
    else

       if (q_interleave) then
          ! Sets comm_direct and comm_recip according to the node numbers given in
          ! direct_nodes and recip_nodes:
          ! comm_direct = nodes in direct_nodes(1:ndirect)
          ! comm_recip  = nodes in recip_nodes(1:nrecip)
          if (q_direct_node) then
             call create_comm_group(comm_charmm, ndirect, direct_nodes, comm_direct)
          else
             call create_comm_group(comm_charmm, nrecip, recip_nodes, comm_recip)
          endif
       else
          ! Splits comm_charmm to comm_direct and comm_recip such that:
          ! mynod <  ndirect => comm_direct
          ! mynod >= ndirect => comm_recip
          if (q_direct_node) then
             call split_comm_group(comm_charmm, ndirect, comm_direct)
          else
             call split_comm_group(comm_charmm, ndirect, comm_recip)
          endif
       endif

       ! Create new communicator "comm_direct_recip" which includes all direct nodes 
       ! plus all recip nodes
       comm_direct_recip = comm_charmm

       ! Set the node ranks after the split
       if (q_direct_node) then
          call mpi_comm_rank(comm_direct, mynod_split, ierror)
       else
          call mpi_comm_rank(comm_recip, mynod_split, ierror)
       endif
       if (ierror /= mpi_success) call wrndie(-5,'<domdec_dr_common>','Error in mpi_comm_rank')

    endif

    if (q_split) then
       if (q_recip_node) then
          if (mynod_split == cons_node_split) q_cons_node = .true.
       endif
       cons_node = ndirect + cons_node_split
    else
       if (mynod == cons_node_split) q_cons_node = .true.
       cons_node = cons_node_split
    endif

    ndirect_prev = ndirect

    return
  end subroutine split_direct_recip

  ! *
  ! * Sets q_direct_node and q_recip_node
  ! *
  subroutine set_q_direct_recip(ndirect_in, nrecip_in, direct_nodes, recip_nodes, &
       q_direct_node_out, q_recip_node_out)
    use parallel,only:mynod
    use domdec_common,only:q_split
    implicit none
    ! Input / Output
    integer, intent(in) :: ndirect_in, nrecip_in
    integer, intent(in) :: direct_nodes(*), recip_nodes(*)
    logical, intent(out) :: q_direct_node_out, q_recip_node_out
    ! Variables
    integer i

    q_direct_node_out = .false.
    q_recip_node_out = .false.

    do i=1,ndirect_in
       if (direct_nodes(i) == mynod) q_direct_node_out = .true.
    enddo

    if (q_split) then
       do i=1,nrecip_in
          if (recip_nodes(i) == mynod) q_recip_node_out = .true.
       enddo
    else
       ! Split OFF, all direct nodes are also reciprocal nodes
       q_recip_node_out = .true.
    endif

    if (q_split .and. (q_direct_node_out .eqv. q_recip_node_out)) then
       call wrndie(-5,'<domdec_dr_common>','For direct/recip split, nodes cannot be both types')
    endif

    if (.not.q_split .and. (.not.q_direct_node_out .or. .not.q_recip_node_out)) then
       call wrndie(-5,'<domdec_dr_common>',&
            'For no split, all nodes must be direct+recip nodes')
    endif

    return
  end subroutine set_q_direct_recip

  subroutine pile_nodes(nnodes, nodes)
    implicit none
    ! Input / Output
    integer, intent(in) :: nnodes
    integer, intent(out) :: nodes(:)
    ! Variables
    integer i

    do i=1,nnodes
       nodes(i) = i - 1
    enddo

    return
  end subroutine pile_nodes

  ! *
  ! * Sets direct and recip nodes after another
  ! *
  subroutine pile_direct_recip(ndirect_in, nrecip_in, direct_nodes, recip_nodes)
    implicit none
    ! Input / Output
    integer, intent(in) :: ndirect_in, nrecip_in
    integer, intent(out) :: direct_nodes(*), recip_nodes(*)
    ! Variables
    integer i, idir, irec

    idir = 0
    irec = 0
    do i=1,ndirect_in + nrecip_in
       if (i <= ndirect_in) then
          idir = idir + 1
          direct_nodes(idir) = i - 1
       else
          irec = irec + 1
          recip_nodes(irec) = i - 1
       endif
    enddo

    if (irec /= nrecip_in .or. idir /= ndirect_in) then
       call wrndie(-5,'<domdec_dr_common>','Invalid number of direct / recip nodes when piling')
    endif

  end subroutine pile_direct_recip

  ! *
  ! * Interleaves direct and recip nodes
  ! *
  subroutine interleave_direct_recip(ndirect_in, nrecip_in, direct_nodes, recip_nodes)
    implicit none
    ! Input / Output
    integer, intent(in) :: ndirect_in, nrecip_in
    integer, intent(out) :: direct_nodes(*), recip_nodes(*)
    ! Variables
    integer i, idir, irec, n
    logical qdir

    qdir = .false.
    idir = 0
    irec = 0

    n = 0
    do i=1,ndirect_in + nrecip_in
       n = n + nrecip_in
       if (n >= ndirect_in + nrecip_in .and. qdir) then
          ! Recip node
          qdir = .false.
          irec = irec + 1
          recip_nodes(irec) = i - 1
          n = n - (ndirect_in + nrecip_in)
       else
          ! Direct node
          qdir = .true.
          idir = idir + 1
          direct_nodes(idir) = i - 1
       endif
    enddo

    if (irec /= nrecip_in .or. idir /= ndirect_in) then
       call wrndie(-5,'<domdec_dr_common>','Invalid number of direct / recip nodes when interleaving')
    endif

    return
  end subroutine interleave_direct_recip

  ! *
  ! * Starts direct/recip split by setting comm_charmm, numnod, and numnod
  ! *
  ! * Called in:
  ! * gete() in energy/eutil.src
  ! * dcntrl() in dynamc/dcntrl.src
  ! *
  subroutine start_split_direct_recip()
    use domdec_common,only:ndirect
    use parallel,only:comm_charmm,numnod,mynod,mynodp
    implicit none

    if (q_direct_node) then
       comm_charmm = comm_direct
       numnod = ndirect
       mynod = mynod_split
       mynodp = mynod_split+1
    else
       comm_charmm = comm_recip
       numnod = nrecip
       mynod = mynod_split
       mynodp = mynod_split+1
    endif

    return
  end subroutine start_split_direct_recip

  ! *
  ! * Stops direct/recip split by resuming original values for comm_charmm, numnod, mynod
  ! *
  subroutine stop_split_direct_recip()
    use parallel,only:comm_charmm,numnod,mynod,mynodp
    implicit none

    comm_charmm = comm_charmm_save
    numnod = numnod_save
    mynod = mynod_save
    mynodp = mynod_save+1

    return
  end subroutine stop_split_direct_recip

  ! *
  ! * Returns the recip node for direct node "direct_node"
  ! *
  integer function get_recip_node(direct_node)
    use domdec_common,only:ndirect
    implicit none
    ! Input
    integer, intent(in) :: direct_node

    get_recip_node = recip_nodes(nrecip*direct_node/ndirect + 1)

    return
  end function get_recip_node

  ! *
  ! * Copies (x, y, z) from direct root node (node 0) to all recip nodes
  ! * NOTE: Can only be called when direct/recip split is ON
  ! *
  subroutine copy_to_recip(x, y, z)
    use mpi
    use psf,only:natom
    use domdec_common,only:ndirect, q_split
    implicit none
    ! Input / Output
    real(chm_real) x(*), y(*), z(*)
    ! Variables
    integer status(MPI_STATUS_SIZE), ierror

    if (.not.q_split) then
       call wrndie(-5,'<domdec_dr_common>','copy_to_recip called when direc/recip split is OFF')
    endif
    
    ! No recip nodes => exit
    if (nrecip == 0) return

    ! first send data from direct root to recip root
    ! NOTE: recip root = recip_nodes(1)
    !       direct root = direct_nodes(1)
    if (q_direct_node .and. mynod_split == 0) then
       call mpi_send(x, natom, mpi_real8, recip_nodes(1), 1, comm_direct_recip, ierror)
       if (ierror /= mpi_success) call wrndie(-5,'<domdec_dr_common>','error in mpi_send')
       call mpi_send(y, natom, mpi_real8, recip_nodes(1), 2, comm_direct_recip, ierror)
       if (ierror /= mpi_success) call wrndie(-5,'<domdec_dr_common>','error in mpi_send')
       call mpi_send(z, natom, mpi_real8, recip_nodes(1), 3, comm_direct_recip, ierror)
       if (ierror /= mpi_success) call wrndie(-5,'<domdec_dr_common>','error in mpi_send')
       ! Direct root node exit
       return
    elseif (q_recip_node .and. mynod_split == 0) then
       call mpi_recv(x, natom, mpi_real8, direct_nodes(1), 1, comm_direct_recip, status, ierror)
       if (ierror /= mpi_success) call wrndie(-5,'<domdec_dr_common>','error in mpi_recv')
       call mpi_recv(y, natom, mpi_real8, direct_nodes(1), 2, comm_direct_recip, status, ierror)
       if (ierror /= mpi_success) call wrndie(-5,'<domdec_dr_common>','error in mpi_recv')
       call mpi_recv(z, natom, mpi_real8, direct_nodes(1), 3, comm_direct_recip, status, ierror)
       if (ierror /= mpi_success) call wrndie(-5,'<domdec_dr_common>','error in mpi_recv')
    elseif (q_direct_node) then
       ! Other direct nodes exit
       return
    endif

    ! send data from recip root to other recip nodes
    call mpi_bcast(x, natom, mpi_real8, 0, comm_recip, ierror)
    if (ierror /= mpi_success) call wrndie(-5,'<domdec_dr_common>','error in mpi_bcast')
    call mpi_bcast(y, natom, mpi_real8, 0, comm_recip, ierror)
    if (ierror /= mpi_success) call wrndie(-5,'<domdec_dr_common>','error in mpi_bcast')
    call mpi_bcast(z, natom, mpi_real8, 0, comm_recip, ierror)
    if (ierror /= mpi_success) call wrndie(-5,'<domdec_dr_common>','error in mpi_bcast')

    return
  end subroutine copy_to_recip

  ! *
  ! * If needed, allocates commbuffer
  ! * If needed, reallocates commbuffer with 50% extra space
  ! *
  subroutine alloc_realloc_commbuffer(commbuffer, req_size)
    use memory
    implicit none
    ! Input / Output
    integer(int_byte), pointer, dimension(:), intent(inout) :: commbuffer
    integer, intent(in) :: req_size
    ! Variables
    integer new_size

    new_size = req_size

    if (associated(commbuffer)) then
       if (size(commbuffer) < req_size) then
          call chmdealloc('domdec_d2r_comm.src','alloc_realloc_commbuffer','commbuffer',&
               size(commbuffer),mibyp=commbuffer)
          new_size = int(req_size*1.5)
       endif
    endif

    if (.not.associated(commbuffer)) then
       call chmalloc('domdec_d2r_comm.src','alloc_realloc_commbuffer','commbuffer',&
            new_size,mibyp=commbuffer)
    endif

    return
  end subroutine alloc_realloc_commbuffer

  ! *
  ! * De-allocates memory for this module
  ! *
  subroutine uninit_dr_common()
    use memory,only:chmdealloc
    use domdec_common,only:nthread
    use nblist_util,only:uninit_array
    implicit none
    integer i

    ! reqbuffer
    if (allocated(reqbuffer)) then
       call chmdealloc('domdec_dr_common.src','uninit_dr_common','reqbuffer',&
            size(reqbuffer),intg=reqbuffer)
    endif

    ! commnode
    if (allocated(commnode)) then
       call chmdealloc('domdec_dr_common.src','uninit_dr_common','commnode',&
            size(commnode),intg=commnode)
    endif

    ! commbuffersize
    if (allocated(commbuffersize)) then
       call chmdealloc('domdec_dr_common.src','uninit_dr_common','commbuffersize',&
            size(commbuffersize),intg=commbuffersize)
    endif

    ! commbuffer
    if (associated(commbuffer)) then
       call chmdealloc('domdec_dr_common.src','uninit_dr_common','commbuffer',&
            size(commbuffer),mibyp=commbuffer)
    endif

    ! commbufferpos
    if (allocated(commbufferpos)) then
       call chmdealloc('domdec_dr_common.src','uninit_dr_common','commbufferpos',&
            size(commbufferpos),intg=commbufferpos)
    endif

    return
  end subroutine uninit_dr_common

!#endif /* (not_cmpi)*/
#endif /* (parallel)*/
#endif /* (domdec_main)*/
end module domdec_dr_common
