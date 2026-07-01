module parallel_groups
  use chm_kinds
#if KEY_PARALLEL==1
  use mpi
  use parallel
  implicit none
  private
  public chmgrp
  integer,parameter :: max_grp_levels=4, max_groups=100, max_comms=100, &
       maxsubgrp=10

  integer current_comm_index, highest_comm_index

  integer grp_charmm, grp_world
  integer comm_world

  type chmgrp
     integer comm,grp
     integer parent_comm,parent_grp,parent_index
     integer level
     integer mynod,numnod
     integer index
     integer numsubgrp
     integer,dimension(maxsubgrp) :: subgrp
  end type chmgrp

  type(chmgrp),pointer :: current_comm
  type(chmgrp),dimension(max_groups),target :: allcomms

  ! Public variables
  public current_comm_index, allcomms

  ! Public subroutines
  public init_chm_groups, split_comm_group, free_comm_group, comm_save, create_comm_group
  
contains
  !-----------------------------------------------------------
  !            INIT_CHM_GROUPS
  !-----------------------------------------------------------
  subroutine init_chm_groups
    implicit none
    integer status

#if KEY_MULTICOM==1
    ! VO stringm : do not initialize if already initialized
    ! The idea is that init routines can be called many times (e.g. to change comms. via multicom)
    logical :: initialized_mpi
    call MPI_INITIALIZED(initialized_mpi,status)
    if (.not.initialized_mpi) & ! initialize
#endif
    ! Initialize comm_charmm to mpi_comm_world and get rank and size
    call mpi_init(status)
    current_comm => allcomms(1)
#if KEY_MULTICOM==1 /*  VO stringm */
    if (.not.initialized_mpi)  &            
#endif
    comm_charmm  = mpi_comm_world
    comm_world   = mpi_comm_world

    call mpi_comm_group(mpi_comm_world,grp_world,status)
#if KEY_MULTICOM==1 /*  VO stringm */
    if (.not.initialized_mpi) &             
#endif
    grp_charmm = grp_world
#if KEY_MULTICOM==1 /*  VO stringm */
    if (comm_charmm.ne.MPI_COMM_NULL) then
     call mpi_comm_group(comm_charmm,grp_charmm,status)
#endif
    call mpi_comm_rank(comm_charmm,mynod,status)
    call mpi_comm_size(comm_charmm,numnod,status)
#if KEY_MULTICOM==1 /*  VO stringm */
     else
      grp_charmm = MPI_GROUP_NULL
      mynod = MPI_UNDEFINED
      numnod = MPI_UNDEFINED
    endif ! comm_charmm
#endif /* VO */

    current_comm_index=1
    highest_comm_index=1

    allcomms(current_comm_index)%level       = 0
#if KEY_MULTICOM==0 /*  VO stringm */
    allcomms(current_comm_index)%comm        = mpi_comm_world
    allcomms(current_comm_index)%grp         = grp_world
#else
    allcomms(current_comm_index)%comm        = comm_charmm
    allcomms(current_comm_index)%grp         = grp_charmm
#endif
    allcomms(current_comm_index)%mynod       = mynod
    allcomms(current_comm_index)%numnod      = numnod
    allcomms(current_comm_index)%index       = current_comm_index
    allcomms(current_comm_index)%parent_comm = -1
    allcomms(current_comm_index)%parent_grp  = -1
    allcomms(current_comm_index)%parent_index= -1
    allcomms(current_comm_index)%numsubgrp   = 0
    allcomms(current_comm_index)%subgrp(1:maxsubgrp)  = 0

    comm_charmm_index=1

  end subroutine init_chm_groups

  ! *
  ! * Frees communicator group
  ! *
  subroutine free_comm_group(comm)
    use stream
    implicit none
    ! Input
    integer comm
    ! Variables
    integer ierror

    call mpi_comm_free(comm, ierror)
    if (ierror /= mpi_success) call wrndie(-5, &
         '<paralgroups>','Error in mpi_comm_free')

    return
  end subroutine free_comm_group

  ! *
  ! * Creates communicator with nodes with ranks in ranks(1:nranks)
  ! *
  subroutine create_comm_group(comm, nranks, ranks, newcomm)
    use stream
    implicit none
    ! Input / Output
    integer, intent(in) :: comm, nranks, ranks(*)
    integer, intent(out) :: newcomm
    ! Variables
    integer orig_group, new_group, ierror

    ! Get handle to comm
    call mpi_comm_group(comm, orig_group, ierror)
    if (ierror /= mpi_success) call wrndie(-5, '<paralgroups>','Error in mpi_comm_group')

    call mpi_group_incl(orig_group, nranks, ranks, new_group, ierror)
    if (ierror /= mpi_success) call wrndie(-5,'<paralgroups>','Error in mpi_group_incl')

    call mpi_comm_create(comm, new_group, newcomm, ierror)
    if (ierror /= mpi_success) call wrndie(-5,'<paralgroups>','Error in mpi_comm_create')

    return
  end subroutine create_comm_group

  ! *
  ! * Splits comm into two communicators such that
  ! * communicator 1: rank <  n0
  ! * communicator 2: rank >= n0
  ! *
  subroutine split_comm_group(comm, n0, newcomm)
    use stream
    use memory
    implicit none
    ! Input / Output
    integer, intent(in) :: comm, n0
    integer, intent(out) :: newcomm
    ! Variables
    integer, allocatable, dimension(:) :: ranks
    integer i, n, n1, rank
    integer orig_group, new_group, ierror

    ! Get total number of nodes
    call mpi_comm_size(comm, n, ierror)
    if (ierror /= mpi_success) call wrndie(-5,'<paralgroups>','Error in mpi_comm_size')

    n1 = n - n0

    ! Get node ranks
    call mpi_comm_rank(comm, rank, ierror)
    if (ierror /= mpi_success) call wrndie(-5,'<paralgroups>','Error in mpi_comm_rank')

    ! Allocate temporary buffer "ranks"
    call chmalloc('parallel_groups.src','split_comm_group','ranks',n,intg=ranks)
       
    ! Get handle to comm
    call mpi_comm_group(comm, orig_group, ierror)
    if (ierror /= mpi_success) call wrndie(-5, '<paralgroups>','Error in mpi_comm_group')

    ! Create new communicators comm0 and comm1
    if (rank < n0) then
       ranks = (/ (i,i=0,n0-1) /)
       call mpi_group_incl(orig_group, n0, ranks, new_group, ierror)
    else
       ranks = (/ (i,i=n0,n-1) /)
       call mpi_group_incl(orig_group, n1, ranks, new_group, ierror)
    endif
    if (ierror /= mpi_success) call wrndie(-5,'<paralgroups>','Error in mpi_group_incl')

    call mpi_comm_create(comm, new_group, newcomm, ierror)
    if (ierror /= mpi_success) call wrndie(-5,'<paralgroups>','Error in mpi_comm_create')

    ! deallocate ranks
    call chmdealloc('paralgroups.src','split_comm_groups','ranks',n,intg=ranks)

    return
  end subroutine split_comm_group

  !-----------------------------------------------------------
  !            comm_save
  !    Does the bookkeeping for communicator in the all_comms database
  !-----------------------------------------------------------
  subroutine comm_save(parentcomm,childcomm,childgrp,nod1,siz1,parentindex,childindex)
    !-- comm_save(parentcomm,childcomm,childgrp,parentindex,childindex)
    use mpi
    use stream,only:outu
    integer,intent(in) :: parentcomm,childcomm,childgrp,parentindex
    integer,intent(out) :: childindex
    integer :: status,nod1,siz1,nchild

    childindex=highest_comm_index+1
    highest_comm_index=childindex
!    write(outu,'(/,"Saving communicator: ",6(a,i3) )' ) &
!         "mynod ",mynod,"  parentcomm ",parentcomm,"  childcomm ",childcomm, &
!         "  childindex ",childindex,"  highest_comm_index", highest_comm_index
    if(childindex > max_comms) &
       call wrndie(-3,"paralgroups.src<Comm_Save> exceeded max_comms", &
            "Increase max_comms in paralgroups.src")

    call mpi_comm_rank(childcomm,nod1,status)
    call mpi_comm_size(childcomm,siz1,status)
    
    allcomms(childindex)%level       = allcomms(parentindex)%level + 1
    allcomms(childindex)%comm        = childcomm
    allcomms(childindex)%grp         = childgrp
    allcomms(childindex)%mynod       = nod1
    allcomms(childindex)%numnod      = siz1
    allcomms(childindex)%index       = childindex
    allcomms(childindex)%parent_comm = allcomms(parentindex)%comm
    allcomms(childindex)%parent_index= parentindex
    allcomms(childindex)%parent_grp  = allcomms(parentindex)%grp

    nchild  = allcomms(parentindex)%numsubgrp+1
    allcomms(parentindex)%numsubgrp  = nchild
    allcomms(parentindex)%subgrp(nchild)  = childindex

    return
  end subroutine comm_save

#endif 
end module parallel_groups
    

