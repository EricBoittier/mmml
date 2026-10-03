module parallel_groups
  use chm_kinds
#if KEY_PARALLEL==1
  use mpi_f08
  use parallel
  implicit none
  private
  public chmgrp
  integer,parameter :: max_grp_levels=4, max_groups=100, max_comms=100, &
       maxsubgrp=10

  integer current_comm_index, highest_comm_index

  TYPE(MPI_Group) :: grp_charmm, grp_world
  TYPE(MPI_Comm) :: comm_world

  type chmgrp
     TYPE(MPI_Comm) :: comm,parent_comm
     TYPE(MPI_Group) :: grp,parent_grp
     integer parent_index
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
    use stream, only: outu, prnlev
    implicit none
    integer status
    logical :: initialized_mpi

    ! initialized_mpi records whether MPI was already up BEFORE CHARMM ran,
    ! i.e. the host (e.g. mpi4py) initialized it.  Keep that value for the
    ! decision below; only initialize MPI here if nobody else has.
    call MPI_INITIALIZED(initialized_mpi,status)
    if (.not.initialized_mpi) then
       call mpi_init(status)
       charmm_owns_mpi = .true.   ! we initialized MPI, so we finalize it
    end if

    ! One unified base-communicator selection for every build (including
    ! stringm/MULTICOM): a host-supplied communicator wins, then the
    ! embedded per-rank-serial default, then the build-specific fallback.
    if (pycharmm_user_comm_set) then
        ! The embedding host (e.g. mpi4py) supplied a base communicator via
        ! set_charmm_comm().  Adopt it: N groups of M ranks then give an
        ! N x (M-node CHARMM) layout (numnod = M within each group).  Highest
        ! priority, so it overrides the embedded-SELF default, the
        ! CHARMM_MULTINODE escape hatch, and any MULTICOM setup.
        !
        ! Adopting means taking a private MPI_Comm_dup, not aliasing the
        ! host's handle -- see adopt_host_comm.
        call adopt_host_comm(comm_charmm)
        comm_world = comm_charmm
    else if (q_pycharmm_embedded) then
        ! pyCHARMM/embedded run: give each rank an independent single-rank
        ! CHARMM (numnod=1) on MPI_COMM_SELF and leave inter-rank
        ! communication to the host (e.g. mpi4py).  Set by init_charmm
        ! regardless of whether the host or CHARMM called MPI_Init, so it is
        ! independent of pycharmm-vs-mpi4py import order.  The CHARMM_MULTINODE
        ! escape hatch clears this flag to fall through to the default below.
        comm_charmm = MPI_COMM_SELF
        comm_world  = MPI_COMM_SELF
    else
#if KEY_MULTICOM==1
        ! stringm/MULTICOM standalone run: the string-method process grid is
        ! set up BEFORE init_chm_groups -- parstrt installs the per-replica
        ! base communicator (MPI_COMM_LOCAL) and MPI is already initialized by
        ! the time we get here (initialized_mpi is .true.).  In that case we
        ! must NOT overwrite comm_charmm: the per-replica local communicator is
        ! exactly the base each replica's CHARMM must run on.  Clobbering it to
        ! MPI_COMM_WORLD collapses all replicas into one group and deadlocks
        ! multi-replica string runs (ftsm-voro, sm0k, smcv-voro).
        !
        ! Only fall back to MPI_COMM_WORLD when CHARMM itself just initialized
        ! MPI (initialized_mpi .false.) -- i.e. there is no prior multicom
        ! split to preserve.
        !
        ! NOTE: this means the CHARMM_MULTINODE escape hatch on a MULTICOM
        ! build with a host-pre-initialized MPI leaves comm_charmm at the
        ! parstrt-set MPI_COMM_LOCAL rather than MPI_COMM_WORLD.  That path is
        ! part of the still-experimental N x M work and is not exercised by the
        ! standalone stringm suite; correctness of the shipping standalone
        ! string method takes priority here.
        if (.not.initialized_mpi) then
            comm_charmm = mpi_comm_world
            comm_world  = mpi_comm_world
        end if
#else
        ! Standalone charmm executable under mpirun, or an embedded run using
        ! the CHARMM_MULTINODE escape hatch: a normal N-node parallel run on
        ! MPI_COMM_WORLD.
        comm_charmm = mpi_comm_world
        comm_world  = mpi_comm_world
#endif
    end if
    current_comm => allcomms(1)

    ! grp_world / grp_charmm = group(s) of CHARMM's base communicator
    ! (MPI_COMM_WORLD, MPI_COMM_SELF, or the per-replica MPI_COMM_LOCAL,
    ! depending on the init path above).
#if KEY_MULTICOM==1 /*  VO stringm */
    ! On a MULTICOM build comm_charmm may be MPI_COMM_NULL on a rank that is
    ! outside the current replica's local communicator; mpi_comm_group/rank/
    ! size must not be called on it.  Guard EVERY such query -- previously the
    ! grp_world query ran unconditionally and crashed on MPI_COMM_NULL.
    if (comm_charmm.ne.MPI_COMM_NULL) then
     call mpi_comm_group(comm_charmm,grp_world,status)
     call mpi_comm_group(comm_charmm,grp_charmm,status)
     call mpi_comm_rank(comm_charmm,mynod,status)
     call mpi_comm_size(comm_charmm,numnod,status)
    else
     grp_world  = MPI_GROUP_NULL
     grp_charmm = MPI_GROUP_NULL
     mynod = MPI_UNDEFINED
     numnod = MPI_UNDEFINED
    endif ! comm_charmm
#else
    call mpi_comm_group(comm_charmm,grp_world,status)
    grp_charmm = grp_world
    call mpi_comm_rank(comm_charmm,mynod,status)
    call mpi_comm_size(comm_charmm,numnod,status)
#endif /* VO */

    current_comm_index=1
    highest_comm_index=1

    allcomms(current_comm_index)%level       = 0
    allcomms(current_comm_index)%comm        = comm_charmm
    allcomms(current_comm_index)%grp         = grp_world
    allcomms(current_comm_index)%mynod       = mynod
    allcomms(current_comm_index)%numnod      = numnod
    allcomms(current_comm_index)%index       = current_comm_index
    allcomms(current_comm_index)%parent_comm = MPI_COMM_NULL
    allcomms(current_comm_index)%parent_grp  = MPI_GROUP_NULL
    allcomms(current_comm_index)%parent_index= -1
    allcomms(current_comm_index)%numsubgrp   = 0
    allcomms(current_comm_index)%subgrp(1:maxsubgrp)  = 0

    comm_charmm_index=1

    ! Report the layout an embedded run actually ended up with.  An N x M
    ! job that quietly came up as the wrong shape (say every rank on
    ! MPI_COMM_SELF because only some ranks called set_mpi_comm) otherwise
    ! looks fine until it deadlocks or produces N copies of one replica;
    ! one line per rank makes the real shape checkable at a glance.
    if (pycharmm_user_comm_set .and. prnlev .ge. 5) then
       write(outu,'(A,I0,A,I0,A)') &
            ' PARALLEL> pyCHARMM base communicator adopted: node ', &
            mynod, ' of ', numnod, ' in this CHARMM group.'
    end if

  end subroutine init_chm_groups

  !-----------------------------------------------------------
  !            NXM_FATAL
  !-----------------------------------------------------------
  !> @brief Report an unrecoverable N x M startup error and kill the job.
  !>
  !> @param[in] msg  Explanation of what went wrong and how to fix it,
  !>                 printed to the CHARMM output unit before aborting.
  !
  ! Deliberately MPI_ABORT rather than wrndie: this runs before CHARMM's
  ! error machinery is fully up, and wrndie(-5) would unwind into
  ! stopch -> PARFIN -> MPI calls on the very communicator we just found to
  ! be unusable.  It also matters that the *job* dies: if one rank exits
  ! alone, every peer is left blocked in the collective it is already in,
  ! which is exactly the hang we are trying to remove.
  subroutine nxm_fatal(msg)
    use stream, only: outu
    implicit none
    character(len=*), intent(in) :: msg
    integer :: status, rank_in_world

    call mpi_comm_rank(mpi_comm_world, rank_in_world, status)
    if (status .ne. MPI_SUCCESS) rank_in_world = -1
    write(outu,'(A,I0,2A)') &
         ' CHARMM> FATAL (pyCHARMM NxM startup, world rank ', &
         rank_in_world, '): ', trim(msg)
    flush(outu)
    call mpi_abort(mpi_comm_world, 1, status)
  end subroutine nxm_fatal

  !-----------------------------------------------------------
  !            ADOPT_HOST_COMM
  !-----------------------------------------------------------
  !> @brief Take a private duplicate of the host-supplied base communicator.
  !>
  !> Reads the handle stored by set_charmm_comm() (module variable
  !> pycharmm_user_comm) and returns CHARMM's own duplicate of it.  Does
  !> not return on error: an unusable handle, or a duplicate that cannot
  !> complete, aborts the job through nxm_fatal.
  !>
  !> @param[out] new_comm  CHARMM's private duplicate of the host
  !>                       communicator, to be used as comm_charmm.
  !
  ! Why duplicate instead of using the host's communicator directly:
  ! CHARMM runs collectives and point-to-point with fixed tags (paral1's
  ! TAG=1) on comm_charmm.  If that is the same communication context the
  ! host script uses for its own mpi4py traffic, a CHARMM message can match
  ! a host receive (or a CHARMM collective can pair with a host collective)
  ! and the run hangs or silently corrupts data.  MPI_Comm_dup gives CHARMM
  ! a context of its own, so host and CHARMM traffic can never cross.
  !
  ! The duplicate is deliberately never freed.  MPI_Comm_free is collective,
  ! and the only place it could run is CHARMM's shutdown, which for an
  ! embedded run is a per-rank destructor with no cross-rank ordering: a
  ! rank calling the free could block forever on a peer that had already
  ! finalized MPI.  One communicator per process, reclaimed by
  ! MPI_Finalize, is much cheaper than that risk.  See PARFIN.
  !
  ! The dup is collective over the host communicator, so it completes only
  ! once every rank of that communicator has reached CHARMM init.  When a
  ! rank never gets here -- the classic N x M mistake of building an M-rank
  ! group but only having some of its ranks call into CHARMM -- a blocking
  ! dup would hang with no message.  We instead poll a non-blocking dup
  ! against a deadline and abort with an explanation naming the likely
  ! cause.  CHARMM_MPI_INIT_TIMEOUT (seconds, 0 = wait forever) overrides
  ! the default for slow or heavily loaded machines.
  subroutine adopt_host_comm(new_comm)
    implicit none
    TYPE(MPI_Comm), intent(out) :: new_comm
    TYPE(MPI_Comm) :: host_comm
    TYPE(MPI_Request) :: request
    TYPE(MPI_Errhandler) :: saved_handler
    integer :: status, probe_status, host_size
    logical :: done
    real(chm_real) :: deadline, timeout

    host_comm%MPI_VAL = pycharmm_user_comm

    ! A rank left out of the host's split (MPI_Comm_split with colour
    ! MPI_UNDEFINED) holds MPI_COMM_NULL.  Every query below would fail on
    ! it, so say so plainly instead.
    if (host_comm .eq. MPI_COMM_NULL) call nxm_fatal( &
         'set_mpi_comm() was given MPI_COMM_NULL on this rank. A rank '// &
         'excluded from the group (Split colour MPI_UNDEFINED) cannot run '// &
         'CHARMM; keep it out of the CHARMM calls entirely, or give it a '// &
         'communicator of its own (e.g. MPI_COMM_SELF).')

    ! Cheap validity probe: a stale or bogus handle fails here, at the point
    ! where it can still be attributed, rather than as a puzzling failure
    ! much later.
    !
    ! The probe only reports anything if MPI is asked to return errors
    ! rather than abort on them.  MPI cannot find a per-communicator error
    ! handler for a handle that is not valid, so it falls back to
    ! MPI_COMM_WORLD's -- which defaults to MPI_ERRORS_ARE_FATAL, killing
    ! the job inside MPI before the message below could ever print.  Switch
    ! MPI_COMM_WORLD to MPI_ERRORS_RETURN for the duration of the probe and
    ! put the caller's handler back afterwards.
    call mpi_comm_get_errhandler(mpi_comm_world, saved_handler, status)
    call mpi_comm_set_errhandler(mpi_comm_world, MPI_ERRORS_RETURN, status)
    ! probe_status is separate from status on purpose: restoring the
    ! handler overwrites status, which would discard the probe's result.
    call mpi_comm_size(host_comm, host_size, probe_status)
    call mpi_comm_set_errhandler(mpi_comm_world, saved_handler, status)
    if (probe_status .ne. MPI_SUCCESS) call nxm_fatal( &
         'the communicator handle passed to set_mpi_comm() is not valid. '// &
         'Check it was not freed (explicitly or by garbage collection) '// &
         'between set_mpi_comm() and the first CHARMM call.')

    timeout = nxm_init_timeout()
    call mpi_comm_idup(host_comm, new_comm, request, status)
    if (status .ne. MPI_SUCCESS) call nxm_fatal( &
         'MPI_Comm_idup failed on the communicator passed to set_mpi_comm().')

    if (timeout .le. 0.0) then
       call mpi_wait(request, MPI_STATUS_IGNORE, status)
    else
       deadline = mpi_wtime() + timeout
       done = .false.
       do while (.not. done)
          call mpi_test(request, done, MPI_STATUS_IGNORE, status)
          if (status .ne. MPI_SUCCESS) call nxm_fatal( &
               'MPI_Test failed while duplicating the communicator passed '// &
               'to set_mpi_comm().')
          ! Nested rather than a single .and.: Fortran does not promise
          ! short-circuit evaluation, so combining these would leave it to
          ! the compiler whether the impure mpi_wtime() is called at all.
          if (.not. done) then
             if (mpi_wtime() .gt. deadline) call nxm_fatal( &
                  'timed out duplicating the communicator passed to '// &
                  'set_mpi_comm(). This rank is waiting for the other '// &
                  'ranks of that communicator to reach CHARMM '// &
                  'initialization, and at least one never did. Every rank '// &
                  'of the communicator must call set_mpi_comm() with it '// &
                  'and then make a CHARMM call. Raise '// &
                  'CHARMM_MPI_INIT_TIMEOUT (seconds, 0 = wait forever) if '// &
                  'the run is merely slow to start.')
          end if
       end do
    end if

  end subroutine adopt_host_comm

  !> @brief Deadline for the startup collective, in seconds.
  !>
  !> Read from the environment variable CHARMM_MPI_INIT_TIMEOUT when it is
  !> set to something parsable, otherwise the 300 s default.
  !>
  !> @return Seconds to wait for the communicator duplicate to complete;
  !>         zero or less means wait indefinitely.
  function nxm_init_timeout() result(timeout)
    implicit none
    real(chm_real) :: timeout
    character(len=32) :: env
    integer :: env_len, env_stat, ios

    timeout = 300.0
    call get_environment_variable('CHARMM_MPI_INIT_TIMEOUT', env, env_len, env_stat)
    if (env_stat .eq. 0 .and. env_len .ge. 1) then
       read(env(1:env_len), *, iostat=ios) timeout
       if (ios .ne. 0) timeout = 300.0
    end if
  end function nxm_init_timeout

  ! *
  ! * Frees communicator group
  ! *
  subroutine free_comm_group(comm)
    use stream
    implicit none
    ! Input
    TYPE(MPI_Comm) :: comm
    ! Variables
    integer ierror

    call mpi_comm_free(comm, ierror)
    if (ierror /= MPI_SUCCESS) call wrndie(-5, &
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
    TYPE(MPI_Comm), intent(in) :: comm
    integer, intent(in) :: nranks, ranks(*)
    TYPE(MPI_Comm), intent(out) :: newcomm
    ! Variables
    TYPE(MPI_Group) :: orig_group, new_group
    integer ierror

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
    TYPE(MPI_Comm), intent(in) :: comm
    integer, intent(in) :: n0
    TYPE(MPI_Comm), intent(out) :: newcomm
    ! Variables
    integer, allocatable, dimension(:) :: ranks
    integer i, n, n1, rank
    TYPE(MPI_Group) :: orig_group, new_group
    integer ierror

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
    use stream,only:outu
    TYPE(MPI_Comm),intent(in) :: parentcomm,childcomm
    TYPE(MPI_Group),intent(in) :: childgrp
    integer,intent(in) :: parentindex
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
    

