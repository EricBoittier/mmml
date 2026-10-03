!> routines for initializing the charmm library before reading input
module api_init
  use, intrinsic :: iso_c_binding
  implicit none
contains

  !> @brief initialize data structures before user reads input into charmm
  !
  !> @return error code, success == 1
  integer(c_int) function init_charmm() bind(c)

    use bases_fcm, only: bnbnd
    use comand, only: comlen
    use ctitla, only: ntitla
    use cmdpar, only: cmdpar_init
    use dimens_fcm, only: set_chsize, set_dimens
    use intcor_module, only: initialize_icr_structs
    use machutil, only: initialize_timers
    use new_timer, only: init_timers, timer_start, t_total
    use param_store, only: param_store_init, set_param
    use parallel, only: q_pycharmm_embedded
#if KEY_CFF==1
    use rtf, only: ucase
#endif
    use startup, only: &
         argumt, &
         startup_machine_dependent_code
    use stream, only: &
         bomlev, iolev, prnlev, wrnlev, &
         istrm, jstrm, nstrm, &
         outu, poutu, &
         lower
    use timerm, only: altlen
    use usermod, only: usrini
    use vangle_mm, only: ptrini

    implicit none

    ! local vars
    integer :: istart, j

    logical :: &
         eof, error, &
         lused, ok, &
         wantqu, &
         qrdcmd, get_next_cmd

    character(len=4) :: winit

    ! for the CHARMM_MULTINODE escape hatch
    character(len=8) :: mn_env
    integer :: mn_len, mn_stat

    ! Set I/O units and zero mscpar totals; default to lower case file names
    outu = poutu
    prnlev = 5
    lower = .true.
    bomlev = 0
    iolev = 1
    wrnlev = 5

    call param_store_init()

    call set_param('BOMLEV',bomlev)
    call set_param('WRNLEV',wrnlev)
    call set_param('PRNLEV',prnlev)
    call set_param('IOLEV',iolev)

#if KEY_CFF==1
    ucase = .true.
#endif

    !     Mark this as an embedded (pyCHARMM) run so the parallel startup
    !     gives each rank an independent single-rank CHARMM (MPI_COMM_SELF)
    !     rather than joining MPI_COMM_WORLD.  Must be set before parstrt,
    !     which runs inside Startup_machine_dependent_code below.  The
    !     standalone charmm executable does not call init_charmm, so it is
    !     unaffected and keeps normal MPI_COMM_WORLD parallel behavior.
    !
    !     Escape hatch: setting the environment variable CHARMM_MULTINODE to
    !     a truthy value (1/y/t, any case) opts back into a genuine
    !     multi-node parallel CHARMM (MPI_COMM_WORLD, numnod=N) for scripts
    !     that issue identical CHARMM commands on every rank and want CHARMM
    !     itself to decompose the work.
    q_pycharmm_embedded = .true.
    call get_environment_variable('CHARMM_MULTINODE', mn_env, mn_len, mn_stat)
    if (mn_stat == 0 .and. mn_len >= 1) then
       if (index('1yYtT', mn_env(1:1)) > 0) q_pycharmm_embedded = .false.
    end if

    !     Start times and do machine specific startup.
    call Startup_machine_dependent_code !used to be called from jobini
    call Initialize_timers   ! used to be jobini

    !     Get the CHARMM command line arguments and initialize for different
    !     platforms different variables. Check if CHARMM should communicate
    !     with QUANTA or not
    call cmdpar_init()
    call argumt(wantqu)

    call set_dimens()

    call init_timers()
    call timer_start(T_total)

    !     Open the input file and read title for run
    !     attention: 'call header' have already been done by now
    nstrm = 1
    istrm = 5
    jstrm(nstrm) = istrm
    eof = .false.
    ntitla = 0

    call initialize_icr_structs()
#if KEY_TSM==1
    call tsminit(.false.)
#endif
    call iniall()

    !     Initialize local variables
    call getpref()
    comlen = 0
    altlen = 0
    istart = 1

    call allocate_all()

    !     Initialize the rest, simply because I cannot figure out right
    !     now whether the stuff in gtnbct needs the allocation first.
    !     Eventually the following lines will probably move into  iniall
    !     above.
    j = 4
    winit = 'INIT'
    call gtnbct(winit,j,bnbnd)

    ! call user defined startup routine -- mfc pulled this from iniall
    call usrini()

    !     Initialize pointers
    call ptrini()
    init_charmm = 1
  end function init_charmm

  subroutine del_charmm() bind(c)
    call stopch('NORMAL STOP')
  end subroutine del_charmm

  !> @brief Give CHARMM the base MPI communicator to run on (embedded use).
  !
  ! Must be called BEFORE init_charmm().  comm_handle is a Fortran MPI
  ! communicator handle (e.g. mpi4py's comm.py2f()).  CHARMM adopts it as
  ! its base communicator, so N mpi4py groups of M ranks give an
  ! N x (M-node CHARMM) layout.  This just stores the handle; the
  ! communicator is adopted later in init_chm_groups.
  !
  ! Any value is treated as a valid handle -- we must NOT reserve negative
  ! values as a "clear" sentinel, because a legitimate Fortran MPI handle
  ! can be negative (e.g. an MPICH handle whose value exceeds 2^31 arrives
  ! here as a negative c_int).  Callers that want the default behaviour
  ! simply never call this routine (the pyCHARMM loader omits the call when
  ! no communicator was chosen).
  subroutine set_charmm_comm(comm_handle) bind(c)
    use parallel, only: pycharmm_user_comm, pycharmm_user_comm_set
    integer(c_int), value :: comm_handle
    pycharmm_user_comm = comm_handle
    pycharmm_user_comm_set = .true.
  end subroutine set_charmm_comm

end module api_init
