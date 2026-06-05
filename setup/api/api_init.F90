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
    use dimens_fcm, only: set_dimens
    use intcor_module, only: initialize_icr_structs
    use machutil, only: initialize_timers
    use new_timer, only: init_timers, timer_start, t_total
    use param_store, only: param_store_init, set_param
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

end module api_init
