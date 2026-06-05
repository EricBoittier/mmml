!> routines to configure and run molecular dynamics
module api_dynamics
  implicit none

#if KEY_LIBRARY == 1

contains

  !> @brief run the dynamics
  !
  ! the out_* arrays should be already allocated to size nrows * ncols
  ! each string of out_names should be already allocated to some max size
  !
  !> param[in] options data structure holding dynamics settings
  !> param[in] in_vx x-component of initial velocity for each atom 1:natom
  !> param[in] in_vy z-component of initial velocity for each atom 1:natom
  !> param[in] in_vz z-component of initial velocity for each atom 1:natom
  !> param[out] out_vx x-component of initial velocity for each atom 1:natom
  !> param[out] out_vy z-component of initial velocity for each atom 1:natom
  !> param[out] out_vz z-component of initial velocity for each atom 1:natom
  integer(c_int) function dynamics_run(options, &
       in_vx, in_vy, in_vz, &
       out_vx, out_vy, out_vz) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use dcntrl_mod, only: dynopt
    use api_types, only: dynamics_settings

    implicit none

    type(dynamics_settings), intent(in) :: options

    real(c_double), dimension(:), optional :: &
         in_vx, in_vy, in_vz, &
         out_vx, out_vy, out_vz

    dynamics_run = 0

    if (present(in_vx)) then
       call dynopt('', 0, options, &
            in_vx, in_vy, in_vz, &
            out_vz, out_vy, out_vz)
    else
       call dynopt('', 0, options)
    end if

    dynamics_run = 1
    call dynamics_reset()
  end function dynamics_run

  !> @brief use Langevin dynamics
  !
  !> @return integer(c_int) 1 if Langevin dynamics was being used
  integer(c_int) function dynamics_use_lang() bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use contrl, only: ilang
    implicit none
    dynamics_use_lang = ilang
    ilang = 1
  end function dynamics_use_lang

  !> @brief change number of steps to be taken in each dynamics run
  !
  !> @param[in] new_nstep the new number of dynamics steps desired
  !> @return the old setting for nstep
  integer(c_int) function dynamics_set_nstep(new_nstep) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use reawri, only: nstep

    implicit none

    integer(c_int), intent(in) :: new_nstep

    dynamics_set_nstep = nstep
    nstep = new_nstep
  end function dynamics_set_nstep

  !> @brief return the number of steps to be taken in each dynamics run
  !
  ! returns the number of dynamics steps which is equal to
  ! the number of energy evaluations
  !
  !> @return integer current number of steps to be taken
  integer(c_int) function dynamics_get_nstep() bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use reawri, only: nstep

    implicit none

    dynamics_get_nstep = nstep
  end function dynamics_get_nstep

  !> @brief set the time step in picoseconds for dynamics
  !
  ! the default is 0.001 picoseconds
  !
  !> @param[in] new_timest new time step in picoseconds
  !> @return real old time step value in picoseconds
  real(c_double) function dynamics_set_timest(new_timest) bind(c)
    use, intrinsic :: iso_c_binding, only: c_double
    use reawri, only: timest

    implicit none

    real(c_double), intent(in) :: new_timest

    dynamics_set_timest = timest
    timest = new_timest
  end function dynamics_set_timest

  !> @brief set the time step for dynamics in AKMA units
  !
  !> @param[in] new_akmast new time step in AKMA units
  !> @return real old time step value in AKMA units
  real(c_double) function dynamics_set_akmast(new_akmast) bind(c)
    use, intrinsic :: iso_c_binding, only: c_double
    use reawri, only: timest, delta
    use consta, only: timfac
    use number, only: zero

    implicit none

    real(c_double), intent(in) :: new_akmast

    dynamics_set_akmast = timest / timfac

    delta = new_akmast
    timest = timfac * delta

    if (delta <= zero) &
         call wrndie(-3, '<dynamics_set_akmast>', 'time step zero or negative')

  end function dynamics_set_akmast

  !> @brief change the freq of regenerating the nonbonded list for dynamics runs
  !
  ! the list is regenerated if the current step number
  ! modulo INBFRQ is zero and if INBFRQ is non-zero.
  ! Specifying zero prevents the non-bonded list from being
  ! regenerated at all.
  ! INBFRQ = -1 --> all lists are updated when necessary
  ! (heuristic test).
  !
  !> @param[in] new_inbfrq the new freq for nonbonded list regeneration
  !> @return integer the old inbfrq
  integer(c_int) function dynamics_set_inbfrq(new_inbfrq) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use contrl, only: inbfrq

    implicit none

    integer(c_int), intent(in) :: new_inbfrq

    dynamics_set_inbfrq = inbfrq
    inbfrq = new_inbfrq
  end function dynamics_set_inbfrq

  !> @brief change the freq of regenerating the hydrogen bond list
  !
  ! analogous to set_inbfrq
  !
  !> @param[in] new_ihbfrq new freq for hydrogen list regeneration
  !> @return integer the old ihbfrq
  integer(c_int) function dynamics_set_ihbfrq(new_ihbfrq) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use hbondm, only: ihbfrq

    implicit none

    integer(c_int), intent(in) :: new_ihbfrq

    dynamics_set_ihbfrq = ihbfrq
    ihbfrq = new_ihbfrq
  end function dynamics_set_ihbfrq

  !> @brief set the initial temperature for dynamics runs
  !
  ! Set the initial temperature at which the velocities have to be
  ! assigned to begin the dynamics run. Important only
  ! for the initial stage of a dynamics run.
  !
  !> @param[in] new_firstt new initial temperature desired
  !> @return real old initial temp
  real(c_double) function dynamics_set_firstt(new_firstt) bind(c)
    use, intrinsic :: iso_c_binding, only: c_double
    use reawri, only: firstt

    implicit none

    real(c_double), intent(in) :: new_firstt

    dynamics_set_firstt = firstt
    firstt = new_firstt
  end function dynamics_set_firstt

  !> @brief change step freq for printing and storing energy data for dynamics runs
  !
  !> @param[in] new_nprint new step frequency desired
  !> @return integer old step frequency
  integer(c_int) function dynamics_set_nprint(new_nprint) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use contrl, only: nprint

    implicit none

    integer(c_int), intent(in) :: new_nprint

    dynamics_set_nprint = nprint
    nprint = new_nprint
  end function dynamics_set_nprint

  !> @brief return step freq for printing and storing energy data for dynamics runs
  !
  !> @return integer step freq for printing and storing energy data for dynamics runs
  integer(c_int) function dynamics_get_nprint() bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use contrl, only: nprint

    implicit none

    dynamics_get_nprint = nprint
  end function dynamics_get_nprint

  !> @brief change freq for checking whether an atom is in Langevin region defined by RBUF
  !
  !> @param[in] new_ilbfrq new frequency desired
  !> @return integer old frequency
  integer(c_int) function dynamics_set_ilbfrq(new_ilbfrq) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use contrl, only: ilbfrq
    implicit none
    integer(c_int), intent(in) :: new_ilbfrq
    dynamics_set_ilbfrq = ilbfrq
    ilbfrq = new_ilbfrq
  end function dynamics_set_ilbfrq

  !> @brief change final equilib temperature for system
  !
  !> @param[in] new_finalt new temperature desired
  !> @return real(c_double) old temperature
  real(c_double) function dynamics_set_finalt(new_finalt) bind(c)
    use, intrinsic :: iso_c_binding, only: c_double
    use reawri, only: finalt
    implicit none
    real(c_double), intent(in) :: new_finalt
    dynamics_set_finalt = finalt
    finalt = new_finalt
  end function dynamics_set_finalt

  !> @brief change temperature increment every IHTFRQ steps for heating stage
  !
  !> @param[in] new_teminc new frequency desired
  !> @return real(c_double) old frequency
  real(c_double) function dynamics_set_teminc(new_teminc) bind(c)
    use, intrinsic :: iso_c_binding, only: c_double
    use reawri, only: teminc
    implicit none
    real(c_double), intent(in) :: new_teminc
    dynamics_set_teminc = teminc
    teminc = new_teminc
  end function dynamics_set_teminc

  !> @brief change temperature at which the starting structure has been equilibrated
  !
  ! Used to assign velocities so that equal
  ! partition of energy will yield the correct equilibrated
  ! temperature.  -999. is a default which causes the
  ! program to assign velocities at T=1.25*FIRSTT.
  !
  !> @param[in] new_tstruc new temperature desired
  !> @return real(c_double) old temperature
  real(c_double) function dynamics_set_tstruc(new_tstruc) bind(c)
    use, intrinsic :: iso_c_binding, only: c_double
    use reawri, only: tstruc
    implicit none
    real(c_double), intent(in) :: new_tstruc
    dynamics_set_tstruc = tstruc
    tstruc = new_tstruc
  end function dynamics_set_tstruc

  !> @brief return the number of integers required to seed the random number generator
  !
  !> @return integer number of integers required to seed the random number generator
  integer(c_int) function dynamics_get_nrand() bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use rndnum, only: nrand
    implicit none
    dynamics_get_nrand = nrand
  end function dynamics_get_nrand

  !> @brief set seed for the random number generator
  !
  ! The seed for the random number generator used for
  ! assigning velocities. If not specified a value based on
  ! the system clock is used; this is the recommended mode, since
  ! it makes each run unique.
  ! One integer, or as many as required by the random number
  ! generator, may be specified. See * note Hbonds: (doc/random.doc)
  !
  !> @params[in] new_rngseeds new seed for the random number generator
  !> @return integer success == 1, error == 0
  integer(c_int) function dynamics_set_rngseeds(new_rngseeds) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use rndnum, only: nrand, rngseeds, qapi_seed_set
    implicit none
    integer(c_int), intent(in) :: new_rngseeds(nrand)
    dynamics_set_rngseeds = 0
    rngseeds(1:nrand) = new_rngseeds(1:nrand)
    qapi_seed_set = .true.
    dynamics_set_rngseeds = 1
  end function dynamics_set_rngseeds

  ! If you are looking for the following settings, they are set by the
  ! options argument to the dynamics_run subroutine
  !
  ! IEQFRQ      0     The step frequency for assigning or scaling velocities to
  !                   FINALT temperature during the equilibration stage of the
  !                   dynamics run.
  !
  ! NTRFRQ      0     The step frequency for stopping the rotation and translation
  !                   of the molecule during dynamics. This operation is done
  !                   automatically after any heating.
  !
  ! ICHECW      1     The option for checking to see if the average temperature
  !                   of the system lies within the allotted temperature window
  !                   (between FINALT+TWINDH and FINALT+TWINDL) every
  !                   IEQFRQ steps.
  !                   .eq. 0 - do not check
  !                            i.e. assign or scale velocities.
  !                   .ne. 0 - check window
  !                            i.e. assign or scale velocities only if average
  !                                 temperature lies outside the window.

  ! TODO
  ! iuncrd 0 iunvel 0 kunit 0 ! unit numbers for writing files

  !> @brief use starting velocities described by iasvel
  !
  !> @return integer(c_int) 1 if start was already on
  function dynamics_use_start() bind(c) result(old_start)
    use, intrinsic :: iso_c_binding, only: c_int
    use contrl, only: irest
    implicit none
    integer(c_int) :: old_start
    old_start = 0
    if (irest == 0) old_start = 1
    irest = 0
  end function dynamics_use_start

  !> @brief use starting velocities described by iasvel
  !
  !> @return integer(c_int) 1 if start was already on
  function dynamics_use_restart() bind(c) result(old_restart)
    use, intrinsic :: iso_c_binding, only: c_int
    use contrl, only: irest
    implicit none
    integer(c_int) :: old_restart
    old_restart = 0
    if (irest == 1) old_restart = 1
    irest = 1
  end function dynamics_use_restart

  !> @brief set twindh high temperature tolerance for equilibration
  !
  !> @return old_twindh old value for twindh
  function dynamics_set_twindh(new_twindh) bind(c) result(old_twindh)
    use, intrinsic :: iso_c_binding, only: c_double
    use reawri, only: twindh
    implicit none
    real(c_double) :: new_twindh, old_twindh
    old_twindh = twindh
    twindh = new_twindh
    if (twindh < 0.0) twindh = -twindh
  end function dynamics_set_twindh

  !> @brief set twindl low temperature tolerance for equilibration
  !
  !> @return old_twindl old value for twindl
  function dynamics_set_twindl(new_twindl) bind(c) result(old_twindl)
    use, intrinsic :: iso_c_binding, only: c_double
    use reawri, only: twindl
    implicit none
    real(c_double) :: new_twindl, old_twindl
    old_twindl = twindl
    twindl = new_twindl
    if (twindl > 0.0) twindl = -twindl
  end function dynamics_set_twindl

  !> @brief set echeck: total energy change tol for each step
  !
  !> @return old_echeck old value for echeck
  function dynamics_set_echeck(new_echeck) bind(c) result(old_echeck)
    use, intrinsic :: iso_c_binding, only: c_double
    use reawri, only: echeck
    implicit none
    real(c_double) :: new_echeck, old_echeck
    old_echeck = echeck
    echeck = new_echeck
  end function dynamics_set_echeck

  !> @brief set nsavc: freq for saving coords to file
  !
  !> @return old_nsavc old value for nsavc
  function dynamics_set_nsavc(new_nsavc) bind(c) result(old_nsavc)
    use, intrinsic :: iso_c_binding, only: c_int
    use reawri, only: nsavc
    implicit none
    integer(c_int) :: new_nsavc, old_nsavc
    old_nsavc = nsavc
    nsavc = new_nsavc
  end function dynamics_set_nsavc

  !> @brief set nsavv: freq for saving velocities to file
  !
  !> @return old_nsavv old value for nsavv
  function dynamics_set_nsavv(new_nsavv) bind(c) result(old_nsavv)
    use, intrinsic :: iso_c_binding, only: c_int
    use reawri, only: nsavv
    implicit none
    integer(c_int) :: new_nsavv, old_nsavv
    old_nsavv = nsavv
    nsavv = new_nsavv
  end function dynamics_set_nsavv

  !> @brief get the friction coefs for the atoms
  !
  !> @param[out] out_fbetas vector of friction coefs for the atoms
  !> @return success
  !>         1 <=> success
  function dynamics_get_fbetas(out_fbetas) bind(c) result(success)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use api_util, only: f2c_logical
    use chm_kinds, only: chm_real
    use cnst_fcm, only: allocate_cnst, fbeta
    use psf, only: natom

    implicit none

    ! args
    real(c_double) :: out_fbetas(*)

    ! locals
    logical :: qsuccess
    real(chm_real) :: work(natom)

    ! result
    integer(c_int) :: success

    qsuccess = .false.
    call allocate_cnst(natom)
    out_fbetas(1:natom) = fbeta(1:natom)
    qsuccess = .true.
    success = f2c_logical(qsuccess)
  end function dynamics_get_fbetas

  !> @brief get the friction coefs for the atoms
  !
  !> @param[in] in_fbetas vector of friction coefs for the atoms
  !> @return success
  !>         1 <=> success
  function dynamics_set_fbetas(in_fbetas) bind(c) result(success)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use api_util, only: f2c_logical
    use chm_kinds, only: chm_real
    use cnst_fcm, only: allocate_cnst, fbeta
    use psf, only: natom

    implicit none

    ! args
    real(c_double) :: in_fbetas(*)

    ! locals
    logical :: qsuccess
    integer :: i

    ! result
    integer(c_int) :: success

    qsuccess = .false.
    call allocate_cnst(natom)
    fbeta(1:natom) = in_fbetas(1:natom)
    qsuccess = .true.
    success = f2c_logical(qsuccess)
  end function dynamics_set_fbetas

  !> @brief print the main coordinate set of the atoms
  function dynamics_open_file(filename, len_fn, read_only, formatted, &
       new_unit) result(success)
    use, intrinsic :: iso_c_binding, only: c_int, c_char
    use api_util, only: c2f_string
    use exfunc, only: lunass
    use machio, only: vopen

    implicit none

    ! args
    character(len=1, kind=c_char) :: filename(*)
    integer(c_int) :: len_fn
    logical :: read_only, formatted
    integer :: new_unit

    ! result
    logical :: success

    ! local vars
    character(len=len_fn) :: fn
    character(len=16) :: access_spec, format_spec
    logical :: qerr

    success = .false.

    if (new_unit .ne. -1) then
       call wrndie(-1, &
            '<api_dynamics%open_file>', &
            'previously file may be open')
    end if

    fn = c2f_string(filename, len_fn)
    new_unit = lunass(90)

    access_spec = 'WRITE'
    if (read_only) access_spec = 'READ'

    format_spec = 'UNFORMATTED'
    if (formatted) format_spec = 'FORMATTED'

    call vopen(new_unit, fn, format_spec, access_spec, qerr, 0)

    if (qerr) then
       call wrndie(-5, &
            '<api_dynamics%open_file>', &
            'cannot open file ' // fn(1:len_fn))
       return
    end if
    success = .true.
  end function dynamics_open_file

  function dynamics_close_file(file_unit) result(success)
    implicit none
    integer :: file_unit
    logical :: success, qerr

    success = .false.

    if (file_unit .eq. -1) then
       success = .true.
       return
    end if

    call vclose(file_unit, 'KEEP', qerr)
    if (qerr) then
       call wrndie(-5, &
            '<api_write%write_coor_pdb>', &
            'cannot close file')
       return
    end if

    file_unit = -1
    success = .true.
  end function dynamics_close_file

  !> @brief some dyn options require resetting before/after each run
  subroutine dynamics_reset()
    use contrl, only: irest, ilang
    use reawri, only: iunrea, iunwri, iuncrd, iunvel, kunit

    implicit none

    logical :: success

    success = .false.

    ilang = 0
    irest = 0

    success = dynamics_close_file(iunrea)
    success = dynamics_close_file(iunwri)
    success = dynamics_close_file(iuncrd)
    success = dynamics_close_file(iunvel)
    success = dynamics_close_file(kunit)
  end subroutine dynamics_reset

  !> @brief open dynamics restart file for writing for the next run
  function dynamics_set_iunwri(filename, len_fn) bind(c) result(success)
    use, intrinsic :: iso_c_binding, only: c_int, c_char
    use reawri, only: iunwri

    implicit none

    ! args
    character(len=1, kind=c_char) :: filename(*)
    integer(c_int) :: len_fn

    ! result
    integer(c_int) :: success

    ! local vars
    logical :: qsuccess

    success = -1

    qsuccess = dynamics_open_file(filename, len_fn, &
         read_only=.false., formatted=.true., new_unit=iunwri)
    if (qsuccess) success = 1
  end function dynamics_set_iunwri

  !> @brief open file for writing coordinates for the next run
  function dynamics_set_iuncrd(filename, len_fn) bind(c) result(success)
    use, intrinsic :: iso_c_binding, only: c_int, c_char
    use reawri, only: iuncrd

    implicit none

    ! args
    character(len=1, kind=c_char) :: filename(*)
    integer(c_int) :: len_fn

    ! result
    integer(c_int) :: success

    ! local vars
    logical :: qsuccess

    success = -1

    qsuccess = dynamics_open_file(filename, len_fn, &
         read_only=.false., formatted=.false., new_unit=iuncrd)
    if (qsuccess) success = 1
  end function dynamics_set_iuncrd

  !> @brief print the main coordinate set of the atoms
  function dynamics_set_iunrea(filename, len_fn) bind(c) result(success)
    use, intrinsic :: iso_c_binding, only: c_int, c_char
    use reawri, only: iunrea

    implicit none

    ! args
    character(len=1, kind=c_char) :: filename(*)
    integer(c_int) :: len_fn

    ! result
    integer(c_int) :: success

    ! local vars
    logical :: qsuccess

    success = -1

    qsuccess = dynamics_open_file(filename, len_fn, &
         read_only=.true., formatted=.true., new_unit=iunrea)
    if (qsuccess) success = 1
  end function dynamics_set_iunrea
#endif /* KEY_LIBRARY */
end module api_dynamics
