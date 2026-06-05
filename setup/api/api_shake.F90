!> routines for applying shake constraints on atoms
module api_shake

  implicit none

contains

  !> @brief deactivate shake
  !
  !> @return shake_off
  !>         1 <=> success
  function shake_off() bind(c) result(success)
    use, intrinsic :: iso_c_binding, only: c_int
    use api_util, only: f2c_logical
    use new_timer, only: timer_stop, t_shakesetup
    use psf, only: natom
    use shake, only: nconst, qshake, qfshake, deallocate_shake
    use stream, only: prnlev, outu
    implicit none
    ! locals
    logical :: qsuccess
    integer(c_int) :: success

    qsuccess = .false.

    qshake = .false.
    qfshake = .false.
    nconst = 0

    if (prnlev >= 2) then
       write(outu, '(a)') ' api_shake> SHAKE constraints removed.'
    end if

    call timer_stop(t_shakesetup)
    call deallocate_shake(natom)

    qsuccess = .true.
    success = f2c_logical(qsuccess)
  end function shake_off

  !> @brief set up shake
  !
  !> @param[in] islct islct(i) & jslct(j) .eq. 1 <=> constrain bonds/angles between atoms i & j
  !> @param[in] jslct see islct
  !> @param[in] icomp constrain the comparison set instead of the main set if icomp == 1
  !> @param[in] iparam use bond dists from param table instead of current coords if iparam == 1
  !> @param[in] ifast use vector parallel algorithm with diff assumptions if ifast == 1
  !> @param[in] iwater index of water residue for the shake fast algorithm
  !> @param[in] tol the allowed relative deviations from the reference values
  !> @param[in] maxiter maximum number of iterations SHAKE tries before giving up
  !> @param[in] scale convergence scale factor for SHAKEA
  !> @param[in] ibonh all bonds involving hydrogens are fixed if ibonh == 1
  !> @param[in] ibond all bonds are fixed if ibond == 1
  !> @param[in] iangh all angles involving hydrogen are fixed if iangh == 1
  !> @param[in] iangl all angles are fixed if iangl == 1
  !> @param[in] inoreset do not reset counters to 0 during setup if inoreset == 1
  !> @return success 1 <=> no error
  function shake_on(islct, jslct, &
       icomp, iparam, &
       ifast, iwater, &
       tol, maxiter, scale, &
       ibonh, ibond, iangh, iangl, &
       inoreset) bind(c) result(success)
    use, intrinsic :: iso_c_binding, only: c_char, c_double, c_int
    use api_util, only: c2f_logical, f2c_logical, c2f_string
    use new_timer, only: timer_start, timer_stop, t_shakesetup
    use psf, only: natom, nres, res
    use shake, only: &
         nconst, qshake, qfshake, &
         allocate_shake, &
         mxiter, shktol, shksca
    use stream, only: prnlev, outu

#if KEY_FSSHK==1
    use fstshk,only: fsrscshk
#endif

    implicit none

    ! args
    integer(c_int) :: islct(*), jslct(*), &
         icomp, ifast, iwater, iparam, &
         ibonh, ibond, iangh, iangl, &
         inoreset, maxiter
    real(c_double) :: tol, scale

    ! locals
    logical :: qcomp, qfast, qparam, &
         qbonh, qbond, qangh, qangl, &
         qnoreset, qsuccess

    integer :: iconb, ires
    logical :: qfshk
    character(len=8) :: renwat

    ! result
    integer(c_int) :: success

    success = -1
    qsuccess = .false.

    call timer_start(T_shakesetup)

#if KEY_FSSHK==1
    ! free memory used by fsshake (if allocated)
    IF (QSHAKE .AND. QFSHAKE) CALL FSRSCSHK
#endif

    ! Initialise some counters.
    ICONB  = 0

    call allocate_shake(natom)

    qcomp = c2f_logical(icomp)
    qfast = c2f_logical(ifast)
    qparam = c2f_logical(iparam)
    qbonh = c2f_logical(ibonh)
    qbond = c2f_logical(ibond)
    qangh = c2f_logical(iangh)
    qangl = c2f_logical(iangl)
    qnoreset = c2f_logical(inoreset)

    if (qbonh) iconb = 1
    if (qbond) iconb = 2
    if (qangh) iconb = 3
    if (qangl) iconb = 4

    ! Update the shake tolerance parameters.
    mxiter = maxiter
    shktol = tol
    shksca = scale

    IF(PRNLEV >= 2) WRITE (OUTU,'(A,D12.4,A,I6)') &
         ' api_shake> SHAKE parameters: TOL = ', SHKTOL, &
         ' MXITer = ', MXITER

    qfshk = qfast

    if (iwater .le. 0) then
       renwat = 'TIP3'  ! default water resn='TIP3'
    else if (0 .lt. iwater .and. iwater .le. nres) then
       renwat = res(iwater)
    else
       renwat = 'TIP3'  ! default water resn='TIP3'
       call wrndie(-2, '<SHKSET>', &
            'No atoms match specified water residue index')
    end if

    qfshake = .false.

    if (qnoreset) iconb = -iconb

    ! Done parsing parameters, now do the actual initializing
    call init_shake(iconb, qparam, qcomp, qfshk, renwat, islct, jslct)

    call timer_stop(t_shakesetup)
    success = f2c_logical(qsuccess)
  end function shake_on
end module api_shake
