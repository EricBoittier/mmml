!> routines for applying constraints on atoms
module api_cons_fix

  implicit none

contains

  !> @brief deactivate fix constraints
  !
  !> @return cons_fix_turn_off
  !>         1 <=> success
  function cons_fix_turn_off(icomp) bind(c) result(success)
    use, intrinsic :: iso_c_binding, only: c_int
    use api_util, only: c2f_logical, f2c_logical
    use coord, only: x, y, z, wmain
    use coordc, only: xcomp, ycomp, zcomp, wcomp
    ! use cnst_fcm, only: allocate_cnst
    use cstran_mod, only: setup_fix_const
    use psf, only: natom, nbond, ntheta

    use code, only: &
#if KEY_CMAP==1
         icct, &
#endif
         mustup, icb, ici, icp, ict

#if KEY_CFF==1 || KEY_MMFF==1
    use ffieldm, only: ffield, cff, mmff
#endif

    use memory, only: chmalloc, chmdealloc
    use stream, only: outu

    implicit none

    ! args
    integer(c_int) :: icomp

    ! locals
    logical :: qcomp, qpurge, qbond, qangle, qphi, qimp, qcmap, &
         qsuccess, mstup
    integer, allocatable, dimension(:) :: mbnd, mtheta
    integer :: selection(natom)

    ! result
    integer(c_int) :: success

    success = -1

#if KEY_MMFF==1 || KEY_CFF==1 /*ffield_fcm*/
    IF (FFIELD == CFF .OR. FFIELD == MMFF) THEN
       call chmalloc(__FILE__, 'cons_fix_setup', 'MBND', NBOND, intg=MBND)
       call chmalloc(__FILE__, 'cons_fix_setup', 'MTHETA', NTHETA, intg=MTHETA)
    else
       call chmalloc(__FILE__, 'cons_fix_setup', 'MBND', 1, intg=MBND)
       call chmalloc(__FILE__, 'cons_fix_setup', 'MTHETA', 1, intg=MTHETA)
    ENDIF
#else /* (ffield_fcm)*/
    call chmalloc(__FILE__, 'cons_fix_setup', 'MBND', 1, intg=MBND)
    call chmalloc(__FILE__, 'cons_fix_setup', 'MTHETA', 1, intg=MTHETA)
#endif /* (ffield_fcm)*/

    mstup = .false.

    qcomp = c2f_logical(icomp)
    qpurge = .false.
    qbond = .false.
    qangle = .false.
    qphi = .false.
    qimp = .false.
    qcmap = .false.

    selection = 0

    ! call allocate_cnst(natom)

    if (qcomp) then
       qsuccess = setup_fix_const(xcomp, ycomp, zcomp, wcomp, selection, &
            qpurge, &
            qbond, qangle, qphi, qimp, &
#if KEY_CMAP == 1
            qcmap, &
#endif
            mstup, mustup, &
            mbnd, mtheta, &
            icb, ici, icp, ict &
#if KEY_CMAP == 1
            , icct &
#endif
            )
    else
       qsuccess = setup_fix_const(x, y, z, wmain, selection, &
            qpurge, &
            qbond, qangle, qphi, qimp, &
#if KEY_CMAP == 1
            qcmap, &
#endif
            mstup, mustup, &
            mbnd, mtheta, &
            icb, ici, icp, ict &
#if KEY_CMAP == 1
            , icct &
#endif
            )
    end if

#if KEY_MMFF==1 || KEY_CFF==1 /*ffield_fcm*/
    IF (FFIELD == CFF .OR. FFIELD == MMFF) THEN
       call chmdealloc(__FILE__, 'cons_fix_setup', 'MBND', NBOND, intg=MBND)
       call chmdealloc(__FILE__, 'cons_fix_setup', 'MTHETA', NTHETA, intg=MTHETA)
    else
       call chmdealloc(__FILE__, 'cons_fix_setup', 'MBND', 1, intg=MBND)
       call chmdealloc(__FILE__, 'cons_fix_setup', 'MTHETA', 1, intg=MTHETA)
    ENDIF
#else /* (ffield_fcm)*/
    call chmdealloc(__FILE__, 'cons_fix_setup', 'MBND', 1, intg=MBND)
    call chmdealloc(__FILE__, 'cons_fix_setup', 'MTHETA', 1, intg=MTHETA)
#endif /* (ffield_fcm)*/

    if (mstup) call psfsum(outu)
    success = f2c_logical(qsuccess)
  end function cons_fix_turn_off

  !> @brief set up fix constraints
  !
  !> @param[in] selection selection(i) .eq. 1 <=> constrain atom i
  !> @param[in] icomp 1 <=> use comparison set
  !> @param[in] ipurge
  !> @param[in] ibond
  !> @param[in] iangle
  !> @param[in] iphi
  !> @param[in] iimp
  !> @param[in] icmap
  !> @return success 1 <=> no error
  function cons_fix_setup(selection, &
       icomp, ipurge, ibond, iangle, iphi, iimp, icmap &
       ) bind(c) result(success)
    use, intrinsic :: iso_c_binding, only: c_int
    use api_util, only: c2f_logical, f2c_logical
    use coord, only: x, y, z, wmain
    use coordc, only: xcomp, ycomp, zcomp, wcomp
    ! use cnst_fcm, only: allocate_cnst
    use cstran_mod, only: setup_fix_const
    use psf, only: nbond, ntheta

    use code, only: &
#if KEY_CMAP==1
         icct, &
#endif
         mustup, icb, ici, icp, ict

#if KEY_CFF==1 || KEY_MMFF==1
    use ffieldm, only: ffield, cff, mmff
#endif

    use memory, only: chmalloc, chmdealloc
    use stream, only: outu

    implicit none

    ! args
    integer(c_int) :: selection(*), &
         icomp, ipurge, ibond, iangle, iphi, iimp, icmap

    ! locals
    logical :: qcomp, qpurge, qbond, qangle, qphi, qimp, qcmap, &
         qsuccess, mstup
    integer, allocatable, dimension(:) :: mbnd, mtheta

    ! result
    integer(c_int) :: success

    success = -1

#if KEY_MMFF==1 || KEY_CFF==1 /*ffield_fcm*/
    IF (FFIELD == CFF .OR. FFIELD == MMFF) THEN
       call chmalloc(__FILE__, 'cons_fix_setup', 'MBND', NBOND, intg=MBND)
       call chmalloc(__FILE__, 'cons_fix_setup', 'MTHETA', NTHETA, intg=MTHETA)
    else
       call chmalloc(__FILE__, 'cons_fix_setup', 'MBND', 1, intg=MBND)
       call chmalloc(__FILE__, 'cons_fix_setup', 'MTHETA', 1, intg=MTHETA)
    ENDIF
#else /* (ffield_fcm)*/
    call chmalloc(__FILE__, 'cons_fix_setup', 'MBND', 1, intg=MBND)
    call chmalloc(__FILE__, 'cons_fix_setup', 'MTHETA', 1, intg=MTHETA)
#endif /* (ffield_fcm)*/

    mstup = .false.

    qcomp = c2f_logical(icomp)
    qpurge = c2f_logical(ipurge)
    qbond = c2f_logical(ibond)
    qangle = c2f_logical(iangle)
    qphi = c2f_logical(iphi)
    qimp = c2f_logical(iimp)
    qcmap = c2f_logical(icmap)

    ! call allocate_cnst(natom)

    if (qcomp) then
       qsuccess = setup_fix_const(xcomp, ycomp, zcomp, wcomp, selection, &
            qpurge, &
            qbond, qangle, qphi, qimp, &
#if KEY_CMAP == 1
            qcmap, &
#endif
            mstup, mustup, &
            mbnd, mtheta, &
            icb, ici, icp, ict &
#if KEY_CMAP == 1
            , icct &
#endif
            )
    else
       qsuccess = setup_fix_const(x, y, z, wmain, selection, &
            qpurge, &
            qbond, qangle, qphi, qimp, &
#if KEY_CMAP == 1
            qcmap, &
#endif
            mstup, mustup, &
            mbnd, mtheta, &
            icb, ici, icp, ict &
#if KEY_CMAP == 1
            , icct &
#endif
            )
    end if

#if KEY_MMFF==1 || KEY_CFF==1 /*ffield_fcm*/
    IF (FFIELD == CFF .OR. FFIELD == MMFF) THEN
       call chmdealloc(__FILE__, 'cons_fix_setup', 'MBND', NBOND, intg=MBND)
       call chmdealloc(__FILE__, 'cons_fix_setup', 'MTHETA', NTHETA, intg=MTHETA)
    else
       call chmdealloc(__FILE__, 'cons_fix_setup', 'MBND', 1, intg=MBND)
       call chmdealloc(__FILE__, 'cons_fix_setup', 'MTHETA', 1, intg=MTHETA)
    ENDIF
#else /* (ffield_fcm)*/
    call chmdealloc(__FILE__, 'cons_fix_setup', 'MBND', 1, intg=MBND)
    call chmdealloc(__FILE__, 'cons_fix_setup', 'MTHETA', 1, intg=MTHETA)
#endif /* (ffield_fcm)*/

    if (mstup) call psfsum(outu)
    success = f2c_logical(qsuccess)
  end function cons_fix_setup
end module api_cons_fix
