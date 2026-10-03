!> API routines for accessing NOE restraint data from Python
!>
!> This module provides C-interoperable functions to access NOE
!> (Nuclear Overhauser Effect) restraint data structures for use
!> with pycharmm's ctypes bindings.
!>
module api_noe
  use, intrinsic :: iso_c_binding, only: c_int, c_double
  use chm_kinds, only: chm_real

  implicit none

contains

  !> @brief Check if NOE facility is active (has restraints)
  !> @return 1 if active (NOENUM > 0), 0 otherwise
  integer(c_int) function noedata_is_active() bind(c)
    use noem, only: noenum
    implicit none
#if KEY_NOMISC==0
    if (noenum > 0) then
       noedata_is_active = 1
    else
       noedata_is_active = 0
    end if
#else
    noedata_is_active = 0
#endif
  end function noedata_is_active

  !> @brief Get number of NOE restraints
  !> @return Number of NOE restraints (NOENUM)
  integer(c_int) function noedata_get_count() bind(c)
    use noem, only: noenum
    implicit none
#if KEY_NOMISC==0
    noedata_get_count = noenum
#else
    noedata_get_count = 0
#endif
  end function noedata_get_count

  !> @brief Get NOE scale factor
  !> @return Scale factor for NOE energies/forces (NOESCA)
  real(c_double) function noedata_get_scale() bind(c)
    use noem, only: noesca
    implicit none
#if KEY_NOMISC==0
    noedata_get_scale = noesca
#else
    noedata_get_scale = 1.0d0
#endif
  end function noedata_get_scale

  !> @brief Set NOE scale factor
  !> @param[in] scale New scale factor
  subroutine noedata_set_scale(scale) bind(c)
    use noem, only: noesca
#if KEY_BLADE==1
    use blade_main, only: system_dirty
#endif
#if KEY_DOMDEC==1
    use domdec_common, only: domdec_system_changed
#endif
    implicit none
    real(c_double), intent(in), value :: scale
#if KEY_NOMISC==0
    noesca = scale

#if KEY_BLADE==1
    system_dirty = .true.
#endif
#if KEY_DOMDEC==1
    call domdec_system_changed()
#endif
#endif
  end subroutine noedata_set_scale

  !> @brief Get parameters for a specific NOE restraint
  !> @param[in] idx 1-based restraint index
  !> @param[out] kmin Force constant for minimum distance
  !> @param[out] rmin Minimum distance
  !> @param[out] kmax Force constant for maximum distance
  !> @param[out] rmax Maximum distance
  !> @param[out] fmax Maximum force
  !> @param[out] tcon Time constant
  !> @param[out] rexp Reciprocal exponent
  !> @return 1 if successful, -1 if index out of range
  integer(c_int) function noedata_get_params(idx, kmin, rmin, kmax, rmax, &
       fmax, tcon, rexp) bind(c)
    use noem, only: noenum, noekmn, noermn, noekmx, noermx, noefmx, &
         noetcn, noeexp
    implicit none
    integer(c_int), intent(in), value :: idx
    real(c_double), intent(out) :: kmin, rmin, kmax, rmax, fmax, tcon, rexp

    noedata_get_params = -1
#if KEY_NOMISC==0
    if (idx < 1 .or. idx > noenum) return

    kmin = noekmn(idx)
    rmin = noermn(idx)
    kmax = noekmx(idx)
    rmax = noermx(idx)
    fmax = noefmx(idx)
    tcon = noetcn(idx)
    rexp = noeexp(idx)
    noedata_get_params = 1
#endif
  end function noedata_get_params

  !> @brief Get soft-core parameters for a specific NOE restraint
  !> @param[in] idx 1-based restraint index
  !> @param[out] rswi Switch distance
  !> @param[out] sexp Soft exponent
  !> @return 1 if successful, -1 if index out of range
  integer(c_int) function noedata_get_soft_params(idx, rswi, sexp) bind(c)
    use noem, only: noenum, noersw, noesex
    implicit none
    integer(c_int), intent(in), value :: idx
    real(c_double), intent(out) :: rswi, sexp

    noedata_get_soft_params = -1
#if KEY_NOMISC==0
    if (idx < 1 .or. idx > noenum) return

    rswi = noersw(idx)
    sexp = noesex(idx)
    noedata_get_soft_params = 1
#endif
  end function noedata_get_soft_params

  !> @brief Check if restraint uses MINDIST averaging
  !> @param[in] idx 1-based restraint index
  !> @return 1 if MINDIST, 0 if not, -1 if index out of range
  integer(c_int) function noedata_is_mindist(idx) bind(c)
    use noem, only: noenum, noemin
    implicit none
    integer(c_int), intent(in), value :: idx

    noedata_is_mindist = -1
#if KEY_NOMISC==0
    if (idx < 1 .or. idx > noenum) return

    if (noemin(idx)) then
       noedata_is_mindist = 1
    else
       noedata_is_mindist = 0
    end if
#endif
  end function noedata_is_mindist

#if KEY_PNOE==1
  !> @brief Check if restraint is a Point NOE (PNOE)
  !> @param[in] idx 1-based restraint index
  !> @return 1 if PNOE, 0 if not, -1 if index out of range
  integer(c_int) function noedata_is_pnoe(idx) bind(c)
    use noem, only: noenum, ispnoe
    implicit none
    integer(c_int), intent(in), value :: idx

    noedata_is_pnoe = -1
    if (idx < 1 .or. idx > noenum) return

    if (ispnoe(idx)) then
       noedata_is_pnoe = 1
    else
       noedata_is_pnoe = 0
    end if
  end function noedata_is_pnoe

  !> @brief Get PNOE reference coordinates
  !> @param[in] idx 1-based restraint index
  !> @param[out] cx X coordinate
  !> @param[out] cy Y coordinate
  !> @param[out] cz Z coordinate
  !> @return 1 if successful, -1 if not PNOE or index out of range
  integer(c_int) function noedata_get_pnoe_coords(idx, cx, cy, cz) bind(c)
    use noem, only: noenum, ispnoe, c0x, c0y, c0z
    implicit none
    integer(c_int), intent(in), value :: idx
    real(c_double), intent(out) :: cx, cy, cz

    noedata_get_pnoe_coords = -1
    if (idx < 1 .or. idx > noenum) return
    if (.not. ispnoe(idx)) return

    cx = c0x(idx)
    cy = c0y(idx)
    cz = c0z(idx)
    noedata_get_pnoe_coords = 1
  end function noedata_get_pnoe_coords

  !> @brief Set PNOE reference coordinates (for moving PNOE)
  !> @param[in] idx 1-based restraint index
  !> @param[in] cx New X coordinate
  !> @param[in] cy New Y coordinate
  !> @param[in] cz New Z coordinate
  !> @return 1 if successful, -1 if not PNOE or index out of range
  integer(c_int) function noedata_set_pnoe_coords(idx, cx, cy, cz) bind(c)
    use noem, only: noenum, ispnoe, c0x, c0y, c0z
#if KEY_BLADE==1
    use blade_main, only: system_dirty
#endif
#if KEY_DOMDEC==1
    use domdec_common, only: domdec_system_changed
#endif
    implicit none
    integer(c_int), intent(in), value :: idx
    real(c_double), intent(in), value :: cx, cy, cz

    noedata_set_pnoe_coords = -1
    if (idx < 1 .or. idx > noenum) return
    if (.not. ispnoe(idx)) return

    c0x(idx) = cx
    c0y(idx) = cy
    c0z(idx) = cz
    noedata_set_pnoe_coords = 1

#if KEY_BLADE==1
    system_dirty = .true.
#endif
#if KEY_DOMDEC==1
    call domdec_system_changed()
#endif
  end function noedata_set_pnoe_coords

  !> @brief Check if PNOE is moving
  !> @param[in] idx 1-based restraint index
  !> @return 1 if moving PNOE, 0 if not, -1 if index out of range
  integer(c_int) function noedata_is_moving_pnoe(idx) bind(c)
    use noem, only: noenum, ispnoe, mvpnoe
    implicit none
    integer(c_int), intent(in), value :: idx

    noedata_is_moving_pnoe = -1
    if (idx < 1 .or. idx > noenum) return
    if (.not. ispnoe(idx)) return

    if (mvpnoe(idx)) then
       noedata_is_moving_pnoe = 1
    else
       noedata_is_moving_pnoe = 0
    end if
  end function noedata_is_moving_pnoe

  !> @brief Get moving PNOE step information
  !> @param[out] current_step Current step number (IMPNOE)
  !> @param[out] total_steps Total number of steps (NMPNOE)
  subroutine noedata_get_mpnoe_steps(current_step, total_steps) bind(c)
    use noem, only: impnoe, nmpnoe
    implicit none
    integer(c_int), intent(out) :: current_step, total_steps

    current_step = impnoe
    total_steps = nmpnoe
  end subroutine noedata_get_mpnoe_steps
#endif

  !> @brief Get all parameters for multiple restraints efficiently
  !> @param[in] n Number of restraints to retrieve
  !> @param[out] kmin_arr Array of kmin values
  !> @param[out] rmin_arr Array of rmin values
  !> @param[out] kmax_arr Array of kmax values
  !> @param[out] rmax_arr Array of rmax values
  !> @return Number of restraints actually copied
  integer(c_int) function noedata_get_params_array(n, kmin_arr, rmin_arr, &
       kmax_arr, rmax_arr) bind(c)
    use noem, only: noenum, noekmn, noermn, noekmx, noermx
    implicit none
    integer(c_int), intent(in), value :: n
    real(c_double), intent(out) :: kmin_arr(*), rmin_arr(*), kmax_arr(*), rmax_arr(*)
    integer :: i, ncopy

    noedata_get_params_array = 0
#if KEY_NOMISC==0
    ncopy = min(n, noenum)
    do i = 1, ncopy
       kmin_arr(i) = noekmn(i)
       rmin_arr(i) = noermn(i)
       kmax_arr(i) = noekmx(i)
       rmax_arr(i) = noermx(i)
    end do
    noedata_get_params_array = ncopy
#endif
  end function noedata_get_params_array

  !> @brief Reset all NOE restraints
  !> @return 1 if successful, 0 if NOE not compiled
  integer(c_int) function noedata_reset() bind(c)
    use noem, only: noenum, noenm2, noesca
#if KEY_DOMDEC==1
    use domdec_common, only: domdec_system_changed
#endif
    implicit none

#if KEY_NOMISC==0
    noenum = 0
    noenm2 = 0
    noesca = 1.0d0
#if KEY_DOMDEC==1
    call domdec_system_changed()
#endif
    noedata_reset = 1
#else
    noedata_reset = 0
#endif
  end function noedata_reset

  !> @brief Add an NOE restraint between two atom groups
  !> @param[in] ni Number of atoms in first selection
  !> @param[in] ilist Array of atom indices in first selection (1-based)
  !> @param[in] nj Number of atoms in second selection
  !> @param[in] jlist Array of atom indices in second selection (1-based)
  !> @param[in] kmin Force constant for minimum distance
  !> @param[in] rmin Minimum distance
  !> @param[in] kmax Force constant for maximum distance
  !> @param[in] rmax Maximum distance
  !> @param[in] fmax Maximum force (default 1.0)
  !> @param[in] tcon Time constant (default 0.0)
  !> @param[in] rexp Reciprocal exponent (default 1.0)
  !> @param[in] rswi Switch distance for soft asymptote (-1 to disable)
  !> @param[in] sexp Soft exponent (default 1.0)
  !> @param[in] mindist 1 if MINDIST averaging, 0 otherwise
  !> @return Index of new restraint (1-based), or -1 on error
  integer(c_int) function noedata_assign(ni, ilist, nj, jlist, &
       kmin, rmin, kmax, rmax, fmax, tcon, rexp, rswi, sexp, mindist) bind(c)
    use noem
    use dimens_fcm, only: noemax
#if KEY_DOMDEC==1
    use domdec_common, only: domdec_system_changed
#endif
    implicit none
    integer(c_int), intent(in), value :: ni, nj, mindist
    integer(c_int), intent(in) :: ilist(*), jlist(*)
    real(c_double), intent(in), value :: kmin, rmin, kmax, rmax
    real(c_double), intent(in), value :: fmax, tcon, rexp, rswi, sexp

    integer :: i, n

    noedata_assign = -1
#if KEY_NOMISC==0
    if (ni < 1 .or. nj < 1) return

    ! Initialize NOE arrays if not yet allocated
    if (.not. allocated(noelis)) call noe_init()

    ! Ensure storage is available
    if (noenum >= noemax) call noe_add_storage()

    noenum = noenum + 1
#if KEY_DOMDEC==1
    call domdec_system_changed()
#endif

    ! Store I selection
    noeipt(noenum) = noenm2 + 1
    noeinm(noenum) = ni
    n = noenm2
    do i = 1, ni
       if (noenm2 >= noemax) call noe_add_storage()
       noenm2 = noenm2 + 1
       noelis(noenm2) = ilist(i)
    end do

    ! Store J selection
    noejpt(noenum) = noenm2 + 1
    noejnm(noenum) = nj
    do i = 1, nj
       if (noenm2 >= noemax) call noe_add_storage()
       noenm2 = noenm2 + 1
       noelis(noenm2) = jlist(i)
    end do

    ! Set parameters
    noekmn(noenum) = kmin
    noermn(noenum) = rmin
    noekmx(noenum) = kmax
    noermx(noenum) = rmax
    noefmx(noenum) = fmax
    noetcn(noenum) = tcon
    noeexp(noenum) = rexp
    noemin(noenum) = (mindist /= 0)
    noersw(noenum) = rswi
    noesex(noenum) = sexp
    noeram(noenum) = 0

    ! Compute average
    if (noermx(noenum) > 0.0d0) then
       noeave(noenum) = 1.0d0 / noermx(noenum)**3
    else
       noeave(noenum) = 0.0d0
    end if

#if KEY_PNOE==1
    ! Not a PNOE
    ispnoe(noenum) = .false.
    c0x(noenum) = 0.0d0
    c0y(noenum) = 0.0d0
    c0z(noenum) = 0.0d0
    mvpnoe(noenum) = .false.
#endif

    noedata_assign = noenum
#endif
  end function noedata_assign

#if KEY_PNOE==1
  !> @brief Add a Point NOE restraint (PNOE)
  !> @param[in] ni Number of atoms in selection
  !> @param[in] ilist Array of atom indices (1-based)
  !> @param[in] cnox X coordinate of reference point
  !> @param[in] cnoy Y coordinate of reference point
  !> @param[in] cnoz Z coordinate of reference point
  !> @param[in] kmin Force constant for minimum distance
  !> @param[in] rmin Minimum distance
  !> @param[in] kmax Force constant for maximum distance
  !> @param[in] rmax Maximum distance
  !> @param[in] fmax Maximum force
  !> @param[in] tcon Time constant
  !> @param[in] rexp Reciprocal exponent
  !> @return Index of new restraint (1-based), or -1 on error
  integer(c_int) function noedata_assign_pnoe(ni, ilist, &
       cnox, cnoy, cnoz, kmin, rmin, kmax, rmax, fmax, tcon, rexp) bind(c)
    use noem
    use dimens_fcm, only: noemax
#if KEY_DOMDEC==1
    use domdec_common, only: domdec_system_changed
#endif
    implicit none
    integer(c_int), intent(in), value :: ni
    integer(c_int), intent(in) :: ilist(*)
    real(c_double), intent(in), value :: cnox, cnoy, cnoz
    real(c_double), intent(in), value :: kmin, rmin, kmax, rmax
    real(c_double), intent(in), value :: fmax, tcon, rexp

    integer :: i

    noedata_assign_pnoe = -1
    if (ni < 1) return

    ! Initialize NOE arrays if not yet allocated
    if (.not. allocated(noelis)) call noe_init()

    ! Ensure storage is available
    if (noenum >= noemax) call noe_add_storage()

    noenum = noenum + 1
#if KEY_DOMDEC==1
    call domdec_system_changed()
#endif

    ! Store I selection
    noeipt(noenum) = noenm2 + 1
    noeinm(noenum) = ni
    do i = 1, ni
       if (noenm2 >= noemax) call noe_add_storage()
       noenm2 = noenm2 + 1
       noelis(noenm2) = ilist(i)
    end do

    ! For PNOE, J selection is not used, but BLaDE requires noejnm=1
    ! Add a dummy J-atom entry using first I-atom (BLaDE only checks count)
    noejpt(noenum) = noenm2 + 1
    if (noenm2 >= noemax) call noe_add_storage()
    noenm2 = noenm2 + 1
    noelis(noenm2) = ilist(1)  ! Dummy entry for BLaDE compatibility
    noejnm(noenum) = 1

    ! Set parameters
    noekmn(noenum) = kmin
    noermn(noenum) = rmin
    noekmx(noenum) = kmax
    noermx(noenum) = rmax
    noefmx(noenum) = fmax
    noetcn(noenum) = tcon
    noeexp(noenum) = rexp
    noemin(noenum) = .false.
    noersw(noenum) = -1.0d0  ! PNOE doesn't use soft asymptote
    noesex(noenum) = 1.0d0
    noeram(noenum) = 0

    ! Compute average
    if (noermx(noenum) > 0.0d0) then
       noeave(noenum) = 1.0d0 / noermx(noenum)**3
    else
       noeave(noenum) = 0.0d0
    end if

    ! Set PNOE-specific fields
    ispnoe(noenum) = .true.
    c0x(noenum) = cnox
    c0y(noenum) = cnoy
    c0z(noenum) = cnoz
    mvpnoe(noenum) = .false.

    noedata_assign_pnoe = noenum
  end function noedata_assign_pnoe

  !> @brief Set moving PNOE target
  !> @param[in] idx 1-based restraint index
  !> @param[in] tnox Target X coordinate
  !> @param[in] tnoy Target Y coordinate
  !> @param[in] tnoz Target Z coordinate
  !> @return 1 if successful, -1 if not PNOE or index out of range
  integer(c_int) function noedata_set_mpnoe_target(idx, tnox, tnoy, tnoz) bind(c)
    use noem, only: noenum, ispnoe, mvpnoe, tc0x, tc0y, tc0z
    implicit none
    integer(c_int), intent(in), value :: idx
    real(c_double), intent(in), value :: tnox, tnoy, tnoz

    noedata_set_mpnoe_target = -1
    if (idx < 1 .or. idx > noenum) return
    if (.not. ispnoe(idx)) return

    tc0x(idx) = tnox
    tc0y(idx) = tnoy
    tc0z(idx) = tnoz
    mvpnoe(idx) = .true.
    noedata_set_mpnoe_target = 1
  end function noedata_set_mpnoe_target

  !> @brief Set number of steps for moving PNOE
  !> @param[in] nsteps Number of steps
  subroutine noedata_set_nmpnoe(nsteps) bind(c)
    use noem, only: nmpnoe, impnoe
    implicit none
    integer(c_int), intent(in), value :: nsteps

    nmpnoe = nsteps
    impnoe = 0
  end subroutine noedata_set_nmpnoe
#endif

  !> @brief Get atom indices for a specific NOE restraint
  !> @param[in] idx 1-based restraint index
  !> @param[out] ni Number of atoms in I selection
  !> @param[out] nj Number of atoms in J selection
  !> @param[out] ilist Array to store I selection atom indices (1-based)
  !> @param[out] jlist Array to store J selection atom indices (1-based)
  !> @param[in] max_atoms Maximum size of ilist/jlist arrays
  !> @return 1 if successful, -1 if index out of range, -2 if arrays too small
  integer(c_int) function noedata_get_atoms(idx, ni, nj, ilist, jlist, max_atoms) bind(c)
    use noem, only: noenum, noeipt, noeinm, noejpt, noejnm, noelis
    implicit none
    integer(c_int), intent(in), value :: idx, max_atoms
    integer(c_int), intent(out) :: ni, nj
    integer(c_int), intent(out) :: ilist(*), jlist(*)

    integer :: i, iptr, jptr

    noedata_get_atoms = -1
    ni = 0
    nj = 0
#if KEY_NOMISC==0
    if (idx < 1 .or. idx > noenum) return

    ni = noeinm(idx)
    nj = noejnm(idx)

    ! Check if arrays are large enough
    if (ni > max_atoms .or. nj > max_atoms) then
       noedata_get_atoms = -2
       return
    end if

    ! Copy I selection atoms
    iptr = noeipt(idx)
    do i = 1, ni
       ilist(i) = noelis(iptr + i - 1)
    end do

    ! Copy J selection atoms
    jptr = noejpt(idx)
    do i = 1, nj
       jlist(i) = noelis(jptr + i - 1)
    end do

    noedata_get_atoms = 1
#endif
  end function noedata_get_atoms

end module api_noe
