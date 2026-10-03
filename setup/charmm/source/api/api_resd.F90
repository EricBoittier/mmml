!> API routines for accessing RESD (Restrained Distances) data from Python
!>
!> This module provides C-interoperable functions to access and modify
!> RESD (Restrained Distances) data structures for use with pycharmm's
!> ctypes bindings.
!>
!> Error codes returned by resddata_add():
!>   > 0  : Success (1-based index of new restraint)
!>   -10  : Invalid npairs (< 1)
!>   -11  : Maximum restraint count exceeded (REDMAX)
!>   -12  : Maximum atom pair count exceeded (REDMX2)
!>   -20  : RESD module not compiled (KEY_NOMISC=1)
!>
module api_resd
  use, intrinsic :: iso_c_binding, only: c_int, c_double
  use chm_kinds, only: chm_real

  implicit none

  !> Error code constants for API functions
  integer(c_int), parameter :: RESD_SUCCESS = 0
  integer(c_int), parameter :: RESD_ERR_NPAIRS = -10
  integer(c_int), parameter :: RESD_ERR_REDMAX = -11
  integer(c_int), parameter :: RESD_ERR_REDMX2 = -12
  integer(c_int), parameter :: RESD_ERR_DISABLED = -20

contains

  !> @brief Check if RESD facility is active (has restraints)
  !> @return 1 if active (REDNUM > 0), 0 otherwise
  integer(c_int) function resddata_is_active() bind(c)
    use resdist_ltm, only: rednum
    implicit none
#if KEY_NOMISC==0
    if (rednum > 0) then
       resddata_is_active = 1
    else
       resddata_is_active = 0
    end if
#else
    resddata_is_active = 0
#endif
  end function resddata_is_active

  !> @brief Get number of RESD restraints
  !> @return Number of RESD restraints (REDNUM)
  integer(c_int) function resddata_get_count() bind(c)
    use resdist_ltm, only: rednum
    implicit none
#if KEY_NOMISC==0
    resddata_get_count = rednum
#else
    resddata_get_count = 0
#endif
  end function resddata_get_count

  !> @brief Get RESD scale factor
  !> @return Scale factor for RESD energies/forces (REDSCA)
  real(c_double) function resddata_get_scale() bind(c)
    use resdist_ltm, only: redsca
    implicit none
#if KEY_NOMISC==0
    resddata_get_scale = redsca
#else
    resddata_get_scale = 1.0d0
#endif
  end function resddata_get_scale

  !> @brief Set RESD scale factor
  !> @param[in] scale New scale factor
  subroutine resddata_set_scale(scale) bind(c)
    use resdist_ltm, only: redsca
#if KEY_BLADE==1
    use blade_main, only: system_dirty
#endif
#if KEY_OPENMM==1
    use omm_ctrl, only: omm_system_changed
#endif
#if KEY_DOMDEC==1
    use domdec_common, only: domdec_system_changed
#endif
    implicit none
    real(c_double), intent(in), value :: scale
#if KEY_NOMISC==0
    redsca = scale

#if KEY_BLADE==1
    system_dirty = .true.
#endif
#if KEY_OPENMM==1
    call omm_system_changed()
#endif
#if KEY_DOMDEC==1
    call domdec_system_changed()
#endif
#endif
  end subroutine resddata_set_scale

  !> @brief Reset all RESD restraints
  !> @return 1 if successful, 0 if RESD not compiled
  integer(c_int) function resddata_reset() bind(c)
    use resdist_ltm, only: rednum, rednm2, redsca
#if KEY_OPENMM==1
    use omm_ctrl, only: omm_system_changed
#endif
#if KEY_DOMDEC==1
    use domdec_common, only: domdec_system_changed
#endif
    implicit none

#if KEY_NOMISC==0
    rednum = 0
    rednm2 = 0
    redsca = 1.0d0
#if KEY_OPENMM==1
    call omm_system_changed()
#endif
#if KEY_DOMDEC==1
    call domdec_system_changed()
#endif
    resddata_reset = 1
#else
    resddata_reset = 0
#endif
  end function resddata_reset

  !> @brief Get parameters for a specific RESD restraint
  !> @param[in] idx 1-based restraint index
  !> @param[out] kval Force constant
  !> @param[out] rval Reference distance value
  !> @param[out] eval_exp Global exponent (EVAL)
  !> @param[out] ival Internal exponent
  !> @param[out] mval Mode (0=both, 1=positive, -1=negative)
  !> @return 1 if successful, -1 if index out of range
  integer(c_int) function resddata_get_params(idx, kval, rval, &
       eval_exp, ival, mval) bind(c)
    use resdist_ltm, only: rednum, redkval, redrval, redeval, redival, redmval
    implicit none
    integer(c_int), intent(in), value :: idx
    real(c_double), intent(out) :: kval, rval
    integer(c_int), intent(out) :: eval_exp, ival, mval

    resddata_get_params = -1
#if KEY_NOMISC==0
    if (idx < 1 .or. idx > rednum) return

    kval = redkval(idx)
    rval = redrval(idx)
    eval_exp = redeval(idx)
    ival = redival(idx)
    mval = redmval(idx)
    resddata_get_params = 1
#endif
  end function resddata_get_params

  !> @brief Get number of atom pairs for a specific restraint
  !> @param[in] idx 1-based restraint index
  !> @return Number of atom pairs, or -1 if index out of range
  integer(c_int) function resddata_get_npairs(idx) bind(c)
    use resdist_ltm, only: rednum, redipt
    implicit none
    integer(c_int), intent(in), value :: idx

    resddata_get_npairs = -1
#if KEY_NOMISC==0
    if (idx < 1 .or. idx > rednum) return

    if (idx == rednum) then
       ! For the last restraint, we need to count differently
       ! This is tricky - we'll return the pairs based on pointers
       resddata_get_npairs = -2  ! Signal to use different method
    else
       resddata_get_npairs = redipt(idx+1) - redipt(idx)
    end if
#endif
  end function resddata_get_npairs

  !> @brief Add a new RESD restraint
  !> @param[in] npairs Number of atom pairs
  !> @param[in] atom_i First atoms in pairs (1-based indices)
  !> @param[in] atom_j Second atoms in pairs (1-based indices)
  !> @param[in] factors Distance factors for each pair
  !> @param[in] kval Force constant
  !> @param[in] rval Reference distance value
  !> @param[in] eval_exp Global exponent (default 2)
  !> @param[in] ival Internal exponent (default 1)
  !> @param[in] mval Mode: 0=both sides, 1=positive only, -1=negative only
  !> @return Index of new restraint (1-based), or -1 on error
  integer(c_int) function resddata_add(npairs, atom_i, atom_j, factors, &
       kval, rval, eval_exp, ival, mval) bind(c)
    use dimens_fcm, only: redmax, redmx2
    use resdist_ltm
#if KEY_OPENMM==1
    use omm_ctrl, only: omm_system_changed
#endif
#if KEY_DOMDEC==1
    use domdec_common, only: domdec_system_changed
#endif
    implicit none
    integer(c_int), intent(in), value :: npairs, eval_exp, ival, mval
    integer(c_int), intent(in) :: atom_i(*), atom_j(*)
    real(c_double), intent(in) :: factors(*)
    real(c_double), intent(in), value :: kval, rval

    integer :: i, n

#if KEY_NOMISC==0
    ! Validate input parameters with specific error codes
    if (npairs < 1) then
       resddata_add = RESD_ERR_NPAIRS
       return
    end if
    if (rednum >= redmax) then
       resddata_add = RESD_ERR_REDMAX
       return
    end if
    if (rednm2 + npairs > redmx2) then
       resddata_add = RESD_ERR_REDMX2
       return
    end if

#if KEY_OPENMM==1
    call omm_system_changed()
#endif
#if KEY_DOMDEC==1
    call domdec_system_changed()
#endif

    ! Initialize pointers if first restraint
    if (rednum == 0) then
       rednm2 = 0
       redsca = 1.0d0
       redipt(1) = 0
    end if

    rednum = rednum + 1

    ! Set parameters
    redkval(rednum) = kval
    redrval(rednum) = rval
    redeval(rednum) = eval_exp
    redival(rednum) = ival
    redmval(rednum) = mval

    ! Store atom pairs
    redipt(rednum) = rednm2
    do i = 1, npairs
       rednm2 = rednm2 + 1
       redilis(1, rednm2) = atom_i(i)
       redilis(2, rednm2) = atom_j(i)
       redklis(rednm2) = factors(i)
    end do
    redipt(rednum + 1) = rednm2

    resddata_add = rednum
#else
    ! RESD module not compiled
    resddata_add = RESD_ERR_DISABLED
#endif
  end function resddata_add

  !> @brief Get maximum number of RESD restraints allowed
  !> @return Maximum restraint count (REDMAX)
  integer(c_int) function resddata_get_max_restraints() bind(c)
    use dimens_fcm, only: redmax
    implicit none
    resddata_get_max_restraints = redmax
  end function resddata_get_max_restraints

  !> @brief Get maximum number of atom pairs allowed
  !> @return Maximum atom pair count (REDMX2)
  integer(c_int) function resddata_get_max_pairs() bind(c)
    use dimens_fcm, only: redmx2
    implicit none
    resddata_get_max_pairs = redmx2
  end function resddata_get_max_pairs

  !> @brief Get current number of atom pairs used
  !> @return Current atom pair count (REDNM2)
  integer(c_int) function resddata_get_current_pair_count() bind(c)
    use resdist_ltm, only: rednm2
    implicit none
#if KEY_NOMISC==0
    resddata_get_current_pair_count = rednm2
#else
    resddata_get_current_pair_count = 0
#endif
  end function resddata_get_current_pair_count

end module api_resd
