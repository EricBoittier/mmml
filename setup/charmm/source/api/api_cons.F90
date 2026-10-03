!> API routines for accessing CONS (constraint) data from Python
!>
!> This module provides C-interoperable functions to access and modify
!> constraint data structures for:
!> - Dihedral restraints (CONS DIHE)
!> - Internal coordinate restraints (CONS IC)
!> - Droplet restraints (CONS DROPLET)
!>
!> For harmonic and fix constraints, see api_cons_harm.F90 and api_cons_fix.F90
!>
module api_cons
  use, intrinsic :: iso_c_binding, only: c_int, c_double
  use chm_kinds, only: chm_real

  implicit none

contains

  !=========================================================================
  ! DIHEDRAL RESTRAINTS (CONS DIHE)
  !=========================================================================

  !> @brief Get number of dihedral restraints
  !> @return Number of dihedral restraints (NCSPHI)
  integer(c_int) function consdata_get_ndihe() bind(c)
    use cnst_fcm, only: ncsphi
    implicit none
    consdata_get_ndihe = ncsphi
  end function consdata_get_ndihe

  !> @brief Clear all dihedral restraints (CLDH)
  subroutine consdata_clear_dihe() bind(c)
    use cnst_fcm, only: ncsphi
    use param_store, only: set_param
#if KEY_BLADE==1
    use blade_main, only: system_dirty
#endif
#if KEY_DOMDEC==1
    use domdec_common, only: domdec_system_changed
#endif
#if KEY_OPENMM==1
    use omm_ctrl, only: omm_system_changed
#endif
    implicit none

    ncsphi = 0
    call set_param('NCSP', ncsphi)
#if KEY_BLADE==1
    system_dirty = .true.
#endif
#if KEY_DOMDEC==1
    call domdec_system_changed()
#endif
#if KEY_OPENMM==1
    call omm_system_changed()
#endif
  end subroutine consdata_clear_dihe

  !> @brief Get parameters for a dihedral restraint
  !> @param[in] idx 1-based restraint index
  !> @param[out] i_atom First atom index
  !> @param[out] j_atom Second atom index
  !> @param[out] k_atom Third atom index
  !> @param[out] l_atom Fourth atom index
  !> @param[out] force Force constant
  !> @param[out] min_angle Minimum angle (radians)
  !> @param[out] period Periodicity
  !> @param[out] width Half-width for flat bottom
  !> @return 1 if successful, -1 if index out of range
  integer(c_int) function consdata_get_dihe_params(idx, i_atom, j_atom, &
       k_atom, l_atom, force, min_angle, period, width) bind(c)
    use cnst_fcm, only: ncsphi, ics, jcs, kcs, lcs, ccsc, ccsb, ccsd, ccsw
    implicit none
    integer(c_int), intent(in), value :: idx
    integer(c_int), intent(out) :: i_atom, j_atom, k_atom, l_atom, period
    real(c_double), intent(out) :: force, min_angle, width

    consdata_get_dihe_params = -1
    if (idx < 1 .or. idx > ncsphi) return

    i_atom = ics(idx)
    j_atom = jcs(idx)
    k_atom = kcs(idx)
    l_atom = lcs(idx)
    force = ccsc(idx)
    min_angle = ccsb(idx)
    period = ccsd(idx)
    width = ccsw(idx)
    consdata_get_dihe_params = 1
  end function consdata_get_dihe_params

  !> @brief Add a dihedral restraint
  !> @param[in] i_atom First atom index (1-based)
  !> @param[in] j_atom Second atom index (1-based)
  !> @param[in] k_atom Third atom index (1-based)
  !> @param[in] l_atom Fourth atom index (1-based)
  !> @param[in] force Force constant
  !> @param[in] min_angle Minimum angle (radians)
  !> @param[in] period Periodicity (0 for harmonic)
  !> @param[in] width Half-width for flat bottom improper
  !> @return Index of new restraint (1-based), or -1 on error
  integer(c_int) function consdata_add_dihe(i_atom, j_atom, k_atom, l_atom, &
       force, min_angle, period, width) bind(c)
    use chm_kinds
    use cnst_fcm, only: ncsphi, ics, jcs, kcs, lcs, ccsc, ccsb, ccsd, ccsw, &
         ccscos, ccssin, iccs
    use consta, only: pi
    use param_store, only: set_param
    use cstran_mod, only: csphi_ensure_capacity
#if KEY_BLADE==1
    use blade_main, only: system_dirty
#endif
#if KEY_DOMDEC==1
    use domdec_common, only: domdec_system_changed
#endif
#if KEY_OPENMM==1
    use omm_ctrl, only: omm_system_changed
#endif
    implicit none
    integer(c_int), intent(in), value :: i_atom, j_atom, k_atom, l_atom, period
    real(c_double), intent(in), value :: force, min_angle, width

    consdata_add_dihe = -1

    ! Ensure capacity (this calls the internal allocation routine)
    ncsphi = ncsphi + 1
    call set_param('NCSP', ncsphi)
    call csphi_ensure_capacity(ncsphi)

    ! Set atom indices
    ics(ncsphi) = i_atom
    jcs(ncsphi) = j_atom
    kcs(ncsphi) = k_atom
    lcs(ncsphi) = l_atom
    iccs(ncsphi) = ncsphi

    ! Set parameters
    ccsc(ncsphi) = force
    ccsb(ncsphi) = min_angle
    ccsd(ncsphi) = period
    ccsw(ncsphi) = width

    ! Compute cos/sin for energy routines
    if (period == 0) then
       ccscos(ncsphi) = cos(min_angle)
       ccssin(ncsphi) = sin(min_angle)
    else
       ccscos(ncsphi) = cos(real(period, chm_real) * min_angle + pi)
       ccssin(ncsphi) = sin(real(period, chm_real) * min_angle + pi)
    end if

#if KEY_BLADE==1
    system_dirty = .true.
#endif
#if KEY_DOMDEC==1
    call domdec_system_changed()
#endif
#if KEY_OPENMM==1
    call omm_system_changed()
#endif

    consdata_add_dihe = ncsphi
  end function consdata_add_dihe

  !=========================================================================
  ! INTERNAL COORDINATE RESTRAINTS (CONS IC)
  !=========================================================================

  !> @brief Check if IC restraints are active
  !> @return 1 if active, 0 otherwise
  integer(c_int) function consdata_ic_is_active() bind(c)
    use cnst_fcm, only: lcic
    implicit none
    if (lcic) then
       consdata_ic_is_active = 1
    else
       consdata_ic_is_active = 0
    end if
  end function consdata_ic_is_active

  !> @brief Get IC restraint parameters
  !> @param[out] bond_force Bond force constant
  !> @param[out] angle_force Angle force constant
  !> @param[out] dihe_force Dihedral force constant
  !> @param[out] impr_force Improper force constant
  !> @param[out] exponent Bond potential exponent
  !> @param[out] upper 1 if upper limit only, 0 otherwise
  subroutine consdata_get_ic_params(bond_force, angle_force, dihe_force, &
       impr_force, exponent, upper) bind(c)
    use cnst_fcm, only: ccbic, cctic, ccpic, cciic, kbexpn, lupper
    implicit none
    real(c_double), intent(out) :: bond_force, angle_force, dihe_force, impr_force
    integer(c_int), intent(out) :: exponent, upper

    bond_force = ccbic
    angle_force = cctic
    dihe_force = ccpic
    impr_force = cciic
    exponent = kbexpn
    if (lupper) then
       upper = 1
    else
       upper = 0
    end if
  end subroutine consdata_get_ic_params

  !> @brief Set IC restraint parameters
  !> @param[in] bond_force Bond force constant (0 to disable)
  !> @param[in] angle_force Angle force constant (0 to disable)
  !> @param[in] dihe_force Dihedral force constant (0 to disable)
  !> @param[in] impr_force Improper force constant (0 to disable)
  !> @param[in] exponent Bond potential exponent (default 2)
  !> @param[in] upper 1 for upper limit only, 0 for both sides
  subroutine consdata_set_ic_params(bond_force, angle_force, dihe_force, &
       impr_force, exponent, upper) bind(c)
    use cnst_fcm, only: lcic, ccbic, cctic, ccpic, cciic, kbexpn, lupper
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
    real(c_double), intent(in), value :: bond_force, angle_force, dihe_force, impr_force
    integer(c_int), intent(in), value :: exponent, upper

    ccbic = bond_force
    cctic = angle_force
    ccpic = dihe_force
    cciic = impr_force
    kbexpn = exponent
    lupper = (upper /= 0)

    ! Set active flag if any force constant is non-zero
    lcic = (ccbic /= 0.0d0 .or. cctic /= 0.0d0 .or. &
            ccpic /= 0.0d0 .or. cciic /= 0.0d0)

#if KEY_BLADE==1
    system_dirty = .true.
#endif
#if KEY_OPENMM==1
    call omm_system_changed()
#endif
#if KEY_DOMDEC==1
    call domdec_system_changed()
#endif
  end subroutine consdata_set_ic_params

  !> @brief Clear IC restraints
  subroutine consdata_clear_ic() bind(c)
    use cnst_fcm, only: lcic, ccbic, cctic, ccpic, cciic
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

    lcic = .false.
    ccbic = 0.0d0
    cctic = 0.0d0
    ccpic = 0.0d0
    cciic = 0.0d0

#if KEY_BLADE==1
    system_dirty = .true.
#endif
#if KEY_OPENMM==1
    call omm_system_changed()
#endif
#if KEY_DOMDEC==1
    call domdec_system_changed()
#endif
  end subroutine consdata_clear_ic

  !=========================================================================
  ! DROPLET RESTRAINTS (CONS DROPLET)
  !=========================================================================

  !> @brief Check if droplet restraints are active
  !> @return 1 if active, 0 otherwise
  integer(c_int) function consdata_droplet_is_active() bind(c)
    use cnst_fcm, only: qqcnst
    implicit none
    if (qqcnst) then
       consdata_droplet_is_active = 1
    else
       consdata_droplet_is_active = 0
    end if
  end function consdata_droplet_is_active

  !> @brief Get droplet restraint parameters
  !> @param[out] force Force constant
  !> @param[out] exponent Exponent (default 4)
  !> @param[out] mass_weight 1 if mass weighted, 0 otherwise
  subroutine consdata_get_droplet_params(force, exponent, mass_weight) bind(c)
    use cnst_fcm, only: kqcnst, kqexpn, lqmass
    implicit none
    real(c_double), intent(out) :: force
    integer(c_int), intent(out) :: exponent, mass_weight

    force = kqcnst
    exponent = kqexpn
    if (lqmass) then
       mass_weight = 1
    else
       mass_weight = 0
    end if
  end subroutine consdata_get_droplet_params

  !> @brief Set droplet restraint parameters
  !> @param[in] force Force constant (0 to disable)
  !> @param[in] exponent Exponent (default 4)
  !> @param[in] mass_weight 1 for mass weighting, 0 otherwise
  subroutine consdata_set_droplet_params(force, exponent, mass_weight) bind(c)
    use cnst_fcm, only: qqcnst, kqcnst, kqexpn, lqmass
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
    real(c_double), intent(in), value :: force
    integer(c_int), intent(in), value :: exponent, mass_weight

    kqcnst = force
    kqexpn = exponent
    lqmass = (mass_weight /= 0)

    ! Set active flag
    qqcnst = (force /= 0.0d0)

#if KEY_BLADE==1
    system_dirty = .true.
#endif
#if KEY_OPENMM==1
    call omm_system_changed()
#endif
#if KEY_DOMDEC==1
    call domdec_system_changed()
#endif
  end subroutine consdata_set_droplet_params

  !> @brief Clear droplet restraints
  subroutine consdata_clear_droplet() bind(c)
    use cnst_fcm, only: qqcnst, kqcnst
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

    qqcnst = .false.
    kqcnst = 0.0d0

#if KEY_BLADE==1
    system_dirty = .true.
#endif
#if KEY_OPENMM==1
    call omm_system_changed()
#endif
#if KEY_DOMDEC==1
    call domdec_system_changed()
#endif
  end subroutine consdata_clear_droplet

end module api_cons
