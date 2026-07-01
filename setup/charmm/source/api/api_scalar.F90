!> routines for manipulating scalar atom properties
module api_scalar

  implicit none

contains

  !> @brief get the charges for the atoms
  !
  !> @param[out] out_charges vector of charges for the atoms
  !> @return success
  !>         1 <=> success
  function scalar_get_charges(out_charges) bind(c) result(success)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use api_util, only: f2c_logical
    use psf, only: natom, cg

    implicit none

    ! args
    real(c_double) :: out_charges(*)

    ! locals
    logical :: qsuccess

    ! result
    integer(c_int) :: success

    qsuccess = .false.

    out_charges(1:natom) = cg(1:natom)

    qsuccess = .true.
    success = f2c_logical(qsuccess)
  end function scalar_get_charges

  !> @brief set the charges for the atoms
  !
  !> @param[in] in_charges vector of charges for the atoms
  !> @param[in] selection if selection(i) == 1 then set atom i charge
  !> @return success
  !>         1 <=> success
  function scalar_set_charges(in_charges, selection) bind(c) result(success)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use api_util, only: f2c_logical
    use psf, only: natom, cg

    implicit none

    ! args
    real(c_double) :: in_charges(*)
    integer(c_int) :: selection(*)

    ! locals
    logical :: qsuccess
    integer :: i

    ! result
    integer(c_int) :: success

    qsuccess = .false.

    do i = 1, natom
       if (selection(i) == 1) cg(i) = in_charges(i)
    end do

    qsuccess = .true.
    success = f2c_logical(qsuccess)
  end function scalar_set_charges

  !> @brief get the masses for the atoms
  !
  !> @param[out] out_masses vector of masses for the atoms
  !> @return success
  !>         1 <=> success
  function scalar_get_masses(out_masses) bind(c) result(success)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use api_util, only: f2c_logical
    use psf, only: natom, amass

    implicit none

    ! args
    real(c_double) :: out_masses(*)

    ! locals
    logical :: qsuccess

    ! result
    integer(c_int) :: success

    qsuccess = .false.

    out_masses(1:natom) = amass(1:natom)

    qsuccess = .true.
    success = f2c_logical(qsuccess)
  end function scalar_get_masses

  !> @brief set the masses for the atoms
  !
  !> @param[in] in_masses vector of masses for the atoms
  !> @param[in] selection if selection(i) == 1 then set atom i masses
  !> @return success
  !>         1 <=> success
  function scalar_set_masses(in_masses, selection) bind(c) result(success)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use api_util, only: f2c_logical
    use psf, only: natom, amass

    implicit none

    ! args
    real(c_double) :: in_masses(*)
    integer(c_int) :: selection(*)

    ! locals
    logical :: qsuccess
    integer :: i

    ! result
    integer(c_int) :: success

    qsuccess = .false.

    do i = 1, natom
       if (selection(i) == 1) amass(i) = in_masses(i)
    end do

    qsuccess = .true.
    success = f2c_logical(qsuccess)
  end function scalar_set_masses

!> @brief get the friction coefs for the atoms
  !
  !> @param[out] out_fbetas vector of friction coefs for the atoms
  !> @return success
  !>         1 <=> success
  function scalar_get_fbetas(out_fbetas) bind(c) result(success)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use api_util, only: f2c_logical
    use cnst_fcm, only: allocate_cnst, fbeta
    use psf, only: natom

    implicit none

    ! args
    real(c_double) :: out_fbetas(*)

    ! locals
    logical :: qsuccess

    ! result
    integer(c_int) :: success

    qsuccess = .false.

    call allocate_cnst(natom)
    out_fbetas(1:natom) = fbeta(1:natom)

    qsuccess = .true.
    success = f2c_logical(qsuccess)
  end function scalar_get_fbetas

  !> @brief set the friction coefs for the atoms
  !
  !> @param[in] in_fbetas vector of friction coefs for the atoms
  !> @param[in] selection if selection(i) == 1 then set atom i friction coef
  !> @return success
  !>         1 <=> success
  function scalar_set_fbetas(in_fbetas, selection) bind(c) result(success)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use api_util, only: f2c_logical
    use cnst_fcm, only: allocate_cnst, fbeta
    use psf, only: natom

    implicit none

    ! args
    real(c_double) :: in_fbetas(*)
    integer(c_int) :: selection(*)

    ! locals
    logical :: qsuccess
    integer :: i

    ! result
    integer(c_int) :: success

    qsuccess = .false.

    call allocate_cnst(natom)

    do i = 1, natom
       if (selection(i) == 1) fbeta(i) = in_fbetas(i)
    end do

    qsuccess = .true.
    success = f2c_logical(qsuccess)
  end function scalar_set_fbetas

  !> @brief get the energy partition array
  !
  !> @param[out] out copy of energy partition array
  !> @return success
  !>         1 <=> success
  function scalar_get_econt(out) bind(c) result(success)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use api_util, only: f2c_logical
    use econtmod, only: econt
    use psf, only: natom

    implicit none

    real(c_double) :: out(*)  ! argument and output
    integer(c_int) :: success  ! result
    logical :: qsuccess  ! local var

    qsuccess = .false.
    if (allocated(econt)) then
       out(1:natom) = econt(1:natom)
       qsuccess = .true.
    end if

    success = f2c_logical(qsuccess)
  end function scalar_get_econt

  !> @brief set the energy partition array
  !
  !> @param[in] new_econt new energy partition array to use
  !> @param[in] selection if selection(i) == 1 then set atom i econt
  !> @return success
  !>         1 <=> success
  function scalar_set_econt(new_econt, selection) bind(c) result(success)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use api_util, only: f2c_logical
    use econtmod, only: econt
    use psf, only: natom

    implicit none

    ! args
    real(c_double) :: new_econt(*)
    integer(c_int) :: selection(*)

    ! locals
    logical :: qsuccess
    integer :: i

    ! result
    integer(c_int) :: success

    qsuccess = .false.

    if (.not. allocated(econt)) then
       success = f2c_logical(qsuccess)
       return
    end if

    do i = 1, natom
       if (selection(i) == 1) econt(i) = new_econt(i)
    end do

    qsuccess = .true.
    success = f2c_logical(qsuccess)
  end function scalar_set_econt

  !> @brief get the free energy difference atom partition
  !
  !> @param[out] output copy of pertepc
  !> @return success
  !>         1 <=> success
  function scalar_get_epcont(out) bind(c) result(success)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use api_util, only: f2c_logical
#if KEY_PERT == 1
    use pert, only: pertepc
#endif
    use psf, only: natom

    implicit none

    real(c_double) :: out(*)  ! argument and output
    integer(c_int) :: success  ! result
    logical :: qsuccess  ! local var

    qsuccess = .false.
#if KEY_PERT == 1
    if (allocated(pertepc)) then
       out(1:natom) = pertepc(1:natom)
       qsuccess = .true.
    end if
#endif /* KEY_PERT */
    success = f2c_logical(qsuccess)
  end function scalar_get_epcont

  !> @brief set the free energy difference atom partition
  !
  !> @param[in] new_vals new scalar values to use
  !> @param[in] selection if selection(i) == 1 then set atom i
  !> @return success
  !>         1 <=> success
  function scalar_set_epcont(new_vals, selection) bind(c) result(success)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use api_util, only: f2c_logical
#if KEY_PERT == 1
    use pert, only: pertepc
#endif
    use psf, only: natom

    implicit none

    ! args
    real(c_double) :: new_vals(*)
    integer(c_int) :: selection(*)

    ! locals
    logical :: qsuccess
    integer :: i

    ! result
    integer(c_int) :: success

    qsuccess = .false.
#if KEY_PERT == 1
    if (.not. allocated(pertepc)) then
       success = f2c_logical(qsuccess)
       return
    end if

    do i = 1, natom
       if (selection(i) == 1) pertepc(i) = new_vals(i)
    end do

    qsuccess = .true.
#endif /* KEY_PERT */
    success = f2c_logical(qsuccess)
  end function scalar_set_epcont

  !> @brief get the harmonic constraint constants
  !
  !> @param[out] output copy of kcnstr
  !> @return success
  !>         1 <=> success
  function scalar_get_constraints(out) bind(c) result(success)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use api_util, only: f2c_logical
    use cnst_fcm, only: allocate_cnst, kcnstr
    use psf, only: natom

    implicit none

    real(c_double) :: out(*)  ! argument and output
    integer(c_int) :: success  ! result
    logical :: qsuccess  ! local var

    qsuccess = .false.
    if (.not. allocated(kcnstr)) then
       call allocate_cnst(natom)
    end if

    out(1:natom) = kcnstr(1:natom)
    qsuccess = .true.

    success = f2c_logical(qsuccess)
  end function scalar_get_constraints

  !> @brief set the harmonic constraint constants
  !
  !> @param[in] new_vals new scalar values to use
  !> @param[in] selection if selection(i) == 1 then set atom i
  !> @return success
  !>         1 <=> success
  function scalar_set_constraints(new_vals, selection) bind(c) result(success)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use api_util, only: f2c_logical
    use cnst_fcm, only: allocate_cnst, kcnstr
    use psf, only: natom

    implicit none

    ! args
    real(c_double) :: new_vals(*)
    integer(c_int) :: selection(*)

    ! locals
    logical :: qsuccess
    integer :: i

    ! result
    integer(c_int) :: success

    qsuccess = .false.
    if (.not. allocated(kcnstr)) then
       call allocate_cnst(natom)
    end if

    do i = 1, natom
       if (selection(i) == 1) kcnstr(i) = new_vals(i)
    end do

    qsuccess = .true.
    success = f2c_logical(qsuccess)
  end function scalar_set_constraints

  !> @brief export a copy of imove: flags indicating which atoms move
  !
  !> @return integer(c_int) flags indicating which atoms move
  function scalar_get_move(out_flags) bind(c) result(success)
    use, intrinsic :: iso_c_binding, only: c_int
    use api_util, only: f2c_logical
    use psf, only: natom, imove

    implicit none

    integer(c_int), dimension(*) :: out_flags
    logical :: qsuccess
    ! result
    integer(c_int) :: success
    qsuccess = .false.

    out_flags(1:natom) = imove(1:natom)

    qsuccess = .true.
    success = f2c_logical(qsuccess)
  end function scalar_get_move

  !> @brief set the flags indicating which atoms move
  !
  !> @param[in] new_vals new scalar values to use
  !> @param[in] selection if selection(i) == 1 then set atom i
  !> @return success
  !>         1 <=> success
  function scalar_set_move(new_vals, selection) bind(c) result(success)
    use, intrinsic :: iso_c_binding, only: c_int
    use api_util, only: f2c_logical
    use psf, only: natom, imove

    implicit none

    ! args
    integer(c_int) :: new_vals(*), selection(*)

    ! locals
    logical :: qsuccess
    integer :: i

    ! result
    integer(c_int) :: success

    qsuccess = .false.

    do i = 1, natom
       if (selection(i) == 1) imove(i) = new_vals(i)
    end do

    qsuccess = .true.
    success = f2c_logical(qsuccess)
  end function scalar_set_move

  !> @brief export a copy of ignore: flags for ignoring atoms
  !
  !> @return integer(c_int) flags for ignoring atoms
  function scalar_get_ignore(out_flags) bind(c) result(success)
    use, intrinsic :: iso_c_binding, only: c_int
    use api_util, only: f2c_logical
    use psf, only: natom

#if KEY_ASPENER == 1
    use surface, only: ignore
#endif

    implicit none

    integer(c_int), dimension(*) :: out_flags
    logical :: qsuccess
    integer(c_int) :: success  ! result

    qsuccess = .false.
#if KEY_ASPENER == 1
    out_flags(1:natom) = ignore(1:natom)
    qsuccess = .true.
#endif  /* KEY_ASPENER */
    success = f2c_logical(qsuccess)
  end function scalar_get_ignore

  !> @brief set the flags indicating wich atoms ignore
  !
  !> @param[in] new_vals new scalar values to use
  !> @param[in] selection if selection(i) == 1 then set atom i
  !> @return success
  !>         1 <=> success
  function scalar_set_ignore(new_vals, selection) bind(c) result(success)
    use, intrinsic :: iso_c_binding, only: c_int
    use api_util, only: f2c_logical
    use psf, only: natom

#if KEY_ASPENER == 1
    use surface, only: ignore
#endif

    implicit none

    ! args
    integer(c_int) :: new_vals(*), selection(*)

    ! locals
    logical :: qsuccess
    integer :: i

    ! result
    integer(c_int) :: success

    qsuccess = .false.
#if KEY_ASPENER == 1
    do i = 1, natom
       if (selection(i) == 1) ignore(i) = new_vals(i)
    end do

    qsuccess = .true.
#endif  /* KEY_ASPENER */
    success = f2c_logical(qsuccess)
  end function scalar_set_ignore

  !> @brief export a copy of atomic solvation parameters
  !
  !> @return real(c_double) atomic solvation parameters
  function scalar_get_aspv(out_vals) bind(c) result(success)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use api_util, only: f2c_logical
    use psf, only: natom

#if KEY_ASPENER == 1
    use surface, only: aspv
#endif

    implicit none

    real(c_double) :: out_vals(*)
    logical :: qsuccess
    integer(c_int) :: success  ! result

    qsuccess = .false.
#if KEY_ASPENER == 1
    out_vals(1:natom) = aspv(1:natom)
    qsuccess = .true.
#endif  /* KEY_ASPENER */
    success = f2c_logical(qsuccess)
  end function scalar_get_aspv

  !> @brief set atomic solvation parameters
  !
  !> @param[in] new_vals new scalar values to use
  !> @param[in] selection if selection(i) == 1 then set atom i
  !> @return success
  !>         1 <=> success
  function scalar_set_aspv(new_vals, selection) bind(c) result(success)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use api_util, only: f2c_logical
    use psf, only: natom

#if KEY_ASPENER == 1
    use surface, only: aspv
#endif

    implicit none

    ! args
    real(c_double) :: new_vals(*)
    integer(c_int) :: selection(*)

    ! locals
    logical :: qsuccess
    integer :: i

    ! result
    integer(c_int) :: success

    qsuccess = .false.
#if KEY_ASPENER == 1
    do i = 1, natom
       if (selection(i) == 1) aspv(i) = new_vals(i)
    end do

    qsuccess = .true.
#endif  /* KEY_ASPENER */
    success = f2c_logical(qsuccess)
  end function scalar_set_aspv

  !> @brief get vdw radius for solvent energy, includes probe radius
  !
  !> @return real(c_double) atomic solvation parameters
  function scalar_get_vdw_surf(out_vals) bind(c) result(success)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use api_util, only: f2c_logical
    use psf, only: natom

#if KEY_ASPENER == 1
    use surface, only: vdw_surf
#endif

    implicit none

    real(c_double) :: out_vals(*)
    logical :: qsuccess
    integer(c_int) :: success  ! result

    qsuccess = .false.
#if KEY_ASPENER == 1
    out_vals(1:natom) = vdw_surf(1:natom)
    qsuccess = .true.
#endif  /* KEY_ASPENER */
    success = f2c_logical(qsuccess)
  end function scalar_get_vdw_surf

  !> @brief set vdw radius for solvent energy, includes probe radius
  !
  !> @param[in] new_vals new scalar values to use
  !> @param[in] selection if selection(i) == 1 then set atom i
  !> @return success
  !>         1 <=> success
  function scalar_set_vdw_surf(new_vals, selection) bind(c) result(success)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use api_util, only: f2c_logical
    use psf, only: natom

#if KEY_ASPENER == 1
    use surface, only: vdw_surf
#endif

    implicit none

    ! args
    real(c_double) :: new_vals(*)
    integer(c_int) :: selection(*)

    ! locals
    logical :: qsuccess
    integer :: i

    ! result
    integer(c_int) :: success

    qsuccess = .false.
#if KEY_ASPENER == 1
    do i = 1, natom
       if (selection(i) == 1) vdw_surf(i) = new_vals(i)
    end do

    qsuccess = .true.
#endif  /* KEY_ASPENER */
    success = f2c_logical(qsuccess)
  end function scalar_set_vdw_surf

  !> @brief get radius scale factor for nonbonded (vdw)
  !
  !> @return real(c_double) radius scale factors
  function scalar_get_rscale(out_vals) bind(c) result(success)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use api_util, only: f2c_logical
    use psf, only: natom, rsclf

    implicit none

    real(c_double) :: out_vals(*)
    logical :: qsuccess
    integer(c_int) :: success  ! result

    qsuccess = .false.
    out_vals(1:natom) = sqrt(abs(rsclf(1:natom)))
    qsuccess = .true.
    success = f2c_logical(qsuccess)
  end function scalar_get_rscale

  !> @brief set radius scale factor for nonbonded (vdw)
  !
  !> @param[in] new_vals new scalar values to use
  !> @param[in] selection if selection(i) == 1 then set atom i
  !> @return success
  !>         1 <=> success
  function scalar_set_rscale(new_vals, selection) bind(c) result(success)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use api_util, only: f2c_logical
    use psf, only: natom, rsclf

    implicit none

    ! args
    real(c_double) :: new_vals(*)
    integer(c_int) :: selection(*)

    ! locals
    logical :: qsuccess
    integer :: i

    ! result
    integer(c_int) :: success

    qsuccess = .false.

    do i = 1, natom
       if (selection(i) == 1) rsclf(i) = new_vals(i) * new_vals(i)
    end do

    qsuccess = .true.
    success = f2c_logical(qsuccess)
  end function scalar_set_rscale

  !> @brief get Weeks, Chandler, Anderson decomp of Lennard-Jones Potential
  !
  !> @return real(c_double) atomic solvation parameters
  function scalar_get_wcad(out_vals) bind(c) result(success)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use api_util, only: f2c_logical
#if KEY_WCA == 1
    use psf, only: natom, wca
#endif

    implicit none

    real(c_double) :: out_vals(*)
    logical :: qsuccess
    integer(c_int) :: success  ! result

    qsuccess = .false.
#if KEY_WCA == 1
    out_vals(1:natom) = wca(1:natom)
    qsuccess = .true.
#endif  /* KEY_WCA */
    success = f2c_logical(qsuccess)
  end function scalar_get_wcad

  !> @brief set Weeks, Chandler, Anderson decomp of Lennard-Jones Potential
  !
  !> @param[in] new_vals new scalar values to use
  !> @param[in] selection if selection(i) == 1 then set atom i
  !> @return success
  !>         1 <=> success
  function scalar_set_wcad(new_vals, selection) bind(c) result(success)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use api_util, only: f2c_logical
#if KEY_WCA == 1
    use psf, only: natom, wca
#endif

    implicit none

    ! args
    real(c_double) :: new_vals(*)
    integer(c_int) :: selection(*)

    ! locals
    logical :: qsuccess
    integer :: i

    ! result
    integer(c_int) :: success

    qsuccess = .false.
#if KEY_WCA == 1
    do i = 1, natom
       if (selection(i) == 1) wca(i) = new_vals(i)
    end do

    qsuccess = .true.
#endif  /* KEY_WCA */
    success = f2c_logical(qsuccess)
  end function scalar_set_wcad

  !> @brief get atom polarizability
  !
  !> @return real(c_double) atom polarizability
  function scalar_get_alpha(out_alpha) bind(c) result(success)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use api_util, only: f2c_logical
    use param, only: itc, alp
    use psf, only: natom, iac

    implicit none

    real(c_double) :: out_alpha(*)
    logical :: qsuccess
    integer(c_int) :: success  ! result

    qsuccess = .false.
    out_alpha(1:natom) = alp(itc(iac(1:natom)))
    qsuccess = .true.
    success = f2c_logical(qsuccess)
  end function scalar_get_alpha

  !> @brief get effective number of electrons
  !
  !> @return real(c_double) effective number of electrons
  function scalar_get_effect(out_eff) bind(c) result(success)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use api_util, only: f2c_logical
    use param, only: itc, eff
    use psf, only: natom, iac

    implicit none

    real(c_double) :: out_eff(*)
    logical :: qsuccess
    integer(c_int) :: success  ! result

    qsuccess = .false.
    out_eff(1:natom) = eff(itc(iac(1:natom)))
    qsuccess = .true.
    success = f2c_logical(qsuccess)
  end function scalar_get_effect

  !> @brief get van der Waals radii
  !
  !> @return real(c_double) van der Waals radii
  function scalar_get_radius(out_vdwr) bind(c) result(success)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use api_util, only: f2c_logical
    use param, only: itc, vdwr
    use psf, only: natom, iac

    implicit none

    real(c_double) :: out_vdwr(*)
    logical :: qsuccess
    integer(c_int) :: success  ! result

    qsuccess = .false.
    out_vdwr(1:natom) = vdwr(itc(iac(1:natom)))
    qsuccess = .true.
    success = f2c_logical(qsuccess)
  end function scalar_get_radius

  !> @brief get FQ Slater orbital principal quantum number
  !
  !> @return real(c_double ) FQ Slater orbital principal quantum number
  function scalar_get_fqprin(out_fqprin) bind(c) result(success)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use api_util, only: f2c_logical
#if KEY_FLUCQ == 1
    use param, only: fqprin
#endif
    use psf, only: natom, iac

    implicit none

    real(c_double) :: out_fqprin(*)
    logical :: qsuccess
    integer(c_int) :: success  ! result

    qsuccess = .false.
#if KEY_FLUCQ == 1
    out_fqprin(1:natom) = fqprin(iac(1:natom))
    qsuccess = .true.
#endif
    success = f2c_logical(qsuccess)
  end function scalar_get_fqprin

  !> @brief get FQ Slater orbital exponent
  !
  !> @return real(c_double ) FQ Slater orbital exponent
  function scalar_get_fqzeta(out_fqzeta) bind(c) result(success)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use api_util, only: f2c_logical
#if KEY_FLUCQ == 1
    use param, only: fqzeta
#endif
    use psf, only: natom, iac

    implicit none

    real(c_double) :: out_fqzeta(*)
    logical :: qsuccess
    integer(c_int) :: success  ! result

    qsuccess = .false.
#if KEY_FLUCQ == 1
    out_fqzeta(1:natom) = fqzeta(iac(1:natom))
    qsuccess = .true.
#endif
    success = f2c_logical(qsuccess)
  end function scalar_get_fqzeta

  !> @brief get FQ electronegativity parameter
  !
  !> @return real(c_double ) FQ electronegativity parameter
  function scalar_get_fqchi(out_fqchi) bind(c) result(success)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use api_util, only: f2c_logical
#if KEY_FLUCQ == 1
    use param, only: fqchi
#endif
    use psf, only: natom, iac

    implicit none

    real(c_double) :: out_fqchi(*)
    logical :: qsuccess
    integer(c_int) :: success  ! result

    qsuccess = .false.
#if KEY_FLUCQ == 1
    out_fqchi(1:natom) = fqchi(iac(1:natom))
    qsuccess = .true.
#endif
    success = f2c_logical(qsuccess)
  end function scalar_get_fqchi

  !> @brief get FQ charge mass
  !
  !> @return real(c_double ) FQ charge mass
  function scalar_get_fqmass(out_fqmass) bind(c) result(success)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use api_util, only: f2c_logical
#if KEY_FLUCQ == 1
    use param, only: fqchma
#endif
    use psf, only: natom, iac

    implicit none

    real(c_double) :: out_fqmass(*)
    logical :: qsuccess
    integer(c_int) :: success  ! result

    qsuccess = .false.
#if KEY_FLUCQ == 1
    out_fqmass(1:natom) = fqchma(iac(1:natom))
    qsuccess = .true.
#endif
    success = f2c_logical(qsuccess)
  end function scalar_get_fqmass

  !> @brief get FQ self-interaction
  !
  !> @return real(c_double ) FQ self-interaction
  function scalar_get_fqjz(out_fqjz) bind(c) result(success)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use api_util, only: f2c_logical
#if KEY_FLUCQ == 1
    use param, only: fqjz
#endif
    use psf, only: natom, iac

    implicit none

    real(c_double) :: out_fqjz(*)
    logical :: qsuccess
    integer(c_int) :: success  ! result

    qsuccess = .false.
#if KEY_FLUCQ == 1
    out_fqjz(1:natom) = fqjz(iac(1:natom))
    qsuccess = .true.
#endif
    success = f2c_logical(qsuccess)
  end function scalar_get_fqjz

  !> @brief get FQ charge force
  !
  !> @return real(c_double ) FQ charge force
  function scalar_get_fqcforce(out_fqcforce) bind(c) result(success)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use api_util, only: f2c_logical
#if KEY_FLUCQ == 1
    use flucqm, only: fqcfor
#endif
    use psf, only: natom

    implicit none

    real(c_double) :: out_fqcforce(*)
    logical :: qsuccess
    integer(c_int) :: success  ! result

    qsuccess = .false.
#if KEY_FLUCQ == 1
    out_fqcforce(1:natom) = fqcfor(1:natom)
    qsuccess = .true.
#endif
    success = f2c_logical(qsuccess)
  end function scalar_get_fqcforce

  !> @brief set FQ charge force
  !
  !> @param[in] new_vals new scalar values to use
  !> @param[in] selection if selection(i) == 1 then set atom i
  !> @return success
  !>         1 <=> success
  function scalar_set_fqcforce(new_vals, selection) bind(c) result(success)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use api_util, only: f2c_logical
    use psf, only: natom
#if KEY_FLUCQ == 1
    use flucqm, only: fqcfor
#endif

    implicit none

    ! args
    real(c_double) :: new_vals(*)
    integer(c_int) :: selection(*)

    ! locals
    logical :: qsuccess
    integer :: i

    ! result
    integer(c_int) :: success

    qsuccess = .false.
#if KEY_FLUCQ == 1
    do i = 1, natom
       if (selection(i) == 1) fqcfor(i) = new_vals(i)
    end do

    qsuccess = .true.
#endif  /* KEY_FLUCQ */
    success = f2c_logical(qsuccess)
  end function scalar_set_fqcforce

  !> @brief get FlucQ charges from last timestep
  !
  !> @return real(c_double ) FlucQ charges from last timestep
  function scalar_get_fqold(out_fqold) bind(c) result(success)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use api_util, only: f2c_logical
#if KEY_FLUCQ == 1
    use flucq, only: fqoldq
#endif
    use psf, only: natom

    implicit none

    real(c_double) :: out_fqold(*)
    logical :: qsuccess
    integer(c_int) :: success  ! result

    qsuccess = .false.
#if KEY_FLUCQ == 1
    out_fqold(1:natom) = fqoldq(1:natom)
    qsuccess = .true.
#endif
    success = f2c_logical(qsuccess)
  end function scalar_get_fqold

  !> @brief set FlucQ charges from last timestep
  !
  !> @param[in] new_vals new scalar values to use
  !> @param[in] selection if selection(i) == 1 then set atom i
  !> @return success
  !>         1 <=> success
  function scalar_set_fqold(new_vals, selection) bind(c) result(success)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use api_util, only: f2c_logical
    use psf, only: natom
#if KEY_FLUCQ == 1
    use flucq, only: fqoldq
#endif

    implicit none

    ! args
    real(c_double) :: new_vals(*)
    integer(c_int) :: selection(*)

    ! locals
    logical :: qsuccess
    integer :: i

    ! result
    integer(c_int) :: success

    qsuccess = .false.
#if KEY_FLUCQ == 1
    do i = 1, natom
       if (selection(i) == 1) fqoldq(i) = new_vals(i)
    end do

    qsuccess = .true.
#endif  /* KEY_FLUCQ */
    success = f2c_logical(qsuccess)
  end function scalar_set_fqold

  !> @brief get variable cutoffs of LJ interaction depending on atom types
  !
  !> @return real(c_double ) variable cutoffs of LJ interaction
  function scalar_get_varc(out_varc) bind(c) result(success)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use api_util, only: f2c_logical
    use varcutm, only: get_varcut
    use psf, only: natom

    implicit none

    real(c_double) :: out_varc(*), buf(1:natom)
    logical :: qsuccess
    integer(c_int) :: success  ! result

    qsuccess = .false.
    call get_varcut(buf, natom)
    out_varc(1:natom) = buf(1:natom)
    qsuccess = .true.
    success = f2c_logical(qsuccess)
  end function scalar_get_varc

  !> @brief set variable cutoffs of LJ interaction depending on atom types
  !
  !> @param[in] new_vals new scalar values to use
  !> @param[in] selection if selection(i) == 1 then set atom i
  !> @return success
  !>         1 <=> success
  function scalar_set_varc(new_vals, selection) bind(c) result(success)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use api_util, only: f2c_logical
    use psf, only: natom
    use varcutm, only: varcut

    implicit none

    ! args
    real(c_double) :: new_vals(*)
    integer(c_int) :: selection(*)

    ! locals
    logical :: qsuccess
    integer :: i

    ! result
    integer(c_int) :: success


    qsuccess = .false.
    if (.not. allocated(varcut)) then
       success = f2c_logical(qsuccess)
       return
    end if

    do i = 1, natom
       if (selection(i) == 1) varcut(i) = new_vals(i)
    end do

    qsuccess = .true.
    success = f2c_logical(qsuccess)
  end function scalar_set_varc

  !> @brief get self-guiding weights for SGLD simulation
  !
  !> @return real(c_double ) Self-guiding weights for SGLD simulation
  function scalar_get_sgwt(out_sgwt) bind(c) result(success)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use api_util, only: f2c_logical
#if KEY_SGLD == 1
    use sgld, only: sgwt, allocate_sgwt
#endif
    use psf, only: natom

    implicit none

    real(c_double) :: out_sgwt(*)
    logical :: qsuccess
    integer(c_int) :: success  ! result

    qsuccess = .false.
#if KEY_SGLD == 1
    if (.not. allocated(sgwt)) then
       call allocate_sgwt(natom)
    end if
    out_sgwt(1:natom) = sgwt(1:natom)
    qsuccess = .true.
#endif
    success = f2c_logical(qsuccess)
  end function scalar_get_sgwt

  !> @brief set self-guiding weights for SGLD simulation
  !
  !> @param[in] new_vals new scalar values to use
  !> @param[in] selection if selection(i) == 1 then set atom i
  !> @return success
  !>         1 <=> success
  function scalar_set_sgwt(new_vals, selection) bind(c) result(success)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use api_util, only: f2c_logical
    use psf, only: natom
#if KEY_SGLD == 1
    use sgld, only: sgwt, allocate_sgwt
#endif

    implicit none

    ! args
    real(c_double) :: new_vals(*)
    integer(c_int) :: selection(*)

    ! locals
    logical :: qsuccess
    integer :: i

    ! result
    integer(c_int) :: success

    qsuccess = .false.
#if KEY_SGLD == 1
    if (.not. allocated(sgwt)) then
       call allocate_sgwt(natom)
    end if

    do i = 1, natom
       if (selection(i) == 1) sgwt(i) = new_vals(i)
    end do

    qsuccess = .true.
#endif  /* KEY_SGLD */
    success = f2c_logical(qsuccess)
  end function scalar_set_sgwt

  !> @brief get apparent friction constants for SGMD/SGLD simulation
  !
  !> @return real(c_double ) apparent friction constants
  function scalar_get_sggamma(out_sggamma) bind(c) result(success)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use api_util, only: f2c_logical
#if KEY_SGLD == 1
    use sgld, only: sggamma, allocate_sggamma
#endif
    use psf, only: natom

    implicit none

    real(c_double) :: out_sggamma(*)
    logical :: qsuccess
    integer(c_int) :: success  ! result

    qsuccess = .false.
#if KEY_SGLD == 1
    if (.not. allocated(sggamma)) then
       call allocate_sggamma(natom)
    end if
    out_sggamma(1:natom) = sggamma(1:natom)
    qsuccess = .true.
#endif
    success = f2c_logical(qsuccess)
  end function scalar_get_sggamma

  !> @brief set apparent friction constants for SGMD/SGLD simulation
  !
  !> @param[in] new_vals new scalar values to use
  !> @param[in] selection if selection(i) == 1 then set atom i
  !> @return success
  !>         1 <=> success
  function scalar_set_sggamma(new_vals, selection) bind(c) result(success)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use api_util, only: f2c_logical
    use psf, only: natom
#if KEY_SGLD == 1
    use sgld, only: sggamma, allocate_sggamma
#endif

    implicit none

    ! args
    real(c_double) :: new_vals(*)
    integer(c_int) :: selection(*)

    ! locals
    logical :: qsuccess
    integer :: i

    ! result
    integer(c_int) :: success

    qsuccess = .false.
#if KEY_SGLD == 1
    if (.not. allocated(sggamma)) then
       call allocate_sggamma(natom)
    end if

    do i = 1, natom
       if (selection(i) == 1) sggamma(i) = new_vals(i)
    end do

    qsuccess = .true.
#endif  /* KEY_SGLD */
    success = f2c_logical(qsuccess)
  end function scalar_set_sggamma
end module api_scalar
