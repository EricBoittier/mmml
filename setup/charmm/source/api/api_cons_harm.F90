!> routines for applying constraints on atoms
module api_cons_harm
  use, intrinsic :: iso_c_binding, only: c_int, c_double

  implicit none

  type, bind(c) :: cons_harm_opts_t
     ! absolute constrains options
     integer(c_int) :: expo
     real(c_double) :: x_scale, y_scale, z_scale

     ! relative constrains options
     integer(c_int) :: q_no_rot, q_no_trans

     ! force constant options
     integer(c_int) :: q_mass, q_weight
     real(c_double) :: force_const
  end type cons_harm_opts_t

contains

  !> @brief deactivate harmonic constraints
  !
  ! this method will also clear the settings
  !
  !> @return cons_harm_turn_off
  !>         1 <=> success
  integer(c_int) function cons_harm_turn_off() bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use cstran_mod, only: clear_harm_const
    implicit none
    cons_harm_turn_off = -1
    call clear_harm_const()
    cons_harm_turn_off = 1
  end function cons_harm_turn_off

  !> @brief set up absolute harmonic constraints
  !
  ! EC = sum over selected atoms of k(i)* [mass(i)] * (x(i)-refx(i))**exponent
  !
  !> @param[in] selection selection(i) .eq. 1 <=> constrain atom i
  !> @param[in] icomp 1 <=> use comparison set
  !> @param[in] opts options for configuring harmonic contraints
  !> @return success 1 <=> no error
  function cons_harm_setup_absolute( &
       selection, icomp, opts) bind(c) result(success)
    use, intrinsic :: iso_c_binding, only: c_int
    use api_util, only: c2f_logical, f2c_logical
    use coord, only: x, y, z, wmain
    use coordc, only: xcomp, ycomp, zcomp, wcomp
    use cnst_fcm, only: allocate_cnst
    use cstran_mod, only: &
         abs_opts_t, force_const_t, &
         setup_harm_const
    use psf, only: natom

    implicit none

    ! args
    integer(c_int) :: selection(*), icomp
    type(cons_harm_opts_t) :: opts

    ! result
    integer(c_int) :: success

    ! locals
    type(abs_opts_t) :: abs_opts
    type(force_const_t) :: force_const_opts
    logical :: qcomp, qsuccess

    integer :: jslct(natom)

    success = -1

    jslct = 0
    qcomp = c2f_logical(icomp)

    abs_opts%expo = opts%expo
    abs_opts%x_scale = opts%x_scale
    abs_opts%y_scale = opts%y_scale
    abs_opts%z_scale = opts%z_scale

    force_const_opts%force_const = opts%force_const
    force_const_opts%mass = c2f_logical(opts%q_mass)
    force_const_opts%weight = c2f_logical(opts%q_weight)

    call allocate_cnst(natom)

    if (qcomp) then
       qsuccess = setup_harm_const( &
            xcomp, ycomp, zcomp, wcomp, &
            x, y, z, wmain, &
            .true., 0, qcomp, &  ! absolute harmonic constraints
            selection, jslct, &
            force_const_opts, abs_opts)
    else
       qsuccess = setup_harm_const( &
            x, y, z, wmain, &
            xcomp, ycomp, zcomp, wcomp, &
            .true., 0, qcomp, &  ! absolute harmonic constraints
            selection, jslct, &
            force_const_opts, abs_opts)
    end if
    success = f2c_logical(qsuccess)
  end function cons_harm_setup_absolute


  !> @brief set up absolute harmonic constraints
  !
  ! EC = sum over selected atoms of k(i)* [mass(i)] * (x(i)-refx(i))**exponent
  !
  !> @param[in] selection selection(i) .eq. 1 <=> constrain atom i
  !> @param[in] icomp 1 <=> use comparison set
  !> @param[in] opts options for configuring harmonic contraints
  !> @return success 1 <=> no error
  function cons_harm_setup_pca( &
       selection, icomp, opts) bind(c) result(success)
    use, intrinsic :: iso_c_binding, only: c_int
    use api_util, only: c2f_logical, f2c_logical
    use coord, only: x, y, z, wmain
    use coordc, only: xcomp, ycomp, zcomp, wcomp
    use cnst_fcm, only: allocate_cnst
    use cstran_mod, only: &
         abs_opts_t, force_const_t, &
         setup_harm_const
    use psf, only: natom

    implicit none

    ! args
    integer(c_int) :: selection(*), icomp
    type(cons_harm_opts_t) :: opts

    ! result
    integer(c_int) :: success

    ! locals
    type(abs_opts_t) :: abs_opts
    type(force_const_t) :: force_const_opts
    logical :: qcomp, qsuccess

    integer :: jslct(natom)

    success = -1

    jslct = 0
    qcomp = c2f_logical(icomp)

    abs_opts%expo = opts%expo
    abs_opts%x_scale = opts%x_scale
    abs_opts%y_scale = opts%y_scale
    abs_opts%z_scale = opts%z_scale

    force_const_opts%force_const = opts%force_const
    force_const_opts%mass = c2f_logical(opts%q_mass)
    force_const_opts%weight = c2f_logical(opts%q_weight)

    call allocate_cnst(natom)

    if (qcomp) then
       qsuccess = setup_harm_const( &
            xcomp, ycomp, zcomp, wcomp, &
            x, y, z, wmain, &
            .true., 3, qcomp, &  ! hrtyp = 3 => pca harmonic constraints
            selection, jslct, &
            force_const_opts, abs_opts)
    else
       qsuccess = setup_harm_const( &
            x, y, z, wmain, &
            xcomp, ycomp, zcomp, wcomp, &
            .true., 3, qcomp, &  ! hrtyp = 3 => pca harmonic constraints
            selection, jslct, &
            force_const_opts, abs_opts)
    end if
    success = f2c_logical(qsuccess)
  end function cons_harm_setup_pca


  !> @brief set up best fit harmonic constraints
  !
  ! EC = sum over selected atoms of k(i)* [mass(i)] * (x(i)-refx(i))**exponent
  !
  !> @param[in] selection selection(i) .eq. 1 <=> constrain atom i
  !> @param[in] icomp 1 <=> use comparison set
  !> @param[in] opts options for configuring harmonic contraints
  !> @return success 1 <=> no error
  function cons_harm_setup_best_fit( &
       selection, icomp, opts) bind(c) result(success)
    use, intrinsic :: iso_c_binding, only: c_int
    use api_util, only: c2f_logical, f2c_logical
    use coord, only: x, y, z, wmain
    use coordc, only: xcomp, ycomp, zcomp, wcomp
    use cnst_fcm, only: allocate_cnst
    use cstran_mod, only: &
         relative_opts_t, force_const_t, &
         setup_harm_const
    use psf, only: natom

    implicit none

    ! args
    integer(c_int) :: selection(*), icomp
    type(cons_harm_opts_t) :: opts

    ! result
    integer(c_int) :: success

    ! locals
    type(relative_opts_t) :: rel_opts
    type(force_const_t) :: force_const_opts
    logical :: qcomp, qsuccess

    integer :: jslct(natom)

    success = -1

    jslct = 0
    qcomp = c2f_logical(icomp)

    rel_opts%no_rot = c2f_logical(opts%q_no_rot)
    rel_opts%no_trans = c2f_logical(opts%q_no_trans)

    force_const_opts%force_const = opts%force_const
    force_const_opts%mass = c2f_logical(opts%q_mass)
    force_const_opts%weight = c2f_logical(opts%q_weight)

    call allocate_cnst(natom)

    if (qcomp) then
       qsuccess = setup_harm_const( &
            xcomp, ycomp, zcomp, wcomp, &
            x, y, z, wmain, &
            .true., 1, qcomp, &  ! best fit harmonic constraints
            selection, jslct, &
            force_const_opts, relative_opts=rel_opts)
    else
       qsuccess = setup_harm_const( &
            x, y, z, wmain, &
            xcomp, ycomp, zcomp, wcomp, &
            .true., 1, qcomp, &  ! best fit harmonic constraints
            selection, jslct, &
            force_const_opts, relative_opts=rel_opts)
    end if
    success = f2c_logical(qsuccess)
  end function cons_harm_setup_best_fit


  !> @brief set up relative harmonic constraints
  !
  ! EC = sum over selected atoms of k(i)* [mass(i)] * (x(i)-refx(i))**exponent
  !
  !> @param[in] selection selection(i) .eq. 1 <=> constrain atom i
  !> @param[in] icomp 1 <=> use comparison set
  !> @param[in] opts options for configuring harmonic contraints
  !> @return success 1 <=> no error
  function cons_harm_setup_relative( &
       iselection, jselection, icomp, opts) bind(c) result(success)
    use, intrinsic :: iso_c_binding, only: c_int
    use api_util, only: c2f_logical, f2c_logical
    use coord, only: x, y, z, wmain
    use coordc, only: xcomp, ycomp, zcomp, wcomp
    use cnst_fcm, only: allocate_cnst
    use cstran_mod, only: &
         relative_opts_t, force_const_t, &
         setup_harm_const
    use psf, only: natom

    implicit none

    ! args
    integer(c_int) :: iselection(*), jselection(*), icomp
    type(cons_harm_opts_t) :: opts

    ! result
    integer(c_int) :: success

    ! locals
    type(relative_opts_t) :: rel_opts
    type(force_const_t) :: force_const_opts
    logical :: qcomp, qsuccess

    success = -1

    qcomp = c2f_logical(icomp)

    rel_opts%no_rot = c2f_logical(opts%q_no_rot)
    rel_opts%no_trans = c2f_logical(opts%q_no_trans)

    force_const_opts%force_const = opts%force_const
    force_const_opts%mass = c2f_logical(opts%q_mass)
    force_const_opts%weight = c2f_logical(opts%q_weight)

    call allocate_cnst(natom)

    if (qcomp) then
       qsuccess = setup_harm_const( &
            xcomp, ycomp, zcomp, wcomp, &
            x, y, z, wmain, &
            .false., 2, qcomp, &  ! relative harmonic constraints
            iselection, jselection, &
            force_const_opts, relative_opts=rel_opts)
    else
       qsuccess = setup_harm_const( &
            x, y, z, wmain, &
            xcomp, ycomp, zcomp, wcomp, &
            .false., 2, qcomp, &  ! relative harmonic constraints
            iselection, jselection, &
            force_const_opts, relative_opts=rel_opts)
    end if
    success = f2c_logical(qsuccess)
  end function cons_harm_setup_relative


  !> @brief set up the force constant options
  !
  ! weight .eq. 1 <=> force param not used
  ! weight .eq. 1 <=> weight array elt. i used to set force const k(i)
  !
  !> @param[in] force_const restraint force constant k
  !> @param[in] imass k * mass of atom for oscillation freq sqrt(k)
  !> @param[in] iweight use weight array for k(i) and ignore force_const param
  !> @return cons_harm_set_force_const
  !>         1 <=> success
  ! integer(c_int) function cons_harm_set_force_const( &
  !      force_const, imass, iweight) bind(c)
  !   use, intrinsic :: iso_c_binding, only: c_int, c_double
  !   use api_util, only: util_convert_logical_c2f

  !   implicit none

  !   ! args
  !   real(c_double) :: force_const
  !   integer(c_int) :: imass, iweight

  !   ! locals
  !   logical :: qmass, qweight

  !   cons_harm_set_force_const = -1

  !   qmass = util_convert_logical_c2f(imass)
  !   qweight = util_convert_logical_c2f(iweight)

  !   cons_harm_set_force_const = 1
  ! end function cons_harm_set_force_const

  !> @brief set up the reference coordinates for harmonic constraints
  !
  !> @param[in] icomp use comparison set for reference coordinates?
  !> @param[in] ikeep use ref coords from previous restraints?
  !> @return integer(c_int) 1 <=> success
  ! integer(c_int) function cons_harm_set_ref_coords(icomp, ikeep) bind(c)
  !   use, intrinsic :: iso_c_binding, only: c_int
  !   use api_util, only: util_convert_logical_c2f

  !   implicit none

  !   ! args
  !   integer(c_int) :: icomp, ikeep

  !   ! locals
  !   logical :: qcomp, qkeep

  !   cons_harm_set_ref_coords = -1

  !   qcomp = util_convert_logical_c2f(icomp)
  !   qkeep = util_convert_logical_c2f(ikeep)

  !   cons_harm_set_ref_coords = 1
  ! end function cons_harm_set_ref_coords

  !> @brief set up absolute harmonic constraints
  !
  ! EC = sum over selected atoms of k(i)* [mass(i)] * (x(i)-refx(i))**exponent
  !
  !> @param[in] selection selection(i) .eq. 1 <=> constrain atom i
  !> @param[in] exponent exponent on difference between atom and ref atom
  !> @param[in] x_scale global scale factor for the x component
  !> @param[in] y_scale global scale factor for the y component
  !> @param[in] z_scale global scale factor for the z component
  !> @return cons_harm_use_absolute 1 <=> success
  ! integer(c_int) function cons_harm_use_absolute( &
  !      selection, exponent, x_scale, y_scale, z_scale) bind(c)
  !   use, intrinsic :: iso_c_binding, only: c_int, c_double

  !   implicit none

  !   ! args
  !   integer(c_int) :: selection(*), exponent
  !   real(c_double) :: x_scale, y_scale, z_scale

  !   cons_harm_use_absolute = -1

  !   cons_harm_use_absolute = 1
  ! end function cons_harm_use_absolute

end module api_cons_harm
