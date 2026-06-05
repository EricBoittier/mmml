!> export copies of param data structures
module api_param
  implicit none
contains

  !> @brief get the size of the atc array (see param_get_atc)
  !
  !> @return integer(c_int) number of atc entries
  function param_get_natc() bind(c) result(n)
    use, intrinsic :: iso_c_binding, only: c_int
    use param, only: natc
    implicit none
    integer(c_int) :: n
    n = natc
  end function param_get_natc

  !> @brief export a copy of atc (chem names for atom type codes)
  !
  ! atc has natc entries and each entry is a string of 8 chars
  !
  !> @param[out] out_atc copy of atc array
  subroutine param_get_atc(out_atc) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_ptr
    use api_util, only: f2c_string
    use param, only: natc, atc

    implicit none

    type(c_ptr), target, dimension(*) :: out_atc
    integer :: i, nchars = 8

    do i = 1, natc
       nchars = len_trim(atc(i))
       call f2c_string(atc(i), out_atc(i), nchars)
    end do
  end subroutine param_get_atc

  !> @brief fills the out_* with atom charges 
  !
  !> @param[out] out_charge charge of atoms, 1:natom
  !> @returns integer(c_int) number of atoms copied to output parameters
  integer(c_int) function param_get_charge(out_charge) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use psf, only: natom, cg

    implicit none

    real(c_double), dimension(*), intent(out) :: out_charge

    out_charge(1:natom) = cg(1:natom)

    param_get_charge = natom
  end function param_get_charge 

  !> @brief fills the out_* with atom vdw radius 
  !
  !> @param[out] out_vdwr vdw radius of atoms, 1:natom
  !> @returns integer(c_int) number of atoms copied to output parameters
  integer(c_int) function param_get_vdwr(out_vdwr) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use param, only: vdwr, itc
    use psf, only: natom, iac

    implicit none

    real(c_double), dimension(*), intent(out) :: out_vdwr

    out_vdwr(1:natom) = vdwr(itc(iac(1:natom)))

    param_get_vdwr = natom
  end function param_get_vdwr

  !> @brief fills the out_* with atom epsilon 
  !
  !> @param[out] out_eps epsilon of atoms, 1:natom
  !> @returns integer(c_int) number of atoms copied to output parameters
  integer(c_int) function param_get_epsilon(out_epsilon) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use param, only: eff, itc
    use psf, only: natom, iac

    implicit none

    real(c_double), dimension(*), intent(out) :: out_epsilon

    out_epsilon(1:natom) = eff(itc(iac(1:natom)))

    param_get_epsilon = natom
  end function param_get_epsilon

end module api_param
