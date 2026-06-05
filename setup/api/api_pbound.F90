!> export settings related to periodic boundary conditions
module api_pbound
  implicit none
contains

#if KEY_PBOUND == 1
  !> @brief export boxinv, boyinv, bozinv parameters
  !
  !> @param[out] x current value of boxinv
  !> @param[out] y current value of boyinv
  !> @param[out] z current value of bozinv
  subroutine pbound_get_boxinv(x, y, z) bind(c)
    use, intrinsic :: iso_c_binding, only: c_double
    use pbound, only: boxinv, boyinv, bozinv
    implicit none
    real(c_double) :: x, y, z
    x = boxinv
    y = boyinv
    z = bozinv
  end subroutine pbound_get_boxinv

  !> @brief export xsize, ysize, zsize
  !
  !> @param[out] x current value of xsize
  !> @param[out] y current value of ysize
  !> @param[out] z current value of zsize
  subroutine pbound_get_size(x, y, z) bind(c)
    use, intrinsic :: iso_c_binding, only: c_double
    use pbound, only: xsize, ysize, zsize
    implicit none
    real(c_double) :: x, y, z
    x = xsize
    y = ysize
    z = zsize
  end subroutine pbound_get_size

  !> @brief export r75 parameter
  !
  !> @return real(c_double) value of the r75 parameter
  function pbound_get_r75() bind(c) result(q)
    use, intrinsic :: iso_c_binding, only: c_double
    use pbound, only: r75
    implicit none
    real(c_double) :: q
    q = r75
  end function pbound_get_r75
  
  !> @brief is the periodic box a truncated octohedral box
  !
  !> @return integer(c_int) 1 if box is truncated octohedral, 0 otherwise
  function pbound_is_to_box() bind(c) result(q)
    use, intrinsic :: iso_c_binding, only: c_int
    use pbound, only: qtoboun
    implicit none
    integer(c_int) :: q
    q = 0
    if (qtoboun) q = 1
  end function pbound_is_to_box

  !> @brief is the periodic box cubic
  !
  !> @return integer(c_int) 1 if box is cubic, 0 otherwise
  function pbound_is_cubic_box() bind(c) result(q)
    use, intrinsic :: iso_c_binding, only: c_int
    use pbound, only: qcuboun
    implicit none
    integer(c_int) :: q
    q = 0
    if (qcuboun) q = 1
  end function pbound_is_cubic_box

  subroutine pbound_pbmove(x, y, z) bind(c)
    use, intrinsic :: iso_c_binding, only: c_double
    implicit none
    real(c_double) :: x, y, z
    call pbmove(x, y, z)
  end subroutine pbound_pbmove

#endif /* KEY_PBOUND */ 

end module api_pbound
