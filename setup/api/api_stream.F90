!> @brief routines for manipulating the input and output units
!
! These functions modify charmm's stream module.
! The stream module contains global variables describing
! default input and output units (analogous to stdin and stdout)
module api_stream
  implicit none
contains

  !> @brief set the verbosity level of charmm
  !
  !> @param[in] new_prnlev the new integer value for prnlev
  !> @return integer old prnlev
  integer(c_int) function stream_set_prnlev(new_prnlev) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use stream, only: prnlev

    implicit none
    
    integer(c_int) :: new_prnlev

    stream_set_prnlev = prnlev
    prnlev = new_prnlev
  end function stream_set_prnlev

  !> @brief set the warning level of charmm
  !
  !> @param[in] new_wrnlev the new integer value for wrnlev
  !> @return integer old wrnlev
  integer(c_int) function stream_set_wrnlev(new_wrnlev) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use chm_kinds, only: wrnlev

    implicit none
    
    integer(c_int) :: new_wrnlev

    stream_set_wrnlev = wrnlev
    wrnlev = new_wrnlev
  end function stream_set_wrnlev

  !> @brief set the bomb level of charmm
  !
  !> @param[in] new_bomlev the new integer value for bomlev
  !> @return integer old bomlev
  integer(c_int) function stream_set_bomlev(new_bomlev) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use chm_kinds, only: bomlev

    implicit none
    
    integer(c_int) :: new_bomlev

    stream_set_bomlev = bomlev
    bomlev = new_bomlev
  end function stream_set_bomlev
  
end module api_stream
