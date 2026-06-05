module api_omm
  implicit none
contains

  !> @brief get the size of the seralized system in characters
  !
  !  This function is called so that storage can be allocated
  !  on the python side before getting the serialized system
  !
  !> @return integer(c_int) number of characters in the serialized system
  function omm_get_system_serial_size() bind(c) result(n)
    use, intrinsic :: iso_c_binding, only: c_char, c_int
#if KEY_OPENMM == 1
    use omm_main, only: serialize_system
#endif /* KEY_OPENMM == 1 */
    implicit none
    integer(c_int) :: n
    character(kind=c_char, len=1), allocatable, dimension(:) :: sys
#if KEY_OPENMM == 1
    call serialize_system(sys)
    n = size(sys)
#else
    n = 0
#endif /* KEY_OPENMM == 1 */
  end function omm_get_system_serial_size

  !> @brief get a serialized system in a format that OpenMM can read
  !
  !  call omm_get_system_serial_size to preallocate storage for out_serial
  !  before calling this subroutine and pass the result as max_size
  !
  !> @param[out] out_serial preallocated char array to hold serialization
  !> @param[in] max_size number of chars that can be safely stored in out_serial
  subroutine omm_get_system_serial(out_serial, max_size) bind(c)
    use, intrinsic :: iso_c_binding, only: c_char, c_int
#if KEY_OPENMM == 1
    use omm_main, only: serialize_system
#endif /* KEY_OPENMM == 1 */
    implicit none

    character(kind=c_char, len=1) :: out_serial(*)
    integer(kind=c_int), value :: max_size
    integer :: current_size, min_size
    character(kind=c_char, len=1), allocatable, dimension(:) :: sys

    current_size = 0
    min_size = 0

#if KEY_OPENMM == 1
    call serialize_system(sys)
    current_size = size(sys)
    min_size = min(max_size, current_size)
    out_serial(1:min_size) = sys(1:min_size)
#endif /* KEY_OPENMM == 1 */
  end subroutine omm_get_system_serial
end module api_omm
