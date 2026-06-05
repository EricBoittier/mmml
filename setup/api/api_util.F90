module api_util
  implicit none

contains

  !> @brief convert a c_int to a fortran logical
  !
  !> @param[in] c_val > 0 for true, .le. 0 for false
  !> @return a Fortran logical (either .true. or .false.)
  function c2f_logical(c_val) result(f_val)
    use, intrinsic :: iso_c_binding, only: c_int

    implicit none

    ! args
    integer(c_int) :: c_val

    ! locals
    logical :: f_val

    f_val = .false.
    if (c_val .gt. 0) f_val = .true.
  end function c2f_logical

  !> @brief convert a fortran logical to a c_int
  !
  !> @param[in] f_val a Fortran logical (either .true. or .false.)
  !> @return c_val 1 for true, 0 for false
  function f2c_logical(f_val) result(c_val)
    use, intrinsic :: iso_c_binding, only: c_int

    implicit none

    ! args
    logical :: f_val

    ! locals
    integer(c_int) :: c_val

    c_val = 0
    if (f_val) c_val = 1
  end function f2c_logical

  !> @brief null term C char array -> Fortran string
  !
  !> @param[in] s a null terminated C character array
  !> @param[in] ns the length of s as a string
  !> @return a Fortran string of length ns with contents s
  function c2f_string(s, ns) result(str)
    use, intrinsic :: iso_c_binding, only: c_char, c_null_char

    implicit none

    character(kind=c_char, len=1) :: s(*)
    integer :: ns

    character(len=ns) :: str

    integer :: i

    str = ' '
    do i = 1, ns
       if (s(i) == c_null_char) exit
       str(i:i) = s(i)
    end do
  end function c2f_string

  !> @brief Fortran string -> C character array
  !
  ! space for the C string will be allocated in this routine
  ! checks fstr for null termination, otherwise uses max_char
  !
  !> @param[in] fstr a fortran string possibly null terminated
  !> @param[out] cstr_ptr C pointer to a C character array
  !> @param[in] max_char max num of chars in fstr to copy
  subroutine f2c_string(fstr, cstr_ptr, max_char)
    use iso_c_binding, only: c_char, c_null_char, c_ptr, c_f_pointer

    implicit none

    ! arguments
    character(len = *), intent(in) :: fstr
    type(c_ptr), target :: cstr_ptr ! ptr to output string
    integer, intent(in) :: max_char

    ! local vars
    character(kind=c_char, len=1), pointer, dimension(:) :: cstr
    integer :: i, len

    len = index(fstr, c_null_char)
    if (len > 0) then
       len = min(len, max_char)
    else
       len = max_char
    end if

    call c_f_pointer(cstr_ptr, cstr, [len])
    do i = 1, len
       cstr(i) = fstr(i:i)
    end do
    !if(len.gt.0) cstr = (/ (fstr(i:i), i = 1, len) /)
  end subroutine f2c_string

end module api_util
