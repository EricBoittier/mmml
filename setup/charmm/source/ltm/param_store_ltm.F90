module param_store
  use chm_kinds, only: chm_real
  implicit none

  !     Module to hold names and values of
  !     miscellaneous parameters that can be substituted
  !     in CHARMM input scripts
  !     by RDCMND.    JULY-1985 LN

  private

  integer, parameter :: alloc_increment = 512, max_name_len = 8

  integer, save :: max_real_params, n_real_params
  character(len=max_name_len), dimension(:), allocatable, save :: real_param_names
  real(chm_real), dimension(:), allocatable, save :: real_param_vals

  integer, save :: max_int_params, n_int_params
  character(len=max_name_len), dimension(:), allocatable, save :: int_param_names
  integer, dimension(:), allocatable, save :: int_param_vals

  integer, save :: max_str_params, n_str_params
  character(len=max_name_len), dimension(:), allocatable, save :: &
    str_param_names, str_param_vals

  interface set_param
    module procedure set_real_param, set_int_param, set_str_param
  end interface set_param

  interface get_param
    module procedure get_real_param, get_int_param, get_str_param
  end interface get_param

  interface find_param
    module procedure find_real_param, find_int_param, find_str_param
  end interface find_param

  public :: param_store_init, &
    set_param, get_param, find_param, &
    write_real_params, write_int_params, write_str_params, &
    get_num_reals, get_reals, &
    get_num_ints, get_ints, &
    get_num_strs, get_strs, &
    max_name_len

contains

  !> initialize storage for parameters
  !! if storage is already allocated, deallocate
  !! so to clear storage after an init,
  !! just call init again
  subroutine param_store_init()
    use number
    use consta
    use dimens_fcm
    use memory, only: chmdealloc

    implicit none

    if (allocated(real_param_names)) then
      call chmdealloc('ltm/param_store_ltm.src', 'param_store_init', &
        'real_param_names', max_real_params, ch8 = real_param_names, &
        qdie = .true.)
    end if

    if (allocated(real_param_vals)) then
      call chmdealloc('ltm/param_store_ltm.src', 'param_store_init', &
        'real_param_vals', max_real_params, crl = real_param_vals, &
        qdie = .true.)
    end if

    n_real_params = 0
    max_real_params = 0
    call more_real_params()

    if (allocated(int_param_names)) then
      call chmdealloc('ltm/param_store_ltm.src', 'param_store_init', &
        'int_param_names', max_int_params, ch8 = int_param_names, &
        qdie = .true.)
    end if

    if (allocated(int_param_vals)) then
      call chmdealloc('ltm/param_store_ltm.src', 'param_store_init', &
        'int_param_vals', max_int_params, intg = int_param_vals, &
        qdie = .true.)
    end if

    n_int_params = 0
    max_int_params = 0
    call more_int_params()

    if (allocated(str_param_names)) then
      call chmdealloc('ltm/param_store_ltm.src', 'param_store_init', &
        'str_param_names', max_str_params, ch8 = str_param_names, &
        qdie = .true.)
    end if

    if (allocated(str_param_vals)) then
      call chmdealloc('ltm/param_store_ltm.src', 'param_store_init', &
        'str_param_vals', max_str_params, ch8 = str_param_vals, &
        qdie = .true.)
    end if

    n_str_params = 0
    max_str_params = 0
    call more_str_params()
  end subroutine param_store_init

  !> allocate more storage for parameters of type real(chm_real)
  !! advances max_real_params by alloc_increment number of elts
  !! note that there are three diff maxes,
  !! one for each diff type of param
  subroutine more_real_params()
    use memory, only: chmalloc, chmrealloc
    implicit none
    integer :: error_code

    error_code = 0
    max_real_params = max_real_params + alloc_increment

    if (.not. allocated(real_param_names)) then
       call chmalloc('ltm/param_store_ltm.src', &
            'more_real_params', 'real_param_names', &
            max_real_params, ch8 = real_param_names, ierr = error_code)
       call handle_alloc_error(error_code, max_real_params, 'real')
    else
       call chmrealloc('ltm/param_store_ltm.src', &
            'more_real_parames', 'real_param_names', &
            max_real_params, ch8 = real_param_names, ierr = error_code)
       call handle_alloc_error(error_code, max_real_params, 'real')
    end if

    if (.not. allocated(real_param_vals)) then
       call chmalloc('ltm/param_store_ltm.src', &
            'more_real_params', 'real_param_vals', &
            max_real_params, crl = real_param_vals, ierr = error_code)
       call handle_alloc_error(error_code, max_real_params, 'real')
    else
       call chmrealloc('ltm/param_store_ltm.src', &
            'more_real_params', 'real_param_vals', &
            max_real_params, crl = real_param_vals, ierr = error_code)
       call handle_alloc_error(error_code, max_real_params, 'real')
    end if
  end subroutine more_real_params

  !> allocate more storage for parameters of type integer
  !! advances max_int_params by alloc_increment number of elts
  subroutine more_int_params()
    use memory, only: chmalloc, chmrealloc
    implicit none
    integer :: error_code

    error_code = 0
    max_int_params = max_int_params + alloc_increment

    if (.not. allocated(int_param_names)) then
       call chmalloc('ltm/param_store_ltm.src', &
            'more_int_params', 'int_param_names', &
            max_int_params, ch8 = int_param_names, ierr = error_code)
       call handle_alloc_error(error_code, max_int_params, 'int')
    else
       call chmrealloc('ltm/param_store_ltm.src', &
            'more_int_parames', 'int_param_names', &
            max_int_params, ch8 = int_param_names, ierr = error_code)
       call handle_alloc_error(error_code, max_int_params, 'int')
    end if

    if (.not. allocated(int_param_vals)) then
       call chmalloc('ltm/param_store_ltm.src', &
            'more_int_params', 'int_param_vals', &
            max_int_params, intg = int_param_vals, ierr = error_code)
       call handle_alloc_error(error_code, max_int_params, 'int')
    else
       call chmrealloc('ltm/param_store_ltm.src', &
            'more_int_params', 'int_param_vals', &
            max_int_params, intg= int_param_vals, ierr = error_code)
       call handle_alloc_error(error_code, max_int_params, 'int')
    end if
  end subroutine more_int_params

  !> allocate more storage for parameters of type character(len=8)
  !! advances max_str_params by alloc_increment number of elts
  subroutine more_str_params()
    use memory, only: chmalloc, chmrealloc
    implicit none
    integer :: error_code

    error_code = 0
    max_str_params = max_str_params + alloc_increment

    if (.not. allocated(str_param_names)) then
       call chmalloc('ltm/param_store_ltm.src', &
            'more_str_params', 'str_param_names', &
            max_str_params, ch8 = str_param_names, ierr = error_code)
       call handle_alloc_error(error_code, max_str_params, 'str')
    else
       call chmrealloc('ltm/param_store_ltm.src', &
            'more_str_parames', 'str_param_names', &
            max_str_params, ch8 = str_param_names, ierr = error_code)
       call handle_alloc_error(error_code, max_str_params, 'str')
    end if

    if (.not. allocated(str_param_vals)) then
       call chmalloc('ltm/param_store_ltm.src', &
            'more_str_params', 'str_param_vals', &
            max_str_params, ch8 = str_param_vals, ierr = error_code)
       call handle_alloc_error(error_code, max_str_params, 'str')
    else
       call chmrealloc('ltm/param_store_ltm.src', &
            'more_str_params', 'str_param_vals', &
            max_str_params, ch8 = str_param_vals, ierr = error_code)
       call handle_alloc_error(error_code, max_str_params, 'str')
    end if
  end subroutine more_str_params

  !> call wrndie for memory mishaps occuring in more_*_params
  !!
  !! @param error_code from chm(re)alloc, 0 if no error
  !! @param max_count max_*_params to be decremented due to error in increase
  !! @param kind_str 'real', 'integer', or 'str' for display to user
  subroutine handle_alloc_error(error_code, max_count, kind_str)
    implicit none
    integer, intent(in) :: error_code
    integer, intent(inout) :: max_count
    character(len=*), intent(in) :: kind_str
    if (error_code /= 0) then
       max_count = max_count - alloc_increment
       call wrndie(-5, '<more_real_params>', &
            'memory allocation failed for ' // &
            kind_str // ' parameter storage')
    end if
  end subroutine handle_alloc_error

  !> sets a real parameter named wrd to new_val
  !! if a real param named wrd exists, overwrites old value with new_val
  !! if dne, add it to the end, expanding storage if necessary
  !!
  !! @param wrd the name of the real param to set
  !! @param new_val the floating point value that wrd real param will take
  subroutine set_real_param(wrd, new_val)
    use chm_kinds, only: chm_real
    implicit none
    character(len=*), intent(in) :: wrd
    real(chm_real), intent(in) :: new_val

    real(chm_real) :: old_val ! unused dummy argument
    logical :: found
    integer :: i

    call find_real_param(wrd, old_val, found, i)
    if (found) then
      real_param_vals(i) = new_val
      return
    end if

    !     couldn't find it. add a new one
    if (n_real_params .ge. max_real_params) then
       call more_real_params()
    end if

    n_real_params = n_real_params + 1
    real_param_names(n_real_params) = wrd
    real_param_vals(n_real_params) = new_val
  end subroutine set_real_param

  !> sets an int param named wrd to new_val
  !! if an int param named wrd exists, overwrites old value with new_val
  !! if dne, add it to the end, expanding storage if necessary
  !!
  !! @param wrd the name of the int param to set
  !! @param new_val the integer value that wrd int param will take
  subroutine set_int_param(wrd, new_val)
    implicit none
    character(len=*), intent(in) :: wrd
    integer, intent(in) :: new_val

    integer :: old_val ! unused dummy argument
    logical :: found
    integer :: i

    call find_int_param(wrd, old_val, found, i)
    if (found) then
      int_param_vals(i) = new_val
      return
    end if

    !     couldn't find it. add a new one
    if (n_int_params .ge. max_int_params) then
       call more_int_params()
    end if

    n_int_params = n_int_params + 1
    int_param_names(n_int_params) = wrd
    int_param_vals(n_int_params) = new_val
  end subroutine set_int_param

  !> sets a string param named wrd to new_val
  !! if a string param named wrd exists, overwrites old value with new_val
  !! if dne, add it to the end, expanding storage if necessary
  !!
  !! @param wrd the name of the string param to set
  !! @param new_val wrd param will be set to the first 8 characters of new_val
  subroutine set_str_param(wrd, new_val)
    implicit none
    character(len=*), intent(in) :: wrd, new_val

    character(len=max_name_len) :: old_val ! unused dummy argument
    logical :: found
    integer :: i

    call find_str_param(wrd, old_val, found, i)
    if (found) then
      str_param_vals(i) = new_val
      return
    end if

    !     couldn't find it. add a new one
    if (n_str_params .ge. max_str_params) then
       call more_str_params()
    end if

    n_str_params = n_str_params + 1
    str_param_names(n_str_params) = wrd
    str_param_vals(n_str_params) = new_val
  end subroutine set_str_param

  !> finds and returns a real parameter
  !! only look for the param in the list of real params
  !! if not found, found is set to false and found_index is set to -1
  !!
  !! @param wrd name used for searching, simple exact string match
  !! @param val if wrd is found, val is set to the stored value, 0.0 otherwise
  !! @param found true if wrd found, false otherwise
  !! @param found_index optional output index of wrd found in real params,
  !!        -1 if not found
  subroutine find_real_param(wrd, val, found, found_index)
    use chm_kinds, only: chm_real
    implicit none
    character(len=*), intent(in) :: wrd
    real(chm_real), intent(out) :: val
    logical, intent(out) :: found
    integer, optional, intent(out) :: found_index

    integer :: i

    ! defaults if not found
    found = .false.
    val = 0.0
    if (present(found_index)) found_index = -1

    do i = 1, n_real_params
       if (wrd == real_param_names(i)) then
          val = real_param_vals(i)
          found = .true.
          if (present(found_index)) found_index = i
          return
       end if
    end do
  end subroutine find_real_param

  !> finds and returns an int parameter
  !! only look for the param in the list of int params
  !! if not found, found is set to false and found_index is set to -1
  !!
  !! @param wrd name used for searching, simple exact string match
  !! @param val if wrd is found, val is set to the stored value, 0 otherwise
  !! @param found true if wrd found, false otherwise
  !! @param found_index optional output index of wrd found in int params,
  !!        -1 if not found
  subroutine find_int_param(wrd, val, found, found_index)
    implicit none
    character(len=*), intent(in) :: wrd
    integer, intent(out) :: val
    logical, intent(out) :: found
    integer, optional, intent(out) :: found_index

    integer :: i

    found = .false.
    val = 0
    if (present(found_index)) found_index = -1

    do i = 1, n_int_params
       if (wrd == int_param_names(i)) then
          val = int_param_vals(i)
          found = .true.
          if (present(found_index)) found_index = i
          return
       end if
    end do
  end subroutine find_int_param

  !> finds and returns an int parameter
  !! only look for the param in the list of int params
  !! if not found, found is set to false and found_index is set to -1
  !!
  !! @param wrd name used for searching, simple exact string match
  !! @param val if wrd is found, val is set to the stored value, 0 otherwise
  !! @param found true if wrd found, false otherwise
  !! @param found_index optional output index of wrd found in int params,
  !!        -1 if not found
  subroutine find_str_param(wrd, val, found, found_index)
    implicit none
    character(len=*), intent(in) :: wrd
    character(len=max_name_len), intent(out) :: val
    logical, intent(out) :: found
    integer, optional, intent(out) :: found_index

    integer :: i

    found = .false.
    val = '        '
    if (present(found_index)) found_index = -1

    do i = 1, n_str_params
       if (wrd == str_param_names(i)) then
          val = str_param_vals(i)
          found = .true.
          if (present(found_index)) found_index = i
          return
       end if
    end do
  end subroutine find_str_param

  !> finds and returns a real parameter
  !! looks for the name wrd in first the real list and then the int list
  !! if not found, an error is reported via wrndie (lvl 2)
  !! uses the find_param interface
  !!
  !! @param wrd name used for searching, simple exact string match
  !! @param val if wrd is found, val is set to the stored value, 0 otherwise
  subroutine get_real_param(wrd, val)
    use chm_kinds, only: chm_real
    implicit none
    character(len=*), intent(in) :: wrd
    real(chm_real), intent(out) :: val

    logical :: found
    integer :: int_val

    call find_param(wrd, val, found)
    if (found) return

    ! if not found among the real params
    ! look amongst the integer params
    ! and cast it to a real
    call find_param(wrd, int_val, found)
    if (found) then
      val = int_val ! cast int to real
      return
    end if

    !     couldnt find it. set value to zero
    call wrndie(-2, '<get_real_param>', &
         'could not find parameter value for ' // wrd)
    val = 0.0
  end subroutine get_real_param

  !> finds and returns an integer parameter
  !! looks for the name wrd in first the int list and then the real list
  !! if not found, an error is reported via wrndie (lvl 2)
  !! uses the find_param interface
  !!
  !! @param wrd name used for searching, simple exact string match
  !! @param val if wrd is found, val is set to the stored value, 0 otherwise
  subroutine get_int_param(wrd, val)
    use chm_kinds, only: chm_real
    implicit none
    character(len=*), intent(in) :: wrd
    integer, intent(out) :: val

    logical :: found
    real(chm_real) :: real_val

    call find_param(wrd, val, found)
    if (found) return

    ! if not found among the int params
    ! look amongst the real params
    ! and cast it to an int
    call find_param(wrd, real_val, found)
    if (found) then
      val = real_val ! cast real to int
      return
    end if

    !     couldnt find it. set value to zero
    call wrndie(-2, '<get_int_param>', &
         'could not find parameter value for ' // wrd)
    val = 0
  end subroutine get_int_param

  !> finds and returns a string parameter
  !! looks for the name wrd in only the str list
  !! if not found, an error is reported via wrndie (lvl 2)
  !! uses the find_param interface
  !!
  !! @param wrd name used for searching, simple exact string match
  !! @param val if wrd is found, val is set to the stored value, else 8 spaces
  subroutine get_str_param(wrd, val)
    implicit none
    character(len=*), intent(in) :: wrd
    character(len=max_name_len), intent(out) :: val

    logical :: found

    call find_param(wrd, val, found)
    if (found) return

    !     couldnt find it. set value to blank
    call wrndie(-2, '<get_str_param>', &
         'could not find parameter value for ' // wrd)
    val = '        '
  end subroutine get_str_param

  !> writes all of the real param names and values
  !! to output unit # out_unit
  !!
  !! @param out_unit unit # to write to
  subroutine write_real_params(out_unit)
    implicit none
    integer, intent(in) :: out_unit
    integer :: i

    do i = 1, n_real_params
      write(out_unit, '(a, a8, a, f16.6)') &
        ' MISCOM: label: ', real_param_names(i), &
        ' value:', real_param_vals(i)
    end do
  end subroutine write_real_params

  !> writes all of the int param names and values
  !! to output unit # out_unit
  !!
  !! @param out_unit unit # to write to
  subroutine write_int_params(out_unit)
    implicit none
    integer, intent(in) :: out_unit
    integer :: i

    do i = 1, n_int_params
      write(out_unit, '(a, a8, a, i10)') &
        ' MISCOM: label: ', int_param_names(i), &
        ' value:', int_param_vals(i)
    end do
  end subroutine write_int_params

  !> writes all of the string param names and values
  !! to output unit # out_unit
  !!
  !! @param out_unit unit # to write to
  subroutine write_str_params(out_unit)
    implicit none
    integer, intent(in) :: out_unit
    integer :: i

    do i = 1, n_str_params
      write(out_unit, '(a, a8, a, a8)') &
        ' MISCOM: label: ', str_param_names(i), &
        ' value: ', str_param_vals(i)
    end do
  end subroutine write_str_params

  function get_num_reals() result(n)
    implicit none
    integer :: n
    n = n_real_params
  end function get_num_reals

  subroutine get_reals(out_names, out_vals)
    use chm_kinds, only: chm_real
    implicit none
    character(len=max_name_len), dimension(n_real_params), intent(out) :: out_names
    real(chm_real), dimension(n_real_params), intent(out) :: out_vals
    out_names(1:n_real_params) = real_param_names(1:n_real_params)
    out_vals(1:n_real_params) = real_param_vals(1:n_real_params)
  end subroutine get_reals

  function get_num_ints() result(n)
    implicit none
    integer :: n
    n = n_int_params
  end function get_num_ints

  subroutine get_ints(out_names, out_vals)
    implicit none
    character(len=max_name_len), dimension(n_int_params), intent(out) :: out_names
    integer, dimension(n_int_params), intent(out) :: out_vals
    out_names(1:n_int_params) = int_param_names(1:n_int_params)
    out_vals(1:n_int_params) = int_param_vals(1:n_int_params)
  end subroutine get_ints

  function get_num_strs() result(n)
    implicit none
    integer :: n
    n = n_str_params
  end function get_num_strs

  subroutine get_strs(out_names, out_vals)
    implicit none
    character(len=max_name_len), dimension(n_str_params), intent(out) :: out_names
    character(len=max_name_len), dimension(n_str_params), intent(out) :: out_vals
    out_names(1:n_str_params) = str_param_names(1:n_str_params)
    out_vals(1:n_str_params) = str_param_vals(1:n_str_params)
  end subroutine get_strs
end module param_store
