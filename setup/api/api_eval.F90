!> routines to evaluate expressions in languages other than python
module api_eval
  implicit none
contains

#if KEY_LIBRARY == 1
  !> @brief evaluate an expression in charmm's native script language
  !
  ! the script array is read from 1 to nchar or c_null_char
  ! which ever comes first
  !
  !> param[in] c_script an expression in charmm's native script language
  !> param[in] nchars number of characters to read and evaluate from script
  !> @return integer(c_int) 1 if successful, otherwise there was an error
  ! integer(c_int) function eval_charmm_script(c_script, c_len) bind(C)
  !   use, intrinsic :: iso_c_binding, only: c_int, c_char, c_null_char

  !   use api_util, only: c2f_string

  !   use comand, only: comlen, comlyn
  !   use dimens_fcm, only: mxcmsz

  !   implicit none

  !   ! args
  !   character(kind=c_char, len=1) :: c_script(*)
  !   integer(c_int), intent(in) :: c_len

  !   ! local vars
  !   logical :: lused = .false.

  !   eval_charmm_script = 0

  !   if (c_len .gt. mxcmsz) then
  !      comlen = mxcmsz
  !   else
  !      comlen = c_len
  !   end if

  !   comlyn(1:comlen) = c2f_string(c_script, comlen)

  !   if (comlen .lt. mxcmsz) comlyn((comlen + 1):mxcmsz) = ' '
  !   comlen = len(trim(comlyn))

  !   call miscom(comlyn, mxcmsz, comlen, lused)
  !   if (.not. lused) call maincomx(comlyn, comlen, lused)

  !   eval_charmm_script = 1
  ! end function eval_charmm_script

  !> @brief evaluate an expression in charmm's native script language
  !
  ! the script array is read from 1 to nchar or c_null_char
  ! which ever comes first
  !
  !> param[in] c_script an expression in charmm's native script language
  !> param[in] nchars number of characters to read and evaluate from script
  !> @return integer(c_int) 1 if successful, otherwise there was an error
  integer(c_int) function eval_charmm_script(c_script, c_len) bind(C)
    use, intrinsic :: iso_c_binding, only: c_int, c_char, c_null_char

    use comand, only: comlen, comlyn
    use ctitla, only: titlea, ntitla
    use dimens_fcm, only: mxcmsz
    use exfunc, only: lunass
    use stream, only: nstrm, istrm, jstrm

    implicit none

    ! args
    character(kind=c_char, len=1) :: c_script(*)
    integer(c_int), value, intent(in) :: c_len

    ! local vars
    integer :: script_unit
    logical :: read_cmd, used, ok, eof, get_next_cmd

    eval_charmm_script = 0

    script_unit = lunass(90)
    open(unit=script_unit, status='SCRATCH')
    write(script_unit, *) c_script(1:c_len)
    rewind(script_unit)

    nstrm = 1
    istrm = script_unit
    jstrm(nstrm) = istrm

    titlea = ' '
    titlea = '* EXECUTING CHARMM SCRIPT FROM PYTHON'
    ntitla = 1

    eof = .false.
    read_cmd = .true.
    used = .false.
    get_next_cmd = .true.
    do while (.true.)
       get_next_cmd = .true.
       do while (get_next_cmd)
          if (read_cmd) then
             call rdcmnd(comlyn, mxcmsz, comlen, istrm, eof, .true., .true., &
                  'CHARMM> ')
          end if
          read_cmd = .true.
          call miscom(comlyn, mxcmsz, comlen, used)
          get_next_cmd = used .and. .not. eof
       end do

       ! If we run out of stuff on a particular stream, pop to the
       ! previous stream. quit when there are no more streams.
       if (eof) then
          if (nstrm > 1) then
             call ppstrm(ok)
          else
             ok = .false.
          end if

          if (.not. ok) then
             close(script_unit)
             eval_charmm_script = 1
             return
          end if

          eof = .false.
          cycle
       end if

       call maincomx(comlyn, comlen, used)
    end do

    close(script_unit)
    eval_charmm_script = 1
  end function eval_charmm_script

  function eval_get_param(name, len_name, val, len_val) result(is_found) bind(c)
    use, intrinsic :: iso_c_binding, only: c_char, c_int, c_null_char

    use api_types, only: &
         found_type, found_int, found_real, found_bool, &
         bool, bool_true, bool_false, &
         found_value

    use api_util, only: c2f_string

    use cmdpar, only: &
         toknam, toklen, &
         valnam, vallen, &
         numpar

    implicit none

     ! args
    character(kind=c_char, len=1) :: name(*), val(*)
    integer(c_int), value, intent(in) :: len_name, len_val

    ! return value
    integer(kind(bool)) :: is_found

       ! local vars
    character(len=len_name) :: f_name
    integer :: len_f, i, j, len_safe

    is_found = bool_false

    f_name = c2f_string(name, len_name)
    len_f = len_name

    do i = 1, numpar
       if (f_name(1:len_f) == toknam(i)(1:toklen(i))) then
          len_safe = min(len_val, vallen(i))
          val(1:len_safe) = (/ (valnam(i)(j:j), j = 1, len_safe) /)
          val(1 + len_safe) = c_null_char
          is_found = bool_true
          return
       end if
    end do
  end function eval_get_param

  function eval_num_params() result(nparams) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use cmdpar, only: numpar
    implicit none
    integer(c_int) :: nparams
    nparams = numpar
  end function eval_num_params

  function eval_max_param_name() result(max_length) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use cmdpar, only: mxtlen
    implicit none
    integer(c_int) :: max_length
    max_length = mxtlen
  end function eval_max_param_name

  function eval_max_param_val() result(max_length) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use cmdpar, only: mxvlen
    implicit none
    integer(c_int) :: max_length
    max_length = mxvlen
  end function eval_max_param_val

  subroutine eval_get_all_params(out_names, out_vals) bind(c)
    use, intrinsic :: iso_c_binding, only: c_ptr

    use api_util, only: f2c_string

    use cmdpar, only: &
         toknam, toklen, &
         valnam, vallen, &
         numpar

    implicit none

    ! args
    type(c_ptr), target, dimension(*) :: out_names, out_vals

    ! local vars
    integer :: i

    do i = 1, numpar
       call f2c_string(toknam(i), out_names(i), toklen(i))
       call f2c_string(valnam(i), out_vals(i), vallen(i))
    end do
  end subroutine eval_get_all_params

  function eval_set_param(name, len_name, val, len_val) result(is_found) bind(c)
    use, intrinsic :: iso_c_binding, only: c_char, c_int, c_null_char

    use api_types, only: &
         found_type, found_int, found_real, found_bool, &
         bool, bool_true, bool_false, &
         found_value

    use api_util, only: c2f_string

    use cmdpar, only: &
         toknam, toklen, mxtlen, &
         valnam, vallen, mxvlen, &
         numpar, maxpar

    implicit none

     ! args
    character(kind=c_char, len=1) :: name(*), val(*)
    integer(c_int), value, intent(in) :: len_name, len_val

    ! return value
    integer(kind(bool)) :: is_found

       ! local vars
    character(len=len_name) :: f_name
    character(len=len_val) :: f_val
    integer :: len_f, i, len_safe, insert_index

    is_found = bool_false
    insert_index = -1

    if (len_name <= 0)  then
       call wrndie(1, &
            '<api_eval%eval_set_param>', &
            'Token empty. Nothing set.')
       return
    end if

    if (len_val <= 0)  then
       call wrndie(1, &
            '<api_eval%eval_set_param>', &
            'Value empty. Nothing set.')
       return
    end if

    f_name = c2f_string(name, len_name)
    len_f = len_name

    do i = 1, numpar
       if (f_name(1:len_f) == toknam(i)(1:toklen(i))) then
          is_found = bool_true
          insert_index = i
          exit
       end if
    end do

    if (insert_index <= 0) then
       numpar = numpar + 1
       insert_index = numpar
    end if

    if (numpar > maxpar) then
       numpar = maxpar
       call wrndie(0, &
            '<api_eval%eval_set_param>', &
            'Parameter table full. Nothing set.')
       return
    end if

    f_val = c2f_string(val, len_val)

    len_safe = min(mxtlen, len_name)
    toknam(insert_index)(1:len_safe) = f_name(1:len_safe)
    toklen(insert_index) = len_safe
    if (len_safe < len_name) then
       call wrndie(1, &
            '<api_eval%eval_set_param>', &
            'Token may have been truncated to ' // f_name(1:len_safe))
    end if

    len_safe = min(mxvlen, len_val)
    valnam(insert_index)(1:len_safe) = f_val(1:len_safe)
    vallen(insert_index) = len_safe
    if (len_safe < len_val) then
       call wrndie(1, &
            '<api_eval%eval_set_param>', &
            'Value may have been truncated to ' // f_val(1:len_safe))
    end if
  end function eval_set_param

  function eval_get_energy_value(name, len) result(found) bind(c)
    use, intrinsic :: iso_c_binding, only: c_char, c_int

    use api_types, only: &
         found_type, found_int, found_real, found_bool, &
         bool, bool_true, bool_false, &
         found_value

    use api_util, only: c2f_string

    use energym, only: &
         lenenp, ceprop, eprop, &
         lenent, ceterm, eterm, &
         lenenv, ceprss, epress

    use param_store, only: find_param
    use rndnum, only: irndsd
    use clcg_mod, only: ranumb
    use string, only: trime

    implicit none

    ! args
    character(kind=c_char, len=1) :: name(*)
    integer(c_int), value, intent(in) :: len

    ! return value
    type(found_value) :: found

    ! local vars
    character(len=len) :: f_name
    integer :: f_name_len, i
    logical :: is_found

    found%is_found = bool_false

    f_name = c2f_string(name, len)
    f_name_len = len
    call trime(f_name, f_name_len)

    do i = 1, lenenp
       if (f_name(1:4) == ceprop(i)) then
          found%is_found = found_real
          found%real_val = eprop(i)
          return
       end if
    end do

    do i = 1, lenent
       if (f_name(1:4) == ceterm(i)) then
          found%is_found = found_real
          found%real_val = eterm(i)
          return
       end if
    end do

    do i = 1, lenenv
       if (f_name(1:4) == ceprss(i)) then
          found%is_found = found_real
          found%real_val = epress(i)
          return
       end if
    end do

    if (f_name(1:4) == 'RAND') then
       found%is_found = found_real
       found%real_val = ranumb()
       return
    end if

    if (f_name(1:4) == 'ISEE') then
       found%is_found = found_int
       found%int_val = irndsd
       return
    end if

    call find_param(f_name, found%real_val, is_found)
    if (is_found) then
       found%is_found = found_real
       return
    end if

    call find_param(f_name, found%int_val, is_found)
    if (is_found) then
       found%is_found = found_int
       return
    end if
  end function eval_get_energy_value

  function builtins_names_max() result(n) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use param_store, only: max_name_len
    implicit none
    integer(c_int) :: n
    n = max_name_len
  end function builtins_names_max

  function builtins_reals_num() result(n) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use param_store, only: get_num_reals
    implicit none
    integer(c_int) :: n
    n = get_num_reals()
  end function builtins_reals_num

  subroutine builtins_reals_get(out_names, out_vals) bind(c)
    use, intrinsic :: iso_c_binding, only: c_double, c_ptr
    use chm_kinds, only: chm_real
    use memory, only: chmalloc, chmdealloc
    use api_util, only: f2c_string
    use param_store, only: get_reals, max_name_len
    implicit none
    type(c_ptr), target, dimension(*) :: out_names
    real(c_double), dimension(*) :: out_vals

    real(chm_real), allocatable, dimension(:) :: vals
    character(len=max_name_len), allocatable, dimension(:) :: names
    integer :: i, n

    n = builtins_reals_num()
    call chmalloc('api/api_eval.inp', 'builtins_reals_get', 'names', &
         n, ch8 = names)
    call chmalloc('api/api_eval.inp', 'builtins_reals_get', 'vals', &
         n, crl = vals)

    call get_reals(names, vals)

    do i = 1, n
       call f2c_string(names(i), out_names(i), min(8, max_name_len))
    end do
    call chmdealloc('api/api_eval.inp', 'builtins_reals_get', 'names', &
         n, ch8 = names)

    out_vals(1:n) = vals(1:n)
    call chmdealloc('api/api_eval.inp', 'builtins_reals_get', 'vals', &
         n, crl = vals)
  end subroutine builtins_reals_get

  function builtins_ints_num() result(n) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use param_store, only: get_num_ints
    implicit none
    integer(c_int) :: n
    n = get_num_ints()
  end function builtins_ints_num

  subroutine builtins_ints_get(out_names, out_vals) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_ptr
    use memory, only: chmalloc, chmdealloc
    use api_util, only: f2c_string
    use param_store, only: get_ints, max_name_len
    implicit none
    type(c_ptr), target, dimension(*) :: out_names
    integer(c_int), dimension(*) :: out_vals

    integer, allocatable, dimension(:) :: vals
    character(len=max_name_len), allocatable, dimension(:) :: names
    integer :: i, n

    n = builtins_ints_num()
    call chmalloc('api/api_eval.inp', 'builtins_ints_get', 'names', &
         n, ch8 = names)
    call chmalloc('api/api_eval.inp', 'builtins_ints_get', 'vals', &
         n, intg = vals)

    call get_ints(names, vals)

    do i = 1, n
       call f2c_string(names(i), out_names(i), min(8, max_name_len))
    end do
    call chmdealloc('api/api_eval.inp', 'builtins_ints_get', 'names', &
         n, ch8 = names)

    out_vals(1:n) = vals(1:n)
    call chmdealloc('api/api_eval.inp', 'builtins_ints_get', 'vals', &
         n, intg = vals)
  end subroutine builtins_ints_get

  function builtins_strs_num() result(n) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use param_store, only: get_num_strs
    implicit none
    integer(c_int) :: n
    n = get_num_strs()
  end function builtins_strs_num

  subroutine builtins_strs_get(out_names, out_vals) bind(c)
    use, intrinsic :: iso_c_binding, only: c_ptr
    use memory, only: chmalloc, chmdealloc
    use api_util, only: f2c_string
    use param_store, only: get_strs, max_name_len
    implicit none
    type(c_ptr), target, dimension(*) :: out_names, out_vals

    character(len=max_name_len), allocatable, dimension(:) :: names, vals
    integer :: i, n

    n = builtins_strs_num()
    call chmalloc('api/api_eval.inp', 'builtins_strs_get', 'names', &
         n, ch8 = names)
    call chmalloc('api/api_eval.inp', 'builtins_strs_get', 'vals', &
         n, ch8 = vals)

    call get_strs(names, vals)

    do i = 1, n
       call f2c_string(names(i), out_names(i), min(8, max_name_len))
       call f2c_string(vals(i), out_vals(i), min(8, max_name_len))
    end do
    call chmdealloc('api/api_eval.inp', 'builtins_strs_get', 'names', &
         n, ch8 = names)
    call chmdealloc('api/api_eval.inp', 'builtins_strs_get', 'vals', &
         n, ch8 = vals)
  end subroutine builtins_strs_get
#endif /* KEY_LIBRARY */

end module api_eval
