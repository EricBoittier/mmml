!> routines to manipulate files at the fortran level
module api_charmm_file
  implicit none
contains

  !> @brief open a file for reading or writing and pass the unit back
  function charmm_file_open(filename, len_fn, &
       to_read, to_write, to_append, &
       formatted, &
       new_unit) result(out_unit) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_char
    use api_util, only: c2f_string
    use exfunc, only: lunass
    use machio, only: vopen

    implicit none

    ! args
    character(len=1, kind=c_char) :: filename(*)
    integer(c_int), value :: &
         len_fn, &
         to_read, to_write, to_append, &
         formatted, &
         new_unit

    ! result
    integer(c_int) :: out_unit

    ! local vars
    character(len=len_fn) :: fn
    character(len=16) :: access_spec, format_spec
    logical :: qerr

    out_unit = -1

    if (new_unit .eq. -1) then
       new_unit = lunass(90)
    end if

    fn = c2f_string(filename, len_fn)

    access_spec = ' '
    if (to_write == 1 .and. to_append == 1) then
       access_spec = 'APPEND'
    else if (to_write == 1) then
       access_spec = 'WRITE'
    else if (to_read == 1) then
       access_spec = 'READ'
    else
       access_spec = 'READ'
       call wrndie(-1, &
            '<api_dynamics%open_file>', &
            'access type not clearly specified, assuming READ')
    end if

    format_spec = 'UNFORMATTED'
    if (formatted == 1) format_spec = 'FORMATTED'

    call vopen(new_unit, fn, format_spec, access_spec, qerr, 0)

    if (qerr) then
       call wrndie(-5, &
            '<api_charmm_file%charmm_file_open>', &
            'cannot open file ' // fn(1:len_fn))
       return
    end if

    out_unit = new_unit
  end function charmm_file_open

  function charmm_file_close(file_unit) result(success) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    implicit none
    integer(c_int), value :: file_unit
    integer(c_int) :: success
    logical :: qerr

    success = 0

    call vclose(file_unit, 'KEEP', qerr)
    if (qerr) then
       call wrndie(-5, &
            '<api_charmm_file%charmm_file_close>', &
            'cannot close file')
       return
    end if

    success = 1
  end function charmm_file_close
end module api_charmm_file
