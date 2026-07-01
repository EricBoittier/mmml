!> routines for writing files to disk
module api_write

  implicit none

contains

  !> @brief print the main coordinate set of the atoms
  integer(c_int) function write_coor_pdb(filename, len_fn, &
                                         selection, comparison) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_char
    use, intrinsic :: iso_fortran_env, only: output_unit

    use coord, only: x, y, z, wmain
    use coordc, only: xcomp, ycomp, zcomp, wcomp
    use coorio_mod, only: cwrite
    use ctitla, only: titlea, ntitla
    use psf, only: natom, res, atype, ibase, nictot, nseg

    use api_util, only: c2f_string
    use exfunc, only: lunass
    use machio, only: vopen

    implicit none

    ! args
    character(len=1, kind=c_char) :: filename(*)
    integer(c_int) :: len_fn, selection(*), comparison

    ! local vars
    integer :: icntrl(20), nres, ntitle
    character(len = 80) :: title

    character(len=len_fn) :: fn
    integer :: new_out_unit
    logical :: qerr

    write_coor_pdb = -1

    icntrl = 0
    nres = nictot(nseg + 1)

    fn = c2f_string(filename, len_fn)
    new_out_unit = lunass(90)
    call vopen(new_out_unit, fn, 'FORMATTED', 'WRITE', qerr, 0)
    if (qerr) then
       call wrndie(-5, &
            '<api_write%write_coor_pdb>', &
            'cannot open file ' // fn(1:len_fn))
       return
    end if

    if (comparison == 0) then
      call cwrite(new_out_unit, titlea, ntitla, icntrl, &
           x, y, z, wmain, &
           res, atype, ibase, nres, &
           natom, selection(1:natom), &
           4, 0, 0, .false., -1)
    else
      call cwrite(new_out_unit, titlea, ntitla, icntrl, &
           xcomp, ycomp, zcomp, wcomp, &
           res, atype, ibase, nres, &
           natom, selection(1:natom), &
           4, 0, 0, .false., -1)
    end if

    call vclose(new_out_unit, 'KEEP', qerr)
    if (qerr) then
       call wrndie(-5, &
            '<api_write%write_coor_pdb>', &
            'cannot close file ' // fn(1:len_fn))
       return
    end if

    write_coor_pdb = 1
  end function write_coor_pdb

  !> @brief write the psf info to a card file
  !
  !> @param[in] filename path of the new file to write
  !> @param[in] len_fn length of the filename in characters
  integer(c_int) function write_psf_card(filename, len_fn) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_char
    use, intrinsic :: iso_fortran_env, only: output_unit

    use api_util, only: c2f_string
    use ctitla, only: titlea, ntitla
    use exfunc, only: lunass
    use machio, only: vopen

    implicit none

    ! args
    character(len=1, kind=c_char) :: filename(*)
    integer(c_int) :: len_fn

    ! locals
    character(len=len_fn) :: fn
    integer :: new_out_unit
    logical :: qerr

    write_psf_card = -1

    fn = c2f_string(filename, len_fn)
    new_out_unit = lunass(90)
    call vopen(new_out_unit, fn, 'FORMATTED', 'WRITE', qerr, 0)
    if (qerr) then
       call wrndie(-5, &
            '<api_write%write_psf_card>', &
            'cannot open file ' // fn(1:len_fn))
       return
    end if

    call psfwr2(new_out_unit, titlea, ntitla)

    call vclose(new_out_unit, 'KEEP', qerr)
    if (qerr) then
       call wrndie(-5, &
            '<api_write%write_psf_card>', &
            'cannot close file ' // fn(1:len_fn))
       return
    end if

    write_psf_card = 1
  end function write_psf_card
end module api_write
