!> @brief routines for reading files and strings
!
! These functions correspond to the read command in charmm.
! There are function for reading input files as well as
! functions for reading strings. The content of the strings
! would be read as a card by the charmm executable.
module api_read
  implicit none
contains

  !> @brief read a topology file given a filename
  !
  !> @param[in] filename a C character array for the input filename
  !> @param[in] len integer length of the filename
  !> @param[in] append integer if .ne. 0 then append contents to prev data
  !> @return integer error code, success is 1
  integer(c_int) function read_rtf_file(filename, len, append) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_char
    use rtf, only: rtfrdr
    use ctitla, only: titleb, ntitlb
    use api_util, only: c2f_string

    implicit none

    character(len=1, kind=c_char), dimension(*) :: filename
    integer(c_int) :: len, append

    integer, parameter :: iunit = 128, max_line = 256
    character(len=len) :: fn_str

    logical :: qappend = .false.

    if (append .ne. 0) qappend = .true.

    fn_str = c2f_string(filename, len)
    open(iunit, file=fn_str, action='read', err=99)
    call rtfrdr(iunit, titleb, ntitlb, 1, .false., qappend)
    close(iunit, err=99)

    read_rtf_file = 1
    return

99  continue
    call wrndie(-5, '<read_rtf_file>', 'io error opening/closing rtf file')
    read_rtf_file = -1
    return
  end function read_rtf_file

  !> @brief read a parameter file given a filename
  !
  !> @param[in] filename a C character array for the input filename
  !> @param[in] fn_len integer length of the filename
  !> @param[in] append integer if .ne. 0 then append contents to prev data
  !> @param[in] flex integer if .ne. 0 then read a flexible parameter file
  !> @return integer error code, success is 1
#if KEY_LIBRARY == 1
  integer(c_int) function read_param_file(filename, fn_len, append, flex) &
       bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_char
    use, intrinsic :: iso_fortran_env, only: output_unit
    use ctitla, only: titleb, ntitlb
    use parmiom, only: parmio
    use rtf, only:atct, natct
    use code, only: mustup
    use bases_fcm, only: bimag, bnbnd
    use stream, only: prnlev
    use api_util, only: c2f_string
    use image_routines_module, only: climag

    implicit none

    character(len=1, kind=c_char), dimension(*) :: filename
    integer(c_int) :: fn_len, append, flex

    integer, parameter :: iunit = 128, max_line = 256
    integer :: j
    character(len = fn_len) :: fn_str, winit

    logical :: qappend = .false., qflex = .false.

    if (append .ne. 0) qappend = .true.
    if (flex .ne. 0) qflex = .true.

    fn_str = c2f_string(filename, fn_len)

    open(iunit, file=fn_str, action='read', err=99)
    call parmio(iunit, ntitlb, titleb, 1, output_unit, natct, atct, &
      qappend, qflex)
    close(iunit, err=99)

    ! now reset major data structures upon parameter file reading
    ! gtnbct, climag and gthbct are not yet in modules

    winit = 'INIT'
    j = 4
    call gtnbct(winit, j, bnbnd)
    call climag(bimag)

    winit = 'INIT'
    j = 4
    call gthbct(winit, j)
    mustup = .true.
    if (prnlev > 2) write (output_unit, '(a)') &
         ' PARMIO> NONBOND, HBOND lists and IMAGE atoms cleared.'

    read_param_file = 1
    return

99  continue
    call wrndie(-5, '<read_param_file>', 'io error opening/closing rtf file')
    read_param_file = -1
    return
  end function read_param_file
#endif

  !> @brief read a psf card given a filename
  !
  !> @param[in] filename a C character array for the input filename
  !> @param[in] fn_len integer length of the filename
  !> @param[in] iappend integer if .ne. 0 then append atoms, otherwise replace
  !> @param[in] ixplor integer if .ne. 0 then read an XPLOR format psf card
  !> @return integer error code, success is 1
  integer(c_int) function read_psf_card(filename, fn_len, iappend, ixplor) &
       bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_char
    use, intrinsic :: iso_fortran_env, only: output_unit

    use api_util, only: c2f_string
    use ctitla, only: titleb, ntitlb
    use exfunc, only: lunass
    use genpsf_m, only: atmini
    use machio, only: vopen
    use psf, only: natom

    implicit none

    ! args
    character(len=1, kind=c_char) :: filename(*)
    integer(c_int) :: fn_len, iappend, ixplor

    ! locals
    integer, parameter :: max_line = 256
    integer :: start_atom_index, new_in_unit
    character(len = fn_len) :: fn_str
    logical :: qappend, qxplor, qerr

    read_psf_card = -1

    qappend = .false.
    if (iappend .ne. 0) qappend = .true.

    qxplor = .false.
    if (ixplor .ne. 0) qxplor = .true.

    fn_str = c2f_string(filename, fn_len)
    new_in_unit = lunass(90)
    call vopen(new_in_unit, fn_str, 'FORMATTED', 'READ', qerr, 0)
    if (qerr) then
       call wrndie(-5, &
            '<api_read%read_psf_card>', &
            'cannot open file ' // fn_str(1:fn_len))
       return
    end if

    call psf_read_formatted(new_in_unit, titleb, ntitlb, qappend)

    call vclose(new_in_unit, 'KEEP', qerr)
    if (qerr) then
       call wrndie(-5, &
            '<api_read%read_psf_card>', &
            'cannot close file ' // filename(1:fn_len))
       return
    end if

    start_atom_index = 1
    if (qappend) start_atom_index = natom + 1

    call atmini(start_atom_index, natom)
    call psfsum(output_unit)

    read_psf_card = 1
  end function read_psf_card

  !> @brief read a pdb file given a filename
  !
  !> @param[in] filename a C character array for the input filename
  !> @param[in] fn_len integer length of the filename
  !> @return integer error code, success is 1
  integer(c_int) function read_pdb(filename, fn_len, iresid) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_char

    use api_util, only: c2f_string
    use coord, only: x, y, z, wmain
    use coorio_mod, only: cread
    use ctitla, only: titleb, ntitlb
    use exfunc, only: lunass
    use machio, only: vopen
    use psf, only: &
         natom, res, nres, atype, ibase, &
         segid, resid, nictot, nseg

    implicit none

    ! args
    character(len=1, kind=c_char) :: filename(*)
    integer(c_int) :: fn_len, iresid

    ! locals
    integer :: new_in_unit, islct(natom), icntrl(20), ifreea(natom)
    character(len = fn_len) :: fn_str
    logical :: qresid, qerr

    integer, parameter :: line_len = 80
    character(len = line_len) :: line

    read_pdb = -1

    qresid = .false.
    if (iresid .ne. 0) qresid = .true.

    fn_str = c2f_string(filename, fn_len)
    new_in_unit = lunass(90)
    call vopen(new_in_unit, fn_str, 'FORMATTED', 'READ', qerr, 0)
    if (qerr) then
       call wrndie(-5, &
            '<api_read%read_psf_card>', &
            'cannot open file ' // fn_str(1:fn_len))
       return
    end if

    islct = 1
    call cread(new_in_unit, titleb, ntitlb, icntrl, x, y, z, wmain, natom, &
       -1, islct, 0, res, nres, atype, ibase, &
       1, ifreea, &
       segid, resid, nictot, nseg, &
       qresid, .false., &
       line, line_len, &
       0, .false., 0)

    call vclose(new_in_unit, 'KEEP', qerr)
    if (qerr) then
       call wrndie(-5, &
            '<api_read%read_psf_card>', &
            'cannot close file ' // filename(1:fn_len))
       return
    end if
    read_pdb = 1
  end function read_pdb

  ! WARNING this function does not work correctly
  ! TODO: fix read_sequence_pdb
  !> @brief read a sequence from a pdb file
  !
  !> @param[in] filename a C character array for the input filename
  !> @param[in] fn_len integer length of the filename
  !> @return integer error code, success is 1
  integer(c_int) function read_sequence_pdb(filename, fn_len) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_char

    use api_util, only: c2f_string
    use coorio_mod, only: cread
    use ctitla, only: titleb, ntitlb, maxtit
    use dimens_fcm, only: mxcmsz
    use exfunc, only: lunass
    use machio, only: vopen
    use psf, only: &
         natom, res, nres, resid, &
         segid, nseg, nictot

    implicit none

    ! args
    character(len=1, kind=c_char) :: filename(*)
    integer(c_int) :: fn_len

    ! locals
    integer :: new_in_unit, start
    character(len = fn_len) :: fn_str
    logical :: qerr

    integer, parameter :: line_len = 80, max_table = 200
    character(len = line_len) :: line
    character(len = 8) :: blank, skip(max_table), alias(2, max_table)

    read_sequence_pdb = -1

    fn_str = c2f_string(filename, fn_len)
    new_in_unit = lunass(90)
    call vopen(new_in_unit, fn_str, 'FORMATTED', 'READ', qerr, 0)
    if (qerr) then
       call wrndie(-5, &
            '<api_read%read_psf_card>', &
            'cannot open file ' // fn_str(1:fn_len))
       return
    end if

    blank = ' '
    start = nictot(nseg + 1)
    call seqrdr('', 0, mxcmsz, new_in_unit, &
         titleb, ntitlb, maxtit, &
         res, nres, resid, &
         4, start, blank, blank, &
         0, 0, skip, &
         0, alias, &
         .true., .false., .false., 1)

    call vclose(new_in_unit, 'KEEP', qerr)
    if (qerr) then
       call wrndie(-5, &
            '<api_read%read_psf_card>', &
            'cannot close file ' // filename(1:fn_len))
       return
    end if
    read_sequence_pdb = 1
  end function read_sequence_pdb

  !> @brief make a new sequence from a string of names
  !
  !> @param[in] seq_str a C char array of the names of elts in the new sequence
  !> @param[in] len_seq_str the length of seq_str as a string
  !> @return error code, success is 1
  integer(c_int) function read_sequence_string(seq_str, len_seq_str) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_char
    use api_util, only: c2f_string
    use param_store, only: set_param
    use psf, only: nres, res, resid, nictot, nseg
    use stream, only: outu, wrnlev, iolev, prnlev, idleng
    use string, only: &
         decodi, encodi, indxa, &
         nexti, nextwd, nexta4, &
         filspc, cnvtuc, trima
#ifdef KEY_RESIZE
    use resize, only: resize_psf
#endif

#if KEY_MMFF==1
    use io, only: data_read, sequence
#endif

    implicit none

    ! inputs
    character(len = 1, kind = c_char) :: seq_str(*)
    integer(c_int) :: len_seq_str

    ! local vars
    character(len = len_seq_str) :: seq
    integer :: len_seq

    integer, parameter :: wdmax = 20
    integer :: wdlen
    character(len = wdmax) :: wd

    integer, parameter :: ncheck = 8
    integer :: chknum(ncheck)
    character(len = 4) :: &
         iblank = '    ', &
         check(ncheck) = &
           (/ 'HIS ' &
            , 'HSD ' &
            , 'HSE ' &
            , 'HSP ' &
            , 'ASP ' &
            , 'GLU ' &
            , 'LYS ' &
            , 'TYR ' &
            /)

    integer :: &
         errcnt, cntres, chktot, &
         start, istart, &
         i, j, k, dummy

    if (iolev <= 0) then
       read_sequence_string = 0
       return
    end if

#if KEY_MMFF==1
    data_read = sequence
#endif

    errcnt = 0
    start = nres + 1
    cntres = 0

    istart = nictot(nseg + 1)

    seq = c2f_string(seq_str, len_seq_str)
    len_seq = len_seq_str

    wdlen = 1
    do while (wdlen /= 0)
       call nextwd(seq, len_seq, wd, wdmax, wdlen)
       if (wdlen /= 0) then
          if (wdlen > 8) then
             if (wrnlev >= 2) write(outu, 220)
220          FORMAT(' ***** Error in SEQRDR ***** Residue name longer than', &
                  ' eight characters.')
             errcnt = errcnt + 1
          end if ! wdlen > 8

          call filspc(wd, 8, wdlen)
          cntres = cntres + 1
          res(nres + cntres) = wd
       end if
    end do ! wdlen /= 0

    nres = nres + cntres
#ifdef KEY_RESIZE
    call resize_psf('api_read.F90','read_sequence_string','NRES',nres, .true.)
#endif
    cntres = nres - start + 1

    if (prnlev >= 2) write(outu, 65) cntres, &
         (res(i)(1:idleng), i = start, nres)
65  format(/10X, 'RESIDUE SEQUENCE -- ', I5, ' RESIDUES', /, (10X,20A))

    chktot = 0

    do j = 1, ncheck
       chknum(j) = 0
    end do

    do i = start, nres
       do j = 1, ncheck
          if (res(i) == check(j)) then
             chknum(j) = chknum(j) + 1
             chktot = chktot + 1
          end if
       end do
    end do

    if (chktot > 0 .and. wrnlev >= 2) &
         write(outu, 67) chktot, (check(j), chknum(j), j = 1, ncheck)

67  format(' ***** Message from SEQRDR ***** THE SYSTEM CONTAINS', I3, &
         ' TITRATABLE GROUPS' / &
         ' THE USER MUST PREDETERMINE THE PROTONATION STATE', &
         ' THROUGH THE SEQUENCE AND RTF' / (10(1X, A4, '-', I3, 1X)))

    if (errcnt > 0) then
       if (wrnlev >= 2) write(outu, 250)

250    format(' Execution terminated due to errors in reading ', &
            'the sequence.')

       call diewrn(0)
    else ! define resid array (not for resid or pdb)
       do i = start, nres
          call encodi(i - istart, resid(i), 8, dummy)
       end do
    end if ! errcnt > 0

    call set_param('SQNRES', cntres)

    read_sequence_string = 1
  end function read_sequence_string

end module api_read
