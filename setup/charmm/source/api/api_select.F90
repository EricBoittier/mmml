!> select sets of atoms to manipulate
module api_select
  implicit none
contains

#if KEY_LIBRARY == 1
  !> @brief select atoms from a range of residues by res name
  !
  !> param[in] flags
  !>            array parallel to atom list, 1 if selected, 0 otherwise
  !> @return success
  !>         1 if success
  function select_print_and_store(flags) result(success)
    use, intrinsic :: iso_c_binding, only: c_int
    use chutil, only: getres, getseg
    use param, only: atc
    use param_store, only: set_param
    use psf, only: iac, atype, nictot, nseg, segid, res, resid, ibase, natom
    use select, only: prntatsl
    use stream, only: prnlev, outu

    implicit none

    ! args
    integer(c_int), intent(in), dimension(*) :: flags

    ! result
    integer(c_int) :: success

    ! locals
    integer :: iseg, ires, iat, nres, nsel
    logical :: is_elt

    success = 0

    if (prnlev >= 6) then
       write(outu, '(a)') &
            ' The following atoms are currently set:'

       call prntatsl(flags, natom, 1, resid, res, ibase, atype, segid, nictot, nseg)
    end if

    do iat = 1, natom
       if (flags(iat) == 1) then
          call set_param('SELATOM', IAT)

          nres = nictot(nseg + 1)
          ires = getres(iat, ibase, nres)
          call set_param('SELIRES', ires)
          call set_param('SELRESI', resid(ires))
          call set_param('SELRESN', res(ires))

          iseg = getseg(ires, nictot, nseg)
          call set_param('SELISEG', iseg)
          call set_param('SELSEGI', segid(iseg))

          call set_param('SELTYPE', atype(iat))

          if (iac(iat) < 1 ) then
             call set_param('SELCHEM', ' ')
          else
             call set_param('SELCHEM', atc(iac(iat)))
          end if

          exit
       end if
    end do

    nsel = sum(flags(1:natom))

    if (prnlev >= 2) then
       write(outu, '(a, i7, a, i7)') &
            ' select>', nsel, &
            ' atoms have been selected out of ', natom
    end if

    call set_param('NSEL', nsel)

    if (nsel == 0) then
       call set_param('SELATOM', 0)
       call set_param('SELIRES', 0)
       call set_param('SELRESI', 'NONE')
       call set_param('SELISEG', 0)
       call set_param('SELSEGI', 'NONE')
       call set_param('SELRESN', 'NONE')
       call set_param('SELTYPE', 'NONE')
       call set_param('SELCHEM', 'NONE')
    end if

    success = 1
  end function select_print_and_store

  !> @brief select atoms from a range of residues by res name
  !
  !> param[in] resname_a
  !>           name of first residue in range
  !> param[in] resname_b
  !>           name of last residue in range
  !> param[out] flags
  !>            array parallel to atom list, 1 if selected, 0 otherwise
  !> @return success
  !>         1 if success
  function select_resname_range(resname_a, resname_b, flags) &
       result(success) bind(c)
    use, intrinsic :: iso_c_binding, only: c_char, c_int
    use api_util, only: c2f_string
    use psf, only: nictot, nseg, res, ibase
    use string, only: eqstwc, ltsteq, trime

    implicit none

    ! args
    character(len=1, kind=c_char), dimension(*) :: resname_a, resname_b
    integer(c_int), dimension(*) :: flags

    ! result
    integer(c_int) :: success

    ! locals
    character(len=8) :: name_a, name_b ! fortran string residue names
    integer :: i, iseg, ires, iat, lres, len_a, len_b
    logical :: qrange, is_elt

    success = 0

    len_a = 8
    name_a = c2f_string(resname_a, len_a)
    call trime(name_a, len_a)

    len_b = 8
    name_b = c2f_string(resname_b, len_b)
    call trime(name_b, len_b)

    qrange = name_a(1:len_a) .ne. name_b(1:len_b)

    do iseg = 1, nseg
       do ires = nictot(iseg) + 1, nictot(iseg + 1)
          lres = 8
          call trime(res(ires), lres)
          if (qrange) then
             is_elt = ltsteq(name_a, len_a, res(ires), lres, .true.)
             if (is_elt) then
                is_elt = ltsteq(res(ires), lres, name_b, len_b, .true.)
             end if
          else
             is_elt = eqstwc(res(ires), lres, name_a, len_a)
          end if

          i = 0
          if (is_elt) i = 1

          do iat = ibase(ires) + 1, ibase(ires + 1)
             flags(iat) = i
          end do
       end do
    end do

    success = select_print_and_store(flags)
  end function select_resname_range

  !> @brief select atoms from a range of residues by res name
  !
  !> param[in] resname_a
  !>           name of first residue in range
  !> param[in] resname_b
  !>           name of last residue in range
  !> param[out] flags
  !>            array parallel to atom list, 1 if selected, 0 otherwise
  !> @return success
  !>         1 if success
  function select_segid_range(segid_a, segid_b, flags) &
       result(success) bind(c)
    use, intrinsic :: iso_c_binding, only: c_char, c_int
    use api_util, only: c2f_string
    use psf, only: nictot, nseg, segid, ibase
    use string, only: eqstwc, ltsteq, trime

    implicit none

    ! args
    character(len=1, kind=c_char), dimension(*) :: segid_a, segid_b
    integer(c_int), dimension(*) :: flags

    ! result
    integer(c_int) :: success

    ! locals
    character(len=8) :: name_a, name_b ! fortran string segment names
    integer :: i, iseg, ires, iat, lseg, len_a, len_b
    logical :: qrange, is_elt

    success = 0

    len_a = 8
    name_a = c2f_string(segid_a, len_a)
    call trime(name_a, len_a)

    len_b = 8
    name_b = c2f_string(segid_b, len_b)
    call trime(name_b, len_b)

    qrange = name_a(1:len_a) .ne. name_b(1:len_b)

    do iseg = 1, nseg
       lseg = 8
       call trime(segid(iseg), lseg)
       if (qrange) then
          is_elt = ltsteq(name_a, len_a, segid(iseg), lseg, .true.)
          if (is_elt) then
             is_elt = ltsteq(segid(iseg), lseg, name_b, len_b, .true.)
          end if
       else
          is_elt = eqstwc(segid(iseg), lseg, name_a, len_a)
       end if

       i = 0
       if (is_elt) i = 1

       do ires = nictot(iseg) + 1, nictot(iseg + 1)
          do iat = ibase(ires) + 1, ibase(ires + 1)
             flags(iat) = i
          end do
       end do
    end do

    success = select_print_and_store(flags)
  end function select_segid_range

  !> @brief test whether atom i is hydrogen
  !
  !> param[in] i
  !>           index of atom to test
  !> @return success
  !>         1 if atom i is hydrogen
  function select_is_hydrog(i) bind(c) result(q)
    use, intrinsic :: iso_c_binding, only: c_int
    use chutil, only: hydrog
    implicit none
    integer :: i
    integer(c_int) :: q
    logical :: test

    q = 0
    test = hydrog(i)
    if (test) q = 1
  end function select_is_hydrog

  !> @brief test whether atom i is lonepair
  !
  !> param[in] i
  !>           index of atom to test
  !> @return success
  !>         1 if atom i is lonepair
  function select_is_lone(i) bind(c) result(q)
    use, intrinsic :: iso_c_binding, only: c_int
    use chutil, only: lone
    implicit none
    integer :: i
    integer(c_int) :: q
    logical :: test

    q = 0
    test = lone(i)
    if (test) q = 1
  end function select_is_lone

  !> @brief test whether atom i has known coords
  !
  !> param[in] i
  !>           index of atom to test
  !> @return success
  !>         1 if atom i has known coords
  function select_is_initial(i) bind(c) result(q)
    use, intrinsic :: iso_c_binding, only: c_int
    use chutil, only: initia
    use coord, only: x, y, z
    use psf, only: natom
    implicit none
    integer :: i
    integer(c_int) :: q
    logical :: test

    q = 0
    if ((i < 1) .or. (i > natom)) return

    test = initia(i, x, y, z)
    if (test) q = 1
  end function select_is_initial

  !> @brief fill out_vals with n numeric property prop of atoms
  !
  !> param[in] prop_str
  !>           an identifier for a numeric property of the atoms
  !> param[out] out_vals
  !>            array that will be filled with values for prop for n atoms
  !> param[in] n
  !>           number of atom prop vals to fetch
  subroutine select_fill_prop(prop_str, out_vals, n)
    use, intrinsic :: iso_c_binding, only: c_double
    use coord, only: x, y, z, wmain
    use coordc, only: xcomp, ycomp, zcomp, wcomp
    use deriv, only: dx, dy, dz
    use param, only: vdwr
    use psf, only: amass, cg

    implicit none

    character(len=*) :: prop_str
    real(c_double) :: out_vals(*)
    integer :: n

    if (prop_str(1:5) == 'wcomp') then
       out_vals(1:n) = wcomp(1:n)
    else if (prop_str(1:5) == 'xcomp') then
       out_vals(1:n) = xcomp(1:n)
    else if (prop_str(1:5) == 'ycomp') then
       out_vals(1:n) = ycomp(1:n)
    else if (prop_str(1:5) == 'zcomp') then
       out_vals(1:n) = zcomp(1:n)
    else if (prop_str(1:5) == 'wmain') then
       out_vals(1:n) = wmain(1:n)
    else if (prop_str(1:4) == 'mass') then
       out_vals(1:n) = amass(1:n)
    else if (prop_str(1:6) == 'charge') then
       out_vals(1:n) = cg(1:n)
    else if (prop_str(1:6) == 'radius') then
       out_vals(1:n) = vdwr(1:n)
    else if (prop_str(1:2) == 'dx') then
       out_vals(1:n) = dx(1:n)
    else if (prop_str(1:2) == 'dy') then
       out_vals(1:n) = dy(1:n)
    else if (prop_str(1:2) == 'dz') then
       out_vals(1:n) = dz(1:n)
    else if (prop_str(1:1) == 'x') then
       out_vals(1:n) = x(1:n)
    else if (prop_str(1:1) == 'y') then
       out_vals(1:n) = y(1:n)
    else if (prop_str(1:1) == 'z') then
       out_vals(1:n) = z(1:n)
    end if
  end subroutine select_fill_prop

  !> @brief fill out_vals with n numeric property prop of atoms
  !
  ! This is a wrapper for select_fill_prop.
  ! The purpose is to clean prop_str first.
  !
  !> param[in] prop_str
  !>           an identifier for a numeric property of the atoms
  !> param[out] out_vals
  !>            array that will be filled with values for prop for n atoms
  !> param[in] n
  !>           number of atom prop vals to fetch
  subroutine select_get_property(prop_str, out_vals, n) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_char, c_double, c_null_char
    implicit none

    character(kind=c_char, len=1), dimension(*) :: prop_str
    real(c_double), dimension(*) :: out_vals
    integer(c_int), value :: n

    character(len=128) :: prop_name
    integer :: i

    logical :: err = .false.

    do i = 1, 128
       if (prop_str(i) .eq. c_null_char) exit

       if (prop_str(i) .ne. ' ') prop_name(i:i) = prop_str(i)
    end do

    if (i .lt. 128) prop_name((i + 1):128) = ' '

    call select_fill_prop(prop_name(1:i), out_vals, n)
  end subroutine select_get_property

  function select_get_max_name() result(out_len) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use selctam, only: mnamsk
    implicit none
    ! output value
    integer(c_int) :: out_len
    out_len = mnamsk
  end function select_get_max_name

  function select_find(name, len_name) bind(c) result(ifound)
    use api_util, only: c2f_string
    use, intrinsic :: iso_c_binding, only: c_char, c_int
    use selctam, only: numsky, lnamsk, mnamsk, namsky
    use string, only: eqst

    implicit none

    ! input args
    character(kind=c_char, len=1) :: name(*)
    integer(c_int), value :: len_name

    ! output value
    integer(c_int) :: ifound

    ! local vars
    integer :: i, len_fname
    character(len=len_name) :: fname

    ifound = 0  ! 0 means not found

    len_fname = len_name
    if (len_name .gt. mnamsk) then
       len_fname = mnamsk
    end if
    fname = c2f_string(name, len_fname)

    do i = 1, numsky
       if (eqst(fname, len_fname, namsky(i), lnamsk(i))) then
          ifound = i
          return
       end if
    end do
  end function select_find

  subroutine select_store(name, len_name, islct, len_islct) bind(c)
    use api_util, only: c2f_string
    use, intrinsic :: iso_c_binding, only: c_char, c_int
    use memory, only: chmalloc, chmdealloc
    use selctam, only: &
         maxsky, mnamsk, &
         numsky, lensky, lnamsk, &
         namsky, ptrsky

    implicit none

    ! input args
    character(kind=c_char, len=1) :: name(*)
    integer(c_int) :: islct(len_islct)
    integer(c_int), value :: len_name, len_islct

    ! local vars
    integer, pointer, dimension(:) :: iptr
    integer :: i, len_fname
    character(len=len_name) :: fname

    len_fname = len_name
    if (len_name .gt. mnamsk) then
       len_fname = mnamsk
    end if
    fname = c2f_string(name, len_fname)

    call chmalloc('api_select.F90', 'api_select%select_store', 'iptr', &
         len_islct, intgp=iptr)
    iptr = islct(1:len_islct)

    i = select_find(name, len_name)
    if ((i .gt. 0) .and. (i .le. numsky)) then
       call chmdealloc('api_select.F90', 'api_select%select_store', &
            'ptrsky(i)%a', lensky(i), intgp=ptrsky(i)%a)
    else if (i .eq. 0) then  ! name not found in store
       if (numsky .eq. maxsky) then
          call chmdealloc('api_select.F90', 'api_select%select_store', &
               'iptr', len_islct, intgp=iptr)
          call wrndie(0, 'api_select%select_store', &
               fname(1:len_fname) // ' selection not stored. ' // &
               'Max selections already reached.')
          return
       end if

       numsky = numsky + 1
       i = numsky

       namsky(i) = fname(1:len_fname)
       lnamsk(i) = len_fname
    else  ! some unanticipated problem
       call chmdealloc('api_select.F90', 'api_select%select_store', &
            'iptr', len_islct, intgp=iptr)
       call wrndie(0, 'api_select%select_store', &
            fname(1:len_fname) // ' selection not stored. ' // &
            'There is an unkown problem.')
       return
    end if

    ptrsky(i)%a => iptr
    lensky(i) = len_islct
  end subroutine select_store

  ! test this and commit it
  subroutine select_delete(name, len_name) bind(c)
    use api_util, only: c2f_string
    use, intrinsic :: iso_c_binding, only: c_char, c_int
    use memory, only: chmdealloc
    use selctam, only: &
         maxsky, mnamsk, &
         numsky, lensky, lnamsk, &
         namsky, ptrsky

    implicit none

    ! input args
    character(kind=c_char, len=1) :: name(*)
    integer(c_int), value :: len_name

    ! local vars
    integer :: del_i, j, len_fname
    character(len=len_name) :: fname

    len_fname = len_name
    if (len_name .gt. mnamsk) then
       len_fname = mnamsk
    end if
    fname = c2f_string(name, len_fname)

    del_i = select_find(name, len_name)
    if ((del_i .le. 0) .or. (del_i .gt. numsky)) then
       call wrndie(0, 'api_select%select_store', &
            fname(1:len_fname) // ' selection not found. ' // &
            'Check the name and try again.')
       return
    end if

    call chmdealloc('api_select.F90', 'api_select%select_store', &
         'ptrsky(del_i)%a', lensky(del_i), intgp=ptrsky(del_i)%a)
    nullify(ptrsky(del_i)%a)
    lensky(del_i) = 0
    namsky(del_i) = ' '
    lnamsk(del_i) = 0

    numsky = numsky - 1
    do j = del_i, numsky
       ptrsky(j)%a => ptrsky(j + 1)%a
       lensky(j) = lensky(j + 1)
       namsky(j) = namsky(j + 1)
       lnamsk(j) = lnamsk(j + 1)

       nullify(ptrsky(j + 1)%a)
       lensky(j + 1) = 0
       namsky(j + 1) = ' '
       lnamsk(j + 1) = 0
    end do
  end subroutine select_delete

  function select_get_num_stored() bind(c) result(n)
    use, intrinsic :: iso_c_binding, only: c_int
    use selctam, only: numsky
    implicit none
    integer(c_int) :: n
    n = numsky
  end function select_get_num_stored

  function select_get_stored_names(out_names, max_names, max_name_len) bind(c) result(n_names)
    use, intrinsic :: iso_c_binding, only: c_int, c_ptr
    use api_util, only: f2c_string
    use selctam, only: &
         maxsky, mnamsk, &
         numsky, lensky, lnamsk, &
         namsky, ptrsky

    implicit none

    ! output argument
    type(c_ptr), target, dimension(*) :: out_names

    ! input arguments
    integer(c_int), value :: max_names, max_name_len

    ! return value
    integer(c_int) :: n_names

    ! local variables
    integer :: i, nchars

    n_names = min(numsky, max_names)

    do i = 1, n_names
       nchars = min(lnamsk(i), max_name_len)
       call f2c_string(namsky(i), out_names(i), nchars)
    end do
  end function select_get_stored_names
#endif /* KEY_LIBRARY */
end module api_select
