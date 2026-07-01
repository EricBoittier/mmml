!> Build a crystal with any space group symmetry
module api_crystal
  implicit none
contains

#if KEY_LIBRARY == 1
  !> @brief initializes the constants for a new crystal from xtltyp, xucell and xtlref
  !
  ! Call only after setting xtltyp, xucell and xtlref
  !
  !> @return success
  !>            1 if success
  integer function crystal_init() result(success)
    use image, only: cutxtl, imgfrq, xtlabc, xdim, xtltyp, xtlref, xucell
    use number, only: fmark, zero, thirty, ten
    use stream, only: prnlev, outu
    implicit none

    integer :: i = 0

    success = 0

    xtlref(1:6) = xucell(1:6)
    if (cutxtl .eq. fmark) then
       cutxtl = max(xucell(1), xucell(2), xucell(3))
       if (cutxtl .gt. thirty) then
          cutxtl = thirty
       else if (cutxtl .lt. ten) then
          cutxtl = ten
       end if
    end if

    call xtlaxs(xtlabc, xucell)
    call xtlsym(xtlabc, xucell, xtltyp, xdim, xtlref)
    call xtlmsr(xucell)
    if (prnlev .ge. 2) then
       call prnxtld(outu, '    ', xtltyp, xucell, &
            .false., zero, .false., [zero])
    end if

    if (imgfrq .le. 0) imgfrq = 50 ! set default value for imgfrq if not set

    success = 1
  end function crystal_init

  !> @brief defines a cubic lattice and constants for a new crystal
  !
  !> param[in] length
  !>           length of all sides a, b and c
  !> @return success
  !>         1 if success
  integer(c_int) function crystal_define_cubic(length) result(success) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use image, only: xtltyp, xucell
    implicit none

    real(c_double), intent(in) :: length

    integer :: i = 0

    success = 0

    xtltyp = 'CUBI'

    xucell(1:3) = length
    xucell(4:6) = 90.0

    success = crystal_init()
  end function crystal_define_cubic

  !> @brief defines a tetragonal lattice and constants for a new crystal
  !
  !> param[in] length_a
  !>           length of sides a and b
  !> param[in] length_c
  !>           length of side c
  !> @return success
  !>         1 if success
  integer(c_int) function crystal_define_tetra(length_a, length_c) result(success) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use image, only: xtltyp, xucell
    implicit none

    real(c_double), intent(in) :: length_a, length_c

    integer :: i = 0

    success = 0

    xtltyp = 'TETR'

    xucell(1:2) = length_a
    xucell(3) = length_c

    xucell(4:6) = 90.0

    success = crystal_init()
  end function crystal_define_tetra

  !> @brief defines a orthorhombic lattice and constants for a new crystal
  !
  !> param[in] length_a
  !>           length of side a
  !> param[in] length_b
  !>           length of side b
  !> param[in] length_c
  !>           length of side c
  !> @return success
  !>         1 if success
  integer(c_int) function crystal_define_ortho(length_a, length_b, length_c) &
       result(success) bind(c)

    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use image, only: xtltyp, xucell
    implicit none

    real(c_double), intent(in) :: length_a, length_b, length_c

    integer :: i = 0

    success = 0

    xtltyp = 'ORTH'

    xucell(1) = length_a
    xucell(2) = length_b
    xucell(3) = length_c

    xucell(4:6) = 90.0

    success = crystal_init()
  end function crystal_define_ortho

  !> @brief defines a monoclinic lattice and constants for a new crystal
  !
  !> param[in] length_a
  !>           length of side a
  !> param[in] length_b
  !>           length of side b
  !> param[in] length_c
  !>           length of side c
  !> param[in] angle_beta
  !>           measure of angle beta in degrees
  !> @return success
  !>         1 if success
  integer(c_int) function crystal_define_mono(length_a, length_b, length_c, &
       angle_beta) result(success) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use image, only: xtltyp, xucell
    implicit none

    real(c_double), intent(in) :: &
         length_a, length_b, length_c, &
         angle_beta

    integer :: i = 0

    success = 0

    xtltyp = 'MONO'

    xucell(1) = length_a
    xucell(2) = length_b
    xucell(3) = length_c

    xucell(4) = 90.0
    xucell(5) = angle_beta
    xucell(6) = 90.0

    success = crystal_init()
  end function crystal_define_mono

  !> @brief defines a triclinic lattice and constants for a new crystal
  !
  !> param[in] length_a
  !>           length of side a
  !> param[in] length_b
  !>           length of side b
  !> param[in] length_c
  !>           length of side c
  !> param[in] angle_alpha
  !>           measure of angle alpha in degrees
  !> param[in] angle_beta
  !>           measure of angle beta in degrees
  !> param[in] angle_gamma
  !>           measure of angle gamma in degrees
  !> @return success
  !>         1 if success
  integer(c_int) function crystal_define_tri(length_a, length_b, length_c, &
       angle_alpha, angle_beta, angle_gamma) result(success) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use image, only: xtltyp, xucell
    implicit none

    real(c_double), intent(in) :: &
         length_a, length_b, length_c, &
         angle_alpha, angle_beta, angle_gamma

    integer :: i = 0

    success = 0

    xtltyp = 'TRIC'

    xucell(1) = length_a
    xucell(2) = length_b
    xucell(3) = length_c

    xucell(4) = angle_alpha
    xucell(5) = angle_beta
    xucell(6) = angle_gamma

    success = crystal_init()
  end function crystal_define_tri

  !> @brief defines a hexa lattice and constants for a new crystal
  !
  !> param[in] length_a
  !>           length of sides a and b
  !> param[in] length_c
  !>           length of side c
  !> @return success
  !>         1 if success
  integer(c_int) function crystal_define_hexa(length_a, length_c) result(success) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use image, only: xtltyp, xucell
    implicit none

    real(c_double), intent(in) :: length_a, length_c

    integer :: i = 0

    success = 0

    xtltyp = 'HEXA'

    xucell(1:2) = length_a
    xucell(3) = length_c

    xucell(4:5) = 90.0
    xucell(6) = 120.0

    success = crystal_init()
  end function crystal_define_hexa

  !> @brief defines a rhombo lattice and constants for a new crystal
  !
  !> param[in] length
  !>           length of all sides (a, b, c)
  !> param[in] angle_alpha
  !>           measure of all angles in degrees (alpha, beta, gamma)
  !> @return success
  !>         1 if success
  integer(c_int) function crystal_define_rhombo(length, angle) result(success) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use image, only: xtltyp, xucell
    implicit none

    real(c_double), intent(in) :: length, angle

    integer :: i = 0

    success = 0

    xtltyp = 'RHOM'

    xucell(1:3) = length
    xucell(4:6) = angle

    success = crystal_init()
  end function crystal_define_rhombo

  !> @brief defines a octa lattice and constants for a new crystal
  !
  !> param[in] length
  !>           length of all sides (a, b, c)
  !> @return success
  !>         1 if success
  integer(c_int) function crystal_define_octa(length) result(success) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use image, only: xtltyp, xucell
    implicit none

    real(c_double), intent(in) :: length

    integer :: i = 0

    success = 0

    xtltyp = 'OCTA'

    xucell(1:3) = length
    xucell(4:6) = 109.4712206344907

    success = crystal_init()
  end function crystal_define_octa

  !> @brief defines a rhombic dodecahedron lattice and constants for a new crystal
  !
  !> param[in] length
  !>           length of all sides (a, b, c)
  !> @return success
  !>         1 if success
  integer(c_int) function crystal_define_rhdo(length) result(success) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use image, only: xtltyp, xucell
    implicit none

    real(c_double), intent(in) :: length

    integer :: i = 0

    success = 0

    xtltyp = 'RHDO'

    xucell(1:3) = length

    xucell(1) = 60.0
    xucell(2) = 90.0
    xucell(3) = 60.0

    success = crystal_init()
  end function crystal_define_rhdo

  !> @brief parses crystal transformations from an array of strings
  !
  !> param[in] sym_ops
  !>           array of strings representing transformations in (X, Y, Z) format
  !> param[in] nops
  !>           number of transformation strings in sym_ops
  !> @return success
  !>         1 if success
  integer function parse_sym_ops(sym_ops, nops) result(success)
    use chm_kinds, only: chm_real
    use image, only: maxsym, xnsymm, xsymop
    use stream, only: prnlev

    implicit none

    ! args
    integer :: nops
    character(len=80), dimension(1:nops) :: sym_ops

    ! locals
    integer :: i, j
    logical :: qerror = .false.

    success = 0

    xnsymm = nops

    if (xnsymm .ge. maxsym) then
       call wrndie(-5, '<parse_sym_ops>', &
            'Too many symmetry operations specified.')
    end if

    ! initialize symmetry operations array
    do i = 1 ,4
       do j = 1 ,3
          xsymop(j, i, 1) = 0
       end do
    end do

    ! the first symmetry operation is always identity
    do i = 1, 3
       xsymop(i, i, 1) = 1
    end do

    xnsymm = xnsymm + 1
    do i = 2, xnsymm
       call xsympa(sym_ops(i - 1), 80, maxsym, xsymop(1, 1, i), qerror)
       if (qerror) then
          call wrndie(-5, '<XBUILD>', 'Symmetry operation parsing error.')
       end if
    end do

    success = 1
  end function parse_sym_ops
  
  !> @brief build the crystal by repeatedly appling specified transformations
  !
  !> param[in] cutoff
  !>           images within cutoff distance are included in transformation list
  !> param[in] sym_ops
  !>           array of strings representing transformations in (X, Y, Z) format
  !> param[in] nops
  !>           number of transformation strings in sym_ops
  !> @return success
  !>         1 if success
  integer(c_int) function crystal_build(cutoff, sym_ops, nops) result(success) bind(c)
    use, intrinsic :: iso_c_binding, only: c_char, c_f_pointer, c_double, c_int, c_ptr
    use api_util, only: c2f_string
    use bases_fcm, only: bimag
    use chm_kinds, only: chm_real
    use coord, only: x, y, z, wmain
    use energym, only: eprop, volume
    use image, only: cutxtl, xnsymm, xtltyp, xucell
    use image_routines_module, only: inimag, reimag
    use memory, only: chmalloc, chmdealloc
    use prssre, only: getvol
    use psf, only: natom
    use select, only: selcta
    use stream, only: prnlev, outu

#if KEY_DOMDEC==1
    use inbnd,only:cutnb
#endif

    implicit none

    ! args
    real(c_double), intent(in) :: cutoff
    integer(c_int), intent(in) :: nops
    type(c_ptr), target, dimension(nops) :: sym_ops

    ! locals
    character(kind=c_char, len=1), pointer, dimension(:) :: str
    character(len=80), dimension(1:nops) :: f_sym_ops
    integer :: i
    integer, allocatable, dimension(:) :: islct
    real(chm_real), allocatable, dimension(:, :, :) :: transf

    success = 0

    if (xtltyp .eq. '    ') then
       call wrndie(-1,'<parse_sym_ops>', &
            'No crystal type defined. Cannot build.')
       return
    end if

    call chmalloc('api_crystal.F90', 'CRYSTL', 'ISLCT', natom, intg=islct)

    call selcta('', 0, islct, x, y, z, wmain, .true.)
    call inimag(bimag, .true.)
    call reimag(bimag, 0, 0)

    cutxtl = cutoff

#if KEY_DOMDEC==1
    cutnb = cutxtl
#endif

    if (prnlev .ge. 5) write(outu, 45) cutxtl

45  format( &
      ' XBUILD> Building all transformations with a minimum atom-atom', &
      /, '         contact distance of less than', F8.2, ' Angstroms.')

    ! must be redone as parse_sym_ops function in this module
    do i = 1, nops
       call c_f_pointer(sym_ops(i), str, [80])
       f_sym_ops(i) = c2f_string(str, 80)
    end do

    success = parse_sym_ops(f_sym_ops, nops)

    call chmalloc('api_crystal.F90', 'CRYSTL', 'TRANSF', 3, 4, xnsymm, crl=transf)

    call xbuild2(natom, x, y, z, islct, transf)

    call chmdealloc('api_crystal.F90', 'CRYSTL', 'ISLCT', natom, intg=islct)
    call chmdealloc('api_crystal.F90', 'CRYSTL', 'TRANSF', 3, 4, xnsymm, crl=transf)

    call getvol(eprop(volume))
    call xtlmsr(xucell)
  end function crystal_build

#endif /* KEY_LIBRARY */
end module api_crystal
