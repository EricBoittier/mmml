!> routines for moving some or all of the atoms
module api_coor

  implicit none

contains


  !> @brief returns current number of atoms in simulation
  !
  !> @returns integer(c_int) number of atoms in simulation
  integer(c_int) function coor_get_natom() bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use psf, only: natom

    implicit none

    coor_get_natom = natom
  end function coor_get_natom

  !> @brief fills the out_* with atom positions
  !
  !> @param[out] out_x x component of atom position, 1:natom
  !> @param[out] out_y y component of atom position, 1:natom
  !> @param[out] out_z z component of atom position, 1:natom
  !> @returns integer(c_int) number of atoms copied to output parameters
  integer(c_int) function coor_get_positions(out_x, out_y, out_z) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use coord, only: x, y, z
    use psf, only: natom

    implicit none

    real(c_double), dimension(*), intent(out) :: out_x, out_y, out_z

    out_x(1:natom) = x(1:natom)
    out_y(1:natom) = y(1:natom)
    out_z(1:natom) = z(1:natom)

    coor_get_positions = natom
  end function coor_get_positions

  !> @brief sets charmm atom positions from in_* arrays of size 1:natom
  !
  !> @param[in] in_x x component of atom position, 1:natom
  !> @param[in] in_y y component of atom position, 1:natom
  !> @param[in] in_z z component of atom position, 1:natom
  !> @returns integer(c_int) number of atoms copied to charmm x, y, z arrays
  integer(c_int) function coor_set_positions(in_x, in_y, in_z) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use coord, only: x, y, z
    use psf, only: natom

    implicit none

    real(c_double), dimension(*), intent(in) :: in_x, in_y, in_z

    x(1:natom) = in_x(1:natom)
    y(1:natom) = in_y(1:natom)
    z(1:natom) = in_z(1:natom)

    coor_set_positions = natom
  end function coor_set_positions

  !> @brief fills the out_* with atom weights
  !
  !> @param[out] out_weights w component or wmain, 1:natom
  !> @returns integer(c_int) number of atoms copied to output parameters
  integer(c_int) function coor_get_weights(out_weights) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use coord, only: wmain
    use psf, only: natom

    implicit none

    real(c_double), dimension(*), intent(out) :: out_weights

    out_weights(1:natom) = wmain(1:natom)
    coor_get_weights = natom
  end function coor_get_weights

  !> @brief sets charmm atom weights from in_weights 1:natom
  !
  !> @param[in] in_weights w component or atom weight, 1:natom
  !> @returns integer(c_int) number of atoms copied to charmm x, y, z arrays
  integer(c_int) function coor_set_weights(in_weights) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use coord, only: wmain
    use psf, only: natom

    implicit none

    real(c_double), dimension(*), intent(in) :: in_weights

    wmain(1:natom) = in_weights(1:natom)
    coor_set_weights = natom
  end function coor_set_weights

  !> @brief fills the out_* with atom forces
  !
  !> @param[out] out_dx x component of atom force, 1:natom
  !> @param[out] out_dy y component of atom force, 1:natom
  !> @param[out] out_dz z component of atom force, 1:natom
  !> @returns integer(c_int) number of atoms copied to output parameters
  integer(c_int) function coor_get_forces(out_dx, out_dy, out_dz) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use deriv, only: dx, dy, dz
    use psf, only: natom

    implicit none

    real(c_double), dimension(*), intent(out) :: out_dx, out_dy, out_dz

    out_dx(1:natom) = dx(1:natom)
    out_dy(1:natom) = dy(1:natom)
    out_dz(1:natom) = dz(1:natom)

    coor_get_forces = natom
  end function coor_get_forces

  !> @brief sets charmm atom forces from in_* arrays of size 1:natom
  !
  !> @param[in] in_dx x component of atom force, 1:natom
  !> @param[in] in_dy y component of atom force, 1:natom
  !> @param[in] in_dz z component of atom force, 1:natom
  !> @returns integer(c_int) number of atoms copied to charmm x, y, z arrays
  integer(c_int) function coor_set_forces(in_dx, in_dy, in_dz) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use deriv, only: dx, dy, dz
    use psf, only: natom

    implicit none

    real(c_double), dimension(*), intent(in) :: in_dx, in_dy, in_dz

    dx(1:natom) = in_dx(1:natom)
    dy(1:natom) = in_dy(1:natom)
    dz(1:natom) = in_dz(1:natom)

    coor_set_forces = natom
  end function coor_set_forces

  !> @brief fills the out_* with atom comparison set
  !
  !> @param[out] out_x x component 1:natom
  !> @param[out] out_y y component 1:natom
  !> @param[out] out_z z component 1:natom
  !> @returns integer(c_int) number of atoms copied to output parameters
  integer(c_int) function coor_get_comparison(out_x, out_y, out_z, out_w) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use coordc, only: xcomp, ycomp, zcomp, wcomp
    use psf, only: natom

    implicit none

    real(c_double), dimension(*), intent(out) :: out_x, out_y, out_z, out_w

    out_x(1:natom) = xcomp(1:natom)
    out_y(1:natom) = ycomp(1:natom)
    out_z(1:natom) = zcomp(1:natom)
    out_w(1:natom) = wcomp(1:natom)

    coor_get_comparison = natom
  end function coor_get_comparison

  !> @brief charmm atom comp set from in_* arrays 1:natom
  !
  !> @param[in] in_x x component 1:natom
  !> @param[in] in_y y component 1:natom
  !> @param[in] in_z z component 1:natom
  !> @returns integer(c_int) number atoms to charmm xcomp, ycomp, zcomp
  integer(c_int) function coor_set_comparison(in_x, in_y, in_z, in_w) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use coordc, only: xcomp, ycomp, zcomp, wcomp
    use psf, only: natom

    implicit none

    real(c_double), dimension(*), intent(in) :: in_x, in_y, in_z, in_w

    xcomp(1:natom) = in_x(1:natom)
    ycomp(1:natom) = in_y(1:natom)
    zcomp(1:natom) = in_z(1:natom)
    wcomp(1:natom) = in_w(1:natom)

    coor_set_comparison = natom
  end function coor_set_comparison

  !> @brief fills the out_* with atom comp2 set
  !
  !> @param[out] out_x x component 1:natom
  !> @param[out] out_y y component 1:natom
  !> @param[out] out_z z component 1:natom
  !> @returns integer(c_int) number of atoms copied to output parameters
  integer(c_int) function coor_get_comp2(out_x, out_y, out_z, out_w) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
#if KEY_COMP2==1
    use coordc, only: xcomp2, ycomp2, zcomp2, wcomp2
#endif
    use psf, only: natom

    implicit none

    real(c_double), dimension(*), intent(out) :: out_x, out_y, out_z, out_w

#if KEY_COMP2==1
    out_x(1:natom) = xcomp2(1:natom)
    out_y(1:natom) = ycomp2(1:natom)
    out_z(1:natom) = zcomp2(1:natom)
    out_w(1:natom) = wcomp2(1:natom)

    coor_get_comp2 = natom
#else
    call wrndie(-3, '<coor_get_comp2>', &
         'support for second comparison sets not included')
    coor_get_comp2 = 0
#endif
  end function coor_get_comp2

  !> @brief charmm atom comp2 set from in_* arrays 1:natom
  !
  !> @param[in] in_x x component 1:natom
  !> @param[in] in_y y component 1:natom
  !> @param[in] in_z z component 1:natom
  !> @returns integer(c_int) number atoms to charmm xcomp, ycomp, zcomp
  integer(c_int) function coor_set_comp2(in_x, in_y, in_z, in_w) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
#if KEY_COMP2==1
    use coordc, only: xcomp2, ycomp2, zcomp2, wcomp2
#endif
    use psf, only: natom

    implicit none

    real(c_double), dimension(*), intent(in) :: in_x, in_y, in_z, in_w

#if KEY_COMP2==1
    xcomp2(1:natom) = in_x(1:natom)
    ycomp2(1:natom) = in_y(1:natom)
    zcomp2(1:natom) = in_z(1:natom)
    wcomp2(1:natom) = in_w(1:natom)

    coor_set_comp2 = natom
#else
    call wrndie(-3, '<coor_set_comp2>', &
         'support for second comparison sets not included')
    coor_set_comp2 = 0
#endif
  end function coor_set_comp2

  !> @brief fills the out_* with atom main set
  !
  !> @param[out] out_x x component 1:natom
  !> @param[out] out_y y component 1:natom
  !> @param[out] out_z z component 1:natom
  !> @returns integer(c_int) number of atoms copied to output parameters
  integer(c_int) function coor_get_main(out_x, out_y, out_z, out_w) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use coord, only: x, y, z, wmain
    use psf, only: natom

    implicit none

    real(c_double), dimension(*), intent(out) :: out_x, out_y, out_z, out_w

    out_x(1:natom) = x(1:natom)
    out_y(1:natom) = y(1:natom)
    out_z(1:natom) = z(1:natom)
    out_w(1:natom) = wmain(1:natom)

    coor_get_main = natom
  end function coor_get_main

  !> @brief charmm atom main set from in_* arrays 1:natom
  !
  !> @param[in] in_x x component 1:natom
  !> @param[in] in_y y component 1:natom
  !> @param[in] in_z z component 1:natom
  !> @returns integer(c_int) number atoms to charmm xcomp, ycomp, zcomp
  integer(c_int) function coor_set_main(in_x, in_y, in_z, in_w) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use coord, only: x, y, z, wmain
    use psf, only: natom

    implicit none

    real(c_double), dimension(*), intent(in) :: in_x, in_y, in_z, in_w

    x(1:natom) = in_x(1:natom)
    y(1:natom) = in_y(1:natom)
    z(1:natom) = in_z(1:natom)
    wmain(1:natom) = in_w(1:natom)

    coor_set_main = natom
  end function coor_set_main

  !> @brief modifies coordinates of all atoms according to the passed flags
  !
  !> @param[in] mass_flag if not zero, coords are mass weighted
  !> @param[in] rms_flag if not zero, use other coord set as rotation refrence
  !> @param[in] noro_flag if not zero, supress rotations, modify one coord set
  integer(c_int) function coor_orient(mass_flag, rms_flag, noro_flag) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int

    use coord, only: x, y, z, wmain
    use coordc, only: xcomp, ycomp, zcomp
    use corsubs, only: orintc
    use memory, only: chmalloc, chmdealloc
    use number, only: anum
    use parallel, only: psnd8
    use psf, only: natom, amass
    use stream, only: prnlev, wrnlev, outu
    implicit none

    ! args
    integer(c_int) :: mass_flag, rms_flag, noro_flag

    ! local vars
    logical :: lmass, lrms, lnoro
    integer :: i, nslct, nmiss

    integer, allocatable, dimension(:) :: islct
    integer, allocatable, dimension(:, :) :: iscr2dim

    coor_orient = -1

    nmiss = 0

    lmass = mass_flag .ne. 0
    lrms = rms_flag .ne. 0
    lnoro = noro_flag .ne. 0

    call chmalloc('corman.src', 'corman', 'islct', natom, intg = islct)
    islct = 1
    nslct = natom

    do i = 1, natom
       if (x(i) == anum) then
          islct(i) = 0
          nmiss = nmiss + 1
       else if (lrms .and. xcomp(i) == anum) then
          islct(i) = 0
          nmiss = nmiss + 1
       end if
    end do

    nslct = nslct - nmiss
    if (nslct <= 0) then
       if(prnlev >= 3) write(outu, '(x, a)') &
            '**WARNING** ALL SELECTED COORDINATES UNDEFINED'
    else
       call chmalloc('corman.src', 'corman', 'iscr2dim', &
            2, natom, intg = iscr2dim)

       call orintc(natom, &
            x, y, z, &
            xcomp, ycomp, zcomp, &
            amass, lmass, lrms, &
            iscr2dim, islct, &
            .false., wmain, lnoro, .true.)

       call chmdealloc('corman.src', 'corman', 'iscr2dim', &
            2, natom, intg=iscr2dim)
       call chmdealloc('corman.src', 'corman', 'islct', natom, intg = islct)

       if (prnlev >= 3) then
          if (lnoro) then
             write(outu, '(x, a, x, a, x, a)') &
                  'SELECTED COORDINATES TRANSLATED IN THE', &
                  'MAIN', &
                  'SET.'
          else
             write(outu, '(x, a, x, a, x, a)') &
                  'ALL COORDINATES ORIENTED IN THE', 'MAIN', &
                  'SET BASED ON SELECTED ATOMS.'
          end if
       end if
    end if

#if KEY_PARALLEL==1
    call psnd8(x, natom)
    call psnd8(y, natom)
    call psnd8(z, natom)
#endif

    if (nmiss > 0 .and. wrnlev >= 2) write(outu, '(x, a, x, i5, x, a)') &
         '**** WARNING **** FOR THIS OPERATION, THERE WERE', nmiss, &
         'MISSING COORDINATES'

    coor_orient = 1
  end function coor_orient

  !> @brief print the main coordinate set of the atoms
  integer(c_int) function coor_print() bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use, intrinsic :: iso_fortran_env, only: output_unit

    use coord, only: x, y, z, wmain
    use coorio_mod, only: cwrite
    use ctitla, only: titlea, ntitla
    use psf, only: natom, res, atype, ibase, nictot, nseg

    implicit none

    ! local vars
    integer :: islct(natom), icntrl(20), nres, ntitle
    character(len = 80) :: title

    coor_print = -1

    islct = 1
    icntrl = 0
    nres = nictot(nseg + 1)

    call cwrite(output_unit, titlea, ntitla, icntrl, &
         x, y, z, wmain, &
         res, atype, ibase, nres, &
         natom, islct, &
         3, 0, 0, .false., -1)

    coor_print = 1
  end function coor_print

  !> @brief print the comparison set of the atoms
  integer(c_int) function coor_print_comp() bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use, intrinsic :: iso_fortran_env, only: output_unit

    use coordc, only: xcomp, ycomp, zcomp, wcomp
    use coorio_mod, only: cwrite
    use ctitla, only: titlea, ntitla
    use psf, only: natom, res, atype, ibase, nictot, nseg

    implicit none

    ! local vars
    integer :: islct(natom), icntrl(20), nres, ntitle
    character(len = 80) :: title

    coor_print_comp = -1

    islct = 1
    icntrl = 0
    nres = nictot(nseg + 1)

    call cwrite(output_unit, titlea, ntitla, icntrl, &
         xcomp, ycomp, zcomp, wcomp, &
         res, atype, ibase, nres, &
         natom, islct, &
         3, 0, 0, .false., -1)

    coor_print_comp = 1
  end function coor_print_comp

  !> @brief copy forces to comparison set
  integer(c_int) function coor_copy_forces(imass, iselection) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int

    use chutil, only: lone
    use coordc, only: xcomp, ycomp, zcomp
    use deriv, only: dx, dy, dz
    use psf, only: amass, natom
    use stream, only: outu, prnlev

#if KEY_DOMDEC == 1
    use domdec_common, only: q_domdec
    use domdec_d2d_comm, only: copy_to_all
#endif

    implicit none

    ! args
    integer(c_int) :: imass, iselection(*)

    ! local vars
    logical :: qmass, qselection(natom)
    integer :: i

    qmass = .false.
    if (imass .gt. 0) qmass = .true.

    qselection = .false.
    do i = 1, natom
       if (iselection(i) .eq. 1) qselection(i) = .true.
    end do

    coor_copy_forces = -1
    xcomp = 0.0
    ycomp = 0.0
    zcomp = 0.0
    do i = 1, natom
       if(qselection(i)) then
          xcomp(i) = dx(i)
          ycomp(i) = dy(i)
          zcomp(i) = dz(i)
          if(qmass .and. .not. lone(i)) then
             xcomp(i) = xcomp(i) / amass(i)
             ycomp(i) = ycomp(i) / amass(i)
             zcomp(i) = zcomp(i) / amass(i)
          end if
       end if
    end do

#if KEY_PARALLEL==1
#if KEY_DOMDEC==1
       if (q_domdec) then
          call copy_to_all(xcomp, ycomp, zcomp)
       else
#endif /* KEY_DOMDEC */

#if KEY_SPACDEC==1
          call spacbr(xcomp, natom, icpumap)
          call spacbr(ycomp, natom, icpumap)
          call spacbr(zcomp, natom, icpumap)
#else
          call vdgbr(xcomp, ycomp, zcomp, 1)
#endif  /* KEY_SPACDEC */

#if KEY_DOMDEC==1
       end if
#endif /* KEY_DOMDEC */
#endif /* KEY_PARALLEL */

       if (prnlev >= 3) then
          write(outu, '(a)') 'selected forces copied to the comparison set'
       end if
       coor_copy_forces = 1
  end function coor_copy_forces

  !> @brief max, min, ave for x, y, z, w over selection of main or comp sets
  !
  !> @param[in] selection selection(i) == 1 <==> atom i is selected for stats
  !> @param[in] qcomp 1 ==> use comparison set for stats
  !> @param[in] qmass 1 ==> place ave values at center of mass
  !> @param[out] out_names C strings of names for each stat value
  !> @param[out] out_vals numeric stat values
  !> @param[out] out_n_selected how many atoms were selected
  !> @param[out] out_n_misses how many selected atoms had to be skipped
  !> @returns status 1 ==> success
  function coor_stat(selection, qcomp, qmass, &
       out_names, out_vals, out_n_selected, out_n_misses) &
       bind(c) result(status)
    use, intrinsic :: iso_c_binding, only: c_double, c_int, &
         c_ptr, c_associated
    use api_util, only: f2c_string, c2f_logical
    use chm_types, only: chm_real
    use coord, only: x, y, z, wmain
    use coordc, only: xcomp, ycomp, zcomp, wcomp
    use corman_mod, only: calc_coor_stats
    use psf, only: natom, amass

    implicit none

    ! input params
    integer(c_int) :: qcomp, qmass, selection(*)

    ! output params
    type(c_ptr), target :: out_names(*)
    real(c_double) :: out_vals(*)
    integer(c_int) :: out_n_selected, out_n_misses

    ! result
    integer(c_int) :: status

    ! locals
    integer :: i
    integer, parameter :: nstats = 13, nchars = 4
    character(len=4) :: f_names(nstats)
    real(chm_real) :: f_vals(nstats)
    logical :: lcomp, lmass

    status = -1

    lcomp = c2f_logical(qcomp)
    lmass = c2f_logical(qmass)

    if (lcomp) then
       call calc_coor_stats(xcomp, ycomp, zcomp, wcomp, &
            amass, natom, selection, lmass, &
            f_names, f_vals, out_n_selected, out_n_misses)
    else
       call calc_coor_stats(x, y, z, wmain, &
            amass, natom, selection, lmass, &
            f_names, f_vals, out_n_selected, out_n_misses)
    end if

    out_vals(1:nstats) = f_vals(1:nstats)

    do i = 1, nstats
       call f2c_string(f_names(i), out_names(i), nchars)
    end do

    status = 1
  end function coor_stat

  !> @brief Returns whether the inpit pair of atoms are excluded, or 14excluded
  !> @brief This function is largely copied from manip/corman3%qexclt for use
  !> @brief with functions that need to know if two atoms are in an exclusion list.
  !
  !> @param[in] in_i <==> atom index i
  !> @param[in] in_j <==> atom index j
  !> @returns status 1, 2 or 3 (see manip/corman3.F90 for their meanings)
  integer(c_int) function qexclt(in_i, in_j) bind(c) result(status)
    use, intrinsic :: iso_c_binding, only: c_int
    use psf, only: natom
    use inbnd, only: nnb14
    use bases_fcm, only: bnbnd

    implicit none

    ! input params
    integer(c_int), VALUE, intent(in) :: in_i, in_j
    ! local variables
    integer(c_int) IS, IQ, IPT, I, J

    if ( in_i == in_j ) then
        ! an atom is always excluded to itself
        status = 1
        return
    endif
    
    status = 3
    ! if exclusion list is not present, ignore exclusions
    if ( nnb14 <= 0 ) return
    ! if images are used, don't worry about image exclusions...
    if ( in_i > natom .OR. in_j > natom ) return

    if ( in_i > in_j ) then
        I = in_j   
        J = in_i   
    else
        I = in_i 
        J = in_j 
    endif

    if ( I == 1 ) then
        IS = 1
    else
        IS = bnbnd%iblo14(I-1)+1
    endif

    IQ =  bnbnd%iblo14(I)
    do IPT = IS, IQ
        if ( bnbnd%inb14(IPT) == J ) then
            status = 1
            return
        endif
        if ( bnbnd%inb14(IPT) == -J ) then
            status = 2
            return
        endif
    enddo
    return

  end function qexclt




end module api_coor
