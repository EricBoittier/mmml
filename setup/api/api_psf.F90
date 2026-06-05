!> export copies of psf data structures
module api_psf
  implicit none
contains

  !> @brief export a copy of natom (number of atoms)
  !
  !> @return integer(c_int) number of atoms from psf
  function psf_get_natom() bind(c) result(n)
    use, intrinsic :: iso_c_binding, only: c_int
    use psf, only: natom
    implicit none
    integer(c_int) :: n
    n = natom
  end function psf_get_natom

  !> @brief export a copy of nres (number of residues)
  !
  !> @return integer(c_int) number of residues from psf
  function psf_get_nres() bind(c) result(n)
    use, intrinsic :: iso_c_binding, only: c_int
    use psf, only: nres
    implicit none
    integer(c_int) :: n
    n = nres
  end function psf_get_nres

  !> @brief export a copy of nseg (number of segments)
  !
  !> @return integer(c_int) number of segments from psf
  function psf_get_nseg() bind(c) result(n)
    use, intrinsic :: iso_c_binding, only: c_int
    use psf, only: nseg
    implicit none
    integer(c_int) :: n
    n = nseg
  end function psf_get_nseg

  !> @brief export a copy of ngrp (number of groups)
  !
  !> @return integer(c_int) number of groups
  function psf_get_ngrp() bind(c) result(n)
    use, intrinsic :: iso_c_binding, only: c_int
    use psf, only: ngrp
    implicit none
    integer(c_int) :: n
    n = ngrp
  end function psf_get_ngrp

  !> @brief export a copy of nbond (number of bonds)
  !
  !> @return integer(c_int) number of bonds
  function psf_get_nbond() bind(c) result(n)
    use, intrinsic :: iso_c_binding, only: c_int
    use psf, only: nbond
    implicit none
    integer(c_int) :: n
    n = nbond
  end function psf_get_nbond

  !> @brief export a copy of iac (param type codes)
  !
  !> @return integer(c_int) array param type codes
  subroutine psf_get_iac(new_iac) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use psf, only: natom, iac

    implicit none

    integer(c_int), dimension(*) :: new_iac

    new_iac(1:natom) = iac(1:natom)
  end subroutine psf_get_iac

  !> @brief export a copy of psf%cg (atom charges)
  !
  !> @return integer(c_double) array atom charges
  subroutine psf_get_charges(out_charges) bind(c)
    use, intrinsic :: iso_c_binding, only: c_double
    use psf, only: natom, cg

    implicit none

    real(c_double), dimension(*) :: out_charges

    out_charges(1:natom) = cg(1:natom)
  end subroutine psf_get_charges

  !> @brief export a copy of amass (atom masses)
  !
  !> @return integer(c_double) array atom masses
  subroutine psf_get_amass(out_amass) bind(c)
    use, intrinsic :: iso_c_binding, only: c_double
    use psf, only: natom, amass

    implicit none

    real(c_double), dimension(*) :: out_amass

    out_amass(1:natom) = amass(1:natom)
  end subroutine psf_get_amass

  !> @brief export a copy of ibase (last atom of each residue)
  !
  !> @return integer(c_int) array last atom of each residue
  subroutine psf_get_ibase(new_ibase) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use psf, only: nres, ibase

    implicit none

    integer(c_int), dimension(*) :: new_ibase
    integer :: n

    n = nres + 1
    new_ibase(1:n) = ibase(1:n)
  end subroutine psf_get_ibase

  !> @brief export a copy of atype (atom name array)
  !
  ! IUPAC Name for each atom
  !
  !> @param[out] out_atype copy of atype array
  subroutine psf_get_atype(out_atype) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_ptr
    use api_util, only: f2c_string
    use psf, only: natom, atype

    implicit none

    type(c_ptr), target, dimension(*) :: out_atype
    integer :: i, nchars

    do i = 1, natom
       nchars = len_trim(atype(i))
       call f2c_string(atype(i), out_atype(i), nchars)
    end do
  end subroutine psf_get_atype

  !> @brief export a copy of res (residue name array)
  !
  ! res has nres entries and each entry is a string of 8 chars
  !
  !> @param[out] out_res copy of residue name array
  subroutine psf_get_res(out_res) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_ptr
    use api_util, only: f2c_string
    use psf, only: nres, res

    implicit none

    type(c_ptr), target, dimension(*) :: out_res
    integer :: i, nchars

    do i = 1, nres
       nchars = len_trim(res(i))
       call f2c_string(res(i), out_res(i), nchars)
    end do
  end subroutine psf_get_res

  !> @brief export a copy of resid (residue identifier array)
  !
  ! resid has nres entries and each entry is a string of 8 chars
  !
  !> @param[out] out_resid copy of resid array
  subroutine psf_get_resid(out_resid) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_ptr
    use api_util, only: f2c_string
    use psf, only: nres, resid

    implicit none

    type(c_ptr), target, dimension(*) :: out_resid
    integer :: i, nchars

    do i = 1, nres
       nchars = len_trim(resid(i))
       call f2c_string(resid(i), out_resid(i), nchars)
    end do
  end subroutine psf_get_resid

  !> @brief export a copy of segid (segment identifier array)
  !
  ! segid has nseg entries and each entry is a string of 8 chars
  !
  !> @param[out] out_segid copy of segid array
  subroutine psf_get_segid(out_segid) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_ptr
    use api_util, only: f2c_string
    use psf, only: nseg, segid

    implicit none

    type(c_ptr), target, dimension(*) :: out_segid
    integer :: i, nchars

    do i = 1, nseg
       nchars = len_trim(segid(i))
       call f2c_string(segid(i), out_segid(i), nchars)
    end do
  end subroutine psf_get_segid

  !> @brief export a copy of nictot (nres for each seg)
  !
  !> @return integer(c_int) array nres for each seg
  subroutine psf_get_nictot(new_nictot) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use psf, only: nseg, nictot

    implicit none

    integer(c_int), dimension(*) :: new_nictot
    integer :: n

    n = nseg + 1
    new_nictot(1:n) = nictot(1:n)
  end subroutine psf_get_nictot

  !> @brief export a copy of igpbs (pointer for 1st atom in each group)
  !
  !> @return integer(c_int) array pointer for 1st atom in each group
  subroutine psf_get_igpbs(new_igpbs) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use psf, only: ngrp, igpbs

    implicit none

    integer(c_int), dimension(*) :: new_igpbs
    integer :: n

    n = ngrp + 1
    new_igpbs(1:n) = igpbs(1:n)
  end subroutine psf_get_igpbs

  !> @brief export a copy of gptyp (code type of each group)
  !
  !> @return integer(c_int) array code type of each group
  subroutine psf_get_igptyp(new_igptyp) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use psf, only: ngrp, igptyp

    implicit none

    integer(c_int), dimension(*) :: new_igptyp

    new_igptyp(1:ngrp) = igptyp(1:ngrp)
  end subroutine psf_get_igptyp

  !> @brief export a copy of ib (1st atom of a bond)
  !
  !> @return integer(c_int) array atom  index for 1st atom of a bond
  subroutine psf_get_ib(new_ib) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use psf, only: ib, nbond

    implicit none

    integer(c_int), dimension(nbond) :: new_ib

    new_ib(1:nbond) = ib(1:nbond)
  end subroutine psf_get_ib

  !> @brief export a copy of jb (2nd atom of a bond)
  !
  !> @return integer(c_int) array atom  index for 2nd atom of a bond
  subroutine psf_get_jb(new_jb) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use psf, only: jb, nbond

    implicit none

    integer(c_int), dimension(nbond) :: new_jb

    new_jb(1:nbond) = jb(1:nbond)
  end subroutine psf_get_jb

  !> @brief delete selected atoms and refrences from psf
  !
  !> @param[in] select_atoms atoms to delete (denoted by 1)
  !> @param[in] isort if 1, then sort the remaining atoms
  !> @return 1 if success
  integer(c_int) function psf_delete_atoms(select_atoms, isort) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use psf, only: natom
    use modpsf, only: delete_atoms

    implicit none

    ! args
    integer(c_int) :: isort, select_atoms(natom)

    ! locals
    logical :: lsort

    lsort = .false.
    if (isort .eq. 1) lsort = .true.

    psf_delete_atoms = -1
    call delete_atoms(select_atoms, lsort, .true.)
    psf_delete_atoms = 1
  end function psf_delete_atoms

  !> @brief delete selected bonds from psf
  !
  ! if atom i and atom j have a bond and
  ! iselect(i) is 1 and jselect(j) is 1
  ! then the bond is deleted
  !
  !> @param[in] iselect atom i of a bond
  !> @param[in] jselect atom j of a bond
  !> @return 1 if success
  integer(c_int) function psf_delete_bonds(iselect, jselect, isort) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use psf, only: natom
    use modpsf, only: delete_bonds

    implicit none

    ! args
    integer(c_int) :: isort, iselect(natom), jselect(natom)

    ! locals
    logical :: lsort

    lsort = .false.
    if (isort .eq. 1) lsort = .true.

    psf_delete_bonds = -1
    call delete_bonds(iselect, jselect, lsort, .false., .true.)
    psf_delete_bonds = 1
  end function psf_delete_bonds

  !> @brief delete selected angles from psf
  !
  !> @param[in] iselect atom i of a angle
  !> @param[in] jselect atom j of a angle
  !> @return 1 if success
  integer(c_int) function psf_delete_angles(iselect, jselect, isort) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use psf, only: natom
    use modpsf, only: delete_angles

    implicit none

    ! args
    integer(c_int) :: isort, iselect(natom), jselect(natom)

    ! locals
    logical :: lsort

    lsort = .false.
    if (isort .eq. 1) lsort = .true.

    psf_delete_angles = -1
    call delete_angles(iselect, jselect, lsort, .false., .true.)
    psf_delete_angles = 1
  end function psf_delete_angles

  !> @brief delete selected dihedrals from psf
  !
  !> @param[in] iselect atom i of a dihedral
  !> @param[in] jselect atom j of a dihedral
  !> @return 1 if success
  integer(c_int) function psf_delete_dihedrals(iselect, jselect, isort) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use psf, only: natom
    use modpsf, only: delete_dihedrals

    implicit none

    ! args
    integer(c_int) :: isort, iselect(natom), jselect(natom)

    ! locals
    logical :: lsort

    lsort = .false.
    if (isort .eq. 1) lsort = .true.

    psf_delete_dihedrals = -1
    call delete_dihedrals(iselect, jselect, lsort, .false., .true.)
    psf_delete_dihedrals = 1
  end function psf_delete_dihedrals

  !> @brief delete selected impropers from psf
  !
  !> @param[in] iselect atom i of a improper
  !> @param[in] jselect atom j of a improper
  !> @return 1 if success
  integer(c_int) function psf_delete_impropers(iselect, jselect, isort) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use psf, only: natom
    use modpsf, only: delete_impropers

    implicit none

    ! args
    integer(c_int) :: isort, iselect(natom), jselect(natom)

    ! locals
    logical :: lsort

    lsort = .false.
    if (isort .eq. 1) lsort = .true.

    psf_delete_impropers = -1
    call delete_impropers(iselect, jselect, lsort, .false., .true.)
    psf_delete_impropers = 1
  end function psf_delete_impropers

  !> @brief delete selected cmaps from psf
  !
  !> @param[in] iselect atom i of a cmap
  !> @param[in] jselect atom j of a cmap
  !> @return 1 if success
  integer(c_int) function psf_delete_cmaps(iselect, jselect, isort) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use psf, only: natom
    use modpsf, only: delete_cmaps

    implicit none

    ! args
    integer(c_int) :: isort, iselect(natom), jselect(natom)

    ! locals
    logical :: lsort

    lsort = .false.
    if (isort .eq. 1) lsort = .true.

    psf_delete_cmaps = -1
    call delete_cmaps(iselect, jselect, lsort, .false., .true.)
    psf_delete_cmaps = 1
  end function psf_delete_cmaps

  !> @brief delete bonds, angles, dihedrals, impropers and cmaps
  !
  !> @param[in] iselect atom i of connectivity
  !> @param[in] jselect atom j of connectivity
  !> @return 1 if success
  integer(c_int) function psf_delete_conn(iselect, jselect, isort) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use psf, only: natom
    use modpsf, only: delete_connectivity

    implicit none

    ! args
    integer(c_int) :: isort, iselect(natom), jselect(natom)

    ! locals
    logical :: lsort

    lsort = .false.
    if (isort .eq. 1) lsort = .true.

    psf_delete_conn = -1
    call delete_connectivity(iselect, jselect, lsort, .false.)
    psf_delete_conn = 1
  end function psf_delete_conn
  
  ! Addition Kai Toepfer May 2022
  !> @brief export a copy of nnb (number of non-bonded exclusions)
  !
  !> @return integer(c_int) number of non-bonded exclusions
  function psf_get_nnb() bind(c) result(n)
    use, intrinsic :: iso_c_binding, only: c_int
    use psf, only: nnb
    implicit none
    integer(c_int) :: n
    n = nnb
  end function psf_get_nnb

  !> @brief Replace atom charges with the values of new_charges
  !
  !> @param[in] new_charges new charge of atoms, 1:natom
  subroutine psf_set_charges(new_charges) bind(c)
    use, intrinsic :: iso_c_binding, only: c_double
    use psf, only: natom, cg

    implicit none

    real(c_double), dimension(*), intent(in) :: new_charges

    cg(1:natom) = new_charges(1:natom)
    
  end subroutine psf_set_charges 

  !> @brief export a copy of psf%inb (non-bonded exclusion)
  !
  !> @return integer(c_int) array of non-bonded exclusion
  subroutine psf_get_iblo_inb(out_iblo, out_inb) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use psf, only: natom, nnb, iblo, inb

    implicit none

    integer(c_int), dimension(*), intent(out) :: out_iblo, out_inb
    
    out_iblo(1:natom) = iblo(1:natom)
    out_inb(1:nnb) = inb(1:nnb)
  end subroutine psf_get_iblo_inb

  !> @brief set psf%(inb,iblo,nnb) (non-bonded exclusion)
  !
  !> @param[in] new_nnb number of atom pairs in inb
  !> @param[in] new_iblo(natom) indexing list for inb
  !> @param[in] new_inb(new_nnb) atom pair definition list
  subroutine psf_set_iblo_inb(new_nnb, new_iblo, new_inb) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use psf, only: natom, nnb, iblo, inb

    implicit none

    integer(c_int), intent(in) :: new_nnb
    integer(c_int), dimension(*), intent(in) :: new_iblo, new_inb

    nnb = new_nnb
    iblo(1:natom) = new_iblo(1:natom)
    inb(1:nnb) = new_inb(1:nnb)
  end subroutine psf_set_iblo_inb
  
end module api_psf
