!> allow api access to dimens variables like chsize and maxa
module api_dimens
  implicit none
contains

  !> @brief print the dimens vars like chsize and maxa to stdout or outu
  subroutine dimens_print() bind(c)
    implicit none
    call print_charmm_sizes() ! lives in iniall.F90
  end subroutine dimens_print

  !> @brief set chsize, the general charmm size variable for particle storage
  !
  !> @param[in] in_chsize new size of storage, affects most arrays in dimens
  subroutine dimens_set_chsize(in_chsize) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use dimens_fcm, only: set_chsize
    implicit none
    integer(c_int) :: in_chsize
    call set_chsize(in_chsize)
  end subroutine dimens_set_chsize

  !> @brief set maxa, the max number of atoms
  !
  !> @param[in] in_maxa new max number of atoms
  subroutine dimens_set_maxa(in_maxa) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use dimens_fcm, only: new_chsize, set_dimen
    implicit none
    integer(c_int) :: in_maxa
    call set_dimen(new_chsize%maxa, in_maxa)
  end subroutine dimens_set_maxa

  !> @brief set maxb, the max number of bonds
  !
  !> @param[in] in_maxb new max number of bonds
  subroutine dimens_set_maxb(in_maxb) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use dimens_fcm, only: new_chsize, set_dimen
    implicit none
    integer(c_int) :: in_maxb
    call set_dimen(new_chsize%maxb, in_maxb)
  end subroutine dimens_set_maxb

  !> @brief set maxt, the max number of angles
  !
  !> @param[in] in_maxt new max number of angles
  subroutine dimens_set_maxt(in_maxt) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use dimens_fcm, only: new_chsize, set_dimen
    implicit none
    integer(c_int) :: in_maxt
    call set_dimen(new_chsize%maxt, in_maxt)
  end subroutine dimens_set_maxt

  !> @brief set maxp, the max number of proper dihedral angles
  !
  !> @param[in] in_maxp new max number of proper dihedral angles
  subroutine dimens_set_maxp(in_maxp) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use dimens_fcm, only: new_chsize, set_dimen
    implicit none
    integer(c_int) :: in_maxp
    call set_dimen(new_chsize%maxp, in_maxp)
  end subroutine dimens_set_maxp

  !> @brief set maximp, the max number of improper dihedral angles
  !
  !> @param[in] in_maximp new max number of improper dihedral angles
  subroutine dimens_set_maximp(in_maximp) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use dimens_fcm, only: new_chsize, set_dimen
    implicit none
    integer(c_int) :: in_maximp
    call set_dimen(new_chsize%maximp, in_maximp)
  end subroutine dimens_set_maximp

  !> @brief set maxnb, the max number of explicit nonbond exclusions
  !
  !> @param[in] in_maxnb new max number of explicit nonbond exclusions
  subroutine dimens_set_maxnb(in_maxnb) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use dimens_fcm, only: new_chsize, set_dimen
    implicit none
    integer(c_int) :: in_maxnb
    call set_dimen(new_chsize%maxnb, in_maxnb)
  end subroutine dimens_set_maxnb

  !> @brief set maxpad, the max number of acceptors and donors
  !
  !> @param[in] in_maxpad new max number of acceptors and donors
  subroutine dimens_set_maxpad(in_maxpad) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use dimens_fcm, only: new_chsize, set_dimen
    implicit none
    integer(c_int) :: in_maxpad
    call set_dimen(new_chsize%maxpad, in_maxpad)
  end subroutine dimens_set_maxpad

  !> @brief set maxres, the max number of residues
  !
  !> @param[in] in_maxres new max number of residues
  subroutine dimens_set_maxres(in_maxres) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use dimens_fcm, only: new_chsize, set_dimen
    implicit none
    integer(c_int) :: in_maxres
    call set_dimen(new_chsize%maxres, in_maxres)
  end subroutine dimens_set_maxres

  !> @brief set maxseg, the max number of segments
  !
  !> @param[in] in_maxseg new max number of segments
  subroutine dimens_set_maxseg(in_maxseg) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use dimens_fcm, only: new_chsize, set_dimen
    implicit none
    integer(c_int) :: in_maxseg
    call set_dimen(new_chsize%maxseg, in_maxseg)
  end subroutine dimens_set_maxseg

  !> @brief set maxcrt, the max number of CMAP dihedrals
  !
  !> @param[in] in_maxcrt new max number of CMAP dihedrals
  subroutine dimens_set_maxcrt(in_maxcrt) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use dimens_fcm, only: new_chsize, set_dimen
    implicit none
    integer(c_int) :: in_maxcrt
#if KEY_CMAP == 1
    call set_dimen(new_chsize%maxcrt, in_maxcrt)
#else
    call wrndie(-1, 'api_dimens%dimens_set_maxcrt', &
         'CMAP code is not compiled; prefx keyword CMAP removed.')
#endif /* KEY_CMAP */
  end subroutine dimens_set_maxcrt

  !> @brief set maxshk, the max number of SHAKE contraints
  !
  !> @param[in] in_maxshk new max number of SHAKE contraints
  subroutine dimens_set_maxshk(in_maxshk) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use dimens_fcm, only: new_chsize, set_dimen
    implicit none
    integer(c_int) :: in_maxshk
    call set_dimen(new_chsize%maxshk, in_maxshk)
  end subroutine dimens_set_maxshk

  !> @brief set maxaim, the max number of atoms including images
  !
  !> @param[in] in_maxaim new max number of atoms including images
  subroutine dimens_set_maxaim(in_maxaim) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use dimens_fcm, only: new_chsize, set_dimen
    implicit none
    integer(c_int) :: in_maxaim
    call set_dimen(new_chsize%maxaim, in_maxaim)
  end subroutine dimens_set_maxaim

  !> @brief set maxgrp, the max number of groups
  !
  !> @param[in] in_maxgrp new max number of groups
  subroutine dimens_set_maxgrp(in_maxgrp) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use dimens_fcm, only: new_chsize, set_dimen
    implicit none
    integer(c_int) :: in_maxgrp
    call set_dimen(new_chsize%maxgrp, in_maxgrp)
  end subroutine dimens_set_maxgrp

  !> @brief set maxnbf, the max number of nonbond fixes
  !
  !> @param[in] in_maxnbf new max number of nonbond fixes
  subroutine dimens_set_maxnbf(in_maxnbf) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use dimens_fcm, only: new_chsize, set_dimen
    implicit none
    integer(c_int) :: in_maxnbf
    call set_dimen(new_chsize%maxnbf, in_maxnbf)
  end subroutine dimens_set_maxnbf

  !> @brief set maxitc, the max number of atom type codes
  !
  ! This will affect the maximum number of vdw lookup values
  !
  !> @param[in] in_maxitc new max number of atom type codes
  subroutine dimens_set_maxitc(in_maxitc) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use dimens_fcm, only: new_chsize, set_dimen
    implicit none
    integer(c_int) :: in_maxitc
    call set_dimen(new_chsize%maxitc, in_maxitc)
    call set_dimen(new_chsize%maxcn, &
         in_maxitc * (in_maxitc + 1) / 2)
  end subroutine dimens_set_maxitc

  subroutine dimens_set_iatbmx(in_iatbmx) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use dimens_fcm, only: new_chsize, set_dimen
    implicit none
    integer(c_int) :: in_iatbmx
    call set_dimen(new_chsize%iatbmx, in_iatbmx)
  end subroutine dimens_set_iatbmx

  ! MAXPAR is the command-parser token-table size (cmdpar), NOT a chsizes
  ! dimension. pyCHARMM needs to grow it so large lingo.charmm_script inputs
  ! do not overflow the table. Called from communicate_dimens BEFORE
  ! init_charmm (so cmdpar's arrays are not yet allocated -> a plain size set
  ! is enough; cmdpar_init allocates with it). If already allocated (post-init
  ! use), cmdpar_reinit grows it in place, preserving existing entries.
  subroutine cmdpar_set_maxpar(in_maxpar) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use cmdpar, only: maxpar, toknam, cmdpar_reinit
    implicit none
    integer(c_int) :: in_maxpar
    if (allocated(toknam)) then
       call cmdpar_reinit(in_maxpar)
    else
       maxpar = in_maxpar
    end if
  end subroutine cmdpar_set_maxpar
end module api_dimens
