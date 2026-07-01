!> routines to configure nonbonded interactions
module api_nbonds
  implicit none
contains

#if KEY_LIBRARY == 1
  !> @brief set update frequency for the nonbonded list
  !
  ! set the update frequency for the non-bonded list. Used in the
  ! subroutine ENERGY() to decide updates. When set to :
  !  0 --> no updates will be done.
  ! +n --> an update is done every time MOD(ECALLS,n).EQ.0 .
  !        (ECALLS is declared in ENERGY.FCM and is
  !        incremented by ICALL every time ENERGY(,,,ICAL)
  !        is called ).
  ! -1 --> an update is done when necessary (default), but only
  !        when ICALL is non zero.
  !
  !> param[in] new_inbfrq new update frequency
  !> @return old_inbfrq old update frequency
  function nbonds_set_inbfrq(new_inbfrq) bind(c) result(old_inbfrq)
    use, intrinsic :: iso_c_binding, only: c_int
    use contrl, only: inbfrq
    implicit none
    integer(c_int) :: old_inbfrq, new_inbfrq
     old_inbfrq = inbfrq
    inbfrq = new_inbfrq
   end function nbonds_set_inbfrq

  !> @brief set update frequency for the image list
  !
  ! set the update frequency for the image list. Used in the
  ! subroutine ENERGY() to decide updates. When set to :
  !  0 --> no updates will be done.
  ! +n --> an update is done every time MOD(ECALLS,n).EQ.0 .
  !        (ECALLS is declared in ENERGY.FCM and is
  !        incremented by ICALL every time ENERGY(,,,ICAL)
  !        is called ).
  ! -1 --> an update is done when necessary (default), but only
  !        when ICALL is non zero.
  !
  !> param[in] new_imgfrq new update frequency
  !> @return old_imgfrq old update frequency
  function nbonds_set_imgfrq(new_imgfrq) bind(c) result(old_imgfrq)
    use, intrinsic :: iso_c_binding, only: c_int
    use image, only: imgfrq
    implicit none
    integer(c_int) :: old_imgfrq, new_imgfrq
     old_imgfrq = imgfrq
    imgfrq = new_imgfrq
   end function nbonds_set_imgfrq

  !> @brief set image update cutoff distance
  !
  !> param[in] new_cutim new starting distance for switching function
  !> @return old_cutim old cutim value
  function nbonds_set_cutim(new_cutim) bind(c) result(old_cutim)
    use, intrinsic :: iso_c_binding, only: c_double
    use image, only: cutim

    implicit none

    real(c_double) :: old_cutim, new_cutim
    logical :: status = .true.

    old_cutim = cutim
    cutim = new_cutim
   end function nbonds_set_cutim

  !> @brief check the order of cutoffs cutnb, ctofnb and ctonnb
  !
  !> @return nbonds_cutoffs_orderly
  !>         false if not cutnb <= ctofnb <= ctonnb, true otherwise
  logical function nbonds_cutoffs_orderly()
    use inbnd, only: cutnb, ctofnb, ctonnb
    use stream, only: prnlev, outu

    implicit none

    nbonds_cutoffs_orderly = .true.

    if (cutnb .le. ctofnb .or. ctofnb .le. ctonnb) then
       if(prnlev .ge. 2) then
          write(outu,'(A,3F10.1)') &
               ' api_nbonds%nbonds_cutoffs_orderly> CUTNB,CTOFNB,CTONNB=', &
               cutnb, ctofnb, ctonnb
       end if ! prnlev >= 2

       nbonds_cutoffs_orderly = .false.

       call wrndie(1, '<api_nbonds%nbonds_cutoffs_orderly>', &
            'CUTNB,CTOFNB,CTONNB are not in correct order.')
    end if ! cutnb .le. ctofnb .or. ctofnb .le. ctofnb
  end function nbonds_cutoffs_orderly

  !> @brief check that the distance between cutnb and ctofnb is less than 1.0
  !
  !> param[out] nbonds_cutoffs_orderly
  !>            false if not cutnb - ctofnb < 1.0, true otherwise
  logical function nbonds_cutoffs_spaced()
    use inbnd, only: cutnb, ctofnb
    use contrl, only: inbfrq

    implicit none

    nbonds_cutoffs_spaced = .true.
    if (cutnb - ctofnb .lt. 1.0 .and. inbfrq .lt. 0) then
       nbonds_cutoffs_spaced = .false.
       call wrndie(1, '<api_nbonds%nbonds_set_cutnb>', &
            'CUTNB and CTOFNB are too close for efficient heuristic update.')
    end if ! cutnb - ctofnb < 1.0 .and. inbfrq .lt. 0
  end function nbonds_cutoffs_spaced

  !> @brief set distance cutoff for generating the list of pairs
  !
  !> param[in] new_cutnb new distance cutoff value
  !> @return old_cutnb old cutnb value
  function nbonds_set_cutnb(new_cutnb) bind(c) result(old_cutnb)
    use, intrinsic :: iso_c_binding, only: c_double
    use bases_fcm, only: bnbnd
    use inbnd, only: cutnb
    use psf, only: natom
    use image, only: cutim, natim
    use pbeq, only: qgsbp, srdist
    use exelecm, only: qextnd
    use stream, only: prnlev, outu

    implicit none

    real(c_double) :: old_cutnb, new_cutnb, old_cutim
    logical :: status = .true.

    old_cutnb = bnbnd%nbdist(1)
    if (new_cutnb /= old_cutnb) then
       bnbnd%nbdist(1) = new_cutnb
       cutnb = new_cutnb
       if (natim > natom .and. cutim < cutnb) then
          old_cutim = cutim
          cutim = new_cutnb
          if (prnlev >= 2 .and. cutim /= old_cutim) then
             write(outu, '(A,/,2(A,F5.1))') &
                  ' ***** Info from api_nbonds%nbonds_set_cutnb *****', &
                  ' CUTIM is reset to ', cutim, &
                  ' from ', old_cutim
          end if ! prnlev >= 2
       end if ! natim > natom
    end if ! new_cutnb /= old_cutnb

#if KEY_PBEQ == 1
    if (qgsbp .and. (.not. qextnd) .and. (cutnb < 2 * srdist)) then
       if (prnlev >= 2) then
          write(outu, '(6X,2(A,F6.2))')  &
               'CUTNB=', new_cutnb, &
               ' 2*SRDIST=', 2 * srdist
       end if ! prnlev >= 2
       call wrndie(-5, '<api_nbonds%nbonds_set_cutnb>', &
            'CUTNB smaller than GSBP radius')
    end if ! qgsbp
#endif /* KEY_PBEQ */

    status = nbonds_cutoffs_spaced()
    status = nbonds_cutoffs_orderly()
  end function nbonds_set_cutnb

  !> @brief set distance from which switching function is used
  !
  !> param[in] new_ctonnb new starting distance for switching function
  !> @return old_ctonnb old ctonnb value
  function nbonds_set_ctonnb(new_ctonnb) bind(c) result(old_ctonnb)
    use, intrinsic :: iso_c_binding, only: c_double
    use bases_fcm, only: bnbnd
    use inbnd, only: ctonnb

    implicit none

    real(c_double) :: old_ctonnb, new_ctonnb
    logical :: status = .true.

    old_ctonnb = bnbnd%nbdist(2)
    bnbnd%nbdist(2) = new_ctonnb
    ctonnb = new_ctonnb
    status = nbonds_cutoffs_orderly()
   end function nbonds_set_ctonnb

  !> @brief set distance at which switching function stops being used
  !
  !> param[in] new_ctofnb new stopping distance for switching function
  !> @return old_ctofnb old ctofnb value
  function nbonds_set_ctofnb(new_ctofnb) bind(c) result(old_ctofnb)
    use, intrinsic :: iso_c_binding, only: c_double
    use bases_fcm, only: bnbnd
    use inbnd, only: ctofnb

    implicit none

    real(c_double) :: old_ctofnb, new_ctofnb
    logical :: status = .true.

    old_ctofnb = bnbnd%nbdist(3)
    bnbnd%nbdist(3) = new_ctofnb
    ctofnb = new_ctofnb

    status = nbonds_cutoffs_spaced()
    status = nbonds_cutoffs_orderly()
  end function nbonds_set_ctofnb

  !> @brief set dielectric constant for extened electrostatics routines
  !
  !> param[in] new_eps new dielectric constant
  !> param[out] nbonds_set_eps old dielectric constant
  function nbonds_set_eps(new_eps) bind(c) result(old_eps)
    use, intrinsic :: iso_c_binding, only: c_double
    use bases_fcm, only: bnbnd
    use inbnd, only: eps
    implicit none
    real(c_double) :: old_eps, new_eps

    old_eps = bnbnd%nbdist(7)
    bnbnd%nbdist(7) = new_eps
    eps = new_eps
  end function nbonds_set_eps

  !> @brief use constant dielectric for radial energy functional form
  !
  ! Energy is proportional to 1/R.
  !
  !> @return nbonds_use_cdie 1 if cdie is on, 0 otherwise
  integer(c_int) function nbonds_use_cdie() bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use bases_fcm, only: bnbnd
    use inbnd, only: lcons

    implicit none

    nbonds_use_cdie = 0
    if (lcons) nbonds_use_cdie = 1

    bnbnd%lnbopt(4) = .true.
    lcons = .true.
  end function nbonds_use_cdie

  !> @brief compute interactions on an atom-atom pair basis
  !
  !> param[out] nbonds_use_atom 1 if atom is on, 0 otherwise
  integer(c_int) function nbonds_use_atom() bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use bases_fcm, only: bnbnd
    use inbnd, only: lgroup, lvatom

    implicit none

    nbonds_use_atom = 0
    if (lvatom) nbonds_use_atom = 1

    bnbnd%lnbopt(6) = .true.
    lvatom = .true.

    bnbnd%lnbopt(3) = .false.
    lgroup = .false.
  end function nbonds_use_atom

  !> @brief compute the van der waal energy term on an atom-atom pair basis
  !
  !> param[out] nbonds_use_vatom 1 if vatom is on, 0 otherwise
  integer(c_int) function nbonds_use_vatom() bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    implicit none
    nbonds_use_vatom = nbonds_use_atom()
  end function nbonds_use_vatom

  !> @brief use switching function on forces only from CTONNB to CTOFNB
  !
  !> param[out] nbonds_use_fswitch 1 if fswitch is on, 0 otherwise
  integer(c_int) function nbonds_use_fswitch() bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use bases_fcm, only: bnbnd
    use inbnd, only: &
         legrom, lshft, &
#if KEY_MMFF == 1
         lmshft, ltrunc, &
#endif
         lfswt

    implicit none

    nbonds_use_fswitch = 0
    if (lfswt) nbonds_use_fswitch = 1

    bnbnd%lnbopt(15) = .false.
    bnbnd%lnbopt(13) = .false.

#if KEY_MMFF == 1
    lmshft = .false.
    ltrunc = .false.
#endif

    bnbnd%lnbopt(5) = .false.
    lshft = .false.

    bnbnd%lnbopt(24) = .false.
    legrom = .false.

    bnbnd%lnbopt(9) = .true.
    lfswt = .true.
  end function nbonds_use_fswitch

  !> @brief use switching function on VDW force from CTONNB to CTOFNB
  !
  !> param[out] nbonds_use_cdie 1 if cdie is on, 0 otherwise
  integer(c_int) function nbonds_use_vfswitch() bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use bases_fcm, only: bnbnd
    use inbnd, only: &
#if KEY_MMFF == 1
         lvfswt, lvgrom, lvtrunc
#else
    lvfswt
#endif

    implicit none

    nbonds_use_vfswitch = 0
    if (lvfswt) nbonds_use_vfswitch = 1

    bnbnd%lnbopt(14) = .false.
    bnbnd%lnbopt(25) = .false.
    
#if KEY_MMFF == 1
    lvtrunc = .false.
    lvgrom = .false.
#endif

    bnbnd%lnbopt(10) = .true.
    lvfswt = .true.
  end function nbonds_use_vfswitch

  ! Addition Kai Toepfer May 2022
  !> @brief get distance from which switching function is used
  !
  !> param[out] old_ctonnb current starting distance for switching function
  !> @return old_ctonnb old ctonnb value
  function nbonds_get_ctonnb(old_ctonnb) bind(c) result(success)
    use, intrinsic :: iso_c_binding, only: c_double, c_int
    use bases_fcm, only: bnbnd
    use inbnd, only: ctonnb

    implicit none

    real(c_double) :: old_ctonnb
    integer(c_int) :: success
    
    success = 0
    old_ctonnb = ctonnb
    !old_ctonnb = bnbnd%nbdist(2)
    success = 1
  end function nbonds_get_ctonnb

  !> @brief get distance at which switching function stops being used
  !
  !> param[out] old_ctofnb current stopping distance for switching function
  !> @return old_ctofnb old ctofnb value
  function nbonds_get_ctofnb(old_ctofnb) bind(c) result(success)
    use, intrinsic :: iso_c_binding, only: c_double, c_int
    use bases_fcm, only: bnbnd
    use inbnd, only: ctofnb

    implicit none

    real(c_double) :: old_ctofnb
    integer(c_int) :: success
    
    success = 0
    old_ctofnb = ctofnb
    !old_ctofnb = bnbnd%nbdist(3)
    success = 1
  end function nbonds_get_ctofnb

  !> @brief Update non bonded exclusion list
  !
  !> @return success
  subroutine nbonds_update_bnbnd() bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use bases_fcm, only: bnbnd
    use nbexcl, only: upinb
    
    implicit none

    call upinb(bnbnd)
    
  end subroutine nbonds_update_bnbnd

#endif /* KEY_LIBRARY */
end module api_nbonds
