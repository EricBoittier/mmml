module hbondm
  use chm_kinds
  use dimens_fcm
  !
  !     The Hydrogen Bond List
  !
  !     Purpose: To store the list of hydrogen bonds
  !
  !     I/O:    Unformatted read: HBREAD in [MK.PROT]HBOND2.FOR
  !             Print and unformatted write: HBWRIT in [MK.PROT]HBOND2.FOR
  !
  !     Notes:  All arrays are indexed by hydrogen bond
  !
  !     Variable   Purpose
  !
  !     CHBINI     Keeps track of which variables have been initialized
  !                by GTHBCT. See that routine for more details.
  !     CTOFHA     Off angle for switching function on angles
  !     CTOFHB     Off distance for switching function on hydrogen bond
  !                distance
  !     CTONHA     On angle for switching function on angles
  !     CTONHB     On distance for switching function on hydrogen bond
  !                distance
  !     CUTHB      Distance cutoff for selecting hydrogen bonds
  !     CUTHBA     Angular cutoff for selecting hydrogen bonds
  !     IHB        Heavy atom donor
  !     JHB        Heavy atom acceptor
  !     KHB        Hydrogen in hydrogen bond (can be zero if no hydrogen)
  !     LHB        Heavy atom acceptor anticedent (zero if not used)
  !     ICH        Codes array for hydrogen bonds
  !     NHB        Number of hydrogen bonds
  !     MAXHB      Dimension of the hydrogen bond arrays
  !     LHBFG      Flag indicating whether acceptor anticedents are used
  !     BEST       Flag specifying that only the best hydrogen bond for
  !                any donor hydrogen will be chosen.
  !
  !     IHBFRQ     The hydrogen-bond update frequency.
  !     QHBPRS     A hydrogen-bond update flag.
  !
  ! integers
  INTEGER CHBINI, ICH(MAXHB), IHB(MAXHB), JHB(MAXHB), KHB(MAXHB), LHB(MAXHB), NHB, IHBFRQ
  ! reals
  real(chm_real)  CTONHB, CTOFHB, CTONHA, CTOFHA, CUTHB, CUTHBA
  ! logicals
  LOGICAL LHBFG, BEST, HBEXCL, QHBPRS
  !
contains
  subroutine hbond_init
    ihbfrq=0
    return
  end subroutine hbond_init
end module hbondm

