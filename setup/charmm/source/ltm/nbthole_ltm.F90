module nbthole
  use chm_kinds
!     NBTHOL     number of pair-specific Thole entries in the parameter file
!     NBTHOLIJ   list of pair-specific Thole read from the parameter
!     NBTHOLXIJ  numerical values of the pairwise Thole factor from the parameter file
!     THOLCUT    cutoff to make up the list of nonbonded Thole shielding
!     NBTHOLP    number of atom pairs in the mini nonbonded list of pair-specific Thole
!     NBTHOLPIMG number of image atom pairs in the mini nonbonded list of pair-specific Thole
!     MAXNBTHOLE default dimension for the mini nonbonded list (can overide it)
!     NBTHOL1    first atom in the mini nonbonded list of pair-specific Thole
!     NBTHOL2    second atom in the mini nonbonded list of pair-specific Thole
!     NBTHOL3    pointer for the pairwise Thole factor in NBTHOLXIJ

!     PPNBTHOLP  number of atom pairs in the mini nonbonded list of pair-specific
!                Thole for state lambda=0 when pert is on
!     PPNBTHOL1  first atom in the mini nonbonded list of pair-specific
!                Thole for state lambda=0 when pert is on
!     PPNBTHOL2  second atom in the mini nonbonded list of pair-specific
!                Thole for state lambda=0 when pert is on
!     PPNBTHOL3  pointer for the pairwise Thole factor in NBTHOLXIJ for state
!                lambda=0 when pert is on

      integer MAXTHOLE
      parameter (MAXTHOLE=100)

      integer MAXNBTHOLE
      integer NBTHOL, NBTHOLIJ(2,MAXTHOLE)
      integer NBTHOLP, NBTHOLPIMG, PPNBTHOLP
      integer,allocatable,dimension(:) :: NBTHOL1, NBTHOL2, NBTHOL3
      integer,allocatable,dimension(:) :: PPNBTHOL1, PPNBTHOL2, PPNBTHOL3

      real(chm_real)  THOLCUT, NBTHOLXIJ(MAXTHOLE)
end module nbthole

