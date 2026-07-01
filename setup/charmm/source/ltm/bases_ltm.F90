!CHARMM Element source/fcm/bases.fcm $Revision: 1.6 $
module bases_fcm

  use chm_kinds
  use chm_types
  !
  !     This module contains the base arrays of dynamically
  !     allocated data structures which refer to the main coordinate set.
  !     It is up to future programmers to see that they are maintained
  !     with approriate dimensions.
  !
  !  BNBND,  LNBND  - Base arrays for nonbond interactions.
  !  BIMAG,  LIMAG  - Base arrays for images.
  !  BNBNDC, LNBNDC - Base arrays for alternate nonbonds   (for TSM).
  !  BIMAGC, LIMAGC - Base arrays for alternate image data (for TSM).
  !  BINTCR, LINTCR - Base arrays for main  IC table.
  !  BINTCS, LINTCS - Base arrays for saved IC table.
  !  BNBNDP, LNBNDP - Base arrays for perturbed nonbond interactions.
  !  BNBNDR, LNBNDR - Base arrays for perturbation reference atoms.
  !  BIMAGP, LIMAGP - Base arrays for perturbed images.
  !  BIMAGR, LIMAGR - Base arrays for perturbed reference images.
! These below have been deleted
  !  BPERTD, LPERTD - Base arrays for PERT general data
  !  BPPSF0, LPPSF0 - Base arrays for PERT saved PSF (lambda=0)
  !  BPRES0, LPRES0 - Base arrays for PERT saved restraints (lambda=0)
  !
!!  INTEGER,PARAMETER :: BASELN=50

!!  integer,dimension(baseln) :: &
!!        BNBND,   LNBND, &
!!        BIMAG,   LIMAG, &
!!       BNBNDC,  LNBNDC, &
!!       BINTCR,  LINTCR, &
!!       BINTCS,  LINTCS, &
!!       BNBNDP,  LNBNDP, &
!!       BNBNDR,  LNBNDR, &
!!       BIMAGC,  LIMAGC, &
!!       BIMAGP,  LIMAGP, &
!!       BIMAGR,  LIMAGR
!!       BPERTD,  LPERTD, &
!!       BPPSF0,  LPPSF0, &
!!       BPRES0,  LPRES0

  type(nonbondDataStructure),target,save :: BNBND,BNBNDC,BNBNDP,BNBNDR

  type(imageDataStructure),target,save   :: BIMAG,BIMAGC,BIMAGP,BIMAGR



end module bases_fcm

