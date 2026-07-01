module resdist_ltm
  use chm_kinds
  use dimens_fcm
  implicit none
  !
  !  data structure (COMMON block) for restraint distances
  !
  ! actual number of distance restraints:
  INTEGER REDNUM

#if KEY_NOMISC==0 /*resdist_fcm*/
  ! actual number of selected atoms
  INTEGER REDNM2
  ! scale factor for energies and forces
  real(chm_real) REDSCA
  ! Atom lists pointer for atom indicies
  INTEGER REDIPT(redmax+1)
  ! Atom lists for indicies
  INTEGER REDILIS(2,redmx2)
  ! Atom lists distance factors
  real(chm_real)  REDKLIS(redmx2)
  ! list for distances and force constants
  real(chm_real) REDRVAL(redmax), REDKVAL(redmax)
  ! list for outside exponents
  INTEGER REDEVAL(redmax)
  ! list for inside individual exponents
  INTEGER REDIVAL(redmax)
  ! list for modes values
  !    (0=complete function: 1=positive only: -1=negative only)
  INTEGER REDMVAL(redmax)
#endif /* (resdist_fcm)  NOMISC*/

contains
  subroutine resdist_iniall
    rednum=0
    return
  end subroutine resdist_iniall
end module resdist_ltm

