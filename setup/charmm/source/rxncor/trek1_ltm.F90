module trek1
  use chm_kinds
  implicit none
  !CHARMM Element source/fcm/trek1.fcm $Revision: 1.4 $

  ! Common-block for TReK version 2.10
  ! By Stefan Fischer; July 5-2003.
  !
  ! Include in  .source/rxncor/adiab.src
  ! Include in  .source/rxncor/travel.src


  real(chm_real)           STEPSZ,STEPLW,DELTAS,SADGRD
  real(chm_real)           PRTOL1,PRTOL2,PROJIN,OSCTOL
  INTEGER          NTANGT,LINMIN,MODREM,SADMIN,NCYCLE,NECALL
  INTEGER          REDLUP,FRAME,DISPLP,INTERP,OSCMAX
  LOGICAL          LORIEN,LHISAD
end module trek1

