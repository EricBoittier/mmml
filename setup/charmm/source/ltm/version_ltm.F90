module version
  implicit none

  !
  !     VERNUM  - version number (simple integer); 0 < VERNUM < 10000 is assumed
  !               for endian check. /LNI February 2013
  !     VERNMC  - version number (character string)
  !
  INTEGER, PARAMETER :: VERNUM=52
  CHARACTER(len=24), PARAMETER :: VERNMC='52a1     August 15, 2026'
  !                                       123456789+123456789+1234
  !
end module version
