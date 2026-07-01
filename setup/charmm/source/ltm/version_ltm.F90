module version
  use chm_kinds

  !
  !     VERNUM  - version number (simple integer); 0 < VERNUM < 10000 is assumed
  !               for endian check. /LNI February 2013
  !     VERNMC  - version number (character string)
  !
  INTEGER, PARAMETER :: VERNUM=48
  CHARACTER(len=24), PARAMETER :: VERNMC='49b1     August 15, 2024'
  !                                       123456789+123456789+1234
  !
end module version
