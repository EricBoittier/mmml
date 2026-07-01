include 'OpenMMFortranModule.f90'

program ommv
  use OpenMM, only: OpenMM_Platform_getOpenMMVersion
  implicit none
  character(len=16) version

  call OpenMM_Platform_getOpenMMVersion(version)
  print '(A)',trim(version)
end program ommv
