module omm_glblopts
   use chm_kinds
   implicit none

#if KEY_OPENMM==1
   character(len=20), save :: omm_platform = ''
   character(len=20), save :: omm_deviceid = ''
   character(len=20), save :: omm_precision = ''

   character(len=20), save :: omm_default_platform = ''
   character(len=20), save :: omm_default_deviceid = ''
   character(len=20), save :: omm_default_precision = ''
#endif
   real(chm_real), save :: torsion_lambda
   logical, save :: &
        qnocpu = .false., &  ! limit platforms to CUDA and OpenCL
        qtor_repex  ! Torsion angle repd is turned on/off

contains

  subroutine init_glblopts()
    implicit none
#if KEY_OPENMM==1
    omm_platform = omm_default_platform
    omm_deviceid = omm_default_deviceid
    omm_precision = omm_default_precision
#endif
  end subroutine init_glblopts

end module omm_glblopts
