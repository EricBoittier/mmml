! Placeholder for Fortran wrapper interface provided by OpenMM.
! https://simtk.org/home/openmm

#if KEY_OPENMM==1

! pgf95 wants 'end subroutine', not just 'end' as in OpenMM 4.1 wrapper

INCLUDE 'OpenMMFortranModule.f90'
INCLUDE 'CharmmOpenMMFortranModule.f90'
INCLUDE 'OpenMMGBSWFortranModule.f90'
INCLUDE 'OpenMMGBMVFortranModule.f90'
INCLUDE 'OpenMMMSESFortranModule.f90' ! AN EXAMPLE OF OPENMM PLUGIN

#else /* KEY_OPENMM */

! humor setmk.com

MODULE OpenMM_Types
END MODULE OpenMM_Types

MODULE OpenMM
END MODULE OpenMM

MODULE OpenMM_Charmm
END MODULE OpenMM_Charmm

MODULE OpenMMGBSW_Types
END MODULE OpenMMGBSW_Types

MODULE OpenMMGBSW
END MODULE OpenMMGBSW

MODULE OpenMMGBMV_Types
END MODULE OpenMMGBMV_Types

MODULE OpenMMGBMV
END MODULE OpenMMGBMV

MODULE OpenMMMSES_Types  ! AN EXAMPLE OF OPENMM PLUGIN
END MODULE OpenMMMSES_Types ! AN EXAMPLE OF OPENMM PLUGIN

MODULE OpenMMMSES  ! AN EXAMPLE OF OPENMM PLUGIN
END MODULE OpenMMMSES  ! AN EXAMPLE OF OPENMM PLUGIN
#endif
