module sizes
  use chm_kinds
  !CHARMM Element source/fcm/sizes.fcm 1.1
#if KEY_QUANTUM==1 /*sizes_fcm*/
  !***********************************************************************
  !   this file contains all the array sizes for use in mopac.
  !
  !     there are only  parameters that the programmer need set:
  !     maxhev = maximum number of heavy atoms (heavy: non-hydrogen atoms)
  !     maxlit = maximum number of hydrogen atoms.
  !     maxtim = default time for a job. (seconds)
  !     maxdmp = default time for automatic restart file generation (secs)
  !
  integer, parameter :: MAXHEV=50  , MAXLIT=50, &
       MAXTIM=3600, MAXDMP=3600
  !
  !***********************************************************************
  !
  !   the following code does not need to be altered by the programmer
  !
  !***********************************************************************
  !
  !    all other parameters are derived functions of these two parameters
  !
  !      name                   definition
  !     numatm         maximum number of atoms allowed.
  !     maxorb         maximum number of orbitals allowed.
  !     maxpar         maximum number of parameters for optimisation.
  !     n2elec         maximum number of two electron integrals allowed.
  !     mpack          area of lower half triangle of density matrix.
  !     morb2          square of the maximum number of orbitals allowed.
  !     maxhes         area of hessian matrix
  !***********************************************************************
  real(chm_real),parameter:: VERSON=4.00D0
  integer,parameter :: NUMATM=MAXHEV+MAXLIT, &
       MAXORB=4*MAXHEV+MAXLIT, &
       MAXPAR=3*NUMATM, &
       MAXBIG=MAXORB*MAXORB*2, &
       N2ELEC=2*(50*MAXHEV*(MAXHEV-1)+10*MAXHEV*MAXLIT+(MAXLIT*(MAXLIT-1))/2), &
       MAXHES=(MAXPAR*(MAXPAR+1))/2,MORB2=MAXORB**2, &
       MPACK=(MAXORB*(MAXORB+1))/2
  !***********************************************************************
  integer,parameter :: NPULAY=MPACK  ! old, NPULAY=1
  !
  !   for short version use line with nmeci=1, for long version use line
  !   with nmeci=10
  !   which is moved to quantm_ltm.src for other memory allocations.
  !          for other memory
  !     integer,parameter :: NMECI=10      ! old, NMECI=1
  !***********************************************************************
#endif /* (sizes_fcm)*/
  !
end module sizes

