module rism
  use chm_kinds
  use dimens_fcm
!CHARMM Element source/fcm/rism.fcm 1.1
#if KEY_RISM==1 /*rism_fcm*/
!---------------------------------------------------------------------
!                Compilation parameters
!---------------------------------------------------------------------
!  PARAMETERS STORAGE
      INTEGER DBASE,DPBASE
      PARAMETER (DBASE=150,DPBASE=(DBASE*(DBASE-1))/2)
!
!  DISTRIBUTION FUNCTION DIMENSIONS:
!  maximum number of points in the r- and k- space for the 
!  discretized distribution functions
      INTEGER DVECT
      PARAMETER (DVECT=512)
!  maximum number of sites in a rism solvent or solute molecule
      INTEGER  DSITV,DSITU
      PARAMETER (DSITV=6,DSITU=20)
!  maximum number of solute molecules
      INTEGER DU,DUU
      PARAMETER (DU=2,DUU=(DU*(DU+1))/2)
!  maximum number of sites
      INTEGER DSITE
      PARAMETER (DSITE=DSITV+DSITU*DU)
!  maximum number of pair of sites of solvent
      INTEGER DPRVV
      PARAMETER (DPRVV=(DSITV*(DSITV+1))/2)
!  maximum number of pair of sites of solute
      INTEGER DPRUU
      PARAMETER (DPRUU=(DSITU*(DSITU+1))/2)
!  maximum number of pair of sites of solute-solvent
      INTEGER DPRUV
      PARAMETER (DPRUV=DSITU*DSITV)
! total maximum number of pair of sites
      INTEGER DPAIR
      PARAMETER (DPAIR=DPRVV+DU*DPRUV+DUU*DPRUU)
!
!     PHYSICAL CONSTANTS:
!     COEFF     Coulomb potential conversion factor E=coeff*q1*q2/r12
      real(chm_real) COEFF
      PARAMETER (COEFF=332.18348D0)
!
#endif /* (rism_fcm)*/
!
end module rism

