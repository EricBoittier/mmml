module cadpac_data
  use chm_kinds, only: chm_real

  implicit none
  
  ! NCADPC: Number of Cadpac atoms
  integer, save :: NCADPC = 0

#if KEY_CADPAC==1 /*cadpac1*/

  !     Defines the data necessary for a GAMESS calculation on a system.

  ! NOTE: DO NOT MODIFY, UNLESS THE PARENT CODE IN GAMESS IS ALSO CHANGED!

  !     Variable  Index    Purpose

  !     MAXCEN             Maximum number of CADPAC atoms allowed
  !     IGMSEL             Selection for calls from cadpac

  INTEGER,PARAMETER :: MAXCEN = 100,MAXN3 = 300

  real(chm_real),save :: CADZIN(MAXCEN), CADCRD(3,MAXCEN)
  INTEGER,save :: NATREL
  CHARACTER(len=10),dimension(maxcen),save :: CADATM

  !     KCHRMM         Flag to tell CADPAC if it is in CHARMM environment
  !     D{X,Y,Z}ELMM   Forces from QM electrons to MM atoms
  !     D{X,Y,Z}REP    Forces from QM nuclei to MM atoms

  !     CADZIN         Stores charge of QM atoms to be passed to CADPAC
  !     CADCRD         Three dimensional array for storing QM coordinates to be
  !                    passed to CADPAC
  !     CADATM         Element name of QM atoms that are passed to CADPAC
  !     MAXCEN         Maximum number of atoms that CADPAC works with
  !     MAXLAT         CADPAC variable - defines the maximum no. of lattice points
  !     COMMON/LATTIC/ A common block from CADPAC that stores information about lattice
  !                    (CHARMM) atoms.

  INTEGER,PARAMETER :: MAXLAT=25120
  INTEGER,save :: KCHRMM
  !     This is currently in CADPAC

  real(chm_real),dimension(maxlat),save :: &
       DXELMM,DYELMM,DZELMM,DXREP,DYREP,DZREP

  real(chm_real),dimension(3,maxlat) :: CLAT
  real(chm_real),dimension(maxlat) :: ZLAT,BREPL,AREPL
  real(chm_real),save :: XALAT,YALAT,ZALAT,CHALAT
  INTEGER,save :: NLAT,NLAT2
#endif /* (cadpac1)*/

end module cadpac_data
