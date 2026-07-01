module icpert
  use chm_kinds
  use chm_types
  use dimens_fcm

#if KEY_TSM==1
  !     Information for internal coordinate perturbations
  !     Modified for CFTI/CFTM, KK 14-Mar-97
  !
  !     Purpose: to store the information needed to do ic perturbations.
  !
  !     Variable    Purpose
  !
  !     MXICP      Maximum number of ic perturbations
  !     NICP        Number of ic perturbations in use
  !     IICP        Flag; 0 => NICP.eq.0, 1 => NICP.gt.0
  !     ICPTYP      Type of ic: 1 => distance, 2 => angle, 3 => dihedral
  !     DZETA       Range of the perturbation
  !     ICPINC      Number of increments (windows) in which the perturbation
  !                 is done
  !     ICPATN      Matrix of atom numbers of atoms in perturbations
  !     NATICP      Total number of atoms involved in perturbations
  !     ICPMV1      Heap pointers for MOVE atom selections
  !     ICPMV2      Heap pointers for MOVE atom selections
  !     NMOV1       Number of atoms in first MOVE selection
  !     NMOV2       Number of atoms in second MOVE selection
  !     LMOV2       Logical indicating double MOVE atom selections
  !     ISLICP      Heap pointer for interaction energy calculation (solute)
  !     JSLICP      Heap pointer for interaction energy calculation (bath)
  !     IUNICP      Unit to which ic perturbation data is written
  !     ISVICP      Print frequency for perturbation output
  !
  !     Variables for CFTI/CFTM conformational free energy
  !     QCFTI       Flag for TI conformational free energy, KK 03/11/93
  !     QCFTM       Flag for TI multidimensional conformational free
  !                 energy, KK 03/24/95
  !     QICWR       Flag for IC write for CFTM calculations
  !     For group analysis in routine BLCFTM
  !     NGRUP  number of coordinate groups 
  !     LGRUP  pointer from coordinate to group number
  !     NGRU   number of coordinates in group
  !     GSYM   character string used to designate group
  !     GAFLU  auxilliary array for errors of group contributions
  !     DIRV   direction of the comformational change. 
  !     IHPCFTI(12) - array of heap pointers for scratch arrays
  !     QPRTSS - flag, .TRUE. means storage arrays are set
  !     LNPRIC -false if printing ic coordinates (default) !RJP
  ! ***** running averages in ic pert (RJP): ******
  !     KTEMPE kT-- for calculating running averages 
  !     RUNITN unit number for writing out running ave
  !     RPRCYC interval (number of steps) for writing out run ave
  !     PEVERY print out every PEVERY perturbation calcs
  !     LRUNNA flag is true when running averages requested
  !     MINFOR minima for forward perts
  !     MINBAC minima for backward perts
  !     SUMFOR sum of exponentials for forward perts
  !     SUMBAC sum of exponentials for backward perts
  !     EAVFOR average delta-E for forward perts
  !     EAVBAC average delta-E for backward perts
  !     ICAVER average (unperturbed) internal coordinates
  !     TSMSTP tsm step counter
  ! ***************************************************
  !
  INTEGER MXICP
  PARAMETER (MXICP=200)
  INTEGER NICP,IICP,ICPTYP(MXICP),ICPINC,ICPATN(4,MXICP), &
       NMOV1(MXICP),NMOV2(MXICP),NATICP,IUNICP,ISVICP

  type(chm_iptr),dimension(MXICP),save :: ICPMV1, ICPMV2
  type(chm_ptr),dimension(12),save :: IHPCFTI

  integer,allocatable,dimension(:) :: islicp, jslicp

  INTEGER RPRCYC,RUNITN,PEVERY,TSMSTP
  real(chm_real) DZETA(MXICP),KTEMPE,MINBAC(MXICP),MINFOR(MXICP), &
       SUMFOR(MXICP),SUMBAC(MXICP),EAVFOR(MXICP),EAVBAC(MXICP), &
       ICAVER(MXICP)
  LOGICAL LMOV2(MXICP),QCFTI,QCFTM,QICWR,QPRTSS,LNPRIC,LRUNNA
  !
  real(chm_real) GAFLU(MXICP),DIRV(MXICP)
  INTEGER NGRUP,LGRUP(MXICP),NGRU(MXICP),YLAMBD
  CHARACTER(len=4) GSYM(MXICP)

contains

  subroutine icpert_init
    implicit none
    lrunna=.false.
    qprtss=.false.
    return
  end subroutine icpert_init

#endif 
end module icpert

