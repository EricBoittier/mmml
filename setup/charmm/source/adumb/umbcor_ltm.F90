module umbcor
  use chm_kinds
#if KEY_ADUMB==1 /*adumb*/
  !     Correlated variables for
  !     Adaptive Umbrella Potentials
  !
  !    RJ Petrella 4.1.01 
  !
  !     Purpose:
  !     To correlate structural variables in an adaptive 
  ! umbrella sampling run to the umbrella coordinates
  !
  !    correlated distances:
  !     NDISTA   number of distances to be correlated w/ umbrella coords
  !     MXDICR   maximum number of correlation distances
  !     DISIAT   
  !     DISJAT   pairs of atoms whose dist's to be correlated w/ umbrella
  !     LCORRV   flag is true if correlations being done
  !     CDISUN   unit numbers for writing out correlated variable files
  !     LPRCRU   flag is true if writing out to file
  !     LCNEED   true when memory not yet allocated for correlations
  !     PCORMD   1/(fraction of the updates which result in printing correlations)
  !     
  ! for rmsd's
  !   
  !      NRMSDI   number of (sub)structures in rmsd calculations
  !      RMSTRS   number of rmsd substructures + x for memory allocation
  !      RMSMMS   number of atoms in rmsd structures + y for mem allocation
  !      RMACNT   counter of atoms in rmsd structures
  !      CRMSU1   unit numbers for writing results for substructures 1
  !      CRMSU2   unit numbers for writing results for substructures 2
  !      ORATCT   number of atoms in rmsd orientations
  !      NSYMAT   number of atoms in rmsd symmetries
  !      
  !      MXRMSD   maximum number of rmsd structures
  !      LPRRMU   flag is true if writing out to file
  !      LCCOR1   true when reference coor set 1 is read
  !      LCCOR2   true when reference coor set 2 is read
  !      MXRMSA   maximum number of atoms within an rmsd substructure
  !      WCRUNI   unit number to which the accumulators are written for restart
  !      RCRUNI   unit number from which accumulators are read for restart
  !
  !   pointers
  !     HPCRRD   pointers to CRRDSM and TOAVDI, arrays
  !     HPTOAV   which carry the sums of the correlated variable
  !              values for each gridpoint and each variable
  !              for either the last trajectory (CRRDSM) or
  !              all trajectories (TOAVDI)
  !      HPCRMS1   for current rms sum1 (RMSSUM1(INSIUM,NRMSDI))
  !      HPCRMS2   for current rms sum2 (RMSSUM1(INSIUM,NRMSDI))
  !      HPTORM1   for cumulative rms sum TORMSD1(INSIUM,NRMSDI))
  !      HPTORM2   for cumulative rms sum TORMSD2(INSIUM,NRMSDI))
  !      HPRMLS   for structure list RMSLST (list of atoms) 
  !      HPRMEN   for pointer to structure list (RMAEND)
  !      HPCO1X,HPCO1Y,HPCO1Z,HPCO2X,HPCO2Y,HPCO2Z
  !               for reference coordinates
  !      HPCOR1   for memory allocation 1  
  !      HPCPYX, HPCPYY, HPCPYZ, HPCP2X, HPCP2Y, HPCP2Z
  !              for temporary storage of coordinates
  !      HPATMN   for storage of selected atom pairs for rmsd calc
  !      HPCOR1,HPCOR2 for rmsd storage arrays
  !      HPNDIS    for distance array
  !CC      HPRMS1,HPRMS2,HPRMS3,HPRMS4   for symmetry dihedrals
  !      HPSYDL  for ic line numbers of symmetry dihedrals 
  !      HPLORI  for whether orientation done (T/F array)
  !      HPSYMN  for symmetry numbers of dihedral angles
  !      HPXCPS,HPYCPS,HPZCPS for temp storage of coors
  !      HPTEMX,HPTEMY,HPTEMZ temporary storage of coors
  !      HPTE2X,HPTE2Y,HPTE2Z temporary storage of coors
  !      HPMNIC,HPMXIC for 1st/last apprnces of substructure in ic table
  !      HPOREN,HPORLS for orientation atom base array and list
  !      HPSYLS,HPSYEN for symmetry atom list and base array
  !
  !  INTEGERS

  integer,PARAMETER :: MXDICR=100, MXRMSD=100

  real(chm_real),allocatable,dimension(:,:) :: HPCRRD,HPTOAV,HPCRMS1,HPTORM1,HPCRMS2,HPTORM2
  integer,allocatable,dimension(:,:) :: HPATMN
  real(chm_real),allocatable,dimension(:) :: HPCPYX,HPCPYY,HPCPYZ,HPCP2X,HPCP2Y,HPCP2Z
  real(chm_real),allocatable,dimension(:) :: HPNDIS,HPCOR1,HPCOR2,HPXCPS,HPYCPS,HPZCPS
  real(chm_real),allocatable,dimension(:) :: HPTEMX,HPTEMY,HPTEMZ,HPTE2X,HPTE2Y,HPTE2Z
  real(chm_real),allocatable,dimension(:) :: HPCO1X,HPCO1Y,HPCO1Z,HPCO2X,HPCO2Y,HPCO2Z

  integer,allocatable,dimension(:) :: HPRMLS,HPORLS,HPSYLS,HPRMEN,HPOREN,HPSYEN
  integer,allocatable,dimension(:) :: HPSYDL,HPSYMN,HPMXIC,HPMNIC  
  integer DISIAT(MXDICR),DISJAT(MXDICR),CDISUN(MXDICR),CRMSU1(MXRMSD),CRMSU2(MXRMSD)
  LOGICAL,allocatable,dimension(:) :: HPLORI
  INTEGER   NRMSDI,RMSTRS,RMSMMS,RMACNT
  INTEGER   ORATCT,NSYMAT,PCORMD
  INTEGER   WCRUNI,RCRUNI,NDISTA
  INTEGER   HPRMSS,MXRMSA


  ! LOGICALS
  LOGICAL LCORRV,LPRCRU,LCNEED,LPRRMU,LCCOR1,LCCOR2

contains

  subroutine adumbcor_iniall()
    implicit none
  ndista=0
  lcorrv=.false.
  lprcru=.false.
  lprrmu=.false.
  rmsmms=-1
  rmstrs=-1
  rmacnt=0
  oratct=0
  nrmsdi=0
  lccor1=.false.
  lccor2=.false.
  mxrmsa=0
  pcormd=1
    return
  end subroutine adumbcor_iniall

#endif /* (adumb)*/
  !
end module umbcor

