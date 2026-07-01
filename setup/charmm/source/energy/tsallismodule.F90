! H Kamberaj November 2007
!
!
! QTTSALL   Flag for Tsallis statistics on torsion angles
!
! QTALPHA   Flag for scaling only torsion potential energy
!
! TSALPHA   U = tsalpha * U
!
! TSQ       =(1-q), where q Tsallis scaling factor
!
! TSEMIN    Ground state potential energy
!
! TSBETA    =1/kBT
!
! TSNDF     nr of degrees of freedom to be weighted according to Tsallis
!
! IASCALE   Flag for atom list which are scaled either according to
!           Tsallis or simple - alpha scaling
!
module tsallis_module

  use chm_kinds
  implicit none
#if KEY_TSALLIS==1 /*tsallis_module*/
  REAL(chm_real), allocatable, save :: TDX(:),TDY(:),TDZ(:)
  REAL(chm_real), save :: tsfact, tsbeta, tsq, tsemin
  real(chm_real), save :: ebias, etsbias, tsalpha
  integer, save :: tsndf
  integer, allocatable, save :: iascale(:)
  real(chm_real), dimension(2), save :: ebtors
  logical, save :: qtsall,qttsall, qtalpha
  ! 
#endif /* (tsallis_module)*/
end module tsallis_module

