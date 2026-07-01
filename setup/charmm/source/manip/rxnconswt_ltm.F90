module rxnconswt
  use chm_kinds
  use dimens_fcm

!
#if KEY_RXNCONS==1
! for plane constraint
  real(chm_real),allocatable,dimension(:) :: WMASS
  real(chm_real),allocatable,dimension(:) :: ATMASS
! logicals
  LOGICAL QCNOTRN,QCNOROT,QCMASS,QCWEIG,QCWCOMP,QCWCOMP2
#endif 
  !
end module rxnconswt

