module rxncons3
  use chm_kinds
  use dimens_fcm

!
#if KEY_RXNCONS==1
! for plane constraint
  real(chm_real),allocatable,dimension(:) :: XREFM,YREFM,ZREFM
  real(chm_real),allocatable,dimension(:) :: XREFA,YREFA,ZREFA
  real(chm_real),allocatable,dimension(:) :: XREFB,YREFB,ZREFB
  real(chm_real),allocatable,dimension(:) :: XTAN,YTAN,ZTAN
  real(chm_real),allocatable,dimension(:) :: XFORC,YFORC,ZFORC
  real(chm_real),allocatable,dimension(:) :: XCURV,YCURV,ZCURV
#endif 
!
end module rxncons3

