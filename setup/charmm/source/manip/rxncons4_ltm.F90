module rxncons4
  use chm_kinds
  use dimens_fcm

!
#if KEY_RXNCONS==1
! for path rmsd constraint
  real(chm_real),allocatable,dimension(:) :: DLTAN,FDOTTAN,FOFFP,dlcrv
      LOGICAL QPTCSCL,QFXREP,QPFORCE,QPTAN
      INTEGER IFXREP,IUTAN
      real(chm_real) PDLENG
#endif 
!
end module rxncons4

