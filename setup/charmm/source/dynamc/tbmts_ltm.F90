module tbmts_ltm
  use chm_kinds
  use dimens_fcm
!
! Multiple Time Scale MD 
! by Masa Watanabe
!
!  QTBMTS    - Main flag indicating MTS usage
!  TBHY     - Flag indicating separations of forces based on bond
!             involving hydrogen atoms only.
!  TBHY1    - Flag indicating mass based force separations.(Fastest bin)
!  TBHY2    - Flag indicating mass based force separations (Middle bin)
!  IMTS     - Integer flag to setup distributions of atoms into the
!             correct bins.
!  IMTF(*)  - Flag to identify the atoms which have less than a specified mass.
!
#if KEY_MTS==0 /*mts_fcm*/
  logical, parameter :: QTBMTS = .false.
#else /* (mts_fcm)*/
  logical :: QTBMTS
  logical :: TBHY, TBHY1, TBHY2
  integer, allocatable :: IMTS(:), IMTF(:)
#endif /* (mts_fcm)*/
!
end module tbmts_ltm

