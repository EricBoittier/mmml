module mfmacons
  use chm_kinds
  use dimens_fcm
!CHARMM Element source/fcm/fmacons.fcm 1.1
! FMACONS.FCM
! ===========
!
#if KEY_FMA==1 /*fmacons_fcm*/
! The Maximum allowable values of level and terms
      integer MXTERM,MXLEVEL
      parameter (MXTERM=20,MXLEVEL=10)
!
! The default values of Level and Terms
      integer DFTERM,DFLEVEL
      parameter (DFTERM=6,DFLEVEL=3)
!
! These values control the average interaction list and neighbour list sizes
      integer AvgNbr, AvgIntrct
      parameter (AvgIntrct=600, AvgNbr=500)
!
      integer START(MXLEVEL),FINISH(MXLEVEL)
!
#endif /* (fmacons_fcm)*/
!
end module mfmacons

