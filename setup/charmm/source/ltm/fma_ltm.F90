module fmam
  use chm_kinds
  use dimens_fcm

! FMA.FCM : FMA setup stuff
! =========
!
#if KEY_FMA==1 /*fma_fcm*/
! LEVEL ... number of levels in FMA 
! TERMS ... number of terms in FMA
      INTEGER LEVEL
      INTEGER TERMS
!
contains

  subroutine fma_init()
    use mfmacons
    implicit none
    Level = DFLevel
    Terms = DFTerm
    return
  end subroutine fma_init


#endif /* (fma_fcm)*/
!
end module fmam

