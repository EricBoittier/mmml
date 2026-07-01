module xdraw
  use chm_kinds
!CHARMM Element source/fcm/xdraw.fcm 1.1
#if KEY_NOGRAPHICS==0
#if KEY_XDISPLAY==1
!
! MACHINE SPECIFIC GRAPHICS VARIABLES
!     XPLA   Number of bit planes
!     XXSZ   Size of X window in X direction 
!     XYSZ   Size of X window in Y direction 
!
  INTEGER XPLA,XXSZ,XYSZ
!
#endif 
#endif 
!
end module xdraw

