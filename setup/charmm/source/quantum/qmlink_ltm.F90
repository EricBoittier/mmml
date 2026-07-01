module qmlinkm
  use chm_kinds
  use dimens_fcm
#if KEY_QUANTUM==1
  use sizes, only : MPACK  
#endif
!CHARMM Element source/fcm/qmlink.fcm 1.1
#if KEY_QUANTUM==1
!
!     Defines the data necessary for a QM calculation on a system.
!         
!  
!     Variable  Index    Purpose 
! 
!     QMLINK             True - containing QM link atoms
!     BT                 C' = BT.C
!     BTM                C  = BTM.C'
!     MQMLNK             Number of QM link atoms
!     IMQLINK            ATOM NUMBER OF QM LINK ATOM
!     JMQLINK            MM ATOMS TO WHICH QM LINK ATOM CONNECTS
!     KMQLINK            QM ATOM TO WHICH QM LINK ATOM CONNECTS
!     MAXQMLINK          MAX. No. of QM LINK ATOMS ALLOWED
!

      integer, parameter :: MAXQMLINK=5,MXQM16=16*MAXQMLINK
      logical, save  :: QMLINK                                                ! COMMON/MLINKL/
      integer, save  :: MQMLNK,IMQLINK(MAXQMLINK),JMQLINK(3,MAXQMLINK), &     ! COMMON/MLINKI/
                        KMQLINK(MAXQMLINK)
      real(chm_real),save :: MBT(MXQM16),MBTM(MXQM16),QMATMQM(MAXQMLINK), &   ! COMMON/MLINKF/
                             MDBTMMM(3,3,MXQM16),PHO(MPACK), &
                             PBHO(MPACK)           ! For UHF-GHO ... PJ 12/2002

! tag for density damping ... PJ 12/2002
      LOGICAL, save ::  DDAMP            ! COMMON /QDAMP/

! density damping factor ... PJ 12/2002
      real(chm_real), save :: PALPHA      ! COMMON/PDAMP/
!
#endif 
end module qmlinkm

