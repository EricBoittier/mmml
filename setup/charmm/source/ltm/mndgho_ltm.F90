module mndgho
  use chm_kinds
  use dimens_fcm

  implicit none
!CHARMM Element source/fcm/mndgho.fcm 1.1 
#if KEY_MNDO97==1
!
!     Defines the data necessary for a QM calculation on a system.
!
!
!     Variable  Index    Purpose
!
!     QLINK              True - containing QM link atoms
!     BT                 C' = BT.C
!     BTM                C  = BTM.C'
!     NATQM              Number of QM atom
!     NQMLNK             Number of QM link atoms
!     IQLINK             ATOM NUMBER OF QM LINK ATOM
!     JQLINK             MM ATOMS TO WHICH QM LINK ATOM CONNECTS
!     KQLINK             QM ATOM TO WHICH QM LINK ATOM CONNECTS
!     MAXQMLINK          MAX. No. of QM LINK ATOMS ALLOWED
!     MAXLM4MND          MAX. dimension for MNDO density matrix
!
! Only an estimate here, need to be changed to be
! strictly consistant with MNDO97
!     integer, parameter :: MAXLM4MND=80000      ! ? for upto 400 AOs.
!                                
!
!     integer, parameter :: MAXQMLINK=10, &
!                           MXQM16=16*MAXQMLINK
!     LOGICAL QLINK
!     INTEGER NATQM,NQMLNK,IQLINK(MAXQMLINK),JQLINK(3,MAXQMLINK),KQLINK(MAXQMLINK)
!     real(chm_real)  BT(MXQM16),BTM(MXQM16),QMATMQ(MAXQMLINK),DBTMMM(3,3,MXQM16),&
!                     PHO(MAXLM4MND),PBHO(MAXLM4MND),FAOA(MAXLM4MND),FAOB(MAXLM4MND)
!     LOGICAL  UHFGHO
!     INTEGER  NORBG

      LOGICAL,save:: QLINK=.false.

#else
      LOGICAL :: QLINK_dummy
!
#endif

end module mndgho
