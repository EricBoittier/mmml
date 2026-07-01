module pshake
  use chm_kinds
  use dimens_fcm
!CHARMM Element source/fcm/pshake.fcm 1.1
!
#if KEY_PERT==1 /*pshakefcm*/
! include file for interactions between PERT and SHAKE
!
! QPSHAKE                      logical flag; is there interaction between
!                              PERT and SHAKE to be worried about.
! PCONSTR                      currently defined statically with size
!                              MAXSHK.  Is the constraint affected by PERT
! PREVLA                       prev. value of lambda, used in PSHKSET
! NPCONST                      number of constraints affected by PERTurbation
! PSHAUX                       auxiliary array used to hold Lagrangian
!                              multipliers as well as dr0/dlamb
! 
! integers
      INTEGER NPCONST
!
      real(chm_real) PREVLA

! pshake on flag 
      LOGICAL QPSHAKE
!
#endif /* (pshakefcm)*/
!
end module pshake

