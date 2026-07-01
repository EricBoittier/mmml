module contrl
  use chm_kinds
  use dimens_fcm
  !
  !     Control variables for minimization and dynamics
  !
  !     Purpose: Setting the modes for the minimization or dynamics
  !     algorithms and the shake constraints applied to the system.
  !
  !     Variables:
  !
  !     INBFRQ      Update frequency for the non-bonded list. Used in the
  !                 subroutine ENERGY() to decide updates. When set to :
  !                  0 --> no updates will be done.
  !                 +n --> an update is done every time MOD(ECALLS,n).EQ.0 .
  !                        (ECALLS is declared in ENERGY.FCM and is
  !                        incremented by ICALL every time ENERGY(,,,ICAL)
  !                        is called ).
  !                 -1 --> an update is done when necessary (default), but only
  !                        when ICALL is non zero.
  !
  !     ILANG       Controls whether langevin dynamics is used
#if KEY_ACE==1
  !     QLBACE      Calculate Langevin dynamics depending on Born solv. radius
#endif 
  !     IREST       Restarting dynamics run
  !     ILBFRQ      Langevin dynamics update frequency
  !     NPRINT      Printing frequency for minimisation
  !     NUPFRQ      Minimum update frequency for minimisation
  !
  !     TOTUPD      incr. by 1 when a nbonds-list update is done from UPDECI.
  !
  !     QSADLE      Do saddle point minimization (minimize on delE**2 surface)
  !
  !     IDDF        Pointer to HEAP for Hessian (and flag)
  !
  !     QNUMER      Use numerical first derivatives by finite differences 
  !                 during minimization
  !     DYNAMQ      true if performing dynamics 
  !     MDSTEP      MD step counter
  !
  !     Variables used in MINMIZ() :
  !     MINXYZ      perform a coordinate minimisation.
  !     LMINUC      do a lattice parameter minimisation.
  !
  !     QSEGSRT     stops rotation/translation on a segment basis
  !
#if KEY_CHEQ==1
  !     QCGMIN     perform charge optimization according to EE principle
#endif 
  !     RCENT       To center the system at the geometric center of
  !                 the first NCRES residues
  real(chm_real),allocatable,dimension(:) :: iddf
  INTEGER  ILANG, ILBFRQ, INBFRQ, IREST, &
       NPRINT, NUPFRQ, TOTUPD , &
#if KEY_MNDO97==1
       ISTEPQM, ISTOPQM,                       & ! namkh 09/12/03
#endif 
       NCRES, MDSTEP                         ! JG 5/02
  
  LOGICAL  MINXYZ, LMINUC, QSADLE, QNUMER,DYNAMQ &
#if KEY_ACE==1
       , QLBACE                         & 
#endif
       , RCENT                          & ! JG 5/02
#if KEY_CHEQ==1
       , QCGMIN                         
#endif
#if KEY_CHEQ==0
  ;          
#endif
  ! stops rotation/translation on a segment basis
  logical :: QSEGSRT

contains

  subroutine contrl_init()
    inbfrq=-1
    ilbfrq=50
    ilang=0
    irest=0
    nprint=10
    mdstep=0
    nupfrq=50
    dynamq=.false.
#if KEY_ACE==1
    qlbace=.false.  
#endif
    !
    lminuc=.false.
    minxyz=.true.
    !
    totupd = 0
    qsadle=.false.
    return
  end subroutine contrl_init
end module contrl

