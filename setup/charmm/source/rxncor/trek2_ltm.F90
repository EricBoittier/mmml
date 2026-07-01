module trek2
  use chm_kinds
  implicit none
!CHARMM Element source/fcm/trek2.fcm $Revision: 1.4 $

! Common-block for TReK version 2.10
! By Stefan Fischer; July 5-2003.
!
! Include in  .source/rxncor/adiab.src
! Include in  .source/rxncor/travel.src

  
  INTEGER              TOTCYC,TOTMIN,INSMIN,RMVMIN,LSTOT1,LSTOT2
  INTEGER              MAXCAL,LASTMV
  LOGICAL              LSCAN1,LSCAN2,LSCAN3

  INTEGER      TOTLUP,CPLUP1,CPLUP2
  LOGICAL      LTMPRJ

  real(chm_real)       ORISTP
  INTEGER      OSSKIP,TOTOSC
  integer,PARAMETER :: HISLEN=60

  real(chm_real)    GRATOL, STPTOL, ENETOL, BRASTP,BRAMAG,BRASCA
  real(chm_real)           MAXTOL, FIRSTP, ATOMAX
  INTEGER          LXEVAL,MODXIT

end module trek2

