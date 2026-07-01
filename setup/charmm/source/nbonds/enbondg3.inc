#if KEY_EXPAND == 1 && KEY_PBOUND == 1

!-------------------------------------------------------------------
! Do PBOUND expansion of code
! ##EXPAND P             .when. PBOUND EXPAND  (expand_pbound)
! ##PASS1 .not.EXPAND
  IF(QBOUN) THEN
     
#undef KEY_EXPAND
#include "enbondg4.inc"
#define KEY_EXPAND 1

! ##PASS2 .not.PBOUND
  ELSE

#undef KEY_PBOUND
#include "enbondg4.inc"
#define KEY_PBOUND 1
     
! ##EXFIN
  ENDIF
! ##EXEND
! ##ENDEX (expand_pbound)

#else /* KEY_EXPAND && KEY_PBOUND */

#define ENBONDG_P_FLAG 1
#include "enbondg4.inc"
#undef ENBONDG_P_FLAG

#endif /* KEY_EXPAND && KEY_PBOUND */
