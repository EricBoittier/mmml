#if KEY_EXPAND == 1 && KEY_PBOUND == 1

!-------------------------------------------------------------------
! Do PBOUND expansion of code
! ##EXPAND  B             .when. PBOUND  EXPAND  (expand_pbound)
! ##PASS1
IF(QBOUN) THEN

#include "nbonda_expand.inc"

! ##PASS2   .not.PBOUND
ELSE

#undef KEY_PBOUND
#include "nbonda_expand.inc"
#define KEY_PBOUND 1
   
! ##EXFIN
ENDIF
! ##EXEND
! ##ENDEX    (expand_pbound)

#else /* KEY_EXPAND && KEY_PBOUND */

#define NBONDA_B_FLAG 1
#include "nbonda_expand.inc"
#undef NBONDA_B_FLAG

#endif /* KEY_EXPAND && KEY_PBOUND */
