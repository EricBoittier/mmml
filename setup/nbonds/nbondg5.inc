#if KEY_EXPAND == 1 && KEY_PERT == 1

! Do PERT expansion of code
! ##EXPAND P              .when. PERT  EXPAND  (expand_pert)
! ##PASS1  .not.EXPAND
IF(QPERT) THEN

#undef KEY_EXPAND
#include "nbondg6.inc"
#define KEY_EXPAND 1

! ##PASS2  .not.PERT
ELSE

#undef KEY_PERT
#include "nbondg6.inc"
#define KEY_PERT 1

! ##EXFIN
ENDIF
! ##EXEND
! ##ENDEX    (expand_pert)

#else /* KEY_EXPAND && KEY_PERT */

#define NBONDG_P_FLAG 1
#include "nbondg6.inc"
#undef NBONDG_P_FLAG

#endif /* KEY_EXPAND && KEY_PERT */
