#if KEY_EXPAND == 1 && KEY_MTS == 1

! Do MTS expansion of code
! ##EXPAND  M             .when. MTS  EXPAND  (expand_mts)
! ##PASS1  .not.EXPAND
IF (QTBMTS) THEN

#undef KEY_EXPAND
#include "nbondg3.inc"
#define KEY_EXPAND 1

! ##PASS2  .not.MTS
ELSE

#undef KEY_MTS
#include "nbondg3.inc"
#define KEY_MTS 1

! ##EXFIN
ENDIF
! ##EXEND
! ##ENDEX    (expand_mts)

#else /* KEY_EXPAND && KEY_MTS */

#define NBONDG_M_FLAG 1
#include "nbondg3.inc"
#undef NBONDG_M_FLAG

#endif /* KEY_EXPAND && KEY_MTS */
