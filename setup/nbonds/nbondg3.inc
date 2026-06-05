#if KEY_EXPAND == 1

! Do IMOVE expansion of code
! ##EXPAND  imove         .when.          EXPAND  (expand_imove)
! ##PASS1   imove  .not.EXPAND
IF(QMOVE) THEN

#undef KEY_EXPAND
#define NBONDG_IMOVE 1
#include "nbondg4.inc"
#undef NBONDG_IMOVE
#define KEY_EXPAND 1

! ##PASS2
ELSE

#include "nbondg4.inc"

! ##EXFIN
ENDIF
! ##EXEND
! ##ENDEX    (expand_imove)

#else /* KEY_EXPAND */

#define NBONDG_IMOVE 1
#include "nbondg4.inc"
#undef NBONDG_IMOVE

#endif /* KEY_EXPAND */
