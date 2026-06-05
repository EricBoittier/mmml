#if NBONDM_EXPAND == 1
  !-------------------------------------------------------------------
  ! Do IMOVE expansion of code
! ##EXPAND  imove         .when.          EXPAND  (expand_imove)

#undef NBONDM_IMOVE

! ##PASS1  imove .not.EXPAND
  IF(QMOVE) THEN

#define NBONDM_IMOVE 1
#undef NBONDM_EXPAND
#include "nbondm_expand.inc"
#undef NBONDM_IMOVE
#define NBONDM_EXPAND 1

! ##PASS2   .not.imove
  ELSE

#include "nbondm_expand.inc"

! ##EXFIN
  ENDIF
! ##EXEND
! ##ENDEX    (expand_imove)

#else  /* NBONDM_EXPAND == 1 */

#define NBONDM_IMOVE 1
#include "nbondm_expand.inc"
#undef NBONDM_IMOVE

#endif  /* NBONDM_EXPAND == 1 */
