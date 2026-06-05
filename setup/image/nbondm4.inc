#if NBONDM_EXPAND == 1 && KEY_TSM == 1

#undef NBONDM_TSM
#define NBONDM_TSM 1

  !-------------------------------------------------------------------
  ! Do TSM expansion of code
! ##EXPAND  T             .when. TSM  EXPAND  (expand_tsm)

#undef NBONDM_TGUARD

! ##PASS1   .not.EXPAND
  IF(LTSM) THEN

#undef NBONDM_EXPAND
#include "nbondm5.inc"
#define NBONDM_EXPAND 1

! ##PASS2   .not.TSM
  ELSE

#undef NBONDM_TSM
#include "nbondm5.inc"
#define NBONDM_TSM 1

! ##EXFIN
  ENDIF
! ##EXEND
! ##ENDEX    (expand_tsm)

#else  /* NBONDM_EXPAND == 1 && KEY_TSM == 1 */

#undef NBONDM_TSM
#if KEY_TSM == 1
#define NBONDM_TSM 1
#endif

#define NBONDM_TGUARD 1
#include "nbondm5.inc"
#undef NBONDM_TGUARD

#endif  /* NBONDM_EXPAND == 1 && KEY_TSM == 1 */
