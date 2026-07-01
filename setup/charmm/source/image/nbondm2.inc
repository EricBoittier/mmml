#if NBONDM_EXPAND == 1 && KEY_REPLICA == 1
  !-------------------------------------------------------------------
  ! Do REPLICA expansion of code

#undef NBONDM_REPLICA
#define NBONDM_REPLICA 1

! ##EXPAND  R             .when. REPLICA  EXPAND  (expand_replica)

#undef NBONDM_RGUARD

! ##PASS1   .not.EXPAND
  IF(QREP) THEN

#undef NBONDM_EXPAND
#include "nbondm3.inc"
#define NBONDM_EXPAND 1

! ##PASS2   .not.REPLICA
  ELSE

#undef NBONDM_REPLICA
#include "nbondm3.inc"
#define NBONDM_REPLICA 1

! ##EXFIN
  ENDIF
! ##EXEND
! ##ENDEX    (expand_replica)
#else  /* NBONDM_EXPAND == 1 && KEY_REPLICA == 1 */

#undef NBONDM_REPLICA
#if KEY_REPLICA == 1
#define NBONDM_REPLICA 1
#endif

#define NBONDM_RGUARD 1
#include "nbondm3.inc"
#undef NBONDM_RGUARD
  
#endif  /* NBONDM_EXPAND == 1 && KEY_REPLICA == 1 */
