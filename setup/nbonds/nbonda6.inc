#if KEY_EXPAND == 1 && KEY_REPLICA == 1

! Do REPLICA expansion of code
! ##EXPAND  R             .when. REPLICA  EXPAND  (expand_replica)
! ##PASS1   .not.EXPAND
IF(QREP) THEN

#undef KEY_EXPAND
#include "nbonda7.inc"
#define KEY_EXPAND 1
   
! ##PASS2   .not.REPLICA
ELSE

#undef KEY_REPLICA
#include "nbonda7.inc"
#define KEY_REPLICA 1

! ##EXFIN
ENDIF
! ##EXEND
! ##ENDEX    (expand_replica)

#else /* KEY_EXPAND && KEY_REPLICA */

#define NBONDA_R_FLAG 1
#include "nbonda7.inc"
#undef NBONDA_R_FLAG

#endif
