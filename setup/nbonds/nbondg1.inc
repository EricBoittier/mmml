#if KEY_EXPAND == 1

! Do LEXS expansion of code
! ##EXPAND  L  lexs       .when.        EXPAND  (expand_lexs)
! ##PASS1   lexs  .not.EXPAND
IF (LEXS) THEN

#undef KEY_EXPAND
#define NBONDG_LEXS 1
#include "nbondg2.inc"
#define KEY_EXPAND 1
#undef NBONDG_LEXS

! ##PASS2
ELSE

#include "nbondg2.inc"

! ##EXFIN
ENDIF
! ##EXEND
! ##ENDEX    (expand_lexs)

#else /* KEY_EXPAND */

#define NBONDG_L_FLAG 1
#define NBONDG_LEXS 1
#include "nbondg2.inc"
#undef NBONDG_L_FLAG
#undef NBONDG_LEXS

#endif /* KEY_EXPAND */
