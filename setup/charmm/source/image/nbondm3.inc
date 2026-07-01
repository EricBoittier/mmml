#if NBONDM_EXPAND == 1
  !-------------------------------------------------------------------
  ! Do LEXS expansion of code
! ##EXPAND  L  lexs       .when.        EXPAND  (expand_lexs)

#undef NBONDM_LGUARD
#undef NBONDM_LEXS

! ##PASS1   lexs  .not.EXPAND
  IF(LEXS) THEN

#define NBONDM_LEXS 1
#undef NBONDM_EXPAND
#include "nbondm4.inc"
#undef NBONDM_LEXS
#define NBONDM_EXPAND 1

! ##PASS2   
  ELSE

#include "nbondm4.inc"

! ##EXFIN
  ENDIF
! ##EXEND
! ##ENDEX    (expand_lexs)

#else  /* NBONDM_EXPAND == 1 */

#define NBONDM_LGUARD 1
#define NBONDM_LEXS 1
#include "nbondm4.inc"
#undef NBONDM_LGUARD
#undef NBONDM_LEXS

#endif  /* NBONDM_EXPAND == 1 */
