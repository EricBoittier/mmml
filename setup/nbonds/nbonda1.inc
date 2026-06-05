#if KEY_EXPAND == 1

!-------------------------------------------------------------------
! Do LEXS expansion of code
! ##EXPAND  L  lexs       .when.        EXPAND  (expand_lexs)
! ##PASS1   lexs  .not.EXPAND
IF(LEXS) THEN

#undef KEY_EXPAND
#define NBONDA_LEXS 1

#include "nbonda2.inc"

#undef NBONDA_LEXS
#define KEY_EXPAND 1
   
! ##PASS2   
ELSE

#include "nbonda2.inc"
   
! ##EXFIN
ENDIF
! ##EXEND
! ##ENDEX    (expand_lexs)

#else /* KEY_EXPAND */

#define NBONDA_L_FLAG 1
#define NBONDA_LEXS 1

#include "nbonda2.inc"

#undef NBONDA_L_FLAG
#undef NBONDA_LEXS

#endif /* KEY_EXPAND */
