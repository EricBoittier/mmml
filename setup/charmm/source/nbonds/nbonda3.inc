#if KEY_EXPAND == 1 && KEY_TSM == 1
!-------------------------------------------------------------------
! Do TSM expansion of code
! ##EXPAND  T             .when. TSM  EXPAND  (expand_tsm)
! ##PASS1   .not.EXPAND
IF(LTSM) THEN

#undef KEY_EXPAND
#include "nbonda4.inc"     
#define KEY_EXPAND 1

! ##PASS2   .not.TSM
ELSE

#undef KEY_TSM
#include "nbonda4.inc"
#define KEY_TSM 1
   
! ##EXFIN
ENDIF
! ##EXEND
! ##ENDEX    (expand_tsm)

#else /* KEY_EXPAND && KEY_TSM */

#define NBONDA_T_FLAG 1
#include "nbonda4.inc"
#undef NBONDA_T_FLAG

#endif /* KEY_EXPAND && KEY_TSM */
