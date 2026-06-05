#if KEY_EXPAND == 1

!-------------------------------------------------------------------
! Do IMOVE expansion of code
! ##EXPAND  imove         .when.          EXPAND  (expand_imove)
! ##PASS1   imove  .not.EXPAND
IF(QMOVE) THEN

#undef KEY_EXPAND
#define NBONDA_IMOVE 1
#include "nbonda6.inc"
#undef NBONDA_IMOVE
#define KEY_EXPAND 1
   
! ##PASS2
ELSE

#include "nbonda6.inc"

! ##EXFIN
ENDIF
! ##EXEND
! ##ENDEX    (expand_imove)

#else /* KEY_EXPAND */

#define NBONDA_IMOVE 1
#include "nbonda6.inc"
#undef NBONDA_IMOVE

#endif /* KEY_EXPAND */
