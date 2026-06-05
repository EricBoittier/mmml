#if KEY_EXPAND == 1 && KEY_PERT == 1

! Do PERT expansion of code                
! ##EXPAND P nopert       .when. PERT  EXPAND  (expand_pert)
! ##PASS1  .not.EXPAND
IF(QPERT) THEN

#undef KEY_EXPAND
#include "nbonda8.inc"
#define KEY_EXPAND 1
   
! ##PASS2  .not.PERT  nopert
ELSE

#undef KEY_PERT
#define NBONDA_NOPERT 1
#include "nbonda8.inc"
#define KEY_PERT 1
#undef NBONDA_NOPERT
   
! ##EXFIN
ENDIF
! ##EXEND
! ##ENDEX    (expand_pert)

#else /* KEY_EXAND && KEY_PERT */

#define NBONDA_P_FLAG 1
#define NBONDA_NOPERT 1

#include "nbonda8.inc"

#undef NBONDA_P_FLAG
#undef NBONDA_NOPERT

#endif /* KEY_EXAND && KEY_PERT */
