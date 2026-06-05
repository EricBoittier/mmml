#if ENBONDA_EXPAND == 1 && KEY_PBOUND == 1

#define ENBONDA_PBOUND 1

!-------------------------------------------------------------------
! Do PBOUND expansion of code
! ##EXPAND  P             .when. PBOUND EXPAND  (expand_pbound)
! ##PASS1   .not.EXPAND
  IF(QBOUN) THEN

#undef ENBONDA_EXPAND
#include "enbonda4.inc"
#define ENBONDA_EXPAND 1
     
! ##PASS2   .not.PBOUND
  ELSE

#undef ENBONDA_PBOUND
#include "enbonda4.inc"
     
! ##EXFIN
  ENDIF
! ##EXEND
! ##ENDEX    (expand_pbound)

#else /* ENBONDA_EXPAND && KEY_PBOUND */

#undef ENBONDA_PBOUND
#if KEY_PBOUND == 1
#define ENBONDA_PBOUND 1
#endif

#define ENBONDA_P_FLAG 1
#include "enbonda4.inc"
#undef ENBONDA_P_FLAG
  
#endif /* ENBONDA_EXPAND && KEY_PBOUND */
