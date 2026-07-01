#if ENBONDA_CFF_EXPAND == 1 && KEY_PBOUND == 1

  !-------------------------------------------------------------------
  ! Do PBOUND expansion of code
! ##EXPAND  P             .when. PBOUND EXPAND  (expand_pbound)

#define ENBONDA_CFF_PBOUND 1

! ##PASS1   .not.EXPAND
  IF(QBOUN) THEN

#undef ENBONDA_CFF_EXPAND
#include "enbonda_cff4.inc"
#define ENBONDA_CFF_EXPAND 1
     
! ##PASS2   .not.PBOUND
  ELSE

#undef ENBONDA_CFF_PBOUND
#include "enbonda_cff4.inc"
#define ENBONDA_CFF_PBOUND 1
     
! ##EXFIN
  ENDIF
! ##EXEND

#else  /* ENBONDA_CFF_EXPAND == 1 && KEY_PBOUND == 1 */

#undef ENBONDA_CFF_PBOUND
#if KEY_PBOUND == 1
#define ENBONDA_CFF_PBOUND 1
#endif  /* KEY_PBOUND == 1 */
  
#define ENBONDA_CFF_PGUARD 1
#include "enbonda_cff4.inc"
#undef ENBONDA_CFF_PGUARD

#endif  /* ENBONDA_CFF_EXPAND == 1 && KEY_PBOUND == 1 */
! ##ENDEX    (expand_pbound)
