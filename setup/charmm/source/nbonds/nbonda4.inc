#if KEY_EXPAND == 1 && KEY_MTS == 1

!-------------------------------------------------------------------
! Do MTS expansion of code
! ##EXPAND  M  nomts      .when. MTS  EXPAND  (expand_mts)
! ##PASS1  .not.EXPAND
IF (QTBMTS) THEN

#undef KEY_EXPAND
#include "nbonda5.inc"
#define KEY_EXPAND 1

! ##PASS2  .not.MTS  nomts
ELSE

#undef KEY_MTS
#define NBONDA_NOMTS 1
#include "nbonda5.inc"
#define KEY_MTS 1
#undef NBONDA_NOMTS

! ##EXFIN
ENDIF
! ##EXEND
! ##ENDEX    (expand_mts)

#else /* KEY_EXPAND && KEY_MTS */

#define NBONDA_M_FLAG 1
#define NBONDA_NOMTS 1

#include "nbonda5.inc"

#undef NBONDA_NOMTS
#undef NBONDA_M_FLAG

#endif /* KEY_EXPAND && KEY_MTS */
