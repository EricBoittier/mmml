#if NBONDM_EXPAND == 1 && KEY_MTS == 1

#undef NBONDM_MTS
#define NBONDM_MTS 1

  !-------------------------------------------------------------------
  ! Do MTS expansion of code
! ##EXPAND  M  nomts      .when. MTS  EXPAND  (expand_mts)

#undef NBONDM_MGUARD
#undef NBONDM_NOMTS

! ##PASS1  .not.EXPAND
  IF (QTBMTS) THEN

#undef NBONDM_EXPAND
#include "nbondm2.inc"
#define NBONDM_EXPAND 1

! ##PASS2  .not.MTS  nomts
  ELSE

#undef NBONDM_MTS
#define NBONDM_NOMTS 1
#include "nbondm2.inc"
#define NBONDM_MTS 1
#undef NBONDM_NOMTS

! ##EXFIN
  ENDIF
! ##EXEND
! ##ENDEX    (expand_mts)
#else  /* NBONDM_EXPAND == 1 && KEY_MTS == 1 */

#undef NBONDM_MTS
#if KEY_MTS == 1
#define NBONDM_MTS 1
#endif

#define NBONDM_NOMTS 1
#define NBONDM_MGUARD 1
#include "nbondm2.inc"

#endif  /* NBONDM_EXPAND == 1 && KEY_MTS == 1 */
