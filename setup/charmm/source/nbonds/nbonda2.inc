#if KEY_QUANTUM == 1 || KEY_GAMESS == 1 || KEY_GAMESSUK == 1 \
  || KEY_CADPAC == 1 || KEY_SCCDFTB == 1 || KEY_QCHEM == 1 \
  || KEY_MNDO97 == 1 || KEY_SQUANTM == 1 || KEY_QTURBO == 1 \
  || KEY_G09 == 1

#define NBONDA_QMMM 1

#endif

#if KEY_EXPAND == 1 && NBONDA_QMMM == 1

! Do QM/MM expansion of code
! ##EXPAND  Q             .when. qmmm  EXPAND  (expand_qmmm)
! ##PASS1   .not.EXPAND
IF(QGMREM) THEN

#undef KEY_EXPAND
#include "nbonda3.inc"
#define KEY_EXPAND 1
   
! ##PASS2 .not.QMMM
ELSE

#undef NBONDA_QMMM
#include "nbonda3.inc"
#define NBONDA_QMMM 1

! ##EXFIN
ENDIF
! ##EXEND
! ##ENDEX    (expand_qmmm)

#else /* KEY_EXPAND && NBONDA_QMMM */

#define NBONDA_Q_FLAG 1
#include "nbonda3.inc"
#undef NBONDA_Q_FLAG

#endif /* KEY_EXPAND && NBONDA_QMMM */
