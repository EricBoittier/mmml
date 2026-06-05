#if KEY_EXPAND == 1

! Do ELECOPT expansion of code
! ##EXPAND  E lrsw lcsw lewald lelse reips ceips .when.  EXPAND  (expand_elecopt)
! ##PASS1  lrsw
IF (LRSW) THEN

#define ENBONDG_LRSW 1
#include "enbondg_expand.inc"
#undef ENBONDG_LRSW
   
! ##PASS2  lcsw
ELSE IF (LCSW) THEN

#define ENBONDG_LCSW 1
#include "enbondg_expand.inc"
#undef ENBONDG_LCSW

! ##PASS3  lewald
ELSE IF(LEWALD) THEN

#define ENBONDG_LEWALD 1
#include "enbondg_expand.inc"
#undef ENBONDG_LEWALD

#if KEY_NBIPS == 1

! ##PASS4  reips   .when. NBIPS
ELSE IF(REIPS) THEN

#define ENBONDG_REIPS 1
#include "enbondg_expand.inc"
#undef ENBONDG_REIPS

! ##PASS5  ceips   .when. NBIPS
ELSE IF(CEIPS) THEN

#define ENBONDG_CEIPS 1
#include "enbondg_expand.inc"
#undef ENBONDG_CEIPS

#endif /* KEY_NBIPS */

! ##PASS6  lelse
ELSE

#define ENBONDG_LELSE 1
#include "enbondg_expand.inc"
#undef ENBONDG_LELSE

! ##EXFIN
ENDIF
! ##EXEND
! ##ENDEX    (expand_elecopt)

#else /* KEY_EXPAND */

#define ENBONDG_E_FLAG 1
#define ENBONDG_LRSW 1
#define ENBONDG_LCSW 1
#define ENBONDG_LEWALD 1
#define ENBONDG_LELSE 1
#define ENBONDG_REIPS 1
#define ENBONDG_CEIPS 1

#include "enbondg_expand.inc"

#undef ENBONDG_E_FLAG
#undef ENBONDG_LRSW
#undef ENBONDG_LCSW
#undef ENBONDG_LEWALD
#undef ENBONDG_LELSE
#undef ENBONDG_REIPS
#undef ENBONDG_CEIPS

#endif /* KEY_EXPAND */
