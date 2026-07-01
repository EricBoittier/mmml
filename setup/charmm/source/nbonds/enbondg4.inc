#if KEY_EXPAND == 1
! Do VDWOPT expansion of code
! ##EXPAND  V lvgrp       .when.  EXPAND  (expand_vdwopt)
! ##PASS1   lvgrp
IF(LVGRP) THEN

#define ENBONDG_LVGRP 1
#include "enbondg5.inc"
#undef ENBONDG_LVGRP

! ##PASS2   .not.EXPAND
ELSE

#undef KEY_EXPAND
#include "enbondg5.inc"
#define KEY_EXPAND 1

! ##EXFIN
ENDIF
! ##EXEND
! ##ENDEX    (expand_vdwopt)

#else /* KEY_EXPAND */

#define ENBONDG_V_FLAG 1
#define ENBONDG_LVGRP 1

#include "enbondg5.inc"

#undef ENBONDG_V_FLAG
#undef ENBONDG_LVGRP

#endif /* KEY_EXPAND */
