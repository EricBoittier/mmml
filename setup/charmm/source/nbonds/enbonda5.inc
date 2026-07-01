#if ENBONDA_EXPAND == 1

!-------------------------------------------------------------------
! Do VDWOPT expansion of code
! ##EXPAND  V dswit dfswit dshft dvips gvswit .when. EXPAND (expand_vdwopt)
! ##PASS1   dswit
  IF (DSWIT) THEN

#define ENBONDA_DSWIT 1
#include "enbonda_expand.inc"
#undef ENBONDA_DSWIT
     
! ##PASS2   dfswit
  ELSE IF (DFSWIT) THEN

#define ENBONDA_DFSWIT 1
#include "enbonda_expand.inc"
#undef ENBONDA_DFSWIT

! ##PASS3   dshft
  ELSE IF (DSHFT) THEN

#define ENBONDA_DSHFT 1
#include "enbonda_expand.inc"
#undef ENBONDA_DSHFT

! ##PASS4A   dbexp  
  ELSE IF (DBEXP) THEN

#define ENBONDA_DBEXP 1
#include "enbonda_expand.inc"
#undef ENBONDA_DBEXP


#if KEY_NBIPS == 1
! ##PASS4B   dvips   .when. NBIPS
  ELSE IF (DVIPS) THEN

#define ENBONDA_DVIPS 1
#include "enbonda_expand.inc"
#undef ENBONDA_DVIPS

! ##PASS4C   dbexpips   
  ELSE IF (DBEXPIPS) THEN

#define ENBONDA_DBEXPIPS 1
#include "enbonda_expand.inc"
#undef ENBONDA_DBEXPIPS

#endif /* KEY_NBIPS */

! ##PASSA   gvswit
  ELSE IF (GVSWIT) THEN

#define ENBONDA_GVSWIT 1
#include "enbonda_expand.inc"
#undef ENBONDA_GVSWIT

! ##PASS5
  ELSE

#include "enbonda_expand.inc"

! ##EXFIN
  ENDIF
! ##EXEND
! ##ENDEX    (expand_vdwopt)

#else /* ENBONDA_EXPAND */

#define ENBONDA_V_FLAG 1
#define ENBONDA_DSWIT 1
#define ENBONDA_DFSWIT 1
#define ENBONDA_DSHFT 1
#define ENBONDA_DBEXP 1
#define ENBONDA_DVIPS 1
#define ENBONDA_DBEXPIPS 1
#define ENBONDA_GVSWIT 1
  
#include "enbonda_expand.inc"

#undef ENBONDA_V_FLAG
#undef ENBONDA_DSWIT
#undef ENBONDA_DFSWIT
#undef ENBONDA_DSHFT
#undef ENBONDA_DBEXP
#undef ENBONDA_DVIPS
#undef ENBONDA_DBEXPIPS
#undef ENBONDA_GVSWIT
  
#endif /* ENBONDA_EXPAND */
