#if ENBONDA_CFF_EXPAND == 1  
  !-------------------------------------------------------------------
  ! Do VDWOPT expansion of code
! ##EXPAND  V dswit dfswit dshft .when.  EXPAND  (expand_vdwopt)

#undef ENBONDA_CFF_VGUARD
#undef ENBONDA_CFF_DSWIT
#undef ENBONDA_CFF_DFSWIT
#undef ENBONDA_CFF_DSHFT

! ##PASS1   dswit
  IF (DSWIT) THEN

#define ENBONDA_CFF_DSWIT 1
#include "enbonda_cff_expand.inc"
#undef ENBONDA_CFF_DSWIT

! ##PASS2   dfswit
  ELSE IF (DFSWIT) THEN

#define ENBONDA_CFF_DFSWIT 1
#include "enbonda_cff_expand.inc"
#undef ENBONDA_CFF_DFSWIT

! ##PASS3   dshft
  ELSE IF (DSHFT) THEN

#define ENBONDA_CFF_DSHFT 1
#include "enbonda_cff_expand.inc"
#undef ENBONDA_CFF_DSHFT

! ##PASS4
  ELSE

#include "enbonda_cff_expand.inc"
     
! ##EXFIN
  ENDIF
! ##EXEND

! ##ENDEX    (expand_vdwopt)

#else  /* ENBONDA_CFF_EXPAND == 1 */

#define ENBONDA_CFF_VGUARD 1
#define ENBONDA_CFF_DSWIT 1
#define ENBONDA_CFF_DFSWIT 1
#define ENBONDA_CFF_DSHFT 1
#include "enbonda_cff_expand.inc"
#undef ENBONDA_CFF_VGUARD
#undef ENBONDA_CFF_DSWIT
#undef ENBONDA_CFF_DFSWIT
#undef ENBONDA_CFF_DSHFT

  
#endif  /* ENBONDA_CFF_EXPAND == 1 */
