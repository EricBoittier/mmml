#if ENBONDA_CFF_EXPAND == 1  
  !-------------------------------------------------------------------
  ! Do ELECOPT expansion of code
! ##EXPAND E lewald rswit rfswit cshft cshift rshft cswit cfswit .when. EXPAND (expand_elecopt)

#undef ENBONDA_CFF_EGUARD
#undef ENBONDA_CFF_LEWALD
#undef ENBONDA_CFF_RSWIT
#undef ENBONDA_CFF_RFSWIT
#undef ENBONDA_CFF_CSHFT
#undef ENBONDA_CFF_CSHIFT
#undef ENBONDA_CFF_RSHFT
#undef ENBONDA_CFF_CSWIT
#undef ENBONDA_CFF_CFSWIT
  
! ##PASS1 lewald
  IF (LEWALD) THEN

#define ENBONDA_CFF_LEWALD 1
#include "enbonda_cff5.inc"
#undef ENBONDA_CFF_LEWALD

! ##PASS2  rswit    .not.EXPAND
  ELSE IF (RSWIT) THEN

#define ENBONDA_CFF_RSWIT 1
#undef ENBONDA_CFF_EXPAND
#include "enbonda_cff5.inc"
#undef ENBONDA_CFF_RSWIT
#define ENBONDA_CFF_EXPAND 1

! ##PASS3  rfswit   .not.EXPAND
  ELSE IF (RFSWIT) THEN

#define ENBONDA_CFF_RFSWIT 1
#undef ENBONDA_CFF_EXPAND
#include "enbonda_cff5.inc"
#undef ENBONDA_CFF_RFSWIT
#define ENBONDA_CFF_EXPAND 1

! ##PASS4  cshft
  ELSE IF (CSHFT) THEN

#define ENBONDA_CFF_CSHFT 1
#include "enbonda_cff5.inc"
#undef ENBONDA_CFF_CSHFT

! ##PASS5  cshift
  ELSE IF (CSHIFT) THEN

#define ENBONDA_CFF_CSHIFT 1
#include "enbonda_cff5.inc"
#undef ENBONDA_CFF_CSHIFT

! ##PASS6  rshft
  ELSE IF (RSHFT) THEN

#define ENBONDA_CFF_RSHFT 1
#include "enbonda_cff5.inc"
#undef ENBONDA_CFF_RSHFT

! ##PASS7  cswit    .not.EXPAND
  ELSE IF (CSWIT) THEN

#define ENBONDA_CFF_CSWIT 1
#undef ENBONDA_CFF_EXPAND
#include "enbonda_cff5.inc"
#undef ENBONDA_CFF_CSWIT
#define ENBONDA_CFF_EXPAND 1

! ##PASS8  cfswit
  ELSE IF (CFSWIT) THEN

#define ENBONDA_CFF_CFSWIT 1
#include "enbonda_cff5.inc"
#undef ENBONDA_CFF_CFSWIT

! ##PASS9
  ELSE

#include "enbonda_cff5.inc"

! ##EXFIN
  ENDIF
! ##EXEND

#else  /* ENBONDA_CFF_EXPAND == 1 */

#define ENBONDA_CFF_EGUARD 1
#define ENBONDA_CFF_LEWALD 1
#define ENBONDA_CFF_RSWIT 1
#define ENBONDA_CFF_RFSWIT 1
#define ENBONDA_CFF_CSHFT 1
#define ENBONDA_CFF_CSHIFT 1
#define ENBONDA_CFF_RSHFT 1
#define ENBONDA_CFF_CSWIT 1
#define ENBONDA_CFF_CFSWIT 1

#include "enbonda_cff5.inc"
  
#undef ENBONDA_CFF_EGUARD
#undef ENBONDA_CFF_LEWALD
#undef ENBONDA_CFF_RSWIT
#undef ENBONDA_CFF_RFSWIT
#undef ENBONDA_CFF_CSHFT
#undef ENBONDA_CFF_CSHIFT
#undef ENBONDA_CFF_RSHFT
#undef ENBONDA_CFF_CSWIT
#undef ENBONDA_CFF_CFSWIT

#endif  /* ENBONDA_CFF_EXPAND == 1 */
! ##ENDEX    (expand_elecopt)
