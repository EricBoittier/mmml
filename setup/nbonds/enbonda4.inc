#if ENBONDA_EXPAND == 1
  
!-------------------------------------------------------------------
! Do ELECOPT expansion of code
! ##EXPAND E lewald rswit rfswit cshft cshift rshft cswit cfswit reips ceips geswit .when. EXPAND (exp_elec)
! ##PASS1 lewald
  IF (LEWALD) THEN

#define ENBONDA_LEWALD 1
#include "enbonda5.inc"
#undef ENBONDA_LEWALD
     
! ##PASS2  rswit    .not.EXPAND
  ELSE IF (RSWIT) THEN

#undef ENBONDA_EXPAND
#define ENBONDA_RSWIT 1
#include "enbonda5.inc"
#undef ENBONDA_RSWIT
     
! ##PASS3  rfswit   .not.EXPAND
  ELSE IF (RFSWIT) THEN

#define ENBONDA_RFSWIT 1
#include "enbonda5.inc"
#undef ENBONDA_RFSWIT
#define ENBONDA_EXPAND 1
     
! ##PASS4  cshft
  ELSE IF (CSHFT) THEN

#define ENBONDA_CSHFT 1
#include "enbonda5.inc"
#undef ENBONDA_CSHFT
     
! ##PASS5  cshift
  ELSE IF (CSHIFT) THEN

#define ENBONDA_CSHIFT 1
#include "enbonda5.inc"
#undef ENBONDA_CSHIFT
     
! ##PASS6  rshft
  ELSE IF (RSHFT) THEN

#define ENBONDA_RSHFT 1
#include "enbonda5.inc"
#undef ENBONDA_RSHFT
     
! ##PASS7  cswit    .not.EXPAND
  ELSE IF (CSWIT) THEN

#undef ENBONDA_EXPAND
#define ENBONDA_CSWIT 1
#include "enbonda5.inc"
#define ENBONDA_EXPAND 1
#undef ENBONDA_CSWIT
     
! ##PASS8  cfswit
  ELSE IF (CFSWIT) THEN

#define ENBONDA_CFSWIT 1
#include "enbonda5.inc"
#undef ENBONDA_CFSWIT
     
#if KEY_NBIPS == 1
     
! ##PASSA  reips   .when. NBIPS
  ELSE IF (REIPS) THEN

#define ENBONDA_REIPS 1
#include "enbonda5.inc"
#undef ENBONDA_REIPS
     
! ##PASSB  ceips   .when. NBIPS
  ELSE IF (CEIPS) THEN

#define ENBONDA_CEIPS 1
#include "enbonda5.inc"
#undef ENBONDA_CEIPS
     
#endif /* KEY_NBIPS */
     
! ##PASSC  geswit
  ELSE IF (GESWIT) THEN

#define ENBONDA_GESWIT 1
#include "enbonda5.inc"
#undef ENBONDA_GESWIT
     
! ##PASS9
  ELSE

#include "enbonda5.inc"
     
! ##EXFIN
  ENDIF
! ##EXEND
! ##ENDEX    (exp_elec)

#else /* ENBONDA_EXPAND */

#define ENBONDA_E_FLAG 1
#define ENBONDA_LEWALD 1
#define ENBONDA_RSWIT 1
#define ENBONDA_RFSWIT 1
#define ENBONDA_CSHFT 1
#define ENBONDA_CSHIFT 1
#define ENBONDA_RSHFT 1
#define ENBONDA_CSWIT 1
#define ENBONDA_CFSWIT 1
#define ENBONDA_REIPS 1
#define ENBONDA_CEIPS 1
#define ENBONDA_GESWIT 1
  
#include "enbonda5.inc"

#undef ENBONDA_E_FLAG
#undef ENBONDA_LEWALD
#undef ENBONDA_RSWIT
#undef ENBONDA_RFSWIT
#undef ENBONDA_CSHFT
#undef ENBONDA_CSHIFT
#undef ENBONDA_RSHFT
#undef ENBONDA_CSWIT
#undef ENBONDA_CFSWIT
#undef ENBONDA_REIPS
#undef ENBONDA_CEIPS
#undef ENBONDA_GESWIT
  
#endif /* ENBONDA_EXPAND */
