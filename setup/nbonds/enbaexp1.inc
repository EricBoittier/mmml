#if KEY_EXPAND == 1
!-------------------------------------------------------------------
! Do ELECOPT expansion of code
! ##EXPAND E0 lewald cshift cfswit rshft eips noelec lkup geswit .when. EXPAND (expand_elecopt)

! ##PASS1 E4 rshft 
  IF (RSHFT)THEN  

#define ENBAEXP_ELECOPT_SUFFIX E4
#define ENBAEXP_E4 1
#define ENBAEXP_RSHFT 1
#include "enbaexp2.inc"
#undef ENBAEXP_ELECOPT_SUFFIX
#undef ENBAEXP_E4
#undef ENBAEXP_RSHFT

! ##PASS2  E5 lkup .when. LNTAB1
#if KEY_LNTAB1 == 1
  ELSE IF (LNBLKUP) THEN

#define ENBAEXP_ELECOPT_SUFFIX E5
#define ENBAEXP_E5 1
#define ENBAEXP_LKUP 1
#include "enbaexp2.inc"
#undef ENBAEXP_ELECOPT_SUFFIX
#undef ENBAEXP_E5
#undef ENBAEXP_LKUP

#endif  /* KEY_LNTAB1 == 1 */

! ##PASS3  E2 cshift
  ELSE IF (CSHIFT) THEN

#define ENBAEXP_ELECOPT_SUFFIX E2
#define ENBAEXP_E2 1
#define ENBAEXP_CSHIFT 1
#include "enbaexp2.inc"
#undef ENBAEXP_ELECOPT_SUFFIX
#undef ENBAEXP_E2
#undef ENBAEXP_CSHIFT

#if ENBAEXP_B1 == 1

! ##PASS4  E8 lewald .when. B1
  ELSE IF (LEWLD.AND.QHYBH) THEN

#define ENBAEXP_ELECOPT_SUFFIX E8
#define ENBAEXP_E8 1
#define ENBAEXP_LEWALD 1
#include "enbaexp2.inc"
#undef ENBAEXP_ELECOPT_SUFFIX
#undef ENBAEXP_E8
#undef ENBAEXP_LEWALD

#endif  /* ENBAEXP_B1 == 1 */

! ##PASS5  E3 cfswit
  ELSE IF (CFSWIT) THEN

#define ENBAEXP_ELECOPT_SUFFIX E3
#define ENBAEXP_E3 1
#define ENBAEXP_CFSWIT 1
#include "enbaexp2.inc"
#undef ENBAEXP_ELECOPT_SUFFIX
#undef ENBAEXP_E3
#undef ENBAEXP_CFSWIT

! ##PASS7 E7 eips
  ELSE IF (LEIPSX) THEN

#define ENBAEXP_ELECOPT_SUFFIX E7
#define ENBAEXP_E7 1
#define ENBAEXP_EIPS 1
#include "enbaexp2.inc"
#undef ENBAEXP_ELECOPT_SUFFIX
#undef ENBAEXP_E7
#undef ENBAEXP_EIPS

! ##PASS8  E1 lewald
  ELSE IF (LEWLD) THEN

#define ENBAEXP_ELECOPT_SUFFIX E1
#define ENBAEXP_E1 1
#define ENBAEXP_LEWALD 1
#include "enbaexp2.inc"
#undef ENBAEXP_ELECOPT_SUFFIX
#undef ENBAEXP_E1
#undef ENBAEXP_LEWALD

! ##PASS9 E9 geswit
  ELSE IF (GESWIT) THEN

#define ENBAEXP_ELECOPT_SUFFIX E9
#define ENBAEXP_E9 1
#define ENBAEXP_GESWIT 1
#include "enbaexp2.inc"
#undef ENBAEXP_ELECOPT_SUFFIX
#undef ENBAEXP_E9
#undef ENBAEXP_GESWIT

! ##PASS6 E6 noelec
  ELSE

#define ENBAEXP_ELECOPT_SUFFIX E6
#define ENBAEXP_E6 1
#define ENBAEXP_NOELEC 1
#include "enbaexp2.inc"
#undef ENBAEXP_ELECOPT_SUFFIX
#undef ENBAEXP_E6
#undef ENBAEXP_NOELEC

! ##EXFIN
  ENDIF
! ##EXEND
! ##ENDEX    (expand_elecopt)
#else  /* KEY_EXPAND */

#define ENBAEXP_ELECOPT_SUFFIX E0
#define ENBAEXP_E0 1

#define ENBAEXP_LEWALD 1
#define ENBAEXP_CSHIFT 1
#define ENBAEXP_CFSWIT 1
#define ENBAEXP_RSHFT 1
#define ENBAEXP_EIPS 1
#define ENBAEXP_NOELEC 1
#define ENBAEXP_LKUP 1
#define ENBAEXP_GESWIT 1

#include "enbaexp2.inc"

#undef ENBAEXP_ELECOPT_SUFFIX
#undef ENBAEXP_E0
#undef ENBAEXP_LEWALD
#undef ENBAEXP_CSHIFT
#undef ENBAEXP_CFSWIT
#undef ENBAEXP_RSHFT
#undef ENBAEXP_EIPS
#undef ENBAEXP_NOELEC
#undef ENBAEXP_LKUP
#undef ENBAEXP_GESWIT

#endif  /* KEY_EXPAND */
