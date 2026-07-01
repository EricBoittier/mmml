! Do ELECOPT expansion of code
#if KEY_EXPAND == 1

! ##EXPAND E0 lewald cshift cfswit rshft eips noelec lkup lewhyb geswit .when. EXPAND (expand_elecopt)

! ##PASS1  E4 rshft

#define ENBAEXP_E_SUFFIX E4
#define ENBAEXP_SUBR_E4 1
#define ENBAEXP_SUBR_RSHFT 1

#include "enbaexp_subr2.inc"

#undef ENBAEXP_E_SUFFIX
#undef ENBAEXP_SUBR_E4
#undef ENBAEXP_SUBR_RSHFT

! ##PASS2  E5 lkup

#define ENBAEXP_E_SUFFIX E5
#define ENBAEXP_SUBR_E5 1
#define ENBAEXP_SUBR_LKUP 1

#include "enbaexp_subr2.inc"

#undef ENBAEXP_E_SUFFIX
#undef ENBAEXP_SUBR_E5
#undef ENBAEXP_SUBR_LKUP

! ##PASS3  E2 cshift

#define ENBAEXP_E_SUFFIX E2
#define ENBAEXP_SUBR_E2 1
#define ENBAEXP_SUBR_CSHIFT 1

#include "enbaexp_subr2.inc"

#undef ENBAEXP_E_SUFFIX
#undef ENBAEXP_SUBR_E2
#undef ENBAEXP_SUBR_CSHIFT

! ##PASS5  E3 cfswit

#define ENBAEXP_E_SUFFIX E3
#define ENBAEXP_SUBR_E3 1
#define ENBAEXP_SUBR_CFSWIT 1

#include "enbaexp_subr2.inc"

#undef ENBAEXP_E_SUFFIX
#undef ENBAEXP_SUBR_E3
#undef ENBAEXP_SUBR_CFSWIT

#if ENBAEXP_SUBR_B1 == 1
! ##PASS4  E8 lewhyb        .when. B1

#define ENBAEXP_E_SUFFIX E8
#define ENBAEXP_SUBR_E8 1
#define ENBAEXP_SUBR_LEWHYB 1

#include "enbaexp_subr2.inc"

#undef ENBAEXP_E_SUFFIX
#undef ENBAEXP_SUBR_E8
#undef ENBAEXP_SUBR_LEWHYB

#endif  /* ENBAEXP_SUBR_B1 */

! ##PASS7  E7 eips

#define ENBAEXP_E_SUFFIX E7
#define ENBAEXP_SUBR_E7 1
#define ENBAEXP_SUBR_EIPS 1

#include "enbaexp_subr2.inc"

#undef ENBAEXP_E_SUFFIX
#undef ENBAEXP_SUBR_E7
#undef ENBAEXP_SUBR_EIPS

! ##PASS8  E9 geswit

#define ENBAEXP_E_SUFFIX E9
#define ENBAEXP_SUBR_E9 1
#define ENBAEXP_SUBR_GESWIT 1

#include "enbaexp_subr2.inc" 

#undef ENBAEXP_E_SUFFIX
#undef ENBAEXP_SUBR_E9
#undef ENBAEXP_SUBR_GESWIT

! ##PASS6  E6 noelec

#define ENBAEXP_E_SUFFIX E6
#define ENBAEXP_SUBR_E6 1
#define ENBAEXP_SUBR_NOELEC 1

#include "enbaexp_subr2.inc"

#undef ENBAEXP_E_SUFFIX
#undef ENBAEXP_SUBR_E6
#undef ENBAEXP_SUBR_NOELEC

! ##PASS8  E1 lewald

#define ENBAEXP_E_SUFFIX E1
#define ENBAEXP_SUBR_E1 1
#define ENBAEXP_SUBR_LEWALD 1

#include "enbaexp_subr2.inc"

#undef ENBAEXP_E_SUFFIX
#undef ENBAEXP_SUBR_E1
#undef ENBAEXP_SUBR_LEWALD

! ##EXEND
! ##ENDEX    (expand_elecopt)

#else  /* KEY_EXPAND */

#define ENBAEXP_E_SUFFIX E0
#define ENBAEXP_SUBR_E0 1

#define ENBAEXP_SUBR_LEWALD 1
#define ENBAEXP_SUBR_CSHIFT 1
#define ENBAEXP_SUBR_CFSWIT 1
#define ENBAEXP_SUBR_RSHFT 1
#define ENBAEXP_SUBR_EIPS 1
#define ENBAEXP_SUBR_NOELEC 1
#define ENBAEXP_SUBR_LKUP 1
#define ENBAEXP_SUBR_LEWHYB 1
#define ENBAEXP_SUBR_GESWIT 1

#include "enbaexp_subr2.inc"

#undef ENBAEXP_E_SUFFIX
#undef ENBAEXP_SUBR_E0

#undef ENBAEXP_SUBR_LEWALD
#undef ENBAEXP_SUBR_CSHIFT
#undef ENBAEXP_SUBR_CFSWIT
#undef ENBAEXP_SUBR_RSHFT
#undef ENBAEXP_SUBR_EIPS
#undef ENBAEXP_SUBR_NOELEC
#undef ENBAEXP_SUBR_LKUP
#undef ENBAEXP_SUBR_LEWHYB
#undef ENBAEXP_SUBR_GESWIT

#endif  /* KEY_EXPAND */
