#if KEY_EXPAND == 1
! Do VDWOPT expansion of code
! ##EXPAND V0 vswit vfswit vshft vlkup vips gvswit .when.  EXPAND  (expand_vdwopt)

! ##PASS1 V1 vswit 

#define ENBAEXP_V_SUFFIX V1
#define ENBAEXP_SUBR_V1 1
#define ENBAEXP_SUBR_VSWIT 1

#include "enbaexp_subr_expand.inc"

#undef ENBAEXP_V_SUFFIX
#undef ENBAEXP_SUBR_VSWIT
#undef ENBAEXP_SUBR_V1

#if ENBAEXP_SUBR_E5 == 1

! ##PASS2 V4 vlkup         .when. E5 

#define ENBAEXP_V_SUFFIX V4
#define ENBAEXP_SUBR_V4 1
#define ENBAEXP_SUBR_VLKUP 1

#include "enbaexp_subr_expand.inc"

#undef ENBAEXP_V_SUFFIX
#undef ENBAEXP_SUBR_VLKUP
#undef ENBAEXP_SUBR_V4

#endif  /* ENBAEXP_SUBR_E5 */

! ##PASS3 V2 vfswit 

#define ENBAEXP_V_SUFFIX V2
#define ENBAEXP_SUBR_V2 1
#define ENBAEXP_SUBR_VFSWIT 1

#include "enbaexp_subr_expand.inc"

#undef ENBAEXP_V_SUFFIX
#undef ENBAEXP_SUBR_VFSWIT
#undef ENBAEXP_SUBR_V2

! ##PASS4 V3 vshft 

#define ENBAEXP_V_SUFFIX V3
#define ENBAEXP_SUBR_V3 1
#define ENBAEXP_SUBR_VSHFT 1

#include "enbaexp_subr_expand.inc"

#undef ENBAEXP_V_SUFFIX
#undef ENBAEXP_SUBR_VSHFT
#undef ENBAEXP_SUBR_V3

! ##PASS5 V5A dbexp

#define ENBAEXP_V_SUFFIX V5A
#define ENBAEXP_SUBR_V5A 1
#define ENBAEXP_SUBR_DBEXP 1

#include "enbaexp_subr_expand.inc"

#undef ENBAEXP_V_SUFFIX
#undef ENBAEXP_SUBR_DBEXP
#undef ENBAEXP_SUBR_V5A


! ##PASS5 V5 vips 

#define ENBAEXP_V_SUFFIX V5
#define ENBAEXP_SUBR_V5 1
#define ENBAEXP_SUBR_VIPS 1

#include "enbaexp_subr_expand.inc"

#undef ENBAEXP_V_SUFFIX
#undef ENBAEXP_SUBR_VIPS
#undef ENBAEXP_SUBR_V5

! ##PASS5 V5b dbexpips 

#define ENBAEXP_V_SUFFIX V5B
#define ENBAEXP_SUBR_V5B 1
#define ENBAEXP_SUBR_DBEXPIPS 1

#include "enbaexp_subr_expand.inc"

#undef ENBAEXP_V_SUFFIX
#undef ENBAEXP_SUBR_DBEXPIPS
#undef ENBAEXP_SUBR_V5B

! ##PASS6 V6 gvswit

#define ENBAEXP_V_SUFFIX V6
#define ENBAEXP_SUBR_V6 1
#define ENBAEXP_SUBR_GVSWIT 1

#include "enbaexp_subr_expand.inc"

#undef ENBAEXP_V_SUFFIX
#undef ENBAEXP_SUBR_GVSWIT
#undef ENBAEXP_SUBR_V6

! ##EXEND
! ##ENDEX    (expand_vdwopt)

#else  /* KEY_EXPAND */

#define ENBAEXP_V_SUFFIX V0

#define ENBAEXP_SUBR_VSWIT 1
#define ENBAEXP_SUBR_VFSWIT 1
#define ENBAEXP_SUBR_VSHFT 1
#define ENBAEXP_SUBR_VLKUP 1
#define ENBAEXP_SUBR_DBEXP 1
#define ENBAEXP_SUBR_VIPS 1
#define ENBAEXP_SUBR_V5 1
#define ENBAEXP_SUBR_DBEXPIPS 1
#define ENBAEXP_SUBR_V5B 1
#define ENBAEXP_SUBR_GVSWIT 1

#include "enbaexp_subr_expand.inc"

#undef ENBAEXP_V_SUFFIX
#undef ENBAEXP_SUBR_V0
#undef ENBAEXP_SUBR_VSWIT
#undef ENBAEXP_SUBR_VFSWIT
#undef ENBAEXP_SUBR_VSHFT
#undef ENBAEXP_SUBR_VLKUP
#undef ENBAEXP_SUBR_DBEXP
#undef ENBAEXP_SUBR_VIPS
#undef ENBAEXP_SUBR_V5
#undef ENBAEXP_SUBR_DBEXPIPS
#undef ENBAEXP_SUBR_V5B
#undef ENBAEXP_SUBR_GVSWIT

#endif  /* KEY_EXPAND */
