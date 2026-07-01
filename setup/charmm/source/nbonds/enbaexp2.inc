#if KEY_EXPAND == 1
  !-------------------------------------------------------------------
  ! Do VDWOPT expansion of code
! ##EXPAND V0 vswit vfswit vshft vlkup vips gvswit .when.  EXPAND  (expand_vdwopt)

! ##PASS1 V1 vswit 
  IF (LVSW) THEN

#define ENBAEXP_VDWOPT_SUFFIX V1
#define ENBAEXP_V1 1
#define ENBAEXP_VSWIT 1
#include "enbaexp_expand.inc"
#undef ENBAEXP_VDWOPT_SUFFIX
#undef ENBAEXP_V1
#undef ENBAEXP_VSWIT

! ##PASS2  V4 vlkup .when. E5
#if ENBAEXP_E0 == 1
  ELSE IF (LNBLKUP) THEN

#define ENBAEXP_VDWOPT_SUFFIX V4
#define ENBAEXP_V4 1
#define ENBAEXP_VLKUP 1
#include "enbaexp_expand.inc"
#undef ENBAEXP_VDWOPT_SUFFIX
#undef ENBAEXP_V4
#undef ENBAEXP_VLKUP

#endif  /* ENBAEXP_E0 == 1 */

! ##PASS3  V2 vfswit 
  ELSE IF (LVFSW) THEN

#define ENBAEXP_VDWOPT_SUFFIX V2
#define ENBAEXP_V2 1
#define ENBAEXP_VFSWIT 1
#include "enbaexp_expand.inc"
#undef ENBAEXP_VDWOPT_SUFFIX
#undef ENBAEXP_V2
#undef ENBAEXP_VFSWIT

! ##PASS4  V3 vshft 
  ELSE IF (LVSH) THEN

#define ENBAEXP_VDWOPT_SUFFIX V3
#define ENBAEXP_V3 1
#define ENBAEXP_VSHFT 1
#include "enbaexp_expand.inc"
#undef ENBAEXP_VDWOPT_SUFFIX
#undef ENBAEXP_V3
#undef ENBAEXP_VSHFT

! ##PASS5A  V5A dbexp
  ELSE IF (DBEXP) THEN

#define ENBAEXP_VDWOPT_SUFFIX V5A
#define ENBAEXP_V5A 1
#define ENBAEXP_DBEXP 1
#include "enbaexp_expand.inc"
#undef ENBAEXP_VDWOPT_SUFFIX
#undef ENBAEXP_V5A
#undef ENBAEXP_DBEXP

! ##PASS5  V5 vips 
  ELSE IF (LVIPSX) THEN

#define ENBAEXP_VDWOPT_SUFFIX V5
#define ENBAEXP_V5 1
#define ENBAEXP_VIPS 1
#include "enbaexp_expand.inc"
#undef ENBAEXP_VDWOPT_SUFFIX
#undef ENBAEXP_V5
#undef ENBAEXP_VIPS

! ##PASS5B  V5B dbexpips 
  ELSE IF (DBEXPIPS) THEN

#define ENBAEXP_VDWOPT_SUFFIX V5B
#define ENBAEXP_V5B 1
#define ENBAEXP_DBEXPIPS 1
#include "enbaexp_expand.inc"
#undef ENBAEXP_VDWOPT_SUFFIX
#undef ENBAEXP_V5B
#undef ENBAEXP_DBEXPIPS

! ##PASS6  V6 gvswit
  ELSE IF (GVSWIT) THEN

#define ENBAEXP_VDWOPT_SUFFIX V6
#define ENBAEXP_V6 1
#define ENBAEXP_GVSWIT 1
#include "enbaexp_expand.inc"
#undef ENBAEXP_VDWOPT_SUFFIX
#undef ENBAEXP_V6
#undef ENBAEXP_GVSWIT

! ##EXFIN
  ELSE
     CALL WRNDIE(-4,'<ENBAEXP>','Bad vdw option')
  ENDIF
! ##EXEND
! ##ENDEX    (expand_vdwopt)

#else  /* KEY_EXPAND */

#define ENBAEXP_VDWOPT_SUFFIX V0
#define ENBAEXP_V0 1

#define ENBAEXP_VSWIT 1
#define ENBAEXP_VFSWIT 1
#define ENBAEXP_VSHFT 1
#define ENBAEXP_VLKUP 1
#define ENBAEXP_VIPS 1
#define ENBAEXP_GVSWIT 1

#include "enbaexp_expand.inc"

#undef ENBAEXP_VDWOPT_SUFFIX
#undef ENBAEXP_V0
#undef ENBAEXP_VSWIT
#undef ENBAEXP_VFSWIT
#undef ENBAEXP_VSHFT
#undef ENBAEXP_VLKUP
#undef ENBAEXP_VIPS
#undef ENBAEXP_GVSWIT

#endif  /* KEY_EXPAND */
