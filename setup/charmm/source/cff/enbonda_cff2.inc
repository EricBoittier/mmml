  !-------------------------------------------------------------------
  ! Do BLOCK expansion of code

#if ENBONDA_CFF_EXPAND == 1 && KEY_BLOCK == 1  

#undef ENBONDA_CFF_BLOCK
#define ENBONDA_CFF_BLOCK 1

! ##EXPAND  B forces       .when. BLOCK  EXPAND  (expand_block)
! ##PASS1   .not.EXPAND .not.forces
  IF(QBLOCK .AND. NOFORC) THEN

#undef ENBONDA_CFF_EXPAND
#undef ENBONDA_CFF_FORCES
#include "enbonda_cff3.inc"

! ##PASS2   .not.EXPAND forces
  ELSE IF(QBLOCK) THEN

#define ENBONDA_CFF_FORCES 1
#include "enbonda_cff3.inc"
#define ENBONDA_CFF_EXPAND 1

! ##PASS3   .not.BLOCK  forces
  ELSE

#undef ENBONDA_CFF_BLOCK
#include "enbonda_cff3.inc"
#define ENBONDA_CFF_BLOCK 1
     
! ##EXFIN
  ENDIF
! ##EXEND

#else  /* ENBONDA_CFF_EXPAND == 1 && KEY_BLOCK == 1 */

#undef ENBONDA_CFF_BLOCK
#if KEY_BLOCK == 1
#define ENBONDA_CFF_BLOCK 1
#endif

#define ENBONDA_CFF_FORCES 1
#define ENBONDA_CFF_BGUARD 1
#include "enbonda_cff3.inc"
#undef ENBONDA_CFF_FORCES
#undef ENBONDA_CFF_BGUARD
    
#endif  /* ENBONDA_CFF_EXPAND == 1 && KEY_BLOCK == 1 */

! ##ENDEX    (expand_block)
