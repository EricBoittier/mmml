#if KEY_EXPAND == 1 && KEY_BLOCK == 1

! Do BLOCK expansion of code
! ##EXPAND  B forces       .when. BLOCK  EXPAND  (expand_block)
! ##PASS1   .not.EXPAND .not.forces
      IF(QBLOCK .AND. NOFORC) THEN

#undef KEY_EXPAND
#include "enbondg2.inc"

! ##PASS2   .not.EXPAND forces
      ELSE IF(QBLOCK) THEN

#define ENBONDG_FORCES 1
#include "enbondg2.inc"
#define KEY_EXPAND 1
         
! ##PASS3   .not.BLOCK  forces
      ELSE

#undef KEY_BLOCK
#include "enbondg2.inc"
#define KEY_BLOCK 1
#undef ENBONDG_FORCES
         
! ##EXFIN
      ENDIF
! ##EXEND
! ##ENDEX    (expand_block)

#else /* KEY_EXPAND && KEY_BLOCK */

#define ENBONDG_B_FLAG 1
#define ENBONDG_FORCES 1
      
#include "enbondg2.inc"

#undef ENBONDG_B_FLAG
#undef ENBONDG_FORCES
      
#endif /* KEY_EXPAND && KEY_BLOCK */
