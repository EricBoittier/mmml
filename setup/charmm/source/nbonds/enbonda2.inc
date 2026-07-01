#if ENBONDA_EXPAND == 1 && KEY_BLOCK == 1

#define ENBONDA_BLOCK 1
  
!-------------------------------------------------------------------
! Do BLOCK expansion of code
! ##EXPAND  B forces       .when. BLOCK  EXPAND  (expand_block)
! ##PASS1   .not.EXPAND .not.forces
  IF(QBLOCK .AND. NOFORC) THEN

#undef ENBONDA_EXPAND
#undef ENBONDA_FORCES

#include "enbonda3.inc"
     
! ##PASS2   .not.EXPAND forces
  ELSE IF(QBLOCK) THEN

#define ENBONDA_FORCES 1
#include "expanda3.inc"

     
! ##PASS3   .not.BLOCK  forces
  ELSE

#undef ENBONDA_BLOCK
#define ENBONDA_EXPAND 1

#include "enbonda3.inc"

     
! ##EXFIN
  ENDIF
! ##EXEND
! ##ENDEX    (expand_block)

#else /* ENBONDA_EXPAND && KEY_BLOCK */

#undef ENBONDA_BLOCK
#if KEY_BLOCK == 1
#define ENBONDA_BLOCK 1
#endif

#define ENBONDA_B_FLAG 1
#define ENBONDA_FORCES 1

#include "enbonda3.inc"

#undef ENBONDA_B_FLAG
#undef ENBONDA_FORCES
  
#endif /* ENBONDA_EXPAND && KEY_BLOCK */
