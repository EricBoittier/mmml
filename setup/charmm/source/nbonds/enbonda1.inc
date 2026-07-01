#if ENBONDA_EXPAND == 1 && KEY_FOURD == 1

#define ENBONDA_FOURD 1
  
!-------------------------------------------------------------------
! Do DIM4 expansion of code
! ##EXPAND  F             .when. FOURD  EXPAND  (expand_dim4)
! ##PASS1   .not.EXPAND
  IF(DIM4) THEN

#undef ENBONDA_EXPAND
#include "enbonda2.inc"
#define ENBONDA_EXPAND 1
     
! ##PASS2   .not.FOURD
  ELSE

#undef ENBONDA_FOURD
#include "enbonda2.inc"
#define ENBONDA_FOURD 1

! ##EXFIN
  ENDIF
! ##EXEND
! ##ENDEX    (expand_dim4)

#else  /* ENBONDA_EXPAND && KEY_FOURD*/

#undef ENBONDA_FOURD
#if KEY_FOURD == 1
#define ENBONDA_FOURD 1
#endif
  
#define ENBONDA_F_FLAG 1
#include "enbonda2.inc"
#undef ENBONDA_F_FLAG
  
#endif /* ENBONDA_EXPAND KEY_FOURD*/
