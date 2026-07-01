  !-------------------------------------------------------------------
  ! Do DIM4 expansion of code

#if ENBONDA_CFF_EXPAND == 1 && KEY_FOURD == 1 

#undef ENBONDA_CFF_FOURD
#define ENBONDA_CFF_FOURD 1  

  ! ##EXPAND  F             .when. FOURD  EXPAND  (expand_dim4)
  ! ##PASS1   .not.EXPAND
  IF(DIM4) THEN
   
#undef ENBONDA_CFF_EXPAND
#include "enbonda_cff2.inc"
#define ENBONDA_CFF_EXPAND 1

! ##PASS2   .not.FOURD
  ELSE

#undef ENBONDA_CFF_FOURD
#include "enbonda_cff2.inc"
#define ENBONDA_CFF_FOURD 1
     
! ##EXFIN
  ENDIF
! ##EXEND

#else /* ENBONDA_CFF_EXPAND && KEY_FOURD */

#undef ENBONDA_CFF_FOURD
#if KEY_FOURD == 1
#define ENBONDA_CFF_FOURD 1
#endif

#define ENBONDA_CFF_FGUARD 1
#include "enbonda_cff2.inc"
#undef ENBONDA_CFF_FGUARD
  
#endif /* ENBONDA_CFF_EXPAND && KEY_FOURD */
! ##ENDEX    (expand_dim4)
