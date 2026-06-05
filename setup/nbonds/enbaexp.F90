! Expanded fast atom list nonbond routine
!
!CC##SET .not.EXPAND
!
!-----------------------------------------------------------------------
SUBROUTINE ENBAEXP(ENB,EEL,LELECX,LVDWX,IFRSTA,NATOMX, &
     CGX,JNBL,INBL, &
     IACNBX, &
     LUSED) 
  !-----------------------------------------------------------------------
  !     Calculate nonbonded interaction energies and forces.
  !
  !     ENB     <- calculated vdw interaction energy
  !     EEL     <- calculated electrostatic interaction energy
  !     LELECX  -> compute electrostatic energy?
  !     LVDWX   -> compute van der Waals energy?
  !     IFIRSTA -> atom index to start main loop with
  !     NATOMX  -> number of atoms
  !     CGX     -> charges
  !     JNBL    -> nonbond atom pair list for index j
  !     INBL    -> nonbond atom pair list for index i
  !     CCNBA, CCNBB, CCNBC, CCNBD -> vdw paramter arrays
  !     IACNBX  -> vdw type array
  !     ERFCT   -> ERFC lookup table
  !     LUSED   <- was anything calculated here?
  !
  !----------------------------------------------------------------------
  !     This is the expanded fast scalar version of the nonboned energy
  !     LFST=0 or LFAST=2 is required to use this routine.
  !                       By Bernard Brooks, June 2004, NIH           
  !
  ! Restricted development:
  !  To ensure optimal performance, new features may be added to this
  !  routine only with concensus from the CHARMM community. - BRB
  !
  !-----------------------------------------------------------------------
  use ewald_1m,only:lewald,erfmod
  use nb_module  ! has ccnba thru d
  
  use chm_kinds
  use consta
  use dimens_fcm
  use number
  use coord
  use deriv
  use param
  use inbnd
#if KEY_BLOCK==1
  use block_fcm           
#endif
  use galgor
  use stream
  implicit none
  real(chm_real)  ENB,EEL
  LOGICAL LELECX,LVDWX
  INTEGER IFRSTA,NATOMX,JNBL(*),INBL(*)
  real(chm_real)  CGX(*)
  INTEGER IACNBX(*)
  LOGICAL LUSED
  !
  LOGICAL ELECFG,RSHFT,RSWIT,CSHIFT,CFSWIT,CSHFT,CSWIT,RSHIFT,RFSWIT
  LOGICAL LEWLD,LVFSW,LVSH,LVSW,NOEWLD         ! BRB fix 27-Sep-2004
  LOGICAL LEIPSX,LVIPSX,REIPS,CEIPS
  LOGICAL GESWIT,GVSWIT 
  INTEGER FIRST
  INTEGER I,NPR
  
  LOGICAL DBEXP, DBEXPIPS

  !---------- Sanity check -------------------------------------
  if(.not. allocated(ccnba))then
     ! How we got here without vdw table filled, who knows?
     call wrndie(-4,"ENBAEXP<enbaexp.src>", &
          "CCNBA not allocated")
  endif
  !
  ! Set flags for electrostatics options (3 options supported at present)
  ELECFG = (LELECX.AND.(EPS /= 0.0))
  LEWLD = LEWALD .AND. ELECFG
  NOEWLD = ELECFG .AND. .NOT.LEWALD
  RSHIFT= .NOT.LCONS .AND.      LSHFT .AND.      LFSWT .AND. NOEWLD &
          .AND. .NOT. LEGROM
  RFSWIT= .NOT.LCONS .AND. .NOT.LSHFT .AND.      LFSWT .AND. NOEWLD &
          .AND. .NOT. LEGROM
  RSHFT = .NOT.LCONS .AND.      LSHFT .AND. .NOT.LFSWT .AND. NOEWLD &
          .AND. .NOT. LEGROM
  RSWIT = .NOT.LCONS .AND. .NOT.LSHFT .AND. .NOT.LFSWT .AND. NOEWLD &
          .AND. .NOT. LEGROM
  CSHIFT=      LCONS .AND.      LSHFT .AND.      LFSWT .AND. NOEWLD &
          .AND. .NOT. LEGROM
  CFSWIT=      LCONS .AND. .NOT.LSHFT .AND.      LFSWT .AND. NOEWLD &
          .AND. .NOT. LEGROM
  CSHFT =      LCONS .AND.      LSHFT .AND. .NOT.LFSWT .AND. NOEWLD &
          .AND. .NOT. LEGROM
  CSWIT =      LCONS .AND. .NOT.LSHFT .AND. .NOT.LFSWT .AND. NOEWLD &
          .AND. .NOT. LEGROM
  GESWIT=LEGROM
  !
  LVFSW=      LVFSWT                   .AND. LVDWX &
       .AND. .NOT. LEGROM .AND. .NOT. LDBEXP
  LVSH = .NOT.LVFSWT .AND.      LVSHFT .AND. LVDWX &
       .AND. .NOT. LEGROM .AND. .NOT. LDBEXP
  LVSW = .NOT.LVFSWT .AND. .NOT.LVSHFT .AND. LVDWX &
       .AND. .NOT. LEGROM .AND. .NOT. LDBEXP
  GVSWIT=LVGROM
  DBEXP=LDBEXP
  !
#if KEY_NBIPS==1
  LEIPSX = LEIPS .AND. ELECFG .AND. .NOT.LEWALD
  CEIPS =  LEIPSX .AND.  LCONS
  REIPS =  LEIPSX .AND.  .NOT. LCONS
  IF(LEIPSX)THEN
     RSHIFT= .FALSE.
     RFSWIT= .FALSE.
     RSHFT = .FALSE.
     RSWIT = .FALSE.
     CSHIFT= .FALSE.
     CFSWIT= .FALSE.
     CSHFT = .FALSE.
     CSWIT = .FALSE.
     GESWIT= .FALSE.
  ENDIF
  !
  DBEXPIPS=LDBEXP .AND. LVIPS .AND. LVDWX
  IF(DBEXPIPS)DBEXP=.FALSE.
  LVIPSX = LVIPS .AND. LVDWX .AND. .NOT. LDBEXP
  IF(LVIPSX.OR.DBEXP.OR.DBEXPIPS)THEN
     LVFSW= .FALSE.
     LVSH = .FALSE.
     LVSW = .FALSE.
     GVSWIT=.FALSE.
  ENDIF
#endif 
  !
  ! check to see if we should be here...
  LUSED = .FALSE.
  IF(LGROUP) RETURN
  ! reject unsupported electrostatic options
  IF(LFMA) RETURN
  IF(LEWLD.AND.ERFMOD >= 0) RETURN
  IF(RSHIFT .OR. RFSWIT .OR. RSWIT) RETURN
  IF(CSHFT .OR. CSWIT) RETURN
  IF(.NOT.LVDWX) RETURN
#if KEY_NBIPS==1
  IF(REIPS) RETURN            
#endif
  IF(LEGROM.AND.((.NOT.LCONS).OR.LSHFT.OR.LFSWT)) RETURN
  IF(LVGROM.AND.(LVSHFT.OR.LVFSWT)) RETURN
  !
  LUSED = .TRUE.
  !
  FIRST = IFRSTA
#if KEY_GENETIC==1
  If(qGA_Ener) then
     FIRST = INT(ENB)
  endif
#endif 

#if KEY_EXPAND == 1 && KEY_BLOCK == 1
  !=======================================================================
  !  Expand control section
  !-------------------------------------------------------------------
  ! Do BLOCK expansion of code
! ##EXPAND  B0  .when. BLOCK  EXPAND  (expand_block)

#define ENBAEXP_BLOCK 1

! ##PASS1   B1
  IF(QBLOCK) THEN

#define ENBAEXP_BLOCK_SUFFIX B1
#define ENBAEXP_B1 1
#include "enbaexp1.inc"
#undef ENBAEXP_BLOCK_SUFFIX
#undef ENBAEXP_B1

! ##PASS2   B2  .not.BLOCK
  ELSE

#undef ENBAEXP_BLOCK
#define ENBAEXP_BLOCK_SUFFIX B2
#define ENBAEXP_B2 1
#include "enbaexp1.inc"
#undef ENBAEXP_BLOCK_SUFFIX
#undef ENBAEXP_B2

! ##EXFIN
  ENDIF 
! ##EXEND
! ##ENDEX    (expand_block)

#else  /* KEY_EXPAND == 1 && KEY_BLOCK == 1 */

#if KEY_BLOCK == 1
#define ENBAEXP_BLOCK 1
#else
#define ENBAEXP_BLOCK 0
#endif

#define ENBAEXP_BLOCK_SUFFIX B0
#define ENBAEXP_B0 1
#include "enbaexp1.inc"

#endif  /* KEY_EXPAND == 1 && KEY_BLOCK == 1 */

125 FORMAT(' ENBAEXP: Using routine ',A,' for energy calculation.')
  RETURN
END SUBROUTINE ENBAEXP

!  Expand routine section
#if KEY_EXPAND == 1 && KEY_BLOCK ==1 

! Do BLOCK expansion of code
! ##EXPAND  B0  .when. BLOCK  EXPAND  (expand_block)

#define ENBAEXP_SUBR_BLOCK 1

! ##PASS1   B1

#define ENBAEXP_B_SUFFIX B1
#define ENBAEXP_SUBR_B1 1
#include "enbaexp_subr1.inc"
#undef ENBAEXP_B_SUFFIX
#undef ENBAEXP_SUBR_B1

! ##PASS2   B2  .not.BLOCK

#undef ENBAEXP_SUBR_BLOCK
#define ENBAEXP_B_SUFFIX B2
#define ENBAEXP_SUBR_B2 1
#include "enbaexp_subr1.inc"
#undef ENBAEXP_B_SUFFIX
#undef ENBAEXP_SUBR_B2

! ##EXEND
! ##ENDEX    (expand_block)

#else  /* KEY_EXPAND == 1 && KEY_BLOCK ==1 */

#if KEY_BLOCK == 1
#define ENBAEXP_SUBR_BLOCK 1
#else  /* KEY_BLOCK */
#define ENBAEXP_SUBR_BLOCK 0
#endif  /* KEY_BLOCK */

#define ENBAEXP_B_SUFFIX B0
#define ENBAEXP_SUBR_B0 1
#include "enbaexp_subr1.inc"

#endif  /* KEY_EXPAND == 1 && KEY_BLOCK ==1 */

SUBROUTINE NULL_ENBAEXP
  RETURN
END SUBROUTINE NULL_ENBAEXP
