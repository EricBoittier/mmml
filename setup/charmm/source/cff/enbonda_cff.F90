#if KEY_CFF==1
SUBROUTINE EVDW_CFF(ENB,EEL,NATOM,JNB,INBLO,CG,RSCLF,CNBA,CNBB, &
     MAXROW,IAC,ITC,NATC,IOFF,LELEC,LVDW,LCONS,LSHFT,LVSHFT,LFSWT, &
     LVFSWT,DX,DY,DZ,X,Y,Z,CTONNB,CTOFNB,EPS,E14FAC, &
#if KEY_FLUCQ==1
     QFLUC,FQCFOR,   & 
#endif
     QECONTX,ECONTX,DD1,IUPT, &
     QSECD)
  !-----------------------------------------------------------------------
  !     THIS ROUTINE CALCULATES NON BONDED INTERACTION ENERGIES AND FORCES
  !
  !     ENB    - vdw energy returned
  !     EEL    - electrostatic energy returned
  !     NATOM  - number of atoms
  !     JNB    - nonbond pair list  (INBLO(NATOM))
  !     INBLO  - pointers into JNB  (NATOM)
  !     CG     - charges  (NATOM)
  !     RSCLF  - Radius Scaling Factor (NATOM)
  !     CNBA   - vdw well distance squared (MAXROW*2)
  !     CNBB   - vdw well depths (MAXROW*2)
  !     MAXROW  - offset for 1-4 interaction in CNBA and CNBB
  !     LELEC,,,,,,,, - logical flags used in BNBND.FCM
  !     ITC(IAC(J))  - lookup pointers into CNBA and CNBB  (NATOM)
  !     CTONNB,CTOFNB - switching function specifiers in real space
  !     EPS - dielectric constant
  !     DD1,QSECD - second derivative specifiers
  !
  !     By Bernard R. Brooks   5/7/81
  !
  !     Include BLOCK energy partition
  !     By Youngdo Won         12/15/90
  !     March, 2008       Updated l-dynamics variables for dma. JLK
  !----------------------------------------------------------------------

  use ewald_1m,only:kappa,lewald,erfmod
  use erfcd_mod,only: erfcd
  use chm_kinds
  use dimens_fcm
  use number
  use econtmod
  use euler
  use fourdm
  use stream
  use consta
  use cff_fcm
#if KEY_PBOUND==1
  use pbound         
#endif
#if KEY_BLOCK==1
  use block_fcm          
  use lambdam         
#endif
#if KEY_DIMB==1
  use dimb         
#endif
  use chutil,only:atomid
  use machutil,only:die

  implicit none
  real(chm_real) ENB,EEL
  INTEGER NATOM
  INTEGER JNB(*)
  INTEGER INBLO(*)
  real(chm_real) CG(*),RSCLF(*),CNBA(*),CNBB(*)
  INTEGER MAXROW
  INTEGER IAC(*),ITC(*)
  INTEGER NATC,IOFF(*)
  LOGICAL LELEC,LVDW,LCONS,LSHFT,LVSHFT,LFSWT,LVFSWT
  real(chm_real) DX(*),DY(*),DZ(*),X(*),Y(*),Z(*)
  real(chm_real) CTONNB,CTOFNB,EPS,E14FAC
  LOGICAL QECONTX
  real(chm_real) ECONTX(*)
  real(chm_real) DD1(*)
  INTEGER IUPT(*)
#if KEY_FLUCQ==1
  LOGICAL QFLUC
  real(chm_real) FQCFOR(*)
#endif 
  LOGICAL QSECD
  !

  !
  real(chm_real) ETEMP1,ETEMP2,ENBPR,EELPR
  real(chm_real) C2OFNB,C4ROF,C2ROF2,CHROF2,C2ONNB
  real(chm_real) RUL3,RUL12,SGSHSQ,ASH6,BSH6
  real(chm_real) CGF,CGT,RS,CGIJ
  real(chm_real) FDXI,FDYI,FDZI,CRXI,CRYI,CRZI,CGT2,DXI,DYI,DZI
  real(chm_real) S,SIG2,R2,SIG6,SIG9,RIJL,RIJU,FUNCT,DFN
  real(chm_real) EN,DEN,DF,DDF,G2,R1,G1,G3,DXIT,DYIT,DZIT
  real(chm_real) AXX,AYY,AZZ,AXY,AXZ,AYZ
  real(chm_real) EADD,EADDR,ON3,ON6,ONOFF2,OFF3,OFF4,OFF5,OFF6, &
       RMIN2,RMIN6, &
       R3,R4,R6,R8,DENOM,ACOEF,BCOEF,CCOEF,DCOEF,TWOA, &
       TWOB,TWOC,TWOD,FOURDL,COVER3,DOVER5,SIXA,SIXD,CONST,CONSTR
  real(chm_real) RECOF3,RECOF6,OFDIF3,OFDIF6,ONOFF3,ONOFF6, &
       R5,CR6,CR12,RJUNK3,RJUNK6,RECOF2,MIN2OF
  INTEGER I,NB,IADD,ITEMP,J,NPR,I1,IACI,JPR,J1,IC,II,JJ,KK
  LOGICAL QAFIRST
  CHARACTER(len=8) SIDDNI,RIDDNI,RESDNI,ACDNI,SIDDNJ,RIDDNJ,RESDNJ,ACDNJ

  !  for ewald summation
  real(chm_real)   DFRS,E14M1,E14F
  real(chm_real)   ERFC2,ERFCX,DERFC

#if KEY_PBOUND==1
  real(chm_real) CORR            
#endif
#if KEY_FOURD==1
  !  4-D variable:
  real(chm_real)   CRFDIMI,DFDIMI
  real(chm_real)   FDFDI,DFDIT
#endif 
  !
  !
  !     QSECD  - NO SECOND DERIVATIVES CALCULATED
  !     ELECFG - DO ELECTROSTATICS HERE
  !     VDWFG  - DO VDW INTERACTION
  !
  !     RSHFT  - DO R-DIELECTRIC WITH SHIFTED POTENTIAL
  !     RSWIT  - DO R-DIELECTRIC WITH SWITCHING FUNCTIONS (DEFAULT)
  !     RFSWIT - DO R-DIELECTRIC WITH FORCE SWITCHING
  !     CSHFT  - DO CONSTANT DIEL. WITH SHIFTED POTENTIAL
  !     CSWIT  - DO CONSTANT DIEL. WITH SWITCHING FUNCTIONS
  !     CSHIFT - DO CONSTANT DIEL. WITH SHIFT: S(r) = (1-r/roff)**2
  !     CFSWIT - DO CONSTANT DIEL. WITH FORCE SWITCHING
  !
  !     DSHFT - VDW DISTANCE SHIFTING
  !     DSWIT - VDW DISTANCE SWITCHING (DEFAULT)
  !
  !     EWALD - EWALD SUMMATION USED
  !
  LOGICAL RSHIFT,RSHFT,RSWIT,RFSWIT,CSHFT,CSWIT,CFSWIT,CSHIFT
  LOGICAL LSECD,ELECFG
  LOGICAL VDWFG,DSWIT,DSHFT,DFSWIT,ELCFG,LUSED
  !
#if KEY_BLOCK==1
  INTEGER IBL,JBL
  real(chm_real)  COEF
#if KEY_DOCK==1
  INTEGER KDOC
  real(chm_real)  DOCFI, DOCFJ
#endif /*  DOCK*/
  real(chm_real) FALPHA
#endif /*  BLOCK*/
  !
#if KEY_FLUCQ==1
  IF(QFLUC) CALL WRNDIE(-4,'<EVDW_CFF>', &
       'No FlucQ implementation for CFF yet')
#endif 
  LSECD = QSECD
  E14M1 = E14FAC-ONE
  ELECFG=LELEC.AND.(EPS /= ZERO)
  ! Set flags for electrostatics options (8 possibilities)
  IF(.NOT.ELECFG .OR. LEWALD) THEN
     RSHFT= .FALSE.
     RSWIT= .FALSE.
     CSWIT= .FALSE.
     CSHFT= .FALSE.
     RSHIFT=.FALSE.
     RFSWIT=.FALSE.
     CSHIFT=.FALSE.
     CFSWIT=.FALSE.
  ELSE
     RSHIFT= .NOT.LCONS .AND.      LSHFT .AND.      LFSWT
     RFSWIT= .NOT.LCONS .AND. .NOT.LSHFT .AND.      LFSWT
     RSHFT = .NOT.LCONS .AND.      LSHFT .AND. .NOT.LFSWT
     RSWIT = .NOT.LCONS .AND. .NOT.LSHFT .AND. .NOT.LFSWT
     CSHIFT=      LCONS .AND.      LSHFT .AND.      LFSWT
     CFSWIT=      LCONS .AND. .NOT.LSHFT .AND.      LFSWT
     CSHFT =      LCONS .AND.      LSHFT .AND. .NOT.LFSWT
     CSWIT =      LCONS .AND. .NOT.LSHFT .AND. .NOT.LFSWT
  ENDIF
  !
  IF(LEWALD .AND. LSECD) CALL WRNDIE(-4,'<EVDW>', &
       'Second derivatives for Ewald have not as yet been implemented.')
  !
  ELCFG=.FALSE.
  QAFIRST=.TRUE.
  !
  ! Set flags for van der Waals options
  VDWFG=LVDW
  IF (VDWFG) THEN
     DFSWIT = LVFSWT
     DSHFT = .NOT.LVFSWT .AND.      LVSHFT
     DSWIT = .NOT.LVFSWT .AND. .NOT.LVSHFT
  ELSE
     DFSWIT = .FALSE.
     DSWIT  = .FALSE.
     DSHFT  = .FALSE.
  ENDIF
  !
  IF(DSHFT) CALL WRNDIE(-4,'<EVDW_CFF>', &
       'VDW distance shifting has not yet been implemented for CFF.')
  !
  IF(DFSWIT) CALL WRNDIE(-4,'<EVDW_CFF>', &
       'VDW force based switching has not yet been implemented for CFF.')
  !
  IF (.NOT.(ELECFG.OR.VDWFG)) RETURN
  !
  NB=0
  !
  C2ONNB=CTONNB*CTONNB
  C2OFNB=CTOFNB*CTOFNB
  !
  IF(RSHIFT) THEN
     CALL WRNDIE(-5,'<EVDW>','FSHIft not supported with RDIE')
     RETURN
  ELSE IF (CFSWIT.OR.RFSWIT) THEN
     !       force-based electrostatic switching coeffs
     IF(CTONNB  <  CTOFNB) THEN
        ONOFF2 = C2ONNB*C2OFNB
        ON3    = C2ONNB*CTONNB
        OFF3   = C2OFNB*CTOFNB
        OFF4   = C2OFNB*C2OFNB
        OFF5   = OFF3*C2OFNB
        DENOM  = ONE/(C2OFNB-C2ONNB)**3
        EADD   = (ONOFF2*(CTOFNB-CTONNB)-(OFF5-ON3*C2ONNB)/FIVE)* &
             EIGHT*DENOM
        ACOEF  = OFF4*(C2OFNB-THREE*C2ONNB)*DENOM
        BCOEF  = SIX*ONOFF2*DENOM
        COVER3 = -(C2ONNB+C2OFNB)*DENOM
        CCOEF  = THREE*COVER3
        DCOEF  = TWO*DENOM
        TWOA   = TWO*ACOEF
        TWOC   = TWO*CCOEF
        FOURDL  = FOUR*DCOEF
        DOVER5 = DCOEF/FIVE
        CONST  = BCOEF*CTOFNB-ACOEF/CTOFNB+COVER3*OFF3+DOVER5*OFF5
        !         rdie
        IF(CTONNB < RSMALL) CALL WRNDIE(-3,'<EVDW>', &
             'Bad CTONNB value specified')
        EADDR  = (TWELVE*ONOFF2*LOG(CTOFNB/CTONNB) - &
             THREE*(OFF4-C2ONNB*C2ONNB))*DENOM
        TWOB   = TWO*BCOEF
        TWOD   = TWO*DCOEF
        SIXA   = SIX*ACOEF
        SIXD   = SIX*DCOEF
        CONSTR=TWOB*LOG(CTOFNB)-ACOEF/C2OFNB+CCOEF*C2OFNB+OFF4*DENOM
     ELSE
        EADD  = -ONE/CTOFNB
        EADDR = -ONE/C2OFNB
     ENDIF
     !
  ELSE IF (LSHFT) THEN
     !     SHIFTED DIELECTRIC COEFFICIENTS
     RECOF2 = ONE/C2OFNB
     MIN2OF = MINTWO/CTOFNB
     C4ROF=ONE/(C2OFNB*C2OFNB)
     C2ROF2=MINTWO/C2OFNB
     CHROF2=-HALF/C2OFNB
  ELSE
     !     SWITCHING ELECTROSTATIC OPTIONS
     IF (CTOFNB > CTONNB) THEN
        RUL3=ONE/(C2OFNB-C2ONNB)**3
        RUL12=RUL3*TWELVE
     ENDIF
  ENDIF
  !
  IF (DSWIT) THEN
     !     VAN DER WAAL DISTANCE SWITCHING COEFFICIENTS
     IF (CTOFNB > CTONNB) THEN
        RUL3=ONE/(C2OFNB-C2ONNB)**3
        RUL12=RUL3*TWELVE
     ENDIF
  ELSE IF (DFSWIT) THEN
     !     vdw force-switching coefficients
     OFF3 = C2OFNB*CTOFNB
     OFF6 = OFF3*OFF3
     RECOF6 = ONE/OFF6
     IF(CTONNB  <  CTOFNB) THEN
        ON3 = C2ONNB*CTONNB
        ON6 = ON3*ON3
        RECOF3 = ONE/OFF3
        OFDIF6 = OFF6/(OFF6 - ON6)
        OFDIF3 = TWO*OFF3/(OFF3 - ON3)
        ONOFF6 = RECOF6/ON6
        ONOFF3 = TWO*RECOF3/ON3
     ELSE
        ONOFF6 = RECOF6*RECOF6
        ONOFF3 = TWO*RECOF6
     ENDIF
  ELSE IF (DSHFT) THEN
     !     VAN DER WAAL DISTANCE SHIFTING
     RECOF2 = ONE/C2OFNB
  ELSE
     IF(VDWFG) CALL DIE
  ENDIF
  !
#if KEY_FOURD==1
  DFDIMI=ZERO
#endif 
  ITEMP=0
  ENB=0.0
  EEL=0.0
  IF(ELECFG) CGF=CCELEC/EPS
  !
  !     Initialize the code look up offsets
  !
  J=0
  DO I=1,NATC
     IOFF(I)=J
     J=J+I
  ENDDO
  !
  !     DO VDW AND ELECTROSTATIC TERMS AS REQUESTED
  !
  !=======================================================================
  !  Expand control section
  !-------------------------------------------------------------------
  ! (disable expand when debug is active)
#if KEY_EXPAND == 1
#define ENBONDA_CFF_EXPAND 1
#endif
  
#if KEY_DEBUG == 1
#undef ENBONDA_CFF_EXPAND
#endif

  !-------------------------------------------------------------------
  ! Do LSECD expansion of code
! ##EXPAND  lsecd analys  .when.    EXPAND  (expand_lsecd)
! ##PASS1   lsecd analys .not.EXPAND
#if ENBONDA_CFF_EXPAND == 1
  IF(LSECD .OR. QATERM .OR. QECONTX) THEN

#undef ENBONDA_CFF_EXPAND
#define ENBONDA_CFF_LSECD 1
#define ENBONDA_CFF_ANALYS 1
#include "enbonda_cff1.inc"
#define ENBONDA_CFF_EXPAND 1
#undef ENBONDA_CFF_LSECD
#undef ENBONDA_CFF_ANALYS

! ##PASS2
  ELSE
     
#include "enbonda_cff1.inc"
     
! ##EXFIN
  ENDIF
  
#else  /* ENBONDA_CFF_EXPAND */

#define ENBONDA_CFF_LSECD 1
#define ENBONDA_CFF_ANALYS 1
#include "enbonda_cff1.inc"

#endif  /* ENBONDA_CFF_EXPAND */

! ##ENDEX    (expand_lsecd)
  !=======================================================================
  !
  ! Format statements
243 FORMAT('ANAL:  VDW: Index        Atom-I', &
       '               ', &
       '    Atom-J                  Dist     ', &
       '      EVDW       EELEC   ', &
       '      Force          Parameters')
244 FORMAT('ANAL:  VDW: Index        Atom-I', &
       '               ', &
       '    Atom-J          ',/ &
       '        Dist            EVDW', &
       '       EELEC   ', &
       '      Force          Parameters')
245 FORMAT('ANAL:  VDW>',I10,I5,4(1X,A),I5,4(1X,A), &
       4F15.6,I5,4F15.6)
246 FORMAT('ANAL:  VDW>',I10,I5,4(1X,A),I5,4(1X,A),/ &
       4F15.6,/I5,4F15.6)
  !
  return
end SUBROUTINE EVDW_CFF

#endif  /* KEY_CFF==1 */

SUBROUTINE NULL_enbonda_CFF
  RETURN
END SUBROUTINE NULL_enbonda_CFF
