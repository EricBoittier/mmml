module enbonda

contains

! Temporarily turn off expand for debugging
! ##SET .not.EXPAND

#define ENBONDA_EXPAND 0

SUBROUTINE EVDW(ENB,EEL,IFRSTA,NATOM,JNB,INBLO,CG,RSCLF,CNBA,CNBB, &
     MAXROW,IAC,ITC,NATC,IOFF, &
     QETEN,QETSR,                                         &
     LELEC,LVDW,LCONS,LSHFT,LVSHFT,LFSWT,LVFSWT,QDBEXP, &
     DX,DY,DZ,X,Y,Z,CTONNB,CTOFNB,CGONNB,CGOFNB,EPS,E14FAC, &
#if KEY_NBIPS==1
     LVIPS,LEIPS,                                          & 
#endif
     LEGROM,LVGROM,                  &
#if KEY_FLUCQ==1
     QFLUC,FQCFOR,                   & 
#endif
#if KEY_WCA==1
     LLSOFT, SCVDWCUTR, WCA,         & 
#endif
     QECONTX,ECONTX,DD1,IUPT,QSECD &
#if KEY_IMCUBES==1
     ,lbycbim                        & 
#endif
     ,QRXNF)
  !-----------------------------------------------------------------------
  !     THIS ROUTINE CALCULATES NON BONDED INTERACTION ENERGIES AND FORCES
  !
  !     ENB    - vdw energy returned
  !     EEL    - electrostatic energy returned
  !     IFRSTA - first atom to look at in INBLO
  !     NATOM  - last atom to look at in INBLO (number of atoms)
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
  !     CGONNB,CGOFNB - GROMACS switching function specifiers in real space
  !     EPS - dielectric constant
  !     DD1,QSECD - second derivative specifiers
  !
  !     By Bernard R. Brooks   5/7/81
  !
  !     Include BLOCK energy partition
  !     By Youngdo Won         12/15/90
  !----------------------------------------------------------------------

  use inbnd, only: grfcon=>RFCON ! Wei Chen 2015

#if KEY_CHEQ==1
  use cheq,only:qcg,qpartbin,   &                  
       DCH,SUMDCH,DDCH                         
#endif
  use ewald,only: kappa,lewald,erfmod
  use erfcd_mod,only:erfcd
  use chm_kinds
  use dimens_fcm
  use number
#if KEY_SCCDFTB==1
  use blockscc_fcm  
#endif
  use sftc          !Cc New PBLOCK
  use econtmod
  use euler
  use fourdm
  use stream
  use consta
#if KEY_NBIPS==1
  use nbips      
#endif
#if KEY_PBOUND==1
  use pbound     
#endif
#if KEY_BLOCK==1
  use block_fcm
  use lambdam     /*ldm*/
#endif
#if KEY_DIMB==1
  use dimb       
#endif
  ! namkh 01/20/04
#if KEY_GAMESS==1 || KEY_GAMESSUK==1 || KEY_MNDO97==1 || KEY_SQUANTM==1 || KEY_QCHEM==1 || KEY_QTURBO==1 || KEY_G09==1
  use gamess_fcm
#endif 
  use gcmc
#if KEY_PERT==1
  use pert
#endif
#if KEY_LJPME==1
  use pmeutil, only: qljpme
  use psf, only: ljc6
  use ewald, only: ljpme_dir
#endif 
  use chutil,only:atomid
  use machutil,only:die
  use parallel,only:mynod
  implicit none

  real(chm_real) ENB,EEL
  INTEGER IFRSTA, NATOM
  INTEGER JNB(*)
  INTEGER INBLO(*)
  real(chm_real) CG(*),RSCLF(*),CNBA(*),CNBB(*)
  INTEGER MAXROW
  INTEGER IAC(*),ITC(*)
  INTEGER NATC,IOFF(*)
  LOGICAL LELEC,LVDW,LCONS,LSHFT,LVSHFT,LFSWT,LVFSWT,QDBEXP
  LOGICAL LEGROM,LVGROM
  real(chm_real) DX(*),DY(*),DZ(*),X(*),Y(*),Z(*)
  real(chm_real) CTONNB,CTOFNB,CGONNB,CGOFNB,EPS,E14FAC,RC3,RC
  LOGICAL QECONTX,QRXNF
  LOGICAL QETEN,QETSR
  real(chm_real) ECONTX(*)
  real(chm_real) DD1(*)
  INTEGER IUPT(*)
  LOGICAL QSECD
#if KEY_WCA==1
  real(chm_real) WCA(*),SCVDWCUTR
  LOGICAL LLSOFT
#endif 

#if KEY_IMCUBES==1
  logical lbycbim                                
#endif
#if KEY_FLUCQ==1
  LOGICAL QFLUC
  real(chm_real) FQCFOR(*)
#endif 
  !
  ! SAPATEL
#if KEY_CHEQ==1
  real(chm_real) HIJ
#endif 
#if KEY_NBIPS==1 /*nbips_comm*/
  !WXW Long range potential using isotropic periodic sum (IPS)
  LOGICAL LEIPS,LVIPS,REIPS,CEIPS,DVIPS,DOIPS
  real(chm_real) U1,U2,U4,U6R,U12R
  real(chm_real) PE,PVC,PVA,DPE,DPVC,DPVA,DDPE,DDPVC,DDPVA
  real(chm_real) ENEP,ENEVC,ENEVA,ENBC
#endif /*   (nbips_comm)*/
  ! SAPATEL
  !
#if KEY_BLOCK==1
  real(chm_real) ENEORG                                 /*ldm*/
#endif
  real(chm_real) ETEMP1,ETEMP2,ENBPR,EELPR
  real(chm_real) C2OFNB,C4ROF,C2ROF2,CHROF2,C2ONNB
  real(chm_real) RUL3,RUL12,SGSHSQ,ASH6,BSH6
  real(chm_real) CGF,CGT,RS,CGIJ
  real(chm_real) FDXI,FDYI,FDZI,CRXI,CRYI,CRZI,CGT2,DXI,DYI,DZI
  real(chm_real) S,SIG2,R2,SIG6,SIG12,RIJL,RIJU,FUNCT,DFN,FAC,SWTMP,ODF
  real(chm_real) EN,DEN,DF,DDF,G2,R1,G1,G3,DXIT,DYIT,DZIT
  real(chm_real) AXX,AYY,AZZ,AXY,AXZ,AYZ
  real(chm_real) EADD,EADDR,ON3,ON6,ONOFF2,OFF3,OFF4,OFF5,OFF6, &
       RMIN2,RMIN6, &
       R3,R4,R6,R8,DENOM,ACOEF,BCOEF,CCOEF,DCOEF,TWOA, &
       TWOB,TWOC,TWOD,FOURDL,COVER3,DOVER5,SIXA,SIXD,CONST,CONSTR
  real(chm_real) RECOF3,RECOF6,OFDIF3,OFDIF6,ONOFF3,ONOFF6, &
       R5,CR6,CR12,RJUNK3,RJUNK6,RECOF2,MIN2OF

  real(chm_real) RMCTON,ALPHA,A6,B6,C6,G62,DG62,DDG62,G63,DG63,DDG63, &
       A12,B12,C12,G122,DG122,DDG122,G123,DG123,DDG123,R12, &
       A,B,C,DG2,DDG2,DG3,DDG3, &
       R13,R14,R7,RMCGON,RMIN12
  real(chm_real) CG2ONNB,CG2OFNB

  LOGICAL DBEXP, DBEXPIPS
  real(chm_real) SIG,AKR,EXPAR,CKR,EXPCR,AKRC,EXPARC,CKRC,EXPCRC

  INTEGER I,NB,IADD,ITEMP,J,NPR,I1,IACI,JPR,J1,IC,II,JJ,KK
  LOGICAL QAFIRST
  CHARACTER(len=8) SIDDNI,RIDDNI,RESDNI,ACDNI,SIDDNJ,RIDDNJ,RESDNJ,ACDNJ
  !
  !  for ewald summation
  real(chm_real)   DFRS,E14M1,E14F
  real(chm_real)   ERFC2,ERFCX,DRFC
  !
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
  !     REIPS   - DO R-DIELECTRIC With IPS method
  !     CEIPS  -  DO CONSTANT DIEL. with IPS method
  !     DVIPS  -  DO VDW with IPS method
  !
  !     DBEXP  -  DO VDW with double exponential potential
  !
  !     DBEXPIPS  -  DO VDW with double exponential IPS potential
  !
  !     GESWIT - DO ELECTROSTATICS WITH GROMACS SWITCH
  !     GVSWIT - DO ELECTROSTATICS WITH GROMACS SWITCH
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
  LOGICAL GESWIT,GVSWIT
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
  real(chm_real) FALPHA   !ldm
#endif /*  BLOCK*/

#if KEY_WCA==1
  real(chm_real) EPRPL, EGLJTMP, EGLJRPL, TMP
#endif 
#if KEY_GCMC==1
  logical lgcmcon  
#endif
#if KEY_PBOUND==1
  real(chm_real) corr 
#endif
  !
#if KEY_PERT==1
  !     REACTANT AND PRODUCT PART CALCULATED BY
  !     BY SEPARATE ROUTINE (SEE EPERT.SRC)
  !c
#if KEY_BLOCK==1
  IF (TQPSSP.or.QBPSSP) THEN
#else /**/
  IF (TQPSSP) THEN
#endif 
     CALL ESSNBA(ENB,EEL,IFRSTA,NATOM,JNB,INBLO,CG,RSCLF,CNBA,CNBB, &
          MAXROW,IAC,ITC,NATC,IOFF,LELEC,LVDW,LCONS,LSHFT,LVSHFT,LFSWT, &
          LVFSWT,QDBEXP, &
#if KEY_NBIPS==1
          LVIPS,LEIPS,                                          & 
#endif
          DX,DY,DZ,X,Y,Z,CTONNB,CTOFNB,EPS,E14FAC, &
          QECONTX,ECONTX,DD1,IUPT,QSECD)
     !
     RETURN
  ENDIF
#endif /* KEY_PERT */
  LSECD=QSECD

  IF(QRXNF)THEN ! Wei Chen 2015
     RC = ONE / CTOFNB
     RC3 = RC / (CTOFNB*CTOFNB)
  ENDIF

  E14M1 = E14FAC-ONE
  ELECFG=LELEC.AND.(EPS /= ZERO)
  ! Set flags for electrostatics options (12 possibilities)
  IF(.NOT.ELECFG .OR. LEWALD) THEN
     RSHFT= .FALSE.
     RSWIT= .FALSE.
     CSWIT= .FALSE.
     CSHFT= .FALSE.
     RSHIFT=.FALSE.
     RFSWIT=.FALSE.
     CSHIFT=.FALSE.
     CFSWIT=.FALSE.
     GESWIT=.FALSE.
     GVSWIT=.FALSE.
#if KEY_NBIPS==1
     REIPS=.FALSE.                  
     CEIPS=.FALSE.                  
#endif
  ELSE
#if KEY_NBIPS==1
     REIPS=.NOT.LCONS .AND.LEIPS                  
     CEIPS=     LCONS .AND.LEIPS                  
#endif
     RSHIFT= .NOT.LCONS .AND.      LSHFT .AND.      LFSWT &
          .AND. .NOT. LEGROM                              &
#if KEY_NBIPS==1
          .AND..NOT.REIPS                      
#else
     ;   
#endif
     RFSWIT= .NOT.LCONS .AND. .NOT.LSHFT .AND.      LFSWT &
          .AND. .NOT. LEGROM                              &
#if KEY_NBIPS==1
          .AND..NOT.REIPS                      
#else
     ;   
#endif
     RSHFT = .NOT.LCONS .AND.      LSHFT .AND. .NOT.LFSWT &
          .AND. .NOT. LEGROM                              &
#if KEY_NBIPS==1
          .AND..NOT.REIPS                      
#else
     ;   
#endif
     RSWIT = .NOT.LCONS .AND. .NOT.LSHFT .AND. .NOT.LFSWT &
          .AND. .NOT. LEGROM                              &
#if KEY_NBIPS==1
          .AND..NOT.REIPS                      
#else
     ;   
#endif
     CSHIFT=      LCONS .AND.      LSHFT .AND.      LFSWT &
          .AND. .NOT. LEGROM                              &
#if KEY_NBIPS==1
          .AND..NOT.CEIPS                      
#else
     ;   
#endif
     CFSWIT=      LCONS .AND. .NOT.LSHFT .AND.      LFSWT &
          .AND. .NOT. LEGROM                              &
#if KEY_NBIPS==1
          .AND..NOT.CEIPS                      
#else
     ;   
#endif
     CSHFT =      LCONS .AND.      LSHFT .AND. .NOT.LFSWT &
          .AND. .NOT. LEGROM                              &
#if KEY_NBIPS==1
          .AND..NOT.CEIPS                      
#else
     ;   
#endif
     CSWIT =      LCONS .AND. .NOT.LSHFT .AND. .NOT.LFSWT &
          .AND. .NOT. LEGROM                              &
#if KEY_NBIPS==1
          .AND..NOT.CEIPS                      
#else
     ;   
#endif
      GESWIT=LEGROM
  ENDIF
  !
  IF(LEWALD .AND. LSECD) CALL WRNDIE(-4,'<EVDW>', &
       'Second derivatives for Ewald have not as yet been implemented.')
  !
#if KEY_FLUCQ==1
  IF(QFLUC.AND.LEWALD) CALL WRNDIE(-4,'<EVDW>', &
       'No FlucQ implementation for Ewald yet')
#endif 
  ELCFG=.FALSE.
  QAFIRST=.TRUE.
  !
  ! Set flags for van der Waals options
  VDWFG=LVDW
  IF (VDWFG) THEN
     DBEXP = QDBEXP
#if KEY_NBIPS==1
     DBEXP = DBEXP .and. .not. LVIPS
     DVIPS  = LVIPS   .and. .not. QDBEXP                             
     DBEXPIPS = QDBEXP .and. LVIPS
#endif
     DFSWIT = LVFSWT &
          .AND. .NOT. LVGROM .and. .not. QDBEXP &
#if KEY_NBIPS==1
          .AND..NOT.LVIPS                      
#else 
     ;   
#endif
     DSHFT = .NOT.LVFSWT .AND.      LVSHFT &
          .AND. .NOT. LVGROM .and. .not. QDBEXP &
#if KEY_NBIPS==1
          .AND..NOT.LVIPS                      
#else
     ;   
#endif
     DSWIT = .NOT.LVFSWT .AND. .NOT.LVSHFT &
          .AND. .NOT. LVGROM .and. .not. QDBEXP &
#if KEY_NBIPS==1
          .AND..NOT.LVIPS                      
#else
     ;   
#endif
     GVSWIT=LVGROM
  ELSE
     DFSWIT = .FALSE.
     DSWIT  = .FALSE.
     DSHFT  = .FALSE.
#if KEY_NBIPS==1
     DVIPS  = .FALSE.                          
#endif
     GVSWIT = .FALSE.
     DBEXP = .FALSE.
     DBEXPIPS = .FALSE.
  ENDIF
  IF (QETEN.OR.QETSR) THEN
     DFSWIT = .FALSE.
     DSHFT  = .FALSE.
     DSWIT  = VDWFG
     GVSWIT = .FALSE.
  ENDIF
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
        FOURDL = FOUR*DCOEF
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
     !     shifted dielectric coefficients
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
     !     van der waal distance switching coefficients
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
     !     van der waal distance shifting
     RECOF2 = ONE/C2OFNB
  ENDIF
  !
#if KEY_FOURD==1
  DFDIMI=ZERO
#endif 
#if KEY_SCCDFTB==1
  if(qsccb) then
     if(idxnbd == 0) dvdlv=0.0d0
     if(idxnbd == 1) dvdlv1=0.0d0
     dvdle=0.0d0
  endif
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

  !     DO VDW AND ELECTROSTATIC TERMS AS REQUESTED
  !
  !=======================================================================
  !  Expand control section
  !-------------------------------------------------------------------
  ! (disable expand when debug is active)
! ##IF DEBUG
! ##SET .not.EXPAND
! ##ENDIF

#if KEY_DEBUG == 1
#define ENBONDA_EXPAND 0
#endif

#if ENBONDA_EXPAND == 1

!-------------------------------------------------------------------
! Do LSECD expansion of code
! ##EXPAND  lsecd analys  .when.    EXPAND  (expand_lsecd)
! ##PASS1   lsecd analys .not.EXPAND
  IF(LSECD .OR. QATERM .OR. QECONTX) THEN

#define ENBONDA_LSECD 1
#define ENBONDA_ANALYS 1

#include "enbonda1.inc"

#undef ENBONDA_LSECD
#undef ENBONDA_ANALYS

! ##PASS2
  ELSE

#include "enbonda1.inc"

! ##EXFIN
  ENDIF
! ##EXEND
! ##ENDEX    (expand_lsecd)
!=======================================================================

#else  /* ENBONDA_EXPAND */

#define ENBONDA_LSECD 1
#define ENBONDA_ANALYS 1

#include "enbonda1.inc"

#endif  /* ENBONDA_EXPAND */

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
       4F15.6,I7,4F15.6)

246 FORMAT('ANAL:  VDW>',I10,I5,4(1X,A),I5,4(1X,A),/ &
       4F15.6,/I7,4F15.6)

  RETURN
END SUBROUTINE EVDW 

end module enbonda
