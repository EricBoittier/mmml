module enbondg
   implicit none

contains

   SUBROUTINE EGROUP(EEL,ENB,NATOM,NST2,JNBG,INBLOG,CG,RSCLF, &
                        DX,DY,DZ,IFRSTG,NGRP,IGPBS,IGPTYP,INB14,IBLO14, &
                        LELEC,LVDW,LCONS,LSHFT,LFSWT,QDBEXP, &
                        CNBA,CNBB,MAXCNX,ITC,IAC,NATC,IOFF,X,Y,Z, &
                        CTONNB,CTOFNB,EPS,E14FAC,WRNMXD,QECONTX,ECONTX, &
#if KEY_NBIPS==1
                        LVIPS,LEIPS,                          & 
#endif
                        QETEN,QETSR,                          &
#if KEY_MC==1
                        LCENTR, &
#endif 
                        XCENT,YCENT,ZCENT,QCENT, &
#if KEY_FLUCQ==1
                        QFLUC,FQCFOR,  & 
#endif
                        DD1,IUPT,QSECD &
#if KEY_WCA==1
                        ,LLSOFT, SCVDWCUTR, WCA &
#endif
#if KEY_LJPME==1
                      , QLJEXC, ELJEXC  &
#endif 
                        ,QEXTND)
!
!     THIS ROUTINE CALCULATES NON BONDED INTERACTION ENERGIES AND
!     FORCES.
!
!     EEL    - electrostatic energy returned
!     ENB    - van der Waals energy returned
!     NATOM  - number of atoms
!     NST2   - number of ST2 waters
!     JNBG   - group pair list  (INBLOG(NGRP))
!     INBLOG - pointers into group pair list  (NGRP)
!     CG     - charges (NATOM)
!     RSCLF  - Radius Scaling Factor (NATOM)
!     DX,DY,DZ - force arrays
!     NGRP   - number of groups total
!     IGPBS  - base array for groups  (NGRP+1)
!     IGPTYP - group type array (0-no charges,1-neutral,2-charged,3-ST2)
!     INB14  - exclusion pair list for atoms  (INB14(NATOM))
!     IBLO14 - pointer into INB14 (NATOM)
!     LELEC,,,,,,LFSWT - flags from BNBND.FCM
!     CNBA,CNBB,MAXCNX,ITC,IAC,NATC,IOFF -vdw parameters and data arrays.
!     X,Y,Z  - coordinates
!     CTONNB,CTOFNB - group switching funtion specifiers
!     EPS    - dielectric constant
!     E14FAC - Electrostatics scale factor for 1-4 interactions.
!     WRNMXD - update warning move distance for extended electrostatics
!     QECONTX - Analysis flag for energy partition.
!     ECONTX  - Energy partition array
!     XCENT,YCENT,ZCENT - center for each group
!     QCENT  - Total change for each group
!     DD1,IUPT,QSECD - Second derivative arrays and flag.
!     LLSOFT    - soft-core flag
!     SCVDWRCUT - soft-core threshold parameter
!
!     QEXTND - Extended flag
!
!     By Bernard R. Brooks    9/22/82
!     Hessian and group force switch method added by BRB 10/12/91
!     General overhaul and code expansion added -    BRB 11/15/96
!
!     QC: UW_031205 add QEXTND to fix for fast off
!         so that extended is consistent with fast "on"
!
  use ewald,only: kappa,lewald,erfmod
  use erfcd_mod,only: erfcd
  use chm_kinds
  use number
  use consta
  use dimens_fcm
  use stream
  use sftc      ! New PBLOCK
#if KEY_NBIPS==1
  use nbips     
#endif
#if KEY_DIMB==1
  use dimb      
#endif
#if KEY_PBOUND==1 /*pbound*/
  use pbound
#endif /*     (pbound)*/
#if KEY_BLOCK==1
  use block_fcm
#endif 
#if KEY_MTS==1
  use tbmts  
#endif
#if KEY_PERT==1
  use pert
#endif 
!pssp
!
! namkh 01/20/04
#if KEY_GAMESS==1 || KEY_GAMESSUK==1 || KEY_MNDO97==1 || KEY_SQUANTM==1 || KEY_QCHEM==1 || KEY_QTURBO==1 || KEY_G09==1
  use gamess_fcm
#endif
#if KEY_LJPME==1
  use pmeutil, only: qljpme, dkappa
  use ewald, only: ljpme_dir
  use energym, only: ljexc, eterm, qeterm
  use psf, only: ljc6
#endif 
      implicit none

      logical :: qdbexp
#if KEY_NBIPS==1 /*nbips_comm*/
!WXW Long range potential using isotropic periodic sum (IPS)
      LOGICAL LEIPS,LVIPS,DOIPS,DVIPS,CEIPS,REIPS
      real(chm_real) U1,U2,U4,U6R,U12R
      real(chm_real) PE,PVC,PVA,DPE,DPVC,DPVA,DDPE,DDPVC,DDPVA
      real(chm_real) ENEP,ENEVC,ENEVA,ENBC
#endif /*   (nbips_comm)*/
      LOGICAL DBEXP, DBEXPIPS
      real(chm_real) SIG,EMIN,AKR,EXPAR,CKR,EXPCR,AKRC,EXPARC,CKRC,EXPCRC
#if KEY_LJPME==1 /* ljpme */
      LOGICAL QLJEXC
      real(chm_real) ELJEXC
#endif /* ljpme */
!
      real(chm_real) EEL,ENB
      INTEGER NATOM,NST2,INBLOG(*)
      INTEGER JNBG(*)
      real(chm_real) CG(*),RSCLF(*)
      real(chm_real) DX(*),DY(*),DZ(*)
      INTEGER IFRSTG,NGRP
      INTEGER IGPBS(*),IGPTYP(*),INB14(*),IBLO14(*)
      LOGICAL LELEC,LVDW,LCONS,LSHFT,LFSWT
      INTEGER MAXCNX,NATC
      real(chm_real) CNBA(*),CNBB(*)
      INTEGER ITC(*),IAC(*)
      INTEGER IOFF(NATC)
      real(chm_real) X(*),Y(*),Z(*)
      real(chm_real) CTONNB,CTOFNB,EPS,E14FAC,WRNMXD
      LOGICAL QECONTX
      real(chm_real) ECONTX(*)
      real(chm_real) XCENT(*),YCENT(*),ZCENT(*),QCENT(*)
      real(chm_real) DD1(*)
      INTEGER IUPT(*)
      LOGICAL QSECD
      LOGICAL QETEN,QETSR
!     QC: UW_031205 fix for fast off
!         so that extended is consistent with fast "on"
      LOGICAL QEXTND

#if KEY_WCA==1
      real(chm_real) WCA(*),SCVDWCUTR
      LOGICAL LLSOFT
#endif 

!
#if KEY_PBOUND==1 /*pbound*/
      real(chm_real) CORR
#endif /*     (pbound)*/
#if KEY_BLOCK==1
      INTEGER IBL,JBL,KK
      real(chm_real)  COEF
#endif /*  BLOCK*/
#if KEY_MC==1
      INTEGER LCENTR
#endif /*  MC*/
#if KEY_FLUCQ==1
      LOGICAL QFLUC
      real(chm_real) FQCFOR(*)
#endif 
#if KEY_MTS==1
      real(chm_real) RR1,RR2,RR3
      real(chm_real) SWF,SWFE
#endif 
!pssp
!
!
! local
      real(chm_real) ETEMP1,ETEMP2,EELPR,ENBPR,ESWADD
      real(chm_real) C2ONNB,C2OFNB,RUL3,RUL12
      real(chm_real) SIG2,SIG6,SIG12,FAC,SWTMP,ODF
      INTEGER IRS,IS,IQ,I,JRS,JSS,JQ,J,II,JJ,IX,JX,IADD
      INTEGER NAT,NB,ITEMP,NPR,JRSPR,NI,NJ
      INTEGER LEX14,NXI,NXIMAX,JSX,INBX,I1,J1,IC
      real(chm_real) DXI,DYI,DZI,DXIT,DYIT,DZIT,DXJT,DYJT,DZJT, &
           DXIC,DYIC,DZIC
      real(chm_real) CGF,FUNCT,CGT,CGP,S,R2,DF,DFN,DFI,DFJ
      real(chm_real) AXX,AXY,AXZ,AYY,AYZ,AZZ,DDF,DDFN
      real(chm_real) SCENT,RIJL,RIJU
      LOGICAL LCSW,LRSW,LVGRP,LSWITR,LEXCL,LST2,LSWIT,LSECD
      LOGICAL SKIP
      real(chm_real) RS,ERFCX,DRFC,E14M1,CGIJ
#if KEY_WCA==1
      real(chm_real) EPRPL, EGLJTMP, EGLJRPL, TMP
#endif 
!
#if KEY_PERT==1
!     REACTANT AND PRODUCT PART CALCULATED BY
!     BY SEPARATE ROUTINE (SEE EPERT.SRC)
#if KEY_BLOCK==1
      IF (TQPSSP.or.QBPSSP) THEN                      !Cc New PBLOCK
#else /**/
      IF (TQPSSP) THEN                      !Cc New PBLOCK
#endif 
         CALL ESSNBG(EEL,ENB,NATOM,NST2,JNBG,INBLOG,CG,RSCLF, &
              DX,DY,DZ,NGRP,IGPBS,IGPTYP,INB14,IBLO14, &
              LELEC,LVDW,LCONS,LSHFT,LFSWT,QDBEXP, &
#if KEY_NBIPS==1
                        LVIPS,LEIPS,                                          & 
#endif
              CNBA,CNBB,MAXCNX,ITC,IAC,NATC,IOFF,X,Y,Z, &
              CTONNB,CTOFNB,EPS,E14FAC,WRNMXD,QECONTX,ECONTX, &
              XCENT,YCENT,ZCENT,QCENT,DD1,IUPT,QSECD)
         RETURN
      ENDIF
#endif /* KEY_PERT */
!
! Set option flags
!
      LSECD=QSECD
      LVGRP=LVDW
      IF (.NOT.(LELEC.OR.LVDW)) RETURN
      IF (EPS == ZERO .AND. .NOT.LVDW) RETURN
      IF (.NOT.LELEC .OR. EPS == ZERO) THEN
        LCSW=.FALSE.
        LRSW=.FALSE.
#if KEY_NBIPS==1
        CEIPS=.FALSE.                          
        REIPS=.FALSE.                          
#endif
      ELSE IF (LEWALD) THEN
        LCSW=.FALSE.
        LRSW=.FALSE.
#if KEY_NBIPS==1
        CEIPS=.FALSE.                          
        REIPS=.FALSE.                          
#endif
      ELSE IF (LCONS) THEN
#if KEY_NBIPS==1
        IF(LEIPS)THEN
          CEIPS=.TRUE.
          REIPS=.FALSE.
          LCSW=.FALSE.
          LRSW=.FALSE.
        ELSE
          CEIPS=.FALSE.
          REIPS=.FALSE.
#endif /* KEY_NBIPS */
          LCSW=.TRUE.
          LRSW=.FALSE.
#if KEY_NBIPS==1
        ENDIF                             
#endif
      ELSE
#if KEY_NBIPS==1
        IF(LEIPS)THEN
          CEIPS=.FALSE.
          REIPS=.TRUE.
          LCSW=.FALSE.
          LRSW=.FALSE.
        ELSE
          CEIPS=.FALSE.
          REIPS=.FALSE.
#endif /* KEY_NBIPS */
          LCSW=.FALSE.
          LRSW=.TRUE.
#if KEY_NBIPS==1
        ENDIF                             
#endif
      ENDIF
#if KEY_NBIPS==1
!      QIPS=CEIPS.OR.REIPS.OR.(LVIPS.AND.LVGRP)      
#endif
      LSWIT=.TRUE.
      IF (LSHFT) THEN
        CALL WRNDIE(-2,'<EGROUP>','Invalid nonbond options')
!       don't return (hidden group shift option).
        LSWIT=.FALSE.
      ENDIF
!
      IF(LEWALD .AND. LSECD) CALL WRNDIE(-4,'<EGROUP>', &
       'Second derivatives for Ewald have not as yet been implemented')
!
#if KEY_FLUCQ==1
      IF(LEWALD.AND.QFLUC) CALL WRNDIE(-4,'<EGROUP>', &
       'No FlucQ implementation for Ewald yet')
#endif 
      IF(LVGRP) THEN
         ! DBEXP flag
         DBEXP = QDBEXP
#if KEY_NBIPS==1
         DBEXP = DBEXP .and. .not. LVIPS
         DVIPS  = LVIPS   .and. .not. QDBEXP                             
         DBEXPIPS = QDBEXP .and. LVIPS
#endif
        J=0
        DO I=1,NATC
          IOFF(I)=J
          J=J+I
        ENDDO
        ENB=ZERO
      ENDIF
      ENBPR=ZERO
!
      E14M1 = E14FAC-ONE
      C2ONNB=CTONNB*CTONNB
      C2OFNB=CTOFNB*CTOFNB
      IF (CTOFNB > CTONNB) THEN
        RUL3=ONE/(C2OFNB-C2ONNB)**3
        RUL12=RUL3*TWELVE
      ENDIF
!
! QC: UW_031205 fix for fast off
!     so that extended is consistent with fast "on"
!..RHS B980630.rs1
!     WRITE (*,*) "QC> EGROUP, EXT",QEXTND
      IF (QEXTND) THEN
!     NO CUTOFFS FROM GROUPS IN LIST
        C2OFNB=RBIG*THOSND*THOSND
        C2ONNB=RBIG*THOSND*THOSND
      ENDIF
!
      CGF=ZERO
      IF (LELEC .AND. EPS /= 0.0) CGF=CCELEC/EPS
!
! Find group centers (no mass weighting)
!
#if KEY_MC==1
      IF (LCENTR  <  1) THEN
#endif 
#if KEY_NBIPS==1
      IF (.NOT.QIPS ) THEN                   
#endif
      DO IRS=IFRSTG,NGRP
        IS=IGPBS(IRS)+1
        IQ=IGPBS(IRS+1)
        NAT=IQ-IS+1
        DXI=ZERO
        DYI=ZERO
        DZI=ZERO
        DO I=IS,IQ
          DXI=DXI+X(I)
          DYI=DYI+Y(I)
          DZI=DZI+Z(I)
        ENDDO
        XCENT(IRS)=DXI/NAT
        YCENT(IRS)=DYI/NAT
        ZCENT(IRS)=DZI/NAT
        IF (LFSWT) THEN
          DXI=ZERO
          DO I=IS,IQ
            DXI=DXI+CG(I)
          ENDDO
          QCENT(IRS)=ZERO
          IF (LCSW) QCENT(IRS)=DXI*SQRT(CGF/CTOFNB)
          IF (LRSW) QCENT(IRS)=DXI*SQRT(CGF/C2OFNB)
        ENDIF
      ENDDO
#if KEY_NBIPS==1
      ENDIF                                   
#endif
#if KEY_MC==1
      ENDIF
#endif 

      EEL=ZERO
      ESWADD=ZERO

!=======================================================================
!  Expand control section
!-------------------------------------------------------------------
! (disable expand when debug is active)
! ##IF DEBUG
! ##SET .not.EXPAND
! ##ENDIF

#if KEY_DEBUG==1
#undef KEY_EXPAND
#endif

#if KEY_EXPAND==1
!-------------------------------------------------------------------
! Do LSECD and analysis expansion of code
! ##EXPAND  lsecd analys  .when.    EXPAND  (expand_lsecd)
! ##PASS1   lsecd analys .not.EXPAND
      IF(LSECD .OR. QECONTX .OR. (NST2 > 0)) THEN

#undef KEY_EXPAND
#define ENBONDG_LSECD 1
#define ENBONDG_ANALYS 1

#include "enbondg1.inc"

#undef ENBONDG_LSECD
#undef ENBONDG_ANALYS
#define KEY_EXPAND 1

! ##PASS2   NOST2
      ELSE


#define KEY_NOST2 1
#include "enbondg1.inc"
         
! ##EXFIN
      ENDIF
! ##EXEND
! ##ENDEX    (expand_lsecd)
#else /* KEY_EXPAND */
      
#define ENBONDG_LSECD 1
#define ENBONDG_ANALYS 1

#include "enbondg1.inc"

#endif /* KEY_EXPAND */
!=======================================================================

      RETURN
   END SUBROUTINE EGROUP

   SUBROUTINE ENBFSG(ENB,EEL,LELECX,LVDWX,JNBGX,INBLGX,IFRSTG,NGRP, &
            IGPBS,IGPTYP,INB14X,IBLO14X,CGX, &
            IACNB,NITCC2,LOWTP, &
#if KEY_BLOCK==1
            IBLOCK,BLCOE,BLCOV,BLCOVR,BLCOVA, & 
#endif
#if KEY_MC==1
            LCENTR,                           & 
#endif
            XCENT,YCENT,ZCENT,QCENT, &
#if KEY_BLOCK==1
            NATOMX,                           &  /*ldm*/
#endif
#if KEY_FLUCQ==1
            QFLUC,FQCFOR,                     & 
#endif
            LUSED, &
#if KEY_WCA==1
            LLSOFT,SCVDWCUTR,WCA,             & 
#endif
            QEXTND)
!-----------------------------------------------------------------------
!     This is the fast scalar version of the group nonboned energy
!     for {constant dielectric} {electrostatic fswitch }
!         {distance dielectric} {electrostatic switch  } {vdW switch}
!     All combinations of these nonbond energy options are supported
!
!     LFAST>=0 and LMACH=0 are required to use this routine.
!
!     EEL     - electrostatic energy returned
!     ENB     - van der Waals energy returned
!     LELECX  - electrostatics flag
!     LVDWX   - van der Waals flag
!     JNBGX   - group pair list  (INBLGX(NGRP))
!     INBLGX  - pointers into group pair list  (NGRP)
!     NGRP    - number of groups total
!     IGPBS   - base array for groups  (NGRP+1)
!     IGPTYP  - group type array (0-no charges,1-neutral,2-charged,3-ST2}
!     INB14X  - exclusion pair list for atoms  (INB14X(NATOM))
!     IBLO14X - pointer into INB14X(NATOM)
!     XCENT,YCENT,ZCENT - center for each group
!     QCENT   - Total change for each group
!  
!     REIPS   - DO R-DIELECTRIC With IPS method
!     CEIPS  -  DO CONSTANT DIEL. with IPS method
!     DVIPS  -  DO VDW with IPS method
!
!     DBEXP  -  DO VDW with double exponential potential
!
!     DBEXPIPS  -  DO VDW with double exponential IPS potential
!

!     18-JAN-91 Youngdo Won, developed from EGROUP (of enbond.src)
!     01-SEP-91 Youngdo Won, BLOCK partition added
!     12-OCT-91 Bernie Brooks, EGRPFS wtih fswitch option
!     22-OCT-91 Youngdo Won, merge EGRPFS into ENBFSG and put it back
!     31-OCT-91 BRB          Add FSWITCH option, select flags,...
!-----------------------------------------------------------------------

      use ewald_1m,only: lewald
      use nb_module  ! has ccnba thru d
      use chm_kinds
      use number
      use consta
      use dimens_fcm
      use bases_fcm
      use coord
      use deriv
      use inbnd
      use param
      use stream
      use pbound
#if KEY_BLOCK==1
      use block_fcm  
#endif
#if KEY_BLOCK==1
      use lambdam     /*ldm*/
#endif
#if KEY_NBIPS==1
      use nbips      
#endif
#if KEY_ASPENER==1
      use eef1_mod   
#endif
#if KEY_MTS==1
      use tbmts      
#endif
      use gcmc
! namkh 01/20/04
#if KEY_GAMESS==1 || KEY_GAMESSUK==1 || KEY_MNDO97==1 || KEY_SQUANTM==1 || KEY_QCHEM==1 || KEY_QTURBO==1 || KEY_G09==1
      use gamess_fcm
#endif 
! For variable cutoffs of LJ interactions
      use varcutm

      implicit none

#if KEY_BLOCK==1 /*ldm*/
      INTEGER NATOMX
      real(chm_real) ENEORG
#endif /*  LDM*/
      real(chm_real)  EEL,ENB
      LOGICAL LELECX,LVDWX
      INTEGER JNBGX(*),INBLGX(*)
      INTEGER IFRSTG,NGRP
      INTEGER IGPBS(*),IGPTYP(*),INB14X(*),IBLO14X(*)
      real(chm_real)  CGX(*)
      INTEGER IACNB(*),NITCC2,LOWTP(*)
      real(chm_real)  XCENT(*),YCENT(*),ZCENT(*),QCENT(*)
      LOGICAL LUSED
#if KEY_WCA==1
      real(chm_real) WCA(*),SCVDWCUTR
      LOGICAL LLSOFT
#endif 
!..RHS B980630.rs1
      LOGICAL QEXTND
!..

#if KEY_BLOCK==1
      INTEGER IBL,JBL,KK, IBLOCK(*)
      real(chm_real)  COEFVD, COEFEL
      real(chm_real)  BLCOE(*),BLCOV(*),BLCOVR(*),BLCOVA(*)
#if KEY_DOCK==1
      INTEGER KDOC
      real(chm_real)  DOCFI, DOCFJ
#endif /*  DOCK*/

!ldm
      real(chm_real) FALPHA
      INTEGER ILDM, JLDM
! LDM
#endif /*  BLOCK*/


#if KEY_FLUCQ==1
      LOGICAL QFLUC
      real(chm_real) FQCFOR(*)
#endif 

#if KEY_ASPENER==1
      real(chm_real) LAMD,IR2
#endif /*  ASPENER*/

#if KEY_NBIPS==1 /*nbips_comm*/
!WXW Long range potential using isotropic periodic sum (IPS)
      LOGICAL LEIPSX,LVIPSX,DOIPS,DVIPS
      real(chm_real) U1,U2,U4,U6R,U12R
      real(chm_real) PE,PVC,PVA,DPE,DPVC,DPVA
      real(chm_real) ENEP,ENEVC,ENEVA,SIPS,DSIPS
#endif /*   (nbips_comm)*/

      LOGICAL DBEXP, DBEXPIPS
      real(chm_real) SIG,Emin,AKR,EXPAR,CKR,EXPCR,AKRC,EXPARC,CKRC,EXPCRC


      real(chm_real) ETEMP1,ETEMP2,EELPR,ENBPR
      real(chm_real) C2ONNB,C2OFNB,RUL3,RUL12,CDIFF
      real(chm_real) CA,CB,TR6,TTPW1,TTPW2,TTP12,SWTMP
      INTEGER IRS,IS,IQ,I,JRS,JS,JQ,J
      INTEGER NAT,NB,ITEMP,NPR,JRSPR,NI,NJ
      INTEGER LEX14,NXI,NXIMAX,JSX,INBX,IVECT,IACI
      real(chm_real) DXI,DYI,DZI,DXIT,DYIT,DZIT,DXJT,DYJT,DZJT, &
            DXIC,DYIC,DZIC
      real(chm_real) CGF,FUNCT,CGT,S,R2,DF,DFN,DFI,DFJ,DFF
      real(chm_real) DFVDW, DFELEC, DFZ
      real(chm_real) SCENT,RIJL,RIJU
      real(chm_real) DK1,DK2,TMPR1,TMPR2
      LOGICAL LST2,LEXCL,LSWITR,LVGRP,LSWIT,LCSW,LRSW
#if KEY_WCA==1
      real(chm_real) EPRPL,EGLJTMP,EGLJRPL,TMP,VDWWL,RMIN6,RMIN2
#endif 
#if KEY_MC==1
      INTEGER LCENTR
#endif 
#if KEY_MTS==1
      real(chm_real) RR1,RR2,RR3
      real(chm_real) SWF,SWFE
#endif 

      !---------- Sanity check -------------------------------------
      if(.not. allocated(ccnba))then
         ! How we got here without vdw table filled, who knows?
         call wrndie(-4,"ENBFSG<enbfast.src>", &
               "CCNBA not allocated")
      endif

      LUSED=.TRUE.
      ENB=ZERO
      EEL=ZERO
!
! Conditional to set options
!
      LVGRP=LVDWX
      IF (.NOT.(LELECX.OR.LVDWX)) RETURN
      IF (EPS == ZERO .AND. .NOT.LVDWX) RETURN
      IF (.NOT.LELECX .OR. EPS == ZERO) THEN
         LCSW=.FALSE.
         LRSW=.FALSE.
      ELSE IF (LCONS) THEN
         LCSW=.TRUE.
         LRSW=.FALSE.
      ELSE
         LCSW=.FALSE.
         LRSW=.TRUE.
      ENDIF
      DBEXP=LDBEXP.AND.LVDWX
#if KEY_NBIPS==1
      DBEXP = DBEXP .and. .not. LVIPS
      DVIPS  = LVIPS   .and. .not. LDBEXP                             
      DBEXPIPS = LDBEXP .and. LVIPS
      LEIPSX=LEIPS.AND.LELECX
      LVIPSX=LVIPS.AND.LVDWX.AND..NOT.LDBEXP
      LSWIT=.NOT.LEIPSX
#else /**/
      LSWIT=.TRUE.
#endif 
      IF (LSHFT) THEN
         CALL WRNDIE(-1,'<ENBFSG>','Invalid GROUP nonbond option.')
!       don't return (hidden group shift option).
         LSWIT=.FALSE.
      ENDIF

      IF(LVSHFT .OR. LVFSWT .OR. LEWALD) THEN
         LUSED=.FALSE.
         RETURN
      ENDIF
#if KEY_PBOUND==1 /*pbound*/
      IF(QBOUN) THEN
         LUSED=.FALSE.
         RETURN
      ENDIF
#endif /* (pbound)*/

      C2ONNB=CTONNB*CTONNB
      C2OFNB=CTOFNB*CTOFNB
      CDIFF=CTOFNB-CTONNB
      IF (CTOFNB > CTONNB) THEN
         RUL3=ONE/(C2OFNB-C2ONNB)**3
         RUL12=RUL3*TWELVE
      ENDIF
!..RHS B980630.rs1
      IF (QEXTND) THEN
!     NO CUTOFFS FROM GROUPS IN LIST
         C2OFNB=RBIG*THOSND*THOSND
         C2ONNB=RBIG*THOSND*THOSND
      ENDIF
!..
      CGF=ZERO
      IF (LELECX .AND. EPS /= 0.0) CGF=CCELEC/EPS

! Find group centers (no mass weighting)
#if KEY_MC==1
      IF (LCENTR  <  1) THEN
#endif 
#if KEY_NBIPS==1
         IF (.NOT.QIPS ) THEN  
#endif
            DO IRS=IFRSTG,NGRP
               IS=IGPBS(IRS)+1
               IQ=IGPBS(IRS+1)
               NAT=IQ-IS+1
               DXI=ZERO
               DYI=ZERO
               DZI=ZERO
               DO I=IS,IQ
                  DXI=DXI+X(I)
                  DYI=DYI+Y(I)
                  DZI=DZI+Z(I)
               ENDDO
               XCENT(IRS)=DXI/NAT
               YCENT(IRS)=DYI/NAT
               ZCENT(IRS)=DZI/NAT
               IF (LFSWT) THEN
                  DXI=ZERO
                  DO I=IS,IQ
                     DXI=DXI+CGX(I)
                  ENDDO
                  QCENT(IRS)=ZERO
                  IF (LCSW) QCENT(IRS)=DXI*SQRT(CGF/CTOFNB)
                  IF (LRSW) QCENT(IRS)=DXI*SQRT(CGF/C2OFNB)
               ENDIF
            ENDDO
#if KEY_NBIPS==1
         ENDIF  
#endif
#if KEY_MC==1
      ENDIF
#endif 

! Loop over pairs of groups in groups list
      loop380: DO IRS=IFRSTG,NGRP
         IF(IRS > 1) THEN
            ITEMP=INBLGX(IRS-1)
            NPR=INBLGX(IRS)-ITEMP
         ELSE
            NPR=INBLGX(IRS)
            ITEMP=0
         ENDIF
         IS=IGPBS(IRS)+1
         IQ=IGPBS(IRS+1)
         NI=IQ-IS+1
         LST2=IGPTYP(IRS) == 3

         loop370: DO JRSPR=1,NPR
            NB=ITEMP+JRSPR
            JRS=JNBGX(NB)
            LEXCL=(JRS < 0)
            IF(LEXCL) JRS=-JRS

!     Use ST2 interaction for this pair
            IF(LST2 .AND. (IGPTYP(JRS) == 3)) CYCLE

            JS=IGPBS(JRS)+1
            JQ=IGPBS(JRS+1)
            NJ=JQ-JS+1
            ETEMP1=ZERO
            ETEMP2=ZERO
            EELPR=ZERO
            ENBPR=ZERO
            FUNCT=ONE
!
!     Process this group pair
!
#if KEY_MTS==1
!---- LONG-SHORT RANGE MTS METHOD
            IF(SLFG) THEN
               SWFE = ONE
               SWF  = ONE
               DXIC=XCENT(IRS)-XCENT(JRS)
               DYIC=YCENT(IRS)-YCENT(JRS)
               DZIC=ZCENT(IRS)-ZCENT(JRS)
               SCENT=DXIC*DXIC+DYIC*DYIC+DZIC*DZIC
               IF((SCENT > RSHL2).AND.(SCENT < RSCUT2)) THEN
                  RR1 = SQRT(SCENT)
                  RR2 = (RR1-RSHL)/RHEAL
                  RR3 = ONE-RR2*RR2*RR2*(6.0*RR2*RR2-15.0*RR2+10.0)
                  IF(SLFG1) SWFE=ZERO
               ELSE IF(SCENT <= RSHL2) THEN
                  RR3 = ONE
                  IF(SLFG2) SWFE=ZERO
               ELSE IF(SCENT >= RSCUT2) THEN
                  RR3 = ZERO
                  IF(SLFG1) SWFE=ZERO
               ENDIF
               IF(SLFG1) SWF = RR3
               IF(SLFG2) SWF = ONE-RR3
            ENDIF
#endif 

!     Use normal electrostatics for this pair
            IF (LEXCL) THEN
!     Check atom exclusion list for exclusions
               loop170: DO I=IS,IQ
                  CGT=CGF*CGX(I)
                  IACI=IACNB(I)
                  IF (I > 1) THEN
                     NXI=IBLO14X(I-1)+1
                  ELSE
                     NXI=1
                  ENDIF
                  NXIMAX=IBLO14X(I)
                  IF (IS == JS) THEN
                     JSX=I+1
                  ELSE
                     JSX=JS
                  ENDIF
#if KEY_GCMC==1
                  if(qgcmc) then
                     IF (.NOT. GCMCON(I)) CYCLE
                  endif
#endif 
                  loop160: DO J=JSX,JQ
#if KEY_GCMC==1
                     if(qgcmc) then
                        IF (.NOT. GCMCON(J)) CYCLE
                     endif
#endif 

! namkh 01/20/04
#if KEY_GAMESS==1 || KEY_GAMESSUK==1 || KEY_MNDO97==1 || KEY_SQUANTM==1 || KEY_QCHEM==1 || KEY_QTURBO==1 || KEY_G09==1
                     if(qmused)then
                     IF((ABS(IGMSEL(I)) == 1.OR.ABS(IGMSEL(I)) == 2) .AND. &
                           (ABS(IGMSEL(J)) == 1.OR.ABS(IGMSEL(J)) == 2) &
                           .AND.QGMREM) CYCLE
                     endif
#endif 

! T.S.
                     DFELEC=0.
                     DFVDW=0.

                     LEX14=0
                     DO WHILE (NXI <= NXIMAX)
                        IF (INB14X(NXI) < 0) THEN
                           INBX=-INB14X(NXI)
                           IF (J == INBX) THEN
                              LEX14 = NITCC2
                              EXIT
                           ENDIF
                           IF (J < INBX) EXIT
                        ELSE
                           IF(J == INB14X(NXI)) CYCLE loop160
                           IF (J < INB14X(NXI)) EXIT
                        ENDIF
                        NXI=NXI+1
                     ENDDO
                     DXI=X(I)-X(J)
                     DYI=Y(I)-Y(J)
                     DZI=Z(I)-Z(J)
!CC Changed by Georgios to prevent overflow of non-interacting blocks
!CC that are on top of each other; Nov. 7 95
                     S=MAX(RSMALL,DXI*DXI+DYI*DYI+DZI*DZI)
!CC
!CC  original code             S=DXI*DXI+DYI*DYI+DZI*DZI
!CC
                     R2=ONE/S

#if KEY_NBIPS==1
                     IF(QIPS)THEN
!WXW Long range potential using isotropic periodic sum (IPS)
                        U2=S*RIPS2R
                        DOIPS=LEX14.LE.0
                     ENDIF
                     IF(LEIPSX)THEN
                        IF(DOIPS)THEN
!WXW Electrostatic potential using isotropic periodic sum (IPS)
                           IF(LCONS)THEN
!WXW Electrostatic potential using isotropic periodic sum (IPS)
!  Electrostatic IPS
!   etr1=1/r+r2*(a1+r2*(a2+r2*(a3+r2*(a4+r2*(a5+a6*r2)))))
!   detr1/dr*r=-1/r+r2*(d1+r2*(d2+r2*(d3+r2*(d4+r2*(d5+d6*r2)))))
!
                              U1=SQRT(U2)
                              PE=ONE/U1+U2*(AIPSE(1)+U2*(AIPSE(2)+U2*(AIPSE(3) &
                                    +U2*(AIPSE(4)+U2*(AIPSE(5)+U2*AIPSE(6))))))-PIPSEC
                              DPE=-ONE/U1+U2*(BIPSE(1)+U2*(BIPSE(2)+U2*(BIPSE(3) &
                                    +U2*(BIPSE(4)+U2*(BIPSE(5)+U2*BIPSE(6))))))
                              ENEP=CGT*CGX(J)*RIPSR
                              EELPR=ENEP*PE
                              DFELEC=ENEP*DPE*R2
                           ELSE
!  Electrostatic IPS
!   etr1=1/r2+r2*(a1+r2*(a2+r2*(a3+r2*(a4+r2*(a5+a6*r2)))))
!   detr1/dr*r1=-1/r2+r2*(d1+r2*(d2+r2*(d3+r2*(d4+r2*(d5+d6*r2)))))
!
                              PE=ONE/U2+U2*(AIPSE(1)+U2*(AIPSE(2)+U2*(AIPSE(3) &
                                    +U2*(AIPSE(4)+U2*(AIPSE(5)+U2*AIPSE(6))))))-PIPSEC
                              DPE=-TWO/U2+U2*(BIPSE(1)+U2*(BIPSE(2)+U2*(BIPSE(3) &
                                    +U2*(BIPSE(4)+U2*(BIPSE(5)+U2*BIPSE(6))))))
                              ENEP=CGT*CGX(J)*RIPS2R
                              EELPR=ENEP*PE
                              DFELEC=ENEP*DPE*R2
                           ENDIF
                        ELSE
                           EELPR=ZERO
                           DFELEC=ZERO
                        ENDIF
                     ELSE
#endif 
#if KEY_ASPENER==1
                        IF (LRSW .AND. LMEMBR .AND. ((LEX14 <= 0).or. &
                        (E14FAC.LT.0.9))) THEN
                           LAMD=SQRT(LAM(I)*LAM(J))
                           IR2=SQRT(R2)
                           EELPR=CGT*CGX(J)*IR2*IR2**LAMD
                           DFELEC=-(ONE+LAMD)*R2*EELPR
                           DFZ= EELPR*LOG(IR2)*(ONE-AEMPIR)/TWO/LAMD
                        ELSE IF (LRSW .AND. LDEBYE) THEN
                           IR2=SQRT(R2)
                           EELPR=CGT*CGX(J)*R2*EXP(-1./IR2/RDEBYE)
                           DFELEC=(R2*MINTWO-IR2/RDEBYE)*EELPR
                        ELSE IF (LRSW) THEN
#else /**/
                        IF (LRSW) THEN
#endif /* ASPENER*/
                           EELPR=CGT*CGX(J)*R2
                           DFELEC=R2*MINTWO*EELPR
                        ELSE
                           EELPR=CGT*CGX(J)*SQRT(R2)
                           DFELEC=-R2*EELPR
                        ENDIF
#if KEY_NBIPS==1
                     ENDIF  
#endif
                        IF(LEX14 > 0) THEN
                           DFELEC=DFELEC*E14FAC
                           EELPR=EELPR*E14FAC
#if KEY_ASPENER==1
                           IF (LMEMBR) THEN
                             IF (E14FAC .LT. 0.9) THEN
                                DFZ=DFZ*E14FAC
                             ELSE
                                DFZ=0
                             ENDIF
                           ENDIF
#endif /* ASPENER*/
                        ENDIF

                     IF (LVGRP) THEN
                        IVECT=LOWTP(MAX(IACNB(J),IACI))+IACNB(J)+IACI+LEX14
                        TR6=R2*R2*R2
                        SIG=CCNBB(IVECT)
                        IF(SIG<RSMALL)THEN
                           ENBPR=ZERO
                           DFVDW=ZERO
                        ELSE IF(DBEXP)THEN
                 !     VAN DER WAAL DOUBLE EXPONENTIAL POTENTIAL
                                         !     E= A*EXP(alpha*R)-B*EXP(beta*R)
                  !
                           EMIN=SIG*SIG/CCNBA(IVECT)/FOUR
                           SIG=(TR6*HALF*SIG/EMIN)**SIXTH
#if KEY_NBIPS==1
                           AKR=AIPSVA(0)/SIG
                           EXPAR=EXP(-AKR)
                           ! energy
                           ENBPR=AIPSVA(1)*EXPAR
                           ! force
                           DF=BIPSVA(1)*EXPAR/SIG
                           CKR=AIPSVC(0)/SIG
                           EXPCR=EXP(-CKR)
                           ! energy
                           ENBPR=EMIN*(ENBPR-AIPSVC(1)*EXPCR)
                           ! force
                           DFVDW=EMIN*(DF-BIPSVC(1)*EXPCR/SIG)*R2
#if ENBONDA_LSECD==1 /*lsecd1 */
                           IF(LSECD) THEN
                             SIG2=SIG*SIG
                             DDF=-EMIN*(BIPSVA(1)*EXPAR*AIPSVA(0)-BIPSVC(1)*EXPCR*AIPSVC(0))*R2*R2/SIG2
                           ENDIF
#endif /*  (lsecd1)*/

                        ELSE IF (DBEXPIPS) THEN                              
                          !     VAN DER WAAL DOUBLE EXPONENTIAL IPS
                          !WXW VDW DE potential using isotropic periodic sum (IPS)
                          EMIN=SIG*SIG/CCNBA(IVECT)/FOUR
                          SIG=(TR6*HALF*SIG/EMIN)**SIXTH
                          IF(DOIPS.AND.SRIPS*SIG>ONE)THEN
                            AKR=AIPSVA(0)/SIG
                            EXPAR=EXP(-AKR)
                            ! energy
                            ENBPR=AIPSVA(1)*(EXPAR+AIPSVA(4)/EXPAR+AIPSVA(5))
                            ! force 
                            DF=BIPSVA(1)*(EXPAR-AIPSVA(4)/EXPAR)/SIG
                            !
                            CKR=AIPSVC(0)/SIG
                            EXPCR=EXP(-CKR)
                            ! energy
                            ENBPR=EMIN*(ENBPR-AIPSVC(1)*(EXPCR+AIPSVC(4)/EXPCR+AIPSVC(5)))
                            ! force
                            DFVDW=EMIN*(DF-BIPSVC(1)*(EXPCR-AIPSVC(4)/EXPCR)/SIG)*R2
#if ENBONDA_LSECD==1 /*lsecd2*/
                            IF(LSECD) THEN
                               SIG2=SIG*SIG
                               DDF=EMIN*(AIPSVA(0)*(BIPSVA(1)*(EXPAR+AIPSVA(4)/EXPAR)) &
                               -AIPSVC(0)*(BIPSVC(1)*(EXPCR+AIPSVC(4)/EXPCR)))*R2*R2/SIG2
                            ENDIF
#endif /*  (lsecd2)*/
                          ELSE
                             ENBPR=ZERO
                             DFVDW=ZERO
                          ENDIF 
                        ELSE IF(DVIPS)THEN
                           !WXW VDW potential using isotropic periodic sum (IPS)
                           IF(DOIPS)THEN
                              U4=U2*U2
                              U6R=ONE/U4/U2
                              U12R=U6R*U6R
            !  L-J r6 term
            !   etr6=1/r6+a0+r2*(a1+r2*(a2+r2*(a3+r2*(a4+r4*(a5+a6*r4)))))
            !   detr6/dr*r1=-6/r6+r2*(d1+r2*(d2+r2*(d3+r2*(d4+r4*(d5+d6*r4)))))
            !
                              PVC=U6R+AIPSVC(0)+U2*(AIPSVC(1)+U2*(AIPSVC(2)+U2*(AIPSVC(3) &
                                    +U2*(AIPSVC(4)+U4*(AIPSVC(5)+U4*AIPSVC(6))))))-PIPSVCC
                              DPVC=-SIX*U6R+U2*(BIPSVC(1)+U2*(BIPSVC(2)+U2*(BIPSVC(3) &
                                    +U2*(BIPSVC(4)+U4*(BIPSVC(5)+U4*BIPSVC(6))))))
            !  L-J r12 term
            !   etr12=1/r12+a0+r2*(a1+r2*(a2+r2*(a3+r4*(a4+r4*(a5+a6*r4)))))
            !   detr12/dr*r1=-6/r6+r2*(d1+r2*(d2+r2*(d3+r4*(d4+r4*(d5+d6*r4)))))
            !
                              PVA=U12R+AIPSVA(0)+U2*(AIPSVA(1)+U2*(AIPSVA(2)+U2*(AIPSVA(3) &
                                    +U4*(AIPSVA(4)+U4*(AIPSVA(5)+U4*AIPSVA(6))))))-PIPSVAC
                              DPVA=-TWELVE*U12R+U2*(BIPSVA(1)+U2*(BIPSVA(2)+U2*(BIPSVA(3) &
                                    +U4*(BIPSVA(4)+U4*(BIPSVA(5)+U4*BIPSVA(6))))))
                              ENEVA =CCNBA(IVECT)*RIPS12R
                              ENEVC =-CCNBB(IVECT)*RIPS6R
                              ENBPR=ENEVA*PVA+ENEVC*PVC
                              DFVDW=(ENEVA*DPVA+ENEVC*DPVC)*R2
                           ELSE
                              ENBPR=ZERO
                              DFVDW=ZERO
                           ENDIF
                        ELSE
#endif 
                           IF (.NOT.QETEN .AND. .NOT.QETSR) THEN
#if KEY_WCA==1
                              IF (CCNBA(IVECT)  ==  0.0) THEN
                                 VDWWL = 0.0
                                 RMIN6 = 0.0
                              ELSE
                                 VDWWL=PT25*CCNBB(IVECT)*CCNBB(IVECT)/CCNBA(IVECT)
                                 RMIN6=HALF*CCNBB(IVECT)/VDWWL
                              ENDIF
                              IF (S*S*S  >  RMIN6) THEN
                                 CA=CCNBA(IVECT)*TR6*TR6
                                 ENBPR=CA-CCNBB(IVECT)*TR6
                                 EGLJTMP = MINSIX*(ENBPR+CA)*R2
                                 EPRPL = 0.0D0
                                 EGLJRPL = 0.0D0
                              ELSE
                                 ENBPR=-VDWWL
                                 EGLJTMP = 0.0D0
                                 IF (LLSOFT) THEN
                                    RMIN2=RMIN6**(1.0D0/3.0D0)
                                    TMP = RMIN2*(1.0-SCVDWCUTR)*(1.0-SCVDWCUTR)
                                    IF (S  <=  (RMIN2 - TMP)) THEN
                                       R2 = 1.0D0/(S + TMP)
                                       TR6 = R2 * R2 * R2
                                       CA = CCNBA(IVECT)*TR6*TR6
                                       EPRPL=CA-CCNBB(IVECT)*TR6
                                       EGLJRPL=MINSIX*R2*(EPRPL+CA)
                                       EPRPL=EPRPL+VDWWL
                                    ELSE
                                       EPRPL=0.0D0
                                       EGLJRPL=0.0D0
                                    ENDIF
                                 ELSE
                                    CA=CCNBA(IVECT)*TR6*TR6
                                    EPRPL=CA-CCNBB(IVECT)*TR6
                                    EGLJRPL=MINSIX*R2*(EPRPL+CA)
                                    EPRPL=EPRPL+VDWWL
                                 ENDIF
                              ENDIF
                              DFVDW = EGLJTMP*WCA(I)*WCA(J)+EGLJRPL
                              ENBPR = (ENBPR*WCA(I)*WCA(J)+EPRPL)
#else /**/
                              CA=CCNBA(IVECT)*TR6*TR6
                              CB=CCNBB(IVECT)*TR6
                              ENBPR=CA-CB
                              DFVDW=-SIX*R2*(ENBPR+CA)
#endif 
                           ELSE IF (QETSR) THEN
                              TTPW1 = EIGHT*FOUR
                              TTPW2 = NINE*NINE*NINE
                              TTP12 = TTPW1*TTPW1/(TTPW2*TTPW2)
                              SWTMP = TTP12*CCNBB(IVECT)*CCNBB(IVECT) &
                                   /(TR6*TR6*CCNBA(IVECT)*CCNBA(IVECT))

                              CA=THIRTN*CCNBA(IVECT)*TR6*TR6
                              ENBPR = (CA-NINE*CCNBB(IVECT)* &
                                   ((TWO*CCNBA(IVECT)/CCNBB(IVECT))**(TWO/THREE))* &
                                   TR6*R2*R2+TWO*CCNBB(IVECT)*TR6 ) &
                                   / ( ONE + SWTMP)
                              DFVDW=R2*MINTWO*((78*CCNBA(IVECT)*TR6*TR6)- &
                                   (45*CCNBB(IVECT)* &
                                   ((TWO*CCNBA(IVECT)/CCNBB(IVECT))**(TWO/THREE))* &
                                   TR6*R2*R2)+(SIX*CCNBB(IVECT)*TR6)) &
                                   / ( ONE + SWTMP) &
                                   - R2*ENBPR*TWELVE*SWTMP/((ONE+SWTMP)*(ONE+SWTMP))
                           ELSE  ! QETEN
                              CA=THIRTN*CCNBA(IVECT)*TR6*TR6
                              ENBPR=CA-NINE*CCNBB(IVECT)* &
                                    ((TWO*CCNBA(IVECT)/CCNBB(IVECT)) &
                                    **(TWO/THREE))* &
                                    TR6*R2*R2+TWO*CCNBB(IVECT)*TR6
                              DFVDW=R2*MINTWO*((78*CCNBA(IVECT)*TR6*TR6)- &
                                    (45*CCNBB(IVECT)* &
                                    ((TWO*CCNBA(IVECT)/CCNBB(IVECT)) &
                                    **(TWO/THREE))* &
                                    TR6*R2*R2)+(SIX*CCNBB(IVECT)*TR6))
                           ENDIF
                        ENDIF  
                     ENDIF

#if KEY_MTS==1
!---- LONG-SHORT RANGE MTS METHOD
                     IF(SLFG) THEN
                        EELPR=EELPR*SWFE
                        ENBPR=ENBPR*SWFE
                        DFELEC=DFELEC*SWF
                        DFVDW=DFVDW*SWF
                     ENDIF
#endif 
!.. RHS 4-7-98 (B980630.rs2)
#if KEY_BLOCK==1 /*abig_block*/
                     IF (QBLOCK) THEN
                        IBL=IBLOCK(I)
                        JBL=IBLOCK(J)
#if KEY_DOCK==1
!                 get asymmetric matrix coefficient
                        DOCFI = 1.0
                        DOCFJ = 1.0
                        IF(QDOCK) THEN
                           KDOC  = (IBL - 1)*NBLOCK + JBL
                           DOCFI = BLDOCP(KDOC)
                           KDOC  = (JBL - 1)*NBLOCK + IBL
                           DOCFJ = BLDOCP(KDOC)
                        ENDIF
#endif /*  DOCK*/
                        KK=MAX(IBL,JBL)
                        KK=KK*(KK-1)/2+MIN(IBL,JBL)
                        IF (QPRNTV) THEN
                           IF (IBL == 1 .OR. JBL == 1 .OR. IBL == JBL) THEN
                              VBELEC(MAX(IBL,JBL)) = VBELEC(MAX(IBL,JBL)) + EELPR
                              VBVDW(MAX(IBL,JBL)) = VBVDW(MAX(IBL,JBL)) + ENBPR
                           ENDIF
                        ENDIF
!ldm
                        IF(QLDM .or. QLMC) THEN
                           JLDM = MAX(IBL,JBL)
                           ILDM = MIN(IBL,JBL)
! first row or diagonal elements exclude (1,1).
                           IF((ILDM >= LSTRT.AND.ILDM == JLDM).OR. &
                                 (ILDM == 1.AND. JLDM >= LSTRT)) THEN
                              FALPHA = (ENBPR + EELPR)
                              LAGMUL = LAGMUL + FALPHA
                              BIFLAM(JLDM) = BIFLAM(JLDM) + FALPHA
                           ENDIF
                           IF(NRST == 2)THEN
                              IF((ILDM >= LSTRT.AND.ILDM == JLDM).OR. &
                                    (ILDM == 2.AND.JLDM >= LSTRT)) THEN
                                 FALPHA=ENBPR + EELPR
                                 BFRST(JLDM) = BFRST(JLDM) + FALPHA
                              ENDIF
                           ENDIF
                        ENDIF
                        DF=DFELEC+DFVDW
                        IF(NRST == 1)THEN
!  BETWEEN PROTEIN(I) AND LIGAND(J)
                           IF(IBL == 1.AND.JBL >= LSTRT) THEN
! ADD FORCE TO I(PROTEIN)
                              ENVDX((JBL-2)*NATOMX+I) = ENVDX((JBL-2)*NATOMX+I) + DXI * DF
                              ENVDY((JBL-2)*NATOMX+I) = ENVDY((JBL-2)*NATOMX+I) + DYI * DF
                              ENVDZ((JBL-2)*NATOMX+I) = ENVDZ((JBL-2)*NATOMX+I) + DZI * DF
! ADD FORCE TO J(LIGAND)
                              ENVDX(J) = ENVDX(J) - DXI * DF
                              ENVDY(J) = ENVDY(J) - DYI * DF
                              ENVDZ(J) = ENVDZ(J) - DZI * DF
!  BETWEEN LIGAND(I) AND PROTEIN(J)
                           ELSE IF(JBL == 1.AND.IBL >= LSTRT) THEN
! ADD FORCE TO J(PROTEIN)
                              ENVDX((IBL-2)*NATOMX+J) = ENVDX((IBL-2)*NATOMX+J) - DXI * DF
                              ENVDY((IBL-2)*NATOMX+J) = ENVDY((IBL-2)*NATOMX+J) - DYI * DF
                              ENVDZ((IBL-2)*NATOMX+J) = ENVDZ((IBL-2)*NATOMX+J) - DZI * DF
! ADD FORCE TO I(LIGAND)
                              ENVDX(I) = ENVDX(I) + DXI * DF
                              ENVDY(I) = ENVDY(I) + DYI * DF
                              ENVDZ(I) = ENVDZ(I) + DZI * DF
!  WITHIN LIGAND
                           ELSE IF(IBL >= LSTRT.AND.IBL == JBL) THEN
                              ENVDX(I) = ENVDX(I) + DXI * DF
                              ENVDY(I) = ENVDY(I) + DYI * DF
                              ENVDZ(I) = ENVDZ(I) + DZI * DF
                              ENVDX(J) = ENVDX(J) - DXI * DF
                              ENVDY(J) = ENVDY(J) - DYI * DF
                              ENVDZ(J) = ENVDZ(J) - DZI * DF
                           ENDIF
                        ELSE IF(NRST == 2)THEN
                           IF(IBL >= LSTRT.AND.JBL == 2)THEN
                              ENVDX(I) = ENVDX(I) + DXI * DF
                              ENVDY(I) = ENVDY(I) + DYI * DF
                              ENVDZ(I) = ENVDZ(I) + DZI * DF
                           ELSE IF(JBL >= LSTRT.AND.IBL == 2)THEN
                              ENVDX(J) = ENVDX(J) - DXI * DF
                              ENVDY(J) = ENVDY(J) - DYI * DF
                              ENVDZ(J) = ENVDZ(J) - DZI * DF
                           ELSE IF(IBL >= LSTRT.AND.IBL == JBL) THEN
                              ENVDX(I) = ENVDX(I) + DXI * DF
                              ENVDY(I) = ENVDY(I) + DYI * DF
                              ENVDZ(I) = ENVDZ(I) + DZI * DF
                              ENVDX(J) = ENVDX(J) - DXI * DF
                              ENVDY(J) = ENVDY(J) - DYI * DF
                              ENVDZ(J) = ENVDZ(J) - DZI * DF
                           ENDIF
                        ELSE IF(NRST == 3)THEN
!  Add force only to ligand
!  Protein(I) AND Ligand(J)
                           IF(IBL == 1.AND.JBL >= LSTRT) THEN
                              ENVDX(J) = ENVDX(J) - DXI * DF
                              ENVDY(J) = ENVDY(J) - DYI * DF
                              ENVDZ(J) = ENVDZ(J) - DZI * DF
!  Protein(J) AND Ligand(I)
                           ELSE IF(JBL == 1.AND.IBL >= LSTRT) THEN
                              ENVDX(I) = ENVDX(I) + DXI * DF
                              ENVDY(I) = ENVDY(I) + DYI * DF
                              ENVDZ(I) = ENVDZ(I) + DZI * DF
!  Within Ligand(I)
                           ELSE IF(IBL >= LSTRT.AND.IBL == JBL) THEN
                              ENVDX(I) = ENVDX(I) + DXI * DF
                              ENVDY(I) = ENVDY(I) + DYI * DF
                              ENVDZ(I) = ENVDZ(I) + DZI * DF
                              ENVDX(J) = ENVDX(J) - DXI * DF
                              ENVDY(J) = ENVDY(J) - DYI * DF
                              ENVDZ(J) = ENVDZ(J) - DZI * DF
                           ENDIF
                        ENDIF
! LDM
#if KEY_DOCK==1
                        IF(QDOCK) THEN
                           EELPR=EELPR*BLCOE(KK)*0.5*(DOCFI + DOCFJ)
                           ENBPR=ENBPR*BLCOV(KK)*0.5*(DOCFI + DOCFJ)
                        ELSE
#endif 
                           EELPR=EELPR*BLCOE(KK)
                           ENBPR=ENBPR*BLCOV(KK)
#if KEY_DOCK==1
                        ENDIF
#endif 
                        DF=DFELEC*BLCOE(KK)+DFVDW*BLCOV(KK)
                     ELSE
!.. ADDED RHS 4-7-98 (B980630.rs2)
#endif /* (abig_block)*/
                        DF=DFELEC+DFVDW
!.. ADDED RHS 4-7-98 (B980630.rs2)
#if KEY_BLOCK==1 /*bbig_block*/
                     ENDIF

                     IF (.NOT. NOFORC) THEN
!.. RHS 4-7-98 (B980630.rs2)
#endif /* (bbig_block)*/
                        DXIT=DXI*DF
                        DYIT=DYI*DF
                        DZIT=DZI*DF
#if KEY_BLOCK==1 /*middle_block*/
#if KEY_DOCK==1
                        IF(QDOCK) THEN
                           DX(I)=DX(I)+DXIT*DOCFI
                           DY(I)=DY(I)+DYIT*DOCFI
                           DZ(I)=DZ(I)+DZIT*DOCFI
                           DX(J)=DX(J)-DXIT*DOCFJ
                           DY(J)=DY(J)-DYIT*DOCFJ
                           DZ(J)=DZ(J)-DZIT*DOCFJ
                        ELSE
#endif /* dock*/
#endif /* (middle_block)*/
                           DX(I)=DX(I)+DXIT
                           DY(I)=DY(I)+DYIT
                           DX(J)=DX(J)-DXIT
                           DY(J)=DY(J)-DYIT
#if KEY_ASPENER==1
                           IF (LRSW .AND. LMEMBR) THEN
! pore model
                              IF (LPOR) THEN
                                 DX(I)=DX(I)+DFZ*LAM(J)*DFDX(I)
                                 DX(J)=DX(J)+DFZ*LAM(I)*DFDX(J)
                                 DY(I)=DY(I)+DFZ*LAM(J)*DFDY(I)
                                 DY(J)=DY(J)+DFZ*LAM(I)*DFDY(J)
                              ENDIF
! curved membrane
                              IF (LCURV) THEN
                                 DZ(I)=DZ(I)+DZIT
                                 DZ(J)=DZ(J)-DZIT
                                 IF (LTUB) THEN
                                    TMPR1= SQRT(Y(I)**2+Z(I)**2)                            
                                    TMPR2= SQRT(Y(J)**2+Z(J)**2)
                                    DK1= DFZ*LAM(J)*DFDZ(I)
                                    DY(I)=DY(I)+ DK1*Y(I)/TMPR1
                                    DZ(I)=DZ(I)+ DK1*Z(I)/TMPR1
                                    DK2= DFZ*LAM(I)*DFDZ(J)
                                    DY(J)=DY(J)+ DK2*Y(J)/TMPR2
                                    DZ(J)=DZ(J)+ DK2*Z(J)/TMPR2                                    
                                 ELSE 
                                    TMPR1= SQRT(X(I)**2+Y(I)**2+Z(I)**2)
                                    TMPR2= SQRT(X(J)**2+Y(J)**2+Z(J)**2)
                                    DK1= DFZ*LAM(J)*DFDZ(I)
                                    DX(I)=DX(I)+ DK1*X(I)/TMPR1                                    
                                    DY(I)=DY(I)+ DK1*Y(I)/TMPR1
                                    DZ(I)=DZ(I)+ DK1*Z(I)/TMPR1
                                    DK2= DFZ*LAM(I)*DFDZ(J)
                                    DX(J)=DX(J)+ DK2*X(J)/TMPR2                                    
                                    DY(J)=DY(J)+ DK2*Y(J)/TMPR2
                                    DZ(J)=DZ(J)+ DK2*Z(J)/TMPR2
                                  ENDIF
                              ELSE
                              DZ(I)=DZ(I)+DZIT+DFZ*LAM(J)*DFDZ(I)
                              DZ(J)=DZ(J)-DZIT+DFZ*LAM(I)*DFDZ(J)
                              ENDIF
                           ELSE
#endif 
                              DZ(I)=DZ(I)+DZIT
                              DZ(J)=DZ(J)-DZIT
#if KEY_ASPENER==1
                           ENDIF  
#endif
#if KEY_BLOCK==1
#if KEY_DOCK==1
                        ENDIF
#endif 
                     ENDIF
#endif 

#if KEY_FLUCQ==1
! Add in electrostatic energy for this pair to FlucQ arrays
                     IF (QFLUC) THEN
                        FQCFOR(I)=FQCFOR(I)+EELPR
                        FQCFOR(J)=FQCFOR(J)+EELPR
                     ENDIF
#endif 
!     Cummulate van der Waals and electrostatic energies
                     ETEMP1=ETEMP1+ENBPR
                     ETEMP2=ETEMP2+EELPR
                  ENDDO loop160
               ENDDO loop170

            ELSE
#if KEY_NBIPS==1
               IF(QIPS)THEN
                  SCENT=ZERO
               ELSE
#endif 

!     Do switching function interaction
                  DXIC=XCENT(IRS)-XCENT(JRS)
                  DYIC=YCENT(IRS)-YCENT(JRS)
                  DZIC=ZCENT(IRS)-ZCENT(JRS)
                  SCENT=DXIC*DXIC+DYIC*DYIC+DZIC*DZIC
                  FUNCT=ONE
#if KEY_NBIPS==1
               ENDIF  
#endif
! modification by EPO 7/10/2006
               if(qvarcut) then
                  IF(VARCUT(IS)+VARCUT(JS) > 0) THEN
!     use the variable cutoff method to define a new LJ cutoff
!     define a new cutoff that is dependent on the atom type
!     (use scalar command to define varcut)
                     CDIFF=CTOFNB-CTONNB
                     C2OFNB=(VARCUT(IS)+VARCUT(JS))/2 + CDIFF
!     get the square of it
                     C2OFNB=C2OFNB*C2OFNB
                     C2ONNB=(VARCUT(IS)+VARCUT(JS))/2
                     C2ONNB=C2ONNB*C2ONNB
                     IF (CTOFNB > CTONNB) THEN
                        RUL3=ONE/(C2OFNB-C2ONNB)**3
                        RUL12=RUL3*TWELVE
                     ENDIF
                  ENDIF
!                  ELSE
!     otherwise use the original values
!yw              C2ONNB=CTONNB*CTONNB
!yw              C2OFNB=CTOFNB*CTOFNB
!yw              CDIFF=CTOFNB-CTONNB
!yw              IF (CTOFNB > CTONNB) THEN
!yw                RUL3=ONE/(C2OFNB-C2ONNB)**3
!yw                RUL12=RUL3*TWELVE
!yw              ENDIF
               endif

               IF(SCENT < C2OFNB) THEN
#if KEY_MTS==1
                  IF(SLFG) THEN
                     IF((SLFG1.AND.(SCENT > RSCUT2))) CYCLE loop370
                     IF((SLFG2.AND.(SCENT < RSHL2))) CYCLE loop370
                  ENDIF
#endif 
                  LSWITR=(SCENT > C2ONNB)
                  IF(LSWITR) THEN
                     RIJL=C2ONNB-SCENT
                     RIJU=C2OFNB-SCENT
                     FUNCT=RIJU*RIJU*(RIJU-3*RIJL)*RUL3
                     DFN=RIJL*RIJU*RUL12
                  ENDIF

                  IF (LFSWT) THEN
                     ETEMP2=-QCENT(IRS)*QCENT(JRS)
#if KEY_BLOCK==1 /*bl_1*/
! Assume all atoms of a group are in the same block.
                     IF (QBLOCK) THEN
                        IBL=IBLOCK(IS)
                        JBL=IBLOCK(JS)
                        KK=MAX(IBL,JBL)
                        KK=KK*(KK-1)/2+MIN(IBL,JBL)
                        IF (QPRNTV) THEN
                           IF (IBL == 1 .OR. JBL == 1 .OR. IBL == JBL) THEN
                              VBELEC(MAX(IBL,JBL)) = VBELEC(MAX(IBL,JBL)) + ETEMP2
                           ENDIF
                        ENDIF
 !ldm
                        IF(QLDM .or. QLMC) THEN
                           JLDM = MAX(IBL,JBL)
                           ILDM = MIN(IBL,JBL)
!  first row or diagonal elements exclude (1,1).
                           IF((ILDM >= LSTRT.AND.ILDM == JLDM).OR. &
                                 (ILDM == 1.AND. JLDM >= LSTRT)) THEN
                              FALPHA = ETEMP2
                              LAGMUL = LAGMUL + FALPHA
                              BIFLAM(JLDM) = BIFLAM(JLDM) + FALPHA
                           ENDIF
                           IF(NRST == 2)THEN
                              IF((ILDM >= LSTRT.AND.ILDM == JLDM).OR. &
                                    (ILDM == 2.AND.JLDM >= LSTRT)) THEN
                                 FALPHA=ETEMP2
                                 BFRST(JLDM) = BFRST(JLDM) + FALPHA
                              ENDIF
                           ENDIF
                        ENDIF
! LDM
                        ETEMP2=ETEMP2*BLCOE(KK)
                     ENDIF
#endif /* (bl_1)*/
                  ENDIF

                  loop220: DO I=IS,IQ
#if KEY_GCMC==1
                     if(qgcmc) then
                        IF (.NOT. GCMCON(I)) CYCLE
                     endif
#endif 
                     CGT=CGF*CGX(I)
                     IACI=IACNB(I)
                     loop210: DO J=JS,JQ
#if KEY_GCMC==1
                        if(qgcmc) then
                           IF (.NOT. GCMCON(J)) CYCLE
                        endif
#endif 

! namkh 01/20/04
#if KEY_GAMESS==1 || KEY_GAMESSUK==1 || KEY_MNDO97==1 || KEY_SQUANTM==1 || KEY_QCHEM==1 || KEY_QTURBO==1 || KEY_G09==1
                        if(qmused)then
                        IF((ABS(IGMSEL(I)) == 1.OR.ABS(IGMSEL(I)) == 2) .AND. &
                              (ABS(IGMSEL(J)) == 1.OR.ABS(IGMSEL(J)) == 2) &
                              .AND.QGMREM) CYCLE
                        endif
#endif 

! T.S.
                        DFELEC=0.
                        DFVDW=0.

                        DXI=X(I)-X(J)
                        DYI=Y(I)-Y(J)
                        DZI=Z(I)-Z(J)
                        S=MAX(RSMALL,DXI*DXI+DYI*DYI+DZI*DZI)
!CCCC                  S=DXI*DXI+DYI*DYI+DZI*DZI
                        R2=ONE/S
#if KEY_NBIPS==1
                        IF(QIPS)THEN
!WXW Long range potential using isotropic periodic sum (IPS)
                           U2=S*RIPS2R
                        ENDIF
                        IF(LEIPSX)THEN
!WXW Electrostatic potential using isotropic periodic sum (IPS)
                           IF(U2 < ONE)THEN
                              IF(LCONS)THEN
!  Electrostatic IPS
!   etr1=1/r+r2*(a1+r2*(a2+r2*(a3+r2*(a4+r2*(a5+a6*r2)))))
!   detr1/dr*r=-1/r+r2*(d1+r2*(d2+r2*(d3+r2*(d4+r2*(d5+d6*r2)))))
!
                                 U1=SQRT(U2)
                                 PE=ONE/U1+U2*(AIPSE(1)+U2*(AIPSE(2)+U2*(AIPSE(3) &
                                       +U2*(AIPSE(4)+U2*(AIPSE(5)+U2*AIPSE(6))))))
                                 DPE=-ONE/U1+U2*(BIPSE(1)+U2*(BIPSE(2)+U2*(BIPSE(3) &
                                       +U2*(BIPSE(4)+U2*(BIPSE(5)+U2*BIPSE(6))))))
                                 ENEP=CGT*CGX(J)*RIPSR
                                 EELPR=ENEP*(PE-PIPSEC)
                                 DFELEC=ENEP*DPE*R2
#if KEY_lsecd==1 /*lsecd2*/
                          IF(LSECD) THEN
                             DDPE=TWO/U1+U2*(BBIPSE(1)+U2*(BBIPSE(2)+U2*(BBIPSE(3) &
                               +U2*(BBIPSE(4)+U2*(BBIPSE(5)+U2*BBIPSE(6))))))
                             DDF=ENEP*DDPE*R2*R2
                          ENDIF
#endif /*  (lsecd2)*/
                              ELSE
!  Electrostatic IPS
!   etr1=1/r2+r2*(a1+r2*(a2+r2*(a3+r2*(a4+r2*(a5+a6*r2)))))
!   detr1/dr*r1=-1/r2+r2*(d1+r2*(d2+r2*(d3+r2*(d4+r2*(d5+d6*r2)))))
!
                                 PE=ONE/U2+U2*(AIPSE(1)+U2*(AIPSE(2)+U2*(AIPSE(3) &
                                       +U2*(AIPSE(4)+U2*(AIPSE(5)+U2*AIPSE(6))))))-PIPSEC
                                 DPE=-TWO/U2+U2*(BIPSE(1)+U2*(BIPSE(2)+U2*(BIPSE(3) &
                                       +U2*(BIPSE(4)+U2*(BIPSE(5)+U2*BIPSE(6))))))
                                 ENEP=CGT*CGX(J)*RIPS2R
                                 EELPR=ENEP*PE
                                 DFELEC=ENEP*DPE*R2
#if KEY_lsecd==1 /*lsecd2*/
                          IF(LSECD) THEN
                             DDPE=SIX/U2+U2*(BBIPSE(1)+U2*(BBIPSE(2)+U2*(BBIPSE(3) &
                               +U2*(BBIPSE(4)+U2*(BBIPSE(5)+U2*BBIPSE(6))))))
                             DDF=ENEP*DDPE*R2*R2
                          ENDIF
#endif /*  (lsecd2)*/
                              ENDIF
                           ELSE
                              EELPR=ZERO
                           ENDIF
                        ELSE
#endif 

#if KEY_ASPENER==1
                           IF (LRSW .AND. LMEMBR) THEN
                              LAMD=SQRT(LAM(I)*LAM(J))
                              IR2=SQRT(R2)
                              EELPR=CGT*CGX(J)*IR2*IR2**LAMD
                              DFELEC=-(ONE+LAMD)*R2*EELPR
                              DFZ= EELPR*LOG(IR2)*(ONE-AEMPIR)/TWO/LAMD
                           ELSE IF (LRSW .AND. LDEBYE) THEN
                              IR2=SQRT(R2)
                              EELPR=CGT*CGX(J)*R2*EXP(-1./IR2/RDEBYE)
                              DFELEC=(R2*MINTWO-IR2/RDEBYE)*EELPR
                           ELSE IF (LRSW) THEN
#else /**/
                           IF (LRSW) THEN
#endif /*  ASPENER*/
                              EELPR=CGT*CGX(J)*R2
                              DFELEC=R2*MINTWO*EELPR
                           ELSE
                              EELPR=CGT*CGX(J)*SQRT(R2)
                              DFELEC=-R2*EELPR
                           ENDIF
#if KEY_NBIPS==1
                        ENDIF  
#endif

                        IF(LVGRP) THEN
                           IVECT=LOWTP(MAX(IACNB(J),IACI))+IACNB(J)+IACI
                           TR6=R2*R2*R2
                           SIG=CCNBB(IVECT)
                           IF (SIG<RSMALL) THEN
                              ENBPR=ZERO 
                              DFVDW=ZERO 
                           ELSE IF (DBEXP) THEN
                              IF(S < C2OFNB)THEN
                                 EMIN=SIG*SIG/CCNBA(IVECT)/FOUR
                                 SIG=(HALF*SIG*TR6/EMIN)**SIXTH
#if KEY_NBIPS==1
                                 AKR=AIPSVA(0)/SIG
                                 EXPAR=EXP(-AKR)
                                 ! energy
                                 ENEVA=AIPSVA(1)*EXPAR
                                 ! force
                                 DPVA=BIPSVA(1)*EXPAR/SIG
                                 !
                                 CKR=AIPSVC(0)/SIG
                                 EXPCR=EXP(-CKR)
                                 ! energy
                                 ENEVC=AIPSVC(1)*EXPCR
                                 ! force
                                 DPVC=BIPSVC(1)*EXPCR/SIG
                                 ENBPR=EMIN*(ENEVA-ENEVC)
                                 DFVDW=EMIN*(DPVA-DPVC)*R2
#endif /* KEY_NBIPS */
                              ELSE
                                 ENBPR=ZERO
                                 DFVDW=ZERO
                              ENDIF
#if KEY_NBIPS==1
                        ELSE IF(LVIPSX)THEN
!WXW VDW potential using isotropic periodic sum (IPS)
                           IF(U2 < ONE)THEN
                              U4=U2*U2
                              U6R=ONE/U4/U2
                              U12R=U6R*U6R
!  L-J r6 term
!   etr6=1/r6+a0+r2*(a1+r2*(a2+r2*(a3+r2*(a4+r4*(a5+a6*r4)))))
!   detr6/dr*r1=-6/r6+r2*(d1+r2*(d2+r2*(d3+r2*(d4+r4*(d5+d6*r4)))))
!
                              PVC=U6R+AIPSVC(0)+U2*(AIPSVC(1)+U2*(AIPSVC(2)+U2*(AIPSVC(3) &
                                    +U2*(AIPSVC(4)+U4*(AIPSVC(5)+U4*AIPSVC(6))))))-PIPSVCC
                              DPVC=-SIX*U6R+U2*(BIPSVC(1)+U2*(BIPSVC(2)+U2*(BIPSVC(3) &
                                    +U2*(BIPSVC(4)+U4*(BIPSVC(5)+U4*BIPSVC(6))))))
!  L-J r12 term
!   etr12=1/r12+a0+r2*(a1+r2*(a2+r2*(a3+r4*(a4+r4*(a5+a6*r4)))))
!   detr12/dr*r1=-6/r6+r2*(d1+r2*(d2+r2*(d3+r4*(d4+r4*(d5+d6*r4)))))
!
                              PVA=U12R+AIPSVA(0)+U2*(AIPSVA(1)+U2*(AIPSVA(2)+U2*(AIPSVA(3) &
                                    +U4*(AIPSVA(4)+U4*(AIPSVA(5)+U4*AIPSVA(6))))))-PIPSVAC
                              DPVA=-TWELVE*U12R+U2*(BIPSVA(1)+U2*(BIPSVA(2)+U2*(BIPSVA(3) &
                                    +U4*(BIPSVA(4)+U4*(BIPSVA(5)+U4*BIPSVA(6))))))
                              ENEVA =CCNBA(IVECT)*RIPS12R
                              ENEVC =-CCNBB(IVECT)*RIPS6R
                              ENBPR=ENEVA*PVA+ENEVC*PVC
                              DFVDW=(ENEVA*DPVA+ENEVC*DPVC)*R2
                           ELSE
                              ENBPR=ZERO
                              DFVDW=ZERO
                           ENDIF 
                        ELSE IF (DBEXPIPS) THEN
                           EMIN=SIG*SIG/CCNBA(IVECT)/FOUR
                           SIG=(HALF*SIG*TR6/EMIN)**SIXTH
                           IF(SRIPS*SIG>ONE)THEN
                               AKR=AIPSVA(0)/SIG
                               EXPAR=EXP(-AKR)
                               !AKRC=AIPSVA(0)*RIPS/SIG
                               !EXPARC=EXP(-AKRC)
                               ! energy
                               !ENEVA=AIPSVA(1)*(EXPAR+EXPARC*(EXPARC/EXPAR+(TWELVE+SIX*EXPARC/AKRC)/AKRC/AKRC))
                               ENEVA=AIPSVA(1)*(EXPAR+AIPSVA(4)/EXPAR+AIPSVA(5))
                               ! force 
                               !DPVA=BIPSVA(1)*(EXPAR-EXPARC*EXPARC/EXPAR)/SIG
                               DPVA=BIPSVA(1)*(EXPAR-AIPSVA(4)/EXPAR)/SIG
                               !
                               CKR=AIPSVC(0)/SIG
                               EXPCR=EXP(-CKR)
                               !CKRC=AIPSVC(0)*RIPS/SIG
                               !EXPCRC=EXP(-CKRC)
                               ! energy
                               !ENEVC=AIPSVC(1)*(EXPCR+EXPCRC*(EXPCRC/EXPCR+(TWELVE+SIX*EXPCRC/CKRC)/CKRC/CKRC))
                               ENEVC=AIPSVC(1)*(EXPCR+AIPSVC(4)/EXPCR+AIPSVC(5))
                               ! force
                               !DPVC=BIPSVC(1)*(EXPCR-EXPCRC*EXPCRC/EXPCR)/SIG
                               DPVC=BIPSVC(1)*(EXPCR-AIPSVC(4)/EXPCR)/SIG
                               ENBPR=EMIN*(ENEVA-ENEVC)
                               DFVDW=EMIN*(DPVA-DPVC)*R2
                           ELSE 
                               ENBPR=ZERO 
                               DFVDW=ZERO 
                           ENDIF
#endif 
                        ELSE
                              IF (.NOT.QETEN .AND. .NOT.QETSR) THEN
#if KEY_WCA==1
                                 IF (CCNBA(IVECT)  ==  0.0) THEN
                                    VDWWL = 0.0
                                    RMIN6 = 0.0
                                 ELSE
                                    VDWWL=PT25*CCNBB(IVECT)*CCNBB(IVECT)/CCNBA(IVECT)
                                    RMIN6=HALF*CCNBB(IVECT)/VDWWL
                                 ENDIF
                                 IF (S*S*S  >  RMIN6) THEN
                                    CA=CCNBA(IVECT)*TR6*TR6
                                    ENBPR=CA-CCNBB(IVECT)*TR6
                                    EGLJTMP = MINSIX*(ENBPR+CA)*R2
                                    EPRPL = 0.0D0
                                    EGLJRPL = 0.0D0
                                 ELSE
                                    ENBPR=-VDWWL
                                    EGLJTMP = 0.0D0
                                    IF (LLSOFT) THEN
                                       RMIN2=RMIN6**(1.0D0/3.0D0)
                                       TMP = RMIN2*(1.0-SCVDWCUTR)*(1.0-SCVDWCUTR)
                                       IF (S  <=  (RMIN2 - TMP)) THEN
                                          R2 = 1.0D0/(S + TMP)
                                          TR6 = R2 * R2 * R2
                                          CA = CCNBA(IVECT)*TR6*TR6
                                          EPRPL=CA-CCNBB(IVECT)*TR6
                                          EGLJRPL=MINSIX*R2*(EPRPL+CA)
                                          EPRPL=EPRPL+VDWWL
                                       ELSE
                                          EPRPL=0.0D0
                                          EGLJRPL=0.0D0
                                       ENDIF
                                    ELSE
                                       CA=CCNBA(IVECT)*TR6*TR6
                                       EPRPL=CA-CCNBB(IVECT)*TR6
                                       EGLJRPL=MINSIX*R2*(EPRPL+CA)
                                       EPRPL=EPRPL+VDWWL
                                    ENDIF
                                 ENDIF
                                 DFVDW = EGLJTMP*WCA(I)*WCA(J)+EGLJRPL
                                 ENBPR = (ENBPR*WCA(I)*WCA(J)+EPRPL)
#else /**/
                                 CA=CCNBA(IVECT)*TR6*TR6
                                 CB=CCNBB(IVECT)*TR6
                                 ENBPR=CA-CB
                                 DFVDW=-SIX*R2*(ENBPR+CA)
#endif 
                              ELSE IF (QETSR) THEN
                                 TTPW1 = EIGHT*FOUR
                                 TTPW2 = NINE*NINE*NINE
                                 TTP12 = TTPW1*TTPW1/(TTPW2*TTPW2)
                                 SWTMP = TTP12*CCNBB(IVECT)*CCNBB(IVECT) &
                                      /(TR6*TR6*CCNBA(IVECT)*CCNBA(IVECT))
                                 
                                 CA=THIRTN*CCNBA(IVECT)*TR6*TR6
                                 ENBPR = (CA-NINE*CCNBB(IVECT)* &
                                      ((TWO*CCNBA(IVECT)/CCNBB(IVECT))**(TWO/THREE))* &
                                      TR6*R2*R2+TWO*CCNBB(IVECT)*TR6 ) &
                                      / ( ONE + SWTMP)
                                 DFVDW=R2*MINTWO*((78*CCNBA(IVECT)*TR6*TR6)- &
                                      (45*CCNBB(IVECT)* &
                                      ((TWO*CCNBA(IVECT)/CCNBB(IVECT))**(TWO/THREE))* &
                                      TR6*R2*R2)+(SIX*CCNBB(IVECT)*TR6)) &
                                      / ( ONE + SWTMP) &
                                      - R2*ENBPR*TWELVE*SWTMP/((ONE+SWTMP)*(ONE+SWTMP))
                              ELSE ! ETEN
                                 CA=THIRTN*CCNBA(IVECT)*TR6*TR6
                                 ENBPR = CA-NINE*CCNBB(IVECT)* &
                                      ((TWO*CCNBA(IVECT)/CCNBB(IVECT)) &
                                      **(TWO/THREE))* &
                                      TR6*R2*R2+TWO*CCNBB(IVECT)*TR6
                                 DFVDW=R2*MINTWO*((78*CCNBA(IVECT)*TR6*TR6)- &
                                      (45*CCNBB(IVECT)* &
                                      ((TWO*CCNBA(IVECT)/CCNBB(IVECT)) &
                                      **(TWO/THREE))* &
                                      TR6*R2*R2)+(SIX*CCNBB(IVECT)*TR6))
                              ENDIF
                           ENDIF  
                        ENDIF

#if KEY_MTS==1
!---- LONG-SHORT RANGE MTS METHOD
                        IF(SLFG) THEN
                           EELPR=EELPR*SWFE
                           ENBPR=ENBPR*SWFE
                           DFELEC=DFELEC*SWF
                           DFVDW=DFVDW*SWF
                        ENDIF
#endif 
#if KEY_BLOCK==1 /*bl_a*/
                        IF (QBLOCK) THEN
                           IBL=IBLOCK(I)
                           JBL=IBLOCK(J)
#if KEY_DOCK==1
!                 get asymmetric matrix coefficient
                           DOCFI = 1.0
                           DOCFJ = 1.0
                           IF(QDOCK) THEN
                              KDOC  = (IBL - 1)*NBLOCK + JBL
                              DOCFI = BLDOCP(KDOC)
                              KDOC  = (JBL - 1)*NBLOCK + IBL
                              DOCFJ = BLDOCP(KDOC)
                           ENDIF
#endif /*  DOCK*/
                           KK=MAX(IBL,JBL)
                           KK=KK*(KK-1)/2+MIN(IBL,JBL)
                           IF (QPRNTV) THEN
                              IF (IBL == 1 .OR. JBL == 1 .OR. IBL == JBL) THEN
                                 VBELEC(MAX(IBL,JBL)) = VBELEC(MAX(IBL,JBL)) + EELPR
                                 VBVDW(MAX(IBL,JBL)) = VBVDW(MAX(IBL,JBL)) + ENBPR
                              ENDIF
                           ENDIF
!ldm
                           IF(QLDM .or. QLMC) THEN
                              JLDM = MAX(IBL,JBL)
                              ILDM = MIN(IBL,JBL)
! first row or diagonal elements exclude (1,1).
                              IF((ILDM >= LSTRT.AND.ILDM == JLDM).OR. &
                                    (ILDM == 1.AND.JLDM >= LSTRT)) THEN
                                 FALPHA = (ENBPR + EELPR)
                                 LAGMUL = LAGMUL + FALPHA
                                 BIFLAM(JLDM) = BIFLAM(JLDM) + FALPHA
                              ENDIF
                              IF(NRST == 2)THEN
                                 IF((ILDM >= LSTRT.AND.ILDM == JLDM).OR. &
                                       (ILDM == 2.AND.JLDM >= LSTRT)) THEN
                                    FALPHA=ENBPR + EELPR
                                    BFRST(JLDM) = BFRST(JLDM) + FALPHA
                                 ENDIF
                              ENDIF
                           ENDIF
                           DF=DFELEC+DFVDW
                           IF(NRST == 1)THEN
!  BETWEEN PROTEIN(I) AND LIGAND(J)
                              IF(IBL == 1.AND.JBL >= LSTRT) THEN
! ADD FORCE TO I(PROTEIN)
                                 ENVDX((JBL-2)*NATOMX+I) = ENVDX((JBL-2)*NATOMX+I) + DXI * DF
                                 ENVDY((JBL-2)*NATOMX+I) = ENVDY((JBL-2)*NATOMX+I) + DYI * DF
                                 ENVDZ((JBL-2)*NATOMX+I) = ENVDZ((JBL-2)*NATOMX+I) + DZI * DF
! ADD FORCE TO J(LIGAND)
                                 ENVDX(J) = ENVDX(J) - DXI * DF
                                 ENVDY(J) = ENVDY(J) - DYI * DF
                                 ENVDZ(J) = ENVDZ(J) - DZI * DF
!  BETWEEN LIGAND(I) AND PROTEIN(J)
                              ELSE IF(JBL == 1.AND.IBL >= LSTRT) THEN
! ADD FORCE TO J(PROTEIN)
                                 ENVDX((IBL-2)*NATOMX+J) = ENVDX((IBL-2)*NATOMX+J) - DXI * DF
                                 ENVDY((IBL-2)*NATOMX+J) = ENVDY((IBL-2)*NATOMX+J) - DYI * DF
                                 ENVDZ((IBL-2)*NATOMX+J) = ENVDZ((IBL-2)*NATOMX+J) - DZI * DF
! ADD FORCE TO I(LIGAND)
                                 ENVDX(I) = ENVDX(I) + DXI * DF
                                 ENVDY(I) = ENVDY(I) + DYI * DF
                                 ENVDZ(I) = ENVDZ(I) + DZI * DF
!  WITHIN LIGAND
                              ELSE IF(IBL >= LSTRT.AND.IBL == JBL) THEN
                                 ENVDX(I) = ENVDX(I) + DXI * DF
                                 ENVDY(I) = ENVDY(I) + DYI * DF
                                 ENVDZ(I) = ENVDZ(I) + DZI * DF
                                 ENVDX(J) = ENVDX(J) - DXI * DF
                                 ENVDY(J) = ENVDY(J) - DYI * DF
                                 ENVDZ(J) = ENVDZ(J) - DZI * DF
                              ENDIF
                           ELSE IF(NRST == 2)THEN
                              IF(IBL >= LSTRT.AND.JBL == 2)THEN
                                 ENVDX(I) = ENVDX(I) + DXI * DF
                                 ENVDY(I) = ENVDY(I) + DYI * DF
                                 ENVDZ(I) = ENVDZ(I) + DZI * DF
                              ELSE IF(JBL >= LSTRT.AND.IBL == 2)THEN
                                 ENVDX(J) = ENVDX(J) - DXI * DF
                                 ENVDY(J) = ENVDY(J) - DYI * DF
                                 ENVDZ(J) = ENVDZ(J) - DZI * DF
                              ELSE IF(IBL >= LSTRT.AND.IBL == JBL)THEN
                                 ENVDX(I) = ENVDX(I) + DXI * DF
                                 ENVDY(I) = ENVDY(I) + DYI * DF
                                 ENVDZ(I) = ENVDZ(I) + DZI * DF
                                 ENVDX(J) = ENVDX(J) - DXI * DF
                                 ENVDY(J) = ENVDY(J) - DYI * DF
                                 ENVDZ(J) = ENVDZ(J) - DZI * DF
                              ENDIF
                           ELSE IF(NRST == 3)THEN
! Add force only to Ligand
! Protein(I) & Ligand(J)
                              IF(IBL == 1.AND.JBL >= LSTRT) THEN
                                 ENVDX(J) = ENVDX(J) - DXI * DF
                                 ENVDY(J) = ENVDY(J) - DYI * DF
                                 ENVDZ(J) = ENVDZ(J) - DZI * DF
! Protein(J) & Ligand(I)
                              ELSE IF(JBL == 1.AND.IBL >= LSTRT) THEN
                                 ENVDX(I) = ENVDX(I) + DXI * DF
                                 ENVDY(I) = ENVDY(I) + DYI * DF
                                 ENVDZ(I) = ENVDZ(I) + DZI * DF
! Within Ligand(I)
                              ELSE IF(IBL >= LSTRT.AND.IBL == JBL) THEN
                                 ENVDX(I) = ENVDX(I) + DXI * DF
                                 ENVDY(I) = ENVDY(I) + DYI * DF
                                 ENVDZ(I) = ENVDZ(I) + DZI * DF
                                 ENVDX(J) = ENVDX(J) - DXI * DF
                                 ENVDY(J) = ENVDY(J) - DYI * DF
                                 ENVDZ(J) = ENVDZ(J) - DZI * DF
                              ENDIF
                           ENDIF
! LDM
#if KEY_DOCK==1
                           IF(QDOCK) THEN
                              EELPR=EELPR*BLCOE(KK)*0.5*(DOCFI + DOCFJ)
                              ENBPR=ENBPR*BLCOV(KK)*0.5*(DOCFI + DOCFJ)
                           ELSE
#endif 
                              EELPR=EELPR*BLCOE(KK)
                              ENBPR=ENBPR*BLCOV(KK)
#if KEY_DOCK==1
                           ENDIF
#endif 
                           DF=DFELEC*BLCOE(KK)+DFVDW*BLCOV(KK)
                        ELSE
!..RHS B980630.rs2, Split the "##IF BLOCK" block
#endif /* (bl_a)*/
                           DF=DFELEC+DFVDW
#if KEY_BLOCK==1 /*abl_a*/
                        ENDIF

                        IF (.NOT. NOFORC) THEN
#endif /* (abl_a)*/

                           IF (LSWITR) DF=DF*FUNCT
                           DXIT=DXI*DF
                           DYIT=DYI*DF
                           DZIT=DZI*DF
#if KEY_BLOCK==1 /*bl_b*/
#if KEY_DOCK==1
                           IF(QDOCK) THEN
                              DX(I)=DX(I)+DXIT*DOCFI
                              DY(I)=DY(I)+DYIT*DOCFI
                              DZ(I)=DZ(I)+DZIT*DOCFI
                              DX(J)=DX(J)-DXIT*DOCFJ
                              DY(J)=DY(J)-DYIT*DOCFJ
                              DZ(J)=DZ(J)-DZIT*DOCFJ
                           ELSE
#endif /* dock*/
#endif /* (bl_b)*/
                              DX(I)=DX(I)+DXIT
                              DY(I)=DY(I)+DYIT
                              DX(J)=DX(J)-DXIT
                              DY(J)=DY(J)-DYIT
#if KEY_ASPENER==1
                              IF (LRSW .AND. LMEMBR) THEN
                                 IF (LSWITR) THEN
! pore model
                                    IF (LPOR) THEN
                                       DX(I)=DX(I)+DFZ*LAM(J)*DFDX(I)*FUNCT
                                       DX(J)=DX(J)+DFZ*LAM(I)*DFDX(J)*FUNCT
                                       DY(I)=DY(I)+DFZ*LAM(J)*DFDY(I)*FUNCT
                                       DY(J)=DY(J)+DFZ*LAM(I)*DFDY(J)*FUNCT
                                    ENDIF
! curved membrane
                                    IF (LCURV) THEN
                                        DZ(I)=DZ(I)+DZIT
                                        DZ(J)=DZ(J)-DZIT
                                       IF (LTUB) THEN
                                         TMPR1= SQRT(Y(I)**2+Z(I)**2)                            
                                         TMPR2= SQRT(Y(J)**2+Z(J)**2) 
                                         DK1= DFZ*LAM(J)*DFDZ(I)*FUNCT
                                         DY(I)=DY(I)+ DK1*Y(I)/TMPR1
                                         DZ(I)=DZ(I)+ DK1*Z(I)/TMPR1
                                         DK2= DFZ*LAM(I)*DFDZ(J)*FUNCT
                                         DY(J)=DY(J)+ DK2*Y(J)/TMPR2
                                         DZ(J)=DZ(J)+ DK2*Z(J)/TMPR2
                                       ELSE
                                         TMPR1= SQRT(X(I)**2+Y(I)**2+Z(I)**2)
                                         TMPR2= SQRT(X(J)**2+Y(J)**2+Z(J)**2)
                                         DK1= DFZ*LAM(J)*DFDZ(I)*FUNCT
                                         DX(I)=DX(I)+ DK1*X(I)/TMPR1                                    
                                         DY(I)=DY(I)+ DK1*Y(I)/TMPR1
                                         DZ(I)=DZ(I)+ DK1*Z(I)/TMPR1
                                         DK2= DFZ*LAM(I)*DFDZ(J)*FUNCT
                                         DX(J)=DX(J)+ DK2*X(J)/TMPR2                                    
                                         DY(J)=DY(J)+ DK2*Y(J)/TMPR2
                                         DZ(J)=DZ(J)+ DK2*Z(J)/TMPR2
                                       ENDIF
                                       
                                    ELSE

                                         DZ(I)=DZ(I)+DZIT+DFZ*LAM(J)*DFDZ(I)*FUNCT
                                         DZ(J)=DZ(J)-DZIT+DFZ*LAM(I)*DFDZ(J)*FUNCT
                                   ENDIF
                                 ELSE
! pore model
                                    IF (LPOR) THEN
                                       DX(I)=DX(I)+DFZ*LAM(J)*DFDX(I)
                                       DX(J)=DX(J)+DFZ*LAM(I)*DFDX(J)
                                       DY(I)=DY(I)+DFZ*LAM(J)*DFDY(I)
                                       DY(J)=DY(J)+DFZ*LAM(I)*DFDY(J)
                                    ENDIF

! curved membrane
                                    IF (LCURV) THEN
                                       DZ(I)=DZ(I)+DZIT
                                       DZ(J)=DZ(J)-DZIT
                                       IF (LTUB) THEN
                                         TMPR1= SQRT(Y(I)**2+Z(I)**2)                            
                                         TMPR2= SQRT(Y(J)**2+Z(J)**2) 
                                         DK1= DFZ*LAM(J)*DFDZ(I)
                                         DY(I)=DY(I)+ DK1*Y(I)/TMPR1
                                         DZ(I)=DZ(I)+ DK1*Z(I)/TMPR1
                                         DK2= DFZ*LAM(I)*DFDZ(J)
                                         DY(J)=DY(J)+ DK2*Y(J)/TMPR2
                                         DZ(J)=DZ(J)+ DK2*Z(J)/TMPR2
                                       ELSE
                                         TMPR1= SQRT(X(I)**2+Y(I)**2+Z(I)**2)
                                         TMPR2= SQRT(X(J)**2+Y(J)**2+Z(J)**2)
                                         DK1= DFZ*LAM(J)*DFDZ(I)
                                         DX(I)=DX(I)+ DK1*X(I)/TMPR1                                    
                                         DY(I)=DY(I)+ DK1*Y(I)/TMPR1
                                         DZ(I)=DZ(I)+ DK1*Z(I)/TMPR1
                                         DK2= DFZ*LAM(I)*DFDZ(J)
                                         DX(J)=DX(J)+ DK2*X(J)/TMPR2                                    
                                         DY(J)=DY(J)+ DK2*Y(J)/TMPR2
                                         DZ(J)=DZ(J)+ DK2*Z(J)/TMPR2
                                       ENDIF                                    
                                    ELSE
                                    DZ(I)=DZ(I)+DZIT+DFZ*LAM(J)*DFDZ(I)
                                    DZ(J)=DZ(J)-DZIT+DFZ*LAM(I)*DFDZ(J)
                                   ENDIF
                                 ENDIF
                              ELSE
#endif 
                                 DZ(I)=DZ(I)+DZIT
                                 DZ(J)=DZ(J)-DZIT
#if KEY_ASPENER==1
                              ENDIF  
#endif
#if KEY_BLOCK==1
#if KEY_DOCK==1
                           ENDIF
#endif 
                        ENDIF
#endif 

#if KEY_FLUCQ==1
! If FlucQ is interactive, sum the interaction energies (scaled by the
! switching function if necessary) into the FlucQ arrays
                        IF (QFLUC) THEN
                           FQCFOR(I)=FQCFOR(I)+EELPR*FUNCT
                           FQCFOR(J)=FQCFOR(J)+EELPR*FUNCT
                        ENDIF
#endif 
                        ETEMP1=ETEMP1+ENBPR
                        ETEMP2=ETEMP2+EELPR
                     ENDDO loop210
                  ENDDO loop220

#if KEY_BLOCK==1
                  IF (.NOT. NOFORC) THEN
#endif 
                     IF(LSWITR) THEN
!     Backtransform center of geometry forces
                        IF(LSWIT) THEN
                           DFF=DFN*(ETEMP1+ETEMP2)
                           DFI=DFF/NI
                           DFJ=DFF/NJ
                           DXIT=DXIC*DFI
                           DYIT=DYIC*DFI
                           DZIT=DZIC*DFI
                           DXJT=DXIC*DFJ
                           DYJT=DYIC*DFJ
                           DZJT=DZIC*DFJ
                           DO I=IS,IQ
                              DX(I)=DX(I)+DXIT
                              DY(I)=DY(I)+DYIT
                              DZ(I)=DZ(I)+DZIT
                           ENDDO
                           DO J=JS,JQ
                              DX(J)=DX(J)-DXJT
                              DY(J)=DY(J)-DYJT
                              DZ(J)=DZ(J)-DZJT
                           ENDDO
                        ENDIF
                     ENDIF
#if KEY_BLOCK==1
                  ENDIF
#endif 

!     Apply the switching function to the energy terms
                  IF(LSWITR) THEN
                     ETEMP1=ETEMP1*FUNCT
                     ETEMP2=ETEMP2*FUNCT
                  ENDIF

               ENDIF
            ENDIF

            EEL=EEL+ETEMP2
            ENB=ENB+ETEMP1
         ENDDO loop370
      ENDDO loop380

      RETURN
   END SUBROUTINE ENBFSG

end module enbondg
