SUBROUTINE NBONDG(X,Y,Z,NNNBG,MXJNBG,JNBG,INBLOG,ING14,IGLO14, &
     CUTNB,CTEXNB,LEXTND,LQUAD,LGRAD,CMPLTD,EPS, &
#if KEY_MTS==1
     NNMT1G,MAXJM1G,JNM1G,INBLM1G, &      
     NNMT2G,MAXJM2G,JNM2G,INBLM2G, &      
#endif
#if KEY_PERT==1
     NNNBGP,MXJNGP,JNBGP,INBLGP, &       
     NNNBGR,MXJNGR,JNBGR,INBLGR, &       
     IGPERT,ING14P,IGL14P, &             
#endif
     RSCMX,RSCMY,RSCMZ,RSQ, &
     RSDX,RSDY,RSDZ,RSQXX,RSQYY,RSQZZ, &
     RSQXY,RSQYZ,RSQZX,RSXMAX,RSYMAX,RSZMAX, &
     RSPOT,RSFX,RSFY,RSFZ, &
     RSGXX,RSGYY,RSGZZ,RSGXY,RSGYZ,RSGZX, &
#if KEY_FOURD==1
     RSFMAX,RSCMF, &                     
#endif
     ATSX,ATSY,ATSZ,ATPOT,ATFX,ATFY,ATFZ, &
     ATGXX,ATGYY,ATGZZ,ATGXY,ATGYZ,ATGZX)
  !-----------------------------------------------------------------------
  !     THIS ROUTINE CONSTRUCTS THE NONBONDED LISTS AND ACCUMULATES
  !     A SET OF ATOM POTENTIALS AND GRADIENTS FOR ELECTROSTATIC
  !     INTERACTIONS OUTSIDE OF THE CUTOFF IF REQUESTED.
  !
  !     ATPOT AND ATF_ HOLD THE POTENTIAL AND ITS FIRST DERIVATIVES
  !     FOR ELECTROSTATIC INTERACTIONS OUTSIDE OF THE CUTOFF (IN UNITS OF
  !     KCAL/MOLE AND ANGSTROMS).
  !
  !     LEXTND IS A FLAG SPECIFYING EXTENDED ELECTROSTATICS
  !     LQUAD IS A FLAG THAT IS SET TO INCLUDE RESIDUE QUADRUPOLE MOMENTS.
  !     GRADIENTS IS A FLAG THAT IS SET TO CALCULATE THE FIELD GRADIENTS.
  !
  !     22-AUG-1981  By Bernard R. Brooks
  !     EXTENDED ELECTROSTATICS BY DAVID STATES
  !
  !
  use chm_kinds
  use exfunc
  use dimens_fcm
  use number
#if KEY_MTS==1
  use tbmts        
#endif
#if KEY_PERT==1
  use pert
  use bases_fcm
#endif 
  use exclm
  use psf
  use stream
  use timerm
  use consta
#if KEY_PARALLEL==1
  use parallel      
#endif
#if KEY_REPLICA==1
  use replica_mod       
#endif
  use blockscc_fcm   !QC: 11/17 based on Xiya

#if KEY_GCMC==1
  use gcmc
#endif 
  use fourdm         !#FOURD
#if KEY_PBOUND==1
  use pbound        
#endif
  !
  ! QC:UW_031205: Make sure that QM/QM group interactions are excluded
#if KEY_GAMESS==1 || KEY_QUANTUM==1 || KEY_CADPAC==1 || KEY_SCCDFTB==1 || KEY_QCHEM==1 || KEY_QTURBO==1 || KEY_G09==1
  use gamess_fcm
  use energym
#endif 
  use chutil,only:atomid
  use machutil,only:die,timre,timrb
  !---   use nbutil_module,only:qinlist
  implicit none
  !
  real(chm_real)  X(*),Y(*),Z(*)
  INTEGER NNNBG,MXJNBG,JNBG(*),INBLOG(*)
  INTEGER IGLO14(*)
  INTEGER ING14(*)
  real(chm_real)  CUTNB,CTEXNB,EPS
  LOGICAL LEXTND,LQUAD,LGRAD,CMPLTD
  logical,external :: qinlist
  real(chm_real)  RSCMX(*),RSCMY(*),RSCMZ(*),RSQ(*)
  real(chm_real)  RSDX(*),RSDY(*),RSDZ(*),RSQXX(*),RSQYY(*),RSQZZ(*)
  real(chm_real)  RSQXY(*),RSQYZ(*),RSQZX(*),RSXMAX(*),RSYMAX(*), &
       RSZMAX(*)
  real(chm_real)  RSPOT(*),RSFX(*),RSFY(*),RSFZ(*)
  real(chm_real)  RSGXX(*),RSGYY(*),RSGZZ(*),RSGXY(*),RSGYZ(*), &
       RSGZX(*)
  real(chm_real)  ATSX(*),ATSY(*),ATSZ(*),ATPOT(*), &
       ATFX(*),ATFY(*),ATFZ(*)
  real(chm_real)  ATGXX(*),ATGYY(*),ATGZZ(*),ATGXY(*), &
       ATGYZ(*),ATGZX(*)
  !
#if KEY_MTS==1 /*mtsdecl*/
  INTEGER NNMT1G,MAXJM1G,JNM1G(*),INBLM1G(*)
  INTEGER NNMT2G,MAXJM2G,JNM2G(*),INBLM2G(*)
#endif /* (mtsdecl)*/
  !
#if KEY_PERT==1 /*pertdecl*/
  INTEGER NNNBGP,MXJNGP,JNBGP(*),INBLGP(*)
  INTEGER NNNBGR,MXJNGR,JNBGR(*),INBLGR(*)
  INTEGER IGPERT(*),ING14P(*),IGL14P(*)
  ! PERT local declarations
  INTEGER NXIP,NXIMXP
  LOGICAL LEX14P
#if KEY_CHEMPERT==1
  !sb   chem pert aux. variable
  integer iprtsu
#endif 
#endif /* (pertdecl)  IF PERT*/
  !
#if KEY_PARALLEL==1 /*pardecl*/
  INTEGER IMYNOD
#endif /* (pardecl)*/
  !
#if KEY_REPLICA==1 /*repdecl*/
  !# <caves>-Aug-4-1993 (Leo Caves)
  integer iRepNo, iRepID
#endif /* (repdecl)  REPLICA*/
  !
  INTEGER ITMP,JTMP
  LOGICAL LTMP, qErr
  !
  ! local storage
  INTEGER I,J,IS,IQ,NAT,NGPE,NGPX,IRS,NXI,NXIMAX
  INTEGER JRS,JS,JQ,IRST,JRST,ISDISP,NREM,IGRPMX
  real(chm_real) CTNBSQ,CTEXSQ,RSDISP
  real(chm_real) XMIN,XMAX,YMIN,YMAX,ZMIN,ZMAX,XD,YD,ZD
  real(chm_real) XD1,YD1,ZD1,R2,R,XI,YI,ZI
  real(chm_real) QSUM,DXT,DYT,DZT,GRPMXS
  real(chm_real) QXXT,QYYT,QZZT,QXYT,QZXT,QYZT,CGT
  real(chm_real) XX,YY,ZZ,XY,YZ,ZX
  real(chm_real) R1,R3,R5,R2X3,R2X5,R2X7,DOT,QXT,QYT,QZT,RQR,CR3
  real(chm_real) TEMP,TEMP2
  LOGICAL LEX14,MOVEFG,QMOVE
  character(len=8) SIDDN,RIDDN,RESDN,ACDN,SIDDN2,RIDDN2,RESDN2,ACDN2
  !
#if KEY_NOST2 != 1 /*st2decl*/
  INTEGER NGST2
  LOGICAL LST2
#endif /* (st2decl)*/
  !
#if KEY_FOURD==1 /*4ddecl*/
  !     4D variable:
  real(chm_real) RSFMAX(*),RSCMF(*)
  real(chm_real) FD,FD1,FMIN,FMAX,DFT
#endif /* (4ddecl)*/
  !
#if KEY_PBOUND==1 /*pbound*/
  real(chm_real) CORR
#endif /*     (pbound)*/
  integer nl0,nl1,nli

  !-----------------------------------------------------------------------
  !
  IF (TIMER > 0) CALL TIMRB
  !
#if KEY_REPLICA==1 /*repsetup*/
  IF (qRep) nRepXG = 0
#endif /* (repsetup)  REPLICA*/
  !
#if KEY_GAMESS==1 || KEY_QUANTUM==1 || KEY_CADPAC==1 || KEY_SCCDFTB==1 || KEY_QCHEM==1 || KEY_QTURBO==1 || KEY_G09==1
  !     QC: UW_031205 set up the QM list if QM energy is to be computed
  !     note that we don't do this for QUANTUM because it handles
  !     group based list by itself
  !     write(*,*) "Inside NBONDG>",QETERM(QMEL),QQMGRP,qmused,qmused_sccdftb
  IF (QETERM(QMEL).and.qmused) THEN
     NQMEXL=0
     ! MH-2011:
     ! The code below is compiled in but never executed if QMUSED==.FALSE.
     ! igmsel and lqmgrp are not allocated in this case - hope this is OK
     ! If not we can always make a fake allocation. All this is new problem
     ! because we want to use dynamic allocation of all data structures!
     ! Tring to FIX properly - it requires testing with all QM packages!
     ! QC: 11/17 EPERT 
     !IF (.NOT.QQMGRP) THEN
     IF (.NOT.QQMGRP &
#if KEY_SCCDFTB==1
         .or. QSCCPERT &  /*Xiya: execute this every time if qsccpert is being called*/
#endif
        ) THEN
        MQMGRP = 0
        DO IRS=1,NGRP
           IS=IGPBS(IRS) + 1
           IQ=IGPBS(IRS+1)
           LQMGRP(IRS)=0
           DO I=IS,IQ
              IF ((IGMSEL(I) == 1).OR.(IGMSEL(I) == 2)) THEN
                 LQMGRP(IRS)=1
                 MQMGRP = MQMGRP + 1
                 EXIT
              ENDIF
           ENDDO
        ENDDO
        QQMGRP=.true.
        IF (PRNLEV >= 5) write(OUTU,*)  &
             "QM groups found: ", MQMGRP," among ",NGRP, &
             " groups"
     ENDIF
  ENDIF
#endif 

  CMPLTD=.FALSE.
  !
  CTNBSQ=CUTNB*CUTNB
  CTEXSQ=CTEXNB*CTEXNB
  QMOVE=.FALSE.
  !brb..07-FEB-99 Add warning for disconnected electrostatic groups.
  GRPMXS=ZERO
  IGRPMX=1
  !
  ! Store the current atom configuration.
  DO I=1,NATOM
     ATSX(I)=X(I)
     ATSY(I)=Y(I)
     ATSZ(I)=Z(I)
  ENDDO
  !
#if KEY_NOMISC != 1 /*exelinit*/
  IF(LEXTND) THEN
     !
     ! Intialize atom potentials.
     DO I=1,NATOM
        ATPOT(I)=0.0
        ATFX(I)=0.0
        ATFY(I)=0.0
        ATFZ(I)=0.0
        IF (LGRAD) THEN
           ATGXX(I)=0.0
           ATGYY(I)=0.0
           ATGZZ(I)=0.0
           ATGXY(I)=0.0
           ATGYZ(I)=0.0
           ATGZX(I)=0.0
        ENDIF
     ENDDO
#if KEY_PERT==1
     ! ------ NKB, extelec with pert -------------------------------
     IF(QPERT) THEN
        atpot0(1:natom) = ATPOT(1:NATOM)
        atfx0 (1:natom) = ATFX (1:NATOM)
        atfy0 (1:natom) = ATFY (1:NATOM)
        atfz0 (1:natom) = ATFZ (1:NATOM)
        atgxx0(1:natom) = ATGXX(1:NATOM)
        atgyy0(1:natom) = ATGYY(1:NATOM)
        atgzz0(1:natom) = ATGZZ(1:NATOM)
        atgxy0(1:natom) = ATGXY(1:NATOM)
        atgyz0(1:natom) = ATGYZ(1:NATOM)
        atgzx0(1:natom) = ATGZX(1:NATOM)
     ENDIF
     ! ----- NKB, extelec with pert ------------------------------
#endif 
     !
  ENDIF
#endif /* (exelinit)  IFN NOMISC*/
  !
  ! Find geometric center for each residue and the multipole moments
  ! for the charge distributions.
  DO I=1,NGRP
     IS=IGPBS(I)+1
     IQ=IGPBS(I+1)
     NAT=IQ-IS+1
     IF(NAT <= 0) CALL DIE
     XMIN=X(IS)
     XMAX=XMIN
     YMIN=Y(IS)
     YMAX=YMIN
     ZMIN=Z(IS)
     ZMAX=ZMIN
     DXT=ZERO
     DYT=ZERO
     DZT=ZERO
#if KEY_FOURD==1 /*4dset1*/
     IF (DIM4) THEN
        FMIN=FDIM(IS)
        FMAX=FMIN
        DFT=ZERO
     ENDIF
#endif /* (4dset1)*/
     DO J=IS,IQ
        !...##IF GCMC
        !            IF (.NOT. GCMCON(J)) GOTO 13
        !...##ENDIF
        DXT=DXT+X(J)
        DYT=DYT+Y(J)
        DZT=DZT+Z(J)
        XMIN=MIN(X(J),XMIN)
        YMIN=MIN(Y(J),YMIN)
        ZMIN=MIN(Z(J),ZMIN)
        XMAX=MAX(X(J),XMAX)
        YMAX=MAX(Y(J),YMAX)
        ZMAX=MAX(Z(J),ZMAX)
#if KEY_FOURD==1 /*4dset2*/
        IF(DIM4) THEN
           DFT=DFT+FDIM(J)
           FMIN=MIN(FDIM(J),FMIN)
           FMAX=MAX(FDIM(J),FMAX)
        ENDIF
#endif /* (4dset2)*/
13   ENDDO
     XD=DXT/NAT
     YD=DYT/NAT
     ZD=DZT/NAT
     !
     ! Size of rectangular box surrounding group.
     RSXMAX(I)=MAX(XMAX-XD,XD-XMIN)
     RSYMAX(I)=MAX(YMAX-YD,YD-YMIN)
     RSZMAX(I)=MAX(ZMAX-ZD,ZD-ZMIN)
     !brb..07-FEB-99 Add warning for disconnected electrostatic groups.
     IF(RSXMAX(I) > GRPMXS) THEN
        GRPMXS=RSXMAX(I)
        IGRPMX=IS
     ENDIF
     IF(RSYMAX(I) > GRPMXS) THEN
        GRPMXS=RSYMAX(I)
        IGRPMX=IS
     ENDIF
     IF(RSZMAX(I) > GRPMXS) THEN
        GRPMXS=RSZMAX(I)
        IGRPMX=IS
     ENDIF
     !brb..07-FEB-99
     ! Size of largest vdw radius of atoms in group times cutoff factor
     ! center of geometry of group.
     RSCMX(I)=XD
     RSCMY(I)=YD
     RSCMZ(I)=ZD
     !
#if KEY_FOURD==1 /*4dset4*/
     IF(DIM4) THEN
        FD=DFT/NAT
        RSFMAX(I)=MAX(FMAX-FD,FD-FMIN)
        RSCMF(I)=FD
     ENDIF
#endif /* (4dset4)*/
     !
#if KEY_NOMISC != 1 /*exelproc*/
     IF(LEXTND) THEN
#if KEY_PERT==1
        !------------- NKB------------------------------------------------------
        IF(QPERT) THEN
           ! get field and gradient for extended electrostatics for lambda=0 state
           ! in PERT
           CALL EXTINT(RSPOT0,RSFX0 , &
                RSFY0 ,RSFZ0 , &
                RSGXX0,RSGYY0, &
                RSGZZ0,RSGXY0, &
                RSGYZ0,RSGZX0, &
                QSUM,DXT,DYT,DZT,QXXT,QYYT,QZZT, &
                QXYT,QZXT,QYZT,EPS,PPCG, &
                X,Y,Z,XD,YD,ZD, &
                RSDX0 ,RSDY0 , &
                RSDZ0 ,RSQ0  , &
                RSQXX0,RSQYY0, &
                RSQZZ0,RSQXY0, &
                RSQYZ0,RSQZX0, &
                LQUAD,I,IS,IQ)
        ENDIF
        ! get field and gradient for extended electrostatics for lambda=1 state
        ! in PERT
        CALL EXTINT(RSPOT,RSFX,RSFY,RSFZ,RSGXX,RSGYY,RSGZZ,RSGXY, &
             RSGYZ,RSGZX,QSUM,DXT,DYT,DZT,QXXT,QYYT,QZZT, &
             QXYT,QZXT,QYZT,EPS,CG, &
             X,Y,Z,XD,YD,ZD,RSDX,RSDY,RSDZ,RSQ, &
             RSQXX,RSQYY,RSQZZ,RSQXY,RSQYZ,RSQZX, &
             LQUAD,I,IS,IQ)
        !------------- NKB----------------------------------------------
#endif 
     ENDIF     ! NKB, end of LEXTND if loop
#endif /* (exelproc)  IFN NOMISC*/
     IF(IMOVEG(I) > 0) QMOVE=.TRUE.
  ENDDO
  !
  !brb..07-FEB-99 Add warning for disconnected electrostatic groups.
  GRPMXS=GRPMXS*TWO
  IF(GRPMXS > TWELVE) THEN
     IF(WRNLEV >= -1) WRITE(OUTU,137) GRPMXS,IGRPMX
137  FORMAT( &
          ' NBONDG>>  Maximum group spatial extent (12A) exceeded.',/ &
          '   Size is',F12.2,' Angstroms and starts with atom:',I8,/ &
          '   Please check group boundary definitions.')
     CALL DIEWRN(-1)
  ENDIF
  !brb..07-FEB-99
  ! Now decide how to treat each residue pair using a rectangular
  ! search and store the disposition in rsdisp and isdisp.
  !
#if KEY_PERT==1 /*pertzero*/
  IF(QPERT) THEN
     NNNBGP=0
     NNNBGR=0
  ENDIF
#endif /* (pertzero)*/
  !
#if KEY_NOST2 != 1 /*st2zero*/
  NGST2=0
#endif /* (st2zero)*/
  NGPE=0
  NGPX=0
  NNNBG=0
  NREM=0
#if KEY_MTS==1
  IF (QTBMTS) THEN
     NNMT1G=0
     NNMT2G=0
  ENDIF
#endif 

!=======================================================================
!  Expand control section
!-------------------------------------------------------------------
! (disable expand when debug is active)
#if KEY_DEBUG == 1
#undef KEY_EXPAND
#endif

#if KEY_EXPAND == 1 && KEY_FOURD == 1
! Do DIM4 expansion of code
! ##EXPAND  F             .when. FOURD  EXPAND  (expand_dim4)
! ##PASS1   .not.EXPAND
  IF(DIM4) THEN

#undef KEY_EXPAND
#include "nbondg1.inc"
#define KEY_EXPAND 1

! ##PASS2   .not.FOURD
  ELSE

#undef KEY_FOURD
#include "nbondg1.inc"
#define KEY_FOURD 1

! ##EXFIN
  ENDIF
! ##EXEND
! ##ENDEX    (expand_dim4)
#else /* KEY_EXPAND && KEY_FOURD */

#define NBONDG_F_FLAG 1
#include "nbondg1.inc"
#undef NBONDG_F_FLAG

#endif /* KEY_EXPAND && KEY_FOURD */
!=======================================================================
  
  CMPLTD=.TRUE.
  !
  ! Termination of the routine.
  !
#if KEY_NOMISC != 1 /*exelend*/
  IF(LEXTND) THEN
#if KEY_PARALLEL==1 /*paraexel*/
#if KEY_VIBPARA != 1
     IF(NUMNOD > 1) THEN
        CALL GCOMB(RSPOT,NGRP)
        CALL GCOMB(RSFX ,NGRP)
        CALL GCOMB(RSFY ,NGRP)
        CALL GCOMB(RSFZ ,NGRP)
        CALL GCOMB(RSGXX,NGRP)
        CALL GCOMB(RSGYY,NGRP)
        CALL GCOMB(RSGZZ,NGRP)
        CALL GCOMB(RSGXY,NGRP)
        CALL GCOMB(RSGYZ,NGRP)
        CALL GCOMB(RSGZX,NGRP)
#if KEY_PERT==1
        !bfix.on.the charmm.org FORUM 07-JUL-2004
        IF(QPERT) THEN
           CALL GCOMB(RSPOT0,NGRP)
           CALL GCOMB(RSFX0 ,NGRP)
           CALL GCOMB(RSFY0 ,NGRP)
           CALL GCOMB(RSFZ0 ,NGRP)
           CALL GCOMB(RSGXX0,NGRP)
           CALL GCOMB(RSGYY0,NGRP)
           CALL GCOMB(RSGZZ0,NGRP)
           CALL GCOMB(RSGXY0,NGRP)
           CALL GCOMB(RSGYZ0,NGRP)
           CALL GCOMB(RSGZX0,NGRP)
        ENDIF
        !bfix...
#endif 
     ENDIF
#endif 
#endif /* (paraexel)*/
     ! ----------------------- NKB ------------------------------------------
#if KEY_PERT==1
     IF(QPERT)THEN
        ! get field and gradient for extended electrostatics for lambda=0 state
        ! in PERT
        CALL EXTATM(NGRP,IGPBS,IS,IQ,X,Y,Z,RSCMX,RSCMY,RSCMZ, &
             RSFX0 , RSFY0 , &
             RSFZ0 , RSGXX0, &
             RSGYY0,RSGZZ0, &
             RSGXY0,RSGYZ0, &
             RSGZX0,RSPOT0, &
             ATSX,ATSY,ATSZ, ATPOT0, &
             ATFX0, ATFY0 , &
             ATFZ0 ,PPCG, &
             ATGXX0,ATGYY0, &
             ATGZZ0,ATGXY0, &
             ATGYZ0,ATGZX0,NATOM,LGRAD)
        ! get field and gradient for extended electrostatics for lambda=1 state
        ! in PERT
        CALL EXTATM(NGRP,IGPBS,IS,IQ,X,Y,Z,RSCMX,RSCMY,RSCMZ,RSFX, &
             RSFY,RSFZ,RSGXX,RSGYY,RSGZZ,RSGXY,RSGYZ,RSGZX, &
             RSPOT,ATSX,ATSY,ATSZ,ATPOT,ATFX,ATFY,ATFZ,CG, &
             ATGXX,ATGYY,ATGZZ,ATGXY,ATGYZ,ATGZX,NATOM,LGRAD)
     ELSE
        CALL EXTATM(NGRP,IGPBS,IS,IQ,X,Y,Z,RSCMX,RSCMY,RSCMZ,RSFX, &
             RSFY,RSFZ,RSGXX,RSGYY,RSGZZ,RSGXY,RSGYZ,RSGZX, &
             RSPOT,ATSX,ATSY,ATSZ,ATPOT,ATFX,ATFY,ATFZ,CG, &
             ATGXX,ATGYY,ATGZZ,ATGXY,ATGYZ,ATGZX,NATOM,LGRAD)
     ENDIF ! (IF QPERT)
     ! ----------------------- NKB ------------------------------------------
#endif 
  ENDIF
#endif /* (exelend)  IFN NOMISC*/
  !
#if KEY_MTS==1
  IF(SLFG.OR.TBHY1) GOTO 711
#endif 
  !
  IF(PRNLEV >= 5) THEN
     WRITE(OUTU,720)
720  FORMAT(/' General group nonbond list generation found:')
     ! Group lists
     WRITE(OUTU,721) NNNBG
721  FORMAT(I9,' GROUP PAIRS WERE FOUND FOR GROUP LIST')
#if KEY_PERT==1 /*pertprint*/
     IF(QPERT) THEN
        WRITE(OUTU,723) NNNBGR
723     FORMAT(I9,' GROUP PAIRS WERE FOUND FOR REACTANT LIST')
        WRITE(OUTU,724) NNNBGP
724     FORMAT(I9,' GROUP PAIRS WERE FOUND FOR PRODUCT  LIST')
     ENDIF
#endif /* (pertprint)*/
#if KEY_NOST2 != 1 /*st2print2*/
     IF(NST2 > 0) WRITE(OUTU,727) NGST2
727  FORMAT(I9,' OF THEM WERE ST2-ST2 INTERACTIONS')
#endif /* (st2print2)*/
     !
#if KEY_NOMISC != 1 /*exelprint*/
     IF(LEXTND) WRITE(OUTU,728) NGPE
728  FORMAT(I9,' PAIRS USED EXTENDED ELECTROSTATICS')
#endif /* (exelprint)  IFN NOMISC*/
     WRITE(OUTU,729) NGPX
729  FORMAT(I9,' GROUP PAIRS WERE BEYOND CUTOFFS')
     WRITE(OUTU,731) NREM
731  FORMAT(I9,' GROUP PAIRS WERE NOT SELECTED')
  ENDIF
  !
#if KEY_MTS==1 /*mtsprint*/
711 CONTINUE
#if KEY_PARALLEL != 1
  IF (QTBMTS .AND. (SLFG .OR. TBHY1)) THEN
     IF(PRNLEV >= 2) THEN
        WRITE(OUTU,720)
        IF(SLFG) THEN
           WRITE(OUTU,714) NNMT2G
           WRITE(OUTU,713) NNMT1G
714        FORMAT(' MTS> ',I9,' LONG-RANGE  GROUP PAIRS WERE FOUND')
713        FORMAT(' MTS> ',I9,' SHORT-RANGE GROUP PAIRS WERE FOUND')
        ENDIF
     ENDIF
  ENDIF
#endif 
#endif /* (mtsprint)*/
#if KEY_REPLICA==1 /*repprint*/
  !# <caves>-Aug-4-1993 (Leo Caves)
  IF (PRNLEV >= 5.AND.qRep) THEN
     WRITE(OUTU,'(I9,A)')  nRepXG,' REPLICA GROUP PAIRS EXCLUDED'
  ENDIF ! PRNLEV

#endif /* (repprint)  REPLICA*/
#if KEY_GAMESS==1 || KEY_QUANTUM==1 || KEY_CADPAC==1 || KEY_SCCDFTB==1 || KEY_QCHEM==1 || KEY_QTURBO==1 || KEY_G09==1
  !          QC: UW_031205
  IF (PRNLEV >= 5.AND.(QETERM(QMEL))) &
       WRITE(OUTU,'(I9,A)') nqmexl,' QM group pairs excluded'
#endif 

  IF (TIMER == 1) THEN
     IF(PRNLEV >= 2) WRITE(OUTU,830) 'TOTAL TIME IN NBONDG'
830  FORMAT(1X,A)
     CALL TIMRE
     CALL TIMRB
  ENDIF
  !
  RETURN
END SUBROUTINE NBONDG
! ----------------------------------------------------------------------------
!-----------------------------------------------------------------------------
! NKB, subroutine extint to initiate the values for field and gradient
!
SUBROUTINE EXTINT(RSPOT,RSFX,RSFY,RSFZ,RSGXX,RSGYY,RSGZZ,RSGXY, &
     RSGYZ,RSGZX,QSUM,DXT,DYT,DZT,QXXT,QYYT,QZZT, &
     QXYT,QZXT,QYZT,EPS,CG, &
     X,Y,Z,XD,YD,ZD,RSDX,RSDY,RSDZ,RSQ, &
     RSQXX,RSQYY,RSQZZ,RSQXY,RSQYZ,RSQZX, &
     LQUAD,I,IS,IQ)

  ! This subroutine created to add the ability to read in charges for the
  ! lambda=0 and lambda=1 states separately when using extended electrostatics
  !
  use chm_kinds
  use exfunc
  use dimens_fcm
  use number
  use pert
  use stream
  use consta
  ! QC_UW031205:
#if KEY_SCCDFTB==1
  use gamess_fcm
#endif 
#if KEY_GCMC==1
  use gcmc
#endif 
  implicit none
  !
  LOGICAL LQUAD
  real(chm_real)  RSQ(*)
  real(chm_real)  RSDX(*),RSDY(*),RSDZ(*),RSQXX(*),RSQYY(*),RSQZZ(*)
  real(chm_real)  RSFX(*),RSFY(*),RSFZ(*)
  real(chm_real)  RSQXY(*),RSQYZ(*),RSQZX(*)
  real(chm_real)  RSPOT(*),RSGXX(*)
  real(chm_real)  RSGYY(*),RSGZZ(*),RSGXY(*),RSGYZ(*),RSGZX(*)
  real(chm_real)  XD,YD,ZD,X(*),Y(*),Z(*),DXT,DYT,DZT
  real(chm_real)  QXT,QYT,QZT,QXXT,QYYT,QZZT,QXYT,QYZT,QZXT,QSUM
  real(chm_real)  CGT,CG(*),EPS
  INTEGER I,J,IS,IQ

  RSPOT(I)=ZERO
  RSFX(I)=ZERO
  RSFY(I)=ZERO
  RSFZ(I)=ZERO
  RSGXX(I)=ZERO
  RSGYY(I)=ZERO
  RSGZZ(I)=ZERO
  RSGXY(I)=ZERO
  RSGYZ(I)=ZERO
  RSGZX(I)=ZERO
  QSUM=ZERO
  DXT=ZERO
  DYT=ZERO
  DZT=ZERO
  QXXT=ZERO
  QYYT=ZERO
  QZZT=ZERO
  QXYT=ZERO
  QZXT=ZERO
  QYZT=ZERO
  DO J=IS,IQ
     CGT=CCELEC*CG(J)/EPS
     ! QC_UW031205:
#if KEY_SCCDFTB==1
    IF (QMUSED) THEN !Puja-bugfix
     IF ((IGMSEL(J) == 1).OR.(IGMSEL(J) == 2)) CGT=ZERO
    ENDIF
#endif 
#if KEY_GCMC==1
     if(qgcmc) then
        IF (.NOT. GCMCON(J)) CYCLE
     endif
#endif 
     QSUM=QSUM+CGT
     DXT=DXT+CGT*X(J)
     DYT=DYT+CGT*Y(J)
     DZT=DZT+CGT*Z(J)
     QXXT=QXXT+CGT*X(J)*X(J)
     QYYT=QYYT+CGT*Y(J)*Y(J)
     QZZT=QZZT+CGT*Z(J)*Z(J)
     QXYT=QXYT+CGT*X(J)*Y(J)
     QZXT=QZXT+CGT*X(J)*Z(J)
     QYZT=QYZT+CGT*Y(J)*Z(J)
  ENDDO
  RSQ(I)=QSUM
  RSDX(I)=DXT-QSUM*XD
  RSDY(I)=DYT-QSUM*YD
  RSDZ(I)=DZT-QSUM*ZD
  IF (LQUAD) THEN
     QXXT=QXXT+QSUM*XD*XD-2.0*DXT*XD
     QYYT=QYYT+QSUM*YD*YD-2.0*DYT*YD
     QZZT=QZZT+QSUM*ZD*ZD-2.0*DZT*ZD
     RSQXX(I)=2.0*QXXT-QYYT-QZZT
     RSQYY(I)=2.0*QYYT-QXXT-QZZT
     RSQZZ(I)=2.0*QZZT-QXXT-QYYT
     RSQXY(I)=3.0*(QXYT+QSUM*XD*YD-DXT*YD-DYT*XD)
     RSQYZ(I)=3.0*(QYZT+QSUM*YD*ZD-DYT*ZD-DZT*YD)
     RSQZX(I)=3.0*(QZXT+QSUM*XD*ZD-DXT*ZD-DZT*XD)
  ENDIF
  RETURN
END SUBROUTINE EXTINT
!
! NKB, subroutine extgrp to calculate field and gradient
!
SUBROUTINE EXTGRP(IGPBS,JRS,IRS,RSCMX,RSCMY,RSCMZ,RSDX, &
     RSDY,RSDZ,RSQXX,RSQYY,RSQZZ,RSQXY,RSQYZ, &
     RSQZX,RSQ,RSFX,RSFY,RSFZ,RSGXX,RSGYY, &
     RSGZZ,RSGXY,RSGYZ,RSGZX,RSPOT,NGPE, &
     LQUAD)

  ! This subroutine created to add the ability to read in charges for the
  ! lambda=0 and lambda=1 states separately when using extended electrostatics
  !
  use chm_kinds
  use exfunc
  use dimens_fcm
  use number
  use pert
  use stream
  use consta
  implicit none
  !
  LOGICAL LQUAD
  real(chm_real)  RSCMX(*),RSCMY(*),RSCMZ(*),RSQ(*)
  real(chm_real)  RSDX(*),RSDY(*),RSDZ(*),RSQXX(*),RSQYY(*),RSQZZ(*)
  real(chm_real)  RSFX(*),RSFY(*),RSFZ(*)
  real(chm_real)  RSQXY(*),RSQYZ(*),RSQZX(*)
  real(chm_real)  RSPOT(*),RSGXX(*)
  real(chm_real)  RSGYY(*),RSGZZ(*),RSGXY(*),RSGYZ(*),RSGZX(*)
  real(chm_real)  XD,YD,ZD,XX,YY,ZZ,XY,YZ,ZX,DXT,DYT,DZT
  real(chm_real)  QXT,QYT,QZT,QXXT,QYYT,QZZT,QXYT,QYZT,QZXT
  real(chm_real)  R1,R2,R3,R4,R5,R2X3,R2X5,R2X7,RQR,R
  real(chm_real)  CGT,CR3,DOT,TEMP,TEMP2
  INTEGER I,J,NGPE,IRS,IGPBS(*),JRS,JS,JQ,IRST,JRST

  JS=IGPBS(JRS)+1
  JQ=IGPBS(JRS+1)
  XD=RSCMX(IRS)-RSCMX(JRS)
  YD=RSCMY(IRS)-RSCMY(JRS)
  ZD=RSCMZ(IRS)-RSCMZ(JRS)
  XX=XD*XD
  YY=YD*YD
  ZZ=ZD*ZD
  XY=XD*YD
  YZ=YD*ZD
  ZX=ZD*XD
  R2=XX+YY+ZZ
  R=SQRT(R2)
  R1=1.0/R
  R3=R1/R2
  R5=R3/R2
  R2X3=3.0/R2
  R2X5=5.0/R2
  R2X7=7.0/R2
  !
  DO J=1,2
     IF (J == 1) THEN
        IRST=IRS
        JRST=JRS
     ELSE
        IRST=JRS
        JRST=IRS
        XD=-XD
        YD=-YD
        ZD=-ZD
     ENDIF
     DXT=RSDX(JRST)*R3
     DYT=RSDY(JRST)*R3
     DZT=RSDZ(JRST)*R3
     DOT=(DXT*XD+DYT*YD+DZT*ZD)
     !
     IF (LQUAD) THEN
        QXXT=RSQXX(JRST)*R5
        QYYT=RSQYY(JRST)*R5
        QZZT=RSQZZ(JRST)*R5
        QXYT=RSQXY(JRST)*R5
        QYZT=RSQYZ(JRST)*R5
        QZXT=RSQZX(JRST)*R5
        QXT=QXXT*XD+QXYT*YD+QZXT*ZD
        QYT=QYYT*YD+QYZT*ZD+QXYT*XD  ! B990105.rjp
        QZT=QZZT*ZD+QZXT*XD+QYZT*YD
        RQR=(QXT*XD+QYT*YD+QZT*ZD)/2.0
        !
        CGT=RSQ(JRST)*R1
        !
        RSPOT(IRST)=RSPOT(IRST)+CGT+DOT+RQR
        !
        CR3=CGT/R2
        DOT=DOT*R2X3
        RQR=RQR*R2X5
        TEMP=CR3+DOT+RQR
        !
        RSFX(IRST)=RSFX(IRST)+DXT+QXT-TEMP*XD
        RSFY(IRST)=RSFY(IRST)+DYT+QYT-TEMP*YD
        RSFZ(IRST)=RSFZ(IRST)+DZT+QZT-TEMP*ZD
        !
        TEMP2=CR3*R2X3+DOT*R2X5+RQR*R2X7
        DXT=DXT*R2X3+QXT*R2X5
        DYT=DYT*R2X3+QYT*R2X5
        DZT=DZT*R2X3+QZT*R2X5
        !
        RSGXX(IRST)=RSGXX(IRST)+TEMP2*XX-2.0*DXT*XD-TEMP+QXXT
        RSGYY(IRST)=RSGYY(IRST)+TEMP2*YY-2.0*DYT*YD-TEMP+QYYT
        RSGZZ(IRST)=RSGZZ(IRST)+TEMP2*ZZ-2.0*DZT*ZD-TEMP+QZZT
        RSGXY(IRST)=RSGXY(IRST)+TEMP2*XY-DXT*YD-DYT*XD+QXYT
        RSGYZ(IRST)=RSGYZ(IRST)+TEMP2*YZ-DYT*ZD-DZT*YD+QYZT
        RSGZX(IRST)=RSGZX(IRST)+TEMP2*ZX-DZT*XD-DXT*ZD+QZXT
     ELSE
        CGT=RSQ(JRST)*R1
        !
        RSPOT(IRST)=RSPOT(IRST)+CGT+DOT
        !
        CR3=CGT/R2
        DOT=DOT*R2X3
        TEMP=CR3+DOT
        !
        RSFX(IRST)=RSFX(IRST)+DXT-TEMP*XD
        RSFY(IRST)=RSFY(IRST)+DYT-TEMP*YD
        RSFZ(IRST)=RSFZ(IRST)+DZT-TEMP*ZD
        !
        TEMP2=CR3*R2X3+DOT*R2X5
        DXT=DXT*R2X3
        DYT=DYT*R2X3
        DZT=DZT*R2X3
        !
        RSGXX(IRST)=RSGXX(IRST)+TEMP2*XX-2.0*DXT*XD
        RSGYY(IRST)=RSGYY(IRST)+TEMP2*YY-2.0*DYT*YD
        RSGZZ(IRST)=RSGZZ(IRST)+TEMP2*ZZ-2.0*DZT*ZD
        RSGXY(IRST)=RSGXY(IRST)+TEMP2*XY-DXT*YD-DYT*XD
        RSGYZ(IRST)=RSGYZ(IRST)+TEMP2*YZ-DYT*ZD-DZT*YD
        RSGZX(IRST)=RSGZX(IRST)+TEMP2*ZX-DZT*XD-DXT*ZD
     ENDIF
  ENDDO
  !
  RETURN
END SUBROUTINE EXTGRP
!
! NKB, subroutine extatm to distribute field and gradient amongst
! individual atoms
!
SUBROUTINE EXTATM(NGRP,IGPBS,IS,IQ,X,Y,Z,RSCMX,RSCMY,RSCMZ,RSFX, &
     RSFY,RSFZ,RSGXX,RSGYY,RSGZZ,RSGXY,RSGYZ,RSGZX, &
     RSPOT,ATSX,ATSY,ATSZ,ATPOT,ATFX,ATFY,ATFZ,CG, &
     ATGXX,ATGYY,ATGZZ,ATGXY,ATGYZ,ATGZX,NATOM,LGRAD)
  !
  ! This subroutine created to add the ability to read in charges for the
  ! lambda=0 and lambda=1 states separately when using extended electrostatics
  !
  use chm_kinds
  use exfunc
  use dimens_fcm
  use number
  use pert
  use stream
  use consta
  ! QC_UW031205:
#if KEY_SCCDFTB==1
  use gamess_fcm
#endif 
#if KEY_GCMC==1
  use gcmc
#endif 
  implicit none
  !
  LOGICAL LGRAD
  real(chm_real)  RSCMX(*),RSCMY(*),RSCMZ(*)
  real(chm_real)  RSFX(*),RSFY(*),RSFZ(*)
  real(chm_real)  RSPOT(*),RSGXX(*)
  real(chm_real)  RSGYY(*),RSGZZ(*),RSGXY(*),RSGYZ(*),RSGZX(*)
  real(chm_real)  ATSX(*),ATSY(*),ATSZ(*),ATPOT(*), &
       ATFX(*),ATFY(*),ATFZ(*)
  real(chm_real)  ATGXX(*),ATGYY(*),ATGZZ(*),ATGXY(*),ATGYZ(*), &
       ATGZX(*)
  real(chm_real)  XI,YI,ZI,X(*),Y(*),Z(*),XY,YZ,ZX,DXT,DYT,DZT
  real(chm_real)  QXT,QYT,QZT
  real(chm_real)  CGT,CG(*)
  INTEGER I,J,NATOM,NGRP,IRS,IGPBS(*),IS,IQ,IRST

  DO IRST=1,NGRP
     IS=IGPBS(IRST)+1
     IQ=IGPBS(IRST+1)
     DO I=IS,IQ
#if KEY_GCMC==1
        if(qgcmc) then
           IF (.NOT. GCMCON(I)) CYCLE
        endif
#endif 
        XI=X(I)-RSCMX(IRST)
        YI=Y(I)-RSCMY(IRST)
        ZI=Z(I)-RSCMZ(IRST)
        QXT=RSGXX(IRST)*XI+RSGXY(IRST)*YI+RSGZX(IRST)*ZI
        QYT=RSGYY(IRST)*YI+RSGXY(IRST)*XI+RSGYZ(IRST)*ZI
        QZT=RSGZZ(IRST)*ZI+RSGZX(IRST)*XI+RSGYZ(IRST)*YI
        ATPOT(I)=ATPOT(I)+RSPOT(IRST)+((2.0*RSFX(IRST)+QXT)*XI+ &
             (2.0*RSFY(IRST)+QYT)*YI+(2.0*RSFZ(IRST)+QZT)*ZI)/2.0
        ATFX(I)=ATFX(I)+RSFX(IRST)+QXT
        ATFY(I)=ATFY(I)+RSFY(IRST)+QYT
        ATFZ(I)=ATFZ(I)+RSFZ(IRST)+QZT
        IF (LGRAD) THEN
           ATGXX(I)=ATGXX(I)+RSGXX(IRST)
           ATGYY(I)=ATGYY(I)+RSGYY(IRST)
           ATGZZ(I)=ATGZZ(I)+RSGZZ(IRST)
           ATGXY(I)=ATGXY(I)+RSGXY(IRST)
           ATGYZ(I)=ATGYZ(I)+RSGYZ(IRST)
           ATGZX(I)=ATGZX(I)+RSGZX(IRST)
        ENDIF
     ENDDO
  ENDDO
  !
  DO I=1,NATOM
     CGT=CG(I)
     ! QC_UW031205:
#if KEY_SCCDFTB==1
    IF (QMUSED) THEN !Puja-bugfix
     IF ((IGMSEL(I) == 1).OR.(IGMSEL(I) == 2)) CGT=ZERO
    ENDIF
#endif 
#if KEY_GCMC==1
     if(qgcmc) then
        IF (.NOT. GCMCON(I)) CYCLE
     endif
#endif 
     ATPOT(I)=ATPOT(I)*CGT/2.0
     ATFX(I)=ATFX(I)*CGT
     ATFY(I)=ATFY(I)*CGT
     ATFZ(I)=ATFZ(I)*CGT
     IF (LGRAD) THEN
        ATGXX(I)=ATGXX(I)*CGT
        ATGYY(I)=ATGYY(I)*CGT
        ATGZZ(I)=ATGZZ(I)*CGT
        ATGXY(I)=ATGXY(I)*CGT
        ATGYZ(I)=ATGYZ(I)*CGT
        ATGZX(I)=ATGZX(I)*CGT
     ENDIF
  ENDDO
  RETURN
END SUBROUTINE EXTATM
!------------------------------------------------------------------------------
