SUBROUTINE NBONDS(X,Y,Z,BNBNDX,BIMAGX)
!-----------------------------------------------------------------------
!     THIS ROUTINE SETS UP THE LIST OF PAIRS WHICH SHOULD BE USED FOR
!     THE NON BONDED ENERGY CALCULATION. JNB AND INBLO STORE THE PAIRS
!     AS FOLLOWS:
!
!             EACH PAIR IS ORDERED BY SEQUENCE NUMBER (LOW ATOM FIRST)
!             THE PAIRS ARE SORTED IN ASSCENDING ORDER ON THE FIRST ATOM
!
!     JNB STORES THE INDICES OF THE SECOND ATOM IN SUCH A LIST
!     INBLO GIVES THE LAST INDEX IN JNB WHICH PAIRS WITH ATOM I
!
!
!     THIS ROUTINE ACTS AS A DISPATCHER FIRST ALLOCATING SPACE (GUESSING
!     HOW MUCH), THEN CALLING THE APPROPRIATE ROUTINE TO ACTUALLY DO
!     THE WORK. IF INSUFFICIENT SPACE WAS AVAILABLE, MORE IS ALLOCATED
!     AND THE PROCEDURE REPEATED.
!
!     8/23/80 DAVID J. STATES
!     10/4/82 Bernard R. Brooks
!
  use ewald,only:lewald
#if KEY_LOOKUP==1
  use LOOKUP,only:qlookup,iwwenr,qvv,qvu,iwoonbl,jwoonbl,nwwo,mxjwwoo,&    
                  wwspltnb,quu,iwoonbli,mxjwwooi,jwoonbli,ivunbli,iuunbli  
#endif
#if KEY_FLUCQ==1
  use flucqm,only: fqcfor                                                  
#endif
#if KEY_FACTS==1
  use facts_module                                                        
#endif
  use chm_kinds
  use chm_types
  use exfunc
  use dimens_fcm
  use number
#if KEY_ACE==1
  use ace_module,only:LACE      
#endif
  use bases_fcm
  use block_fcm
  use datstr,only:copy_nbond_excl
  use exclm
  use exelecm
  use fast
  use ffieldm
  use fourdm
  use image
  use image_routines_module, only: reimag, transo
  use inbnd
  use machdep
  use mmffm
  use psf
  use pert
  use stream
  use tbmts
  use tsms_mod
  use tsmh
  use quantm
  use parallel
  use replica_mod
  use gamess_fcm
  use pbound
  use nbthole
#if KEY_FLUCQ==1
  use flucq
#endif 
#if KEY_MNDO97==1
  use mndo97
#endif 
#if KEY_SQUANTM==1
  use squantm
#endif 
  use dcm_fcm, only: qdcm,dcmnbup
#if KEY_NO_BYCC==0
! added below--RJP
  use bycc_mod
  use mexclar
  use contrl
#endif 

#if KEY_MSCALE==1
  use mscalemod, only: qmscale                       
#endif
  use memory
  use datstr,only:useddt_nbond,useddt_image
  use machutil,only:die
#if KEY_DOMDEC==1
  use nblist_types,only:nblist
  use nblist_builder,only:ns_xfast
  use domdec_common,only:q_domdec
#endif 
  use param_store, only: set_param
!---   use nbutil_module,only:renbnd,getbnd,setbnd,prnbct
!
  implicit none
!
  integer,allocatable,dimension(:) :: WRKAR,RIMGLS,PIMGLS,NUMMP,HIMPRTN, &
       MMM,LSTM0,CNTN,XO,YO,ZO,LSTN0,XOO,YOO,ZOO,ORDCBN,PNTNER,PNTDIS,PARTNN, &
       PARTND,DUMMYX
  real(chm_real),allocatable,dimension(:) :: HAFMAR
  real(chm_real),allocatable,dimension(:) :: MEANX
  real(chm_real),allocatable,dimension(:) :: MEANY
  real(chm_real),allocatable,dimension(:) :: MEANZ
  real(chm_real),allocatable,dimension(:) :: RSQ,RSDX,RSDY,RSDZ,RSPOT, &
       RSFX,RSFY,RSFZ,RSGXX,RSGYY,RSGZZ,RSGXY,RSGYZ,RSGZX,RSQXX,RSQYY,RSQZZ, &
       RSQXY,RSQYZ,RSQZX
  real(chm_real) X(*),Y(*),Z(*)
!!  INTEGER BNBNDX(*),LNBNDX(*),BIMAGX(*),LIMAGX(*)
  type(nonbondDataStructure) BNBNDX
  type(imageDataStructure) BIMAGX

  !
  INTEGER BIGGST,BIGGR,MAXJNB,MXJNBG,OLDLST,NTRNSX,IVAL
  INTEGER MAXJMB,MXJMBG
  SAVE    MAXJMB,MXJMBG
  real(chm_real),allocatable,dimension(:) :: RSCMX,RSCMY,RSCMZ
  real(chm_real),allocatable,dimension(:) :: RSXMAX,RSYMAX,RSZMAX
  integer,allocatable,dimension(:) :: RSDISP
  real(chm_real) ERXN,RMXGES,RVAL,RFACT
  LOGICAL CMPLTD,LSLOWNB
  !  --added RJP-------------------------
  real(chm_real) NUMER2,DENOM2,NUMER3,DENOM3
  real(chm_real) P3,M2,NACLCU,MEMFAC,RNUMNOD
  INTEGER IMAXJNB,CCMAXJNB,CCMXJNBG
  SAVE CCMAXJNB,CCMXJNBG
  INTEGER MAXACL,TEMPN,TEMPN2
  SAVE MAXACL
  integer,PARAMETER :: MXNCUB=20000
  INTEGER IACLCU
  real(chm_real) PRTDEN,RLIMIT,AMPLIT,CTNBFR,CTNBSX
  real(chm_real) CTNBCU,THRDTM,SECDTM,NCTPRT
  real(chm_real) RNATOM,RHLFMT,RNGRP,RGHFMT
  real(chm_real) XMIN,XMAX,YMIN,YMAX,ZMIN,ZMAX
  INTEGER N1,N2,N3,I,J
  LOGICAL LBYCCF
  integer, dimension(1), target :: pzer1 = [0]
  integer, dimension(:), pointer :: tpjnb,tpinblo, &
       trjnb, trinblo, tpinb14, tpiblo14
  integer, dimension(:), pointer :: tjnbg,tinblog, &
       ting14,tiglo14,tpjnbg,tpinblog,trjnbg,trinblog,tping14,tpiglo14
  !
  ! --------------------------------------
#if KEY_MTS==1
  integer, target, dimension(1) :: mts_dummy = (/ 0 /)
  INTEGER NNMT1,NNMT2,MAXJM1,MAXJM2
  INTEGER MXJMT1,MXJMT2,NNMT5,NNMT6
  INTEGER NNMT1G,NNMT2G,MAXJM1G,MAXJM2G
  INTEGER MXJMT1G,MXJMT2G
  SAVE    MXJMT1,MXJMT2,MXJMT1G,MXJMT2G
#endif 
#if KEY_PERT==1
  INTEGER MXJNGP,MXJNGR,MAXJNP,MAXJNR
  INTEGER NNNBP,NNNBR,NNNBGP,NNNBGR
  INTEGER MXJMBP,MXJMBR,MXJMGP,MXJMGR
  SAVE    MXJNGP,MXJNGR,MAXJNP,MAXJNR,MXJMBP,MXJMBR,MXJMGP,MXJMGR
#endif 
  real(chm_real),allocatable,dimension(:) :: RSFMAX,RSCMF
#if KEY_FACTS==1
  integer :: fct1ac,fct2ac
  integer :: fct3ac,fct4ac
#endif 
#if KEY_LOOKUP==1
  INTEGER NBOO,NBVU,NBUU
  real(chm_real) WON
#endif 
#if KEY_IMCUBES==1
  logical resize,doimages
  save resize
  integer CUMAXNB,CUMAXMB,natimcu
  save CUMAXNB,CUMAXMB
  data resize/.true./
  data CUMAXNB,CUMAXMB/1,1/
#endif 
  ! --added MXJNBG/0/ below -RJP
  DATA MAXJNB/0/,MXJNBG/0/
  DATA MAXJMB/0/
  !  Added below --RJP
  DATA CCMAXJNB/0/,MAXACL/0/
  DATA CCMXJNBG/0/
#if KEY_MTS==1
  DATA MAXJM1/0/,MAXJM2/0/,MXJMT1/0/,MXJMT2/0/
  DATA MAXJM1G/0/,MAXJM2G/0/,MXJMT1G/0/,MXJMT2G/0/
#endif 
#if KEY_PERT==1
  DATA MAXJNP/0/,MXJNGP/0/,MAXJNR/0/,MXJNGR/0/
  DATA MXJMGP/0/,MXJMBP/0/,MXJMGR/0/,MXJMBR/0/
#endif 
!  DATA RSQ,RSDX,RSDY,RSDZ,RSPOT,RSFX,RSFY,RSFZ,RSGXX,RSGYY,RSGZZ, &
!       RSGXY,RSGYZ,RSGZX,RSQXX,RSQYY,RSQZZ,RSQXY,RSQYZ,RSQZX &
!       /20*1/

  integer ierr
  real(chm_real) :: adum(1)
  integer maxjnb_tmp

  ! Test data structures
  !
  IF (.NOT. USEDDT_nbond(BNBND)) THEN
     CALL WRNDIE(-3,'<NBONDS>', &
          'Nonbond data structure is not defined.')
  ENDIF

#if KEY_PERT==1
  IF(QPERT) THEN
    CALL COPY_nbond_excl(BNBNDP,BNBND) 
  ENDIF
#endif 

  CALL GETBND(BNBND,.TRUE.)
#if KEY_REPLICA==1
  IF (qRep) THEN
     nRepXG = 0
     nRepXA = 0
  ENDIF ! qRep
#endif /*  REPLICA*/

#if KEY_TSM==1
  !     This will cause the next call to tsme to update the TSM non-bond
  !     lists.
  IF (QTSM) THEN
     UPPERT = .TRUE.
     IF(PIGSET) CALL PIGCVSET(X,Y,Z)
     !mfc now allocated arrays     
     !mfc     RIMGLS=1
     !mfc     PIMGLS=2
  ENDIF
#endif 
  NTRNSX=NTRANS
#if KEY_PBOUND==1 /*pbound*/
  IF(NTRANS > 0.and..not.qboun) THEN
#else /* (pbound)*/
  IF(NTRANS > 0) THEN
#endif /*     (pbound)*/
     IF (.NOT. USEDDT_image(BIMAG)) THEN
        IF(WRNLEV >= 2) WRITE(OUTU,23)
23      FORMAT(' <NBONDS>: Image data structure is not defined.')
        NTRNSX=0
     ENDIF
  ENDIF
!
#if KEY_LOOKUP==1
! Turn on energy calculation if necessary
  IF(QLOOKUP .AND. IWWENR ==  -1) IWWENR= 1
#endif 
  IF(PRNLEV >= 5) WRITE(OUTU,2000) NBXMOD
2000 FORMAT(' Generating nonbond list with Exclusion mode =',I2)
  !
  !=======================================================================
  !
  !  Check for conflicting options in nonbond usage.
  !
  !    CHARMM Developers: Please keep this section up to date!
  !
  ! Nonbond Methods Support table:
  !                      1   2   3   4   5   6   7   8   9  10  11  12
  ! 1. LEXS              o
  ! 2. EXELEC/RXNFLD     x   o
  ! 3. PERT              -   x   o
  ! 4. MTS               -   x?  x?  o
  ! 5. TSM               -   x   x   x   o
  ! 6. QUANTUM/GAMESS    -   x   x   x   x   o
  !    CADPAC/GAMESSUK/SCCDFTB/MNDO97/SQUANTM
  ! 7. PARALLEL          -   -   -!  -   ??  -   o
  ! 8. REPLICA/BLOCK     -   x   ??  -?  x   x   -   o
  ! 9. PBOUND            -   x?  -   ??  x?  x?  -   -   o
  ! 10. EWALD            x   x   -!  ??  x   -6  -   -   -   o
  ! 11. ST2              -   -   x   x?  x   x   x?  x   -   x   o
  ! 12. DIM4             -   x   x   x   ??  x   -   ??  -   x   x   o
  ! 13. MMFF             -   x   -   -   x   x   -   -   -!  -!  x   x   o
  ! 15. ACE              x   x   x   x   x   x   -   x   x?  x   x   x   x? x  o
  !
  !     ATOM lists       -   x   -   -   -   -   -   -   -   -   x   -   -  -
  !     GROUP lists      -   -   -   -   x   x   -   x   -   -!  -   x   x  -
  !
  !  Key: x (not supported), - (supported), o (self),
  !       x? (probably not supported), -? (probably supported), ?? (don't know)
  !       -! (development in progress)
  !       -6 (supported with QUANTUM, SQUANTM and MNDO97)
  !==
  !==  1. LEXS
  !==
  IF(LEXS) THEN
#if KEY_NOMISC==0
     IF(QEXTND .OR. QRXNFL) CALL WRNDIE(-4,'<NBONDS>',    & 
#endif
#if KEY_NOMISC==0
          'LEXS is incompatible with EXELEC or RXNFLD')    
#endif
     IF(LEWALD) CALL WRNDIE(-4,'<NBONDS>', &
          'LEXS is incompatible with EWALD')
  ENDIF
  !==
  !==  2. EXELEC/RXNFLD
  !==
#if KEY_NOMISC==0 /*exeltest2*/
  ! next two lines included by NKB, to put extended electrostatics
  ! in PERT but keep warning for rxnfld
#if KEY_PERT==1
  IF(QPERT.AND.QRXNFL) CALL WRNDIE(-4,'<NBONDS>',     & 
#endif
#if KEY_PERT==1
       'PERT is incompatible with RXNFLD')             
#endif
  IF(QEXTND .OR. QRXNFL) THEN
     ! NKB, next two lines commented out to make PERT compatible with extelec
#if KEY_PERT==1
     !         IF(QPERT) CALL WRNDIE(-4,'<NBONDS>',               
     !     &      'PERT is incompatible with EXELEC or RXNFLD')   
#endif
#if KEY_MTS==1
     IF(TBHY1.OR.SLFG) CALL WRNDIE(-1,'<NBONDS>',        & 
#endif
#if KEY_MTS==1
          'EXELEC with MTS is untested - good luck')      
#endif
#if KEY_TSM==1
     IF (QTSM) CALL WRNDIE(-4,'<NBONDS>',                & 
          'TSM is incompatible with EXELEC or RXNFLD')    
#endif
! QC_UW_0217
#if KEY_SCCDFTB==0
#if KEY_GAMESS==1 || KEY_GAMESSUK==1 || KEY_QUANTUM==1 || KEY_CADPAC==1 || \
    KEY_QCHEM==1 || KEY_MNDO97==1 || KEY_SQUANTM==1 || KEY_QTURBO==1 || KEY_G09==1
     IF(QGMREM) CALL WRNDIE(-4,'<NBONDS>', &
          'EXELEC or RXNFLD is incompatible with QM/MM Methods')
#endif 
#endif
#if KEY_REPLICA==1
     IF (qRep) CALL WRNDIE(-4,'<NBONDS>',                & 
          'REPLICA is incompatible with EXELEC or RXNFLD') 
#endif
#if KEY_BLOCK==1
     IF (QBLOCK) CALL WRNDIE(-4,'<NBONDS>',              & 
          'BLOCK is incompatible with EXELEC or RXNFLD')   
#endif
#if KEY_PBOUND==1
     If(qBoun) CALL WRNDIE(-1,'<NBONDS>',                & 
          'PBOUND with EXELEC or RXNFLD is untested')      
#endif
     IF(LEWALD) CALL WRNDIE(-4,'<NBONDS>', &
          'EWALD is incompatible with EXELEC or RXNFLD')
#if KEY_FOURD==1
     IF (DIM4) CALL WRNDIE(-4,'<NBONDS>',                & 
          'DIM4 is incompatible with EXELEC or RXNFLD')    
#endif
     IF (.NOT.LGROUP) CALL WRNDIE(-4,'<NBONDS>', &
          'Atom lists are incompatible with EXELEC or RXNFLD')
#if KEY_MMFF==1
     IF(FFIELD == MMFF) CALL WRNDIE(-4,'<NBONDS>',       & 
          'MMFF is incompatible with EXELEC or RXNFLD ')   
#endif
  ENDIF
#endif /* (exeltest2)*/
  !==
  !==  3. PERT
  !==
#if KEY_PERT==1 /*pertinit*/
  ! test for PSF consistency
  IF(QPERT) THEN
     IF(NATOM /= NATOMP) CALL WRNDIE(-4,'<NBONDS>', &
          'Number of atoms does not match.')
#if KEY_MTS==1
     IF(TBHY1.OR.SLFG) CALL WRNDIE(-1,'<NBONDS>', 'PERT with MTS is untested - good luck')    
     IF (QTSM) CALL WRNDIE(-4,'<NBONDS>','PERT is incompatible with TSM')
#endif
#if KEY_GAMESS==1 || KEY_GAMESSUK==1 || KEY_QUANTUM==1 || KEY_CADPAC==1 || \
                      KEY_MNDO97==1 || KEY_SQUANTM==1
!   QC: 11/17 - SCCDFTB actually works with PERT now
!   KEY_SCCDFTB==1 || KEY_MNDO97==1 || KEY_SQUANTM==1 
    IF(QGMREM .AND. .NOT. qmused_qchem) &
         CALL WRNDIE(-4,'<NBONDS>','PERT is incompatible with QM/MM methods')
#endif 

#if KEY_REPLICA==1
     IF (qRep) CALL WRNDIE(-1,'<NBONDS>','REPLICA is untested with PERT - good luck')    
#endif
#if KEY_BLOCK==1
     IF (QBLOCK) THEN                                      
#endif
#if KEY_MSCALE==1
        IF(.NOT.QMSCALE) THEN                              
#endif
#if KEY_BLOCK==1
           CALL WRNDIE(-1,'<NBONDS>', 'BLOCK is untested with PERT - good luck') 
#endif
#if KEY_MSCALE==1
        ENDIF                                              
#endif
#if KEY_BLOCK==1
     ENDIF                                                 
#endif
#if KEY_PBOUND==1
     If(qBoun) CALL WRNDIE(-1,'<NBONDS>',                & 
#endif
#if KEY_PBOUND==1
          'PBOUND is untested with PERT - good luck')      
#endif
     !C         IF(LEWALD) CALL WRNDIE(-1,'<NBONDS>',
     !C     &     'EWALD is untested with PERT - good luck')
#if KEY_NOST2==0
     IF(NST2 > 0) CALL WRNDIE(-4,'<NBONDS>',            & 
#endif
#if KEY_NOST2==0
          'ST2 waters not supported with PERT')            
#endif
#if KEY_FOURD==1
     IF (DIM4) CALL WRNDIE(-4,'<NBONDS>',                & 
#endif
#if KEY_FOURD==1
          'DIM4 is incompatible with PERT')                
#endif
     !        With PERT, Number of groups and size of each group must match.
     IF(LGROUP .AND. NGRP /= NGRPP) CALL WRNDIE(-4,'<NBONDS>', &
          'Number of groups does not match for GROUP option with PERT')
  ENDIF  ! (QPERT)
#endif /* (pertinit)  IF PERT*/
  !==
  !==  4. MTS
  !==
#if KEY_MTS==1 /*mtstests*/
  IF(TBHY1.OR.SLFG) THEN
#if KEY_TSM==1
     IF (QTSM) CALL WRNDIE(-4,'<NBONDS>','MTS is incompatible with TSM')                  
#endif
#if KEY_GAMESS==1 || KEY_GAMESSUK==1 || KEY_QUANTUM==1 || KEY_CADPAC==1 || \
    KEY_SCCDFTB==1 || KEY_QCHEM==1 || KEY_MNDO97==1 || KEY_SQUANTM==1 || \
    KEY_QTURBO==1 || KEY_G09==1
     IF(QGMREM) CALL WRNDIE(-4,'<NBONDS>', &
          'MTS is incompatible with QM/MM methods')
#endif 
#if KEY_REPLICA==1
     IF (qRep) CALL WRNDIE(-1,'<NBONDS>','REPLICA is untested with MTS - good luck')      
#endif
#if KEY_BLOCK==1
     IF (QBLOCK) CALL WRNDIE(-1,'<NBONDS>','BLOCK is untested with MTS - good luck')        
#endif
#if KEY_PBOUND==1
     If(qBoun) CALL WRNDIE(-1,'<NBONDS>','PBOUND is untested with MTS - good luck')       
#endif
     IF(LEWALD) CALL WRNDIE(-4,'<NBONDS>', &
          'EWALD is untested with MTS - good luck')
#if KEY_NOST2==0
     IF(NST2 > 0) CALL WRNDIE(-1,'<NBONDS>', 'ST2 waters MTS is untested - good luck')        
#endif
#if KEY_FOURD==1
     IF (DIM4) CALL WRNDIE(-4,'<NBONDS>', 'DIM4 is incompatible with MTS')                 
#endif
  ENDIF  ! (TBHY1.OR.SLFG)
#endif /* (mtstests)*/
  !==
  !==  5. TSM
  !==
#if KEY_TSM==1 /*tsmtest*/
  tsm1: IF (QTSM) THEN
#if KEY_GAMESS==1 || KEY_GAMESSUK==1 || KEY_QUANTUM==1 || KEY_CADPAC==1 || \
    KEY_SCCDFTB==1 || KEY_QCHEM==1 || KEY_QTURBO==1 || KEY_G09==1
     IF(QGMREM) CALL WRNDIE(-4,'<NBONDS>', &
          'TSM is incompatible with QM/MM Method')
#endif 
#if KEY_PARALLEL==1
     !CC         IF(NUMNOD >= 2) CALL WRNDIE(-1,'<NBONDS>',         
#endif
#if KEY_PARALLEL==1
     !CC     &     'PARALLEL with TSM is untested - good luck')     
#endif
#if KEY_REPLICA==1
     IF (qRep) CALL WRNDIE(-4,'<NBONDS>',                & 
#endif
#if KEY_REPLICA==1
          'REPLICA is incompatible with TSM')              
#endif
#if KEY_BLOCK==1
     IF (QBLOCK) CALL WRNDIE(-4,'<NBONDS>',              & 
#endif
#if KEY_BLOCK==1
          'BLOCK is incompatible with TSM')                
#endif
#if KEY_PBOUND==1
     If(qBoun) CALL WRNDIE(-1,'<NBONDS>',                & 
#endif
#if KEY_PBOUND==1
          'PBOUND with TSM is untested - good luck')  
#endif
     IF(LEWALD) CALL WRNDIE(-4,'<NBONDS>', &
          'EWALD is incompatible with TSM')
#if KEY_NOST2==0
     IF(NST2 > 0) CALL WRNDIE(-4,'<NBONDS>',            & 
#endif
#if KEY_NOST2==0
          'ST2 waters not supported with TSM')             
#endif
#if KEY_FOURD==1
     IF (DIM4) CALL WRNDIE(-1,'<NBONDS>',                & 
#endif
#if KEY_FOURD==1
          'DIM4 is untested with TSM - good luck')         
#endif
     IF(LGROUP) CALL WRNDIE(-4,'<NBONDS>', &
          'Group lists are not supported with TSM')
#if KEY_MMFF==1
     IF(FFIELD == MMFF) CALL WRNDIE(-4,'<NBONDS>',       & 
#endif
#if KEY_MMFF==1
          'TSM is not implemented for MMFF')               
#endif
  ENDIF tsm1
#endif /* (tsmtest)*/
  !==
  !==  6. QUANTUM/GAMESS/GAMESSUK/CADPAC/SCCDFTB/MNDO97/SQUANTM
  !==
#if KEY_GAMESS==1 || KEY_GAMESSUK==1 || KEY_CADPAC==1 || KEY_QUANTUM==1 || \
    KEY_SCCDFTB==1 || KEY_QCHEM==1 || KEY_MNDO97==1 || KEY_SQUANTM==1 || \
    KEY_QTURBO==1 || KEY_G09==1 /*qmtest*/
  gmrem1: IF(QGMREM) THEN
#if KEY_REPLICA==1
     !CC         IF (qRep) CALL WRNDIE(-4,'<NBONDS>',               
#endif
#if KEY_REPLICA==1
     !CC     &    'REPLICA is incompatible with QM/MM method')      
#endif
#if KEY_BLOCK==1
     !CC         IF (QBLOCK) CALL WRNDIE(-4,'<NBONDS>',             
#endif
#if KEY_BLOCK==1
     !CC     &    'BLOCK is incompatible with QM/MM method')        
#endif
#if KEY_PBOUND==1
     If(qBoun) CALL WRNDIE(-1,'<NBONDS>',                   & 
#endif
#if KEY_PBOUND==1
          'PBOUND with QM/MM method is untested')            
#endif
     !CC##IF GAMESS GAMESSUK CADPAC
     !CC         IF(LEWALD) CALL WRNDIE(-4,'<NBONDS>',
     !CC     &     'EWALD is incompatible with QM/MM method')
     !CC##ENDIF
#if KEY_NOST2==0
     IF(NST2 > 0) CALL WRNDIE(-4,'<NBONDS>',               & 
#endif
#if KEY_NOST2==0
          'ST2 waters not supported with QM/MM method')         
#endif
#if KEY_FOURD==1
     IF (DIM4) CALL WRNDIE(-4,'<NBONDS>',                   & 
#endif
#if KEY_FOURD==1
          'DIM4 is incompatible with QM/MM method')           
#endif
     !-checked-OK-C##IF GAMESS GAMESSUK CADPAC
     !-checked-OK-C         IF(LGROUP) CALL WRNDIE(-4,'<NBONDS>',
     !-checked-OK-C     &   'Group lists are not supported with QM/MM method')
     !-checked-OK-C##ENDIF
#if KEY_MMFF==1
     IF(FFIELD == MMFF) CALL WRNDIE(-4,'<NBONDS>',          & 
#endif
#if KEY_MMFF==1
          'MMFF is incompatible with QM/MM method')           
#endif
#if KEY_MNDO97==1 || KEY_SQUANTM==1
#if KEY_REPLICA==1
     IF (qRep) CALL WRNDIE(-4,'<NBONDS>',                   & 
#endif
#if KEY_REPLICA==1
          'REPLICA is not supported with MNDO97 method')       
#endif
#if KEY_BLOCK==1
     IF (QBLOCK) CALL WRNDIE(-4,'<NBONDS>',                 & 
#endif
#if KEY_BLOCK==1
          'BLOCK is not supported with MNDO97 method')         
#endif
#if KEY_MNDO97==0
     IF(.NOT.LGROUP.AND.LEWALD) CALL WRNDIE(-4,'<NBONDS>', &
          'Atom lists are not compatible with EWALD and QM/MM method')
#endif
#endif 
  ENDIF gmrem1
#endif /* (qmtest)*/
  !==
  !==  7. PARALLEL
  !==
#if KEY_PARALLEL==1
  numnodgt2: IF(NUMNOD >= 2) THEN
#if KEY_NOST2==0
     IF(NST2 > 0) CALL WRNDIE(-4,'<NBONDS>',            & 
#endif
#if KEY_NOST2==0
          'ST2 waters not supported with PARALLEL methods')  
#endif
  ENDIF numnodgt2
#endif 
  !==
  !==  8. REPLICA/BLOCK
  !==
#if KEY_REPLICA==1
  IF (qRep) THEN
     IF(LEWALD) CALL WRNDIE(-4,'<NBONDS>', &
          'EWALD is incompatible with REPLICA')
#if KEY_NOST2==0
     IF(NST2 > 0) CALL WRNDIE(-4,'<NBONDS>',            & 
#endif
#if KEY_NOST2==0
          'ST2 waters not supported with REPLICA')         
#endif
#if KEY_FOURD==1
     IF (DIM4) CALL WRNDIE(-1,'<NBONDS>',                & 
#endif
#if KEY_FOURD==1
          'DIM4 is untested with REPLICA - good luck')     
#endif
     IF(LGROUP) CALL WRNDIE(-4,'<NBONDS>', &
          'Group lists are not supported with REPLICA')
  ENDIF
#endif /*  REPLICA*/
#if KEY_BLOCK==1
  IF (QBLOCK) THEN
     !.ab.         IF(LEWALD) CALL WRNDIE(-4,'<NBONDS>',
     !.ab.Ok with HYBH.
     ! av_080628         IF(LEWALD.AND..NOT.QHYBH) CALL WRNDIE(-4,'<NBONDS>',
     ! av_080628     &     'EWALD is incompatible with BLOCK')
#if KEY_NOST2==0
     IF(NST2 > 0) CALL WRNDIE(-4,'<NBONDS>',            & 
#endif
#if KEY_NOST2==0
          'ST2 waters not supported with BLOCK')           
#endif
#if KEY_FOURD==1
     IF (DIM4) CALL WRNDIE(-1,'<NBONDS>',                & 
#endif
#if KEY_FOURD==1
          'DIM4 is untested with BLOCK - good luck')       
#endif
     IF(LGROUP) CALL WRNDIE(-4,'<NBONDS>', &
          'Group lists are not supported with BLOCK')
  ENDIF
#endif 
  !==
  !==  9. PBOUND   (tests compteted)
  !==
  !==  10. EWALD
  !==
  IF(LEWALD) THEN
#if KEY_NOST2==0
     IF(NST2 > 0) CALL WRNDIE(-4,'<NBONDS>',            & 
#endif
#if KEY_NOST2==0
          'ST2 waters not supported with EWALD')           
#endif
#if KEY_FOURD==1
     IF (DIM4) CALL WRNDIE(-4,'<NBONDS>',                & 
#endif
#if KEY_FOURD==1
          'DIM4 is not supported with EWALD')              
#endif
     !CC         IF(LGROUP) CALL WRNDIE(-4,'<NBONDS>',
     !CC     &     'Group lists are not yet supported with EWALD')
  ENDIF
  !==
  !==  11. ST2
  !==
#if KEY_NOST2==0 /*st2warn*/
  IF(NST2 > 0) THEN
#if KEY_FOURD==1
     IF (DIM4) CALL WRNDIE(-4,'<NBONDS>',                & 
#endif
#if KEY_FOURD==1
          'DIM4 is not supported with ST2')                
#endif
     IF(.NOT.LGROUP) CALL WRNDIE(-4,'<NBONDS>', &
          'Atom lists are not yet supported with ST2')
#if KEY_MMFF==1
     IF(FFIELD == MMFF) CALL WRNDIE(-4,'<NBONDS>',       & 
#endif
#if KEY_MMFF==1
          'ST2 is not supported with MMFF')                
#endif
  ENDIF
#endif /* (st2warn)  IFN NOST2*/
  !==
  !==  12. DIM4
  !==
#if KEY_FOURD==1
  IF(DIM4) THEN
     IF(LGROUP) CALL WRNDIE(-4,'<NBONDS>', &
          'Group lists are not yet supported with DIM4')
#if KEY_MMFF==1
     IF(FFIELD == MMFF) CALL WRNDIE(-4,'<NBONDS>',       & 
#endif
#if KEY_MMFF==1
          'DIM4 is not supported with MMFF')               
#endif
  ENDIF
#endif 
  !==
  !==  14.  MTS
  !==
  !==
  !==  15. ACE
  !==
#if KEY_ACE==1 /*acetests*/
  IF(LACE) THEN
     IF(LEXS) CALL WRNDIE(-1,'<NBONDS>', &
          'ACE is untested with LEXS - good luck!')
#if KEY_NOMISC==0
     IF(QEXTND) CALL WRNDIE(-4,'<NBONDS>',               & 
#endif
#if KEY_NOMISC==0
          'ACE not yet implemented with EXELEC')           
#endif
#if KEY_NOMISC==0
     IF(QRXNFL) CALL WRNDIE(-4,'<NBONDS>',               & 
#endif
#if KEY_NOMISC==0
          'ACE incompatible with RXNFLD')                  
#endif
#if KEY_PERT==1
     IF(QPERT) CALL WRNDIE(-4,'<NBONDS>',                & 
#endif
#if KEY_PERT==1
          'ACE is incompatible? with PERT')                
#endif
#if KEY_MTS==1
     IF(TBHY1.OR.SLFG) CALL WRNDIE(-4,'<NBONDS>',        & 
#endif
#if KEY_MTS==1
          'ACE not yet implemented with MTS')              
#endif
#if KEY_TSM==1
     IF (QTSM) CALL WRNDIE(-4,'<NBONDS>',                & 
#endif
#if KEY_TSM==1
          'ACE is incompatible? with TSM')                 
#endif
#if KEY_GAMESS==1 || KEY_GAMESSUK==1 || KEY_QUANTUM==1 || KEY_CADPAC==1 || \
    KEY_SCCDFTB==1 || KEY_QCHEM==1 || KEY_MNDO97==1 || KEY_SQUANTM==1 || \
    KEY_QTURBO==1 || KEY_G09==1
     IF(QGMREM) CALL WRNDIE(-4,'<NBONDS>', &
          'ACE is incompatible? with QM/MM method')
#endif 
#if KEY_PBOUND==1
     IF(qBoun) CALL WRNDIE(-1,'<NBONDS>',                & 
#endif
#if KEY_PBOUND==1
          'ACE is untested with PBOUND - good luck!')      
#endif
     IF(LEWALD) CALL WRNDIE(-4,'<NBONDS>', &
          'ACE is incompatible with EWALD')
#if KEY_NOST2==0
     IF(NST2 > 0) CALL WRNDIE(-1,'<NBONDS>',            & 
#endif
#if KEY_NOST2==0
          'ACE is untested with ST2 waters - good luck')   
#endif
#if KEY_FOURD==1
     IF (DIM4) CALL WRNDIE(-4,'<NBONDS>',                & 
#endif
#if KEY_FOURD==1
          'ACE not yet implemented with DIM4')             
#endif
#if KEY_MMFF==1
     IF(FFIELD == MMFF) CALL WRNDIE(-1,'<NBONDS>',       & 
#endif
#if KEY_MMFF==1
          'ACE is untested with MMFF - good luck!')        
#endif
  ENDIF
#endif /* (acetests)*/
  IF(QDCM) CALL DCMNBUP() ! Notify DCM module of nbond list update
  !==
  !   Done with option compatibility testing.
  !=======================================================================
  !
  ! Allocate space for coordinates set: "since last update"
   call allocate_inbnd(natom)  

  ! Allocate temporary work space
  ! --Edited and reconfigured this section, -RJP 4.7.00
#if KEY_NO_BYCC==0
  IF(LBYCC) THEN
     call chmalloc('nbonds.src','NBONDS','NUMMP',MXNCUB,intg=NUMMP)
     call chmalloc('nbonds.src','NBONDS','HIMPRTN',MXNCUB,intg=HIMPRTN)
     call chmalloc('nbonds.src','NBONDS','MMM',MXNCUB,intg=MMM)
     call chmalloc('nbonds.src','NBONDS','LSTM0',MXNCUB,intg=LSTM0)
     call chmalloc('nbonds.src','NBONDS','CNTN',MXNCUB,intg=CNTN)
     call chmalloc('nbonds.src','NBONDS','XO',MXNCUB,intg=XO)
     call chmalloc('nbonds.src','NBONDS','YO',MXNCUB,intg=YO)
     call chmalloc('nbonds.src','NBONDS','ZO',MXNCUB,intg=ZO)
     IF(.NOT.LGROUP) THEN
        call chmalloc('nbonds.src','NBONDS','LSTN0',NACTC,intg=LSTN0)
        call chmalloc('nbonds.src','NBONDS','XOO',NACTC,intg=XOO)
        call chmalloc('nbonds.src','NBONDS','YOO',NACTC,intg=YOO)
        call chmalloc('nbonds.src','NBONDS','ZOO',NACTC,intg=ZOO)
        call chmalloc('nbonds.src','NBONDS','ORDCBN',NACTC,intg=ORDCBN)
        call chmalloc('nbonds.src','NBONDS','PNTNER',NACTC,intg=PNTNER)
        call chmalloc('nbonds.src','NBONDS','PNTDIS',NACTC,intg=PNTDIS)
        call chmalloc('nbonds.src','NBONDS','PARTNN',NACTC,intg=PARTNN)
        call chmalloc('nbonds.src','NBONDS','PARTND',NACTC,intg=PARTND)
        call chmalloc('nbonds.src','NBONDS','DUMMYX',NACTC,intg=DUMMYX)
        call chmalloc('nbonds.src','NBONDS','HAFMAR',NACTC,crl=HAFMAR)
        call chmalloc('nbonds.src','NBONDS','MEANX',NACTC,crl=MEANX)
        call chmalloc('nbonds.src','NBONDS','MEANY',NACTC,crl=MEANY)
        call chmalloc('nbonds.src','NBONDS','MEANZ',NACTC,crl=MEANZ)
     ELSE
        call chmalloc('nbonds.src','NBONDS','LSTN0',NGRP,intg=LSTN0)
        call chmalloc('nbonds.src','NBONDS','XOO',NGRP,intg=XOO)
        call chmalloc('nbonds.src','NBONDS','YOO',NGRP,intg=YOO)
        call chmalloc('nbonds.src','NBONDS','ZOO',NGRP,intg=ZOO)
        call chmalloc('nbonds.src','NBONDS','ORDCBN',NGRP,intg=ORDCBN)
     ENDIF
  ENDIF
#endif 
  nccngr: IF((.NOT.LBYCC).OR.(LGROUP)) THEN
     NGRPT=MAX(NGRP,NIMGRP)
     if (.not.allocated(rscmx)) then
        call chmalloc('nbonds.src','NBONDS','rscmx',NGRPt,crl=rscmx)
        call chmalloc('nbonds.src','NBONDS','rscmx',NGRPt,crl=rscmy)
        call chmalloc('nbonds.src','NBONDS','rscmz',NGRPt,crl=rscmz)
        call chmalloc('nbonds.src','NBONDS','rsxmax', NGRPt,crl=rsxmax)
        call chmalloc('nbonds.src','NBONDS','rsymax', NGRPt,crl=rsymax)
        call chmalloc('nbonds.src','NBONDS','rszmax', NGRPt,crl=rszmax)
     elseif (ngrpt > size(rscmx)) then
        call chmrealloc('nbonds.src','NBONDS','rscmx',NGRPt,crl=rscmx)
        call chmrealloc('nbonds.src','NBONDS','rscmx',NGRPt,crl=rscmy)
        call chmrealloc('nbonds.src','NBONDS','rscmz',NGRPt,crl=rscmz)
        call chmrealloc('nbonds.src','NBONDS','rsxmax', NGRPt,crl=rsxmax)
        call chmrealloc('nbonds.src','NBONDS','rsymax', NGRPt,crl=rsymax)
        call chmrealloc('nbonds.src','NBONDS','rszmax', NGRPt,crl=rszmax)
     endif
     IF(QEXTND) THEN
        call chmalloc('nbonds.src','NBONDS','RSQ',NGRP,crl=RSQ)
        call chmalloc('nbonds.src','NBONDS','RSDX',NGRP,crl=RSDX)
        call chmalloc('nbonds.src','NBONDS','RSDY',NGRP,crl=RSDY)
        call chmalloc('nbonds.src','NBONDS','RSDZ',NGRP,crl=RSDZ)
        call chmalloc('nbonds.src','NBONDS','RSPOT',NGRP,crl=RSPOT)
        call chmalloc('nbonds.src','NBONDS','RSFX',NGRP,crl=RSFX)
        call chmalloc('nbonds.src','NBONDS','RSFY',NGRP,crl=RSFY)
        call chmalloc('nbonds.src','NBONDS','RSFZ',NGRP,crl=RSFZ)
        call chmalloc('nbonds.src','NBONDS','RSGXX',NGRP,crl=RSGXX)
        call chmalloc('nbonds.src','NBONDS','RSGYY',NGRP,crl=RSGYY)
        call chmalloc('nbonds.src','NBONDS','RSGZZ',NGRP,crl=RSGZZ)
        call chmalloc('nbonds.src','NBONDS','RSGXY',NGRP,crl=RSGXY)
        call chmalloc('nbonds.src','NBONDS','RSGYZ',NGRP,crl=RSGYZ)
        call chmalloc('nbonds.src','NBONDS','RSGZX',NGRP,crl=RSGZX)

        IF (QXQUAD) THEN
           call chmalloc('nbonds.src','NBONDS','RSQXX',NGRP,crl=RSQXX)
           call chmalloc('nbonds.src','NBONDS','RSQYY',NGRP,crl=RSQYY)
           call chmalloc('nbonds.src','NBONDS','RSQZZ',NGRP,crl=RSQZZ)
           call chmalloc('nbonds.src','NBONDS','RSQXY',NGRP,crl=RSQXY)
           call chmalloc('nbonds.src','NBONDS','RSQYZ',NGRP,crl=RSQYZ)
           call chmalloc('nbonds.src','NBONDS','RSQZX',NGRP,crl=RSQZX)
        ENDIF
     ENDIF
     call chmalloc('nbonds.src','NBONDS','rsdisp', NGRPt,intg=rsdisp)
#if KEY_FOURD==1
     IF (DIM4) THEN
        call chmalloc('nbonds.src','NBONDS','rszmax', NGRPt,crl=rsfmax)
        call chmalloc('nbonds.src','NBONDS','rszmax', NGRPt,crl=rscmf)
     ELSE
        call chmalloc('nbonds.src','NBONDS','rszmax', 1,crl=rsfmax)
        call chmalloc('nbonds.src','NBONDS','rszmax', 1,crl=rscmf)
     ENDIF
#endif 
  ENDIF nccngr
  !
  !=======================================================================
! count the possible number of pairs for the thole shielding lists
!
!
      IF(NBTHOL.GT.0) THEN
        if(MAXNBTHOLE.LT.0)then
! just use the dimension passed in the parameter file
        MAXNBTHOLE=abs(MAXNBTHOLE)
        if (prnlev >= 2) write(outu,'(a,i6,a)') ' Allocate space for ',MAXNBTHOLE, &
             ' Thole shielding pairs '
  call chmalloc('nbonds.src','NBONDS','NBTHOL1',MAXNBTHOLE,intg=NBTHOL1)
  call chmalloc('nbonds.src','NBONDS','NBTHOL2',MAXNBTHOLE,intg=NBTHOL2)
  call chmalloc('nbonds.src','NBONDS','NBTHOL3',MAXNBTHOLE,intg=NBTHOL3)
  call chmalloc('nbonds.src','NBONDS','PPNBTHOL1',MAXNBTHOLE,intg=PPNBTHOL1)
  call chmalloc('nbonds.src','NBONDS','PPNBTHOL2',MAXNBTHOLE,intg=PPNBTHOL2)
  call chmalloc('nbonds.src','NBONDS','PPNBTHOL3',MAXNBTHOLE,intg=PPNBTHOL3)

        elseif(MAXNBTHOLE.eq.0)then
! automatically set the dimension needed
        xmin=x(1)
        xmax=x(1)
        ymin=y(1)
        ymax=y(1)
        zmin=z(1)
        zmax=z(1)
        do i=1,NATOM
           if(x(i).lt.xmin) xmin=x(i)
           if(x(i).gt.xmax) xmax=x(i)
           if(y(i).lt.ymin) ymin=y(i)
           if(y(i).gt.ymax) ymax=y(i)
           if(z(i).lt.zmin) zmin=z(i)
           if(z(i).gt.zmax) zmax=z(i)
        enddo
!       write(outu,*) xmin,xmax,ymin,ymax,zmin,zmax
!       write(outu,*) (xmax-xmin)*(ymax-ymin)*(zmax-zmin)
!       write(outu,*) THOLCUT**3/((xmax-xmin)*(ymax-ymin)*(zmax-zmin))
        n3=0
        do i=1,NBTHOL
           n1=0
           n2=0
           do j=1,NATOM
              if(IAC(j).eq.NBTHOLIJ(1,i))then
                 n1=n1+1
              endif
              if(IAC(j).eq.NBTHOLIJ(2,i))then
                 n2=n2+1
              endif
           enddo
           n3 = n3 + n1*n2
        enddo
        n3 = n3*(THOLCUT**3/((xmax-xmin)*(ymax-ymin)*(zmax-zmin)))
        MAXNBTHOLE = max(n3,MAXNBTHOLE)+10

        if (prnlev >= 2) write(outu,'(a,i6,a)') ' Allocate space for ',MAXNBTHOLE, &
             ' Thole shielding pairs '
        call chmalloc('nbonds.src','NBONDS','NBTHOL1',MAXNBTHOLE,intg=NBTHOL1)
  ! NBTHOL1    first atom in the mini nonbonded list of pair-specific Thole
        call chmalloc('nbonds.src','NBONDS','NBTHOL2',MAXNBTHOLE,intg=NBTHOL2)
  ! NBTHOL2    second atom in the mini nonbonded list of pair-specific Thole
        call chmalloc('nbonds.src','NBONDS','NBTHOL3',MAXNBTHOLE,intg=NBTHOL3)
  ! NBTHOL3    pointer for the pairwise Thole factor in NBTHOLXIJ
        call chmalloc('nbonds.src','NBONDS','PPNBTHOL1',MAXNBTHOLE,intg=PPNBTHOL1)
  ! PPNBTHOL1    first atom in the mini nonbonded list of pair-specific Thole, state lambda=0
        call chmalloc('nbonds.src','NBONDS','PPNBTHOL2',MAXNBTHOLE,intg=PPNBTHOL2)
  ! PPNBTHOL2    second atom in the mini nonbonded list of pair-specific Thole, state lambda=0
        call chmalloc('nbonds.src','NBONDS','PPNBTHOL3',MAXNBTHOLE,intg=PPNBTHOL3)
  ! PPNBTHOL3  pointer for the pairwise Thole factor in NBTHOLXIJ, state lambda=0

        endif
      ENDIF
  !
  ! Allocate long term space for lists
  !
  !     TO ALLOCATE SPACE FOR JNB MAKE A GUESS EMPERICALLY, AND COMPARE IT
  !     TO THE CURRENTLY ALLOCATED SPACE. USE THE LARGER OF THE TWO AND
  !     COMPARE THAT TO THE MAXIMUM POSSIBLE NUMBER OF PAIRS, REDUCING
  !     IT IF NECESSARY.
  !
  !CC This code overhauled - BRB & MH - 11/11/91
  ! Estimate half of maximum space for a dense system (water)
  ! assuming 10A**3 for each atom. 0.20944=4/3*PI/10/2/2
  RMXGES=NBSCAL*CUTNB**3*NATOM*0.10472
  RVAL=(NBSCAL*(CUTNB+ONE)**3*NGRP*NGRP*0.10472)/NATOM
  BIGGR=NGRP*((NGRP+2)/2)
  !+ln
  IF(NATOM  <  65536)THEN
     BIGGST=NATOM*((NATOM-1)/2)+1
  ELSE
     BIGGST=2147483647
  ENDIF
  !-ln
  ! BEGIN G. Lamoureux
  IF (QDRUDE) THEN
     IF(NATOM >= 32768) THEN
        BIGGST=2147483647
     ELSE
        BIGGST=2*NATOM*NATOM+1
     ENDIF
  ENDIF
  ! END G. Lamoureux
  !
#if KEY_PERT==1
  IF(QPERT) THEN
     IF(LGROUP) THEN
        RFACT=2*NIPERT*RVAL
        RFACT=RFACT/NATOM + 100
        IF(RFACT > BIGGR) RFACT=BIGGR
        IF(INT(RFACT) > MXJNGR) MXJNGR=INT(RFACT)
        IF(INT(RFACT) > MXJNGP) MXJNGP=INT(RFACT)
        RFACT=(NATOM-NIPERT)*RVAL
        RVAL=RFACT/NATOM + 100
        MAXJNB=0
        MAXJNP=0
        MAXJNR=0
     ELSE
        RFACT=2*NIPERT*RMXGES
        RFACT=RFACT/NATOM + 100
        IF(RFACT > BIGGST) RFACT=BIGGST
        IF(INT(RFACT) > MAXJNR) MAXJNR=INT(RFACT)
        IF(INT(RFACT) > MAXJNP) MAXJNP=INT(RFACT)
        RFACT=(NATOM-NIPERT)*RMXGES
        RMXGES=RFACT/NATOM + 100
        MXJNBG=0
        MXJNGP=0
        MXJNGR=0
     ENDIF
     NNNBP=0
     NNNBR=0
     NNNBGP=0
     NNNBGR=0
  ENDIF
#endif 
  !
  IF(LGROUP) THEN
     IF(RVAL > BIGGR) RVAL=BIGGR
     IF(INT(RVAL) > MXJNBG) MXJNBG=INT(RVAL)
#if KEY_NOST2==0
  ELSE IF(NST2 > 0) THEN
     RVAL=NBSCAL*CUTNB**3*NST2*THIRD*0.10472
     BIGGR=NST2*(NST2-1)/2 +1
     IF(RVAL > BIGGR) RVAL=BIGGR
     IF(INT(RVAL) > MXJNBG) MXJNBG=INT(RVAL)
#endif 
  ELSE
     BIGGR=0
     MXJNBG=0
  ENDIF
  !
  IF(RMXGES > BIGGST) RMXGES=BIGGST
  !
  IF (LGROUP) MAXJNB=0
#if KEY_IMCUBES==1
  !        =========================================================
  IF(LBYCBIM &
#if KEY_DOMDEC==1
       .and. .not.q_domdec &  
#endif
       )THEN
     natimcu=natim
     IF(NTRNSX <= 0) natimcu=natom
     doimages=.true.
     if(natimcu == natom)doimages=.false.
     cuMAXnB=max(cumaxnb,1)
     cuMAXMB=max(cumaxmb,1)
     MXJNBG=0
     MXJMBG=0
     CALL RENBND(BNBND,CUMAXNB,NATOM,NGRP,MXJNBG)
     if (doimages) CALL REIMAG(BIMAG,CUMAXMB,MXJMBG)
  elseIF (RMXGES > MAXJNB) then
     MAXJNB=RMXGES
  ENDIF
  !        ========================================================
#else /**/
  IF (RMXGES > MAXJNB) MAXJNB=RMXGES
#endif 
  !
#if KEY_MTS==1
  IF(TBHY1.OR.SLFG) THEN
     MAXJM1=MAXJNB
     MAXJM2=MAXJNB
     NNMT1G = 0
     NNMT2G = 0
     NNMT1  = 0
     NNMT2  = 0
  ENDIF
#endif 
  !
#if KEY_PARALLEL==1 /*paratest*/
#if KEY_VIBPARA==0
  RVAL=ONE
  IF(NUMNOD >= 2) THEN
     IF(NGRP > NUMNOD*10) RVAL=ONE/NUMNOD
     IF(NGRP > NUMNOD*6)  RVAL=ONEPT5/NUMNOD
     IF(NGRP > NUMNOD*3)  RVAL=TWO/NUMNOD
  ENDIF
#if KEY_PARASCAL==1 /*parasc*/
  IF(.NOT.QPSRNB) RVAL=ONE
#endif /* (parasc)*/
  MXJNBG=MXJNBG*RVAL
  MAXJNB_tmp=MAXJNB*RVAL
#if KEY_PERT==1
  IF(QPERT) THEN
     MAXJNP=MAXJNP*RVAL
     MAXJNR=MAXJNR*RVAL
     MXJNGP=MXJNGP*RVAL
     MXJNGR=MXJNGR*RVAL
  ENDIF
#endif 
#endif 
  if (maxjnb_tmp > maxjnb) maxjnb = maxjnb_tmp
#endif /* (paratest)*/
  !CC
  !=======================================================================
  ! .  should we use the vector/parallel routines for nbond generation?
  !
  ! Developers: Please keep this section up to date....
  !
  ! Nonbond List Generation Routine:
  !  a.  NBONDA  - (general)   generate atom lists
  !  b.  NBONDG  - (general)   generate group lists
  !  c.  NBNDGC  -             Nonbond list by cubes
  !  f.  NBONDMA - (general)   image lists (atom)
  !  g.  NBONDMG - (general)   image lists (group)
  !  h.  NBNDCCF (fastest)     
  !  i.  NBNDCCO
  !  j.  NBNDGG  (groups)
  !  k.  NBNDGCM - (IMCUBES)   atom and image lists
  !
  ! Nonbond Methods Support table:
  !                      a   b   c     f   g   h   i   j   k
  !                     --  --  --    --  --  --  --  --  --
  ! 1. LEXS              o   o   x     x   x   x   x   x    ?
  ! 2. EXELEC/RXNFLD     x   o   o     x   x   x   x   o    ?
  ! 3. PERT              o   o   x     o   o   x   x   x    o
  ! 4. MTS               o   o   x     o   o   x   x   x    ?
  ! 5. TSM               o   x   o     o   x   x   o   x    o
  ! 6. QUANTUM/GAMESS    o   x   x     x   x   ?   ?   ?    ?
  !    CADPAC/GAMESSUK/SCCDFTB/MNDO97/SQUANTM
  ! 7. PARALLEL          o   o   x     o   o   x   o  bug   o
  ! 8. REPLICA           o   x   o     o   x   x   x   x    o
  ! 9. PBOUND            o   o   x    (o) (o)  x   x   x    x
  ! 10. EWALD            o   o   o     o   o   x   x   x    o
  ! 11. ST2              x   o   x     x   o   x   x   x    ?
  ! 12. DIM4             o   x   x     o   x   x   x   x    ?
  ! 13. MMFF             o   x   o     o   x   o   o   x    o
  !
  !  x - not supported     o - supported     (o) - implicitly supported  ? - not known
  !
  ! NOTE-- addded "LBYCC" flag to below if tests--RJP
  ! Try to use fast if requested...
  LSLOWNB = (FASTER < 0.OR.(.NOT.QFASTNB))
  ! fast routines do not support GROUP, or exclude segment options.
  IF(LEXS.OR.LGROUP.OR.QEXTND) LSLOWNB = .TRUE.
#if KEY_PARALLEL==1
  ! fast routines do not support parallel
  IF(NUMNOD >= 2) THEN
     LSLOWNB=.TRUE.
     LBYCU=.FALSE.
     !CC         LBYCC=.FALSE.
  ENDIF
#endif 
#if KEY_PERT==1
  ! fast routines do not support PERT
  IF(QPERT) THEN
     LSLOWNB=.TRUE.
     LBYCU=.FALSE.
     LBYCC=.FALSE.
     !sbbug disable use of BYCB as well
#if KEY_IMCUBES==1
     LBYCBIM=.FALSE.
#endif 
     !sbbug ... are we now catching all inappropriate list routines???
  ENDIF
#endif 
#if KEY_MTS==1
  ! fast routines do not support MTS
  IF(TBHY1.OR.SLFG) THEN
     LSLOWNB=.TRUE.
     LBYCU=.FALSE.
     LBYCC=.FALSE.
  ENDIF
#endif 
#if KEY_NOST2==0
  ! fast routines do not support ST2 water model
  IF(NST2 > 0) THEN
     LSLOWNB=.TRUE.
     LBYCU=.FALSE.
     LBYCC=.FALSE.
  ENDIF
#endif 
  LSLOWNB = .TRUE.
  !
#if KEY_GAMESS==1 || KEY_GAMESSUK==1 || KEY_QUANTUM==1 || KEY_CADPAC==1 || \
    KEY_SCCDFTB==1 || KEY_QCHEM==1 || KEY_MNDO97==1 || KEY_SQUANTM==1 || \
    KEY_QTURBO==1 || KEY_G09==1
  ! fast routines do not support QUANTUM/GAMESS/CADPAC/SCCDFTB/MNDO97/SQUANTM
  ! it would be better to use qmused ??!!
  IF(QGMREM) THEN
     LSLOWNB = .TRUE.
     LBYCU=.FALSE.
     LBYCC=.FALSE.
  ENDIF
#endif 
#if KEY_TSM==1 /*tsmtest*/
  ! Some fast routines do not support TSM
  IF (QTSM) THEN
     LSLOWNB = .TRUE.
  ENDIF
#endif /* (tsmtest)*/
#if KEY_PBOUND==1 /*pbound*/
  If(qBoun) then
     lbycu=.false.
     lbycc=.false.
#if KEY_IMCUBES==1
     lbycbim=.false.    
#endif
  Endif
#endif /*      (pbound)*/
#if KEY_FOURD==1 /*4dadd*/
  IF (DIM4) THEN
     LSLOWNB=.TRUE.
     LBYCU=.FALSE.
     LBYCC=.FALSE.
  ENDIF
#endif /* (4dadd)*/
#if KEY_DOMDEC==1
  if (.not.q_domdec) then 
#endif
#if KEY_IMCUBES==1
     if(ntrnsx > 0) then
        if (lbycbim) &
             call transo(x, y, z, &
             (/ zero /), (/ zero /), (/ zero /), &
             .false., .false., &
             (/ zero /), &
             natom,ntrans,imtrns, &
             bimag%imatpt,bimag%imattr,norot,natim &
#if KEY_FLUCQ==1
             ,qfluc,cg,fqcfor           & 
#endif
             )
     endif
#endif 
#if KEY_DOMDEC==1
  endif  
#endif
  !
  !=======================================================================
  ! Generate nonbond list.
  IF (LBYCC.AND.(NTRANS > 0)) CALL WRNDIE(-5,'<NBONDS>', &
       'BYCC not currently compatible with images. Use BYCBimages')
#if KEY_SPACDEC==1
  IF (.NOT.LBYCC) CALL WRNDIE(-5,'<NBONDS>', &
       'SPACDEC compilation requires BYCC option in nbonds command')
  IF (LGROUP) CALL WRNDIE(-5,'<NBONDS>', &
       'SPACDEC compilation not currently compatible with groups')
#endif 
#if KEY_NO_BYCC==0 /*no_bycc*/
  bycc1: IF (LBYCC) THEN
     !   options
     LBYCCF = .TRUE. !fastest
     ! test for options
#if KEY_TSM==1
     IF(QTSM) THEN
        LBYCCF = .FALSE.
     ENDIF
#endif 
#if KEY_SPACDEC==1
     LBYCCF = .FALSE.
#endif 
     !   allocation for nonbonded list
     IF (.NOT. LGROUP) THEN
        PRTDEN = 0.1019 !atom density
        NCTPRT = NACTVE
     ELSE
        PRTDEN = 0.031 !group density
        NCTPRT = NACTG
     ENDIF
     !   allocation for nonbonded listt
     RLIMIT = ((0.2387324*NCTPRT)/PRTDEN)**(0.333333333)
     RLIMIT = 2*RLIMIT
     IF (CUTNB <= RLIMIT) THEN
        AMPLIT = 0.5235988*PRTDEN
        AMPLIT = AMPLIT*AMPLIT
        CTNBCU = CUTNB*CUTNB*CUTNB
        CTNBFR = CTNBCU*CUTNB
        CTNBSX = CTNBFR*CUTNB*CUTNB
        THRDTM = (7.6394373/PRTDEN)*NCTPRT*CTNBCU
        SECDTM = (NCTPRT/PRTDEN)**(0.666666666)
        SECDTM =  -6.9270256*SECDTM*CTNBFR
        IMAXJNB = NBSCAL*(INT(AMPLIT*(CTNBSX + SECDTM + THRDTM))  &
             + 1000)
     ELSE
        IMAXJNB = NBSCAL*(NCTPRT*NCTPRT)
     ENDIF
     IF (.NOT. LGROUP) THEN !using atom list
#if KEY_PARALLEL==1
        IF(NUMNOD == 1) THEN 
#endif
           CCMAXJNB = MAX(IMAXJNB,CCMAXJNB)
           MXJNBG = 0
           NNNBG = 0
           CCMXJNBG = 0
           ! allocations for cluster-cluster lists
           P3 = (CUTNB+3)*(CUTNB+3)*(CUTNB+3)
           NACLCU = NACTC**3
           NACLCU = NACTC*NACTC
           NACLCU = NACLCU*NACTC
           NUMER2 = P3*0.95/NACTVE
           NUMER2 = NUMER2*NACLCU
           DENOM2 = 5*NACTC +  P3*1.6*NACTC/NACTVE
           MAXACL = MAX(INT(NUMER2/DENOM2) + 4*NACTC,MAXACL)
           M2 = (CUTNB-2)*(CUTNB-2)*(CUTNB-2)
           NUMER3 = M2*0.9/NACTVE
           NUMER3 = NUMER3*NACLCU
           DENOM3 = M2*1.6*NACTC/NACTVE
           DENOM3 = DENOM3 + 5*NACTC
           MAXACN = MAX(INT(NUMER3/DENOM3) +  &
                INT(2*NACTC)+1000,MAXACN)
           MAXACD = MAX(MAXACL - MAXACD + 1000,MAXACD)
           IF (MAXACD < NACTC) MAXACD = NACTC
           IF (MAXACN < NACTC) MAXACN = NACTC
           IF ((PRNLEV > 2).AND.(.NOT.(DYNAMQ.OR.MINXYZ))) THEN
              WRITE(OUTU,*) 'ALLOCATING SPACE FOR ',MAXACN, &
                   ' NEAR ACTIVE CLUSTER PAIRS'
              WRITE(OUTU,*) 'AND FOR ',MAXACD, &
                   ' DISTANT ACTIVE CLUSTER PAIRS'
           ENDIF
           MAXACD = MAX(MAXACD,INT(CCMAXJNB/9))
           MAXACN = MAX(MAXACN,INT(CCMAXJNB/9))
#if KEY_PARALLEL==1
        ENDIF  /*if numnod is one*/
#endif
#if KEY_PARALLEL==1 /*memory*/
        IF(NUMNOD > 1) THEN
           RNUMNOD = NUMNOD
           MEMFAC = LOG(RNUMNOD)/1.5 + 1
           MAXACD = MEMFAC*INT(IMAXJNB/(9*NUMNOD))
           MAXACN = MEMFAC*INT(IMAXJNB/(9*NUMNOD))
           CCMAXJNB = INT(IMAXJNB/NUMNOD)*MEMFAC
        ENDIF
#endif /* (memory)*/
     ELSE   !if using group list
        CCMXJNBG =  MAX(IMAXJNB,CCMXJNBG)
        MAXJNB = 0
        NNNB = 0
        CCMAXJNB = 0
     ENDIF
  ENDIF bycc1
  !      SAVNBS = NBSCAL
#endif /*  (no_bycc)*/
  !
  !      WRITE(6,*) 'MAXACD is ',MAXACD
  !      WRITE(6,*) 'MAXACN is ',MAXACN
  !      WRITE(6,*) 'CCMAXJNB is ',CCMAXJNB


  !---------------------------------------------------------------------
  !      Loop till Completed
  !---------------------------------------------------------------------

  CMPLTD=.FALSE.
  complete: DO WHILE (.NOT.CMPLTD)
     !
     ! Allocate memory for first pass
     !   (or reallocate memory on subsequent passes)
     ! Added below -RJP
#if KEY_NO_BYCC==0
     IF (LBYCC) THEN
        IF (.NOT.LGROUP) THEN
           MAXJNB = CCMAXJNB
           RNATOM = NATOM
           RHLFMT = RNATOM*(RNATOM-1)/2+1
           IF(RHLFMT < MAXJNB) MAXJNB = INT(RHLFMT)
           call chmalloc('nbonds.src','NBONDS','HPMXCD',MAXACD,intg=HPMXCD)
           call chmalloc('nbonds.src','NBONDS','HPMXCN',MAXACN,intg=HPMXCN)
        ELSE
           MXJNBG = CCMXJNBG
           RNGRP = NGRP
           RGHFMT = RNGRP*((RNGRP+2)/2)
           IF(RGHFMT < MXJNBG) MXJNBG = INT(RGHFMT)
        ENDIF
     ENDIF
     if(lbycc)call chmalloc('nbonds.src','NBONDS','WRKAR',MXNCUB*27,intg=WRKAR)
#endif 
!
#if KEY_IMCUBES==1
     if(.not.lbycbim)then
#endif 
        CALL RENBND(BNBND,MAXJNB,NATOM,NGRP,MXJNBG)
#if KEY_IMCUBES==1
     endif
#endif 

#if KEY_MTS==1
     IF(TBHY1.OR.SLFG) THEN
        CALL RENBND(BNBNM1,MAXJM1,NATOM,NGRP,MXJNBG)
        CALL RENBND(BNBNM2,MAXJM2,NATOM,NGRP,MXJNBG)
     ENDIF
#endif 
!
#if KEY_PERT==1
     IF(QPERT) THEN
        CALL RENBND(BNBNDP,MAXJNP,NATOMP,NGRPP,MXJNGP)
        CALL RENBND(BNBNDR,MAXJNR,NATOM,NGRP,MXJNGR)
     ENDIF
#endif 
     !
     IF(PRNLEV >= 5) THEN
        WRITE(OUTU,210) MAXJNB,MXJNBG
210     FORMAT (' == PRIMARY == SPACE FOR',I9,' ATOM PAIRS AND', &
             I9,' GROUP PAIRS')
#if KEY_PERT==1
        IF(QPERT) THEN
           WRITE(OUTU,212) MAXJNR,MXJNGR
212        FORMAT (' SPACE FOR',I9,' REACTANT ATOM PAIRS, AND', &
                I9,' GROUP PAIRS.')
           WRITE(OUTU,211) MAXJNP,MXJNGP
211        FORMAT (' SPACE FOR',I9,' PRODUCT  ATOM PAIRS, AND', &
                I9,' GROUP PAIRS.')
        ENDIF
#endif 
        !
#if KEY_FACTS==1
        IF (FCTRUN) THEN
           WRITE(OUTU,222) MXFCAB
222        FORMAT (' ==  FACTS  == SPACE FOR',I9, &
                ' FACTS ATOM PAIRS ')
        ENDIF
#endif 
     ENDIF
     !
     !-----------------------------------------------------------------------
     !     Generate the list by cubes.
     !-----------------------------------------------------------------------
#if KEY_IMCUBES==1
     IF (LBYCBIM &
#if KEY_DOMDEC==1
          .and. .not.q_domdec &  
#endif
          ) THEN
        MXJMBG=0
        IF(PRNLEV >= 5) WRITE(OUTU,4410) MAXJMB,MXJMBG
4410    FORMAT (' == IMAGES === SPACE FOR',I9,' ATOM PAIRS AND', &
             I9,' GROUP PAIRS')
        !-----------------------------------------------------------------------
        !       Run a first pass to get the size of the list and
        !          load balance
        !-----------------------------------------------------------------------
215     continue
        if(resize)then
           IF(PRNLEV >= 5) then
              write(outu,*) " "
              write(outu,*) "========================================="
              write(outu,*) "             TRIAL run of list..........."
              write(outu,*) "========================================="
           endif
           if(doimages)then
              bimagx%NIMNBX=0
              bimagx%NIMNBS=0
           endif
           IF(PRNLEV > 6) WRITE(OUTU,125) 'NBNDGCM'
           CALL NBNDGCM(NATIMcu,NIMGRP, &
                NNNB, bnbnd%JNB, MAXJNB, &
                bnbnd%INBLO, X, Y, Z, &
                bimag%NIMNB,bimag%IMJNB,MAXJMB, &
                bimag%IMBLO, &
                bimag%IMINB,bimag%IMIBLO, &
                bimag%NIMNBS, bimag%IMJNBS, bimag%IMBLOS, &
                NNNBG, MXJMBG, &
                bnbnd%JNBG, &
                bnbnd%INBLOG, bnbnd%INB14, &
                bnbnd%IBLO14, &
                bimag%IMING, bimag%IMIGLO, &
                liminv,iminv,ntrans,bimag%IMATPT, &
                bimag%IMATTR, &
                CUTNB, CTEXNB, LGROUP, &
                QEXTND, QXQUAD, QXGRAD, LVATOM, WRNMIN, CMPLTD, EPS, &
#if KEY_TSM==1
                QTSM,REACLS,PRODLS, &
#endif 
                RSCMX,RSCMY,RSCMZ, &
                RSQ, &
                RSDX,RSDY,RSDZ,RSQXX, &
                RSQYY,RSQZZ,RSQXY,RSQYZ, &
                RSQZX, &
                RSXMAX,RSYMAX,RSZMAX,RSDISP, &
                RSPOT,RSFX, RSFY, &
                RSFZ,RSGXX,RSGYY,RSGZZ, &
                RSGXY,RSGYZ,RSGZX,ATSX,ATSY, &
                ATSZ, ATPOT, ATFX, ATFY, &
                ATFZ,  ATGXX, ATGYY, &
                ATGZZ, &
                ATGXY, ATGYZ, ATGZX, &
                RESIZE   )
           CUMAXNB=nnnb
           CUMAXmb= bimag%NIMNB
           CALL RENBND(BNBND,CUMAXNB,NATOM,NGRP,MXJNBG)
           if (doimages) CALL REIMAG(BIMAG,CUMAXMB,MXJMBG)
        endif
        !-----------------------------------------------------------------------
        !     Generate the list by cubes.
        !-----------------------------------------------------------------------
        maxjnb=CUMAXNB
        maxjmb=CUMAXMB
        IF(PRNLEV > 6) WRITE(OUTU,125) 'NBNBGCM'
        CALL NBNDGCM(NATIMcu,NIMGRP, &
             NNNB, bnbnd%JNB, MAXJNB, &
             bnbnd%INBLO, X, Y, Z, &
             bimag%NIMNB,bimag%IMJNB, MAXJMB, &
             bimag%IMBLO, &
             bimag%IMINB,bimag%IMIBLO, &
             bimag%NIMNBS, bimag%IMJNBS, bimag%IMBLOS, &
             NNNBG,  MXJMBG, &
             bnbnd%JNBG, &
             bnbnd%INBLOG, bnbnd%INB14, &
             bnbnd%IBLO14, &
             bimag%IMING, bimag%IMIGLO, &
             liminv,iminv,ntrans,bimag%IMATPT, &
             bimag%IMATTR, &
             CUTNB, CTEXNB, LGROUP, &
             QEXTND, QXQUAD, QXGRAD, LVATOM, WRNMIN, CMPLTD, EPS, &
#if KEY_TSM==1
             QTSM,REACLS,PRODLS, &
#endif 
             RSCMX,RSCMY,RSCMZ, &
             RSQ, &
             RSDX,RSDY,RSDZ,RSQXX, &
             RSQYY,RSQZZ,RSQXY,RSQYZ, &
             RSQZX, &
             RSXMAX,RSYMAX,RSZMAX,RSDISP, &
             RSPOT,RSFX,RSFY, &
             RSFZ,RSGXX,RSGYY,RSGZZ, &
             RSGXY,RSGYZ,RSGZX,ATSX,ATSY, &
             ATSZ, ATPOT,ATFX, ATFY,ATFZ, ATGXX, ATGYY, &
             ATGZZ, ATGXY, ATGYZ, ATGZX, &
             RESIZE   )
        if(.not. cmpltd)then
           resize=.true.
           goto 215
        endif
     ELSE IF (LBYCU &
#if KEY_DOMDEC==1
          .and. .not.q_domdec &  
#endif
          ) THEN
#else /**/
     IF (LBYCU &
#if KEY_DOMDEC==1
          .and. .not.q_domdec &  
#endif
          ) THEN
#endif 
#if KEY_NO_BYCU==0
        IF(PRNLEV > 6) WRITE(OUTU,125) 'NBNBGC'
        CALL NBNDGC(NNNB, bnbnd%JNB, MAXJNB, &
             bnbnd%INBLO, X, Y, Z, NNNBG, MXJNBG, bnbnd%JNBG, &
             bnbnd%INBLOG, bnbnd%INB14, bnbnd%IBLO14, &
             bnbnd%ING14, bnbnd%IGLO14, CUTNB, CTEXNB, LGROUP, &
             QEXTND, QXQUAD, QXGRAD, LVATOM, WRNMIN, CMPLTD, EPS, &
#if KEY_TSM==1
             QTSM,REACLS,PRODLS, &
#endif 
             RSCMX, RSCMY, RSCMZ, &
             RSQ, &
             RSDX,  RSDY,  RSDZ,  RSQXX, &
             RSQYY, RSQZZ, RSQXY, RSQYZ, &
             RSQZX, &
             RSXMAX,RSYMAX,RSZMAX,RSDISP, &
             RSPOT, RSFX,  RSFY, &
             RSFZ,  RSGXX, RSGYY, RSGZZ, &
             RSGXY, RSGYZ, RSGZX, ATSX, ATSY, &
             ATSZ, ATPOT, ATFX, ATFY, ATFZ, ATGXX, ATGYY, ATGZZ, &
             ATGXY, ATGYZ, ATGZX)
        !-----------------------------------------------------------------------
        ! Added below--RJP
        !-----------------------------------------------------------------------
#endif 
#if KEY_NO_BYCC==0
     ELSE IF (LBYCC.AND.LGROUP &
#if KEY_DOMDEC==1
          .and. .not.q_domdec &  
#endif
          ) THEN
        ! Generate the group list by clusters-in-cubes.
        IF(PRNLEV > 6) WRITE(OUTU,125) 'NBNBGG'
        CALL NBNDGG(X, Y, Z, NNNBG, MXJNBG, bnbnd%JNBG, &
             bnbnd%INBLOG, bnbnd%ING14, bnbnd%IGLO14, &
             CUTNB, CTEXNB, &
             QEXTND, QXQUAD, QXGRAD, WRNMIN, CMPLTD, EPS, MXNCUB, &
             LSTN0,XOO,YOO,ZOO,ORDCBN, &
             NUMMP,HIMPRTN,MMM,WRKAR, &
             LSTM0,CNTN,XO,YO,ZO, &
             HPGEXL,HPGEXP,HPNEXG,    & !added 5.25.02 RJP
             RSCMX, RSCMY, RSCMZ, &
             RSQ, &
             RSDX,  RSDY,  RSDZ,  RSQXX, &
             RSQYY, RSQZZ, RSQXY, RSQYZ, &
             RSQZX, &
             RSXMAX,RSYMAX,RSZMAX,RSDISP, &
             RSPOT, RSFX,  RSFY, &
             RSFZ,  RSGXX, RSGYY, RSGZZ, &
             RSGXY, RSGYZ, RSGZX, ATSX, ATSY, &
             ATSZ, ATPOT, ATFX, ATFY, &
             ATFZ,  ATGXX, ATGYY, ATGZZ, &
             ATGXY, ATGYZ, ATGZX)
        call chmdealloc('nbonds.src','NBONDS','WRKAR',MXNCUB*27,intg=WRKAR)
           !
     ELSE IF (LBYCC &
#if KEY_DOMDEC==1
          .and. .not.q_domdec &  
#endif
          ) THEN
        IF(LBYCCF) THEN  !fastest version
           IF(PRNLEV > 6) WRITE(OUTU,125) 'NBNDCCF'
           ! Generate the atom list by clusters-in-cubes (fastest version).
           CALL NBNDCCF(NNNB, bnbnd%JNB, MAXJNB, &
                bnbnd%INBLO, X, Y, Z, &
                bnbnd%INB14, bnbnd%IBLO14, &
                CUTNB,HPMXCN,HPMXCD,TEMPN2,TEMPN, &
                CMPLTD,PNTNER,PNTDIS,PARTNN, &
                PARTND,ORDCBN,XOO,YOO,ZOO, &
                DUMMYX,HAFMAR,MEANX,MEANY,MEANZ, &
                LSTN0,NUMMP, &
                HIMPRTN,WRKAR,MMM,LSTM0, &
                CNTN,XO,YO,ZO,MXNCUB, &
                HPAEXL,HPAEXP,HPNEXA, &
                ATSX,ATSY,ATSZ)
           call chmdealloc('nbonds.src','NBONDS','WRKAR',MXNCUB*27,intg=WRKAR)
           call chmdealloc('nbonds.src','NBONDS','HPMXCD',MAXACD,intg=HPMXCD)
           call chmdealloc('nbonds.src','NBONDS','HPMXCN',MAXACN,intg=HPMXCN)
        ELSE  !options necessary
           ! Generate the atom list by clusters-in-cubes (options version).
           CALL NBNDCCO(NNNB, bnbnd%JNB, MAXJNB, &
                bnbnd%INBLO, X, Y, Z, &
                bnbnd%INB14, bnbnd%IBLO14, &
                CUTNB,HPMXCN,HPMXCD,TEMPN2,TEMPN, &
                CMPLTD,PNTNER,PNTDIS,PARTNN, &
                PARTND,ORDCBN,XOO,YOO, &
                ZOO,DUMMYX,HAFMAR,MEANX, &
                MEANY,MEANZ,LSTN0,NUMMP, &
                HIMPRTN,WRKAR,MMM,LSTM0, &
                CNTN,XO,YO,ZO, &
#if KEY_TSM==1
                QTSM,REACLS,PRODLS, &
#endif 
                MXNCUB,HPAEXL,HPAEXP,HPNEXA, &
                ATSX,ATSY,ATSZ)
           !
           call chmdealloc('nbonds.src','NBONDS','WRKAR',MXNCUB*27,intg=WRKAR)
           call chmdealloc('nbonds.src','NBONDS','HPMXCD',MAXACD,intg=HPMXCD)
           call chmdealloc('nbonds.src','NBONDS','HPMXCN',MAXACN,intg=HPMXCN)
        ENDIF
        !-----------------------------------------------------------------------
#endif 
#if KEY_DOMDEC==1
     elseif (q_domdec) then
        call ns_xfast(nnnb, nblist, x, y, z, &
             bnbnd%inb14, bnbnd%iblo14, cmpltd, atsx, atsy, atsz)
#endif 
     ELSE IF (LSLOWNB .AND. LGROUP) THEN
        ! Generate group lists
        IF(PRNLEV > 6) WRITE(OUTU,125) 'NBONDG'
        ! these should be separated, one by one, but just for fun we do them together"
        if (associated(bnbnd%jnbg)) then
           tjnbg => bnbnd%JNBG
           tinblog => bnbnd%INBLOG
           ting14 => bnbnd%ING14
           tiglo14 => bnbnd%IGLO14
           tpjnbg => bnbndp%JNBG
           tpinblog => bnbndp%INBLOG
           trjnbg => bnbndr%JNBG
           trinblog => bnbndr%INBLOG
           tping14 => bnbndr%ING14
           tpiglo14 => bnbndr%IGLO14
        else
           tjnbg => pzer1
           tinblog => pzer1
           ting14 => pzer1
           tiglo14 => pzer1
           tpjnbg => pzer1
           tpinblog => pzer1
           trjnbg => pzer1
           trinblog => pzer1
           tping14 => pzer1
           tpiglo14 => pzer1
        endif

#if KEY_MTS == 1
           if (.not. qtbmts) then
              bnbnm1%jnbg => mts_dummy
              bnbnm1%inblog => mts_dummy

              bnbnm2%jnbg => mts_dummy
              bnbnm2%inblog => mts_dummy
           end if
#endif /* KEY_MTS */

        CALL NBONDG(X,Y,Z, NNNBG, MXJNBG, tJNBG, &
             tINBLOG, tING14, tIGLO14, &
             CUTNB, CTEXNB, QEXTND, QXQUAD, QXGRAD, CMPLTD, EPS, &
#if KEY_MTS==1
             NNMT1G, MAXJM1G, bnbnm1%JNBG, bnbnm1%INBLOG, &
             NNMT2G, MAXJM2G, bnbnm2%JNBG, bnbnm2%INBLOG, &
#endif 
#if KEY_PERT==1
             NNNBGP, MXJNGP, tpJNBG, tpINBLOG, &
             NNNBGR, MXJNGR, trJNBG, trINBLOG, &
             PERTIG,tpING14, tpIGLO14, &
#endif /*  IF PERT*/
             RSCMX,RSCMY,RSCMZ, &
             RSQ,RSDX, &
             RSDY,RSDZ,RSQXX,RSQYY, RSQZZ, &
             RSQXY,RSQYZ,RSQZX, &
             RSXMAX,RSYMAX,RSZMAX, &
             RSPOT, &
             RSFX,RSFY,RSFZ,RSGXX,RSGYY, &
             RSGZZ,RSGXY,RSGYZ,RSGZX, &
#if KEY_FOURD==1
             RSFMAX,RSCMF, &
#endif 
             ATSX,ATSY,ATSZ, ATPOT, ATFX , ATFY, &
             ATFZ,  ATGXX, ATGYY, ATGZZ, &
             ATGXY, ATGYZ, ATGZX)

#if KEY_MTS == 1
           if (.not. qtbmts) then
              bnbnm1%jnbg => null()
              bnbnm1%inblog => null()

              bnbnm2%jnbg => null()
              bnbnm2%inblog => null()
           end if
#endif /* KEY_MTS */

        !-----------------------------------------------------------------------
     ELSE IF (LSLOWNB) THEN
        ! Generate atom lists
        IF(PRNLEV > 6) WRITE(OUTU,125) 'NBONDA'
#if KEY_FACTS==1 /*facts_block_1*/
        facts2: IF (FCTRUN) THEN
           call FCTNBA(nnnb, BNBND%jnb, maxjnb, BNBND%inblo, &
                       x, y, z,                              &
                       BNBND%inb14, BNBND%iblo14,            &
                       cutnb, wrnmin, cmpltd,                &
                       fct1ac, fct2ac,                       &
                       FCTBND%fct1ilo, FCTBND%fct1jnb,       &
                       FCTBND%fct2ilo, FCTBND%fct2jnb,       &
                       rscmx , rscmy , rscmz,                &
                       rsxmax, rsymax, rszmax, rsdisp,       &
                       atsx  , atsy  , atsz )
        ELSE facts2
#endif /* (facts_block_1)*/
           ! these should be separated, one by one, but just for fun we do them together"
           if (associated(bnbndp%jnb)) then
              tpjnb => bnbndp%jnb
              tpinblo => bnbndp%inblo
              trjnb => bnbndr%jnb
              trinblo => bnbndr%inblo
              tpinb14 => bnbndr%inb14
              tpiblo14 => bnbndr%iblo14
           else
              tpjnb => pzer1
              tpinblo => pzer1
              trjnb => pzer1
              trinblo => pzer1
              tpinb14 => pzer1
              tpiblo14 => pzer1
           endif
!@!           write(*,*)'NBONDS>associated: pjnb, pinblo, rjnb, rinblo, pinb14, piblo14 ?', &
!@!                associated(bnbndp%jnb),associated(bnbndp%inblo),&
!@!                associated(bnbndr%jnb),associated(bnbndr%inblo),&
!@!                associated(bnbndr%inb14),associated(bnbndr%iblo14)

#if KEY_MTS == 1
           if (.not. qtbmts) then
              bnbnm1%jnb => mts_dummy
              bnbnm1%inblo => mts_dummy

              bnbnm2%jnb => mts_dummy
              bnbnm2%inblo => mts_dummy
           end if
#endif /* KEY_MTS */

           CALL NBONDA(NNNB, bnbnd%JNB, MAXJNB, &
                bnbnd%INBLO, X, Y, Z, bnbnd%INB14, &
                bnbnd%IBLO14, CUTNB, WRNMIN, CMPLTD, &
#if KEY_MTS==1
                NNMT1, MAXJM1, bnbnm1%JNB, bnbnm1%INBLO, &
                NNMT2, MAXJM2, bnbnm2%JNB, bnbnm2%INBLO, &
#endif 
#if KEY_PERT==1
                NNNBP, MAXJNP, tpJNB, tpINBLO, &
                NNNBR, MAXJNR, trJNB, trINBLO, &
                PERTIP,tpINB14, tpIBLO14, &
#endif /*  IF PERT*/
#if KEY_TSM==1
                QTSM,REACLS,PRODLS, &
#endif 
                RSCMX,RSCMY,RSCMZ, &
                RSXMAX,RSYMAX,RSZMAX,RSDISP, &
#if KEY_FOURD==1
                RSFMAX,RSCMF, &
#endif 
                ATSX,ATSY,ATSZ, &
                MAXNBTHOLE, NBTHOL, NBTHOLIJ, NBTHOLP, &
                NBTHOL1, NBTHOL2, NBTHOL3, PPNBTHOLP, & 
                PPNBTHOL1, PPNBTHOL2, PPNBTHOL3, THOLCUT)

#if KEY_MTS == 1
           if (.not. qtbmts) then
              bnbnm1%jnb => null()
              bnbnm1%inblo => null()

              bnbnm2%jnb => null()
              bnbnm2%inblo => null()
           end if
#endif /* KEY_MTS */

           ! ENDIF
#if KEY_FACTS==1
        endif facts2  !  (fctrun)
#endif 
        !-----------------------------------------------------------------------
     ELSE
        CALL WRNDIE(-3,'<NBONDS>','Not valid FASTNB option.')
     ENDIF
     !-----------------------------------------------------------------------
     notcomplete: IF (.NOT.CMPLTD) THEN
        ! oops - not enough memory was allocated.
        !   get more and try again....
        IF (MAXJNB > BIGGST.OR.MXJNBG.GT.BIGGR) CALL WRNDIE(-4, &
             '<NBONDS>','GENERATING TOO MANY CONTACTS')
#if KEY_NO_BYCC==0
        IF (.NOT.LBYCC) THEN
#endif 
           IF(NNNB > MAXJNB) MAXJNB=(MAXJNB*1.5+20)
           IF(NNNBG > MXJNBG) MXJNBG=(MXJNBG*1.5+20)
#if KEY_IMCUBES==1
#if KEY_IMCUBES==1
           IF (LBYCBIM) THEN 
#endif
              IF (MAXJMB > BIGGST.OR.MXJMBG.GT.BIGGR) CALL WRNDIE(-4, &
                   '<NBONDS>','GENERATING TOO MANY IM CONTACTS')
              IF(bimag%NIMNB > MAXJMB) MAXJMB=(MAXJMB*1.5+20)
           ENDIF         ! sb050718.fix
#endif 
#if KEY_NO_BYCC==0
        ELSE             !if BYCC
           IF (.NOT. LGROUP) THEN
              IF (NNNB  >  MAXJNB) CCMAXJNB = INT(MAXJNB*1.5)
              IF (TEMPN2 > MAXACN) MAXACN = INT(MAXACN*1.5)
              IF (TEMPN > MAXACD) MAXACD = INT(MAXACD*1.5)
              IF ((PRNLEV > 2).AND.(.NOT.(DYNAMQ.OR.MINXYZ))) THEN
                 WRITE(OUTU,*) '(RE)ALLOCATING SPACE FOR ',MAXACN, &
                      ' NEAR ACTIVE CLUSTER PAIRS'
                 WRITE(OUTU,*) 'AND FOR ',MAXACD, &
                      ' DISTANT ACTIVE CLUSTER PAIRS'
              ENDIF
           ELSE          !if group-list
              IF (NNNBG  >  MXJNBG) CCMXJNBG = INT(MXJNBG*1.5)
           ENDIF
        ENDIF
#endif 
        ! ****************************************************************--RJP
#if KEY_MTS==1
        IF(TBHY1.OR.SLFG) THEN
           IF(NNMT1 > MAXJM1) MAXJM1=(MAXJM1*1.5+20)
           IF(NNMT2 > MAXJM2) MAXJM2=(MAXJM2*1.5+20)
           IF(NNMT1G > MAXJM1G) MAXJM1G=(MAXJM1G*1.5+20)
           IF(NNMT2G > MAXJM2G) MAXJM2G=(MAXJM2G*1.5+20)
        ENDIF
#endif 
#if KEY_PERT==1
        IF (QPERT) THEN
           IF (MAXJNP > BIGGST.OR.MXJNGP.GT.BIGGR) CALL WRNDIE(-4, &
                '<NBONDS>','GENERATING TOO MANY CONTACTS (P)')
           IF(NNNBP > MAXJNP) THEN
              MAXJNP=(MAXJNP*1.5+20)
              MAXJNR=(MAXJNR*1.5+20)
           ENDIF
           IF(NNNBGP > MXJNGP) THEN
              MXJNGP=(MXJNGP*1.5+20)
              MXJNGR=(MXJNGR*1.5+20)
           ENDIF
           !
           IF (MAXJNR > BIGGST.OR.MXJNGR.GT.BIGGR) CALL WRNDIE(-4, &
                '<NBONDS>','GENERATING TOO MANY CONTACTS (R)')
           IF(NNNBR > MAXJNR) MAXJNR=(MAXJNR*1.5+20)
           IF(NNNBGR > MXJNGR) MXJNGR=(MXJNGR*1.5+20)
        ENDIF
#endif 
#if KEY_FACTS==1
        if (fctrun) then
           if ((fct1ac > mxfcab).or.(fct2ac > mxfcab)) then
              mxfcab =(mxfcab*1.5+100)
              call FCTGROW_ints('nbonds.src', 'NBONDS', 'fct1jnb', FCTBND%fct1jnb, mxfcab)
              call FCTGROW_ints('nbonds.src', 'NBONDS', 'fct2jnb', FCTBND%fct2jnb, mxfcab)
           endif
        endif
#endif 
     ENDIF notcomplete
  ENDDO complete
  !
  !=======================================================================
  !  Save counters in various data structures for MTS and PERT
#if KEY_MTS==1
  IF (QTBMTS) THEN
     NNMT5=NNNB
     NNMT6=NNNBG
     IF(TBHY1.OR.SLFG) THEN
        CALL SETBND(BNBND)
        CALL GETBND(BNBNM1,.FALSE.)
        NNNB=NNMT1
        NNNBG=NNMT1G
        CALL SETBND(BNBNM1)
        CALL GETBND(BNBNM2,.FALSE.)
        NNNB=NNMT2
        NNNBG=NNMT2G
        CALL SETBND(BNBNM2)
        CALL GETBND(BNBND,.TRUE.)
     ENDIF
     NNNB = NNMT5
     NNNBG=NNMT6
  ENDIF
#endif 
  !
#if KEY_PERT==1
  IF(QPERT) THEN
     ! put the nonbond list counts on the appropriate data structure.
     CALL SETBND(BNBND)
     CALL GETBND(BNBNDP,.FALSE.)
     NNNB=NNNBP
     NNNBG=NNNBGP
     CALL SETBND(BNBNDP)
     CALL GETBND(BNBNDR,.FALSE.)
     NNNB=NNNBR
     NNNBG=NNNBGR
     CALL SETBND(BNBNDR)
     CALL GETBND(BNBND,.TRUE.)
  ENDIF
#endif 
  !=======================================================================
  ! Process (split) the nonbond list based on QM atom selections.
  ! This has been moved last part to support IMAGEs
  ! namkh 09/24/04
  !...##IF QUANTUM
  !      IF (NATQM > 0) THEN
  !        IF(PRNLEV > 6) WRITE(OUTU,125) 'NBNDQM'
  !        CALL NBNDQM
  !      ENDIF
  !...##ENDIF
  !=======================================================================
  !
  ! Do image nonbond list update.
  !
#if KEY_DOMDEC==1
  if(.not.q_domdec) then        
#endif
#if KEY_PBOUND==1
  IF(.not.qBoun) then           
#endif
#if KEY_IMCUBES==1
  IF(.not.LBYCBIM) then         
#endif
        IF(NTRNSX > 0) THEN
           !
           !  Create image coordinates
           !
           CALL TRANSO(X, Y, Z, &
                (/ zero /), (/ zero /), (/ zero /), &
                .FALSE., .FALSE., &
                (/ zero /), &
                NATOM,NTRANS,IMTRNS, &
                bimag%IMATPT,bimag%IMATTR,NOROT,NATIM &
#if KEY_FLUCQ==1
                ,QFLUC,CG,FQCFOR           & 
#endif
                )
           !
           IF(CUTIM < CUTNB) CALL WRNDIE(0, &
                '<NBONDS>','CUTNB is larger than CUTIM: List incomplete.')
           !
           !=======================================================================
           ! should we use the vector/parallel routines for image nbond generation?
           LSLOWNB = (FASTER < 0.OR.(.NOT.QFASTNB))
           IF(LEXS.OR.LGROUP) LSLOWNB = .TRUE.
#if KEY_PARALLEL==1
           ! fast routines do not support parallel
           IF(NUMNOD >= 2) LSLOWNB=.TRUE.
#endif 
#if KEY_PERT==1
           ! fast routines do not support PERT
           IF(QPERT) LSLOWNB=.TRUE.
#endif 
#if KEY_MTS==1
           ! fast routines do not support MTS
           IF(TBHY1.OR.SLFG) LSLOWNB=.TRUE.
#endif 
#if KEY_NOST2==0
           IF(NST2 > 0) LSLOWNB = .TRUE.
#endif 
#if KEY_TSM==1 /*tsmtest*/
           ! Some fast routines do not support TSM
           IF (QTSM) LSLOWNB = .TRUE.
#endif /* (tsmtest)*/
#if KEY_FOURD==1 /*4dadd*/
           IF (DIM4) LSLOWNB=.TRUE.
#endif /* (4dadd)*/
           !=======================================================================
           RVAL=NATIM-NATOM
           RVAL=(RVAL+20)/NATOM
           IF(CUTIM > CUTNB) THEN
              RFACT=CUTNB/CUTIM
              IF(RFACT < HALF) RFACT=HALF
              RVAL=RVAL*RFACT**2
           ENDIF
           IF(.NOT.LGROUP) THEN
              !  imscal added by Scott Feller, NIH, Aug 95
              IVAL=(NNNB+100)*RVAL/2
              IF(MAXJMB < IVAL) MAXJMB=IVAL*2*IMSCAL
              ! do not allow allocation of size zero array
              ! always reserve space for at least 100 interactions
              maxjmb = max(100, maxjmb)
              MXJMBG=0
           ENDIF
           IF(LGROUP) MAXJMB=0
           IF(LGROUP.OR.NST2 > 0) THEN
              IVAL=(NNNBG+100)*RVAL
              IF(IVAL > MXJMBG) MXJMBG=IVAL
           ENDIF
#if KEY_MTS==1
           IF(TBHY1.OR.SLFG) THEN
              MXJMT1=MAXJMB
              MXJMT2=MAXJMB
              MXJMT1G=MXJMBG
              MXJMT2G=MXJMBG
           ENDIF
#endif 
           !
#if KEY_PERT==1
           IF(QPERT) THEN
              LSLOWNB=.TRUE.
              IF(LGROUP) THEN
                 IVAL=(NNNBGP+100)*RVAL/2
                 IF(MXJMGP*2 < IVAL) MXJMGP=IVAL
                 IVAL=(NNNBGR+100)*RVAL/2
                 IF(MXJMGR*2 < IVAL) MXJMGR=IVAL
                 MXJMBP=0
                 MXJMBR=0
              ELSE
                 IVAL=(NNNBP+100)*RVAL/2
                 IF(MXJMBP*2 < IVAL) MXJMBP=IVAL
                 IVAL=(NNNBR+100)*RVAL/2
                 IF(MXJMBR*2 < IVAL) MXJMBR=IVAL
                 MXJMGP=0
                 MXJMGR=0
              ENDIF
           ENDIF
#endif 
           !
#if KEY_TSM==1
           !     If tsm is being used, the reactant and product lists must be
           !     expanded.
           IF (QTSM) THEN
              call chmalloc('nbonds.src','NBONDS','RIMGLS',NATIM,intg=RIMGLS)
              CALL EXTSL2(NATOM,REACLS,NATIM, &
                   RIMGLS,bimag%IMATTR)
              call chmalloc('nbonds.src','NBONDS','PIMGLS',NATIM,intg=PIMGLS)
              CALL EXTSL2(NATOM,PRODLS,NATIM, &
                   PIMGLS,bimag%IMATTR)
!           ELSE
!              RIMGLS=1
!              PIMGLS=2
           ENDIF
#endif 
           !=======================================================================
           ! Now set up image atoms and image-primary nonbond list
           !
           CMPLTD=.FALSE.
           DO WHILE (.NOT.CMPLTD)
              !
              CALL REIMAG(BIMAG,MAXJMB,MXJMBG)
              IF(PRNLEV >= 5) WRITE(OUTU,410) MAXJMB,MXJMBG
410           FORMAT (' SPACE FOR',I9,' ATOM PAIRS AND',I9,' GROUP PAIRS')
#if KEY_MTS==1
              !
              IF(TBHY1.OR.SLFG) THEN
                 CALL REIMAG(BIMTS1,MXJMT1,MXJMT1G)
                 CALL REIMAG(BIMTS2,MXJMT2,MXJMT2G)
                 IF(PRNLEV >= 5) WRITE(OUTU,1401) MXJMT1,MXJMT1G
                 IF(PRNLEV >= 5) WRITE(OUTU,1410) MXJMT2,MXJMT2G
              ENDIF
1401          FORMAT (' SPACE FOR',I9,' ATOM PAIRS AND',I9,' GROUP PAIRS')
1410          FORMAT (' SPACE FOR',I9,' ATOM PAIRS AND',I9,' GROUP PAIRS')
#endif 
              !
#if KEY_PERT==1
              IF(QPERT) THEN
                 CALL REIMAG(BIMAGP,MXJMBP,MXJMGP)
                 CALL REIMAG(BIMAGR,MXJMBR,MXJMGR)
                 IF(PRNLEV >= 5) WRITE(OUTU,412) MXJMBR,MXJMGR
412              FORMAT (' SPACE FOR',I9,' REACTANT ATOM PAIRS, AND', &
                      I9,' GROUP PAIRS.')
                 IF(PRNLEV >= 5) WRITE(OUTU,411) MXJMBP,MXJMGP
411              FORMAT (' SPACE FOR',I9,' PRODUCT  ATOM PAIRS, AND', &
                      I9,' GROUP PAIRS.')
              ENDIF
#endif 
#if KEY_FACTS==1
              if (fctrun .AND. prnlev >= 5) then
                 write(outu,228) mxfcib
228              FORMAT (' ==  FACTS  == SPACE FOR',I9, &
                         ' FACTS IMAGE ATOM PAIRS ')
              endif
#endif 
              !
              !-----------------------------------------------------------------------
#if KEY_DEBUG==1
              WRITE(OUTU,'(A,I20)') 'BIMAG%IMINB ',  BIMAG%IMINB
              WRITE(OUTU,'(A,I20)') 'BIMAG%IMIBLO ',  BIMAG%IMIBLO
              WRITE(OUTU,'(A,I20)') 'BIMAG%IMING ',  BIMAG%IMING
              WRITE(OUTU,'(A,I20)') 'BIMAG%IMIGLO ',  BIMAG%IMIGLO
              WRITE(OUTU,'(A,I20)') 'BIMAG%NIMNB ',  BIMAG%NIMNB
              WRITE(OUTU,'(A,I20)') 'BIMAG%IMJNB ',  BIMAG%IMJNB
              WRITE(OUTU,'(A,I20)') 'BIMAG%IMBLO ',  BIMAG%IMBLO
              WRITE(OUTU,'(A,I20)') 'BIMAG%NIMNBG ',  BIMAG%NIMNBG
              WRITE(OUTU,'(A,I20)') 'BIMAG%IMJNBG ',  BIMAG%IMJNBG
              WRITE(OUTU,'(A,I20)') 'BIMAG%IMBLOG ',  BIMAG%IMBLOG
              WRITE(OUTU,'(A,I20)') 'BIMAG%NIMNBS ',  BIMAG%NIMNBS
              WRITE(OUTU,'(A,I20)') 'BIMAG%IMJNBS ',  BIMAG%IMJNBS
              WRITE(OUTU,'(A,I20)') 'BIMAG%IMBLOS ',  BIMAG%IMBLOS
              WRITE(OUTU,'(A,I20)') 'BIMAG%NIMNBX ',  BIMAG%NIMNBX
              WRITE(OUTU,'(A,I20)') 'BIMAG%IMJNBX ',  BIMAG%IMJNBX
              WRITE(OUTU,'(A,I20)') 'BIMAG%IMBLOX ',  BIMAG%IMBLOX
              WRITE(OUTU,'(A,I20)') 'BIMAG%IMATTR ',  BIMAG%IMATTR
              WRITE(OUTU,'(A,I20)') 'BIMAG%IMATPT ',  BIMAG%IMATPT
              WRITE(OUTU,'(A,I20)') 'BIMAG%IMINB ',  BIMAG%IMINB
              WRITE(OUTU,'(A,I20)') 'BIMAG%IMIBLO ',  BIMAG%IMIBLO
              WRITE(OUTU,'(A,I20)') 'BIMAG%IMING ',  BIMAG%IMING
              WRITE(OUTU,'(A,I20)') 'BIMAG%IMIGLO ',  BIMAG%IMIGLO
              WRITE(OUTU,'(A,I20)') 'BIMAGP%NIMNB ',  BIMAGP%NIMNB
              WRITE(OUTU,'(A,I20)') 'BIMAGP%IMJNB ',  BIMAGP%IMJNB
              WRITE(OUTU,'(A,I20)') 'BIMAGP%IMBLO ',  BIMAGP%IMBLO
              WRITE(OUTU,'(A,I20)') 'BIMAGP%NIMNBG ',  BIMAGP%NIMNBG
              WRITE(OUTU,'(A,I20)') 'BIMAGP%IMJNBG ',  BIMAGP%IMJNBG
              WRITE(OUTU,'(A,I20)') 'BIMAGP%IMBLOG ',  BIMAGP%IMBLOG
              WRITE(OUTU,'(A,I20)') 'BIMAGP%NIMNBS ',  BIMAGP%NIMNBS
              WRITE(OUTU,'(A,I20)') 'BIMAGP%IMJNBS ',  BIMAGP%IMJNBS
              WRITE(OUTU,'(A,I20)') 'BIMAGP%IMBLOS ',  BIMAGP%IMBLOS
              WRITE(OUTU,'(A,I20)') 'BIMAGP%NIMNBX ',  BIMAGP%NIMNBX
              WRITE(OUTU,'(A,I20)') 'BIMAGP%IMJNBX ',  BIMAGP%IMJNBX
              WRITE(OUTU,'(A,I20)') 'BIMAGP%IMBLOX ',  BIMAGP%IMBLOX
              WRITE(OUTU,'(A,I20)') 'BIMAGR%NIMNB ',  BIMAGR%NIMNB
              WRITE(OUTU,'(A,I20)') 'BIMAGR%IMJNB ',  BIMAGR%IMJNB
              WRITE(OUTU,'(A,I20)') 'BIMAGR%IMBLO ',  BIMAGR%IMBLO
              WRITE(OUTU,'(A,I20)') 'BIMAGR%NIMNBG ',  BIMAGR%NIMNBG
              WRITE(OUTU,'(A,I20)') 'BIMAGR%IMJNBG ',  BIMAGR%IMJNBG
              WRITE(OUTU,'(A,I20)') 'BIMAGR%IMBLOG ',  BIMAGR%IMBLOG
              WRITE(OUTU,'(A,I20)') 'BIMAGR%NIMNBS ',  BIMAGR%NIMNBS
              WRITE(OUTU,'(A,I20)') 'BIMAGR%IMJNBS ',  BIMAGR%IMJNBS
              WRITE(OUTU,'(A,I20)') 'BIMAGR%IMBLOS ',  BIMAGR%IMBLOS
              WRITE(OUTU,'(A,I20)') 'BIMAGR%NIMNBX ',  BIMAGR%NIMNBX
              WRITE(OUTU,'(A,I20)') 'BIMAGR%IMJNBX ',  BIMAGR%IMJNBX
              WRITE(OUTU,'(A,I20)') 'BIMAGR%IMBLOX ',  BIMAGR%IMBLOX
#endif 
              !-----------------------------------------------------------------------
              IF (LSLOWNB .AND. LGROUP) THEN
                 ! Use the generic group image nonbond list generation routine
                 !
                 IF(PRNLEV > 6) WRITE(OUTU,125) 'NBONDMG'
                 CALL NBONDMG(X, Y, Z, MXJMBG, &
                      bimag%IMING,  bimag%IMIGLO, &
                      bimag%NIMNBG, bimag%IMJNBG, bimag%IMBLOG, &
                      bimag%NIMNBX, bimag%IMJNBX, bimag%IMBLOX, &
                      NTRANS, NIMGRP, bimag%IMATTR, bimag%IMATPT, &
                      LIMINV, IMINV, CUTNB, CMPLTD, &
#if KEY_MTS==1
                      MXJMT1G, MXJMT2G, &
                      bimts1%NIMNBG, bimts1%IMJNBG, bimts1%IMBLOG, &
                      bimts2%NIMNBG, bimts2%IMJNBG, bimts2%IMBLOG, &
                      bimts1%NIMNBX, bimts1%IMJNBX, bimts1%IMBLOX, &
                      bimts2%NIMNBX, bimts2%IMJNBX, bimts2%IMBLOX, &
#endif 
#if KEY_PERT==1
                      PERTIG, &
                      bimag%IMING, bimag%IMIGLO, MXJMGP, MXJMGR, &
                      bimagp%NIMNBG, bimagp%IMJNBG, bimagp%IMBLOG, &
                      bimagp%NIMNBX, bimagp%IMJNBX, bimagp%IMBLOX, &
                      bimagr%NIMNBG, bimagr%IMJNBG, bimagr%IMBLOG, &
                      bimagr%NIMNBX, bimagr%IMJNBX, bimagr%IMBLOX, &
#endif /*  IF PERT*/
                      RSCMX,  RSCMY,  RSCMZ, &
                      RSXMAX,RSYMAX,RSZMAX)
                 !-----------------------------------------------------------------------
              ELSE IF (LSLOWNB) THEN
                 ! Use the generic atom image nonbond list generation routine
                 !
                 IF(PRNLEV > 6) WRITE(OUTU,125) 'NBONDMA'
#if KEY_FACTS==1 /*facts_block_2*/
                 IF (FCTRUN) THEN
                    call FCTNMA(x, y, z, maxjmb,                          &
                                BIMAG%iminb , BIMAG%imiblo,               &
                                BIMAG%nimnb , BIMAG%imjnb , BIMAG%imblo,  &
                                BIMAG%nimnbs, BIMAG%imjnbs, BIMAG%imblos, &
                                ntrans, nimgrp,                           &
                                BIMAG%imattr, BIMAG%imatpt,               &
                                liminv, iminv, cutnb, wrnmin, cmpltd,     &
                                fct3ac,fct4ac,                            &
                                FCTBND%fct3ilo, FCTBND%fct3jnb,           &
                                FCTBND%fct4ilo, FCTBND%fct4jnb,           &
                                rscmx,  rscmy,  rscmz,                    &
                                rsxmax, rsymax, rszmax, rsdisp )
                 ELSE
#endif /* (facts_block_2)*/
                    CALL NBONDMA(X, Y, Z, MAXJMB, bimag%IMINB, &
                         bimag%IMIBLO, &
                         bimag%NIMNB,  bimag%IMJNB,  bimag%IMBLO, &
                         bimag%NIMNBS, bimag%IMJNBS, bimag%IMBLOS, &
                         NTRANS, NIMGRP, bimag%IMATTR, bimag%IMATPT, &
                         LIMINV, IMINV, CUTNB, WRNMIN, CMPLTD, &
#if KEY_MTS==1
                         MXJMT1, MXJMT2, &
                         bimts1%NIMNB, bimts1%IMJNB, bimts1%IMBLO, &
                         bimts2%NIMNB, bimts2%IMJNB, bimts2%IMBLO, &
                         bimts1%NIMNBS, bimts1%IMJNBS, bimts1%IMBLOS, &
                         bimts2%NIMNBS, bimts2%IMJNBS, bimts2%IMBLOS, &
#endif 
#if KEY_PERT==1
                         PERTIP, &
                         bimag%IMINB,   bimag%IMIBLO,  MXJMBP, MXJMBR, &
                         bimagp%NIMNB,  bimagp%IMJNB,  bimagp%IMBLO, &
                         bimagp%NIMNBS, bimagp%IMJNBS, bimagp%IMBLOS, &
                         bimagr%NIMNB,  bimagr%IMJNB,  bimagr%IMBLO, &
                         bimagr%NIMNBS, bimagr%IMJNBS, bimagr%IMBLOS, &
#endif /*  IF PERT*/
#if KEY_TSM==1
                         QTSM,RIMGLS,PIMGLS, &      
#endif
                         RSCMX,  RSCMY,  RSCMZ, &
                         RSXMAX,RSYMAX,RSZMAX,RSDISP, &
      MAXNBTHOLE, NBTHOL, NBTHOLIJ, NBTHOLP, NBTHOLPIMG, &
      NBTHOL1, NBTHOL2, NBTHOL3, PPNBTHOLP, PPNBTHOL1, PPNBTHOL2, PPNBTHOL3, THOLCUT)

#if KEY_FACTS==1
                 ENDIF
#endif 
                 !-----------------------------------------------------------------------
              ELSE
                 CALL WRNDIE(-3,'<NBONDS>','Not valid FASTNB image opt.')
              ENDIF
              !
              IF(.NOT.CMPLTD) THEN
                 IF(bimag%NIMNB  > MAXJMB) MAXJMB=MAXJMB*1.5+20
                 IF(bimag%NIMNBG > MXJMBG) MXJMBG=MXJMBG*1.5+20
                 !
#if KEY_MTS==1
                 IF(TBHY1.OR.SLFG) THEN
                    IF(bimts1%NIMNB > MXJMT1) MXJMT1=MXJMT1*1.5+20
                    IF(bimts2%NIMNB > MXJMT2) MXJMT2=MXJMT2*1.5+20
                    IF(bimts1%NIMNBG > MXJMT1G) &
                         MXJMT1G=MXJMT1G*1.5+20
                    IF(bimts2%NIMNBG > MXJMT2G) &
                         MXJMT2G=MXJMT2G*1.5+20
                 ENDIF
                 !
#endif 
#if KEY_PERT==1
                 IF(QPERT) THEN
                    IF(bimagp%NIMNB  > MXJMBP) MXJMBP=MXJMBP*1.5+20
                    IF(bimagp%NIMNBG > MXJMGP) MXJMGP=MXJMGP*1.5+20
                    IF(bimagr%NIMNB  > MXJMBR) MXJMBR=MXJMBR*1.5+20
                    IF(bimagr%NIMNBG > MXJMGR) MXJMGR=MXJMGR*1.5+20
                 ENDIF
#endif /*  IF PERT*/
#if KEY_FACTS==1
                 if (fctrun) then
                    if ((fct3ac > mxfcib).or.(fct4ac > mxfcib)) then
                       mxfcib =(mxfcib*1.5+100)
                       call FCTGROW_ints('nbonds.src', 'NBONDS', 'fct3jnb', FCTBND%fct3jnb, mxfcib)
                       call FCTGROW_ints('nbonds.src', 'NBONDS', 'fct4jnb', FCTBND%fct4jnb, mxfcib)
                    endif
                 endif
#endif 
              ENDIF
           ENDDO
#if KEY_TSM==1
           IF (QTSM) THEN
              call chmdealloc('nbonds.src','NBONDS','RIMGLS',NATIM,intg=RIMGLS)
              call chmdealloc('nbonds.src','NBONDS','PIMGLS',NATIM,intg=PIMGLS)
           ENDIF
#endif 
        ENDIF ! (NTRNSX > 0)
#if KEY_IMCUBES==1
  ENDIF                        
#endif
#if KEY_PBOUND==1
  ENDIF                        
#endif
#if KEY_DOMDEC==1
  endif                        
#endif
  !
  !=======================================================================
  ! Process (split) the nonbond list based on QM atom selections.
  ! Moved here to support IMAGE facility.
  ! namkh 09/24/04
#if KEY_QUANTUM==1 || KEY_MNDO97==1 || KEY_SQUANTM==1 /*qntm_0*/
#if KEY_QUANTUM==1
  IF (NATQM > 0) THEN
#elif KEY_SQUANTM==1
  IF (NATQM(1) > 0) THEN
#elif KEY_MNDO97==1
  IF (NUMAT > 0) THEN
#endif 
     IF(PRNLEV > 6) WRITE(OUTU,125) 'NBNDQM'
     CALL NBNDQM(X,Y,Z)
  ENDIF
#endif /* (qntm_0)*/
  !
  !=======================================================================
  ! Free up all stack space.
#if KEY_NO_BYCC==0
  IF(LBYCC) THEN
     call chmdealloc('nbonds.src','NBONDS','NUMMP',MXNCUB,intg=NUMMP)
     call chmdealloc('nbonds.src','NBONDS','HIMPRTN',MXNCUB,intg=HIMPRTN)
     call chmdealloc('nbonds.src','NBONDS','MMM',MXNCUB,intg=MMM)
     call chmdealloc('nbonds.src','NBONDS','LSTM0',MXNCUB,intg=LSTM0)
     call chmdealloc('nbonds.src','NBONDS','CNTN',MXNCUB,intg=CNTN)
     call chmdealloc('nbonds.src','NBONDS','XO',MXNCUB,intg=XO)
     call chmdealloc('nbonds.src','NBONDS','YO',MXNCUB,intg=YO)
     call chmdealloc('nbonds.src','NBONDS','ZO',MXNCUB,intg=ZO)
     IF(.NOT.LGROUP) THEN
        call chmdealloc('nbonds.src','NBONDS','LSTN0',NACTC,intg=LSTN0)
        call chmdealloc('nbonds.src','NBONDS','XOO',NACTC,intg=XOO)
        call chmdealloc('nbonds.src','NBONDS','YOO',NACTC,intg=YOO)
        call chmdealloc('nbonds.src','NBONDS','ZOO',NACTC,intg=ZOO)
        call chmdealloc('nbonds.src','NBONDS','ORDCBN',NACTC,intg=ORDCBN)
        call chmdealloc('nbonds.src','NBONDS','PNTNER',NACTC,intg=PNTNER)
        call chmdealloc('nbonds.src','NBONDS','PNTDIS',NACTC,intg=PNTDIS)
        call chmdealloc('nbonds.src','NBONDS','PARTNN',NACTC,intg=PARTNN)
        call chmdealloc('nbonds.src','NBONDS','PARTND',NACTC,intg=PARTND)
        call chmdealloc('nbonds.src','NBONDS','DUMMYX',NACTC,intg=DUMMYX)
        call chmdealloc('nbonds.src','NBONDS','HAFMAR',NACTC,crl=HAFMAR)
        call chmdealloc('nbonds.src','NBONDS','MEANX',NACTC,crl=MEANX)
        call chmdealloc('nbonds.src','NBONDS','MEANY',NACTC,crl=MEANY)
        call chmdealloc('nbonds.src','NBONDS','MEANZ',NACTC,crl=MEANZ)
     ELSE
        call chmdealloc('nbonds.src','NBONDS','LSTN0',NGRP,intg=LSTN0)
        call chmdealloc('nbonds.src','NBONDS','XOO',NGRP,intg=XOO)
        call chmdealloc('nbonds.src','NBONDS','YOO',NGRP,intg=YOO)
        call chmdealloc('nbonds.src','NBONDS','ZOO',NGRP,intg=ZOO)
        call chmdealloc('nbonds.src','NBONDS','ORDCBN',NGRP,intg=ORDCBN)
     ENDIF
  ENDIF
#endif 
  nccngr_deallocate: IF((.NOT.LBYCC).OR.(LGROUP)) THEN
     NGRPT=MAX(NGRP,NIMGRP)
     call chmdealloc('nbonds.src','NBONDS','rscmx',NGRPt,crl=rscmx)
     call chmdealloc('nbonds.src','NBONDS','rscmy',NGRPt,crl=rscmy)
     call chmdealloc('nbonds.src','NBONDS','rscmz',NGRPt,crl=rscmz)
     IF(QEXTND) THEN
        call chmdealloc('nbonds.src','NBONDS','RSQ',NGRP,crl=RSQ)
        call chmdealloc('nbonds.src','NBONDS','RSDX',NGRP,crl=RSDX)
        call chmdealloc('nbonds.src','NBONDS','RSDY',NGRP,crl=RSDY)
        call chmdealloc('nbonds.src','NBONDS','RSDZ',NGRP,crl=RSDZ)
        call chmdealloc('nbonds.src','NBONDS','RSPOT',NGRP,crl=RSPOT)
        call chmdealloc('nbonds.src','NBONDS','RSFX',NGRP,crl=RSFX)
        call chmdealloc('nbonds.src','NBONDS','RSFY',NGRP,crl=RSFY)
        call chmdealloc('nbonds.src','NBONDS','RSFZ',NGRP,crl=RSFZ)
        call chmdealloc('nbonds.src','NBONDS','RSGXX',NGRP,crl=RSGXX)
        call chmdealloc('nbonds.src','NBONDS','RSGYY',NGRP,crl=RSGYY)
        call chmdealloc('nbonds.src','NBONDS','RSGZZ',NGRP,crl=RSGZZ)
        call chmdealloc('nbonds.src','NBONDS','RSGXY',NGRP,crl=RSGXY)
        call chmdealloc('nbonds.src','NBONDS','RSGYZ',NGRP,crl=RSGYZ)
        call chmdealloc('nbonds.src','NBONDS','RSGZX',NGRP,crl=RSGZX)

        IF (QXQUAD) THEN
           call chmdealloc('nbonds.src','NBONDS','RSQXX',NGRP,crl=RSQXX)
           call chmdealloc('nbonds.src','NBONDS','RSQYY',NGRP,crl=RSQYY)
           call chmdealloc('nbonds.src','NBONDS','RSQZZ',NGRP,crl=RSQZZ)
           call chmdealloc('nbonds.src','NBONDS','RSQXY',NGRP,crl=RSQXY)
           call chmdealloc('nbonds.src','NBONDS','RSQYZ',NGRP,crl=RSQYZ)
           call chmdealloc('nbonds.src','NBONDS','RSQZX',NGRP,crl=RSQZX)
        ENDIF
     ENDIF
     call chmdealloc('nbonds.src','NBONDS','rsxmax', NGRPt,crl=rsxmax)
     call chmdealloc('nbonds.src','NBONDS','rsymax', NGRPt,crl=rsymax)
     call chmdealloc('nbonds.src','NBONDS','rszmax', NGRPt,crl=rszmax)
     call chmdealloc('nbonds.src','NBONDS','rsdisp', NGRPt,intg=rsdisp)
#if KEY_FOURD==1
     IF (DIM4) THEN
        call chmdealloc('nbonds.src','NBONDS','rszmax', NGRPt,crl=rsfmax)
        call chmdealloc('nbonds.src','NBONDS','rszmax', NGRPt,crl=rscmf)
     ELSE
        call chmdealloc('nbonds.src','NBONDS','rszmax', 1,crl=rsfmax)
        call chmdealloc('nbonds.src','NBONDS','rszmax', 1,crl=rscmf)
     ENDIF
#endif 
  ENDIF nccngr_deallocate

  IF((.NOT.LBYCC).OR.(LGROUP)) THEN
     if(allocated(rszmax))then
        call chmdealloc('nbonds.src','NBONDS','rszmax', 1,crl=rsfmax)
        call chmdealloc('nbonds.src','NBONDS','rszmax', 1,crl=rscmf)
     endif
#if KEY_FOURD==1
     if(allocated(rszmax))then
        call chmdealloc('nbonds.src','NBONDS','rszmax', 1,crl=rsfmax)
        call chmdealloc('nbonds.src','NBONDS','rszmax', 1,crl=rscmf)
     endif
#endif 
  ENDIF
  
  !
#if KEY_NOMISC==0
  IF (QRXNFL .AND. RXNMOD == 'NONBOND') THEN
     IF (.NOT.QEXTND) THEN
        IF(WRNLEV >= 2) WRITE(OUTU,225)
225     FORMAT(' *****  ERROR  ***** ', &
             'ERXNFL CALLED IN NBONDS WITHOUT ', &
             'EXTENDED ELECTROSTATICS')
        CALL DIE
     ENDIF
     CALL ERXNFL(ERXN,NATOM,CG,X,Y,Z,adum,adum,adum,ATPOT, &
          ATFX,ATFY,ATFZ, &
          1,RXNORD,EPS,EPSEXT,RXNSHL,MPMMNT, &
          RXMMNT,ERXMNT, &
          LEGPN,LEGPND)
  ENDIF
#endif 
                        !
#if KEY_MTS==1
  IF(.NOT.SLFG) THEN
#endif 
     IF (WRNLEV > 5) CALL PRNBCT(BNBND)
     CALL SETBND(BNBND)
#if KEY_MTS==1
  ENDIF
#endif 
  !
  !# <caves>-Aug-6-1993 (Leo Caves)  capture number of pairs
  CALL set_param('NNBA',NNNB)
  CALL set_param('NNBG',NNNBG)
  CALL set_param('NNBI',XNNNB)
  !
#if KEY_REPLICA==1
  !# <caves>-Aug-6-1993 (Leo Caves) Capture number of exclusions due to replicas
  IF (qRep) THEN
     CALL set_param('NRXA',nRepXA)
     CALL set_param('NRXG',nRepXG)
  ENDIF ! qRep
#endif /*  REPLICA*/
  !
#if KEY_LOOKUP==1
  ! Extract the OO (,OH,HO, and HH) list(s) from the non-bond lists
  !
  ! No check has been made that atom lists are in use, or concerning
  ! effects on various other methods
  !
  lookup_0: IF(QVV.OR.QVU)THEN
     IF(PRNLEV > 6) WRITE(OUTU,125) 'WWSPLTNB'
     ! Space allocation
     IF(.NOT.ALLOCATED(IWOONBL))THEN
        !...##IF IMCUBES
        !            IF(LBYCBIM)THEN
        ! Need similar for PBUCBES
        !              ALLOCATE(IWOONBL(2*NATOM))
        !            ELSE
        !...##ENDIF
        ALLOCATE(IWOONBL(NATOM))
#if KEY_IMCUBES==1
        !            ENDIF                
#endif
     ENDIF
     IF(.NOT.ALLOCATED(JWOONBL))THEN
        WON=FLOAT(NWWO)/FLOAT(NATOM)
        MXJWWOO=NNNB*WON*WON+2
        ALLOCATE(JWOONBL(MXJWWOO))
     ENDIF
     IF(PRNLEV >= 5) WRITE(OUTU,219) ' PRIMARY ',MXJWWOO
219  FORMAT (A,'SPACE FOR',I12,' SOLVENT-SOLVENT OO PAIRS')

     CALL WWSPLTNB(CMPLTD,.FALSE.,1,NATOM, &
          bnbnd%INBLO,bnbnd%JNB, &
          MXJWWOO,IWOONBL,JWOONBL, &
#if KEY_IMCUBES==1
          LBYCBIM,                                  & 
#endif
          NNNB,NBOO,NBVU,NBUU)
     IF(.NOT. CMPLTD) THEN
        ! Resize to now exactly known sizes and redo the split
        ! Use a small margin to reduce number of deallocate/allocate events
        DEALLOCATE(JWOONBL)
        MXJWWOO=1.1*NBOO+2
        ALLOCATE(JWOONBL(MXJWWOO))
        IF(PRNLEV >= 5) WRITE(OUTU,219) ' RESIZE PRIMARY ',MXJWWOO
        CALL WWSPLTNB(CMPLTD,.FALSE.,1,NATOM, &
             bnbnd%INBLO,bnbnd%JNB, &
             MXJWWOO,IWOONBL,JWOONBL, &
#if KEY_IMCUBES==1
             LBYCBIM,                                  & 
#endif
             NNNB,NBOO,NBVU,NBUU)
        IF(.NOT. CMPLTD) CALL WRNDIE(-4,'<NBONDS>', &
             'Internal error ww-processing INBLO')
     ENDIF
     !
     CALL set_param('NNOO',NBOO)
     CALL set_param('NNVU',NBVU)
     CALL set_param('NNUU',NBUU)
     CALL set_param('NNNB',NNNB)
     IF(PRNLEV >= 5)THEN
        WRITE(OUTU,780) 'primary',NBOO,NBVU,NBUU,NNNB
     ENDIF
     IF(NTRNSX > 0) THEN
        ! Space allocation, based on existing image list size
        IF(.NOT.ALLOCATED(IWOONBLI))THEN
           ALLOCATE(IWOONBLI(MAXAIM))
#if KEY_IMCUBES==1
           !               ENDIF                                   
#endif
        ENDIF
        IF(.NOT.ALLOCATED(JWOONBLI))THEN
           MXJWWOOI=bimag%NIMNB*(FLOAT(NWWO)/FLOAT(NATOM))**2
           ALLOCATE(JWOONBLI(MXJWWOOI))
        ENDIF
        IF(PRNLEV >= 5) WRITE(OUTU,219)  'IMAGE ',MXJWWOOI
        ! Zero entries for primary atoms
        IF(QVV) IWOONBLI(1:NATOM)=0
        IF(QVU) IVUNBLI(1:NATOM)=0
        IF(QUU) IUUNBLI(1:NATOM)=0
        CALL WWSPLTNB(CMPLTD,.TRUE.,1,NATIM, &
             bimag%IMBLO,bimag%IMJNB, &
             MXJWWOOI,IWOONBLI,JWOONBLI, &
#if KEY_IMCUBES==1
             LBYCBIM,                                        & 
#endif
             bimag%NIMNB,NBOO,NBVU,NBUU)
        IF(.NOT. CMPLTD) THEN
           ! Resize to now exactly known sizes and redo the split
           ! Use a small margin to reduce number of deallocate/allocate events
           DEALLOCATE(JWOONBLI)
           MXJWWOOI=1.1*NBOO+2
           ALLOCATE(JWOONBLI(MXJWWOOI))
           IF(PRNLEV >= 5) WRITE(OUTU,219) 'RESIZE IMAGE ',MXJWWOOI
           CALL WWSPLTNB(CMPLTD,.TRUE.,1,NATIM, &
                bimag%IMBLO,bimag%IMJNB, &
                MXJWWOOI,IWOONBLI,JWOONBLI, &
#if KEY_IMCUBES==1
                LBYCBIM,                                    & 
#endif
                bimag%NIMNB,NBOO,NBVU,NBUU)
           !
           IF(.NOT. CMPLTD) CALL WRNDIE(-4,'<NBONDS>', &
                'Internal error ww-processing IMBLO')
        ENDIF
        CALL set_param('NIMGOO',NBOO)
        CALL set_param('NIMGVU',NBVU)
        CALL set_param('NIMGUU',NBUU)
        CALL set_param('NIMGVV', bimag%NIMNB)
        IF(PRNLEV >= 5)THEN
           WRITE(OUTU,780) 'image',NBOO,NBVU,NBUU, &
                bimag%NIMNB
        ENDIF
     ENDIF
  ENDIF lookup_0
780 FORMAT(' Processed ',A,' lookup lists' &
         /'    Solvent-solvent OO pairs:',I12, &
         /'        Solvent-solute pairs:',I12, &
         /'         Solute-solute pairs:',I12, &
         /' Pairs left on standard list:',I12,/)
#endif /*  LOOKUP*/
  RETURN
  !
125 FORMAT(' NBONDS: Using routine ',A,' for list generation.')
  !
END SUBROUTINE NBONDS

