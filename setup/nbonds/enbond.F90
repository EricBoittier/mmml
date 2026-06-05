module enbond_mod

contains

SUBROUTINE ENBOND(ENB,EEL,BNBND,IFRSTA, &
     NATOM,CG,RSCLF,NGRP,IGPBS,IGPTYP,IAC,IACNBX,DX,DY,DZ, &
     X,Y,Z,QECONTX,ECONTX,QEWEX,EELX,DD1,IUPT,QSECD,QVDW,QELEC, &
#if KEY_FLUCQ==1
     QFLUC,FQCFOR,               & 
#endif
     QST2,NST2,EST2,QEXTND &
#if KEY_WCA==1
     ,LLSOFT, SCVDWCUTR, WCA      & 
#endif
#if KEY_LJPME==1
    , QLJEXC, ELJEXC  & 
#endif
     )

  !-----------------------------------------------------------------------
  !
  !     This routine sets up, allocates temporary memory, and then calls a
  !     specific vdw and electrostatics routine.  It determines the correct
  !     routine based on user methods, options, and selected FAST options.
  !
  !     Note: A development practice which does not conform to CHARMM rules
  !           has led to features that are ONLY available in specific fast
  !           routines and not in the generic (slow) routine.  This may not
  !           be handled properly and wrong results may be generated.
  !           Users: Beware... Developers: Don't do it...  - BRB
  !
  !     By Bernard R. Brooks  (mostly)  1983
  !     FORTRAN version By Youngdo Won 12/15/90
  !     MMFF added by Jay Banks 20 Oct 95
  !
  use ewald,only: ewldex, &
#if KEY_FASTEW==1
       intcrecsz, realcrecsz, & 
#endif
       lewald, &
       KAPPA,EWVIRIAL,ERFMOD,EWLDEX ! FOR DCM
#if KEY_FEWMFC==1 /*mfc_fast_ewald*/
#if KEY_FEWSB==0 /*fewsb*/
#if KEY_CHEQ==1 /*cheq*/
  use ewald,only: rewald95_cheq
#endif /* (cheq)*/
  use ewald,only: rewald95
#else /* (fewsb)*/
  use ewald,only: rewald95_sb
#endif /* (fewsb)*/
#else /* (mfc_fast_ewald)*/
  use ewald,only: rewald
#endif /* (mfc_fast_ewald)*/



  use new_timer,only:timer_start,timer_stop,T_ewald,T_dir  & 
       ,T_ips,T_ipsnb                                     
#if KEY_CHEQ==1
  use cheq,only: qcg      
#endif
  use memory
#if KEY_GRAPE==1
  use grapemod,only: enbgrap                              
#endif

  use chm_kinds
  use chm_types
  use consta
  use dimens_fcm
  use number
  use enbonda
  use enbondg
#if KEY_BLOCK==1
  use block_fcm
  use lambdam
#endif 
  use etablem
  use fast
  use inbnd
  use nbips
  use param
  use stream
  use fmam
  use timerm
  use eintern_fast,only:fastst
  use parallel
  use pbound
  use grape
  use ffieldm
  use mmffm
  use tbmts
#if KEY_PERT==1
  use pert, only: QPERT
#endif
#if KEY_SCPISM==1
  use scpismm,only:scpevdwf,scpism         
#endif
  use mccent
  use gcmc
#if KEY_ACE==1
  use ace_module
  use code
#endif 
#if KEY_PIPF==1
  use pipfm      
#endif
  use memory
  use mtp_fcm
  use mtpl_fcm

  use dcm_fcm,only: QDCM,DCME

#if KEY_DOMDEC==1
  use enbxfast,only:enbcalc_xfast  
  use domdec_common,only:q_domdec, q_gpu  
  use ebonded_domdec,only:e14_domdec
#endif
#if KEY_DOMDEC_GPU==1
  use domdec_util_gpu_mod,only:range_start,range_stop  
#endif
#if KEY_LJPME==1
  use pmeutil, only: QLJPME
  use energym, only: QETERM, LJEXC
  use ewald, only: ljpmeex
#endif
  !---   use nbutil_module,only:getbnd
  !
  implicit none
  integer,allocatable,dimension(:) :: ioff,aceiof
  character(len=10) :: file = "enbond.src"
  character(len=9) :: routine = "enbond"
  real(chm_real)  ENB,EEL
  !!      INTEGER BNBND(*)
  type(nonbondDataStructure) BNBND
  INTEGER NATOM, IFRSTA
  real(chm_real)  CG(*), RSCLF(*)
  INTEGER NGRP
  INTEGER IGPBS(*),IGPTYP(*),IAC(*),IACNBX(*)
  real(chm_real)  DX(*),DY(*),DZ(*),X(*),Y(*),Z(*)
  LOGICAL QECONTX
  real(chm_real) ECONTX(*)
  LOGICAL QEWEX
  real(chm_real)  EELX
  real(chm_real)  DD1(*)
  INTEGER IUPT(*)
  LOGICAL QSECD,QVDW,QELEC,QST2,QEXTND
  INTEGER NST2, I
  real(chm_real)  EST2

#if KEY_FLUCQ==1
  LOGICAL QFLUC
  real(chm_real)  FQCFOR(*)
#endif 
#if KEY_WCA==1
  real(chm_real) WCA(*), SCVDWCUTR
  LOGICAL LLSOFT
#endif 
  !
  LOGICAL LVDWX,LELECX,LEWLDX,IFALSE,LUSED
#if KEY_FASTEW==1
  ! Cache storage for FAST EWald version of REWALD; Scott Brozell TSRI
  ! Including these files is required; they are already included above
  INTEGER MAXPAIRS  ! maximum number of pairs
  INTEGER SIZEINTC  ! size of the interger cache
  INTEGER SIZErealC ! size of the real(chm_real) cache
  INTEGER,allocatable,dimension(:) :: INTCACHE  ! the integer cache STACK pointer
  real(chm_real),allocatable,dimension(:) :: &
       realCACHE ! the real(chm_real) cache STACK pointer
  !.ab.
  INTEGER ICSZ,RCSZ
#endif /*  FASTEW*/
  !
  real(chm_real) ELECL,EVDWL,SVCTOF,SVCTON
#if KEY_ACE==1
  INTEGER NTEMP
#endif 
  real(chm_real),parameter :: eta0(1,1) = reshape((/ ZERO /), (/ 1, 1 /))
#if KEY_LJPME==1
    LOGICAL  QLJEXC
    real(chm_real) ELJEXC
#endif
  integer, dimension(1), target :: pzer1 = [0]
  integer, dimension(:), pointer :: tjnb, tinblo

  IF(NATOM <= 0) RETURN
  !
  !     Make sure we have the current non-bond flags and cut-offs. This
  !     will handle any changes made due to pert or tsm.
  CALL GETBND(BNBND,.TRUE.)
  !
  LUSED=.FALSE.
  ENB  = ZERO
  EEL  = ZERO
  EST2 = ZERO
  !
  LVDWX=QVDW.AND.LVDW
  LELECX=QELEC.AND.LELEC
  LEWLDX=LEWALD.AND.LELECX

#if KEY_LJPME==1
  ! Try to catch all known incompatibilities (due to not being implemented,
  ! rather than an inherent problem) with the LJPME code.
  if(qljpme) then
#if KEY_DOMDEC==1
     if(.not.q_domdec) then
#endif
        if(faster .ne. -1) then
           CALL WRNDIE(-5,'<ENBOND>', 'LJPMe requires either DOMDec or FASTer OFF')
        endif
#if KEY_DOMDEC==1
     endif
#endif

#if KEY_DOMDEC_GPU==1
     if (q_gpu) CALL WRNDIE(-5,'<ENBOND>', 'LJPMe is not compatible with DOMDec GPU')
#endif

#if KEY_BLOCK==1
     if (qblock) CALL WRNDIE(-5,'<ENBOND>', 'LJPMe is not compatible with BLOCK')
#endif

#if KEY_PBOUND==1
     if(QBOUN) CALL WRNDIE(-5,'<ENBOND>', 'LJPMe is not compatible with BOUNd')
#endif

#if KEY_PERT==1
     if(QPERT) CALL WRNDIE(-5,'<ENBOND>', 'LJPMe is not compatible with PERT')
#endif
  endif
#endif /* LJPME */
  !
  !=======================================================================
  !
  ! See if nonbond energy tables are in use. if so, pass all control to ETABLE.
  !
#if KEY_PARALLEL==1
  IF(MYNOD == 0) THEN          
#endif
#if KEY_VIBPARA==0
#if KEY_NOMISC==0
     IF(NTABTP > 0) THEN
        IF(LVDWX.OR.LELECX) THEN
           IF(PRNLEV > 6) WRITE(OUTU,125) 'ETABLE'
           CALL ETABLE(ENB,EEL,NATOM,X,Y,Z,DX,DY,DZ,EPS, &
                CTOFNB,BNBND%INBLO,BNBND%JNB,IAC,CG, &
                NTABTP,NTABSQ,NTABLN,NTABST,ITBITC,TABDEN,TABHOM,TABRHO, &
                IPTTB1,IPTTB2,IPTTB3,IPTRT, &
                DD1,IUPT,QSECD &
#if KEY_IMCUBES==1
                ,lbycbim         & 
#endif
                )
        ENDIF
        RETURN
     ENDIF
#endif 
#endif 

#if KEY_PARALLEL==1
  ENDIF                        
#endif
  !
  !=======================================================================
  ! Are there ST2 water molecules?
  !
#if KEY_NOST2==0
  if(nst2 > 0.and.qst2) then
     if(qsecd) call wrndie(-3,'<enbond>', &
          ' can not use second derivatives with st2.')
     if(prnlev > 6) write(outu,125) 'enst2'
     call enst2(est2,x,y,z,dx,dy,dz,bnbnd%jnbg, &
          bnbnd%inblog,ngrp,ctonnb,ctofnb,qextnd)
     !-- don't return unless all atoms are st2
     if(nst2 == natom) return
  endif
#endif
  !--- 
  !=======================================================================
  !--- 
  !---  Test if fast routine is allowed and requested.
  !--- 
  if(faster >= 0) call fastst(x,y,z,dx,dy,dz,qsecd)
  !---
  !=======================================================================
  !---  See if special hardware (GRAPE) is in use
  !---
#if KEY_GRAPE==1 /*mdgrape_1*/
  grapeif:if(lgrape.or.lnocut) then
     !---
     !---     call mdgrape-1,2,3 or GPU routines
     !---
     call enbgrap(natom,lelecx,x,y,z,dx,dy,dz,enb,eel, &
          bnbnd%inb14,bnbnd%iblo14, &
#if KEY_FLUCQ==1
          qfluc,fqcfor,                           & 
#endif
          lused)
     !
     if(lused) then
        if(prnlev > 6) write(outu,125) 'enbgrap'
        return
     endif
  endif grapeif
#endif /* (mdgrape_1)*/

  !=======================================================================
#if KEY_NBIPS==1 /*nbips*/
  !---
  !--- See if IPS algorithm is in use. 
  !---
  !WXW  IPS for nonbonded pairs
  IF(QIPSONLY)THEN
     call timer_start(T_ips)     
     call timer_start(T_ipsnb)     
     IF(PRNLEV > 6) WRITE(OUTU,125) 'ENBIPS'
     CALL ENBIPS(ENB,EEL,IFRSTA,NATOM,QVDW,QELEC,LCONS, &
          BNBND%JNB,BNBND%INBLO, &
          CG,RSCLF,CNBA,CNBB,MAXCN,IAC,ITC,NATC,EPS, &
          DX,DY,DZ,X,Y,Z,QECONTX,ECONTX,DD1,IUPT,QSECD &
#if KEY_IMCUBES==1
          ,lbycbim                           &
#endif
          )
     call timer_stop(T_ipsnb)     
     call timer_stop(T_ips)     
     RETURN
  ENDIF
#endif /*  (nbips)*/

  !=====================================================================
  !--- Fast group-group routines

  group_nb:if(lgroup) then

     !---  reject other unsupported methods
#if KEY_MMFF==1 /*mmff_fast*/
     !---  no mmff nonbond fast group routines
     if(ffield == mmff) goto 200
#endif /* (mmff_fast)*/
#if KEY_BLOCK==1
     !-- spliting the vdw up in block currently not supported in group-group routine
     IF(QBVSPLT) goto 200
     !.ab.
     IF(QHYBH.AND.LFAST >= 0) CALL WRNDIE(-5,'<ENBOND>', &
          'HYBH and GROUP-NBond incompatible.')
     !.ab.
#endif 
     if(lfast < 0) goto 200

#if KEY_MC==1
     !---If centering is done in MC do not reallocate space
     if (lcentr  ==  0) then
        call mc_alloc_centers('enbond.src', 'ENBOND', NGRP)
     ENDIF
#else /*  MC*/
     call mc_alloc_centers('enbond.src', 'ENBOND', NGRP)
#endif /*  MC*/
     call enbfsg(enb,eel,lelecx,lvdwx,bnbnd%jnbg, &
          bnbnd%inblog,ifrsta,ngrp,igpbs,igptyp, &
          bnbnd%inb14,bnbnd%iblo14,cg, &
          iacnbx,nitcc2,lowtp, &
#if KEY_BLOCK==1
          iblckp,blcoee,blcoev,blcoevr,blcoeva, &     
#endif
#if KEY_MC==1
          lcentr,                       & 
#endif
          xcent,ycent,zcent,qcent, &
#if KEY_BLOCK==1
          natom,                        &  /*ldm*/
#endif
#if KEY_FLUCQ==1
          qfluc,fqcfor,                 & 
#endif
          lused, &
#if KEY_WCA==1
          llsoft, scvdwcutr, wca,       & 
#endif
          qextnd)
     if(lused) then
        if(prnlev > 6) write(outu,125) 'ENBFSG'
#if KEY_MC==1
        if (lcentr  ==  0) then
           call mc_dealloc_centers('enbond.src', 'ENBOND', NGRP)
        endif
#endif /*  MC*/
        goto 300
     endif
#if KEY_MC==1
     !--- If centering was not done in MC free space
     if (lcentr  ==  0) then
        call mc_dealloc_centers('enbond.src', 'ENBOND', NGRP)
     endif
#endif /*  MC*/
     !yw...23-JUL-2004 c32p2 fix
     goto 200
  endif Group_NB
  ! ------------- End of Group fast routines ---------------
  !=======================================================================
  !--- 
  !---  If fma is to be used

  IF (LFMA) then
#if KEY_FMA==1 /*fma_nb*/
     !heck for conflicts:
     IF(.NOT.LCONS) CALL WRNDIE(-3,'<ENBOND>', &
          'The FMA method does not support R-dielectric')
     !
#if KEY_DEBUG==1
     if(TIMER > 1) CALL WRTTIM(' FASTMA: Before FMA calculation') 
#endif
     IF(PRNLEV > 6) WRITE(OUTU,125) 'FMA'
     call FMA(Eel, Enb, (Level+1), Terms)
#if KEY_DEBUG==1
     if(Timer > 1) CALL WRTTIM(' FASTMA: after FMA calculation')  
#endif
     !
     ! then correct the FMA results for exclusions
     Elecl = Zero
     if (NNB14  >  0) then
        SVCTOF = CTOFNB
        SVCTON = CTONNB
        CTOFNB = THOSND
        CTONNB = THOSND
        IFALSE = .FALSE.
        CALL ENBFS8(EVDWL,ELECL,LELECX,IFALSE,IFRSTA,NATOM,CG, &
             BNBND%INB14,BNBND%IBLO14, &
             IACNBX,NITCC2,LOWTP, &
#if KEY_BLOCK==1
             IBLCKP,BLCOEE,BLCOEV,             & 
#endif
#if KEY_BLOCK==1
             BLCOEVR,BLCOEVA,                        & 
#endif
#if KEY_CHEQ==1
             eta0,.FALSE.,                                       & 
#endif
#if KEY_FLUCQ==1
             QFLUC,FQCFOR,                                       & 
#endif
#if KEY_WCA==1
             LLSOFT,SCVDWCUTR,WCA,                               & 
#endif
             LUSED,QGRF)         ! GRF -- Wei Chen 2015
#if KEY_DEBUG==1
        if(Timer > 1) CALL WRTTIM                               & 
#endif
#if KEY_DEBUG==1
             (' FASTMA: after exclusion calculation')  
#endif
        EEL = EEL - ElecL
        CTOFNB = SVCTOF
        CTONNB = SVCTON
     endif
     ! Finally compute VDW if required
     if (LVdwx) then
        EVdwl = ZERO
        CALL ENBFS8(EVDWL,ELECL,IFALSE,LVDWX,IFRSTA,NATOM,CG, &
             BNBND%JNB,BNBND%INBLO, &
             IACNBX,NITCC2,LOWTP, &
#if KEY_BLOCK==1
             IBLCKP,BLCOEE,BLCOEV,             & 
#endif
#if KEY_BLOCK==1
             BLCOEVR,BLCOEVA,                        & 
#endif
#if KEY_CHEQ==1
             eta0,.FALSE.,                                       & 
#endif
#if KEY_FLUCQ==1
             QFLUC,FQCFOR,                                       & 
#endif
#if KEY_WCA==1
             LLSOFT,SCVDWCUTR,WCA,                               & 
#endif
             LUSED,QGRF)      ! GRF -- Wei Chen 2015
#if KEY_DEBUG==1
        if(Timer > 1) CALL WRTTIM                               & 
#endif
#if KEY_DEBUG==1
             (' FASTMA: after VDW calculation')                 
#endif
        Enb = Enb + EVdwl
     endif
     RETURN
#else /* (fma_nb)*/
     CALL WRNDIE(3,'<ENBOND>','No FMA code compiled.')
     !--- don't return - use generic routine instead.
     GOTO 200
#endif /* (fma_nb)*/
  ENDIF
  !
  !=======================================================================
  !
  ! Ace is to be used
#if KEY_ACE==1 /*ace_nb*/
  Ace_NB:IF (LACE) THEN
     !--- allocate memory
     NTEMP=GETNNB(BNBND%INBLO,NATOM)
     IF(NPAIR < NTEMP) THEN
        IF(NPAIR > 0) THEN
           call chmdealloc('enbond.src','ENBOND','SWIT',NPAIR,crl=SWIT)
           call chmdealloc('enbond.src','ENBOND','DSWIT',NPAIR,crl=DSWIT)
           call chmdealloc('enbond.src','ENBOND','DISTM',NPAIR,crl=DISTM)
           call chmdealloc('enbond.src','ENBOND','XFSDI',2*NPAIR,crl=XFSDI)
           call chmdealloc('enbond.src','ENBOND','YFSDI',2*NPAIR,crl=YFSDI)
           call chmdealloc('enbond.src','ENBOND','ZFSDI',2*NPAIR,crl=ZFSDI)
        ENDIF
        !--- add 5% to accomodate fluctuations in size of pair list:
        NPAIR=1+INT(real(NTEMP)*1.05)
        call chmalloc('enbond.src','ENBOND','SWIT',NPAIR,crl=SWIT)
        call chmalloc('enbond.src','ENBOND','DSWIT',NPAIR,crl=DSWIT)
        call chmalloc('enbond.src','ENBOND','DISTM',NPAIR,crl=DISTM)
        call chmalloc('enbond.src','ENBOND','XFSDI',2*NPAIR,crl=XFSDI)
        call chmalloc('enbond.src','ENBOND','YFSDI',2*NPAIR,crl=YFSDI)
        call chmalloc('enbond.src','ENBOND','ZFSDI',2*NPAIR,crl=ZFSDI)
        SWIT(1:NPAIR) = ZERO
        DSWIT(1:NPAIR) = ZERO
        DISTM(1:NPAIR) = ZERO
     ENDIF
     NTEMP=GETNNB(BNBND%IBLO14,NATOM)
     IF(NPA14 < NTEMP) THEN
        IF(NPA14 > 0) THEN
           call chmdealloc('enbond.src','ENBOND','SA14P' ,NPA14,crl=SA14P)
           call chmdealloc('enbond.src','ENBOND','SWA14P',NPA14,crl=SWA14P)
        ENDIF
        NPA14=NTEMP
        call chmalloc('enbond.src','ENBOND','SA14P', NPA14,crl=SA14P)
        call chmalloc('enbond.src','ENBOND','SWA14P',NPA14,crl=SWA14P)
        SA14P(1:NPA14) = ZERO
        SWA14P(1:NPA14) = ZERO
     ENDIF
     IF(NATACE < NATOM) THEN
        IF(NATACE > 0) THEN

           call chmdealloc('enbond.src','ENBOND','CGIACP',NATACE,crl=CGIACP)

           call chmdealloc('enbond.src','ENBOND','CGSACP',NATACE,crl=CGSACP)
           call chmdealloc('enbond.src','ENBOND','desdbp',NATACE,crl=desdbp)
           call chmdealloc('enbond.src','ENBOND','ESFIX',NATACE,crl=ESFIX)
           call chmdealloc('enbond.src','ENBOND','ESELF',NATACE,crl=ESELF)
           call chmdealloc('enbond.src','ENBOND','BSOLV',NATACE,crl=BSOLV)
           call chmdealloc('enbond.src','ENBOND','DISUM',NATACE,crl=DISUM)
           call chmdealloc('enbond.src','ENBOND','XFREA',NATACE,crl=XFREA)
           call chmdealloc('enbond.src','ENBOND','YFREA',NATACE,crl=YFREA)
           call chmdealloc('enbond.src','ENBOND','ZFREA',NATACE,crl=ZFREA)
           call chmdealloc('enbond.src','ENBOND','CG2',NATACE,crl=CG2)
           call chmdealloc('enbond.src','ENBOND','DBDE',NATACE,crl=DBDE)
        ENDIF
        NATACE=NATOM

        call chmalloc('enbond.src','ENBOND','CGIACP',NATOM,crl=CGIACP)

        call chmalloc('enbond.src','ENBOND','CGSACP',NATOM,crl=CGSACP)

        call chmalloc('enbond.src','ENBOND','DESDBP',NATOM,crl=DESDBP)
        call chmalloc('enbond.src','ENBOND','ESFIX',NATOM,crl=ESFIX)
        call chmalloc('enbond.src','ENBOND','ESELF',NATOM,crl=ESELF)

        call chmalloc('enbond.src','ENBOND','BSOLV',NATOM,crl=BSOLV)


        call chmalloc('enbond.src','ENBOND','DISUM',NATOM,crl=DISUM)
        call chmalloc('enbond.src','ENBOND','XFREA',NATOM,crl=XFREA)
        call chmalloc('enbond.src','ENBOND','YFREA',NATOM,crl=YFREA)
        call chmalloc('enbond.src','ENBOND','ZFREA',NATOM,crl=ZFREA)
        call chmalloc('enbond.src','ENBOND','CG2',NATOM,crl=CG2)
        call chmalloc('enbond.src','ENBOND','DBDE',NATOM,crl=DBDE)
        !--- initialize BORN radii (currently not essential; this
        !--- is done in case a future, iterative self energy potential
        !--- is called where the Born radii should start from a
        !--- first, sensible guess):
        CALL INIB(NATOM,BSOLV)
     ENDIF
     !--- initialize self energy contributions of fixed atoms;
     !--- set charge arrays to be used for self and interaction
     !--- energy, depending on ionic scaling factors, and
     !--- calculate self energy contribution for nb-exclusion atom
     !--- pairs and store their (fixed) distance, switching function:
     IF((QFIACE == 0).OR.(ACEUP)) THEN
        call chmalloc('enbond.src','ENBOND','ACEIOF',MAXATC,intg=ACEIOF)
        CALL IESFIX(LELECX,NATOM, &
#if KEY_BLOCK==1
             IBLCKP,BLCOEE,BLCOEV,BLCOEVR,BLCOEVA, &      
#endif
             ACEIOF,CGIACP,CGSACP, &
             BNBND%INB14,BNBND%IBLO14, &
             SA14P,SWA14P, &
             ESFIX)
        call chmdealloc('enbond.src','ENBOND','ACEIOF',MAXATC,intg=ACEIOF)
        QFIACE=1
        ACEUP=.FALSE.
     ENDIF
     !       calc. ace electrostatic energy
     CALL ENACE(ENB,EEL,LELECX,LVDWX,NATOM, &
          BNBND%JNB,BNBND%INBLO, &
          BNBND%INB14,BNBND%IBLO14, &
#if KEY_BLOCK==1
          IBLCKP,BLCOEE,BLCOEV,BLCOEVR,BLCOEVA, &    
#endif
          ESELF,ESFIX, &
          CGIACP,CGSACP,DESDBP, &
          BSOLV,DISUM,DISTM, &
          SA14P,SWA14P, &
          XFREA,YFREA,ZFREA, &
          XFSDI,YFSDI,ZFSDI, &
          CG2,DBDE,SWIT,DSWIT)
     IF(PRNLEV > 6) WRITE(OUTU,125) 'ENACE'
     !       calc. VDW
     IF (LVDWX) THEN
        EVDWL = ZERO
        IFALSE = .FALSE.
        CALL ENBFS8(EVDWL,ELECL,IFALSE,LVDWX,IFRSTA,NATOM,CG, &
             BNBND%JNB,BNBND%INBLO, &
             IACNBX,NITCC2,LOWTP, &
#if KEY_BLOCK==1
             IBLCKP,BLCOEE,BLCOEV,    & 
#endif
#if KEY_BLOCK==1
             BLCOEVR,BLCOEVA,               & 
#endif
#if KEY_CHEQ==1
             eta0,.FALSE.,                              & 
#endif
#if KEY_FLUCQ==1
             QFLUC,FQCFOR,                              & 
#endif
#if KEY_WCA==1
             LLSOFT,SCVDWCUTR,WCA,                      & 
#endif
             LUSED,QGRF)      ! GRF -- Wei Chen 2015
        ENB = ENB + EVDWL
        IF(LUSED) THEN
           IF(PRNLEV > 6) WRITE(OUTU,125) 'ENBFS8(vdW)'
        ENDIF
     ENDIF
     RETURN
  ENDIF Ace_NB
#endif /* (ace_nb)*/
  !
  !=======================================================================
  ! Fast atom-atom routines
  !

  IF(LFAST < 0) GOTO 200
#if KEY_MMFF==1 /*mmff_fast*/
  !     no mmff nonbond fast atom-atom routines
  IF(FFIELD == MMFF) GOTO 200
#endif /* (mmff_fast)*/
#if KEY_MTPL==1
  ! MTPL requires FAST OFF
  IF(QMTPL) GOTO 200
#endif /* mtpl */

  IF(QDCM) GOTO 200

  !-----------------------------------------------------------------------
  !--- 
  !---  Fast routine selection:
  !---     Try to use the "highest" selected fast routine.
  !---     If options are not allowed, go to the next...
  !---     To see what routine actually was used, set PRNLEV to 7 or more.
  !--- 
  !-----------------------------------------------------------------------
  !---  Check if expanded FAST routines can be used

  IF(LFAST == 1) GOTO 180   ! generic fast explicitly specified
  !--- reject other unsupported methods
#if KEY_NOST2==0
  IF(NST2 > 0.AND.QST2) GOTO 180    
#endif
#if KEY_FLUCQ==1
  IF(QFLUC)  GOTO 180        
#endif
#if KEY_CHEQ==1
  IF(QCG)    GOTO 180        
#endif
#if KEY_PBOUND==1
  IF(QBOUN)  GOTO 180        
#endif
#if KEY_MTS==1
  IF(SLFG)   GOTO 180        
#endif
  IF(QETEN .OR. QETSR)  GOTO 180        

  !-------------- QQQQQ NOTE:  QGCMC needs to be set properly  QQQQQQQ

#if KEY_GCMC==1
  IF(QGCMC)  GOTO 180        
#endif
#if KEY_SOFTVDW==1
  IF(QGAMIN) GOTO 180        
#endif
#if KEY_SCPISM==1
  IF(SCPISM) GOTO 180        
#endif
#if KEY_CFF==1
  IF(FFIELD == CFF) GOTO 180 
#endif
#if KEY_BLOCK==1 /*block_tests*/
  IF(QBLOCK) THEN
     IF(NOFORC) GOTO 180
     IF(QPRNTV) GOTO 180
#if KEY_DOCK==1
     IF(QDOCK)  GOTO 180    
#endif
     IF(QLDM)   GOTO 180    !ldm
  ENDIF
#endif /* (block_tests)*/
#if KEY_DOMDEC==1
  if (.not.q_domdec) goto 170  ! Expanded routines 
#endif

  !------------------------------------------------------------------
  !          DOMDEC non-bonded routines can be used
  !------------------------------------------------------------------
  IF(LEWLDX) THEN
     call timer_start(T_ewald)
     call timer_start(T_dir)
  endif
  !  ----------- ENBCALC_XFAST -----------------------
#if KEY_DOMDEC==1
  call enbcalc_xfast(x, y, z, cg, dx, dy, dz, eel, enb, eelx, lused)
#endif
  if(lewldx) then
     call timer_stop(T_dir)  
     call timer_stop(T_ewald)
  endif
  if(lused) then
     IF(PRNLEV > 6) WRITE(OUTU,125) 'ENBCALC_XFAST'
     !-------------------------------------------------------------------
     ! PJ 06/2005
#if KEY_PIPF==1
     if (qpipf) then
        if(prnlev > 6) write(outu,125) 'epipf'
        call epfdrv(ifrsta,natom,bnbnd%jnb,bnbnd%inblo, &
             cg,maxcn,iac,itc,lelecx,lcons,lshft,lvshft,lfswt, &
             lvfswt,dx,dy,dz,x,y,z,ctonnb,ctofnb,eps,e14fac,alp, &
             dd1,iupt,qsecd &
             )
     endif
#endif 
     !------- expand routine returned not used, go to safe slow routines
     goto 300
  endif

  !------------------------------------------------------------------
  !          Expanded fastest routines can be used
  !------------------------------------------------------------------
170 continue
  IF(LEWLDX) THEN
     call timer_start(T_ewald)
     call timer_start(T_dir)  
  endif
  !  ----------- ENBAEXP -----------------------
  if(associated(bnbnd%jnb)) then
     tjnb => bnbnd%jnb
     tinblo => bnbnd%inblo
  else
     tjnb => pzer1
     tinblo => pzer1
  endif
  call enbaexp(enb,eel,lelecx,lvdwx,ifrsta,natom,cg, &
       tjnb,tinblo, &
       iacnbx, &
       lused)
  if(lewldx) then
     call timer_stop(T_dir)  
     call timer_stop(T_ewald)
  endif
  if(lused) then
     IF(PRNLEV > 6) WRITE(OUTU,125) 'ENBAEXP'
     !-------------------------------------------------------------------
     ! PJ 06/2005
#if KEY_PIPF==1
     if (qpipf) then
        if(prnlev > 6) write(outu,125) 'epipf'
        call epfdrv(ifrsta,natom,bnbnd%jnb,bnbnd%inblo, &
             cg,maxcn,iac,itc,lelecx,lcons,lshft,lvshft,lfswt, &
             lvfswt,dx,dy,dz,x,y,z,ctonnb,ctofnb,eps,e14fac,alp, &
             dd1,iupt,qsecd &
             )
     endif
#endif 
     !------- expand routine returned not used, go to safe slow routines
     goto 300
  endif

  !--------------------------------------------------------------------
  !         Fast 1 routines
  !--------------------------------------------------------------------
180 CONTINUE

  Faster_1:IF(LEWLDX) THEN
     !--- call the fast ewald nonbond routine
     call timer_start(T_ewald) 
     call timer_start(T_dir) 

#if KEY_FEWMFC==1 /*fast_ew_mfc*/
     !======================== FASTER EWALD with Vector constructs =======================
     maxpairs = getmaxnpr(natom,bnbnd%inblo)
#if KEY_FEWSB==1 /*sb*/
     call rewald95_sb(enb,eel,natom,bnbnd%jnb, &
          bnbnd%inblo,&
          lelecx,lvdwx, &
          iacnbx,nitcc2,lowtp, &
          dx,dy,dz,x,y,z,cg, &
          maxpairs,lused)
     IF(lused .and. PRNLEV > 6) WRITE(OUTU,125) 'REWALD95_SB'


#if KEY_CHEQ==1
#error  FEWSB does not work with CHEQ
#endif 


#else /* (sb)*/
#if KEY_CHEQ==1 /*cheq*/
     if(qcg) then
        call rewald95_cheq(enb,eel,natom,bnbnd%jnb, &
             bnbnd%inblo,&
             lelecx,lvdwx, &
             iacnbx,nitcc2,lowtp, &
             dx,dy,dz,x,y,z,cg, &
             maxpairs,lused)
     else
#endif /* (cheq)*/
        call rewald95(enb,eel,natom,bnbnd%jnb, &
             bnbnd%inblo,&
             lelecx,lvdwx, &
             iacnbx,nitcc2,lowtp, &
             dx,dy,dz,x,y,z,cg, &
             maxpairs,lused)
#if KEY_CHEQ==1
     endif                  
#endif
     IF(lused .and. PRNLEV > 6) WRITE(OUTU,125) 'REWALD95 '
     !         endif

#endif /* (sb)*/
     !------------ End of faster ewald with Vector constructs ------------------------

#else /* (fast_ew_mfc)*/
     !=============================================================================================
     !------------ REWALD original and with older faster ewald with localized data ----------------
#if KEY_FASTEW==1
     maxpairs = getmaxnpr(natom,bnbnd%inblo)
     !--- cache storage for fast ewald version of rewald; 
     !---                           scott brozell tsri
     if(prnlev > 7) write(outu,235) maxpairs
     !--- 8 is >6, but otherwise is a random number
235  format(' enbond: the maximum number of pairs is ',i10)
     !.ab.Larger size for dE/dlambda...
     icsz = intcrecsz
     rcsz = realcrecsz
#if KEY_BLOCK==1
     IF(QHYBH) THEN
        icsz = icsz + 1
        rcsz = rcsz + 2
     ENDIF
#endif 
     sizeintc  = maxpairs*icsz
     sizerealc = maxpairs*rcsz
     call chmalloc(file,routine,"intcache", sizeintc, intg=intcache)
     call chmalloc(file,routine,"realcache",sizerealc, crl=realcache)
#endif 
     !.ab.Not ported. Is it necessary ? Contact A.Blondel.
#if KEY_BLOCK==1
     IF(QHYBH.AND.LFAST >= 0) CALL WRNDIE(-5,'<ENBOND>',  & 
#endif
#if KEY_BLOCK==1
          'HYBH and REWALD95 incompatible, see code.')   
#endif

     call rewald(enb,eel,natom,bnbnd%jnb, &
          bnbnd%inblo,&
          lelecx,lvdwx, &
          iacnbx,nitcc2,lowtp, &
          dx,dy,dz,x,y,z,cg, &
#if KEY_FLUCQ==1
          qfluc,fqcfor,                      & 
#endif
#if KEY_FASTEW==1
          intcache,ICSZ,realcache,RCSZ,  & 
#endif
          lused)
     IF(lused .and. PRNLEV > 6) WRITE(OUTU,125) 'REWALD'
#if KEY_FASTEW==1 /*few*/
     !---     Cache storage cleanup for FAST EWald version of REWALD
     call chmdealloc(file,routine,"intcache", sizeintc,  intg=intcache)
     call chmdealloc(file,routine,"realcache",sizerealc, crl=realcache)
#endif /*  (few)*/
#endif /* (fast_ew_mfc)*/
     call timer_stop(T_dir) 
     call timer_stop(T_ewald) 
     if(lused) goto 300
  else Faster_1
     !---------------------------------------------------------------
#if KEY_CFF==1 /*cff*/
     FFIELD_cff:if(ffield == cff) then
        call enbfs8_cff(enb,eel,lelecx,lvdwx,ifrsta,natom,cg, &
             bnbnd%jnb,bnbnd%inblo, &
             iacnbx,nitcc2,lowtp, &
#if KEY_BLOCK==1
             iblckp,blcoee,blcoev,blcoevr,blcoeva, &   
#endif
#if KEY_FLUCQ==1
             qfluc,fqcfor,                               & 
#endif
             lused)
        if(lused) then
           if(prnlev > 6) write(outu,125) 'enbfs8_cff'
           return
        endif
     ENDIF FFIELD_cff
#endif /* (cff)*/
     !----------------------------------------------------------------
#if KEY_SCPISM==1
     Scpismnb:IF (SCPISM) THEN
        !     
        CALL SCPEVDWF(ENB,EEL,LELECX,LVDWX,IFRSTA,NATOM,CG, &
             BNBND%JNB,BNBND%INBLO, &
             IACNBX,NITCC2,LOWTP, &
#if KEY_BLOCK==1
             IBLCKP,BLCOEE,BLCOEV,BLCOEVR,BLCOEVA, &      
#endif
             LUSED)
        IF(LUSED) THEN
           IF(PRNLEV > 6) WRITE(OUTU,125) 'SCPEVDWF'
           RETURN
        ENDIF
     ENDIF Scpismnb
#endif 
     !-----------------------------------------------------------------
     if(associated(bnbnd%jnb)) then
        tjnb => bnbnd%jnb
        tinblo => bnbnd%inblo
     else
        tjnb => pzer1
        tinblo => pzer1
     endif
     CALL ENBFS8(ENB,EEL,LELECX,LVDWX,IFRSTA,NATOM,CG, &
          tJNB,tINBLO, &
          IACNBX,NITCC2,LOWTP, &
#if KEY_BLOCK==1
          IBLCKP,BLCOEE,BLCOEV,                    & 
          BLCOEVR,BLCOEVA,                         & 
#endif
#if KEY_CHEQ==1
          eta0,.FALSE.,                            & 
#endif
#if KEY_FLUCQ==1
          QFLUC,FQCFOR,                            & 
#endif
#if KEY_WCA==1
          LLSOFT,SCVDWCUTR,WCA,                    & 
#endif
          LUSED,QGRF)    ! GRF -- Wei Chen 2015
     !-----------------------------------------------------------------------
     ! additional call for atomic multipoles (MTP module)
     IF (QMTP) THEN
        CALL MTPX(NATOM,BNBND%JNB,BNBND%INBLO)
     ENDIF
#if KEY_MTPL==1
     IF (QMTPL) THEN
        CALL MTPLX(NATOM,BNBND%JNB,BNBND%INBLO)
     ENDIF
#endif /* mtpl */

     IF (QDCM) THEN
        CALL DCME(NATOM,BNBND%INBLO,BNBND%JNB,ENB,EEL,BNBND%IBLO14, &
           BNBND%INB14,QECONTX,ECONTX,EPS,NNB14,KAPPA,ERFMOD,EWLDEX, &
           X,Y,Z,DX,DY,DZ,CTONNB,CTOFNB &
#if KEY_FLUCQ==1
           ,QFLUC,FQCFOR     &
#endif
           )
     ENDIF

     !-----------------------------------------------------------------------
     IF(LUSED) THEN
        IF(PRNLEV > 6) WRITE(OUTU,125) 'ENBFS8'
        !----------------------------------------------------------------
        !     PJ 06/2005
#if KEY_PIPF==1
        IF (QPIPF) THEN
           IF(PRNLEV > 6) WRITE(OUTU,125) 'EPIPF'
           CALL EPFDRV(IFRSTA,NATOM,BNBND%JNB, &
                BNBND%INBLO, &
                CG,MAXCN,IAC,ITC,LELECX,LCONS,LSHFT,LVSHFT,LFSWT, &
                LVFSWT,DX,DY,DZ,X,Y,Z,CTONNB,CTOFNB,EPS,E14FAC,ALP, &
                DD1,IUPT,QSECD &
                )
        ENDIF
#endif 
        !-------------------------------------------------------------------
        RETURN
     ENDIF

  ENDIF Faster_1
  !              End of Faster 1 routines
  !===================================================================
  !---  
  !---  If we get here, call the generic (slow) nonbond energy routines
  !---  in which all options are supported.
  !---  
200 CONTINUE
  call chmalloc('enbond.src','ENBOND','ioff',natc,intg=ioff)
  !--- 
  !---  QQQQ need a list of error messages for methods 
  !---           that are not supported
  !---  in the "slow" routines (e.g. GCMC, SOFTVDW,...)
  !--- ======================================================
  !---  Generic group-group list
  !--- 
  IF(LGROUP) THEN

#if KEY_MC==1
     !--- If centering is done in MC do not reallocate space
     IF (LCENTR  ==  0) THEN
        call mc_alloc_centers('enbond.src', 'ENBOND', NGRP)
     ENDIF
#else /*  MC*/
     call mc_alloc_centers('enbond.src', 'ENBOND', NGRP)
#endif /*  MC*/
     if(prnlev > 6) write(outu,125) 'egroup'
     call egroup(eel,enb,natom,nst2, &
          bnbnd%jnbg,bnbnd%inblog, &
          cg,rsclf,dx,dy,dz,ifrsta,ngrp,igpbs,igptyp, &
          bnbnd%inb14,bnbnd%iblo14, &
          lelecx,lvdwx,lcons,lshft,lfswt,ldbexp, &
          cnba,cnbb,maxcn,itc,iac,natc,ioff,x,y,z, &
          ctonnb,ctofnb,eps,e14fac,wrnmxd,qecontx,econtx, &
#if KEY_NBIPS==1
          lvips, leips,                                      & 
#endif
          qeten,qetsr,                                        & 
#if KEY_MC==1
          lcentr,                                            & 
#endif
          xcent,ycent,zcent,qcent, &
#if KEY_FLUCQ==1
          qfluc,fqcfor,                                      & 
#endif
          dd1,iupt,qsecd &
#if KEY_WCA==1
          ,llsoft, scvdwcutr, wca                             & 
#endif
#if KEY_LJPME==1
          , QLJEXC, ELJEXC  &
#endif

          ,QEXTND)

     !     QC: UW_031205 fix for fast off 
     !     so that extended is consistent with fast "on"

#if KEY_MC==1
     !        If centering was not done in MC free space
     IF (LCENTR  ==  0) THEN
        call mc_dealloc_centers('enbond.src', 'ENBOND', NGRP)
     ENDIF
#endif /*  MC*/
     GOTO 300 
  ENDIF
  !
  !=======================================================================
  ! Generic atom-atom list  (charmm, cff, and mmff versions).
  !
#if KEY_MMFF==1 || KEY_CFF==1
  if(ffield == charmm .or. ffield == amberffn) then
#endif 
     IF(PRNLEV > 6) WRITE(OUTU,125) 'EVDW'

     if(associated(bnbnd%jnb)) then
        tjnb => bnbnd%jnb
        tinblo => bnbnd%inblo
     else
        tjnb => pzer1
        tinblo => pzer1
     endif
     
     CALL EVDW(ENB,EEL,IFRSTA,NATOM,tJNB, &
          tINBLO, &
          CG,RSCLF,CNBA,CNBB,MAXCN,IAC,ITC,NATC,IOFF, &
          QETEN,QETSR,                                    & 
          LELECX,LVDWX,LCONS,LSHFT,LVSHFT,LFSWT,LVFSWT,LDBEXP, &
          DX,DY,DZ,X,Y,Z,CTONNB,CTOFNB,CGONNB,CGOFNB,EPS,E14FAC, &
#if KEY_NBIPS==1
          LVIPS,LEIPS,                               & 
#endif
          LEGROM,LVGROM,                             &
#if KEY_FLUCQ==1
          QFLUC,FQCFOR,                              & 
#endif
#if KEY_WCA==1
          LLSOFT, SCVDWCUTR, WCA,                    & 
#endif
          QECONTX,ECONTX,DD1,IUPT,QSECD &
#if KEY_IMCUBES==1
          ,lbycbim                           & 
#endif
          ,QGRF)   ! GRF -- Wei Chen 2015
     !-------------------------------------------------------------------
     ! PJ 06/2005
#if KEY_PIPF==1
     IF (QPIPF) THEN
        IF(PRNLEV > 6) WRITE(OUTU,125) 'EPIPF'
     ENDIF
     CALL EPFDRV(IFRSTA,NATOM,BNBND%JNB,BNBND%INBLO, &
          CG,MAXCN,IAC,ITC,LELECX,LCONS,LSHFT,LVSHFT,LFSWT, &
          LVFSWT,DX,DY,DZ,X,Y,Z,CTONNB,CTOFNB,EPS,E14FAC,ALP, &
          DD1,IUPT,QSECD &
          )
#endif 
     !-----------------------------------------------------------------------
     ! additional call for atomic multipoles (MTP module)
#if KEY_MTPL==1
     IF (QMTPL) THEN
        CALL MTPLX(NATOM,BNBND%JNB,BNBND%INBLO)
     ENDIF
#endif /* mtpl */

     IF (QDCM) THEN
        CALL DCME(NATOM,BNBND%INBLO,BNBND%JNB,ENB,EEL,BNBND%IBLO14, &
          BNBND%INB14,QECONTX,ECONTX,EPS,NNB14,KAPPA,ERFMOD,EWLDEX, &
          X,Y,Z,DX,DY,DZ,CTONNB,CTOFNB &
#if KEY_FLUCQ==1
           ,QFLUC,FQCFOR               &
#endif
           )
     ENDIF

     !-------------------------------------------------------------------
#if KEY_MMFF==1
  ELSEIF(FFIELD == MMFF) THEN
     !yw...02-Feb-96 EVDW_MM call is moved here from EVDW
     IF(PRNLEV > 6) WRITE(OUTU,125) 'EVDW_MM'
     CALL EVDW_MM(ENB,EEL,NATOM, &
          BNBND%JNB,BNBND%INBLO,CG,CNBA,CNBB,MAXCN, &
          IAC,ITC,NATC,IOFF, &
          LELECX,LVDWX,LCONS,LSHFT,LVSHFT,LFSWT,LVFSWT, &
          DX,DY,DZ,X,Y,Z,CTONNB,CTOFNB,EPS,E14FAC, &
          QECONTX,ECONTX,DD1,IUPT,QSECD,LVTRUNC,CTVTRN,LMSHFT, &
#if KEY_FLUCQ==1
          QFLUC,FQCFOR,    & 
#endif
          DELBUF,GAMBUF,DELQ &
#if KEY_IMCUBES==1
          ,lbycbim                           & 
#endif
          )
#endif 
#if KEY_CFF==1
  ELSEIF(FFIELD == CFF) THEN
     IF(PRNLEV > 6) WRITE(OUTU,125) 'EVDW_CFF'
     CALL EVDW_CFF(ENB,EEL,NATOM, &
          BNBND%JNB,BNBND%INBLO, &
          CG,RSCLF,CNBA,CNBB,MAXCN,IAC,ITC,NATC,IOFF, &
          LELECX,LVDWX,LCONS,LSHFT,LVSHFT,LFSWT,LVFSWT, &
          DX,DY,DZ,X,Y,Z,CTONNB,CTOFNB,EPS,E14FAC, &
#if KEY_FLUCQ==1
          QFLUC,FQCFOR,    & 
#endif
          QECONTX,ECONTX,DD1,IUPT, &
          QSECD)
#endif 
#if KEY_MMFF==1 || KEY_CFF==1
  ENDIF
#endif 
  !
  !=======================================================================
  !---       non-bond exclusion for ewald
  !
300 CONTINUE
#if KEY_DOMDEC==1
  if (q_domdec) then
#if KEY_DOMDEC_GPU==1
     if (q_gpu) call range_start('e14_domdec')
#endif
     call e14_domdec(lvdwx, lewldx, qewex, &
#if KEY_LJPME==1
                     qljpme, qljexc, eljexc, & 
#endif
     enb, eel, eelx)
#if KEY_DOMDEC_GPU==1
     if (q_gpu) call range_stop()
#endif
  else ! q_domdec
#if KEY_LJPME==1
  ! If we're computing LJPME, add in the excluded terms now
    if(QLJPME .and. QETERM(LJEXC) .and. NNB14 .gt. 0) then
        CALL LJPMEEX(NATOM,BNBND%IBLO14,BNBND%INB14,NNB14,CTONNB,CTOFNB,DX,DY,DZ,X,Y,Z)
    endif
#endif /*ljpme*/
  endif ! q_domdec
#else /*domdec*/
#if KEY_LJPME==1
  ! If we're computing LJPME, add in the excluded terms now
    if(QLJPME .and. QETERM(LJEXC) .and. NNB14 .gt. 0) then
        CALL LJPMEEX(NATOM,BNBND%IBLO14,BNBND%INB14,NNB14,CTONNB,CTOFNB,DX,DY,DZ,X,Y,Z)
    endif
#endif /*ljpme*/
#endif /*domdec*/
  IF(LEWLDX .AND. NNB14 > 0 .AND. QEWEX) THEN
     IF(PRNLEV > 6) WRITE(OUTU,125) 'EWLDEX'
     ! NOTE: for DOMDEC, we do not call ewldex. This is because the 1-4 exclusion is
     !       already taken care of in the above call to enbcalc_xfast
#if KEY_DOMDEC==1
     if (.not.q_domdec) then  
#endif
      ! NOTE: DCM module modifies CG list so needs to call ewldex locally
      if (.not.QDCM) then ! DCM
        CALL EWLDEX(EELX,NATOM, &
             BNBND%IBLO14,BNBND%INB14, &
             NNB14,CG,CTONNB,CTOFNB,EPS, &
             DX,DY,DZ,X,Y,Z,QECONTX,ECONTX &
#if KEY_FLUCQ==1
             ,QFLUC,FQCFOR     & 
#endif
             )
      endif ! DCM
#if KEY_DOMDEC==1
     endif  
#endif
  ENDIF
  !
  !=======================================================================
  !
  if(allocated(ioff))call chmdealloc('enbond.src','ENBOND','ioff',natc,intg=ioff)
  RETURN
  !
125 FORMAT(' ENBOND: Using routine ',A,' for energy calculation.')
  !
  !
  !=======================================================================
  !
END subroutine enbond

SUBROUTINE ETENSET

  !---  Note: does not support CFF or MMFF
  !---  Note: does not support IMAGE
  !---  Note: does not support GRAPE (gravity pipe machine)
  !---  Note: does not support ewald
  !---  Note: does not support soft core vdw
  !---  Note: does not support vdw shift, vdw force switch, or vdw switch
  !---  Note: does not support multi-body dynamics
  !---  Note: fully supports BLOCK
  !---  Note: fully supports "slow" energy evaluation (eg. for NMA)


  !---  Sets up ten-twelve potential. 

  !--- Author: John Karanicolas

  !---    SYNTAX:  ETEN ON
  !---      or     ETEN OFF
  !--- 
  !---   Note: Any parameter not containing "ON" 
  !---                will turn the 10-12 potential off.

  use chm_kinds
  use chm_types
  use dimens_fcm
  use comand
  use number
  use inbnd
  use stream
  use string
  use bases_fcm
  !---   use nbutil_module,only:setbnd
  !
  !
#if KEY_PARALLEL==1
  use parallel
#endif 
  implicit none

  !--- Passed variables
  !--- parse the command line info, set QETEN accordingly

  qeten=(indxa(comlyn,comlen,'ON') > 0)
  comlen = 0
  call setbnd(bnbnd)

  !--- Print whether it's being turned on or off,
  !--- provided the print level is 5 or more

  if(prnlev >= 5   &
#if KEY_PARALLEL==1
       .and.mynod == 0 & 
#endif
       ) then
     if(qeten) then
        WRITE(OUTU,'(a)') &
             'ETENSET : The 10-12-6 potential has been turned on '
        WRITE(OUTU,'(a)') &
             'ETENSET : The coefficients being used are 13, 18, 4 '
     else
        WRITE(OUTU,'(a)') &
             'ETENSET : The 10-12-6 potential has been turned off '
     endif
  endif
  return
end subroutine etenset

SUBROUTINE ETSRSET

  !---  Note: does not support CFF or MMFF
  !---  Note: does not support IMAGE
  !---  Note: does not support GRAPE (gravity pipe machine)
  !---  Note: does not support ewald
  !---  Note: does not support soft core vdw
  !---  Note: does not support vdw shift, vdw force switch, or vdw switch
  !---  Note: does not support multi-body dynamics
  !---  Note: fully supports BLOCK
  !---  Note: fully supports "slow" energy evaluation (eg. for NMA)


  !---  Sets up ten-twelve potential. 

  !--- Original author: John Karanicolas
  !--- Adapted from ETENSET by Alex Dickson

  !---    SYNTAX:  ETSR ON
  !---      or     ETSR OFF
  !--- 
  !---   Note: Any parameter not containing "ON" 
  !---                will turn the 10-12 potential off.

  use chm_kinds
  use chm_types
  use dimens_fcm
  use comand
  use number
  use inbnd
  use stream
  use string
  use bases_fcm
  !---   use nbutil_module,only:setbnd
  !
  !
#if KEY_PARALLEL==1
  use parallel
#endif 
  implicit none

  !--- Passed variables
  !--- parse the command line info, set QETSR accordingly

  qetsr=(indxa(comlyn,comlen,'ON') > 0)
  comlen = 0
  call setbnd(bnbnd)

  !--- Print whether it's being turned on or off,
  !--- provided the print level is 5 or more

  if(prnlev >= 5   &
#if KEY_PARALLEL==1
       .and.mynod == 0 & 
#endif
       ) then
     if(qetsr) then
        if (qeten) then
           WRITE(OUTU,'(a)') &
                'ETSRSET : ETSR is overriding the ETEN potential'
        endif

        WRITE(OUTU,'(a)') &
             'ETSRSET : The short range 10-12-6 potential has been turned on '
        WRITE(OUTU,'(a)') &
             'ETSRSET : The coefficients being used are 13, 18, 4 '
        WRITE(OUTU,'(a)') &
             'ETSRSET : Denominator term:  1 + (2r/3sigma)^12 '
     else
        WRITE(OUTU,'(a)') &
             'ETSRSET : The short range 10-12-6 potential has been turned off '
     endif
  endif
  return
end subroutine etsrset

end module enbond_mod

