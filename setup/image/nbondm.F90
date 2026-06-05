SUBROUTINE NBONDMG(X,Y,Z,MXJNBG,IMING,IMIGLO, &
     NIMNBG,IMJNBG,IMBLOG,NIMNBX,IMJNBX,IMBLOX, &
     NTRANS,NIMGRP,IMATTR,IMATPT,LIMINV,IMINV, &
     CUTNB,CMPLTD, &
#if KEY_MTS==1
     MXJNB1G,MXJNB2G, &
     NIMT1G,IMJM1G,IMBM1G,NIMT2G,IMJM2G,IMBM2G, &
     NIMT1X,IMJM1X,IMBM1X,NIMT2X,IMJM2X,IMBM2X, &
#endif /*  IF MTS*/
#if KEY_PERT==1
     IGPERT,IMINGP,IMIGLP,MXJMGP,MXJMGR, &
     NIMNGP,IMJNGP,IMBLGP,NIMNXP,IMJNXP,IMBLXP, &
     NIMNGR,IMJNGR,IMBLGR,NIMNXR,IMJNXR,IMBLXR, &
#endif /*  IF PERT*/
     RSCMX,RSCMY,RSCMZ,RSXMAX,RSYMAX,RSZMAX)
  !
  !     THIS ROUTINE CONSTRUCTS THE NONBONDED LISTS AND ACCUMULATES
  !     A SET OF ATOM POTENTIALS AND GRADIENTS FOR ELECTROSTATIC
  !     INTERACTIONS OUTSIDE OF THE CUTOFF IF REQUESTED.
  !
  !     GRADIENTS IS A FLAG THAT IS SET TO CALCULATE THE FIELD GRADIENTS.
  !
  !     By Bernard R. Brooks  22-AUG-1981
  !     Overhauled (atom lists removed) - BRB - October 25, 1996
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
#endif
  use exclm
  use psf
  use stream
  use timerm
#if KEY_PARALLEL==1
  use parallel    
#endif
#if KEY_REPLICA==1
  use replica_mod     
#endif
  use chutil,only:atomid
  use machutil,only:timrb,timre,die
  !---   use nbutil_module,only:qinlist
  implicit none
  !
  INTEGER NIMNBG,MXJNBG,NIMNBX,NTRANS,NIMGRP
  real(chm_real)  CUTNB
  logical,external :: qinlist
  !
  real(chm_real)  X(*),Y(*),Z(*)
  INTEGER IMBLOG(*),IMBLOX(*)
  INTEGER IMJNBG(*),IMJNBX(*)
  INTEGER IMING(*)
  INTEGER IMIGLO(*)
  INTEGER IMATTR(*),IMATPT(*)
  INTEGER IMINV(*)
  LOGICAL CMPLTD,LEX14,LIMINV
  LOGICAL MOVEFG,LSELF,LIMALX
  real(chm_real)  RSCMX(*),RSCMY(*),RSCMZ(*)
  real(chm_real)  RSXMAX(*),RSYMAX(*),RSZMAX(*)
  !
#if KEY_MTS==1 /*mtsdecl*/
  INTEGER MXJNB1G,MXJNB2G
  INTEGER NIMT1G,IMJM1G(*),IMBM1G(*)
  INTEGER NIMT2G,IMJM2G(*),IMBM2G(*)
  INTEGER NIMT1X,IMJM1X(*),IMBM1X(*)
  INTEGER NIMT2X,IMJM2X(*),IMBM2X(*)
#endif /* (mtsdecl)*/
  !
#if KEY_PERT==1
  INTEGER IGPERT(*)
  INTEGER IMINGP(*),IMIGLP(*)
  INTEGER MXJMGP,MXJMGR
  INTEGER NIMNGP,IMJNGP(*),IMBLGP(*)
  INTEGER NIMNXP,IMJNXP(*),IMBLXP(*)
  INTEGER NIMNGR,IMJNGR(*),IMBLGR(*)
  INTEGER NIMNXR,IMJNXR(*),IMBLXR(*)
  ! PERT local declarations
  INTEGER NXIP,NXIMXP
  LOGICAL LEX14P
#if KEY_CHEMPERT==1
  !sbcp   chem pert aux. variable
  integer iprtsu
#endif 
#endif  /* KEY_PERT */ 
  !
#if KEY_PARALLEL==1
  INTEGER IMYNOD
#endif 
  !
#if KEY_REPLICA==1 /*repdecl*/
  integer iRepNo, iRepID
#endif /* (repdecl)  REPLICA*/
  !
  INTEGER ITMP,JTMP
  LOGICAL LTMP
  !
  ! local storage
  real(chm_real) CTNBSQ,R2,RSDISP
  real(chm_real) XD,YD,ZD,XMIN,YMIN,ZMIN,XMAX,YMAX,ZMAX,XI,YI,ZI
  INTEGER NAT,NGPX,I,J,IS,IQ,ISDISP,NGRPX,NREM
  INTEGER IRS,JRS,KRS,KRSX,ITRANS
  INTEGER NXI,NXIMAX,JS,JQ,INBX,IX14
  CHARACTER(len=8) SIDDN,RIDDN,RESDN,ACDN,SIDDN2,RIDDN2,RESDN2,ACDN2
  !
#if KEY_NOST2==0 /*st2decl*/
  INTEGER NGST2
  LOGICAL LST2
#endif /* (st2decl)*/
  !
  !-----------------------------------------------------------------------
  !
  IF (TIMER.GT.0) CALL TIMRB
  !
#if KEY_REPLICA==1 /*repsetup*/
  IF (qRep) nRepXG = 0
#endif /* (repsetup)  REPLICA*/
  !
  CMPLTD=.FALSE.
  !
  CTNBSQ=CUTNB*CUTNB
  !
  ! Fill in null values for the "real" atom interactions.
  DO I=1,NGRP
     IMBLOG(I)=0
     IMBLOX(I)=0
  ENDDO
#if KEY_MTS==1
  IF (QTBMTS) THEN
     DO I=1,NGRP
        IMBM1G(I)=0
        IMBM1X(I)=0
        IMBM2G(I)=0
        IMBM2X(I)=0
     ENDDO
  ENDIF
#endif 
#if KEY_PERT==1
  IF(QPERT) THEN
     DO I=1,NGRP
        IMBLGP(I)=0
        IMBLXP(I)=0
        IMBLGR(I)=0
        IMBLXR(I)=0
     ENDDO
  ENDIF
#endif 
  !
  ! Find geometric center for each residue.
  ! The arrays for the first NGRP groups are assumed to be already
  ! correctly filled from the call to NBONDG.
  !
  DO I=NGRP+1,NIMGRP
     IS=IGPBS(I)+1
     IQ=IGPBS(I+1)
     NAT=IQ-IS+1
     IF(NAT.LE.0) CALL DIE
     XMIN=X(IS)
     XMAX=XMIN
     YMIN=Y(IS)
     YMAX=YMIN
     ZMIN=Z(IS)
     ZMAX=ZMIN
     XD=ZERO
     YD=ZERO
     ZD=ZERO
     DO J=IS,IQ
        XD=XD+X(J)
        YD=YD+Y(J)
        ZD=ZD+Z(J)
        XMIN=MIN(X(J),XMIN)
        YMIN=MIN(Y(J),YMIN)
        ZMIN=MIN(Z(J),ZMIN)
        XMAX=MAX(X(J),XMAX)
        YMAX=MAX(Y(J),YMAX)
        ZMAX=MAX(Z(J),ZMAX)
     ENDDO
     XD=XD/NAT
     YD=YD/NAT
     ZD=ZD/NAT
     !
     ! Size of rectangular box surrounding group
     RSXMAX(I)=MAX(XMAX-XD,XD-XMIN)
     RSYMAX(I)=MAX(YMAX-YD,YD-YMIN)
     RSZMAX(I)=MAX(ZMAX-ZD,ZD-ZMIN)
     ! Center of geometry of group
     RSCMX(I)=XD
     RSCMY(I)=YD
     RSCMZ(I)=ZD
  ENDDO
  !
  ! Now decide how to treat each residue pair using a rectangular
  ! search and store the disposition in rsdisp and isdisp.
  !
#if KEY_PERT==1
  IF(QPERT) THEN
     NIMNGP=0
     NIMNGR=0
     NIMNXP=0
     NIMNXR=0
  ENDIF
#endif 
  !
#if KEY_NOST2==0
  NGST2=0
#endif 
  NGPX=0
  NREM=0
  NIMNBG=0
  NIMNBX=0
  !
#if KEY_MTS==1
  IF (QTBMTS) THEN
     NIMT1G=0
     NIMT2G=0
     NIMT1X=0
     NIMT2X=0
  ENDIF
#endif 
  !
  !=======================================================================
  !  Main loop begin
  !=======================================================================
  !
  IRS=NGRP
  DO ITRANS=1,NTRANS
     !
     NGRPX=NGRP
     IF(.NOT.LIMINV) THEN
        LIMALX=.FALSE.
     ELSE IF (IMINV(ITRANS).EQ.ITRANS) THEN
        LIMALX=.FALSE.
     ELSE IF (IMINV(ITRANS).GT.ITRANS) THEN
        LIMALX=.TRUE.
     ELSE IF (IMINV(ITRANS).LT.ITRANS) THEN
        !     throw out this entire image, but first fill list arrays
        DO KRS=1,NGRP
           IS=IGPBS(IRS+1)+1
           IF(IMATTR(IS).EQ.IGPBS(KRS)+1.AND.IS.LE.IMATPT(ITRANS))THEN
              IRS=IRS+1
              !
              IMBLOG(IRS)=NIMNBG
              IMBLOX(IRS)=NIMNBX
#if KEY_MTS==1
              IF (QTBMTS) THEN
                 IMBM1G(IRS)=NIMT1G
                 IMBM2G(IRS)=NIMT2G
                 IMBM1X(IRS)=NIMT1X
                 IMBM2X(IRS)=NIMT2X
              ENDIF
#endif 
#if KEY_PERT==1
              IF(QPERT) THEN
                 IMBLGP(IRS)=NIMNGP
                 IMBLGR(IRS)=NIMNGR
                 IMBLXP(IRS)=NIMNXP
                 IMBLXR(IRS)=NIMNXR
              ENDIF
#endif 
           ENDIF
        ENDDO
        !
        NGRPX=0
     ENDIF
     !
     DO KRS=1,NGRPX
        IS=IGPBS(IRS+1)+1
        IF(IMATTR(IS).EQ.IGPBS(KRS)+1 .AND. IS.LE.IMATPT(ITRANS)) THEN
           IRS=IRS+1
           !
           IF(LEXS) THEN
              ITMP=IGPBS(KRS)+1
              CALL ATOMID(ITMP,SIDDN,RIDDN,RESDN,ACDN)
              IF(NEXS.GT.0) LTMP=QINLIST(SIDDN,SLIST,NEXS)
           ENDIF
           !
#if KEY_NOST2==0
           LST2=IGPTYP(KRS).EQ.3
#endif 
           !
           ! set up group exclusion pointers
           NXI=IMIGLO(IRS-1)+1
           NXIMAX=IMIGLO(IRS)
           !
#if KEY_REPLICA==1 /*reptest*/
           !
           IF (qRep) THEN
              iRepNo = repNoG(kRs)
              iRepID = repID(iRepNo)
#if KEY_REPDEB==1 /*repdebug*/
              IF(qRepDB) WRITE(outU,*) 'IG:ID:No ', iRs,iRepID, iRepNo
#endif /* (repdebug)  REPDEB*/
           ENDIF
#endif /* (reptest)*/
           !
#if KEY_PERT==1
           IF(QPERT) THEN
              NXIP=IMIGLP(IRS-1)+1
              NXIMXP=IMIGLP(IRS)
           ENDIF
#endif /*  IF PERT*/
           !
           IF(LIMALX) THEN
              KRSX=1
           ELSE
              KRSX=KRS
           ENDIF
           !
#if KEY_PARALLEL==1
           IMYNOD = MOD(KRSX+MYNOD,NUMNOD)
           DO JRS=KRSX+IMYNOD,NGRP,NUMNOD
#else  /* KEY_PARALLEL */
           DO JRS=KRSX,NGRP
#endif  /* KEY_PARALLEL */
              ISDISP=0
              !
              !------------------------------------------------------------------
              IF(LEXS) THEN
                 JTMP=IGPBS(JRS)+1
                 CALL ATOMID(JTMP,SIDDN2,RIDDN2,RESDN2,ACDN2)
                 !     nonbonded interactions between different segments will be excluded
                 IF (NEXS.LE.0) THEN
                    IF(SIDDN.NE.SIDDN2) ISDISP=-2
                 ELSE
                    IF(LTMP.OR.QINLIST(SIDDN2,SLIST,NEXS)) ISDISP=-2
                 ENDIF
              ENDIF
              !
              !------------------------------------------------------------------
              IF(IMOVEG(IRS).GT.0 .AND. IMOVEG(JRS).GT.0) ISDISP=-2
              !
              !------------------------------------------------------------------
#if KEY_REPLICA==1 /*repmain*/
              IF (qRep) THEN
#if KEY_REPDEB==1 /*repdb1*/
                 IF (qRepDB) THEN
                    WRITE(outU,*) 'JG:ID:No ',jRs, &
                         repID(repNoG(jRs)),repNoG(jRs)
                 ENDIF
#endif /* (repdb1)*/
                 IF ( iRepID .EQ. repID(repNoG(jRs)) .AND. &
                      iRepNo .NE. repNoG(jRs) )   THEN
                    nRepXG = nRepXG + 1
#if KEY_REPDEB==1 /*repdb2*/
                    IF (qRepDB)WRITE(outU,*)' *****EXCLUDING GROUP PAIR'
#endif /* (repdb2)*/
                    ISDISP=-2
                 ENDIF
              ENDIF
#endif /* (repmain)  REPLICA*/
              !------------------------------------------------------------------
              IF(ISDISP.GE.0) THEN
                 ! find distances between group centers
                 XD=RSCMX(IRS)-RSCMX(JRS)
                 YD=RSCMY(IRS)-RSCMY(JRS)
                 ZD=RSCMZ(IRS)-RSCMZ(JRS)
                 RSDISP=XD*XD+YD*YD+ZD*ZD
                 IF(RSDISP.GT.CTNBSQ) ISDISP=-1
              ENDIF
              !------------------------------------------------------------------
              !
              DO WHILE(NXI.LE.NXIMAX .AND. JRS.GT.IMING(NXI))
                 NXI=NXI+1
              ENDDO
              IF(NXI.GT.NXIMAX) THEN
                 LEX14=.FALSE.
              ELSE IF(JRS.EQ.IMING(NXI)) THEN
                 LEX14=.TRUE.
              ELSE
                 LEX14=.FALSE.
              ENDIF
#if KEY_PERT==1
              IF(QPERT) THEN
                 LEX14P=.FALSE.
                 DO WHILE(NXIP.LE.NXIMXP .AND. JRS.GT.IMINGP(NXIP))
                    NXIP=NXIP+1
                 ENDDO
                 IF(NXIP.GT.NXIMXP) THEN
                    LEX14P=.FALSE.
                 ELSE IF(JRS.EQ.IMINGP(NXIP)) THEN
                    LEX14P=.TRUE.
                 ELSE
                    LEX14P=.FALSE.
                 ENDIF
              ENDIF
#endif /*  IF PERT*/
              !
              !============== GROUP - GROUP INTERACTIONS =============================
              !
              ! Setup groups list
              !
              IF(ISDISP.LE.-2) THEN
                 IF(ISDISP.EQ.-2) NREM=NREM+1
#if KEY_NOST2==0
                 ! exclude pair from VDW list if both are ST2's
              ELSE IF(LST2.AND.(IGPTYP(JRS).EQ.3)) THEN
                 IF(ISDISP.GE.0) THEN
                    NGST2=NGST2+1
                    IF(KRS.NE.JRS .OR. LIMALX) THEN
                       NIMNBG=NIMNBG+1
                       IF(NIMNBG.GT.MXJNBG) RETURN
                       IMJNBG(NIMNBG)=JRS
                    ELSE
                       NIMNBX=NIMNBX+1
                       IMJNBX(NIMNBX)=JRS
                    ENDIF
                 ENDIF
#endif /*  IFN NOST2*/
                 !
#if KEY_MTS==1
                 !
                 !---------------- MTS Group List -----------------------------------
                 !
              ELSE IF (QTBMTS .AND. SLFG) THEN
                 IF (ISDISP.GE.0) THEN
                    IF (KRS.NE.JRS .OR. LIMALX) THEN
                       NIMNBG=NIMNBG+1
                       IF (NIMNBG.GT.MXJNBG) RETURN
                       IF(LEX14) THEN
                          IMJNBG(NIMNBG)=-JRS
                       ELSE
                          IMJNBG(NIMNBG)=JRS
                       ENDIF
                       ! Longe range
                       IF(RSDISP.GE.RSHL2T) THEN
                          NIMT2G=NIMT2G+1
                          IF(NIMT2G.GT.MXJNB2G) RETURN
                          IF(LEX14) THEN
                             IMJM2G(NIMT2G)=-JRS
                          ELSE
                             IMJM2G(NIMT2G)=JRS
                          ENDIF
                       ENDIF
                       ! Short range
                       IF(RSDISP.LE.RSCUT2T) THEN
                          NIMT1G=NIMT1G+1
                          IF(NIMT1G.GT.MXJNB1G) RETURN
                          IF(LEX14) THEN
                             IMJM1G(NIMT1G)=-JRS
                          ELSE
                             IMJM1G(NIMT1G)=JRS
                          ENDIF
                       ENDIF
                    ELSE
                       NIMNBX=NIMNBX+1
                       IF(LEX14) THEN
                          IMJNBX(NIMNBX)=-JRS
                       ELSE
                          IMJNBX(NIMNBX)=JRS
                       ENDIF
                       ! Longe range
                       IF(RSDISP.GE.RSHL2T) THEN
                          NIMT2X=NIMT2X+1
                          IF(LEX14) THEN
                             IMJM2X(NIMT2X)=-JRS
                          ELSE
                             IMJM2X(NIMT2X)=JRS
                          ENDIF
                       ENDIF
                       ! Short range
                       IF(RSDISP.LE.RSCUT2T) THEN
                          NIMT1X=NIMT1X+1
                          IF(LEX14) THEN
                             IMJM1X(NIMT1X)=-JRS
                          ELSE
                             IMJM1X(NIMT1X)=JRS
                          ENDIF
                       ENDIF
                    ENDIF
                    ! Excluded pair.
                    NGPX=NGPX+1
                 ENDIF
                 !
                 !------------------ End of MTS Group List -----------------------------
#endif 
                 !
#if KEY_PERT==1
              ELSE IF(QPERT) THEN
                 IF (ISDISP.GE.0) THEN
#if KEY_CHEMPERT==1
                    !sb handle chem pert
                    iprtsu=igpert(irs)+igpert(jrs)
                    if (((.not.qchemp).and.(iprtsu.eq.0))  .or. &
                         ((     qchemp).and.(iprtsu.ne.1))) then
#else /**/
                    IF(IGPERT(IRS)+IGPERT(JRS).EQ.0) THEN
#endif 
                       IF(LEX14.NEQV.LEX14P) CALL WRNDIE(-3,'<NBPERT>', &
                            'Bad selection of changed atoms in PERT command.')
                       IF (KRS.NE.JRS .OR. LIMALX) THEN
                          NIMNBG=NIMNBG+1
                          IF (NIMNBG.GT.MXJNBG) RETURN
                          IF(LEX14) THEN
                             IMJNBG(NIMNBG)=-JRS
                          ELSE
                             IMJNBG(NIMNBG)=JRS
                          ENDIF
                       ELSE
                          NIMNBX=NIMNBX+1
                          IF(LEX14) THEN
                             IMJNBX(NIMNBX)=-JRS
                          ELSE
                             IMJNBX(NIMNBX)=JRS
                          ENDIF
                       ENDIF
                    ELSE
                       IF (KRS.NE.JRS .OR. LIMALX) THEN
                          NIMNGP=NIMNGP+1
                          IF (NIMNGP.GT.MXJMGP) RETURN
                          IF(LEX14) THEN
                             IMJNGP(NIMNGP)=-JRS
                          ELSE
                             IMJNGP(NIMNGP)=JRS
                          ENDIF
                       ELSE
                          NIMNXP=NIMNXP+1
                          IF(LEX14) THEN
                             IMJNXP(NIMNXP)=-JRS
                          ELSE
                             IMJNXP(NIMNXP)=JRS
                          ENDIF
                       ENDIF
                       !
                       IF (KRS.NE.JRS .OR. LIMALX) THEN
                          NIMNGR=NIMNGR+1
                          IF (NIMNGR.GT.MXJMGR) RETURN
                          IF(LEX14P) THEN
                             IMJNGR(NIMNGR)=-JRS
                          ELSE
                             IMJNGR(NIMNGR)=JRS
                          ENDIF
                       ELSE
                          NIMNXR=NIMNXR+1
                          IF(LEX14P) THEN
                             IMJNXR(NIMNXR)=-JRS
                          ELSE
                             IMJNXR(NIMNXR)=JRS
                          ENDIF
                       ENDIF
                    ENDIF
                 ELSE
                    ! Excluded pair.
                    NGPX=NGPX+1
                 ENDIF
#endif 
                 !
              ELSE IF (LEX14) THEN
                 ! Close contact between groups with exclusions.
                 IF (KRS.NE.JRS .OR. LIMALX) THEN
                    NIMNBG=NIMNBG+1
                    IF (NIMNBG.GT.MXJNBG) RETURN
                    IMJNBG(NIMNBG)=-JRS
                 ELSE
                    NIMNBX=NIMNBX+1
                    IMJNBX(NIMNBX)=-JRS
                 ENDIF
              ELSE IF (ISDISP.GE.0) THEN
                 ! Close contact between groups.
                 IF (KRS.NE.JRS .OR. LIMALX) THEN
                    NIMNBG=NIMNBG+1
                    IF (NIMNBG.GT.MXJNBG) RETURN
                    IMJNBG(NIMNBG)=JRS
                 ELSE
                    NIMNBX=NIMNBX+1
                    IMJNBX(NIMNBX)=JRS
                 ENDIF
              ELSE
                 ! Excluded pair
                 NGPX=NGPX+1
              ENDIF
           ENDDO ! JRS
           IMBLOG(IRS)=NIMNBG
           IMBLOX(IRS)=NIMNBX
#if KEY_MTS==1
           IF (QTBMTS) THEN
              IMBM1G(IRS)=NIMT1G
              IMBM2G(IRS)=NIMT2G
              IMBM1X(IRS)=NIMT1X
              IMBM2X(IRS)=NIMT2X
           ENDIF
#endif 
#if KEY_PERT==1
           IF(QPERT) THEN
              IMBLGP(IRS)=NIMNGP
              IMBLXP(IRS)=NIMNXP
              IMBLGR(IRS)=NIMNGR
              IMBLXR(IRS)=NIMNXR
           ENDIF
#endif 
        ENDIF
     ENDDO  ! KRS
  ENDDO  ! ITRANS
  !=======================================================================
  !  Main loop end
  !=======================================================================
  !
  CMPLTD=.TRUE.
  !
  IF(PRNLEV.GE.5) THEN
     WRITE(OUTU,720)
720  FORMAT(/' Image group nonbond list generation found:')
     ! Group lists
     WRITE(OUTU,721) NIMNBG
721  FORMAT(I9,' GROUP PAIRS WERE FOUND FOR GROUP LIST')
     IF(NIMNBX.GT.0) WRITE(OUTU,722) NIMNBX
722  FORMAT(I9,' GROUP PAIRS WERE FOUND FOR GROUP SELF LIST')
#if KEY_PERT==1
     IF(QPERT) THEN
        WRITE(OUTU,723) NIMNGR
723     FORMAT(I9,' GROUP PAIRS WERE FOUND FOR REACTANT LIST')
        IF(NIMNXR.GT.0) WRITE(OUTU,724) NIMNXR
724     FORMAT(I9,' GROUP PAIRS WERE FOUND FOR REACTANT SELF LIST')
        WRITE(OUTU,725) NIMNGP
725     FORMAT(I9,' GROUP PAIRS WERE FOUND FOR PRODUCT  LIST')
        IF(NIMNXP.GT.0) WRITE(OUTU,726) NIMNXP
726     FORMAT(I9,' GROUP PAIRS WERE FOUND FOR PRODUCT  SELF LIST')
     ENDIF
#endif 
#if KEY_NOST2==0
     IF(NST2.GT.0) WRITE(OUTU,727) NGST2
727  FORMAT(I9,' OF THEM WERE ST2-ST2 INTERACTIONS')
#endif 
     WRITE(OUTU,729) NGPX
729  FORMAT(I9,' GROUP PAIRS WERE BEYOND CUTOFFS')
  ENDIF
  !
#if KEY_MTS==1
  IF (QTBMTS) THEN
     IF(PRNLEV.GT.2) THEN
        IF(TBHY1.OR.SLFG) WRITE(OUTU,4720)
4720    FORMAT(/,' Image nonbond list generation found for MTS:')
        IF(SLFG) THEN
           WRITE(OUTU,6731) NIMT1G
6731       FORMAT(I9,' GROUP PAIRS FOUND FOR SHORT RANGE LIST')
           IF(NIMT1X.GT.0) WRITE(OUTU,6732) NIMT1X
6732       FORMAT(I9,' GROUP PAIRS FOUND FOR SHORT RANGE SELF LIST')
           WRITE(OUTU,5731) NIMT2G
5731       FORMAT(I9,' GROUP PAIRS FOUND FOR LONG RANGE LIST')
           IF(NIMT2X.GT.0) WRITE(OUTU,5732) NIMT2X
5732       FORMAT(I9,' GROUP PAIRS FOUND FOR LONG RANGE SELF LIST')
        ENDIF
     ENDIF
  ENDIF
#endif 
  !
#if KEY_REPLICA==1 /*repprint*/
  !# <caves>-Aug-4-1993 (Leo Caves)
  IF (PRNLEV.GE.5.AND.qRep) THEN
     WRITE(OUTU,'(I9,A)')  nRepXG,' REPLICA GROUP PAIRS EXCLUDED'
  ENDIF ! PRNLEV
#endif /* (repprint)  REPLICA*/
  !
  IF (TIMER.EQ.1) THEN
     IF(PRNLEV.GE.2) WRITE(OUTU,130) 'TOTAL TIME IN NBONDMG'
130  FORMAT(1X,A)
     CALL TIMRE
     CALL TIMRB
  ENDIF
  !
  IF(PRNLEV.GT.8) THEN
     WRITE(OUTU,888) 'NIMNBG',NIMNBG
     WRITE(OUTU,888) 'IMBLOG',(IMBLOG(I),I=1,NGRPT)
     WRITE(OUTU,888) 'IMJNBG',(IMJNBG(I),I=1,NIMNBG)
     WRITE(OUTU,888) 'NIMNBX',NIMNBX
     WRITE(OUTU,888) 'IMBLOX',(IMBLOX(I),I=1,NGRPT)
     WRITE(OUTU,888) 'IMJNBX',(IMJNBX(I),I=1,NIMNBX)
#if KEY_PERT==1
     IF(QPERT) THEN
        WRITE(OUTU,888) 'NIMNGR',NIMNGR
        WRITE(OUTU,888) 'IMBLGR',(IMBLGR(I),I=1,NGRPT)
        WRITE(OUTU,888) 'IMJNGR',(IMJNGR(I),I=1,NIMNGR)
        WRITE(OUTU,888) 'NIMNXR',NIMNXR
        WRITE(OUTU,888) 'IMBLXR',(IMBLXR(I),I=1,NGRPT)
        WRITE(OUTU,888) 'IMJNXR',(IMJNXR(I),I=1,NIMNXR)
        WRITE(OUTU,888) 'NIMNGP',NIMNGP
        WRITE(OUTU,888) 'IMBLGP',(IMBLGP(I),I=1,NGRPT)
        WRITE(OUTU,888) 'IMJNGP',(IMJNGP(I),I=1,NIMNGP)
        WRITE(OUTU,888) 'NIMNXP',NIMNXP
        WRITE(OUTU,888) 'IMBLXP',(IMBLXP(I),I=1,NGRPT)
        WRITE(OUTU,888) 'IMJNXP',(IMJNXP(I),I=1,NIMNXP)
     ENDIF
#endif 
888  FORMAT(2X,A6,': '/,(20I5))
  ENDIF
  RETURN
END SUBROUTINE NBONDMG

SUBROUTINE NBONDMA(X,Y,Z,MXJNB,IMINB,IMIBLO, &
     NIMNB, IMJNB, IMBLO, NIMNBS,IMJNBS,IMBLOS, &
     NTRANS,NIMGRP,IMATTR,IMATPT,LIMINV,IMINV, &
     CUTNB,WRNMIN,CMPLTD, &
#if KEY_MTS==1
  MXJNB1,MXJNB2, &
       NIMMT1,IMJM1,IMBM1,NIMMT2,IMJM2,IMBM2, &
       NIMT1S,IMJM1S,IMBM1S,NIMT2S,IMJM2S,IMBM2S, &
#endif /*  IF MTS*/
#if KEY_PERT==1
  IPERT,IMINBP,IMIBLP,MXJMBP,MXJMBR, &
       NIMNBP,IMJNBP,IMBLOP,NIMNSP,IMJNSP,IMBLSP, &
       NIMNBR,IMJNBR,IMBLOR,NIMNSR,IMJNSR,IMBLSR, &
#endif /*  IF PERT*/
#if KEY_TSM==1
  LTSM,REACLS,PRODLS, &
#endif 
  RSCMX,RSCMY,RSCMZ,RSXMAX,RSYMAX,RSZMAX,RSDISP, &
  MAXNBTHOLE, NBTHOL, NBTHOLIJ, NBTHOLP, &
  NBTHOLPIMG, NBTHOL1, NBTHOL2, NBTHOL3, &
  PPNBTHOLP, PPNBTHOL1, PPNBTHOL2, PPNBTHOL3, THOLCUT)

  !
  ! THIS ROUTINE CONSTRUCTS THE IMAGE ATOM BASED NONBONDED LISTS
  !
  !     By Bernard R. Brooks  22-AUG-1981
  !     Overhauled (group lists removed) - BRB - October 25, 1996
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
#endif
  use exclm
  use psf
  use stream
  use timerm
#if KEY_PARALLEL==1
  use parallel    
#endif
#if KEY_REPLICA==1
  use replica_mod     
#endif
  use chutil,only:initia,atomid
  use machutil,only:timrb,timre,die
  !---   use nbutil_module,only:qinlist
  implicit none
  !
  INTEGER NIMNB,MXJNB,NIMNBS,NTRANS,NIMGRP
  real(chm_real)  CUTNB,WRNMIN
  logical,external :: qinlist
  !
  real(chm_real)  X(*),Y(*),Z(*)
  INTEGER IMBLO(*),IMBLOS(*)
  INTEGER IMJNB(*),IMJNBS(*)
  INTEGER IMINB(*)
  INTEGER IMIBLO(*)
  INTEGER IMATTR(*),IMATPT(*)
  INTEGER RSDISP(*)
  INTEGER IMINV(*)
  real(chm_real)  RSCMX(*),RSCMY(*),RSCMZ(*)
  real(chm_real)  RSXMAX(*),RSYMAX(*),RSZMAX(*)
! Drude thole shielding
      INTEGER MAXNBTHOLE, NBTHOL, NBTHOLIJ(2,*)
      INTEGER NBTHOLP, NBTHOLPIMG
      INTEGER NBTHOL1(*), NBTHOL2(*), NBTHOL3(*)
      INTEGER PPNBTHOL1(*), PPNBTHOL2(*), PPNBTHOL3(*), PPNBTHOLP
      real(chm_real)  THOLCUT
  !
#if KEY_MTS==1 /*mtsdecl*/
  INTEGER MXJNB1,MXJNB2
  INTEGER NIMT1S,IMJM1S(*),IMBM1S(*)
  INTEGER NIMT2S,IMJM2S(*),IMBM2S(*)
  INTEGER NIMMT1,IMJM1(*),IMBM1(*)
  INTEGER NIMMT2,IMJM2(*),IMBM2(*)
#endif /* (mtsdecl)*/
  !
#if KEY_PERT==1 /*pertdecl*/
  INTEGER IPERT(*),IMINBP(*),IMIBLP(*)
  INTEGER MXJMBP,MXJMBR
  INTEGER NIMNBP,IMJNBP(*),IMBLOP(*)
  INTEGER NIMNSP,IMJNSP(*),IMBLSP(*)
  INTEGER NIMNBR,IMJNBR(*),IMBLOR(*)
  INTEGER NIMNSR,IMJNSR(*),IMBLSR(*)
  ! PERT local declarations
  INTEGER NXIP,NXIMXP
  LOGICAL LEX14P
#if KEY_CHEMPERT==1
  !sbcp chem pert aux. variable
  integer iprtsu
  !sb
#endif 
#endif /* (pertdecl)*/
#if KEY_TSM==1
  LOGICAL LTSM
  INTEGER REACLS(*),PRODLS(*)
#endif 
  !
#if KEY_PARALLEL==1
  INTEGER IMYNOD
#endif 
  !
#if KEY_REPLICA==1 /*repdecl*/
  integer iRepNo, iRepID
#endif /* (repdecl)  REPLICA*/
  !
  INTEGER ITMP,JTMP
  LOGICAL LTMP
  !
  ! local storage
  real(chm_real) CTNBSQ,WMINSQ,R2
  real(chm_real) XD,YD,ZD,XMIN,YMIN,ZMIN,XMAX,YMAX,ZMAX, &
       XD1,YD1,ZD1,XI,YI,ZI
  LOGICAL CMPLTD,LIMINV,MOVEFG,QMOVE,LSELF,LIMALX,DOIT
  INTEGER NAT,NGAT,NGPX,I,J,IS,IQ
  INTEGER IRS,JRS,KRS,KRSX,ITRANS
  INTEGER NXI,NXIMAX,JS,JQ,INBX,IX14,IX14P,NGRPX
  CHARACTER(len=8) SIDDN,RIDDN,RESDN,ACDN,SIDDN2,RIDDN2,RESDN2,ACDN2
  !
  !-----------------------------------------------------------------------
! save number of thole pairs before counting images
  NBTHOLPIMG = NBTHOLP
  !
  IF (TIMER.GT.0) CALL TIMRB
#if KEY_REPLICA==1 /*repsetup*/
  IF (qRep) nRepXA = 0
#endif /* (repsetup)  REPLICA*/
  !
  CMPLTD=.FALSE.
  CTNBSQ=CUTNB*CUTNB
  WMINSQ=WRNMIN*WRNMIN
  QMOVE=.FALSE.
  !
  ! Fill in null values for the "real" atom interactions.
  DO I=1,NATOM
     IMBLO(I)=0
     IMBLOS(I)=0
  ENDDO
#if KEY_MTS==1
  IF (QTBMTS) THEN
     DO I=1,NATOM
        IMBM1(I)=0
        IMBM2(I)=0
        IMBM1S(I)=0
        IMBM2S(I)=0
     ENDDO
  ENDIF
#endif 
  !
#if KEY_PERT==1
  IF(QPERT) THEN
     DO I=1,NATOM
        IMBLOP(I)=0
        IMBLSP(I)=0
        IMBLOR(I)=0
        IMBLSR(I)=0
     ENDDO
  ENDIF
#endif 
  !
  ! Find geometric center for each group
  ! The arrays for the first NGRP groups are assumed to be already
  ! correctly filled from the call to NBONDG.
  !
  DO I=NGRP+1,NIMGRP
     IS=IGPBS(I)+1
     IQ=IGPBS(I+1)
     NAT=IQ-IS+1
     IF(NAT.LE.0) CALL DIE
     XMIN=X(IS)
     XMAX=XMIN
     YMIN=Y(IS)
     YMAX=YMIN
     ZMIN=Z(IS)
     ZMAX=ZMIN
     DO J=IS,IQ
        XMIN=MIN(X(J),XMIN)
        YMIN=MIN(Y(J),YMIN)
        ZMIN=MIN(Z(J),ZMIN)
        XMAX=MAX(X(J),XMAX)
        YMAX=MAX(Y(J),YMAX)
        ZMAX=MAX(Z(J),ZMAX)
     ENDDO
     XD=HALF*(XMIN+XMAX)
     YD=HALF*(YMIN+YMAX)
     ZD=HALF*(ZMIN+ZMAX)
     !
     ! Size of rectangular box surrounding group
     RSXMAX(I)=XMAX-XD
     RSYMAX(I)=YMAX-YD
     RSZMAX(I)=ZMAX-ZD
     ! Center of group defined by the box
     RSCMX(I)=XD
     RSCMY(I)=YD
     RSCMZ(I)=ZD
     IF(IMOVEG(I).GT.0) QMOVE=.TRUE.
  ENDDO
  !
  ! Now decide how to treat each residue pair using a rectangular
  ! search and store the disposition in rsdisp.
  !
  NIMNB=0
  NIMNBS=0
  !
#if KEY_MTS==1
  IF (QTBMTS) THEN
     NIMMT1=0
     NIMMT2=0
     NIMT1S=0
     NIMT2S=0
  ENDIF
#endif 
  !
#if KEY_PERT==1
  IF(QPERT) THEN
     NIMNBR=0
     NIMNBP=0
     NIMNSR=0
     NIMNSP=0
  ENDIF
#endif 
  NGAT=0
  NGPX=0
  IRS=NGRP
  !
  !=======================================================================
  !  Expand control section
  !-------------------------------------------------------------------
#if KEY_EXPAND == 1
#define NBONDM_EXPAND 1
#endif
  ! (disable expand when debug is active)
#if KEY_DEBUG
#undef NBONDM_EXPAND
#endif
  !-------------------------------------------------------------------
  ! Do PERT expansion of code                
! ##EXPAND P nopert       .when. PERT  EXPAND  (expand_pert)
#if NBONDM_EXPAND == 1 && KEY_PERT == 1

#define NBONDM_PERT 1

! ##PASS1  .not.EXPAND
  IF(QPERT) THEN

#undef NBONDM_EXPAND
#include "nbondm1.inc"
#define NBONDM_EXPAND 1

! ##PASS2  .not.PERT  nopert
  ELSE

#undef NBONDM_PERT
#define NBONDM_NOPERT 1
#include "nbondm1.inc"
#define NBONDM_PERT 1
#undef NBONDM_NOPERT

! ##EXFIN
  ENDIF
! ##EXEND

#else  /* NBONDM_EXPAND == 1 */

#undef NBONDM_PERT
#if KEY_PERT == 1
#define NBONDM_PERT 1
#endif

#define NBONDM_PGUARD 1
#define NBONDM_NOPERT 1
#include "nbondm1.inc"
#undef NBONDM_PGUARD
#undef NBONDM_NOPERT

#endif  /* NBONDM_EXPAND == 1 */
! ##ENDEX    (expand_pert)
  !=======================================================================
  !
245 FORMAT(' WARNING: ATOMS',4(1X,A), &
         ' AND',4(1X,A),' ONLY',F5.2,' A. APART')
  !
  CMPLTD=.TRUE.
  !
  IF(PRNLEV.GE.5) THEN
     WRITE(OUTU,720)
720  FORMAT(/' Image nonbond list generation found:')
     ! Atom lists
     WRITE(OUTU,731) NIMNB
731  FORMAT(I9,' ATOM PAIRS WERE FOUND FOR ATOM LIST')
     WRITE(OUTU,732) NIMNBS
732  FORMAT(I9,' ATOM PAIRS WERE FOUND FOR ATOM SELF LIST')
     IF(NBTHOLP-NBTHOLPIMG.GT.0) THEN
         write(outu,'(i9,a)') NBTHOLP-NBTHOLPIMG, &
                      ' IMAGE THOLE SHIELDING PAIRS FOUND '
     ENDIF
#if KEY_PERT==1
     IF(QPERT) THEN
        WRITE(OUTU,734) NIMNBR
734     FORMAT(I9,' ATOM PAIRS WERE FOUND FOR REACTANT LIST')
        IF(NIMNSR.GT.0) WRITE(OUTU,735) NIMNSR
735     FORMAT(I9,' ATOM PAIRS WERE FOUND FOR REACTANT SELF LIST')
        WRITE(OUTU,736) NIMNBP
736     FORMAT(I9,' ATOM PAIRS WERE FOUND FOR PRODUCT  LIST')
        IF(NIMNSP.GT.0) WRITE(OUTU,737) NIMNSP
737     FORMAT(I9,' ATOM PAIRS WERE FOUND FOR PRODUCT  SELF LIST')
     ENDIF
#endif 
     WRITE(OUTU,739) NGAT
739  FORMAT(I9,' GROUP PAIRS REQUIRED ATOM SEARCHES'/)
  ENDIF
  !
#if KEY_MTS==1
  IF (QTBMTS) THEN
     IF(PRNLEV.GT.2) THEN
        IF(TBHY1.OR.SLFG) WRITE(OUTU,4720)
4720    FORMAT(/,' Image nonbond list generation found for MTS:')
        IF(TBHY1) THEN
           WRITE(OUTU,1731) NIMMT1
1731       FORMAT(I9,' ATOM PAIRS WERE FOUND FOR FAST MODE LIST')
           IF(NIMT1S.GT.0) WRITE(OUTU,1732) NIMT1S
1732       FORMAT(I9,' ATOM PAIRS WERE FOUND FOR FAST SELF LIST')
           WRITE(OUTU,2731) NIMMT2
2731       FORMAT(I9,' ATOM PAIRS WERE FOUND FOR SLOW MODE LIST')
           IF(NIMT2S.GT.0) WRITE(OUTU,2732) NIMT2S
2732       FORMAT(I9,' ATOM PAIRS WERE FOUND FOR SLOW SELF LIST')
        ENDIF
        IF(SLFG) THEN
           WRITE(OUTU,3731) NIMMT1
3731       FORMAT(I9,' ATOM PAIRS FOUND FOR SHORT RANGE LIST')
           IF(NIMT1S.GT.0) WRITE(OUTU,3732) NIMT1S
3732       FORMAT(I9,' ATOM PAIRS FOUND FOR SHORT RANGE SELF LIST')
           WRITE(OUTU,4731) NIMMT2
4731       FORMAT(I9,' ATOM PAIRS FOUND FOR LONG RANGE LIST')
           IF(NIMT1S.GT.0) WRITE(OUTU,4732) NIMT2S
4732       FORMAT(I9,' ATOM PAIRS FOUND FOR LONG RANGE SELF LIST')
        ENDIF
     ENDIF
  ENDIF
#endif   /* KEY_MTS */
  !
#if KEY_REPLICA==1 /*repprint*/
  !# <caves>-Aug-4-1993 (Leo Caves)
  IF (PRNLEV.GE.5.AND.qRep) THEN
     WRITE(OUTU,'(I9,A/)') nRepXA,' REPLICA ATOM  PAIRS EXCLUDED'
  ENDIF ! PRNLEV
#endif /* (repprint)  REPLICA*/
  !
  IF (TIMER.EQ.1) THEN
     IF(PRNLEV.GE.2) WRITE(OUTU,130) 'TOTAL TIME IN NBONDMA'
130  FORMAT(1X,A)
     CALL TIMRE
     CALL TIMRB
  ENDIF
  !
  IF(PRNLEV.GT.8) THEN
     WRITE(OUTU,888) 'NIMNB ',NIMNB
     WRITE(OUTU,888) 'IMBLO ',(IMBLO(I),I=1,NATOMT)
     WRITE(OUTU,888) 'IMJNB ',(IMJNB(I),I=1,NIMNB)
     WRITE(OUTU,888) 'NIMNBS',NIMNBS
     WRITE(OUTU,888) 'IMBLOS',(IMBLOS(I),I=1,NATOMT)
     WRITE(OUTU,888) 'IMJNBS',(IMJNBS(I),I=1,NIMNBS)
#if KEY_MTS==1
     IF(TBHY1.OR.SLFG) THEN
        WRITE(OUTU,888) 'NIMMT1 ',NIMMT1
        WRITE(OUTU,888) 'IMBM1  ',(IMBM1(I),I=1,NATOMT)
        WRITE(OUTU,888) 'IMJM1  ',(IMJM1(I),I=1,NIMMT1)
        WRITE(OUTU,888) 'NIMT1S ',NIMT1S
        WRITE(OUTU,888) 'IMBM1S ',(IMBM1S(I),I=1,NATOMT)
        WRITE(OUTU,888) 'IMJM1S ',(IMJM1S(I),I=1,NIMT1S)
        WRITE(OUTU,888) 'NIMMT2 ',NIMMT2
        WRITE(OUTU,888) 'IMBM2  ',(IMBM2(I),I=1,NATOMT)
        WRITE(OUTU,888) 'IMJM2  ',(IMJM2(I),I=1,NIMMT2)
        WRITE(OUTU,888) 'NIMT2S ',NIMT2S
        WRITE(OUTU,888) 'IMBM2S ',(IMBM2S(I),I=1,NATOMT)
        WRITE(OUTU,888) 'IMJM2S ',(IMJM2S(I),I=1,NIMT2S)
     ENDIF
#endif 
     !
#if KEY_PERT==1
     IF(QPERT) THEN
        WRITE(OUTU,888) 'NIMNBR',NIMNBR
        WRITE(OUTU,888) 'IMBLOR',(IMBLOR(I),I=1,NATOMT)
        WRITE(OUTU,888) 'IMJNBR',(IMJNBR(I),I=1,NIMNBR)
        WRITE(OUTU,888) 'NIMNSR',NIMNSR
        WRITE(OUTU,888) 'IMBLSR',(IMBLSR(I),I=1,NATOMT)
        WRITE(OUTU,888) 'IMJNSR',(IMJNSR(I),I=1,NIMNSR)
        WRITE(OUTU,888) 'NIMNBP',NIMNBP
        WRITE(OUTU,888) 'IMBLOP',(IMBLOP(I),I=1,NATOMT)
        WRITE(OUTU,888) 'IMJNBP',(IMJNBP(I),I=1,NIMNBP)
        WRITE(OUTU,888) 'NIMNSP',NIMNSP
        WRITE(OUTU,888) 'IMBLSP',(IMBLSP(I),I=1,NATOMT)
        WRITE(OUTU,888) 'IMJNSP',(IMJNSP(I),I=1,NIMNSP)
     ENDIF
#endif 
888  FORMAT(2X,A6,': '/,(20I5))
  ENDIF
  RETURN
END SUBROUTINE NBONDMA 
