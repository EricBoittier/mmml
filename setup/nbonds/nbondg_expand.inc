  DO IRS=1,NGRP
#if NBONDG_LEXS==1 /*lexs_test1*/
     !RCZ  25-OCT-91
#if NBONDG_L_FLAG ==1
     IF(LEXS) THEN                                       
#endif
        ITMP=IGPBS(IRS)+1
        CALL ATOMID(ITMP,SIDDN,RIDDN,RESDN,ACDN)
        IF(NEXS > 0) LTMP=QINLIST(SIDDN,SLIST,NEXS)
#if NBONDG_L_FLAG ==1
     ENDIF                                               
#endif
     !RCZ
#endif /* (lexs_test1)*/
#if KEY_NOST2 != 1
     LST2=IGPTYP(IRS) == 3
#endif 
     !
     ! set group exclusion pointers
     IF(IRS > 1) THEN
        NXI=IGLO14(IRS-1)+1
     ELSE
        NXI=1
     ENDIF
     NXIMAX=IGLO14(IRS)
     !
#if KEY_REPLICA==1 /*reptest*/
     !# <caves>-Aug-4-1993 (Leo Caves)
#if NBONDG_R_FLAG ==1
     IF (qRep) THEN                                      
#endif
        iRepNo = repNoG(iRs)
        iRepID = repID(iRepNo)
#if KEY_REPDEB==1 /*repdebug*/
        IF(qRepDB) WRITE(outU,*) 'IG:ID:No ', iRs,iRepID, iRepNo
#endif /* (repdebug)  REPDEB*/
#if NBONDG_R_FLAG ==1
     ENDIF                                               
#endif
#endif /* (reptest)*/
     !
#if KEY_PERT==1 /*pertcheck*/
#if NBONDG_P_FLAG ==1
     IF(QPERT) THEN                                      
#endif
        IF(IRS > 1) THEN
           NXIP=IGL14P(IRS-1)+1
        ELSE
           NXIP=1
        ENDIF
        NXIMXP=IGL14P(IRS)
#if NBONDG_P_FLAG ==1
     ENDIF                                               
#endif
#endif /* (pertcheck)  IF PERT*/
     !
     nl0=irs
     nl1=ngrp
     nli=1
#if KEY_PARALLEL==1 /*paragroup*/
#if KEY_PARAFULL==1 /*parfgroup*/
     IMYNOD=MOD(IRS+MYNOD,NUMNOD)
     nl0 = IRS+IMYNOD
     nl1 = NGRP
     nli = NUMNOD
#elif KEY_SPACDEC==1 /*parfgroup*/
     !       This is probably not needed when SPACDEC
#endif /* (parfgroup)*/
#endif /* (paragroup)*/
     do jrs = nl0,nl1,nli
#if KEY_PARALLEL==1 /*paragroup*/
#if KEY_VIBPARA==1
        ISDISP=0
#else /**/
#if KEY_PARAFULL==1 /*parfgroup*/
        ISDISP=0
#elif KEY_PARASCAL==1 /*parfgroup*/
        ISDISP=0
        IF(QPSRNB) THEN
           IS=IGPBS(IRS)+1
           JS=IGPBS(JRS)+1
           IF(MYNOD /= IPMAT(IPBLOCK(IS),IPBLOCK(JS))) ISDISP=-3
        ENDIF
#endif /* (parfgroup)*/
#endif 
#else /* (paragroup)*/
        ISDISP=0
#endif /* (paragroup)*/
        !
        !------------------------------------------------------------------
#if NBONDG_LEXS==1 /*lexs_test2*/
        !rcz..B960628.rcz, enabling the EXSG option
#if NBONDG_L_FLAG ==1
        IF(LEXS) THEN                                    
#endif
           JTMP=IGPBS(JRS)+1
           CALL ATOMID(JTMP,SIDDN2,RIDDN2,RESDN2,ACDN2)
           IF(PRNLEV > 7) THEN
              WRITE(OUTU,*) ' NBONDG> RCZDEBUG> IRS,ITMP=',IRS,ITMP
              WRITE(OUTU,*) ' NBONDG> RCZDEBUG> JRS,JTMP=',JRS,JTMP
              WRITE(OUTU,*) ' NBONDG> RCZDEBUG> SIDDN,SIDDN2=', &
                   SIDDN(1:idleng),SIDDN2(1:idleng)
           ENDIF
           !     nonbonded interactions between different segments will be excluded
           IF (NEXS <= 0) THEN
              IF(SIDDN /= SIDDN2) ISDISP=-2
           ELSE
              IF(LTMP.OR.QINLIST(SIDDN2,SLIST,NEXS)) ISDISP=-2
           ENDIF
#if NBONDG_L_FLAG ==1
        ENDIF                                            
#endif
        !RCZ 91/05/31
#endif /*  (lexs_test2)*/
        !------------------------------------------------------------------
        !
        !------------------------------------------------------------------
#if NBONDG_IMOVE==1 /*imove_test*/
        IF(IMOVEG(IRS) > 0 .AND. IMOVEG(JRS) > 0) ISDISP=-2
#endif /* (imove_test)*/
        !
        !------------------------------------------------------------------
#if KEY_REPLICA==1 /*repmain*/
        !# <caves>-Aug-4-1993 (Leo Caves)
        ! Replica Exclusions (Group Exclusion)
        ! Rationale: If groups belong to same subsystem (repID) but are not in same
        ! replica unit (repNoG) - then EXCLUDE the group pair
#if NBONDG_R_FLAG ==1
        IF (qRep) THEN                                   
#endif
#if KEY_REPDEB==1 /*repdb1*/
           IF (qRepDB) THEN
              WRITE(outU,*) 'JG:ID:No ',jRs, &
                   repID(repNoG(jRs)),repNoG(jRs)
           ENDIF
#endif /* (repdb1)*/
           IF ( iRepID  ==  repID(repNoG(jRs)) .AND. &
                iRepNo  /=  repNoG(jRs) )   THEN
              nRepXG = nRepXG + 1
#if KEY_REPDEB==1 /*repdb2*/
              IF (qRepDB)WRITE(outu,'(a)') &
                   ' *****EXCLUDING GROUP PAIR'
#endif /* (repdb2)*/
              ISDISP=-2
           ENDIF
#if NBONDG_R_FLAG ==1
        ENDIF                                            
#endif
#endif /* (repmain)  REPLICA*/
        !------------------------------------------------------------------
        !          QC: UW_031205: fix for QM groups
#if KEY_GAMESS==1 || KEY_QUANTUM==1 || KEY_CADPAC==1 || KEY_SCCDFTB==1 || KEY_QCHEM==1 || KEY_QTURBO==1 || KEY_G09==1
        IF (QETERM(QMEL).and.qmused) THEN
           !          skip this pair if both are QM
           IF ((LQMGRP(IRS) == 1).AND.(LQMGRP(JRS) == 1)) THEN
              ISDISP=-2
              NQMEXL=NQMEXL + 1
              IF (PRNLEV >= 7) write(outu,'(a,i8)') &
                   ' *****EXCLUDING QM PAIR',NQMEXL
           ENDIF
        ENDIF
#endif 
        !
        IF(ISDISP >= 0) THEN
           ! find distances between group centers
           XD=RSCMX(IRS)-RSCMX(JRS)
           YD=RSCMY(IRS)-RSCMY(JRS)
           ZD=RSCMZ(IRS)-RSCMZ(JRS)
#if KEY_PBOUND==1 /*pbound*/
#if NBONDG_B_FLAG ==1
           If(qBoun) then                                
#endif
              If(qCUBoun.or.qTOBoun) then
                 XD      = BOXINV * XD
                 YD      = BOYINV * YD
                 ZD      = BOZINV * ZD
                 xd = xd - nint(xd)
                 yd = yd - nint(yd)
                 zd = zd - nint(zd)
!!$                 IF(XD >  HALF) XD = XD - ONE
!!$                 IF(XD <  -HALF) XD = XD + ONE
!!$                 IF(YD >  HALF) YD = YD - ONE
!!$                 IF(YD <  -HALF) YD = YD + ONE
!!$                 IF(ZD >  HALF) ZD = ZD - ONE
!!$                 IF(ZD <  -HALF) ZD = ZD + ONE
                 If (qTOBoun) Then
                    CORR = HALF * AINT ( R75 * ( ABS( XD ) + &
                         ABS( YD ) + &
                         ABS( ZD ) ) )
                    XD      = XD    - SIGN( CORR,  XD  )
                    YD      = YD    - SIGN( CORR,  YD  )
                    ZD      = ZD    - SIGN( CORR,  ZD  )
                 Endif
                 XD      = XSIZE * XD
                 YD      = YSIZE * YD
                 ZD      = ZSIZE * ZD
              Else
                 Call PBMove(XD, YD, ZD)
              Endif
#if NBONDG_B_FLAG ==1
           Endif                                         
#endif
#endif /*  (pbound)*/
           !
           RSDISP=XD*XD+YD*YD+ZD*ZD
#if KEY_FOURD==1 /*4ddist*/
#if NBONDG_F_FLAG ==1
           IF(DIM4) THEN                                 
#endif
              FD1=MAX(ABS(FD)-RSFMAX(IRS)-RSFMAX(JRS),ZERO)
              RSDISP=RSDISP+FD1*FD1
#if NBONDG_F_FLAG ==1
           ENDIF                                         
#endif
#endif /* (4ddist)*/
           IF(RSDISP > CTNBSQ) ISDISP=-1
        ENDIF
        !
        DO WHILE(NXI <= NXIMAX .AND. JRS > ING14(NXI))
           NXI=NXI+1
        ENDDO
        IF(NXI > NXIMAX) THEN
           LEX14=IRS == JRS
        ELSE IF(JRS == ING14(NXI)) THEN
           LEX14=.TRUE.
        ELSE
           LEX14=IRS == JRS
        ENDIF
#if KEY_PERT==1 /*pertmain1*/
#if NBONDG_P_FLAG ==1
        IF(QPERT) THEN                                   
#endif
           DO WHILE(NXIP <= NXIMXP .AND. JRS > ING14P(NXIP))
              NXIP=NXIP+1
           ENDDO
           IF(NXIP > NXIMXP) THEN
              LEX14P=IRS == JRS
           ELSE IF(JRS == ING14P(NXIP)) THEN
              LEX14P=.TRUE.
           ELSE
              LEX14P=IRS == JRS
           ENDIF
#if NBONDG_P_FLAG ==1
        ENDIF                                            
#endif
#endif /* (pertmain1)  IF PERT*/
        !
        !============== GROUP - GROUP INTERACTIONS =============================
        !
        ! Setup groups list.
        !
        IF(ISDISP <= -2) THEN
           IF(ISDISP == -2) NREM=NREM+1
#if KEY_NOST2 != 1 /*st2main*/
           !    exclude pair from VDW list if both are ST2's
        ELSE IF(LST2.AND.(IGPTYP(JRS) == 3)) THEN
           IF(ISDISP >= 0) THEN
              IF(.NOT.LEX14) THEN
                 NGST2=NGST2+1
                 NNNBG=NNNBG+1
                 IF(NNNBG > MXJNBG) RETURN
                 JNBG(NNNBG)=JRS
              ENDIF
           ENDIF
#endif /* (st2main)  IFN NOST2*/
           !
#if KEY_PERT==1 /*pertmain2*/
        ELSE IF(QPERT) THEN
           IF (ISDISP >= 0) THEN

#if KEY_CHEMPERT==1
              iprtsu=igpert(irs)+igpert(jrs)    
#endif
              if( &
                   !sbcp handle chem pert   
#if KEY_CHEMPERT==1
                 ((.not.qchemp).and.(iprtsu == 0)).or. &       
#endif
#if KEY_CHEMPERT==1
                      ((     qchemp).and.(iprtsu /= 1)) )then  
#else
                 IGPERT(IRS)+IGPERT(JRS) == 0 ) then           
#endif
                 IF(LEX14.NEQV.LEX14P) CALL WRNDIE(-3,'<NBPERT>', &
                      'Bad selection of changed atoms in PERT command.')
                 NNNBG=NNNBG+1
                 IF (NNNBG > MXJNBG) RETURN
                 IF(LEX14) THEN
                    JNBG(NNNBG)=-JRS
                 ELSE
                    JNBG(NNNBG)=JRS
                 ENDIF
              ELSE
                 NNNBGP=NNNBGP+1
                 IF (NNNBGP > MXJNGP) RETURN
                 IF(LEX14) THEN
                    JNBGP(NNNBGP)=-JRS
                 ELSE
                    JNBGP(NNNBGP)=JRS
                 ENDIF
                 !
                 NNNBGR=NNNBGR+1
                 IF (NNNBGR > MXJNGR) RETURN
                 IF(LEX14P) THEN
                    JNBGR(NNNBGR)=-JRS
                 ELSE
                    JNBGR(NNNBGR)=JRS
                 ENDIF
              ENDIF
              !--------NKB, to make pert compatible with extended electrostatics -----------
           ELSE IF (RSDISP < CTEXSQ .AND. LEXTND) THEN
              ! get field and gradient for extended electrostatics for lambda=0 state
              ! in PERT
              CALL EXTGRP(IGPBS,JRS,IRS,RSCMX,RSCMY,RSCMZ, &
                   RSDX0 ,RSDY0 , &
                   RSDZ0 ,RSQXX0, &
                   RSQYY0,RSQZZ0, &
                   RSQXY0,RSQYZ0, &
                   RSQZX0,RSQ0  , &
                   RSFX0 ,RSFY0 , &
                   RSFZ0 ,RSGXX0, &
                   RSGYY0,RSGZZ0, &
                   RSGXY0,RSGYZ0, &
                   RSGZX0,RSPOT0,NGPE,LQUAD)
              ! get field and gradient for extended electrostatics for lambda=1 state
              ! in PERT
              CALL EXTGRP(IGPBS,JRS,IRS,RSCMX,RSCMY,RSCMZ,RSDX,RSDY, &
                   RSDZ,RSQXX,RSQYY,RSQZZ,RSQXY,RSQYZ,RSQZX, &
                   RSQ,RSFX,RSFY,RSFZ,RSGXX,RSGYY,RSGZZ,RSGXY, &
                   RSGYZ,RSGZX,RSPOT,NGPE,LQUAD)
              ! -------- end of addition by NKB -------------------------------
              NGPE=NGPE+1
           ELSE
              ! Excluded pair.
              NGPX=NGPX+1
           ENDIF
#endif /* (pertmain2)*/
           !
#if KEY_MTS==1
        ELSE IF(SLFG .AND. QTBMTS) THEN
           IF(ISDISP >= 0) THEN
              NNNBG=NNNBG+1
              IF (NNNBG > MXJNBG) RETURN
              IF(LEX14) THEN
                 JNBG(NNNBG)=-JRS
              ELSE
                 JNBG(NNNBG)=JRS
              ENDIF
              !
              IF(RSDISP >= RSHL2T) THEN
                 NNMT2G=NNMT2G+1
                 IF (NNMT2G > MAXJM2G) RETURN
                 IF(LEX14) THEN
                    JNM2G(NNMT2G)=-JRS
                 ELSE
                    JNM2G(NNMT2G)=JRS
                 ENDIF
              ENDIF
              !
              IF(RSDISP <= RSCUT2T) THEN
                 NNMT1G=NNMT1G+1
                 IF (NNMT1G > MAXJM1G) RETURN
                 IF(LEX14) THEN
                    JNM1G(NNMT1G)=-JRS
                 ELSE
                    JNM1G(NNMT1G)=JRS
                 ENDIF
              ENDIF
           ELSE
              ! Excluded pair.
              NGPX=NGPX+1
           ENDIF
#endif 
        ELSE IF (LEX14) THEN
           ! Close contact between groups with exclusions.
           NNNBG=NNNBG+1
           IF (NNNBG > MXJNBG) RETURN
           JNBG(NNNBG)=-JRS
           !
        ELSE IF(ISDISP >= 0) THEN
           ! Close contact between groups.
           NNNBG=NNNBG+1
           IF (NNNBG > MXJNBG) RETURN
           JNBG(NNNBG)=JRS
           !
#if KEY_NOMISC != 1 /*exelmain*/
        ELSE IF (RSDISP < CTEXSQ .AND. LEXTND) THEN
           !
           ! DO-EXTENDED-ELECTROSTATICS-PAIR
           ! If the residues are far enough apart, do only a single evaluation
           ! and use second order polynomial interpolation.
           !
           !-------------------------- NKB ---------------------------------------
           CALL EXTGRP(IGPBS,JRS,IRS,RSCMX,RSCMY,RSCMZ,RSDX,RSDY,RSDZ, &
                RSQXX,RSQYY,RSQZZ,RSQXY,RSQYZ,RSQZX,RSQ,RSFX, &
                RSFY,RSFZ,RSGXX,RSGYY,RSGZZ,RSGXY,RSGYZ,RSGZX, &
                RSPOT,NGPE,LQUAD)
           !-------------------------- NKB ---------------------------------------
           NGPE=NGPE+1
#endif /* (exelmain)  IFN NOMISC*/
        ELSE
           ! Excluded pair.
           NGPX=NGPX+1
        ENDIF
     ENDDO ! jrs
     INBLOG(IRS)=NNNBG
#if KEY_MTS==1
     IF(SLFG) THEN
        INBLM1G(IRS)=NNMT1G
        INBLM2G(IRS)=NNMT2G
     ENDIF
#endif 
#if KEY_PERT==1 /*pertgend*/
#if NBONDG_P_FLAG ==1
     IF(QPERT) THEN                                      
#endif
        INBLGR(IRS)=NNNBGR
        INBLGP(IRS)=NNNBGP
#if NBONDG_P_FLAG ==1
     ENDIF                                               
#endif
#endif /* (pertgend)*/
     !
  ENDDO    !  IRS
