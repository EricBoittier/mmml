  DO IRS=1,NGRP
#if NBONDA_LEXS == 1 /*lexs1*/
     !RCZ  25-OCT-91
#if NBONDA_L_FLAG == 1
     IF(LEXS) THEN
#endif
        ITMP=IGPBS(IRS)+1
        CALL ATOMID(ITMP,SIDDN,RIDDN,RESDN,ACDN)
        IF(NEXS > 0) LTMP=QINLIST(SIDDN,SLIST,NEXS)
#if NBONDA_L_FLAG == 1
     ENDIF
#endif
     !RCZ
#endif /* (lexs1)*/
     !
#if KEY_REPLICA == 1 /*reptest*/
     !# <caves>-Aug-4-1993 (Leo Caves)
#if NBONDA_R_FLAG == 1
     IF (qRep) THEN
#endif
        iRepNo = repNoG(iRs)
        iRepID = repID(iRepNo)
#if KEY_REPDEB == 1 /*repdebug*/
        IF(qRepDB) WRITE(outU,*) 'IG:ID:No ', iRs,iRepID, iRepNo
#endif /* (repdebug)  REPDEB*/
#if NBONDA_R_FLAG == 1
     ENDIF
#endif
#endif /* (reptest)*/
     !
     !
     nl0=irs
     nl1=ngrp
     nli=1
#if KEY_PARALLEL == 1 /*paragroup*/
#if KEY_PARAFULL == 1 /*parfgroup*/
     IMYNOD=MOD(IRS+MYNOD,NUMNOD)
     nl0 = IRS+IMYNOD
     nl1 = NGRP
     nli = NUMNOD
#elif KEY_SPACDEC == 1 /*parfgroup*/
     !       This is probably not used with SPACDEC
     !
#endif /* (parfgroup)*/
#endif /* (paragroup)*/
     do jrs=nl0,nl1,nli
#if KEY_PARALLEL == 1 /*paragroup*/
#if KEY_VIBPARA == 1
        RSDISP(JRS)=0
#else /**/
#if KEY_PARAFULL == 1 /*parfgroup*/
        RSDISP(JRS)=0
#elif KEY_PARASCAL == 1 /*parfgroup*/
        RSDISP(JRS)=0
        IF(QPSRNB) THEN
           IS=IGPBS(IRS)+1
           JS=IGPBS(JRS)+1
           IF(MYNOD /= IPMAT(IPBLOCK(IS),IPBLOCK(JS))) RSDISP(JRS)=-3
        ENDIF
#endif /* (parfgroup)*/
#endif
#else /* (paragroup)*/
        RSDISP(JRS)=0
#endif /* (paragroup)*/
        !
        !rcz..B960628.rcz, enabling the EXSG option
#if NBONDA_LEXS == 1 /*lexs2*/
#if NBONDA_L_FLAG == 1
        IF(LEXS) THEN
#endif
           JTMP=IGPBS(JRS)+1
           CALL ATOMID(JTMP,SIDDN2,RIDDN2,RESDN2,ACDN2)
           IF(PRNLEV > 7) THEN
              WRITE(OUTU,*) ' NBONDA> RCZDEBUG> IRS,ITMP=',IRS,ITMP
              WRITE(OUTU,*) ' NBONDA> RCZDEBUG> JRS,JTMP=',JRS,JTMP
              WRITE(OUTU,*) ' NBONDA> RCZDEBUG> SIDDN,SIDDN2=', &
                   SIDDN(1:idleng),SIDDN2(1:idleng)
           ENDIF
           !     nonbonded interactions between different segments will be excluded
           IF (NEXS <= 0) THEN
              IF(SIDDN /= SIDDN2) RSDISP(JRS)=-2
           ELSE
              IF(LTMP.OR.QINLIST(SIDDN2,SLIST,NEXS)) RSDISP(JRS)=-2
           ENDIF
#if NBONDA_L_FLAG == 1
        ENDIF
#endif
        !RCZ 91/05/31
#endif /* (lexs2)*/
        !
        !------------------------------------------------------------------
#if NBONDA_IMOVE == 1
        IF(IMOVEG(IRS) > 0 .AND. IMOVEG(JRS).GT.0) RSDISP(JRS)=-2
#endif
        !
        !------------------------------------------------------------------
#if KEY_REPLICA == 1 /*repmain*/
        !# <caves>-Aug-4-1993 (Leo Caves)
        ! Replica Exclusions (Group Exclusion)
        ! Rationale: If groups belong to same subsystem (repID) but are not in same
        ! replica unit (repNoG) - then EXCLUDE the group pair
#if NBONDA_R_FLAG == 1
        IF (qRep) THEN
#endif
#if KEY_REPDEB == 1 /*repdb1*/
           IF (qRepDB) THEN
              WRITE(outU,*) 'JG:ID:No ',jRs, &
                   repID(repNoG(jRs)),repNoG(jRs)
           ENDIF
#endif /* (repdb1)*/
           IF ( iRepID  ==  repID(repNoG(jRs)) .AND. &
                iRepNo  /=  repNoG(jRs) )   THEN
              nRepXG = nRepXG + 1
#if KEY_REPDEB == 1 /*repdb2*/
              IF (qRepDB)WRITE(outU,*)' *****EXCLUDING GROUP PAIR'
#endif /* (repdb2)*/
              RSDISP(JRS)=-2
           ENDIF
#if NBONDA_R_FLAG == 1
        ENDIF
#endif
#endif /* (repmain)  REPLICA*/
        !------------------------------------------------------------------
        !
        IF(RSDISP(JRS) >= 0) THEN
           ! find distances between group centers
           XD=RSCMX(IRS)-RSCMX(JRS)
           YD=RSCMY(IRS)-RSCMY(JRS)
           ZD=RSCMZ(IRS)-RSCMZ(JRS)
#if KEY_PBOUND == 1 /*pbound*/
#if NBONDA_B_FLAG == 1
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
#if NBONDA_B_FLAG == 1
           Endif
#endif
#endif /*      (pbound)*/
           !
           XD1=MAX(ABS(XD)-RSXMAX(IRS)-RSXMAX(JRS),ZERO)
           YD1=MAX(ABS(YD)-RSYMAX(IRS)-RSYMAX(JRS),ZERO)
           ZD1=MAX(ABS(ZD)-RSZMAX(IRS)-RSZMAX(JRS),ZERO)
           R2=XD1*XD1+YD1*YD1+ZD1*ZD1
#if KEY_FOURD == 1 /*4ddist*/
#if NBONDA_F_FLAG== 1
           IF(DIM4) THEN
#endif
              FD=RSCMF(IRS)-RSCMF(JRS)
              FD1=MAX(ABS(FD)-RSFMAX(IRS)-RSFMAX(JRS),ZERO)
              R2=R2+FD1*FD1
#if NBONDA_F_FLAG== 1
           ENDIF
#endif
#endif /* (4ddist)*/
           IF(R2 > CTNBSQ) THEN
#if KEY_DEBUG == 1
              if(prnlev >= 7) write(outu,447) irs,jrs,sqrt(r2)
447           format(' NBONDA: Rejecting group pair irs,jrs,r:',2I5,F10.5)
#endif
              RSDISP(JRS)=-1
           ELSE
              NGAT=NGAT+1
#if KEY_MTS == 1 /*mtsnoclose*/
#if NBONDA_M_FLAG== 1
              IF (.NOT. QTBMTS) THEN
#endif
#endif /*  (mtsnoclose)*/
#if NBONDA_NOMTS == 1 /*nomts1*/
                 XD1=ABS(XD)+RSXMAX(IRS)+RSXMAX(JRS)
                 YD1=ABS(YD)+RSYMAX(IRS)+RSYMAX(JRS)
                 ZD1=ABS(ZD)+RSZMAX(IRS)+RSZMAX(JRS)
                 R2=XD1*XD1+YD1*YD1+ZD1*ZD1
#if KEY_FOURD == 1 /*4ddist2*/
#if NBONDA_F_FLAG== 1
                 IF(DIM4) THEN
#endif
                    FD1=ABS(FD)+RSFMAX(IRS)+RSFMAX(JRS)
                    R2=R2+FD1*FD1
#if NBONDA_F_FLAG== 1
                 ENDIF
#endif
#endif /* (4ddist2)*/
                 IF(R2 < CTNBSQ) THEN
                    RSDISP(JRS)=1
                    NGAT=NGAT-1
                 ENDIF
#endif /*  (nomts1)  nomts nombmts*/
#if KEY_MTS == 1 /*mtsnoclose*/
#if NBONDA_M_FLAG== 1
              ENDIF
#endif
#endif /*  (mtsnoclose)*/
           ENDIF
        ENDIF  ! (RSDISP(JRS) >= 0)
     ENDDO   ! JRS
     !
     !==============  ATOM - ATOM  INTERACTIONS =============================
     !
     ! Construct the jnb array searching only close residue
     ! pairs (RSDISP>=0)
     !
     IS=IGPBS(IRS)+1
     IQ=IGPBS(IRS+1)
     DO I=IS,IQ
        IF (I > 1) THEN
           NXI=IBLO14(I-1)+1
        ELSE
           NXI=1
        ENDIF
        NXIMAX=IBLO14(I)
#if KEY_PERT == 1 /*pertatom1*/
#if NBONDA_P_FLAG == 1
        IF(QPERT) THEN
#endif
           IF (I > 1) THEN
              NXIP=IBL14P(I-1)+1
           ELSE
              NXIP=1
           ENDIF
           NXIMXP=IBL14P(I)
#if NBONDA_P_FLAG == 1
        ENDIF
#endif
#endif /* (pertatom1)*/
        XI=X(I)
        YI=Y(I)
        ZI=Z(I)
#if KEY_FOURD == 1
#if NBONDA_F_FLAG== 1
        IF(DIM4) THEN
#endif
           FDI=FDIM(I)
#if NBONDA_F_FLAG== 1
        ENDIF
#endif
#endif
#if NBONDA_IMOVE == 1
        MOVEFG=IMOVE(I) > 0
#endif
#if KEY_REPLICA == 1 /*repatom1*/
        !# <caves>Aug-4-1993 (Leo Caves)
#if NBONDA_R_FLAG == 1
        IF (qRep) THEN
#endif
           iRepNo = repNoA(i)
           iRepID = repID(iRepNo)
#if KEY_REPDEB == 1 /*repatomdeb*/
           IF(qRepDB) WRITE(outU,*) 'IA:ID:No ',i,iRepID, iRepNo
#endif /* (repatomdeb)*/
#if NBONDA_R_FLAG == 1
        ENDIF
#endif
#endif /* (repatom1)  REPLICA*/
        !
        nl0=irs
        nl1=ngrp
        nli=1
#if KEY_PARALLEL == 1 /*paraatom*/
#if KEY_PARAFULL == 1 /*parfatom*/
        IMYNOD=MOD(IRS+MYNOD,NUMNOD)
        nl0 = IRS+IMYNOD
        nli = NUMNOD
#endif /* (parfatom)*/
#endif /* (paraatom)*/
        DO JRS=nl0,nl1,nli
           IF(RSDISP(JRS) >= 0) THEN
              JS=IGPBS(JRS)+1
              JQ=IGPBS(JRS+1)
              IF(I >= JS) JS=I+1
              DO J=JS,JQ
                 !
#if KEY_DEBUG == 1
                 if(prnlev >= 7) write(outu,457) irs,jrs,rsdisp(jrs),i,j
457              format(' NBONDA: Examining atom pair irs,jrs,rsdisp,i,j:',5I5)
#endif
                 !
                 !=======================================================================
                 ! Check to see if this atom pair needs to be examined.
                 !
#if NBONDA_IMOVE == 1
                 DOIT=.NOT.(MOVEFG.AND.(IMOVE(J) > 0))
#else
                 DOIT=.TRUE.
#endif
                 !-----------------------------------------------------------------------
#if KEY_REPLICA == 1 /*repatom2*/
                 !# <caves>-Aug-4-1993 (Leo Caves)
                 ! Replica Exclusions (Atom Exclusion)
                 ! Rationale: If atoms belong to same subsystem (repID) but are not in
                 ! same replica unit (repNoA) - then EXCLUDE the atom pair
                 ! This check is probably not necessary, due to the nature of the
                 ! replication process.
                 !
#if NBONDA_R_FLAG == 1
                 IF (qRep) THEN
#endif
#if KEY_REPDEB == 1
                    IF (qRepDB) WRITE(outU,*) 'JA:ID:No ',j,            &
                         repID(repNoA(j)),repNoA(j)
#endif
                    IF ( iRepID  ==  repID(repNoA(j)) .AND. &
                         iRepNo  /=  repNoA(j) )   THEN
                       nRepXA = nRepXA + 1
#if KEY_REPDEB == 1
                       IF (qRepDB) WRITE(outU,*)                        &
                            ' *****EXCLUDING ATOM PAIR'
#endif
                       DOIT=.FALSE.
                    ENDIF
#if NBONDA_R_FLAG == 1
                 ENDIF
#endif
#endif /* (repatom2)  REPLICA*/
                 !
                 !-----------------------------------------------------------------------
#if KEY_TSM == 1 /*tsmatom*/
#if NBONDA_T_FLAG == 1
                 IF(LTSM) THEN
#endif
                    IF(REACLS(I) == 1 .AND.PRODLS(J).EQ.1.OR. &
                         REACLS(J) == 1 .AND.PRODLS(I).EQ.1) DOIT=.FALSE.
#if NBONDA_T_FLAG == 1
                 ENDIF
#endif
#endif /* (tsmatom)*/

#if NBONDA_QMMM == 1
                 !-----------------------------------------------------------------------
! ##IF GAMESS GAMESSUK QUANTUM CADPAC SCCDFTB QCHEM MNDO97 SQUANTM QTURBO G09
                 IF(QGMREM) THEN
                    !C OLD                    IF(IGMSEL(I) > 0 .AND. IGMSEL(J).GT.0) DOIT=.FALSE.
                    IF(((ABS(IGMSEL(I)) == 1).OR.(ABS(IGMSEL(I)).EQ.2)) &
                         .AND.((ABS(IGMSEL(J)) == 1).OR.(ABS(IGMSEL(J)).EQ.2))) &
                         DOIT=.FALSE.
                 ENDIF
! ##ENDIF
                 !-----------------------------------------------------------------------
#endif /* NBONDA_QMMM */

                 IF(DOIT) THEN
                    !
                    ! See if pair is in excluded list
                    !
                    INBX=IABS(INB14(NXI))
                    DO WHILE(NXI <= NXIMAX .AND. J > INBX)
                       NXI=NXI+1
                       INBX=IABS(INB14(NXI))
                    ENDDO
                    IF(NXI > NXIMAX) THEN
                       IX14=J
                    ELSE IF(J == INB14(NXI)) THEN  ! exclusion found
                       IX14=0
                    ELSE
                       IF(J == INBX) THEN
                          IX14=-J    ! it's a 1-4 pair
                       ELSE
                          IX14=J
                       ENDIF
                    ENDIF
#if KEY_PERT == 1 /*pertatomx*/
                    ! See if pair is in perturbed excluded list
#if NBONDA_P_FLAG == 1
                    IF(QPERT) THEN
#endif
                       !sbcp handle chem pert
#if KEY_CHEMPERT == 1
                       iprtsu=ipert(i)+ipert(j)
#endif
                       if ( &
#if KEY_CHEMPERT == 1
                            ((.not.qchemp).and.(iprtsu /= 0)).or. &
                            ((     qchemp).and.(iprtsu == 1)) &
#else
                            IPERT(I)+IPERT(J) /= 0 &
#endif
                            ) then
                          INBX=IABS(INB14P(NXIP))
                          DO WHILE(NXIP <= NXIMXP .AND. J > INBX)
                             NXIP=NXIP+1
                             INBX=IABS(INB14P(NXIP))
                          ENDDO
                          IF(NXIP > NXIMXP) THEN
                             IX14P=J
                          ELSE IF(J == INB14P(NXIP)) THEN  ! exclusion found
                             IX14P=0
                          ELSE
                             IF(J == INBX) THEN
                                IX14P=-J    ! it's a 1-4 pair
                             ELSE
                                IX14P=J
                             ENDIF
                          ENDIF
                       ELSE
                          IX14P=0
                       ENDIF
                       IF(IX14 == 0 .AND. IX14P.EQ.0) DOIT=.FALSE.
#if NBONDA_P_FLAG == 1
                    ELSE
                       IF(IX14 == 0) DOIT=.FALSE.
                    ENDIF
#endif
#else /*  (pertatomx)*/
#if NBONDA_NOPERT == 1
                    IF(IX14 == 0) DOIT=.FALSE.
#endif
#endif /* (pertatomx)  IF PERT*/

                    !-----------------------------------------------------------------------
                    IF(RSDISP(JRS) == 0 .AND. DOIT) THEN
                       !                      compute the atom-atom distance
                       XD=XI-X(J)
                       YD=YI-Y(J)
                       ZD=ZI-Z(J)
#if KEY_PBOUND == 1 /*pbound*/
#if NBONDA_B_FLAG == 1
                       If(qBoun) then
#endif
                          If(qCUBoun.or.qTOBoun) then
                             XD      = BOXINV * XD
                             YD      = BOYINV * YD
                             ZD      = BOZINV * ZD
                             XD      = XD    - INT(XD + SIGN(HALF,XD))
                             YD      = YD    - INT(YD + SIGN(HALF,YD))
                             ZD      = ZD    - INT(ZD + SIGN(HALF,ZD))
                             If (qTOBoun) Then
                                CORR = HALF*AINT(R75*( ABS( XD ) + &
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
#if NBONDA_B_FLAG == 1
                       Endif
#endif
#endif /*   (pbound)*/
                       R2=XD*XD+YD*YD+ZD*ZD
#if KEY_FOURD == 1 /*4dadd*/
#if NBONDA_F_FLAG== 1
                       IF(DIM4) THEN
#endif
                          FD=FDI-FDIM(J)
                          R2=R2+FD*FD
#if NBONDA_F_FLAG== 1
                       ENDIF
#endif
#endif /* (4dadd)*/
                       IF (R2 <= CTNBSQ) THEN
                          IF(R2 < WMINSQ) THEN
#if KEY_FOURD == 1
#if NBONDA_F_FLAG== 1
                             IF(.NOT.DIM4) THEN
#endif
#endif
#if NBONDA_NOFOURD == 1
                                IF(WRNLEV >= 3) THEN
                                   !                          write information
                                   CALL ATOMID(I,SIDDN,RIDDN,RESDN,ACDN)
                                   CALL ATOMID(J,SIDDN2,RIDDN2,RESDN2,ACDN2)
                                   WRITE(OUTU,245) SIDDN(1:idleng), &
                                        RIDDN(1:idleng),RESDN(1:idleng), &
                                        ACDN(1:idleng), &
                                        SIDDN2(1:idleng),RIDDN2(1:idleng), &
                                        RESDN2(1:idleng),ACDN2(1:idleng), &
                                        SQRT(R2)
                                   IF(.NOT.(INITIA(I,X,Y,Z).AND. &
                                        INITIA(J,X,Y,Z)))THEN
                                      CALL WRNDIE(-2,'<NBONDG>', &
                                           'Some atoms have undefined positions')
                                      CMPLTD=.TRUE.
                                      RETURN
                                   ENDIF
                                ENDIF
#endif /*    nofourd*/
#if KEY_FOURD == 1
#if NBONDA_F_FLAG== 1
                             ENDIF
#endif
#endif
                          ENDIF
#if KEY_DEBUG == 1
                          if(prnlev >= 7) write(outu,456) i,j,sqrt(r2)
456                       format('         Distance is OK for atom pair          :', &
                               15                           x,2I5,F12.4)
#endif
                       ELSE
                          DOIT=.FALSE.
#if KEY_DEBUG == 1
                          if(prnlev >= 7) write(outu,458) i,j,sqrt(r2)
458                       format('         Rejecting atom pair due to distance   :', &
                               15                           x,2I5,F12.4)
#endif
                       ENDIF
                    ELSE
#if KEY_DEBUG == 1
                       if(prnlev >= 7 .and. .not.DOIT) write(outu,461) i,j
461                    format('         Rejecting atom pair not by distance   :', &
                            15                        x,2I5,F12.4)
#endif
                    ENDIF  ! (RSDISP(JRS) == 0 .AND. DOIT)
                    !-----------------------------------------------------------------------
                    IF(DOIT) THEN

#if KEY_DEBUG == 1
                       if(prnlev >= 7) write(outu,459) i,j
459                    format('         Accepting atom pair                   :', &
                            15x,2I5)
#endif

                          if(.not.(isdrude(i).or.isdrude(j))) then

                             R2=((x(i)-x(j))**2+(y(i)-y(j))**2+(z(i)-z(j))**2)

                             if(r2.lt.tholcut) then

                                call MKTHOLELIST(i,j,IAC(I),IAC(J), &
                                     MAXNBTHOLE, NBTHOL, NBTHOLIJ, NBTHOLP, &
                                     NBTHOL1, NBTHOL2, NBTHOL3, R2)
#if KEY_PERT == 1
                                IF(QPERT) THEN
                                   call MKTHOLELIST(i,j,PPIAC(I),PPIAC(J), &
                                        MAXNBTHOLE, NBTHOL, NBTHOLIJ, PPNBTHOLP, &
                                        PPNBTHOL1, PPNBTHOL2, PPNBTHOL3, R2)
                                ENDIF
#endif

                             endif
                          endif

                       ! Add it to the list
#if KEY_PERT == 1 /*pertatom3*/
#if NBONDA_P_FLAG == 1
                       IF(QPERT) THEN
#endif
                          !sbcp handle chem pert
#if KEY_CHEMPERT == 1
                          iprtsu=ipert(i)+ipert(j)
#endif
                          if ( &
#if KEY_CHEMPERT == 1
                               ((.not.qchemp).and.(iprtsu /= 0)).or. &
                               ((     qchemp).and.(iprtsu == 1)) &
#else
                               IPERT(I)+IPERT(J) /= 0          &
#endif
                               ) then
                             ! Add if it belongs in the product list
                             IF(IX14 /= 0) THEN
                                NNNBP=NNNBP+1
                                IF (NNNBP > MAXJNP) RETURN
                                JNBP(NNNBP)=IX14
                             ENDIF
                             ! Add if it belongs in the reactant list
                             IF(IX14P /= 0) THEN
                                NNNBR=NNNBR+1
                                IF (NNNBR > MAXJNR) RETURN
                                JNBR(NNNBR)=IX14P
                             ENDIF
                          ELSE
                             ! Add if it belongs in the main list
                             IF(IX14 /= 0) THEN
                                NNNB=NNNB+1
                                IF (NNNB > MAXJNB) RETURN
                                JNB(NNNB)=IX14
                             ENDIF
                          ENDIF
#if NBONDA_P_FLAG == 1
                       ELSE
#endif
#endif /* (pertatom3)  IF PERT*/
#if NBONDA_NOPERT == 1
                          IF(IX14 /= 0) THEN
                             ! Add if it belongs in the main list
                             NNNB=NNNB+1
                             IF (NNNB > MAXJNB) RETURN
                             JNB(NNNB)=IX14
#if KEY_MTS == 1 /*mtsatom2*/
#if NBONDA_M_FLAG== 1
                             IF (QTBMTS) THEN
#endif
                                !
                                !-------- Short-long range selection
                                !
                                IF(SLFG) THEN
                                   IF(R2 >= RSHL2T) THEN
                                      NNMT2=NNMT2+1
                                      IF (NNMT2 > MAXJM2) RETURN
                                      JNM2(NNMT2)=IX14
                                   ENDIF
                                   IF(R2 <= RSCUT2T) THEN
                                      NNMT1=NNMT1+1
                                      IF (NNMT1 > MAXJM1) RETURN
                                      JNM1(NNMT1)=IX14
                                   ENDIF
                                ENDIF
                                !
                                !-------- Hydrogen interaction selection
                                !
                                IF(TBHY1) THEN
                                   IF(IMTF(I)+IMTF(J) /= 0) THEN
                                      NNMT1=NNMT1+1
                                      IF(NNMT1 > MAXJM1) RETURN
                                      JNM1(NNMT1)=IX14
                                   ELSE
                                      NNMT2=NNMT2+1
                                      IF(NNMT2 > MAXJM2) RETURN
                                      JNM2(NNMT2)=IX14
                                   ENDIF
                                ENDIF
#if NBONDA_M_FLAG== 1
                             ENDIF
#endif
#endif /* (mtsatom2)  IF MTS*/
                             !
                          ENDIF
#endif /*  nopert*/
#if KEY_PERT == 1 /*pertatom4*/
#if NBONDA_P_FLAG == 1
                       ENDIF
#endif
#endif /* (pertatom4)*/
                       !
                    ENDIF  !  (DOIT)
                 ENDIF   !  (DOIT)
              ENDDO   ! J
           ENDIF   ! (RSDISP(JRS) >= 0)
        ENDDO   ! JRS
        INBLO(I)=NNNB
#if KEY_MTS == 1 /*mtsend*/
#if NBONDA_M_FLAG== 1
        IF (QTBMTS) THEN
#endif
           IF(TBHY1.OR.SLFG) THEN
              INBLM1(I)=NNMT1
              INBLM2(I)=NNMT2
           ENDIF
#if NBONDA_M_FLAG== 1
        ENDIF
#endif
#endif /* (mtsend)  IF MTS*/
#if KEY_PERT == 1 /*pertend*/
#if NBONDA_P_FLAG == 1
        IF(QPERT) THEN
#endif
           INBLOP(I)=NNNBP
           INBLOR(I)=NNNBR
#if NBONDA_P_FLAG == 1
        ENDIF
#endif
#endif /* (pertend)*/
     ENDDO   ! I
  ENDDO    ! IRS
