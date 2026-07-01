  !=======================================================================
  !  Main loop begin
  !=======================================================================
  !
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
              IS=IGPBS(IRS)+1
              IQ=IGPBS(IRS+1)
              DO I=IS,IQ
                 IMBLO(I)=NIMNB
                 IMBLOS(I)=NIMNBS
              ENDDO
              !
#if NBONDM_MTS==1
#if NBONDM_MGUARD==1
              IF (QTBMTS) THEN                            
#endif
                 DO I=IS,IQ
                    IMBM1(I)=NIMMT1
                    IMBM2(I)=NIMMT2
                    IMBM1S(I)=NIMT1S
                    IMBM2S(I)=NIMT2S
                 ENDDO
#if NBONDM_MGUARD==1
              ENDIF                                       
#endif
#endif 
              !
#if NBONDM_PERT==1
#if NBONDM_PGUARD==1
              IF(QPERT) THEN                              
#endif
                 DO I=IS,IQ
                    IMBLOR(I)=NIMNBR
                    IMBLOP(I)=NIMNBP
                    IMBLSR(I)=NIMNSR
                    IMBLSP(I)=NIMNSP
                 ENDDO
#if NBONDM_PGUARD==1
              ENDIF                                       
#endif
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
#if NBONDM_LEXS==1 /*lexs1*/
#if NBONDM_LGUARD==1
           IF(LEXS) THEN                                   
#endif
              ITMP=IGPBS(KRS)+1
              CALL ATOMID(ITMP,SIDDN,RIDDN,RESDN,ACDN)
              IF(NEXS.GT.0) LTMP=QINLIST(SIDDN,SLIST,NEXS)
#if NBONDM_LGUARD==1
           ENDIF                                           
#endif
#endif /*  (lexs1)*/
           !
#if NBONDM_REPLICA==1 /*reptest*/
#if NBONDM_RGUARD==1
           IF (qRep) THEN                                  
#endif
              iRepNo = repNoG(kRs)
              iRepID = repID(iRepNo)
#if KEY_REPDEB==1 /*repdebug*/
              IF(qRepDB) WRITE(outU,*) 'IG:ID:No ', iRs,iRepID, iRepNo
#endif /* (repdebug)  REPDEB*/
#if NBONDM_RGUARD==1
           ENDIF                                           
#endif
#endif /* (reptest)*/
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
              RSDISP(JRS)=0
#else /**/
           DO JRS=KRSX,NGRP
              RSDISP(JRS)=0
#endif 
              !
              !rcz..B960628.rcz, enabling the EXSG option
#if NBONDM_LEXS==1 /*lexs2*/
#if NBONDM_LGUARD==1
              IF(LEXS) THEN                                 
#endif
                 JTMP=IGPBS(JRS)+1
                 CALL ATOMID(JTMP,SIDDN2,RIDDN2,RESDN2,ACDN2)
                 !     nonbonded interactions between different segments will be excluded
                 IF (NEXS.LE.0) THEN
                    IF(SIDDN.NE.SIDDN2) RSDISP(JRS)=-2
                 ELSE
                    IF(LTMP.OR.QINLIST(SIDDN2,SLIST,NEXS)) &
                         RSDISP(JRS)=-2
                 ENDIF
#if NBONDM_LGUARD==1
              ENDIF                                         
#endif
              !RCZ 91/05/31
#endif /* (lexs2)*/
#if NBONDM_IMOVE==1
              IF(IMOVEG(KRS).GT.0.AND.IMOVEG(JRS).GT.0) RSDISP(JRS)=-2 
#endif
              !
              !------------------------------------------------------------------
#if NBONDM_REPLICA==1 /*repmain*/
#if NBONDM_RGUARD==1
              IF (qRep) THEN                                
#endif
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
                    RSDISP(JRS)=-2
                 ENDIF
#if NBONDM_RGUARD==1
              ENDIF                                         
#endif
#endif /* (repmain)  REPLICA*/
              !------------------------------------------------------------------
              !
              IF(RSDISP(JRS).GE.0) THEN
                 ! find the minimum distance between the solid rectangles.
                 XD=RSCMX(IRS)-RSCMX(JRS)
                 YD=RSCMY(IRS)-RSCMY(JRS)
                 ZD=RSCMZ(IRS)-RSCMZ(JRS)
                 XD1=MAX(ABS(XD)-RSXMAX(IRS)-RSXMAX(JRS),ZERO)
                 YD1=MAX(ABS(YD)-RSYMAX(IRS)-RSYMAX(JRS),ZERO)
                 ZD1=MAX(ABS(ZD)-RSZMAX(IRS)-RSZMAX(JRS),ZERO)
                 R2=XD1*XD1+YD1*YD1+ZD1*ZD1
                 IF(R2.GT.CTNBSQ) THEN
                    RSDISP(JRS)=-1
                 ELSE
                    NGAT=NGAT+1
#if NBONDM_MTS==1 /*mtsnoclose*/
#if NBONDM_MGUARD==1
                    IF (.NOT. QTBMTS) THEN                  
#endif
#endif /*  (mtsnoclose)*/
#if NBONDM_NOMTS==1
                       XD1=ABS(XD)+RSXMAX(IRS)+RSXMAX(JRS)
                       YD1=ABS(YD)+RSYMAX(IRS)+RSYMAX(JRS)
                       ZD1=ABS(ZD)+RSZMAX(IRS)+RSZMAX(JRS)
                       R2=XD1*XD1+YD1*YD1+ZD1*ZD1
                       IF(R2.LT.CTNBSQ) THEN
                          RSDISP(JRS)=1
                          NGAT=NGAT-1
                       ENDIF
#endif /*   nomts*/
#if NBONDM_MTS==1 /*mtsnoclose*/
#if NBONDM_MGUARD==1
                    ENDIF                                   
#endif
#endif /*  (mtsnoclose)*/
                 ENDIF
              ENDIF
           ENDDO  ! JRS
           !
           !==============  ATOM - ATOM  INTERACTIONS =============================
           !
           ! Construct the jnb array searching only close residue
           ! pairs (R2<CTNBSQ)
           !
           IS=IGPBS(IRS)+1
           IQ=IGPBS(IRS+1)
           DO I=IS,IQ
              NXI=IMIBLO(I-1)+1
              NXIMAX=IMIBLO(I)
#if NBONDM_PERT==1
#if NBONDM_PGUARD==1
              IF(QPERT) THEN                                
#endif
                 NXIP=IMIBLP(I-1)+1
                 NXIMXP=IMIBLP(I)
#if NBONDM_PGUARD==1
              ENDIF                                         
#endif
#endif /*  IF PERT*/
              XI=X(I)
              YI=Y(I)
              ZI=Z(I)
#if NBONDM_IMOVE==1
              MOVEFG=IMOVE(I).GT.0                          
#endif
              !
#if NBONDM_REPLICA==1 /*repatom1*/
              !# <caves>Aug-4-1993 (Leo Caves)
#if NBONDM_RGUARD==1
              IF (qRep) THEN                              
#endif
                 iRepNo = repNoA(IMATTR(i))
                 iRepID = repID(iRepNo)
#if KEY_REPDEB==1 /*repatomdeb*/
                 IF(qRepDB) WRITE(outU,*) 'IA:ID:No ',i,iRepID, iRepNo
#endif /* (repatomdeb)*/
#if NBONDM_RGUARD==1
              ENDIF                                       
#endif
#endif /* (repatom1)  REPLICA*/
              !
#if KEY_PARALLEL==1
              IMYNOD = MOD(KRSX+MYNOD,NUMNOD)
              DO JRS=KRSX+IMYNOD,NGRP,NUMNOD
#else /**/
              DO JRS=KRSX,NGRP
#endif 
                 IF(RSDISP(JRS).GE.0) THEN
                    JS=IGPBS(JRS)+1
                    JQ=IGPBS(JRS+1)
                    !
                    LSELF=.FALSE.
                    IF(JRS.EQ.KRS) THEN
                       IF(.NOT.LIMALX) THEN
                          LSELF=.TRUE.
                          JS=JS+I-IS
                       ENDIF
                    ENDIF
                    !
                    DO J=JS,JQ
                       ! ==================================================================
                       ! Check to see if this atom pair needs to be examined.
                       !
#if NBONDM_IMOVE==1
                       DOIT=.NOT.(MOVEFG.AND.(IMOVE(J).GT.0))  
#else
                       DOIT=.TRUE.                             
#endif
                       !-----------------------------------------------------------------------
#if NBONDM_REPLICA==1 /*repatom2*/
                       !
#if NBONDM_RGUARD==1
                       IF (qRep) THEN                        
#endif
#if KEY_REPDEB==1
                          IF (qRepDB) THEN
                             WRITE(outU,*) 'JA:ID:No ',j, &
                                  repID(repNoA(j)),repNoA(j)
                          ENDIF
#endif /*  REPDEB*/
                          IF ( iRepID .EQ. repID(repNoA(j)) .AND. &
                               iRepNo .NE. repNoA(j) )   THEN
                             nRepXA = nRepXA + 1
#if KEY_REPDEB==1
                             IF (qRepDB) WRITE(outU,*)' *****EXCLUDING ATOM PAIR'
#endif /*  REPDEB*/
                             DOIT=.FALSE.
                          ENDIF
#if NBONDM_RGUARD==1
                       ENDIF                                 
#endif
#endif /* (repatom2)  REPLICA*/
                       !-----------------------------------------------------------------------
#if NBONDM_TSM==1 /*tsmatom*/
#if NBONDM_TGUARD==1
                       IF(LTSM) THEN                         
#endif
                          IF(REACLS(I).EQ.1 .AND.PRODLS(J).EQ.1.OR. &
                               REACLS(J).EQ.1 .AND.PRODLS(I).EQ.1) DOIT=.FALSE.
#if NBONDM_TGUARD==1
                       ENDIF                                 
#endif
#endif /* (tsmatom)*/
                       !-----------------------------------------------------------------------
                       IF(DOIT) THEN
                          !
                          !
                          ! See if pair is in excluded list
                          !
                          INBX=IABS(IMINB(NXI))
                          DO WHILE(NXI.LE.NXIMAX .AND. J.GT.INBX)
                             NXI=NXI+1
                             INBX=IABS(IMINB(NXI))
                          ENDDO
                          IF(NXI.GT.NXIMAX) THEN
                             IX14=J
                          ELSE IF(J.EQ.IMINB(NXI)) THEN  ! exclusion found
                             IX14=0
                          ELSE
                             IF(J.EQ.INBX) THEN
                                IX14=-J    ! it's a 1-4 pair
                             ELSE
                                IX14=J
                             ENDIF
                          ENDIF
#if NBONDM_PERT==1 /*pertatomx*/
                          ! See if pair is in perturbed excluded list
#if NBONDM_PGUARD==1
                          IF(QPERT) THEN                      
#endif
#if KEY_CHEMPERT==1
                             !sb handle chem pert
                             iprtsu=ipert(i)+ipert(j)
                             if (((.not.qchemp).and.(iprtsu.ne.0)).or. &
                                  ((     qchemp).and.(iprtsu.eq.1))) then
#else /**/
                             IF(IPERT(I)+IPERT(J).NE.0) THEN
#endif 
                                INBX=IABS(IMINBP(NXIP))
                                DO WHILE(NXIP.LE.NXIMXP .AND. J.GT.INBX)
                                   NXIP=NXIP+1
                                   INBX=IABS(IMINBP(NXIP))
                                ENDDO
                                IF(NXIP.GT.NXIMXP) THEN
                                   IX14P=J
                                ELSE IF(J.EQ.IMINBP(NXIP)) THEN ! exclusion found
                                   IX14P=0
                                ELSE
                                   IF(J.EQ.INBX) THEN
                                      IX14P=-J    ! it's a 1-4 pair
                                   ELSE
                                      IX14P=J
                                   ENDIF
                                ENDIF
                             ELSE
                                IX14P=0
                             ENDIF
                             IF(IX14.EQ.0 .AND. IX14P.EQ.0) DOIT=.FALSE.
#if NBONDM_PGUARD==1
                          ELSE                                
#endif
#if NBONDM_PGUARD==1
                             IF(IX14.EQ.0) DOIT=.FALSE.        
#endif
#if NBONDM_PGUARD==1
                          ENDIF                               
#endif
#else /*  (pertatomx)*/
#if NBONDM_NOPERT==1
                          IF(IX14.EQ.0) DOIT=.FALSE.          
#endif
#endif /* (pertatomx)  IF PERT*/
                          !
                          IF(RSDISP(JRS).EQ.0 .AND. DOIT) THEN
                             XD=XI-X(J)
                             YD=YI-Y(J)
                             ZD=ZI-Z(J)
                             R2=XD*XD+YD*YD+ZD*ZD
                             !
                             IF (R2.LT.CTNBSQ) THEN
                                IF (R2.LT.WMINSQ) THEN
                                   IF(WRNLEV.GE.3) THEN
                                      CALL ATOMID(I,SIDDN,RIDDN,RESDN,ACDN)
                                      CALL ATOMID(J,SIDDN2,RIDDN2,RESDN2,ACDN2)
                                      WRITE(OUTU,245)  &
                                           SIDDN(1:idleng),RIDDN(1:idleng), &
                                           RESDN(1:idleng),ACDN(1:idleng), &
                                           SIDDN2(1:idleng),RIDDN2(1:idleng), &
                                           RESDN2(1:idleng),ACDN2(1:idleng), &
                                           SQRT(R2)
                                      IF(.NOT.(INITIA(I,X,Y,Z).AND. &
                                           INITIA(J,X,Y,Z))) &
                                           CALL WRNDIE(-2,'<NBONDMA>', &
                                           'Some atoms have undefined positions')
                                   ENDIF
                                ENDIF
                             ELSE
                                DOIT=.FALSE.
                             ENDIF
                          ENDIF  ! (RSDISP(JRS).EQ.0 .AND. DOIT)
                          !-----------------------------------------------------------------------
                          IF(DOIT) THEN
                             ! Add it to the list
!                         CALL ATOMID(I,SIDDN,RIDDN,RESDN,ACDN)
!                         CALL ATOMID(J,SIDDN2,RIDDN2,RESDN2,ACDN2)
!                         WRITE(*,*) I,J,
!    $                          SIDDN(1:idleng),RIDDN(1:idleng),
!    $                          RESDN(1:idleng),ACDN(1:idleng),
!    $                          SIDDN2(1:idleng),RIDDN2(1:idleng),
!    $                          RESDN2(1:idleng),ACDN2(1:idleng),
!    $                          SQRT(R2)

                            if( (.not.(isdrude(i).or.isdrude(j)))) then
                               R2=((x(i)-x(j))**2+(y(i)-y(j))**2+(z(i)-z(j))**2)
                               if(r2.lt.tholcut) then
                                 CALL MKTHOLELIST(i,j,IAC(I),IAC(J), &
                                   MAXNBTHOLE, NBTHOL, NBTHOLIJ, NBTHOLP, &
                                   NBTHOL1, NBTHOL2, NBTHOL3, R2)
!                                  write(*,'(3i5,9f12.3)') NBTHOLP,i,j,sqrt(R2)

#if NBONDM_PERT==1
                                 IF(QPERT) THEN
                                   call MKTHOLELIST(i,j,PPIAC(I),PPIAC(J), &
                                     MAXNBTHOLE, NBTHOL, NBTHOLIJ, PPNBTHOLP, &
                                     PPNBTHOL1, PPNBTHOL2, PPNBTHOL3, R2)
                                 ENDIF
#endif 

                               endif
                            endif

#if NBONDM_PERT==1 /*pertatom3*/
#if NBONDM_PGUARD==1
                             IF(QPERT) THEN                    
#endif
#if KEY_CHEMPERT==1
                                iprtsu=ipert(i)+ipert(j)
                                if (((.not.qchemp).and.(iprtsu.ne.0)).or. &
                                     ((     qchemp).and.(iprtsu.eq.1))) then
#else /**/
                                IF(IPERT(I)+IPERT(J).NE.0) THEN
#endif 
                                   ! Add if it belongs in the product list
                                   IF(IX14.NE.0) THEN
                                      IF(LSELF) THEN
                                         IF(J.EQ.JS) THEN
                                            NIMNSP=NIMNSP+1
                                            IMJNSP(NIMNSP)=IX14
                                         ELSE
                                            NIMNBP=NIMNBP+1
                                            IF (NIMNBP.GT.MXJMBP) RETURN
                                            IMJNBP(NIMNBP)=IX14
                                         ENDIF
                                      ELSE
                                         NIMNBP=NIMNBP+1
                                         IF (NIMNBP.GT.MXJMBP) RETURN
                                         IMJNBP(NIMNBP)=IX14
                                      ENDIF
                                   ENDIF
                                   ! Add if it belongs in the reactant list
                                   IF(IX14P.NE.0) THEN
                                      IF(LSELF) THEN
                                         IF(J.EQ.JS) THEN
                                            NIMNSR=NIMNSR+1
                                            IMJNSR(NIMNSR)=IX14P
                                         ELSE
                                            NIMNBR=NIMNBR+1
                                            IF (NIMNBR.GT.MXJMBR) RETURN
                                            IMJNBR(NIMNBR)=IX14P
                                         ENDIF
                                      ELSE
                                         NIMNBR=NIMNBR+1
                                         IF (NIMNBR.GT.MXJMBR) RETURN
                                         IMJNBR(NIMNBR)=IX14P
                                      ENDIF
                                   ENDIF
                                ELSE
                                   ! Add if it belongs in the main list
                                   IF(IX14.NE.0) THEN
                                      IF(LSELF) THEN
                                         IF(J.EQ.JS) THEN
                                            NIMNBS=NIMNBS+1
                                            IMJNBS(NIMNBS)=IX14
                                         ELSE
                                            NIMNB=NIMNB+1
                                            IF (NIMNB.GT.MXJNB) RETURN
                                            IMJNB(NIMNB)=IX14
                                         ENDIF
                                      ELSE
                                         NIMNB=NIMNB+1
                                         IF (NIMNB.GT.MXJNB) RETURN
                                         IMJNB(NIMNB)=IX14
                                      ENDIF
                                   ENDIF
                                ENDIF
#if NBONDM_PGUARD==1
                             ELSE                              
#endif
#endif /* (pertatom3)  IF PERT*/
#if NBONDM_NOPERT==1
                                !                        
                                IF(IX14.NE.0) THEN
                                   IF(LSELF) THEN
                                      IF(J.EQ.JS) THEN
                                         NIMNBS=NIMNBS+1
                                         IMJNBS(NIMNBS)=IX14
                                      ELSE
                                         NIMNB=NIMNB+1
                                         IF (NIMNB.GT.MXJNB) RETURN
                                         IMJNB(NIMNB)=IX14
                                      ENDIF
                                   ELSE
                                      NIMNB=NIMNB+1
                                      IF (NIMNB.GT.MXJNB) RETURN
                                      IMJNB(NIMNB)=IX14
                                   ENDIF
#if NBONDM_MTS==1
                                   !
#if NBONDM_MGUARD==1
                                   IF(TBMTS) THEN                
#endif
                                      !
                                      !---------------- Short-long range force selection
                                      !
                                      IF(SLFG) THEN
                                         IF(RSDISP(JRS).NE.0) THEN
                                            ! compute R2 only if we didn't above
                                            XD=XI-X(J)
                                            YD=YI-Y(J)
                                            ZD=ZI-Z(J)
                                            R2=XD*XD+YD*YD+ZD*ZD
                                         ENDIF  ! (RSDISP(JRS).NE.0)
                                         IF(R2.GE.RSHL2T) THEN
                                            IF(LSELF) THEN
                                               IF(J.EQ.JS) THEN
                                                  NIMT2S=NIMT2S+1
                                                  IMJM2S(NIMT2S)=IX14
                                               ELSE
                                                  NIMMT2=NIMMT2+1
                                                  IF(NIMMT2.GT.MXJNB2) RETURN
                                                  IMJM2(NIMMT2)=IX14
                                               ENDIF
                                            ELSE
                                               NIMMT2=NIMMT2+1
                                               IF(NIMMT2.GT.MXJNB2) RETURN
                                               IMJM2(NIMMT2)=IX14
                                            ENDIF
                                         ENDIF
                                         IF(R2.LE.RSCUT2T) THEN
                                            IF(LSELF) THEN
                                               IF(J.EQ.JS) THEN
                                                  NIMT1S=NIMT1S+1
                                                  IMJM1S(NIMT1S)=IX14
                                               ELSE
                                                  NIMMT1=NIMMT1+1
                                                  IF(NIMMT1.GT.MXJNB1) RETURN
                                                  IMJM1(NIMMT1)=IX14
                                               ENDIF
                                            ELSE
                                               NIMMT1=NIMMT1+1
                                               IF(NIMMT1.GT.MXJNB1) RETURN
                                               IMJM1(NIMMT1)=IX14
                                            ENDIF
                                         ENDIF
                                      ENDIF
                                      !
                                      !---------------- Mass difference selection
                                      !
                                      IF(TBHY1) THEN
                                         IF(IMTF(I)+IMTF(J).NE.0) THEN
                                            IF(LSELF) THEN
                                               IF(J.EQ.JS) THEN
                                                  NIMT1S=NIMT1S+1
                                                  IMJM1S(NIMT1S)=IX14
                                               ELSE
                                                  NIMMT1=NIMMT1+1
                                                  IF(NIMMT1.GT.MXJNB1) RETURN
                                                  IMJM1(NIMMT1)=IX14
                                               ENDIF
                                            ELSE
                                               NIMMT1=NIMMT1+1
                                               IF(NIMMT1.GT.MXJNB1) RETURN
                                               IMJM1(NIMMT1)=IX14
                                            ENDIF
                                         ELSE
                                            IF(LSELF) THEN
                                               IF(J.EQ.JS) THEN
                                                  NIMT2S=NIMT2S+1
                                                  IMJM2S(NIMT2S)=IX14
                                               ELSE
                                                  NIMMT2=NIMMT2+1
                                                  IF(NIMMT2.GT.MXJNB2) RETURN
                                                  IMJM2(NIMMT2)=IX14
                                               ENDIF
                                            ELSE
                                               NIMMT2=NIMMT2+1
                                               IF(NIMMT2.GT.MXJNB2) RETURN
                                               IMJM2(NIMMT2)=IX14
                                            ENDIF
                                         ENDIF
                                      ENDIF
#if NBONDM_MGUARD==1
                                   ENDIF                         
#endif
#endif 
                                   !            write(6,444) nimnb,i,j,imjnb(nimnb)
                                   ! 444        format(20X,4I10)
                                ENDIF
#endif /*  nopert*/
#if NBONDM_PERT==1
#if NBONDM_PGUARD==1
                             ENDIF                             
#endif
#endif /*  IF PERT*/
                             !
                          ENDIF  !  (DOIT)
                       ENDIF  !  (DOIT)
                    ENDDO  ! J
                 ENDIF  ! (RSDISP(JRS).GE.0)
              ENDDO  ! JRS
              IMBLO(I)=NIMNB
              IMBLOS(I)=NIMNBS
#if NBONDM_MTS==1
#if NBONDM_MGUARD==1
              IF (QTBMTS) THEN                            
#endif
                 IMBM1(I)=NIMMT1
                 IMBM2(I)=NIMMT2
                 IMBM1S(I)=NIMT1S
                 IMBM2S(I)=NIMT2S
#if NBONDM_MGUARD==1
              ENDIF                                       
#endif
#endif  /* NBONDM_MTS */
#if NBONDM_PERT==1
#if NBONDM_PGUARD==1
              IF(QPERT) THEN                              
#endif
                 IMBLOP(I)=NIMNBP
                 IMBLSP(I)=NIMNSP
                 IMBLOR(I)=NIMNBR
                 IMBLSR(I)=NIMNSR
#if NBONDM_PGUARD==1
              ENDIF                                       
#endif
#endif  /* NBONDM_PERT==1 */
           ENDDO  ! I
        ENDIF
        ! end of KRS loop
     ENDDO   ! KRS
     ! end of ITRANS loop
  ENDDO  ! ITRANS
  !
  !=======================================================================
