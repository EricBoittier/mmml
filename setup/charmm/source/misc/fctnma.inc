   DO ITRANS=1,NTRANS

      NGRPX=NGRP
      IF(.NOT.LIMINV) THEN
         LIMALX=.FALSE.
      ELSE IF (IMINV(ITRANS)==ITRANS) THEN
         LIMALX=.FALSE.
      ELSE IF (IMINV(ITRANS) > ITRANS) THEN
         LIMALX=.TRUE.
      ELSE IF (IMINV(ITRANS) < ITRANS) THEN
         ! throw out this entire image, but first fill list arrays
         DO KRS=1,NGRP
            IS=IGPBS(IRS+1)+1
            IF(IMATTR(IS) == IGPBS(KRS)+1.AND.IS <= IMATPT(ITRANS))THEN
               IRS=IRS+1

               IS=IGPBS(IRS)+1
               IQ=IGPBS(IRS+1)
               DO I=IS,IQ
                  IMBLO(I)=NIMNB
                  IMBLOS(I)=NIMNBS
               ENDDO
            ENDIF
         ENDDO

         NGRPX=0
      ENDIF

      DO KRS=1,NGRPX
         IS=IGPBS(IRS+1)+1
         IF(IMATTR(IS) == IGPBS(KRS)+1 .AND. IS <= IMATPT(ITRANS)) THEN
            IRS=IRS+1
#ifdef FACTS_REPLICA /*reptest*/
#ifdef FACTS_DEBUG
            IF (qRep) THEN                                  
#endif
               iRepNo = repNoG(kRs)
               iRepID = repID(iRepNo)
#if KEY_REPDEB==1 /*repdebug*/
               IF(qRepDB) write(outu,*) 'IG:ID:No ', iRs,iRepID, iRepNo
#endif /* (repdebug)  REPDEB*/
#ifdef FACTS_DEBUG
            ENDIF                                           
#endif
#endif /* (reptest)*/
            IF(LIMALX) THEN
               KRSX=1
            ELSE
               KRSX=KRS
            ENDIF
#if KEY_PARALLEL==1
            IMYNOD = MOD(KRSX+MYNOD,NUMNOD)
            DO JRS=KRSX+IMYNOD,NGRP,NUMNOD
               RSDISP(JRS)=0
#else /**/
            DO JRS=KRSX,NGRP
               RSDISP(JRS)=0
#endif 
               ! FACTS FIX
#ifdef FACTS_IMOVE
               ! IF(IMOVEG(KRS) > 0.AND.IMOVEG(JRS) > 0) RSDISP(JRS)=-2 
#endif
!------------------------------------------------------------------
#ifdef FACTS_REPLICA /*repmain*/
#ifdef FACTS_DEBUG
               IF (qRep) THEN                                
#endif
#if KEY_REPDEB==1 /*repdb1*/
                  IF (qRepDB) THEN
                     write(outu,*) 'JG:ID:No ',jRs,        &
                           repID(repNoG(jRs)),repNoG(jRs)
                  ENDIF
#endif /* (repdb1)*/
                  IF ( iRepID == repID(repNoG(jRs)) .AND.  &
                     iRepNo  /=  repNoG(jRs) )   THEN
                     nRepXG = nRepXG + 1
#if KEY_REPDEB==1 /*repdb2*/
                     IF (qRepDB)write(outu,*)' *****EXCLUDING GROUP PAIR'
#endif /* (repdb2)*/
                     RSDISP(JRS)=-2
                  ENDIF
#ifdef FACTS_DEBUG
               ENDIF                                         
#endif
#endif /* (repmain)  REPLICA*/
!------------------------------------------------------------------
               IF(RSDISP(JRS) >= 0) THEN
                  ! find the minimum distance between the solid rectangles.
                  XD=RSCMX(IRS)-RSCMX(JRS)
                  YD=RSCMY(IRS)-RSCMY(JRS)
                  ZD=RSCMZ(IRS)-RSCMZ(JRS)
   
                  XD1=MAX(ABS(XD)-RSXMAX(IRS)-RSXMAX(JRS),ZERO)
                  YD1=MAX(ABS(YD)-RSYMAX(IRS)-RSYMAX(JRS),ZERO)
                  ZD1=MAX(ABS(ZD)-RSZMAX(IRS)-RSZMAX(JRS),ZERO)
                  R2=XD1*XD1+YD1*YD1+ZD1*ZD1
                  IF (R2  >  FCTHCS) THEN
                     RSDISP(JRS)=-1
                  ELSE
                     NGAT=NGAT+1
                        XD1=ABS(XD)+RSXMAX(IRS)+RSXMAX(JRS)
                        YD1=ABS(YD)+RSYMAX(IRS)+RSYMAX(JRS)
                        ZD1=ABS(ZD)+RSZMAX(IRS)+RSZMAX(JRS)
                        R2=XD1*XD1+YD1*YD1+ZD1*ZD1
                        IF (R2  <  FCTLCS) THEN
                           RSDISP(JRS)=1
                           NGAT=NGAT-1
                        ENDIF
                  ENDIF
               ENDIF  !  (RSDISP(JRS) >= 0)
            ENDDO  !  (JRS=KRSX,NGRP)
!==============  ATOM - ATOM  INTERACTIONS =============================
            ! Construct the jnb array searching only close residue
            ! pairs (R2<CTNBSQ)
            IS=IGPBS(IRS)+1
            IQ=IGPBS(IRS+1)
            DO I=IS,IQ
               NXI=IMIBLO(I-1)+1
               NXIMAX=IMIBLO(I)
   
               XI=X(I)
               YI=Y(I)
               ZI=Z(I)
#ifdef FACTS_IMOVE
               MOVEFG=IMOVE(I) > 0                          
#endif
#ifdef FACTS_REPLICA /*repatom1*/
               !# <caves>Aug-4-1993 (Leo Caves)
#ifdef FACTS_DEBUG
               IF (qRep) THEN                              
#endif
                  iRepNo = repNoA(IMATTR(i))
                  iRepID = repID(iRepNo)
#if KEY_REPDEB==1 /*repatomdeb*/
                  IF(qRepDB) write(outu,*) 'IA:ID:No ',i,iRepID, iRepNo
#endif /* (repatomdeb)*/
#ifdef FACTS_DEBUG
               ENDIF                                       
#endif
#endif /* (repatom1)  REPLICA*/
#if KEY_PARALLEL==1
               IMYNOD = MOD(KRSX+MYNOD,NUMNOD)
               DO JRS=KRSX+IMYNOD,NGRP,NUMNOD
#else /**/
               DO JRS=KRSX,NGRP
#endif 
                  IF(RSDISP(JRS) >= 0) THEN
                     JS=IGPBS(JRS)+1
                     JQ=IGPBS(JRS+1)
   
                     LSELF=.FALSE.
                     IF(JRS == KRS) THEN
                        IF(.NOT.LIMALX) THEN
                        LSELF=.TRUE.
                        JS=JS+I-IS
                        ENDIF
                     ENDIF
                     ! -----------
                     DO J=JS,JQ
! ==================================================================
! Check to see if this atom pair needs to be examined.
                        ! FACTS FIX
#ifdef FACTS_IMOVE
                        ! DOIT=.NOT.(MOVEFG.AND.(IMOVE(J) > 0))  
#endif
                        DOIT=.TRUE.                             ! # # .not.imove
                        ! write(*,*) 'DOIT - 1:',I,J,DOIT
!-----------------------------------------------------------------------
#ifdef FACTS_REPLICA /*repatom2*/
#ifdef FACTS_DEBUG
                        IF (qRep) THEN                        
#endif
#if KEY_REPDEB==1
                           IF (qRepDB) THEN
                              write(outu,*) 'JA:ID:No ',j,      &
                                    repID(repNoA(j)),repNoA(j)
                           ENDIF
#endif /*  REPDEB*/
                           IF ( iRepID == repID(repNoA(j)) .AND. &
                              iRepNo  /=  repNoA(j) ) THEN
                                 nRepXA = nRepXA + 1
#if KEY_REPDEB==1
                                 IF (qRepDB) write(outu,*)' *****EXCLUDING ATOM PAIR'
#endif /*  REPDEB*/
                                 DOIT=.FALSE.
                           ENDIF
#ifdef FACTS_DEBUG
                        ENDIF                                 
#endif
#endif /* (repatom2)  REPLICA*/
!-----------------------------------------------------------------------
!-----------------------------------------------------------------------
                        IF(DOIT) THEN  !  1. (DOIT)
                           ! See if pair is in excluded list
                           INBX=IABS(IMINB(NXI))
                           DO WHILE(NXI <= NXIMAX .AND. J > INBX)
                              NXI=NXI+1
                              INBX=IABS(IMINB(NXI))
                           ENDDO
                           IF(NXI > NXIMAX) THEN
                              IX14=J
                           ELSE IF(J==IMINB(NXI)) THEN  ! exclusion found
                              IX14=0
                              FCTINC=J
                           ELSE
                              IF(J==INBX) THEN
                                 IX14=-J    ! it's a 1-4 pair
                              ELSE
                                 IX14=J
                              ENDIF
                           ENDIF
      
                           IF(IX14==0) DOIT=.FALSE.
      
                           IF(RSDISP(JRS)==0 .AND. DOIT) THEN
                              XD=XI-X(J)
                              YD=YI-Y(J)
                              ZD=ZI-Z(J)
      
                              R2=XD*XD+YD*YD+ZD*ZD
      
                              IF (R2  <  FCTHCS) THEN
                                 IF (R2 < WMINSQ) THEN
                                    IF(WRNLEV >= 3) THEN
                                       CALL ATOMID(I,SIDDN,RIDDN,RESDN,ACDN)
                                       CALL ATOMID(J,SIDDN2,RIDDN2,RESDN2,ACDN2)
                                       write(outu,245)                            &
                                          SIDDN(1:idleng),RIDDN(1:idleng),      &
                                          RESDN(1:idleng),ACDN(1:idleng),       &
                                          SIDDN2(1:idleng),RIDDN2(1:idleng),    &
                                          RESDN2(1:idleng),ACDN2(1:idleng),     &
                                          SQRT(R2)
                                       IF(.NOT.(INITIA(I,X,Y,Z).AND.            &
                                                INITIA(J,X,Y,Z)))               &
                                                CALL WRNDIE(-2,'<FCTNMA>',      &
                                                     'Some atoms have undefined positions')
                                    ENDIF
                                 ENDIF
                              ELSE
                                 DOIT=.FALSE.
                              ENDIF
                           ENDIF  ! (RSDISP(JRS)==0 .AND. DOIT)
                           IF (RSDISP(JRS) == 1 .AND. DOIT) THEN
                              R2=0.0D0
                           ENDIF
                           ! write(*,*) 'DOIT - 2:',I,J,DOIT
                           !-----------------------------------------------------------------------
                           IF(DOIT) THEN
                              ! Add it to the list
                              IF(IX14 /= 0) THEN
                                 IF (R2 < FCT1LN) THEN
                                    FCT3AC=FCT3AC+1
                                    IF (FCT3AC  >  MXFCIB) RETURN
                                    FCT3LB(FCT3AC)=IABS(IX14)
                                 ENDIF
#ifdef FACTS_IMOVE
                                 IF(.NOT.(MOVEFG.AND.(IMOVE(J) > 0))) THEN     
#endif
                                    IF (R2 < CTNBSQ) THEN
                                       IF (LSELF) THEN
                                          IF (J == JS) THEN
                                             NIMNBS=NIMNBS+1
                                             IMJNBS(NIMNBS)=IX14
                                          ELSE
                                             NIMNB=NIMNB+1
                                             IF (NIMNB > MXJNB) RETURN
                                             IMJNB(NIMNB)=IX14
                                          ENDIF
                                       ELSE
                                          NIMNB=NIMNB+1
                                          IF (NIMNB  >  MXJNB) RETURN
                                          IMJNB(NIMNB)=IX14
                                       ENDIF
                                    ENDIF
                                    IF (R2  <  FCT2LN) THEN
                                       IF ((CG(I)          /= 0.0D0)  .AND. &
                                          (CG(IABS(IX14)) /= 0.0D0)) THEN
                                          FCT4AC=FCT4AC+1
                                          IF (FCT4AC  >  MXFCIB) RETURN
                                          FCT4LB(FCT4AC)=IABS(IX14)
                                       ENDIF
                                    ENDIF
#ifdef FACTS_IMOVE
                                 ENDIF                                         
#endif
                                 !            write(6,444) nimnb,i,j,imjnb(nimnb)
                                 ! 444        format(20X,4I10)
                              ENDIF  !  (IX14 /= 0)
                           ELSE   !  2. (DOIT)
                              IF (IX14 == 0) THEN
                                 FCT3AC=FCT3AC+1
                                 IF (FCT3AC  >  MXFCIB) RETURN
                                 FCT3LB(FCT3AC)=FCTINC
#ifdef FACTS_IMOVE
                                 IF(.NOT.(MOVEFG.AND.(IMOVE(J) > 0))) THEN        
#endif
                                    IF ((CG(I)       /=  0.0D0)  .AND. &
                                       (CG(FCTINC)  /=  0.0D0)) THEN
                                       FCT4AC=FCT4AC+1
                                       IF (FCT4AC  >  MXFCIB) RETURN
                                       FCT4LB(FCT4AC)=FCTINC
                                    ENDIF
#ifdef FACTS_IMOVE
                                 ENDIF                                            
#endif
                              ENDIF  !  (IX14 == 0)
                           ENDIF  !  2. (DOIT)
                        ENDIF  ! 1. (DOIT)
                     ENDDO  !  J
                  ENDIF  !  (RSDISP(JRS) >= 0)
               ENDDO  ! JRS
               IMBLO(I)  = NIMNB
               IMBLOS(I) = NIMNBS
               FCT3LL(I) = FCT3AC
               FCT4LL(I) = FCT4AC
            ENDDO  ! (I=IS,IQ)
         ENDIF  !  (IMATTR(IS) == IGPBS(KRS)+1 .AND. IS <= IMATPT(ITRANS))
! end of KRS loop
      ENDDO   !  (KRS=1,NGRPX)
! end of ITRANS loop
   ENDDO  ! ITRANS
