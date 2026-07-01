  DO IRS=1,NGRP
#ifdef FACTS_REPLICA /*reptest*/
      !# <caves>-Aug-4-1993 (Leo Caves)
#ifdef FACTS_DEBUG
      IF (qRep) THEN                                  
#endif
         iRepNo = repNoG(iRs)
         iRepID = repID(iRepNo)
#if KEY_REPDEB==1 /*repdebug*/
         IF(qRepDB) write(outu,*) 'IG:ID:No ', iRs,iRepID, iRepNo
#endif /* (repdebug)  REPDEB*/
#ifdef FACTS_DEBUG
      ENDIF                                           
#endif
#endif /* (reptest)*/

! -----------------------
#if KEY_PARALLEL==1 /*paragroup*/
#if KEY_PARAFULL==1 /*parfgroup*/
      IMYNOD=MOD(IRS+MYNOD,NUMNOD)
      DO JRS=IRS+IMYNOD,NGRP,NUMNOD
         RSDISP(JRS)=0
#endif /* (parfgroup)*/
#else /* (paragroup)*/
      DO JRS=IRS,NGRP
         RSDISP(JRS)=0
#endif /* (paragroup)*/
!------------------------------------------------------------------
! FACTS FIX
#ifdef FACTS_IMOVE
         ! IF(IMOVEG(IRS) > 0 .AND. IMOVEG(JRS) > 0) RSDISP(JRS)=-2 
#endif
!
!------------------------------------------------------------------
#ifdef FACTS_REPLICA /*repmain*/
!# <caves>-Aug-4-1993 (Leo Caves)
! Replica Exclusions (Group Exclusion)
! Rationale: If groups belong to same subsystem (repID) but are not in same
! replica unit (repNoG) - then EXCLUDE the group pair
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
              ! find distances between group centers
            XD=RSCMX(IRS)-RSCMX(JRS)
            YD=RSCMY(IRS)-RSCMY(JRS)
            ZD=RSCMZ(IRS)-RSCMZ(JRS)
            !
            XD1=MAX(ABS(XD)-RSXMAX(IRS)-RSXMAX(JRS),ZERO)
            YD1=MAX(ABS(YD)-RSYMAX(IRS)-RSYMAX(JRS),ZERO)
            ZD1=MAX(ABS(ZD)-RSZMAX(IRS)-RSZMAX(JRS),ZERO)
            R2=XD1*XD1+YD1*YD1+ZD1*ZD1

            IF (R2  >  FCTHCS) THEN
#if KEY_DEBUG==1
               if(prnlev >= 7) write(outu,447) irs,jrs,sqrt(r2)                
447               format(' NBONDA: Rejecting group pair irs,jrs,r:',2I5,F10.5) 
#endif
               RSDISP(JRS)=-1
            ELSE
               NGAT=NGAT+1

               XD1=ABS(XD)+RSXMAX(IRS)+RSXMAX(JRS)
               YD1=ABS(YD)+RSYMAX(IRS)+RSYMAX(JRS)
               ZD1=ABS(ZD)+RSZMAX(IRS)+RSZMAX(JRS)
               R2 =XD1*XD1+YD1*YD1+ZD1*ZD1

               IF (R2 < FCTLCS) THEN
                  RSDISP(JRS)=1
                  NGAT=NGAT-1
               ENDIF
            ENDIF
         ENDIF  ! (RSDISP(JRS) >= 0)
      ENDDO   ! JRS - jrs=nl0,nl1,nli
!==============  ATOM - ATOM  INTERACTIONS =============================
!
! Construct the jnb array searching only close residue
! pairs (RSDISP>=0)

      IS=IGPBS(IRS)+1
      IQ=IGPBS(IRS+1)
      DO I=IS,IQ
         IF (I > 1) THEN
            NXI=IBLO14(I-1)+1
         ELSE
            NXI=1
         ENDIF
         NXIMAX=IBLO14(I)

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
            iRepNo = repNoA(i)
            iRepID = repID(iRepNo)
#if KEY_REPDEB==1 /*repatomdeb*/
            IF(qRepDB) write(outu,*) 'IA:ID:No ',i,iRepID, iRepNo
#endif /* (repatomdeb)*/
#ifdef FACTS_DEBUG
         ENDIF                                      
#endif
#endif /* (repatom1)  REPLICA*/
!

#if KEY_PARALLEL==1 /*paraatom*/
#if KEY_PARAFULL==1 /*parfatom*/
         IMYNOD=MOD(IRS+MYNOD,NUMNOD)
         DO JRS=IRS+IMYNOD,NGRP,NUMNOD
#endif /* (parfatom)*/
#else /* (paraatom)*/
         DO JRS=IRS,NGRP
#endif /* (paraatom)*/
            IF(RSDISP(JRS) >= 0) THEN
               JS=IGPBS(JRS)+1
               JQ=IGPBS(JRS+1)
               IF(I >= JS) JS=I+1
               DO J=JS,JQ
                  !
#if KEY_DEBUG==1
                  if(prnlev >= 7) write(outu,457) irs,jrs,rsdisp(jrs),i,j           
457                  format(' NBONDA: Examining atom pair irs,jrs,rsdisp,i,j:',5I5) 
#endif
!=======================================================================
! Check to see if this atom pair needs to be examined.
!
! FACTS FIX
#ifdef FACTS_IMOVE
                  ! DOIT=.NOT.(MOVEFG.AND.(IMOVE(J) > 0))    
#endif
                  DOIT=.TRUE.                                ! # # .not.imove
                  ! write(*,*) 'DOIT - 1:',I,J,DOIT
!-----------------------------------------------------------------------
#ifdef FACTS_REPLICA /*repatom2*/
!# <caves>-Aug-4-1993 (Leo Caves)
! Replica Exclusions (Atom Exclusion)
! Rationale: If atoms belong to same subsystem (repID) but are not in
! same replica unit (repNoA) - then EXCLUDE the atom pair
! This check is probably not necessary, due to the nature of the
! replication process.
#ifdef FACTS_DEBUG
                  IF (qRep) THEN                      
#endif
#if KEY_REPDEB==1
                    IF (qRepDB) write(outu,*) 'JA:ID:No ',j, &         
                               repID(repNoA(j)),repNoA(j)              
#endif
                    IF ( iRepID == repID(repNoA(j)) .AND.    &
                         iRepNo  /=  repNoA(j) )   THEN
                       nRepXA = nRepXA + 1
#if KEY_REPDEB==1
                       IF (qRepDB) write(outu,*)             &         
                               ' *****EXCLUDING ATOM PAIR'             
#endif
                       DOIT=.FALSE.
                    ENDIF
#ifdef FACTS_DEBUG
                  ENDIF                               
#endif
#endif /* (repatom2)  REPLICA*/
!-----------------------------------------------------------------------
!-----------------------------------------------------------------------
!-----------------------------------------------------------------------
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
                     ELSE IF(J==INB14(NXI)) THEN  ! exclusion found
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
!-----------------------------------------------------------------------
                     IF(RSDISP(JRS)==0 .AND. DOIT) THEN
                        ! compute the atom-atom distance         
                        XD=XI-X(J)
                        YD=YI-Y(J)
                        ZD=ZI-Z(J)

                        R2=XD*XD+YD*YD+ZD*ZD

                        IF (R2  <  FCTHCS) THEN
                           IF(R2 < WMINSQ) THEN
                              IF(WRNLEV >= 3) THEN
                              ! Write information
                              CALL ATOMID(I,SIDDN,RIDDN,RESDN,ACDN)
                              CALL ATOMID(J,SIDDN2,RIDDN2,RESDN2,ACDN2)
                              write(outu,245) SIDDN(1:idleng),                &
                                    RIDDN(1:idleng),RESDN(1:idleng),          &
                                    ACDN(1:idleng),                           &
                                    SIDDN2(1:idleng),RIDDN2(1:idleng),        &
                                    RESDN2(1:idleng),ACDN2(1:idleng),         &
                                    SQRT(R2)
                                 IF(.NOT.(INITIA(I,X,Y,Z).AND.                &
                                          INITIA(J,X,Y,Z)))THEN
                                    CALL WRNDIE(-2,'<NBONDG>',                &
                                       'Some atoms have undefined positions')
                                    CMPLTD=.TRUE.
                                    RETURN
                                 ENDIF
                              ENDIF
                           ENDIF
#if KEY_DEBUG==1
                           if(prnlev >= 7) write(outu,456) i,j,sqrt(r2)                    
456                        format('         Distance is OK for atom pair          :' , &   
                                  15x,2I5,F12.4)                                           
#endif
                        ELSE
                           DOIT=.FALSE.
#if KEY_DEBUG==1
                           if(prnlev >= 7) write(outu,458) i,j,sqrt(r2)                    
458                        format('         Rejecting atom pair due to distance   :' , &   
                                  15x,2I5,F12.4)                                           
#endif
                        ENDIF
                     ELSE
#if KEY_DEBUG==1
                        if(prnlev >= 7 .and. .not.DOIT) write(outu,461) i,j                
461                     format('         Rejecting atom pair not by distance   :'    , &   
                               15x,2I5,F12.4)                                              
#endif
                     ENDIF  ! (RSDISP(JRS)==0 .AND. DOIT)
                     ! FACTS ADDITION
                     IF (RSDISP(JRS) == 1 .AND. DOIT) THEN
                        R2=0.0D0
                     ENDIF
                     ! write(*,*) 'DOIT - 2:',I,J,DOIT
!-----------------------------------------------------------------------
                     IF(DOIT) THEN
#if KEY_DEBUG==1
                        if(prnlev >= 7) write(outu,459) i,j                                
459                     format('         Accepting atom pair                   :',15x,2I5) 
#endif
                        ! Add it to the list
                        IF(IX14 /= 0) THEN
                           IF (R2 < FCT1LN) THEN
                              FCT1AC=FCT1AC+1
                              IF (FCT1AC > MXFCAB) RETURN
                              FCT1LB(FCT1AC)=IABS(IX14)
                           ENDIF
#ifdef FACTS_IMOVE
                           IF(.NOT.(MOVEFG.AND.(IMOVE(J) > 0))) THEN   
#endif
                              ! Add if it belongs in the main list
                              IF (R2 < CTNBSQ) THEN
                                 NNNB=NNNB+1
                                 IF (NNNB  >  MAXJNB) RETURN
                                 JNB(NNNB)=IX14
                              ENDIF
                              IF (R2 < FCT2LN) THEN
                                 IF ((CG(I)          /=  0.0D0)  .AND. &
                                     (CG(IABS(IX14)) /=  0.0D0)) THEN
                                    FCT2AC=FCT2AC+1
                                    IF (FCT2AC  >  MXFCAB) RETURN
                                    FCT2LB(FCT2AC)=IABS(IX14)
                                 ENDIF
                              ENDIF
#ifdef FACTS_IMOVE
                           ENDIF                                        
#endif
                        ENDIF  !  (IX14 /= 0)
                     !
                     ELSE   !  (DOIT)
                        IF (IX14 == 0) THEN
                           FCT1AC=FCT1AC+1
                           IF (FCT1AC > MXFCAB) RETURN
                           FCT1LB(FCT1AC)=FCTINC
#ifdef FACTS_IMOVE
                           IF(.NOT.(MOVEFG.AND.(IMOVE(J) > 0))) THEN   
#endif
                              IF ((CG(I)      /= 0.0D0)  .AND. &
                                  (CG(FCTINC) /= 0.0D0)) THEN
                                 FCT2AC=FCT2AC+1
                                 IF (FCT2AC  >  MXFCAB) RETURN
                                 FCT2LB(FCT2AC)=FCTINC
                              ENDIF
#ifdef FACTS_IMOVE
                           ENDIF                                        
#endif
                        ENDIF  !  (IX14 == 0)
                     ENDIF  !  (DOIT)
                  ENDIF  !  (DOIT)
               ENDDO  !  J
            ENDIF  !  (RSDISP(JRS) >= 0)
         ENDDO  !  JRS
         ! Add 2nd atom of pair
         INBLO(I)  = NNNB
         FCT1LL(I) = FCT1AC
         FCT2LL(I) = FCT2AC
      ENDDO  ! I
   ENDDO  ! IRS
