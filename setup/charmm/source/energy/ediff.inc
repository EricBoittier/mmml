!  Main loop begin
      DO I=IATMIN,IATMAX
        IF(I.GT.1) THEN
          NXI=IBLO14(I-1)+1
        ELSE
          NXI=1
        ENDIF
        NXIMAX=IBLO14(I)
        XI=X(I)
        YI=Y(I)
        ZI=Z(I)
        IACI=IACNB(I)
        CGT=CGF*CG(I)
        DTX=ZERO
        DTY=ZERO
        DTZ=ZERO
!
        JS=1
        JQ=NATOMX
        IF(I.GE.JS .AND. IATMIN.LE.NATOMX) JS=I+1
        DO J=JS,JQ
          IF(ISLCT(I).EQ.1 .OR. ISLCT(J).EQ.1) THEN
!
! See if pair is in excluded list
!
            INBX=IABS(INB14(NXI))
            DO WHILE(NXI.LE.NXIMAX .AND. J.GT.INBX)
              NXI=NXI+1
              INBX=IABS(INB14(NXI))
            ENDDO
            IF(NXI.GT.NXIMAX) THEN
              IX14=J
            ELSE IF(J.EQ.INB14(NXI)) THEN  ! exclusion found
              IX14=0
            ELSE
              IF(J.EQ.INBX) THEN
                 IX14=-J    ! it's a 1-4 pair
              ELSE
                 IX14=J
              ENDIF
            ENDIF
            IF(IX14.NE.0) THEN
!             compute the atom-atom distance
              XD=XI-X(J)
              YD=YI-Y(J)
              ZD=ZI-Z(J)
              
#ifdef EDIFF_PBOUND  /* pbound */
#if EDIFF_DEBUG==1
              If(qBoun) then                       
#endif
                If(qCUBoun.or.qTOBoun) then
                  XD = BOXINV * XD
                  YD = BOYINV * YD
                  ZD = BOZINV * ZD
                  XD = XD - INT(XD + SIGN(HALF,XD))
                  YD = YD - INT(YD + SIGN(HALF,YD))
                  ZD = ZD - INT(ZD + SIGN(HALF,ZD))
                  If (qTOBoun) Then
                     CORR = HALF*AINT(R75*(ABS(XD)+ABS(YD)+ABS(ZD)))
                     XD = XD - SIGN( CORR,  XD  )
                     YD = YD - SIGN( CORR,  YD  )
                     ZD = ZD - SIGN( CORR,  ZD  )
                  Endif
                  XD = XSIZE * XD
                  YD = YSIZE * YD
                  ZD = ZSIZE * ZD
                Else
                    Call PBMove(XD, YD, ZD)
                Endif
#if EDIFF_DEBUG==1
              Endif                                
#endif
#endif  /* pbound */
              R2=XD*XD+YD*YD+ZD*ZD
              S2=MAX(RSMALL,R2)
              IF(S2.LT.C2OFNB) THEN
                TFELEC=ZERO
                TFVDW=ZERO
                LOUTER=(S2.GT.C2ONNB)
!
!     Electrostatic / van der Waals switch function
                IF (SWITCH) THEN
                  IF (LOUTER) THEN
                    RIJL=C2ONNB-S2
                    RIJU=C2OFNB-S2
                    FSW=RIJU*RIJU*(RIJU-THREE*RIJL)*RUL3
                    DFSW=RIJL*RIJU*RUL12
                  ENDIF
                ENDIF
                IVECT=LOWTP(MAX(IACNB(J),IACI))+IACNB(J)+IACI
                CH=CGT*CG(J)
                IF(IX14.LT.0) THEN
                  CH=CH*E14FAC
                  IVECT=IVECT+NITCC2
                ENDIF
!
                TR2=ONE/S2
                TR6=TR2*TR2*TR2
!
!------ Electrostatic energies (only if there are charges)
!
                IF(CH.NE.ZERO) THEN
                  IF (LCONS) THEN
                    R1 = SQRT(TR2)
! cdie original shift
                    IF (CSHFT) THEN
                      FSH=ONE+S2*CTROF2
                      CH=CH*R1*FSH
                      ENE=CH*FSH
                      TFELEC= -ENE*TR2+C4ROF2*CH
! cdie force shift
                    ELSE IF (CSHIFT) THEN
                      CH=CH*R1
                      ENE=CH*(ONE + S2*(MIN2OF*R1-CTROF2))
                      TFELEC= - CH*(CTROF2 + TR2)
! cdie force switch
                    ELSE IF (CFSWIT) THEN
                      IF (LOUTER) THEN
                        ENE = CH*( R1* (ACOEF - S2*(BCOEF + S2*(COVER3 &
                                    + DOVER5*S2))) + CONST)
                        TFELEC = - CH*R1*( ACOEF*TR2 + BCOEF + &
                                      S2*(CCOEF + DCOEF*S2) )
                      ELSE
                         ENE = CH*(R1+EADD)
                         TFELEC = - CH*R1*TR2
                      ENDIF
! cdie switch
                    ELSE IF (CSWIT) THEN
                      IF (LOUTER) THEN
                        CH=CH*R1
                        ENE=CH*FSW
                        TFELEC= -ENE*TR2+CH*DFSW
                      ELSE
                        ENE=CH*R1
                        TFELEC = -ENE*TR2
                      ENDIF
! no elec
                    ELSE
                      ENE=ZERO
                      TFELEC = ZERO
                    ENDIF
                  ELSE
! rdie original shift
                    IF (RSHFT) THEN
                      FSH=ONE+S2*CTROF2
                      CH=CH*TR2*FSH
                      ENE=CH*FSH
                      TFELEC= -TWO*ENE*TR2+C4ROF2*CH
! rdie switch
                    ELSE IF (RSWIT) THEN
                      IF (LOUTER) THEN
                        CH=CH*TR2
                        ENE=CH*FSW
                        TFELEC= -TWO*ENE*TR2+CH*DFSW
                      ELSE
                        ENE=CH*TR2
                        TFELEC= -TWO*ENE*TR2
                      ENDIF
! no elec
                    ELSE
                      ENE=ZERO
                      TFELEC = ZERO
                    ENDIF
                  ENDIF
                ELSE
                  ENE=ZERO
                  TFELEC = ZERO
                ENDIF
!
!------ End of Electrostatic energies
!
!------ VDW energies
!
             IF (.NOT.QETEN .AND. .NOT. QETSR) THEN
!
! vdw shift
                IF (LVSH) THEN
                  CA=CCNBA(IVECT)*TR6*TR6
                  ENEVDW = CA-CCNBB(IVECT)*TR6
                  CC=S2*S2*S2*CCNBC(IVECT)
                  ENN=ENEVDW-CC+CCNBD(IVECT)
                  TFVDW=MINSIX*(ENEVDW+CA+CC)*TR2
! vdw force switch
                ELSE IF(LVFSW) THEN
                  IF (LOUTER) THEN
                    IF(.NOT.LCONS .OR. CH.EQ.ZERO) R1 = SQRT(TR2)
                    R3 = R1*TR2
                    RJUNK6 = TR6-RECOF6
                    RJUNK3 = R3-RECOF3
                    CR12 = CCNBA(IVECT)*OFDIF6*RJUNK6
                    CR6  = CCNBB(IVECT)*OFDIF3*RJUNK3
                    ENN = CR12*RJUNK6 - CR6*RJUNK3
                    TFVDW = TR2*(SIX*CR6*R3 - TWELVE*CR12*TR6)
                  ELSE
                    CA=CCNBA(IVECT)*TR6*TR6
                    ENEVDW = CA-CCNBB(IVECT)*TR6
                    ENN = ENEVDW+CCNBB(IVECT)*ONOFF3-CCNBA(IVECT)*ONOFF6
                    TFVDW = MINSIX*TR2*(ENEVDW+CA)
                  ENDIF
! vdw switch
                ELSE IF(LVSW) THEN
                  CA=CCNBA(IVECT)*TR6*TR6
                  IF (LOUTER) THEN
                    ENEVDW = CA-CCNBB(IVECT)*TR6
                    ENN=ENEVDW*FSW
                    TFVDW = ENEVDW*DFSW-SIX*TR2*(ENN+CA*FSW)
                  ELSE
                    ENN = CA-CCNBB(IVECT)*TR6
                    TFVDW =  MINSIX*TR2*(ENN+CA)
                  ENDIF
! no vdw
                ELSE
                  ENN = ZERO
                ENDIF
             ELSE IF (QETSR) THEN
                               ! V_ETSR = V_ETEN / (1 + (2*r/(3*sigma))^12)
               ! Note:   CCNBA = eps * sigma^12
               !         CCNBB = 2 * eps * sigma^6
               ! SWTMP = (2*r/(3*sigma))^12
               ! TTP12 = (2/3)^12 /4
               TTPW1 = EIGHT*FOUR
               TTPW2 = NINE*NINE*NINE
               TTP12 = TTPW1*TTPW1/(TTPW2*TTPW2)
               SWTMP = TTP12*CCNBB(IVECT)*CCNBB(IVECT) &
                    /(TR6*TR6*CCNBA(IVECT)*CCNBA(IVECT))

               CA=THIRTN*CCNBA(IVECT)*TR6*TR6
               ENEVDW = (CA-NINE*CCNBB(IVECT)* &
                    ((TWO*CCNBA(IVECT)/CCNBB(IVECT))**(TWO/THREE))* &
                    TR6*TR2*TR2+TWO*CCNBB(IVECT)*TR6 ) &
                    / ( ONE + SWTMP) 
               ENN=ENEVDW
               TFVDW=TR2*MINTWO*((78*CCNBA(IVECT)*TR6*TR6)- &
                    (45*CCNBB(IVECT)* &
                    ((TWO*CCNBA(IVECT)/CCNBB(IVECT))**(TWO/THREE))* &
                    TR6*TR2*TR2)+(SIX*CCNBB(IVECT)*TR6)) &
                    / ( ONE + SWTMP) &
                    - TR2*ENEVDW*TWELVE*SWTMP/(ONE+SWTMP)    
             ELSE   ! QETEN = TRUE
                CA=THIRTN*CCNBA(IVECT)*TR6*TR6
                ENEVDW = CA-(NINE+NINE)*CCNBB(IVECT)* &
                        ((TWO*CCNBA(IVECT)/CCNBB(IVECT))**(TWO/THREE))* &
                        TR6*TR2*TR2+TWO*CCNBB(IVECT)*TR6
                ENN=ENEVDW
                TFVDW=TR2*MINTWO*((78*CCNBA(IVECT)*TR6*TR6)- &
                        (45*CCNBB(IVECT)* &
                        ((TWO*CCNBA(IVECT)/CCNBB(IVECT))**(TWO/THREE))* &
                        TR6*TR2*TR2)+(EIGHT*CCNBB(IVECT)*TR6))
             ENDIF
!
!------ End of VDW energies
!
!
#if KEY_BLOCK==1 /*block_1*/
                IF(QBLOCK) THEN
                  IBL=IBLCKP(I)
                  JBL=IBLCKP(J)
                  KK=MAX(IBL,JBL)
                  KK=KK*(KK-1)/2+MIN(IBL,JBL)
                  ENE=ENE*BLCOEE(KK)
                  ENN=ENN*BLCOEV(KK)
                  TF=TFELEC*BLCOEE(KK)+TFVDW*BLCOEV(KK)
                ELSE
                  TF=TFELEC+TFVDW
                ENDIF
!
                IF(.NOT.NOFORC) THEN
#else /*  (block_1)*/
                  TF=TFELEC+TFVDW
#endif /* (block_1)*/
!
                  XD=XD*TF
                  YD=YD*TF
                  ZD=ZD*TF
                  DTX=DTX+XD
                  DTY=DTY+YD
                  DTZ=DTZ+ZD
                  DX(J)=DX(J)-XD
                  DY(J)=DY(J)-YD
                  DZ(J)=DZ(J)-ZD
#if KEY_BLOCK==1 /*block_3*/
                ENDIF
#endif /* (block_3)*/
!
                ENB=ENB+ENN
                EEL=EEL+ENE
              ENDIF   ! (R2.LE.C2OFNB)
            ENDIF   ! (IX14.NE.0)
          ENDIF   ! (ISLCT...)
        ENDDO   ! (J)
        DX(I)=DX(I)+DTX
        DY(I)=DY(I)+DTY
        DZ(I)=DZ(I)+DTZ
      ENDDO   ! (I)
!  Main loop end
