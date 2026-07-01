  DO I=1,NATOM
     NPR=INBLO(I)-ITEMP
     ITEMP=INBLO(I)
     IF(NPR > 0) THEN
        ETEMP1=0.0
        ETEMP2=0.0
        I1=ITC(IAC(I))
        IACI=IOFF(I1)
        !
        IF (ELECFG) THEN
           CGT=CGF*CG(I)
           ELCFG=(CGT /= 0.0)
        ENDIF
        !
        !     USE FDXI,FDYI,FDZI FOR ITH COMPONENT OF FORCE VECTORS
        !     USE CRXI,CRYI,CRZI FOR ITH COMPONENT OF THE COORDINATES
        !
        FDXI=DX(I)
        FDYI=DY(I)
        FDZI=DZ(I)
        CRXI=X(I)
        CRYI=Y(I)
        CRZI=Z(I)
#if ENBONDA_CFF_FOURD==1
#if ENBONDA_CFF_FGUARD ==1
        IF(DIM4) THEN                                     
#endif
           FDFDI=DFDIM(I)
           CRFDIMI=FDIM(I)
#if ENBONDA_CFF_FGUARD ==1
        ENDIF                                             
#endif
#endif 
        !
        DO JPR=1,NPR
           NB=NB+1
           IF (JNB(NB) < 0) THEN
              CGT2=CGT*E14FAC
              E14F = E14M1
              J=-JNB(NB)
              J1=ITC(IAC(J))
           ELSE
              CGT2=CGT
              E14F=ZERO
              J=JNB(NB)
              J1=ITC(IAC(J))
           ENDIF
           IC=MNO(I1,J1)
           !
           DXI=CRXI-X(J)
           DYI=CRYI-Y(J)
           DZI=CRZI-Z(J)
#if ENBONDA_CFF_PBOUND==1 /*pbound*/
#if ENBONDA_CFF_PGUARD==1
           If(qBoun) then                                    
#endif
              If(qCUBoun.or.qTOBoun) then
                 DXI = BOXINV * DXI
                 DYI = BOYINV * DYI
                 DZI = BOZINV * DZI
                 dxi = dxi - nint(dxi)
                 dyi = dyi - nint(dyi)
                 dzi = dzi - nint(dzi)
!!$                 IF(DXI >   HALF) DXI = DXI - ONE
!!$                 IF(DXI <  -HALF) DXI = DXI + ONE
!!$                 IF(DYI >   HALF) DYI = DYI - ONE
!!$                 IF(DYI <  -HALF) DYI = DYI + ONE
!!$                 IF(DZI >   HALF) DZI = DZI - ONE
!!$                 IF(DZI <  -HALF) DZI = DZI + ONE
                 If (qTOBoun) Then
                    CORR = HALF * AINT ( R75 * (ABS(DXI) + &
                         ABS(DYI) + &
                         ABS(DZI)))
                    DXI = DXI - SIGN( CORR,  DXI  )
                    DYI = DYI - SIGN( CORR,  DYI  )
                    DZI = DZI - SIGN( CORR,  DZI  )
                 Endif
                 DXI = XSIZE * DXI
                 DYI = YSIZE * DYI
                 DZI = ZSIZE * DZI
              Else
                 Call PBMove(DXI, DYI, DZI)
              Endif
#if ENBONDA_CFF_PGUARD==1
           Endif                                             
#endif
#endif /*      (pbound)*/
           S=DXI*DXI+DYI*DYI+DZI*DZI
#if ENBONDA_CFF_FOURD==1
#if ENBONDA_CFF_FGUARD ==1
           IF(DIM4) THEN                                     
#endif
              DFDIMI=CRFDIMI-FDIM(J)
              IF(DIM4ON(6) == 1) S=S+DFDIMI*DFDIMI
#if ENBONDA_CFF_FGUARD ==1
           ENDIF                                             
#endif
#endif 
           R2=1.0/S
           R1 = SQRT(R2)
           !
           !     TO COMPUTE VDW INTERACTION FOR THIS PAIR
           !
           ! LUSED = vdw calculation done
           LUSED = .FALSE.
           !
#if ENBONDA_CFF_DSWIT==1 /*vdwopt*/
#if ENBONDA_CFF_VGUARD==1
           IF (DSWIT) THEN                                   
#endif
              !     VAN DER WAAL DISTANCE SWITCHING FUNCTION
              !
              IF (S < C2OFNB) THEN
                 SIG2=RSCLF(I)*RSCLF(J)*R2
                 SIG6=SIG2*SIG2*SIG2
                 SIG9=SQRT(SIG6)*SIG6
                 IF (S > C2ONNB) THEN
                    RIJL=C2ONNB-S
                    RIJU=C2OFNB-S
                    FUNCT=RIJU*RIJU*(RIJU-THREE*RIJL)*RUL3
                    DFN=RIJL*RIJU*RUL12
                    EN=CNBA(IC)*SIG9-CNBB(IC)*SIG6
                    ENBPR=(FUNCT*EN)
                    DEN=-R2*(9.0*CNBA(IC)*SIG9-SIX*CNBB(IC)*SIG6)
                    DF=DFN*EN+FUNCT*DEN
#if ENBONDA_CFF_LSECD==1 /*lsecd1*/
                    IF(LSECD) THEN
                       DDF=FUNCT*(CNBA(IC)*90.*SIG9- &
                            CNBB(IC)*42.0*SIG6)*R2*R2+ &
                            TWO*DFN*DEN+EN*(DFN*R2-TWO*(RIJU+RIJL)*RUL12)
                    ENDIF
#endif /*  (lsecd1)*/
                 ELSE
                    ENBPR=CNBA(IC)*SIG9-CNBB(IC)*SIG6
                    DF = -R2*(9.0*CNBA(IC)*SIG9-SIX*CNBB(IC)*SIG6)
#if ENBONDA_CFF_LSECD==1 /*lsecd2*/
                    IF(LSECD) THEN
                       DDF=(CNBA(IC)*90.*SIG9-CNBB(IC)*42.0*SIG6)*R2*R2
                    ENDIF
#endif /*  (lsecd2)*/
                 ENDIF
                 LUSED=.TRUE.
              ENDIF
              !
#endif /* (vdwopt)*/
#if ENBONDA_CFF_DFSWIT==1 /*vdwopt*/
#if ENBONDA_CFF_VGUARD==1
           ELSE IF (DFSWIT) THEN                             
#endif
              !             vdw force-based switching
              IF (S  <  C2OFNB) THEN
                 RMIN2 = RSCLF(I)*RSCLF(J)*CNBA(IC)
                 RMIN6 = RMIN2*RMIN2*RMIN2
                 R3 = R1*R2
                 IF (S > C2ONNB) THEN
                    R5 = R3*R2
                    R6 = R3*R3
                    R8 = R6*R2
                    CR12 = CNBB(IC)*RMIN6*RMIN6*OFDIF6
                    CR6  = CNBB(IC)*RMIN6*OFDIF3
                    RJUNK6 = R6-RECOF6
                    RJUNK3 = R3-RECOF3
                    ENBPR = CR12*RJUNK6*RJUNK6 - CR6*RJUNK3*RJUNK3
                    DF = SIX*CR6*RJUNK3*R5 - TWELVE*CR12*RJUNK6*R8
#if ENBONDA_CFF_LSECD==1 /*lsecd3*/
                    IF(LSECD) THEN
                       DDF = (CR12*(72.D0*R8*R6 + 84.D0*RJUNK6*R8) - &
                            CR6 * (18.D0*R8 + 24.D0*RJUNK3*R5))*R2
                    ENDIF
#endif /*  (lsecd3)*/
                 ELSE
                    SIG2=RMIN2*R2
                    SIG6=SIG2*SIG2*SIG2
                    SIG9=SQRT(SIG6)*SIG6
                    ENBPR = CNBA(IC)*SIG9-CNBB(IC)*SIG6 + RMIN6*(ONOFF3 - &
                         RMIN6*ONOFF6)
                    DF = (6.0*CNBB(IC)*SIG6 - 9.0*CNBA(IC)*SIG9)*R2
#if ENBONDA_CFF_LSECD==1 /*lsecd4*/
                    IF(LSECD) THEN
                       DDF=(CNBA(IC)*90.D0*SIG9-CNBB(IC)*42.D0*SIG6)*R2*R2
                    ENDIF
#endif /*  (lsecd4)*/
                 ENDIF
                 LUSED = .TRUE.
              ENDIF
              !
#endif /* (vdwopt)*/
#if ENBONDA_CFF_DSHFT==1 /*vdwopt*/
#if ENBONDA_CFF_VGUARD==1
           ELSE IF (DSHFT) THEN                              
#endif
              !     VAN DER WAAL DISTANCE SHIFTED FUNCTION
              !     E= CA/R**9 - CB/R**6 -CC*R**6 +CD
              !
              !             IF (S < C2OFNB) THEN
              !               SGSHSQ=RSCLF(I)*RSCLF(J)*CNBA(IC)
              !               SIG2=SGSHSQ*R2
              !               SIG6=SIG2*SIG2*SIG2
              !               SIG9=SIG6*SQRT(SIG6)
              !               SGSHSQ=SGSHSQ*RECOF2
              !               SGSHSQ=SGSHSQ*SGSHSQ*SGSHSQ
              !               ASH6=TWO*(SGSHSQ-ONE)*SGSHSQ*SGSHSQ/SIG6
              !               BSH6=(THREE*SGSHSQ-FOUR)*SGSHSQ
              !               ENBPR=(CNBB(IC)*(SIG9-SIG6-SIG6+ASH6-BSH6))
              !
              !               DF=CNBB(IC)*R2*(TWELVE*(SIG6-SIG9)+SIX*ASH6)
              !...##IF lsecd (lsecd5)
              !               IF(LSECD) THEN
              !                 DDF=(CNBA(IC)*90.0*SIG9-CNBB(IC)*42.0*SIG6
              !    1               +THIRTY*ASH6)*R2*R2
              !               ENDIF
              !...##ENDIF  (lsecd5)
              !               LUSED=.TRUE.
              !             ENDIF
              !
#if ENBONDA_CFF_VGUARD==1
           ENDIF                                             
#endif
#endif /*  (vdwopt)*/
           !
#if ENBONDA_CFF_FOURD==1
#if ENBONDA_CFF_FGUARD ==1
           IF(DIM4) THEN                                     
#endif
              IF(DIM4ON(6) == 1) S=DXI*DXI+DYI*DYI+DZI*DZI
              IF(DIM4ON(7) == 1) S=S+DFDIMI*DFDIMI
#if ENBONDA_CFF_FGUARD ==1
           ENDIF                                             
#endif
#endif 
           !
           !     do electrostatics
           EELPR=0.0
           IF (ELCFG) THEN
              IF (CG(J) /= 0.0) THEN
                 IF(S < C2OFNB) THEN
                    IF (.NOT.LUSED) THEN
                       DF=0.0
                       DDF=0.0
                       LUSED=.TRUE.
                       ENBPR=0.0
                    ENDIF
#if ENBONDA_CFF_LEWALD==1 /*elecopts*/
#if ENBONDA_CFF_EGUARD==1
                    IF (LEWALD) THEN                            
#endif
                       RS=ONE/R1
                       CALL ERFCD(RS,KAPPA,ERFCX,DERFC,ERFMOD)
                       ERFC2=ERFCX + E14F
                       CGIJ = CGT*CG(J)
                       EELPR=CGIJ*ERFC2*R1
                       DFRS=(CGIJ*DERFC + EELPR)*R2
                       DF=DF - DFRS
#endif /* (elecopts)*/
#if ENBONDA_CFF_RSWIT==1 /*elecopts*/
#if ENBONDA_CFF_EGUARD==1
                    ELSE IF (RSWIT) THEN                        
#endif
                       !
                       !     to do-switching-function-inner-section
                       G2=CGT2*CG(J)*R2
                       IF (S > C2ONNB) THEN
                          RIJL=C2ONNB-S
                          RIJU=C2OFNB-S
                          FUNCT=RIJU*RIJU*(RIJU-3*RIJL)*RUL3
                          DFN=RIJL*RIJU*RUL12
                          EELPR=FUNCT*G2
                          DEN=R2*MINTWO*G2
                          DF=DF+DFN*G2+FUNCT*DEN
#if ENBONDA_CFF_LSECD==1 /*lsecd6*/
                          IF(LSECD) THEN
                             DDF=DDF+(FUNCT*SIX*G2)*R2*R2+TWO*DFN*DEN+ &
                                  G2*(DFN*R2-TWO*(RIJU+RIJL)*RUL12)
                          ENDIF
#endif /*  (lsecd6)*/
                       ELSE
                          EELPR=G2
                          DF=DF+R2*MINTWO*G2
#if ENBONDA_CFF_LSECD==1 /*lsecd7*/
                          IF(LSECD) DDF=DDF+SIX*G2*R2*R2
#endif /*  (lsecd7)*/
                       ENDIF
                       !
#endif /* (elecopts)*/
#if ENBONDA_CFF_RFSWIT==1 /*elecopts*/
#if ENBONDA_CFF_EGUARD==1
                    ELSE IF (RFSWIT) THEN                       
#endif
                       !       force-based switching with rdie
                       G2=CGT2*CG(J)
                       R4 = R2*R2
                       IF (S  >  C2ONNB) THEN
                          EELPR = G2*(ACOEF*R2 + TWOB*LOG(R1) - &
                               S*(CCOEF + S*DENOM) + CONSTR)
                          DF = DF - G2*(TWOA*R4+TWOB*R2+TWOC+TWOD*S)
#if ENBONDA_CFF_LSECD==1 /*lsecd8*/
                          IF(LSECD) THEN
                             DDF = DDF+G2*((SIXA*R2+TWOB)*R4-TWOC*R2-SIXD)
                          ENDIF
#endif /*  (lsecd8)*/
                       ELSE
                          EELPR = G2*(R2 + EADDR)
                          DF = DF - TWO*G2*R4
#if ENBONDA_CFF_LSECD==1 /*lsecd9*/
                          IF(LSECD) DDF = DDF + SIX*G2*R2*R4
#endif /*  (lsecd9)*/
                       ENDIF
                       !
#endif /* (elecopts)*/
#if ENBONDA_CFF_CSHFT==1 /*elecopts*/
#if ENBONDA_CFF_EGUARD==1
                    ELSE IF (CSHFT) THEN                        
#endif
                       !
                       !     TO DO-CONSTANT-SHIFT-INNER-SECTION
                       !     FOR FUNCTION SHIFTING, THE FORM OF THE POTENTIAL IS
                       !     EEL=QI*QJ/EPS*(1./R)*(1.0 - 2.0*R**2/CTOFNB**2 + R**4/CTOFNB**4)
                       !     EEL=0.0  ( R > CTOFNB )
                       !
                       G1=CGT2*CG(J)*R1
                       G2=G1*S*C2ROF2
                       G3=G2*S*CHROF2
                       EELPR=(G1+G2+G3)
                       DF=DF+R2*(G2-G1+THREE*G3)
#if ENBONDA_CFF_LSECD==1 /*lsecd10*/
                       IF(LSECD) DDF=DDF+(TWO*G1+SIX*G3)*R2*R2
#endif /*  (lsecd10)*/
                       !
#endif /* (elecopts)*/
#if ENBONDA_CFF_CSHIFT==1 /*elecopts*/
#if ENBONDA_CFF_EGUARD==1
                    ELSE IF (CSHIFT) THEN                       
#endif
                       !
                       !     TO DO-CONSTANT-SHIFT-INNER-SECTION
                       !     FOR FUNCTION SHIFTING, THE FORM OF THE POTENTIAL IS
                       !     EEL=QI*QJ/EPS*(1./R)*(1.0 - 2.0*R/CTOFNB + R**2/CTOFNB**2)
                       !     EEL=0.0  ( R > CTOFNB )
                       !
                       G1=CGT2*CG(J)*R1
                       EELPR=G1*(ONE + S*(RECOF2 + MIN2OF*R1))
                       DF=DF + G1*(RECOF2 - R2)
#if ENBONDA_CFF_LSECD==1 /*lsecd11*/
                       IF(LSECD) DDF=DDF + TWO*G1*R2*R2
#endif /*  (lsecd11)*/
                       !
#endif /* (elecopts)*/
#if ENBONDA_CFF_RSHFT==1 /*elecopts*/
#if ENBONDA_CFF_EGUARD==1
                    ELSE IF (RSHFT) THEN                        
#endif
                       !
                       !     TO DO-SHIFT-INNER-SECTION
                       !     FOR FUNCTION SHIFTING, THE FORM OF THE POTENTIAL IS
                       !     EEL=QI*QJ/EPS*(1./R**2 + R**2/(CTOFNB**4) - 2.0/CTOFNB
                       !     EEL=0.0  ( R > CTOFNB )
                       !
                       G2=CGT2*CG(J)
                       G1=G2*S*C4ROF
                       G3=G2*C2ROF2
                       G2=G2*R2
                       EELPR=(G1+G2+G3)
                       DF=DF+R2*(TWO*(G1-G2))
#if ENBONDA_CFF_LSECD==1 /*lsecd12*/
                       IF(LSECD) DDF=DDF+(SIX*G2+TWO*G1)*R2*R2
#endif /*  (lsecd12)*/
                       !
#endif /* (elecopts)*/
#if ENBONDA_CFF_CSWIT==1 /*elecopts*/
#if ENBONDA_CFF_EGUARD==1
                    ELSE IF (CSWIT) THEN                        
#endif
                       !
                       !     TO DO-CONSTANT-DIELECTRIC-INNER-SECTION
                       G1=CGT2*CG(J)*R1
                       IF (S > C2ONNB) THEN
                          RIJL=C2ONNB-S
                          RIJU=C2OFNB-S
                          FUNCT=RIJU*RIJU*(RIJU-3*RIJL)*RUL3
                          DFN=RIJL*RIJU*RUL12
                          EELPR=(FUNCT*G1)
                          DEN=-R2*G1
                          DF=DF+DFN*G1+FUNCT*DEN
#if ENBONDA_CFF_LSECD==1 /*lsecd13*/
                          IF(LSECD) THEN
                             DDF=DDF+TWO*(FUNCT*G1*R2*R2+DFN*DEN)+ &
                                  G1*(DFN*R2-TWO*(RIJU+RIJL)*RUL12)
                          ENDIF
#endif /*  (lsecd13)*/
                       ELSE
                          EELPR=(G1)
                          DF=DF-R2*G1
#if ENBONDA_CFF_LSECD==1 /*lsecd14*/
                          IF(LSECD) DDF=DDF+TWO*G1*R2*R2
#endif /*  (lsecd14)*/
                       ENDIF
                       !
#endif /* (elecopts)*/
#if ENBONDA_CFF_CFSWIT==1 /*elecopts*/
#if ENBONDA_CFF_EGUARD==1
                    ELSE IF (CFSWIT) THEN                       
#endif
                       !                   cdie force-based switching
                       R3 = R1*R2
                       G1 = CGT2*CG(J)
                       IF (S  >  C2ONNB) THEN
                          EELPR = G1*(R1*(ACOEF-S*(BCOEF+ &
                               S*(COVER3+DOVER5*S))) + CONST)
                          DF = DF-G1*R1*(ACOEF*R2+BCOEF+S*(CCOEF+DCOEF*S))
#if ENBONDA_CFF_LSECD==1 /*lsecd15*/
                          IF(LSECD) THEN
                             DDF=DDF+G1*(TWOA*R3*R2-(TWOC+FOURDL*S)*R1)
                          ENDIF
#endif /*  (lsecd15)*/
                       ELSE
                          EELPR = G1*(R1+EADD)
                          DF = DF - G1*R3
#if ENBONDA_CFF_LSECD==1 /*lsecd16*/
                          IF(LSECD) DDF = DDF + TWO*G1*R3*R2
#endif /*  (lsecd16)*/
                       ENDIF
#if ENBONDA_CFF_EGUARD==1
                    ELSE                                        
                       CALL DIE                                  
                    ENDIF                                       
#endif
#endif /*  (elecopts)*/
                 ENDIF
              ENDIF
           ENDIF
           !
           IF(LUSED) THEN
              !
#if ENBONDA_CFF_BLOCK==1
#if ENBONDA_CFF_BGUARD==1
              IF (QBLOCK) THEN                                
#endif
                 IBL = IBLCKP(I)
                 JBL = IBLCKP(J)
#if KEY_DOCK==1
                 !               get asymmetric matrix coefficient
                 DOCFI = 1.0
                 DOCFJ = 1.0
                 IF(QDOCK) THEN
                    KDOC  = (IBL - 1)*NBLOCK + JBL
                    DOCFI = BLDOCP(KDOC)
                    KDOC  = (JBL - 1)*NBLOCK + IBL
                    DOCFJ = BLDOCP(KDOC)
                 ENDIF
#endif /*  DOCK*/
                 IF (JBL  <  IBL) THEN
                    KK=IBL
                    IBL=JBL
                    JBL=KK
                 ENDIF
                 KK=IBL+JBL*(JBL-1)/2
                 COEF = BLCOEP(KK)
                 IF(QLDM) THEN
                    !                 first row or diagonal elements exclude (1,1).
                    IF(IBL /= 1.AND.IBL == JBL) THEN
                       FALPHA = ENBPR + EELPR
                       LAGMUL = LAGMUL + FALPHA
                       BIFLAM(JBL) = BIFLAM(JBL) + FALPHA
                    ELSE IF(IBL == 1.AND.JBL >= 2) THEN
                       FALPHA = ENBPR + EELPR
                       LAGMUL = LAGMUL + FALPHA
                       BIFLAM(JBL) = BIFLAM(JBL) + FALPHA
                    ENDIF
                 ENDIF
#if KEY_DOCK==1
                 IF(QDOCK) THEN
                    !                 Factor 0.5 to make sure no double counting
                    ENBPR=ENBPR*COEF*0.5*(DOCFI + DOCFJ)
                    EELPR=EELPR*COEF*0.5*(DOCFI + DOCFJ)
                 ELSE
#endif /*  DOCK*/
                    ENBPR=ENBPR*COEF
                    EELPR=EELPR*COEF
#if KEY_DOCK==1
                 ENDIF
#endif 
                 DF=DF*COEF
#if ENBONDA_CFF_LSECD==1 /*lsecd17*/
                 IF (LSECD) DDF=DDF*COEF
#endif /*  (lsecd17)*/
#if ENBONDA_CFF_BGUARD==1
              ENDIF                                           
#endif
              !
#if ENBONDA_CFF_BGUARD==1
              IF (.NOT. NOFORC) THEN                          
#endif
#endif /*  BLOCK*/
#if ENBONDA_CFF_FORCES==1 /*forces_main*/
                 DXIT=DXI*DF
                 DYIT=DYI*DF
                 DZIT=DZI*DF

#if KEY_DOCK==1
                 IF(QDOCK) THEN
                    FDXI=FDXI+DXIT*DOCFI
                    FDYI=FDYI+DYIT*DOCFI
                    FDZI=FDZI+DZIT*DOCFI
                    DX(J)=DX(J)-DXIT*DOCFJ
                    DY(J)=DY(J)-DYIT*DOCFJ
                    DZ(J)=DZ(J)-DZIT*DOCFJ
                 ELSE
#endif /* dock*/
                    FDXI=FDXI+DXIT
                    FDYI=FDYI+DYIT
                    FDZI=FDZI+DZIT
                    DX(J)=DX(J)-DXIT
                    DY(J)=DY(J)-DYIT
                    DZ(J)=DZ(J)-DZIT
#if KEY_DOCK==1
                 ENDIF
#endif 
#if ENBONDA_CFF_FOURD==1 /*4daddfj*/
#if ENBONDA_CFF_FGUARD ==1
                 IF(DIM4) THEN                                 
#endif
                    DFDIT=DFDIMI*DF
                    FDFDI=FDFDI+DFDIT
                    DFDIM(J)=DFDIM(J)-DFDIT
#if ENBONDA_CFF_FGUARD ==1
                 ENDIF                                         
#endif
#endif /* (4daddfj)*/
                 !     to calculate-second-derivatives
                 !
#if ENBONDA_CFF_LSECD==1 /*lsecd_main*/
                 IF(LSECD) THEN
                    !
#if KEY_TNPACK==1
                    !pd...TNPACK: 2 lignes suivantes Philippe, 28-Jul-95
                    IF(.NOT.(JNB(NB) > 0 .AND. QLOC1)) THEN
                       !pd...
#endif 
                       !
                       DDF=DDF-DF*R2
                       !
                       !     NOW UPDATE DERIVATIVE MATRICIES
                       !
                       AXX=DXI*DXI*DDF+DF
                       AYY=DYI*DYI*DDF+DF
                       AZZ=DZI*DZI*DDF+DF
                       AXY=DXI*DYI*DDF
                       AXZ=DXI*DZI*DDF
                       AYZ=DYI*DZI*DDF
#if KEY_DIMB==1
                       IF(QCMPCT) THEN
                          CALL EVDCMP(I,J,AXX,AYY,AZZ,AXY,AXZ,AYZ,DD1, &
                               PINBCM,PJNBCM)
                       ELSE
#endif /*  DIMB*/
                          !
                          II=3*I-2
                          JJ=3*J-2
                          !
                          IADD=IUPT(II)+II
                          DD1(IADD)=DD1(IADD)+AXX
                          IADD=IUPT(II+1)+II+1
                          DD1(IADD)=DD1(IADD)+AYY
                          IADD=IUPT(II+2)+II+2
                          DD1(IADD)=DD1(IADD)+AZZ
                          IADD=IUPT(II)+II+1
                          DD1(IADD)=DD1(IADD)+AXY
                          IADD=IUPT(II)+II+2
                          DD1(IADD)=DD1(IADD)+AXZ
                          IADD=IUPT(II+1)+II+2
                          DD1(IADD)=DD1(IADD)+AYZ
                          !
                          IADD=IUPT(JJ)+JJ
                          DD1(IADD)=DD1(IADD)+AXX
                          IADD=IUPT(JJ+1)+JJ+1
                          DD1(IADD)=DD1(IADD)+AYY
                          IADD=IUPT(JJ+2)+JJ+2
                          DD1(IADD)=DD1(IADD)+AZZ
                          IADD=IUPT(JJ)+JJ+1
                          DD1(IADD)=DD1(IADD)+AXY
                          IADD=IUPT(JJ)+JJ+2
                          DD1(IADD)=DD1(IADD)+AXZ
                          IADD=IUPT(JJ+1)+JJ+2
                          DD1(IADD)=DD1(IADD)+AYZ
                          !
                          IF (JJ < II) THEN
                             IADD=IUPT(JJ)+II
                             DD1(IADD)=DD1(IADD)-AXX
                             IADD=IUPT(JJ+1)+II+1
                             DD1(IADD)=DD1(IADD)-AYY
                             IADD=IUPT(JJ+2)+II+2
                             DD1(IADD)=DD1(IADD)-AZZ
                             IADD=IUPT(JJ)+II+1
                             DD1(IADD)=DD1(IADD)-AXY
                             IADD=IUPT(JJ+1)+II
                             DD1(IADD)=DD1(IADD)-AXY
                             IADD=IUPT(JJ)+II+2
                             DD1(IADD)=DD1(IADD)-AXZ
                             IADD=IUPT(JJ+2)+II
                             DD1(IADD)=DD1(IADD)-AXZ
                             IADD=IUPT(JJ+1)+II+2
                             DD1(IADD)=DD1(IADD)-AYZ
                             IADD=IUPT(JJ+2)+II+1
                             DD1(IADD)=DD1(IADD)-AYZ
                          ELSE
                             IADD=IUPT(II)+JJ
                             DD1(IADD)=DD1(IADD)-AXX
                             IADD=IUPT(II+1)+JJ+1
                             DD1(IADD)=DD1(IADD)-AYY
                             IADD=IUPT(II+2)+JJ+2
                             DD1(IADD)=DD1(IADD)-AZZ
                             IADD=IUPT(II+1)+JJ
                             DD1(IADD)=DD1(IADD)-AXY
                             IADD=IUPT(II)+JJ+1
                             DD1(IADD)=DD1(IADD)-AXY
                             IADD=IUPT(II+2)+JJ
                             DD1(IADD)=DD1(IADD)-AXZ
                             IADD=IUPT(II)+JJ+2
                             DD1(IADD)=DD1(IADD)-AXZ
                             IADD=IUPT(II+2)+JJ+1
                             DD1(IADD)=DD1(IADD)-AYZ
                             IADD=IUPT(II+1)+JJ+2
                             DD1(IADD)=DD1(IADD)-AYZ
                          ENDIF
                          !
#if KEY_DIMB==1
                       ENDIF  ! (QCMPCT)
#endif /*  DIMB*/
#if KEY_TNPACK==1
                       !pd...TNPACK: 28-Jul-95
                    ENDIF
#endif 
                    !
                 ENDIF  ! (LSECD)
#endif /*  (lsecd_main)*/
                 !
#endif /*  (forces_main)*/
#if ENBONDA_CFF_BGUARD==1 && ENBONDA_CFF_BLOCK==1
              ENDIF  ! (.NOT.NOFORC)                  
#endif
              ETEMP1=ETEMP1+ENBPR
              ETEMP2=ETEMP2+EELPR
              !
#if ENBONDA_CFF_ANALYS==1 /*analys*/
              IF(QATERM) THEN
                 IF(QANBND) THEN
                    KK=ANSLCT(I)+ANSLCT(J)
                    IF(KK == 2 .OR. (KK >= 1 .AND. .NOT.QAONLY)) THEN
                       IF(QAUNIT < 0) THEN
                          II=OUTU
                       ELSE
                          II=QAUNIT
                       ENDIF
                       !
                       IF(PRNLEV >= 5) THEN
                          IF(QAFIRST) THEN
                             IF(QLONGL) THEN
                                WRITE(II,243)
                             ELSE
                                WRITE(II,244)
                             ENDIF
                             QAFIRST=.FALSE.
                          ENDIF
                          CALL ATOMID(I,SIDDNI,RIDDNI,RESDNI,ACDNI)
                          CALL ATOMID(J,SIDDNJ,RIDDNJ,RESDNJ,ACDNJ)
                          IF(QLONGL) THEN
                             WRITE(II,245) NB,I,SIDDNI(1:idleng), &
                                  RIDDNI(1:idleng),RESDNI(1:idleng), &
                                  ACDNI(1:idleng), &
                                  J,SIDDNJ(1:idleng),RIDDNJ(1:idleng), &
                                  RESDNJ(1:idleng),ACDNJ(1:idleng), &
                                  ONE/SQRT(R2),ENBPR,EELPR,DF,IC, &
                                  RSCLF(I)*RSCLF(J)*CNBA(IC), &
                                  CNBB(IC),CG(I),CG(J)
                          ELSE
                             WRITE(II,246) NB,I,SIDDNI(1:idleng), &
                                  RIDDNI(1:idleng),RESDNI(1:idleng), &
                                  ACDNI(1:idleng), &
                                  J,SIDDNJ(1:idleng),RIDDNJ(1:idleng), &
                                  RESDNJ(1:idleng),ACDNJ(1:idleng), &
                                  ONE/SQRT(R2),ENBPR,EELPR,DF,IC, &
                                  RSCLF(I)*RSCLF(J)*CNBA(IC), &
                                  CNBB(IC),CG(I),CG(J)
                          ENDIF
                       ENDIF
                    ENDIF
                 ENDIF
              ENDIF
              !
              IF(QECONTX) THEN
                 S=HALF*(ENBPR+EELPR)
                 ECONTX(I)=ECONTX(I)+S
                 ECONTX(J)=ECONTX(J)+S
              ENDIF
#endif /* (analys)*/
           ENDIF ! (LUSED)
        ENDDO  ! JPR
        !
        !     RESTORE ITH COMPONENT OF FORCE IN THE ARRAY
        !
#if ENBONDA_CFF_BGUARD==1 && ENBONDA_CFF_BLOCK==1
        IF (.NOT. NOFORC) THEN                      
#endif
#if ENBONDA_CFF_FORCES==1 /*forces_set*/
           DX(I)=FDXI
           DY(I)=FDYI
           DZ(I)=FDZI
#if ENBONDA_CFF_FOURD==1 /*4daddfi*/
#if ENBONDA_CFF_FGUARD ==1
           IF(DIM4) THEN                                     
#endif
              DFDIM(I)=FDFDI
#if ENBONDA_CFF_FGUARD ==1
           ENDIF                                             
#endif
#endif /* (4daddfi)*/
#endif /* (forces_set)*/
#if ENBONDA_CFF_BGUARD==1 && ENBONDA_CFF_BLOCK==1
        ENDIF                                       
#endif
        !
        ENB=ENB+ETEMP1
        EEL=EEL+ETEMP2

     ENDIF  ! (NPR > 0)
  ENDDO  ! I
