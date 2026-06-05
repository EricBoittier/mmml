  DO I=IFRSTA,NATOM
#if KEY_IMCUBES==1
     if(lbycbim)itemp=inblo(I+natom)
     NB=ITEMP
#endif 
     NPR=INBLO(I)-ITEMP
     ITEMP=INBLO(I)
#if KEY_GCMC==1
     !        Y. Deng 04-26-2005 Make invisible for grand canonical MC
     if(qgcmc) then
        lgcmcon = gcmcon(i)
     else
        lgcmcon = .true.
     endif
     
     IF (NPR > 0 .and. lGCMCON) THEN
#else /**/
     IF(NPR > 0) THEN
#endif 
        ETEMP1=0.0
        ETEMP2=0.0
#if KEY_SCCDFTB==1
        if(qsccb) then
           dvdltmp1=0.0d0
           dvdltmp2=0.0d0
        endif
#endif 
        I1=ITC(IAC(I))
        IACI=IOFF(I1)
        !
        IF (ELECFG) THEN
           CGT=CGF*CG(I)
           ELCFG=(CGT /= 0.0)
           ! SAPATEL
#if KEY_CHEQ==1
           IF (QCG)  ELCFG=.TRUE.
#endif 
           ! SAPATEL
        ELSE
           CGT=ZERO
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

#if ENBONDA_FOURD==1
#if ENBONDA_F_FLAG==1
        IF(DIM4) THEN                                     
#endif
           FDFDI=DFDIM(I)
           CRFDIMI=FDIM(I)
#if ENBONDA_F_FLAG==1
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
              IF (I1 < J1) THEN
                 IC=IOFF(J1)+I1+MAXROW
              ELSE
                 IC=IACI+J1+MAXROW
              ENDIF
#if KEY_NBIPS==1
#if ENBONDA_CEIPS==1 || ENBONDA_REIPS==1 || ENBONDA_DVIPS==1 || ENBONDA_DBEXPIPS==1
              DOIPS=.FALSE.                   
#endif
#endif 
           ELSE
#if KEY_NBIPS==1
#if ENBONDA_CEIPS==1 || ENBONDA_REIPS==1 || ENBONDA_DVIPS==1 || ENBONDA_DBEXPIPS==1
              DOIPS=LEIPS.OR.LVIPS            
#endif
#endif 
              CGT2=CGT
              E14F=ZERO
              J=JNB(NB)
              J1=ITC(IAC(J))
              IF (I1 < J1) THEN
                 IC=IOFF(J1)+I1
              ELSE
                 IC=IACI+J1
              ENDIF
           ENDIF
           
           ! namkh 01/20/04
#if KEY_GAMESS==1 || KEY_GAMESSUK==1 || KEY_MNDO97==1 || KEY_SQUANTM==1 || KEY_QCHEM==1 || KEY_QTURBO==1 || KEY_G09==1
           if(qmused)then
           IF((ABS(IGMSEL(I)) == 1.OR.ABS(IGMSEL(I)).EQ.2) .AND. &
                (ABS(IGMSEL(J)) == 1.OR.ABS(IGMSEL(J)).EQ.2) &
                .AND.QGMREM) GOTO 241
           endif
#endif 
           
#if KEY_GCMC==1
           if(qgcmc) then
              IF (.NOT. GCMCON(J)) GOTO 30
           endif
#endif 
           !
           DXI=CRXI-X(J)
           DYI=CRYI-Y(J)
           DZI=CRZI-Z(J)



#if KEY_MSMMPT==1
     IF (NATOM.EQ.MMPTNATOMX) THEN 
       IF (PBCXSIZE.GT.0.D0 .AND. PBCYSIZE.GT.0.D0 .AND. PBCZSIZE.GT.0.D0) THEN
   
          IF (DXI.LE.-0.5D0*PBCXSIZE) THEN
            DXI=DXI+PBCXSIZE
          ELSE IF (DXI.GT.0.5D0*PBCXSIZE) THEN
            DXI=DXI-PBCXSIZE
          ENDIF
          
          IF (DYI.LE.-0.5D0*PBCYSIZE) THEN
            DYI=DYI+PBCYSIZE
          ELSE IF (DYI.GT.0.5D0*PBCYSIZE) THEN
            DYI=DYI-PBCYSIZE
          ENDIF
          
          IF (DZI.LE.-0.5D0*PBCZSIZE) THEN
            DZI=DZI+PBCZSIZE
          ELSE IF (DZI.GT.0.5D0*PBCZSIZE) THEN
            DZI=DZI-PBCZSIZE
          ENDIF
   
       ENDIF
     ENDIF
#endif

#if ENBONDA_PBOUND==1
#if ENBONDA_P_FLAG==1
           If(qBoun) then                                    
#endif
              If(qCUBoun.or.qTOBoun) then
                 DXI = BOXINV * DXI
                 DYI = BOYINV * DYI
                 DZI = BOZINV * DZI
                 dxi = dxi - nint(dxi)
                 dyi = dyi - nint(dyi)
                 dzi = dzi - nint(dzi)
!!$        IF(DXI >   HALF) DXI = DXI - ONE
!!$        IF(DXI <  -HALF) DXI = DXI + ONE
!!$        IF(DYI >   HALF) DYI = DYI - ONE
!!$        IF(DYI <  -HALF) DYI = DYI + ONE
!!$        IF(DZI >   HALF) DZI = DZI - ONE
!!$        IF(DZI <  -HALF) DZI = DZI + ONE
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
#if ENBONDA_P_FLAG==1
           Endif                                             
#endif
#endif /* ENBONDA_PBOUND */
           S=DXI*DXI+DYI*DYI+DZI*DZI
#if ENBONDA_FOURD==1
#if ENBONDA_F_FLAG==1
           IF(DIM4) THEN                                     
#endif
              DFDIMI=CRFDIMI-FDIM(J)
              IF(DIM4ON(6) == 1) S=S+DFDIMI*DFDIMI
#if ENBONDA_F_FLAG==1
           ENDIF                                             
#endif
#endif 
           ! BY S.B. 03/16/00
           IF (S  <  RSMALL) S=RSMALL
           ! BY S.B. 03/16/00
           R2=1.0/S
           R1 = SQRT(R2)
#if KEY_NBIPS==1
#if ENBONDA_CEIPS==1 || ENBONDA_REIPS==1 || ENBONDA_DVIPS==1 || ENBONDA_DBEXPIPS==1
           DOIPS=DOIPS.AND.S < RIPS2
           IF(DOIPS)THEN
              !WXW Long range potential using isotropic periodic sum (IPS)
              U2=S*RIPS2R
           ENDIF
#endif 
#endif 
           !
           !     TO COMPUTE VDW INTERACTION FOR THIS PAIR
           !
           ! LUSED = vdw calculation done
           LUSED = .FALSE.
           !
#if ENBONDA_DSWIT==1 /*vdwopt*/
#if ENBONDA_V_FLAG==1
           IF (DSWIT) THEN                                   
#endif
              !     VAN DER WAAL DISTANCE SWITCHING FUNCTION
              !
              IF (S < C2OFNB) THEN
                 SIG2=RSCLF(I)*RSCLF(J)*CNBA(IC)*R2
                 SIG6=SIG2*SIG2*SIG2
                 SIG12=SIG6*SIG6
                 IF (S > C2ONNB) THEN
                    RIJL=C2ONNB-S
                    RIJU=C2OFNB-S
                    FUNCT=RIJU*RIJU*(RIJU-THREE*RIJL)*RUL3
                    DFN=RIJL*RIJU*RUL12
#if KEY_WCA==1
                    EN=WCA(I)*WCA(J)*CNBB(IC)*(SIG12-SIG6-SIG6)
#else /**/
                    EN=CNBB(IC)*(SIG12-SIG6-SIG6)
#endif 
#if KEY_LJPME==1
                    if(qljpme) call ljpme_dir(ljc6(i), ljc6(j), s, en, dfn, CNBA(IC), CNBB(IC), .false.)
#endif
                    ENBPR=(FUNCT*EN)
#if KEY_WCA==1
                    DEN=WCA(I)*WCA(J)*CNBB(IC)*R2*TWELVE*(SIG6-SIG12)
#else /**/
                    DEN=CNBB(IC)*R2*TWELVE*(SIG6-SIG12)
#endif 
                    DF=DFN*EN+FUNCT*DEN
#if ENBONDA_LSECD==1 /*lsecd1*/
                    IF(LSECD) THEN
                       DDF=FUNCT*CNBB(IC)*(156.*SIG12-84.0*SIG6)*R2*R2+ &
                            TWO*DFN*DEN+EN*(DFN*R2-TWO*(RIJU+RIJL)*RUL12)
                    ENDIF
#endif /*  (lsecd1)*/
                 ELSE IF (.NOT.QETEN .AND. .NOT.QETSR) THEN
                    !  FOR THE GO MODEL 10-12, ASSUME THAT THE CUTOFFS ARE LONG (NO EXCL.)
#if KEY_WCA==1
                    ! the cut-off is assumed to be much larger than
                    ! the atom radii, only the attractive part needs swith funtion
                    IF (S  >  CNBA(IC)) THEN
                       ENBPR=CNBB(IC)*(SIG12-SIG6-SIG6)
                       EGLJTMP = CNBB(IC)*TWELVE*(SIG6-SIG12)*R2
                       EPRPL = 0.0D0
                       EGLJRPL = 0.0D0
                    ELSE
                       ENBPR=-CNBB(IC)
                       EGLJTMP = 0.0D0
                       IF (LLSOFT) THEN
                          TMP = CNBA(IC)*(1.0-SCVDWCUTR)*(1.0-SCVDWCUTR)
                          IF (S  <=  (CNBA(IC) - TMP)) THEN
                             R2 = 1.0D0/(S + TMP)
                             SIG2=RSCLF(I)*RSCLF(J)*CNBA(IC)*R2
                             SIG6=SIG2*SIG2*SIG2
                             SIG12=SIG6*SIG6
                             EPRPL=CNBB(IC)*((SIG12-SIG6-SIG6)+1.0D0)
                             EGLJRPL=CNBB(IC)*TWELVE*(SIG6-SIG12)*R2
                          ELSE
                             EPRPL=0.0D0
                             EGLJRPL=0.0D0
                          ENDIF
                       ELSE
                          !                      SIG2=RSCLF(I)*RSCLF(J)*CNBA(IC)*R2
                          !                      SIG6=SIG2*SIG2*SIG2
                          !                      SIG12=SIG6*SIG6
                          EPRPL=CNBB(IC)*((SIG12-SIG6-SIG6)+1.0D0)
                          EGLJRPL=CNBB(IC)*TWELVE*(SIG6-SIG12)*R2
                       ENDIF
                    ENDIF
                    DF=EGLJTMP*WCA(I)*WCA(J)+EGLJRPL
                    ENBPR=(ENBPR*WCA(I)*WCA(J)+EPRPL)
#else /**/
                    ENBPR=(CNBB(IC)*(SIG12-SIG6-SIG6))
                    DF=CNBB(IC)*R2*12.0*(SIG6-SIG12)
#endif 
#if KEY_LJPME==1
                    if(qljpme) call ljpme_dir(ljc6(i), ljc6(j), s, enbpr, df, CNBA(IC), CNBB(IC), .false.)
#endif
#if ENBONDA_LSECD==1 /*lsecd2*/
                    IF(LSECD) THEN
                       DDF=CNBB(IC)*(156.*SIG12-84.0*SIG6)*R2*R2
                    ENDIF
#endif /*  (lsecd2)*/
                 ELSE IF (QETSR) THEN ! QETSR
                    FAC=FOUR/(NINE*SIG2)
                    SWTMP=(FAC*FAC)**3
                    ENBPR=(CNBB(IC)*(THIRTN*SIG12-NINE*TWO*SIG6 &
                         *SIG2*SIG2+FOUR*SIG6)) &
                         / (ONE+SWTMP)
                    ! DF = 1/R dV/dr
                    DF=CNBB(IC)*R2*TWELVE*(FIFTN*SIG6*SIG2*SIG2 &
                         -THIRTN*SIG12-TWO*SIG6) &
                         / (ONE+SWTMP) &
                         - R2*ENBPR*TWELVE*SWTMP &
                         /(ONE+SWTMP)
#if ENBONDA_LSECD==1 /*lsecd2*/
                    IF(LSECD) THEN
                       ! DDF = 1/R^2 d2V/dr2

                       ODF=CNBB(IC)*R2*TWELVE*(FIFTN*SIG6*SIG2*SIG2 &
                            -THIRTN*SIG12-TWO*SIG6)

                       DDF=(CNBB(IC)*TWELVE*(169.0*SIG12- &
                            165.0*SIG6*SIG2*SIG2+14.0*SIG6)*R2*R2) &
                            / ( ONE+SWTMP) &
                            - TWO*R2*TWELVE*ODF*SWTMP/((ONE+SWTMP)*(ONE+SWTMP)) &
                            - R2*R2*TWELVE*ENBPR*SWTMP/(ONE+SWTMP) &
                            * (MINTWO*TWELVE*SWTMP/(ONE+SWTMP) + ELEVEN)
                    ENDIF
#endif /*  (lsecd2)*/
                 ELSE ! QETEN
                    ENBPR=(CNBB(IC)*(THIRTN*SIG12-NINE*TWO*SIG6 &
                         *SIG2*SIG2+FOUR*SIG6))
                    DF=CNBB(IC)*R2*TWELVE*(FIFTN*SIG6*SIG2*SIG2 &
                         -THIRTN*SIG12-TWO*SIG6)
#if ENBONDA_LSECD==1 /*lsecd2*/
                    IF(LSECD) THEN
                       DDF=(CNBB(IC)*TWELVE*(169.0*SIG12- &
                            165.0*SIG6*SIG2*SIG2+14.0*SIG6)*R2*R2)
                    ENDIF
#endif /*  (lsecd2)*/
                 ENDIF
                 LUSED=.TRUE.
              ENDIF
              !
#endif /* (vdwopt)*/
#if ENBONDA_DFSWIT==1 /*vdwopt*/
#if ENBONDA_V_FLAG==1
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
#if ENBONDA_LSECD==1 /*lsecd3*/
                    IF(LSECD) THEN
                       DDF = (CR12*(72.D0*R8*R6 + 84.D0*RJUNK6*R8) - &
                            CR6 * (18.D0*R8 + 24.D0*RJUNK3*R5))*R2
                    ENDIF
#endif /*  (lsecd3)*/
                 ELSE
                    SIG2=RMIN2*R2
                    SIG6=SIG2*SIG2*SIG2
                    SIG12=SIG6*SIG6
                    ENBPR = CNBB(IC)*(SIG12-SIG6-SIG6 + RMIN6*(ONOFF3 - &
                         RMIN6*ONOFF6))
                    DF = CNBB(IC)*R2*TWELVE*(SIG6-SIG12)
#if ENBONDA_LSECD==1 /*lsecd4*/
                    IF(LSECD) THEN
                       DDF = CNBB(IC)*(156.D0*SIG12-84.D0*SIG6)*R2*R2
                    ENDIF
#endif /*  (lsecd4)*/
                 ENDIF
                 LUSED = .TRUE.
              ENDIF
              !
#endif /* (vdwopt)*/
#if ENBONDA_GVSWIT==1 /*vdwopt*/
#if ENBONDA_V_FLAG==1
           ELSE IF (GVSWIT) THEN                             
#endif
!                           vdw GROMACS switching
!              WRITE(*,*) 'MGL GVSWIT> CTONNB ',CTONNB
!              WRITE(*,*) 'MGL GVSWIT> CTOFNB ',CTOFNB
              IF (S  <  C2OFNB) THEN
 
!                  Brief nomenclature note: 
!                  CTONNB <-> Rshift
!                  CTOFNB <-> Rcut
                 
                 RMCTON = SQRT(S) - CTONNB
                 ALPHA=6.0
                 A6= ALPHA*((ALPHA+1.0)*CTONNB-(ALPHA+4.0)*CTOFNB)
                 A6= A6/((CTOFNB**(ALPHA+2))*(CTOFNB-CTONNB)**2)
                 B6=-ALPHA*((ALPHA+1.0)*CTONNB-(ALPHA+3.0)*CTOFNB)
                 B6= B6/((CTOFNB**(ALPHA+2))*(CTOFNB-CTONNB)**3)
                 C6= 1.0/(CTOFNB**ALPHA)-(A6/3.)*(CTOFNB-CTONNB)**3 &
                      -(B6/4.)*(CTOFNB-CTONNB)**4
                 G62=(A6/3.0)*(RMCTON)**3
                 G62=(A6/3.0)*(RMCTON)**3
                 DG62=A6*(RMCTON)**2
                 DDG62=2.0*A6*(RMCTON)
                 G63=(B6/4.0)*(RMCTON)**4
                 DG63=B6*(RMCTON)**3
                 DDG63=3.0*B6*(RMCTON)**2

                 ALPHA=12.0
                 A12= ALPHA*((ALPHA+1.0)*CTONNB-(ALPHA+4.0)*CTOFNB)
                 A12= A12/((CTOFNB**(ALPHA+2))*(CTOFNB-CTONNB)**2)
                 B12=-ALPHA*((ALPHA+1.0)*CTONNB-(ALPHA+3.0)*CTOFNB)
                 B12= B12/((CTOFNB**(ALPHA+2))*(CTOFNB-CTONNB)**3)
                 C12= 1.0/(CTOFNB**ALPHA)-(A12/3.)*(CTOFNB-CTONNB)**3 &
                      -(B12/4.)*(CTOFNB-CTONNB)**4
                 G122=(A12/3.0)*(RMCTON)**3
                 DG122=A12*(RMCTON)**2
                 DDG122=2.0*A12*(RMCTON)
                 G123=(B12/4.0)*(RMCTON)**4
                 DG123=B12*(RMCTON)**3
                 DDG123=3.0*B12*(RMCTON)**2

                 RMIN2 = RSCLF(I)*RSCLF(J)*CNBA(IC)
                 RMIN6 = RMIN2*RMIN2*RMIN2
                 RMIN12=RMIN6*RMIN6
                 R3 = R1*R2
                 R6 = R3*R3
                 R7 = R6*R1
                 R12 = R6*R6
                 R13 = R12*R1
                 R14 = R13*R1

                 IF (S > C2ONNB) THEN
!                     SHIFTED.
                    ENBPR= CNBB(IC)*(     RMIN12*(R12-G122-G123-C12) &
                                    -2.0*RMIN6 *(R6 -G62 -G63 -C6))
                    DF =   CNBB(IC)*R1*(RMIN12*( -12.0*R13-DG122-DG123) &
                                    -2.0*RMIN6*(-6.0*R7 -DG62 -DG63))
#if ENBONDA_LSECD==1 /*lsecd20*/
                    IF(LSECD) THEN
                       DDF = CNBB(IC)*R2*(RMIN12*(156.0*R14-DDG122 &
                           -DDG123)-2.0*RMIN6*( 42.0*R7 -DDG62 -DDG63))
#endif /* (lsecd20)*/
                    ENDIF
                 ELSE
!                     UNSHIFTED.
                    ENBPR= CNBB(IC)*(     RMIN12*(R12-C12) &
                                    - 2.0*RMIN6* (R6- C6))
                    DF =   CNBB(IC)*R1*(RMIN12*(   -12.0*R13) &
                                    -2.0*RMIN6*(-6.0*R7 ))
#if ENBONDA_LSECD==1 /*lsecd21*/
                    IF(LSECD) THEN
                       DDF = CNBB(IC)*R2*(RMIN12*(156.0*R14) &
                                  -2.0*RMIN6*( 42.0*R7))
#endif /* (lsecd21)*/
                    ENDIF
                 ENDIF
                 LUSED = .TRUE.
              ENDIF
!              
#endif /* (vdwopt)*/
#if ENBONDA_DSHFT==1 /*vdwopt*/
#if ENBONDA_V_FLAG==1
           ELSE IF (DSHFT) THEN                              
#endif
              !     VAN DER WAAL DISTANCE SHIFTED FUNCTION
              !     E= CA/R**12 - CB/R**6 -CC*R**6 +CD
              !
              IF (S < C2OFNB) THEN
                 SGSHSQ=RSCLF(I)*RSCLF(J)*CNBA(IC)
                 SIG2=SGSHSQ*R2
                 SIG6=SIG2*SIG2*SIG2
                 SIG12=SIG6*SIG6
                 SGSHSQ=SGSHSQ*RECOF2
                 SGSHSQ=SGSHSQ*SGSHSQ*SGSHSQ
                 !sb  if one of the rsclf factors is zero, the original line contains a
                 !    division by zero! The following may carry a performance penalty, but
                 !    these are the slow routines ...            
                 !sb                ASH6=TWO*(SGSHSQ-ONE)*SGSHSQ*SGSHSQ/SIG6
                 ash6=TWO*(SGSHSQ-ONE)*SGSHSQ*((recof2/r2)**3)
                 BSH6=(THREE*SGSHSQ-FOUR)*SGSHSQ
                 !
                 ENBPR=(CNBB(IC)*(SIG12-SIG6-SIG6+ASH6-BSH6))
                 DF=CNBB(IC)*R2*(TWELVE*(SIG6-SIG12)+SIX*ASH6)
#if KEY_LJPME==1
                 if(qljpme) then
                     ! remove the extra shifting terms so that the shifted potential is
                     ! consistent with that proposed in Wennberg LJ-PME paper
                     enbpr = enbpr - CNBB(IC)*(ASH6-BSH6)
                     df = df - CNBB(IC)*R2*SIX*ASH6
                     call ljpme_dir(ljc6(i), ljc6(j), s, enbpr, df, CNBA(IC), CNBB(IC), .true.)
                 endif
#endif
#if ENBONDA_LSECD==1 /*lsecd5*/
                 IF(LSECD) THEN
                    DDF=CNBB(IC)*R2*R2*(156.0*SIG12-84.0*SIG6+THIRTY*ASH6)
                 ENDIF
#endif /*  (lsecd5)*/
                 LUSED=.TRUE.
              ENDIF
              !
#endif /* (vdwopt)*/
#if ENBONDA_DBEXP==1 /*vdwopt*/
#if ENBONDA_V_FLAG==1
           ELSE IF (DBEXP) THEN                              
#endif
              !     VAN DER WAAL DOUBLE EXPONENTIAL POTENTIAL
              !     E= A*EXP(alpha*R)-B*EXP(beta*R)
              !
              SIG2=RSCLF(I)*RSCLF(J)*CNBA(IC)
              IF (S < C2OFNB.AND.SIG2>RSMALL) THEN
                 SIG=SQRT(SIG2)
#if KEY_NBIPS==1
                AKR=AIPSVA(0)/SIG/R1
                EXPAR=EXP(-AKR)
                ! energy
                ENBPR=AIPSVA(1)*EXPAR
                ! force
                DF=BIPSVA(1)*EXPAR/SIG
                CKR=AIPSVC(0)/SIG/R1
                EXPCR=EXP(-CKR)
                ! energy
                ENBPR=CNBB(IC)*(ENBPR-AIPSVC(1)*EXPCR)
                ! force
                DF=CNBB(IC)*(DF-BIPSVC(1)*EXPCR/SIG)*R1
#endif /* KEY_NBIPS */
#if KEY_LJPME==1
                 if(qljpme) then
                     call ljpme_dir(ljc6(i), ljc6(j), s, enbpr, df, CNBA(IC), CNBB(IC), .true.)
                 endif
#endif
#if ENBONDA_LSECD==1 /*lsecd6*/
                 IF(LSECD) THEN
                    SIG2=SIG*SIG
#if KEY_NBIPS==1 /*nbips_vips*/
                    DDF=-CNBB(IC)*(BIPSVA(1)*EXPAR*AIPSVA(0)-BIPSVC(1)*EXPCR*AIPSVC(0))*R2/SIG2
#endif
                 ENDIF
#endif /*  (lsecd6)*/
                 LUSED=.TRUE.
              ENDIF
              !
#endif /* (vdwopt)*/
#if KEY_NBIPS==1 /*nbips_vips*/
#if ENBONDA_DVIPS==1 /*vdwopt*/
#if ENBONDA_V_FLAG==1
           ELSE IF (DVIPS) THEN                              
#endif
              !     VAN DER WAAL IPS
              !WXW VDW potential using isotropic periodic sum (IPS)
              IF (DOIPS) THEN
                 !WXW   1-4  interaction will be calculated at subroutine EEXIPS
                 U4=U2*U2
                 U6R=ONE/U4/U2
                 U12R=U6R*U6R
                 !  L-J r6 term
                 !   etr6=1/r6+a0+r2*(a1+r2*(a2+r2*(a3+r2*(a4+r4*(a5+a6*r4)))))
                 !   detr6/dr*r1=-6/r6+r2*(d1+r2*(d2+r2*(d3+r2*(d4+r4*(d5+d6*r4)))))
                 !
                 PVC=U6R+AIPSVC(0)+U2*(AIPSVC(1)+U2*(AIPSVC(2)+U2*(AIPSVC(3) &
                      +U2*(AIPSVC(4)+U4*(AIPSVC(5)+U4*AIPSVC(6))))))-PIPSVCC
                 DPVC=-SIX*U6R+U2*(BIPSVC(1)+U2*(BIPSVC(2)+U2*(BIPSVC(3) &
                      +U2*(BIPSVC(4)+U4*(BIPSVC(5)+U4*BIPSVC(6))))))
                 !  L-J r12 term
                 !   etr12=1/r12+a0+r2*(a1+r2*(a2+r2*(a3+r4*(a4+r4*(a5+a6*r4)))))
                 !   detr12/dr*r1=-6/r6+r2*(d1+r2*(d2+r2*(d3+r4*(d4+r4*(d5+d6*r4)))))
                 !
                 PVA=U12R+AIPSVA(0)+U2*(AIPSVA(1)+U2*(AIPSVA(2)+U2*(AIPSVA(3) &
                      +U4*(AIPSVA(4)+U4*(AIPSVA(5)+U4*AIPSVA(6))))))-PIPSVAC
                 DPVA=-TWELVE*U12R+U2*(BIPSVA(1)+U2*(BIPSVA(2)+U2*(BIPSVA(3) &
                      +U4*(BIPSVA(4)+U4*(BIPSVA(5)+U4*BIPSVA(6))))))
                 SGSHSQ=RSCLF(I)*RSCLF(J)*CNBA(IC)
                 SIG2=SGSHSQ*RIPS2R
                 SIG6=SIG2*SIG2*SIG2
                 SIG12=SIG6*SIG6
                 ENEVA =CNBB(IC)*SIG12
                 ENEVC =-TWO*CNBB(IC)*SIG6
                 ENBPR=ENEVA*PVA+ENEVC*PVC
                 DF=(ENEVA*DPVA+ENEVC*DPVC)*R2
                 LUSED=.TRUE.
#if ENBONDA_LSECD==1 /*lsecd2*/
                 IF(LSECD) THEN
                    DDPVC=42.0D0*U6R+U2*(BBIPSVC(1)+U2*(BBIPSVC(2)+U2*(BBIPSVC(3) &
                      +U2*(BBIPSVC(4)+U4*(BBIPSVC(5)+U4*BBIPSVC(6))))))
                    DDPVA=156.0D0*U12R+U2*(BBIPSVA(1)+U2*(BBIPSVA(2)+U2*(BBIPSVA(3) &
                      +U4*(BBIPSVA(4)+U4*(BBIPSVA(5)+U4*BBIPSVA(6))))))
                    DDF=(ENEVA*DDPVA+ENEVC*DDPVC)*R2*R2
                 ENDIF
#endif /*  (lsecd2)*/
              ELSE
                 ENBPR=ZERO
                 DF=ZERO
              ENDIF
#endif /*  (vdwopt)*/
#if ENBONDA_DBEXPIPS==1 /*vdwopt*/
#if ENBONDA_V_FLAG==1
           ELSE IF (DBEXPIPS) THEN                              
#endif
              !     VAN DER WAAL DOUBLE EXPONENTIAL IPS
              !WXW VDW DE potential using isotropic periodic sum (IPS)
              SIG2=RSCLF(I)*RSCLF(J)*CNBA(IC)
              IF (DOIPS.AND.SIG2>RSMALL) THEN
                 !WXW   1-4  interaction will be calculated at subroutine EEXIPS
               SIG=SQRT(SIG2)
               IF(SRIPS*SIG*R1>ONE)THEN
                AKR=AIPSVA(0)/SIG/R1
                EXPAR=EXP(-AKR)
                !AKRC=AIPSVA(0)*RIPS/SIG
                !EXPARC=EXP(-AKRC)
                ! energy
                ENBPR=AIPSVA(1)*(EXPAR+AIPSVA(4)/EXPAR+AIPSVA(5))
                ! force 
                DF=BIPSVA(1)*(EXPAR-AIPSVA(4)/EXPAR)/SIG
                !
                CKR=AIPSVC(0)/SIG/R1
                EXPCR=EXP(-CKR)
                !CKRC=AIPSVC(0)*RIPS/SIG
                !EXPCRC=EXP(-CKRC)
                ! energy
                ENBPR=CNBB(IC)*(ENBPR-AIPSVC(1)*(EXPCR+AIPSVC(4)/EXPCR+AIPSVC(5)))
                ! force
                DF=CNBB(IC)*(DF-BIPSVC(1)*(EXPCR-AIPSVC(4)/EXPCR)/SIG)*R1
#if ENBONDA_LSECD==1 /*lsecd7*/
                 IF(LSECD) THEN
                    SIG2=SIG*SIG
                    DDF=CNBB(IC)*(AIPSVA(0)*(BIPSVA(1)*(EXPAR+AIPSVA(4)/EXPAR)) &
                    -AIPSVC(0)*(BIPSVC(1)*(EXPCR+AIPSVC(4)/EXPCR)))*R2/SIG2
                 ENDIF
#endif /*  (lsecd7)*/
                ELSE
                   ENBPR=ZERO 
                   DF=ZERO 
                ENDIF
                 LUSED=.TRUE.
              ENDIF
              !
#endif /*  (vdwopt)*/
#endif /*  (nbips_dbexpips)*/
#if ENBONDA_V_FLAG==1
           ENDIF                                             
#endif
           !
#if ENBONDA_FOURD==1
#if ENBONDA_F_FLAG==1
           IF(DIM4) THEN                                     
#endif
              IF(DIM4ON(6) == 1) S=DXI*DXI+DYI*DYI+DZI*DZI
              IF(DIM4ON(7) == 1) S=S+DFDIMI*DFDIMI
#if ENBONDA_F_FLAG==1
           ENDIF                                             
#endif
#endif 

           !
           !     do electrostatics
           EELPR=0.0
           IF (ELCFG) THEN
              ! SAPATEL
#if KEY_CHEQ==1
              IF ((CG(J) /= 0.0).OR.QCG) THEN
#else /**/
              IF (CG(J) /= 0.0) THEN
#endif 
! SAPATEL
                 IF(S < C2OFNB) THEN
                    IF (.NOT.LUSED) THEN
                       DF=0.0
                       DDF=0.0
                       LUSED=.TRUE.
                       ENBPR=0.0
                    ENDIF

                    IF (QRXNF) THEN ! Wei Chen  2015
                       G1=CGT2*CG(J)*R1
                       EELPR = CGT2*CG(J)*(R1 + (HALF*GRFCON*RC3/R2) - ((ONE+HALF*GRFCON)*RC))
                       DEN = CGT2*CG(J)*(GRFCON*RC3-R1*R2)
                       DF = DF + DEN

#if ENBONDA_LEWALD==1 /*elecopts*/
#if ENBONDA_E_FLAG==1
                    ELSE IF (LEWALD) THEN                            
#endif
                       RS=ONE/R1
                       CALL ERFCD(RS,KAPPA,ERFCX,DRFC,ERFMOD)
                       ERFC2=ERFCX + E14F
                       CGIJ = CGT*CG(J)
                       ! SAPATEL
#if KEY_CHEQ==1
                       HIJ=ERFC2*R1
                       EELPR=HIJ*CGIJ
                       HIJ=CGF*HIJ
#else /**/
                       EELPR=CGIJ*ERFC2*R1
#endif 
                       ! SAPATEL
                       DFRS=(CGIJ*DRFC + EELPR)*R2
                       DF=DF - DFRS
#endif /* (elecopts)*/
#if ENBONDA_RSWIT==1 /*elecopts*/
#if ENBONDA_E_FLAG==1
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
                          ! SAPATEL
#if KEY_CHEQ==1
                          HIJ=FUNCT*CGF*R2
#endif 
                          ! SAPATEL
                          EELPR=FUNCT*G2
                          DEN=R2*MINTWO*G2
                          DF=DF+DFN*G2+FUNCT*DEN
#if ENBONDA_LSECD==1 /*lsecd6*/
                          IF(LSECD) THEN
                             DDF=DDF+(FUNCT*SIX*G2)*R2*R2+TWO*DFN*DEN+ &
                                  G2*(DFN*R2-TWO*(RIJU+RIJL)*RUL12)
                          ENDIF
#endif /*  (lsecd6)*/
                       ELSE
                          EELPR=G2
                          ! SAPATEL
#if KEY_CHEQ==1
                          HIJ=CGF*R2
#endif 
                          ! SAPATEL
                          DF=DF+R2*MINTWO*G2
#if ENBONDA_LSECD==1 /*lsecd7*/
                          IF(LSECD) DDF=DDF+SIX*G2*R2*R2
#endif /*  (lsecd7)*/
                       ENDIF
                       !
#endif /* (elecopts)*/
#if ENBONDA_RFSWIT==1 /*elecopts*/
#if ENBONDA_E_FLAG==1
                    ELSE IF (RFSWIT) THEN                       
#endif
                       !       force-based switching with rdie
                       G2=CGT2*CG(J)
                       R4 = R2*R2
                       IF (S  >  C2ONNB) THEN
                          ! SAPATEL
#if KEY_CHEQ==1
                          EELPR = (ACOEF*R2 + TWOB*LOG(R1) - &
                               S*(CCOEF + S*DENOM) + CONSTR)
                          HIJ=EELPR*CGF
                          EELPR=EELPR*G2
#else /**/
                          EELPR = G2*(ACOEF*R2 + TWOB*LOG(R1) - &
                               S*(CCOEF + S*DENOM) + CONSTR)
#endif 
                          ! SAPATEL
                          
                          DF = DF - G2*(TWOA*R4+TWOB*R2+TWOC+TWOD*S)
#if ENBONDA_LSECD==1 /*lsecd8*/
                          IF(LSECD) THEN
                             DDF = DDF+G2*((SIXA*R2+TWOB)*R4-TWOC*R2-SIXD)
                          ENDIF
#endif /*  (lsecd8)*/
                       ELSE
                          EELPR = G2*(R2 + EADDR)
                          ! SAPATEL
#if KEY_CHEQ==1
                          HIJ=CGF*(R2 + EADDR)
#endif 
                          ! SAPATEL
                          DF = DF - TWO*G2*R4
#if ENBONDA_LSECD==1 /*lsecd9*/
                          IF(LSECD) DDF = DDF + SIX*G2*R2*R4
#endif /*  (lsecd9)*/
                       ENDIF
                       !
#endif /* (elecopts)*/
#if ENBONDA_CSHFT==1 /*elecopts*/
#if ENBONDA_E_FLAG==1
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
                       ! SAPATEL
#if KEY_CHEQ==1
                       HIJ=CGF*R1*(1.0+S*C2ROF2*(1.0+S*CHROF2))
#endif 
                       ! SAPATEL
                       DF=DF+R2*(G2-G1+THREE*G3)
#if ENBONDA_LSECD==1 /*lsecd10*/
                       IF(LSECD) DDF=DDF+(TWO*G1+SIX*G3)*R2*R2
#endif /*  (lsecd10)*/
                       !
#endif /* (elecopts)*/
#if ENBONDA_CSHIFT==1 /*elecopts*/
#if ENBONDA_E_FLAG==1
                    ELSE IF (CSHIFT) THEN                       
#endif
                       !
                       !     TO DO-CONSTANT-SHIFT-INNER-SECTION
                       !     FOR FUNCTION SHIFTING, THE FORM OF THE POTENTIAL IS
                       !     EEL=QI*QJ/EPS*(1./R)*(1.0 - 2.0*R/CTOFNB + R**2/CTOFNB**2)
                       !     EEL=0.0  ( R > CTOFNB )
                       !
                       G1=CGT2*CG(J)*R1
                       ! SAPATEL
#if KEY_CHEQ==1
                       EELPR=(ONE + S*(RECOF2 + MIN2OF*R1))
                       HIJ=EELPR*CGF*R1
                       EELPR=G1*EELPR
#else /**/
                       EELPR=G1*(ONE + S*(RECOF2 + MIN2OF*R1))
#endif 
                       ! SAPATEL
                       DF=DF + G1*(RECOF2 - R2)
#if ENBONDA_LSECD==1 /*lsecd11*/
                       IF(LSECD) DDF=DDF + TWO*G1*R2*R2
#endif /*  (lsecd11)*/
                       !
#endif /* (elecopts)*/
#if ENBONDA_RSHFT==1 /*elecopts*/
#if ENBONDA_E_FLAG==1
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
                       ! SAPATEL
#if KEY_CHEQ==1
                       HIJ=CGF*(S*C4ROF+C2ROF2+R2)
#endif 
                       ! SAPATEL
                       DF=DF+R2*(TWO*(G1-G2))
#if ENBONDA_LSECD==1 /*lsecd12*/
                       IF(LSECD) DDF=DDF+(SIX*G2+TWO*G1)*R2*R2
#endif /*  (lsecd12)*/
                       !
#endif /* (elecopts)*/
#if ENBONDA_CSWIT==1 /*elecopts*/
#if ENBONDA_E_FLAG==1
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
                          ! SAPATEL
#if KEY_CHEQ==1
                          HIJ=CGF*R1*FUNCT
#endif 
                          ! SAPATEL
                          DEN=-R2*G1
                          DF=DF+DFN*G1+FUNCT*DEN
#if ENBONDA_LSECD==1 /*lsecd13*/
                          IF(LSECD) THEN
                             DDF=DDF+TWO*(FUNCT*G1*R2*R2+DFN*DEN)+ &
                                  G1*(DFN*R2-TWO*(RIJU+RIJL)*RUL12)
                          ENDIF
#endif /*  (lsecd13)*/
                       ELSE
                          EELPR=(G1)
                          ! SAPATEL
#if KEY_CHEQ==1
                          HIJ=CGF*R1
#endif 
                          ! SAPATEL
                          DF=DF-R2*G1
#if ENBONDA_LSECD==1 /*lsecd14*/
                          IF(LSECD) DDF=DDF+TWO*G1*R2*R2
#endif /*  (lsecd14)*/
                       ENDIF
                       !
#endif /* (elecopts)*/
#if ENBONDA_CFSWIT==1 /*elecopts*/
#if ENBONDA_E_FLAG==1
                    ELSE IF (CFSWIT) THEN                       
#endif
                       !                   cdie force-based switching
                       R3 = R1*R2
                       G1 = CGT2*CG(J)
                       IF (S  >  C2ONNB) THEN
                          ! SAPATEL
#if KEY_CHEQ==1
                          EELPR = (R1*(ACOEF-S*(BCOEF+ &
                               S*(COVER3+DOVER5*S))) + CONST)
                          HIJ=CGF*EELPR
                          EELPR=G1*EELPR
#else /**/
                          EELPR = G1*(R1*(ACOEF-S*(BCOEF+ &
                               S*(COVER3+DOVER5*S))) + CONST)
#endif 
                          ! SAPATEL

                          DF = DF-G1*R1*(ACOEF*R2+BCOEF+S*(CCOEF+DCOEF*S))
#if ENBONDA_LSECD==1 /*lsecd15*/
                          IF(LSECD) THEN
                             DDF=DDF+G1*(TWOA*R3*R2-(TWOC+FOURDL*S)*R1)
                          ENDIF
#endif /*  (lsecd15)*/
                       ELSE
                          EELPR = G1*(R1+EADD)
                          ! SAPATEL
#if KEY_CHEQ==1
                          HIJ=CGF*(R1+EADD)
#endif 
                          !  SAPATEL
                          DF = DF - G1*R3
#if ENBONDA_LSECD==1 /*lsecd16*/
                          IF(LSECD) DDF = DDF + TWO*G1*R3*R2
#endif /*  (lsecd16)*/
                       ENDIF
#endif /*  (elecopts)*/
#if ENBONDA_GESWIT==1 /*elecopts*/
#if ENBONDA_E_FLAG==1
                    ELSE IF (GESWIT) THEN                       
#endif
!                                          cdie GROMACS switching

                       CG2ONNB = CGONNB * CGONNB
                       CG2OFNB = CGOFNB * CGOFNB
!                       WRITE(*,*) 'MGL GESWIT> CGONNB ',CGONNB
!                       WRITE(*,*) 'MGL GESWIT> CGOFNB ',CGOFNB
                       RMCGON = SQRT(S) - CGONNB
                       ALPHA=1.0
                       A= ALPHA*((ALPHA+1.0)*CGONNB-(ALPHA+4.0)*CGOFNB)
                       A= A/((CGOFNB**(ALPHA+2))*(CGOFNB-CGONNB)**2)

                       B= -ALPHA*((ALPHA+1.0)*CGONNB-(ALPHA+3.0)*CGOFNB)
                       B= B/((CGOFNB**(ALPHA+2))*(CGOFNB-CGONNB)**3)

                       C= 1.0/(CGOFNB**ALPHA)-(A/3.)*(CGOFNB-CGONNB)**3 &
                         -(B/4.)*(CGOFNB-CGONNB)**4

                       G2=(A/3.0)*(RMCGON)**3
                       DG2=A*(RMCGON)**2
                       DDG2=2.0*A*(RMCGON)
                       G3=(B/4.0)*(RMCGON)**4
                       DG3=B*(RMCGON)**3
                       DDG3=3.0*B*(RMCGON)**2
                       EELPR=CGT2*CG(J)*(R1-G2-G3-C)

                       IF (S  >  CG2ONNB) THEN
                          EELPR=CGT2*CG(J)*(R1-G2-G3-C)
                          DF= DF + CGT2*CG(J)*R1*(-R2-DG2-DG3)
#if ENBONDA_LSECD==1 /*lsecd18*/
                          IF(LSECD) DDF=DDF + &
                                       CGT2*CG(J)*R2*(2.0*R3-DDG2-DDG3)
#endif /* (lsecd18)*/
                       ELSE
                          EELPR=CGT2*CG(J)*(R1-C)
                          DF = DF - CGT2*CG(J)*R3
#if ENBONDA_LSECD==1 /* lsecd19 */
                          IF(LSECD) DDF=DDF+ 3.0*CGT2*CG(J)*R4
#endif /* lsecd19 */
                       ENDIF
#endif /*  (elecopts)*/
#if KEY_NBIPS==1 /*nbips_elecs*/
#if ENBONDA_REIPS==1 /*elecopts*/
#if ENBONDA_E_FLAG==1
                    ELSE IF (REIPS) THEN                        
#endif
                       IF(DOIPS)THEN
                      !
                       !     TO R-dielectric IPS
                       !
                          G2=CGT2*CG(J)
                          !  Electrostatic IPS
                          !   etr1=1/r2+r2*(a1+r2*(a2+r2*(a3+r2*(a4+r2*(a5+a6*r2)))))
                          !   detr1/dr*r1=-1/r2+r2*(d1+r2*(d2+r2*(d3+r2*(d4+r2*(d5+d6*r2)))))
                          !
                          PE=ONE/U2+U2*(AIPSE(1)+U2*(AIPSE(2)+U2*(AIPSE(3) &
                               +U2*(AIPSE(4)+U2*(AIPSE(5)+U2*AIPSE(6))))))-PIPSEC
                          DPE=-TWO/U2+U2*(BIPSE(1)+U2*(BIPSE(2)+U2*(BIPSE(3) &
                               +U2*(BIPSE(4)+U2*(BIPSE(5)+U2*BIPSE(6))))))
                          ENEP=G2*RIPS2R
                          EELPR=ENEP*PE
                          DF  =DF+ENEP*DPE*R2
#if ENBONDA_LSECD==1 /*lsecd2*/
                          IF(LSECD) THEN
                             DDPE=SIX/U2+U2*(BBIPSE(1)+U2*(BBIPSE(2)+U2*(BBIPSE(3) &
                               +U2*(BBIPSE(4)+U2*(BBIPSE(5)+U2*BBIPSE(6))))))
                             DDF=DDF+ENEP*DDPE*R2*R2
                          ENDIF
#endif /*  (lsecd2)*/
                      ELSE
                         EELPR=ZERO
                      ENDIF
                       !
#endif /*  (elecopts)*/
#if ENBONDA_CEIPS==1 /*elecopts*/
#if ENBONDA_E_FLAG==1
                    ELSE IF (CEIPS) THEN                        
#endif
                      IF(DOIPS)THEN
                       !
                       !     TO constant-dielectric IPS
                       !
                          G2=CGT2*CG(J)
                          U1=SQRT(U2)
                          !  Electrostatic IPS
                          !   etr1=1/r+r2*(a1+r2*(a2+r2*(a3+r2*(a4+r2*(a5+a6*r2)))))
                          !   detr1/dr*r=-1/r+r2*(d1+r2*(d2+r2*(d3+r2*(d4+r2*(d5+d6*r2)))))
                          !
                          PE=ONE/U1+U2*(AIPSE(1)+U2*(AIPSE(2)+U2*(AIPSE(3) &
                               +U2*(AIPSE(4)+U2*(AIPSE(5)+U2*AIPSE(6))))))-PIPSEC
                          DPE=-ONE/U1+U2*(BIPSE(1)+U2*(BIPSE(2)+U2*(BIPSE(3) &
                               +U2*(BIPSE(4)+U2*(BIPSE(5)+U2*BIPSE(6))))))
                          ENEP=G2*RIPSR
                          EELPR=ENEP*PE
                          DF =DF+ENEP*DPE*R2
            IF(LSECD) THEN
               DDPE=TWO/U1+U2*(BBIPSE(1)+U2*(BBIPSE(2)+U2*(BBIPSE(3) &
                    +U2*(BBIPSE(4)+U2*(BBIPSE(5)+U2*BBIPSE(6))))))
               DDF=CGIJ*DDPE/R2/R2
            ENDIF
                      ELSE
                         EELPR=ZERO
                      ENDIF
                       !
#endif /*  (elecopts)*/
#endif /* (nbips_elecs)*/
#if ENBONDA_E_FLAG==1
                    ELSE                                        
                       CALL DIE                                  
                    ENDIF                                       
#endif
                 ENDIF
              ENDIF
           ENDIF
           !
           IF(LUSED) THEN
              !
#if ENBONDA_BLOCK==1
#if ENBONDA_B_FLAG==1
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
!ldm
                 IF(NRST == 1)THEN
                    !                  DF=DFV+DFE
                    !  BETWEEN PROTEIN(I) AND LIGAND(J)
                    IF(IBL == 1.AND.JBL >= LSTRT) THEN
                       ! ADD FORCE TO I(PROTEIN)
                       ENVDX((JBL-2)*NATOM+I) = ENVDX((JBL-2)*NATOM+I) + DXI * DF
                       ENVDY((JBL-2)*NATOM+I) = ENVDY((JBL-2)*NATOM+I) + DYI * DF
                       ENVDZ((JBL-2)*NATOM+I) = ENVDZ((JBL-2)*NATOM+I) + DZI * DF
                       ! ADD FORCE TO J(LIGAND)
                       ENVDX(J) = ENVDX(J) - DXI * DF
                       ENVDY(J) = ENVDY(J) - DYI * DF
                       ENVDZ(J) = ENVDZ(J) - DZI * DF
                       !  BETWEEN LIGAND(I) AND PROTEIN(J)
                    ELSE IF(JBL == 1.AND.IBL >= LSTRT) THEN
                       ! ADD FORCE TO J(PROTEIN)
                       ENVDX((IBL-2)*NATOM+J) = ENVDX((IBL-2)*NATOM+J) - DXI * DF
                       ENVDY((IBL-2)*NATOM+J) = ENVDY((IBL-2)*NATOM+J) - DYI * DF
                       ENVDZ((IBL-2)*NATOM+J) = ENVDZ((IBL-2)*NATOM+J) - DZI * DF
                       ! ADD FORCE TO I(LIGAND)
                       ENVDX(I) = ENVDX(I) + DXI * DF
                       ENVDY(I) = ENVDY(I) + DYI * DF
                       ENVDZ(I) = ENVDZ(I) + DZI * DF
                       !  WITHIN LIGAND
                    ELSE IF(IBL >= LSTRT.AND.IBL == JBL) THEN
                       ENVDX(I) = ENVDX(I) + DXI * DF
                       ENVDY(I) = ENVDY(I) + DYI * DF
                       ENVDZ(I) = ENVDZ(I) + DZI * DF
                       ENVDX(J) = ENVDX(J) - DXI * DF
                       ENVDY(J) = ENVDY(J) - DYI * DF
                       ENVDZ(J) = ENVDZ(J) - DZI * DF
                    ENDIF
                 ELSE IF(NRST == 2)THEN
                    IF(IBL >= LSTRT.AND.JBL == 2)THEN
                       ENVDX(I) = ENVDX(I) + DXI * DF
                       ENVDY(I) = ENVDY(I) + DYI * DF
                       ENVDZ(I) = ENVDZ(I) + DZI * DF
                    ELSE IF(JBL >= LSTRT.AND.IBL == 2)THEN
                       ENVDX(J) = ENVDX(J) - DXI * DF
                       ENVDY(J) = ENVDY(J) - DYI * DF
                       ENVDZ(J) = ENVDZ(J) - DZI * DF
                    ELSE IF(IBL >= LSTRT.AND.IBL == JBL)THEN
                       ENVDX(I) = ENVDX(I) + DXI * DF
                       ENVDY(I) = ENVDY(I) + DYI * DF
                       ENVDZ(I) = ENVDZ(I) + DZI * DF
                       ENVDX(J) = ENVDX(J) - DXI * DF
                       ENVDY(J) = ENVDY(J) - DYI * DF
                       ENVDZ(J) = ENVDZ(J) - DZI * DF
                    ENDIF
                 ELSE IF(NRST == 3)THEN
                    ! Add force only to ligand
                    ! protein(I) and ligand(J)
                    IF(IBL == 1.AND.JBL >= LSTRT) THEN
                       ENVDX(J) = ENVDX(J) - DXI * DF
                       ENVDY(J) = ENVDY(J) - DYI * DF
                       ENVDZ(J) = ENVDZ(J) - DZI * DF
                       ! protein(J) and ligand(I)
                    ELSE IF(JBL == 1.AND.IBL >= LSTRT) THEN
                       ENVDX(I) = ENVDX(I) + DXI * DF
                       ENVDY(I) = ENVDY(I) + DYI * DF
                       ENVDZ(I) = ENVDZ(I) + DZI * DF
                       ! Within ligand(I)
                    ELSE IF(IBL >= LSTRT.AND.IBL == JBL) THEN
                       ENVDX(I) = ENVDX(I) + DXI * DF
                       ENVDY(I) = ENVDY(I) + DYI * DF
                       ENVDZ(I) = ENVDZ(I) + DZI * DF
                       ENVDX(J) = ENVDX(J) - DXI * DF
                       ENVDY(J) = ENVDY(J) - DYI * DF
                       ENVDZ(J) = ENVDZ(J) - DZI * DF
                    ENDIF
                 ENDIF
! LDM
                 IF (JBL  <  IBL) THEN
                    KK=IBL
                    IBL=JBL
                    JBL=KK
                 ENDIF
                 KK=IBL+JBL*(JBL-1)/2
                 COEF = BLCOEP(KK)
                 IF (QPRNTV) THEN
                    IF (IBL == 1 .OR. JBL == 1 .OR. IBL == JBL) THEN
                       VBELEC(MAX(IBL,JBL)) = VBELEC(MAX(IBL,JBL)) + EELPR
                       VBVDW(MAX(IBL,JBL)) = VBVDW(MAX(IBL,JBL)) + ENBPR
                    ENDIF
                 ENDIF
!ldm
                 IF (QLDM .or. QLMC) THEN
                    ! first row or diagonal elements exclude (1,1).
                    IF((IBL == 1.AND.JBL >= LSTRT) &
                         .OR.( (JBL == 1.AND.IBL >= LSTRT) &
                         .or.  (IBL >= LSTRT.AND.IBL == JBL)) ) THEN
                       FALPHA = ENBPR + EELPR
#if KEY_PERT==1
                       if(QPERT) FALPHA = FALPHA*PERTLAM  
#endif
                       LAGMUL = LAGMUL + FALPHA
                       BIFLAM(JBL) = BIFLAM(JBL) + FALPHA
                    ENDIF
                    IF(NRST == 2)THEN
                       ! within ligand i
                       IF(IBL >= LSTRT.AND.IBL == JBL)THEN
                          FALPHA=ENBPR + EELPR
#if KEY_PERT==1
                          if(QPERT) FALPHA = FALPHA*PERTLAM  
#endif
                          BFRST(IBL) = BFRST(IBL) + FALPHA
                          ! between ligand I & virtual atoms
                       ELSE IF((IBL == 2.AND.JBL >= LSTRT).OR. &
                            (JBL == 2.AND.IBL >= LSTRT))THEN
                          FALPHA=ENBPR + EELPR
#if KEY_PERT==1
                          if(QPERT) FALPHA = FALPHA*PERTLAM  
#endif
                          BFRST(MAX(JBL,IBL)) = BFRST(MAX(JBL,IBL)) + FALPHA
                       ENDIF
                    ENDIF
                 ENDIF
! LDM
#if KEY_DOCK==1
                 IF(QDOCK) THEN
                    ENBPR=ENBPR*COEF*0.5*(DOCFI + DOCFJ)
                    EELPR=EELPR*COEF*0.5*(DOCFI + DOCFJ)
                 ELSE
#endif /*  DOCK*/
#if KEY_SCCDFTB==1
                    if(qsccb) then
                       dvdl0=enbpr*cdvdl(kk)
                       dvdl1=eelpr*cdvdl(kk)
                    endif
#endif 
                    ENBPR=ENBPR*COEF
                    EELPR=EELPR*COEF
#if KEY_DOCK==1
                 ENDIF
#endif 
                 DF=DF*COEF
#if ENBONDA_LSECD==1 /*lsecd17*/
                 IF (LSECD) DDF=DDF*COEF
#endif /*  (lsecd17)*/
#if ENBONDA_B_FLAG==1
              ENDIF                                           
#endif
              !
#if ENBONDA_B_FLAG==1
              IF (.NOT. NOFORC) THEN                          
#endif
#endif /*  BLOCK*/
#if ENBONDA_FORCES==1 /*forces_main*/
                 DXIT=DXI*DF
                 DYIT=DYI*DF
                 DZIT=DZI*DF
#if KEY_FLUCQ==1
                 ! Add in interaction energy to FlucQ arrays
                 IF (QFLUC) THEN
                    FQCFOR(I)=FQCFOR(I)+EELPR
                    FQCFOR(J)=FQCFOR(J)+EELPR
                 ENDIF
#endif 

#if ENBONDA_BLOCK==1
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
#endif 
                    FDXI=FDXI+DXIT
                    FDYI=FDYI+DYIT
                    FDZI=FDZI+DZIT
                    DX(J)=DX(J)-DXIT
                    DY(J)=DY(J)-DYIT
                    DZ(J)=DZ(J)-DZIT
#if ENBONDA_BLOCK==1
#if KEY_DOCK==1
                 ENDIF
#endif 
#endif 
#if ENBONDA_FOURD==1 /*4daddfj*/
#if ENBONDA_F_FLAG==1
                 IF(DIM4) THEN                                 
#endif
                    DFDIT=DFDIMI*DF
                    FDFDI=FDFDI+DFDIT
                    DFDIM(J)=DFDIM(J)-DFDIT
#if ENBONDA_F_FLAG==1
                 ENDIF                                         
#endif
#endif /* (4daddfj)*/
                 ! SAPATEL
#if KEY_CHEQ==1
                 IF (QCG) THEN
                    IF (S < C2OFNB) THEN
                       IF (QPARTBIN(I) /= 0) DCH(I)=DCH(I)+HIJ*CG(J)
                       IF (QPARTBIN(J) /= 0) DCH(J)=DCH(J)+HIJ*CG(I)
                    ENDIF
                 ENDIF !qcg
#endif 
                 ! SAPATEL
                 !     to calculate-second-derivatives
                 !
#if ENBONDA_LSECD==1 /*lsecd_main*/
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
#endif /*  (forces_main)*/
#if ENBONDA_B_FLAG==1 && ENBONDA_BLOCK==1
              ENDIF  ! (.NOT.NOFORC)                  
#endif
              ETEMP1=ETEMP1+ENBPR
              ETEMP2=ETEMP2+EELPR
#if KEY_SCCDFTB==1
              if(qsccb) then
                 dvdltmp1=dvdltmp1+dvdl0
                 dvdltmp2=dvdltmp2+dvdl1
              endif
#endif 
              !
#if ENBONDA_ANALYS==1 /*analys*/
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
#if KEY_GCMC==1
30         CONTINUE
#endif 

241        CONTINUE
        ENDDO  ! JPR
        !
        !     RESTORE ITH COMPONENT OF FORCE IN THE ARRAY
        !
#if ENBONDA_B_FLAG==1 && ENBONDA_BLOCK==1
        IF (.NOT. NOFORC) THEN                      
#endif
#if ENBONDA_FORCES==1 /*forces_set*/
           DX(I)=FDXI
           DY(I)=FDYI
           DZ(I)=FDZI
#if ENBONDA_FOURD==1 /*4daddfi*/
#if ENBONDA_F_FLAG==1
           IF(DIM4) THEN                                     
#endif
              DFDIM(I)=FDFDI
#if ENBONDA_F_FLAG==1
           ENDIF                                             
#endif
#endif /* (4daddfi)*/
#endif /* (forces_set)*/
#if ENBONDA_B_FLAG==1 && ENBONDA_BLOCK==1
        ENDIF                                       
#endif
        !
        ENB=ENB+ETEMP1
        EEL=EEL+ETEMP2
#if KEY_SCCDFTB==1
        if(qsccb) then
           if(idxnbd == 0) dvdlv=dvdlv+dvdltmp1
           if(idxnbd == 1) dvdlv1=dvdlv1+dvdltmp1
           dvdle=dvdle+dvdltmp2
        endif
#endif 
        !
     ENDIF  ! (NPR > 0)
  ENDDO  ! I
