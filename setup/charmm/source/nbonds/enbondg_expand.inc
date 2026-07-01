!=======================================================================
!   Main loop begin
!=======================================================================
!     loop over pairs of groups in groups list
!
      NB=0
      ITEMP=0
      DO IRS=IFRSTG,NGRP
        NPR=INBLOG(IRS)-ITEMP
        ITEMP=INBLOG(IRS)
        IS=IGPBS(IRS)+1
        IQ=IGPBS(IRS+1)
        NI=IQ-IS+1
#if KEY_NOST2 != 1
        LST2=IGPTYP(IRS) == 3
#endif 
!
        DO JRSPR=1,NPR
          NB=NB+1
          JRS=JNBG(NB)
          LEXCL=(JRS < 0)
          JRS=IABS(JRS)
          JSS=IGPBS(JRS)+1
          JQ=IGPBS(JRS+1)
          NJ=JQ-JSS+1
          ETEMP1=ZERO
          ETEMP2=ZERO
          FUNCT=ONE
!
!     PROCESS THIS GROUP PAIR
!
! ##IF MTS .not.NOST2  (skip_tests)
#if KEY_MTS == 1 || KEY_NOST2 != 1          
          SKIP=.FALSE.
#if KEY_MTS==1
!---- LONG-SHORT RANGE MTS METHOD
!     do the exclusions only in the short range (ignore distances)
          SKIP=(SLFG .AND. LEXCL .AND. SLFG2)
#endif 
#if KEY_NOST2 != 1
!     Use ST2 interaction for this pair
          IF(LST2 .AND. (IGPTYP(JRS) == 3)) SKIP=.TRUE.
#endif 
          IF(SKIP) THEN
             CONTINUE
          ELSE IF (LEXCL) THEN
#else /*  (skip_tests)*/
          IF (LEXCL) THEN
#endif /* (skip_tests)*/
!
!     check atom exclusion list for exclusions
            IF(IS > 1) THEN
              NXIMAX=IBLO14(IS-1)
            ELSE
              NXIMAX=0
            ENDIF
            DO I=IS,IQ
              NXI=NXIMAX+1
              NXIMAX=IBLO14(I)
              CGT=CGF*CG(I)
              IF (IS == JSS) THEN
                JSX=I+1
              ELSE
                JSX=JSS
              ENDIF
              DO J=JSX,JQ
!
! namkh 01/20/04
#if KEY_GAMESS==1 || KEY_GAMESSUK==1 || KEY_MNDO97==1 || KEY_SQUANTM==1 || KEY_QCHEM==1 || KEY_QTURBO==1 || KEY_G09==1
                 if(qmused)then
                 IF((ABS(IGMSEL(I)) == 1.OR.ABS(IGMSEL(I)) == 2) .AND. &
                    (ABS(IGMSEL(J)) == 1.OR.ABS(IGMSEL(J)) == 2) &
                     .AND. QGMREM ) CYCLE
                 endif
#endif 
!
!
! See if pair is in excluded list
!
                CGP = CGT*CG(J)
                INBX=IABS(INB14(NXI))
                DO WHILE(NXI <= NXIMAX .AND. J > INBX)
                  NXI=NXI+1
                  INBX=IABS(INB14(NXI))
                ENDDO
                IF(NXI > NXIMAX) THEN
                  LEX14=0
                ELSE IF(J == INB14(NXI)) THEN  ! exclusion found
                  LEX14=-1
                ELSE
                  IF(J == INBX) THEN
                    LEX14=MAXCNX   ! it's a 1-4 pair
                    CGP = CGP*E14FAC
                  ELSE
                    LEX14=0
                  ENDIF
                ENDIF
                IF(LEX14 >= 0) THEN
                  DXI=X(I)-X(J)
                  DYI=Y(I)-Y(J)
                  DZI=Z(I)-Z(J)
#if KEY_PBOUND==1 /*pbound*/
#if ENBONDG_P_FLAG==1
                  If(qBoun) then                            
#endif
                     If(qCUBoun.or.qTOBoun) then
                        DXI = BOXINV * DXI
                        DYI = BOYINV * DYI
                        DZI = BOZINV * DZI
                        dxi = dxi - nint(dxi)
                        dyi = dyi - nint(dyi)
                        dzi = dzi - nint(dzi)
!!$                        IF(DXI >   HALF) DXI = DXI - ONE
!!$                        IF(DXI <  -HALF) DXI = DXI + ONE
!!$                        IF(DYI >   HALF) DYI = DYI - ONE
!!$                        IF(DYI <  -HALF) DYI = DYI + ONE
!!$                        IF(DZI >   HALF) DZI = DZI - ONE
!!$                        IF(DZI <  -HALF) DZI = DZI + ONE
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
#if ENBONDG_P_FLAG==1
                  Endif                                     
#endif
#endif /*      (pbound)*/
                  S=DXI*DXI+DYI*DYI+DZI*DZI
                  R2=ONE/S
#if KEY_NBIPS==1
              IF(QIPS)THEN
!WXW Long range potential using isotropic periodic sum (IPS)
                DOIPS=LEX14 <= 0
                U2=S*RIPS2R
              ENDIF
#endif 
!--------------------
#if ENBONDG_LRSW==1 /*elec_opt*/
#if ENBONDG_E_FLAG==1
                  IF (LRSW) THEN                            
#endif
                    EELPR=CGP*R2
                    DF=R2*MINTWO*EELPR
#if ENBONDG_LSECD==1
                    IF(LSECD) DDF=SIX*EELPR*R2*R2           
#endif
#endif /*  (elec_opt)*/
!--------------------
#if ENBONDG_LCSW==1 /*elec_opt*/
#if ENBONDG_E_FLAG==1
                  ELSE IF(LCSW) THEN                        
#endif
                    EELPR=CGP*SQRT(R2)
                    DF=-R2*EELPR
#if ENBONDG_LSECD==1
                    IF(LSECD) DDF=TWO*EELPR*R2*R2           
#endif
#endif /*  (elec_opt)*/
!--------------------
#if ENBONDG_LEWALD==1 /*elec_opt*/
#if ENBONDG_E_FLAG==1
                  ELSE IF(LEWALD) THEN                      
#endif
                    RS=SQRT(S)
                    CALL ERFCD(RS,KAPPA,ERFCX,DRFC,ERFMOD)
                    IF(LEX14 > 0) ERFCX=ERFCX + E14M1
                    CGIJ = CGT*CG(J)
                    EELPR=CGIJ*ERFCX*RS*R2
                    DF=-(CGIJ*DRFC + EELPR)*R2
#if ENBONDG_LSECD==1
                    IF(LSECD) DDF=ZERO                      
#endif
!
#endif /*  (elec_opt)*/
!--------------------
#if KEY_NBIPS==1 /*nbips_elecs*/
#if ENBONDG_REIPS==1 /*elec_opt*/
#if ENBONDG_E_FLAG==1
                  ELSE IF(REIPS) THEN                                
#endif
                   IF(DOIPS) THEN
!  Electrostatic IPS
!   etr1=1/r2+r2*(a1+r2*(a2+r2*(a3+r2*(a4+r2*(a5+a6*r2)))))
!   detr1/dr*r1=-1/r2+r2*(d1+r2*(d2+r2*(d3+r2*(d4+r2*(d5+d6*r2)))))
!
            PE=ONE/U2+U2*(AIPSE(1)+U2*(AIPSE(2)+U2*(AIPSE(3) &
             +U2*(AIPSE(4)+U2*(AIPSE(5)+U2*AIPSE(6))))))-PIPSEC
            DPE=-TWO/U2+U2*(BIPSE(1)+U2*(BIPSE(2)+U2*(BIPSE(3) &
             +U2*(BIPSE(4)+U2*(BIPSE(5)+U2*BIPSE(6))))))
                    ENEP=CGP*RIPS2R
                    EELPR=ENEP*PE
                    DF=ENEP*DPE*R2
#if ENBONDG_LSECD==1 /*lsecd2*/
                    IF(LSECD) THEN
                       DDPE=SIX/U2+U2*(BBIPSE(1)+U2*(BBIPSE(2)+U2*(BBIPSE(3) &
                               +U2*(BBIPSE(4)+U2*(BBIPSE(5)+U2*BBIPSE(6))))))
                       DDF=ENEP*DDPE*R2*R2
                    ENDIF
#endif /*  (lsecd2)*/
                   ELSE
                    EELPR=ZERO
                    DF=ZERO
                   ENDIF
!
#endif /*  (elec_opt)*/
!--------------------
#if ENBONDG_CEIPS==1 /*elec_opt*/
#if ENBONDG_E_FLAG==1
                  ELSE IF(CEIPS) THEN                                
#endif
                   IF(DOIPS) THEN
!  Electrostatic IPS
!   etr1=1/r+r2*(a1+r2*(a2+r2*(a3+r2*(a4+r2*(a5+a6*r2)))))
!   detr1/dr*r=-1/r+r2*(d1+r2*(d2+r2*(d3+r2*(d4+r2*(d5+d6*r2)))))
!
            U1=SQRT(U2)
            PE=ONE/U1+U2*(AIPSE(1)+U2*(AIPSE(2)+U2*(AIPSE(3) &
             +U2*(AIPSE(4)+U2*(AIPSE(5)+U2*AIPSE(6))))))
            DPE=-ONE/U1+U2*(BIPSE(1)+U2*(BIPSE(2)+U2*(BIPSE(3) &
             +U2*(BIPSE(4)+U2*(BIPSE(5)+U2*BIPSE(6))))))
            ENEP=CGP*RIPSR
            EELPR=ENEP*(PE-PIPSEC)
            DF=ENEP*DPE*R2
#if ENBONDG_LSECD==1 /*lsecd2*/
            IF(LSECD) THEN
               DDPE=TWO/U1+U2*(BBIPSE(1)+U2*(BBIPSE(2)+U2*(BBIPSE(3) &
                        +U2*(BBIPSE(4)+U2*(BBIPSE(5)+U2*BBIPSE(6))))))
               DDF=ENEP*DDPE*R2*R2
            ENDIF
#endif /*  (lsecd2)*/
                   ELSE
                    EELPR=ZERO
                    DF=ZERO
                   ENDIF
!
#endif /*  (elec_opt)*/
#endif /*  (nbips_elecs)*/
!--------------------
#if ENBONDG_LELSE==1 /*elec_opt*/
#if ENBONDG_E_FLAG==1
                  ELSE                                      
#endif
                    EELPR=ZERO
                    DF=ZERO
#if ENBONDG_LSECD==1
                    IF(LSECD) DDF=ZERO                      
#endif
#if ENBONDG_E_FLAG==1
                  ENDIF                                     
#endif
#endif /*  (elec_opt)*/
!--------------------
!
#if ENBONDG_LVGRP==1 /*lvgrp1*/
#if ENBONDG_V_FLAG==1
                  IF(LVGRP) THEN                            
#endif
                    I1=ITC(IAC(I))
                    J1=ITC(IAC(J))
                    IC=IOFF(MAX(I1,J1))+MIN(I1,J1)+LEX14
#if KEY_WCA==1
                    IF (S  >  CNBA(IC)) THEN
                      SIG2=RSCLF(I)*RSCLF(J)*CNBA(IC)*R2
                      SIG6=SIG2*SIG2*SIG2
                      SIG12=SIG6*SIG6
                      ENBPR=CNBB(IC)*(SIG12-SIG6-SIG6)
                      EGLJTMP = CNBB(IC)*TWELVE*(SIG6-SIG12)*R2
                      EPRPL = 0.0D0
                      EGLJRPL = 0.0D0
                    ELSE
                      ENBPR=-CNBB(IC)
                      EGLJTMP = 0.0D0
                      IF (LLSOFT) THEN
                        TMP = CNBA(IC)*(1.0-SCVDWCUTR)*(1.0-SCVDWCUTR)
!      write(*,*) '&&&', I, J, SCVDWCUTR
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
                        SIG2=RSCLF(I)*RSCLF(J)*CNBA(IC)*R2
                        SIG6=SIG2*SIG2*SIG2
                        SIG12=SIG6*SIG6
                        EPRPL=CNBB(IC)*((SIG12-SIG6-SIG6)+1.0D0)
                        EGLJRPL=CNBB(IC)*TWELVE*(SIG6-SIG12)*R2
                      ENDIF
                    ENDIF
#else /**/
                    SIG2=RSCLF(I)*RSCLF(J)*CNBA(IC)*R2
!                   use standard vdw function
                    SIG6=SIG2*SIG2*SIG2
                    SIG12=SIG6*SIG6
#endif 
                    IF(SIG2<RSMALL)THEN
                            ENBPR=ZERO
                    ELSE IF(DBEXP)THEN
                        !     VAN DER WAAL DOUBLE EXPONENTIAL POTENTIAL
                        !     E= A*EXP(alpha*R)-B*EXP(beta*R)
                        !
                        SIG=SQRT(SIG2)
#if KEY_NBIPS==1 /*nbips_vdw*/
                        AKR=AIPSVA(0)/SIG
                        EXPAR=EXP(-AKR)
                        ! energy
                        ENBPR=AIPSVA(1)*EXPAR
                        ! force
                        DPVA=BIPSVA(1)*EXPAR/SIG
                        CKR=AIPSVC(0)/SIG
                        EXPCR=EXP(-CKR)
                        ! energy
                        ENBPR=CNBB(IC)*(ENBPR-AIPSVC(1)*EXPCR)
                        ! force
                        DPVC=BIPSVC(1)*EXPCR/SIG
                        DF=DF+CNBB(IC)*(DPVA-DPVC)*R2
#if ENBONDA_LSECD==1 /*lsecd1*/
                        IF(LSECD) THEN
                           DDF=DDF-CNBB(IC)*(BIPSVA(1)*EXPAR*AIPSVA(0)-BIPSVC(1)*EXPCR*AIPSVC(0))*R2*R2/SIG2
                        ENDIF
#endif /*  (lsecd1)*/

                    !WXW VDW potential using isotropic periodic sum (IPS)
                    ELSE IF (DBEXPIPS) THEN                              
                      !     VAN DER WAAL DOUBLE EXPONENTIAL IPS
                      !WXW VDW DE potential using isotropic periodic sum (IPS)
                      !WXW   1-4  interaction will be calculated at subroutine EEXIPS
                      SIG=SQRT(SIG2)
                      IF(DOIPS.AND.SRIPS*SIG>ONE)THEN
                        AKR=AIPSVA(0)/SIG
                        EXPAR=EXP(-AKR)
                        ! energy
                        ENBPR=AIPSVA(1)*(EXPAR+AIPSVA(4)/EXPAR+AIPSVA(5))
                        ! force 
                        DPVA=BIPSVA(1)*(EXPAR-AIPSVA(4)/EXPAR)/SIG
                        !
                        CKR=AIPSVC(0)/SIG
                        EXPCR=EXP(-CKR)
                        ! energy
                        ENBPR=CNBB(IC)*(ENBPR-AIPSVC(1)*(EXPCR+AIPSVC(4)/EXPCR+AIPSVC(5)))
                        ! force
                        DPVC=BIPSVC(1)*(EXPCR-AIPSVC(4)/EXPCR)/SIG
                        DF=DF+CNBB(IC)*(DPVA-DPVC)*R2
#if ENBONDA_LSECD==1 /*lsecd2*/
                        IF(LSECD) THEN
                           DDF=DDF+CNBB(IC)*(AIPSVA(0)*(BIPSVA(1)*(EXPAR+AIPSVA(4)/EXPAR)) &
                           -AIPSVC(0)*(BIPSVC(1)*(EXPCR+AIPSVC(4)/EXPCR)))*R2*R2/SIG2
                        ENDIF
#endif /*  (lsecd2)*/
                      ELSE 
                        ENBPR=ZERO
                      ENDIF
                    ELSE IF (DVIPS) THEN
!WXW VDW potential using isotropic periodic sum (IPS)
                      IF(DOIPS)THEN
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
                        ENEVA =CNBB(IC)*SIG12/U12R
                        ENEVC =-TWO*CNBB(IC)*SIG6/U6R
                        ENBPR=ENEVA*PVA+ENEVC*PVC
                        DF=DF+(ENEVA*DPVA+ENEVC*DPVC)*R2
#if ENBONDG_LSECD==1 /*lsecd2*/
                        IF(LSECD) THEN
                           DDPVC=42.0D0*U6R+U2*(BBIPSVC(1)+U2*(BBIPSVC(2)+U2*(BBIPSVC(3) &
                             +U2*(BBIPSVC(4)+U4*(BBIPSVC(5)+U4*BBIPSVC(6))))))
                           DDPVA=156.0D0*U12R+U2*(BBIPSVA(1)+U2*(BBIPSVA(2)+U2*(BBIPSVA(3) &
                             +U4*(BBIPSVA(4)+U4*(BBIPSVA(5)+U4*BBIPSVA(6))))))
                           DDF=DDF+(ENEVA*DDPVA+ENEVC*DDPVC)*R2*R2
                        ENDIF
#endif /*  (lsecd2)*/
                      ELSE
                       ENBPR=ZERO
                      ENDIF
#endif /* (nbips_vdw)*/
                    ELSE 
                      IF (.NOT.QETEN .AND. .NOT.QETSR) THEN
#if KEY_WCA==1
                        DF=DF+(EGLJTMP*WCA(I)*WCA(J)+EGLJRPL)
                        ENBPR=(ENBPR*WCA(I)*WCA(J)+EPRPL)
#if ENBONDG_LSECD==1
                        IF(LSECD) DDF=DDF+                       & 
                          CNBB(IC)*(156.*SIG12-84.0*SIG6)*R2*R2 
#endif
#else /**/
                        DF=DF+CNBB(IC)*TWELVE*R2*(SIG6-SIG12)
                        ENBPR=CNBB(IC)*(SIG12-SIG6-SIG6)
#if KEY_LJPME==1
            if(qljpme) call ljpme_dir(ljc6(I), ljc6(j), s, enbpr, df, CNBA(IC), CNBB(IC), .false.)
#endif

#if ENBONDG_LSECD==1
                        IF(LSECD) DDF=DDF+                       & 
                            CNBB(IC)*(156.*SIG12-84.0*SIG6)*R2*R2 
#endif
#endif 
                      ELSE IF (QETSR) THEN 
                         FAC=FOUR/(NINE*SIG2)
                         SWTMP=(FAC*FAC)**3
                         ENBPR=(CNBB(IC)*(THIRTN*SIG12-NINE*TWO*SIG6 &
                              *SIG2*SIG2+FOUR*SIG6)) &
                              / (ONE+SWTMP)
                         ! DF = 1/R dV/dr
                         DF=DF+CNBB(IC)*R2*TWELVE*(FIFTN*SIG6*SIG2*SIG2 &
                              -THIRTN*SIG12-TWO*SIG6) &
                              / (ONE+SWTMP) &
                              - R2*ENBPR*TWELVE*SWTMP &
                              /(ONE+SWTMP)
                         IF(LSECD) THEN
                            ODF=CNBB(IC)*R2*TWELVE*(FIFTN*SIG6*SIG2*SIG2 &
                                 -THIRTN*SIG12-TWO*SIG6)

                            DDF=DDF+(CNBB(IC)*TWELVE*(169.0*SIG12- &
                                 165.0*SIG6*SIG2*SIG2+14.0*SIG6)*R2*R2) &
                                 / ( ONE+SWTMP) &
                                 - TWO*R2*TWELVE*ODF*SWTMP/((ONE+SWTMP)*(ONE+SWTMP)) &
                                 - R2*R2*TWELVE*ENBPR*SWTMP/(ONE+SWTMP) &
                                 * (MINTWO*TWELVE*SWTMP/(ONE+SWTMP) + ELEVEN)
                         ENDIF
                      ELSE ! QETEN
                         DF=DF+CNBB(IC)*TWELVE*R2* &
                              (FIFTN*SIG6*SIG2*SIG2-THIRTN*SIG12-TWO*SIG6)
#if ENBONDG_LSECD==1
                         IF(LSECD) DDF=DDF+                       & 
#endif          
#if ENBONDG_LSECD==1
                              CNBB(IC)*TWELVE*(169.0*SIG12-165.0*   & 
#endif          
#if ENBONDG_LSECD==1
                              SIG6*SIG2*SIG2+14.0*SIG2)*R2*R2      
#endif          
                         ENBPR=(CNBB(IC)*(THIRTN*SIG12-NINE*TWO*SIG6* &
                              SIG2*SIG2+FOUR*SIG6))
                      ENDIF
                    ENDIF         /* SIG */           
#if ENBONDG_V_FLAG==1
       ENDIF                                     
#endif
#endif /* (lvgrp1)*/

!
#if KEY_BLOCK==1
                  IF (QBLOCK) THEN
                     IBL = IBLCKP(I)
                     JBL = IBLCKP(J)
                     KK=MAX(IBL,JBL)
                     KK=KK*(KK-1)/2+MIN(IBL,JBL)
                     COEF = BLCOEP(KK)
                     EELPR=EELPR*COEF
                     ENBPR=ENBPR*COEF
                     DF=DF*COEF
#if ENBONDG_LSECD==1
                     IF(LSECD) DDF=DDF*COEF                 
#endif
                  ENDIF
!
                  IF (.NOT. NOFORC) THEN
#endif 
                     DXIT=DXI*DF
                     DYIT=DYI*DF
                     DZIT=DZI*DF
                     DX(I)=DX(I)+DXIT
                     DY(I)=DY(I)+DYIT
                     DZ(I)=DZ(I)+DZIT
                     DX(J)=DX(J)-DXIT
                     DY(J)=DY(J)-DYIT
                     DZ(J)=DZ(J)-DZIT
#if KEY_FLUCQ==1
! Add in this interaction to the FlucQ arrays
                     IF (QFLUC) THEN
                        FQCFOR(I)=FQCFOR(I)+EELPR
                        FQCFOR(J)=FQCFOR(J)+EELPR
                     ENDIF
#endif 
#if KEY_BLOCK==1
                  ENDIF
#endif 
                  ETEMP1=ETEMP1+ENBPR
                  ETEMP2=ETEMP2+EELPR
!
! Calculate second derivatives
#if ENBONDG_LSECD==1 /*lsecd_main1*/
                  IF(LSECD) THEN
!
                    DDF=DDF-DF*R2
!
                    AXX=DXI*DXI*DDF+DF
                    AYY=DYI*DYI*DDF+DF
                    AZZ=DZI*DZI*DDF+DF
                    AXY=DXI*DYI*DDF
                    AXZ=DXI*DZI*DDF
                    AYZ=DYI*DZI*DDF
!
#if KEY_DIMB==1
                    IF(QCMPCT) THEN
!                    CALL WRNDIE(-3,'<EGROUP>','EGROUP not yet ',
!     1                'implemented for compact second derivatives')
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
#if KEY_DIMB==1
                    ENDIF
#endif /*  DIMB*/
                  ENDIF
#endif /*  (lsecd_main1)*/
#if ENBONDG_ANALYS==1 /*analys1*/
                  IF(QECONTX) THEN
                    S=HALF*(ENBPR+EELPR)
                    ECONTX(I)=ECONTX(I)+S
                    ECONTX(J)=ECONTX(J)+S
                  ENDIF
#endif /*  (analys1)*/
!
                ENDIF
!
!!  XX  !!! namkh 01/20/04
!!  XX  !!  ##IF GAMESS GAMESSUK MNDO97 SQUANTM QCHEM QTURBO G09
!!  XX  !!                END IF
!!  XX  !!  ##ENDIF
!
              ENDDO
            ENDDO
          ELSE
!
#if KEY_NBIPS==1
            IF(QIPS)THEN
              SCENT=ZERO
            ELSE
#endif 
! Do switching function interaction
            DXIC=XCENT(IRS)-XCENT(JRS)
            DYIC=YCENT(IRS)-YCENT(JRS)
            DZIC=ZCENT(IRS)-ZCENT(JRS)
#if KEY_PBOUND==1 /*pbound*/
#if ENBONDG_P_FLAG==1
            If(qBoun) then                                  
#endif
               If(qCUBoun.or.qTOBoun) then
                  DXIC = BOXINV * DXIC
                  DYIC = BOYINV * DYIC
                  DZIC = BOZINV * DZIC
                  dxic = dxic - nint(dxic)
                  dyic = dyic - nint(dyic)
                  dzic = dzic - nint(dzic)
!!$                  IF(DXIC >   HALF) DXIC = DXIC - ONE
!!$                  IF(DXIC <  -HALF) DXIC = DXIC + ONE
!!$                  IF(DYIC >   HALF) DYIC = DYIC - ONE
!!$                  IF(DYIC <  -HALF) DYIC = DYIC + ONE
!!$                  IF(DZIC >   HALF) DZIC = DZIC - ONE
!!$                  IF(DZIC <  -HALF) DZIC = DZIC + ONE
                  If (qTOBoun) Then
                     CORR = HALF * AINT ( R75 * (ABS(DXIC) + &
                                                 ABS(DYIC) + &
                                                 ABS(DZIC)))
                     DXIC = DXIC - SIGN( CORR,  DXIC  )
                     DYIC = DYIC - SIGN( CORR,  DYIC  )
                     DZIC = DZIC - SIGN( CORR,  DZIC  )
                  Endif
                  DXIC = XSIZE * DXIC
                  DYIC = YSIZE * DYIC
                  DZIC = ZSIZE * DZIC
               Else
                  Call PBMove(DXIC, DYIC, DZIC)
               Endif
#if ENBONDG_P_FLAG==1
            Endif                                           
#endif
#endif /*      (pbound)*/
            SCENT=DXIC*DXIC+DYIC*DYIC+DZIC*DZIC
#if KEY_NBIPS==1
            ENDIF                                
#endif
            IF(SCENT < C2OFNB) THEN
!
#if KEY_MTS==1
!---- LONG-SHORT RANGE MTS METHOD
             SKIP=.FALSE.
             IF(SLFG) THEN
                SWFE = ONE
                SWF  = ONE
                IF(SCENT <= RSHL2) THEN
                   SKIP=SLFG2
                ELSE IF(SCENT >= RSCUT2) THEN
                   SKIP=SLFG1
                ELSE
                   RR1 = SQRT(SCENT)
                   RR2 = (RR1-RSHL)/RHEAL
                   RR3 = ONE-RR2*RR2*RR2*(6.0*RR2*RR2-15.0*RR2+10.0)
                   IF(SLFG1) THEN
                      SWFE=ZERO
                      SWF = RR3
                   ENDIF
                   IF(SLFG2) SWF = ONE-RR3
                ENDIF
             ENDIF
             IF(.NOT.SKIP) THEN
#endif 
!
              LSWITR=(SCENT > C2ONNB)
              IF(LSWITR) THEN
                RIJL=C2ONNB-SCENT
                RIJU=C2OFNB-SCENT
                FUNCT=RIJU*RIJU*(RIJU-3*RIJL)*RUL3
                DFN=RIJL*RIJU*RUL12
#if ENBONDG_LSECD==1
                IF(LSECD) DDFN=MINTWO*(RIJU+RIJL)*RUL12     
#endif
              ELSE
                FUNCT=ONE
              ENDIF
              IF(LFSWT) THEN
                 ESWADD=-QCENT(IRS)*QCENT(JRS)
#if KEY_BLOCK==1
! Assume all atoms of a group are in the same block.
                 IF (QBLOCK) THEN
                    IBL = IBLCKP(IS)
                    JBL = IBLCKP(JSS)
                    KK=MAX(IBL,JBL)
                    KK=KK*(KK-1)/2+MIN(IBL,JBL)
                    ETEMP2 = ETEMP2 * BLCOEP(KK)
                 ENDIF
#endif 
              ENDIF
!
              DO I=IS,IQ
                CGT=CGF*CG(I)
                DO J=JSS,JQ
!
! namkh 01/20/04
#if KEY_GAMESS==1 || KEY_GAMESSUK==1 || KEY_MNDO97==1 || KEY_SQUANTM==1 || KEY_QCHEM==1 || KEY_QTURBO==1 || KEY_G09==1
                  if(qmused)then
                  IF((ABS(IGMSEL(I)) == 1.OR.ABS(IGMSEL(I)) == 2) .AND. &
                     (ABS(IGMSEL(J)) == 1.OR.ABS(IGMSEL(J)) == 2) &
                      .AND.QGMREM  ) CYCLE
                  endif
#endif 
!
                  DXI=X(I)-X(J)
                  DYI=Y(I)-Y(J)
                  DZI=Z(I)-Z(J)
#if KEY_PBOUND==1 /*pbound*/
#if ENBONDG_P_FLAG==1
                  If(qBoun) then                            
#endif
                     If(qCUBoun.or.qTOBoun) then
                        DXI = BOXINV * DXI
                        DYI = BOYINV * DYI
                        DZI = BOZINV * DZI
                        dxi = dxi - nint(dxi)
                        dyi = dyi - nint(dyi)
                        dzi = dzi - nint(dzi)
!!$                        IF(DXI >   HALF) DXI = DXI - ONE
!!$                        IF(DXI <  -HALF) DXI = DXI + ONE
!!$                        IF(DYI >   HALF) DYI = DYI - ONE
!!$                        IF(DYI <  -HALF) DYI = DYI + ONE
!!$                        IF(DZI >   HALF) DZI = DZI - ONE
!!$                        IF(DZI <  -HALF) DZI = DZI + ONE
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
#if ENBONDG_P_FLAG==1
                  Endif                                     
#endif
#endif /*      (pbound)*/
                  S=DXI*DXI+DYI*DYI+DZI*DZI
                  R2=ONE/S
#if KEY_NBIPS==1
              IF(QIPS)THEN
!WXW Long range potential using isotropic periodic sum (IPS)
                DOIPS=S < RIPS2
                U2=S*RIPS2R
              ENDIF
#endif 
!
!--------------------
#if ENBONDG_LRSW==1 /*elec_opt*/
#if ENBONDG_E_FLAG==1
                  IF (LRSW) THEN                            
#endif
                    EELPR=CGT*CG(J)*R2
                    DF=-R2*(TWO*EELPR)
#if ENBONDG_LSECD==1
                    IF(LSECD) DDF=SIX*EELPR*R2*R2           
#endif
#endif /*  (elec_opt)*/
!--------------------
#if ENBONDG_LCSW==1 /*elec_opt*/
#if ENBONDG_E_FLAG==1
                  ELSE IF(LCSW) THEN                        
#endif
                    EELPR=CGT*CG(J)*SQRT(R2)
                    DF=-R2*EELPR
#if ENBONDG_LSECD==1
                    IF(LSECD) DDF=TWO*EELPR*R2*R2           
#endif
#endif /*  (elec_opt)*/
!--------------------
#if ENBONDG_LEWALD==1 /*elec_opt*/
#if ENBONDG_E_FLAG==1
                  ELSE IF(LEWALD) THEN                      
#endif
                    RS=SQRT(S)
                    CALL ERFCD(RS,KAPPA,ERFCX,DRFC,ERFMOD)
                    CGIJ = CGT*CG(J)
                    EELPR=CGIJ*ERFCX*RS*R2
                    DF=-(CGIJ*DRFC + EELPR)*R2
#if ENBONDG_LSECD==1
                    IF(LSECD) DDF=ZERO                      
#endif
!
#endif /*  (elec_opt)*/
!--------------------
#if KEY_NBIPS==1 /*nbips_elecs*/
#if ENBONDG_REIPS==1 /*elec_opt*/
#if ENBONDG_E_FLAG==1
                  ELSE IF(REIPS) THEN                                
#endif
                   IF(DOIPS) THEN
!  Electrostatic IPS
!   etr1=1/r2+r2*(a1+r2*(a2+r2*(a3+r2*(a4+r2*(a5+a6*r2)))))
!   detr1/dr*r1=-1/r2+r2*(d1+r2*(d2+r2*(d3+r2*(d4+r2*(d5+d6*r2)))))
!
            PE=ONE/U2+U2*(AIPSE(1)+U2*(AIPSE(2)+U2*(AIPSE(3) &
             +U2*(AIPSE(4)+U2*(AIPSE(5)+U2*AIPSE(6))))))-PIPSEC
            DPE=-TWO/U2+U2*(BIPSE(1)+U2*(BIPSE(2)+U2*(BIPSE(3) &
             +U2*(BIPSE(4)+U2*(BIPSE(5)+U2*BIPSE(6))))))
                    ENEP=CGT*CG(J)*RIPS2R
                    EELPR=ENEP*PE
                    DF=ENEP*DPE*R2
#if ENBONDG_LSECD==1 /*lsecd2*/
                          IF(LSECD) THEN
                             DDPE=SIX/U2+U2*(BBIPSE(1)+U2*(BBIPSE(2)+U2*(BBIPSE(3) &
                               +U2*(BBIPSE(4)+U2*(BBIPSE(5)+U2*BBIPSE(6))))))
                             DDF=DDF+ENEP*DDPE*R2*R2
                          ENDIF
#endif /*  (lsecd2)*/
                   ELSE
                    EELPR=ZERO
                    DF=ZERO
                   ENDIF
!
#endif /*  (elec_opt)*/
!--------------------
#if ENBONDG_CEIPS==1 /*elec_opt*/
#if ENBONDG_E_FLAG==1
                  ELSE IF(CEIPS) THEN                                
#endif
                   IF(DOIPS) THEN
!  Electrostatic IPS
!   etr1=1/r+r2*(a1+r2*(a2+r2*(a3+r2*(a4+r2*(a5+a6*r2)))))
!   detr1/dr*r=-1/r+r2*(d1+r2*(d2+r2*(d3+r2*(d4+r2*(d5+d6*r2)))))
!
            U1=SQRT(U2)
            PE=ONE/U1+U2*(AIPSE(1)+U2*(AIPSE(2)+U2*(AIPSE(3) &
             +U2*(AIPSE(4)+U2*(AIPSE(5)+U2*AIPSE(6))))))
            DPE=-ONE/U1+U2*(BIPSE(1)+U2*(BIPSE(2)+U2*(BIPSE(3) &
             +U2*(BIPSE(4)+U2*(BIPSE(5)+U2*BIPSE(6))))))
            ENEP=CGT*CG(J)*RIPSR
            EELPR=ENEP*(PE-PIPSEC)
            DF=ENEP*DPE*R2
#if ENBONDG_LSECD==1 /*lsecd2*/
                          IF(LSECD) THEN
                             DDPE=TWO/U1+U2*(BBIPSE(1)+U2*(BBIPSE(2)+U2*(BBIPSE(3) &
                               +U2*(BBIPSE(4)+U2*(BBIPSE(5)+U2*BBIPSE(6))))))
                             DDF=DDF+ENEP*DDPE*R2*R2
                          ENDIF
#endif /*  (lsecd2)*/
                   ELSE
                    EELPR=ZERO
                    DF=ZERO
                   ENDIF
!
#endif /*  (elec_opt)*/
#endif /*  (nbips_elecs)*/
!--------------------
#if ENBONDG_LELSE==1 /*elec_opt*/
#if ENBONDG_E_FLAG==1
                  ELSE                                      
#endif
                    EELPR=ZERO
                    DF=ZERO
#if ENBONDG_LSECD==1
                    IF(LSECD) DDF=ZERO                      
#endif
#if ENBONDG_E_FLAG==1
                  ENDIF                                     
#endif
#endif /*  (elec_opt)*/
!--------------------
!
#if ENBONDG_LVGRP==1 /*lvgrp2*/
#if ENBONDG_V_FLAG==1
                  IF(LVGRP) THEN                            
#endif
                    I1=ITC(IAC(I))
                    J1=ITC(IAC(J))
                    IC=IOFF(MAX(I1,J1))+MIN(I1,J1)
#if KEY_WCA==1
                    IF (S  >  CNBA(IC)) THEN
                      SIG2=RSCLF(I)*RSCLF(J)*CNBA(IC)*R2
                      SIG6=SIG2*SIG2*SIG2
                      SIG12=SIG6*SIG6
                      ENBPR=CNBB(IC)*(SIG12-SIG6-SIG6)
                      EGLJTMP = CNBB(IC)*TWELVE*(SIG6-SIG12)*R2
                      EPRPL = 0.0D0
                      EGLJRPL = 0.0D0
                    ELSE
                      ENBPR=-CNBB(IC)
                      EGLJTMP = 0.0D0
                      IF (LLSOFT) THEN
                        TMP = CNBA(IC)*(1.0-SCVDWCUTR)*(1.0-SCVDWCUTR)
!      write(*,*) '$$$', I, J, SCVDWCUTR
                        IF (S  <  (CNBA(IC)-TMP)) THEN
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
                        SIG2=RSCLF(I)*RSCLF(J)*CNBA(IC)*R2
                        SIG6=SIG2*SIG2*SIG2
                        SIG12=SIG6*SIG6
                        EPRPL=CNBB(IC)*((SIG12-SIG6-SIG6)+1.0D0)
                        EGLJRPL=CNBB(IC)*TWELVE*(SIG6-SIG12)*R2
                      ENDIF
                    ENDIF
#else /**/
                    SIG2=RSCLF(I)*RSCLF(J)*CNBA(IC)*R2
!                   use standard vdw function
                    SIG6=SIG2*SIG2*SIG2
                    SIG12=SIG6*SIG6
#endif 
                    IF(SIG2<RSMALL)THEN
                            ENBPR=ZERO
                    ELSE IF(DBEXP)THEN
                      !     VAN DER WAAL DOUBLE EXPONENTIAL POTENTIAL
                      !     E= A*EXP(alpha*R)-B*EXP(beta*R)
                      !
                      IF(S < C2OFNB)THEN
                         SIG=SQRT(SIG2)
#if KEY_NBIPS == 1
                         AKR=AIPSVA(0)/SIG
                        EXPAR=EXP(-AKR)
                        ! energy
                        ENBPR=AIPSVA(1)*EXPAR
                        ! force
                        DPVA=BIPSVA(1)*EXPAR/SIG
                        CKR=AIPSVC(0)/SIG
                        EXPCR=EXP(-CKR)
                        ! energy
                        ENBPR=CNBB(IC)*(ENBPR-AIPSVC(1)*EXPCR)
                        ! force
                        DPVC=BIPSVC(1)*EXPCR/SIG
                        DF=DF+CNBB(IC)*(DPVA-DPVC)*R2
#if ENBONDA_LSECD==1 /*lsecd1*/
                        IF(LSECD) THEN
                           DDF=DDF-CNBB(IC)*(BIPSVA(1)*EXPAR*AIPSVA(0)-BIPSVC(1)*EXPCR*AIPSVC(0))*R2*R2/SIG2
                        ENDIF
#endif /*  (lsecd1)*/
#endif  /* KEY_NBIPS */
                      ELSE
                         ENBPR=ZERO
                      ENDIF
#if KEY_NBIPS==1 /*nbips_vdw*/
                    !WXW VDW potential using isotropic periodic sum (IPS)
                    ELSE IF (DBEXPIPS) THEN                              
                      !     VAN DER WAAL DOUBLE EXPONENTIAL IPS
                      !WXW VDW DE potential using isotropic periodic sum (IPS)
                      !WXW   1-4  interaction will be calculated at subroutine EEXIPS
                      SIG=SQRT(SIG2)
                      IF(DOIPS.AND.SRIPS*SIG>ONE)THEN
                      AKR=AIPSVA(0)/SIG
                      EXPAR=EXP(-AKR)
                      ! energy
                      ENBPR=AIPSVA(1)*(EXPAR+AIPSVA(4)/EXPAR+AIPSVA(5))
                      ! force 
                      DPVA=BIPSVA(1)*(EXPAR-AIPSVA(4)/EXPAR)/SIG
                      !
                      CKR=AIPSVC(0)/SIG
                      EXPCR=EXP(-CKR)
                      ! energy
                      ENBPR=CNBB(IC)*(ENBPR-AIPSVC(1)*(EXPCR+AIPSVC(4)/EXPCR+AIPSVC(5)))
                      ! force
                      DPVC=BIPSVC(1)*(EXPCR-AIPSVC(4)/EXPCR)/SIG
                      DF=DF+CNBB(IC)*(DPVA-DPVC)*R2
#if ENBONDA_LSECD==1 /*lsecd2*/
                      IF(LSECD) THEN
                         DDF=DDF+CNBB(IC)*(AIPSVA(0)*(BIPSVA(1)*(EXPAR+AIPSVA(4)/EXPAR)) &
                         -AIPSVC(0)*(BIPSVC(1)*(EXPCR+AIPSVC(4)/EXPCR)))*R2*R2/SIG2
                      ENDIF
#endif /*  (lsecd2)*/
                    ELSE 
                      ENBPR=ZERO
                    ENDIF
                  ELSE IF (DVIPS) THEN
                    IF(DOIPS)THEN
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
                      ENEVA =CNBB(IC)*SIG12/U12R
                      ENEVC =-TWO*CNBB(IC)*SIG6/U6R
                      ENBPR=ENEVA*PVA+ENEVC*PVC
                      DF=DF+(ENEVA*DPVA+ENEVC*DPVC)*R2
#if ENBONDG_LSECD==1 /*lsecd3*/
                      IF(LSECD) THEN
                         DDPVC=42.0D0*U6R+U2*(BBIPSVC(1)+U2*(BBIPSVC(2)+U2*(BBIPSVC(3) &
                           +U2*(BBIPSVC(4)+U4*(BBIPSVC(5)+U4*BBIPSVC(6))))))
                         DDPVA=156.0D0*U12R+U2*(BBIPSVA(1)+U2*(BBIPSVA(2)+U2*(BBIPSVA(3) &
                           +U4*(BBIPSVA(4)+U4*(BBIPSVA(5)+U4*BBIPSVA(6))))))
                         DDF=DDF+(ENEVA*DDPVA+ENEVC*DDPVC)*R2*R2
                      ENDIF
#endif /*  (lsecd3)*/
                    ELSE
                      ENBPR=ZERO
                    ENDIF
#endif /* (nbips_vdw)*/
                  ELSE
                    IF (.NOT.QETEN .AND. .NOT.QETSR) THEN
#if KEY_WCA==1
!      write(*,*) 'INTER', I, J, ENBPR, EPRPL
                    DF=DF+EGLJTMP*WCA(I)*WCA(J)+EGLJRPL
                    ENBPR=ENBPR*WCA(I)*WCA(J)+EPRPL
#if ENBONDG_LSECD==1
                    IF(LSECD) DDF=DDF+                       & 
                      CNBB(IC)*(156.*SIG12-84.0*SIG6)*R2*R2 
#endif
#else /**/
                    DF=DF+CNBB(IC)*TWELVE*R2*(SIG6-SIG12)
                    ENBPR=CNBB(IC)*(SIG12-SIG6-SIG6)

#if ENBONDG_LSECD==1
                    IF(LSECD) DDF=DDF+                       & 
                      CNBB(IC)*(156.*SIG12-84.0*SIG6)*R2*R2 
#endif
#endif 
                  ELSE IF (QETSR) THEN 
                    FAC=FOUR/(NINE*SIG2)
                    SWTMP=(FAC*FAC)**3
                    ENBPR=(CNBB(IC)*(THIRTN*SIG12-NINE*TWO*SIG6 &
                         *SIG2*SIG2+FOUR*SIG6)) &
                         / (ONE+SWTMP)
                    ! DF = 1/R dV/dr
                    DF=DF+CNBB(IC)*R2*TWELVE*(FIFTN*SIG6*SIG2*SIG2 &
                         -THIRTN*SIG12-TWO*SIG6) &
                         / (ONE+SWTMP) &
                         - R2*ENBPR*TWELVE*SWTMP &
                         /((ONE+SWTMP)*(ONE+SWTMP))
                    IF(LSECD) THEN
                       ODF=CNBB(IC)*R2*TWELVE*(FIFTN*SIG6*SIG2*SIG2 &
                            -THIRTN*SIG12-TWO*SIG6)

                       DDF=DDF+(CNBB(IC)*TWELVE*(169.0*SIG12- &
                            165.0*SIG6*SIG2*SIG2+14.0*SIG6)*R2*R2) &
                            / ( ONE+SWTMP) &
                            - TWO*R2*TWELVE*ODF*SWTMP/((ONE+SWTMP)*(ONE+SWTMP)) &
                            - R2*R2*TWELVE*ENBPR*SWTMP/(ONE+SWTMP) &
                            * (MINTWO*TWELVE*SWTMP/(ONE+SWTMP) + ELEVEN)
                    ENDIF
                  ELSE ! QETEN
                    DF=DF+CNBB(IC)*TWELVE*R2* &
                     (FIFTN*SIG6*SIG2*SIG2-THIRTN*SIG12-TWO*SIG6)
#if ENBONDG_LSECD==1
                    IF(LSECD) DDF=DDF+                       & 
                      CNBB(IC)*TWELVE*(169.0*SIG12-165.0*   & 
                      SIG6*SIG2*SIG2+14.0*SIG2)*R2*R2      
#endif
                    ENBPR=(CNBB(IC)*(THIRTN*SIG12-NINE*TWO*SIG6 &
                      *SIG2*SIG2+FOUR*SIG6))
                  ENDIF
                ENDIF              /* dbexp */                       
#if ENBONDG_V_FLAG==1
              ENDIF                                     
#endif
#endif /*  (lvgrp2)*/


#if KEY_LJPME==1
    if(qljpme) call ljpme_dir(ljc6(I), ljc6(j), s, enbpr, df, CNBA(IC), CNBB(IC), .false.)
#endif
!
#if KEY_MTS==1
!---- LONG-SHORT RANGE MTS METHOD
                  IF(SLFG) THEN
                      EELPR=EELPR*SWFE
                      ENBPR=ENBPR*SWFE
                      DF=DF*SWF
#if ENBONDG_LSECD==1
                      IF(LSECD) DDF=DDF*SWF                 
#endif
                  ENDIF
#endif 

#if KEY_BLOCK==1
                  IF (QBLOCK) THEN
                     IBL = IBLCKP(I)
                     JBL = IBLCKP(J)
                     KK=MAX(IBL,JBL)
                     KK=KK*(KK-1)/2+MIN(IBL,JBL)
                     COEF = BLCOEP(KK)
                     EELPR=EELPR*COEF
                     ENBPR=ENBPR*COEF
                     DF=DF*COEF
#if ENBONDG_LSECD==1
                     IF(LSECD) DDF=DDF*COEF                 
#endif
                  ENDIF
!
                  IF (.NOT. NOFORC) THEN
#endif 
#if ENBONDG_LSECD==1 /*lsecd10*/
                     IF(LSECD) THEN
                       DDF=DDF*FUNCT
                       DFI=DFN*DF
                     ENDIF
#endif /*  (lsecd10)*/
                     DF=DF*FUNCT
!
                     DXIT=DXI*DF
                     DYIT=DYI*DF
                     DZIT=DZI*DF
                     DX(I)=DX(I)+DXIT
                     DY(I)=DY(I)+DYIT
                     DZ(I)=DZ(I)+DZIT
                     DX(J)=DX(J)-DXIT
                     DY(J)=DY(J)-DYIT
                     DZ(J)=DZ(J)-DZIT
#if KEY_FLUCQ==1
! Add in this interaction to the FlucQ arrays, modified by
! the switching function if necessary
                     IF (QFLUC) THEN
                        FQCFOR(I)=FQCFOR(I)+EELPR*FUNCT
                        FQCFOR(J)=FQCFOR(J)+EELPR*FUNCT
                     ENDIF
#endif 
#if KEY_BLOCK==1
                  ENDIF
#endif 
                  ETEMP1=ETEMP1+ENBPR
                  ETEMP2=ETEMP2+EELPR
!
! Calculate second derivatives
#if ENBONDG_LSECD==1 /*lsecd_main2*/
                  IF(LSECD) THEN
! calculate d2F*S term
                    DDF=DDF-DF*R2
!
                    AXX=DXI*DXI*DDF+DF
                    AYY=DYI*DYI*DDF+DF
                    AZZ=DZI*DZI*DDF+DF
                    AXY=DXI*DYI*DDF
                    AXZ=DXI*DZI*DDF
                    AYZ=DYI*DZI*DDF
!
#if KEY_DIMB==1
                    IF(QCMPCT) THEN
                        CALL EVDCMP(I,J,AXX,AYY,AZZ,AXY,AXZ,AYZ,DD1, &
                                    PINBCM,PJNBCM)
                        IF(LSWITR) THEN
                          IF(LSWIT) THEN
                           CALL EG1CMP(I,J,IS,IQ,JSS,JQ,NI,NJ,DFI, &
                                       DXIC,DYIC,DZIC,DXI,DYI,DZI,DD1, &
                                       PINBCM,PJNBCM)
                          ENDIF
                        ENDIF

                    ELSE
#endif 
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
                    IF(LSWITR) THEN
! calculate 2*dF*dS term
                      IF(LSWIT) THEN
                        DFJ=DFI/NJ
                        DFI=DFI/NI
                        DXIT=DXIC*DFI
                        DYIT=DYIC*DFI
                        DZIT=DZIC*DFI
                        DXJT=DXIC*DFJ
                        DYJT=DYIC*DFJ
                        DZJT=DZIC*DFJ
                        DO IX=IS,IQ
                          II=3*I-2
                          JJ=3*IX-2
                          IF (II == JJ) THEN
                            IADD=IUPT(II)+II
                            DD1(IADD)=DD1(IADD)+DXI*DXIT*TWO
                            IADD=IUPT(II+1)+II+1
                            DD1(IADD)=DD1(IADD)+DYI*DYIT*TWO
                            IADD=IUPT(II+2)+II+2
                            DD1(IADD)=DD1(IADD)+DZI*DZIT*TWO
                            IADD=IUPT(II)+II+1
                            DD1(IADD)=DD1(IADD)+DXI*DYIT+DYI*DXIT
                            IADD=IUPT(II)+II+2
                            DD1(IADD)=DD1(IADD)+DXI*DZIT+DZI*DXIT
                            IADD=IUPT(II+1)+II+2
                            DD1(IADD)=DD1(IADD)+DYI*DZIT+DZI*DYIT
                          ELSE IF (JJ < II) THEN
                            IADD=IUPT(JJ)+II
                            DD1(IADD)=DD1(IADD)+DXI*DXIT
                            IADD=IUPT(JJ+1)+II+1
                            DD1(IADD)=DD1(IADD)+DYI*DYIT
                            IADD=IUPT(JJ+2)+II+2
                            DD1(IADD)=DD1(IADD)+DZI*DZIT
                            IADD=IUPT(JJ)+II+1
                            DD1(IADD)=DD1(IADD)+DYI*DXIT
                            IADD=IUPT(JJ+1)+II
                            DD1(IADD)=DD1(IADD)+DXI*DYIT
                            IADD=IUPT(JJ)+II+2
                            DD1(IADD)=DD1(IADD)+DZI*DXIT
                            IADD=IUPT(JJ+2)+II
                            DD1(IADD)=DD1(IADD)+DXI*DZIT
                            IADD=IUPT(JJ+1)+II+2
                            DD1(IADD)=DD1(IADD)+DZI*DYIT
                            IADD=IUPT(JJ+2)+II+1
                            DD1(IADD)=DD1(IADD)+DYI*DZIT
                          ELSE
                            IADD=IUPT(II)+JJ
                            DD1(IADD)=DD1(IADD)+DXI*DXIT
                            IADD=IUPT(II+1)+JJ+1
                            DD1(IADD)=DD1(IADD)+DYI*DYIT
                            IADD=IUPT(II+2)+JJ+2
                            DD1(IADD)=DD1(IADD)+DZI*DZIT
                            IADD=IUPT(II+1)+JJ
                            DD1(IADD)=DD1(IADD)+DYI*DXIT
                            IADD=IUPT(II)+JJ+1
                            DD1(IADD)=DD1(IADD)+DXI*DYIT
                            IADD=IUPT(II+2)+JJ
                            DD1(IADD)=DD1(IADD)+DZI*DXIT
                            IADD=IUPT(II)+JJ+2
                            DD1(IADD)=DD1(IADD)+DXI*DZIT
                            IADD=IUPT(II+2)+JJ+1
                            DD1(IADD)=DD1(IADD)+DZI*DYIT
                            IADD=IUPT(II+1)+JJ+2
                            DD1(IADD)=DD1(IADD)+DYI*DZIT
                          ENDIF
                          II=3*J-2
                          JJ=3*IX-2
                          IF (JJ < II) THEN
                            IADD=IUPT(JJ)+II
                            DD1(IADD)=DD1(IADD)-DXI*DXIT
                            IADD=IUPT(JJ+1)+II+1
                            DD1(IADD)=DD1(IADD)-DYI*DYIT
                            IADD=IUPT(JJ+2)+II+2
                            DD1(IADD)=DD1(IADD)-DZI*DZIT
                            IADD=IUPT(JJ)+II+1
                            DD1(IADD)=DD1(IADD)-DYI*DXIT
                            IADD=IUPT(JJ+1)+II
                            DD1(IADD)=DD1(IADD)-DXI*DYIT
                            IADD=IUPT(JJ)+II+2
                            DD1(IADD)=DD1(IADD)-DZI*DXIT
                            IADD=IUPT(JJ+2)+II
                            DD1(IADD)=DD1(IADD)-DXI*DZIT
                            IADD=IUPT(JJ+1)+II+2
                            DD1(IADD)=DD1(IADD)-DZI*DYIT
                            IADD=IUPT(JJ+2)+II+1
                            DD1(IADD)=DD1(IADD)-DYI*DZIT
                          ELSE
                            IADD=IUPT(II)+JJ
                            DD1(IADD)=DD1(IADD)-DXI*DXIT
                            IADD=IUPT(II+1)+JJ+1
                            DD1(IADD)=DD1(IADD)-DYI*DYIT
                            IADD=IUPT(II+2)+JJ+2
                            DD1(IADD)=DD1(IADD)-DZI*DZIT
                            IADD=IUPT(II+1)+JJ
                            DD1(IADD)=DD1(IADD)-DYI*DXIT
                            IADD=IUPT(II)+JJ+1
                            DD1(IADD)=DD1(IADD)-DXI*DYIT
                            IADD=IUPT(II+2)+JJ
                            DD1(IADD)=DD1(IADD)-DZI*DXIT
                            IADD=IUPT(II)+JJ+2
                            DD1(IADD)=DD1(IADD)-DXI*DZIT
                            IADD=IUPT(II+2)+JJ+1
                            DD1(IADD)=DD1(IADD)-DZI*DYIT
                            IADD=IUPT(II+1)+JJ+2
                            DD1(IADD)=DD1(IADD)-DYI*DZIT
                          ENDIF
                        ENDDO
                        DO JX=JSS,JQ
                          II=3*I-2
                          JJ=3*JX-2
                          IF (JJ < II) THEN
                            IADD=IUPT(JJ)+II
                            DD1(IADD)=DD1(IADD)-DXI*DXJT
                            IADD=IUPT(JJ+1)+II+1
                            DD1(IADD)=DD1(IADD)-DYI*DYJT
                            IADD=IUPT(JJ+2)+II+2
                            DD1(IADD)=DD1(IADD)-DZI*DZJT
                            IADD=IUPT(JJ)+II+1
                            DD1(IADD)=DD1(IADD)-DYI*DXJT
                            IADD=IUPT(JJ+1)+II
                            DD1(IADD)=DD1(IADD)-DXI*DYJT
                            IADD=IUPT(JJ)+II+2
                            DD1(IADD)=DD1(IADD)-DZI*DXJT
                            IADD=IUPT(JJ+2)+II
                            DD1(IADD)=DD1(IADD)-DXI*DZJT
                            IADD=IUPT(JJ+1)+II+2
                            DD1(IADD)=DD1(IADD)-DZI*DYJT
                            IADD=IUPT(JJ+2)+II+1
                            DD1(IADD)=DD1(IADD)-DYI*DZJT
                          ELSE
                            IADD=IUPT(II)+JJ
                            DD1(IADD)=DD1(IADD)-DXI*DXJT
                            IADD=IUPT(II+1)+JJ+1
                            DD1(IADD)=DD1(IADD)-DYI*DYJT
                            IADD=IUPT(II+2)+JJ+2
                            DD1(IADD)=DD1(IADD)-DZI*DZJT
                            IADD=IUPT(II+1)+JJ
                            DD1(IADD)=DD1(IADD)-DYI*DXJT
                            IADD=IUPT(II)+JJ+1
                            DD1(IADD)=DD1(IADD)-DXI*DYJT
                            IADD=IUPT(II+2)+JJ
                            DD1(IADD)=DD1(IADD)-DZI*DXJT
                            IADD=IUPT(II)+JJ+2
                            DD1(IADD)=DD1(IADD)-DXI*DZJT
                            IADD=IUPT(II+2)+JJ+1
                            DD1(IADD)=DD1(IADD)-DZI*DYJT
                            IADD=IUPT(II+1)+JJ+2
                            DD1(IADD)=DD1(IADD)-DYI*DZJT
                          ENDIF
                          II=3*J-2
                          JJ=3*JX-2
                          IF (II == JJ) THEN
                            IADD=IUPT(II)+II
                            DD1(IADD)=DD1(IADD)+DXI*DXJT*TWO
                            IADD=IUPT(II+1)+II+1
                            DD1(IADD)=DD1(IADD)+DYI*DYJT*TWO
                            IADD=IUPT(II+2)+II+2
                            DD1(IADD)=DD1(IADD)+DZI*DZJT*TWO
                            IADD=IUPT(II)+II+1
                            DD1(IADD)=DD1(IADD)+DXI*DYJT+DYI*DXJT
                            IADD=IUPT(II)+II+2
                            DD1(IADD)=DD1(IADD)+DXI*DZJT+DZI*DXJT
                            IADD=IUPT(II+1)+II+2
                            DD1(IADD)=DD1(IADD)+DYI*DZJT+DZI*DYJT
                          ELSE IF (JJ < II) THEN
                            IADD=IUPT(JJ)+II
                            DD1(IADD)=DD1(IADD)+DXI*DXJT
                            IADD=IUPT(JJ+1)+II+1
                            DD1(IADD)=DD1(IADD)+DYI*DYJT
                            IADD=IUPT(JJ+2)+II+2
                            DD1(IADD)=DD1(IADD)+DZI*DZJT
                            IADD=IUPT(JJ)+II+1
                            DD1(IADD)=DD1(IADD)+DYI*DXJT
                            IADD=IUPT(JJ+1)+II
                            DD1(IADD)=DD1(IADD)+DXI*DYJT
                            IADD=IUPT(JJ)+II+2
                            DD1(IADD)=DD1(IADD)+DZI*DXJT
                            IADD=IUPT(JJ+2)+II
                            DD1(IADD)=DD1(IADD)+DXI*DZJT
                            IADD=IUPT(JJ+1)+II+2
                            DD1(IADD)=DD1(IADD)+DZI*DYJT
                            IADD=IUPT(JJ+2)+II+1
                            DD1(IADD)=DD1(IADD)+DYI*DZJT
                          ELSE
                            IADD=IUPT(II)+JJ
                            DD1(IADD)=DD1(IADD)+DXI*DXJT
                            IADD=IUPT(II+1)+JJ+1
                            DD1(IADD)=DD1(IADD)+DYI*DYJT
                            IADD=IUPT(II+2)+JJ+2
                            DD1(IADD)=DD1(IADD)+DZI*DZJT
                            IADD=IUPT(II+1)+JJ
                            DD1(IADD)=DD1(IADD)+DYI*DXJT
                            IADD=IUPT(II)+JJ+1
                            DD1(IADD)=DD1(IADD)+DXI*DYJT
                            IADD=IUPT(II+2)+JJ
                            DD1(IADD)=DD1(IADD)+DZI*DXJT
                            IADD=IUPT(II)+JJ+2
                            DD1(IADD)=DD1(IADD)+DXI*DZJT
                            IADD=IUPT(II+2)+JJ+1
                            DD1(IADD)=DD1(IADD)+DZI*DYJT
                            IADD=IUPT(II+1)+JJ+2
                            DD1(IADD)=DD1(IADD)+DYI*DZJT
                          ENDIF
                        ENDDO
                      ENDIF
                    ENDIF
#if KEY_DIMB==1
                    ENDIF
#endif /*  DIMB*/
                  ENDIF
#endif /*  (lsecd_main2)*/
#if ENBONDG_ANALYS==1 /*analys2*/
                  IF(QECONTX) THEN
                    S=HALF*(ENBPR+EELPR+ESWADD/(NI*NJ))*FUNCT
                    ECONTX(I)=ECONTX(I)+S
                    ECONTX(J)=ECONTX(J)+S
                  ENDIF
#endif /*  (analys2)*/
!
!
!!  XX  MH11:  We cycle now :-)
!!  XX  !!! namkh 01/20/04
!!  XX  !!   ##IF GAMESS GAMESSUK MNDO97 SQUANTM QCHEM QTURBO G09
!!  XX  !!                   END IF
!!  XX  !!   ##ENDIF
!
                ENDDO
              ENDDO
!
              ETEMP2=ETEMP2+ESWADD
              IF(LSWITR) THEN
! Backtransform center of geometry forces
                IF(LSWIT) THEN
                  DFN=DFN*(ETEMP1+ETEMP2)
                  DFI=DFN/NI
                  DFJ=DFN/NJ
#if KEY_BLOCK==1
                  IF (.NOT. NOFORC) THEN
#endif 
                     DXIT=DXIC*DFI
                     DYIT=DYIC*DFI
                     DZIT=DZIC*DFI
                     DXJT=DXIC*DFJ
                     DYJT=DYIC*DFJ
                     DZJT=DZIC*DFJ
                     DO I=IS,IQ
                       DX(I)=DX(I)+DXIT
                       DY(I)=DY(I)+DYIT
                       DZ(I)=DZ(I)+DZIT
                     ENDDO
                     DO J=JSS,JQ
                       DX(J)=DX(J)-DXJT
                       DY(J)=DY(J)-DYJT
                       DZ(J)=DZ(J)-DZJT
                     ENDDO
#if KEY_BLOCK==1
                  ENDIF
#endif 
                ENDIF
!
! Calculate second derivatives
#if ENBONDG_LSECD==1 /*lsecd_main3*/
                IF(LSECD) THEN
#if KEY_DIMB==1
                  IF(QCMPCT) THEN
                     CALL EG2CMP(IS,IQ,JSS,JQ,NI,NJ,DFN,DDFN, &
                                 ETEMP1,ETEMP2,DXIC,DYIC,DZIC,DD1, &
                                 PINBCM,PJNBCM)
                  ELSE
#endif /*  DIMB*/
!
! calculate F*d2S term
                    DDF=DDFN*(ETEMP1+ETEMP2)
                    DFI=DDF/(NI*NI)
                    DFJ=DFN/(NI*NI)
                    DO I=IS,IQ
                      DO IX=I,IQ
                        II=3*I-2
                        JJ=3*IX-2
                        IF (II == JJ) THEN
                          IADD=IUPT(II)+II
                          DD1(IADD)=DD1(IADD)+DFI*DXIC*DXIC+DFJ
                          IADD=IUPT(II+1)+II+1
                          DD1(IADD)=DD1(IADD)+DFI*DYIC*DYIC+DFJ
                          IADD=IUPT(II+2)+II+2
                          DD1(IADD)=DD1(IADD)+DFI*DZIC*DZIC+DFJ
                          IADD=IUPT(II)+II+1
                          DD1(IADD)=DD1(IADD)+DFI*DXIC*DYIC
                          IADD=IUPT(II)+II+2
                          DD1(IADD)=DD1(IADD)+DFI*DXIC*DZIC
                          IADD=IUPT(II+1)+II+2
                          DD1(IADD)=DD1(IADD)+DFI*DYIC*DZIC
                        ELSE
                          IADD=IUPT(II)+JJ
                          DD1(IADD)=DD1(IADD)+DFI*DXIC*DXIC+DFJ
                          IADD=IUPT(II+1)+JJ+1
                          DD1(IADD)=DD1(IADD)+DFI*DYIC*DYIC+DFJ
                          IADD=IUPT(II+2)+JJ+2
                          DD1(IADD)=DD1(IADD)+DFI*DZIC*DZIC+DFJ
                          IADD=IUPT(II+1)+JJ
                          DD1(IADD)=DD1(IADD)+DFI*DYIC*DXIC
                          IADD=IUPT(II)+JJ+1
                          DD1(IADD)=DD1(IADD)+DFI*DXIC*DYIC
                          IADD=IUPT(II+2)+JJ
                          DD1(IADD)=DD1(IADD)+DFI*DZIC*DXIC
                          IADD=IUPT(II)+JJ+2
                          DD1(IADD)=DD1(IADD)+DFI*DXIC*DZIC
                          IADD=IUPT(II+2)+JJ+1
                          DD1(IADD)=DD1(IADD)+DFI*DZIC*DYIC
                          IADD=IUPT(II+1)+JJ+2
                          DD1(IADD)=DD1(IADD)+DFI*DYIC*DZIC
                        ENDIF
                      ENDDO
                    ENDDO
!
                    DFI=DDF/(NJ*NJ)
                    DFJ=DFN/(NJ*NJ)
                    DO J=JSS,JQ
                      DO JX=J,JQ
                        II=3*J-2
                        JJ=3*JX-2
                        IF (II == JJ) THEN
                          IADD=IUPT(II)+II
                          DD1(IADD)=DD1(IADD)+DFI*DXIC*DXIC+DFJ
                          IADD=IUPT(II+1)+II+1
                          DD1(IADD)=DD1(IADD)+DFI*DYIC*DYIC+DFJ
                          IADD=IUPT(II+2)+II+2
                          DD1(IADD)=DD1(IADD)+DFI*DZIC*DZIC+DFJ
                          IADD=IUPT(II)+II+1
                          DD1(IADD)=DD1(IADD)+DFI*DXIC*DYIC
                          IADD=IUPT(II)+II+2
                          DD1(IADD)=DD1(IADD)+DFI*DXIC*DZIC
                          IADD=IUPT(II+1)+II+2
                          DD1(IADD)=DD1(IADD)+DFI*DYIC*DZIC
                        ELSE
                          IADD=IUPT(II)+JJ
                          DD1(IADD)=DD1(IADD)+DFI*DXIC*DXIC+DFJ
                          IADD=IUPT(II+1)+JJ+1
                          DD1(IADD)=DD1(IADD)+DFI*DYIC*DYIC+DFJ
                          IADD=IUPT(II+2)+JJ+2
                          DD1(IADD)=DD1(IADD)+DFI*DZIC*DZIC+DFJ
                          IADD=IUPT(II+1)+JJ
                          DD1(IADD)=DD1(IADD)+DFI*DYIC*DXIC
                          IADD=IUPT(II)+JJ+1
                          DD1(IADD)=DD1(IADD)+DFI*DXIC*DYIC
                          IADD=IUPT(II+2)+JJ
                          DD1(IADD)=DD1(IADD)+DFI*DZIC*DXIC
                          IADD=IUPT(II)+JJ+2
                          DD1(IADD)=DD1(IADD)+DFI*DXIC*DZIC
                          IADD=IUPT(II+2)+JJ+1
                          DD1(IADD)=DD1(IADD)+DFI*DZIC*DYIC
                          IADD=IUPT(II+1)+JJ+2
                          DD1(IADD)=DD1(IADD)+DFI*DYIC*DZIC
                        ENDIF
                      ENDDO
                    ENDDO
!
                    DFI=DDF/(NI*NJ)
                    DFJ=DFN/(NI*NJ)
                    DO I=IS,IQ
                      DO J=JSS,JQ
                        II=3*I-2
                        JJ=3*J-2
                        IF (JJ < II) THEN
                          IADD=IUPT(JJ)+II
                          DD1(IADD)=DD1(IADD)-DFI*DXIC*DXIC-DFJ
                          IADD=IUPT(JJ+1)+II+1
                          DD1(IADD)=DD1(IADD)-DFI*DYIC*DYIC-DFJ
                          IADD=IUPT(JJ+2)+II+2
                          DD1(IADD)=DD1(IADD)-DFI*DZIC*DZIC-DFJ
                          IADD=IUPT(JJ)+II+1
                          DD1(IADD)=DD1(IADD)-DFI*DYIC*DXIC
                          IADD=IUPT(JJ+1)+II
                          DD1(IADD)=DD1(IADD)-DFI*DXIC*DYIC
                          IADD=IUPT(JJ)+II+2
                          DD1(IADD)=DD1(IADD)-DFI*DZIC*DXIC
                          IADD=IUPT(JJ+2)+II
                          DD1(IADD)=DD1(IADD)-DFI*DXIC*DZIC
                          IADD=IUPT(JJ+1)+II+2
                          DD1(IADD)=DD1(IADD)-DFI*DZIC*DYIC
                          IADD=IUPT(JJ+2)+II+1
                          DD1(IADD)=DD1(IADD)-DFI*DYIC*DZIC
                        ELSE
                          IADD=IUPT(II)+JJ
                          DD1(IADD)=DD1(IADD)-DFI*DXIC*DXIC-DFJ
                          IADD=IUPT(II+1)+JJ+1
                          DD1(IADD)=DD1(IADD)-DFI*DYIC*DYIC-DFJ
                          IADD=IUPT(II+2)+JJ+2
                          DD1(IADD)=DD1(IADD)-DFI*DZIC*DZIC-DFJ
                          IADD=IUPT(II+1)+JJ
                          DD1(IADD)=DD1(IADD)-DFI*DYIC*DXIC
                          IADD=IUPT(II)+JJ+1
                          DD1(IADD)=DD1(IADD)-DFI*DXIC*DYIC
                          IADD=IUPT(II+2)+JJ
                          DD1(IADD)=DD1(IADD)-DFI*DZIC*DXIC
                          IADD=IUPT(II)+JJ+2
                          DD1(IADD)=DD1(IADD)-DFI*DXIC*DZIC
                          IADD=IUPT(II+2)+JJ+1
                          DD1(IADD)=DD1(IADD)-DFI*DZIC*DYIC
                          IADD=IUPT(II+1)+JJ+2
                          DD1(IADD)=DD1(IADD)-DFI*DYIC*DZIC
                        ENDIF
                      ENDDO
                    ENDDO
#if KEY_DIMB==1
                  ENDIF
#endif /*  DIMB*/
                ENDIF
#endif /*  (lsecd_main3)*/
!
                ETEMP1=ETEMP1*FUNCT
                ETEMP2=ETEMP2*FUNCT
              ENDIF  ! (LSWITR)
#if KEY_MTS==1
             ENDIF  ! (.NOT.SKIP)                           
#endif
            ENDIF  ! (SCENT < C2OFNB)
          ENDIF  ! (LEXCL)
          EEL=EEL+ETEMP2
          ENB=ENB+ETEMP1
        ENDDO  ! JRSPR
      ENDDO  ! IRS
!=======================================================================
!   Main loop end
!=======================================================================
