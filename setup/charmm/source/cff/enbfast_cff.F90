#if KEY_CFF==1
SUBROUTINE ENBFS8_CFF(ENB,EEL,LELECX,LVDWX,IFRSTA,NATOMX, &
     CGX,JNBL,INBL, &
     IACNB,NITCC2,LOWTP, &
#if KEY_BLOCK==1
     IBLOCK,BLCOE,BLCOV,             & 
#endif
#if KEY_BLOCK==1
     BLCOVR,BLCOVA,                  & 
#endif
#if KEY_FLUCQ==1
     QFLUC,FQCFOR,                   & 
#endif
     LUSED)
  !----------------------------------------------------------------------
  !     This file is actually a copy of enbfs8.src with a few lines of
  !     code changed to calculate a 9-6 potential instead of a 12-6.
  !     This change was only made for the standard calculation so none
  !     of the special features () are supported yet.  Rick Lapp
  !
  !     This is the fast scalar version of the nonboned energy terms
  !     for {constant dielectric} {electrostatic shifting} {vdW shifting}
  !         {distance dielectric} {electrostatic switch  } {vdW switch  }
  !     All combinations of these nonbond energy options are supported
  !
  !     January 11, 1990  Youngdo Won
  !     October 16, 1991  Force-based methods added.  PJS
  !     May 25, 1998      Modified to use 9-6 potential for CFF forcefield
  !                       Rick Lapp.
  !     March, 2008       Updated l-dynamics variables for dma. JLK
  !-----------------------------------------------------------------------

  use chm_kinds
  use nb_module      !  has ccnba thru d
  use consta
  use dimens_fcm
  use number
  use coord
  use deriv
  use param
  use inbnd
#if KEY_BLOCK==1
  use block_fcm
  use pert  !Cc New PBLOCK
#endif 
  use lambdam
#if KEY_PBOUND==1
  use pbound     
#endif
#if KEY_MTS==1
  use tbmts      
#endif
#if KEY_GENETIC==1
  use galgor     
#endif
  implicit none
  real(chm_real)  ENB,EEL
  LOGICAL LELECX,LVDWX
  INTEGER IFRSTA,NATOMX,JNBL(*),INBL(*)
  real(chm_real) CGX(*)
  INTEGER IACNB(*),NITCC2,LOWTP(*)
#if KEY_FLUCQ==1
  LOGICAL QFLUC
  real(chm_real) FQCFOR(*)
#endif 
  LOGICAL LUSED

#if KEY_BLOCK==1
  INTEGER IBL,JBL,KK, IBLOCK(*)
  real(chm_real)  BLCOE(*),BLCOV(*),BLCOVR(*),BLCOVA(*)
#if KEY_DOCK==1
  INTEGER KDOC
  real(chm_real)  DOCFI, DOCFJ
#endif 
  real(chm_real) FALPHA
  INTEGER ILDM, JLDM
#endif /*  BLOCK*/
#if KEY_PBOUND==1 /*pbound*/
  real(chm_real) CORR
#endif /*     (pbound)*/
#if KEY_MTS==1
  real(chm_real)  RR1,RR2,RR3
  real(chm_real)  SWF,SWFE
#endif 

  INTEGER IVECT,JVECT,KVECT
  real(chm_real) CA,CC,CH,ENE,ENN
  real(chm_real) TF,TX,TY,TZ,DTX,DTY,DTZ
  real(chm_real) TFELEC,TFVDW
  real(chm_real) S2,TR1,TR2,TR6,FSW,DFSW,FSH
  real(chm_real) EADD,ON3,ON6,ONOFF2,OFF3,OFF4,OFF5,OFF6,R1,R3, &
       DENOM,ACOEF,BCOEF,CCOEF,DCOEF,COVER3,DOVER5,CONST,ENEVDW
  real(chm_real) RECOF3,RECOF6,OFDIF3,OFDIF6,ONOFF3,ONOFF6, &
       CR6,CR9,RJUNK3,RJUNK6,MIN2OF
  !
  real(chm_real) C2ONNB,C2OFNB,CTROF2,C4ROF2,RUL3,RUL12,RIJL,RIJU
  real(chm_real) CGF,CGT,CRXI,CRYI,CRZI
  INTEGER ITEMP,I,J,NPR,MAXCU2,IACI
  LOGICAL ELECFG,LOUTER,RSHFT,RSWIT,CSHIFT,CFSWIT,CSHFT,CSWIT, &
       SWITCH,LVSW,LVSH,LVFSW,RSHIFT,RFSWIT
  !
  integer firsti
#if KEY_GENETIC==1
  INTEGER First        
#endif
  logical lcubes

  !---------- Sanity check -------------------------------------
  if(.not. allocated(ccnba))then
     ! How we got here without vdw table filled, who knows?
     call wrndie(-4,"ENBFS8_cff<enbfast_cff.src>", &
          "CCNBA not allocated")
  endif

#if KEY_GENETIC==1
  First = IFRSTA
  If(qGA_Ener) then
     First = Int(ENB)
  endif
#endif 
  LUSED=.TRUE.
  ENB=ZERO
  EEL=ZERO
  ELECFG=(LELECX.AND.(EPS /= ZERO))
  IF (.NOT.(LVDWX.OR.ELECFG)) RETURN
  CGF=ZERO
  IF (ELECFG) CGF=CCELEC/EPS
  !
  ! Set flags for electrostatics options (6 options supported)
  RSHIFT= .NOT.LCONS .AND.      LSHFT .AND.      LFSWT .AND. ELECFG
  RFSWIT= .NOT.LCONS .AND. .NOT.LSHFT .AND.      LFSWT .AND. ELECFG
  RSHFT = .NOT.LCONS .AND.      LSHFT .AND. .NOT.LFSWT .AND. ELECFG
  RSWIT = .NOT.LCONS .AND. .NOT.LSHFT .AND. .NOT.LFSWT .AND. ELECFG
  CSHIFT=      LCONS .AND.      LSHFT .AND.      LFSWT .AND. ELECFG
  CFSWIT=      LCONS .AND. .NOT.LSHFT .AND.      LFSWT .AND. ELECFG
  CSHFT =      LCONS .AND.      LSHFT .AND. .NOT.LFSWT .AND. ELECFG
  CSWIT =      LCONS .AND. .NOT.LSHFT .AND. .NOT.LFSWT .AND. ELECFG
  !
  IF(RSHIFT .OR. RFSWIT) THEN
     LUSED=.FALSE.
     RETURN
  ENDIF
  !
  LVFSW=      LVFSWT                   .AND. LVDWX
  LVSH = .NOT.LVFSWT .AND.      LVSHFT .AND. LVDWX
  LVSW = .NOT.LVFSWT .AND. .NOT.LVSHFT .AND. LVDWX
  SWITCH= RSWIT.OR.CSWIT.OR.LVSW
  !
#if KEY_FLUCQ==1
  IF(QFLUC) CALL WRNDIE(-4,'<ENBFS8_CFF>', &
       'No FlucQ implementation for CFF.')
#endif 
  IF(LVSH) CALL WRNDIE(-4,'<ENBFS8_CFF>', &
       'VDW Distance shifting has not as yet been implemented for CFF.')
  !
  IF(LVFSW) CALL WRNDIE(-4,'<ENBFS8_CFF>', &
       'VDW Force based switching has not yet been implemented for CFF.')
  !
  C2OFNB=CTOFNB*CTOFNB
  C2ONNB=CTONNB*CTONNB
  CTROF2=-ONE/C2OFNB
  C4ROF2=FOUR*CTROF2
  IF (CSHIFT) MIN2OF = MINTWO/CTOFNB
  !
  IF (SWITCH) THEN
     IF (CTOFNB > CTONNB) THEN
        RUL3=ONE/(C2OFNB-C2ONNB)**3
        RUL12=TWELVE*RUL3
     ENDIF
  ENDIF
  IF (CFSWIT) THEN
     !       force-based cdie switching coeffs
     IF(CTONNB  <  CTOFNB) THEN
        ONOFF2 = C2ONNB*C2OFNB
        ON3    = C2ONNB*CTONNB
        OFF3   = C2OFNB*CTOFNB
        OFF4   = C2OFNB*C2OFNB
        OFF5   = OFF3*C2OFNB
        DENOM  = ONE/(C2OFNB-C2ONNB)**3
        EADD   = (ONOFF2*(CTOFNB-CTONNB)-(OFF5-ON3*C2ONNB)/FIVE)* &
             EIGHT*DENOM
        ACOEF  = OFF4*(C2OFNB-THREE*C2ONNB)*DENOM
        BCOEF  = SIX*ONOFF2*DENOM
        COVER3 = -(C2ONNB+C2OFNB)*DENOM
        CCOEF  = THREE*COVER3
        DCOEF  = TWO*DENOM
        DOVER5 = DCOEF/FIVE
        CONST  = BCOEF*CTOFNB-ACOEF/CTOFNB+COVER3*OFF3+DOVER5*OFF5
     ELSE
        EADD  = -ONE/CTOFNB
     END IF
  ENDIF

  !     DO VDW AND ELECTROSTATIC TERMS AS REQUESTED

#if KEY_GENETIC==1
  firsti=first
#else /**/
  firsti=ifrsta
#endif 
  loop60: DO I=Firsti,NATOMX
#if KEY_IMCUBES==1
#if KEY_IMCUBES==1
     lcubes=lbycbim     
#endif
     if(lcubes)then
        ITEMP=INBL(I+NATOMX)
        NPR=INBL(I)-ITEMP
     else
#endif 
        IF(I > 1) THEN
           ITEMP=INBL(I-1)
           NPR=INBL(I)-ITEMP
        ELSE
           NPR=INBL(I)
           ITEMP=0
        ENDIF
#if KEY_IMCUBES==1
     endif
#endif 
     IF (NPR == 0) then
        ITEMP=INBL(I)
        cycle loop60
     endif
     IACI=IACNB(I)
     CGT=CGF*CGX(I)
     CRXI=X(I)
     CRYI=Y(I)
     CRZI=Z(I)
     DTX=ZERO
     DTY=ZERO
     DTZ=ZERO
     !
     loop30: DO J=1,NPR
        ! T.S. initialize TFELEC,TFVDW
        TFELEC=ZERO
        TFVDW=ZERO
        KVECT=JNBL(ITEMP+J)
        JVECT=ABS(KVECT)
        !
        TX=CRXI-X(JVECT)
        TY=CRYI-Y(JVECT)
        TZ=CRZI-Z(JVECT)
#if KEY_PBOUND==1 /*pbound*/
        If(qBoun) then
           If(qCUBoun.or.qTOBoun) then
              TX      = BOXINV * TX
              TY      = BOYINV * TY
              TZ      = BOZINV * TZ
              tx = tx - nint(tx)
              ty = ty - nint(ty)
              tz = tz - nint(tz)
!!$              IF(TX >  HALF) TX = TX - ONE
!!$              IF(TX <  -HALF) TX = TX + ONE
!!$              IF(TY >  HALF) TY = TY - ONE
!!$              IF(TY <  -HALF) TY = TY + ONE
!!$              IF(TZ >  HALF) TZ = TZ - ONE
!!$              IF(TZ <  -HALF) TZ = TZ + ONE
              If (qTOBoun) Then
                 CORR = HALF * AINT ( R75 * ( ABS ( TX ) + &
                      ABS ( TY ) + &
                      ABS ( TZ ) ) )
                 TX      = TX    - SIGN ( CORR,  TX  )
                 TY      = TY    - SIGN ( CORR,  TY  )
                 TZ      = TZ    - SIGN ( CORR,  TZ  )
              Endif
              TX      = XSIZE * TX
              TY      = YSIZE * TY
              TZ      = ZSIZE * TZ
           Else
              Call PBMove(TX, TY, TZ)
           Endif
        Endif
#endif /*      (pbound)*/
        S2=MAX(RSMALL,TX*TX+TY*TY+TZ*TZ)
        IF (S2 < C2OFNB) THEN
#if KEY_MTS==1
           IF(SLFG) THEN
              IF((SLFG1.AND.(S2 > RSCUT2))) GOTO 999
              IF((SLFG2.AND.(S2 < RSHL2)))  GOTO 999
           ENDIF
#endif 
           LOUTER=(S2 > C2ONNB)
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
           IVECT=LOWTP(MAX(IACNB(JVECT),IACI))+IACNB(JVECT)+IACI
           CH=CGT*CGX(JVECT)
           IF(KVECT < 0) THEN
              if (Lfma .and. (.not. lvdwx)) cycle loop30
              CH=CH*E14FAC
              IVECT=IVECT+NITCC2
           ENDIF
           !
           TR2=ONE/S2
           TR1=SQRT(TR2)
           TR6=TR2*TR2*TR2
           !
           !------ Electrostatic energies (only if there are charges)
           !
           IF(CH /= ZERO) THEN
              IF (LCONS) THEN
                 R1 = TR1
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
           ! vdw shift
           IF (LVSH) THEN
              CA=CCNBA(IVECT)*TR6*TR2*TR1
              ENEVDW = CA-CCNBB(IVECT)*TR6
              CC=S2*S2*S2*CCNBC(IVECT)
              ENN=ENEVDW-CC+CCNBD(IVECT)
              TFVDW=MINSIX*(ENEVDW+0.5*CA+CC)*TR2
              ! vdw force switch
              !              ELSE IF(LVFSW) THEN
              !                 IF (LOUTER) THEN
              !                   IF(.NOT.LCONS .OR. CH == ZERO) R1 = TR1
              !                   R3 = R1*TR2
              !                   RJUNK6 = TR6-RECOF6
              !                   RJUNK3 = R3-RECOF3
              !                   CR9  = CCNBA(IVECT)*OFDIF6*RJUNK6
              !                   CR6  = CCNBB(IVECT)*OFDIF3*RJUNK3
              !                   ENN = CR9*RJUNK6 - CR6*RJUNK3
              !                   TFVDW = TR2*(SIX*CR6*R3 - NINE*CR9*TR6)
              !                 ELSE
              !                   CA=CCNBA(IVECT)*TR6*TR2*TR1
              !                   ENEVDW = CA-CCNBB(IVECT)*TR6
              !                   ENN = ENEVDW+CCNBB(IVECT)*ONOFF3-CCNBA(IVECT)*ONOFF6
              !                   TFVDW = MINSIX*TR2*(ENEVDW+0.5*CA)
              !                 END IF
              ! vdw switch
           ELSE IF(LVSW) THEN
              CA=CCNBA(IVECT)*TR6*TR2*TR1
              IF (LOUTER) THEN
                 ENEVDW = CA-CCNBB(IVECT)*TR6
                 ENN=ENEVDW*FSW
                 TFVDW = ENEVDW*DFSW-SIX*TR2*(ENN+0.5*CA*FSW)
              ELSE
                 ENN = CA-CCNBB(IVECT)*TR6
                 TFVDW =  MINSIX*TR2*(ENN+0.5*CA)
              END IF
              ! no vdw
           ELSE
              ENN = ZERO
           ENDIF
           !
           !------ End of VDW energies
           !
#if KEY_MTS==1
           !------ LONG-SHORT RANGE MTS METHOD
           IF(SLFG) THEN
              SWFE = ONE
              SWF  = ONE
              IF((S2 >= RSHL2).AND.(S2 <= RSCUT2)) THEN
                 RR1 = SQRT(S2)
                 RR2 = (RR1 - RSHL)/RHEAL
                 RR3 = ONE-RR2*RR2*RR2*(6.0*RR2*RR2-15.0*RR2+10.0)
                 IF(SLFG1) SWFE=ZERO
              ELSE IF(S2 < RSHL2) THEN
                 RR3 = ONE
                 IF(SLFG2) SWFE=ZERO
              ELSE IF(S2 > RSCUT2) THEN
                 RR3 = ZERO
                 IF(SLFG1) SWFE=ZERO
              ENDIF
              IF(SLFG1) SWF=RR3
              IF(SLFG2) SWF=ONE-RR3
              ENE=ENE*SWFE
              ENN=ENN*SWFE
              TFELEC=TFELEC*SWF
              TFVDW=TFVDW*SWF
           ENDIF
#endif 
           !
#if KEY_BLOCK==1 /*block_1*/
           IF (QBLOCK) THEN
              IBL=IBLOCK(I)
              JBL=IBLOCK(JVECT)
#if KEY_DOCK==1
              !                 get asymmetric matrix coefficient
              DOCFI = 1.0
              DOCFJ = 1.0
              IF(QDOCK) THEN
                 KDOC  = (IBL - 1)*NBLOCK + JBL
                 DOCFI = BLDOCP(KDOC)
                 KDOC  = (JBL - 1)*NBLOCK + IBL
                 DOCFJ = BLDOCP(KDOC)
              ENDIF
#endif /*  DOCK*/
              KK=MAX(IBL,JBL)
              KK=KK*(KK-1)/2+MIN(IBL,JBL)
              IF(QLDM) THEN
                 JLDM = MAX(IBL,JBL)
                 ILDM = MIN(IBL,JBL)
                 !                   first row or diagonal elements exclude (1,1).
                 IF(ILDM /= 1.AND.ILDM == JLDM) THEN
                    FALPHA = (ENE + ENN)
#if KEY_PERT==1
                    if(QPERT) FALPHA = FALPHA*PERTLAM  
#endif
                    LAGMUL = LAGMUL + FALPHA
                    BIFLAM(JLDM) = BIFLAM(JLDM) + FALPHA
                 ELSE IF(ILDM == 1.AND.JLDM >= 2) THEN
                    FALPHA = (ENE + ENN)
#if KEY_PERT==1
                    if(QPERT) FALPHA = FALPHA*PERTLAM  
#endif
                    LAGMUL = LAGMUL + FALPHA
                    BIFLAM(JLDM) = BIFLAM(JLDM) + FALPHA
                 ENDIF
              ENDIF
#if KEY_DOCK==1
              IF(QDOCK) THEN
                 ENE=ENE*BLCOE(KK)*0.5*(DOCFI + DOCFJ)
                 ENN=ENN*BLCOV(KK)*0.5*(DOCFI + DOCFJ)
              ELSE
#endif 
                 ENE=ENE*BLCOE(KK)
                 ENN=ENN*BLCOV(KK)
#if KEY_DOCK==1
              ENDIF
#endif 
              TF=TFELEC*BLCOE(KK)+TFVDW*BLCOV(KK)
           ELSE
              TF=TFELEC+TFVDW
           ENDIF
           !
           IF (.NOT. NOFORC) THEN
              TF=TFELEC+TFVDW
#endif /* (block_1)*/
              !
              TX=TX*TF
              TY=TY*TF

              TZ=TZ*TF
#if KEY_BLOCK==1 /*block_2*/
#if KEY_DOCK==1
              IF(QDOCK) THEN
                 DTX=DTX+TX*DOCFI
                 DTY=DTY+TY*DOCFI
                 DTZ=DTZ+TZ*DOCFI
                 if (Lfma .and. (.not. lvdwx)) then
                    DX(JVECT)=DX(JVECT)+TX*DOCFJ
                    DY(JVECT)=DY(JVECT)+TY*DOCFJ
                    DZ(JVECT)=DZ(JVECT)+TZ*DOCFJ
                 else
                    DX(JVECT)=DX(JVECT)-TX*DOCFJ
                    DY(JVECT)=DY(JVECT)-TY*DOCFJ
                    DZ(JVECT)=DZ(JVECT)-TZ*DOCFJ
                 endif
              ELSE
#endif /* dock*/
#endif /* (block_2)*/
                 DTX=DTX+TX
                 DTY=DTY+TY
                 DTZ=DTZ+TZ
                 if (Lfma .and. (.not. lvdwx)) then
                    DX(JVECT)=DX(JVECT)+TX
                    DY(JVECT)=DY(JVECT)+TY
                    DZ(JVECT)=DZ(JVECT)+TZ
                 else
                    DX(JVECT)=DX(JVECT)-TX
                    DY(JVECT)=DY(JVECT)-TY
                    DZ(JVECT)=DZ(JVECT)-TZ
                 endif
#if KEY_BLOCK==1 /*block_3*/
#if KEY_DOCK==1
              ENDIF    
#endif
           ENDIF
#endif /* (block_3)*/
           !
           ENB=ENB+ENN
           EEL=EEL+ENE
999        CONTINUE
        ENDIF
     enddo loop30
     !
     !     RESTORE i-TH COMPONENT OF FORCE IN THE ARRAY
#if KEY_BLOCK==1
     IF (.NOT. NOFORC) THEN   
#endif
        if (Lfma .and. (.not. lvdwx)) then
           DX(I)=DX(I)-DTX
           DY(I)=DY(I)-DTY
           DZ(I)=DZ(I)-DTZ
        else
           DX(I)=DX(I)+DTX
           DY(I)=DY(I)+DTY
           DZ(I)=DZ(I)+DTZ
        endif
#if KEY_BLOCK==1
     ENDIF                   
#endif
     ITEMP=INBL(I)
  enddo loop60
  !
  return
end SUBROUTINE ENBFS8_CFF

#endif 

SUBROUTINE NULL_enbfast_CFF
  RETURN
END SUBROUTINE NULL_enbfast_CFF

