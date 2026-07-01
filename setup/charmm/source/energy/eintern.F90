module eintern

contains

SUBROUTINE EBOND(EB,IB,JB,ICB,NBOND,CBC,CBB,DX,DY,DZ,X,Y,Z, &
     QECONTX,ECONTX,ICONBH,ISKB,DD1,IUPT,QSECD,KBEXPN,LUPPER)

  !-----------------------------------------------------------------------
  !     CALCULATES BOND ENERGIES AND FORCES.
  !     IT DOES NOT CALCULATE BOND ENERGY AND FORCES FOR ALL BONDS
  !     WITH A NONZERO VALUE IN ISKB ONLY WHEN ICONBH .NE. 0
  !
  !     By Bernard R. Brooks  (mostly)  1981-1983
  !
  !     BLOCK incorporated (including seccond derivatives)
  !     By Youngdo Won, 12/17/90
  !     SPASIBA Force Field added by P. Lagant and R. Stote (11/01)
  !     SPASIBA Force Field removed for c34b2 and c35a1 releases
  !
  use chm_kinds
  use dimens_fcm
  use number
  use fourdm
#if KEY_SCCDFTB==1
  use blockscc_fcm  
#endif
#if KEY_VALBOND==1
  use valbond, only: vbmet,vbinid,vbhyp      
#endif
  use block_fcm
  use lambdam
  use pert  !Cc New PBLOCK
  use dimb
  use econtmod
  use stream
#if KEY_PARALLEL==1
  use parallel
#endif 
#if KEY_ACTBOND==1
  use actclus_mod 
#endif
  use chutil,only:atomid
  implicit none
  !
#if KEY_PARALLEL==1
  LOGICAL NOPARS     
#endif
  real(chm_real) EB
  INTEGER IB(*),JB(*),ICB(*)
  INTEGER NBOND,KBEXPN
  real(chm_real) CBC(*),CBB(*)
  real(chm_real) DX(*),DY(*),DZ(*),X(*),Y(*),Z(*)
  LOGICAL QECONTX
  real(chm_real) ECONTX(*)
  INTEGER ICONBH
  INTEGER ISKB(*)
  real(chm_real) DD1(*)
  INTEGER IUPT(*)
  LOGICAL QSECD,LUPPER
  !
  real(chm_real) RX,RY,RZ,S2,S,R,DB,DF,E,DXI,DYI,DZI,DDF,A
  INTEGER I,MM,J,IC,IADD,JJ,II,KK
  LOGICAL NOCONS,QAFIRST
  CHARACTER(len=8) SIDDNI,RIDDNI,RESDNI,ACDNI,SIDDNJ,RIDDNJ,RESDNJ,ACDNJ
  !
#if KEY_FOURD==1 /*4ddecl*/
  !     4-D variable:
  real(chm_real) RFDIM,DFDIMI,DFDI
#endif /* (4ddecl)*/
#if KEY_BLOCK==1
  INTEGER IBL, JBL, KDOC
  real(chm_real) COEF, DOCFI, DOCFJ
!ldm
  real(chm_real) UNSCALE, FALPHA
!ldm
#endif /*  BLOCK*/
#if KEY_ACTBOND==1 /*actbond0*/
  INTEGER III,UPLIMIT
#endif /* (actbond0)*/
  !*****************************************************************
  !...##IF ACTBOND (actbond0)
  !      IF(.NOT.QBACTON) CALL BACTIVDEF
  !...##ENDIF (actbond0)
  
  !
#if KEY_SCCDFTB==1
#if KEY_PARALLEL==1
  IF (MYNOD.EQ.0) THEN
#endif 
     if(qsccb) then
        if(idxbnd.eq.0) dvdlb=zero
        if(idxbnd.eq.1) dvdlub=zero
     endif
#if KEY_PARALLEL==1
  ENDIF
#endif 

#endif 
  EB=ZERO
  E=ZERO
  NOCONS=(ICONBH.GT.0)
  IF (NBOND.EQ.0) RETURN
  !
  QAFIRST=.TRUE.
  !
#if KEY_PARALLEL==1 /*parabond*/
#if KEY_VIBPARA==1
  DO 10 MM=1,NBOND
#else /**/

#if KEY_PARAFULL==1 /*parfbond*/
#if KEY_PARASCAL==1 /*parastest*/
#error  'Illegal parallel compile option'
#endif /* (parastest)*/
  DO MM=MYNODP,NBOND,NUMNOD
#elif KEY_PARASCAL==1 /*parfbond*/
  NOPARS=(ICONBH.GE.0)
  DO MM=1,NBOND
     IF(NOPARS) THEN
        IF(JPMAT(IPBLOCK(IB(MM)),IPBLOCK(JB(MM))).NE.MYNOD) GOTO 10
     ENDIF
#elif KEY_SPACDEC==1 /*parfbond*/
  DO MM=1,NBOND
     IF(ICPUMAP(IB(MM)) /= MYNOD) GOTO 10
#else /* (parfbond)*/
#error  'Illegal parallel compile option'
#endif /* (parfbond)*/
#endif 
#else /* (parabond)*/
#if KEY_ACTBOND==1 /*actbond*/
  IF(QBACTON) THEN
     UPLIMIT = NACTBND
  ELSE
     UPLIMIT = NBOND
  ENDIF
  DO III = 1,UPLIMIT
     IF (QBACTON) THEN
        MM = ACTBOND(III)
     ELSE
        MM = III
     ENDIF
     !      DO 10 III = 1,NACTBND
     !       MM = ACTBOND(III)
#else /* (actbond)*/
  DO MM=1,NBOND
#endif /* (actbond)*/
#endif /* (parabond)*/
     !
     I=IB(MM)
     IF(NOCONS) THEN
        IF(ISKB(MM).NE.0) GOTO 10
     ENDIF
     J=JB(MM)
     IC=ICB(MM)
     IF(IC.EQ.0) GOTO 10
     IF(CBC(IC).EQ.ZERO) GOTO 10
#if KEY_VALBOND==1
     IF(VBINID)THEN                             
#endif
#if KEY_VALBOND==1
       IF(VBMET(I).OR.VBMET(J)) THEN            
#endif
#if KEY_VALBOND==1
         IF (VBHYP(I) .OR. VBHYP(J)) GOTO 10    
#endif
#if KEY_VALBOND==1
       ENDIF                                    
#endif
#if KEY_VALBOND==1
     ENDIF                                      
#endif
     RX=X(I)-X(J)
     RY=Y(I)-Y(J)
     RZ=Z(I)-Z(J)
     
#if KEY_MSMMPT==1
     IF (PBCXSIZE.GT.0.D0 .AND. PBCYSIZE.GT.0.D0 .AND. PBCZSIZE.GT.0.D0) THEN
     
        IF (RX.LE.-0.5D0*PBCXSIZE) THEN
          RX=RX+PBCXSIZE
        ELSE IF (RX.GT.0.5D0*PBCXSIZE) THEN
          RX=RX-PBCXSIZE
        ENDIF
        
        IF (RY.LE.-0.5D0*PBCYSIZE) THEN
          RY=RY+PBCYSIZE
        ELSE IF (RY.GT.0.5D0*PBCYSIZE) THEN
          RY=RY-PBCYSIZE
        ENDIF
        
        IF (RZ.LE.-0.5D0*PBCZSIZE) THEN
          RZ=RZ+PBCZSIZE
        ELSE IF (RZ.GT.0.5D0*PBCZSIZE) THEN
          RZ=RZ-PBCZSIZE
        ENDIF
     ENDIF
#endif
                                                                              
     S2=RX*RX + RY*RY + RZ*RZ
#if KEY_FOURD==1 /*4dadd*/
     IF (DIM4ON(1).EQ.1) THEN
        RFDIM=FDIM(I)-FDIM(J)
        S2=S2 + RFDIM*RFDIM
     ENDIF
#endif /* (4dadd)*/
     S=SQRT(S2)
     IF(LUPPER .AND. (S.LT.CBB(IC))) GOTO 10
     IF(CBB(IC).EQ.ZERO) THEN
        DB=S2
        IF(KBEXPN.EQ.2) THEN
           R=TWO
           DF=CBC(IC)
           DDF=ZERO
        ELSE
           R=KBEXPN
           DF=CBC(IC)*S**(KBEXPN-2)
           DDF=(KBEXPN-2)*CBC(IC)*S**(KBEXPN-4)*R
        ENDIF
     ELSE
        IF(S.EQ.ZERO) GOTO 10
        DB=S-CBB(IC)
        IF(KBEXPN.EQ.2) THEN
           R=TWO/S
           DF=CBC(IC)*DB
           DDF=TWO*CBC(IC)
           DDF=(DDF-R*DF)/S2
        ELSE
           R=KBEXPN/S
           DF=CBC(IC)*DB**(KBEXPN-1)
           DDF=KBEXPN*(KBEXPN-1)*CBC(IC)*DB**(KBEXPN-2)
           DDF=(DDF-R*DF)/S2
        ENDIF
     ENDIF
     !
#if KEY_BLOCK==1
     IF (QBLOCK) THEN
        IBL = IBLCKP(I)
        JBL = IBLCKP(J)
#if KEY_DOCK==1
        !         get asymmetric matrix coefficient
        DOCFI = 1.0
        DOCFJ = 1.0
        IF(QDOCK) THEN
           KDOC  = (IBL - 1)*NBLOCK + JBL
           DOCFI = BLDOCP(KDOC)
           KDOC  = (JBL - 1)*NBLOCK + IBL
           DOCFJ = BLDOCP(KDOC)
        ENDIF
#endif /*  DOCK*/
        IF (JBL .LT. IBL) THEN
           KK=JBL
           JBL=IBL
           IBL=KK
        ENDIF
        KK=IBL+JBL*(JBL-1)/2
        COEF = BLCOEB(KK)
        IF (QPRNTV .AND. .NOT. QNOBO) THEN
           IF (IBL == 1 .OR. JBL == 1 .OR. IBL == JBL) THEN
              VBBOND(JBL) = VBBOND(JBL) + DF * DB
           ENDIF
        ENDIF
#if KEY_BLOCK==1 /*ldm_1*/ /*ldm*/
        IF (QLDM .or. QLMC) THEN
           ! first row or diagonal elements exclude (1,1).
           UNSCALE = 0.0
           IF ((IBL == 1 .AND. JBL >= LSTRT) .or. &
                (IBL >= LSTRT .AND. IBL == JBL)) UNSCALE = DF
        ENDIF
        IF(QNOBO) COEF = 1.0
        IF(RSTP.AND. .NOT. QNOBO)THEN
           IF ((IBL == 1 .AND. JBL >= LSTRT) .or. &
                (IBL >= LSTRT .AND. IBL == JBL)) THEN
              ENVDX(I) = ENVDX(I) + DF * R * RX
              ENVDY(I) = ENVDY(I) + DF * R * RY
              ENVDZ(I) = ENVDZ(I) + DF * R * RZ
              ENVDX(J) = ENVDX(J) - DF * R * RX
              ENVDY(J) = ENVDY(J) - DF * R * RY
              ENVDZ(J) = ENVDZ(J) - DF * R * RZ
           ENDIF
        ENDIF
#endif /*  (ldm_1)*/

#if KEY_SCCDFTB==1
#if KEY_PARALLEL==1
        IF (MYNOD.EQ.0) THEN
#endif 
           if(qsccb) then
              dvdl0=df*cdvdl(kk)
           endif
#if KEY_PARALLEL==1
        ENDIF
#endif 

#endif 
        DF=DF*COEF
        IF (QSECD) DDF=DDF*COEF
     ENDIF
#if KEY_BLOCK==1 /*ldm*/
     IF ((QLDM .or. QLMC) &
          .AND. .NOT. QNOBO) then
        IF ((IBL == 1 .AND. JBL >= LSTRT) .or. &
             (IBL >= LSTRT .AND. IBL == JBL)) THEN
           FALPHA = UNSCALE*DB
#if KEY_PERT==1
           if(QPERT) FALPHA = FALPHA*PERTLAM  !Cc New PBLOCK 
#endif
           LAGMUL = LAGMUL + FALPHA
           BIFLAM(JBL) = BIFLAM(JBL) + FALPHA
           IF (NRST == 2) THEN
              BFRST(JBL) = BFRST(JBL) + FALPHA
           ENDIF
        ENDIF
     ENDIF
#endif 
#if KEY_DOCK==1
     IF(QDOCK) THEN
        !         Factor 0.5 to make sure no double counting
        E=DF*DB*0.5*(DOCFI + DOCFJ)
     ELSE
#endif 
        E=DF*DB
#if KEY_DOCK==1
     ENDIF
#endif 
#else /*   BLOCK*/
     E=DF*DB
#endif /*  BLOCK*/
     EB=EB+E

#if KEY_SCCDFTB==1
#if KEY_PARALLEL==1
     IF (MYNOD.EQ.0) THEN
#endif 
        if(qsccb) then
           if(idxbnd.eq.0) dvdlb=dvdlb+dvdl0*db
           if(idxbnd.eq.1) dvdlub=dvdlub+dvdl0*db
        endif
#if KEY_PARALLEL==1
     ENDIF
#endif 

#endif 
     !
     IF(QATERM) THEN
        KK=ANSLCT(I)+ANSLCT(J)
        IF(KK.EQ.2 .OR. (KK.GE.1 .AND. .NOT.QAONLY)) THEN
           IF(QAUNIT.LT.0) THEN
              II=OUTU
           ELSE
              II=QAUNIT
           ENDIF
           !
           IF(PRNLEV.GE.5) THEN
              IF(QAFIRST) THEN
                 IF(QLONGL) THEN
                    WRITE(II,243)
                 ELSE
                    WRITE(II,244)
                 ENDIF
243              FORMAT('ANAL: BOND: Index        Atom-I               ', &
                      '    Atom-J                  Dist           Energy   ', &
                      '      Force            Parameters')
244              FORMAT('ANAL: BOND: Index        Atom-I               ', &
                      '    Atom-J          ',/ &
                      '        Dist           Energy   ', &
                      '      Force            Parameters')
                 QAFIRST=.FALSE.
              ENDIF
              CALL ATOMID(I,SIDDNI,RIDDNI,RESDNI,ACDNI)
              CALL ATOMID(J,SIDDNJ,RIDDNJ,RESDNJ,ACDNJ)
              IF(QLONGL) THEN
                 WRITE(II,245) MM,I,SIDDNI(1:idleng),RIDDNI(1:idleng), &
                      RESDNI(1:idleng),ACDNI(1:idleng), &
                      J,SIDDNJ(1:idleng),RIDDNJ(1:idleng), &
                      RESDNJ(1:idleng),ACDNJ(1:idleng), &
                      S,E,DF,IC,CBB(IC),CBC(IC)
              ELSE
                 WRITE(II,246) MM,I,SIDDNI(1:idleng),RIDDNI(1:idleng), &
                      RESDNI(1:idleng),ACDNI(1:idleng), &
                      J,SIDDNJ(1:idleng),RIDDNJ(1:idleng), &
                      RESDNJ(1:idleng),ACDNJ(1:idleng), &
                      S,E,DF,IC,CBB(IC),CBC(IC)
              ENDIF
245           FORMAT('ANAL: BOND>',2I5,4(1X,A),I5,4(1X,A), &
                   3F15.6,I5,2F15.6)
246           FORMAT('ANAL: BOND>',2I5,4(1X,A),I5,4(1X,A),/ &
                   3F15.6,I5,2F15.6)
           ENDIF
        ENDIF
     ENDIF
     !
     IF(QECONTX) THEN
        E=E*HALF
        ECONTX(I)=ECONTX(I)+E
        ECONTX(J)=ECONTX(J)+E
     ENDIF
#if KEY_BLOCK==1
     IF (.NOT. NOFORC) THEN
#endif /*  BLOCK*/
        DF=DF*R
        !
        DXI=RX*DF
        DYI=RY*DF
        DZI=RZ*DF
#if KEY_BLOCK==1
#if KEY_DOCK==1
        IF(QDOCK) THEN
           DX(I)=DX(I)+DXI*DOCFI
           DY(I)=DY(I)+DYI*DOCFI
           DZ(I)=DZ(I)+DZI*DOCFI
           DX(J)=DX(J)-DXI*DOCFJ
           DY(J)=DY(J)-DYI*DOCFJ
           DZ(J)=DZ(J)-DZI*DOCFJ
        ELSE
#endif /* DOCK*/
#endif 
           DX(I)=DX(I)+DXI
           DY(I)=DY(I)+DYI
           DZ(I)=DZ(I)+DZI
           DX(J)=DX(J)-DXI
           DY(J)=DY(J)-DYI
           DZ(J)=DZ(J)-DZI
#if KEY_BLOCK==1
#if KEY_DOCK==1
        ENDIF
#endif 
#endif 
#if KEY_FOURD==1 /*4daddf*/
        IF(DIM4ON(1).EQ.1) THEN
           DFDI=RFDIM*DF
           DFDIM(I)=DFDIM(I)+DFDI
           DFDIM(J)=DFDIM(J)-DFDI
        ENDIF
#endif /* (4daddf)*/
        !
#if KEY_IPRESS==1
        IF(QIPRSS) THEN
           PVIR(I)=PVIR(I)+S2*DF
           PVIR(J)=PVIR(J)+S2*DF
        ENDIF
#endif 
        !
        !       SECOND DERIVATIVE PART
        IF (QSECD) THEN

#if KEY_DIMB==1
           IF(QCMPCT) THEN
              CALL EBNCMP(I,J,RX,RY,RZ,DF,DDF,DD1,PINBCM, &
                   PJNBCM,KBEXPN)
           ELSE
#endif /*  DIMB*/

           IF(KBEXPN .NE. 2) THEN
              CALL WRNDIE(-3,'<EBOND> ','NO SECOND DERIV. FOR ' &
                   //'ANHARMONIC BONDS')
           ENDIF
           !
           IF (J.LT.I) THEN
              JJ=3*I-2
              II=3*J-2
           ELSE
              JJ=3*J-2
              II=3*I-2
           ENDIF
           !
           A=RX*RX*DDF+DF
           IADD=IUPT(II)+JJ
           DD1(IADD)=DD1(IADD)-A
           IADD=IUPT(II)+II
           DD1(IADD)=DD1(IADD)+A
           IADD=IUPT(JJ)+JJ
           DD1(IADD)=DD1(IADD)+A
           !
           A=RY*RY*DDF+DF
           IADD=IUPT(II+1)+JJ+1
           DD1(IADD)=DD1(IADD)-A
           IADD=IUPT(II+1)+II+1
           DD1(IADD)=DD1(IADD)+A
           IADD=IUPT(JJ+1)+JJ+1
           DD1(IADD)=DD1(IADD)+A
           !
           A=RZ*RZ*DDF+DF
           IADD=IUPT(II+2)+JJ+2
           DD1(IADD)=DD1(IADD)-A
           IADD=IUPT(II+2)+II+2
           DD1(IADD)=DD1(IADD)+A
           IADD=IUPT(JJ+2)+JJ+2
           DD1(IADD)=DD1(IADD)+A
           !
           A=RX*RY*DDF
           IADD=IUPT(II)+JJ+1
           DD1(IADD)=DD1(IADD)-A
           IADD=IUPT(II+1)+JJ
           DD1(IADD)=DD1(IADD)-A
           IADD=IUPT(II)+II+1
           DD1(IADD)=DD1(IADD)+A
           IADD=IUPT(JJ)+JJ+1
           DD1(IADD)=DD1(IADD)+A
           !
           A=RX*RZ*DDF
           IADD=IUPT(II)+JJ+2
           DD1(IADD)=DD1(IADD)-A
           IADD=IUPT(II+2)+JJ
           DD1(IADD)=DD1(IADD)-A
           IADD=IUPT(II)+II+2
           DD1(IADD)=DD1(IADD)+A
           IADD=IUPT(JJ)+JJ+2
           DD1(IADD)=DD1(IADD)+A
           !
           A=RY*RZ*DDF
           IADD=IUPT(II+1)+JJ+2
           DD1(IADD)=DD1(IADD)-A
           IADD=IUPT(II+2)+JJ+1
           DD1(IADD)=DD1(IADD)-A
           IADD=IUPT(II+1)+II+2
           DD1(IADD)=DD1(IADD)+A
           IADD=IUPT(JJ+1)+JJ+2
           DD1(IADD)=DD1(IADD)+A
#if KEY_DIMB==1
        ENDIF  
#endif
     ENDIF

#if KEY_BLOCK==1
     ENDIF
#endif /*  BLOCK*/

10   CONTINUE
  ENDDO
  RETURN
END SUBROUTINE EBOND

SUBROUTINE EANGLE(ET,IT,JT,KT,ICT,NTHETA,CTC,CTB,DX,DY,DZ,X,Y,Z, &
     QECONTX,ECONTX,ICONAH,ISKT,DD1,IUPT,QSECD &
     )

  !-----------------------------------------------------------------------
  !     CALCULATES BOND ANGLES AND BOND ANGLE ENERGIES.
  !     IT DOES NOT CALCULATE ANGLE ENERGY AND FORCES FOR ALL ANGLES
  !     WITH A NONZERO VALUE IN ISKT ARRAY ONLY WHEN ICONAH .NE. 0
  !
  !     By Bernard R. Brooks  (mostly)  1981-1983
  !
  use chm_kinds
  use dimens_fcm
  use number
#if KEY_ACTBOND==1
  use actclus_mod   
#endif
  use block_fcm
#if KEY_SCCDFTB==1
  use blockscc_fcm  
#endif
  use fourdm
#if KEY_BLOCK==1 || KEY_SPACDEC==1
#endif /*  BLOCK SPACDEC*/
  use block_fcm
  use lambdam
  use pert  !Cc New PBLOCK
  use dimb
#if KEY_PARALLEL==1
  use parallel     
#endif
  use stream
  use consta
  use econtmod
  use chutil,only:atomid
  implicit none
#if KEY_PARALLEL==1
  LOGICAL NOPARS
#endif 
#if KEY_BLOCK==1
  real(chm_real) DFORG   /*ldm*/
#endif
  real(chm_real) ET
  INTEGER IT(*),JT(*),KT(*),ICT(*)
  INTEGER NTHETA
  real(chm_real) CTC(*),CTB(*)
  real(chm_real) DX(*),DY(*),DZ(*),X(*),Y(*),Z(*)
  LOGICAL QECONTX
  real(chm_real) ECONTX(*)
  INTEGER ICONAH
  INTEGER ISKT(*)
  real(chm_real) DD1(*)
  INTEGER IUPT(*)
  LOGICAL QSECD
  !
  real(chm_real) DXI,DYI,DZI,DXJ,DYJ,DZJ,RI2,RJ2,RI,RJ
  real(chm_real) RIR,RJR,DXIR,DYIR,DZIR,DXJR,DYJR,DZJR,CST,AT, &
       DA,DF,E
  real(chm_real) ST2R,STR,DTXI,DTXJ,DTYI,DTYJ,DTZI,DTZJ
  real(chm_real) DFX,DFY,DFZ,DGX,DGY,DGZ,DDF,RI2RF,RJ2RF,RIRJF
  real(chm_real) DDXIXI,DDYIYI,DDZIZI,DDXJXJ,DDYJYJ,DDZJZJ
  real(chm_real) DDXIXJ,DDYIYJ,DDZIZJ,DDXIYI,DDXIZI,DDYIZI
  real(chm_real) DDXJYJ,DDXJZJ,DDYJZJ,DDXIYJ,DDYIXJ,DDXIZJ
  real(chm_real) DDZIXJ,DDYIZJ,DDZIYJ,A,SMALLV
  real(chm_real) ST,ST2
  INTEGER I,NWARN,ITH,J,K,IC,JJ,II,KK,IADD
  LOGICAL NOCONS,IJTEST,IKTEST,JKTEST,QAFIRST
  CHARACTER(len=8) SIDDNI,RIDDNI,RESDNI,ACDNI,SIDDNJ,RIDDNJ,RESDNJ,ACDNJ
  CHARACTER(len=8) SIDDNK,RIDDNK,RESDNK,ACDNK
  !
#if KEY_FOURD==1 /*4ddecl*/
  !     4-D variables:
  real(chm_real) DFDIMI,DFDIMJ,DFDIMIR,DFDIMJR,DTFDI,DTFDJ,DFFD,DGFD
#endif /* (4ddecl)*/
#if KEY_BLOCK==1
  real(chm_real) COEF, DOCFI, DOCFJ, DOCFK, DOCFJ1
  INTEGER IBL, JBL, KDOC
!ldm
  real(chm_real) UNSCALE, FALPHA
!ldm
#endif /*  BLOCK*/
#if KEY_ACTBOND==1 /*actbond0*/
  INTEGER III, UPLIMIT
#endif /* (actbond0)*/
  !
  ! ********************************************************************
  !...##IF ACTBOND (actbond0)
  !      IF(.NOT.QBACTON) CALL BACTIVDEF
  !...##ENDIF (actbond0)
  !
  ET=ZERO
#if KEY_SCCDFTB==1
  if(qsccb) dvdla=zero  
#endif
  SMALLV=RPRECI
  NWARN=0
  NOCONS=(ICONAH.GT.0)
  IF(NTHETA.EQ.0) RETURN
  QAFIRST=.TRUE.
  !
#if KEY_PARALLEL==1 /*paraangle*/
#if KEY_VIBPARA==1
  DO ITH=1,NTHETA
#else /**/

#if KEY_PARAFULL==1 /*parfangle*/
  DO ITH=MYNODP,NTHETA,NUMNOD
#elif KEY_PARASCAL==1 /*parfangle*/
  NOPARS=(ICONAH.GE.0)
  DO ITH=1,NTHETA
     IF(NOPARS) THEN
        II=IPBLOCK(IT(ITH))
        JJ=IPBLOCK(JT(ITH))
        KK=IPBLOCK(KT(ITH))
        IC=II
        IF(II.EQ.JJ) IC=KK
        IF(KK.NE.JJ .AND. KK.NE.IC) THEN
           !           the angle spans three block.
           !           make sure that we have the coordinates.
           CALL PSADDTOCL(KT(ITH),JPMAT(JJ,IC))
        ENDIF
        IF(JPMAT(JJ,IC).NE.MYNOD) GOTO 20
     ENDIF
#elif KEY_SPACDEC==1 /*parfangle*/
  DO ITH=1,NTHETA
     IF(ICPUMAP(IT(ITH)) /= MYNOD) GOTO 20
#endif /* (parfangle)*/
#endif 
#else /* (paraangle)*/
#if KEY_ACTBOND==1 /*actbond*/
  IF(QBACTON) THEN
     UPLIMIT = NACTANG
  ELSE
     UPLIMIT = NTHETA
  ENDIF
  DO III = 1,UPLIMIT
     IF (QBACTON) THEN
        ITH = ACTANGL(III)
     ELSE
        ITH = III
     ENDIF
     !      DO 20 III = 1,NACTANG
     !       ITH = ACTANGL(III)
#else /* (actbond)*/
  DO ITH=1,NTHETA
#endif /* (actbond)*/
#endif /* (paraangle)*/
     !
     I=IT(ITH)
     IF(NOCONS) THEN
        IF(ISKT(ITH).NE.0) GOTO 20
     ENDIF
     J=JT(ITH)
     K=KT(ITH)
     IC=ICT(ITH)
     IF(IC.EQ.0) GOTO 20
     DXI=X(I)-X(J)
     DYI=Y(I)-Y(J)
     DZI=Z(I)-Z(J)
     DXJ=X(K)-X(J)
     DYJ=Y(K)-Y(J)
     DZJ=Z(K)-Z(J)
     
#if KEY_MSMMPT==1
     IF (PBCXSIZE.GT.0.D0 .AND. PBCYSIZE.GT.0.D0 .AND. &
     PBCZSIZE.GT.0.D0) THEN
     
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
        
        IF (DXJ.LE.-0.5D0*PBCXSIZE) THEN
          DXJ=DXJ+PBCXSIZE
        ELSE IF (DXJ.GT.0.5D0*PBCXSIZE) THEN
          DXJ=DXJ-PBCXSIZE
        ENDIF
        
        IF (DYJ.LE.-0.5D0*PBCYSIZE) THEN
          DYJ=DYJ+PBCYSIZE
        ELSE IF (DYJ.GT.0.5D0*PBCYSIZE) THEN
          DYJ=DYJ-PBCYSIZE
        ENDIF
        
        IF (DZJ.LE.-0.5D0*PBCZSIZE) THEN
          DZJ=DZJ+PBCZSIZE
        ELSE IF (DZJ.GT.0.5D0*PBCZSIZE) THEN
          DZJ=DZJ-PBCZSIZE
        ENDIF
     ENDIF
#endif
                                                         
     RI2=DXI*DXI+DYI*DYI+DZI*DZI
     RJ2=DXJ*DXJ+DYJ*DYJ+DZJ*DZJ
#if KEY_FOURD==1 /*4dang1*/
     IF (DIM4ON(2).EQ.1) THEN
        DFDIMI=FDIM(I)-FDIM(J)
        DFDIMJ=FDIM(K)-FDIM(J)
        RI2=RI2+DFDIMI*DFDIMI
        RJ2=RJ2+DFDIMJ*DFDIMJ
     ENDIF
#endif /* (4dang1)*/
     RI=SQRT(RI2)
     RJ=SQRT(RJ2)
     RIR=ONE/RI
     RJR=ONE/RJ
     DXIR=DXI*RIR
     DYIR=DYI*RIR
     DZIR=DZI*RIR
     DXJR=DXJ*RJR
     DYJR=DYJ*RJR
     DZJR=DZJ*RJR
     CST=DXIR*DXJR+DYIR*DYJR+DZIR*DZJR
#if KEY_FOURD==1 /*4dang2*/
     IF(DIM4ON(2).EQ.1) THEN
        DFDIMIR=DFDIMI*RIR
        DFDIMJR=DFDIMJ*RJR
        CST=CST+DFDIMIR*DFDIMJR
     ENDIF
#endif /* (4dang2)*/
     !
     !
     ! IMPORTANT NOTE: the CHARMM potential has issues related to
     ! dividing by zero when you take the derivative w.r.t. cos
     ! theta. The GROMACS potential does *NOT*. So, we write
     ! everything out explicitly here and skip the "flat angle"
     ! testing as well as the multiplication by 1/sin theta later
     ! on. Don't add the 1/sin theta business back in unless you 1)
     ! have a good reason 2) feel like double-checking for
     ! divide-by-zero errors all over the place.
     !

     IF((ABS(CST).GE.COSMAX).AND.(CTB(IC).GE.0)) THEN
        IF(ABS(CST).GT.ONE) CST=SIGN(ONE,CST)
        AT=ACOS(CST)
        IF(CTB(IC).GE.ZERO) THEN
           DA=AT-CTB(IC)
        ELSE
           DA=AT-ACOS(ONE+CTB(IC))
        ENDIF
        IF(ABS(DA).GT.0.1) THEN
           NWARN=NWARN+1
           IF((NWARN.LE.5 .AND. WRNLEV.GE.5) .OR. WRNLEV.GE.6) THEN
              WRITE(OUTU,10) ITH,I,J,K
10            FORMAT(' WARNING FROM EANGLE. Angle',I5, &
                   '  is almost linear.', &
                   /' Derivatives may be affected for atoms:',3I5)
              WRITE(OUTU,101) 'I ATOM:',X(I),Y(I),Z(I)
              WRITE(OUTU,101) 'J ATOM:',X(J),Y(J),Z(J)
              WRITE(OUTU,101) 'K ATOM:',X(K),Y(K),Z(K)
              WRITE(OUTU,101) 'DXIR  :',DXIR,DYIR,DZIR
              WRITE(OUTU,101) 'DXJR  :',DXJR,DYJR,DZJR
              WRITE(OUTU,101) 'CST   :',CST,AT*RADDEG,DA*RADDEG
101           FORMAT(5X,A,5F15.5)
           ENDIF
        ENDIF
     ENDIF
     !
     AT=ACOS(CST)
     !
     ! Check if we want to use the cosine potential (negative angle) or
     ! the normal CHARMM potential
     ! CTB(IC) = theta_0
     ! CTC(IC) = K
     IF (CTB(IC).GE.0) THEN
        IF(PRNLEV.GT.6) THEN
           WRITE(OUTU,'(A,I5)') 'EANGLE> Using CHARMM angle function for angle ', IC
        ENDIF
        DA=AT-CTB(IC)
        DF=CTC(IC)*DA
        DDF=CTC(IC)
        E=DF*DA
     ELSE
        IF(PRNLEV.GT.6) THEN
           WRITE(OUTU,'(A,I5)') 'EANGLE> Using GROMACS-style angle function for angle ', IC
        ENDIF
        DA=CST-(CTB(IC)+ONE) ! = cos(theta)-cos(theta_not)
        ! This is actually DF/2 because the generic code does DF=DF+DF later.
        !E=(CTC(IC)*DA*DA)/2.0
        DF= CTC(IC)*DA/2.0
        E= DF*DA
        ! Tim's first pass at second derivatives. Untested.
        ! DDF=-(CTC(IC)*(DA*CST-ST2))/2.0 
     ENDIF
#if KEY_BLOCK==1
     IF (QBLOCK) THEN
        IBL = IBLCKP(I)
        JBL = IBLCKP(J)
        KK = IBLCKP(K)
#if KEY_DOCK==1
        !         two pairs in an angle (i,j), (k,j)
        DOCFI = 1.0
        DOCFJ = 1.0
        DOCFK = 1.0
        DOCFJ1 = 1.0
        IF(QDOCK) THEN
           KDOC  = (IBL - 1)*NBLOCK + JBL
           DOCFI = BLDOCP(KDOC)
           KDOC  = (JBL - 1)*NBLOCK + IBL
           DOCFJ = BLDOCP(KDOC)
           KDOC  = (KK - 1)*NBLOCK + JBL
           DOCFK = BLDOCP(KDOC)
           KDOC  = (JBL - 1)*NBLOCK + KK
           DOCFJ1 = BLDOCP(KDOC)
        ENDIF
#endif /*  DOCK*/
        IF (IBL .EQ. JBL) JBL=KK
        IF (JBL .LT. IBL) THEN
           KK=JBL
           JBL=IBL
           IBL=KK
        ENDIF
        KK=IBL+JBL*(JBL-1)/2
        COEF = BLCOEA(KK)
        IF (QPRNTV .AND. .NOT. QNOAN) THEN
           IF (IBL == 1 .OR. JBL == 1 .OR. IBL == JBL) THEN
              VBANG(JBL) = VBANG(JBL) + DF * DA
           ENDIF
        ENDIF
#if KEY_BLOCK==1 /*ldm_2*/ /*ldm*/
        IF (QLDM .or. QLMC) THEN
           ! first row or diagonal elements exclude (1,1).
           UNSCALE = 0.0
           IF ((IBL == 1 .AND. JBL >= LSTRT) .or. &
                (IBL >= LSTRT .AND. IBL == JBL)) UNSCALE = DF
        ENDIF
        IF(QNOAN) COEF = 1.0
        IF (RSTP .AND. .NOT. QNOAN) THEN
           IF ((IBL == 1 .AND. JBL >= LSTRT) .or. &
                (IBL >= LSTRT .AND. IBL == JBL)) THEN
              DFORG = DF
           ENDIF
        ENDIF
#endif /* (ldm_2)*/

#if KEY_SCCDFTB==1
#if KEY_PARALLEL==1
        IF (MYNOD.EQ.0) THEN
#endif 

           if(qsccb) then
              dvdl0=df*cdvdl(kk)
           endif
#if KEY_PARALLEL==1
        ENDIF
#endif 

#endif 

        E=E*COEF
        DF=DF*COEF
        DDF=DDF*COEF
     ENDIF
#if KEY_BLOCK==1 /*ldm*/
     IF ((QLDM .or. QLMC) .AND. .NOT. QNOAN) THEN
        IF ((IBL == 1 .AND. JBL >= LSTRT) .or. &
             (IBL >= LSTRT .AND. IBL == JBL)) THEN
           FALPHA = UNSCALE*DA
#if KEY_PERT==1
           if(QPERT) FALPHA = FALPHA*PERTLAM  !Cc New PBLOCK 
#endif
           LAGMUL = LAGMUL + FALPHA
           BIFLAM(JBL) = BIFLAM(JBL) + FALPHA
           IF (NRST == 2) THEN
              BFRST(JBL) = BFRST(JBL) + FALPHA
           ENDIF
        ENDIF
     ENDIF
#endif /* LDM*/
#if KEY_DOCK==1
     IF(QDOCK) THEN
        !         Factor 0.25 to make sure no double counting
        E=E*0.25*(DOCFI+DOCFJ+DOCFK+DOCFJ1)
     ENDIF
#endif 
#endif /*  BLOCK*/
     !
     ET=ET+E

#if KEY_SCCDFTB==1
#if KEY_PARALLEL==1
     IF (MYNOD.EQ.0) THEN
#endif 

        if(qsccb) then
           dvdla=dvdla+dvdl0*da
        endif
#if KEY_PARALLEL==1
     ENDIF
#endif 

#endif 

     DF=DF+DF
     !
     IF(QATERM) THEN
        KK=ANSLCT(I)+ANSLCT(J)+ANSLCT(K)
        IF(KK.EQ.3 .OR. (KK.GE.1 .AND. .NOT.QAONLY)) THEN
           IF(QAUNIT.LT.0) THEN
              II=OUTU
           ELSE
              II=QAUNIT
           ENDIF
           !
           IF(PRNLEV.GE.5) THEN
              IF(QAFIRST) THEN
                 IF(QLONGL) THEN
                    WRITE(II,243)
                 ELSE
                    WRITE(II,244)
                 ENDIF
243              FORMAT('ANAL: ANGL: Index        Atom-I             ', &
                      '      Atom-J                   Atom-K                ', &
                      '  Angle          Energy   ', &
                      '      Force            Parameters')
244              FORMAT('ANAL: ANGL: Index        Atom-I             ', &
                      '      Atom-J                   Atom-K          ', &
                      /'        Angle          Energy   ', &
                      '      Force            Parameters')
                 QAFIRST=.FALSE.
              ENDIF
              CALL ATOMID(I,SIDDNI,RIDDNI,RESDNI,ACDNI)
              CALL ATOMID(J,SIDDNJ,RIDDNJ,RESDNJ,ACDNJ)
              CALL ATOMID(K,SIDDNK,RIDDNK,RESDNK,ACDNK)
              IF(QLONGL) THEN
                 WRITE(II,246) ITH,I,SIDDNI(1:idleng),RIDDNI(1:idleng), &
                      RESDNI(1:idleng),ACDNI(1:idleng), &
                      J,SIDDNJ(1:idleng),RIDDNJ(1:idleng), &
                      RESDNJ(1:idleng),ACDNJ(1:idleng), &
                      K,SIDDNK(1:idleng),RIDDNK(1:idleng), &
                      RESDNK(1:idleng),ACDNK(1:idleng), &
                      AT*RADDEG,E,DF,IC,CTB(IC)*RADDEG,CTC(IC)
              ELSE
                 WRITE(II,245) ITH,I,SIDDNI(1:idleng),RIDDNI(1:idleng), &
                      RESDNI(1:idleng),ACDNI(1:idleng), &
                      J,SIDDNJ(1:idleng),RIDDNJ(1:idleng), &
                      RESDNJ(1:idleng),ACDNJ(1:idleng), &
                      K,SIDDNK(1:idleng),RIDDNK(1:idleng), &
                      RESDNK(1:idleng),ACDNK(1:idleng), &
                      AT*RADDEG,E,DF,IC,CTB(IC)*RADDEG,CTC(IC)
              ENDIF
245           FORMAT('ANAL: ANGL>',2I5,4(1X,A),I5,4(1X,A),I5,4(1X,A), &
                   /3F15.6,I5,2F15.6)
246           FORMAT('ANAL: ANGL>',2I5,4(1X,A),I5,4(1X,A),I5,4(1X,A), &
                   3F15.6,I5,2F15.6)
           ENDIF
        ENDIF
     ENDIF
     !
     IF(QECONTX) THEN
        E=E*THIRD
        ECONTX(I)=ECONTX(I)+E
        ECONTX(J)=ECONTX(J)+E
        ECONTX(K)=ECONTX(K)+E
     ENDIF
#if KEY_BLOCK==1
     IF (.NOT. NOFORC) THEN
#endif /*  BLOCK*/
        !
        IF(CTB(IC).GE.0) THEN
           ! As mentioned above, no funny business necessary with GROMACS
           IF(ABS(CST).GE.0.99999) THEN
              IF(PRNLEV.GE.8) THEN
                 WRITE(OUTU,'(A)') 'EANGLE> FORCE: NON-GROMACS ANGLE IS FLAT'
              ENDIF
              ST2R=ONE/(ONE-CST*CST+SMALLV)
              STR=SQRT(ST2R)
              IF(CTB(IC).LT.PT001) THEN
                 DF=MINTWO*CTC(IC)*(ONE+DA*DA*SIXTH)
              ELSE IF(PI-CTB(IC).LT.PT001) THEN
                 DF=TWO*CTC(IC)*(ONE+DA*DA*SIXTH)
              ELSE
                 DF=-DF*STR
              ENDIF
           ELSE
              IF(PRNLEV.GE.8) THEN
                 WRITE(OUTU,'(A)') 'EANGLE> FORCE: ANGLE NOT FLAT'
              ENDIF
              ST2R=ONE/(ONE-CST*CST)
              STR=SQRT(ST2R)
              DF=-DF*STR
           ENDIF
        ENDIF
        !
        DTXI=RIR*(DXJR-CST*DXIR)
        DTXJ=RJR*(DXIR-CST*DXJR)
        DTYI=RIR*(DYJR-CST*DYIR)
        DTYJ=RJR*(DYIR-CST*DYJR)
        DTZI=RIR*(DZJR-CST*DZIR)
        DTZJ=RJR*(DZIR-CST*DZJR)
        !
#if KEY_BLOCK==1
#if KEY_DOCK==1
        IF(QDOCK) THEN
           DFX=DF*DTXI
           DGX=DF*DTXJ
           DX(I)=DX(I)+DFX*DOCFI
           DX(K)=DX(K)+DGX*DOCFK
           DX(J)=DX(J)-DFX*DOCFJ-DGX*DOCFJ1
           !
           DFY=DF*DTYI
           DGY=DF*DTYJ
           DY(I)=DY(I)+DFY*DOCFI
           DY(K)=DY(K)+DGY*DOCFK
           DY(J)=DY(J)-DFY*DOCFJ-DGY*DOCFJ1
           !
           DFZ=DF*DTZI
           DGZ=DF*DTZJ
           DZ(I)=DZ(I)+DFZ*DOCFI
           DZ(K)=DZ(K)+DGZ*DOCFK
           DZ(J)=DZ(J)-DFZ*DOCFJ-DGZ*DOCFJ1
        ELSE
#endif 
#endif 

           DFX=DF*DTXI
           DGX=DF*DTXJ
           DX(I)=DX(I)+DFX
           DX(K)=DX(K)+DGX
           DX(J)=DX(J)-DFX-DGX
           !
           DFY=DF*DTYI
           DGY=DF*DTYJ
           DY(I)=DY(I)+DFY
           DY(K)=DY(K)+DGY
           DY(J)=DY(J)-DFY-DGY
           !
           DFZ=DF*DTZI
           DGZ=DF*DTZJ
           DZ(I)=DZ(I)+DFZ
           DZ(K)=DZ(K)+DGZ
           DZ(J)=DZ(J)-DFZ-DGZ
#if KEY_BLOCK==1
#if KEY_DOCK==1
        ENDIF
#endif 
#endif 
#if KEY_BLOCK==1 /*ldm*/
        IF(RSTP.AND. .NOT. QNOAN)THEN
           IF ((IBL == 1 .AND. JBL >= LSTRT) .or. &
                (IBL >= LSTRT .AND. IBL == JBL)) THEN
              ENVDX(I) = ENVDX(I) + DFORG * DTXI
              ENVDY(I) = ENVDY(I) + DFORG * DTYI
              ENVDZ(I) = ENVDZ(I) + DFORG * DTZI
              ENVDX(K) = ENVDX(K) + DFORG * DTXJ
              ENVDY(K) = ENVDY(K) + DFORG * DTYJ
              ENVDZ(K) = ENVDZ(K) + DFORG * DTZJ
              ENVDX(J) = ENVDX(J) - DFORG * (DTXI + DTXJ)
              ENVDY(J) = ENVDY(J) - DFORG * (DTYI + DTYJ)
              ENVDZ(J) = ENVDZ(J) - DFORG * (DTZI + DTZJ)
           ENDIF
        ENDIF
#endif /*   LDM*/
#if KEY_FOURD==1 /*4dang3*/
        IF(DIM4ON(2).EQ.1) THEN
           DTFDI=RIR*(DFDIMJR-CST*DFDIMIR)
           DTFDJ=RJR*(DFDIMIR-CST*DFDIMJR)
           DFFD=DF*DTFDI
           DGFD=DF*DTFDJ
           DFDIM(I)=DFDIM(I)+DFFD
           DFDIM(K)=DFDIM(K)+DGFD
           DFDIM(J)=DFDIM(J)-DFFD-DGFD
        ENDIF
#endif /* (4dang3)*/
        !
#if KEY_IPRESS==1
        IF(QIPRSS) THEN
           PVIR(I)=PVIR(I)+DFX*DXI+DFY*DYI+DFZ*DZI
           PVIR(J)=PVIR(J)+DFX*DXI+DFY*DYI+DFZ*DZI+DGX*DXJ+DGY*DYJ+DGZ*DZJ
           PVIR(K)=PVIR(K)+DGX*DXJ+DGY*DYJ+DGZ*DZJ
        ENDIF
#endif 
        !       SECOND DERIVATIVE PART
        IF (QSECD) THEN
           !
           IF (CTB(IC).LT.0) THEN
              CALL WRNDIE(-3,'<EANGLE> ','NO SECOND DERIV. FOR ' &
                   //'GROMACS ANGLES')
           ENDIF

           DDF=TWO*DDF*ST2R*(ONE-CST*STR*DA)
           !
           RI2RF=RIR*RIR*DF
           RJ2RF=RJR*RJR*DF
           RIRJF=RIR*RJR*DF
           !
           DDXIXI=RI2RF*(CST*(DXIR*DXIR-ONE)-TWO*DXI*DTXI)+DDF*DTXI*DTXI
           DDYIYI=RI2RF*(CST*(DYIR*DYIR-ONE)-TWO*DYI*DTYI)+DDF*DTYI*DTYI
           DDZIZI=RI2RF*(CST*(DZIR*DZIR-ONE)-TWO*DZI*DTZI)+DDF*DTZI*DTZI
           DDXJXJ=RJ2RF*(CST*(DXJR*DXJR-ONE)-TWO*DXJ*DTXJ)+DDF*DTXJ*DTXJ
           DDYJYJ=RJ2RF*(CST*(DYJR*DYJR-ONE)-TWO*DYJ*DTYJ)+DDF*DTYJ*DTYJ
           DDZJZJ=RJ2RF*(CST*(DZJR*DZJR-ONE)-TWO*DZJ*DTZJ)+DDF*DTZJ*DTZJ
           !
           DDXIXJ=RIRJF*(ONE-DXIR*DXIR-DXJR*DXJR+CST*DXIR*DXJR)+DDF &
                *DTXI*DTXJ
           DDYIYJ=RIRJF*(ONE-DYIR*DYIR-DYJR*DYJR+CST*DYIR*DYJR)+DDF &
                *DTYI*DTYJ
           DDZIZJ=RIRJF*(ONE-DZIR*DZIR-DZJR*DZJR+CST*DZIR*DZJR)+DDF &
                *DTZI*DTZJ
           !
           DDXIYI=RI2RF*(CST*DXIR*DYIR-DXI*DTYI-DYI*DTXI)+DDF*DTXI*DTYI
           DDXIZI=RI2RF*(CST*DXIR*DZIR-DXI*DTZI-DZI*DTXI)+DDF*DTXI*DTZI
           DDYIZI=RI2RF*(CST*DYIR*DZIR-DYI*DTZI-DZI*DTYI)+DDF*DTYI*DTZI
           !
           DDXJYJ=RJ2RF*(CST*DXJR*DYJR-DXJ*DTYJ-DYJ*DTXJ)+DDF*DTXJ*DTYJ
           DDXJZJ=RJ2RF*(CST*DXJR*DZJR-DXJ*DTZJ-DZJ*DTXJ)+DDF*DTXJ*DTZJ
           DDYJZJ=RJ2RF*(CST*DYJR*DZJR-DYJ*DTZJ-DZJ*DTYJ)+DDF*DTYJ*DTZJ
           !
           A=DXIR*DYIR+DXJR*DYJR
           DDXIYJ=RIRJF*(CST*DXIR*DYJR-A)+DDF*DTXI*DTYJ
           DDYIXJ=RIRJF*(CST*DYIR*DXJR-A)+DDF*DTYI*DTXJ
           A=DXIR*DZIR+DXJR*DZJR
           DDXIZJ=RIRJF*(CST*DXIR*DZJR-A)+DDF*DTXI*DTZJ
           DDZIXJ=RIRJF*(CST*DZIR*DXJR-A)+DDF*DTZI*DTXJ
           A=DYIR*DZIR+DYJR*DZJR
           DDYIZJ=RIRJF*(CST*DYIR*DZJR-A)+DDF*DTYI*DTZJ
           DDZIYJ=RIRJF*(CST*DZIR*DYJR-A)+DDF*DTZI*DTYJ
           !
           !
#if KEY_DIMB==1
           IF(QCMPCT) THEN
              CALL EANCMP(I,J,K,DDXIXI,DDYIYI,DDZIZI,DDXJXJ,DDYJYJ, &
                   DDZJZJ,DDXIXJ,DDYIYJ,DDZIZJ,DDXIYI,DDXIZI, &
                   DDYIZI,DDXJYJ,DDXJZJ,DDYJZJ,DDXIYJ,DDYIXJ, &
                   DDXIZJ,DDZIXJ,DDYIZJ,DDZIYJ,DD1, &
                   PINBCM,PJNBCM)
           ELSE
#endif /*  DIMB*/

              II=3*I-2
              JJ=3*J-2
              KK=3*K-2
              IJTEST=(J.LT.I)
              IKTEST=(K.LT.I)
              JKTEST=(K.LT.J)
              !
              IF (IKTEST) THEN
                 IADD=IUPT(KK)+II
                 DD1(IADD)=DD1(IADD)+DDXIXJ
                 IADD=IUPT(KK+1)+II+1
                 DD1(IADD)=DD1(IADD)+DDYIYJ
                 IADD=IUPT(KK+2)+II+2
                 DD1(IADD)=DD1(IADD)+DDZIZJ
                 IADD=IUPT(KK)+II+1
                 DD1(IADD)=DD1(IADD)+DDYIXJ
                 IADD=IUPT(KK+1)+II
                 DD1(IADD)=DD1(IADD)+DDXIYJ
                 IADD=IUPT(KK)+II+2
                 DD1(IADD)=DD1(IADD)+DDZIXJ
                 IADD=IUPT(KK+2)+II
                 DD1(IADD)=DD1(IADD)+DDXIZJ
                 IADD=IUPT(KK+1)+II+2
                 DD1(IADD)=DD1(IADD)+DDZIYJ
                 IADD=IUPT(KK+2)+II+1
                 DD1(IADD)=DD1(IADD)+DDYIZJ
              ELSE
                 IADD=IUPT(II)+KK
                 DD1(IADD)=DD1(IADD)+DDXIXJ
                 IADD=IUPT(II+1)+KK+1
                 DD1(IADD)=DD1(IADD)+DDYIYJ
                 IADD=IUPT(II+2)+KK+2
                 DD1(IADD)=DD1(IADD)+DDZIZJ
                 IADD=IUPT(II+1)+KK
                 DD1(IADD)=DD1(IADD)+DDYIXJ
                 IADD=IUPT(II)+KK+1
                 DD1(IADD)=DD1(IADD)+DDXIYJ
                 IADD=IUPT(II+2)+KK
                 DD1(IADD)=DD1(IADD)+DDZIXJ
                 IADD=IUPT(II)+KK+2
                 DD1(IADD)=DD1(IADD)+DDXIZJ
                 IADD=IUPT(II+2)+KK+1
                 DD1(IADD)=DD1(IADD)+DDZIYJ
                 IADD=IUPT(II+1)+KK+2
                 DD1(IADD)=DD1(IADD)+DDYIZJ
              ENDIF
              !
              IF (IJTEST) THEN
                 IADD=IUPT(JJ)+II
                 DD1(IADD)=DD1(IADD)-DDXIXJ-DDXIXI
                 IADD=IUPT(JJ+1)+II+1
                 DD1(IADD)=DD1(IADD)-DDYIYJ-DDYIYI
                 IADD=IUPT(JJ+2)+II+2
                 DD1(IADD)=DD1(IADD)-DDZIZJ-DDZIZI
                 IADD=IUPT(JJ)+II+1
                 DD1(IADD)=DD1(IADD)-DDXIYI-DDYIXJ
                 IADD=IUPT(JJ+1)+II
                 DD1(IADD)=DD1(IADD)-DDXIYI-DDXIYJ
                 IADD=IUPT(JJ)+II+2
                 DD1(IADD)=DD1(IADD)-DDXIZI-DDZIXJ
                 IADD=IUPT(JJ+2)+II
                 DD1(IADD)=DD1(IADD)-DDXIZI-DDXIZJ
                 IADD=IUPT(JJ+1)+II+2
                 DD1(IADD)=DD1(IADD)-DDYIZI-DDZIYJ
                 IADD=IUPT(JJ+2)+II+1
                 DD1(IADD)=DD1(IADD)-DDYIZI-DDYIZJ
              ELSE
                 IADD=IUPT(II)+JJ
                 DD1(IADD)=DD1(IADD)-DDXIXJ-DDXIXI
                 IADD=IUPT(II+1)+JJ+1
                 DD1(IADD)=DD1(IADD)-DDYIYJ-DDYIYI
                 IADD=IUPT(II+2)+JJ+2
                 DD1(IADD)=DD1(IADD)-DDZIZJ-DDZIZI
                 IADD=IUPT(II+1)+JJ
                 DD1(IADD)=DD1(IADD)-DDXIYI-DDYIXJ
                 IADD=IUPT(II)+JJ+1
                 DD1(IADD)=DD1(IADD)-DDXIYI-DDXIYJ
                 IADD=IUPT(II+2)+JJ
                 DD1(IADD)=DD1(IADD)-DDXIZI-DDZIXJ
                 IADD=IUPT(II)+JJ+2
                 DD1(IADD)=DD1(IADD)-DDXIZI-DDXIZJ
                 IADD=IUPT(II+2)+JJ+1
                 DD1(IADD)=DD1(IADD)-DDYIZI-DDZIYJ
                 IADD=IUPT(II+1)+JJ+2
                 DD1(IADD)=DD1(IADD)-DDYIZI-DDYIZJ
              ENDIF
              IF (JKTEST) THEN
                 IADD=IUPT(KK)+JJ
                 DD1(IADD)=DD1(IADD)-DDXIXJ-DDXJXJ
                 IADD=IUPT(KK+1)+JJ+1
                 DD1(IADD)=DD1(IADD)-DDYIYJ-DDYJYJ
                 IADD=IUPT(KK+2)+JJ+2
                 DD1(IADD)=DD1(IADD)-DDZIZJ-DDZJZJ
                 IADD=IUPT(KK)+JJ+1
                 DD1(IADD)=DD1(IADD)-DDXJYJ-DDYIXJ
                 IADD=IUPT(KK+1)+JJ
                 DD1(IADD)=DD1(IADD)-DDXJYJ-DDXIYJ
                 IADD=IUPT(KK)+JJ+2
                 DD1(IADD)=DD1(IADD)-DDXJZJ-DDZIXJ
                 IADD=IUPT(KK+2)+JJ
                 DD1(IADD)=DD1(IADD)-DDXJZJ-DDXIZJ
                 IADD=IUPT(KK+1)+JJ+2
                 DD1(IADD)=DD1(IADD)-DDYJZJ-DDZIYJ
                 IADD=IUPT(KK+2)+JJ+1
                 DD1(IADD)=DD1(IADD)-DDYJZJ-DDYIZJ
              ELSE
                 IADD=IUPT(JJ)+KK
                 DD1(IADD)=DD1(IADD)-DDXIXJ-DDXJXJ
                 IADD=IUPT(JJ+1)+KK+1
                 DD1(IADD)=DD1(IADD)-DDYIYJ-DDYJYJ
                 IADD=IUPT(JJ+2)+KK+2
                 DD1(IADD)=DD1(IADD)-DDZIZJ-DDZJZJ
                 IADD=IUPT(JJ+1)+KK
                 DD1(IADD)=DD1(IADD)-DDXJYJ-DDYIXJ
                 IADD=IUPT(JJ)+KK+1
                 DD1(IADD)=DD1(IADD)-DDXJYJ-DDXIYJ
                 IADD=IUPT(JJ+2)+KK
                 DD1(IADD)=DD1(IADD)-DDXJZJ-DDZIXJ
                 IADD=IUPT(JJ)+KK+2
                 DD1(IADD)=DD1(IADD)-DDXJZJ-DDXIZJ
                 IADD=IUPT(JJ+2)+KK+1
                 DD1(IADD)=DD1(IADD)-DDYJZJ-DDZIYJ
                 IADD=IUPT(JJ+1)+KK+2
                 DD1(IADD)=DD1(IADD)-DDYJZJ-DDYIZJ
              ENDIF
              !
              !         DIAGONAL TERMS
              IADD=IUPT(II)+II
              DD1(IADD)=DD1(IADD)+DDXIXI
              IADD=IUPT(II+1)+II+1
              DD1(IADD)=DD1(IADD)+DDYIYI
              IADD=IUPT(II+2)+II+2
              DD1(IADD)=DD1(IADD)+DDZIZI
              IADD=IUPT(II)+II+1
              DD1(IADD)=DD1(IADD)+DDXIYI
              IADD=IUPT(II)+II+2
              DD1(IADD)=DD1(IADD)+DDXIZI
              IADD=IUPT(II+1)+II+2
              DD1(IADD)=DD1(IADD)+DDYIZI
              !
              IADD=IUPT(KK)+KK
              DD1(IADD)=DD1(IADD)+DDXJXJ
              IADD=IUPT(KK+1)+KK+1
              DD1(IADD)=DD1(IADD)+DDYJYJ
              IADD=IUPT(KK+2)+KK+2
              DD1(IADD)=DD1(IADD)+DDZJZJ
              IADD=IUPT(KK)+KK+1
              DD1(IADD)=DD1(IADD)+DDXJYJ
              IADD=IUPT(KK)+KK+2
              DD1(IADD)=DD1(IADD)+DDXJZJ
              IADD=IUPT(KK+1)+KK+2
              DD1(IADD)=DD1(IADD)+DDYJZJ
              !
              IADD=IUPT(JJ)+JJ
              DD1(IADD)=DD1(IADD)+DDXIXI+DDXJXJ+DDXIXJ+DDXIXJ
              IADD=IUPT(JJ+1)+JJ+1
              DD1(IADD)=DD1(IADD)+DDYIYI+DDYJYJ+DDYIYJ+DDYIYJ
              IADD=IUPT(JJ+2)+JJ+2
              DD1(IADD)=DD1(IADD)+DDZIZI+DDZJZJ+DDZIZJ+DDZIZJ
              IADD=IUPT(JJ)+JJ+1
              DD1(IADD)=DD1(IADD)+DDXIYI+DDXJYJ+DDXIYJ+DDYIXJ
              IADD=IUPT(JJ)+JJ+2
              DD1(IADD)=DD1(IADD)+DDXIZI+DDXJZJ+DDXIZJ+DDZIXJ
              IADD=IUPT(JJ+1)+JJ+2
              DD1(IADD)=DD1(IADD)+DDYIZI+DDYJZJ+DDYIZJ+DDZIYJ
#if KEY_DIMB==1
           ENDIF  
#endif

        ENDIF

#if KEY_BLOCK==1
     ENDIF
#endif /*  BLOCK*/
     !
20   CONTINUE
  ENDDO
  !
  IF(NWARN.GT.5 .AND. WRNLEV.GE.2) WRITE(OUTU,30) NWARN
30 FORMAT(' TOTAL OF',I6,' WARNINGS FROM EANGLE')
  !
  RETURN
END SUBROUTINE EANGLE

subroutine ephi(ep,ip,jp,kp,lp,icp,nphi,cpc,cpd,cpb,cpcos, &
     cpsin,dx,dy,dz,x,y,z,qcpw,cpw, &
     qecontx,econtx,iconhp,iskp,dd1,iupt,qsecd &
     )

  !-----------------------------------------------------------------------
  !     CALCULATES EITHER TORSION ANGLES OR IMPROPER TORSION
  !     ANGLES AND THEIR ENERGIES. FIRST DERIVATIVES ARE ADDED
  !     TO DX, DY, DZ AND IF QSECD SECOND DERIVATIVES TO DD1.
  !
  !     ENERGY TERMS ARE EXPRESSED AS A FUNCTION OF PHI TO AVOID
  !     ALL PROBLEMS AS DIHEDRALS BECOME PLANAR.
  !     THE FUNCTIONAL FORMS ARE:
  !
  !     E = K*(1+COS(n*Phi-Phi0))
  !     Where:
  !     n IS A POSITIVE INTEGER COS(n*Phi) AND SIN(n*Phi)
  !            ARE CALCULATED BY RECURRENCE TO AVOID LIMITATION ON n
  !     K IS THE FORCE CONSTANT IN kcal/mol
  !     Phi0/n IS A MAXIMUM IN ENERGY.
  !
  !     FOR IMPROPER DIHEDRALS, THE ENERGY TERM IS GIVEN BY:
  !     E = K*(Phi-phi0)**2
  !     WHERE
  !     K IS THE FORCE CONSTANT IN kcal/mol/rad^2
  !     Phi0 IS THE MINIMUM IN ENERGY.
  !
  !     If QCPW is specified, then the improper becomes
  !     a flat bottom with quadratic walls
  !     E= K*( MAX(0, ABS(Phi-phi0)-cpw )**2
  !
  !     THE CONSTRAINTS ON DIHEDRALS CAN USE EITHER FORM.
  !     FOR THE COSINE FORM, Phi0 IS PhiMin*n+PI. THIS DONE AT THE
  !     PARSER LEVEL SO THAT THE GIVEN PhiMin BECOMES A MINIMUM.
  !
  !     The parameters of the routine are:
  !
  !     EP          <- Dihedral Energy
  !     IP,JP,KP,LP(phi) -> atom number for the members of dihedrals
  !     ICP(phi)    -> parameter number associated with dihedral (type)
  !     NPHI        ->  Number of dihedrals
  !     CPC(type)   -> K value as explained above
  !     CPD(type)   -> if 0: flags Improper form. if CPD<0 flag for multiple
  !                    dihedral n=-CPD. if CPD>0 last term n=CPD.
  !     CPB(type)   -> Phi0 value as explained above (warnings) (radians)
  !     CPCOS(type) -> Cos(Phi0)
  !     CPSIN(type) -> Sin(Phi0)
  !     DX,DY,DZ(atom) <-> Force matrices
  !     X,Y,Z(atom) -> Coordinate matrices
  !     QCPW        -> Flag indicating flat bottom impropers are in use
  !     CPW(type)   -> half width of flat bottom potential (degrees)
  !     QECONT      -> Flag for energy/atom statistics
  !     ECONT(atom) <- matrix of energy/atom
  !     ICONHP      -> Flag to activate the skipping procedure
  !     ISKP(phi)   -> matrix of flag to skip dihedral. Skip if ISKIP.ne.0
  !     DD1        <-> Second derivative matrix (upper triangle)
  !     IUPT(atom)  -> Index function for DD1
  !     QSECD       -> Second derivative flag.
  !
  !
  !     By Bernard R. Brooks    1981
  !
  !     New formulation and comments by:
  !
  !           Arnaud Blondel    1994
  !
  !     See details in: AB&MK J.Comp.Chem (1996), 17, 1132-1141.
  !
  use chm_kinds
  use dimens_fcm
  use number
#if KEY_SCCDFTB==1
  use blockscc_fcm  
#endif
#if KEY_PARALLEL==1
  use parallel  
#endif
  use block_fcm
  use lambdam
  use pert  !Cc New PBLOCK
  use dimb
  use consta
  use econtmod
  use stream
  use galgor_ltm
  use chutil,only:atomid
  !
  implicit none
  !
#if KEY_PARALLEL==1
  LOGICAL NOPARS  
#endif
  real(chm_real) EP
  INTEGER IP(*),JP(*),KP(*),LP(*),ICP(*)
  INTEGER NPHI
  real(chm_real) CPC(*),CPB(*),CPCOS(*),CPSIN(*)
  INTEGER CPD(*)
  real(chm_real) DX(*),DY(*),DZ(*),X(*),Y(*),Z(*)
  LOGICAL QCPW
  real(chm_real) CPW(*)
  LOGICAL QECONTX
  real(chm_real) ECONTX(*)
  INTEGER ICONHP
  INTEGER ISKP(*)
  real(chm_real) DD1(*)
  INTEGER IUPT(*)
  LOGICAL QSECD
  !
  real(chm_real) CPBIC,E1,DF1,DDF1,FX,FY,FZ,GX,GY,GZ,HX,HY,HZ
  real(chm_real) AX,AY,AZ,BX,BY,BZ,RA2,RB2,RA2R,RB2R,RG2,RG,RGR,RGR2
  real(chm_real) RABR,CP,AP,SP,E,DF,DDF,CA,SA,ARG,APR
  real(chm_real) GAA,GBB,FG,HG,FGA,HGB,FGRG2,HGRG2,DFRG3
  real(chm_real) DFX,DFY,DFZ,DHX,DHY,DHZ,DGX,DGY,DGZ
  real(chm_real) DTFX,DTFY,DTFZ,DTHX,DTHY,DTHZ,DTGX,DTGY,DTGZ
  real(chm_real) GAFX,GAFY,GAFZ,GBHX,GBHY,GBHZ
  real(chm_real) FAGX,FAGY,FAGZ,HBGX,HBGY,HBGZ
  real(chm_real) DDFGH(45)
  INTEGER NWARN,NWARNX,IPHI,I,J,K,L,IC,IPER,NPER
  INTEGER II,JJ,KK,LL,IADD
  LOGICAL IJTEST,IKTEST,ILTEST,JKTEST,JLTEST,KLTEST
  LOGICAL LREP,NOCONS,QAFIRST
  CHARACTER(len=8) SIDDNI,RIDDNI,RESDNI,ACDNI,SIDDNJ,RIDDNJ,RESDNJ,ACDNJ
  CHARACTER(len=8) SIDDNK,RIDDNK,RESDNK,ACDNK,SIDDNL,RIDDNL,RESDNL,ACDNL
  !
#if KEY_BLOCK==1
  INTEGER IBL, JBL, KKK, LLL, KDOC
  real(chm_real)  COEF, DOCFI, DOCFJ, DOCFK, DOCFJ1, DOCFK1, DOCFL
!ldm
  real(chm_real) UNSCALE, FALPHA
  real(chm_real) DFORG
!ldm
#endif /*  BLOCK*/
  !
  real(chm_real), parameter :: RXMIN=0.005D0, RXMIN2=0.000025D0
  !
#if KEY_GENETIC==1
  INTEGER First
  First = 1
  If(qGA_Ener) First = Int(EP)
#endif 
  EP=ZERO

#if KEY_SCCDFTB==1
#if KEY_PARALLEL==1
  IF (MYNOD.EQ.0) THEN
#endif 

     if(qsccb) then
        if(idxphi.eq.0) dvdlp=zero
        if(idxphi.eq.1) dvdlip=zero
        if(idxphi.eq.2) dvdlcp=zero
     endif
#if KEY_PARALLEL==1
  ENDIF
#endif 

#endif 

  NOCONS=(ICONHP.GT.0)
  IF(NPHI.LE.0) RETURN
  NWARN=0
  NWARNX=0
  QAFIRST=.TRUE.
  !
#if KEY_PARALLEL==1 /*paraphi*/
#if KEY_VIBPARA==1
  DO IPHI=1,NPHI
#else /**/

#if KEY_PARAFULL==1 /*parfphi*/
  DO IPHI=MYNODP,NPHI,NUMNOD
#elif KEY_PARASCAL==1 /*parfphi*/
  NOPARS=(ICONHP.GE.0)
#if KEY_GENETIC==1
  DO IPHI=First,NPHI
#else /**/
  DO IPHI=1,NPHI
#endif 
     IF(NOPARS) THEN
        II=JPBLOCK(IP(IPHI))
        JJ=JPBLOCK(JP(IPHI))
        KK=JPBLOCK(KP(IPHI))
        LL=JPBLOCK(LP(IPHI))
        IA=JJ
        IB=KK
        IF(IA.EQ.IB) IB=LL
        IF(IA.EQ.IB) IB=II
        IF(II.NE.IA .AND. II.NE.IB) THEN
           CALL PSADDTOCL(IP(IPHI),JPMAT(IA,IB))
        ENDIF
        IF(LL.NE.IA .AND. LL.NE.IB) THEN
           CALL PSADDTOCL(LP(IPHI),JPMAT(IA,IB))
        ENDIF
        IF(JPMAT(IA,IB).NE.MYNOD) GOTO 160
     ENDIF
#elif KEY_SPACDEC==1 /*parfphi*/
  DO IPHI=1,NPHI
     IF(MYNOD /= ICPUMAP(IP(IPHI))) GOTO 160
#endif /* (parfphi)*/
#endif 
#else /* (paraphi)*/
#if KEY_GENETIC==1
  DO IPHI=First,NPHI
#else /**/
  DO IPHI=1,NPHI
#endif 
#endif /* (paraphi)*/
     !
     I=IP(IPHI)
     IF(NOCONS) THEN
        IF(ISKP(IPHI).NE.0) GOTO 160
     ENDIF
     J=JP(IPHI)
     K=KP(IPHI)
     L=LP(IPHI)
     IC=ICP(IPHI)
     IF(IC.EQ.0) GOTO 160
     ! F=Ri-Rj, G=Rj-Rk, H-Rl-Rk.
     FX=X(I)-X(J)
     FY=Y(I)-Y(J)
     FZ=Z(I)-Z(J)
     GX=X(J)-X(K)
     GY=Y(J)-Y(K)
     GZ=Z(J)-Z(K)
     HX=X(L)-X(K)
     HY=Y(L)-Y(K)
     HZ=Z(L)-Z(K)
     ! A=F^G, B=H^G
     AX=FY*GZ-FZ*GY
     AY=FZ*GX-FX*GZ
     AZ=FX*GY-FY*GX
     BX=HY*GZ-HZ*GY
     BY=HZ*GX-HX*GZ
     BZ=HX*GY-HY*GX
     ! RG=|G|, RGR=1/|G|
     RA2=AX*AX+AY*AY+AZ*AZ
     RB2=BX*BX+BY*BY+BZ*BZ
     RG2=GX*GX+GY*GY+GZ*GZ
     RG=SQRT(RG2)
     ! Warnings have been simplified.
     IF((RA2.LE.RXMIN2).OR.(RB2.LE.RXMIN2).OR.(RG.LE.RXMIN)) THEN
        NWARN=NWARN+1
        IF((NWARN.LE.5 .AND. WRNLEV.GE.5) .OR. WRNLEV.GE.6) THEN
           WRITE(OUTU,20) IPHI,I,J,K,L
20         FORMAT(' EPHI: WARNING.  dihedral',I5,' is almost linear.'/ &
                ' derivatives may be affected for atoms:',4I5)
        ENDIF
        GOTO 160
     ENDIF
     !
     RGR=ONE/RG
     RA2R=ONE/RA2
     RB2R=ONE/RB2
     RABR=SQRT(RA2R*RB2R)
     ! CP=cos(phi)
     CP=(AX*BX+AY*BY+AZ*BZ)*RABR
     ! SP=sin(phi), Note that sin(phi).G/|G|=B^A/(|A|.|B|)
     ! which can be simplify to sin(phi)=|G|H.A/(|A|.|B|)
     SP=RG*RABR*(AX*HX+AY*HY+AZ*HZ)
     !
     ! Energy and derivative contributions.
     !
     IF (CPD(IC).NE.0) THEN
        !
        ! Set up for the proper dihedrals.
        !
        E=ZERO
        DF=ZERO
        DDF=ZERO
30      CONTINUE
        IPER=CPD(IC)
        IF (IPER.GE.0) THEN
           LREP=.FALSE.
        ELSE
           LREP=.TRUE.
           IPER=-IPER
        ENDIF
        !
        E1=ONE
        DF1=ZERO
        !alculation of cos(n*phi-phi0) and sin(n*phi-phi0).
        DO NPER=1,IPER
           DDF1=E1*CP-DF1*SP
           DF1=E1*SP+DF1*CP
           E1=DDF1
        enddo
           E1=E1*CPCOS(IC)+DF1*CPSIN(IC)
           DF1=DF1*CPCOS(IC)-DDF1*CPSIN(IC)
           DF1=-IPER*DF1
           DDF1=-IPER*IPER*E1
           E1=ONE+E1
           !
           IF (IPER.EQ.0) THEN
              E1=ONE
              DF1=ZERO
              DDF1=ZERO
           ENDIF
           !
           ARG=CPC(IC)
           E=E+ARG*E1
           DF=DF+ARG*DF1
           DDF=DDF+ARG*DDF1
           !
           IF(LREP) THEN
              IC=IC+1
              GOTO 30
           ENDIF
           !
           ! Set up for the improper dihedrals.
           !
        ELSE
           !
           !alcul of cos(phi-phi0),sin(phi-phi0) and (Phi-Phi0).
           CA=CP*CPCOS(IC)+SP*CPSIN(IC)
           SA=SP*CPCOS(IC)-CP*CPSIN(IC)
           IF (CA.GT.PTONE ) THEN
              AP=ASIN(SA)
           ELSE
              AP=SIGN(ACOS(MAX(CA,MINONE)),SA)
              ! Warning is now triggered at deltaphi=84.26...deg (used to be 90).
              NWARNX=NWARNX+1
              IF((NWARNX.LE.5 .AND. WRNLEV.GE.5) .OR. WRNLEV.GE.6) THEN
                 APR=AP*RADDEG
                 CPBIC=CPB(IC)*RADDEG
                 WRITE(OUTU,80) IPHI,APR,CPBIC,I,J,K,L
80               FORMAT(' EPHI: WARNING. bent improper torsion angle', &
                      ' is '//'far ','from minimum for;'/3X,' IPHI=',I5, &
                      '  with deltaPHI=',F9.4,' MIN=',F9.4,' ATOMS:',4I5)
              ENDIF
           ENDIF
           !
           IF(QCPW) THEN
              APR=CPW(IC)*DEGRAD
              IF(ABS(AP).GT.APR) THEN
                 IF(AP.GT.ZERO) THEN
                    AP=AP-APR
                 ELSE
                    AP=AP+APR
                 ENDIF
                 DDF=TWO*CPC(IC)
                 DF=DDF*AP
                 E=HALF*DF*AP
              ELSE
                 DDF=ZERO
                 DF=ZERO
                 E=ZERO
              ENDIF
           ELSE
              DDF=TWO*CPC(IC)
              DF=DDF*AP
              E=HALF*DF*AP
           ENDIF
           !
        ENDIF
        !
#if KEY_BLOCK==1 /*big_block*/
        qblocktest: IF (QBLOCK) THEN
           IBL = IBLCKP(I)
           JBL = IBLCKP(J)
           KKK = IBLCKP(K)
           LLL = IBLCKP(L)
#if KEY_DOCK==1
           !         three pairs (i,j), (k,j) and (k,l)
           DOCFI = 1.0
           DOCFJ = 1.0
           DOCFK = 1.0
           DOCFJ1 = 1.0
           DOCFL = 1.0
           DOCFK1 = 1.0
           IF(QDOCK) THEN
              KDOC  = (IBL - 1)*NBLOCK + JBL
              DOCFI = BLDOCP(KDOC)
              KDOC  = (JBL - 1)*NBLOCK + IBL
              DOCFJ = BLDOCP(KDOC)
              KDOC  = (KKK - 1)*NBLOCK + JBL
              DOCFK = BLDOCP(KDOC)
              KDOC  = (JBL - 1)*NBLOCK + KKK
              DOCFJ1 = BLDOCP(KDOC)
              KDOC  = (KKK - 1)*NBLOCK + LLL
              DOCFK1 = BLDOCP(KDOC)
              KDOC  = (LLL - 1)*NBLOCK + KKK
              DOCFL = BLDOCP(KDOC)
           ENDIF
#endif /*  DOCK*/
           IF (IBL .EQ. JBL) JBL=KKK
           IF (IBL .EQ. JBL) JBL=LLL
           IF (JBL .LT. IBL) THEN
              KKK=JBL
              JBL=IBL
              IBL=KKK
           ENDIF
           KKK=IBL+JBL*(JBL-1)/2
           COEF = BLCOED(KKK)

           IF (QPRNTV) THEN
              IF (IBL == 1 .OR. JBL == 1 .OR. IBL == JBL) THEN
                 IF (QNOIM .AND. (CPD(IC) == 0)) THEN
                    VBIMPR(JBL) = VBIMPR(JBL) + E
                 ELSE IF (QNOPH .AND. (CPD(IC) /= 0)) THEN
                    VBTORS(JBL) = VBTORS(JBL) + E
                 ENDIF
              ENDIF
           ENDIF
#if KEY_BLOCK==1 /*ldm*/
           IF (QLDM .or. QLMC) THEN
              ! first row or diagonal elements exclude (1,1).
              UNSCALE = 0.0
              IF ((IBL == 1 .AND. JBL >= LSTRT) .or. &
                   (IBL >= LSTRT .AND. IBL == JBL)) UNSCALE = E
           ENDIF
#endif /*     LDM*/
           IF ( QNOIM .AND. (CPD(IC).EQ.0) ) COEF=1.0
           IF ( QNOPH .AND. (CPD(IC).NE.0) ) COEF=1.0
#if KEY_DOCK==1
           IF(QDOCK) THEN
              E=E*COEF*(DOCFI+DOCFJ+DOCFK+DOCFJ1+ &
                   DOCFK1+DOCFL)/6.0
           ELSE
#endif 

#if KEY_SCCDFTB==1
#if KEY_PARALLEL==1
              IF (MYNOD.EQ.0) THEN
#endif 

                 if(qsccb) then
                    dvdl0=e*cdvdl(kkk)
                 endif
#if KEY_PARALLEL==1
              ENDIF
#endif 

#endif 

              E=E*COEF
#if KEY_DOCK==1
           ENDIF
#endif 
#if KEY_BLOCK==1
           DFORG=DF        /*ldm*/
#endif
           DF=DF*COEF
           DDF=DDF*COEF
        ENDIF qblocktest
        ! Set the energy.
#if KEY_BLOCK==1 /*ldm_5*/ /*ldm*/
        IF (QLDM .or. QLMC) THEN
           IF ((.NOT. QNOIM .AND. CPD(IC) == 0) .OR. &
                (.NOT. QNOPH .AND. CPD(IC) /= 0)) THEN
              IF ((IBL == 1 .AND. JBL >= LSTRT) .or. &
                   (IBL >= LSTRT .AND. IBL == JBL)) THEN
                 FALPHA = UNSCALE
#if KEY_PERT==1
                 if(QPERT) FALPHA = FALPHA*PERTLAM  !Cc New PBLOCK 
#endif
                 LAGMUL = LAGMUL + FALPHA
                 BIFLAM(JBL) = BIFLAM(JBL) + FALPHA
                 IF (NRST == 2) THEN
                    BFRST(JBL) = BFRST(JBL) + FALPHA
                 ENDIF
              ENDIF
           ENDIF
        ENDIF
#endif /*  (ldm_5)*/
#endif /* (big_block)*/
        EP=EP+E

#if KEY_SCCDFTB==1
#if KEY_PARALLEL==1
        IF (MYNOD.EQ.0) THEN
#endif 

           if(qsccb) then
              if(idxphi.eq.0) dvdlp=dvdlp+dvdl0
              if(idxphi.eq.1) dvdlip=dvdlip+dvdl0
              if(idxphi.eq.2) dvdlcp=dvdlcp+dvdl0
           endif
#if KEY_PARALLEL==1
        ENDIF
#endif 

#endif 

        !
        !brb...19-Jul-94 New ANAL terms
        qatermtest: IF(QATERM) THEN
           KK=ANSLCT(I)+ANSLCT(J)+ANSLCT(K)+ANSLCT(L)
           IF(KK.EQ.4 .OR. (KK.GE.1 .AND. .NOT.QAONLY)) THEN
              IF(QAUNIT.LT.0) THEN
                 II=OUTU
              ELSE
                 II=QAUNIT
              ENDIF
              !
              IF (CP.GT.PTONE ) THEN
                 AP=ASIN(SP)
              ELSE
                 AP=SIGN(ACOS(MAX(CP,MINONE)),SP)
              ENDIF
              !
              IF(PRNLEV.GE.5) THEN
                 IF(QAFIRST) THEN
                    IF(QLONGL) THEN
                       WRITE(II,243)
                    ELSE
                       WRITE(II,244)
                    ENDIF
243                 FORMAT('ANAL: DIHE: Index        Atom-I              ', &
                         '     Atom-J                   Atom-K         ', &
                         '          Atom-L          ', &
                         '        Dihedral       Energy   ', &
                         '      Force           Parameters')
244                 FORMAT('ANAL: DIHE: Index        Atom-I              ', &
                         '     Atom-J',/ &
                         '                         Atom-K         ', &
                         '          Atom-L          ',/ &
                         '        Dihedral       Energy   ', &
                         '      Force           Parameters')
                    QAFIRST=.FALSE.
                 ENDIF
                 CALL ATOMID(I,SIDDNI,RIDDNI,RESDNI,ACDNI)
                 CALL ATOMID(J,SIDDNJ,RIDDNJ,RESDNJ,ACDNJ)
                 CALL ATOMID(K,SIDDNK,RIDDNK,RESDNK,ACDNK)
                 CALL ATOMID(L,SIDDNL,RIDDNL,RESDNL,ACDNL)
                 IF(QLONGL) THEN
                    WRITE(II,245) IPHI,I,SIDDNI(1:idleng),RIDDNI(1:idleng), &
                         RESDNI(1:idleng),ACDNI(1:idleng), &
                         J,SIDDNJ(1:idleng),RIDDNJ(1:idleng), &
                         RESDNJ(1:idleng),ACDNJ(1:idleng), &
                         K,SIDDNK(1:idleng),RIDDNK(1:idleng), &
                         RESDNK(1:idleng),ACDNK(1:idleng), &
                         L,SIDDNL(1:idleng),RIDDNL(1:idleng), &
                         RESDNL(1:idleng),ACDNL(1:idleng), &
                         AP*RADDEG,E,DF,IC,CPB(IC)*RADDEG, &
                         CPC(IC),CPD(IC)
                 ELSE
                    WRITE(II,246) IPHI,I,SIDDNI(1:idleng),RIDDNI(1:idleng), &
                         RESDNI(1:idleng),ACDNI(1:idleng), &
                         J,SIDDNJ(1:idleng),RIDDNJ(1:idleng), &
                         RESDNJ(1:idleng),ACDNJ(1:idleng), &
                         K,SIDDNK(1:idleng),RIDDNK(1:idleng), &
                         RESDNK(1:idleng),ACDNK(1:idleng), &
                         L,SIDDNL(1:idleng),RIDDNL(1:idleng), &
                         RESDNL(1:idleng),ACDNL(1:idleng), &
                         AP*RADDEG,E,DF,IC,CPB(IC)*RADDEG, &
                         CPC(IC),CPD(IC)
                 ENDIF
245              FORMAT('ANAL: DIHE>',2I5,4(1X,A),I5,4(1X,A),I5,4(1X,A), &
                      I5,4(1X,A),3F15.6,I5,2F15.6,I5)
246              FORMAT('ANAL: DIHE>',2I5,4(1X,A),I5,4(1X,A),/ &
                      I5,4(1X,A),I5,4(1X,A),/ &
                      3F15.6,I5,2F15.6,I5)
              ENDIF
           ENDIF
        ENDIF qatermtest
        !
        ! Contribution on atoms.
        IF(QECONTX) THEN
           E=E*PT25
           ECONTX(I)=ECONTX(I)+E
           ECONTX(J)=ECONTX(J)+E
           ECONTX(K)=ECONTX(K)+E
           ECONTX(L)=ECONTX(L)+E
        ENDIF
        !
        ! Compute derivatives wrt catesian coordinates.
        ! this section is for first derivatives only.
        !
#if KEY_BLOCK==1
        IF (.NOT. NOFORC) THEN
#endif /*  BLOCK*/
           ! GAA=-|G|/A^2, GBB=|G|/B^2, FG=F.G, HG=H.G
           !  FGA=F.G/(|G|A^2), HGB=H.G/(|G|B^2)
           FG=FX*GX+FY*GY+FZ*GZ
           HG=HX*GX+HY*GY+HZ*GZ
           FGA=FG*RA2R*RGR
           HGB=HG*RB2R*RGR
           GAA=-RA2R*RG
           GBB=RB2R*RG
           ! DTFi=d(phi)/dFi, DTGi=d(phi)/dGi, DTHi=d(phi)/dHi. (used in 2nd deriv)
           DTFX=GAA*AX
           DTFY=GAA*AY
           DTFZ=GAA*AZ
           DTGX=FGA*AX-HGB*BX
           DTGY=FGA*AY-HGB*BY
           DTGZ=FGA*AZ-HGB*BZ
           DTHX=GBB*BX
           DTHY=GBB*BY
           DTHZ=GBB*BZ
           ! DFi=dE/dFi, DGi=dE/dGi, DHi=dE/dHi.
           DFX=DF*DTFX
           DFY=DF*DTFY
           DFZ=DF*DTFZ
           DGX=DF*DTGX
           DGY=DF*DTGY
           DGZ=DF*DTGZ
           DHX=DF*DTHX
           DHY=DF*DTHY
           DHZ=DF*DTHZ
           ! Distribute over Ri.
#if KEY_BLOCK==1
#if KEY_DOCK==1
           IF(QDOCK) THEN
              DX(I)=DX(I)+DFX*DOCFI
              DY(I)=DY(I)+DFY*DOCFI
              DZ(I)=DZ(I)+DFZ*DOCFI
              DX(J)=DX(J)-DFX*DOCFJ+DGX*DOCFJ1
              DY(J)=DY(J)-DFY*DOCFJ+DGY*DOCFJ1
              DZ(J)=DZ(J)-DFZ*DOCFJ+DGZ*DOCFJ1
              DX(K)=DX(K)-DHX*DOCFK1-DGX*DOCFK
              DY(K)=DY(K)-DHY*DOCFK1-DGY*DOCFK
              DZ(K)=DZ(K)-DHZ*DOCFK1-DGZ*DOCFK
              DX(L)=DX(L)+DHX*DOCFL
              DY(L)=DY(L)+DHY*DOCFL
              DZ(L)=DZ(L)+DHZ*DOCFL
           ELSE
#endif 
#endif 
              DX(I)=DX(I)+DFX
              DY(I)=DY(I)+DFY
              DZ(I)=DZ(I)+DFZ
              DX(J)=DX(J)-DFX+DGX
              DY(J)=DY(J)-DFY+DGY
              DZ(J)=DZ(J)-DFZ+DGZ
              DX(K)=DX(K)-DHX-DGX
              DY(K)=DY(K)-DHY-DGY
              DZ(K)=DZ(K)-DHZ-DGZ
              DX(L)=DX(L)+DHX
              DY(L)=DY(L)+DHY
              DZ(L)=DZ(L)+DHZ
#if KEY_BLOCK==1
#if KEY_DOCK==1
           ENDIF
#endif 
#endif 
#if KEY_BLOCK==1 /*ldm*/
           IF(RSTP)THEN
              IF ((.NOT. QNOIM .AND. CPD(IC) == 0) .OR. &
                   (.NOT. QNOPH .AND. CPD(IC) /= 0)) THEN
                 IF ((IBL == 1 .AND. JBL >= LSTRT) .or. &
                      (IBL >= LSTRT .AND. IBL == JBL)) THEN
                    DFX=DFORG*DTFX
                    DFY=DFORG*DTFY
                    DFZ=DFORG*DTFZ
                    DGX=DFORG*DTGX
                    DGY=DFORG*DTGY
                    DGZ=DFORG*DTGZ
                    DHX=DFORG*DTHX
                    DHY=DFORG*DTHY
                    DHZ=DFORG*DTHZ
                    ENVDX(I) = ENVDX(I) + DFX
                    ENVDY(I) = ENVDY(I) + DFY
                    ENVDZ(I) = ENVDZ(I) + DFZ
                    ENVDX(J) = ENVDX(J) - DFX + DGX
                    ENVDY(J) = ENVDY(J) - DFY + DGY
                    ENVDZ(J) = ENVDZ(J) - DFZ + DGZ
                    ENVDX(K) = ENVDX(K) - DHX - DGX
                    ENVDY(K) = ENVDY(K) - DHY - DGY
                    ENVDZ(K) = ENVDZ(K) - DHZ - DGZ
                    ENVDX(L) = ENVDX(L) + DHX
                    ENVDY(L) = ENVDY(L) + DHY
                    ENVDZ(L) = ENVDZ(L) + DHZ
                 ENDIF
              ENDIF
           ENDIF
#endif /*  LDM*/
           !
#if KEY_IPRESS==1
           IF(QIPRSS) THEN
              PVIR(I)=PVIR(I)+DFX*FX+DFY*FY+DFZ*FZ
              PVIR(J)=PVIR(J)+DFX*FX+DFY*FY+DFZ*FZ+DGX*GX+DGY*GY+DGZ*GZ
              PVIR(K)=PVIR(K)+DGX*GX+DGY*GY+DGZ*GZ+DHX*HX+DHY*HY+DHZ*HZ
              PVIR(L)=PVIR(L)+DHX*HX+DHY*HY+DHZ*HZ
           ENDIF
#endif 
           !
           ! Second derivative part.
           !
           IF (QSECD) THEN
              !
              ! RGR2=1/G.G,FGRG2=(F.G)/(G.G),HGRG2=(H.G)/(G.G),DFRG3=(dE/dPhi)/|G|^3
              RGR2=RGR*RGR
              FGRG2=FG*RGR2
              HGRG2=HG*RGR2
              DFRG3=DF*RGR2*RGR
              ! GAF=-G^A/A.A, GBH=-G^B/B.B, FAG=F^A/A.A, HBG=-H^B/B.B
              GAFX=RA2R*(AY*GZ-AZ*GY)
              GAFY=RA2R*(AZ*GX-AX*GZ)
              GAFZ=RA2R*(AX*GY-AY*GX)
              GBHX=RB2R*(BY*GZ-BZ*GY)
              GBHY=RB2R*(BZ*GX-BX*GZ)
              GBHZ=RB2R*(BX*GY-BY*GX)
              FAGX=RA2R*(FY*AZ-FZ*AY)
              FAGY=RA2R*(FZ*AX-FX*AZ)
              FAGZ=RA2R*(FX*AY-FY*AX)
              HBGX=RB2R*(BY*HZ-BZ*HY)
              HBGY=RB2R*(BZ*HX-BX*HZ)
              HBGZ=RB2R*(BX*HY-BY*HX)
              ! What are the indexes ?
              ! ddE/dX.dY= DDFGH(n)      Fx, Fy, Fz,|Gx, Gy, Gz,|Hx, Hy, Hz. X/
              !                          ----------------------------------- / Y
              !               n=          1   2   4 | 7  11  16 |22  29  37   Fx
              !                               3   5 | 8  12  17 |23  30  38   Fy
              !                                   6 | 9  13  18 |24  31  39   Fz
              !                                     ------------------------
              !                                      10  14  19 |25  32  40   Gx
              !                                          15  20 |26  33  41   Gy
              !                                              21 |27  34  42   Gz
              !                                                 ------------
              !                                                  28  35  43   Hx
              !                                                      36  44   Hy
              !                                                          45   Hz
              !
              ! ddE/dF.dF
              DDFGH(1) =DDF*DTFX*DTFX+TWO*DFX*GAFX
              DDFGH(2) =DDF*DTFX*DTFY+DFX*GAFY+DFY*GAFX
              DDFGH(3) =DDF*DTFY*DTFY+TWO*DFY*GAFY
              DDFGH(4) =DDF*DTFX*DTFZ+DFX*GAFZ+DFZ*GAFX
              DDFGH(5) =DDF*DTFY*DTFZ+DFY*GAFZ+DFZ*GAFY
              DDFGH(6) =DDF*DTFZ*DTFZ+TWO*DFZ*GAFZ
              ! ddE/dF.dG
              DDFGH(7) =DDF*DTFX*DTGX+FAGX*DFX-FGRG2*DFX*GAFX
              DDFGH(8) =DDF*DTFY*DTGX+FAGY*DFX-FGRG2*DFY*GAFX
              DDFGH(9) =DDF*DTFZ*DTGX+FAGZ*DFX-FGRG2*DFZ*GAFX
              DDFGH(11)=DDF*DTFX*DTGY+FAGX*DFY-FGRG2*DFX*GAFY
              DDFGH(12)=DDF*DTFY*DTGY+FAGY*DFY-FGRG2*DFY*GAFY
              DDFGH(13)=DDF*DTFZ*DTGY+FAGZ*DFY-FGRG2*DFZ*GAFY
              DDFGH(16)=DDF*DTFX*DTGZ+FAGX*DFZ-FGRG2*DFX*GAFZ
              DDFGH(17)=DDF*DTFY*DTGZ+FAGY*DFZ-FGRG2*DFY*GAFZ
              DDFGH(18)=DDF*DTFZ*DTGZ+FAGZ*DFZ-FGRG2*DFZ*GAFZ
              ! ddE/dF.dH
              DDFGH(22)=DDF*DTFX*DTHX
              DDFGH(23)=DDF*DTFY*DTHX
              DDFGH(24)=DDF*DTFZ*DTHX
              DDFGH(29)=DDF*DTFX*DTHY
              DDFGH(30)=DDF*DTFY*DTHY
              DDFGH(31)=DDF*DTFZ*DTHY
              DDFGH(37)=DDF*DTFX*DTHZ
              DDFGH(38)=DDF*DTFY*DTHZ
              DDFGH(39)=DDF*DTFZ*DTHZ
              ! ddE/dG.dG
              DDFGH(10)=DDF*DTGX*DTGX-DFRG3*(GAFX*AX-GBHX*BX) &
                   -TWO*FGRG2*DFX*FAGX+TWO*HGRG2*DHX*HBGX
              DDFGH(14)=DDF*DTGX*DTGY &
                   -HALF*DFRG3*(GAFX*AY+GAFY*AX-GBHX*BY-GBHY*BX) &
                   -FGRG2*(DFX*FAGY+DFY*FAGX)+HGRG2*(DHX*HBGY+DHY*HBGX)
              DDFGH(15)=DDF*DTGY*DTGY-DFRG3*(GAFY*AY-GBHY*BY) &
                   -TWO*FGRG2*DFY*FAGY+TWO*HGRG2*DHY*HBGY
              DDFGH(19)=DDF*DTGX*DTGZ &
                   -HALF*DFRG3*(GAFX*AZ+GAFZ*AX-GBHX*BZ-GBHZ*BX) &
                   -FGRG2*(DFX*FAGZ+DFZ*FAGX)+HGRG2*(DHX*HBGZ+DHZ*HBGX)
              DDFGH(20)=DDF*DTGY*DTGZ &
                   -HALF*DFRG3*(GAFY*AZ+GAFZ*AY-GBHY*BZ-GBHZ*BY) &
                   -FGRG2*(DFY*FAGZ+DFZ*FAGY)+HGRG2*(DHY*HBGZ+DHZ*HBGY)
              DDFGH(21)=DDF*DTGZ*DTGZ-DFRG3*(GAFZ*AZ-GBHZ*BZ) &
                   -TWO*FGRG2*DFZ*FAGZ+TWO*HGRG2*DHZ*HBGZ
              ! ddE/dG.dH
              DDFGH(25)=DDF*DTGX*DTHX-DHX*HBGX-HGRG2*GBHX*DHX
              DDFGH(26)=DDF*DTGY*DTHX-DHY*HBGX-HGRG2*GBHY*DHX
              DDFGH(27)=DDF*DTGZ*DTHX-DHZ*HBGX-HGRG2*GBHZ*DHX
              DDFGH(32)=DDF*DTGX*DTHY-DHX*HBGY-HGRG2*GBHX*DHY
              DDFGH(33)=DDF*DTGY*DTHY-DHY*HBGY-HGRG2*GBHY*DHY
              DDFGH(34)=DDF*DTGZ*DTHY-DHZ*HBGY-HGRG2*GBHZ*DHY
              DDFGH(40)=DDF*DTGX*DTHZ-DHX*HBGZ-HGRG2*GBHX*DHZ
              DDFGH(41)=DDF*DTGY*DTHZ-DHY*HBGZ-HGRG2*GBHY*DHZ
              DDFGH(42)=DDF*DTGZ*DTHZ-DHZ*HBGZ-HGRG2*GBHZ*DHZ
              ! ddE/dH.dH
              DDFGH(28)=DDF*DTHX*DTHX+TWO*DHX*GBHX
              DDFGH(35)=DDF*DTHX*DTHY+DHX*GBHY+DHY*GBHX
              DDFGH(36)=DDF*DTHY*DTHY+TWO*DHY*GBHY
              DDFGH(43)=DDF*DTHX*DTHZ+DHX*GBHZ+DHZ*GBHX
              DDFGH(44)=DDF*DTHY*DTHZ+DHY*GBHZ+DHZ*GBHY
              DDFGH(45)=DDF*DTHZ*DTHZ+TWO*DHZ*GBHZ
              !
              ! Now scatter ddE/(dFGH)^2 through (dFGH/dRiRjRkRl)^2
              !
#if KEY_DIMB==1
              qcmpcttest: IF (QCMPCT) THEN
                 CALL EPHCMP(I,J,K,L,DDFGH,DD1,PINBCM,PJNBCM)
              ELSE
#endif /*  DIMB*/

                 II=3*I-2
                 JJ=3*J-2
                 KK=3*K-2
                 LL=3*L-2
                 IJTEST=(J.LT.I)
                 IKTEST=(K.LT.I)
                 JKTEST=(K.LT.J)
                 ILTEST=(L.LT.I)
                 JLTEST=(L.LT.J)
                 KLTEST=(L.LT.K)
                 !
                 IADD=IUPT(II)+II
                 DD1(IADD)=DD1(IADD)+DDFGH(1)
                 IADD=IUPT(II+1)+II+1
                 DD1(IADD)=DD1(IADD)+DDFGH(3)
                 IADD=IUPT(II+2)+II+2
                 DD1(IADD)=DD1(IADD)+DDFGH(6)
                 IADD=IUPT(II)+II+1
                 DD1(IADD)=DD1(IADD)+DDFGH(2)
                 IADD=IUPT(II)+II+2
                 DD1(IADD)=DD1(IADD)+DDFGH(4)
                 IADD=IUPT(II+1)+II+2
                 DD1(IADD)=DD1(IADD)+DDFGH(5)
                 !
                 IADD=IUPT(LL)+LL
                 DD1(IADD)=DD1(IADD)+DDFGH(28)
                 IADD=IUPT(LL+1)+LL+1
                 DD1(IADD)=DD1(IADD)+DDFGH(36)
                 IADD=IUPT(LL+2)+LL+2
                 DD1(IADD)=DD1(IADD)+DDFGH(45)
                 IADD=IUPT(LL)+LL+1
                 DD1(IADD)=DD1(IADD)+DDFGH(35)
                 IADD=IUPT(LL)+LL+2
                 DD1(IADD)=DD1(IADD)+DDFGH(43)
                 IADD=IUPT(LL+1)+LL+2
                 DD1(IADD)=DD1(IADD)+DDFGH(44)
                 !
                 IADD=IUPT(JJ)+JJ
                 DD1(IADD)=DD1(IADD)+DDFGH(1)+DDFGH(10)-DDFGH(7)-DDFGH(7)
                 IADD=IUPT(JJ+1)+JJ+1
                 DD1(IADD)=DD1(IADD)+DDFGH(3)+DDFGH(15)-DDFGH(12)-DDFGH(12)
                 IADD=IUPT(JJ+2)+JJ+2
                 DD1(IADD)=DD1(IADD)+DDFGH(6)+DDFGH(21)-DDFGH(18)-DDFGH(18)
                 IADD=IUPT(JJ)+JJ+1
                 DD1(IADD)=DD1(IADD)+DDFGH(2)+DDFGH(14)-DDFGH(11)-DDFGH(8)
                 IADD=IUPT(JJ)+JJ+2
                 DD1(IADD)=DD1(IADD)+DDFGH(4)+DDFGH(19)-DDFGH(16)-DDFGH(9)
                 IADD=IUPT(JJ+1)+JJ+2
                 DD1(IADD)=DD1(IADD)+DDFGH(5)+DDFGH(20)-DDFGH(17)-DDFGH(13)
                 !
                 IADD=IUPT(KK)+KK
                 DD1(IADD)=DD1(IADD)+DDFGH(28)+DDFGH(10)+DDFGH(25)+DDFGH(25)
                 IADD=IUPT(KK+1)+KK+1
                 DD1(IADD)=DD1(IADD)+DDFGH(36)+DDFGH(15)+DDFGH(33)+DDFGH(33)
                 IADD=IUPT(KK+2)+KK+2
                 DD1(IADD)=DD1(IADD)+DDFGH(45)+DDFGH(21)+DDFGH(42)+DDFGH(42)
                 IADD=IUPT(KK)+KK+1
                 DD1(IADD)=DD1(IADD)+DDFGH(35)+DDFGH(14)+DDFGH(32)+DDFGH(26)
                 IADD=IUPT(KK)+KK+2
                 DD1(IADD)=DD1(IADD)+DDFGH(43)+DDFGH(19)+DDFGH(40)+DDFGH(27)
                 IADD=IUPT(KK+1)+KK+2
                 DD1(IADD)=DD1(IADD)+DDFGH(44)+DDFGH(20)+DDFGH(41)+DDFGH(34)
                 !
                 IF (IJTEST) THEN
                    IADD=IUPT(JJ)+II
                    DD1(IADD)=DD1(IADD)+DDFGH(7)-DDFGH(1)
                    IADD=IUPT(JJ+1)+II+1
                    DD1(IADD)=DD1(IADD)+DDFGH(12)-DDFGH(3)
                    IADD=IUPT(JJ+2)+II+2
                    DD1(IADD)=DD1(IADD)+DDFGH(18)-DDFGH(6)
                    IADD=IUPT(JJ)+II+1
                    DD1(IADD)=DD1(IADD)+DDFGH(8)-DDFGH(2)
                    IADD=IUPT(JJ+1)+II
                    DD1(IADD)=DD1(IADD)+DDFGH(11)-DDFGH(2)
                    IADD=IUPT(JJ)+II+2
                    DD1(IADD)=DD1(IADD)+DDFGH(9)-DDFGH(4)
                    IADD=IUPT(JJ+2)+II
                    DD1(IADD)=DD1(IADD)+DDFGH(16)-DDFGH(4)
                    IADD=IUPT(JJ+1)+II+2
                    DD1(IADD)=DD1(IADD)+DDFGH(13)-DDFGH(5)
                    IADD=IUPT(JJ+2)+II+1
                    DD1(IADD)=DD1(IADD)+DDFGH(17)-DDFGH(5)
                 ELSE
                    IADD=IUPT(II)+JJ
                    DD1(IADD)=DD1(IADD)+DDFGH(7)-DDFGH(1)
                    IADD=IUPT(II+1)+JJ+1
                    DD1(IADD)=DD1(IADD)+DDFGH(12)-DDFGH(3)
                    IADD=IUPT(II+2)+JJ+2
                    DD1(IADD)=DD1(IADD)+DDFGH(18)-DDFGH(6)
                    IADD=IUPT(II+1)+JJ
                    DD1(IADD)=DD1(IADD)+DDFGH(8)-DDFGH(2)
                    IADD=IUPT(II)+JJ+1
                    DD1(IADD)=DD1(IADD)+DDFGH(11)-DDFGH(2)
                    IADD=IUPT(II+2)+JJ
                    DD1(IADD)=DD1(IADD)+DDFGH(9)-DDFGH(4)
                    IADD=IUPT(II)+JJ+2
                    DD1(IADD)=DD1(IADD)+DDFGH(16)-DDFGH(4)
                    IADD=IUPT(II+2)+JJ+1
                    DD1(IADD)=DD1(IADD)+DDFGH(13)-DDFGH(5)
                    IADD=IUPT(II+1)+JJ+2
                    DD1(IADD)=DD1(IADD)+DDFGH(17)-DDFGH(5)
                 ENDIF
                 !
                 IF (IKTEST) THEN
                    IADD=IUPT(KK)+II
                    DD1(IADD)=DD1(IADD)-DDFGH(7)-DDFGH(22)
                    IADD=IUPT(KK+1)+II+1
                    DD1(IADD)=DD1(IADD)-DDFGH(12)-DDFGH(30)
                    IADD=IUPT(KK+2)+II+2
                    DD1(IADD)=DD1(IADD)-DDFGH(18)-DDFGH(39)
                    IADD=IUPT(KK)+II+1
                    DD1(IADD)=DD1(IADD)-DDFGH(8)-DDFGH(23)
                    IADD=IUPT(KK+1)+II
                    DD1(IADD)=DD1(IADD)-DDFGH(11)-DDFGH(29)
                    IADD=IUPT(KK)+II+2
                    DD1(IADD)=DD1(IADD)-DDFGH(9)-DDFGH(24)
                    IADD=IUPT(KK+2)+II
                    DD1(IADD)=DD1(IADD)-DDFGH(16)-DDFGH(37)
                    IADD=IUPT(KK+1)+II+2
                    DD1(IADD)=DD1(IADD)-DDFGH(13)-DDFGH(31)
                    IADD=IUPT(KK+2)+II+1
                    DD1(IADD)=DD1(IADD)-DDFGH(17)-DDFGH(38)
                 ELSE
                    IADD=IUPT(II)+KK
                    DD1(IADD)=DD1(IADD)-DDFGH(7)-DDFGH(22)
                    IADD=IUPT(II+1)+KK+1
                    DD1(IADD)=DD1(IADD)-DDFGH(12)-DDFGH(30)
                    IADD=IUPT(II+2)+KK+2
                    DD1(IADD)=DD1(IADD)-DDFGH(18)-DDFGH(39)
                    IADD=IUPT(II+1)+KK
                    DD1(IADD)=DD1(IADD)-DDFGH(8)-DDFGH(23)
                    IADD=IUPT(II)+KK+1
                    DD1(IADD)=DD1(IADD)-DDFGH(11)-DDFGH(29)
                    IADD=IUPT(II+2)+KK
                    DD1(IADD)=DD1(IADD)-DDFGH(9)-DDFGH(24)
                    IADD=IUPT(II)+KK+2
                    DD1(IADD)=DD1(IADD)-DDFGH(16)-DDFGH(37)
                    IADD=IUPT(II+2)+KK+1
                    DD1(IADD)=DD1(IADD)-DDFGH(13)-DDFGH(31)
                    IADD=IUPT(II+1)+KK+2
                    DD1(IADD)=DD1(IADD)-DDFGH(17)-DDFGH(38)
                 ENDIF
                 !
                 IF (ILTEST) THEN
                    IADD=IUPT(LL)+II
                    DD1(IADD)=DD1(IADD)+DDFGH(22)
                    IADD=IUPT(LL+1)+II+1
                    DD1(IADD)=DD1(IADD)+DDFGH(30)
                    IADD=IUPT(LL+2)+II+2
                    DD1(IADD)=DD1(IADD)+DDFGH(39)
                    IADD=IUPT(LL)+II+1
                    DD1(IADD)=DD1(IADD)+DDFGH(23)
                    IADD=IUPT(LL+1)+II
                    DD1(IADD)=DD1(IADD)+DDFGH(29)
                    IADD=IUPT(LL)+II+2
                    DD1(IADD)=DD1(IADD)+DDFGH(24)
                    IADD=IUPT(LL+2)+II
                    DD1(IADD)=DD1(IADD)+DDFGH(37)
                    IADD=IUPT(LL+1)+II+2
                    DD1(IADD)=DD1(IADD)+DDFGH(31)
                    IADD=IUPT(LL+2)+II+1
                    DD1(IADD)=DD1(IADD)+DDFGH(38)
                 ELSE
                    IADD=IUPT(II)+LL
                    DD1(IADD)=DD1(IADD)+DDFGH(22)
                    IADD=IUPT(II+1)+LL+1
                    DD1(IADD)=DD1(IADD)+DDFGH(30)
                    IADD=IUPT(II+2)+LL+2
                    DD1(IADD)=DD1(IADD)+DDFGH(39)
                    IADD=IUPT(II+1)+LL
                    DD1(IADD)=DD1(IADD)+DDFGH(23)
                    IADD=IUPT(II)+LL+1
                    DD1(IADD)=DD1(IADD)+DDFGH(29)
                    IADD=IUPT(II+2)+LL
                    DD1(IADD)=DD1(IADD)+DDFGH(24)
                    IADD=IUPT(II)+LL+2
                    DD1(IADD)=DD1(IADD)+DDFGH(37)
                    IADD=IUPT(II+2)+LL+1
                    DD1(IADD)=DD1(IADD)+DDFGH(31)
                    IADD=IUPT(II+1)+LL+2
                    DD1(IADD)=DD1(IADD)+DDFGH(38)
                 ENDIF
                 !
                 IF (JKTEST) THEN
                    IADD=IUPT(KK)+JJ
                    DD1(IADD)=DD1(IADD)+DDFGH(7)+DDFGH(22)-DDFGH(10)-DDFGH(25)
                    IADD=IUPT(KK+1)+JJ+1
                    DD1(IADD)=DD1(IADD)+DDFGH(12)+DDFGH(30)-DDFGH(15)-DDFGH(33)
                    IADD=IUPT(KK+2)+JJ+2
                    DD1(IADD)=DD1(IADD)+DDFGH(18)+DDFGH(39)-DDFGH(21)-DDFGH(42)
                    IADD=IUPT(KK)+JJ+1
                    DD1(IADD)=DD1(IADD)+DDFGH(8)+DDFGH(23)-DDFGH(14)-DDFGH(26)
                    IADD=IUPT(KK+1)+JJ
                    DD1(IADD)=DD1(IADD)+DDFGH(11)+DDFGH(29)-DDFGH(14)-DDFGH(32)
                    IADD=IUPT(KK)+JJ+2
                    DD1(IADD)=DD1(IADD)+DDFGH(9)+DDFGH(24)-DDFGH(19)-DDFGH(27)
                    IADD=IUPT(KK+2)+JJ
                    DD1(IADD)=DD1(IADD)+DDFGH(16)+DDFGH(37)-DDFGH(19)-DDFGH(40)
                    IADD=IUPT(KK+1)+JJ+2
                    DD1(IADD)=DD1(IADD)+DDFGH(13)+DDFGH(31)-DDFGH(20)-DDFGH(34)
                    IADD=IUPT(KK+2)+JJ+1
                    DD1(IADD)=DD1(IADD)+DDFGH(17)+DDFGH(38)-DDFGH(20)-DDFGH(41)
                 ELSE
                    IADD=IUPT(JJ)+KK
                    DD1(IADD)=DD1(IADD)+DDFGH(7)+DDFGH(22)-DDFGH(10)-DDFGH(25)
                    IADD=IUPT(JJ+1)+KK+1
                    DD1(IADD)=DD1(IADD)+DDFGH(12)+DDFGH(30)-DDFGH(15)-DDFGH(33)
                    IADD=IUPT(JJ+2)+KK+2
                    DD1(IADD)=DD1(IADD)+DDFGH(18)+DDFGH(39)-DDFGH(21)-DDFGH(42)
                    IADD=IUPT(JJ+1)+KK
                    DD1(IADD)=DD1(IADD)+DDFGH(8)+DDFGH(23)-DDFGH(14)-DDFGH(26)
                    IADD=IUPT(JJ)+KK+1
                    DD1(IADD)=DD1(IADD)+DDFGH(11)+DDFGH(29)-DDFGH(14)-DDFGH(32)
                    IADD=IUPT(JJ+2)+KK
                    DD1(IADD)=DD1(IADD)+DDFGH(9)+DDFGH(24)-DDFGH(19)-DDFGH(27)
                    IADD=IUPT(JJ)+KK+2
                    DD1(IADD)=DD1(IADD)+DDFGH(16)+DDFGH(37)-DDFGH(19)-DDFGH(40)
                    IADD=IUPT(JJ+2)+KK+1
                    DD1(IADD)=DD1(IADD)+DDFGH(13)+DDFGH(31)-DDFGH(20)-DDFGH(34)
                    IADD=IUPT(JJ+1)+KK+2
                    DD1(IADD)=DD1(IADD)+DDFGH(17)+DDFGH(38)-DDFGH(20)-DDFGH(41)
                 ENDIF
                 !
                 IF (JLTEST) THEN
                    IADD=IUPT(LL)+JJ
                    DD1(IADD)=DD1(IADD)+DDFGH(25)-DDFGH(22)
                    IADD=IUPT(LL+1)+JJ+1
                    DD1(IADD)=DD1(IADD)+DDFGH(33)-DDFGH(30)
                    IADD=IUPT(LL+2)+JJ+2
                    DD1(IADD)=DD1(IADD)+DDFGH(42)-DDFGH(39)
                    IADD=IUPT(LL)+JJ+1
                    DD1(IADD)=DD1(IADD)+DDFGH(26)-DDFGH(23)
                    IADD=IUPT(LL+1)+JJ
                    DD1(IADD)=DD1(IADD)+DDFGH(32)-DDFGH(29)
                    IADD=IUPT(LL)+JJ+2
                    DD1(IADD)=DD1(IADD)+DDFGH(27)-DDFGH(24)
                    IADD=IUPT(LL+2)+JJ
                    DD1(IADD)=DD1(IADD)+DDFGH(40)-DDFGH(37)
                    IADD=IUPT(LL+1)+JJ+2
                    DD1(IADD)=DD1(IADD)+DDFGH(34)-DDFGH(31)
                    IADD=IUPT(LL+2)+JJ+1
                    DD1(IADD)=DD1(IADD)+DDFGH(41)-DDFGH(38)
                 ELSE
                    IADD=IUPT(JJ)+LL
                    DD1(IADD)=DD1(IADD)+DDFGH(25)-DDFGH(22)
                    IADD=IUPT(JJ+1)+LL+1
                    DD1(IADD)=DD1(IADD)+DDFGH(33)-DDFGH(30)
                    IADD=IUPT(JJ+2)+LL+2
                    DD1(IADD)=DD1(IADD)+DDFGH(42)-DDFGH(39)
                    IADD=IUPT(JJ+1)+LL
                    DD1(IADD)=DD1(IADD)+DDFGH(26)-DDFGH(23)
                    IADD=IUPT(JJ)+LL+1
                    DD1(IADD)=DD1(IADD)+DDFGH(32)-DDFGH(29)
                    IADD=IUPT(JJ+2)+LL
                    DD1(IADD)=DD1(IADD)+DDFGH(27)-DDFGH(24)
                    IADD=IUPT(JJ)+LL+2
                    DD1(IADD)=DD1(IADD)+DDFGH(40)-DDFGH(37)
                    IADD=IUPT(JJ+2)+LL+1
                    DD1(IADD)=DD1(IADD)+DDFGH(34)-DDFGH(31)
                    IADD=IUPT(JJ+1)+LL+2
                    DD1(IADD)=DD1(IADD)+DDFGH(41)-DDFGH(38)
                 ENDIF
                 !
                 IF (KLTEST) THEN
                    IADD=IUPT(LL)+KK
                    DD1(IADD)=DD1(IADD)-DDFGH(25)-DDFGH(28)
                    IADD=IUPT(LL+1)+KK+1
                    DD1(IADD)=DD1(IADD)-DDFGH(33)-DDFGH(36)
                    IADD=IUPT(LL+2)+KK+2
                    DD1(IADD)=DD1(IADD)-DDFGH(42)-DDFGH(45)
                    IADD=IUPT(LL)+KK+1
                    DD1(IADD)=DD1(IADD)-DDFGH(26)-DDFGH(35)
                    IADD=IUPT(LL+1)+KK
                    DD1(IADD)=DD1(IADD)-DDFGH(32)-DDFGH(35)
                    IADD=IUPT(LL)+KK+2
                    DD1(IADD)=DD1(IADD)-DDFGH(27)-DDFGH(43)
                    IADD=IUPT(LL+2)+KK
                    DD1(IADD)=DD1(IADD)-DDFGH(40)-DDFGH(43)
                    IADD=IUPT(LL+1)+KK+2
                    DD1(IADD)=DD1(IADD)-DDFGH(34)-DDFGH(44)
                    IADD=IUPT(LL+2)+KK+1
                    DD1(IADD)=DD1(IADD)-DDFGH(41)-DDFGH(44)
                 ELSE
                    IADD=IUPT(KK)+LL
                    DD1(IADD)=DD1(IADD)-DDFGH(25)-DDFGH(28)
                    IADD=IUPT(KK+1)+LL+1
                    DD1(IADD)=DD1(IADD)-DDFGH(33)-DDFGH(36)
                    IADD=IUPT(KK+2)+LL+2
                    DD1(IADD)=DD1(IADD)-DDFGH(42)-DDFGH(45)
                    IADD=IUPT(KK+1)+LL
                    DD1(IADD)=DD1(IADD)-DDFGH(26)-DDFGH(35)
                    IADD=IUPT(KK)+LL+1
                    DD1(IADD)=DD1(IADD)-DDFGH(32)-DDFGH(35)
                    IADD=IUPT(KK+2)+LL
                    DD1(IADD)=DD1(IADD)-DDFGH(27)-DDFGH(43)
                    IADD=IUPT(KK)+LL+2
                    DD1(IADD)=DD1(IADD)-DDFGH(40)-DDFGH(43)
                    IADD=IUPT(KK+2)+LL+1
                    DD1(IADD)=DD1(IADD)-DDFGH(34)-DDFGH(44)
                    IADD=IUPT(KK+1)+LL+2
                    DD1(IADD)=DD1(IADD)-DDFGH(41)-DDFGH(44)
                 ENDIF

#if KEY_DIMB==1
              ENDIF qcmpcttest  
#endif

           ENDIF
#if KEY_BLOCK==1
        ENDIF
#endif /*  BLOCK*/
        !
160     CONTINUE
10      CONTINUE
   ENDDO
   !
   NWARN=NWARN+NWARNX
   IF(NWARN.GT.5 .AND. WRNLEV.GE.2) WRITE(OUTU,170) NWARN
170  FORMAT(' TOTAL OF',I6,' WARNINGS FROM EPHI')
   !
   RETURN
END SUBROUTINE EPHI

 end module eintern

