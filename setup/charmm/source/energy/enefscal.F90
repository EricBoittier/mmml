module eintern_fast
  implicit none
  private

  ! Public subroutines
  public fastst, ebondfs, eanglfs, ephifs, eiphifs, bactiv, mkbactive, &
       ebondfaster, eanglfaster, eanglfastergrom, ephifaster, eiphifaster
  
contains

  SUBROUTINE FASTST(XA,YA,ZA,DXA,DYA,DZA,QSECD)
    !-----------------------------------------------------------------------
    !     Calculates the energy and forces as fast as possible and assumes
    !     that no fancy stuff is being tried. It also assumes that the main
    !     coordinate set holds the active coordinates. The PSF is included
    !     in most subroutines, so these routines are not as flexible as the
    !     normal energy routines. Also for nonbonded interactions, only one
    !     option is supported (see code for latest options).
    !
    !     The FASTER variable determines what operation takes place.
    !     FASTer command allows user specified action.
    !
    !     FASTER = -1  : Always use slow routines
    !     FASTER =  0  : Use best fast routines possible, no error if not...
    !     FASTER =  1  : Use generic fast routines ! Error if cannot use
    !     FASTER =  2  : Use expanded fast routines ! Error if cannot use
    !
    !     FASTer routines are chosen based on LFAST
    !     LFAST = -1, do not use FAST routines
    !              0, use the best possible based on methods used (default)
    !              1, use generic fast only
    !              2, use limited fast only
    !
    !     LFAST is determined from FASTER and the methods and options used.
    !           For example if one uses a method which needs a full
    !           Hessian, then LFAST will be -1 when FASTER is -1 or 0.
    !           If FASTER is 1 or 2, then an error message will result.
    !
    !     14-JUL-83 Bernard R. Brooks
    !     31-Aug-91 Youngdo Won, LFAST routing
    !     22-OCT-91              Check if nonbond options are valid
    !-----------------------------------------------------------------------
    use chm_kinds
    use dimens_fcm
    use coord
    use deriv
    use fast
    use fourdm
    use inbnd
    use param
    use psf
    use stream
    !pssp SEE COMMENT BELOW
#if KEY_PERT==1
    use pert
#endif 
    use machutil,only:locdif

    implicit none
    real(chm_real) XA(*),YA(*),ZA(*)
    real(chm_real) DXA(*),DYA(*),DZA(*)
    LOGICAL QSECD
    INTEGER DUMMY
    INTEGER OLDFST,INWRN,NBWRN
    DATA    INWRN,NBWRN/0,0/
    !
    OLDFST=LFAST
    LFAST=-1
    IF (FASTER.LT.0) RETURN
#if KEY_PERT==1
    !pssp
    !     DEPENDING ON THE VALUE PAIR QPSSP/TQPSSP WE WANT TO FORCE THE
    !     USE OF THE SLOW ROUTINES, I.E. FORCE CALCULATION OF REACTANT
    !     AND PRODUCT PART WITH SLOW ROUTINES. N.B.: ENVIRONMENT PART
    !     (NORMALLY THE BULK OF INTERACTIONS) IS COMPUTED NORMALLY WITH
    !     FASTEST SUPPORTED ROUTINE
    IF (QPSSP.AND.TQPSSP) RETURN
    !pssp
#endif 
    IF (QSECD) THEN
       IF (OLDFST.GT.0 .AND. WRNLEV.GE.2) WRITE (OUTU,'(2A,/A)') &
            ' FASTST> Second derivatives are not supported', &
            ' with FAST option.', &
            '         Using generic routines (enables second derivatives).'
       RETURN
    ENDIF
#if KEY_FOURD==1
    IF (DIM4) THEN
       IF (OLDFST.GT.0 .AND. WRNLEV.GE.2) WRITE (OUTU,'(A,/A)') &
            ' FASTST> 4th dimension is not supported with FAST option.', &
            '         Using generic routines (enables 4th dimension code).'
       RETURN
    ENDIF
#endif 
    IF (LOCDIF(DX,DXA,DUMMY).NE.0) GOTO 100
    IF (LOCDIF(DY,DYA,DUMMY).NE.0) GOTO 100
    IF (LOCDIF(DZ,DZA,DUMMY).NE.0) GOTO 100
    IF (LOCDIF(X,XA,DUMMY).NE.0) GOTO 100
    IF (LOCDIF(Y,YA,DUMMY).NE.0) GOTO 100
    IF (LOCDIF(Z,ZA,DUMMY).NE.0) GOTO 100
    !
    LFAST=FASTER
    RETURN
    !
100 CONTINUE
    IF (FASTER.LE.0) RETURN
    IF (INWRN.EQ.0 .AND. WRNLEV.GE.2) WRITE (OUTU,33)
    INWRN=INWRN+1
    !
33  FORMAT(' FASTST> Bad array location in memory for ', &
         'FAST routines.')
    RETURN
  END SUBROUTINE FASTST

  SUBROUTINE EBONDFS(EB,NBOND,IB,JB,ICB,CBB,CBC)
    !-----------------------------------------------------------------------
    !     calculates bond energies and forces as fast as possible
    !     Fast SCALAR version
    !     14-JUL-1983, Bernard R. Brooks
    !     31-Aug-1991, Youngdo Won
    !-----------------------------------------------------------------------
    use chm_kinds
    use dimens_fcm
    use number
#if KEY_BLOCK==1
    use block_fcm
    use lambdam
    use pert  !Cc New PBLOCK
#endif /*  BLOCK*/
    use deriv
    use coord
    use stream
    use parallel
#if KEY_GRAPE==1
    use grape,only:lfmm
#endif
#if KEY_SPACDEC==1
    use spacdec      
#endif
#if KEY_ACTBOND==1
    use actclus_mod  
#endif
#if KEY_GENETIC==1
    use galgor_ltm,only:qGA_Ener    
#endif
    implicit none
    real(chm_real) EB
    INTEGER NBOND
    INTEGER IB(*),JB(*),ICB(*)
    real(chm_real) CBB(*),CBC(*)
    !
    real(chm_real) RX,RY,RZ,S,R,DB,DF,DXI,DYI,DZI
    INTEGER MM,I,J,IC,IIMM
#if KEY_BLOCK==1
    INTEGER IBL,JBL,KK, KDOC
    real(chm_real) COEF, DOCFI, DOCFJ
!ldm
    real(chm_real) UNSCALE, FALPHA
!ldm
#endif /*  BLOCK*/
#if KEY_ACTBOND==1 /*actbond0*/
    INTEGER UPLIMIT
#endif /* (actbond0)*/
    integer ii,indfirst,indlast,indinc

    !
    !     WRITE(OUTU,'(A)') 'in EBONDFS'
    !
#if KEY_GENETIC==1
    INTEGER FIRST
    ! **************************************************
    First = 1
    If(qGA_Ener) First = Int(EB)
#endif /* GENETIC*/
    !
#if KEY_ACTBOND==1 /*actbond0*/
    !      IF(.NOT.QBACTON) CALL BACTIVDEF
#if KEY_GENETIC==1 /*gentest0*/
    CALL WRNDIE(-5,'<EBONDFS> ','GENETIC AND ACTBOND KEYWORDS' &
         //' ARE INCOMPATIBLE')
#endif /* (gentest0)*/
#endif /* (actbond0)*/
    EB=ZERO

    IF (NBOND == 0) RETURN

       !
       indfirst=1
       indlast=nbond
       indinc=1

#if KEY_PARALLEL==1 /*parabond*/
#if KEY_PARAFULL==1 /*parfbond*/
       indfirst = MYNODP
       indlast  = NBOND
       indinc   = NUMNOD
#elif KEY_PARASCAL==1 /*parfbond*/
#if KEY_GENETIC==1
       indfirst=First
       indlast=NBOND
#endif 
#elif KEY_SPACDEC==1 /*parfbond*/
       ! the above default is OK here
#else /* (parfbond)*/
#error  'Illegal parallel compile option'
#endif /* (parfbond)*/

#else /* (parabond)*/

#if KEY_GENETIC==1 /*genebond*/
     indfirst=First
     indlast=NBOND
#endif /*genebond*/
#endif /* (parabond)*/

#if KEY_ACTBOND==1 /*actbond for active bonds */
     indfirst = 1
     indinc = 1
     IF (QBACTON) THEN
        IF (QUREYBR) THEN  !UB term
           UPLIMIT = NACTANG
        ELSE
           UPLIMIT = NACTBND
        ENDIF
     ELSE !active bonds off
        UPLIMIT = NBOND
     ENDIF
     indlast=UPLIMIT
#if KEY_PARALLEL==1 /*paract*/
     indinc=NUMNOD
     indfirst=MYNODP
#endif /*paract*/
#endif /*actbond*/

#if KEY_GRAPE==1
     if(lfmm)then
        indfirst=1
        indlast=nbond
        indinc=1
     endif
#endif
    loop60: DO ii=indfirst,indlast,indinc

       mm = ii

#if KEY_ACTBOND==1 /*actbond2*/ /*for active bonds*/
       IF (QBACTON) THEN
         IF (QUREYBR) THEN  !UB term
           MM = ACTANGL(II)
         ELSE
           MM = ACTBOND(II)
         ENDIF
       ENDIF
#endif /*actbond2*/
       !
       I=IB(MM)
       J=JB(MM)
#if KEY_GRAPE==1
       if(lfmm) then
          if((fmmcpu(i)/=1).and.(fmmcpu(j)/=1)) cycle loop60
       endif
#endif
       IF (I.LE.0) cycle loop60
       IC=ICB(MM)
       IF (IC.EQ.0) cycle loop60
       IF (CBC(IC).EQ.0.0) cycle loop60
       RX=X(I)-X(J)
       RY=Y(I)-Y(J)
       RZ=Z(I)-Z(J)
       S=SQRT(RX*RX+RY*RY+RZ*RZ)
       IF (S.EQ.0.0) cycle loop60
       R=2.0/S
       DB=S-CBB(IC)
       DF=CBC(IC)*DB
#if KEY_BLOCK==1 /*blocker*/
       IF (QBLOCK) THEN
!ldm
          if (qmld) then
             if (qldm_ureybr) then
                call msld_scale_force(mldureyc(mm),mldureyi(mm),mldureyj(mm),&
                     blcoep(mldureyc(mm)),df,df*db)
             else
                call msld_bond_scale_force(qnobo,iblckp(i),iblckp(j),df,df*db)
             endif
          else
             IBL = IBLCKP(I)
             JBL = IBLCKP(J)
!LDM
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
             IF (JBL.LT.IBL) THEN
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
!ldm
             IF (QLDM .or. QLMC) THEN
                !     first row or diagonal elements exclude (1,1).
                UNSCALE = 0.0
                IF ((IBL == 1 .AND. JBL >= LSTRT) .or. &
                     (IBL >= LSTRT .AND. IBL == JBL)) UNSCALE = DF
                IF (RSTP .AND. .NOT. QNOBO) THEN
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
             ENDIF
!LDM
             IF(QNOBO) COEF = 1.0
             DF=DF*COEF
!ldm
             IF ((QLDM .or. QLMC) .AND. .NOT. QNOBO) then
                IF ((IBL == 1 .AND. JBL >= LSTRT) .or. &
                     (IBL >= LSTRT .AND. IBL == JBL)) THEN
                   FALPHA = UNSCALE*DB
#if KEY_PERT==1
                   if(QPERT) FALPHA = FALPHA*PERTLAM  
#endif
                   LAGMUL = LAGMUL + FALPHA
                   BIFLAM(JBL) = BIFLAM(JBL) + FALPHA
                   IF (NRST == 2) BFRST(JBL) = BFRST(JBL) + FALPHA
                ENDIF
             ENDIF
          endif
!ldm
       ENDIF
#if KEY_DOCK==1
       IF(QDOCK) THEN
          !         Factor 0.5 to make sure no double counting
          EB=EB+DF*DB*0.5*(DOCFI + DOCFJ)
       ELSE
#endif /*  DOCK*/
#endif /* (blocker)*/
          EB=EB+DF*DB
#if KEY_BLOCK==1
#if KEY_DOCK==1
       ENDIF
#endif /*  DOCK*/
#endif /*  BLOCK*/
       !
#if KEY_BLOCK==1
       IF (.NOT.NOFORC) THEN
#endif 
          DF=DF*R
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
       ENDIF
#endif 
       !
    enddo loop60
    RETURN
  END SUBROUTINE EBONDFS

  SUBROUTINE EANGLFS(ET)
    !-----------------------------------------------------------------------
    !     calculates bond angles and bond angle energies as fast as possible
    !     Fast SCALAR version
    !     14-JUL-1983, Bernard R. Brooks
    !     31-Aug-1991, Youngdo Won
    !-----------------------------------------------------------------------
    use chm_kinds
    use block_fcm
    use dimens_fcm
    use number
    use psf
    use deriv
    use coord
    use code
    use param
    use stream
    use consta
    !
#if KEY_BLOCK==1 || KEY_SPACDEC==1 /*block1*/
    use block_fcm
    use lambdam
    use pert  !Cc New PBLOCK
#endif /* (block1)*/
    use parallel
#if KEY_GRAPE==1
    use grape,only:lfmm
#endif
#if KEY_SPACDEC==1
    use spacdec       
#endif
#if KEY_ACTBOND==1
    use actclus_mod   
#endif
#if KEY_GENETIC==1
    use galgor_ltm    
#endif
    implicit none
    !
    real(chm_real) ET
    !
    real(chm_real) DXI,DYI,DZI,DXJ,DYJ,DZJ,SMALLV,ST2R,STR
    real(chm_real) RI,RJ,RIR,RJR,DXIR,DYIR,DZIR,DXJR,DYJR,DZJR
    real(chm_real) CST,AT,DA,DF,DTXI,DTXJ,DTYI,DTYJ,DTZI,DTZJ,DTI,DTJ
    INTEGER NWARN,ITH,I,J,K,IC,IIMM
#if KEY_BLOCK==1 /*block2*/
    real(chm_real) COEF, DOCFI, DOCFJ, DOCFK, DOCFJ1
    INTEGER IBL, JBL, KK, KDOC
!ldm
    real(chm_real) UNSCALE, FALPHA
    real(chm_real) DFA
!ldm
#endif /* (block2)*/
#if KEY_ACTBOND==1 /*actbond0*/
    INTEGER UPLIMIT
#endif /* (actbond0)*/
    integer ii,indfirst,indlast,indinc

#if KEY_GENETIC==1
    INTEGER First   
#endif


    ! *********************************************************
#if KEY_GENETIC==1
    First = 1
    If(qGA_Ener) then
       First = Int(ET)
    Else
       Lstangl = NTHETA
    EndIf
#endif 
    ET=ZERO
    SMALLV=RPRECI
    NWARN=0

    IF (NTHETA == 0) RETURN

       !
       indfirst=1
       indlast=ntheta
       indinc=1

#if KEY_PARALLEL==1 /*paraangle*/
#if KEY_PARAFULL==1 /*parfangle*/
       indfirst=MYNODP
       indlast=NTHETA
       indinc=NUMNOD
#elif KEY_PARASCAL==1 /*parfangle*/
#if KEY_GENETIC==1
       indfirst=First
       indlast=LstAngl
#endif 
#endif /* (parfangle)*/

#else /* (paraangle)*/

#if KEY_GENETIC==1 /*geneang*/
       indfirst=First
       indlast=LstAngl
#endif /* (geneang)*/
#endif /* (paraangle)*/

#if KEY_ACTBOND==1 /*actangle*/ /*for active angles*/
       indfirst=1
       indinc = 1
       if(QBACTON) THEN
          UPLIMIT = NACTANG
       else
          UPLIMIT = NTHETA
       endif
       indlast=UPLIMIT
#if KEY_PARALLEL==1 /*paractang*/
       indinc=NUMNOD
       indfirst=MYNODP
#endif /*paractang*/
#endif /*actangle*/

#if KEY_GRAPE==1
       if(lfmm)then
          indfirst=1
          indlast=ntheta
          indinc=1
       endif
#endif

    loop50: do ii=indfirst,indlast,indinc
       ith=ii
#if KEY_PARALLEL==1 /*paraangle*/
#if KEY_PARASCAL==1 /*parfangle*/
          IF(PSICT(ITH).NE.MYNOD) cycle loop50
#endif /*parfangle*/
#endif /*paraangle*/

#if KEY_ACTBOND==1 /*actang2*/ /*for active angles*/
          IF(QBACTON) ITH = ACTANGL(II)
#endif /* actang2*/

       i=it(ith)
       j=jt(ith)
       k=kt(ith)
#if KEY_GRAPE==1
       if(lfmm) then
          if((fmmcpu(i)/=1).and.(fmmcpu(j)/=1).and.(fmmcpu(k)/=1)) cycle loop50
       endif
#endif
       if (i.le.0) cycle loop50
       ic=ict(ith)
       if (ic.eq.0) cycle loop50
       dxi=x(i)-x(j)
       dyi=y(i)-y(j)
       dzi=z(i)-z(j)
       dxj=x(k)-x(j)
       dyj=y(k)-y(j)
       dzj=z(k)-z(j)
       ri=sqrt(dxi*dxi+dyi*dyi+dzi*dzi)
       if (ri.eq.0.0) cycle loop50
       rj=sqrt(dxj*dxj+dyj*dyj+dzj*dzj)
       if (rj.eq.0.0) cycle loop50
       rir=1.0/ri
       rjr=1.0/rj
       dxir=dxi*rir
       dyir=dyi*rir
       dzir=dzi*rir
       dxjr=dxj*rjr
       dyjr=dyj*rjr
       dzjr=dzj*rjr
       cst=dxir*dxjr+dyir*dyjr+dzir*dzjr
       !
       if(abs(cst).ge.0.999999) then
          if(abs(cst).gt.one) cst=sign(one,cst)
          at=acos(cst)
          da=at-ctb(ic)
          if(abs(da).gt.0.1) then
             nwarn=nwarn+1
             if((nwarn.le.5 .and. wrnlev.ge.5) .or. wrnlev.ge.6) then
                WRITE(OUTU,10) ITH,I,J,K
10              FORMAT(' WARNING FROM EANGLFS. Angle',I5, &
                     '  is almost linear.', &
                     /' Derivatives may be affected for atoms:',3I5)
                WRITE(OUTU,101) 'I ATOM:',X(I),Y(I),Z(I)
                WRITE(OUTU,101) 'J ATOM:',X(J),Y(J),Z(J)
                WRITE(OUTU,101) 'K ATOM:',X(K),Y(K),Z(K)
                WRITE(OUTU,101) 'DXIR  :',DXIR,DYIR,DZIR
                WRITE(OUTU,101) 'DXJR  :',DXJR,DYJR,DZJR
                WRITE(OUTU,101) 'CST   :',CST,AT*RADDEG,DA*RADDEG
101             FORMAT(5X,A,5F15.5)
             ENDIF
          ENDIF
       ENDIF
       !
       AT=ACOS(CST)
       DA=AT-CTB(IC)
       DF=CTC(IC)*DA
       !
#if KEY_BLOCK==1 /*block3*/
       IF (QBLOCK) THEN
!ldm
          if (qmld) then
             call msld_scale_force(mldangc(ith),mldangi(ith),mldangj(ith),&
                  blcoep(mldangc(ith)),df,df*da)
          else
! LDM
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
             IF (IBL.EQ.JBL) JBL=KK
             IF (JBL.LT.IBL) THEN
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
!ldm
             IF (QLDM .or. QLMC) THEN
                ! first row or diagonal elements exclude (1,1).
                UNSCALE = 0.0
                IF ((IBL == 1 .AND. JBL >= LSTRT) .or. &
                     (IBL >= LSTRT .AND. IBL == JBL)) UNSCALE = DF
             ENDIF
             IF(RSTP.AND. .NOT. QNOAN)THEN
                IF ((IBL == 1 .AND. JBL >= LSTRT) .or. &
                     (IBL >= LSTRT .AND. IBL == JBL)) THEN
                   IF(ABS(CST).GE.0.999999) THEN
                      ST2R=ONE/(ONE-CST*CST+SMALLV)
                      STR=SQRT(ST2R)
                      IF(CTB(IC).LT.PT001) THEN
                         DFA=MINTWO*CTC(IC)*(ONE+DA*DA*SIXTH)
                      ELSE IF(PI-CTB(IC).LT.PT001) THEN
                         DFA=TWO*CTC(IC)*(ONE+DA*DA*SIXTH)
                      ELSE
                         DFA=MINTWO*DF*STR
                      ENDIF
                   ELSE
                      ST2R=ONE/(ONE-CST*CST)
                      STR=SQRT(ST2R)
                      DFA=MINTWO*DF*STR
                   ENDIF

                   DTXI=RIR*(DXJR-CST*DXIR)
                   DTXJ=RJR*(DXIR-CST*DXJR)
                   DTYI=RIR*(DYJR-CST*DYIR)
                   DTYJ=RJR*(DYIR-CST*DYJR)
                   DTZI=RIR*(DZJR-CST*DZIR)
                   DTZJ=RJR*(DZIR-CST*DZJR)

                   ENVDX(I) = ENVDX(I) + DFA * DTXI
                   ENVDY(I) = ENVDY(I) + DFA * DTYI
                   ENVDZ(I) = ENVDZ(I) + DFA * DTZI
                   ENVDX(K) = ENVDX(K) + DFA * DTXJ
                   ENVDY(K) = ENVDY(K) + DFA * DTYJ
                   ENVDZ(K) = ENVDZ(K) + DFA * DTZJ
                   ENVDX(J) = ENVDX(J) - DFA * (DTXI + DTXJ)
                   ENVDY(J) = ENVDY(J) - DFA * (DTYI + DTYJ)
                   ENVDZ(J) = ENVDZ(J) - DFA * (DTZI + DTZJ)
                ENDIF
             ENDIF
!LDM
             IF(QNOAN) COEF = 1.0
             DF=DF*COEF
!ldm
             IF ((QLDM .or. QLMC) .AND. .NOT. QNOAN) THEN
                IF ((IBL == 1 .AND. JBL >= LSTRT) .or. &
                     (IBL >= LSTRT .AND. IBL == JBL)) THEN
                   FALPHA = UNSCALE*DA
#if KEY_PERT==1
                   if(QPERT) FALPHA = FALPHA*PERTLAM  
#endif
                   LAGMUL = LAGMUL + FALPHA
                   BIFLAM(JBL) = BIFLAM(JBL) + FALPHA
                   IF (NRST == 2) BFRST(JBL) = BFRST(JBL) + FALPHA
                ENDIF
             ENDIF
          endif
       ENDIF

#if KEY_DOCK==1
       IF(QDOCK) THEN
          ET= ET+DF*DA*0.25*(DOCFI+DOCFJ+DOCFK+DOCFJ1)
       ELSE
#endif 
#endif /* (block3)*/
          ET= ET+DF*DA
#if KEY_BLOCK==1 /*block4*/
#if KEY_DOCK==1
       ENDIF
#endif 
       IF (.NOT.NOFORC) THEN
#endif /* (block4)*/
          IF(ABS(CST).GE.0.999999) THEN
             ST2R=ONE/(ONE-CST*CST+SMALLV)
             STR=SQRT(ST2R)
             IF(CTB(IC).LT.PT001) THEN
                DF=TWO*MINTWO*CTC(IC)*(ONE+DA*DA*SIXTH)
             ELSE IF(PI-CTB(IC).LT.PT001) THEN
                DF=TWO*TWO*CTC(IC)*(ONE+DA*DA*SIXTH)
             ELSE
                DF=MINTWO*DF*STR
             ENDIF
          ELSE
             ST2R=ONE/(ONE-CST*CST)
             STR=SQRT(ST2R)
             DF=MINTWO*DF*STR
          ENDIF
          !
          DTXI=RIR*(DXJR-CST*DXIR)
          DTXJ=RJR*(DXIR-CST*DXJR)
          DTYI=RIR*(DYJR-CST*DYIR)
          DTYJ=RJR*(DYIR-CST*DYJR)
          DTZI=RIR*(DZJR-CST*DZIR)
          DTZJ=RJR*(DZIR-CST*DZJR)
          !
#if KEY_BLOCK==1 /*block5*/
#if KEY_DOCK==1
          IF(QDOCK) THEN
             DTI=DF*DTXI
             DTJ=DF*DTXJ
             DX(I)=DX(I)+DTI*DOCFI
             DX(K)=DX(K)+DTJ*DOCFK
             DX(J)=DX(J)-DTI*DOCFJ-DTJ*DOCFJ1
             !
             DTI=DF*DTYI
             DTJ=DF*DTYJ
             DY(I)=DY(I)+DTI*DOCFI
             DY(K)=DY(K)+DTJ*DOCFK
             DY(J)=DY(J)-DTI*DOCFJ-DTJ*DOCFJ1
             !
             DTI=DF*DTZI
             DTJ=DF*DTZJ
             DZ(I)=DZ(I)+DTI*DOCFI
             DZ(K)=DZ(K)+DTJ*DOCFK
             DZ(J)=DZ(J)-DTI*DOCFJ-DTJ*DOCFJ1
          ELSE
#endif 
#endif /* (block5)*/
             DTI=DF*DTXI
             DTJ=DF*DTXJ
             DX(I)=DX(I)+DTI
             DX(K)=DX(K)+DTJ
             DX(J)=DX(J)-DTI-DTJ
             !
             DTI=DF*DTYI
             DTJ=DF*DTYJ
             DY(I)=DY(I)+DTI
             DY(K)=DY(K)+DTJ
             DY(J)=DY(J)-DTI-DTJ
             !
             DTI=DF*DTZI
             DTJ=DF*DTZJ
             DZ(I)=DZ(I)+DTI
             DZ(K)=DZ(K)+DTJ
             DZ(J)=DZ(J)-DTI-DTJ
#if KEY_BLOCK==1 /*block6*/
#if KEY_DOCK==1
          ENDIF
#endif 
       ENDIF
#endif /* (block6)*/
    enddo loop50
    !
    IF(NWARN.GT.5 .AND. WRNLEV.GE.3) WRITE(OUTU,455) NWARN
455 FORMAT(' EANGLFS> Total of',I6,' warnings.')
    !
    RETURN
  END SUBROUTINE EANGLFS

  SUBROUTINE EPHIFS(EP)
    !-----------------------------------------------------------------------
    !     Fast SCALAR version dihedral energy and force routine.
    !     Dihedral energy terms are expressed as a function of PHI.
    !     This avoids all problems as dihedrals become planar.
    !     The functional form is:
    !
    !     E = K*(1+COS(n*Phi-Phi0))
    !     Where:
    !     n IS A POSITIVE INTEGER COS(n(Phi-Phi0) AND SIN(n(Phi-Phi0)
    !            ARE CALCULATED BY RECURENCE TO AVOID LIMITATION ON n
    !     K IS THE FORCE CONSTANT IN kcal/mol/rad
    !     Phi0/n IS A MAXIMUM IN ENERGY.
    !
    !     The parameter of the routine is:
    !     EP: Diheral Energy returned
    !     Data come form param.f90
    !
    !     31-Aug-1991, Youngdo Won
    !
    !     New formulation by
    !           Arnaud Blondel    1994
    !
    !    MODIFIED BY HIQMET KAMBERAJ
    !    ADDED: TSALLIS MD ON TORSIONS ANGLES
    !    2007
    !
    !-----------------------------------------------------------------------
#if KEY_TSALLIS==1
    use tsallis_module,only : tdx,tdy,tdz,ebtors, &
         tsndf,qtalpha,qttsall,tsalpha,iascale
#endif 
    use chm_kinds
    use dimens_fcm
    use number
    use code
    use consta
    use coord
    use deriv
    use param
    use psf
    use stream
    use parallel

#if KEY_BLOCK==1
    use lambdam
    use block_fcm
    use pert  !Cc New PBLOCK
#endif
#if KEY_GRAPE==1
    use grape,only:lfmm
#endif
#if KEY_SPACDEC==1
    use spacdec    
#endif
#if KEY_ACTBOND==1
    use actclus_mod 
#endif
#if KEY_GENETIC==1
    use galgor_ltm     
#endif
    implicit none
    real(chm_real) EP
    !
#if KEY_BLOCK==1
    INTEGER IBL,JBL,KKK,LLL, KDOC
    real(chm_real)  COEF, DOCFI, DOCFJ, DOCFK, DOCFJ1, DOCFK1, DOCFL
    !.ab.
    real(chm_real) FR,FP,DFR,DFP,DPR,DPP
    !.ab.
!ldm
    real(chm_real) UNSCALE, FALPHA
    real(chm_real)  DFORG, RA2RORG, RB2RORG
! LDM
#endif /*  BLOCK*/

    !
    real(chm_real)  FX,FY,FZ,GX,GY,GZ,HX,HY,HZ
    real(chm_real)  AX,AY,AZ,BX,BY,BZ,RA2,RB2,RG,RGR
    real(chm_real)  RA2R,RB2R,RABR,CT,CX,CY,CZ,ST
    real(chm_real)  GAA,GBB,FG,HG,FGA,HGB
    real(chm_real)  E,DF,E1,DF1,DDF1,ARG
    real(chm_real)  DFX,DFY,DFZ,DHX,DHY,DHZ,DGX,DGY,DGZ
    INTEGER NWARN,IPHI,I,J,K,L,IC,IPER,NPER,IIMM
    LOGICAL LREP
    !
    real(chm_real)  RXMIN,RXMIN2
    PARAMETER (RXMIN=0.005,RXMIN2=0.000025D0)

    ! Tsallis scaling of energy
#if KEY_TSALLIS==1
    !  local variables
    integer :: fi,fj,fk,fl, sum1
    logical :: tsallismd
#endif 

#if KEY_ACTBOND==1
    INTEGER UPLIMIT      
#endif
    integer ii,indfirst,indlast,indinc
#if KEY_GENETIC==1
    INTEGER First   
#endif



    !
    !     Initialize the torsion energy to zero
    ! ********************************************************
#if KEY_GENETIC==1
    First = 1
    If(qGA_Ener) then
       First = Int(EP)
    Else
       LstPhi = NPHI
    EndIf
#endif 
    !
    EP=ZERO
#if KEY_BLOCK==1
    IF (QHYBH) THEN
       DPR=ZERO
       DPP=ZERO
       FR=(THREE*HYBHLB-FOUR)*(HYBHLB-ONE)/FOUR
       DFR=(SIX*HYBHLB-SEVEN)/FOUR
       FP=(THREE*HYBHLB+ONE)*HYBHLB/FOUR
       DFP=(SIX*HYBHLB+ONE)/FOUR
    ENDIF
#endif 
    !.ab.
    IF (NPHI == 0) RETURN
    NWARN=0
    !
#if KEY_TSALLIS==1
    EBTORS(1)=ZERO    
#endif
    !

    indfirst=1
    indlast=nphi
    indinc=1

#if KEY_PARALLEL==1 /*paraphi*/
#if KEY_PARAFULL==1 /*parfphi*/
    indfirst=MYNODP
    indlast=NPHI
    indinc=NUMNOD
#elif KEY_PARASCAL==1 /*parfphi*/
#if KEY_GENETIC==1
    indfirst=First
    indlast=LstPhi
#endif 
#endif /* (parfphi)*/

#else /* (paraphi)*/

#if KEY_GENETIC==1 /*genephi*/
    indfirst=First
    indlast=LstPhi
#endif /* (genephi)*/
#endif /* (paraphi)*/

#if KEY_ACTBOND==1 /*actphi*/ /*for active dihedrals*/
    indfirst = 1
    INDINC = 1
    IF(QBACTON) THEN
       UPLIMIT = NACTDIH
    ELSE
       UPLIMIT = NPHI
    ENDIF
    indlast=UPLIMIT
#if KEY_PARALLEL /*paractphi*/
    indinc=NUMNOD
    indfirst=MYNODP
#endif /*paractphi*/
#endif /*actphi*/

#if KEY_GRAPE==1
    if(lfmm)then
       indfirst=1
       indlast=nphi
       indinc=1
    endif
#endif

    loop70: do ii=indfirst,indlast,indinc
       iphi=ii
#if KEY_ACTBOND==1 /*actphi2*/ /*for active dihedrals*/
       IF (QBACTON) THEN
          IPHI = ACTDIHE(II)
       ELSE
          IPHI = II
       ENDIF
#endif /*actphi2*/

#if KEY_PARALLEL==1 /*paraphi*/
#if KEY_PARASCAL==1 /*parfphi*/
          IF(PSICP(IPHI).NE.MYNOD) cycle loop70
#endif /* (parfphi)*/
#endif /*paraphi*/

       I=IP(IPHI)
       J=JP(IPHI)
       K=KP(IPHI)
       L=LP(IPHI)
#if KEY_GRAPE==1
       if(lfmm)then
          if((fmmcpu(i)/=1).and.(fmmcpu(j)/=1).and.(fmmcpu(k)/=1) &
               .and.(fmmcpu(l)/=1)) cycle loop70
       endif
#endif
       IC=ICP(IPHI)
       IF (IC.EQ.0) cycle loop70
       IF (CPD(IC).EQ.0) cycle loop70
       ! H Kamberaj (ScalingMD)
#if KEY_TSALLIS==1
       if (qttsall .or. qtalpha) then

          FI = IASCALE(I)
          FJ = IASCALE(J)
          FK = IASCALE(K)
          FL = IASCALE(L)
          sum1 = fi+fj+fk+fl
          tsallismd = (sum1 == 4)
!          TSALLISMD =  ( &
!               (FI .EQ. 1) .AND. (FJ .EQ. 1) .AND. &
!               (FK .EQ. 1) .AND. (FL .EQ. 1) &
!               )
          IF (qttsall .and. TSALLISMD) TSNDF=TSNDF+1

          IF(qtalpha .and. TSALLISMD) IC=IC*TSALPHA
       ENDIF
#endif 
       !
       !  F=Ri-Rj, G=Rj-Rk, H-Rl-Rk.
       FX=X(I)-X(J)
       FY=Y(I)-Y(J)
       FZ=Z(I)-Z(J)
       GX=X(J)-X(K)
       GY=Y(J)-Y(K)
       GZ=Z(J)-Z(K)
       HX=X(L)-X(K)
       HY=Y(L)-Y(K)
       HZ=Z(L)-Z(K)
       ! A=F^G, B=H^G.
       AX=FY*GZ-FZ*GY
       AY=FZ*GX-FX*GZ
       AZ=FX*GY-FY*GX
       BX=HY*GZ-HZ*GY
       BY=HZ*GX-HX*GZ
       BZ=HX*GY-HY*GX
       !
       RA2=AX*AX+AY*AY+AZ*AZ
       RB2=BX*BX+BY*BY+BZ*BZ
       RG=SQRT(GX*GX+GY*GY+GZ*GZ)
       ! Warnings have been simplified.
       IF((RA2.LE.RXMIN2).OR.(RB2.LE.RXMIN2).OR.(RG.LE.RXMIN)) THEN
          NWARN=NWARN+1
          IF((NWARN.LE.5 .AND. WRNLEV.GE.5) .OR. WRNLEV.GE.6) &
               WRITE(OUTU,20) IPHI,I,J,K,L
20        FORMAT(' EPHIFS: WARNING.  dihedral',I5,' is almost linear.'/ &
               ' derivatives may be affected for atoms:',4I5)
          cycle loop70
       ENDIF
       !
       RGR=ONE/RG
       RA2R=ONE/RA2
       RB2R=ONE/RB2
       RABR=SQRT(RA2R*RB2R)
       ! CT=cos(phi)
       CT=(AX*BX+AY*BY+AZ*BZ)*RABR
       !
       ! ST=sin(phi), Note that sin(phi).G/|G|=B^A/(|A|.|B|)
       ! which can be simplify to sin(phi)=|G|H.A/(|A|.|B|)
       ST=RG*RABR*(AX*HX+AY*HY+AZ*HZ)
       !
       !     Energy and derivative contributions.
       E=ZERO
       DF=ZERO
30     CONTINUE
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
       DDF1=ZERO
       !Calculation of cos(n*phi-phi0) and sin(n*phi-phi0).
       DO NPER=1,IPER
          DDF1=E1*CT-DF1*ST
          DF1=E1*ST+DF1*CT
          E1=DDF1
       enddo
       E1=E1*CPCOS(IC)+DF1*CPSIN(IC)
       DF1=DF1*CPCOS(IC)-DDF1*CPSIN(IC)
       DF1=-IPER*DF1
       E1=ONE+E1
       !brb...03-Jul-2004 Zero-period dihedral bugfix
       IF(IPER.EQ.0) E1=ONE
       !
       ARG=CPC(IC)
       E=E+ARG*E1
       DF=DF+ARG*DF1
       !
       IF(LREP) THEN
          IC=IC+1
          GOTO 30
       ENDIF
       !
#if KEY_BLOCK==1
       IF (QBLOCK) THEN
!ldm
          if (qmld) then
             call msld_scale_enerforce(mlddihc(iphi),mlddihi(iphi),mlddihj(iphi),&
                  blcoep(mlddihc(iphi)),df,e)
          else
! LDM
             IBL = IBLCKP(I)
             JBL = IBLCKP(J)
             KKK = IBLCKP(K)
             LLL = IBLCKP(L)
#if KEY_DOCK==1
             !         three pairs  (i,j), (k,j), (k,l)
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
             IF (IBL.EQ.JBL) JBL=KKK
             IF (IBL.EQ.JBL) JBL=LLL
             IF (JBL.LT.IBL) THEN
                KKK=JBL
                JBL=IBL
                IBL=KKK
             ENDIF
             KKK=IBL+JBL*(JBL-1)/2
             !.ab.QHYBH stuff.kkk=1(bl11,Main),kkk=2,3(bl12,bl22,Reactant),
             !.ab. kkk=4,6(bl13,bl33,Prod),kkk=5(bl23,don't see).
             !.ab. kkk=2,3,4,6: E accumulates througth DP(R/P) to avoid
             !.ab. if blocks below (E=0.+see below).
             IF (QHYBH) THEN
                IF (KKK.NE.1) THEN
                   IF (KKK.EQ.5) THEN
                      DF=ZERO
                   ELSE IF (KKK.LT.4) THEN
                      DF=FR*DF
                      DPR=DPR+E
                   ELSE
                      DF=FP*DF
                      DPP=DPP+E
                   ENDIF
                   E=ZERO
                ENDIF
             ELSE
                !.ab.
                COEF = BLCOED(KKK)
                IF (QPRNTV .AND. .NOT. QNOPH) THEN
                   IF (IBL == 1 .OR. JBL == 1 .OR. IBL == JBL) THEN
                      VBTORS(JBL) = VBTORS(JBL) + E
                   ENDIF
                ENDIF
!ldm
                IF (QLDM .or. QLMC) THEN
                   ! first row or diagonal elements exclude (1,1).
                   UNSCALE = 0.0
                   IF ((IBL == 1 .AND. JBL >= LSTRT) .or. &
                        (IBL >= LSTRT .AND. IBL == JBL)) UNSCALE = E
                ENDIF
!LDM
                IF (QNOPH) COEF = 1.0
#if KEY_DOCK==1
                IF(QDOCK) THEN
                   E=E*COEF*(DOCFI+DOCFJ+DOCFK+DOCFJ1+ &
                        DOCFK1+DOCFL)/6.0
                ELSE
#endif 
                   E=E*COEF
#if KEY_DOCK==1
                ENDIF
#endif 
                DFORG=DF           !ldm
                DF=DF*COEF
                !.ab.elseof(qhybh)
             ENDIF
             !.ab.
!ldm
             IF (QLDM .or. QLMC) THEN
                IF(.NOT.QNOPH) THEN
                   IF ((IBL == 1 .AND. JBL >= LSTRT) .or. &
                        (IBL >= LSTRT .AND. IBL == JBL)) THEN
                      FALPHA = UNSCALE
#if KEY_PERT==1
                      if(QPERT) FALPHA = FALPHA*PERTLAM  
#endif
                      LAGMUL = LAGMUL + FALPHA
                      BIFLAM(JBL) = BIFLAM(JBL) + FALPHA
                      IF (NRST == 2) BFRST(JBL) = BFRST(JBL) + FALPHA
                   ENDIF
                ENDIF
             ENDIF
          endif
       ENDIF
#endif /*  BLOCK*/
       !
       !
       !     Cumulate the energy
       ! H Kamberaj 2007 (Tsallis MD)
#if KEY_TSALLIS==1
       if (qttsall .and. TSALLISMD) then
          Ebtors(1)=Ebtors(1) + E
       else
          EP=EP + E
       endif
#else /**/
       EP=EP + E
#endif 
       !
       !     Compute derivatives wrt catesian coordinates.
       !
#if KEY_BLOCK==1
       IF (.NOT.NOFORC) THEN
#endif /*  BLOCK*/
          ! GAA=dE/dphi.|G|/A^2, GBB=dE/dphi.|G|/B^2, FG=F.G, HG=H.G
          !  FGA=dE/dphi*F.G/(|G|A^2), HGB=dE/dphi*H.G/(|G|B^2)
          FG=FX*GX+FY*GY+FZ*GZ
          HG=HX*GX+HY*GY+HZ*GZ
#if KEY_BLOCK==1
!ldm
          RA2RORG=RA2R
          RB2RORG=RB2R
 !LDM
#endif /*  BLOCK*/
          RA2R=DF*RA2R
          RB2R=DF*RB2R
          FGA=FG*RA2R*RGR
          HGB=HG*RB2R*RGR
          GAA=RA2R*RG
          GBB=RB2R*RG
          ! DFi=dE/dFi, DGi=dE/dGi, DHi=dE/dHi.
          DFX=-GAA*AX
          DFY=-GAA*AY
          DFZ=-GAA*AZ
          DGX=FGA*AX-HGB*BX
          DGY=FGA*AY-HGB*BY
          DGZ=FGA*AZ-HGB*BZ
          DHX=GBB*BX
          DHY=GBB*BY
          DHZ=GBB*BZ
          ! Distribute over Ri.
#if KEY_BLOCK==1
#if KEY_DOCK==1
          IF(QDOCK) THEN
             ! H Kamberaj (2007)
#if KEY_TSALLIS==1
             IF ( qttsall .and. TSALLISMD ) THEN

                TDX(I)=TDX(I)+DFX*DOCFI
                TDY(I)=TDY(I)+DFY*DOCFI
                TDZ(I)=TDZ(I)+DFZ*DOCFI
                TDX(J)=TDX(J)-DFX*DOCFJ+DGX*DOCFJ1
                TDY(J)=TDY(J)-DFY*DOCFJ+DGY*DOCFJ1
                TDZ(J)=TDZ(J)-DFZ*DOCFJ+DGZ*DOCFJ1
                TDX(K)=TDX(K)-DHX*DOCFK1-DGX*DOCFK
                TDY(K)=TDY(K)-DHY*DOCFK1-DGY*DOCFK
                TDZ(K)=TDZ(K)-DHZ*DOCFK1-DGZ*DOCFK
                TDX(L)=TDX(L)+DHX*DOCFL
                TDY(L)=TDY(L)+DHY*DOCFL
                TDZ(L)=TDZ(L)+DHZ*DOCFL
             ELSE
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
             ENDIF
#else /**/
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
#endif 
          ELSE
#endif 
#endif 
             ! H Kamberaj, Feb 2007 (Tsallis MD)
#if KEY_TSALLIS==1
             IF ( (qttsall) .and. TSALLISMD) THEN

                TDX(I)=TDX(I)+DFX
                TDY(I)=TDY(I)+DFY
                TDZ(I)=TDZ(I)+DFZ
                TDX(J)=TDX(J)-DFX+DGX
                TDY(J)=TDY(J)-DFY+DGY
                TDZ(J)=TDZ(J)-DFZ+DGZ
                TDX(K)=TDX(K)-DHX-DGX
                TDY(K)=TDY(K)-DHY-DGY
                TDZ(K)=TDZ(K)-DHZ-DGZ
                TDX(L)=TDX(L)+DHX
                TDY(L)=TDY(L)+DHY
                TDZ(L)=TDZ(L)+DHZ
             ELSE
                dx(i)=dx(i)+dfx
                dy(i)=dy(i)+dfy
                dz(i)=dz(i)+dfz
                dx(j)=dx(j)-dfx+dgx
                dy(j)=dy(j)-dfy+dgy
                dz(j)=dz(j)-dfz+dgz
                dx(k)=dx(k)-dhx-dgx
                dy(k)=dy(k)-dhy-dgy
                dz(k)=dz(k)-dhz-dgz
                dx(l)=dx(l)+dhx
                dy(l)=dy(l)+dhy
                dz(l)=dz(l)+dhz
             ENDIF
#else /**/
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
#endif 
#if KEY_BLOCK==1
#if KEY_DOCK==1
          ENDIF
#endif 
          !
       ENDIF
#endif /*  BLOCK*/
#if KEY_BLOCK==1 /*ldm*/
       IF (RSTP .AND. .NOT. QNOPH) THEN
          IF ((IBL == 1 .AND. JBL >= LSTRT) .or. &
               (IBL >= LSTRT .AND. IBL == JBL)) THEN
             RA2R=DFORG*RA2RORG
             RB2R=DFORG*RB2RORG
             FGA=FG*RA2R*RGR
             HGB=HG*RB2R*RGR
             GAA=RA2R*RG
             GBB=RB2R*RG
             DFX=-GAA*AX
             DFY=-GAA*AY
             DFZ=-GAA*AZ
             DGX=FGA*AX-HGB*BX
             DGY=FGA*AY-HGB*BY
             DGZ=FGA*AZ-HGB*BZ
             DHX=GBB*BX
             DHY=GBB*BY
             DHZ=GBB*BZ
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
#endif /* LDM*/
       !
    enddo loop70
    !
    IF (NWARN.GT.5 .AND. WRNLEV.GE.3) WRITE (OUTU,75) NWARN
75  FORMAT(' EPHIFS> Total of',I6,' WARNINGs issued.')
    !
    !.ab.
#if KEY_BLOCK==1
    IF (QHYBH) THEN
       EP=EP+FR*DPR+FP*DPP
       DPR=DFR*DPR
       DPP=DFP*DPP
       CALL SUMHYB(IHYBH,DPR,DPP)
    ENDIF
#endif 
    !.ab.
    RETURN
  END SUBROUTINE EPHIFS

  SUBROUTINE EIPHIFS(EIP)
    !-----------------------------------------------------------------------
    !     Fast SCALAR version improper torsion energy and force routine.
    !
    !     For improper dihedrals, the energy term is given by:
    !     E = K*(Phi-phi0)**2
    !     WHERE
    !     K IS THE FORCE CONSTANT IN kcal/mol/rad
    !     Phi0 IS THE MINIMUM IN ENERGY.
    !
    !     The parameter of the routine is:
    !     EPI: Diheral Energy returned
    !     Data come form param.f90
    !
    !     31-Aug-1991, Youngdo Won
    !
    !     New formulation by
    !           Arnaud Blondel    1994
    !
    !-----------------------------------------------------------------------
    use chm_kinds
    use dimens_fcm
    use number
    use psf
    use code
    use consta
    use coord
    use deriv
    use param
    use stream
#if KEY_BLOCK==1
    use lambdam
    use block_fcm
    use pert  !Cc New PBLOCK
#endif 
    use parallel
#if KEY_GRAPE==1
    use grape,only:lfmm
#endif
#if KEY_SPACDEC==1
    use spacdec    
#endif
#if KEY_ACTBOND==1
    use actclus_mod 
#endif
#if KEY_GENETIC==1
    use galgor_ltm     
#endif
    !
    !
    implicit none
    real(chm_real) EIP
#if KEY_BLOCK==1
    INTEGER IBL,JBL,KKK,LLL, KDOC
    real(chm_real)  COEF, DOCFI, DOCFJ, DOCFK, DOCFJ1, DOCFK1, DOCFL
    !.ab.
    real(chm_real) FR,FP,DFR,DFP,DIR,DIP
    !.ab.
!ldm
    real(chm_real) UNSCALE, FALPHA
    real(chm_real)  DFORG, RA2RORG, RB2RORG
!LDM
#endif /*  BLOCK*/
    real(chm_real) FX,FY,FZ,GX,GY,GZ,HX,HY,HZ
    real(chm_real) AX,AY,AZ,BX,BY,BZ,RA2,RB2,RG,RGR
    real(chm_real) RA2R,RB2R,RABR,CT,AP,CX,CY,CZ,ST,CA,SA
    real(chm_real) GAA,GBB,FG,HG,FGA,HGB
    real(chm_real) E,DF
    real(chm_real) DFX,DFY,DFZ,DHX,DHY,DHZ,DGX,DGY,DGZ
    INTEGER NWARN,NWARNX,IPHI,I,J,K,L,IC,IIMM
#if KEY_OPLS==1
    real(chm_real) E1,DF1,DDF1,ARG
    INTEGER IPER,NPER
    LOGICAL LREP
#endif 
    !
    real(chm_real)  RXMIN,RXMIN2
    PARAMETER (RXMIN=0.005D0,RXMIN2=0.000025D0)
    !
#if KEY_ACTBOND==1 /*actbond0*/
    INTEGER UPLIMIT
#endif /* (actbond0)*/
    integer ii,indfirst,indlast,indinc
#if KEY_GENETIC==1
    INTEGER First    
#endif


    ! ************************************************************
#if KEY_GENETIC==1
    First = 1
    If(qGA_Ener) then
       First = Int(EIP)
    Else
       LstImPhi = NIMPHI
    EndIf
#endif 
    !
#if KEY_ACTBOND==1 /*actbond0*/
    !      IF(.NOT.QBACTON) CALL BACTIVDEF
#if KEY_GENETIC==1 /*gentest0*/
    CALL WRNDIE(-5,'<EIPHIFS> ','GENETIC AND ACTBOND KEYWORDS' &
         //' ARE INCOMPATIBLE')
#endif /* (gentest0)*/
#endif /* (actbond0)*/
    EIP=ZERO
    !.ab.
#if KEY_BLOCK==1
    IF (QHYBH) THEN
       DIR=ZERO
       DIP=ZERO
       FR=(THREE*HYBHLB-FOUR)*(HYBHLB-ONE)/FOUR
       DFR=(SIX*HYBHLB-SEVEN)/FOUR
       FP=(THREE*HYBHLB+ONE)*HYBHLB/FOUR
       DFP=(SIX*HYBHLB+ONE)/FOUR
    ENDIF
#endif 
    !.ab.
    IF (NIMPHI == 0) RETURN
    NWARN=0
    NWARNX=0
    !

    indfirst=1
    indlast=nimphi
    indinc=1

#if KEY_PARALLEL==1 /*paraimphi*/
#if KEY_PARAFULL==1 /*parfimphi*/
    indfirst = MYNODP
    indlast  = NIMPHI
    indinc   = NUMNOD
#elif KEY_PARASCAL==1 /*parfimphi*/
#if KEY_GENETIC==1
    indfirst=First
    indlast=LstImPhi
#endif 
#endif /* (parfimphi)*/
#else /* (paraimphi)*/
#if KEY_GENETIC==1 /*geneimp*/
    indfirst=First
    indlast=LstImPhi
#endif /* (geneimp)*/
#endif /* (paraimphi)*/

#if KEY_ACTBOND==1 /*actimp*/ /*for active impropers*/
    indfirst=1
    indinc=1
    IF(QBACTON) THEN
       UPLIMIT = NACTIMP
    ELSE
       UPLIMIT = NIMPHI
    ENDIF
    indlast=UPLIMIT
#if KEY_PARALLEL==1 /*paractimp*/
    indinc=NUMNOD
    indfirst=MYNODP
#endif /*paractimp*/
#endif /*actimp*/

#if KEY_GRAPE==1
    if(lfmm)then
       indfirst=1
       indlast=nimphi
       indinc=1
    endif
#endif

    loop110:do ii=indfirst,indlast,indinc
       iphi=ii

#if KEY_PARALLEL==1 /*paraimphi*/
#if KEY_PARASCAL==1 /*parfimphi*/
       IF(PSICI(IPHI).NE.MYNOD) cycle loop110
#endif /* (parfimphi)*/
#endif /* (paraimphi)*/

#if KEY_ACTBOND==1 /*actimp2*/ /*for active impropers*/
       IF(QBACTON) IPHI = ACTIMPR(II)
#endif /*actimp2*/
       !
       I=IM(IPHI)
       J=JM(IPHI)
       K=KM(IPHI)
       L=LM(IPHI)
#if KEY_GRAPE==1
       if(lfmm)then
          if((fmmcpu(i)/=1).and.(fmmcpu(j)/=1).and.(fmmcpu(k)/=1) &
               .and.(fmmcpu(l)/=1)) cycle loop110
       endif
#endif
       IC=ICI(IPHI)
       IF (IC.EQ.0) cycle loop110
#if KEY_OPLS==0
       IF (CID(IC).NE.0) THEN
          CALL WRNDIE(-3,'<EIPHIFS>', &
               'Bad periodicity value for improper dihedral angles.')
          cycle loop110
       ENDIF
#endif 
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
       ! A=F^G, B=H^G.
       AX=FY*GZ-FZ*GY
       AY=FZ*GX-FX*GZ
       AZ=FX*GY-FY*GX
       BX=HY*GZ-HZ*GY
       BY=HZ*GX-HX*GZ
       BZ=HX*GY-HY*GX
       !
       RA2=AX*AX+AY*AY+AZ*AZ
       RB2=BX*BX+BY*BY+BZ*BZ
       RG=SQRT(GX*GX+GY*GY+GZ*GZ)
       ! Warnings have been simplified.
       IF((RA2.LE.RXMIN2).OR.(RB2.LE.RXMIN2).OR.(RG.LE.RXMIN)) THEN
          NWARN=NWARN+1
          IF((NWARN.LE.5 .AND. WRNLEV.GE.5) .OR. WRNLEV.GE.6) &
               WRITE(OUTU,20) IPHI,I,J,K,L
20        FORMAT(' EIPHIFS: WARNING.  dihedral',I5,' is almost linear.'/ &
               ' derivatives may be affected for atoms:',4I5)
          cycle loop110
       ENDIF
       !
       RGR=ONE/RG
       RA2R=ONE/RA2
       RB2R=ONE/RB2
       RABR=SQRT(RA2R*RB2R)
       ! CT=cos(phi)
       CT=(AX*BX+AY*BY+AZ*BZ)*RABR
       !
       ! ST=sin(phi), Note that sin(phi).G/|G|=B^A/(|A|.|B|)
       ! which can be simplify to sin(phi)=|G|H.A/(|A|.|B|)
       ST=RG*RABR*(AX*HX+AY*HY+AZ*HZ)
#if KEY_OPLS==1
       IF (CID(IC).NE.0) THEN

          E=ZERO
          DF=ZERO
35        CONTINUE
          IPER=CID(IC)
          IF (IPER.GE.0) THEN
             LREP=.FALSE.
          ELSE
             LREP=.TRUE.
             IPER=-IPER
          ENDIF
          !
          E1=ONE
          DF1=ZERO
          !Calculation of cos(n*phi-phi0) and sin(n*phi-phi0).
          DO NPER=1,IPER
             DDF1=E1*CT-DF1*ST
             DF1=E1*ST+DF1*CT
             E1=DDF1
          enddo
          E1=E1*CICOS(IC)+DF1*CISIN(IC)
          DF1=DF1*CICOS(IC)-DDF1*CISIN(IC)
          DF1=-IPER*DF1
          E1=ONE+E1
          !
          ARG=CIC(IC)
          E=E+ARG*E1
          DF=DF+ARG*DF1
          !
          IF(LREP) THEN
             IC=IC+1
             GOTO 35
          ENDIF
          !

          ! Use harmonic potential
          !
       ELSE
          !
#endif 
          !
          !Calcul of cos(phi-phi0),sin(phi-phi0) and (Phi-Phi0).
          CA=CT*CICOS(IC)+ST*CISIN(IC)
          SA=ST*CICOS(IC)-CT*CISIN(IC)
          IF (CA.GT.PTONE ) THEN
             AP=ASIN(SA)
          ELSE
             AP=SIGN(ACOS(MAX(CA,MINONE)),SA)
             ! Warning is now triggered at deltaphi=84.26...deg (used to be 90).
             NWARNX=NWARNX+1
             IF((NWARNX.LE.5 .AND. WRNLEV.GE.5) .OR. WRNLEV.GE.6) THEN
                WRITE(OUTU,80) IPHI,AP*RADDEG,CIB(IC)*RADDEG,I,J,K,L
80              FORMAT(' EPHI: WARNING. bent improper torsion angle',  &
                     ' is '//'far ','from minimum for;'/3X,' IPHI=',I5, &
                     '  with deltaPHI=',F9.4,' MIN=',F9.4,' ATOMS:',4I5)
             ENDIF
          ENDIF
          !
          DF=CIC(IC)*AP
          E=DF*AP
          DF=TWO*DF
#if KEY_OPLS==1
       ENDIF   
#endif

       !
#if KEY_BLOCK==1
       IF (QBLOCK) THEN
          IBL = IBLCKP(I)
          JBL = IBLCKP(J)
          KKK = IBLCKP(K)
          LLL = IBLCKP(L)
#if KEY_DOCK==1
          !         three pairs  (i,j), (k,j), (k,l)
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
          IF (IBL.EQ.JBL) JBL=KKK
          IF (IBL.EQ.JBL) JBL=LLL
          IF (JBL.LT.IBL) THEN
             KKK=JBL
             JBL=IBL
             IBL=KKK
          ENDIF
          KKK=IBL+JBL*(JBL-1)/2
          !.ab.QHYBH stuff.comments in EPHIFS.
          IF (QHYBH) THEN
             IF (KKK.NE.1) THEN
                IF (KKK.EQ.5) THEN
                   DF=ZERO
                ELSE IF (KKK.LT.4) THEN
                   DF=FR*DF
                   DIR=DIR+E
                ELSE
                   DF=FP*DF
                   DIP=DIP+E
                ENDIF
                E=ZERO
             ENDIF
          ELSE
             !.ab.
             COEF = BLCOED(KKK)
             IF (QPRNTV .AND. .NOT. QNOIM) THEN
                IF (IBL == 1 .OR. JBL == 1 .OR. IBL == JBL) THEN
                   VBIMPR(JBL) = VBIMPR(JBL) + E
                ENDIF
             ENDIF
!ldm
             IF (QLDM .or. QLMC) THEN
                ! first row or diagonal elements exclude (1,1).
                UNSCALE = 0.0
                IF ((IBL == 1 .AND. JBL >= LSTRT) .or. &
                     (IBL >= LSTRT .AND. IBL == JBL)) UNSCALE = E
             ENDIF
! LDM
             IF (QNOIM) COEF = 1.0
#if KEY_DOCK==1
             IF(QDOCK) THEN
                E=E*COEF*(DOCFI+DOCFJ+DOCFK+DOCFJ1+ &
                     DOCFK1+DOCFL)/6.0
             ELSE
#endif 
                E=E*COEF
#if KEY_DOCK==1
             ENDIF
#endif 
             DFORG=DF     !ldm
             DF=DF*COEF
             !.ab.elseof(qhybh)
          ENDIF
          !.ab.
       ENDIF
!ldm
       IF (QLDM .or. QLMC) THEN
          IF ((IBL == 1 .AND. JBL >= LSTRT) .or. &
               (IBL >= LSTRT .AND. IBL == JBL)) THEN
             FALPHA = UNSCALE
#if KEY_PERT==1
             if(QPERT) FALPHA = FALPHA*PERTLAM  
#endif
             LAGMUL = LAGMUL + FALPHA
             BIFLAM(JBL) = BIFLAM(JBL) + FALPHA
             IF (NRST == 2) BFRST(JBL) = BFRST(JBL) + FALPHA
          ENDIF
       ENDIF
! LDM
#endif /*  BLOCK*/
       !
       EIP=EIP+E
       !
       !     Compute derivatives wrt catesian coordinates.
       !
#if KEY_BLOCK==1
       IF (.NOT. NOFORC) THEN
#endif /*  BLOCK*/
          ! GAA=dE/dphi.|G|/A^2, GBB=dE/dphi.|G|/B^2, FG=F.G, HG=H.G
          !  FGA=dE/dphi*F.G/(|G|A^2), HGB=dE/dphi*H.G/(|G|B^2)
          FG=FX*GX+FY*GY+FZ*GZ
          HG=HX*GX+HY*GY+HZ*GZ
#if KEY_BLOCK==1 /*ldm*/
          RA2RORG=RA2R
          RB2RORG=RB2R
#endif /* ldm*/
          RA2R=DF*RA2R
          RB2R=DF*RB2R
          FGA=FG*RA2R*RGR
          HGB=HG*RB2R*RGR
          GAA=RA2R*RG
          GBB=RB2R*RG
          ! DFi=dE/dFi, DGi=dE/dGi, DHi=dE/dHi.
          DFX=-GAA*AX
          DFY=-GAA*AY
          DFZ=-GAA*AZ
          DGX=FGA*AX-HGB*BX
          DGY=FGA*AY-HGB*BY
          DGZ=FGA*AZ-HGB*BZ
          DHX=GBB*BX
          DHY=GBB*BY
          DHZ=GBB*BZ
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
          !
!ldm
          IF (RSTP .AND. .NOT. QNOIM) THEN
             IF ((IBL == 1 .AND. JBL >= LSTRT) .or. &
                  (IBL >= LSTRT .AND. IBL == JBL)) THEN
                RA2R=DFORG*RA2RORG
                RB2R=DFORG*RB2RORG
                FGA=FG*RA2R*RGR
                HGB=HG*RB2R*RGR
                GAA=RA2R*RG
                GBB=RB2R*RG
                DFX=-GAA*AX
                DFY=-GAA*AY
                DFZ=-GAA*AZ
                DGX=FGA*AX-HGB*BX
                DGY=FGA*AY-HGB*BY
                DGZ=FGA*AZ-HGB*BZ
                DHX=GBB*BX
                DHY=GBB*BY
                DHZ=GBB*BZ
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
!LDM
       ENDIF
#endif /*  BLOCK*/
       !
    enddo loop110
    !
    NWARN=NWARN+NWARNX
    IF (NWARN.GT.5 .AND. WRNLEV.GE.3) WRITE (OUTU,175) NWARN
175 FORMAT(' EIPHIFS> Total of',I6,' WARNINGs issued.')
    !
    !.ab.
#if KEY_BLOCK==1
    IF (QHYBH) THEN
       EIP=EIP+FR*DIR+FP*DIP
       DIR=DFR*DIR
       DIP=DFP*DIP
       CALL SUMHYB(IHYBH,DIR,DIP)
    ENDIF
#endif 
    !.ab.
    RETURN
  END SUBROUTINE EIPHIFS
  !
  SUBROUTINE BACTIV(COMLYN,COMLEN)
    use chm_kinds
    use dimens_fcm
    use psf
    use stream
    use coord
    use select
    !  parses BACT command and sets up call to MKBACTIVE,
    !  which creates the active bond, angle, dihe, and improper arrays
    ! . Passed variables.
    implicit none
    CHARACTER COMLYN*(*)
    INTEGER   COMLEN
    !
    INTEGER II
    INTEGER FLACT(MAXA) !local

    FLACT(1:natom) = 0
    ! -----------------------------------------------
    CALL SELCTA(COMLYN,COMLEN,FLACT,X,Y,Z,WMAIN,.TRUE.)
    !
    CALL MKBACTIVE(FLACT)
    RETURN
  END SUBROUTINE BACTIV
  !
  !
  SUBROUTINE BACTIVDEF
    use chm_kinds
    use dimens_fcm
    use psf
    !  sets up default active bond, angle, dihe and improper arrays
    !
    implicit none
    INTEGER II
    INTEGER FLACT(MAXA) !local

    FLACT(1:natom) = 1
    CALL MKBACTIVE(FLACT)
    RETURN
  END SUBROUTINE BACTIVDEF
  !
  !
  SUBROUTINE MKBACTIVE(FLACT)
    use chm_kinds
    use dimens_fcm
    use psf
    use actclus_mod
    use stream
    use memory
#if KEY_PARALLEL==1
    use parallel,only: MYNODP
#endif
    ! sets up active atom arrays for bonded energies
    !
    implicit none
    INTEGER FLACT(*) !every atom in system should be flagged 1 or 0
    INTEGER  ATM,MM,ITH,IPHI,IIMP
    INTEGER PROD
    !
    !these arrays are allocated and deallocated together
    if(allocated(ACTBOND)) then
       call chmdealloc('enefscal','MKBACTIV','ACTBOND',NBOND,intg=ACTBOND)
       call chmdealloc('enefscal','MKBACTIV','ACTANGL',NTHETA,intg=ACTANGL)
       call chmdealloc('enefscal','MKBACTIV','ACTDIHE',NPHI,intg=ACTDIHE)
       call chmdealloc('enefscal','MKBACTIV','ACTIMPR',NIMPHI,intg=ACTIMPR)
    endif

    call chmalloc('enefscal','MKBACTIV','ACTBOND',NBOND,intg=ACTBOND)
    call chmalloc('enefscal','MKBACTIV','ACTANGL',NTHETA,intg=ACTANGL)
    call chmalloc('enefscal','MKBACTIV','ACTDIHE',NPHI,intg=ACTDIHE)
    call chmalloc('enefscal','MKBACTIV','ACTIMPR',NIMPHI,intg=ACTIMPR)

    NACTBND = 0
    NACTANG = 0
    NACTDIH = 0
    NACTIMP = 0
    DO MM = 1,NBOND
       PROD = FLACT(IB(MM))*FLACT(JB(MM))
       IF(PROD.EQ.1) THEN
          NACTBND = NACTBND + 1
          IF(NACTBND.GT.NBOND) THEN
             WRITE(6,*) 'NACTBND ',NACTBND,' NBOND ',NBOND
             CALL WRNDIE(-3,'<MKBACTIVE>', &
                  'TOO MANY ACTIVE BONDS')
          ENDIF
          ACTBOND(NACTBND) = MM
       ENDIF
    ENDDO

    DO ITH = 1,NTHETA
       PROD =FLACT(IT(ITH))*FLACT(JT(ITH))*FLACT(KT(ITH))
       IF(PROD.EQ.1) THEN
          NACTANG = NACTANG + 1
          IF(NACTANG.GT.NTHETA) CALL WRNDIE(-3,'<MKBACTIVE>', &
               'TOO MANY ACTIVE ANGLES')
          ACTANGL(NACTANG) = ITH
       ENDIF
    ENDDO
    !       WRITE(OUTU,'(I8,A14)') NACTANG,' ACTIVE ANGLES '
    DO IPHI = 1,NPHI
       PROD =FLACT(IP(IPHI))*FLACT(JP(IPHI))* &
            FLACT(KP(IPHI))*FLACT(LP(IPHI))
       IF(PROD.EQ.1) THEN
          NACTDIH = NACTDIH + 1
          IF(NACTDIH.GT.NPHI) CALL WRNDIE(-3,'<MKBACTIVE>', &
               'TOO MANY ACTIVE DIHEDRALS')
          ACTDIHE(NACTDIH) = IPHI
       ENDIF
    ENDDO
    !       WRITE(OUTU,'(I8,A14)') NACTDIH,' ACTIVE DIHEDRALS '
    DO IIMP = 1,NIMPHI
       PROD = FLACT(IM(IIMP))*FLACT(JM(IIMP))* &
            FLACT(KM(IIMP))*FLACT(LM(IIMP))
       IF(PROD.EQ.1) THEN
          NACTIMP = NACTIMP + 1
          IF(NACTIMP.GT.NIMPHI) CALL WRNDIE(-3,'<MKBACTIVE>', &
               'TOO MANY ACTIVE IMPROPERS')
          ACTIMPR(NACTIMP) = IIMP
       ENDIF
    ENDDO
    !       WRITE(OUTU,'(I8,A14)') NACTIMP,' ACTIVE IMPROPERS '
    QBACTON = .TRUE.
#if KEY_PARALLEL==1
    if(MYNODP.eq.1) then
#endif
1010 FORMAT(3X,A,1X,I10,1X,A,I10,1X,A,1X,I10,1X,A,1X,I10)
    WRITE(OUTU,1010) &
         'TOTAL BONDS=',NBOND,'ANGS=', &
         NTHETA,'DIHEDRS=',NPHI,'IMPROPS=',NIMPHI
    WRITE(OUTU,1010) &
         'ACTIV BONDS=',NACTBND,'ANGS=', &
         NACTANG,'DIHEDRS=',NACTDIH,'IMPROPS=',NACTIMP
#if KEY_PARALLEL==1
    endif
#endif
    !
    RETURN
  END SUBROUTINE MKBACTIVE
  !--- ##ENDIF ACTBOND

  !  !-----------------------------------------------------------------------!
  !  ! ABOUT FASTER ROUTINES
  !  ! This module contains stripped down, fast versions of the bond,
  !  ! angle, dihedral and impropers. It is intended for routines where
  !  ! extra features are neither required nor desired. Please do not add
  !  ! your special features to this module without the consensus of the
  !  ! CHARMM development community, as it may drastically effect the speed
  !  ! of CHARMM for all users.
  !  ! Routines converted so far:
  !  !  EBONDFASTER
  !  !  EANGLFASTER
  !  !  EANGLFASTERGROM
  !  !  EPHIFASTER
  !  !  EIPHIFASTER
  !  ! The following have been stripped, and thus must be ruled out in 
  !  ! ecntrl.src:
  !  !  ACTBOND
  !  !  BLOCK
  !  !  DOCK
  !  !  GENETIC
  !  !  LDM
  !  !-----------------------------------------------------------------------!

  SUBROUTINE EBONDFASTER(EB,NBOND,IB,JB,ICB,CBB,CBC)
    !-----------------------------------------------------------------------
    !     calculates bond energies and forces as fast as possible
    !     Fast SCALAR version
    !     14-JUL-1983, Bernard R. Brooks
    !     31-Aug-1991, Youngdo Won
    !     01-Feb-2011, Michael G. Lerner, Tim Miller, Bernard R. Brooks
    !-----------------------------------------------------------------------
    use chm_kinds
    use dimens_fcm
    use number
    use deriv
    use coord
    use stream
    use parallel
    use spacdec
    implicit none
#ifdef _OPENMP
    integer omp_get_thread_num 
#endif
    real(chm_real) EB
    INTEGER NBOND
    INTEGER IB(*),JB(*),ICB(*)
    real(chm_real) CBB(:),CBC(:)
    !
    real(chm_real) RX,RY,RZ,S,R,DB,DF,DXI,DYI,DZI
    INTEGER MM,I,J,IC,IIMM
    integer ii,indfirst,indlast,indinc
    !
    !     WRITE(OUTU,'(A)') 'in EBONDFASTER'
    !
    !
    EB=ZERO
    IF (NBOND == 0) RETURN
    !
    indfirst=1
    indlast=nbond
    indinc=1
#if KEY_PARALLEL==1 /*parabond*/
#if KEY_PARAFULL==1 /*parfbond*/
    indfirst = MYNODP
    indlast  = NBOND
    indinc   = NUMNOD
#elif KEY_PARASCAL==1 /*parfbond*/
#elif KEY_SPACDEC==1 /*parfbond*/
    ! the above default is OK here
#else /* (parfbond)*/
#error  'Illegal parallel compile option'
#endif /* (parfbond)*/

#else /* (parabond)*/

#endif /* (parabond)*/

    do ii=indfirst,indlast,indinc
       mm=ii
#if KEY_PARALLEL==1 /*parabond*/
#if KEY_PARAFULL==1 /*parfbond*/
#elif KEY_SPACDEC==1 /*parfbond*/
       IF(ICPUMAP(IB(MM)) /= MYNOD) cycle
       !...##ELIF SPACDEC (parfbond)  ! This is in the future
       !      DO 60 IIMM=1,NMYBOND
       !         MM=MYBOND(IIMM)
#endif /* (parfbond)*/
#else /* (parabond)*/
#endif /* (parabond)*/
       !
       I=IB(MM)
       J=JB(MM)
       IF (I.LE.0) cycle
       IC=ICB(MM)
       IF (IC.EQ.0) cycle
       IF (CBC(IC).EQ.0.0) cycle
       RX=X(I)-X(J)
       RY=Y(I)-Y(J)
       RZ=Z(I)-Z(J)
       S=SQRT(RX*RX+RY*RY+RZ*RZ)
       IF (S.EQ.0.0) cycle
       R=2.0/S
       DB=S-CBB(IC)
       DF=CBC(IC)*DB
       EB=EB+DF*DB
       !
       DF=DF*R
       DXI=RX*DF
       DYI=RY*DF
       DZI=RZ*DF
       DX(I)=DX(I)+DXI
       DY(I)=DY(I)+DYI
       DZ(I)=DZ(I)+DZI
       DX(J)=DX(J)-DXI
       DY(J)=DY(J)-DYI
       DZ(J)=DZ(J)-DZI
       !
    enddo

    RETURN
  END SUBROUTINE EBONDFASTER

  SUBROUTINE EANGLFASTER(ET)
    !-----------------------------------------------------------------------
    !     calculates bond angles and bond angle energies as fast as possible
    !     Fast SCALAR version
    !     14-JUL-1983, Bernard R. Brooks
    !     31-Aug-1991, Youngdo Won
    !     01-Feb-2011, Michael G. Lerner, Tim Miller, Bernard R. Brooks
    !-----------------------------------------------------------------------
    use chm_kinds
    use block_fcm
    use dimens_fcm
    use number
    use psf
    use deriv
    use coord
    use code
    use param
    use stream
    use consta
    !
    use parallel
#if KEY_SPACDEC==1
    use spacdec  
#endif
    implicit none
    !
    real(chm_real) ET
    !
    real(chm_real) DXI,DYI,DZI,DXJ,DYJ,DZJ,SMALLV,ST2R,STR
    real(chm_real) RI,RJ,RIR,RJR,DXIR,DYIR,DZIR,DXJR,DYJR,DZJR
    real(chm_real) CST,AT,DA,DF,DTXI,DTXJ,DTYI,DTYJ,DTZI,DTZJ,DTI,DTJ
    INTEGER NWARN,ITH,I,J,K,IC,IIMM
    integer ii,indfirst,indlast,indinc


    ! *********************************************************

    ET=ZERO
    SMALLV=RPRECI
    NWARN=0
    IF (NTHETA == 0) RETURN
    !
    indfirst=1
    indlast=ntheta
    indinc=1

#if KEY_PARALLEL==1 /*paraangle*/
#if KEY_PARAFULL==1 /*parfangle*/
    indfirst=MYNODP
    indlast=NTHETA
    indinc=NUMNOD
#elif KEY_PARASCAL==1 /*parfangle*/
#endif /* (parfangle)*/

#else /* (paraangle)*/

#endif /* (paraangle)*/

    do ii=indfirst,indlast,indinc
       ith=ii
#if KEY_PARALLEL==1 /*paraangle*/
#if KEY_PARASCAL==1 /*parfangle*/
       IF(PSICT(ITH).NE.MYNOD) cycle 
#elif KEY_SPACDEC==1 /*parfangle*/
       IF(MYNOD /= ICPUMAP(IT(ITH))) cycle 
       !...##ELIF SPACDEC (parfangle)  ! future...
       !      DO 50 IIMM=1,NMYANGL
       !         ITH=MYANGL(IIMM)
#endif /* (parfangle)*/
#else /* (paraangle)*/
#endif /* (paraangle)*/

       i=it(ith)
       j=jt(ith)
       k=kt(ith)
       if (i.le.0) cycle 
       ic=ict(ith)
       if (ic.eq.0) cycle 
       dxi=x(i)-x(j)
       dyi=y(i)-y(j)
       dzi=z(i)-z(j)
       dxj=x(k)-x(j)
       dyj=y(k)-y(j)
       dzj=z(k)-z(j)
       ri=sqrt(dxi*dxi+dyi*dyi+dzi*dzi)
       if (ri.eq.0.0) cycle 
       rj=sqrt(dxj*dxj+dyj*dyj+dzj*dzj)
       if (rj.eq.0.0) cycle 
       rir=1.0/ri
       rjr=1.0/rj
       dxir=dxi*rir
       dyir=dyi*rir
       dzir=dzi*rir
       dxjr=dxj*rjr
       dyjr=dyj*rjr
       dzjr=dzj*rjr
       cst=dxir*dxjr+dyir*dyjr+dzir*dzjr
       !
       IF(ABS(CST).GE.0.999999) THEN
          IF(ABS(CST).GT.ONE) CST=SIGN(ONE,CST)
          AT=ACOS(CST)
          DA=AT-CTB(IC)
          DF=CTC(IC)*DA
          !     
          ET= ET+DF*DA

          IF(ABS(DA).GT.0.1) THEN
             NWARN=NWARN+1
             IF((NWARN.LE.5 .AND. WRNLEV.GE.5) .OR. WRNLEV.GE.6) THEN
                WRITE(OUTU,10) ITH,I,J,K
10              FORMAT(' WARNING FROM EANGLFASTER. Angle',I5, &
                     '  is almost linear.', &
                     /' Derivatives may be affected for atoms:',3I5)
                WRITE(OUTU,101) 'I ATOM:',X(I),Y(I),Z(I)
                WRITE(OUTU,101) 'J ATOM:',X(J),Y(J),Z(J)
                WRITE(OUTU,101) 'K ATOM:',X(K),Y(K),Z(K)
                WRITE(OUTU,101) 'DXIR  :',DXIR,DYIR,DZIR
                WRITE(OUTU,101) 'DXJR  :',DXJR,DYJR,DZJR
                WRITE(OUTU,101) 'CST   :',CST,AT*RADDEG,DA*RADDEG
101             FORMAT(5X,A,5F15.5)
             ENDIF
          ENDIF
          ST2R=ONE/(ONE-CST*CST+SMALLV)
          STR=SQRT(ST2R)
          IF(CTB(IC).LT.PT001) THEN
             DF=MINTWO*CTC(IC)*(ONE+DA*DA*SIXTH)
          ELSE IF(PI-CTB(IC).LT.PT001) THEN
             DF=TWO*CTC(IC)*(ONE+DA*DA*SIXTH)
          ELSE
             DF=MINTWO*DF*STR
          ENDIF


       ELSE
          AT=ACOS(CST)
          DA=AT-CTB(IC)
          DF=CTC(IC)*DA
          ET= ET+DF*DA

          ST2R=ONE/(ONE-CST*CST)
          STR=SQRT(ST2R)
          DF=MINTWO*DF*STR
       ENDIF
       !
       DTXI=RIR*(DXJR-CST*DXIR)
       DTXJ=RJR*(DXIR-CST*DXJR)
       DTYI=RIR*(DYJR-CST*DYIR)
       DTYJ=RJR*(DYIR-CST*DYJR)
       DTZI=RIR*(DZJR-CST*DZIR)
       DTZJ=RJR*(DZIR-CST*DZJR)
       !
       DTI=DF*DTXI
       DTJ=DF*DTXJ
       DX(I)=DX(I)+DTI
       DX(K)=DX(K)+DTJ
       DX(J)=DX(J)-DTI-DTJ
       !
       DTI=DF*DTYI
       DTJ=DF*DTYJ
       DY(I)=DY(I)+DTI
       DY(K)=DY(K)+DTJ
       DY(J)=DY(J)-DTI-DTJ
       !
       DTI=DF*DTZI
       DTJ=DF*DTZJ
       DZ(I)=DZ(I)+DTI
       DZ(K)=DZ(K)+DTJ
       DZ(J)=DZ(J)-DTI-DTJ
    enddo

    !
    IF(NWARN.GT.5 .AND. WRNLEV.GE.3) WRITE(OUTU,455) NWARN
455 FORMAT(' EANGLFASTER> Total of',I6,' warnings.')
    !
    RETURN
  END SUBROUTINE EANGLFASTER

  SUBROUTINE EANGLFASTERGROM(ET)
    !-----------------------------------------------------------------------
    !     calculates bond angles and bond angle energies as fast as possible
    !     Fast SCALAR version
    !     14-JUL-1983, Bernard R. Brooks
    !     31-Aug-1991, Youngdo Won
    !     01-Feb-2011, Michael G. Lerner, Tim Miller, Bernard R. Brooks
    !-----------------------------------------------------------------------
    use chm_kinds
    use block_fcm
    use dimens_fcm
    use number
    use psf
    use deriv
    use coord
    use code
    use param
    use stream
    use consta
    !
    use parallel
#if KEY_SPACDEC==1
    use spacdec  
#endif
    implicit none
    !
    real(chm_real) ET
    !
    real(chm_real) DXI,DYI,DZI,DXJ,DYJ,DZJ,SMALLV,ST2R,STR
    real(chm_real) RI,RJ,RIR,RJR,DXIR,DYIR,DZIR,DXJR,DYJR,DZJR
    real(chm_real) CST,AT,DA,DF,DTXI,DTXJ,DTYI,DTYJ,DTZI,DTZJ,DTI,DTJ
    INTEGER NWARN,ITH,I,J,K,IC,IIMM
    integer ii,indfirst,indlast,indinc



    ! *********************************************************

    ET=ZERO
    SMALLV=RPRECI
    NWARN=0
    IF (NTHETA == 0) RETURN
    !
    indfirst=1
    indlast=ntheta
    indinc=1

#if KEY_PARALLEL==1 /*paraangle*/
#if KEY_PARAFULL==1 /*parfangle*/
    indfirst=MYNODP
    indlast=NTHETA
    indinc=NUMNOD
#elif KEY_PARASCAL==1 /*parfangle*/
#endif /* (parfangle)*/

#else /* (paraangle)*/

#endif /* (paraangle)*/

    loop50: do ii=indfirst,indlast,indinc
       ith=ii

#if KEY_PARALLEL==1 /*paraangle*/
#if KEY_PARASCAL==1 /*parfangle*/
       IF(PSICT(ITH).NE.MYNOD) cycle loop50
#elif KEY_SPACDEC==1 /*parfangle*/
       IF(MYNOD /= ICPUMAP(IT(ITH))) cycle loop50
       !...##ELIF SPACDEC (parfangle)  ! future...
       !      DO 50 IIMM=1,NMYANGL
       !         ITH=MYANGL(IIMM)
#endif /* (parfangle)*/
#else /* (paraangle)*/
#endif /* (paraangle)*/

       i=it(ith)
       j=jt(ith)
       k=kt(ith)
       if (i.le.0) cycle loop50
       ic=ict(ith)
       if (ic.eq.0) cycle loop50
       dxi=x(i)-x(j)
       dyi=y(i)-y(j)
       dzi=z(i)-z(j)
       dxj=x(k)-x(j)
       dyj=y(k)-y(j)
       dzj=z(k)-z(j)
       ri=sqrt(dxi*dxi+dyi*dyi+dzi*dzi)
       if (ri.eq.0.0) cycle loop50
       rj=sqrt(dxj*dxj+dyj*dyj+dzj*dzj)
       if (rj.eq.0.0) cycle loop50
       rir=1.0/ri
       rjr=1.0/rj
       dxir=dxi*rir
       dyir=dyi*rir
       dzir=dzi*rir
       dxjr=dxj*rjr
       dyjr=dyj*rjr
       dzjr=dzj*rjr
       cst=dxir*dxjr+dyir*dyjr+dzir*dzjr

       ! ! = cos(theta)-cos(theta_not)
       DA=CST-(CTB(IC)+ONE)
       DF=CTC(IC)*DA
       ET=ET+DF*DA/2.0
       !ET=ET+CTC(IC)*DA*DA/2.0
       !     
       !
       DTXI=RIR*(DXJR-CST*DXIR)
       DTXJ=RJR*(DXIR-CST*DXJR)
       DTYI=RIR*(DYJR-CST*DYIR)
       DTYJ=RJR*(DYIR-CST*DYJR)
       DTZI=RIR*(DZJR-CST*DZIR)
       DTZJ=RJR*(DZIR-CST*DZJR)
       !
       DTI=DF*DTXI
       DTJ=DF*DTXJ
       DX(I)=DX(I)+DTI
       DX(K)=DX(K)+DTJ
       DX(J)=DX(J)-DTI-DTJ
       !
       DTI=DF*DTYI
       DTJ=DF*DTYJ
       DY(I)=DY(I)+DTI
       DY(K)=DY(K)+DTJ
       DY(J)=DY(J)-DTI-DTJ
       !
       DTI=DF*DTZI
       DTJ=DF*DTZJ
       DZ(I)=DZ(I)+DTI
       DZ(K)=DZ(K)+DTJ
       DZ(J)=DZ(J)-DTI-DTJ
    enddo loop50
    !
    IF(NWARN.GT.5 .AND. WRNLEV.GE.3) WRITE(OUTU,455) NWARN
455 FORMAT(' EANGLFASTERGROM> Total of',I6,' warnings.')
    !
    RETURN
  END SUBROUTINE EANGLFASTERGROM

  SUBROUTINE EPHIFASTER(EP)
    !-----------------------------------------------------------------------
    !     Fast SCALAR version dihedral energy and force routine.
    !     Dihedral energy terms are expressed as a function of PHI.
    !     This avoids all problems as dihedrals become planar.
    !     The functional form is:
    !
    !     E = K*(1+COS(n*Phi-Phi0))
    !     Where:
    !     n IS A POSITIVE INTEGER COS(n(Phi-Phi0) AND SIN(n(Phi-Phi0)
    !            ARE CALCULATED BY RECURENCE TO AVOID LIMITATION ON n
    !     K IS THE FORCE CONSTANT IN kcal/mol/rad
    !     Phi0/n IS A MAXIMUM IN ENERGY.
    !
    !     The parameter of the routine is:
    !     EP: Diheral Energy returned
    !     Data come form param.f90
    !
    !     31-Aug-1991, Youngdo Won
    !
    !     New formulation by
    !           Arnaud Blondel    1994
    !
    !    MODIFIED BY HIQMET KAMBERAJ
    !    ADDED: TSALLIS MD ON TORSIONS ANGLES
    !    2007
    !
    !     01-Feb-2011, Michael G. Lerner, Tim Miller, Bernard R. Brooks
    !
    !-----------------------------------------------------------------------
    use chm_kinds
    use dimens_fcm
    use number
    use code
    use consta
    use coord
    use deriv
    use param
    use psf
    use stream
    use parallel
#if KEY_SPACDEC==1
    use spacdec  
#endif
    implicit none
    real(chm_real) EP
    !

    !
    real(chm_real)  FX,FY,FZ,GX,GY,GZ,HX,HY,HZ
    real(chm_real)  AX,AY,AZ,BX,BY,BZ,RA2,RB2,RG,RGR
    real(chm_real)  RA2R,RB2R,RABR,CT,CX,CY,CZ,ST
    real(chm_real)  GAA,GBB,FG,HG,FGA,HGB
    real(chm_real)  E,DF,E1,DF1,DDF1,ARG
    real(chm_real)  DFX,DFY,DFZ,DHX,DHY,DHZ,DGX,DGY,DGZ
    INTEGER NWARN,IPHI,I,J,K,L,IC,IPER,NPER,IIMM
    LOGICAL LREP
    !
    real(chm_real)  RXMIN,RXMIN2
    PARAMETER (RXMIN=0.005,RXMIN2=0.000025D0)


    integer ii,indfirst,indlast,indinc



    !
    !     Initialize the torsion energy to zero
    ! ********************************************************
    !
    EP=ZERO
    !.ab.
    IF (NPHI == 0) RETURN
    NWARN=0
    !
    !

    indfirst=1
    indlast=nphi
    indinc=1

#if KEY_PARALLEL==1 /*paraphi*/
#if KEY_PARAFULL==1 /*parfphi*/
    indfirst=MYNODP
    indlast=NPHI
    indinc=NUMNOD
#elif KEY_PARASCAL==1 /*parfphi*/
#endif /* (parfphi)*/

#else /* (paraphi)*/

#endif /* (paraphi)*/

    loop70: do ii=indfirst,indlast,indinc
       iphi=ii

#if KEY_PARALLEL==1 /*paraphi*/
#if KEY_SPACDEC==1 /*parfphi*/
       IF(MYNOD /= ICPUMAP(IP(IPHI))) cycle loop70
       !...##ELIF SPACDEC (parfphi)
       !      DO 70 IIMM=1,NMYDIHE
       !         IPHI=MYDIHE(IIMM)
#elif KEY_PARASCAL==1 /*parfphi*/
       IF(PSICP(IPHI).NE.MYNOD) cycle loop70
#endif /* (parfphi)*/
#else /* (paraphi)*/
#endif /* (paraphi)*/
       !
       I=IP(IPHI)
       J=JP(IPHI)
       K=KP(IPHI)
       L=LP(IPHI)
       IC=ICP(IPHI)
       IF (IC.EQ.0) cycle loop70
       IF (CPD(IC).EQ.0) cycle loop70
       !  F=Ri-Rj, G=Rj-Rk, H-Rl-Rk.
       FX=X(I)-X(J)
       FY=Y(I)-Y(J)
       FZ=Z(I)-Z(J)
       GX=X(J)-X(K)
       GY=Y(J)-Y(K)
       GZ=Z(J)-Z(K)
       HX=X(L)-X(K)
       HY=Y(L)-Y(K)
       HZ=Z(L)-Z(K)
       ! A=F^G, B=H^G.
       AX=FY*GZ-FZ*GY
       AY=FZ*GX-FX*GZ
       AZ=FX*GY-FY*GX
       BX=HY*GZ-HZ*GY
       BY=HZ*GX-HX*GZ
       BZ=HX*GY-HY*GX
       !
       RA2=AX*AX+AY*AY+AZ*AZ
       RB2=BX*BX+BY*BY+BZ*BZ
       RG=SQRT(GX*GX+GY*GY+GZ*GZ)
       ! Warnings have been simplified.
       IF((RA2.LE.RXMIN2).OR.(RB2.LE.RXMIN2).OR.(RG.LE.RXMIN)) THEN
          NWARN=NWARN+1
          IF((NWARN.LE.5 .AND. WRNLEV.GE.5) .OR. WRNLEV.GE.6) &
               WRITE(OUTU,20) IPHI,I,J,K,L
20        FORMAT(' EPHIFASTER: WARNING.  dihedral',I5,' is almost linear.'/ &
               ' derivatives may be affected for atoms:',4I5)
          cycle loop70
       ENDIF
       !
       RGR=ONE/RG
       RA2R=ONE/RA2
       RB2R=ONE/RB2
       RABR=SQRT(RA2R*RB2R)
       ! CT=cos(phi)
       CT=(AX*BX+AY*BY+AZ*BZ)*RABR
       !
       ! ST=sin(phi), Note that sin(phi).G/|G|=B^A/(|A|.|B|)
       ! which can be simplify to sin(phi)=|G|H.A/(|A|.|B|)
       ST=RG*RABR*(AX*HX+AY*HY+AZ*HZ)
       !
       !     Energy and derivative contributions.
       E=ZERO
       DF=ZERO
30     CONTINUE
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
       DDF1=ZERO
       !Calculation of cos(n*phi-phi0) and sin(n*phi-phi0).
       DO NPER=1,IPER
          DDF1=E1*CT-DF1*ST
          DF1=E1*ST+DF1*CT
          E1=DDF1
       enddo
       E1=E1*CPCOS(IC)+DF1*CPSIN(IC)
       DF1=DF1*CPCOS(IC)-DDF1*CPSIN(IC)
       DF1=-IPER*DF1
       E1=ONE+E1
       !brb...03-Jul-2004 Zero-period dihedral bugfix
       IF(IPER.EQ.0) E1=ONE
       !
       ARG=CPC(IC)
       E=E+ARG*E1
       DF=DF+ARG*DF1
       !
       IF(LREP) THEN
          IC=IC+1
          GOTO 30
       ENDIF
       !
       !
       !
       EP=EP + E
       !
       !     Compute derivatives wrt catesian coordinates.
       !
       ! GAA=dE/dphi.|G|/A^2, GBB=dE/dphi.|G|/B^2, FG=F.G, HG=H.G
       !  FGA=dE/dphi*F.G/(|G|A^2), HGB=dE/dphi*H.G/(|G|B^2)
       FG=FX*GX+FY*GY+FZ*GZ
       HG=HX*GX+HY*GY+HZ*GZ
       RA2R=DF*RA2R
       RB2R=DF*RB2R
       FGA=FG*RA2R*RGR
       HGB=HG*RB2R*RGR
       GAA=RA2R*RG
       GBB=RB2R*RG
       ! DFi=dE/dFi, DGi=dE/dGi, DHi=dE/dHi.
       DFX=-GAA*AX
       DFY=-GAA*AY
       DFZ=-GAA*AZ
       DGX=FGA*AX-HGB*BX
       DGY=FGA*AY-HGB*BY
       DGZ=FGA*AZ-HGB*BZ
       DHX=GBB*BX
       DHY=GBB*BY
       DHZ=GBB*BZ
       ! Distribute over Ri.
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
       !
    enddo loop70

    !
    IF (NWARN.GT.5 .AND. WRNLEV.GE.3) WRITE (OUTU,75) NWARN
75  FORMAT(' EPHIFASTER> Total of',I6,' WARNINGs issued.')
    !
    !.ab.
    !.ab.
    RETURN
  END SUBROUTINE EPHIFASTER

  SUBROUTINE EIPHIFASTER(EIP)
    !-----------------------------------------------------------------------
    !     Fast SCALAR version improper torsion energy and force routine.
    !
    !     For improper dihedrals, the energy term is given by:
    !     E = K*(Phi-phi0)**2
    !     WHERE
    !     K IS THE FORCE CONSTANT IN kcal/mol/rad
    !     Phi0 IS THE MINIMUM IN ENERGY.
    !
    !     The parameter of the routine is:
    !     EPI: Diheral Energy returned
    !     Data come form param.f90
    !
    !     31-Aug-1991, Youngdo Won
    !
    !     New formulation by
    !           Arnaud Blondel    1994
    !
    !     01-Feb-2011, Michael G. Lerner, Tim Miller, Bernard R. Brooks
    !
    !-----------------------------------------------------------------------
    use chm_kinds
    use dimens_fcm
    use number
    use psf
    use actclus_mod
    use code
    use consta
    use coord
    use deriv
    use param
    use stream
    use parallel
#if KEY_SPACDEC==1
    use spacdec  
#endif
    !
    implicit none
    real(chm_real) EIP
    real(chm_real) FX,FY,FZ,GX,GY,GZ,HX,HY,HZ
    real(chm_real) AX,AY,AZ,BX,BY,BZ,RA2,RB2,RG,RGR
    real(chm_real) RA2R,RB2R,RABR,CT,AP,CX,CY,CZ,ST,CA,SA
    real(chm_real) GAA,GBB,FG,HG,FGA,HGB
    real(chm_real) E,DF
    real(chm_real) DFX,DFY,DFZ,DHX,DHY,DHZ,DGX,DGY,DGZ
    INTEGER NWARN,NWARNX,IPHI,I,J,K,L,IC,IIMM
#if KEY_OPLS==1
    real(chm_real) E1,DF1,DDF1,ARG
    INTEGER IPER,NPER
    LOGICAL LREP
#endif 
    !
    real(chm_real)  RXMIN,RXMIN2
    PARAMETER (RXMIN=0.005D0,RXMIN2=0.000025D0)
    !
    integer ii,indfirst,indlast,indinc
#if KEY_ACTBOND==1
    integer :: UPLIMIT 
#endif

    ! ************************************************************
    !
    EIP=ZERO
    !.ab.
    !.ab.
    IF (NIMPHI == 0) RETURN
    NWARN=0
    NWARNX=0
    !

    indfirst=1
    indlast=nimphi
    indinc=1

#if KEY_PARALLEL==1 /*paraimphi*/
#if KEY_PARAFULL==1 /*parfimphi*/
    indfirst = MYNODP
    indlast  = NIMPHI
    indinc   = NUMNOD
#elif KEY_PARASCAL==1 /*parfimphi*/
#endif /*parfimphi*/
#endif /*paraimphi*/

#if KEY_ACTBOND==1 /*actimpf*/ /*for active impropers*/
     indfirst=1
     indlast=1
     IF(QBACTON) THEN
        UPLIMIT = NACTIMP
     ELSE
        UPLIMIT = NIMPHI
     ENDIF
     indlast=UPLIMIT
#if KEY_PARALLEL==1
     indinc=NUMNOD
     indfirst=MYNODP
#endif
#endif /*actimpf*/

    loop110: do ii=indfirst,indlast,indinc
       iphi=ii

#if KEY_ACTBOND==1 /*actimpf2*/ /*for active impropers*/
       IF(QBACTON) IPHI = ACTIMPR(II)
#endif /*actimpf2*/

#if KEY_PARALLEL==1 /*paraimphi*/
#if KEY_PARASCAL==1 /*parfimphi*/
       IF(PSICI(IPHI).NE.MYNOD) cycle loop110
#endif /*parfimphi*/
#endif /*paraimphi*/

       !
       I=IM(IPHI)
       J=JM(IPHI)
       K=KM(IPHI)
       L=LM(IPHI)
       IC=ICI(IPHI)
       IF (IC.EQ.0) cycle loop110
#if KEY_OPLS==0
       IF (CID(IC).NE.0) THEN
          CALL WRNDIE(-3,'<EIPHIFS>', &
               'Bad periodicity value for improper dihedral angles.')
          cycle loop110
       ENDIF
#endif 
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
       ! A=F^G, B=H^G.
       AX=FY*GZ-FZ*GY
       AY=FZ*GX-FX*GZ
       AZ=FX*GY-FY*GX
       BX=HY*GZ-HZ*GY
       BY=HZ*GX-HX*GZ
       BZ=HX*GY-HY*GX
       !
       RA2=AX*AX+AY*AY+AZ*AZ
       RB2=BX*BX+BY*BY+BZ*BZ
       RG=SQRT(GX*GX+GY*GY+GZ*GZ)
       ! Warnings have been simplified.
       IF((RA2.LE.RXMIN2).OR.(RB2.LE.RXMIN2).OR.(RG.LE.RXMIN)) THEN
          NWARN=NWARN+1
          IF((NWARN.LE.5 .AND. WRNLEV.GE.5) .OR. WRNLEV.GE.6) &
               WRITE(OUTU,20) IPHI,I,J,K,L
20        FORMAT(' EIPHIFS: WARNING.  dihedral',I5,' is almost linear.'/ &
               ' derivatives may be affected for atoms:',4I5)
          cycle loop110
       ENDIF
       !
       RGR=ONE/RG
       RA2R=ONE/RA2
       RB2R=ONE/RB2
       RABR=SQRT(RA2R*RB2R)
       ! CT=cos(phi)
       CT=(AX*BX+AY*BY+AZ*BZ)*RABR
       !
       ! ST=sin(phi), Note that sin(phi).G/|G|=B^A/(|A|.|B|)
       ! which can be simplify to sin(phi)=|G|H.A/(|A|.|B|)
       ST=RG*RABR*(AX*HX+AY*HY+AZ*HZ)
#if KEY_OPLS==1
       IF (CID(IC).NE.0) THEN

          E=ZERO
          DF=ZERO
35        CONTINUE
          IPER=CID(IC)
          IF (IPER.GE.0) THEN
             LREP=.FALSE.
          ELSE
             LREP=.TRUE.
             IPER=-IPER
          ENDIF
          !
          E1=ONE
          DF1=ZERO
          !Calculation of cos(n*phi-phi0) and sin(n*phi-phi0).
          DO NPER=1,IPER
             DDF1=E1*CT-DF1*ST
             DF1=E1*ST+DF1*CT
             E1=DDF1
          enddo
          E1=E1*CICOS(IC)+DF1*CISIN(IC)
          DF1=DF1*CICOS(IC)-DDF1*CISIN(IC)
          DF1=-IPER*DF1
          E1=ONE+E1
          !
          ARG=CIC(IC)
          E=E+ARG*E1
          DF=DF+ARG*DF1
          !
          IF(LREP) THEN
             IC=IC+1
             GOTO 35
          ENDIF
          !

          ! Use harmonic potential
          !
       ELSE
          !
#endif 
          !
          !Calcul of cos(phi-phi0),sin(phi-phi0) and (Phi-Phi0).
          CA=CT*CICOS(IC)+ST*CISIN(IC)
          SA=ST*CICOS(IC)-CT*CISIN(IC)
          IF (CA.GT.PTONE ) THEN
             AP=ASIN(SA)
          ELSE
             AP=SIGN(ACOS(MAX(CA,MINONE)),SA)
             ! Warning is now triggered at deltaphi=84.26...deg (used to be 90).
             NWARNX=NWARNX+1
             IF((NWARNX.LE.5 .AND. WRNLEV.GE.5) .OR. WRNLEV.GE.6) THEN
                WRITE(OUTU,80) IPHI,AP*RADDEG,CIB(IC)*RADDEG,I,J,K,L
80              FORMAT(' EPHI: WARNING. bent improper torsion angle',  &
                     ' is '//'far ','from minimum for;'/3X,' IPHI=',I5, &
                     '  with deltaPHI=',F9.4,' MIN=',F9.4,' ATOMS:',4I5)
             ENDIF
          ENDIF
          !
          DF=CIC(IC)*AP
          E=DF*AP
          DF=TWO*DF
#if KEY_OPLS==1
       ENDIF   
#endif

       !
       !
       EIP=EIP+E
       !
       !     Compute derivatives wrt catesian coordinates.
       !
       ! GAA=dE/dphi.|G|/A^2, GBB=dE/dphi.|G|/B^2, FG=F.G, HG=H.G
       !  FGA=dE/dphi*F.G/(|G|A^2), HGB=dE/dphi*H.G/(|G|B^2)
       FG=FX*GX+FY*GY+FZ*GZ
       HG=HX*GX+HY*GY+HZ*GZ
       RA2R=DF*RA2R
       RB2R=DF*RB2R
       FGA=FG*RA2R*RGR
       HGB=HG*RB2R*RGR
       GAA=RA2R*RG
       GBB=RB2R*RG
       ! DFi=dE/dFi, DGi=dE/dGi, DHi=dE/dHi.
       DFX=-GAA*AX
       DFY=-GAA*AY
       DFZ=-GAA*AZ
       DGX=FGA*AX-HGB*BX
       DGY=FGA*AY-HGB*BY
       DGZ=FGA*AZ-HGB*BZ
       DHX=GBB*BX
       DHY=GBB*BY
       DHZ=GBB*BZ
       ! Distribute over Ri.
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
       !
    enddo loop110
    !
    NWARN=NWARN+NWARNX
    IF (NWARN.GT.5 .AND. WRNLEV.GE.3) WRITE (OUTU,175) NWARN
175 FORMAT(' EIPHIFASTER> Total of',I6,' WARNINGs issued.')
    !
    !.ab.
    !.ab.
    RETURN
  END SUBROUTINE EIPHIFASTER

!!$  subroutine enefscal_dummy()
!!$    return
!!$  end subroutine enefscal_dummy

end module eintern_fast

