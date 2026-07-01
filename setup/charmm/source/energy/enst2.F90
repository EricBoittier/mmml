#if KEY_NOST2==0
SUBROUTINE ENST2(EST2,X,Y,Z,DX,DY,DZ,JNBG,INBLOG,NGRPX, &
     CTONNB,CTOFNB,LEXTND)
  !-----------------------------------------------------------------------
  !     THIS ROUTINE CALCULATES THE WATER-WATER INTERACTION ENERGY
  !     AND DERIVATIVES FOR ST2-TYPE WATER PAIRS. THE WATERS ARE NOT
  !     ASSUMED TO BE RIGID, BUT RIGIDITY IS ENFORCED BY FIXST2.
  !     THIS LIMITS DYNAMICS CALCULATIONS WITH ST2 WATERS TO
  !     VERLET-TYPE INTEGRATION.
  !     MINIMIZATIONS USING THIS ROUTINE TO COMPUTE THE ST2 INTERACTIONS
  !     MUST BE SET UP WITH SHKST2 AND FXFST2 (FORCE CORECTION).
  !     THIS ROUTINE DOES NOT COMPUTE SECOND DERIVATIVES.
  !
  !     By  Bernard Brooks and John Brady  17-MAY-1983
  !
  use chm_kinds
  use dimens_fcm
  !
  use psf
  use stream
  use consta
  use chutil,only:atomid
  !
  implicit none
  INTEGER   INBLOG(*),NGRPX
  INTEGER   JNBG(*)
  real(chm_real) DX(*),DY(*),DZ(*)
  real(chm_real) CTOFNB, CTONNB
  real(chm_real) DDDXI,DDDYI,DDDZI,EST2PR,EST2
  real(chm_real) X(*),Y(*),Z(*)
  CHARACTER(len=8) SIDDN,RIDDN,RESDN,ACDN,SIDDN2,RIDDN2,RESDN2,ACDN2
  LOGICAL LEXTND
  !
  real(chm_real) RUL3,CTNBSQ,CTON,CTOF,RUL3X,QH2,SIGSQ,FOREPS
  real(chm_real) DXI,DYI,DZI,S,ROO,SW,DSW,RIJL,RIJU,R2,SIG2, &
       R6,R12,EN,DD
  real(chm_real) DDXI,DDYI,DDZI,DSWDXO,DSWDYO,DSWDZO,RIJ2,RIJ,D2
  real(chm_real) FXO,FYO,FZO
  INTEGER ITEMP,IRS,NB,NPR,IS,IQ,IH1,L,JRS,JS,JQ
  INTEGER JH1,IPT,II,JJ
  INTEGER IPHASE(16)
  real(chm_real), parameter :: RL=2.0160, RU=3.1287, QH=0.2357, SIGMA=3.10
  real(chm_real), parameter :: EPSLON=7.5750E-02
  DATA IPHASE/1,1,-1,-1,1,1,-1,-1,-1,-1,1,1,-1,-1,1,1/
  !
  !     The routine uses the group non-bonded interactionlists which is
  !     required for IMAGES and when building hydrogens (HBUILD).
  !     This approach is also consistant with methods used for general
  !     van der Waal and electrostatic interaction terms. BRB 8/11/83
  !
  RUL3=1.0/((RU-RL)**3)
  CTNBSQ=CTOFNB*CTOFNB
  CTON=CTONNB
  CTOF=CTOFNB
  IF (CTOF.GT.CTON) THEN
     RUL3X=1.0/((CTOF-CTON)**3)
  ELSE
     RUL3X=0.0
  ENDIF
  !
  QH2=QH**2
  QH2=QH2*CCELEC
  SIGSQ=SIGMA**2
  FOREPS=4.0*EPSLON
  !
  !     LOOP OVER GROUPS
  !
  EST2=0.0
  ITEMP=0
  !
  !     use non-bonded list
  DO IRS=1,NGRPX
     NB=ITEMP
     NPR=IABS(INBLOG(IRS))-ITEMP
     ITEMP=IABS(INBLOG(IRS))
     IF(INBLOG(IRS).LT.0) THEN
        NPR=NPR-1
        NB=NB+1
     ENDIF
     IF(NPR.GT.0 .AND. IGPTYP(IRS).EQ.3) THEN
        IS=IGPBS(IRS)+1
        IQ=IGPBS(IRS+1)
        IH1=IS+1
        DO L=1,NPR
           NB=NB+1
           JRS=JNBG(NB)
           IF (IGPTYP(JRS).EQ.3) THEN
              JS=IGPBS(JRS)+1
              JQ=IGPBS(JRS+1)
              JH1=JS+1
              !
              !     EVALUATE-ENERGY-BETWEEN-ST2-IS-AND-ST2-JS
              !     
              !     COMPUTE THE OXYGEN-OXYGEN CONTRIBUTIONS AND THE SWITCHING FUNCTION
              !
              DXI=X(IS)-X(JS)
              DYI=Y(IS)-Y(JS)
              DZI=Z(IS)-Z(JS)
              S=DXI*DXI+DYI*DYI+DZI*DZI
              IF(.NOT.(.NOT.LEXTND .AND. (S.GE.CTNBSQ))) THEN
                 ROO=SQRT(S)
                 !
                 IF (ROO.GE.RU .AND. ROO.LE.CTON ) THEN
                    SW=1.0
                    DSW=0.0
                 ELSE IF (ROO.LE.RL) THEN
                    SW=0.0
                    DSW=0.0
                 ELSE IF (ROO.LT.RU) THEN
                    !                 INNER SWITCHING REGION
                    RIJL=ROO-RL
                    RIJU=RU-ROO
                    SW=RIJL*RIJL*(3.0*RIJU+RIJL)*RUL3
                    DSW=6.0*RIJL*RIJU*RUL3
                    DSW=DSW/ROO
                 ELSE IF (LEXTND) THEN
                    SW=1.0
                    DSW=0.0
                 ELSE 
                    !                 OUTER SWITCHING REGION
                    RIJL=CTOF-ROO
                    RIJU=CTON-ROO
                    SW=RIJL*RIJL*(RIJL-3.0*RIJU)*RUL3X
                    DSW=6.0*RIJL*RIJU*RUL3X
                    DSW=DSW/ROO
                 ENDIF
                 !
                 R2=1.0/S
                 SIG2=SIGSQ*R2
                 R6=SIG2*SIG2*SIG2
                 R12=R6*R6
                 EN=FOREPS*(R12-R6)
                 EST2PR=EN
                 DD=FOREPS*R2*(6.0*R6-12.0*R12)
                 DDXI=DD*DXI
                 DDYI=DD*DYI
                 DDZI=DD*DZI
                 DX(IS)=DX(IS)+DDXI
                 DY(IS)=DY(IS)+DDYI
                 DZ(IS)=DZ(IS)+DDZI
                 DX(JS)=DX(JS)-DDXI
                 DY(JS)=DY(JS)-DDYI
                 DZ(JS)=DZ(JS)-DDZI
                 DSWDXO=DSW*DXI
                 DSWDYO=DSW*DYI
                 DSWDZO=DSW*DZI
                 IF (ROO.LE.RL) THEN
                    IF (WRNLEV.GT.5) THEN
                       CALL ATOMID(IS,SIDDN,RIDDN,RESDN,ACDN)
                       CALL ATOMID(JS,SIDDN2,RIDDN2,RESDN2,ACDN2)
                       WRITE(OUTU,5) SIDDN(1:idleng),RIDDN(1:idleng), &
                            RESDN(1:idleng),ACDN(1:idleng), &
                            SIDDN2(1:idleng),RIDDN2(1:idleng), &
                            RESDN2(1:idleng),ACDN2(1:idleng),ROO
5                      FORMAT(' ENST2: Atoms',4(1X,A),' and',4(1X,A), &
                            ' only',F5.2,' A. apart')
                    ENDIF
                 ELSE
                    !
                    IPT=0
                    DO II=IH1,IQ
                       DO JJ=JH1,JQ
                          IPT=IPT+1
                          !
                          DXI=X(II)-X(JJ)
                          DYI=Y(II)-Y(JJ)
                          DZI=Z(II)-Z(JJ)
                          !
                          RIJ2=DXI*DXI+DYI*DYI+DZI*DZI
                          RIJ=SQRT(RIJ2)
                          !
                          D2=QH2/RIJ
                          IF(IPHASE(IPT).LT.0) D2=-D2
                          !
                          EST2PR=EST2PR+D2*SW
                          !
                          DD=(-D2/RIJ2)*SW
                          DDDXI=DD*DXI
                          DDDYI=DD*DYI
                          DDDZI=DD*DZI
                          !
                          DX(II)=DX(II)+DDDXI
                          DY(II)=DY(II)+DDDYI
                          DZ(II)=DZ(II)+DDDZI
                          DX(JJ)=DX(JJ)-DDDXI
                          DY(JJ)=DY(JJ)-DDDYI
                          DZ(JJ)=DZ(JJ)-DDDZI
                          !
                          IF(DSW.NE.0.0) THEN
                             FXO=DSWDXO*D2
                             FYO=DSWDYO*D2
                             FZO=DSWDZO*D2
                             DX(IS)=DX(IS)+FXO
                             DY(IS)=DY(IS)+FYO
                             DZ(IS)=DZ(IS)+FZO
                             DX(JS)=DX(JS)-FXO
                             DY(JS)=DY(JS)-FYO
                             DZ(JS)=DZ(JS)-FZO
                          ENDIF
                       ENDDO
                    ENDDO
                 ENDIF
                 EST2=EST2+EST2PR
              ENDIF
           ENDIF
        ENDDO
     ENDIF
  END DO
  !
  RETURN
END SUBROUTINE ENST2

SUBROUTINE FIXST2(X,Y,Z,MASSW)
  !-----------------------------------------------------------------------
  !     THIS ROUTINE FINDS THE POSITION OF THE ST2 LONE PAIRS AND
  !     MODIFIES LONE PAIR POSITIONS IF NOT CORRECT (RIGID MODEL)
  !     THERE IS ALSO A CHECK AS TO WHETHER THE ST2 IS CORRECT
  !     (CHARGES, NUMBER OF ATOMS, ...)
  !     MASSW IS CODED AS FOLLOWS;
  !     -1 = ASSUME O IS FIXED, AND GIVE NO WEIGHTING AT ALL TO LP'S
  !     0 = ASSUME ALL ATOMS HAVE EQUAL WEIGHT (1.0) INCLUDING LP'S
  !     1 = USE THE MASS ARRAY (AND MAKE LARGE IF ATOM IS FIXED
  !
  !     By Bernard Brooks, 22-APR-1983
  !
  !
  use chm_kinds
  use dimens_fcm
  use psf
  use corsubs,only:rotls1
  use chutil,only:initia
  !
  implicit none
  real(chm_real) X(*),Y(*),Z(*)
  INTEGER MASSW
  !
  real(chm_real) BMASS(5)
  real(chm_real) XROT(5),YROT(5),ZROT(5)
  real(chm_real) XIDEAL(5),YIDEAL(5),ZIDEAL(5)
  INTEGER ATOMPR(2,5)
  real(chm_real) ALPH,BETH,ALPL,BETL
  INTEGER J,I,IS,IL,IPT
  INTEGER, parameter :: NATST2=5
  real(chm_real), parameter :: BIGMAS=1.0E6, ROH=1.0, ROL=0.8
  DATA ATOMPR/1,1,2,2,3,3,4,4,5,5/
  !
  !
  IF(NST2.LE.0) RETURN
  BMASS(4)=0.0
  BMASS(5)=0.0
  !
  !
  !     Fit ST2 to ideal geometry
  !
  !     ROH is the ST2 hydrogen oxygen distance
  !     ROL is the ST2 lone-pair oxygen distance
  !
  ALPH=ROH*SQRT(1.0D0/3.0D0)
  BETH=ROH*SQRT(2.0D0/3.0D0)
  ALPL=ROL*SQRT(1.0D0/3.0D0)
  BETL=ROL*SQRT(2.0D0/3.0D0)
  !
  DO J=1,NATST2
     XIDEAL(J)=0.0
     YIDEAL(J)=0.0
     ZIDEAL(J)=0.0
  ENDDO
  XIDEAL(2)=ALPH
  YIDEAL(2)=BETH
  XIDEAL(3)=ALPH
  YIDEAL(3)=-BETH
  XIDEAL(4)=-ALPL
  ZIDEAL(4)=-BETL
  XIDEAL(5)=-ALPL
  ZIDEAL(5)=BETL
  !
  DO I=1,NGRP
     IF(IGPTYP(I).EQ.3) THEN
        !
        IS=IGPBS(I)+1
        IL=IGPBS(I+1)
        IF(IL-IS.NE.4) CALL WRNDIE(-4,'FIXST2', &
             'ST2 MUST HAVE 5 ATOMS')
        !
        !     Fix IMOVE just in case it is wrong
        IMOVE(IS+3)=-1
        IMOVE(IS+4)=-1
        !
        !
        IF (IMOVE(IS).EQ.0 .AND. IMOVE(IS+1).EQ.0 &
             .AND. IMOVE(IS+2).EQ.0) THEN
           IPT=IS
           DO J=1,NATST2
              XROT(J)=XIDEAL(J)
              YROT(J)=YIDEAL(J)
              ZROT(J)=ZIDEAL(J)
              IF (MASSW.EQ.0) THEN
                 BMASS(J)=1.0
              ELSE
                 BMASS(J)=AMASS(J+IS-1)
              ENDIF
              IF(IMOVE(IPT).GT.0) BMASS(J)=BIGMAS
              IF(.NOT.INITIA(IPT,X,Y,Z)) BMASS(J)=0.0
              IPT=IPT+1
           ENDDO
           IF(MASSW.EQ.-1) BMASS(1)=BIGMAS
           !
           !     Least squares fit of Xrot to X,Y,Z with weights determined by
           !     BMASS
           !
           CALL ROTLS1(X(IS),Y(IS),Z(IS),XROT,YROT,ZROT,NATST2, &
                ATOMPR,NATST2,BMASS,.FALSE.,.FALSE.)
           !
           !     COPY TO COORDINATE ARRAYS
           IPT=1
           DO J=IS,IL
              X(J)=XROT(IPT)
              Y(J)=YROT(IPT)
              Z(J)=ZROT(IPT)
              IPT=IPT+1
           ENDDO
        ENDIF
     ENDIF
  ENDDO
  !
  RETURN
END SUBROUTINE FIXST2

SUBROUTINE SHKST2(X,Y,Z,XREF,YREF,ZREF,LMASS,NITER)
  !-----------------------------------------------------------------------
  !     THIS ROUTINE SHAKES X,Y,Z WITH RESPECT TO XREF,YREF,ZREF.
  !     THIS CODE IS PARALLEL TO SHAKEC, BUT ONLY TREATS ST2 MOLECULES.
  !
  !     By Bernard R. Brooks    May 1983
  !
  use chm_kinds
  use dimens_fcm
  use psf
  use shake
  use stream
#if KEY_TSM==1
  use tsms_mod
#endif 
  !
  implicit none
  real(chm_real) X(*),Y(*),Z(*),XREF(*),YREF(*),ZREF(*)
  LOGICAL LMASS
  INTEGER NITER
  !
  !
  real(chm_real), parameter :: ROH=1.0, ROL=0.8
  real(chm_real) BMASS(3)
  !
  real(chm_real) GX(5),GY(5),GZ(5),RX(5),RY(5),RZ(5)
  real(chm_real) ALPH,BETH,ALPL,BETL,TOL2
  INTEGER IGRP,IS,IL,KPT,J,I,NIT,LL
  real(chm_real) TOLER,XPIJ,YPIJ,ZPIJ,DIFF,XIJ,YIJ,ZIJ,RRPR, &
       AMSI,AMSJ,ACOR
  real(chm_real) AX,AY,AZ,AT,BX,BY,BZ,BT,CX,CY,CZ
  LOGICAL READY
  real(chm_real) CONST2(3)
  !
  INTEGER LRIGID,LSHAKE
  DATA LRIGID/1/,LSHAKE/1/
  !
  IF(NST2.LE.0) RETURN
#if KEY_TSM==1
  IF(QTSM.AND.PIGSET) THEN
     CALL PIGCVSET(XREF,YREF,ZREF)
     CALL PIGCVSET(X,Y,Z)
  ENDIF
#endif 
  !
  !     ROH is the ST2 hydrogen oxygen distance
  !     ROL is the ST2 lone-pair oxygen distance
  !
  ALPH=ROH*SQRT(1.0D0/3.0D0)
  BETH=ROH*SQRT(2.0D0/3.0D0)
  ALPL=ROL*SQRT(1.0D0/3.0D0)
  BETL=ROL*SQRT(2.0D0/3.0D0)
  !
  !     CONST2 CONTAINS ST2 INTERNAL DISTANCES SQUARED
  !
  CONST2(1)=ROH*ROH
  CONST2(2)=CONST2(1)
  CONST2(3)=4.0*BETH*BETH
  !
  TOL2=2.0*SHKTOL
  NITER=0
  !
  !     LOOP OVER ST2 GROUPS
  !     LOOP OVER SHAKE ITERATIONS
  !
  DO IGRP=1,NGRP
     IF(IGPTYP(IGRP).EQ.3) THEN
        IS=IGPBS(IGRP)+1
        !
        !
        IL=IGPBS(IGRP+1)
        IF(IL-IS.NE.4) CALL WRNDIE(-4,'<SHKST2>','ST2 MUST HAVE 5 ' &
             //'ATOMS')
        !
        !
        !         COPY COORDINATES AND REFERENCE SET TO TEMPORARY SET
        !
        KPT=IS
        DO J=1,3
           GX(J)=X(KPT)
           GY(J)=Y(KPT)
           GZ(J)=Z(KPT)
           RX(J)=XREF(KPT)
           RY(J)=YREF(KPT)
           RZ(J)=ZREF(KPT)
           KPT=KPT+1
        ENDDO
        !
        IF(LSHAKE.NE.0) THEN
           DO I=1,3
              IF (LMASS) THEN
                 BMASS(I)=1.0/AMASS(I+IS-1)
              ELSE
                 BMASS(I)=1.0
              ENDIF
              IF(IMOVE(I+IS-1).GT.0) BMASS(I)=0.0
           ENDDO
           !
           NIT=0
10         CONTINUE
           IF (NIT.GT.MXITER) THEN
              IF(WRNLEV.GE.2) WRITE(OUTU,311) MXITER
              CALL DIEWRN(-2)
#if KEY_TSM==1
              IF(QTSM.AND.PIGSET) CALL PIGCVSET(X,Y,Z)
#endif 
              IF(.TRUE.) RETURN
           ENDIF
           READY=.TRUE.
           LL=0
           DO I=1,2
              DO J=I+1,3
                 LL=LL+1
                 IF (IMOVE(I+IS-1).EQ.0 .OR. IMOVE(J+IS-1).EQ.0) THEN
                    TOLER=CONST2(LL)
                    XPIJ=GX(I)-GX(J)
                    YPIJ=GY(I)-GY(J)
                    ZPIJ=GZ(I)-GZ(J)
                    DIFF=TOLER-XPIJ*XPIJ-YPIJ*YPIJ-ZPIJ*ZPIJ
                    !
                    !                 COMPARE DIFFERENCE IN R**2 WITH TOLERANCE
                    !
                    IF (ABS(DIFF).LT.TOLER*TOL2) GOTO 100
                    !
                    !                 DETERMINE OLD ( OR REFERENCE ) BOND DIRECTION
                    !
                    XIJ=RX(I)-RX(J)
                    YIJ=RY(I)-RY(J)
                    ZIJ=RZ(I)-RZ(J)
                    RRPR=XIJ*XPIJ+YIJ*YPIJ+ZIJ*ZPIJ
                    IF (RRPR.LT.TOLER*1.E-6) THEN
                       !                   DEVIATION-TOO-LARGE
                       IF(WRNLEV.GE.2) WRITE(OUTU,321) NITER,LL,I,J
                       CALL DIEWRN(-2)
#if KEY_TSM==1
                       IF(QTSM.AND.PIGSET) CALL PIGCVSET(X,Y,Z)
#endif 
                       IF(.TRUE.) RETURN
                    ENDIF
                    AMSI=BMASS(I)
                    AMSJ=BMASS(J)
                    ACOR=DIFF/(RRPR*(AMSI+AMSJ)*2.0)
                    !
                    !                 SHIFT NEW COORDINATES ALONG OLD BOND DIRECTION
                    !
                    XIJ=XIJ*ACOR
                    GX(I)=GX(I)+XIJ*AMSI
                    GX(J)=GX(J)-XIJ*AMSJ
                    YIJ=YIJ*ACOR
                    GY(I)=GY(I)+YIJ*AMSI
                    GY(J)=GY(J)-YIJ*AMSJ
                    ZIJ=ZIJ*ACOR
                    GZ(I)=GZ(I)+ZIJ*AMSI
                    GZ(J)=GZ(J)-ZIJ*AMSJ
                    !
                    !                 SET FLAGS
                    !
                    READY=.FALSE.
                 ENDIF
100              CONTINUE
              ENDDO
           ENDDO
           NIT=NIT+1
           !
           !           ITERATION COMPLETE
           !
           IF (.NOT.(READY)) GOTO 10
           IF(NIT.GT.NITER) NITER=NIT
        ENDIF
        !
        !         CALCULATE THE LONE PAIR POSITIONS
        !
        DO J=2,3
           RX(J)=GX(J)-GX(1)
           RY(J)=GY(J)-GY(1)
           RZ(J)=GZ(J)-GZ(1)
        ENDDO
        !
        AX=RX(2)+RX(3)
        AY=RY(2)+RY(3)
        AZ=RZ(2)+RZ(3)
        AT=AX*AX+AY*AY+AZ*AZ
        AT=-1.0/SQRT(AT)
        AX=AX*AT
        AY=AY*AT
        AZ=AZ*AT
        !
        BX=RY(2)*RZ(3)-RZ(2)*RY(3)
        BY=RZ(2)*RX(3)-RX(2)*RZ(3)
        BZ=RX(2)*RY(3)-RY(2)*RX(3)
        BT=BX*BX+BY*BY+BZ*BZ
        BT=1.0/SQRT(BT)
        BX=BX*BT
        BY=BY*BT
        BZ=BZ*BT
        !
        GX(4)=ALPL*AX+BETL*BX+GX(1)
        GY(4)=ALPL*AY+BETL*BY+GY(1)
        GZ(4)=ALPL*AZ+BETL*BZ+GZ(1)
        GX(5)=ALPL*AX-BETL*BX+GX(1)
        GY(5)=ALPL*AY-BETL*BY+GY(1)
        GZ(5)=ALPL*AZ-BETL*BZ+GZ(1)
        !
        !
        !         REPOSITION THE HYDROGEN ATOMS IF RIGID IS SPECIFIED
        IF (LRIGID.NE.0) THEN
           CX=AY*BZ-AZ*BY
           CY=AZ*BX-AX*BZ
           CZ=AX*BY-AY*BX
           !
           GX(3)=GX(1)-ALPH*AX+BETH*CX
           GX(2)=GX(1)-ALPH*AX-BETH*CX
           GY(3)=GY(1)-ALPH*AY+BETH*CY
           GY(2)=GY(1)-ALPH*AY-BETH*CY
           GZ(3)=GZ(1)-ALPH*AZ+BETH*CZ
           GZ(2)=GZ(1)-ALPH*AZ-BETH*CZ
        ENDIF
        !
        !         COPY TEMPORARY COORDINATES BACK
        !
        KPT=IS
        DO J=1,5
           X(KPT)=GX(J)
           Y(KPT)=GY(J)
           Z(KPT)=GZ(J)
           KPT=KPT+1
        ENDDO
        !
        !
     ENDIF
  ENDDO
311 FORMAT(' ***** ERROR IN SHKST2 ***** COORDINATE RESETTING', &
       ' WAS NOT ACCOMPLISHED IN',I6,' ITERATIONS',/)
321 FORMAT(' ***** ERROR IN SHKST2 ***** DEVIATION IN SHAKE', &
       ' TOO LARGE',4I6,/)
  !
  IF (PRNLEV.GE.6) WRITE(OUTU,331) NITER
331 FORMAT(' SHKST2: Worst ST2 shaked in',I5,' iterations.')
#if KEY_TSM==1
  IF(QTSM.AND.PIGSET) CALL PIGCVSET(X,Y,Z)
#endif 
  !
  RETURN
END SUBROUTINE SHKST2

SUBROUTINE FXFST2(X,Y,Z,DX,DY,DZ,LMASS)
  !-----------------------------------------------------------------------
  !     This routine will redistribute forces on ST2'S so that net forces
  !     and torques remain constant, while all internal forces and
  !     forces at lone pairs are removed. It will work on any set of atoms
  !     and lone pairs with minor modification.
  !
  !     LMASS is coded as follows:
  !     .FALSE. = assume all atoms have equal weight (1.0) excluding l.p.'s,
  !     .TRUE.  = assume all atoms have weight equal to physical mass.
  !     Lone pairs have always mass zero.
  !
  !     The method assumes that ST2's are correctly represented in the PSF
  !     i.e. the atom number sequence for an ST2 has to be (oxygen,
  !     hydrogen1, hydrogen2, lone pair 1, lone pair 2).
  !
  !     This routine is the result of a series of attempts to keep ST2's
  !     rigid in both dynamics and minimization methods by Bernard Brooks
  !     and Axel Brunger, 1-JUN-1983.
  !
  !
  use chm_kinds
  use dimens_fcm
  !
  use psf
  !
  implicit none
  real(chm_real) X(*),Y(*),Z(*)
  real(chm_real)  DX(*), DY(*), DZ(*)
  LOGICAL LMASS
  !
  INTEGER IS, IGPR, I, J, NRED
  real(chm_real), parameter :: BIGMAS=1.0E8, MINDET=1.0E-16
  integer, parameter :: NATST2=5, NLONE=2, NM=3
  !
  INTEGER LH(NM), MH(NM)
  real(chm_real)  XR(NATST2), YR(NATST2), ZR(NATST2)
  real(chm_real)  XCM, YCM, ZCM, ARMASS(NATST2), AMASST
  real(chm_real)  FXCM, FYCM, FZCM, TXCM, TYCM, TZCM
  real(chm_real)  XX, XY, XZ, YY, YZ, ZZ, TCM(NM,NM), D
  real(chm_real)  OXCM, OYCM, OZCM
  !
  !
  IF(NST2.LE.0) RETURN
  !
  NRED=NATST2-NLONE
  !
  DO IGPR=1,NGRP
     IF(IGPTYP(IGPR).EQ.3) THEN
        IS=IGPBS(IGPR)+1
        !
        !     get masses for ST2 atoms and center of mass XCM, YCM, ZCM
        !     ( R = sum { Ri Mi } / M,  M = sum Mi ).
        !     The mass of a fixed atom is set to the big value BIGMAS, the mass
        !     of a lone pair is set to zero.
        !
        XCM=0.0
        YCM=0.0
        ZCM=0.0
        AMASST=0.0
        DO J=1,NATST2
           I=(IS+J-1)
           XR(J)=X(I)
           YR(J)=Y(I)
           ZR(J)=Z(I)
           IF (NATST2-J.LT.NLONE) THEN
              ARMASS(J)=0.0
           ELSE IF (IMOVE(I).GT.0) THEN
              ARMASS(J)=BIGMAS
           ELSE IF (LMASS) THEN
              ARMASS(J)=AMASS(I)
           ELSE
              ARMASS(J)=1.0
           ENDIF
           XCM=XCM+XR(J)*ARMASS(J)
           YCM=YCM+YR(J)*ARMASS(J)
           ZCM=ZCM+ZR(J)*ARMASS(J)
           AMASST=AMASST+ARMASS(J)
        ENDDO
        XCM=XCM/AMASST
        YCM=YCM/AMASST
        ZCM=ZCM/AMASST
        !
        !     translate temporary coordinates XR,YR,ZR to center of mass
        !
        DO J=1,NATST2
           XR(J)=XR(J)-XCM
           YR(J)=YR(J)-YCM
           ZR(J)=ZR(J)-ZCM
        ENDDO
        !
        !     now evaluate the center of mass force FXCM,FYCM,FZCM
        !     ( F = sum Fi ) and
        !     torque TXCM,TYCM,TZCM  ( T =  sum {Ri X Fi} )
        !
        FXCM=0.0
        FYCM=0.0
        FZCM=0.0
        TXCM=0.0
        TYCM=0.0
        TZCM=0.0
        DO J=1,NATST2
           I=(IS+J-1)
           FXCM=FXCM+DX(I)
           FYCM=FYCM+DY(I)
           FZCM=FZCM+DZ(I)
           TXCM=TXCM+(YR(J)*DZ(I)-ZR(J)*DY(I))
           TYCM=TYCM+(ZR(J)*DX(I)-XR(J)*DZ(I))
           TZCM=TZCM+(XR(J)*DY(I)-YR(J)*DX(I))
        ENDDO
        !
        !     now evaluate the inertia tensor TCM for the reduced ST2 
        !     (without lone pairs)
        !
        XX=0.0
        XY=0.0
        XZ=0.0
        YY=0.0
        YZ=0.0
        ZZ=0.0
        DO J=1,NRED
           XX=XX+XR(J)*XR(J)*ARMASS(J)
           XY=XY+XR(J)*YR(J)*ARMASS(J)
           XZ=XZ+XR(J)*ZR(J)*ARMASS(J)
           YY=YY+YR(J)*YR(J)*ARMASS(J)
           YZ=YZ+YR(J)*ZR(J)*ARMASS(J)
           ZZ=ZZ+ZR(J)*ZR(J)*ARMASS(J)
        ENDDO
        TCM(1,1)=YY+ZZ
        TCM(2,1)=-XY
        TCM(3,1)=-XZ
        TCM(1,2)=-XY
        TCM(2,2)=XX+ZZ
        TCM(3,2)=-YZ
        TCM(1,3)=-XZ
        TCM(2,3)=-YZ
        TCM(3,3)=XX+YY
        !
        !     invert the inertia tensor TCM[1..3,1..3]
        !
        CALL MINVST(TCM,NM,D,LH,MH)
        IF (ABS(D).LT.MINDET) THEN
           CALL WRNDIE(-2,'FXFST2','Determinant of TCM zero.')
        ELSE
           !
           !     get the angular acceleration vector OXCM, OYCM, OZCM
           !     by using the relation ( T = I * O )
           !
           OXCM=TXCM*TCM(1,1)+TYCM*TCM(1,2)+TZCM*TCM(1,3)
           OYCM=TXCM*TCM(2,1)+TYCM*TCM(2,2)+TZCM*TCM(2,3)
           OZCM=TXCM*TCM(3,1)+TYCM*TCM(3,2)+TZCM*TCM(3,3)
           !
           !     get back the forces with no internal contributions, i.e.
           !     ( Fi = { F/M + O X Ri } / Mi ),
           !     set the forces for lone pairs to zero.
           !
           FXCM=FXCM/AMASST
           FYCM=FYCM/AMASST
           FZCM=FZCM/AMASST
           DO J=1,NRED
              I=(IS+J-1)
              DX(I)=(FXCM+OYCM*ZR(J)-OZCM*YR(J))*ARMASS(J)
              DY(I)=(FYCM+OZCM*XR(J)-OXCM*ZR(J))*ARMASS(J)
              DZ(I)=(FZCM+OXCM*YR(J)-OYCM*XR(J))*ARMASS(J)
           ENDDO
           DO J=NRED+1,NATST2
              I=(IS+J-1)
              DX(I)=0.0
              DY(I)=0.0
              DZ(I)=0.0
           ENDDO
        ENDIF
        !
     ENDIF
  ENDDO
  !
  RETURN
END SUBROUTINE FXFST2

SUBROUTINE FXLST2(X,Y,Z,DX,DY,DZ)
  !-----------------------------------------------------------------------
  !     THIS ROUTINE REDISTRIBUTES FORCES ON ST2'S SO THAT NET FORCE
  !     AND TORQUE REMAIN CONSTANT. ZERO OUT FORCES ON LONE PAIRS.
  !     THIS ALGORITHM ASSUMES THAT RIGID ST2'S ARE IN USE.
  !     INTERNAL FORCES ARE NOT REMOVED.
  !
  !     By Bernard R. Brooks    May 1983
  !
  use chm_kinds
  use dimens_fcm
  use number
  use psf
  implicit none
  real(chm_real) X(*),Y(*),Z(*),DX(*),DY(*),DZ(*)
  !
  !
  real(chm_real) ROH,ROL,ALPHA,BETA,OPTA,FXP,FXM,FYP,FYM,FZP,FZM
  real(chm_real) X12,Y12,Z12,X31,Y31,Z31,X23,Y23,Z23
  INTEGER IGPR,IS
  !
  ROH=ONE
  ROL=0.80
  !
  ALPHA=ROL/(TWO*ROH)
  BETA=ALPHA*SQRT(THREE)
  OPTA=ONE+(TWO*ALPHA)
  !
  !
  DO IGPR=1,NGRP
     IF(IGPTYP(IGPR).EQ.3) THEN
        IS=IGPBS(IGPR)+1
        !
        FXP=DX(IS+3)+DX(IS+4)
        FXM=DX(IS+3)-DX(IS+4)
        FYP=DY(IS+3)+DY(IS+4)
        FYM=DY(IS+3)-DY(IS+4)
        FZP=DZ(IS+3)+DZ(IS+4)
        FZM=DZ(IS+3)-DZ(IS+4)
        !
        X12=X(IS)-X(IS+1)
        Y12=Y(IS)-Y(IS+1)
        Z12=Z(IS)-Z(IS+1)
        X31=X(IS+2)-X(IS)
        Y31=Y(IS+2)-Y(IS)
        Z31=Z(IS+2)-Z(IS)
        X23=X(IS+1)-X(IS+2)
        Y23=Y(IS+1)-Y(IS+2)
        Z23=Z(IS+1)-Z(IS+2)
        !
        DX(IS)=DX(IS)+OPTA*FXP+BETA*(FZM*Y23-FYM*Z23)
        DY(IS)=DY(IS)+OPTA*FYP+BETA*(FXM*Z23-FZM*X23)
        DZ(IS)=DZ(IS)+OPTA*FZP+BETA*(FYM*X23-FXM*Y23)
        !
        DX(IS+1)=DX(IS+1)-ALPHA*FXP+BETA*(FZM*Y31-FYM*Z31)
        DY(IS+1)=DY(IS+1)-ALPHA*FYP+BETA*(FXM*Z31-FZM*X31)
        DZ(IS+1)=DZ(IS+1)-ALPHA*FZP+BETA*(FYM*X31-FXM*Y31)
        !
        DX(IS+2)=DX(IS+2)-ALPHA*FXP+BETA*(FZM*Y12-FYM*Z12)
        DY(IS+2)=DY(IS+2)-ALPHA*FYP+BETA*(FXM*Z12-FZM*X12)
        DZ(IS+2)=DZ(IS+2)-ALPHA*FZP+BETA*(FYM*X12-FXM*Y12)
        !
        DX(IS+3)=0.0
        DY(IS+3)=0.0
        DZ(IS+3)=0.0
        !
        DX(IS+4)=0.0
        DY(IS+4)=0.0
        DZ(IS+4)=0.0
        !
     ENDIF
  ENDDO
  !
  RETURN
END SUBROUTINE FXLST2

SUBROUTINE ST2WRN(DISTOF,ANGLON,X,Y,Z,NONPOL)
  !-----------------------------------------------------------------------
  !     Subroutine returns a warning if a hydrogen or a lone pair of an
  !     ST2 water point to a nonpolar heavy atom (e.g. methyl group)
  !     within a specified range of distances (0.0<...<DISTOFF) and
  !     a specified range of angles (ANGLON<...<180.0).
  !
  !     Typical warning ranges are (0.0...5.2 A.) and (140...180 deg.).
  !     These approximate values were obtained by molecular dynamics
  !     calculations on dipeptides in ST2 water. (P.Rossky & M.Karplus,
  !     JACS 101 pp. 1913 (1979) and J.Brady & Karplus, to be published).
  !
  !     Routine applies only to ST2 water.
  !
  !     6-FEB-83 Axel Brunger
  !
  use chm_kinds
  use dimens_fcm
  !
  use psf
  use stream
  use chutil,only:atomid,lone,hydrog

  implicit none
  !
  real(chm_real)    DISTOF, ANGLON
  real(chm_real) X(*),Y(*),Z(*)
  INTEGER   NONPOL(NATOM)
  !
  INTEGER   I, IG, J, IST2, INP
  real(chm_real)    DSTOF2, CANGON
  real(chm_real)    X1D, Y1D, Z1D, X2D, Y2D, Z2D, DIST2, DIST12,  &
       DIST22
  real(chm_real)    CSANGL
  !
  CHARACTER(len=8) SID, RID, CRES, AC, SID2, RID2, CRES2, AC2
  !
  real(chm_real)    RAD
  DATA      RAD/0.174532925D-01/
  !
  IF(PRNLEV.GE.2) WRITE(OUTU,25) DISTOF, ANGLON
25 FORMAT(/' ST2WRN called with DISTOF = ',F10.4, &
       '   and ANGLON = ',F10.4)
  DSTOF2=DISTOF*DISTOF
  CANGON=-COS(ANGLON*RAD)
  !
  !     GET-LIST-OF-NONPOLAR-ATOMS
  !
  !     exclude donors, acceptors, ST2 groups, hydrogens and lone pairs,
  !     donor and acceptor antecedents.
  !
  DO I=1,NATOM
     NONPOL(I)=1
  ENDDO
  DO I=1,NDON
     IF (IDON(I).GT.0) NONPOL(IDON(I))=0
  ENDDO
  DO I=1,NACC
     IF (IACC(I).GT.0) NONPOL(IACC(I))=0
  ENDDO
  !
  !     now all acceptor antecedents and donor antecedents
  !     are excluded from the non-polar list.
  !
  DO I=1,NBOND
     IF (IB(I).GT.0.AND.JB(I).GT.0) THEN
        IF (NONPOL(IB(I)).EQ.0) THEN
           NONPOL(JB(I))=2
        ELSE IF (NONPOL(JB(I)).EQ.0) THEN
           NONPOL(IB(I))=2
        ENDIF
     ENDIF
  ENDDO
  DO I=1,NATOM
     IF (NONPOL(I).EQ.2) NONPOL(I)=0
  ENDDO
  !
  !     now we get rid of ST2 atoms and of hydrogens
  !
  DO J=1,NGRP
     IF (IGPTYP(J).EQ.3) THEN
        DO I=IGPBS(J)+1,IGPBS(J+1)
           NONPOL(I)=0
        ENDDO
     ENDIF
  ENDDO
  DO  I=1,NATOM
     IF (HYDROG(I).OR.LONE(I)) NONPOL(I)=0
  ENDDO
  !
  DO IG=1,NGRP
     IF(IGPTYP(IG).EQ.3) THEN
        IST2=IGPBS(IG)+1
        !     DO-DISTANCE-AND-ANGLE-TEST-TO-NONPOLAR-GROUPS
        DO INP=1,NATOM
           IF (NONPOL(INP).EQ.1) THEN
              X1D=X(INP)-X(IST2)
              Y1D=Y(INP)-Y(IST2)
              Z1D=Z(INP)-Z(IST2)
              DIST2=(X1D*X1D+Y1D*Y1D+Z1D*Z1D)
              IF (DIST2.LE.1.0E-4) THEN
                 !     DISTANCES-ZERO
                 IF(WRNLEV.GE.2) THEN
                    CALL ATOMID(INP,SID,RID,CRES,AC)
                    CALL ATOMID(I,SID2,RID2,CRES2,AC2)
                    WRITE(OUTU,115) SID2(1:idleng),RID2(1:idleng), &
                         CRES2(1:idleng),AC2(1:idleng), &
                         SID(1:idleng),RID(1:idleng), &
                         CRES(1:idleng),AC(1:idleng)
                 ENDIF
                 CALL DIEWRN(-4)
              ELSE
                 IF (DIST2.LT.DSTOF2) THEN
                    !
                    !     ST2 heavy atom and nonpolar atom are within spec. range.
                    !
                    DO I=IST2+1,IST2+4
                       X1D=X(I)-X(IST2)
                       Y1D=Y(I)-Y(IST2)
                       Z1D=Z(I)-Z(IST2)
                       X2D=X(INP)-X(IST2)
                       Y2D=Y(INP)-Y(IST2)
                       Z2D=Z(INP)-Z(IST2)
                       DIST12=X1D*X1D+Y1D*Y1D+Z1D*Z1D
                       DIST22=X2D*X2D+Y2D*Y2D+Z2D*Z2D
                       IF (DIST12.LE.1.0E-4 .OR. DIST22.LE.1.0E-4) THEN
                          !     DISTANCES-ZERO
                          IF(WRNLEV.GE.2) THEN
                             CALL ATOMID(INP,SID,RID,CRES,AC)
                             CALL ATOMID(I,SID2,RID2,CRES2,AC2)
                             WRITE(OUTU,115) SID2(1:idleng),RID2(1:idleng), &
                                  CRES2(1:idleng),AC2(1:idleng), &
                                  SID(1:idleng),RID(1:idleng), &
                                  CRES(1:idleng),AC(1:idleng)
                          ENDIF
                          CALL DIEWRN(-4)
                       ELSE
                          CSANGL=(X1D*X2D+Y1D*Y2D+Z1D*Z1D)/ &
                               SQRT(DIST12*DIST22)
                          IF (CANGON.LT.CSANGL) THEN
                             !
                             !     a warning message should be printed
                             !
                             IF(WRNLEV.GE.2) THEN
                                CALL ATOMID(INP,SID,RID,CRES,AC)
                                CALL ATOMID(I,SID2,RID2,CRES2,AC2)
                                WRITE(OUTU,125) SID(1:idleng),RID(1:idleng), &
                                     CRES(1:idleng),AC(1:idleng), &
                                     SID2(1:idleng),RID2(1:idleng), &
                                     CRES2(1:idleng),AC2(1:idleng), &
                                     ACOS(-CSANGL)/RAD,SQRT(DIST2)
                             ENDIF
                          ENDIF
                       ENDIF
                    ENDDO
                 ENDIF
              ENDIF
           ENDIF
        ENDDO
        !
     ENDIF
  ENDDO
  !
  RETURN
  !
115 FORMAT(' ERROR FROM ST2WRN: distance zero between ', &
       4(1X,A),' and atom',4(1X,A))
125 FORMAT(/' ST2WRN: atom1 ',4(1X,A),' and atom2',4(1X,A),/, &
       '         form  (atom1-OX2-atom2) angle= ',F12.5,' and', &
       ' (atom1-OX2) distance= ',F10.5)
  !
END SUBROUTINE ST2WRN

SUBROUTINE MINVST(A,N,D,L,M)
  !-----------------------------------------------------------------------
  !     MINV INVERTS THE N*N MATRIX A, USING THE STANDARD GAUSS-
  !     JORDAN METHOD. THE DETERMINANT IS ALSO CALCULATED. IT HAS BEEN
  !
  !     A(1..N,1..N) = MATRIX TO BE INVERTED; DELIVERED WITH THE INVERSE
  !     N = ORDER OF MATRIX A
  !     D = DELIVERED WITH THE DETERMINANT OF A
  !     L,M(1..N) = DUMMY ARRAYS
  !
  !     Included by: W.F. VAN GUNSTEREN   APR. 1979
  !
  use chm_kinds
  use number
  use stream
  implicit none
  !
  real(chm_real) A(*),D
  INTEGER N
  INTEGER L(*),M(*)
  !
  INTEGER NK,K,KK,J,IZ,I,IJ,KI,JI,JP,JK,IK,KJ,JQ,JR
  real(chm_real) BIGA,HOLD
  !
  !***SEARCH FOR LARGEST ELEMENT
  D=1.0
  NK=-N
  DO K=1,N
     NK=NK+N
     L(K)=K
     M(K)=K
     KK=NK+K
     BIGA=A(KK)
     DO J=K,N
        IZ=N*(J-1)
        DO I=K,N
           IJ=IZ+I
           IF(ABS(BIGA).LT.ABS(A(IJ))) THEN
              BIGA=A(IJ)
              L(K)=I
              M(K)=J
           ENDIF
        ENDDO
     ENDDO
     !
     !***INTERCHANGE ROWS
     J=L(K)
     IF(J.GT.K) THEN
        KI=K-N
        DO I=1,N
           KI=KI+N
           HOLD=-A(KI)
           JI=KI-K+J
           A(KI)=A(JI)
           A(JI)=HOLD
        ENDDO
     ENDIF
     !
     !***INTERCHANGE COLUMNS
     I=M(K)
     IF(I.GT.K) THEN
        JP=N*(I-1)
        DO J=1,N
           JK=NK+J
           JI=JP+J
           HOLD=-A(JK)
           A(JK)=A(JI)
           A(JI)=HOLD
        ENDDO
     ENDIF
     !
     !***DIVIDE COLUMN BY MINUS PIVOT
     IF(BIGA.EQ.ZERO) THEN
        D=0.0
        RETURN
     ENDIF
     DO I=1,N
        IF(I.NE.K) THEN
           IK=NK+I
           A(IK)=A(IK)/(-BIGA)
        ENDIF
     ENDDO
     !
     !***REDUCE MATRIX
     DO I=1,N
        IK=NK+I
        HOLD=A(IK)
        IJ=I-N
        DO J=1,N
           IJ=IJ+N
           IF(I.NE.K) THEN
              IF(J.NE.K) THEN
                 KJ=IJ-I+K
                 A(IJ)=HOLD*A(KJ)+A(IJ)
              ENDIF
           ENDIF
        ENDDO
     ENDDO
     !
     !***DIVIDE ROW BY PIVOT
     KJ=K-N
     DO J=1,N
        KJ=KJ+N
        IF(J.NE.K) A(KJ)=A(KJ)/BIGA
     ENDDO
     !
     !***PRODUCT OF PIVOTS
     D=D*BIGA
     !
     !***REPLACE PIVOT BY RECIPROCAL
     A(KK)=1.0/BIGA
  ENDDO
  !
  !***FINAL ROW AND COLUMN INTERCHANGE
  DO K=N,1,-1
     I=L(K)
     IF(I.GT.K) THEN
        JQ=N*(K-1)
        JR=N*(I-1)
        DO J=1,N
           JK=JQ+J
           HOLD=A(JK)
           JI=JR+J
           A(JK)=-A(JI)
           A(JI)=HOLD
        ENDDO
     ENDIF
     J=M(K)
     IF(J.GT.K) THEN
        KI=K-N
        DO I=1,N
           KI=KI+N
           HOLD=A(KI)
           JI=KI-K+J
           A(KI)=-A(JI)
           A(JI)=HOLD
        ENDDO
     ENDIF
  ENDDO
  RETURN
END SUBROUTINE MINVST
#else /**/
SUBROUTINE NULL_ST
  RETURN
END SUBROUTINE NULL_ST
#endif /*  NOST2*/


