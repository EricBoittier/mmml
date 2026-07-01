#if KEY_ADUMB==1 /*adumb*/
!
! Routines specific for specific energy terms
!
SUBROUTINE UM1PHI(IU,JU,KU,LU,INDEX,ICRD,NPHI,X,Y,Z)
  !
  !     CALCULATES TORSION ANGLES AND CORESPONDING INTERNAL COORDINATE
  !     FOR UMBRELLA POTENTIAL
  !
  !     The parameters of the routine are:
  !     IP,JP,KP,LP: Matrices of atom number for the members of dihedrals
  !     ICRD: Internal coordinate
  !     X,Y,Z: Coordinate matrices
  !
  !
  !     Adapted from EPHI by C. Bartels, 1996
  !
  !
  use chm_kinds
  use dimens_fcm
  use number
  use consta
  use stream
  !
  implicit none
  real(chm_real) ICRD(*)
  INTEGER IU(*),JU(*),KU(*),LU(*),INDEX(*),NPHI
  real(chm_real) X(*),Y(*),Z(*)
  !
  LOGICAL QAFIRST
  INTEGER NWARN,I,J,K,L,IPHI
  real(chm_real) FX,FY,FZ,GX,GY,GZ,HX,HY,HZ,AX,AY,AZ,BX,BY,BZ
  real(chm_real) RA2,RB2,RG2,RG,RA2R,RB2R,RGR,RABR,CP,SP,AP
  !
  real(chm_real)  RXMIN,RXMIN2
  PARAMETER (RXMIN=0.005D0,RXMIN2=0.000025D0)
  !
  NWARN=0
  QAFIRST=.TRUE.
  !
  DO IPHI=1,NPHI
     I=IU(IPHI)
     J=JU(IPHI)
     K=KU(IPHI)
     L=LU(IPHI)
     !
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
     IF((RA2 <= RXMIN2).OR.(RB2.LE.RXMIN2).OR.(RG.LE.RXMIN)) THEN
        NWARN=NWARN+1
        IF((NWARN <= 5 .AND. WRNLEV >= 5) .OR. WRNLEV.GE.6) THEN
           IF(PRNLEV > 2)WRITE(OUTU,20) IPHI,I,J,K,L
20         FORMAT(' EPHI: WARNING.  dihedral',I5,' is almost linear.'/ &
                ' derivatives may be affected for atoms:',4I5)
        ENDIF
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
     IF (CP > PTONE ) THEN
        AP=ASIN(SP)
     ELSE
        AP=SIGN(ACOS(MAX(CP,MINONE)),SP)
     ENDIF
     ! Internal coordinat in range [0,1)
     ICRD(INDEX(IPHI))=(AP+PI)/TWOPI
     !
  enddo
  !
  IF(NWARN > 1 .AND. WRNLEV >= 2.AND.PRNLEV.GT.2) &
       WRITE(OUTU,170) NWARN
170 FORMAT(' TOTAL OF',I6,' WARNINGS FROM EPHI')
  !
  RETURN
END SUBROUTINE UM1PHI
!
!--------------------------------------------------------------------
SUBROUTINE UM2PHI(EP,IP,JP,KP,LP,INDEX,NPHI,DNT, &
     DX,DY,DZ,X,Y,Z,QECONTX,ECONTX)
  !
  !     CALCULATES EITHER TORSION ANGLES ENERGIES FROM UMBRELLA POTENTIALS. 
  !     FIRST DERIVATIVES ARE ADDED TO DX, DY, DZ.
  !     NO SECOND DERIVATIVES ARE AVAILABLE.
  !
  !     ENERGY TERMS ARE EXPRESSED AS A FUNCTION OF PHI TO AVOID
  !     ALL PROBLEMS AS DIHEDRALS BECOME PLANAR.
  !
  use chm_kinds
  use dimens_fcm
  use number
  use consta
  use stream
  !
  implicit none
  real(chm_real) EP
  INTEGER IP(*),JP(*),KP(*),LP(*),INDEX(*)
  INTEGER NPHI
  real(chm_real) DNT(*),DX(*),DY(*),DZ(*),X(*),Y(*),Z(*)
  LOGICAL QECONTX
  real(chm_real) ECONTX(*)
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
  !
  real(chm_real)  RXMIN,RXMIN2
  PARAMETER (RXMIN=0.005D0,RXMIN2=0.000025D0)
  !
  EP=ZERO
  IF(NPHI <= 0) RETURN
  NWARN=0
  NWARNX=0
  QAFIRST=.TRUE.
  !
  loop10: DO IPHI=1,NPHI
     !
     I=IP(IPHI)
     J=JP(IPHI)
     K=KP(IPHI)
     L=LP(IPHI)
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
     IF((RA2 <= RXMIN2).OR.(RB2.LE.RXMIN2).OR.(RG.LE.RXMIN)) THEN
        NWARN=NWARN+1
        IF((NWARN <= 5 .AND. WRNLEV >= 5) .OR. WRNLEV.GE.6) THEN
           IF(PRNLEV > 2)WRITE(OUTU,20) IPHI,I,J,K,L
20         FORMAT(' EPHI: WARNING.  dihedral',I5,' is almost linear.'/ &
                ' derivatives may be affected for atoms:',4I5)
        ENDIF
        cycle loop10
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
     E=ZERO
     DF= -DNT(INDEX(IPHI))/TWOPI
     !          WRITE(OUTU,180) IPHI,DF
     !  180        FORMAT(' Int. Crd. ',I2,' force on angle ',G15.5)
     DDF=ZERO
     ! Set the energy.
     EP=EP+E
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
  enddo loop10
  !
  NWARN=NWARN+NWARNX
  IF(NWARN > 5.AND.WRNLEV >= 2.AND.PRNLEV.GT.2) &
       WRITE(OUTU,170) NWARN
170 FORMAT(' TOTAL OF',I6,' WARNINGS FROM EPHI')
  !
  RETURN
END SUBROUTINE UM2PHI
!
!-----------------------------------------------------------------------
!
SUBROUTINE UM1EN(E,N)
  !
  !     CALCULATES INTERNAL COORDINATE FROM POTENTIAL ENERGY CONTRIBUTIONS
  !
  !     The parameters of the routine are:
  !     E: energy contributions
  !     N: number of energy contributions
  !
  !     C. Bartels, 1996
  !
  !
  use chm_kinds
  use dimens_fcm
  use number
  use umb
  use stream
  !
  implicit none
  real(chm_real)  E(*)
  INTEGER N
  !
  real(chm_real)  ETOT
  INTEGER I
  !
  ! sum up energy terms
  ETOT=0.0
  DO I=1,N
     ETOT=ETOT+E(I)
  enddo
  ! Internal coordinate such that [EMINUM,EMAXUM) --> [1/4,3/4)
  COUM(IEUM)=(ETOT-EMINUM)/((EMAXUM-EMINUM)*2.0)+0.25
  !
  RETURN
END SUBROUTINE UM1EN
!
!-----------------------------------------------------------------------
!
SUBROUTINE UM2EN(NA,DX,DY,DZ)
  !
  !     CALCULATES FORCES FOR ENERGY UMBRELLA
  !
  !     The parameters of the routine are:
  !     DX,DY,DZ: forces onto atoms
  !     NA      : number of atoms
  !
  !     C. Bartels, 1996
  !
  !
  use chm_kinds
  use dimens_fcm
  use number
  use umb
  !
  implicit none
  real(chm_real)  DX(*),DY(*),DZ(*)
  INTEGER NA
  !
  real(chm_real)  F
  INTEGER I
  !
  F=1.0-DUM(IEUM)/((EMAXUM-EMINUM)*2.0)
  DX(1:na)=DX(1:na)*F
  DY(1:na)=DY(1:na)*F
  DZ(1:na)=DZ(1:na)*F
  !
  RETURN
END SUBROUTINE UM2EN
!
!-----------------------------------------------------------------------
! New type of umbrella coordinate on which we have currently a project running.
! Thus, please do not use. C. Bartels, August 1998
!
SUBROUTINE UM1NOE(NAT,X,Y,Z)
  !
  !     CALCULATES INTERNAL COORDINATE FROM HARMONIC CONSTRAINT UMBRELLA
  !
  !     The parameters of the routine are:
  !     NAT:   number of atoms
  !     X,Y,Z: atom coordinates
  !
  !     C. Bartels, 1998
  !
  use chm_kinds
  use dimens_fcm
  use number
  use umb
  use stream
#if KEY_PNOE==1
  use noem  
#endif
  !
  !
  implicit none
  INTEGER NAT
  real(chm_real)  X(NAT),Y(NAT),Z(NAT)
  INTEGER I,IDUMMY(1)
  real(chm_real)  RDUMMY(1),TOTE,ECONT,ENOE
  !
  TOTE=0.0
  DO I=1,NNUM
     ENOE=0.0
     !       initialize forces
     CALL UM3NOE(NAT, &
          STFNUM(I)%x,STFNUM(I)%y,STFNUM(I)%z)
     !       calc energy and forces for each NOE data set
     IF(NUMNUM(I) > 0) THEN
#if KEY_PNOE==1
        !     Do not know why PNOE was set to FALSE by default in the
        !     original version, which would allow only a single point NOE.
        !     Changed to the actual value of the IsPNOE array (TRUE or FALSE
        !     for each NOE). Easiest way to make compatible.
        !     M. Schaefer, Dec 2004.
#endif 
        CALL NOECNS(ENOE, &
             STFNUM(I)%x,STFNUM(I)%y,STFNUM(I)%z, &
             X,Y,Z,.FALSE.,RDUMMY, &
             NUMNUM(I),SCANUM(I), &
             IPTNUM(I),JPTNUM(I),INMNUM(I), &
             JNMNUM(I),LISNUM(I),EXPNUM(I), &
             RMNNUM(I),KMNNUM(I),RMXNUM(I), &
             KMXNUM(I),FMXNUM(I),TCNNUM(I), &
             AVENUM(I),MINNUM(I), &
             RDUMMY,IDUMMY,.FALSE. &
                                !JH (soft asymptote)
             ,RSWNUM(I),SEXNUM(I),RAMNUM(I) &
#if KEY_PNOE==1
             , IsPNOE, C0X, C0Y, C0Z &
             , MVPNOE,OC0X,OC0Y,OC0Z &
             ,TC0X,TC0Y,TC0Z &
             , NMPNOE, IMPNOE &
#endif 
        )

     ENDIF
     ENUM(I)=ENOE/NORMNU(I)
     ECONT=ENUM(I)/(ENUM(I)+ELIMNU(I))
     TOTE=TOTE+ECONT
  ENDDO
  ! Internal coordinate such that [EMINNU,EMAXNU) --> [0,1)
  COUM(INUM)=(TOTE-EMINNU)/(EMAXNU-EMINNU)
  !
  RETURN
END SUBROUTINE UM1NOE
!
!-----------------------------------------------------------------------
! New type of umbrella coordinate on which we have currently a project running.
! Thus, please do not use. C. Bartels, August 1998
!
SUBROUTINE UM2NOE(NAT,DX,DY,DZ)
  !
  !     CALCULATES INTERNAL COORDINATE FROM HARMONIC CONSTRAINT UMBRELLA
  !
  !     The parameters of the routine are:
  !     NAT:   number of atoms
  !     X,Y,Z: atom coordinates
  !
  !     C. Bartels, 1998
  !
  !
  use chm_kinds
  use dimens_fcm
  use number
  use umb
  use stream
  !
  implicit none
  INTEGER NAT
  real(chm_real)  DX(NAT),DY(NAT),DZ(NAT)
  !
  INTEGER I
  real(chm_real)  F1,F2,F3,TMP
  !
  F1=-DUM(INUM)/(EMAXNU-EMINNU)
  DO I=1,NNUM
     F2=F1/NORMNU(I)
     TMP=1.0/(ENUM(I)+ELIMNU(I))
     F3=F2*TMP*(1-ENUM(I)*TMP)
     !        add forces
     CALL UM4NOE(NAT,F3,DX,DY,DZ, &
          STFNUM(I)%x,STFNUM(I)%y,STFNUM(I)%z)
  ENDDO
  !
  RETURN
END SUBROUTINE UM2NOE
!
!-----------------------------------------------------------------------
! New type of umbrella coordinate on which we have currently a project running.
! Thus, please do not use. C. Bartels, August 1998
!
SUBROUTINE UM3NOE(NAT,FX,FY,FZ)
  !
  !     Initialize force arrays
  !
  !     The parameters of the routine are:
  !     NAT:   number of atoms
  !     FX,FY,FZ: force arrays
  !
  !     C. Bartels, 1998
  !
  !
  use chm_kinds    
  use dimens_fcm
  use number
  !
  implicit none
  INTEGER NAT
  real(chm_real)  FX(NAT),FY(NAT),FZ(NAT)
  !
  INTEGER I
  !
  DO I=1,NAT
     FX(I)=0.0
     FY(I)=0.0
     FZ(I)=0.0
  ENDDO
  !
  RETURN
END SUBROUTINE UM3NOE
!
!
!-----------------------------------------------------------------------
! New type of umbrella coordinate on which we have currently a project running.
! Thus, please do not use. C. Bartels, August 1998
!
SUBROUTINE UM4NOE(NAT,F,DX,DY,DZ,FX,FY,FZ)
  !
  !     add umbrella forces
  !
  !     The parameters of the routine are:
  !     NAT:   number of atoms
  !     FX,FY,FZ: noe force arrays
  !     F: factor to multiply forces
  !     DX,DY,DZ: arrays to accumulate forces
  !
  !     C. Bartels, 1998
  !
  !
  use chm_kinds
  use dimens_fcm
  use number
  !
  implicit none
  INTEGER NAT
  real(chm_real)  FX(NAT),FY(NAT),FZ(NAT),DX(NAT),DY(NAT),DZ(NAT),F
  !
  INTEGER I
  !
  DO I=1,NAT
     DX(I)=DX(I)+F*FX(I)
     DY(I)=DY(I)+F*FY(I)
     DZ(I)=DZ(I)+F*FZ(I)
  ENDDO
  !
  return
end SUBROUTINE UM4NOE

#if KEY_ADUMBRXNCOR==1 /*adumbrxncor*/
      SUBROUTINE UM1RXN(icrd,X,Y,Z)
!
!     Interface from ADUMB to RXNCOR to permit adaptive umbrella sampling on coordinates defined with RXNCOR
!     
!
!     The parameters of the routine are:
!     ICRD: Internal coordinate
!     X,Y,Z: Coordinate matrices
!
!
!     based on subroutines RXNENE and UM1PHI, by J. Spiriti, 2010
!
!
      use chm_kinds
      use dimens_fcm
      use rxncom
      use number
      use consta
      use stream
      use umb
      use rxenemod,only: ascend
      implicit none
!
      real(chm_real) :: ICRD(*),a,b
      INTEGER :: irxncor, ir, index
      real(chm_real)  X(*),Y(*),Z(*)
!
      LOGICAL QAFIRST
      INTEGER NWARN
      real*8 value
!
      NWARN=0
      QAFIRST=.TRUE.
!
      call ascend !I think this determines the values of reaction coordinates
      DO 10 irxncor=1,numbrxn
        ir=rxncorindex(irxncor) !which reaction coordinate in RXNCOR's list?
        index=treelo(ir) !which node number?
        value=delval(1,index)
        a=umbmin(irxncor)
        b=umbmax(irxncor)-a
        if ((value.lt.a).or.(value.gt.b)) call wrndie(-2,'<um1rxn>',&
          'reaction coordinate out of range')
! Internal coordinat in range [0,1)
        ICRD(umbindex(irxncor))=(value-a)/b
!
   10 CONTINUE
!This enables the use of rxncor trace under these conditions.  But ADUMB's statistics are used for wham, not RXNCOR's.
      if ((umbmdstep  /=  old_mdstep) .or. (umbmdstep == 0)) then
          old_mdstep =  umbmdstep
  
!          call rxnstt(delval,rxncnt,sttstp,delstp,nmlstt, &
!            lodel,hidel,deldel,treelo, &
!            nrxncr,rxntrn,trcfrq,rxntrc,trunit,deftyp)
     
      end if

      RETURN
      END SUBROUTINE UM1RXN

      SUBROUTINE UM2rxn(DNT,DX,DY,DZ,X,Y,Z)
!
!     CALCULATES EITHER TORSION ANGLES ENERGIES FROM UMBRELLA POTENTIALS. 
!     FIRST DERIVATIVES ARE ADDED TO DX, DY, DZ.
!     NO SECOND DERIVATIVES ARE AVAILABLE.
!
!     ENERGY TERMS ARE EXPRESSED AS A FUNCTION OF PHI TO AVOID
!     ALL PROBLEMS AS DIHEDRALS BECOME PLANAR.
!
      use chm_kinds
      use dimens_fcm
      use number
      use consta
      use stream
      use rxncom
      use umb
      use rxenemod,only:decend
      implicit none
!
      REAL(chm_real) DNT(*),DX(*),DY(*),DZ(*),X(*),Y(*),Z(*)
      real(chm_real) e,df
!
      integer irxncor, ir, index
      LOGICAL LREP,NOCONS,QAFIRST
!
      REAL*8  RXMIN,RXMIN2
      PARAMETER (RXMIN=0.005D0,RXMIN2=0.000025D0)
!
      QAFIRST=.TRUE.
!
      DO 10 irxncor=1,numbrxn
!
!          ir=rxncorindex(irxncor)          
          ir=rxncorindex(irxncor) !which reaction coordinate in RXNCOR's list?
          index=treelo(ir) !which node number?
 
!we need to work out these derivatives and then stick them into delder.                    
          E=ZERO
          DF= -DNT(umbindex(irxncor))/(umbmax(irxncor)-umbmin(irxncor))
          delder(1,index)=df
!the call to DECEND will work out the remainder of the derivatives and add onto the forces for this reaction coordinate only
          call decend(ir,umbfrm,kumbpr,dl0ptr)

!          CALL DECEND(DELDER,DELVAL,DEFTYP,REFNOD,NGRAPH,IR,
!     -                HEAP(TREELO),HEAP(TREEHI),HEAP(UMBFRM),
!     -                HEAP(KUMBPR),HEAP(DL0PTR))        
!          WRITE(OUTU,180) IPHI,DF
!  180        FORMAT(' Int. Crd. ',I2,' force on angle ',G15.5)
! Set the energy.
!
  160   CONTINUE
   10 CONTINUE
!
!
      RETURN
      END SUBROUTINE UM2RXN


#endif /* (adumbrxncor)*/


#endif /* (adumb)*/

SUBROUTINE NULL_UM1
  RETURN
END SUBROUTINE NULL_UM1

