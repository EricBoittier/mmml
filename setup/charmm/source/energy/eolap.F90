MODULE EOLAPMOD
  use chm_kinds
  use memory
  IMPLICIT NONE
  !
CONTAINS
  !
#if KEY_OVERLAP==0 /*main_olap*/
  SUBROUTINE OLAPENER(E,DX,DY,DZ)
    real(chm_real) E,DX(*),DY(*),DZ(*)
    CALL WRNDIE(-1,'<OLAP>','OLAP code not compiled.')
    return
  end subroutine olapener
#else /* (main_olap)*/
  SUBROUTINE OLAPENER(E,DX,DY,DZ)
    !----------------------------------------------------------------------
  use olapmod, only:iolap
  use dimens_fcm
  use exfunc
  use number
  use stream
    !
  use olap
  use psf
  use param
  use coord
    implicit none
    !
    !     First this routine calculates all the overlap integrals.
    !
    real(chm_real) E,DX(*),DY(*),DZ(*)
    !
    real(chm_real) NORMWF
    INTEGER NDERS
    real(chm_real), allocatable, dimension(:) :: isx,isy,isz

  NDERS=NATOM*NSYST*(NSYST+1)/2
  call chmalloc("eolap.src","olapener","isx",nders,crl=isx)
  call chmalloc("eolap.src","olapener","isy",nders,crl=isy)
  call chmalloc("eolap.src","olapener","isz",nders,crl=isz)
  !
  E=ZERO
  !
  IF(VOLWF.NE.ZERO) THEN
     NORMWF=VOLWF / (VOLWF+CHAWF+ESPWF)
     CALL OLAPTWO(NATOM,NSYST,NOLAP,IOLAP,MAXINDS,X,Y,Z, &
          WMAIN,VDWR,CESP,WOLAPG,OLAPG,ITC,IAC,QDOLAP,NORMWF,1,NDERS, &
          S,ISX,ISY,ISZ,E,DX,DY,DZ,SUPERW,SYSW)
  ENDIF
  IF(CHAWF.NE.ZERO) THEN
     NORMWF=CHAWF / (VOLWF+CHAWF+ESPWF)
     CALL OLAPTWO(NATOM,NSYST,NOLAP,IOLAP,MAXINDS,X,Y,Z, &
          WMAIN,VDWR,CESP,WOLAPG,OLAPG,ITC,IAC,QDOLAP,NORMWF,2,NDERS, &
          S,ISX,ISY,ISZ,E,DX,DY,DZ,SUPERW,SYSW)
  ENDIF
  IF(ESPWF.NE.ZERO) THEN
     NORMWF=ESPWF / (VOLWF+CHAWF+ESPWF)
     CALL OLAPTWO(NATOM,NSYST,NOLAP,IOLAP,MAXINDS,X,Y,Z, &
          WMAIN,VDWR,CESP,WOLAPG,OLAPG,ITC,IAC,QDOLAP,NORMWF,3,NDERS, &
          S,ISX,ISY,ISZ,E,DX,DY,DZ,SUPERW,SYSW)
  ENDIF
  IF(PRNLEV.GE.2 .AND. QDOLAP)THEN
     WRITE(OUTU,*)'--> E(tot)=',-E
  ENDIF
  call chmdealloc("eolap.src","olapener","isx",nders,crl=isx)
  call chmdealloc("eolap.src","olapener","isy",nders,crl=isy)
  call chmdealloc("eolap.src","olapener","isz",nders,crl=isz)
  RETURN
  !
END SUBROUTINE OLAPENER
!
SUBROUTINE OLAPTWO(NATOM,NSYST,NOLAP,IOLAP,MAXINDS,X,Y,Z, &
     W,R,Q,WOLAPG,GM,ITC,IAC,QDOLAP,NORMWF,IFORMULA,NDERS, &
     S,SX,SY,SZ,E,DX,DY,DZ,SUW,SW)
  !----------------------------------------------------------------------

  use number
  use stream
  !
  !     This routine calculates all the overlap integrals and
  !     their derivatives.
  !
  implicit none
  !
  INTEGER NATOM,NSYST,NOLAP(*),IOLAP(*),MAXINDS,ITC(*),IAC(*)
  INTEGER IFORMULA,NDERS
  real(chm_real) X(*),Y(*),Z(*),W(*),R(*),Q(*),WOLAPG,GM
  real(chm_real) E,NORMWF,SUW
  real(chm_real) S(*),SX(*),SY(*),SZ(*),DX(*),DY(*),DZ(*),SW(*)
  LOGICAL QDOLAP
  !
  INTEGER I,J,K,L,IX,JX,IPT,IP,II,JJ,IJ
  real(chm_real) G,GX,GY,GZ,G2,D,DD,DQ,D_X,D_Y,D_Z
  real(chm_real) T,T1,T2,SII,SJJ,SIJ,TX,TY,TZ
  !
  S (1:NSYST*(NSYST+1)/2) = ZERO
  SX(1:NDERS) = ZERO
  SY(1:NDERS) = ZERO
  SZ(1:NDERS) = ZERO
  !
  DO I = 1, NATOM
     DO J = 1, I
        DO K = NOLAP(I),NOLAP(I+1)-1
           IX=IOLAP(K)
           IF(IX.GT.0) THEN
              DO L = NOLAP(J), NOLAP(J+1)-1
                 JX=IOLAP(L)
                 !     Not only diagonal term for the atoms of the same system!
                 IF(JX.GT.0) THEN
                    IF(IX.GT.JX)THEN
                       IPT=(IX-1)*IX/2+JX
                    ELSE
                       IPT=(JX-1)*JX/2+IX
                    ENDIF
                    !     Get the integrals and their derivatives
                    !     0. Common things
                    D=R(ITC(IAC(I)))**2 + R(ITC(IAC(J)))**2
                    D_X=X(I)-X(J)
                    D_Y=Y(I)-Y(J)
                    D_Z=Z(I)-Z(J)
                    DD=HALF/D
                    DQ=D_X*D_X + D_Y*D_Y + D_Z*D_Z
                    !     1. Volume overlap?
                    IF(IFORMULA.EQ.1) THEN
                       IF(ABS(W(I)*W(I)+W(J)*W(J)).GT.RSMALL) THEN
                          G= W(I)*W(J) * EXP(-DD*DQ) * &
                               (EIGHT*DD*R(ITC(IAC(I)))*R(ITC(IAC(J))))**1.5
                          G2=-TWO*G*DD
                       ELSE
                          G=ZERO
                          G2=ZERO
                       ENDIF
                    ENDIF
                    !     2. Charge overlap?
                    IF(IFORMULA.EQ.2) THEN
                       IF(ABS(Q(I)*Q(I)+Q(J)*Q(J)).GT.RSMALL) THEN
                          G= Q(I)*Q(J) * EXP(-DD*DQ) * &
                               (EIGHT*DD*R(ITC(IAC(I)))*R(ITC(IAC(J))))**1.5
                          G2=-TWO*G*DD
                       ELSE
                          G=ZERO
                          G2=ZERO
                       ENDIF
                    ENDIF
                    !     3. Electrostatic potential overlap?
                    IF(IFORMULA.EQ.3) THEN
                       DQ=SQRT(DQ)
                       IF(DQ.NE.ZERO) THEN
                          G=WOLAPG*Q(I)*Q(J)*EXP(-DQ/GM)
                          G2=-G/(GM*DQ)
                       ELSE
                          G=WOLAPG*Q(I)*Q(J)
                          G2=ZERO
                       ENDIF
                    ENDIF
                    !     If the same atoms have been compared, halve the overlap
                    IF(I.EQ.J)THEN
                       G=G/TWO
                       G2=G2/TWO
                    ENDIF
                    GX=G2*D_X
                    GY=G2*D_Y
                    GZ=G2*D_Z
                    !     Pack the overlap integral and the derivatives
                    S(IPT)=S(IPT)+G
                    IP=(IPT-1)*NATOM+I
                    SX(IP) = SX(IP) + GX
                    SY(IP) = SY(IP) + GY
                    SZ(IP) = SZ(IP) + GZ
                    IP=(IPT-1)*NATOM+J
                    SX(IP) = SX(IP) - GX
                    SY(IP) = SY(IP) - GY
                    SZ(IP) = SZ(IP) - GZ
                 ENDIF
              ENDDO
           ENDIF
        ENDDO
     ENDDO
  ENDDO
  !     Calculate the Hodgkin index and the forces
  DO I=2,NSYST
     DO J=1,I-1
        II=I*(I+1)/2
        JJ=J*(J+1)/2
        IJ=(I-1)*I/2+J
        !     Individual term and overall weighting
        SII=SW(I)*SW(I)*S(II)
        SIJ=SW(I)*SW(J)*S(IJ)
        SJJ=SW(J)*SW(J)*S(JJ)
        T=ZERO
        IF(ABS(SII+SJJ).GT.RSMALL)T=SIJ/(SII+SJJ)
        IF(QDOLAP .AND. PRNLEV.GE.2) WRITE(OUTU,*) &
             'E(',IFORMULA,')=',T,', Derivatives (weighted):'
        E=E-SUW*NORMWF*T
        !     Derivatives
        T1=SII+SJJ
        T2=ZERO
        IF(ABS(T1).GT.RSMALL)T2=SUW*NORMWF/T1/T1
        DO K=1,NATOM
           TX=-T2*(T1*SW(I)*SW(J)*SX((IJ-1)*NATOM+K) &
                -SIJ*(SW(I)*SW(I)*SX((II-1)*NATOM+K) &
                +SW(J)*SW(J)*SX((JJ-1)*NATOM+K)))
           TY=-T2*(T1*SW(I)*SW(J)*SY((IJ-1)*NATOM+K) &
                -SIJ*(SW(I)*SW(I)*SY((II-1)*NATOM+K) &
                +SW(J)*SW(J)*SY((JJ-1)*NATOM+K)))
           TZ=-T2*(T1*SW(I)*SW(J)*SZ((IJ-1)*NATOM+K) &
                -SIJ*(SW(I)*SW(I)*SZ((II-1)*NATOM+K) &
                +SW(J)*SW(J)*SZ((JJ-1)*NATOM+K)))
           DX(K)=DX(K)+TX
           DY(K)=DY(K)+TY
           DZ(K)=DZ(K)+TZ
           IF(QDOLAP .AND. PRNLEV.GE.2) WRITE(OUTU,*) &
                'Atom: ',K,', DX=',TX,', DY=',TY, ', DZ=',TZ
        ENDDO
     ENDDO
  ENDDO
  RETURN
END SUBROUTINE OLAPTWO
!
#endif /* (main_olap)*/

END MODULE EOLAPMOD

