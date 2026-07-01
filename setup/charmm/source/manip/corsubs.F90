module corsubs
  use chm_kinds
  implicit none
  !-----------------------------------------------------------------------
  !
  !  The AXIS vector from the COOR AXIS command (and others)
  !  Used by the COOR ROTA and COOR TRANs commands (and others)
  !
  !    QAXISC  -  Flag indicating valid axis data
  !    AXISX   -  The X component of the direction vector
  !    AXISY   -  The Y component of the direction vector
  !    AXISZ   -  The Z component of the direction vector
  !    AXISR   -  The magnitude of the direction vector
  !    AXISCX  -  The X value of the vector origin
  !    AXISCY  -  The Y value of the vector origin
  !    AXISCZ  -  The Z value of the vector origin
  !
  LOGICAL,save :: QAXISC
  real(chm_real),save :: AXISX,AXISY,AXISZ,AXISR,AXISCX,AXISCY,AXISCZ

contains
  SUBROUTINE CDIPOLE(NATOM,X,Y,Z,CG,NSEL,IDX,D1,D2,D3,QTOT,OXYZ, &
       LMASS,AMASS)
    !-----------------------------------------------------------------------
    ! Compute dipole moment (D1,D2,D3) and total charge (QTOT) of NSEL atoms
    ! indexed by IDX. If QTOT  /=  0 and ORIENT == TRUE the dipole moment is 
    ! computed using the geometric center of the atoms as origin for the 
    ! coordinate system. Also set substitution variables.
    ! Lennart Nilsson, NOV 96
    !
    ! OXYZ keyword suppresses the origin move for ions
    ! Victor Anisimov, OCT 03
    !      
    ! LMASS keyword moves origin to center of mass (not center of origin)
    ! Tibor Rudas, APR 04
    !
  use consta
  use number
  use param_store, only: set_param

  implicit none

    INTEGER NATOM,NSEL,IDX(NSEL)
    real(chm_real) X(NATOM),Y(NATOM),Z(NATOM),CG(NATOM),D1,D2,D3,QTOT
    LOGICAL OXYZ,LMASS
    real(chm_real) AMASS(*)
    !
    INTEGER I,ID
    real(chm_real) XCM,YCM,ZCM,MASS,TMASS
    !
    QTOT=ZERO
    XCM=ZERO
    YCM=ZERO
    ZCM=ZERO
    D1=ZERO
    D2=ZERO
    D3=ZERO
    TMASS=ZERO
    IF(NSEL  <=  0) RETURN
    DO I=1,NSEL
       QTOT=QTOT+CG(IDX(I))
    ENDDO
    IF(ABS(QTOT) > RSMALL.AND.OXYZ)THEN
       DO I=1,NSEL
          ID=IDX(I)
          MASS=ONE
          IF(LMASS) THEN
             MASS=AMASS(ID)
             TMASS=TMASS+MASS
          ENDIF
          XCM=XCM+(X(ID)*MASS)
          YCM=YCM+(Y(ID)*MASS)
          ZCM=ZCM+(Z(ID)*MASS)
       ENDDO
       MASS=NSEL
       IF(LMASS) MASS=TMASS
       XCM=XCM/MASS
       YCM=YCM/MASS
       ZCM=ZCM/MASS
    ENDIF
    DO I=1,NSEL
       ID=IDX(I)
       D1=D1+(X(ID)-XCM) * CG(ID)
       D2=D2+(Y(ID)-YCM) * CG(ID)
       D3=D3+(Z(ID)-ZCM) * CG(ID)
    ENDDO
    D1=D1*DEBYEC
    D2=D2*DEBYEC
    D3=D3*DEBYEC
    call set_param('XDIP',D1)
    call set_param('YDIP',D2)
    call set_param('ZDIP',D3)
    call set_param('CHARGE',QTOT)
    RETURN
  END SUBROUTINE CDIPOLE

  SUBROUTINE ORINTC(NAT,X,Y,Z,XCOMP,YCOMP,ZCOMP,AMASS,LMASS,LRMS, &
       ATOMIN,ISLCT,LWEIG,WMAIN,LNORO,LPRINT)
    !-----------------------------------------------------------------------
    !  THIS ROUTINE IS CALLED BY THE ORIENT COMMAND AND ROTATES THE
    !  MOLECULE SO THAT IT SITS ON THE ORIGIN WITH NO OFF-DIAGONAL
    !  MOMENTS. LMASS CAUSES A MASS WEIGHTING TO BE DONE.
    !  LRMS=.TRUE. WILL CAUSE THE ROTATION TO BE WRT THE OTHER SET.
    !
    !     By Bernard R. Brooks   1982

  use number
  use stream
  use param_store, only: set_param

  implicit none

    INTEGER NAT
    real(chm_real) X(*),Y(*),Z(*),XCOMP(*),YCOMP(*),ZCOMP(*),WMAIN(*)
    real(chm_real) AMASS(*)
    LOGICAL LMASS,LRMS
    INTEGER ATOMIN(2,*)
    INTEGER ISLCT(*)
    LOGICAL LWEIG,LNORO,LPRINT

    real(chm_real) U(9),RN(3),PHI,EV(3)
    real(chm_real) XC,YC,ZC,XN,YN,ZN
    INTEGER NPR,N,I,J
    LOGICAL LPRNT

    N=NAT
    LPRNT = (LPRINT .AND. PRNLEV >= 2)

    IF(LRMS) THEN
       NPR=0
       DO I=1,N
          IF(ISLCT(I) == 1) THEN
             NPR=NPR+1
             ATOMIN(1,NPR)=I
             ATOMIN(2,NPR)=I
          ENDIF
       ENDDO
       !
       CALL ROTLSQ(XCOMP,YCOMP,ZCOMP,N,X,Y,Z,N,ATOMIN,NPR, &
            LMASS,AMASS,AMASS,LWEIG,WMAIN,LNORO,LPRNT)
       !
    ELSE
       IF(LPRNT) WRITE(OUTU,15)
15     FORMAT(/' ORIENT THE COORDINATES TO ALIGN WITH AXIS'/)
       !
       ! Process best translation
       !
       CALL LSQP2(NAT,X,Y,Z,AMASS,LMASS, &
            ISLCT,LWEIG,WMAIN,LNORO,LPRINT,XC,YC,ZC,U,EV)
       !
       IF(LPRNT) WRITE(OUTU,25) XC,YC,ZC
25     FORMAT(' CENTER OF ATOMS BEFORE TRANSLATION',3F12.5)
       !
       call set_param('XCEN',ZERO)
       call set_param('YCEN',ZERO)
       call set_param('ZCEN',ZERO)
       XC=-XC
       YC=-YC
       ZC=-ZC
       call set_param('XMOV',XC)
       call set_param('YMOV',YC)
       call set_param('ZMOV',ZC)
       DO I=1,N
          X(I)=X(I)+XC
          Y(I)=Y(I)+YC
          Z(I)=Z(I)+ZC
       ENDDO
       !
       IF(LNORO) RETURN
       !
       ! Process best rotation
       !
       CALL FNDROT(U,RN,PHI,LPRNT)
       call set_param('THET',PHI)
       QAXISC=.TRUE.
       AXISCX= ZERO
       AXISCY= ZERO
       AXISCZ= ZERO
       AXISR = ONE
       AXISX = RN(1)
       AXISY = RN(2)
       AXISZ = RN(3)
       call set_param('XAXI',AXISX)
       call set_param('YAXI',AXISY)
       call set_param('ZAXI',AXISZ)
       call set_param('RAXI',AXISR)
       call set_param('XCEN',AXISCX)
       call set_param('YCEN',AXISCY)
       call set_param('ZCEN',AXISCZ)
       !
       DO I=1,N
          XN=U(1)*X(I)+U(2)*Y(I)+U(3)*Z(I)
          YN=U(4)*X(I)+U(5)*Y(I)+U(6)*Z(I)
          ZN=U(7)*X(I)+U(8)*Y(I)+U(9)*Z(I)
          X(I)=XN
          Y(I)=YN
          Z(I)=ZN
       ENDDO
       !
    ENDIF
    !
    RETURN
  END SUBROUTINE ORINTC

  SUBROUTINE LSQP2(NAT,X,Y,Z,AMASS,LMASS,ISLCT,LWEIG,WMAIN, &
       LNORO,LPRINT,XCM,YCM,ZCM,U,EV)
    !-----------------------------------------------------------------------
    !     By Bernard R. Brooks   1982 & 1997
    !
  use number
  use stream

    INTEGER NAT
    real(chm_real) X(*),Y(*),Z(*),AMASS(*),WMAIN(*)
    INTEGER ISLCT(*)
    LOGICAL LMASS,LNORO,LWEIG,LPRINT
    real(chm_real) XCM,YCM,ZCM,U(9),EV(3)
    !
    real(chm_real) SCR(24),AMOM(6),RN(3),PHI
    real(chm_real) AMASST,AM,XX,XY,XZ,YY,YZ,ZZ,DET
    real(chm_real) XC,YC,ZC
    INTEGER N,I,J,IPT
    LOGICAL LPRNT
    !
    N=NAT
    LPRNT = (LPRINT .AND. PRNLEV >= 2)
    !
    ! Calculate the center of mass
    !
    XC=ZERO
    YC=ZERO
    ZC=ZERO
    AMASST=ZERO
    DO I=1,N
       IF(ISLCT(I) == 1) THEN
          IF(LMASS) THEN
             AM=AMASS(I)
          ELSE
             AM=ONE
          ENDIF
          IF(LWEIG) AM=AM*WMAIN(I)
          XC=XC+X(I)*AM
          YC=YC+Y(I)*AM
          ZC=ZC+Z(I)*AM
          AMASST=AMASST+AM
       ENDIF
    ENDDO
    XC=XC/AMASST
    YC=YC/AMASST
    ZC=ZC/AMASST
    XCM=XC
    YCM=YC
    ZCM=ZC
    !
    IF(LNORO) RETURN
    !
    ! Process best rotation calculation
    !
    XX=ZERO
    XY=ZERO
    XZ=ZERO
    YY=ZERO
    YZ=ZERO
    ZZ=ZERO
    DO I=1,N
       IF(ISLCT(I) == 1) THEN
          IF(LMASS) THEN
             AM=AMASS(I)
          ELSE
             AM=1.D0
          ENDIF
          IF(LWEIG) AM=AM*WMAIN(I)
          XX=XX+(X(I)-XC)*(X(I)-XC)*AM
          XY=XY+(X(I)-XC)*(Y(I)-YC)*AM
          XZ=XZ+(X(I)-XC)*(Z(I)-ZC)*AM
          YY=YY+(Y(I)-YC)*(Y(I)-YC)*AM
          YZ=YZ+(Y(I)-YC)*(Z(I)-ZC)*AM
          ZZ=ZZ+(Z(I)-ZC)*(Z(I)-ZC)*AM
       ENDIF
    ENDDO
    !
    AMOM(1)=ZZ
    AMOM(2)=YZ
    AMOM(3)=XZ
    AMOM(4)=YY
    AMOM(5)=XY
    AMOM(6)=XX
    IF(LPRNT) WRITE(OUTU,105) AMOM
105 FORMAT(' MOMENTS'/3F16.8,/16X,2F16.8,/32X,F16.8/)
    !
    CALL DIAGQ(3,3,AMOM,U,SCR(4),SCR(7),SCR(10),SCR(13),EV, &
         SCR(16),SCR(19),SCR(22),0)
    !
    DO I=1,3
       DET=U(I)
       U(I)=U(I+6)
       U(I+6)=DET
    ENDDO
    DO I=1,3
       IPT=(I-1)*3
       DET=U(IPT+1)
       U(IPT+1)=U(IPT+3)
       U(IPT+3)=DET
       IF(U(IPT+I) < ZERO) THEN
          DO J=1,3
             IPT=IPT+1
             U(IPT)=-U(IPT)
          ENDDO
       ENDIF
    ENDDO
    DET=U(1)*(U(5)*U(9)-U(6)*U(8))+U(2)*(U(6)*U(7)-U(4)*U(9))+ &
         U(3)*(U(4)*U(8)-U(5)*U(7))
    IF(DET < ZERO) THEN
       U(7)=-U(7)
       U(8)=-U(8)
       U(9)=-U(9)
       DET=-DET
    ENDIF
    IF(ABS(DET-ONE) > 1.D-4) WRITE(OUTU,203) DET
203 FORMAT(/' ***** WARNING ***** FROM LSQP. ROTATION MATRIX IS', &
         ' NOT UNITARY.'/,' DETERMINANT=',F14.8/)
    IF(LPRNT) WRITE(OUTU,205) U
205 FORMAT(' Transpose of the rotation matrix',3(/1X,3F12.6))
    !
    RETURN
  END SUBROUTINE LSQP2

  SUBROUTINE ROTLSQ(X1,Y1,Z1,NATOM1,X2,Y2,Z2,NATOM2,ATOMPR,NPAIR, &
       LMASS,AMASS1,AMASS2,QWGHT,KWMULT,LNOROT,LPRINT)
    !-----------------------------------------------------------------------
    !     THE PROGRAM ROTATES COORDINATE SET 2 RESULTING IN A 2 SUCH THAT
    !     THE SUM OF THE SQUARE OF THE DISTANCE BETWEEN EACH COORDINATE IN 1
    !     AND 2 IS A MINIMUM.
    !     THE LEAST SQUARE MINIMIZATION IS DONE ONLY WITH RESPECT TO THE
    !     ATOMS REFERRED TO IN THE PAIR ARRAY. THE ROTATION MATRIX THAT IS
    !     CALCULATED IS APPLIED TO THE ENTIRE SECOND COORDINATE SET.
    !     BERNARD R. BROOKS
    !
  use chm_kinds
  use exfunc
  use number
  use memory
  use param_store, only: set_param

    implicit none
    !
    INTEGER NATOM1,NATOM2,NPAIR
    real(chm_real) X1(*),Y1(*),Z1(*),X2(*),Y2(*),Z2(*)
    real(chm_real) AMASS1(*),AMASS2(*)
    real(chm_real) KWMULT(*)
    LOGICAL LMASS,QWGHT,LNOROT,LPRINT
    INTEGER ATOMPR(2,*)
    real(chm_real),allocatable,dimension(:) :: mass
    !
    call chmalloc('rotlsq.src','ROTLSQ','MASS',NPAIR,crl=MASS)
    CALL PKMASS(AMASS1,AMASS2,MASS,ATOMPR,NPAIR,LMASS,QWGHT,KWMULT)
    CALL ROTLS1(X1,Y1,Z1,X2,Y2,Z2,NATOM2,ATOMPR,NPAIR,MASS,LPRINT,LNOROT)
    call chmdealloc('rotlsq.src','ROTLSQ','MASS',NPAIR,crl=MASS)

    RETURN
  END SUBROUTINE ROTLSQ

  SUBROUTINE ROTLS1(XA,YA,ZA,XB,YB,ZB,NATOMB,ATOMPR,NPAIR, &
       BMASS,LPRINTP,LNOROT)
    !-----------------------------------------------------------------------
    !     THIS ROUTINE DOES THE ACTUAL ROTATION OF B TO MATCH A. THE NEW
    !     ARRAY B IS RETURNED IN X.
    !     THIS ROUTINE WRITTEN BY B. BROOKS , ADAPTED FROM ACTA CRYST
    !     (1976) A32,922 W. KABSCH
    !
  use chm_kinds
  use number
  use stream
  use consta
  use param_store, only: set_param

    implicit none
    !
    INTEGER NATOMB,NPAIR
    real(chm_real) BMASS(NPAIR)
    real(chm_real) XA(*),YA(*),ZA(*),XB(*),YB(*),ZB(*)
    INTEGER ATOMPR(2,NPAIR)
    LOGICAL LPRINTP,LNOROT
    !
    !
    real(chm_real) R(9),U(9),EVA(3),DEVA(3,3)
    real(chm_real) RN(3), PHI
    real(chm_real) CMXA,CMYA,CMZA,CMXB,CMYB,CMZB,CMXC,CMYC,CMZC
    real(chm_real) XI,YI,ZI,XJ,YJ,ZJ
    real(chm_real) TMASS,RMST,RMSV,CST,FACT,RSHIFT
    INTEGER K,KA,KB,I
    LOGICAL LPRINT2, QEVW, LPRINT
    !
    LPRINT=LPRINTP
    IF(PRNLEV <= 2)LPRINT=.FALSE.
    !
    LPRINT2=(LPRINT.AND.(PRNLEV > 6))
    !
    CMXA=0.0
    CMYA=0.0
    CMZA=0.0
    CMXB=0.0
    CMYB=0.0
    CMZB=0.0
    TMASS=0.0
    DO K=1,NPAIR
       KA=ATOMPR(1,K)
       KB=ATOMPR(2,K)
       IF (XA(KA) /= ANUM .AND. XB(KB) /= ANUM) THEN
          CMXA=CMXA+XA(KA)*BMASS(K)
          CMYA=CMYA+YA(KA)*BMASS(K)
          CMZA=CMZA+ZA(KA)*BMASS(K)
          CMXB=CMXB+XB(KB)*BMASS(K)
          CMYB=CMYB+YB(KB)*BMASS(K)
          CMZB=CMZB+ZB(KB)*BMASS(K)
          TMASS=TMASS+BMASS(K)
       ENDIF
    ENDDO
    CMXA=CMXA/TMASS
    CMYA=CMYA/TMASS
    CMZA=CMZA/TMASS
    CMXB=CMXB/TMASS
    CMYB=CMYB/TMASS
    CMZB=CMZB/TMASS
    DO K=1,NATOMB
       IF(XB(K) /= ANUM) THEN
          XB(K)=XB(K)-CMXB
          YB(K)=YB(K)-CMYB
          ZB(K)=ZB(K)-CMZB
       ENDIF
    ENDDO
    !
    CMXC=CMXA-CMXB
    CMYC=CMYA-CMYB
    CMZC=CMZA-CMZB
    call set_param('XMOV',CMXC)
    call set_param('YMOV',CMYC)
    call set_param('ZMOV',CMZC)
    IF (LPRINT) THEN
       WRITE(OUTU,44) CMXB,CMYB,CMZB
       WRITE(OUTU,45) CMXA,CMYA,CMZA
       WRITE(OUTU,46) CMXC,CMYC,CMZC
    ENDIF
44  FORMAT(' CENTER OF ATOMS BEFORE TRANSLATION',3F12.5)
45  FORMAT(' CENTER OF REFERENCE COORDINATE SET',3F12.5)
46  FORMAT(' NET TRANSLATION OF ROTATED ATOMS  ',3F12.5)
    !
    IF (LNOROT) THEN
       !
       !       USE A UNIT ROTATION MATRIX. NO ROTATION IS SPECIFIED
       !
       DO K=1,NATOMB
          IF (XB(K) /= ANUM) THEN
             XB(K)=XB(K)+CMXA
             YB(K)=YB(K)+CMYA
             ZB(K)=ZB(K)+CMZA
          ENDIF
       ENDDO
       !
    ELSE
       !
       !       COMPUTE ROTATION MATRIX FROM LAGRANGIAN
       !
       DO I=1,9
          R(I)=0.0
       ENDDO
       DO K=1,NPAIR
          KA=ATOMPR(1,K)
          KB=ATOMPR(2,K)
          IF (XA(KA) /= ANUM .AND. XB(KB) /= ANUM) THEN
             XI=XB(KB)*BMASS(K)
             YI=YB(KB)*BMASS(K)
             ZI=ZB(KB)*BMASS(K)
             XJ=XA(KA)-CMXA
             YJ=YA(KA)-CMYA
             ZJ=ZA(KA)-CMZA
             R(1)=R(1)+XI*XJ
             R(2)=R(2)+XI*YJ
             R(3)=R(3)+XI*ZJ
             R(4)=R(4)+YI*XJ
             R(5)=R(5)+YI*YJ
             R(6)=R(6)+YI*ZJ
             R(7)=R(7)+ZI*XJ
             R(8)=R(8)+ZI*YJ
             R(9)=R(9)+ZI*ZJ
          ENDIF
       ENDDO
       !
       CALL FROTU(R,EVA,DEVA,U,ZERO,QEVW,LPRINT2)
       !
       ! rotate/translate the atoms in set B to match set A.
       DO K=1,NATOMB
          IF (XB(K) /= ANUM) THEN
             CMXC=U(1)*XB(K)+U(4)*YB(K)+U(7)*ZB(K)+CMXA
             CMYC=U(2)*XB(K)+U(5)*YB(K)+U(8)*ZB(K)+CMYA
             ZB(K)=U(3)*XB(K)+U(6)*YB(K)+U(9)*ZB(K)+CMZA
             XB(K)=CMXC
             YB(K)=CMYC
          ENDIF
       ENDDO
       !
       IF (LPRINT) WRITE(OUTU,55) U
55     FORMAT(' ROTATION MATRIX',3(/1X,3F12.6))
       CALL FNDROT(U,RN,PHI,LPRINT)
       !
       ! Compute center of rotation (if it's significant)
       IF(ABS(PHI) >= ONE) THEN
          IF(ABS(PHI) > 179.0) THEN
             CST=MINONE
          ELSE
             CST=COS(DEGRAD*PHI)
          ENDIF
          XI=CMXB-(U(1)*CMXB+U(4)*CMYB+U(7)*CMZB)
          YI=CMYB-(U(2)*CMXB+U(5)*CMYB+U(8)*CMZB)
          ZI=CMZB-(U(3)*CMXB+U(6)*CMYB+U(9)*CMZB)
          XI=XI+CMXA-(U(1)*CMXA+U(2)*CMYA+U(3)*CMZA)
          YI=YI+CMYA-(U(4)*CMXA+U(5)*CMYA+U(6)*CMZA)
          ZI=ZI+CMZA-(U(7)*CMXA+U(8)*CMYA+U(9)*CMZA)
          FACT=HALF/(ONE-CST)
          XI=XI*FACT
          YI=YI*FACT
          ZI=ZI*FACT
          FACT=RN(1)*(CMXA+CMXB)+RN(2)*(CMYA+CMYB)+RN(3)*(CMZA+CMZB)
          XI=XI+HALF*FACT*RN(1)
          YI=YI+HALF*FACT*RN(2)
          ZI=ZI+HALF*FACT*RN(3)
          RSHIFT=RN(1)*(CMXA-CMXB)+RN(2)*(CMYA-CMYB)+RN(3)*(CMZA-CMZB)
          IF (LPRINT) THEN
             WRITE(OUTU,65) XI,YI,ZI,RSHIFT
65           FORMAT(' CENTER OF ROTATION ',3F10.6,'  SHIFT IS',F10.6/)
          ENDIF
       ELSE
          !         don't bother for very small rotations...
          XI=ZERO
          YI=ZERO
          ZI=ZERO
          RSHIFT=ZERO
       ENDIF
       !
       ! Set substitution variables
       call set_param('THET',PHI)
       call set_param('SHIFT',RSHIFT)
       QAXISC=.TRUE.
       AXISCX= XI
       AXISCY= YI
       AXISCZ= ZI
       AXISR = ONE
       AXISX = RN(1)
       AXISY = RN(2)
       AXISZ = RN(3)
       call set_param('XAXI',AXISX)
       call set_param('YAXI',AXISY)
       call set_param('ZAXI',AXISZ)
       call set_param('RAXI',AXISR)
       call set_param('XCEN',AXISCX)
       call set_param('YCEN',AXISCY)
       call set_param('ZCEN',AXISCZ)
    ENDIF
    !
    RMST=0.0
    DO K=1,NPAIR
       KA=ATOMPR(1,K)
       KB=ATOMPR(2,K)
       IF (XA(KA) /= ANUM .AND. XB(KB) /= ANUM) RMST=RMST+ &
            BMASS(K)*((XB(KB)-XA(KA))**2+(YB(KB)-YA(KA))**2+ &
            (ZB(KB)-ZA(KA))**2)
    ENDDO
    RMSV=SQRT(RMST/TMASS)
    call set_param('RMS ',RMSV)
    IF(LPRINT) WRITE(OUTU,14) RMST,TMASS,RMSV
14  FORMAT(' TOTAL SQUARE DIFF IS',F12.4,'  DENOMINATOR IS',F12.4,/ &
         '       THUS RMS DIFF IS',F12.6)
    !
    RETURN
  END SUBROUTINE ROTLS1

  SUBROUTINE FROTU(R,EVA,DEVA,U,EVWID,QEVW,QPRINT)
    !-----------------------------------------------------------------------
    !     THIS ROUTINE SOLVES THE CONSTRAINED MINIMIZATION EQUATION
    !     USING LAGRANGE MULTIPLIERS.
    !     BERNARD R. BROOKS
    !
    ! Input:
    !     R(3,3)  - R matrix    ( R(1,3)=sum(Xai*Zbi*MASSi) )
    !     QPRINT  - Flag indicating a verbose output
    !     EVWID
    ! Output:
    !     U(3,3)  - Resultant rotation matrix.
    !
  use chm_kinds
  use number
  use stream
  use vector
    implicit none
    !
    real(chm_real) R(3,3),EVA(3),DEVA(3,3),U(3,3),EVWID
    LOGICAL QEVW,QPRINT
    !
    real(chm_real) W(6),A(3,3),B(3,3),SCR(24),RTA(3,3)
    real(chm_real) EV(3),RT(3,3),BT(3,3),S(3,3),T(3,3),DSW(3,3)
    !
    INTEGER I,IPT,J,K,JP,JQ,KP,KQ
    real(chm_real) TRACE,EVS,DET,DOT,DRAT,SWVAL,DSWVAL
    !
    QEVW=.FALSE.
    !
    IF(QPRINT) write(6,56) ' The R matrix:',R
56  format(A/,(10X,3F12.5))
    !
    CALL DETM33(R, DET)
    IF(QPRINT) write(6,56) ' The Rdet value:',DET
    !
    IPT=0
    DO I=1,3
       DO J=I,3
          IPT=IPT+1
          W(IPT)=ZERO
          DO K=1,3
             W(IPT)=W(IPT)+R(J,K)*R(I,K)
          ENDDO
       ENDDO
    ENDDO
    !
    IF(QPRINT) write(6,56) ' The W matrix:',W
    !
    ! Handle the case where just one atom is selected.
    TRACE=W(1)+W(4)+W(6)
    IF(TRACE < TENM5) THEN
       EVA  = zero
       DEVA = zero
       U    = zero
       DO I=1,3
          U(I,I)=ONE
       ENDDO
       RETURN
    ENDIF
    !
    CALL DIAGQ(3,3,W,A,SCR(4),SCR(7),SCR(10),SCR(13),EV, &
         SCR(16),SCR(19),SCR(22),0)
    !
    IF(QPRINT) write(6,56) ' The A matrix:',A
    IF(QPRINT) write(6,56) ' The EV matrix:',EV
    !
    DO I=1,3
       EVA(I)=SQRT(ABS(EV(I)))
    ENDDO
    IF(DET < ZERO) EVA(1)=-EVA(1)
    !
    CALL TRANSPS(RT,R,3,3)
    CALL MULNXN(RTA,RT,A,3)
    !
    DO J=1,3
       EVS=ZERO
       IF(ABS(EVA(J)) > TENM5) EVS=ONE/EVA(J)
       DO I=1,3
          B(I,J)=RTA(I,J)*EVS
       ENDDO
    ENDDO
    !
    IF(QPRINT) write(6,56) ' The B matrix:',B
    !
    CALL DETM33(A, DET)
    IF(QPRINT) write(6,56) ' The Adet value:',DET
    !
    ! handle the colinear case (two zero eigenvalues)
    !
    CALL NORMALL(B(1,3),3)
    IF(ABS(EVA(2)) <= TENM5) THEN
       B(1,2)=ZERO
       B(2,2)=ZERO
       B(3,2)=ONE
       CALL ORTHOG(B(1,2),B(1,3),3)
       CALL DOTPR(B(1,2),B(1,2),3,DOT)
       IF(DOT < TENM5) THEN
          B(1,2)=ZERO
          B(2,2)=ONE
          B(3,2)=ZERO
          CALL ORTHOG(B(1,2),B(1,3),3)
       ENDIF
    ENDIF
    CALL NORMALL(B(1,2),3)
    !
    ! handle the coplanar case (one zero eigenvalue)
    !
    IF(ABS(EVA(1)) <= TENM5) THEN
       DO K=1,3
          KP=K+1
          KQ=K+2
          IF(KP > 3) KP=KP-3
          IF(KQ > 3) KQ=KQ-3
          B(K,1)=B(KP,2)*B(KQ,3)-B(KP,3)*B(KQ,2)
          IF(DET < ZERO) B(K,1)=-B(K,1)
       ENDDO
    ENDIF
    CALL NORMALL(B(1,1),3)
    !
    IF(QPRINT) write(6,56) ' The B matrix:',B
    !
    CALL TRANSPS(BT,B,3,3)
    CALL MULNXN(U,A,BT,3)
    !
    IF(QPRINT) write(6,56) ' The U matrix:',U
    !
    ! Check to insure unity (as opposed to anti-unitary)
    CALL DETM33(U, DET)
    IF(ABS(DET-ONE) > TENM5 .AND. WRNLEV > 0) WRITE(OUTU,55) DET
55  FORMAT(/' ***** WARNING ***** FROM FROTU. ROTATION MATRIX IS ', &
         'NOT UNITARY.'/,'  THE DETERMINANT IS',F14.8/)
    !
    deva = u
    !
    !--------------------------------------------------------------------
    ! Adjust the results if near a "fault"
    ! compute both solutions and use a switching function to get
    ! a linear combination.
    !
    IF(EVA(1)+EVA(2) > EVWID) RETURN
    IF(EVA(1)*EVA(2) > 0.0) RETURN
    IF(EVA(2)-EVA(1) < TENM5) RETURN
    !
    QEVW=.TRUE.
    !
    DRAT=(EVWID-EVA(1)-EVA(2))/EVWID
    ! quadratic formula
    SWVAL=HALF*DRAT**2
    DSWVAL= TWO*(EVA(2)+EVA(1)) * DRAT/EVWID
    !L      SWVAL=HALF*DRAT
    !L      DSWVAL= (EVA(2)+EVA(1))/EVWID
    !
    IF(QPRINT) write(6,56) ' The SWVAL value:',SWVAL
    !
    DO J=1,2
       EVS=1.0/EVA(J)
       DO I=1,3
          B(I,J)=RTA(I,J)*EVS
       ENDDO
    ENDDO
    !
    B(1:3,3)=zero
    CALL NORMALL(B(1,2),3)
    CALL NORMALL(B(1,1),3)
    !
    IF(QPRINT) write(6,56) ' The BSW matrix:',B
    !
    CALL TRANSPS(BT,B,3,3)
    CALL MULNXN(DSW,A,BT,3)
    !
    IF(QPRINT) write(6,56) ' The DSW matrix:',DSW
    !
    ! Now average the solutions
    !
    DO I=1,3
       DO J=1,3
          DEVA(I,J) = DEVA(I,J) + DSW(I,J)*(DSWVAL-TWO*SWVAL)
       ENDDO
    ENDDO
    DO I=1,2
       EVA(I)=EVA(I)*(ONE-TWO*SWVAL)
    ENDDO
    !--------------------------------------------------------------------
    !
    RETURN
  END SUBROUTINE FROTU

  SUBROUTINE PKMASS(AMASS1,AMASS2,BMASS,ATOMPR,NPAIR,LMASS, &
       QWGHT,KWMULT)
    !-----------------------------------------------------------------------
    !     THIS ROUTINE PACKS THE BMASS ARRAY WITH MASSES FROM SELECTED
    !     ATOMS. 1.0 IS USED IF LMASS IS FALSE
    !     IF QWGHT IS TRUE THE PRODUCT OF BMASS AND KWMULT IS FORMED AND
    !     STORED IN BMASS
    !
    !     Author: Bernie Brooks
    !
  use chm_kinds
  use number
  use stream
    implicit none
    !
    INTEGER NPAIR
    real(chm_real) AMASS1(*),AMASS2(*)
    real(chm_real) KWMULT(*)
    INTEGER ATOMPR(2,*)
    real(chm_real) BMASS(*)
    LOGICAL LMASS,QWGHT
    INTEGER NWARN,K,KX,KY
    !
    IF (LMASS) THEN
       NWARN=0
       DO K=1,NPAIR
          KX=ATOMPR(1,K)
          KY=ATOMPR(2,K)
          BMASS(K)=AMASS1(KX)
          IF(AMASS1(KX) /= AMASS2(KY)) THEN
             NWARN=NWARN+1
             BMASS(K)=SQRT(AMASS1(KX)*AMASS2(KY))
          ENDIF
       ENDDO
       IF(NWARN > 0 .AND. WRNLEV >= 2) WRITE(OUTU,45) NWARN
    ELSE
       DO K=1,NPAIR
          BMASS(K)=ONE
       ENDDO
    ENDIF
    IF (QWGHT) THEN
       DO K=1,NPAIR
          BMASS(K)=BMASS(K)*KWMULT(ATOMPR(1,K))
       ENDDO
    ENDIF
    !
45  FORMAT(/' *** WARNING *** MASSES DONT MATCH FOR THIS', &
         ' HOMOLOGY FOR',I5,' ATOMS. RESULTS WILL USE GEOMETRIC MEAN.')
    !
    RETURN
  END SUBROUTINE PKMASS

  SUBROUTINE FNDROT(UX,RN,PHI,LPRINT)
    !-----------------------------------------------------------------------
    !     This routine finds the direction and magnitude of a 3x3
    !     rotation matrix
    !     BERNARD R. BROOKS (overhauled Feb. 1998 - BRB)
    !
    !   UX(3,3)  - Input 3x3 rotation matrix.  Checked for Det(UX)=1
    !   RN(3)    - Returned normalized rotation vector (or zero if UX=1)
    !   PHI      - Returned angle of rotation (in degrees)
    !   LPRINT   - Logical flag requesting a print of RN and PHI.
    !
  use chm_kinds
  use vector
  use number
  use stream
  use consta
    !
    implicit none
    !
    real(chm_real) UX(3,3),RN(3),PHI
    LOGICAL LPRINT
    !
    real(chm_real) RX(6),RY(6),U(3,3)
    LOGICAL LOK, QPRINT
    INTEGER I,J
    real(chm_real) DET,FACT
    !
    QPRINT = LPRINT .AND. (PRNLEV > 2)
    !
    ! Test to make sure it is unitary
    !
    CALL DETM33(UX, DET)
    IF(ABS(DET-ONE) > TENM5) THEN
       IF(WRNLEV >= 2) WRITE(OUTU,15) DET
15     FORMAT(' **** WARNING FROM FNDROT **** INPUT MATRIX IS NOT', &
            ' UNITARY. DET=',F12.6)
       RETURN
    ENDIF
    !
    RN(1)=(UX(3,2)-UX(2,3))*HALF
    RN(2)=(UX(1,3)-UX(3,1))*HALF
    RN(3)=(UX(2,1)-UX(1,2))*HALF
    CALL DOTPR(RN,RN,3,FACT)
    IF(FACT < TENM8) THEN
       !     angle of rotation is either 0.0 or 180.0, use cosine method
       RN(1)=SQRT(ABS((ONE+UX(1,1)-UX(2,2)-UX(3,3))*0.5))
       RN(2)=SQRT(ABS((ONE-UX(1,1)+UX(2,2)-UX(3,3))*0.5))
       RN(3)=SQRT(ABS((ONE-UX(1,1)-UX(2,2)+UX(3,3))*0.5))
       IF(UX(2,1)+UX(1,2) < ZERO) RN(2)=-RN(2)
       IF(UX(1,3)+UX(3,1) < ZERO) RN(3)=-RN(3)
       IF(UX(3,2)+UX(2,3) < ZERO .AND. &
            RN(2)*RN(3) > ZERO) RN(3)=-RN(3)
       CALL DOTPR(RN,RN,3,FACT)
       IF(FACT < TENM5) THEN
          IF(QPRINT) WRITE(6,'(A)') &
               ' VERY LITTLE ROTATION: NO AXIS FOUND'
          RN(1:3)=zero
          PHI=ZERO
          RETURN
       ELSE
          CALL NORMALL(RN,3)
          PHI=180.0
       ENDIF
    ELSE
       !       use sine method
       CALL NORMALL(RN,3)
       FACT=SQRT(FACT)
       IF(FACT > ONE) FACT=ONE
       FACT=ASIN(FACT)
       PHI=RADDEG*FACT
       !
       ! Determine if angle>90 degrees
       RY(1)=RN(2)*RN(3)
       RY(2)=RN(1)*RN(3)
       RY(3)=RN(1)*RN(2)
       RY(4)=RN(2)**2+RN(3)**2
       RY(5)=RN(1)**2+RN(3)**2
       RY(6)=RN(1)**2+RN(2)**2
       RX(1)=(UX(3,2)+UX(2,3))*0.5
       RX(2)=(UX(1,3)+UX(3,1))*0.5
       RX(3)=(UX(2,1)+UX(1,2))*0.5
       RX(4)=ONE-UX(1,1)
       RX(5)=ONE-UX(2,2)
       RX(6)=ONE-UX(3,3)
       DO I=1,6
          RX(I)=RX(I)-RY(I)
       ENDDO
       CALL DOTPR(RX,RY,6,FACT)
       IF(FACT > ZERO) PHI=180.0-PHI
    ENDIF
    !
    IF(QPRINT) WRITE(6,35) RN,PHI
35  FORMAT(' AXIS OF ROTATION IS',3F10.6,'  ANGLE IS',F8.2/)
    !
    CALL FNDU(U,RN,PHI,LOK)
    DO I=1,3
       DO J=1,3
          LOK=LOK.AND.(ABS(U(I,J)-UX(I,J)) < TENM5)
       ENDDO
    ENDDO
    IF (.NOT.(LOK)) THEN
       IF(WRNLEV >= 2) WRITE(OUTU,45)
       IF(WRNLEV >= 2) WRITE(OUTU,55) U
    ENDIF
45  FORMAT(/' ** WARNING ** IN FNDROT. ROTATION AXIS POORLY', &
         ' RESOLVED')
55  FORMAT(' RESULTANT ROTATION MATRIX',3(/1X,3F12.6))
    !
    RETURN
  END SUBROUTINE FNDROT

  SUBROUTINE FNDU(U,RN,PHI,LOK)
    !-----------------------------------------------------------------------
    !     This routine finds the unitary transformation matrix
    !     given an axis and an angle about that axis.  It also
    !     normalizes the rotation axis vector.
    !
    !     BERNARD R. BROOKS   (overhauled Feb. 1998 - BRB)
    !
    !   U(3,3)   - Returnted 3x3 rotation matrix.
    !   RN(3)    - Input rotation vector (!!It will be normalized!!)
    !   PHI      - Input angle of rotation (in degrees)
    !   LOK      - Returned logical flag indicating proper function
    !
  use chm_kinds
  use vector
  use number
  use consta
  use stream
    implicit none
    !
    real(chm_real) U(3,3),RN(3),PHI
    LOGICAL LOK
    !
    real(chm_real) PHIR,FACT,T(3,3),T2(3,3),VAL1,VAL2
    INTEGER I,J
    !
    PHIR=PHI*DEGRAD
    FACT=DOTVEC(RN,RN,3)
    IF(ABS(FACT) < TENM8) THEN
       IF(WRNLEV >= 2) WRITE(OUTU,35) FACT
35     FORMAT(/' **** WARNING ***** THE ROTATION AXIS MUST BE', &
            ' SPECIFIED.'/,' NO ROTATION CAN TAKE PLACE',D12.6)
       LOK=.FALSE.
       RETURN
    ENDIF
    !
    CALL NORMALL(RN,3)
    !
    DO I=1,3
       DO J=1,3
          U(I,J)=ZERO
       ENDDO
       U(I,I)=ONE
       T(I,I)=ZERO
    ENDDO
    !
    T(3,2)= RN(1)
    T(2,3)=-RN(1)
    T(1,3)= RN(2)
    T(3,1)=-RN(2)
    T(2,1)= RN(3)
    T(1,2)=-RN(3)
    !
    VAL1=SIN(PHIR)
    VAL2=ONE-COS(PHIR)
    !
    CALL ADDCTV(U,T,9,VAL1)
    CALL MULNXN(T2,T,T,3)
    CALL ADDCTV(U,T2,9,VAL2)
    !
    LOK=.TRUE.
    !
    RETURN
  END SUBROUTINE FNDU

  SUBROUTINE GETCOM(SET,NSET,X,Y,Z,MASS,CH, &
       XREF,YREF,ZREF,QREF)
    !
    !     this routine computes the center of mass of a set of NSET atoms
    !     whose atom numbers are stored in SET. X, Y and Z are the
    !     coordinates. MASS holds the mass of each atom and CH its charge.
    !     c.o.m position is returned in XREF, YREF and ZREF while Qref
    !     holds the sum of charges
    !
    !     Tibor Rudas Jan 2003 - Jun 2003
    !
  use exfunc
  use number
    implicit none
    !
    INTEGER SET(*),NSET
    real(chm_real) X(*),Y(*),Z(*),MASS(*),CH(*)
    real(chm_real) XREF,YREF,ZREF,QREF
    !
    INTEGER I,K
    real(chm_real) SMASS
    !
    XREF=ZERO
    YREF=ZERO
    ZREF=ZERO
    QREF=ZERO
    SMASS=ZERO
    !
    !     check for some special cases
    IF(NSET <= 0) THEN
       !        i.e. we were passed an empty set -> return 0/0/0
       RETURN
    ELSE IF(NSET == 1) THEN
       !        i.e. set contains only one atom -> pass it's coors
       XREF=X(SET(1))
       YREF=Y(SET(1))
       ZREF=Z(SET(1))
       QREF=CH(SET(1))
       RETURN
    ENDIF
    !
    DO I=1,NSET
       K=SET(I)
       XREF=XREF+(X(K)*MASS(K))
       YREF=YREF+(Y(K)*MASS(K))
       ZREF=ZREF+(Z(K)*MASS(K))
       QREF=QREF+CH(K)
       SMASS=SMASS+MASS(K)
    ENDDO
    !
    !     check for missing masses in AMASS (happened once...)
    IF(SMASS <= ZERO) &
         CALL WRNDIE(-5,'<GETCOM>', &
         'AMASS does not contain the atom masses.')

    XREF=XREF/SMASS
    YREF=YREF/SMASS
    ZREF=ZREF/SMASS
    !
    RETURN
  END SUBROUTINE GETCOM
  !
end module corsubs

