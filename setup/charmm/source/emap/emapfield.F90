#if KEY_EMAP==1 /*emap*/
!  Subroutines for molecular modeling with map objects
!                         By XIONGWU WU  Nov, 2004
!
!   1. Using map objects to represent large macromolecules can be more
!   efficient than using all-atom representations.
!   2. Interaction between map objects is handled by field map objects. The 
!   interaction between a pair of map objects can be thought as a map object
!   interacts with the field of another object.
!   3. A field map object describes the eletrostatic potential throughout the 
!   space.  It covers the whole space from -infinity to infinity using a 
!   reduced coordinate.
!             X=x/(x+a),Y=y/(y+b),Z=z/(z+c)
!   These reduced coordinated take values:  X:(-1,1) Y:(-1,1) Z:(-1,1)  
!   Parameter (a,b,c) represent the dimension of real molecules.
!   4. Molecular map objects describe the charge distribution in the space.
!   5. Core-indexes are used to mark the core and surface of the molecule.
!   6. Interactions are catagoried based on core-indeces:
!         Assume molecular map object 1 interacts with field map object 2.
!         C1(i) and C2(i) represent the core-indeces of grid point i.
!         C1(i)=0    No interaction
!         C1(i)>0 and C2(i)=0  Electrostatic: e1(i)F2(i)
!                              e1 charge, F2 potential
!         C1(i)>0 and C2(i)=1  Solvation of object 2:  s2*e2**2  (s--solvation
!         C1(i)=1 and C2(i)>0  Solvation of object 1:  s1*e1**2      parameter)
!         C1(i)=1 and C2(i)=1  vdw attraction     -p1-p2   (p-potential well)
!         C1(i)>1 and C2(i)>1  vdw repulsive     p1p2(C1(i)C2(i))**m
!         
!
SUBROUTINE EMAPFGEN(NAT,X,Y,Z,AMASS,CG,ISLCT, &
     IDEMP,DX,DY,DZ,AX,AY,AZ,LGRIDCG)
  !-----------------------------------------------------------------------
  !
  !   This routine generate a electrostatic field map from a set of atoms
  !     Each grid point presents a electrostatic potential if it is outside
  !     of the core or an electric charge if it is inside the core.
  !     The potential is the sum of that from all atoms: phi=SUM(ei/ri)
  !     The charge is the sum of that from all atoms in the neighboring
  !       lattic: e(i,j,k)=SUM(ei(x2-x)(y2-y)(z2-z)/(x2-x1)/(y2-y1)/(z2-z1))
  !
  use chm_kinds
  use memory
  use dimens_fcm
  use number
  use stream
  use exfunc
  use emapmod
  implicit none
  !
  real(chm_real4),allocatable,dimension(:) :: CGTMP
  LOGICAL LGRIDCG
  INTEGER IDEMP,NX,NY,NZ,NAT,ISLCT(*)
  real(chm_real) EPS,CG(*),X(*),Y(*),Z(*),AMASS(*)
  real(chm_real) DX,DY,DZ,AX,AY,AZ,CGSUM
  real(chm_real) XI,YI,ZI,CX,CY,CZ
  INTEGER I,NGRID,ICORE,NATI
  INTEGER IX,IY,IZ,MNX,MNY,MNZ,MXX,MXY,MXZ
  !
  !  Calculate selected atoms
  MNX=100000
  MNY=100000
  MNZ=100000
  MXX=-100000
  MXY=-100000
  MXZ=-100000
  NATI=0
  CGSUM=ZERO
  CX=ZERO
  CY=ZERO
  CZ=ZERO
  loop100: DO I=1,NAT
     IF(ISLCT(I) /= 1) cycle loop100
     NATI=NATI+1
     CGSUM=CGSUM+CG(I)
     XI=X(I)
     YI=Y(I)
     ZI=Z(I)
     CX=CX+XI
     CY=CY+YI
     CZ=CZ+ZI
     IX=NINT(XI/DX)
     IY=NINT(YI/DY)
     IZ=NINT(ZI/DZ)
     IF(MNX > IX)MNX=IX
     IF(MNY > IY)MNY=IY
     IF(MNZ > IZ)MNZ=IZ
     IF(MXX < IX)MXX=IX
     IF(MXY < IY)MXY=IY
     IF(MXZ < IZ)MXZ=IZ
  enddo loop100

  CX=CX/NATI
  CY=CY/NATI
  CZ=CZ/NATI
  CGSUM=CGSUM/NATI
  NX=2*INT((MXX-MNX+2)/2)+3
  NY=2*INT((MXY-MNY+2)/2)+3
  NZ=2*INT((MXZ-MNZ+2)/2)+3
  NGRID=NX*NY*NZ
  call chmalloc('emapfield.src','EMAPFGEN','empcrd(IDEMP)%x ',NATI,crlp=empcrd(IDEMP)%x )
  call chmalloc('emapfield.src','EMAPFGEN','empcrd(IDEMP)%y ',NATI,crlp=empcrd(IDEMP)%y )
  call chmalloc('emapfield.src','EMAPFGEN','empcrd(IDEMP)%z ',NATI,crlp=empcrd(IDEMP)%z )
  NATEMAP(IDEMP)=NATI
  call chmalloc('emapfield.src','EMAPFGEN','empgrd(IDEMP)%rho',NGRID, &
       cr4p=empgrd(IDEMP)%rho)
  call chmalloc('emapfield.src','EMAPFGEN','empgrd(idemp)%core',NGRID, &
       intgp=empgrd(IDEMP)%core)
  call chmalloc('emapfield.src','EMAPFGEN','empgrd(IDEMP)%ddrho',NGRID, &
       cr4p=empgrd(IDEMP)%ddrho)
  CALL EMAPINIT(IDEMP,ZERO)
  IF(MEMAPX(IDEMP)==-999999)MEMAPX(IDEMP)=MNX-2
  IF(MEMAPY(IDEMP)==-999999)MEMAPY(IDEMP)=MNY-2
  IF(MEMAPZ(IDEMP)==-999999)MEMAPZ(IDEMP)=MNZ-2
  IF(LEMAPX(IDEMP)==-999999)LEMAPX(IDEMP)=NX
  IF(LEMAPY(IDEMP)==-999999)LEMAPY(IDEMP)=NY
  IF(LEMAPZ(IDEMP)==-999999)LEMAPZ(IDEMP)=NZ
  DEMAPX(IDEMP)=DX
  DEMAPY(IDEMP)=DY
  DEMAPZ(IDEMP)=DZ
  AEMAPX(IDEMP)=AX
  AEMAPY(IDEMP)=AY
  AEMAPZ(IDEMP)=AZ
  CEMAPX(IDEMP)=CX
  CEMAPY(IDEMP)=CY
  CEMAPZ(IDEMP)=CZ
  !  Create temperary storage array
  call chmalloc('emapfield.src','EMAPFGEN','CGTMP',NATI,cr4=CGTMP)

  CALL FIELD2EMP(LGRIDCG,NAT,EMDIELC,CGSUM,CG,X,Y,Z,AMASS,ISLCT, &
       MEMAPX(IDEMP),MEMAPY(IDEMP),MEMAPZ(IDEMP), &
       LEMAPX(IDEMP),LEMAPY(IDEMP),LEMAPZ(IDEMP), &
       DEMAPX(IDEMP),DEMAPY(IDEMP),DEMAPZ(IDEMP), &
       AEMAPX(IDEMP),AEMAPY(IDEMP),AEMAPZ(IDEMP), &
       NCREMAP(IDEMP),NATEMAP(IDEMP),CGTMP, &
       empcrd(IDEMP)%x,empcrd(IDEMP)%y,empcrd(IDEMP)%z, &
       empgrd(IDEMP)%rho,empgrd(IDEMP)%ddrho,empgrd(IDEMP)%core)
  IF(PRNLEV > 2)THEN
     write(outu,1010)IDEMP
     WRITE(OUTU,1020)MEMAPX(IDEMP),MEMAPY(IDEMP),MEMAPZ(IDEMP)
     WRITE(OUTU,1030)LEMAPX(IDEMP),LEMAPY(IDEMP),LEMAPZ(IDEMP)
     WRITE(OUTU,1040)DEMAPX(IDEMP),DEMAPY(IDEMP),DEMAPZ(IDEMP)
     WRITE(OUTU,1050)AEMAPX(IDEMP),AEMAPY(IDEMP),AEMAPZ(IDEMP)
     WRITE(OUTU,1060)NGRID,NCREMAP(IDEMP),NATEMAP(IDEMP)
  ENDIF
  call chmdealloc('emapfield.src','EMAPFGEN','CGTMP',NATI,cr4=CGTMP)
1010 FORMAT("Field map ",I3," is built from protein structure")
1020 FORMAT("MX,MY,MZ= ",3I6)
1030 FORMAT("LX,LY,LZ= ",3I6)
1040 FORMAT("DX,DY,DZ= ",3F10.4)
1050 FORMAT("AX,AY,AZ= ",3F10.4)
1060 FORMAT("NGRID= ",I8,"    NCORE= ",I8,"    NEMATM= ",I8)
1070 FORMAT("Average mass density is= ",F15.8)
  RETURN
END SUBROUTINE EMAPFGEN


SUBROUTINE FIELD2EMP(LGRIDCG,NAT,EPS,CGSUM,CG,X,Y,Z, &
     AMASS,ISLCT,MSX,MSY,MSZ,NX,NY,NZ, &
     DX,DY,DZ,AX,AY,AZ, &
     NCORE,NATI,CGR,XR,YR,ZR,RHO,DDR,CORE)
  !-----------------------------------------------------------------------
  !  This routine  generat filed from coordinates
  !                        
  use chm_kinds
  use dimens_fcm
  use number
  use consta
  implicit none
  LOGICAL LGRIDCG
  INTEGER MSX,MSY,MSZ,NX,NY,NZ,NAT,NATI,NGRID,NCORE
  INTEGER ISLCT(*),CORE(*)
  real(chm_real) X(*),Y(*),Z(*),XR(*),YR(*),ZR(*),AMASS(*),CG(*)
  real(chm_real) DX,DY,DZ,DX1,DY1,DZ1,AX,AY,AZ,EPS,CGSUM
  REAL*4 RHO(*),DDR(*),CGR(*)
  real(chm_real) XS,YS,ZS,XI,YI,ZI,XJ,YJ,ZJ,XIJ,YIJ,ZIJ,RIJ,EELE, &
       CGF,CGI
  real(chm_real) DTX1,DTY1,DTZ1,DTX,DTY,DTZ
  INTEGER IX,IY,IZ,IX1,IY1,IZ1
  INTEGER I
  INTEGER IXYZ000,IXYZ100,IXYZ010,IXYZ110,IXYZ001,IXYZ101
  INTEGER IXYZ011,IXYZ111,IXYZ,IXYZ1
  INTEGER NXYZMAP
  NGRID=NX*NY*NZ
  XS=MSX*DX
  YS=MSY*DY
  ZS=MSZ*DZ
  DO I=1,NGRID
     RHO(I)=ZERO
     DDR(I)=ZERO
     CORE(I)=0
  END DO
  ! interpolate charges to protein field map
  NATI=0
  loop200: DO I=1,NAT
     IF(ISLCT(I) /= 1) cycle loop200
     NATI=NATI+1
     CGI=CG(I) 
     XI=X(I)
     YI=Y(I)
     ZI=Z(I)
     CGR(NATI)=CGI
     XR(NATI)=XI
     YR(NATI)=YI
     ZR(NATI)=ZI
     IX=INT((XI-XS)/DX+ONE)
     IY=INT((YI-YS)/DY+ONE)
     IZ=INT((ZI-ZS)/DZ+ONE)
     IX1=IX+1
     IY1=IY+1
     IZ1=IZ+1
     IF(IX<1.OR.IY<1.OR.IZ<1)cycle loop200
     IF(IX1>NX.OR.IY1>NY.OR.IZ1>NZ)cycle loop200
     DTX1=IX-(XI-XS)/DX
     DTY1=IY-(YI-YS)/DY
     DTZ1=IZ-(ZI-ZS)/DZ
     DTX=ONE-DTX1
     DTY=ONE-DTY1
     DTZ=ONE-DTZ1
     IXYZ000=NXYZMAP(IX,IY,IZ,NX,NY,NZ)
     CORE(IXYZ000)=1
     RHO(IXYZ000)=RHO(IXYZ000)+CGI*DTX1*DTY1*DTZ1
     IXYZ100=NXYZMAP(IX1,IY,IZ,NX,NY,NZ)
     CORE(IXYZ100)=1
     RHO(IXYZ100)=RHO(IXYZ100)+CGI*DTX*DTY1*DTZ1
     IXYZ010=NXYZMAP(IX,IY1,IZ,NX,NY,NZ)
     CORE(IXYZ010)=1
     RHO(IXYZ010)=RHO(IXYZ010)+CGI*DTX1*DTY*DTZ1
     IXYZ110=NXYZMAP(IX1,IY1,IZ,NX,NY,NZ)
     CORE(IXYZ110)=1
     RHO(IXYZ110)=RHO(IXYZ110)+CGI*DTX*DTY*DTZ1
     IXYZ001=NXYZMAP(IX,IY,IZ1,NX,NY,NZ)
     CORE(IXYZ001)=1
     RHO(IXYZ001)=RHO(IXYZ001)+CGI*DTX1*DTY1*DTZ
     IXYZ101=NXYZMAP(IX1,IY,IZ1,NX,NY,NZ)
     CORE(IXYZ101)=1
     RHO(IXYZ101)=RHO(IXYZ101)+CGI*DTX*DTY1*DTZ
     IXYZ011=NXYZMAP(IX,IY1,IZ1,NX,NY,NZ)
     CORE(IXYZ011)=1
     RHO(IXYZ011)=RHO(IXYZ011)+CGI*DTX1*DTY*DTZ
     IXYZ111=NXYZMAP(IX1,IY1,IZ1,NX,NY,NZ)
     CORE(IXYZ111)=1
     RHO(IXYZ111)=RHO(IXYZ111)+CGI*DTX*DTY*DTZ
  enddo loop200
  IF(NATI==0)RETURN
  !  Update core indexes
  CALL BUILDCORE(NX,NY,NZ,CORE)
  NCORE=0
  DO IX=1,NX
     DO IY=1,NY
        DO IZ=1,NZ
           IXYZ=NXYZMAP(IX,IY,IZ,NX,NY,NZ)
           IF(CORE(IXYZ) > 0)NCORE=NCORE+1
        ENDDO
     ENDDO
  ENDDO
  !    Calculate electrostatic potentials of non-core grids 
  CGF=CCELEC/EPS
  DX1=TWO/(NX-1)
  DY1=TWO/(NY-1)
  DZ1=TWO/(NZ-1)
  DO IX=2,NX-1
     XI=(IX-1)*DX1-ONE 
     XJ=AX*XI/(ONE-ABS(XI))
     DO IY=2,NY-1
        YI=(IY-1)*DY1-ONE
        YJ=AY*YI/(ONE-ABS(YI))
        loop500: DO IZ=2,NZ-1
           ZI=(IZ-1)*DZ1-ONE
           ZJ=AZ*ZI/(ONE-ABS(ZI))
           IX1=NINT((XJ-XS)/DX)+1
           IY1=NINT((YJ-YS)/DY)+1
           IZ1=NINT((ZJ-ZS)/DZ)+1
           IF(IX1*(NX-IX1) > 0 .AND.IY1*(NY-IY1).GT.0 &
                .AND.IZ1*(NZ-IZ1) > 0)THEN
              IXYZ1=NXYZMAP(IX1,IY1,IZ1,NX,NY,NZ)
              IF(CORE(IXYZ1) > 0)THEN
                 IX1=IX1
                 cycle loop500
              ENDIF
           ENDIF
           IXYZ=NXYZMAP(IX,IY,IZ,NX,NY,NZ)
           EELE=ZERO 
           IF(LGRIDCG)THEN
              !    Calculate electrostatic potentials  from grid charges
              DO IX1=1,NX
                 DO IY1=1,NY
                    DO IZ1=1,NZ
                       IXYZ1=NXYZMAP(IX1,IY1,IZ1,NX,NY,NZ)
                       IF(CORE(IXYZ1) > 0)THEN
                          CGI=RHO(IXYZ1)*CGF
                          XI=(IX1-1)*DX+XS 
                          YI=(IY1-1)*DY+YS
                          ZI=(IZ1-1)*DZ+ZS
                          XIJ=XI-XJ
                          YIJ=YI-YJ
                          ZIJ=ZI-ZJ
                          RIJ=SQRT(XIJ*XIJ+YIJ*YIJ+ZIJ*ZIJ)
                          EELE=EELE+CGI/RIJ
                       ENDIF
                    ENDDO
                 ENDDO
              ENDDO
           ELSE
              !    Calculate electrostatic potentials  from atom charges
              DO I=1,NATI
                 CGI=CGR(I)*CGF
                 XI=XR(I)
                 YI=YR(I)
                 ZI=ZR(I)
                 XIJ=XI-XJ
                 YIJ=YI-YJ
                 ZIJ=ZI-ZJ
                 RIJ=SQRT(XIJ*XIJ+YIJ*YIJ+ZIJ*ZIJ)
                 EELE=EELE+CGI/RIJ
              ENDDO
           ENDIF
           DDR(IXYZ)=EELE
        ENDDO loop500
     ENDDO
  ENDDO
  RETURN 
END SUBROUTINE FIELD2EMP


SUBROUTINE EMAPERR(IDRIG1,IDRIG2,ECORE,EELE,ESOLV,ECONS)
  !-----------------------------------------------------------------------
  !   This routeine calculate the interaction between two rigid fragments 
  !
  use chm_kinds
  use dimens_fcm
  use number
  use emapmod
  implicit none
  INTEGER IDEMP1,IDEMP2,IDRIG1,IDRIG2,IDRIG
  real(chm_real) ECORE,EELE,ESOLV,ECONS
  LOGICAL LDDR,LCORE,NEWRIG
  IDEMP1=IDEMRIG(IDRIG1)
  IDEMP2=IDEMRIG(IDRIG2)
  !  Create scratch rigid fragments for the rigid fragment
  NEWRIG=CHKRIGNM('FMAPRIG',8,IDRIG)
  IF(.NOT.NEWRIG)CALL WRNDIE(0,'<MC>', &
       'CANNOT CREAT SCRATCH rigid ID:'//EMRIGID(IDRIG))
  CALL EMAPRIGDUP(IDRIG1,IDRIG)
  IF(LPBC)CALL EMAPRPBC(IDRIG,IDRIG2)
  CALL EMAPRTRN(IDRIG,IDRIG2)
  ECORE=ZERO
  EELE=ZERO
  ESOLV=ZERO
  ECONS=ZERO
  CALL EMAPERM1(IDEMP2,IDRIG, empgrd(IDEMP2)%rho, &
       empgrd(IDEMP2)%ddrho,empgrd(IDEMP2)%core, &
       empgrd(IDEMP1)%rho,empgrd(IDEMP1)%ddrho, &
       empgrd(IDEMP1)%core,TEMRIG(1,IDRIG),REMRIG(1,IDRIG), &
       ECORE,EELE,ESOLV,ECONS)
  CALL RMEMRIG(IDRIG)
  RETURN
END SUBROUTINE EMAPERR


SUBROUTINE EMAPERM(IDEMP,IDRIG,ECORE,EELE,ESOLV,ECONS)
  !-----------------------------------------------------------------------
  !   This routeine calculate the interaction between molecular map IDEMPM and 
  !   a field map IDEMPF
  !
  use chm_kinds
  use dimens_fcm
  use number
  use emapmod
  implicit none
  INTEGER IDEMP,IDEMP1,IDRIG
  real(chm_real) ECORE,EELE,ESOLV,ECONS
  LOGICAL LDDR,LCORE
  IDEMP1=IDEMRIG(IDRIG)
  ECORE=ZERO
  EELE=ZERO
  ESOLV=ZERO
  ECONS=ZERO
  CALL EMAPERM1(IDEMP,IDRIG, &
       empgrd(IDEMP)%rho, &
       empgrd(IDEMP)%ddrho, &
       empgrd(IDEMP)%core, &
       empgrd(IDEMP1)%rho, &
       empgrd(IDEMP1)%ddrho, &
       empgrd(IDEMP1)%core, &
       TEMRIG(1,IDRIG),REMRIG(1,IDRIG), &
       ECORE,EELE,ESOLV,ECONS)
  RETURN
END SUBROUTINE EMAPERM


SUBROUTINE EMAPERM1(IDEMP1,IDRIG,RHO1,DDR1,CORE1, &
     RHO2,DDR2,CORE2,TR,U,ECORE,EELE,ESOLV,ECONS)
  !-----------------------------------------------------------------------
  !  This routine calculate the Molecular Map-Field Map interaction
  !
  use chm_kinds
  use dimens_fcm
  use number
  use emapmod
  implicit none
  INTEGER IDEMP1,IDEMP2,IDRIG,CORE1(*),CORE2(*)
  INTEGER IXYZ000,IXYZ100,IXYZ010,IXYZ110,IXYZ001,IXYZ101
  INTEGER IXYZ011,IXYZ111
  real(chm_real) CI000,CI100,CI010,CI110,CI001,CI101,CI011,CI111
  real(chm_real) CORE1I,CORE2I,CISUM,RC1,RC2
  REAL*4 RHO1(*),DDR1(*),RHO2(*),DDR2(*)
  real(chm_real) PHI000,PHI100,PHI010,PHI110,PHI001, &
       PHI101,PHI011,PHI111
  real(chm_real) PHI,DPHIX,DPHIY,DPHIZ,DPHIXY,DPHIZX,DPHIYZ,DPHIXYZ
  real(chm_real) TR(3),U(3,3),ECORE,EELE,ESOLV,ECONS,ESOLV1,ECORE1
  real(chm_real) CGI,CGJ,PSOLV1,PSOLV2,PSELE,PCORE
  real(chm_real) ENGCORE,ENGSOLV,ENGBIND
  INTEGER NXYZMAP
  INTEGER I,J,IXYZ1,IXYZ2
  INTEGER NX1,NY1,NZ1,NX2,NY2,NZ2
  INTEGER IX,IY,IZ,IX1,IY1,IZ1,IX2,IY2,IZ2
  real(chm_real) XS1,YS1,ZS1,X1,Y1,Z1,XS2,YS2,ZS2,X2,Y2,Z2
  real(chm_real) DTX,DTY,DTZ,XI,YI,ZI
  real(chm_real) DX1,DY1,DZ1,DX2,DY2,DZ2,DFX,DFY,DFZ
  real(chm_real) AX1,AY1,AZ1,AX2,AY2,AZ2
  real(chm_real) CX1,CY1,CZ1,CX2,CY2,CZ2
  IDEMP2=IDEMRIG(IDRIG)
  NX1=LEMAPX(IDEMP1)
  NY1=LEMAPY(IDEMP1)
  NZ1=LEMAPZ(IDEMP1)
  NX2=LEMAPX(IDEMP2)
  NY2=LEMAPY(IDEMP2)
  NZ2=LEMAPZ(IDEMP2)
  DX1=DEMAPX(IDEMP1)
  DY1=DEMAPY(IDEMP1)
  DZ1=DEMAPZ(IDEMP1)
  DX2=DEMAPX(IDEMP2)
  DY2=DEMAPY(IDEMP2)
  DZ2=DEMAPZ(IDEMP2)
  CX2=CEMAPX(IDEMP2)
  CY2=CEMAPY(IDEMP2)
  CZ2=CEMAPZ(IDEMP2)
  AX1=AEMAPX(IDEMP1)
  AY1=AEMAPY(IDEMP1)
  AZ1=AEMAPZ(IDEMP1)
  DFX=TWO/(NX1-1)
  DFY=TWO/(NY1-1)
  DFZ=TWO/(NZ1-1)
  XS1=MEMAPX(IDEMP1)*DX1
  YS1=MEMAPY(IDEMP1)*DY1
  ZS1=MEMAPZ(IDEMP1)*DZ1
  XS2=MEMAPX(IDEMP2)*DX2
  YS2=MEMAPY(IDEMP2)*DY2
  ZS2=MEMAPZ(IDEMP2)*DZ2
  !  energy parameters
  RC1=(DX1*DY1*DZ1)**(ONE/THREE)
  RC2=(DX2*DY2*DZ2)**(ONE/THREE)
  !  Solvation proportional to surface  of molecule 2
  !   and proportional to the charge density on the surface
  !      PSOLV1=EMAPEPS*RC2*RC2/(RC1)**6
  !      PSOLV2=EMAPEPS*RC2*RC2/(RC2)**6
  PSOLV1=EMAPEPS*RC2*RC2/(RC1)**3
  PSOLV2=EMAPEPS*RC2*RC2/(RC2)**3
  !  Electric binding propertional to charge density of map1
  !      PSELE=EMAPEPE/RC1**3
  PSELE=EMAPEPE*RC2*RC2/SQRT(RC1**3*RC2**3)
  !  vdw binding proportional to surface of molecule 2
  !      PCORE=EMAPEPC*RC2*RC2
  PCORE=EMAPEPC*RC2*RC2/RC1
  !  Loop over all core area of molecule 2
  DO IZ2=1,NZ2
     Z2=(IZ2-1)*DZ2+ZS2-CZ2
     DO IY2=1,NY2
        Y2=(IY2-1)*DY2+YS2-CY2
        loop100: DO IX2=1,NX2
           IXYZ2=NXYZMAP(IX2,IY2,IZ2,NX2,NY2,NZ2)
           CORE2I=CORE2(IXYZ2)
           IF(CORE2I <= HALF)cycle loop100
           CGJ=RHO2(IXYZ2)
           X2=(IX2-1)*DX2+XS2-CX2
           X1=U(1,1)*X2+U(1,2)*Y2+U(1,3)*Z2+TR(1)+CX2
           Y1=U(2,1)*X2+U(2,2)*Y2+U(2,3)*Z2+TR(2)+CY2
           Z1=U(3,1)*X2+U(3,2)*Y2+U(3,3)*Z2+TR(3)+CZ2
           ! Molecular map interaction
           IX=INT((X1-XS1)/DX1+ONE)
           IY=INT((Y1-YS1)/DY1+ONE)
           IZ=INT((Z1-ZS1)/DZ1+ONE)
           IF(  &
                IX*(NX1-IX) > 0 .and. &
                IY*(NY1-IY) > 0 .and. &
                IZ*(NZ1-IZ) > 0        )    then    !GOTO 500
              IX1=IX+1
              IY1=IY+1
              IZ1=IZ+1
              DTX=(X1-XS1)/DX1+ONE-IX
              DTY=(Y1-YS1)/DY1+ONE-IY
              DTZ=(Z1-ZS1)/DZ1+ONE-IZ
              IXYZ000=NXYZMAP(IX,IY,IZ,NX1,NY1,NZ1)
              CI000=CORE1(IXYZ000)
              PHI000=RHO1(IXYZ000)
              IXYZ100=NXYZMAP(IX1,IY,IZ,NX1,NY1,NZ1)
              CI100=CORE1(IXYZ100)
              PHI100=RHO1(IXYZ100)
              IXYZ010=NXYZMAP(IX,IY1,IZ,NX1,NY1,NZ1)
              CI010=CORE1(IXYZ010)
              PHI010=RHO1(IXYZ010)
              IXYZ110=NXYZMAP(IX1,IY1,IZ,NX1,NY1,NZ1)
              CI110=CORE1(IXYZ110)
              PHI110=RHO1(IXYZ110)
              IXYZ001=NXYZMAP(IX,IY,IZ1,NX1,NY1,NZ1)
              CI001=CORE1(IXYZ001)
              PHI001=RHO1(IXYZ001)
              IXYZ101=NXYZMAP(IX1,IY,IZ1,NX1,NY1,NZ1)
              CI101=CORE1(IXYZ101)
              PHI101=RHO1(IXYZ101)
              IXYZ011=NXYZMAP(IX,IY1,IZ1,NX1,NY1,NZ1)
              CI011=CORE1(IXYZ011)
              PHI011=RHO1(IXYZ011)
              IXYZ111=NXYZMAP(IX1,IY1,IZ1,NX1,NY1,NZ1)
              CI111=CORE1(IXYZ111)
              PHI111=RHO1(IXYZ111)
              CISUM=CI000+CI100+CI010+CI110+CI001+CI101+CI011+CI111
              IF(CISUM > HALF)THEN
                 !  This grid contact or overlap with the core of molecule 1
                 !  Caculate core vdw interaction
                 DPHIX=CI100-CI000 
                 DPHIY=CI010-CI000
                 DPHIZ=CI001-CI000
                 DPHIXY=CI110-CI000-DPHIX-DPHIY
                 DPHIYZ=CI011-CI000-DPHIZ-DPHIY
                 DPHIZX=CI101-CI000-DPHIX-DPHIZ
                 DPHIXYZ=CI111-CI000-DPHIX-DPHIY-DPHIZ &
                      -DPHIXY-DPHIYZ-DPHIZX
                 CORE1I=CI000+DTX*DPHIX+DTY*DPHIY+DTZ*DPHIZ &
                      +DTX*DTY*DPHIXY+DTZ*DTY*DPHIYZ+DTX*DTZ*DPHIZX &
                      +DTX*DTY*DTZ*DPHIXYZ
                 ECORE=ECORE+ &
                      ENGCORE(CORE2I,CORE1I,PCORE)
                 !  Calculate solvation energy
                 DPHIX=PHI100-PHI000
                 DPHIY=PHI010-PHI000
                 DPHIZ=PHI001-PHI000
                 DPHIXY=PHI110-PHI000-DPHIX-DPHIY
                 DPHIYZ=PHI011-PHI000-DPHIZ-DPHIY
                 DPHIZX=PHI101-PHI000-DPHIX-DPHIZ
                 DPHIXYZ=PHI111-PHI000-DPHIX-DPHIY-DPHIZ &
                      -DPHIXY-DPHIYZ-DPHIZX
                 CGI=PHI000+DTX*DPHIX+DTY*DPHIY+DTZ*DPHIZ &
                      +DTX*DTY*DPHIXY+DTZ*DTY*DPHIYZ+DTX*DTZ*DPHIZX &
                      +DTX*DTY*DTZ*DPHIXYZ
                 ESOLV=ESOLV+ &
                      ENGSOLV(CGI,CGJ,CORE1I,CORE2I,RC1,RC2,PSOLV1,PSOLV2)
                 ECONS=ECONS+ENGBIND(CGI,CGJ,PSELE)
              ENDIF
              !500        CONTINUE
           endif
           !  Calculate electric field energy
           XI=X1/(ABS(X1)+AX1)
           YI=Y1/(ABS(Y1)+AY1)
           ZI=Z1/(ABS(Z1)+AZ1)
           IX=INT((ONE+XI)/DFX)+1
           IY=INT((ONE+YI)/DFY)+1
           IZ=INT((ONE+ZI)/DFZ)+1
           IX1=IX+1
           IY1=IY+1
           IZ1=IZ+1
           DTX=(XI+ONE)/DFX+ONE-IX
           DTY=(YI+ONE)/DFY+ONE-IY
           DTZ=(ZI+ONE)/DFZ+ONE-IZ
           IXYZ000=NXYZMAP(IX,IY,IZ,NX1,NY1,NZ1)
           PHI000=DDR1(IXYZ000)
           IXYZ100=NXYZMAP(IX1,IY,IZ,NX1,NY1,NZ1)
           PHI100=DDR1(IXYZ100)
           IXYZ010=NXYZMAP(IX,IY1,IZ,NX1,NY1,NZ1)
           PHI010=DDR1(IXYZ010)
           IXYZ110=NXYZMAP(IX1,IY1,IZ,NX1,NY1,NZ1)
           PHI110=DDR1(IXYZ110)
           IXYZ001=NXYZMAP(IX,IY,IZ1,NX1,NY1,NZ1)
           PHI001=DDR1(IXYZ001)
           IXYZ101=NXYZMAP(IX1,IY,IZ1,NX1,NY1,NZ1)
           PHI101=DDR1(IXYZ101)
           IXYZ011=NXYZMAP(IX,IY1,IZ1,NX1,NY1,NZ1)
           PHI011=DDR1(IXYZ011)
           IXYZ111=NXYZMAP(IX1,IY1,IZ1,NX1,NY1,NZ1)
           PHI111=DDR1(IXYZ111)
           !  This grid does not contact or overlap with the core of molecule 1
           DPHIX=PHI100-PHI000
           DPHIY=PHI010-PHI000
           DPHIZ=PHI001-PHI000
           DPHIXY=PHI110-PHI000-DPHIX-DPHIY
           DPHIYZ=PHI011-PHI000-DPHIZ-DPHIY
           DPHIZX=PHI101-PHI000-DPHIX-DPHIZ
           DPHIXYZ=PHI111-PHI000-DPHIX-DPHIY-DPHIZ &
                -DPHIXY-DPHIYZ-DPHIZX
           PHI=PHI000+DTX*DPHIX+DTY*DPHIY+DTZ*DPHIZ &
                +DTX*DTY*DPHIXY+DTZ*DTY*DPHIYZ+DTX*DTZ*DPHIZX &
                +DTX*DTY*DTZ*DPHIXYZ
           EELE=EELE+CGJ*PHI
           !            ECONS=ECONS-PHI*PHI
        enddo loop100
     ENDDO
  ENDDO
  RETURN
END SUBROUTINE EMAPERM1


function ENGSOLV(CGI,CGJ,COREI,COREJ, &
     RCI,RCJ,FSOLV1,FSOLV2)  &
     result(engsolv_1)
  !-----------------------------------------------------------------------
  !  Calculate solvation energy
  !     Solvation energy only occure to the surface grids
  !     Solvation energy is: Es=fs*ei*ei*s
  !
  use chm_kinds
  use dimens_fcm
  use number
  implicit none
  INTEGER I
  real(chm_real)  CGI,CGJ,COREI,COREJ,RCW6,engsolv_1
  real(chm_real)  RCI,RCJ,FSOLV1,FSOLV2,SI,SJ,CIN,CJN
  !  Assuming surface cut at core=2,  1/(1+(core/2)^6)
  RCW6=64.00D0
  !   overlap from core=1 Angstroms, core^6/(1+core^6)
  CIN=COREI**6
  SI=RCW6/(RCW6+CIN)
  CJN=COREJ**6
  SJ=RCW6/(RCW6+CJN)
  ENGSOLV_1=FSOLV1*CGI*CGI*SI*CJN/(ONE+CJN) &
       +FSOLV2*CGJ*CGJ*SJ*CIN/(ONE+CIN)
  RETURN
END function ENGSOLV


function ENGBIND(CGI,CGJ,FBIND) result(engbind_1)
  !-----------------------------------------------------------------------
  !  Calculate contact binding energy
  !     Binding energy  occures when two surface contact
  !     Binding energy is: Ebind=fbind*ei*ej*s
  !
  use chm_kinds
  use dimens_fcm
  use number
  implicit none
  real(chm_real) ENGBIND_1
  real(chm_real)  CGI,CGJ
  real(chm_real)  FBIND
  ENGBIND_1=FBIND*CGI*CGJ
  RETURN
END function ENGBIND


function ENGCORE(COREI,COREJ,FVDW) result(engcore_1)
  !-----------------------------------------------------------------------
  !  Calculate overlapping energy
  !     Ecore=
  !
  use chm_kinds
  use dimens_fcm
  use number
  implicit none
  real(chm_real)  COREI,COREJ,ENGCORE_1
  real(chm_real)  FVDW,CIJ,CIN,CJN
  !  vdw energy is based on the product to allow insertion: cij=corei*corej
  !      defined by 4((cij/3)^2-(cij/3))
  CIJ=(COREI*COREJ)/THREE
  ENGCORE_1=FOUR*FVDW*(CIJ*CIJ-CIJ)
  RETURN
END function ENGCORE


function VFACTOR(XI,YI,ZI,AX,AY,AZ) result(vfactor_1)
  !-----------------------------------------------------------------------
  !  Calculate surface factor  in field map
  !     grids are from (-NX,-NY,-NZ) to (NX,NY,NZ)
  !
  use chm_kinds
  use dimens_fcm
  use number
  implicit none
  real(chm_real) XI,YI,ZI,AX,AY,AZ,AXYZ,DTXYZ,vfactor_1
  !
  AXYZ=AX*AY*AZ
  DTXYZ=(ONE-ABS(XI))*(ONE-ABS(YI))*(ONE-ABS(ZI))
  VFACTOR_1=AXYZ/DTXYZ/DTXYZ
  RETURN
END function VFACTOR


INTEGER function NXYZMAP(IX,IY,IZ,NX,NY,NZ)
  !-----------------------------------------------------------------------
  !  Calculate index number in a field map
  !     grids are from (1,1,1) to (NX,NY,NZ)
  !
  use chm_kinds
  use dimens_fcm
  use number
  implicit none
  INTEGER IX,IY,IZ,NX,NY,NZ
  !
  NXYZMAP=IX+NX*(IY-1+NY*(IZ-1))
  RETURN
END function NXYZMAP

INTEGER function NXYZCALC(IX,IY,IZ,NX,NY,NZ)
  !-----------------------------------------------------------------------
  !  Calculate index number in a field map
  !     grids are from (-NX,-NY,-NZ) to (NX,NY,NZ)
  !
  use chm_kinds
  use dimens_fcm
  use number
  implicit none
  INTEGER IX,IY,IZ,NX,NY,NZ
  !
  NXYZCALC=(NX*NY*NZ+1)/2+IX+NX*(IY+NY*IZ)
  return
end function NXYZCALC

#endif /* (emap)*/

SUBROUTINE NULL_EMFIELD
  RETURN
END SUBROUTINE NULL_EMFIELD

