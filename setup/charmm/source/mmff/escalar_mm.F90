module escalar_mm
  implicit none

#if KEY_MMFF==1
contains

!=======================================================================
! real(chm_real) FUNCTION CANGLE : COMPUTES THE COSINE OF THE VALENCE ANGLE
! BETWEEN ATOMS I, J, AND K
!=======================================================================
FUNCTION CANGLE(DERIVS,I,J,K,NDIRC,X,Y,Z) result(cangle_rtn)
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !  Written by T. Halgren but based on the adaptation by R. Czerminski
  !  of Molecular Simulations, Inc. of the similar subroutine vangle,
  !  itself based on code developed at Merck and Co., Inc. by T. Halgren
  !  and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! Jay Banks 19 Oct 95: removed MAXCRD as argument, replaced it with
  ! MAXAIM as dimension
  ! Jay Banks 27 Nov 95: changed XYZ array to X,Y,Z
  !
  !  28 Nov 95 Jay Banks: removed space from all "GO TO"s, as suggested in
  !  developer.doc.
  !
  !  29 Nov 95 Jay Banks: changed arithmetic IFs to IF ... THEN ... ELSE.
  !
  use chm_kinds
  use dimens_fcm
  use mmffm
  implicit none
  !
  integer DERIVS, i, j, k, ndirc
  real(chm_real)  x(*),y(*),z(*),cangle_rtn
  !
  integer n, n1, n2
  real(chm_real)  costh, di, dii, dij, dik, dil
  real(chm_real)  dj, djj, djk, djl, dk, dkk, dkl, dl, dll
  real(chm_real)  rij2, rijrjk, rjk2
  !
  COMMON/DTDQ/DI(3),DJ(3),DK(3),DL(3)
  COMMON/D2TDQ2/DII(3,3),DIJ(3,3),DIK(3,3),DIL(3,3),DJJ(3,3), &
       DJK(3,3),DJL(3,3),DKK(3,3),DKL(3,3),DLL(3,3)
  !
  !   GET NEEDED DIRECTION COSINES IF NOT ALREADY IN PLACE
  !
  IF(NDIRC.NE.0) CALL DIRCOS(I,J,K,0,X,Y,Z)
  !     CALCULATE COSINE OF ANGLE
  COSTH=0.0D0
  DO N=1,3
     COSTH=COSTH+EJI(N)*EJK(N)
  ENDDO
  !   MAKE SURE VALUE IS IN RANGE
  IF(COSTH.LT.-1.D0) COSTH=-1.D0
  IF(COSTH.GT.1.D0) COSTH=1.D0
  CANGLE_rtn=COSTH
  IF(derivs.EQ.0) RETURN
  !     CALCULATE FIRST DERIVATIVES
  DO N=1,3
     DI(N)=(EJK(N)+CANGLE_RTN*EIJ(N))/EIJ(4)
     DK(N)=(-EIJ(N)-CANGLE_RTN*EJK(N))/EJK(4)
     DJ(N)=-DI(N)-DK(N)
  ENDDO
  !
  IF(derivs.LE.1) RETURN
  !     WRITE(6,*) ' CANGLE_RTN = ',CANGLE_RTN
  !     CALCULATE SECOND DERIVATIVES
  RIJ2=1.0D0/(EIJ(4)*EIJ(4))
  RJK2=1.0D0/(EJK(4)*EJK(4))
  RIJRJK=1.D0/(EIJ(4)*EJK(4))
  DO N1=1,3
     DO N2=1,N1
        IF(N1.NE.N2) THEN
           !
           !         CASE 2: D2E/DXIDYI, ETC (N1 .NE. N2)
           !
           DII(N1,N2)=(EIJ(N2)*EJK(N1) + EIJ(N1)*EJK(N2) &
                + 3.D0*CANGLE_RTN*EIJ(N1)*EIJ(N2))*RIJ2
           DII(N2,N1)=DII(N1,N2)
           DKK(N1,N2)=(EJK(N1)*EIJ(N2) + EIJ(N1)*EJK(N2) &
                + 3.D0*CANGLE_RTN*EJK(N1)*EJK(N2))*RJK2
           DKK(N2,N1)=DKK(N1,N2)
           !
           !         CASE 6: D2E/DXJDYJ, ETC (N1 .NE. N2)
           !
           DJJ(N1,N2)=DII(N1,N2)+DKK(N1,N2) &
                -RIJRJK*(2.D0*(EIJ(N1)*EIJ(N2)+EJK(N1)*EJK(N2)) &
                +CANGLE_RTN*(EIJ(N2)*EJK(N1)+EIJ(N1)*EJK(N2)))
           DJJ(N2,N1)=DJJ(N1,N2)
           !
        ELSE
           !
           !         CASE 1: D2E/DXI**2, ETC. (N1 = N2)
           !
           DII(N1,N2)=RIJ2*(2.D0*EIJ(N2)*EJK(N1) &
                +CANGLE_RTN*(3.D0*EIJ(N1)*EIJ(N2) - 1.D0))
           DKK(N1,N2)=RJK2*(2.D0*EJK(N2)*EIJ(N1) &
                +CANGLE_RTN*(3.D0*EJK(N1)*EJK(N2) - 1.D0))
           !
           !         CASE 5: D2E/DXJ**2, ETC  (N1 = N2)
           DJJ(N1,N2)=DII(N1,N2)+DKK(N1,N2) +RIJRJK* &
                (2.D0*(1.D0-EIJ(N1)*EIJ(N2)-EJK(N1)*EJK(N2)) &
                -CANGLE_RTN*(EIJ(N1)*EJK(N2)+EIJ(N2)*EJK(N1)))
           !
        ENDIF
     ENDDO
  ENDDO
  DO N1=1,3
     DO N2=1,3
        IF(N1.NE.N2) THEN
           !
           !         CASE 8: D2E/DXIDYK, ETC (N1 .NE. N2)
           !
           DIK(N1,N2)=-RIJRJK*(EIJ(N1)*EIJ(N2)+EJK(N1)*EJK(N2) &
                +CANGLE_RTN*EIJ(N1)*EJK(N2))
           !     .               -CANGLE_RTN*EIJ(N1)*EJK(N2))
           !
        ELSE
           !
           !         CASE 7: D2E/DXIDYK, ETC (N1 = N2)
           !
           DIK(N1,N2)=-RIJRJK*(EIJ(N1)*EIJ(N2)+EJK(N1)*EJK(N2)-1.D0 &
                +CANGLE_RTN*EIJ(N1)*EJK(N2))
           !     .               -CANGLE_RTN*EIJ(N1)*EJK(N2))
           !
        ENDIF
        !
        !         CASES 3 AND 4: D2E/DXIDXJ, D2E/DXIDYJ, ETC (N1 .NE. N2)
        !         NOTE: (Jay Banks, 29-NOV-95)  In previous code (with
        !         arithmetic IF), these lines executed for BOTH N1.EQ.N2 and
        !         N1.NE.N2.  So I left it that way, despite the above comment.
        !
        DIJ(N1,N2)=-(DII(N1,N2)+DIK(N1,N2))
        DJK(N1,N2)=-(DKK(N1,N2)+DIK(N1,N2))
        !
     ENDDO
  ENDDO
  !
  RETURN
END FUNCTION CANGLE

! ======================================================
! SUBROUTINE DIRCOS : CALCULATE DIRECTION COSINES
! FOR COMBINATIONS OF INPUT ATOMS I,J,K,L
! ======================================================
SUBROUTINE DIRCOS(I,J,K,L,X,Y,Z)
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   Adapted by R. Czerminski of Molecular Simulations, Inc. from
  !  code developed at Merck and Co., Inc. by T. Halgren and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !
  !  27 Nov 95 Jay Banks: changed XYZ array to X,Y,Z.
  !
  use chm_kinds
  use dimens_fcm
  use mmffm
  implicit none
  !
  integer i, j, k, l
  real(chm_real)  X(*),Y(*),Z(*)
  !
  integer n
  !
  !   SUPPRESS FLOATING-POINT UNDERFLOW MESSAGES
  !
  !      CALL LIB$FLT_UNDER(0)
  !
  EIJ(1)=X(J)-X(I)
  EIJ(2)=Y(J)-Y(I)
  EIJ(3)=Z(J)-Z(I)
  EIJ(4)=0.
  DO N=1,3
     EIJ(4)=EIJ(4)+EIJ(N)*EIJ(N)
  ENDDO
  EIJ(4)=SQRT(EIJ(4))
  DO N=1,3
     EIJ(N)=EIJ(N)/EIJ(4)
     EJI(N)=-EIJ(N)
  ENDDO
  EJI(4)=EIJ(4)
  IF(K.LE.0) RETURN
  EIK(1)=X(K)-X(I)
  EIK(2)=Y(K)-Y(I)
  EIK(3)=Z(K)-Z(I)
  EIK(4)=0.
  DO N=1,3
     EIK(4)=EIK(4)+EIK(N)*EIK(N)
  ENDDO
  EIK(4)=SQRT(EIK(4))
  DO N=1,3
     EIK(N)=EIK(N)/EIK(4)
     EKI(N)=-EIK(N)
  ENDDO
  EKI(4)=EIK(4)
  EJK(1)=X(K)-X(J)
  EJK(2)=Y(K)-Y(J)
  EJK(3)=Z(K)-Z(J)
  EJK(4)=0.
  DO N=1,3
     EJK(4)=EJK(4)+EJK(N)*EJK(N)
  ENDDO
  EJK(4)=SQRT(EJK(4))
  DO N=1,3
     EJK(N)=EJK(N)/EJK(4)
     EKJ(N)=-EJK(N)
  ENDDO
  EKJ(4)=EJK(4)
  IF(L.LE.0) RETURN
  EKL(1)=X(L)-X(K)
  EKL(2)=Y(L)-Y(K)
  EKL(3)=Z(L)-Z(K)
  EKL(4)=0.
  DO N=1,3
     EKL(4)=EKL(4)+EKL(N)*EKL(N)
  ENDDO
  EKL(4)=SQRT(EKL(4))
  DO N=1,3
     EKL(N)=EKL(N)/EKL(4)
     ELK(N)=-EKL(N)
  ENDDO
  ELK(4)=EKL(4)
  EJL(1)=X(L)-X(J)
  EJL(2)=Y(L)-Y(J)
  EJL(3)=Z(L)-Z(J)
  EJL(4)=0.
  DO N=1,3
     EJL(4)=EJL(4)+EJL(N)*EJL(N)
  ENDDO
  EJL(4)=SQRT(EJL(4))
  DO N=1,3
     EJL(N)=EJL(N)/EJL(4)
     ELJ(N)=-EJL(N)
  ENDDO
  ELJ(4)=EJL(4)
  !
  RETURN
END SUBROUTINE DIRCOS

!=======================================================================
! SUBROUTINE EANGLE_MM : Routine for angle-bending in subject molecule
!=======================================================================
!23456789 123456789 123456789 123456789 123456789 123456789 123456789 12
!
SUBROUTINE EANGLE_MM(ETH,ITHETAX,JTHETAX,KTHETAX,ICT,NTHETAX, &
     myAnglEq,myAnglFC,cbndx, &
     X,Y,Z,DX,DY,DZ,HESS,IUPT,derivs, &
     QECONT,ECONT,ICONAH,ISKT)
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   Adapted by R. Czerminski of Molecular Simulations, Inc. from
  !  code developed at Merck and Co., Inc. by T. Halgren and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !
  ! Jay Banks 25 Oct 95: changed local variable name IA to IIA, and
  ! argument cbnd to cbndx, to avoid conflict with variables in common
  ! (mmff.f90).
  !
  ! Jay Banks 27 Nov 95: changed XYZ, DXYZ arrays to X, Y, Z, DX, DY, DZ.
  !
  !  28 Nov 95 Jay Banks: removed space from all "GO TO"s, as suggested in
  !  developer.doc.
  !
  !  29 Nov 95 Jay Banks: changed arithmetic IFs to IF ... THEN ... ELSE.
  !
  !  29 Nov 95 Jay Banks: added BLOCK code.
  !
  !  19 May 2011 Antti-Pekka Hynninen: added support for domdec
  !
  use chm_kinds
  use dimens_fcm
  use consta
  use exfunc
  use number
  use mmffm
  use parallel
  use psf
  use stream
#if KEY_BLOCK==1
  use block_fcm,only: blcoep,iblckp,noforc,qblock
#endif /*  BLOCK*/
#if KEY_DOMDEC_MMFF==1 /* domdec and mmff are incompatible */
  use domdec_bonded,only:nangletbl,angletbl  
#endif
  implicit none
  !
  real(chm_real)  ETH           ! total angle energy
  integer ITHETAX(*),JTHETAX(*),KTHETAX(*) ! angle list
  integer ict(*)        ! angle parameter pointer
  integer nTHETAX        ! number of angles
  real(chm_real)  myAnglEq(*)    ! angle equilibrium value
  real(chm_real)  myAnglFC(*)    ! angle bending fc
  real(chm_real)  cbndx          ! the MMFF cubic-bend parameter
  !                             (normally -0.4 rad)
  real(chm_real)  x(*),y(*),z(*) ! coordinates
  real(chm_real)  dx(*),dy(*),dz(*)! first derivatives
  real(chm_real)  HESS(*)          ! second derivatives
  INTEGER IUPT(*)
  integer derivs          ! 0=only energy, 1=first der.
  !                           ! 2=sec. der. diag. 3=sd all
  LOGICAL QECONT
  real(chm_real)  ECONT(*)
  INTEGER ICONAH
  INTEGER ISKT(*)
  !
  integer AtomType,t_class
  external AtomType,t_class
  !
  real(chm_real) STRAIN
  COMMON/DTDQ/DI(3),DJ(3),DK(3),DL(3)
  COMMON/D2TDQ2/DII(3,3),DIJ(3,3),DIK(3,3),DIL(3,3),DJJ(3,3), &
       DJK(3,3),DJL(3,3),DKK(3,3),DKL(3,3),DLL(3,3)
  !
  character(len=10) value
  integer iia, idx, im1t3, imj, imk, iml
  integer ja, jdx, jm1t3, jmk, jml, ka, kdx, km1t3, kml
  integer la, ldx, lm1t3, mii, mij, mik, mil, mjj, mjk, mjl, mkk
  integer mj, mkl, mll, n, n1, n2, ngl
  logical constr
  real(chm_real)  c1, c2
  real(chm_real)  di, dii, dij, dik, dil, dj, djj, djk, djl
  real(chm_real)  dk, dkk, dkl, dl, dll, dth, enth
  real(chm_real)  theta, ctheta
#if KEY_BLOCK==1
  real(chm_real) COEF
  INTEGER IBL, JBL, KK
#endif /*  BLOCK*/
#if KEY_DOMDEC_MMFF==1 /* domdec and mmff are incompatible */
  integer nn  
#endif
  !
  !     CALCULATE STERIC ENERGY ASSOCIATED WITH DEFORMATION OF ANGLE
  !     IDX,JDX,KDX.
  !
  !   SUPPRESS UNDERFLOAT MESSAGES IF ON VAX
  !
  !      CALL LIB$FLT_UNDER(0)
  !
  ETH=ZERO
  if(NTHETAX.le.0) return
#if KEY_DOMDEC_MMFF==1 /* domdec and mmff are incompatible */
  if (nangletbl == 0) return  
#endif
  !
  !rcz 93/03/05 - all references to LDX should be removed
  !               this is dead code !!!
  !
  ldx=0
  !
  if(QECONT .and. prnlev.ge.2) WRITE(OUTU,7010)
7010 FORMAT(/,'      A N G L E   B E N D I N G         '// &
       ' -------ATOMS-------   -ATOM TYPES-   FF     VALENCE', &
       '      IDEAL                 STRAIN     FORCE',/, &
       '  I       J       K     I    J    K  CLASS    ANGLE ', &
       '      ANGLE      DIFF.      ENERGY   CONSTANT'/ &
       1X,97('-'))
  !
  !
  constr=ICONAH.ne.0
  !
#if KEY_DOMDEC_MMFF==1 /* domdec and mmff are incompatible */
  do nn=1,nangletbl
     ngl = angletbl(nn)
#else /* (domdec)*/
#if KEY_PARALLEL==1
  DO NGL=MYNODP,NTHETAX,NUMNOD
#else /**/
  DO NGL=1,NTHETAX
#endif 
#endif /* (domdec)*/
     if(constr) then
        if(iskt(ngl).ne.0) goto 4000
     endif
     if(ict(ngl).le.0) goto 4000
     !        write(outu,*) 'ngl, ict(ngl) =',ngl,ict(ngl)
     IDX=ITHETAX(NGL)
     JDX=JTHETAX(NGL)
     KDX=KTHETAX(NGL)
     !        write(outu,*) ' idx,jdx,kdx =',idx,jdx,kdx
#if KEY_BLOCK==1
     IF (QBLOCK) THEN
        IBL = IBLCKP(IDX)
        JBL = IBLCKP(JDX)
        KK = IBLCKP(KDX)
        IF (IBL .EQ. JBL) JBL=KK
        IF (JBL .LT. IBL) THEN
           KK=JBL
           JBL=IBL
           IBL=KK
        ENDIF
        KK=IBL+JBL*(JBL-1)/2
        COEF = BLCOEP(KK)
     ENDIF
#endif /*  BLOCK*/
     !
     !   USE SPECIAL ROUTINE IF THIS IS A LINEAR CENTER
     !
     MJ=MTYPE(JDX)
     !        write(outu,*) ' mj = ', mj, ', mlinba(mj) = ', mlinba(mj)
     IF(MLINBA(MJ).NE.0) THEN
        !           write(outu,*) 'CALLING LINEAR ANGLE ROUTINE'
        CALL eanglin_mm(NGL,IDX,JDX,KDX,ict,ctheta, &
             myAnglFC,X,Y,Z,DX,DY,DZ,HESS,derivs &
#if KEY_BLOCK==1
             ,COEF &
#endif 
             )
        THETA=ACOS(CTHETA)
        DTH=THETA-myAnglEq(ICT(NGL))
        STRAIN=myAnglFC(ICT(NGL))*(1.D0+CTHETA)
#if KEY_BLOCK==1
        IF (QBLOCK) THEN
           STRAIN=STRAIN*COEF
        ENDIF
#endif /*  BLOCK*/
        ETH=ETH+STRAIN
        GOTO 3800
     ENDIF
     THETA = VANGLE(derivs,IDX,JDX,KDX,1,X,Y,Z)
100  STRAIN = 0.0D0
     DTH=THETA-myAnglEq(ICT(NGL))
     !     QUADRATIC POTENTIAL
200  C2=myAnglFC(ICT(NGL))
     C1=C2*DTH
     STRAIN=HALF*C1*DTH
     !     ADD NTH DEGREE TERM
     ENTH=cbndx*DTH
     STRAIN=STRAIN*(ONE+ENTH)
     !     STOT=STRAIN+SN
     !D     WRITE(OUTU,33) IDX,JDX,KDX,LDX,THETA,AnglEq(ict(ngl)),DTH,
     !D    . STRAIN-ENTH,ENTH,STRAIN
33   FORMAT(4I5,6F10.5)
#if KEY_BLOCK==1
     IF (QBLOCK) THEN
        STRAIN=STRAIN*COEF
        C1=C1*COEF
        C2=C2*COEF
     ENDIF
#endif /*  BLOCK*/
600  ETH=ETH+STRAIN
#if KEY_BLOCK==1
     IF (.NOT. NOFORC) THEN
#endif /*  BLOCK*/
        !
        IF(derivs.EQ.0) GOTO 3800
        !     CALCULATE FIRST DERIVATIVES
        C1=C1*(ONE+ONEPT5*ENTH)
        IF (LDX.GT.0) THEN
           DX(LDX) = DX(LDX) + C1*DL(1)
           DY(LDX) = DY(LDX) + C1*DL(2)
           DZ(LDX) = DZ(LDX) + C1*DL(3)
        ENDIF
        DX(IDX) = DX(IDX) + C1*DI(1)
        DY(IDX) = DY(IDX) + C1*DI(2)
        DZ(IDX) = DZ(IDX) + C1*DI(3)
        DX(JDX) = DX(JDX) + C1*DJ(1)
        DY(JDX) = DY(JDX) + C1*DJ(2)
        DZ(JDX) = DZ(JDX) + C1*DJ(3)
        DX(KDX) = DX(KDX) + C1*DK(1)
        DY(KDX) = DY(KDX) + C1*DK(2)
        DZ(KDX) = DZ(KDX) + C1*DK(3)
        IF(derivs.EQ.1) GOTO 3800
        !     CALCULATE SECOND DERIVATIVES
        C2=C2*(ONE+THREE*ENTH)
1100    IM1T3=(IDX-1)*3
        JM1T3=(JDX-1)*3
        KM1T3=(KDX-1)*3
        IMJ=IDX-JDX
        IMK=IDX-KDX
        JMK=JDX-KDX
        IF(LDX.GT.0) THEN
           LM1T3=(LDX-1)*3
           IML=IDX-LDX
           JML=JDX-LDX
           KML=KDX-LDX
        ENDIF
        DO N1=1,3
           IIA=(IM1T3+N1)*(IM1T3+N1-1)/2
           JA=(JM1T3+N1)*(JM1T3+N1-1)/2
           KA=(KM1T3+N1)*(KM1T3+N1-1)/2
           IF(LDX.GT.0) THEN
              LA=(LM1T3+N1)*(LM1T3+N1-1)/2
           ENDIF
           DO N2=1,N1
              MII=IIA+IM1T3+N2
              MJJ=JA+JM1T3+N2
              MKK=KA+KM1T3+N2
              HESS(MII)=HESS(MII)+C2*DI(N1)*DI(N2)+C1*DII(N1,N2)
              HESS(MJJ)=HESS(MJJ)+C2*DJ(N1)*DJ(N2)+C1*DJJ(N1,N2)
              HESS(MKK)=HESS(MKK)+C2*DK(N1)*DK(N2)+C1*DKK(N1,N2)
              IF(LDX.GT.0) THEN
                 MLL=LA+LM1T3+N2
                 HESS(MLL)=HESS(MLL)+C2*DL(N1)*DL(N2)+C1*DLL(N1,N2)
              ENDIF
           ENDDO
           IF(derivs.EQ.2) GOTO 3700
           DO N2=1,3
              IF(IMJ.GE.0) THEN
                 MIJ=IIA+JM1T3+N2
                 HESS(MIJ)=HESS(MIJ)+C2*DI(N1)*DJ(N2)+C1*DIJ(N1,N2)
              ELSE
                 MIJ=JA+IM1T3+N2
                 HESS(MIJ)=HESS(MIJ)+C2*DI(N2)*DJ(N1)+C1*DIJ(N2,N1)
              ENDIF
              IF(IMK.GE.0) THEN
                 MIK=IIA+KM1T3+N2
                 HESS(MIK)=HESS(MIK)+C2*DI(N1)*DK(N2)+C1*DIK(N1,N2)
              ELSE
                 MIK=KA+IM1T3+N2
                 HESS(MIK)=HESS(MIK)+C2*DI(N2)*DK(N1)+C1*DIK(N2,N1)
              ENDIF
              IF(JMK.GE.0) THEN
                 MJK=JA+KM1T3+N2
                 HESS(MJK)=HESS(MJK)+C2*DJ(N1)*DK(N2)+C1*DJK(N1,N2)
              ELSE
                 MJK=KA+JM1T3+N2
                 HESS(MJK)=HESS(MJK)+C2*DJ(N2)*DK(N1)+C1*DJK(N2,N1)
              ENDIF
              IF(LDX.GT.0) THEN
                 IF(IML.GE.0) THEN
                    MIL=IIA+LM1T3+N2
                    HESS(MIL)=HESS(MIL)+C2*DI(N1)*DL(N2)+C1*DIL(N1,N2)
                 ELSE
                    MIL=LA+IM1T3+N2
                    HESS(MIL)=HESS(MIL)+C2*DI(N2)*DL(N1)+C1*DIL(N2,N1)
                 ENDIF
                 IF(JML.GE.0) THEN
                    MJL=JA+LM1T3+N2
                    HESS(MJL)=HESS(MJL)+C2*DJ(N1)*DL(N2)+C1*DJL(N1,N2)
                 ELSE
                    MJL=LA+JM1T3+N2
                    HESS(MJL)=HESS(MJL)+C2*DJ(N2)*DL(N1)+C1*DJL(N2,N1)
                 ENDIF
                 IF(KML.GE.0) THEN
                    MKL=KA+LM1T3+N2
                    HESS(MKL)=HESS(MKL)+C2*DK(N1)*DL(N2)+C1*DKL(N1,N2)
                 ELSE
                    MKL=LA+KM1T3+N2
                    HESS(MKL)=HESS(MKL)+C2*DK(N2)*DL(N1)+C1*DKL(N2,N1)
                 ENDIF
              ENDIF
           ENDDO
3700       CONTINUE
        ENDDO
#if KEY_BLOCK==1
     ENDIF  ! (.NOT.NOFORC)
#endif /*  BLOCK*/
3800 CONTINUE
     if(QECONT .and. prnlev.ge.2) WRITE(outu,7100) &
          AtName(IDX), QNAME(JDX), AtName(KDX), &
          AtomType(IDX),AtomType(JDX),AtomType(KDX), t_class(NGL), &
          THETA*RADDEG, myAnglEq(ICT(NGL))*RADDEG, DTH*RADDEG, &
          STRAIN, myAnglFC(ICT(NGL))/MDAKCAL
7100 FORMAT(1X,3(A,1X),I3,I5,I5,I5,'    ', &
          F8.3,4F11.3)
4000 CONTINUE
  ENDDO
  if(QECONT .and. prnlev.ge.2) then
     CALL WFLOAT(4,10,VALUE,ETH)
     WRITE(OUTU,8200) VALUE
8200 FORMAT(/'     TOTAL ANGLE STRAIN ENERGY = ',A//)
  endif
  !
  RETURN
END SUBROUTINE EANGLE_MM

!=======================================================================
! SUBROUTINE EANGLIN_MM : Routine for linear angle-bending in 
!                         subject molecule
!=======================================================================
!23456789 123456789 123456789 123456789 123456789 123456789 123456789 12
SUBROUTINE EANGLIN_MM(NGL,IDX,JDX,KDX,ICT,ctheta, &
     myAnglFC,X,Y,Z,DX,DY,DZ,HESS,derivs &
#if KEY_BLOCK==1
     ,COEF &
#endif 
     )
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !  Written by T. Halgren but based on the adaptation by R. Czerminski 
  !  of Molecular Simulations, Inc. of the similar subroutine eangles_mm,
  !  itself based on code developed at Merck and Co., Inc. by T. Halgren 
  !  and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! Jay Banks 19 October 1995: removed MAXCRD and MAXDER as arguments,
  ! replaced them with MAXAIM in dimensions (and added ##INCLUDE
  ! dimens.f90)
  !
  ! Jay Banks 25 Oct 95: changed local variable name IA to IIA, to avoid 
  ! conflict with variable in common (mmff.f90).
  !
  ! Jay Banks 27 Nov 95: changed XYZ, DXYZ to X, Y, Z, DX, DY, DZ.
  !
  !  28 Nov 95 Jay Banks: removed space from all "GO TO"s, as suggested in
  !  developer.doc.
  !
  !  29 Nov 95 Jay Banks: changed arithmetic IFs to IF ... THEN ... ELSE.
  !
  !  29 Nov 95 Jay Banks: added BLOCK code.
  !
  use chm_kinds
  use dimens_fcm
  use consta
  use mmffm
  use number
  use stream
#if KEY_BLOCK==1
  use block_fcm
#endif /*  BLOCK*/
  implicit none
  !
  integer ict(*)        ! angle parameter pointer
  real(chm_real)  myAnglFC(*)    ! angle bending fc
  real(chm_real)  x(*),y(*),z(*) ! coordinates
  real(chm_real)  dx(*),dy(*),dz(*)! first derivatives
  real(chm_real)  HESS(*)          ! second derivatives
  integer derivs          ! 0=only energy, 1=first der.
  !                           ! 2=sec. der. diag. 3=sd all
  integer AtomType,t_class
  external AtomType,t_class
  !
  COMMON/DTDQ/DI(3),DJ(3),DK(3),DL(3)
  COMMON/D2TDQ2/DII(3,3),DIJ(3,3),DIK(3,3),DIL(3,3),DJJ(3,3), &
       DJK(3,3),DJL(3,3),DKK(3,3),DKL(3,3),DLL(3,3)
  !
  character(len=10) value
  integer iia, idx, im1t3, imj, imk
  integer ja, jdx, jm1t3, jmk, ka, kdx, km1t3
  integer mii, mij, mik, mjj, mjk, mkk
  integer n, n1, n2, ngl
  logical constr
  real(chm_real)  c2
  real(chm_real)  di, dii, dij, dik, dj, djj, djk, dk, dkk
  real(chm_real)  dl, dil, djl, dkl, dll
  real(chm_real)  ctheta
#if KEY_BLOCK==1
  real(chm_real) COEF
#endif /*  BLOCK*/
  !
  !     CALCULATE STERIC ENERGY ASSOCIATED WITH DEFORMATION OF LINEAR 
  !     ANGLE IDX,JDX,KDX.
  !
  !        write(outu,*) 'ngl, ict(ngl) =',ngl,ict(ngl)
  CTHETA = CANGLE(derivs,IDX,JDX,KDX,1,X,Y,Z)
  C2=myAnglFC(ICT(NGL))
  IF(derivs.EQ.0) GOTO 3800
#if KEY_BLOCK==1
  IF (QBLOCK) THEN
     C2=C2*COEF
  ENDIF
  IF(.NOT. NOFORC) THEN
#endif /*  BLOCK*/
     !     CALCULATE FIRST DERIVATIVES
     DX(IDX) = DX(IDX) + C2*DI(1)
     DY(IDX) = DY(IDX) + C2*DI(2)
     DZ(IDX) = DZ(IDX) + C2*DI(3)
     DX(JDX) = DX(JDX) + C2*DJ(1)
     DY(JDX) = DY(JDX) + C2*DJ(2)
     DZ(JDX) = DZ(JDX) + C2*DJ(3)
     DX(KDX) = DX(KDX) + C2*DK(1)
     DY(KDX) = DY(KDX) + C2*DK(2)
     DZ(KDX) = DZ(KDX) + C2*DK(3)
     IF(derivs.EQ.1) GOTO 3800
     !     CALCULATE SECOND DERIVATIVES
1100 IM1T3=(IDX-1)*3
     JM1T3=(JDX-1)*3
     KM1T3=(KDX-1)*3
     IMJ=IDX-JDX
     IMK=IDX-KDX
     JMK=JDX-KDX
     DO N1=1,3
        IIA=(IM1T3+N1)*(IM1T3+N1-1)/2
        JA=(JM1T3+N1)*(JM1T3+N1-1)/2
        KA=(KM1T3+N1)*(KM1T3+N1-1)/2
        DO N2=1,N1
           MII=IIA+IM1T3+N2
           MJJ=JA+JM1T3+N2
           MKK=KA+KM1T3+N2
           HESS(MII)=HESS(MII)+C2*DII(N1,N2)
           HESS(MJJ)=HESS(MJJ)+C2*DJJ(N1,N2)
           HESS(MKK)=HESS(MKK)+C2*DKK(N1,N2)
        ENDDO
        IF(derivs.EQ.2) GOTO 3700
        DO N2=1,3
           IF(IMJ.GE.0) THEN
              MIJ=IIA+JM1T3+N2
              HESS(MIJ)=HESS(MIJ)+C2*DIJ(N1,N2)
           ELSE
              MIJ=JA+IM1T3+N2
              HESS(MIJ)=HESS(MIJ)+C2*DIJ(N2,N1)
           ENDIF
           IF(IMK.GE.0) THEN
              MIK=IIA+KM1T3+N2
              HESS(MIK)=HESS(MIK)+C2*DIK(N1,N2)
           ELSE
              MIK=KA+IM1T3+N2
              HESS(MIK)=HESS(MIK)+C2*DIK(N2,N1)
           ENDIF
           IF(JMK.GE.0) THEN
              MJK=JA+KM1T3+N2
              HESS(MJK)=HESS(MJK)+C2*DJK(N1,N2)
           ELSE
              MJK=KA+JM1T3+N2
              HESS(MJK)=HESS(MJK)+C2*DJK(N2,N1)
           ENDIF
        ENDDO
3700    CONTINUE
     ENDDO
#if KEY_BLOCK==1
  ENDIF  ! (.NOT.NOFORC)
#endif /*  BLOCK*/
3800 CONTINUE
  !
  RETURN
END SUBROUTINE EANGLIN_MM

!=======================================================================
! SUBROUTINE EBOND_MM : STERIC ENERGY ASSOCIATED WITH BOND STRETCHING
! CALCULATE STERIC ENERGY ASSOCIATED WITH BOND STRETCHING FOR BOND IB.
!=======================================================================
!
SUBROUTINE EBOND_MM(EB,IB,JB,ICB,NBOND,myBondEq,myBondFC,CSTRX, &
     X,Y,Z,DX,DY,DZ,HESS,IUPT,derivs, &
     QECONT,ECONT,ICONBH,ISKB)
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   Adapted by R. Czerminski of Molecular Simulations, Inc. from
  !  code developed at Merck and Co., Inc. by T. Halgren and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !
  ! Jay Banks 25 Oct 95: changed local variable name IA to IIA, to avoid 
  ! conflict with variable in common (mmff.f90).
  ! Jay Banks 25 Oct 95: changed argument cstr to cstrx, to avoid conflict
  ! with variable in common (mmff.f90).
  !
  ! Jay Banks 27 Nov 95: changed XYZ, DXYZ to X, Y, Z, DX, DY, DZ.
  !
  !  28 Nov 95 Jay Banks: removed space from all "GO TO"s, as suggested in
  !  developer.doc.
  !
  !  29 Nov 95 Jay Banks: changed arithmetic IFs to IF ... THEN ... ELSE.
  !
  !  29 Nov 95 Jay Banks: put in BLOCK code.
  !
  !  19 May 2011 Antti-Pekka Hynninen: added support for domdec
  !
  use chm_kinds
  use dimens_fcm
  use consta
  use exfunc
  use number
  use parallel
  use stream
  use string
  use mmffm
#if KEY_BLOCK==1
  use block_fcm
#endif /*  BLOCK*/
#if KEY_DOMDEC_MMFF==1 /* domdec and mmff are incompatible */
  use domdec_bonded,only:nbondtbl,bondtbl 
#endif
  implicit none
  !
  real(chm_real)  EB               ! total bond energy
  integer IB(*),JB(*)
  integer icb(*)           ! pointers for bond parameters
  integer nbond            ! number of bonds
  real(chm_real)  myBondEq(*)        ! bond equilibrium value
  real(chm_real)  myBondFC(*)        ! stretch force constant
  real(chm_real)  CSTRX            ! MMFF cubic-stretch force constant;
  !                                normally 2 A^^-1
  real(chm_real)  X(*),Y(*),Z(*)    ! coordinates
  real(chm_real)  DX(*),DY(*),DZ(*)   ! first derivatives
  real(chm_real)  HESS(*)          ! second derivatives
  INTEGER IUPT(*)
  integer derivs           ! flag for derivatives calculation
  LOGICAL QECONT
  real(chm_real)  ECONT(*)
  INTEGER ICONBH
  INTEGER ISKB(*)
  !
  integer AtomType,b_class
  external AtomType,b_class
  !
  character(len=10) value
  integer i, iia, im1t3, imj, j, ja
  integer jm1t3, mii, mij, mjj, n, n1, n2
  integer nb
  logical constr
  real(chm_real)  c1, c2, csdr, dr
  !     real(chm_real)  cutoff
  real(chm_real)  f4, fcsdr2, haa, hab
  real(chm_real)  DI(3),DJ(3)
  real(chm_real)  STRAIN
#if KEY_BLOCK==1
  INTEGER IBL, JBL, KK
  real(chm_real) COEF
#endif /*  BLOCK*/
#if KEY_DOMDEC_MMFF==1 /* domdec and mmff are incompatible */
  integer nn   
#endif
  !
  !   SUPPRESS FLOATING-POINT UNDERFLOW MESSAGES
  !
  !     CALL LIB$FLT_UNDER(0)
  eb=0
  IF (NBOND.EQ.0) RETURN
#if KEY_DOMDEC_MMFF==1 /* domdec and mmff are incompatible */
  if (nbondtbl == 0) return 
#endif
  !     CUTOFF=0.33D0
  F4=7.D0/12.D0
  !
  IF(QECONT .AND. PRNLEV.GE.2) WRITE(OUTU,6510)
6510 FORMAT(/,'        B O N D   S T R E T C H I N G      ',// &
          ' ------ATOMNAMES------   ATOM TYPES   FF     BOND     IDEAL', &
          '             STRAIN     FORCE'/ &
          '   I          J',12X,'I    J','   CLASS',3X, &
          'LENGTH   LENGTH    DIFF.    ENERGY   CONSTANT'/ &
          1X,88('-'))
  !
  constr=ICONBH.ne.0
  !
#if KEY_DOMDEC_MMFF==1 /* domdec and mmff are incompatible */
  do nn=1,nbondtbl
     nb = bondtbl(nn)
#else /* (domdec)*/
#if KEY_PARALLEL==1
  DO NB=MYNODP,NBOND,NUMNOD
#else /**/
  DO NB=1,NBOND
#endif 
#endif /* (domdec)*/
     if(constr) then
        if(iskb(nb).ne.0) goto 2000
     endif
     i=IB(nb)
     j=JB(nb)
     IMJ=I-J
     !
     !     THE ARRAY EIJ HOLDS THE X,Y,Z COMPONENTS OF THE UNIT VECTOR
     !     FROM ATOM I TO ATOM J.  EIJ(4) IS THE BOND LENGTH.
     !     BL IS THE IDEAL BOND LENGTH.
     !     EIJ(N) = -EJI(N)
     !
     EIJ(1)=X(J)-X(I)
     EIJ(2)=Y(J)-Y(I)
     EIJ(3)=Z(J)-Z(I)
     EIJ(4)=0.0D0
     DO N=1,3
        EIJ(4)=EIJ(4)+EIJ(N)*EIJ(N)
     ENDDO
     IF(EIJ(4).LE.0.) then
        write(scrtch,'(a,2i5)') 'ZERO BOND LENGTH: ',i,j
        call wrndie(-3,'<ebond_mm>',scrtch(:23))
        eij(4)=rsmall
     endif
     EIJ(4)=SQRT(EIJ(4))
     DO N=1,3
        EIJ(N)=EIJ(N)/EIJ(4)
        EJI(N)=-EIJ(N)
     ENDDO
     EJI(4)=EIJ(4)
     !
     !     WRITE(OUTU,*) NB,I,J,EIJ(4),BL(NB)
     !     CALCULATE STERIC ENERGY
     !
     
     !        dr=abs(bl(nb)-BondEq(icb(nb)))+abs(bk(nb)-BondFC(icb(nb)))
     !        if(dr.gt.RSMALL) then
     !          WRITE(OUTU,'(a)')      ' ebond_mm> debug: parameter error...'
     !          WRITE(OUTU,'(a,2i5)')  ' ebond_mm> nb,icb(nb)=',nb,icb(nb)
     !          dr=BondFC(icb(nb))
     !          WRITE(OUTU,'(a,2f10.5)') ' ebond_mm> bk,BondFC=',bk(nb),dr
     !          WRITE(OUTU,'(a,2f10.5)') ' ebond_mm> bl,BondEq=',
     !     &                             bl(nb),BondEq(icb(nb))
     !          stop
     !        endif
     
     DR=EIJ(4)-myBondEq(ICB(NB))
     C2=myBondFC(ICB(NB))
     C1=C2*DR
     STRAIN=C1*DR*HALF
     !     ADD NTH DEGREE TERM(S)
     CSDR=CSTRX*DR
     FCSDR2=F4*CSDR*CSDR
     STRAIN=STRAIN*(ONE+CSDR+FCSDR2)
     C1=C1*(ONE+ONEPT5*CSDR+TWO*FCSDR2)
     C2=C2*(ONE+THREE*CSDR+SIX*FCSDR2)
#if KEY_BLOCK==1
     IF (QBLOCK) THEN
        IBL = IBLCKP(I)
        JBL = IBLCKP(J)
        IF (JBL .LT. IBL) THEN
           KK=JBL
           JBL=IBL
           IBL=KK
        ENDIF
        KK=IBL+JBL*(JBL-1)/2
        COEF = BLCOEP(KK)
        STRAIN=STRAIN*COEF
        C1=C1*COEF
        C2=C2*COEF
     ENDIF
#endif /*  BLOCK*/
     EB=EB+STRAIN
#if KEY_BLOCK==1
     IF (.NOT. NOFORC) THEN
#endif /*  BLOCK*/
        IF(derivs.EQ.0) GOTO 1900
        !     CALCULATE FIRST DERIVATIVES
        DO N=1,3
           DI(N)=EJI(N)
           DJ(N)=EIJ(N)
        ENDDO
        DX(I)=DX(I)+C1*DI(1)
        DY(I)=DY(I)+C1*DI(2)
        DZ(I)=DZ(I)+C1*DI(3)
        DX(J)=DX(J)+C1*DJ(1)
        DY(J)=DY(J)+C1*DJ(2)
        DZ(J)=DZ(J)+C1*DJ(3)
        IF(derivs.EQ.1) GOTO 1900
        !     CALCULATE SECOND DERIVATIVES
700     IM1T3=(I-1)*3
        JM1T3=(J-1)*3
        DO N1=1,3
           IIA=(IM1T3+N1)*(IM1T3+N1-1)/2
           JA=(JM1T3+N1)*(JM1T3+N1-1)/2
           DO N2=1,N1
              MII=IIA+IM1T3+N2
              MJJ=JA+JM1T3+N2
              IF(N1.NE.N2) THEN
                 HAA=EIJ(N1)*EJI(N2)/EIJ(4)
              ELSE
                 HAA=(1.0D0+EIJ(N1)*EJI(N2))/EIJ(4)
              ENDIF
              HESS(MII)=HESS(MII)+C2*DI(N1)*DI(N2)+C1*HAA
              HESS(MJJ)=HESS(MJJ)+C2*DJ(N1)*DJ(N2)+C1*HAA
           ENDDO
           IF(derivs.EQ.2) GOTO 1800
           DO N2=1,3
              IF(IMJ.GE.0) THEN
                 MIJ=IIA+JM1T3+N2
              ELSE
                 MIJ=JA+IM1T3+N2
              ENDIF
              IF(N1.NE.N2) THEN
                 HAB=EIJ(N1)*EIJ(N2)/EIJ(4)
              ELSE
                 HAB=(EIJ(N1)*EIJ(N2)-1.0D0)/EIJ(4)
              ENDIF
              HESS(MIJ)=HESS(MIJ)+C2*DI(N1)*DJ(N2)+C1*HAB
           ENDDO
1800       CONTINUE
        ENDDO
1900          CONTINUE
#if KEY_BLOCK==1
     ENDIF  !(.NOT.NOFORC)
#endif /*  BLOCK*/
     IF(QECONT .AND. PRNLEV.GE.2) WRITE(OUTU,6511) &
          QNAME(I),QNAME(J),AtomType(I),AtomType(J),b_class(NB),EIJ(4), &
          myBondEq(ICB(NB)),DR, STRAIN, myBondFC(ICB(NB))/MDAKCAL
6511 FORMAT(1X,2(A,1X),2I5,I6,3X,F8.3,1X,F8.3,1X,F8.3,2X,F8.3,2X,F8.3)
2000 CONTINUE
  ENDDO
  IF(QECONT .AND. PRNLEV.GE.2) THEN
     CALL WFLOAT(4,10,VALUE,eb)
     write(outu,2200) value
2200 FORMAT(/'      TOTAL BOND STRAIN ENERGY = ',A//)
  endif
  !
  RETURN
END SUBROUTINE EBOND_MM

!=======================================================================
! SUBROUTINE EOOPL : STERIC ENERGY ASSOCIATED WITH OUT-OF-PLANE ANGLE
! CALCULATE STERIC ENERGY ASSOCIATED WITH OUT-OF-PLANE ANGLE
! BENDING INVOLVING ATOM JDX.
!=======================================================================
!
SUBROUTINE EOOPL(eoop,IT,JT,KT,LTHETA,icoop,ntheta, &
     OoplFC,x,y,z,dx,dy,dz,HESS,derivs, &
     QECONT,ECONT,ICONOP,ISKO)
        !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   Adapted by R. Czerminski of Molecular Simulations, Inc. from
  !  code developed at Merck and Co., Inc. by T. Halgren and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !
  ! Jay Banks 25 Oct 95: changed local variable name IA to IIA, to avoid 
  ! conflict with variable in common (mmff.f90).
  !
  ! Jay Banks 27 Nov 95: changed XYZ, DXYZ to X, Y, Z, DX, DY, DZ.
  !
  !  28 Nov 95 Jay Banks: removed space from all "GO TO"s, as suggested in
  !  developer.doc.
  !
  !  29 Nov 95 Jay Banks: changed arithmetic IFs to IF ... THEN ... ELSE.
  !
  !  30 Nov 95 Jay Banks: added BLOCK code.
  !
  !  19 May 2011 Antti-Pekka Hynninen: added support for domdec
  !
  use chm_kinds
  use dimens_fcm
  use consta
  use exfunc
  use number
  use stream
  use parallel
#if KEY_BLOCK==1
  use block_fcm
#endif /*  BLOCK*/
#if KEY_DOMDEC_MMFF==1 /* domdec and mmff are incompatible */
  use domdec_bonded,only:nangletbl,angletbl  
#endif
  implicit none
  !
  real(chm_real)  eoop                ! total oop energy
  integer IT(*),JT(*) ! oop list
  integer KT(*),LTHETA(*) ! oop list
  integer icoop(*)            ! pointer to parameters
  integer ntheta              ! number of angles
  real(chm_real)  OoplFC(*)           ! force constants
  real(chm_real)  x(*),y(*),z(*)       ! coordinates
  real(chm_real)  dx(*),dy(*),dz(*)      ! first derivatives
  real(chm_real)  HESS(*)             ! second derivatives
  integer derivs              ! derivatives flag
  LOGICAL QECONT
  real(chm_real)  ECONT(*)
  INTEGER ICONOP,ISKO(*)      ! IF(ICONOP.NE.0), consult ISKO to
  !                                 ! determine which angles to skip.
  !
  CHARACTER(len=10) VALUE
  integer iia, idx, im1t3, imj, imk, iml
  integer ja, jdx, jm1t3, jmk, jml, jo, ka, kdx, km1t3
  integer kml, la, ldx, lm1t3, mii, mij, mik, mil, mjj, mjk
  integer mjl, mkk, mkl, mll, n1, n2, nn
  real(chm_real)  c1, c2, delta, di, dii, dij
  real(chm_real)  dik, dil, dj, djj, djk, djl, dk, dkk, dkl, dl, dll
  logical constr
  real(chm_real)  STRAIN
  !
#if KEY_BLOCK==1
  INTEGER IBL, JBL, KKK, LLL
  real(chm_real)  COEF
#endif /*  BLOCK*/
#if KEY_DOMDEC_MMFF==1 /* domdec and mmff are incompatible */
  integer jj  
#endif
        !
  COMMON/DDDQ/DI(3),DJ(3),DK(3),DL(3)
  COMMON/D2DDQ2/DII(3,3),DIJ(3,3),DIK(3,3),DIL(3,3),DJJ(3,3), &
       DJK(3,3),DJL(3,3),DKK(3,3),DKL(3,3),DLL(3,3)
  !
  integer AtomType
  external AtomType
  !
  !   SUPPRESS FLOATING-POINT UNDERFLOW MESSAGES
  !
  !      CALL LIB$FLT_UNDER(0)
  !
  !
  if(ntheta.le.0) return
  !
  constr=ICONOP.ne.0
  !
  !
  if(QECONT .and. prnlev.ge.2) WRITE(OUTU,7000) ' WILSON'
7000 FORMAT(/,' O U T - O F - P L A N E    B E N D I N G       '/ &
          '                                                 ',A/ &
          ' -----------ATOMS-----------  --ATOM TYPES--   OUT-OF-PLANE', &
          '   STERIC      FORCE'/ &
          '  I -- J -- K ... L            I  J  K  L         ANGLE ', &
          '      ENERGY    CONSTANT'/1X,80('-'))
  !
#if KEY_DOMDEC_MMFF==1 /* domdec and mmff are incompatible */
  do jj=1,nangletbl
     JO = angletbl(nn)
#else /* (domdec)*/
#if KEY_PARALLEL==1
  DO JO=MYNODP,NTHETA,NUMNOD
#else /**/
  DO JO=1,NTHETA
#endif 
#endif /* (domdec)*/
     if(constr) then
        if(isko(JO).ne.0) goto 4000
     endif
     IDX = IT(JO)
     JDX = JT(JO)
     KDX = KT(JO)
     LDX = LTHETA(JO)
     IF(ldx.EQ.0) GOTO 4000
     if(icoop(jo).le.0) goto 4000
     !       WRITE(OUTU,100) JO,IDX,JDX,KDX,LDX
     ! 100 FORMAT(I4,10I4)
     !     IT IS ASSUMED THAT THE PREFERRED VALUE OF DELTA IS ZERO (I.E.,
     !     A PLANAR CONFORMATJON FOR ALL FOUR ATOMS) IN ALL FORCE FIELDS.
     !     QUADRATIC POTENTIAL
     !     mmff uses the wilson OOP ANGLE
     !   COMPUTE ANGLE
     DELTA = OOPNGL(derivs,IDX,JDX,KDX,LDX,X,Y,Z)
400  STRAIN=0.0D0
     c2=OoplFC(icoop(jo))
     C1=C2*DELTA
     STRAIN=C1*DELTA*HALF
     !
#if KEY_BLOCK==1
     IF (QBLOCK) THEN
        IBL = IBLCKP(IDX)
        JBL = IBLCKP(JDX)
        KKK = IBLCKP(KDX)
        LLL = IBLCKP(LDX)
        IF (IBL .EQ. JBL) JBL=KKK
        IF (IBL .EQ. JBL) JBL=LLL
        IF (JBL .LT. IBL) THEN
           KKK=JBL
           JBL=IBL
           IBL=KKK
        ENDIF
        KKK=IBL+JBL*(JBL-1)/2
        COEF = BLCOEP(KKK)
        STRAIN=STRAIN*COEF
        C1=C1*COEF
        C2=C2*COEF
     ENDIF
#endif /*  BLOCK*/
1000 eoop=eoop+STRAIN
     !      WRITE(OUTU,310) JO,IDX,JDX,KDX,LDX
     !    .,DELTA*57.3,C2,STRAIN,SN
     ! 310 FORMAT(I4,4I4,4F10.4)
#if KEY_BLOCK==1
     IF (.NOT. NOFORC) THEN
#endif /*  BLOCK*/
        IF(derivs.EQ.0) GOTO 3500
        !     CALCULATE FIRST DERIVATIVES
        DX(IDX)=DX(IDX)+C1*DI(1)
        DY(IDX)=DY(IDX)+C1*DI(2)
        DZ(IDX)=DZ(IDX)+C1*DI(3)
        DX(JDX)=DX(JDX)+C1*DJ(1)
        DY(JDX)=DY(JDX)+C1*DJ(2)
        DZ(JDX)=DZ(JDX)+C1*DJ(3)
        DX(KDX)=DX(KDX)+C1*DK(1)
        DY(KDX)=DY(KDX)+C1*DK(2)
        DZ(KDX)=DZ(KDX)+C1*DK(3)
        DX(LDX)=DX(LDX)+C1*DL(1)
        DY(LDX)=DY(LDX)+C1*DL(2)
        DZ(LDX)=DZ(LDX)+C1*DL(3)
        IF(derivs.EQ.1) GOTO 3500
        !     CALCULATE SECOND DERIVATIVES
1400    IM1T3=(IDX-1)*3
        JM1T3=(JDX-1)*3
        KM1T3=(KDX-1)*3
        LM1T3=(LDX-1)*3
        IMJ=IDX-JDX
        IMK=IDX-KDX
        IML=IDX-LDX
        JMK=JDX-KDX
        JML=JDX-LDX
        KML=KDX-LDX
        DO N1=1,3
           IIA=(IM1T3+N1)*(IM1T3+N1-1)/2
           JA=(JM1T3+N1)*(JM1T3+N1-1)/2
           KA=(KM1T3+N1)*(KM1T3+N1-1)/2
           LA=(LM1T3+N1)*(LM1T3+N1-1)/2
           DO N2=1,N1
              MII=IIA+IM1T3+N2
              MJJ=JA+JM1T3+N2
              MKK=KA+KM1T3+N2
              MLL=LA+LM1T3+N2
              HESS(MII)=HESS(MII)+C2*DI(N1)*DI(N2)+C1*DII(N1,N2)
              HESS(MJJ)=HESS(MJJ)+C2*DJ(N1)*DJ(N2)+C1*DJJ(N1,N2)
              HESS(MKK)=HESS(MKK)+C2*DK(N1)*DK(N2)+C1*DKK(N1,N2)
              HESS(MLL)=HESS(MLL)+C2*DL(N1)*DL(N2)+C1*DLL(N1,N2)
           ENDDO
           IF(derivs.EQ.2) GOTO 3400
           DO N2=1,3
              IF(IMJ.GE.0) THEN
                 MIJ=IIA+JM1T3+N2
                 HESS(MIJ)=HESS(MIJ)+C2*DI(N1)*DJ(N2)+C1*DIJ(N1,N2)
              ELSE
                 MIJ=JA+IM1T3+N2
                 HESS(MIJ)=HESS(MIJ)+C2*DI(N2)*DJ(N1)+C1*DIJ(N2,N1)
              ENDIF
              IF(IMK.GE.0) THEN
                 MIK=IIA+KM1T3+N2
                 HESS(MIK)=HESS(MIK)+C2*DI(N1)*DK(N2)+C1*DIK(N1,N2)
              ELSE
                 MIK=KA+IM1T3+N2
                 HESS(MIK)=HESS(MIK)+C2*DI(N2)*DK(N1)+C1*DIK(N2,N1)
              ENDIF
              IF(IML.GE.0) THEN
                 MIL=IIA+LM1T3+N2
                 HESS(MIL)=HESS(MIL)+C2*DI(N1)*DL(N2)+C1*DIL(N1,N2)
              ELSE
                 MIL=LA+IM1T3+N2
                 HESS(MIL)=HESS(MIL)+C2*DI(N2)*DL(N1)+C1*DIL(N2,N1)
              ENDIF
              IF(JMK.GE.0) THEN
                 MJK=JA+KM1T3+N2
                 HESS(MJK)=HESS(MJK)+C2*DJ(N1)*DK(N2)+C1*DJK(N1,N2)
              ELSE
                 MJK=KA+JM1T3+N2
                 HESS(MJK)=HESS(MJK)+C2*DJ(N2)*DK(N1)+C1*DJK(N2,N1)
              ENDIF
              IF(JML.GE.0) THEN
                 MJL=JA+LM1T3+N2
                 HESS(MJL)=HESS(MJL)+C2*DJ(N1)*DL(N2)+C1*DJL(N1,N2)
              ELSE
                 MJL=LA+JM1T3+N2
                 HESS(MJL)=HESS(MJL)+C2*DJ(N2)*DL(N1)+C1*DJL(N2,N1)
              ENDIF
              IF(KML.GE.0) THEN
                 MKL=KA+LM1T3+N2
                 HESS(MKL)=HESS(MKL)+C2*DK(N1)*DL(N2)+C1*DKL(N1,N2)
              ELSE
                 MKL=LA+KM1T3+N2
                 HESS(MKL)=HESS(MKL)+C2*DK(N2)*DL(N1)+C1*DKL(N2,N1)
              ENDIF
           ENDDO
3400       CONTINUE
        ENDDO
3500    CONTINUE
#if KEY_BLOCK==1
     ENDIF  ! (.NOT.NOFORC)
#endif /*  BLOCK*/
     !
     if(QECONT .and. prnlev.ge.2) &
          WRITE(OUTU,7105) QNAME(LDX),AtName(IDX),AtName(JDX),AtName(KDX), &
          AtomType(LDX), AtomType(IDX), AtomType(JDX), AtomType(KDX), &
          DELTA*RADDEG,STRAIN, C2/MDAKCAL
7105 FORMAT(' ',4(A,1X),'  ',4I3,'      ',F8.3,'    ',F8.3,'   ',F9.4)
4000 CONTINUE
  ENDDO
  if(QECONT .and. prnlev.ge.2) then
     CALL WFLOAT(4,10,VALUE,EOOP)
     WRITE(OUTU,8200) VALUE
8200 FORMAT(/'     TOTAL OUT-OF-PLANE STRAIN ENERGY = ',A//)
  endif
  !
  RETURN
END SUBROUTINE EOOPL

!=======================================================================
! SUBROUTINE EPHI_MM : torsional interactions in subject molecule
!=======================================================================
!
SUBROUTINE EPHI_MM(EDIHE,IP,JP,KP,LP,ICP,NPHI, &
     myTorFC,X,Y,Z,DX,DY,DZ,HESS,IUPT,DERIVS, &
     QECONT,ECONT,ICONHP,ISKP)
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   Adapted by R. Czerminski of Molecular Simulations, Inc. from
  !  code developed at Merck and Co., Inc. by T. Halgren and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !
  ! Jay Banks 25 Oct 95: changed local variable name IA to IIA, to avoid 
  ! conflict with variable in common (mmff.f90).
  !
  ! Jay Banks 27 Nov 95: changed XYZ, DXYZ to X, Y, Z, DX, DY, DZ.
  !
  !  28 Nov 95 Jay Banks: removed space from all "GO TO"s, as suggested in
  !  developer.doc.
  !
  !  29 Nov 95 Jay Banks: changed arithmetic IFs to IF ... THEN ... ELSE.
  !
  !  30 Nov 95 Jay Banks: added BLOCK code.
  !
  !  19 May 2011 Antti-Pekka Hynninen, added support for domdec
  !
  use chm_kinds
  use dimens_fcm
  use consta
  use exfunc
  use mmffm
  use stream
  use string
  use parallel
#if KEY_BLOCK==1
  use block_fcm
#endif /*  BLOCK*/
#if KEY_DOMDEC_MMFF==1 /* domdec and mmff are incompatible */
  use domdec_bonded,only:ndihetbl,dihetbl  
#endif
  implicit none
  !
  real(chm_real)  EDIHE         ! dihedral energy
  integer IP(*),JP(*),KP(*),LP(*) ! dihedral list
  integer icp(*)        ! dihedral parameter pointer
  integer nphi          ! number of dihedrals
  real(chm_real)  myTorFC(3,MAXCP/3)  ! 1,2,3-fold torsion parameter
  !     real(chm_real)  V1(*)         ! 1-fold torsion parameter
  !     real(chm_real)  V2(*)         ! 2-fold torsion parameter
  !     real(chm_real)  V3(*)         ! 3-fold torsion parameter
  real(chm_real)  X(*),Y(*),Z(*)   ! coordinates
  real(chm_real)  DX(*),DY(*),DZ(*)! first derivatives
  real(chm_real)  HESS(*)       ! second derivatives
  integer DERIVS          ! derivatives flag
  LOGICAL QECONT
  real(chm_real)  ECONT(*)
  INTEGER ICONHP
  INTEGER ISKP(*)
  INTEGER IUPT(*)
  !
  integer AtomType,p_class
  external AtomType,p_class
  !
  character(len=10) value
  integer i, iia, idx, ifld, im1t3, imj, imk
  integer iml, ja, jdx, jm1t3, jmk, jml, ka, kdx, km1t3, kml
  integer la, ldx, lm1t3, mii, mij, mik, mil, mjj, mjk, mjl
  integer mkk, mkl, mll, n, n1, n2, nd
  logical constr
  real(chm_real)  arg, c1, c2, di, dii, dij, dik, dil, dj,  &
       djj, djk, djl
  real(chm_real)  dk, dkk, dkl, dl, dll, phi
  real(chm_real)  V(3), STRAIN
  !
#if KEY_BLOCK==1
  INTEGER IBL, JBL, KKK, LLL
  real(chm_real)  COEF
#endif /*  BLOCK*/
#if KEY_DOMDEC_MMFF==1 /* domdec and mmff are incompatible */
  integer nn 
#endif
  !
  COMMON/DPDQ/DI(3),DJ(3),DK(3),DL(3)
  COMMON/D2PDQ2/DII(3,3),DIJ(3,3),DIK(3,3),DIL(3,3),DJJ(3,3), &
       DJK(3,3),DJL(3,3),DKK(3,3),DKL(3,3),DLL(3,3)
  !
  !   SUPPRESS FLOATING-POINT UNDERFLOW MESSAGES
  !
  !      CALL LIB$FLT_UNDER(0)
  !
  !     CALCULATE STERIC ENERGY ASSOCIATED WITH TWISTING INVOLVING
  !     ATOM JDX.
  !
  EDIHE=0
  IF(NPHI.LE.0) RETURN
#if KEY_DOMDEC_MMFF==1 /* domdec and mmff are incompatible */
  if (ndihetbl == 0) return  
#endif

  !...##IF DEBUG
  !      call getenv('DEBUG_EPHI_MM',SCRTCH)
  !      DEBUG=SCRTCH.ne.' '
  !...##ENDIF
  !
  constr=ICONHP.ne.0
  if(QECONT .and. prnlev.ge.2 &
#if KEY_DEBUG==1
       .or.DEBUG &
#endif 
       ) WRITE(outu,7000)
7000 FORMAT(/,'    T O R S I O N A L    '// &
          ' --------------ATOMS--------------  ---ATOM TYPES---   FF   ', &
          '  TORSION    STERIC    --FORCE CONSTANTS--'/ &
          '  I       J          K       L        I   J   K   L   CLASS  ', &
          '  ANGLE     ENERGY     V1      V2      V3'/1X,102('-'))
  
  IFLD=3
#if KEY_DOMDEC_MMFF==1 /* domdec and mmff are incompatible */
  do nn=1,ndihetbl
     nd = dihetbl(nn)
#else /* (domdec)*/
#if KEY_PARALLEL==1
  DO ND=MYNODP,NPHI,NUMNOD
#else /**/
  DO ND=1,NPHI
#endif 
#endif /* (domdec)*/
     if(constr) then
        if(iskp(nd).ne.0) goto 3000
     endif
     if(icp(nd).le.0) goto 3000
     IDX=IP(ND)
     JDX=JP(ND)
     KDX=KP(ND)
     LDX=LP(ND)
     PHI=TORNGL(DERIVS,IDX,JDX,KDX,LDX,X,Y,Z)
     STRAIN=0.0D0
     !     IFOLD TERM FUNCTION
     IF(ND.LT.1.OR.ND.GT.MAXP) then
        WRITE(SCRTCH,'(a,3i5)') ' MAXP, ND, NPHI',MAXP,ND,NPHI
        call wrndie(-5,'<diheds>',SCRTCH(:31))
     ENDIF
     !       WRITE(6,*) ' NPHI,IDX ...',NPHI,IDX,JDX,KDX,LDX
     V(1)=myTorFC(1,ICP(nd))
     V(2)=-myTorFC(2,ICP(nd))
     V(3)=myTorFC(3,ICP(nd))
     !
     !   SKIP THIS TORSION IF ALL FORCE COMPONENTS ARE ZERO
     !       IF(V(1).EQ.0..AND.V(2).EQ.0..AND.V(3).EQ.0.) GOTO 3000
     ! NOTE: this test has been moved to komegam where icp is set
     !       to 0 for all zero force constants
     !
     STRAIN=V(1)-V(2)+V(3)
     C1=0.0D0
     C2=0.0D0
     DO I=1,IFLD
        ARG=I*PHI
        STRAIN=STRAIN + V(I)*COS(ARG)
        C1=C1 + V(I)*I*SIN(ARG)
        C2=C2 + V(I)*I*I*COS(ARG)
     ENDDO
     C1=-C1
     C2=-C2
     !
#if KEY_BLOCK==1
     IF (QBLOCK) THEN
        IBL = IBLCKP(IDX)
        JBL = IBLCKP(JDX)
        KKK = IBLCKP(KDX)
        LLL = IBLCKP(LDX)
        IF (IBL .EQ. JBL) JBL=KKK
        IF (IBL .EQ. JBL) JBL=LLL
        IF (JBL .LT. IBL) THEN
           KKK=JBL
           JBL=IBL
           IBL=KKK
        ENDIF
        KKK=IBL+JBL*(JBL-1)/2
        COEF = BLCOEP(KKK)
        STRAIN=STRAIN*COEF
        C1=C1*COEF
        C2=C2*COEF
     ENDIF
#endif /*  BLOCK*/
200  EDIHE=EDIHE+STRAIN
     !     WRITE(OUTU,*) IDX,JDX,KDX,LDX,nd,STRAIN
#if KEY_BLOCK==1
     IF (.NOT. NOFORC) THEN
#endif /*  BLOCK*/
        IF(DERIVS.EQ.0) GOTO 2600
        !     CALCULATE FIRST DERIVATIVES
        DX(IDX)=DX(IDX)+C1*DI(1)
        DY(IDX)=DY(IDX)+C1*DI(2)
        DZ(IDX)=DZ(IDX)+C1*DI(3)
        DX(JDX)=DX(JDX)+C1*DJ(1)
        DY(JDX)=DY(JDX)+C1*DJ(2)
        DZ(JDX)=DZ(JDX)+C1*DJ(3)
        DX(KDX)=DX(KDX)+C1*DK(1)
        DY(KDX)=DY(KDX)+C1*DK(2)
        DZ(KDX)=DZ(KDX)+C1*DK(3)
        DX(LDX)=DX(LDX)+C1*DL(1)
        DY(LDX)=DY(LDX)+C1*DL(2)
        DZ(LDX)=DZ(LDX)+C1*DL(3)
        IF(DERIVS.EQ.1) GOTO 2600
        !     CALCULATE SECOND DERIVATIVES
500     IM1T3=(IDX-1)*3
        JM1T3=(JDX-1)*3
        KM1T3=(KDX-1)*3
        LM1T3=(LDX-1)*3
        IMJ=IDX-JDX
        IMK=IDX-KDX
        IML=IDX-LDX
        JMK=JDX-KDX
        JML=JDX-LDX
        KML=KDX-LDX
        DO N1=1,3
           IIA=(IM1T3+N1)*(IM1T3+N1-1)/2
           JA=(JM1T3+N1)*(JM1T3+N1-1)/2
           KA=(KM1T3+N1)*(KM1T3+N1-1)/2
           LA=(LM1T3+N1)*(LM1T3+N1-1)/2
           DO N2=1,N1
              MII=IIA+IM1T3+N2
              MJJ=JA+JM1T3+N2
              MKK=KA+KM1T3+N2
              MLL=LA+LM1T3+N2
              HESS(MII)=HESS(MII)+C2*DI(N1)*DI(N2)+C1*DII(N1,N2)
              HESS(MJJ)=HESS(MJJ)+C2*DJ(N1)*DJ(N2)+C1*DJJ(N1,N2)
              HESS(MKK)=HESS(MKK)+C2*DK(N1)*DK(N2)+C1*DKK(N1,N2)
              HESS(MLL)=HESS(MLL)+C2*DL(N1)*DL(N2)+C1*DLL(N1,N2)
           ENDDO
           IF(DERIVS.EQ.2) GOTO 2500
           DO N2=1,3
              IF(IMJ.GE.0) THEN
                 MIJ=IIA+JM1T3+N2
                 HESS(MIJ)=HESS(MIJ)+C2*DI(N1)*DJ(N2)+C1*DIJ(N1,N2)
              ELSE
                 MIJ=JA+IM1T3+N2
                 HESS(MIJ)=HESS(MIJ)+C2*DI(N2)*DJ(N1)+C1*DIJ(N2,N1)
              ENDIF
              IF(IMK.GE.0) THEN
                 MIK=IIA+KM1T3+N2
                 HESS(MIK)=HESS(MIK)+C2*DI(N1)*DK(N2)+C1*DIK(N1,N2)
              ELSE
                 MIK=KA+IM1T3+N2
                 HESS(MIK)=HESS(MIK)+C2*DI(N2)*DK(N1)+C1*DIK(N2,N1)
              ENDIF
              IF(IML.GE.0) THEN
                 MIL=IIA+LM1T3+N2
                 HESS(MIL)=HESS(MIL)+C2*DI(N1)*DL(N2)
              ELSE
                 MIL=LA+IM1T3+N2
                 HESS(MIL)=HESS(MIL)+C2*DI(N2)*DL(N1)
              ENDIF
              IF(JMK.GE.0) THEN
                 MJK=JA+KM1T3+N2
                 HESS(MJK)=HESS(MJK)+C2*DJ(N1)*DK(N2)+C1*DJK(N1,N2)
              ELSE
                 MJK=KA+JM1T3+N2
                 HESS(MJK)=HESS(MJK)+C2*DJ(N2)*DK(N1)+C1*DJK(N2,N1)
              ENDIF
              IF(JML.GE.0) THEN
                 MJL=JA+LM1T3+N2
                 HESS(MJL)=HESS(MJL)+C2*DJ(N1)*DL(N2)+C1*DJL(N1,N2)
              ELSE
                 MJL=LA+JM1T3+N2
                 HESS(MJL)=HESS(MJL)+C2*DJ(N2)*DL(N1)+C1*DJL(N2,N1)
              ENDIF
              IF(KML.GE.0) THEN
                 MKL=KA+LM1T3+N2
                 HESS(MKL)=HESS(MKL)+C2*DK(N1)*DL(N2)+C1*DKL(N1,N2)
              ELSE
                 MKL=LA+KM1T3+N2
                 HESS(MKL)=HESS(MKL)+C2*DK(N2)*DL(N1)+C1*DKL(N2,N1)
              ENDIF
           ENDDO
2500       CONTINUE
        ENDDO
2600    CONTINUE
#if KEY_BLOCK==1
     ENDIF  ! (.NOT.NOFORC)
#endif /*  BLOCK*/
     !
     if(QECONT .and. prnlev.ge.2 &
#if KEY_DEBUG==1
          .or.DEBUG &
#endif 
          ) WRITE(outu,7105) &
          AtName(IDX),QNAME(JDX),QNAME(KDX),AtName(LDX), &
          AtomType(IDX), AtomType(JDX), AtomType(KDX), AtomType(LDX), &
          p_class(ND),PHI*RADDEG,STRAIN, myTorFC(1,ICP(nd))*2.0, &
          myTorFC(2,ICP(nd))*2.0,myTorFC(3,ICP(nd))*2.0
7105 FORMAT(' ',4(A,1X),'  ',4I4,I6,'    ',F8.3,'  ',F8.3,3F8.3)
3000 CONTINUE
  ENDDO
  if(QECONT .and. prnlev.ge.2) then
     CALL WFLOAT(4,10,VALUE,EDIHE)
     WRITE(OUTU,8200) VALUE
8200 FORMAT(/'   TOTAL TORSION STRAIN ENERGY = ',A//)
  endif
  !
  RETURN
END SUBROUTINE EPHI_MM

!=======================================================================
! SUBROUTINE STRBND : stretch-bend interactions in subject molecule
!=======================================================================
!
SUBROUTINE ESTRBND(ESTRB,ITHETAX,JTHETAX,KTHETAX,ICT,NTHETAX, &
     myAnglEq,IBONDX,JBONDX,ICB,myBondEq, &
     StrbListX,ICSTBNX,STBNPX, &
     X,Y,Z,DX,DY,DZ,HESS,derivs, &
     QECONT,ECONT,ICONAH,ISKT)
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   Adapted by R. Czerminski of Molecular Simulations, Inc. from
  !  code developed at Merck and Co., Inc. by T. Halgren and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !
  ! Jay Banks 25 Oct 95: changed local variable name IA to IIA, and 
  ! arguments ICSTBN and STBNP to ICSTBNX and STBNPX, to avoid conflict
  ! with variables in common (mmff.f90).
  !
  ! Jay Banks 27 Nov 95: changed XYZ, DXYZ to X, Y, Z, DX, DY, DZ.
  !
  ! Jay Banks 28 Nov 95: changed DOUBLE PRECISION STRAIN to real(chm_real).
  !
  !  28 Nov 95 Jay Banks: removed space from all "GO TO"s, as suggested in
  !  developer.doc.
  !
  !  29 Nov 95 Jay Banks: changed arithmetic IFs to IF ... THEN ... ELSE.
  !
  !  29 Nov 95 Jay Banks: added BLOCK code.
  !
  !  01 Dec 95 Jay Banks: added code to print out energy contributions.
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use consta
  use mmffm
  use parallel
  use psf
  use stream
#if KEY_BLOCK==1
  use block_fcm,only:blcoep,iblckp,noforc,qblock
#endif /*  BLOCK*/
  implicit none
  !
  !
  integer AtomType
  external AtomType
  !
  real(chm_real)  ESTRB              ! stretch-bend energy
  integer ITHETAX(*),JTHETAX(*)! angle list
  integer KTHETAX(*)
  integer NTHETAX             ! number of angles
  integer ICT(*)             ! angle parameter pointer
  real(chm_real)  myAnglEq(*)  ! angle equilibrium value
  integer IBONDX(*),JBONDX(*)  ! bond list
  integer ICB(*)             ! bond parameter pointer
  real(chm_real)  myBondEq(*) ! equilibrium bond lengths
  integer StrbListX(2,*)      ! bonds forming angle
  integer ICSTBNX(*)          ! stretch-bend parameter pointer
  real(chm_real)  STBNPX(2,*)         ! stretch-bend parameters
  real(chm_real)  X(*),Y(*),Z(*)      ! coordinates
  real(chm_real)  DX(*),DY(*),DZ(*)     ! first derivatives
  real(chm_real)  HESS(*)            ! second derivatives
  integer DERIVS             ! derivatives flag
  LOGICAL QECONT
  real(chm_real)  ECONT(*)
  INTEGER ICONAH             ! if nonzero, consult ISKT
  INTEGER ISKT(*)            ! if ISKT(I) (and ICONAH) nonzero, skip
  !                                ! angle I
  !
  integer iia, idx, nb, im1t3
  integer imj, imk, iml, ja, jdx, jm1t3, jmk, jml, ka, kdx
  integer km1t3, kml, la, ldx, lm1t3, mii, mij, mik
  integer mil, mjj, mjk, mjl, mkk, mkl, mll, n, n1, n2
  integer nd, nth, nn
  real(chm_real)  c2, cr, ct, di, dii, dij, dik, dil, dj
  real(chm_real)  djj, djk, djl, dk, dkk, dkl, dl, dll, dr, dth
  real(chm_real)  fk,ssb,theta
  logical constr
  character(len=80) SCRTCH
  character(len=10) value
  !
  real(chm_real) STRAIN
  COMMON/DTDQ/DI(3),DJ(3),DK(3),DL(3)
  COMMON/D2TDQ2/DII(3,3),DIJ(3,3),DIK(3,3),DIL(3,3),DJJ(3,3), &
       DJK(3,3),DJL(3,3),DKK(3,3),DKL(3,3),DLL(3,3)
#if KEY_BLOCK==1
  real(chm_real) COEF
  INTEGER IBL, JBL, KK
#endif /*  BLOCK*/
  !
  !     CALCULATE STERIC ENERGY ASSOCIATED WITH STRETCH-BEND OF ANGLE
  !     IDX,JDX,KDX.
  !
  !   SUPPRESS FLOATING-POINT UNDERFLOW MESSAGES
  !
  !      CALL LIB$FLT_UNDER(0)
  !
  !...##IF DEBUG
  !      call getenv('DEBUG_ESTRBND',SCRTCH)
  !      DEBUG=SCRTCH.ne.' '
  !      if(DEBUG) then
  !         write(outu,'(a,i5,f10.3)') ' DEBUG_ESTRBND> NTHETAX=',
  !     &                                               NTHETAX
  !         WRITE(OUTU,'(A)')' DEBUG_ESTRBND> nth,nb,I,J,K,L,ND,ICSTBNX,FK='
  !      endif
  !...##ENDIF
  ESTRB=0
  if(NTHETAX.le.0) return
  !
  constr=ICONAH.ne.0
  !
  !
  IF(QECONT.AND.PRNLEV.GE.2) WRITE(OUTU,7010)
7010 FORMAT(/,'      S T R E T C H   B E N D I N G         '// &
          ' -------ATOMS-------  -ATOM TYPES-    FF     VALENCE', &
          '      DELTA      DELTA     STRAIN      F CON ',/, &
          '  I       J       K     I    J    K  CLASS    ANGLE ', &
          '      ANGLE      R(I,J)    ENERGY       I-J' / &
          1X,97('-'))
  !
#if KEY_PARALLEL==1
  DO nth=MYNODP,NTHETAX,NUMNOD
#else /**/
  DO nth=1,NTHETAX
#endif 
     if(constr) then
        if(iskt(nth).ne.0) goto 3400
     endif
     if(icstbnx(nth).eq.0) goto 3400
     IDX=ITHETAX(nth)
     JDX=JTHETAX(nth)
     KDX=KTHETAX(nth)
     !
     !   do not include stretch-bend term if this is a "linear bond angle"
     !
     ! T.A. Halgren change (B980629.wy)
     !       if(mlinba(mtype(jdx)).ne.0) return
     if(mlinba(mtype(jdx)).ne.0) goto 3400
     ! end of change
     THETA=VANGLE(derivs,IDX,JDX,KDX,1,X,Y,Z)
     DTH=THETA-myAnglEq(ICT(nth))
     STRAIN=0.0D0
     DO ND=1,2
        if(ICSTBNX(nth).gt.0) then
           FK=STBNPX(ND,ICSTBNX(nth))
        else
           FK=STBNPX(3-ND,ABS(ICSTBNX(nth)))
        endif
        IF(FK.EQ.0.) GOTO 3300
        nb=StrbListX(ND,nth)
        if(nb.le.0 .or. nb.gt.MAXB) then
           write(SCRTCH,'(a,2i6)') &
                'nb out of range: nb,MAXB=',nb,MAXB
           call wrndie(-5,'<ESTRBND>',SCRTCH(:37))
           return
        endif
        LDX=IBONDX(nb)
        IF(LDX.EQ.JDX) LDX=JBONDX(nb)
        !...##IF DEBUG
        !      if(DEBUG) WRITE(6,'(A,8I5,F10.3)') ' DEBUG_ESTRBND> ',
        !     &          nth,nb,IDX,JDX,KDX,LDX,ND,ICSTBNX(nth),FK
        !      WRITE(6,'(A,8I5,F10.3)') ' DEBUG_ESTRBND> ',
        !     &          nth,nb,IDX,JDX,KDX,LDX,ND,ICSTBNX(nth),FK
        !...##ENDIF
        !       ADD NEEDED E() ELEMENTS
        EJL(1)=X(LDX)-X(JDX)
        EJL(2)=Y(LDX)-Y(JDX)
        EJL(3)=Z(LDX)-Z(JDX)
        EJL(4)=0.
        DO N=1,3
           EJL(4)=EJL(4)+EJL(N)*EJL(N)
        ENDDO
        EJL(4)=SQRT(EJL(4))
        ELJ(4)=EJL(4)
        DO N=1,3
           EJL(N)=EJL(N)/EJL(4)
           ELJ(N)=-EJL(N)
        ENDDO
        DR=EJL(4)-myBondEq(ICB(nb))
100     SSB=FK*DR*DTH
        CT=FK*DTH
        CR=FK*DR
        C2=FK
#if KEY_BLOCK==1
        IF (QBLOCK) THEN
           IBL = IBLCKP(IDX)
           JBL = IBLCKP(JDX)
           KK = IBLCKP(KDX)
           IF (IBL .EQ. JBL) JBL=KK
           IF (JBL .LT. IBL) THEN
              KK=JBL
              JBL=IBL
              IBL=KK
           ENDIF
           KK=IBL+JBL*(JBL-1)/2
           COEF = BLCOEP(KK)
           SSB=SSB*COEF
           CR=CR*COEF
           CT=CT*COEF
           C2=C2*COEF
        ENDIF
#endif /*  BLOCK*/
200     STRAIN=STRAIN+SSB
        !...##IF DEBUG
        !      if(DEBUG)
        !     & WRITE(OUTU,'(A,5F10.3)') ' DEBUG_ESTRBND> DR,DTH,FK,SSB,STRAIN='
        !     &                ,DR,DTH,FK,SSB,STRAIN
        !      WRITE(OUTU,'(A,5F10.3)') ' DEBUG_ESTRBND> DR,DTH,FK,SSB,STRAIN=',
        !     &                DR,DTH,FK,SSB,STRAIN
        !...##ENDIF
        IF(QECONT.AND.PRNLEV.GE.2) THEN
           IF(ND.EQ.1) THEN
              WRITE(OUTU,7100)AtNAME(IDX),QNAME(JDX),AtNAME(KDX), &
                   AtomTYPE(IDX),AtomTYPE(JDX),AtomTYPE(KDX),MDSTBN(NTH), &
                   THETA*RADDEG,DTH*RADDEG,DR,SSB, &
                   FK/MDAKCAL
           ELSE IF(ND.EQ.2) THEN
              WRITE(OUTU,7100)AtNAME(KDX),QNAME(JDX),AtNAME(IDX), &
                   AtomTYPE(KDX),AtomTYPE(JDX),AtomTYPE(IDX),MDSTBN(NTH), &
                   THETA*RADDEG,DTH*RADDEG,DR,SSB, &
                   FK/MDAKCAL
           ENDIF
7100       FORMAT(1X,3(A,1X),I3,I5,I5,I5,'    ', &
                F8.3,4F11.3)
        ENDIF
        IF(derivs.EQ.0) GOTO 3300
        !     CALCULATE FIRST DERIVATIVES
#if KEY_BLOCK==1
        IF (.NOT. NOFORC) THEN
#endif /*  BLOCK*/
           DX(IDX)=DX(IDX)+CR*DI(1)
           DY(IDX)=DY(IDX)+CR*DI(2)
           DZ(IDX)=DZ(IDX)+CR*DI(3)
           DX(JDX)=DX(JDX)+CT*ELJ(1)+CR*DJ(1)
           DY(JDX)=DY(JDX)+CT*ELJ(2)+CR*DJ(2)
           DZ(JDX)=DZ(JDX)+CT*ELJ(3)+CR*DJ(3)
           DX(KDX)=DX(KDX)+CR*DK(1)
           DY(KDX)=DY(KDX)+CR*DK(2)
           DZ(KDX)=DZ(KDX)+CR*DK(3)
           DX(LDX)=DX(LDX)+CT*EJL(1)
           DY(LDX)=DY(LDX)+CT*EJL(2)
           DZ(LDX)=DZ(LDX)+CT*EJL(3)
           IF(derivs.EQ.1) GOTO 3300
           !     CALCULATE SECOND DERIVATIVES
500        CT=CT/ELJ(4)
           IMJ=IDX-JDX
           IMK=IDX-KDX
           IML=IDX-LDX
           JMK=JDX-KDX
           JML=JDX-LDX
           KML=KDX-LDX
           IM1T3=(IDX-1)*3
           JM1T3=(JDX-1)*3
           KM1T3=(KDX-1)*3
           LM1T3=(LDX-1)*3
           DO N1=1,3
              IIA=(IM1T3+N1)*(IM1T3+N1-1)/2
              JA=(JM1T3+N1)*(JM1T3+N1-1)/2
              KA=(KM1T3+N1)*(KM1T3+N1-1)/2
              LA=(LM1T3+N1)*(LM1T3+N1-1)/2
              DO N2=1,N1
                 MII=IIA+IM1T3+N2
                 MJJ=JA+JM1T3+N2
                 MKK=KA+KM1T3+N2
                 MLL=LA+LM1T3+N2
                 HESS(MII)=HESS(MII)+CR*DII(N1,N2)
                 HESS(MJJ)=HESS(MJJ)+C2*(ELJ(N1)*DJ(N2)+ELJ(N2) &
                      *DJ(N1)) + CR*DJJ(N1,N2)+CT*EJL(N1) &
                      *ELJ(N2)
                 HESS(MKK)=HESS(MKK)+CR*DKK(N1,N2)
                 HESS(MLL)=HESS(MLL)+CT*EJL(N1)*ELJ(N2)
                 IF(IML.EQ.0) THEN
                    HESS(MII)=HESS(MII)+C2*(EJI(N1)*DI(N2)+EJI(N2) &
                         *DI(N1))
                 ELSE IF(KML.EQ.0) THEN
                    HESS(MKK)=HESS(MKK)+C2*(EJK(N1)*DK(N2)+EJK(N2) &
                         *DK(N1))
                 ENDIF
                 IF(N1.EQ.N2) THEN
                    HESS(MJJ)=HESS(MJJ)+CT
                    HESS(MLL)=HESS(MLL)+CT
                 ENDIF
              ENDDO
              IF(derivs.EQ.2) GOTO 3200
              DO N2=1,3
                 IF(IMJ.GE.0) THEN
                    MIJ=IIA+JM1T3+N2
                    HESS(MIJ)=HESS(MIJ)+C2*ELJ(N2)*DI(N1)+CR*DIJ(N1,N2)
                 ELSE
                    MIJ=JA+IM1T3+N2
                    HESS(MIJ)=HESS(MIJ)+C2*ELJ(N1)*DI(N2)+CR*DIJ(N2,N1)
                 ENDIF
                 IF(IMK.GE.0) THEN
                    MIK=IIA+KM1T3+N2
                    HESS(MIK)=HESS(MIK)+CR*DIK(N1,N2)
                 ELSE
                    MIK=KA+IM1T3+N2
                    HESS(MIK)=HESS(MIK)+CR*DIK(N2,N1)
                 ENDIF
                 IF(IML.GT.0) THEN
                    MIL=IIA+LM1T3+N2
                    HESS(MIL)=HESS(MIL)+C2*EJL(N2)*DI(N1)
                 ELSE IF(IML.LT.0) THEN
                    MIL=LA+IM1T3+N2
                    HESS(MIL)=HESS(MIL)+C2*EJL(N1)*DI(N2)
                 ENDIF
                 IF(JMK.GE.0) THEN
                    MJK=JA+KM1T3+N2
                    HESS(MJK)=HESS(MJK)+C2*ELJ(N1)*DK(N2)+CR*DJK(N1,N2)
                 ELSE
                    MJK=KA+JM1T3+N2
                    HESS(MJK)=HESS(MJK)+C2*ELJ(N2)*DK(N1)+CR*DJK(N2,N1)
                 ENDIF
                 IF(JML.GE.0) THEN
                    MJL=JA+LM1T3+N2
                    HESS(MJL)=HESS(MJL)+C2*DJ(N1)*EJL(N2) &
                         +CT*EJL(N1)*EJL(N2)
                 ELSE
                    MJL=LA+JM1T3+N2
                    HESS(MJL)=HESS(MJL)+C2*DJ(N2)*EJL(N1) &
                         +CT*EJL(N2)*EJL(N1)
                 ENDIF
                 IF(KML.GT.0) THEN
                    MKL=KA+LM1T3+N2
                    HESS(MKL)=HESS(MKL)+C2*DK(N1)*EJL(N2)
                 ELSE IF(KML.LT.0) THEN
                    MKL=LA+KM1T3+N2
                    HESS(MKL)=HESS(MKL)+C2*DK(N2)*EJL(N1)
                 ENDIF
                 IF(N1.EQ.N2) THEN
                    HESS(MJL)=HESS(MJL)-CT
                 ENDIF
              ENDDO
3200          CONTINUE
           ENDDO
#if KEY_BLOCK==1
        ENDIF  ! (.NOT.NOFORC)
#endif /*  BLOCK*/
3300    CONTINUE
     ENDDO
     ESTRB=ESTRB+STRAIN
3400 CONTINUE
  ENDDO
  if(QECONT .and. prnlev.ge.2) then
     CALL WFLOAT(4,10,VALUE,ESTRB)
     WRITE(OUTU,8200) VALUE
8200 FORMAT(/'     TOTAL STRETCH-BEND STRAIN ENERGY = ',A//)
  endif
  !
  RETURN
END SUBROUTINE ESTRBND

!=======================================================================
! real(chm_real) FUNCTION OOPNGL : CALCULATES THE OOPL ANGLE
! CALCULATES THE ANGLE (IN RADIANS) BY WHICH VECTOR <J,L> DEVIATES
! FROM THE PLANE DEFINED BY ATOMS I, J, AND K
!=======================================================================
FUNCTION OOPNGL(DERIVS,I,J,K,L,X,Y,Z) result(oopngl_rtn)
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   Adapted by R. Czerminski of Molecular Simulations, Inc. from
  !  code developed at Merck and Co., Inc. by T. Halgren and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !
  !  Jay Banks 27 Nov 95: changed XYZ array to X, Y, Z.
  !
  !  28 Nov 95 Jay Banks: removed space from all "GO TO"s, as suggested in
  !  developer.doc.
  !
  !  29 Nov 95 Jay Banks: changed arithmetic IFs to IF ... THEN ... ELSE.
  !
  use chm_kinds
  use dimens_fcm
  use mmffm
  use number,only:zero,one
  implicit none
  !
  integer DERIVS, i, j, k, l
  real(chm_real)  x(*),y(*),z(*),oopngl_rtn
  !
  integer m, n1, n2, nn
  real(chm_real)  cd2r, cdst, cdst2, cdstji, cdstjk, cosdl
  real(chm_real)  costh, cotth, di, dii, dij, dik, dil, dj, &
       djj, djk, djl
  real(chm_real)  dk, dkk, dkl, dl, dll, eii, ekk
  real(chm_real)  ONEPM, rji, rjk, rjl, sindl, sinth, tandl, tdrjl
  real(chm_real)  tds2t, tdst, theta, ti, tj, tk, tl, ttcs

  COMMON/DDDQ/DI(3),DJ(3),DK(3),DL(3)
  COMMON/D2DDQ2/DII(3,3),DIJ(3,3),DIK(3,3),DIL(3,3),DJJ(3,3), &
       DJK(3,3),DJL(3,3),DKK(3,3),DKL(3,3),DLL(3,3)
  COMMON/DTDQ/TI(3),TJ(3),TK(3),TL(3)
  !
  real(chm_real) A(3),B(3),C(3), &
       FJI(3,3),FJK(3,3),FJL(3,3)
  real(chm_real) :: DKRON(1:3,1:3)   !=(/ & 
!       one ,zero,zero, & 
!       zero,one ,zero, &
!       zero,zero,one /)

  dkron=zero
  dkron(1,1)=one
  dkron(2,2)=one
  dkron(3,3)=one
  !
  !   SUPPRESS UNDERFLOAT MESSAGES IF ON VAX
  !
  !      CALL LIB$FLT_UNDER(0)
  !
  !   GET NEEDED DIRECTION COSINES
  CALL DIRCOS(I,J,K,L,X,Y,Z)
  !   COMPUTE ANGLE
  A(1)=EJI(2)*EJK(3)-EJK(2)*EJI(3)
  A(2)=EJI(3)*EJK(1)-EJK(3)*EJI(1)
  A(3)=EJI(1)*EJK(2)-EJK(1)*EJI(2)
  SINDL=0.0D0
  DO NN=1,3
     SINDL=SINDL+A(NN)*EJL(NN)
  ENDDO
  THETA=VANGLE(1,I,J,K,0,X,Y,Z)
  SINTH=SIN(THETA)
  SINDL=SINDL/SINTH
  OOPNGL_RTN=ASIN(SINDL)
  IF(DERIVS.EQ.0) RETURN
  !     CALCULATE FIRST DERIVATIVES
  B(1)=EJK(2)*EJL(3)-EJL(2)*EJK(3)
  B(2)=EJK(3)*EJL(1)-EJL(3)*EJK(1)
  B(3)=EJK(1)*EJL(2)-EJL(1)*EJK(2)
  C(1)=EJL(2)*EJI(3)-EJI(2)*EJL(3)
  C(2)=EJL(3)*EJI(1)-EJI(3)*EJL(1)
  C(3)=EJL(1)*EJI(2)-EJI(1)*EJL(2)
  COSDL=COS(OOPNGL_RTN)
  COSTH=COS(THETA)
  CDST=1.0D0/(COSDL*SINTH)
  TANDL=SINDL/COSDL
  TDST=TANDL/SINTH
  TDS2T=TDST/SINTH
  RJL=1.0D0/EJL(4)
  RJI=1.0D0/EJI(4)
  RJK=1.0D0/EJK(4)
  DO NN=1,3
     DL(NN)=(A(NN)*CDST+TANDL*ELJ(NN))*RJL
     DI(NN)=(B(NN)*CDST+TDS2T*(EIJ(NN)+COSTH*EJK(NN)))*RJI
     DK(NN)=(C(NN)*CDST+TDS2T*(EKJ(NN)+COSTH*EJI(NN)))*RJK
     DJ(NN)=-(DI(NN)+DK(NN)+DL(NN))
  ENDDO
  IF(DERIVS.LE.1) RETURN
  !     CALCULATE SECOND DERIVATIVES
  DO N1=1,3
     DO N2=N1,3
        IF(N1.EQ.N2) THEN
           FJI(N1,N2)=0.0D0
           FJK(N1,N2)=0.0D0
           FJL(N1,N2)=0.0D0
        ELSE IF(N1.LT.N2) THEN
           M=6-N1-N2
           ONEPM=(-1.0D0)**(N2-N1)
           FJI(N1,N2)=ONEPM*EJI(M)
           FJI(N2,N1)=-FJI(N1,N2)
           FJK(N1,N2)=ONEPM*EJK(M)
           FJK(N2,N1)=-FJK(N1,N2)
           FJL(N1,N2)=ONEPM*EJL(M)
           FJL(N2,N1)=-FJL(N1,N2)
        ENDIF
     ENDDO
  ENDDO
  COTTH=COSTH/SINTH
  CDST2=CDST*CDST
  TTCS=2.0D0*TDS2T*COTTH
  CDSTJI=CDST*RJI
  CDSTJK=CDST*RJK
  CD2R=1.0D0/(COSDL*COSDL)
  TDRJL=TANDL*RJL
  DO N1=1,3
     EII=EIJ(N1)+COSTH*EJK(N1)
     EKK=EKJ(N1)+COSTH*EJI(N1)
     DO N2=1,3
        DII(N1,N2)=(CDST*B(N1)*(TANDL*DI(N2)-COTTH*TI(N2)) &
             +EII*(CDST2*DI(N2)-TTCS*TI(N2)) &
             +TDS2T*((EIJ(N1)*EIJ(N2)-DKRON(N1,N2))*RJI &
             +SINTH*EKJ(N1)*TI(N2))+EIJ(N2)*DI(N1))*RJI
        DKK(N1,N2)=(CDST*C(N1)*(TANDL*DK(N2)-COTTH*TK(N2)) &
             +EKK*(CDST2*DK(N2)-TTCS*TK(N2)) &
             +TDS2T*((EKJ(N1)*EKJ(N2)-DKRON(N1,N2))*RJK &
             +SINTH*EIJ(N1)*TK(N2))+EKJ(N2)*DK(N1))*RJK
        DLL(N1,N2)=(CDST*A(N1)*TANDL*DL(N2) &
             +CD2R*ELJ(N1)*DL(N2)+ELJ(N2)*DL(N1) &
             +TDRJL*(ELJ(N1)*ELJ(N2)-DKRON(N1,N2)))*RJL
        DIK(N1,N2)=(CDSTJK*(EKJ(N2)*B(N1)+FJL(N2,N1)) &
             +CDST*B(N1)*(TANDL*DK(N2)-COTTH*TK(N2)) &
             +EII*(CDST2*DK(N2)-TTCS*TK(N2))-TDS2T* &
             (COSTH*RJK*(EJK(N1)*EJK(N2)-DKRON(N1,N2)) &
             +SINTH*EJK(N1)*TK(N2)))*RJI
        DIL(N1,N2)=(CDSTJI*(EIJ(N1)*A(N2)+FJK(N1,N2)) &
             +CDST*A(N2)*(TANDL*DI(N1)-COTTH*TI(N1)) &
             +CD2R*ELJ(N2)*DI(N1))*RJL
        DKL(N1,N2)=(CDSTJK*(EKJ(N1)*A(N2)+FJI(N2,N1)) &
             +CDST*A(N2)*(TANDL*DK(N1)-COTTH*TK(N1)) &
             +CD2R*ELJ(N2)*DK(N1))*RJL
     ENDDO
  ENDDO
  DO N1=1,3
     DO N2=1,3
        DJK(N1,N2)=-(DIK(N1,N2)+DKK(N1,N2)+DKL(N2,N1))
        DIJ(N1,N2)=-(DII(N1,N2)+DIK(N1,N2)+DIL(N1,N2))
        DJL(N1,N2)=-(DIL(N1,N2)+DKL(N1,N2)+DLL(N1,N2))
     ENDDO
  ENDDO
  DO N1=1,3
     DO N2=1,3
        DJJ(N1,N2)=-(DIJ(N1,N2)+DJK(N2,N1)+DJL(N2,N1))
     ENDDO
  ENDDO
  !
  RETURN
END FUNCTION OOPNGL

!=======================================================================
! real(chm_real) FUNCTION TORNGL : COMPUTES THE TORSION ANGLE BETWEEN THE PLANE
! FUNCTION COMPUTES THE TORSION ANGLE BETWEEN THE PLANE DEFINED BY
! ATOMS I, J, K AND THE PLANE DEFINED BY ATOMS J, K, AND L.
! ANGLE RETURNED IN RADIANS.
! A POSITIVE TORSION ANGLE DENOTES A CLOCKWISE ROTATION OF PLANE
! J, K, L WITH RESPECT TO THE REFERENCE PLANE I, J, K WHEN VIEWING
! THE ASSEMBLY FROM J TO K.
!=======================================================================
!
FUNCTION TORNGL(DERIVS,I,J,K,L,X,Y,Z) result(torngl_rtn)
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   Adapted by R. Czerminski of Molecular Simulations, Inc. from
  !  code developed at Merck and Co., Inc. by T. Halgren and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !
  !  Jay Banks 27 Nov 95: changed XYZ array to X, Y, Z.
  !
  !  29 Nov 95 Jay Banks: changed arithmetic IFs to IF ... THEN ... ELSE.
  !
  use chm_kinds
  use dimens_fcm
  use mmffm
  implicit none
  !
  integer DERIVS, i, j, k, l
  real(chm_real)  x(*),y(*),z(*),torngl_rtn
  !
  integer n, n1, n1p1, n2, n3, nn
  real(chm_real)  cc, cjr, costhj, costhk, cotthj
  real(chm_real)  cotthk, d1, d2, di, dii, dij, dik, dil, dj, djj
  real(chm_real)  djk, djl, dk, dkk, dkl, dl, dll
  real(chm_real)  rrcj, rrck, rrj, rrk, rrsj, rrsk, rs2j,  &
       rs2k, rsj, rsk
  real(chm_real)  sinthj, sinthk, thj, thk, tkj(3)
  real(chm_real)  tkk(3), tkl(3), t1, t2, t3, tdummy
  real(chm_real)  sij(4),sji(4),sik(4),ski(4),sjk(4), &
       skj(4),sjl(4),slj(4)
  real(chm_real)  skl(4),slk(4)
  real(chm_real)  A(3),B(3),C(3),CIJ(3),CJK(3),CKL(3), &
       TJI(3),TJJ(3),TJK(3)
  real(chm_real)  DI2(3),DL2(3),FIJ(3,3),FJKI(3,3),FJKL(3,3), &
       FKL(3,3)

  COMMON/DTDQ/T1(3),T2(3),T3(3),TDUMMY(3)
  COMMON/DPDQ/DI(3),DJ(3),DK(3),DL(3)
  COMMON/D2PDQ2/DII(3,3),DIJ(3,3),DIK(3,3),DIL(3,3),DJJ(3,3), &
       DJK(3,3),DJL(3,3),DKK(3,3),DKL(3,3),DLL(3,3)
  !
  !   GET AND STORE NEEDED DIRECTION COSINES
  !
  CALL DIRCOS(I,J,K,L,X,Y,Z)
  !   COMPUTE ANGLE
  A(1)=EIJ(2)*EJK(3)-EJK(2)*EIJ(3)
  A(2)=EIJ(3)*EJK(1)-EJK(3)*EIJ(1)
  A(3)=EIJ(1)*EJK(2)-EJK(1)*EIJ(2)
  B(1)=EJK(2)*EKL(3)-EKL(2)*EJK(3)
  B(2)=EJK(3)*EKL(1)-EKL(3)*EJK(1)
  B(3)=EJK(1)*EKL(2)-EKL(1)*EJK(2)
  C(1)=A(2)*B(3)-B(2)*A(3)
  C(2)=A(3)*B(1)-B(3)*A(1)
  C(3)=A(1)*B(2)-B(1)*A(2)
  D1=C(1)*EJK(1)+C(2)*EJK(2)+C(3)*EJK(3)
  D2=A(1)*B(1) + A(2)*B(2) + A(3)*B(3)
  TORNGL_rtn = ATAN2(D1,D2)
  IF(DERIVS.EQ.0) RETURN
  !
  !     CALCULATE FIRST DERIVATIVES; DO NOT NEED TO CALL FOR DIRECTION
  !     COSINES TO BE RECALCUATED, SINCE THEY ARE ALREADY IN PLACE
  !     FOR I,J,K
  !
  THJ=VANGLE(1,I,J,K,0,X,Y,Z)
  !
  !   LOAD THE ANGULAR FIRST DERIVATIVE TERMS RELATED TO CENTER J
  !
  DO N=1,3
     TJI(N)=T1(N)
     TJJ(N)=T2(N)
     TJK(N)=T3(N)
  ENDDO
  COSTHJ=COS(THJ)
  SINTHJ=SIN(THJ)
  !
  !   COPY THE DIRECTION COSINES TO SAVE ARRAYS, AS NEW J-K-L VALUES
  !   WILL BE CALCULATED BY ANGLE AND WILL OVERWRITE THE ORIGINALS
  !
  DO N=1,4
     SIJ(N)=EIJ(N)
     SJI(N)=EJI(N)
     SIK(N)=EIK(N)
     SKI(N)=EKI(N)
     SJK(N)=EJK(N)
     SKJ(N)=EKJ(N)
     SJL(N)=EJL(N)
     SLJ(N)=ELJ(N)
     SKL(N)=EKL(N)
     SLK(N)=EKL(N)
  ENDDO
  !
  !   COMPUTE ANGLE J-K-L, FIRST RECOMPUTING THE NEEDED DIRECTION COSINES
  !
  THK=VANGLE(1,J,K,L,1,X,Y,Z)
  !
  !   LOAD THE ANGULAR DERIVATIVE TERMS RELATED TO CENTER K
  !
  DO N=1,3
     TKJ(N)=T1(N)
     TKK(N)=T2(N)
     TKL(N)=T3(N)
  ENDDO
  !
  !   RESTORE THE DIRECTION COSINES FROM THE SAVE ARRAYS
  !
  DO N=1,4
     EIJ(N)=SIJ(N)
     EJI(N)=SJI(N)
     EIK(N)=SIK(N)
     EKI(N)=SKI(N)
     EJK(N)=SJK(N)
     EKJ(N)=SKJ(N)
     EJL(N)=SJL(N)
     ELJ(N)=SLJ(N)
     EKL(N)=SKL(N)
     ELK(N)=SKL(N)
  ENDDO
  COSTHK=COS(THK)
  SINTHK=SIN(THK)
  RSJ=EIJ(4)*SINTHJ
  RSK=EKL(4)*SINTHK
  RS2J=1.0D0/(RSJ*SINTHJ)
  RS2K=1.0D0/(RSK*SINTHK)
  RRJ=EIJ(4)/EJK(4)
  RRK=EKL(4)/EJK(4)
  RRCJ=RRJ*COSTHJ
  RRCK=RRK*COSTHK
  DO N=1,3
     DI(N)=-A(N)*RS2J
     DL(N)= B(N)*RS2K
     DJ(N)=(RRCJ-1.0D0)*DI(N)-RRCK*DL(N)
     DK(N)=-(DI(N)+DJ(N)+DL(N))
  ENDDO
  IF(DERIVS.LE.1) RETURN
  !     CALCULATE SECOND DERIVATIVES
300 RRSJ=RRJ*SINTHJ
  RRSK=RRK*SINTHK
  DO NN=1,3
     CIJ(NN)=EIJ(NN)/EIJ(4)
     CJK(NN)=EJK(NN)/EJK(4)
     CKL(NN)=EKL(NN)/EKL(4)
  ENDDO
  DO N1=1,2
     N1P1=N1+1
     DO N2=N1P1,3
        N3=6-N1-N2
        CC=(-1.0D0)**(N2-N1)
        FIJ(N1,N2)=CC*EIJ(N3)/EJK(4)
        FJKI(N1,N2)=CC*EJK(N3)/EIJ(4)
        FJKL(N1,N2)=CC*EJK(N3)/EKL(4)
        FKL(N1,N2)=CC*EKL(N3)/EJK(4)
        FIJ(N2,N1)=-FIJ(N1,N2)
        FJKI(N2,N1)=-FJKI(N1,N2)
        FJKL(N2,N1)=-FJKL(N1,N2)
        FKL(N2,N1)=-FKL(N1,N2)
     ENDDO
  ENDDO
  COTTHJ=COSTHJ/SINTHJ
  COTTHK=COSTHK/SINTHK
  CJR=COSTHJ/EJK(4)
  DO NN=1,3
     DI2(NN)=2.0D0*DI(NN)
     DL2(NN)=2.0D0*DL(NN)
     DII(NN,NN)=DI2(NN)*(CIJ(NN)-COTTHJ*TJI(NN))
     DLL(NN,NN)=-DL2(NN)*(CKL(NN)+COTTHK*TKL(NN))
  ENDDO
  DO N1=1,2
     N1P1=N1+1
     DO N2=N1P1,3
        DII(N1,N2)= DI2(N1)*(CIJ(N2)-COTTHJ*TJI(N2))+FJKI(N2,N1)*RS2J
        DLL(N1,N2)=-DL2(N1)*(CKL(N2)+COTTHK*TKL(N2))+FJKL(N1,N2)*RS2K
        DII(N2,N1)=DII(N1,N2)
        DLL(N2,N1)=DLL(N1,N2)
     ENDDO
  ENDDO
  DO N1=1,3
     DO N2=1,3
        DIL(N1,N2)=0.0D0
        DIK(N1,N2)=-(CJK(N2)+2.0D0*COTTHJ*TJK(N2))*DI(N1)
        DJL(N1,N2)=(CJK(N1)-2.0D0*COTTHK*TKJ(N1))*DL(N2)
        IF(N1.NE.N2) THEN
           DIK(N1,N2)=DIK(N1,N2)+FIJ(N2,N1)*RS2J
           DJL(N1,N2)=DJL(N1,N2)+FKL(N2,N1)*RS2K
        ENDIF
        DIJ(N1,N2)=-(DII(N1,N2)+DIK(N1,N2))
        DKL(N1,N2)=-(DLL(N1,N2)+DJL(N1,N2))
     ENDDO
  ENDDO
  DO NN=1,3
     DJJ(NN,NN)=(EIJ(NN)*CJR-TJJ(NN)*RRSJ+CJK(NN)*RRCJ)*DI(NN) + &
          (RRCJ-1.0D0)*DIJ(NN,NN) - RRCK*DJL(NN,NN) + &
          (RRSK*TKJ(NN)-CJK(NN)*RRCK)*DL(NN)
  ENDDO
  DO N1=1,2
     N1P1=N1+1
     DO N2=N1P1,3
        DJJ(N1,N2)=(EIJ(N2)*CJR-TJJ(N2)*RRSJ+CJK(N2)*RRCJ)*DI(N1) + &
             (RRCJ-1.0D0)*DIJ(N1,N2) - RRCK*DJL(N2,N1) + &
             (RRSK*TKJ(N2)-CJK(N2)*RRCK)*DL(N1)
        DJJ(N2,N1)=DJJ(N1,N2)
     ENDDO
  ENDDO
  DO N1=1,3
     DO N2=1,3
        DJK(N1,N2)=-(DIJ(N2,N1)+DJJ(N1,N2)+DJL(N1,N2))
     ENDDO
     DKK(N1,N1)=-(DIK(N1,N1)+DJK(N1,N1)+DKL(N1,N1))
  ENDDO
  DO N1=1,2
     N1P1=N1+1
     DO N2=N1P1,3
        DKK(N1,N2)=-(DIK(N1,N2)+DJK(N1,N2)+DKL(N2,N1))
        DKK(N2,N1)=DKK(N1,N2)
     ENDDO
  ENDDO
  !
  RETURN
END FUNCTION TORNGL

!=======================================================================
! real(chm_real) FUNCTION VANGLE : COMPUTES THE VALENCE ANGLE
! COMPUTES THE VALENCE ANGLE BETWEEN ATOMS I, J, AND K IN RADIANS
!=======================================================================
FUNCTION VANGLE(DERIVS,I,J,K,NDIRC,X,Y,Z) result(vangle_rtn)
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   Adapted by R. Czerminski of Molecular Simulations, Inc. from
  !  code developed at Merck and Co., Inc. by T. Halgren and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !
  !  Jay Banks 27 Nov 95: changed XYZ array to X, Y, Z.
  !
  !  28 Nov 95 Jay Banks: removed space from all "GO TO"s, as suggested in
  !  developer.doc.
  !
  !  29 Nov 95 Jay Banks: changed arithmetic IFs to IF ... THEN ... ELSE.
  !
  use chm_kinds
  use dimens_fcm
  use mmffm
  implicit none
  !
  integer DERIVS, i, j, k, ndirc
  real(chm_real)  x(*),y(*),z(*),vangle_rtn
  !
  integer n, n1, n2
  real(chm_real)  bij, bjk, costh, cotth, di, dii, dij, dik, dil
  real(chm_real)  dj, djj, djk, djl, dk, dkk, dkl, dl, dll
  real(chm_real)  rij2, rijks, rjk2
  real(chm_real)  sinth, tanth, theta
  !
  COMMON/DTDQ/DI(3),DJ(3),DK(3),DL(3)
  COMMON/D2TDQ2/DII(3,3),DIJ(3,3),DIK(3,3),DIL(3,3),DJJ(3,3), &
       DJK(3,3),DJL(3,3),DKK(3,3),DKL(3,3),DLL(3,3)
  real(chm_real) FIJ(3),FJK(3)
  !
  !   SUPPRESS FLOATING-POINT UNDERFLOW MESSAGES
  !
  !      CALL LIB$FLT_UNDER(0)
  !
  !   GET NEEDED DIRECTION COSINES IF NOT ALREADY IN PLACE
  !
  IF(NDIRC.NE.0) CALL DIRCOS(I,J,K,0,X,Y,Z)
  !     CALCULATE ANGLE
  COSTH=0.0D0
  DO N=1,3
     FIJ(N)=EIJ(N)/EIJ(4)
     FJK(N)=EJK(N)/EJK(4)
     COSTH=COSTH+EJI(N)*EJK(N)
  ENDDO
  !   MAKE SURE VALUE IS IN RANGE
  IF(COSTH.LT.-1.) COSTH=-1.
  IF(COSTH.GT.1.) COSTH=1.
  THETA=ACOS(COSTH)
  VANGLE_RTN=THETA
  IF(DERIVS.EQ.0) RETURN
  !     CALCULATE FIRST DERIVATIVES
  SINTH=SIN(THETA)
  IF(SINTH.NE.0) THEN
     BIJ=1.0D0/(EIJ(4)*SINTH)
     BJK=1.0D0/(EJK(4)*SINTH)
     !
     ! NOTE: (Jay Banks, 29 Nov 95)  THIS LOGIC (OR THE "GO TO 500" THAT USED
     ! TO BE HERE) MEANS THAT !!!TANTH IS NOT CALCULATED UNLESS SINTH.EQ.0!!!
     ! THIS IS OK BECAUSE TANTH IS NOT *USED* UNLESS SINTH.EQ.0, EITHER.
     ! ACCORDING TO R. NACHBAR OF MERCK & CO., TANTH MAY BEEN USED HERE
     ! (RATHER THAN JUST ZERO) BECAUSE THERE WAS SOME THOUGHT OF USING IT FOR
     ! SINTH *NEAR* ZERO AS WELL AS EQUAL.  SO LEAVE IT IN FOR NOW, OR MAYBE
     ! SEE IF THERE'S A SIGNIFICANT SPEEDUP FROM LEAVING IT OUT.
     !
  ELSE IF(COSTH.NE.0) THEN
     TANTH=SINTH/COSTH
  ENDIF
  DO N=1,3
     IF(SINTH.EQ.0) THEN
        DI(N)=TANTH*FIJ(N)
        DK(N)=-TANTH*FJK(N)
     ELSE
        DI(N)=(EJI(N)*COSTH+EKJ(N))*BIJ
        DK(N)=(EJK(N)*COSTH+EIJ(N))*BJK
     ENDIF
     DJ(N)=-(DI(N)+DK(N))
  ENDDO
  IF(DERIVS.LE.1) RETURN
  !     CALCULATE SECOND DERIVATIVES
  IF(SINTH.NE.0) THEN
     RIJ2=1.0D0/(EIJ(4)*EIJ(4))
     RJK2=1.0D0/(EJK(4)*EJK(4))
     RIJKS=1.0D0/(EIJ(4)*EJK(4)*SINTH)
     COTTH=COSTH/SINTH
     DO N1=1,3
        DO N2=1,N1
           IF(N1.NE.N2) THEN
              DII(N1,N2)=FIJ(N1)*DI(N2)+FIJ(N2)*DI(N1)- &
                   COTTH*(DI(N1)*DI(N2)+FIJ(N1)*FIJ(N2))
              DII(N2,N1)=DII(N1,N2)
              DKK(N1,N2)=-(FJK(N1)*DK(N2)+FJK(N2)*DK(N1)+ &
                   COTTH*(DK(N1)*DK(N2)+FJK(N1)*FJK(N2)))
              DKK(N2,N1)=DKK(N1,N2)
              DJJ(N1,N2)=DII(N1,N2)+DKK(N1,N2)- &
                   (DI(N1)*DK(N2)+DI(N2)*DK(N1))*COTTH+ &
                   FIJ(N1)*DK(N2)-FJK(N1)*DI(N2)+(EJK(N1)*EJK(N2)+ &
                   EIJ(N1)*EIJ(N2))*RIJKS
              DJJ(N2,N1)=DJJ(N1,N2)
           ELSE
              DII(N1,N2)=FIJ(N1)*DI(N2)+FIJ(N2)*DI(N1)- &
                   COTTH*(DI(N1)*DI(N2)+FIJ(N1)*FIJ(N2)-RIJ2)
              DKK(N1,N2)=-(FJK(N1)*DK(N2)+FJK(N2)*DK(N1)+ &
                   COTTH*(DK(N1)*DK(N2)+FJK(N1)*FJK(N2)-RJK2))
              DJJ(N1,N2)=DII(N1,N2)+DKK(N1,N2)- &
                   (DI(N1)*DK(N2)+DI(N2)*DK(N1))*COTTH+ &
                   FIJ(N1)*DK(N2)-FJK(N1)*DI(N2)+(EJK(N1)*EJK(N2)+ &
                   EIJ(N1)*EIJ(N2)-2.0D0)*RIJKS
           ENDIF
        ENDDO
     ENDDO
     DO N1=1,3
        DO N2=1,3
           IF(N1.NE.N2) THEN
              DIK(N1,N2)=DK(N2)*(FIJ(N1)-COTTH*DI(N1))+ &
                   EJK(N1)*EJK(N2)*RIJKS
           ELSE
              DIK(N1,N2)=DK(N2)*(FIJ(N1)-COTTH*DI(N1))+ &
                   (EJK(N1)*EJK(N2)-1.0D0)*RIJKS
           ENDIF
           DIJ(N1,N2)=-(DII(N1,N2)+DIK(N1,N2))
           DJK(N1,N2)=-(DKK(N1,N2)+DIK(N1,N2))
        ENDDO
     ENDDO
  ELSE
     DO N1=1,3
        DO N2=1,3
           DII(N1,N2)=0.0D0
           DIJ(N1,N2)=0.0D0
           DIK(N1,N2)=0.0D0
           DJJ(N1,N2)=0.0D0
           DJK(N1,N2)=0.0D0
           DKK(N1,N2)=0.0D0
        ENDDO
     ENDDO
  ENDIF
  !
  RETURN
END FUNCTION VANGLE
#endif 

end module escalar_mm

