#if KEY_FOURD==0 /*4dinert*/
SUBROUTINE NULL_INRMIN
  return
end SUBROUTINE NULL_INRMIN
#else /* (4dinert)*/
#if KEY_UNUSED==1 /*inrmin_unused*/
SUBROUTINE INRMIN(VARB,NVAR)
  !-----------------------------------------------------------------------
  !     This routine will use the inertia tensor to determine
  !     the best rotation for the back projection.     
  !
  use dimens_fcm
  use number
  use stream
  !
  real(chm_real) VARB(*)
  INTEGER NVAR
  !
  !
  LOGICAL LPRNT
  real(chm_real) SUMR2,SUM, X(MAXAIM), Y(MAXAIM), Z(MAXAIM),  &
       FDIM(MAXAIM)
  real(chm_real) DET, RN(4), PHI
  INTEGER IQ, IP, P, I, NATOM, IPT, J 
  real(chm_real) RXRX, RXRY, RXRZ, RXRFD, RYRX, RYRY, RYRZ, RYRFD
  real(chm_real) RZRX, RZRY, RZRZ, RZRFD, RFDRX, RFDRY, RFDRZ,  &
       RFDRFD
  real(chm_real) INERTA(4,4),V(4,4),TEMP(16),U(16),AMOM(10)
  real(chm_real) XP,YP,ZP,FDP,SCR(32),XC,YC,ZC,FDC
  !     -------------------------------------------------------------- 
  !     VARIABLE DEFINITIONS:
  !     NM = DIMENSION OF MATRIX=4
  !     N = ORDER OF MATRIX=4
  !
  !     I(x,x)=SUM[Y**2 + Z**2 + FDIM**2]
  !     I(y,y)=SUM[X**2 + Z**2 + FDIM**2]
  !     I(z,z)=SUM[X**2 + Y**2 + FDIM**2]
  !     I(fdim,fdim)=SUM[X**2 + Y**2 + Z**2]
  !     I(x,y)=I(y,x)=-SUM[X*Y]
  !     I(x,z)=I(z,x)=-SUM[X*Z]
  !     I(x,fdim)=I(fdim,x)=-SUM[X*FDIM]
  !     I(y,z)=I(z,y)=-SUM[Y*Z]
  !     I(y,fdim)=I(fdim,y)=-SUM[Y*FDIM]
  !     I(z,fdim)=I(fdim,z)=-SUM[Z*FDIM]
  !
  !     RXRX,RXRY,etc all components of the inertia tensor. 
  !
  !     ---------------------------------------------------------------
  ! 
  !     First need to put Varb into seperate Uectors:  X,Y,Z,FDIM
  !     Note: this does the same as SUBROUTINE PUTVR0.
  !    
  !     Written by Elan Eisenmesser
  if(prnlev.ge.2) WRITE(outu,*)'ROTATION APPLIED TO PRINCIPAL AXES'
  P = 0
  NATOM = NVAR/4
  DO I = 1,NATOM
     X(I) = VARB(P + 1)
     Y(I) = VARB(P + 2)
     Z(I) = VARB(P + 3)
     FDIM(I) = VARB(P + 4)
     P = P + 4
  ENDDO
  !
  !
  !     Center coordinates.
  !
  XC=ZERO
  YC=ZERO
  ZC=ZERO
  FDC=ZERO
  DO I = 1,NATOM
     XC=XC + X(I)
     YC=YC + Y(I)
     ZC=ZC + Z(I)
     FDC=FDC + FDIM(I)
  ENDDO
  XC=XC/NATOM
  YC=YC/NATOM
  ZC=ZC/NATOM 
  FDC=FDC/NATOM
  DO I = 1,NATOM
     X(I)=X(I) - XC
     Y(I)=Y(I) - YC
     Z(I)=Z(I) - ZC
     FDIM(I)=FDIM(I) - FDC
  ENDDO
  !
  !     NOW BEGIN TO MAKE INERTIA TENSOR
  !
  RXRX = ZERO
  RXRY = ZERO
  RXRZ = ZERO
  RXRFD = ZERO
  RYRX = ZERO
  RYRY = ZERO
  RYRZ = ZERO
  RYRFD = ZERO
  RZRX = ZERO
  RZRY = ZERO
  RZRZ = ZERO
  RZRFD = ZERO
  RFDRX = ZERO
  RFDRY = ZERO
  RFDRZ = ZERO
  RFDRFD = ZERO

  DO I=1,NATOM
     RXRX = RXRX + Y(I)*Y(I)+Z(I)*Z(I)+FDIM(I)*FDIM(I) 
     RXRY = RXRY - X(I)*Y(I)
     RXRZ = RXRZ - X(I)*Z(I)
     RXRFD = RXRFD - X(I)*FDIM(I)
     RYRX = RYRX - Y(I)*X(I)
     RYRY = RYRY + X(I)*X(I)+Z(I)*Z(I)+FDIM(I)+FDIM(I)
     RYRZ = RYRZ - Y(I)*Z(I)
     RYRFD = RYRFD - Y(I)*FDIM(I)
     RZRX = RZRX - Z(I)*X(I)
     RZRY = RZRY - Z(I)*Y(I)
     RZRZ = RZRZ + X(I)*X(I)+Y(I)*Y(I)+FDIM(I)*FDIM(I)
     RZRFD = RZRFD - Z(I)*FDIM(I)
     RFDRX = RFDRX - FDIM(I)*X(I)
     RFDRY = RFDRY - FDIM(I)*Y(I)
     RFDRZ = RFDRZ - FDIM(I)*Z(I)
     RFDRFD = RFDRFD + X(I)*X(I)+Y(I)*Y(I)+Z(I)*Z(I)
  ENDDO
  !
  !     NOW ALL THE COMPONENTS OF THE INERTIA TENSOR IN 3D and.
  !
  INERTA(1,1) =  RXRX
  INERTA(1,2) =  RXRY
  INERTA(1,3) =  RXRZ
  INERTA(2,1) =  RYRX
  INERTA(2,2) =  RYRY
  INERTA(2,3) =  RYRZ
  INERTA(3,1) =  RZRX
  INERTA(3,2) =  RZRY
  INERTA(3,3) =  RZRZ
  INERTA(1,4) =  RXRFD
  INERTA(2,4) =  RYRFD
  INERTA(3,4) =  RZRFD
  INERTA(4,1) =  RFDRX
  INERTA(4,2) =  RFDRY
  INERTA(4,3) =  RFDRZ
  INERTA(4,4) =  RFDRFD
  !
  AMOM(1)=INERTA(4,4)
  AMOM(2)=INERTA(3,4)
  AMOM(3)=INERTA(2,4)
  AMOM(4)=INERTA(1,4)
  AMOM(5)=INERTA(3,3)
  AMOM(6)=INERTA(2,3)
  AMOM(7)=INERTA(1,3)
  AMOM(8)=INERTA(2,2)
  AMOM(9)=INERTA(1,2)
  AMOM(10)=INERTA(1,1)
  !
  !     The actual rotation matrix in 4-D:
  !
  CALL DIAGQ(4,4,AMOM,U,SCR(5),SCR(9),SCR(13), &
       SCR(17),SCR(1),SCR(21),SCR(25), &
       SCR(29),0)
  DO I=1,4
     IPT=(I-1)*4
     DET=U(IPT+1)
     U(IPT+1)=U(IPT+4)
     U(IPT+4)=DET
     IF(U(IPT+I).LT.ZERO) THEN
        DO J=1,4
           IPT=IPT+1
           U(IPT)=-U(IPT)
        ENDDO
     ENDIF
  ENDDO
  DO I=1,4
     IPT=(I-1)*4
     DET=U(IPT+2)
     U(IPT+2)=U(IPT+3)
     U(IPT+3)=DET
     IF(U(IPT+I).LT.ZERO) THEN
        DO J=1,4
           IPT=IPT+1
           U(IPT)=-U(IPT)
        ENDDO
     ENDIF
  ENDDO
  DET=U(1)*(U(6)*(U(11)*U(16)-U(12)*U(15))-U(7)*(U(10)*U(16)- &
       U(12)*U(14))+U(8)*(U(10)*U(15)-U(11)*U(14))) - &
       U(2)*(U(5)*(U(11)*U(16)-U(12)*U(15))-U(7)*(U(9)*U(16)- &
       U(12)*U(13))+U(8)*(U(9)*U(15)-U(11)*U(13))) + &
       U(3)*(U(5)*(U(10)*U(16)-U(12)*U(14))-U(6)*(U(9)*U(16)- &
       U(12)*U(13))+U(8)*(U(9)*U(14)-U(10)*U(13))) - &
       U(4)*(U(5)*(U(10)*U(15)-U(11)*U(14))-U(6)*(U(9)*U(15)- &
       U(11)*U(13))+U(7)*(U(9)*U(14)-U(10)*U(13))) 
  IF(DET.LT.ZERO) THEN
     U(13)=-U(13)
     U(14)=-U(14)
     U(15)=-U(15)
     U(16)=-U(16)
     DET=-DET
  ENDIF
  !
  !      Change to principal axes for 4-d.
  !
  DO I=1,NATOM
     XP=U(1)*X(I)+U(2)*Y(I)+U(3)*Z(I)+U(4)*FDIM(I) 
     YP=U(5)*X(I)+U(6)*Y(I)+U(7)*Z(I)+U(8)*FDIM(I)
     ZP=U(9)*X(I)+U(10)*Y(I)+U(11)*Z(I)+U(12)*FDIM(I)
     FDP=U(13)*X(I)+U(14)*Y(I)+U(15)*Z(I)+U(16)*FDIM(I)
     X(I)=XP
     Y(I)=YP
     Z(I)=ZP
     FDIM(I)=FDP
  ENDDO
  !
  !     Now put X,Y,Z,FDIM back into Varb.
  !     Note: this is the same as SUBROUTINE GETVR0.
  !
  P = 0
  DO I = 1,NATOM
     VARB(P + 1) = X(I)
     VARB(P + 2) = Y(I)
     VARB(P + 3) = Z(I)
     VARB(P + 4) = FDIM(I)
     P = P + 4
  ENDDO
  RETURN
END SUBROUTINE INRMIN

#endif /* (inrmin_unused)*/
SUBROUTINE INRDYN(X,Y,Z,FDIM,VX,VY,VZ,VFD,NATOM)
  !-----------------------------------------------------------------------
  !     This routine will use the inertia tensor to determine
  !     the best rotation for a back projection.
  !     INRDYN is for a back projection during dynamics.     
  !
  use chm_kinds
  use dimens_fcm
  use number
  use stream
  implicit none
  !
  LOGICAL LPRNT
  real(chm_real) SUMR2,SUM, X(*), Y(*), Z(*), FDIM(*)
  real(chm_real) VX(*),VY(*),VZ(*),VFD(*)
  real(chm_real)  DET, RN(4), PHI
  INTEGER IQ, IP, P, I, NATOM, IPT, J 
  real(chm_real) RXRX, RXRY, RXRZ, RXRFD, RYRX, RYRY, RYRZ, RYRFD
  real(chm_real) RZRX, RZRY, RZRZ, RZRFD, RFDRX, RFDRY, RFDRZ,  &
       RFDRFD
  real(chm_real) INERTA(4,4),V(4,4),TEMP(16),U(16),AMOM(10)
  real(chm_real) XP,YP,ZP,FDP,SCR(32),XC,YC,ZC,FDC
  real(chm_real) VXP,VYP,VZP,VFDP
  !     -------------------------------------------------------------- 
  !     VARIABLE DEFINITIONS:
  !     NM = DIMENSION OF MATRIX=4
  !     N = ORDER OF MATRIX=4
  !
  !     I(x,x)=SUM[Y**2 + Z**2 + FDIM**2]
  !     I(y,y)=SUM[X**2 + Z**2 + FDIM**2]
  !     I(z,z)=SUM[X**2 + Y**2 + FDIM**2]
  !     I(fdim,fdim)=SUM[X**2 + Y**2 + Z**2]
  !     I(x,y)=I(y,x)=-SUM[X*Y]
  !     I(x,z)=I(z,x)=-SUM[X*Z]
  !     I(x,fdim)=I(fdim,x)=-SUM[X*FDIM]
  !     I(y,z)=I(z,y)=-SUM[Y*Z]
  !     I(y,fdim)=I(fdim,y)=-SUM[Y*FDIM]
  !     I(z,fdim)=I(fdim,z)=-SUM[Z*FDIM]
  !
  !     RXRX,RXRY,etc all components of the inertia tensor. 
  !
  !     ---------------------------------------------------------------
  ! 
  !
  !     Center coordinates.
  !
  if(prnlev.ge.2) WRITE(outu,*)'ROTATION APPLIED TO PRINCIPAL AXES'

  XC=ZERO
  YC=ZERO
  ZC=ZERO
  FDC=ZERO
  DO I = 1,NATOM
     XC=XC + X(I)
     YC=YC + Y(I)
     ZC=ZC + Z(I)
     FDC=FDC + FDIM(I)
  ENDDO
  XC=XC/NATOM
  YC=YC/NATOM
  ZC=ZC/NATOM 
  FDC=FDC/NATOM
  DO I = 1,NATOM
     X(I)=X(I) - XC
     Y(I)=Y(I) - YC
     Z(I)=Z(I) - ZC
     FDIM(I)=FDIM(I) - FDC
  ENDDO
  !
  !     NOW BEGIN TO MAKE INERTIA TENSOR
  !
  RXRX = ZERO
  RXRY = ZERO
  RXRZ = ZERO
  RXRFD = ZERO
  RYRX = ZERO
  RYRY = ZERO
  RYRZ = ZERO
  RYRFD = ZERO
  RZRX = ZERO
  RZRY = ZERO
  RZRZ = ZERO
  RZRFD = ZERO
  RFDRX = ZERO
  RFDRY = ZERO
  RFDRZ = ZERO
  RFDRFD = ZERO

  DO I=1,NATOM
     RXRX = RXRX + Y(I)*Y(I)+Z(I)*Z(I)+FDIM(I)*FDIM(I) 
     RXRY = RXRY - X(I)*Y(I)
     RXRZ = RXRZ - X(I)*Z(I)
     RXRFD = RXRFD - X(I)*FDIM(I)
     RYRX = RYRX - Y(I)*X(I)
     RYRY = RYRY + X(I)*X(I)+Z(I)*Z(I)+FDIM(I)+FDIM(I)
     RYRZ = RYRZ - Y(I)*Z(I)
     RYRFD = RYRFD - Y(I)*FDIM(I)
     RZRX = RZRX - Z(I)*X(I)
     RZRY = RZRY - Z(I)*Y(I)
     RZRZ = RZRZ + X(I)*X(I)+Y(I)*Y(I)+FDIM(I)*FDIM(I)
     RZRFD = RZRFD - Z(I)*FDIM(I)
     RFDRX = RFDRX - FDIM(I)*X(I)
     RFDRY = RFDRY - FDIM(I)*Y(I)
     RFDRZ = RFDRZ - FDIM(I)*Z(I)
     RFDRFD = RFDRFD + X(I)*X(I)+Y(I)*Y(I)+Z(I)*Z(I)
  ENDDO
  !
  !     NOW ALL THE COMPONENTS OF THE INERTIA TENSOR IN 3D and.
  !
  INERTA(1,1) =  RXRX
  INERTA(1,2) =  RXRY
  INERTA(1,3) =  RXRZ
  INERTA(2,1) =  RYRX
  INERTA(2,2) =  RYRY
  INERTA(2,3) =  RYRZ
  INERTA(3,1) =  RZRX
  INERTA(3,2) =  RZRY
  INERTA(3,3) =  RZRZ
  INERTA(1,4) =  RXRFD
  INERTA(2,4) =  RYRFD
  INERTA(3,4) =  RZRFD
  INERTA(4,1) =  RFDRX
  INERTA(4,2) =  RFDRY
  INERTA(4,3) =  RFDRZ
  INERTA(4,4) =  RFDRFD
  !
  AMOM(1)=INERTA(4,4)
  AMOM(2)=INERTA(3,4)
  AMOM(3)=INERTA(2,4)
  AMOM(4)=INERTA(1,4)
  AMOM(5)=INERTA(3,3)
  AMOM(6)=INERTA(2,3)
  AMOM(7)=INERTA(1,3)
  AMOM(8)=INERTA(2,2)
  AMOM(9)=INERTA(1,2)
  AMOM(10)=INERTA(1,1)
  !
  !     The actual rotation matrix in 4-D:
  !
  CALL DIAGQ(4,4,AMOM,U,SCR(5),SCR(9),SCR(13), &
       SCR(17),SCR(1),SCR(21),SCR(25), &
       SCR(29),0)
  DO I=1,4
     IPT=(I-1)*4
     DET=U(IPT+1)
     U(IPT+1)=U(IPT+4)
     U(IPT+4)=DET
     IF(U(IPT+I).LT.ZERO) THEN
        DO J=1,4
           IPT=IPT+1
           U(IPT)=-U(IPT)
        ENDDO
     ENDIF
  ENDDO
  DO I=1,4
     IPT=(I-1)*4
     DET=U(IPT+2)
     U(IPT+2)=U(IPT+3)
     U(IPT+3)=DET
     IF(U(IPT+I).LT.ZERO) THEN
        DO J=1,4
           IPT=IPT+1
           U(IPT)=-U(IPT)
        ENDDO
     ENDIF
  ENDDO
  DET=U(1)*(U(6)*(U(11)*U(16)-U(12)*U(15))-U(7)*(U(10)*U(16)- &
       U(12)*U(14))+U(8)*(U(10)*U(15)-U(11)*U(14))) - &
       U(2)*(U(5)*(U(11)*U(16)-U(12)*U(15))-U(7)*(U(9)*U(16)- &
       U(12)*U(13))+U(8)*(U(9)*U(15)-U(11)*U(13))) + &
       U(3)*(U(5)*(U(10)*U(16)-U(12)*U(14))-U(6)*(U(9)*U(16)- &
       U(12)*U(13))+U(8)*(U(9)*U(14)-U(10)*U(13))) - &
       U(4)*(U(5)*(U(10)*U(15)-U(11)*U(14))-U(6)*(U(9)*U(15)- &
       U(11)*U(13))+U(7)*(U(9)*U(14)-U(10)*U(13))) 
  IF(DET.LT.ZERO) THEN
     U(13)=-U(13)
     U(14)=-U(14)
     U(15)=-U(15)
     U(16)=-U(16)
     DET=-DET
  ENDIF
  !
  !      Change to principal axes for 4-d.
  !
  DO I=1,NATOM
     XP=U(1)*X(I)+U(2)*Y(I)+U(3)*Z(I)+U(4)*FDIM(I) 
     YP=U(5)*X(I)+U(6)*Y(I)+U(7)*Z(I)+U(8)*FDIM(I)
     ZP=U(9)*X(I)+U(10)*Y(I)+U(11)*Z(I)+U(12)*FDIM(I)
     FDP=U(13)*X(I)+U(14)*Y(I)+U(15)*Z(I)+U(16)*FDIM(I)
     X(I)=XP
     Y(I)=YP
     Z(I)=ZP
     FDIM(I)=FDP
  ENDDO
  !
  ! Velocities also rotated:
  !
  DO I=1,NATOM
     VXP=U(1)*VX(I)+U(2)*VY(I)+U(3)*VZ(I)+U(4)*VFD(I) 
     VYP=U(5)*VX(I)+U(6)*VY(I)+U(7)*VZ(I)+U(8)*VFD(I)
     VZP=U(9)*VX(I)+U(10)*VY(I)+U(11)*VZ(I)+U(12)*VFD(I)
     VFDP=U(13)*VX(I)+U(14)*VY(I)+U(15)*VZ(I)+U(16)*VFD(I)
     VX(I)=VXP
     VY(I)=VYP
     VZ(I)=VZP
     VFD(I)=VFDP
  ENDDO
  !
  RETURN
END SUBROUTINE INRDYN
#endif /* (4dinert)*/

