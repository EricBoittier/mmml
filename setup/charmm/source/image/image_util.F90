module image_util

  contains

  ! to process these atoms
  SUBROUTINE PROCATOMS(IMXCEN,IMYCEN,IMZCEN,NTRANS,IMTRNS,X,Y,Z, &
       LDYNA,XOLD,YOLD,ZOLD,VX,VY,VZ, &
       IMIN,ITRAN,IS,IQ)

    use chm_kinds, only: chm_real
#if KEY_SGLD==1
  use sgld,only:qsgld,qsgmd,sgx1,sgy1,sgz1,sgx2,sgy2,sgz2      
#endif

    implicit none

    ! subroutine arguments
    real(chm_real)  IMXCEN,IMYCEN,IMZCEN
    INTEGER NTRANS
    real(chm_real) IMTRNS(12,*)
    real(chm_real) X(*),Y(*),Z(*)
    real(chm_real) XOLD(*),YOLD(*),ZOLD(*),VX(*),VY(*),VZ(*)
    INTEGER LDYNA,IMIN,ITRAN,IS,IQ

    ! local variables
    INTEGER I
    real(chm_real)  XMAX,XMIN,YMAX,YMIN,ZMAX,ZMIN,XCEN,YCEN,ZCEN
    real(chm_real)  RMIN,XN,YN,ZN
    real(chm_real)  xf, yf, zf

    real(chm_real) r2(ntrans)

    XMAX=X(IS)
    XMIN=X(IS)
    YMAX=Y(IS)
    YMIN=Y(IS)
    ZMAX=Z(IS)
    ZMIN=Z(IS)
    DO I=IS+1,IQ
       IF(X(I).LT.XMIN) XMIN=X(I)
       IF(X(I).GT.XMAX) XMAX=X(I)
       IF(Y(I).LT.YMIN) YMIN=Y(I)
       IF(Y(I).GT.YMAX) YMAX=Y(I)
       IF(Z(I).LT.ZMIN) ZMIN=Z(I)
       IF(Z(I).GT.ZMAX) ZMAX=Z(I)
    ENDDO

    XCEN=(XMAX+XMIN)*0.5
    YCEN=(YMAX+YMIN)*0.5
    ZCEN=(ZMAX+ZMIN)*0.5

    DO ITRAN=1,NTRANS
       XN=XCEN*IMTRNS(1,ITRAN)+YCEN*IMTRNS(2,ITRAN)+ &
            ZCEN*IMTRNS(3,ITRAN)+IMTRNS(10,ITRAN)-IMXCEN
       YN=XCEN*IMTRNS(4,ITRAN)+YCEN*IMTRNS(5,ITRAN)+ &
            ZCEN*IMTRNS(6,ITRAN)+IMTRNS(11,ITRAN)-IMYCEN
       ZN=XCEN*IMTRNS(7,ITRAN)+YCEN*IMTRNS(8,ITRAN)+ &
            ZCEN*IMTRNS(9,ITRAN)+IMTRNS(12,ITRAN)-IMZCEN
       R2(itran) = XN*XN+YN*YN+ZN*ZN
    end do

    rmin = (XCEN-IMXCEN)**2+(YCEN-IMYCEN)**2+(ZCEN-IMZCEN)**2
    IMIN = 0
    do i = 1, ntrans
       IF (R2(i) .LT. RMIN) THEN
          RMIN = R2(i)
          IMIN = i
       END IF
    END DO
    ITRAN = IMIN

    IF (IMIN .le. 0) return

    !       THERE IS A BETTER TRANSFORMATION
    !       NOW TRANSFORM TO IT.
    DO I=IS,IQ
       XN=X(I)*IMTRNS(1,ITRAN)+Y(I)*IMTRNS(2,ITRAN)+ &
            Z(I)*IMTRNS(3,ITRAN)+IMTRNS(10,ITRAN)
       YN=X(I)*IMTRNS(4,ITRAN)+Y(I)*IMTRNS(5,ITRAN)+ &
            Z(I)*IMTRNS(6,ITRAN)+IMTRNS(11,ITRAN)
       ZN=X(I)*IMTRNS(7,ITRAN)+Y(I)*IMTRNS(8,ITRAN)+ &
            Z(I)*IMTRNS(9,ITRAN)+IMTRNS(12,ITRAN)

       X(I)=XN
       Y(I)=YN
       Z(I)=ZN
    ENDDO

    !       For dynamics, must also modify previous coordinates
    !       In case it goes to a two step verlet, also rotate the velocities
    IF(LDYNA.GE.1) THEN
       DO I=IS,IQ
          XN=XOLD(I)*IMTRNS(1,ITRAN)+YOLD(I)*IMTRNS(2,ITRAN)+ &
               ZOLD(I)*IMTRNS(3,ITRAN)
          YN=XOLD(I)*IMTRNS(4,ITRAN)+YOLD(I)*IMTRNS(5,ITRAN)+ &
               ZOLD(I)*IMTRNS(6,ITRAN)
          ZN=XOLD(I)*IMTRNS(7,ITRAN)+YOLD(I)*IMTRNS(8,ITRAN)+ &
               ZOLD(I)*IMTRNS(9,ITRAN)

          XOLD(I)=XN
          YOLD(I)=YN
          ZOLD(I)=ZN
       ENDDO
    ENDIF

    IF(LDYNA.LE.-1) THEN
       DO I=IS,IQ
          XN=XOLD(I)*IMTRNS(1,ITRAN)+YOLD(I)*IMTRNS(2,ITRAN)+ &
               ZOLD(I)*IMTRNS(3,ITRAN)+IMTRNS(10,ITRAN)
          YN=XOLD(I)*IMTRNS(4,ITRAN)+YOLD(I)*IMTRNS(5,ITRAN)+ &
               ZOLD(I)*IMTRNS(6,ITRAN)+IMTRNS(11,ITRAN)
          ZN=XOLD(I)*IMTRNS(7,ITRAN)+YOLD(I)*IMTRNS(8,ITRAN)+ &
               ZOLD(I)*IMTRNS(9,ITRAN)+IMTRNS(12,ITRAN)

          XOLD(I)=XN
          YOLD(I)=YN
          ZOLD(I)=ZN
       ENDDO
    ENDIF

    IF(ABS(LDYNA).EQ.1) THEN
       DO I=IS,IQ
          XN=VX(I)*IMTRNS(1,ITRAN)+VY(I)*IMTRNS(2,ITRAN)+ &
               VZ(I)*IMTRNS(3,ITRAN)
          YN=VX(I)*IMTRNS(4,ITRAN)+VY(I)*IMTRNS(5,ITRAN)+ &
               VZ(I)*IMTRNS(6,ITRAN)
          ZN=VX(I)*IMTRNS(7,ITRAN)+VY(I)*IMTRNS(8,ITRAN)+ &
               VZ(I)*IMTRNS(9,ITRAN)

          VX(I)=XN
          VY(I)=YN
          VZ(I)=ZN
       ENDDO
    ENDIF
#if KEY_SGLD==1
     IF(QSGLD.OR.QSGMD) THEN
       DO I=IS,IQ
         XN=SGX1(I)*IMTRNS(1,ITRAN)+SGY1(I)*IMTRNS(2,ITRAN)+ &
             SGZ1(I)*IMTRNS(3,ITRAN)+IMTRNS(10,ITRAN)
         YN=SGX1(I)*IMTRNS(4,ITRAN)+SGY1(I)*IMTRNS(5,ITRAN)+ &
             SGZ1(I)*IMTRNS(6,ITRAN)+IMTRNS(11,ITRAN)
         ZN=SGX1(I)*IMTRNS(7,ITRAN)+SGY1(I)*IMTRNS(8,ITRAN)+ &
             SGZ1(I)*IMTRNS(9,ITRAN)+IMTRNS(12,ITRAN)
         SGX1(I)=XN
         SGY1(I)=YN
         SGZ1(I)=ZN
         XN=SGX2(I)*IMTRNS(1,ITRAN)+SGY2(I)*IMTRNS(2,ITRAN)+ &
             SGZ2(I)*IMTRNS(3,ITRAN)+IMTRNS(10,ITRAN)
         YN=SGX2(I)*IMTRNS(4,ITRAN)+SGY2(I)*IMTRNS(5,ITRAN)+ &
             SGZ2(I)*IMTRNS(6,ITRAN)+IMTRNS(11,ITRAN)
         ZN=SGX2(I)*IMTRNS(7,ITRAN)+SGY2(I)*IMTRNS(8,ITRAN)+ &
             SGZ2(I)*IMTRNS(9,ITRAN)+IMTRNS(12,ITRAN)
         SGX2(I)=XN
         SGY2(I)=YN
         SGZ2(I)=ZN
       ENDDO
     ENDIF
#endif 

    RETURN
  end SUBROUTINE PROCATOMS

end module image_util
