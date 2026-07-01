module quick
  use chm_kinds
  use dimens_fcm
  implicit none

contains

#if KEY_NOMISC==0 /*nomisc*/
  SUBROUTINE QUICKA
    !
    ! Quick and easy analysis facility for atoms
    !
    use chm_kinds
    use dimens_fcm
    use exfunc
    use number
    !
    use comand
    use consta
    use psf
    use corsubs, only: &
      axisx,axisy,axisz,axisr, &
      axiscx,axiscy,axiscz,qaxisc
    use stream
    use string
    use memory
    use pucker_mod, only: pucker6cp
    use param_store, only: set_param

    implicit none
    !
    ! . Local variables.
    integer,allocatable,dimension(:) :: ISLCT
    INTEGER II
    real(chm_real) XIAT,YIAT,ZIAT,XJAT,YJAT,ZJAT, &
         XKAT,YKAT,ZKAT,XLAT,YLAT,ZLAT,XMAT,YMAT,ZMAT,XNAT,YNAT,ZNAT
    real(chm_real) R2,TH,PHI,THETA,Q,RA,RB,RC
    real(chm_real) AX,AY,AZ,BX,BY,BZ,CX,CY,CZ,FX,FY,FZ,GX,GY,GZ, &
         HX,HY,HZ
    real(chm_real) FR,GR,HR,CST
    LOGICAL BAD,ALLINT
    CHARACTER(len=8) WRD,ALP
    INTEGER WRDLEN
    CHARACTER(len=40) OLDLYN
    INTEGER OLDLEN
    !
    ! parse the command line
    CALL TRIMA(COMLYN,COMLEN)
    IF(COMLEN == 0) THEN
       ! Process general information command
       IF(PRNLEV >= 2) WRITE(OUTU,'(A)')' QUICKA: No atoms requested.'
       RETURN
    ENDIF
    !
    ! Check to see if there are only integers on the line (old syntax)
    ! If all words are integers, the ALLINT will be .true.
    ALLINT=.FALSE.
    IF(COMLEN > 40) GOTO 20
    CALL COPYST(OLDLYN,40,OLDLEN,COMLYN,COMLEN)
10  CONTINUE
    WRD=NEXTA8(OLDLYN,OLDLEN)
    WRDLEN=8
    CALL SPLITI(WRD,II,ALP,WRDLEN)
    IF(WRDLEN > 0) GOTO 20
    CALL TRIMA(OLDLYN,OLDLEN)
    IF(OLDLEN > 0) GOTO 10
    ALLINT=.TRUE.
20  CONTINUE
    !
    call chmalloc('quicka.src','QUICKA','ISLCT',NATOM,intg=ISLCT)
    BAD=.FALSE.
    !-----------------------------------------------------------------------
    CALL GETNEXTP(COMLYN,COMLEN,ISLCT,XIAT,YIAT,ZIAT,BAD,ALLINT)
    IF(BAD) THEN
       call chmdealloc('quicka.src','QUICKA','ISLCT',NATOM,intg=ISLCT)
       RETURN
    ENDIF
    IF(COMLEN == 0) THEN
       ! Process atom command
       !
       IF(QAXISC) THEN
          !          compute projection of position onto corman defined axis.
          AX=XIAT-AXISCX
          AY=YIAT-AXISCY
          AZ=ZIAT-AXISCZ
          RA=SQRT(AX*AX+AY*AY+AZ*AZ)
          CST=(AX*AXISX+AY*AXISY+AZ*AXISZ)/AXISR**2
          CX=AXISX*CST
          CY=AXISY*CST
          CZ=AXISZ*CST
          RC=SQRT(CX*CX+CY*CY+CZ*CZ)
          BX=AX-CX
          BY=AY-CY
          BZ=AZ-CZ
          RB=SQRT(BX*BX+BY*BY+BZ*BZ)
          IF(PRNLEV > 3) THEN
             WRITE(OUTU,44) RA
44           FORMAT(' QUICKA: The distance from axis center is:',F12.5)
             WRITE(OUTU,45) RB
45           FORMAT(' QUICKA: The distance from axis vector is:',F12.5)
             WRITE(OUTU,46) RC
46           FORMAT(' QUICKA: Projected distance along axis is:',F12.5)
          ENDIF
       ENDIF
       !
       call set_param('XVAL',XIAT)
       call set_param('YVAL',YIAT)
       call set_param('ZVAL',ZIAT)
       IF(PRNLEV >= 2) WRITE(OUTU,105)
       call chmdealloc('quicka.src','QUICKA','ISLCT',NATOM,intg=ISLCT)
       RETURN
    ENDIF
    !-----------------------------------------------------------------------
    CALL GETNEXTP(COMLYN,COMLEN,ISLCT,XJAT,YJAT,ZJAT,BAD,ALLINT)
    IF(BAD) THEN
       call chmdealloc('quicka.src','QUICKA','ISLCT',NATOM,intg=ISLCT)
       RETURN
    ENDIF
    IF(COMLEN == 0) THEN
       ! Process distance command
       !
       R2=(XIAT-XJAT)**2+(YIAT-YJAT)**2+(ZIAT-ZJAT)**2
       R2=SQRT(R2)
       IF(PRNLEV >= 2) WRITE(OUTU,55) R2
55     FORMAT(' QUICKA: The distance is:',F12.5)
       !
       call set_param('DIST',R2)
       IF(PRNLEV >= 2) WRITE(OUTU,105)
       call chmdealloc('quicka.src','QUICKA','ISLCT',NATOM,intg=ISLCT)
       RETURN
    ENDIF
    !
    !-----------------------------------------------------------------------
    CALL GETNEXTP(COMLYN,COMLEN,islct, &
         XKAT,YKAT,ZKAT,BAD,ALLINT)
    IF(BAD) THEN
       call chmdealloc('quicka.src','QUICKA','ISLCT',NATOM,intg=ISLCT)
       RETURN
    ENDIF
    IF(COMLEN == 0) THEN
       ! Process angle command
       !
       FX=XIAT-XJAT
       FY=YIAT-YJAT
       FZ=ZIAT-ZJAT
       FR=SQRT(FX*FX + FY*FY + FZ*FZ)
       IF(FR < 0.000001) THEN
          WRITE(OUTU,'(A)') ' QUICKA: No valid angle.'
          WRITE(OUTU,105)
          call chmdealloc('quicka.src','QUICKA','ISLCT',NATOM,intg=ISLCT)
          RETURN
       ENDIF
       FX=FX/FR
       FY=FY/FR
       FZ=FZ/FR
       !
       GX=XKAT-XJAT
       GY=YKAT-YJAT
       GZ=ZKAT-ZJAT
       GR=SQRT(GX*GX+GY*GY+GZ*GZ)
       IF(GR < 0.000001) THEN
          IF(WRNLEV >= 2) WRITE(OUTU,'(A)') ' QUICKA: No valid angle.'
          IF(WRNLEV >= 2) WRITE(OUTU,105)
          call chmdealloc('quicka.src','QUICKA','ISLCT',NATOM,intg=ISLCT)
          RETURN
       ENDIF
       GX=GX/GR
       GY=GY/GR
       GZ=GZ/GR
       CST=FX*GX+FY*GY+FZ*GZ
       !
       IF(ABS(CST) >= COSMAX) THEN
          TH=NINETY-SIGN(NINETY,CST)
       ELSE
          TH=ACOS(CST)*RADDEG
       ENDIF
       IF(PRNLEV >= 2) WRITE(OUTU,65) TH
65     FORMAT(' QUICKA: The angle is:',F10.3)
       !
       call set_param('THET',TH)
       IF(PRNLEV >= 2) WRITE(OUTU,105)
       call chmdealloc('quicka.src','QUICKA','ISLCT',NATOM,intg=ISLCT)
       RETURN
    ENDIF
    !
    !-----------------------------------------------------------------------
    CALL GETNEXTP(COMLYN,COMLEN,islct, &
         XLAT,YLAT,ZLAT,BAD,ALLINT)
    IF(BAD) THEN
       call chmdealloc('quicka.src','QUICKA','ISLCT',NATOM,intg=ISLCT)
       RETURN
    ENDIF
    IF(COMLEN == 0) THEN
       ! Process dihedral command
       !
       FX=XIAT-XJAT
       FY=YIAT-YJAT
       FZ=ZIAT-ZJAT
       FR=FX*FX + FY*FY + FZ*FZ
       FR=SQRT(FR)
       IF(FR < 0.000001) THEN
          IF(WRNLEV >= 2) WRITE(OUTU,'(A)') &
               ' QUICKA: No valid dihedral.'
          IF(WRNLEV >= 2) WRITE(OUTU,105)
          call chmdealloc('quicka.src','QUICKA','ISLCT',NATOM,intg=ISLCT)
          RETURN
       ENDIF
       FX=FX/FR
       FY=FY/FR
       FZ=FZ/FR
       !
       HX=XLAT-XKAT
       HY=YLAT-YKAT
       HZ=ZLAT-ZKAT
       HR=HX*HX + HY*HY + HZ*HZ
       HR=SQRT(HR)
       IF(HR < 0.000001) THEN
          IF(WRNLEV >= 2) WRITE(OUTU,'(A)') &
               ' QUICKA: No valid dihedral.'
          IF(WRNLEV >= 2) WRITE(OUTU,105)
          call chmdealloc('quicka.src','QUICKA','ISLCT',NATOM,intg=ISLCT)
          RETURN
       ENDIF
       HX=HX/HR
       HY=HY/HR
       HZ=HZ/HR
       !
       GX=XJAT-XKAT
       GY=YJAT-YKAT
       GZ=ZJAT-ZKAT
       GR=GX*GX+GY*GY+GZ*GZ
       GR=SQRT(GR)
       IF(GR < 0.000001) THEN
          IF(WRNLEV >= 2) WRITE(OUTU,'(A)') &
               ' QUICKA: No valid dihedral.'
          IF(WRNLEV >= 2) WRITE(OUTU,105)
          call chmdealloc('quicka.src','QUICKA','ISLCT',NATOM,intg=ISLCT)
          RETURN
       ENDIF
       GX=GX/GR
       GY=GY/GR
       GZ=GZ/GR
       !
       CST=-FX*GX-FY*GY-FZ*GZ
       IF(ABS(CST) >= COSMAX) THEN
          IF(WRNLEV >= 2) WRITE(OUTU,'(A)') &
               ' QUICKA: No valid dihedral.'
          IF(WRNLEV >= 2) WRITE(OUTU,105)
          call chmdealloc('quicka.src','QUICKA','ISLCT',NATOM,intg=ISLCT)
          RETURN
       ENDIF
       CST=HX*GX+HY*GY+HZ*GZ
       IF(ABS(CST) >= COSMAX) THEN
          IF(WRNLEV >= 2) WRITE(OUTU,'(A)') &
               ' QUICKA: No valid dihedral.'
          IF(WRNLEV >= 2) WRITE(OUTU,105)
          call chmdealloc('quicka.src','QUICKA','ISLCT',NATOM,intg=ISLCT)
          RETURN
       ENDIF
       !
       AX=FY*GZ-FZ*GY
       AY=FZ*GX-FX*GZ
       AZ=FX*GY-FY*GX
       BX=HY*GZ-HZ*GY
       BY=HZ*GX-HX*GZ
       BZ=HX*GY-HY*GX
       !
       RA=AX*AX+AY*AY+AZ*AZ
       RB=BX*BX+BY*BY+BZ*BZ
       RA=SQRT(RA)
       RB=SQRT(RB)
       IF(RA <= 0.000001) THEN
          IF(WRNLEV >= 2) WRITE(OUTU,'(A)') &
               ' QUICKA: No valid dihedral.'
          IF(WRNLEV >= 2) WRITE(OUTU,105)
          call chmdealloc('quicka.src','QUICKA','ISLCT',NATOM,intg=ISLCT)
          RETURN
       ENDIF
       IF(RB <= 0.000001) THEN
          IF(WRNLEV >= 2) WRITE(OUTU,'(A)') &
               ' QUICKA: No valid dihedral.'
          IF(WRNLEV >= 2) WRITE(OUTU,105)
          call chmdealloc('quicka.src','QUICKA','ISLCT',NATOM,intg=ISLCT)
          RETURN
       ENDIF
       !
       AX=AX/RA
       AY=AY/RA
       AZ=AZ/RA
       BX=BX/RB
       BY=BY/RB
       BZ=BZ/RB
       !
       CST=AX*BX+AY*BY+AZ*BZ
       IF(ABS(CST) >= ONE) CST=SIGN(ONE,CST)
       PHI=ACOS(CST)
       CX=AY*BZ-AZ*BY
       CY=AZ*BX-AX*BZ
       CZ=AX*BY-AY*BX
       IF(GX*CX+GY*CY+GZ*CZ > 0.0) PHI=-PHI
       PHI=PHI*RADDEG
       IF(PRNLEV >= 2) WRITE(OUTU,75) PHI
75     FORMAT(' QUICKA: The dihedral is:',F10.3)
       !        
       call set_param('PHI ',PHI)
       IF(PRNLEV >= 2) WRITE(OUTU,105)
       call chmdealloc('quicka.src','QUICKA','ISLCT',NATOM,intg=ISLCT)
       RETURN
    ENDIF
    !-----------------------------------------------------------------------
    CALL GETNEXTP(COMLYN,COMLEN,islct, &
         XMAT,YMAT,ZMAT,BAD,ALLINT)
    IF(BAD) THEN
       call chmdealloc('quicka.src','QUICKA','ISLCT',NATOM,intg=ISLCT)
       RETURN
    ENDIF
    if (comlen > 0) then
       CALL GETNEXTP(COMLYN,COMLEN,islct, &
            XNAT,YNAT,ZNAT,BAD,ALLINT)
       IF(BAD) THEN
          call chmdealloc('quicka.src','QUICKA','ISLCT',NATOM,intg=ISLCT)
          RETURN
       ENDIF
       IF(COMLEN == 0) THEN
          ! Process puckering command
          call pucker6cp((/xiat,xjat,xkat,xlat,xmat,xnat/), &
               (/yiat,yjat,ykat,ylat,ymat,ynat/), (/ziat,zjat,zkat,zlat,zmat,znat/), &
               Q, theta, phi)
          if (prnlev > 2) write (outu,'(a,3f10.5)') &
               ' QUICKA: The puckering (Q,THET,PHI) is:',Q,theta,phi
          call set_param('Q   ',Q)
          call set_param('THET',theta)
          call set_param('PHI ',phi)
          IF(PRNLEV >= 2) WRITE(OUTU,105)
          call chmdealloc('quicka.src','QUICKA','ISLCT',NATOM,intg=ISLCT)
          return
       endif
    endif
    !
    !-----------------------------------------------------------------------
    ! Just print atom data
    DO WHILE(COMLEN > 0)
       CALL GETNEXTP(COMLYN,COMLEN,islct,XIAT,YIAT,ZIAT,BAD,ALLINT)
       IF(BAD) THEN
          call chmdealloc('quicka.src','QUICKA','ISLCT',NATOM,intg=ISLCT)
          RETURN
       ENDIF
    ENDDO
    !
    IF(PRNLEV >= 2) WRITE(OUTU,105)
    call chmdealloc('quicka.src','QUICKA','ISLCT',NATOM,intg=ISLCT)
    RETURN
    !
105 FORMAT(/)
    !=======================================================================
  END SUBROUTINE QUICKA

  SUBROUTINE GETNEXTP(COMLYN,COMLEN,ISLCT,XVAL,YVAL,ZVAL,BAD,ALLINT)
    !
    use chm_kinds
    use dimens_fcm
    use select
    use stream
    use string
    use psf
    use coord
    use coordc
    use chutil

    implicit none
    !
    CHARACTER(len=*) COMLYN
    INTEGER COMLEN
    INTEGER ISLCT(*)
    real(chm_real) XVAL,YVAL,ZVAL
    LOGICAL BAD,ALLINT
    !
    CHARACTER(len=8) SIDDN,RIDDN,RESDN,ACDN
    INTEGER I,II,NII,NSEL,iia(1)
    real(chm_real) TOTM,AM,XT,YT,ZT
    LOGICAL QMASS,QCOMP
    CHARACTER(len=4) WRD,WRD2
    !
    WRD=CURRA4(COMLYN,COMLEN)
    IF(WRD == 'SELE') THEN
       CALL SELCTA(COMLYN,COMLEN,ISLCT,X,Y,Z,WMAIN,.TRUE.)
       WRD2=CURRA4(COMLYN,COMLEN)
       IF(WRD2 == 'MASS') THEN
          WRD2=NEXTA4(COMLYN,COMLEN)
          QMASS=.TRUE.
       ELSE
          WRD2='GEOM'
          QMASS=.FALSE.
       ENDIF
       WRD=CURRA4(COMLYN,COMLEN)
       IF(WRD == 'COMP') THEN
          WRD=NEXTA4(COMLYN,COMLEN)
          QCOMP=.TRUE.
       ELSE
          WRD=' '
          QCOMP=.FALSE.
       ENDIF
       !
       NSEL=0
       TOTM=0.0
       XT=0.0
       YT=0.0
       ZT=0.0
       DO I=1,NATOM
          IF(ISLCT(I) /= 0) THEN
             NSEL=NSEL+1
             AM=1.0
             IF(QMASS) AM=AMASS(I)
             TOTM=TOTM+AM
             IF(QCOMP) THEN
                IF(.NOT.INITIA(I,XCOMP,YCOMP,ZCOMP)) THEN
                   IF(PRNLEV > 2) THEN
                      CALL ATOMID(I,SIDDN,RIDDN,RESDN,ACDN)
                      WRITE(OUTU,34) I, &
                           SIDDN(1:idleng),RIDDN(1:idleng), &
                           RESDN(1:idleng),ACDN(1:idleng)
                   ENDIF
                   BAD=.TRUE.
                   RETURN
                ENDIF
                XT=XT+XCOMP(I)*AM
                YT=YT+YCOMP(I)*AM
                ZT=ZT+ZCOMP(I)*AM
             ELSE
                IF(.NOT.INITIA(I,X,Y,Z)) THEN
                   IF(PRNLEV > 2) THEN
                      CALL ATOMID(I,SIDDN,RIDDN,RESDN,ACDN)
                      WRITE(OUTU,35) I, &
                           SIDDN(1:idleng),RIDDN(1:idleng), &
                           RESDN(1:idleng),ACDN(1:idleng)
                   ENDIF
                   BAD=.TRUE.
                   RETURN
                ENDIF
                XT=XT+X(I)*AM
                YT=YT+Y(I)*AM
                ZT=ZT+Z(I)*AM
             ENDIF
          ENDIF
       ENDDO
       IF(NSEL == 0) THEN
          IF(WRNLEV > 2) WRITE(OUTU,24)
24        FORMAT(' QUICKA:  No selected atoms.')
          BAD=.TRUE.
          RETURN
       ENDIF
       IF(TOTM > 0.0) THEN
          XVAL=XT/TOTM
          YVAL=YT/TOTM
          ZVAL=ZT/TOTM
       ELSE
          XVAL=0.0
          YVAL=0.0
          ZVAL=0.0
       ENDIF
       !
       IF(PRNLEV > 3) WRITE(OUTU,54) WRD2,NSEL,XVAL,YVAL,ZVAL,WRD
    ELSE
       !
       IF(ALLINT) THEN
          II=NEXTI(COMLYN,COMLEN)
          NII=1
       ELSE
          CALL NXTATM(IIa,NII,1,COMLYN,COMLEN,ISLCT, &
               SEGID,RESID,ATYPE,IBASE,NICTOT,NSEG,RES,NATOM)
          ii=iia(1)
       ENDIF
       !
       IF(NII /= 1) THEN
          IF(WRNLEV > 2) WRITE(OUTU,25)
25        FORMAT(' QUICKA:  Specified atom does not exist.')
          BAD=.TRUE.
          RETURN
       ENDIF
       !
       WRD=CURRA4(COMLYN,COMLEN)
       IF(WRD == 'COMP') THEN
          WRD=NEXTA4(COMLYN,COMLEN)
          QCOMP=.TRUE.
       ELSE
          WRD=' '
          QCOMP=.FALSE.
       ENDIF
       IF(QCOMP) THEN
          IF(.NOT.INITIA(II,XCOMP,YCOMP,ZCOMP)) THEN
             IF(PRNLEV > 2) THEN
                CALL ATOMID(II,SIDDN,RIDDN,RESDN,ACDN)
                WRITE(OUTU,34) II, &
                     SIDDN(1:idleng),RIDDN(1:idleng), &
                     RESDN(1:idleng),ACDN(1:idleng)
             ENDIF
             BAD=.TRUE.
             RETURN
          ENDIF
          XVAL=XCOMP(II)
          YVAL=YCOMP(II)
          ZVAL=ZCOMP(II)
       ELSE
          IF(.NOT.INITIA(II,X,Y,Z)) THEN
             IF(PRNLEV > 2) THEN
                CALL ATOMID(II,SIDDN,RIDDN,RESDN,ACDN)
                WRITE(OUTU,35) II, &
                     SIDDN(1:idleng),RIDDN(1:idleng), &
                     RESDN(1:idleng),ACDN(1:idleng)
             ENDIF
             BAD=.TRUE.
             RETURN
          ENDIF
          XVAL=X(II)
          YVAL=Y(II)
          ZVAL=Z(II)
       ENDIF
       !
       CALL ATOMID(II,SIDDN,RIDDN,RESDN,ACDN)
       IF(PRNLEV > 3) WRITE(OUTU,55)  &
            II,SIDDN(1:idleng),RIDDN(1:idleng), &
            RESDN(1:idleng),ACDN(1:idleng), &
            XVAL,YVAL,ZVAL,WRD
    ENDIF
    !
    CALL TRIMA(COMLYN,COMLEN)
    !
34  FORMAT(' QUICKA: Atom',I8,4(1X,A), &
         ' does not have a valid COMP position.')
35  FORMAT(' QUICKA: Atom',I8,4(1X,A), &
         ' does not have a valid position.')
54  FORMAT(' QUICKA: Center of ',A4,' of ',I8, &
         ' selected atoms is at:',3F10.5,2X,A4)
55  FORMAT(' QUICKA: Atom',I8,4(1X,A),' is at:',3F10.5,2X,A)
    !
    return
  end SUBROUTINE GETNEXTP
#endif /*  (nomisc)*/

  SUBROUTINE NULL_QK
    RETURN
  END SUBROUTINE NULL_QK

end module quick

