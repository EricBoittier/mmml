!CHARMM Element source/manip/rxcons_2.src $Revision: 1.16 $
module rxcons2
contains


#if KEY_RXNCONS==0 /*rxcons_2*/
  SUBROUTINE RX_2_SETUP(COMLYN,COMLEN)
    INTEGER COMLEN
    CHARACTER(len=*) COMLYN
    return
  end SUBROUTINE RX_2_SETUP
#else /* (rxcons_2)*/
  SUBROUTINE RX_2_SETUP(COMLYN,COMLEN)

    use dimens_fcm
    use number
    use coord
    use memory
    use shake
    use psf
    use select
    use stream
    use rxncons
    use chm_kinds
    implicit none
    integer,allocatable,dimension(:) :: ISLCT
    integer,allocatable,dimension(:) :: JSLCT
    INTEGER COMLEN
    CHARACTER(len=*) COMLYN

    ! . Local variables.
    LOGICAL ERR
    !
    ! . set up counter
    !

    IF(.NOT.LCLEAN) THEN
       call chmalloc('rxcons_2.src','RX_2_SETUP','IRXATM',NRXATM,intg=IRXATM)
       call chmalloc('rxcons_2.src','RX_2_SETUP','XCORD',NRXATM,crl=XCORD)
       call chmalloc('rxcons_2.src','RX_2_SETUP','YCORD',NRXATM,crl=YCORD)
       call chmalloc('rxcons_2.src','RX_2_SETUP','ZCORD',NRXATM,crl=ZCORD)
       call chmalloc('rxcons_2.src','RX_2_SETUP','XREFC',NRXATM,crl=XREFC)
       call chmalloc('rxcons_2.src','RX_2_SETUP','YREFC',NRXATM,crl=YREFC)
       call chmalloc('rxcons_2.src','RX_2_SETUP','ZREFC',NRXATM,crl=ZREFC)

       ! . Parse the double atom selection

       call chmalloc('rxcons_2.src','RX_2_SETUP','ISLCT',NATOM,intg=ISLCT)
       call chmalloc('rxcons_2.src','RX_2_SETUP','jSLCT',NATOM,intg=jSLCT)
       CALL SELCTD(COMLYN,COMLEN,ISLCT,JSLCT,X,Y,Z,WMAIN,.TRUE.,ERR)
       !
       ! . Set
       !
       CALL RXNCHK_2(X,Y,Z,NRXATM,LCALRX,VRXNCS,LRXPRN, &
            ISLCT,JSLCT,IRXATM,XCORD,YCORD,ZCORD)

       IF(IRXCNS.GT.0) LRXCNS=.TRUE.
       !
       IF(LRXCNS) QHOLO=.TRUE.

       call chmdealloc('rxcons_2.src','RX_2_SETUP','ISLCT',NATOM,intg=ISLCT)
       call chmdealloc('rxcons_2.src','RX_2_SETUP','jSLCT',NATOM,intg=jSLCT)

    ELSE ! LCLEAN

       call chmdealloc('rxcons_2.src','RX_2_SETUP','IRXATM',NRXATM,intg=IRXATM)
       call chmdealloc('rxcons_2.src','RX_2_SETUP','XCORD',NRXATM,crl=XCORD)
       call chmdealloc('rxcons_2.src','RX_2_SETUP','YCORD',NRXATM,crl=YCORD)
       call chmdealloc('rxcons_2.src','RX_2_SETUP','ZCORD',NRXATM,crl=ZCORD)
       call chmdealloc('rxcons_2.src','RX_2_SETUP','XREFC',NRXATM,crl=XREFC)
       call chmdealloc('rxcons_2.src','RX_2_SETUP','YREFC',NRXATM,crl=YREFC)
       call chmdealloc('rxcons_2.src','RX_2_SETUP','ZREFC',NRXATM,crl=ZREFC)

    ENDIF

    RETURN
  END SUBROUTINE RX_2_SETUP

  SUBROUTINE RXNCHK_2(X,Y,Z,NRXATM,QCALRX,VRXNCS,LRXPRN, &
       ISLCT,JSLCT,IRXATM,XCORD,YCORD,ZCORD)
    !
    use chm_kinds
    use dimens_fcm
    use select
    use number
    use psf
    use stream
    !
    implicit none
    !
    ! . Parsed variables
    !
    INTEGER ISLCT(*),JSLCT(*),IRXATM(*),NRXATM
    real(chm_real) X(*),Y(*),Z(*),VRXNCS
    real(chm_real) XCORD(*),YCORD(*),ZCORD(*)
    LOGICAL QCALRX,LRXPRN
    !
    ! . Local variables
    !
    INTEGER I,INT,JNT,NTRXN,J,ITEMP
    real(chm_real) D
    INT=NSELCT(NATOM,ISLCT)
    JNT=NSELCT(NATOM,JSLCT)
    NTRXN=INT+JNT

    if (prnlev >= 2) then
       WRITE(OUTU,24) 'The bond length reaction coordinate is selected'
       WRITE(OUTU,25) 'number of atoms defining the rxn: ', NRXATM
    endif

    IF(INT.NE.1) THEN
       WRITE(OUTU,25) 'The number of atoms in the first selection: ' &
            , INT
       CALL WRNDIE(-3,'<RXNCHK1>', &
            'exactly one atom is needed for the first atom selection')
    ENDIF

    IF(JNT.NE.1) THEN
       WRITE(OUTU,25) 'The number of atoms in the second selection: ' &
            , JNT
       CALL WRNDIE(-3,'<RXNCHK1>', &
            'exactly one atoms are needed for the second atom selection')
    ENDIF

    J=0

    DO I=1,NATOM
       IF(ISLCT(I).EQ.1) THEN
          J=J+1
          IF(J.GT.NRXATM) CALL WRNDIE(-3,'<RXNCHK1>', &
               'exactly two aoms are needed for this rxn type')
          IRXATM(J)=I
       ENDIF
    ENDDO
    DO I=1,NATOM
       IF(JSLCT(I).EQ.1) THEN
          J=J+1
          IF(J.GT.NRXATM) CALL WRNDIE(-3,'<RXNCHK1>', &
               'exactly two aoms are needed for this rxn type')
          IRXATM(J)=I
       ENDIF
    ENDDO
    !
    ! . Test printing
    !
    IF(LRXPRN)THEN
       if (prnlev >= 2) WRITE(OUTU,24)'Index of selected atoms'
       DO I=1,NRXATM
          if (prnlev >= 2) WRITE(OUTU,10) I, IRXATM(I)
       ENDDO
    ENDIF
    !
    ! . COPY coordinates from current coordinate set
    !
    DO I=1,NRXATM
       ITEMP=IRXATM(I)
       XCORD(I)=X(ITEMP)
       YCORD(I)=Y(ITEMP)
       ZCORD(I)=Z(ITEMP)
       !       WRITE(OUTU,11) I, IRXATM(I), XCORD(I), YCORD(I), ZCORD(I)
    ENDDO

    CALL BDIS12(XCORD,YCORD,ZCORD,D)
    !
    ! . Test printing
    !
    if (prnlev >= 2) then
       WRITE(OUTU,26) 'The set value of rxn coord: ', VRXNCS
       WRITE(OUTU,26) 'The rxn coord from the current coord  : ', D
    endif

    IF(ABS(D-VRXNCS).GT.RSMALL) THEN
       CALL RXNPALN1_2(XCORD,YCORD,ZCORD,D,VRXNCS)
       if (prnlev >= 2) then
          WRITE(OUTU,26) 'The set value of rxn coord: ', VRXNCS
          WRITE(OUTU,27) 'The adjusted rxn coord    : ', D
       endif
       DO I=1,NRXATM
          ITEMP=IRXATM(I)
          X(ITEMP)=XCORD(I)
          Y(ITEMP)=YCORD(I)
          Z(ITEMP)=ZCORD(I)
       ENDDO
    ENDIF

24  FORMAT('RXNCONS> ',A)
25  FORMAT('RXNCONS> ',A,I5)
26  FORMAT('RXNCONS> ',A,1F16.6)
27  FORMAT('RXNCONS> ',A,3F16.6)
10  FORMAT(2I5)
11  FORMAT(2I5,3F16.6)
12  FORMAT(2F16.6)
    RETURN
  END SUBROUTINE RXNCHK_2

  SUBROUTINE BDIS12(XCORD,YCORD,ZCORD,D)
    use chm_kinds
    use dimens_fcm
    use vector
    use number
    implicit none
    !
    ! . Parsed variables
    !
    real(chm_real) XCORD(*),YCORD(*),ZCORD(*),D
    !
    !
    ! . Local variables
    !
    real(chm_real) DRVEC(3)

    DRVEC(1)=XCORD(2)-XCORD(1)
    DRVEC(2)=YCORD(2)-YCORD(1)
    DRVEC(3)=ZCORD(2)-ZCORD(1)
    D=SQRT(DOTVEC(DRVEC,DRVEC,3))

    RETURN
  END SUBROUTINE BDIS12

  SUBROUTINE RXNPALN1_2(XCORD,YCORD,ZCORD,D,VRXNCS)
    use chm_kinds
    use dimens_fcm
    use vector
    use number
    use stream
    implicit none
    !
    ! . Parsed variables
    !
    real(chm_real) XCORD(*),YCORD(*),ZCORD(*),D,VRXNCS
    !
    ! . Local variables
    !
    INTEGER I,IC
    real(chm_real) XD(2),YD(2),ZD(2),LM ! not the best way
    real(chm_real) A(3),PD2(2),PFPL

    DO I=1,2
       XD(I)=XCORD(I)
       YD(I)=YCORD(I)
       ZD(I)=ZCORD(I)
    ENDDO

    IC=1

100 CONTINUE

    LM=ZERO

    CALL PFPRTWO_2(XD,YD,ZD,A)
    PFPL=DOTVEC(A,A,3)

    LM=-(D-VRXNCS)/PFPL
    XD(2)=XD(2)+LM*A(1)*HALF
    YD(2)=YD(2)+LM*A(2)*HALF
    ZD(2)=ZD(2)+LM*A(3)*HALF
    XD(1)=XD(1)-LM*A(1)*HALF
    YD(1)=YD(1)-LM*A(2)*HALF
    ZD(1)=ZD(1)-LM*A(3)*HALF

    CALL BDIS12(XD,YD,ZD,D)

    !      WRITE(OUTU,10) I, D, VRXNCS, LM

    IC=IC+1

    IF(IC.GT.2000) CALL WRNDIE(-3,'<RXNPALN1>', &
         'alignment does not converge')

    IF(ABS(D-VRXNCS).GT.RSMALL) GOTO 100

    DO I=1,2
       XCORD(I)=XD(I)
       YCORD(I)=YD(I)
       ZCORD(I)=ZD(I)
    ENDDO

10  FORMAT(I5,3F16.6)
    RETURN
  END SUBROUTINE RXNPALN1_2

  SUBROUTINE PFPRTWO_2(XCD,YCD,ZCD,PD2)
    use chm_kinds
    use dimens_fcm
    use vector
    use number
    implicit none
    !
    ! . Parsed variables
    !
    real(chm_real) XCD(*),YCD(*),ZCD(*),PD2(*)
    !
    ! . Local variables
    !
    INTEGER I
    real(chm_real) DRVEC(3),TEMP,D1 ! not the best way

    DRVEC(1)=XCD(2)-XCD(1)
    DRVEC(2)=YCD(2)-YCD(1)
    DRVEC(3)=ZCD(2)-ZCD(1)
    D1=SQRT(DOTVEC(DRVEC,DRVEC,3))

    TEMP=ZERO
    IF(D1.GT.RSMALL) TEMP=ONE/D1

    DO I=1,3
       PD2(I)=DRVEC(I)*TEMP
    ENDDO

    RETURN
  END SUBROUTINE PFPRTWO_2

  SUBROUTINE RXNCONS2(X,Y,Z,XREF,YREF,ZREF,AMASS,NATOM,IRXCNS, &
       NRXATM,LRXCNS,LDYNA,VRXNCS,LAGRANM,MXITER,PLAGRANM, &
       LRXPRN,IRXATM,XCORD,YCORD,ZCORD, &
       XREFC,YREFC,ZREFC,DTSQ,ZFAC,GFAC)
    !
    use chm_kinds
    use dimens_fcm
    use exfunc
    use number
    use stream
    use parallel
    use machutil,only:eclock
    implicit none
    !
    ! . Parsed variables
    !
    real(chm_real) X(*),Y(*),Z(*),XREF(*),YREF(*),ZREF(*)
    real(chm_real) AMASS(*)
    INTEGER NATOM,IRXCNS,NRXATM,IRXATM(*),MXITER
    LOGICAL LRXCNS,LDYNA,PLAGRANM,LRXPRN
    real(chm_real) VRXNCS,LAGRANM,XCORD(*),YCORD(*),ZCORD(*)
    real(chm_real) XREFC(*),YREFC(*),ZREFC(*),DTSQ,ZFAC,GFAC
    !
#if KEY_PARALLEL==1
    real(chm_real) GCARR(10),TIMMER
#endif 
    !
    ! . Local variables
    !
    INTEGER I, IA
    INTEGER ATFRST,ATLAST

    ! . Just testing
    !
    !      ATFRST=1
    !      ATLAST=NATOM
    !...##IF PARALLEL
    !...##IF PARAFULL
    !C     Define the atom bounds for this processor.
    !      IF(LDYNA) THEN
    !        ATFRST=1+IPARPT(MYNOD)
    !        ATLAST=IPARPT(MYNODP)
    !      ENDIF
    !...##ENDIF
    !...##ENDIF
    !
    !
    !. Copy coordinates
    !
#if KEY_PARALLEL==1
    CALL PSYNC()
    TIMMER=ECLOCK()
    CALL VDGBR(X,Y,Z,1)
    CALL VDGBR(XREF,YREF,ZREF,1)
    TMERI(TIMGCOMM)=TMERI(TIMGCOMM)+ECLOCK()-TIMMER
    CALL PSYNC()
    TIMMER=ECLOCK()
#endif 

    DO I=1,NRXATM
       IA=IRXATM(I)
       XCORD(I)=X(IA)
       YCORD(I)=Y(IA)
       ZCORD(I)=Z(IA)
       XREFC(I)=XREF(IA)
       YREFC(I)=YREF(IA)
       ZREFC(I)=ZREF(IA)
       !        IF(MYNOD.EQ.0) THEN
       !          WRITE(OUTU,*)' before shake, X ',IA,X(IA),Y(IA),Z(IA)
       !          WRITE(OUTU,*)' before shake, XREF ',IA,XREF(IA),YREF(IA),
       !     &                 ZREF(IA)
       !        ENDIF
    ENDDO

    LAGRANM=ZERO

    CALL RXNCONS_22(AMASS,NATOM,NRXATM,VRXNCS,LAGRANM,MXITER,IRXATM, &
         LRXPRN, &
         XCORD,YCORD,ZCORD,XREFC,YREFC,ZREFC,ZFAC,GFAC)

    DO I=1,NRXATM
       IA=IRXATM(I)
       X(IA)=XCORD(I)
       Y(IA)=YCORD(I)
       Z(IA)=ZCORD(I)
       !        IF(MYNOD.EQ.0) THEN
       !          WRITE(OUTU,*)' after shake, X ',IA,X(IA),Y(IA),Z(IA)
       !          WRITE(OUTU,*)' after shake, XREF ',IA,XREF(IA),YREF(IA),
       !     &                 ZREF(IA)
       !        ENDIF
    ENDDO

    !       IF(PLAGRANM) THEN
    !...##IF PARALLEL
    !         IF(MYNOD.EQ.0) THEN
    !...##ENDIF
    !         WRITE(OUTU,20) 'RXNCONS> LAGRANGE MULTIPLIER (kcal/mol/A) IS:',
    !     &   LAGRANM, ' Z^-1/2= ', ONE/SQRT(ZFAC), ' G= ', GFAC

    !...##IF PARALLEL
    !         ENDIF
    !...##ENDIF
    !       ENDIF

10  FORMAT(A,2I5,3F16.6)
20  FORMAT(3(A,1F16.6))

    RETURN
  END SUBROUTINE RXNCONS2


  SUBROUTINE RXNCONS_22(AMASS,NATOM,NRXATM,VRXNCS,LAGRANM,MXITER, &
       IRXATM,LRXPRN, &
       XCORD,YCORD,ZCORD,XREFC,YREFC,ZREFC, &
       ZFAC,GFAC)

    use chm_kinds
    use dimens_fcm
    use vector
    use number
    use stream
#if KEY_PARALLEL==1
    use parallel
#endif 

    implicit none
    !
    ! . Parsed variables
    !
    real(chm_real) AMASS(*)
    INTEGER NATOM,NRXATM,IRXATM(*),MXITER
    real(chm_real) VRXNCS,LAGRANM,XCORD(*),YCORD(*),ZCORD(*)
    real(chm_real) XREFC(*),YREFC(*),ZREFC(*),ZFAC,GFAC
    LOGICAL LRXPRN
    !
    !
    INTEGER I,IA, ITER
    real(chm_real) MASS(3),R2MR1(3)
    real(chm_real) PP1(3),PP2MPP1(3)
    real(chm_real) RREF1(3),RREF2(3),RREF3(3),FUN,DFUNDL,LOLD
    real(chm_real) R2MR1r(3),A

    ITER=0
    LAGRANM=ZERO
    ZFAC=ZERO
    GFAC=ZERO

    DO I=1,NRXATM
       IA=IRXATM(I)
       MASS(I)=ONE/AMASS(IA)
    ENDDO

    IF(ITER.EQ.0) THEN
       R2MR1r(1)=XREFC(2)-XREFC(1)
       R2MR1r(2)=YREFC(2)-YREFC(1)
       R2MR1r(3)=ZREFC(2)-ZREFC(1)

       A=ONE/SQRT(DOTVEC(R2MR1r,R2MR1r,3))

       DO I=1,3
          PP2MPP1(I)=TWO*R2MR1r(I)*A*MASS(1)
       ENDDO

       R2MR1r(1)=XCORD(2)-XCORD(1)
       R2MR1r(2)=YCORD(2)-YCORD(1)
       R2MR1r(3)=ZCORD(2)-ZCORD(1)

       ITER=1
       LOLD=LAGRANM
    ENDIF

100 CONTINUE

    DO I=1,3
       R2MR1(I)=R2MR1r(I)+LAGRANM*PP2MPP1(I)
    ENDDO

    FUN=SQRT(DOTVEC(R2MR1,R2MR1,3))-VRXNCS

    !...##IF PARALLEL
    !      IF(MYNOD.EQ.0) THEN
    !...##ENDIF
    !        WRITE(OUTU,20) ITER, LAGRANM, FUN,SQRT(DOTVEC(R2MR1,R2MR1,3))
    !...##IF PARALLEL
    !      ENDIF
    !...##ENDIF

    IF(ABS(FUN).GT.RSMALL) THEN
       A=DOTVEC(R2MR1,PP2MPP1,3)/SQRT(DOTVEC(R2MR1,R2MR1,3))
       DFUNDL=A
       LAGRANM=LOLD-FUN/DFUNDL
       ITER=ITER+1
       LOLD=LAGRANM
       IF(ITER.GT.MXITER) THEN
          WRITE(OUTU,311) MXITER
          CALL DIEWRN(-2)
          RETURN
       ENDIF
       GOTO 100
    ENDIF

    DO I=1,3
       PP1(I)=HALF*PP2MPP1(I)
    ENDDO

    XCORD(1)=XCORD(1)-LAGRANM*PP1(1)
    YCORD(1)=YCORD(1)-LAGRANM*PP1(2)
    ZCORD(1)=ZCORD(1)-LAGRANM*PP1(3)
    XCORD(2)=XCORD(2)+LAGRANM*PP1(1)
    YCORD(2)=YCORD(2)+LAGRANM*PP1(2)
    ZCORD(2)=ZCORD(2)+LAGRANM*PP1(3)

    R2MR1r(1)=XCORD(2)-XCORD(1)
    R2MR1r(2)=YCORD(2)-YCORD(1)
    R2MR1r(3)=ZCORD(2)-ZCORD(1)

    A=ONE/SQRT(DOTVEC(R2MR1r,R2MR1r,3))

    ZFAC=ONE

    GFAC=ZERO

20  FORMAT(I5,4F16.6)
311 FORMAT(' ***** ERROR IN RXNCONS_2 ***** COORDINATE SETTING ', &
         ' WAS NOT ACCOMPLISHED IN',I6,' ITERATIONS',/)

    RETURN
  END SUBROUTINE RXNCONS_22

  SUBROUTINE RXNCONS_2F(XREF,YREF,ZREF,DX,DY,DZ,NATOM,IRXCNS, &
       NRXATM,LRXPRN,IRXATM)

    use chm_kinds
    use dimens_fcm
    use vector
    use number
    use stream
    implicit none
    !
    ! . Parsed variables
    !
    real(chm_real) XREF(*),YREF(*),ZREF(*),DX(*),DY(*),DZ(*)
    INTEGER NATOM,IRXCNS,NRXATM,IRXATM(*)
    LOGICAL LRXPRN
    !
    !
    ! . Local variables
    !
    INTEGER I, J, IA, IB
    real(chm_real) R2MR1(3),A,RXTAN(3,NRXATM),SUM,FD

    IA=IRXATM(1)
    IB=IRXATM(2)
    R2MR1(1)=XREF(IB)-XREF(IA)
    R2MR1(2)=YREF(IB)-YREF(IA)
    R2MR1(3)=ZREF(IB)-ZREF(IA)
    A=ONE/SQRT(DOTVEC(R2MR1,R2MR1,3))

    DO I=1,3
       R2MR1(I)=R2MR1(I)*A
    ENDDO

    DO I=1,3
       RXTAN(I,1)=-R2MR1(I)
       RXTAN(I,2)= R2MR1(I)
    ENDDO

    SUM=ZERO
    DO I=1,NRXATM
       DO J=1,3
          SUM=SUM+RXTAN(J,I)*RXTAN(J,I)
       ENDDO
    ENDDO

    SUM=ONE/SQRT(SUM)
    DO I=1,NRXATM
       DO J=1,3
          RXTAN(J,I)=RXTAN(J,I)*SUM
       ENDDO
    ENDDO

    FD=ZERO
    DO I=1,NRXATM
       IA=IRXATM(I)
       FD=FD+RXTAN(1,I)*DX(IA)
       FD=FD+RXTAN(2,I)*DY(IA)
       FD=FD+RXTAN(3,I)*DZ(IA)
    ENDDO

    DO I=1,NRXATM
       IA=IRXATM(I)
       DX(IA)=DX(IA)-FD*RXTAN(1,I)
       DY(IA)=DY(IA)-FD*RXTAN(2,I)
       DZ(IA)=DZ(IA)-FD*RXTAN(3,I)
    ENDDO
    RETURN
  END SUBROUTINE RXNCONS_2F
#endif /* (rxcons_2)*/
end module rxcons2

