module rxcons1
contains

#if KEY_RXNCONS==0 /*rxcons_1*/
  SUBROUTINE RX_1_SETUP(COMLYN,COMLEN)
    INTEGER COMLEN
    CHARACTER(len=*) COMLYN
    return
  end SUBROUTINE RX_1_SETUP
#else /* (rxcons_1)*/
  SUBROUTINE RX_1_SETUP(COMLYN,COMLEN)
    use rxncons
    use chm_kinds
    use dimens_fcm
    use number
    use coord
    use memory
    use shake
    use psf
    use select
    use stream
    !
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
       call chmalloc('rxcons_1.src','RX_1_SETUP','IRXATM',NRXATM,intg=IRXATM)
       call chmalloc('rxcons_1.src','RX_1_SETUP','XCORD',NRXATM,crl=XCORD)
       call chmalloc('rxcons_1.src','RX_1_SETUP','YCORD',NRXATM,crl=YCORD)
       call chmalloc('rxcons_1.src','RX_1_SETUP','ZCORD',NRXATM,crl=ZCORD)
       call chmalloc('rxcons_1.src','RX_1_SETUP','XREFC',NRXATM,crl=XREFC)
       call chmalloc('rxcons_1.src','RX_1_SETUP','YREFC',NRXATM,crl=YREFC)
       call chmalloc('rxcons_1.src','RX_1_SETUP','ZREFC',NRXATM,crl=ZREFC)

       ! . Parse the double atom selection

       call chmalloc('rxcons_1.src','RX_1_SETUP','ISLCT',NATOM,intg=ISLCT)
       call chmalloc('rxcons_1.src','RX_1_SETUP','JSLCT',NATOM,intg=JSLCT)
       CALL SELCTD(COMLYN,COMLEN,ISLCT,JSLCT,X,Y,Z,WMAIN,.TRUE.,ERR)
       !
       ! . Check if the coordinates read satify the constraint
       !
       CALL RXNCHK_1(X,Y,Z,NRXATM,LCALRX,VRXNCS,LRXPRN, &
            ISLCT,JSLCT,IRXATM,XCORD,YCORD,ZCORD)

       IF(IRXCNS.GT.0) LRXCNS=.TRUE.
       !
       IF(LRXCNS) QHOLO=.TRUE.

       call chmdealloc('rxcons_1.src','RX_1_SETUP','ISLCT',NATOM,intg=ISLCT)
       call chmdealloc('rxcons_1.src','RX_1_SETUP','JSLCT',NATOM,intg=JSLCT)

    ELSE ! LCLEAN

       call chmdealloc('rxcons_1.src','RX_1_SETUP','IRXATM', &
            NRXATM,intg=IRXATM)
       call chmdealloc('rxcons_1.src','RX_1_SETUP','XCORD',NRXATM,crl=XCORD)
       call chmdealloc('rxcons_1.src','RX_1_SETUP','YCORD',NRXATM,crl=YCORD)
       call chmdealloc('rxcons_1.src','RX_1_SETUP','ZCORD',NRXATM,crl=ZCORD)
       call chmdealloc('rxcons_1.src','RX_1_SETUP','XREFC',NRXATM,crl=XREFC)
       call chmdealloc('rxcons_1.src','RX_1_SETUP','YREFC',NRXATM,crl=YREFC)
       call chmdealloc('rxcons_1.src','RX_1_SETUP','ZREFC',NRXATM,crl=ZREFC)

    ENDIF
    RETURN
  END subroutine RX_1_SETUP

  
  SUBROUTINE RXNCHK_1(X,Y,Z,NRXATM,QCALRX,VRXNCS,LRXPRN, &
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
    real(chm_real) D1,D2,D1MD2
    INT=NSELCT(NATOM,ISLCT)
    JNT=NSELCT(NATOM,JSLCT)
    NTRXN=INT+JNT

    if (prnlev >= 2) then
       WRITE(OUTU,24) 'The d1-d2 reaction coordinate is selected'
       WRITE(OUTU,25) 'number of atoms defining the rxn: ', NRXATM
    endif

    IF(INT.NE.1) THEN
       WRITE(OUTU,25) 'The number of atoms in the first selection: ' &
            , INT
       CALL WRNDIE(-3,'<RXNCHK1>', &
            'exactly one atom is needed for the first atom selection')
    ENDIF

    IF(JNT.NE.2) THEN
       WRITE(OUTU,25) 'The number of atoms in the second selection: ' &
            , JNT
       CALL WRNDIE(-3,'<RXNCHK1>', &
            'exactly two atoms are needed for the second atom selection')
    ENDIF

    J=0

    DO I=1,NATOM
       IF(ISLCT(I).EQ.1) THEN
          J=J+1
          IF(J.GT.NRXATM) CALL WRNDIE(-3,'<RXNCHK1>', &
               'exactly three atoms are needed for this rxn type')
          IRXATM(J)=I
       ENDIF
    ENDDO
    DO I=1,NATOM
       IF(JSLCT(I).EQ.1) THEN
          J=J+1
          IF(J.GT.NRXATM) CALL WRNDIE(-3,'<RXNCHK1>', &
               'exactly three atoms are needed for this rxn type')
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

    CALL D1MIND2(XCORD,YCORD,ZCORD,D1,D2,D1MD2)
    !
    ! . Test printing
    !
    if (prnlev >= 2) then
       WRITE(OUTU,26) 'The set value of rxn coord: ', VRXNCS
       WRITE(OUTU,26) 'The rxn coord from the current coord  : ', D1MD2
    endif

    IF(ABS(D1MD2-VRXNCS).GT.RSMALL) THEN
       CALL RXNPALN1_1(XCORD,YCORD,ZCORD,D1,D2,D1MD2,VRXNCS)
       if (prnlev >= 2) then
          WRITE(OUTU,26) 'The set value of rxn coord: ', VRXNCS
          WRITE(OUTU,27) 'The adjusted rxn coord    : ', D1,D2,D1MD2
       endif
       DO I=1,NRXATM
          ITEMP=IRXATM(I)
          X(ITEMP)=XCORD(I)
          Y(ITEMP)=YCORD(I)
          Z(ITEMP)=ZCORD(I)
       ENDDO
    ENDIF

24  FORMAT('RXNCONS: ',A)
25  FORMAT('RXNCONS: ',A,I5)
26  FORMAT('RXNCONS: ',A,1F16.6)
27  FORMAT('RXNCONS: ',A,3F16.6)
10  FORMAT(2I5)
11  FORMAT(2I5,3F16.6)
12  FORMAT(2F16.6)
    RETURN
  END SUBROUTINE RXNCHK_1

  SUBROUTINE D1MIND2(XCORD,YCORD,ZCORD,D1,D2,D1MD2)
    !
    use chm_kinds
    use dimens_fcm
    use vector
    use number
    implicit none
    !
    ! . Parsed variables
    !
    real(chm_real) XCORD(*),YCORD(*),ZCORD(*),D1,D2,D1MD2
    !
    !
    ! . Local variables
    !
    real(chm_real) DRVEC(3)

    DRVEC(1)=XCORD(2)-XCORD(1)
    DRVEC(2)=YCORD(2)-YCORD(1)
    DRVEC(3)=ZCORD(2)-ZCORD(1)
    D1=SQRT(DOTVEC(DRVEC,DRVEC,3))

    DRVEC(1)=XCORD(3)-XCORD(2)
    DRVEC(2)=YCORD(3)-YCORD(2)
    DRVEC(3)=ZCORD(3)-ZCORD(2)
    D2=SQRT(DOTVEC(DRVEC,DRVEC,3))

    D1MD2=D1-D2

    RETURN
  END SUBROUTINE D1MIND2

  SUBROUTINE RXNPALN1_1(XCORD,YCORD,ZCORD,D1,D2,D1MD2,VRXNCS)
    !
    use chm_kinds
    use dimens_fcm
    use vector
    use number
    use stream
    !
    implicit none
    !
    ! . Parsed variables
    !
    real(chm_real) XCORD(*),YCORD(*),ZCORD(*),D1,D2,D1MD2,VRXNCS
    !
    ! . Local variables
    !
    INTEGER I,IC
    real(chm_real) DRVEC(3),XD(3),YD(3),ZD(3),LM ! not the best way
    real(chm_real) A(3),PD2(3),PFPL

    DO I=1,3
       XD(I)=XCORD(I)
       YD(I)=YCORD(I)
       ZD(I)=ZCORD(I)
    ENDDO

    IC=1

100 CONTINUE

    LM=ZERO

    CALL PFPRTWO1_1(XD,YD,ZD,A)
    PFPL=DOTVEC(A,A,3)

    LM=-(D1MD2-VRXNCS)/PFPL
    XD(2)=XD(2)+LM*A(1)
    YD(2)=YD(2)+LM*A(2)
    ZD(2)=ZD(2)+LM*A(3)

    CALL D1MIND2(XD,YD,ZD,D1,D2,D1MD2)

    !      WRITE(OUTU,10) I, D1MD2, VRXNCS, LM

    IC=IC+1

    IF(IC.GT.2000) CALL WRNDIE(-3,'<RXNPALN1>', &
         'alignment does not converge')

    IF(ABS(D1MD2-VRXNCS).GT.RSMALL) GOTO 100

    DO I=1,3
       XCORD(I)=XD(I)
       YCORD(I)=YD(I)
       ZCORD(I)=ZD(I)
    ENDDO

10  FORMAT(I5,3F16.6)
    RETURN
  END SUBROUTINE RXNPALN1_1

  SUBROUTINE PFPRTWO1_1(XCD,YCD,ZCD,PD2)
    !
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
    !
    ! . Local variables
    !
    INTEGER I
    real(chm_real) DRVEC(3),TEMP,D1,D2 ! not the best way

    DRVEC(1)=XCD(2)-XCD(1)
    DRVEC(2)=YCD(2)-YCD(1)
    DRVEC(3)=ZCD(2)-ZCD(1)
    D1=SQRT(DOTVEC(DRVEC,DRVEC,3))

    TEMP=ZERO
    IF(D1.GT.RSMALL) TEMP=ONE/D1

    DO I=1,3
       PD2(I)=DRVEC(I)*TEMP
    ENDDO

    DRVEC(1)=XCD(3)-XCD(2)
    DRVEC(2)=YCD(3)-YCD(2)
    DRVEC(3)=ZCD(3)-ZCD(2)
    D2=SQRT(DOTVEC(DRVEC,DRVEC,3))

    TEMP=ZERO
    IF(D2.GT.RSMALL) TEMP=ONE/D2

    DO I=1,3
       PD2(I)=PD2(I)+DRVEC(I)*TEMP
    ENDDO

    RETURN
  END SUBROUTINE PFPRTWO1_1

  SUBROUTINE RXNCONS_1(X,Y,Z,XREF,YREF,ZREF,AMASS,NATOM,IRXCNS, &
       NRXATM,LRXCNS,LDYNA,VRXNCS,LAGRANM,MXITER,PLAGRANM, &
       LRXPRN,IRXATM,XCORD,YCORD,ZCORD,XREFC,YREFC,ZREFC, &
       DTSQ,ZFAC,GFAC)
    !
    use chm_kinds
    use dimens_fcm
    use exfunc
    use number
    use stream
#if KEY_PARALLEL==1
    use parallel     
#endif
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

    CALL RXNCONS_12(AMASS,NATOM,NRXATM,VRXNCS,LAGRANM,MXITER,LRXPRN, &
         IRXATM,XCORD,YCORD,ZCORD,XREFC,YREFC,ZREFC, &
         ZFAC,GFAC)

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
  END SUBROUTINE RXNCONS_1

  SUBROUTINE RXNCONS_12(AMASS,NATOM,NRXATM,VRXNCS,LAGRANM,MXITER, &
       LRXPRN,IRXATM,XCORD,YCORD,ZCORD, &
       XREFC,YREFC,ZREFC,ZFAC,GFAC)

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
    LOGICAL LRXPRN
    real(chm_real) VRXNCS,LAGRANM,XCORD(*),YCORD(*),ZCORD(*)
    real(chm_real) XREFC(*),YREFC(*),ZREFC(*),ZFAC,GFAC

    !
    INTEGER I,IA, ITER
    real(chm_real) MASS(3),R2MR1(3),R3MR2(3),PP1(3),PP2(3),PP3(3)
    real(chm_real) PP2MPP1(3),PP3MPP2(3)
    real(chm_real) RREF1(3),RREF2(3),RREF3(3),FUN,DFUNDL,LOLD
    real(chm_real) R2MR1r(3),R3MR2r(3),A,B

    ITER=0
    LAGRANM=ZERO
    ZFAC=ZERO
    GFAC=ZERO

    DO I=1,3
       IA=IRXATM(I)
       MASS(I)=ONE/AMASS(IA)
    ENDDO

    IF(ITER.EQ.0) THEN
       R2MR1r(1)=XREFC(2)-XREFC(1)
       R2MR1r(2)=YREFC(2)-YREFC(1)
       R2MR1r(3)=ZREFC(2)-ZREFC(1)
       R3MR2r(1)=XREFC(3)-XREFC(2)
       R3MR2r(2)=YREFC(3)-YREFC(2)
       R3MR2r(3)=ZREFC(3)-ZREFC(2)

       A=ONE/SQRT(DOTVEC(R2MR1r,R2MR1r,3))
       B=ONE/SQRT(DOTVEC(R3MR2r,R3MR2r,3))

       DO I=1,3
          PP1(I)=-R2MR1r(I)*A*MASS(1)
          PP2(I)=(R2MR1r(I)*A+R3MR2r(I)*B)*MASS(2)
          PP3(I)=-R3MR2r(I)*B*MASS(3)
          PP2MPP1(I)=PP2(I)-PP1(I)
          PP3MPP2(I)=PP3(I)-PP2(I)
       ENDDO

       R2MR1r(1)=XCORD(2)-XCORD(1)
       R2MR1r(2)=YCORD(2)-YCORD(1)
       R2MR1r(3)=ZCORD(2)-ZCORD(1)
       R3MR2r(1)=XCORD(3)-XCORD(2)
       R3MR2r(2)=YCORD(3)-YCORD(2)
       R3MR2r(3)=ZCORD(3)-ZCORD(2)

       ITER=1
       LOLD=LAGRANM

    ENDIF

100 CONTINUE

    DO I=1,3
       R2MR1(I)=R2MR1r(I)+LAGRANM*PP2MPP1(I)
       R3MR2(I)=R3MR2r(I)+LAGRANM*PP3MPP2(I)
    ENDDO

    FUN=SQRT(DOTVEC(R2MR1,R2MR1,3))-SQRT(DOTVEC(R3MR2,R3MR2,3))- &
         VRXNCS

    !...##IF PARALLEL
    !      IF(MYNOD.EQ.0) THEN
    !...##ENDIF
    !        WRITE(OUTU,20) ITER, LAGRANM, FUN,SQRT(DOTVEC(R2MR1,R2MR1,3)),
    !     &                 SQRT(DOTVEC(R3MR2,R3MR2,3))
    !...##IF PARALLEL
    !      ENDIF
    !...##ENDIF

    IF(ABS(FUN).GT.RSMALL) THEN
       A=DOTVEC(R2MR1,PP2MPP1,3)/SQRT(DOTVEC(R2MR1,R2MR1,3))
       B=DOTVEC(R3MR2,PP3MPP2,3)/SQRT(DOTVEC(R3MR2,R3MR2,3))
       DFUNDL=A-B
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

    XCORD(1)=XCORD(1)+LAGRANM*PP1(1)
    YCORD(1)=YCORD(1)+LAGRANM*PP1(2)
    ZCORD(1)=ZCORD(1)+LAGRANM*PP1(3)
    XCORD(2)=XCORD(2)+LAGRANM*PP2(1)
    YCORD(2)=YCORD(2)+LAGRANM*PP2(2)
    ZCORD(2)=ZCORD(2)+LAGRANM*PP2(3)
    XCORD(3)=XCORD(3)+LAGRANM*PP3(1)
    YCORD(3)=YCORD(3)+LAGRANM*PP3(2)
    ZCORD(3)=ZCORD(3)+LAGRANM*PP3(3)

    R2MR1r(1)=XCORD(2)-XCORD(1)
    R2MR1r(2)=YCORD(2)-YCORD(1)
    R2MR1r(3)=ZCORD(2)-ZCORD(1)
    R3MR2r(1)=XCORD(3)-XCORD(2)
    R3MR2r(2)=YCORD(3)-YCORD(2)
    R3MR2r(3)=ZCORD(3)-ZCORD(2)

    A=ONE/SQRT(DOTVEC(R2MR1r,R2MR1r,3))
    B=ONE/SQRT(DOTVEC(R3MR2r,R3MR2r,3))

    ZFAC=ONE*MASS(1)+TWO*(ONE+DOTVEC(R2MR1r,R3MR2r,3)*A*B)*MASS(2) &
         +ONE*MASS(3)

    ZFAC=SQRT(ZFAC)
    GFAC=ZERO

20  FORMAT(I5,4F16.6)
311 FORMAT(' ***** ERROR IN RXNCONS_1 ***** COORDINATE SETTING ', &
         ' WAS NOT ACCOMPLISHED IN',I6,' ITERATIONS',/)

    RETURN
  END SUBROUTINE RXNCONS_12

  SUBROUTINE RXNCONS_1F(XREF,YREF,ZREF,DX,DY,DZ,NATOM,IRXCNS, &
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
    ! . Local variables
    !
    INTEGER I, J, IA, IB, IC
    real(chm_real) R2MR1(3),R3MR2(3),A,B,RXTAN(3,NRXATM),SUM,FD

    IF(IRXCNS.EQ.1) THEN
       IA=IRXATM(1)
       IB=IRXATM(2)
       IC=IRXATM(3)
       R2MR1(1)=XREF(IB)-XREF(IA)
       R2MR1(2)=YREF(IB)-YREF(IA)
       R2MR1(3)=ZREF(IB)-ZREF(IA)
       R3MR2(1)=XREF(IC)-XREF(IB)
       R3MR2(2)=YREF(IC)-YREF(IB)
       R3MR2(3)=ZREF(IC)-ZREF(IB)
       A=ONE/SQRT(DOTVEC(R2MR1,R2MR1,3))
       B=ONE/SQRT(DOTVEC(R3MR2,R3MR2,3))
       DO I=1,3
          R2MR1(I)=R2MR1(I)*A
          R3MR2(I)=R3MR2(I)*B
       ENDDO
       DO I=1,3
          RXTAN(I,1)=-R2MR1(I)
          RXTAN(I,2)=R2MR1(I)+R3MR2(I)
          RXTAN(I,3)=-R3MR2(I)
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
    ENDIF
    RETURN
  END SUBROUTINE RXNCONS_1F
#endif /* (rxcons_1)*/

end module rxcons1

