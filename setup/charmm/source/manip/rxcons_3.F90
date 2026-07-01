module rxcons3
  use chm_kinds
  use dimens_fcm
  implicit none
contains
  !
#if KEY_RXNCONS==0 /*rxcons_3*/
  SUBROUTINE RX_3_SETUP(COMLYN,COMLEN)
    INTEGER COMLEN
    CHARACTER(len=*) COMLYN
    return
  end SUBROUTINE RX_3_SETUP
#else /* (rxcons_3)*/
  SUBROUTINE RX_3_SETUP(COMLYN,COMLEN)
    !
    use exfunc
    use number
    use coord
    use coordc
    use memory
    use shake
    use psf
    use stream
    !
    use rxncons
    use rxnconswt
    use rxncons3
    INTEGER COMLEN
    CHARACTER(len=*) COMLYN

    ! . Local variables.
    INTEGER ISLCT, JSLCT
    LOGICAL ERR
    !
    ! . set up counter
    !
    IF(.NOT.LCLEAN) THEN
       call chmalloc('rxcons_3.src','RX_3_SETUP','IRXATM',NRXATM,intg=IRXATM)
       call chmalloc('rxcons_3.src','RX_3_SETUP','XCORD',NRXATM,crl=XCORD)
       call chmalloc('rxcons_3.src','RX_3_SETUP','YCORD',NRXATM,crl=YCORD)
       call chmalloc('rxcons_3.src','RX_3_SETUP','ZCORD',NRXATM,crl=ZCORD)
       call chmalloc('rxcons_3.src','RX_3_SETUP','XREFC',NRXATM,crl=XREFC)
       call chmalloc('rxcons_3.src','RX_3_SETUP','YREFC',NRXATM,crl=YREFC)
       call chmalloc('rxcons_3.src','RX_3_SETUP','ZREFC',NRXATM,crl=ZREFC)
       call chmalloc('rxcons_3.src','RX_3_SETUP','XFORC',NRXATM,crl=XFORC)
       call chmalloc('rxcons_3.src','RX_3_SETUP','YFORC',NRXATM,crl=YFORC)
       call chmalloc('rxcons_3.src','RX_3_SETUP','ZFORC',NRXATM,crl=ZFORC)
       call chmalloc('rxcons_3.src','RX_3_SETUP','XREFM',NRXATM,crl=XREFM)
       call chmalloc('rxcons_3.src','RX_3_SETUP','YREFM',NRXATM,crl=YREFM)
       call chmalloc('rxcons_3.src','RX_3_SETUP','ZREFM',NRXATM,crl=ZREFM)
       call chmalloc('rxcons_3.src','RX_3_SETUP','XREFA',NRXATM,crl=XREFA)
       call chmalloc('rxcons_3.src','RX_3_SETUP','YREFA',NRXATM,crl=YREFA)
       call chmalloc('rxcons_3.src','RX_3_SETUP','ZREFA',NRXATM,crl=ZREFA)
       call chmalloc('rxcons_3.src','RX_3_SETUP','XREFB',NRXATM,crl=XREFB)
       call chmalloc('rxcons_3.src','RX_3_SETUP','YREFB',NRXATM,crl=YREFB)
       call chmalloc('rxcons_3.src','RX_3_SETUP','ZREFB',NRXATM,crl=ZREFB)
       call chmalloc('rxcons_3.src','RX_3_SETUP','XTAN',NRXATM,crl=XTAN)
       call chmalloc('rxcons_3.src','RX_3_SETUP','YTAN',NRXATM,crl=YTAN)
       call chmalloc('rxcons_3.src','RX_3_SETUP','ZTAN',NRXATM,crl=ZTAN)
       call chmalloc('rxcons_3.src','RX_3_SETUP','XCURV',NRXATM,crl=XCURV)
       call chmalloc('rxcons_3.src','RX_3_SETUP','YCURV',NRXATM,crl=YCURV)
       call chmalloc('rxcons_3.src','RX_3_SETUP','ZCURV',NRXATM,crl=ZCURV)
       call chmalloc('rxcons_3.src','RX_3_SETUP','WMASS',NRXATM,crl=WMASS)
       call chmalloc('rxcons_3.src','RX_3_SETUP','ATMASS',NRXATM,crl=ATMASS)

       CALL RXNSET_3(NRXATM,LCALRX,VRXNCS,IRXATM, &
            QCNOTRN,QCNOROT,QCMASS,QCWEIG,QCWCOMP, &
#if KEY_COMP2==1
            QCWCOMP2,  &                                 
#endif
            LRXPRN,XCORD,YCORD,ZCORD, &
            XREFM,YREFM,ZREFM, &
            XREFA,YREFA,ZREFA, &
            XREFB,YREFB,ZREFB, &
            XTAN,YTAN,ZTAN, &
            WMASS,ATMASS,LPRID)

       IF(IRXCNS.GT.0) LRXCNS=.TRUE.
       !
       IF(LRXCNS) QHOLO=.TRUE.

    ELSE ! LCLEAN

       call chmdealloc('rxcons_3.src','RX_3_SETUP','IRXATM',NRXATM,intg=IRXATM)
       call chmdealloc('rxcons_3.src','RX_3_SETUP','XCORD',NRXATM,crl=XCORD)
       call chmdealloc('rxcons_3.src','RX_3_SETUP','YCORD',NRXATM,crl=YCORD)
       call chmdealloc('rxcons_3.src','RX_3_SETUP','ZCORD',NRXATM,crl=ZCORD)
       call chmdealloc('rxcons_3.src','RX_3_SETUP','XREFC',NRXATM,crl=XREFC)
       call chmdealloc('rxcons_3.src','RX_3_SETUP','YREFC',NRXATM,crl=YREFC)
       call chmdealloc('rxcons_3.src','RX_3_SETUP','ZREFC',NRXATM,crl=ZREFC)
       call chmdealloc('rxcons_3.src','RX_3_SETUP','XFORC',NRXATM,crl=XFORC)
       call chmdealloc('rxcons_3.src','RX_3_SETUP','YFORC',NRXATM,crl=YFORC)
       call chmdealloc('rxcons_3.src','RX_3_SETUP','ZFORC',NRXATM,crl=ZFORC)
       call chmdealloc('rxcons_3.src','RX_3_SETUP','XREFM',NRXATM,crl=XREFM)
       call chmdealloc('rxcons_3.src','RX_3_SETUP','YREFM',NRXATM,crl=YREFM)
       call chmdealloc('rxcons_3.src','RX_3_SETUP','ZREFM',NRXATM,crl=ZREFM)
       call chmdealloc('rxcons_3.src','RX_3_SETUP','XREFA',NRXATM,crl=XREFA)
       call chmdealloc('rxcons_3.src','RX_3_SETUP','YREFA',NRXATM,crl=YREFA)
       call chmdealloc('rxcons_3.src','RX_3_SETUP','ZREFA',NRXATM,crl=ZREFA)
       call chmdealloc('rxcons_3.src','RX_3_SETUP','XREFB',NRXATM,crl=XREFB)
       call chmdealloc('rxcons_3.src','RX_3_SETUP','YREFB',NRXATM,crl=YREFB)
       call chmdealloc('rxcons_3.src','RX_3_SETUP','ZREFB',NRXATM,crl=ZREFB)
       call chmdealloc('rxcons_3.src','RX_3_SETUP','XTAN',NRXATM,crl=XTAN)
       call chmdealloc('rxcons_3.src','RX_3_SETUP','YTAN',NRXATM,crl=YTAN)
       call chmdealloc('rxcons_3.src','RX_3_SETUP','ZTAN',NRXATM,crl=ZTAN)
       call chmdealloc('rxcons_3.src','RX_3_SETUP','XCURV',NRXATM,crl=XCURV)
       call chmdealloc('rxcons_3.src','RX_3_SETUP','YCURV',NRXATM,crl=YCURV)
       call chmdealloc('rxcons_3.src','RX_3_SETUP','ZCURV',NRXATM,crl=ZCURV)
       call chmdealloc('rxcons_3.src','RX_3_SETUP','WMASS',NRXATM,crl=WMASS)
       call chmdealloc('rxcons_3.src','RX_3_SETUP','ATMASS',NRXATM,crl=ATMASS)


    ENDIF

    RETURN
  END SUBROUTINE RX_3_SETUP

  SUBROUTINE RXNSET_3(NRXATM,QCALRX,VRXNCS,IRXATM, &
       QCNOTRN,QCNOROT,QCMASS,QCWEIG,QCWCOMP, &
#if KEY_COMP2==1
       QCWCOMP2,  & 
#endif
       LRXPRN,XCORD,YCORD,ZCORD,XREFM,YREFM,ZREFM, &
       XREFA,YREFA,ZREFA,XREFB,YREFB,ZREFB, &
       XTAN,YTAN,ZTAN, &
       WMASS,ATMASS,LPRID)
    !
    use exfunc
    use number
    use psf
    use stream
    use coord
    use coordc
    use cnst_fcm
    use lonepr
    use memory
    !
    ! . Parsed variables
    !
    integer,allocatable,dimension(:) :: ATMPR
    real(chm_real),allocatable,dimension(:) :: RA
    real(chm_real),allocatable,dimension(:) :: RB
    INTEGER IRXATM(*),NRXATM,LPRID(*)
    real(chm_real) VRXNCS
    real(chm_real) XCORD(*),YCORD(*),ZCORD(*)
    real(chm_real) XREFM(*),YREFM(*),ZREFM(*)
    real(chm_real) XREFA(*),YREFA(*),ZREFA(*)
    real(chm_real) XREFB(*),YREFB(*),ZREFB(*)
    real(chm_real) XTAN(*), YTAN(*), ZTAN(*)
    real(chm_real) WMASS(*),ATMASS(*)
    LOGICAL QCALRX,QCNOTRN,QCNOROT,QCMASS,QCWEIG,QCWCOMP,LRXPRN
#if KEY_COMP2==1
    LOGICAL QCWCOMP2
#endif 
    !
    ! . Local variables
    !
    INTEGER I,J,JJ,ITEMP
    INTEGER N,ILP,IPT,JPT,JH,K
    real(chm_real) XD,YD,ZD
    real(chm_real) TEMP,DRDTTAN,DRDTNRM,LAMDA,D1,D2,DTDR

    if (prnlev >= 2) WRITE(OUTU,24) 'The plane constraint is selected'
    if (prnlev >= 2) WRITE(OUTU,25) 'number of atoms defining the rxn: ', NRXATM

    WMASS (1:NRXATM)=zero
    ATMASS(1:NRXATM)=zero

    DO I=1,NATOM
       LPRID(I)=0
    ENDDO

    IF(QCWEIG)THEN
       J=0
       DO I=1,NATOM
          IF(WMAIN(I).GT.ZERO)THEN
             J=J+1
             IRXATM(J)=I
             WMASS(J)=WMAIN(I)
             IF(IMOVE(I).NE.-1)THEN
                ATMASS(J)=ONE/AMASS(I)
             ELSE
                ATMASS(J)=ONE/WMASS(J)
             ENDIF
             IF(QCMASS)WMASS(J)=WMASS(J)*AMASS(I)
          ENDIF
       ENDDO
    ELSEIF(QCWCOMP)THEN
       J=0
       DO I=1,NATOM
          IF(WCOMP(I).GT.ZERO)THEN
             J=J+1
             IRXATM(J)=I
             WMASS(J)=WCOMP(I)
             IF(IMOVE(I).NE.-1)THEN
                ATMASS(J)=ONE/AMASS(I)
             ELSE
                ATMASS(J)=ONE/WMASS(J)
             ENDIF
             IF(QCMASS)WMASS(J)=WMASS(J)*AMASS(I)
          ENDIF
       ENDDO
#if KEY_COMP2==1
    ELSEIF(QCWCOMP2)THEN
       J=0
       DO I=1,NATOM
          IF(WCOMP2(I).GT.ZERO)THEN
             J=J+1
             IRXATM(J)=I
             WMASS(J)=WCOMP2(I)
             IF(IMOVE(I).NE.-1)THEN
                ATMASS(J)=ONE/AMASS(I)
             ELSE
                ATMASS(J)=ONE/WMASS(J)
             ENDIF
             IF(QCMASS)WMASS(J)=WMASS(J)*AMASS(I)
          ENDIF
       ENDDO
#endif 
    ELSE
       J=0
       DO I=1,NATOM
          J=J+1
          IRXATM(J)=I
          WMASS(J)=ONE
          IF(IMOVE(I).NE.-1)THEN
             ATMASS(J)=ONE/AMASS(I)
          ELSE
             ATMASS(J)=ONE/WMASS(J)
          ENDIF
          IF(QCMASS)WMASS(J)=WMASS(J)*AMASS(I)
       ENDDO
    ENDIF

    IF(NUMLP.GT.0)THEN
       DO ILP=NUMLP,1,-1
          N=LPNHOST(ILP)
          IF(N.LE.3)CALL WRNDIE(-1,'<RXNSET_3>', &
               '  plane cons only allows cent type of lonepares')

          IPT=LPHPTR(ILP)
          I=LPHOST(IPT)

          LPRID(I)=ILP
       ENDDO

       DO J=1,NRXATM
          I=IRXATM(J)
          IF(IMOVE(I).EQ.-1)THEN
             ILP=LPRID(I)
             N=LPNHOST(ILP)
             IPT=LPHPTR(ILP)
             IF(WMASS(J).GT.ZERO)THEN
                DO K=1,N
                   JH=LPHOST(IPT+K)
                   DO JJ=1,NRXATM
                      IF(JH.EQ.IRXATM(JJ).AND.JH.NE.I)THEN
                         IF(WMASS(JH).GT.ZERO) THEN
                            CALL WRNDIE(-1,'<RXNSET_3>', &
                                 'lonepair and hosts cannot both have non-zero weight')
                         ENDIF
                      ENDIF
                   ENDDO
                ENDDO ! K=1,N
             ENDIF
          ENDIF
       ENDDO ! J=1,NRXATM
    ENDIF
    !
    ! . Test printing
    !
    IF(LRXPRN)THEN
       WRITE(OUTU,24)'Index of selected atoms'
       DO I=1,NRXATM
          WRITE(OUTU,10) I, IRXATM(I)
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
       XREFM(I)=X(ITEMP)
       YREFM(I)=Y(ITEMP)
       ZREFM(I)=Z(ITEMP)
       XREFA(I)=XCOMP(ITEMP)
       YREFA(I)=YCOMP(ITEMP)
       ZREFA(I)=ZCOMP(ITEMP)
       XREFB(I)=REFX(ITEMP)
       YREFB(I)=REFY(ITEMP)
       ZREFB(I)=REFZ(ITEMP)
    ENDDO
    ! Test printing
    IF(LRXPRN) THEN
       WRITE(OUTU,24) 'Current coordinate'
       DO I=1,NRXATM
          WRITE(OUTU,11) I, IRXATM(I), XCORD(I), YCORD(I), ZCORD(I)
       ENDDO
       WRITE(OUTU,24) 'Coordinate of pathreference, I'
       DO I=1,NRXATM
          WRITE(OUTU,11) I, IRXATM(I), XREFM(I), YREFM(I), ZREFM(I)
       ENDDO
       WRITE(OUTU,24) 'Coordinate of pathreference, I-1'
       DO I=1,NRXATM
          WRITE(OUTU,11) I, IRXATM(I), XREFA(I), YREFA(I), ZREFA(I)
       ENDDO
       WRITE(OUTU,24) 'Coordinate of pathreference, I+1'
       DO I=1,NRXATM
          WRITE(OUTU,11) I, IRXATM(I), XREFB(I), YREFB(I), ZREFB(I)
       ENDDO
    ENDIF

    call chmalloc('rxcons_3.src','RXNSET_3','ATMPR',2*NRXATM,intg=ATMPR)
    call chmalloc('rxcons_3.src','RXNSET_3','RA',3*NRXATM,crl=RA)
    call chmalloc('rxcons_3.src','RXNSET_3','RB',3*NRXATM,crl=RB)

    CALL CALCTAU(NRXATM,XREFM,YREFM,ZREFM,XREFA,YREFA,ZREFA, &
         XREFB,YREFB,ZREFB,XTAN,YTAN,ZTAN,WMASS, &
         QCNOTRN,QCNOROT,VRXNCS, &
         ATMPR,RA,RB,.TRUE.,DTDR)

    DO I=1,NRXATM
       XCORD(I)=XREFM(I)
       YCORD(I)=YREFM(I)
       ZCORD(I)=ZREFM(I)

       ITEMP=IRXATM(I)
       IF(IMOVE(ITEMP).NE.-1)THEN
          X(ITEMP)=XCORD(I)
          Y(ITEMP)=YCORD(I)
          Z(ITEMP)=ZCORD(I)
       ELSE
          ILP=LPRID(ITEMP)

          N=LPNHOST(ILP)
          IPT=LPHPTR(ILP)

          XD=XCORD(I)-X(ITEMP)
          YD=YCORD(I)-Y(ITEMP)
          ZD=ZCORD(I)-Z(ITEMP)

          X(ITEMP)=XCORD(I)
          Y(ITEMP)=YCORD(I)
          Z(ITEMP)=ZCORD(I)

          DO K=1,N
             JPT=LPHOST(IPT+K)
             X(JPT)=X(JPT)+XD
             Y(JPT)=Y(JPT)+YD
             Z(JPT)=Z(JPT)+ZD
          ENDDO
       ENDIF
    ENDDO

    IF(LRXPRN) THEN
       WRITE(OUTU,24) 'Corrected Current coordinate'
       DO I=1,NRXATM
          WRITE(OUTU,11) I, IRXATM(I), XCORD(I), YCORD(I), ZCORD(I)
       ENDDO
       WRITE(OUTU,24) 'Corrected Coordinate of path reference, I'
       DO I=1,NRXATM
          WRITE(OUTU,11) I, IRXATM(I), XREFM(I), YREFM(I), ZREFM(I)
       ENDDO
    ENDIF

    call chmdealloc('rxcons_3.src','RXNSET_3','ATMPR',2*NRXATM,intg=ATMPR)
    call chmdealloc('rxcons_3.src','RXNSET_3','RA',3*NRXATM,crl=RA)
    call chmdealloc('rxcons_3.src','RXNSET_3','RB',3*NRXATM,crl=RB)

24  FORMAT('RXNCONS> ',A)
25  FORMAT('RXNCONS> ',A,I5)
10  FORMAT(2I5)
11  FORMAT(2I5,3F16.6)
12  FORMAT(2F16.6)
    RETURN
  END SUBROUTINE RXNSET_3

  SUBROUTINE CALCTAU(NRXATM,XREFM,YREFM,ZREFM,XREFA,YREFA,ZREFA, &
       XREFB,YREFB,ZREFB,XTAN,YTAN,ZTAN,WMASS,QCNOTRN, &
       QCNOROT,VRXNCS,ATMPR,RA,RB,QEQDIS,DTDR)

    use exfunc
    use number
    use psf
    use stream

    real(chm_real) XREFM(*),YREFM(*),ZREFM(*), &
         XREFA(*),YREFA(*),ZREFA(*)
    real(chm_real) XREFB(*),YREFB(*),ZREFB(*),XTAN(*),YTAN(*),ZTAN(*)
    real(chm_real) WMASS(*),VRXNCS
    INTEGER NRXATM
    INTEGER ATMPR(2,NRXATM)
    real(chm_real) RA(3,NRXATM),RB(3,NRXATM),DTDR
    LOGICAL QCNOROT,QCNOTRN,QEQDIS

    INTEGER I,J,K
    real(chm_real) WTOT,DISTAN,DEVA(3,3),EVWID
    real(chm_real) SUM,SUM1,ROTRB,DIFF,A,B,TEMP,FACT,D1,D2
    LOGICAL LPRINT,LPRINT1

    EVWID = ZERO ! never set, if you know what it should be plz update

    LPRINT=(PRNLEV.GT.5)
    LPRINT1=(PRNLEV.GT.9)

    DO I=1,NRXATM
       ATMPR(1,I)=I
       ATMPR(2,I)=I
    ENDDO

    CALL ECBSTF(XREFM,YREFM,ZREFM,XREFB,YREFB,ZREFB,ATMPR,NRXATM, &
         QCNOROT,QCNOTRN,LPRINT1, (/ ZERO /), .FALSE.,WMASS, &
         WTOT,DISTAN,RA,RB,DEVA,EVWID)

    TEMP=ONE/WTOT
    D2=SQRT(DISTAN*TEMP)

    IF(D2.LT.TENM5) THEN
       CALL WRNDIE(-2,'<CALCTAU>','zero tangent length i+1 - i')
    ENDIF

    IF(LPRINT) THEN
       WRITE(OUTU,68) 'I+1 - I', D2
    ENDIF

    !      WRITE(OUTU,68) 'I+1 - I', D2

    DO K=1,NRXATM
       FACT=WMASS(K)*TEMP*HALF
       ROTRB=ZERO
       DO J=1,3
          ROTRB=ROTRB+DEVA(1,J)*RB(J,K)
       ENDDO
       XTAN(K)=(ROTRB-RA(1,K))*FACT ! jwchu 12162002

       ROTRB=ZERO
       DO J=1,3
          ROTRB=ROTRB+DEVA(2,J)*RB(J,K)
       ENDDO
       YTAN(K)=(ROTRB-RA(2,K))*FACT ! jwchu 12162002

       ROTRB=ZERO
       DO J=1,3
          ROTRB=ROTRB+DEVA(3,J)*RB(J,K)
       ENDDO
       ZTAN(K)=(ROTRB-RA(3,K))*FACT ! jwchu 12162002
    ENDDO

    CALL ECBSTF(XREFM,YREFM,ZREFM,XREFA,YREFA,ZREFA,ATMPR,NRXATM, &
         QCNOROT,QCNOTRN,LPRINT1, (/ ZERO /), .FALSE.,WMASS, &
         WTOT,DISTAN,RA,RB,DEVA,EVWID)

    D1=SQRT(DISTAN*TEMP)
    IF(D1.LT.TENM5) THEN
       CALL WRNDIE(-2,'<CALCTAU>','zero tangent length i - i-1')
    ENDIF
    IF(LPRINT) THEN
       WRITE(OUTU,68) 'I - I-1', D1
    ENDIF

    !      WRITE(OUTU,68) 'I - I-1', D1

    VRXNCS=HALF*(D1-D2)

    A=ONE/D1
    B=ONE/D2
    DO K=1,NRXATM
       FACT=WMASS(K)*TEMP*HALF
       ROTRB=ZERO
       DO J=1,3
          ROTRB=ROTRB+DEVA(1,J)*RB(J,K)
       ENDDO
       XTAN(K)=(RA(1,K)-ROTRB)*FACT*A+XTAN(K)*B ! jwchu 12162002

       ROTRB=ZERO
       DO J=1,3
          ROTRB=ROTRB+DEVA(2,J)*RB(J,K)
       ENDDO
       YTAN(K)=(RA(2,K)-ROTRB)*FACT*A+YTAN(K)*B ! jwchu 12162002

       ROTRB=ZERO
       DO J=1,3
          ROTRB=ROTRB+DEVA(3,J)*RB(J,K)
       ENDDO
       ZTAN(K)=(RA(3,K)-ROTRB)*FACT*A+ZTAN(K)*B ! jwchu 12162002
    ENDDO

    SUM=ZERO
    DO I=1,NRXATM
       SUM=SUM+XTAN(I)*XTAN(I)+ &
            YTAN(I)*YTAN(I)+ &
            ZTAN(I)*ZTAN(I)
    ENDDO

    SUM=SQRT(SUM)
    DTDR=SUM
    SUM=ONE/SUM

    DO I=1,NRXATM
       XTAN(I)=XTAN(I)*SUM
       YTAN(I)=YTAN(I)*SUM
       ZTAN(I)=ZTAN(I)*SUM
    ENDDO

    IF(QEQDIS)THEN
       IF(ABS(VRXNCS).GT.TENM8) THEN
          CALL EQDIS(NRXATM,XREFM,YREFM,ZREFM,XREFA,YREFA,ZREFA, &
               XREFB,YREFB,ZREFB,XTAN,YTAN,ZTAN, &
               ATMPR,QCNOROT,QCNOTRN, &
               WMASS,WTOT,DISTAN,RA,RB,DEVA,EVWID,VRXNCS,DTDR)
       ENDIF
    ENDIF

58  FORMAT('CALCTAU> ',3F14.5)
68  FORMAT('CALCTAU> ',A,2F14.5)
78  FORMAT(I5)
88  FORMAT('CALCTAU> ',A)
98  FORMAT('CALCTAU> ',1F14.5)

    RETURN
  END SUBROUTINE CALCTAU

  SUBROUTINE EQDIS(NRXATM,XREFM,YREFM,ZREFM,XREFA,YREFA,ZREFA, &
       XREFB,YREFB,ZREFB,XTAN,YTAN,ZTAN, &
       ATMPR,QCNOROT,QCNOTRN, &
       WMASS,WTOT,DISTAN,RA,RB,DEVA,EVWID,DIFF,DTDR)
    use vector
    use number
    use psf
    use stream

    real(chm_real) XREFM(*),YREFM(*),ZREFM(*), &
         XREFA(*),YREFA(*),ZREFA(*)
    real(chm_real) XREFB(*),YREFB(*),ZREFB(*),XTAN(*),YTAN(*),ZTAN(*)
    real(chm_real) WMASS(*)
    INTEGER NRXATM
    INTEGER ATMPR(2,NRXATM)
    real(chm_real) RA(3,NRXATM),RB(3,NRXATM),DIFF,DTDR
    LOGICAL QCNOROT,QCNOTRN

    INTEGER I,J,K,ITER
    real(chm_real) WTOT,DISTAN,DEVA(3,3),EVWID
    real(chm_real) SUM,SUM1,ROTRB,D1,D2,LAMDA,DFDL,A,B,LOLD,TEMP,FACT
    LOGICAL LPRINT,LPRINT1

    LPRINT=(PRNLEV.GT.5)
    LPRINT1=(PRNLEV.GT.9)

    LAMDA=ZERO
    LOLD=ZERO
    ITER=0

1111 CONTINUE

    DFDL=DOTVEC(XTAN,XTAN,NRXATM)+DOTVEC(YTAN,YTAN,NRXATM)+ &
         DOTVEC(ZTAN,ZTAN,NRXATM)

    LAMDA=-DIFF/DFDL
    !      LAMDA=-DIFF*HALF

    DO I=1,NRXATM
       XREFM(I)=XREFM(I)+LAMDA*XTAN(I)
       YREFM(I)=YREFM(I)+LAMDA*YTAN(I)
       ZREFM(I)=ZREFM(I)+LAMDA*ZTAN(I)
    ENDDO

    CALL ECBSTF(XREFM,YREFM,ZREFM,XREFB,YREFB,ZREFB,ATMPR,NRXATM, &
         QCNOROT,QCNOTRN,LPRINT1, (/ ZERO /), .FALSE.,WMASS, &
         WTOT,DISTAN,RA,RB,DEVA,EVWID)

    TEMP=ONE/WTOT

    D2=SQRT(DISTAN/WTOT)
    DO K=1,NRXATM
       FACT=TEMP*WMASS(K)*HALF
       ROTRB=ZERO
       DO J=1,3
          ROTRB=ROTRB+DEVA(1,J)*RB(J,K)
       ENDDO
       XTAN(K)=(ROTRB-RA(1,K))*FACT ! jwchu 12162002

       ROTRB=ZERO
       DO J=1,3
          ROTRB=ROTRB+DEVA(2,J)*RB(J,K)
       ENDDO
       YTAN(K)=(ROTRB-RA(2,K))*FACT ! jwchu 12162002

       ROTRB=ZERO
       DO J=1,3
          ROTRB=ROTRB+DEVA(3,J)*RB(J,K)
       ENDDO
       ZTAN(K)=(ROTRB-RA(3,K))*FACT ! jwchu 12162002
    ENDDO
    CALL ECBSTF(XREFM,YREFM,ZREFM,XREFA,YREFA,ZREFA,ATMPR,NRXATM, &
         QCNOROT,QCNOTRN,LPRINT1, (/ ZERO /), .FALSE.,WMASS, &
         WTOT,DISTAN,RA,RB,DEVA,EVWID)

    D1=SQRT(DISTAN/WTOT)

    DIFF=(D1-D2)*HALF
    ITER=ITER+1

    IF(LPRINT) WRITE(OUTU,18) 'Iteration: ',ITER,D1,D2,DIFF,DFDL,LAMDA
    !      WRITE(OUTU,18) 'Iteration: ',ITER,D1,D2,DIFF,DFDL,LAMDA

    IF(ABS(DIFF).GT.TENM8) THEN
       LOLD=LAMDA
       A=ONE/D1
       B=ONE/D2
       DO K=1,NRXATM
          FACT=WMASS(K)*TEMP*HALF
          ROTRB=ZERO
          DO J=1,3
             ROTRB=ROTRB+DEVA(1,J)*RB(J,K)
          ENDDO
          XTAN(K)=(RA(1,K)-ROTRB)*FACT*A+XTAN(K)*B ! jwchu 12162002

          ROTRB=ZERO
          DO J=1,3
             ROTRB=ROTRB+DEVA(2,J)*RB(J,K)
          ENDDO
          YTAN(K)=(RA(2,K)-ROTRB)*FACT*A+YTAN(K)*B ! jwchu 12162002

          ROTRB=ZERO
          DO J=1,3
             ROTRB=ROTRB+DEVA(3,J)*RB(J,K)
          ENDDO
          ZTAN(K)=(RA(3,K)-ROTRB)*FACT*A+ZTAN(K)*B ! jwchu 12162002
       ENDDO
       IF(ITER.GT.500) THEN
          CALL WRNDIE(-3,'<EQDIS>', &
               'Distance between i,i-1 and i+1,i can not be made equal')
       ENDIF
       GOTO 1111
    ENDIF

18  FORMAT('EQDIS> ',A,I5,5E16.6)
    RETURN
  END SUBROUTINE EQDIS

  SUBROUTINE RXNCONS_3(X,Y,Z,XREF,YREF,ZREF,AMASS,NATOM,IRXCNS, &
       NRXATM,LRXCNS,LDYNA,VRXNCS,LAGRANM,FRCONS,MXITER, &
       PLAGRANM,LRXPRN,IRXATM, &
       XCORD,YCORD,ZCORD,XREFC,YREFC,ZREFC,DTSQ, &
       ZFAC,GFAC,RCNSDL,RCNSDS,RCNSDSM,RCNSDC, &
       RCNSDCM,RCNSDR,RCNSDRC,RCNSAMF,IMOVE,NUMLP, &
       LPNHOST,LPHPTR,LPHOST,LPVALUE,LPWGHT,LPRID)

    !
    use exfunc
    use number
    use deriv
    use stream
    use parallel
    use machutil,only:eclock
    !
    ! . Parsed variables
    real(chm_real) X(*),Y(*),Z(*),XREF(*),YREF(*),ZREF(*)
    real(chm_real) AMASS(*)
    INTEGER NATOM,IRXCNS,NRXATM,IRXATM(*),MXITER
    LOGICAL LRXCNS,LDYNA,PLAGRANM,LRXPRN
    real(chm_real) VRXNCS,LAGRANM,FRCONS,XCORD(*),YCORD(*),ZCORD(*)
    real(chm_real) XREFC(*),YREFC(*),ZREFC(*),DTSQ,ZFAC,GFAC, &
         RCNSDL,RCNSDR,RCNSDS,RCNSDSM,RCNSDC,RCNSDCM,RCNSDRC,RCNSAMF
    INTEGER LPRID(*)
    INTEGER IMOVE(*),NUMLP,LPNHOST(*),LPHPTR(*),LPHOST(*)
    real(chm_real) LPVALUE(3,*)
    LOGICAL LPWGHT(*)
    !
#if KEY_PARALLEL==1
    real(chm_real) GCARR(10),TIMMER
#endif 
    !
    ! . Local variables
    !
    INTEGER I, IA
    INTEGER ATFRST,ATLAST
    INTEGER N,ILP,IPT,JPT,JH,K
    real(chm_real) AM,MTOT,XD,YD,ZD

    real(chm_real) sum1,sum2,sumx,sumy,sumz,sum3,sum4
    real(chm_real) sum3x,sum3y,sum3z

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
    !        WRITE(OUTU,*)' Hi ', MYNODP, ATFRST, ATLAST
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
    !      DO I=1,NRXATM
    !        IA=IRXATM(I)
    !        IF(IMOVE(IA).EQ.-1)THEN
    !          ILP=LPRID(IA)
    !
    !         N=LPNHOST(ILP)
    !          IPT=LPHPTR(ILP)
    !
    !          XD=X(IA)-XREF(IA)
    !          YD=Y(IA)-YREF(IA)
    !          ZD=Z(IA)-ZREF(IA)
    !
    !          sum1=xd*xd+yd*yd+zd*zd
    !          sum1=sqrt(sum1)
    !
    !          sum2=zero
    !          sum3=zero
    !          sumx=zero
    !          sumy=zero
    !          sumz=zero
    !
    !          sum3x=zero
    !          sum3y=zero
    !          sum3z=zero
    !
    !          DO K=1,N
    !
    !           JPT=LPHOST(IPT+K)
    !            XD=X(JPT)-XREF(JPT)
    !            YD=Y(JPT)-YREF(JPT)
    !            ZD=Z(JPT)-ZREF(JPT)
    !            sum2=sum2+(XD*XD+YD*YD+ZD*ZD)
    !
    !            sumx=sumx+dx(jpt)
    !            sumy=sumy+dy(jpt)
    !            sumz=sumz+dz(jpt)
    !
    !            sum3x=sum3x+dx(jpt)*dx(jpt)
    !            sum3y=sum3y+dy(jpt)*dy(jpt)
    !            sum3z=sum3z+dz(jpt)*dz(jpt)
    !            sum3=sum3+sum3x+sum3y+sum3z
    !
    !          ENDDO
    !
    !          DO K=1,N
    !            JPT=LPHOST(IPT+K)
    !            write(*,'(A,2I5,6F12.4)')
    !     $      'forces ',IA,JPT,dx(jpt),dy(jpt),dz(jpt),sumx,sumy,sumz
    !          ENDDO
    !
    !          sum2=sqrt(sum2/dble(n))
    !          sum4=sqrt((sumx*sumx+sumy*sumy+sumz*sumz))
    !          sum3=sqrt(sum3/dble(n))
    !c          sum3=sqrt(sum3)
    !        ENDIF
    !        write(*,'(A,I5,6F12.4)')'summary ',
    !     $          IA,sum1,sum2,sum2/sum1,sum4,sum3,sum3/sum4
    !      ENDDO

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

    CALL RXNCONS_32(NATOM,NRXATM,VRXNCS,LAGRANM,FRCONS,  &
         MXITER,LRXPRN,IRXATM,                &
         XCORD,YCORD,ZCORD,XREFC,YREFC,ZREFC, &
         ZFAC,GFAC,                           &
         RCNSDL,RCNSDS,RCNSDSM,RCNSDC,RCNSDCM,&
         RCNSDR,RCNSDRC,RCNSAMF,              &
         DX,DY,DZ,LPRID)

    !      DO I=1,NRXATM
    !        IA=IRXATM(I)
    !        X(IA)=XCORD(I)
    !        Y(IA)=YCORD(I)
    !        Z(IA)=ZCORD(I)
    !        IF(MYNOD.EQ.0) THEN
    !          WRITE(OUTU,*)' after shake, X ',IA,X(IA),Y(IA),Z(IA)
    !          WRITE(OUTU,*)' after shake, XREF ',IA,XREF(IA),YREF(IA),
    !     &                 ZREF(IA)
    !        ENDIF
    !      ENDDO

    DO I=1,NRXATM
       IA=IRXATM(I)
       IF(IMOVE(IA).NE.-1)THEN
          X(IA)=XCORD(I)
          Y(IA)=YCORD(I)
          Z(IA)=ZCORD(I)
       ELSE
          ILP=LPRID(IA)

          N=LPNHOST(ILP)
          IPT=LPHPTR(ILP)

          XD=XCORD(I)-X(IA)
          YD=YCORD(I)-Y(IA)
          ZD=ZCORD(I)-Z(IA)

          X(IA)=XCORD(I)
          Y(IA)=YCORD(I)
          Z(IA)=ZCORD(I)

          DO K=1,N
             JPT=LPHOST(IPT+K)
             X(JPT)=X(JPT)+XD
             Y(JPT)=Y(JPT)+YD
             Z(JPT)=Z(JPT)+ZD
          ENDDO
       ENDIF
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
  END SUBROUTINE RXNCONS_3

  SUBROUTINE RXNCONS_32(NATOM,NRXATM,VRXNCS,LAGRANM,FRCONS,        &
       MXITER,LRXPRN,                             &
       IRXATM,XCORD,YCORD,ZCORD,XREFC,YREFC,ZREFC,&
       ZFAC,GFAC,RCNSDL,RCNSDS,RCNSDSM,RCNSDC,    &
       RCNSDCM,RCNSDR,RCNSDRC,RCNSAMF,            &
       DX,DY,DZ,LPRID)

    use exfunc
    use number
    use stream
    use memory
    use rxnconswt

    use rxncons3

    ! . Parsed variables
    !
    integer,allocatable,dimension(:) :: ATMPR
    real(chm_real),allocatable,dimension(:) :: RA
    real(chm_real),allocatable,dimension(:) :: RB
    real(chm_real),allocatable,dimension(:) :: XTMP
    real(chm_real),allocatable,dimension(:) :: YTMP
    real(chm_real),allocatable,dimension(:) :: ZTMP
    real(chm_real),allocatable,dimension(:) :: XTMP1
    real(chm_real),allocatable,dimension(:) :: YTMP1
    real(chm_real),allocatable,dimension(:) :: ZTMP1
    INTEGER NATOM,NRXATM,IRXATM(*),MXITER
    REAL(chm_real) VRXNCS,LAGRANM,FRCONS,XCORD(*),YCORD(*),ZCORD(*)
    REAL(chm_real) XREFC(*),YREFC(*),ZREFC(*),ZFAC,GFAC
    REAL(chm_real) RCNSDL,RCNSDS,RCNSDSM,RCNSDC,RCNSDCM,RCNSDR,RCNSDRC,RCNSAMF
    REAL(chm_real) DX(*),DY(*),DZ(*)
    INTEGER LPRID(*)
    LOGICAL LRXPRN
    !
    !
    ! . Local variables
    !
    real(chm_real) DRDTTAN,DRDTNRM,D1,D2,DTDR

    !
    ! Prepare force vector base on REFC
    !
    call chmalloc('rxcons_3.src','RXNCONS_32','ATMPR',2*NRXATM,intg=ATMPR)
    call chmalloc('rxcons_3.src','RXNCONS_32','RA',3*NRXATM,crl=RA)
    call chmalloc('rxcons_3.src','RXNCONS_32','RB',3*NRXATM,crl=RB)

    CALL CALCTAU(NRXATM,     XCORD ,     YCORD ,     ZCORD , &
         XREFA,YREFA,ZREFA, &
         XREFB,YREFB,ZREFB, &
         XTAN ,YTAN ,ZTAN , &
         WMASS,QCNOTRN,QCNOROT,VRXNCS, &
         ATMPR,RA,RB,.FALSE.,DTDR)

    call chmalloc('rxcons_3.src','RXNCONS_32','XTMP',NRXATM,crl=XTMP)
    call chmalloc('rxcons_3.src','RXNCONS_32','YTMP',NRXATM,crl=YTMP)
    call chmalloc('rxcons_3.src','RXNCONS_32','ZTMP',NRXATM,crl=ZTMP)
    call chmalloc('rxcons_3.src','RXNCONS_32','XTMP1',NRXATM,crl=XTMP1)
    call chmalloc('rxcons_3.src','RXNCONS_32','YTMP1',NRXATM,crl=YTMP1)
    call chmalloc('rxcons_3.src','RXNCONS_32','ZTMP1',NRXATM,crl=ZTMP1)

    LAGRANM=ZERO
    IF(ABS(VRXNCS).GT.RSMALL) THEN

       CALL CALCTAU(NRXATM,     XREFC ,     YREFC ,     ZREFC , &
            XREFA,YREFA,ZREFA, &
            XREFB,YREFB,ZREFB, &
            XTAN ,YTAN ,ZTAN , &
            WMASS,QCNOTRN,QCNOROT,VRXNCS, &
            ATMPR,RA,RB,.FALSE.,DTDR)

       CALL SHKRXN_3(NRXATM,XTAN,YTAN,ZTAN, &
            XCORD,YCORD,ZCORD,XREFC,YREFC,ZREFC, &
            XTMP, YTMP, ZTMP, &
            XTMP1,YTMP1,ZTMP1, &
            XREFA,YREFA,ZREFA, &
            XREFB,YREFB,ZREFB, &
            WMASS,QCNOTRN,QCNOROT,ATMASS,VRXNCS, &
            ATMPR,RA,RB,MXITER,LAGRANM, &
            D1,D2,DTDR)

    ENDIF

    CALL CALCTAU(NRXATM,     XREFC ,     YREFC ,     ZREFC ,  &
         XREFA,YREFA,ZREFA,&
         XREFB,YREFB,ZREFB,&
         XTAN ,YTAN ,ZTAN ,&
         WMASS,QCNOTRN,QCNOROT,VRXNCS, &
         ATMPR,RA,RB,.FALSE.,DTDR)

    FRCONS=ZERO
    CALL GETFORC(NRXATM,DX,DY,DZ,XFORC,YFORC,ZFORC, &
         XTAN,YTAN,ZTAN, &
         IRXATM,LPRID,FRCONS)


    CALL GETFACTORS(NRXATM,   XCORD,     YCORD,     ZCORD  , &
         XREFM,YREFM,ZREFM, &
         XREFA,YREFA,ZREFA, &
         XREFB,YREFB,ZREFB, &
         XTAN ,YTAN ,ZTAN , &
         XCURV,YCURV,ZCURV, &
         XTMP ,YTMP ,ZTMP , &
         XTMP1,YTMP1,ZTMP1, &
         WMASS,QCNOTRN,QCNOROT,VRXNCS, &
         ATMPR,RA,RB,ATMASS, &
         .FALSE.,D1,D2,ZFAC,GFAC,             &
         RCNSDL,RCNSDS,RCNSDSM,RCNSDC,RCNSDCM,&
         RCNSDR,RCNSDRC,RCNSAMF)

    call chmdealloc('rxcons_3.src','RXNCONS_32','XTMP',NRXATM,crl=XTMP)
    call chmdealloc('rxcons_3.src','RXNCONS_32','YTMP',NRXATM,crl=YTMP)
    call chmdealloc('rxcons_3.src','RXNCONS_32','ZTMP',NRXATM,crl=ZTMP)
    call chmdealloc('rxcons_3.src','RXNCONS_32','XTMP1',NRXATM,crl=XTMP1)
    call chmdealloc('rxcons_3.src','RXNCONS_32','YTMP1',NRXATM,crl=YTMP1)
    call chmdealloc('rxcons_3.src','RXNCONS_32','ZTMP1',NRXATM,crl=ZTMP1)

    call chmdealloc('rxcons_3.src','RXNCONS_32','ATMPR',2*NRXATM,intg=ATMPR)
    call chmdealloc('rxcons_3.src','RXNCONS_32','RA',3*NRXATM,crl=RA)
    call chmdealloc('rxcons_3.src','RXNCONS_32','RB',3*NRXATM,crl=RB)

    RETURN
  END SUBROUTINE RXNCONS_32

  SUBROUTINE SHKRXN_3(NRXATM,XTAN,YTAN,ZTAN,XN,YN,ZN,XO,YO,ZO, &
       XTMP,YTMP,ZTMP, &
       XTMP1,YTMP1,ZTMP1, &
       XREFA,YREFA,ZREFA, &
       XREFB,YREFB,ZREFB, &
       WMASS,QCNOTRN,QCNOROT,ATMASS,VRXNCS, &
       ATMPR,RA,RB,MXITER,LAGRANM,D1,D2,DTDR)

    use vector
    use number
    use stream
    !
    ! . Parsed variables
    !
    INTEGER NRXATM,MXITER,ATMPR(2,NRXATM)

    real(chm_real) VRXNCS,LAGRANM
    real(chm_real) XTAN(*),YTAN(*),ZTAN(*),XN(*),YN(*),ZN(*)
    real(chm_real) XO(*),YO(*),ZO(*)
    real(chm_real) XREFA(*),YREFA(*),ZREFA(*)
    real(chm_real) XREFB(*),YREFB(*),ZREFB(*)
    real(chm_real) XTMP(*),YTMP(*),ZTMP(*)
    real(chm_real) XTMP1(*),YTMP1(*),ZTMP1(*)
    real(chm_real) WMASS(*),ATMASS(*)
    real(chm_real) RA(3,NRXATM),RB(3,NRXATM),D1,D2,DTDR

    LOGICAL QCNOTRN,QCNOROT
    !
    !
    ! . Local variables
    !
    INTEGER I,J,K,ITER
    real(chm_real) WTOT,DISTAN,EVWID,DEVA(3,3)
    real(chm_real) ROTRB,FDTTAN,DFDL,LOLD,DTDR1
    LOGICAL LPRINT

    LPRINT=(PRNLEV.GT.5)
    ITER=0
    LAGRANM=ZERO
    LOLD=ZERO
    FDTTAN=ZERO
    DFDL=ZERO

    WTOT=ZERO
    DO I=1,NRXATM
       WTOT=WTOT+ONE/ATMASS(I)
    ENDDO

    DO I=1,NRXATM
       XTAN(I)=XTAN(I)*ATMASS(I)*DTDR*WTOT
       YTAN(I)=YTAN(I)*ATMASS(I)*DTDR*WTOT
       ZTAN(I)=ZTAN(I)*ATMASS(I)*DTDR*WTOT
    ENDDO

2222 CONTINUE

    ITER=ITER+1
    DO I=1,NRXATM
       XTMP(I)=XN(I)+LAGRANM*XTAN(I)
       YTMP(I)=YN(I)+LAGRANM*YTAN(I)
       ZTMP(I)=ZN(I)+LAGRANM*ZTAN(I)
    ENDDO

    CALL CALCTAU(NRXATM,XTMP,YTMP,ZTMP,XREFA,YREFA,ZREFA, &
         XREFB,YREFB,ZREFB,XTMP1,YTMP1,ZTMP1,WMASS, &
         QCNOTRN,QCNOROT,VRXNCS, &
         ATMPR,RA,RB,.FALSE.,DTDR1)

    IF(LPRINT)WRITE(OUTU,18) &
         ' Iteration: ',ITER,DTDR,LAGRANM,VRXNCS,DFDL

    !      WRITE(OUTU,18)
    !     $       ' Iteration: ',ITER,DTDR,LAGRANM,VRXNCS,DFDL

    IF(ABS(VRXNCS).GT.RSMALL) THEN
       LOLD=LAGRANM

       FDTTAN=DOTVEC(XTMP1,XTAN,NRXATM)+DOTVEC(YTMP1,YTAN,NRXATM)+ &
            DOTVEC(ZTMP1,ZTAN,NRXATM)

       FDTTAN=FDTTAN*DTDR1

       DFDL=FDTTAN

       LAGRANM=LOLD-VRXNCS/DFDL

       IF(ITER.GT.MXITER) THEN
          CALL WRNDIE(-3,'<SHKRXN2>', &
               'Reaction coordinate type 3 does not converge')
       ENDIF

       GOTO 2222
    ENDIF

    DO I=1,NRXATM
       XN(I)=XTMP(I)
       YN(I)=YTMP(I)
       ZN(I)=ZTMP(I)
    ENDDO

    LAGRANM=LAGRANM*DTDR*WTOT

    !      write(*,19)' lag ',LAGRANM,LAGRANM/0.0004184D0,DTDR,WTOT

18  FORMAT('SHKRXN3> ',A,I5,5E16.6)
19  FORMAT('SHKRXN3> ',A,5E16.6)
    RETURN
  END SUBROUTINE SHKRXN_3

  SUBROUTINE RXNCONS_3F(XREF,YREF,ZREF,DX,DY,DZ,NATOM,IRXCNS, &
       NRXATM,LRXPRN,IRXATM,XCORD,YCORD,ZCORD,IMOVE, &
       AMASS,NUMLP,LPNHOST,LPHPTR,LPHOST,LPVALUE,LPWGHT, &
       LPRID)

    !
    use exfunc
    use number
    use stream
    use memory
    use rxnconswt
    use rxncons3
    !
    ! . Parsed variables
    !
    integer,allocatable,dimension(:) :: ATMPR
    real(chm_real),allocatable,dimension(:) :: RA
    real(chm_real),allocatable,dimension(:) :: RB
    real(chm_real) XREF(*),YREF(*),ZREF(*),DX(*),DY(*),DZ(*)
    INTEGER NATOM,IRXCNS,NRXATM,IRXATM(*)
    real(chm_real) XCORD(*),YCORD(*),ZCORD(*)
    LOGICAL LRXPRN

    INTEGER LPRID(*)
    INTEGER IMOVE(*),NUMLP,LPNHOST(*),LPHPTR(*),LPHOST(*)
    real(chm_real) LPVALUE(3,*),AMASS(*)
    LOGICAL LPWGHT(*)

    !
    ! . Local variables
    !
    INTEGER I,IA
    real(chm_real) DRDTNRM,VRXNCS,D1,D2,DTDR

    DO I=1,NRXATM
       IA=IRXATM(I)
       XCORD(I)=XREF(IA)
       YCORD(I)=YREF(IA)
       ZCORD(I)=ZREF(IA)
    ENDDO
    !
    ! Prepare force vector base on REFC
    !
    call chmalloc('rxcons_3.src','RXNCONS_32','ATMPR',2*NRXATM,intg=ATMPR)
    call chmalloc('rxcons_3.src','RXNCONS_32','RA',3*NRXATM,crl=RA)
    call chmalloc('rxcons_3.src','RXNCONS_32','RB',3*NRXATM,crl=RB)

    CALL CALCTAU(NRXATM,     XCORD ,     YCORD ,     ZCORD , &
         XREFA,YREFA,ZREFA, &
         XREFB,YREFB,ZREFB, &
         XTAN ,YTAN ,ZTAN , &
         WMASS,QCNOTRN,QCNOROT,VRXNCS, &
         ATMPR,RA,RB,.FALSE.,DTDR)

    CALL PROJF_3(NRXATM,NATOM,IRXATM, &
         XTAN,YTAN,ZTAN,DX,DY,DZ, &
         IMOVE,AMASS,NUMLP, &
         LPNHOST,LPHPTR,LPHOST,LPVALUE,LPWGHT,LPRID)

    call chmdealloc('rxcons_3.src','RXNCONS_32','ATMPR',2*NRXATM,intg=ATMPR)
    call chmdealloc('rxcons_3.src','RXNCONS_32','RA',3*NRXATM,crl=RA)
    call chmdealloc('rxcons_3.src','RXNCONS_32','RB',3*NRXATM,crl=RB)

    RETURN
  END SUBROUTINE RXNCONS_3F

  SUBROUTINE PROJF_3(NRXATM,NATOM,IRXATM, &
       XTAN,YTAN,ZTAN,DX,DY,DZ,IMOVE,AMASS,NUMLP, &
       LPNHOST,LPHPTR,LPHOST,LPVALUE,LPWGHT,LPRID)
    !
    use exfunc
    use number
    use stream
    !
    ! . Parsed variables
    !
    real(chm_real) XTAN(*),YTAN(*),ZTAN(*),DX(*),DY(*),DZ(*)
    INTEGER NATOM,IRXCNS,NRXATM,IRXATM(*)

    INTEGER LPRID(*)
    INTEGER IMOVE(*),NUMLP,LPNHOST(*),LPHPTR(*),LPHOST(*)
    real(chm_real) LPVALUE(3,*),AMASS(*)
    LOGICAL LPWGHT(*)

    !
    INTEGER I,IA
    real(chm_real) SUM,FD,A
    INTEGER ILP,N,K,IPT,JPT
    real(chm_real) XF,YF,ZF

    !      A=DOTVEC(XTAN,XTAN,NRXATM)+
    !     &  DOTVEC(YTAN,YTAN,NRXATM)+
    !     &  DOTVEC(ZTAN,ZTAN,NRXATM)
    !
    !      A=ONE/SQRT(A)
    !
    !      DO I=1,NRXATM
    !        XTAN(I)=XTAN(I)*A
    !        YTAN(I)=YTAN(I)*A
    !        ZTAN(I)=ZTAN(I)*A
    !      ENDDO

    FD=ZERO
    DO I=1,NRXATM
       IA=IRXATM(I)
       IF(IMOVE(IA).NE.-1)THEN
          FD=FD+XTAN(I)*DX(IA)
          FD=FD+YTAN(I)*DY(IA)
          FD=FD+ZTAN(I)*DZ(IA)
       ELSE
          XF=ZERO
          YF=ZERO
          ZF=ZERO

          ILP=LPRID(IA)
          N=LPNHOST(ILP)
          IPT=LPHPTR(ILP)

          DO K=1,N
             JPT=LPHOST(IPT+K)
             XF=XF+DX(JPT)
             YF=YF+DY(JPT)
             ZF=ZF+DZ(JPT)
          ENDDO

          FD=FD+XTAN(I)*XF
          FD=FD+YTAN(I)*YF
          FD=FD+ZTAN(I)*ZF
          !            write(*,*)'ooo ',jrep,j,dx(ia),dy(ia),dz(ia)
       ENDIF
    ENDDO

    DO I=1,NRXATM
       IA=IRXATM(I)
       IF(IMOVE(IA).NE.-1)THEN
          DX(IA)=DX(IA)-FD*XTAN(I)
          DY(IA)=DY(IA)-FD*YTAN(I)
          DZ(IA)=DZ(IA)-FD*ZTAN(I)
       ELSE
          ILP=LPRID(IA)
          N=LPNHOST(ILP)
          IPT=LPHPTR(ILP)

          DO K=1,N
             JPT=LPHOST(IPT+K)
             !            DX(JPT)=DX(JPT)-FD*XTAN(I)*AM
             !            DY(JPT)=DY(JPT)-FD*YTAN(I)*AM
             !            DZ(JPT)=DZ(JPT)-FD*ZTAN(I)*AM
          ENDDO
       ENDIF
    ENDDO

    RETURN
  END SUBROUTINE PROJF_3

  SUBROUTINE GETFACTORS(NRXATM,XCORD,YCORD,ZCORD,          &
       XREFM,YREFM,ZREFM,XREFA,YREFA,ZREFA,  &
       XREFB,YREFB,ZREFB,                    &
       XTAN,YTAN,ZTAN,XCURV,YCURV,ZCURV,     &
       XM1,  YM1,  ZM1, XP1, YP1, ZP1,       &
       WMASS,QCNOTRN,                        &
       QCNOROT,VRXNCS,ATMPR,RA,RB,ATMASS,    &
       QEQDIS,D1,D2,ZFAC,GFAC,               &
       RCNSDL,RCNSDS,RCNSDSM,RCNSDC,RCNSDCM, &
       RCNSDR,RCNSDRC,RCNSAMF)


    use exfunc
    use number
    use psf
    use stream
    implicit none
    real(chm_real) XCORD(*),YCORD(*),ZCORD(*)
    real(chm_real) XREFM(*),YREFM(*),ZREFM(*), &
         XREFA(*),YREFA(*),ZREFA(*)
    real(chm_real) XREFB(*),YREFB(*),ZREFB(*),XTAN(*),YTAN(*),ZTAN(*)
    real(chm_real) XM1(*),YM1(*),ZM1(*),XP1(*),YP1(*),ZP1(*)
    real(chm_real) WMASS(*),ATMASS(*)
    real(chm_real) VRXNCS,XCURV(*),YCURV(*),ZCURV(*)
    INTEGER NRXATM
    INTEGER ATMPR(2,NRXATM)
    real(chm_real) RA(3,NRXATM),RB(3,NRXATM),D1,D2
    real(chm_real) ZFAC,GFAC,RCNSDL,RCNSDR
    REAL(chm_real) RCNSDS,RCNSDSM,RCNSDC,RCNSDCM,RCNSDRC,RCNSAMF
    LOGICAL QCNOROT,QCNOTRN,QEQDIS

    INTEGER I,J,K
    real(chm_real) WTOT,DISTAN,DEVA(3,3),EVWID,TEMP
    real(chm_real) SUM,SUM1,ROTRB,DIFF,A,B,FACT,DOT
    real(chm_real) MM,MP,PP
    LOGICAL LPRINT,LPRINT1

    EVWID = ZERO ! never set, not sure of intent

    LPRINT=(PRNLEV.GT.5)
    LPRINT1=(PRNLEV.GT.9)

    DO I=1,NRXATM
       ATMPR(1,I)=I
       ATMPR(2,I)=I
    ENDDO

    CALL ECBSTF(XCORD,YCORD,ZCORD,XREFB,YREFB,ZREFB,ATMPR,NRXATM, &
         QCNOROT,QCNOTRN,LPRINT1, (/ ZERO /), .FALSE.,WMASS, &
         WTOT,DISTAN,RA,RB,DEVA,EVWID)

    TEMP=ONE/WTOT

    D2=SQRT(DISTAN*TEMP)

    IF(D2.LT.TENM5) THEN
       CALL WRNDIE(-2,'<GETFACTORS>','zero tangent length i+1 - i')
    ENDIF

    IF(LPRINT) THEN
       WRITE(OUTU,68) 'I+1 - I', D2
    ENDIF

    DO K=1,NRXATM
       ROTRB=ZERO
       DO J=1,3
          ROTRB=ROTRB+DEVA(1,J)*RB(J,K)
       ENDDO
       XP1(K)=ROTRB

       ROTRB=ZERO
       DO J=1,3
          ROTRB=ROTRB+DEVA(2,J)*RB(J,K)
       ENDDO
       YP1(K)=ROTRB

       ROTRB=ZERO
       DO J=1,3
          ROTRB=ROTRB+DEVA(3,J)*RB(J,K)
       ENDDO
       ZP1(K)=ROTRB
    ENDDO

    CALL ECBSTF(XCORD,YCORD,ZCORD,XREFA,YREFA,ZREFA,ATMPR,NRXATM, &
         QCNOROT,QCNOTRN,LPRINT1, (/ ZERO /), .FALSE.,WMASS, &
         WTOT,DISTAN,RA,RB,DEVA,EVWID)

    D1=SQRT(DISTAN*TEMP)
    IF(D1.LT.TENM5) THEN
       CALL WRNDIE(-2,'<GATFACTORS>','zero tangent length i - i-1')
    ENDIF
    IF(LPRINT) THEN
       WRITE(OUTU,68) 'I - I-1', D1,D1/SQRT(WTOT)
    ENDIF

    VRXNCS=HALF*(D1-D2)

    DO K=1,NRXATM
       ROTRB=ZERO
       DO J=1,3
          ROTRB=ROTRB+DEVA(1,J)*RB(J,K)
       ENDDO
       XM1(K)=ROTRB

       ROTRB=ZERO
       DO J=1,3
          ROTRB=ROTRB+DEVA(2,J)*RB(J,K)
       ENDDO
       YM1(K)=ROTRB

       ROTRB=ZERO
       DO J=1,3
          ROTRB=ROTRB+DEVA(3,J)*RB(J,K)
       ENDDO
       ZM1(K)=ROTRB
    ENDDO

    DO K=1,NRXATM
       XTAN(K) =(XP1(K)-XM1(K))
       YTAN(K) =(YP1(K)-YM1(K))
       ZTAN(K) =(ZP1(K)-ZM1(K))
       XCURV(K)=(XP1(K)+XM1(K))*HALF
       YCURV(K)=(YP1(K)+YM1(K))*HALF
       ZCURV(K)=(ZP1(K)+ZM1(K))*HALF
    ENDDO

    SUM=ZERO
    SUM1=ZERO
    DO K=1,NRXATM
       FACT=WMASS(K)*TEMP
       DOT=XTAN(K)*XTAN(K)+ &
            YTAN(K)*YTAN(K)+ &
            ZTAN(K)*ZTAN(K)

       SUM1=SUM1+DOT*FACT
       SUM =SUM +DOT*FACT*FACT
    ENDDO

    RCNSDS  = HALF*SQRT(SUM1)
    RCNSDSM = HALF*SQRT(SUM)

    DO K=1,NRXATM
       FACT=WMASS(K)*TEMP
       XTAN(K)=XTAN(K)*FACT
       YTAN(K)=YTAN(K)*FACT
       ZTAN(K)=ZTAN(K)*FACT
    ENDDO

    SUM=ZERO
    DO K=1,NRXATM
       SUM=SUM+XTAN(K)*XTAN(K)+ &
            YTAN(K)*YTAN(K)+ &
            ZTAN(K)*ZTAN(K)
    ENDDO

    SUM=ONE/SQRT(SUM)
    DO K=1,NRXATM
       XTAN(K)=XTAN(K)*SUM
       YTAN(K)=YTAN(K)*SUM
       ZTAN(K)=ZTAN(K)*SUM
    ENDDO

    CALL ECBSTF(XCORD,YCORD,ZCORD,XREFM,YREFM,ZREFM,ATMPR,NRXATM, &
         QCNOROT,QCNOTRN,LPRINT1, (/ ZERO /), .FALSE.,WMASS, &
         WTOT,DISTAN,RA,RB,DEVA,EVWID)

    RCNSDR=SQRT(DISTAN/WTOT)

    DO K=1,NRXATM
       ROTRB=ZERO
       DO J=1,3
          ROTRB=ROTRB+DEVA(1,J)*RB(J,K)
       ENDDO
       XM1(K)  = RA(1,K)-ROTRB
       XCURV(K)=XCURV(K)-ROTRB

       ROTRB=ZERO
       DO J=1,3
          ROTRB=ROTRB+DEVA(2,J)*RB(J,K)
       ENDDO
       YM1(K)  = RA(2,K)-ROTRB
       YCURV(K)=YCURV(K)-ROTRB

       ROTRB=ZERO
       DO J=1,3
          ROTRB=ROTRB+DEVA(3,J)*RB(J,K)
       ENDDO
       ZM1(K)  = RA(3,K)-ROTRB
       ZCURV(K)=ZCURV(K)-ROTRB
    ENDDO

    SUM=ZERO
    SUM1=ZERO
    DO K=1,NRXATM
       FACT=WMASS(K)*TEMP
       DOT=XCURV(K)*XCURV(K)+ &
            YCURV(K)*YCURV(K)+ &
            ZCURV(K)*ZCURV(K)

       SUM1=SUM1+DOT*FACT
       SUM =SUM +DOT*FACT*FACT
    ENDDO

    RCNSDC  = SQRT(SUM1)
    RCNSDCM = SQRT(SUM)

    SUM=ONE/RCNSDCM

    DO K=1,NRXATM
       FACT=WMASS(K)*TEMP
       XCURV(K)=XCURV(K)*FACT*SUM
       YCURV(K)=YCURV(K)*FACT*SUM
       ZCURV(K)=ZCURV(K)*FACT*SUM
    ENDDO

    SUM=ZERO
    DO K=1,NRXATM
       SUM=SUM+XCURV(K)*XM1(K)+ &
            YCURV(K)*YM1(K)+ &
            ZCURV(K)*ZM1(K)
    ENDDO

    RCNSDRC=-SUM
    RCNSDRC=RCNSDRC*RCNSDCM/RCNSDC

    CALL ECBSTF(XREFM,YREFM,ZREFM,XREFA,YREFA,ZREFA,ATMPR,NRXATM, &
         QCNOROT,QCNOTRN,LPRINT1, (/ ZERO /), .FALSE.,WMASS, &
         WTOT,DISTAN,RA,RB,DEVA,EVWID)
    RCNSDL=SQRT(DISTAN*TEMP)

    RCNSAMF=RCNSDRC*RCNSDC/RCNSDL/RCNSDS

    ZFAC=ZERO
    MM=ZERO
    MP=ZERO
    PP=ZERO
    DO K=1,NRXATM
       FACT=XTAN(K)*XTAN(K)+ &
            YTAN(K)*YTAN(K)+ &
            ZTAN(K)*ZTAN(K)
       FACT=FACT*ATMASS(K)
       ZFAC=ZFAC+FACT

       !        MM=MM+ATMASS(K)*(
       !     $        XM1(K)*XM1(K)+
       !     $        YM1(K)*YM1(K)+
       !     $        ZM1(K)*ZM1(K))
       !
       !        MP=MP+ATMASS(K)*(
       !     $        XP1(K)*XM1(K)+
       !     $        YP1(K)*YM1(K)+
       !     $        ZP1(K)*ZM1(K))
       !
       !        PP=PP+ATMASS(K)*(
       !     $        XP1(K)*XP1(K)+
       !     $        YP1(K)*YP1(K)+
       !     $        ZP1(K)*ZP1(K))
    ENDDO

    !      GFAC=-(MM*MM+MM*MP*TWO+MP*MP)+
    !     $      (MP*MP+PP*MP*TWO+PP*PP)
    !      write(*,'(6F12.5)')mm,mp,pp,zfac,gfac,GFAC/(ZFAC*ZFAC)
    !      GFAC=GFAC/(ZFAC*ZFAC)

    GFAC=ZERO

    IF(LPRINT) THEN
       WRITE(OUTU,88)'       FACTORS '
       WRITE(OUTU,59)ZFAC,GFAC,RCNSDL,RCNSDS,RCNSDSM,RCNSDC,RCNSDCM, &
            RCNSDR,RCNSDRC,RCNSAMF
    ENDIF

58  FORMAT('GETFACTOR> ',3F14.5)
59  FORMAT('GETFACTOR> ',2F12.4,8F16.6)
68  FORMAT('GETFACTOR> ',A,2F14.5)
78  FORMAT(I5)
88  FORMAT('GETFACTOR> ',A)
98  FORMAT('GETFACTOR> ',1F14.5)
    RETURN
  END SUBROUTINE GETFACTORS

  SUBROUTINE GETFORC(NRXATM,DX,DY,DZ,XFORC,YFORC,ZFORC, &
       XTAN,YTAN,ZTAN,IRXATM,LPRID,FRCONS)

    use exfunc
    use number
    use stream
    use dimens_fcm
    use psf
    use lonepr
    implicit none

    INTEGER NRXATM
    REAL(chm_real) DX(*),DY(*),DZ(*)
    REAL(chm_real) XFORC(*),YFORC(*),ZFORC(*)
    REAL(chm_real) XTAN(*),YTAN(*),ZTAN(*),FRCONS
    INTEGER IRXATM(*),LPRID(*)

    INTEGER I,IA,ILP,IPT,JPT,K,N
    REAL(chm_real) XF,YF,ZF

    XFORC(1:NRXATM)=zero
    YFORC(1:NRXATM)=zero
    ZFORC(1:NRXATM)=zero

    DO I=1,NRXATM
       IA=IRXATM(I)
       IF(IMOVE(IA).NE.-1)THEN
          XFORC(I)=DX(IA)
          YFORC(I)=DY(IA)
          ZFORC(I)=DZ(IA)
       ELSE
          XF=ZERO
          YF=ZERO
          ZF=ZERO

          ILP=LPRID(IA)
          N=LPNHOST(ILP)
          IPT=LPHPTR(ILP)

          DO K=1,N
             JPT=LPHOST(IPT+K)
             XF=XF+DX(JPT)
             YF=YF+DY(JPT)
             ZF=ZF+DZ(JPT)
          ENDDO

          XFORC(I)=XF
          YFORC(I)=YF
          ZFORC(I)=ZF
          !            write(*,*)'ooo ',jrep,j,dx(ia),dy(ia),dz(ia)
       ENDIF
    ENDDO

    FRCONS=ZERO
    DO I=1,NRXATM
       FRCONS=FRCONS+XFORC(I)*XTAN(I)+ &
            YFORC(I)*YTAN(I)+ &
            ZFORC(I)*ZTAN(I)
    ENDDO
    !
    RETURN
  END SUBROUTINE GETFORC
#endif /* (rxcons_3)*/

end module rxcons3

