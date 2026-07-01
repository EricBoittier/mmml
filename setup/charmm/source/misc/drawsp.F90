#if KEY_NOMISC==0
SUBROUTINE DRAWSP(COMLYN,COMLEN,X,Y,Z,WMAIN, &
     XCOMP,YCOMP,ZCOMP,ISLCT)
  !-----------------------------------------------------------------------
  !     SETS UP FOR MOLECULE DRAWING USING D.J. STATES PLOTTING
  !     PROGRAM PLT2.  MOD by LN MAY 1986
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use number
  use psf
  use param
  use hbondm
  use code
  use stream
  use string
  use machutil,only:die

  implicit none
  !
  CHARACTER(len=*) COMLYN
  INTEGER COMLEN
  real(chm_real) X(*),Y(*),Z(*),WMAIN(*)
  real(chm_real) XCOMP(*),YCOMP(*),ZCOMP(*)
  INTEGER ISLCT(*)
  !
  !
  INTEGER IUNIT,NOMO
  real(chm_real) DFACT
  !
  IUNIT=GTRMI(COMLYN,COMLEN,'UNIT',-1)
  IF(IUNIT.LT.0) CALL DIE
  DFACT=GTRMF(COMLYN,COMLEN,'DFAC',ZERO)
  NOMO=0
  IF(INDXA(COMLYN,COMLEN,'NOMO').GT.0) NOMO=1
  !
  !     Selection should be done before DRAWSP is called?
  !     CALL SELCTA(COMLYN,COMLEN,ISLCT,X,Y,Z,WMAIN,.TRUE.)
  !
  IF(IUNIT.NE.OUTU .AND. IOLEV.LT.0) RETURN
  IF(IUNIT.EQ.OUTU .AND. PRNLEV.LE.2) RETURN
  !
  CALL DRMOLV(NATOM,X,Y,Z,XCOMP,YCOMP,ZCOMP, &
       DFACT,IB,JB,ICB,CBC,NBOND,IHB,JHB,KHB,NHB,IUNIT,NOMO,ISLCT)
  !
  RETURN
END SUBROUTINE DRAWSP

SUBROUTINE DRMOLV(NATOM,X,Y,Z,XCOMP,YCOMP,ZCOMP,DFACT, &
     IB,JB,ICB,CBC,NBOND,IHB,JHB,KHB,NHB,IUNIT,NOMO,ISLCT)
  !-----------------------------------------------------------------------
  !     Actaully prepares the drawing for PLT2. Called by DRAWSP.
  !
  use chm_kinds
  implicit none
  INTEGER NATOM
  real(chm_real) XCOMP(*),YCOMP(*),ZCOMP(*)
  real(chm_real) X(*),Y(*),Z(*)
  real(chm_real) DFACT,CBC(*)
  INTEGER IB(*),JB(*),IHB(*),JHB(*),KHB(*),ICB(*)
  INTEGER ISLCT(*)
  INTEGER IUNIT,NBOND,NHB,NOMO
  !
  INTEGER I,I1,I2,IX
  real(chm_real) X1,Y1,Z1,XMIN,XMAX,YMIN,YMAX
  !
  !     Find min&max values to give scaling information...
  XMIN=9999.9
  YMIN=XMIN
  XMAX=-XMIN
  YMAX=XMAX
  IF(NOMO.EQ.0) THEN
     DO I=1,NBOND
        I1=IB(I)
        I2=JB(I)
        IF (.NOT.(I1.LE.0 .OR. I2.LE.0)) THEN
           IF(ISLCT(I1).GT.0 .AND. ISLCT(I2).GT.0) THEN
              XMIN=MIN(XMIN,X(I1),X(I2))
              XMAX=MAX(XMAX,X(I1),X(I2))
              YMIN=MIN(YMIN,Y(I1),Y(I2))
              YMAX=MAX(YMAX,Y(I1),Y(I2))
           ENDIF
        ENDIF
     ENDDO
     !     And the hydrogen bonds
     DO I=1,NHB
        I1=IHB(I)
        I2=JHB(I)
        IF(KHB(I).GT.0) I1=KHB(I)
        IF(ISLCT(I1).GT.0 .AND. ISLCT(I2).GT.0) THEN
           XMIN=MIN(XMIN,X(I1),X(I2))
           XMAX=MAX(XMAX,X(I1),X(I2))
           YMIN=MIN(YMIN,Y(I1),Y(I2))
           YMAX=MAX(YMAX,Y(I1),Y(I2))
        ENDIF
     ENDDO
  ENDIF
  !
  !     Now check possible derivative influence on scaling..
  !
  IF(DFACT .GT. 0.0) THEN
     DO I=1,NATOM
        IF(ISLCT(I).GT.0) THEN
           X1=X(I)+XCOMP(I)*DFACT
           Y1=Y(I)+YCOMP(I)*DFACT
           XMIN=MIN(XMIN,X(I),X1)
           XMAX=MAX(XMAX,X(I),X1)
           YMIN=MIN(YMIN,Y(I),Y1)
           YMAX=MAX(YMAX,Y(I),Y1)
        ENDIF
     ENDDO
  ENDIF
  !
  !     Assume quadratic plotting area
  !
  IF (XMAX-XMIN .GT. YMAX-YMIN) THEN
     YMAX=YMIN+(XMAX-XMIN)
  ELSE
     XMAX=XMIN+(YMAX-YMIN)
  ENDIF
  !
  WRITE(IUNIT,50) XMIN,XMAX,YMIN,YMAX
50 FORMAT('XMIN ',F10.4/'XMAX ',F10.4 &
       /'YMIN ',F10.4/'YMAX ',F10.4/'SET ')
  !
  IF(NOMO.NE.0) GOTO 80
  DO I=1,NBOND
     I1=IB(I)
     I2=JB(I)
     IF (.NOT.(I1.LE.0 .OR. I2.LE.0)) THEN
        IF(ISLCT(I1).GT.0 .AND. ISLCT(I2).GT.0) THEN
           IX=ICB(I)
           IF (CBC(IX).GT.500.) THEN
              WRITE(IUNIT,55) X(I1),Y(I1),Z(I1),X(I2),Y(I2),Z(I2)
           ELSE
              WRITE(IUNIT,65) X(I1),Y(I1),Z(I1),X(I2),Y(I2),Z(I2)
           ENDIF
        ENDIF
     ENDIF
  ENDDO
55 FORMAT('LIN ',6F12.6)
65 FORMAT('LIN ',6F12.6,A)
  !
  DO I=1,NHB
     I1=IHB(I)
     I2=JHB(I)
     IF(KHB(I).GT.0) I1=KHB(I)
     IF(ISLCT(I1).GT.0 .AND. ISLCT(I2).GT.0) THEN
        WRITE(IUNIT,65) X(I1),Y(I1),Z(I1),X(I2),Y(I2),Z(I2),' DOT'
     ENDIF
  ENDDO
  IF(DFACT.LE.0.0) RETURN
  !
80 CONTINUE
  DO I=1,NATOM
     IF(ISLCT(I).GT.0) THEN
        X1=X(I)+XCOMP(I)*DFACT
        Y1=Y(I)+YCOMP(I)*DFACT
        Z1=Z(I)+ZCOMP(I)*DFACT
        WRITE(IUNIT,55) X(I),Y(I),Z(I),X1,Y1,Z1
     ENDIF
  ENDDO
  !
END SUBROUTINE DRMOLV
#else /**/
SUBROUTINE NULL_DP
  RETURN
END SUBROUTINE NULL_DP
#endif 


