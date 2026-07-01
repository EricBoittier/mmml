! Utility Graphics routines
!
SUBROUTINE GETCOLOR(ICOL,WRD,IDEF)
  !
  ! This routine gets the color index from that color name.
  !
  use chm_kinds
  implicit none
  !
  INTEGER ICOL,IDEF
  CHARACTER(len=4) WRD
  !
#if KEY_NOGRAPHICS==0
  !
  INTEGER ID
  !
  ID=IDEF
  !
  IF(WRD.EQ.'    ') THEN
     ICOL=ID
  ELSE IF(WRD.EQ.'NONE') THEN
     ICOL=0
  ELSE IF(WRD.EQ.'RED ') THEN
     ICOL=1
     ! light gray
  ELSE IF(WRD.EQ.'BLAC') THEN
     ICOL=2
  ELSE IF(WRD.EQ.'YELL') THEN
     ICOL=3
  ELSE IF(WRD.EQ.'GREE') THEN
     ICOL=4
  ELSE IF(WRD.EQ.'WHIT') THEN
     ICOL=5
  ELSE IF(WRD.EQ.'BLUE') THEN
     ICOL=6
  ELSE IF(WRD.EQ.'CYAN') THEN
     ICOL=7
  ELSE IF(WRD.EQ.'MAGE') THEN
     ICOL=8
     ! dark gray
  ELSE IF(WRD.EQ.'GRAY') THEN
     ICOL=9
  ELSE IF(WRD.EQ.'ORAN') THEN
     ICOL=10
  ELSE IF(WRD.EQ.'BROW') THEN
     ICOL=11
  ELSE IF(WRD.EQ.'PURP') THEN
     ICOL=12
  ELSE IF(WRD.EQ.'TURQ') THEN
     ICOL=13
  ELSE IF(WRD.EQ.'CHAR') THEN
     ICOL=14
     ! dark blue
  ELSE IF(WRD.EQ.'DKBL') THEN
     ICOL=15
     ! By number
  ELSE IF(WRD.EQ.'0   ') THEN
     ICOL=0
  ELSE IF(WRD.EQ.'1   ') THEN
     ICOL=1
  ELSE IF(WRD.EQ.'2   ') THEN
     ICOL=2
  ELSE IF(WRD.EQ.'3   ') THEN
     ICOL=3
  ELSE IF(WRD.EQ.'4   ') THEN
     ICOL=4
  ELSE IF(WRD.EQ.'5   ') THEN
     ICOL=5
  ELSE IF(WRD.EQ.'6   ') THEN
     ICOL=6
  ELSE IF(WRD.EQ.'7   ') THEN
     ICOL=7
  ELSE IF(WRD.EQ.'8   ') THEN
     ICOL=8
  ELSE IF(WRD.EQ.'9   ') THEN
     ICOL=9
  ELSE IF(WRD.EQ.'10  ') THEN
     ICOL=10
  ELSE IF(WRD.EQ.'11  ') THEN
     ICOL=11
  ELSE IF(WRD.EQ.'12  ') THEN
     ICOL=12
  ELSE IF(WRD.EQ.'13  ') THEN
     ICOL=13
  ELSE IF(WRD.EQ.'14  ') THEN
     ICOL=14
  ELSE IF(WRD.EQ.'15  ') THEN
     ICOL=15
  ELSE
     ICOL=-1
     CALL WRNDIE(0,'<GETCOLOR>','BAD COLOR TEXT')
  ENDIF
  !
  RETURN
END SUBROUTINE GETCOLOR
!
SUBROUTINE ASSDEFCOLOR(NATOM,ATYPE,ICOLOR,ICOLRC)
  !
  use chm_kinds
  implicit none
  !
  INTEGER NATOM
  INTEGER ICOLOR(*),ICOLRC(*)
  CHARACTER(len=*) ATYPE(*)
  !
  INTEGER I,J
  CHARACTER(len=1) CFIRST
  !
  ! and assign default radii
  DO I=1,NATOM
     CFIRST=ATYPE(I)
     IF(CFIRST.EQ.'C') THEN
        J=2
     ELSE IF(CFIRST.EQ.'H') THEN
        J=5
     ELSE IF(CFIRST.EQ.'O') THEN
        J=1
     ELSE IF(CFIRST.EQ.'N') THEN
        J=6
     ELSE IF(CFIRST.EQ.'S') THEN
        J=3
     ELSE IF(CFIRST.EQ.'F') THEN
        J=3
     ELSE IF(CFIRST.EQ.'P') THEN
        J=4
     ELSE
        J=7
     ENDIF
     ICOLOR(I)=J
     ICOLRC(I)=J
  ENDDO
  RETURN
END SUBROUTINE ASSDEFCOLOR
!
SUBROUTINE ASSDEFRADII(NATOM,ATYPE,RADII)
  !
  use chm_kinds
  implicit none
  !
  INTEGER NATOM
  real(chm_real) RADII(*)
  CHARACTER(len=*) ATYPE(*)
  !
  INTEGER I
  CHARACTER(len=1) CFIRST

  ! and assign default radii
  DO I=1,NATOM
     CFIRST=ATYPE(I)
     IF(CFIRST.EQ.'C') THEN
        RADII(I)=1.6
     ELSE IF(CFIRST.EQ.'H') THEN
        RADII(I)=1.0
     ELSE IF(CFIRST.EQ.'O') THEN
        RADII(I)=1.35
     ELSE IF(CFIRST.EQ.'N') THEN
        RADII(I)=1.7
     ELSE IF(CFIRST.EQ.'S') THEN
        RADII(I)=1.9
     ELSE IF(CFIRST.EQ.'F') THEN
        RADII(I)=1.05
     ELSE IF(CFIRST.EQ.'P') THEN
        RADII(I)=1.9
     ELSE
        RADII(I)=1.9
     ENDIF
  ENDDO
  RETURN
END SUBROUTINE ASSDEFRADII
!
SUBROUTINE GRSETUP(QFIRST)
  !
  !  RM Venable <*> August 1993
  !  added for nodisplay and SG GL displays; needed for derived files
  !  initializes pseudo-display used for PLUTO, LIGHT, HPGL, PS files
  !  set vars in common block /GRAPH/, coordinate transform, etc.
  !
  use chm_kinds
  use dimens_fcm
  use stream
  use graph
  implicit none

  LOGICAL QFIRST
  INTEGER I,J

  !
  ! INITIALIZE USCREN(4,4) MATRIX
  !
  DO I=1,4
     DO J=1,4
        USCREN(J,I)=0.0
     ENDDO
  ENDDO
  !
  GRDUPA=32.0
  USCREN(1,1)=32.0
  USCREN(2,2)=-32.0
  USCREN(3,3)=32.0
  USCREN(4,4)=1.0
  USCREN(1,4)=640.0D0
  USCREN(2,4)=512.0D0
  !
  IF (QFIRST) THEN
     IGRWIDTH=1
     IGRFONT=2
     DSTERO=18.0D0
     ! initialize the axes tips
     DO I=1,7
        DO J=1,3
           AXEXYZ(J,I) = 0.0
        ENDDO
        AXEXYZ(4,I) =1.0
     ENDDO
     AXEXYZ(1,1) =  25.0
     AXEXYZ(1,2) = -25.0
     AXEXYZ(2,3) =  25.0
     AXEXYZ(2,4) = -25.0
     AXEXYZ(3,5) =  25.0
     AXEXYZ(3,6) = -25.0
     ! establish the 8-bit color map for color postscript
     ! 192 = 16 colors, 12 intensity levels (zcue)
     I=192
     CALL INICLRMAP(I)
  ENDIF

  RETURN
END SUBROUTINE GRSETUP

SUBROUTINE RESETVIEW(ULAB,UMOD)
  !
  use chm_kinds
  implicit none
  !
  real(chm_real) ULAB(4,4),UMOD(4,4)
  !
  INTEGER I,J
  !
  ! reset the view matrix.
  DO I=1,4
     DO J=1,4
        ULAB(J,I)=0.0
        UMOD(J,I)=0.0
     ENDDO
     ULAB(I,I)=1.0
     UMOD(I,I)=1.0
  ENDDO
  CALL CREATETRANS(.FALSE.)
  RETURN
END SUBROUTINE RESETVIEW
!
SUBROUTINE MODIFYTRANS(UMOD,QLAB,QDEBUG)
  !
  use chm_kinds
  use dimens_fcm
  use graph
  implicit none
  !
  real(chm_real) UMOD(4,4)
  LOGICAL QLAB,QDEBUG
  !
  real(chm_real) UTEMP(4,4)
  INTEGER I,J
  !
  IF(QLAB) THEN
     CALL VECMMUL(UMOD,ULAB,UTEMP)
  ELSE
     CALL VECMMUL(ULAB,UMOD,UTEMP)
  ENDIF
  !
  IF(QDEBUG) CALL WRMATX('MOD TRANS:',UMOD)
  !
  DO I=1,4
     DO J=1,4
        ULAB(J,I)=UTEMP(J,I)
        UMOD(J,I)=0.0
     ENDDO
     UMOD(I,I)=1.0
  ENDDO
  !
  ! create-the-transformation-matrix
  CALL CREATETRANS(QDEBUG)
  RETURN
END SUBROUTINE MODIFYTRANS
!
SUBROUTINE CREATETRANS(QDEBUG)
  !
  use chm_kinds
  use dimens_fcm
  use graph
  use stream
  implicit none
  !
  LOGICAL QDEBUG
  !
  real(chm_real) UTEMP(4,4)
  !
  IF(QDEBUG) CALL WRMATX('LAB FRAME:',ULAB)
  IF(QDEBUG) CALL WRMATX('SCREEN U :',USCREN)
  IF(QSTERO) THEN
     ! no stereo X offset for povray; images written to separate files
     IF(QPOVR) THEN
        USTERL(1,4) = 0.0
        USTERR(1,4) = 0.0
     ELSE
        USTERR(1,4)=DSTERO*0.5
        USTERL(1,4)=-DSTERO*0.5
     ENDIF
     CALL VECMMUL(USTERL,ULAB,UTEMP)
     CALL VECMMUL(USCREN,UTEMP,ULEFT)
     CALL VECMMUL(USTERR,ULAB,UTEMP)
     CALL VECMMUL(USCREN,UTEMP,URIGHT)
     IF(QDEBUG) THEN
        CALL WRMATX('FINAL USL:',USTERL)
        CALL WRMATX('FINAL USR:',USTERR)
        CALL WRMATX('FINAL UL :',ULEFT)
        CALL WRMATX('FINAL UR :',URIGHT)
     ENDIF
  ELSE
     CALL VECMMUL(USCREN,ULAB,ULEFT)
     IF(QDEBUG) CALL WRMATX('FINAL U  :',ULEFT)
  ENDIF
  IF(PRNLEV.GT.5) WRITE(OUTU,435) ULAB
435 FORMAT(' LABORATORY FRAME TRANSLATION/ROTATION MATRIX:', &
       4(/20X,4F12.6))
  RETURN
END SUBROUTINE CREATETRANS
#else /*  IFN NOGRAPHICS*/
  RETURN
END SUBROUTINE GETCOLOR
#endif /*  IFN NOGRAPHICS*/

