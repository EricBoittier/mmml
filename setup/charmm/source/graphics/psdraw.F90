#if KEY_NOGRAPHICS==1
SUBROUTINE NULL_PS
  RETURN
END SUBROUTINE NULL_PS
#else /**/
SUBROUTINE PSDRAW(IUNIT,ISTRT,ISTOP,IAX,IAY,IAZ,IACOL,IARAD, &
     IHBIX,ISTER,INDRW,FINDRW,IVX,IVY)
  !
  ! THIS ROUTINE GENERATES A POSTSCRIPT FILE FROM THE SORTED ATOM LIST
  !
  !     IUNIT  - FORTRAN UNIT NO. FOR POSTSCRIPT FILE
  !     ISTRT  - START POINTER INTO INDEXS
  !     ISTOP  - STOP    "       "     "
  !     IAX    - ATOM X POSITONS IN SCREEN COORDINATES
  !     IAY    -   "  Y    "     "     "       "
  !     IAZ    -   "  Z    "     "     "       "
  !     IVX    - VECTOR END X POSITONS IN SCREEN COORDINATES
  !     IVY    -   "     "  Y    "     "     "       "
  !     IACOL  - ATOM COLOR REGISTER VALUES
  !     IARAD  - ATOM RADII
  !     IHBIX  - INTERNAL FLAG ARRAY; LAST INDEX OF DONOR, NO. HB
  !     ISTER  - IMAGE TYPE FLAG; -1 = LEFT, 0 = MONO, 1 = RIGHT
  !     INDRW  - INITIAL CALL; BEGIN A NEW PICTURE
  !     FINDRW - FINAL CALL; PICTURE FINISHED, PRINT TITLE, SHOWPAGE
  !
  use chm_kinds
  use dimens_fcm
  use graph
  use stream
  use hbondm
  implicit none

  INTEGER IUNIT,ISTRT,ISTOP,ISTER
  INTEGER IAX(*),IAY(*),IACOL(*),IARAD(*)
  INTEGER IAZ(*), IHBIX(3,*), IVX(*),IVY(*)
  LOGICAL INDRW,FINDRW

  INTEGER IRAD,CCOL,IBD, IMDPT
  INTEGER KXMX,KYMX,KXMN,KYMN
  INTEGER, PARAMETER :: KZER=0
  INTEGER HBWIDTH,ITEMP,JTEMP
  INTEGER IER,J,I,II,IAT,JAT,IX,IY, IS
  INTEGER IZH,IZL,ILEV,NLEV,KFONT, IH, JH
  CHARACTER(len=20) TXTFNT(4),SYMFNT(4)

  TXTFNT(1) = 'TxVS setfont'
  TXTFNT(2) = 'TxSM setfont'
  TXTFNT(3) = 'TxME setfont'
  TXTFNT(4) = 'TxLA setfont'
  SYMFNT(1) = 'SyVS setfont'
  SYMFNT(2) = 'SySM setfont'
  SYMFNT(3) = 'SyME setfont'
  SYMFNT(4) = 'SyLA setfont'

  IF (PRNLEV.GT.5) WRITE(OUTU,34) 1+ISTOP-ISTRT
34 FORMAT(' ',I8,' atoms will be displayed')

  ! determine z=0 point (xy plane); drawing split into 2 loops
  DO II = ISTRT,ISTOP
     IAT=INDEXS(II)
     IF (IAZ(II).GT.0) GOTO 40
  END DO
40 IMDPT = II-1

  ! setup clipping for A size page, used primarily for stereo mode
  IF (QPSORI) THEN
     ! landscape
     KXMX = 3960
     KYMX = 3060
     KXMN = -3960
     KYMN = -3060
  ELSE
     ! portrait
     KXMX = 3060
     KYMX = 3960
     KXMN = -3060
     KYMN = -3960
  ENDIF

  IF (ISTER.EQ.-1) THEN
     ! left eye image
     WRITE (IUNIT,'(A)') 'grestore gsave'
     WRITE (IUNIT,920) KXMN,KYMN,KZER,KYMN,KZER,KYMX,KXMN,KYMX, &
          KXMN,KYMN
  ELSEIF (ISTER.EQ.0) THEN
     ! monoscopic image
     WRITE (IUNIT,920) KXMN,KYMN,KXMX,KYMN,KXMX,KYMX,KXMN,KYMX, &
          KXMN,KYMN
  ELSEIF (ISTER.EQ.1) THEN
     ! right eye image
     WRITE (IUNIT,'(A)') 'grestore gsave'
     WRITE (IUNIT,920) KZER,KYMN,KXMX,KYMN,KXMX,KYMX,KZER,KYMX, &
          KZER,KYMN
  ENDIF

920 FORMAT ('newpath ',2I6,' moveto ',2I6,' lineto ',2I6,' lineto ', &
       2I6,' lineto ',2I6,' lineto closepath clip',/,'newpath')

  ! init for display of hydrogen bonds; first, setup for Z cueing
  IF (QDHBON .AND. NHB.GT.0) THEN
     IF (ZCUEH.GE.ZCUEL) THEN
        IZL=ZCUEL*GRDUPA
        IZH=ZCUEH*GRDUPA
     ELSE
        IZL=IAZ(INDEXS(ISTRT))
        IZH=IAZ(INDEXS(ISTOP))
     ENDIF
     NLEV=IGRZLEV
     ! init array of H-bonds to display; up to 3 per donor
     DO J=1,3
        DO II = ISTRT,ISTOP
           IHBIX(J,II) = 0
        END DO
     END DO
     ! IHB Heavy atom donor
     ! JHB Heavy atom acceptor
     ! KHB Hydrogen in hydrogen bond (can be zero if no hydrogen)
     ! NHB Number of hydrogen bonds
     DO II=1,NHB
        IAT=KHB(II)
        JAT=JHB(II)
        IF(IAT.LE.0) IAT=IHB(II)
        IF(IAT.LE.0) GOTO 155
        IF(JAT.LE.0) GOTO 155
        IAT=INDEXR(IAT)
        JAT=INDEXR(JHB(II))
        IF(IAT.LE.0) GOTO 155
        IF(JAT.LE.0) GOTO 155
        IF(INDEXB(IAT).LT.ISTRT) GOTO 155
        IF(INDEXB(IAT).GT.ISTOP) GOTO 155
        IF(INDEXB(JAT).LT.ISTRT) GOTO 155
        IF(INDEXB(JAT).GT.ISTOP) GOTO 155
        ! the donor is indexed by IAT, the acceptor by JAT
        IF (IHBIX(1,IAT).EQ.0) THEN
           IHBIX(1,IAT) = JAT
        ELSE IF (IHBIX(2,IAT).EQ.0) THEN
           IHBIX(2,IAT) = JAT
        ELSE IF (IHBIX(3,IAT).EQ.0) THEN
           IHBIX(3,IAT) = JAT
        ELSE
           CALL WRNDIE(0,'<PSDRAW>','more than 3 H-bonds per donor')
        ENDIF
155     CONTINUE
     ENDDO
  ENDIF

  !  main drawing section; start by enforcing solid line type
  WRITE (IUNIT,910) IGRWIDTH
  WRITE (IUNIT,'(A)') '[] 0 setdash'

  ! first draw the objects with negative Z atom coord 
  DO IS = ISTRT, IMDPT
     IAT=INDEXS(IS)
     CCOL=IACOL(IAT)
     CALL SETPSCOLR(IUNIT,QPSCLR,QPSBBK,CCOL,COLOR_MAP)

     IF (QDATOM) THEN
        IRAD=IARAD(IAT)
        IF (IRAD.GT.0) THEN
           WRITE(IUNIT,820) IAX(IAT),IAY(IAT),IAX(IAT),IAY(IAT),IRAD
820        FORMAT(2I7,' Mv ',3I7,' At')
        ENDIF
     ENDIF

     IF (QDBOND) THEN
        DO J=1,NATBON(IAT)
           JAT=IATBON(J,IAT)
           IX=(IAX(IAT)+IAX(JAT))/2
           IY=(IAY(IAT)+IAY(JAT))/2
           WRITE (IUNIT,810) IX,IY,IAX(IAT),IAY(IAT)
        END DO
     ENDIF

     IF (QDHBON .AND. NHB.GT.0) THEN
        IF (IHBIX(1,IAT).GT.0) THEN
           ! apply h-bond z shading based on donor Z coord
           IX=IAZ(IAT)+IAZ(IHBIX(1,IAT))
           I=IGHBCO
           IF (IX.LT.IZH) THEN
              IF (IX.LE.IZL) THEN
                 I = I +16*(NLEV-1)
              ELSE
                 ILEV=(NLEV*(IZH-IX))/(IZH-IZL)
                 I = I +16*ILEV
              ENDIF
           ENDIF
           ! set color, width, and dashlength
           CALL SETPSCOLR(IUNIT,QPSCLR,QPSBBK,I,COLOR_MAP)
           WRITE (IUNIT,910) IGHBWI
           IF (IGHBTY.GT.0) WRITE (IUNIT,930) IGHBTY
           ! draw up to 3 H-bonds for this donor
           DO IH = 1,3
              JH = IHBIX(IH,IAT)
              IF (JH.NE.0) THEN
                 WRITE (IUNIT,810) IAX(JH),IAY(JH), IAX(IAT),IAY(IAT)
              ENDIF
           END DO
           ! reset to bond width, solid line
           WRITE (IUNIT,910) IGRWIDTH
           WRITE (IUNIT,'(A)') '[] 0 setdash'
        ENDIF
     ENDIF

     IF (QDVECT) THEN
       CALL SETPSCOLR(IUNIT,QPSCLR,QPSBBK,IVECCO,COLOR_MAP)
       WRITE(IUNIT,910) IVECWI
       WRITE(IUNIT,810) IAX(IAT),IAY(IAT),IVX(IAT),IVY(IAT)
     ENDIF

     IF (QLABEL) THEN
        II=INDEXP(IAT)
        IF (IGRLBL(II).NE.0) THEN
           KFONT=IGRLBL(II)
           I=KFONT/256
           CALL SETPSCOLR(IUNIT,QPSCLR,QPSBBK,I,COLOR_MAP)
           I=MOD(KFONT,256)
           WRITE (IUNIT,'(A)') TXTFNT(I)
           ITEMP=IAX(IAT)+(2+IGRWIDTH)*10
           JTEMP=IAY(IAT)+(2+IGRWIDTH)*10
           WRITE (IUNIT,830) ITEMP,JTEMP,TGRLBL(II)(1:IGRLLN(II))
        ENDIF
     ENDIF

  END DO

  ! show the axes; dotted for negative, positive labeled
  IF (QDAXE) THEN
     WRITE (IUNIT,910) IGRWIDTH
910  FORMAT (I4,' 10 mul setlinewidth')
     CCOL = 7
     CALL SETPSCOLR(IUNIT,QPSCLR,QPSBBK,CCOL,COLOR_MAP)
     WRITE (IUNIT,'(A)') '[] 0 setdash'
     WRITE (IUNIT,810) INT(AXETRN(1,1)), INT(AXETRN(2,1)), &
          INT(AXETRN(1,7)), INT(AXETRN(2,7))
810  FORMAT(4I7,' Ve')
     WRITE (IUNIT,810) INT(AXETRN(1,3)), INT(AXETRN(2,3)), &
          INT(AXETRN(1,7)), INT(AXETRN(2,7))
     WRITE (IUNIT,810) INT(AXETRN(1,5)), INT(AXETRN(2,5)), &
          INT(AXETRN(1,7)), INT(AXETRN(2,7))
     WRITE (IUNIT,930) IGHBTY
930  FORMAT('[',I3,' 10 mul ] 0 setdash')
     WRITE (IUNIT,810) INT(AXETRN(1,2)), INT(AXETRN(2,2)), &
          INT(AXETRN(1,7)), INT(AXETRN(2,7))
     WRITE (IUNIT,810) INT(AXETRN(1,4)), INT(AXETRN(2,4)), &
          INT(AXETRN(1,7)), INT(AXETRN(2,7))
     WRITE (IUNIT,810) INT(AXETRN(1,6)), INT(AXETRN(2,6)), &
          INT(AXETRN(1,7)), INT(AXETRN(2,7))
     WRITE (IUNIT,'(A)') TXTFNT(IGRFONT)
     ! label the positive ends of the axes lines
     WRITE (IUNIT,830) INT(AXETRN(1,1)),INT(AXETRN(2,1)),'X'
830  FORMAT(2I7,' Mv (',A,') show')
     WRITE (IUNIT,830) INT(AXETRN(1,3)),INT(AXETRN(2,3)),'Y'
     WRITE (IUNIT,830) INT(AXETRN(1,5)),INT(AXETRN(2,5)),'Z'
     WRITE (IUNIT,'(A)') 'newpath'
  ENDIF

  ! now draw the objects with positive atom coords
  WRITE (IUNIT,910) IGRWIDTH
  WRITE (IUNIT,'(A)') '[] 0 setdash'
  DO IS = IMDPT+1,ISTOP
     IAT=INDEXS(IS)
     CCOL=IACOL(IAT)
     CALL SETPSCOLR(IUNIT,QPSCLR,QPSBBK,CCOL,COLOR_MAP)

     IF (QDATOM) THEN
        IRAD=IARAD(IAT)
        IF (IRAD.GT.0) THEN
           WRITE(IUNIT,820) IAX(IAT),IAY(IAT),IAX(IAT),IAY(IAT),IRAD
        ENDIF
     ENDIF

     IF (QDBOND) THEN
        DO J=1,NATBON(IAT)
           JAT=IATBON(J,IAT)
           IX=(IAX(IAT)+IAX(JAT))/2
           IY=(IAY(IAT)+IAY(JAT))/2
           WRITE (IUNIT,810) IX,IY,IAX(IAT),IAY(IAT)
        END DO
     ENDIF

     IF (QDHBON .AND. NHB.GT.0) THEN
        IF (IHBIX(1,IAT).GT.0) THEN
           ! apply h-bond z shading based on donor Z coord
           IX=IAZ(IAT)+IAZ(IHBIX(1,IAT))
           I=IGHBCO
           IF (IX.LT.IZH) THEN
              IF (IX.LE.IZL) THEN
                 I = I +16*(NLEV-1)
              ELSE
                 ILEV=(NLEV*(IZH-IX))/(IZH-IZL)
                 I = I +16*ILEV
              ENDIF
           ENDIF
           CALL SETPSCOLR(IUNIT,QPSCLR,QPSBBK,I,COLOR_MAP)
           WRITE (IUNIT,910) IGHBWI
           IF (IGHBTY.GT.0) WRITE (IUNIT,930) IGHBTY
           ! draw up to 3 H-bonds for this donor
           DO IH = 1,3
              JH = IHBIX(IH,IAT)
              IF (JH.NE.0) THEN
                 WRITE (IUNIT,810) IAX(JH),IAY(JH), IAX(IAT),IAY(IAT)
              ENDIF
           END DO
           WRITE (IUNIT,910) IGRWIDTH
           WRITE (IUNIT,'(A)') '[] 0 setdash'
        ENDIF
     ENDIF

     IF (QDVECT) THEN
       CALL SETPSCOLR(IUNIT,QPSCLR,QPSBBK,IVECCO,COLOR_MAP)
       WRITE(IUNIT,910) IVECWI
       WRITE(IUNIT,810) IAX(IAT),IAY(IAT),IVX(IAT),IVY(IAT)
     ENDIF

     IF (QLABEL) THEN
        II=INDEXP(IAT)
        IF (IGRLBL(II).NE.0) THEN
           KFONT=IGRLBL(II)
           I=KFONT/256
           CALL SETPSCOLR(IUNIT,QPSCLR,QPSBBK,I,COLOR_MAP)
           I=MOD(KFONT,256)
           WRITE (IUNIT,'(A)') TXTFNT(I)
           ITEMP=IAX(IAT)+(2+IGRWIDTH)*10
           JTEMP=IAY(IAT)+(2+IGRWIDTH)*10
           WRITE (IUNIT,830) ITEMP,JTEMP,TGRLBL(II)(1:IGRLLN(II))
        ENDIF
     ENDIF

  END DO

  !
  ! final stuff before finishing page
  !
  IF (FINDRW.AND.QERASE) THEN
     WRITE (IUNIT,'(A)') 'grestore'
     IF (QTITLE) THEN
        WRITE (IUNIT,'(A)') TXTFNT(IGRFONT)
        I=5
        CALL SETPSCOLR(IUNIT,QPSCLR,QPSBBK,I,COLOR_MAP)
        WRITE (IUNIT,830) KXMN+320,KYMN+320, GTITLE(1:IGRTLEN)
     ENDIF
     WRITE (IUNIT,'(A)') 'showpage'
     WRITE (IUNIT,'(A)') '%%Trailer'
     WRITE (IUNIT,'(A)') 'end CHARMMjob restore'
  ENDIF
  !
  RETURN
END SUBROUTINE PSDRAW

SUBROUTINE PSCETRM(IUNIT)
  !
  ! terminate PS file in ERAse OFF mode; PSC UNIT N TERM
  !

  use chm_kinds
  use dimens_fcm
  use graph
  implicit none

  INTEGER IUNIT
  LOGICAL QERROR
  INTEGER KXMN, KXMX, KYMN, KYMX, I
  CHARACTER(len=20) TXTFNT(4),SYMFNT(4)

  TXTFNT(1) = 'TxVS setfont'
  TXTFNT(2) = 'TxSM setfont'
  TXTFNT(3) = 'TxME setfont'
  TXTFNT(4) = 'TxLA setfont'
  SYMFNT(1) = 'SyVS setfont'
  SYMFNT(2) = 'SySM setfont'
  SYMFNT(3) = 'SyME setfont'
  SYMFNT(4) = 'SyLA setfont'

  IF (QPSORI) THEN
     ! landscape
     KXMX = 3960
     KYMX = 3060
     KXMN = -3960
     KYMN = -3060
  ELSE
     ! portrait
     KXMX = 3060
     KYMX = 3960
     KXMN = -3060
     KYMN = -3960
  ENDIF

  WRITE (IUNIT,'(A)') 'grestore'
  IF (QTITLE) THEN
     WRITE (IUNIT,'(A)') TXTFNT(IGRFONT)
     I=5
     CALL SETPSCOLR(IUNIT,QPSCLR,QPSBBK,I,COLOR_MAP)
     WRITE (IUNIT,830) KXMN+320,KYMN+320, GTITLE(1:IGRTLEN)
830  FORMAT(2I7,' Mv (',A,') show')
  ENDIF
  WRITE (IUNIT,'(A)') 'showpage'
  WRITE (IUNIT,'(A)') '%%Trailer'
  WRITE (IUNIT,'(A)') 'end CHARMMjob restore'
  CALL VCLOSE(IUNIT,'KEEP',QERROR)

  RETURN
END SUBROUTINE PSCETRM


SUBROUTINE PSCINIT(IUNIT,QINIT)
  !
  !  modify USCREN for postscript device coordinates
  !  send initialization sequence to IUNIT
  !  qinit flag added for 'erase off' option; true on first call only
  !  qpscolr flag is true for color postscript (graph.f90)
  !  qpsorie flag is true for landscape format (graph.f90)
  !
  use chm_kinds
  use dimens_fcm
  use graph
  use stream
  use startup,only:getnam,sysid
  use machutil,only:daytim

  implicit none

  INTEGER IUNIT,IMO,IDA,IYR,IHR,IMI,ISE,I,J
  CHARACTER(len=16) CUSERNAM
  CHARACTER(len=80) CSYSNAM
  CHARACTER(len=32) TIMBUF
  CHARACTER(len=1) CSL,CSP,CCO
  LOGICAL QINIT

  !
  ! roughly 28 decipoints per cm; preserve initial 1 cm / A scaling
  ! scaled by 10 for better resolution with integer coords (280 units/A)
  GRDUPA=280.d0
  DO I=1,4
     DO J=1,4
        USCREN(J,I) = 0.d0
     ENDDO
  ENDDO
  USCREN(1,1)=280.d0
  USCREN(2,2)=280.d0
  USCREN(3,3)=280.d0
  USCREN(4,4)=1.d0
  !
  IF(IOLEV.LT.0) RETURN
  !
  IF (QINIT) THEN
     CSL = '/'
     CSP = ' '
     CCO = ':'
     WRITE (IUNIT,'(A)') '%!PS-Adobe-1.0'
     WRITE (IUNIT,'(A)') '%%DocumentFonts: Times-Bold Symbol'
     WRITE (IUNIT,'(A)') '%%Title: '//GTITLE(1:IGRTLEN)
     CALL GETNAM(CUSERNAM)
     CALL SYSID(CSYSNAM)
     WRITE (IUNIT,'(A)') '%%Creator: CHARMM '//CUSERNAM//CSYSNAM
     CALL DAYTIM(IMO,IDA,IYR,IHR,IMI,ISE)
     WRITE (TIMBUF,'(I2,A,I2,A,I2,A,I2,A,I2,A,I2)') &
          IMO,CSL,IDA,CSL,IYR,CSP,IHR,CCO,IMI,CCO,ISE
     WRITE (IUNIT,'(A)') '%%CreationDate: '//TIMBUF
     WRITE (IUNIT,'(A)') '%%For: '//CUSERNAM
     WRITE (IUNIT,'(A)') '%%Pages: 1'
     WRITE (IUNIT,'(A)') '%%BoundingBox: 0 0 612 792'
     WRITE (IUNIT,'(A)') '%%EndComments'
     WRITE (IUNIT,'(A)') 'save /CHARMMjob exch def'
     WRITE (IUNIT,'(A)') '50 dict begin'
     WRITE (IUNIT,'(A)') '.1 .1 scale'
     WRITE (IUNIT,'(A)') '3060 3960 translate'
     ! make a black background for color w/o reversal
     IF (QPSCLR.AND.QPSBBK) WRITE (IUNIT,918)
918  FORMAT ('newpath 0. 0. 0. setrgbcolor ',/, &
          '-3060 -3960 moveto 3060 -3960 lineto 3060 3960 lineto ', &
          '-3060 3960 lineto -3060 -3960 lineto closepath fill',/, &
          'newpath')
     IF (QPSORI) THEN
        WRITE (IUNIT,'(A)') '90 rotate'
     ENDIF
     WRITE (IUNIT,'(A)') '1 setlinecap'
     WRITE (IUNIT,'(A)') '1 setlinejoin'
     !     postscript macro defs, esp. fonts
     WRITE (IUNIT,'(A)') '/Ve { moveto lineto stroke } def'
     WRITE (IUNIT,'(A)') '/At { 0 360 arc fill } def'
     WRITE (IUNIT,'(A)') '/Mv { moveto } def'
     WRITE (IUNIT,'(A)') '/Sg { setgray } def'
     WRITE (IUNIT,'(A)') '/Sc { setrgbcolor } def'
     WRITE (IUNIT,'(A)') &
          '/TxVS /Times-Bold findfont 120 scalefont def'
     WRITE (IUNIT,'(A)') &
          '/TxSM /Times-Bold findfont 140 scalefont def'
     WRITE (IUNIT,'(A)') &
          '/TxME /Times-Bold findfont 180 scalefont def'
     WRITE (IUNIT,'(A)') &
          '/TxLA /Times-Bold findfont 240 scalefont def'
     WRITE (IUNIT,'(A)') &
          '/SyVS /Symbol findfont 120 scalefont def'
     WRITE (IUNIT,'(A)') &
          '/SySM /Symbol findfont 140 scalefont def'
     WRITE (IUNIT,'(A)') &
          '/SyME /Symbol findfont 180 scalefont def'
     WRITE (IUNIT,'(A)') &
          '/SyLA /Symbol findfont 240 scalefont def'
     WRITE (IUNIT,'(A)') 'gsave'
     WRITE (IUNIT,'(A)') '%%EndProlog'
     WRITE (IUNIT,'(A)') '%%Page: 0 1'
  ENDIF
  RETURN
END SUBROUTINE PSCINIT

SUBROUTINE SETPSCOLR(IUNIT,QCOLR,QBBK,ICOLR,MAP)
  !
  use chm_kinds
  use stream
  implicit none
  !
  LOGICAL QCOLR,QBBK
  INTEGER ICOLR,MAP(0:*),MC,IUNIT,J100, LCOLR
  real(chm_real) RED,GREEN,BLUE, SGRY, DFF,A

  DATA LCOLR /-1/

  DFF = 255.0
  J100 = 256
  IF (ICOLR.LT.0 .OR. ICOLR.GT.191) WRITE(OUTU,951) ICOLR
951 FORMAT ('Bad color index passed to SETPSCOLR: ',I12)
  ! suppress setrgbcolor or setgray when no color change
  IF (ICOLR.EQ.LCOLR) RETURN
  LCOLR=ICOLR
  MC = MAP(ICOLR)
  BLUE = REAL( MOD(MC,J100)) / DFF
  MC = MC / J100
  GREEN = REAL( MOD(MC,J100)) / DFF
  MC = MC / J100
  RED = REAL( MOD(MC,J100)) / DFF
  MC = MOD(ICOLR,16)
  ! inversion of whites and grays (for white background or B&W mode)
  IF ( (.NOT. QBBK .OR. .NOT. QCOLR) .AND.  &
       (MC.EQ.2 .OR. MC.EQ.5 .OR. MC.EQ.9)) THEN
     RED = 1.D0 - RED
     GREEN = 1.D0 - GREEN
     BLUE = 1.D0 - BLUE
  ENDIF
  IF (QCOLR) THEN
     WRITE (IUNIT,901) RED,GREEN,BLUE
901  FORMAT(3F9.3,' Sc')
  ELSE
     ! apply the NTSC formula to convert color to grayscale
     SGRY = 0.299*RED + 0.587*GREEN + 0.114*BLUE
     WRITE (IUNIT,902) SGRY
902  FORMAT(F7.3,' Sg')
  ENDIF
  RETURN
END SUBROUTINE SETPSCOLR
#endif 

