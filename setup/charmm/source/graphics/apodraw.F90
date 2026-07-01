#if KEY_NOGRAPHICS==1 || KEY_NODISPLAY==1 /*nono*/
SUBROUTINE NULL_AD
  return
end SUBROUTINE NULL_AD
#else /* (nono)*/
SUBROUTINE DRAWT3(ISTRT,ISTOP,IAX,IAY,IAZ,IACOL,IARAD, &
     IHBIX,ISTER,INDRW,FINDRW,IVX,IVY)
  !
  ! THIS ROUTINE DISPLAYS THE SORTED ATOM LIST
  !
  !     ISTRT - START POINTER INTO INDEXS
  !     ISTOP - STOP    "       "     "
  !     IAX   - ATOM X POSITONS IN SCREEN COORDINATES
  !     IAY   -   "  Y    "     "     "       "
  !     IAZ   -   "  Z    "     "     "       "
  !     IVX   - VECTOR X OFFSET IN SCREEN COORDINATES
  !     IVY   -    "   Y    "    "    "       "
  !     IACOL - ATOM COLOR REGISTER VALUES
  !     IARAD - ATOM RADII
  !     IHBIX - INTERNAL ARRAY; HBOND ATOM INDICES
  !     ISTER - IMAGE TYPE FLAG; -1=STEREO LEFT, 0=MONO, 1=STEREO RIGHT
  !     INDRW - INITIAL CALL; BEGIN A NEW PICTURE
  !     FINDRW- FINAL CALL; PICTURE FINISHED (SWAP BUFFERS)
  !
  use chm_kinds
  use dimens_fcm
  use graph
  use xdraw
  use stream
  use hbondm
  implicit none

  INTEGER ISTRT,ISTOP,ISTER
  INTEGER IAX(*),IAY(*),IACOL(*),IARAD(*)
  INTEGER IAZ(*), IHBIX(3,*)
  INTEGER :: IVX(*),IVY(*) ! comp vector args :: rvenable
  LOGICAL INDRW,FINDRW

  INTEGER MAXBUF,LIMBUF
  PARAMETER (MAXBUF=512,LIMBUF=480)
#if KEY_XDISPLAY==1
  INTEGER*2 XSEGM(4,MAXBUF)
#endif 
  INTEGER*4 IBD,ISHFT
  INTEGER IER,J,I,II,GRBUFF,NOGRBFF,IAT,JAT,IX,IY, IS
  INTEGER IZH,IZL,ILEV,NLEV,KFONT,CCOL, IH, JH, IMDPT
  !
#if KEY_XDISPLAY==0 /*main_xdisplay*/
  CALL WRNDIE(-1,'DRAWT3>','Screen drawing only under X11.')
  RETURN
#else /* (main_xdisplay)*/
  !
  !c      WRITE(OUTU,34) 1+ISTOP-ISTRT
  !c  34  FORMAT(' ',I8,' atoms will be displayed')

  ! determine z=0 point (xy plane); drawing split into 2 loops
  DO II = ISTRT,ISTOP
     IAT=INDEXS(II)
     IF (IAZ(II).GT.0) GOTO 40
  END DO
40 IMDPT = II-1

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
           CALL WRNDIE(0,'DRAWT3>','more than 3 H-bonds per donor')
        ENDIF
155     CONTINUE
     ENDDO
  ENDIF

  !
  !  set to draw in undisplayed buffer
  !  main drawing section
  !
#if KEY_XDISPLAY==1
  IF(INDRW.AND.QERASE) CALL XCLEAR
  !  X clipping based on the stereo flag
  CALL XCLIPDRW(ISTER)
  ! reset to bond width, solid line
  CALL XLINEWIDTH(IGRWIDTH)
  CALL XLINESTYLE(0)
#endif 
  CCOL=-1
  ! first draw the objects with negative Z atom coord 
  DO IS = ISTRT, IMDPT
     IAT=INDEXS(IS)
     CCOL=IACOL(IAT)
#if KEY_XDISPLAY==1
     CALL XCOLOR(CCOL)
#endif 
     ! display atom if enabled
     IF (QDATOM) THEN
        IF (IARAD(IAT).GT.0) THEN
#if KEY_XDISPLAY==1
           CALL XFARC(IAX(IAT),IAY(IAT),IARAD(IAT))
#endif 
        ENDIF
     ENDIF

     ! display bonds for this atom
     IF (QDBOND) THEN
        IBD = 0
        DO J=1,NATBON(IAT)
           JAT=IATBON(J,IAT)
           IX=(IAX(IAT)+IAX(JAT))/2
           IY=(IAY(IAT)+IAY(JAT))/2
           IBD=IBD+1
#if KEY_XDISPLAY==1
           XSEGM(1,IBD)=IAX(IAT)
           XSEGM(2,IBD)=IAY(IAT)
           XSEGM(3,IBD)=IX
           XSEGM(4,IBD)=IY
#endif 
        END DO
#if KEY_XDISPLAY==1
        CALL XMULTILINE(XSEGM,IBD)
#endif 
     ENDIF

     ! display H-bonds if the list has any entries and if enabled
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
#if KEY_XDISPLAY==1
           CALL XCOLOR(I)
           CALL XLINEWIDTH(IGHBWI)
           IF(IGHBTY.GT.0) CALL XLINESTYLE(1)
#endif 
           ! draw up to 3 H-bonds for this donor
           IBD = 0
           DO IH = 1,3
              JH = IHBIX(IH,IAT)
              IF (JH.NE.0) THEN
                 IBD=IBD+1
#if KEY_XDISPLAY==1
                 XSEGM(1,IBD)=IAX(IAT)
                 XSEGM(2,IBD)=IAY(IAT)
                 XSEGM(3,IBD)=IAX(JH)
                 XSEGM(4,IBD)=IAY(JH)
#endif 
              ENDIF
           END DO
#if KEY_XDISPLAY==1
           CALL XMULTILINE(XSEGM,IBD)
           ! reset to bond width, solid line
           CALL XLINEWIDTH(IGRWIDTH)
           CALL XLINESTYLE(0)
#endif 
        ENDIF
     ENDIF

     ! finally, display the labels if enabled
     IF (QLABEL) THEN
        II=INDEXP(IAT)
        IF (IGRLBL(II).NE.0) THEN
           KFONT=IGRLBL(II)
           I=KFONT/256
#if KEY_XDISPLAY==1
           CALL XCOLOR(I)
#endif 
           I=MOD(KFONT,256)
#if KEY_XDISPLAY==1
           CALL XSELFONT(I)
           CALL XTEXT(TGRLBL(II),IGRLLN(II),IAX(IAT)+2+IGRWIDTH, &
                IAY(IAT)-(2+IGRWIDTH))
#endif 
        ENDIF
     ENDIF

     ! Tim Miller, let's draw vectors if enabled
     IF (QDVECT) THEN
        XSEGM(1,1)=IAX(IAT)
        XSEGM(2,1)=IAY(IAT)
        XSEGM(3,1)=IVX(IAT)
        XSEGM(4,1)=IVY(IAT)
#if KEY_XDISPLAY==1
        CALL XCOLOR(IVECCO)
        CALL XLINEWIDTH(IVECWI)
        IBD = 1
        CALL XMULTILINE(XSEGM,IBD)
        CALL XLINEWIDTH(IGRWIDTH)
#endif 
     ENDIF

  END DO

  ! show the axes; dotted for negative, positive labeled
  IF (QDAXE) THEN
     CCOL = IGAXCO
#if KEY_XDISPLAY==1
     ! setup lines in GC
     CALL XLINEWIDTH(IGRWIDTH)
     !       0 - solid, >0 dashed
     CALL XLINESTYLE(0)
     CALL XCOLOR(CCOL)
     XSEGM(1,1) = AXETRN(1,7)
     XSEGM(2,1) = AXETRN(2,7)
     XSEGM(3,1) = AXETRN(1,1)
     XSEGM(4,1) = AXETRN(2,1)
     XSEGM(1,2) = AXETRN(1,7)
     XSEGM(2,2) = AXETRN(2,7)
     XSEGM(3,2) = AXETRN(1,3)
     XSEGM(4,2) = AXETRN(2,3)
     XSEGM(1,3) = AXETRN(1,7)
     XSEGM(2,3) = AXETRN(2,7)
     XSEGM(3,3) = AXETRN(1,5)
     XSEGM(4,3) = AXETRN(2,5)
     IBD = 3
     CALL XMULTILINE(XSEGM,IBD)
     CALL XLINESTYLE(1)
     !
     XSEGM(1,1) = AXETRN(1,7)
     XSEGM(2,1) = AXETRN(2,7)
     XSEGM(3,1) = AXETRN(1,2)
     XSEGM(4,1) = AXETRN(2,2)
     XSEGM(1,2) = AXETRN(1,7)
     XSEGM(2,2) = AXETRN(2,7)
     XSEGM(3,2) = AXETRN(1,4)
     XSEGM(4,2) = AXETRN(2,4)
     XSEGM(1,3) = AXETRN(1,7)
     XSEGM(2,3) = AXETRN(2,7)
     XSEGM(3,3) = AXETRN(1,6)
     XSEGM(4,3) = AXETRN(2,6)
     IBD = 3
     CALL XMULTILINE(XSEGM,IBD)
     CALL XLINESTYLE(0)
     ! Put labels on axes
     CCOL = IGAXCO
     CALL XCOLOR(CCOL)
     CALL XSELFONT(IGRFONT)
     II=1
     CALL XTEXT('X',II,INT(AXETRN(1,1)),INT(AXETRN(2,1)))
     CALL XTEXT('Y',II,INT(AXETRN(1,3)),INT(AXETRN(2,3)))
     CALL XTEXT('Z',II,INT(AXETRN(1,5)),INT(AXETRN(2,5)))
#endif 
  ENDIF

  ! reset to bond width, solid line
#if KEY_XDISPLAY==1
  CALL XLINEWIDTH(IGRWIDTH)
  CALL XLINESTYLE(0)
#endif 
  ! now draw the objects with positive atom coords
  DO IS = IMDPT+1,ISTOP
     IAT=INDEXS(IS)
     CCOL=IACOL(IAT)
#if KEY_XDISPLAY==1
     CALL XCOLOR(CCOL)
#endif 
     ! display atom if enabled
     IF (QDATOM) THEN
        IF (IARAD(IAT).GT.0) THEN
#if KEY_XDISPLAY==1
           CALL XFARC(IAX(IAT),IAY(IAT),IARAD(IAT))
#endif 
        ENDIF
     ENDIF

     ! display bonds for this atom if enabled
     IF (QDBOND) THEN
        IBD = 0
        DO J=1,NATBON(IAT)
           JAT=IATBON(J,IAT)
           IX=(IAX(IAT)+IAX(JAT))/2
           IY=(IAY(IAT)+IAY(JAT))/2
           IBD=IBD+1
#if KEY_XDISPLAY==1
           XSEGM(1,IBD)=IAX(IAT)
           XSEGM(2,IBD)=IAY(IAT)
           XSEGM(3,IBD)=IX
           XSEGM(4,IBD)=IY
#endif 
        END DO
#if KEY_XDISPLAY==1
        CALL XMULTILINE(XSEGM,IBD)
#endif 
     ENDIF

     ! display H-bonds if the list has any entries and if enabled
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
#if KEY_XDISPLAY==1
           CALL XCOLOR(I)
           CALL XLINEWIDTH(IGHBWI)
           IF(IGHBTY.GT.0) CALL XLINESTYLE(1)
#endif 
           ! draw up to 3 H-bonds for this donor
           IBD = 0
           DO IH = 1,3
              JH = IHBIX(IH,IAT)
              IF (JH.NE.0) THEN
                 IBD=IBD+1
#if KEY_XDISPLAY==1
                 XSEGM(1,IBD)=IAX(IAT)
                 XSEGM(2,IBD)=IAY(IAT)
                 XSEGM(3,IBD)=IAX(JH)
                 XSEGM(4,IBD)=IAY(JH)
#endif 
              ENDIF
           END DO
#if KEY_XDISPLAY==1
           CALL XMULTILINE(XSEGM,IBD)
#endif 
           ! reset to bond width, solid line
#if KEY_XDISPLAY==1
           CALL XLINEWIDTH(IGRWIDTH)
           CALL XLINESTYLE(0)
#endif 
        ENDIF
     ENDIF

     ! finally, display the labels if enabled
     IF (QLABEL) THEN
        II=INDEXP(IAT)
        KFONT=IGRLBL(II)
        IF (KFONT.NE.0) THEN
           I=KFONT/256
#if KEY_XDISPLAY==1
           CALL XCOLOR(I)
#endif 
           I=MOD(KFONT,256)
#if KEY_XDISPLAY==1
           CALL XSELFONT(I)
           CALL XTEXT(TGRLBL(II),IGRLLN(II),IAX(IAT)+2+IGRWIDTH, &
                IAY(IAT)-(2+IGRWIDTH))
#endif 
        ENDIF
     ENDIF

     ! Tim Miller, let's draw vectors if enabled
     IF (QDVECT) THEN
        XSEGM(1,1)=IAX(IAT)
        XSEGM(2,1)=IAY(IAT)
        XSEGM(3,1)=IVX(IAT)
        XSEGM(4,1)=IVY(IAT)
#if KEY_XDISPLAY==1
        CALL XCOLOR(IVECCO)
        CALL XLINEWIDTH(IVECWI)
        IBD=1
        CALL XMULTILINE(XSEGM,IBD)
        CALL XLINEWIDTH(IGRWIDTH)
#endif 
     ENDIF

  END DO

  !
  !  now display the buffered image via swap or copy
  !
  IF (FINDRW) THEN
#if KEY_XDISPLAY==1
     !  Reset clipping to the whole drawable area (mono)
     CALL XCLIPDRW(0)
#endif 
     IF (QTITLE) THEN
#if KEY_XDISPLAY==1
        CALL XSELFONT(IGRFONT)
        CALL XCOLOR(IGTICO)
        CALL XTEXT(GTITLE,IGRTLEN,25,XYSZ-25)
#endif 
     ENDIF
#if KEY_XDISPLAY==1
     CALL XSHOWUP()
#endif 
     !     FIN ! ok to swap or copy; findrw=true
  ENDIF
  !
  RETURN
#endif /* (main_xdisplay)*/
END SUBROUTINE DRAWT3
#endif /* (nono)*/

