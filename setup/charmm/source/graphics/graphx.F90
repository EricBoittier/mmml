#if KEY_NOGRAPHICS==1 /*nographx*/
! trap for call to GRAPHX when the code wasn't included by pref.dat
SUBROUTINE GRAPHX
  CALL WRNDIE(1,'<GRAPHX>','Graphics and plot code not compiled.')
  RETURN
END SUBROUTINE GRAPHX
#else /* (nographx)*/
SUBROUTINE GRAPHX
  !
  ! setup vertex storage arrays on HEAP for POV objects
  ! no code trap kept in first routine in file
  !         <*> January 1998 <*> RM Venable <*>
  !
  use chm_kinds
  use dimens_fcm
  use psf
  use exfunc
  use comand
  use graph
  use memory
  use string

  implicit none

  ! default NVERtex for POV objects is the number of atoms; 4 col array
  HNPOVRTX = GTRMI(COMLYN,COMLEN,'NVER',NATOM)
  call chmalloc('graphx.src','GRAPHX','HPOVRTX',4*HNPOVRTX,crl=HPOVRTX)
  call chmalloc('graphx.src','GRAPHX','HTRNVRTX',4*HNPOVRTX,crl=HTRNVRTX)
  call GRPHX2(HPOVRTX,HTRNVRTX,HNPOVRTX)
  call chmdealloc('graphx.src','GRAPHX','HPOVRTX',4*HNPOVRTX,crl=HPOVRTX)
  call chmdealloc('graphx.src','GRAPHX','HTRNVRTX',4*HNPOVRTX,crl=HTRNVRTX)
  RETURN
END SUBROUTINE GRAPHX

SUBROUTINE GRPHX2(POVRTX,TRNVRTX,NPOVRTX)

  !
  ! This set of routines provides CHARMM the capability of displaying
  ! molecular structures when run on a graphics workstation.
  !     Bernard R. Brooks - January,1988
  !
  !                 <*> August 1993 <*> RM Venable <*>
  ! SG stuff added, display portion only (kept Fortran HPGL, LIGHT, etc.)
  ! adopted axes display from SG code (anonymous donation via RJL)
  ! pseudo device for NODISPLAY mode, any file creation w/o Apollo device
  !
  !          <*> November 1993 <*> RM Venable, M Hodoscek <*>
  ! X11 display window added, developed under HP-UX; Apollo GPR model
  ! followed, using roughly equivalent cover routines in xdisp.c
  ! PostScript level 1 output file also developed using the same model;
  ! color and greyscale, portrait and landscape
  !
  !                <*> November 1994 <*> RM Venable <*>
  ! xdisp.c modified extensively; double-buffering, clipping, StaticColor,
  ! symbol fonts, window title, modified colormap calls; apodraw.src
  ! changed to accomodate clipping and double-buffering
  !
  ! misc. bug fixes; axis labels in X; added GLDISPLAY pref.dat keyword
  !
  !                <*> December 1997 <*> RM Venable <*>
  ! removed legacy Apollo (APOLLO) and HPGL code
  ! removed marginal SGI code (GLDISPLAY)
  ! added POV-Ray output format
  ! changed rendering model; draw all Z-sorted objects in one pass
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  !
  use psf
  use param
  use hbondm
  use graph
  use xdraw
  use stream
  use comand
  use coord
  use coordc
  use corman_mod, only: corcom
  use consta
  use ctitla
  use traj_mod,only:reatrj,wrttrj,trajio
  use string
  use memory
  use chutil
  use intcor_module,only:intcr2
  use select

  implicit none

  !
  !       10        20        30        40        50        60        70
  !---+----|----+----|----+----|----+----|----+----|----+----|----+----|--
  !
  integer,allocatable,dimension(:) :: ISLCT
  INTEGER I,II,J,IREP,NREP,LBLSZ,LBLLEN,LBLATM,LBLCLR
  INTEGER IUNIT,NISLCT,JSLCT,K,L,LWTEXT,IWRP
  real(chm_real) XCGR,YCGR,ZCGR,SCGR,CST,SNT,THETA,R
  real(chm_real) UTEMP(4,4),UMOD(4,4),WGT,WGTSUM
  CHARACTER(len=4) WRD
  CHARACTER(len=8) TSEG,TRES,TRSN,TTYP
  CHARACTER(len=8) USRLBL
  CHARACTER(len=3) WRD3
  CHARACTER(len=12) WTEXT
  LOGICAL LUSED,EOF,OK,QLAB,REDRAW,QCOMP,QOFF,QDEBUG,QMASS,QFRST
  LOGICAL QLSEGI,QLRESI,QLRESN,QLTYPE,QLCHEM,QLCHAR,QLWEIG,QSELE
  LOGICAL QLUSER,QLATNU
  !
  ! vertex array for POV objects; uses HEAP for user expansion
  !         <*> January 1998 <*> RM Venable <*>
  !
  INTEGER NPOVRTX
  real(chm_real) POVRTX(4,NPOVRTX), TRNVRTX(4,NPOVRTX)
  !SB: the usual Watcom (il)logic. There are SAVE statements in graph.f90
  SAVE
  !
  !
  QPSCR=.FALSE.
  QPOVR=.FALSE.
  QPLUTO=.FALSE.
  QNOWIN=.FALSE.
  QDAXE=.FALSE.
  EOF=.FALSE.
  IGAXCO=7
  IGTICO=5
  IGHBCO=10
  IGHBWI=4
  IGHBTY=4
  NPVOBJ = 0
  !
  IF(.NOT.QGRDEV) THEN
     ! process-on-command
     ! set all flags to default values
     QGRDEV=.TRUE.
     QFULLS=.FALSE.
     QDMAIN=.TRUE.
     QTITLE=.TRUE.
     GTITLE = TITLEA(1)
     IGRTLEN = LEN(GTITLE)
     CALL TRIMA(GTITLE,IGRTLEN)
     QDCOMP=.FALSE.
     QDVECT=.FALSE.
     QDVEHD=.FALSE.
     QSTERO=.FALSE.
     TSTERO=7.0
     ! process-reset-command
     CALL RESETVIEW(ULAB,UMOD)
     GRSCAL=1.0
     ZCLPL=-999.0
     ZCLPH=999.0
     ZCUEL=-999.0
     ZCUEH=-999.0
#if KEY_XDISPLAY==1 /*xdisp1*/
     IF (INDXA(COMLYN,COMLEN,'NOWI').GT.0) QNOWIN=.TRUE.
     IF (QNOWIN) THEN
        CALL GRSETUP(.TRUE.)
     ELSE
        XPLA=GTRMI(COMLYN,COMLEN,'XPLA',8)
        XXSZ=GTRMI(COMLYN,COMLEN,'XXSZ',800)
        XYSZ=GTRMI(COMLYN,COMLEN,'XYSZ',800)
        IWRP=GTRMI(COMLYN,COMLEN,'SNAP',0)
        CALL XSETWARP(IWRP)
        CALL GRPHINQ(QFULLS)
        CALL GRPHINIT(QFULLS,.TRUE.)
     ENDIF
#else /* (xdisp1)*/
     CALL GRSETUP(.TRUE.)
#endif /* (xdisp1)*/

     ! assign-default-colors
     IVECCO=3
     IVECWI=1
     CALL ASSDEFCOLOR(NATOM,ATYPE,ICOLOR,ICOLRC)
     CALL ASSDEFRADII(NATOM,ATYPE,RADII)
     IGRASIZ=1.0
     IGRBSIZ=0.0
     !
     ! create-the-transformation-matrix
     CALL CREATETRANS(QDEBUG)
     QZAUTO=.FALSE.
     QDATOM=.FALSE.
     QDBOND=.TRUE.
     QDHBON=.FALSE.

     ! process-atom-selection
     NGRSEL=NATOM
     DO I=1,NATOM
        IGRSEL(I)=1
     ENDDO
     CALL GRAPNB(NATBON,IATBON,IGRSEL,INDEXR,INDEXP,NGRSEL)
#if KEY_XDISPLAY==1 /*xdisp2*/
  ELSE
     IF (INDXA(COMLYN,COMLEN,'NOWI').GT.0) QNOWIN=.TRUE.
     IF (QNOWIN) CALL GRSETUP(.FALSE.)
#endif /* (xdisp2)*/
  ENDIF
  QAUTO=.TRUE.
  QERASE=.TRUE.
  REDRAW=.FALSE.
  QDEBUG=.FALSE.
  GOTO 110
  !
  !     GRAPHICS command parsing loop
  !
100 CONTINUE
  CALL XTRANE(COMLYN,COMLEN,'GRAPHX')
105 CONTINUE
  CALL RDCMND(COMLYN,MXCMSZ,COMLEN,ISTRM,EOF,.TRUE., &
       .TRUE.,'GRAPHX> ')
  CALL MISCOM(COMLYN,MXCMSZ,COMLEN,LUSED)
  IF(LUSED.AND..NOT.EOF) GOTO 105
  !
  IF(EOF) THEN
     IF (NSTRM.EQ.1) THEN
        ! just exit the command parser. leave everything intact
        RETURN
     ENDIF
     CALL PPSTRM(OK)
     EOF=.FALSE.
     GOTO 100
  ENDIF
110 CONTINUE
  WRD=NEXTA4(COMLYN,COMLEN)
  WRD3=WRD
  !-----------------------------------------------------------------------
  !-----------------------------------------------------------------------
  IF(WRD.EQ.'    ') THEN
     GOTO 100
     !-----------------------------------------------------------------------
  ELSE IF(WRD.EQ.'IC  ') THEN
     ! process-internal-coord-commands
     !
     IUNIT=ISTRM
     NISLCT=NATOM
     call chmalloc('graphx.src','GRPHX2','ISLCT',NISLCT,intg=ISLCT)
     QCOMP=(INDXA(COMLYN,COMLEN,'COMP').GT.0)
     IF(QCOMP) THEN
        IF(PRNLEV.GE.3) WRITE(OUTU,237)
237     FORMAT(' THE COMPARISON COORDINATES WILL BE USED')
!old:        call INTCR2(XCOMP,YCOMP,ZCOMP,WCOMP,X,Y,Z,WMAIN, &
!old:             COMLYN,COMLEN,CBB,CTB,NATC,KCB,KCT,NCB,NCT,ISLCT,ATC)
        call INTCR2(XCOMP,YCOMP,ZCOMP,WCOMP,X,Y,Z,WMAIN,COMLYN,COMLEN,ISLCT)
     ELSE
        call INTCR2(X,Y,Z,WMAIN,XCOMP,YCOMP,ZCOMP,WCOMP,COMLYN,COMLEN,ISLCT)
     ENDIF
     call chmdealloc('graphx.src','GRPHX2','ISLCT',NISLCT,intg=ISLCT)
     REDRAW=QAUTO
     !-----------------------------------------------------------------------
  ELSE IF(WRD.EQ.'COOR') THEN
     ! process-coord-manipulation-commands
     !
     CALL CORCOM(COMLYN,COMLEN)
     ! (For the case of the COOR ORIE VIEW command)
     ! modify-the-transformation-matrix
     CALL MODIFYTRANS(UMOD,QLAB,QDEBUG)
     REDRAW=QAUTO
     !-----------------------------------------------------------------------
  ELSE IF(WRD.EQ.'TRAJ') THEN
     ! process-trajio-commands; code from charmm_main
     IF (INDXA(COMLYN,COMLEN,'READ').GT.0) THEN
        IF (INDXA(COMLYN,COMLEN,'COMP').LE.0) THEN
           CALL REATRJ(X,Y,Z)
        ELSE
           CALL REATRJ(XCOMP,YCOMP,ZCOMP)
        ENDIF
        REDRAW=QAUTO
     ELSE IF (INDXA(COMLYN,COMLEN,'WRIT').GT.0) THEN
        IF (INDXA(COMLYN,COMLEN,'COMP').LE.0) THEN
           CALL WRTTRJ(X,Y,Z)
        ELSE
           CALL WRTTRJ(XCOMP,YCOMP,ZCOMP)
        ENDIF
     ELSE
        CALL TRAJIO
     ENDIF
     !-----------------------------------------------------------------------
  ELSE IF(WRD.EQ.'DEBU') THEN
     QDEBUG=(PRNLEV.GE.1)
     !-----------------------------------------------------------------------
  ELSE IF(WRD3.EQ.'FUL') THEN
#if KEY_NODISPLAY==0 /*full*/
     ! process-full-screen
     QFULLS=.NOT.QFULLS
#if KEY_XDISPLAY==1 /*xdisp3*/
     IF(PRNLEV.GT.2) WRITE(OUTU,*)'No full screen mode in X.'
#endif /* (xdisp3)*/
#endif /* (full)*/
     !-----------------------------------------------------------------------
  ELSE IF(WRD3.EQ.'INT') THEN
#if KEY_NODISPLAY==0
     ! process-interactive-mode
#if KEY_XDISPLAY==1
     IF(PRNLEV.GT.2) WRITE(OUTU,*)'No interactive mode in X yet.'
#endif 
#endif 
     !-----------------------------------------------------------------------
  ELSE IF(WRD3.EQ.'FON') THEN
#if KEY_NODISPLAY==0
     ! process-font-change
#if KEY_XDISPLAY==1
     CALL TRIMA(COMLYN,COMLEN)
     WRD=NEXTA4(COMLYN,COMLEN)
     IF(WRD.EQ.'VSMA') THEN
        CALL GRAFONT(1,.TRUE.)
     ELSE IF(WRD.EQ.'SMAL') THEN
        CALL GRAFONT(2,.TRUE.)
     ELSE IF(WRD.EQ.'MEDI') THEN
        CALL GRAFONT(3,.TRUE.)
     ELSE IF(WRD.EQ.'LARG') THEN
        CALL GRAFONT(4,.TRUE.)
     ELSE
        CALL GRAFONT(3,.TRUE.)
     ENDIF
     CALL TRIMA(COMLYN,COMLEN)
     IF (COMLEN.GT.0) THEN
        WRD=NEXTA4(COMLYN,COMLEN)
        IF(WRD.EQ.'COLO') THEN
           WRD=NEXTA4(COMLYN,COMLEN)
           CALL GETCOLOR(IGTICO,WRD,-1)
        ENDIF
     ENDIF
     COMLEN=0
     REDRAW=QAUTO
#endif 
#endif 
     !-----------------------------------------------------------------------
  ELSE IF(WRD3.EQ.'LBL') THEN
     ! process-label-command
     ! set atom selection flags and optional user label
     QFRST=.FALSE.
     QSELE=.FALSE.
     QLUSER=.FALSE.
     USRLBL='        '
     LBLLEN=0
     CALL TRIMA(COMLYN,COMLEN)
     IF (INDX(COMLYN,COMLEN,'USER',4).GT.0) THEN
        CALL GTRMWD(COMLYN,COMLEN,'USER',4,USRLBL,8,LBLLEN)
        QLUSER=.TRUE.
     ENDIF
     IF (INDX(COMLYN,COMLEN,'SELE',4).GT.0) QSELE=.TRUE.
     ! process-atom-selection
     CALL SELCTA(COMLYN,COMLEN,IGRSEL,X,Y,Z,WMAIN,.TRUE.)
     IF(INDXA(COMLYN,COMLEN,'FIRS').GT.0) QFRST=.TRUE.
     !  label size spec
     WRD=GTRMA(COMLYN,COMLEN,'SIZE')
     IF(WRD.EQ.'VSMA') THEN
        LBLSZ=1
     ELSE IF(WRD.EQ.'SMAL') THEN
        LBLSZ=2
     ELSE IF(WRD.EQ.'MEDI') THEN
        LBLSZ=3
     ELSE IF(WRD.EQ.'LARG') THEN
        LBLSZ=4
     ELSE
        LBLSZ=2
     ENDIF
     IF (QDEBUG) WRITE(OUTU,*) 'LBLSZ=',LBLSZ,' ',WRD
     !  label color spec
     WRD=GTRMA(COMLYN,COMLEN,'COLO')
     !  (label color default is yellow)
     CALL GETCOLOR(LBLCLR,WRD,3)
     IF(LBLCLR.LT.0) LBLCLR=3
     IF (QDEBUG) WRITE(OUTU,*) 'LBLCLR=',LBLCLR,' ',WRD
     !  check for initialization flag
     IF(INDXA(COMLYN,COMLEN,'INIT').GT.0) THEN
        DO I=1,NATOM
           IGRLBL(I)=0
        ENDDO
     ELSE
        !  label type options
        CALL TRIMA(COMLYN,COMLEN)
        IF((COMLEN.EQ.0).AND. .NOT. QLUSER )THEN
           QLRESN=.TRUE.
           QLRESI=.TRUE.
           LBLLEN=8
        ELSE
           QLSEGI=.FALSE.
           QLRESI=.FALSE.
           QLRESN=.FALSE.
           QLTYPE=.FALSE.
           QLCHEM=.FALSE.
           QLCHAR=.FALSE.
           QLWEIG=.FALSE.
           IF(INDXA(COMLYN,COMLEN,'SEGI').GT.0) THEN
              QLSEGI=.TRUE.
              LBLLEN=LBLLEN+4
           ENDIF
           IF(INDXA(COMLYN,COMLEN,'RESI').GT.0) THEN
              QLRESI=.TRUE.
              LBLLEN=LBLLEN+4
           ENDIF
           IF(INDXA(COMLYN,COMLEN,'RESN').GT.0) THEN
              QLRESN=.TRUE.
              LBLLEN=LBLLEN+4
           ENDIF
           IF(INDXA(COMLYN,COMLEN,'ATNU').GT.0 .OR. &
                INDXA(COMLYN,COMLEN,'ATNO').GT.0) THEN
              QLATNU=.TRUE.
              LBLLEN=LBLLEN+5
           ENDIF
           IF(INDXA(COMLYN,COMLEN,'TYPE').GT.0) THEN
              QLTYPE=.TRUE.
              LBLLEN=LBLLEN+4
           ENDIF
           IF(INDXA(COMLYN,COMLEN,'CHEM').GT.0) THEN
              QLCHEM=.TRUE.
              LBLLEN=LBLLEN+4
           ENDIF
           IF(INDXA(COMLYN,COMLEN,'CHAR').GT.0) THEN
              QLCHAR=.TRUE.
              LBLLEN=LBLLEN+12
           ENDIF
           IF(INDXA(COMLYN,COMLEN,'WEIG').GT.0) THEN
              QLWEIG=.TRUE.
              LBLLEN=LBLLEN+12
           ENDIF
           IF (QDEBUG) WRITE(OUTU,*) 'LBLLEN=',LBLLEN
           !  invalid label spec exceptions
           IF (QLWEIG.AND.QLCHAR) THEN
              IF(WRNLEV.GE.2) WRITE(OUTU,*) &
                   ' ** invalid label spec: CHARge & WEIGht **'
              GOTO 100
           ENDIF
           IF (LBLLEN.GT.24) THEN
              IF(PRNLEV.GE.2) WRITE(OUTU,*) &
                   ' ** invalid label spec: too many options **'
              GOTO 100
           ENDIF
        ENDIF
        IF (QDEBUG) WRITE(OUTU,747) QSELE,QFRST
747     FORMAT('[ select = ',L1,' ]     [ first = ',L1,' ]')
        !  process label atom selection vector
        !  first atom of selected residues
        IF(QFRST.AND.QSELE) THEN
           DO I=1,NATOM
              IF(IGRSEL(I).EQ.1) THEN
                 IGRSEL(I)=0
                 K=GETRES(I,IBASE,NRES)
                 DO J=1+IBASE(K),IBASE(K+1)
                    IGRSEL(J)=0
                 ENDDO
                 IGRSEL(1+IBASE(K))=1
              ENDIF
           ENDDO
           !  label selected atoms only
        ELSE IF(QSELE) THEN
           CONTINUE
           !  default; label first atom of each residue
        ELSE
           DO I=1,NATOM
              IGRSEL(I)=0
           ENDDO
           DO I=1,NRES
              LBLATM=1+IBASE(I)
              IGRSEL(LBLATM)=1
           ENDDO
        ENDIF
        DO I=1,NATOM
           IF(IGRSEL(I).EQ.1) THEN
              LBLATM=I
              IGRLBL(LBLATM)=LBLSZ+256*LBLCLR
              IGRLLN(LBLATM)=LBLLEN
              ! build-label-text
              CALL ATOMID(LBLATM,TSEG,TRES,TRSN,TTYP)
              L=0
              DO II=1,9
                 WTEXT=' '
                 LWTEXT=4
                 IF(II.EQ.1 .AND. QLSEGI) WTEXT=TSEG
                 IF(II.EQ.2 .AND. QLRESN) WTEXT=TRSN
                 IF(II.EQ.3 .AND. QLRESI) WTEXT=TRES
                 IF(II.EQ.4 .AND. QLTYPE) WTEXT=TTYP
                 IF(II.EQ.5 .AND. QLATNU) THEN
                    WRITE(WTEXT,'(I5)') I
                    LWTEXT=5
                    CALL TRIMA(WTEXT,LWTEXT)
                 ENDIF
                 IF(II.EQ.6 .AND. QLCHEM) WTEXT=ATC(IAC(LBLATM))
                 IF(II.EQ.7 .AND. QLUSER) THEN
                    WTEXT=USRLBL
                    LWTEXT=8
                 ENDIF
                 IF(II.EQ.8 .AND. QLCHAR) THEN
                    WRITE(WTEXT,'(F12.6)') CG(LBLATM)
                    LWTEXT=12
                    CALL TRIMA(WTEXT,LWTEXT)
                    ! delete trailing zeroes.
                    DO WHILE(WTEXT(LWTEXT:LWTEXT).EQ.'0')
                       LWTEXT=LWTEXT-1
                    ENDDO
                    ! don't delete trailing decimal point.
                    !C                        IF(WTEXT(LWTEXT:LWTEXT).EQ.'.') LWTEXT=LWTEXT-1
                 ENDIF
                 !
                 IF(II.EQ.9 .AND. QLWEIG) THEN
                    WRITE(WTEXT,'(1PG12.5)') WMAIN(LBLATM)
                    LWTEXT=12
                    CALL TRIMA(WTEXT,LWTEXT)
                    ! delete trailing zeroes.
                    DO WHILE(WTEXT(LWTEXT:LWTEXT).EQ.'0')
                       LWTEXT=LWTEXT-1
                    ENDDO
                    ! delete trailing decimal point.
                    IF(WTEXT(LWTEXT:LWTEXT).EQ.'.') LWTEXT=LWTEXT-1
                 ENDIF
                 !
                 IF(WTEXT.NE.' ') THEN
                    TGRLBL(LBLATM)(L+1:L+LWTEXT)=WTEXT(1:LWTEXT)
                    L=L+LWTEXT
                    CALL TRIME(TGRLBL(LBLATM),L)
                    L=L+1
                 ENDIF
              ENDDO
           ENDIF
        ENDDO
        !
     ENDIF
     REDRAW=QAUTO
     !-----------------------------------------------------------------------
  ELSE IF(WRD3.EQ.'HBS') THEN
     ! process-hbond-style-command
     CALL TRIMA(COMLYN,COMLEN)
     IF(COMLEN.EQ.0) THEN
        IGHBCO=10
        IGHBWI=4
        IGHBTY=4
     ELSE
        !  hbond color spec
        WRD=GTRMA(COMLYN,COMLEN,'COLO')
        CALL GETCOLOR(IGHBCO,WRD,IGHBCO)
        IF(IGHBCO.LT.0) IGHBCO=10
        I=GTRMI(COMLYN,COMLEN,'WIDT',-1)
        IF(I.GT.0) THEN
           IGHBWI=I
        ELSE
           IGHBWI=4
        ENDIF
        I=GTRMI(COMLYN,COMLEN,'DASH',-1)
        IF(I.GE.0) THEN
           IGHBTY=I
        ELSE
           IGHBTY=0
        ENDIF
     ENDIF
     REDRAW=QAUTO
     !-----------------------------------------------------------------------
  ELSE IF(WRD3.EQ.'SCA') THEN
     ! process-scale-command
     QLAB=(INDXA(COMLYN,COMLEN,'LAB').GT.0)
     QLAB=.NOT.(INDXA(COMLYN,COMLEN,'MOL').GT.0)
     SCGR=NEXTF(COMLYN,COMLEN)
     NREP=GTRMI(COMLYN,COMLEN,'REP',1)
     DO IREP=1,NREP
        GRSCAL=GRSCAL*SCGR
        DO I=1,3
           UMOD(I,I)=SCGR
        ENDDO
        ! modify-the-transformation-matrix
        CALL MODIFYTRANS(UMOD,QLAB,QDEBUG)
        IF(QAUTO) CALL DRAWIT(-1,X,Y,Z,XCOMP,YCOMP,ZCOMP)
     ENDDO
     !-----------------------------------------------------------------------
  ELSE IF(WRD3.EQ.'DRA') THEN
     ! process-draw-command
     CALL TRIME(COMLYN,COMLEN)
     IF(COMLEN.GT.0) THEN
        ! process-atom-selection
        CALL SELCTA(COMLYN,COMLEN,IGRSEL,X,Y,Z,WMAIN,.TRUE.)
        NGRSEL=0
        DO I=1,NATOM
           IF(IGRSEL(I).EQ.1) THEN
              IF(INITIA(I,X,Y,Z)) THEN
                 NGRSEL=NGRSEL+1
              ELSE
                 IGRSEL(I)=0
              ENDIF
           ENDIF
        ENDDO
        CALL GRAPNB(NATBON,IATBON,IGRSEL,INDEXR,INDEXP,NGRSEL)
     ENDIF
     CALL DRAWIT(-1,X,Y,Z,XCOMP,YCOMP,ZCOMP)
     !-----------------------------------------------------------------------
  ELSE IF(WRD3.EQ.'DEF') THEN
     ! assign-default-colors
     CALL ASSDEFCOLOR(NATOM,ATYPE,ICOLOR,ICOLRC)
     CALL ASSDEFRADII(NATOM,ATYPE,RADII)
     IGRASIZ=1.0
     IGRBSIZ=0.0
     REDRAW=QAUTO
     !-----------------------------------------------------------------------
  ELSE IF(WRD3.EQ.'BOX') THEN
     ! process-boxsize-command
     QLAB=(INDXA(COMLYN,COMLEN,'LAB').GT.0)
     QLAB=.NOT.(INDXA(COMLYN,COMLEN,'MOL').GT.0)
     SCGR=NEXTF(COMLYN,COMLEN)
     IF(SCGR.LE.0.0) THEN
        SCGR=1.0
     ELSE
        SCGR=16.0/SCGR
     ENDIF

     DO I=1,3
        UMOD(I,I)=SCGR/GRSCAL
     ENDDO
     GRSCAL=SCGR
     ! modify-the-transformation-matrix
     CALL MODIFYTRANS(UMOD,QLAB,QDEBUG)
     REDRAW=QAUTO
     !-----------------------------------------------------------------------
  ELSE IF(WRD3.EQ.'MAX') THEN
     ! process-maximum-window-command
     SCGR=0.0
     DO I=1,NATOM
        IF(INDEXR(I).NE.0) THEN
           IF(INITIA(I,X,Y,Z)) THEN
              R=X(I)**2+Y(I)**2+Z(I)**2
              IF(R.GT.SCGR) SCGR=R
           ENDIF
        ENDIF
     ENDDO
     SCGR=10.0/SQRT(SCGR)
     REDRAW=QAUTO
     DO I=1,3
        UMOD(I,I)=SCGR/GRSCAL
     ENDDO
     GRSCAL=SCGR
     ! modify-the-transformation-matrix
     CALL MODIFYTRANS(UMOD,QLAB,QDEBUG)
     !-----------------------------------------------------------------------
  ELSE IF(WRD3.EQ.'RES') THEN
     ! process-reset-command
     CALL RESETVIEW(ULAB,UMOD)
     GRSCAL=1.0
     REDRAW=QAUTO
     !-----------------------------------------------------------------------
  ELSE IF(WRD3.EQ.'CEN') THEN
     ! process-center-command
     QLAB=.FALSE.
     ! process-atom-selection
     CALL SELCTA(COMLYN,COMLEN,IGRSEL,X,Y,Z,WMAIN,.TRUE.)
     QMASS=(INDXA(COMLYN,COMLEN,'MASS').GT.0)
     XCGR=0.0
     YCGR=0.0
     ZCGR=0.0
     WGTSUM=0.0
     DO I=1,NATOM
        IF(IGRSEL(I).EQ.1) THEN
           IF(QMASS) THEN
              WGT=AMASS(I)
           ELSE
              WGT=1.0
           ENDIF
           WGTSUM=WGTSUM+WGT
           XCGR=XCGR+X(I)*WGT
           YCGR=YCGR+Y(I)*WGT
           ZCGR=ZCGR+Z(I)*WGT
        ENDIF
     ENDDO
     IF(WGT.GT.0.0) THEN
        XCGR=XCGR/WGTSUM
        YCGR=YCGR/WGTSUM
        ZCGR=ZCGR/WGTSUM
        IF(PRNLEV.GE.3) WRITE(OUTU,1044) XCGR,YCGR,ZCGR
1044    FORMAT(/' GRAPHX: Structure centered at ',3F12.5)
        !
        UMOD(1,4)=-XCGR
        UMOD(2,4)=-YCGR
        UMOD(3,4)=-ZCGR
        ULAB(1,4)=0.0
        ULAB(2,4)=0.0
        ULAB(3,4)=0.0
        ! modify-the-transformation-matrix
        CALL MODIFYTRANS(UMOD,QLAB,QDEBUG)
        REDRAW=QAUTO
     ELSE
        CALL WRNDIE(0,'<GRAPHX>','No atoms (or no mass) specified')
     ENDIF
     !-----------------------------------------------------------------------
  ELSE IF(WRD3.EQ.'POI') THEN
     ! process-point-command
     QLAB=.FALSE.
     XCGR=NEXTF(COMLYN,COMLEN)
     YCGR=NEXTF(COMLYN,COMLEN)
     ZCGR=NEXTF(COMLYN,COMLEN)
     UMOD(1,4)=-XCGR
     UMOD(2,4)=-YCGR
     UMOD(3,4)=-ZCGR
     ULAB(1,4)=0.0
     ULAB(2,4)=0.0
     ULAB(3,4)=0.0
     ! modify-the-transformation-matrix
     CALL MODIFYTRANS(UMOD,QLAB,QDEBUG)
     REDRAW=QAUTO
     !-----------------------------------------------------------------------
  ELSE IF(WRD3.EQ.'ROT') THEN
     ! process-rotate-command
     QLAB=(INDXA(COMLYN,COMLEN,'LAB').GT.0)
     QLAB=.NOT.(INDXA(COMLYN,COMLEN,'MOL').GT.0)
     XCGR=NEXTF(COMLYN,COMLEN)
     YCGR=NEXTF(COMLYN,COMLEN)
     ZCGR=NEXTF(COMLYN,COMLEN)
     NREP=GTRMI(COMLYN,COMLEN,'REP',1)
     DO IREP=1,NREP
        !
        ! DO ROTATIONS IN X,Y,Z ORDER.
        !
        IF(XCGR.NE.0.0) THEN
           THETA=XCGR*PI/180.0
           SNT=SIN(THETA)
           CST=COS(THETA)
           UMOD(2,2)=CST
           UMOD(3,3)=CST
           UMOD(2,3)=SNT
           UMOD(3,2)=-SNT
           ! modify-the-transformation-matrix
           CALL MODIFYTRANS(UMOD,QLAB,QDEBUG)
        ENDIF
        !
        IF(YCGR.NE.0.0) THEN
           THETA=YCGR*PI/180.0
           SNT=SIN(THETA)
           CST=COS(THETA)
           UMOD(3,3)=CST
           UMOD(1,1)=CST
           UMOD(3,1)=SNT
           UMOD(1,3)=-SNT
           ! modify-the-transformation-matrix
           CALL MODIFYTRANS(UMOD,QLAB,QDEBUG)
        ENDIF
        !
        IF(ZCGR.NE.0.0) THEN
           THETA=ZCGR*PI/180.0
           SNT=SIN(THETA)
           CST=COS(THETA)
           UMOD(1,1)=CST
           UMOD(2,2)=CST
           UMOD(1,2)=SNT
           UMOD(2,1)=-SNT
           ! modify-the-transformation-matrix
           CALL MODIFYTRANS(UMOD,QLAB,QDEBUG)
        ENDIF
        !
        IF(QAUTO) CALL DRAWIT(-1,X,Y,Z,XCOMP,YCOMP,ZCOMP)
     ENDDO
     !-----------------------------------------------------------------------
  ELSE IF((WRD.EQ.'TRA ').OR.(WRD.EQ.'TRAN')) THEN
     ! process-translate-command; changed to allow TRAJ, rmv jun2001
     QLAB=(INDXA(COMLYN,COMLEN,'LAB').GT.0)
     QLAB=.NOT.(INDXA(COMLYN,COMLEN,'MOL').GT.0)
     XCGR=NEXTF(COMLYN,COMLEN)
     YCGR=NEXTF(COMLYN,COMLEN)
     ZCGR=NEXTF(COMLYN,COMLEN)
     NREP=GTRMI(COMLYN,COMLEN,'REP',1)
     DO IREP=1,NREP
        UMOD(1,4)=XCGR
        UMOD(2,4)=YCGR
        UMOD(3,4)=ZCGR
        ! modify-the-transformation-matrix
        CALL MODIFYTRANS(UMOD,QLAB,QDEBUG)
        IF(QAUTO) CALL DRAWIT(-1,X,Y,Z,XCOMP,YCOMP,ZCOMP)
     ENDDO
     !-----------------------------------------------------------------------
  ELSE IF(WRD3.EQ.'ASS') THEN
     ! process-assign-command
     GRSCAL=1.0
     DO I=1,4
        DO J=1,4
           ULAB(J,I)=NEXTF(COMLYN,COMLEN)
           UMOD(J,I)=0.0
        ENDDO
        UMOD(I,I)=1.0
     ENDDO
     ! create-the-transformation-matrix
     CALL CREATETRANS(QDEBUG)
     REDRAW=QAUTO
     !-----------------------------------------------------------------------
  ELSE IF(WRD3.EQ.'ZCL') THEN
     ! process-z-clip-command
     XCGR=NEXTF(COMLYN,COMLEN)
     CALL TRIMA(COMLYN,COMLEN)
     IF(COMLEN.LE.0) THEN
        ZCLPL=-XCGR
        ZCLPH=XCGR
     ELSE
        ZCLPL=XCGR
        ZCLPH=NEXTF(COMLYN,COMLEN)
     ENDIF
     IF (ZCLPL.EQ.0.0) THEN
        ZCLPL = -999.0
        ZCLPH =  999.0
     ENDIF
     IF(QZAUTO) THEN
        ZCUEL=ZCLPL
        ZCUEH=ZCLPH
        IF(PRNLEV.GE.3) WRITE(OUTU,1053) ZCUEL,ZCUEH
     ENDIF
     IF(PRNLEV.GE.3) WRITE(OUTU,1052) ZCLPL,ZCLPH
1052 FORMAT(' GRAPHX: Z clipping range: ',2F12.5)
     REDRAW=QAUTO
     !-----------------------------------------------------------------------
  ELSE IF(WRD3.EQ.'ZCU') THEN
     ! process-z-cue-command
     QZAUTO=.FALSE.
     IF(INDXA(COMLYN,COMLEN,'AUTO').GT.0) THEN
        ZCUEL=ZCLPL
        ZCUEH=ZCLPH
        QZAUTO=.TRUE.
     ELSE IF(INDXA(COMLYN,COMLEN,'OFF').GT.0) THEN
        ZCUEL=-999.0
        ZCUEH=-999.0
     ELSE
        XCGR=NEXTF(COMLYN,COMLEN)
        CALL TRIMA(COMLYN,COMLEN)
        IF(COMLEN.LE.0) THEN
           ZCUEL=-XCGR
           ZCUEH=XCGR
        ELSE
           ZCUEL=XCGR
           ZCUEH=NEXTF(COMLYN,COMLEN)
        ENDIF
     ENDIF
     IF(PRNLEV.GE.3) WRITE(OUTU,1053) ZCUEL,ZCUEH
1053 FORMAT(' GRAPHX: Z queueing range: ',2F12.5)
     REDRAW=QAUTO
     !-----------------------------------------------------------------------
  ELSE IF(WRD3.EQ.'STE') THEN
     ! process-stereo-command
     QOFF=.NOT.(INDXA(COMLYN,COMLEN,'ON').GT.0)
     QOFF=(INDXA(COMLYN,COMLEN,'OFF').GT.0)
     QSTERO=.NOT.QOFF
     IF(QSTERO) THEN
        CALL TRIMA(COMLYN,COMLEN)
        IF(COMLEN.GT.0) DSTERO=NEXTF(COMLYN,COMLEN)
        CALL TRIMA(COMLYN,COMLEN)
        IF(COMLEN.GT.0) TSTERO=NEXTF(COMLYN,COMLEN)
        !
        DO I=1,4
           DO J=1,4
              USTERL(J,I)=0.0
              USTERR(J,I)=0.0
           ENDDO
           USTERL(I,I)=1.0
           USTERR(I,I)=1.0
        ENDDO
        !
        USTERR(1,4)=DSTERO*0.5
        USTERL(1,4)=-DSTERO*0.5
        THETA=TSTERO*0.5*PI/180.0
        CST=COS(THETA)
        SNT=SIN(THETA)
        USTERL(1,1)=CST
        USTERR(1,1)=CST
        USTERL(3,3)=CST
        USTERR(3,3)=CST
        USTERL(1,3)=SNT
        USTERR(1,3)=-SNT
        USTERL(3,1)=-SNT
        USTERR(3,1)=SNT
        !
     ENDIF
     ! create-the-transformation-matrix
     CALL CREATETRANS(QDEBUG)
     REDRAW=QAUTO
     !-----------------------------------------------------------------------
  ELSE IF(WRD3.EQ.'AXE') THEN
     ! process-axes-command
     CALL TRIMA(COMLYN,COMLEN)
     IF (INDXA(COMLYN,COMLEN,'DEFA').GT.0) THEN
        AXEXYZ(1,1) =  25.0
        AXEXYZ(1,2) = -25.0
        AXEXYZ(2,3) =  25.0
        AXEXYZ(2,4) = -25.0
        AXEXYZ(3,5) =  25.0
        AXEXYZ(3,6) = -25.0
     ELSEIF (COMLEN.GT.0) THEN
180     WRD = NEXTA4(COMLYN,COMLEN)
        IF (WRD.EQ.'XLIM') THEN
           AXEXYZ(1,2) = NEXTF(COMLYN,COMLEN)
           AXEXYZ(1,1) = NEXTF(COMLYN,COMLEN)
        ELSEIF (WRD.EQ.'YLIM') THEN
           AXEXYZ(2,4) = NEXTF(COMLYN,COMLEN)
           AXEXYZ(2,3) = NEXTF(COMLYN,COMLEN)
        ELSEIF (WRD.EQ.'ZLIM') THEN
           AXEXYZ(3,6) = NEXTF(COMLYN,COMLEN)
           AXEXYZ(3,5) = NEXTF(COMLYN,COMLEN)
        ELSEIF (WRD.EQ.'COLO') THEN
           WRD = NEXTA4(COMLYN,COMLEN)
           CALL GETCOLOR(IGAXCO,WRD,-1)
        ENDIF
        CALL TRIMA(COMLYN,COMLEN)
        IF(COMLEN.GT.0) GOTO 180
     ENDIF
     !-----------------------------------------------------------------------
  ELSE IF(WRD3.EQ.'COL') THEN
     ! process-color-command
     QCOMP=(INDXA(COMLYN,COMLEN,'COMP').GT.0)
     ! process-atom-selection
     CALL SELCTA(COMLYN,COMLEN,IGRSEL,X,Y,Z,WMAIN,.TRUE.)
     WRD=NEXTA4(COMLYN,COMLEN)
     CALL GETCOLOR(J,WRD,-1)
     IF(J.LT.0) GOTO 300
     ! process brightness option as a scale factor
     CALL TRIMA(COMLYN,COMLEN)
     IF(COMLEN.GT.0) THEN
        IF (INDXA(COMLYN,COMLEN,'LEV').GT.0) THEN
           I=NEXTI(COMLYN,COMLEN)
        ELSE
           WGT=NEXTF(COMLYN,COMLEN)
           I=IGRZLEV*(1.0-WGT)
        ENDIF
        IF(I.LE.0) I=0
        IF(I.GE.IGRZLEV) I=IGRZLEV-1
        J=J+16*I
     ENDIF
     !
     IF(QCOMP) THEN
        DO I=1,NATOM
           IF(IGRSEL(I).EQ.1) ICOLRC(I)=J
        ENDDO
     ELSE
        DO I=1,NATOM
           IF(IGRSEL(I).EQ.1) ICOLOR(I)=J
        ENDDO
     ENDIF
300  CONTINUE
     REDRAW=QAUTO
     !-----------------------------------------------------------------------
  ELSE IF(WRD3.EQ.'LIN') THEN
     ! process-line-width-command
     IGRWIDTH=NEXTI(COMLYN,COMLEN)
     REDRAW=QAUTO
     !-----------------------------------------------------------------------
  ELSE IF(WRD3.EQ.'RAD') THEN
     ! process-radii-command
     ! process-atom-selection
     CALL SELCTA(COMLYN,COMLEN,IGRSEL,X,Y,Z,WMAIN,.TRUE.)
     IF(INDXA(COMLYN,COMLEN,'DEF').GT.0) THEN
        CALL ASSDEFRADII(NATOM,ATYPE,RADII)
     ENDIF
     !
     IF(INDXA(COMLYN,COMLEN,'PAR').GT.0) THEN
        DO I=1,NATOM
           IF(IGRSEL(I).EQ.1) RADII(I)=VDWR(ITC(IAC(I)))
        ENDDO
     ENDIF
     !
     CALL TRIMA(COMLYN,COMLEN)
     IF(COMLEN.GT.0) THEN
        IGRASIZ=NEXTF(COMLYN,COMLEN)
        IF(IGRASIZ.LT.0.0) IGRASIZ=0.125
        DO I=1,NATOM
           IF(IGRSEL(I).EQ.1) RADII(I)=RADII(I)*IGRASIZ
        ENDDO
     ENDIF
     !
     IGRASIZ=1.0
     CALL TRIMA(COMLYN,COMLEN)
     IF(COMLEN.GT.0) IGRBSIZ=NEXTF(COMLYN,COMLEN)
     IF(IGRBSIZ.LE.0.0) IGRBSIZ=0.0
     REDRAW=QAUTO
#if KEY_XDISPLAY==1
     !-----------------------------------------------------------------------
  ELSE IF(WRD3.EQ.'SNA') THEN
     ! process-snap-command
     QOFF=.NOT.(INDXA(COMLYN,COMLEN,'ON').GT.0)
     QOFF=(INDXA(COMLYN,COMLEN,'OFF').GT.0)
     CALL TRIMA(COMLYN,COMLEN)
     IF (QOFF) THEN
        IWRP = 0
     ELSE
        IWRP = 1
     ENDIF
     CALL XSETWARP(IWRP)
#endif 
     !-----------------------------------------------------------------------
  ELSE IF(WRD3.EQ.'DIS') THEN
     ! process-display-command
     QOFF=.NOT.(INDXA(COMLYN,COMLEN,'ON').GT.0)
     QOFF=(INDXA(COMLYN,COMLEN,'OFF').GT.0)
     CALL TRIMA(COMLYN,COMLEN)
     !
     ! TURN OFF SELECTED DISPLAY COMPONENTS
     IF(QOFF) THEN
        IF(COMLEN.LE.0) THEN
           QDMAIN=.FALSE.
           QDCOMP=.FALSE.
           QDVECT=.FALSE.
        ELSE
           IF(INDXA(COMLYN,COMLEN,'MAIN').GT.0) QDMAIN=.FALSE.
           IF(INDXA(COMLYN,COMLEN,'COMP').GT.0) QDCOMP=.FALSE.
           IF(INDXA(COMLYN,COMLEN,'VECT').GT.0) QDVECT=.FALSE.
           IF(INDXA(COMLYN,COMLEN,'ATOM').GT.0) QDATOM=.FALSE.
           IF(INDXA(COMLYN,COMLEN,'HBON').GT.0) QDHBON=.FALSE.
           IF(INDXA(COMLYN,COMLEN,'BOND').GT.0) QDBOND=.FALSE.
           IF(INDXA(COMLYN,COMLEN,'TEXT').GT.0) QTITLE=.FALSE.
           IF(INDXA(COMLYN,COMLEN,'LABE').GT.0) QLABEL=.FALSE.
           IF(INDXA(COMLYN,COMLEN,'AXES').GT.0) QDAXE=.FALSE.
        ENDIF
        !
        ! TURN ON SELECTED DISPLAY COMPONENTS
     ELSE
        IF(INDXA(COMLYN,COMLEN,'MAIN').GT.0) QDMAIN=.TRUE.
        IF(INDXA(COMLYN,COMLEN,'COMP').GT.0) QDCOMP=.TRUE.
        IF(INDXA(COMLYN,COMLEN,'VECT').GT.0) QDVECT=.TRUE.
        IF(INDXA(COMLYN,COMLEN,'BOND').GT.0) QDBOND=.TRUE.
        IF(INDXA(COMLYN,COMLEN,'ATOM').GT.0) QDATOM=.TRUE.
        IF(INDXA(COMLYN,COMLEN,'HBON').GT.0) QDHBON=.TRUE.
        IF(INDXA(COMLYN,COMLEN,'TEXT').GT.0) QTITLE=.TRUE.
        IF(INDXA(COMLYN,COMLEN,'LABE').GT.0) QLABEL=.TRUE.
        IF(INDXA(COMLYN,COMLEN,'AXES').GT.0) QDAXE=.TRUE.
     ENDIF
     IF(QDCOMP.AND.QDVECT) THEN
       CALL WRNDIE(1,'<GRAPHX>','Cannot DISPLAY both COMP and VECT, VECT disabled')
       QDVECT=.FALSE.
     ENDIF
     REDRAW=QAUTO
     !-----------------------------------------------------------------------
  ELSE IF(WRD3.EQ.'MAK') THEN
     ! process-make-sor-command  <RMV> use Fortran version for now
     IUNIT=GTRMI(COMLYN,COMLEN,'UNIT',OUTU)
     CALL DRAWIT(IUNIT,X,Y,Z,XCOMP,YCOMP,ZCOMP)
     !-----------------------------------------------------------------------
  ELSE IF(WRD3.EQ.'AUT') THEN
     ! process-auto-command
     IF(INDXA(COMLYN,COMLEN,'OFF').GT.0) THEN
        QAUTO=.FALSE.
     ELSE IF(INDXA(COMLYN,COMLEN,'ON').GT.0) THEN
        QAUTO=.TRUE.
     ELSE
        QAUTO=.NOT.QAUTO
     ENDIF
     IF(PRNLEV.GE.3) THEN
        IF(QAUTO) THEN
           WRITE(OUTU,55) 'AUTO MODE ENABLED.'
        ELSE
           WRITE(OUTU,55) 'AUTO MODE DISABLED.'
        ENDIF
55      FORMAT(1X,A)
     ENDIF
     REDRAW=QAUTO
     !-----------------------------------------------------------------------
  ELSE IF(WRD3.EQ.'ERA') THEN
     ! process-erase-command
     IF(INDXA(COMLYN,COMLEN,'OFF').GT.0) THEN
        QERASE=.FALSE.
     ELSE IF(INDXA(COMLYN,COMLEN,'ON').GT.0) THEN
        QERASE=.TRUE.
     ELSE
        QERASE=.NOT.QERASE
     ENDIF
     !-----------------------------------------------------------------------
  ELSE IF(WRD3.EQ.'PSC') THEN
     ! process-postscript-command  <RMV>
     IUNIT=GTRMI(COMLYN,COMLEN,'UNIT',OUTU)
     QPSCLR=(INDXA(COMLYN,COMLEN,'COLO').GT.0)
     QPSORI=(.NOT.(INDXA(COMLYN,COMLEN,'PORT').GT.0))
     QPSBBK=(.NOT.(INDXA(COMLYN,COMLEN,'BWRE').GT.0))
     IF(QERASE) THEN
        CALL PSCINIT(IUNIT,.TRUE.)
     ELSE
        IF(INDXA(COMLYN,COMLEN,'INIT').GT.0) THEN
           CALL PSCINIT(IUNIT,.TRUE.)
        ELSEIF(INDXA(COMLYN,COMLEN,'TERM').GT.0) THEN
           CALL PSCETRM(IUNIT)
           GOTO 200
        ELSE
           CALL PSCINIT(IUNIT,.FALSE.)
        ENDIF
     ENDIF
     QPSCR=.TRUE.
     ! create-the-transformation-matrix
     CALL CREATETRANS(QDEBUG)
     CALL DRAWIT(IUNIT,X,Y,Z,XCOMP,YCOMP,ZCOMP)
     QPSCR=.FALSE.
     !
     ! bypass drawing for 'PSC UNIT n TERM' (needed for ERAse OFF overlays)
     !
200  CONTINUE
#if KEY_NODISPLAY==1
     CALL GRSETUP(.FALSE.)
#elif KEY_XDISPLAY==1
     IF (QNOWIN) THEN
        CALL GRSETUP(.FALSE.)
     ELSE
        CALL GRPHINQ(QFULLS)
     ENDIF
#endif 
     ! create-the-transformation-matrix
     CALL CREATETRANS(QDEBUG)
     !-----------------------------------------------------------------------
  ELSE IF(WRD3.EQ.'POV') THEN
     ! process-povray-command  <*> rvenable Apr 1997
     IUNIT=GTRMI(COMLYN,COMLEN,'UNIT',OUTU)
     ! object code disabled for now
     !        IF (INDXA(COMLYN,COMLEN,'OBJ').GT.0) THEN
     !C unit assumed to be open and not the same file as the POV file
     ! not-an-object, must be writing atoms and/or bonds to POV file
     IGPOVIU=GTRMI(COMLYN,COMLEN,'INCL',-1)
     !          IPOVOBJ=GTRMI(COMLYN,COMLEN,'UOBJ',-1)
     IPOVOBJ=-1
     QPOVR=.TRUE.
     IF(QERASE) THEN
        CALL POVINIT(IUNIT,.TRUE.)
        IF (QSTERO) CALL POVINIT(IUNIT+1,.TRUE.)
     ELSE
        IF(INDXA(COMLYN,COMLEN,'INIT').GT.0) THEN
           CALL POVINIT(IUNIT,.TRUE.)
           IF (QSTERO) CALL POVINIT(IUNIT+1,.TRUE.)
        ELSEIF(INDXA(COMLYN,COMLEN,'TERM').GT.0) THEN
           CALL POVETRM(IUNIT)
           GOTO 210
        ELSE
           CALL POVINIT(IUNIT,.FALSE.)
           IF (QSTERO) CALL POVINIT(IUNIT+1,.FALSE.)
        ENDIF
     ENDIF
     ! create-the-transformation-matrix
     CALL CREATETRANS(QDEBUG)
     CALL DRAWIT(IUNIT,X,Y,Z,XCOMP,YCOMP,ZCOMP)
     !
     ! bypass drawing for 'POV UNIT n TERM' (needed for ERAse OFF overlays)
210  CONTINUE
     NPVOBJ = 0
     QPOVR=.FALSE.

     ! end else not-an-object
     !        ENDIF

     ! reset the transformation matrix to the screen or pseudo-screen
#if KEY_NODISPLAY==1
     CALL GRSETUP(.FALSE.)
#elif KEY_XDISPLAY==1
     IF (QNOWIN) THEN
        CALL GRSETUP(.FALSE.)
     ELSE
        CALL GRPHINQ(QFULLS)
     ENDIF
#endif 
     ! create-the-transformation-matrix
     CALL CREATETRANS(QDEBUG)

     !-----------------------------------------------------------------------
  ELSE IF(WRD3.EQ.'PLU') THEN
     ! process-pluto-command
     IUNIT=GTRMI(COMLYN,COMLEN,'UNIT',OUTU)
     QPLUTO=.TRUE.
     CALL DRAWIT(IUNIT,X,Y,Z,XCOMP,YCOMP,ZCOMP)
     QPLUTO=.FALSE.
     !-----------------------------------------------------------------------
  ELSE IF(WRD3.EQ.'OFF') THEN
     ! process-off-command
#if KEY_NODISPLAY==1
     QGRDEV=.FALSE.
     ! terminate the graphics
#elif KEY_XDISPLAY==1
     IF (QNOWIN) THEN
        QGRDEV=.FALSE.
     ELSE
        CALL GRPHTERM
     ENDIF
#endif 
     RETURN
     !-----------------------------------------------------------------------
  ELSE IF(WRD3.EQ.'TEX') THEN
     ! process-text-command
     CALL TRIMA(COMLYN,COMLEN)
     IGRTLEN=COMLEN
     IF(COMLEN.GT.0) THEN
        GTITLE=COMLYN(1:COMLEN)
        QTITLE=.TRUE.
        ! blank out quotes needed to prevent upper case conversion
        IF (GTITLE(1:1).EQ.'"') GTITLE(1:1) = ' '
        IF (GTITLE(COMLEN:COMLEN).EQ.'"') &
             GTITLE(COMLEN:COMLEN) = ' '
     ELSE
        QTITLE=.FALSE.
     ENDIF
     COMLEN=0
     REDRAW=QAUTO
     !-----------------------------------------------------------------------
  ELSE IF(WRD3.EQ.'VEC') THEN
     ! PROCESS-VECTOR COMMAND
     WRD=GTRMA(COMLYN,COMLEN,'COLO')
     CALL GETCOLOR(IVECCO,WRD,3)
     IVECWI=GTRMI(COMLYN,COMLEN,'WIDT',1)
     IF(INDXA(COMLYN,COMLEN,'HEAD').GT.0) QDVEHD=.TRUE.
 
     !-----------------------------------------------------------------------
  ELSE IF(WRD3.EQ.'HEL') THEN
     ! PROCESS-HELP-COMMAND
     IF(PRNLEV.GE.5) THEN
        WRITE(OUTU,24) 'These commands affect what is viewed:'
        WRITE(OUTU,24) '   DISplay [ON/OFF] [MAIN] [COMP] [VECT]', &
             ' [ATOM] [BOND] [TEXT] [HBONds] [LABEls] [AXES]'
        WRITE(OUTU,24) '   COLor name brightfactor [COMP] ', &
             'atom-selection'
        WRITE(OUTU,24) '      NONE   RED  BLACk   YELLow GREEn  WHITe'
        WRITE(OUTU,24) '      BLUE   CYAN MAGEnta GRAY   ORANge BROWn'
        WRITE(OUTU,24) '      PURPle TURQuoise    CHARtreuse  DKBLue'
        WRITE(OUTU,24) '   LBL  label-type  label-atoms', &
             ' SIZE label-size', &
             ' COLOr label-color'
        WRITE(OUTU,24) '      label-type  = INIT SEGId RESN RESId', &
             ' TYPE CHEM CHARge WEIGht USER user-label'
        WRITE(OUTU,24) '      label-atoms = FIRSt and/or', &
             ' atom-selection'
        WRITE(OUTU,24) '      label-size  = SMALl NORMal MEDIum LARGe'
        WRITE(OUTU,24) '      label-color = see COLOR command'
        WRITE(OUTU,24) '      user-label  = up to 8 characters'
        WRITE(OUTU,24) '   LINe iwidth  (bonds or vectors; pixels)'
        WRITE(OUTU,24) '   HBStyle COLOr name WIDTh int DASH int'
        WRITE(OUTU,24) '   RADii [DEFaults] [PARam] scale [bond]', &
             ' atom-selection'
        WRITE(OUTU,24) '   DEFault-colors'
        WRITE(OUTU,24) '   AXEs [XLIM xmin xmax] [YLIM ymin ymax]'
        WRITE(OUTU,24) '        [ZLIM zmin zmax] default: -/+ 25.0'
        WRITE(OUTU,24) '   STEreo [ON/OFF] [dist] [angle]'
        WRITE(OUTU,24) '   DRAw [atom-selection]'
        WRITE(OUTU,24) '   FUL :: toggle full screen graphics'
        WRITE(OUTU,24) '   INTeractive:: full screen interactive mode'
        WRITE(OUTU,24) '      help= [HELP] or ?    exit= E'
        WRITE(OUTU,24) '   ERAse [ON/OFF]  :: erase between drawings'
        WRITE(OUTU,24) '   FONt [ SMALl | NORMal | MEDIum | LARGe ]'
        WRITE(OUTU,24) '   TEXt [text-body] :: display a title'
        WRITE(OUTU,24) '   OFF  :: disable all graphics and exit.'
        WRITE(OUTU,24) 'These commands change the view only:'
        WRITE(OUTU,24) '   RESet'
        WRITE(OUTU,24) '   SCAle  factor [MOL/LAB] [REP int]'
        WRITE(OUTU,24) '   BOXsize size  [MOL/LAB]'
        WRITE(OUTU,24) '   CENter [atom-selection]'
        WRITE(OUTU,24) '   MAXwindow'
        WRITE(OUTU,24) '   POInt  x y z'
        WRITE(OUTU,24) '   ROTate rx ry rz  [MOL/LAB] [REP int]'
        WRITE(OUTU,24) '   TRAnslate x y z  [MOL/LAB] [REP int]'
        WRITE(OUTU,24) '   ASSign-transformation-matrix', &
             ' 16x(real-number)'
        WRITE(OUTU,24) '   ZCLip [low] high'
        WRITE(OUTU,24) '   ZCUe [[low] high ]/[AUTO]'
        WRITE(OUTU,24) 'These commands do not affect the display:'
        WRITE(OUTU,24) '   AUTo [ON/OFF]:: redraw after every command'
        WRITE(OUTU,24) '   BITmap [file]:: screen to bitmap file.'
        WRITE(OUTU,24) '   HELp'
#if KEY_XDISPLAY==1
        WRITE(OUTU,24) '   SNAp [ON/OFF] :: "snap" pointer to window'
#endif 
        WRITE(OUTU,24) '   EXEcute pathname'
        WRITE(OUTU,24) '   END   ::exit from command parser only.'
        WRITE(OUTU,24) 'NOTE: the UNIT must be OPENed first...'
        WRITE(OUTU,24) '   *  ... and is NOT closed (overlay)'
        WRITE(OUTU,24) '   POV *  UNIT n  [ INCLude n ]'
        WRITE(OUTU,24) '   PSC *  UNIT n  [ COLOr ] [ PORTrait ]'
        WRITE(OUTU,24) '   PLUto  UNIT n :: PLUTO format plot file'
        WRITE(OUTU,24) '   MAKe   UNIT n :: LIGHT file.'
24      FORMAT(' GRAPHX HELP> ',10A)
     ENDIF
#if KEY_UNIX==1
     !-----------------------------------------------------------------------
  ELSE IF(WRD3.EQ.'EXE') THEN
     ! process-execute-command
     CALL TRIMA(COMLYN,COMLEN)
     IF(COMLEN.GT.0) CALL GRPSHELL(COMLYN(1:COMLEN))
     COMLEN=0
#endif 
     !-----------------------------------------------------------------------
  ELSE IF(WRD3.EQ.'END') THEN
     ! process-end-command
     ! Just exit the command parser. leave everything intact
     RETURN
     !-----------------------------------------------------------------------
  ELSE
     CALL WRNDIE(1,'<GRAPHX>','UNRECOGNIZED GRAPHICS COMMAND')
  ENDIF
  !-----------------------------------------------------------------------
  !-----------------------------------------------------------------------
  IF(REDRAW) THEN
     CALL DRAWIT(-1,X,Y,Z,XCOMP,YCOMP,ZCOMP)
     REDRAW=.FALSE.
  ENDIF
  GOTO 100
  !
  !====================================================================
END SUBROUTINE GRPHX2
#endif /* (nographx)*/

