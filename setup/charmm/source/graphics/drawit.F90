!CHARMM Element source/graphics/drawit.src 1.1
#if KEY_NOGRAPHICS==0
SUBROUTINE DRAWIT(IUNIT,X,Y,Z,XCOMP,YCOMP,ZCOMP)
  !
  !  THIS ROUTINE DRAWS THE CURRENT COORDINATES USING THE
  !  VIEW AND SELECTION ALREADY DEFINED
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use psf
  use graph
  use stream
  use memory
  implicit none
  real(chm_real),allocatable,dimension(:) :: XYZ
  real(chm_real),allocatable,dimension(:) :: XYZS
  integer,allocatable,dimension(:) :: IXV
  integer,allocatable,dimension(:) :: IYV
  integer,allocatable,dimension(:) :: IZV
  integer,allocatable,dimension(:) :: ICOLR
  integer,allocatable,dimension(:) :: IARAD
  integer,allocatable,dimension(:) :: IHBIX
!   next four added for vectors in comp :: rvenable
  real(chm_real),allocatable,dimension(:) :: XYZC
  real(chm_real),allocatable,dimension(:) :: XYZCS
  integer,allocatable,dimension(:) :: IXVC
  integer,allocatable,dimension(:) :: IYVC
  INTEGER IUNIT
  real(chm_real) X(*),Y(*),Z(*)
  real(chm_real) XCOMP(*),YCOMP(*),ZCOMP(*)
  !
  !
  !
  LOGICAL INDRW,FINDRW,ERROR
  !
  IF (.NOT.QGRDEV) RETURN
  IF (NGRSEL.LE.0) RETURN
  !
  call chmalloc('drawit.src','DRAWIT','XYZ',NGRSEL*4,crl=XYZ)
  call chmalloc('drawit.src','DRAWIT','XYZS',NGRSEL*4,crl=XYZS)
  call chmalloc('drawit.src','DRAWIT','IXV',NGRSEL,intg=IXV)
  call chmalloc('drawit.src','DRAWIT','IYV',NGRSEL,intg=IYV)
  call chmalloc('drawit.src','DRAWIT','IZV',NGRSEL,intg=IZV)
  call chmalloc('drawit.src','DRAWIT','ICOLR',NGRSEL,intg=ICOLR)
  call chmalloc('drawit.src','DRAWIT','IARAD',NGRSEL,intg=IARAD)
  call chmalloc('drawit.src','DRAWIT','IHBIX',NGRSEL*3,intg=IHBIX)
  call chmalloc('drawit.src','DRAWIT','XYZC',NGRSEL*4,crl=XYZC)
  call chmalloc('drawit.src','DRAWIT','XYZCS',NGRSEL*4,crl=XYZCS)
  call chmalloc('drawit.src','DRAWIT','IXVC',NGRSEL,intg=IXVC)
  call chmalloc('drawit.src','DRAWIT','IYVC',NGRSEL,intg=IYVC)
  !
  INDRW=.TRUE.
  FINDRW=.NOT.QDCOMP

  ! the last four args to DRAWT2 are for COMP VECTors
  IF (QDMAIN) THEN
     IF (QSTERO) THEN
        call DRAWT2(X,Y,Z,XYZ,XYZS,IXV,IYV,IZV,ICOLR, &
             IARAD,IHBIX,ICOLOR,ULEFT,-1,INDRW,.FALSE.,IUNIT,HNPOVRTX, &
             HPOVRTX,HTRNVRTX,XYZC,XYZCS,IXVC,IYVC)
        INDRW=.FALSE.
        call DRAWT2(X,Y,Z,XYZ,XYZS,IXV,IYV,IZV,ICOLR, &
             IARAD,IHBIX,ICOLOR,URIGHT,1,INDRW,FINDRW,IUNIT,HNPOVRTX, &
             HPOVRTX,HTRNVRTX,XYZC,XYZCS,IXVC,IYVC)
     ELSE
        call DRAWT2(X,Y,Z,XYZ,XYZS,IXV,IYV,IZV,ICOLR, &
             IARAD,IHBIX,ICOLOR,ULEFT,0,INDRW,FINDRW,IUNIT,HNPOVRTX, &
             HPOVRTX,HTRNVRTX,XYZC,XYZCS,IXVC,IYVC)
        INDRW=.FALSE.
     ENDIF
  ENDIF
  !
  IF (QDCOMP.AND. (.NOT.QDVECT)) THEN
     FINDRW=.TRUE.
     IF (QSTERO) THEN
        call DRAWT2(XCOMP,YCOMP,ZCOMP,XYZ,XYZS,IXV,IYV,IZV, &
             ICOLR,IARAD,IHBIX,ICOLRC,ULEFT,-1,INDRW,.FALSE.,IUNIT, &
             HNPOVRTX,HPOVRTX,HTRNVRTX,XYZC,XYZCS,IXVC,IYVC)
        INDRW=.FALSE.
        call DRAWT2(XCOMP,YCOMP,ZCOMP,XYZ,XYZS,IXV,IYV,IZV, &
             ICOLR,IARAD,IHBIX,ICOLRC,URIGHT,1,INDRW,FINDRW,IUNIT, &
             HNPOVRTX,HPOVRTX,HTRNVRTX,XYZC,XYZCS,IXVC,IYVC)
     ELSE
        call DRAWT2(XCOMP,YCOMP,ZCOMP,XYZ,XYZS,IXV,IYV,IZV, &
             ICOLR,IARAD,IHBIX,ICOLRC,ULEFT,0,INDRW,FINDRW,IUNIT, &
             HNPOVRTX,HPOVRTX,HTRNVRTX,XYZC,XYZCS,IXVC,IYVC)
     ENDIF
  ENDIF

  call chmdealloc('drawit.src','DRAWIT','XYZ',NGRSEL*4,crl=XYZ)
  call chmdealloc('drawit.src','DRAWIT','XYZS',NGRSEL*4,crl=XYZS)
  call chmdealloc('drawit.src','DRAWIT','IXV',NGRSEL,intg=IXV)
  call chmdealloc('drawit.src','DRAWIT','IYV',NGRSEL,intg=IYV)
  call chmdealloc('drawit.src','DRAWIT','IZV',NGRSEL,intg=IZV)
  call chmdealloc('drawit.src','DRAWIT','ICOLR',NGRSEL,intg=ICOLR)
  call chmdealloc('drawit.src','DRAWIT','IARAD',NGRSEL,intg=IARAD)
  call chmdealloc('drawit.src','DRAWIT','IHBIX',NGRSEL*3,intg=IHBIX)
  call chmdealloc('drawit.src','DRAWIT','XYZC',NGRSEL*4,crl=XYZC)
  call chmdealloc('drawit.src','DRAWIT','XYZCS',NGRSEL*4,crl=XYZCS)
  call chmdealloc('drawit.src','DRAWIT','IXVC',NGRSEL,intg=IXVC)
  call chmdealloc('drawit.src','DRAWIT','IYVC',NGRSEL,intg=IYVC)

  IF (QERASE.AND.(IUNIT.GE.0).AND.IOLEV.GT.0) THEN
     CALL VCLOSE(IUNIT,'KEEP',ERROR)
  ENDIF
  RETURN
END SUBROUTINE DRAWIT

SUBROUTINE DRAWT2(X,Y,Z,XYZ,XYZS,IXV,IYV,IZV,ICOLR,IARAD, &
     IHBIX,ICOLRP,UTRANS,ISTERO,INDRW,FINDRW,IUNIT, &
     NPOVRTX,POVRTX,TRNVRTX,XYZC,XYZCS,IXVC,IYVC)
  !
  !
  !  THIS ROUTINE DRAWS THE CURRENT COORDINATES USING THE
  !  VIEW AND SELECTION ALREADY DEFINED
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use graph
  !CCC  use param
  use psf
  use stream
  ! use coordc, only: xcomp, ycomp, zcomp
  use coordc
  implicit none
  !
  real(chm_real) X(*),Y(*),Z(*)
  real(chm_real) XYZ(4,*),XYZS(4,*)
  INTEGER IXV(*),IYV(*),ICOLR(*),ICOLRP(*),IARAD(*)
  ! arguments for vectors in comp :: rvenable
  real(chm_real) :: XYZC(4,*),XYZCS(4,*)
  INTEGER :: IXVC(*),IYVC(*)
  INTEGER IZV(*),IHBIX(*), NPOVRTX
  real(chm_real) POVRTX(4,NPOVRTX), TRNVRTX(4,NPOVRTX)
  real(chm_real) UTRANS(4,4), UTMP(4,4), UXSCL(4,4)
  INTEGER ISTERO
  LOGICAL INDRW,FINDRW
  INTEGER IUNIT

  !
  INTEGER ISTOP,ISTRT,IZH,IZL
  INTEGER I,J,ILEV,NLEV,II
  real(chm_real) ARAD, XGR
  SAVE ISTRT,ISTOP
  !
  ! GENERATE COORDINATE MATRIX (4XN)
  !
  DO II=1,NGRSEL
     I=INDEXP(II)
     XYZ(1,II)=X(I)
     XYZ(2,II)=Y(I)
     XYZ(3,II)=Z(I)
     XYZ(4,II)=1.0
  ENDDO
  !
  ! TRANSFORM COORDINATES; FIRST UNDO SCALING FOR POVRAY
  !
  IF (QPOVR.AND.INDRW) THEN
     XGR = 1.0 / GRSCAL
     DO I=1,4
        DO J=1,4
           UTMP(I,J) = UTRANS(I,J)
           UXSCL(I,J) = 0.0
        ENDDO
        UXSCL(I,I) = XGR
     ENDDO
     UXSCL(4,4) = 1.0
     CALL VECMMUL(UTMP,UXSCL,UTRANS)
  ENDIF

  CALL VECMMULN(UTRANS,XYZ,4,4,NGRSEL,XYZS)
  CALL VECMMULN(UTRANS,AXEXYZ,4,4,7,AXETRN)
  IF (NPVOBJ.GT.0) THEN
     I = KPVOBJ(2,NPVOBJ) + KPVOBJ(3,NPVOBJ)
     CALL VECMMULN(UTRANS,POVRTX,4,4,NPOVRTX,TRNVRTX)
  ENDIF
  !
  ! SORT BONDS BY Z VALUES
  !
  IF (ISTERO.LE.0) THEN
     DO I=1,NGRSEL
        IZV(I)=XYZS(3,I)
     ENDDO
     CALL SORTP(NGRSEL,INDEXS,ORDER5,IZV,0,0,0,0,0,0,1)
     IF (QDHBON) THEN
        DO I=1,NGRSEL
           INDEXB(INDEXS(I))=I
        ENDDO
     ENDIF
     !
     ! DO Z CLIPPING
     IZL=ZCLPL*GRDUPA*GRSCAL
     IZH=ZCLPH*GRDUPA*GRSCAL
     !
     DO I=1,NGRSEL
        J=INDEXS(I)
        IF (IZV(J).GE.IZL) THEN
           ISTRT=I
           GOTO 100
        ENDIF
     ENDDO
     ISTRT=NGRSEL
100  CONTINUE
     !
     DO I=NGRSEL,1,-1
        J=INDEXS(I)
        IF (IZV(J).LE.IZH) THEN
           ISTOP=I
           GOTO 200
        ENDIF
     ENDDO
     ISTOP=1
200  CONTINUE
     !
     ! DO Z SHADING
     IF (ZCUEH.GE.ZCUEL) THEN
        IZL=ZCUEL*GRDUPA*GRSCAL
        IZH=ZCUEH*GRDUPA*GRSCAL
     ELSE
        IZL=IZV(INDEXS(ISTRT))
        IZH=IZV(INDEXS(ISTOP))
     ENDIF
     !
     DO I=ISTRT,ISTOP
        J=INDEXS(I)
        ICOLR(J)=ICOLRP(INDEXP(J))
        ILEV=ICOLR(J)/16
        NLEV=IGRZLEV-ILEV
        IF (IZV(J).LT.IZH) THEN
           IF (IZV(J).LE.IZL) THEN
              ICOLR(J)=ICOLR(J)+16*(NLEV-1)
           ELSE
              ILEV=(NLEV*(IZH-IZV(J)))/(IZH-IZL)
              ICOLR(J)=ICOLR(J)+16*ILEV
           ENDIF
        ENDIF
     ENDDO
     !
     ! FIN ! IF(ISTERO.LE.0)
  ENDIF

  ! Tim Miller: deal with the comparison coordinates, in case
  ! the vector option is used.
  if (QDVECT) then
    do ii=1,ngrsel
      i=indexp(ii)
      xyzc(1,ii)=xcomp(i)+x(i)
      xyzc(2,ii)=ycomp(i)+y(i)
      xyzc(3,ii)=zcomp(i)+z(i)
      xyzc(4,ii)=1.0
    enddo
    call vecmmuln(utrans,xyzc,4,4,ngrsel,xyzcs)
    do i=1,ngrsel
      ixvc(i)=xyzcs(1,i)
      iyvc(i)=xyzcs(2,i)
    enddo
  endif

  ! convert x,y to integer (z done above); calc display radii
  DO I=1,NGRSEL
     IXV(I)=XYZS(1,I)
     IYV(I)=XYZS(2,I)
     IF (IGRASIZ.GT.0.0) THEN
        ARAD=RADII(INDEXP(I))
        IARAD(I)=(GRDUPA*ARAD*GRSCAL)
        IF (QPOVR) THEN
           XYZS(4,I)=GRDUPA*ARAD
        ELSE
           XYZS(4,I)=GRDUPA*ARAD*GRSCAL
        ENDIF
     ELSE
        IARAD(I)=0
        XYZS(4,I)=0.0
     ENDIF
  ENDDO
  !
  ! DISPLAY IT ALL
  !
  IF (IUNIT.LT.0) THEN
#if KEY_NODISPLAY==1
     RETURN
#else /**/
#if KEY_XDISPLAY==1
     IF (.NOT. QNOWIN) THEN
       CALL DRAWT3(ISTRT,ISTOP,IXV,IYV,IZV, &
            ICOLR,IARAD,IHBIX,ISTERO,INDRW,FINDRW,IXVC,IYVC)
     ENDIF
#else /**/
     RETURN
#endif 
#endif 
  ELSE
     ! begin ++ CONDITIONAL
     IF (QPSCR) THEN
        !  start or add to postscript display file -- rmv -- nov93
        CALL PSDRAW(IUNIT,ISTRT,ISTOP,IXV,IYV,IZV,ICOLR,IARAD, &
             IHBIX,ISTERO,INDRW,FINDRW,IXVC,IYVC)
     ELSEIF (QPOVR) THEN
        !  start or add to povray input file -- rmv --  begun mar97
        IF ((.NOT.INDRW).AND.(.NOT.FINDRW).AND.(ISTERO.EQ.1)) THEN
           CALL POVDFN(IUNIT,ISTRT,ISTOP,XYZS,ICOLR,ISTERO, &
                .TRUE.,FINDRW,NPOVRTX,TRNVRTX)
        ELSE
           CALL POVDFN(IUNIT,ISTRT,ISTOP,XYZS,ICOLR,ISTERO, &
                INDRW,FINDRW,NPOVRTX,TRNVRTX,XYZCS)
        ENDIF
     ELSEIF (QPLUTO) THEN
        !  make FDAT file for use with PLUTO program -- rmv -- april 1990
        IF (INDRW) CALL MAKFDAT(IUNIT,ISTRT,ISTOP,XYZS)
     ELSE
        CALL MAKSOR(IUNIT,ISTRT,ISTOP,XYZS,ICOLR,IARAD,ISTERO, &
             DSTERO)
        ! end ++ CONDITIONAL
     ENDIF
     !     FIN ! else not screen
  ENDIF

  !
  RETURN
END SUBROUTINE DRAWT2

SUBROUTINE MAKSOR(IUNIT,ISTRT,ISTOP,XYZS,ICOLR, &
     IARAD,ISTERO,STSEP)
  !
  use chm_kinds
  use dimens_fcm
  use graph
  use comand
  use stream
  use string
  implicit none
  !
  INTEGER IUNIT,ISTRT,ISTOP,ISTERO
  real(chm_real) XYZS(4,*)
  INTEGER ICOLR(*),IARAD(*)
  !
  !
  INTEGER I,J,K,II,JJ,IREFL,ITRAN,JARAD,ISTSEP
  INTEGER NBAT,IBSAT,IAT,JAT
  real(chm_real) STSEP,XYZV(3),FACT
  !
  IF(IOLEV.LT.0) RETURN
  !
  !* ENERTEST PICTURE
  !*  X    Y    Z   RADI COLR ISTER REFL TRAN
  !*
  !  533  769   82   25    5    0    4    0
  !
  IF (ISTERO.LE.0) THEN

     CALL TRIMA(COMLYN,COMLEN)
     IF (COMLEN.GT.0) WRITE(IUNIT,21) '*',COMLYN(1:COMLEN)
     WRITE(IUNIT,21) '*'
21   FORMAT(2A)
  ENDIF
  !
  ISTSEP=STSEP*GRDUPA
  IF (ISTERO.EQ.0) WRITE(IUNIT,22) 0,0,100,960
  IF (ISTERO.EQ.-1) WRITE(IUNIT,22) 0,ISTSEP,100,960
22 FORMAT(12I5)
  !
  IF (QDATOM) THEN
     IREFL=4
     ITRAN=0
     DO II=ISTRT,ISTOP
        IAT=INDEXS(II)
        WRITE(IUNIT,23) XYZS(1,IAT),XYZS(2,IAT),XYZS(3,IAT), &
             IARAD(IAT),ICOLR(IAT),ISTERO,IREFL,ITRAN
23      FORMAT(3F10.3,12I5)
     ENDDO
  ENDIF
  !
  IF (QDBOND .AND. IGRBSIZ.GT.0.0) THEN
     DO I=1,NGRSEL
        INDEXB(INDEXS(I))=I
     ENDDO
     IREFL=4
     ITRAN=0
     IBSAT=0
     !C      IF (QDATOM) IBSAT=2
     NBAT=10
     JARAD=(GRDUPA*GRSCAL*IGRBSIZ)
     !
     DO II=ISTRT,ISTOP
        !C        WRITE(OUTU,88) II
        !C  88    FORMAT(' PROCESSING ATOM ',I5)
        IAT=INDEXS(II)
        DO J=1,NATBON(IAT)
           JAT=IATBON(J,IAT)
           JJ=INDEXB(JAT)
           !C          WRITE(OUTU,89) II,IAT,J,JJ,JAT
           !C  89      FORMAT(' PROCESSING BOND ',5I5)
           FACT=0.0
           DO K=1,3
              FACT=FACT+(XYZS(K,IAT)-XYZS(K,JAT))**2
           ENDDO
           !C          FACT=SQRT(FACT)
           IF (FACT.GT.1.0) THEN
              IF (JJ.GT.0) THEN
                 DO I=IBSAT,NBAT
                    FACT=I
                    FACT=FACT/(NBAT+NBAT+1)
                    DO K=1,3
                       XYZV(K)=(1.0-FACT)*XYZS(K,IAT)+FACT*XYZS(K,JAT)
                    ENDDO
                    !
                    WRITE(IUNIT,23) XYZV(1),XYZV(2),XYZV(3), &
                         JARAD,ICOLR(IAT),ISTERO,IREFL,ITRAN
                 ENDDO
              ENDIF
           ENDIF
        ENDDO
     ENDDO
  ENDIF
  !
  RETURN
END SUBROUTINE MAKSOR

SUBROUTINE GRAPNB(NATBON,IATBON,IGRSEL,INDEXR,INDEXP, &
     NGRSEL)
  !
  ! THIS ROUTINE MAKES AN (XC) TYPE REPRESENTATION OF THE BOND LIST.
  !
  use chm_kinds
  use dimens_fcm
  use psf
  use stream
  implicit none
  !
  ! iatbmx = 8 and is a parameter set in dimens.f90
  !
  INTEGER NGRSEL,INDEXR(*),INDEXP(*)
  INTEGER NATBON(*),IATBON(IATBMX,*)
  INTEGER IGRSEL(*)
  !
  INTEGER IBT,JBT,I,J
  !
  IF (NGRSEL.LT.1) RETURN
  IF (IOLEV.LT.0) RETURN
  !
  J=0
  DO I=1,NATOM
     INDEXR(I)=0
     IF (IGRSEL(I).EQ.1) THEN
        J=J+1
        INDEXR(I)=J
        INDEXP(J)=I
        NATBON(J)=0
     ENDIF
  ENDDO
  !
  DO I=1,NBOND
     IBT=IB(I)
     JBT=JB(I)
     IF (IBT.GT.0 .AND. JBT.GT.0) THEN
        IBT=INDEXR(IBT)
        JBT=INDEXR(JBT)
        IF (IBT.GT.0 .AND. JBT.GT.0) THEN
           !
           IF (NATBON(IBT).LT.IATBMX) THEN
              NATBON(IBT)=NATBON(IBT)+1
              IATBON(NATBON(IBT),IBT)=JBT
           ENDIF
           !
           IF(NATBON(JBT).LT.IATBMX) THEN
              NATBON(JBT)=NATBON(JBT)+1
              IATBON(NATBON(JBT),JBT)=IBT
           ENDIF
           !
        ENDIF
     ENDIF
  ENDDO
  !
  RETURN
END SUBROUTINE GRAPNB

SUBROUTINE WRMATX(NAME,U)
  !
  ! PRINT OUT A TRANSFORMATION MATRIX
  !
  use chm_kinds
  use stream
  implicit none
  !
  CHARACTER(len=*) NAME
  real(chm_real) U(4,4)
  INTEGER I,J
  !
  IF (PRNLEV.LT.2) RETURN
  WRITE(OUTU,44) NAME
44 FORMAT(' MATRIX FOR ',A)
  DO I=1,4
     WRITE(OUTU,33) (U(I,J),J=1,4)
  ENDDO
33 FORMAT(4F10.5)
  RETURN
END SUBROUTINE WRMATX

SUBROUTINE MAKFDAT(IUNIT,ISTRT,ISTOP,XYZS)
  !  make an FDAT file for PLUTO -- 800 card max at present
  !  (N.B. pluto has 400 atom max as supplied by Cambridge)
  !  not a full-fledged FDAT entry, only data needed for plot
  !  uses current view transform, atom selection, and Z clip
  !  rvenable <*> april 1990
  !
  !  args:
  !
  !     IUNIT   output file unit
  !     ISTRT   first atom index
  !     ISTOP   last atom index
  !     XYZS    atom coordinates
  !
  !
  use chm_kinds
  use dimens_fcm
  use psf
  use graph
  use stream
  implicit none
  !
  INTEGER IUNIT,ISTRT,ISTOP,NATM,NCRD,NAC,I,J,IAT,IC
  INTEGER NCON,NCC,KCON(5000),K,K1,K2,NICD
  INTEGER, PARAMETER :: MXACRD=2000
  CHARACTER(len=80) CFDAT(MXACRD)
  CHARACTER(len=8)  CNATM(3)
  real(chm_real) XYZS(4,*),AFAC
  !
  IF (IOLEV.LT.0) RETURN
  !
  !  afac includes a factor of 1/32, since coordinates are in pixels
  AFAC=3.125E2
  !  initialize directory record
  DO I=1,80
     CFDAT(1)(I:I)='0'
  ENDDO
  CFDAT(1)(1:1)='#'
  CFDAT(1)(2:9)='        '
  CFDAT(1)(11:11)='3'
  CFDAT(1)(18:23)='      '
  !  set CELL=1
  CFDAT(1)(57:57)='1'
  !  atom format 2; I7 coord field, A x 10**5
  CFDAT(1)(59:59)='2'
  !  no. of radii datum
  !C      CFDAT(1)(42:44)='  8'
  !  cubic unit cell, 100 A sides
  WRITE(CFDAT(2),910) 100,100,100,900,900,900,1,1,1,1,1,1, &
       0,0,0,0,0,0,0,0,0,'        ',0,0,'    '
910 FORMAT(6I6,6I1,6I2,3I3,A8,I3,I2,A4)
  !  crystal radii
  !C      CFDAT(3)='C  68H  23N  68O  68P 105S 102F  64CL 99'
  !C      NICD=3

  NICD=2
  NATM=1+(ISTOP-ISTRT)
  IF (NATM.GE.1000) THEN
     CALL WRNDIE(1,'<MKFDAT>','999 atom limit exceeded')
     RETURN
  ENDIF
  NAC=(NATM+2)/3
  WRITE(CFDAT(1)(45:47),'(I3)') NATM

  !  construct atom cards, 3 atoms per card; names from PSF
920 FORMAT(A,1X,3I7,1X,A,1X,3I7,1X,A,1X,3I7)
  IAT=ISTRT
  DO IC=NICD+1,NAC+NICD
     IF (IAT+2.LE.ISTOP) THEN
        DO J=1,3
           CNATM(J)=ATYPE(INDEXP(J+IAT-1))
        ENDDO
        WRITE(CFDAT(IC),920) CNATM(1)(1:idleng), &
             INT(AFAC*XYZS(1,IAT)), &
             INT(AFAC*XYZS(2,IAT)),-1*INT(AFAC*XYZS(3,IAT)), &
             CNATM(2)(1:idleng),INT(AFAC*XYZS(1,IAT+1)), &
             INT(AFAC*XYZS(2,IAT+1)),-1*INT(AFAC*XYZS(3,IAT+1)), &
             CNATM(3)(1:idleng),INT(AFAC*XYZS(1,IAT+2)), &
             INT(AFAC*XYZS(2,IAT+2)),-1*INT(AFAC*XYZS(3,IAT+2))
     ELSE
        ! begin ++ CONDITIONAL
        IF (IAT.EQ.ISTOP) THEN
           CNATM(1)=ATYPE(INDEXP(IAT))
           WRITE(CFDAT(IC),920) CNATM(1)(1:IDLENG), &
                INT(AFAC*XYZS(1,IAT)), &
                INT(AFAC*XYZS(2,IAT)),-1*INT(AFAC*XYZS(3,IAT))
        ELSEIF (IAT+1.EQ.ISTOP) THEN
           DO J=1,2
              CNATM(J)=ATYPE(INDEXP(J+IAT-1))
           ENDDO
           WRITE(CFDAT(IC),920) CNATM(1)(1:idleng), &
                INT(AFAC*XYZS(1,IAT)), &
                INT(AFAC*XYZS(2,IAT)),-1*INT(AFAC*XYZS(3,IAT)), &
                CNATM(2)(1:idleng),INT(AFAC*XYZS(1,IAT+1)), &
                INT(AFAC*XYZS(2,IAT+1)),-1*INT(AFAC*XYZS(3,IAT+1))
        ELSE
           GOTO 200
        ENDIF
     ENDIF
     IAT=IAT+3
     !     FIN ! do atom cards
  ENDDO

  !  connectivity cards; first bond to all atoms

  NCON=NATM
  DO I=1,NATM
     J=I+ISTRT-1
     IF (NATBON(J).GE.1) THEN
        KCON(I)=1+IATBON(1,J)-ISTRT
     ELSE
        KCON(I)=0
     ENDIF
  ENDDO

  !  additional (more than 1) bonds for each atom

  DO I=1,NATM
     J=I+ISTRT-1
     IF (NATBON(J).GE.2) THEN
        DO K=2,NATBON(J)
           KCON(NCON+1)=I
           KCON(NCON+2)=1+IATBON(K,J)-ISTRT
           NCON=NCON+2
        ENDDO
     ENDIF
  ENDDO

  IF (NCON.LT.1000) THEN
     WRITE(CFDAT(1)(54:56),'(I3)') NCON
  ELSE
     WRITE(CFDAT(1)(54:56),'(I3)') MOD(NCON,1000)
     WRITE(CFDAT(1)(76:76),'(I1)') NCON / 1000
  ENDIF

  !  determine format & card count; write card images

  IF (NATM.GE.100) THEN
     NCC=1+(NCON/26)
     IF (0.EQ.MOD(NCON,26)) NCC=NCC-1
     NCRD=NICD+NAC+NCC
     K1=1
     K2=26
     DO I=NICD+1+NAC,NCRD
        IF (K2.GT.NCON) K2=NCON
        WRITE(CFDAT(I),922) (KCON(J),J=K1,K2)
922     FORMAT(26I3,2X)
        K1=K1+26
        K2=K2+26
     ENDDO
  ELSE
     NCC=1+(NCON/40)
     IF (0.EQ.MOD(NCON,40)) NCC=NCC-1
     NCRD=NICD+NAC+NCC
     K1=1
     K2=40
     DO I=NICD+1+NAC,NCRD
        IF (K2.GT.NCON) K2=NCON
        WRITE(CFDAT(I),924) (KCON(J),J=K1,K2)
924     FORMAT(40I2)
        K1=K1+40
        K2=K2+40
     ENDDO
  ENDIF

  NCRD=NICD+NAC+NCC
  WRITE(CFDAT(1)(24:26),'(I3)') NCRD

  !  now write the file

200 CONTINUE
  DO I=1,NCRD
     WRITE (IUNIT,930) CFDAT(I)
  ENDDO
930 FORMAT (A80)
  CLOSE(IUNIT)
  RETURN
END SUBROUTINE MAKFDAT

SUBROUTINE VECMMUL(UA,UB,UC)
  !
  ! Do a single precision 4x4 matrix multiply
  !
  use chm_kinds
  implicit none
  real(chm_real) UA(4,4),UB(4,4),UC(4,4)
  INTEGER I,J,K
  real(chm_real) UCT
  !
  DO I=1,4
     DO J=1,4
        UCT=0.0
        DO K=1,4
           UCT=UCT+UA(I,K)*UB(K,J)
        ENDDO
        UC(I,J)=UCT
     ENDDO
  ENDDO
  !
  RETURN
END SUBROUTINE VECMMUL

SUBROUTINE VECMMULN(UTRANS,XYZ,M,N,S,XYZS)
  !
  ! Do a single precision matrix multiply
  !
  use chm_kinds
  implicit none
  INTEGER M,N,S
  real(chm_real) UTRANS(M,N),XYZ(N,S),XYZS(M,S)
  INTEGER I,J,K
  real(chm_real) UCT
  !
  !
  DO I=1,M
     DO J=1,S
        UCT=0.0
        DO K=1,N
           UCT=UCT+UTRANS(I,K)*XYZ(K,J)
        ENDDO
        XYZS(I,J)=UCT
     ENDDO
  ENDDO
  !
  RETURN
END SUBROUTINE VECMMULN
#else /**/
SUBROUTINE DRAWIT(IUNIT,X,Y,Z,XCOMP,YCOMP,ZCOMP)
  use chm_kinds
  implicit none
  INTEGER IUNIT
  real(chm_real) X(*),Y(*),Z(*)
  real(chm_real) XCOMP(*),YCOMP(*),ZCOMP(*)
  RETURN
END SUBROUTINE DRAWIT
#endif /*  IFN NOGRAPHICS*/

