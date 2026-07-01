#if KEY_NOGRAPHICS==1 /*mainnogrhpx*/
SUBROUTINE NULL_AG
  RETURN
END SUBROUTINE NULL_AG
#else /* (mainnogrhpx)*/
SUBROUTINE GRPHINQ(QSIZE)
  !
  !  initializes graphic display characteristics such as
  !  bitmap size, availability of double-buffering & color, etc.
  !  set vars in common block /GRAPH/ used subsequently by GRPHINIT
  !  and for screen coordinate transform, etc.
  !
  use chm_kinds
  use dimens_fcm
  use stream
  use graph
#if KEY_XDISPLAY==1
  use xdraw  
#endif
  implicit none

  LOGICAL QSIZE
  INTEGER I,J
  !
  ! INITIALIZE USCREN(4,4) MATRIX
  ! ....THIS SECTION NEEDS WORK TO GET PROPER ASPECT RATIOS FOR
  !     ALL OF THE VARIOUS MACHINE TYPES....
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
#if KEY_XDISPLAY==1 /*xdisp2*/
  USCREN(1,4)=XXSZ/2
  USCREN(2,4)=XYSZ/2
#endif /* (xdisp2)*/
  !
  RETURN
END SUBROUTINE GRPHINQ

SUBROUTINE GRPHINIT(QSIZE,QFIRST)
  !
  !  open the graphics device; call GRPHINQ first for setup
  !
  use chm_kinds
  use dimens_fcm
  use graph
#if KEY_XDISPLAY==1
  use xdraw  
#endif
  implicit none

  LOGICAL QSIZE,QFIRST
  INTEGER*4 IMAPCNT
  INTEGER IER,I,J

  IF (QFIRST) THEN
     IGRWIDTH=1
     IGRFONT=2
#if KEY_XDISPLAY==1 /*xdisp3*/
     DSTERO=0.5*XXSZ/32.D0
#endif /* (xdisp3)*/
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
     ! initialize the POV graphic object control array
     DO I=1,3
        DO J=1,MAXPVO
           KPVOBJ(I,J) = -1
        ENDDO
     ENDDO
#if KEY_NODISPLAY==0 /*nodisp*/
     IMAPCNT=192
     CALL INICLRMAP(IMAPCNT)
#endif /* (nodisp)*/
#if KEY_XDISPLAY==1 /*xdisp4*/
     !
     !     Initialize X display of size XXSZ,XYSZ
     !
     CALL XINITDISP(XXSZ,XYSZ,XPLA,IMAPCNT,COLOR_MAP)
     !
     !     Fonts...
     !
     CALL XFONTINIT
#endif /* (xdisp4)*/
     !
  ENDIF
  RETURN
END SUBROUTINE GRPHINIT

SUBROUTINE INICLRMAP(ISZ)
  !
  !     Initializes colormap: 16 colors in 12 intensities
  !
  use chm_kinds
  use dimens_fcm
  use graph
  implicit none

  INTEGER ISZ,I,ICOL_REDUCE,J
  !   The HEX constant syntax is maybe HPUX specific
  ! black; hex 000000
  COLOR_MAP(0)= 0
  ! red; hex FF0000
  COLOR_MAP(1)=16711680
  ! light gray; hex C8C8C8
  COLOR_MAP(2)=13158600
  ! yellow; hex FFFF00
  COLOR_MAP(3)=16776960
  ! green; hex 00FF00
  COLOR_MAP(4)=65280
  ! white; hex FFFFFF
  COLOR_MAP(5)=16777215
  ! light blue; hex 8088FF
  COLOR_MAP(6)=8423679
  ! cyan; hex 00FFFF
  COLOR_MAP(7)=65535
  ! magenta; hex FF00FF
  COLOR_MAP(8)=16711935
  ! dark gray; hex 989898
  COLOR_MAP(9)=10000536
  ! orange; hex FFC000
  COLOR_MAP(10)=16760832
  ! brown; hex B06000
  COLOR_MAP(11)=11558912
  ! purple; hex C000B0
  COLOR_MAP(12)=12583088
  ! turquoise; hex 00C0F0
  COLOR_MAP(13)=49392
  ! chartreuse; hex C0FF00
  COLOR_MAP(14)=12648192
  ! blue; hex 0000FF
  COLOR_MAP(15)=255
  IGRZLEV=12
  DO I=1,11
     DO J=0,15
        COLOR_MAP(I*16+J)=ICOL_REDUCE(COLOR_MAP(J),I)
     ENDDO
  ENDDO
  RETURN
END SUBROUTINE INICLRMAP

INTEGER FUNCTION ICOL_REDUCE(ICVAL,ICLEV)
  !
  !  compute the reduced intensity given the original value
  !  and the reduction level
  !
  use chm_kinds
  use dimens_fcm
  use graph
  implicit none

  INTEGER ICVAL,ICLEV,IR,IG,IB
  real(chm_real) ZLEV
  !
  ZLEV=FLOAT(2+IGRZLEV-ICLEV)/(2+IGRZLEV)
  ZLEV=SQRT(ZLEV)
  IB=MOD(ICVAL,256)
  IB=IB*ZLEV
  IG=MOD((ICVAL/256),256)
  IG=IG*ZLEV
  IR=MOD((ICVAL/65536),256)
  IR=IR*ZLEV
  ICOL_REDUCE= IR*65536 + IG*256 + IB
  RETURN
END FUNCTION ICOL_REDUCE

SUBROUTINE GRAFONT(IFNT,LTEXT)
  !
  use chm_kinds
  use dimens_fcm
  use graph
  implicit none

  INTEGER IER,IFNT
  LOGICAL LTEXT
  IF (LTEXT) IGRFONT=IFNT
#if KEY_XDISPLAY==1 /*xdisp5*/
  ! MH-12: Should this be protected against graph nowin?
  IF(.NOT.QNOWIN) CALL XSELFONT(IFNT)
#endif /* (xdisp5)*/
  RETURN
END SUBROUTINE GRAFONT

SUBROUTINE GRPHTERM
  !
  !  terminate graphics mode; free buffers, bitmaps, restore windows
  !
  use chm_kinds
  use dimens_fcm
  use graph
  implicit none
#if KEY_XDISPLAY==1 /*xdisp6*/
  CALL XDISPOFF
#endif /* (xdisp6)*/
  RETURN
END SUBROUTINE GRPHTERM
!
SUBROUTINE GRPSHELL(PATHNAME)
  !-----------------------------------------------------------------------
  !
  ! THIS ROUTINE EXECUTES A SHELL COMMAND
  !
  use chm_kinds
  use stream
  use cstuff, only: fsystem
  implicit none
  CHARACTER(len=*) PATHNAME

  INTEGER PTHLEN, IRET
  !
  PTHLEN=LEN(PATHNAME)
  IF(PRNLEV.GE.3) WRITE(OUTU,44) ' Invoking the command: "', &
       PATHNAME(1:PTHLEN),'"'
44 FORMAT(4A)
  !
  !     Replace call to SYSTEM() with combination of 
  !     vfork() and exec()
  !     to avoid replicating all memory!
  !
  IRET = FSYSTEM(PATHNAME,PTHLEN)
  IF(PRNLEV.GE.3.and.iret<0) WRITE(OUTU,45) iret
45 format('Error: EXE returned ',I4)
  RETURN
END SUBROUTINE GRPSHELL
#endif /* (mainnogrhpx)*/

