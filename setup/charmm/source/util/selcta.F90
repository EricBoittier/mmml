module select
  implicit none

  !    NSELCT - The number of selected atoms from a selection array.
  !    NSELCTV- The number of a type of atom from a selection array.
  !    ATMSEL - Get an atom selection and return the number selected

contains

SUBROUTINE SELCTA(COMLYN,COMLEN,FLAGS,X,Y,Z,WMAIN,QCOOR)
  !-----------------------------------------------------------------------
  !     Atom selection. This routine is a front end for atom
  !     selection.
  !
  use chm_kinds
  use dimens_fcm
  use psf
  implicit none
  !
  INTEGER FLAGS(*)
  real(chm_real) X(*),Y(*),Z(*),WMAIN(*)
  LOGICAL   QCOOR
  INTEGER COMLEN
  character(len=*) COMLYN
  INTEGER MODE,DFLT,I
  !
  !     call recursive selection routine
  !
  MODE=0
  DFLT=1
  CALL SELRPN(COMLYN,COMLEN,FLAGS,NATOM,DFLT,MODE,.FALSE.,1,' ',0, &
       RESID,RES,IBASE,SEGID,NICTOT,NSEG,QCOOR,X,Y,Z,QCOOR,1,WMAIN)
  !
  IF(MODE < 0) THEN
     DO I=1,NATOM
        FLAGS(I)=0
     ENDDO
  ENDIF
  RETURN
END SUBROUTINE SELCTA

SUBROUTINE SELCTD(COMLYN,COMLEN,ISLCT,JSLCT,X,Y,Z,WMAIN,QCOOR,ERR)
  !-----------------------------------------------------------------------
  !     Double atom selection. This routine calls atom selection twice.
  !     If only a single atom selection is given in the command line,
  !     both selection arrays are returned with that selection.
  !     If no selection is given, both selections are returned with all
  !     atoms. In the event of an error, the ERR flag is set and no atoms
  !     will be selected.
  !
  !        Bernard R. Brooks    11-Nov-1984
  !
  use chm_kinds
  use dimens_fcm
  use psf
  implicit none
  !
  INTEGER ISLCT(*),JSLCT(*)
  real(chm_real) X(*),Y(*),Z(*),WMAIN(*)
  LOGICAL   QCOOR,ERR
  INTEGER COMLEN
  character(len=*) COMLYN
  INTEGER MODE,I,K,NSLCT,NSLCT2
  !
  !     call recursive selection routine twice
  !
  ERR=.TRUE.
  MODE=0
  CALL SELRPN(COMLYN,COMLEN,ISLCT,NATOM,1,MODE, &
       .FALSE.,1,' ',0,RESID,RES,IBASE,SEGID,NICTOT,NSEG, &
       QCOOR,X,Y,Z,QCOOR,1,WMAIN)
  IF(MODE /= 0) THEN
     CALL WRNDIE(0,'<SELCTD>','ATOM SELECTION ERROR')
     DO I=1,NATOM
        ISLCT(I)=0
     ENDDO
     DO I=1,NATOM
        JSLCT(I)=0
     ENDDO
     RETURN
  ENDIF
  !
  NSLCT=0
  DO K=1,NATOM
     JSLCT(K)=ISLCT(K)
     IF(ISLCT(K) == 1) NSLCT=NSLCT+1
  ENDDO
  IF(NSLCT == 0) THEN
     CALL WRNDIE(0,'<SELCTD>','ZERO ATOMS SELECTED')
     RETURN
  ENDIF
  !
  ! SELECT SECOND SET OF ATOMS
  MODE=1
  CALL SELRPN(COMLYN,COMLEN,JSLCT,NATOM,1,MODE, &
       .FALSE.,1,' ',0,RESID,RES,IBASE,SEGID,NICTOT,NSEG, &
       QCOOR,X,Y,Z,QCOOR,1,WMAIN)
  IF(MODE < 0) THEN
     CALL WRNDIE(0,'<SELCTD>','ATOM SELECTION ERROR FOR SECOND')
     DO I=1,NATOM
        JSLCT(I)=0
     ENDDO
     RETURN
  ENDIF
  !
  NSLCT2=0
  DO K=1,NATOM
     IF(JSLCT(K) == 1) NSLCT2=NSLCT2+1
  ENDDO
  IF(NSLCT2 == 0) THEN
     CALL WRNDIE(0,'<SELCTD>','ZERO ATOMS SELECTED FOR SECOND')
     RETURN
  ENDIF
  !
  ERR=.FALSE.
  RETURN
END SUBROUTINE SELCTD

SUBROUTINE SELRPN(COMLYN,COMLEN,FLAGS,NFLAGS,DFLT,MODE, &
     QTAG,TAGLEN,TAGS,KEY, &
     XRESID,XRES,XIBASE,XSEGID,XNICTO,XNSEG, &
     QCOOR,X,Y,Z, &
     QPROP,NPROP,DATA)
  !-----------------------------------------------------------------------
  !     Recursive atom (tag) selection routine.
  !
  !      COMLYN  -  Command line
  !      COMLEN  -  Length of command line
  !      FLAGS   -  Selection array (over tags or atoms)
  !      NFLAGS  -  Number of tags or atoms
  !      DFLT    -  Default initialization option
  !      MODE    -  Operation mode
  !    *  QTAG    -  Flag defining a tag selection (not atom)  *(unused)
  !    *  TAGLEN  -  Length of each tag
  !    *  TAGS    -  Array of tags
  !    *  KEY     -  Key atoms for tags
  !      XRESID  -  RESID array to use
  !      XRES    -  RES array to use
  !      XIBASE  -  IBASE array to use
  !      XSEGID  -  SEGID array to use
  !      XNICTO  -  NICTOT array to use
  !      XNSEG   -  Number of segments to use
  !      QCOOR   -  Flag defining if coordinates are valid
  !      X,Y,Z,  -  Coordinates
  !      QPROP   -  Flag defining if properties are valid
  !    *  NPROP   -  Number of properties
  !      DATA    -  Property data
  !
  !                    * - vestigial
  !
  !     Recursive atom selection routine.
  !
  !     MODE: if equal 0 on input FLAGS are set to default DFLT if no
  !           selection specified. If equal 1 FLAGS are untouched if
  !           no selection specified.
  !     If on output MODE=-1: SELCTC encountered an error.
  !
  !     7-FEB-83 Axel Brunger
  !     Converted to Fortran and overhauled
  !                        - Bernard Brooks, NIH,  September 1992
  !
  use chm_kinds
  use memory
  implicit none
  !
  real(chm_real),allocatable,dimension(:) :: IARRAY
  character(len=*)   COMLYN
  INTEGER   COMLEN
  INTEGER FLAGS(*)
  INTEGER   NFLAGS, DFLT, MODE
  LOGICAL   QTAG
  INTEGER   TAGLEN ! XXX not used
  character(len=*) TAGS ! XXX not used
  INTEGER   KEY ! XXX not used
  character(len=*)  XRESID(*), XRES(*)
  INTEGER XIBASE(*)
  character(len=*)  XSEGID(*)
  INTEGER   XNICTO(*), XNSEG
  LOGICAL   QCOOR
  real(chm_real) X(*),Y(*),Z(*)
  LOGICAL   QPROP
  INTEGER   NPROP
  real(chm_real)    DATA(*)
  !
  IF(QPROP) THEN
     IF(NPROP /= 1) CALL WRNDIE(-4,'<SELRPN>','Bad NPROP value')
  ENDIF
  !
  call chmalloc('selcta.src','SELRPN','IARRAY',NFLAGS,crl=IARRAY)

  CALL SELRPN2(COMLYN,COMLEN,FLAGS,NFLAGS,DFLT,MODE, &
       XRESID,XRES,XIBASE,XSEGID,XNICTO,XNSEG, &
       QCOOR,X,Y,Z,QPROP,DATA,IARRAY)
  !
  call chmdealloc('selcta.src','SELRPN','IARRAY',NFLAGS,crl=IARRAY)
  RETURN
END SUBROUTINE SELRPN

SUBROUTINE SELRPN2(COMLYN,COMLEN,FLAGS,NFLAGS,DFLT,MODE, &
     XRESID,XRES,XIBASE,XSEGID,XNICTO,XNSEG, &
     QCOOR,X,Y,Z,QPROP,DATA,ARRAY)
  !-----------------------------------------------------------------------
  ! See comments in SELRPN for information.
  !-----------------------------------------------------------------------
  use chm_kinds
  use chm_types
  use dimens_fcm
  use number
  use psf
  use param
  use storage
  use selctam
  use stream
  use string
#if KEY_PBOUND==1
  use pbound    
#endif
#if KEY_GCMC==1
  use gcmc      
#endif
  use chutil
  use usermod,only: usrsel
  use param_store, only: set_param

  implicit none

  character(len=*)   COMLYN
  INTEGER   COMLEN
  INTEGER FLAGS(*)
  INTEGER   NFLAGS, DFLT, MODE
  character(len=*)  XRESID(*), XRES(*)
  INTEGER XIBASE(*)
  character(len=*)  XSEGID(*)
  INTEGER   XNICTO(*), XNSEG
  LOGICAL   QCOOR
  real(chm_real) X(*),Y(*),Z(*)
  LOGICAL   QPROP
  real(chm_real)    DATA(*),ARRAY(:)
  !
  !***FBS MODS
#if KEY_PBOUND==1 /*pbound*/
  real(chm_real) CORR
  real(chm_real) my_nint
#endif /*     (pbound)*/
  !***FBS MODS
  ! local variables
  INTEGER CMLEN
  character(len=MXCMSZ) CMLYN
  !
  INTEGER   RETMAX
  PARAMETER (RETMAX=50)
  INTEGER   RETADR(RETMAX)
  !
  INTEGER      QSTKLN, QINDX1, QINDX2, QLIFT
  logical,pointer,dimension(:) :: QST, QLST
  integer,parameter :: MXLIFT=30
  type(chm_lptr) :: QPOINT(MXLIFT)
  !
  INTEGER   WDMAX, WDLEN, WDSHMX, WDSHLN
  PARAMETER (WDMAX=MAXSKY,WDSHMX=132)
  character(len=WDMAX) WD
  character(len=WDSHMX) WDSH
  !
  LOGICAL   ELEMNT
  LOGICAL   QSHOW, QRANGE, QABS, OK, ERR
  !***FBS MODS
  LOGICAL   QPERI
  !***END OF FBS MODS
  INTEGER   ISEG, IRES, IAT, IATS1
  INTEGER   IAT2, NUMB, NUMB1, NUMB2
  INTEGER   I, ISELE, IV, IVMIN, IVMAX, IBOND
  INTEGER   J, IDLEN, WIDTH, K, IATS2
  INTEGER   IEND, IPROP, IROPR, LEVEL, XNRES
  INTEGER   I700
  real(chm_real) RMAX,RMAX2,DIST2,DX,DY,DZ

  
  real(chm_real) RMAXRT3,D3,RX,RY,RZ
  real(chm_real) XMIN,XMAX,YMIN,YMAX,ZMIN,ZMAX
  real(chm_real) XYZMIN,XYZMAX,XYCMIN,XYCMAX,XBZMIN,XBZMAX,XBCMIN,XBCMAX


  real(chm_real) XREF, YREF, ZREF, CUTOFF, DATAPT, TOL
  !
  INTEGER    MXTGSZ
  PARAMETER (MXTGSZ=20)
  character(len=MXTGSZ)    TAG1
  INTEGER   LTAG1
  character(len=8) SEG1,RES1,RESN1,RESID1,SEG2,RESN2
  INTEGER   LSEG1, LRESN1, LRESD1, LSEG2, LRESN2
  character(len=8) RESID2,TYPEX,TYPE1,TYPE2
  INTEGER   LRESD2,LTYPE1,LTYPE2
  INTEGER   LSEG, LRES, LRESD, LTYPEX, LATTYP
  character(len=8)   ALPH, ALPH1, ALPH2, ATTYP
  character(len=4)   WPROP
  !
  character(len=1) :: SBRA='(',SKET=')'

  interface
    subroutine selprop(wprop,array,nflags,err)
      use chm_kinds
      character(len=4) :: wprop
      real(chm_real) :: array(:)
      integer :: nflags
      logical :: err
    end subroutine selprop
  end interface

  ! initialize working stack
  QSTKLN = NFLAGS
  QLIFT = 0
  QST => null()
  QLST => null()

  WDLEN=0
  ISELE=INDXA(COMLYN,COMLEN,'SELE')
  IEND=INDX(COMLYN,COMLEN,' END',4)
  IF(ISELE == 0 .AND. IEND > 0) THEN
     CALL WRNDIE(0,'<SELRPN>','SELEction keyword missing')
     WDLEN=COMLEN
     IF(WDLEN > WDMAX) WDLEN=WDMAX
     WD=COMLYN(1:WDLEN)
     GOTO 900
     !
  ELSE IF(ISELE > 0 .AND. IEND == 0) THEN
     CALL WRNDIE(0,'<SELRPN>','ENDselection terminator missing')
     WDLEN=COMLEN
     IF(WDLEN > WDMAX) WDLEN=WDMAX
     WD=COMLYN(1:WDLEN)
     GOTO 900
     !
  ELSE IF(ISELE == 0.AND.IEND.EQ.0.AND.MODE.EQ.0) THEN
     DO IAT=1,NFLAGS
        FLAGS(IAT)=DFLT
     ENDDO
     RETURN
     !
  ELSE IF(ISELE == 0.AND.IEND.EQ.0.AND.MODE.EQ.1) THEN
     RETURN
     !
  ELSE IF(ISELE > IEND) THEN
     CALL WRNDIE(0,'<SELRPN>','END preceeds SELE keyword')
     WDLEN=COMLEN
     IF(WDLEN > WDMAX) WDLEN=WDMAX
     WD=COMLYN(1:WDLEN)
     GOTO 900
     !
  ENDIF
  !
  IEND=IEND+3
  CMLEN=IEND-ISELE+1
  CMLYN=COMLYN(ISELE:IEND)
42 CONTINUE
  IF(IEND < COMLEN) THEN
     IEND=IEND+1
     IF(COMLYN(IEND:IEND) /= ' ') GOTO 42
     IEND=IEND-1
  ENDIF
  IF(IEND < COMLEN .AND. ISELE > 1) THEN
     SCRTCH=COMLYN(1:ISELE-1)//COMLYN(IEND+1:COMLEN)
  ELSE IF(IEND < COMLEN) THEN
     SCRTCH=COMLYN(IEND+1:COMLEN)
  ELSE IF(ISELE > 1) THEN
     SCRTCH=COMLYN(1:ISELE-1)
  ELSE
     SCRTCH=' '
  ENDIF
  COMLEN=COMLEN+ISELE-IEND-1
  COMLYN=SCRTCH(1:COMLEN)
  ISELE=1
  !
#if KEY_DEBUG==1
  WRITE(OUTU,44) 'ENTERING ATOM SELECTION PARSING'
  CALL PRNTST(OUTU,CMLYN,CMLEN,10,-1)
  CALL PRNTST(OUTU,COMLYN,COMLEN,10,-1)
44 FORMAT('SELRPN:: ',A)
#endif 
  !
  ! process-recursive-parsing
  !
  QSHOW=.FALSE.
  LEVEL=1
  !
  CALL NXTJUNK(CMLYN,CMLEN,ISELE,WD,WDLEN,WDMAX,QSHOW,OK)
  IF(.NOT.OK) GOTO 900
#if KEY_DEBUG==1
  WRITE(OUTU,44) '   DROP AT LEVEL 0 (11)'
  CALL PRNTST(OUTU,WD,WDLEN,13,-1)
  CALL PRNTST(OUTU,CMLYN,CMLEN,13,-1)
#endif 

  ! ** invoke procedure expression **
  RETADR(LEVEL)=11
  GOTO 1110
1111 CONTINUE
  ! ** return label                **
#if KEY_DEBUG==1
  WRITE(OUTU,44) '   BACK TO LEVEL 0 (11)'
  CALL PRNTST(OUTU,WD,WDLEN,13,-1)
  CALL PRNTST(OUTU,CMLYN,CMLEN,13,-1)
#endif 
  !
  IF (WD(1:3) /= 'END') THEN
     CALL WRNDIE(0,'SELRPN','Unbalanced boolean expression')
     GOTO 900
  ENDIF
  !
  IF (QLIFT /= 1) THEN
     CALL WRNDIE(0,'SELRPN','Problem with STACK usage. Check code')
     GOTO 900
  ENDIF
  !
  IF (LEVEL /= 1) THEN
     CALL WRNDIE(0,'SELRPN','Problem with LEVEL usage. Check code')
     GOTO 900
  ENDIF
  !
  ! ** return      **
  GOTO 9999
  ! ** return      **
  !---- END STARTUP CODE ------------------------------------------------
  !
  !\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
  !//////////////////////////////////////////////////////////////////////
  !
  !---- BEGIN PROCEDURE EXPRESSION --------------------------------------
1110 CONTINUE
  !     go-down-one-level
  LEVEL=LEVEL+1
  IF(LEVEL > RETMAX) THEN
     CALL WRNDIE(-4,'<SELRPN>','Return address overflow (RETMAX)')
     GOTO 900
  ENDIF
  !
#if KEY_DEBUG==1
  WRITE(OUTU,44) '      DROP AT LEVEL 1 (21)'
  CALL PRNTST(OUTU,WD,WDLEN,16,-1)
  CALL PRNTST(OUTU,CMLYN,CMLEN,16,-1)
#endif 
  !
  ! ** invoke procedure term    **
  RETADR(LEVEL)=21
  GOTO 2220
2221 CONTINUE
  ! ** return label             **
#if KEY_DEBUG==1
  WRITE(OUTU,44) '      BACK TO LEVEL 1 (21)'
  CALL PRNTST(OUTU,WD,WDLEN,16,-1)
  CALL PRNTST(OUTU,CMLYN,CMLEN,16,-1)
#endif 
  !
  !=======================================================================
1130 CONTINUE
  IF(WD /= '.OR.') GOTO 1140
  CALL NXTJUNK(CMLYN,CMLEN,ISELE,WD,WDLEN,WDMAX,QSHOW,OK)
  IF(.NOT.OK) GOTO 900
  !
#if KEY_DEBUG==1
  WRITE(OUTU,44) '      DROP AT LEVEL 1 WITHIN .OR. (22)'
  CALL PRNTST(OUTU,WD,WDLEN,16,-1)
  CALL PRNTST(OUTU,CMLYN,CMLEN,16,-1)
#endif 
  ! ** invoke procedure term    **
  RETADR(LEVEL)=22
  GOTO 2220
2222 CONTINUE
  ! ** return label             **
#if KEY_DEBUG==1
  WRITE(OUTU,44) '      BACK AT LEVEL 1 WITHIN .OR. (22)'
  CALL PRNTST(OUTU,WD,WDLEN,16,-1)
  CALL PRNTST(OUTU,CMLYN,CMLEN,16,-1)
#endif 
  !
  ! process-or-operator
  QLST(1:QSTKLN) = QST(1:QSTKLN) .OR. QLST(1:QSTKLN)

  if (.not. back_stack()) goto 900
  goto 1130
  !
1140 CONTINUE
  !=======================================================================
  !
  !     go-up-one-level
  LEVEL=LEVEL-1
  IF(LEVEL <= 0) THEN
     CALL WRNDIE(-4,'<SELRPN>', &
          'Return address underflow. Check code')
     GOTO 900
  ENDIF
  !
#if KEY_DEBUG==1
  WRITE(OUTU,44) '      END OF LEVEL 1'
  CALL PRNTST(OUTU,WD,WDLEN,16,-1)
  CALL PRNTST(OUTU,CMLYN,CMLEN,16,-1)
#endif 
  !
  !     return to address:
  IF (RETADR(LEVEL) == 11) GOTO 1111
  IF (RETADR(LEVEL) == 12) GOTO 1112
  !     unknown-return-address
  CALL WRNDIE(-4,'<SELRPN>','Return address unknown. Check code')
  GOTO 900
  !---- END PROCEDURE EXPRESSION ---------------------------------------
  !
  !\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
  !//////////////////////////////////////////////////////////////////////
  !
  !---- BEGIN PROCEDURE TERM -------------------------------------------
2220 CONTINUE
  ! go-down-one-level
  LEVEL=LEVEL+1
  IF(LEVEL > RETMAX) THEN
     CALL WRNDIE(-4,'<SELRPN>','Return address overflow (RETMAX)')
     GOTO 900
  ENDIF
  !
#if KEY_DEBUG==1
  WRITE(OUTU,44) '         DROP AT LEVEL 2 (31)'
  CALL PRNTST(OUTU,WD,WDLEN,19,-1)
  CALL PRNTST(OUTU,CMLYN,CMLEN,19,-1)
#endif 
  ! ** invoke procedure factor   **
  RETADR(LEVEL)=31
  GOTO 3330
3331 CONTINUE
  ! ** return label              **
#if KEY_DEBUG==1
  WRITE(OUTU,44) '         BACK AT LEVEL 2 (31)'
  CALL PRNTST(OUTU,WD,WDLEN,19,-1)
  CALL PRNTST(OUTU,CMLYN,CMLEN,19,-1)
#endif 
  !
  !=======================================================================
2230 CONTINUE
  IF(WD(1:4) /= '.AND') GOTO 2240
  CALL NXTJUNK(CMLYN,CMLEN,ISELE,WD,WDLEN,WDMAX,QSHOW,OK)
  IF(.NOT.OK) GOTO 900
  !
#if KEY_DEBUG==1
  WRITE(OUTU,44) '         DROP AT LEVEL 2 WITHIN .AND. (32)'
  CALL PRNTST(OUTU,WD,WDLEN,19,-1)
  CALL PRNTST(OUTU,CMLYN,CMLEN,19,-1)
#endif 
  ! ** invoke procedure factor   **
  RETADR(LEVEL)=32
  GOTO 3330
3332 CONTINUE
  ! ** return label              **
  !
#if KEY_DEBUG==1
  WRITE(OUTU,44) '         BACK AT LEVEL 2 WITHIN .AND. (32)'
  CALL PRNTST(OUTU,WD,WDLEN,19,-1)
  CALL PRNTST(OUTU,CMLYN,CMLEN,19,-1)
#endif 
  ! process-and-operator
  QLST(1:QSTKLN) = QST(1:QSTKLN) .AND. QLST(1:QSTKLN)

  if (.not. back_stack()) goto 900
  goto 2230
  !
2240 CONTINUE
  !=======================================================================
  ! go-up-one-level
  LEVEL=LEVEL-1
  IF(LEVEL <= 0) THEN
     CALL WRNDIE(-4,'<SELRPN>', &
          'Return address underflow. Check code')
     GOTO 900
  ENDIF
  !
#if KEY_DEBUG==1
  WRITE(OUTU,44) '         END OF LEVEL 2'
  CALL PRNTST(OUTU,WD,WDLEN,19,-1)
  CALL PRNTST(OUTU,CMLYN,CMLEN,19,-1)
#endif 
  !     return to address:
  IF (RETADR(LEVEL) == 21) GOTO 2221
  IF (RETADR(LEVEL) == 22) GOTO 2222
  !     unknown-return-address
  CALL WRNDIE(-4,'<SELRPN>','Return address unknown. Check code')
  GOTO 900
  !---- END PROCEDURE TERM ----------------------------------------------
  !
  !\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
  !//////////////////////////////////////////////////////////////////////
  !
  !---- BEGIN PROCEDURE TERM2 -------------------------------------------
3330 CONTINUE
  ! go-down-one-level
  LEVEL=LEVEL+1
  IF(LEVEL > RETMAX) THEN
     CALL WRNDIE(-4,'<SELRPN>','Return address overflow (RETMAX)')
     GOTO 900
  ENDIF
  !
#if KEY_DEBUG==1
  WRITE(OUTU,44) '            DROP AT LEVEL 3 (41)'
  CALL PRNTST(OUTU,WD,WDLEN,22,-1)
  CALL PRNTST(OUTU,CMLYN,CMLEN,22,-1)
#endif 
  ! ** invoke procedure factor   **
  RETADR(LEVEL)=41
  GOTO 4440
4441 CONTINUE
#if KEY_DEBUG==1
  WRITE(OUTU,44) '            BACK AT LEVEL 3 (41)'
  CALL PRNTST(OUTU,WD,WDLEN,22,-1)
  CALL PRNTST(OUTU,CMLYN,CMLEN,22,-1)
#endif 
  ! ** return label              **
  !
  !=======================================================================
3340 CONTINUE
  IF(WD(1:4) /= '.ARO') GOTO 3350
  !
  ! process-around-operator
  IF(.NOT.QCOOR) THEN
     CALL WRNDIE(0,'<SELRPN>', &
          '.AROUND. not allowed because coordinates not present')
     GOTO 900
  ENDIF
  !
  CALL NXTJUNK(CMLYN,CMLEN,ISELE,WD,WDLEN,WDMAX,QSHOW,OK)
  IF(.NOT.OK) GOTO 900
  !
#if KEY_DEBUG==1
  WRITE(OUTU,44) '            PROCESS .AROUND. AT LEVEL 3'
  CALL PRNTST(OUTU,WD,WDLEN,22,-1)
  CALL PRNTST(OUTU,CMLYN,CMLEN,22,-1)
#endif 

  RMAX=DECODF(WD,WDLEN)
  RMAX2=RMAX*RMAX
  RMAXRT3=RMAX*SQRT(THREE)

  if (.not. push_stack()) goto 900

! Faster .around. operator - BRB - 6/1/2020

  XMIN= ANUM
  XMAX=-ANUM
  YMIN= ANUM
  YMAX=-ANUM
  ZMIN= ANUM
  ZMAX=-ANUM
  XYZMIN= ANUM
  XYZMAX=-ANUM
  XYCMIN= ANUM
  XYCMAX=-ANUM
  XBZMIN= ANUM
  XBZMAX=-ANUM
  XBCMIN= ANUM
  XBCMAX=-ANUM

  QINDX1=0
  DO IAT=1,NFLAGS
     IF(.NOT. INITIA(IAT,X,Y,Z)) THEN
        CALL WRNDIE(0,'<SELRPN>', &
             'Some atom coordinates unknown')
        !              terminate-parsing
        GOTO 900
     ENDIF
     QINDX1=QINDX1+1
     QST(QINDX1) = QLST(QINDX1)

     IF(QLST(QINDX1)) THEN
! Find "around" atoms limits in 14 different directions
        DX=X(IAT)
        DY=Y(IAT)
        DZ=Z(IAT)

        IF(DX < XMIN) XMIN=DX
        IF(DX > XMAX) XMAX=DX
        IF(DY < YMIN) YMIN=DY
        IF(DY > YMAX) YMAX=DY
        IF(DZ < ZMIN) ZMIN=DZ
        IF(DZ > ZMAX) ZMAX=DZ
        D3=DX+DY+DZ
        IF(D3 < XYZMIN) XYZMIN=D3
        IF(D3 > XYZMAX) XYZMAX=D3
        D3=DX+DY-DZ
        IF(D3 < XYCMIN) XYCMIN=D3
        IF(D3 > XYCMAX) XYCMAX=D3
        D3=DX-DY+DZ
        IF(D3 < XBZMIN) XBZMIN=D3
        IF(D3 > XBZMAX) XBZMAX=D3
        D3=DX-DY-DZ
        IF(D3 < XBCMIN) XBCMIN=D3
        IF(D3 > XBCMAX) XBCMAX=D3
     ENDIF
  ENDDO
  IF(XMIN == ANUM) GOTO 630  ! No atoms selected to search around

  QINDX1=0
  DO IAT=1,NFLAGS
     QINDX1=QINDX1+1
! is atom already selected? then keep without searching
     IF(QST(QINDX1)) GOTO 620
! is the atom close enough to max-min boundaries? If not, then exclude
     RX=X(IAT)
     RY=Y(IAT)
     RZ=Z(IAT)
     IF( RX-RMAX > XMAX ) GOTO 620
     IF( RX+RMAX < XMIN ) GOTO 620
     IF( RY-RMAX > YMAX ) GOTO 620
     IF( RY+RMAX < YMIN ) GOTO 620
     IF( RZ-RMAX > ZMAX ) GOTO 620
     IF( RZ+RMAX < ZMIN ) GOTO 620
     D3=RX+RY+RZ
     IF( D3-RMAXRT3 > XYZMAX ) GOTO 620
     IF( D3+RMAXRT3 < XYZMIN ) GOTO 620
     D3=RX+RY-RZ
     IF( D3-RMAXRT3 > XYCMAX ) GOTO 620
     IF( D3+RMAXRT3 < XYCMIN ) GOTO 620
     D3=RX-RY+RZ
     IF( D3-RMAXRT3 > XBZMAX ) GOTO 620
     IF( D3+RMAXRT3 < XBZMIN ) GOTO 620
     D3=RX-RY-RZ
     IF( D3-RMAXRT3 > XBCMAX ) GOTO 620
     IF( D3+RMAXRT3 < XBCMIN ) GOTO 620
!
! OK... Still here? Atom may be close. Search until found (or not)
     QINDX2=1
     DO IAT2=1,NFLAGS
        IF(QLST(QINDX2)) THEN
           DX=RX-X(IAT2)
           IF(DX >  RMAX) GOTO 610
           IF(DX < -RMAX) GOTO 610
           DY=RY-Y(IAT2)
           IF(DY >  RMAX) GOTO 610
           IF(DY < -RMAX) GOTO 610
           DZ=RZ-Z(IAT2)
           IF(DZ >  RMAX) GOTO 610
           IF(DZ < -RMAX) GOTO 610
           DIST2=DX*DX+DY*DY+DZ*DZ
           IF (DIST2 <= RMAX2) THEN
              QST(QINDX1)=.TRUE.
              GOTO 620
           ENDIF
610        CONTINUE
        ENDIF
        QINDX2=QINDX2+1
     ENDDO
620  CONTINUE
  ENDDO
630 CONTINUE
  !
  QLST(1:QSTKLN) = QST(1:QSTKLN)
  CALL NXTJUNK(CMLYN,CMLEN,ISELE,WD,WDLEN,WDMAX,QSHOW,OK)
  IF(.NOT.OK) GOTO 900

  if (.not. back_stack()) goto 900
  goto 3340
  !
3350 CONTINUE
  !=======================================================================
  IF(WD(1:4) /= '.SUB') GOTO 3360
  !
  ! process-subset-operator
  CALL NXTJUNK(CMLYN,CMLEN,ISELE,WD,WDLEN,WDMAX,QSHOW,OK)
  IF(.NOT.OK) GOTO 900
  IVMIN=DECODI(WD,WDLEN)
  IVMAX=IVMIN
  !
  CALL NXTJUNK(CMLYN,CMLEN,ISELE,WD,WDLEN,WDMAX,QSHOW,OK)
  IF(.NOT.OK) GOTO 900
  !
  QRANGE=(WD == ':')
  IF (QRANGE) THEN
     CALL NXTJUNK(CMLYN,CMLEN,ISELE,WD,WDLEN,WDMAX,QSHOW,OK)
     IF(.NOT.OK) GOTO 900
     IVMAX=DECODI(WD,WDLEN)
     CALL NXTJUNK(CMLYN,CMLEN,ISELE,WD,WDLEN,WDMAX,QSHOW,OK)
     IF(.NOT.OK) GOTO 900
  ENDIF
  !
#if KEY_DEBUG==1
  WRITE(OUTU,44) '            PROCESS .SUBSET. AT LEVEL 3'
  CALL PRNTST(OUTU,WD,WDLEN,22,-1)
  CALL PRNTST(OUTU,CMLYN,CMLEN,22,-1)
#endif 

  if (.not. push_stack()) goto 900
  QST(1:QSTKLN) = .FALSE.
  !
  IV=0
  DO QINDX2 = 1, QSTKLN
     IF(QLST(QINDX2)) THEN
        IV=IV+1
        IF(IV <= IVMAX .AND. IV >= IVMIN) THEN
           QINDX1=QINDX2
           QST(QINDX1)=.TRUE.
        ENDIF
     ENDIF
  ENDDO
  !
  QLST(1:QSTKLN) = QST(1:QSTKLN)

  if (.not. back_stack()) goto 900
  goto 3340
  !
3360 CONTINUE
  !=======================================================================
  !
  ! GO-UP-ONE-LEVEL
  LEVEL=LEVEL-1
  IF(LEVEL <= 0) THEN
     CALL WRNDIE(-4,'<SELRPN>', &
          'Return address underflow. Check code')
     GOTO 900
  ENDIF
  !
#if KEY_DEBUG==1
  WRITE(OUTU,44) '            END OF LEVEL 3'
  CALL PRNTST(OUTU,WD,WDLEN,22,-1)
  CALL PRNTST(OUTU,CMLYN,CMLEN,22,-1)
#endif 
  !     return to address:
  IF (RETADR(LEVEL) == 31) GOTO 3331
  IF (RETADR(LEVEL) == 32) GOTO 3332
  !     unknown-return-address
  CALL WRNDIE(-4,'<SELRPN>','Return address unknown. Check code')
  GOTO 900
  !---- END PROCEDURE TERM2 ----------------------------------------------
  !
  !\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
  !//////////////////////////////////////////////////////////////////////
  !
  !---- BEGIN PROCEDURE FACTOR -------------------------------------------
4440 CONTINUE
  !     go-down-one-level
  LEVEL=LEVEL+1
  IF(LEVEL > RETMAX) THEN
     CALL WRNDIE(-4,'<SELRPN>','Return address overflow (RETMAX)')
     GOTO 900
  ENDIF
  !
#if KEY_DEBUG==1
  WRITE(OUTU,44) '               START AT LEVEL 4'
  CALL PRNTST(OUTU,WD,WDLEN,25,-1)
  CALL PRNTST(OUTU,CMLYN,CMLEN,25,-1)
#endif 
  !=======================================================================
  IF(WD(1:4) /= '.NOT') GOTO 5010
  !
  CALL NXTJUNK(CMLYN,CMLEN,ISELE,WD,WDLEN,WDMAX,QSHOW,OK)
  IF(.NOT.OK) GOTO 900
  !
#if KEY_DEBUG==1
  WRITE(OUTU,44) '               DROP AT LEVEL 4 IN .NOT. (42)'
  CALL PRNTST(OUTU,WD,WDLEN,25,-1)
  CALL PRNTST(OUTU,CMLYN,CMLEN,25,-1)
#endif 
  ! ** invoke procedure factor  **
  RETADR(LEVEL)=42
  GOTO 4440
4442 CONTINUE
  ! ** return label             **
#if KEY_DEBUG==1
  WRITE(OUTU,44) '               BACK AT LEVEL 4 IN .NOT. (42)'
  CALL PRNTST(OUTU,WD,WDLEN,25,-1)
  CALL PRNTST(OUTU,CMLYN,CMLEN,25,-1)
#endif 
  !
  ! process-not-operator
  QST(1:QSTKLN) = .NOT. QST(1:QSTKLN)
  !
  GOTO 6000
5010 CONTINUE
  !=======================================================================
  IF(WD(1:4) /= '.BON') GOTO 5020
  ! selected all atoms bonded to current set
  CALL NXTJUNK(CMLYN,CMLEN,ISELE,WD,WDLEN,WDMAX,QSHOW,OK)
  IF(.NOT.OK) GOTO 900
  !
#if KEY_DEBUG==1
  WRITE(OUTU,44) '               DROP AT LEVEL 4 IN .BONDED. (43)'
  CALL PRNTST(OUTU,WD,WDLEN,25,-1)
  CALL PRNTST(OUTU,CMLYN,CMLEN,25,-1)
#endif 
  ! ** invoke procedure factor  **
  RETADR(LEVEL)=43
  GOTO 4440
4443 CONTINUE
  ! ** return label             **
#if KEY_DEBUG==1
  WRITE(OUTU,44) '               BACK AT LEVEL 4 IN .BONDED. (43)'
  CALL PRNTST(OUTU,WD,WDLEN,25,-1)
  CALL PRNTST(OUTU,CMLYN,CMLEN,25,-1)
#endif 
  !
  ! process-bonded-operator
  !

  if (.not. push_stack()) goto 900
  !
  QST(1:QSTKLN) = .FALSE.
  !
  K=NBOND
  IF(NFLAGS.GT.NATOM .AND. NBONDT2.GT.NBOND) K=NBONDT2 ! allow selection of extra image bonds
  DO IBOND = 1,K
     IAT2 = IB(IBOND)
     QINDX2 = IAT2
     IF (QLST(QINDX2)) THEN
        QINDX1 = JB(IBOND)
        QST(QINDX1) = .TRUE.
     ENDIF
     IAT2 = JB(IBOND)
     QINDX2 = IAT2
     IF (QLST(QINDX2)) THEN
        QINDX1 = IB(IBOND)
        QST(QINDX1) = .TRUE.
     ENDIF
  ENDDO
  QLST(1:QSTKLN) = QST(1:QSTKLN)

  if (.not. back_stack()) goto 900
  goto 6000
  !
5020 CONTINUE
  !=======================================================================
  IF(WD(1:4) /= '.BYR') GOTO 5030
  !
  CALL NXTJUNK(CMLYN,CMLEN,ISELE,WD,WDLEN,WDMAX,QSHOW,OK)
  IF(.NOT.OK) GOTO 900
  !
#if KEY_DEBUG==1
  WRITE(OUTU,44) '               DROP AT LEVEL 4 IN .BYRES. (44)'
  CALL PRNTST(OUTU,WD,WDLEN,25,-1)
  CALL PRNTST(OUTU,CMLYN,CMLEN,25,-1)
#endif 
  ! ** invoke procedure factor  **
  RETADR(LEVEL)=44
  GOTO 4440
4444 CONTINUE
  ! ** return label             **
  !
#if KEY_DEBUG==1
  WRITE(OUTU,44) '               BACK AT LEVEL 4 IN .BYRES. (44)'
  CALL PRNTST(OUTU,WD,WDLEN,25,-1)
  CALL PRNTST(OUTU,CMLYN,CMLEN,25,-1)
#endif 
  ! process-byres-operator
  !
  DO ISEG=1,XNSEG
     DO IRES=XNICTO(ISEG)+1,XNICTO(ISEG+1)
        IAT2=XIBASE(IRES)
        ELEMNT=.FALSE.
        DO WHILE(.NOT.ELEMNT .AND. IAT2 < XIBASE(IRES+1))
           IAT2=IAT2+1
           IF (QST(IAT2)) THEN
              ELEMNT=.TRUE.
              DO IAT=XIBASE(IRES)+1,XIBASE(IRES+1)
                 QST(IAT) = ELEMNT
              ENDDO
           ENDIF
        ENDDO
     ENDDO
  ENDDO
  !
  GOTO 6000
5030 CONTINUE
  !=======================================================================
  IF(WD(1:4) /= '.BYG') GOTO 5040
  !
  CALL NXTJUNK(CMLYN,CMLEN,ISELE,WD,WDLEN,WDMAX,QSHOW,OK)
  IF(.NOT.OK) GOTO 900
  !
#if KEY_DEBUG==1
  WRITE(OUTU,44) '               DROP AT LEVEL 4 IN .BYGROUP. (45)'
  CALL PRNTST(OUTU,WD,WDLEN,25,-1)
  CALL PRNTST(OUTU,CMLYN,CMLEN,25,-1)
#endif 
  ! ** invoke procedure factor  **
  RETADR(LEVEL)=45
  GOTO 4440
4445 CONTINUE
  ! ** return label             **
  !
#if KEY_DEBUG==1
  WRITE(OUTU,44) '               BACK AT LEVEL 4 IN .BYGROUP. (45)'
  CALL PRNTST(OUTU,WD,WDLEN,25,-1)
  CALL PRNTST(OUTU,CMLYN,CMLEN,25,-1)
#endif 
  ! process-bygroup-operator
  !
  DO IRES=1,NGRPT
     IAT2=IGPBS(IRES)
     IF(IAT2 < NFLAGS) THEN
        ELEMNT=.FALSE.
        DO WHILE(.NOT.ELEMNT .AND. IAT2 < IGPBS(IRES+1))
           IAT2=IAT2+1
           IF (QST(IAT2)) THEN
              ELEMNT=.TRUE.
              DO IAT=IGPBS(IRES)+1,IGPBS(IRES+1)
                 QST(IAT) = ELEMNT
              ENDDO
           ENDIF
        ENDDO
     ENDIF
  ENDDO
  !
  GOTO 6000
5040 CONTINUE
  !=======================================================================
  IF(WD /= SBRA) GOTO 5050
  !
  CALL NXTJUNK(CMLYN,CMLEN,ISELE,WD,WDLEN,WDMAX,QSHOW,OK)
  IF(.NOT.OK) GOTO 900
  !
#if KEY_DEBUG==1
  WRITE(OUTU,44) '               DROP AT LEVEL 4 IN "(" (12)'
  CALL PRNTST(OUTU,WD,WDLEN,25,-1)
  CALL PRNTST(OUTU,CMLYN,CMLEN,25,-1)
#endif 
  ! ** invoke procedure expression **
  RETADR(LEVEL)=12
  GOTO 1110
1112 CONTINUE
  ! ** return label                **
#if KEY_DEBUG==1
  WRITE(OUTU,44) '               BACK AT LEVEL 4 IN "(" (12)'
  CALL PRNTST(OUTU,WD,WDLEN,25,-1)
  CALL PRNTST(OUTU,CMLYN,CMLEN,25,-1)
#endif 
  IF (WD /= SKET) THEN
     CALL WRNDIE(0,'<SELRPN>','Unbalanced boolean expression')
     GOTO 900
  ENDIF
  CALL NXTJUNK(CMLYN,CMLEN,ISELE,WD,WDLEN,WDMAX,QSHOW,OK)
  IF(.NOT.OK) GOTO 900
  !
  GOTO 6000
5050 CONTINUE

  if (.not. push_stack()) goto 900

  ! parse-if-stored-selections
  J=0
5060 CONTINUE
  J=J+1
  IF(J > NUMSKY) GOTO 5070
  IF(.NOT.EQST(WD,WDLEN,NAMSKY(J),LNAMSK(J))) GOTO 5060
#if KEY_DEBUG==1
  WRITE(OUTU,44) '               PROCESS KEYWORK AT LEVEL 4'
  CALL PRNTST(OUTU,WD,WDLEN,25,-1)
  CALL PRNTST(OUTU,CMLYN,CMLEN,25,-1)
#endif 
  CALL SELSKY(QST, NFLAGS, PTRSKY(J)%a, LENSKY(J))
  CALL NXTJUNK(CMLYN,CMLEN,ISELE,WD,WDLEN,WDMAX,QSHOW,OK)
  IF(.NOT.OK) GOTO 900
  !
  GOTO 6000
5070 CONTINUE
  !-----------------------------------------------------------------
  !     parse-token
  !-----------------------------------------------------------------
  IF(WD == 'ALL') THEN
#if KEY_DEBUG==1
     WRITE(OUTU,44) '               PROCESS ALL AT LEVEL 4'
     CALL PRNTST(OUTU,WD,WDLEN,25,-1)
     CALL PRNTST(OUTU,CMLYN,CMLEN,25,-1)
#endif 
     QST(1:QSTKLN) = .TRUE.
     CALL NXTJUNK(CMLYN,CMLEN,ISELE,WD,WDLEN,WDMAX,QSHOW,OK)
     IF(.NOT.OK) GOTO 900
     !
     !-----------------------------------------------------------------
  ELSE IF(WD == 'NONE') THEN
#if KEY_DEBUG==1
     WRITE(OUTU,44) '               PROCESS NONE AT LEVEL 4'
     CALL PRNTST(OUTU,WD,WDLEN,25,-1)
     CALL PRNTST(OUTU,CMLYN,CMLEN,25,-1)
#endif 
     QST(1:QSTKLN) = .FALSE.
     CALL NXTJUNK(CMLYN,CMLEN,ISELE,WD,WDLEN,WDMAX,QSHOW,OK)
     IF(.NOT.OK) GOTO 900
     !
     !-----------------------------------------------------------------
  ELSE IF(WD == 'USER') THEN
#if KEY_DEBUG==1
     WRITE(OUTU,44) '               PROCESS USER AT LEVEL 4'
     CALL PRNTST(OUTU,WD,WDLEN,25,-1)
     CALL PRNTST(OUTU,CMLYN,CMLEN,25,-1)
#endif 
     DO IAT=1,NFLAGS
        FLAGS(IAT)=0
     ENDDO
     CALL USRSEL(NFLAGS,FLAGS,X,Y,Z,QCOOR)
     DO IAT=1,NFLAGS
        QST(IAT) = (FLAGS(IAT) /= 0)
     ENDDO
     CALL NXTJUNK(CMLYN,CMLEN,ISELE,WD,WDLEN,WDMAX,QSHOW,OK)
     IF(.NOT.OK) GOTO 900
     !
     !-----------------------------------------------------------------
  ELSE IF(WD(1:4) == 'PREV') THEN
#if KEY_DEBUG==1
     WRITE(OUTU,44) '               PROCESS PREV AT LEVEL 4'
     CALL PRNTST(OUTU,WD,WDLEN,25,-1)
     CALL PRNTST(OUTU,CMLYN,CMLEN,25,-1)
#endif 
     DO IAT=1,NFLAGS
        QST(IAT) = (FLAGS(IAT) == 1)
     ENDDO
     CALL NXTJUNK(CMLYN,CMLEN,ISELE,WD,WDLEN,WDMAX,QSHOW,OK)
     IF(.NOT.OK) GOTO 900
     !
     !-----------------------------------------------------------------
  ELSE IF(WD(1:4) == 'RECA') THEN
#if KEY_DEBUG==1
     WRITE(OUTU,44) '               PROCESS RECALL AT LEVEL 4'
     CALL PRNTST(OUTU,WD,WDLEN,25,-1)
     CALL PRNTST(OUTU,CMLYN,CMLEN,25,-1)
#endif 
     CALL NXTJUNK(CMLYN,CMLEN,ISELE,WD,WDLEN,WDMAX,QSHOW,OK)
     IF(.NOT.OK) GOTO 900
     J=NEXTI(WD,WDLEN)
     IF(.NOT. (J >= 1.AND.J <= MAXSTO)) THEN
        CALL WRNDIE(1,'<SELRPN>','Store number out of range')
        GOTO 900
     ELSE IF(.not. allocated(ptrsto(j)%a)) then
        CALL WRNDIE(1,'<SELRPN>','Store empty')
        GOTO 900
     ELSE IF(ptrSTO(J)%len /= NFLAGS) THEN
        CALL WRNDIE(1,'<SELRPN>', &
             'Dim. mismatch - Cannot use if PSF changes')
        GOTO 900
     ELSE
        CALL SELRCL(QST, PTRSTO(J)%a, NFLAGS)
        CALL NXTJUNK(CMLYN,CMLEN,ISELE,WD,WDLEN,WDMAX,QSHOW,OK)
        IF(.NOT.OK) GOTO 900
     ENDIF
     !
     !-----------------------------------------------------------------
  ELSE IF(WD(1:4) == 'BYNU') THEN
#if KEY_DEBUG==1
     WRITE(OUTU,44) '               PROCESS BYNUM AT LEVEL 4'
     CALL PRNTST(OUTU,WD,WDLEN,25,-1)
     CALL PRNTST(OUTU,CMLYN,CMLEN,25,-1)
#endif 
     CALL NXTJUNK(CMLYN,CMLEN,ISELE,WD,WDLEN,WDMAX,QSHOW,OK)
     IF(.NOT.OK) GOTO 900
     CALL COPYST(ATTYP,8,LATTYP,WD,WDLEN)
     !        inquire-if-range-specified
     CALL NXTJUNK(CMLYN,CMLEN,ISELE,WD,WDLEN,WDMAX,QSHOW,OK)
     IF(.NOT.OK) GOTO 900
     QRANGE=(WD == ':')
     IF (QRANGE) THEN
        IATS1=DECODI(ATTYP,LATTYP)
        IF (IATS1 <= 0.OR.IATS1 > NFLAGS) GOTO 900
        CALL NXTJUNK(CMLYN,CMLEN,ISELE,WD,WDLEN,WDMAX,QSHOW,OK)
        IF(.NOT.OK) GOTO 900
        IATS2=DECODI(WD,WDLEN)
        IF (IATS2 <= 0.OR.IATS2 > NFLAGS) GOTO 900
        CALL NXTJUNK(CMLYN,CMLEN,ISELE,WD,WDLEN,WDMAX,QSHOW,OK)
        IF(.NOT.OK) GOTO 900
     ENDIF
     DO IAT=1,NFLAGS
        IF(QRANGE) THEN
           ELEMNT=(IATS1 <= IAT.AND.IAT.LE.IATS2)
        ELSE
           CALL ENCODI(IAT,TAG1,MXTGSZ,LTAG1)
           ELEMNT=EQSTWC(TAG1,LTAG1,ATTYP,LATTYP)
        ENDIF
        QST(IAT) = ELEMNT
     ENDDO
     !
     !-----------------------------------------------------------------
  ELSE IF(WD(1:4) == 'SEGI') THEN
#if KEY_DEBUG==1
     WRITE(OUTU,44) '               PROCESS SEGI AT LEVEL 4'
     CALL PRNTST(OUTU,WD,WDLEN,25,-1)
     CALL PRNTST(OUTU,CMLYN,CMLEN,25,-1)
#endif 
     CALL NXTJUNK(CMLYN,CMLEN,ISELE,WD,WDLEN,WDMAX,QSHOW,OK)
     IF(.NOT.OK) GOTO 900
     CALL COPYST(SEG1,8,LSEG1,WD,WDLEN)
     !        inquire-if-range-specified
     CALL NXTJUNK(CMLYN,CMLEN,ISELE,WD,WDLEN,WDMAX,QSHOW,OK)
     IF(.NOT.OK) GOTO 900
     QRANGE=(WD == ':')
     IF (QRANGE) THEN
        CALL NXTJUNK(CMLYN,CMLEN,ISELE,WD,WDLEN,WDMAX,QSHOW,OK)
        IF(.NOT.OK) GOTO 900
        CALL COPYST(SEG2,8,LSEG2,WD,WDLEN)
        CALL NXTJUNK(CMLYN,CMLEN,ISELE,WD,WDLEN,WDMAX,QSHOW,OK)
        IF(.NOT.OK) GOTO 900
     ENDIF
     DO ISEG=1,XNSEG
        LSEG=8
        CALL TRIME(XSEGID(ISEG),LSEG)
        IF(QRANGE) THEN
           ELEMNT=LTSTEQ(SEG1,LSEG1,XSEGID(ISEG),LSEG,.TRUE.)
           IF (ELEMNT) ELEMNT=LTSTEQ(XSEGID(ISEG),LSEG, &
                SEG2,LSEG2,.TRUE.)
        ELSE
           ELEMNT=(EQSTWC(XSEGID(ISEG),LSEG,SEG1,LSEG1))
        ENDIF
        DO IRES=XNICTO(ISEG)+1,XNICTO(ISEG+1)
           DO IAT=XIBASE(IRES)+1,XIBASE(IRES+1)
              QST(IAT) = ELEMNT
           ENDDO
        ENDDO
     ENDDO
     !
     !-----------------------------------------------------------------
  ELSE IF(WD(1:4) == 'ISEG') THEN
#if KEY_DEBUG==1
     WRITE(OUTU,44) '               PROCESS ISEG AT LEVEL 4'
     CALL PRNTST(OUTU,WD,WDLEN,25,-1)
     CALL PRNTST(OUTU,CMLYN,CMLEN,25,-1)
#endif 
     CALL NXTJUNK(CMLYN,CMLEN,ISELE,WD,WDLEN,WDMAX,QSHOW,OK)
     IF(.NOT.OK) GOTO 900
     CALL COPYST(SEG1,8,LSEG1,WD,WDLEN)
     !        inquire-if-range-specified
     CALL NXTJUNK(CMLYN,CMLEN,ISELE,WD,WDLEN,WDMAX,QSHOW,OK)
     IF(.NOT.OK) GOTO 900
     QRANGE=(WD == ':')
     IF (QRANGE) THEN
        NUMB1=DECODI(SEG1,LSEG1)
        IF (NUMB1 <= 0.OR.NUMB1 > XNSEG) GOTO 900
        CALL NXTJUNK(CMLYN,CMLEN,ISELE,WD,WDLEN,WDMAX,QSHOW,OK)
        IF(.NOT.OK) GOTO 900
        NUMB2=DECODI(WD,WDLEN)
        IF (NUMB2 <= 0.OR.NUMB2 > XNSEG) GOTO 900
        CALL NXTJUNK(CMLYN,CMLEN,ISELE,WD,WDLEN,WDMAX,QSHOW,OK)
        IF(.NOT.OK) GOTO 900
     ENDIF
     DO ISEG=1,XNSEG
        IF(QRANGE) THEN
           ELEMNT=(NUMB1 <= ISEG.AND.ISEG.LE.NUMB2)
        ELSE
           CALL ENCODI(ISEG,TAG1,MXTGSZ,LTAG1)
           ELEMNT=EQSTWC(TAG1,LTAG1,SEG1,LSEG1)
        ENDIF
        DO IRES=XNICTO(ISEG)+1,XNICTO(ISEG+1)
           DO IAT=XIBASE(IRES)+1,XIBASE(IRES+1)
              QST(IAT) = ELEMNT
           ENDDO
        ENDDO
     ENDDO
     !
     !-----------------------------------------------------------------
  ELSE IF(WD(1:4) == 'RESN') THEN
     CALL NXTJUNK(CMLYN,CMLEN,ISELE,WD,WDLEN,WDMAX,QSHOW,OK)
#if KEY_DEBUG==1
     WRITE(OUTU,44) '               PROCESS RESN AT LEVEL 4'
     CALL PRNTST(OUTU,WD,WDLEN,25,-1)
     CALL PRNTST(OUTU,CMLYN,CMLEN,25,-1)
#endif 
     IF(.NOT.OK) GOTO 900
     CALL COPYST(RESN1,8,LRESN1,WD,WDLEN)
     !        inquire-if-range-specified
     CALL NXTJUNK(CMLYN,CMLEN,ISELE,WD,WDLEN,WDMAX,QSHOW,OK)
     IF(.NOT.OK) GOTO 900
     QRANGE=(WD == ':')
     IF (QRANGE) THEN
        CALL NXTJUNK(CMLYN,CMLEN,ISELE,WD,WDLEN,WDMAX,QSHOW,OK)
        IF(.NOT.OK) GOTO 900
        CALL COPYST(RESN2,8,LRESN2,WD,WDLEN)
        CALL NXTJUNK(CMLYN,CMLEN,ISELE,WD,WDLEN,WDMAX,QSHOW,OK)
        IF(.NOT.OK) GOTO 900
     ENDIF
     DO ISEG=1,XNSEG
        DO IRES=XNICTO(ISEG)+1,XNICTO(ISEG+1)
           LRES=8
           CALL TRIME(XRES(IRES),LRES)
           IF(QRANGE) THEN
              ELEMNT=LTSTEQ(RESN1,LRESN1,XRES(IRES),LRES, &
                   .TRUE.)
              IF (ELEMNT) ELEMNT=LTSTEQ(XRES(IRES),LRES, &
                   RESN2,LRESN2,.TRUE.)
           ELSE
              ELEMNT=(EQSTWC(XRES(IRES),LRES,RESN1,LRESN1))
           ENDIF
           DO IAT=XIBASE(IRES)+1,XIBASE(IRES+1)
              QST(IAT) = ELEMNT
           ENDDO
        ENDDO
     ENDDO
     !
     !-----------------------------------------------------------------
  ELSE IF(WD(1:4) == 'RESI') THEN
#if KEY_DEBUG==1
     WRITE(OUTU,44) '               PROCESS RESI AT LEVEL 4'
     CALL PRNTST(OUTU,WD,WDLEN,25,-1)
     CALL PRNTST(OUTU,CMLYN,CMLEN,25,-1)
#endif 
     CALL NXTJUNK(CMLYN,CMLEN,ISELE,WD,WDLEN,WDMAX,QSHOW,OK)
     IF(.NOT.OK) GOTO 900
     CALL COPYST(RESID1,8,LRESD1,WD,WDLEN)
     !        inquire-if-range-specified
     CALL NXTJUNK(CMLYN,CMLEN,ISELE,WD,WDLEN,WDMAX,QSHOW,OK)
     IF(.NOT.OK) GOTO 900
     QRANGE=(WD == ':')
     IF (QRANGE) THEN
        CALL SPLITI(RESID1,NUMB1,ALPH1,LRESD1)
        CALL NXTJUNK(CMLYN,CMLEN,ISELE,WD,WDLEN,WDMAX,QSHOW,OK)
        IF(.NOT.OK) GOTO 900
        CALL COPYST(RESID2,8,LRESD2,WD,WDLEN)
        CALL SPLITI(RESID2,NUMB2,ALPH2,LRESD2)
        CALL NXTJUNK(CMLYN,CMLEN,ISELE,WD,WDLEN,WDMAX,QSHOW,OK)
        IF(.NOT.OK) GOTO 900
     ENDIF
     DO ISEG=1,XNSEG
        DO IRES=XNICTO(ISEG)+1,XNICTO(ISEG+1)
           LRESD=8
           CALL TRIME(XRESID(IRES),LRESD)
           IF(QRANGE) THEN
              LRESD=8
              CALL SPLITI(XRESID(IRES),NUMB,ALPH,LRESD)
              IF(NUMB1 == NUMB2) THEN
                 ELEMNT=NUMB1 == NUMB
                 IF (ELEMNT) ELEMNT=LTSTEQ(ALPH1,LRESD1, &
                      XRESID(IRES),LRESD,.TRUE.)
                 IF (ELEMNT) ELEMNT=LTSTEQ(XRESID(IRES),LRESD, &
                      ALPH2,LRESD2,.TRUE.)
              ELSE
                 ELEMNT=NUMB1 <= NUMB.AND.NUMB.LE.NUMB2
              ENDIF
           ELSE
              ELEMNT=(EQSTWC(XRESID(IRES),LRESD,RESID1,LRESD1))
           ENDIF
           DO IAT=XIBASE(IRES)+1,XIBASE(IRES+1)
              QST(IAT) = ELEMNT
           ENDDO
        ENDDO
     ENDDO
     !
     !-----------------------------------------------------------------
  ELSE IF(WD(1:4) == 'IRES') THEN
     XNRES=XNICTO(XNSEG+1)
     CALL NXTJUNK(CMLYN,CMLEN,ISELE,WD,WDLEN,WDMAX,QSHOW,OK)
#if KEY_DEBUG==1
     WRITE(OUTU,44) '               PROCESS IRES AT LEVEL 4'
     CALL PRNTST(OUTU,WD,WDLEN,25,-1)
     CALL PRNTST(OUTU,CMLYN,CMLEN,25,-1)
#endif 
     IF(.NOT.OK) GOTO 900
     CALL COPYST(RESID1,8,LRESD1,WD,WDLEN)
     !        inquire-if-range-specified
     CALL NXTJUNK(CMLYN,CMLEN,ISELE,WD,WDLEN,WDMAX,QSHOW,OK)
     IF(.NOT.OK) GOTO 900
     QRANGE=(WD == ':')
     IF (QRANGE) THEN
        NUMB1=DECODI(RESID1,LRESD1)
        IF (NUMB1 <= 0.OR.NUMB1 > XNRES) GOTO 900
        CALL NXTJUNK(CMLYN,CMLEN,ISELE,WD,WDLEN,WDMAX,QSHOW,OK)
        IF(.NOT.OK) GOTO 900
        NUMB2=DECODI(WD,WDLEN)
        IF (NUMB2 <= 0.OR.NUMB2 > XNRES) GOTO 900
        CALL NXTJUNK(CMLYN,CMLEN,ISELE,WD,WDLEN,WDMAX,QSHOW,OK)
        IF(.NOT.OK) GOTO 900
     ENDIF
     DO ISEG=1,XNSEG
        DO IRES=XNICTO(ISEG)+1,XNICTO(ISEG+1)
           IF(QRANGE) THEN
              ELEMNT=(NUMB1 <= IRES.AND.IRES.LE.NUMB2)
           ELSE
              CALL ENCODI(IRES,TAG1,MXTGSZ,LTAG1)
              ELEMNT=EQSTWC(TAG1,LTAG1,RESID1,LRESD1)
           ENDIF
           DO IAT=XIBASE(IRES)+1,XIBASE(IRES+1)
              QST(IAT) = ELEMNT
           ENDDO
        ENDDO
     ENDDO
     !
     !-----------------------------------------------------------------
  ELSE IF(WD(1:4) == 'IGRO') THEN
     CALL NXTJUNK(CMLYN,CMLEN,ISELE,WD,WDLEN,WDMAX,QSHOW,OK)
     IF(.NOT.OK) GOTO 900
#if KEY_DEBUG==1
     WRITE(OUTU,44) '               PROCESS IGROUP AT LEVEL 4'
     CALL PRNTST(OUTU,WD,WDLEN,25,-1)
     CALL PRNTST(OUTU,CMLYN,CMLEN,25,-1)
#endif 
     CALL COPYST(RESID1,8,LRESD1,WD,WDLEN)
     !        inquire-if-range-specified
     CALL NXTJUNK(CMLYN,CMLEN,ISELE,WD,WDLEN,WDMAX,QSHOW,OK)
     IF(.NOT.OK) GOTO 900
     QRANGE=(WD == ':')
     IF (QRANGE) THEN
        NUMB1=DECODI(RESID1,LRESD1)
        IF (NUMB1 <= 0.OR.NUMB1 > NGRPT) GOTO 900
        CALL NXTJUNK(CMLYN,CMLEN,ISELE,WD,WDLEN,WDMAX,QSHOW,OK)
        IF(.NOT.OK) GOTO 900
        NUMB2=DECODI(WD,WDLEN)
        IF (NUMB2 <= 0.OR.NUMB2 > NGRPT) GOTO 900
        CALL NXTJUNK(CMLYN,CMLEN,ISELE,WD,WDLEN,WDMAX,QSHOW,OK)
        IF(.NOT.OK) GOTO 900
     ENDIF
     DO IRES=1,NGRPT
        IF(IGPBS(IRES+1) <= NFLAGS) THEN
           IF(QRANGE) THEN
              ELEMNT=(NUMB1 <= IRES.AND.IRES.LE.NUMB2)
           ELSE
              CALL ENCODI(IRES,TAG1,MXTGSZ,LTAG1)
              ELEMNT=EQSTWC(TAG1,LTAG1,RESID1,LRESD1)
           ENDIF
           DO IAT=IGPBS(IRES)+1,IGPBS(IRES+1)
              QST(IAT) = ELEMNT
           ENDDO
        ENDIF
     ENDDO
     !
     !-----------------------------------------------------------------
  ELSE IF(WD == 'TYPE') THEN
     CALL NXTJUNK(CMLYN,CMLEN,ISELE,WD,WDLEN,WDMAX,QSHOW,OK)
     IF(.NOT.OK) GOTO 900
#if KEY_DEBUG==1
     WRITE(OUTU,44) '               PROCESS TYPE AT LEVEL 4'
     CALL PRNTST(OUTU,WD,WDLEN,25,-1)
     CALL PRNTST(OUTU,CMLYN,CMLEN,25,-1)
#endif 
     CALL COPYST(TYPE1,8,LTYPE1,WD,WDLEN)
     !        inquire-if-range-specified
     CALL NXTJUNK(CMLYN,CMLEN,ISELE,WD,WDLEN,WDMAX,QSHOW,OK)
     IF(.NOT.OK) GOTO 900
     QRANGE=(WD == ':')
     IF (QRANGE) THEN
        CALL NXTJUNK(CMLYN,CMLEN,ISELE,WD,WDLEN,WDMAX,QSHOW,OK)
        IF(.NOT.OK) GOTO 900
        CALL COPYST(TYPE2,8,LTYPE2,WD,WDLEN)
        CALL NXTJUNK(CMLYN,CMLEN,ISELE,WD,WDLEN,WDMAX,QSHOW,OK)
        IF(.NOT.OK) GOTO 900
     ENDIF
     DO IAT=1,NFLAGS
        TYPEX=ATYPE(IAT)
        LTYPEX=8
        CALL TRIME(TYPEX,LTYPEX)
        IF(QRANGE) THEN
           ELEMNT=LTSTEQ(TYPE1,LTYPE1,TYPEX,LTYPEX,.TRUE.)
           IF(ELEMNT) ELEMNT=LTSTEQ(TYPEX,LTYPEX,TYPE2,LTYPE2, &
                .TRUE.)
        ELSE
           ELEMNT=(EQSTWC(TYPEX,LTYPEX,TYPE1,LTYPE1))
        ENDIF
        QST(IAT) = ELEMNT
     ENDDO
     !
     !-----------------------------------------------------------------
  ELSE IF(WD(1:4) == 'CHEM') THEN
     CALL NXTJUNK(CMLYN,CMLEN,ISELE,WD,WDLEN,WDMAX,QSHOW,OK)
     IF(.NOT.OK) GOTO 900
#if KEY_DEBUG==1
     WRITE(OUTU,44) '               PROCESS CHEM AT LEVEL 4'
     CALL PRNTST(OUTU,WD,WDLEN,25,-1)
     CALL PRNTST(OUTU,CMLYN,CMLEN,25,-1)
#endif 
     CALL COPYST(TYPE1,8,LTYPE1,WD,WDLEN)
     !        inquire-if-range-specified
     CALL NXTJUNK(CMLYN,CMLEN,ISELE,WD,WDLEN,WDMAX,QSHOW,OK)
     IF(.NOT.OK) GOTO 900
     QRANGE=(WD == ':')
     IF (QRANGE) THEN
        CALL NXTJUNK(CMLYN,CMLEN,ISELE,WD,WDLEN,WDMAX,QSHOW,OK)
        IF(.NOT.OK) GOTO 900
        CALL COPYST(TYPE2,8,LTYPE2,WD,WDLEN)
        CALL NXTJUNK(CMLYN,CMLEN,ISELE,WD,WDLEN,WDMAX,QSHOW,OK)
        IF(.NOT.OK) GOTO 900
     ENDIF
     DO IAT=1,NFLAGS
        IF(IAC(IAT) > NATC) THEN
           CALL WRNDIE(0,'<SELRPN>', &
                'No VDW parameters available for CHEM token')
           GOTO 900
        ENDIF
        TYPEX=ATC(IAC(IAT))
        LTYPEX=8
        CALL TRIME(TYPEX,LTYPEX)
        IF(QRANGE) THEN
           ELEMNT=LTSTEQ(TYPE1,LTYPE1,TYPEX,LTYPEX,.TRUE.)
           IF (ELEMNT) ELEMNT=LTSTEQ(TYPEX,LTYPEX, &
                TYPE2,LTYPE2,.TRUE.)
        ELSE
           ELEMNT=(EQSTWC(TYPEX,LTYPEX,TYPE1,LTYPE1))
        ENDIF
        QST(IAT) = ELEMNT
     ENDDO
     !
     !-----------------------------------------------------------------
  ELSE IF(WD == 'ATOM') THEN
     CALL NXTJUNK(CMLYN,CMLEN,ISELE,WD,WDLEN,WDMAX,QSHOW,OK)
     IF(.NOT.OK) GOTO 900
#if KEY_DEBUG==1
     WRITE(OUTU,44) '               PROCESS ATOM AT LEVEL 4'
     CALL PRNTST(OUTU,WD,WDLEN,25,-1)
     CALL PRNTST(OUTU,CMLYN,CMLEN,25,-1)
#endif 
     CALL COPYST(SEG1,8,LSEG1,WD,WDLEN)
     CALL NXTJUNK(CMLYN,CMLEN,ISELE,WD,WDLEN,WDMAX,QSHOW,OK)
     IF(.NOT.OK) GOTO 900
     CALL COPYST(RES1,8,LRESD1,WD,WDLEN)
     CALL NXTJUNK(CMLYN,CMLEN,ISELE,WD,WDLEN,WDMAX,QSHOW,OK)
     IF(.NOT.OK) GOTO 900
     CALL COPYST(TAG1,MXTGSZ,LTAG1,WD,WDLEN)
     DO ISEG=1,XNSEG
        DO IRES=XNICTO(ISEG)+1,XNICTO(ISEG+1)
           DO IAT=XIBASE(IRES)+1,XIBASE(IRES+1)
              ELEMNT=EQSTWC(XSEGID(ISEG),8,SEG1,LSEG1)
              IF (ELEMNT) ELEMNT=EQSTWC(XRESID(IRES),8, &
                   RES1,LRESD1)
              IF (ELEMNT) ELEMNT=EQSTWC(ATYPE(IAT),8,TAG1,LTAG1)
              QST(IAT) = ELEMNT
           ENDDO
        ENDDO
     ENDDO
     CALL NXTJUNK(CMLYN,CMLEN,ISELE,WD,WDLEN,WDMAX,QSHOW,OK)
     IF(.NOT.OK) GOTO 900
     !
     !-----------------------------------------------------------------
  ELSE IF(WD(1:4) == 'POIN') THEN
     CALL NXTJUNK(CMLYN,CMLEN,ISELE,WD,WDLEN,WDMAX,QSHOW,OK)
     IF(.NOT.OK) GOTO 900
#if KEY_DEBUG==1
     WRITE(OUTU,44) '               PROCESS POINT AT LEVEL 4'
     CALL PRNTST(OUTU,WD,WDLEN,25,-1)
     CALL PRNTST(OUTU,CMLYN,CMLEN,25,-1)
#endif 
     XREF=DECODF(WD,WDLEN)
     CALL NXTJUNK(CMLYN,CMLEN,ISELE,WD,WDLEN,WDMAX,QSHOW,OK)
     IF(.NOT.OK) GOTO 900
     YREF=DECODF(WD,WDLEN)
     CALL NXTJUNK(CMLYN,CMLEN,ISELE,WD,WDLEN,WDMAX,QSHOW,OK)
     IF(.NOT.OK) GOTO 900
     ZREF=DECODF(WD,WDLEN)
     CALL NXTJUNK(CMLYN,CMLEN,ISELE,WD,WDLEN,WDMAX,QSHOW,OK)
     IF(.NOT.OK) GOTO 900
     IF(WD == 'CUT') THEN
        CALL NXTJUNK(CMLYN,CMLEN,ISELE,WD,WDLEN,WDMAX,QSHOW,OK)
        IF(.NOT.OK) GOTO 900
        RMAX=DECODF(WD,WDLEN)
        CALL NXTJUNK(CMLYN,CMLEN,ISELE,WD,WDLEN,WDMAX,QSHOW,OK)
        IF(.NOT.OK) GOTO 900
     ELSE
        RMAX=8.0
     ENDIF
     !***FBS MODS
     QPERI=.FALSE.
     IF(WD(1:4) == 'PERI') THEN
#if KEY_PBOUND==1 /*pbound_peri*/
        IF(qBoun) THEN
           QPERI=.TRUE.
        ELSE
           CALL WRNDIE(0,'<SELRPN>', &
                'pbound is not set up: PERI is ignored')
        ENDIF
        CALL NXTJUNK(CMLYN,CMLEN,ISELE,WD,WDLEN,WDMAX,QSHOW,OK)
        IF(.NOT.OK) GOTO 900
#else /*  (pbound_peri)*/
        CALL WRNDIE(0,'<SELRPN>', &
             'pbound is not compiled: PERI is ignored')
#endif /* (pbound_peri)*/
     ENDIF
     !***END OF FBS MODS
     RMAX2=RMAX*RMAX
     DO IAT=1,NFLAGS
        DX=XREF-X(IAT)
        DY=YREF-Y(IAT)
        DZ=ZREF-Z(IAT)
        !***FBS MODS
#if KEY_PBOUND==1 /*pbound_main*/
        IF(QPERI) THEN
           If(qCUBoun.or.qTOBoun) then
              DX      = BOXINV * DX
              DY      = BOYINV * DY
              DZ      = BOZINV * DZ
              IF(DX >  HALF) DX = DX - ONE
              IF(DX <  -HALF) DX = DX + ONE
              IF(DY >  HALF) DY = DY - ONE
              IF(DY <  -HALF) DY = DY + ONE
              IF(DZ >  HALF) DZ = DZ - ONE
              IF(DZ <  -HALF) DZ = DZ + ONE
              If (qTOBoun) Then
                 CORR = HALF * AINT ( R75 * ( ABS ( DX ) + &
                      ABS ( DY ) + &
                      ABS ( DZ ) ) )
                 DX      = DX    - SIGN ( CORR,  DX  )
                 DY      = DY    - SIGN ( CORR,  DY  )
                 DZ      = DZ    - SIGN ( CORR,  DZ  )
              Endif
              DX      = XSIZE * DX
              DY      = YSIZE * DY
              DZ      = ZSIZE * DZ
           Else
              Call PBMove(DX, DY, DZ)
           Endif
        ENDIF
#endif /* (pbound_main)*/
        !***FBS MODS
        !
        DIST2=DX*DX+DY*DY+DZ*DZ
        QST(IAT) = (DIST2 <= RMAX2)
     ENDDO
     !
     !-----------------------------------------------------------------
  ELSE IF(WD(1:4) == 'PROP') THEN
     CALL NXTJUNK(CMLYN,CMLEN,ISELE,WD,WDLEN,WDMAX,QSHOW,OK)
     IF(.NOT.OK) GOTO 900
#if KEY_DEBUG==1
     WRITE(OUTU,44) '               PROCESS PROP AT LEVEL 4'
     CALL PRNTST(OUTU,WD,WDLEN,25,-1)
     CALL PRNTST(OUTU,CMLYN,CMLEN,25,-1)
#endif 
     IF(WD == 'ABS') THEN
        QABS=.TRUE.
        CALL NXTJUNK(CMLYN,CMLEN,ISELE,WD,WDLEN,WDMAX,QSHOW,OK)
        IF(.NOT.OK) GOTO 900
     ELSE
        QABS=.FALSE.
     ENDIF
     IF(WD == '1   ') THEN
        IPROP=1
        IF(.NOT.QPROP) THEN
           CALL WRNDIE(0,'<SELRPN>','No properties present')
           GOTO 900
        ENDIF
        CALL WRNDIE(-1,'<SELRPN>', &
             'PROP 1 is obsolete, use: PROP WMAIN or PROP WCOMP')
     ELSE
        IPROP=0
        WPROP=WD
        CALL SELPROP(WPROP,ARRAY,NFLAGS,ERR)
        IF(ERR) GOTO 900
     ENDIF
     CALL NXTJUNK(CMLYN,CMLEN,ISELE,WD,WDLEN,WDMAX,QSHOW,OK)
     IF(.NOT.OK) GOTO 900
     IF(WD == '.LT.' .OR. WD.EQ.'LT') THEN
        IROPR=1
     ELSE IF(WD == '.LE.' .OR. WD.EQ.'LE') THEN
        IROPR=2
     ELSE IF(WD == '.GT.' .OR. WD.EQ.'GT') THEN
        IROPR=3
     ELSE IF(WD == '.GE.' .OR. WD.EQ.'GE') THEN
        IROPR=4
     ELSE IF(WD == '.EQ.' .OR. WD.EQ.'EQ') THEN
        IROPR=5
     ELSE IF(WD == '.NE.' .OR. WD.EQ.'NE') THEN
        IROPR=6
     ELSE IF(WD == '.AE.' .OR. WD.EQ.'AE') THEN
        IROPR=7
     ELSE
        CALL WRNDIE(0,'<SELRPN>','Unknown relational operator')
        GOTO 900
     ENDIF
     CALL NXTJUNK(CMLYN,CMLEN,ISELE,WD,WDLEN,WDMAX,QSHOW,OK)
     IF(.NOT.OK) GOTO 900
     CUTOFF=DECODF(WD,WDLEN)
     TOL=ABS(1.0E-10*CUTOFF)
     DO IAT=1,NFLAGS
        IF(IPROP > 0) THEN
           DATAPT=DATA(IAT)
        ELSE
           DATAPT=ARRAY(IAT)
        ENDIF
        IF (QABS.AND.DATAPT < 0.0) DATAPT=-DATAPT
        !CCCC
        ! Try a more direct simple approach - BRB
        !C            IF(IROPR == 1) THEN
        !C                ELEMNT=((DATAPT-CUTOFF) < -TOL)
        !C            ELSE IF(IROPR == 2) THEN
        !C                ELEMNT=((DATAPT-CUTOFF) <= TOL)
        !C            ELSE IF(IROPR == 3) THEN
        !C                ELEMNT=((DATAPT-CUTOFF) > TOL)
        !C            ELSE IF(IROPR == 4) THEN
        !C                ELEMNT=((DATAPT-CUTOFF) >= -TOL)
        !C            ELSE IF(IROPR == 5) THEN
        !C                ELEMNT=(ABS(DATAPT-CUTOFF) < TOL)
        !C            ELSE IF(IROPR == 6) THEN
        !C                ELEMNT=(ABS(DATAPT-CUTOFF) >= TOL)
        !C            ELSE IF(IROPR == 7) THEN
        !C                ELEMNT=(ABS(DATAPT-CUTOFF) < TOL)
        !C            ENDIF
        !CCCC
        IF(IROPR == 1) THEN
           ELEMNT=(DATAPT < CUTOFF)
        ELSE IF(IROPR == 2) THEN
           ELEMNT=(DATAPT <= CUTOFF)
        ELSE IF(IROPR == 3) THEN
           ELEMNT=(DATAPT > CUTOFF)
        ELSE IF(IROPR == 4) THEN
           ELEMNT=(DATAPT >= CUTOFF)
        ELSE IF(IROPR == 5) THEN
           ELEMNT=(DATAPT == CUTOFF)
        ELSE IF(IROPR == 6) THEN
           ELEMNT=(DATAPT /= CUTOFF)
        ELSE IF(IROPR == 7) THEN
           ELEMNT=(ABS(DATAPT-CUTOFF) < PT0001)
        ENDIF
        !CCCC
        QST(IAT) = ELEMNT
     ENDDO
     CALL NXTJUNK(CMLYN,CMLEN,ISELE,WD,WDLEN,WDMAX,QSHOW,OK)
     IF(.NOT.OK) GOTO 900
     !
     !-----------------------------------------------------------------
  ELSE IF(WD(1:4) == 'INIT') THEN
     CALL NXTJUNK(CMLYN,CMLEN,ISELE,WD,WDLEN,WDMAX,QSHOW,OK)
     IF(.NOT.OK) GOTO 900
#if KEY_DEBUG==1
     WRITE(OUTU,44) '               PROCESS INIT AT LEVEL 4'
     CALL PRNTST(OUTU,WD,WDLEN,25,-1)
     CALL PRNTST(OUTU,CMLYN,CMLEN,25,-1)
#endif 
     DO IAT=1,NFLAGS
        QST(IAT) = INITIA(IAT,X,Y,Z)
     ENDDO
     !
     !-----------------------------------------------------------------
  ELSE IF(WD(1:4) == 'LONE') THEN
     CALL NXTJUNK(CMLYN,CMLEN,ISELE,WD,WDLEN,WDMAX,QSHOW,OK)
     IF(.NOT.OK) GOTO 900
#if KEY_DEBUG==1
     WRITE(OUTU,44) '               PROCESS LONE AT LEVEL 4'
     CALL PRNTST(OUTU,WD,WDLEN,25,-1)
     CALL PRNTST(OUTU,CMLYN,CMLEN,25,-1)
#endif 
     DO IAT=1,NFLAGS
        QST(IAT) = LONE(IAT)
     ENDDO
     !
     !-----------------------------------------------------------------
  ELSE IF(WD(1:4) == 'HYDR') THEN
     CALL NXTJUNK(CMLYN,CMLEN,ISELE,WD,WDLEN,WDMAX,QSHOW,OK)
     IF(.NOT.OK) GOTO 900
#if KEY_DEBUG==1
     WRITE(OUTU,44) '               PROCESS HYDR AT LEVEL 4'
     CALL PRNTST(OUTU,WD,WDLEN,25,-1)
     CALL PRNTST(OUTU,CMLYN,CMLEN,25,-1)
#endif 
     DO IAT=1,NFLAGS
        QST(IAT) = HYDROG(IAT)
     ENDDO
     !
     !-----------------------------------------------------------------
#if KEY_GCMC==1
  ELSE IF(WD(1:4) == 'GCMC') THEN
     if(qgcmc) then
        CALL NXTJUNK(CMLYN,CMLEN,ISELE,WD,WDLEN,WDMAX,QSHOW,OK)
     else
        ok = .false.
     endif
     IF(.NOT.OK) GOTO 900
#if KEY_DEBUG==1
     WRITE(OUTU,44) '               PROCESS GCMC AT LEVEL 4'
     CALL PRNTST(OUTU,WD,WDLEN,25,-1)
     CALL PRNTST(OUTU,CMLYN,CMLEN,25,-1)
#endif 
     DO IAT=1,NFLAGS
        QST(IAT) = GCMCON(IAT)
     ENDDO
#endif 
     !-----------------------------------------------------------------
  ELSE
     IF(WRNLEV >= 2) THEN
        WRITE(OUTU,7003)
        CALL PRNTST(OUTU,WD,WDLEN,1,-1)
     ENDIF
     CALL WRNDIE(0,'<SELRPN>','Unrecognized token')
     GOTO 900
  ENDIF
  !-------------------------
6000 CONTINUE
  !
  !     note that before returning from PARSE-TOKEN the string WD updated
  !
  !     go-up-one-level
  LEVEL=LEVEL-1
  IF(LEVEL <= 0) THEN
     CALL WRNDIE(-4,'<SELRPN>', &
          'Return address underflow. Check code')
     GOTO 900
  ENDIF
  !
#if KEY_DEBUG==1
  WRITE(OUTU,44) '               END OF LEVEL 4'
  CALL PRNTST(OUTU,WD,WDLEN,25,-1)
  CALL PRNTST(OUTU,CMLYN,CMLEN,25,-1)
#endif 
  !     return to address:
  IF (RETADR(LEVEL) == 41) GOTO 4441
  IF (RETADR(LEVEL) == 42) GOTO 4442
  IF (RETADR(LEVEL) == 43) GOTO 4443
  IF (RETADR(LEVEL) == 44) GOTO 4444
  IF (RETADR(LEVEL) == 45) GOTO 4445
  !     unknown-return-address
  CALL WRNDIE(-4,'<SELRPN>','Return address unknown. Check code')
  GOTO 900
  !---- END PROCEDURE FACTOR ---------------------------------------------
  !
  !\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
  !//////////////////////////////////////////////////////////////////////
  !
  !---- BEGIN FINISH UP --------------------------------------------------
9999 CONTINUE
  !
  !
  QINDX1=1
  DO IAT=1,NFLAGS
     IF(QST(QINDX1)) THEN
        FLAGS(IAT)=1
     ELSE
        FLAGS(IAT)=0
     ENDIF
     QINDX1=QINDX1+1
  ENDDO
  IF (QSHOW) THEN
     !
     ! show-the-current-set
     !
     IF(PRNLEV >= 2) WRITE(OUTU,7001)
7001 FORMAT(' The following atoms are currently set:')
     CALL PRNTATSL(FLAGS,NFLAGS,1,XRESID,XRES,XIBASE,ATYPE, &
          XSEGID,XNICTO,XNSEG)
     !----
     !
  ENDIF
  K=0
  DO IAT=1,NFLAGS
     IF (FLAGS(IAT) == 1) THEN
        IF(K == 0) THEN
           CALL set_param('SELATOM',IAT)
           XNRES=XNICTO(XNSEG+1)
           IRES=GETRES(IAT,XIBASE,XNRES)
           CALL set_param('SELIRES',IRES)
           CALL set_param('SELRESI',XRESID(IRES))
           ISEG=GETSEG(IRES,XNICTO,XNSEG)
           CALL set_param('SELISEG',ISEG)
           CALL set_param('SELSEGI',XSEGID(ISEG))
           CALL set_param('SELRESN',RES(IRES))
           CALL set_param('SELTYPE',ATYPE(IAT))
           if(iac(iat) < 1 )then
              CALL set_param('SELCHEM'," ")
           else
              CALL set_param('SELCHEM',ATC(IAC(IAT)))
           endif
        ENDIF
        K=K+1
     ENDIF
  ENDDO
  IF(PRNLEV >= 2) WRITE(OUTU,2000) K,NFLAGS
2000 FORMAT(' SELRPN>',I7,' atoms have been selected out of',I7)
  CALL set_param('NSEL',K)
  !
  IF(K == 0) THEN
     CALL set_param('SELATOM',0)
     CALL set_param('SELIRES',0)
     CALL set_param('SELRESI','NONE')
     CALL set_param('SELISEG',0)
     CALL set_param('SELSEGI','NONE')
     CALL set_param('SELRESN','NONE')
     CALL set_param('SELTYPE','NONE')
     CALL set_param('SELCHEM','NONE')
  ENDIF
  !
  do while (QLIFT >= 1)
     deallocate(QPOINT(QLIFT)%A)
     QLIFT = QLIFT - 1
  enddo
  RETURN
  !
  !
  !=======================================================================
  !
  ! to terminate-parsing
900 CONTINUE
  IF(WRNLEV >= 2) THEN
     WRITE(OUTU,7003)
7003 FORMAT(' Last word parsed in SELRPN: ')
     CALL PRNTST(OUTU,WD,WDLEN,1,-1)
  ENDIF
  do while (QLIFT >= 1)
     deallocate(QPOINT(QLIFT)%A)
     QLIFT = QLIFT - 1
  enddo
  MODE=-1
  RETURN

contains

  ! Adds a level to QPOINT.
  logical function push_stack() result(ok)
#if KEY_DEBUG==1
    WRITE(OUTU,44) '                  PUSH STACK'
44 FORMAT('SELRPN:: ',A)
    CALL PRNTST(OUTU,WD,WDLEN,28,-1)
    CALL PRNTST(OUTU,CMLYN,CMLEN,28,-1)
#endif 
    IF (QLIFT >= MXLIFT) THEN
       CALL WRNDIE(-4,'<SELRPN>', &
            'Stack overflow. Simplify atom selections')
       !     terminate-parsing
       ok=.false.
       return
    ENDIF
    QLIFT = QLIFT + 1
    allocate(QPOINT(QLIFT)%A(QSTKLN))
    QLST => QST
    QST => QPOINT(QLIFT)%A
    ok=.true.
    return
  end function push_stack

  ! Removes a level from QPOINT.
  logical function back_stack() result(ok)
#if KEY_DEBUG==1
    WRITE(OUTU,44) '                  BACK STACK'
44 FORMAT('SELRPN:: ',A)
    CALL PRNTST(OUTU,WD,WDLEN,28,-1)
    CALL PRNTST(OUTU,CMLYN,CMLEN,28,-1)
#endif 
    IF (QLIFT < 1) THEN
       CALL WRNDIE(-4,'<SELRPN>', &
            'Stack underflow. Check code')
       !     terminate-parsing
       ok=.false.
       return
    ENDIF
    deallocate(QPOINT(QLIFT)%A)
    QLIFT = QLIFT - 1
    QST => QLST
    IF (QLIFT > 1) THEN
       QLST => QPOINT(QLIFT-1)%A
    ELSE
       QLST => null()
    ENDIF
    ok=.true.
    return
  end function back_stack

END SUBROUTINE SELRPN2

SUBROUTINE NXTJUNK(CMLYN,CMLEN,ISELE,WD,WDLEN,WDMAX,QSHOW,OK)
  !-----------------------------------------------------------------------
  ! Parse the next word in the atom selection
  !
  use chm_kinds
  use string
  implicit none
  !
  character(len=*) CMLYN
  INTEGER CMLEN,ISELE
  character(len=*) WD
  INTEGER WDLEN,WDMAX
  LOGICAL QSHOW,OK
  !
  character(len=4) WD2
  !
  CALL NEXTOP(CMLYN,ISELE,CMLEN,WD,WDMAX,WDLEN)
  WD2=WD
  IF(WD2 == 'SHOW') THEN
     QSHOW=.TRUE.
     CALL NEXTOP(CMLYN,ISELE,CMLEN,WD,WDMAX,WDLEN)
  ENDIF
  IF (WDLEN == 0) THEN
     CALL WRNDIE(0,'<SELRPN>','Unexpected end during parsing')
     OK=.FALSE.
     RETURN
  ENDIF
  CALL FILSPC(WD,WDMAX,WDLEN)
  OK=.TRUE.
  !
  RETURN
END SUBROUTINE NXTJUNK

SUBROUTINE SELRCL(QSTACK,STORE,NFLAGS)
  !-----------------------------------------------------------------------
  !     Subroutine fills QSTACK according to the values in STORE
  !     QSTACK=(STORE >= 1.0)
  !
  use chm_kinds
  implicit none
  LOGICAL QSTACK(*)
  real(chm_real)  STORE(*)
  INTEGER NFLAGS, I
  !
  DO I=1,NFLAGS
     QSTACK(I)=(STORE(I) >= 1.0)
  ENDDO
  RETURN
END SUBROUTINE SELRCL

SUBROUTINE SELSKY(QSTACK,NFLAGS,STORE,NSTORE)
  !-----------------------------------------------------------------------
  !     Subroutine fills QSTACK according to the values in STORE
  !
  use chm_kinds
  implicit none
  LOGICAL QSTACK(*)
  INTEGER  STORE(*)
  INTEGER NFLAGS, NSTORE
  !
  INTEGER I
  !
  IF(NFLAGS < NSTORE) CALL WRNDIE(-3,'<SELSKY>', &
       'Dim. mismatch - Definitions not valid after PSF change')
  !
  DO I=1,NSTORE
     QSTACK(I)=(STORE(I) == 1)
  ENDDO
  !
  ! Use this code for defines using image atoms.
  IF(NFLAGS > NSTORE) THEN
     DO I=NSTORE+1,NFLAGS
        QSTACK(I)=.FALSE.
     ENDDO
  ENDIF
  !
  RETURN
END SUBROUTINE SELSKY

SUBROUTINE FILSKY(COMLYN,COMLEN,LUSED,QSEL,ISEL)
  !-----------------------------------------------------------------------
  !     THIS ROUTINE FILLS SELECTION KEYWORDS
  !     IF(QSEL) usel ISEL array instead of parsing selection
  !
  use chm_kinds
  use dimens_fcm
  use selctam
  use coord
  use psf
  use memory
  use string
  implicit none
  integer,pointer,dimension(:) :: IPTR
  character(len=*) COMLYN
  INTEGER COMLEN
  LOGICAL LUSED,QSEL
  INTEGER NUMP,J
  INTEGER ISEL(*)
  !
  !     CHECK TO SEE IF SELECTION KEYWORD IS PRESENT.
  !     IF NOT, EXIT
  !
  IF(INDX(COMLYN,COMLEN,'SELE',4) == 0 .AND. .NOT. QSEL) THEN
     CALL WRNDIE(0,'<FILSKY>','No selection keyword specified.')
     LUSED=.FALSE.
     RETURN
  ENDIF
  !
  LUSED=.TRUE.
  NUMP=NUMSKY+1
  call chmalloc('selcta.src','FILSKY','IPTR',NATOM,intgp=IPTR)
  IF(QSEL)THEN
     IPTR=ISEL(1:NATOM)
  ELSE 
    CALL SELCTA(COMLYN,COMLEN,IPTR,X,Y,Z,WMAIN,.TRUE.)
  ENDIF
  !
  CALL TRIME(COMLYN,COMLEN)
  IF(COMLEN <= 0) GOTO 900
  CALL NEXTWD(COMLYN,COMLEN,NAMSKY(NUMP),MNAMSK,LNAMSK(NUMP))
  CALL TRIME(COMLYN,COMLEN)
  IF(COMLEN > 0) GOTO 900
  !
  DO J=1,NUMSKY
     IF(EQST(NAMSKY(NUMP),LNAMSK(NUMP),NAMSKY(J),LNAMSK(J))) THEN
        call chmdealloc('selcta.src','FILSKY','PTRSKY(J)%a',LENSKY(J),intgp=PTRSKY(J)%a)
        PTRSKY(J)%a => IPTR
        LENSKY(J)=NATOM
        RETURN
     ENDIF
  ENDDO
  !
  IF(NUMP >= MAXSKY) THEN
     CALL WRNDIE(0,'<FILSKY>','Overflow in number of definitions.')
     RETURN
  ENDIF
  !
  NUMSKY=NUMP
  LENSKY(NUMP)=NATOM
  PTRSKY(NUMP)%a => IPTR
  RETURN
  !
  ! to crap-out
900 CONTINUE
  CALL WRNDIE(0,'<FILSKY>','Syntax error in DEFIne command')
  call chmdealloc('selcta.src','FILSKY','IPTR',NATOM,intgp=IPTR)
  RETURN
  !
END SUBROUTINE FILSKY

INTEGER FUNCTION NSELCT(NATOM,ISLCT)
  !-----------------------------------------------------------------------
  !     Simply counts the number of selected atoms
  !
  use chm_kinds
  implicit none
  INTEGER NATOM
  INTEGER ISLCT(natom)
  !
  INTEGER NS,NW,I
  !
  NS=0
  NW=0

  DO I=1,NATOM
     IF(ISLCT(I) == 1) THEN
        NS=NS+1
     ELSE IF(ISLCT(I) /= 0) THEN
        NW=NW+1
     ENDIF
  ENDDO
  NSELCT=NS
  IF(NW > 0) THEN
     CALL WRNDIE(-3,'<NSELCT>','Some selection values invalid')
  ENDIF
  RETURN
END FUNCTION NSELCT
INTEGER FUNCTION NSELCTV(NATOM,ISLCT,IVAL)
  !-----------------------------------------------------------------------
  !     Simply counts the number of selected atoms with a particular value
  !
  use chm_kinds
  implicit none
  INTEGER NATOM
  INTEGER ISLCT(*),IVAL
  !
  INTEGER NS,I,IV
  !
  NS=0
  IV=IVAL
  DO I=1,NATOM
     IF(ISLCT(I) == IV) NS=NS+1
  ENDDO
  NSELCTV=NS
  RETURN
END FUNCTION NSELCTV

INTEGER FUNCTION AtmSel ( comlyn, clen, iSel, qComp )
  !-----------------------------------------------------------------------
  !# <caves>-Mar-19-1991 (Leo Caves)
  !
  !     Get an Atom Selection.
  !     MAIN/COMP set can be requested for selections involving
  !     coordinates/weights.
  !     Returns number of selected atoms.
  !
  use chm_kinds
  use dimens_fcm
  use psf
  use coord
  use coordc
  implicit none
  !
  !     Passed Variables
  character(len=*) comlyn
  INTEGER clen, iSel(*)
  LOGICAL qComp
  !
  !...Get the selection.
  IF ( qComp ) THEN
     CALL SelctA ( comlyn, clen, iSel, &
          xCOMP,yCOMP,zCOMP, wCOMP, .TRUE. )
  ELSE
     CALL SelctA ( comlyn, clen, iSel, &
          x,y,z, wmain, .TRUE. )
  ENDIF
  !...Get the no. selected.
  AtmSel = NSelct (nAtom,iSel)
  !
  !...Exit.
  RETURN
END FUNCTION AtmSel

SUBROUTINE PRNTATSL(FLAGS,NFLAGS,IVAL,XRESID,XRES,XIBASE,ATYPE, &
     XSEGID,XNICTO,XNSEG)
  !
  !     This procedure prints a listing of the atoms that are currently
  !     set.
  !-----------------------------------------------------------------------
  use chm_kinds
  use dimens_fcm
  use number
  use stream
  use string
  !
  implicit none
  !
  INTEGER FLAGS(*)
  INTEGER   NFLAGS, IVAL
  character(len=*)  XRESID(*), XRES(*), ATYPE(*)
  INTEGER XIBASE(*)
  character(len=*)  XSEGID(*)
  INTEGER   XNICTO(*), XNSEG
  !
  INTEGER I,J,ISEG,IRES,WIDTH,IDLEN
  INTEGER WDSHMX, WDSHLN
  PARAMETER (WDSHMX=132)
  character(len=WDSHMX) WDSH
  !
  IF(PRNLEV >= 2) WRITE(OUTU,7002)
7002 FORMAT('SEGId RESId RESName  .. TYPEs ..')
  !
  WIDTH=idleng
  DO ISEG=1,XNSEG
     DO IRES=XNICTO(ISEG)+1,XNICTO(ISEG+1)
        WDSHLN=0
        CALL ADDST(WDSH,WDSHMX,WDSHLN,XSEGID(ISEG),idleng)
        CALL ADDSTA(WDSH,WDSHMX,WDSHLN,' ')
        CALL ADDST(WDSH,WDSHMX,WDSHLN,XRESID(IRES),idleng)
        CALL ADDSTA(WDSH,WDSHMX,WDSHLN,' ')
        CALL ADDST(WDSH,WDSHMX,WDSHLN,XRES(IRES),idleng)
        IDLEN=WDSHLN
        J=0
        DO I=XIBASE(IRES)+1,XIBASE(IRES+1)
           IF (FLAGS(I) == IVAL) THEN
              IF (WDSHLN+1+WIDTH > 132) THEN
                 CALL PRNTST(OUTU,WDSH,WDSHLN,1,132)
                 J=0
                 WDSHLN=0
                 CALL CPYSPC(WDSH,WDSHMX,WDSHLN,IDLEN)
              ENDIF
              CALL TRIME(WDSH,WDSHLN)
              CALL ADDSTA(WDSH,WDSHMX,WDSHLN,' ')
              CALL ADDST(WDSH,WDSHMX,WDSHLN,ATYPE(I),idleng)
              J=J+1
           ENDIF
        ENDDO
        IF (J > 0) CALL PRNTST(OUTU,WDSH,WDSHLN,1,132)
     ENDDO
  ENDDO
  !
  RETURN
END SUBROUTINE PRNTATSL

  SUBROUTINE NXTATM(QAT,NQAT,MAXQAT,COMLYN,COMLEN,ISLCT, &
       SEGID,RESID,ATYPE,IBASE,NICTOT,NSEG,RES,NATOM)
    !-----------------------------------------------------------------------
    ! This function parses an atom and return the index of that atom
    ! A return value of zero means the atom was not found.
    !     QAT     - List of selected atoms returned
    !     NQAT    - Number of parsed atoms returned
    !     MAXQAT  - Maximum number of atoms to parse from command line
    !     COMLYN,COMLEN - The command line
    !     ISLCT   - Scratch array for selecting atoms through SELRPN
    !     SEGID,RESID,ATYPE,IBASE,NICTOT,NSEG,RES,NATOM - PSF elements
    !
    use stream
    use dimens_fcm
    use coord
    use chutil
    use string

    INTEGER QAT(*),NQAT,MAXQAT
    CHARACTER(LEN=*) COMLYN
    INTEGER COMLEN,ISLCT(*)
    CHARACTER(LEN=*) ATYPE(*),RESID(*),SEGID(*),RES(*)
    INTEGER NICTOT(*),IBASE(*),NSEG,NATOM
    LOGICAL TIMP

    INTEGER IIRES,IISEG,IATOM,ILEN,NRES,DFLT,MODE,I
    CHARACTER(LEN=4) WRD
    CHARACTER(LEN=8) SEGX,RESX,ATOM
    !
    NQAT=0
    !
    WRD=CURRA4(COMLYN,COMLEN)

    ! check for bynumber keyword.
    IF(WRD == 'BYNU') THEN
       !
       DO WHILE(NQAT < MAXQAT .AND. COMLEN > 0)
          WRD=CURRA4(COMLYN,COMLEN)
          IF(WRD == 'BYNU') WRD=NEXTA4(COMLYN,COMLEN)
          IATOM=NEXTI(COMLYN,COMLEN)
          CALL TRIMA(COMLYN,COMLEN)
          !
          NQAT=NQAT+1
          QAT(NQAT)=IATOM
          IF (IATOM == 0) THEN
             IF(WRNLEV >= 2) WRITE(OUTU,91)
91           FORMAT(' ERROR IN NXTATM: Unrecognizable ATOM number.')
          ENDIF
       ENDDO
       RETURN
    ENDIF
    !
    ! check for atom selection parsing.
    IF(WRD == 'SELE') THEN
       DFLT=0
       MODE=0
       CALL SELRPN(COMLYN,COMLEN,ISLCT,NATOM,DFLT,MODE,.FALSE., &
            1,' ',0,RESID,RES,IBASE,SEGID,NICTOT,NSEG, &
            .TRUE.,X,Y,Z,.TRUE.,1,WMAIN)
       IF(MODE < 0) THEN
          DO I=1,NATOM
             ISLCT(I)=0
          ENDDO
       ENDIF
       DO I=1,NATOM
          IF(ISLCT(I) == 1) THEN
             NQAT=NQAT+1
             QAT(NQAT)=I
             IF(NQAT == MAXQAT) RETURN
          ENDIF
       ENDDO
       RETURN
    ENDIF
    !
    DO WHILE(NQAT < MAXQAT .AND. COMLEN > 0)
       SEGX=CURRA8(COMLYN,COMLEN)
       DO IISEG=1,NSEG
          IF(SEGX == SEGID(IISEG)) GOTO 100
       ENDDO
       !
       ! couldn't find this segment, try as residue number/atom name pair.
       ILEN=4
       CALL SPLITI(SEGX,IIRES,RESX,ILEN)
       !
       IF(IIRES <= 0 .OR. RESX /= ' ') THEN
          ! can't find this atom (bad SEGID or residue number)
          IF(WRNLEV >= 2) WRITE(OUTU,54) SEGX(1:idleng)
54        FORMAT(' ERROR IN NXTATM: Unrecognizable SEGID or residue', &
               ' number "',A,'".')
          NQAT=1
          QAT(NQAT)=0
          COMLEN=0
          RETURN
       ENDIF
       !
       IIRES=NEXTI(COMLYN,COMLEN)
       ATOM=NEXTA8(COMLYN,COMLEN)
       TIMP=(ATOM(1:1) == '*')
       NRES=NICTOT(NSEG+1)
       IATOM=MATOM(IIRES,ATOM,ATYPE,IBASE,1,NRES,.TRUE.)
       IF(IATOM <= 0) THEN
          ! can't find this atom (bad atom name)
          IF(WRNLEV >= 2) WRITE(OUTU,74) ATOM(1:idleng),SEGX(1:idleng)
74        FORMAT(' ERROR IN NXTATM: Unrecognizable ATOM "',A, &
               '" for residue number: ',A)
          IATOM=0
       ENDIF
       GOTO 200
       !
100    CONTINUE
       ! Parse as SEGID,RESID,ATOMID syntax.
       !
       SEGX=NEXTA8(COMLYN,COMLEN)
       RESX=NEXTA8(COMLYN,COMLEN)
       ATOM=NEXTA8(COMLYN,COMLEN)
       CALL TRIMA(COMLYN,COMLEN)
       TIMP=(ATOM(1:1) == '*')
       IATOM=GETATN(SEGX,RESX,ATOM,SEGID,RESID,ATYPE,IBASE,NICTOT,NSEG)
       !
       IF (IATOM == 0) THEN
          ! can't find this atom (bad RESID or ATOMID)
          IF(WRNLEV >= 2) WRITE(OUTU,94) ATOM(1:idleng),SEGX(1:idleng)
94        FORMAT(' ERROR IN NXTATM: Unrecognizable ATOM "',A, &
               '" or RESID "',A,'" for segment "',A,'".')
          IATOM=0
       ENDIF

200    CONTINUE
       IF(TIMP) IATOM=-IATOM
       NQAT=NQAT+1
       QAT(NQAT)=IATOM
    ENDDO
    !
    RETURN
  END SUBROUTINE NXTATM

end module select

