module graph
  use chm_kinds
  use dimens_fcm
#if KEY_NOGRAPHICS==0
!
!     CHARMM-GRAPHICS common block
!
!     Purpose: to store information between subsequent calls to
!     CHARMM-GRAPHICS
!
!     QPSCR  - TRUE::GENERATING POSTSCRIPT FILE
!     QPOVR  - TRUE::GENERATING POVRAY 3.x FILE
!     QPSCLR - TRUE::POSTSCRIPT IN COLOR, NOT GREYSCALE
!     QPSBBK - TRUE::POSTSCRIPT IN COLOR, BLACK BACKGROUND
!     QPSORI - TRUE::POSTSCRIPT LANDSCAPE, NOT PORTRAIT MODE
!     QPLUTO - TRUE::GENERATING PLUTO PLOT FILE
!     QTITLE - TRUE::TITLE DISPLAY ENABLED
!     QSTERO - TRUE::STEREO GRAPHICS (SIDE-BY-SIDE)
!     QGRDEV - TRUE::GRAPHICS DEVICE OPEN
!     QAUTO  - TRUE::REDISPLAY MOLECULE AFTER EVERY COMMAND
!     QERASE - TRUE::SCREEN ERASE/NEW PAGE BETWEEN DRAWINGS
!     QDATOM - TRUE::DISPLAY ATOMS AS FILLED CIRCLES
!     QDBOND - TRUE::DISPLAY BONDS
!     QLABEL - TRUE::DISPLAY LABELS
!     QDHBON - TRUE::DISPLAY HBONDS
!     QDMAIN - TRUE::DISPLAY THE MAIN COORDINATES
!     QDCOMP - TRUE::DISPLAY THE COMP COORDINATES
!     QDAXE  - TRUE::DISPLAY LAB FRAME AXES
!     QDVECT - TRUE::DISPLAY THE COMP COORDINATES AS VECTORS
!     QDVEHD - TRUE::INCLUDE ARROW HEADS ON VECTORS (rvenable)
!     QFULLS - TRUE::USE THE FULL SCREEN
!     QZAUTO - TRUE::ZCUE FOLLOWS ZCLIPPING
!     QNOWIN - TRUE::NO WINDOW MODE; PRODUCE DERIVED FILES ONLY
!
!     USCREN(4,4) - TRANSFORM FROM LAB TO SCREEN
!     USTERL(4,4) - SIDE-BY-SIDE STEREO MATRIX LEFT
!     USTERR(4,4) - SIDE-BY-SIDE STEREO MATRIX RIGHT
!     ULAB(4,4)   - TRANSFORM FROM MOLECULE TO LAB
!     ULEFT(4,4)  - COMPLETE LEFT TRANSFOR
!     URIGHT(4,4) - COMPLETE RIGHT TRANSFORM
!
!     AXEXYZ(4,7) - COORDS OF LAB FRAME AXES TIPS, CENTER
!     AXETRN(4,7) - AXES TIPS USING CURRENT GRAPHICS TRANSFORM
!
!     HNPOVRTX         - NO. OF VERTICES ALLOCATED ONTO THE HEAP
!     HPOVRTX          - POINTER TO VERTEX DATA IN THE HEAP
!     HTRNVRTX         - POINTER TO TRANSFORMED VERTEX DATA
!     IGPOVIU          - UNIT NO. FOR POV INCLUDE; OVERRIDE DEFAULTS
!     IPOVOBJ          - UNIT NO. FOR OPTIONAL POV OBJECT FILE
!     MAXPVO           - MAXIMUM NUMBER OF POV OBJECTS
!     NPVOBJ           - NUMBER OF POV OBJECTS
!     PVOTYP(5)        - 1=Sphr, 2=Cyln, 3=Slab, 4=Tria, 5=Smth
!     KPVOBJ(3,MAXPVO) - POV OBJECT CONTROL ARRAY
!                          [1] OBJECT TYPE; SEE PVOTYP
!                          [2] INDEX OFFSET INTO POVRTX ARRAY (0..)
!                          [3] NO. OF VERTICES FOR THE OBJECT
!
!     TSTERO - STEREO ANGLE
!     DSTERO - STEREO SEPARATION DISTANCE IN LAB FRAME
!     GRSCAL - TOTAL GRAPHICS SCALE FACTOR
!     GRDUPA - GRAPHICS DEVICE UNITS PER ANGSTROM
!     ZCLPL - LOW  Z CLIP VALUE
!     ZCLPH - HIGH Z CLIP VALUE
!     ZCUEL - LOW  Z CUE VALUE
!     ZCUEH - HIGH Z CUE VALUE
!     IGRZLEV- NO. OF Z DEPTH CUEING LEVELS
!
!     NGRSEL        - NUMBER OF SELECTED ATOMS
!     ICOLOR(NATOM) - ATOM COLOR CODES
!     ICOLRC(NATOM) - ATOM COLOR CODES FOR COMPARISON
!     RADII(NATOM)  - ATOM RADII FOR EACH ATOM
!     IGRSEL(NATOM) - UTILITY ATOM DISPLAY SELECTION
!     IGRLBL(NATOM) - LABEL DISPLAY CODE; 0=NO, 1-4 = FONT INDEX
!     IGRLLN(NATOM) - LABEL LENGTH
!     IGHBCO        - HBOND COLOR
!     IGHBWI        - HBOND WIDTH
!     IGHBTY        - HBOND LINE TYPE; 0 = NONE, 1-N = DASH LENGTH
!     IGRWIDTH      - LINE WIDTH IN PIXELS
!     IGRASIZ - ATOM SIZE FACTOR; RADIUS OF 1A ATOM AT SCALE 1.0
!     IGRBSIZ - BOND SIZE FACTOR; DIAMETER OF 0.1A
!     IGRTLEN       - GRAPHICS TITLE LINE LENGTH
!     IGRFONT       - FONT SELECTION
!     IGTICO        - GRAPHICS TITLE COLOR
!     IGAXCO        - AXES COLOR
!     IVECCO        - VECTOR COLOR
!     IVECWI        - VECTOR WIDTH
!
!     COLOR_MAP(0:191) - RGB DATA FOR APOLLO,X,POSTSCRIPT
!     NATBON(NATOM) - NUMBER OF BONDS EACH ATOM HAS
!     IATBON(IATBMX,NATOM) - OTHER ATOM OF EACH BOND
!     INDEXR(NATOM) - INDEX FROM PSF TO COMPRESSED ARRAYS
!     INDEXP(NATOM) - INDEX FROM COMPRESSED TO PSF ARRAYS
!     INDEXS(NATOM) - INDEX FROM SORTED TO COMPRESSED ARRAYS
!     INDEXB(NATOM) - INDEX FROM COMPRESSED TO SORTED ARRAYS (HBONDS)
!     TGRLBL(NATOM) - LABEL DISPLAY TEXT
!     GTITLE        - GRAPHICS TITLE TEXT
!
      INTEGER, PARAMETER :: MAXPVO=256
      real(chm_real),allocatable,dimension(:) :: HPOVRTX
      real(chm_real),allocatable,dimension(:) :: HTRNVRTX
      LOGICAL QSTERO,QGRDEV,QAUTO,QTITLE,QLABEL,QDVEHD
      LOGICAL QDMAIN,QDCOMP,QDVECT,QDATOM,QDBOND,QDHBON,QDAXE
      LOGICAL QFULLS,QZAUTO,QPLUTO,QERASE
      LOGICAL QPSCR,QPSCLR,QPSORI,QPSBBK,QNOWIN,QPOVR
      INTEGER NGRSEL,IGRZLEV,NPVOBJ,KPVOBJ(3,MAXPVO)
      INTEGER*4 COLOR_MAP(0:191)
      real(chm_real) USCREN(4,4),USTERL(4,4),USTERR(4,4)
      real(chm_real) ULAB(4,4),ULEFT(4,4),URIGHT(4,4)
      real(chm_real) TSTERO,DSTERO,GRSCAL,ZCLPL,ZCLPH,ZCUEL,ZCUEH
      INTEGER IGTICO,IGAXCO
      INTEGER IGRTLEN,IGHBCO,IGHBWI,IGHBTY
      INTEGER IGPOVIU,IPOVOBJ
      INTEGER IGRWIDTH,IGRFONT
      INTEGER HNPOVRTX
      INTEGER IVECCO,IVECWI
      real(chm_real) IGRASIZ,IGRBSIZ
      real(chm_real) GRDUPA,AXEXYZ(4,7),AXETRN(4,7)
      INTEGER,save,allocatable,dimension(:) :: INDEXP,INDEXR,INDEXS,INDEXB
      INTEGER,save,allocatable,dimension(:) :: ICOLOR,ICOLRC,IGRSEL
      real(chm_real),save,allocatable,dimension(:) :: RADII
      INTEGER,save,allocatable,dimension(:) :: NATBON,IGRLBL,IGRLLN
      INTEGER,save,allocatable,dimension(:,:) :: IATBON
!
      CHARACTER(len=80) GTITLE
      CHARACTER(len=24),save,allocatable,dimension(:) :: TGRLBL
      CHARACTER(len=4) PVOTYP(5)
!
contains
  subroutine graph_ltm_init()
    use memory
    qgrdev=.false.
    call chmalloc('graph_ltm.src','graph_ltm_init','indexp',maxa,intg=indexp)
    call chmalloc('graph_ltm.src','graph_ltm_init','indexr',maxa,intg=indexr)
    call chmalloc('graph_ltm.src','graph_ltm_init','indexs',maxa,intg=indexs)
    call chmalloc('graph_ltm.src','graph_ltm_init','indexb',maxa,intg=indexb)
    call chmalloc('graph_ltm.src','graph_ltm_init','icolor',maxa,intg=icolor)
    call chmalloc('graph_ltm.src','graph_ltm_init','icolrc',maxa,intg=icolrc)
    call chmalloc('graph_ltm.src','graph_ltm_init','igrsel',maxa,intg=igrsel)
    call chmalloc('graph_ltm.src','graph_ltm_init','natbon',maxa,intg=natbon)
    call chmalloc('graph_ltm.src','graph_ltm_init','igrlbl',maxa,intg=igrlbl)
    call chmalloc('graph_ltm.src','graph_ltm_init','igrlln',maxa,intg=igrlln)
    call chmalloc('graph_ltm.src','graph_ltm_init','radii',maxa,crl=radii)
    allocate(tgrlbl(maxa),iatbon(iatbmx,maxa))
    return
  end subroutine graph_ltm_init
#endif 
!
end module graph

