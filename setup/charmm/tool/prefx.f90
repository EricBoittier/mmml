PROGRAM PREFLX
!=======================================================================
!
! Preprocess CHARMM source to handle machine control
!    and INCLUDE files in a consistant manner.
!
!  compiler keywords read from file "preflx.dat" or "pref.dat"
!
! processed control commands (examples)
!
!-----------------------------------------------------------------------
!  ##IF    SCALAR         - process following code if "scalar" on list.
!  ##IFN   VAX SUN IRIS   - process following code if none on list.
!  ##ELIF  APOLLO         - after and IF, alternate processing.
!  ##ELSE                 - after and IF or IFN, process the rest.
!  ##ENDIF                - terminates IF, IFN, ELSE, or ELIF.
!-----------------------------------------------------------------------
!   - Any flag may be preceeded by the 5 characters ".not." (lower case)
!     to specify the complement of the flag (e.g. ".not.REPLICA").
!-----------------------------------------------------------------------
!  ##ERROR 'text'         - indicates error within IF section
!                     (usually used after an ##ELSE condition)
!-----------------------------------------------------------------------
!  ##KEYWORDS LIST unit   - inserts a fortran write lines of all current
!                           keywords to the selected write unit.
!                           "unit" may be a variable name or a number
!                           but it is limited to 8 characters maximum.
!  ##KEYWORDS FILL count array  - fills integer variable count with the
!                           number of active keys and a character*12
!                           array in the program with the current keys.
!-----------------------------------------------------------------------
!  ##INCLUDE '~/charmm_fcm/psf.fcm'   - includes a file inline
!   (some parsing of name may be done and the directory may be modified)
!              (parsing of name is performed on non-UNIX machines)
!
!-----------------------------------------------------------------------
!  ##USE some_module_name,only:some_variable  - includes a use statement
!
!-----------------------------------------------------------------------
!  ##EXPAND flag1 flag2 ...  .when.  flagc1 flagc2 ... (identifier)
!                         - setup a section for code expansion 1
!     Expand subcommands (immediately following the ##EXPAND):
!           ##PASS1 flag1 flag2 ...
!           ##PASS2 flag1 flag2 ...
!           ##PASS3 ...   - code sections and conditions for each pass
!           ##PASS[n] ...
!           ##EXFIN       - code section for the termination of the expand section
!           ##EXEND       - end of expansion specification
!  ##ENDEX   (identifier)
!     (the identifier is required and must match the corresponding ##EXPAND).
!      For each pass, the specified flags are temporarily set (or .not. set)
!      as requested.  If all of the conditions for the code expansion (flags
!      specified after the .when. construct) are not set, then all flags from
!      the ##EXPAND line (before the .when.) are temporarily set and no code
!      expansion is processed.
!-----------------------------------------------------------------------
!  Inline substitution:  {V*} is replaced by the most recent two letter
!  keyword starting with the letter "V".  This only works with any two
!  letter keywords and issues an error if none is found.
!-----------------------------------------------------------------------
!  ##SET   flag1 flag2 .. - set a flag at the current expand level
!-----------------------------------------------------------------------
! nesting of IF's is allowed, parsing is allowed in included files.
! nesting of include files is allowed.
!-----------------------------------------------------------------------
!   - Allows the construct " !## flag1 flag2 ..." at the end of a fortran line
!     for conditional processing.
!     For example, the following constructs are equivalent:
!           Standard format:
!                 ##IF LONGLINE
!                       QLONGL=.TRUE.
!                 ##ELSE
!                       QLONGL=.FALSE.
!                 ##ENDIF
!           Compact format:
!                       QLONGL=.TRUE.     !##LONGLINE        ! comments
!                       QLONGL=.FALSE.    !##.not.LONGLINE   ! comments
!     Both "and" and "or" conditions can be used:
!           !##PERT  !##PARALLEL   - An "AND" conditional compile
!           !##PERT PARALLEL       - AN "OR" conditional compile
!
!   - Allows "!" comments following valid FORTRAN lines
!-----------------------------------------------------------------------
!   - Allows control line identifiers.  For example:
!         ##IF PERT (pertprint)
!            ...
!         ##ELSE    (pertprint)
!            ...
!         ##ENDIF   (pertprint)
!     This construct checks the identifiers and prints a fatal
!     error message if the identifiers do not match.
!-----------------------------------------------------------------------
!   - Allows the include directory to be specified in preflx.dat
!         FCMDIR=directory_name ! Use the specified directory
!         FCMDIR=LOCAL          ! Use the local directory
!         FCMDIR=CURRENT        ! Use the one in the ##INCLUDE line
!-----------------------------------------------------------------------
!  Keyword Rules
!
!   - Keyword may not exceed 12 characters in length.
!   - Global keywords should be all uppercase
!   - Local  keywords should be all lowercase
!   - Keywords should otherwiswe follow fortran standards for naming
!   - Recommendation: Avoid one and two letter keywords
!-----------------------------------------------------------------------
!  Reserved Keywords when Running PREFLX
!
!   SINGLE        - will invoke a conversion to SP.
!   PUTFCM        - include files are explicitly copied to the source.
!   VMS           - Use VAX VMS directory names are
!   REMIMPNON     - Remove all "IMPLICIT NONE" lines
!   UPPERCASE     - Convert all source code (not text) to uppercase
!   INCOMMENTS ** - copy comments from all included files.
!   FCMDIR        - (See above)
!   END           - Stop reading keywords (any following keywords are ignored)
!
!      ** - May be specified by "##SET" or "##SET .not." lines in the source
!-----------------------------------------------------------------------
!   - Tabs in the source code are fatal errors.
!-----------------------------------------------------------------------
!   - Checks for fortran line lengths exceeding 72 characters.
!-----------------------------------------------------------------------
!   - For CHARMM usage, the following conventions are used:
!         Global flags (set in the file pref.dat or preflx.dat):
!                Upper case (e.g. "PERT")
!         Local flags (defined by ##SET or ##EXPAND or ##PASSSn):
!                Lower case (e.g. "nopert")
!         Local non-expand flags (##EXPAND line)
!                Single character uppercase (e.g. "P")
!-----------------------------------------------------------------------
!
! WARNING:: This source file has PREFLX constructs.  To use this
!     without a previous version of PREF, PREFLX, or PREFX,
!     some hand editing is required (search for "##").
!
!================================================================
!  Original version: Bernard R. Brooks  - October 31, 1989
!  Contributions from Youngdo Won and Ryszard Czerminski
!  Overhauled November 4, 1993 - BRB
!  Code expansion added October 21, 1996 - BRB
!  List option added July 29, 2003 - BRB
!  Added line numbers to expansion, converted to F95 - Mike Garrahan, 2010
!
! MXKEYS   - The maximum number of active conditional keys
! MXEXPL   - The maximum number of expand levels
! MXCLN    - maximum number of control lines between EXPAND and EXEND
   INTEGER,PARAMETER :: MXKEYS=300, MXEXPL=20
   INTEGER,PARAMETER :: MAXCLN=50

! Information associated with one level of EXPAND nesting.
   type expand_context
      integer :: unit
      character(len=12) :: fname
      integer :: lineno, offset
      character(len=12) :: keys(MXKEYS)
      integer :: nkeys
      character(len=120) :: ctrl_text(MAXCLN)
      integer :: ctrl_index
   end type expand_context

   call driver

contains

   subroutine driver
      implicit none

! NSTRM    - current stream number for include file
! MXSTRM   - maximum file unit number for nested include files
! OUTU     - output unit for fortran source file
! LSTRM(*) - The ## structure level at entry for each include file.
!            This prevents ## structures from starting
!            and ending in different files.
      integer nstrm,outu
      integer,parameter :: mxstrm=80
      integer lstrm(mxstrm)
! NLEV     - The current ## structure level (how many nested IFs)
! MXLEV    - The maximum number of nested IFs
! ILEV(*)  - Level processing code:
!               -1 = do not export any code for the rest of this construct
!                0 = do not export code, but do so on a subsequent ##ELIF 
!                    where conditions are true, or on a subsequent ##ELSE. 
!                1 = export code
! LEVID(*) - Character data used for checking for ID mismatches.
! MATCHID  - local variable for the current ID
      integer nlev
      integer,parameter :: mxlev=40
      integer ilev(mxlev)
      character(len=120) ::  levid(mxlev),matchid

! IEXPL    - The current expand level
! QEXP     - The flag for each include file indicating an expanded section
      integer iexpl
      type(expand_context),target :: exstk(mxexpl)
      type(expand_context),pointer :: ctx, parent, child
      logical qexp(mxexpl+21)
      integer lnkeys(mxkeys)

! LLENA    - The length of the current input line
! LINEA    - The current input line
! CFEED    - A comment line full of hyphens
! FCMDIR   - Name of directory in which to seek include files
      integer llena,llenb
      character(len=1000) :: linea,lineb
      character(len=72) :: cfeed
      character(len=60) :: fcmdir,fcmd1
!
! just temporary variables
      INTEGER I,IPT,JPT,KPT,IKEY
      character(len=8) :: UNITNAME
!
! MATCH    - Logical flag indicting that a keyname match was found
! TABCH    - Logical flag to check for TABs 
! LBLANK   - Logical flag indicating a ' ' has been found
! LPAREN   - Logical flag indicating a '(' has been found
! PUTFCM   - Logical flag to copy and process include files
! SINGLE   - Logical flag to convert to single precision
! QVMS     - Logical flag to create VAX like directory names
! QRIMP    - Logical flag to remove IMPLICIT NONE statements in the code
! QUPPRC   - Logical flag to convert code to upper case before processing
! QINCOM   - Logical flag to include comments in included files
!
      LOGICAL MATCH,TABCH,PUTFCM,SINGLE,QVMS
      LOGICAL QRIMP,QUPPRC,QINCOM

      logical freeform

!
!.##IF MACINTOSH
!.##ENDIF
!
! STIN     - unit number for initial input
! STOUT    - unit number for output
! MSGU     - unit number for error messages
      INTEGER STIN,STOUT,MSGU
!.##IF APOLLO MACINTOSH
!.##ELSE
      PARAMETER (STIN=5,STOUT=6,MSGU=3)
!.##ENDIF
      SAVE
!
! Check for tabs?
      TABCH =.TRUE.

   !-------- default not fixed format ---------------------------
      freeform = .true.
!
! start of code
!
! On Apollo, read from unit 20(+) and write to unit 19.
      IEXPL = 1
      ctx => exstk(IEXPL)
      ctx%lineno = 0
      ctx%fname = ''
      ctx%unit = STIN
      OUTU = STOUT
!
! create CFEED
      CFEED(1:1)='!'
      DO I=2,72
         CFEED(I:I)='-'
      ENDDO
!
! open the input and output files if appropriate (machine dependent)
      CALL OPENF(ctx%unit,OUTU,MSGU)
!
! get keynames for this run
      CALL SETKYS(ctx%nkeys,ctx%keys,FCMDIR,MSGU)
!
! Check to see if conversion to single precision is warranted.
! !!!Note: Only the "SINGLE" key will cause conversion to SP. - BRB !!!
      SINGLE=.FALSE.
      DO IKEY=1,ctx%nkeys
         IF(ctx%keys(IKEY) == 'SINGLE') SINGLE=.TRUE.
      ENDDO
!
! Copy include files?  We may need to change this...
!.##IF APOLLO
!.##ELSE
      PUTFCM=.FALSE.
      DO IKEY=1,ctx%nkeys
         IF(ctx%keys(IKEY) == 'PUTFCM') PUTFCM=.TRUE.
      ENDDO
!.##ENDIF
!
! Is it a VAX?
      QVMS=.FALSE.
      DO IKEY=1,ctx%nkeys
         IF(ctx%keys(IKEY) == 'VMS') QVMS=.TRUE.
      ENDDO
!
! Remove IMPLICIT NONE?
      QRIMP=.FALSE.
      DO IKEY=1,ctx%nkeys
         IF(ctx%keys(IKEY) == 'REMIMPNON') QRIMP=.TRUE.
      ENDDO
!
! Convert code to uppercase?
      QUPPRC=.FALSE.
      DO IKEY=1,ctx%nkeys
         IF(ctx%keys(IKEY) == 'UPPERCASE') QUPPRC=.TRUE.
      ENDDO
!
! Include comments?
      QINCOM=.FALSE.
      DO IKEY=1,ctx%nkeys
         IF(ctx%keys(IKEY) == 'INCOMMENTS') QINCOM=.TRUE.
      ENDDO
!
! some Macintosh code so that it can do all of the files in a single run.
! It gets the files from a list found on the file "preflx.list".
!.##IF MACINTOSH
!.##ENDIF
!
!
! setup values to start a new file for processing
      NLEV=1
      ILEV(1)=1
      LSTRM(1)=1
      LEVID(1)='LEVEL 1'
!
!=======================================================================
!
! MAIN LOOP START
!
 100  CONTINUE
! read the next line
      READ(ctx%unit,20,END=200) LINEA
      ctx%lineno = ctx%lineno + 1
  20  FORMAT(20A)
! Always output freeform from now on
!      freeform = freeform .or. (index(LINEA,"FREEFORMFTN") > 0)
! See if there are any "!" on this line and remove flags not set
      CALL CHKLINE(LINEA,LLENA,ctx,MSGU, &
                   TABCH,QUPPRC,freeform)
! search for and comment out 'IMPLICIT NONE' lines
      IF(QRIMP) THEN
         I=INDEX(LINEA,'IMPLICIT NONE')
         IF (I > 0) LINEA(1:1)='!'
      ENDIF
!
!=======================================================================
! process non control lines
      IF(LINEA(1:2) /=  '##') THEN
         IF(ILEV(NLEV) == 1) THEN
            CALL OUTLINE(LINEA,OUTU,ctx%unit,STIN,MSGU,SINGLE,QINCOM &
                 ,freeform)
         ELSE
            WRITE(OUTU,20)
         ENDIF
!=======================================================================
! process control line
      ELSE IF(LINEA(1:5) == '##IF ') THEN
! process-if-command
!   check-for-keyname-match
         CALL KEYMATCH(LINEA,LLENA,MATCH,MATCHID,.FALSE.,ctx)
!------ This will go at the beginning of the file instead of in pref.dat --------
!         if(index(matchid,"FREEFORMATFORTRAN")  /= 0)then
!            print *,"Found FREEFORMATFORTRAN"
!            freeform=.true.
!         endif
!-------------------------------------------------------------------------------
! add a new level
         NLEV=NLEV+1
         IF(NLEV > MXLEV) STOP 'FATAL: TOO MANY LEVELS'
         LEVID(NLEV)=MATCHID
         CALL WRITE_COMMENT(ILEV,NLEV,OUTU,LINEA,LLENA)
! set the correct ILEV value for this new ##IF construct
         IF(ILEV(NLEV-1) == 1) THEN
            IF(MATCH) THEN
               ILEV(NLEV)=1
            ELSE
               ILEV(NLEV)=0
            ENDIF
         ELSE
            ILEV(NLEV)=-1
         ENDIF
!=======================================================================
      ELSE IF(LINEA(1:6) == '##IFN ') THEN
! process-ifn-command
!   check-for-keyname-match
         CALL KEYMATCH(LINEA,LLENA,MATCH,MATCHID,.FALSE.,ctx)
! add a new level
         NLEV=NLEV+1
         IF(NLEV > MXLEV) STOP 'FATAL: TOO MANY LEVELS'
         LEVID(NLEV)=MATCHID
         CALL WRITE_COMMENT(ILEV,NLEV,OUTU,LINEA,LLENA)
! set the correct ILEV value for this new ##IFN construct
         IF(ILEV(NLEV-1) == 1) THEN
            IF(MATCH) THEN
               ILEV(NLEV)=0
            ELSE
               ILEV(NLEV)=1
            ENDIF
         ELSE
            ILEV(NLEV)=-1
         ENDIF
!=======================================================================
      ELSE IF((LINEA(1:7) == '##ELIF ') .OR. &
              (LINEA(1:9) == '##ELSEIF ')) THEN
! process-elif-command
         CALL WRITE_COMMENT(ILEV,NLEV,OUTU,LINEA,LLENA)
!   check-for-keyname-match
         CALL KEYMATCH(LINEA,LLENA,MATCH,MATCHID,.FALSE.,ctx)
         IF(LEVID(NLEV) /= MATCHID) THEN
! oops, as bad ID...
            WRITE(MSGU,20) 'ERROR: ID MISMATCH:'
            WRITE(MSGU,20) '    PREVIOUS: ',LEVID(NLEV)
            WRITE(MSGU,20) '    CURRENT:  ',MATCHID
            STOP  'ERROR: ID MISMATCH:'
         ENDIF
! modify the current ILEV value for this ##IF construct
         IF(ILEV(NLEV) == 1) ILEV(NLEV)=-1
         IF(ILEV(NLEV) == 0) THEN
            IF(MATCH) ILEV(NLEV)=1
         ENDIF
!=======================================================================
      ELSE IF(LINEA(1:7) == '##ELSE ') THEN
! process-else-command
!   check-for-keyname-match
         CALL KEYMATCH(LINEA,LLENA,MATCH,MATCHID,.FALSE.,ctx)
         IF(LEVID(NLEV) /= MATCHID) THEN
! oops, as bad ID...
            WRITE(MSGU,20) 'ERROR: ID MISMATCH:'
            WRITE(MSGU,20) '    PREVIOUS: ',LEVID(NLEV)
            WRITE(MSGU,20) '    CURRENT:  ',MATCHID
            STOP  'ERROR: ID MISMATCH:'
         ENDIF
         CALL WRITE_COMMENT(ILEV,NLEV,OUTU,LINEA,LLENA)
! modify the current ILEV value for this ##IF construct
         IF(ILEV(NLEV) == 1) ILEV(NLEV)=-1
         IF(ILEV(NLEV) == 0) ILEV(NLEV)=1
!=======================================================================
      ELSE IF(LINEA(1:8) == '##ENDIF ') THEN
! process-endif-command
!   check-for-keyname-match
         CALL KEYMATCH(LINEA,LLENA,MATCH,MATCHID,.FALSE.,ctx)
         IF(LEVID(NLEV) /= MATCHID) THEN
            WRITE(MSGU,20) 'ERROR: ID MISMATCH:'
            WRITE(MSGU,20) '    PREVIOUS: ',LEVID(NLEV)
            WRITE(MSGU,20) '    CURRENT:  ',MATCHID
            STOP  'ERROR: ID MISMATCH:'
         ENDIF
         CALL WRITE_COMMENT(ILEV,NLEV,OUTU,LINEA,LLENA)
! remove this current level (done with ##IF construct)
         NLEV=NLEV-1
! Oops, too many ##ENDIFs
         IF(NLEV == 0) STOP 'ERROR with ENDIF COUNT'
!=======================================================================
      ELSE IF(LINEA(1:8) == '##ERROR ') THEN
! process-error-command
         IF(ILEV(NLEV) == 1) THEN
            WRITE(OUTU,20) '!',('.',I=1,NLEV-1),LINEA
            WRITE(MSGU,20) ' ERROR: Bad selection in parsing IF block'
            WRITE(MSGU,20) '!',('.',I=1,NLEV-1),LINEA
            STOP  'ERROR: Programmed ERROR encountered'
         ELSE
            WRITE(OUTU,20)
         ENDIF
!=======================================================================
      ELSE IF(LINEA(1:6) == '##SET ') THEN
! process-set-command
         IF(ILEV(NLEV) == 1) THEN
! add a temporary flag to the current level
            WRITE(OUTU,20) '!:::',LINEA(1:LLENA)
            CALL ADDKYS(LINEA,MSGU,ctx%nkeys,ctx%keys,QINCOM)
         ELSE
            WRITE(OUTU,20)
         ENDIF
!=======================================================================
      ELSE IF(LINEA(1:16) == '##KEYWORDS LIST ') THEN
! process-keywords-list-command
!
! LIST mode: Create code that looks something like:
!00000000111111111122222222223333333333444444444455555555556666666666777
!23456789012345678901234567890123456789012345678901234567890123456789012
!     WRITE(OUTU    ,'(1X)')
!     WRITE(OUTU    ,'(1X,A)') 'KEYWORD LIST: Number of keys: xxx.'
!     WRITE(OUTU    ,'(6X,A)')
!    &'  KEY1 KEY2 KEY3 ...                                            '
!     WRITE(OUTU    ,'(6X,A)')
!    &'  KEY12 KEY13 KEY14 ...                                         '
!     WRITE(OUTU    ,'(1X)')
!     WRITE(OUTU    ,'(1X)')
!
         IF(ILEV(NLEV) == 1) THEN
           WRITE(OUTU,20) '!:::',LINEA(1:LLENA)
! first get unit name
           LINEA(1:16)=' '
!           IF(freeform) LINEA(1:18)=' '
           CALL TRIMA(LINEA,LLENA)
           IF(LLENA.LE.0) STOP 'ERROR: LIST command syntax error.'
           JPT=LLENA
           DO IPT=LLENA,2,-1
             IF(LINEA(IPT:IPT) == ' ') JPT=IPT-1
           ENDDO
           IF(JPT > 8) STOP 'ERROR: LIST command file name too long.'
           UNITNAME=LINEA(1:JPT)
! write a blank line
           LINEA="      WRITE(        ,'(1X)')"
           LINEA(13:20)=UNITNAME
           LLENA=LINLEN(LINEA)
           WRITE(OUTU,20) LINEA(1:LLENA)
! write out number of active keys
           LINEA=' '
           LINEA( 1:36)="      WRITE(xxxxxxxx,'(1X,A)') 'KEYW"
           LINEA(37:72)="ORD LIST: Number of keys: xxx.'     "
           LINEA(13:20)=UNITNAME
           WRITE(LINEA(62:65),'(I4)') ctx%nkeys
           LLENA=LINLEN(LINEA)
           WRITE(OUTU,20) LINEA(1:LLENA)
! get key lengths
           DO IPT=1,ctx%nkeys
             LNKEYS(IPT)=12
             CALL TRIMA(ctx%keys(IPT),LNKEYS(IPT))
           ENDDO
! now write out the keys
           IF(freeform)THEN
             LINEA="      WRITE(        ,'(6X,A)')&"
           ELSE
             LINEA="      WRITE(        ,'(6X,A)')"
           ENDIF  
           LINEA(13:20)=UNITNAME
           LLENA=LINLEN(LINEA)
           JPT=8
           LINEB=' '
           LINEB( 1:36)="     &'                             "
           LINEB(37:72)="                                   '"
           LLENB=72
           DO IPT=1,ctx%nkeys
             IF(JPT+LNKEYS(IPT) > 72) THEN
               WRITE(OUTU,20) LINEA(1:LLENA)
               WRITE(OUTU,20) LINEB(1:LLENB)
               JPT=8
               LINEB( 1:36)="     &'                             "
               LINEB(37:72)="                                   '"
             ENDIF
             LINEB(JPT:JPT+LNKEYS(IPT)-1) = ctx%keys(IPT)
             JPT=JPT+LNKEYS(IPT)+1
           ENDDO
           IF(JPT > 8) THEN
             WRITE(OUTU,20) LINEA(1:LLENA)
             WRITE(OUTU,20) LINEB(1:LLENB)
           ENDIF
! write two blank lines
           LINEA="      WRITE(        ,'(1X)')"
           LINEA(13:20)=UNITNAME
           LLENA=LINLEN(LINEA)
           WRITE(OUTU,20) LINEA(1:LLENA)
           WRITE(OUTU,20) '!:::'
           WRITE(OUTU,'(A,I0)') '# ', ctx%lineno + 1
         ENDIF
!=======================================================================
      ELSE IF(LINEA(1:16) == '##KEYWORDS FILL ') THEN
! process-keywords-fill-command
!
! FILL mode: Create code that looks something like:
!
!00000000111111111122222222223333333333444444444455555555556666666666777
!23456789012345678901234567890123456789012345678901234567890123456789012
!     COUNT    = xxxx
!     ARRAY    (   1) = 'KEY1        '
!     ARRAY    (   2) = 'KEY2        '
!
         IF(ILEV(NLEV) == 1) THEN
!
! first get count name
           WRITE(OUTU,20) '!:::',LINEA(1:LLENA)
           LINEA(1:16)=' '
           CALL TRIMA(LINEA,LLENA)
           IF(LLENA.LE.0) STOP 'ERROR: FILL command syntax error.'
           JPT=LLENA
           DO IPT=LLENA,2,-1
             IF(LINEA(IPT:IPT) == ' ') JPT=IPT-1
           ENDDO
           IF(JPT > 8) STOP 'ERROR: FILL command count name too long.'
           UNITNAME=LINEA(1:JPT)
! write the count
           LINEB='                ='
           LINEB(7:16)=UNITNAME
           WRITE(LINEB(19:22),'(I4)') ctx%nkeys
           LLENB=22
           WRITE(OUTU,20) LINEB(1:LLENB)
! now get array name
           LINEA(1:JPT)=' '
           CALL TRIMA(LINEA,LLENA)
           IF(LLENA.LE.0) STOP 'ERROR: FILL command syntax error.'
           JPT=LLENA
           DO IPT=LLENA,2,-1
             IF(LINEA(IPT:IPT) == ' ') JPT=IPT-1
           ENDDO
           IF(JPT > 8) STOP 'ERROR: FILL command count name too long.'
           UNITNAME=LINEA(1:JPT)
! write the array
           DO IPT=1,ctx%nkeys
             LINEA="               (    ) = '            '"
             LINEA(7:14)=UNITNAME
             WRITE(LINEA(17:20),'(I4)') IPT
             LINEA(26:37) = ctx%keys(IPT)
             LLENA=38
             WRITE(OUTU,20) LINEA(1:LLENA)
           ENDDO
           WRITE(OUTU,20) '!:::'
           WRITE(OUTU,'(A,I0)') '# ', ctx%lineno + 1
         ENDIF
!=======================================================================
      ELSE IF(LINEA(1:10) == '##INCLUDE ') THEN
! process-include-command
         IF(ILEV(NLEV) == 1) THEN
            WRITE (MSGU, 75) 'include found, line numbers may be wrong'
 75         FORMAT ('  WARNING:: ', A)
            IF (PUTFCM) THEN
               WRITE(OUTU,20) '!:::',LINEA(1:LLENA)
               LSTRM(ctx%unit)=NLEV
               IF (ctx%unit == STIN) THEN
                  NSTRM=21
               ELSE
                  NSTRM=NSTRM+1
               ENDIF
               QEXP(NSTRM)=.FALSE.
               IF(NSTRM > MXSTRM) STOP 'ERROR: Too many include levels'
               LLENA=120
            ENDIF
            CALL TRIMA(LINEA,LLENA)
! handle machine specific include processing (directory name,...)
! TODO use EXSTK?
            CALL INCLDE(NSTRM,OUTU,LINEA,LLENA,PUTFCM,QVMS,FCMDIR)
         ELSE
            WRITE(OUTU,20)
         ENDIF
!=======================================================================
      ELSE IF(LINEA(1:9) == '##EXPAND ') THEN
! process-expand-command
         I=MAX(INDEX(LINEA,'.WHEN.'),INDEX(LINEA,'.when.'))
         IF(ILEV(NLEV) == 1) THEN
            CALL WRITE_COMMENT(ILEV,NLEV,OUTU,LINEA,LLENA)
! determine if the expand will be active.
            IF(I > 0) THEN
               LINEB=LINEA(I:)
               CALL KEYMATCH(LINEB,LLENB,MATCH,MATCHID,.TRUE.,ctx)
            ELSE
               MATCH=.TRUE.
            ENDIF
         ELSE
            WRITE(OUTU,20)
            MATCH=.FALSE.
         ENDIF
!
         IF(MATCH) THEN
! Process this expand section
            LSTRM(ctx%unit) = NLEV
            child => exstk(IEXPL+1)
            IF (ctx%unit == STIN) THEN
               child%unit = 21
            ELSE
               child%unit = ctx%unit + 1
            ENDIF
            IF (child%unit > MXSTRM) STOP 'ERROR: Too many expand levels'
            QEXP(child%unit) = .TRUE.
!
! get MATCHID for this strucutre
            CALL KEYMATCH(LINEA,LLENA,MATCH,MATCHID,.FALSE.,ctx)
            IF(MATCHID == ' ') STOP '##EXPAND with no match id'
! copy current flags for the new expand level
            IF (IEXPL >= MXEXPL) STOP 'FATAL: TOO MANY EXPAND LEVELS'
            call copy_keys(child, ctx)
            IEXPL=IEXPL+1
! create temporary include file for expanded code
            WRITE(child%fname,445) child%unit
 445        FORMAT('FOR0',I2,'.DAT')
!.##IF APOLLO
!.##ELSE
            OPEN(UNIT=child%unit,FORM='FORMATTED',FILE=child%fname, &
                  ACCESS='SEQUENTIAL',STATUS='UNKNOWN')
!.##ENDIF
! get the control lines for this expand construct
            IPT=1
            DO
               READ(ctx%unit,20,END=490) LINEA
               ctx%lineno = ctx%lineno + 1
               LLENA=LINLEN(LINEA)
               IF(LLENA.LE.0) CYCLE
               child%ctrl_text(IPT) = LINEA
               IPT=IPT+1
               IF(IPT > MAXCLN) STOP 'Too many expand control lines'
               IF(LINEA(1:7) /= '##EXEND') CYCLE
               EXIT
 490           STOP 'End of file found in an expand control section'
            ENDDO
! copy the body of the expand section into a temporary file
            child%offset = ctx%lineno
            DO
               READ(ctx%unit,20,END=495) LINEA
               ctx%lineno = ctx%lineno + 1
               LLENA=LINLEN(LINEA)
               IF(LINEA(1:7) /= '##ENDEX') THEN
                  WRITE(child%unit,20) LINEA(1:LLENA)
                  CYCLE
               ENDIF
! ##ENDEX found - check to see if it is the current one.
               CALL KEYMATCH(LINEA,LLENA,MATCH,LINEB,.FALSE.,child)
               IF(LINEB /= MATCHID) THEN
                  WRITE(child%unit,20) LINEA(1:LLENA)
                  CYCLE
               ENDIF
               EXIT
 495           STOP 'End of file found in an expand section body'
            ENDDO
! match found, save temp file for subsequent read.
            CLOSE(UNIT=child%unit)
            ctx => child
            ctx%lineno = ctx%offset
!--------------------------
! OK, control section copied and temp file created - process first pass.
!     (there must be at least one pass.)
            ctx%ctrl_index = 1
            LINEA = ctx%ctrl_text(1)
            IF(LINEA(1:6) /= '##PASS') &
                  STOP 'Expand section has no first pass'
!           Process this first pass in an expanded section
            GOTO 700
!--------------------------
         ELSE  ! .not. MATCH
! Do not process this expand section (treat as an ##IF)
            CALL KEYMATCH(LINEA,LLENA,MATCH,MATCHID,.FALSE.,ctx)
! add a new level
            NLEV=NLEV+1
            IF(NLEV > MXLEV) STOP 'FATAL: TOO MANY LEVELS'
            LEVID(NLEV)=MATCHID
            ILEV(NLEV)=ILEV(NLEV-1)
            IF(IEXPL.GE.MXEXPL) STOP 'FATAL: TOO MANY EXPAND LEVELS'
            child => exstk(IEXPL+1)
            child = ctx  ! copies all fields
            IEXPL=IEXPL+1
! turn on temporary flags
            IF(ILEV(NLEV) == 1) THEN
               IF(I > 0) LINEA(I:)=' '
               CALL ADDKYS(LINEA,MSGU, &
                     child%nkeys,child%keys,QINCOM)
            ENDIF
! skip the control lines for this unused expand construct
            ctx => child
            DO
               READ(ctx%unit,20,END=559) LINEA
               ctx%lineno = ctx%lineno + 1
               WRITE(OUTU,20)
               LLENA=LINLEN(LINEA)
               IF(LLENA.LE.0) CYCLE
               IF(LINEA(1:7) /= '##EXEND') CYCLE
               EXIT
 559           STOP 'End of file found in an expand control section'
            ENDDO
         ENDIF  ! MATCH
!=======================================================================
      ELSE IF(LINEA(1:8) == '##ENDEX ') THEN
! process-endex-command
!  if we get here, then the expand must be inactive.  Just treat
!  this as if it was an ##endif if the code is active.
         CALL KEYMATCH(LINEA,LLENA,MATCH,MATCHID,.FALSE.,ctx)
         IF(LEVID(NLEV) /= MATCHID) THEN
            WRITE(MSGU,20) 'ERROR: ID MISMATCH:'
            WRITE(MSGU,20) '    PREVIOUS: ',LEVID(NLEV)
            WRITE(MSGU,20) '    CURRENT:  ',MATCHID
            STOP  'ERROR: ID MISMATCH:'
         ENDIF
         CALL WRITE_COMMENT(ILEV,NLEV,OUTU,LINEA,LLENA)
! remove this current level (done with ##EXPAND construct)
         NLEV=NLEV-1
         IEXPL=IEXPL-1
! Oops, too many ##ENDEXs
         IF(NLEV == 0 .OR. IEXPL == 0) STOP 'ERROR with ENDEX COUNT'
         ctx => exstk(IEXPL)
!=======================================================================
      ELSE IF(LINEA(1:6) == '##USE ') THEN
         if(ilev(nlev) == 1) then
            WRITE(OUTU,20) '      use'//LINEA(6:LLENA)
         else
            WRITE(OUTU,20)
         endif
!=======================================================================
      ELSE IF(LINEA(1:10) == '##FORTRAN ') THEN
         WRITE(OUTU,20) '!'//LINEA(1:LLENA)
!
!=======================================================================
      ELSE IF(LINEA(1:8) == '##FLECS ') THEN
         WRITE(OUTU,20) '!'//LINEA(1:LLENA)
         STOP 'ERROR: FLECS is no longer supported.'
!=======================================================================
      ELSE
! pass through as comments unrecognized commands, and issue warning
         WRITE(MSGU,35) LINEA(1:LLENA)
  35     FORMAT(' WARNING::: unrecognized command line: ',A)
         IF(ILEV(NLEV) == 1) WRITE(OUTU,20) '!'//LINEA(1:LLENA)
         STOP 'ERROR: Unrecognized ## construct encountered'
!=======================================================================
      ENDIF
!=======================================================================
      GOTO 100
!=======================================================================
!
! process end of file encountered
 200  CONTINUE
      CLOSE(UNIT=ctx%unit)
! Is this the main file?
      IF (ctx%unit == STIN) GOTO 800

      parent => exstk(IEXPL-1)
!
! check to see if the level is correct.  Do not allow ##IF constructs
! to start and end in different files.
      IF(NLEV > LSTRM(parent%unit)) &
            STOP 'ERROR, end of file found in IF block in expand'
      IF(NLEV < LSTRM(parent%unit)) &
            STOP 'ERROR, too many ENDIFs in expand file'
!-------------------------
      IF(.NOT. QEXP(ctx%unit)) THEN
         GOTO 100
      ENDIF
!=======================================================================
 600  CONTINUE
! This was an expanded section of code.  Process next pass, or
! proces termination of the expanded section.
      LINEA = ctx%ctrl_text(ctx%ctrl_index)
      IF(LINEA(1:6) == '##PASS') THEN
! copy keys for next pass
         parent => exstk(IEXPL-1)
         call copy_keys(ctx, parent)
! process next pass
         GOTO 700
      ELSE IF(LINEA(1:7) == '##EXFIN') THEN
! process section finish
         DO
            ctx%ctrl_index = ctx%ctrl_index + 1
            LINEA = ctx%ctrl_text(ctx%ctrl_index)
            IF(LINEA(1:2) /= '##') THEN
               CALL CHKLINE(LINEA,LLENA,ctx,MSGU, &
                     TABCH,QUPPRC,freeform)
               CALL OUTLINE(LINEA,OUTU,ctx%unit,STIN,MSGU,SINGLE,QINCOM &
                     ,freeform)
               CYCLE
            ELSE
               IF(LINEA(1:7) /= '##EXEND') &
                  STOP '##EXEND not found where expected'
            ENDIF
            EXIT
         ENDDO
      ELSE IF(LINEA(1:7) == '##EXEND') THEN
! terminate expand section
      ELSE IF(LINEA(1:2) == '##') THEN
         STOP 'Unrecognized expand section ## subcommand'
      ELSE
         STOP 'No expand section ## subcommand where expected'
      ENDIF
! done with this expand section...
      IEXPL=IEXPL-1
!-------------------------
! Go back to the previous file.
      ctx => exstk(IEXPL)
      IF (IEXPL == 1) WRITE(OUTU,'(A,I0)') '# ', ctx%lineno + 1
      GOTO 100
!
!=======================================================================
 700  CONTINUE
! Process another expand pass (or do first) in an expanded section
!   (assume key from previous level are already copied)
!
! Determine if the pass will be active
      I=MAX(INDEX(LINEA,'.WHEN.'),INDEX(LINEA,'.when.'))
      IF(I > 0) THEN
         LINEB=LINEA(I:)
         CALL KEYMATCH(LINEB,LLENB,MATCH,MATCHID,.TRUE.,ctx)
         LINEA(I:)=' '
      ELSE
         MATCH=.TRUE.
      ENDIF
!
      IF (MATCH) THEN
! process this pass
         CALL ADDKYS(LINEA,MSGU, &
               ctx%nkeys,ctx%keys,QINCOM)
         DO
            ctx%ctrl_index = ctx%ctrl_index + 1
            LINEA = ctx%ctrl_text(ctx%ctrl_index)
            IF(LINEA(1:2) /= '##') THEN
               CALL CHKLINE(LINEA,LLENA,ctx,MSGU, &
                     TABCH,QUPPRC,freeform)
               CALL OUTLINE(LINEA,OUTU,ctx%unit,STIN,MSGU,SINGLE,QINCOM &
                     ,freeform)
               CYCLE
            ENDIF
            EXIT
         ENDDO
! invoke the tempory file to get the body of the expanded section
         LINEA=''''//ctx%fname//''''
         LLENA=120
         CALL TRIMA(LINEA,LLENA)
         FCMD1='LOCAL'
         CALL INCLDE(ctx%unit,OUTU,LINEA,LLENA,PUTFCM,QVMS,FCMD1)
         ctx%lineno = ctx%offset
         WRITE(OUTU,'(A,I0)') '# ', ctx%offset + 1
         GOTO 100
      ELSE
! ignore this pass
         DO
            ctx%ctrl_index = ctx%ctrl_index + 1
            LINEA = ctx%ctrl_text(ctx%ctrl_index)
            IF(LINEA(1:2) /= '##') CYCLE
            EXIT
         ENDDO
         GOTO 600
      ENDIF
!
!=======================================================================
 800  CONTINUE
! Now we are done processing this file.  Do a level check.
      IF(NLEV /= 1) STOP 'ERROR, end of file found in IF block'
      IF(IEXPL /= 1) STOP 'ERROR, end of file found in EXPAND block'
!
!.##IF MACINTOSH
!.##ENDIF
!
!      STOP ! VO : (1) stop causes a stop message to be printed on some
!                      compilers and the preflx script traps those messages
!                      (export NO_STOP_MESSAGE to environment does not
!                      always work)
!                  (2) driver was called from the main program, we should
!                      return to it
       return
   END SUBROUTINE DRIVER
!=======================================================================
   SUBROUTINE WRITE_COMMENT(ILEV, NLEV, OUTU, LINEA, LLENA)
      INTEGER ILEV(*)
      INTEGER NLEV
      INTEGER OUTU
      CHARACTER(LEN=*) :: LINEA
      INTEGER LLENA

      ! write out auto indented comment
      IF (NLEV < 2) THEN
         WRITE (OUTU, '(20a)')
      ELSE IF (ILEV(NLEV-1) == 1) THEN
         WRITE (OUTU, '(20a)') '!', ('.', I=1, NLEV-1), LINEA(1:LLENA)
      ELSE
         WRITE (OUTU, '(20a)')
      ENDIF
   END SUBROUTINE WRITE_COMMENT
!=======================================================================
   SUBROUTINE OUTLINE(LINEA,OUTU,NSTRM,STIN,MSGU,SINGLE,QINCOM &
           ,freeform)
!
! Write out the current line after some processing and checking
!
!     LINEA - line to write out
!     OUTU  - unit number for current output
!     NSTRM - unit number for current  input
!     STIN  - unit number for standard input
!     MSGU  - unit number for error message (e.g. line too long)
!     SINGLE- logical flag for single precision conversion check
!     QINCOM- logical flag for writing comment liness in include files
!
!.##IFN APOLLO
      IMPLICIT NONE
!.##ENDIF
      logical freeform
      character(len=*) :: LINEA
      INTEGER OUTU,NSTRM,STIN,MSGU
      LOGICAL SINGLE,QINCOM
!
      INTEGER LLENA
      SAVE
!
      IF(LINEA == ' ') THEN
         WRITE(OUTU,20)
         RETURN
      ENDIF
      LLENA=LINLEN(LINEA)
!
      IF(LINEA(1:1) == 'C'.OR.LINEA(1:1) == 'c') THEN
! This is a comment line, write it out if not in an include file
         IF(NSTRM == STIN .OR. QINCOM) WRITE(OUTU,20) LINEA(1:LLENA)
      ELSE IF(ICHAR(LINEA(1:1)) == 12) THEN
! This line starts with a form feed, ignore it
         LINEA(1:1)='!'
         WRITE(OUTU,20)
      ELSE
         IF(SINGLE) CALL SINGLES(LINEA)
         WRITE(OUTU,20) LINEA(1:LLENA)
  20     FORMAT(20A)
      ENDIF
      RETURN
   END SUBROUTINE OUTLINE
!=======================================================================
!=======================================================================
   SUBROUTINE CHKLINE(LINEA,LLENA,ctx,MSGU, &
                      TABCH,QUPPRC,freeform)
!
! Check the current line for !## flags and for ! in general.
!
!.##IFN APOLLO
      IMPLICIT NONE
!.##ENDIF
      logical freeform
      character(len=*) :: LINEA
      INTEGER LLENA,MSGU
      type(expand_context) :: ctx
      LOGICAL TABCH,QUPPRC
!
      INTEGER I,J,LLENB
      LOGICAL SQF,MATCH
      character(len=120) :: MATCHID,LINEB
      character(len=12) :: WRD
      SAVE
!
      LLENA=LINLEN(LINEA)
! see if the line has any nasty tabs in it
      IF(TABCH) THEN
         DO I=1,LLENA
            IF(ICHAR(LINEA(I:I)) == 9) &
                    STOP 'FATAL ERROR: TABS IN THIS FILE!'
         ENDDO
      ENDIF
!
      IF(freeform) then
         if(LLENA > 200) &
           WRITE(MSGU,77) LINEA(1:LLENA)
 77      FORMAT( &
              ' WARNING:: Free Format Line longer than 200 chars:', &
              ' Errors possible.'/,'   "',A,'"')
      else
         if(LLENA > 200) &
              WRITE(MSGU,76) LINEA(1:LLENA)
 76      FORMAT('  WARNING:: Line longer than 200 characters:', &
              ' Errors possible.'/,'   "',A,'"')
      endif
!
      IF(LINEA(1:1) == 'C') GOTO 220
      IF(LINEA(1:1) == 'c') GOTO 220
      IF(LINEA(1:1) == '!') GOTO 220
! See if there are any "{" on this line for substituions of the
! type {V*} => V0
  80  CONTINUE
      DO I=2,LLENA-3
        IF(LINEA(I:I) == '{') THEN
          IF(LINEA(I+2:I+3) == '*}') THEN
!           found a substitution...replace it
            DO J=ctx%NKEYS,1,-1
              WRD=ctx%KEYS(J)
              IF(WRD(1:1) == LINEA(I+1:I+1)) THEN
                IF(WRD(3:) == ' ') THEN
!                 found a match...
                  LINEB(1:I-1)=LINEA(1:I-1)
                  LINEB(I:I+1)=WRD(1:2)
                  LINEB(I+2:)=LINEA(I+4:)
                  LINEA=LINEB
                  LLENA=LLENA-2
                  GOTO 80
                ENDIF
              ENDIF
            ENDDO
!           found no match
            STOP 'ERROR: NO MATCH OF TWO LETTER SUBSTITUTION KEYWORD'
          ENDIF
        ENDIF
      ENDDO
!
! See if there are any "!" on this line
      DO I=1,LLENA
         IF(LINEA(I:I) == '!') THEN
            IF(QUPPRC) THEN
               SQF=.FALSE.
               DO J=1,I
                  IF(LINEA(J:J) == '''') SQF=.NOT.SQF
                  IF(.NOT.SQF) CALL UPPERCASE(LINEA(J:J),1)
               ENDDO
            ENDIF
            GOTO 210
         ENDIF
      ENDDO
!
! No exclamation mark, now check uppercase flag
      IF(QUPPRC) THEN
         SQF=.FALSE.
         DO I=1,LLENA
            IF(LINEA(I:I) == '''') SQF=.NOT.SQF
!           dont uppercase text in quotes.
            IF(.NOT.SQF) CALL UPPERCASE(LINEA(I:I),1)
         ENDDO
      ENDIF
!
      RETURN
 210  CONTINUE
! we found at least one, now lets process the line
      SQF=.FALSE.
      DO I=1,LLENA
         IF(LINEA(I:I) == '''') SQF=.NOT.SQF
         IF(LINEA(I:I) == '!') THEN
           IF(.NOT.SQF) THEN
!            here's a ! comment, remove it
             J=I+2
!            check for inline option parsing
             IF(LINEA(I:J) == '!##') THEN
                LINEB=LINEA(I:)
                LLENB=LLENA-I+1
                DO J=2,LLENB
                   IF(LINEB(J:J) == '!') LINEB(J:)=' '
                ENDDO
                LINEB(3:3)=' '
                CALL KEYMATCH(LINEB,LLENB,MATCH,MATCHID,.FALSE.,ctx)
                IF(MATCH) THEN
                   LINEA(I:I)=' '
                   DO J=I+1,LLENA
                      IF(LINEA(J:J) == '!') GOTO 215
                      LINEA(J:J)=' '
                   ENDDO
                ELSE
                   LINEA(1:2)='!:'
                   GOTO 220
                ENDIF
              else if (trim(LINEA(1:1)) /= '#') then
!               it's just a simple comment - remove, but leave cpp lines alone
                LINEA(I:I)=' '
                DO J=I+1,LLENA
                   IF(LINEA(J:J) == '!') GOTO 215
                   LINEA(J:J)=' '
                ENDDO
             ENDIF
           ENDIF
         ENDIF
 215     CONTINUE
      ENDDO
 220  CONTINUE
      LLENA=LINLEN(LINEA)
      RETURN
   END SUBROUTINE CHKLINE
!=======================================================================
!=======================================================================
   SUBROUTINE UPPERCASE(ST,STLEN)
!
! This routine converts a string to upper case.
! This routine is ASCII code dependant.
!
      INTEGER STLEN
      character(len=*) :: ST
!
      INTEGER ILITA,ILITZ,ICHUCA,IBIGA,ICHR,I
      SAVE
!
      ICHUCA=ICHAR('A')
      ILITA=ICHAR('a')
      ILITZ=ICHAR('z')
      IBIGA=ICHUCA-ILITA
!
      DO I=1,STLEN
        ICHR=ICHAR(ST(I:I))
        IF(ICHR.GE.ILITA.AND.ICHR.LE.ILITZ) ST(I:I)=CHAR(ICHR+IBIGA)
      ENDDO
!
      RETURN
   END SUBROUTINE UPPERCASE
!=======================================================================
!=======================================================================
   SUBROUTINE KEYMATCH(LINE,LLEN,MATCH,MATCHID,QALL,ctx)
!
!  Determine if flags on the line match the current key set.
! to check-for-keyname-match
!
!.##IFN APOLLO
      IMPLICIT NONE
!.##ENDIF
!
      INTEGER LLEN
      character(len=*) :: LINE
      LOGICAL MATCH
      character(len=*) :: MATCHID
      LOGICAL QALL
      type(expand_context),intent(in) :: ctx

      LOGICAL LBLANK,LPAREN,QM
      INTEGER IPT,JPT,I
      SAVE
!
      LLEN=LEN(LINE)
      CALL TRIMA(LINE,LLEN)
      MATCH=QALL
      MATCHID=' '
      LBLANK=.FALSE.
      LPAREN=.FALSE.
      DO IPT=1,LLEN
         IF(LINE(IPT:IPT) == ' ') THEN
            LBLANK=.NOT.LPAREN
         ELSE IF(LINE(IPT:IPT) == '(') THEN
            LPAREN=.TRUE.
            LBLANK=.FALSE.
            JPT=IPT
         ELSE IF(LINE(IPT:IPT) == ')') THEN
            IF(.NOT.LPAREN) &
                  STOP 'ERROR: NO MATCH OF ID PARENTHESES'
            LPAREN=.FALSE.
            MATCHID=LINE(JPT:IPT)
            LBLANK=.FALSE.
         ELSE
            IF(LBLANK) THEN
               JPT=IPT+1
  40           CONTINUE
               IF(LINE(JPT:JPT) /= ' ') THEN
                  JPT=JPT+1
                  IF(JPT.LE.LLEN) GOTO 40
               ENDIF
               IF(LINE(IPT:IPT+4) == '.NOT.' .OR. &
                  LINE(IPT:IPT+4) == '.not.') THEN
                  QM=.FALSE.
                  DO I=1,ctx%nkeys
                     IF (ctx%keys(I) == LINE(IPT+5:JPT)) QM=.TRUE.
                  ENDDO
                  IF(.NOT.QALL .AND. .NOT.QM) MATCH=.TRUE.
                  IF(QALL .AND. QM) MATCH=.FALSE.
               ELSE
                  QM=.FALSE.
                  DO I=1,ctx%nkeys
                     IF (ctx%keys(I) == LINE(IPT:JPT)) QM=.TRUE.
                  ENDDO
                  IF(.NOT.QALL .AND. QM) MATCH=.TRUE.
                  IF(QALL .AND. .NOT.QM) MATCH=.FALSE.
               ENDIF
            ENDIF
            LBLANK=.FALSE.
         ENDIF
      ENDDO
      RETURN
   END SUBROUTINE KEYMATCH
!=======================================================================
   SUBROUTINE TRIMA(LINE,LLEN)
!
! trims a character string
!
!.##IFN APOLLO
      IMPLICIT NONE
!.##ENDIF
      INTEGER LLEN,IPT,JPT
      character(len=*) :: LINE
      character(len=120) :: LTEMP
      SAVE
!
      DO IPT=1,LLEN
         IF(LINE(IPT:IPT) /= ' ') GOTO 100
      ENDDO
      LLEN=0
      RETURN
 100  CONTINUE
!
      DO JPT=LLEN,1,-1
         IF(LINE(JPT:JPT) /= ' ') GOTO 200
      ENDDO
 200  CONTINUE
      LTEMP=LINE(IPT:JPT)
      LINE=LTEMP
      LLEN=JPT-IPT+1
      RETURN
   END SUBROUTINE TRIMA
!=======================================================================
   INTEGER FUNCTION LINLEN(LINE)
!
! trims a character string
!
!.##IFN APOLLO
      IMPLICIT NONE
!.##ENDIF
      character(len=*) :: LINE
!
      INTEGER JPT,LLEN
      SAVE
!
      LLEN=LEN(LINE)
      DO JPT=LLEN,1,-1
         IF(LINE(JPT:JPT) /= ' ') GOTO 200
      ENDDO
      JPT=0
 200  CONTINUE
      LINLEN=JPT
      RETURN
   END FUNCTION LINLEN
!=======================================================================
   SUBROUTINE OPENF(INU,OUTU,MSGU)
!
!  MACHINE SPECIFIC OPENING INITIAL FILES
!
!.##IFN APOLLO
      IMPLICIT NONE
!.##ENDIF
      INTEGER INU,OUTU,MSGU
      SAVE
! begin
!
! open FLECS input file on unit 1
!.##IF APOLLO
!.##ELSE
      OPEN(UNIT=MSGU,FILE='prefx.msg',FORM='FORMATTED', &
           ACCESS='SEQUENTIAL',STATUS='UNKNOWN')
!.##ENDIF
      RETURN
   END SUBROUTINE OPENF
!=======================================================================
   SUBROUTINE INCLDE(INU,OUTU,LINE,LLEN,PUTFCM,QVMS,FCMDIR)
!
! MACHINE SPECIFIC OPEN INCLUDE FILE
!
!.##IFN APOLLO
      IMPLICIT NONE
!.##ENDIF
      INTEGER INU,OUTU,LLEN
      character(len=*) :: LINE
      LOGICAL PUTFCM,QVMS
      character(len=60) :: FCMDIR
!
      INTEGER LNLINE,IPT,JPT,KPT,LPT
      character(len=60) :: NLINE
      character(len=1) :: DIRCHR
      SAVE
! begin
!
      DO IPT=1,LLEN
         IF(LINE(IPT:IPT) == '''') GOTO 100
      ENDDO
      IF(IPT > LLEN) STOP 'ERROR: cannot find include file.'
 100  CONTINUE
      DO JPT=LLEN,IPT,-1
         IF(LINE(JPT:JPT) == '''') GOTO 200
      ENDDO
 200  CONTINUE
      IF(IPT == JPT) STOP 'ERROR: cannot find include file.'
      IPT=IPT+1
      JPT=JPT-1
!
      NLINE=LINE(IPT:JPT)
      LNLINE=JPT-IPT+1
!
!.##IF MACINTOSH
!.##ELSE
      IF (QVMS) THEN
         DIRCHR=':'
      ELSE
         DIRCHR='/'
      ENDIF
!.##ENDIF
!
      KPT=JPT
 50   CONTINUE
      KPT=KPT-1
      IF(KPT == IPT) GOTO 60
      IF(LINE(KPT:KPT) /= '/'.AND.LINE(KPT:KPT) /= ':') GOTO 50
      KPT=KPT+1
 60   CONTINUE
!
! process name translation
!   (default is given name)
!
! use given name
      IF(FCMDIR == ' ' .OR. FCMDIR == 'CURRENT') THEN
         IF (QVMS) THEN
            DO KPT=IPT,JPT
               IF(LINE(KPT:KPT) == '/') LINE(KPT:KPT)=':'
            ENDDO
         ENDIF
! use local directory
      ELSE IF(FCMDIR == 'LOCAL') THEN
         NLINE=LINE(KPT:JPT)
         LNLINE=JPT-KPT+1
! use provided directory
      ELSE
         LPT=0
 70      CONTINUE
         LPT=LPT+1
         IF(FCMDIR(LPT:LPT) /= ' '.AND.LPT < 60) GOTO 70
         LPT=LPT-1
         NLINE = FCMDIR(1:LPT) // DIRCHR // LINE(KPT:JPT)
         LNLINE=LPT+1+JPT-KPT+1
      ENDIF
!
! open include file
!.##IF APOLLO
!.##ELSE
      IF (PUTFCM) THEN
!     open the include file
         OPEN(UNIT=INU,FORM='FORMATTED',FILE=NLINE(1:LNLINE), &
              ACCESS='SEQUENTIAL',STATUS='OLD')
      ELSE
            WRITE (OUTU,'(A)') '      INCLUDE ''' // NLINE(1:LNLINE) &
                               // ''''
      ENDIF
!
!.##ENDIF
!
      RETURN
   END SUBROUTINE INCLDE
!=======================================================================
   SUBROUTINE SETKYS(NKEYS,KEYS,FCMDIR,MSGU)
!
! machine specific get of keynames
!
!.##IFN APOLLO
      IMPLICIT NONE
!.##ENDIF
      INTEGER NKEYS
      CHARACTER(len=12) :: KEYS(MXKEYS)
      character(len=60) :: FCMDIR
      INTEGER MSGU
!
      INTEGER LNKEYS(MXKEYS)
      character(len=120) :: LINE
      INTEGER I,J,PFDU
      PARAMETER(PFDU=4)
      SAVE
!
! no more default keys... All explicit.
      NKEYS=0
      FCMDIR=' '
!
! open the local preflx.dat file (use defaults if it doesnt exist)
!.##IF APOLLO
!.##ELSE
      OPEN(UNIT=PFDU,FORM='FORMATTED',FILE='preflx.dat', &
           ACCESS='SEQUENTIAL',STATUS='OLD',ERR=180)
      GOTO 190
 180  CONTINUE
      OPEN(UNIT=PFDU,FORM='FORMATTED',FILE='pref.dat', &
           ACCESS='SEQUENTIAL',STATUS='OLD',ERR=200)
 190  CONTINUE
!.##ENDIF
!
      NKEYS=1
  50  CONTINUE
         READ(PFDU,20,END=100) LINE
  20     FORMAT(A)
         IF(LINE(1:7) == 'FCMDIR=') THEN
            FCMDIR=LINE(8:)
            GOTO 50
         ENDIF
         KEYS(NKEYS)=LINE(1:12)
         IF(KEYS(NKEYS) == 'END') GOTO 100
         IF(KEYS(NKEYS) /= ' ') NKEYS=NKEYS+1
         IF(NKEYS > MXKEYS) STOP 'FATAL: TOO MANY KEYS'
      GOTO 50
 100  CONTINUE
      NKEYS=NKEYS-1
!
 200  CONTINUE
!
      DO I=1,NKEYS
         DO J=1,I-1
            IF(KEYS(I) == KEYS(J)) THEN
               WRITE(MSGU,21) KEYS(I)
  21           FORMAT(' Duplicate key found: ',A)
               KEYS(I)='...'
            ENDIF
         ENDDO
      ENDDO
!
      J=0
      DO I=1,NKEYS
         IF(J > 0) KEYS(I-J)=KEYS(I)
         IF(KEYS(I) == '...') J=J+1
      ENDDO
      NKEYS=NKEYS-J
      DO I=1,NKEYS
         LNKEYS(I)=12
         CALL TRIMA(KEYS(I),LNKEYS(I))
      ENDDO
!
      IF(NKEYS > 0) THEN
         WRITE(MSGU,22) NKEYS,(KEYS(I)(1:LNKEYS(I)),I=1,NKEYS)
  22     FORMAT(I4,' Conditional keys: ',12(A,1X)/, &
              ('     Conditional keys: ',12(A,1X)))
      ELSE
         WRITE(MSGU,23)
  23     FORMAT(' No condition keys used.')
      ENDIF
!
      RETURN
   END SUBROUTINE SETKYS
!
!=======================================================================
!=======================================================================
   SUBROUTINE ADDKYS(LINE,MSGU,NKEYS,KEYS,QINCOM)
!
! add or remove keys from the current list
!
!.##IFN APOLLO
      IMPLICIT NONE
!.##ENDIF
      character(len=*) ::LINE
      INTEGER MSGU,NKEYS
      CHARACTER(len=12) :: KEYS(MXKEYS)
      LOGICAL QINCOM
!
      INTEGER I,J,IPT,JPT,LLEN
!
      INTEGER LNKEYS(MXKEYS)
      LOGICAL LBLANK,LPAREN,QM
      SAVE
!
      LLEN=LEN(LINE)
      CALL TRIMA(LINE,LLEN)
      LBLANK=.FALSE.
      LPAREN=.FALSE.
      DO IPT=1,LLEN
         IF(LINE(IPT:IPT) == ' ') THEN
            LBLANK=.NOT.LPAREN
         ELSE IF(LINE(IPT:IPT) == '(') THEN
            LPAREN=.TRUE.
            LBLANK=.FALSE.
         ELSE IF(LINE(IPT:IPT) == ')') THEN
            IF(.NOT.LPAREN) &
                  STOP 'ERROR: NO MATCH OF ID PARENTHESES'
            LPAREN=.FALSE.
            LBLANK=.FALSE.
         ELSE
            IF(LBLANK) THEN
               JPT=IPT+1
  40           CONTINUE
               IF(LINE(JPT:JPT) /= ' ') THEN
                  JPT=JPT+1
                  IF(JPT.LE.LLEN) GOTO 40
               ENDIF
               IF(LINE(IPT:IPT+4) == '.NOT.' .OR. &
                  LINE(IPT:IPT+4) == '.not.') THEN
                  DO I=1,NKEYS
                     IF(KEYS(I) == LINE(IPT+5:JPT)) KEYS(I)='...'
                  ENDDO
                  IF(LINE(IPT+5:JPT) == 'INCOMMENTS') QINCOM=.FALSE.
               ELSE
                  QM=.FALSE.
                  DO I=1,NKEYS
                     IF(KEYS(I) == LINE(IPT:JPT)) QM=.TRUE.
                  ENDDO
                  IF(.NOT.QM) THEN
                     IF(LINE(IPT:JPT) /= ' ') NKEYS=NKEYS+1
                     IF(NKEYS > MXKEYS) STOP 'FATAL: TOO MANY KEYS'
                     KEYS(NKEYS)=LINE(IPT:JPT)
                  ENDIF
                  IF(LINE(IPT:JPT) == 'INCOMMENTS') QINCOM=.TRUE.
               ENDIF
            ENDIF
            LBLANK=.FALSE.
         ENDIF
      ENDDO
!
! garbage collect old keys.
      J=0
      DO I=1,NKEYS
         IF(J > 0) KEYS(I-J)=KEYS(I)
         IF(KEYS(I) == '...') J=J+1
      ENDDO
      NKEYS=NKEYS-J
! trim all keys to set lengths
      DO I=1,NKEYS
         LNKEYS(I)=12
         CALL TRIMA(KEYS(I),LNKEYS(I))
      ENDDO
!
      IF(NKEYS > 0) THEN
         WRITE(MSGU,22) NKEYS,(KEYS(I)(1:LNKEYS(I)),I=1,NKEYS)
  22     FORMAT(I4,' Conditional keys: ',12(A,1X)/, &
              ('     Conditional keys: ',12(A,1X)))
      ELSE
         WRITE(MSGU,23)
  23     FORMAT(' No condition keys used.')
      ENDIF
      RETURN
   END SUBROUTINE ADDKYS

   ! Copies the keyword list from src to dest.
   subroutine copy_keys(dest, src)
      type(expand_context),intent(inout) :: dest
      type(expand_context),intent(in) :: src
      integer :: nkeys
      nkeys = src%nkeys
      dest%nkeys = nkeys
      dest%keys(1:nkeys) = src%keys(1:nkeys)
      return
   end subroutine copy_keys
!
!=======================================================================
!=======================================================================
   SUBROUTINE SINGLES(LINE)
!-----------------------------------------------------------------------
!
! Replaces the following 32-bit machine specific features and
! converts everything into standard FORTRAN-77 single precision code
!
! 32-bit machine (VAX, Convex etc.) to 64-bit computers (CRAY)
!
! Double precision conversion:
! ====================================================================
! REAL*8            = REAL            real*8            = real
! COMPLEX*16        = COMPLEX         complex*16        = complex
! INTEGER*2         = INTEGER         integer*2         = integer
! *2(               = (
! *8(               = (
! CDABS(            = ABS(            cdabs(            = abs(
! DABS(             = ABS(            dabs(             = abs(
! CDSIN(            = SIN(            cdsin(            = sin(
! DSIN(             = SIN(            dsin(             = sin(
! CDCOS(            = COS(            cdcos(            = cos(
! DCOS(             = COS(            dcos(             = cos(
! CDTAN(            = TAN(            cdtan(            = tan(
! DTAN(             = TAN(            dtan(             = tan(
! DASIN(            = ASIN(           dasin(            = asin(
! DACOS(            = ACOS(           dacos(            = acos(
! DATAN(            = ATAN(           datan(            = atan(
! DMIN1(            = MIN(            dmin1(            = min(
! DMAX1(            = MAX(            dmax1(            = max(
! SNGL(             = (               sngl(             = (
! DBLE(             = REAL(           dble(             = real(
! DCMPLX(           = CMPLX(          dcmplx(           = cmplx(
! DCONJG(           = CONJG(          dconjg(           = conjg(
! DIMAG(            = AIMAG(          dimag(            = aimag(
! DLOG(             = LOG(            dlog(             = log(
! DMOD(             = AMOD(           dmod(             = amod(
! DSIGN(            = SIGN(           dsign(            = sign(
! DSQRT(            = SQRT(           dsqrt(            = sqrt(
!
! Youngdo Won 08-09-90
! ====================
!
!.##IFN APOLLO
      IMPLICIT NONE
!.##ENDIF
      character(len=120) :: LINE
!
      INTEGER I
      LOGICAL DONE
      SAVE
!
! DO NOT REPLACE DOUBLE PRECISION  - BRB
!C      I=INDEX(LINE,'DOUBLE PRECISION')
!C      IF (I > 0) LINE(I:I+15)='REAL            '
!C      I=INDEX(LINE,'double precision')
!C      IF (I > 0) LINE(I:I+15)='real            '
!
      I=INDEX(LINE,'REAL*8')
      IF (I > 0) LINE(I:I+5)='REAL  '
      I=INDEX(LINE,'real*8')
      IF (I > 0) LINE(I:I+5)='real  '
!
      I=INDEX(LINE,'COMPLEX*16')
      IF (I > 0) LINE(I:I+9)='COMPLEX   '
      I=INDEX(LINE,'complex*16')
      IF (I > 0) LINE(I:I+9)='complex   '
!
      I=INDEX(LINE,'INTEGER*2')
      IF (I > 0) LINE(I:I+8)='INTEGER  '
      I=INDEX(LINE,'integer*2')
      IF (I > 0) LINE(I:I+8)='integer  '
!
      I=INDEX(LINE,'*2(')
      IF (I > 0) LINE(I:I+2)='(  '
!
      I=INDEX(LINE,'*8(')
      IF (I > 0) LINE(I:I+2)='(  '
!
   20 CONTINUE
      DONE=.TRUE.
!
      I=INDEX(LINE,'CDABS(')
      IF (I > 0)  THEN
      LINE(I:I+5)='  ABS('
      DONE=.FALSE.
      ENDIF
      I=INDEX(LINE,'cdabs(')
      IF (I > 0)  THEN
      LINE(I:I+5)='  abs('
      DONE=.FALSE.
      ENDIF
!
      I=INDEX(LINE,'DABS(')
      IF (I > 0)  THEN
      LINE(I:I+4)=' ABS('
      DONE=.FALSE.
      ENDIF
      I=INDEX(LINE,'dabs(')
      IF (I > 0)  THEN
      LINE(I:I+4)=' abs('
      DONE=.FALSE.
      ENDIF
!
      I=INDEX(LINE,'CDSIN(')
      IF (I > 0)  THEN
      LINE(I:I+5)='  SIN('
      DONE=.FALSE.
      ENDIF
      I=INDEX(LINE,'cdsin(')
      IF (I > 0)  THEN
      LINE(I:I+5)='  sin('
      DONE=.FALSE.
      ENDIF
!
      I=INDEX(LINE,'DSIN(')
      IF (I > 0)  THEN
      LINE(I:I+4)=' SIN('
      DONE=.FALSE.
      ENDIF
      I=INDEX(LINE,'dsin(')
      IF (I > 0)  THEN
      LINE(I:I+4)=' sin('
      DONE=.FALSE.
      ENDIF
!
      I=INDEX(LINE,'CDCOS(')
      IF (I > 0)  THEN
      LINE(I:I+5)='  COS('
      DONE=.FALSE.
      ENDIF
      I=INDEX(LINE,'cdcos(')
      IF (I > 0)  THEN
      LINE(I:I+5)='  cos('
      DONE=.FALSE.
      ENDIF
!
      I=INDEX(LINE,'DCOS(')
      IF (I > 0)  THEN
      LINE(I:I+4)=' COS('
      DONE=.FALSE.
      ENDIF
      I=INDEX(LINE,'dcos(')
      IF (I > 0)  THEN
      LINE(I:I+4)=' cos('
      DONE=.FALSE.
      ENDIF
!
      I=INDEX(LINE,'CDTAN(')
      IF (I > 0)  THEN
      LINE(I:I+5)=' TAN('
      DONE=.FALSE.
      ENDIF
      I=INDEX(LINE,'cdtan(')
      IF (I > 0)  THEN
      LINE(I:I+5)=' tan('
      DONE=.FALSE.
      ENDIF
!
      I=INDEX(LINE,'DTAN(')
      IF (I > 0)  THEN
      LINE(I:I+4)=' TAN('
      DONE=.FALSE.
      ENDIF
      I=INDEX(LINE,'dtan(')
      IF (I > 0)  THEN
      LINE(I:I+4)=' tan('
      DONE=.FALSE.
      ENDIF
!
      I=INDEX(LINE,'DASIN(')
      IF (I > 0)  THEN
      LINE(I:I+5)=' ASIN('
      DONE=.FALSE.
      ENDIF
      I=INDEX(LINE,'dasin(')
      IF (I > 0)  THEN
      LINE(I:I+5)=' asin('
      DONE=.FALSE.
      ENDIF
!
      I=INDEX(LINE,'DACOS(')
      IF (I > 0)  THEN
      LINE(I:I+5)=' ACOS('
      DONE=.FALSE.
      ENDIF
      I=INDEX(LINE,'dacos(')
      IF (I > 0)  THEN
      LINE(I:I+5)=' acos('
      DONE=.FALSE.
      ENDIF
!
      I=INDEX(LINE,'DATAN(')
      IF (I > 0)  THEN
      LINE(I:I+5)=' ATAN('
      DONE=.FALSE.
      ENDIF
      I=INDEX(LINE,'datan(')
      IF (I > 0)  THEN
      LINE(I:I+5)=' atan('
      DONE=.FALSE.
      ENDIF
!
      I=INDEX(LINE,'DMIN1(')
      IF (I > 0)  THEN
      LINE(I:I+5)='  MIN('
      DONE=.FALSE.
      ENDIF
      I=INDEX(LINE,'dmin1(')
      IF (I > 0)  THEN
      LINE(I:I+5)='  min('
      DONE=.FALSE.
      ENDIF
!
      I=INDEX(LINE,'DMAX1(')
      IF (I > 0)  THEN
      LINE(I:I+5)='  MAX('
      DONE=.FALSE.
      ENDIF
      I=INDEX(LINE,'dmax1(')
      IF (I > 0)  THEN
      LINE(I:I+5)='  max('
      DONE=.FALSE.
      ENDIF
!
      I=INDEX(LINE,'SNGL(')
      IF (I > 0)  THEN
      LINE(I:I+4)='    ('
      DONE=.FALSE.
      ENDIF
      I=INDEX(LINE,'sngl(')
      IF (I > 0)  THEN
      LINE(I:I+4)='    ('
      DONE=.FALSE.
      ENDIF
!
! Conver DBLE to REAL - BRB
      I=INDEX(LINE,'DBLE(')
      IF (I > 0)  THEN
      LINE(I:I+4)='REAL('
      DONE=.FALSE.
      ENDIF
      I=INDEX(LINE,'dble(')
      IF (I > 0)  THEN
      LINE(I:I+4)='real('
      DONE=.FALSE.
      ENDIF
!
      I=INDEX(LINE,'DCMPLX(')
      IF (I > 0)  THEN
      LINE(I:I+6)=' CMPLX('
      DONE=.FALSE.
      ENDIF
      I=INDEX(LINE,'dcmplx(')
      IF (I > 0)  THEN
      LINE(I:I+6)=' cmplx('
      DONE=.FALSE.
      ENDIF
!
      I=INDEX(LINE,'DCONJG(')
      IF (I > 0)  THEN
      LINE(I:I+6)=' CONJG('
      DONE=.FALSE.
      ENDIF
      I=INDEX(LINE,'dconjg(')
      IF (I > 0)  THEN
      LINE(I:I+6)=' conjg('
      DONE=.FALSE.
      ENDIF
!
! There should be no DFLOATs (not part of F77) - BRB
!      I=INDEX(LINE,'DFLOAT(')
!      IF (I > 0)  THEN
!      LINE(I:I+6)=' FLOAT('
!      DONE=.FALSE.
!      ENDIF
!      I=INDEX(LINE,'dfloat(')
!      IF (I > 0)  THEN
!      LINE(I:I+6)=' float('
!      DONE=.FALSE.
!      ENDIF
!
      I=INDEX(LINE,'DIMAG(')
      IF (I > 0)  THEN
      LINE(I:I+4)='AIMAG('
      DONE=.FALSE.
      ENDIF
      I=INDEX(LINE,'dimag(')
      IF (I > 0)  THEN
      LINE(I:I+4)='aimag('
      DONE=.FALSE.
      ENDIF
!
      I=INDEX(LINE,'DLOG(')
      IF (I > 0)  THEN
      LINE(I:I+4)=' LOG('
      DONE=.FALSE.
      ENDIF
      I=INDEX(LINE,'dlog(')
      IF (I > 0)  THEN
      LINE(I:I+4)=' log('
      DONE=.FALSE.
      ENDIF
!
      I=INDEX(LINE,'DMOD(')
      IF (I > 0)  THEN
      LINE(I:I+4)='AMOD('
      DONE=.FALSE.
      ENDIF
      I=INDEX(LINE,'dmod(')
      IF (I > 0)  THEN
      LINE(I:I+4)='amod('
      DONE=.FALSE.
      ENDIF
!
! There should be no DREALs (not part of F77) - BRB
!      I=INDEX(LINE,'DREAL(')
!      IF (I > 0)  THEN
!      LINE(I:I+5)=' REAL('
!      DONE=.FALSE.
!      ENDIF
!      I=INDEX(LINE,'dreal(')
!      IF (I > 0)  THEN
!      LINE(I:I+5)=' real('
!      DONE=.FALSE.
!      ENDIF
!
      I=INDEX(LINE,'DSIGN(')
      IF (I > 0)  THEN
      LINE(I:I+5)=' SIGN('
      DONE=.FALSE.
      ENDIF
      I=INDEX(LINE,'dsign(')
      IF (I > 0)  THEN
      LINE(I:I+5)=' sign('
      DONE=.FALSE.
      ENDIF
!
      I=INDEX(LINE,'DSQRT(')
      IF (I > 0)  THEN
      LINE(I:I+5)=' SQRT('
      DONE=.FALSE.
      ENDIF
      I=INDEX(LINE,'dsqrt(')
      IF (I > 0)  THEN
      LINE(I:I+5)=' sqrt('
      DONE=.FALSE.
      ENDIF
!
!ln...Add the following 07-Jan-2005 for SINGLE versions
      I=INDEX(LINE,'DEXP(')
      IF (I > 0)  THEN
      LINE(I:I+4)=' EXP('
      DONE=.FALSE.
      ENDIF
      I=INDEX(LINE,'dexp(')
      IF (I > 0)  THEN
      LINE(I:I+4)=' exp('
      DONE=.FALSE.
      ENDIF
!
      I=INDEX(LINE,'IDINT(')
      IF (I > 0)  THEN
      LINE(I:I+5)='  INT('
      DONE=.FALSE.
      ENDIF
      I=INDEX(LINE,'idint(')
      IF (I > 0)  THEN
      LINE(I:I+5)='  int('
      DONE=.FALSE.
      ENDIF

!
      I=INDEX(LINE,'DINT(')
      IF (I > 0)  THEN
      LINE(I:I+4)=' INT('
      DONE=.FALSE.
      ENDIF
      I=INDEX(LINE,'dint(')
      IF (I > 0)  THEN
      LINE(I:I+4)=' int('
      DONE=.FALSE.
      ENDIF
!ln...05-Jan-2005

      IF (.NOT.(DONE)) GOTO 20
!
! convert <CR> into C
!      IF (ICHAR(LINE(1:1)) == 12) LINE(1:1)='!'
!
! comment out 'IMPLICIT NONE'
!      I=INDEX(LINE,'IMPLICIT NONE')
!      IF (I > 0) LINE(1:1)='!'
!
      RETURN
   END SUBROUTINE SINGLES

END PROGRAM PREFLX
