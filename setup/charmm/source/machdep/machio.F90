module machio
  use chm_kinds
  implicit none

  ! holds machine dependent information about FORTRAN units 1...99
  ! and their associated files

  ! list for free FORTRAN units
  ! see routine FREEUN

  ! JMS 10/2011.  GAMUS requires opening/closing more files, so we need more units.
#if KEY_GAMUS==1
  INTEGER,PARAMETER :: MAXUN=999
#else /**/
  INTEGER,PARAMETER :: MAXUN=99
#endif 
  INTEGER :: IFREEU(MAXUN)

  CHARACTER(len=60) COSPDN(MAXUN)
  CHARACTER(len=8)  COSDN(MAXUN)


contains
subroutine close_open(fname)
! check if file named fname is open;if so, close it 
use stream
character(len=*) fname
!local
integer unit
logical qopen
!
  ! BIOVIA Code Start : Enhancement for QM/MM
  if (fname(1:9) .eq. '//./pipe/' .or. fname(1:9) .eq. '//./PIPE/') then
    return
  endif
  ! BIOVIA Code End
  inquire(file=fname,opened=qopen,number=unit)
  if(qopen)then 
    IF(WRNLEV >= 2 .and. prnlev > 5) WRITE(OUTU,'(A,/,2A)') &
          ' CLOSE_OPEN> ***** WARNING ***** another unit is already ', &
          '         assigned to the file -', &
          ' it will be disconnected first.'
    CALL VCLOSE(UNIT,'KEEP',QOPEN)
  endif
end subroutine close_open
!end module machio

!CHARMM Element source/machdep/machio.src 1.1
!
SUBROUTINE VOPEN(UNIT,FILEX,FORM,ACCESS,ERR,DIRRLEN,CONV)
  !----------------------------------------------------------------------
  !     Opens a file.
  !
  !      ++++++++++++++++++++++++++++++++
  !      machine dependent version
  !      ++++++++++++++++++++++++++++++++
  !
  !      Axel Brunger, 30-APR-85
  !      =======================
  !
  use chm_kinds
!  use machio
  use stream
  use string
  use exfunc
  use parallel

  implicit none
#if KEY_PARALLEL==1
  CHARACTER(len=8) fnodnum
#endif 
  INTEGER, INTENT(IN) :: DIRRLEN
  INTEGER UNIT
  CHARACTER(len=*) FILEX, FORM, ACCESS
  CHARACTER(len=*), OPTIONAL :: CONV
  LOGICAL ERR
  !      local
  INTEGER I, FLEN, RECLEN
  LOGICAL EXIST,NOQUOTES
  !
! BIOVIA Code Start
#if KEY_UNIX==1 && KEY_LICENSE==0
! BIOVIA Code End
  INTEGER, PARAMETER :: MXFILE=128
#else /**/
  INTEGER, PARAMETER :: MXFILE=256
#endif 
  CHARACTER(len=MXFILE) FILE
  CHARACTER(len=10) CRCONT
  LOGICAL QCONV
  integer*4 firstreclen
  !
  !     begin
  QCONV=PRESENT(CONV)
  ERR=.FALSE.
  !
  !
  !     Don't open files on a slave process.
#if KEY_PARALLEL==1
#if KEY_PRLLOUT==1
  if(mynod.ne.0)then
     write(fnodnum,'("cout",i4.4)')mynod
     write(outu,*) "Node ",mynod," opening ",fnodnum
     open (unit=6, file=fnodnum,status="NEW")
  endif
#endif 
#endif 
  !---- Especially for parallel slaves, return if IO is turned off
  IF(IOLEV.LT.0) RETURN
  !
  !---------- $$INITIAL$$ Initialize stdio and bookkeeping, then return----
  IF(FILEX.EQ.'$$INITIAL$$') THEN
     !
     !     NOTE: for -input, -output arguments this is not needed
     !           However no action for now!
     !           std input/output already opened in argumt()!
     !

     !        special section for initialization
#if KEY_UNIX==1
     !     no need to open standard i/o files
     ! Unless.....
#if KEY_FILEINPUT==1
     open (unit=5, file="charmm.inp",status="OLD")
#endif 
#if KEY_PARALLEL==1
#if KEY_PRLLOUT==1
     if(mynod.ne.0)then
        write(fnodnum,'("cout",i4.4)')mynod
        write(outu,*) "Node ",mynod," opening ",fnodnum
        open (unit=6, file=fnodnum,status="NEW")
     endif
#endif 
#endif 
#if KEY_FILEOUTPUT==1
     open (unit=6, file="charmm.out",status="NEW")
#endif 
#else /**/
#error  'Unrecognized machine type in VOPEN(1)'
#endif 
     !
     !     Initialize list of free FORTRAN units.
     !     It is assumed that system input is assigned to unit 5
     !     and system output is assigned to unit OUTU and all other
     !     units are deassigned.
     DO I=1,MAXUN
        IFREEU(I)=0
     ENDDO
     IFREEU(UNIT)=10
     IFREEU(OUTU)=1
     RETURN
  ENDIF
  !--------------End of $$INITIAL$$ -------------------------------
  !
  IF (FORM.EQ.'FORMATTED') THEN
#if KEY_STRINGM==1 /*  VO : need long records for stringm i/o */
     RECLEN=(2**14 - 1) * 2 + 1
#else /*  VO */
     RECLEN=256
#endif
     CRCONT='LIST'
  ELSE IF (FORM.EQ.'UNFORMATTED') THEN
     RECLEN=1048576
     CRCONT='NONE'
  ELSE
     IF(WRNLEV.GE.2) WRITE(OUTU,'(3A)') &
          ' %VOPEN-ERR: unknown format qualifier "',FORM,'"'
     ERR=.TRUE.
     RETURN
  ENDIF
  !
  FLEN=STRLNG(FILEX)
  IF(FLEN.GT.MXFILE) CALL WRNDIE(-5,'<VOPEN>','FLEN.GT.MXFILE')
  FILE=FILEX(1:FLEN)
  !
  !     Process name conversion if necessary
  !
#if KEY_UNIX==1
  !     remove quotes
  NOQUOTES=.TRUE.
  IF(FILEX(1:1).EQ.'"') THEN
     NOQUOTES=.FALSE.
     IF (FILEX(FLEN:FLEN).EQ.'"') THEN
        FILE=FILEX(2:FLEN-1)
        FLEN=FLEN-2
     ELSE
        FILE=FILEX(2:FLEN)
        FLEN=FLEN-1
     ENDIF
  ENDIF
#if KEY_UNIX==1
  CALL EXPNAM(FILE,FLEN,ERR)
  IF (ERR) RETURN
#endif 
#else /**/
#error  'Unrecognized machine type in VOPEN(2)'
#endif 
  !
  CALL TRIMA(FILE,FLEN)
#if KEY_UNIX==1
  IF(NOQUOTES) THEN
     IF (ACCESS.EQ.'READ') THEN
        !     ---- for read try UPPER or lower case files
        INQUIRE(FILE=FILE,EXIST=EXIST)
        IF(.NOT.EXIST) CALL CNVTLC(FILE,FLEN)
     ELSE
        !     ---- for write & append use UPPER or lower case file
        !     depending on LOWER value (initially FILE is in UPPER case)
        IF(LOWER) CALL CNVTLC(FILE,FLEN)
     ENDIF
  ENDIF
#endif 
  IF(PRNLEV.GE.2) WRITE(OUTU,'(3A)') &
       ' VOPEN> Attempting to open::',FILE(1:FLEN),'::'
  !
  IF (ACCESS.EQ.'APPEND') THEN
        IF(IOLEV.LT.0) RETURN
#if KEY_OSX==1
     OPEN(FILE=FILE(1:FLEN),FORM=FORM,STATUS='OLD', &
          ACCESS='SEQUENTIAL',ERR=9999,UNIT=UNIT)
#elif KEY_GNU==1
#if KEY_NOCONVERT==0 && __PGI == 0 && KEY_PATHSCALE==0
     IF(QCONV.AND. FORM == 'UNFORMATTED')THEN
        OPEN(FILE=FILE(1:FLEN),FORM=FORM,STATUS='OLD', &
             ACCESS='SEQUENTIAL',position='APPEND',ERR=9999,UNIT=UNIT,CONVERT=CONV &
#if KEY_STRINGM==1 /*  VO: stringm, added record length */
     &      ,RECL=RECLEN & 
#endif
     &      )
     ELSE
#endif 
     OPEN(FILE=FILE(1:FLEN),FORM=FORM,STATUS='OLD', &
          ACCESS='SEQUENTIAL',position='APPEND',ERR=9999,UNIT=UNIT &
#if KEY_STRINGM==1 /*  VO: stringm, added record length */
     &       ,RECL=RECLEN & 
#endif
     &   )

#if KEY_NOCONVERT==0 && __PGI == 0 && KEY_PATHSCALE==0
     ENDIF 
#endif 
#elif KEY_UNIX==1
     CALL CLOSE_OPEN(FILE(1:FLEN))
#if KEY_NOCONVERT==0 && __PGI == 0 && KEY_PATHSCALE==0
     IF(QCONV.AND. FORM == 'UNFORMATTED')THEN
        OPEN(FILE=FILE(1:FLEN),FORM=FORM,STATUS='OLD', &
             ACCESS='SEQUENTIAL',ERR=9999,UNIT=UNIT,RECL=RECLEN,CONVERT=CONV)
     ELSE
#endif 
     OPEN(FILE=FILE(1:FLEN),FORM=FORM,STATUS='OLD', &
          ACCESS='SEQUENTIAL',ERR=9999,UNIT=UNIT,RECL=RECLEN)
     ENDFILE (UNIT=UNIT)
#if KEY_NOCONVERT==0 && __PGI == 0 && KEY_PATHSCALE==0
     ENDIF 
#endif 
#else /**/
#error  'Unrecognized machine type in VOPEN(3)'
#endif 
     !
  ELSE IF (ACCESS.EQ.'READ') THEN
#if KEY_UNIX==1 || KEY_OSX==1
     CALL CLOSE_OPEN(FILE(1:FLEN))
#if KEY_NOCONVERT==0 && __PGI == 0 && KEY_PATHSCALE==0
     IF(QCONV.AND. FORM == 'UNFORMATTED')THEN
        OPEN(FILE=FILE(1:FLEN),FORM=FORM,STATUS='OLD', &
             ACCESS='SEQUENTIAL',ERR=9999,UNIT=UNIT,CONVERT=CONV &
#if KEY_STRINGM==1 /*  VO: stringm, added record length */
     &      ,RECL=RECLEN & 
#endif
     &      )
     ELSE
#endif 
     OPEN(FILE=FILE(1:FLEN),FORM=FORM,STATUS='OLD', &
          ACCESS='SEQUENTIAL',ERR=9999,UNIT=UNIT &
#if KEY_STRINGM==1 /*  VO: stringm, added record length */
     &   ,RECL=RECLEN & 
#endif
     &   )
#if KEY_NOCONVERT==0 && __PGI == 0 && KEY_PATHSCALE==0
     ENDIF 
#endif 
#else /**/
#error  'Unrecognized machine type in VOPEN(4)'
#endif 
     !
  ELSE IF (ACCESS.EQ.'WRITE') THEN
        IF(IOLEV.LT.0) RETURN
#if KEY_OSX==1
     OPEN(FILE=FILE(1:FLEN),FORM=FORM,STATUS='UNKNOWN', &
          ACCESS='SEQUENTIAL',ERR=9999,UNIT=UNIT)
#elif KEY_UNIX==1
     CALL CLOSE_OPEN(FILE(1:FLEN))
#if KEY_NOCONVERT==0 && KEY_PATHSCALE==0
     IF(QCONV.AND. FORM == 'UNFORMATTED')THEN
        OPEN(FILE=FILE(1:FLEN),FORM=FORM,STATUS='UNKNOWN', &
             ACCESS='SEQUENTIAL',UNIT=UNIT,CONVERT=CONV)
!             ACCESS='SEQUENTIAL',ERR=9999,UNIT=UNIT,CONVERT=CONV)
     ELSE
#endif 
     OPEN(FILE=FILE(1:FLEN),FORM=FORM,STATUS='UNKNOWN', &
          ACCESS='SEQUENTIAL',ERR=9999,UNIT=UNIT)
#if KEY_NOCONVERT==0 && KEY_PATHSCALE==0
     ENDIF 
#endif 
#else /**/
#error  'Unrecognized machine type in VOPEN(5)'
#endif 
     !
  ELSE IF (ACCESS.EQ.'DIRECT') THEN
     IF(DIRRLEN.LT.1) &
        CALL WRNDIE(-5,"VOPEN","DIRECT I/O USED WITH AN INVALID RECORD LENGTH")
     CALL CLOSE_OPEN(FILE(1:FLEN))
#if KEY_NOCONVERT==0 && KEY_PATHSCALE==0
     IF(QCONV.AND. FORM == 'UNFORMATTED')THEN
        OPEN(FILE=FILE(1:FLEN),FORM=FORM,ACTION='READWRITE', &
          STATUS='UNKNOWN',ACCESS='DIRECT',ERR=9999,UNIT=UNIT, &
          RECL=DIRRLEN,CONVERT=CONV)
     ELSE
#endif 
     OPEN(FILE=FILE(1:FLEN),FORM=FORM,ACTION='READWRITE', &
          STATUS='UNKNOWN',ACCESS='DIRECT',ERR=9999,UNIT=UNIT, &
          RECL=DIRRLEN)
#if KEY_NOCONVERT==0 && KEY_PATHSCALE==0
     ENDIF 
#endif 
  ELSE IF (ACCESS.EQ.'EMAP') THEN
#if KEY_PATHSCALE==0
     OPEN(FILE=FILE(1:FLEN),FORM='UNFORMATTED',ACTION='READWRITE', &
          STATUS='UNKNOWN',ACCESS='STREAM',ERR=9999,UNIT=UNIT)
#else
     OPEN(FILE=FILE(1:FLEN),FORM='UNFORMATTED',ACTION='READWRITE', &
          STATUS='UNKNOWN',ACCESS='DIRECT',RECL=1,ERR=9999,UNIT=UNIT)
#endif
  ELSE
     IF(WRNLEV.GE.2) WRITE(OUTU,'(3A)') &
          ' %VOPEN-ERR: unknown access qualilifer "',ACCESS,'"'
     ERR=.TRUE.
  ENDIF
  GOTO 8888
9999 ERR=.TRUE.
8888 CONTINUE
  !
  IF (.NOT.ERR) THEN
     !
     !     put appropriate code in IFREEU array:
     !        +10 read formatted
     !        +1  write/append formatted
     !        -1  write/append unformatted
     !        -10 read unformatted
     IF (FORM.EQ.'FORMATTED') THEN
        IFREEU(UNIT)=1
     ELSE
        IFREEU(UNIT)=-1
     ENDIF
     IF (ACCESS.EQ.'READ') IFREEU(UNIT)=IFREEU(UNIT)*10
  ENDIF
  !
  RETURN
END SUBROUTINE VOPEN
end module machio

SUBROUTINE VINQRE(MODE,NAME,MAXLEN,LENGTH,QOPEN,QFORM,QWRITE, &
     UNIT)
  !----------------------------------------------------------------------
  !     file inquiry by file name or FORTRAN unit
  !     Flag QOPEN indicates whether file or unit is "open".
  !     Flag QFORM indicates whether file was opened formatted.
  !     Flag QWRITe indicates whether file was opened write-access.
  !     For inquiry by unit MAXLEN has to be specified (max length of NAME
  !     and LENGTH returns with the length of NAME.
  !     For inquiry by file the two names INPUT and OUTPUT are reserved
  !     for the standard input and output channels 5 and OUTU.
  !
  use chm_kinds
  use machio
  use stream
  use string
  use parallel
  use dimens_fcm

  implicit none
#if KEY_UNIX==1
  !     for environment variable expansion
  CHARACTER(len=MXCMSZ) ENAME
  INTEGER ELEN
#endif 
  CHARACTER(len=*) MODE, NAME
  INTEGER MAXLEN, LENGTH
  LOGICAL QOPEN, QFORM, QWRITE
  INTEGER UNIT
  !
  !     begin
  QOPEN=.TRUE.

  LENGTH=0
  !
  !     Don't check for slave processes.
  IF(IOLEV.LT.0) RETURN
  !
  IF (MODE.EQ.'FILE'.AND.NAME.EQ.'INPUT') THEN
     UNIT=5
  ELSE IF (MODE.EQ.'FILE'.AND.NAME.EQ.'OUTPUT') THEN
     UNIT=OUTU
#if KEY_UNIX==1
  ELSE IF (MODE.EQ.'FILE') THEN
     !     added for expanding environment variables shf 4/90
     ENAME = NAME
     CALL EXPNAM(ENAME,ELEN,QOPEN)
     QOPEN = .NOT.QOPEN
     UNIT = -1
     IF(QOPEN) THEN
        INQUIRE(FILE=ENAME(1:ELEN),OPENED=QOPEN,NUMBER=UNIT)
        IF (UNIT.LE.0.OR.UNIT.GT.99) QOPEN=.FALSE.
     ENDIF
#else /**/
  ELSE IF (MODE.EQ.'FILE') THEN
     INQUIRE(FILE=NAME,OPENED=QOPEN,NUMBER=UNIT)
#endif 
  ELSE IF (MODE.EQ.'UNIT') THEN
     NAME(1:MAXLEN) = ' '
     INQUIRE(UNIT=UNIT,OPENED=QOPEN,NAME=NAME)
     LENGTH=MAXLEN
     CALL TRIME(NAME,LENGTH)
     !
  ENDIF
  !
  !     if file is open then get QFORM and QWRITE flags
  IF (QOPEN) THEN
     IF ((IFREEU(UNIT).EQ.+1).OR.(IFREEU(UNIT).EQ.+7)) THEN
        QFORM=.TRUE.
        QWRITE=.TRUE.
     ELSE IF ((IFREEU(UNIT).EQ.+10).OR.(IFREEU(UNIT).EQ.+70)) THEN
        QFORM=.TRUE.
        QWRITE=.FALSE.
     ELSE IF ((IFREEU(UNIT).EQ.-1).OR.(IFREEU(UNIT).EQ.-7)) THEN
        QFORM=.FALSE.
        QWRITE=.TRUE.
     ELSE IF ((IFREEU(UNIT).EQ.-10).OR.(IFREEU(UNIT).EQ.-70)) THEN
        QFORM=.FALSE.
        QWRITE=.FALSE.
     ELSE IF (IFREEU(UNIT).EQ.0) THEN
        QFORM=.FALSE.
        QWRITE=.FALSE.
        !C        IF(WRNLEV.GE.2) WRITE(OUTU,45) UNIT
        !C    45  FORMAT(' VINQRE: File',I4,' open without using VOPEN.')
     ENDIF
  ENDIF
  RETURN
END SUBROUTINE VINQRE

SUBROUTINE VCLOSE(UNIT,DISPOS,ERR)
  !----------------------------------------------------------------------
  !     closes a file/unit with disposition DISPOS and sets the
  !     corresponding IFREEU element to zero.
  !
  !     DONT CLOSE UNIT 5 OR OUTU !!
  !
  !      By Axel Brunger, 21-APR-85
  !
  use chm_kinds
  use machio
  use stream
  use parallel
  implicit none
  INTEGER UNIT
  CHARACTER(len=*) DISPOS
  character(len=6) dispose
  LOGICAL ERR
  !
  !     begin
  ERR=.FALSE.
  !     Don't close any files if a slave process.
  IF(IOLEV.LT.0) RETURN
  !
  IF (UNIT.NE.OUTU.AND.UNIT.NE.5) THEN
     IF(PRNLEV.GE.2) WRITE(OUTU,45) UNIT,DISPOS
45   FORMAT(' VCLOSE: Closing unit',I5,' with status "',A,'"')
     if( index(dispos,"KEEP") > 0 ) then
        CLOSE(UNIT=UNIT,status="KEEP",ERR=9999)
     elseif(index(dispos,"DELETE") > 0) then
        CLOSE(UNIT=UNIT,status="DELETE",ERR=9999)
     else
        write(outu,'(a,a)') &
             "ERROR in DISPOS argument to VCLOSE: ",dispos
        write(outu,'(a)') &
             "     Changing to KEEP"
        CLOSE(UNIT=UNIT,status="KEEP",ERR=9999)
     endif
     IFREEU(UNIT)=0
  ENDIF
  return
  
9999 CONTINUE
  IF(WRNLEV.GE.2) then
     !         WRITE(OUTU,'(" VCLOSE-ERR: error during close ")')
  endif
  ERR=.TRUE.
  RETURN
END SUBROUTINE VCLOSE

SUBROUTINE TRYORO(UNIT,FMT)
  !----------------------------------------------------------------------
  use chm_kinds
  implicit none
  INTEGER UNIT
  CHARACTER(len=*) FMT
  !
  RETURN
END SUBROUTINE TRYORO

SUBROUTINE SAVEIT(IUNIT)
  !-----------------------------------------------------------------------
  !     Close and reopen the output file so that the output will be
  !     saved if the VAX system crashes.
  !-----------------------------------------------------------------------
  use chm_kinds
  use dimens_fcm
  use stream
  use parallel
  implicit none
  !
  INTEGER IUNIT,ISTAT
  logical qopn
#if KEY_OSX==1
  !     22-OCT-91, Youngdo Won
  !     IBM RS/6000 machines require restart files saved
  !                 does not support ACCESS='APPEND'
  CHARACTER(len=128) FILENM
  CHARACTER(len=16) FORMN
  INTEGER      RECLEN
  IF(IUNIT.EQ.5 .OR. IUNIT.EQ.OUTU) RETURN
  IF(IOLEV.LT.0) RETURN
  INQUIRE (UNIT=IUNIT,NAME=FILENM,FORM=FORMN,RECL=RECLEN)
  CLOSE (UNIT=IUNIT)
  OPEN (UNIT=IUNIT,STATUS='OLD',FILE=FILENM,FORM=FORMN)
#elif KEY_GNU==1
  ! In parallel/parallel this must be checked.
  ! but it doesnt hurt if we do it always
  inquire(unit=iunit,opened=qopn)
  if(qopn)call FLUSH (IUNIT)
#else /**/
  !     Do nothing for the other machine types: Apollo, Cray
#endif 
  RETURN
END SUBROUTINE SAVEIT

SUBROUTINE EXPNAM(PATH1,LEN2,ERR)
  !----------------------------------------------------------------------
  ! EXPNAM:  Expand the environment variable into the path name.
  !          Allows file names in the CHARMM open statement of the
  !          type: $env_var/file.name or $env_var.  The variable
  !          PATH1 on input contains the file path (potentially)
  !          containing the environment variable.  On output
  !          it contains the expanded path.  The $ character must
  !          be the first non-blank character in the string.
  !          Note to programmers:  PATH1 must be long enough to contain
  !          the expanded path.
  !
  !     written by: Stephen Fleischman 6/88.
  !
  use, intrinsic :: iso_fortran_env
  use chm_kinds
  use dimens_fcm
  use comand
  use stream
  use string
  use parallel
  use repdstr
  use cstuff, only: expand_tilde
  
  implicit none
  
  CHARACTER(len=*) PATH1
  CHARACTER(len=MXCMSZ) PATH2
  CHARACTER(len=5) AP
  INTEGER LEN1,LEN2,I1,I2,I3,APLEN, status
  LOGICAL GO,ERR,EXPND

! BIOVIA Code Start : Fix for Windows
#if KEY_WIN32==1 || KEY_WIN64==1
  LOGICAL ISCOLON
  ISCOLON=.FALSE.
#endif
! BIOVIA Code End

  LEN1 = LEN(PATH1)
  CALL TRIMA(PATH1,LEN1)
  LEN2 = LEN1
#if KEY_UNIX==1 /*unix*/
  PATH2 = ' '
  ERR = .FALSE.
  I1 = 0
  IF (PATH1(1:1).EQ.'$') I1 = 2
  I2 = 1
  GO = .TRUE.
  IF (GO) THEN
100  CONTINUE
     IF (I2.GT.LEN1) THEN
        GO = .FALSE.
     ELSE IF (PATH1(I2:I2).EQ.'/') THEN
        GO = .FALSE.
     ELSE IF (PATH1(I2:I2).EQ.':') THEN
! BIOVIA Code Start : Fix for Windows
#if KEY_WIN32==1 || KEY_WIN64==1
        ISCOLON=.TRUE.
#endif
! BIOVIA Code End
        GO = .FALSE.
        I1 = 1
     ELSE
        I2 = I2 + 1
     ENDIF
     IF (GO) GOTO 100
  ENDIF
  I2=I2-1
  ERR = .FALSE.
  IF (I1.EQ.0) THEN
     EXPND = .FALSE.
  ELSE
     EXPND = .TRUE.
  ENDIF
  IF (I1.EQ.2.AND.I2.EQ.1) THEN
     CALL WRNDIE(0,'<EXPNAM>', &
          'The environment variable contained only the dollar sign.')
     ERR = .TRUE.
     EXPND = .FALSE.
  ELSE IF (I1.EQ.2.AND.I2.EQ.LEN1-1) THEN
     CALL WRNDIE(0,'<EXPNAM>', &
          'The last char in the path was a /, i.e. path is a ' &
          //'directory.')
     ERR = .TRUE.
     EXPND = .FALSE.
  ELSE IF (I1.EQ.1.AND.I2.EQ.1) THEN
! BIOVIA Code Start : Fix for Windows
#if KEY_WIN32==1 || KEY_WIN64==1
     IF (ISCOLON .NE. .TRUE.) THEN
#endif
! BIOVIA Code End
     CALL WRNDIE(0,'<EXPNAM>', &
          'The environment variable contained only a colon.')
     ERR = .TRUE.
! BIOVIA Code Start : Fix for Windows
#if KEY_WIN32==1 || KEY_WIN64==1
     ENDIF
#endif
! BIOVIA Code End
     EXPND = .FALSE.
  ELSE IF (I1.EQ.1.AND.I2.EQ.LEN1-1) THEN
     CALL WRNDIE(0,'<EXPNAM>', &
          'The last char in the path was a :, i.e. path is a ' &
          //'directory.')
     ERR = .TRUE.
     EXPND = .FALSE.
  ENDIF
  !
  IF (EXPND .OR. PATH1(1:1).EQ.'~' ) THEN
     !
     !     Check for ~<user-name>/... case
     !
     status = 0
     IF (PATH1(1:1) .EQ. '~' .AND. PATH1(2:2) .NE. '/') THEN
        I3 = INDEX(PATH1, '/')
        status = expand_tilde(path1(1:i3), i3, path2, len2)
     ELSE
        IF (EXPND) call get_environment_variable(PATH1(I1:I2),PATH2)
        IF (PATH1(1:1).EQ.'~') THEN
           call get_environment_variable('HOME',PATH2)
           I2=0
        ENDIF
        LEN2 = LEN(PATH2)
        CALL TRIMA(PATH2,LEN2)
     ENDIF
#if KEY_WIN32==0 && KEY_WIN64==0
     IF (status .ne. 0 .or. LEN2 .LE. 0) THEN
        IF(WRNLEV.GE.2) WRITE(OUTU,'(1X,3A)') '<EXPNAM>', &
             'The environment variable ',PATH1(I1:I2),' does not exist.'
        CALL WRNDIE(0,'<EXPNAM>', &
             'Error in the environment variable.')
        ERR = .TRUE.
        RETURN
     ENDIF
#endif /* KEY_WIN32==0 && KEY_WIN64==0 */
     IF( (LEN2 + LEN1 - I2) .GT. LEN(PATH1)) THEN
        IF(WRNLEV.GE.2) WRITE(OUTU,*) '<EXPNAM>', &
             'The character variable containing the path is not long', &
             'enough to accept the expanded path.'
        CALL WRNDIE(0,'<EXPNAM>','Too long path.')
     ENDIF
     IF (EXPND .AND. PATH2(LEN2:LEN2).NE.'/') THEN
        LEN2 = LEN2+1
        PATH2(LEN2:LEN2) = '/'
     ENDIF
     PATH2(LEN2+1:) = PATH1(I2+2:LEN1)
     LEN2 = LEN(PATH2)
     CALL TRIMA(PATH2,LEN2)
     PATH1 = PATH2(1:LEN2)
  ENDIF
#endif /* (unix)*/

#if KEY_REPDSTR==1
  !     Append the replica number to filename
  !     before October 2008 it used to be process number
  !     which turned out not very practical to handle!
  IF(QREPDSTR.and.qrepioset)THEN
     if(qrepdnames)then
        WRITE(AP,'(i5)')IREPDSTR
        APLEN=5
        CALL TRIMA(AP,APLEN)
        CALL ADDST(PATH1,MXCMSZ,LEN2,'_',1)
        CALL ADDST(PATH1,MXCMSZ,LEN2,AP,APLEN)
        !C         write(*,*)'EXPNAM>gme,file=',MYNODG,PATH1(1:LEN2)
     endif
  ENDIF
#endif 
  RETURN
END SUBROUTINE EXPNAM

SUBROUTINE GFLUSH(IUNIT)
  use chm_kinds
  implicit none
  INTEGER IUNIT
  RETURN
END SUBROUTINE GFLUSH

#if KEY_OSX==1
SUBROUTINE FLUSH(IUNIT)
  use chm_kinds
  implicit none
  INTEGER IUNIT
  flush(iunit)
  return
end SUBROUTINE FLUSH
#endif 

SUBROUTINE FILBAK(IUNIT,PBMLEV,ERR)
  use chm_kinds
  implicit none
  INTEGER IUNIT,PBMLEV
  LOGICAL ERR
  RETURN
END SUBROUTINE FILBAK
