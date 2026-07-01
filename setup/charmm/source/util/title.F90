SUBROUTINE RDTITL(TITLE,NTITL,IUNIT,IOP)
  !-----------------------------------------------------------------------
  !     This routine tries to read a title. If a title is already
  !     present (i.e. NTITL /= 0) the current title is untouched
  !     when no title is found.
  !
  !     All title lines MUST begin with a "*". The last line is indicated
  !     by just a '*' with nothing except blanks behind it.
  !     If the current input line contains no star a backspace on the
  !     current input unit is performed. In case of interactive use
  !     the error handler of BACKSPACE yields to an error message.
  !
  !     The titles are read using RDCMND, thus parameter substitutions are
  !     allowed.
  !
  !     1-FEB-83 AB
  !
  use chm_kinds
  use stream
  use string
  use parallel
  ! VO string v
#if KEY_MULTICOM==1
  use multicom_aux                                   
#endif
#if KEY_STRINGM==1
  use machio,only: ifreeu                            
#endif
  ! VO string ^
  !
  use repdstr
  use machutil,only:die
  !
  implicit none
  INTEGER IWIDTH, IWMAX, MAXTIT
  PARAMETER (IWIDTH=80, IWMAX=100, MAXTIT=32)
  character(len=IWIDTH) TITLE(*)
  INTEGER   NTITL, IUNIT,IOP
  !
  CHARACTER (len=IWMAX) LINE
  LOGICAL   EOF, DONE, INIT
  INTEGER   LINLEN,I,IADD,LINLEN0
  !
  integer*4 ntitl4
  integer int8traju
  ! VO stringm
#if KEY_STRINGM==1
  integer :: oldiol
  logical :: qstr
#endif
  !
  character(len=8) SNONE
  DATA SNONE/'* NONE *'/
#if KEY_STRINGM==1
  ! VO stringm v
  qstr=iunit.le.size(ifreeu) ! protect against OOB on ifreeu
  if (qstr) qstr=((MOD(IFREEU(IUNIT),8).eq.0).and.(IFREEU(IUNIT).ne.0).and.(ME_LOCAL.eq.0))
  if (qstr) then
   oldiol=iolev
   iolev=1
  endif
#endif
  !
#if KEY_REPDSTR==1
  IF(.NOT.QREPDSTR) THEN            
#endif
     IF(IOLEV < 0 &
#if KEY_MULTICOM==1 /*  VO string */
     & .and.ME_PARSER.ne.0 &        
#endif
     &           ) RETURN
#if KEY_REPDSTR==1
  ENDIF                             
#endif

  IF (IOP == -2) THEN
     !       ------- READ BINARY PSF TITLE
     READ(IUNIT) (TITLE(I),I=1,NTITL)
     !     
  ELSE IF (IOP == -1) THEN
     !       --------- READ BINARY FILE TITLE
       int8traju=0
           READ(IUNIT) NTITL4,(TITLE(I),I=1,NTITL4)
           ntitl=ntitl4
        !         READ(IUNIT) NTITL,(TITLE(I),I=1,NTITL)
     !
  ELSE IF (IOP == 0) THEN
     !     READ CARD FORMAT
     !
     INIT=.FALSE.
     EOF=.FALSE.
     DONE=.FALSE.
10   CONTINUE
     ! LIBRARY DEBUG
     !  Again here is where there is a problem because rdcmnd tries to
     !  read from unit 5, sdin.
     CALL RDCMND(LINE,IWMAX,LINLEN,IUNIT,EOF,.FALSE.,.FALSE.,' ')

     IF (.NOT.EOF) THEN
        LINLEN0=LINLEN
        IF(LINLEN0 < 1)LINLEN0=1
        IF (IUNIT == ISTRM) THEN
           IF(PRNLEV >= 2)WRITE(OUTU,'(2A)')' RDTITL> ',LINE(1:LINLEN0)
        ELSE
           IF(PRNLEV >= 2)WRITE(OUTU,'(2A)') ' TITLE> ',LINE(1:LINLEN0)
        ENDIF
     ENDIF
     IF (EOF) THEN
        DONE=.TRUE.
     ELSE IF (EQSTA(LINE,LINLEN,'*')) THEN
        DONE=.TRUE.
     ELSE IF (EQSTA(LINE,LINLEN,'#')) THEN
        DONE=.TRUE.
     ELSE IF ((EQSTA(LINE,1,'#').OR.EQSTA(LINE,1,'*')) &
          .AND.NTITL < MAXTIT) THEN
        IF (.NOT.INIT) THEN
           INIT=.TRUE.
           IADD=0
           IF (.NOT.(EQSTA(LINE,1,'#'))) NTITL=0
        ENDIF
        IF (EQSTA(LINE,1,'#')) LINE(1:1)='*'
        !
        IADD=IADD+1
        IF(NTITL < IADD) NTITL=IADD
        CALL FILSPC(LINE,IWIDTH,LINLEN)
        CALL COPYST(TITLE(IADD),IWIDTH,LINLEN,LINE,IWIDTH)
     ELSE IF ((EQSTA(LINE,1,'#').OR.EQSTA(LINE,1,'*')) &
          .AND.NTITL >= MAXTIT) THEN
        CALL WRNDIE(0,'<RDTITL>','Title is greater than 32 lines')
        DONE=.TRUE.
     ELSE
        !
        !         no title found...
        !
        ! LIBRARY DEBUG
        ! this too presents a problem since you cannot backspace
        ! commands from the python stream I expect.
        BACKSPACE(UNIT=IUNIT,ERR=70)

        GOTO 80
70      CALL WRNDIE(1,'<RDTITL>','Title expected.')
80      DONE=.TRUE.
     ENDIF
     IF (.NOT.(DONE)) GOTO 10
     !
     IF (.NOT.INIT .AND. WRNLEV >= 2) WRITE(OUTU,'(A)') &
          ' RDTITL> No title read.'
     !
  ELSE
     IF(WRNLEV >= 2) WRITE(OUTU,'(A)') &
          ' RDTITL> Programmer info. Invalid IOP'
     CALL DIE
  ENDIF
  !
  IF(NTITL <= 0) THEN
     CALL WRNDIE(1,'<RDTITL>','Title expected.')
     NTITL=1
     TITLE(1)=SNONE
  ENDIF
  !
#if KEY_STRINGM==1 /*  restore iolev */
  if (qstr) iolev=oldiol 
#endif
  !
  RETURN
END SUBROUTINE RDTITL

SUBROUTINE WRTITL(TITLE,NTITL,IUNIT,IOP)
  !-----------------------------------------------------------------------
  !     This routine writes titles. It accepts an type option;
  !     -2  binary psf format, add nothing
  !     -1  binary output, add time and date
  !      0  card output, add time and date
  !      1  print title, add nothing, include leader (WRTITL:)
  !      2  update date only (no output)
  !      3  output in brief card format (no first char or last line)
  !
  !     For the card and binary options,
  !     the date and username will be added at the end.
  !     If a date and username is already present, it will be superceeded.
  !     For the print option, the date and time information is
  !     left as it was.
  !
  !     1-FEB-83 AB
  !
  use chm_kinds
  use stream
  use string
  use parallel
  use repdstr
  !
  ! VO string
#if KEY_STRINGM==1
  use multicom_aux         
#endif
#if KEY_STRINGM==1
  use machio, only: ifreeu 
#endif
  ! VO string
  !
  implicit none
  INTEGER IWIDTH, MAXTIT
  PARAMETER (IWIDTH=80, MAXTIT=32)
  character(len=IWIDTH) TITLE(*)
  INTEGER   NTITL,IUNIT,IOP
  integer*4 ntitl4
  character(len=IWIDTH) LINE
  !
  INTEGER   J, ILEN
  !
#if KEY_STRINGM==1
  ! VO stringm v
  logical :: qstr
  qstr=(iunit.le.size(ifreeu)).and.(iunit.gt.0) ! protect against OOB on ifreeu
  if (qstr) qstr=((MOD(IFREEU(IUNIT),8).eq.0).and.(IFREEU(IUNIT).ne.0).and.(ME_LOCAL.eq.0))
#endif
  ! VO ^
  !
#if KEY_REPDSTR==1
  IF(.NOT.QREPDSTR) THEN                           
#endif
     IF(IUNIT == OUTU .AND. PRNLEV < 2) RETURN
     IF(IUNIT /= OUTU .AND. IOLEV < 0 &
!    VO : do not return if this is a head node in an ensemble or string
#if KEY_STRINGM==1
     &      .and..not.qstr            &         
#endif
     &                                ) RETURN
#if KEY_REPDSTR==1
  ENDIF                                            
#endif
  !
  IF (IOP == -2) THEN
     !     for binary psf format
     ntitl4=ntitl
     WRITE(IUNIT) (TITLE(J),J=1,NTITL4)
     !
  ELSE IF (IOP == -1) THEN
     !     for binary output
     CALL DATEID(TITLE,NTITL)
     ntitl4=ntitl
     WRITE(IUNIT) NTITL4,(TITLE(J),J=1,NTITL)
     !
  ELSE IF (IOP == 0) THEN
     !     for card output
     CALL DATEID(TITLE,NTITL)
     DO J=1,NTITL
        CALL COPYST(LINE,IWIDTH,ILEN,TITLE(J),IWIDTH)
        CALL TRIME(LINE,ILEN)
        WRITE(IUNIT,'(A)') LINE(1:ILEN)
     enddo
     WRITE(IUNIT,'(A)') '*'
     !
  ELSE IF (IOP == +1) THEN
     !     just print out current title
     DO J=1,NTITL
        CALL COPYST(LINE,IWIDTH,ILEN,TITLE(J),IWIDTH)
        CALL TRIME(LINE,ILEN)
        WRITE(IUNIT,'(2A)') ' TITLE>  ',LINE(1:ILEN)
     enddo
     WRITE(IUNIT,'(A)') ' TITLE>  *'
     !
  ELSE IF (IOP == +2) THEN
     !     update date and username
     CALL DATEID(TITLE,NTITL)
     !
  ELSE IF (IOP == +3) THEN
     !     just print out current title in brief format
     DO J=1,NTITL
        CALL COPYST(LINE,IWIDTH,ILEN,TITLE(J),IWIDTH)
        CALL TRIME(LINE,ILEN)
        WRITE(IUNIT,'(A)') LINE(2:ILEN)
     enddo
     !
  ENDIF
  !
  RETURN
END SUBROUTINE WRTITL

SUBROUTINE DATEID(TITLE,NTITL)
  !-----------------------------------------------------------------------
  !     Procedure PUT-IN-DATE-AND-USER
  !     Introduced while converting to FORTRAN by Youngdo Won
  use chm_kinds
  use exfunc
  use machutil,only:daytim
  use startup
  use string
  implicit none
  INTEGER  IWIDTH, MAXTIT
  PARAMETER (IWIDTH=80, MAXTIT=32)
  character(len=IWIDTH) TITLE(*)
  INTEGER  NTITL
  !
  INTEGER  K, L, M, II, JJ, KK
  ! BIOVIA Code Start : Bug fix
  character(len=24) USERNM
  ! BIOVIA Code End
  character(len=8)  SDATE
  DATA SDATE/'*  DATE:'/
  !
  if(ntitl == 0 ) then
     ntitl=ntitl+1
  else
     IF (INDX(TITLE(NTITL),IWIDTH,SDATE,8) == 0) NTITL=NTITL+1
  endif
  IF (NTITL > MAXTIT) THEN
     CALL WRNDIE(1,'<WRTITL>', &
          'Maximum number of title lines exceeded')
  ELSE
#if KEY_STRINGM==0
     CALL DAYTIM(K,L,M,II,JJ,KK)
     CALL GETNAM(USERNM)
     WRITE(TITLE(NTITL),600) SDATE,K,L,M,II,JJ,KK,USERNM
#else
! VO : disable system calls because they are problematic on some platforms on which string is used
     write(title(ntitl),'(2A)') sdate, ' UNAVAILABLE WHEN THE STRINGM MODULE IS USED'
#endif
  ENDIF
  ! BIOVIA Code Start : Bug fix
600 FORMAT(A8,4X,I2,'/',I2,'/',I2,5X,I2,':',I2,':',I2,6X, &
       'CREATED BY USER: ',A24)
  ! BIOVIA Code End
  !
  RETURN
END SUBROUTINE DATEID

SUBROUTINE CLOLGU(COMLYN,COMLEN,UNIT)
  !-----------------------------------------------------------------------
  !     This routine closes a logical unit.  It checks the command line
  !     for the unit number.
  !
  !     Author: David States
  !
  use chm_kinds
  use exfunc
  use stream
  use string
  implicit none
  !
  character(len=*) COMLYN
  INTEGER COMLEN,UNIT
  LOGICAL OPENUN, ERROR
  INTEGER I
  character(len=10) DISPOS
  character(len=80) NAME
  LOGICAL QFORM,QWRITE
  INTEGER MAXLEN, LENGTH
  PARAMETER (MAXLEN=80)
  !
  UNIT=GTRMI(COMLYN,COMLEN,'UNIT',-1)
  DISPOS=GTRMA(COMLYN,COMLEN,'DISP')
  IF (DISPOS == ' ') DISPOS='KEEP'
  IF (DISPOS == 'DELE') DISPOS='DELETE'
  IF (UNIT < 0) THEN
     CALL WRNDIE(0,'<CLOLGU>','NO UNIT NUMBER SPECIFIED')
     RETURN
  ENDIF

  name = ' '  ! initialize name
  CALL VINQRE('UNIT',NAME,MAXLEN,LENGTH,OPENUN,QFORM,QWRITE,UNIT)
  IF (.NOT.OPENUN) THEN
     IF(WRNLEV >= 2) WRITE (OUTU,'(2A)') &
          ' CLOLGU>  *****  WARNING  *****', &
          ' Attempt to close unit that was not open.'
     RETURN
  ENDIF
  I=10
  CALL TRIME(DISPOS,I)
  CALL VCLOSE(UNIT,DISPOS(1:I),ERROR)
  RETURN
END SUBROUTINE CLOLGU

SUBROUTINE OPNLGU(COMLYN,COMLEN,UNIT)
  !-----------------------------------------------------------------------
  !     THIS ROUTINE PARSES THE COMMAND LINE FOR A LOGICAL UNIT NUMBER
  !     FOLLOWING THE KEYWORD "UNIT", CHECKS FOR KEYWORD SPECIFICATIONS:
  !     FORMATTED ("FORM" OR "CARD")
  !     UNFORMATTED ("UNFO" OR "FILE")
  !     READONLY AND SHARED ("READ")
  !     WRITE ENABLED ("WRIT")
  !     APPEND TO CURRENT FILE ("APPE")
  !     DIRECT ACCESS ("DIRECT")
  !     RECORD SIZE FOR DIRECT ACCESS ("RECL")
  !     IT PICKS UP A RECORD SIZE AFTER RECL IN UNITS USED BY A GIVEN
  !     MACHINE (SUCH AS BYTES OR LONGWORDS). THEN IT USES THE REMAINDER
  !     OF THE LINE AS A FILE NAME TO OPEN THE APPROPRIATE LOGICAL UNIT
  !     TO THAT FILE.
  !
  !     FOR NEW FILES THE DEFAULT IS UNFORMATTED. OLD FILES WILL BE TESTED
  !     BY READING A LINE AND SEEING WHETHER IT CONTAINS VALID ASCII, AND
  !     THE FORMATTING SET ACCORDINGLY. IF "READ" OR "WRITE" ARE NOT
  !     SPECIFIED IT WILL TRY TO OPEN AN OLD FILE FOR READING. IF THAT
  !     FAILS IT WILL OPEN A NEW FILE.
  !
  !     Authors: David States
  !     Robert Bruccoleri
  !
  !     Direct Access added by William Laidig 6/2/88, fixed to work
  !     properly in the F95 code by Tim Miller, June, 2011.
  !    
  !     Conversion of binary representation (compiler dependent) CONV keyword. LNI, February 2013
  use chm_kinds
  use exfunc
  use machio, only:vopen
  use stream
  use string
  use parallel
  use param_store, only: set_param

  implicit none

  CHARACTER FRMSPC*11,ACCSPC*17,DALEN*10
  character(len=*) COMLYN
  INTEGER COMLEN,UNIT
  !
  INTEGER MAXLEN, LENGTH
  PARAMETER (MAXLEN=256)
  CHARACTER FILENM*(MAXLEN), NAME*(MAXLEN)
  !
  LOGICAL OPENUN,ERROR, QFORM, QWRITE
  INTEGER FILLEN,I,UNIT2,NRECSZ, K
#if KEY_NOCONVERT==0
  CHARACTER(LEN=20) CONV
#endif 
! BIOVIA Code Start : Fix for Windows
#if KEY_WIN32==1
  CHARACTER*(MAXLEN) PIPESTRU,PIPESTRL
  PIPESTRU='//./PIPE/'
  PIPESTRL='//./pipe/'
#endif
! BIOVIA Code End
  !
  CALL set_param('IOSTAT',-1)
  !
  !     initialize NAME & FILENM (valgrind's complaint)
  DO I=1,MAXLEN
     NAME(I:I)=' '
     FILENM(I:I)=' '
  ENDDO
  !
! BIOVIA Code Start : Fix for Windows
#if KEY_GNU==1
  CALL GTRMWD_SPACE(COMLYN,COMLEN,'NAME',4,FILENM,MAXLEN,FILLEN)
#else
  CALL GTRMWD(COMLYN,COMLEN,'NAME',4,FILENM,MAXLEN,FILLEN)
#endif
! BIOVIA Code End
  IF (FILLEN == 0) THEN
     CALL WRNDIE(0,'<OPNLGU>','NO FILE NAME GIVEN')
     UNIT=-1
     RETURN
  ENDIF

  ! rvenable feb97
  ! remove any embedded double quotes from @param substitution
  K = FILLEN
  DO I=2,FILLEN
     IF (FILENM(I:I) == '"') THEN
        FILENM(I:) = FILENM(I+1:)
        K = K-1
     ENDIF
  ENDDO
  IF (K /= FILLEN) FILLEN = K
  !
  !
  UNIT=GTRMI(COMLYN,COMLEN,'UNIT',-1)
  IF (UNIT < 0) THEN
     CALL WRNDIE(0,'<OPNLGU>','NO UNIT NUMBER SPECIFIED')
     RETURN
  ENDIF
  CALL VINQRE('UNIT',NAME,MAXLEN,LENGTH,OPENUN,QFORM,QWRITE,UNIT)
  IF (OPENUN) THEN
     IF(WRNLEV >= 2 .and. prnlev >= 2) WRITE(OUTU,'(2A)') &
          ' OPNLGU> Unit already open.', &
          ' The old file will be closed first.'
     CALL VCLOSE(UNIT,'KEEP',ERROR)
  ENDIF
  !
! BIOVIA Code Start : Fix for Windows
#if KEY_WIN32==1
  IF((.NOT.EQST(FILENM,9,PIPESTRU,9)).AND.(.NOT.EQST(FILENM,9,PIPESTRL,9))) THEN
#endif
  CALL VINQRE('FILE',FILENM,I,I,OPENUN,QFORM,QWRITE,UNIT2)
#if KEY_WIN32==1
  ENDIF
#endif
! BIOVIA Code End
  IF (OPENUN) THEN
     IF(WRNLEV >= 2 .and. prnlev >= 2) WRITE(OUTU,'(A,/,2A)') &
          ' OPNLGU> ***** WARNING ***** another unit is already ', &
          '         assigned to the file -', &
          ' it will be disconnected first.'
     CALL VCLOSE(UNIT2,'KEEP',ERROR)
  ENDIF
  !
#if KEY_NOCONVERT==0
  CALL GTRMWA(COMLYN,COMLEN,'CONV',4,CONV,LEN(CONV),K)
  IF(K<=0) CONV='NATIVE'
#endif 
  !
  IF (INDXA(COMLYN,COMLEN,'UNFO') > 0) THEN
     FRMSPC='UNFORMATTED'
  ELSE IF (INDXA(COMLYN,COMLEN,'FILE') > 0) THEN
     FRMSPC='UNFORMATTED'
  ELSE IF (INDXA(COMLYN,COMLEN,'FORM') > 0) THEN
     FRMSPC='FORMATTED'
  ELSE IF (INDXA(COMLYN,COMLEN,'CARD') > 0) THEN
     FRMSPC='FORMATTED'
  ELSE
     CALL WRNDIE(1,'<OPNLGU>', &
          'NO FORMATTING SPECIFICATION, IT WILL BE OPENED UNFORMATTED')
     FRMSPC='UNFORMATTED'
  ENDIF
  !
  ACCSPC='READ'
  IF (INDXA(COMLYN,COMLEN,'APPE') > 0) THEN
     ACCSPC='APPEND'
  ELSE IF (INDXA(COMLYN,COMLEN,'READ') > 0) THEN
     ACCSPC='READ'
  ELSE IF (INDXA(COMLYN,COMLEN,'WRIT') > 0) THEN
     ACCSPC='WRITE'
  ELSE IF (INDXA(COMLYN,COMLEN,'DIRE') > 0) THEN
     ACCSPC='DIRECT'
  ELSE IF (INDXA(COMLYN,COMLEN,'EMAP') > 0) THEN
     ACCSPC='EMAP'
     FRMSPC='UNFORMATTED'
  ELSE
     CALL VOPEN(UNIT,FILENM,FRMSPC,'READ',ERROR,0 &
#if KEY_NOCONVERT==0
     ,CONV=CONV)
#else /**/
     )
#endif 
     IF (ERROR) THEN
        IF(WRNLEV >= 2 .and. prnlev >= 2) WRITE(OUTU,55)
        ACCSPC='WRITE'
     ELSE
        IF(WRNLEV >= 2 .and. prnlev >= 2) WRITE(OUTU,65)
        GOTO 100
     ENDIF
  ENDIF
55 FORMAT(/ &
       ' *****  WARNING  ***** NEITHER READ OR WRITE SPECIFIED'/ &
       ' SINCE NO OLD FILE WAS FOUND A NEW ONE WILL BE CREATED')
65 FORMAT(/ &
       ' *****  WARNING  ***** NEITHER READ NOR WRITE SPECIFIED'/ &
       ' SINCE AN OLD FILE EXISTED IT WILL BE OPENED READONLY')
  !
  IF(ACCSPC == 'DIRECT') THEN
     NRECSZ=GTRMI(COMLYN,COMLEN,'RECL',-1)
     IF(NRECSZ <= 0) THEN
        CALL WRNDIE(0,'<OPNLGU>', &
             'Direct Access record size too small.')
        UNIT=-1
        RETURN
     ENDIF
     !
     ! Tim Miller: not exactly sure why this was here...
     ! but I think it's screwing me up; commenting out
     !IF(IOLEV < 0 .AND. ACCSPC /= 'READ') RETURN
  ELSE
     NRECSZ=-1
  ENDIF
  !
  CALL VOPEN(UNIT,FILENM,FRMSPC,ACCSPC,ERROR,NRECSZ &
#if KEY_NOCONVERT==0
     ,CONV=CONV)
#else /**/
     )
#endif 

  IF(ERROR) THEN
     IF(WRNLEV >= 2 .and. prnlev >= 2) WRITE(OUTU,75) UNIT,FILENM
     CALL WRNDIE(0,'<OPNLGU>','"OPEN" not possible.')
     UNIT=-1
     RETURN
  ENDIF
75 FORMAT(' OPNLGU> Unit',I4,' cannot be opened as ',A)
  !
100 CONTINUE
  !
  CALL set_param('IOSTAT',1)
  !
  IF(PRNLEV < 2) RETURN
  CALL VINQRE('UNIT',NAME,MAXLEN,LENGTH,OPENUN,QFORM,QWRITE,UNIT)
  IF (ACCSPC == 'APPEND') THEN
     WRITE(OUTU,115) UNIT,NAME(1:LENGTH)
  ELSE IF (ACCSPC == 'READ') THEN
     WRITE(OUTU,125) UNIT,NAME(1:LENGTH)
  ELSE IF (ACCSPC == 'WRITE') THEN
     WRITE(OUTU,135) UNIT,NAME(1:LENGTH)
  ELSE IF (ACCSPC == 'DIRECT') THEN
     WRITE(OUTU,145) UNIT,NAME(1:LENGTH),NRECSZ
  ELSE IF (ACCSPC == 'EMAP') THEN
     WRITE(OUTU,155) UNIT,NAME(1:LENGTH)
  ENDIF
115 FORMAT(' OPNLGU> Unit',I3,' opened for APPEND access to ',A)
125 FORMAT(' OPNLGU> Unit',I3,' opened for READONLY access to ',A)
135 FORMAT(' OPNLGU> Unit',I3,' opened for WRITE access to ',A)
145 FORMAT(' OPNLGU> Unit',I3,' opened for DIRECT access to ',A,/, &
       ' with record size of ',I10)
155 FORMAT(' OPNLGU> Unit',I3,' opened for EMAP access to ',A)
#if KEY_NOCONVERT==0
  IF(PRNLEV > 5 .AND. .NOT. QFORM)  WRITE(OUTU,165) CONV
165 FORMAT(' OPNLGU> Using unformatted conversion mode: ', A)
#endif 
  !
  RETURN
END SUBROUTINE OPNLGU

SUBROUTINE PUSTRM(COMLYN,MXCMSZ,COMLEN)
  !-----------------------------------------------------------------------
  !     Pushes the UNIT onto the input stream stack.
  !
  !     Author: Bernie Brooks
  !
  use chm_kinds
  use ctitla
  use exfunc
  use machio,only: maxun
  use stream
  !
#if KEY_MULTICOM==1 /*  VO : stringm : Modifications to allow different streams on different communicators */
  use string, only:encodi,gtrmi,indxa,joinwd,addsta,addst,nextwd,gtrma10, indx, itoa
  use multicom_aux
#if KEY_MPI==1
  use mpi
#endif
#else  /* VO */
  use string, only:encodi,gtrmi,indxa,joinwd,addsta,addst,nextwd
#endif /* VO ^ */
  !
  implicit none
  character(len=*) COMLYN
  INTEGER COMLEN,MXCMSZ
  !
  INTEGER UNIT,UN1,UN999
  LOGICAL QOPEN
  INTEGER WDLEN,IDUMMY
  character(len=4) WD
  !
  INTEGER MAXLEN, LENGTH
  PARAMETER (MAXLEN=128)
  character(len=MAXLEN) :: NAME = ' '
  LOGICAL QFORM,QWRITE
  !
#if KEY_MULTICOM==1 /*  VO : string v */
  character(len=20) :: comm_name=''
  integer :: l=len(comm_name), mpi_err
  integer*4 :: comm_parser, me_comm_parser, size_comm_parser
  logical :: qcomm
#endif /* VO : string ^ */
  !
  UN1=10
  UN999=MAXUN
  idummy=0
  name = ' '

#if KEY_MULTICOM==1 /* (mcom)    VO string v */
! search for communicator spec
  qcomm=(indx(comlyn,comlen,'PARSE',5).gt.0)
  if (qcomm) then
! read communicator name
   comm_name=gtrma10(comlyn,comlen,'PARSE')
   comm_name=adjustl(comm_name); l=len_trim(comm_name)
   if (l.le.0) then
    call wrndie(0,'PUSTRM>',' COMMUNICATOR NAME NOT SPECIFIED. NOTHING DONE.')
    return
   else
! cycle through valid options
    l=5
    if (comm_name(1:5).eq.'WORLD') then
!    this should not be necessary since 'GLOBAL' wraps 'WORLD'
     comm_parser=MPI_COMM_WORLD; 
    elseif (comm_name(1:5).eq.'LOCAL') then
     comm_parser=MPI_COMM_LOCAL
    elseif (comm_name(1:5).eq.'GROUP') then ! for backward compatibility
     comm_parser=MPI_COMM_LOCAL
     comm_name='LOCAL'
    elseif (comm_name(1:6).eq.'GLOBAL') then
     comm_parser=MPI_COMM_GLOBAL
     l=6
    else
     call wrndie(0,'PUSTRM>',' '//comm_name(1:l)//&
     &' IS NOT A VALID COMMUNICATOR. NOTHING DONE.')
     return
    endif
   endif ! (l<0)
  else  ! qcomm
   comm_parser=MPI_COMM_PARSER ! keep current communicator
  endif ! qcomm
  !
  call MPI_COMM_RANK(comm_parser, me_comm_parser, mpi_err)  ! fetch new rank & size
  call MPI_COMM_SIZE(comm_parser, size_comm_parser, mpi_err)
  !
  UNIT=GTRMI(COMLYN,COMLEN,'UNIT',-999)
  if (me_comm_parser.ne.0) then ! non-root ( on the new parser communicator )
  ! keep Nilssen's code here, but do not return
  ! LNI: get rid of the filename
    CALL NEXTWD(COMLYN,COMLEN,NAME,MAXLEN,LENGTH)
  !
  else ! me_comm_parser=0 : only the root(s) will process 
  !
#else /* (mcom)  VO stringm */
  !
  UNIT=GTRMI(COMLYN,COMLEN,'UNIT',-999)
  !
  IF(IOLEV < 0) THEN
  ! LNI: We need to get rid of the filename to allow arguments to be found...
    CALL NEXTWD(COMLYN,COMLEN,NAME,MAXLEN,LENGTH)
!
    RETURN
  ENDIF
#endif /*(mcom)  VO stringm */
  !
  IF(UNIT == -999) THEN
     UNIT=UN999
10   CONTINUE
     !
     CALL VINQRE('UNIT',NAME,MAXLEN,LENGTH,QOPEN,QFORM,QWRITE,UNIT)
     IF (QOPEN) THEN
        UNIT=UNIT-1
        GOTO 10
     ENDIF
     CALL JOINWD(COMLYN,MXCMSZ,COMLEN,'NAME',4)
     CALL ADDSTA(COMLYN,MXCMSZ,COMLEN,' UNIT ')
     CALL ENCODI(UNIT,WD,4,WDLEN)
     CALL ADDST(COMLYN,MXCMSZ,COMLEN,WD,WDLEN)
     CALL ADDSTA(COMLYN,MXCMSZ,COMLEN,' READ CARD')
     CALL OPNLGU(COMLYN,COMLEN,IDUMMY)
     IF(IDUMMY < 0) GOTO 999 ! VO string
  ENDIF
  IF(UNIT <= UN1.OR.UNIT > UN999) THEN
     CALL WRNDIE(0,'<PUSTRM>','Illegal logical unit')
     IDUMMY=-1 ; GOTO 999 ! VO string
  ENDIF
  IF (.NOT.(INDXA(COMLYN,COMLEN,'NORE') > 0)) THEN
     IF (reallow) THEN
        REWIND UNIT
     ENDIF
  ENDIF
  !
  IF (NSTRM >= MXSTRM) THEN
     CALL WRNDIE(0,'<PUSTRM>', &
          'STACK OVERFLOW IN STREAM INPUT REQUEST')
     IDUMMY=-1 ! VO stringm
  ENDIF
!
  999 continue  ! VO stringm
!
#if KEY_MULTICOM==1 /* (mcom)   VO string v */
! switch to the new stream and update communicator, if applicable
  endif ! comm_parser
! broadcast of error flag idummy:
  call MPI_BCAST(idummy,1,MPI_INTEGER,0,comm_parser,mpi_err)
  if (idummy<0) return
  ! make sure all ranks have the same unit (not strictly required)
  call MPI_BCAST(unit,1,MPI_INTEGER,0,comm_parser,mpi_err)
!
  comm_strm(nstrm) = MPI_COMM_PARSER                      ! store current communicator
! switch to new communicator
  MPI_COMM_PARSER = comm_parser
  ME_PARSER=me_comm_parser
  SIZE_PARSER=size_comm_parser
!
#else /* (mcom) */
  if (idummy<0) return
#endif /*(mcom)  VO stringm ^ */
  !
  NSTRM=NSTRM+1
  ISTRM=UNIT
  IF(PRNLEV >= 2) WRITE(OUTU,65) UNIT
65 FORMAT(/20X,'INPUT STREAM SWITCHING TO UNIT ',I5)
#if KEY_MULTICOM==1 /*  VO string v */
  if (qcomm.and.prnlev.ge.2) write(outu,66) comm_name(1:l)
66 FORMAT(' PUSTRM>',' SETTING PARSER COMMUNICATOR TO "',A,'".')
#endif /* VO string ^ */
  CALL RDTITL(TITLEB,NTITLB,ISTRM,0)
  JSTRM(NSTRM)=ISTRM

  RETURN
END SUBROUTINE PUSTRM

SUBROUTINE PPSTRM(OK)
  !-----------------------------------------------------------------------
  !     This routine causes the input stream to be popped.
  !     Pop the stream stack.
  !
  !     Author: Bernie Brooks
  !
  use chm_kinds
  use stream

  ! VO string v
#if KEY_MULTICOM==1
  use multicom_aux, only : MPI_COMM_PARSER, SIZE_PARSER, ME_PARSER 
#endif
#if KEY_MULTICOM==1
  use mpi                                                          
#endif
  use parallel,only: psnd4
#if KEY_REPDSTR==1
  use repdstr, only: qrepdstr, qrdqtt
  use parallel, only: comm_charmm
  use mpi
#endif
  ! VO string ^

  implicit none
  LOGICAL OK, CLOSERR
  integer ierr
#if KEY_MULTICOM==1
  integer :: bug                                                   
#endif
  !
# if KEY_REPDSTR==1
  ! If using repd and we are inside a stream and this is
  ! the final stream, indicate we will no longer be inside a stream.
  if (qrepdstr) then
    if (qrdqtt.and.nstrm.eq.2) qrdqtt = .false.
    ! Since nstrm is never modified on nodes with iolev < 0, send result
    ! to other threads
    call mpi_bcast(qrdqtt, 1, MPI_LOGICAL, 0, comm_charmm, ierr)
  endif
# endif
#if KEY_MULTICOM==0 /* (mcom)    VO string v */
  !
  IF(IOLEV < 0) THEN
#if KEY_PARALLEL==1
     ! The psnd4 call here are so iolev < 0 processes stay in sync with
     ! the other processes (see psnd4 call below).
     CALL PSND4(OK,1) 
#endif
     RETURN
  ENDIF
#endif /*(mcom)           VO string ^ */

  IF (NSTRM <= 1) THEN
     OK=.FALSE.
  ELSE
     !     close the current stream file
     !     mikem 11/13/91: it's a question, if we really can use a regular
     !     CLOSE stmt, or have to call VCLOSE for that
     !     Right now it seems to me that we have to call VCLOSE
     !     but we may want to reconsider this later
     !     IF(ISTRM /= OUTU .AND. ISTRM.NE.5) CLOSE (UNIT=ISTRM)
#if KEY_MULTICOM==1 /*  VO stringm */
     if (ME_PARSER.eq.0) then                                     
#endif
     IF (ISTRM /= OUTU) CALL VCLOSE(JSTRM(NSTRM),'KEEP',CLOSERR)
     IF (CLOSERR) CALL WRNDIE(0,'<PPSTRM>', &
          'Problems occured during closing unit')
#if KEY_MULTICOM==1 /*  VO stringm */
     endif                                                        
#endif
     NSTRM=NSTRM-1
     ISTRM=JSTRM(NSTRM)
     !
#if KEY_MULTICOM==1 /* (mcom) */
     ! VO string :  restore old communicator (assuming it is still valid)
     MPI_COMM_PARSER=comm_strm(nstrm)
     call MPI_COMM_RANK(MPI_COMM_PARSER, ME_PARSER, bug)  ! fetch new rank & size
     call MPI_COMM_SIZE(MPI_COMM_PARSER, SIZE_PARSER, bug)
     ! VO string
#endif /*(mcom) */
     !
     IF(PRNLEV >= 2) WRITE(OUTU,85) ISTRM
     OK=.TRUE.
  ENDIF

#if KEY_PARALLEL==1 && KEY_MULTICOM==0
  call psnd4(ok,1)
#endif

85 FORMAT(/20X,'RETURNING TO INPUT STREAM ',I5)

  return
end subroutine ppstrm

SUBROUTINE CHKLIM
  !-----------------------------------------------------------------------
  !     Returns TRUE in ATLIM if either the CPU-time limit or the clocktim
  !     deadline, as given in the CHARMM DEADline command, has been
  !     reached. The return variable LIMTYP also tells you which limit
  !     it was: LIMTYP='NONE','CPU ','CLK ', or 'BOTH'
  !
  use chm_kinds
  use dimens_fcm
  use fast
  use stream
  use timerm
  use machutil,only:jobdat,daytim

  implicit none

  real(chm_real)  DELTA,ELATIM
  INTEGER DAY,MONTH,YEAR,HOUR,MINUTE,SECOND
  INTEGER IODIR,NPFLTS,IOBUF
  !
  ATLIM=.FALSE.
  LIMTYP='NONE'
  IF(CPULIM > 0.0) THEN
     CALL JOBDAT(ELATIM,DELTA,IOBUF,IODIR,NPFLTS)
     IF(DELTA-CPUINI >= CPULIM) THEN
        ATLIM=.TRUE.
        LIMTYP='CPU '
     ENDIF
  ENDIF
  !
  IF(DEADHR >= 0) THEN
     CALL DAYTIM(MONTH,DAY,YEAR,HOUR,MINUTE,SECOND)
     IF(PASMID == 0) PASMID=DAY
     DELTA=24.0*(DAY-PASMID)+(HOUR-DEADHR)+(MINUTE-DEADMN)/60.0
     IF(DELTA > 0.0) THEN
        !
        IF(PRNLEV >= 2) WRITE(OUTU,'(A,/,6I8)') &
             ' DAY     ,PASMID  ,HOUR    ,DEADHR  ,MINUTE  ,DEADMN', &
             DAY,PASMID,HOUR,DEADHR,MINUTE,DEADMN
        IF(PRNLEV >= 2) WRITE(OUTU,'(A,/,4I8)') &
             ' TIMER   ,WRNLEV  ,BOMLEV  ,FASTER', &
             TIMER,WRNLEV,BOMLEV,FASTER
        !
        ATLIM=.TRUE.
        !
        IF (LIMTYP == 'CPU') THEN
           LIMTYP='BOTH'
        ELSE
           LIMTYP='CLK '
        ENDIF
        !
     ENDIF
  ENDIF
  RETURN
END SUBROUTINE CHKLIM

INTEGER FUNCTION LUNASS(IUNIT)
  !-----------------------------------------------------------------------
  !     Simple function to find the first free to use logical unit number
  !     starting with IUNIT and going up.....
  !
  !     Lennart Nilsson, NOV-87
  !
  INTEGER IUNIT,LUN
  LOGICAL QOPEN
  !
  character(len=80) NAME
  LOGICAL QFORM,QWRITE
  INTEGER MAXLEN, LENGTH
  PARAMETER (MAXLEN=80)
  !
  LUN=IUNIT-1
10 CONTINUE
  LUN=LUN+1

  name = ' '  ! initialize name variable as blank
  CALL VINQRE('UNIT',NAME,MAXLEN,LENGTH,QOPEN,QFORM,QWRITE,LUN)
  IF (QOPEN) GOTO 10
  LUNASS=LUN
  !
  RETURN
END FUNCTION LUNASS

