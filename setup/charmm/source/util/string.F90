module string
  use chm_kinds
  use dimens_fcm

  implicit none

  !    TEMPORARY STRING MANIPULATION VARIABLES
  !     NOTE: SCRMAX must match MXCMSZ of COMAND.FCM
  ! extended to include commonly used character codes (ICHAR) (see below)

  integer scrlen,swdlen,fmtlen
  !     The parameter SCRMAX must be reduced for some machines (IBM).
  !     It may also be reduced to save memory on non-virtual machines.

  integer, parameter :: swdmax=20, fmtmax=20
  character(len=swdmax) swdtch
  character(len=fmtmax) fmtwd
  character(len=scrmax) scrtch
  character(len=4) ffour

  integer :: gtrmim_err = 0    ! error flag for gtrmim() routine

  ! some decimal character codes useful in string routines
  ! initialized in main/iniall.src
  ! there is a fudge for the TAB character, as I cannot place an explicit
  ! tab in the source to get the character code. 
  ! the character code for TAB is explicitly checked for both ASCII and EBCDIC.
  !
  ! LC -lower case, UC - uppercase
  integer :: ichtab
  integer, parameter :: ICHQUO = ichar('"'), &
       ICHZERO = ichar('0'), ICHNINE = ichar('9'), &
       ICHUCA = ichar('A'), ICHUCZ = ichar('Z'), &
       ICHLCA = ichar('a'), ICHLCZ = ichar('z')
  character(len=1), parameter, private :: BLANK = ' '
  !
  ! CHARACTER(len=4) functions
  !    GTRMA  - Get and remove keyword string from the command line.
  !    NEXTA4 - Get and remove the next 4-character word from the line.
  !    CURRA4 - Same as NEXTA4 except don't remove the word.
  ! CHARACTER(len=6) functions
  !    NEXTA6 - Get and remove the next 6-character word from the line.
  ! CHARACTER(len=8) functions
  !    NEXTA8 - Get and remove the next 8-character word from the line.
  ! CHARACTER(len=20) functions
  !    NEXT20 - Get and remove the next 20-character word from the line.
  ! INTEGER functions
  !    DECODI - Get an integer value from a string
  !    GTRMI  - Get and remove keyword integer value from the command line.
  !    INDX   - search a line for an abbreviated substring
  !    INDXA  - search a line for a substring and remove the substring
  !    INDXAF - utility string function (use INDXRA instead)
  !    INDXRA - utility string function (use INDXA instead)
  !    NEXTI  - Get and remove the next integer value from a line.
  !    STRLNG - Find the length of a string (non-blank)
  ! LOGICAL functions
  !    CHECQUE- verifies if word2 is in word1 and removes it
  !    DOTRIM - trim a string
  !    EQST   - are two stings equal?
  !    EQSTA  - are two abbreviated strings equal?
  !    EQSTWC - are two strings equal with wildcarding?
  !    EQWDWC - utility string function (use EQSTWC instead)
  !    LTSTEQ - Is the alphabetic order of two strings correct?
  !    QDIGIT - Is a character a digit?
  !    QALPHA - Is a character an alphabetic?
  ! real(chm_real) functions
  !    DECODF - Get the R*8 value from a string
  !    GTRMF  - Get and remove keyword real value from the command line
  !    NEXTF  - Get and remove the next real value from the command line.
  !

  interface GTRMF
    module procedure gtrmf_crl, gtrmf_alt
  end interface GTRMF
!
!  VO additional aux. functions used stringm and pnm
  interface itoa
   module procedure itoa4, itoa8
  end interface itoa
!

contains
  subroutine string_iniall()
    fmtlen=0
    ! set correct tab code
    if      (ichlca == 97) then
       ! ascii tab
       ichtab=9
    else if (ichlca == 129) then
       ! ebcdic tab
       ichtab=5
    else
       CALL WrnDie(0,'string.src<string_iniall>', &
            'ASCII/EBCDIC coding unknown. TAB handling may be incorrect')
       ichtab=9
    endif
    return
  end subroutine string_iniall

  subroutine prntst(iunit,st,stlen,nover,ncol)
    !-----------------------------------------------------------------------
    !     Prints a string to a specified unit.
    !         nover gives the number of preceeding blanks to print
    !         ncol gives the number of columns
    !     B R Brooks - 4/83

    use stream

    integer iunit,stlen,nover,ncol
    character(len=*) st
    integer istrt,istp,nact,i

    if(prnlev <= 1) return

    nact=ncol-nover
    if(ncol <= 0) nact=stlen
    istp=0
    do
       istrt=istp+1
       istp=istp+nact
       if(istp > stlen) istp=stlen
       write(iunit,20) (BLANK,i=1,nover),(st(i:i),i=istrt,istp)
       if (istp == stlen) exit
    enddo
20  format(132a1)
    return
  end subroutine prntst

  !
  !     THIS GROUP OF STRING COMMANDS DEALS WITH THE COPYING OF STRINGS
  !
  SUBROUTINE COPYST(ST1,ST1SIZ,ST1LEN,ST2,ST2LEN)
    !-----------------------------------------------------------------------
    !     COPYS ST2 INTO ST1.
    !
    !     Author: Robert Bruccoleri
    !
    INTEGER ST1SIZ,ST1LEN,ST2LEN
    CHARACTER(len=*) ST1,ST2
    !
    CALL COPSUB(ST1,ST1SIZ,ST1LEN,ST2,1,ST2LEN)
    RETURN
  END SUBROUTINE COPYST

  SUBROUTINE COPSUB(ST1,ST1SIZ,ST1LEN,ST2,START,ISTP)
    !-----------------------------------------------------------------------
    !     Copys substring ST2(START:ISTP) into ST1.
    !
    !     Axel Brunger, 26-MAR-83
    !
    implicit none
    CHARACTER(len=*), intent(inout) :: ST1 ! substr copied here 1:n
    INTEGER, intent(in) :: ST1SIZ ! st1 capacity
    integer, intent(out) :: ST1LEN ! min of substr len and st1 capacity
    CHARACTER(len=*), intent(in) :: ST2 ! substr is in st2(start:istp)
    INTEGER, intent(in) :: &
         START, & ! substr start idx
         ISTP ! substr final idx inclusive

    INTEGER :: &
         LIM, & ! length of new string, may be trunc b/c of st1siz
         substring_size ! size of original substr requested

    character(len=len(st1)) buffer ! in case st1 = st2, cant memcpy st1 <- st2

    substring_size = istp - start + 1

    LIM=0
    IF (substring_size > 0 ) THEN
       LIM=MIN(ST1SIZ, substring_size)
       IF(LIM <= 0 .or. start < 0 ) THEN
          ST1=' '
          ST1LEN=0
          RETURN
       ENDIF
       buffer = ST2(START:ISTP)
       ST1(1:lim) = buffer(1:lim) 
       IF(substring_size > ST1SIZ) THEN
          CALL WRNDIE(1,'<COPSUB>','DESTINATION STRING TOO SMALL')
       ENDIF
    ENDIF
    ST1LEN=LIM ! zero is copy cant be done
    RETURN
  END SUBROUTINE COPSUB

  SUBROUTINE CPYSPC(ST,STSIZ,STLEN,NBLANK)
    !-----------------------------------------------------------------------
    !     ADDS NBLANK SPACES TO ST
    !
    !     Author: Robert Bruccoleri
    !
    INTEGER STSIZ,STLEN,NBLANK
    CHARACTER(len=*) ST
    INTEGER LIM
    !
    IF(NBLANK <= 0) RETURN
    IF(STLEN+NBLANK > STSIZ) THEN
       CALL WRNDIE(1,'<CPYSPC>','STRING TOO SMALL')
    ENDIF
    LIM=MIN(STSIZ,STLEN+NBLANK)
    CALL FILSPC(ST,LIM,STLEN)
    STLEN=LIM
    RETURN
  END SUBROUTINE CPYSPC

  SUBROUTINE FILSPC(ST,STMAX,STLEN)
    !-----------------------------------------------------------------------
    !     THIS ROUTINE FILLS THE UNUSED PART OF ST WITH SPACES.
    !
    !     Author: Robert Bruccoleri
    !
    INTEGER STMAX,STLEN
    CHARACTER(len=*) ST
    !
    IF (STLEN > 0) THEN
       IF(STMAX > STLEN) ST(STLEN+1:STMAX)=' '
    ELSE
       ST=' '
    ENDIF
    RETURN
  END SUBROUTINE FILSPC

  !
  !     THIS GROUP OF STRING COMMANDS DEALS WITH JOINING STRINGS TOGETHER
  !
  SUBROUTINE ADDST(ST,STMAX,STLEN,ADST,ADLEN)
    !-----------------------------------------------------------------------
    !     CONCATENATES ADST ONTO ST
    !
    !     Author: Robert Bruccoleri
    !
    INTEGER STMAX,STLEN,ADLEN,LIM
    CHARACTER(len=*) ST,ADST
    !
    LIM=ADLEN+STLEN
    IF(LIM > STMAX) THEN
       CALL WRNDIE(1,'<ADDST> ','TRUNCATION HAS OCCOURRED')
       LIM=STMAX
    ENDIF
    IF(LIM <= STLEN) RETURN
    ST(STLEN+1:LIM)=ADST(1:LIM-STLEN)
    STLEN=LIM
    RETURN
  END SUBROUTINE ADDST

  SUBROUTINE ADDSTA(ST,STMAX,STLEN,KEYWRD)
    !-----------------------------------------------------------------------
    !     CONCATENATES ADST ONTO ST
    !
    !     Author: Robert Bruccoleri
    !
    CHARACTER(len=*) ST,KEYWRD
    INTEGER STLEN,STMAX
    !
    CALL ADDST(ST,STMAX,STLEN,KEYWRD,LEN(KEYWRD))
    !
    RETURN
  END SUBROUTINE ADDSTA

  SUBROUTINE JOINWD(ST,STMAX,STLEN,WD,WDLEN)
    !-----------------------------------------------------------------------
    !     DOES THE INVERSE OF NEXTWD, I.E. JOINDS WD ONTO THE BEGINNING OF
    !     ST.
    !
    INTEGER STMAX,STLEN,WDLEN
    CHARACTER(len=*) ST,WD
    !
    IF(WDLEN <= 0) RETURN
    IF(STLEN+WDLEN >= STMAX) THEN
       CALL WRNDIE(-2,'<JOINWD>','STRING TOO SMALL')
    ENDIF
    IF (STLEN > 0) THEN
       SCRTCH=WD(1:WDLEN)//' '//ST(1:STLEN)
       ST=SCRTCH
       STLEN=WDLEN+1+STLEN
    ELSE
       ST=WD(1:WDLEN)
       STLEN=WDLEN
    ENDIF
    RETURN
  END SUBROUTINE JOINWD

  !
  !     THIS GROUP OF STRING COMMANDS DEALS WITH SEARCHING
  !
  LOGICAL FUNCTION CHECQUE(C1,C2)
    !-----------------------------------------------------------------------
    !     This subroutine verifies if c2 is in c1 and removes it
    !yw...04-Aug-93 from RISM
    !
    CHARACTER(len=*) C1,C2
    LOGICAL JUNK
    INTEGER I1,I2,LENC1,LENC2
    !
    I1=INDEX(C1,C2)
    CHECQUE=I1 /= 0
    IF(CHECQUE)THEN
       LENC1=LEN(C1)
       LENC2=LEN(C2)
       I2=INDEX(C1(I1+LENC2:LENC1),' ')+I1+LENC2-1
       !
       !     Remove the word until the next ' '
       !  
       SCRTCH=C1
       C1(I1:LENC1)=SCRTCH(I2:LENC1)
       JUNK=DOTRIM(C1)
    ENDIF
    !
    RETURN
  END FUNCTION CHECQUE

  INTEGER FUNCTION INDX(ST,STLEN,TARGET,TARLEN)
    !-----------------------------------------------------------------------
    !     THIS ROUTINE PERFORMS THE INDEX FUNCTION ON STRINGS WHICH ARE
    !     STORED IN CHARACTER ARRAYS.
    !
    !     Author: Robert Bruccoleri
    !
    INTEGER STLEN,TARLEN
    CHARACTER(len=*) ST,TARGET
    INTEGER LIM,I
    !
    INDX=0
    IF (STLEN <= 0) RETURN
    IF(TARLEN > STLEN) RETURN
    IF (TARLEN <= 0) THEN
       INDX=1
       RETURN
    ENDIF
    LIM=STLEN-TARLEN+1
    DO I=1,LIM
       IF(ST(I:I+TARLEN-1) == TARGET(1:TARLEN)) THEN
          INDX=I
          RETURN
       ENDIF
    ENDDO
    RETURN
  END FUNCTION INDX

  !
  !     THIS GROUP OF STRING COMMANDS DEALS WITH PICKING OFF THE FIRST WORD
  !
  SUBROUTINE NEXTWD(ST,STLEN,WD,WDMAX,WDLEN)
    !-----------------------------------------------------------------------
    !     THIS ROUTINE WILL REMOVE A WORD FROM ST AND PLACE IT
    !     IN WD. A WORD IS DEFINED AS A SEQUENCE OF NON-BLANK CHARACTERS.
    !
    !     Author: Robert Bruccoleri
    !
    INTEGER STLEN,WDMAX,WDLEN
    CHARACTER(len=*) ST,WD
    INTEGER IND,IPT,IPT2
    !
    !     FIRST LOOK PAST ANY LEADING BLANKS.
    !
    IND=0
1   IND=IND+1
    IF(IND > STLEN) THEN
       WDLEN=0
       STLEN=0
       ST=BLANK
       WD=BLANK
       RETURN
    ENDIF
    IF(ST(IND:IND) == BLANK) GOTO 1
    !
    !     NOW GET THE WORD
    !
    IPT=IND
2   IPT=IPT+1
    IF(IPT > STLEN) THEN
       STLEN=0
       WDLEN=IPT-IND
       IF(WDLEN > WDMAX) THEN
          WDLEN=WDMAX
          IPT=WDMAX+IND
       ENDIF
       WD=ST(IND:IPT-1)
       ST=BLANK
       RETURN
    ENDIF
    IF(ST(IPT:IPT) /= BLANK) GOTO 2
    !
    !     NOW WE HAVE THE WORD, DELETE IT FROM ST
    !
    IPT2=IPT
    WDLEN=IPT-IND
    IF(WDLEN > WDMAX) THEN
       WDLEN=WDMAX
       IPT=WDMAX+IND
    ENDIF
    WD=ST(IND:IPT-1)
    SCRTCH=ST(IPT2:STLEN)
    ST=SCRTCH
    STLEN=STLEN-IPT2+1
    RETURN
  END SUBROUTINE NEXTWD
  
    SUBROUTINE NEXTWD100(ST,STLEN,WD,WDLEN)
    !-----------------------------------------------------------------------
    !   As NEXTWD, this subroutine will move a word from ST to WD 
    !   A word is defined as a sequence of non-blank characters.
    !   This subroutine is a modificatio of the original NEXTWD from Robert 
    !   Bruccoleri, under the guidance of  Joshua Buckner.
    !   A difference from the original routine is that the WDMAX is hardcoded 
    !   in the subroutine as WORLD_MAX=100 
    !   
    !    Pezzella 05.06.2019
    !-----------------------------------------------------------------------
     
    implicit none

    !-----------------------------------------------------------------------
    !                   Variables
    ! -> Parsed:
    ! st: string to divide 
    ! wd: word to save 
    ! stlen and wdlen: length of st and wd respectively 
    ! -> Local 
    ! word_max: parameter that set the maximum lenght of the word
    ! blamk_i: number of blank characters. 
    ! temp_st and temp_len : temporary variables, equivalent to st and stlen
    !
    !-----------------------------------------------------------------------

    character(len=*), intent(inout) :: st, wd
    integer, intent(inout) :: stlen, wdlen
    integer, parameter :: word_max = 100
    integer :: blank_i, temp_len
    character(len=stlen) :: temp_str
    character(len=word_max) :: word

    blank_i = 0
    temp_len = 0

    ! remove leading and trailing whitespace
    temp_str = adjustl(trim(st))
    temp_len = len(temp_str)
    if (temp_len == 0) then ! no words, nothing to do
       wd = ' '
       wdlen = 0
       st = ' '
       stlen = 0
       call wrndie(5,'<util>','NEXTWD100, the line parsed is empty.')
       return
    end if

    ! find the first blank and clean them.     
    blank_i = index(temp_str, ' ')
    ! no blanks, copy whole string
    if (blank_i == 0) then 
       wdlen = len(temp_str)
    else
       wdlen = blank_i - 1
    end if

    ! case wdlen exceeds world_max it will be resized.
    if (wdlen > word_max) wdlen = word_max
    
    ! assign the word to the WD parameter
    wd = temp_str(1:wdlen)
 
    ! removed the newly stored word from the string ST
    ! length(wd) <  length(st)
    if (wdlen .lt. stlen) then 
       st = temp_str((wdlen + 1):stlen)
       stlen = len(trim(st))
    else ! st is only one word
       st = ' '
       stlen = 0
    end if


    RETURN
    END SUBROUTINE NEXTWD100
  
  
  

  CHARACTER(len=4) FUNCTION CURRA4(ST,STLEN)
    !-----------------------------------------------------------------------
    !     This function will find the first word from ST.
    !     This routine does not delete the current word from ST.
    !
    INTEGER STLEN
    CHARACTER(len=*) ST
    INTEGER IND,IPT
    !
    !     FIRST LOOK PAST ANY LEADING BLANKS.
    !
    CURRA4='    '
    IND=0
1   IND=IND+1
    IF(IND > STLEN) THEN
       RETURN
    ENDIF
    IF(ST(IND:IND) == ' ') GOTO 1
    !
    !     NOW GET THE WORD
    !
    IPT=IND
2   IPT=IPT+1
    IF(IPT > STLEN) GOTO 3
    IF(ST(IPT:IPT) /= ' ' .AND. IPT-IND < 4) GOTO 2
    !
3   CURRA4=ST(IND:IPT-1)
    RETURN
  END FUNCTION CURRA4

  CHARACTER(len=8) FUNCTION CURRA8(ST,STLEN)
    !-----------------------------------------------------------------------
    !     This function will find the first word from ST.
    !     This routine does not delete the current word from ST.
    !
    INTEGER STLEN
    CHARACTER(len=*) ST
    INTEGER IND,IPT
    !
    !     FIRST LOOK PAST ANY LEADING BLANKS.
    !
    CURRA8='        '
    IND=0
1   IND=IND+1
    IF(IND > STLEN) THEN
       RETURN
    ENDIF
    IF(ST(IND:IND) == ' ') GOTO 1
    !
    !     NOW GET THE WORD
    !
    IPT=IND
2   IPT=IPT+1
    IF(IPT > STLEN) GOTO 3
    IF(ST(IPT:IPT) /= ' ' .AND. IPT-IND < 8) GOTO 2
    !
3   CURRA8=ST(IND:IPT-1)
    RETURN
  END FUNCTION CURRA8

  SUBROUTINE NEXTOP(ST,START,STLEN,WD,WDMAX,WDLEN)
    !-----------------------------------------------------------------------
    !     This routine will remove an opword from ST and place it
    !     in WD. An opword is defined as a sequence of non-blank
    !     characters or one of the following symbols ':','(',')'.
    !     The search is started at character index START.
    !
    !     Axel Brunger, 25-MAR-83
    !     overhauled 2-JUL-84 - BRB
    !
    CHARACTER(len=*) ST,WD
    INTEGER   START, STLEN, WDMAX, WDLEN
    !
    INTEGER   IND, IPT
    LOGICAL   CONDIT
    CHARACTER(len=1) STEMP
    !
    WD=' '
    WDLEN=0
    IF (STLEN < START) RETURN
    !
    IND=START-1
10  CONTINUE
    IND=IND+1
    IF (.NOT.(ST(IND:IND) /= ' '.OR.IND >= STLEN)) GOTO 10
    STEMP=ST(IND:IND)
    IF (STEMP == ' ') THEN
       STLEN=START-1
       RETURN
    ENDIF
    !
    IPT=IND
30  CONTINUE
    CONDIT=(STEMP == ')'.OR.STEMP.EQ.'('.OR.STEMP.EQ.':' &
         .OR.IND >= STLEN)
    IND=IND+1
    IF (.NOT.CONDIT) THEN
       STEMP=ST(IND:IND)
       CONDIT=(STEMP == ' '.OR.STEMP.EQ.')'.OR.STEMP.EQ.'(' &
            .OR.STEMP == ':')
    ENDIF
    IF (.NOT.(CONDIT)) GOTO 30
    WD=ST(IPT:IND-1)
    WDLEN=MIN(IND-IPT,WDMAX)
    !
    IF (START > 1) THEN
       IF (STLEN >= IND) THEN
          SCRTCH=ST(1:START-1)//ST(IND:STLEN)
       ELSE
          SCRTCH=ST(1:START-1)
       ENDIF
    ELSE
       IF (STLEN >= IND) THEN
          SCRTCH=ST(IND:STLEN)
       ELSE
          SCRTCH=' '
       ENDIF
    ENDIF
    !
    ST=SCRTCH
    STLEN=STLEN-IND+START
    RETURN
  END SUBROUTINE NEXTOP

  FUNCTION NEXTF(ST,STLEN) result(nextff)
    !-----------------------------------------------------------------------
    !     Reads the next word off of ST using NEXTWD, converts it to a
    !     floating point number, and returns it.
    !
    !     Author: Robert Bruccoleri
    !
    INTEGER STLEN
    CHARACTER(len=*) ST
    real(chm_real) nextff

    CALL NEXTWD(ST,STLEN,SWDTCH,SWDMAX,SWDLEN)
    NEXTFf=DECODF(SWDTCH,SWDLEN)
    RETURN
  END FUNCTION NEXTF

  INTEGER FUNCTION NEXTI(ST,STLEN)
    !-----------------------------------------------------------------------
    !     Reads the next word off of ST using NEXTWD, converts it to an
    !     integer, and returns it.
    !
    !     Author: Robert Bruccoleri
    !
    INTEGER STLEN
    CHARACTER(len=*) ST

    CALL NEXTWD(ST,STLEN,SWDTCH,SWDMAX,SWDLEN)
    NEXTI=DECODI(SWDTCH,SWDLEN)
    RETURN
  END FUNCTION NEXTI

  CHARACTER(len=4) FUNCTION NEXTA4(ST,STLEN)
    !-----------------------------------------------------------------------
    !     Gets the next word off of ST into a four byte word. The word will
    !     be blank filled. Declare this function as needed so that Fortran
    !     doesn't do any type conversion on assignment.
    !
    !     Author: Robert Bruccoleri
    !
    INTEGER STLEN
    CHARACTER(len=*) ST
    INTEGER L
    !
    NEXTA4='    '
    L=LEN(NEXTA4)
    CALL NEXTWD(ST,STLEN,SWDTCH,L,SWDLEN)
    IF (SWDLEN > 0) THEN
       NEXTA4=SWDTCH(1:SWDLEN)
    ENDIF
    RETURN
  END FUNCTION NEXTA4

  CHARACTER(len=6) FUNCTION NEXTA6(ST,STLEN)
    !-----------------------------------------------------------------------
    !     Gets the next word off of ST into a six byte word. The word will
    !     be blank filled. Declare this function as needed so that Fortran
    !     doesn't do any type conversion on assignment.
    !
    !     Author: Robert Bruccoleri
    !
    INTEGER STLEN
    CHARACTER(len=*) ST
    INTEGER L
    !
    NEXTA6='      '
    L=LEN(NEXTA6)
    CALL NEXTWD(ST,STLEN,SWDTCH,L,SWDLEN)
    IF (SWDLEN > 0) THEN
       NEXTA6=SWDTCH(1:SWDLEN)
    ENDIF
    RETURN
  END FUNCTION NEXTA6

  CHARACTER(len=8) FUNCTION NEXTA8(ST,STLEN)
    !-----------------------------------------------------------------------
    !     Gets the next word off of ST into a six byte word. The word will
    !     be blank filled. Declare this function as needed so that Fortran
    !     doesn't do any type conversion on assignment.
    !
    !     Author: Robert Bruccoleri
    !
    INTEGER STLEN
    CHARACTER(len=*) ST
    INTEGER L
    !
    NEXTA8='        '
    L=LEN(NEXTA8)
    CALL NEXTWD(ST,STLEN,SWDTCH,L,SWDLEN)
    IF (SWDLEN > 0) THEN
       NEXTA8=SWDTCH(1:SWDLEN)
    ENDIF
    RETURN
  END FUNCTION NEXTA8

  ! I-Jen Chen started. 12/28/99
  !
  CHARACTER(len=20) FUNCTION NEXT20(ST,STLEN)
    !----------------------------------------------------------------------
    !     Gets the next word off of ST into a six byte word. The word will
    !     be blank filled. Declare this function as needed so that Fortran
    !     doesn't do any type conversion on assignment.
    !
    !     Author: Robert Bruccoleri
    !
    INTEGER STLEN
    CHARACTER(len=*) ST
    INTEGER L
    !
    NEXT20='                    '
    L=LEN(NEXT20)
    CALL NEXTWD(ST,STLEN,SWDTCH,L,SWDLEN)
    IF (SWDLEN > 0) THEN
       NEXT20=SWDTCH(1:SWDLEN)
    ENDIF
    RETURN
  END FUNCTION NEXT20

  ! I-Jen left.

  CHARACTER(len=100) FUNCTION NEXT100(ST,STLEN) !marcpezz 11/04/2019
    !----------------------------------------------------------------------
    !     Gets the next word off of ST into a hundred  byte word. The word will
    !     be blank filled. Declare this function as needed so that Fortran
    !     doesn't do any type conversion on assignment.
    !     SWDTCH is substituted by SWDTCH100, that has dimension 100.
    !      
    !      Pezzella 15.05.2019
    INTEGER STLEN
    CHARACTER(len=*) ST
    character(len=100) :: SWDTCH100
    INTEGER L
    !
    NEXT100='                                                                                                    '
    L=LEN(NEXT100)
    CALL NEXTWD100(ST,STLEN,SWDTCH100,SWDLEN)
    IF (SWDLEN > 0) THEN
       NEXT100=SWDTCH100(1:SWDLEN)
    ENDIF
    RETURN
  END FUNCTION NEXT100

  SUBROUTINE NXTWDA(ST,STLEN,WD,WDMAX,WDLEN,START)
    !-----------------------------------------------------------------------
    !     This routine will remove a word from ST and place it
    !     in WD. A word is defined as a sequence of non-blank
    !     characters. The search is started at character index START.
    !
    !     Axel Brunger, 25-MAR-83
    !     overhauled 2-JUL-84 - BRB
    !
    CHARACTER(len=*) ST,WD
    INTEGER   START, STLEN, WDLEN
    INTEGER , intent(in) ::  WDMAX
    !
    INTEGER   IND, IPT
    LOGICAL   CONDIT
    CHARACTER(len=1) STEMP
    !
    WD=' '
    WDLEN=0
    IF (STLEN < START) RETURN
    !
    IND=START-1
10  CONTINUE
    IND=IND+1
    IF (.NOT.(ST(IND:IND) /= ' '.OR.IND >= STLEN)) GOTO 10
    STEMP=ST(IND:IND)
    IF (STEMP == ' ') THEN
       STLEN=START-1
       RETURN
    ENDIF
    !
    IPT=IND
30  CONTINUE
    CONDIT=(IND >= STLEN)
    IND=IND+1
    IF (.NOT.CONDIT) THEN
       STEMP=ST(IND:IND)
       CONDIT=(STEMP == ' ')
    ENDIF
    IF (.NOT.(CONDIT)) GOTO 30
    WD=ST(IPT:IND-1)
    WDLEN=MIN(IND-IPT,WDMAX)
    !
    IF (START > 1) THEN
       IF (STLEN >= IND) THEN
          SCRTCH=ST(1:START-1)//ST(IND:STLEN)
       ELSE
          SCRTCH=ST(1:START-1)
       ENDIF
    ELSE
       IF (STLEN >= IND) THEN
          SCRTCH=ST(IND:STLEN)
       ELSE
          SCRTCH=' '
       ENDIF
    ENDIF
    !
    ST=SCRTCH
    STLEN=STLEN-IND+START
    RETURN
  END SUBROUTINE NXTWDA
  
  ! BIOVIA Code Start : Bug fix
  SUBROUTINE NXTWDA_SPACE(ST,STLEN,WD,WDMAX,WDLEN,START)
    !-----------------------------------------------------------------------
    !     This routine will remove a word from ST and place it
    !     in WD. A word is defined as a sequence of non-blank
    !     characters. The search is started at character index START.
    !
    !     Axel Brunger, 25-MAR-83
    !     overhauled 2-JUL-84 - BRB
    !
    use chm_kinds
    use dimens_fcm
    !
    CHARACTER*(*) ST,WD
    INTEGER   START, STLEN, WDMAX, WDLEN
    !
    INTEGER   IND, IPT
    LOGICAL   CONDIT
    CHARACTER*1 STEMP
    LOGICAL DQUOTE
    !
    DQUOTE=.FALSE.
    WD=' '
    WDLEN=0
    IF (STLEN.LT.START) RETURN
    !
    IND=START-1
10  CONTINUE
    IND=IND+1
    IF (.NOT.(ST(IND:IND).NE.' '.OR.IND.GE.STLEN)) GOTO 10
    STEMP=ST(IND:IND)
    IF (STEMP.EQ.' ') THEN
       STLEN=START-1
       RETURN
    ENDIF
    !
    IF(STEMP.EQ.'"') DQUOTE=.TRUE.
    IPT=IND
30  CONTINUE
    CONDIT=(IND.GE.STLEN)
    IND=IND+1
    IF (.NOT.CONDIT) THEN
       STEMP=ST(IND:IND)
       IF(.NOT.DQUOTE) THEN
          CONDIT=(STEMP.EQ.' ')
       ELSE
          CONDIT=(STEMP.EQ.'"')
          if (condit) ind = ind + 1
       ENDIF
    ENDIF
    IF (.NOT.(CONDIT)) GOTO 30
       ! IF(ST(IPT:IPT).EQ.'"') THEN
       !    IPT=IPT+1
       ! ENDIF
       WD=ST(IPT:IND-1)
       WDLEN=MIN(IND-IPT,WDMAX)
       !
       IF (START.GT.1) THEN
          IF (STLEN.GE.IND) THEN
             ! IF(ST(IND:IND).EQ.'"') THEN
             !    SCRTCH=ST(1:START-1)//ST(IND+1:STLEN)
             ! ELSE
                SCRTCH=ST(1:START-1)//ST(IND:STLEN)
             ! ENDIF
          ELSE
             SCRTCH=ST(1:START-1)
          ENDIF
       ELSE
       IF (STLEN.GE.IND) THEN
          ! IF(ST(IND:IND).EQ.'"') THEN
          !    SCRTCH=ST(IND+1:STLEN)
          ! ELSE
             SCRTCH=ST(IND:STLEN)
          ! ENDIF
       ELSE
          SCRTCH=' '
       ENDIF
    ENDIF
    !
    IF(ST(IND:IND).EQ.'"') THEN
       STLEN=STLEN-IND+START-1
    ELSE
       STLEN=STLEN-IND+START
    ENDIF
    ST=SCRTCH
    RETURN
  END SUBROUTINE NXTWDA_SPACE
  ! BIOVIA Code End
  
  !
  !     THIS GROUP OF STRING COMMANDS DEALS WITH PICKING OFF AN INTERNAL WORD
  !
  SUBROUTINE GTRMWD(ST,STLEN,KEY,KEYLEN,WD,WDMAX,WDLEN)
    !-----------------------------------------------------------------------
    !     THIS ROUTINE PICKS UP THE WORD AFTER THE KEY WORD KEY IN THE
    !     STRING ST AND PLACES IT INTO WD. THE KEY AND SUCCEEDING WORD ARE
    !     THEN DELETED FROM ST. THE KEY MUST BE FOLLOWED BY A BLANK SO THAT
    !     JOINED WORDS WILL NOT WORK.
    !
    INTEGER STLEN,KEYLEN,WDMAX,WDLEN,IND
    CHARACTER(len=*) ST,KEY,WD
    !
    WDLEN=0
    IND=INDXRA(ST,STLEN,KEY,KEYLEN,.FALSE.)
    IF(IND > 0) THEN
       CALL NXTWDA(ST,STLEN,WD,WDMAX,WDLEN,IND)
    ENDIF
    RETURN
  END SUBROUTINE GTRMWD
  
  ! BIOVIA Code Start : Bug fix
  SUBROUTINE GTRMWD_SPACE(ST,STLEN,KEY,KEYLEN,WD,WDMAX,WDLEN)
    !-----------------------------------------------------------------------
    !     THIS ROUTINE PICKS UP THE WORD AFTER THE KEY WORD KEY IN THE
    !     STRING ST AND PLACES IT INTO WD. THE KEY AND SUCCEEDING WORD ARE
    !     THEN DELETED FROM ST. THE KEY MUST BE FOLLOWED BY A BLANK SO THAT
    !     JOINED WORDS WILL NOT WORK.
    !
    use chm_kinds
    use exfunc
    !
    INTEGER STLEN,KEYLEN,WDMAX,WDLEN,IND
    CHARACTER*(*) ST,KEY,WD
    !
    WDLEN=0
    IND=INDXRA(ST,STLEN,KEY,KEYLEN,.FALSE.)
    IF(IND.GT.0) THEN
#if KEY_GNU==1
       CALL NXTWDA_SPACE(ST,STLEN,WD,WDMAX,WDLEN,IND)
#else
       CALL NXTWDA(ST,STLEN,WD,WDMAX,WDLEN,IND)
#endif
    ENDIF
    RETURN
  END SUBROUTINE GTRMWD_SPACE
  ! BIOVIA Code End

  SUBROUTINE GTRMWA(ST,STLEN,KEY,KEYLEN,WD,WDMAX,WDLEN)
    !-----------------------------------------------------------------------
    !     GeT and ReMove Word Abreviated
    !     Gets and removes a word from ST prefixed by the keyword KEY. This
    !     routine is similar to GTRMWD except it allows KEY to be an
    !     abbreviation.
    !     This routine should remain internal to string.flx
    !
    use exfunc
    !
    INTEGER STLEN,KEYLEN,WDMAX,WDLEN
    CHARACTER(len=*) ST,KEY,WD
    !
    INTEGER IND
    !
    WDLEN=0
    IND=INDXRA(ST,STLEN,KEY,KEYLEN,.TRUE.)
    IF(IND > 0) THEN
       CALL NXTWDA(ST,STLEN,WD,WDMAX,WDLEN,IND)
    ENDIF
    RETURN
  END SUBROUTINE GTRMWA

  SUBROUTINE GTRMWAM(M,ST,STLEN,KEY,KEYLEN,WDS,WDMAX,WDLEN)
    !-----------------------------------------------------------------------
    !     GeT and ReMove Word Abreviated Many
    !     Gets and removes M words from ST prefixed by the keyword KEY. This
    !     routine is similar to GTRMWD except it allows KEY to be an
    !     abbreviation.
    !     This routine should remain internal to string.flx
    !
    use exfunc
    !
    integer M
    INTEGER STLEN,KEYLEN,WDMAX,WDLEN(1:M)
    CHARACTER(len=*) ST,KEY,WDS(1:M)
    !
    INTEGER IND,I
    !
    WDLEN(1:m)=0
    IND=INDXRA(ST,STLEN,KEY,KEYLEN,.TRUE.)
    IF(IND > 0) THEN
       do i=1,M
          CALL NXTWDA(ST,STLEN,WDS(I),WDMAX,WDLEN(i),IND)
       enddo
    ENDIF
    RETURN
  END SUBROUTINE GTRMWAM

  SUBROUTINE GETBETW(C1,C2,C3,C4)
    !-----------------------------------------------------------------------
    !     Look in the command line c1 for the word(s) placed
    !     just between c2  and c3 and put it in c4.
    !yw...04-Aug-93 from RISM
    !
    CHARACTER(len=*) C1,C2,C3,C4

    INTEGER LENC1,LENC2,LENC3
    INTEGER I2,I3

    LENC1=LEN(C1)
    LENC2=LEN(C2)
    LENC3=LEN(C3)
    C4=' '
    I2=INDEX(C1,C2)
    I3=INDEX(C1(I2+LENC2:LENC1),C3)+I2+LENC2-1
    !
    !     If c2 or c3 are not in c1 return
    !
    IF((I3 == 0).OR.(I2.EQ.0)) RETURN 
    C4=C1(I2+LENC2:I3-1)
    C1(I2:LENC1)=C1(I3+LENC3:LENC1)
    RETURN
  END SUBROUTINE GETBETW

  INTEGER FUNCTION INDXA(ST,STLEN,KEYWRD)
    !-----------------------------------------------------------------------
    CHARACTER(len=*) ST,KEYWRD
    INTEGER STLEN
    !
    INDXA=INDXRA(ST,STLEN,KEYWRD,LEN(KEYWRD),.TRUE.)
    RETURN
  END FUNCTION INDXA

  INTEGER FUNCTION GTRMI(ST,STLEN,KEYWRD,DEFNUM)
    !-----------------------------------------------------------------------
    CHARACTER(len=*) ST
    INTEGER STLEN
    CHARACTER(len=*) KEYWRD
    INTEGER DEFNUM,NUM

    NUM=DEFNUM
    IF(STLEN > 0) THEN
       CALL GTRMWA(ST,STLEN,KEYWRD,LEN(KEYWRD),SWDTCH,SWDMAX,SWDLEN)
       IF (SWDLEN /= 0) NUM=DECODI(SWDTCH,SWDLEN)
    ENDIF
    GTRMI=NUM
    RETURN
  END FUNCTION GTRMI

  SUBROUTINE GTRMIM(M,ST,STLEN,KEYWRD,DEFNUMS,NUMS,QPRESENT)
    !-----------------------------------------------------------------------
    !
    ! Extension of gtrmi function
    ! needed for iseed commands with many seed numbers
    ! gets M numbers into NUMS array
    ! defnums is also an array
    !
    use stream

    CHARACTER(len=*) ST
    INTEGER STLEN,M,I,J
    CHARACTER(len=*) KEYWRD
    LOGICAL QPRESENT,problem,qoldiseed
    INTEGER DEFNUMS(*),NUMS(*)
    character(len=swdmax),allocatable,dimension(:) :: lswdtch
    integer,allocatable,dimension(:) :: lswdlen
    integer,parameter :: maxrest=1000
    integer rstindx,rstlen0,rstlen1
    character(len=maxrest) strestore,strestore0,strestore1
    integer restoreindx
    !
    NUMS(1:m)=DEFNUMS(1:m)
    qpresent = .false.
    IF(STLEN > 0) THEN
       ! not using chmalloc here - it is character array
       allocate(lswdtch(m),lswdlen(m))
       if(stlen < maxrest) then
          rstlen0=stlen
          strestore0(1:stlen)=st(1:stlen)
       endif

       CALL GTRMWAM(M,ST,STLEN,KEYWRD,LEN(KEYWRD),LSWDTCH,SWDMAX,LSWDLEN)

       if(stlen < maxrest) then
          rstlen1=stlen
          strestore1(1:stlen)=st(1:stlen)
       endif

       gtrmim_err=0
       qoldiseed=.false.
       restoreindx=1
       do i = 1, m
          ! do some extra error checking here... Good for random()
          ! We also take care of the old iseed <int>
          ! syntax here: put err=-100
          problem=.false.
          if (lswdlen(i) > 0) then
             qpresent = .true.
             if (.not. isnum(lswdtch(i),lswdlen(i))) then
                problem=.true.
             else
                NUMS(i)=DECODI(LSWDTCH(i),LSWDLEN(i))
             endif
          endif
          if(qoldiseed)then
             strestore(restoreindx:restoreindx+lswdlen(i)+1) = &
                  ' '//lswdtch(i)(1:lswdlen(i))
             restoreindx=restoreindx+lswdlen(i)+1
          endif
          if(problem)then
             gtrmim_err=gtrmim_err-1
          else
             ! only the first is a number for old ISEED
             if(i == 1) qoldiseed = .true.
          endif
       enddo
       !
       ! Here we can put back to the parse string ST,STLEN
       ! Do this for ISEED only ???
       !
       if(qoldiseed.and.(gtrmim_err < 0)) then
          ! Error -100 for old iseed
          gtrmim_err=-100
          ! Put back M-1 elements from parse string
          if (rstlen0 < maxrest) then
             ! we need to calculate the right place of insertion:
             rstindx=0
             do j = 1, stlen
                rstindx=rstindx+1
                if ( strestore1(j:j) /= strestore0(j:j) ) exit
             enddo
             ! we dont need the strestore0 anymore
             ! use it as temporary buffer:
             strestore0(1:rstindx)=st(1:rstindx)
             strestore0(rstindx+1:rstindx+restoreindx) = &
                  strestore(1:restoreindx)
             strestore0(rstindx+restoreindx+1:stlen+restoreindx) = &
                  st(rstindx+1:stlen)
             ! now put it back to st:
             stlen = stlen + restoreindx
             st(1:stlen) = strestore0(1:stlen)
          endif
       endif

       deallocate(lswdtch,lswdlen)

    ENDIF

    RETURN
  END SUBROUTINE GTRMIM

  SUBROUTINE GTRMFM(M,ST,STLEN,KEYWRD,DEFNUMS,NUMS,QPRESENT)
    !-----------------------------------------------------------------------
    !
    ! Extension of gtrmf function 
    ! currently this routine is not used
    ! but it was easy to make it from gtrmim :-)
    ! gets M numbers into NUMS array
    ! defnums is also an array
    !
    CHARACTER(len=*) ST
    INTEGER STLEN,M,I
    CHARACTER(len=*) KEYWRD
    LOGICAL QPRESENT
    real(chm_real) DEFNUMS(1:m),NUMS(1:m)
    character(len=swdmax),allocatable,dimension(:) :: lswdtch
    integer,allocatable,dimension(:) :: lswdlen
    !
    NUMS(1:m)=DEFNUMS(1:m)
    qpresent = .false.
    IF(STLEN > 0) THEN
       allocate(lswdtch(m),lswdlen(m))  ! not using chmalloc here - it is internal
       CALL GTRMWAM(M,ST,STLEN,KEYWRD,LEN(KEYWRD),LSWDTCH,SWDMAX,LSWDLEN)
       do i = 1, m
          IF (LSWDLEN(i) /= 0) NUMS(i)=DECODF(LSWDTCH(i),LSWDLEN(i))
          IF (LSWDLEN(i) /= 0) qpresent = .true.
       enddo
       deallocate(lswdtch,lswdlen)
    ENDIF

    RETURN
  END SUBROUTINE GTRMFM

  FUNCTION GTRMF_CRL(ST,STLEN,KEYWRD,DEFNUM) result(gtrmff)
    !-----------------------------------------------------------------------
    !
    CHARACTER(len=*) ST
    INTEGER STLEN
    CHARACTER(len=*) KEYWRD
    real(chm_real) DEFNUM,NUM,gtrmff

    NUM=DEFNUM
    IF(STLEN > 0) THEN
       CALL GTRMWA(ST,STLEN,KEYWRD,LEN(KEYWRD),SWDTCH,SWDMAX,SWDLEN)
       IF (SWDLEN /= 0) NUM=DECODF(SWDTCH,SWDLEN)
    ENDIF
    GTRMFf=NUM
    RETURN
  END FUNCTION GTRMF_CRL

  function gtrmf_alt(st,stlen,keywrd,defnum) result(gtrmff)
    character(len=*) :: st, keywrd
    integer :: stlen
#if KEY_SINGLE==1
    real(chm_real8) :: defnum, gtrmff  
#endif
#if KEY_SINGLE==0
    real(chm_real4) :: defnum, gtrmff  
#endif
    real(chm_real) :: defnum_crl

    defnum_crl = defnum
    gtrmff = gtrmf_crl(st,stlen,keywrd,defnum_crl)
  end function gtrmf_alt

  CHARACTER(len=8) FUNCTION GTRMA(ST,STLEN,KEYWRD)
    !-----------------------------------------------------------------------
    !
    CHARACTER(len=*) ST
    INTEGER STLEN
    CHARACTER(len=*) KEYWRD
    INTEGER WDLEN
    !
    GTRMA='        '
    IF(STLEN > 0) THEN
       WDLEN=LEN(GTRMA)
       CALL GTRMWA(ST,STLEN,KEYWRD,LEN(KEYWRD),SWDTCH,WDLEN,SWDLEN)
       IF (SWDLEN /= 0) GTRMA=SWDTCH(1:SWDLEN)
    ENDIF
    !
    RETURN
  END FUNCTION GTRMA
    !
#if KEY_STRINGM==1 || KEY_MULTICOM==1 /*  VO: additional functions for mcom processing */
  CHARACTER(len=10) FUNCTION GTRMA10(ST,STLEN,KEYWRD)
    !-----------------------------------------------------------------------
    !
    CHARACTER(len=*) ST
    INTEGER STLEN
    CHARACTER(len=*) KEYWRD
    INTEGER WDLEN
    !
    GTRMA10=''
    IF(STLEN > 0) THEN
       WDLEN=LEN(GTRMA10)
       CALL GTRMWA(ST,STLEN,KEYWRD,LEN(KEYWRD),SWDTCH,WDLEN,SWDLEN)
       IF (SWDLEN /= 0) GTRMA10=SWDTCH(1:SWDLEN)
    ENDIF
    !
    RETURN
  END FUNCTION GTRMA10
#endif /* VO */
    !
  INTEGER FUNCTION INDXRA(ST,STLEN,KEY,KEYLEN,LABREV)
    !-----------------------------------------------------------------------
    !     Performs the INDX function except it removes the word in which
    !     KEY was found. The KEY must be preceded by a blank or the
    !     start of the string if LABREV is true.
    !     It must be followed by a blank if LABREV is false.
    !     This function can search for and handle abbreviations. In writing
    !     this routine, one must realize that one must check every occurence
    !     of KEY to see if it has the requisite blanks.
    !
    INTEGER STLEN,KEYLEN
    CHARACTER(len=*) ST,KEY
    LOGICAL LABREV
    INTEGER  IND,III
    !
    IND=0
10  CONTINUE
    III=0
    IND=INDXAF(ST,STLEN,KEY,KEYLEN,IND)
    IF (.NOT.(IND == 0)) THEN
       IF (IND == 1) THEN
          III=IND
       ELSE IF (ST(IND-1:IND-1) == BLANK) THEN
          III=IND
       ENDIF
       IF (LABREV) THEN
       ELSE IF (IND+KEYLEN-1 == STLEN) THEN
       ELSE IF (ST(IND+KEYLEN:IND+KEYLEN) == BLANK) THEN
       ELSE
          III=0
       ENDIF
    ENDIF
    IF (.NOT.(IND == 0 .OR. III > 0)) GOTO 10
    IF (III > 0) CALL ZAPWD(ST,STLEN,III)
    INDXRA=III
    RETURN
  END FUNCTION INDXRA

  SUBROUTINE ZAPWD(ST,STLEN,IND)
    !-----------------------------------------------------------------------
    !     Replaces the word starting at IND in ST to blanks. There had
    !     better be a non-blank character at ST(IND), otherwise ZAPWD will
    !     die.
    !
    !     This routine is to remain internal to STRING.FLX  - BRB
    !
    !     Author: Robert Bruccoleri
    !
    INTEGER STLEN,IND
    CHARACTER(len=*) ST
    !
    CALL NXTWDA(ST,STLEN,SCRTCH,SCRMAX,SCRLEN,IND)
    RETURN
  END SUBROUTINE ZAPWD

  INTEGER FUNCTION INDXAF(ST,STLEN,KEY,KEYLEN,IND)
    !-----------------------------------------------------------------------
    !     Searches the string, ST, for KEY, starting at ST(IND+1). If the
    !     search fails, IND is set to zero.
    !
    !     This routine is to remain internal to STRING.FLX  - BRB
    !
    !     Author: Robert Bruccoleri
    !
    INTEGER STLEN,KEYLEN,IND
    CHARACTER(len=*) ST,KEY
    INTEGER J,K,L
    !
    IF(IND < 0) CALL DIEWRN(-2)
    INDXAF=0
    IF(KEYLEN <= 0) RETURN
    J=IND+1
    IF(J+KEYLEN-1 > STLEN) RETURN
    L=STLEN-J+1
    k = index(st(j:stlen), key(1:keylen))
    IF(K == 0) RETURN
    INDXAF=K+J-1
    RETURN
  END FUNCTION INDXAF

  !
  !     THIS GROUP OF STRING COMMANDS DEALS WITH STRING COMPARISONS
  !
  LOGICAL FUNCTION EQST(ST1,LEN1,ST2,LEN2)
    !-----------------------------------------------------------------------
    !     COMPARES TWO STRINGS AND RETURNS TRUE IF THEY ARE EQUAL.
    !
    CHARACTER(len=*) ST1,ST2
    INTEGER LEN1,LEN2
    !
    IF(LEN1 <= 0 .AND. LEN2.LE.0) THEN
       EQST=.TRUE.
       RETURN
    ENDIF
    IF(LEN1 /= LEN2) THEN
       EQST=.FALSE.
       RETURN
    ENDIF
    EQST=ST1(1:LEN1) == ST2(1:LEN2)
    RETURN
  END FUNCTION EQST

  LOGICAL FUNCTION EQSTA(ST,STLEN,KEYWRD)
    !-----------------------------------------------------------------------
    CHARACTER(len=*) ST,KEYWRD
    INTEGER STLEN
    !
    EQSTA=EQST(ST,STLEN,KEYWRD,LEN(KEYWRD))
    RETURN
  END FUNCTION EQSTA

  LOGICAL FUNCTION LTSTEQ(ST1,ST1LEN,ST2,ST2LEN,LEQFG)
    !-----------------------------------------------------------------------
    !     Determines whether string st1 is lexicographically less or equal
    !     than string st2, i.e. st1 <= st2. If one string is equal to the
    !     beginning of the other string it is considered to be less than it.
    !     Axel Brunger, FEB-83
    !
    CHARACTER(len=*) ST1,ST2
    INTEGER   ST1LEN,ST2LEN
    LOGICAL LEQFG
    !
    LOGICAL DONE
    INTEGER I
    !
    I=0
10  CONTINUE
    DONE=.TRUE.
    I=I+1
    IF (I > ST2LEN.AND.ST2LEN == ST1LEN.AND.LEQFG) THEN
       LTSTEQ=.TRUE.
    ELSE IF (I > ST2LEN) THEN
       LTSTEQ=.FALSE.
    ELSE IF (I > ST1LEN) THEN
       LTSTEQ=.TRUE.
    ELSE IF (ST1(I:I) < ST2(I:I)) THEN
       LTSTEQ=.TRUE.
    ELSE IF (ST1(I:I) == ST2(I:I)) THEN
       DONE=.FALSE.
    ELSE
       LTSTEQ=.FALSE.
    ENDIF
    IF (.NOT.(DONE)) GOTO 10
    RETURN
  END FUNCTION LTSTEQ


  !===========================================================================
  ! Returns true if strings st and wc are equal, allowing wildcards in wc
  ! as described in select.doc.
  logical function eqstwc(st,stlen,wc,wclen) result(eqstwc_rtn)
    use exfunc
    integer :: stlen,wclen
    character(len=*) :: st,wc

    integer stl,wcl,oldlst,ret
    character(len=1) :: &
         star  ='*', &
         number='#', &
         percnt='%', &
         plus  ='+', &
         backsl='\'    !' this comment helps emacs know that the string has ended.
    character(len=1) :: digits(10)=(/'0','1','2','3','4','5','6','7','8','9'/)

    stl=stlen
    wcl=wclen
    call trime(st,stl)
    call trime(wc,wcl)
    call match_tail(1, 1)
    return

  contains

    ! Sets eqstwc_rtn to .true. if st(istp:) and wc(iwcp:) match,
    ! or .false. if a mismatch is found.
    recursive subroutine match_tail(istp, iwcp)
      integer,intent(in) :: istp, iwcp
      integer :: ist, iwc

      ist = istp
      iwc = iwcp
      do
         if(ist > stl .and. iwc.gt.wcl) then
            eqstwc_rtn=.true.
            exit
         else if(iwc > wcl) then
            eqstwc_rtn=.false.
            exit
         else if(wc(iwc:iwc) == star) then
            ! match any 0 or more chars in st
            iwc=iwc+1
            call match_tail(ist, iwc)
            if (eqstwc_rtn) exit
            do while(ist <= stl)
               ist=ist+1
               call match_tail(ist, iwc)
               if (eqstwc_rtn) return
            enddo
            eqstwc_rtn=.false.
            exit
         else if(wc(iwc:iwc) == number) then
            ! match 0 or more digits in st
            iwc=iwc+1
            call match_tail(ist, iwc)
            if (eqstwc_rtn) exit
            do while(ist <= stl)
               if (srchws(digits,10,st(ist:ist)) == 0) then
                  eqstwc_rtn=.false.
                  return
               endif
               ist=ist+1
               call match_tail(ist, iwc)
               if (eqstwc_rtn) return
            enddo
            eqstwc_rtn=.false.
            exit
         else if(ist > stl) then
            eqstwc_rtn=.false.
            exit
         else if(wc(iwc:iwc) == percnt) then
            ! match any 1 char in st
            iwc=iwc+1
            ist=ist+1
         else if(wc(iwc:iwc) == plus) then
            ! match 1 digit in st
            if (srchws(digits,10,st(ist:ist)) == 0) then
               eqstwc_rtn=.false.
               exit
            endif
            ist=ist+1
            iwc=iwc+1
         else if(wc(iwc:iwc) == backsl) then
            ! next char in wc is literal
            iwc=iwc+1
            if (iwc > wcl) then
               eqstwc_rtn=st(ist:ist) == backsl .and. ist.eq.stl
               exit
            else if(st(ist:ist) == wc(iwc:iwc)) then
               iwc=iwc+1
               ist=ist+1
            else
               eqstwc_rtn=.false.
               exit
            endif
         else if(st(ist:ist) == wc(iwc:iwc)) then
            iwc=iwc+1
            ist=ist+1
         else
            eqstwc_rtn=.false.
            exit
         endif
      enddo
      return
    end subroutine match_tail

  end function eqstwc

  LOGICAL FUNCTION EQWDWC(ST,WCWRD)
    !-----------------------------------------------------------------------
    !
    CHARACTER(len=*) ST,WCWRD
    !
    EQWDWC=EQSTWC(ST,LEN(ST),WCWRD,LEN(WCWRD))
    RETURN
  END FUNCTION EQWDWC

  !
  !     THIS GROUP OF STRING COMMANDS DEALS WITH STRING MANIPULATION
  !
  LOGICAL FUNCTION DOTRIM(ST)
    !-----------------------------------------------------------------------
    !     This subroutine removes the blanks on the left
    !yw...04-Aug-93 from RISM
    CHARACTER(len=*) ST
    INTEGER LENGTH,I
    !
    LENGTH=LEN(ST)
    SCRTCH=ST
    DO I=1,LENGTH
       DOTRIM=ST(I:I) /= ' '
       IF(DOTRIM)THEN
          ST(1:LENGTH)=SCRTCH(I:LENGTH)
          RETURN
       ENDIF
    ENDDO
    !
    RETURN
  END FUNCTION DOTRIM

  subroutine trime(st,stlen)
    ! this routine sets stlen to the index of the first
    ! non blank character from the left starting at stlen.
    ! it also blanks out extraneous characters past stlen.
    implicit none
    integer, intent(inout) :: stlen
    character(len=*), intent(inout) :: st
    integer :: old_len
    if (stlen .eq. 0) return
    old_len = len(st)
    if (stlen < old_len) st((stlen + 1):old_len) = blank
    stlen = len_trim(st(1:stlen))
  end subroutine trime

  subroutine trimb(st,stlen)
    ! this routine removes leading blanks by shifting st to the left
    ! and then reducing stlen
    implicit none
    character(len=*), intent(inout) :: st
    integer, intent(inout) :: stlen
    if (stlen .eq. 0) return
    st(1:stlen) = adjustl(st(1:stlen))
    stlen = len_trim(st(1:stlen))
  end subroutine trimb

  SUBROUTINE TRIMA(ST,STLEN)
    !-----------------------------------------------------------------------
    !     THIS ROUTINE DECREMENTS STLEN UNTIL IT POINTS TO THE FIRST
    !     NON BLANK CHARACTER. IT HAS THE EFFECT OF REMOVING THE
    !     THE TRAILING SPACES. IT ALSO REMOVES FRONT END BLANKS
    !     B. R. BROOKS 3/83
    !
    CHARACTER(len=*) ST
    INTEGER STLEN
    !
    CALL TRIME(ST,STLEN)
    CALL TRIMB(ST,STLEN)
    RETURN
  END SUBROUTINE TRIMA

  SUBROUTINE CMPRST(ST,STLEN)
    !-----------------------------------------------------------------------
    !     This routine compresses a string by removing all excess
    !     blanks. Any group of blanks is replaced by a single blank.
    !     Blanks within double quotes will not be compressed.
    !
    !     Bernard R. Brooks    9/84
    !
    CHARACTER(len=*) ST
    INTEGER STLEN
    !
    INTEGER I,IPT
    LOGICAL LBLANK,LDBLQ
    CHARACTER(len=1) :: SDBLQ='"', CHX
    !
    IPT=0
    ! Logic modified to remove leading blanks - BRB
    LBLANK=.TRUE.
    LDBLQ=.FALSE.
    DO I=1,STLEN
       IF(ST(I:I) == SDBLQ) LDBLQ=.NOT.LDBLQ
       IF (LBLANK) THEN
          IF(ST(I:I) /= BLANK .OR. LDBLQ) THEN
             LBLANK=.FALSE.
             IPT=IPT+1
          ENDIF
       ELSE
          LBLANK=(ST(I:I) == BLANK)
          IPT=IPT+1
       ENDIF
       IF(IPT /= I .AND. IPT > 0) THEN
          CHX=ST(I:I)
          ST(IPT:IPT)=CHX
       ENDIF
    ENDDO
    !
    STLEN=IPT
    RETURN
  END SUBROUTINE CMPRST

  subroutine cnvtuc(st,stlen)
    !-----------------------------------------------------------------------
    !     This routine converts a string to upper case and removes
    !     nonacceptable control characters. This routine is ASCII code
    !     dependant.
    !     Substring portions in double quotes are not converted.
    !     The double quotes are not removed.
    !
    !     Modified routine by: Stephen H. Fleischman 2/88
    !# <caves>-Jan-17-1994 (Leo Caves) use character codes from string.f90

    integer,intent(in) :: stlen
    character(len=stlen),intent(inout) :: st

    integer itab,ilita,ilitz,ibiga,ichr,i,iquote
    logical capson

    ilita = ICHLCA
    ilitz = ICHLCZ
    ibiga = ICHUCA - ilita
    iquote = ICHQUO
    itab = ichtab
    capson = .true.


    do i=1,stlen
       ichr=ichar(st(i:i))
       if (ichr == iquote) then
          capson = .not.capson
       else if (ichr == itab) then
          st(i:i) = BLANK
       else if (capson) then
          if(ichr >= ilita.and.ichr <= ilitz) st(i:i)=char(ichr+ibiga)
       endif
    enddo

    return
  end subroutine cnvtuc

  SUBROUTINE CNVTLC(ST,STLEN)
    !-----------------------------------------------------------------------
    !     This routine converts a string to lower case
    !# <caves>-Jan-17-1994 (Leo Caves) use character codes from string.f90
    !
    INTEGER STLEN
    CHARACTER(len=*) ST
    !
    LOGICAL CAPSON
    INTEGER ILITA,IBIGZ,IBIGA,ICHR,I,IQUOTE,ITAB
    !
    !
    IBIGA=ICHUCA
    IBIGZ=ICHUCZ
    ILITA=ICHLCa-IBIGA
    !
    IQUOTE= ICHQUO
    ITAB = ICHTAB
    CAPSON = .TRUE.
    !
    DO I=1,STLEN
       ICHR=ICHAR(ST(I:I))
       IF (ICHR == IQUOTE) THEN
          CAPSON = .NOT.CAPSON
       ELSE IF (ICHR == ITAB) THEN
          ST(I:I)=BLANK
       ELSE IF (CAPSON) THEN
          IF(ICHR >= IBIGA.AND.ICHR <= IBIGZ) ST(I:I)=CHAR(ICHR+ILITA)
       ENDIF
    ENDDO
    !
    RETURN
  END SUBROUTINE CNVTLC

  INTEGER FUNCTION STRLNG(STRING)
    !-----------------------------------------------------------------------
    CHARACTER(len=*) STRING
    INTEGER SLEN,I
    !     length of string
    SLEN = LEN(STRING)
    STRLNG = 0
    DO I = SLEN,1,-1
       IF((STRING(I:I) /= ' ').AND.(ICHAR(STRING(I:I)).NE.0)) THEN
          STRLNG = I
          GOTO 100
       END IF
    ENDDO
100 CONTINUE
    !
    RETURN
  END FUNCTION STRLNG

  SUBROUTINE SPLITI(STRING,NUM,ALPHA,STLEN)
    !-----------------------------------------------------------------------
    !     Axel Brunger, 26-JAN-83
    !
    !     For the string 'repeat{digit} repeat{alpha}'
    !     the routine splits the numerical and the alphanumerical part
    !     and stores the integer in NUM ,the latter part in ALPHA.
    !     Leading blanks are removed. On output the length of ALPHA is
    !     stored in STLEN. However, ALPHA is filled up with blanks to
    !     STLEN on input.
    !     NUM is 0 if no integers are present.
    !
    INTEGER STLEN
    CHARACTER(len=*) STRING, ALPHA
    INTEGER NUM

    INTEGER IND, J, LEN
    CHARACTER(len=1) :: NUMBER(10)=(/'0','1','2','3','4','5','6','7','8','9'/)
    !
    !
    LEN=STLEN
    CALL TRIME(STRING,LEN)
    IF(LEN == 0) THEN
       NUM=0
       ALPHA=' '
       RETURN
    ENDIF
    !
    IND=0
10  CONTINUE
    IND=IND+1
    J=0
20  CONTINUE
    J=J+1
    IF (.NOT.(NUMBER(J) == STRING(IND:IND).OR.J.EQ.10)) GOTO 20
    IF (.NOT.(IND == LEN.OR.NUMBER(J) /= STRING(IND:IND))) GOTO 10
    IF (IND == LEN.AND.NUMBER(J).EQ.STRING(IND:IND)) IND=IND+1
    DO J=1,IND-1
       ALPHA(J:J)=STRING(J:J)
    ENDDO
    IF (IND > 1) THEN
       NUM=DECODI(ALPHA,IND-1)
    ELSE
       NUM=0
    ENDIF
    ALPHA=' '
    DO J=1,LEN-IND+1
       ALPHA(J:J)=STRING(IND+J-1:IND+J-1)
    ENDDO
    STLEN=LEN-IND+1
    RETURN
  END SUBROUTINE SPLITI
  !
  !     THIS GROUP OF STRING COMMANDS DEALS WITH DATA CONVERSION
  !
  SUBROUTINE ENCODI(I,ST,MXSTLN,STLEN)
    !-----------------------------------------------------------------------
    !     ENCODE THE INTEGER I INTO THE STRING ST
    !
    !     BRB -  3-JUL-1984
    !
    INTEGER I,MXSTLN,STLEN
    CHARACTER(len=*) ST
    !
    WRITE(SCRTCH,'(I20)') I
    SCRLEN=20
    CALL TRIMA(SCRTCH,SCRLEN)
    !
    IF(SCRLEN > MXSTLN) THEN
       CALL WRNDIE(-1,'<ENCODI>','STRING TOO SMALL')
       ST=' '
       STLEN=0
       RETURN
    ENDIF
    !
    ST=SCRTCH(1:SCRLEN)
    STLEN=SCRLEN
    RETURN
  END SUBROUTINE ENCODI

  SUBROUTINE ENCODF(R,ST,MXSTLN,STLEN)
    !-----------------------------------------------------------------------
    !     THIS WILL ENCODE THE REAL NUMBER R INTO THE STRING ST AND WILL
    !     ATTEMPT TO SHORTEN THE ENCODING AS MUCH AS POSSIBLE. THE NUMBER IS
    !     CONVERTED USING A 1PG14.6 FORMAT. THE TECHNIQUES USED FOR
    !     SHORTENING ARE REMOVAL OF LEADING BLANKS, REMOVAL OF TRAILING
    !     BLANKS, REMOVAL OF A SUPERFLUOUS DECIMAL POINT AND REMOVAL OF
    !     LEADING ZEROES IN THE EXPONENT.
    !
    !     Author: Robert Bruccoleri
    !
    use exfunc
    use number
    use consta
    !
    real(chm_real) R
    INTEGER MXSTLN,STLEN
    CHARACTER(len=*) ST
    !
    CHARACTER(len=20) WD
    INTEGER WDLEN,IPT,I
    !
    IF (FMTLEN > 0) THEN
       !
       !     process stored format for coding
       !
       if (index(fmtwd(1:fmtlen), "I") > 0) then
          !     process with an integer format
          I=R+SIGN(HALF,R)
          WRITE(WD,FMTWD(1:FMTLEN)) I
       ELSE
          WRITE(WD,FMTWD(1:FMTLEN)) R
       ENDIF
       WDLEN=20
       CALL TRIME(WD,WDLEN)
       ST=WD
       STLEN=WDLEN
    ELSE
       !
       !     process standard format coding
       !
       WRITE(WD,'(1PG17.9)') R
       WDLEN=20
       !
       !     add a leading zero if it is missing
       ipt = index(wd(1:wdlen), " .")
       IF(IPT > 0) WD(IPT:IPT)='0'
       !
       CALL TRIMA(WD,WDLEN)
       !
       !     trim off zeroes following the decimal

       ipt = index(wd(1:wdlen), "E")

       IF (IPT > 0) THEN
          READ(WD(IPT+1:WDLEN),'(I3)') I
          IF(I == 0) WDLEN=IPT-1
       ELSE
          IPT=WDLEN+1
       ENDIF
       I=IPT
20     CONTINUE
       IPT=IPT-1
       IF (.NOT.(IPT == 1 .OR. WD(IPT:IPT) /= '0')) GOTO 20
       IF(WD(IPT:IPT) == '.') IPT=IPT-1
       !
       IF (I <= WDLEN) THEN
          ST=WD(1:IPT)//WD(I:WDLEN)
          STLEN=WDLEN-I+IPT+1
       ELSE
          !ab...Bugfix 20-Dec-94, .0 encodig bugfixed
          IF ((R == 0.0) .AND. (IPT.EQ.0)) THEN
             ST='0'
             STLEN=1
          ELSE
             ST=WD(1:IPT)
             STLEN=IPT
          ENDIF
          !ab...
       ENDIF
       !
    ENDIF
    !
    IF(STLEN > MXSTLN) THEN
       CALL WRNDIE(0,'<ENCODF>','STRING TOO SMALL')
       STLEN=0
       RETURN
    ENDIF
    !
    RETURN
  END SUBROUTINE ENCODF

  INTEGER FUNCTION DECODI(ST,STLEN)
    !-----------------------------------------------------------------------
    !     Converts the string into an integer. A Fortran DECODE statement is
    !     used.
    !
    !     Author: Robert Bruccoleri
    !
    use stream
    !
    INTEGER STLEN
    CHARACTER(len=*) ST
    CHARACTER(len=6) FMT
    INTEGER L
    real(chm_real) REALINT

    DECODI=0
    L=STLEN
    CALL TRIME(ST,L)
    IF(L == 0) THEN
       IF(WRNLEV >= 2) WRITE(OUTU,'(1X,A)') &
            'WARNING from DECODI -- Zero length string being converted to 0'
       RETURN
    ENDIF
    if(.not.isnum(st,L)) goto 290
    WRITE(FMT,200) L
200 FORMAT('(I',I3,')')
    READ(ST,FMT,ERR=290) L
    DECODI=L
    RETURN
    !
290 CONTINUE
    !yw...The number may be in scientific notation.  07-Aug-95
    REALINT=DECODF(ST,STLEN)
    IF(REALINT == 0.0 .AND. WRNLEV >= 2) THEN
       if (prnlev >= 2) WRITE(OUTU,'(A)') &
            ' WARNING FROM DECODI - COULD NOT CONVERT STRING'
       if (prnlev >= 2) CALL PRNTST(OUTU,ST,STLEN,1,80)
       if (prnlev >= 2) WRITE(OUTU,'(A)') 'ZERO WILL BE RETURNED.'
       DECODI=0
    ELSE
       L=INT(REALINT)
       REALINT=ABS(REALINT-L)
       IF(REALINT > 0.001 .AND. WRNLEV >= 2) THEN
          if (prnlev >= 2) WRITE(OUTU,'(A)') &
               ' WARNING FROM DECODI - REAL to INTEGER conversion'
          if (prnlev >= 2) CALL PRNTST(OUTU,ST,STLEN,1,80)
          if (prnlev >= 2) WRITE(OUTU,'(A)') 'Integer portion will be returned.'
       ENDIF
       DECODI=L
    ENDIF
    !yw...
    !
    RETURN
  END FUNCTION DECODI

  FUNCTION DECODF(ST,STLEN) result(decodff)
    !-----------------------------------------------------------------------
    !     DECODES THE STRING INTO A REAL NUMBER AND RETURNS ITS VALUE. IT
    !     USES A F<STLEN>.0 FORMAT SO THAT THE ABSENCE OF A DECIMAL POINT
    !     WILL NOT MESS THINGS UP.
    !
    !     Author: Robert Bruccoleri
    !
    use stream
    INTEGER STLEN
    CHARACTER(len=*) ST
    CHARACTER(len=7) FMT
    INTEGER IOFF,L
    real(chm_real) VAL,decodff

    DECODFf=0.0
    L=STLEN
    IOFF=1
    CALL TRIME(ST,L)
    IF (L == 0) THEN
! better trap ...
       call wrndie(1,'<DECODF>','Zero length string')
       IF (WRNLEV  >=  2) WRITE(OUTU,'(2A)') &
            ' WARNING from DECODF --', &
            ' Zero length string being converted to 0.'
       RETURN
    ENDIF
    IF (L > 99) THEN
       CALL WRNDIE(0,'<DECODF>', &
            'STRING HAS TOO MANY DIGITS. ZERO USED')
       RETURN
    ENDIF
    !
    IF(L > 2 .AND. EQSTA(ST,2,'--')) THEN
       L=L-2
       IOFF=3
    ENDIF
    if(.not.isnum(st,L)) goto 290
    WRITE(FMT,195) '(F',L,'.0)'
195 FORMAT(A,I2,A)
    IF(STLEN < IOFF) RETURN
    READ(ST(IOFF:STLEN),FMT,ERR=290) VAL
    DECODFf=VAL
    RETURN
290 IF (WRNLEV  >=  2 .and. prnlev >= 2) WRITE(OUTU,'(A)') &
        ' WARNING FROM DECODF - COULD NOT CONVERT STRING'
    if (prnlev >= 2) CALL PRNTST(OUTU,ST,STLEN,1,80)
    call wrndie(0,'<DECODF>','Could not convert string')
    if (prnlev >= 2) WRITE(OUTU,'(A)') 'ZERO WILL BE RETURNED.'
    DECODFf=0.0
    RETURN
  END FUNCTION DECODF

  LOGICAL FUNCTION QDIGIT(CH)
    !# <caves>-Aug-26-1993 (Leo Caves)
    !
    !...Is the passed character a digit ?
    !
    !...Passed Variables.
    CHARACTER(len=1) CH
    !...Local.
    INTEGER ICH
    !...Begin Executable Code. 
    !
    ICH = ICHAR(CH)
    QDIGIT = ((ICH >= ICHZERO).AND.(ICH <= ICHNINE))
    !...Exit.
    RETURN
  END FUNCTION QDIGIT
  !
  LOGICAL FUNCTION QALPHA(CH)
    !# <caves>-Aug-26-1993 (Leo Caves)
    !
    !...Is the passed character an alpha ?
    !
    !...Passed Variables.
    CHARACTER(len=1) CH
    !...Local.
    INTEGER ICH
    !...Begin Executable Code. 
    !
    ICH = ICHAR(CH)
    QALPHA = ((ICH >= ICHUCA).AND.(ICH <= ICHUCZ) .OR.  &
         (ICH >= ICHLCa).AND.(ICH <= ICHLCz) ) 
    !...Exit.
    RETURN
  END FUNCTION QALPHA


  logical function isnum(st,stlen)
    !-----------------------------------------------------------------------
    !     check if the string ST is a number
    !     use QDIGIT(CH) to check a single digit
    !     assume ST is trimmed
    character(len=*) st
    integer       stlen

    !     local
    integer i

    isnum = .true.
    do i=1,stlen
       if(.not.qdigit(st(i:i))) then
          if(st(i:i) == '+' .or. st(i:i).eq.'-' .or. &
               st(i:i) == '.' .or. st(i:i).eq.' ' .or. &
               st(i:i) == 'E' .or. st(i:i).eq.'e' .or. &
               st(i:i) == 'D' .or. st(i:i).eq.'d') then
             continue
          else
             isnum=.false.
          endif
       endif
    enddo

    return
  end function isnum

  logical function qxform()
    !-----------------------------------------------------------------------
    !     Use extended format for i/o?
    !
    use ffieldm
    use param
    use psf
    use stream

    integer i

    qxform=.false.

    if (qextfmt) then
       qxform=.true.
       return
    endif

    if (qoldfmt) then
       qxform=.false.
       return
    endif

    if (qnewfmt) then
       qxform=.true.
       return
    endif

    !     find if extended format is needed
    if (natom >= 100000) then
       qxform=.true.
       idleng=8
       if (prnlev >= 2) write(outu,'(a,/,a)') &
            ' QXFORM> Expanded format used.', &
            '         The number of atoms exceeds 100,000.'
       return
    endif

    do i=1,nseg
       if (segid(i)(5:5) /= blank) then
          qxform=.true.
          idleng=8
          if (prnlev >= 2) write(outu,'(a,/,a)') &
               ' QXFORM> Expanded format used.', &
               '         More than 4 character SEGID used.'
          return
       endif
    enddo

    do i=1,nres
       if (resid(i)(5:5) /= blank) then
          qxform=.true.
          idleng=8
          if (prnlev >= 2) write(outu,'(a,/,a)') &
               ' QXFORM> Expanded format used.', &
               '         More than 4 character RESID used.'
          return
       endif
       if (res(i)(5:5) /= blank) then
          qxform=.true.
          idleng=8
          if (prnlev >= 2) write(outu,'(a,/,a)') &
               ' QXFORM> Expanded format used.', &
               '         More than 4 character residue name used.'
          return
       endif
    enddo

    do i=1,natom
       if (atype(i)(5:5) /= blank) then
          qxform=.true.
          idleng=8
          if (prnlev >= 2) write(outu,'(a,/,a)') &
               ' QXFORM> Expanded format used.', &
               '         More than 4 character atom name used.'
          return
       endif
    enddo

    !yw 13-Jan-2006 add checking param.f90 ATC*8
#if KEY_MMFF==1 || KEY_CFF==1
    if(ffield == charmm .or. ffield == amberffn) then
       do i=1,natc
          if (atc(i)(5:5) /= blank) then
             qxform=.true.
             idleng=8
             if (prnlev >= 2) write(outu,'(a,/,a)') &
                  ' QXFORM> Expanded format used.', &
                  '         More than 4 character atom type used.'
             return
          endif
       enddo
    endif
#endif 
    return
  end function qxform
     !
! additional functions for stringm and pnm
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! V. Ovchinnikov 2010-13; convert numbers to string
   function itoa4(i)
   use number, only : one
   integer*4 :: i
   integer*4, parameter :: ione=1, itwo=2
!   character(len=((ceiling(log10(one*abs(i)+1.)))+1)) :: itoa
!   character(len=((ceiling(log10(one*abs(i)+1.)))+count((/i.lt.0/)) )) :: itoa4 ! does not work with Pathscale
   character(len=max(((ceiling(log10(one*abs(i)+1.)))+ (ione-sign(ione,i))/itwo ),1)) :: itoa4
   character(len=80) :: b 
   write(b,*) i
   itoa4=adjustl(b)
   end function itoa4
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
   function itoa8(i)
   use number, only : one
   integer*8 :: i
   integer*8, parameter :: ione=1, itwo=2
!   character(len=((ceiling(log10(one*abs(i)+1.)))+1)) :: itoa
!   character(len=((ceiling(log10(one*abs(i)+1.)))+count((/i.lt.0/)) )) :: itoa8
   character(len=max(((ceiling(log10(one*abs(i)+1.)))+ (ione-sign(ione,i))/itwo ),1)) :: itoa8
   character(len=80) :: b 
   write(b,*) i
   itoa8=adjustl(b)
   end function itoa8
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
   function ftoa(f)
   real(chm_real) :: f
   character(len=15) :: ftoa
   write(ftoa,'(G15.10)') f
   end function ftoa
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
   function ltoa(l)
   logical :: l
   character(len=3) :: ltoa
   if (l) then ; ltoa='YES' ; else ; ltoa='NO ' ; endif
   end function ltoa
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
end module string
