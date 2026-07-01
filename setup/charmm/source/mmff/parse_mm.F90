#if KEY_MMFF==1
! =====================================================================
! SUBROUTINE GTLINE : Get a new line from the input unit
! =====================================================================
SUBROUTINE GTLINE(LINE,IXSTAT,IZSTAT,NCHAR,NEW)
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   Adapted by R. Czerminski of Molecular Simulations, Inc. from
  !  code developed at Merck and Co., Inc. by T. Halgren and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !
  !  28 Nov 95 Jay Banks: removed space from all "GO TO"s, as suggested in
  !  developer.doc.
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use mmffm
  use stream
  implicit none
  !
  integer ixstat, izstat, nchar, nmax, null
  integer nmax2
  CHARACTER(len=*) LINE
  LOGICAL :: NEW,FIRST=.TRUE.
  !
  !     DATA NULL/0/
  save first

  null=0
  !
  !   BRANCH IF THIS IS THE FIRST CALL TO GTLINE
  IF(FIRST) GOTO 50
  !   BRANCH IF THE CALLING SUBROUTINE INSISTS THAT A NEW RECORD
  !   BE READ
  !      IF(NEW) WRITE(6,*) ' NEW RECORD REQUESTED'
  IF(NEW) GOTO 50
  !   ALSO BRANCH IF NOTHING IS LEFT IN THE CURRENT LINE BUFFER
  IF(LINE.EQ.' ') GOTO 50
  !
  !   ALSO BRANCH IF THE FIRST CHARACTER IS "!", INDICATING A COMMENT
  !   FIELD WHICH IS TO SIMPLY BE IGNORED BY dimens
  !
  IF(LINE(1:1).EQ.'!') GOTO 50
  !   FIND THE LENGTH OF THE RESIDUAL LINE BUFFER AND RETURN
  !   TO THE CALLING SUBROUTINE
  NCHAR=LEN_TRIM(LINE)
  RETURN
  !   READ THE NEXT INPUT LINE.  ECHO TO APPROPRIATE OUTPUT UNITS
  !   AS DETERMINED BY IXSTAT, IZSTAT, AND LOGICAL VARIABLE OBEY.
  !   ALSO, PROCESS FURTHER IF THE LINE BEGINS WITH ONE OR MORE
  !   SPECIAL CHARACTERS
  !
50 CONTINUE
  FIRST=.FALSE.
  LINE=' '
  READ(ISTRM,'(A)',END=200) LINE
  !   CALCULATE THE LENGTH OF THE CHARACTER STRING
  NCHAR=LEN_TRIM(LINE)
  NMAX=MAX(1,NCHAR)
  NMAX2=MAX(2,NMAX)
  !
  !   CHECK TO SEE WHETHER THIS IS A COMMENT LINE WHICH
  !   IS SIMPLY TO BE IGNORED (KEY IS THAT THE FIRST CHARACTER IS "!")
  !
  IF(LINE(1:1).EQ.'!') THEN
     GOTO 50
  ENDIF
  !   also send LINE TO UNIT IOZ (=IO IN BATCH RUNS BUT =IZ IN LIVE RUNS)
  !   IF IZSTAT=1 AND IF IOZ.NE.0 (NOTE IOZ IS ARTIFICIALLY SET TO
  !   ZERO ON OCCASIONS WHEN HARD-COPY OUTPUT IS NOT BEING KEPT
  !     IF(IZSTAT.EQ.1.AND.IOZ.NE.0) WRITE(OUTU,'(A,A)') ' # ',
  !     IF(IZSTAT.EQ.1) WRITE(OUTU,'(A,A)') ' # ',LINE(1:NMAX)
  RETURN
  !
200 CONTINUE
  !   NULL LINE (<CNTL-Z>) ENCOUNTERED IN A "LIVE" RUN - ASK "REENTER"
  NULL=NULL+1
  IF(NULL.GT.10) THEN
     CALL SAYC(' *** TOO MANY NULL LINES ... TERMINATING ***')
     CALL WRNDIE(-5,'<gtline>','ERROR MSG')
  ENDIF
  CALL SAYC(' NULL LINE ENCOUNTERED -- REENTER:')
  GOTO 50
  !
END SUBROUTINE GTLINE

! SUBROUTINE GTWORD : COPY FIRST TOKEN FROM CHBUF TO TMPBUF AND
! RIGHT-JUSTIFY; LEFT-JUSTIFY REMAINDER OF CHBUF
! ============================================================
SUBROUTINE GTWORD(TMPBUF,CHBUF,OK,RJUST)
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   Adapted by R. Czerminski of Molecular Simulations, Inc. from
  !  code developed at Merck and Co., Inc. by T. Halgren and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !  16 Oct 95 Jay Banks: changed variable names "IM" and "KM" to "IIM"
  !                       and "KKM" to avoid conflict with CHARMM names
  !
  !  28 Nov 95 Jay Banks: removed space from all "GO TO"s, as suggested in
  !  developer.doc.
  !
  use chm_kinds
  implicit none
  !
  integer i, iim, k, kkm, limit, low, ltmp, m
  INTEGER, PARAMETER :: LF=80
  CHARACTER(len=*) TMPBUF,CHBUF
  CHARACTER(len=LF) CLONE
  LOGICAL OK,RJUST
  !
  OK=.TRUE.
  TMPBUF=' '
  LOW=1
  LIMIT=LEN(CHBUF)
  DO I=LOW,LIMIT
     IIM=I
     IF(CHBUF(I:I).NE.' ') GOTO 20
  ENDDO
  !   IF FELL THROUGH, REST OF CHBUF IS BLANK
  !   SET OK=.FALSE. TO TELL CALLING ROUTINE THAT NO INFORMATION
  !   IS BEING RETURNED
  OK=.FALSE.
  RETURN
20 CONTINUE
  !   CURRENT WORD (TOKEN) STARTS HERE
  LOW=IIM
  M=0
  DO K=LOW,LIMIT
     IF(CHBUF(K:K).EQ.' ') GOTO 600
     KKM=K
     M=M+1
  ENDDO
600 CONTINUE
  !   OK - WORD STARTS IN POSITION LOW AND EXTENDS TO POSITION KKM
  !   COPY TO TMPBUF; RIGHT-JUSTIFY IF SO REQUESTED
  LTMP=LEN(TMPBUF)
  TMPBUF=' '
  IF(RJUST) THEN
     TMPBUF(LTMP-M+1:LTMP)=CHBUF(LOW:KKM)
  ELSE
     TMPBUF(1:)=CHBUF(LOW:KKM)
  ENDIF
  !   NOW CLONE CHBUF AND THEN COPY RESIDUAL PART BACK INTO CHBUF ON
  !   RETURN
  IF(KKM+2.LE.LIMIT) THEN
     CLONE(1:)=CHBUF(KKM+2:)
     CHBUF(1:)=CLONE(1:)
     !        WRITE(6,*) ' NEW CHBUF = ',CHBUF
  ELSE
     CHBUF=' '
  ENDIF
  !
  RETURN
END SUBROUTINE GTWORD

! ====================================================================
! SUBROUTINE RDWORD : READ A CHARACTER WORD IN A FORMAT;
! CONVERT TO UPPER CASE IF SO SPECIFIED
! ====================================================================
SUBROUTINE RDWORD(IUPPER,IXSTAT,IZSTAT,NCHAR,CHANS)
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   Adapted by R. Czerminski of Molecular Simulations, Inc. from
  !  code developed at Merck and Co., Inc. by T. Halgren and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use mmffm
  use stream
  implicit none
  !
  integer iupper, ixstat, izstat, nchar
  !
  CHARACTER(len=80),save :: LINE
  CHARACTER(len=*) CHANS
  LOGICAL RDNEW,OK
  !
  !   SETTING RDNEW=.TRUE. FORCES A NEW INPUT RECORD TO BE READ
  RDNEW=.TRUE.
  RDNEW=.FALSE.
  CALL GTLINE(LINE,IXSTAT,IZSTAT,NCHAR,RDNEW)
  !   LOAD ONLY FIRST TOKEN (WORD) ON THE LINE
  CALL GTWORD(CHANS,LINE,OK,.FALSE.)
  NCHAR=LEN_TRIM(CHANS)
  !   CONVERT TO UPPER CASE IF IUPPER=1
  IF(IUPPER.EQ.1) CALL MUPCASE(CHANS,CHANS)
  !
  RETURN
END SUBROUTINE RDWORD

! ================================================================
! FUNCTION QNAME : MATCH ATOM NUMBER TO ATOM NAME
! ================================================================
CHARACTER(len=*) FUNCTION QNAME(I)
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   Adapted by R. Czerminski of Molecular Simulations, Inc. from
  !  code developed at Merck and Co., Inc. by T. Halgren and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !
  use chm_kinds
  use dimens_fcm
  use psf
  use mmffm
  implicit none

  integer i, i1, i10, i100, i1000
  integer lenq, lenr
  !
  CHARACTER(len=8),save :: QFORM
  CHARACTER(len=1) :: DIGIT(11)=(/'1','2','3','4','5','6','7','8','9','0',' '/)
  !
  IF(I.LE.0.OR.I.GT.MAXAIM) THEN
     QNAME=' '
     RETURN
  ENDIF
  QFORM='SEQNUM'
  IF(QFORM.EQ.'SEQNUM'.AND.I.LE.NATOM) THEN
     QNAME=AtNames(I)
     !
     !   NOW PLACE SEQUENCE NUMBER AFTER THE ATOM NAME IF THIS ATOM
     !   IS IN THE "SUBJECT MOLECULE" (I.LE.NATOM)
     LENQ=LEN_TRIM(QNAME)
     !   NOTE: THE FOLLOWING COMMENTED-OUT CODE APPEARS STRAIGHTFORWARD
     !   AND WORKS ON THE VAX, BUT DOESN'T WORK ON THE IBM BECAUSE ONE
     !   EVIDENTLY CAN'T USE THE I/O SYSTEM (TO DO THE INTERNAL WRITE INTO
     !   THE CHARACTER VARIABLE QSEQ) WHILE USING THE I/O SYSTEM (IN THE
     !   FORTRAN "WRITE(,) ... QNAME(I)" STATEMENT WHICH COMMONLY
     !   INVOKES FUNCTION QNAME.
     !        WRITE(QSEQ,'(A,I3)') ' #',NSEQ(I)
     !        QNAME(LENQ+1:)=QSEQ(1:5)
     !   THEREFORE, WE DO IT THE OLD-FASHION WAY:
     I1000=I/1000
     I100=(I-1000*I1000)/100
     I10=(I-1000*I1000-100*I100)/10
     I1=(I-1000*I1000-100*I100-10*I10)
     IF(I1000.EQ.0) I1000=11
     IF(I100.EQ.0) THEN
        IF(I1000.EQ.11) I100=11
        IF(I1000.NE.11) I100=10
     ENDIF
     IF(I10.EQ.0) THEN
        IF(I100.EQ.11) I10=11
        IF(I100.NE.11) I10=10
     ENDIF
     IF(I1.EQ.0) I1=10
     IF(I.LT.10) THEN
        QNAME(LENQ+1:) = ' #'//DIGIT(I1)
     ELSE IF(I.LT.100) THEN
        QNAME(LENQ+1:) = ' #'//DIGIT(I10)//DIGIT(I1)
     ELSE IF(I.LT.1000) THEN
        QNAME(LENQ+1:) = ' #'//DIGIT(I100)//DIGIT(I10)//DIGIT(I1)
     ELSE
        QNAME(LENQ+1:) = '#'//DIGIT(I1000)//DIGIT(I100)// &
             DIGIT(I10)//DIGIT(I1)
     ENDIF
  ELSE IF(QFORM.EQ.'RESNAME'.OR.I.GT.NATOM) THEN
     !
     !   USE THE "RESIDUE.ATOM_NAME" FORMAT FOR ALL CONTEXT-MOLECULE
     !   ATOMS (I.GT.NATOM) AND, IF QFORM='RESNAME', ALSO FOR
     !   THE SUBJECT-MOLECULE ATOMS
     !
     LENR=LEN_TRIM(RESNAME(I))
     QNAME=RESNAME(I)(1:LENR)//'.'//AtNames(I)
  ENDIF
  !
  RETURN
END FUNCTION QNAME

! =============================================================
! SUBROUTINE QUERY : WRITE A PROMPT REQUESTING INTERACTIVE INPUT,
! AND KEEP CURSOR AT END OF THE PROMPT LINE
! =============================================================
SUBROUTINE  QUERY(UNIT,LINE)
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   Adapted by R. Czerminski of Molecular Simulations, Inc. from
  !  code developed at Merck and Co., Inc. by T. Halgren and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !
  !  Jay Banks 22 Nov 95: added PRNLEV.GE.2 test.  Also added ##INCLUDEs
  !
  use chm_kinds
  use stream
  implicit none
  !
  CHARACTER(len=*) LINE
  integer unit
  !
  !
  IF(UNIT.NE.0.AND.PRNLEV.GE.2) WRITE(UNIT,'(A,$)') LINE
  !
  RETURN
END SUBROUTINE QUERY

! =============================================================
! SUBROUTINE SAYC : Print output character message
! =============================================================
SUBROUTINE SAYC(C)
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   Adapted by R. Czerminski of Molecular Simulations, Inc. from
  !  code developed at Merck and Co., Inc. by T. Halgren and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !
  use chm_kinds
  use stream
  implicit none
  !
  CHARACTER(len=*) C
  integer lenc

  !      -- OUTPUT TO APPROP LOGICAL UNITS
  !      PARAMETER(LFMT=20)
  !      CHARACTER*(LFMT) FMT
  !      LENC = LEN(C)
  !      WRITE(FMT,'(A,I2.2,A)') '(A', LENC, ')'
  LENC=LEN_TRIM(C)
  IF(LENC.GT.0) THEN
     if(prnlev.ge.2) WRITE(OUTU,'(A)') C(1:LENC)
     !     IF(IZ.GT.0) WRITE(IZ,'(A)') C(1:LENC)
  ELSE
     if(prnlev.ge.2) WRITE(OUTU,'(A)') ' '
     !     IF(IZ.GT.0) WRITE(IZ,'(A)') ' '
  ENDIF
  !
  RETURN
END SUBROUTINE SAYC

! =======
! subroutine mupcase : a routine to convert a string to upper case.
! =======
subroutine mupcase( source, destin )
  !***********************************************************************
  ! a routine to convert a string to upper case.
  !***********************************************************************
  use chm_kinds
  implicit none
  !
  !
  !...parameters
  !
  character(len=*)      source
  character(len=*)      destin
  !
  !...variables
  !
  integer dlen, high, ii, kk, low
  character(len=1) :: upparr(26)=(/ &
       'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', &
       'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', &
       'U', 'V', 'W', 'X', 'Y', 'Z'/)
  low  = ichar('a')
  high = ichar('z')
  !
  !...
  !
  dlen = len( destin )
  destin = source
  do ii=1, dlen
     kk = ichar( source(ii:ii) )
     if( kk.ge.low .and. kk.le.high ) then
        destin(ii:ii) = upparr( kk - low + 1 )
     endif
  enddo
  !
  !...
  !
  RETURN
END subroutine mupcase

! ==================================================================
! SUBROUTINE WFLOAT : write a real variable into a character string
! ==================================================================
!
SUBROUTINE WFLOAT(IFORM,ILEN,LINE,VALUE)
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   Adapted by R. Czerminski of Molecular Simulations, Inc. from
  !  code developed at Merck and Co., Inc. by T. Halgren and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !
  use chm_kinds
  implicit none
  !
  integer iform, ilen, ivalue, ixp, nbase
  real(chm_real) value
  CHARACTER(len=*) LINE
  !
  !  FIELD WIDTH 12 .......
  !
  IF(IFORM.EQ.5.AND.ILEN.EQ.12) THEN
     WRITE(LINE(1:ILEN),'(F12.5)',ERR=10) VALUE
     IF(LINE(1:1).NE.'*') RETURN
10   CONTINUE
     WRITE(LINE(1:ILEN),'(1PG12.5)') VALUE
  ELSE IF(IFORM.EQ.4.AND.ILEN.EQ.12) THEN
     WRITE(LINE(1:ILEN),'(F12.4)',ERR=15) VALUE
     IF(LINE(1:1).NE.'*') RETURN
15   CONTINUE
     WRITE(LINE(1:ILEN),'(1PG12.4)') VALUE
  ELSE IF(IFORM.EQ.3.AND.ILEN.EQ.12) THEN
     WRITE(LINE(1:ILEN),'(F12.3)',ERR=16) VALUE
     IF(LINE(1:1).NE.'*') RETURN
16   CONTINUE
     WRITE(LINE(1:ILEN),'(1PG12.3)') VALUE
     !
     !  FIELD WIDTH 10 .......
     !
  ELSE IF(IFORM.EQ.3.AND.ILEN.EQ.10) THEN
     WRITE(LINE(1:ILEN),'(F10.3)',ERR=20) VALUE
     IF(LINE(1:1).NE.'*') RETURN
20   CONTINUE
     WRITE(LINE(1:ILEN),'(1PG10.3)') VALUE
  ELSE IF(IFORM.EQ.4.AND.ILEN.EQ.10) THEN
     WRITE(LINE(1:ILEN),'(F10.4)',ERR=30) VALUE
     IF(LINE(1:1).NE.'*') RETURN
30   CONTINUE
     WRITE(LINE(1:ILEN),'(1PG10.4)',ERR=21) VALUE
     IF(LINE(1:1).NE.'*') RETURN
21   CONTINUE
     WRITE(LINE(1:ILEN),'(1PG10.3)') VALUE
  ELSE IF(IFORM.EQ.5.AND.ILEN.EQ.10) THEN
     WRITE(LINE(1:ILEN),'(F10.5)',ERR=31) VALUE
     IF(LINE(1:1).NE.'*') RETURN
31   CONTINUE
     WRITE(LINE(1:ILEN),'(1PG10.4)',ERR=22) VALUE
     IF(LINE(1:1).NE.'*') RETURN
22   CONTINUE
     WRITE(LINE(1:ILEN),'(1PG10.3)') VALUE
     !
     !  FIELD WIDTH 9 .......
     !
  ELSE IF(IFORM.EQ.4.AND.ILEN.EQ.9) THEN
     WRITE(LINE(1:ILEN),'(F9.4)',ERR=50) VALUE
     IF(LINE(1:1).NE.'*') RETURN
50   CONTINUE
     WRITE(LINE(1:ILEN),'(1PG9.3)',ERR=55) VALUE
     IF(LINE(1:1).NE.'*') RETURN
55   CONTINUE
     WRITE(LINE(1:ILEN),'(1PG9.2)') VALUE
  ELSE IF(IFORM.EQ.3.AND.ILEN.EQ.9) THEN
     WRITE(LINE(1:ILEN),'(F9.3)',ERR=51) VALUE
     IF(LINE(1:1).NE.'*') RETURN
51   CONTINUE
     WRITE(LINE(1:ILEN),'(1PG9.3)',ERR=56) VALUE
     IF(LINE(1:1).NE.'*') RETURN
56   CONTINUE
     WRITE(LINE(1:ILEN),'(1PG9.2)') VALUE
  ELSE IF(IFORM.EQ.2.AND.ILEN.EQ.9) THEN
     WRITE(LINE(1:ILEN),'(F9.2)',ERR=57) VALUE
     IF(LINE(1:1).NE.'*') RETURN
57   CONTINUE
     WRITE(LINE(1:ILEN),'(1PG9.2)') VALUE
  ELSE IF(IFORM.EQ.1.AND.ILEN.EQ.9) THEN
     WRITE(LINE(1:ILEN),'(F9.1)',ERR=60) VALUE
     IF(LINE(1:1).NE.'*') RETURN
60   CONTINUE
     WRITE(LINE(1:ILEN),'(1PG9.1)') VALUE
     !
     !  FIELD WIDTH 8 .......
     !
  ELSE IF(IFORM.EQ.3.AND.ILEN.EQ.8) THEN
     WRITE(LINE(1:ILEN),'(F8.3)',ERR=70) VALUE
     IF(LINE(1:1).NE.'*') RETURN
70   CONTINUE
     WRITE(LINE(1:ILEN),'(1PG8.2)',ERR=80) VALUE
     IF(LINE(1:1).NE.'*') RETURN
80   CONTINUE
     WRITE(LINE(1:ILEN),'(1PG8.1)',ERR=90) VALUE
     IF(LINE(1:1).NE.'*') RETURN
90   CONTINUE
     IXP=LOG10(ABS(VALUE))+1
     NBASE=5
     IF(VALUE.LT.0.) NBASE=4
     IVALUE=VALUE*10.**(NBASE-IXP)
     WRITE(LINE(1:ILEN),'(I5,A,I2.2)') IVALUE,'E',IXP-NBASE
  ELSE IF(IFORM.EQ.2.AND.ILEN.EQ.8) THEN
     WRITE(LINE(1:ILEN),'(F8.2)',ERR=71) VALUE
     IF(LINE(1:1).NE.'*') RETURN
71   CONTINUE
     WRITE(LINE(1:ILEN),'(1PG8.2)',ERR=81) VALUE
     IF(LINE(1:1).NE.'*') RETURN
81   CONTINUE
     WRITE(LINE(1:ILEN),'(1PG8.1)',ERR=91) VALUE
     IF(LINE(1:1).NE.'*') RETURN
91   CONTINUE
     IXP=LOG10(ABS(VALUE))+1
     NBASE=5
     IF(VALUE.LT.0.) NBASE=4
     IVALUE=VALUE*10.**(NBASE-IXP)
     WRITE(LINE(1:ILEN),'(I5,A,I2.2)') IVALUE,'E',IXP-NBASE
  ELSE IF(IFORM.EQ.1.AND.ILEN.EQ.8) THEN
     WRITE(LINE(1:ILEN),'(F8.1)',ERR=72) VALUE
     IF(LINE(1:1).NE.'*') RETURN
     RETURN
72   CONTINUE
     WRITE(LINE(1:ILEN),'(1PG8.2)',ERR=82) VALUE
     IF(LINE(1:1).NE.'*') RETURN
82   CONTINUE
     WRITE(LINE(1:ILEN),'(1PG8.1)',ERR=92) VALUE
     IF(LINE(1:1).NE.'*') RETURN
92   CONTINUE
     IXP=LOG10(ABS(VALUE))+1
     NBASE=5
     IF(VALUE.LT.0.) NBASE=4
     IVALUE=VALUE*10.**(NBASE-IXP)
     WRITE(LINE(1:ILEN),'(I5,A,I2.2)') IVALUE,'E',IXP-NBASE
     !
     !  FIELD WIDTH 6 .......
     !
  ELSE IF(IFORM.EQ.3.AND.ILEN.EQ.6) THEN
     WRITE(LINE(1:ILEN),'(F6.3)',ERR=170) VALUE
     IF(LINE(1:1).NE.'*') RETURN
170  CONTINUE
     WRITE(LINE(1:ILEN),'(F6.2)',ERR=180) VALUE
     IF(LINE(1:1).NE.'*') RETURN
180  CONTINUE
     WRITE(LINE(1:ILEN),'(F6.1)',ERR=190) VALUE
     IF(LINE(1:1).NE.'*') RETURN
190  CONTINUE
     WRITE(LINE(1:ILEN),'(F6.0)',ERR=200) VALUE
     IF(LINE(1:1).NE.'*') RETURN
200  CONTINUE
     WRITE(LINE(1:ILEN),'(I6)',ERR=210) VALUE
     IF(LINE(1:1).NE.'*') RETURN
210  CONTINUE
     !
     !  IF HERE, THE NUMBER JUST WON'T FIT!  TOUGH LUCK.
     !
  ELSE IF(IFORM.EQ.2.AND.ILEN.EQ.6) THEN
     WRITE(LINE(1:ILEN),'(F6.2)',ERR=181) VALUE
     IF(LINE(1:1).NE.'*') RETURN
181  CONTINUE
     WRITE(LINE(1:ILEN),'(F6.1)',ERR=191) VALUE
     IF(LINE(1:1).NE.'*') RETURN
191  CONTINUE
     WRITE(LINE(1:ILEN),'(F6.0)',ERR=201) VALUE
     IF(LINE(1:1).NE.'*') RETURN
201  CONTINUE
     WRITE(LINE(1:ILEN),'(I6)',ERR=211) VALUE
     IF(LINE(1:1).NE.'*') RETURN
211  CONTINUE
     !
     !  IF HERE, THE NUMBER JUST WON'T FIT!  TOUGH LUCK.
     !
  ELSE IF(IFORM.EQ.1.AND.ILEN.EQ.6) THEN
     WRITE(LINE(1:ILEN),'(F6.1)',ERR=192) VALUE
     IF(LINE(1:1).NE.'*') RETURN
192  CONTINUE
     WRITE(LINE(1:ILEN),'(F6.0)',ERR=202) VALUE
     IF(LINE(1:1).NE.'*') RETURN
202  CONTINUE
     WRITE(LINE(1:ILEN),'(I6)',ERR=212) VALUE
     IF(LINE(1:1).NE.'*') RETURN
212  CONTINUE
     !
     !  IF HERE, THE NUMBER JUST WON'T FIT!  TOUGH LUCK.
     !
  ENDIF
  !
  RETURN
END SUBROUTINE WFLOAT
#else /**/
SUBROUTINE NULL_parse_MM
  RETURN
END SUBROUTINE NULL_parse_MM
#endif 


