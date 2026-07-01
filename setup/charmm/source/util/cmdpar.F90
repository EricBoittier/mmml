module cmdpar
  use chm_kinds

  implicit none
  !BEGIN DOC
  !     Parameters for command line substitution.
  !# <caves>-Jan-15-1994 (Leo Caves) Extended parameter table.
  !
  ! * SYNTAX * 
  ! # setting parameters
  ! SET <token> [=] <value>
  ! <token>:== string
  ! <value>:== string
  ! # substituting parameters on the command line
  ! [ @<token> ]  
  ! [ @{<token>} ] (the brackets "protect" token from surrounding alphanumerics)
  !
  ! * PARAMETERS *
  ! # parameter: data-type: description
  ! MAXPAR: integer: the number of entries in the parameter table.
  ! MXTLEN: integer: maximum size of the token string 
  ! MXVLEN: integer: maximum size of the value string 
  !
  !
  ! # array: data-type: description
  ! ( CPARC the characters  )
  ! TOKNAM(MAXPAR): character*(MXTLEN): the token strings 
  ! VALNAM(MAXPAR): character*(MXVLEN): the value strings 
  ! ( CPARI the integers. )
  ! NUMPAR: integer: number of entries in the parameter table
  ! TOKLEN(MAXPAR): integer: the token string length
  ! VALLEN(MAXPAR): integer: the value string length
  !
  ! * FUNCTIONS * (declared in exfunc.fcm)
  ! # prototype
  ! INTEGER FUNCTION PARNUM(token,tokLen) 
  ! # passed arguments
  ! integer tokLen            (untouched on exit)
  ! character*(tokLen) token  (untouched on exit)
  ! # description
  ! This function performs a lookup into the parameter table based on
  ! the passed token string. The return value is the index into the
  ! parameter table. If token is not found, return value is -1. 
  ! # location in CHARMM source: util/cmdpar.src 
  !
  ! # prototype
  ! INTEGER FUNCTION PARINS(token,tokLen,value,valLen)   
  ! # passed arguments
  ! integer tokLen,valLen                 (untouched on exit)
  ! character token*tokLen, value*valLen  (untouched on exit)
  ! # description
  ! Install a token into the parameter table, with its associated value.
  ! If successful return 0, if error encountered return -1.
  ! # location in CHARMM source: util/cmdpar.src 
  !
  ! # prototype
  !  INTEGER FUNCTION PARSUB ( COMLYN, COMMAX, COMLEN, QPRINT )
  ! # passed arguments
  ! integer commax, comlen  (comlen modified on exit)
  ! character comlyn*commax (modified on exit)
  ! logical qprint          (untouched on exit)
  ! # description
  ! perform substitution of parameters (signalled by '@') in passed string.
  ! string and its length are modified appropriately.
  ! If successful return 0, if error encountered return -1.
  ! # location in CHARMM source: util/cmdpar.src 
  !
  ! # prototype
  !  INTEGER FUNCTION PARMAN ( COMLYN, COMLEN )
  ! # passed arguments
  ! integer comlen   (modified on exit)
  ! character comlyn (modified on exit)
  ! # description
  ! Parameter manipulation of form <par> <asOp> [ <unOp> ] arg1 [ <biOp> arg2 ]
  ! (simple numerical modification of parameters)
  ! see code for details.
  ! If successful return 0, if error encountered return -1.
  ! # location in CHARMM source: util/cmdpar.src 
  !END DOC
  !
  ! * DATA STRUCTURE *
  ! dimensioning: max size of table and of token and value strings
  INTEGER,PARAMETER :: MAXPAR=10240,MXTLEN=32,MXVLEN=128

  ! the tokens
  CHARACTER(len=MXTLEN),save :: TOKNAM(MAXPAR)

  ! the values
  CHARACTER(len=MXVLEN),save :: VALNAM(MAXPAR)

  ! lengths of token and value strings
  integer,dimension(maxpar),save :: toklen,vallen

  ! number of current symbols in table
  integer,save :: numpar
  
!==================================================================
contains

  !-------------------------------------------------------------------------
  !               CMDPAR_INIT
  !-------------------------------------------------------------------------
  subroutine cmdpar_init()
    integer :: i
    numpar = 0
    do i=1,maxpar
       toklen(i)=0
       vallen(i)=0
       toknam(i) = ' '
       valnam(i) = ' '
    enddo
    return
  end subroutine cmdpar_init

  !-------------------------------------------------------------------------
  !               PARNUM
  !-------------------------------------------------------------------------
  integer function parnum(token,length)
    !# <caves>-Aug-13-1993 (Leo Caves)
    !
    !     Is there an entry in the parameter table associated with passed token ?
    !     If found : return pointer into parameter table
    !     else       return -1
    !
    ! passed
    character(len=*) token
    integer length
    ! local
    logical qfnd
    integer ipar,tlen
    ! begin
    qfnd = .false. 
    tlen = min(length,mxtlen)
    parnum = -1
    if(numpar == 0) return
    ! do the search (linear lookup!)
    ipar = 1
    ! do while
10  if ( (.not. qfnd) .and. (ipar  <=  numpar)) then
       if(tlen == toklen(ipar)) then
          if(token(1:tlen) == toknam(ipar)(1:tlen)) then
             qfnd = .true. 
             parnum=ipar
          endif
       endif
       ipar = ipar + 1
       goto 10
    endif
    ! end do while
    return
  end function parnum
  

  !-------------------------------------------------------------------------
  !               PARINS
  !-------------------------------------------------------------------------
  integer function parins(token,tlen,value,length)
    !# <caves>-Aug-13-1993 (Leo Caves)
    !
    !     Install a parameter token, with its associate value.
    !     If the parameter exists in the table, then replace its value.

    use stream

    ! passed.
    character(len=*) token, value
    integer tlen, length,i

    ! functions.
    ! local
    integer ipar

    ! begin
    ! check token
    if(tlen >= mxtlen) then
       CALL WrnDie(1,'<ParIns>','Token may have been truncated.')
       tlen = mxtlen
    elseif(tlen <= 0)  then
       CALL WrnDie(1,'<ParIns>','Token empty: Ignored.') 
       tlen = 0
       parins = -1
       return
    endif
    ! check value
    if(length >= mxvlen) then
       CALL WrnDie(1,'<ParIns>','Value may have been truncated.')
       length = mxvlen
    elseif(length < 0)  then
       ! empty value is valid.
       length = 0
    endif

    ipar = parnum(token,tlen)

    if(ipar < 0)then
       ! install
       if(numpar+1 > maxpar)then
          CALL WrnDie(0,'<ParIns>','Parameter table full. Ignored.')
          parins = -1
          return
       endif
       numpar = numpar + 1

       ipar = numpar
       toknam(ipar) = token(1:tlen)
       toklen(ipar) = tlen

    endif
    ! replace value
    vallen(ipar) = length
    if(length > 0)then
       valnam(ipar) = value(1:length)
    else
       valnam(ipar) = ' '
    endif
    !
    if(prnlev >= 2) call parwri(outu,ipar,1) 
    parins = 1
    ! exit.
    return
  end function parins


  !-------------------------------------------------------------------------
  !               PARMAN
  !-------------------------------------------------------------------------
  INTEGER FUNCTION PARMAN(comLyn, comLen)
    !# <caves>-Mar-30-1993 (Leo Caves)
    !
    !...Simple parameter manipulation
    !   Syntax:
    !   LET par <asOp> [ <unOp> ] arg1 [ <biOp> arg2 ]
    !   for definitions see DATA statements below.
    !
    !   Of course its not a full recursive expression evaluator
    !   But I hope that its useful.
    !
    !...Global Variables.
  use dimens_fcm
  use chm_kinds
  use string
  use exfunc
  use number
  use stream
    implicit none
    !
    !...Passed Variables.
    character(len=*) comLyn
    INTEGER comLen
    !
    !...Local Variables. 
    LOGICAL qNew, qUn, qBi, qAs
    INTEGER uOp,bOp,aOp,iLen,iPar
    real(chm_real) lval, expr, arg1, arg2
    CHARACTER(len=80) unOps, biOps, asOps
    ! a problem with DATA initialization below
    !      DATA unOps /'SINE|COSI|TANG|EXPO|LOG1|LOGE|SQRT|'//
    !     &            'ABSO|INTE|NINT|ASIN|ACOS|ATAN|MAXI|MINI|'/
    DATA biOps /'* |/ |- |+ |**|% '/
    DATA asOps /'= |+=|-=|*=|/='/
    !
    !...Begin Executable Code. 
    !
    ! initialize
    unOps = 'SINE|COSI|TANG|EXPO|LOG1|LOGE|SQRT|' // &
         'ABSO|INTE|NINT|ASIN|ACOS|ATAN|MAXI|MINI|'
    qNew = .TRUE.
    qAs  = .FALSE. 
    qUn  = .FALSE.
    qBi  = .FALSE.
    expr = zero
    lval = zero
    !
    ! grab the parameter which is to be modified 
    CALL NEXTWD(COMLYN,COMLEN,SWDTCH,SWDMAX,SWDLEN)
    IF(SWDLEN < 1) THEN
       PARMAN = -1
       RETURN 
    ENDIF
    !
    IPAR = PARNUM(SWDTCH,SWDLEN)
    IF(IPAR <= 0) THEN 
       CALL WrnDie(0,'<ParMan>', &
            'Non-existent parameter. '//swdtch(1:swdlen))
       PARMAN = -1
       RETURN
    ENDIF
    LVAL=DECODF(VALNAM(IPAR),VALLEN(IPAR))
    
    ! parse Assignment operator
    CALL TRIME(COMLYN,COMLEN)
    IF(COMLEN < 1)THEN
       CALL WrnDie(0,'<ParMan>','Bad no. of args')
       PARMAN = -1
       RETURN
    ENDIF
    CALL NEXTWD(COMLYN,COMLEN,SWDTCH,SWDMAX,SWDLEN)
    iLen = MAX(1,MIN(2,swdlen))
    aop = index(asops, swdtch(1:ilen))
    IF (aOp > 0) THEN 
       qAs = .TRUE.
       aOp = int(aOp/3) + 1
    ELSE
       CALL WrnDie(0,'<ParMan>', &
            'Bad assignment operator. '//swdtch(1:swdlen))
       PARMAN = -1
       RETURN
    ENDIF
    
    ! grab next token
    CALL TRIME(COMLYN,COMLEN)
    IF(COMLEN < 1)THEN
       CALL WrnDie(0,'<ParMan>','Bad no. of args')
       PARMAN = -1
       RETURN
    ENDIF
    !
    CALL NEXTWD(COMLYN,COMLEN,SWDTCH,SWDMAX,SWDLEN)
    !
    CALL TRIME(COMLYN,COMLEN) 
    ! if no other args assume simple assignment
    qAs = COMLEN < 1
    IF (qAs) THEN
       expr=DECODF(SWDTCH,SWDLEN)
    ELSE
       !
       ! not a simple assignment so check for unary or binary op.
       !
       ! look for unary ops in the function table
       iLen = MAX(3,MIN(4,swdlen))
       uop = index(unops, swdtch(1:ilen))
       IF (uOp > 0) THEN
          ! unary operator
          uOp = int(uOp/5) + 1
          qUn = .TRUE.
          ! grab argument 
          CALL TRIME(COMLYN,COMLEN)
          IF(COMLEN < 1)THEN
             CALL WrnDie(0,'<ParMan>','Bad no. of args')
             PARMAN = -1
             RETURN
          ENDIF
          CALL NEXTWD(COMLYN,COMLEN,SWDTCH,SWDMAX,SWDLEN)
          arg1=DECODF(SWDTCH,SWDLEN)
          ! any more args for un-op ( I am breaking my definitions here !)
          ! this is done to put in simple max/min functions
          CALL TRIME(COMLYN,COMLEN)
          IF(COMLEN > 0)THEN
             CALL NEXTWD(COMLYN,COMLEN,SWDTCH,SWDMAX,SWDLEN)
             arg2=DECODF(SWDTCH,SWDLEN)
          ENDIF
       ELSE
          ! assume binary op. 
          ! arg 1
          arg1=DECODF(SWDTCH,SWDLEN)
          !
          ! pick up the binary operator
          CALL TRIME(COMLYN,COMLEN)
          IF(COMLEN < 1)THEN
             CALL WrnDie(0,'<ParMan>','Bad no. of args')
             PARMAN = -1
             RETURN
          ENDIF
          !
          CALL NEXTWD(COMLYN,COMLEN,SWDTCH,SWDMAX,SWDLEN)
          iLen = MAX(1,MIN(2,swdlen))
          bop = index(biops, swdtch(1:ilen))
          IF (bOp > 0) THEN
             bOp = int(bOp/3) + 1
             qBi = .TRUE.
          ELSE
             CALL WrnDie(0,'<ParMan>', &
                  'Bad binary operator. '//swdtch(1:swdlen))
             PARMAN = -1
             RETURN
          ENDIF
          ! arg 2
          CALL TRIME(COMLYN,COMLEN)
          IF(COMLEN < 1)THEN
             CALL WrnDie(0,'<ParMan>','Bad no. of args')
             PARMAN = -1
             RETURN
          ENDIF
          CALL NEXTWD(COMLYN,COMLEN,SWDTCH,SWDMAX,SWDLEN)
          arg2=DECODF(SWDTCH,SWDLEN)
          !
       ENDIF
    ENDIF ! qAs
    !
    !     unOps = 'SINE:COSI:TANG:EXPO:LOG1:LOGE:SQRT:ABSO:INTE:NINT'  
    !
    IF (qUn) THEN
       IF     (uOp ==  1) THEN
          expr = SIN(arg1)
       ELSEIF (uOp ==  2) THEN
          expr = COS(arg1)
       ELSEIF (uOp ==  3) THEN
          expr = TAN(arg1)
       ELSEIF (uOp ==  4) THEN
          expr = EXP(arg1)
       ELSEIF (uOp ==  5) THEN
          expr = LOG10(arg1)
       ELSEIF (uOp ==  6) THEN
          expr = LOG(arg1)
       ELSEIF (uOp ==  7) THEN
          IF(arg1 >= zero) THEN
             expr = SQRT(arg1)
          ELSE  
             CALL WrnDie(0,'<ParMan>','Neg. arg. to SQRT')
             PARMAN = -1
             RETURN 
          ENDIF
       ELSEIF (uOp ==  8) THEN
          expr = ABS(arg1)
       ELSEIF (uOp ==  9) THEN
          expr = INT(arg1)
       ELSEIF (uOp == 10) THEN
          expr = NINT(arg1)
       ELSEIF ((uOp >= 11) .AND. (uOp <= 13)) THEN
          IF(abs(arg1) <= one) THEN
             IF(uOp == 11) expr = ASIN(arg1)
             IF(uOp == 12) expr = ACOS(arg1)
             IF(uOp == 13) expr = ATAN(arg1)
          ELSE  
             CALL WrnDie(0,'<ParMan>', &
                  'Bad arg to arc trig. function')
             PARMAN = -1
             RETURN 
          ENDIF
       ELSEIF (uOp == 14) THEN
          expr = MAX(arg1,arg2)
       ELSEIF (uOp == 15) THEN
          expr = MIN(arg1,arg2)
       ELSE
          CALL WrnDie(0,'<ParMan>','Invalid Unary opcode.')
          PARMAN = -1
          RETURN 
       ENDIF
    ENDIF
    !
    !     biOps = '* :/ :- :+ :**:% '
    IF (qBi) THEN
       IF     (bOp ==  1) THEN
          expr = arg1 * arg2 
       ELSEIF (bOp ==  2) THEN
          IF(arg2 /= zero) THEN     ! Bugfix B000329.ddb arg1 -> arg2
             expr = arg1 / arg2
          ELSE  
             CALL WrnDie(0,'<ParMan>','Divide by zero.')
             PARMAN = -1
             RETURN
          ENDIF
       ELSEIF (bOp ==  3) THEN
          expr = arg1 - arg2
       ELSEIF (bOp ==  4) THEN
          expr = arg1 + arg2 
       ELSEIF (bOp ==  5) THEN
          expr = arg1 ** arg2 
       ELSEIF (bOp ==  6) THEN
          expr = mod(arg1,arg2)
       ELSE
          CALL WrnDie(0,'<ParMan>','Invalid Binary opcode.')
          PARMAN = -1
          RETURN 
       ENDIF
    ENDIF
    !
    ! make the final assignment
    !     asOps = '= |+=|-=|*=|/='  
    !      IF (qAs) THEN
    IF     (aOp ==  1) THEN
       lval = expr 
    ELSEIF (aOp ==  2) THEN
       lval = lval + expr 
    ELSEIF (aOp ==  3) THEN
       lval = lval - expr 
    ELSEIF (aOp ==  4) THEN
       lval = lval * expr 
    ELSEIF (aOp ==  5) THEN
       IF(expr /= zero) THEN
          lval = lval / expr 
       ELSE  
          CALL WrnDie(0,'<ParMan>','Divide by zero.')
          PARMAN = -1
          RETURN
       ENDIF
    ELSE
       CALL WrnDie(0,'<ParMan>','Invalid Assignment opcode.')
       PARMAN = -1
       RETURN
    ENDIF
    !
    ! restore the parameter value
    !
    CALL ENCODF(LVAL,VALNAM(IPAR),MXVLEN,VALLEN(IPAR))
    !
    IF(PRNLEV >= 2) CALL PARWRI(OUTU,IPAR,1)
    !
    PARMAN = 0
    !...Exit.
    RETURN
  END FUNCTION PARMAN
  
  !-------------------------------------------------------------------------
  !               PARWRI
  !-------------------------------------------------------------------------
  SUBROUTINE PARWRI(UNIT, IPAR, MODE)
    !# <caves>-Aug-19-1993 (Leo Caves)
    !
    !...Simply print the specified parameter (token and value) to the specified unit
    !   Function provides a standard output format for parameter.
    !   Mode indicates whether the parameter has been substituted or set
    !   mode: 1 set ; mode: 0 substituted 
    !
  use chm_kinds
    implicit none
    !...Global Variables.
    !
    !...Passed Variables.
    INTEGER UNIT,IPAR,MODE
    !...Local
    character(len=4) AS
    !...Begin Executable Code. 
    !
    !
    IF( (IPAR < 0) .OR. (IPAR > NUMPAR) ) THEN
       CALL WRNDIE(-1,'<ParWri>','Bad parameter number passed.')
       RETURN
    ELSE
       !
       IF     (MODE == 0) THEN
          AS = ' -> '
       ELSEIF (MODE == 1) THEN
          AS = ' <- '
       ELSE
          AS = ' == '
       ENDIF
       !
       IF(VALLEN(IPAR) > 0)THEN
          WRITE(UNIT,'(6A)') &
               ' Parameter: ',TOKNAM(IPAR)(1:TOKLEN(IPAR)),AS,'"', &
               VALNAM(IPAR)(1:VALLEN(IPAR)),'"'
       ELSE
          WRITE(UNIT,'(4A)') &
               ' Parameter: ', TOKNAM(IPAR)(1:TOKLEN(IPAR)),AS,'"" <empty>'
       ENDIF
    ENDIF
    
    !...Exit.
    RETURN
  END SUBROUTINE PARWRI
  
  !-------------------------------------------------------------------------
  !               PARSE1
  !-------------------------------------------------------------------------
  SUBROUTINE PARSE1(COMLYN, COMMAX, COMLEN, QPRINT)
    !
    !  Pete Steinbach's precursor to PARSUB, called to first substitite
    !  parameters preceeded by '@@'.  Allows parameters to reference
    !  array elements.                         Mar 20, 1998
    !
  use chm_kinds
  use string
    implicit none
    !...Passed Variables.
    character(len=*) COMLYN
    INTEGER COMMAX,COMLEN
    LOGICAL QPRINT
    
    !...Local.
    INTEGER IPAR,I,J,IATAT,SUBLEN
    ! parameter substitution operators
    character(len=2) ATAT 
    DATA ATAT/'@@'/
    
    iatat = index(comlyn(1:comlen), atat(1:2))
    IF(IATAT > 0) THEN
       I = COMLEN-1
50     I = I-1
       IF(COMLYN(I:I+1)  ==  ATAT) THEN
          SUBLEN = COMLEN-I
          IPAR = PARSUB(COMLYN(I+1:COMLEN),COMMAX-I,SUBLEN,QPRINT)
          IF(IPAR < 0) CALL WrnDie(0,'<PARSE1>', &
               'Error in parameter substitution.')
          ! Overwrite the first '@'
          DO J = 1,SUBLEN
             COMLYN(I-1+J:I-1+J) = COMLYN(I+J:I+J)
          ENDDO
          COMLEN = I-1+SUBLEN
       ENDIF
       IF(I  >  IATAT) GOTO 50
    ENDIF
    IPAR = PARSUB(COMLYN,COMMAX,COMLEN,QPRINT)
    IF(IPAR < 0) CALL WrnDie(0,'<PARSE1>', &
         'Error in parameter substitution.')
    RETURN
  END SUBROUTINE PARSE1
  
  !-------------------------------------------------------------------------
  !               PARSUB
  !-------------------------------------------------------------------------
  INTEGER FUNCTION PARSUB(COMLYN, COMMAX, COMLEN, QPRINT)
    !# <caves>-Aug-19-1993 (Leo Caves)
    !
    !...Perform parameter substitutions on the passed string.
    !
  use dimens_fcm
  use chm_kinds
  use string
  use stream
  use exfunc
    implicit none
    !...Global Variables.
    
    
    !...Passed Variables.
    character(len=*) COMLYN
    INTEGER COMMAX,COMLEN
    LOGICAL QPRINT

    !...Local.
    CHARACTER(len=MXTLEN) TOKN
    INTEGER TLEN,IPAR,IBEG,IEND,IATMK,IENDMK
    character(len=1) ATMARK,SYMBRA,SYMKET
    ! parameter substitution operators
    INTEGER iParOp
    INTEGER IQUES, IHASH
    PARAMETER(IQUES=1,IHASH=2) 
    character(len=2) PAROP 
    DATA ATMARK/'@'/,SYMBRA/'{'/,SYMKET/'}'/
    DATA PAROP /'?#'/
    !
    !...Begin Executution.
    !
    PARSUB = 0
    ! BEGIN { @ parameter substitution }
    ! is there a parameter to substitute ?
    ! DO WHILE (IATMK > 0)
300 CONTINUE
    iatmk = index(comlyn(1:comlen), atmark(1:1))
    IF (IATMK > 0) THEN
       iBeg = IATMK + 1
       ! check for parameter operator
       iParOp = INDX(parOp,2,COMLYN(iBeg:iBeg),1)
       IF(iParOp > 0) iBeg = iBeg + 1
       !
       ! deal with brace delimited token 
       IF(COMLYN(iBeg:iBeg) == SYMBRA)THEN
          iBeg = iBeg + 1
          iendmk = index(comlyn(1:comlen), symket(1:1))
          IF(IENDMK > 0) THEN
             iEnd = IENDMK - 1
          ELSE
             CALL WrnDie(-1,'<RDCMND>', &
                  'Bad parameter syntax: No closing brace') 
             COMLEN = 0
             PARSUB = -1
             RETURN
          ENDIF
       ELSE
          IPAR = TOKEND(COMLYN(iBeg:COMLEN),COMLEN-iBeg+1)
          IEND = iBeg + IPAR - 1
          IENDMK = IEND 
       ENDIF
       ! grab token
       TOKN = COMLYN(iBeg:iEnd)
       TLEN = iEnd - iBeg + 1 
       ! lookup value based on token
       IPAR = PARNUM(TOKN,TLEN)
       ! <-------------------------------------------------- COMPATIBILITY
       IF(IPAR < 0)THEN
          ! backward compatibility (@a-z or @A-Z): lookup for single character token
          IPAR = PARNUM(TOKN,1)
          IF (IPAR > 0) THEN
             IENDMK = iBeg 
             TLEN = 1
          ELSE
             ! <-------------------------------------------------- COMPATIBILITY
             IF(iParOp /= IQUES) THEN
                IF(WRNLEV >= 2) WRITE(OUTU,*)  &
                     ' * WARNING * <PARSUB> Command ignored.', &
                     ' Token not found: >',TOKN(1:TLEN),'<'
                COMLEN=0
                PARSUB = -1
                RETURN
             ENDIF
          ENDIF
       ENDIF
       ! do the substitution.
       ! capture remainder of command line into temporary buffer
       CALL COPSUB(SCRTCH,SCRMAX,SCRLEN,COMLYN,IENDMK+1,COMLEN)
       ! reduce command line appropriately
       COMLEN=IATMK - 1
       ! substitute the parameter value (if it has a length)
       IF(iParOp == IQUES) THEN
          ! value substition based on question mark operator: is token present ?
          IF(IPAR > 0) THEN
             TOKN = '1' 
          ELSE
             TOKN = '0' 
          ENDIF
          CALL ADDST(COMLYN,COMMAX,COMLEN,TOKN,1) 
          ! diagnostic ?
       ELSE
          ! normal value substitution
          IF(VALLEN(IPAR) > 0) CALL ADDST(COMLYN,COMMAX,COMLEN, &
               VALNAM(IPAR),VALLEN(IPAR))
          IF (QPRINT .AND. PRNLEV >= 3) CALL PARWRI(OUTU,IPAR,0) 
       ENDIF
       ! concatenate the remainder of the command line (back from the scratch string)
       CALL ADDST(COMLYN,COMMAX,COMLEN,SCRTCH,SCRLEN)
       !
       GOTO 300
    ENDIF
    ! ENDDO WHILE (IPAR)
    ! END { @ parameter substitution }
    !
    !...Exit
    PARSUB = 0
    RETURN
  END FUNCTION PARSUB
  
  INTEGER FUNCTION TOKEND(TOKEN,TOKLEN)
    !# <caves>-Aug-26-1993 (Leo Caves)
    !
    !...Find the position of the end of a token. 
    !   A token is delimited by any non-alphanumeric.
    !
  use chm_kinds
    implicit none
    !...Global Variables.
    !
    !...Passed Variables.
    character(len=*) TOKEN
    INTEGER TOKLEN
    !
    !...Local Variables. 
    LOGICAL qSTEP
    INTEGER IPOS
    !
    !...Begin Executable Code. 
    !
    TOKEND = 0

    ! dispatch bad length
    IF(TOKLEN < 1) RETURN
    ! creep forward until a non-alphanueric is found 
    IPOS = 1
    do while( IPOS <= TOKLEN )
       IF(QTOKDEL(TOKEN(IPOS:IPOS))) then
          exit
       endif
       IPOS = IPOS+1
    enddo

    TOKEND = IPOS - 1
    !...Exit.
    RETURN
  END FUNCTION TOKEND
  !
  LOGICAL FUNCTION QTOKDEL(CH)
    !# <caves>-Aug-26-1993 (Leo Caves)
    !
    !...Is the passed character a token delimiter ? (i.e. a non-alphanumeric)
    !
  use chm_kinds
  use string
    implicit none
    !...Global Variables.
    !
    !...Passed Variables.
    CHARACTER(len=1) CH

    !...Local.
    INTEGER IUND,IHYP
    !
    !...Begin Executable Code. 
    !
    QTOKDEL = .TRUE.
    !
    IF ( QDIGIT(CH) .OR. QALPHA(CH) ) QTOKDEL = .FALSE.
    !
    !...Exit.
    RETURN
  END FUNCTION QTOKDEL
end module cmdpar

