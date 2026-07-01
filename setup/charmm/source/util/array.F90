#if KEY_GAMESS==1
#endif 
  !
  SUBROUTINE AVALUE(MAP,ARRAY,DIM,MARK)
    !-----------------------------------------------------------------------
    !     Routine maps values of "ARRAY" using the "MAP" function.
    !     Warning: MAP and ARRAY should be well defined, no checking
    !     is done. This routine assumes that ARRAY has INTEGER
    !     elements.
    !
    !      23-JAN-83 Axel Brunger
    !
  use chm_kinds
    implicit none
    INTEGER   DIM,MAP(*)
    INTEGER ARRAY(*)
    INTEGER   MARK
    INTEGER   I
    !
    DO I=1,DIM
       IF (ARRAY(I) == MARK) THEN
          ! ARRAY(I)=MARK ! hwm220723: Why re-assign?
          continue ! hwm220723: Simply continue
       ELSE IF (ARRAY(I) > 0) THEN
          ARRAY(I)=MAP(ARRAY(I))
       ELSE IF (ARRAY(I) < 0) THEN
          ARRAY(I)=-MAP(-ARRAY(I))
       ENDIF
    ENDDO
    RETURN
  END SUBROUTINE AVALUE

  SUBROUTINE AINDEX(INVMAP,ARRAY,DIM,WORK)
    !-----------------------------------------------------------------------
    !     Routine maps indices of "ARRAY" using "INVMAP" function.
    !     First, result is copied to "WORK" array and then back to "ARRAY".
    !     Warning: no checking is done for correct dimensions, etc.
    !     Routine applies to REAL arrays.
    !
    !      23-JAN-83 Axel Brunger
    !
  use chm_kinds
    implicit none
    INTEGER   DIM,INVMAP(*)
    real(chm_real)    ARRAY(*),WORK(*)
    INTEGER   I
    !
    DO I=1,DIM
       WORK(I)=ARRAY(INVMAP(I))
    ENDDO
    DO I=1,DIM
       ARRAY(I)=WORK(I)
    ENDDO
    RETURN
  END SUBROUTINE AINDEX

  SUBROUTINE AINDXC(INVMAP,ARRAY,DIM,WORK)
    !-----------------------------------------------------------------------
    !     Routine maps indices of "ARRAY" using "INVMAP" function.
    !     First, result is copied to "WORK" array and then back to "ARRAY".
    !     Warning: no checking is done for correct dimensions, etc.
    !     Routine applies to character(len=4) arrays.
    !
    !      23-JAN-83 Axel Brunger
    !      15-JAN-2003 Youngdo Won, expansion of character*4 to character*8
  use chm_kinds
    implicit none
    INTEGER   DIM,INVMAP(*)
    character(len=*) ARRAY(*),WORK(*)
    INTEGER   I
    !
    DO I=1,DIM
       WORK(I)=ARRAY(INVMAP(I))
    ENDDO
    DO I=1,DIM
       ARRAY(I)=WORK(I)
    ENDDO
    RETURN
  END SUBROUTINE AINDXC

  SUBROUTINE AINDXL(INVMAP,ARRAY,DIM,WORK)
    !-----------------------------------------------------------------------
    !     Same routine as AINDEX. Applies to LOGICAL arrays (needed for DRUDES)
    !
  use chm_kinds
    implicit none
    INTEGER   DIM,INVMAP(*)
    LOGICAL   ARRAY(*),WORK(*)
    INTEGER   I
    !  
    DO I=1,DIM
       WORK(I)=ARRAY(INVMAP(I))
    ENDDO
    DO I=1,DIM 
       ARRAY(I)=WORK(I)
    ENDDO
    RETURN
  END SUBROUTINE AINDXL

  SUBROUTINE AINDX4(INVMAP,ARRAY,DIM,WORK)
    !-----------------------------------------------------------------------
    !     Same routine as AINDEX. Applies to INTEGER arrays.
    !
  use chm_kinds
    implicit none
    INTEGER   DIM,INVMAP(*)
    INTEGER   ARRAY(*),WORK(*)
    INTEGER   I
    !
    DO I=1,DIM
       WORK(I)=ARRAY(INVMAP(I))
    ENDDO
    DO I=1,DIM
       ARRAY(I)=WORK(I)
    ENDDO
    RETURN
  END SUBROUTINE AINDX4

  SUBROUTINE AINDX8(INVMAP,ARRAY,DIM,WORK)
    !-----------------------------------------------------------------------
    !     Same routine as AINDEX. Applies to chm_int8 arrays.
    !
  use chm_kinds
    implicit none
    INTEGER   DIM,INVMAP(*)
    INTEGER(chm_int8) :: ARRAY(*),WORK(*)
    INTEGER   I

    DO I=1,DIM
       WORK(I)=ARRAY(INVMAP(I))
    ENDDO
    DO I=1,DIM
       ARRAY(I)=WORK(I)
    ENDDO
    RETURN
  END SUBROUTINE AINDX8

  SUBROUTINE AINDX_PTR(invmap, array, dim)
  use chm_types
    implicit none
    integer :: dim, invmap(*)
    type(chm_ptr) :: array(*)
    type(chm_ptr) :: work(dim)
    integer :: i

    do i = 1, dim
       work(i) = array(invmap(i))
    enddo
    array(1:dim) = work(1:dim)

  END SUBROUTINE AINDX_PTR

  SUBROUTINE SHFTIC(NUMBER,IOPT,INDX,MA,IA,JA,KA,LA, &
       MB,IB,JB,KB,LB)
    !-----------------------------------------------------------------------
    !     THIS ROUTINE REMAPS INTO A TEMPORARY INTERNAL COORDINATE
    !     ARRAY (IA,JA ETC) FROM (IB,JB ETC).
    !     i.e. SHIFTS I.C.
    !
    !      Author: S. Swaminathan
    !
  use chm_kinds
    implicit none
    INTEGER NUMBER,IOPT
    INTEGER INDX(*)
    INTEGER IA(*),JA(*),KA(*),LA(*),MA(*)
    INTEGER IB(*),JB(*),KB(*),LB(*),MB(*)
    INTEGER INDEX,I
    !
    loop110:DO I=1,NUMBER
       INDEX=INDX(I)
       MA(I)=MB(INDEX)
       IA(I)=IB(INDEX)
       JA(I)=JB(INDEX)
       IF(IOPT == 2) cycle loop110
       KA(I)=KB(INDEX)
       IF(IOPT == 3) cycle loop110
       LA(I)=LB(INDEX)
    enddo loop110
    RETURN
  END SUBROUTINE SHFTIC

  SUBROUTINE COPYD( FROMX, FROMY, FROMZ, TOX, TOY, TOZ , N )
    !-----------------------------------------------------------------------
    !     Copies FROM one 3N dimensional vector TO another.

  use chm_kinds
    implicit none
    INTEGER    N, I
    real(chm_real)  FROMX(N),FROMY(N),FROMZ(N),TOX(N),TOY(N),TOZ(N)

    DO I=1,N
       TOX(I) = FROMX(I)
       TOY(I) = FROMY(I)
       TOZ(I) = FROMZ(I)
    ENDDO

    RETURN
  END SUBROUTINE COPYD


  !
  !     (binary) search routines for looking up key's in (multiple) arrays
  !     ==================================================================
  !
  INTEGER FUNCTION NINDX(NUMBER,NARRAY,NLEN)
    !-----------------------------------------------------------------------
    !     This routine determines the lowest index of "number" in NARRAY
    !     using a binary search. NARRAY and NUMBER have to be of type I*4.
    !     NARRAY is assumed to be sorted in ascending order.
    !
    !      Barry Olafson
    !      Axel Brunger, 13-ARP-83
    !
  use chm_kinds
    implicit none
    INTEGER   NUMBER, NARRAY(*), NLEN
    !
    INTEGER   ISTART, ISTOP, N
    !
    ISTART=1
    ISTOP=NLEN
    N=(ISTART+ISTOP)/2
    !
    IF (NARRAY(N) /= NUMBER.AND..NOT.ISTART >= ISTOP) THEN
30     CONTINUE
       IF (NARRAY(N) > NUMBER) THEN
          ISTOP=N-1
       ELSE
          ISTART=N+1
       ENDIF
       N=(ISTART+ISTOP)/2
       IF (NARRAY(N) /= NUMBER.AND..NOT.ISTART >= ISTOP) GOTO 30
    ENDIF
    !
    !      THE FOLLOWING LOOKS FUNNY,
    !      BUT IS NEEDED TO ACCOMODATE A BUG IN THE CRAY COMPILER
    !
    IF (N > 1.AND.NARRAY(N) == NUMBER) THEN
60     CONTINUE
       N=N-1
       IF (N > 1.AND.NARRAY(N) == NUMBER) GOTO 60
    ENDIF
    IF (.NOT.(N == 1.AND.NARRAY(1).EQ.NUMBER)) N=N+1
    IF (NARRAY(N) == NUMBER) THEN
       NINDX=N
    ELSE
       NINDX=0
    ENDIF
    !
    !CC  THIS IS WHAT IT SHOULD LOOK LIKE
    !CC      WHEN (NARRAY(N) == NUMBER)
    !CC      WHILE (N > 1.AND.NARRAY(N) == NUMBER) N=N-1
    !CC      IF (.NOT.(N == 1.AND.NARRAY(1).EQ.NUMBER)) N=N+1
    !CC      NINDX=N
    !CC      FIN
    !CC      ELSE NINDX=0
    !
    RETURN
  END FUNCTION NINDX

  INTEGER FUNCTION NINDX8(NUMBER,NARRAY,NLEN)
    !-----------------------------------------------------------------------
    !     This routine determines the lowest index of "number" in NARRAY
    !     using a binary search. NARRAY and NUMBER have to be of type chm_int8.
    !     NARRAY is assumed to be sorted in ascending order.
    !
    !      Barry Olafson
    !      Axel Brunger, 13-ARP-83
    !
  use chm_kinds
    implicit none
    integer(chm_int8) :: number, narray(nlen)
    integer nlen
    integer   istart, istop, n,i

    istart=1
    istop=nlen
    n=(istart+istop)/2

    do while(narray(n) /= number  .and.  .not. (istart >= istop) ) 
       if (narray(n) > number) then
          istop=n-1
          if(istop < istart) exit
       else
          istart=n+1
       endif
       n=(istart+istop)/2
    enddo

    !      THE FOLLOWING LOOKS FUNNY,
    !      BUT IS NEEDED TO ACCOMODATE A BUG IN THE CRAY COMPILER

    ! mfc.. this stuff seems to find the lowest instance of a multiple instance of
    !       a value in an array if the number to be matched is equal to athat value
    do while (n > 1.and.narray(n) == number) 
       n=n-1
    enddo

    if (.not.(n == 1.and.narray(1).eq.number) .and. n < nlen) n=n+1
    if (narray(n) == number) then
       nindx8=n
    else
       nindx8=0
    endif

    !CC  THIS IS WHAT IT SHOULD LOOK LIKE
    !CC      WHEN (NARRAY(N) == NUMBER)
    !CC      WHILE (N > 1.AND.NARRAY(N) == NUMBER) N=N-1
    !CC      IF (.NOT.(N == 1.AND.NARRAY(1).EQ.NUMBER)) N=N+1
    !CC      NINDX=N
    !CC      FIN
    !CC      ELSE NINDX=0

    RETURN
  END FUNCTION NINDX8

  INTEGER FUNCTION FIND52(X1,X2,X3,X4,X5,Y1,Y2,Y3,Y4,Y5, &
       DIM,WIDTH,MARK)
    !-----------------------------------------------------------------------
    !     Linear search through X1(I),...,XWIDTH(I) for I=1,DIM to locate
    !     element Y1,...,YWIDTH. If not found FIND52 is set to MARK
    !     X1,...,X5 are of type INTEGER, Y1,...Y5 are of type INTEGER
    !     
    !      23-JAN-83 Axel Brunger
    !
  use chm_kinds
    implicit none
    INTEGER   DIM
    INTEGER   X1(DIM), X2(DIM), X3(DIM), X4(DIM), X5(DIM)
    INTEGER   Y1, Y2, Y3, Y4, Y5
    INTEGER   WIDTH, MARK
    !
    INTEGER   I
    LOGICAL   FOUND
    !
    !     bug with patch setup if there is no IC yet!
    !C called in modpsf.src
    IF(DIM <= 0)THEN
       FIND52=MARK
       RETURN
    ENDIF
    !
    I=0
20  CONTINUE
    I=I+1
    FOUND=(Y1 == X1(I))
    IF (FOUND.AND.WIDTH /= 1) THEN
       FOUND=(Y2 == X2(I))
       IF (FOUND.AND.WIDTH /= 2) THEN
          FOUND=(Y3 == X3(I))
          IF (FOUND.AND.WIDTH /= 3) THEN
             FOUND=(Y4 == X4(I))
             IF (FOUND.AND.WIDTH /= 4) THEN
                FOUND=(Y5 == X5(I))
             ENDIF
          ENDIF
       ENDIF
    ENDIF
    IF (.NOT.(FOUND.OR.I >= DIM)) GOTO 20
    IF (FOUND) THEN
       FIND52=I
    ELSE
       FIND52=MARK
    ENDIF
    RETURN
  END FUNCTION FIND52

  INTEGER FUNCTION SRCHWD(WORDS,NUMWRD,WORD)
    !-----------------------------------------------------------------------
    !     Searches the words array for word and returns its index
    !     if found and zero otherwise. Applies to INTEGER arrays.
    !
    !      Author: Robert Bruccoleri
    !
  use chm_kinds
    implicit none
    INTEGER  WORDS(*), NUMWRD, WORD, I
    !
    SRCHWD=0
    IF (NUMWRD > 0) THEN
       I=0
40     CONTINUE
       I=I+1
       IF (.NOT.(I >= NUMWRD.OR.WORD == WORDS(I))) GOTO 40
       IF (WORD == WORDS(I)) SRCHWD=I
    ENDIF
    RETURN
  END FUNCTION SRCHWD

  INTEGER FUNCTION SRCHWS(WORDS,NUMWRD,WORD)
    !-----------------------------------------------------------------------
    !     Searches the words array for word and returns its index
    !     if found and zero otherwise. Applies to character*4 arrays.
    !     (character*4 arrays are represented by INTEGER arrays)
    !
  use chm_kinds
    implicit none
    character(len=*)  WORDS(*), WORD
    INTEGER  NUMWRD, I
    !
    SRCHWS=0
    IF (NUMWRD > 0) THEN
       I=0
80     CONTINUE
       I=I+1
       IF (.NOT.(I >= NUMWRD.OR.WORD == WORDS(I))) GOTO 80
       IF (WORD == WORDS(I)) SRCHWS=I
    ENDIF
    RETURN
  END FUNCTION SRCHWS

#if KEY_CADPAC==0 && KEY_GAMESSUK==0 && KEY_GAMESS==0 /*cp_guk*/
  SUBROUTINE DAXPY(N,DA,DX,INCX,DY,INCY)
    ! --------------------------------------------------------
    !  CONSTANT TIMES A VECTOR PLUS A VECTOR.
    !  USES UNROLLED LOOPS FOR INCREMENTS EQUAL TO ONE.
    !  JACK DONGARRA, LINPACK, 3/11/78.
    ! --------------------------------------------------------
  use chm_kinds
    implicit none
    !
    real(chm_real) DX(*), DY(*), DA
    INTEGER I,INCX,INCY,IX,IY,M,MP1,N
    !
    IF (N <= 0) RETURN
    IF (DA  ==  0.0D0) RETURN
    IF (INCX /= 1 .or. INCY /= 1) then
       ! --------------------------------------------------------
       !  CODE FOR UNEQUAL INCREMENTS OR EQUAL INCREMENTS
       !  NOT EQUAL TO 1
       ! --------------------------------------------------------
       IX = 1
       IY = 1
       IF (INCX  <  0) IX = (-N+1)*INCX + 1
       IF (INCY  <  0) IY = (-N+1)*INCY + 1
       DO I = 1,N
          DY(IY) = DY(IY) + DA*DX(IX)
          IX = IX + INCX
          IY = IY + INCY
       enddo
       RETURN
    endif
    ! --------------------------------------------------------
    !  CODE FOR BOTH INCREMENTS EQUAL TO 1
    !  CLEAN-UP LOOP
    ! --------------------------------------------------------
    M = MOD(N,4)
    IF (M  /=  0) then
       DY(1:m) = DY(1:m) + DA*DX(1:m)
       IF (N  <  4) RETURN
    endif
    MP1 = M + 1
    DO I = MP1,N,4
       DY(I) = DY(I) + DA*DX(I)
       DY(I + 1) = DY(I + 1) + DA*DX(I + 1)
       DY(I + 2) = DY(I + 2) + DA*DX(I + 2)
       DY(I + 3) = DY(I + 3) + DA*DX(I + 3)
    enddo
    RETURN
  END SUBROUTINE DAXPY
#endif /* (cp_guk)*/


#if KEY_GAMESSUK==0 && KEY_GAMESS==0 /*guk0*/
  ! copy 2 of 2
  SUBROUTINE DCOPY(N,DX,INCX,DY,INCY)
    ! --------------------------------------------------------
    !  COPIES A VECTOR, X, TO A VECTOR, Y.
    !  USES UNROLLED LOOPS FOR INCREMENTS EQUAL TO ONE.
    !  JACK DONGARRA, LINPACK, 3/11/78.
    ! --------------------------------------------------------
  use chm_kinds
    implicit none
    !
    real(chm_real) DX(*), DY(*)
    INTEGER I,INCX,INCY,IX,IY,M,MP1,N
    !
    IF (N  <=  0) RETURN
    IF (INCX /= 1 .or. INCY /= 1) then
       ! --------------------------------------------------------
       !  CODE FOR UNEQUAL INCREMENTS OR EQUAL INCREMENTS
       !  NOT EQUAL TO 1
       ! --------------------------------------------------------
       IX = 1
       IY = 1
       IF (INCX  <  0) IX = (-N+1)*INCX + 1
       IF (INCY  <  0) IY = (-N+1)*INCY + 1
       DO I = 1,N
          DY(IY) = DX(IX)
          IX = IX + INCX
          IY = IY + INCY
       enddo
       RETURN
    endif
    ! --------------------------------------------------------
    !  CODE FOR BOTH INCREMENTS EQUAL TO 1
    !  CLEAN-UP LOOP
    ! --------------------------------------------------------
    M = MOD(N,7)
    IF (M  /=  0) then
       DY(1:m) = DX(1:m)
       IF (N  <  7) RETURN
    endif
    MP1 = M + 1
    DO I = MP1,N,7
       DY(I) = DX(I)
       DY(I + 1) = DX(I + 1)
       DY(I + 2) = DX(I + 2)
       DY(I + 3) = DX(I + 3)
       DY(I + 4) = DX(I + 4)
       DY(I + 5) = DX(I + 5)
       DY(I + 6) = DX(I + 6)
    enddo
    RETURN
  END SUBROUTINE DCOPY
#endif /* (guk0) */
  ! JZ_UW12: REAL*4 version
#if KEY_GAMESSUK==0 || KEY_MNDO97==0 /* (guk) */
  SUBROUTINE DCOPYR4(N,DX,INCX,DY,INCY)
    ! --------------------------------------------------------
    !  COPIES A VECTOR, X, TO A VECTOR, Y.
    !  USES UNROLLED LOOPS FOR INCREMENTS EQUAL TO ONE.
    !  JACK DONGARRA, LINPACK, 3/11/78.
    ! --------------------------------------------------------
  use chm_kinds
    implicit none
    !
    real(chm_real4) DX(*), DY(*)
    INTEGER I,INCX,INCY,IX,IY,M,MP1,N
    !
    IF (N  <=  0) RETURN
    IF (INCX /= 1 .or. INCY /= 1) then
       ! --------------------------------------------------------
       !  CODE FOR UNEQUAL INCREMENTS OR EQUAL INCREMENTS
       !  NOT EQUAL TO 1
       ! --------------------------------------------------------
       IX = 1
       IY = 1
       IF (INCX  <  0) IX = (-N+1)*INCX + 1
       IF (INCY  <  0) IY = (-N+1)*INCY + 1
       DO I = 1,N
          DY(IY) = DX(IX)
          IX = IX + INCX
          IY = IY + INCY
       enddo
       RETURN
    endif
    ! --------------------------------------------------------
    !  CODE FOR BOTH INCREMENTS EQUAL TO 1
    !  CLEAN-UP LOOP
    ! --------------------------------------------------------
    M = MOD(N,7)
    IF (M  /=  0) then
       DY(1:m) = DX(1:m)
       IF (N  <  7) RETURN
    endif
    MP1 = M + 1
    DO I = MP1,N,7
       DY(I) = DX(I)
       DY(I + 1) = DX(I + 1)
       DY(I + 2) = DX(I + 2)
       DY(I + 3) = DX(I + 3)
       DY(I + 4) = DX(I + 4)
       DY(I + 5) = DX(I + 5)
       DY(I + 6) = DX(I + 6)
    enddo
    RETURN
  END SUBROUTINE DCOPYR4
  ! JZ_UW12: INTEGER version
  SUBROUTINE DCOPYI4(N,DX,INCX,DY,INCY)
    ! --------------------------------------------------------
    !  COPIES A VECTOR, X, TO A VECTOR, Y.
    !  USES UNROLLED LOOPS FOR INCREMENTS EQUAL TO ONE.
    !  JACK DONGARRA, LINPACK, 3/11/78.
    ! --------------------------------------------------------
  use chm_kinds
    implicit none
    !
    INTEGER DX(*), DY(*)
    INTEGER I,INCX,INCY,IX,IY,M,MP1,N
    !
    IF (N  <=  0) RETURN
    IF (INCX /= 1 .or. INCY /= 1) then
       ! --------------------------------------------------------
       !  CODE FOR UNEQUAL INCREMENTS OR EQUAL INCREMENTS
       !  NOT EQUAL TO 1
       ! --------------------------------------------------------
       IX = 1
       IY = 1
       IF (INCX  <  0) IX = (-N+1)*INCX + 1
       IF (INCY  <  0) IY = (-N+1)*INCY + 1
       DO I = 1,N
          DY(IY) = DX(IX)
          IX = IX + INCX
          IY = IY + INCY
       enddo
       RETURN
    endif
    ! --------------------------------------------------------
    !  CODE FOR BOTH INCREMENTS EQUAL TO 1
    !  CLEAN-UP LOOP
    ! --------------------------------------------------------
    M = MOD(N,7)
    IF (M  /=  0) then
       DY(1:m) = DX(1:m)
       IF (N  <  7) RETURN
    endif
    MP1 = M + 1
    DO I = MP1,N,7
       DY(I) = DX(I)
       DY(I + 1) = DX(I + 1)
       DY(I + 2) = DX(I + 2)
       DY(I + 3) = DX(I + 3)
       DY(I + 4) = DX(I + 4)
       DY(I + 5) = DX(I + 5)
       DY(I + 6) = DX(I + 6)
    enddo
    RETURN
  END SUBROUTINE DCOPYI4
#endif /* (guk)*/


#if KEY_CADPAC==0 && KEY_GAMESSUK==0 && KEY_GAMESS==0 /*cp_guk*/
  FUNCTION DDOT(N,DX,INCX,DY,INCY) result(ddot_rtn)
    !
    ! --------------------------------------------------------
    !  FORMS THE DOT PRODUCT OF TWO VECTORS.
    !  USES UNROLLED LOOPS FOR INCREMENTS EQUAL TO ONE.
    !  JACK DONGARRA, LINPACK, 3/11/78.
    ! --------------------------------------------------------
  use chm_kinds
    implicit none
    !
    real(chm_real) DX(*), DY(*), DTEMP, ddot_rtn
    INTEGER I,INCX,INCY,IX,IY,M,MP1,N
    !
    DDOT_rtn = 0.0D0
    DTEMP = 0.0D0
    IF (N  <=  0) RETURN
    IF (INCX /= 1 .or. INCY /= 1) then
       ! --------------------------------------------------------
       !  CODE FOR UNEQUAL INCREMENTS OR EQUAL INCREMENTS
       !  NOT EQUAL TO 1
       ! --------------------------------------------------------
       IX = 1
       IY = 1
       IF (INCX  <  0) IX = (-N+1)*INCX + 1
       IF (INCY  <  0) IY = (-N+1)*INCY + 1
       DO I = 1,N
          DTEMP = DTEMP + DX(IX)*DY(IY)
          IX = IX + INCX
          IY = IY + INCY
       enddo
       DDOT_rtn = DTEMP
       RETURN
    endif
    ! --------------------------------------------------------
    !   CODE FOR BOTH INCREMENTS EQUAL TO 1
    !   CLEAN-UP LOOP
    ! --------------------------------------------------------
    M = MOD(N,5)
    IF (M  /=  0) then
       DO I = 1,M
          DTEMP = DTEMP + DX(I)*DY(I)
       enddo
       IF (N  <  5) GO TO 60
    endif
    MP1 = M + 1
    DO I = MP1,N,5
       DTEMP = DTEMP + DX(I)*DY(I) + DX(I + 1)*DY(I + 1) + &
            DX(I + 2)*DY(I + 2) + DX(I + 3)*DY(I + 3) + DX(I + 4)*DY(I + 4)
    enddo
60  DDOT_rtn = DTEMP
    RETURN
  END FUNCTION DDOT
#endif /* (cp_guk)*/


  SUBROUTINE RMCOPY(NX,NY,A,IXA,IYA,B,IXB,IYB)
    !
    ! This routine copies/transposes one 2-d array into another
    ! The Y dimension is the "inner" and should be stride 1 where possible
    !     real(chm_real) version
    !
  use chm_kinds
    implicit none
    !
    INTEGER NX,NY,IXA,IYA,IXB,IYB
    real(chm_real) A(*),B(*)
    !
    INTEGER I,J,IPT,JPT,IPT1,JPT1
    !
    IPT1=1
    JPT1=1
    DO I=1,NX
       IPT=IPT1
       JPT=JPT1
       DO J=1,NY
          B(JPT)=A(IPT)
          IPT=IPT+IYA
          JPT=JPT+IYB
       ENDDO
       IPT1=IPT1+IXA
       JPT1=JPT1+IXB
    ENDDO
    !
    RETURN
  END SUBROUTINE RMCOPY
  !
  SUBROUTINE CMCOPY(NX,NY,A,IXA,IYA,B,IXB,IYB)
    !
    ! This routine copies/transposes one 2-d array into another
    ! The Y dimension is the "inner" and should be stride 1 where possible
    !     COMPLEX*16 version
    !
  use chm_kinds
    implicit none
    !
    INTEGER NX,NY,IXA,IYA,IXB,IYB
    COMPLEX*16 A(*),B(*)
    !
    INTEGER I,J,IPT,JPT,IPT1,JPT1
    !
    IPT1=1
    JPT1=1
    DO I=1,NX
       IPT=IPT1
       JPT=JPT1
       DO J=1,NY
          B(JPT)=A(IPT)
          IPT=IPT+IYA
          JPT=JPT+IYB
       ENDDO
       IPT1=IPT1+IXA
       JPT1=JPT1+IXB
    ENDDO
    !
    RETURN
  END SUBROUTINE CMCOPY
  !
  ! JZ_UW12: Initialization routines
  !          taken from old CHARMM
      SUBROUTINE FILLI4(A,N,VALUE)
!-----------------------------------------------------------------------
!      FILLS A WITH VALUE
!      Author: Robert Bruccoleri
!
      use chm_kinds
      implicit none
      INTEGER   A(*),VALUE
      INTEGER N,I
!
      DO I=1,N
        A(I)=VALUE
      ENDDO
      RETURN
      END SUBROUTINE FILLI4
!
      SUBROUTINE FILLR8(A,N,VALUE)
!-----------------------------------------------------------------------
!     FILLS A WITH VALUE
!
!      Author: Robert Bruccoleri
!
      use chm_kinds
      implicit none
      real(chm_real) A(*),VALUE
      INTEGER N,I
!
      DO I=1,N
        A(I)=VALUE
      ENDDO
      RETURN
      END SUBROUTINE FILLR8
!  
      SUBROUTINE FILLR41(A,N,VALUE)
!-----------------------------------------------------------------------
!     FILLS A WITH VALUE
!     No Byte-specification in FILLR4 as found in correl/rdfsol.src
!
      use chm_kinds
      implicit none
      real(chm_real4) A(*),VALUE
      INTEGER N,I
!
      DO I=1,N
        A(I)=VALUE
      ENDDO
      RETURN
      END SUBROUTINE FILLR41

