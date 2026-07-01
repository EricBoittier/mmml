SUBROUTINE qcksrt(n,arr,ibrr)
  !------------------------------------------------------------------------
  ! Quick sorting of arr (used in MAYER)
  !
  use chm_kinds
  implicit none
  integer n,ibrr,m,nstack,istack,jstack,l,ir,ib,j,i,iq
  real(chm_real) arr,fm,fa,fc,fmi,a,fx
  parameter (m=7,nstack=50,fm=7875.,fa=211.,fc=1663.,fmi=1./fm)
  dimension arr(n),ibrr(n),istack(nstack)
  !
  jstack=0
  l=1
  ir=n
  fx=0.
10 if(ir-l < m)then
     do j=l+1,ir
        a=arr(j)
        ib=ibrr(j)
        do i=j-1,1,-1
           if(arr(i) <= a)goto 12
           arr(i+1)=arr(i)
           ibrr(i+1)=ibrr(i)
        enddo
        i=0
12      arr(i+1)=a
        ibrr(i+1)=ib
     enddo
     if(jstack == 0)return
     ir=istack(jstack)
     l=istack(jstack-1)
     jstack=jstack-2
  else
     i=l
     j=ir
     fx=mod(fx*fa+fc,fm)
     iq=l+(ir-l+1)*(fx*fmi)
     a=arr(iq)
     ib=ibrr(iq)
     arr(iq)=arr(l)
     ibrr(iq)=ibrr(l)
20   continue
21   if(j > 0)then
        if(a < arr(j))then
           j=j-1
           goto 21
        endif
     endif
     if(j <= i)then
        arr(i)=a
        ibrr(i)=ib
        goto 30
     endif
     arr(i)=arr(j)
     ibrr(i)=ibrr(j)
     i=i+1
22   if(i <= n)then
        if(a > arr(i))then
           i=i+1
           goto 22
        endif
     endif
     if(j <= i)then
        arr(j)=a
        ibrr(j)=ib
        i=j
        goto 30
     endif
     arr(j)=arr(i)
     ibrr(j)=ibrr(i)
     j=j-1
     goto 20
30   jstack=jstack+2
     if(jstack > nstack) then
        call wrndie (0,'<QCKSRT>','nstack must be made larger.')
     endif
     if(ir-i >= i-l)then
        istack(jstack)=ir
        istack(jstack-1)=i+1
        ir=i-1
     else
        istack(jstack)=i-1
        istack(jstack-1)=l
        l=i+1
     endif
  endif
  goto 10
end SUBROUTINE qcksrt

integer function sort_depth(N)
  !
  !     Notes added 6/5/90 by Tom Ngo--here is how to justify choice of
  !     size for STORE array:
  !
  !     1.  This particular implementation of Quicksort has a maximum
  !         recursion level of (ceil(logbase2(N))-1).  To see this,
  !         consider an array containing N = 2**M cells.  The maximum
  !         stack space is used if each partitioning operation divides
  !         the array exactly in two.  At its largest, the stack
  !         contains indices for subarrays of length 2**(M-1), 2**(M-2),
  !         ..., 2.  Thus, the number of elements in the stack is
  !         M-1 = logbase2(N) - 1.  In general, N is not a power of two,
  !         so you "round up" to the next power of two by taking ceil()
  !         of the logbase2().
  !
  !     2.  Getting a weaker, but more easily calculated upper bound:
  !
  !                 2 * ( ceil(logbase2(N)) - 1 )
  !            <=   2 * ( floor(logbase2(N)) )
  !            ==   4 * ( 0.5 * floor(logbase2(N)) )
  !            <=   4 * ( floor( 0.5 * logbase2(N) + 1 ) )    [*]
  !            <=   4 * ( floor( logbaseE(2) * logbase2(N) + 1 ) )
  !            ==   4 * ( floor( logbaseE(N) + 1 ) )
  !
  !     Why the +1 in the expression labelled "[*]"?  Consider, for
  !     example, the case logbase2(N) = 3.8.  I know there is no such
  !     case in reality, but you will see the point:
  !
  !                   4 * ( 0.5 * floor( logbase2(N) )   =   6
  !           but...  4 * ( floor(0.5 * logbase2(N)) )   =   4
  !
  !     Thus, taking the 0.5 inside the floor() function can cause the
  !     expression to decrease through rounding.  The +1 is sufficient
  !     to guarantee that this will not happen.  Prior to CMS version *2
  !     of CHARMM 22, the +1 was absent, and this caused crashes.
  !
  integer,intent(in) :: N
  ! assume N > 0
  sort_depth = int(1 + log(real(N)) * 4)
end function sort_depth

SUBROUTINE SORTP(N,PERM,ORDERL,A1,A2,A3,A4,A5,A6,A7,A8)
  !-----------------------------------------------------------------------
  !     Generalised quicksort routine
  !     PERMutation version
  !
  !     Structure A1,A2,A3,A4,A5,A6 of length N sorted as defined in
  !     logical function ORDERL. The algorithm used is quicksort.
  !
  !     Original version by Robert Bruccoleri
  !     Generalised to sort on multiple arrays with an external ORDERL
  !     function by Axel Brunger, 15-MAR-83
  !
  !     Converted to FORTRAN 9/8/92 by Tom Ngo.  I tried not to shuffle
  !     statements around too much; in particular, I left the two FLECS
  !     procedures where they were, and used pairs of (non-assigned)
  !     GOTOs to accomplish the effect of the FLECS procedure call.  The
  !     least trivial parts of the conversion were the REPEAT UNTIL
  !     constructs, for which there is no equivalent in FORTRAN.  In two
  !     easy cases I rewrote the loops clearly.  In the remaining cases
  !     I replaced the REPEAT UNTIL constructs by equivalent IF-GOTO
  !     loops, at some expense to readability.
  !
  use chm_kinds
  use memory
  implicit none

  INTEGER   N, PERM(*)
  logical,external :: ORDERL
  integer,external :: sort_depth
  INTEGER A1(*),A2(*),A3(*),A4(*),A5(*),A6(*),A7(*),A8(*)
  integer,allocatable,dimension(:) :: STORE

  IF (N /= 0) THEN
     call chmalloc('sort.src','SORTP','STORE',sort_depth(N),intg=STORE)
     CALL SORTP2(N,PERM,ORDERL,A1,A2,A3,A4,A5,A6,A7,A8,STORE)
     call chmdealloc('sort.src','SORTP','STORE',sort_depth(N),intg=STORE)
  END IF
  !
  RETURN
END SUBROUTINE SORTP

SUBROUTINE SRT8P(N,PERM,ORDERL,A1,A2,A3,A4,A5,A6,A7,A8)
  ! duplicated from SORTP to accommodate chm_int8 for A1
  use chm_kinds
  use memory
  implicit none
  !
  INTEGER   N, PERM(*)
  logical,external :: ORDERL
  integer,external :: sort_depth
  integer(chm_int8) :: a1(*)
  INTEGER A2(*),A3(*),A4(*),A5(*),A6(*),A7(*),A8(*)
  integer,allocatable,dimension(:) :: STORE

  IF (N /= 0) THEN
     call chmalloc('sort.src','SRT8P','STORE',sort_depth(N),intg=STORE)
     CALL SRT8P2(N,PERM,ORDERL,A1,A2,A3,A4,A5,A6,A7,A8,STORE)
     call chmdealloc('sort.src','SRT8P','STORE',sort_depth(N),intg=STORE)
  END IF
  !
  RETURN
END SUBROUTINE SRT8P

SUBROUTINE SORTP2(N,PERM,ORDERL,A1,A2,A3,A4,A5,A6,A7,A8,STORE)
  !
  !     See SORTP above.
  !     Robert Bruccoleri
  !     Axel Brunger, 15-MAR-83
  !
  use clcg_mod,only:random
  use chm_kinds
  implicit none
  INTEGER   N, PERM(*)
  INTEGER   A1(*), A2(*), A3(*), A4(*), A5(*), A6(*), A7(*), A8(*)
  INTEGER   STORE(*)
  !
  LOGICAL   ORDERL
  EXTERNAL  ORDERL
  !
  INTEGER   I, END0, END1, L1, L2, START, START2, TOP, TEMP
  INTEGER   DUMMY, RNDIND
  LOGICAL   CONDIT
  !
  DATA DUMMY/0/
  !
  DO I = 1, N
     PERM(I)=I
  END DO
  !
  TOP=0
  START=1
  END0=N
  !
3333 IF(END0 == START) GOTO 1111
  IF(END0 /= START+1) GOTO 2222
  IF (.NOT. ORDERL(PERM(START),PERM(END0),A1,A2,A3,A4,A5,A6,A7,A8)) &
       THEN
     TEMP=PERM(START)
     PERM(START)=PERM(END0)
     PERM(END0)=TEMP
  END IF
1111 IF(TOP == 0) GOTO 9999
  END0=STORE(TOP)
  START=STORE(TOP-1)
  TOP=TOP-2
  GOTO 3333
  !
2222 CONTINUE
  GOTO 4444                 ! PROCESS-PARTITION
4445 CONTINUE                  ! return from PROCESS-PARTITION
  L1=END1-START
  L2=END0-START2
  IF (L1 <= L2) THEN
     !
     !     L2>=L1 DO L1 FIRST
     !
     TOP=TOP+2
     STORE(TOP-1)=START2
     STORE(TOP)=END0
     END0=END1
  ELSE
     TOP=TOP+2
     STORE(TOP-1)=START
     STORE(TOP)=END1
     START=START2
  END IF
  GOTO 3333
  !
9999 CONTINUE
  !
  RETURN
  !
4444 CONTINUE                  ! TO PROCESS-PARTITION
  !
  RNDIND=INT((END0-START)*RANDOM(DUMMY))+START
  START2=START-1
  END1=END0+1
  !
101 START2=START2+1
  CONDIT=ORDERL(PERM(START2),PERM(RNDIND),A1,A2,A3,A4,A5,A6,A7,A8)
  IF (CONDIT .AND. START2 /= END0) GOTO 101 ! REPEAT UNTIL
  !
201 END1=END1-1
  CONDIT=ORDERL(PERM(RNDIND),PERM(END1),A1,A2,A3,A4,A5,A6,A7,A8)
  IF (CONDIT .AND. END1 /= START) GOTO 201
  !
  IF (START2 < END1) THEN
     TEMP=PERM(START2)
     PERM(START2)=PERM(END1)
     PERM(END1)=TEMP
     GOTO 101
  END IF
  !
  !
  IF (START2 < RNDIND) THEN
     TEMP=PERM(START2)
     PERM(START2)=PERM(RNDIND)
     PERM(RNDIND)=TEMP
     START2=START2+1
  ELSE
     IF (RNDIND < END1) THEN
        TEMP=PERM(END1)
        PERM(END1)=PERM(RNDIND)
        PERM(RNDIND)=TEMP
        END1=END1-1
     END IF
  END IF
  !
  GOTO 4445                 ! FIN PROCESS-PARTITION
  !
END SUBROUTINE SORTP2

SUBROUTINE SRT8P2(N,PERM,ORDERL,A1,A2,A3,A4,A5,A6,A7,A8,STORE)
  !
  !     See SRT8P above.
  !     Robert Bruccoleri
  !     Axel Brunger, 15-MAR-83
  !
  use clcg_mod,only:random
  use chm_kinds
  implicit none
  INTEGER   N, PERM(*)
  INTEGER(chm_int8) :: A1(*)
  INTEGER   A2(*), A3(*), A4(*), A5(*), A6(*), A7(*), A8(*), STORE(*)
  !
  LOGICAL   ORDERL
  EXTERNAL  ORDERL
  !
  INTEGER   I, END0, END1, L1, L2, START, START2, TOP, TEMP
  INTEGER   DUMMY, RNDIND
  LOGICAL   CONDIT
  !
  DATA DUMMY/0/
  !
  DO I = 1, N
     PERM(I)=I
  END DO
  !
  TOP=0
  START=1
  END0=N
  !
3333 IF(END0 == START) GOTO 1111
  IF(END0 /= START+1) GOTO 2222
  IF (.NOT. ORDERL(PERM(START),PERM(END0),A1,A2,A3,A4,A5,A6,A7,A8)) &
       THEN
     TEMP=PERM(START)
     PERM(START)=PERM(END0)
     PERM(END0)=TEMP
  END IF
1111 IF(TOP == 0) GOTO 9999
  END0=STORE(TOP)
  START=STORE(TOP-1)
  TOP=TOP-2
  GOTO 3333
  !
2222 CONTINUE
  GOTO 4444                 ! PROCESS-PARTITION
4445 CONTINUE                  ! return from PROCESS-PARTITION
  L1=END1-START
  L2=END0-START2
  IF (L1 <= L2) THEN
     !
     !     L2>=L1 DO L1 FIRST
     !
     TOP=TOP+2
     STORE(TOP-1)=START2
     STORE(TOP)=END0
     END0=END1
  ELSE
     TOP=TOP+2
     STORE(TOP-1)=START
     STORE(TOP)=END1
     START=START2
  END IF
  GOTO 3333
  !
9999 CONTINUE
  !
  RETURN
  !
4444 CONTINUE                  ! TO PROCESS-PARTITION
  !
  RNDIND=INT((END0-START)*RANDOM(DUMMY))+START
  START2=START-1
  END1=END0+1
  !
101 START2=START2+1
  CONDIT=ORDERL(PERM(START2),PERM(RNDIND),A1,A2,A3,A4,A5,A6,A7,A8)
  IF (CONDIT .AND. START2 /= END0) GOTO 101 ! REPEAT UNTIL
  !
201 END1=END1-1
  CONDIT=ORDERL(PERM(RNDIND),PERM(END1),A1,A2,A3,A4,A5,A6,A7,A8)
  IF (CONDIT .AND. END1 /= START) GOTO 201
  !
  IF (START2 < END1) THEN
     TEMP=PERM(START2)
     PERM(START2)=PERM(END1)
     PERM(END1)=TEMP
     GOTO 101
  END IF
  !
  !
  IF (START2 < RNDIND) THEN
     TEMP=PERM(START2)
     PERM(START2)=PERM(RNDIND)
     PERM(RNDIND)=TEMP
     START2=START2+1
  ELSE
     IF (RNDIND < END1) THEN
        TEMP=PERM(END1)
        PERM(END1)=PERM(RNDIND)
        PERM(RNDIND)=TEMP
        END1=END1-1
     END IF
  END IF
  !
  GOTO 4445                 ! FIN PROCESS-PARTITION
  !
END SUBROUTINE SRT8P2

SUBROUTINE SORT(N,EXCHL,ORDERL,A1,A2,A3,A4,A5,A6,A7,A8)
  !
  !     Generalised quicksort routine
  !     EXCHange version
  !
  !     Structure A1,A2,A3,A4,A5,A6,A7,A8 of length N sorted as defined
  !     in logical function ORDERL. The algorithm used is quicksort.
  !     EXCHange of elements defined in subroutine EXCHL.
  !
  !     Original version by Robert Bruccoleri
  !     Generalised to sort on multiple arrays with an external ORDERL
  !     function by Axel Brunger, 15-MAR-83
  !
  use chm_kinds
  use memory
  implicit none

  INTEGER   N
  EXTERNAL  EXCHL
  logical,external :: ORDERL
  integer,external :: sort_depth
  INTEGER A1(*),A2(*),A3(*),A4(*),A5(*),A6(*),A7(*),A8(*)
  integer,allocatable,dimension(:) :: STORE

  IF (N /= 0) THEN
     call chmalloc('sort.src','SORT','STORE',sort_depth(N),intg=STORE)
     CALL SORT2(N,EXCHL,ORDERL,A1,A2,A3,A4,A5,A6,A7,A8,STORE)
     call chmdealloc('sort.src','SORT','STORE',sort_depth(N),intg=STORE)
  END IF
  !
  RETURN
END SUBROUTINE SORT

SUBROUTINE SORT2(N,EXCHL,ORDERL,A1,A2,A3,A4,A5,A6,A7,A8,STORE)
  !
  !     See SORT above.
  !     Robert Bruccoleri
  !     Axel Brunger, 15-MAR-83
  !
  use clcg_mod,only:random
  use chm_kinds
  implicit none
  INTEGER   N
  LOGICAL   ORDERL
  EXTERNAL  ORDERL,EXCHL
  INTEGER   A1(*), A2(*), A3(*), A4(*), A5(*), A6(*), A7(*), A8(*)
  INTEGER   STORE(*)
  !
  !
  INTEGER   END0, END1, L1, L2, START, START2, TOP
  INTEGER   DUMMY, RNDIND
  LOGICAL   CONDIT
  !
  DATA DUMMY/0/
  !
  TOP=0
  START=1
  END0=N
  !
3333 IF(END0 == START) GOTO 1111
  IF(END0 /= START+1) GOTO 2222
  IF (.NOT. ORDERL(START,END0,A1,A2,A3,A4,A5,A6,A7,A8)) THEN
     CALL EXCHL(START,END0,A1,A2,A3,A4,A5,A6,A7,A8)
  END IF
1111 IF(TOP == 0) GOTO 9999
  END0=STORE(TOP)
  START=STORE(TOP-1)
  TOP=TOP-2
  GOTO 3333
  !
2222 CONTINUE
  GOTO 4444                 ! PROCESS-PARTITION-SORT2
4445 CONTINUE                  ! return from PROCESS-PARTITION-SORT2
  L1=END1-START
  L2=END0-START2
  IF (L1 <= L2) THEN
     !
     !     L2>=L1 DO L1 FIRST
     !
     TOP=TOP+2
     STORE(TOP-1)=START2
     STORE(TOP)=END0
     END0=END1
  ELSE
     TOP=TOP+2
     STORE(TOP-1)=START
     STORE(TOP)=END1
     START=START2
  END IF
  GOTO 3333
  !
9999 CONTINUE
  !
  RETURN
  !
4444 CONTINUE                  ! TO PROCESS-PARTITION-SORT2
  !
  RNDIND=INT((END0-START)*RANDOM(DUMMY))+START
  START2=START-1
  END1=END0+1
  !
101 START2=START2+1
  CONDIT=ORDERL(START2,RNDIND,A1,A2,A3,A4,A5,A6,A7,A8)
  IF (CONDIT .AND. START2 /= END0) GOTO 101
  !
201 END1=END1-1
  CONDIT=ORDERL(RNDIND,END1,A1,A2,A3,A4,A5,A6,A7,A8)
  IF (CONDIT .AND. END1 /= START) GOTO 201
  !
  IF (START2 < END1) THEN
     CALL EXCHL(START2,END1,A1,A2,A3,A4,A5,A6,A7,A8)
     GOTO 101
  END IF
  !
  IF (START2 < RNDIND) THEN
     CALL EXCHL(START2,RNDIND,A1,A2,A3,A4,A5,A6,A7,A8)
  ELSE
     IF (RNDIND < END1) THEN
        CALL EXCHL(END1,RNDIND,A1,A2,A3,A4,A5,A6,A7,A8)
     END IF
  END IF
  !
  GOTO 4445                 ! FIN PROCESS-PARTITION-SORT2
  !
END SUBROUTINE SORT2

LOGICAL FUNCTION ORDER5(K,L,X1,X2,X3,X4,X5,X6,X7,WIDTH)
  !
  !     True if X1(K) <= X1(L),...,XWIDTH(K) <= XWIDTH(L), K <= L.
  !     (in canonical order)
  !
  !     Axel Brunger, 15-MAR-83
  !
  use chm_kinds
  implicit none
  INTEGER   K, L
  INTEGER X1(*), X2(*), X3(*), X4(*), X5(*), X6(*), X7(*)
  INTEGER   WIDTH
  !
  !
  !
  ORDER5=(X1(K) <= X1(L))
  IF (WIDTH /= 1 .AND. X1(K) == X1(L)) THEN
     ORDER5=(X2(K) <= X2(L))
     IF (WIDTH /= 2 .AND. X2(K) == X2(L)) THEN
        ORDER5=(X3(K) <= X3(L))
        IF (WIDTH /= 3 .AND. X3(K) == X3(L)) THEN
           ORDER5=(X4(K) <= X4(L))
           IF (WIDTH /= 4 .AND. X4(K) == X4(L)) THEN
              ORDER5=(X5(K) <= X5(L))
              IF (WIDTH /= 5 .AND. X5(K) == X5(L)) THEN
                 ORDER5=(X6(K) <= X6(L))
                 IF (WIDTH /= 6 .AND. X6(K) == X6(L)) THEN
                    ORDER5=(X7(K) <= X7(L))
                    IF (WIDTH == 7.AND.X7(K).EQ.X7(L)) ORDER5=(K <= L)
                 END IF
                 IF (WIDTH == 6.AND.X6(K).EQ.X6(L)) ORDER5=(K <= L)
              END IF
              IF (WIDTH == 5.AND.X5(K).EQ.X5(L)) ORDER5=(K <= L)
           END IF
           IF (WIDTH == 4.AND.X4(K).EQ.X4(L)) ORDER5=(K <= L)
        END IF
        IF (WIDTH == 3.AND.X3(K).EQ.X3(L)) ORDER5=(K <= L)
     END IF
     IF (WIDTH == 2.AND.X2(K).EQ.X2(L)) ORDER5=(K <= L)
  END IF
  IF (WIDTH == 1.AND.X1(K).EQ.X1(L)) ORDER5=(K <= L)
  !
  RETURN
END FUNCTION ORDER5

#if KEY_CMAP==1
LOGICAL FUNCTION ORDER8(K,L,X1,X2,X3,X4,X5,X6,X7,X8)
  !
  !     True if X1(K) <= X1(L),...,X8(K) <= X8(L), K <= L.
  !     (in canonical order)
  !
  !
  use chm_kinds
  implicit none
  INTEGER K, L
  INTEGER X1(*),X2(*),X3(*),X4(*),X5(*),X6(*),X7(*),X8(*)
  INTEGER WIDTH
  !
  !
  !
  ORDER8=(X1(K) <= X1(L))
  IF (X1(K) == X1(L)) THEN
     ORDER8=(X2(K) <= X2(L))
     IF (X2(K) == X2(L)) THEN
        ORDER8=(X3(K) <= X3(L))
        IF (X3(K) == X3(L)) THEN
           ORDER8=(X4(K) <= X4(L))
           IF (X4(K) == X4(L)) THEN
              ORDER8=(X5(K) <= X5(L))
              IF (X5(K) == X5(L)) THEN
                 ORDER8=(X6(K) <= X6(L))
                 IF (X6(K) == X6(L)) THEN
                    ORDER8=(X7(K) <= X7(L))
                    IF (X7(K) == X7(L)) THEN
                       ORDER8=(X8(K) <= X8(L))
                       IF (X8(K) == X8(L)) ORDER8=(K <= L)
                    END IF
                    IF (X7(K) == X7(L)) ORDER8=(K <= L)
                 END IF
                 IF (X6(K) == X6(L)) ORDER8=(K <= L)
              END IF
              IF (X5(K) == X5(L)) ORDER8=(K <= L)
           END IF
           IF (X4(K) == X4(L)) ORDER8=(K <= L)
        END IF
        IF (X3(K) == X3(L)) ORDER8=(K <= L)
     END IF
     IF (X2(K) == X2(L)) ORDER8=(K <= L)
  END IF
  IF (X1(K) == X1(L)) ORDER8=(K <= L)
  !
  RETURN
END FUNCTION ORDER8
#endif 

!----------------------------------------------------------------------

SUBROUTINE EXCH5(INDEX1,INDEX2,A1,A2,A3,A4,A5,A6,A7,WIDTH)
  !
  !     Exchanges A1(INDEX1) and A1(INDEX2),...,
  !                           AWIDTH(INDEX1) and AWIDTH(INDEX2) .
  !
  !     NOV-82 Axel Brunger
  !
  use chm_kinds
  implicit none
  INTEGER A1(*), A2(*), A3(*), A4(*), A5(*), A6(*), A7(*), EXCHG
  INTEGER   INDEX1, INDEX2, WIDTH
  !
  !
  !
  EXCHG=A1(INDEX1)
  A1(INDEX1)=A1(INDEX2)
  A1(INDEX2)=EXCHG
  IF (WIDTH /= 1) THEN
     EXCHG=A2(INDEX1)
     A2(INDEX1)=A2(INDEX2)
     A2(INDEX2)=EXCHG
     IF (WIDTH /= 2) THEN
        EXCHG=A3(INDEX1)
        A3(INDEX1)=A3(INDEX2)
        A3(INDEX2)=EXCHG
        IF (WIDTH /= 3) THEN
           EXCHG=A4(INDEX1)
           A4(INDEX1)=A4(INDEX2)
           A4(INDEX2)=EXCHG
           IF (WIDTH /= 4) THEN
              EXCHG=A5(INDEX1)
              A5(INDEX1)=A5(INDEX2)
              A5(INDEX2)=EXCHG
              IF (WIDTH /= 5) THEN
                 EXCHG=A6(INDEX1)
                 A6(INDEX1)=A6(INDEX2)
                 A6(INDEX2)=EXCHG
                 IF (WIDTH /= 6) THEN
                    EXCHG=A7(INDEX1)
                    A7(INDEX1)=A7(INDEX2)
                    A7(INDEX2)=EXCHG
                 END IF
              END IF
           END IF
        END IF
     END IF
  END IF
  RETURN
END SUBROUTINE EXCH5

#if KEY_CMAP==1
SUBROUTINE EXCH8(INDEX1,INDEX2,A1,A2,A3,A4,A5,A6,A7,A8)
  !
  !     Exchanges A1(INDEX1) and A1(INDEX2),...,
  !                           A8(INDEX1) and A8(INDEX2) .
  !
  use chm_kinds
  implicit none
  INTEGER A1(*), A2(*), A3(*), A4(*), A5(*), A6(*), A7(*), A8(*)
  INTEGER EXCHG
  INTEGER INDEX1, INDEX2
  !
  !
  !
  EXCHG=A1(INDEX1)
  A1(INDEX1)=A1(INDEX2)
  A1(INDEX2)=EXCHG
  EXCHG=A2(INDEX1)
  A2(INDEX1)=A2(INDEX2)
  A2(INDEX2)=EXCHG
  EXCHG=A3(INDEX1)
  A3(INDEX1)=A3(INDEX2)
  A3(INDEX2)=EXCHG
  EXCHG=A4(INDEX1)
  A4(INDEX1)=A4(INDEX2)
  A4(INDEX2)=EXCHG
  EXCHG=A5(INDEX1)
  A5(INDEX1)=A5(INDEX2)
  A5(INDEX2)=EXCHG
  EXCHG=A6(INDEX1)
  A6(INDEX1)=A6(INDEX2)
  A6(INDEX2)=EXCHG
  EXCHG=A7(INDEX1)
  A7(INDEX1)=A7(INDEX2)
  A7(INDEX2)=EXCHG
  EXCHG=A8(INDEX1)
  A8(INDEX1)=A8(INDEX2)
  A8(INDEX2)=EXCHG
  RETURN
END SUBROUTINE EXCH8
#endif 

#if KEY_DIMS==1

  SUBROUTINE DIMSEXCH5R(INDEX1,INDEX2,A1,A2,A3,A4,A5,A6,A7,WIDTH)
    !
    !     Exchanges A1(INDEX1) and A1(INDEX2),...,
    !                           AWIDTH(INDEX1) and AWIDTH(INDEX2) .
    !
    !     NOV-82 Axel Brunger
    !
    !     Jul-07 Juan R. Perilla
    !
    use chm_kinds
    real(chm_real)   A1(*), A3(*), A5(*), A7(*)
    INTEGER  A2(*), A4(*), A6(*)
    real(chm_real) EXCHG
    INTEGER   INDEX1, INDEX2, WIDTH
!
!
!
    EXCHG=A1(INDEX1)
    A1(INDEX1)=A1(INDEX2)
    A1(INDEX2)=EXCHG
    IF (WIDTH /= 1) THEN
       EXCHG=A2(INDEX1)
       A2(INDEX1)=A2(INDEX2)
       A2(INDEX2)=EXCHG
       IF (WIDTH /= 2) THEN
          EXCHG=A3(INDEX1)
          A3(INDEX1)=A3(INDEX2)
          A3(INDEX2)=EXCHG
          IF (WIDTH /= 3) THEN
             EXCHG=A4(INDEX1)
             A4(INDEX1)=A4(INDEX2)
             A4(INDEX2)=EXCHG
             IF (WIDTH /= 4) THEN
                EXCHG=A5(INDEX1)
                A5(INDEX1)=A5(INDEX2)
                A5(INDEX2)=EXCHG
                IF (WIDTH /= 5) THEN
                   EXCHG=A6(INDEX1)
                   A6(INDEX1)=A6(INDEX2)
                   A6(INDEX2)=EXCHG
                   IF (WIDTH /= 6) THEN
                      EXCHG=A7(INDEX1)
                      A7(INDEX1)=A7(INDEX2)
                      A7(INDEX2)=EXCHG
                   END IF
                END IF
             END IF
          END IF
       END IF
    END IF
    RETURN
  END SUBROUTINE DIMSEXCH5R

!----------------------------------------------------------------------

  LOGICAL FUNCTION DIMSORDERR(K,L,ARRAY,D1,D2,D3,D4,D5,D6,D7)
    !
    !     SORT order function for REAL's
    !     Axel Brunger, 15-MAR-83
    !
    !     Jul-07 Juan R. Perilla
    !
    use chm_kinds

    INTEGER   K, L
    real(chm_real) ARRAY(*)
    INTEGER   D1(*), D3(*), D5(*), D7(*)
    real(chm_real)  D2(*), D4(*), D6(*)


!
    INTEGER   I
    !
    !     I=0
    !     REPEAT UNTIL (ARRAY(I,K) /= ARRAY(I,L).OR.I >= WIDTH)
    !
    IF( ARRAY(K)  /=  ARRAY(L) ) THEN
       DIMSORDERR=(ARRAY(K) <= ARRAY(L))
       GOTO 9999
    END IF
    !
    !     FIN
    !
    DIMSORDERR=K <= L
    !
9999 CONTINUE
    RETURN
  END FUNCTION DIMSORDERR

  !-------------------------------------------------------------

#endif

!----------------------------------------------------------------------

LOGICAL FUNCTION ORDER(K,L,ARRAY,WIDTH,D1,D2,D3,D4,D5,D6)
  !
  !     SORT order function for INTEGER's
  !     Axel Brunger, 15-MAR-83
  !
  use chm_kinds
  implicit none
  INTEGER   WIDTH
  INTEGER   K, L, ARRAY(WIDTH,*), D1, D2, D3, D4, D5, D6
  !
  INTEGER   I
  !
  DO I = 1, WIDTH
     IF(ARRAY(I,K)  /=  ARRAY(I,L)) THEN
        ORDER = (ARRAY(I,K)  <=  ARRAY(I,L))
        GOTO 9999
     END IF
  END DO
  ORDER = K <= L
  !
9999 RETURN
END FUNCTION ORDER

LOGICAL FUNCTION ORDERI8(K,L,ARRAY,WIDTH,D1,D2,D3,D4,D5,D6)
  !
  !     SORT order function for chm_int8, copied from ORDER
  !     Axel Brunger, 15-MAR-83
  !
  use chm_kinds
  implicit none
  INTEGER   WIDTH
  INTEGER   K, L, D2, D3, D4, D5, D6
  integer(chm_int8) :: ARRAY(WIDTH,*), D1
  !
  INTEGER   I
  !
  DO I = 1, WIDTH
     IF(ARRAY(I,K)  /=  ARRAY(I,L)) THEN
        ORDERI8 = (ARRAY(I,K)  <=  ARRAY(I,L))
        GOTO 9999
     END IF
  END DO
  ORDERI8 = K <= L
  !
9999 RETURN
END FUNCTION ORDERI8

SUBROUTINE EXCH(K,L,ARRAY,WIDTH,D1,D2,D3,D4,D5,D6)
  !
  !     SORT exchange function for INTEGER's
  !     Axel Brunger, 18-MAR-83
  !
  use chm_kinds
  implicit none
  INTEGER   WIDTH
  INTEGER   K, L, ARRAY(WIDTH,*), D1, D2, D3, D4, D5, D6
  !
  INTEGER   I, ITEMP
  !
  DO I = 1, WIDTH
     ITEMP=ARRAY(I,K)
     ARRAY(I,K)=ARRAY(I,L)
     ARRAY(I,L)=ITEMP
  END DO
  !
  RETURN
END SUBROUTINE EXCH
!----------------------------------------------------------------------

LOGICAL FUNCTION ORDERR(K,L,ARRAY,WIDTH,D1,D2,D3,D4,D5,D6)
  !
  !     SORT order function for REAL's
  !     Axel Brunger, 15-MAR-83
  !
  use chm_kinds
  implicit none
  INTEGER   WIDTH
  INTEGER   K, L
  real(chm_real)    ARRAY(WIDTH,*), D1, D2, D3, D4, D5, D6
  !
  INTEGER   I
  !
  !     I=0
  !     REPEAT UNTIL (ARRAY(I,K) /= ARRAY(I,L).OR.I >= WIDTH)
  !
  DO I = 1, WIDTH
     IF( ARRAY(I,K)  /=  ARRAY(I,L) ) THEN
        ORDERR=(ARRAY(I,K) <= ARRAY(I,L))
        GOTO 9999
     END IF
  END DO
  !
  !     FIN
  !
  ORDERR=K <= L
  !
9999 CONTINUE
  RETURN
END FUNCTION ORDERR
!----------------------------------------------------------------------

SUBROUTINE EXCHR(K,L,ARRAY,WIDTH,D1,D2,D3,D4,D5,D6)
  !
  !     SORT exchange routine for REAL's
  !     Axel Brunger, 18-MAR-83
  !
  use chm_kinds
  implicit none
  INTEGER   WIDTH
  INTEGER   K, L
  real(chm_real)    ARRAY(WIDTH,*), D1, D2, D3, D4, D5, D6
  !
  INTEGER   I
  real(chm_real)    ITEMP
  !
  DO I = 1, WIDTH
     ITEMP=ARRAY(I,K)
     ARRAY(I,K)=ARRAY(I,L)
     ARRAY(I,L)=ITEMP
  END DO
  !
  RETURN
END SUBROUTINE EXCHR

SUBROUTINE SORTAG(A,N,TAG)
  !
  use chm_kinds
  implicit none
  INTEGER N,TAG(*)
  real(chm_real) A(N)
  !
  real(chm_real) T, TT
  INTEGER I, L, M, J, K, IJ
  INTEGER TG, IU(16), IL(16)
  !
  DO  I=1,N
     TAG(I)=I
  enddo
  M=1
  I=1
  J=N
5 IF(I >= J) GOTO 70
10 K=I
  IJ=(J+I)/2
  T=A(IJ)
  IF(A(I) <= T) GOTO 20
  A(IJ)=A(I)
  A(I)=T
  T=A(IJ)
  TG=TAG(IJ)
  TAG(IJ)=TAG(I)
  TAG(I)=TG
20 L=J
  IF(A(J) >= T) GOTO 40
  A(IJ)=A(J)
  A(J)=T
  T=A(IJ)
  TG=TAG(IJ)
  TAG(IJ)=TAG(J)
  TAG(J)=TG
  IF(A(I) <= T) GOTO 40
  A(IJ)=A(I)
  A(I)=T
  T=A(IJ)
  TG=TAG(IJ)
  TAG(IJ)=TAG(I)
  TAG(I)=TG
  GOTO 40
30 A(L)=A(K)
  A(K)=TT
  TG=TAG(L)
  TAG(L)=TAG(K)
  TAG(K)=TG
40 L=L-1
  IF(A(L) > T) GOTO 40
  TT=A(L)
50 K=K+1
  IF(A(K) < T) GOTO 50
  IF(K <= L) GOTO 30
  IF(L-I <= J-K) GOTO 60
  IL(M)=I
  IU(M)=L
  I=K
  M=M+1
  GOTO 80
60 IL(M)=K
  IU(M)=J
  J=L
  M=M+1
  GOTO 80
70 M=M-1
  IF(M == 0) RETURN
  I=IL(M)
  J=IU(M)
80 IF(J-I >= 1) GOTO 10
  IF(I == 1) GOTO 5
  I=I-1
90 I=I+1
  IF(I == J) GOTO 70
  T=A(I+1)
  IF(A(I) <= T) GOTO 90
  TG=TAG(I+1)
  K=I
100 A(K+1)=A(K)
  TAG(K+1)=TAG(K)
  K=K-1
  IF(T < A(K)) GOTO 100
  A(K+1)=T
  TAG(K+1)=TG
  GOTO 90
END SUBROUTINE SORTAG
!
SUBROUTINE INDIXX(N,ARRIN,INDX)
  !------------------------------------------
  ! From "Numerical Recipes", based on QuickSort.
  ! Indexes an array ARRIN(1:N), i.e., outputs the array INDX(1:N) such that
  ! ARRIN(INDX(j)) is in ascending order for j=1,2,...,N.
  ! The input quantities N and ARRIN are not changed.
  !
  !
  ! global
  use chm_kinds
  implicit none
  ! passed
  INTEGER ARRIN(*),Q
  INTEGER INDX(*),N
  ! local
  INTEGER I,J,L,IR,INDXT
  !
  ! begin
  !
  DO J=1,N
     INDX(J)=J
  enddo
  L=N/2+1
  IR=N
10 CONTINUE
  IF(L > 1)THEN
     L=L-1
     INDXT=INDX(L)
     Q=ARRIN(INDXT)
  ELSE
     INDXT=INDX(IR)
     Q=ARRIN(INDXT)
     INDX(IR)=INDX(1)
     IR=IR-1
     IF(IR == 1)THEN
        INDX(1)=INDXT
        RETURN
     ENDIF
  ENDIF
  I=L
  J=L+L
20 IF(J <= IR)THEN
     IF(J < IR)THEN
        IF(ARRIN(INDX(J)) < ARRIN(INDX(J+1)))J=J+1
     ENDIF
     IF(Q < ARRIN(INDX(J)))THEN
        INDX(I)=INDX(J)
        I=J
        J=J+J
     ELSE
        J=IR+1
     ENDIF
     GO TO 20
  ENDIF
  INDX(I)=INDXT
  GO TO 10
END SUBROUTINE INDIXX

!---- Sort ARRAY(1..N) in ascending order via Quicksort ---------
!  extracted from nbndgc and modified - mkg 2009
subroutine qsort_ints(ARRAY, N)
 integer,intent(in) :: N
 integer,dimension(N),intent(inout) :: ARRAY

 logical :: DONE
 integer :: RECURS
 real :: RNDVAL
 ! array indices
 integer,dimension(N) :: BOUNDS
 integer :: LO, HI, PRL, PRH, RNDIND
 ! array element values
 integer :: SWP, PIVOT
 !
 !     In any given iteration of Quicksort the task is to sort the
 !     subarray in the range of indices LO..HI.
 !
 !     The basic operation is "partitioning": First we choose a
 !     PIVOT, which is a value that hopefully will split the array
 !     into two pieces of roughly equal size.  Using a pair of
 !     cursors PRL and PRH we sweep from the edges and proceed
 !     inward, swapping elements between the two cursors as
 !     necessary until they meet in the middle.  At that point,
 !     everything above the cursors exceeds PIVOT, and everthing
 !     below the cursor is less than PIVOT.
 !
 !     More precisely:  at the end of the partitioning, we must
 !     end up with ARRAY(LO..PRL)  <=  PIVOT, and
 !     ARRAY(PRH+1..HI)  >=  PIVOT.  If PIVOT happens to be the
 !     largest element in ARRAY, then (PRL,PRH) should end up equal to
 !     (HI,HI).  If PIVOT happens to be the smallest
 !     element in ARRAY, then (PRL,PRH) should end up equal to
 !     (LO,LO).
 !
 !     This particular implementation is taken from Sedgewick,
 !     "Algorithms", Addison-Wesley, 1988, p. 118.  It is modified
 !     so that arrays of length 2 or less are handled as special
 !     cases, and that pivots are chosen at random.
 !
 !     After partitioning we sort the arrays LO..PRL and
 !     PRH+1..HI recursively.  Since FORTRAN is not recursive,
 !     we must emulate recursion using a stack.  I use BOUNDS for
 !     this:  I save the bounds of one subarray in a pair of
 !     locations in ARRAY, then go on to do the other subarray
 !     immediately.
 !
 LO = 1
 HI = N

 RECURS = 0
 !
 !     At the top of the loop we have a number of subarrays to sort:
 !     LO..HI, and anything on the stack.  Our top priority
 !     is to sort LO..HI; the stack contains deferred tasks.
 !     The body of the loop is a giant IF-ELSE:  either
 !     LO..HI is small enough to be done by a direct method
 !     or it must be done by recursive Quicksort.  In the former
 !     case, we end by grabbing a new task from the stack.
 !
 !     None of the loops in this sort are vectorizable.
 !
 DONE = .FALSE.
 DO WHILE (.NOT. DONE)
    !
    IF (HI - LO <= 1) THEN

       !           If exactly two elements, swap if necessary
       IF (HI - LO == 1) THEN ! Exactly two elems
          IF (ARRAY(LO) > ARRAY(HI)) THEN
             SWP = ARRAY(LO)
             ARRAY(LO) = ARRAY(HI)
             ARRAY(HI) = SWP
          ENDIF
       ENDIF

       !           LO..HI is done, so grab next task off stack
       DONE = (RECURS < 2)
       IF (.NOT. DONE) THEN
          LO = BOUNDS(RECURS - 1)
          HI = BOUNDS(RECURS)
          RECURS = RECURS - 2
       ENDIF

    ELSE

       !           Choose PIVOT at random from among elements in subarray
       CALL RANDOM_NUMBER(RNDVAL)
       RNDIND = INT((HI - LO + 0.9) * RNDVAL) + LO
       PIVOT = ARRAY(RNDIND)
       ! Move PIVOT to HI so we're free to rearrange other elements
       ARRAY(RNDIND) = ARRAY(HI)
       ARRAY(HI) = PIVOT

       !           Partition subarray LO..HI-1 about PIVOT by
       !           sweeping inward from edges using cursors PRL and PRH
       PRL = LO
       PRH = HI - 1
       DO
          DO WHILE (PRL < HI .AND. ARRAY(PRL) < PIVOT)
             PRL = PRL + 1
          ENDDO
          DO WHILE (PRH > LO .AND. ARRAY(PRH) > PIVOT)
             PRH = PRH - 1
          ENDDO
          IF (PRL >= PRH) EXIT
          ! elements PRL and PRH are each on the wrong side of PIVOT
          SWP = ARRAY(PRL)
          ARRAY(PRL) = ARRAY(PRH)
          ARRAY(PRH) = SWP
          PRL = PRL + 1
          PRH = PRH - 1
       ENDDO
       IF (PRL < HI) THEN
          ! restore PIVOT to its proper place
          SWP = ARRAY(PRL)
          ARRAY(PRL) = ARRAY(HI)
          ARRAY(HI) = SWP
       ENDIF

       !           Save larger subarray on stack for later sorting;
       !           return to sort smaller one immediately
       RECURS = RECURS + 2
       IF (PRL - LO > HI - PRL) THEN
          BOUNDS(RECURS - 1) = LO
          BOUNDS(RECURS) = PRL - 1
          LO = PRL + 1
       ELSE
          BOUNDS(RECURS - 1) = PRL + 1
          BOUNDS(RECURS) = HI
          HI = PRL - 1
       ENDIF

    ENDIF

 ENDDO
end subroutine qsort_ints
!--------------------------------------------------------------------
  SUBROUTINE qisort(a,n,t)
  use chm_kinds

    !     NON-RECURSIVE STACK VERSION OF QUICKSORT FROM N.WIRTH'S PASCAL
    !     BOOK, 'ALGORITHMS + DATA STRUCTURES = PROGRAMS'.

    !     SINGLE PRECISION, ALSO CHANGES THE ORDER OF THE ASSOCIATED ARRAY T.

    IMPLICIT NONE

    INTEGER, INTENT(IN)    :: n
!    real(kind=chm_real) :: a(n)
    integer :: a(n)
    INTEGER, INTENT(INOUT) :: t(n)

    !     Local Variables

    INTEGER                :: i, j, k, l, r, s, stackl(15), stackr(15), ww
    real(kind=chm_real)    :: w, x

    s = 1
    stackl(1) = 1
    stackr(1) = n

    !     KEEP TAKING THE TOP REQUEST FROM THE STACK UNTIL S = 0.

10  CONTINUE
    l = stackl(s)
    r = stackr(s)
    s = s - 1

    !     KEEP SPLITTING A(L), ... , A(R) UNTIL L >= R.

20  CONTINUE
    i = l
    j = r
    k = (l+r) / 2
    x = a(k)

    !     REPEAT UNTIL I > J.

    DO
       DO
          IF (a(i).LT.x) THEN                ! Search from lower end
             i = i + 1
             CYCLE
          ELSE
             EXIT
          END IF
       END DO

       DO
          IF (x.LT.a(j)) THEN                ! Search from upper end
             j = j - 1
             CYCLE
          ELSE
             EXIT
          END IF
       END DO

       IF (i.LE.j) THEN                     ! Swap positions i & j
          w = a(i)
          ww = t(i)
          a(i) = a(j)
          t(i) = t(j)
          a(j) = w
          t(j) = ww
          i = i + 1
          j = j - 1
          IF (i.GT.j) EXIT
       ELSE
          EXIT
       END IF
    END DO

    IF (j-l.GE.r-i) THEN
       IF (l.LT.j) THEN
          s = s + 1
          stackl(s) = l
          stackr(s) = j
       END IF
       l = i
    ELSE
       IF (i.LT.r) THEN
          s = s + 1
          stackl(s) = i
          stackr(s) = r
       END IF
       r = j
    END IF

    IF (l.LT.r) GO TO 20
    IF (s.NE.0) GO TO 10
    RETURN
  END SUBROUTINE qisort
