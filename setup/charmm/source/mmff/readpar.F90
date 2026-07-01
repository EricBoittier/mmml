#if KEY_MMFF==1
! ===========================================================
! SUBROUTINE BIFIND : BINARY SEARCH OF 1D ARRAY
!   SEARCH ELEMENTS 1,KK OF ARRAY KV FOR ELEMENT I WHERE
!   KV(I)=K.  BINARY SEARCH IS USED, AS ELEMENTS 1,KK ARE
!   UNDERSTOOD TO BE ORDERED IN INCREASING VALUE.
!   IF BINARY SEARCH FAILS TO FIND ELEMENT I, THEN A STRAIGHT
!   LINEAR SEARCH IS PERFORMED ON ELEMENTS KK+1,KMAX
!   (IF KMAX>KK); THESE ELEMENTS ARE NOT ASSUMED TO BE
!   ORDERED IN VALUE
!   IF ELEMENT I IS FOUND, NEWPAR IS RETURNED AS .FALSE.,
!   INDICATING SUCCESS.  FULL IS ALSO RETURNED AS .FALSE.
!   IF ELEMENT I IS NOT FOUND, NEWPAR IS RETURNED AS .TRUE.
!   IF IROUTE=1 AND KMX<LIM, KMX IS INCREMENTED BY 1, I IS
!   SET EQUAL TO THE INCREMENTED VALUE OF KMX, AND FULL IS
!   RETURNED AS .TRUE.
!   IF IROUTE=2, KMX IS NOT INCREMENTED AND I IS NOT ASSIGNED A
!   VALUE.  FULL IS RETURNED AS .FALSE. IF KMX<LIM
! ===========================================================
!
SUBROUTINE BIFIND(K,KV,KK,I,KMX,LIM,NEWPAR,FULL,IROUTE)
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   Adapted by R. Czerminski of Molecular Simulations, Inc. from
  !  code developed at Merck and Co., Inc. by T. Halgren and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !  16 Oct 95 Jay Banks: changed variable name "KM" to "KK" to avoid
  !                       conflict with CHARMM name
  !
  use chm_kinds
  implicit none
  !
  LOGICAL NEWPAR,FULL
  INTEGER lim,KV(LIM)
  integer i, iroute, k, kk, kmx
  integer NINDX  ! , BSEARCH
  external NINDX ! , BSEARCH
  !
  NEWPAR=.FALSE.
  FULL=.FALSE.

  i=NINDX(K,KV,KK)
  if(i.gt.0) return
  !
  !   BINARY SEARCH FAILED -- CARRY OUT LINEAR SEARCH ON
  !   REMAINING ELEMENTS, IF ANY
  !
  IF(KMX.GT.KK) then
     DO I=KK+1,KMX
        IF(KV(I).EQ.K) return
     enddo
  endif
  !
  !   FELL THROUGH LOOP, SO LINEAR SEARCH ALSO FAILED
  !
  NEWPAR=.TRUE.
  IF(KMX.LT.LIM) THEN
     FULL=.FALSE.
     if(IROUTE.EQ.1) THEN
        KMX=KMX+1
        I=KMX
     endif
  ELSE
     FULL=.TRUE.
  ENDIF
  RETURN
END SUBROUTINE BIFIND

SUBROUTINE BIFIND8(K,KV,KK,I,KMX,LIM,NEWPAR,FULL,IROUTE)
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   Adapted by R. Czerminski of Molecular Simulations, Inc. from
  !  code developed at Merck and Co., Inc. by T. Halgren and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !  16 Oct 95 Jay Banks: changed variable name "KM" to "KK" to avoid
  !                       conflict with CHARMM name
  !
  use chm_kinds
  implicit none
  !
  LOGICAL NEWPAR,FULL
  INTEGER lim
  integer(chm_int8) :: k,kv(lim)
  integer i, iroute, kk, kmx
  integer NINDX8  ! , BSEARCH
  external NINDX8 ! , BSEARCH
  !
  NEWPAR=.FALSE.
  FULL=.FALSE.

  i=NINDX8(K,KV,KK)
  if(i.gt.0) return
  !
  !   BINARY SEARCH FAILED -- CARRY OUT LINEAR SEARCH ON
  !   REMAINING ELEMENTS, IF ANY
  !
  IF(KMX.GT.KK) then
     DO I=KK+1,KMX
        IF(KV(I).EQ.K) return
     enddo
  endif
  !
  !   FELL THROUGH LOOP, SO LINEAR SEARCH ALSO FAILED
  !
  NEWPAR=.TRUE.
  IF(KMX.LT.LIM) THEN
     FULL=.FALSE.
     if(IROUTE.EQ.1) THEN
        KMX=KMX+1
        I=KMX
     endif
  ELSE
     FULL=.TRUE.
  ENDIF
  RETURN
END SUBROUTINE BIFIND8

! =====================================================================
! SUBROUTINE BONDCON : PACK/UNPACK A CANONICALLY-ORDERED INDEX FOR BOND
!   PACK OR UNPACK A CANONICALLY-ORDERED INDEX FOR BOND LENGTHS OR
!   BOND-CHARGE INCREMENTS
! =====================================================================
!
SUBROUTINE BONDCON(PUTGET,I,J,MCLASS,IJM)
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   Adapted by R. Czerminski of Molecular Simulations, Inc. from
  !  code developed at Merck and Co., Inc. by T. Halgren and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !
  use chm_kinds
  use dimens_fcm
  use ffieldm
  use string
  implicit none
  !
  CHARACTER(len=*) PUTGET ! 'PUT' or 'GET'
  integer i, j         ! atom types
  integer mclass       !? class
  integer ijm          ! packed index if 'PUT'
  !                          ! to be unpacked index if 'GET'
  integer, parameter :: MCMAX=8, MX=MAXDEFI
  integer, parameter :: MAXBTYPE=16383 ! Maximum atom type for which packing
  ! formula fits in a 4-byte integer

  real(chm_real), parameter :: HALF=0.5D0, TWO=2.0D0
  INTEGER THREE
  THREE(I,J,MCLASS)=MCLASS+MCMAX*(MX*MIN(I,J)+MAX(I,J))
  !
  if (MX .gt. MAXBTYPE) CALL WRNDIE(-5, '<bondcon>', &
       'MAXDEFI too big for index packing')
  !
  IF(PUTGET.EQ.'PUT') THEN      ! PACK THE INDEX
     IJM=THREE(I,J,MCLASS)
  ELSE IF(PUTGET.EQ.'GET') THEN ! UNPACK THE INDEX
     I=IJM/MX/MCMAX           ! TO RECOVER I,J,MCLASS
     J=(IJM-MCMAX*MX*I)/MCMAX
     MCLASS=(IJM-MCMAX*MX*I-MCMAX*J)
  ELSE
     CALL WRNDIE(-5,'<bondcon>','Invalid PUTGET string')
  ENDIF
#if KEY_DEBUG==1
  if(MCLASS.ge.MCMAX .or. MCLASS.lt.0) then
     write(SCRTCH,'(a,2i5)') 'invalid MCLASS; MCLASS,MCMAX=', &
          MCLASS,MCMAX
     call wrndie(-5,'<bondcon>',SCRTCH(:39))
  endif
  if(I.ge.MX .or. J.ge.MX .or. I.lt.0 .or. J.lt.0) then
     write(SCRTCH,'(a,3i5)') 'Invalid I or J; I,J,MX=',I,J,MX
     call wrndie(-5,'<bondcon>',SCRTCH(:48))
  endif
#endif 
  RETURN
END SUBROUTINE BONDCON

!
! Jay Banks 01 Nov 95: copied from util/find_loc.f in MSI version.
! Jay Banks 22 Nov 95: replaced in-line implicit none with ##INCLUDE
!
SUBROUTINE ffind_loc(UNIT,KEY,EOF) ! 
  ! finds first line in the UNIT  beginning with KEY string
  !  returns EOF = true if line  was not found
  use chm_kinds
  implicit none
  integer UNIT
  character(len=*) KEY
  logical EOF
  !
  character(len=80) line
  rewind(UNIT)
  EOF=.FALSE.
  do while (.not.EOF)
     read(UNIT,'(a)',end=800) line
     if(line(1:len(KEY)).eq.KEY) return
  enddo
800 EOF=.TRUE.
  !
  RETURN
END SUBROUTINE ffind_loc

! =======================================================
! SUBROUTINE OMEGCON : PACK/UNPACK A CANONICALLY-ORDERED INDEX FOR OMEGA
! (DIHEDRAL) ANGLES
! =======================================================
!
SUBROUTINE OMEGCON(PUTGET,I,J,K,L,MCLASS,IJKLM)
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   Adapted by R. Czerminski of Molecular Simulations, Inc. from
  !  code developed at Merck and Co., Inc. by T. Halgren and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !
  use chm_kinds
  use dimens_fcm
  implicit none
  !
  CHARACTER(len=*) PUTGET
  integer(chm_int8) :: ijklm
  integer i, j, k, l, mclass

  integer, parameter :: MCMAX=6, MX=MAXDEFI/2
  integer, parameter :: MAXOTYPE=137 ! Maximum atom type for which packing
  ! formula fits in a 4-byte integera
  !
  INTEGER FIVE
  FIVE(I,J,K,L,MCLASS)= &
       MCLASS+MCMAX*(MX**3*J+MX**2*K+MX*I+L)
  !
  if (MX .gt. MAXOTYPE) CALL WRNDIE(-5, '<omegcon>', &
       'MAXDEFI too big for index packing')
  !
  IF(PUTGET.EQ.'PUT'.OR.PUTGET.EQ.'put') THEN ! PACK THE INDEX
     !
     IJKLM=MIN(FIVE(I,J,K,L,MCLASS),FIVE(L,K,J,I,MCLASS))
  ELSE IF(PUTGET.EQ.'GET'.or.PUTGET.EQ.'get') THEN
     !
     !   UNPACK THE INDEX TO RECOVER I,J,K,L,MCLASS
     !
     J=IJKLM/MX**3/MCMAX
     K=(IJKLM-MCMAX*MX**3*J)/MX**2/MCMAX
     I=(IJKLM-MCMAX*MX**3*J-MCMAX*MX**2*K)/MX/MCMAX
     L=(IJKLM-MCMAX*MX**3*J-MCMAX*MX**2*K-MCMAX*MX*I)/MCMAX
     MCLASS=(IJKLM-MCMAX*MX**3*J-MCMAX*MX**2*K-MCMAX*MX*I-MCMAX*L)
  ELSE
     CALL WRNDIE(-5,'<omegcon>','Invalid PUTGET string')
  ENDIF
#if KEY_DEBUG==1
  if(MCLASS.ge.MCMAX .or. MCLASS.lt.0) &
       call wrndie(-5,'<omegcon>','Invalid MCLASS')
  if(I.GE.MX .or. J.GE.MX .or. K.GE.MX .or. L.GE.MX .or. &
       I.LT.0  .or. J.LT.0  .or. K.LT.0  .or. L.LT.0 ) &
       call wrndie(-5,'<omegcon>','Invalid I,J,K or L')
#endif 
  !
  RETURN
END SUBROUTINE OMEGCON

! =======================================================
! SUBROUTINE OOPCON : PACK/UNPACK A CANONICALLY-ORDERED INDEX FOR OOPL
! OUT-OF-PLANE ANGLES
! =======================================================
!
SUBROUTINE OOPCON(PUTGET,I,J,K,L,IJKL)
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   Adapted by R. Czerminski of Molecular Simulations, Inc. from
  !  code developed at Merck and Co., Inc. by T. Halgren and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !
  use chm_kinds
  use dimens_fcm
  implicit none
  !
  CHARACTER(len=*) PUTGET
  integer i, ijkl, j, k, l
  integer, parameter :: MX=MAXDEFI/2
  integer, parameter :: MAXOPTYPE=215 ! Maximum atom type for which packing
  ! formula fits in a 4-byte integer
  !
  INTEGER FOUR
  FOUR(I,J,K,L)=MX**3*J+MX**2*I+MX*K+L
  !
  if (MX .gt. MAXOPTYPE) CALL WRNDIE(-5, '<oopcon>', &
       'MAXDEFI too big for index packing')
  !
  IF(PUTGET.EQ.'PUT') THEN ! PACK THE INDEX
     !
     IJKL=MIN(FOUR(I,J,K,L),FOUR(K,J,I,L), &
          FOUR(L,J,I,K),FOUR(I,J,L,K), &
          FOUR(K,J,L,I),FOUR(L,J,K,I))

  ELSE IF(PUTGET.EQ.'GET') THEN ! UNPACK THE INDEX
     ! TO RECOVER I,J,K,L,MCLASS
     J=IJKL/MX**3
     I=(IJKL-MX**3*J)/MX**2
     K=(IJKL-MX**3*J-MX**2*I)/MX
     L=(IJKL-MX**3*J-MX**2*I-MX*K)
  ELSE
     CALL WRNDIE(-5,'<oopcon>','Invalid PUTGET string')
  ENDIF
#if KEY_DEBUG==1
  if(I.GE.MX .or. J.GE.MX .or. K.GE.MX .or. L.GE.MX .or. &
       I.lt.0  .or. J.lt.0  .or. K.lt.0  .or. L.lt.0  ) &
       call wrndie(-5,'<oopcon>','I,J,K or L out of range ')
#endif 
  !
  RETURN
END SUBROUTINE OOPCON

! ===========================================================
! SUBROUTINE RDAROM: READ IN AROMATIC SYMBOLIC TYPES
! AS A FUNCTION OF THE ORIGINALLY ASSIGNED SYMBOLIC TYPES
! ===========================================================
SUBROUTINE RDAROM(LUTRAROM,ERROR)
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   Adapted by R. Czerminski of Molecular Simulations, Inc. from
  !  code developed at Merck and Co., Inc. by T. Halgren and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !
  !  Jay Banks 22 Nov 95: added IF (PRNLEV.GE.2) to WRITE statement
  !
  !  Jay Banks 27 Nov 95: changed above to IF (WRNLEV.GE.2), because
  !  followed by CALL WRNDIE
  !
  !  28 Nov 95 Jay Banks: removed space from all "GO TO"s, as suggested in
  !  developer.doc.
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use mmffm
  use param
  use stream
  implicit none
  !
  integer lutrarom
  logical error
  !
  CHARACTER(len=80) LINE
  !     CHARACTER*80 TRAROMF
  !
  !   FIRST GET AND OPEN THE FILE
  !
  !     CALL PFILES('TRAROM',TRAROMF,LTRAROM,LUTRAROM,OKPAR)
  !     IF(.NOT.OKPAR) RETURN
  error=.false.
  NTRAROM=0
10 CONTINUE
  READ(LUTRAROM,'(A)') LINE
  IF(LINE(1:1).EQ.'$') GOTO 40
  IF(LINE(1:1).EQ.'*') GOTO 10
  IF(NTRAROM.GE.MAXATC) THEN
     IF (WRNLEV.GE.2) WRITE(OUTU,'(A,I6,A/A)') &
          ' ***** Too parameters read from MMFFAROM '// &
          '... max = ',MAXATC,' *****',' *** Execution terminated ****'
     CALL WRNDIE(-5,'<rdarom>','Too parameters read from'// &
          ' MMFFFAROM')
  ENDIF
  NTRAROM=NTRAROM+1
  READ(LINE,'(A,4X,A,5I5)',ERR=250) &
       SYMOLD(NTRAROM),SYMAROM(NTRAROM), &
       IANUM(NTRAROM),IRSIZE(NTRAROM),L5POS(NTRAROM), &
       LIMCAT(NTRAROM),LN5AN(NTRAROM)
  GOTO 10
40 CONTINUE
  !     LRB=LPATH(TRAROMF)
  !     LEND=LLEN(TRAROMF)
  if(prnlev.ge.2) WRITE(OUTU,'(A,I5,A,I5)') &
       'rdarom>',NTRAROM, &
       ' MMFF AROMATIC SYMBOLIC TYPE DEFINITIONS READ FROM UNIT ', &
       LUTRAROM
  !  50 FORMAT(I5,' MMFF AROMATIC SYMBOLIC TYPE DEFINITIONS ',
  !    . 'READ FROM ',A)
  !     CALL VCLOSE(LUTRAROM,'KEEP',ERROR)
  RETURN
250 CONTINUE
  CALL WRNDIE(-5,'<rdarom>','error on reading parameter file')
  !
  RETURN
END SUBROUTINE RDAROM

! ===========================================================
! SUBROUTINE RDBNDK : READ MMFF DEFAULT-RULE BOND STRETCHING
!                     PARAMETERS
! ===========================================================
!
SUBROUTINE RDBONDK(LUBONDK,ERROR)
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
  use param
  use stream
  use mmffm
  implicit none
  !
  integer na, nb
  real(chm_real) ba, bb
  integer lubondk
  logical error
  !
  CHARACTER(len=80) LINE
  !     CHARACTER*80 BNDCONF
  !     integer llen
  !     external llen
  !
  !   FIRST GET AND OPEN THE FILE
  !
  !     CALL PFILES('BNDCON',BNDCONF,LBNDCON,LUBONDK,OKPAR)
  !     IF(.NOT.OKPAR) RETURN
  error=.false.
  NBNDCON=0
10 CONTINUE
  READ(LUBONDK,'(A)') LINE
  IF(LINE(1:1).EQ.'$') GOTO 40
  IF(LINE(1:1).EQ.'*') GOTO 10
  READ(LINE,'(2I5,2F10.5)',ERR=250) NA,NB,BA,BB
  NBNDCON=NBNDCON+1
  IA(1,NBNDCON)=NA
  IA(2,NBNDCON)=NB
  BL1(NBNDCON)=BA
  BK1(NBNDCON)=BB
  GOTO 10
40 CONTINUE
  !     LRB=LPATH(BNDCONF)
  !     LEND=LLEN(BNDCONF)
  if(prnlev.ge.2) WRITE(OUTU,'(a,i5,a,i5)') &
       'rdbondk> ',NBNDCON, &
       ' MMFF DEFAULT-RULE STRECHING CONSTANTS READ FROM UNIT ', &
       LUBONDK
  !     CALL VCLOSE(LUBONDK,'KEEP',ERROR)
  RETURN
250 CONTINUE
  CALL WRNDIE(-5,'<rdbondk>','error reading parameter file')
  !
  RETURN
END SUBROUTINE RDBONDK

! ======================================================================
! SUBROUTINE RDBONDM : READS AND WRITES STRETCHING CONSTANTS
! READS AND WRITES STANDARD AND SUPPLEMENTARY STRETCHING CONSTANTS
! FOR THE MMFF FORCE FIELD
! ======================================================================
!
SUBROUTINE RDBONDM(NS,LUBOND,ISUPP,ERROR)
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   Adapted by R. Czerminski of Molecular Simulations, Inc. from
  !  code developed at Merck and Co., Inc. by T. Halgren and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !
  !  Jay Banks 22 Nov 95: added IF (PRNLEV.GE.2) to WRITE statement.
  !
  !  Jay Banks 27-Nov-95: changed IF (PRNLEV.GE.2) to IF (WRNLEV.GE.2) for
  !  WRITE statements followed by CALL WRNDIE.
  !
  !  28 Nov 95 Jay Banks: removed space from all "GO TO"s, as suggested in
  !  developer.doc.
  !
  !  28 Nov 95 Jay Banks: changed PRNLEV test to WRNLEV for printing error
  !  conditions.
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use mmffm
  use consta
  use param
  use stream
  implicit none
  !
  integer i, ii, ISUPP
  integer j, k, kk
  !     integer lbond, lend, lrb, lubond, mclass
  integer lubond, mclass
  integer ns, KCBLAST
  real(chm_real)  fc, stdbl
  !
  CHARACTER(len=80) LINE
  !     CHARACTER*80 BONDF
  CHARACTER(len=36) SOURCE
  LOGICAL NEWPAR,FULL,ERROR

  error=.false.
  !
  !   ZERO ARRAYS
  !
  BondFC(1:maxcb)=0.
  BondEq(1:maxcb)=0.
  KCB(1:maxcb)=0
  b_source(1:maxcb)=' '

  !
  !   READ BOND STRETCHING AND BOND DIPOLE PARAMETERS
  !   FIRST GET AND OPEN THE FILE
  !     CALL PFILES('BONDPAR',BONDF,LBOND,LUBOND,OKPAR)
  !     IF(.NOT.OKPAR) RETURN
  NCB=0
  KCBLAST=-1
  loop10: do while(.true.)
     READ(LUBOND,'(A)',END=40) LINE
     IF(LINE(1:1).EQ.'$') exit loop10
     IF(LINE(1:1).EQ.'*') cycle loop10
     READ(LINE,20) MCLASS, I,J,FC,STDBL,SOURCE
20   FORMAT(I1,I4,I5,2F10.3,3X,A)
     IF(NCB.GE.MAXCB) THEN
        IF (WRNLEV.GE.2) WRITE(OUTU,'(A,I6,A/A)') &
             ' ***** Too many bond-stretch parameters read '// &
             '... max = ',MAXCB,' *****',' *** Execution terminated ****'
        CALL WRNDIE(-5,'<rdbondm>','Too many bond-stretch'// &
             ' parameters')
     ENDIF
     NCB=NCB+1
     !
     !   FORM PACKED INDEX
     !
     CALL BONDCON('PUT',I,J,MCLASS,KCB(NCB))
     !
     !   IS CANONICAL ORDER BEING OBEYED?
     !
     IF(KCB(NCB).LE.KCBLAST) THEN
        if(wrnlev.ge.2) WRITE(OUTU,'(A/A/A)') &
             ' *** Canonical order violated: misordered line follows: ***', &
             LINE(1:78),' *** Execution terminated ****'
        CALL WRNDIE(-5,'<rdbondm>','Canonical order violated')
     ELSE
        KCBLAST=KCB(NCB)
     ENDIF
     !
     BondFC(NCB)=FC*MDAKCAL
     BondEq(NCB)=STDBL
     b_source(NCB)=source
  enddo loop10
40 CONTINUE
  !     LRB=LPATH(BONDF)
  !     LEND=LLEN(BONDF)
  if(prnlev.ge.2) WRITE(OUTU,'(a,i5,a,i5)')'rdbondm> ',NCB, &
       ' BOND STRETCHING PARAMETERS READ FROM UNIT ',LUBOND
  !     CALL VCLOSE(LUBOND,'KEEP',ERROR)
  !
700 CONTINUE
  !
  !  READ IN CHANGED STRETCHING CONSTANTS
  !
  NCBT=NCB
  IF (NS.EQ.0) RETURN
  if(prnlev.ge.2) WRITE(OUTU,900) NS
900 FORMAT(I5,' SUPPLEMENTARY STRETCHING PARAMETERS READ')
  call ffind_loc(ISUPP,'BOND',ERROR)
  if(ERROR) then
     call wrndie(-1,'<RDBONDM>','EOF in supplementary file')
     return
  endif
  DO I=1,NS
     READ(ISUPP,1000) MCLASS,II,KK,FC,STDBL,SOURCE
1000 FORMAT(I1,I4,I5,2F10.3,3X,A)
     !  FIND THE BOND IN THE KB-LIST, IF IT'S NOT THERE - ADD IT.
     CALL BONDCON('PUT',II,KK,MCLASS,K)
     CALL BIFIND(K,KCB,NCB,J,NCBT,MAXCB,NEWPAR,FULL,1)
     IF(.NOT.FULL) THEN
        IF(NEWPAR) KCB(J)=K
        BondFC(J)=FC*MDAKCAL
        BondEq(J)=STDBL
     ELSE
        !  IF THIS POINT IS REACHED THE KB ARRAY IS FILLED
        if(wrnlev.ge.2) WRITE(OUTU,1400) I
        !     IF(IZ.NE.0) WRITE(IZ,1400) I
1400    FORMAT(/'* * * * * ERROR - THE ARRAY OF BOND TYPES IS FILLED', &
             '.  BOND CARD #',I3,' IS SKIPPED.')
        !       INIT=1
     ENDIF
  enddo
  !
  RETURN
END SUBROUTINE RDBONDM

! ===========================================================
! SUBROUTINE RDCHGM : READ MMFF BOND-INCREMENT CHARGE PARAMETERS
! ===========================================================
!
SUBROUTINE RDCHGM(NQ,LUCHG,ISUPP,ERROR)
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   Adapted by R. Czerminski of Molecular Simulations, Inc. from
  !  code developed at Merck and Co., Inc. by T. Halgren and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !
  ! Jay Banks 25 Oct 95: changed local variable name IA to IIA, to avoid 
  ! conflict with variable in module (mmff.f90).
  !
  ! Jay Banks 22 Nov 95: added IF (PRNLEV.GE.2) to WRITE statement.
  !
  !  Jay Banks 27-Nov-95: changed IF (PRNLEV.GE.2) to IF (WRNLEV.GE.2) for
  !  WRITE statements followed by CALL WRNDIE.
  !
  !  28 Nov 95 Jay Banks: removed space from all "GO TO"s, as suggested in
  !  developer.doc.
  !
  !  28 Nov 95 Jay Banks: changed PRNLEV test to WRNLEV for printing error
  !  conditions.
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use param
  use stream
  use mmffm
  implicit none
  !
  integer i, iia, ISUPP, it12, j, ja, k
  integer luchg, mclass, n, nq
  integer ITLAST
  real(chm_real)  bciij
  !
  CHARACTER(len=80) LINE
  LOGICAL ERROR,NEWPAR,FULL
  !     CHARACTER*80 CHGFLE
  !     integer llen
  !     external llen
  !
  !    READ MMFF BOND-INCREMENT CHARGE PARAMETERS
  !   FIRST GET AND OPEN THE BOND-INCREMENTS FILE
  !      write(outu,*) ' calling pfiles'
  !     CALL PFILES('CHGPAR',CHGFLE,LCHG,LUCHG,OKPAR)
  !      write(outu,*) ' okpar =',okpar
  !     IF(.NOT.OKPAR) RETURN
  error=.false.
  NCQ=0
  ITLAST=-1
10 CONTINUE
  READ(LUCHG,'(A)',END=40) LINE
  IF(LINE(1:1).EQ.'*') GOTO 10
  IF(LINE(1:1).EQ.'$') GOTO 40
  READ(LINE,20,ERR=30) MCLASS,I,J,BCIIJ
20 FORMAT(I1,I4,I5,F10.5)
  IF(NCQ.GE.MAXCB) THEN
     IF (WRNLEV.GE.2) WRITE(OUTU,'(A,I6,A/A)') &
          ' ***** Too many bond charge increments read '// &
          '... max = ',MAXCB,' *****',' *** Execution terminated ****'
     CALL WRNDIE(-5,'<rdchgm>','Too many bond charge increment'// &
          ' parameters')
  ENDIF
  NCQ=NCQ+1
  !      write(outu,*) ' ncq =',ncq
  !      write(outu,*) line(:50)
  CALL BONDCON('PUT',I,J,MCLASS,IT12)
  !
  !   IS CANONICAL ORDER BEING OBEYED?
  !
  IF(IT12.LE.ITLAST) THEN
     if(wrnlev.ge.2) WRITE(OUTU,'(A/A/A)') &
          ' *** Canonical order violated: misordered line follows: ***', &
          LINE(1:78),' *** Execution terminated ****'
     CALL WRNDIE(-5,'<rdchgm>','Canonical order violated')
  ELSE
     ITLAST=IT12
  ENDIF
  !
  !   STORE THIS BOND-CHARGE INCREMENT
  !
  BCI(NCQ)=BCIIJ
  ICHGNDX(NCQ)=IT12
  GOTO 10
30 CONTINUE
  !     IF(IZ.NE.0) WRITE(IZ,*) ' ERROR ON READING MMFFCHG FILE. STOPPING'
  CALL WRNDIE(-5,'<rdchgm>','ERROR ON READING MMFFCHG FILE')
40 CONTINUE
  !     CALL VCLOSE(LUCHG,'KEEP',ERROR)
  !     LRB=LPATH(CHGFLE)
  !     LEND=LLEN(CHGFLE)
  if(prnlev.ge.2) WRITE(OUTU,'(a,i5,a,i5)') 'rdchgm> ',NCQ, &
       ' BOND-INCREMENT CHARGES READ FROM UNIT ',LUCHG
  NCQT=NCQ
  !   READ SUPPLEMENTARY CHARGE-INCREMENT PARAMETERS, IF NQ IS NONZERO
  IF(NQ.EQ.0) RETURN
  if(prnlev.ge.2) WRITE(OUTU,90) NQ
90 FORMAT(I5,' SUPPLEMENTARY CHARGE-INCREMENT PARAMETERS READ')
  DO K=1,NQ
     READ(ISUPP,95) IIA,JA,BCIIJ
95   FORMAT(2I5,F10.5)
     IF(IIA.GT.JA) BCIIJ=-BCIIJ
     MCLASS=0
     CALL BONDCON('PUT',IIA,JA,MCLASS,IT12)
     !   SEE IF ALREADY IN DATA BASE
     CALL BIFIND(IT12,ICHGNDX,NCQ,N,NCQT,MAXCB,NEWPAR,FULL,1)
     IF(.NOT.FULL) THEN
        IF(NEWPAR) ICHGNDX(N)=IT12
        BCI(N)=BCIIJ
     ELSE
        if(wrnlev.ge.2) WRITE(OUTU,110) K
        !     IF(IZ.NE.0) WRITE(IZ,110) K
110     FORMAT(/' *** THE ARRAY OF CHARGE INCREMENTS IS FULL ...', &
             ' PARAMETER CARD ',I3,' IS SKIPPED')
     ENDIF
  enddo
  !
  RETURN
END SUBROUTINE RDCHGM

! ===============================================================
! SUBROUTINE RDDEFI : READ IN MMFF ATOM TYPE DEFINITIONS AND ALIASES
! ===============================================================
SUBROUTINE RDDEFI(LUMMFF,ERROR)
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   Adapted by R. Czerminski of Molecular Simulations, Inc. from
  !  code developed at Merck and Co., Inc. by T. Halgren and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !
  !  Jay Banks 27-Nov-95: changed IF (PRNLEV.GE.2) to IF (WRNLEV.GE.2) for
  !  WRITE statement (formerly) followed by CALL WRNDIE.
  !
  !  28 Nov 95 Jay Banks: removed space from all "GO TO"s, as suggested in
  !  developer.doc.
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use mmffm, only: maxdefi, mdef
  use rtf, only: ctype, intdef
  use stream

  use allocation, only: chmalloc
  use deallocation, only: chmdealloc

  implicit none

  integer, intent(in) :: lummff

  integer j, nmmff

  ! an array to hold values read in for mdef
  ! after reading all the values,
  ! mdef will be allocated and then the values
  ! for mdef_tmp will be copied over
  integer, dimension(5, maxdefi) :: mdef_tmp

  CHARACTER(len=4) SymbN ! scratch space for symbolic name
  CHARACTER(len=80) LINE
  logical, intent(out) :: error ! unused, should be removed
  
  error=.false.
  NMMFF=0
  do
    READ(LUMMFF,'(A)') LINE

    IF(LINE(1:1).EQ.'$') exit
    IF(LINE(1:1).EQ.'*') cycle

    if(NMMFF.ge.MAXDEFI) then
      call wrndie(-1,'<rdequv>',' too many definitions')
      exit
    endif

    NMMFF=NMMFF+1
    READ(LINE,'(3X,A4,5I5)') SymbN, (mdef_tmp(J,NMMFF),J=1,5)

    !... test if symbolic atom type corresponds to numeric atom type...
    j=1
    do while (SymbN.ne.CTYPE(j)(1:4) .and. j.lt.MAXATC)
      j=j+1
    end do

    if(intdef(j) .ne. mdef_tmp(1,NMMFF) .and. wrnlev .ge. 2) &
      write(outu,'(3(1x,a),2I5)') &
        'rdequv> symbn,ctype,j,MDEF(1,NMMFF)=', &
        symbn,ctype(j),j, mdef_tmp(1,NMMFF)
  end do

  if(prnlev.ge.2) WRITE(OUTU,'(a,i5,a,i5)') 'rdequv> ',NMMFF, &
       ' ATOM-TYPE DEFINITIONS READ FROM UNIT ',LUMMFF

  ! allocate mdef to be exactly 5 x nmmff
  if(allocated(mdef)) call chmdealloc('readpar.src', 'RDDEFI', &
    'mdef', 5, nmmff, intg = mdef)
  call chmalloc('readpar.src', 'RDDEFI', 'mdef', 5, nmmff, intg = mdef)

  ! copy read data into mmffm::mdef
  mdef = mdef_tmp(1:5, 1:nmmff)
  RETURN
END SUBROUTINE RDDEFI

! ======================================================================
! SUBROUTINE RDDFSB : READ DEFAULT STRETCH-BEND PARAMETERS
! ======================================================================
!
SUBROUTINE RDDFSB(LUSTBN,ERROR)
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
  use consta
  use param
  use stream
  implicit none
  !
  integer i, j
  integer lustbn
  integer k
  real(chm_real)  f1, f2
  logical error
  CHARACTER(len=80) LINE
  !
  error=.false.
  !
  !   NOW READ DEFAULT STRETCH-BEND PARAMETERS
  !   FIRST GET AND OPEN THE FILE
  !      CALL PFILES('DFSBPAR',STBNF,LSTBN,LUSTBN,OKPAR)
  !      IF(.NOT.OKPAR) RETURN
  !
  ! PREZERO THE FORCE-CONSTANT ARRAY
  !
  DO I=0,4
     DO J=1,4
        DO K=0,4
           DEFSBK(I,J,K)=0.
        ENDDO
     ENDDO
  ENDDO
  !
  !   READ THE FILE AND INSERT THE ENTRIES
  !
  NCDFSB=0
110 CONTINUE
  READ(LUSTBN,'(A)') LINE
  IF(LINE(1:1).EQ.'$') GOTO 140
  IF(LINE(1:1).EQ.'*') GOTO 110
  NCDFSB=NCDFSB+1
  READ(LINE,120) I,J,K,F1,F2
120 FORMAT(3I5,2F10.3)
  IF(I.LT.K) THEN
     DEFSBK(I,J,K)=F1
     DEFSBK(K,J,I)=F2
  ELSE
     DEFSBK(I,J,K)=F2
     DEFSBK(K,J,I)=F1
  ENDIF
  GOTO 110
140 CONTINUE
  !      LRB=LPATH(STBNF)
  !      LEND=LLEN(STBNF)
  !      WRITE(IPF,150) NCDFSB,STBNF(LRB+1:LEND)
  IF(PRNLEV.gt.2) WRITE(OUTU,150) NCDFSB,lustbn
150 FORMAT(I5,' DEFAULT STRETCH-BEND PARAMETERS READ FROM UNIT ',i5)
  CALL VCLOSE(LUSTBN,'KEEP',ERROR)
  !
  RETURN
END SUBROUTINE RDDFSB

! ===========================================================
! SUBROUTINE RDHDEF : READ IN HYDROGEN ATOM SYMBOLIC TYPES
! AS A FUNCTION OF THE SYMBOLIC TYPES OF THE PARENT ATOM TO
! WHICH EACH SUCH HYDROGEN IS ATTACHED
! ===========================================================
!
SUBROUTINE RDHDEF(LUTRHYD,ERROR)
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   Adapted by R. Czerminski of Molecular Simulations, Inc. from
  !  code developed at Merck and Co., Inc. by T. Halgren and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !
  ! Jay Banks 08 Nov 95: changed  variable PTYPE (mmff.f90: parent
  ! atom type) to PATYPE to avoid conflict with CHARMM variable.
  !
  ! Jay Banks 22 Nov 95: added IF (PRNLEV.GE.2) to WRITE statement.
  !
  !  Jay Banks 27-Nov-95: changed IF (PRNLEV.GE.2) to IF (WRNLEV.GE.2) for
  !  WRITE statement followed by CALL WRNDIE.
  !
  !  28 Nov 95 Jay Banks: removed space from all "GO TO"s, as suggested in
  !  developer.doc.
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use param
  use stream
  use mmffm
  implicit none
  !
  integer lutrhyd
  logical error
  CHARACTER(len=80) LINE
  !
  !   FIRST GET AND OPEN THE FILE
  !
  !     CALL PFILES('TRHYD',TRHYDF,LTRHYD,LUTRHYD,OKPAR)
  !     IF(.NOT.OKPAR) RETURN
  error=.false.
  NTRHYD=0
10 CONTINUE
  READ(LUTRHYD,'(A)') LINE
  IF(LINE(1:1).EQ.'$') GOTO 40
  IF(LINE(1:1).EQ.'*') GOTO 10
  IF(NTRHYD.GE.MAXATC) THEN
     IF (WRNLEV.GE.2) WRITE(OUTU,'(A,I6,A/A)') &
          ' ***** Too parameters read from MMFFHDEF '// &
          '... max = ',MAXATC,' *****',' *** Execution terminated ****'
     CALL WRNDIE(-5,'<rdhdef>','Too parameters read from'// &
          ' MMFFHDEF')
  ENDIF
  NTRHYD=NTRHYD+1
  READ(LINE,'(2X,A,4X,A)',ERR=250) PATYPE(NTRHYD),HDTYPE(NTRHYD)
  GOTO 10
40 CONTINUE
  !     LRB=LPATH(TRHYDF)
  !     LEND=LLEN(TRHYDF)
  if(prnlev.ge.2) WRITE(OUTU,'(a,i5,a,i5)') 'RDHDEF> ',NTRHYD, &
       ' MMFF HYDROGEN SYMBOLIC TYPE DEFINITIONS READ FROM UNIT ',LUTRHYD
  !     CALL VCLOSE(LUTRHYD,'KEEP',ERROR)
  RETURN
250 CONTINUE
  CALL WRNDIE(-5,'<RDHDEF>','error reading mmffhdef.par')
  !
  RETURN
END SUBROUTINE RDHDEF

! ====================================================================
! SUBROUTINE RDOOPM : READ OUT-OF-PLANE BENDING PARAMETERS
! ====================================================================
!
SUBROUTINE RDOOPM(NO,LUOOP,ISUPP,ERROR)
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   Adapted by R. Czerminski of Molecular Simulations, Inc. from
  !  code developed at Merck and Co., Inc. by T. Halgren and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !  16 Oct 95 Jay Banks: changed variable names IT, JT, KT to
  !       ITH, JTH, KTH to avoid conflict with CHARMM names
  !
  !  22 Nov 95 Jay Banks: added IF (PRNLEV.GE.2) to a WRITE statement.
  !
  !  Jay Banks 27-Nov-95: changed IF (PRNLEV.GE.2) to IF (WRNLEV.GE.2) for
  !  WRITE statements followed by CALL WRNDIE.
  !
  !  28 Nov 95 Jay Banks: removed space from all "GO TO"s, as suggested in
  !  developer.doc.
  !
  !  28 Nov 95 Jay Banks: changed PRNLEV test to WRNLEV for printing error
  !  conditions.
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use mmffm
  use consta
  use param
  use stream
  implicit none
  !
  integer i, isupp, ith, j, jth, k
  integer koop
  integer kth, KOLAST, l, lt
  integer luoop, no
  real(chm_real)  fc
  integer moop(2)
  real(chm_real) oopk(2)
  integer m
  logical mmff94s
  !
  CHARACTER(len=80) LINE
  CHARACTER(len=36) SOURCE
  LOGICAL ERROR,NEWPAR,FULL
  !     integer llen
  !     external llen
  !
  !   set replacement parameters for mmff94s
  !
  mmff94s=.false.
  !      mmff94s=.true.
  MOOP(1)=10
  MOOP(2)=40
  OOPK(1)=0.015
  OOPK(2)=0.03
  !
  !   FIRST GET AND OPEN THE FILE
  !
  !     CALL PFILES('OOPPAR',OOPF,LOOP,LUOOP,OKPAR)
  !     IF(.NOT.OKPAR) RETURN
  error=.false.
  !  NOW READ IN THE PARAMETERS
  NCOOP=0
  KOLAST=-1
210 CONTINUE
  READ(LUOOP,'(A)') LINE
  IF(LINE(1:1).EQ.'$') GOTO 240
  IF(LINE(1:1).EQ.'*') GOTO 210
  READ(LINE,220) I,J,K,L,FC,SOURCE
220 FORMAT(4I5,F10.3,A)
  CALL OOPCON('PUT',I,J,K,L,KOOP)
  !
  !   IS CANONICAL ORDER BEING OBEYED?
  !
  IF(KOOP.LE.KOLAST) THEN
     if(wrnlev.ge.2) WRITE(OUTU,'(A/A/A)') &
          ' *** Canonical order violated: misordered line follows: ***', &
          LINE(1:78),' *** Execution terminated ****'
     CALL WRNDIE(-5,'<rdoopm>','Canonical order violated')
  ELSE
     KOLAST=KOOP
  ENDIF
  IF(NCOOP.GE.MAXCT) THEN
     IF (WRNLEV.GE.2) WRITE(OUTU,'(A,I6,A/A)') &
          ' ***** Too many out-of-plane bending parameters read in '// &
          '... max = ',MAXCT,' *****',' *** Execution terminated ****'
     CALL WRNDIE(-5,'<rdoopm>','Too many out-of-plane bending'// &
          ' parameters')
  ENDIF
  IF(MMFF94S) THEN
     DO M=1,2
        IF(J.EQ.MOOP(M)) FC=OOPK(M)
     ENDDO
  ENDIF
  NCOOP=NCOOP+1
  KCOOP(NCOOP)=KOOP
  OoplFC(NCOOP)=FC*MDAKCAL
  !C      O_SOURCE(NCOOP)=SOURCE
  GOTO 210
240 CONTINUE
  NCOOPT=NCOOP
  !     LRB=LPATH(OOPF)
  !     LEND=LLEN(OOPF)
  if(prnlev.ge.2) WRITE(OUTU,'(a,i5,a,i5)') 'rdoopm> ',NCOOP, &
       ' OUT-OF-PLANE PARAMETERS READ FROM UNIT ',LUOOP
  !
  !   READ SUPPLEMENTAL OUT-OF-PLANE CONSTANTS, IF ANY
  !
  IF(NO /= 0) then ! GOTO 5000
     if(prnlev.ge.2) WRITE(OUTU,'(a,i5,a)') 'rdoopm> ',NO, &
          ' SUPPLEMENTARY OUT-OF-PLANE PARAMETERS READ'
     loop4950: DO I=1,NO
        READ(isupp,220) ITH,JTH,KTH,LT,FC,SOURCE
        !  220 FORMAT(4I5,F10.3,A)
        CALL OOPCON('PUT',ITH,JTH,KTH,LT,KOOP)
        CALL BIFIND(KOOP,KCOOP,NCOOP,J,NCOOPT,MAXCT,NEWPAR,FULL,1)
        IF(.NOT.FULL) THEN
           IF(NEWPAR) KCOOP(J)=KOOP
           if(newpar) NCOOPT=NCOOPT+1
           OoplFC(J)=FC*MDAKCAL
           !C         O_SOURCE(J)=SOURCE
        ELSE
           !  IF THIS POINT IS REACHED, THE KCOOP ARRAY IS FILLED
           if(wrnlev.ge.2) WRITE(OUTU,4800) I
           !     IF(IZ.NE.0) WRITE(IZ,4800) I
4800       FORMAT(/'* * * * * ERROR - THE ARRAY OF OUT-OF-PLANE ANGLE ', &
                'TYPES IS FILLED.  ANGLE CARD',I3,' IS SKIPPED')
           cycle loop4950
        ENDIF
     enddo loop4950
  endif
  !
  RETURN
END SUBROUTINE RDOOPM

! ===========================================================
! SUBROUTINE rdpbci : READ MMFF BOND-INCREMENT CHARGE PARAMETERS
! ===========================================================
!
SUBROUTINE rdpbci(LUCHG,ERROR)
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
  !#INCLUDE '~/charmm_fcm/exfunc.f90'
  use param
  use stream
  use mmffm
  implicit none
  !
  integer i
  integer luchg, npbci
  real(chm_real)  fqa, pb
  !
  CHARACTER(len=80) LINE
  LOGICAL ERROR
  !
  error=.false.
  !
  !   INITIALIZE PARTIAL-BOND-CHARGE-INCREMENT CHARGES AND LOGICALS
  !   TO .FALSE.
  !
  !      write(outu,*) ' entering rdpbci'
  DO I=1,MAXDEFI
     PBCI(I)=0.
     FCADJ(I)=0.
     LPBCI(I)=.FALSE.
  ENDDO
  !
  !   NOW OPEN AND READ THE PARTIAL-BOND-CHARGE INCREMENTS FILE (USED IN
  !   ASSIGNING DEFAULT CHARGES
  !
  !     CALL PFILES('PBCIPAR',CHGFLE,LCHG,LUCHG,OKPAR)
  !     IF(.NOT.OKPAR) RETURN
  NPBCI=0
210 CONTINUE
  READ(LUCHG,'(A)',END=240) LINE
  IF(LINE(1:1).EQ.'*') GOTO 210
  IF(LINE(1:1).EQ.'$') GOTO 240
  NPBCI=NPBCI+1
  READ(LINE,220,ERR=230) I,PB,FQA
  !0    1    0.0000    0.0000   Derived pbci
220 FORMAT(I5,2F10.5)
  !
  !   STORE THIS PARTIAL-BOND-CHARGE INCREMENT AND FORMAL-CHARGE ADJUSTMENT
  !   FACTOR
  !
  PBCI(I)=PB
  FQA=ABS(FQA)
  IF(FQA.LT.0.) THEN
     CALL WRNDIE(-5,'<rdpbci>','Negative FCAJD value found in file')
  ENDIF
  FCADJ(I)=FQA
  LPBCI(I)=.TRUE.
  GOTO 210
230 CONTINUE
  CALL WRNDIE(-5,'<rdpbci>','ERROR ON READING MMFFPBCI FILE')
240 CONTINUE
  !     CALL VCLOSE(LUCHG,'KEEP',ERROR)
  !     LRB=LPATH(CHGFLE)
  !     LEND=LLEN(CHGFLE)
  if(prnlev.ge.2) WRITE(OUTU,'(a,i5,a,i5)') 'rdpbci> ',NPBCI, &
       ' PARTIAL-BOND-INCREMENT CHARGES READ FROM UNIT ',LUCHG
  !
  RETURN
END SUBROUTINE rdpbci

! ===========================================================
! SUBROUTINE RDPROPM : READ MMFF ATOM-TYPE PROPERTIES
! ===========================================================
!
SUBROUTINE RDPROPM(LUATPRP,ERROR)
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   Adapted by R. Czerminski of Molecular Simulations, Inc. from
  !  code developed at Merck and Co., Inc. by T. Halgren and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !
  !  Jay Banks 22 Nov 95: added IF (PRNLEV.GE.2) to a WRITE statement.
  !
  !  Jay Banks 27-Nov-95: changed IF (PRNLEV.GE.2) to IF (WRNLEV.GE.2) for
  !  WRITE statement followed by CALL WRNDIE.
  !
  !  28 Nov 95 Jay Banks: removed space from all "GO TO"s, as suggested in
  !  developer.doc.
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use mmffm
  use rtf, only: atnumt,reallocate_natct
  use param
  use stream
  implicit none
  !
  integer luatprp, natprop, m, atnumtm
  logical error
  !
  CHARACTER(len=80) LINE

  !   FIRST GET AND OPEN THE FILE
  !     CALL PFILES('ATPROP',ATPROPF,LATPROP,LUATPRP,OKPAR)
  !     IF(.NOT.OKPAR) RETURN
  error=.false.
  NATPROP=0
10 CONTINUE
  READ(LUATPRP,'(A)') LINE
  IF(LINE(1:1).EQ.'$') GOTO 40
  IF(LINE(1:1).EQ.'*') GOTO 10
  READ(LINE,'(9I5)',ERR=250) M,AtNumTM,MCOORD(M),MVALMIN(M), &
       MPILP(M),MLTBND(M),MAROM(M),MLINBA(M),MSP2SGL(M)
  call reallocate_natct(m)
  atnumt(m)=atnumtm
  IF(M.GT.MAXDEFI) THEN
     IF (WRNLEV.GE.2) WRITE(OUTU,'(A,I6,A/A)') &
          ' ***** MMFF numerical atom type too large '// &
          '... max = ',MAXDEFI,' *****',' *** Execution terminated ***'
     CALL WRNDIE(-5,'<rdpropm>','Too large a MMFF numerical '// &
          'atom type')
  ENDIF
  NATPROP=NATPROP+1
  MVALMAX(M)=MVALMIN(M)
  IF(MVALMIN(M).GT.10) THEN
     MVALMIN(M)=MVALMIN(M)/10
     MVALMAX(M)=MVALMAX(M)-10*MVALMIN(M)
  ENDIF
  GOTO 10
40 CONTINUE
  if(prnlev.ge.2) WRITE(OUTU,'(a,i5,a,i5)') 'rdpropm> ',NATPROP, &
       ' MMFF ATOM-TYPE PROPERTY RECORDS READ FROM UNIT ',LUATPRP
  RETURN
250 CONTINUE
  CALL WRNDIE(-5,'<rdpropm>','error reading mmffprop.par')
  !
  RETURN
END SUBROUTINE RDPROPM

! ======================================================================
! SUBROUTINE RDSTBNM : READ REGULAR STRETCH-BEND PARAMETERS
! ======================================================================
!
SUBROUTINE RDSTBNM(LUSTBN,ERROR)
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   Adapted by R. Czerminski of Molecular Simulations, Inc. from
  !  code developed at Merck and Co., Inc. by T. Halgren and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !
  !  Jay Banks 22 Nov 95: added IF (PRNLEV.GE.2) to a WRITE statement.
  !
  !  Jay Banks 27-Nov-95: changed IF (PRNLEV.GE.2) to IF (WRNLEV.GE.2) for
  !  WRITE statements followed by CALL WRNDIE.
  !
  !  28 Nov 95 Jay Banks: removed space from all "GO TO"s, as suggested in
  !  developer.doc.
  !
  use chm_kinds
  !...##INCLUDE '~/charmm_fcm/debug.f90'
  use dimens_fcm
  use exfunc
  use mmffm
  use consta
  use param
  use stream
  use string
  implicit none
  !
  integer i, j
  integer lustbn
  integer k, ksb, KSBLAST, mcij, mcijk, mcjk, mclass
  real(chm_real)  f1, f2
  logical error
  !
  CHARACTER(len=80) LINE
  CHARACTER(len=36) SOURCE
  !
  !...##IF DEBUG
  !      call getenv('DEBUG_RDSTBNM',SCRTCH)
  !      DEBUG=SCRTCH.ne.' '
  !...##ENDIF
  !
  error=.false.
  NCSB=0
  KSBLAST=-1
10 CONTINUE
  READ(LUSTBN,'(A)') LINE
  IF(LINE(1:1).EQ.'$') GOTO 40
  IF(LINE(1:1).EQ.'*') GOTO 10
  READ(LINE,20) MCLASS,I,J,K,F1,F2,SOURCE
20 FORMAT(I1,I4,2I5,2F10.3,A)
  !
  !   FORM THE PACKED, CANONICALLY ORDERED INDEX
  !
  !   PROVIDE "-MCLASS" TO STBNCON TO TELL IT TO COMPUTE MCIJ, MCJK
  !   AND MCIJK FROM |MCLASS|, RATHER THAN VICE-VERSA, IN THE EVENT
  !   THAT MCLASS IS NON-ZERO
  !
  MCIJ=0
  MCJK=0
  MCIJK=0
  !...##IF DEBUG
  !      if(DEBUG) write(OUTU,'(a,3i4,4i3,i10)') 'DEBUG_RDSTBNM> b'//
  !     & ' I,J,K,MCLASS,MCIJ,MCJK,MCIJK,KSB =',
  !     &   I,J,K,MCLASS,MCIJ,MCJK,MCIJK,KSB
  !...##ENDIF
  CALL STBNCON('PUT',I,J,K,-MCLASS,MCIJ,MCJK,MCIJK,KSB)
  !...##IF DEBUG
  !      if(DEBUG) write(OUTU,'(a,3i4,4i3,i10)') 'DEBUG_RDSTBNM> a'//
  !     & ' I,J,K,MCLASS,MCIJ,MCJK,MCIJK,KSB =',
  !     &   I,J,K,MCLASS,MCIJ,MCJK,MCIJK,KSB
  !...##ENDIF
  !      SUBROUTINE STBNCON(PUTGET,I,J,K,MCLASS,MCIJ,MCJK,MCIJK,IJKM)
  !
  !   IS CANONICAL ORDER BEING OBEYED?
  !
  IF(KSB.LE.KSBLAST) THEN
     if(wrnlev.ge.2) WRITE(OUTU,'(A/A/A)') &
          ' *** Canonical order violated: misordered line follows: ***', &
          LINE(1:78),' *** Execution terminated ****'
     CALL WRNDIE(-5,'<rdomegm>','Canonical order violated')
  ELSE
     KSBLAST=KSB
  ENDIF
  IF(NCSB.GE.MAXCT) THEN
     IF (WRNLEV.GE.2) WRITE(OUTU,'(A,I6,A/A)') &
          ' ***** Too many stretch-bend parameters read '// &
          '... max = ',MAXCT,' *****',' ***** Execution terminated' &
          //' *****'
     CALL WRNDIE(-5,'<rdstbnm>','Too many stretch-bend'// &
          ' parameters')
  ENDIF
  NCSB=NCSB+1
  KCSTBN(NCSB)=KSB
  STBNP(1,NCSB)=F1*MDAKCAL
  STBNP(2,NCSB)=F2*MDAKCAL
  BA_SOURCE(NCSB)=SOURCE
  GOTO 10
40 CONTINUE
  NCSBT=NCSB
  if(prnlev.ge.2) WRITE(OUTU,'(a,i5,a,i5)') 'rdstbnm> ',NCSB, &
       ' STRETCH-BEND PARAMETERS READ FROM UNIT ',LUSTBN
  !
  RETURN
END SUBROUTINE RDSTBNM

! ===============================================================
! SUBROUTINE RDSYMB : READ IN MMFF SYMBOLIC TO NUMERIC ATOM TYPE TABLE
! ===============================================================
SUBROUTINE RDSYMB(LUSYMT,ERROR)
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   Adapted by R. Czerminski of Molecular Simulations, Inc. from
  !  code developed at Merck and Co., Inc. by T. Halgren and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !
  !  Jay Banks 22 Nov 95: Added IF (PRNLEV.GE.2) to two WRITE statements.
  !  Changed (OUTU,*) to explicit format in another WRITE statement.
  !
  !  Jay Banks 27 Nov 95: Changed test of PRNLEV to WRNLEV when WRITE is
  !  followed by CALL WRNDIE.  CALL WRNDIE is actually commented out there
  !  (mass not assigned), so added its former message to what's printed
  !  under WRNLEV.
  !
  !  28 Nov 95 Jay Banks: removed space from all "GO TO"s, as suggested in
  !  developer.doc.
  !
  !  28 Nov 95 Jay Banks: changed PRNLEV test to WRNLEV for printing error
  !  conditions.
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use param
  use rtf, only: ctype, intdef, armass, natct, atnumt, reallocate_natct
  use stream
  use mmffm
  implicit none
  !
  integer i, j, lusymt, nsymb
  logical error
  real(chm_real) mass
  !
  CHARACTER(len=80) LINE
  !
  error=.false.
  NSYMB=0
10 CONTINUE
  READ(LUSYMT,'(A)') LINE
  IF(LINE(1:1).EQ.'$') GOTO 40
  IF(LINE(1:1).EQ.'*') GOTO 10
  if(NSYMB.ge.MAXATC) then
     IF (WRNLEV.GE.2) WRITE(OUTU,'(A,I6,A/A)') &
          ' ***** Too many symbolic MMFF atom types read '// &
          '... max = ',MAXATC,' *****',' *** Execution terminated ****'
     call wrndie(-5,'<RDSYMB>',' too many atom type symbols')
  else
     NSYMB=NSYMB+1
     call reallocate_natct(nsymb)
     READ(LINE,'(2X,A4,I5)') CTYPE(NSYMB),INTDEF(NSYMB)
     if(INTDEF(NSYMB).gt.0 .and. INTDEF(NSYMB).le.MAXDEFI) then
        ATC(INTDEF(NSYMB))=CTYPE(NSYMB)
        NATCT=MAX(NATCT,INTDEF(NSYMB))
     else
        IF (WRNLEV.GE.2) WRITE(OUTU,'(A,I6,A/A)') &
             ' ***** Too large a numerical MMFF atom type '// &
             '... max = ',MAXDEFI,' *****',' *** Execution terminated ****'
        call wrndie(-5,'<RDSYMB>','Numeric atom type out of range')
     endif
     !
     !   USE ENTRIES IN  BLOCK /mmffprop/ TO ASSIGN ATOMIC NUMBERS
     !   AND MASSES for the numeric atom types
     !
     !       AtNumT(INTDEF(NSYMB))=MSPEC(INTDEF(NSYMB))
     IF(abs(AtNumT(INTDEF(NSYMB))) == 0 .OR. &
          abs(AtNumT(INTDEF(NSYMB))).GT.MaxAtN) then
        if(wrnlev.ge.2) then
           write(outu,'(a)') line
           write(outu,'(A,3I6)') &
                ' nsymb,intdef(nsymb),AtNumT(intdef(nsymb))=', &
                nsymb, intdef(nsymb), AtNumT(intdef(nsymb))
        endif
        call wrndie(-3,'<RDSYMB>','invalid atomic number')
        mass=0.
     else
        mass=ElementMass(abs(AtNumT(INTDEF(NSYMB))))
     endif
     if(mass.le.0.) then
        if(wrnlev.ge.2) then
           write(outu,'(a)')  &
                'RDSYMB: mass not assigned for the following:'
           write(outu,'(a)') line
        endif
        !          call wrndie(-3,'<RDSYMB>','mass not assigned')
     endif
     !       AtNumT(NSYMB)=anum ! this will destroy information read from RTF
     ARMASS(INTDEF(NSYMB))=mass
     !---
     GOTO 10
  endif
40 CONTINUE
  !     LRB=LPATH(SYMTRF)
  !     LEND=LLEN(SYMTRF)
  if(prnlev.ge.2) WRITE(OUTU,'(a,i5,a,i5)') 'RDSYMB> ', NSYMB, &
       ' SYMBOLIC-NUMERIC ATOM-TYPE DEFINITIONS READ FROM UNIT ',LUSYMT
  !
  NATC=NATCT
  !
  ! check for repetitions...
  !
  if(wrnlev.ge.2) then
     do i=1,NSYMB-1
        do j=i+1,NSYMB
           if(CTYPE(i).eq.CTYPE(j)) write(outu,'(a,a)') &
                ' type repetition: ',CTYPE(i)
        enddo
     enddo
  endif
  !
  ! print masses if prnlev high enough
  !
  if(prnlev.gt.6) then
     write(outu,'(a)') &
          ' CTYPE,INTDEF,ATC' ! symbolic type, num type, symb type
     do i=1,NSYMB
        write(outu,'(1x,a,i4,1x,a)')   & ! ,1x,i5,1x,a,f12.6)')
             CTYPE(I),INTDEF(I),ATC(INTDEF(I))   ! ,
        !    & AtNumT(I),ElementName(AtNumT(I)),ElementMass(AtNumT(I))
        !ARMASS(INTDEF(I))
     enddo
  endif
  !     CALL VCLOSE(LUSYMT,'KEEP',ERROR)
  !
  RETURN
END SUBROUTINE RDSYMB

! ======================================================================
! SUBROUTINE RDTHETM : READ REGULAR ANGLE-BENDING PARAMETERS
! ======================================================================
!
SUBROUTINE RDTHETM(NB,LUANG,ISUPP,ERROR)
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   Adapted by R. Czerminski of Molecular Simulations, Inc. from
  !  code developed at Merck and Co., Inc. by T. Halgren and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !  16 Oct 95 Jay Banks: changed variable names "IB" and "IT" to "IBmmff"
  !                       and "ITmmff" to avoid conflict with CHARMM names
  !
  !  Jay Banks 25 Oct 95: changed variable IA to IAmmff, to avoid conflict
  !  with variable in module (mmff.f90).
  !
  !  Jay Banks 22 Nov 95: added IF (PRNLEV.GE.2) to a WRITE statement.
  !
  !  Jay Banks 27-Nov-95: changed IF (PRNLEV.GE.2) to IF (WRNLEV.GE.2) for
  !  WRITE statements followed by CALL WRNDIE.
  !
  !  28 Nov 95 Jay Banks: removed space from all "GO TO"s, as suggested in
  !  developer.doc.
  !
  !  28 Nov 95 Jay Banks: changed PRNLEV test to WRNLEV for printing error
  !  conditions.
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use mmffm
  use consta
  use param
  use stream
  implicit none
  !
  integer i, ISUPP, j
  integer luang, nb
  integer k, ka, KALAST, mclass
  integer iammff, ibmmff, ic, itmmff, l
  logical error
  real(chm_real)  a1, fc
  !
  CHARACTER(len=80) LINE
  CHARACTER(len=36) SOURCE
  LOGICAL NEWPAR,FULL
  !
  error=.false.
  NCT=0
  KALAST=-1
10 CONTINUE
  READ(LUANG,'(A)') LINE
  IF(LINE(1:1).EQ.'$') GOTO 40
  IF(LINE(1:1).EQ.'*') GOTO 10
  READ(LINE,20) MCLASS,I,J,K,FC,A1,SOURCE
20 FORMAT(I1,I4,2I5,2F10.3,3X,A)
  !
  !   FORM THE PACKED, CANONICALLY ORDERED INDEX
  !
  CALL THETCON('PUT',I,J,K,MCLASS,KA)
  !
  !   IS CANONICAL ORDER BEING OBEYED?
  !
  IF(KA.LE.KALAST) THEN
     if(wrnlev.ge.2) WRITE(OUTU,'(A/A/A)') &
          ' *** Canonical order violated: misordered line follows: ***', &
          LINE(1:78),' *** Execution terminated ****'
     CALL WRNDIE(-5,'<rdthetm>','Canonical order violated')
  ELSE
     KALAST=KA
  ENDIF
  IF(NCT.GT.MAXCT) THEN
     IF (WRNLEV.GE.2) WRITE(OUTU,'(A,I6,A/A)') &
          ' ***** Too many angle-bending parameters read in ... max = ', &
          MAXCT,' *****',' *** Execution terminated ****'
     CALL WRNDIE(-5,'<rdthetm>','Too many angle-bending parameters')
  ENDIF
  NCT=NCT+1
  KCT(NCT)=KA
  AnglFC(NCT)=FC*MDAKCAL
  AnglEq(NCT)=A1*degrad
  GOTO 10
40 CONTINUE
  if(prnlev.ge.2) WRITE(OUTU,'(A,I5,A,I5)') &
       'rdthetm>',NCT,' ANGLE-BENDING PARAMETERS READ FROM UNIT ',LUANG
  !  50 FORMAT(I5,' ANGLE-BENDING PARAMETERS READ FROM ',A)
  !  READ IN SUPPLEMENTARY BENDING PARAMETERS IF NB.NE.0
  NCTT=NCT
  IF (NB.EQ.0) return
  if(prnlev.ge.2) WRITE(OUTU,'(A,I5,A)') &
       'rdthetm>',NB,' SUPPLEMENTARY BENDING PARAMETERS READ'
  call ffind_loc(ISUPP,'ANGLE',ERROR)
  if(ERROR) then
     call wrndie(-1,'<RDTHETM>','EOF in supplementary file')
     return
  endif
  DO I=1,NB
     READ(ISUPP,2400) MCLASS,IAMMFF,IBmmff,IC,FC,A1,SOURCE(1:33)
2400 FORMAT(I1,I4,2I5,2F10.3,3X,A)
     !  FOR REGULAR BENDING CONSTANTS, SCAN THE KCT LIST TO LOCATE ANGLE.
     !  IF IT'S NOT THERE, ADD IT.  IF A 'J' IS NOT READ IN, A VALUE OF ONE
     !  IS ASSUMED AND ANGLES FOR J=2,3 ARE ZEROED OUT (EV. DEP. IS SURPRES).
     !
     !   FORM THE PACKED INDEX
     !
     CALL THETCON('PUT',IAMMFF,IBmmff,IC,MCLASS,ITmmff)
     CALL BIFIND(ITmmff,KCT,NCT,L,NCTT,MAXCT,NEWPAR,FULL,1)
     IF(.NOT.FULL) THEN
        IF(NEWPAR) KCT(L)=ITmmff
        AnglFC(L)=FC*MDAKCAL
        AnglEq(L)=A1*degrad
     ELSE
        !  IF THIS POINT IS REACHED THE KCT ARRAY IS FILLED
        if(wrnlev.ge.2) WRITE(OUTU,4300) I
        !     IF(IZ.NE.0) WRITE(IZ,4300) I
4300    FORMAT(/'* * * * * ERROR - THE ARRAY OF ANGLE TYPES IS ', &
             'FILLED.  ANGLE CARD',I3,' IS SKIPPED')
        !       INIT=1
     ENDIF
  enddo
  !
  RETURN
END SUBROUTINE RDTHETM

!======================================================================
! SUBROUTINE RDTORS : READS MAIN AND SUPPLEMENTARY TORSIONAL CONSTANTS
! FOR THE MMFF FORCE FIELD
!======================================================================
!
SUBROUTINE RDTORS(NT,LUTOR,ISUPP,ERROR)
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   Adapted by R. Czerminski of Molecular Simulations, Inc. from
  !  code developed at Merck and Co., Inc. by T. Halgren and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !  16 Oct 95 Jay Banks: changed variable name KT to KTOR
  !                       to avoid conflict with CHARMM name
  !
  !  22 Nov 95 Jay Banks: added IF (PRNLEV.GE.2) to three WRITE
  !                       statements.
  !
  !  Jay Banks 27-Nov-95: changed IF (PRNLEV.GE.2) to IF (WRNLEV.GE.2) for
  !  WRITE statements followed by CALL WRNDIE.
  !
  !  28 Nov 95 Jay Banks: removed space from all "GO TO"s, as suggested in
  !  developer.doc.
  !
  !  28 Nov 95 Jay Banks: changed PRNLEV test to WRNLEV for printing error
  !  conditions.
  !
  use chm_kinds
  use dimens_fcm
  use consta
  use mmffm
  use param
  use stream
  implicit none
  !
  integer i, ISUPP
  integer j, k, jj
  integer(chm_int8) :: KTOR, KTLAST
  integer l
  integer lutor, maxcol, mclass, n
  integer nline, nstar, nt
  integer namper, natsign, npound
  real(chm_real)  t1, t2, t3
  !
  CHARACTER(len=80) LINE
  CHARACTER(len=36) SOURCE
  LOGICAL ERROR,NEWPAR,FULL
  !     CHARACTER*80 TORF
  !
  error=.false.
  !   READ REGULAR TORSION PARAMETERS
  NCP=0
  KTLAST=-1
  loop10: do while(.true.)
     READ(LUTOR,'(A)') LINE
     IF(LINE(1:1).EQ.'$') exit loop10
     IF(LINE(1:1).EQ.'*') cycle loop10
     READ(LINE,20) MCLASS,I,J,K,L,T1,T2,T3,SOURCE
20   FORMAT(I1,I4,3I5,3F8.3,A)
     !
     !   FORM PACKED INDEX
     !
     CALL OMEGCON('PUT',I,J,K,L,MCLASS,KTOR)
     !      IF(I*J*K*L.EQ.1) WRITE(6,*) ' 1 1 1 1, KTOR = ',KTOR
     !     WRITE(12,*) I,J,K,L,MCLASS
     !
     !   IS CANONICAL ORDER BEING OBEYED?
     !
     IF(KTOR.LE.KTLAST) THEN
        IF (WRNLEV.GE.2) WRITE(OUTU,'(A/A/A)') &
             ' *** Canonical order violated: misordered line follows: ***', &
             LINE(1:78),' *** Execution terminated ****'
        CALL WRNDIE(-5,'<RDTORS>','Canonical order violated')
     ELSE
        KTLAST=KTOR
     ENDIF
     IF(NCP.GE.MAXCP) THEN
        IF (WRNLEV.GE.2) WRITE(OUTU,'(A,I6,A/A)') &
             ' ***** Too many dihedral angle parameters read in '// &
             '... max = ',MAXCP,' *****',' *** Execution terminated ****'
        CALL WRNDIE(-5,'<rdtors>','Too many dihedral angle'// &
             ' parameters')
     ENDIF
     NCP=NCP+1
     !      if(mod(ncp,100).eq.0)
     !     . write(outu,*) ncp,' ',line(:60)
     KCP(NCP)=KTOR
     CPC(3 * NCP - 2) = T1 * 0.5
     CPC(3 * NCP - 1) = T2 * 0.5
     CPC(3 * NCP) = T3 * 0.5
     !
     !-------------------------------------------------------
     ! if everything is OK this should allow to use
     ! charmm EPHI routine to calculate torsion energy
     !
     jj=3*(NCP-1)
     TorPr(jj+1)=-1   ! = CPD in CHARMM (periodicity)
     TorPr(jj+2)=-2   ! - indicates that there are more terms
     TorPr(jj+3)=3    ! compare with parmio.src
     !                       IF (KCP(I).EQ.KCP(I+1)) CPD(I) = - ABS(CPD(I))
     TorPh(jj+1)=0.   ! = CPB in CHARMM (phase)
     TorPh(jj+2)=PI
     TorPh(jj+3)=0.
     !-------------------------------------------------------
     !
     t_source(NCP)=source
  enddo loop10

  if(prnlev.gt.2) WRITE(OUTU,'(a,i5,a,i5)') 'RDTORS> ',NCP, &
       ' TORSION PARAMETERS READ FROM UNIT ',LUTOR
  NCPT=NCP
  !-----------------------------------------------------------------------
  IF (NT.EQ.0) RETURN
  !  READ CHANGED CONSTANTS (NT.GT.0)
  if(prnlev.gt.2) WRITE(OUTU,'(I5,A)') &
       NT,' SUPPLEMENTARY TORSION ANGLE PARAMETERS READ'
  call ffind_loc(ISUPP,'TORSION',ERROR)
  if(ERROR) then
     call wrndie(-1,'<RDTORS>','EOF in supplementary file')
     return
  endif
  DO N=1,NT
     !   READ TORSION-PARAMETER LINE INTO CHARACTER BUFFER
     READ(ISUPP,'(A)') LINE
     IF(LINE(1:1).EQ.'*') THEN
        !   REMOVE THE "*" (WHICH INDICATES A PARAMETER LINE TO BE IGNORED
        !   IN TORSION FITTING BY PROGRAM TORFIT) FROM COL 1 AND SHIFT
        !   EVERYTHING ONE COLUMN TO THE LEFT
        NLINE=LEN_TRIM(LINE)
        NLINE=MAX(NLINE,2)
        DO J=1,NLINE-1
           LINE(J:J)=LINE(J+1:J+1)
        enddo
        LINE(NLINE:NLINE)=' '
     ENDIF
     !   NOW CHANGE ANY EMBEDDED *'S TO BLANKS
     MAXCOL=46
1085 CONTINUE
     NSTAR=INDEX(LINE,'*')
     IF(NSTAR.GT.0.AND.NSTAR.LT.MAXCOL) THEN
        LINE(NSTAR:NSTAR)=' '
        GOTO 1085
     ENDIF
     !   NOW CHANGE ANY EMBEDDED #'S TO BLANKS
1090 CONTINUE
     NPOUND=INDEX(LINE,'#')
     IF(NPOUND.GT.0.AND.NPOUND.LT.MAXCOL) THEN
        LINE(NPOUND:NPOUND)=' '
        GOTO 1090
     ENDIF
     !   NOW CHANGE ANY EMBEDDED &'S TO BLANKS
1091 CONTINUE
     NAMPER=INDEX(LINE,'&')
     IF(NAMPER.GT.0.AND.NAMPER.LT.MAXCOL) THEN
        LINE(NAMPER:NAMPER)=' '
        GOTO 1091
     ENDIF
     !   NOW CHANGE ANY EMBEDDED @'S TO BLANKS
1092 CONTINUE
     NATSIGN=INDEX(LINE,'@')
     IF(NATSIGN.GT.0.AND.NATSIGN.LT.MAXCOL) THEN
        LINE(NATSIGN:NATSIGN)=' '
        GOTO 1092
     ENDIF
     !   FINALLY, READ THE SOUGHT VARIABLES FROM "LINE"
     READ(LINE,1100) MCLASS,I,J,K,L,T1,T2,T3,SOURCE
1100 FORMAT(I1,I4,3I5,3F8.3,A)
     !
     !   FORM PACKED, CANONICALLY ORDERED INDEX
     !
     CALL OMEGCON('PUT',I,J,K,L,MCLASS,KTOR)
     !  FIND THE ANGLE IN THE KCP LIST.  IF IT'S NOT THERE - ADD IT.
     CALL BIFIND8(KTOR,KCP,NCP,J,NCPT,MAXCP,NEWPAR,FULL,1)
     IF(.NOT.FULL) THEN
        IF(NEWPAR) KCP(J)=KTOR
        CPC(3 * J - 2) = T1 * 0.5
        CPC(3 * J - 1) = T2 * 0.5
        CPC(3 * J) = T3 * 0.5
        jj=3*(j-1)
        TorPr(jj+1)=-1   ! = CPD in CHARMM (periodicity)
        TorPr(jj+2)=-2   ! - indicates that there are more terms
        TorPr(jj+3)=3    ! compare with parmio.src
        !                       IF (KCP(I).EQ.KCP(I+1)) CPD(I) = - ABS(CPD(I))
        TorPh(jj+1)=0.   ! = CPB in CHARMM (phase)
        TorPh(jj+2)=PI
        TorPh(jj+3)=0.
     ELSE
        !  IF THIS POINT IS REACHED, THE KCP ARRAY IS FILLED
        IF (WRNLEV.GE.2) WRITE(OUTU,2400) I
        !     IF(IZ.NE.0) WRITE(IZ,2400) I
2400    FORMAT (/'* * * * * ERROR - THE ARRAY OF TORSIONAL ANGLE ', &
             'TYPES IS FILLED.  TORSIONAL CARD #',I3,' IS SKIPPED')
     ENDIF
  enddo
  !
  RETURN
END SUBROUTINE RDTORS

! ===================================================================
! SUBROUTINE RDVDWM : READ VAN DER WAALS PARAMETERS
! ===================================================================
!
SUBROUTINE RDVDWM(NV,LUVDW,ISUPP,ERROR)
  !
  !  16 Oct 95 Jay Banks: changed variable name IT to ITP
  !                       to avoid conflict with CHARMM name
  !
  !  22 Nov 95 Jay Banks: added IF (PRNLEV.GE.2) to a WRITE statement.
  !
  !  27 Nov 95 Jay Banks: Changed two PRNLEV tests to WRNLEV.
  !
  !  28 Nov 95 Jay Banks: removed space from all "GO TO"s, as suggested in
  !  developer.doc.
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use mmffm
  use inbnd
  use memory
  implicit none
  real(chm_real),allocatable,dimension(:) :: AFACTR
  real(chm_real),allocatable,dimension(:) :: GFACTR
  real(chm_real),allocatable,dimension(:) :: ALPHA
  real(chm_real),allocatable,dimension(:) :: EFFN
  integer,allocatable,dimension(:) :: ioff
  integer NV,LUVDW,ISUPP
  logical ERROR
  !
  character(len=1),allocatable,dimension(:) :: dachar
  !
  error=.false.
  call chmalloc('readpar.src','RDVDWM','AFACTR',MAXATC,crl=AFACTR)
  call chmalloc('readpar.src','RDVDWM','GFACTR',MAXATC,crl=GFACTR)
  call chmalloc('readpar.src','RDVDWM','ALPHA',MAXATC,crl=ALPHA)
  call chmalloc('readpar.src','RDVDWM','EFFN',MAXATC,crl=EFFN)
  call chmalloc('readpar.src','RDVDWM','dachar',MAXATC,ch1=dachar)
  call chmalloc('readpar.src','RDVDWM','ioff',MAXATC,intg=ioff)
  !
  call RDVDWM2(NV,LUVDW,ISUPP,error,AFACTR,GFACTR,ALPHA, &
       EFFN,DACHAR,ioff)
  call chmdealloc('readpar.src','RDVDWM','AFACTR',MAXATC,crl=AFACTR)
  call chmdealloc('readpar.src','RDVDWM','GFACTR',MAXATC,crl=GFACTR)
  call chmdealloc('readpar.src','RDVDWM','ALPHA',MAXATC,crl=ALPHA)
  call chmdealloc('readpar.src','RDVDWM','EFFN',MAXATC,crl=EFFN)
  call chmdealloc('readpar.src','RDVDWM','dachar',MAXATC,ch1=dachar)
  call chmdealloc('readpar.src','RDVDWM','ioff',MAXATC,intg=ioff)
  return
end SUBROUTINE RDVDWM
!
SUBROUTINE RDVDWM2(NV,LUVDW,ISUPP,error, &
     AFACTR,GFACTR,ALPHA,EFFN,DACHAR,IOFF)
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   Adapted by R. Czerminski of Molecular Simulations, Inc. from
  !  code developed at Merck and Co., Inc. by T. Halgren and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !
  use chm_kinds
  use dimens_fcm
  use vector
  use mmffm
  use inbnd
  use number
  use param
  use stream
  implicit none
  !
  LOGICAL ERROR
  real(chm_real) AFACTR(MAXATC),GFACTR(MAXATC),ALPHA(MAXATC), &
       EFFN(MAXATC)
  CHARACTER(len=1) DACHAR(MAXATC)
  integer ioff(MAXATC)
  !
  integer i, ISUPP, itp, j, ij
  integer luvdw
  integer nv, nvdw
  integer id, jd
  real(chm_real)  afact
  real(chm_real)  afacts, bfact, bfacts, darad, daeps, darads,  &
       daepss
  real(chm_real)  pexp, pexps
  real(chm_real)  afactit, alphait, effnit, epsfact, gfactit, gij
  real(chm_real)  radii, radjj, yij
  !
  CHARACTER(len=80) LINE
  CHARACTER(len=1) DACHIT
  !     CHARACTER*80 VDWFLE
  !     integer llen
  !     external llen
  !
  DO I = 1,MAXATC
     DACHAR(I)=' '
  ENDDO
  !
  j=0             ! Initialize the code look up offsets
  do i=1,MAXATC   ! for upper triangle matrices
     ioff(i)=j    ! (i-1)*i/2
     j=j+i        ! to make storage compatible with charmm
  enddo
  !
  NVDW=-1
10 CONTINUE
  READ(LUVDW,'(A)') LINE
  IF(LINE(1:1).EQ.'$') GOTO 40
  IF(LINE(1:1).EQ.'*') GOTO 10
  IF(NVDW.EQ.-1) THEN
     READ(LINE,'(F8.3,F10.3,F11.3,F9.3,F11.3)') &
          PEXP,AFACT,BFACT,DARAD,DAEPS
     NVDW=0
     GOTO 10
  ELSE
     READ(LINE,20) ITP,ALPHA(ITP),EFFN(ITP),AFACTR(ITP), &
          GFACTR(ITP),DACHAR(ITP)
     IF(ITP.GT.MAXDEFI) THEN
        IF (WRNLEV.GE.2) WRITE(OUTU,'(A,I6,A/A)') &
             ' ***** Too large a numerical atom type encountered ...'// &
             ' max = ',MAXDEFI,' *****',' *** Execution terminated ****'
        CALL WRNDIE(-5,'<rdvdwm>','Numerical atom type too large')
     ENDIF
     !         WRITE(OUTU,20) ITP,ALPHA(ITP),EFFN(ITP),AFACTR(ITP),
     !     .   GFACTR(ITP),DACHAR(ITP)
20   FORMAT(I5,4F10.3,1X,A1)
     NVDW=MAX(NVDW,ITP)
     GOTO 10
  ENDIF
40 CONTINUE
  !      WRITE(6,*) ' VDWFLE ',VDWFLE
  !     LRB=LPATH(VDWFLE)
  !     LEND=LLEN(VDWFLE)
  !      WRITE(6,*) ' LRB ',LRB
  if(prnlev.ge.2) WRITE(OUTU,'(a,i5,a,i5)') 'rdvdwm2> ',NVDW, &
       ' VDW PARAMETERS READ FROM UNIT ',LUVDW
  !-----------------------------------------------------------------------
  !  READ ANY NEW OR CHANGED CONSTANTS IF NV.NE.0
  !
  NVDWM=NVDW
  IF (NV.EQ.0) GOTO 1400
  if(prnlev.ge.2) WRITE(OUTU,800) NV
  !     IF(IZ.NE.0) WRITE(IZ,800) NV
800 FORMAT(/1H0,9X,'THE FOLLOWING',I3,' VDW PARAMETERS ARE READ'// &
       3X,'TYPE   ALPHA      NEFF      AFACTR    GFACTR   ', &
       ' D/A')
  call ffind_loc(ISUPP,'VDW',ERROR)
  if(ERROR) then
     call wrndie(-1,'<RDVDWM2>','EOF in supplementary file')
     return
  endif
  READ(ISUPP,*) PEXPS,AFACTS,BFACTS,DARADS,DAEPSS
  IF(PEXPS.NE.0.) PEXP=PEXPS
  IF(AFACTS.NE.0) AFACT=AFACTS
  IF(BFACTS.NE.0) BFACT=BFACTS
  IF(DARADS.NE.0) DARAD=DARADS
  IF(DAEPSS.NE.0) DAEPS=DAEPSS
  DO I=1,NV
     READ(ISUPP,1000) ITP,ALPHAIT,EFFNIT,AFACTIT,GFACTIT, &
          DACHIT
1000 FORMAT(I5,4F10.3,1X,A1)
     if(prnlev.ge.2) WRITE(OUTU,1100) ITP,ALPHAIT,EFFNIT,AFACTIT, &
          GFACTIT,DACHIT
     !      if(iz.ne.0) WRITE (iz,1100) ITP,ALPHAIT,EFFNIT,AFACTIT,
     !    .  GFACTIT,DACHIT
1100 FORMAT(5X,I3,2X,4F10.3,A4)
     !     CHECK TO BE SURE 'ITP' DOES NOT EXCEED MAXIMUM
     !     NUMERIC ATOM TYPE (=MAXATC)
     IF (ITP.LE.MAXDEFI .and. ITP.gt.0) THEN
        ALPHA(ITP)=abs(ALPHAIT)
        EFFN(ITP)=max(abs(EFFNIT),HALF)
        AFACTR(ITP)=AFACTIT
        GFACTR(ITP)=GFACTIT
        DACHAR(ITP)=DACHIT
        NVDWM=MAX(NVDWM,ITP)
     ELSE
        if(wrnlev.ge.2) then
           WRITE(OUTU,1100) ITP,ALPHAIT,EFFNIT,AFACTIT,GFACTIT,DACHIT
           WRITE(OUTU,1200)
1200       FORMAT(1H0,'* * * * * ERROR - ATOM TYPE OUT OF RANGE, ', &
                'ABOVE CARD IS SKIPPED')
        ENDIF
     ENDIF
  enddo
  if(prnlev.ge.2) WRITE(OUTU,900) NV
900 FORMAT(I5,' SUPPLEMENTARY MMFF VDW PARAMETERS READ ')
1400 CONTINUE
  !
  !   NOW CALCULATE THE VDW RADII AND WELL DEPTHS TO BE USED IN THE
  !   MMFF CALCULATION
  !
  EPSFACT=181.16
  !
  !   epsfact yields eps(i,j) in kcal/mol;  it is the conversion factor
  !   shown in eq 35 of T. A. Halgren, J. Am. Chem. Soc.,114, 7827-7843
  !   (1992).  The other quantities in the expression for eps(i,j) below
  !   are named in analogy to their names in eq 35 of that paper and
  !   are in the units defined there.
  !
  loop2000: DO I=1,NVDWM
     IF(DACHAR(I).EQ.' ') cycle loop2000
     ID=0
     IF(DACHAR(I).EQ.'D') ID=1
     IF(DACHAR(I).EQ.'A') ID=-1
     !   COMPUTE RSTAR FOR I-I INTERACTION
     RADII=AFACTR(I)*ALPHA(I)**PEXP
     VDWR(I)=RADII
     loop2100: DO J=1,I
        ij=(I-1)*I/2 + J
        IF(DACHAR(J).EQ.' ') cycle loop2100
        JD=0
        IF(DACHAR(J).EQ.'D') JD=1
        IF(DACHAR(J).EQ.'A') JD=-1
        !   COMPUTE RSTAR FOR J-J INTERACTION
        RADJJ=AFACTR(J)*ALPHA(J)**PEXP
        !   USE MODIFIED ARITHMETIC MEAN TO GET RSTAR(I,J)
        IF(ID.EQ.1.OR.JD.EQ.1) THEN
           GIJ=1.
        ELSE
           !              if(RADII+RADJJ.gt.0.0D0) then 
           YIJ=(RADII-RADJJ)/(RADII+RADJJ)
           !              else
           !                 YIJ=0.0D0
           !                 call wrndie(-1,'<RDVDW>','RADII+RADJJ.eq.0')
           !              endif
           GIJ=1. + AFACT*(1.-EXP(-BFACT*YIJ**2))
        ENDIF
        RSTAR(IJ)=0.5*GIJ*(RADII+RADJJ)
        !           RSTAR(I*(I-1)/2+J)=0.5*GIJ*(RADII+RADJJ)
        !            RSTAR(I,J)=(RADII**3+RADJJ**3)/(RADII**2+RADJJ**2)
        !           RSTAR(J,I)=RSTAR(I,J)
        !   COMPUTE ESTAR(I,J)
        !           ESTAR(I*(I-1)/2+J)=EPSFACT*GFACTR(I)*GFACTR(J)*
        ESTAR(IJ)=EPSFACT*GFACTR(I)*GFACTR(J)* &
             ALPHA(I)*ALPHA(J)/ &
             (SQRT(ALPHA(I)/EFFN(I))+SQRT(ALPHA(J)/EFFN(J)))/ &
             RSTAR(IJ)**6
        !    .      RSTAR(I*(I-1)/2+J)**6
        !           ESTAR(J,I)=ESTAR(I,J)
        IF(ID*JD.EQ.-1) THEN
           RSTAR(IJ)=RSTAR(IJ)*DARAD
           ESTAR(IJ)=ESTAR(IJ)*DAEPS
           !              RSTAR(I*(I-1)/2+J)=RSTAR(I*(I-1)/2+J)*DARAD
           !              ESTAR(I*(I-1)/2+J)=ESTAR(I*(I-1)/2+J)*DAEPS
           !               RSTAR(J,I)=RSTAR(I,J)
           !               ESTAR(J,I)=ESTAR(I,J)
        ENDIF
     enddo loop2100
     !D        WRITE(OUTU,*) I,ALPHA(I),EFFN(I),ESTAR(I,I),RSTAR(I,I)
  enddo loop2000
  !
  !.....copy nonbonded parameters for 1-4 interactions
  !
  rstar(maxcn+1:maxcn+ij) = RSTAR(1:ij)
  estar(maxcn+1:maxcn+ij) = ESTAR(1:ij)
  call scalr8(ESTAR(MAXCN+1),ij,v14fac)
  !
  RETURN
END SUBROUTINE RDVDWM2

! =====================================================================
! SUBROUTINE THETCON : PACK/UNPACK INDEX FOR THETA (BOND) ANGLES
!   PACK OR UNPACK A CANONICALLY-ORDERED INDEX FOR THETA (BOND)
!   ANGLES
! =====================================================================
!
SUBROUTINE THETCON(PUTGET,I,J,K,MCLASS,IJKM)
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   Adapted by R. Czerminski of Molecular Simulations, Inc. from
  !  code developed at Merck and Co., Inc. by T. Halgren and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !
  use chm_kinds
  use dimens_fcm
  use ffieldm
  use param
  use mmffm
  implicit none
  CHARACTER(len=*) PUTGET
  integer i, ijkm, j, k, mclass

  integer, parameter :: MCMAX=10,MX=MAXDEFI
  integer, parameter :: MAXTTYPE=598 ! maximum atom type for which packing
  ! formula fits in a 4-byte integer
  real(chm_real), parameter :: HALF=0.5D0, TWO=2.0D0

  INTEGER FOUR
  FOUR(I,J,K,MCLASS)=MCLASS+MCMAX*(MX**2*J+MX*MIN(I,K)+MAX(I,K))
  !
  if (MX .gt. MAXTTYPE) CALL WRNDIE(-5, '<thetcon>', &
       'MAXDEFI too big for index packing')
  IF(PUTGET.EQ.'PUT') THEN     ! PACK THE INDEX
     IJKM=FOUR(I,J,K,MCLASS)
  ELSEIF(PUTGET.EQ.'GET') THEN ! UNPACK THE INDEX
     J=IJKM/MX**2/MCMAX        ! TO RECOVER I,J,K,MCLASS
     I=(IJKM-MCMAX*MX**2*J)/MX/MCMAX
     K=(IJKM-MCMAX*MX**2*J-MCMAX*MX*I)/MCMAX
     MCLASS=(IJKM-MCMAX*MX**2*J-MCMAX*MX*I-MCMAX*K)
  ELSE
     CALL WRNDIE(-5,'<thetcon>','Invalid PUTGET string')
  ENDIF
#if KEY_DEBUG==1
  if(MCLASS.ge.MCMAX .or. MCLASS.lt.0) &
       call wrndie(-5,'<thetcon>','MCLASS.ge.MCMAX')
  if(I.GE.MX .or. J.GE.MX .or. K.GE.MX .or. &
       I.lt.0  .or. J.lt.0  .or. K.lt.0  ) &
       call wrndie(-5,'<thetcon>','I,J or K out of range')
#endif 
  !
  RETURN
END SUBROUTINE THETCON

! =====================================================================
! SUBROUTINE STBNCON : PACK/UNPACK A CANONICALLY-ORDERED INDEX FOR
! STRETCH-BEND INTERACTIONS
! =====================================================================
!
SUBROUTINE STBNCON(PUTGET,I,J,K,MCLASS,MCIJ,MCJK,MCIJK,IJKM)
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
  use string
  implicit none
  CHARACTER(len=*) PUTGET
  !
  integer i, ijkm, ijkm1, ijkm2, j, k, mcij, mcijk, mcjk
  integer mclass, mijk, mkji, mijk0, mkji0
  integer, parameter :: MCMAX=20,MX=MAXDEFI
  integer, parameter :: MAXSBTYPE=474 ! Maximum atom type for which packing
  ! formula fits in a 4-byte integer
  !
  INTEGER THREEM
  THREEM(I,J,K,MCLASS)= &
       MCLASS+MCMAX*(MX**2*J+MX*I+K)
  !
  if (MX .gt. MAXSBTYPE) CALL WRNDIE(-5, '<stbncon>', &
       'MAXDEFI too big for index packing')
  !
  IF(PUTGET.EQ.'PUT') THEN
     !
     !   IF A NEGATIVE VALUE FOR MCLASS IS PROVIDED, COMPUTE MCXX FROM
     !   ITS ABSOLUTE VALUE
     !
     IF(MCLASS.LT.0) THEN
        MCLASS=-MCLASS
        MCIJ=0
        MCJK=0
        MCIJK=0
        IF(MCLASS.EQ.1) THEN
           MCIJK=1
           MCIJ=1
        ELSE IF(MCLASS.EQ.2) THEN
           MCIJK=1
           MCJK=1
        ELSE IF(MCLASS.EQ.3) THEN
           MCIJK=2
           MCIJ=1
           MCJK=1
        ELSE IF(MCLASS.EQ.4) THEN
           MCIJK=4
        ELSE IF(MCLASS.EQ.5) THEN
           MCIJK=3
        ELSE IF(MCLASS.EQ.6) THEN
           MCIJK=5
           MCIJ=1
           MCJK=0
        ELSE IF(MCLASS.EQ.7) THEN
           MCIJK=5
           MCIJ=0
           MCJK=1
        ELSE IF(MCLASS.EQ.8) THEN
           MCIJK=6
           MCIJ=1
           MCJK=1
        ELSE IF(MCLASS.EQ.9) THEN
           MCIJK=7
           MCIJ=1
           MCJK=0
        ELSE IF(MCLASS.EQ.10) THEN
           MCIJK=7
           MCIJ=0
           MCJK=1
        ELSE IF(MCLASS.EQ.11) THEN
           MCIJK=8
           MCIJ=1
           MCJK=1
        ENDIF
     ENDIF
     !
     !   TEST FOR VALID INPUT
     !
     IF(MCIJK.LT.0.OR.MCIJK.GT.8) THEN
        write(SCRTCH,'(a,i5)') 'Invalid MCIJK =',MCIJK
        call wrndie(-5,'<stbncon>',SCRTCH(:20))
     ELSE IF(MCIJ.LT.0.OR.MCIJ.GT.1) THEN
        write(SCRTCH,'(a,i5)') 'Invalid MCIJ  =',MCIJK
        call wrndie(-5,'<stbncon>',SCRTCH(:20))
     ELSE IF(MCJK.LT.0.OR.MCJK.GT.1) THEN
        write(SCRTCH,'(a,i5)') 'Invalid MCJK  =',MCJK
        call wrndie(-5,'<stbncon>',SCRTCH(:20))
     ENDIF
     IF(MCIJK.EQ.0.AND.MCIJ.EQ.0.AND.MCJK.EQ.0) GOTO 10
     IF(MCIJK.EQ.1.AND.MCIJ.EQ.1.AND.MCJK.EQ.0) GOTO 10
     IF(MCIJK.EQ.1.AND.MCIJ.EQ.0.AND.MCJK.EQ.1) GOTO 10
     IF(MCIJK.EQ.2.AND.MCIJ.EQ.1.AND.MCJK.EQ.1) GOTO 10
     IF(MCIJK.EQ.3.AND.MCIJ.EQ.0.AND.MCJK.EQ.0) GOTO 10
     IF(MCIJK.EQ.4.AND.MCIJ.EQ.0.AND.MCJK.EQ.0) GOTO 10
     IF(MCIJK.EQ.5.AND.MCIJ.EQ.1.AND.MCJK.EQ.0) GOTO 10
     IF(MCIJK.EQ.5.AND.MCIJ.EQ.0.AND.MCJK.EQ.1) GOTO 10
     IF(MCIJK.EQ.6.AND.MCIJ.EQ.1.AND.MCJK.EQ.1) GOTO 10
     IF(MCIJK.EQ.7.AND.MCIJ.EQ.1.AND.MCJK.EQ.0) GOTO 10
     IF(MCIJK.EQ.7.AND.MCIJ.EQ.0.AND.MCJK.EQ.1) GOTO 10
     IF(MCIJK.EQ.8.AND.MCIJ.EQ.1.AND.MCJK.EQ.1) GOTO 10
     !
     !  IF STILL HERE, THE COMBINATION OF MCIJK. MCIJ AND MCJK IS INVALID
     !
     write(SCRTCH,'(a,3i5)') 'Invalid MCIJ,MCJK,MCIJK =', &
          MCIJ,MCJK,MCIJK
     call wrndie(-5,'<stbncon>',SCRTCH(:40))
10   CONTINUE
     !
     !   PACK THE INDEX
     !
     MIJK0=2*(MCIJK-MCIJ) + (MCIJK-MCJK)
     IF(MIJK0.LE.3) THEN
        MIJK=MIJK0
     ELSE IF(MIJK0.EQ.9) THEN
        MIJK=5
     ELSE IF(MIJK0.EQ.12) THEN
        MIJK=4
     ELSE IF(MIJK0.EQ.13) THEN
        MIJK=6
     ELSE IF(MIJK0.EQ.14) THEN
        MIJK=7
     ELSE IF(MIJK0.EQ.15) THEN
        MIJK=8
     ELSE IF(MIJK0.EQ.19) THEN
        MIJK=9
     ELSE IF(MIJK0.EQ.20) THEN
        MIJK=10
     ELSE IF(MIJK0.EQ.21) THEN
        MIJK=11
     ELSE
        write(SCRTCH,'(a,i5)') 'Invalid MIJK0 =',MIJK0
        call wrndie(-5,'<stbncon>',SCRTCH(:20))
     ENDIF
     MKJI0=2*(MCIJK-MCJK) + (MCIJK-MCIJ)
     IF(MKJI0.LE.3) THEN
        MKJI=MKJI0
     ELSE IF(MKJI0.EQ.9) THEN
        MKJI=5
     ELSE IF(MKJI0.EQ.12) THEN
        MKJI=4
     ELSE IF(MKJI0.EQ.13) THEN
        MKJI=6
     ELSE IF(MKJI0.EQ.14) THEN
        MKJI=7
     ELSE IF(MKJI0.EQ.15) THEN
        MKJI=8
     ELSE IF(MKJI0.EQ.19) THEN
        MKJI=9
     ELSE IF(MKJI0.EQ.20) THEN
        MKJI=10
     ELSE IF(MKJI0.EQ.21) THEN
        MKJI=11
     ELSE
        write(SCRTCH,'(a,i5)') 'Invalid MKJI0 =',MKJI0
        call wrndie(-5,'<stbncon>',SCRTCH(:20))
     ENDIF
     IJKM1=THREEM(I,J,K,MIJK)
     IJKM2=THREEM(K,J,I,MKJI)
     IF(IJKM1.LE.IJKM2) THEN
        MCLASS=MIJK
        IJKM=IJKM1
     ELSE
        MCLASS=MKJI
        IJKM=IJKM2
     ENDIF
     !
  ELSE IF(PUTGET.EQ.'GET') THEN ! UNPACK THE INDEX
     !                                   ! TO RECOVER I,J,K,MCLASS
     !
     J=IJKM/MX**2/MCMAX
     I=(IJKM-MCMAX*MX**2*J)/MX/MCMAX
     K=(IJKM-MCMAX*MX**2*J-MCMAX*MX*I)/MCMAX
     MCLASS=(IJKM-MCMAX*MX**2*J-MCMAX*MX*I-MCMAX*K)
     !
     !   UNPACK MCLASS
     !
     MCIJ=0
     MCJK=0
     MCIJK=0
     IF(MCLASS.EQ.1) THEN
        MCIJK=1
        MCIJ=1
     ELSE IF(MCLASS.EQ.2) THEN
        MCIJK=1
        MCJK=1
     ELSE IF(MCLASS.EQ.3) THEN
        MCIJK=2
        MCIJ=1
        MCJK=1
     ELSE IF(MCLASS.EQ.4) THEN
        MCIJK=4
     ELSE IF(MCLASS.EQ.5) THEN
        MCIJK=3
     ELSE IF(MCLASS.EQ.6) THEN
        MCIJK=5
        MCIJ=1
        MCJK=0
     ELSE IF(MCLASS.EQ.7) THEN
        MCIJK=5
        MCIJ=0
        MCJK=1
     ELSE IF(MCLASS.EQ.8) THEN
        MCIJK=6
        MCIJ=1
        MCJK=1
     ELSE IF(MCLASS.EQ.9) THEN
        MCIJK=7
        MCIJ=1
        MCJK=0
     ELSE IF(MCLASS.EQ.10) THEN
        MCIJK=7
        MCIJ=0
        MCJK=1
     ELSE IF(MCLASS.EQ.11) THEN
        MCIJK=8
        MCIJ=1
        MCJK=1
     ENDIF
  ELSE
     CALL WRNDIE(-5,'<stbncon>','Invalid PUTGET string')
  ENDIF
#if KEY_DEBUG==1
  if(abs(MCLASS).ge.MCMAX) &
       call wrndie(-5,'<stbncon>','MCLASS.ge.MCMAX')
  if(I.GE.MX .or. J.GE.MX .or. K.GE.MX .or. &
       I.LT.0  .or. J.LT.0  .or. K.LT.0) &
       call wrndie(-5,'<stbncon>','Invalid I,J or K')
#endif 
  !
  RETURN
END SUBROUTINE STBNCON

#endif 

SUBROUTINE NULL_readpar
  RETURN
END SUBROUTINE NULL_readpar

