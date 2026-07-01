module testch_m
  implicit none
contains

SUBROUTINE TESTCH(COMLYN,COMLEN)
  !
  !     TESTCH processes the CHARMM TEST commands.
  !
  use chm_kinds
  use chm_types
  use number
  use memory
  use allocdat
  use deallocdat
  use alloccomp
  use dimens_fcm
  use bases_fcm
  use coord
  use coordc
  use fourdm
  use image
  use image_routines_module, only: inimag
  use psf
  use genpsf_m, only: atmini
  use timerm
  use parallel
  use string
  use egrad, only: testfd
  use vibcom, only: testsd

  implicit none
  ! . Passed variables.
  real(chm_real),allocatable,dimension(:) :: IDDA
  real(chm_real),allocatable,dimension(:) :: IDDF
  real(chm_real),allocatable,dimension(:) :: IDXFD
  real(chm_real),allocatable,dimension(:) :: IDXAD
  real(chm_real),allocatable,dimension(:) :: IXNEW
  real(chm_real),allocatable,dimension(:) :: IYNEW
  real(chm_real),allocatable,dimension(:) :: IZNEW
#if KEY_FOURD==1
  real(chm_real),allocatable,dimension(:) :: IFDNEW
#endif 
  real(chm_real),allocatable,dimension(:) :: IDXF
  real(chm_real),allocatable,dimension(:) :: IDYF
  real(chm_real),allocatable,dimension(:) :: IDZF
  integer,allocatable,dimension(:) :: ISLCT
  integer,allocatable,dimension(:) :: JSLCT
  CHARACTER(len=*) COMLYN
  INTEGER   COMLEN
  ! . Local variables.
  CHARACTER(len=4) WRD, WINIT
  INTEGER   I,J, NODES
  INTEGER   LENV
  LOGICAL   LCOMP, OK, LCRYS, LHOMO,QTIME,QSYNC
  ! . Local variables for memory module
  LOGICAL :: LPSUCC,LPFAIL,LPRALLOC,LPRDEALL
  LOGICAL :: LNOFILE,LNOPROC,LNONAME,LNORANK,LNOTYPE,LNOSIZE, &
       LCASEINS, LMMVERB
  LOGICAL :: LACCUMALLOCDB,LACCUMDEALLOCDB,LNOALLOCDIE,LNODEALLDIE, &
       LPRNALLOCF,LPRNDEALLOCF
  LOGICAL :: LRESALLOC,LRESDEALL
  !
  !C##IF NOMISC
  !C      CALL WRNDIE(-1,'<TESTCH>','TEST code is not compiled.')
  !C      RETURN
  !C##ELSE
  !
  WRD=NEXTA4(COMLYN,COMLEN)
  ! . Branch on the option.
  !=======================================================================
  IF(WRD.EQ.'FIRS') THEN
     ! . Test first derivatives.
     LCRYS=(INDXA(COMLYN,COMLEN,'CRYS').GT.0)
     LHOMO=(INDXA(COMLYN,COMLEN,'HOMO').GT.0)
     LENV=3*NATOM+XDIM
#if KEY_FOURD==1
     IF(DIM4) LENV=LENV+NATOM
#endif 
     call chmalloc('testch.src','TESTCH','IDXFD',LENV,crl=IDXFD)
     call chmalloc('testch.src','TESTCH','IDXAD',LENV,crl=IDXAD)
     call chmalloc('testch.src','TESTCH','IXNEW',NATOM,crl=IXNEW)
     call chmalloc('testch.src','TESTCH','IYNEW',NATOM,crl=IYNEW)
     call chmalloc('testch.src','TESTCH','IZNEW',NATOM,crl=IZNEW)
#if KEY_FOURD==1
     call chmalloc('testch.src','TESTCH','IFDNEW',NATOM,crl=IFDNEW)
#endif 
     call chmalloc('testch.src','TESTCH','ISLCT',NATOM,intg=ISLCT)
     call TESTFD(COMLYN,COMLEN,NATOM,X,Y,Z,WMAIN, &
          IXNEW,IYNEW,IZNEW, &
#if KEY_FOURD==1
          DIM4,FDIM,IFDNEW,DFDIM, & 
#endif
#if KEY_FOURD==0
          .FALSE.,[ZERO],[ZERO],[ZERO], & 
#endif
          IDXFD,IDXAD,ISLCT,IMOVE,LCRYS,LHOMO)
     call chmdealloc('testch.src','TESTCH','IDXFD',LENV,crl=IDXFD)
     call chmdealloc('testch.src','TESTCH','IDXAD',LENV,crl=IDXAD)
     call chmdealloc('testch.src','TESTCH','IXNEW',NATOM,crl=IXNEW)
     call chmdealloc('testch.src','TESTCH','IYNEW',NATOM,crl=IYNEW)
     call chmdealloc('testch.src','TESTCH','IZNEW',NATOM,crl=IZNEW)
#if KEY_FOURD==1
     call chmdealloc('testch.src','TESTCH','IFDNEW',NATOM,crl=IFDNEW)
#endif 
     call chmdealloc('testch.src','TESTCH','ISLCT',NATOM,intg=ISLCT)

     !
     !=======================================================================
  ELSE IF(WRD.EQ.'SECO') THEN
     ! . Test the second derivatives. System should not be to BIG...
     call chmalloc('testch.src','TESTCH','IDDA', &
          (3*NATOM*(3*NATOM+1))/2,crl=IDDA)
     call chmalloc('testch.src','TESTCH','IDDF', &
          (3*NATOM*(3*NATOM+1))/2,crl=IDDF)
     call chmalloc('testch.src','TESTCH','IXNEW',NATOM,crl=IXNEW)
     call chmalloc('testch.src','TESTCH','IYNEW',NATOM,crl=IYNEW)
     call chmalloc('testch.src','TESTCH','IZNEW',NATOM,crl=IZNEW)
     call chmalloc('testch.src','TESTCH','IDXF',NATOM,crl=IDXF)
     call chmalloc('testch.src','TESTCH','IDYF',NATOM,crl=IDYF)
     call chmalloc('testch.src','TESTCH','IDZF',NATOM,crl=IDZF)
     call chmalloc('testch.src','TESTCH','ISLCT',NATOM,intg=ISLCT)
     call chmalloc('testch.src','TESTCH','JSLCT',NATOM,intg=JSLCT)

     call TESTSD(COMLYN,COMLEN,NATOM,X,Y,Z,WMAIN,IDDA,IDDF,IXNEW, &
          IYNEW,IZNEW,IDXF,IDYF,IDZF,islct,jslct)
     call chmdealloc('testch.src','TESTCH','IDDF', &
          (3*NATOM*(3*NATOM+1))/2,crl=IDDF)
     call chmdealloc('testch.src','TESTCH','IDDA', &
          (3*NATOM*(3*NATOM+1))/2,crl=IDDA)
     call chmdealloc('testch.src','TESTCH','IXNEW',NATOM,crl=IXNEW)
     call chmdealloc('testch.src','TESTCH','IYNEW',NATOM,crl=IYNEW)
     call chmdealloc('testch.src','TESTCH','IZNEW',NATOM,crl=IZNEW)
     call chmdealloc('testch.src','TESTCH','IDXF',NATOM,crl=IDXF)
     call chmdealloc('testch.src','TESTCH','IDYF',NATOM,crl=IDYF)
     call chmdealloc('testch.src','TESTCH','IDZF',NATOM,crl=IDZF)
     call chmdealloc('testch.src','TESTCH','ISLCT',NATOM,intg=ISLCT)
     call chmdealloc('testch.src','TESTCH','JSLCT',NATOM,intg=JSLCT)
     !
     !=======================================================================
  ELSE IF(WRD.EQ.'COOR') THEN
     ! . Test the coordinates.
     LCOMP=(INDXA(COMLYN,COMLEN,'COMP').GT.0)
     IF(LCOMP) THEN
        CALL TSTCRD(OK,XCOMP,YCOMP,ZCOMP,NATOM)
     ELSE
        CALL TSTCRD(OK,X,Y,Z,NATOM)
     ENDIF
     !=======================================================================
     ! . Initialise lots of things.
  ELSE IF(WRD.EQ.'INIT') THEN
     CALL ATMINI(1,MAXA)
     WINIT='INIT'
     J=4
     CALL GTNBCT(WINIT,J,BNBND)
     CALL INIMAG(BIMAG,.TRUE.)
     WINIT='INIT'
     J=4
     CALL GTHBCT(WINIT,J)
     !
     !=======================================================================
     !
     !=======================================================================
#if KEY_PARALLEL==1
  ELSE IF(WRD.EQ.'PARA') THEN
     J=NATOM
     IF(J.LT.MAXNODE) J=MAXNODE
     call chmalloc('testch.src','TESTCH','IXNEW',J,crl=IXNEW)
     call TESTPAR8('X',X,IXNEW,NATOM)
     call TESTPAR8('Y',Y,IXNEW,NATOM)
     call TESTPAR8('Z',Z,IXNEW,NATOM)
     call chmdealloc('testch.src','TESTCH','IXNEW',J,crl=IXNEW)
     !
  ELSE IF(WRD.EQ.'COMM') THEN
     ! simple communications testing
     I=GTRMI(COMLYN,COMLEN,'SIZE',8192)
     J=GTRMI(COMLYN,COMLEN,'COUN',1)
     QTIME=INDXA(COMLYN,COMLEN,'TIME') .GT.0
     QSYNC=INDXA(COMLYN,COMLEN,'SYNC') .GT.0
     call chmalloc('testch.src','TESTCH','IXNEW',I,crl=IXNEW)
     call chmalloc('testch.src','TESTCH','IYNEW',I,crl=IYNEW)
     call TESTCOMM(I,J,IXNEW,IYNEW,QTIME,QSYNC)
     call chmdealloc('testch.src','TESTCH','IXNEW',I,crl=IXNEW)
     call chmdealloc('testch.src','TESTCH','IYNEW',I,crl=IYNEW)
#endif 
     !=======================================================================
     ! . Test connectivity and PSF commands.
  ELSE IF(WRD.EQ.'CONN') THEN
     CALL CONECT(NBOND,IB,JB)
  ELSE IF(WRD.EQ.'PSF') THEN
     CALL TESPSF
     !
     !=======================================================================
     ! . Test parameters.
     !  note: change command name due to conflict with parallel - BRB
  ELSE IF(WRD.EQ.'PARM') THEN
     CALL TESTPARA(COMLYN,COMLEN)
     !
     !=======================================================================
     ! . Test memory 
     !
  ELSE IF(WRD.EQ.'MEMO') THEN
     WRD=NEXTA4(COMLYN,COMLEN)
     ! if accumulating databases (initialized as false in allocdat_ltm, deallocdat_ltm)
     memwrd: IF (WRD.EQ.'ACCU') THEN  
        allocdbarsz = 10000
        dealldbarsz = 10000
        fallocdbarsz = 10000
        fdealldbarsz = 10000
        ! IF turning on accumulation
        IF (INDXA(COMLYN,COMLEN,'STOP').LE.0) THEN
           LACCUMALLOCDB = .false.
           LACCUMDEALLOCDB = .false.
           IF (INDXA(COMLYN,COMLEN,'ALLO').GT.0) LACCUMALLOCDB = .true.
           IF (INDXA(COMLYN,COMLEN,'DEAL').GT.0) LACCUMDEALLOCDB=.true.
           IF (LACCUMALLOCDB) qaccumallocdb = .true.
           IF (LACCUMDEALLOCDB) qaccumdeallocdb=.true.
           ! if neither alloc or dealloc specified, accumulate both
           IF((.not.LACCUMALLOCDB).and.(.not.LACCUMDEALLOCDB)) then
              qaccumallocdb = .true.
              qaccumdeallocdb=.true.
           ENDIF
           WRITE(6,*) 'Accumulating memory database(s)'
        ELSE 
           ! else if stopping accumulation
           LACCUMALLOCDB = .true.
           LACCUMDEALLOCDB = .true.
           IF (INDXA(COMLYN,COMLEN,'ALLO').GT.0) LACCUMALLOCDB = .false.
           IF (INDXA(COMLYN,COMLEN,'DEAL').GT.0) LACCUMDEALLOCDB=.false.
           IF (.not.LACCUMALLOCDB) qaccumallocdb = .false.
           IF (.not.LACCUMDEALLOCDB) qaccumdeallocdb = .false.
           ! if neither alloc or dealloc specified, stop both
           IF((LACCUMALLOCDB).and.(LACCUMDEALLOCDB)) then
              qaccumallocdb = .false.
              qaccumdeallocdb=.false.
           ENDIF
           WRITE(6,*) 'Stopping accumulation of memory database(s)'
        ENDIF !if stopping or starting accumulation  
        ! whether to die on allocation errors, global setting, intialized as false
     ELSE IF (WRD.EQ.'NODI') THEN memwrd
        LNOALLOCDIE = .false.  !local variables
        LNODEALLDIE = .false.
        IF (INDXA(COMLYN,COMLEN,'ALLO').GT.0) LNOALLOCDIE = .true.
        IF (INDXA(COMLYN,COMLEN,'DEAL').GT.0) LNODEALLDIE = .true.
        IF (LNOALLOCDIE) qnoallocdie = .true.
        IF (LNODEALLDIE) qnodealldie = .true.
        ! if neither alloc nor dealloc specified, set both to survive
        IF((.not.LNOALLOCDIE).and.(.not.LNODEALLDIE)) then
           qnoallocdie = .true.
           qnodealldie=.true.
        ENDIF
        WRITE(6,*) 'Preventing termination due to memory errors'
        ! toggles to die 
     ELSE IF (WRD.EQ.'DIE') THEN memwrd
        LNOALLOCDIE = .true.  !local variables
        LNODEALLDIE = .true.
        IF (INDXA(COMLYN,COMLEN,'ALLO').GT.0) LNOALLOCDIE = .false.
        IF (INDXA(COMLYN,COMLEN,'DEAL').GT.0) LNODEALLDIE = .false.
        ! if neither alloc nor dealloc specified, set both to die
        IF((LNOALLOCDIE).and.(LNODEALLDIE)) then
           qnoallocdie = .false.
           qnodealldie = .false.
        ENDIF
        WRITE(6,*) 'Restoring termination due to memory errors'
        ! print database
     ELSE IF (WRD.EQ.'PRDB') THEN  memwrd
        LPSUCC = .false.
        LPFAIL = .false.
        IF (INDXA(COMLYN,COMLEN,'SUCC').GT.0) LPSUCC = .true. 
        IF (INDXA(COMLYN,COMLEN,'FAIL').GT.0) LPFAIL = .true. 
        ! if neither successes or failures specified, print both
        IF ((.NOT.LPSUCC).AND.(.NOT.LPFAIL)) THEN
           LPSUCC = .true.
           LPFAIL = .true.
        ENDIF
        LPRALLOC = .false.
        LPRDEALL = .false.
        IF (INDXA(COMLYN,COMLEN,'ALLO').GT.0) LPRALLOC = .true.
        IF (INDXA(COMLYN,COMLEN,'DEAL').GT.0) LPRDEALL = .true. 
        ! if neither alloc nor dealloc specified, print both
        IF ((.NOT.LPRALLOC).AND.(.NOT.LPRDEALL)) THEN
           LPRALLOC = .true.
           LPRDEALL = .true.
        ENDIF
        IF (LPRALLOC) CALL prn_allocdb(qpsucc=LPSUCC,qpfail=LPFAIL)
        IF (LPRDEALL) CALL prn_deallocdb(qpsucc=LPSUCC,qpfail=LPFAIL)
        ! print allocations/deallocations on the fly
     ELSE IF (WRD.EQ.'PFLY') THEN  memwrd
        ! if starting to print on the fly
        LPRNALLOCF = .false. 
        LPRNDEALLOCF = .false.
        IF (INDXA(COMLYN,COMLEN,'STOP').LE.0) THEN
           IF (INDXA(COMLYN,COMLEN,'ALLO').GT.0) LPRNALLOCF = .true.
           IF (INDXA(COMLYN,COMLEN,'DEAL').GT.0) LPRNDEALLOCF = .true.
           IF (LPRNALLOCF) qprnallocf = .true.
           IF (LPRNDEALLOCF) qprndeallocf=.true.
           ! if neither alloc nor dealloc specified, print both
           IF((.not.LPRNALLOCF).and.(.not.LPRNDEALLOCF)) then
              qprnallocf = .true.
              qprndeallocf=.true.
           ENDIF
           write(6,*) 'To print memory information at runtime'
        ELSE 
           ! else if turning off printing on the fly
           LPRNALLOCF = .true.
           LPRNDEALLOCF = .true.
           IF (INDXA(COMLYN,COMLEN,'ALLO').GT.0) LPRNALLOCF = .false.
           IF (INDXA(COMLYN,COMLEN,'DEAL').GT.0) LPRNDEALLOCF = .false.
           ! if neither alloc nor dealloc specified, turn off printing for both
           IF((LPRNALLOCF).and.(LPRNDEALLOCF)) then
              qprnallocf = .false.
              qprndeallocf=.false.
           ENDIF
           write(6,*) 'To stop printing of memory information at runtime'
        ENDIF ! if starting or stopping print on the fly
        ! comparison of databases
     ELSE IF (WRD.EQ.'COMP') THEN memwrd
        ! decide on comparison criteria
        LNOFILE = .false.
        LNOPROC = .false.
        LNONAME = .false.
        LNOTYPE = .false.
        LNORANK = .false.
        LNOSIZE = .false.
        LCASEINS = .false.  !case insensitive
        LMMVERB = .false.  !verbose
        IF (INDXA(COMLYN,COMLEN,'NOFI').GT.0) LNOFILE = .true.
        IF (INDXA(COMLYN,COMLEN,'NOPR').GT.0) LNOPROC = .true.
        IF (INDXA(COMLYN,COMLEN,'NONA').GT.0) LNONAME = .true.
        IF (INDXA(COMLYN,COMLEN,'NOTY').GT.0) LNOTYPE = .true.
        IF (INDXA(COMLYN,COMLEN,'NORA').GT.0) LNORANK = .true.
        IF (INDXA(COMLYN,COMLEN,'NOSI').GT.0) LNOSIZE = .true.
        IF (INDXA(COMLYN,COMLEN,'INSE').GT.0) LCASEINS = .true.
        IF (INDXA(COMLYN,COMLEN,'VERB').GT.0) LMMVERB = .true.
        call compalloc(qnofile=LNOFILE,qnoproc=LNOPROC, &
             qnoname=LNONAME,qnotype=LNOTYPE,qnorank=LNORANK,qnosize=LNOSIZE, &
             qcaseins=LCASEINS,qmmverb=LMMVERB)
        ! reset (reinitialize) databases
     ELSE IF (WRD.EQ.'RESE') THEN memwrd
        LRESALLOC = .false.
        LRESDEALL = .false.
        IF (INDXA(COMLYN,COMLEN,'ALLO').GT.0) LRESALLOC = .true.
        IF (INDXA(COMLYN,COMLEN,'DEAL').GT.0) LRESDEALL = .true.
        IF (LRESALLOC) call reset_allocdb 
        IF (LRESDEALL) call reset_deallocdb 
        ! if neither alloc nor dealloc specified, print both
        IF((.not.LRESALLOC).and.(.not.LRESDEALL)) then
           call reset_allocdb 
           call reset_deallocdb
        ENDIF
        write(6,*) 'Resetting memory databases'
     ENDIF memwrd ! parse second word in memory command (third word overall)
  ELSE
     CALL WRNDIE(0,'<TESTCH>','UNRECOGNIZED TEST OPTION')
  ENDIF
  !
  RETURN
END SUBROUTINE TESTCH

SUBROUTINE CONECT(NBONDX,IBX,JBX)
  !
  ! routine determines connectivity of a specified set of atoms by
  ! making a recursive search on the chemical bond graph given by
  ! IBX(i),JBX(i),i=1,NBONDX
  !
  ! ----------------------------------------------------
  !
  use chm_kinds
  use dimens_fcm
  use memory
  use exfunc
  !
  ! input/output
  use psf
  implicit none
  integer,allocatable,dimension(:) :: NATBON
  integer,allocatable,dimension(:) :: IATBON
  integer,allocatable,dimension(:) :: SUBSET
  integer,allocatable,dimension(:) :: COMSET
  integer,allocatable,dimension(:) :: LIST
  integer,allocatable,dimension(:) :: VERTEX
  integer,allocatable,dimension(:) :: IND
  integer,allocatable,dimension(:) :: RETADR
  INTEGER NBONDX
  INTEGER IBX(*), JBX(*)
  ! local
  INTEGER MAXLEV
  !
  MAXLEV=NBONDX
  call chmalloc('testch.src','CONECT','NATBON',NATOM,intg=NATBON)
  call chmalloc('testch.src','CONECT','IATBON',NATOM*IATBMX,intg=IATBON)
  call chmalloc('testch.src','CONECT','SUBSET',NATOM,intg=SUBSET)
  call chmalloc('testch.src','CONECT','COMSET',NATOM,intg=COMSET)
  call chmalloc('testch.src','CONECT','LIST',NATOM,intg=LIST)
  call chmalloc('testch.src','CONECT','VERTEX',MAXLEV,intg=VERTEX)
  call chmalloc('testch.src','CONECT','IND',MAXLEV,intg=IND)
  call chmalloc('testch.src','CONECT','RETADR',MAXLEV,intg=RETADR)
  call CONNE2(NBONDX,IBX,JBX,NATBON,IATBON,SUBSET,COMSET, &
       LIST,MAXLEV,VERTEX,IND,RETADR)
  call chmdealloc('testch.src','CONECT','NATBON',NATOM,intg=NATBON)
  call chmdealloc('testch.src','CONECT','IATBON',NATOM*IATBMX,intg=IATBON)
  call chmdealloc('testch.src','CONECT','SUBSET',NATOM,intg=SUBSET)
  call chmdealloc('testch.src','CONECT','COMSET',NATOM,intg=COMSET)
  call chmdealloc('testch.src','CONECT','LIST',NATOM,intg=LIST)
  call chmdealloc('testch.src','CONECT','VERTEX',MAXLEV,intg=VERTEX)
  call chmdealloc('testch.src','CONECT','IND',MAXLEV,intg=IND)
  call chmdealloc('testch.src','CONECT','RETADR',MAXLEV,intg=RETADR)

  RETURN
END SUBROUTINE CONECT

SUBROUTINE CONNE2(NBONDX,IBX,JBX,NATBON,IATBON, &
     SUBSET,COMSET,LIST,MAXLEV,VERTEX,IND,RETADR)
  !
  ! see CONNEC above
  use chm_kinds
  use dimens_fcm
  ! input/output
  use psf
  use comand
  use coord
  use select
  use stream
  use string
  use chutil,only:atomid

  implicit none
  INTEGER NBONDX
  INTEGER IBX(*), JBX(*)
  INTEGER NATBON(*),IATBON(IATBMX,*)
  INTEGER SUBSET(*), COMSET(*)
  INTEGER LIST(*), MAXLEV, VERTEX(*), IND(*), RETADR(*)
  ! local
  INTEGER I, IBT, JBT, MSET, LEVEL, PASSED, NEXT, M
  LOGICAL QCOMM, HIT, QPRINT
  CHARACTER(len=4) WRD
  CHARACTER(len=4) CSTAR
  character(len=8) SI, RI, RE, AT
  INTEGER, PARAMETER :: MARK=-1
  ! begin
  !
  ! command parsing
  DO I=1,NATOM
     SUBSET(I)=1
  ENDDO
  QCOMM=.FALSE.
  QPRINT=.FALSE.
  WRD=NEXTA4(COMLYN,COMLEN)
  DO WHILE(WRD.NE.' ')
     IF(WRD.EQ.'SUBS') THEN
        CALL SELCTA(COMLYN,COMLEN,SUBSET,X,Y,Z,WMAIN,.TRUE.)
     ELSE IF(WRD.EQ.'COMM') THEN
        QCOMM=.TRUE.
        CALL SELCTA(COMLYN,COMLEN,COMSET,X,Y,Z,WMAIN,.TRUE.)
     ELSE IF(WRD.EQ.'PRIN') THEN
        QPRINT=.TRUE.
     ELSE
        CALL WRNDIE(0,'<CONNE2>','unkown option')
     ENDIF
     WRD=NEXTA4(COMLYN,COMLEN)
  ENDDO
  !
  ! first make list of all bonds of all atoms (vertices)
  DO I=1,NATOM
     NATBON(I)=0
  ENDDO
  DO I=1,NBONDX
     IBT=IBX(I)
     JBT=JBX(I)
     NATBON(IBT)=NATBON(IBT)+1
     NATBON(JBT)=NATBON(JBT)+1
     IF (NATBON(IBT).GT.IATBMX.OR.NATBON(JBT).GT.IATBMX) THEN
        CALL WRNDIE(-1,'<CONNEC>','IATBMX exceeded')
     ENDIF
     IATBON(NATBON(JBT),JBT)=IBT
     IATBON(NATBON(IBT),IBT)=JBT
  ENDDO
  !
  ! do search in bond graph
  DO I=1,NATOM
     LIST(I)=MARK
  ENDDO
  LEVEL=1
  MSET=0
  !
  I=0
  !CC      DO WHILE(I.LT.NATOM)
101 CONTINUE
  I=I+1
  !CC         IF (LIST(I).EQ.MARK.AND.SUBSET(I).EQ.1) THEN
  IF (.NOT.(LIST(I).EQ.MARK.AND.SUBSET(I).EQ.1)) GOTO 102
  MSET=MSET+1
  !
  ! *invoke PROCEDURE SEARCH (PASSED:=VERTEX)
  PASSED=I
  RETADR(LEVEL)=1
  GOTO 2000
2001 CONTINUE
  ! *return address
  !
  !CC         ENDIF
102 CONTINUE
  !CC      ENDDO
  IF(I.LT.NATOM) GOTO 101
  GOTO 2999
  !
  !---- PROCEDURE SEARCH ( CALL BY VALUE: PASSED->VERTEX ) --------------
  !     LOCAL: IND
  ! *head
2000 CONTINUE
  LEVEL=LEVEL+1
  IF (LEVEL.GT.MAXLEV) CALL WRNDIE(-2,'<CONNE2>','MAXLEV exceeded')
  VERTEX(LEVEL)=PASSED
  !
  ! *begin
  LIST(VERTEX(LEVEL))=MSET
  !
  ! * *loop head: FOR { IND(LEVEL)=1,NATBON(VERTEX(LEVEL) }
  IND(LEVEL)=0
7711 CONTINUE
  ! * *loop iteration
  IND(LEVEL)=IND(LEVEL)+1
  IF (IND(LEVEL).GT.NATBON(VERTEX(LEVEL))) GOTO 7722
  ! * *loop begin
  !
  NEXT=IATBON(IND(LEVEL),VERTEX(LEVEL))
  IF (LIST(NEXT).NE.-1.AND.LIST(NEXT).NE.MSET) THEN
     CALL WRNDIE(0,'<CONNE2>','check algorithm')
  ENDIF
  !CC      IF (LIST(NEXT).EQ.MARK.AND.SUBSET(NEXT).EQ.1) THEN
  IF (.NOT.(LIST(NEXT).EQ.MARK.AND.SUBSET(NEXT).EQ.1)) GOTO 202
  !
  ! * * *invoke procedure SEARCH ( PASSED:=NEXT )
  PASSED=NEXT
  RETADR(LEVEL)=2
  GOTO 2000
2002 CONTINUE
  ! * * *return label
  !
  !CC      ENDIF
202 CONTINUE
  !
  ! * *end loop
  GOTO 7711
7722 CONTINUE
  ! * *exit loop
  !
  ! *return to address
  LEVEL=LEVEL-1
  IF (LEVEL.LE.0) CALL WRNDIE(-1,'<CONNE2>','Level underflow')
  IF (RETADR(LEVEL).EQ.2) GOTO 2002
  IF (RETADR(LEVEL).EQ.1) GOTO 2001
  CALL WRNDIE(-2,'<CONNE2>','Unknown return address, check code')
  !--END PROCEDURE SEARCH-----------------------------------------------
  !
2999 CONTINUE
  !
  ! make printouts
  IF(PRNLEV.GE.2) WRITE(OUTU,9000) &
       ' CONNECt: selected atoms form ',MSET,' disconnected set(s)'
  !
  IF (QCOMM) THEN
     DO M=1,MSET
        HIT=.FALSE.
        I=1
        DO WHILE (.NOT. HIT .AND. I.LE.NATOM)
           IF (LIST(I).EQ.M.AND.COMSET(I).EQ.1) HIT=.TRUE.
           I=I+1
        ENDDO
        IF(HIT) THEN
           IF(PRNLEV.GE.2) WRITE(OUTU,9000) &
                '          set ',M,' contains atoms of COMMon set'
        ELSE
           IF(PRNLEV.GE.2) WRITE(OUTU,9000) &
                '          set ',M,' does not contain any atoms of COMMon set'
        ENDIF
     ENDDO
  ENDIF
  !
  IF (QPRINT) THEN
     DO M=1,MSET
        IF(PRNLEV.GE.2) WRITE(OUTU,9000)
        IF(PRNLEV.GE.2) WRITE(OUTU,9000) &
             ' CONNECt: list of all atoms in set ',M
        DO I=1,NATOM
           IF (LIST(I).EQ.M) THEN
              IF(QCOMM.AND.COMSET(I).EQ.1) THEN
                 CSTAR='   *'
              ELSE
                 CSTAR='    '
              ENDIF
              CALL ATOMID(I,SI,RI,RE,AT)
              IF(PRNLEV.GE.2) WRITE(OUTU,9010)'        ',CSTAR, &
                   '  (',SI(1:idleng),' ',RI(1:idleng),' ', &
                   RE(1:idleng),' ',AT(1:idleng),')'
9010          FORMAT(11A)
           ENDIF
        ENDDO
     ENDDO
  ENDIF
  !
9000 FORMAT(A,I6,A)
  RETURN
END SUBROUTINE CONNE2

SUBROUTINE TESPSF
  !
  ! routine tests PSF for out-of-range entries and duplications
  !
  ! Axel Brunger, Cambridge, September 1983
  ! ---------------------------------------
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  !
  ! input/output
  use psf
  use stream
  use ffieldm
  use mmffm
  use chutil,only:atomid

  implicit none
  ! local
  LOGICAL CONDIT
  INTEGER I,J
  EXTERNAL  EXCH5
  CHARACTER(len=8) S1,R1,N1,A1,S2,R2,N2,A2,S3,R3,N3,A3,S4,R4,N4,A4
#if KEY_CMAP==1
  EXTERNAL EXCH8
  CHARACTER(len=8) S5,R5,N5,A5,S6,R6,N6,A6,S7,R7,N7,A7,S8,R8,N8,A8
#endif 
  ! begin
  !
  ! check out of range entries
  DO I=1,NBOND
     CONDIT=(IB(I).GT.0.AND.IB(I).LE.NATOM.AND. &
          JB(I).GT.0.AND.JB(I).LE.NATOM)
     IF (.NOT.CONDIT) THEN
        IF(WRNLEV.GE.2) WRITE(OUTU,9010) &
             ' TESPSF-ERROR: bond ',I,' out-of-range'
     ENDIF
  ENDDO
  !
  DO I=1,NTHETA
     CONDIT=(IT(I).GT.0.AND.IT(I).LE.NATOM.AND. &
          JT(I).GT.0.AND.JT(I).LE.NATOM.AND. &
          KT(I).GT.0.AND.KT(I).LE.NATOM)
     IF (.NOT.CONDIT) THEN
        IF(WRNLEV.GE.2) WRITE(OUTU,9010) &
             ' TESPSF-ERROR: angle ',I,' out-of-range'
     ENDIF
  ENDDO
  !
  DO I=1,NPHI
     CONDIT=(IP(I).GT.0.AND.IP(I).LE.NATOM.AND. &
          JP(I).GT.0.AND.JP(I).LE.NATOM.AND. &
          KP(I).GT.0.AND.KP(I).LE.NATOM.AND. &
          LP(I).GT.0.AND.KP(I).LE.NATOM)
     IF (.NOT.CONDIT) THEN
        IF(WRNLEV.GE.2) WRITE(OUTU,9010) &
             ' TESPSF-ERROR: dihedral ',I,' out-of-range'
     ENDIF
  ENDDO
  !
  DO I=1,NIMPHI
     CONDIT=(IM(I).GT.0.AND.IM(I).LE.NATOM.AND. &
          JM(I).GT.0.AND.JM(I).LE.NATOM.AND. &
          KM(I).GT.0.AND.KM(I).LE.NATOM.AND. &
          LM(I).GT.0.AND.LM(I).LE.NATOM)
     IF (.NOT.CONDIT) THEN
        IF(WRNLEV.GE.2) WRITE(OUTU,9010) &
             ' TESPSF-ERROR: improper ',I,' out-of-range'
     ENDIF
  ENDDO

#if KEY_CMAP==1
  DO I=1,NCRTERM
     CONDIT=(I1CT(I).GT.0.AND.I1CT(I).LE.NATOM.AND. &
          J1CT(I).GT.0.AND.J1CT(I).LE.NATOM.AND. &
          K1CT(I).GT.0.AND.K1CT(I).LE.NATOM.AND. &
          L1CT(I).GT.0.AND.L1CT(I).LE.NATOM.AND. &
          I2CT(I).GT.0.AND.I2CT(I).LE.NATOM.AND. &
          J2CT(I).GT.0.AND.J2CT(I).LE.NATOM.AND. &
          K2CT(I).GT.0.AND.K2CT(I).LE.NATOM.AND. &
          L2CT(I).GT.0.AND.L2CT(I).LE.NATOM)
     IF (.NOT.CONDIT) THEN
        IF(WRNLEV.GE.2) WRITE(OUTU,9010) &
             ' TESPSF-ERROR: cross-term ',I,' out-of-range'
     ENDIF
  ENDDO
#endif 
  !
  DO I=1,NDON
     CONDIT=(IDON(I).GT.0.AND.IDON(I).LE.NATOM.AND. &
          IHD1(I).GE.0.AND.IHD1(I).LE.NATOM)
     IF (.NOT.CONDIT) THEN
        IF(WRNLEV.GE.2) WRITE(OUTU,9010) &
             ' TESPSF-ERROR: donor ',I,' out-of-range'
     ENDIF
  ENDDO
  !
  DO I=1,NACC
     CONDIT=(IACC(I).GT.0.AND.IACC(I).LE.NATOM.AND. &
          IAC1(I).GE.0.AND.IAC1(I).LE.NATOM)
     IF (.NOT.CONDIT) THEN
        IF(WRNLEV.GE.2) WRITE(OUTU,9010) &
             ' TESPSF-ERROR: acceptor ',I,' out-of-range'
     ENDIF
  ENDDO
  !
  ! now sort all lists
#if KEY_MMFF==1
  IF(FFIELD.eq.MMFF) THEN
     CALL SORT(NBOND,EXCH5,ORDER5,IB,JB,BondType,ICBAUTO,0,0,0,4)
     CALL SORT(NTHETA,EXCH5,ORDER5,IT,JT,KT,LTHETA,0,0,0,4)
  ELSE
     CALL SORT(NBOND,EXCH5,ORDER5,IB,JB,ICBAUTO,0,0,0,0,3)
     CALL SORT(NTHETA,EXCH5,ORDER5,IT,JT,KT,0,0,0,0,3)
  ENDIF
#else /**/
  CALL SORT(NBOND,EXCH5,ORDER5,IB,JB,ICBAUTO,0,0,0,0,3)
  CALL SORT(NTHETA,EXCH5,ORDER5,IT,JT,KT,0,0,0,0,3)
#endif 

  CALL SORT(NPHI,EXCH5,ORDER5,IP,JP,KP,LP,0,0,0,4)
  CALL SORT(NPAUTO,EXCH5,ORDER5,IPAUTO,JPAUTO,KPAUTO,LPAUTO,ICPAUTO,0,0,5)
  CALL SORT(NIMPHI,EXCH5,ORDER5,IM,JM,KM,LM,0,0,0,4)
#if KEY_CMAP==1
  CALL SORT(NCRTERM,EXCH8,ORDER8, &
       I1CT,J1CT,K1CT,L1CT,I2CT,J2CT,K2CT,L2CT)
#endif 

  CALL SORT(NDON,EXCH5,ORDER5,IDON,IHD1,0,0,0,0,0,2)
  CALL SORT(NACC,EXCH5,ORDER5,IACC,IAC1,0,0,0,0,0,2)
  !
  ! now check for duplications
  DO I=2,NBOND
     CONDIT=(IB(I).EQ.IB(I-1).AND. &
          JB(I).EQ.JB(I-1))
     IF (CONDIT) THEN
        J=IB(I)
        CALL ATOMID(J,S1,R1,N1,A1)
        J=JB(I)
        CALL ATOMID(J,S2,R2,N2,A2)
        IF(WRNLEV.GE.2) WRITE(OUTU,9020) &
             ' TESPSF-ERROR: Duplication of bond ', &
             I-1,' and ',I,'  IDs: (', &
             S1(1:idleng),' ',R1(1:idleng),' ',A1(1:idleng),')(', &
             S2(1:idleng),' ',R2(1:idleng),' ',A2(1:idleng),')'
     ENDIF
  ENDDO
  !
  DO I=2,NTHETA
     CONDIT=(IT(I).EQ.IT(I-1).AND. &
          JT(I).EQ.JT(I-1).AND. &
          KT(I).EQ.KT(I-1))
     IF (CONDIT) THEN
        J=IT(I)
        CALL ATOMID(J,S1,R1,N1,A1)
        J=JT(I)
        CALL ATOMID(J,S2,R2,N2,A2)
        J=KT(I)
        CALL ATOMID(J,S3,R3,N3,A3)
        IF(WRNLEV.GE.2) WRITE(OUTU,9020) &
             ' TESPSF-ERROR: Duplication of angle ', &
             I-1,' and ',I,'  IDs: (', &
             S1(1:idleng),' ',R1(1:idleng),' ',A1(1:idleng),')(', &
             S2(1:idleng),' ',R2(1:idleng),' ',A2(1:idleng),')(', &
             S3(1:idleng),' ',R3(1:idleng),' ',A3(1:idleng),')'
     ENDIF
  ENDDO
  !
  DO I=2,NPHI
     CONDIT=(IP(I).EQ.IP(I-1).AND. &
          JP(I).EQ.JP(I-1).AND. &
          KP(I).EQ.KP(I-1).AND. &
          LP(I).EQ.LP(I-1))
     IF (CONDIT) THEN
        J=IP(I)
        CALL ATOMID(J,S1,R1,N1,A1)
        J=JP(I)
        CALL ATOMID(J,S2,R2,N2,A2)
        J=KP(I)
        CALL ATOMID(J,S3,R3,N3,A3)
        J=LP(I)
        CALL ATOMID(J,S4,R4,N4,A4)
        IF(WRNLEV.GE.2) WRITE(OUTU,9020) &
             ' TESPSF-ERROR: Duplication of dihedral ', &
             I-1,' and ',I,'  IDs: (',S1,' ',R1,' ',A1,')(', &
             S1(1:idleng),' ',R1(1:idleng),' ',A1(1:idleng),')(', &
             S2(1:idleng),' ',R2(1:idleng),' ',A2(1:idleng),')(', &
             S3(1:idleng),' ',R3(1:idleng),' ',A3(1:idleng),')(', &
             S4(1:idleng),' ',R4(1:idleng),' ',A4(1:idleng),')'
     ENDIF
  ENDDO
  !
  DO I=2,NPAUTO
     CONDIT=(IPAUTO(I).EQ.IPAUTO(I-1).AND. &
             JPAUTO(I).EQ.JPAUTO(I-1).AND. &
             KPAUTO(I).EQ.KPAUTO(I-1).AND. &
             LPAUTO(I).EQ.LPAUTO(I-1))
     IF (CONDIT) THEN
        J=IPAUTO(I)
        CALL ATOMID(J,S1,R1,N1,A1)
        J=JPAUTO(I)
        CALL ATOMID(J,S2,R2,N2,A2)
        J=KPAUTO(I)
        CALL ATOMID(J,S3,R3,N3,A3)
        J=LPAUTO(I)
        IF(J>0) then
           CALL ATOMID(J,S4,R4,N4,A4)
           IF(WRNLEV.GE.2) WRITE(OUTU,9020) &
             ' TESPSF-ERROR: Duplication of autogen dihedral ', &
             I-1,' and ',I,'  IDs: (',S1,' ',R1,' ',A1,')(', &
             S1(1:idleng),' ',R1(1:idleng),' ',A1(1:idleng),')(', &
             S2(1:idleng),' ',R2(1:idleng),' ',A2(1:idleng),')(', &
             S3(1:idleng),' ',R3(1:idleng),' ',A3(1:idleng),')(', &
             S4(1:idleng),' ',R4(1:idleng),' ',A4(1:idleng),')'
        else
           IF(WRNLEV.GE.2) WRITE(OUTU,9020) &
             ' TESPSF-ERROR: Duplication of autogen angle ', &
             I-1,' and ',I,'  IDs: (',S1,' ',R1,' ',A1,')(', &
             S1(1:idleng),' ',R1(1:idleng),' ',A1(1:idleng),')(', &
             S2(1:idleng),' ',R2(1:idleng),' ',A2(1:idleng),')(', &
             S3(1:idleng),' ',R3(1:idleng),' ',A3(1:idleng),')'
        endif
     ENDIF
  ENDDO
  !
  DO I=2,NIMPHI
     CONDIT=(IM(I).EQ.IM(I-1).AND. &
          JM(I).EQ.JM(I-1).AND. &
          KM(I).EQ.KM(I-1).AND. &
          LM(I).EQ.LM(I-1))
     IF (CONDIT) THEN
        J=IM(I)
        CALL ATOMID(J,S1,R1,N1,A1)
        J=JM(I)
        CALL ATOMID(J,S2,R2,N2,A2)
        J=KM(I)
        CALL ATOMID(J,S3,R3,N3,A3)
        J=LM(I)
        CALL ATOMID(J,S4,R4,N4,A4)
        IF(WRNLEV.GE.2) WRITE(OUTU,9020) &
             ' TESPSF-ERROR: Duplication of improper ', &
             I-1,' and ',I,'  IDs: (', &
             S1(1:idleng),' ',R1(1:idleng),' ',A1(1:idleng),')(', &
             S2(1:idleng),' ',R2(1:idleng),' ',A2(1:idleng),')(', &
             S3(1:idleng),' ',R3(1:idleng),' ',A3(1:idleng),')(', &
             S4(1:idleng),' ',R4(1:idleng),' ',A4(1:idleng),')'
     ENDIF
  ENDDO

#if KEY_CMAP==1
  DO I=2,NCRTERM
     CONDIT=(I1CT(I).EQ.I1CT(I-1).AND. &
          J1CT(I).EQ.J1CT(I-1).AND. &
          K1CT(I).EQ.K1CT(I-1).AND. &
          L1CT(I).EQ.L1CT(I-1).AND. &
          I2CT(I).EQ.I2CT(I-1).AND. &
          J2CT(I).EQ.J2CT(I-1).AND. &
          K2CT(I).EQ.K2CT(I-1).AND. &
          L2CT(I).EQ.L2CT(I-1))
     IF (CONDIT) THEN
        J=I1CT(I)
        CALL ATOMID(J,S1,R1,N1,A1)
        J=J1CT(I)
        CALL ATOMID(J,S2,R2,N2,A2)
        J=K1CT(I)
        CALL ATOMID(J,S3,R3,N3,A3)
        J=L1CT(I)
        CALL ATOMID(J,S4,R4,N4,A4)
        J=I2CT(I)
        CALL ATOMID(J,S5,R5,N5,A5)
        J=J2CT(I)
        CALL ATOMID(J,S6,R6,N6,A6)
        J=K2CT(I)
        CALL ATOMID(J,S7,R7,N7,A7)
        J=L2CT(I)
        CALL ATOMID(J,S8,R8,N8,A8)

        IF(WRNLEV.GE.2) WRITE(OUTU,9020) &
             ' TESPSF-ERROR: Duplication of cross-term ', &
             I-1,' and ',I,'  IDs: (', &
             S1(1:idleng),' ',R1(1:idleng),' ',A1(1:idleng),')(', &
             S2(1:idleng),' ',R2(1:idleng),' ',A2(1:idleng),')(', &
             S3(1:idleng),' ',R3(1:idleng),' ',A3(1:idleng),')(', &
             S4(1:idleng),' ',R4(1:idleng),' ',A4(1:idleng),')(', &
             S5(1:idleng),' ',R5(1:idleng),' ',A5(1:idleng),')(', &
             S6(1:idleng),' ',R6(1:idleng),' ',A6(1:idleng),')(', &
             S7(1:idleng),' ',R7(1:idleng),' ',A7(1:idleng),')(', &
             S8(1:idleng),' ',R8(1:idleng),' ',A8(1:idleng),')'
     ENDIF
  ENDDO
#endif 

  !
  DO I=2,NDON
     CONDIT=(IDON(I).EQ.IDON(I-1).AND. &
          IHD1(I).EQ.IHD1(I-1))
     IF (CONDIT) THEN
        J=IDON(I)
        CALL ATOMID(J,S1,R1,N1,A1)
        J=IHD1(I)
        CALL ATOMID(J,S2,R2,N2,A2)
        IF(WRNLEV.GE.2) WRITE(OUTU,9020) &
             ' TESPSF-ERROR: Duplication of donor ', &
             I-1,' and ',I,'  IDs: (', &
             S1(1:idleng),' ',R1(1:idleng),' ',A1(1:idleng),')(', &
             S2(1:idleng),' ',R2(1:idleng),' ',A2(1:idleng),')'
     ENDIF
  ENDDO
  !
  DO I=2,NACC
     CONDIT=(IACC(I).EQ.IACC(I-1).AND. &
          IAC1(I).EQ.IAC1(I-1))
     IF (CONDIT) THEN
        J=IACC(I)
        CALL ATOMID(J,S1,R1,N1,A1)
        J=IAC1(I)
        CALL ATOMID(J,S2,R2,N2,A2)
        IF(WRNLEV.GE.2) WRITE(OUTU,9020) &
             ' TESPSF-ERROR: Duplication of acceptor ', &
             I-1,' and ',I,'    IDs: (', &
             S1(1:idleng),' ',R1(1:idleng),' ',A1(1:idleng),')(', &
             S2(1:idleng),' ',R2(1:idleng),' ',A2(1:idleng),')'
     ENDIF
  ENDDO
  !
9010 FORMAT(A,I6,A)
9020 FORMAT(A,I6,A,I6,/,12(A,A),A)
  RETURN
END SUBROUTINE TESPSF
!C##ENDIF
#if KEY_PARALLEL==1

SUBROUTINE TESTPAR8(ARRAY,X,XR,N)
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use parallel
  use psf
  use stream
  implicit none
  !
  CHARACTER(len=*) ARRAY
  INTEGER N
  real(chm_real) X(N),XR(N)
  !
  ! input/output
  !
  INTEGER I
  real(chm_real) NORM,ERRX
  !
  IF(NUMNOD.LE.1) THEN
     IF(PRNLEV.GT.2) WRITE(OUTU,88) ARRAY,0.0
     RETURN
  ENDIF
  !
  DO I=1,N
     XR(I)=X(I)
  ENDDO
  !
  CALL PSYNC()
  CALL GCOMB(XR,N)
  !
  NORM=1.0/NUMNOD
  ERRX=0.0
  DO I=1,N
     ERRX=ERRX+ABS(X(I)-XR(I)*NORM)
  ENDDO
  DO I=1,NUMNOD
     XR(I)=0.0
  ENDDO
  XR(MYNODP)=ERRX
  CALL GCOMB(XR,NUMNOD)
  ERRX=0.0
  DO I=1,NUMNOD
     ERRX=ERRX+XR(I)
  ENDDO
  IF(PRNLEV.GT.2) WRITE(OUTU,88) ARRAY,ERRX
88 FORMAT('TEST PARALLEL:: The total error for "',A,'" is:',F20.10)
  !
  CALL PSND8(X,N)
  CALL PSYNC()
  !
  RETURN
END SUBROUTINE TESTPAR8

SUBROUTINE TESTCOMM(N,M,X,Y,QTIME,QSYNC)
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use parallel
  ! input/output
  use psf
  use stream
  use machutil,only:eclock
  implicit none
  !
  INTEGER N,M
  real(chm_real) X(N),Y(N)
  LOGICAL QTIME,QSYNC
  !
  !
  INTEGER I,J
  real(chm_real) TIM1,TIM2,TIM3,TIM4,TIM5
  !
  IF(PRNLEV.GE.2) WRITE(OUTU,'(A,I4,A,I5,A)') &
       'Testing parallel communication using',M,'packets with', &
       N,' 8-byte words'
  !
  ! Set up fake IPARPT array
  DO I=0,NUMNOD
     IPARPT(I)=I*N/NUMNOD
  ENDDO
  IF(QTIME)TIM1=ECLOCK()
  DO I=1,M
     CALL PSND8(X,N)
  ENDDO
  IF(QTIME)THEN
     TIM1=ECLOCK()-TIM1
     TIM2=ECLOCK()
  ENDIF
  ! General communications scheme - broadcast
  DO I=1,M
     CALL AGVDGB(X,IPARPT,NUMNOD,MYNOD,IPPMAP)
  ENDDO
  IF(QTIME)THEN
     TIM2=ECLOCK()-TIM2
     TIM3=ECLOCK()
  ENDIF
  ! General communications scheme - sum
  DO I=1,M
     CALL AGVDGS(X,IPARPT,Y,NUMNOD,MYNOD,IPPMAP)
  ENDDO
  IF(QTIME)THEN
     TIM3=ECLOCK()-TIM3
  ENDIF
  TIM4=0.0
  TIM5=0.0
  IF(2**NODDIM .EQ. NUMNOD)THEN
     ! Recursive doubling  communications scheme - broadcast
     IF(QTIME)THEN
        TIM4=ECLOCK()
     ENDIF
     DO I=1,M
        CALL AVDGBR(X,IPARPT,NODDIM,MYNOD,IPPMAP)
     ENDDO
     IF(QTIME)THEN
        TIM4=ECLOCK()-TIM4
        TIM5=ECLOCK()
     ENDIF
     ! Recursive doubling communications scheme - sum
     DO I=1,M
        CALL AVDGSUM(X,IPARPT,Y,NODDIM,MYNOD,IPPMAP)
     ENDDO
     IF(QTIME)THEN
        TIM5=ECLOCK()-TIM5
     ENDIF
  ENDIF
  IF(PRNLEV.GE.2)THEN
     WRITE(OUTU,100) NUMNOD,M,N*8,TIM1,TIM2,TIM3,TIM4,TIM5
100  FORMAT(' COMM> NODES NUM BYTES', &
          '     PSND8    AGVDGB    AGVDGS    AVDGBR   AVDGSUM',/, &
          ' COMM:',I5,I5,I6,5F10.5)
  ENDIF
  RETURN
END SUBROUTINE TESTCOMM
#endif /*  PARALLEL*/

SUBROUTINE TESTPARA(COMLYN,COMLEN)
  !
  ! This routine tests the parameter tables.
  !
  ! Arnaud. Blondel - Under devloppement. 1994
  ! ------------------------------------------
  !
  use chm_kinds
  use dimens_fcm
  ! param.f90(s)
  use param
  use vangle_mm
  use cnst_fcm
  !
  use stream
  use string
  use exfunc

  implicit none
  ! . Passed variables.
  CHARACTER(len=*) COMLYN
  INTEGER   COMLEN
  ! . Local variables.
  INTEGER I,J
  CHARACTER(len=4) WRD
  !
  WRD=NEXTA4(COMLYN,COMLEN)
  !
  IF(WRD.EQ.'TRIG') THEN
     WRD=NEXTA4(COMLYN,COMLEN)
     IF(WRD.EQ.'DIHE') THEN
        WRITE(POUTU,*) ' TESTPARA> Writting the Dihedral', &
             ' Parameters tables'
        WRITE(POUTU,*) 'NPHI Constant  Periode    Angle', &
             '           Cos            Sin'
        DO I=1,NCP
           WRITE(POUTU,17) &
                I,CPC(I),CPD(I),CPB(I),CPCOS(I),CPSIN(I)
17         FORMAT(I4,F9.5,I6,5X,3G15.8,/, &
                '--------------------------------------')
        ENDDO
        !
     ELSE IF(WRD.EQ.'CDIH') THEN
        WRITE(POUTU,*) ' TESTPARA> Writting the Dihedral', &
             ' Constraints Parameters tables'
        WRITE(POUTU,*) 'NPHI Constant  Periode    Angle', &
             '           Width'
        DO I=1,NCSPHI
           WRITE(POUTU,17) &
                I,CCSC(I),CCSD(I),CCSB(I),CCSW(I)
        ENDDO
        !
     ELSE IF(WRD.EQ.'VDIH') THEN
        WRITE(POUTU,*) ' TESTPARA> Writting the Dihedral', &
             'Vector Parameters tables'
        CALL PRTVPHI(NPHIV,VIND, &
             VCPC,VCPD,VCPB, &
             VCPCOS,VCPSIN)

     ELSE IF(WRD.EQ.'IMPR') THEN
        WRITE(POUTU,*) ' TESTPARA> Writting the Improper', &
             ' Parameters tables'
        WRITE(POUTU,*) 'NIMP Constant  Periode    Angle', &
             '           Cos            Sin'
        DO I=1,NCI
           WRITE(POUTU,17) &
                I,CIC(I),CID(I),CIB(I),CICOS(I),CISIN(I)
        ENDDO
        !
     ENDIF
  ENDIF
  RETURN
END SUBROUTINE TESTPARA

SUBROUTINE PRTVPHI(NPHIV,VIND,VCPC,VCPD, &
     VCPB,VCPCOS,VCPSIN)
  !
  ! This routine tests prints the vector parameter tables.
  ! ------------------------------------------------------
  !
  !   A.Blondel 1994.
  !
  use chm_kinds
  use dimens_fcm
  use stream
  use code
  implicit none
  !
  INTEGER VIND(*),VCPD(*),NPHIV
  real(chm_real) VCPC(*),VCPB(*),VCPCOS(*),VCPSIN(*)
  !
  INTEGER I
  !
  WRITE(POUTU,23)
23 FORMAT('     #  IC       K     n          PhiO', &
       '     Cos       Sin',/)
  DO I=1,NPHIV
     WRITE(POUTU,27) I,ICP(VIND(I)),VCPC(I),VCPD(I), &
          VCPB(I),VCPCOS(I),VCPSIN(I)
27   FORMAT(I6,I4,F9.5,I6,5X,3F9.5)
     WRITE(POUTU,*) '---------------------------------------'
  ENDDO
  RETURN
END SUBROUTINE PRTVPHI

end module testch_m

