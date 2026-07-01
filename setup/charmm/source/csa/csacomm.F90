module csacommmod
  use chm_kinds
  implicit none
#if KEY_CSA==1 || KEY_DISTENE==1 /*csa_data*/
  !
  !     CSA, DISTENE work only in parallel
  !
  !     These are the mscale module global data
  !
  !     NSUBS                  Total number of systems
  !     QCSA                   Flag: Are we using this method?
  !     INTERCOM (nsubs)       Intercomunicators for MPI_COMM_SPAWNed processes
  !     MSTLPR   (nsubs)       program names
  !     MSTPROC  (nsubs)       how many processes on each subsytem (parallel/parallel)
  !     MSTLINP  (nsubs)       input scripts for subsystems
  !     MSTLOUT  (nsubs)       output files for susbsystems
  !     MAXREQS                Maximum communication wait requests
  !
  integer, save,public :: mstlpr, mstproc, mstlinp, mstlout
  logical, save, public :: qcsa
  integer(chm_int4),allocatable,dimension(:),save,public :: intercomm
  INTEGER, save, public :: NSUBS
  INTEGER, PARAMETER :: MAXREQS=15
  !
#endif /* (csa_data)*/
  !
CONTAINS
  !
#if KEY_CSA==0 && KEY_DISTENE==0 /*main_csa*/
  SUBROUTINE MASTERDSTR(COMLYN,COMLEN)
    CHARACTER(len=*) COMLYN
    INTEGER   COMLEN
    CALL WRNDIE(-1,'<CSA>','CSA code not compiled.')
    RETURN
  END SUBROUTINE MASTERDSTR
#else /* (main_csa)*/
  SUBROUTINE MASTERDSTR(COMLYN,COMLEN)
    !----------------------------------------------------------------------
    !
    ! Setup for distribution of compute processes
    !
    ! Written in October 2007 by Milan Hodoscek
    ! Code design by Bernie Brooks
    !
    !
  use exfunc
  use dimens_fcm
  use number
  use psf
  use coord
  use stream
  use string
  use parallel
    !
#if KEY_PARALLEL==1
  use mpi                   
#endif
    !
    CHARACTER(len=*) COMLYN
    INTEGER   COMLEN
    !
    CHARACTER(len=100) :: MSTPROG,MSTINP, MSTOUT, COMNDS, ARRARG
    CHARACTER(LEN=100), allocatable, dimension(:) :: SARRARG
    CHARACTER(len=5) :: NAMUNIQ
    INTEGER :: I,J,NEWLEN
    CHARACTER(len=1), parameter :: QUOTE = '"'
    !
    INTEGER :: ALLOC_STAT, NCALLS
    !
    !     We need integer*4 here, on -i8 or -fdefault-integer-8 compilations
    !     but no need for #INTEGER8 here :-)
    INTEGER(chm_int4) :: MAXPROCS
    INTEGER(chm_int4) NPROC,MPIINFO4,MPICOMM4,ME4,MPISELF,MPIERRS(1)
    INTEGER(chm_int4) ME,NP,MPIDP,MEROOT,IONE,IERR,MPIVER,MPISUBVER
    !
    real(chm_real) RR
    !
  !
  !     Are we using the correct version of MPI
  !
  CALL MPI_GET_VERSION(MPIVER,MPISUBVER,IERR)
  !C      write(*,*)'MPI_Version = ', MPIVER, MPISUBVER
  IF (MPIVER.LT.2) CALL WRNDIE(-5,'<CSA>', &
       'CSA/DISTENE code needs to be compiled with MPI-2(OpenMPI).')
  !
  QCSA=.TRUE.
  !
  NSUBS=GTRMI(COMLYN,COMLEN,'NSUB',1)
  !
  IF(PRNLEV.GE.2) WRITE(OUTU,'(A,I5)') &
       'MASTer> Number of subsystems =', nsubs
  !
  !
  allocate(sarrarg(5), intercomm(nsubs), stat = alloc_stat )
  !
  if (alloc_stat.ne.0) write(outu,*)'CSA>allocate error'
  !
  !     get the number of processes to use for this system:
  MSTPROC=GTRMI(COMLYN,COMLEN,'NPRO',1)
  !
  !     get the program name:
  CALL GTRMWA(COMLYN,COMLEN,'PROG',4, &
       MSTPROG,100,MSTLPR)
  !
  !     get the input name:
  CALL GTRMWA(COMLYN,COMLEN,'INPU',4, &
       MSTINP,100,MSTLINP)
  !
  !     get the output name:
  CALL GTRMWA(COMLYN,COMLEN,'OUTP',4, &
       MSTOUT,100,MSTLOUT)
  !
  !
  !     Now we can execute the setup defined in the MASTer command:
  !
  MPIINFO4=MPI_INFO_NULL
  !
  !     Here we could have one error per process for checking. Ignore for now!
  MPIERRS(1)=MPI_ERRCODES_IGNORE(1)
  MPISELF=MPI_COMM_SELF
  MPICOMM4=COMM_CHARMM
  ME4=0
  !
  !     We need more ifs for non-specified parameters !!!
  IF((MSTPROG(1:1).EQ.QUOTE).AND. &
       (MSTPROG(MSTLPR:MSTLPR).EQ.QUOTE)) THEN
     COMNDS=MSTPROG(2:MSTLPR-1)
  ELSE
     COMNDS=MSTPROG(1:MSTLPR)
  ENDIF
  SARRARG(1)=' -input '
  IF((MSTINP(1:1).EQ.QUOTE).AND. &
       (MSTINP(MSTLINP:MSTLINP).EQ.QUOTE)) THEN
     SARRARG(2)=MSTINP(2:MSTLINP-1)
  ELSE
     SARRARG(2)=MSTINP(1:MSTLINP)
  ENDIF
  SARRARG(3)=' -output '
  NEWLEN=MSTLOUT
  IF((MSTOUT(1:1).EQ.QUOTE).AND. &
       (MSTOUT(MSTLOUT:MSTLOUT).EQ.QUOTE)) THEN
     ARRARG=MSTOUT(2:MSTLOUT-1)
     NEWLEN=MSTLOUT-2
  ELSE
     ARRARG=MSTOUT(1:MSTLOUT)
  ENDIF
  SARRARG(5)=' '
  MAXPROCS=MSTPROC
  !
  !     Spawn nsubs groups of compute processes
  DO I=1,NSUBS
     !
     !     Output files should have unique names
     WRITE(NAMUNIQ,'(I4.4)')I
     SARRARG(4)=ARRARG(1:NEWLEN)//'_'//NAMUNIQ
     !C         write(*,*)'MASTER>output name = ',sarrarg(4)(1:newlen+5)
     CALL MPI_COMM_SPAWN(COMNDS,SARRARG,MAXPROCS, &
          MPIINFO4,ME4,MPICOMM4,INTERCOMM(I),MPIERRS,IERR)
     !
  ENDDO
  !
  RETURN
END SUBROUTINE MASTERDSTR
!
SUBROUTINE ETRAJ(COMLYN,COMLEN)
  !----------------------------------------------------------------------
  !
  !     This routine reads in the trajectory in the master
  !     script and sends coordinates to the subsystems.
  !     Then it gets the energy from them,
  !
  use exfunc
  use stream
  use string
  use dimens_fcm
  use ctitla
  use energym
  use number
  use parallel
  use psf
  use cvio
  use mpi
  !
  CHARACTER(len=*) COMLYN
  INTEGER   COMLEN
  !
  INTEGER NSTOP,NBEGN,SKIP,ISTEP,ISTATS,NDEGF
  INTEGER NFRAMES,NC,NCR,I,J,K,FLAG,M
  INTEGER NFREAT,NUNIT,FIRSTU,IUNIT,NFILE,NSAVV
  INTEGER IPT,JPT,KPT,MYFLAG
  CHARACTER(LEN=4), parameter ::  HDR1='CORD', HDR2='VELD'
  !
  REAL(chm_real) :: DELTA
  REAL(chm_real), allocatable, dimension(:,:) :: XETERM,XEPROP
  REAL(chm_real), allocatable, dimension(:) :: E,X,Y,Z,DX,DY,DZ,fdim
  REAL(chm_real4), allocatable, dimension(:) :: TEMP
  INTEGER, allocatable, dimension(:) :: FREEAT,PROCMAP
  !
  !     MPI stuff (integer*4 - no need for ##INTEGER8)
  !
  !
  INTEGER(chm_int4), allocatable, dimension(:) :: REQ1
  INTEGER(chm_int4), allocatable, dimension(:,:) :: REQ2
  INTEGER(chm_int4) :: WSTAT(MPI_STATUS_SIZE),JJ,MXX,IERR
  !
  LOGICAL :: QSYNCH
#if KEY_CHEQ==1
  LOGICAL :: QCG                                        
#endif
  !
  QSYNCH = INDXA(COMLYN,COMLEN,'SYNC').GT.0

  CALL TRJSPC(COMLYN,COMLEN,NUNIT,FIRSTU,NBEGN,SKIP,NSTOP)
  !      CALL TRAJIO -> this seems to be not working ???
  !
  !      write(*,*)'ETRAJ>TRJSPC:nstop,nbegn,skip=',nstop,nbegn,skip
  !      write(*,*)'ETRAJ>TRJSPC:nunit,firstu=',nunit,firstu
  !
  NFRAMES=(NSTOP-NBEGN)/SKIP+1
  !
  NC=NFRAMES/NSUBS
  NCR=MOD(NFRAMES,NSUBS)
  !C      write(*,*)'ETRAJ>nframes,nc,nr=',nframes,nc,ncr
  !
  !     ******************************************************
  !     ******************  IMPORTANT NOTE *******************
  !     ******************************************************
  !
  !     Since we are using non-blocking communication the
  !     buffers are subject to race condition problems.
  !     So all the results that come from the subsystems must
  !     be stored in the (1:NSUBS) arrays or arrays of arrays!
  !
  !     See, for example XEPROP, XETERM arrays!
  !
  !     Another issue is that the order of results is not
  !     consecutive. To make it so, we need arrays of the
  !     (1:NFRAMES) size. Then of course we don't need (1:NSUBS)
  !
  !     The above needs to be done only for the non-blocking
  !     part, ie the data coming from the subsystems to the
  !     master!
  !
  !     ******************************************************
  !     ******************  IMPORTANT NOTE *******************
  !     ******************************************************

  !
  ALLOCATE(XETERM(LENENT,NSUBS),XEPROP(LENENP,NSUBS))
  !
  ALLOCATE(X(NATOM),Y(NATOM),Z(NATOM),DX(NATOM),DY(NATOM),DZ(NATOM))
  ALLOCATE(E(NFRAMES),FREEAT(NATOM),TEMP(NATOM),fdim(natom))
  ALLOCATE(REQ1(NSUBS),REQ2(NSUBS,MAXREQS),PROCMAP(NSUBS))
  !
  IPT=0
  JPT=0
  IUNIT=FIRSTU
  ISTATS = 1  ! start the trajectory read
  !
  !
  !                Sending to slaves
  !                IFLAG=MOD(FLAG,10)+1
  !
  !              0 finish the slaves
  !              1 coordinates
  !              2 internal coordinates
  !
  !                Receiving from slaves
  !                JFLAG=FLAG/10+1
  !
  !              1 only energy (always, so we can do non-blocking easily!!!)
  !              2 coordinates
  !              3 forces
  !              4 coordinates+forces
  !              5 ....
  !
  FLAG=11   ! this for sending the coordinates and get the energy back
  ! for more communication increase the value and
  ! extend the program...
  !
  !     This is synchronous code just to check the performance of
  !     parallel stuff. It must be slow so it is not recommended to ever use!
  !     It is here to show the differences with (a)synchronous communications
  !
  IF(QSYNCH) THEN
     !CC  This was working, but is blocking a lot!!
     !     LATER:
     !     get rid of this NC, NFRAMES business
     !     - just check the number of frames in the file(s)...
     !
     DO I = 1, NC
        !
        !     First send all the data to everybody
        !     This has to be modified when we want everybody busy all the time
        !     But that's just the improvement of this I=1,NC loop
        !     For now we have a simple case:
        !      - everybody the same number of atoms
        !      - wait on everybody to finish one cycle and then proceed
        !
        DO J=1, NSUBS
           IPT=IPT+1
           IF(IPT.LE.NFRAMES)THEN
              !C Not working ???               CALL REATRJ(X,Y,Z)
              CALL READCV(X,Y,Z, &
#if KEY_CHEQ==1
                   CG,QCG,                          & 
#endif
                   TEMP,NATOM,FREEAT,NFREAT, &
                   FIRSTU,NUNIT,IUNIT,NFILE, &
                   ISTEP,ISTATS,NDEGF,DELTA, &
                   NBEGN,NSTOP,SKIP,NSAVV,HDR1,HDR2, &
                   TITLEB,NTITLB,.FALSE.,fdim,.TRUE.)
              !               write(*,*)'ETRAJ>ipt,j,i,flg,istats=',ipt,j,i,flag,istats
              CALL SUBSENDC(IPT,J,FLAG,X,Y,Z)
           ENDIF
        ENDDO
        !
        !     Now wait for the results to come
        !     This code will block untill all subsystems finish
        !     Improve: make everybody busy all the time
        !
        DO J=1,NSUBS
           JPT=JPT+1
           IF(JPT.LE.NFRAMES)THEN
              CALL SUBRECES(J,FLAG,XETERM(1,J),XEPROP(1,J))
              E(JPT)=XEPROP(EPOT,J)
              CALL PRINTE(OUTU, XEPROP(1,J), XETERM(1,J),'ETRJ','ENR', &
                   .FALSE.,JPT, ZERO, ZERO, .FALSE.)
           ENDIF
        ENDDO
     ENDDO
     ! '
     GOTO 12
     !
  ENDIF
  !.....................................NEW................
  !
  !     The following block of code deals with the I/O so
  !     only node=0 wants to do it
  !
  IF(MYNOD.GT.0) GOTO 12
  !
  !     First send the data to all available subsystems
  !
  DO J=1, NSUBS
     IF(IPT.LT.NFRAMES)THEN
        IPT=IPT+1
        !C Not working ???               CALL REATRJ(X,Y,Z)
        CALL READCV(X,Y,Z, &
#if KEY_CHEQ==1
             CG,QCG,         & 
#endif
             TEMP,NATOM,FREEAT,NFREAT, &
             FIRSTU,NUNIT,IUNIT,NFILE, &
             ISTEP,ISTATS,NDEGF,DELTA, &
             NBEGN,NSTOP,SKIP,NSAVV,HDR1,HDR2, &
             TITLEB,NTITLB,.FALSE.,fdim,.TRUE.)
        !     write(*,*)'ETRAJ>ipt,j,i,flg,istats=',ipt,j,i,flag,istats
        !C      write(*,*)'ETRAJ-readcv>ipt,j,flag=',ipt,j,flag
        CALL SUBSENDC(IPT,J,FLAG,X,Y,Z)
        !C            write(*,*)'ETRAJ-subsendc>ipt,j,flag=',ipt,j,flag
     ENDIF
  ENDDO
  !
  !     Now post the receive requests for the above subsytems
  !     NOTE that M is the same for everybody!
  !
  DO J=1,NSUBS
     IF(JPT.LT.NFRAMES)THEN
        JPT=JPT+1
        CALL SUBRECE(J,FLAG,XETERM(1,J),XEPROP(1,J),REQ1,REQ2,M)
        !C            write(*,*)'ETRAJ-subrece>jpt,j,flag=',jpt,j,flag
        PROCMAP(J)=JPT
     ENDIF
  ENDDO
  !
  !     Now wait for any of the above to finish and start a new
  !     data on it....
  !
  KPT=0
  MXX=NSUBS
  !
  IF (JPT.EQ.NFRAMES) GOTO 11
  !
  !     The main loop for non-blocking communication
  !     Every group of processors is busy all the time
  !
10 CONTINUE
  !
  CALL MPI_WAITANY(MXX,REQ1,JJ,WSTAT,IERR)
  !C      write(*,*)'ETRAJ-waitany>ipt,jj,mxx=',ipt,jj,mxx
  !
  !     Processor JJ finished, wait for the rest of its receives
  !      
  DO J=2,M
     CALL MPI_WAIT(REQ2(JJ,J),WSTAT,IERR)
     !C         write(*,*)'ETRAJ-wait>jpt,jj,m=',jpt,jj,m
  ENDDO
  !
  !     The data from this processor are ready to process
  !
  E(PROCMAP(JJ))=XEPROP(EPOT,JJ)
  CALL PRINTE(OUTU, XEPROP(1,JJ), XETERM(1,JJ), 'ETRJ', 'ENR', &
       .FALSE.,PROCMAP(JJ), ZERO, ZERO, .FALSE.)
  !
  JPT=JPT+1
  PROCMAP(JJ)=JPT    ! Update the status who is doing what
  !
  !     All data processed!
  !     Let's make this one busy now:
  !
  J=JJ
  IPT=IPT+1
  IF(IPT.LE.NFRAMES)THEN
     CALL READCV(X,Y,Z, &
#if KEY_CHEQ==1
          CG,QCG,            & 
#endif
          TEMP,NATOM,FREEAT,NFREAT, &
          FIRSTU,NUNIT,IUNIT,NFILE, &
          ISTEP,ISTATS,NDEGF,DELTA, &
          NBEGN,NSTOP,SKIP,NSAVV,HDR1,HDR2, &
          TITLEB,NTITLB,.FALSE.,fdim,.TRUE.)
     !C         write(*,*)'ETRAJ-readcv>ipt,j,flag=',ipt,j,flag
     CALL SUBSENDC(IPT,J,FLAG,X,Y,Z)
     !C         write(*,*)'ETRAJ-subsendc>ipt,j,flag=',ipt,j,flag
     CALL SUBRECE(J,FLAG,XETERM(1,J),XEPROP(1,J),REQ1,REQ2,M)
     !C         write(*,*)'ETRAJ-subrece>ipt,j,m=',ipt,j,m
  ENDIF
  !
  !     Are we at the last frame?
  IF(JPT.EQ.NFRAMES) GOTO 11
  !
  !     Goto wait for next
  !
  GOTO 10
  !
11 CONTINUE
  !
  !     Now we have NSUBS request still pending
  !     Do them in the loop one by one!
  !     Also check for cases when NFRAMES < NSUBS
  KPT=NSUBS
  IF(KPT.GT.NFRAMES)KPT=NFRAMES
  !
  MXX=KPT
  DO J = 1, KPT
     CALL MPI_WAITANY(MXX,REQ1,JJ,WSTAT,IERR)
     DO K=2,M
        CALL MPI_WAIT(REQ2(JJ,K),WSTAT,IERR)
     ENDDO
     !     JJ data ready to process
     E(PROCMAP(JJ))=XEPROP(EPOT,JJ)
     CALL PRINTE(OUTU, XEPROP(1,JJ), XETERM(1,JJ), 'ETRJ', 'ENR', &
          .FALSE.,PROCMAP(JJ), ZERO, ZERO, .FALSE.)
     !
     !     Compress REQ1, REQ2, PROCMAP arrays
     !     Not sure if this is needed, but if yes then it comes here:
     !
     !     For now we ignore this case!

     !
  ENDDO
  !
  !     Send everybody that we are finished with this trajectory!
  !
  DO I=1,NSUBS
     MYFLAG=0
     CALL SUBSENDC(IPT,I,MYFLAG,X,Y,Z)
     !C         write(*,*)'ETRAJ-subsendc>ipt,jpt,i,myflag=',ipt,jpt,i,myflag
  ENDDO
  !
12 CONTINUE
  !
  !     The above code executed on the 0-th node only.
  !     If needed this is a good place to broadcast the data
  !     using CALL PSND8(), etc
  !
  DEALLOCATE(X,Y,Z,DX,DY,DZ,XETERM,XEPROP,E,FREEAT,TEMP)
  DEALLOCATE(REQ1,REQ2,PROCMAP,fdim)
  !
  RETURN
END SUBROUTINE ETRAJ
!
SUBROUTINE SUBSENDC(IFRAME,ISUB,FLAG,X,Y,Z)
  !----------------------------------------------------------------------
  !
  !
  use dimens_fcm
  use psf
  use parallel
  use mpi
  !
  !     Input parameters:
  !     IFRAME - current frame at master, send it to the compute node
  !     ISUB   - current subsystem that the data should be send to
  !     FLAG   - controlling flag: what to send or finish
  !
  INTEGER IFRAME,ISUB,FLAG
  real(chm_real) X(*),Y(*),Z(*)
  !
  !
  INTEGER(chm_int4) IERR,REQ,M_STAT(MPI_STATUS_SIZE),MTO,MMTYPE,MLEN,MINT
  INTEGER(chm_int4) :: BUF(3)
  !
  !     Send the flag to I-th set of compute servers
  !
  IF (MYNOD.EQ.0) THEN
     MTO=0
     MMTYPE=1
     MLEN=3
     MINT = MPI_INTEGER
     BUF(1)=FLAG
     BUF(2)=ISUB
     BUF(3)=IFRAME
     CALL MPI_ISEND(BUF,MLEN,MINT,MTO,MMTYPE,INTERCOMM(ISUB), &
          REQ,IERR)
     IF(IERR.NE.MPI_SUCCESS)  &
          CALL WRNDIE(-4,'<SUBSENDC>','MPI ISEND ERROR IN SUBSENDC')
     CALL MPI_WAIT(REQ,M_STAT,IERR)
     IF(IERR.NE.MPI_SUCCESS)  &
          CALL WRNDIE(-4,'<SUBSENDC>','MPI WAIT ERROR IN SUBSENDC')
  ENDIF
  !     
  !     Finish the I-th slave's loop
  !
  IF(FLAG.EQ.0)THEN
     RETURN
  ENDIF
  !
  !      write(*,*)'SUBSENDC>x(1),x(9)=',x(1),x(9)
  !
  !     Send the coordinates to I-th set of compute servers
  !     Does it need to be FLAG protected?
  !     Is there a situation we don't need coordinates?
  !
  IF (MYNOD.EQ.0) THEN
     MTO=0
     MMTYPE=2
     MLEN=NATOM
     MINT = MPI_DOUBLE_PRECISION
     CALL MPI_ISEND(X,MLEN,MINT,MTO,MMTYPE,INTERCOMM(ISUB),REQ,IERR)
     IF(IERR.NE.MPI_SUCCESS)  &
          CALL WRNDIE(-4,'<SUBSENDC>','MPI ISEND ERROR IN SUBSENDC')
     CALL MPI_WAIT(REQ,M_STAT,IERR)
     IF(IERR.NE.MPI_SUCCESS)  &
          CALL WRNDIE(-4,'<SUBSENDC>','MPI WAIT ERROR IN SUBSENDC')
     CALL MPI_ISEND(Y,MLEN,MINT,MTO,MMTYPE,INTERCOMM(ISUB),REQ,IERR)
     IF(IERR.NE.MPI_SUCCESS)  &
          CALL WRNDIE(-4,'<SUBSENDC>','MPI ISEND ERROR IN SUBSENDC')
     CALL MPI_WAIT(REQ,M_STAT,IERR)
     IF(IERR.NE.MPI_SUCCESS)  &
          CALL WRNDIE(-4,'<SUBSENDC>','MPI WAIT ERROR IN SUBSENDC')
     CALL MPI_ISEND(Z,MLEN,MINT,MTO,MMTYPE,INTERCOMM(ISUB),REQ,IERR)
     IF(IERR.NE.MPI_SUCCESS)  &
          CALL WRNDIE(-4,'<SUBSENDC>','MPI ISEND ERROR IN SUBSENDC')
     CALL MPI_WAIT(REQ,M_STAT,IERR)
     IF(IERR.NE.MPI_SUCCESS)  &
          CALL WRNDIE(-4,'<SUBSENDC>','MPI WAIT ERROR IN SUBSENDC')
  ENDIF
  !
  !     Send internal coordinates when mod(flag,10).gt.1 here:
  !
  !
  RETURN
END SUBROUTINE SUBSENDC
!
SUBROUTINE SUBRECE(I,FLAG,XETERM,XEPROP,REQ1,REQ2,M)
  !----------------------------------------------------------------------
  !
  !
  use energym
  use parallel
  use mpi
  !
  INTEGER I,M,FLAG
  real(chm_real) XEPROP(*),XETERM(*)
  !
  !
  !     MPI stuff (integer*4 - no need for ##INTEGER8)
  !
  INTEGER(CHM_INT4) :: BUF,MLEN,MINT,MFROM,MMTYPE,MCOMM,REQ,IERR
  INTEGER(CHM_INT4) :: M_STAT(MPI_STATUS_SIZE),REQ1(*),REQ2(NSUBS,*)
  !
  !     This code is good only for I/O stuff so there should be
  !     no global communication inside! It only posts the receive.
  !     The waiting code is in the calling routine.
  !
  IF (MYNOD.EQ.0) THEN
     !
     MFROM=0
     MMTYPE=3
     MLEN=LENENT
     MINT=MPI_DOUBLE_PRECISION
     !
     CALL MPI_IRECV(XETERM,MLEN,MINT,MFROM,MMTYPE, &
          INTERCOMM(I),REQ,IERR)
     IF(IERR.NE.MPI_SUCCESS)  &
          CALL WRNDIE(-4,'<SUBRECE>','MPI IRECV ERROR IN SUBRECE')
     M=1
     REQ1(I)=REQ
     REQ2(I,M)=REQ
     MLEN=LENENP
     MMTYPE=4
     CALL MPI_IRECV(XEPROP,MLEN,MINT,MFROM,MMTYPE, &
          INTERCOMM(I),REQ,IERR)
     IF(IERR.NE.MPI_SUCCESS)  &
          CALL WRNDIE(-4,'<SUBRECE>','MPI IRECV ERROR IN SUBRECE')
     M=M+1
     IF(M.GT.MAXREQS) &
          CALL WRNDIE(-4,'<SUBRECE>','Too many pending requests.')
     REQ2(I,M)=REQ
     !
  ENDIF
  !
  RETURN
END SUBROUTINE SUBRECE
!
!
SUBROUTINE SUBRECES(I,FLAG,XETERM,XEPROP)
  !----------------------------------------------------------------------
  use energym
  use parallel
  use mpi

  INTEGER I,FLAG
  real(chm_real) XEPROP(*),XETERM(*)
  !
  !     MPI stuff (integer*4 - no need for ##INTEGER8)
  !
  INTEGER(CHM_INT4) :: BUF,MLEN,MINT,MFROM,MMTYPE,MCOMM,REQ,IERR
  INTEGER(CHM_INT4) :: M_STAT(MPI_STATUS_SIZE)
  !
  IF (MYNOD.EQ.0) THEN
     !
     MFROM=0
     MMTYPE=3
     MLEN=LENENT
     MINT=MPI_DOUBLE_PRECISION
     !
     CALL MPI_IRECV(XETERM,MLEN,MINT,MFROM,MMTYPE, &
          INTERCOMM(I),REQ,IERR)
     IF(IERR.NE.MPI_SUCCESS)  &
          CALL WRNDIE(-4,'<SUBRECE>','MPI IRECV ERROR IN SUBRECE')
     CALL MPI_WAIT(REQ,M_STAT,IERR)
     IF(IERR.NE.MPI_SUCCESS)  &
          CALL WRNDIE(-4,'<SUBRECE>','MPI WAIT ERROR IN SUBRECE')
     MLEN=LENENP
     MMTYPE=4
     CALL MPI_IRECV(XEPROP,MLEN,MINT,MFROM,MMTYPE, &
          INTERCOMM(I),REQ,IERR)
     IF(IERR.NE.MPI_SUCCESS)  &
          CALL WRNDIE(-4,'<SUBRECE>','MPI IRECV ERROR IN SUBRECE')
     CALL MPI_WAIT(REQ,M_STAT,IERR)
     IF(IERR.NE.MPI_SUCCESS)  &
          CALL WRNDIE(-4,'<SUBRECE>','MPI WAIT ERROR IN SUBRECE')
     !
     !CC      write(*,*)'SUBRECE>I,Energy=',I,E
     !
  ENDIF
  !     broadcast this stuff!
  !C      CALL PSND8(E,1)
  !
  RETURN
END SUBROUTINE SUBRECES
!
SUBROUTINE CALCRECE(COMLYN,COMLEN)
  !----------------------------------------------------------------------
  !
  !     This is called when RECEive command is specified on
  !     the slave nodes. It gets the data according to the
  !     FLAG specification (FLAG is a global variable in this module)
  !

  use exfunc
  use stream
  use dimens_fcm
  use psf
  use coord
  use cmdpar
  use parallel
  use string
  !
  use mpi
  !
  CHARACTER(len=*) COMLYN
  INTEGER   COMLEN
  !
  INTEGER I,WD2LEN,WD3LEN,FLAG,ISUB,IFRAME,IPAR,LL
  real(chm_real) E
  LOGICAL :: QIFDONE, OK, EOF
  INTEGER, PARAMETER :: WDMAX=50, LFRAM=10
  CHARACTER(LEN=WDMAX) :: WRD2,WRD3
  CHARACTER(LEN=LFRAM) :: SFRAM
  CHARACTER(LEN=4) :: WRD
  !
  !     MPI stuff (integer*4 - no need for ##INTEGER8)
  !
  INTEGER(CHM_INT4) :: MLEN,MINT,MFROM,MMTYPE,MCOMM,REQ,IERR
  INTEGER(CHM_INT4) :: M_STAT(MPI_STATUS_SIZE), BUF(3)
  !
  !     The original sintax is:
  !
  !     RECEive [IFDONE word] repeat(flags)
  !
  !     how to deal with these flags??? NOT YET...
  !
  !
  !     First get the flag from master controling which
  !     data to get from it.
  !     This is performed only on the first processor
  !
  IF (MYNOD.EQ.0) THEN
     CALL MPI_COMM_GET_PARENT(MCOMM,IERR)
     !
     MFROM=0
     MMTYPE=1
     MLEN=3
     MINT=MPI_INTEGER
     !
     CALL MPI_IRECV(BUF,MLEN,MINT,MFROM,MMTYPE,MCOMM,REQ,IERR)
     IF(IERR.NE.MPI_SUCCESS)  &
          CALL WRNDIE(-4,'<CALCRECE>','MPI IRECV ERROR IN CALCRECE')
     CALL MPI_WAIT(REQ,M_STAT,IERR)
     IF(IERR.NE.MPI_SUCCESS)  &
          CALL WRNDIE(-4,'<CALCRECE>','MPI WAIT ERROR IN CALCRECE')
     !
     FLAG=BUF(1)
     ISUB=BUF(2)
     IFRAME=BUF(3)
  ENDIF
  !     broadcast this stuff! Not sure if all are needed
  CALL PSND4(FLAG,1)
  CALL PSND4(ISUB,1)
  CALL PSND4(IFRAME,1)
  !      write(*,*)'CALCRECE>flag=',flag
  !
  !     Check for FLAG=0
  !
  IF(FLAG.EQ.0) GOTO 200
  !
  !     Get the coordinates (FLAG??)
  !
  IF (MYNOD.EQ.0) THEN
     CALL MPI_COMM_GET_PARENT(MCOMM,IERR)
     !
     MFROM=0
     MMTYPE=2
     MLEN=NATOM
     MINT=MPI_DOUBLE_PRECISION
     !
     CALL MPI_IRECV(X,MLEN,MINT,MFROM,MMTYPE,MCOMM,REQ,IERR)
     IF(IERR.NE.MPI_SUCCESS)  &
          CALL WRNDIE(-4,'<CALCRECE>','MPI IRECV ERROR IN CALCRECE')
     CALL MPI_WAIT(REQ,M_STAT,IERR)
     IF(IERR.NE.MPI_SUCCESS)  &
          CALL WRNDIE(-4,'<CALCRECE>','MPI WAIT ERROR IN CALCRECE')
     CALL MPI_IRECV(Y,MLEN,MINT,MFROM,MMTYPE,MCOMM,REQ,IERR)
     IF(IERR.NE.MPI_SUCCESS)  &
          CALL WRNDIE(-4,'<CALCRECE>','MPI IRECV ERROR IN CALCRECE')
     CALL MPI_WAIT(REQ,M_STAT,IERR)
     IF(IERR.NE.MPI_SUCCESS)  &
          CALL WRNDIE(-4,'<CALCRECE>','MPI WAIT ERROR IN CALCRECE')
     CALL MPI_IRECV(Z,MLEN,MINT,MFROM,MMTYPE,MCOMM,REQ,IERR)
     IF(IERR.NE.MPI_SUCCESS)  &
          CALL WRNDIE(-4,'<CALCRECE>','MPI IRECV ERROR IN CALCRECE')
     CALL MPI_WAIT(REQ,M_STAT,IERR)
     IF(IERR.NE.MPI_SUCCESS)  &
          CALL WRNDIE(-4,'<CALCRECE>','MPI WAIT ERROR IN CALCRECE')
     !
  ENDIF
  !     broadcast this stuff!
  CALL PSND8(X,NATOM)
  CALL PSND8(Y,NATOM)
  CALL PSND8(Z,NATOM)
  !
  !      write(*,*)'CALCRECE>x(1),x(9)=',x(1),x(9)

  !
  !     Parse the repeat(flags)....
  !
  !     We always get the following from master even if the
  !     input script is not using it. Put them in variables
  !
  !     1. frame number       ( @FRAME )
  !     2. subsystem number   ( @SUBSYS )
  !
  WRITE(SFRAM,'(I10)')IFRAME
  LL=LFRAM
  CALL TRIMA(SFRAM,LL)
  IPAR=PARINS('FRAME',5,SFRAM,LL)
  WRITE(SFRAM,'(I10)')ISUB
  LL=LFRAM
  CALL TRIMA(SFRAM,LL)
  IPAR=PARINS('SUBSYS',6,SFRAM,LL)
  !
  !     This code has to be the last thing in parsing the
  !     RECEive command because at the end comlyn is overwritten
  !
  !     If IFDONE keyword specified we need to check also
  !     for FLAG=0 condition. FLAG is received by MPI
  !     from MASTER
  !
200 CONTINUE
  !
  QIFDONE=(INDXA(COMLYN,COMLEN,'IFDO').GT.0)
  IF (QIFDONE) THEN
     IF(FLAG.EQ.0) THEN
        CALL NEXTWD(COMLYN,COMLEN,WRD2,WDMAX,WD2LEN)
        IF(IOLEV.GT.0) THEN
           REWIND ISTRM
           OK=.FALSE.
           EOF=.FALSE.
111        CONTINUE
           READ(ISTRM,'(A)',ERR=945) COMLYN(1:80)
           COMLEN=80
           CALL CNVTUC(COMLYN,COMLEN)
           WRD=NEXTA4(COMLYN,COMLEN)
           IF (WRD.EQ.'LABE') THEN
              CALL NEXTWD(COMLYN,COMLEN,WRD3,WDMAX,WD3LEN)
              OK=EQST(WRD2,WD2LEN,WRD3,WD3LEN)
           ENDIF
           IF (.NOT.(OK.OR.EOF)) GOTO 111
           !
           IF (.NOT.EOF) GOTO 845
945        CALL WRNDIE(-2,'<RECEive>','Unable to find IFDONE label')
845        CONTINUE
        ENDIF
     ELSE
        CALL NEXTWD(COMLYN,COMLEN,WRD2,WDMAX,WD2LEN)
     ENDIF
  ENDIF
  !
  RETURN
END SUBROUTINE CALCRECE
!
SUBROUTINE CALCTRAN(COMLYN,COMLEN)
  !----------------------------------------------------------------------
  !
  !     This is called when TRANsmit command is specified on
  !     the slave nodes. It gets the data according to the
  !     FLAG specification (FLAG is a global variable in this module)
  !
  !
  use exfunc
  use number
  use stream
  use dimens_fcm
  use psf
  use coord
  use energym
  use parallel
  !
  use mpi
  !
  CHARACTER(len=*) COMLYN
  INTEGER   COMLEN
  !
  INTEGER I
  real(chm_real) E
  !
  !     MPI stuff (integer*4 - no need for ##INTEGER8)
  !
  INTEGER(CHM_INT4) :: MLEN,MINT,MTO,MMTYPE,MCOMM,REQ,IERR
  INTEGER(CHM_INT4) :: M_STAT(MPI_STATUS_SIZE)
  !
  !     Nothing to parse yet
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !
  !     Send the energy
  !
  CALL MPI_COMM_GET_PARENT(MCOMM,IERR)
  !C      call mpi_comm_remote_size(mcomm,mlen,ierr)
  !C      write(*,*)'CALCTRAN>remote_size=',mlen
  !C      call mpi_comm_rank(mcomm,mlen,ierr)
  !C      write(*,*)'CALCTRAN>intercom_rank=',mlen
  !C      call mpi_comm_size(mcomm,mlen,ierr)
  !C      write(*,*)'CALCTRAN>intercomm_size=',mlen
  !
  IF (MYNOD.EQ.0) THEN
     !
     MTO=0
     !         MTO=MPI_ROOT
     MMTYPE=3
     MLEN=LENENT
     MINT=MPI_DOUBLE_PRECISION
     !
     CALL MPI_ISEND(ETERM,MLEN,MINT,MTO,MMTYPE,MCOMM,REQ,IERR)
     IF(IERR.NE.MPI_SUCCESS)  &
          CALL WRNDIE(-4,'<CALCTRAN>','MPI ISEND ERROR IN CALCTRAN')
     CALL MPI_WAIT(REQ,M_STAT,IERR)
     IF(IERR.NE.MPI_SUCCESS)  &
          CALL WRNDIE(-4,'<CALCTRAN>','MPI WAIT ERROR IN CALCTRAN')
     MLEN=LENENP
     MMTYPE=4
     CALL MPI_ISEND(EPROP,MLEN,MINT,MTO,MMTYPE,MCOMM,REQ,IERR)
     IF(IERR.NE.MPI_SUCCESS)  &
          CALL WRNDIE(-4,'<CALCTRAN>','MPI ISEND ERROR IN CALCTRAN')
     CALL MPI_WAIT(REQ,M_STAT,IERR)
     IF(IERR.NE.MPI_SUCCESS)  &
          CALL WRNDIE(-4,'<CALCTRAN>','MPI WAIT ERROR IN CALCTRAN')
     !
     !         CALL PRINTE(OUTU, EPROP, ETERM, 'TRAN', 'ENR',
     !     $        .FALSE.,0, ZERO, ZERO, .TRUE.)
  ENDIF
  !
  !      write(*,*)'CALCTRAN>e=',e
  !
  RETURN
END SUBROUTINE CALCTRAN
!
#endif /* (main_csa)*/
!
END MODULE CSACOMMMOD

