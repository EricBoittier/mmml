MODULE MSCALEMOD

  use chm_kinds
  use chm_types
  use bases_fcm
  implicit none

#if KEY_MSCALE==1  /* mscale_data */
  !     These are the mscale module global data
  !
  !     NSUBS                  Total number of systems
  !     QMSCALE                Flag: Are we using this method?
  !     KEYNAME  (20,nsubs)    Array to store the names of the subsystems (for EDS)
  !     MSYSEL   (natom,nsubs) Array to store subsystem selections
  !     INTERCOM (nsubs)       Intercomunicators for MPI_COMM_SPAWNed processes
  !     MSCLPR   (nsubs)       program names
  !     MSCLPROC (nsubs)       how many processes on each subsystem (parallel/parallel)
  !     MSCLFPROC(nsubs)       how many processes on each subsystem - trough SYSTEM()
  !     MSCLINP  (nsubs)       input scripts for subsystems
  !     MSCLOUT  (nsubs)       output files for subsystems
  !     FMSCL    (nsubs)       Scale factors
  !     QMSCLAM  (nsubs)       Flag: replace FMSCL by LAMBDA
  !     QMSCMLAM (nsubs)       Flag: replace FMSCL by 1-LAMBDA
  !     NSUBSAT  (nsubs)       Number of atoms in each selection
  !     INTERCOMM(nsubs)       Communicators between main and servers
  !     REPDCOMM (nsubs,nrepdstr) communicators for REPDSTR
  !
  !
  integer, parameter :: nmslen = 20 ! max subsystem keyname length

  character(len=nmslen), allocatable, dimension(:), save, public :: keyname
  integer, allocatable, dimension(:,:),save,public :: msysel
  integer, allocatable, dimension(:),save,public :: nsubsat,msclpr, &
       msclproc,msclfproc,msclinp,msclout
  logical, save, public :: qmscale, qatomsub, qcrysub, qchrgsub
  integer(chm_int4),allocatable,dimension(:),save,public :: intercomm
  logical, allocatable, dimension(:),save,public :: qmsclam, &
       qmscmlam, qatom, qcryst, qfchrg
  REAL(CHM_REAL), allocatable, dimension(:), save, public :: FMSCL
  INTEGER, save, public :: NSUBS
  LOGICAL, allocatable, dimension(:),save,public :: qmsclfinite
  LOGICAL, save, public :: subfinite, qtrqsub, qsubsdbg, qmscmain
  LOGICAL, save, public :: qmscunin,qmscunou
  real(chm_real), allocatable, dimension(:), save, public ::  &
       msclstep
  real(chm_real), save, public :: substep 
  integer, allocatable, dimension(:), save :: ntrqsub  
  logical, allocatable, dimension(:), save :: qtorquea  
  integer, save                            :: upen     ! write subsys PE to this file
  logical, save                            :: qredupen
  integer(chm_int4),allocatable,dimension(:,:) :: repdcomm
  integer,allocatable,dimension(:) :: repdnsubs
  !
#endif /* mscale_data */
  !
  interface
     integer(c_int32_t) function getpid() bind(C)
       use iso_c_binding
       implicit none
     end function getpid
  end interface

CONTAINS
  
  !
#if KEY_MSCALE==0 /* main_mscale */
  SUBROUTINE MSCALE(MODE,COMLYN,COMLEN)
    CHARACTER COMLYN*(*)
    INTEGER COMLEN,MODE

    CALL WRNDIE(-1,'<MSCALE>','MSCALE code not compiled.')
    RETURN
  END SUBROUTINE MSCALE
#else  /* main_mscale */

  subroutine mscale_iniall()
    qmscale=.false.
    qredupen=.false.
    return
  end subroutine mscale_iniall


  SUBROUTINE MSCALE(MODE,COMLYN,COMLEN)
    !----------------------------------------------------------------------
    !
    ! Main driver for MSCAle command
    !
    ! Written in June 2007 by Milan Hodoscek
    ! Code design by Bernie Brooks
    !
    ! 2nd Derivative support added 05/2008
    ! H. Lee Woodcock, M. Hodoscek, B.R. Brooks
    !
    use exfunc
    use dimens_fcm
    use number
    use psf
    use coord
    use stream
    use select
    use string
#if KEY_REPDSTR==1
    use repdstr,only: qrepdstr,irepdstr,nrepdstr,psetloc,psetglob
#endif
#if KEY_TORQUE==1
    use torque, only: qtorque, trqati, ntrqcenter
#endif
    use memory
    !
#if KEY_PARALLEL==1
    use parallel
    use mpi      
#endif

    implicit none

    CHARACTER COMLYN*(*)
    INTEGER   MODE,COMLEN
    !
    LOGICAL :: EOF,LUSED,QERR,QMSCIGNORE
    CHARACTER(len=4) :: WRD
    CHARACTER(len=nmslen) :: NMSUB
    CHARACTER(len=100) :: ICOMNDS
    CHARACTER(len=100), allocatable, dimension(:) :: MSCPROG, &
         MSCINP, MSCOUT
    CHARACTER(LEN=100), allocatable, dimension(:) :: COMNDS, SARRARG
    CHARACTER(LEN=100), allocatable, dimension(:,:) :: ARRARG
    LOGICAL, ALLOCATABLE, DIMENSION(:) :: QMSAMBER
    INTEGER :: IUNIT,ISYST,I,J
    CHARACTER(len=1), parameter :: QUOTE = '"'
    !
    INTEGER :: ALLOC_STAT, NCALLS, MAXNSUBS
    !
    !     We need integer*4 (=chm_int4) here, on -i8 or -fdefault-integer-8 compilations
    !     but no need for #INTEGER8 here :-)
    INTEGER(chm_int4), allocatable, dimension(:) :: ARRINFO, MAXPROCS
    INTEGER(chm_int4), allocatable, dimension(:) :: ARRERR
    INTEGER(chm_int4) NPROC,MPIINFO4,MPICOMM4,ME4,MPISELF,MPIERRS(1)
    INTEGER(chm_int4) ME,NP,MPIDP,MEROOT,IONE,IERR,MCG,INTERCOMG
    INTEGER(chm_int4) myrepsiz,repdgrp,repdcom,reprank,myrep
    INTEGER(chm_int4) repdnp,mpicomm4l
    INTEGER(chm_int4), allocatable, dimension(:) :: myrepgrp
    !
    real(chm_real) RR
    !
  !
  !
  !     Deal with the case when SERVer command is used
  !     this should be the call to energy controlled from the master
  !     process. For now we have only 
  !
  IF (MODE.EQ.1) THEN
     !
     !     The code inside this IF must be executed only on subsystems
     !
     QMSCMAIN=.FALSE.
     !
     NCALLS=GTRMI(COMLYN,COMLEN,'NCAL',0)
     !
     !     Atom keyword for server command too (just in case):
     !
     QATOMSUB=(INDXA(COMLYN,COMLEN,'ATOM').GT.0)
     !
     !     CRYSTAL keyword for server command too
     !
     QCRYSUB=(INDXA(COMLYN,COMLEN,'CRYS').GT.0)
     !
     !     Do we want to allow the charges of the subsystem to be modified?
     !
     QCHRGSUB=(INDXA(COMLYN,COMLEN,'FCHA').GT.0)
     !
     ! Check if we want extra debugging information from the subsystem
     !
     QSUBSDBG=(INDXA(COMLYN,COMLEN,'DEBU').GT.0)
     !
     !     TORQue keyword should exist for subsys too
     QTRQSUB=(INDXA(COMLYN,COMLEN,'TORQ').GT.0)
#if KEY_TORQUE==0  /* torqe_compiled */
     IF(QTRQSUB) THEN
        CALL WRNDIE(-3,'<MSCALE>','TORQUE code not compiled.')
        QTRQSUB=.false.
     ENDIF
#else /* torqe_compiled */
     IF(QTRQSUB) THEN
        CALL WRNDIE(-3,'<MSCALE>','TORQUE code not developed yet.')
        QTRQSUB=.false.
     ENDIF
#endif  /* torqe_compiled */
     !
     !     Finite difference Hessians:
     SUBFINITE=.FALSE.
     SUBFINITE=(INDXA(COMLYN,COMLEN,'FINI').GT.0)
     SUBSTEP=GTRMF(COMLYN,COMLEN,'STEP',PT05)
     !        write(*,*)'SERVER> FINITE = ',SUBFINITE
     !        write(*,*)'SERVER> STEPSIZE = ',SUBSTEP
     !
     CALL LOOPENE(NCALLS)
     !
     RETURN
  ENDIF

  ! Allow for "special purpose" MSCALE invokations where EMSCALE is not
  ! to be called from ENERGY, but may be called from other points in the
  ! code. The main purpose for this is to do implicit solvent calcs. of
  ! explicit solvent systems for discrete state constant PH MD. In principle
  ! when EMSCALE gets called should be totally context-dependent, but this
  ! scheme may need to become more flexible.
  !
  ! Tim Miller: August, 2012.
  IF(INDXA(COMLYN,COMLEN,'SPEC').GT.0) THEN
     QMSCALE=.FALSE.
  ELSE
     QMSCALE=.TRUE.
  ENDIF

  !
  NSUBS=GTRMI(COMLYN,COMLEN,'NSUB',1)
  UPEN=GTRMI(COMLYN,COMLEN,'UPEN',-1)
  QREDUPEN=INDXA(COMLYN,COMLEN,'REDU') > 0
  ! 
  QMSCMAIN=.TRUE.
  !
  IF(PRNLEV >= 2) WRITE(OUTU,'(A,I5)') &
       'MSCAle> Number of subsystems =', nsubs
  !
  !  NSUBS < 1 is a special case for qrepdstr=.true. Put  QMSCIGNORE=.true.
  !  In this case we can not allocate, etc, but
  !  we need the parser and some of the rest of the code here for such a case.
  QMSCIGNORE = NSUBS < 1
  !
  !     The rest of this subroutine is executed on the main CHARMM
  !
  IF(.NOT.QMSCIGNORE) THEN
  allocate( msysel(natom,nsubs), &
       comnds(nsubs),sarrarg(7),arrarg(7,nsubs),qmsclam(nsubs), &
       nsubsat(nsubs),msclpr(nsubs),msclproc(nsubs),fmscl(nsubs), &
       mscprog(nsubs), mscinp(nsubs),mscout(nsubs),msclinp(nsubs), &
       arrinfo(nsubs), maxprocs(nsubs),msclout(nsubs),qatom(nsubs), &
       intercomm(nsubs),qmscmlam(nsubs),msclfproc(nsubs), &
       qcryst(nsubs), qfchrg(nsubs), QMSCLFINITE(nsubs),MSCLSTEP(nsubs), &
       qtorquea(nsubs),ntrqsub(nsubs),qmsamber(nsubs), &
       keyname(nsubs), stat = alloc_stat )
  !
  if (alloc_stat.ne.0) write(outu,*)'MSCALE>allocate error'
  ENDIF
  QMSCUNIN=INDXA(COMLYN,COMLEN,'NUIN') > 0  ! usefull mostly for REPDSTR
  QMSCUNOU=INDXA(COMLYN,COMLEN,'UNOU') > 0  ! usefull mostly for REPDSTR
  !
  !     Parse the commands within MSCAle block
  !
  EOF=.FALSE.
  ISYST=0
100 CONTINUE
  CALL XTRANE(COMLYN,COMLEN,'MSCALE')
  LUSED=.TRUE.
  DO WHILE (LUSED.AND. .NOT.EOF)
     CALL RDCMND(COMLYN,MXCMSZ,COMLEN,ISTRM,EOF,.TRUE., &
          .TRUE.,'MSCALE> ')
     CALL MISCOM(COMLYN,MXCMSZ,COMLEN,LUSED)
  ENDDO
  WRD=NEXTA4(COMLYN,COMLEN)
  IF (WRD.EQ.'    ') GOTO 100
  IF(WRD.EQ.'SUBS') THEN
     ISYST=ISYST+1
     !     subsystem name:
     CALL NEXTWD(COMLYN,COMLEN,KEYNAME(ISYST),NMSLEN,J)
     !
     !     Finite difference Hessians: 
     !        QMSCLFINITE(ISYST)=(INDXA(COMLYN,COMLEN,'FINI').GT.0)
     !        MSCLSTEP(ISYST)=GTRMF(COMLYN,COMLEN,'STEP',PT05)

     QTORQUEA(ISYST)=(INDXA(COMLYN,COMLEN,'TORQ').GT.0)
#if KEY_TORQUE==1 /* torqe_parse */
     IF(QTORQUEA(ISYST).AND.(.NOT.qtorque)) THEN
        CALL WRNDIE(-4,'<MSCALE>','TORQUES NOT TURNED ON')
     ENDIF
#else /* torqe_parse */
     IF(QTORQUEA(ISYST)) THEN
        CALL WRNDIE(-4,'<MSCALE>','TORQUE CODE NOT COMPILED')
     ENDIF
     QTORQUEA(ISYST)=.FALSE.
#endif /* torqe_parse */

     !     AMBEr keyword
     QMSAMBER(ISYST)=(INDXA(COMLYN,COMLEN,'AMBE').GT.0)
     !     Crystal keyword:
     QCRYST(ISYST)=(INDXA(COMLYN,COMLEN,'CRYS').GT.0)
     !
     ! FCHRg key word for charge transfer
     QFCHRG(ISYST)=(INDXA(COMLYN,COMLEN,'FCHA').GT.0)
     !
     !     Atom keyword:
     QATOM(ISYST)=(INDXA(COMLYN,COMLEN,'ATOM').GT.0)
     !
     !     lambda-1 keyword (for lambda 0 case):
     QMSCMLAM(ISYST)=(INDXA(COMLYN,COMLEN,'MLAM').GT.0)
     !
     !     lambda keyword (for lambda 1 case): 
     QMSCLAM(ISYST)=(INDXA(COMLYN,COMLEN,'LAMB').GT.0)
     !
     !     get the coefficients if no lambda present
     IF(.NOT.QMSCLAM(ISYST)) &
          FMSCL(ISYST)=GTRMF(COMLYN,COMLEN,'COEF',ONE)
     !
     !     get the number of processes to use for this system:
     MSCLPROC(ISYST)=GTRMI(COMLYN,COMLEN,'NPRO',1)
     !
     !     get the forwarding number of processes to use in call system()
     MSCLFPROC(ISYST)=GTRMI(COMLYN,COMLEN,'FNPR',1)
     !
     !     get the program name:
     CALL GTRMWA(COMLYN,COMLEN,'PROG',4, &
          MSCPROG(ISYST),100,MSCLPR(ISYST))
     !
     !     get the input name:
     CALL GTRMWA(COMLYN,COMLEN,'INPU',4, &
          MSCINP(ISYST),100,MSCLINP(ISYST))
     !
     !     get the output name:
     CALL GTRMWA(COMLYN,COMLEN,'OUTP',4, &
          MSCOUT(ISYST),100,MSCLOUT(ISYST))
     !
     !     get the selection for this system:
     CALL SELCTA(COMLYN,COMLEN,MSYSEL(1,ISYST),X,Y,Z,WMAIN,.TRUE.)
     !

     ! populate ntrqsub
     NTRQSUB(ISYST)=0
#if KEY_TORQUE==1 /* tourqe_count */
     IF(QTORQUEA(ISYST)) THEN
        DO I=1,ntrqcenter
           IF(MSYSEL(trqati(I),ISYST) == 1) THEN
              NTRQSUB(ISYST)=NTRQSUB(ISYST)+1
           ENDIF
        ENDDO
        IF(NTRQSUB(ISYST).EQ.0) THEN
           CALL WRNDIE(-2,'<MSCALE>', &
                       'TORQUE used in subsystem but no centers selected')
        ENDIF
        IF(PRNLEV.GT.3) THEN
           WRITE(OUTU,'(A,I4,A,I4)') 'MSCALE> There are ', NTRQSUB(ISYST), &
                                  ' torque centers in subsystem ',ISYST
        ENDIF
     ENDIF
#endif /* tourqe_count */

  ELSE IF (WRD.EQ.'SYSD') THEN
     !
     IF(PRNLEV >= 2) THEN
        DO I=1,ISYST
           write(outu,*)'SYSTEM:',i
           write(outu,*)'MSCALE>program name=',mscprog(i)(1:msclpr(i))
           write(outu,*)'MSCALE>input name=',mscinp(i)(1:msclinp(i))
           write(outu,*)'MSCALE>output name=',mscout(i)(1:msclout(i))
           write(outu,'(''number of processes:'',i6)')MSCLPROC(i)
           write(outu,*)'selection:'
           write(outu,'(40i2)')(MSYSEL(j,i),j=1,natom)
        ENDDO
     ENDIF
     !
  ELSE IF (WRD.EQ.'END ') THEN
     GOTO 200
  ELSE
     WRITE(OUTU,*) 'WRD = ', WRD
     CALL WRNDIE(0,'<MSCALE>','UNRECOGNIZED COMMAND')
  ENDIF
  GOTO 100

  !
200 CONTINUE
  !
  QERR=NSUBS /= ISYST
#if KEY_REPDSTR==1
  IF(QREPDSTR.AND.(NSUBS < 1)) QERR=.FALSE. 
#endif
  IF(QERR)CALL WRNDIE(-5,'<MSCALE>', &
       'Wrong number of subsystems.')
  !
  !      write(*,*)'MSCALE>command processed.'
  !
  !     Now we can execute the setup defined in the MSCAle command:
  !
  MPIINFO4=MPI_INFO_NULL
  !
  !     Here we could have one error per process for checking. Ignore for now!
  MPIERRS(1)=MPI_ERRCODES_IGNORE(1)
  MPISELF=MPI_COMM_SELF
  MPICOMM4=COMM_CHARMM
  ME4=0
  ! me4=0 is OK also for REPDSTR, because we always start from 0 in each group
  !
  ! ***********************************************************
  ! *
  ! * MSCALE with REPDSTR, October 2010 (Milan Hodoscek)
  ! * Further development July 2012
  ! *
  ! ***********************************************************
  !
  ! Plan A:
  ! -------
  ! The easy way to include REPDSTR in this is to
  ! create a separate group for each irepdstr
  !
  ! But there are problems with this code in OpenMPI library.
  !
  ! Plan B: was not worth doing.
  !         There seems to be support in OpenMPI for SPAWN with GROUPS
  !         But only with new versions (Tested 1.6???)...
  !
  ! There is more development for this support in c36a5q/source/mscale
  ! from October 24, 2010 version. Namely there is support for more flexible
  ! setup. I am not sure if now we support that some replicas don't have any
  ! mscale command in it ???
  ! But I think we support different number of subsystems in each. Check if empty
  ! mpi_comm_spawn() works with zero processes???
  ! Also I/O is by hand now, if different replicas want different:
  !  programs, inputs, outputs, etc...
  !
#if KEY_REPDSTR==1
  if(qrepdstr)then
     mpicomm4l=mpicomm4  ! Localize this variable, just in case it is needed later...
     ! --------------------------
     ! create a new group
     ! these are global ranks!
     myrepsiz=numnodg/nrepdstr
     call chmalloc('mscale.src','MSCALE','MYREPGRP',myrepsiz,intg=myrepgrp)
     do i=1,myrepsiz
        myrepgrp(i) = irepdstr*myrepsiz+i-1
     enddo
     call mpi_comm_group(mpicomm4l,mcg,ierr)
     call mpi_group_incl(mcg,myrepsiz,myrepgrp,repdgrp,ierr)
     ! create repdcom to be used in subsequent calls regarding
     ! this replica/group
     call mpi_comm_create(mpicomm4l,repdgrp,repdcom,ierr)
     call mpi_comm_rank(repdcom,reprank,ierr)
     ! do these need to be the same???
!     write(outu,*)'mynod,reprank,repdcom=',mynod,reprank,repdcom  ! are they the same
     ! in the case of repd we use group communicator
     mpicomm4=repdcom
     !=================================================
  endif
#endif

  !     We need more ifs for non-specified parameters !!!
  DO I=1,ISYST
     IF((MSCPROG(I)(1:1).EQ.QUOTE).AND. &
          (MSCPROG(I)(MSCLPR(I):MSCLPR(I)).EQ.QUOTE)) THEN
        COMNDS(I)=MSCPROG(I)(2:MSCLPR(I)-1)
     ELSE
        COMNDS(I)=MSCPROG(I)(1:MSCLPR(I))
     ENDIF
     IF(QMSAMBER(I)) THEN
        ARRARG(1,I)='-i '
        SARRARG(1)='-i '
     ELSE
        ARRARG(1,I)='-input '   ! there should be no space before '-' symbol
        SARRARG(1)='-input '    ! this makes mpich2 happy!
     ENDIF
     IF((MSCINP(I)(1:1).EQ.QUOTE).AND. &
          (MSCINP(I)(MSCLINP(I):MSCLINP(I)).EQ.QUOTE)) THEN
        ARRARG(2,I)=MSCINP(I)(2:MSCLINP(I)-1)
        SARRARG(2)=MSCINP(I)(2:MSCLINP(I)-1)
        IF(LOWER) CALL CNVTLC(SARRARG(2),MSCLINP(I)-2)
     ELSE
        ARRARG(2,I)=MSCINP(I)(1:MSCLINP(I))
        SARRARG(2)=MSCINP(I)(1:MSCLINP(I))
        IF(LOWER) CALL CNVTLC(SARRARG(2),MSCLINP(I))
     ENDIF
     IF(QMSAMBER(I)) THEN
        ARRARG(3,I)='-o '
        SARRARG(3)='-o '
     ELSE
        ARRARG(3,I)='-output '
        SARRARG(3)='-output '
     ENDIF
     IF((MSCOUT(I)(1:1).EQ.QUOTE).AND. &
          (MSCOUT(I)(MSCLOUT(I):MSCLOUT(I)).EQ.QUOTE)) THEN
        ARRARG(4,I)=MSCOUT(I)(2:MSCLOUT(I)-1)
        SARRARG(4)=MSCOUT(I)(2:MSCLOUT(I)-1)
        IF(LOWER) CALL CNVTLC(SARRARG(4),MSCLOUT(I)-2)
     ELSE
        ARRARG(4,I)=MSCOUT(I)(1:MSCLOUT(I))
        SARRARG(4)=MSCOUT(I)(1:MSCLOUT(I))
        IF(LOWER) CALL CNVTLC(SARRARG(4),MSCLOUT(I))
     ENDIF
     IF(QMSAMBER(I)) THEN
        ARRARG(5,I)='-O '
        SARRARG(5)='-O '
        ARRARG(6,I)='-server '
        SARRARG(6)='-server '
        ARRARG(7,I)=' '
        SARRARG(7)=' '
     ELSE  
        ARRARG(5,I)=' '
        SARRARG(5)=' '
        ARRARG(6,I)=' '
        SARRARG(6)=' '
        ARRARG(7,I)=' '
        SARRARG(7)=' '
     ENDIF
     MAXPROCS(I)=MSCLPROC(I)
     !         write(*,*)'MSCALE>nproc=',nproc

     CALL MPI_COMM_SPAWN(COMNDS(I),SARRARG,MAXPROCS(I), &
          MPIINFO4,ME4,MPICOMM4,INTERCOMM(I),MPIERRS,IERR)
     !
  ENDDO
  !
  !     Unfortunately this doesn't work very well. The problem is in
  !     argv[][] order plus it works for one set of arguments :-(  /this is for openmpi/
  !     Maybe in the future there will be a need for communication
  !     between clients. This call makes it EASY!
  !
  !      NPROC=ISYST
  !      MPIINFO4=MPI_INFO_NULL
  !      ME4=0
  !      write(*,*)'MSCALE>nproc=',nproc
  !      CALL MPI_COMM_SPAWN_MULTIPLE(NPROC,COMNDS,ARRARG,MAXPROCS,
  !     $     MPIINFO4,ME4,MPICOMM4,INTERCOMM,ARRERR,IERR)
  !
  !
  !C      call sleep(10)
  !

  if(upen.gt.0) then
     write(upen,'(A1)',advance='no')'#'
     do i=1,nsubs
        write(upen,'(A20,X)',advance='no') keyname(i)
     enddo
     write(upen,'(a)') ' '
  endif

  RETURN
END SUBROUTINE MSCALE
!
SUBROUTINE LOOPENE(NCALLS)
  !----------------------------------------------------------------------
  !
  !     This is the central routine in the MSCAle setup for non master
  !     scripts. It performs:
  !     - communicates control parameter to know when it is finished
  !        currently nothing yet...
  !     - communication of coordinates
  !     - calls UPDECI(), ENERGY() routines
  !     - communcation of energy, forces, virial, etc
  !
  !     SERVer [NCALls <int>] should be the last command in the script.
  !     Maybe very good idea to execute ENERgy before MSCAle command
  !     to specify nonbond options, etc
  !
  !     We basically call PARFIN and STOPCH at the end here...
  !
  !     How do we deal with update? Is it in ENERGY?
  !     We must call UPDECI() here... 
  !     what about ISTEP = we get it from the main process!!!
  !
  use stream
  use dimens_fcm
  use number
  use psf
  use coord
  use deriv
  use energym
  use eutil
  use bases_fcm
  use heurist,only:updeci
  !
#if KEY_PARALLEL==1
  use mpi  
  use parallel
#endif
  !
  LOGICAL QSECD
  INTEGER NCALLS,NAT3,N6
  !
  INTEGER(chm_int4) COMPARENT,COMWORLD,ME,NP,IERR,MPIDP,MEROOT,IONE
  real(chm_real) RR
  REAL(CHM_REAL), allocatable, dimension(:) :: DDX
  !
  INTEGER ISTEP,I,LASTST
  !
  !     Endless loop or NCALLS here:
  !
  QSECD=.FALSE. ! Set to false by default 
  ISTEP=0
  !
100 CONTINUE
  !     Receive the data from the MAIN
  !
  !C      write(*,*)'LOOPENE>before SERVERRECEIVE()'
  CALL SERVERRECEIVE(X,Y,Z,QSECD)
  CALL UPDECI(ISTEP,X,Y,Z,WMAIN,0,(/zero/),(/zero/),(/zero/),(/zero/),(/zero/),(/zero/))
  ISTEP=ISTEP+1
  IF(QSECD) THEN 
     !Set dimension of 2nd derivative matix (upper triangular) 
     NAT3=NATOM*3
     N6=(NAT3*(NAT3+1))/2
     allocate(DDX(N6))

     DO I=1,N6
        DDX(I)=ZERO
     ENDDO

     !        write(*,*)'MSCALE> getting 2nd derivatives... ',SUBFINITE
     IF(SUBFINITE) THEN 
        !           write(*,*)'MSCALE> calling msclvib... ',SUBFINITE, NATOM
        !           write(*,*)'MSCALE> calling msclvib... ',SUBSTEP 
        CALL MSCLVIB(X,Y,Z,NATOM,SUBSTEP,DDX)
        SUBFINITE=.FALSE.
     ELSE
        !           write(*,*)'********* calling normal energy *********'
        CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,1,NAT3,DDX)
     ENDIF

  ELSE 
     CALL GETE(X,Y,Z,X,Y,Z,0)
  ENDIF

  !
  !     Send the data to the main
  !     energy stuff, forces, virial
  !
  !     write(*,*)'number of atoms in subsystem... ',NATOM
  CALL SERVERBROADCAST(NATOM,LENENT,DX,DY,DZ,ETERM,EPRESS,EPROP,QSECD,DDX)
  !
  !     This routine is basically endless loop unless NCALls <int>
  !     is specified in the input script.
  !     It is finished when either some control parameter
  !     comes from the emscale0 or emscale or when MPI_FINALIZE()
  !     or...
  !     MPI_BCAST() is OK because it waits until some data come from
  !     the main group.

  IF(QSECD) DEALLOCATE(DDX)

  !
  IF((NCALLS.NE.0).AND.(ISTEP.GT.NCALLS)) GOTO 900
  !
  GOTO 100
  !
900 CONTINUE
  !     Finish this program (STOPCH??)
  !
  RETURN
END SUBROUTINE LOOPENE
!
!     SUBROUTINE QCHEM(E,DX,DY,DZ,CGX,AMASSX,IACX,NDD1,DD1,QSECD,IUPT,
!    &                 JUPT,PTRSTO)

!
  SUBROUTINE MSCALEFIN
    !----------------------------------------------------------------------
    !
    !     This routine must be called from MAIN only
    !
    use stream
    use repdstr
    !
#if KEY_PARALLEL==1
    use parallel
    use mpi    
#endif
    !
    !
    INTEGER I,J,IPT,NS,ALL_STAT
    INTEGER(chm_int4) MEROOT,MPIDP,N,IERR,MPIINT,IONE,INTCOMM,NCONTROL
    !
    IF(MYNOD.EQ.0)THEN
       MEROOT=MPI_ROOT
    ELSE
       MEROOT=MPI_PROC_NULL
    ENDIF
    MPIDP=MPI_DOUBLE_PRECISION
    MPIINT=MPI_INTEGER
    IONE=1
    !
    DO I=1,NSUBS
!Not yet /Plan B/:       IF(QREPDSTR)INTCOMM=REPDCOMM(I,IREPDSTR+1)
       INTCOMM=INTERCOMM(I)
       ! Perform the job controlling communication:
       ! We want to finish now...
       NCONTROL=-1
       CALL MPI_BCAST(NCONTROL,IONE,MPIINT,MEROOT,INTCOMM,IERR)
    ENDDO
    RETURN
  END SUBROUTINE MSCALEFIN
!
SUBROUTINE MAINBROADCAST(NATOMX,X,Y,Z,QSECD)
  !----------------------------------------------------------------------
  !
  !     MAIN: Broadcast the coordinates to all SUBSYSTEMS
  !
  use stream
  use dimens_fcm
  use rtf,only:atct
  use psf
  use image
  use linkatom
#if KEY_TORQUE==1
  use torque,only:trqrotm,torque_getrot 
#endif
  use linkatom,only:findel
  use parallel
  !
#if KEY_PARALLEL==1
  use mpi      
#endif
  !
  LOGICAL QSECD
  INTEGER NATOMX
  real(chm_real) X(*),Y(*),Z(*)
  !
  INTEGER I,J,K,L,IPT,NS,ALL_STAT,NCONTROL
  INTEGER(CHM_INT4) MEROOT,MPIDP,N,IERR,MPIINT,IONE,M,ISIX,MPILOG
  REAL(CHM_REAL), allocatable, dimension(:) :: XS,YS,ZS,CGS,AN
  CHARACTER(LEN=6) :: ELE ! the same as ATCT
  !
  IF(MYNOD.EQ.0)THEN
     MEROOT=MPI_ROOT
  ELSE
     MEROOT=MPI_PROC_NULL
  ENDIF
  MPIDP=MPI_DOUBLE_PRECISION
  MPIINT=MPI_INTEGER
  MPILOG=MPI_LOGICAL
  IONE=1
  ISIX=6
  !
  ALLOCATE(XS(NATOMX),YS(NATOMX),ZS(NATOMX),AN(NATOMX), &
       CGS(NATOMX),STAT=ALL_STAT)
  !
  !     Fill the communication arrays from the selections
  !
  DO I=1,NSUBS
     ! Perform the job controlling communication:
     ! We always want to continue using slaves here
     NCONTROL=1
     CALL MPI_BCAST(NCONTROL,IONE,MPIINT,MEROOT,INTERCOMM(I),IERR)

     IPT=0
     DO J=1,NATOMX
        IF(MSYSEL(J,I).EQ.1)THEN
           IPT=IPT+1
           XS(IPT)=X(J)
           YS(IPT)=Y(J)
           ZS(IPT)=Z(J)
           CGS(IPT)=CG(J)

           !     Here we fill the Atomic Numbers array
           IF(QATOM(I))THEN
              CALL FINDEL(ATCT(IAC(J)),AMASS(J),J,ELE,AN(IPT), &
                   .FALSE.)
           ENDIF
        ENDIF
     ENDDO
     N=IPT
     NSUBSAT(I)=IPT
     M=MSCLFPROC(I)
     !
     !     First check for QATOM flag and send the atomic data
     !     This is mainly used in non-charmm subsystems
     !
     IF(QATOM(I))THEN
        CALL MPI_BCAST(N,IONE,MPIINT,MEROOT,INTERCOMM(I),IERR)
        CALL MPI_BCAST(M,IONE,MPIINT,MEROOT,INTERCOMM(I),IERR)
        CALL MPI_BCAST(AN,N,MPIDP,MEROOT,INTERCOMM(I),IERR)
     ENDIF
     !
     IF(QCRYST(I))THEN
        CALL MPI_BCAST(XTLTYP,4,MPI_CHARACTER,MEROOT,INTERCOMM(I),IERR)
        CALL MPI_BCAST(XUCELL,ISIX,MPIDP,MEROOT,INTERCOMM(I),IERR)
     ENDIF
     !
     IF(QFCHRG(I))THEN
        CALL MPI_BCAST(CGS,N,MPIDP,MEROOT,INTERCOMM(I),IERR)
     ENDIF
     !
     CALL MPI_BCAST(XS,N,MPIDP,MEROOT,INTERCOMM(I),IERR)
     CALL MPI_BCAST(YS,N,MPIDP,MEROOT,INTERCOMM(I),IERR)
     CALL MPI_BCAST(ZS,N,MPIDP,MEROOT,INTERCOMM(I),IERR)

     ! add sending of finite and step arrays... and QSECD 
     !        write(*,*)'MAINBROADCAST> QSECD = ', QSECD

     !        IF(QSECD) THEN
     CALL MPI_BCAST(QSECD,IONE,MPILOG,MEROOT,INTERCOMM(I),IERR)
     !           write(*,*)'Broadcast Coords --> Subs:  writing QSECD',QSECD
     !        ENDIF 

#if KEY_TORQUE==1 /* tourqe_sent */
     IF(QTORQUEA(I)) THEN
        CALL TORQUE_GETROT(MSYSEL(1,I))
        if(prnlev.ge.7) then
          do k=1,ntrqsub(i)
            if(prnlev.ge.3) write(*,*) 'MAINBROADCAST> rot matrix for atom ', k
            do j=1,3
               write(*,'(3F13.6)') trqrotm(1,j,k), trqrotm(2,j,k),trqrotm(3,j,k)
            enddo
          enddo
        endif
        CALL MPI_BCAST(trqrotm,9*NTRQSUB(I),MPIDP,MEROOT,INTERCOMM(I),IERR)
     ENDIF
#endif /* tourqe_sent */

  ENDDO
  !
  DEALLOCATE(XS,YS,ZS,AN)
  !
  RETURN
END SUBROUTINE MAINBROADCAST
!
SUBROUTINE MAINRECEIVE(INIT,NATOM,LENENT,DX,DY,DZ,ETERM,EPRESS,EPROP,QSECD,DD1,QDYNCALL)
  !----------------------------------------------------------------------
  !
  !     MAIN: Get the results from SUBSYSTEMS
  !
  !     Place the data in the proper vectors in the main
  !     Perform the scaling here
  !
  use stream
  use dimens_fcm
  use number
  use eutil
#if KEY_EDS==1
  use energym,only:viri,vire,virke,lenenv,eds
#else
  use energym,only:viri,vire,virke,lenenv
#endif
  use image,only:dxtl
#if KEY_TORQUE==1
  use torque,only:trqtxyz,torque_makeforce
#endif
#if KEY_EDS==1
  use edsmod 
#endif
  use memory
  use consta
  !
#if KEY_PARALLEL==1
  use parallel
  use mpi          
#endif
  !
  LOGICAL QSECD,QDYNCALL
  INTEGER INIT,NATOM,LENENT
  real(chm_real) DX(*),DY(*),DZ(*),ETERM(*),EPRESS(*),EPROP(*),DD1(*)
  !
  INTEGER I,J,IPT,IST,NAT3,IAT,JPT,KPT,JJ,II
  INTEGER(chm_int4) MEROOT,MPIDP,COMPARENT,IERR,N,N6
  real(chm_real) FACTSYS
  REAL(CHM_REAL), allocatable, dimension(:) :: XSUB,YSUB,ZSUB, &
       ESUB,DDX2
  real(chm_real) :: EPSUBS(lenenv)
  real(chm_real) :: DXTLSUBS(6)
  real(chm_real) :: sviri, svire, svirke
  integer, allocatable, dimension(:) :: ISEL3,IUPT,JUPT

  real(chm_real)                            :: esubstot ! probably wrong MH-12

#if KEY_EDS==1
  real(chm_real),allocatable,dimension(:)   :: edsdx, edsdy, edsdz, scratch
  real(chm_real),allocatable,dimension(:)   :: edseterm
  real(chm_real)                            :: edsoff, edsexptot, edsexp, tmpexp
  real(chm_real)                            :: edsepot, totedsene
  real(chm_real)                            :: maxsafeene
  real(chm_real)                            :: eref,accum,edsviri,edsvire,edsvirke
  real(chm_real),dimension(6)               :: edsdxtl
  real(chm_real),dimension(lenenv)          :: edsepress
  character(len=120)                        :: errmsg

  maxsafeene=log(rbigst)
  IF(QEDS) THEN
     call chmalloc('mscale.src','MAINRECEIVE','edsdx',NATOM,crl=edsdx)
     call chmalloc('mscale.src','MAINRECEIVE','edsdy',NATOM,crl=edsdy)
     call chmalloc('mscale.src','MAINRECEIVE','edsdz',NATOM,crl=edsdz)
     call chmalloc('mscale.src','MAINRECEIVE','edseterm',LENENT,crl=edseterm)

     edsviri=zero
     edsvire=zero
     edsvirke=zero
     do i=1,lenent
        edseterm(i)=zero
     enddo
     do i=1,lenenv
        edsepress(i)=zero
     enddo
     edsdxtl(:)=zero
     do i=1,natom
        edsdx(i)=zero
        edsdy(i)=zero
        edsdz(i)=zero
     enddo
  ENDIF
  edsexptot = 0.0
  eref = 999999999.99
  accum = 0.0
#endif

  !     write(*,*)'MSCALE> number of atoms (NATOM) = ',NATOM

  !
  !
  !     If INIT=-1 pick only MLAMBDA cases, if INIT=-2 pick LAMBDA ones
  !     No scaling in this case (EPERT does the job!)
  !
  MEROOT=0
  MPIDP=MPI_DOUBLE_PRECISION
  !

  DO I = 1, NSUBS
     IF((INIT.GE.0).OR.((INIT.EQ.-1).AND.QMSCMLAM(I)) &
          .OR.((INIT.EQ.-2).AND.QMSCLAM(I))) THEN
        !
        N=NSUBSAT(I)
        NAT3=NATOM*3
        N6=(NAT3*(NAT3+1))/2
        !write(*,*)'MSCALE MAIN> N6 = ',N6
        !write(*,*)'MSCALE QSECD = ', QSECD
        !
        ALLOCATE(XSUB(N),YSUB(N),ZSUB(N),ESUB(LENENT),STAT=IST)
        IF(QSECD) THEN
           ALLOCATE(DDX2(N6))
           DO j=1,N6
              DDX2(j)=ZERO
           ENDDO
        ENDIF
        !
        CALL MPI_BCAST(XSUB,N,MPIDP,MEROOT,INTERCOMM(I),IERR)
        CALL MPI_BCAST(YSUB,N,MPIDP,MEROOT,INTERCOMM(I),IERR)
        CALL MPI_BCAST(ZSUB,N,MPIDP,MEROOT,INTERCOMM(I),IERR)
        N=LENENT
        CALL MPI_BCAST(ESUB,N,MPIDP,MEROOT,INTERCOMM(I),IERR)

        IF(QDYNCALL.OR.(QREDUPEN.EQV..FALSE.)) THEN
           IF(UPEN.GT.0) THEN
              ESUBSTOT=ZERO
              DO J=1,LENENT
                 ESUBSTOT=ESUBSTOT+ESUB(J)
              ENDDO
              WRITE(UPEN,'(X,E20.8,X)',ADVANCE='NO') esubstot
           ENDIF
        ENDIF

        IF(QMSCLAM(I).OR.QMSCMLAM(I))THEN
           FACTSYS=ONE
#if KEY_EDS==1
        ELSE IF(QEDS) THEN
           FACTSYS=ONE
#endif

        ELSE
           FACTSYS=FMSCL(I)
        ENDIF

        IF(QCRYST(I)) THEN
           CALL MPI_BCAST(DXTLSUBS,6,MPIDP,MEROOT,INTERCOMM(I),IERR)
           CALL MPI_BCAST(SVIRI,1,MPIDP,MEROOT,INTERCOMM(I),IERR)
           CALL MPI_BCAST(SVIRE,1,MPIDP,MEROOT,INTERCOMM(I),IERR)
           CALL MPI_BCAST(SVIRKE,1,MPIDP,MEROOT,INTERCOMM(I),IERR)
           CALL MPI_BCAST(EPSUBS,LENENV,MPIDP,MEROOT,INTERCOMM(I),IERR)
        ENDIF

        IF(QSECD) THEN
           CALL MPI_BCAST(DDX2,N6,MPIDP,MEROOT,INTERCOMM(I),IERR)
        ENDIF
#if KEY_TORQUE==1 /* torque_receive */
        IF(QTORQUEA(I)) THEN
           CALL MPI_BCAST(trqtxyz,3*NTRQSUB(I),MPIDP,MEROOT,INTERCOMM(I),IERR)
           if(prnlev.ge.7) then
             do j=1,ntrqsub(i)
               if(prnlev.ge.3) &
                  write(*,*) 'MAINRECEIVE> GOT TRQ on atom ', j, ' = ', trqtxyz(1,j),trqtxyz(2,j),trqtxyz(3,j)
             enddo
           endif
           CALL TORQUE_MAKEFORCE(MSYSEL(1,I))
        ENDIF
#endif /* torque_receive */

        ! ---------------------------------------------
        ! add reading and mapping of Hessian (i.e. DD1) 
        ! ---------------------------------------------

        !           IF(QSECD) THEN
        !              DO J=1,N6
        !                 write(*,*)'DDX22 = ',DDX2(J)
        !              ENDDO
        !           ENDIF 
        !           write(*,*)'MSCALE> Mapping Subsystem Hessian to Full...'
        !
        !     Hessians: 
        !

        IF(QSECD) THEN
           ! Set up second derivative arrays
           allocate(ISEL3(NATOM*3),IUPT(NATOM*3))
           IUPT=1
           CALL FILUPT(IUPT,NAT3)
           !WRITE(*,*)'IUPT(1) = ',IUPT(1)
           !WRITE(*,*)'IUPT(81) = ',IUPT(NAT3),NAT3

           JPT=0
           DO IAT=1,NATOM
              ISEL3(JPT+1)=MSYSEL(IAT,I)       ! tm fix
              ISEL3(JPT+2)=MSYSEL(IAT,I)       ! tm fix
              ISEL3(JPT+3)=MSYSEL(IAT,I)       ! tm fix
              JPT=JPT+3
           ENDDO

           JPT=0
           DO II=1,NAT3
              IF(ISEL3(II).EQ.1)THEN
                 DO JJ=II,NAT3                 ! tm fix
                    IF(ISEL3(JJ).EQ.1) THEN
                       JPT=JPT+1
                       KPT=IUPT(II)+JJ
                       !                           write(*,*)'DD1,KPT,DDX2,JPT,IUPT,II,JJ = ',
                       !    &                      DD1(KPT),KPT,DDX2(JPT),JPT,IUPT(II),
                       !    &                      II,JJ
                       DD1(KPT)=DD1(KPT)+(DDX2(JPT)*FACTSYS)
                    ENDIF
                 ENDDO
              ENDIF
           ENDDO
        ENDIF


#if KEY_EDS==1
        IF(QEDS) THEN
           ! EDS ENERGY

           ESUBSTOT=ZERO
           DO J=1,LENENT
              ! GK 12/2019 Check whether energy contribution is safe ... if it is too high, scale it down 
              ! This is a dirty hack to deal with dummy atoms 
              if(ESUB(J) .gt. 1.0D-2/RPRECI  ) then 
                 CALL WRNDIE(-1,'<MSCALE>','HIGH ENERGY CONTRIBUTION IN SUBSYSTEM')
                 ESUB(J) = 1.0D-2/RPRECI  
              endif ! GK 
              EDSETERM(J)=EDSETERM(J)+ESUB(J)
              ESUBSTOT=ESUBSTOT+ESUB(J)
           ENDDO

           EDSOFF=-19999999.
           DO J=1,NEDS
              IF(EDSLIST(J)%MSKEY.EQ.KEYNAME(I)) THEN
                 EDSOFF=EDSLIST(J)%EOFFSET
              ENDIF
           ENDDO
           IF(EDSOFF.LT.-9999999.) THEN
              CALL WRNDIE(-3,'<MAINRECEIVE>','NO EDS OFFSET FOUND FOR SUBSYSTEM ... SKIPPING')
              CYCLE
           ENDIF
           IF(QEDSDYNOFF) THEN
              TMPEXP=ONE
              EDSEXP=ZERO
              ESUBSTOT=ESUBSTOT-EDSOFF
              IF(I.EQ.1) THEN
                 ! first subsys
                 EREF=ESUBSTOT
                 EDSEXP=ONE
              ELSE
                 IF(ESUBSTOT.LT.EREF) THEN
                    IF(EDSTEMP.GT.0) THEN
                       ! There's a good chance that this scale factor is wrong...
                       TMPEXP=EXP((ESUBSTOT-EREF)/(kboltz*edstemp))
                    ELSE
                       TMPEXP=0.0
                    ENDIF
                    EDSEXPTOT=EDSEXPTOT*TMPEXP
                    EREF=ESUBSTOT
                    EDSEXP=ONE
                 ENDIF
              ENDIF
              IF(edstemp.gt.0.0) THEN
                 EDSEXP=EXP((-1.0/(kboltz*edstemp))*(ESUBSTOT-EREF))
              ENDIF
              EDSEXPTOT=EDSEXPTOT+EDSEXP

              ! EDS FORCES
              IPT=0
              DO J=1,NATOM
                 IF(MSYSEL(J,I).EQ.1)THEN
                    IPT=IPT+1
                    EDSDX(J)=(EDSDX(J)*TMPEXP)+(EDSEXP*XSUB(IPT))
                    EDSDY(J)=(EDSDY(J)*TMPEXP)+(EDSEXP*YSUB(IPT))
                    EDSDZ(J)=(EDSDZ(J)*TMPEXP)+(EDSEXP*ZSUB(IPT))
                 ENDIF
              ENDDO

              IF(QCRYST(I)) THEN
                 DO II=1,LENENV
                    EDSEPRESS(II)=(EDSEPRESS(II)*TMPEXP)+(EDSEXP*EPSUBS(II))
                 ENDDO
                 DO II=1,6
                    EDSDXTL(II)=(EDSDXTL(II)*TMPEXP)+(EDSEXP*DXTLSUBS(II))
                 ENDDO
                 EDSVIRI=(EDSVIRI*TMPEXP)+(SVIRI*EDSEXP)
                 EDSVIRE=(EDSVIRE*TMPEXP)+(SVIRE*EDSEXP)
                 EDSVIRKE=(EDSVIRKE*TMPEXP)+(SVIRKE*EDSEXP)
              ENDIF

           ! User-specified offset (No dynamic offset)
           ELSE
              EREF=ZERO

              EDSEXP=(-1.0/(kboltz*edstemp))*(ESUBSTOT-EDSOFF)

              IF(EDSEXP.GE.MAXSAFEENE) THEN
                 WRITE(outu,'(a,i3,a,2f14.4)') 'MAINRECEIVE> PROBLEM WITH SUBSYS ',I,' ESUBSTOT,EDSOFF = ', &
                                               ESUBSTOT,EDSOFF
                 WRITE(errmsg,'(A,F18.6,A)') 'EDS OFFSET ENERGY (', EDSEXP, ') IS OUTSIDE SAFE RANGE'
                 CALL WRNDIE(-3,'<MSCALE>',errmsg)
              ENDIF
              EDSEXP=EXP(EDSEXP)
              EDSEXPTOT=EDSEXPTOT+EDSEXP

              ! EDS FORCES
              IPT=0
              DO J=1,NATOM
                 IF(MSYSEL(J,I).EQ.1)THEN
                    IPT=IPT+1
                    EDSDX(J)=EDSDX(J)+(EDSEXP*XSUB(IPT))
                    EDSDY(J)=EDSDY(J)+(EDSEXP*YSUB(IPT))
                    EDSDZ(J)=EDSDZ(J)+(EDSEXP*ZSUB(IPT))
                 ENDIF
              ENDDO

              IF(QCRYST(I)) THEN
                 DO II=1,LENENV
                    EDSEPRESS(II)=EDSEPRESS(II)+(EDSEXP*EPSUBS(II))
                 ENDDO
                 DO II=1,6
                    EDSDXTL(II)=EDSDXTL(II)+(EDSEXP*DXTLSUBS(II))
                 ENDDO
                 EDSVIRI=EDSVIRI+(SVIRI*EDSEXP)
                 EDSVIRE=EDSVIRE+(SVIRE*EDSEXP)
                 EDSVIRKE=EDSVIRKE+(SVIRKE*EDSEXP)
              ENDIF

           ENDIF

        ELSE
#endif

        IF(QCRYST(I)) THEN
           DO II=1,LENENV
              EPRESS(II)=EPRESS(II)+(EPSUBS(II)*FACTSYS)
           ENDDO
           DO II=1,6
              DXTL(II)=DXTL(II)+(DXTLSUBS(II)*FACTSYS)
           ENDDO
           EPROP(VIRI)=EPROP(VIRI)+(SVIRI*FACTSYS)
           EPROP(VIRE)=EPROP(VIRE)+(SVIRE*FACTSYS)
           EPROP(VIRKE)=EPROP(VIRKE)+(SVIRKE*FACTSYS)
        ENDIF


        !
        !     Forces:
        !
        IPT=0
        !            IF(INIT.LT.0)THEN
        ! in the case of PERT we can overwrite the data
        ! the summing up is in the EPERT() routine
        ! 
        !               DO J = 1, NATOM
        !                  IF(MSYSEL(J,I).EQ.1)THEN
        !                     IPT=IPT+1
        !                     DX(J)=XSUB(IPT)
        !                     DY(J)=YSUB(IPT)
        !                     DZ(J)=ZSUB(IPT)
        !                  ENDIF
        !               ENDDO
        !            ELSE
        DO J = 1, NATOM
           IF(MSYSEL(J,I).EQ.1)THEN
              IPT=IPT+1
              DX(J)=DX(J)+XSUB(IPT)*FACTSYS
              DY(J)=DY(J)+YSUB(IPT)*FACTSYS
              DZ(J)=DZ(J)+ZSUB(IPT)*FACTSYS
           ENDIF
        ENDDO
        !            ENDIF
        !
        !     Energy:
        !
        !           for PERT we overwrite (maybe we should always do this???)
        !            IF(INIT.LT.0)THEN
        !               DO J = 1, LENENT
        !                  ETERM(J) = ESUB(J)
        !               ENDDO
        !            ELSE
        DO J = 1, LENENT
           ETERM(J) = ETERM(J)+ESUB(J)*FACTSYS
        ENDDO
        !            ENDIF
        !
        DEALLOCATE(XSUB,YSUB,ZSUB,ESUB)
        !           write(*,*)'MSCALE> Deallocate DDX2...',QSECD
        IF(QSECD) DEALLOCATE(DDX2,ISEL3,IUPT)
        !
#if KEY_EDS==1
        ENDIF 
#endif
     ENDIF
     !
  ENDDO

#if KEY_EDS==1
  IF(QEDS) THEN
     ! final energy
     TOTEDSENE=ZERO
     DO I=1,LENENT
        ETERM(I)=ETERM(I)+EDSETERM(I)
        TOTEDSENE=TOTEDSENE+EDSETERM(I)
     ENDDO
     EDSEPOT=EREF-KBOLTZ*EDSTEMP*LOG(EDSEXPTOT)
     ETERM(EDS)=EDSEPOT-TOTEDSENE

     IF(QDYNCALL.OR.(QREDUPEN.EQV..FALSE.)) THEN
        IF(UPEN.GT.0) THEN
           IF(QEDSDYNOFF) THEN
              WRITE(UPEN,'(A,E21.6,A,E18.6,A,E18.6)',ADVANCE='NO') 'TOT(EDS) = ',TOTEDSENE, ' POT(EDS) = ', &
                                                           EDSEPOT, ' EREF = ', EREF
           ELSE
              WRITE(UPEN,'(A,E21.6,A,E18.6)',ADVANCE='NO') 'TOT(EDS) = ',TOTEDSENE, ' POT(EDS) = ', &
                                                           EDSEPOT
           ENDIF
        ENDIF
     ENDIF

     ! final forces
     DO I=1,NATOM
        DX(I)=DX(I)+EDSDX(I)/EDSEXPTOT
        DY(I)=DY(I)+EDSDY(I)/EDSEXPTOT
        DZ(I)=DZ(I)+EDSDZ(I)/EDSEXPTOT
     ENDDO

     ! Lame hack -- if one subsystem is crystal, assume they all are
     IF(QCRYST(1)) THEN
        DO I=1,LENENV
           EPRESS(I)=EPRESS(I)+EDSEPRESS(I)/EDSEXPTOT
        ENDDO
        DO I=1,6
           DXTL(I)=DXTL(I)+EDSDXTL(I)/EDSEXPTOT
        ENDDO
        EPROP(VIRI)=EPROP(VIRI)+EDSVIRI/EDSEXPTOT
        EPROP(VIRE)=EPROP(VIRE)+EDSVIRE/EDSEXPTOT
        EPROP(VIRKE)=EPROP(VIRKE)+EDSVIRKE/EDSEXPTOT
     ENDIF

     call chmdealloc('mscale.src','MAINRECEIVE','edsdx',NATOM,crl=edsdx)
     call chmdealloc('mscale.src','MAINRECEIVE','edsdy',NATOM,crl=edsdy)
     call chmdealloc('mscale.src','MAINRECEIVE','edsdz',NATOM,crl=edsdz)
     call chmdealloc('mscale.src','MAINRECEIVE','edseterm',LENENT,crl=edseterm)
  ENDIF
#endif

  IF(QDYNCALL.OR.(QREDUPEN.EQV..FALSE.)) THEN  
     IF(UPEN.GT.0) WRITE(UPEN,'(A)') ' '
  ENDIF

  !
  RETURN
END SUBROUTINE MAINRECEIVE
!
SUBROUTINE SERVERBROADCAST(NATOM,LENENT,DX,DY,DZ,ETERM,EPRESS,EPROP,QSECD,DDX)
  !----------------------------------------------------------------------
  !
  !     SUBSYSTEM: Broadcast the results to MAIN
  !
  use stream
  use dimens_fcm
  use number
  use image, only: dxtl
#if KEY_TORQUE==1
  use torque, only: trqtxyz,ntrqcenter 
#endif
  use energym, only: lenenv, viri, vire, virke
  !
#if KEY_PARALLEL==1
  use parallel
  use mpi       
#endif
  !
  LOGICAL QSECD
  INTEGER NATOM,LENENT,NAT3,I,II
  real(chm_real) DX(*),DY(*),DZ(*),ETERM(*),EPRESS(*),EPROP(*),DDX(*)
  !
  INTEGER(chm_int4) MEROOT,MPIDP,COMPARENT,IERR,N,N6,LEN4
  !
  LEN4=LENENT
  N=NATOM
  NAT3=NATOM*3
  N6=((NAT3)*(NAT3+1))/2
  !     write(*,*)'MSCALE SUBS> N6 = ',N6
  !     write(*,*)'MSCALE QSECD = ', QSECD
  !     IF(QSECD) THEN
  !        DO I=1,171
  !           write(*,*)'DDX = ', DDX(I)
  !        ENDDO
  !        write(*,*)'MSCALE DDX(1) = ',DDX(1)
  !        write(*,*)'MSCALE DDX(171) = ',DDX(171)
  !     ENDIF 

  CALL MPI_COMM_GET_PARENT(COMPARENT,IERR)
  !
  IF(MYNOD.EQ.0)THEN
     MEROOT=MPI_ROOT
  ELSE
     MEROOT=MPI_PROC_NULL
  ENDIF
  !
  MPIDP=MPI_DOUBLE_PRECISION
  IF(QSUBSDBG) THEN
    call printe(6,EPROP,ETERM,'SUBS','ENR',.TRUE., &
                1,1,1,.TRUE.)
  ENDIF

  !
  !C      write(*,*)'SERVERBROADCAST>comparent,natom,lenent=',
  !C     $     comparent,natom,lenent
  CALL MPI_BCAST(DX,N,MPIDP,MEROOT,COMPARENT,IERR)
  CALL MPI_BCAST(DY,N,MPIDP,MEROOT,COMPARENT,IERR)
  CALL MPI_BCAST(DZ,N,MPIDP,MEROOT,COMPARENT,IERR)
  CALL MPI_BCAST(ETERM,LEN4,MPIDP,MEROOT,COMPARENT,IERR)

  IF(QCRYSUB) THEN
     CALL MPI_BCAST(DXTL,6,MPIDP,MEROOT,COMPARENT,IERR)
     CALL MPI_BCAST(EPROP(VIRI),1,MPIDP,MEROOT,COMPARENT,IERR)
     CALL MPI_BCAST(EPROP(VIRE),1,MPIDP,MEROOT,COMPARENT,IERR)
     CALL MPI_BCAST(EPROP(VIRKE),1,MPIDP,MEROOT,COMPARENT,IERR)
     CALL MPI_BCAST(EPRESS,LENENV,MPIDP,MEROOT,COMPARENT,IERR)
  ENDIF

  IF(QSECD) THEN
     CALL MPI_BCAST(DDX,N6,MPIDP,MEROOT,COMPARENT,IERR)
  ENDIF
  !
#if KEY_TORQUE==1
  IF(QTRQSUB) THEN
     CALL MPI_BCAST(trqtxyz,3*ntrqcenter,MPIDP,MEROOT,COMPARENT,IERR)
  ENDIF
#endif
  !
  RETURN
END SUBROUTINE SERVERBROADCAST
!
!
SUBROUTINE SERVERRECEIVE(X,Y,Z,QSECD)
  !----------------------------------------------------------------------
  !
  !     SUBSYTEMS: get the coordinates from the MAIN
  !
  use stream
  use dimens_fcm
  use psf
  use image
#if KEY_TORQUE==1
  use torque, only:trqrotm,ntrqcenter 
#endif
  use parallel
  use memory
  !
#if KEY_PARALLEL==1
  use mpi   
#endif
  !
  LOGICAL QSECD
  real(chm_real) X(*),Y(*),Z(*)
  !
  REAL(CHM_REAL), allocatable, dimension(:) :: AN
  INTEGER :: ALL_STAT,NCONTROL
  INTEGER(chm_int4) :: IONE,NX,N,MEROOT,MPIDP,COMPARENT,MPIINT,IERR,MX
  INTEGER(chm_int4) :: ISIX,MPILOG,I
  CHARACTER(len=4) :: MPTYP
  real(chm_real),allocatable,dimension(:,:,:) :: TRANSF
  !
  CALL MPI_COMM_GET_PARENT(COMPARENT,IERR)
  !
  MEROOT=0
  IONE=1
  ISIX=6
  MPIDP=MPI_DOUBLE_PRECISION
  MPIINT=MPI_INTEGER
  MPILOG=MPI_LOGICAL
  N=NATOM
  !
  ! Perform the job controlling communication:
  ! MAIN broadcasts a stop flag
  CALL MPI_BCAST(NCONTROL,IONE,MPIINT,MEROOT,COMPARENT,IERR)
  IF(NCONTROL < 0) THEN
     CALL STOPCH('NORMAL STOP (MSCALE SLAVE)')
  ENDIF
  !
  !     ATOM in CHARMM: We get this data, but we ignore it for now
  IF(QATOMSUB)THEN
     ALLOCATE(AN(N),STAT=ALL_STAT)
     CALL MPI_BCAST(NX,IONE,MPIINT,MEROOT,COMPARENT,IERR)
     CALL MPI_BCAST(MX,IONE,MPIINT,MEROOT,COMPARENT,IERR)
     IF(NX.NE.N)THEN
        WRITE(OUTU,*)'Problem with communication of ATOM data.'
        WRITE(OUTU,*)'Expected data =', N
        WRITE(OUTU,*)'Transfered data =', NX
     ENDIF
     CALL MPI_BCAST(AN,N,MPIDP,MEROOT,COMPARENT,IERR)
     DEALLOCATE(AN)
  ENDIF
  IF(QCRYSUB)THEN
     CALL MPI_BCAST(MPTYP,4,MPI_CHARACTER,MEROOT,COMPARENT,IERR)
     IF(MPTYP.NE.XTLTYP) THEN
        WRITE(*,*) 'MPTYP = ', MPTYP, ' but XTLTYP = ', XTLTYP
        CALL WRNDIE(-2,'<SERVERRECEIVE>', &
             'Crystal types do not match: do NOT run constant pressure dynamics!')
     ENDIF

     CALL MPI_BCAST(XUCELL,ISIX,MPIDP,MEROOT,COMPARENT,IERR)
     CALL XTLAXS(XTLABC,XUCELL)
     CALL XTLMSR(XUCELL)
     call chmalloc('mscale.src','SERVERRECEIVE','TRANSF',3,4,XNSYMM,crl=TRANSF)
     CALL IMFILL(TRANSF,.FALSE.)
     call chmdealloc('mscale.src','SERVERRECEIVE','TRANSF',3,4,XNSYMM,crl=TRANSF)

  ENDIF
  !
  IF(QCHRGSUB)THEN
     CALL MPI_BCAST(CG,N,MPIDP,MEROOT,COMPARENT,IERR)
  ENDIF
  !
  CALL MPI_BCAST(X,N,MPIDP,MEROOT,COMPARENT,IERR)
  CALL MPI_BCAST(Y,N,MPIDP,MEROOT,COMPARENT,IERR)
  CALL MPI_BCAST(Z,N,MPIDP,MEROOT,COMPARENT,IERR)

  ! get flags from main... finite, step and qsecd... 
  !     IF(QSECD) THEN
  CALL MPI_BCAST(QSECD,IONE,MPILOG,MEROOT,COMPARENT,IERR)
  !        write(*,*)'get coords: subs --> main ',QSECD
  !        write(*,*)'SERVERRECEIVE> QSECD = ', QSECD
  !     ENDIF 
  !
#if KEY_TORQUE==1
  IF(QTRQSUB) THEN
     CALL MPI_BCAST(trqrotm ,9*ntrqcenter,MPIDP,MEROOT,COMPARENT,IERR)
  ENDIF
#endif
  !

  RETURN
END SUBROUTINE SERVERRECEIVE

SUBROUTINE MSCLVIB(X,Y,Z,NATOM,SUBSTEP,DDX) 
  !----------------------------------------------------------------------
  !
  !     Sets up and passes necessary information to MSCLNORMDS 
  !     to compute Hessians via finite differences 
  !
  use dimens_fcm
  use number
  use exfunc
  !
  use deriv
  !
  real(chm_real) X(:),Y(:),Z(:),DDX(*),SUBSTEP
  INTEGER NATOM,NDIM,NAT3,N6,JSPACE
  REAL(CHM_REAL), allocatable, dimension(:) :: DDS

  NAT3=NATOM*3
  NDIM=NAT3
  N6=((NAT3)*(NAT3+1))/2
  CALL MSCLNORMDS(X,Y,Z,NAT3,DDX,SUBSTEP)
  !     deallocate(DDS)
  RETURN
END SUBROUTINE MSCLVIB

SUBROUTINE MSCLNORMDS(X,Y,Z,NAT3,DDX,SUBSTEP)
  !----------------------------------------------------------------------
  !
  !     SETS UP AND DIAGONALIZES THE SECOND DERIVATIVE MATRIX
  !  (This is a stripped down version for doing only finite differences)
  !
  !     By Bernard R. Brooks   1982
  !
  use dimens_fcm
  use number
  use exfunc
  use deriv
  use consta
  use energym
  use stream
  use bases_fcm
  use vibcom
  !
  INTEGER NAT3
  real(chm_real) X(:),Y(:),Z(:)
  real(chm_real) DDX(*)
  !
  real(chm_real)  SUBSTEP
  !
  !
  INTEGER NATOM,NATP,N6,I
  REAL(CHM_REAL), allocatable, dimension(:) :: XREF,YREF,ZREF, &
       DXF,DYF,DZF

  !
  NATOM=NAT3/3
  !
  N6=(NAT3*(NAT3+1))/2
  DO I=1,N6
     DDX(I)=ZERO
  ENDDO
  !
  CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,0)
  allocate(XREF(NATOM),YREF(NATOM),ZREF(NATOM),DXF(NATOM), &
       DYF(NATOM),DZF(NATOM))
  CALL GENSD2(NAT3,X,Y,Z,XREF,YREF,ZREF, &
       DDX,DXF,DYF,DZF,SUBSTEP,.FALSE., &
       3*NATOM,.FALSE.) ! JZ_UW12
  !
  !...#if KEY_PARALLEL==1
  !      CALL VDGBR(DX,DY,DZ,1)
  !      CALL GCOMB(DDX,N6)
  !...#endif
  !
  IF(PRNLEV >= 2) THEN
     CALL PRINTE(OUTU, EPROP, ETERM, 'VIBR', 'ENR', .TRUE., &
          1, ZERO, ZERO, .TRUE.)
  ENDIF
  RETURN 
END SUBROUTINE MSCLNORMDS

SUBROUTINE EMSCALE(INIT,NATOM,LENENT,X,Y,Z,DX,DY,DZ,ETERM, &
     EPRESS,EPROP,QSECD,DD1,QDYNCALL)
  !----------------------------------------------------------------------
  !
  !     This is the central routine in the MSCAle setup for master
  !     script. It performs:
  !     - communicates coordinates to subsystems
  !     - communicates control parameter for the slave (if needed???)
  !     - communcation of energy, forces, virial, etc
  !     - places these data into the right place in DX(),DY(),DZ()
  !     - it is called from the ENERGY() routine:
  !
  !     INIT - flag to control where this routine is called.
  !            if INIT=1 then just send the coordinates to the subsystems
  !            if INIT=0 then gether the info from subsystems
  !            This way we overlap the calculations of subsystems with the main
  !
  !
  !
  use stream

  !
  LOGICAL QSECD,QDYNCALL
  INTEGER INIT,NATOM,LENENT,I
  real(chm_real) X(*),Y(*),Z(*),DX(*),DY(*),DZ(*),ETERM(*),EPRESS(*),EPROP(*),DD1(*)
  !

  !     IF(QSECD) THEN 
  !        DO I=1,NATOM*3 
  !           write(*,*)'DD1 = ', DD1(I)
  !        ENDDO 
  !     ENDIF 

  IF(INIT.EQ.1)THEN
     !
     !     Pack/send the coordinates to subsystems
     !
     CALL MAINBROADCAST(NATOM,X,Y,Z,QSECD)
     !
     RETURN
  ENDIF
  !
  !     Receive data from the subsystems, scale it, place it in the
  !     right order, etc...
  !
  CALL MAINRECEIVE(INIT,NATOM,LENENT,DX,DY,DZ,ETERM,EPRESS,EPROP,QSECD,DD1,QDYNCALL)
  !

  RETURN
END SUBROUTINE EMSCALE

#endif  /* main_mscale */
!
END MODULE MSCALEMOD
