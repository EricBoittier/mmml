module parallel
  use chm_kinds
  use dimens_fcm
  implicit none

  ! silencing warnings in F90 and above
  interface psnd4
     subroutine PSND4s(var,len)
       integer var,len
     end subroutine PSND4s
     subroutine PSND4m(arr,len)
       integer arr(*),len
     end subroutine PSND4m
     subroutine PSND4l(arr,len)
       logical arr
       integer len
     end subroutine PSND4l
     subroutine PSND4lm(arr,len)
       logical arr(*)
       integer len
     end subroutine PSND4lm
     subroutine PSND4r(arr,len)
       use chm_kinds
       real(chm_real4)  arr(*)
       integer len
     end subroutine PSND4r
  end interface psnd4
  interface psnd8
     subroutine PSND8s(var,len)
       use chm_kinds
       integer len
       real(chm_real8) var
     end subroutine PSND8s
     subroutine PSND8m(arr,len)
       use chm_kinds
       integer len
       real(chm_real8) arr(*)
     end subroutine PSND8m
     subroutine PSND8i(arr,len)
       use chm_kinds
       integer len
       integer(chm_int8) arr(*)
     end subroutine PSND8i
  end interface psnd8

  interface gcomb
     subroutine gcombs(rvar,lngth) ! scalar version
       use chm_kinds
       integer,optional :: lngth
       real(chm_real) :: rvar
     end subroutine gcombs
     subroutine gcombsi(ivar,lngth)
       integer :: ivar
       integer,optional :: lngth
     end subroutine gcombsi
     subroutine gcombr(rarr,lenarr)
       use chm_kinds
       real(chm_real) :: rarr(*)
       integer :: lenarr
     end subroutine gcombr
     subroutine gcombi(iarr,lenarr)
       integer :: iarr(*)
       integer :: lenarr
     end subroutine gcombi
  end interface gcomb

#if KEY_PARALLEL==0 /*parallel_serial*/

  ! Fallback values for serial compile
  integer, parameter :: NUMNOD = 1, MYNOD = 0, MYNODP = 1, maxnode=1, MYNODG = 0, &
       MYNODGP = 1
  logical :: vib_finite = .false.

contains

  subroutine parallel_iniall()

    return
  end subroutine parallel_iniall

  integer function nnod()
    nnod=1
    return
  end function nnod

  integer function mnod()
    mnod=0
    return
  end function mnod

#else /* (parallel_serial)*/

  ! COMM_CHARMM - Communicator for this CHARMM instance
  !
  integer COMM_CHARMM
  integer comm_charmm_index
  integer MASTER_NODE

  !  MAXNODE - The maximum number of nodes that will participate.
  !  TMGNUM  - Number of timing partitions (i.e. energy, wait, combine...)
  !
#if KEY_REPDSTR==1
  integer,PARAMETER :: MAXNODE=4096,TMGNUM=7
#elif KEY_STRINGM==1 /* VO : allow very large numbers of nodes (should be dynamic) */
  integer,parameter :: MAXNODE=2**15,TMGNUM=7
#else /**/
  integer,PARAMETER :: MAXNODE=2048,TMGNUM=7
#endif
  !
  !  TMERI   - Timer values for various terms.
  !  TSUMI   - Timer sums for various terms.
  !
  !     TMERI(1) - external energy
  !     TMERI(2) - internal energy
  !     TMERI(3) - wait (shows unbalance: currently in energy only)
  !     TMERI(4) - global communication
  !     TMERI(5) - PME communication
  !     TMERI(6) - nbondg - nonbond list generation
  !     TMERI(7) - dcntrl - integration

  integer,parameter :: TIMEXTE=1,TIMINTE=2,TIMWAIT=3,TIMGCOMM=4
  integer,parameter :: TIMECOMM=5,TIMNBOND=6,TIMDCNTRL=7

  !     INTEGER NNOD, MNOD, NPTWO
  !     EXTERNAL NNOD, MNOD, NPTWO
  !
  !  IMERI   - Number of times trough timer loop
  !  MYNOD   - Number of this node
  !  MYNODP  - Number of this node +1
  !  NUMNOD  - Total number of nodes
  !  NODDIM  - Dimension of hypercube (if appropriate).
  !  INODE() - Mapping of logical processors to logical nodes.
  !  DPCOMM  - Double precision communication?
  !  IPARPT()- Boundary for atoms in vector distributed global sum.
  !  JPARPT()- Boundary for other than atoms for vector distribute.
  !  NPARPT()- Blockcounts for IPARPT
  !  IPPMAP()- Mapping of logical nodes to physical nodes.
  !  TIDS()  - Mapping between task ID and mynode numbers.
  !  NUMNODG - Number of nodes (global)
  !  MYNODG  - Which node is this in the global parallel system
  !  IRNGFL  - Flag to properly start prallel random number generator
  !  QVSEND  - Flag to broadcast initial velocities to all processes
  !  QQMPAR  - is replica path using parallel QM code flag
  !  CPUMAP  - Mapping of atoms per CPU for Spacial Decomposition
  !  PARHOST - Names of te host computers in a parallel system
  !  PARHLEN - Lengths of the strings in PARHOST table
  !  mpi_integer_size - Size of the integer in mpi
  !  mpi_real8_size - Size of the real8 in mpi
  !  mpi_real4_size - Size of the real4 in mpi
  !
  !
  INTEGER :: IMERI, MYNOD, MYNODP, NUMNOD, NODDIM,  DPCOMM

  INTEGER :: PLNOD0,NUMNODG,MYNODG,IRNGFL,IOCPUMP
  INTEGER :: MYNODGP
  CHARACTER(len=80) :: PARHOST(maxnode)
  integer mpi_integer_size, mpi_real8_size, mpi_real4_size

  integer,allocatable,dimension(:) :: ICPUMAP,fmmcpu

#if KEY_PARINFNTY==1
  logical :: qinfinity
#endif
  !
  !  QMPI    - flag telling mopac not to use parallel code (uses path-integral parallel)
  LOGICAL QMPI
  logical :: qsplit=.false.
  integer :: INODE(MAXNODE), &
       IPARPT(0:MAXNODE), JPARPT(0:MAXNODE), &
       NPARPT(0:MAXNODE),IPPMAP(0:MAXNODE), &
       PARHLEN(MAXNODE)
  real(chm_real) :: TMERI(TMGNUM), TSUMI(TMGNUM)

  LOGICAL QVSEND,QQMPAR,QICPUMAP


#if KEY_SCCDFTB==1
  !  QC: UW_031205 Add arrays that save boundary k-vectors for SCC/MM eWald
  !
  !  IKVCSC()- Boundary for k-vectors in vector distributed global sum.
  integer :: IKVCSC(0:MAXNODE),IMMLSC(0:MAXNODE)
#endif

#if KEY_PARASCAL==1 /*parasfcm*/
  !  MAXIPB  - Maximum number of blocks.
  !  NPBLOCK - Number of parallel blocks
  !  IPBLOCK - In what block does an atom lie (range 1 to 4 for 10 nodes)
  !  JPBLOCK - What processor will integrate atom (range 0 to 9 for 10)
  !  IPMAT(,) - Processor list for reduced communication.
  !  JPMAT(,) - What processor will process interactions between blocks.
  !  QPSRNB  - Generate a reduced size nonbond list (distributed)?
  !
  INTEGER,PARAMETER :: MAXIPB=8
  INTEGER NPBLOCK
  LOGICAL QPSRNB
  integer :: IPMAT(MAXIPB,MAXIPB),JPMAT(MAXIPB,MAXIPB), &
       IPBLOCK(MAXA),JPBLOCK(MAXA)
  !
  !  PSICB  -  What processor will calculate this bond
  !  PSICT  -  What processor will calculate this angle
  !  PSICP  -  What processor will calculate this dihedral
  !  PSICI  -  What processor will calculate this improper dihedral
  !
  integer :: PSICB(MAXB),PSICT(MAXT),PSICP(MAXP),PSICI(MAXIMP)
  !

#endif /* (parasfcm)  parascal*/

#if KEY_CMPI==1
  ! values differ from standard MPI, so use different names
  integer, parameter :: CMPI_COMM_WORLD = 0
  integer, parameter :: CMPI_SUM = 0, CMPI_BOR = 1
  integer, parameter :: CMPI_DOUBLE_PRECISION = 0, CMPI_INTEGER = 1, CMPI_BYTE = 2
#endif

!*************************************************************************************
contains                  !CONTAINS
!*************************************************************************************

  subroutine parallel_iniall()
#if KEY_SPACDEC==1
    QICPUMAP=.TRUE.
#endif
    QMPI =.FALSE.
    return
  end subroutine parallel_iniall

  !-----------------------------------------------------------------------
  integer function nptwo()
    !     Returns log_2(NNOD()). This is LOCAL!!!
    !
    integer i, nd

    i=lnnod()
    nd=-1
    do while ( i > 0 )
       nd=nd+1
       i=i/2
    enddo
    nptwo=nd
    return
  end function nptwo


  INTEGER FUNCTION MNOD()
    !-----------------------------------------------------------------------
    !
    ! Reports node number for this node
    !
! for DDI:     INTEGER DDI_NP,DDI_ME
! for DDI:     CALL DDI_NPROC(DDI_NP,DDI_ME)
! for DDI:     MNOD = DDI_ME
    !
#if KEY_MPI==1 /*socket1*/
    use mpi
    INTEGER STATUS
    CALL MPI_COMM_RANK(COMM_CHARMM,MNOD,STATUS)
#endif /* (socket1)*/
    !
    RETURN
  END FUNCTION MNOD

  INTEGER FUNCTION NNOD()
    !-----------------------------------------------------------------------
    !
    ! How many nodes are in the parallel system.
    !
! for DDI:     INTEGER DDI_NP,DDI_ME
! for DDI:     CALL DDI_NPROC(DDI_NP,DDI_ME)
! for DDI:     NNOD = DDI_NP
    !
#if KEY_MPI==1
    use mpi
    INTEGER STATUS
    CALL MPI_COMM_SIZE(COMM_CHARMM,NNOD,STATUS)
#endif
    !
    RETURN
  END FUNCTION NNOD

  integer function lnnod()
    !-----------------------------------------------------------------------
    !
    ! reports number of nodes. local routine: ippmap ready
    !
    lnnod=numnod
    !
    return
  end function lnnod

  !
  INTEGER FUNCTION LMNOD()
    !-----------------------------------------------------------------------
    !
    ! Reports node number for this node. Local routine: IPPMAP ready
    !
    !
    LMNOD=MYNOD
    !
    RETURN
  END FUNCTION LMNOD
  !
  !
#if KEY_PARASCAL==1 /*parascal*/
  INTEGER FUNCTION NUMXPART()
    !-----------------------------------------------------------------------
    !
    ! Reports number of nodes on for PARASCAL on hypercube:
    ! load 0 1 2 3 4 5 6 7 8 9 <program-name> doesn't work!
    !
    NUMXPART = (NPART+1)*NPART/2
    RETURN
  END FUNCTION NUMXPART
#endif /* (parascal)*/

#endif /* (parallel_serial)*/

end module parallel
