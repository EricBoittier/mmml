module paral4
use chm_kinds
use parallel
implicit none
#if KEY_PARALLEL==1 /*paral4*/
!------------------------------------------------------------------------------------------------
contains
        subroutine setcpustruc
! creates neighborhoods of cpus, composed of head node (rank zero) and base nodes (non-zero).
! headcom == communicator between all headnodes in system
! hoodcom == communicator within a neighborhood
! basecom == communicator among base nodes in a neighborhood
!                                                --RJP
        use chm_kinds
        use dimens_fcm
        use parallel
        use cpustruc 
        !
        use exfunc
        use memory
        use comand  
        use string
        INCLUDE  'mpif.h'

        integer(chm_int4) :: myid, ierr,NUMPROC,MPIVER,MPISUBVER,IERROR,NEWRANK,orig_group
        integer(chm_int4) :: headgrp,mylocrank,mylocsize,headnsz,orig_size,STATUS,basegrp
        integer(chm_int4),allocatable,dimension(:) :: baserank
!        integer(chm_int4),dimension(4) :: headrank
        real(chm_real) :: RATIO,RNUMPROC,RNUMHOOD,RMYID
        real(chm_real),parameter :: TOL=1D-8
        integer :: II,JJ,DATASIZE,NODE,NOTHER,LO,HI,CPU,HOOD,CNT,LOCNODE
        real(chm_real),allocatable,dimension(:) :: DATAS,DATAR
        logical :: QTRANSP,QTEST
!----------------------------------------------------------------------------------------- 

        CALL MPI_GET_VERSION(MPIVER,MPISUBVER,IERR)
!        WRITE(6,*) 'MYNODP is ',MYNODP,' mpi version is ',MPIVER

        call MPI_COMM_RANK( MPI_COMM_WORLD, myid, ierr )
!        WRITE(6,*) 'MYNODP is ',MYNODP,' myid is ',myid
!        MYID = MYNOD + 100
        RMYID = MYID
        call MPI_COMM_SIZE( MPI_COMM_WORLD, NUMPROC, ierr ) 
!        WRITE(6,*) 'MYNODP ',MYNODP,' NUMPROC is ',myid
        RNUMPROC = NUMPROC
!
!          Extract the original group handle
        call MPI_COMM_GROUP(MPI_COMM_WORLD, orig_group, ierr)
!        WRITE(6,*) 'PAST COMM GROUP'
        call MPI_COMM_SIZE(MPI_COMM_WORLD,orig_size,ierr)
!        write(6,*) 'size of mpi_com_world is ',orig_size

        NUMHOOD = GTRMI(COMLYN,COMLEN,'NEIG',1) 
        QTRANSP = .FALSE.
        IF(INDXA(COMLYN,COMLEN,'TRAN').GT.0) QTRANSP = .TRUE.
        QTEST = .FALSE.
        IF(INDXA(COMLYN,COMLEN,'TEST').GT.0) QTEST = .TRUE.
!        write(6,*) 'MYNODP ',MYNODP,' NUMHOOD ',NUMHOOD
        RNUMHOOD = NUMHOOD
!        RNATOM = NATOM
!        NATOMHD = IDINT(RNATOM/RNUMHOOD)
        
        IF(RNUMHOOD.GT.RNUMPROC) THEN
         IF(MYNODP.EQ.1) THEN
          WRITE(6,*) '# OF NEIGHBORHOODS EXCEEDS # OF CPUS'
          CALL WRNDIE(-5,'<SETCPUSTRUC>',' TOO MANY CPU NEIGHBORHOODS')
         ENDIF
        ENDIF

        RATIO = RNUMPROC/RNUMHOOD
        NUMNODH = IDINT(RNUMPROC/NUMHOOD+0.5)
        NUMNODHI4 = NUMNODH  !int4 
        RNUMNODH = NUMNODH
!        WRITE(6,*) 'RATIO IS ',RATIO,' NUMNODH is ',NUMNODH 
        IF(RATIO-NUMNODH.GT.TOL) THEN
         IF(MYNODP.EQ.1) THEN
          WRITE(6,*) 'WARNING:  TOTAL # CPUS NOT EVENLY DIVISIBLE BY # NEIGHBORHOODS'
         ENDIF
        ENDIF
! allocate space for local arrays
        if(.not.allocated(headrank)) call chmalloc('spaccom.src','setcpustruc','headrank',NUMHOOD,ci4=headrank)
        if(.not.allocated(baserank)) call chmalloc('spaccom.src','setcpustruc','baserank',NUMNODH-1,ci4=baserank)
        if(.not.allocated(NEIGHMAPP)) call chmalloc('spaccom.src','setcpustruc','NEIGHMAPP',NUMNOD,intg=NEIGHMAPP)
        if(allocated(LOC2GLBP)) call chmdealloc('spaccom.src','setcpustruc','LOC2GLBP',NUMNODH,intg=LOC2GLBP)
        call chmalloc('spaccom.src','setcpustruc','LOC2GLBP',NUMNODH,intg=LOC2GLBP)
!        WRITE(6,*) 'AFTER ALLOCATION, mynod p = ',mynodp

! determine order of cubes to be taken in during spatial assignment
      cellordtyp = 'default'  ! grid order
      IF(INDXA(COMLYN,COMLEN,'CONT').GT.0) cellordtyp = 'continu'  !continous 'lawnmower' pattern
!
      QWRSPACMAP = .false.
      IF(INDXA(COMLYN,COMLEN,'WCUB').GT.0)  QWRSPACMAP = .true.  !write out coordinates of cubes, by cpu

!        WRITE(6,*) 'AFTER INDXA, mynod p = ',mynodp
      if(.not.allocated(HOODCPULST)) call chmalloc('spaccom.src','SETCPUSTRUC','HOODCPULST',NUMNOD,intg=HOODCPULST)
      if(.not.allocated(HOODCPUHI)) call chmalloc('spaccom.src','SETCPUSTRUC','HOODCPUHI',NUMHOOD,intg=HOODCPUHI)
      if(.not.allocated(HOODCPUMNY)) call chmalloc('spaccom.src','SETCPUSTRUC','HOODCPUMNY',NUMHOOD,intg=HOODCPUMNY)

      IF(.NOT.QTRANSP) THEN  !a neighborhood consists of consecutive node numbers ("by sections")
!-------------------------------------
        MYNHOOD = IDINT(RMYID/RNUMNODH)  !which neighborhood
        IMYNHDP = MYNHOOD + 1
        MYKEY = MOD(MYID,NUMNODH)
        MYKEYP = MYKEY + 1
        IMYKEYP = MYKEYP
!        WRITE(6,*) 'MYID IS ',MYID,' MYNHOOD IS ',MYNHOOD 
!        WRITE(6,*) 'MYID IS ',MYID,' MYKEY IS ',MYKEY
        call MPI_COMM_SPLIT(MPI_COMM_WORLD, MYNHOOD, MYKEY, HOODCOM, IERROR)
!        WRITE(6,*) 'MYNODP is ',MYNODP,' HOODCOM IS ',HOODCOM
! given the global rank, what is the neighborhood
        DO NODE = 1,NUMNOD
          NEIGHMAPP(NODE) = IDINT((NODE-1)/RNUMNODH) + 1    
!         IF(MYNODP.EQ.1) THEN
!          WRITE(6,*) 'FOR NODE ',NODE,' NEIGHMAPP IS ',NEIGHMAPP(NODE)
!         ENDIF
        ENDDO
!        LOCALRANKH = 0
!        DO NODE = 1,NUMNOD
!         
!        ENDDO
!        GLOBRNKH = 0
!        DO NODE = 1,NUMNOD 
!         GLOBRNKH(NODE) = (NUMHOOD-1)+LOCALRANK   
!        ENDDO
        call MPI_Comm_rank(HOODCOM, newrank, ierror)

!        write(6,*) 'Global ', myid,' is now local rank ',newrank,' MYNHOOD ',MYNHOOD
!        call MPI_Comm_size(NYNHOOD, sizeh, ierror)
! mapping from local rankp to global rankp
         DO LOCNODE = 1,NUMNODH
           LOC2GLBP(LOCNODE) = LOCNODE + MYNHOOD*NUMNODH
         ENDDO
! create matrix for head node ranks
        DO II = 1,NUMHOOD
          headrank(II) = (II -1)*NUMNODH
!          WRITE(6,*) 'MYNODP Is ',MYNODP,' HEADRANK OF ',II,' IS ',headrank(II)
        ENDDO 
! create group for head nodes
!        write(6,*) 'MYNODP ',MYNODP,' just before mpigroupincl'
!        WRITE(6,*) 'NUMHOOD IS ',NUMHOOD,' HEADRANK is ',HEADRANK
! group creation must be carried out on all nodes:
        call MPI_GROUP_INCL(orig_group, NUMHOOD, headrank, headgrp, ierr) 
!        write(6,*) 'MYNODP ',MYNODP,' just after mpigroupincl'
        call MPI_Group_rank(headgrp, mylocrank, ierr)
        call MPI_Group_size(headgrp, mylocsize, ierr)
!        WRITE(6,*) 'MYNODP is ',MYNODP,' mylocrank in headgrp is ',mylocrank,' mylocsize is ',mylocsize 
! create head node communicator
        call MPI_COMM_CREATE(MPI_COMM_WORLD,headgrp,HEADCOM,ierr) 
! presumably you cannot ask about groups you are not in:
        IF(MYKEY.EQ.0) THEN  !if headnode
         call MPI_COMM_SIZE(HEADCOM,headnsz,ierr)
!        WRITE(6,*) 'MYNODP ',MYNODP,' HEADNSZ IS ',HEADNSZ
        ENDIF
! create the channel or communicator for base nodes (this is a local intraneighborhood communicator)
        IF(NUMNODH.GT.1) THEN
         DO II = 1,NUMNODH-1
          baserank(II) = II + MYNHOOD*NUMNODH 
!          WRITE(6,*) 'MYNODP Is ',MYNODP,' BASERANK OF ',II,' IS ',baserank(II)
         ENDDO 
         call MPI_GROUP_INCL(orig_group,NUMNODH-1, baserank, basegrp, ierr)
         call MPI_Group_rank(basegrp, mylocrank, ierr)
         call MPI_Group_size(basegrp, mylocsize, ierr)
!         WRITE(6,*) 'MYNODP is ',MYNODP,' mylocrank in basegrp is ',mylocrank,' mylocsize is ',mylocsize
         call MPI_COMM_CREATE(MPI_COMM_WORLD,basegrp,BASECOM,ierr)
!         WRITE(6,*) 'MYNODP is ',MYNODP,' after MPI_COMM_CREATE '
        ENDIF !if base nodes exist
        CPU = 0
        DO HOOD = 1,NUMHOOD
         DO II = 1,NUMNODH
           CPU = CPU + 1
           HOODCPULST(CPU) = CPU 
         ENDDO
         HOODCPUHI(HOOD) = CPU
         HOODCPUMNY(HOOD) = NUMNODH
        ENDDO
        DO HOOD = 1, NUMHOOD
         HI = HOODCPUHI(HOOD)
         LO = HI - HOODCPUMNY(HOOD) + 1
         DO II = LO,HI
           CPU = HOODCPULST(II)
!           WRITE(6,*) '<<HOOD ',HOOD,' HOLDS CPU ',CPU,' NEIGHMAPP ',NEIGHMAPP(CPU) 
         ENDDO
        ENDDO
!----------------------------------------- 
      ELSE !take transpose
! take the transpose of the NODE,RANK matrix
        MYNHOOD = MOD(MYID,NUMHOOD)
        IMYNHDP = MYNHOOD + 1
        MYKEY = INT(RMYID/RNUMHOOD)
        MYKEYP = MYKEY + 1
        IMYKEYP = MYKEYP
!        WRITE(6,*) 'MYID IS ',MYID,' MYNHOOD IS ',MYNHOOD 
!        WRITE(6,*) 'MYID IS ',MYID,' MYKEY IS ',MYKEY
        
        DO NODE = 1,NUMNOD
          NEIGHMAPP(NODE) = MOD((NODE-1),NUMHOOD) + 1
         IF(MYNODP.EQ.1) THEN
          WRITE(6,*) 'FOR NODE ',NODE,' NEIGHMAPP IS ',NEIGHMAPP(NODE) 
         ENDIF
        ENDDO
! 
        call MPI_COMM_SPLIT(MPI_COMM_WORLD, MYNHOOD, MYKEY, HOODCOM, IERROR)
!        WRITE(6,*) 'MYNODP is ',MYNODP,' HOODCOM IS ',HOODCOM

        call MPI_Comm_rank(HOODCOM, newrank, ierror)

!        write(6,*) 'Global ', myid,' is now local rank ',newrank,' MYNHOOD ',MYNHOOD
! mapping from local rankp to global rankp
         DO LOCNODE = 1,NUMNODH
           LOC2GLBP(LOCNODE) = (LOCNODE-1)*NUMHOOD + IMYNHDP
         ENDDO
! create matrix for head node ranks
        DO II = 1,NUMHOOD
          headrank(II) = II -1
!          WRITE(6,*) 'MYNODP Is ',MYNODP,' HEADRANK OF ',II,' IS ',headrank(II)
        ENDDO
! create group for head nodes
! group creation must be carried out on all nodes:
        call MPI_GROUP_INCL(orig_group, NUMHOOD, headrank, headgrp, ierr) 
        call MPI_Group_rank(headgrp, mylocrank, ierr)
        call MPI_Group_size(headgrp, mylocsize, ierr)
! create head node communicator
        call MPI_COMM_CREATE(MPI_COMM_WORLD,headgrp,HEADCOM,ierr) 
! presumably you cannot ask about groups you are not in:
        IF(MYKEY.EQ.0) THEN  !if I'm a head node
         call MPI_COMM_SIZE(HEADCOM,headnsz,ierr)
        ENDIF
! create the channel or communicator for base nodes (this is a local intraneighborhood communicator)
        IF(NUMNODH.GT.1) THEN
         DO II = 1,NUMNODH-1
          baserank(II) = II*NUMHOOD + MYNHOOD
!          WRITE(6,*) 'MYNODP Is ',MYNODP,' BASERANK OF ',II,' IS ',baserank(II)
         ENDDO
         call MPI_GROUP_INCL(orig_group,NUMNODH-1, baserank, basegrp, ierr)
         call MPI_Group_rank(basegrp, mylocrank, ierr)
         call MPI_Group_size(basegrp, mylocsize, ierr)
         call MPI_COMM_CREATE(MPI_COMM_WORLD,basegrp,BASECOM,ierr)
!
        ENDIF !if base nodes exist
        CNT = 0
        DO HOOD = 1,NUMHOOD
         DO II = 1,NUMNODH
           CNT = CNT + 1
           CPU = (II-1)*NUMHOOD + HOOD
           HOODCPULST(CNT) = CPU 
         ENDDO
         HOODCPUHI(HOOD) = CNT
         HOODCPUMNY(HOOD) = NUMNODH
        ENDDO
        DO HOOD = 1, NUMHOOD
         HI = HOODCPUHI(HOOD)
         LO = HI - HOODCPUMNY(HOOD) + 1
         DO II = LO,HI
           CPU = HOODCPULST(II)
           WRITE(6,*) '<<HOOD ',HOOD,' HOLDS CPU ',CPU,' NEIGHMAPP ',NEIGHMAPP(CPU)
         ENDDO
        ENDDO
      
      ENDIF  !transpose or not
!----------------------------------------------------
! determine my rank in base communicator
      IF(MYKEY.NE.0) THEN  !not head nodes
       call MPI_Comm_rank(BASECOM,mylocrank,ierr) 
       MYKEYB = mylocrank
       MYKEYBP = MYKEYB+1
       IMYKEYBP = MYKEYBP
!       WRITE(6,*) 'MYNODP ',MYNODP,' MY RANK IN BASECOM IS ',MYKEYB,' IMYKEYPB ',IMYKEYBP
      ENDIF

      IF(QTEST) THEN
! tests
        DATASIZE = 10
        if(.not.allocated(DATAS)) call chmalloc('spaccom.src','setcpustruc','DATAS',DATASIZE,crl=DATAS)
        if(.not.allocated(DATAR)) call chmalloc('spaccom.src','setcpustruc','DATAR',DATASIZE,crl=DATAR)
        DO JJ = 1,DATASIZE
           DATAS(JJ) = MYKEY + MYNHOOD*NUMNODH + JJ
           WRITE(6,*) 'MYNODP ',MYNODP,' JJ ',JJ,' DATAS ',DATAS(JJ)
        ENDDO
! first add everything everywhere
        DATAR =0
        CALL MPI_ALLREDUCE(DATAS,DATAR,DATASIZE,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,STATUS)
        DO JJ = 1,DATASIZE
          WRITE(6,*) 'MYNODP ',MYNODP,' JJ ',JJ,' DATAR-ALL ',DATAR(JJ)
        ENDDO
!
! now add only within the head nodes
        DATAR =0
        IF(MYKEY.EQ.0) THEN
         CALL MPI_ALLREDUCE(DATAS,DATAR,DATASIZE,MPI_DOUBLE_PRECISION,MPI_SUM,HEADCOM,STATUS)
        ENDIF
        DO JJ = 1,DATASIZE
          WRITE(6,*) 'MYNODP ',MYNODP,' JJ ',JJ,' DATAR-HEAD ',DATAR(JJ)
        ENDDO
! now add only within your neighborhood
        DATAR =0
        CALL MPI_ALLREDUCE(DATAS,DATAR,DATASIZE,MPI_DOUBLE_PRECISION,MPI_SUM,HOODCOM,STATUS)
        DO JJ = 1,DATASIZE
          WRITE(6,*) 'MYNODP ',MYNODP,' JJ ',JJ,' DATAR-HOOD ',DATAR(JJ)
        ENDDO
!broadcast
        DATAR =0
        CALL MPI_BCAST(DATAS,DATAR,MPI_DOUBLE_PRECISION,0,HOODCOM,STATUS)
        DO JJ = 1,DATASIZE
          WRITE(6,*) 'MYNODP ',MYNODP,' JJ ',JJ,' DATAR-BCAST ',DATAR(JJ)
        ENDDO
! add within your local base (non-head) nodes
! now add only within your neighborhood
        DATAR =0
        IF(MYKEY.NE.0) THEN
         CALL MPI_ALLREDUCE(DATAS,DATAR,DATASIZE,MPI_DOUBLE_PRECISION,MPI_SUM,BASECOM,STATUS)
        ENDIF
        DO JJ = 1,DATASIZE
          WRITE(6,*) 'MYNODP ',MYNODP,' JJ ',JJ,' DATAR-BASE ',DATAR(JJ)
        ENDDO
      ENDIF

! set up standard node arrays for communication
      call cpu_commarr

! CPUs in my neighborhood, by local rank
    if(allocated(OTHERCPU_HD)) deallocate(OTHERCPU_HD)
    allocate(OTHERCPU_HD(NUMNODH-1))
    if(allocated(RNDOTHMP_HD)) deallocate(RNDOTHMP_HD)
    allocate(RNDOTHMP_HD(NUMNODH-1))

    NOTHER = 0
    do NODE = 1,NUMNODH
     if(NODE.ne.IMYKEYP) then
       NOTHER = NOTHER + 1
       OTHERCPU_HD(NOTHER) = NODE
       RNDOTHMP_HD(NOTHER) = NODE - 1
!       WRITE(6,*) 'MYNODP ',MYNODP,' NOTHER ',NOTHER,' NODEH ',NODE,' RNDOTHMP_HD ',RNDOTHMP_HD(NOTHER)
     endif
    enddo 

! CPUs in my neighborhood, by MPI_COMM_WORLD rank
     if(allocated(OTHERCPU_LN)) deallocate(OTHERCPU_LN)
     allocate(OTHERCPU_LN(NUMNODH-1))
      CNT = 0
      HI = HOODCPUHI(IMYNHDP)
      LO = HI-HOODCPUMNY(IMYNHDP) + 1
       DO II = LO,HI
        NODE = HOODCPULST(II)
        IF(NODE.NE.MYNODP) then
         CNT = CNT + 1
         OTHERCPU_LN(CNT) = NODE 
        ENDIF
       ENDDO

!-------
! this is for testspacsr
!!-------
!       TSQACTUAL = .false.
!       TSNTRIAL = GTRMI(COMLYN,COMLEN,'NTRI',100)
!!      WRITE(6,*) 'MYNODP ',MYNODP,' GOT TO HERE IN TESTSPACSR 0.5'
!       TSNATOMX = GTRMI(COMLYN,COMLEN,'NATO',1000)
!!      WRITE(6,*) 'MYNODP ',MYNODP,' TSNATOMX IS ',TSNATOMX
!       IF(INDXA(COMLYN,COMLEN,'ACTU').GT.0) TSQACTUAL = .TRUE.
!!
!! set up Atom Decomposition for hybrid method
!      CALL SPACAD_SETUP
!      CALL SPACADF_SETUP
!!      WRITE(6,*) 'END OF SETCPUSTRUC'
!!      CALL MPI_BARRIER(MPI_COMM_WORLD,STATUS)
!!      STOP
!!     DO II = 1,NUMNODH
!!       WRITE(6,*) 'MYNODP is ',MYNODP,' IMYNHDP ',IMYNHDP,' LOCNODEP ',II,' LOC2GLBP ',LOC2GLBP(II)
!!     ENDDO
        end subroutine setcpustruc
!-------------------------------------------------------------------------------------------------
   subroutine po_comm_r(SENDAR,SENDSZ,RECVAR,HOODCOM,NUMNODH,IMYKEYP,HEADCOM,NUMNOD)
! post office algorithm for reals
! breaks single all-to-all into three steps involving neighborhoods
! reduces the number of messages overall and can speed communication
! relative to a single allgather, particularly on large numbers of cpus
!                                                 --RJP
!   use cpustruc
   use memory
   implicit none
   real(chm_real),intent(in),dimension(:) :: SENDAR
   real(chm_real),intent(in),dimension(:) :: RECVAR
   integer,intent(in) :: SENDSZ
! these are local variables in this context:
   integer(chm_int4),intent(in) :: HOODCOM !communicator within a neighborhood
   integer(chm_int4),intent(in) :: HEADCOM !communicator between head nodes of neighborhoods
   integer,intent(in) :: NUMNODH  !number of nodes in a neighborhood
   integer,intent(in) :: NUMNOD  !number of nodes in all neighborhoods
   integer,intent(in) :: IMYKEYP !rank within neighborhood communicator
!
   include 'mpif.h'

 !locals
   real(chm_real),allocatable,dimension(:) :: RECVARH
   integer :: II
   integer(chm_int4) :: ierr

   if(allocated(RECVARH)) call chmdealloc('paral4.src','po_comm_r','RECVARH',NUMNODH*SENDSZ,crl=RECVARH)
   call chmalloc('paral4.src','po_comm_r','RECVARH',NUMNODH*SENDSZ,crl=RECVARH)
!   allocate(RECVARH(NUMNODH*SENDSZ))
   RECVARH = 0

   call MPI_GATHER(SENDAR,SENDSZ,MPI_DOUBLE_PRECISION,RECVARH,SENDSZ,MPI_DOUBLE_PRECISION,0,HOODCOM,ierr)

   if(IMYKEYP.eq.1) then

    call MPI_ALLGATHER(RECVARH,NUMNODH*SENDSZ,MPI_DOUBLE_PRECISION,RECVAR,NUMNODH*SENDSZ, &
     MPI_DOUBLE_PRECISION,HEADCOM,ierr)

   endif
   call chmdealloc('paral4.src','po_comm_r','RECVARH',NUMNODH*SENDSZ,crl=RECVARH)
!deallocate(RECVARH)

   call MPI_BCAST(RECVAR,NUMNOD*SENDSZ,MPI_DOUBLE_PRECISION,0,HOODCOM,ierr)

   end subroutine po_comm_r

!-------------------------------------------------------------------------------------------------
   subroutine po_comm_i(SENDAR,SENDSZ,RECVAR,HOODCOM,NUMNODH,IMYKEYP,HEADCOM,NUMNOD)
!post office algorithm for integers
!   use cpustruc 
   implicit none
   integer,intent(in),dimension(:) :: SENDAR
   integer,intent(in),dimension(:) :: RECVAR
   integer,intent(in) :: SENDSZ 
! the following are local variables here:
   integer(chm_int4),intent(in) :: HOODCOM !communicator within a neighborhood
   integer(chm_int4),intent(in) :: HEADCOM !communicator between head nodes of neighborhoods
   integer,intent(in) :: NUMNODH  !number of nodes in a neighborhood
   integer,intent(in) :: NUMNOD  !number of nodes in all neighborhoods
   integer,intent(in) :: IMYKEYP !rank within neighborhood communicator

   include 'mpif.h'

 !locals
   integer,allocatable,dimension(:) :: RECVARH
   integer :: II
   integer(chm_int4) :: ierr

   allocate(RECVARH(NUMNODH*SENDSZ))   
   RECVARH = 0

   call MPI_GATHER(SENDAR,SENDSZ,MPI_INTEGER,RECVARH,SENDSZ,MPI_INTEGER,0,HOODCOM,ierr)

   if(IMYKEYP.eq.1) then
   
    call MPI_ALLGATHER(RECVARH,NUMNODH*SENDSZ,MPI_INTEGER,RECVAR,NUMNODH*SENDSZ, &
     MPI_INTEGER,HEADCOM,ierr) 
 
   endif
   deallocate(RECVARH)

   call MPI_BCAST(RECVAR,NUMNOD*SENDSZ,MPI_INTEGER,0,HOODCOM,ierr)
   
   end subroutine po_comm_i
!-------------------------------------------------------------------------------------------------
   subroutine test_po
! test of post office algorithm   
   use cpustruc
   use string,only: GTRMI
   use comand
   use parallel,only: MYNODP,NUMNOD
   integer :: BASE, NODE,SZ,II,JJ,KK
   real(chm_real),allocatable,dimension(:) :: SENDAR
   real(chm_real),allocatable,dimension(:) :: RECVAR
   integer,allocatable,dimension(:) :: SENDARI
   integer,allocatable,dimension(:) :: RECVARI
   
   SZ = GTRMI(COMLYN,COMLEN,'SIZE',10)
   allocate(SENDAR(SZ))

   do II = 1,SZ
    SENDAR(II) =  REAL((MYNODP-1)*SZ+II-1)
    WRITE(6,*) 'MYNODP ',MYNODP,' SENDAR ',SENDAR(II) 
   enddo

   allocate(RECVAR(SZ*NUMNOD))
   RECVAR = 0
   call po_comm_r(SENDAR,SZ,RECVAR,HOODCOM,NUMNODH,IMYKEYP,HEADCOM,NUMNOD)

   do II = 1,NUMNOD
    do JJ = 1,SZ
      KK = (II-1)*SZ+JJ
!     WRITE(6,*) 'MYNODP ',MYNODP,' HOOD ',II,' NUM ',KK,' RECVAR ',RECVAR(KK)
     WRITE(MYNODP+100,*) 'MYNODP ',MYNODP,' HOOD ',II,' NUM ',KK,' RECVAR ',RECVAR(KK)
    enddo
   enddo

   deallocate(SENDAR)
   deallocate(RECVAR)

   allocate(SENDARI(SZ))
   do II = 1,SZ
    SENDARI(II) =  (MYNODP-1)*SZ+II-1
    WRITE(6,*) 'MYNODP ',MYNODP,' SENDARI ',SENDARI(II)
   enddo

   allocate(RECVARI(SZ*NUMNOD))
   RECVARI = 0
   call po_comm_i(SENDARI,SZ,RECVARI,HOODCOM,NUMNODH,IMYKEYP,HEADCOM,NUMNOD)

   do II = 1,NUMNOD
    do JJ = 1,SZ
      KK = (II-1)*SZ+JJ
     WRITE(200+mynodp,*) 'MYNODP ',MYNODP,' HOOD ',II,' NUM ',KK,' RECVARI ',RECVARI(KK)
    enddo
   enddo

   deallocate(SENDARI)
   deallocate(RECVARI)
  
   end subroutine test_po

   subroutine cpu_commarr
! creates some general arrays for communication
   use cpustruc
   implicit none
   integer NODE,NOTHER
! set up standard node arrays for communication
     if(allocated(OTHERCPUS)) deallocate(OTHERCPUS)
     allocate(OTHERCPUS(NUMNOD-1))
     if(allocated(CPUIDENT))  deallocate(CPUIDENT)
     allocate(CPUIDENT(NUMNOD))
     if(allocated(CPUONES)) deallocate(CPUONES)
     allocate(CPUONES(NUMNOD))
     NOTHER = 0
     do NODE = 1,NUMNOD
       CPUIDENT(NODE) = NODE
       CPUONES(NODE) = 1
       if(NODE.NE.MYNODP) then
        NOTHER = NOTHER + 1
        OTHERCPUS(NOTHER) = NODE
       endif
     enddo
    QCPUONES=.true.
   end subroutine cpu_commarr

#endif /*paral4*/

end module paral4
