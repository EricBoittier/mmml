  module async_util2
  use cpustruc
 
#if KEY_PARALLEL==1
   contains

   subroutine exlistas(NTOTCPU,NSELCPU,CPULIST,MYNODPX,LISTBYCPU,LISTHI,LISTMNY, &
     RLISTBYCPU,RLISTHI,RLISTMNY)
!
! this routine exchanges integer lists (e.g. atom numbers) between CPUS, with non-blocking calls.
! it first exchanges the number of atoms in the lists; then it exchanges the lists themselves.
   use asynccomg 
   use chm_types, only: arofar_i4,arofar_i8 
   use memory
   use parallel, only: gcomb
!LMYSIMATM_L   !temporary  
   implicit none
   integer,intent(in),dimension(:) :: CPULIST !list of selected cpus
   integer,intent(in),dimension(:) :: LISTBYCPU,LISTHI,LISTMNY
   integer,intent(out),dimension(:) :: RLISTBYCPU,RLISTHI,RLISTMNY !send/rec lists
   integer,intent(in) ::  NTOTCPU,NSELCPU,MYNODPX  !tot # of cpus in indices of LISTHI, RLISTHI; # cpus selected
!    
! local
   integer :: IBEG,IEND,II,NODE,CNT,ATM
   integer :: RRND,SRND,ADD,CPU
   integer,allocatable,dimension(:) :: RNODAR,SNODAR  
   integer(chm_int4),allocatable,dimension(:),save :: LOCSHAND,LOCRHAND
#if KEY_INTEGER8==1
   type(arofar_i8),allocatable,dimension(:),save :: LOCRBUF,LOCSBUF
#else /**/
   type(arofar_i4),allocatable,dimension(:),save :: LOCRBUF,LOCSBUF
#endif 
!   integer,allocatable,dimension(:),save :: NULLRECV
   integer,dimension(1),save :: TEST1AR !temporary
   logical :: QTESTING !temporary
   integer,allocatable,dimension(:) :: CPU1,CPUM
! end of declarations
!---------------------------------------------------------------------------------------
   call chmalloc('async_util.src','exlistas','CPUM',NTOTCPU,intg=CPUM)
   call chmalloc('async_util.src','exlistas','CPU1',NTOTCPU,intg=CPU1)
   call chmalloc('async_util.src','exlistas','RNODAR',NTOTCPU,intg=RNODAR)  !overallocated
   call chmalloc('async_util.src','exlistas','SNODAR',NTOTCPU,intg=SNODAR)  !overallocated
!
   do CPU = 1,NTOTCPU
    CPUM(CPU) = CPU
    CPU1(CPU) = 1
   enddo
   qtesting = .true.  !temporary
   if(NTOTCPU.lt.1) THEN
     WRITE(6,*) '<async_exlist> ERROR: BAD NUMBER OF CPUS '
     stop
   endif
! exchange lengths     
   CALL ASYNCLST_SG(NSELCPU,CPULIST,LISTMNY,CPUM,CPU1,LOCSHAND,  &
#if KEY_INTEGER8==1
    psi8buf=LOCSBUF,plabint=5000) 
#endif
#if KEY_INTEGER8==0
    psibuf=LOCSBUF,plabint=5000) 
#endif

   CALL ASYNCLST_RG(NSELCPU,CPULIST,CPU1,LOCRHAND, &
#if KEY_INTEGER8==1
   pri8buf=LOCRBUF,plabint=5010) 
#endif
#if KEY_INTEGER8==0
   pribuf=LOCRBUF,plabint=5010) 
#endif

   WRITE(6,*) 'calling ASYNCLST_WRG from exlistas' 
   CALL ASYNCLST_WRG(NSELCPU,CPULIST,RLISTMNY,CPUM,CPU1,LOCRHAND,  &
#if KEY_INTEGER8==1
    pri8buf=LOCRBUF,plabint=5020) 
#endif
#if KEY_INTEGER8==0
    pribuf=LOCRBUF,plabint=5020) 
#endif

   CALL ASYNCLST_WSG(NSELCPU,CPULIST,CPU1,LOCSHAND, &
#if KEY_INTEGER8==1
     psi8buf=LOCSBUF) 
#endif
#if KEY_INTEGER8==0
     psibuf=LOCSBUF) 
#endif

   CNT = 0
   do NODE = 1, NTOTCPU
     if(NODE.NE.MYNODPX) THEN
     CNT = CNT + RLISTMNY(NODE)
     endif
   enddo
!   WRITE(6,*) 'RRMYNODPX ',MYNODPX,' my image atoms ',NMYSIMATM_L,' my total atoms ',NMYSATM_L
   WRITE(6,*) 'RRMYNODPX ',MYNODPX,' RECEIVING TOTAL ',CNT,' ATOMS, plus self-partners = ', &
     CNT + LISTMNY(MYNODPX)
!, ' NMYSIMATM_L ',NMYSIMATM_L
   QTESTING = .true.
   if(QTESTING) THEN
    TEST1AR(1) = CNT + LISTMNY(MYNODPX)
    CALL GCOMB(TEST1AR,1)
    WRITE(6,*) 'TOTAL RECEIVED IMAGES + SELF-PARTNERS ',TEST1AR(1)
   endif
   
   CNT = 0
   RRND = 0
   SRND = 0
   do NODE = 1,NTOTCPU
    ADD = RLISTMNY(NODE)
!receive node array
    if((ADD.gt.0).and.(NODE.ne.MYNODPX)) then
     RRND = RRND + 1
     RNODAR(RRND) = NODE
    endif
    CNT = CNT + ADD
    RLISTHI(NODE) = CNT
!   WRITE(6,*) 'RRMYNODPX ',MYNODPX,' set to rec ',RLISTMNY(NODE),' atoms fr cpu ',NODE
! send node array
    if((NODE.ne.MYNODPX).and.(LISTMNY(NODE).ne.0)) then
     SRND = SRND + 1
     SNODAR(SRND) = NODE
    endif
   enddo
   WRITE(6,*) 'TTTMYNODPX ',MYNODPX,' REC TOTAL ',CNT,' IMPATMS for tot of ',CNT+LISTMNY(MYNODPX)
   if(RRND.lt.1) THEN
     WRITE(6,*) '<async_exlist> ERROR: BAD NUMBER OF RECEIVE CPUS '
     stop
   endif
   if(SRND.lt.1) THEN
     WRITE(6,*) '<async_exlist> ERROR: BAD NUMBER OF SEND CPUS '
     stop
   endif

   CNT = 0
   do NODE = 1,NTOTCPU
     IEND = RLISTHI(NODE)
     IBEG = IEND - RLISTMNY(NODE) + 1
     do II = IBEG,IEND
       CNT = CNT + 1
      ATM = RLISTBYCPU(II)
!      WRITE(6,*) '77MYNODPX ',MYNODPX,' RECEIVING ATM ',ATM,' fr node ',NODE
     enddo
   enddo
   WRITE(6,*) 'YYMYNODPX ',MYNODPX,' TOTAL ATMS in RLISTBYCPUs ',CNT,' PLUS my own ', &
    CNT+LISTMNY(MYNODPX)
!   if(QTESTING) THEN
!    if(CNT+LISTMNY(MYNODPX).NE.NMYSIMATM_L ) then
!      WRITE(6,*) '<NBNDCCO> # OF IMG ATOM PARTNRS RECEIVD + INTRINSCS ',CNT, 'DOES NOT MATCH # OF IMG ATOMS ', &
!       NMYSIMATM_L
!      STOP
!    endif
!   endif

! now exchange lists 

   CALL ASYNCLST_SG(SRND,SNODAR,LISTBYCPU,LISTHI,LISTMNY,LOCSHAND, &
#if KEY_INTEGER8==1
    psi8buf=LOCSBUF,plabint=5100) 
#endif
#if KEY_INTEGER8==0
    psibuf=LOCSBUF,plabint=5100) 
#endif

!   WRITE(6,*) '&&MYNODP_O2 ',MYNODPX,' HOODP ',IMYNHDP,' SHAND ',LOCSHAND(1)
   CALL ASYNCLST_RG(RRND,RNODAR,RLISTMNY,LOCRHAND, &
#if KEY_INTEGER8==1
   pri8buf=LOCRBUF,plabint=5010) 
#endif
#if KEY_INTEGER8==0
   pribuf=LOCRBUF,plabint=5010) 
#endif

!   WRITE(6,*) '&&MYNODP_O3 ',MYNODPX,' HOODP ',IMYNHDP,' RHAND ',LOCRHAND(1)
   WRITE(6,*) 'calling ASYNCLST_WRG from exlistas' 
   CALL ASYNCLST_WRG(RRND,RNODAR,RLISTBYCPU,RLISTHI,RLISTMNY,LOCRHAND, &
#if KEY_INTEGER8==1
    pri8buf=LOCRBUF,plabint=5120) 
#endif
#if KEY_INTEGER8==0
    pribuf=LOCRBUF,plabint=5120) 
#endif

   CALL ASYNCLST_WSG(SRND,SNODAR,LISTMNY,LOCSHAND, &
#if KEY_INTEGER8==1
     psi8buf=LOCSBUF) 
#endif
#if KEY_INTEGER8==0
     psibuf=LOCSBUF) 
#endif

   CNT = 0
   do NODE = 1,NTOTCPU
     IEND = RLISTHI(NODE)
     IBEG = IEND - RLISTMNY(NODE) + 1
     do II = IBEG,IEND
      CNT = CNT + 1
      ATM = RLISTBYCPU(II)
!      WRITE(6,*) '77MYNODP ',MYNODPX,' RECEIVING ATM ',ATM,' fr node ',NODE
     enddo
   enddo
   WRITE(6,*) 'YYMYNODP ',MYNODPX,' TOTAL ATMS in RLISTBYCPUs ',CNT,' PLUS my own ', &
    CNT+LISTMNY(MYNODPX)
!   if(QTESTING) THEN
!    if(CNT+LISTMNY(MYNODPX).NE.NMYSIMATM_L ) then
!      WRITE(6,*) '<NBNDCCO> # OF IMG ATOM PARTNRS RECEIVD + INTRINSCS ',CNT, 'DOES NOT MATCH # OF IMG ATOMS ', &
!       NMYSIMATM_L
!      STOP
!    endif
!   endif
   call chmdealloc('async_util.src','exlistas','CPUM',NTOTCPU,intg=CPUM)
   call chmdealloc('async_util.src','exlistas','CPU1',NTOTCPU,intg=CPU1)
   call chmdealloc('async_util.src','exlistas','RNODAR',NTOTCPU,intg=RNODAR)  !overallocated
   call chmdealloc('async_util.src','exlistas','SNODAR',NTOTCPU,intg=SNODAR)  !overallocated
   return 
   end subroutine exlistas 
!----------------------------------------------------------------------------------------------
!----------------------------------------------------------------------------------------------
!----------------------------------------------------------------------------------------------
  subroutine cleannull(NROUND,NODARR,LENGTH,NROUNDN,NODARRN,NMAP,NMAPN)
   use memory
   implicit none
   integer,intent(in) :: NROUND
   integer,intent(in),dimension(:) :: NODARR,LENGTH
   integer,intent(inout),allocatable,dimension(:) :: NODARRN
   integer,intent(out) :: NROUNDN
   integer,intent(in),dimension(:),optional :: NMAP
   integer,intent(inout),allocatable,dimension(:),optional :: NMAPN

! local
   integer :: ROUND,TESTND
! **********end of declarations ********************************
   if(((present(NMAP)).and..not.present(NMAPN)).or. &
    ((present(NMAPN)).and..not.present(NMAP))) then 
    write(6,*) '<CLEANNULL> ERROR: must pass both new and old arrays to utility' 
   endif
   NROUNDN = 0
   do ROUND = 1,NROUND
     TESTND = NODARR(ROUND)
      if (TESTND.LE.0) CYCLE ! wipe out rounds with no receive node
!      if(TESTND.lt.0) then
!       WRITE(6,*) 'BAD SENDNODE ON CPU ',MYNODP,' ROUND ',ROUND,' SENDND ',SENDND
!       STOP
!       CALL WRNDIE(-5,'<ASYNCLST_SG>','RANKP < 0, BAD SEND NODE')
!      endif
      if(LENGTH(TESTND).le.0) CYCLE   ! wipe out rounds with zero-length data
     NROUNDN = NROUNDN + 1
   enddo
   if (allocated(NODARRN)) call chmdealloc('async_util.src','cleannull','NODARRN',NROUNDN,intg=NODARRN)
   call chmalloc('async_util.src','cleannull','NODARRN',NROUNDN,intg=NODARRN)
   if(present(NMAPN)) then
    if (allocated(NMAPN)) call chmdealloc('async_util.src','cleannull','NMAPN',NROUNDN,intg=NMAPN)
    call chmalloc('async_util.src','cleannull','NMAPN',NROUNDN,intg=NMAPN)
   endif
   NROUNDN = 0
   do ROUND = 1,NROUND
     TESTND = NODARR(ROUND)
     if (TESTND.LE.0) CYCLE ! wipe out rounds with no receive node
     if(LENGTH(TESTND).le.0) CYCLE   ! wipe out rounds with zero-length data
     NROUNDN = NROUNDN + 1
     NODARRN(NROUNDN)=NODARR(ROUND) 
    if(present(NMAP))  NMAPN(NROUNDN)=NMAP(ROUND)
   enddo
  end subroutine cleannull
!----------------------------------------------------------------------------------
!----------------------------------------------------------------------------------
  subroutine CPUEXCHINT(NUMNODGR,OTHCPUS,MYINTEGER,NUMNODES,OTHERSINT)
! exchange single integers among a group of CPUS  
! expects communication between all pairwise combinations of CPUs in the group
!                                            --RJP 7/2011
  use chm_types,only: arofar_i4,arofar_i8
  use parallel,only: MYNODP !temporary
  use memory
  use asynccomg
  implicit none
  integer,dimension(:),intent(in) :: OTHCPUS  ! list of other cpus in the group
  integer,intent(in) :: NUMNODGR  !number of cpus in this group (not including me)
  integer,intent(in) :: MYINTEGER  
  integer,intent(in) :: NUMNODES  !number of nodes in system (dimension of OTHERSINT)
  integer,allocatable,dimension(:),intent(out) :: OTHERSINT !integers received from others
!local
  integer,allocatable,dimension(:) :: NMYATTMP
  integer :: II,NODE
#if KEY_INTEGER8==1
  type(arofar_i8),allocatable,dimension(:),save :: LOCRBUF,LOCSBUF
#else /**/
  type(arofar_i4),allocatable,dimension(:),save :: LOCRBUF,LOCSBUF
#endif 
  integer(chm_int4),allocatable,dimension(:) :: LOCSHAND,LOCRHAND
!  integer,allocatable,dimension(:) :: NULLRECV
  include 'mpif.h'  !temporary
  integer(chm_int4) :: status !temporary
! end of decl

  call chmalloc('nbndcco.src','NBNDCCO','NMYATTMP',1,intg=NMYATTMP)
  call chmalloc('nbndcco.src','NBNDCCO','OTHERSINT',NUMNODES,intg=OTHERSINT)

  OTHERSINT = 0
  NMYATTMP = MYINTEGER  !scalar=> array

  call ASYNCLST_RG(NUMNODGR,OTHCPUS,CPUONES,LOCRHAND, &
#if KEY_INTEGER8==1
  pri8buf=LOCRBUF,plabint=2010) 
#endif
#if KEY_INTEGER8==0
  pribuf=LOCRBUF,plabint=2010) 
#endif

  call ASYNCLST_SG(NUMNODGR,OTHCPUS,NMYATTMP,CPUONES,CPUONES,LOCSHAND,  &
#if KEY_INTEGER8==1
   psi8buf=LOCSBUF,plabint=2000) 
#endif
#if KEY_INTEGER8==0
   psibuf=LOCSBUF,plabint=2000) 
#endif

  call ASYNCLST_WRG(NUMNODGR,OTHCPUS,OTHERSINT,CPUIDENT,CPUONES,LOCRHAND,  &
#if KEY_INTEGER8==1
  pri8buf=LOCRBUF,plabint=2020) 
#endif
#if KEY_INTEGER8==0
  pribuf=LOCRBUF,plabint=2020) 
#endif

!  WRITE(6,*) 'MYNODP ',MYNODP,' GOT TO HERE 3'
  call ASYNCLST_WSG(NUMNODGR,OTHCPUS,CPUONES,LOCSHAND, &
#if KEY_INTEGER8==1
    psi8buf=LOCSBUF) 
#endif
#if KEY_INTEGER8==0
    psibuf=LOCSBUF) 
#endif

  do II = 1,NUMNODGR
   NODE = OTHCPUS(II)
!   WRITE(6,*) 'MYNODP ',MYNODP,' WILL REC ',OTHERSINT(NODE),' atoms from node ',NODE
  enddo

  call chmdealloc('nbndcco.src','NBNDCCO','NMYATTMP',1,intg=NMYATTMP)
  call chmdealloc('nbndcco.src','NBNDCCO','OTHERSINT',NUMNODES,intg=OTHERSINT)

  end subroutine CPUEXCHINT

  subroutine CPUEXCHINT2(NCPUSEND,CPUSEND,NCPURECV,CPURECV,MYINTEGER,NUMNODES,OTHERSINT)
! exchange single integers among CPUS, given a send group and receive group (of CPUs)
!                                            --RJP 7/2011
  use chm_types,only : arofar_i4,arofar_i8
  use parallel,only: MYNODP !temporary
!  use async_util,only: cleannull
  use memory
  use asynccomg
  implicit none
  integer,dimension(:),intent(in) :: CPUSEND,CPURECV  ! list of other cpus in the group
  integer,intent(in) :: NCPUSEND,NCPURECV  !number of cpus in this group (not including me)
  integer,intent(in) :: MYINTEGER  
  integer,intent(in) :: NUMNODES  !number of nodes in system (dimension of OTHERSINT)
  integer,allocatable,dimension(:),intent(out) :: OTHERSINT !integers received from others
!local
  integer,allocatable,dimension(:) :: NMYATTMP
  integer :: II,NODE
#if KEY_INTEGER8==1
  type(arofar_i8),allocatable,dimension(:),save :: LOCRBUF,LOCSBUF
#else /**/
  type(arofar_i4),allocatable,dimension(:),save :: LOCRBUF,LOCSBUF
#endif 
  integer(chm_int4),allocatable,dimension(:) :: LOCSHAND,LOCRHAND
!  integer,allocatable,dimension(:) :: NULLRECV
  include 'mpif.h'  !temporary
  integer(chm_int4) :: status !temporary
  integer,dimension(:),allocatable,save :: RNODARR,SNODARR
  integer :: NRNODARR,NSNODARR
! end of decl

!  call chmalloc('nbndcco.src','NBNDCCO','NMYATTMP',1,intg=NMYATTMP)
  call chmalloc('nbndcco.src','NBNDCCO','NMYATTMP',NUMNODES,intg=NMYATTMP)
  if(allocated(OTHERSINT)) call chmdealloc('nbndcco.src','NBNDCCO','OTHERSINT',NUMNODES,intg=OTHERSINT)
  call chmalloc('nbndcco.src','NBNDCCO','OTHERSINT',NUMNODES,intg=OTHERSINT)

  OTHERSINT = 0
  NMYATTMP = MYINTEGER  !scalar=> array

!  WRITE(6,*) 'CPUONES AT 1 is ',CPUONES(1)
!  WRITE(6,*) 'CPUONES AT 2 is ',CPUONES(2)
!  WRITE(6,*) 'CPUONES AT 3 is ',CPUONES(3)
!  WRITE(6,*) 'CPUONES AT 4 is ',CPUONES(4)
!  WRITE(6,*) 'NCPUSEND ',NCPUSEND,' NCPURECV ',NCPURECV

  call cleannull(NCPURECV,CPURECV,CPUONES,NRNODARR,RNODARR)
  call cleannull(NCPUSEND,CPUSEND,CPUONES,NSNODARR,SNODARR)

!  call parstoperr('<cpuexchint2>',' succesfully past cleannulls')
  if(MYNODP.eq.1) then
    WRITE(6,*) 'NODE 1 in cpuexchint, NCPURECV is ',NCPURECV,' NCPUSEND ',NCPUSEND
  endif

  call ASYNCLST_RG(NRNODARR,RNODARR,CPUONES,LOCRHAND, &
#if KEY_INTEGER8==1
  pri8buf=LOCRBUF,plabint=2110) 
#endif
#if KEY_INTEGER8==0
  pribuf=LOCRBUF,plabint=2110) 
#endif

  call ASYNCLST_SG(NSNODARR,SNODARR,NMYATTMP,CPUONES,CPUONES,LOCSHAND,  &
#if KEY_INTEGER8==1
   psi8buf=LOCSBUF,plabint=2100) 
#endif
#if KEY_INTEGER8==0
   psibuf=LOCSBUF,plabint=2100) 
#endif

  call ASYNCLST_WRG(NRNODARR,RNODARR,OTHERSINT,CPUIDENT,CPUONES,LOCRHAND,  &
#if KEY_INTEGER8==1
   pri8buf=LOCRBUF,plabint=2120) 
#endif
#if KEY_INTEGER8==0
   pribuf=LOCRBUF,plabint=2120) 
#endif

  call ASYNCLST_WSG(NSNODARR,SNODARR,CPUONES,LOCSHAND, &
#if KEY_INTEGER8==1
     psi8buf=LOCSBUF) 
#endif
#if KEY_INTEGER8==0
     psibuf=LOCSBUF) 
#endif

  do II = 1,NRNODARR
   NODE = RNODARR(II)
!   WRITE(6,*) 'MYNODP ',MYNODP,' WILL REC ',OTHERSINT(NODE),' atoms from node ',NODE
  enddo

!  call chmdealloc('nbndcco.src','NBNDCCO','NMYATTMP',1,intg=NMYATTMP)
  call chmdealloc('nbndcco.src','NBNDCCO','NMYATTMP',NUMNODES,intg=NMYATTMP)

  end subroutine CPUEXCHINT2

  subroutine CPUEXCHLIST(NCPUSEND,CPUSEND,SENDLIST,SENDHI,SENDMNY, &
  NCPURECV,CPURECV,RECVHI,RECVMNY,NUMNODES,RECVLIST,PCCATOR,PSXX,PSYY,PSZZ,PRXX,PRYY,PRZZ,plabint)
! exchange lists of integers among CPUs, given a send group and receive group of CPUS  
! optionally, 3 real arrays may be passed (PSXX, PSYY, PSZZ; e.g. coordinates).
! As an additional option, separate send and receive (PRXX,...) arrays may be used.
!   
!                                            --RJP  7/2011
  use chm_types,only: arofar_i4,arofar,arofar_i8
  use parallel,only: MYNODP !temporary
  use memory
  use asynccomg
!  use async_util,only: cleannull
  implicit none
  integer,dimension(:),intent(in) :: CPUSEND,CPURECV  ! list of other cpus in the group
  integer,intent(in) :: NCPUSEND,NCPURECV  !number of cpus in this group (not including me)
  integer,dimension(:),intent(in) :: SENDLIST,SENDHI,SENDMNY  ! list of integers to send
  integer,dimension(:),intent(in) :: RECVMNY  ! length of list blocks to receive
  integer,dimension(:),intent(out) :: RECVLIST,RECVHI  !recvhi could probably be eliminated
  integer,intent(in) :: NUMNODES  !number of nodes in ccator (dimension of OTHERSINT)
  integer(chm_int4),intent(in),optional :: PCCATOR  !communicator
  integer(chm_int4),intent(in),optional :: plabint  !communicator
  real(chm_real),intent(inout),optional,dimension(:),target :: PSXX,PSYY,PSZZ
  real(chm_real),intent(inout),optional,dimension(:),target :: PRXX,PRYY,PRZZ
!local
  integer :: II,NODE,PT,PX,PY,PZ
  type(arofar),allocatable,dimension(:),save :: LOCSBUFR,LOCRBUFR
#if KEY_INTEGER8==1
  type(arofar_i8),allocatable,dimension(:),save :: LOCRBUF,LOCSBUF
#else /**/
  type(arofar_i4),allocatable,dimension(:),save :: LOCRBUF,LOCSBUF
#endif 
  integer(chm_int4),allocatable,dimension(:) :: LOCSHAND,LOCRHAND
!  integer,allocatable,dimension(:) :: NULLRECV
  include 'mpif.h'  !temporary
  integer(chm_int4) :: status !temporary
  integer(chm_int4) :: ccator  !communicator
  integer :: NODE2 !temporary
  logical :: QXYZDATA,QXYZDATA2
  real(chm_real),pointer,dimension(:) :: RECVX,RECVY,RECVZ
  integer,dimension(:),allocatable,save :: RNODARR,SNODARR
  integer :: NRNODARR,NSNODARR
  integer :: labint
! end of decl
  
  ccator = MPI_COMM_WORLD
  if(present(pccator)) ccator = pccator
!  WRITE(6,*) 'MYNODP ',MYNODP,' in CPUEXCHLST , NCPURECV is ',NCPURECV, ' NCPUSEND is ',NCPUSEND
!  WRITE(6,*) 'MYNODP ',MYNODP,' in CPUEXCHLST , SENDMNY(1) is ',SENDMNY(1)

  QXYZDATA = .false.
  !process send x,y,z
  PX = 0
  if(present(PSXX)) PX = 1
  PY = 0
  if(present(PSYY)) PY = 1
  PZ = 0
  if(present(PSZZ)) PZ = 1
  PT = PX+PY+PZ
  if(PT.GT.0) then
    if (PT.LT.3) then
     WRITE(6,*) '<CPUEXCHLIST> ERROR: if passing X,Y,Z, must pass all three arrays'
     STOP
    else
     QXYZDATA = .true.
    endif
  endif

  QXYZDATA2 = .false.
  !process recv x,y,z
  PX = 0
  if(present(PRXX)) PX = 1
  PY = 0
  if(present(PRYY)) PY = 1
  PZ = 0
  if(present(PRZZ)) PZ = 1
  PT = PX+PY+PZ
  if(PT.GT.0) then
    if (PT.LT.3) then
     WRITE(6,*) '<CPUEXCHLIST> ERROR: if passing X,Y,Z, must pass all three arrays'
     STOP
    else
     QXYZDATA2 = .true.
    endif
  endif

  if((QXYZDATA2).and..not.(QXYZDATA)) then
    WRITE(6,*) '<CPUEXCHLIST> ERROR: must pass either send or both send and receive x,y,z buffers'
    STOP
  endif

 if(MYNODP.eq.9) then
  do II = 1,NCPUSEND
    NODE = CPUSEND(II)
   WRITE(6,*) '>>>exchlist node 9 sending ',SENDMNY(NODE),' to node ',NODE,' NCPUSEND ',NCPUSEND
  enddo
 endif

 if(MYNODP.eq.1) then
  do II = 1,NCPUSEND
    NODE = CPUSEND(II)
   WRITE(6,*) '>>>exchlist node 1 sending ',SENDMNY(NODE),' to node ',NODE,' NCPUSEND ',NCPUSEND
  enddo
 endif

  call cleannull(NCPURECV,CPURECV,RECVMNY,NRNODARR,RNODARR)
  call cleannull(NCPUSEND,CPUSEND,SENDMNY,NSNODARR,SNODARR)

  if(.not.QXYZDATA) then

  labint = 2220
  if(present(plabint)) labint=plabint
   call ASYNCLST_RG(NRNODARR,RNODARR,RECVMNY,LOCRHAND, &
#if KEY_INTEGER8==1
   pri8buf=LOCRBUF,pccator=CCATOR,plabint=2210) 
#endif
#if KEY_INTEGER8==0
   pribuf=LOCRBUF,pccator=CCATOR,plabint=2210) 
#endif

   call ASYNCLST_SG(NSNODARR,SNODARR,SENDLIST,SENDHI,SENDMNY,LOCSHAND,  &
#if KEY_INTEGER8==1
    psi8buf=LOCSBUF,pccator=CCATOR,plabint=2200) 
#endif
#if KEY_INTEGER8==0
    psibuf=LOCSBUF,pccator=CCATOR,plabint=2200) 
#endif

   call ASYNCLST_WRG(NRNODARR,RNODARR,RECVLIST,RECVHI,RECVMNY,LOCRHAND,  &
#if KEY_INTEGER8==1
    pri8buf=LOCRBUF,plabint=labint) 
#endif
#if KEY_INTEGER8==0
    pribuf=LOCRBUF,plabint=labint) 
#endif

!   WRITE(6,*) 'MYNODP ',MYNODP,' calling ASYNCLST_WRG from CPUEXCHLIST' 
   call ASYNCLST_WSG(NSNODARR,SNODARR,SENDMNY,LOCSHAND, &
#if KEY_INTEGER8==1
   psi8buf=LOCSBUF) 
#endif
#if KEY_INTEGER8==0
   psibuf=LOCSBUF) 
#endif

  else !qxyzdata

   RECVX => PSXX
   RECVY => PSYY
   RECVZ => PSZZ
   if(QXYZDATA2) then
    RECVX => PRXX
    RECVY => PRYY
    RECVZ => PRZZ
   endif

   labint = 2420
   if(present(plabint)) labint=plabint

   call ASYNCLST_RG(NRNODARR,RNODARR,RECVMNY, &
   LOCRHAND,prrbuf=LOCRBUFR,pccator=CCATOR,pdatawid=3,plabint=2410)

   call ASYNCLST_SG(NSNODARR,SNODARR,SENDLIST,SENDHI,SENDMNY,  &
    LOCSHAND,psrbuf=LOCSBUFR,pccator=CCATOR,pxx=PSXX,pyy=PSYY,pzz=PSZZ,plabint=2400)

   call ASYNCLST_WRG(NRNODARR,RNODARR,RECVLIST,RECVHI,RECVMNY,  &
    LOCRHAND,prrbuf=LOCRBUFR,pxx=RECVX,pyy=RECVY,pzz=RECVZ,plabint=labint)

   call ASYNCLST_WSG(NSNODARR,SNODARR,SENDMNY,LOCSHAND,psrbuf=LOCSBUFR)
 
!       CALL ASYNCLST_RG(NIMATRRND,IMATRNODAR,YRIMPATM_CPUMNY, &
!      LOCRHAND,prrbuf=LOCRBUFR,pdatawid=3)
!
!       CALL ASYNCLST_SG(NIMATSRND,IMATSNODAR,MYIMPATM_CPU,MYIMPATM_CPUHI,MYIMPATM_CPUMNY, &
!       LOCSHAND,psrbuf=LOCSBUFR,PXX=X,PYY=Y,PZZ=Z)
!
!      CALL ASYNCLST_WRG(NIMATRRND,IMATRNODAR,YRIMPATM_CPU,YRIMPATM_CPUHI,YRIMPATM_CPUMNY, &
!       LOCRHAND,prrbuf=LOCRBUFR,PXX=X,PYY=Y,PZZ=Z)
!
!       CALL ASYNCLST_WSG(NIMATSRND,IMATSNODAR,MYIMPATM_CPUMNY,LOCSHAND,psrbuf=LOCSBUFR)
!
  endif !if qarray or not 

!  do II = 1,NRNODARR
!   NODE = RNODARR(II)
!!   WRITE(6,*) 'MYNODP ',MYNODP,' WILL REC ',RECVMNY(NODE),' atoms from node ',NODE
!  enddo

  end subroutine CPUEXCHLIST

!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
   subroutine mpicomp_send_rec(NNTCPUSEN,NTCPUSEN,NNTCPUREC,NTCPUREC,pqverb,plabint)
! compares sends and receives of all cpus in the system, for testing
! since global communication, slow for large numbers of CPUS
!    --RJP 5/2012
   use parallel,only: NUMNOD,MYNODP
   use nbndcc_utilb,only: parstoperr
   use chm_types
   implicit none 
   integer,dimension(:),allocatable,intent(in) :: NTCPUSEN,NTCPUREC
   integer,intent(in) :: NNTCPUSEN,NNTCPUREC
   integer,optional,intent(in) :: plabint
   logical,optional,intent(in) :: pqverb
   include 'mpif.h'
! local
   integer(chm_int4) :: status
   integer,dimension(:,:),allocatable :: DEBUGRECS,DEBUGSENDS,TOTALREC,TOTALSEN
   integer,dimension(:),allocatable :: WORKREC,WORKSEN
   integer :: NODE,NODE2,II
   integer :: labint
   logical :: qverb
!-------------------------------------------------------------------------------
   qverb = .false.
   if(present(pqverb)) qverb = pqverb
  ! label
   labint = 0
   if(present(plabint)) labint = plabint

   if(MYNODP.EQ.1) WRITE(6,*) '>>>>>>>>>>> mpicomp_send_rec begun <<< label ',plabint
   CALL MPI_BARRIER(MPI_COMM_WORLD,status)

   if(allocated(DEBUGRECS)) deallocate(DEBUGRECS)
   if(allocated(DEBUGSENDS)) deallocate(DEBUGSENDS)
   if(allocated(TOTALREC)) deallocate(TOTALREC)
   if(allocated(TOTALSEN)) deallocate(TOTALSEN)
   if(allocated(WORKREC)) deallocate(WORKREC)
   if(allocated(WORKSEN)) deallocate(WORKSEN)
   allocate(DEBUGRECS(NUMNOD,NUMNOD))
   allocate(DEBUGSENDS(NUMNOD,NUMNOD))
   allocate(TOTALREC(NUMNOD,NUMNOD))
   allocate(TOTALSEN(NUMNOD,NUMNOD))
   allocate(WORKREC(NUMNOD))
   allocate(WORKSEN(NUMNOD))

   DEBUGRECS = 0 !arrays
   DEBUGSENDS = 0
   TOTALREC = 0
   TOTALSEN = 0
   WORKREC = 0
   WORKSEN = 0

   DO II = 1,NNTCPUREC
     if(qverb) WRITE(6,*) '>>COMM IM NODE ',MYNODP,' RECEIVING FROM ',NTCPUREC(II)
     DEBUGRECS(MYNODP,NTCPUREC(II)) = 1
   ENDDO

   DO II = 1,NNTCPUSEN
     if(qverb) WRITE(6,*) '>>COMM IM NODE ',MYNODP,' SENDING TO ',NTCPUSEN(II)
     DEBUGSENDS(MYNODP,NTCPUSEN(II)) = 1
   ENDDO

   DO NODE = 1,NUMNOD
    WORKREC = 0 !array
    WORKSEN = 0 !array
    DO NODE2 = 1,NUMNOD
      WORKREC(NODE2)=DEBUGRECS(NODE,NODE2)
      WORKSEN(NODE2)=DEBUGSENDS(NODE,NODE2)
    ENDDO
    CALL IGCOMB(WORKSEN,NUMNOD)
    CALL IGCOMB(WORKREC,NUMNOD)
    DO NODE2 = 1,NUMNOD
     TOTALREC(NODE,NODE2) = WORKREC(NODE2)
     TOTALSEN(NODE,NODE2) = WORKSEN(NODE2)
    ENDDO
   ENDDO
   if(MYNODP.EQ.1) then
    DO NODE = 1,NUMNOD
     DO NODE2 = 1,NUMNOD
   if(qverb) then
    WRITE(6,*) 'NODE1 NODE2 ',NODE,NODE2,' TOTREC 1,2 ',TOTALREC(NODE,NODE2),' TOTSEN 2,1 ',TOTALSEN(NODE2,NODE)
   endif
      if((TOTALREC(NODE,NODE2).NE.TOTALSEN(NODE2,NODE))) then
       WRITE(6,*) 'ERROR: SENDS DO NOT CORRESPOND WITH RECEIVES'
       WRITE(6,*) 'ERROR: NODE ',NODE,' REC FR NODE ',NODE2,' is ',TOTALREC(NODE,NODE2)
       WRITE(6,*) 'ERROR: NODE ',NODE2,' SEND TO NODE ',NODE,' is ',TOTALSEN(NODE2,NODE)
       call parstoperr('<mpicomp_send_rec>',' discrepancy between sends and receives')
      endif
     ENDDO
    ENDDO
   WRITE(6,*) '>>>>>>>>>>> mpicomp_send_rec passed <<< label ',plabint
    
   endif !if mynodp eq. 1

   end subroutine mpicomp_send_rec
!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------

#endif 
  end module async_util2
!-------------------------------------------------------------------------------
