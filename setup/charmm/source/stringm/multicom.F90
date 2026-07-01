! ********************************************************************************!
! Preprocessed by GNU CPP !
! ********************************************************************************!



!
! code extension that allows users to define communicators manually
! required by string module
!
! NOTE: all routines in this module are meant to be executed on ALL processes
!
!
      module multicom
#if (KEY_STRINGM==1) || (KEY_MULTICOM==1) /* automatically protect all code */
      use ivector ! container for vector of ints
       use stream
!
      implicit none
      private
!
! encapsulated communicator type
!
      type comm_type
       type (int_vector) :: nodes
! had to set types below to int4 to work correctly on 64bit gnu
       integer*4 :: me
       integer*4 :: size
       integer*4 :: comm_id
       integer*4 :: group_id
       integer*4 :: parent_id ! ID of the parent communicator ( N/A to first entry )
      end type comm_type
!
! variables
  character(len=132)::info(21)=(/'','','','','','','','','','','','','','','','','','','','',''/);! output buffer
!
      type (comm_type), pointer, private, save :: communicators(:)=>NULL() ! array of communicators
      integer, save, private :: ncomm=0 ! number of defined communicators
      logical, save, public :: multicom_initialized=.false.
      integer, save, private :: first_me=-1, first_size=0
!
      integer, parameter, private :: mcom_expand = 10
!



!
! subroutines (parallel only)

! access qualifiers



      public multicom_main ! parser
      public multicom_init ! initialization
      private multicom_add ! add communicator
      private multicom_list ! list communicators
      private multicom_set ! set a communicator to a communicator in this module; depends on parallel.fcm
      public multicom_parinit ! reinitialize parallel run (use after communicators have been changed)
      public multicom_cleanup



      public multicom_safe_reset ! set some communicators to safe values
      private multicom_barrier ! call MPI_BARRIER on any of the known communicators

      public multicom_permute_string_ranks

!
      contains






!=====================================================================================
       subroutine multicom_main(comlyn, comlen)
! multicom command parser
       use parselist
       use multicom_aux
!
       use string
       use mpi
!
       character*(*), intent(inout) :: comlyn
       integer, intent(inout) :: comlen
!
! local variables
       character(len=8) :: keyword
       character(len=len("MULTICOM_MAIN>") ),parameter::whoami="MULTICOM_MAIN>";!macro
       integer*4 :: ncpu, me, bug
       integer :: newcomm
       integer :: i, j, ind, klen=0
       integer :: nlocal=-1, nrep=-1
       logical :: qens, qstr
       type (int_vector) :: list
!
       call MPI_COMM_SIZE(MPI_COMM_WORLD, ncpu, bug)
       call MPI_COMM_RANK(MPI_COMM_WORLD, me, bug)
!
       keyword=nexta8(comlyn,comlen)
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       if ( keyword(1:4).eq.'INIT'(1:4) ) then
        if (me.eq.0) &
& call wrndie(0,whoami,trim('REINITIALIZING USER-DEFINED COMMUNICATOR LIST.'))
        call multicom_init()
        call multicom_list()
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       elseif ( keyword(1:3).eq.'ADD'(1:3) ) then
!
! parse node list
        call trima(comlyn, comlen)
        call ilist_parse(list, comlyn)
!
        do i=1,list%last
         if (list%i(i).lt.0.or.list%i(i).ge.ncpu) then
          call wrndie(0,whoami,trim('ONE OR MORE NODES OUT OF RANGE. NOTHING DONE.'))
          call int_vector_done(list)
          return
         endif
!
        enddo
!
        if (list%last.eq.0) then
         call wrndie(0,whoami,trim(' NO NODES SPECIFIED.'))
        else
!
         newcomm=multicom_add(list%i(1:list%last)+1) ! offsetting nodes at 0 and adding 1 here permits users to think of the list as ranks in COMM_WORLD
! ! conceptually, this is incorrect because the indices point into the a node array (indexed from 1)
! ! of the first communicator (on top of WORLD), but I am making this compromise for simplicity
         if (newcomm.lt.0) then
          call wrndie(0,whoami,trim(' COULD NOT ADD COMMUNICATOR.'))
         else
          write(info,222) whoami, newcomm;
          if(prnlev.ge. 3) write(OUTU,'(A)') pack(info,info.ne.'');info=''; ! print if verbosity >= 3
         endif
        endif
        call int_vector_done(list)
 222 FORMAT(/A, ' ADDED COMMUNICATOR ',I4,'.')
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       elseif ( keyword(1:4).eq.'BARR'(1:4) ) then
        keyword=nexta8(comlyn,comlen) ! communicator name, e.g. LOCAL
        call multicom_barrier(keyword)
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       elseif ( keyword(1:3).eq.'SET'(1:3) ) then
        keyword=nexta8(comlyn,comlen) ! communicator name, e.g. LOCAL
! process communicator list (essentially same code as for add; communicator indices start at 1)
! here, inode indexes communicators
        comlyn(comlen+1:)=''; comlyn=adjustl(comlyn); comlen=len_trim(comlyn)
        call ilist_parse(list, comlyn)
! check range
        do i=1, list%last
         if (list%i(i).lt.1.or.list%i(i).gt.ncomm) then
          call wrndie(0,whoami,trim(' ONE OR MORE COMMUNICATORS OUT OF RANGE. NOTHING DONE.'))
          call int_vector_done(list)
          return
         endif
        enddo ! check
!
        if (list%last.eq.0) then
         call wrndie(0,whoami,trim(' NO COMMUNICATORS SPECIFIED.'))
        else
         call multicom_set(keyword, list%i(1:list%last)) ! pass subset of array -- only the communnicators
! warn if LOCAL modified
         if (len(keyword).ge.5) then
          if (keyword(1:5).eq.'LOCAL'.or.keyword(1:5).eq.'GROUP') then ! keep group for backward compatibility
           if (first_me.eq.0) then
            write(info, '(2A)') whoami, &
     & ' WARNING: MPI_COMM_LOCAL CHANGED (REINITIALIZING PARALLEL RUN ?).'
            write(OUTU,'(A)') pack(info,info.ne.'');info='';
           endif
           call multicom_parinit()
          endif ! local
         endif ! len
        endif ! ind
!
        call int_vector_done(list)
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       elseif ( keyword(1:4).eq.'LIST'(1:4) ) then
        call multicom_list()
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       elseif ( keyword(1:4).eq.'DONE'(1:4) ) then
        if (me.ge.0) &
     & call wrndie(0,whoami,trim('REMOVING USER-DEFINED COMMUNICATOR LIST.'))
        call multicom_cleanup()
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! now have to reinitialize communicators; this is not done here
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       elseif ( keyword(1:7).eq.'PARINIT'(1:7) ) then
!ccccccc call parallel initialization routines (use after comm_local has been modified)
        call multicom_parinit()
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!ccccc in this section, provide "canned" communicator setups for various tasks, ]
!cccc e.g. ENSEMBLE/STRING modules
       elseif (.false. &
#if (KEY_ENSEMBLE==1)
     & .or. ( keyword(1:4).eq.'ENSE'(1:4) ) & 
#endif
#if (KEY_STRINGM==1)
     & .or. ( keyword(1:4).eq.'STRI'(1:4) ) & 
#endif
     & ) then
! setup a standard ensemble/string case
        qens=( keyword(1:4).eq.'ENSE'(1:4) )
        qstr=( keyword(1:4).eq.'STRI'(1:4) )
!
        klen=len(keyword)
        keyword=nexta8(comlyn,comlen)
        nlocal=decodi(keyword, klen)
        if (nlocal.le.0) then
         write(info(1),*)' INVALID NUMBER OF PROCESSORS PER REPLICA (',keyword,').';call wrndie(0,whoami,trim(info(1)))
        else

         call gtrmwd(comlyn, comlen, 'BY', 2, keyword, klen, i)
         nrep=decodi(keyword, i)
         if (nrep.le.0) then
          write(info(1),*)' INVALID NUMBER OF REPLICAS (',keyword,').';call wrndie(0,whoami,trim(info(1)))
         elseif (ncpu.lt.nrep*nlocal) then
          call wrndie(0,whoami,trim(' REQUESTING TOO MANY PROCESSORS. NOTHING DONE.'))
         else
! set up for a (nlocal x nrep) run
! use existing routines
          if (qens) keyword='ENSEMBLE'
          if (qstr) keyword='STRING'
          if (first_me.eq.0) then
           write(info, 333) whoami, keyword, whoami, nrep, nlocal
 333 FORMAT(/A, ' WILL SET UP COMMUNICATORS FOR ',A, &
     & /,A,' ON ', I5, ' REPLICA(S) WITH ', I5, &
     & ' PROCESSORS PER REPLICA.')
           write(OUTU,'(A)') pack(info,info.ne.'');info='';
          endif ! first_me
!
          call multicom_init() ! (re)initialize
! define nrep communicators (local)
          do i=1, nrep
           ind = multicom_add( (/ ((i-1)*nlocal+j, j=1,nlocal) /) )
          enddo
!
          call multicom_set('LOCAL', (/ (i+1, i=1, nrep)/)) ! set local communicator
! define ENSBL/STRNG
          ind = multicom_add( (/ ((j-1)*nlocal+1, j=1, nrep) /) )
          if (qens) call multicom_set('ENSBL', (/ nrep+2/) )
          if (qstr) call multicom_set('STRNG', (/ nrep+2/) )
!
          call multicom_parinit()
! warn about 'unused' processors
          if (first_me.eq.0) then
           if (first_size.gt.nlocal*nrep) then
            write(info,444) whoami,first_size-nlocal*nrep, keyword
 444 FORMAT(A, I5, ' PROCESSORS WILL NOT BE USED FOR ', &
     & A,' CALCULATIONS.');
            write(OUTU,'(A)') pack(info,info.ne.'');info='';
           endif ! first_size
          endif ! first_me
         endif ! nrep
        endif ! nlocal
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       else
            write(info(1),*)' UNRECOGNIZED SUBCOMMAND: ',keyword;call wrndie(0,whoami,trim(info(1)))
       endif
!
       end subroutine multicom_main
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine multicom_parinit()
       use pme_module, only: qpme, pmesh_setup, pmesh_clear; use ewald
       use ewald ! ugly hack to work around module detection bug in setmk.com
       use stream
       character(len=len("MULTICOM_PARINIT>") ),parameter::whoami="MULTICOM_PARINIT>";!macro
!
!
! if (first_me.eq.0) write(outu, '(2A)') whoami,' CALLING PARSTRT'
       call PARSTRT()
! write(0,*) 'BACK FROM PARSTART'
! if (LEWALD.and.QPME) then
! if (first_me.eq.0) write(outu, '(2A)') whoami,
! & ' CALLING PMESH_CLEAR'
! call PMESH_CLEAR()
!
! if (first_me.eq.0) write(outu, '(2A)') whoami,
! & ' CALLING PMESH_SETUP'
! call PMESH_SETUP()
! endif
!
       end subroutine multicom_parinit
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine multicom_init()
       use mpi
! locals
       integer*4 :: communicator
       integer :: i, j
       integer*4 :: ii, jj
       integer*4 :: bug=0
       character(len=len("MULTICOM_INIT>") ),parameter::whoami="MULTICOM_INIT>";!macro
       logical :: mpiinit
!
       if (multicom_initialized) then
        call multicom_cleanup()
       endif
!
! allocate communicator array:
!
       allocate(communicators(mcom_expand))
!
       if (size(communicators).lt.1) then
        call wrndie(0,whoami,trim(' INTERNAL ERROR : COMMUNICATOR ARRAY HAS NONPOSITIVE SIZE.'))
        call multicom_cleanup()
        return
       endif
!
       ncomm=0 ! number of communicators
       do i=1, size(communicators)
        communicators(i)%me=MPI_UNDEFINED
        communicators(i)%size=1
        communicators(i)%comm_id=MPI_COMM_NULL
        communicators(i)%group_id=MPI_GROUP_NULL
        communicators(i)%parent_id=-1
       enddo
!
       call mpi_initialized(mpiinit,bug)
       if (.not.mpiinit) call mpi_init(bug)
       communicator=MPI_COMM_WORLD
!
       ncomm=ncomm+1
!
       call MPI_COMM_GROUP(communicator, communicators(ncomm)%group_id, bug)
       call MPI_GROUP_SIZE(communicators(ncomm)%group_id, ii, jj)
       call MPI_COMM_RANK(communicator, communicators(ncomm)%me, bug)
       call MPI_COMM_SIZE(communicator, communicators(ncomm)%size, bug)
       communicators(ncomm)%comm_id=communicator
       communicators(ncomm)%parent_id=0
       first_me=communicators(ncomm)%me
       first_size=communicators(ncomm)%size
!
! store procs:
       do i=0,first_size-1
        j=int_vector_uadd(communicators(ncomm)%nodes, i)
       enddo
!
       multicom_initialized=.true. ! must be set before calling set below
!
       end subroutine multicom_init
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       function multicom_add(nodes,comm)
! define a new communicator & return pointer into communicator array
! all communicators are defined from the absolute one defined in multicom_init
! it is currently MPI_COMM_WORLD; for a communicator parent_id=n means that the
! _values_ in the nodes array correspond to _indices_ into the parent`s nodes array,
! except when n=0, in thich case the _values_ in the nodes array are actual ranks in the
! absolute communicator; these ranks are also group ranks, which can be used to create
! additional groups and communicators; in the current implementation, parent_id is 0
! for all communicators;
!
       use mpi
!
       integer, optional :: comm ! omission implies 1st communicator (WORLD)
       integer :: nodes(:)
! locals
       integer :: comm_id, node_id
       integer :: i, j
       integer :: multicom_add
       integer*4 :: bug=0, nproc
       integer*4, allocatable :: list(:)
!
       type (comm_type), pointer :: newcomms(:)=>NULL() ! reallocation
!
       character(len=len("MULTICOM_ADD>") ),parameter::whoami="MULTICOM_ADD>";!macro
!
       multicom_add=-1
       if (.not.multicom_initialized) call multicom_init()
!
       if (present(comm)) then
        comm_id=comm
       else
        comm_id=1
       endif
!
       if (comm_id.gt.ncomm.or.comm_id.lt.1) then
        call wrndie(0,whoami,trim(' INVALID PARENT COMMUNICATOR.'))
        return
       endif
!
! parse processor array
       if (ncomm.eq.size(communicators)) then
! reallocate
        allocate(newcomms(ncomm+mcom_expand))
        newcomms(1:ncomm)=communicators
        do i=1,ncomm
! make sure nodelist pointers are copied
         newcomms(i)%nodes%i=>communicators(i)%nodes%i
         nullify(communicators(i)%nodes%i)
        enddo
        deallocate(communicators)
        communicators=>newcomms
        if (ncomm.eq.size(communicators)) then
         call wrndie(0,whoami,trim(' INTERNAL ERROR : COULD NOT EXPAND COMMUNICATOR ARRAY.'))
         return
        endif
       endif
!
       ncomm=ncomm+1
!
       do i=1,SIZE(nodes)
! determine node_id relative to first communicator (WORLD)
        j=comm_id
        node_id=nodes(i) ! node relative to the requested comm.
        do while (j.gt.0) ! want to get node index relative to the absolute communicator
         node_id=communicators(j)%nodes%i(node_id) ! node id in parent communicator of j
         j =communicators(j)%parent_id
        enddo
! save absolute node ID
        j=int_vector_uadd(communicators(ncomm)%nodes, node_id) ! relative to 1
       enddo
!
       nproc=communicators(ncomm)%nodes%last
       allocate(list(nproc)); ! stay with 4-bit ints for MPI
       list=communicators(ncomm)%nodes%i(1:nproc)
! define new group
       call MPI_GROUP_INCL(communicators(1)%group_id, &
     & nproc, list, & ! 4-bit integer array
     & communicators(ncomm)%group_id, bug)
!
! check for empty group
       if (communicators(ncomm)%group_id.eq.MPI_GROUP_EMPTY) then
        call wrndie(0,whoami,trim(' NO VALID NODES FOUND. NOTHING DONE.'))
        ncomm=ncomm-1
        return
       endif
!
! call MPI_GROUP_SIZE(communicators(ncomm)%group_id, i,j)
! create communicator based on new group; on some nodes this will return MPI_COMM_NULL
! although the corresponding group should never be NULL
       call MPI_COMM_CREATE(communicators(1)%comm_id, &
     & communicators(ncomm)%group_id, &
     & communicators(ncomm)%comm_id, bug)
!
! obtain size + rank from communicator
       if (communicators(ncomm)%comm_id.ne.MPI_COMM_NULL) then
        call MPI_COMM_RANK(communicators(ncomm)%comm_id, &
     & communicators(ncomm)%me, bug)
        call MPI_COMM_SIZE(communicators(ncomm)%comm_id, &
     & communicators(ncomm)%size, bug)
! since we expressed all nodes relative to absolute comm, parent ID is 0
        communicators(ncomm)%parent_id=0
       else
        communicators(ncomm)%size=1 ! for consistency with many communication routines, which bail when size=1
        communicators(ncomm)%parent_id=0 ! parent is 0 -- see above
        communicators(ncomm)%me=MPI_UNDEFINED
       endif
!
       multicom_add=ncomm
!
       if (allocated(list)) deallocate(list)
!
       end function multicom_add
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine multicom_cleanup()
! remove all traces of multicom, but leave MPI_COMM_WORLD intact
       use mpi
!
       integer :: i
       integer*4 :: bug
       integer*4 :: result
!
       if (associated(communicators)) then
        do i=1,ncomm
         call int_vector_done(communicators(i)%nodes)
! destroy group & communicators
         if (communicators(i)%comm_id.ne.MPI_COMM_NULL) then ! it is an error to use MPI_COMM_NULL in compare
       call MPI_COMM_COMPARE(MPI_COMM_WORLD,communicators(i)%comm_id, &
     & result, bug)
! write(0,*) first_me, communicators(i)%comm_id, i
!
          if (result.ne.MPI_IDENT) &
     & call MPI_COMM_FREE(communicators(i)%comm_id, bug) ! keep world
         endif
!cccccccc
         if (communicators(i)%group_id.ne.MPI_GROUP_NULL) &
     & call MPI_GROUP_FREE(communicators(i)%group_id, bug)
        enddo
        ncomm=0
        deallocate(communicators)
       endif
!
! reset communicators
!#ifdef 1
       call multicom_safe_reset()
!#endif // __CHARMM
       call multicom_parinit()
!
       multicom_initialized=.false.
!
       end subroutine multicom_cleanup
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine multicom_list()
! list currently defined communicators
! use mpi
       use stream
!
       integer :: i
! integer :: bug
! integer*4 :: j,k ! aardvark
       character(len=len("MULTICOM_LIST>") ),parameter::whoami="MULTICOM_LIST>";!macro
!
       if (.not.multicom_initialized) then
        call wrndie(0,whoami,trim(' COMMUNICATOR MODULE NOT INITIALIZED.'))
        return
       endif
!
       if (ncomm.lt.1) then
        call wrndie(0,whoami,trim(' INTERNAL ERROR: NO COMMUNICATORS DEFINED.'))
        return
       endif
!
       if (communicators(1)%me.eq.0) then
         write(info, 111) whoami ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ! 0 writes
         do i=1, ncomm
           write(info, 112) whoami, i, communicators(i)%parent_id ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
           write(info, '(" MULTICOM_LIST> ", 10I5)') &
     & communicators(i)%nodes%i(1:communicators(i)%nodes%last) ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
         enddo
       endif
 111 FORMAT (/A,' THE FOLLOWING COMMUNICATORS ARE DEFINED:')
 112 FORMAT (' ',A, ' #', I5, ' NODE LIST (RELATIVE TO COMMUNICATOR #',I5,'):')
!
! aardvaark
! j= MPI_UNDEFINED
! if (MPI_COMM_LOCAL.ne.MPI_COMM_NULL)
! & call MPI_COMM_RANK(MPI_COMM_LOCAL,j,bug)
! write(900+first_me,*) first_me, MPI_COMM_LOCAL, j
! if (SIZE_LOCAL.gt.1) CALL MPI_BARRIER(MPI_COMM_LOCAL,bug)
! aardvaark
       end subroutine multicom_list
!
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine multicom_barrier(comm_name)
       use multicom_aux
       use string
       use stream
       use mpi
!
       character(len=*) :: comm_name
! local vars
       integer :: s, m, l
       integer*4 :: c, ierror
!
       character(len=len("MULTICOM_BARRIER>") ),parameter::whoami="MULTICOM_BARRIER>";!macro
!
       if (.not.multicom_initialized) then
        call wrndie(0,whoami,trim(' COMMUNICATOR MODULE NOT INITIALIZED. NOTHING DONE.'))
        return
       endif
!
       comm_name=adjustl(comm_name); l=len_trim(comm_name)
       if (l.le.0) then
        call wrndie(0,whoami,trim(' COMMUNICATOR NAME NOT SPECIFIED. NOTHING DONE.'))
        return
       endif
!
! cycle through existing communicator list
       l=5 ! length of communicator id string
!
       if (comm_name(1:5).eq.'WORLD') then
! this should not be necessary since 'GLOBAL' wraps 'WORLD'; primarily for debugging
        c=MPI_COMM_WORLD; s=1; m=1;
       elseif (comm_name(1:5).eq.'LOCAL') then
        c=MPI_COMM_LOCAL; s=SIZE_LOCAL; m=ME_LOCAL;
       elseif (comm_name(1:5).eq.'GROUP') then ! for backward compatibility
        c=MPI_COMM_LOCAL; s=SIZE_LOCAL; m=ME_LOCAL;
       elseif (comm_name(1:6).eq.'GLOBAL') then
        c=MPI_COMM_GLOBAL; s=SIZE_GLOBAL; m=ME_GLOBAL;
        l=6
#if (KEY_ENSEMBLE==1)
       elseif (comm_name(1:5).eq.'ENSBL') then 
#endif
#if (KEY_ENSEMBLE==1)
        c=MPI_COMM_ENSBL; s=SIZE_ENSBL; m=ME_ENSBL; 
#endif
#if (KEY_STRINGM==1)
       elseif (comm_name(1:5).eq.'STRNG') then 
#endif
#if (KEY_STRINGM==1)
        c=MPI_COMM_STRNG; s=SIZE_STRNG; m=ME_STRNG; 
#endif
       elseif (comm_name(1:6).eq.'PARSER') then
        c=MPI_COMM_PARSER; s=SIZE_PARSER; m=ME_PARSER;
        l=6
       else
        write(info(1),*)comm_name,' IS NOT IN USE. NOTHING DONE.';call wrndie(0,whoami,trim(info(1)))
        return
       endif
!
! if (first_me.eq.0) then
       if (m.eq.0) then ! all roots print
         write(info, '(4A)') whoami, &
     & ' CALLING MPI_BARRIER ON COMMUNICATOR "',comm_name(1:l),'".'; write(OUTU,'(A)') pack(info,info.ne.'');info='';
       endif
!
       if (c.ne.MPI_COMM_NULL.and.m.ne.MPI_UNDEFINED.and.s.gt.1) &
     & call MPI_BARRIER(c, ierror)
!
       end subroutine multicom_barrier
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine multicom_set(comm_name, commlist)
! set various communicators based on the multicom 'database'
! note: all processors are meant to execute this routine
       use multicom_aux;
       use stream
       use mpi
!
       character(len=*) :: comm_name
       integer :: commlist(:)
! local vars
       character(len=len("MULTICOM_SET>") ),parameter::whoami="MULTICOM_SET>";!macro
       integer :: llist
       integer :: comm_id, node_id, parent_id
       integer :: numnodes, c, m, s
       integer :: itype, bug, lcomm, i, j
       logical :: found=.false., foundg=.false.
!
       integer :: flag(0:first_size-1), flagg(0:first_size-1)
!
       if (.not.multicom_initialized) then
        call wrndie(0,whoami,trim(' COMMUNICATOR MODULE NOT INITIALIZED. NOTHING DONE.'))
        return
       endif
!
! first check whether the communicator ids are valid
       llist=SIZE(commlist)
       do i=1, llist
        if (commlist(i).gt.ncomm.or.commlist(i).lt.1) then
         call wrndie(0,whoami,trim(' SOME COMMUNICATOR IDs INVALID. NOTHING DONE.'))
         return
        endif
       enddo
!
       flag=0
       flagg=0
!
       lcomm=len(comm_name)
!
! loop over all communicators. Note that no check for overlapping groups is made
! in the case a node belonging to multiple groups,
! the last communicator assignment overwrites previous ones
!
       do j=1, llist
        comm_id=commlist(j)
! define dummy vars to shorten code
! note that any of these might be undefined
        c=communicators(comm_id)%comm_id ! could be MPI_COMM_NULL on this node
        m=communicators(comm_id)%me ! could be MPI_COMM_UNDEFINED
        s=communicators(comm_id)%size
!
! detect if null communicator (unlikely, just bug check)
        found=c.eq.MPI_COMM_NULL
        call MPI_ALLREDUCE(found, foundg, 1, MPI_LOGICAL, MPI_LAND, &
     & communicators(1)%comm_id, bug)
        if (foundg) then
         call wrndie(0,whoami,trim(' UNEXPECTED ERROR: COMMUNICATOR HANDLE NULL ON ALL NODES.'))
         return
        endif
!
        numnodes=communicators(comm_id)%nodes%last
!
! write(0,*) communicators(comm_id)%nodes%i(1:numnodes)
        do i=1, numnodes
!
         node_id =communicators(comm_id)%nodes%i(i) ! node id in parent (possibly 0)
         parent_id=communicators(comm_id)%parent_id ! possibly 0
! write(700+ME_GLOBAL,*) parent_id ! aa
!
         do while (parent_id.ne.0)
          node_id =communicators(parent_id)%nodes%i(node_id) ! node_id in parent of parent
          parent_id=communicators(parent_id)%parent_id ! parent of parent (possibly 0)
         enddo
! write(0,*) numnodes, node_id, comm_id, parent_id
!
! set communicator
         if (first_me .eq. node_id) then
! check communicator list
          if (lcomm.ge.5) then
!
           flag(node_id)=node_id+1 ! this flag means that a communicator assignment is successful
!
           if (comm_name(1:5).eq.'LOCAL') then
            MPI_COMM_LOCAL=c; SIZE_LOCAL=s; ME_LOCAL=m;
           elseif (comm_name(1:5).eq.'GROUP') then ! for backward compatibility
            MPI_COMM_LOCAL=c; SIZE_LOCAL=s; ME_LOCAL=m;
!
#if (KEY_ENSEMBLE==1)
           elseif (comm_name(1:5).eq.'ENSBL') then 
#endif
#if (KEY_ENSEMBLE==1)
            MPI_COMM_ENSBL=c; SIZE_ENSBL=s; ME_ENSBL=m 
#endif
!
#if (KEY_STRINGM==1)
           elseif (comm_name(1:5).eq.'STRNG') then 
#endif
#if (KEY_STRINGM==1)
            MPI_COMM_STRNG=c; SIZE_STRNG=s; ME_STRNG=m 
#endif
!
!
!
!
           elseif (comm_name(1:6).eq.'PARSER') then
            MPI_COMM_PARSER=c; SIZE_PARSER=s; ME_PARSER=m
           else
            flag(node_id)=0
           endif
          endif ! lcomm
!
         endif ! first_me
!
        enddo ! i=1, numnodes
       enddo ! over commlist array (j)
!
!
#if (KEY_INTEGER8==0)
       itype=MPI_INTEGER 
#endif
#if (KEY_INTEGER8==1)
       itype=MPI_INTEGER8 
#endif
! combine flagg; have to do some ad-hoc gymnastics because of integer size
       call MPI_ALLREDUCE(flag, flagg, &
     & first_size, itype, MPI_MAX, communicators(1)%comm_id, bug)
! flagg array is positive for nodes on which assignment worked, and 0
! for which it did not; for the latter nodes, explicitly set
! communicator to null, size to 1, and me to MPI_UNDEFINED:
       c=MPI_COMM_NULL; s=1; m=MPI_UNDEFINED
       if (flag(first_me).eq.0) then
         if (comm_name(1:5).eq.'LOCAL') then
          MPI_COMM_LOCAL=c; SIZE_LOCAL=s; ME_LOCAL=m;
         elseif (comm_name(1:5).eq.'GROUP') then ! for compatibility with older scripts
          MPI_COMM_LOCAL=c; SIZE_LOCAL=s; ME_LOCAL=m;
!
#if (KEY_ENSEMBLE==1)
         elseif (comm_name(1:5).eq.'ENSBL') then 
#endif
#if (KEY_ENSEMBLE==1)
          MPI_COMM_ENSBL=c; SIZE_ENSBL=s; ME_ENSBL=m 
#endif
!
#if (KEY_STRINGM==1)
         elseif (comm_name(1:5).eq.'STRNG') then 
#endif
#if (KEY_STRINGM==1)
          MPI_COMM_STRNG=c; SIZE_STRNG=s; ME_STRNG=m 
#endif
!
!
!
!
         elseif (comm_name(1:6).eq.'PARSER') then
          MPI_COMM_PARSER=c; SIZE_PARSER=s; ME_PARSER=m
         endif
       endif ! lcomm
!
       llist=sum(min(flagg,1)) ! # nodes with successful assignments
       found=(llist.gt.0)
       if (.not.found) then
        write(info(1),*)comm_name,' IS NOT IN USE ON REQUESTED NODES.';call wrndie(0,whoami,trim(info(1)))
       else
        if (first_me.eq.0) then
         write(info(1),*)'ASSIGNED "',trim(comm_name),'" COMMUNICATOR ON THE FOLLOWING NODES:';write(OUTU,'(3A)') whoami,' ',trim(info(1));info(1)=''
! VO: 10.2012: if the number of nodes is very large, the node list
! can overrun the info array; therefore, print in sections (instead of single line below)
! write(info, '(" MULTICOM_SET> ",20I5)') &
! & PACK(flagg, flagg.gt.0)-1 ! node numbering starts from 0
! write(OUTU,'(A)') pack(info,info.ne.'');info='';
! reuse "flag" -- not used from here on:
         flag=PACK(flagg, flagg.gt.0)-1
         j=20*size(info) ! max number of node ids that will fit in the buffer
         do i=0, llist-1, j
          write(info, '(" MULTICOM_SET> ",20I5)') flag(i:min(i+j-1,llist-1)) ! node numbering starts from 0
          write(OUTU,'(A)') pack(info,info.ne.'');info='';
         enddo
!
         if (llist.lt.first_size) & ! warn that on some nodes, communicator is still unassigned
     & write(info, '(" ",2A,I5,A)') whoami, &
     & ' COMMUNICATOR "'//trim(comm_name)//'" UNASSIGNED ON ',        &
     & first_size-llist, ' NODES.' ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
        endif ! first_me
       endif ! .not.found
!
       end subroutine multicom_set
!====================================================================
#if (KEY_STRINGM==1)
       subroutine multicom_permute_string_ranks(ranks)
       use multicom_aux;
       use mpi
! assume that the string communicator exists (MPI_COMM_STRNG)
! loop over all communicators & find the string communicator
! create a new communicator using the nodes of the string comm.,
! using WORLD as parent. Delete the existing string communicator
! and set the new communicator to MPI_COMM STRNG.
!
       integer :: ranks(:)
!
       integer :: i, j, k, comm, node_id, parent_id
       logical :: found
       integer :: itype
       integer*4 :: bug
       character(len=len("MULTICOM_PERMUTE_STRING_RANKS>") ),parameter::whoami="MULTICOM_PERMUTE_STRING_RANKS>";!macro
!
       if (.not.multicom_initialized) then
        call wrndie(0,whoami,trim(' COMMUNICATOR MODULE NOT INITIALIZED. NOTHING DONE.'))
        return
       endif
!
#if (KEY_INTEGER8==0)
       itype=MPI_INTEGER 
#endif
#if (KEY_INTEGER8==1)
       itype=MPI_INTEGER8 
#endif
!
       if (ncomm.lt.1) then
        call wrndie(0,whoami,trim(' INTERNAL ERROR: NO COMMUNICATORS DEFINED.'))
        return
       endif
!
       comm=0
       if (MPI_COMM_STRNG.ne.MPI_COMM_NULL.and.ME_STRNG.eq.0) then
        found=.false.
        do while (.not.found.and.comm.lt.ncomm)
         comm=comm+1
         found=(communicators(comm)%comm_id.eq.MPI_COMM_STRNG)
        enddo
        if (.not.found) comm=0
       endif
! reduce
       call MPI_ALLREDUCE(comm, i, 1, itype, MPI_BOR,MPI_COMM_GLOBAL, bug)
       comm=i
       if (comm.eq.0) then
        call wrndie(0,whoami,trim(' INTERNAL ERROR: MPI_COMM_STRNG NOT FOUND.'))
        return
       endif
!
! add new communicator
! write(800+ME_GLOBAL,*) comm
! write(800+ME_GLOBAL,*) ranks-1
! write(800+ME_GLOBAL,*)
! & communicators(comm)%nodes%i(1:size(ranks))
!
!
       j=multicom_add(ranks, comm)
! assing MPI_COMM_STRNG to new communicator
       do i=1, communicators(j)%nodes%last
!
         node_id =communicators(j)%nodes%i(i) ! node_id in j`s parent
         parent_id=communicators(j)%parent_id ! j`s parent
!
         do while (parent_id.ne.0) ! want to get node index relative to the global communicator (WORLD)
          node_id =communicators(parent_id)%nodes%i(node_id) ! node id in parent of parent
          parent_id=communicators(parent_id)%parent_id
         enddo
! set communicator
         if (first_me .eq. node_id ) then
          MPI_COMM_STRNG=communicators(j)%comm_id
          SIZE_STRNG= communicators(j)%size
          ME_STRNG= communicators(j)%me
         endif ! first_me
!
       enddo ! i=1, numnodes
! destroy old communicator and group
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! do not do any checks -- assume rest of code works correctly
       if (communicators(comm)%comm_id.ne.MPI_COMM_NULL) & ! comm will be null on slave nodes
     & call MPI_COMM_FREE(communicators(comm)%comm_id, bug)
       call MPI_GROUP_FREE(communicators(comm)%group_id, bug) ! group will be defined on all nodes
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! overwrite old communicator by new communicator;
       communicators(comm)%me =communicators(j)%me
       communicators(comm)%size =communicators(j)%size
       communicators(comm)%comm_id =communicators(j)%comm_id
       communicators(comm)%group_id =communicators(j)%group_id
       communicators(comm)%parent_id=communicators(j)%parent_id
       call int_vector_done(communicators(comm)%nodes)
       do i=1,communicators(j)%nodes%last
        k=int_vector_uadd(communicators(comm)%nodes, &
     & communicators(j)%nodes%i(i))
       enddo
! remove old string communicator
       call int_vector_done(communicators(j)%nodes)
       ncomm=ncomm-1
! write(800+ME_GLOBAL,*)
! & communicators(comm)%nodes%i(1:size(ranks)) !aa
! close(800+ME_GLOBAL)
!
       end subroutine multicom_permute_string_ranks
#endif
!====================================================================
       subroutine multicom_safe_reset()
       use multicom_aux
       use mpi
! guess reset failsafe communicator values ; called after multicom_cleanup
! to avoid problems if multicom finalized in the middle of a program
       MPI_COMM_LOCAL=MPI_COMM_GLOBAL
       SIZE_LOCAL=SIZE_GLOBAL
       ME_LOCAL=ME_GLOBAL
!
! reset parser communicator to global
       MPI_COMM_PARSER=MPI_COMM_GLOBAL
       SIZE_PARSER=SIZE_GLOBAL
       ME_PARSER=ME_GLOBAL
!
!
! Other communicators
#if (KEY_ENSEMBLE==1)
  MPI_COMM_ENSBL=MPI_COMM_NULL; ME_ENSBL=MPI_UNDEFINED; SIZE_ENSBL=1; 
#endif
#if (KEY_STRINGM==1)
  MPI_COMM_STRNG=MPI_COMM_NULL; ME_STRNG=MPI_UNDEFINED; SIZE_STRNG=1; 
#endif
! add lines as new communicator names are defined
!
       end subroutine multicom_safe_reset
!====================================================================
#endif /* automatically protect all code */
      end module multicom
!====================================================================
