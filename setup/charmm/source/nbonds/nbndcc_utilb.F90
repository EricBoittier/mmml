 module nbndcc_utilb
 use dimens_fcm
 use chm_kinds

 implicit none

 contains
!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
 subroutine parstoperr(SUBNAME,ERRORNM,PSTATUS,PWARN,PQSINGLE,PQSTOP,PQERROR)
#if KEY_PARALLEL==1
 use parallel,only: MYNODP 
#endif
! provides a clean message print or stop across all cpus.
! returns without action if warning level (def -5) is > charmm WRNLEV.
! by default, prints passed message, preceeded by "***ERROR'
! and stops                                       --RJP  6/2012
 implicit none
 character(len=*),intent(in) :: SUBNAME,ERRORNM
 integer,intent(in),optional :: PWARN !passed warning level
 integer(chm_int4),optional :: PSTATUS !mpi status var
 logical,intent(in),optional :: PQSINGLE,PQSTOP,PQERROR !flags for writing to 
! single cpu, stopping after print, printing "ERROR", all true by default.
#if KEY_PARALLEL==1
 include 'mpif.h' 
#endif
!local
 logical :: QSINGLE,QSTOP,QERROR 
 integer(chm_int4) :: STATUS
 character(len=10) :: STRING
 integer :: WRN
! end of declarations
 WRN = -5
 if(present(PWARN)) WRN = PWARN
 if(WRN>WRNLEV) return

 QSINGLE = .true.
 if(present(PQSINGLE)) QSINGLE=PQSINGLE
 QSTOP = .true.
 if(present(PQSTOP)) QSTOP = PQSTOP
 QERROR = .true.
 if(present(PQERROR)) QERROR = PQERROR
 
 STRING = '********* '
 if(QERROR) STRING = '***ERROR: '

#if KEY_PARALLEL==1
 if((MYNODP.eq.1).or.(.not.QSINGLE)) then  
#endif
   WRITE(6,*) '************************************************'
   WRITE(6,*) STRING,'<',SUBNAME,'>',ERRORNM,'**************' 
   WRITE(6,*) '************************************************'
#if KEY_PARALLEL==1
 endif  
#endif

#if KEY_PARALLEL==1
! write(6,*) 'mynodp ',MYNODP,' calling mpi barrier'
 call MPI_BARRIER(MPI_COMM_WORLD,STATUS)
 if(present(PSTATUS)) PSTATUS=STATUS
! write(6,*) 'mynodp ',MYNODP,' called mpi barrier'
#endif

 if(QSTOP.and.WRN<=BOMLEV) then
#if KEY_PARALLEL==1
! write(6,*) 'mynodp ',MYNODP,' calling finalize '
  call MPI_FINALIZE
! write(6,*) 'mynodp ',MYNODP,' called finalize '
#endif
  stop
 endif
  
 end subroutine parstoperr

#if KEY_PARALLEL==1
!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
   subroutine mpicomp_send_rec(NNTCPUSEN,NTCPUSEN,NNTCPUREC,NTCPUREC,pqverb,plabint)
! given the arrays of cpus to which each cpu is sending and from which it is 
! receiving, this algorithm checks the entire network for self-consistency,
! by comparing sends and receives of all cpus in the system.
! since there is global communication, is slow for large numbers of CPUS
! requires on the order of NUMNOD^2 words of memory
!                                                            --RJP 5/2012
   use parallel,only: NUMNOD,MYNODP
   implicit none
   integer,dimension(:),allocatable,intent(in) :: NTCPUSEN,NTCPUREC !arrays of sending/receiving nodes
   integer,intent(in) :: NNTCPUSEN,NNTCPUREC !length of send/receive arrays
   integer,optional,intent(in) :: plabint  !integer label for debugging
   logical,optional,intent(in) :: pqverb !verbose flag
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
 end module nbndcc_utilb
