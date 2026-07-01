! Utility routines for parallel CHARMM on MIMD machines.
!
!  NOTE: This routine will be converted to the new fortran
!        only after all the heap and stacks are gone, ie
!        there will be no need to compile CHARMM with
!        -i8 or -fdefault-integer-8 compiler flags!!!
!
! Machines currently supported:
!       Intel iPSC/860 (gamma-Touchstone)  (## INTEL)
!       Intel delta-Touchstone (both ## INTEL and DELTA)
!       Intel Paragon (both ## INTEL and PARAGON)
!       Ethernet cluster (diverse workstations) (## ETHER)
!       Token Ring (Apollo workstations) (## RING)
!       FDDI Ring or Ethernet (homogeneous workstations) (## SOCKET)
!       Thinking Machines CM-5 (## CM5)
!       IBM SPn (## IBMSP)
!       PVM with any platform which supports it.  (## PVM)
!           (NOTE: on majority of workstation clusters it is
!                  recommended to use ##SOCKET instead of ##PVM,
!                  but they are both fully functional)
!           PowerChallenge (PVM-3.3.7) doesn't support PVMDataInPlace
!                          - use (## SGIMP)
!
! Synchronous (one way) I/O controlled by (## SYNCHRON)
!
! Entire section controlled by ##IF PARALLEL
!
! Written by Bernie Brooks & Milan Hodoscek,
!          Original version From April, 1991
!          New Global sum routines from April 1992
!          (Based on methods by G. Fox and R. Van de Geijn)
!          (Intel iPSC/860 assistance from Stan Erwin)
!
! Completely rewritten for easy support of many platforms, June 1993
! CM5 support revamped, July 1994
! Some reorganization to better support different libraries on the same
! platform, July 1997
!
! This file is organized in 3 layers:
! 1. layer: routines which are called from the rest of CHARMM
!           This is implemented as direct call to MPI library
!           In case of bad implementation CMPI calls should be used.
! 2. layer: machine independent implementation of CMPI calls
! 3. layer: machine dependent implementation of layer 2
! For details see comments under LAYER labels below.
!
! NEW: This file was split in 3 - paral1.f90, paral2.fcm, paral3.fcm
!      each LAYER is in its own file
!
! brbrooks@helix.nih.gov, milan@par10.mgsl.dcrt.nih.gov
! nagle@tammy.harvard.edu, tamayo@think.com (CM5)
!
! Please check with B. Brooks before publication for current
! appropriate acknowledgement.  Thanks.
!


!-----------------------------------------------------------------------
!                        PARSTRT
! Startup for parallel code.
!-----------------------------------------------------------------------
subroutine parstrt
  !  This code must be run when CHARMM is started to properly setup all
  !  data communication.
#if KEY_PARALLEL == 1
  use chm_kinds
  !
  use dimens_fcm
  use parallel
  use parallel_groups
  use param_store, only: set_param
  use chm_types
  use stream
  use number
  use mpi
#if KEY_MULTICOM==1
  ! VO string v
  use multicom, only: multicom_init, multicom_safe_reset
  use multicom_aux
  ! VO string ^
#endif
#endif /* KEY_PARALLEL */

#ifdef _OPENMP
  use domdec_common, only: nthread
  use omp_lib, only: &
       omp_get_max_threads, &
       omp_set_num_threads
#endif

  implicit none

#if KEY_PARALLEL == 1
  integer :: i, status

#if KEY_MULTICOM==1 /*  VO string v */
  logical :: initialized_mpi=.false.
#endif /* VO string ^ */
#endif /* KEY_PARALLEL */

#ifdef _OPENMP
  character(len=10) :: nthreads_str = ' '

  call get_environment_variable('OMP_NUM_THREADS', nthreads_str)
  if (trim(nthreads_str) .eq. '') call omp_set_num_threads(1)

  nthread = omp_get_max_threads()
#endif /* _OPENMP */

#if KEY_PARALLEL == 1
  !     Initialize
  ! The CMPI_COMM_WORLD and other flags in parallel_ltm.src
  ! are internal to some CMPI routines in paral2.src.
  ! For CMPI to work properly we need both CMPI_COMM_WORLD and
  ! standard MPI_COMM_WORLD that we get from "use mpi"
  ! MPI_COMM_WORLD has to be used for comm_charmm even when CMPI!!
#if KEY_MULTICOM==1
  ! VO: if multicom is used, we can call parinit to reset communicator and node info,
  !     but without calling mpi_init (or using mpi_world); thus, one can basically
  !     re-initialize with a different communicator on-the-fly at runtime
#endif
  !
 comm_charmm = mpi_comm_world
  master_node = 0

#if KEY_CMPI==1 /*pll_cmpi1*/
  call cmpi_init(status&
#if KEY_MULTICOM==1
  &                     ,initialized_mpi &               /* VO stringm*/
#endif
  &                                     )
#else /* (pll_cmpi1)*/
#if KEY_MULTICOM==1 /*  VO stringm v */
  call mpi_initialized(initialized_mpi, status)
  if (initialized_mpi) then
   comm_charmm = MPI_COMM_LOCAL
   mynod       = ME_LOCAL
   numnod      = SIZE_LOCAL
  else
   comm_charmm = mpi_comm_world
  endif
#endif /* VO stringm ^ */
  call init_chm_groups
  if(numnod.gt.maxnode) call wrndie(-5,'<PARSTRT>','Too many processors acquired')
  call cube(mynod,numnod,ippmap)
#endif /* (pll_cmpi1)*/
  !
#if KEY_MULTICOM==1 /*  VO stringm */
  if (.not.initialized_mpi) then
   MPI_COMM_GLOBAL=MPI_COMM_WORLD;SIZE_GLOBAL=NUMNOD;ME_GLOBAL=MYNOD
   call multicom_safe_reset()
   call multicom_init()
  endif
#endif /* VO stringm */

  mynodp = mynod+1

  call mpi_type_size(MPI_INTEGER, mpi_integer_size, status)
  call mpi_type_size(MPI_REAL8, mpi_real8_size, status)
  call mpi_type_size(MPI_REAL4, mpi_real4_size, status)

  !     For parallel/parallel QM replica/path
  qqmpar=.false.

  !     Initialize the global info (for parallel/parallel)
  mynodg=mynod
  numnodg=numnod
  mynodgp = mynodg+1

  plnod0=prnlev
  if(mynod.ne.0) plnod0=0
  !
#if KEY_MULTICOM==1 /* (multicom)  VO stringm v */
  !
  ! Use MPI-compliant calls so that if gencomm is undefined and 
  ! the number of CPU is not a power of two,
  ! we avoid gbor & survive this routine
  !
  if (numnod.gt.1) then
#if KEY_MPI==1 /* (mpi) */
   call MPI_ALLREDUCE(MPI_IN_PLACE,plnod0,1,MPI_INTEGER,MPI_BOR,&
     &                   COMM_CHARMM,STATUS)
#endif /*(mpi) */
  endif
  if(ME_GLOBAL.ne.0) then ! VO : by default suppress output except on global root
  !
#else /* (multicom)  VO stringm multicom ^ */
  call gbor(plnod0,1)
  if(mynod.ne.0) then
#endif /*(multicom) */
  !
     prnlev=-1
     wrnlev=-5
     iolev=-1
  endif
  dpcomm=1

  call set_param('MYNODE',mynod)
  call set_param('NUMNODE',numnod)
  !
#if KEY_MULTICOM==1 /*  VO stringm */
  call set_param('MYNODEG',ME_GLOBAL)
  call set_param('NUMNODEG',SIZE_GLOBAL)
#endif /* VO stringm */
  !
  do i=1,maxnode
     inode(i)=mod(i,numnod)
  enddo

  !     Initialize timing array

  imeri=0
  tmeri(1:tmgnum)=zero

  !     Check priority of the main process and assign the same on the
  !     rest of processes
! BIOVIA Code Start : Fix for Windows
#if KEY_WIN32==1 || WIN64==1
  !
#elif KEY_GNU==1 || KEY_OSX==1
! BIOVIA Code End
  i=0
  if(mynod.eq.0) call getppr(i)
  call psnd4(i,1)
  if(mynod.ne.0) call setppr(i)
#endif 
  !
  ! handle the infinite network speed option
#if KEY_PARINFNTY==1
  qinfinity = .true.                   
#endif

#endif /* KEY_PARALLEL */
end subroutine parstrt

#if KEY_PARALLEL==1 /*pllmain*/
!-----------------------------------------------------------------------
! LAYER 1
! Routines that are called from the rest of CHARMM and for efficiency
! implemented as MPI or variuos other message passing libraries.
!-----------------------------------------------------------------------

subroutine vdgsum(x,y,z,mode)
  !-----------------------------------------------------------------------
  !
  !    Wraping routine for vector distributed global sum.
  !
  !    MODE=0 - Do just the inner sum for PARSCAL
  !         (assume contributions are only on processors containing block)
  !    MODE=1 - Do the complete global sum (outer followed by inner)
  !             (assume values can be on any node)
  !    MODE=2 - Do just the outer sum.
  !             (assume values anywhere, but only needed by block nodes)
  !
  ! The MODE values 0 and 1 are identical and MODE 2 is ignored for PARAFULL.
  !
  use chm_kinds
  use dimens_fcm
  use parallel
  !
  use exfunc
  use memory
#if KEY_CMPI==0
  use mpi      
#endif
  !
  implicit none
  real(chm_real) X(*), Y(*), Z(*)
  integer mode

  real(chm_real),allocatable,dimension(:) :: w
  integer n, status
  !
#if KEY_PARAFULL==1
#if KEY_PARASCAL==1
#error  'Illegal parallel compile options'
#endif 
  if(mode.eq.2) return
#elif KEY_PARASCAL==1 || KEY_SPACDEC==1
#else /**/
#error  'Illegal parallel compile options'
#endif 
  !
#if KEY_PARINFNTY==1
  if(qinfinity) return                 
#endif
  if(numnod.eq.1) return
  n=iparpt(numnod)
  call chmalloc('paral1.src','VDGSUM','W',N,crl=W)
#if KEY_CMPI==1 /*Charmm_MPI*/
  !     Uses CHARMM MPI implementation
  call cmpi_red_scat(x,w,nparpt,cmpi_double_precision, &
       cmpi_sum,cmpi_comm_world,status)
  x(iparpt(mynod)+1:iparpt(mynod)+nparpt(mynod)) = w(1:nparpt(mynod))
  call cmpi_red_scat(y,w,nparpt, &
       cmpi_double_precision,cmpi_sum,cmpi_comm_world, &
       status)
  y(iparpt(mynod)+1 : iparpt(mynod)+nparpt(mynod)) = w(1:nparpt(mynod))
  call cmpi_red_scat(z,w,nparpt, &
       cmpi_double_precision,cmpi_sum,cmpi_comm_world, &
       status)
  z(iparpt(mynod)+1 : iparpt(mynod)+nparpt(mynod)) = w(1:nparpt(mynod))
#else /* (Charmm_MPI)*/
  !     Uses compliant calls.
  call mpi_reduce_scatter(x,w,nparpt, &
       mpi_double_precision,mpi_sum,comm_charmm, &
       status)
  x(iparpt(mynod)+1 : iparpt(mynod)+nparpt(mynod)) = w(1:nparpt(mynod))
  call mpi_reduce_scatter(y,w,nparpt, &
       mpi_double_precision,mpi_sum,comm_charmm, &
       status)
  y(iparpt(mynod)+1 : iparpt(mynod)+nparpt(mynod)) = w(1:nparpt(mynod))
  call mpi_reduce_scatter(z,w,nparpt, &
       mpi_double_precision,mpi_sum,comm_charmm, &
       status)
  z(iparpt(mynod)+1 : iparpt(mynod)+nparpt(mynod)) = w(1:nparpt(mynod))
#endif /* (Charmm_MPI)*/
  call chmdealloc('paral1.src','VDGSUM','W',N,crl=W)
  !
  return
end subroutine vdgsum

SUBROUTINE RI1VDGBR(X)
  !-----------------------------------------------------------------------
  !
  !    Wraping routine for integer vector distributed global broadcast.
  !    One array only!
  !    FIXME: This routine needs to be standardized
  !
  use chm_kinds
  use parallel
  use memory
#if KEY_MPI==1
  use mpi      
#endif
  implicit none
  !
  real(chm_real),allocatable,dimension(:) :: WPAR
  real(chm_real) x(*)
  integer npar,status
  !
#if KEY_PARINFNTY==1
  if(qinfinity) return                 
#endif
  if(numnod.eq.1) return

  npar=iparpt(numnod)
  call chmalloc('paral1.src','RI1VDGBR','WPAR',NPAR,crl=WPAR)
#if KEY_CMPI==1
  CALL CMPI_GATHERV(X,NPARPT(MYNOD), &
       CMPI_DOUBLE_PRECISION,WPAR,NPARPT,IPARPT, &
       CMPI_DOUBLE_PRECISION,CMPI_COMM_WORLD,STATUS)
  X(1:NPAR) = WPAR(1:NPAR)
#else /**/
  call mpi_allgatherv(x(iparpt(mynod)+1),nparpt(mynod), &
       MPI_DOUBLE_PRECISION,WPAR,NPARPT,IPARPT, &
       MPI_DOUBLE_PRECISION,COMM_CHARMM,STATUS)
  x(1:npar) = wpar(1:npar)
#endif 
  call chmdealloc('paral1.src','RI1VDGBR','WPAR',NPAR,crl=wpar)
  return
END SUBROUTINE RI1VDGBR

subroutine vdgbr(x,y,z,mode)
  !-----------------------------------------------------------------------
  !
  !    Wraping routine for vector distributed global broadcast.
  !
  !
  !    MODE=0 - Do just the inner broadcast.
  !          (assume values are needed only on processors containing block).
  !    MODE=1 - Do the complete global broadcast (inner followed by outer).
  !          (assume values are needed everywhere)
  !    MODE=2 - Do just the outer broadcast.
  !          (assume values are correct for all nodes containing block).
  !
  ! The MODE values 0 and 1 are identical and MODE 2 is ignored for PARAFULL.
  !
  use chm_kinds
  !
  use dimens_fcm
  use parallel
  use exfunc
  use memory
#if KEY_CMPI==0
  use mpi      
#endif
  !
  implicit none
  !
  real(chm_real) x(*), y(*), z(*)
  integer mode

  real(chm_real),allocatable,dimension(:) :: W
  integer n,status
  !
  !     =====================================
  !
#if KEY_PARAFULL==1
#if KEY_PARASCAL==1
#error  'Illegal parallel compile options'
#endif 
  if(mode.eq.2) return
#elif KEY_PARASCAL==1 || KEY_SPACDEC==1
#else /**/
#error  'Illegal parallel compile options'
#endif 
  !
#if KEY_PARINFNTY==1
  if(qinfinity) return                 
#endif
  if(numnod.eq.1) return
  n=iparpt(numnod)
  call chmalloc('paral1.src','VDGBR','W',N,crl=W)
#if KEY_CMPI==1
  !     X is all vector not partial as in standard MPI.
  !     In the current implementation NPARPT is ignored
  call cmpi_gatherv(x,nparpt(mynod), &
       cmpi_double_precision,w,nparpt,iparpt, &
       cmpi_double_precision,cmpi_comm_world,status)
  x(1:n) = w(1:n)
  call cmpi_gatherv(y,nparpt(mynod), &
       cmpi_double_precision,w,nparpt,iparpt, &
       cmpi_double_precision,cmpi_comm_world,status)

  y(1:n) = w(1:n)
  call cmpi_gatherv(z,nparpt(mynod), &
       cmpi_double_precision,w,nparpt,iparpt, &
       cmpi_double_precision,cmpi_comm_world,status)
  z(1:n) = w(1:n)
#else /**/
  !     Uses compliant calls
  call mpi_allgatherv(x(iparpt(mynod)+1),nparpt(mynod), &
       mpi_double_precision,w,nparpt,iparpt, &
       mpi_double_precision,comm_charmm,status)
  x(1:n) = w(1:n)
  call mpi_allgatherv(y(iparpt(mynod)+1),nparpt(mynod), &
       mpi_double_precision,w,nparpt,iparpt, &
       mpi_double_precision,comm_charmm,status)
  y(1:n) = w(1:n)
  call mpi_allgatherv(z(iparpt(mynod)+1),nparpt(mynod), &
       mpi_double_precision,w,nparpt,iparpt, &
       mpi_double_precision,comm_charmm,status)
  z(1:n) = w(1:n)
#endif 

  call chmdealloc('paral1.src','VDGBR','W',N,crl=W)

  return
end subroutine vdgbr

SUBROUTINE VDGBRE(X,KPARPT)
  !-----------------------------------------------------------------------
  !
  !     Global vector distributed broadcast for single vector partitioned
  !     in KPARPT array.
  !
  use chm_kinds

  use dimens_fcm
  use parallel
  use exfunc
  use memory
#if KEY_CMPI==0
  use mpi      
#endif

  implicit none

  real(chm_real),allocatable,dimension(:) :: W
  real(chm_real) X(*)
  integer kparpt(0:*)
  integer n,status,i,counts(0:maxnode)
  !
#if KEY_PARINFNTY==1
  if(qinfinity) return                 
#endif
  if(numnod.eq.1) return
#if KEY_CMPI==1
  !     For PME performance reason call inplace routine!
  call cimpi_gatherv(x,cmpi_double_precision,kparpt, &
       comm_charmm,status)
#else /**/
  n=kparpt(numnod)
  call chmalloc('paral1.src','VDGBRE','W',N,crl=W)
  !
  do i = 0, numnod-1
     counts(i)=kparpt(i+1)-kparpt(i)
  enddo

  call mpi_allgatherv(x(kparpt(mynod)+1),counts(mynod), &
       mpi_double_precision,w,counts,kparpt, &
       mpi_double_precision,comm_charmm,status)
  x(1:n) = w(1:n)
  call chmdealloc('paral1.src','VDGBRE','W',N,crl=W)
#endif 

  return
end subroutine vdgbre

subroutine gcombs(x,n)
  !-----------------------------------------------------------------------
  !
  !     Global COMBine routine. For scalar real
  !
  use chm_kinds
  use dimens_fcm
  use parallel,only: numnod, comm_charmm
#if KEY_CMPI==1
  use parallel,only: cmpi_double_precision,cmpi_sum
#endif
  use stream
  !
  use exfunc
  use memory
#if KEY_CMPI==0
  use mpi      
#endif
  !
  implicit none
  !
  real(chm_real) X
  integer, optional :: n

  real(chm_real),allocatable,dimension(:) :: W
  integer status
  integer i,nn
  !
#if KEY_PARINFNTY==1
  if(qinfinity) return                 
#endif
  if(numnod.eq.1) return
! for DDI:  call ddi_gsumf(3334,x,n)
  if (present(n)) then
     if ( n /= 1 ) CALL WRNDIE(-5,'<GCOMBS>', &
     'length either 1 or not specified at all.')
     nn = n
  else
     nn = 1
  endif
#if KEY_CMPI==1
  call chmalloc('paral1.src','gcombs','w',nn,crl=w)
  call cimpi_allred(x,w,nn,cmpi_double_precision,cmpi_sum, &
       comm_charmm,status)
  call chmdealloc('paral1.src','gcombs','w',nn,crl=w)
#else /**/
  !     Uses compliant calls.
  call chmalloc('paral1.src','GCOMBS','W',NN,crl=W)
  call mpi_allreduce(x,w,nn,mpi_double_precision,mpi_sum, &
       comm_charmm,status)
  x = w(1)
  call chmdealloc('paral1.src','GCOMBS','W',NN,crl=w)
  !
#endif 
  !
  return
end subroutine gcombs

subroutine gcombsi(x,n)
  !-----------------------------------------------------------------------
  !
  !     Global COMBine routine. For scalar integers
  !
  use chm_kinds
  use dimens_fcm
  use parallel,only: numnod, comm_charmm
#if KEY_CMPI==1
  use parallel,only: cmpi_integer,cmpi_sum
#endif
  use stream
  !
  use exfunc
  use memory
#if KEY_CMPI==0
  use mpi      
#endif
  !
  implicit none
  !
  integer ,allocatable,dimension(:) :: W
  integer X
  integer, optional :: n

  integer status
  integer nn, i
  !
#if KEY_PARINFNTY==1
  if(qinfinity) return                 
#endif
  if(numnod.eq.1) return

  if (present(n)) then
     if ( n /= 1 ) CALL WRNDIE(-5,'<GCOMBSI>', &
     'length either 1 or not specified at all.')
     nn = n
  else
     nn = 1
  endif

! for DDI:  call ddi_gsumf(3334,x,n)
#if KEY_CMPI==1
  call chmalloc('paral1.src','gcombsi','w',nn,intg=w)
  call cimpi_allred(x,w,nn,cmpi_integer,cmpi_sum, &
       comm_charmm,status)
  call chmdealloc('paral1.src','gcombsi','w',nn,intg=w)
#else /**/
  !     Uses compliant calls.
  call chmalloc('paral1.src','GCOMBSI','W',NN,intg=W)
  call mpi_allreduce(x,w,nn,mpi_integer,mpi_sum, &
       comm_charmm,status)
  x = w(1)
  call chmdealloc('paral1.src','GCOMBSI','W',NN,intg=w)
  !
#endif 
  !
  return
end subroutine gcombsi

subroutine gcombr(x,n)
  !-----------------------------------------------------------------------
  !
  !     Global COMBine routine. For real arrays
  !
  use chm_kinds
  use dimens_fcm
  use parallel,only: numnod, comm_charmm
#if KEY_CMPI==1
  use parallel,only: cmpi_double_precision,cmpi_sum
#endif
  use stream
  !
  use exfunc
  use memory
#if KEY_CMPI==0
  use mpi      
#endif
  !
  implicit none
  !
  real(chm_real),allocatable,dimension(:) :: W
  real(chm_real) X(*)

  integer status
  integer n, i
  !
#if KEY_PARINFNTY==1
  if(qinfinity) return                 
#endif
  if(numnod.eq.1) return
! for DDI:  call ddi_gsumf(3334,x,n)
#if KEY_CMPI==1
  call chmalloc('paral1.src','gcombr','w',n,crl=w)
  call cimpi_allred(x,w,n,cmpi_double_precision,cmpi_sum, &
       comm_charmm,status)
  call chmdealloc('paral1.src','gcombr','w',n,crl=w)
#else /**/
  !     Uses compliant calls.
  call chmalloc('paral1.src','GCOMBR','W',N,crl=W)
  call mpi_allreduce(x,w,n,mpi_double_precision,mpi_sum, &
       comm_charmm,status)
  x(1:n) = w(1:n)
  call chmdealloc('paral1.src','GCOMBR','W',N,crl=w)
  !
#endif 
  !
  return
end subroutine gcombr
subroutine gcombi(x,n)
  !-----------------------------------------------------------------------
  !
  !     Global COMBine routine. For integer arrays
  !
  use chm_kinds
  use dimens_fcm
  use parallel,only: numnod, comm_charmm
#if KEY_CMPI==1
  use parallel,only: cmpi_integer,cmpi_sum
#endif
  use stream
  !
  use exfunc
  use memory
#if KEY_CMPI==0
  use mpi      
#endif
  !
  implicit none
  !
  integer,allocatable,dimension(:) :: W
  integer X(*)

  integer status
  integer n, i
  !
#if KEY_PARINFNTY==1
  if(qinfinity) return                 
#endif
  if(numnod.eq.1) return
! for DDI:  call ddi_gsumf(3334,x,n)
#if KEY_CMPI==1
  call chmalloc('paral1.src','gcombi','w',n,intg=w)
  call cimpi_allred(x,w,n,cmpi_integer,cmpi_sum, &
       comm_charmm,status)
  call chmdealloc('paral1.src','gcombi','w',n,intg=w)
#else /**/
  !     Uses compliant calls.
  call chmalloc('paral1.src','GCOMBI','W',N,intg=W)
  call mpi_allreduce(x,w,n,mpi_double_precision,mpi_sum, &
       comm_charmm,status)
  x(1:n) = w(1:n)
  call chmdealloc('paral1.src','GCOMBI','W',N,intg=w)
  !
#endif 
  !
  return
end subroutine gcombi

subroutine comb_to_root(x, n)
  !---------------------------------------------------------------
  ! Combines (sums) values to the root node by calling mpi_reduce
  ! NOTE: x is changed only in the root node
  ! APH 2011
  !
  use chm_kinds
  use memory
  use mpi
  use parallel
  implicit none
  ! Input / Output
  real(chm_real) x(*)
  integer n
  ! Variables
  real(chm_real), allocatable, dimension(:) :: w
  integer ierror

  if (n <= 0) return
  
  if (mynod == 0) then
     call chmalloc('paral1.src','comb_to_root','w',n,crl=w)
     call mpi_reduce(x, w, n, MPI_REAL8, MPI_SUM, 0, COMM_CHARMM, ierror)
     x(1:n) = w(1:n)
     call chmdealloc('paral1.src','comb_to_root','w',n,crl=w)
  else
     ! NOTE: x in receive buffer is ignored for non-root nodes
     call mpi_reduce(x, x, n, MPI_REAL8, MPI_SUM, 0, COMM_CHARMM, ierror)
  endif

  return
end subroutine comb_to_root

!%% AvdV addition (for gnu only; not tested for other machines)
!#if KEY_TMD==1 /*pll_tmd*/
SUBROUTINE GCOMBMAX(X)
  !-----------------------------------------------------------------------
  !
  !     Global COMBine routine; returns the maximum element
  !
  use chm_kinds
  use dimens_fcm
  use parallel
  !
  use exfunc
  use memory
#if KEY_CMPI==0
  use mpi      
#endif
#if KEY_STRINGM==1 && KEY_MPI==1 /*  VO stringm */
  use mpi      
#endif
  !
  implicit none
  !
  real(chm_real) X,Y(2)
  !

  INTEGER STATUS

  !
  !
#if KEY_PARINFNTY==1
  if(qinfinity) return                 
#endif
  IF(NUMNOD.EQ.1) RETURN
  !     Uses compliant calls.
#if KEY_CMPI==1
#if KEY_MPI==0
  CALL WRNDIE(-5,'<GCOMBMAX>','Implemented only for ##MPI.')
  RETURN
#else /**/
! this is not good for CMPI
!  CALL MPI_ALLREDUCE(X,Y,1,MPI_2DOUBLE_PRECISION,MPI_MAXLOC, &
!       COMM_CHARMM,STATUS)
!  X=Y(1)
#endif 
  !
#else /**/
  CALL MPI_ALLREDUCE(X,Y,1,MPI_2DOUBLE_PRECISION,MPI_MAXLOC, &
       COMM_CHARMM,STATUS)
  X=Y(1)
#endif 
  !
  RETURN
END SUBROUTINE GCOMBMAX
!
!#endif /* (pll_tmd)*/
!%% AvdV end addition
!
SUBROUTINE IVDGBRE(X,KPARPT)
  !-----------------------------------------------------------------------
  !
  !     Global vector distributed broadcast for single vector partitioned
  !     in KPARPT array.
  !
  use chm_kinds

  use dimens_fcm
  use parallel
  use exfunc
  use memory
#if KEY_CMPI==0
  use mpi
#endif

  implicit none

  integer,allocatable,dimension(:) :: W
  INTEGER X(*)
  integer kparpt(0:*)
  integer n,status,i,counts(0:maxnode)
  !
#if KEY_PARINFNTY==1
  if(qinfinity) return
#endif
  if(numnod.eq.1) return
#if KEY_CMPI==1
  !     For PME performance reason call inplace routine!
  call cimpi_gatherv(x,CMPI_INTEGER,kparpt, &
       comm_charmm,status)
#else /**/
  n=kparpt(numnod)
  call chmalloc('paral1.src','IVDGBRE','W',N,intg=W)
  !
  do i = 0, numnod-1
     counts(i)=kparpt(i+1)-kparpt(i)
  enddo

  call mpi_allgatherv(x(kparpt(mynod)+1),counts(mynod), &
       MPI_INTEGER,w,counts,kparpt, &
       MPI_INTEGER,comm_charmm,status)
  x(1:n) = w(1:n)
  call chmdealloc('paral1.src','IVDGBRE','W',N,intg=W)
#endif

  return
end subroutine ivdgbre
!
SUBROUTINE IGCOMB(X,N)
  !-----------------------------------------------------------------------
  !
  !     Global COMBine routine. Integer version.
  !
  use chm_kinds
  !
  use dimens_fcm
  use parallel
  !
  use exfunc
  use memory
#if KEY_CMPI==0
  use mpi      
#endif
  !
  implicit none
  !
  integer,allocatable,dimension(:) :: W
  INTEGER X(*)

  INTEGER STATUS
  INTEGER N, I
  !
#if KEY_PARINFNTY==1
  if(qinfinity) return                 
#endif
  IF(NUMNOD.EQ.1) RETURN
#if KEY_CMPI==1
  call chmalloc('paral1.src','IGCOMB','W',N,intg=W)
  CALL CIMPI_ALLRED(X,W,N,CMPI_INTEGER,CMPI_SUM, &
       COMM_CHARMM,STATUS)
  call chmdealloc('paral1.src','IGCOMB','W',N,intg=w)
#else /**/
  !     Uses compliant calls.
  call chmalloc('paral1.src','IGCOMB','W',N,intg=W)
  CALL MPI_ALLREDUCE(X,W,N,MPI_INTEGER,MPI_SUM, &
       COMM_CHARMM,STATUS)
  X(1:N) = W(1:N)
  call chmdealloc('paral1.src','IGCOMB','W',N,intg=w)
  !
#endif 
  !
  RETURN
END SUBROUTINE IGCOMB
!
SUBROUTINE GBOR(X,N)
  !-----------------------------------------------------------------------
  !
  !     Global Bitwise OR routine.
  !
  use chm_kinds
  use dimens_fcm
  use parallel
  !
  use exfunc
  use memory
#if KEY_CMPI==0
  use mpi      
#endif
  !
  implicit none
  integer,allocatable,dimension(:) :: W
  INTEGER X(*), N
  !
  INTEGER STATUS,I
  !
#if KEY_PARINFNTY==1
  if(qinfinity) return                 
#endif
  IF(NUMNOD.EQ.1) RETURN
  call chmalloc('paral1.src','GBOR','W',N,intg=W)
#if KEY_CMPI==1
  !     Call inplace routine
  CALL CIMPI_ALLRED(X,W,N,CMPI_INTEGER,CMPI_BOR, &
       COMM_CHARMM,STATUS)
#else /**/
  !     Uses compliant calls.
  CALL MPI_ALLREDUCE(X,W,N,MPI_INTEGER,MPI_BOR, &
       COMM_CHARMM,STATUS)
  X(1:N) = W(1:N)
#endif 
  !
  call chmdealloc('paral1.src','GBOR','W',N,intg=W)
  !
  RETURN
END SUBROUTINE GBOR
!
SUBROUTINE PSYNC()
  !-----------------------------------------------------------------------
  !
  !     This is a wrapper routine for global sync.
  !
  use chm_kinds
  !
  use dimens_fcm
  use parallel
  use memory
#if KEY_CMPI==0
  use mpi      
#endif
  implicit none
  INTEGER STATUS
  !
#if KEY_PARINFNTY==1
  if(qinfinity) return                 
#endif
  IF(NUMNOD.EQ.1) RETURN
#if KEY_CMPI==1
  ! CMPI_COMM_WORLD is local to some of the cmpi_xxx routines in
  ! paral2.src file. We cannot use COMM_CHARMM here !!!!
  CALL CMPI_BARRIER(CMPI_COMM_WORLD,STATUS)
#else /**/
  CALL MPI_BARRIER(COMM_CHARMM,STATUS)
#endif 
  !
  RETURN
END SUBROUTINE PSYNC
!
SUBROUTINE SWAPD(N,X,Y,Z,XPL,YPL,ZPL,XPR,YPR,ZPR,LPEER,RPEER)
  !-----------------------------------------------------------------------
  ! This routine performs a double swap of data between
  ! {X,Y,Z} and {XPL,YPL,ZPL}, and at the same time also {X,Y,Z} and {XPR,YPR,ZPR}.
  ! X,Y,Z is on MYNOD, XPL,YPL,ZPL is on LPEER,XPR,YPR,ZPR, is on RPEER.
  ! If RPEER=-1 or LPEER=-1, then skip the swap in this direction, ie call GRECSEN
  !
  use chm_kinds
  use dimens_fcm
  use parallel
  use repdstr
  !
  implicit none
  INTEGER N,LPEER,RPEER,LPEERC,RPEERC
  real(chm_real) X(*),Y(*),Z(*),XPL(*),YPL(*),ZPL(*)
  real(chm_real) XPR(*),YPR(*),ZPR(*)
  !
#if KEY_REPDSTR==1
  IF(QREPDSTR)THEN
     LPEERC=LPEER*NUMNOD
     RPEERC=RPEER*NUMNOD
  ELSE
     LPEERC=LPEER
     RPEERC=RPEER
  ENDIF
#else /**/
  LPEERC=LPEER
  RPEERC=RPEER
#endif 
  CALL SWAPD1(N,X,XPL,XPR,LPEERC,RPEERC)
  CALL SWAPD1(N,Y,YPL,YPR,LPEERC,RPEERC)
  CALL SWAPD1(N,Z,ZPL,ZPR,LPEERC,RPEERC)
  !
  RETURN
END SUBROUTINE SWAPD
!
!
#if KEY_REPDSTR==1
SUBROUTINE REPDBR(X)
  !-----------------------------------------------------------------------
  ! This routine performs a parallel/parallel broadcast for
  ! distributed replica code
  !
  use epathmod
  use dimens_fcm
  use parallel
  use repdstr
  !
  implicit none
  !
  real(chm_real) x(*)
  !
  !     This routine is modified from agvdgba in paral3.src
  !     agvdgba is vector ditributed global brodcast
  !     using ring topology, ie suitable for any number of
  !     replicas. Also it does processor mapping for
  !     parallel/parallel
  !
  !      call agvdgba(x,arepdmap,nrepdstr,irepdstr,prepdmap)
  !      wouldnt work because of processor/replica assignment problems
  !      so we make a new one here:
  !
  INTEGER ITYPE,IBIT,ISTEP,ME,IOFF,JOFF
  INTEGER ILEN,JLEN
  !
  ITYPE=1
  ME=irepdstr
  DO ISTEP = 1, nrepdstr-1
     IBIT=nrepdstr-ISTEP+ME
     IOFF=MOD(IBIT+1,nrepdstr)
     JOFF=MOD(IBIT,nrepdstr)
     ILEN=arepdmap(IOFF+1)-arepdmap(IOFF)
     JLEN=arepdmap(JOFF+1)-arepdmap(JOFF)
     CALL GRECSENR(irepdstr,prepdmap,nrepdstr,ITYPE,X(arepdmap(JOFF)+1),JLEN, &
          X(arepdmap(IOFF)+1),ILEN)
     ITYPE=ITYPE+1
  ENDDO
  !
  RETURN
END SUBROUTINE REPDBR
#endif 
!
SUBROUTINE PSNDC(ARRAY, LENGTH)
  !-----------------------------------------------------------------------
  ! This routine performs on node 0 broadcast to all other nodes
  ! and receive from node 0 on all other nodes.
  ! Usually called after read on node 0. For character arrays.
  !
  use chm_kinds
  !
  use dimens_fcm
  use parallel
#if KEY_CMPI==1
  use parallel,only: cmpi_byte,cmpi_comm_world
#endif
#if KEY_CMPI==0
  use mpi      
#endif
  implicit none
  CHARACTER(len=*) ARRAY(*)
  INTEGER LENGTH
  INTEGER STATUS,LENST

  IF((NUMNOD.EQ.1).OR.(LENGTH.EQ.0)) RETURN
  LENST = LEN(ARRAY(1))

#if KEY_CMPI==1
       CALL CMPI_BCAST(ARRAY,LENST*LENGTH,CMPI_BYTE,MASTER_NODE,CMPI_COMM_WORLD, &
       STATUS)
#else /**/
       CALL MPI_BCAST(ARRAY,LENST*LENGTH,MPI_BYTE,MASTER_NODE,COMM_CHARMM, &
       STATUS)
#endif 
  !
  RETURN
END SUBROUTINE PSNDC
!
SUBROUTINE PSNDC2(ARRAY, LENGTH)
  !-----------------------------------------------------------------------
! This is introduced by Anmol for taking care of two dimensional lone pair center array.  
! Currently this is the only case of two dimensional array.             
! Notice that it does not use length coming to subroutine, rather a fixed 1000/1000 array is broadcasted
! This is because host of lone pair center MLPCT is having fixed 2 D size curently.                    
  !
  use chm_kinds
  !
  use dimens_fcm
  use parallel
#if KEY_CMPI==0
  use mpi      
#endif
  implicit none
  CHARACTER(len=*) ARRAY(1000,1000)
  INTEGER LENGTH
  INTEGER STATUS,LENST

  IF((NUMNOD.EQ.1).OR.(LENGTH.EQ.0)) RETURN
  LENST = LEN(ARRAY(1,1))

#if KEY_CMPI==1
       CALL CMPI_BCAST(ARRAY,LENST*1000*1000,CMPI_BYTE,MASTER_NODE,CMPI_COMM_WORLD, &                  
       STATUS)                                                                                         
#else /**/
      CALL MPI_BCAST(ARRAY,LENST*1000*1000,MPI_BYTE,MASTER_NODE,COMM_CHARMM, &
       STATUS)  
#endif 
  !
  RETURN
END SUBROUTINE PSNDC2
!
SUBROUTINE PSND4m(ARRAY, LENGTH)
  !-----------------------------------------------------------------------
  ! This routine performs on node 0 broadcast to all other nodes
  ! and receive from node 0 on all other nodes.
  ! Usually called after read on node 0. For single precision arrays.
  !
  use chm_kinds
  !
  use dimens_fcm
  use parallel,only:comm_charmm,master_node,numnod
#if KEY_CMPI==1
  use parallel,only: cmpi_byte,cmpi_comm_world
#endif
#if KEY_CMPI==0
  use mpi
#endif
  !
  implicit none
  INTEGER ARRAY(*)
  INTEGER LENGTH
  INTEGER STATUS
  !
  IF(NUMNOD.EQ.1) RETURN
#if KEY_CMPI==1
  CALL CMPI_BCAST(ARRAY,4*LENGTH,CMPI_BYTE,MASTER_NODE,CMPI_COMM_WORLD,STATUS)
#else /**/
  CALL MPI_BCAST(ARRAY,4*LENGTH,MPI_BYTE,MASTER_NODE,COMM_CHARMM,STATUS)
#endif 
  RETURN
END SUBROUTINE PSND4M
SUBROUTINE PSND4r(ARRAY, LENGTH)
  !-----------------------------------------------------------------------
  ! This routine performs on node 0 broadcast to all other nodes
  ! and receive from node 0 on all other nodes.
  ! Usually called after read on node 0. For single precision arrays.
  !
  use chm_kinds
  !
  use dimens_fcm
  use parallel,only:comm_charmm,master_node,numnod
#if KEY_CMPI==1
  use parallel,only: cmpi_byte,cmpi_comm_world
#endif
#if KEY_CMPI==0
  use mpi
#endif
  !
  implicit none
  real(chm_real4) ARRAY(*)
  INTEGER LENGTH
  INTEGER STATUS
  !
  IF(NUMNOD.EQ.1) RETURN
#if KEY_CMPI==1
  CALL CMPI_BCAST(ARRAY,4*LENGTH,CMPI_BYTE,MASTER_NODE,CMPI_COMM_WORLD,STATUS)
#else /**/
  CALL MPI_BCAST(ARRAY,4*LENGTH,MPI_BYTE,MASTER_NODE,COMM_CHARMM,STATUS)
#endif 
  RETURN
END SUBROUTINE PSND4R
SUBROUTINE PSND4s(variable, LENGTH)
  !-----------------------------------------------------------------------
  ! This routine performs on node 0 broadcast to all other nodes
  ! and receive from node 0 on all other nodes.
  ! Usually called after read on node 0. For single precision arrays.
  !
  use chm_kinds
  !
  use dimens_fcm
  use parallel,only:comm_charmm,master_node,numnod
#if KEY_CMPI==1
  use parallel,only: cmpi_byte,cmpi_comm_world
#endif
#if KEY_CMPI==0
  use mpi
#endif
  !
  implicit none
  INTEGER variable
  INTEGER LENGTH
  INTEGER STATUS
  !
  IF(NUMNOD.EQ.1) RETURN
  if(length/=1) CALL WRNDIE(5,'<PSND4S>','this routine should work for length=1 only')

#if KEY_CMPI==1
  CALL CMPI_BCAST(variable,4*LENGTH,CMPI_BYTE,MASTER_NODE,CMPI_COMM_WORLD,STATUS)
#else /**/
  CALL MPI_BCAST(variable,4*LENGTH,MPI_BYTE,MASTER_NODE,COMM_CHARMM,STATUS)
#endif 
  RETURN
END SUBROUTINE PSND4S
SUBROUTINE PSND4l(variable, LENGTH)
  !-----------------------------------------------------------------------
  ! This routine performs on node 0 broadcast to all other nodes
  ! and receive from node 0 on all other nodes.
  ! Usually called after read on node 0. For single precision arrays.
  !
  use chm_kinds
  !
  use dimens_fcm
  use parallel,only:comm_charmm,master_node,numnod
#if KEY_CMPI==1
  use parallel,only: cmpi_byte,cmpi_comm_world
#endif
#if KEY_CMPI==0
  use mpi
#endif
  !
  implicit none
  logical variable
  INTEGER LENGTH
  INTEGER STATUS
  !
  IF(NUMNOD.EQ.1) RETURN
  if(length/=1) CALL WRNDIE(5,'<PSND4L>','this routine should work for length=1 only')

#if KEY_CMPI==1
  CALL CMPI_BCAST(variable,4*LENGTH,CMPI_BYTE,MASTER_NODE,CMPI_COMM_WORLD,STATUS)
#else /**/
  CALL MPI_BCAST(variable,4*LENGTH,MPI_BYTE,MASTER_NODE,COMM_CHARMM,STATUS)
#endif 
  RETURN
END SUBROUTINE PSND4l
SUBROUTINE PSND4lm(variable, LENGTH)
  !-----------------------------------------------------------------------
  ! This routine performs on node 0 broadcast to all other nodes
  ! and receive from node 0 on all other nodes.
  ! Usually called after read on node 0. For single precision arrays.
  !
  use chm_kinds
  !
  use dimens_fcm
  use parallel,only:comm_charmm,master_node,numnod
#if KEY_CMPI==1
  use parallel,only: cmpi_byte,cmpi_comm_world
#endif
#if KEY_CMPI==0
  use mpi
#endif
  !
  implicit none
  logical variable(*)
  INTEGER LENGTH
  INTEGER STATUS
  !
  IF(NUMNOD.EQ.1) RETURN
#if KEY_CMPI==1
  CALL CMPI_BCAST(variable,4*LENGTH,CMPI_BYTE,MASTER_NODE,CMPI_COMM_WORLD,STATUS)
#else /**/
  CALL MPI_BCAST(variable,4*LENGTH,MPI_BYTE,MASTER_NODE,COMM_CHARMM,STATUS)
#endif 
  RETURN
END SUBROUTINE PSND4lm

subroutine psnd4_comm(comm,array, length)
  !-----------------------------------------------------------------------
  ! This routine performs on node 0 broadcast to all other nodes
  ! and receive from node 0 on all other nodes.
  ! Usually called after read on node 0. For single precision arrays.
  !
  use chm_kinds
  !
  use dimens_fcm
  use parallel
#if KEY_CMPI==0
  use mpi      
#endif

  implicit none
  integer array(*)
  integer length,comm
  integer status,comm_size

  call mpi_comm_size(comm,comm_size,status)
  if(comm_size == 1) return
#if KEY_CMPI==1
  CALL CMPI_BCAST(ARRAY,4*LENGTH,CMPI_BYTE,MASTER_NODE,CMPI_COMM_WORLD,STATUS)
#else /**/
  call mpi_bcast(array,4*length,mpi_byte,0,comm,status)
#endif 
  return
end subroutine psnd4_comm

SUBROUTINE PSND8m(ARRAY, LENGTH)
  !-----------------------------------------------------------------------
  ! This routine performs on node 0 broadcast to all other nodes
  ! and receive from node 0 on all other nodes.
  ! Usually called after read on node 0. For real(chm_real) precision arrays.
  !
  use chm_kinds
  !
  use dimens_fcm
  use parallel,only:comm_charmm,master_node,numnod
#if KEY_CMPI==1
  use parallel,only: cmpi_byte,cmpi_comm_world
#endif
#if KEY_CMPI==0
  use mpi      
#endif
  !
  implicit none
  real(chm_real) ARRAY(*)
  INTEGER LENGTH
  INTEGER STATUS
  !
  IF(NUMNOD.EQ.1) RETURN
#if KEY_CMPI==1
  CALL CMPI_BCAST(ARRAY,8*LENGTH,CMPI_BYTE,MASTER_NODE,CMPI_COMM_WORLD,STATUS)
#else /**/
  CALL MPI_BCAST(ARRAY,8*LENGTH,MPI_BYTE,MASTER_NODE,COMM_CHARMM,STATUS)
#endif 
  RETURN
END SUBROUTINE PSND8m
SUBROUTINE PSND8i(ARRAY, LENGTH)
  !-----------------------------------------------------------------------
  ! This routine performs on node 0 broadcast to all other nodes
  ! and receive from node 0 on all other nodes.
  ! Usually called after read on node 0. For real(chm_real) precision arrays.
  !
  use chm_kinds
  !
  use dimens_fcm
  use parallel,only:comm_charmm,master_node,numnod
#if KEY_CMPI==1
  use parallel,only: cmpi_byte,cmpi_comm_world
#endif
#if KEY_CMPI==0
  use mpi      
#endif
  !
  implicit none
  integer(chm_int8) ARRAY(*)
  INTEGER LENGTH
  INTEGER STATUS
  !
  IF(NUMNOD.EQ.1) RETURN
#if KEY_CMPI==1
  CALL CMPI_BCAST(ARRAY,8*LENGTH,CMPI_BYTE,MASTER_NODE,CMPI_COMM_WORLD,STATUS)
#else /**/
  CALL MPI_BCAST(ARRAY,8*LENGTH,MPI_BYTE,MASTER_NODE,COMM_CHARMM,STATUS)
#endif 
  RETURN
END SUBROUTINE PSND8i
SUBROUTINE PSND8s(ARRAY, LENGTH)
  !-----------------------------------------------------------------------
  ! This routine performs on node 0 broadcast to all other nodes
  ! and receive from node 0 on all other nodes.
  ! Usually called after read on node 0. For real(chm_real) precision arrays.
  !
  use chm_kinds
  !
  use dimens_fcm
  use parallel,only:comm_charmm,master_node,numnod
#if KEY_CMPI==1
  use parallel,only: cmpi_byte,cmpi_comm_world
#endif
#if KEY_CMPI==0
  use mpi      
#endif
  !
  implicit none
  real(chm_real) ARRAY
  INTEGER LENGTH
  INTEGER STATUS
  !
  IF(NUMNOD.EQ.1) RETURN
#if KEY_CMPI==1
  CALL CMPI_BCAST(ARRAY,8*LENGTH,CMPI_BYTE,MASTER_NODE,CMPI_COMM_WORLD,STATUS)
#else /**/
  CALL MPI_BCAST(ARRAY,8*LENGTH,MPI_BYTE,MASTER_NODE,COMM_CHARMM,STATUS)
#endif 
  RETURN
END SUBROUTINE PSND8s

subroutine psnd8_comm(comm,array, length)
  !-----------------------------------------------------------------------
  ! this routine performs on node 0 broadcast to all other nodes
  ! and receive from node 0 on all other nodes.
  ! usually called after read on node 0. for real(chm_real) precision arrays.
  !
  use chm_kinds
  !
  use dimens_fcm
  use parallel
  use mpi      
  !
  implicit none
  integer,intent(in) :: length,comm
  real(chm_real),intent(inout),dimension(length) :: array
  integer status,comm_size,nod0

  call mpi_comm_size(comm,comm_size,status)
  nod0 = 0
  if(comm_size == 1) return
  call mpi_bcast(array,8*length,mpi_byte,nod0,comm,status)
  return

end subroutine psnd8_comm

#if KEY_ENSEMBLE==1
SUBROUTINE PSYNC_WORLD()
  !-----------------------------------------------------------------------
  !
  !     This is a wrapper routine for global sync.
  !
  use chm_kinds
  !
  use dimens_fcm
  use parallel
  use ensemble, only: nensem
  use memory
#if KEY_CMPI==0
  use mpi      
#endif
  implicit none
  INTEGER STATUS
  !
#if KEY_PARINFNTY==1
  if(qinfinity) return                 
#endif
  if (numnod == 1 .and. nensem == 1) return
#if KEY_CMPI==1
  CALL CMPI_BARRIER(CMPI_COMM_WORLD,STATUS)
#else /**/
  CALL MPI_BARRIER(MPI_COMM_WORLD,STATUS)
#endif 
  !
  RETURN
END SUBROUTINE PSYNC_WORLD

SUBROUTINE PSNDC_WORLD(ARRAY, LENGTH)
  !-----------------------------------------------------------------------
  ! This routine performs on node 0 broadcast to all other nodes
  ! and receive from node 0 on all other nodes.
  ! Usually called after read on node 0. For character arrays.
  !
  use chm_kinds
  !
  use dimens_fcm
  use parallel
  use ensemble, only: nensem
#if KEY_CMPI==0
  use mpi      
#endif
  implicit none
  CHARACTER(len=*) ARRAY(*)
  INTEGER LENGTH
  INTEGER STATUS,LENST
  !
  IF(((numnod == 1 .and. nensem == 1)).OR.(LENGTH.EQ.0)) RETURN
  LENST = LEN(ARRAY(1))
#if KEY_CMPI==1
  CALL CMPI_BCAST(ARRAY,LENST*LENGTH,CMPI_BYTE,0,CMPI_COMM_WORLD, &
       STATUS)
#else /**/
  CALL MPI_BCAST(ARRAY,LENST*LENGTH,MPI_BYTE,0,MPI_COMM_WORLD, &
       STATUS)
#endif 
  !
  RETURN
END SUBROUTINE PSNDC_WORLD
!
SUBROUTINE PSND4_WORLD(ARRAY, LENGTH)
  !-----------------------------------------------------------------------
  ! This routine performs on node 0 broadcast to all other nodes
  ! and receive from node 0 on all other nodes.
  ! Usually called after read on node 0. For single precision arrays.
  !
  use chm_kinds
  !
  use dimens_fcm
  use parallel
  use ensemble, only: nensem
#if KEY_CMPI==0
  use mpi      
#endif
  !
  implicit none
  INTEGER ARRAY(*)
  INTEGER LENGTH
  INTEGER STATUS
  !
  if (numnod == 1 .and. nensem == 1) return
#if KEY_CMPI==1
  CALL CMPI_BCAST(ARRAY,4*LENGTH,CMPI_BYTE,0,CMPI_COMM_WORLD,STATUS)
#else /**/
  CALL MPI_BCAST(ARRAY,4*LENGTH,MPI_BYTE,0,MPI_COMM_WORLD,STATUS)
#endif 
  RETURN
END SUBROUTINE PSND4_WORLD
!
SUBROUTINE PSND8_WORLD(ARRAY, LENGTH)
  !-----------------------------------------------------------------------
  ! This routine performs on node 0 broadcast to all other nodes
  ! and receive from node 0 on all other nodes.
  ! Usually called after read on node 0. For real(chm_real) precision arrays.
  !
  use chm_kinds
  !
  use dimens_fcm
  use parallel
  use ensemble, only: nensem
#if KEY_CMPI==0
  use mpi      
#endif
  !
  implicit none
  real(chm_real) ARRAY(*)
  INTEGER LENGTH
  INTEGER STATUS
  !
  if (numnod == 1 .and. nensem == 1) return
#if KEY_CMPI==1
  CALL CMPI_BCAST(ARRAY,8*LENGTH,CMPI_BYTE,0,CMPI_COMM_WORLD,STATUS)
#else /**/
  CALL MPI_BCAST(ARRAY,8*LENGTH,MPI_BYTE,0,MPI_COMM_WORLD,STATUS)
#endif 
  RETURN
END SUBROUTINE PSND8_WORLD
#endif 
!
SUBROUTINE LOCSPACCOMX(TO,FROM,LS,MS,XS,YS,ZS,LR,MR,XR,YR,ZR)
  !-----------------------------------------------------------------------
  !
  !     This routine makes one or two way communication. In the two way
  !     case we don't need to worry about the order of send & receive, which
  !     prevents the deadlocks.
  !
  !     All receiving arrays have the same size LR (atoms on the same CPU)
  !     All sending arrays have the same size LS (atoms on the same CPU)
  !     Maybe not used any more...
  !
  !         ****** NO 64 BIT SUPPORT ??? *******
  !
  use chm_kinds
  implicit none
  !
  INTEGER TO,FROM,LS,LR,MS(*),MR(*)
  real(chm_real) XS(*),YS(*),ZS(*),XR(*),YR(*),ZR(*)
  !
  return
  CALL GRECSEN(TO,1,MR,LR/2+1,MS,LS/2+1)
  CALL GRECSEN(TO,2,XR,LR,XS,LS)
  CALL GRECSEN(TO,3,YR,LR,YS,LS)
  CALL GRECSEN(TO,4,ZR,LR,ZS,LS)
  !
  RETURN
END SUBROUTINE LOCSPACCOMX
!
SUBROUTINE LOCINTCOM(TO,FROM,IS,IR)
  !-----------------------------------------------------------------------
  !
  !     This routine makes one or two way communication. In the two way
  !     case we don't need to worry about the order of send & receive, which
  !     prevents the deadlocks.
  !
  !     It works for 1 integer
  !
  use chm_kinds
  use parallel
#if KEY_MPI==1
  use mpi      
#endif
  implicit none
  !
  INTEGER TO,FROM,IS,IR
  !
#if KEY_MPI==1
  !
  !     We need everything integer*4 here, so it works in
  !     all environments ie. 32 bit and 64 bit
  !
  INTEGER*4 M_STAT(MPI_STATUS_SIZE,2),IERR,REQ(2),IX,TAG
  INTEGER*4 LTO,LFROM,LIS,LIR
  INTEGER*4 IONE,mpiint,mpicomw
  !
  TAG=1
  IONE=1
  LTO=TO
  LFROM=FROM
  LIS=IS
  MPIINT=MPI_INTEGER
  MPICOMW=COMM_CHARMM
  !
  IX=0
  IF(TO.GT.-1)THEN
     IX=IX+1
     CALL MPI_ISEND(LIS,IONE,MPIINT,LTO,TAG,MPICOMW,REQ(IX),IERR)
  ENDIF
  IF(FROM.GT.-1)THEN
     IX=IX+1
     CALL MPI_IRECV(LIR,IONE,MPIINT,LFROM,TAG,MPICOMW,REQ(IX),IERR)
  ENDIF
  !
  IF((TO.GT.-1).OR.(FROM.GT.-1))THEN
     CALL MPI_WAITALL(IX,REQ,M_STAT,IERR)
  ENDIF
  ir=lir
  !
#else /**/
  CALL WRNDIE(-1,'<LOCINTCOM>','Implemented only for ##MPI.')
#endif 
  RETURN
END SUBROUTINE LOCINTCOM
!
SUBROUTINE LOCSPACCOM(TO,FROM,LS,MS,XS,YS,ZS,LR,MR,XR,YR,ZR)
  !-----------------------------------------------------------------------
  !
  !     This routine makes one or two way communication. In the two way
  !     case we don't need to worry about the order of send & receive, which
  !     prevents the deadlocks.
  !
  !     All receiving arrays have the same size LR (atoms on the same CPU)
  !     All sending arrays have the same size LS (atoms on the same CPU)
  !
  use chm_kinds
  use parallel
#if KEY_MPI==1
  use mpi      
#endif
#if KEY_MULTICOM==1 /*  VO stringm */
  use multicom_aux 
#endif
  implicit none
  !
  INTEGER TO,FROM,LS,LR,MS(*),MR(*)
  real(chm_real) XS(*),YS(*),ZS(*),XR(*),YR(*),ZR(*)
  !
#if KEY_MPI==1
  !
  !     We need everything integer*4 here, so it works in
  !     all environments ie. 32 bit and 64 bit
  !
  INTEGER*4 M_STAT(MPI_STATUS_SIZE,2),IERR,REQ(2),IX,TAG
  INTEGER*4 lto,lfrom,lls,llr,mpicomw,mpiint,mpidprec,i
  integer*4,allocatable,dimension(:) :: lms,lmr
  !
  TAG=1
  mpiint=MPI_INTEGER
  mpidprec=MPI_DOUBLE_PRECISION
  mpicomw=COMM_CHARMM
  lls=ls
  llr=lr
  lto=to
  lfrom=from
  !
#if KEY_MULTICOM==1 /*  VO stringm */
  if (MPI_COMM_LOCAL.eq.MPI_COMM_NULL) return 
#endif
  !
  IX=0
  IF((TO.GT.-1).and.(ls.gt.0))THEN
     allocate(lms(ls))
     lms(1:ls) = ms(1:ls)
     IX=IX+1
     CALL MPI_ISEND(LMS,LLS,MPIINT,LTO,TAG,MPICOMW,REQ(IX),IERR)
  ENDIF
  IF((FROM.GT.-1).and.(lr.gt.0))THEN
     IX=IX+1
     allocate(lmr(lr))
     CALL MPI_IRECV(LMR,LLR,MPIINT,LFROM,TAG,MPICOMW,REQ(IX),IERR)
  ENDIF
  !
  IF((TO.GT.-1).OR.(FROM.GT.-1))THEN
     CALL MPI_WAITALL(IX,REQ,M_STAT,IERR)
  ENDIF
  !
  !     safe to deallocate now...
  IF((TO.GT.-1).and.(ls.gt.0))THEN
     deallocate(lms)
  endif
  IF((FROM.GT.-1).and.(lr.gt.0))THEN
     mr(1:lr) = lmr(1:lr)
     deallocate(lmr)
  endif
  !
  IX=0
  IF(TO.GT.-1)THEN
     IX=IX+1
     CALL MPI_ISEND(XS,LLS,MPIDPREC,LTO,TAG,MPICOMW,REQ(IX),IERR)
  ENDIF
  IF(FROM.GT.-1)THEN
     IX=IX+1
     CALL MPI_IRECV(XR,LLR,MPIDPREC,LFROM,TAG,MPICOMW,REQ(IX),IERR)
  ENDIF
  !
  IF((TO.GT.-1).OR.(FROM.GT.-1))THEN
     CALL MPI_WAITALL(IX,REQ,M_STAT,IERR)
  ENDIF
  !
  IX=0
  IF(TO.GT.-1)THEN
     IX=IX+1
     CALL MPI_ISEND(YS,LLS,MPIDPREC,LTO,TAG,MPICOMW,REQ(IX),IERR)
  ENDIF
  IF(FROM.GT.-1)THEN
     IX=IX+1
     CALL MPI_IRECV(YR,LLR,MPIDPREC,LFROM,TAG,MPICOMW,REQ(IX),IERR)
  ENDIF
  !     
  IF((TO.GT.-1).OR.(FROM.GT.-1))THEN
     CALL MPI_WAITALL(IX,REQ,M_STAT,IERR)
  ENDIF
  !
  IX=0
  IF(TO.GT.-1)THEN
     IX=IX+1
     CALL MPI_ISEND(ZS,LLS,MPIDPREC,LTO,TAG,MPICOMW,REQ(IX),IERR)
  ENDIF
  IF(FROM.GT.-1)THEN
     IX=IX+1
     CALL MPI_IRECV(ZR,LLR,MPIDPREC,LFROM,TAG,MPICOMW,REQ(IX),IERR)
  ENDIF
  !     
  IF((TO.GT.-1).OR.(FROM.GT.-1))THEN
     CALL MPI_WAITALL(IX,REQ,M_STAT,IERR)
  ENDIF
  !
#else /**/
  CALL WRNDIE(-1,'<LOCSPACCOM>','Implemented only for ##MPI.')
#endif 
  RETURN
END SUBROUTINE LOCSPACCOM
!
SUBROUTINE LOCFORCCOM(TO,FROM,LS,XS,YS,ZS,LR,XR,YR,ZR)
  !-----------------------------------------------------------------------
  !
  !     This routine makes one or two way communication. In the two way
  !     case we don't need to worry about the order of send & receive, which
  !     prevents the deadlocks.
  !
  !     All receiving arrays have the same size LR (atoms on the same CPU)
  !     All sending arrays have the same size LS (atoms on the same CPU)
  !
  use chm_kinds
  use parallel
#if KEY_MPI==1
  use mpi      
#endif
#if KEY_MULTICOM==1 /*  VO stringm */
  use multicom_aux 
#endif
  implicit none
  !
  INTEGER TO,FROM,LS,LR
  real(chm_real) XS(*),YS(*),ZS(*),XR(*),YR(*),ZR(*)
  !
#if KEY_MPI==1
  !
  !     We need everything integer*4 here, so it works in
  !     all environments ie. 32 bit and 64 bit
  !
  INTEGER*4 M_STAT(MPI_STATUS_SIZE,2),IERR,REQ(2),IX,TAG
  integer*4 lto,lfrom,lls,llr,mpidprec,mpicomw
  !
  TAG=1
  mpidprec=MPI_DOUBLE_PRECISION
  mpicomw=COMM_CHARMM
  lto=to
  lfrom=from
  lls=ls
  llr=lr
  !
  !      IX=1
  !      IF(TO.GT.-1)THEN
  !         CALL MPI_ISEND(MS,LS,MPI_INTEGER,TO,TAG,MPI_COMM_WORLD,
  !     $        REQ(IX),IERR)
  !      ENDIF
  !      IF(FROM.GT.-1)THEN
  !         IX=IX+1
  !         CALL MPI_IRECV(MR,LR,MPI_INTEGER,FROM,TAG,MPI_COMM_WORLD,
  !     $        REQ(IX),IERR)
  !      ENDIF
  !
  !      IF((TO.GT.-1).OR.(FROM.GT.-1))THEN
  !         CALL MPI_WAITALL(IX,REQ,M_STAT,IERR)
  !      ENDIF
  !
#if KEY_MULTICOM==1
  if (MPI_COMM_LOCAL.eq.MPI_COMM_NULL) return 
#endif
  !
  IX=0
  IF(TO.GT.-1)THEN
     IX=IX+1
     CALL MPI_ISEND(XS,LLS,MPIDPREC,LTO,TAG,MPICOMW,REQ(IX),IERR)
  ENDIF
  IF(FROM.GT.-1)THEN
     IX=IX+1
     CALL MPI_IRECV(XR,LLR,MPIDPREC,LFROM,TAG,MPICOMW,REQ(IX),IERR)
  ENDIF
  !
  IF((TO.GT.-1).OR.(FROM.GT.-1))THEN
     CALL MPI_WAITALL(IX,REQ,M_STAT,IERR)
  ENDIF
  !
  IX=0
  IF(TO.GT.-1)THEN
     IX=IX+1
     CALL MPI_ISEND(YS,LLS,MPIDPREC,LTO,TAG,MPICOMW,REQ(IX),IERR)
  ENDIF
  IF(FROM.GT.-1)THEN
     IX=IX+1
     CALL MPI_IRECV(YR,LLR,MPIDPREC,LFROM,TAG,MPICOMW,REQ(IX),IERR)
  ENDIF
  !
  IF((TO.GT.-1).OR.(FROM.GT.-1))THEN
     CALL MPI_WAITALL(IX,REQ,M_STAT,IERR)
  ENDIF
  !
  IX=0
  IF(TO.GT.-1)THEN
     IX=IX+1
     CALL MPI_ISEND(ZS,LLS,MPIDPREC,LTO,TAG,MPICOMW,REQ(IX),IERR)
  ENDIF
  IF(FROM.GT.-1)THEN
     IX=IX+1
     CALL MPI_IRECV(ZR,LLR,MPIDPREC,LFROM,TAG,MPICOMW,REQ(IX),IERR)
  ENDIF
  !
  IF((TO.GT.-1).OR.(FROM.GT.-1))THEN
     CALL MPI_WAITALL(IX,REQ,M_STAT,IERR)
  ENDIF
  !
#else /**/
  CALL WRNDIE(-1,'<LOCSPACCOM>','Implemented only for ##MPI.')
#endif 
  RETURN
END SUBROUTINE LOCFORCCOM
!
SUBROUTINE LOCMAPCOM(TO,FROM,LS,MS,LR,MR)
  !
  !-----------------------------------------------------------------------
  !
  !     This routine makes one or two way communication. In the two way
  !     case we don't need to worry about the order of send & receive, which
  !     prevents the deadlocks.
  !
  !     All receiving arrays have the same size LR (atoms on the same CPU)
  !     All sending arrays have the same size LS (atoms on the same CPU)
  !
  use chm_kinds
  use parallel
#if KEY_MPI==1
  use mpi      
#endif
#if KEY_MULTICOM==1 /*  VO stringm */
  use multicom_aux 
#endif
  implicit none
  !
  INTEGER TO,FROM,LS,LR,MS(*),MR(*)
  !
#if KEY_MPI==1
  !
  !     We need everything integer*4 here, so it works in
  !     all environments ie. 32 bit and 64 bit
  !
  INTEGER*4 M_STAT(MPI_STATUS_SIZE,2),IERR,REQ(2),IX,TAG
  integer*4 lto,lfrom,lls,llr,mpiint,mpicomw
  integer*4,allocatable,dimension(:) :: lms,lmr
  !
  TAG=1
  mpicomw=COMM_CHARMM
  mpiint=MPI_INTEGER
  lto=to
  lfrom=from
  lls=ls
  llr=lr
  !
#if KEY_MULTICOM==1 /*  VO stringm */
  if (MPI_COMM_LOCAL.eq.MPI_COMM_NULL) return 
#endif
  !
  IX=0
  IF(TO.GT.-1)THEN
     IX=IX+1
     allocate(lms(ls))
     lms(1:ls)=ms(1:ls)
     CALL MPI_ISEND(LMS,LLS,MPIINT,LTO,TAG,MPICOMW,REQ(IX),IERR)
  ENDIF
  IF(FROM.GT.-1)THEN
     IX=IX+1
     allocate(lmr(lr))
     CALL MPI_IRECV(LMR,LLR,MPIINT,LFROM,TAG,MPICOMW,REQ(IX),IERR)
  ENDIF

  IF((TO.GT.-1).OR.(FROM.GT.-1))THEN
     CALL MPI_WAITALL(IX,REQ,M_STAT,IERR)
  ENDIF
  !     now safe to deallocate....
  if(to.gt.-1)then
     deallocate(lms)
  endif
  if(from.gt.-1)then
     mr(1:lr)=lmr(1:lr)
     deallocate(lmr)
  endif
  !
#else /**/
  CALL WRNDIE(-1,'<LOCMAPCOM>','Implemented only for ##MPI.')
#endif 
  !
  RETURN
END SUBROUTINE LOCMAPCOM
!
!
SUBROUTINE SPACBR(X,N,CPUMAP)
  !-----------------------------------------------------------------------
  !
  !     This routine broadcasts an array in the spacial decomposed system
  !     It is not efficient yet, since we need to prepare the correct
  !     IPPMAP to use VDGBR. But making IPPMAP needs additional communication
  !     which is not implemented yet.
  !
  !
  use chm_kinds

  !
  use number
  use dimens_fcm
  use parallel
  !
  implicit none
  real(chm_real) X(*)
  INTEGER N,CPUMAP(*)
  !
  INTEGER I
  !
  DO I=1, N
     IF(MYNOD.NE.CPUMAP(I)) X(I) = ZERO
  ENDDO
  !
  CALL GCOMB(X,N)
  !
  RETURN
END SUBROUTINE SPACBR
!
SUBROUTINE BALPRN()
  !-----------------------------------------------------------------------
  !
  !     This routine prints load balance information at the end
  !
  use chm_kinds
  !
  use number
  use stream
  use dimens_fcm
  use parallel
  use memory
  !
  implicit none
  INTEGER I,J
  real(chm_real) TSUMT
  real(chm_real),allocatable,dimension(:) :: TIMONE
  real(chm_real),allocatable,dimension(:,:) :: TIMALL
  !
  ! in case of infinite network speed resume communication
#if KEY_PARINFNTY==1
  qinfinity = .false.                  
#endif
  !
  !     subtract the ewald communication TIMECOMM from TIMEXTE
  !     because of double counting in the code!!
  tmeri(timexte)=tmeri(timexte)-tmeri(timecomm)
  !
  call chmalloc('paral1.src','BALPRN','TIMONE', &
       numnodg,crl=timone)
  call chmalloc('paral1.src','BALPRN','TIMALL', &
       numnodg,tmgnum,crl=timall)
  IF(PRNLEV.GE.2) THEN
     WRITE(OUTU,*)'Parallel load balance (sec.):'
     WRITE(OUTU,*) &
          'Node Eext      Eint   Wait    Comm    EComm   List   Integ   Total'
     TSUMT=ZERO
     DO I = 1, TMGNUM
        TSUMT=TSUMT+TMERI(I)
     ENDDO
     WRITE(OUTU,1510)0, (TMERI(I), I=1,TMGNUM),TSUMT
  ENDIF
  !
  DO I = 1, TMGNUM
     TSUMI(I) = TMERI(I)
  ENDDO
  !
  IF (NUMNOD.GT.1) THEN
     !
     DO I = 1, TMGNUM
        !mh..09-FEB-98 fix for T3E port
        ! use gcomb() routine instead send/receive
        DO J = 1, NUMNOD
           TIMONE(J) = ZERO
        ENDDO
        TIMONE(MYNODP) = TMERI(I)
        CALL GCOMB(TIMONE,NUMNOD)
        DO J = 2, NUMNOD
           TIMALL(J,I) = TIMONE(J)
        ENDDO
     ENDDO
     DO I = 2, NUMNOD
        TSUMT=ZERO
        DO J = 1, TMGNUM
           TSUMT=TSUMT+TIMALL(I,J)
        ENDDO
        IF(PRNLEV.GE.2) WRITE(OUTU,1510) I-1, &
             (TIMALL(I,J), J=1, TMGNUM),TSUMT
        DO J = 1, TMGNUM
           TSUMI(J) = TSUMI(J) + TIMALL(I,J)
        ENDDO
     ENDDO
     IF(PRNLEV.GE.2) THEN
        DO I = 1, TMGNUM
           TSUMI(I) = TSUMI(I)/NUMNOD
        ENDDO
        TSUMT=ZERO
        DO I = 1, TMGNUM
           TSUMT=TSUMT+TSUMI(I)
        ENDDO
        WRITE(OUTU,'(A)') 'PARALLEL> Average timing for all nodes:'
        WRITE(OUTU,1510) &
             NUMNOD,(TSUMI(I), I = 1, TMGNUM),TSUMT
     ENDIF
  ENDIF
  call chmdealloc('paral1.src','BALPRN','TIMONE', &
       numnodg,crl=timone)
  call chmdealloc('paral1.src','BALPRN','TIMALL', &
       numnodg,tmgnum,crl=timall)
1510 FORMAT(I4,13F8.1)
  !
  RETURN
END SUBROUTINE BALPRN
!
SUBROUTINE PARFIN
  !-----------------------------------------------------------------------
  !
  ! Finishing routine - required for some platforms or libraries
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use psf
  !use parallel,only:
  use memory
#if KEY_MULTICOM==1 /*  VO stringm */
  use multicom, only: multicom_cleanup            
#endif
  !
  implicit none
  INTEGER STATUS
#if KEY_GAMESSUK==1
  INTEGER ISTATUS
#endif 
  !
#if KEY_SPACDEC==1
  !     This is not the best place:
  !     what happens if there is delete atom in the script ???
  !
  if(.NOT.QICPUMAP)THEN
     call chmdealloc('paral1.src','PARFIN','ICPUMAP',NATOM,intg=ICPUMAP)
     QICPUMAP=.TRUE.
  ENDIF
#endif 

!  ##IFN ENSEMBLE (ensemble)
#if KEY_CMPI==1 /*cmpilib*/
#if KEY_MPI==1 /*mpilib*/

#if KEY_GAMESSUK==1 /*gukmpi*/
  CALL PARALLEL_END(ISTATUS)
! for DDI:  STATUS=0
! for DDI:  CALL DDI_PEND(STATUS)
#else /* (gukmpi)*/
#if KEY_MULTICOM==1 /*  VO stringm */
  call multicom_cleanup()                  
#endif
  CALL MPI_FINALIZE(STATUS)
#endif /* (gukmpi)*/
#else /* (mpilib)*/
  STATUS=0
#endif /* (mpilib)*/
#else /* (cmpilib)*/
#if KEY_MULTICOM==1 /*  VO stringm */
  call multicom_cleanup()                  
#endif
  CALL MPI_FINALIZE(STATUS)
#endif /* (cmpilib)*/
!  ##ENDIF (ensemble)
  RETURN
END SUBROUTINE PARFIN
!
SUBROUTINE CUBE(MYNOD,NUMNOD,IPPMAP)
  !-----------------------------------------------------------------------
  use chm_kinds
  implicit none
  !
  !     Defines IPPMAP for standard cube network node numbers
  !
  !
  INTEGER MYNOD, NUMNOD, IPPMAP(0:*)
  INTEGER I
  !
  DO I = 0, NUMNOD
     IPPMAP(I)=I
  ENDDO
  RETURN
End SUBROUTINE CUBE
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
#endif /* (pllmain)  PARALLEL*/

#if KEY_PARCMD==1 /*pllpcmd*/
#if KEY_PARALLEL==0 /*pll*/
!==================== PARALLEL COMMAND ==================================

SUBROUTINE PARCMD(COMLYN,COMLEN)
  CHARACTER(len=*) COMLYN
  INTEGER   COMLEN
  CALL WRNDIE(-1,'<PARCMD>','Parallel code not compiled.')
  RETURN
END SUBROUTINE PARCMD

#else /* (pll)*/
!
SUBROUTINE PARCMD(COMLYN,COMLEN)
  !-----------------------------------------------------------------------
  !
  !     PARAllel command parser for controlling parallel execution
  !     Syntax:
  !
  !     PARAllel CONCurrent <int> ...
  !
  !         CONCurrent <int>   specify how many concurrent jobs
  !                            to run in the system
  !
  !     PARAllel FIFO <int>    specify FIFO scheduler in LoBoS with
  !                            static priority <int>
  !
  !     PARAllel BUFF <int>    specify buffer size for send/receive 
  !                            calls. <int> is in real(chm_real) units
  !
  !     PARAllel INFO          Prints the hostname information for each process
  !                            Also fills arrays PARHOST, PARHLEN in parallel.f90
  !
  !     PARAllel RPATH <nrep>  CYCL
  !              RPATH <nrep>  specify the real number of replication
  !                            for the whole system
  !              CYCL          Flag for CYCLyc RPATH command (must match RPATH CYCL)
  !
  !     PARAllel SPLIt <int1> <int2>
  !                           Splits a parallel job into 2 separate jobs
  !                           using <int1> processes for the first job and <int2> for
  !                           the second. Default is int1=numnod-1,int2=1
  !
  use chm_kinds
  use exfunc
  use stream
  use string
  use parallel
  use param_store, only: set_param
#if KEY_GRAPE==1
  use grape, only: qgpusplit                     
  use grapemod, only: igpungr,igpungk,igpunch    
#endif

  implicit none
  ! . Passed variables.
  CHARACTER(len=*) COMLYN
  INTEGER   COMLEN
  INTEGER I,MEOLD,NOLD
  INTEGER CLEN,IPGRP,ISTPRI
  CHARACTER(len=100) CCOM
  INTEGER LIREP,HIREP,NTREP,nsplit(2),isplit(2),jsplit
  LOGICAL QCYCL,QINFORM,lqsplit,lqgpusplit
  !
  !     PARAllel INFO
  QINFORM=(INDXA(COMLYN,COMLEN,'INFO').GT.0)
  IF(QINFORM.AND.(PRNLEV>=2))THEN
     DO I=0,NUMNOD-1
        WRITE(OUTU,*)'Process ',i,' executed on host ', &
             PARHOST(I+1)(1:PARHLEN(I+1))
     ENDDO
  ENDIF
  !
#if KEY_GRAPE==1
  lqgpusplit = index(comlyn,'GPUS') > 0
  if(lqgpusplit) then
     qgpusplit = (INDXA(COMLYN,COMLEN,'GPUS').GT.0)
     igpungr=GTRMI(COMLYN,COMLEN,'NGR',0)
     igpungk=GTRMI(COMLYN,COMLEN,'NGK',0)
     igpunch=GTRMI(COMLYN,COMLEN,'NCH',0)
     ! perform some checking and setup defaults:
     if((igpungr==0).and.(igpungk==0).and.(igpunch==0)) then
        igpungr=numnod
        qgpusplit=.false.
     else if (igpungk > 1) then
        call wrndie(-5,'<PARCMD>','NGK > 1 does not work. GPU PME not parallel')
     else if ((igpungr == numnod).or.(igpunch == numnod)) then
        if((igpungr+igpungk+igpunch) /= numnod) call wrndie(-5,'<PARCMD>', &
             'Wrong NGR,NGK, or NCH specified.')
        qgpusplit=.false.
     else
        if((igpungr+igpungk+igpunch) /= numnod) call wrndie(-5,'<PARCMD>', &
             'Wrong NGR,NGK, or NCH specified.')
     endif
     ! Print the setup:
     if(prnlev>=2) then
        if(qgpusplit)then
           write(outu,*)'Performing split GPU/CPU calculation'
           write(outu,*)'Number of GPUs used in this run:',igpungr
           write(outu,*)'Number of GPUs used for PME:    ',igpungk
           write(outu,*)'Number of additional CPUs used: ',igpunch
        else
           write(outu,*)'Not Performing split GPU/CPU calculation'
        endif
     endif
  endif
#endif 
  lqsplit = index(comlyn,'SPLI') > 0
  if(lqsplit) then
     nsplit(1)=numnod-1  ! mainly used by GPU (KSPACE is separated)
     nsplit(2)=1
     call gtrmim(2,comlyn,comlen,'SPLI',nsplit,isplit,qsplit)
     if(isplit(1)+isplit(2) /= numnod) call wrndie(-5,'<PARCMD>', &
          'PARA SPLIT>Sum must be total number of processes')
     if(qsplit)then
        if(mynodg < isplit(1))then
           numnod=isplit(1)
           jsplit=0
        else
           numnod=isplit(2)
           mynod=mynodg-isplit(1)
           jsplit=1
        endif
        mynodp=numnod+1
        noddim=nptwo()
        do i=0,numnod
           ippmap(i)=i+jsplit*isplit(1)
        enddo
        do i=1,maxnode
           inode(i)=mod(i,numnod)
        enddo
        if(mynodg == 0) then
           write(outu,*)'PARAllel> Splitting this job in two two separate runs'
           write(outu,'(a,i0,a,i0,a)')' PARAllel> using ',isplit(1),' and ',isplit(2),' processes'
        endif
     endif
  endif
  !
#if KEY_REPLICA==1
  NTREP=GTRMI(COMLYN,COMLEN,'RPAT',0)
  QCYCL=(INDXA(COMLYN,COMLEN,'CYCL').GT.0)
  IF(NTREP.GT.0)THEN
     CALL set_param('LIREP',LIREP)
     CALL set_param('HIREP',HIREP)
     CALL set_param('NTREP',NTREP)
     IF(PRNLEV.GE.2)THEN
        WRITE(outu,*)'Replica path method will be used with',NTREP, &
             ' replications'
        WRITE(outu,*)'using',numnod,' procesors'
     ENDIF
  ENDIF
#endif 
  !
  !     This code is for Linux "real time" scheduling
  !     with the static priority ISTPRI
!!! Comented out, because not available everywhere
!!! check rtsched() in cstuff.c
!!!  ISTPRI=GTRMI(COMLYN,COMLEN,'FIFO',0)
!!!  CALL PSND4(ISTPRI,1)
!!!  IF(ISTPRI.GT.0)CALL RTSCHED(ISTPRI)

  !
  RETURN
END SUBROUTINE PARCMD

#endif /* (pll)*/
#endif /* (pllpcmd)*/

subroutine pll_dummy()
  return
end subroutine pll_dummy

