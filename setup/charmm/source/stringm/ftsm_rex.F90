! ********************************************************************************!
! Preprocessed by GNU CPP !
! ********************************************************************************!
!
! FTSM_REX.MOD
!
! REPLICA EXCHANGE MODULE FOR THE FINITE TEMPERATURE STRING METHOD
      module ftsm_rex
#if (KEY_STRINGM==1) /*  automatically protect all code */
!
#if (KEY_STRINGM==1)
!
      use chm_kinds
      use ivector
      use ftsm_var, only: nstring, ftsm_initialized
!
      implicit none
!
      private
!
! custom type for sending rex data by MPI
      type rex_string_datatype
       real(chm_real) :: dpar0, dperp0, drms0, kpara, kperp, krms, evolve_expo_mem
       real(chm_real) :: avforce(3) ! running average force arrays for FE integration
       integer :: num_evolve_samples
       logical :: ftsm_mini_on, evolve_expo_on, evolve_aver_on, qrms_upper_bound
      end type rex_string_datatype
!
      integer*4, save, public :: rex_string_data_mpi
!
      real(chm_real), save, public :: rex_beta
      integer, save, pointer, public :: rex_map(:)
      type (int_vector), save, public :: rex_log
!
      logical, save, public :: rex_initialized=.false.
!
      public rex_string_datatype
      public ftsm_rex_init
      public ftsm_rex_done
      public ftsm_rex_set_temp
      public ftsm_rex_print_map
      public ftsm_rex_read_map
      public ftsm_rex_print_log
!
      contains
!====================================================================================================
       subroutine ftsm_rex_init(temp)
      use consta
      use number
      use mpi
!
#if (KEY_PARALLEL==1)
#if (KEY_SINGLE==1)
 integer :: mpifloat=MPI_REAL 
#endif
#if (KEY_SINGLE==0)
 integer :: mpifloat=MPI_REAL8 
#endif
#if (KEY_INTEGER8==0)
 integer :: mpiint=MPI_INTEGER 
#endif
#if (KEY_INTEGER8==1)
 integer :: mpiint=MPI_INTEGER8 
#endif
 integer :: mpichar=MPI_CHARACTER
 integer :: mpibool=MPI_LOGICAL
#endif
!
       real(chm_real), optional :: temp
       real(chm_real) :: t
       integer :: i
       integer*4 :: counts(0:2), oldtypes(0:2), ierr ! , extent, offsets(0:2)
! integer :: offsets(0:1), counts(0:1), oldtypes(0:1), extent, ierr
       integer(kind=MPI_ADDRESS_KIND) :: lb, extent, offsets(0:2)
!
       if (.not.rex_initialized) then
        if (present(temp)) then ; t=temp; else ; t=three*100; endif
        if (t.gt.0) rex_beta=one/(t*kboltz)
        if (.not.ftsm_initialized) return
!
        allocate(rex_map(nstring))
        rex_map=(/ (i, i=0,nstring-1) /)
        call int_vector_reinit(rex_log)
        rex_initialized=.true.
! initialize MPI type for rex ; arrays must match type def above
        offsets(0)=0; oldtypes(0)=mpifloat; counts(0)=10
        call MPI_TYPE_GET_EXTENT(mpifloat,lb,extent,ierr)
!
        offsets(1)=offsets(0)+counts(0)*extent; oldtypes(1)=mpiint ; counts(1)=1
        call MPI_TYPE_GET_EXTENT(mpiint,lb,extent,ierr)
!
        offsets(2)=offsets(1)+counts(1)*extent; oldtypes(2)=mpibool ; counts(2)=4
!
        call MPI_TYPE_CREATE_STRUCT(3, counts, offsets, oldtypes, rex_string_data_mpi, ierr)
! call MPI_TYPE_STRUCT(3, counts, offsets, oldtypes, rex_string_data_mpi, ierr)
        call MPI_TYPE_COMMIT(rex_string_data_mpi, ierr)
!
       endif
!
       end subroutine ftsm_rex_init
!!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine ftsm_rex_done()
       use mpi
       integer*4 :: i
       if (rex_initialized) then
        deallocate(rex_map)
        call int_vector_done(rex_log)
        rex_initialized=.false.
        call mpi_type_free(rex_string_data_mpi,i)
       endif
       end subroutine ftsm_rex_done
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine ftsm_rex_set_temp(temp)
      use consta
      use number
       real(chm_real), optional :: temp
       real(chm_real) :: t
!
       if (.not.rex_initialized) call ftsm_rex_init()
       if (present(temp)) then ; t=temp; else ; t=three*100; endif
       if (t.gt.0) rex_beta=one/(t*kboltz)
       end subroutine ftsm_rex_set_temp
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine ftsm_rex_print_map(iunit,fmt)
      use stream
! only root process should call
       integer :: iunit
       character(len=*), optional :: fmt
! local
       integer :: i
       character(80) :: frm
       character(len=len("FTSM_REX_PRINT_MAP>") ),parameter::whoami="FTSM_REX_PRINT_MAP>";!macro
! begin
       if (.not.rex_initialized) then
! call wrndie(0,whoami,trim('REX NOT INITIALIZED.'))
! return
        call ftsm_rex_init()
       endif
!
       if (.not.present(fmt)) then
        write(frm,'("(",I5,"I5)")') nstring
       else
        frm=fmt
       endif
       write(iunit,frm) (/ (i, i=0,nstring-1) /)
       write(iunit,frm) rex_map(1:nstring)
       end subroutine ftsm_rex_print_map
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine ftsm_rex_read_map(iunit)
      use stream
      use multicom_aux;
      use mpi
      use parallel, only: psnd4, psnd8
!
#if (KEY_PARALLEL==1)
#if (KEY_SINGLE==1)
 integer :: mpifloat=MPI_REAL 
#endif
#if (KEY_SINGLE==0)
 integer :: mpifloat=MPI_REAL8 
#endif
#if (KEY_INTEGER8==0)
 integer :: mpiint=MPI_INTEGER 
#endif
#if (KEY_INTEGER8==1)
 integer :: mpiint=MPI_INTEGER8 
#endif
 integer :: mpichar=MPI_CHARACTER
 integer :: mpibool=MPI_LOGICAL
#endif
!
       integer :: iunit, ierror
       character(len=len("FTSM_REX_READ_MAP>") ),parameter::whoami="FTSM_REX_READ_MAP>";!macro
! begin
       if (.not.rex_initialized) then
        call ftsm_rex_init()
       endif
!
       if (MPI_COMM_STRNG.ne.MPI_COMM_NULL) then
        if (ME_STRNG.eq.0) then
         read(iunit,*) rex_map(1:nstring) ! first row contains indices 0 -- nstring-1
         read(iunit,*) rex_map(1:nstring) ! second row is what we want
         if (any(rex_map.lt.0)) call wrndie(0,whoami,trim('READ NEGATIVE RANK.'))
        endif ! ME_
        if (SIZE_STRNG.gt.1) call mpi_bcast(rex_map,nstring,mpiint,0,MPI_COMM_STRNG,ierror)
       endif ! MPI_COMM
! broadcast to slave nodes
       if (ME_LOCAL.ne.MPI_UNDEFINED.and.SIZE_LOCAL.gt.1) &
! & call MPI_BCAST(cv%rex_map, nstring, MPI_INTEGER,
! & 0,MPI_COMM_LOCAL,ierr)
#if (KEY_INTEGER8==0)
     & call PSND4(rex_map,nstring) 
#endif
#if (KEY_INTEGER8==1)
     & call PSND8(rex_map,nstring) 
#endif
!
       end subroutine ftsm_rex_read_map
!================================================================
       subroutine ftsm_rex_print_log(iunit, fmt)
! assume that unit is prepared
! NOTE that this is a global print!
      use multicom_aux;
      use mpi
!
       integer :: iunit
       character(len=*), optional :: fmt
! local
       character(80) :: frm
       integer :: i
       integer*4 :: rex_log_size4(nstring)
       integer :: rex_log_size8(nstring)
       integer*4 :: rex_log_disp4(nstring)
       integer :: total_size
       integer, pointer, dimension(:) :: rex_log_all
       integer :: ierror
       logical :: qroot
!
#if (KEY_PARALLEL==1)
#if (KEY_SINGLE==1)
 integer :: mpifloat=MPI_REAL 
#endif
#if (KEY_SINGLE==0)
 integer :: mpifloat=MPI_REAL8 
#endif
#if (KEY_INTEGER8==0)
 integer :: mpiint=MPI_INTEGER 
#endif
#if (KEY_INTEGER8==1)
 integer :: mpiint=MPI_INTEGER8 
#endif
 integer :: mpichar=MPI_CHARACTER
 integer :: mpibool=MPI_LOGICAL
#endif
! do work
! gather all data on root
       qroot=MPI_COMM_STRNG.ne.MPI_COMM_NULL
!
       if (.not.rex_initialized) call ftsm_rex_init()
!
       if (qroot.and.SIZE_STRNG.gt.1) then
! calculate size of logs
        rex_log_size8=0
! call MPI_ALLGATHER(rex_log%last,1,type,
! & rex_log_size8,1,type,
! & MPI_COMM_STRNG,error)
        call MPI_GATHER(rex_log%last,1,mpiint, &
     & rex_log_size8,1,mpiint, &
     & 0,MPI_COMM_STRNG,ierror)
        call mpi_bcast(rex_log_size8,nstring,mpiint,0,MPI_COMM_STRNG,ierror)
!
        total_size=sum(rex_log_size8)
        rex_log_size4=rex_log_size8 ! type cast to 4 byte integer
! allocate space to hold entire log
        allocate(rex_log_all(total_size))
! calculate send displacements
        rex_log_disp4(1)=0;
        do i=1,SIZE_STRNG-1
         rex_log_disp4(i+1)=rex_log_disp4(i)+rex_log_size4(i)
        enddo
! now gather the logs
! call MPI_ALLGATHERV(rex_log%i,rex_log%last,type,
! & rex_log_all,rex_log_size4,rex_log_disp4,type,
! & MPI_COMM_STRNG,ierror)
        call MPI_GATHERV(rex_log%i,rex_log%last,mpiint, &
     & rex_log_all,rex_log_size4,rex_log_disp4,mpiint, &
     & 0,MPI_COMM_STRNG,ierror)
!
        if (.not.present(fmt)) then
         frm='(2I5,I13)'
        else
         frm=fmt
        endif
!
      if(ME_STRNG.eq.0.and.total_size.gt.0) write(iunit,frm) rex_log_all
!
        call int_vector_reinit(rex_log) ! erase log
        deallocate(rex_log_all)
       endif ! STRNG
       end subroutine ftsm_rex_print_log
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
#endif
#endif /* automatically protect all code */
      end module ftsm_rex
!
