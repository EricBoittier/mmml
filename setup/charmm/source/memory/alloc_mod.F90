MODULE allocation
  !
  ! This module handles explicit memory allocation through calls to
  !  chmalloc(), which take the form:
  ! 
  !  call chmalloc('filename','procname','arrayname',size,(siz2,)
  !     & (siz3,) 
  !     & [array type keyword]= arrayname,lbou=lbound,(lbou2=lbound2,)
  !     & (lbou3=lbound3,) ierr=errorname,
  !     & qdie=[.true. or .false.])
  !
  !     'filename' is the name of the file in which the array to be
  !        allocated appears
  !     'procname' is the name of the procedure in which the array appears 
  !     'arrayname' is the name of the array
  !     errorname is the integer which holds the error value returned in ierr
  !     qdie is the logical dummy argument whose specified value determines
  !        whether execution will halt on an error (unless overridden)
  !     size, siz2, siz3 are the (integer) extents of the array in 
  !        each dimension, as appropriate. (Only arrays of rank 1, 2, or 3
  !        are currently supported by this facility.)
  !     lbou,lbou2,lbou3 are the dummy arguments whose specified integer
  !        values (lbound,lbound2,lbound3) give the lower bounds of the array.
  !
  !     The list of supported array types is as follows:
  !
  !      keyword    type
  !      crl       chm_real
  !      crlp      pointer to chm_real
  !      mcrlp     pointer to chm_real allocated using MPI_ALLOC_MEM
  !      cr4       chm_real4
  !      cr4p      pointer to chm_real4
  !      mcr4p     pointer to chm_real4 allocated using MPI_ALLOC_MEM
  !      cr8       chm_real8
  !      rlg       general real
  !      intg      general integer
  !      intgp     pointer to general integer
  !      iby       int_byte
  !      mibyp     pointer to int_byte allocated using MPI_ALLOC_MEM
  !      ci2       chm_int2
  !      ci4       chm_int4
  !      ci8       chm_int8
  !      ch1       character len=1
  !      ch2       character len=2
  !      ch4       character len=4
  !      ch6       character len=6
  !      ch8       character len=8
  !      ch16      character len=16
  !      ch36      character len=36 (added cb3 for mmff)
  !      ch80      character len=80 (added mfc for parallel host names)
  !      log       logical
  !      cmpx      chm_cmpx complex(double precision)
  !
  !      The filename, procname, arrayname, and size arguments are required.
  !      The dummy argument keyword [array type keyword = ] should be specified
  !        in all cases.  The "ierr =" , "qdie =" , and lower bound dummy
  !        argument keywords need to be specified when the corresponding
  !        arguments are being used.  The array type keyword should immediately
  !        follow the size argument(s) in the calling sequence. The order of 
  !        any additional optional arguments is less important. 
  !
  !      An example of an allocation of a 1D array with this facility is:
  ! 
  !      call chmalloc('testlong','mainproc','intarr',1000,
  !     & intg=intarr)
  !     
  !      For a 2D array allocation, in which the lower bound is -100 and 
  !      the error status is returned in the variable myerr, without halting
  !      execution, 
  !
  !      call chmalloc('testlong','mainproc','intarr',100,200,
  !     & intg=intarr2d,lbou=-100,ierr=myerr,qdie=.false.)
  !
  !      There are special subroutines, alloc_vchar and dealloc_vchar, 
  !      that handle (1D) arrays of characters having any specified length.
  !
  !     Thanks to L. Nilsson, P. Sherwood, R. Venable, B.R. Brooks, 
  !     C. Brooks, T. Miller, M. Crowley, Y. Won, and B. Saybasili
  !
  !                                         --RJP  Nov 2008
  use chm_kinds
  use chm_types
  implicit none

  INTERFACE chmalloc
     module procedure alloc_1d
     module procedure alloc_2d
     module procedure alloc_3d
     module procedure alloc_4d
  end INTERFACE

CONTAINS
  !______________________________________________________________________
  !______________________________________________________________________
  !
  SUBROUTINE alloc_1d(filename,procname,arrayname,size, &
       crl,crlp,mcrlp,cr4,cr4p,mcr4p,cr8, &
       rlg,intg,intgp,iby,mibyp,ci2,ci4,ci8, &
       ch1,ch2,ch4,ch6,ch8,ch16,ch20,ch36,ch80, &
       log,cmpx,qdie,ierr,lbou)
    !
    !  This routine handles explicit 1D array allocations
    !                                                  --RJP Nov 2008
#if KEY_PARALLEL==1
    use mpi
    use iso_c_binding
    use parallel,only:mpi_real8_size, mpi_real4_size
#endif 
    implicit none
    character(len=*),intent(in) :: filename,procname,arrayname
    integer,intent(in) :: size
    integer,optional,intent(in) :: lbou
    logical,optional,intent(in) :: qdie
    integer,optional,intent(out) :: ierr
    ! passed arrays, all optional arguments
    real(kind=chm_real),allocatable,optional,dimension(:) :: crl 
    real(kind=chm_real),pointer,optional,dimension(:) :: crlp 
    real(kind=chm_real),pointer,optional,dimension(:) :: mcrlp 
    real(kind=chm_real4),allocatable,optional,dimension(:) :: cr4 
    real(kind=chm_real4),pointer,optional,dimension(:) :: cr4p 
    real(kind=chm_real4),pointer,optional,dimension(:) :: mcr4p 
    real(kind=chm_real8),allocatable,optional,dimension(:) :: cr8 
    real,allocatable,optional,dimension(:) :: rlg
    integer,allocatable,optional,dimension(:) :: intg
    integer,pointer,optional,dimension(:) :: intgp
    integer(kind=int_byte),allocatable,optional,dimension(:) :: iby
    integer(kind=int_byte),pointer,optional,dimension(:) :: mibyp
    integer(kind=chm_int2),allocatable,optional,dimension(:) :: ci2
    integer(kind=chm_int4),allocatable,optional,dimension(:) :: ci4
    integer(kind=chm_int8),allocatable,optional,dimension(:) :: ci8
    character(len=1),allocatable,optional,dimension(:) :: ch1 
    character(len=2),allocatable,optional,dimension(:) :: ch2
    character(len=4),allocatable,optional,dimension(:) :: ch4
    character(len=6),allocatable,optional,dimension(:) :: ch6
    character(len=8),allocatable,optional,dimension(:) :: ch8
    character(len=16),allocatable,optional,dimension(:) :: ch16
    character(len=20),allocatable,optional,dimension(:) :: ch20
    character(len=36),allocatable,optional,dimension(:) :: ch36
    character(len=80),allocatable,optional,dimension(:) :: ch80
    logical,allocatable,optional,dimension(:) :: log
    complex(kind=chm_cmpx),allocatable,optional,dimension(:) :: cmpx 
    ! local variables
    logical :: locqdie
    integer :: hit, locierr,loclb,locub
#if KEY_PARALLEL==1
    type(c_ptr) :: cptr
    integer(kind=MPI_ADDRESS_KIND) mpisize, mpiptr
#endif 
    !***********************************************************************
    ! end of variable declarations
    !***********************************************************************
    ! determine bounds
    loclb = 1
    if(present(lbou)) then
       loclb = lbou
    endif
    locub = loclb + size -1  
    ! determine whether to die on errors
    locqdie = .true.
    if(present(qdie)) then
       locqdie = qdie
    endif
    !----------------------------------------------------------------------
    ! check to see which array type is present in arguments
    ! most common types first
    ! chm_real
    if(present(crl)) then
       allocate(crl(loclb:locub),stat=locierr)
       call proc_alloc_outpt(filename,procname,arrayname,size, &
            'crl',1,locierr,locqdie,kind(crl),isallocated=allocated(crl))
    elseif(present(crlp)) then
       allocate(crlp(loclb:locub),stat=locierr)
       call proc_alloc_outpt(filename,procname,arrayname,size, &
            'crlp',1,locierr,locqdie,kind(crlp),isallocated=associated(crlp))
       ! general integer
    else if (present(intg)) then
       allocate(intg(loclb:locub),stat=locierr)
       call proc_alloc_outpt(filename,procname,arrayname,size, &
            'intg',1,locierr,locqdie,kind(intg),isallocated=allocated(intg))
    else if (present(intgp)) then
       allocate(intgp(loclb:locub),stat=locierr)
       call proc_alloc_outpt(filename,procname,arrayname,size, &
            'intgp',1,locierr,locqdie,kind(intgp),isallocated=associated(intgp))
       ! general real 
    else if(present(rlg)) then
       allocate(rlg(loclb:locub),stat=locierr)
       call proc_alloc_outpt(filename,procname,arrayname,size, &
            'rlg',1,locierr,locqdie,kind(rlg),isallocated=allocated(rlg))
       ! character(len=6)
    else if(present(ch6)) then
       allocate(ch6(loclb:locub),stat=locierr)
#ifdef KEY_RESIZE
       ch6=''
#endif              
       call proc_alloc_outpt(filename,procname,arrayname,size, &
            'ch6',1,locierr,locqdie,kind(ch6),isallocated=allocated(ch6))
       ! character(len=8)
    else if(present(ch8)) then
       allocate(ch8(loclb:locub),stat=locierr)
#ifdef KEY_RESIZE
       ch8=''
#endif              
       call proc_alloc_outpt(filename,procname,arrayname,size, &
            'ch8',1,locierr,locqdie,kind(ch8),isallocated=allocated(ch8))
       ! chm_real4
    else if(present(cr4)) then
       allocate(cr4(loclb:locub),stat=locierr)
       call proc_alloc_outpt(filename,procname,arrayname,size, &
            'cr4',1,locierr,locqdie,kind(cr4),isallocated=allocated(cr4))
    else if(present(cr4p)) then
       allocate(cr4p(loclb:locub),stat=locierr)
       call proc_alloc_outpt(filename,procname,arrayname,size, &
            'cr4p',1,locierr,locqdie,kind(cr4p),isallocated=associated(cr4p))
       ! chm_int8
    else if(present(ci8)) then
       allocate(ci8(loclb:locub),stat=locierr)
       call proc_alloc_outpt(filename,procname,arrayname,size, &
            'ci8',1,locierr,locqdie,kind(ci8),isallocated=allocated(ci8))
       ! logical
    else if (present(log)) then
       allocate(log(loclb:locub),stat=locierr)
       call proc_alloc_outpt(filename,procname,arrayname,size, &
            'log',1,locierr,locqdie,kind(log),isallocated=allocated(log))
       ! chm_cmpx
    else if(present(cmpx)) then
       allocate(cmpx(loclb:locub),stat=locierr)
       call proc_alloc_outpt(filename,procname,arrayname,size, &
            'cmpx',1,locierr,locqdie,kind(cmpx),isallocated=allocated(cmpx))
       ! less common types
       ! chm_real8
    else if(present(cr8)) then
       allocate(cr8(loclb:locub),stat=locierr)
       call proc_alloc_outpt(filename,procname,arrayname,size, &
            'cr8',1,locierr,locqdie,kind(cr8),isallocated=allocated(cr8))
       ! chm_int4
    else if(present(ci4)) then
       allocate(ci4(loclb:locub),stat=locierr)
       call proc_alloc_outpt(filename,procname,arrayname,size, &
            'ci4',1,locierr,locqdie,kind(ci4),isallocated=allocated(ci4))
       ! int_byte
    else if(present(iby)) then
       allocate(iby(loclb:locub),stat=locierr)
       call proc_alloc_outpt(filename,procname,arrayname,size, &
            'iby',1,locierr,locqdie,kind(iby),isallocated=allocated(iby))
       ! int_byte with using MPI allocation
    else if(present(mibyp)) then
#if KEY_PARALLEL==1
       if (loclb /= 1) then
          allocate(mibyp(loclb:locub),stat=locierr)
          call wrndie(0,'<ALLOC_1D>','INVALID LOWER BOUND, ALLOCATING NON MPI MEMORY')
       else
          mpisize = locub-loclb+1
          call mpi_alloc_mem(mpisize, MPI_INFO_NULL, mpiptr, locierr)
          if (locierr /= MPI_SUCCESS) then
             locierr = 1
          else
             locierr = 0
             cptr = transfer(mpiptr, cptr)
             call c_f_pointer(cptr, mibyp, [locub-loclb+1])
          endif
       endif
#else /**/
       allocate(mibyp(loclb:locub),stat=locierr)
#endif 
       call proc_alloc_outpt(filename,procname,arrayname,size, &
            'mibyp',1,locierr,locqdie,kind(mibyp),isallocated=associated(mibyp))
       ! chm_int2
    else if(present(ci2)) then
       allocate(ci2(loclb:locub),stat=locierr)
       call proc_alloc_outpt(filename,procname,arrayname,size, &
            'ci2',1,locierr,locqdie,kind(ci2),isallocated=allocated(ci2))
       ! character(len=1)
    else if(present(ch1)) then
       allocate(ch1(loclb:locub),stat=locierr)
#ifdef KEY_RESIZE
       ch1=''
#endif              
       call proc_alloc_outpt(filename,procname,arrayname,size, &
            'ch1',1,locierr,locqdie,kind(ch1),isallocated=allocated(ch1))
       ! character(len=2)
    else if(present(ch2)) then
       allocate(ch2(loclb:locub),stat=locierr)
#ifdef KEY_RESIZE
       ch2=''
#endif              
       call proc_alloc_outpt(filename,procname,arrayname,size, &
            'ch2',1,locierr,locqdie,kind(ch2),isallocated=allocated(ch2))
       ! character(len=4)
    else if(present(ch4)) then
       allocate(ch4(loclb:locub),stat=locierr)
#ifdef KEY_RESIZE
       ch4=''
#endif              
       call proc_alloc_outpt(filename,procname,arrayname,size, &
            'ch4',1,locierr,locqdie,kind(ch4),isallocated=allocated(ch4))
       ! character(len=16)
    else if(present(ch16)) then
       allocate(ch16(loclb:locub),stat=locierr)
#ifdef KEY_RESIZE
       ch16=''
#endif              
       call proc_alloc_outpt(filename,procname,arrayname,size, &
            'ch16',1,locierr,locqdie,kind(ch16),isallocated=allocated(ch16))
       ! character(len=16)
    else if(present(ch20)) then
       allocate(ch20(loclb:locub),stat=locierr)
#ifdef KEY_RESIZE
       ch20=''
#endif              
       call proc_alloc_outpt(filename,procname,arrayname,size, &
            'ch20',1,locierr,locqdie,kind(ch20),isallocated=allocated(ch20))
    else if(present(ch36)) then
       allocate(ch36(loclb:locub),stat=locierr)
#ifdef KEY_RESIZE
       ch36=''
#endif              
       call proc_alloc_outpt(filename,procname,arrayname,size, &
            'ch36',1,locierr,locqdie,kind(ch36),isallocated=allocated(ch36))
    else if(present(ch80)) then
       allocate(ch80(loclb:locub),stat=locierr)
#ifdef KEY_RESIZE
       ch80=''
#endif              
       call proc_alloc_outpt(filename,procname,arrayname,size, &
            'ch80',1,locierr,locqdie,kind(ch80),isallocated=allocated(ch80))
    else if(present(mcrlp)) then
#if KEY_PARALLEL==1
       if (loclb /= 1) then
          call wrndie(-5,'<ALLOC_1D>','MPI memory (mcrlp) must have lower bound=1')
       else
          mpisize = mpi_real8_size*(locub-loclb+1)
          call mpi_alloc_mem(mpisize, MPI_INFO_NULL, mpiptr, locierr)
          if (locierr /= MPI_SUCCESS) then
             locierr = 1
          else
             locierr = 0
             cptr = transfer(mpiptr, cptr)
             call c_f_pointer(cptr, mcrlp, [locub-loclb+1])
          endif
       endif
#else /**/
       allocate(mcrlp(loclb:locub),stat=locierr)
#endif 
       call proc_alloc_outpt(filename,procname,arrayname,size, &
            'mcrlp',1,locierr,locqdie,kind(mcrlp),isallocated=associated(mcrlp))
    else if(present(mcr4p)) then
#if KEY_PARALLEL==1
       if (loclb /= 1) then
          call wrndie(-5,'<ALLOC_1D>','MPI memory (mcr4p) must have lower bound=1')
       else
          mpisize = mpi_real4_size*(locub-loclb+1)
          call mpi_alloc_mem(mpisize, MPI_INFO_NULL, mpiptr, locierr)
          if (locierr /= MPI_SUCCESS) then
             locierr = 1
          else
             locierr = 0
             cptr = transfer(mpiptr, cptr)
             call c_f_pointer(cptr, mcr4p, [locub-loclb+1])
          endif
       endif
#else /**/
       allocate(mcr4p(loclb:locub),stat=locierr)
#endif 
       call proc_alloc_outpt(filename,procname,arrayname,size, &
            'mcr4p',1,locierr,locqdie,kind(mcr4p),isallocated=associated(mcr4p))
    else 
       ! write warning if no array match
       write(6,*) 'all1d No array matched list of available types.',filename,procname,arrayname
       !        if(qdie) then call wrndie...
    endif
    ! save error status
    if(present(ierr)) then
       ierr = locierr 
    endif
  END SUBROUTINE alloc_1d
  !______________________________________________________________________
  !______________________________________________________________________
  !
  SUBROUTINE alloc_2d(filename,procname,arrayname,size,siz2,crl,crlp,mcrlp,rlg, &
       cr4,cr4p,cr8,intg,intgp,ci2,ci4,ci8,iby,ch1,ch2,ch4,ch6,ch8,ch16,log,qdie,ierr, &
       lbou,lbou2)
    !
    !  This routine handles explicit 2D array allocations
    !                                                     -- RJP Nov 2008 
#if KEY_PARALLEL==1
    use mpi
    use iso_c_binding
    use parallel,only:mpi_real8_size
#endif 
    implicit none
    character(len=*),intent(in) :: filename,procname,arrayname
    integer,intent(in) :: size,siz2
    integer,optional,intent(in) :: lbou,lbou2
    logical,optional,intent(in) :: qdie
    integer,optional,intent(out) :: ierr
    ! passed arrays, all optional arguments
    real(kind=chm_real),allocatable,optional,dimension(:,:) :: crl
    real(kind=chm_real),pointer,optional,dimension(:,:) :: crlp
    real(kind=chm_real),pointer,optional,dimension(:,:) :: mcrlp
    real(kind=chm_real4),allocatable,optional,dimension(:,:) :: cr4
    real(kind=chm_real4),pointer,optional,dimension(:,:) :: cr4p
    real(kind=chm_real8),allocatable,optional,dimension(:,:) :: cr8
    real,allocatable,optional,dimension(:,:) :: rlg
    integer,allocatable,optional,dimension(:,:) :: intg
    integer,pointer,optional,dimension(:,:) :: intgp
    integer(kind=int_byte),allocatable,optional,dimension(:,:) :: iby
    integer(kind=chm_int2),allocatable,optional,dimension(:,:) :: ci2
    integer(kind=chm_int4),allocatable,optional,dimension(:,:) :: ci4
    integer(kind=chm_int8),allocatable,optional,dimension(:,:) :: ci8
    character(len=1),allocatable,optional,dimension(:,:) :: ch1
    character(len=2),allocatable,optional,dimension(:,:) :: ch2
    character(len=4),allocatable,optional,dimension(:,:) :: ch4
    character(len=6),allocatable,optional,dimension(:,:) :: ch6
    character(len=8),allocatable,optional,dimension(:,:) :: ch8
    character(len=16),allocatable,optional,dimension(:,:) :: ch16
    logical,allocatable,optional,dimension(:,:) :: log
    !
    ! local variables
    logical :: locqdie
    integer :: hit, locierr,loclb1,locub1,loclb2,locub2
#if KEY_PARALLEL==1
    type(c_ptr) :: cptr
    integer(kind=MPI_ADDRESS_KIND) mpisize, mpiptr
#endif 
    !***********************************************************************
    ! end of variable declarations
    !***********************************************************************
    ! determine bounds
    loclb1 = 1
    if(present(lbou)) then
       loclb1 = lbou
    endif
    locub1 = loclb1 + size -1  
    !
    loclb2 = 1
    if(present(lbou2)) then
       loclb2 = lbou2
    endif
    locub2 = loclb2 + siz2 -1
    !
    ! determine whether to die on errors
    locqdie = .true.
    if(present(qdie)) then
       locqdie = qdie
    endif
    !----------------------------------------------------------------------
    ! check to see which array type is present in arguments
    ! chm_real
    if(present(crl)) then
       allocate(crl(loclb1:locub1,loclb2:locub2),stat=locierr)
       call proc_alloc_outpt(filename,procname,arrayname,size, &
            'crl',2,locierr,locqdie,kind(crl),sz2p=siz2,isallocated=allocated(crl))
    elseif(present(crlp)) then
       allocate(crlp(loclb1:locub1,loclb2:locub2),stat=locierr)
       call proc_alloc_outpt(filename,procname,arrayname,size, &
            'crlp',2,locierr,locqdie,kind(crlp),sz2p=siz2,isallocated=associated(crlp))
       ! general integer
    else if (present(intg)) then
       allocate(intg(loclb1:locub1,loclb2:locub2),stat=locierr)
       call proc_alloc_outpt(filename,procname,arrayname,size, &
            'intg',2,locierr,locqdie,kind(intg),sz2p=siz2,isallocated=allocated(intg))
    else if (present(intgp)) then
       allocate(intgp(loclb1:locub1,loclb2:locub2),stat=locierr)
       call proc_alloc_outpt(filename,procname,arrayname,size, &
            'intgp',2,locierr,locqdie,kind(intgp),sz2p=siz2,isallocated=associated(intgp))
       ! general real
    else if(present(rlg)) then
       allocate(rlg(loclb1:locub1,loclb2:locub2),stat=locierr)
       call proc_alloc_outpt(filename,procname,arrayname,size, &
            'rlg',2,locierr,locqdie,kind(rlg),sz2p=siz2,isallocated=allocated(rlg))
       ! character(len=6)
    else if(present(ch6)) then
       allocate(ch6(loclb1:locub1,loclb2:locub2),stat=locierr)
#ifdef KEY_RESIZE
       ch6=''
#endif              
       call proc_alloc_outpt(filename,procname,arrayname,size, &
            'ch6',2,locierr,locqdie,kind(ch6),sz2p=siz2,isallocated=allocated(ch6))
       ! character(len=8)
    else if(present(ch8)) then
       allocate(ch8(loclb1:locub1,loclb2:locub2),stat=locierr)
#ifdef KEY_RESIZE
       ch8=''
#endif              
       call proc_alloc_outpt(filename,procname,arrayname,size, &
            'ch8',2,locierr,locqdie,kind(ch8),sz2p=siz2,isallocated=allocated(ch8))
       ! chm_real4
    else if(present(cr4)) then
       allocate(cr4(loclb1:locub1,loclb2:locub2),stat=locierr)
       call proc_alloc_outpt(filename,procname,arrayname,size, &
            'cr4',2,locierr,locqdie,kind(cr4),sz2p=siz2,isallocated=allocated(cr4))
    else if(present(cr4p)) then
       allocate(cr4(loclb1:locub1,loclb2:locub2),stat=locierr)
       call proc_alloc_outpt(filename,procname,arrayname,size, &
            'cr4p',2,locierr,locqdie,kind(cr4),sz2p=siz2,isallocated=allocated(cr4))
       ! chm_int8
    else if(present(ci8)) then
       allocate(ci8(loclb1:locub1,loclb2:locub2),stat=locierr)
       call proc_alloc_outpt(filename,procname,arrayname,size, &
            'ci8',2,locierr,locqdie,kind(ci8),sz2p=siz2,isallocated=allocated(ci8))
       ! logical
    else if (present(log)) then
       allocate(log(loclb1:locub1,loclb2:locub2),stat=locierr)
       call proc_alloc_outpt(filename,procname,arrayname,size, &
            'log',2,locierr,locqdie,kind(log),sz2p=siz2,isallocated=allocated(log))
       ! less common types
       ! chm_real8
    else if(present(cr8)) then
       allocate(cr8(loclb1:locub1,loclb2:locub2),stat=locierr)
       call proc_alloc_outpt(filename,procname,arrayname,size, &
            'cr8',2,locierr,locqdie,kind(cr8),sz2p=siz2,isallocated=allocated(cr8))
       ! chm_int4
    else if(present(ci4)) then
       allocate(ci4(loclb1:locub1,loclb2:locub2),stat=locierr)
       call proc_alloc_outpt(filename,procname,arrayname,size, &
            'ci4',2,locierr,locqdie,kind(ci4),sz2p=siz2,isallocated=allocated(ci4))
       ! int_byte
    else if(present(iby)) then
       allocate(iby(loclb1:locub1,loclb2:locub2),stat=locierr)
       call proc_alloc_outpt(filename,procname,arrayname,size, &
            'iby',2,locierr,locqdie,kind(iby),sz2p=siz2,isallocated=allocated(iby))
       ! chm_int2
    else if(present(ci2)) then
       allocate(ci2(loclb1:locub1,loclb2:locub2),stat=locierr)
       call proc_alloc_outpt(filename,procname,arrayname,size, &
            'ci2',2,locierr,locqdie,kind(ci2),sz2p=siz2,isallocated=allocated(ci2))
       ! character(len=1)
    else if(present(ch1)) then
       allocate(ch1(loclb1:locub1,loclb2:locub2),stat=locierr)
#ifdef KEY_RESIZE
       ch1=''
#endif              
       call proc_alloc_outpt(filename,procname,arrayname,size, &
            'ch1',2,locierr,locqdie,kind(ch1),sz2p=siz2,isallocated=allocated(ch1))
       ! character(len=2)
    else if(present(ch2)) then
       allocate(ch2(loclb1:locub1,loclb2:locub2),stat=locierr)
#ifdef KEY_RESIZE
       ch2=''
#endif              
       call proc_alloc_outpt(filename,procname,arrayname,size, &
            'ch2',2,locierr,locqdie,kind(ch2),sz2p=siz2,isallocated=allocated(ch2))
       ! character(len=4)
    else if(present(ch4)) then
       allocate(ch4(loclb1:locub1,loclb2:locub2),stat=locierr)
#ifdef KEY_RESIZE
       ch4=''
#endif              
       call proc_alloc_outpt(filename,procname,arrayname,size, &
            'ch4',2,locierr,locqdie,kind(ch4),sz2p=siz2,isallocated=allocated(ch4))
       ! character(len=16)
    else if(present(ch16)) then
       allocate(ch16(loclb1:locub1,loclb2:locub2),stat=locierr)
#ifdef KEY_RESIZE
       ch16=''
#endif              
       call proc_alloc_outpt(filename,procname,arrayname,size, &
            'cr16',2,locierr,locqdie,kind(ch16),sz2p=siz2,isallocated=allocated(ch16))
    else if(present(mcrlp)) then
#if KEY_PARALLEL==1
       if (loclb1 /= 1 .or. loclb2 /= 1) then
          call wrndie(-5,'<ALLOC_1D>','MPI memory (mcrlp) must have lower bound=1')
       else
          mpisize = mpi_real8_size*(locub1-loclb1+1)*(locub2-loclb2+1)
          call mpi_alloc_mem(mpisize, MPI_INFO_NULL, mpiptr, locierr)
          if (locierr /= MPI_SUCCESS) then
             locierr = 1
          else
             locierr = 0
             cptr = transfer(mpiptr, cptr)
             call c_f_pointer(cptr, mcrlp, [locub1-loclb1+1, locub2-loclb2+1])
          endif
       endif
#else /**/
       allocate(mcrlp(loclb1:locub1,loclb2:locub2),stat=locierr)
#endif 
       call proc_alloc_outpt(filename,procname,arrayname,size, &
            'mcrlp',2,locierr,locqdie,kind(mcrlp),sz2p=siz2,isallocated=associated(mcrlp))
    else
       ! write warning if no array match
       write(6,*) 'all2d No array matched list of available types.',filename,procname,arrayname
       !        if(qdie) then call wrndie...
    endif
    ! save error status
    if(present(ierr)) then
       ierr = locierr 
    endif
  END SUBROUTINE alloc_2d
  !______________________________________________________________________
  !______________________________________________________________________
  !
  SUBROUTINE alloc_3d(filename,procname,arrayname,size,siz2,siz3, &
       crl,cr4,cr8,rlg,intg,ci2,ci4,ci8,iby,ch1,ch2,ch4,ch6,ch8,ch16,log, &
       cmpx,qdie,ierr,lbou,lbou2,lbou3)
    !
    !  This routine handles explicit 3D array allocations
    !                                                --RJP Nov 2008
    implicit none
    character(len=*),intent(in) :: filename,procname,arrayname
    integer,intent(in) :: size,siz2,siz3
    integer,optional,intent(in) :: lbou,lbou2,lbou3
    logical,optional,intent(in) :: qdie
    integer,optional,intent(out) :: ierr
    ! passed arrays, all optional arguments
    real(kind=chm_real),allocatable,optional,dimension(:,:,:) :: crl
    real(kind=chm_real4),allocatable,optional,dimension(:,:,:) :: cr4
    real(kind=chm_real8),allocatable,optional,dimension(:,:,:) :: cr8
    real,allocatable,optional,dimension(:,:,:) :: rlg
    integer,allocatable,optional,dimension(:,:,:) :: intg
    integer(kind=int_byte),allocatable,optional, &
         dimension(:,:,:) :: iby
    integer(kind=chm_int2),allocatable,optional, &
         dimension(:,:,:) :: ci2
    integer(kind=chm_int4),allocatable,optional, &
         dimension(:,:,:) :: ci4
    integer(kind=chm_int8),allocatable,optional, &
         dimension(:,:,:) :: ci8
    character(len=1),allocatable,optional,dimension(:,:,:) :: ch1
    character(len=2),allocatable,optional,dimension(:,:,:) :: ch2
    character(len=4),allocatable,optional,dimension(:,:,:) :: ch4
    character(len=6),allocatable,optional,dimension(:,:,:) :: ch6
    character(len=8),allocatable,optional,dimension(:,:,:) :: ch8
    character(len=16),allocatable,optional,dimension(:,:,:) :: ch16
    logical,allocatable,optional,dimension(:,:,:) :: log
    complex(kind=chm_cmpx),allocatable,optional,dimension(:,:,:) :: cmpx 
    !
    ! local variables
    logical :: locqdie
    integer :: hit, locierr,loclb1,locub1,loclb2,locub2,loclb3,locub3
    !***********************************************************************
    ! end of variable declarations
    !***********************************************************************
    ! determine bounds
    loclb1 = 1
    if(present(lbou)) then
       loclb1 = lbou
    endif
    locub1 = loclb1 + size -1  
    !
    loclb2 = 1
    if(present(lbou2)) then
       loclb2 = lbou2
    endif
    locub2 = loclb2 + siz2 -1
    !
    loclb3 = 1
    if(present(lbou3)) then
       loclb3 = lbou3
    endif
    locub3 = loclb3 + siz3 -1
    !
    ! determine whether to die on errors
    locqdie = .true.
    if(present(qdie)) then
       locqdie = qdie
    endif
    !----------------------------------------------------------------------
    ! check to see which array type is present in arguments
    ! chm_real
    if(present(crl)) then
       allocate(crl(loclb1:locub1,loclb2:locub2,loclb3:locub3), &
            stat=locierr)
       call proc_alloc_outpt(filename,procname,arrayname,size, &
            'crl',3,locierr,locqdie,kind(crl),sz2p=siz2,sz3p=siz3,isallocated=allocated(crl))
       ! general integer
    else if(present(intg)) then
       allocate(intg(loclb1:locub1,loclb2:locub2,loclb3:locub3), &
            stat=locierr)
       call proc_alloc_outpt(filename,procname,arrayname,size, &
            'intg',3,locierr,locqdie,kind(intg),sz2p=siz2,sz3p=siz3,isallocated=allocated(intg))
       ! general real
    else if(present(rlg)) then
       allocate(rlg(loclb1:locub1,loclb2:locub2,loclb3:locub3), &
            stat=locierr)
       call proc_alloc_outpt(filename,procname,arrayname,size, &
            'rlg',3,locierr,locqdie,kind(rlg),sz2p=siz2,sz3p=siz3,isallocated=allocated(rlg))
       ! character(len=6)
    else if(present(ch6)) then
       allocate(ch6(loclb1:locub1,loclb2:locub2,loclb3:locub3), &
            stat=locierr)
#ifdef KEY_RESIZE
       ch6=''
#endif              
       call proc_alloc_outpt(filename,procname,arrayname,size, &
            'ch6',3,locierr,locqdie,kind(ch6),sz2p=siz2,sz3p=siz3,isallocated=allocated(ch6))
       ! character(len=8)
    else if(present(ch8)) then
       allocate(ch8(loclb1:locub1,loclb2:locub2,loclb3:locub3), &
            stat=locierr)
#ifdef KEY_RESIZE
       ch8=''
#endif              
       call proc_alloc_outpt(filename,procname,arrayname,size, &
            'ch8',3,locierr,locqdie,kind(ch8),sz2p=siz2,sz3p=siz3,isallocated=allocated(ch8))
       !! chm_real4
    else if(present(cr4)) then
       allocate(cr4(loclb1:locub1,loclb2:locub2,loclb3:locub3), &
            stat=locierr)
       call proc_alloc_outpt(filename,procname,arrayname,size, &
            'cr4',3,locierr,locqdie,kind(cr4),sz2p=siz2,sz3p=siz3,isallocated=allocated(cr4))
       ! chm_int8
    else if(present(ci8)) then
       allocate(ci8(loclb1:locub1,loclb2:locub2,loclb3:locub3), &
            stat=locierr)
       call proc_alloc_outpt(filename,procname,arrayname,size, &
            'ci8',3,locierr,locqdie,kind(ci8),sz2p=siz2,sz3p=siz3,isallocated=allocated(ci8))
       ! logical
    else if(present(log)) then
       allocate(log(loclb1:locub1,loclb2:locub2,loclb3:locub3), &
            stat=locierr)
       call proc_alloc_outpt(filename,procname,arrayname,size, &
            'log',3,locierr,locqdie,kind(log),sz2p=siz2,sz3p=siz3,isallocated=allocated(log))
       ! chm_cmpx
    else if(present(cmpx)) then
       allocate(cmpx(loclb1:locub1,loclb2:locub2,loclb3:locub3), &
            stat=locierr)
       call proc_alloc_outpt(filename,procname,arrayname,size, &
            'cmpx',3,locierr,locqdie,kind(cmpx),sz2p=siz2,sz3p=siz3,isallocated=allocated(cmpx))
       !less common types
       ! chm_real8
    else if(present(cr8)) then
       allocate(cr8(loclb1:locub1,loclb2:locub2,loclb3:locub3), &
            stat=locierr)
       call proc_alloc_outpt(filename,procname,arrayname,size, &
            'cr8',3,locierr,locqdie,kind(cr8),sz2p=siz2,sz3p=siz3,isallocated=allocated(cr8))
       ! chm_int4
    else if(present(ci4)) then
       allocate(ci4(loclb1:locub1,loclb2:locub2,loclb3:locub3), &
            stat=locierr)
       call proc_alloc_outpt(filename,procname,arrayname,size, &
            'ci4',3,locierr,locqdie,kind(ci4),sz2p=siz2,sz3p=siz3,isallocated=allocated(ci4))
       ! int_byte
    else if(present(iby)) then
       allocate(iby(loclb1:locub1,loclb2:locub2,loclb3:locub3), &
            stat=locierr)
       call proc_alloc_outpt(filename,procname,arrayname,size, &
            'iby',3,locierr,locqdie,kind(iby),sz2p=siz2,sz3p=siz3,isallocated=allocated(iby))
       ! chm_int2
    else if(present(ci2)) then
       allocate(ci2(loclb1:locub1,loclb2:locub2,loclb3:locub3), &
            stat=locierr)
       call proc_alloc_outpt(filename,procname,arrayname,size, &
            'ci2',3,locierr,locqdie,kind(ci2),sz2p=siz2,sz3p=siz3,isallocated=allocated(ci2))
       ! character(len=1)
    else if(present(ch1)) then
       allocate(ch1(loclb1:locub1,loclb2:locub2,loclb3:locub3), &
            stat=locierr)
#ifdef KEY_RESIZE
       ch1=''
#endif              
       call proc_alloc_outpt(filename,procname,arrayname,size, &
            'ch1',3,locierr,locqdie,kind(ch1),sz2p=siz2,sz3p=siz3,isallocated=allocated(ch1))
       ! character(len=2)
    else if(present(ch2)) then
       allocate(ch2(loclb1:locub1,loclb2:locub2,loclb3:locub3), &
            stat=locierr)
#ifdef KEY_RESIZE
       ch2=''
#endif              
       call proc_alloc_outpt(filename,procname,arrayname,size, &
            'ch2',3,locierr,locqdie,kind(ch2),sz2p=siz2,sz3p=siz3,isallocated=allocated(ch2))
       ! character(len=4)
    else if(present(ch4)) then
       allocate(ch4(loclb1:locub1,loclb2:locub2,loclb3:locub3), &
            stat=locierr)
#ifdef KEY_RESIZE
       ch4=''
#endif              
       call proc_alloc_outpt(filename,procname,arrayname,size, &
            'ch4',3,locierr,locqdie,kind(ch4),sz2p=siz2,sz3p=siz3,isallocated=allocated(ch4))
       ! character(len=16)
    else if(present(ch16)) then
       allocate(ch16(loclb1:locub1,loclb2:locub2,loclb3:locub3), &
            stat=locierr)
       call proc_alloc_outpt(filename,procname,arrayname,size, &
            'ch16',3,locierr,locqdie,kind(ch16),sz2p=siz2,sz3p=siz3,isallocated=allocated(ch16))
    else
       ! write warning if no array match
       write(6,*) 'all3d No array matched list of available types.',filename,procname,arrayname
       !        if(qdie) then call wrndie...
    endif
    ! save error status
    if(present(ierr)) then
       ierr = locierr 
    endif
  END SUBROUTINE alloc_3d
  !______________________________________________________________________
  !______________________________________________________________________
  !
  SUBROUTINE alloc_4d(filename,procname,arrayname,size,siz2,siz3,siz4, &
       crl,cr4,cr8,rlg,intg,ci2,ci4,ci8,iby,ch1,ch2,ch4,ch6,ch8,ch16,log, &
       cmpx,qdie,ierr,lbou,lbou2,lbou3,lbou4)
    !
    !  This routine handles explicit 3D array allocations
    !                                                --RJP Nov 2008
    implicit none
    character(len=*),intent(in) :: filename,procname,arrayname
    integer,intent(in) :: size,siz2,siz3,siz4
    integer,optional,intent(in) :: lbou,lbou2,lbou3,lbou4
    logical,optional,intent(in) :: qdie
    integer,optional,intent(out) :: ierr
    ! passed arrays, all optional arguments
    real(kind=chm_real),allocatable,optional,dimension(:,:,:,:) :: crl
    real(kind=chm_real4),allocatable,optional,dimension(:,:,:,:) :: cr4
    real(kind=chm_real8),allocatable,optional,dimension(:,:,:,:) :: cr8
    real,allocatable,optional,dimension(:,:,:,:) :: rlg
    integer,allocatable,optional,dimension(:,:,:,:) :: intg
    integer(kind=int_byte),allocatable,optional, &
         dimension(:,:,:,:) :: iby
    integer(kind=chm_int2),allocatable,optional, &
         dimension(:,:,:,:) :: ci2
    integer(kind=chm_int4),allocatable,optional, &
         dimension(:,:,:,:) :: ci4
    integer(kind=chm_int8),allocatable,optional, &
         dimension(:,:,:,:) :: ci8
    character(len=1),allocatable,optional,dimension(:,:,:,:) :: ch1
    character(len=2),allocatable,optional,dimension(:,:,:,:) :: ch2
    character(len=4),allocatable,optional,dimension(:,:,:,:) :: ch4
    character(len=6),allocatable,optional,dimension(:,:,:,:) :: ch6
    character(len=8),allocatable,optional,dimension(:,:,:,:) :: ch8
    character(len=16),allocatable,optional,dimension(:,:,:,:) :: ch16
    logical,allocatable,optional,dimension(:,:,:,:) :: log
    complex(kind=chm_cmpx),allocatable,optional,dimension(:,:,:,:) :: cmpx 
    !
    ! local variables
    logical :: locqdie
    integer :: hit, locierr,loclb1,locub1,loclb2,locub2,loclb3,locub3,loclb4,locub4
    !***********************************************************************
    ! end of variable declarations
    !***********************************************************************
    ! determine bounds
    loclb1 = 1
    if(present(lbou)) then
       loclb1 = lbou
    endif
    locub1 = loclb1 + size -1  
    !
    loclb2 = 1
    if(present(lbou2)) then
       loclb2 = lbou2
    endif
    locub2 = loclb2 + siz2 -1
    !
    loclb3 = 1
    if(present(lbou3)) then
       loclb3 = lbou3
    endif
    locub3 = loclb3 + siz3 -1
    !
    loclb4 = 1
    if(present(lbou4)) then
       loclb4 = lbou4
    endif
    locub4 = loclb4 + siz4 -1
    !
    ! determine whether to die on errors
    locqdie = .true.
    if(present(qdie)) then
       locqdie = qdie
    endif
    !----------------------------------------------------------------------
    ! check to see which array type is present in arguments
    ! chm_real
    if(present(crl)) then
       allocate(crl(loclb1:locub1,loclb2:locub2,loclb3:locub3,loclb4:locub4), &
            stat=locierr)
       call proc_alloc_outpt(filename,procname,arrayname,size, &
            'crl',3,locierr,locqdie,kind(crl), &
            sz2p=siz2,sz3p=siz3,sz4p=siz4,isallocated=allocated(crl))
       ! general integer
    else if(present(intg)) then
       allocate(intg(loclb1:locub1,loclb2:locub2,loclb3:locub3,loclb4:locub4), &
            stat=locierr)
       call proc_alloc_outpt(filename,procname,arrayname,size, &
            'intg',3,locierr,locqdie,kind(intg),sz2p=siz2,sz3p=siz3,sz4p=siz4, &
            isallocated=allocated(intg))
       ! general real
    else if(present(rlg)) then
       allocate(rlg(loclb1:locub1,loclb2:locub2,loclb3:locub3,loclb4:locub4), &
            stat=locierr)
       call proc_alloc_outpt(filename,procname,arrayname,size, &
            'rlg',3,locierr,locqdie,kind(rlg),sz2p=siz2,sz3p=siz3,sz4p=siz4, &
            isallocated=allocated(rlg))
       ! character(len=6)
    else if(present(ch6)) then
       allocate(ch6(loclb1:locub1,loclb2:locub2,loclb3:locub3,loclb4:locub4), &
            stat=locierr)
#ifdef KEY_RESIZE
       ch6=''
#endif              
       call proc_alloc_outpt(filename,procname,arrayname,size, &
            'ch6',3,locierr,locqdie,kind(ch6),sz2p=siz2,sz3p=siz3,sz4p=siz4, &
            isallocated=allocated(ch6))
       ! character(len=8)
    else if(present(ch8)) then
       allocate(ch8(loclb1:locub1,loclb2:locub2,loclb3:locub3,loclb4:locub4), &
            stat=locierr)
#ifdef KEY_RESIZE
       ch8=''
#endif              
       call proc_alloc_outpt(filename,procname,arrayname,size, &
            'ch8',3,locierr,locqdie,kind(ch8),sz2p=siz2,sz3p=siz3,sz4p=siz4, &
            isallocated=allocated(ch8))
       !! chm_real4
    else if(present(cr4)) then
       allocate(cr4(loclb1:locub1,loclb2:locub2,loclb3:locub3,loclb4:locub4), &
            stat=locierr)
       call proc_alloc_outpt(filename,procname,arrayname,size, &
            'cr4',3,locierr,locqdie,kind(cr4),sz2p=siz2,sz3p=siz3,sz4p=siz4, &
            isallocated=allocated(cr4))
       ! chm_int8
    else if(present(ci8)) then
       allocate(ci8(loclb1:locub1,loclb2:locub2,loclb3:locub3,loclb4:locub4), &
            stat=locierr)
       call proc_alloc_outpt(filename,procname,arrayname,size, &
            'ci8',3,locierr,locqdie,kind(ci8),sz2p=siz2,sz3p=siz3,sz4p=siz4, &
            isallocated=allocated(ci8))
       ! logical
    else if(present(log)) then
       allocate(log(loclb1:locub1,loclb2:locub2,loclb3:locub3,loclb4:locub4), &
            stat=locierr)
       call proc_alloc_outpt(filename,procname,arrayname,size, &
            'log',3,locierr,locqdie,kind(log),sz2p=siz2,sz3p=siz3,sz4p=siz4, &
            isallocated=allocated(log))
       ! chm_cmpx
    else if(present(cmpx)) then
       allocate(cmpx(loclb1:locub1,loclb2:locub2,loclb3:locub3,loclb4:locub4), &
            stat=locierr)
       call proc_alloc_outpt(filename,procname,arrayname,size, &
            'cmpx',3,locierr,locqdie,kind(cmpx),sz2p=siz2,sz3p=siz3,sz4p=siz4, &
            isallocated=allocated(cmpx))
       !less common types
       ! chm_real8
    else if(present(cr8)) then
       allocate(cr8(loclb1:locub1,loclb2:locub2,loclb3:locub3,loclb4:locub4), &
            stat=locierr)
       call proc_alloc_outpt(filename,procname,arrayname,size, &
            'cr8',3,locierr,locqdie,kind(cr8),sz2p=siz2,sz3p=siz3,sz4p=siz4, &
            isallocated=allocated(cr8))
       ! chm_int4
    else if(present(ci4)) then
       allocate(ci4(loclb1:locub1,loclb2:locub2,loclb3:locub3,loclb4:locub4), &
            stat=locierr)
       call proc_alloc_outpt(filename,procname,arrayname,size, &
            'ci4',3,locierr,locqdie,kind(ci4),sz2p=siz2,sz3p=siz3,sz4p=siz4, &
            isallocated=allocated(ci4))
       ! int_byte
    else if(present(iby)) then
       allocate(iby(loclb1:locub1,loclb2:locub2,loclb3:locub3,loclb4:locub4), &
            stat=locierr)
       call proc_alloc_outpt(filename,procname,arrayname,size, &
            'iby',3,locierr,locqdie,kind(iby),sz2p=siz2,sz3p=siz3,sz4p=siz4 &
            ,isallocated=allocated(iby))
       ! chm_int2
    else if(present(ci2)) then
       allocate(ci2(loclb1:locub1,loclb2:locub2,loclb3:locub3,loclb4:locub4), &
            stat=locierr)
       call proc_alloc_outpt(filename,procname,arrayname,size, &
            'ci2',3,locierr,locqdie,kind(ci2),sz2p=siz2,sz3p=siz3,sz4p=siz4, &
            isallocated=allocated(ci2))
       ! character(len=1)
    else if(present(ch1)) then
       allocate(ch1(loclb1:locub1,loclb2:locub2,loclb3:locub3,loclb4:locub4), &
            stat=locierr)
#ifdef KEY_RESIZE
       ch1=''
#endif              
       call proc_alloc_outpt(filename,procname,arrayname,size, &
            'ch1',3,locierr,locqdie,kind(ch1),sz2p=siz2,sz3p=siz3,sz4p=siz4, &
            isallocated=allocated(ch1))
       ! character(len=2)
    else if(present(ch2)) then
       allocate(ch2(loclb1:locub1,loclb2:locub2,loclb3:locub3,loclb4:locub4), &
            stat=locierr)
#ifdef KEY_RESIZE
       ch2=''
#endif              
       call proc_alloc_outpt(filename,procname,arrayname,size, &
            'ch2',3,locierr,locqdie,kind(ch2),sz2p=siz2,sz3p=siz3,sz4p=siz4, &
            isallocated=allocated(ch2))
       ! character(len=4)
    else if(present(ch4)) then
       allocate(ch4(loclb1:locub1,loclb2:locub2,loclb3:locub3,loclb4:locub4), &
            stat=locierr)
#ifdef KEY_RESIZE
       ch4=''
#endif              
       call proc_alloc_outpt(filename,procname,arrayname,size, &
            'ch4',3,locierr,locqdie,kind(ch4),sz2p=siz2,sz3p=siz3,sz4p=siz4, &
            isallocated=allocated(ch4))
       ! character(len=16)
    else if(present(ch16)) then
       allocate(ch16(loclb1:locub1,loclb2:locub2,loclb3:locub3,loclb4:locub4), &
            stat=locierr)
#ifdef KEY_RESIZE
       ch16=''
#endif              
       call proc_alloc_outpt(filename,procname,arrayname,size, &
            'ch16',3,locierr,locqdie,kind(ch16),sz2p=siz2,sz3p=siz3,sz4p=siz4, &
            isallocated=allocated(ch16))
    else
       ! write warning if no array match
       write(6,*) 'all4d No array matched list of available types.',filename,procname,arrayname
       !        if(qdie) then call wrndie...
    endif
    ! save error status
    if(present(ierr)) then
       ierr = locierr 
    endif
  end SUBROUTINE alloc_4d
  !______________________________________________________________________
  !______________________________________________________________________
  !
  SUBROUTINE alloc_vchar(filename,procname,arrayname,size, &
       cha,chlen,qdie,ierr,lbou)
    !
    !     This is a special-case routine which handles character arrays
    !     of variable (programmer-specified) length.  These arrays need
    !     to be deallocated with corresponding calls to dealloc_vchar.
    !                                         --RJP Nov 2008
    implicit none
    integer :: chlen
    character(len=chlen),allocatable,dimension(:) :: cha
    character(len=*),intent(in) :: filename,procname,arrayname
    integer,intent(in) :: size
    integer,optional,intent(in) :: lbou
    logical,optional,intent(in) :: qdie
    integer,optional,intent(out) :: ierr
    ! local variables
    logical :: locqdie
    integer :: locierr,loclb,locub
    !***************************************************************

    write(6,*) 'TO HERE 0+++++++++++++++'
    ! test for presence of qdie
    locqdie = .true.
    if(present(qdie)) then
       locqdie = qdie
    endif
    ! determine bounds
    loclb = 1
    if(present(lbou)) then
       loclb = lbou
    endif
    locub = loclb + size - 1  
    ! determine whether to die on errors
    locqdie = .true.
    if(present(qdie)) then
       locqdie = qdie
    endif
    write(6,*) 'TO HERE 1+++++++++++++++'
    ! process allocation
    allocate(cha(loclb:locub),stat=locierr)
#ifdef KEY_RESIZE
       cha=''
#endif              
    write(6,*) 'TO HERE 2+++++++++++++++'
    call proc_alloc_outpt(filename,procname,arrayname,size, &
         'vchar',1,locierr,locqdie,kind(cha),isallocated=allocated(cha))
    write(6,*) 'TO HERE 3+++++++++++++++'
    !
    ! save error status
    if(present(ierr)) then
       ierr = locierr
    endif
  end SUBROUTINE alloc_vchar
  !______________________________________________________________________
  !______________________________________________________________________
  !
  SUBROUTINE proc_alloc_outpt(filename,procname,arrayname, &
       size,arrtype,arrank,ierrp,qdiep,arkind,sz2p,sz3p,sz4p,isallocated)
    !
    !     This routine processes data associated with the memory
    !     allocation for an array
    !                                       --RJP Nov 2008
  use allocdat
    character(len=*),intent(in) :: filename,procname,arrayname
    character(len=*),intent(in) :: arrtype
    integer,intent(in) :: arrank,ierrp,arkind
    logical,intent(in) :: qdiep 
    integer,intent(in) :: size
    integer,optional,intent(in) :: sz2p,sz3p,sz4p
    logical,optional,intent(in) :: isallocated
    !
    !     loca
    logical :: qdie
    !
    !!     qprnallocf    true if printing allocations on the fly
    !     qprndeallocf  true if printing deallocations on the fly
    !     qaccumallocdb   true if accumulating database
    !***************************************************************
    !
    qdie = qdiep
    if(qnoallocdie) qdie=.false.  !if set globally, don't die
    if (ierrp.gt.0) then  !if unsuccessful allocation
       call allocerr(filename,procname,arrayname,size,qdie,isallocated)
       if (qaccumallocdb) call accum_allocfail(filename,procname, &
            arrayname,size,arrtype,arrank,arkind,sz2p,sz3p,sz4p)
       !if accumulating database, call accumulation routine
    else 
       if (qaccumallocdb) call accum_allocdb(filename,procname, &
            arrayname,size,arrtype,arrank,arkind,sz2p,sz3p,sz4p)
    endif
    ! if printing on the fly
    if(qprnallocf) call prnalloc_fly(filename,procname,arrayname, &
         size,arrtype,arrank,arkind,ierrp,sz2p,sz3p,sz4p)
    !

  END SUBROUTINE proc_alloc_outpt
  !______________________________________________________________________
  !______________________________________________________________________
  SUBROUTINE allocerr(filename,procname,arrayname,size,qdie,isallocated)
    ! printout for allocation errors            RJP Nov 2008
    character(len=*),intent(in) :: filename,procname,arrayname
    integer,intent(in) :: size
    logical,intent(in) :: qdie,isallocated
    !***************************************************************
    !
    WRITE(6,'(A25,1X,A10,1X,A12,1X,A10,1X,A9,1X,A10,i20)') &
         'MEMORY ALLOCATION ERROR @',arrayname,'OF PROCEDURE', &
         procname,'IN SOURCE',filename,size
    if(isallocated)then
       write(6,"(a)") "Array already allocated"
    else
       write(6,"(a)") "Unknown reason for error"
    endif
    if(qdie) then
       STOP
       !       call wrndie(-5,
       !     & procname,'array not deallocated')
    endif
  END SUBROUTINE allocerr
  !______________________________________________________________________
  !______________________________________________________________________
  !
  SUBROUTINE accum_allocdb(filename,procname,arrayname, &
       size,arrtype,arrank,arkind,siz2,siz3,siz4)
    !
    !     Adds the passed array information to the database for the 
    !     corresponding successful allocation. 
    !     Allocates memory for the database initially.
    !                                                  RJP Nov 2008
  use allocdat
    character(len=*),intent(in) :: filename,procname,arrayname
    character(len=*),intent(in) :: arrtype
    integer,intent(in) :: arrank,arkind
    integer,intent(in) :: size
    integer,optional,intent(in) :: siz2,siz3,siz4
    ! local variables
    integer :: ii,locsize
    !
    !***************************************************************
    !
    !       allocdbarsz = 10000 !should be set in TEST command 
    if(.not.allocated(arnamar)) then !create general flag here?
       !         write(6,*) 'allocating memory for database'
       allocate(filnamar(allocdbarsz),arnamar(allocdbarsz), &
            artypar(allocdbarsz),procnamar(allocdbarsz), &
            arrankar(allocdbarsz),arsizear(allocdbarsz), &
            arkindar(allocdbarsz))
       ! initialize arrays
       do ii = 1,allocdbarsz
          filnamar(ii)=' '
          arnamar(ii)=' '
          artypar(ii)=' '
          procnamar(ii)=' '
          arrankar(ii)=0
          arsizear(ii)=0
          arkindar(ii)=0
       enddo
       nallocentr=0
    endif
    locsize=size
    if(present(siz2)) locsize=locsize*siz2
    if(present(siz3)) locsize=locsize*siz3
    if(present(siz4)) locsize=locsize*siz4
    !
    if(nallocentr.lt.allocdbarsz) then
       nallocentr = nallocentr + 1
       filnamar(nallocentr)=filename
       procnamar(nallocentr)=procname
       arnamar(nallocentr)=arrayname
       artypar(nallocentr)=arrtype
       arrankar(nallocentr)=arrank
       arsizear(nallocentr)=locsize  !size is the overall size
       arkindar(nallocentr)=arkind   !kind value
    else
       write(6,'(A65)') &
            'WARNING: # of alloctd arrys > analysis limit (TEST command)'
    endif
  end SUBROUTINE accum_allocdb
  !______________________________________________________________________
  !______________________________________________________________________
  !
  SUBROUTINE prn_allocdb(qpsucc,qpfail)
    !    
    !     Prints out the accumulated database information, for
    !     either successful or failed allocations, or both
    !                                         --RJP Nov 2008
  use allocdat
    logical,intent(in) :: qpsucc,qpfail
    ! local variables
    integer :: nar
    real(kind=chm_real) :: ssum,msum
    logical :: locqdiep
    !***************************************************************
    !
    !   loop over database and print out      
    ! 
    if(qpsucc) then 
       if(.not.(qaccumallocdb)) write(6,'(5X,A63)') &
            'WARNING:  NOT CURRENTLY ACCUMULATING SUCCESSFUL ALLOC DATABASE'
       if(nallocentr.gt.0) then
          write(6,'(5X,A70)') &
               '--------------------------------------------------------------'
          write(6,'(25X,A41)') 'SUCCESSFULLY MEMORY MOD-ALLOCATED ARRAYS'
          write(6,'(5X,A70)') &
               '--------------------------------------------------------------'
          write(6,'(10X,A8,1X,A6,5X,A7,1X,A8,2X,A6,3X,A4,1X,A12,3X,A4)')  &
               'number','file','process','name', &
               'type','rank','size','kind'
          ssum = 0
          msum = 0 
          do nar = 1,nallocentr
             ssum = ssum + arsizear(nar)
             !        write(6,*) 'array is ',nar,'kind is ',arkindar(nar)
             msum = msum + arsizear(nar)*arkindar(nar)
             call prn_allocdpt('ALLOCDB>',nar,filnamar(nar),procnamar(nar), &
                  arnamar(nar),artypar(nar),arrankar(nar),arsizear(nar), &
                  arkindar(nar))
          enddo
          write(6,'(18X,A34,11X,F15.0)')  &
               'Total words successfully allocated',ssum
          write(6,'(18X,A44,1X,F15.0)') &
               'Estimated total bytes successfully allocated',msum
       else
          write(6,'(5X,A70)') &
               '--------------------------------------------------------------'
          write(6,'(24X,A37)') 'NO SUCCESSFUL MEMORY MOD ALLOCATIONS'
          write(6,'(5X,A70)') &
               '--------------------------------------------------------------'
       endif ! if # alloc > 0
    endif !if printing successes
    !---------------------------------------------------------------------
    if(qpfail) then
       if(.not.(qaccumallocdb)) write(6,'(5X,A59)') &
            'WARNING:  NOT CURRENTLY ACCUMULATING FAILED ALLOC DATABASE'
       if(nfallocentr.gt.0) then
          write(6,'(5X,A70)') &
               '--------------------------------------------------------------'
          write(6,'(27X,A38)') 'FAILED MEMORY MOD ALLOCATION ATTEMPTS'
          write(6,'(5X,A70)') &
               '--------------------------------------------------------------'
          write(6,'(12X,A8,1X,A6,5X,A7,1X,A8,2X,A6,3X,A4,1X,A12)') &
               'number','file','process','name', &
               'type','rank','size'
          ssum = 0
          msum = 0
          do nar = 1,nfallocentr
             ssum = ssum + farsizear(nar)
             msum = msum + farkindar(nar)*farsizear(nar)
             call prn_allocdpt('ALLFLDB>',nar,filnamar(nar),procnamar(nar), &
                  arnamar(nar),artypar(nar),arrankar(nar),arsizear(nar), &
                  arkindar(nar))
          enddo
          write(6,'(A56,7X,F15.0)') &
               'Total words unsuccessfully allocated ',ssum
          write(6,'(16X,A46,1X,F15.0)') &
               'Estimated total bytes unsuccessfully allocated',msum
       else
          write(6,'(5X,A70)') &
               '--------------------------------------------------------------'
          write(6,'(30X,A33)') 'NO FAILED MEMORY MOD ALLOCATIONS'
          write(6,'(5X,A70)') &
               '--------------------------------------------------------------'
       endif !if # > 0
    endif
    !
  end SUBROUTINE prn_allocdb
  !______________________________________________________________________
  !______________________________________________________________________
  !
  SUBROUTINE prnalloc_fly(filename,procname,arrayname, &
       size,artype,arrank,arkind,ierrp,siz2,siz3,siz4)
    !
    !     Prints information about an allocation (successful or failed)
    !     immediately after it has occurred--i.e. "on the fly"
    !                                           --RJP Nov 2008
  use allocdat
    character(len=*),intent(in) :: filename,procname,arrayname
    character(len=*),intent(in) :: artype
    integer,intent(in) :: arrank,ierrp,size,arkind
    integer,optional,intent(in) :: siz2,siz3,siz4
    ! local variables
    integer :: locsize
    !***************************************************************
    !      
    locsize=size
    if(present(siz2)) locsize=locsize*siz2
    if(present(siz3)) locsize=locsize*siz3
    if(present(siz4)) locsize=locsize*siz4
    if(ierrp.eq.0) then
       write(6,'(A)') 'Successfully Allocated Array' 
       write(6,'(10X,A8,1X,A6,5X,A7,1X,A8,2X,A6,3X,A4,1X,A12,3X,A4)') &
            'number','file','process','name', &
            'type','rank','size','kind'
       call prn_allocdpt('ALLOCOK>',nallocentr,filename,procname, &
            arrayname,artype,arrank,locsize,arkind)
    else
       write(6,'(A)') '****Could NOT Allocate Array****'
       write(6,'(10X,A8,1X,A6,5X,A7,1X,A8,2X,A6,3X,A4,1X,A12,3X,A4)') &
            'number','file','process','name','type','rank','size','kind'
       call prn_allocdpt('ALLOCFL>',nfallocentr,filename,procname, &
            arrayname,artype,arrank,locsize,arkind)
    endif
    write(6,*) 'end of routine prnalloc_fly'
  end SUBROUTINE prnalloc_fly
  !______________________________________________________________________
  !______________________________________________________________________
  !
  SUBROUTINE accum_allocfail(filename,procname,arrayname, &
       size,arrtype,arrank,arkind,siz2,siz3,siz4)
    !
    !     Adds the passed array information to the database for the
    !     corresponding failed allocation.
    !     Allocates memory for the failure database initially.
    !                                        --RJP Nov 2008
  use allocdat
    character(len=*),intent(in) :: filename,procname,arrayname
    character(len=*),intent(in) :: arrtype
    integer,intent(in) :: arrank,arkind
    !      logical,intent(in) :: qdiep
    integer,intent(in) :: size
    integer,optional,intent(in) :: siz2,siz3,siz4
    ! local variables
    integer :: ii,locsize
    !***************************************************************
    !
    !       fallocdbarsz = 1000 !should be set in TEST command 
    if(.not.allocated(farnamar)) then !create general flag here?
       !         write(6,*) 'allocating memory for database'
       ! these should be deallocated at end of run
       allocate(ffilnamar(fallocdbarsz),farnamar(fallocdbarsz), &
            fartypar(fallocdbarsz),fprocnamar(fallocdbarsz), &
            farrankar(fallocdbarsz),farsizear(fallocdbarsz), &
            farkindar(fallocdbarsz))
       ! initialize arrays
       do ii = 1,fallocdbarsz
          ffilnamar(ii)=' '
          farnamar(ii)=' '
          fartypar(ii)=' '
          fprocnamar(ii)=' '
          farrankar(ii)=0
          farsizear(ii)=0
          farkindar(ii)=0
       enddo
       nfallocentr=0
    endif
    !      write(6,*) 'CALLED ACCUM_ALLOCFAIL'
    !      write(6,*) 'nfallocentr is',nfallocentr
    locsize=size
    if(present(siz2)) locsize=locsize*siz2
    if(present(siz3)) locsize=locsize*siz3
    if(present(siz4)) locsize=locsize*siz4
    !
    if(nfallocentr.lt.fallocdbarsz) then
       nfallocentr = nfallocentr + 1
       ffilnamar(nfallocentr)=filename
       fprocnamar(nfallocentr)=procname
       farnamar(nfallocentr)=arrayname
       fartypar(nfallocentr)=arrtype
       farrankar(nfallocentr)=arrank
       farsizear(nfallocentr)=locsize  !size is the overall size
       farkindar(nfallocentr)=arkind
    else
       write(6,'(A65)') &
            'WARNING: # failed alloctd arrys > analysis limit (TEST command)'
    endif
  END SUBROUTINE accum_allocfail
  !______________________________________________________________________
  !______________________________________________________________________
  !
  SUBROUTINE prn_allocdpt(label,num,afile,aproc,aname,atype,arank, &
       asize,akind,int1)
    !
    !     prints a single data point (line) of the allocation database
    !
    character(len=*),intent(in) :: afile,aproc,aname,label
    character(len=*),intent(in) :: atype
    integer,intent(in) :: num,arank,asize,akind
    integer,optional,intent(in) :: int1  !optional integer
    !
    if(.not.present(int1)) then
       write(6,'(A8,1X,I9,1X,A8,1X,A10,1X,A10,1X,A6,1X,I4,1X,I15,1X,I3)') &
            label,num,afile,aproc,aname,atype,arank,asize,akind
       !
    else
       write(6,'(A8,1X,I9,1X,A8,1X,A10,1X,A10,1X,A6,1X,I4,1X,I15,1X,I3,1X,I3)') &
            label,num,afile,aproc,aname,atype,arank,asize,akind,int1
    endif
    !
  END SUBROUTINE prn_allocdpt
  !______________________________________________________________________
  !______________________________________________________________________
  SUBROUTINE reset_allocdb
    !     initializes successful and failed allocation databases
    !
  use allocdat
    integer :: ii
    ! initialize alloc arrays
    WRITE(6,*) 'INITIALIZING ALLOCATION DATABASES'
    if(allocated(filnamar)) then
       do ii = 1,allocdbarsz
          filnamar(ii)=' '
          arnamar(ii)=' '
          artypar(ii)=' '
          procnamar(ii)=' '
          arrankar(ii)=0
          arsizear(ii)=0
          arkindar(ii)=0
       enddo
    endif
    nallocentr=0
    ! initialize falloc arrays
    if(allocated(ffilnamar)) then
       do ii = 1,fallocdbarsz
          ffilnamar(ii)=' '
          farnamar(ii)=' '
          fartypar(ii)=' '
          fprocnamar(ii)=' '
          farrankar(ii)=0
          farsizear(ii)=0
          farkindar(ii)=0
       enddo
    endif
    nfallocentr=0
    !
  END SUBROUTINE reset_allocdb
  !

  subroutine allocate_chm_xyz(fil,routine,var,a,n)
    type(chm_xyz),allocatable,dimension(:) :: a
    character(len=*),intent(in) :: fil,routine,var
    integer,intent(in) :: n
    integer :: alloc_err

    allocate(a(n),stat=alloc_err)
    
    call proc_alloc_outpt(fil,routine,var,n,'chm_xyz',1,alloc_err,.false.,-999)
    return
  end subroutine allocate_chm_xyz

  subroutine chmalloc_chm_xyz(fil,routine,var,n,a)
    type(chm_xyz) :: a
    character(len=*),intent(in) :: fil,routine,var
    integer,intent(in) :: n
    call alloc_1d(fil,routine,var,n,crl=a%x)
    call alloc_1d(fil,routine,var,n,crl=a%y)
    call alloc_1d(fil,routine,var,n,crl=a%z)
    return
  end subroutine chmalloc_chm_xyz

  subroutine chmalloc_chm_array(fil,routine,var,n,p)
    type(chm_array),dimension(:),allocatable :: p
    character(len=*) :: fil,routine,var
    integer,intent(in) :: n
    integer alloc_err
    allocate(p(n),stat=alloc_err)
    call proc_alloc_outpt(fil,routine,var,n,'chm_array',1,alloc_err,.false.,-999)
  end subroutine chmalloc_chm_array

  subroutine chmalloc_chm_iptr(fil,routine,var,n,p)
    type(chm_iarray),dimension(:),allocatable :: p
    character(len=*) :: fil,routine,var
    integer,intent(in) :: n
    integer alloc_err
    allocate(p(n),stat=alloc_err)
    call proc_alloc_outpt(fil,routine,var,n,'chm_array',1,alloc_err,.false.,-999)
  end subroutine chmalloc_chm_iptr


END MODULE allocation

