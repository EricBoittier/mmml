MODULE deallocation
  !
  ! This module handles explicit memory deallocation through calls to
  !  chmdealloc(), which take the form:
  !
  !  call chmdealloc('filename','procname','arrayname',size1,(size2,)
  !     & (size3,) 
  !     & [array type keyword]= arrayname,ierr=errorname,
  !     & qdie=[.true. or .false.]) 
  !
  !     See allocation module for definitions and further explanation.
  !     As in the case of the chmalloc calls (allocations), 
  !     the size arguments and corresponding dummy argument keywords 
  !     are required in the chmdealloc calls.  Here, they are used for
  !     internal checking (comparison of specified or "intended" size with
  !     actual size of the deallocated array), and preserve the 
  !     symmetry of the allocation and deallocation code structures.
  !
  !                                        --RJP  Nov 2008
  use chm_kinds
  use chm_types
  implicit none

  INTERFACE chmdealloc
     module procedure dealloc_1d
     module procedure dealloc_2d
     module procedure dealloc_3d
     module procedure dealloc_4d
  END INTERFACE

CONTAINS
  !______________________________________________________________________ 
  !______________________________________________________________________
  !
  SUBROUTINE dealloc_1d(filename,procname,arrayname,siz1, &
       crl,crlp,mcrlp,cr4,cr4p,mcr4p, &
       cr8,rlg,intg,intgp,iby,mibyp,ci2,ci4,ci8, &
       ch1,ch2,ch4,ch6,ch8,ch16,ch20,ch36,ch80, &
       log,cmpx,qdie,ierr)
    !
    !      Handles explicit 1D deallocations  
    !                                               --RJP Nov 2008
    !
#if KEY_PARALLEL==1
    use mpi  
#endif
    implicit none
    !
    character(len=*),intent(in) :: filename,procname,arrayname
    logical,optional,intent(in) :: qdie
    integer,intent(in) :: siz1
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
    integer :: hit, locierr,loclb,locub,dsize
    !***********************************************************************
    ! end of variable declarations
    !***********************************************************************
    dsize = 0
    ! check to see whether to die on errors
    locqdie = .true.
    if(present(qdie)) locqdie = qdie
    ! check to see which array type is present in arguments
    ! most common types first
    ! chm_real
    if(present(crl)) then
       if (allocated(crl)) dsize = size(crl)
       deallocate(crl,stat=locierr)
       call proc_dealloc_outpt(filename,procname,arrayname,dsize, &
            'crl',1,locierr,locqdie,kind(crl),siz1)
    elseif(present(crlp)) then
       if (associated(crlp)) dsize = size(crlp)
       deallocate(crlp,stat=locierr)
       call proc_dealloc_outpt(filename,procname,arrayname,dsize, &
            'crlp',1,locierr,locqdie,kind(crlp),siz1)
       ! general integer
    else if (present(intg)) then
       if (allocated(intg)) dsize = size(intg)
       deallocate(intg,stat=locierr)
       call proc_dealloc_outpt(filename,procname,arrayname,dsize, &
            'intg',1,locierr,locqdie,kind(intg),siz1)
    else if (present(intgp)) then
       if (associated(intgp)) dsize = size(intgp)
       deallocate(intgp,stat=locierr)
       call proc_dealloc_outpt(filename,procname,arrayname,dsize, &
            'intgp',1,locierr,locqdie,kind(intgp),siz1)
       ! general real
    else if (present(rlg)) then
       if (allocated(rlg)) dsize = size(rlg)
       deallocate(rlg,stat=locierr)
       call proc_dealloc_outpt(filename,procname,arrayname,dsize, &
            'rlg',1,locierr,locqdie,kind(rlg),siz1)
       ! character(len=6)
    else if(present(ch6)) then
       if (allocated(ch6)) dsize = size(ch6)
       deallocate(ch6,stat=locierr)
       call proc_dealloc_outpt(filename,procname,arrayname,dsize, &
            'ch6',1,locierr,locqdie,kind(ch6),siz1)
       ! character(len=8)
    else if(present(ch8)) then
       if (allocated(ch8)) dsize = size(ch8)
       deallocate(ch8,stat=locierr)
       call proc_dealloc_outpt(filename,procname,arrayname,dsize, &
            'ch8',1,locierr,locqdie,kind(ch8),siz1)
       ! chm_real4
    else if(present(cr4)) then
       if (allocated(cr4)) dsize = size(cr4)
       deallocate(cr4,stat=locierr)
       call proc_dealloc_outpt(filename,procname,arrayname,dsize, &
            'cr4',1,locierr,locqdie,kind(cr4),siz1)
    else if(present(cr4p)) then
       if (associated(cr4p)) dsize = size(cr4p)
       deallocate(cr4p,stat=locierr)
       call proc_dealloc_outpt(filename,procname,arrayname,dsize, &
            'cr4p',1,locierr,locqdie,kind(cr4p),siz1)
       ! chm_int8
    else if(present(ci8)) then
       if (allocated(ci8)) dsize = size(ci8)
       deallocate(ci8,stat=locierr)
       call proc_dealloc_outpt(filename,procname,arrayname,dsize, &
            'ci8',1,locierr,locqdie,kind(ci8),siz1)
       ! logical
    else if (present(log)) then
       if (allocated(log)) dsize = size(log)
       deallocate(log,stat=locierr)
       call proc_dealloc_outpt(filename,procname,arrayname,dsize, &
            'log',1,locierr,locqdie,kind(log),siz1)
       ! chm_cmpx
    else if(present(cmpx)) then
       if (allocated(cmpx)) dsize = size(cmpx)
       deallocate(cmpx,stat=locierr)
       call proc_dealloc_outpt(filename,procname,arrayname,dsize, &
            'cmpx',1,locierr,locqdie,kind(cmpx),siz1)
       ! less common types
       ! chm_real8
    else if(present(cr8)) then
       if (allocated(cr8)) dsize = size(cr8)
       deallocate(cr8,stat=locierr)
       call proc_dealloc_outpt(filename,procname,arrayname,dsize, &
            'cr8',1,locierr,locqdie,kind(cr8),siz1)
       ! chm_int4
    else if(present(ci4)) then
       if (allocated(ci4)) dsize = size(ci4)
       deallocate(ci4,stat=locierr)
       call proc_dealloc_outpt(filename,procname,arrayname,dsize, &
            'ci4',1,locierr,locqdie,kind(ci4),siz1)
       ! int_byte
    else if(present(iby)) then
       if (allocated(iby)) dsize = size(iby)
       deallocate(iby,stat=locierr)
       call proc_dealloc_outpt(filename,procname,arrayname,dsize, &
            'iby',1,locierr,locqdie,kind(iby),siz1)

    else if(present(mibyp)) then
       if (associated(mibyp)) dsize = size(mibyp)
#if KEY_PARALLEL==1
       call mpi_free_mem(mibyp, locierr)
       if (locierr /= MPI_SUCCESS) then
          locierr = 1
       else
          locierr = 0
       endif
       nullify(mibyp)
#else /**/
       deallocate(mibyp,stat=locierr)
#endif 
       call proc_dealloc_outpt(filename,procname,arrayname,dsize, &
            'mibyp',1,locierr,locqdie,kind(mibyp),siz1)
       ! chm_int2
    else if(present(ci2)) then
       if (allocated(ci2)) dsize = size(ci2)
       deallocate(ci2,stat=locierr)
       call proc_dealloc_outpt(filename,procname,arrayname,dsize, &
            'ci2',1,locierr,locqdie,kind(ci2),siz1)
       ! character(len=1)
    else if(present(ch1)) then
       if (allocated(ch1)) dsize = size(ch1)
       deallocate(ch1,stat=locierr)
       call proc_dealloc_outpt(filename,procname,arrayname,dsize, &
            'ch1',1,locierr,locqdie,kind(ch1),siz1)
       ! character(len=2)
    else if(present(ch2)) then
       if (allocated(ch2)) dsize = size(ch2)
       deallocate(ch2,stat=locierr)
       call proc_dealloc_outpt(filename,procname,arrayname,dsize, &
            'ch2',1,locierr,locqdie,kind(ch2),siz1)
       ! character(len=4)
    else if(present(ch4)) then
       if (allocated(ch4)) dsize = size(ch4)
       deallocate(ch4,stat=locierr)
       call proc_dealloc_outpt(filename,procname,arrayname,dsize, &
            'ch4',1,locierr,locqdie,kind(ch4),siz1)
       ! character(len=16)
    else if(present(ch16)) then
       if (allocated(ch16)) dsize = size(ch16)
       deallocate(ch16,stat=locierr)
       call proc_dealloc_outpt(filename,procname,arrayname,dsize, &
            'ch16',1,locierr,locqdie,kind(ch16),siz1)
       ! character(len=20)
    else if(present(ch20)) then
       if (allocated(ch20)) dsize = size(ch20)
       deallocate(ch20,stat=locierr)
       call proc_dealloc_outpt(filename,procname,arrayname,dsize, &
            'ch20',1,locierr,locqdie,kind(ch20),siz1)
    else if(present(ch36)) then
       if (allocated(ch36)) dsize = size(ch36)
       deallocate(ch36,stat=locierr)
       call proc_dealloc_outpt(filename,procname,arrayname,dsize, &
            'ch36',1,locierr,locqdie,kind(ch36),siz1)
    else if(present(ch80)) then
       if (allocated(ch80)) dsize = size(ch80)
       deallocate(ch80,stat=locierr)
       call proc_dealloc_outpt(filename,procname,arrayname,dsize, &
            'ch80',1,locierr,locqdie,kind(ch80),siz1)
    elseif(present(mcrlp)) then
       if (associated(mcrlp)) dsize = size(mcrlp)
#if KEY_PARALLEL==1
       call mpi_free_mem(mcrlp, locierr)
       if (locierr /= MPI_SUCCESS) then
          locierr = 1
       else
          locierr = 0
       endif
       nullify(mcrlp)
#else /**/
       deallocate(mcrlp,stat=locierr)
#endif 
       call proc_dealloc_outpt(filename,procname,arrayname,dsize, &
            'mcrlp',1,locierr,locqdie,kind(mcrlp),siz1)
    elseif(present(mcr4p)) then
       if (associated(mcr4p)) dsize = size(mcr4p)
#if KEY_PARALLEL==1
       call mpi_free_mem(mcr4p, locierr)
       if (locierr /= MPI_SUCCESS) then
          locierr = 1
       else
          locierr = 0
       endif
       nullify(mcr4p)
#else /**/
       deallocate(mcr4p,stat=locierr)
#endif 
       call proc_dealloc_outpt(filename,procname,arrayname,dsize, &
            'mcr4p',1,locierr,locqdie,kind(mcr4p),siz1)
    else
       ! write warning if no array match
       write(6,*) 'deall1 No array matched list of available types.',filename,procname,arrayname
       !        if(qdie) then call wrndie...
    endif
    ! save error status
    if(present(ierr)) then
       ierr = locierr 
    endif
  end SUBROUTINE dealloc_1d
  !______________________________________________________________________ 
  !______________________________________________________________________
  !
  SUBROUTINE dealloc_2d(filename,procname,arrayname,siz1,siz2, &
       crl,crlp,mcrlp,cr4,cr4p,cr8,rlg,intg,intgp,iby,ci2,ci4,ci8,ch1,ch2,ch4,ch6,ch8,ch16,log, &
       qdie,ierr)
    !
    !      Handles explicit 2D deallocations   --RJP Nov 2008 
    !
#if KEY_PARALLEL==1
    use mpi  
#endif
    implicit none
    !      
    character(len=*),intent(in) :: filename,procname,arrayname
    logical,optional,intent(in) :: qdie
    integer,intent(in) :: siz1,siz2
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
    logical :: locqdie,verbose=.true.
    integer :: hit, locierr,dsize
    !***********************************************************************
    ! end of variable declarations
    !***********************************************************************
    dsize = 0
    ! determine whether to die on errors
    locqdie = .true.
    if(present(qdie)) locqdie = qdie
    !----------------------------------------------------------------------
    ! check to see which array type is present in arguments
    ! chm_real
    if(present(crl)) then
       if (allocated(crl)) dsize = size(crl)
       if(verbose)then
!          print *,"dealloc_2d attempt crl",arrayname,siz1,siz2,dsize
       endif
       deallocate(crl,stat=locierr)
       call proc_dealloc_outpt(filename,procname,arrayname,dsize, &
            'crl',2,locierr,locqdie,kind(crl),siz1,siz2)
       ! general integer
    elseif(present(crlp)) then
       if (associated(crlp)) dsize = size(crlp)
       if(verbose)then
!          print *,"dealloc_2d attempt crlp",arrayname,siz1,siz2,dsize
       endif
       deallocate(crlp,stat=locierr)
       call proc_dealloc_outpt(filename,procname,arrayname,dsize, &
            'crlp',2,locierr,locqdie,kind(crlp),siz1,siz2)
       ! general integer
    else if (present(intg)) then
       if (allocated(intg)) dsize = size(intg)
       deallocate(intg,stat=locierr)
       call proc_dealloc_outpt(filename,procname,arrayname,dsize, &
            'intg',2,locierr,locqdie,kind(intg),siz1,siz2)
    else if (present(intgp)) then
       if (associated(intgp)) dsize = size(intgp)
       deallocate(intgp,stat=locierr)
       call proc_dealloc_outpt(filename,procname,arrayname,dsize, &
            'intgp',2,locierr,locqdie,kind(intgp),siz1,siz2)
       ! general real
    else if (present(rlg)) then
       if (allocated(rlg)) dsize = size(rlg)
       deallocate(rlg,stat=locierr)
       call proc_dealloc_outpt(filename,procname,arrayname,dsize, &
            'rlg',2,locierr,locqdie,kind(rlg),siz1,siz2)
       ! character(len=6)
    else if(present(ch6)) then
       if (allocated(ch6)) dsize = size(ch6)
       deallocate(ch6,stat=locierr)
       call proc_dealloc_outpt(filename,procname,arrayname,dsize, &
            'ch6',2,locierr,locqdie,kind(ch6),siz1,siz2)
       ! character(len=8)
    else if(present(ch8)) then
       if (allocated(ch8)) dsize = size(ch8)
       deallocate(ch8,stat=locierr)
       call proc_dealloc_outpt(filename,procname,arrayname,dsize, &
            'ch8',2,locierr,locqdie,kind(ch8),siz1,siz2)
       ! chm_real4
    else if(present(cr4)) then
       if (allocated(cr4)) dsize = size(cr4)
       deallocate(cr4,stat=locierr)
       call proc_dealloc_outpt(filename,procname,arrayname,dsize, &
            'cr4',2,locierr,locqdie,kind(cr4),siz1,siz2)
    else if(present(cr4p)) then
       if (associated(cr4p)) dsize = size(cr4p)
       deallocate(cr4p,stat=locierr)
       call proc_dealloc_outpt(filename,procname,arrayname,dsize, &
            'cr4p',2,locierr,locqdie,kind(cr4p),siz1,siz2)
       ! chm_int8
    else if(present(ci8)) then
       if (allocated(ci8)) dsize = size(ci8)
       deallocate(ci8,stat=locierr)
       call proc_dealloc_outpt(filename,procname,arrayname,dsize, &
            'ci8',2,locierr,locqdie,kind(ci8),siz1,siz2)
       ! logical
    else if (present(log)) then
       if (allocated(log)) dsize = size(log)
       deallocate(log,stat=locierr)
       call proc_dealloc_outpt(filename,procname,arrayname,dsize, &
            'log',2,locierr,locqdie,kind(log),siz1,siz2)
       ! less common types
       ! chm_real8
    else if(present(cr8)) then
       if (allocated(cr8)) dsize = size(cr8)
       deallocate(cr8,stat=locierr)
       call proc_dealloc_outpt(filename,procname,arrayname,dsize, &
            'cr8',2,locierr,locqdie,kind(cr8),siz1,siz2)
       ! chm_int4
    else if(present(ci4)) then
       if (allocated(ci4)) dsize = size(ci4)
       deallocate(ci4,stat=locierr)
       call proc_dealloc_outpt(filename,procname,arrayname,dsize, &
            'ci4',2,locierr,locqdie,kind(ci4),siz1,siz2)
       ! int_byte
    else if(present(iby)) then
       if (allocated(iby)) dsize = size(iby)
       deallocate(iby,stat=locierr)
       call proc_dealloc_outpt(filename,procname,arrayname,dsize, &
            'iby',2,locierr,locqdie,kind(iby),siz1,siz2)
       ! chm_int2
    else if(present(ci2)) then
       if (allocated(ci2)) dsize = size(ci2)
       deallocate(ci2,stat=locierr)
       call proc_dealloc_outpt(filename,procname,arrayname,dsize, &
            'ci2',2,locierr,locqdie,kind(ci2),siz1,siz2)
       ! character(len=1)
    else if(present(ch1)) then
       if (allocated(ch1)) dsize = size(ch1)
       deallocate(ch1,stat=locierr)
       call proc_dealloc_outpt(filename,procname,arrayname,dsize, &
            'ch1',2,locierr,locqdie,kind(ch1),siz1,siz2)
       ! character(len=2)
    else if(present(ch2)) then
       if (allocated(ch2)) dsize = size(ch2)
       deallocate(ch2,stat=locierr)
       call proc_dealloc_outpt(filename,procname,arrayname,dsize, &
            'ch2',2,locierr,locqdie,kind(ch2),siz1,siz2)
       ! character(len=4)
    else if(present(ch4)) then
       if (allocated(ch4)) dsize = size(ch4)
       deallocate(ch4,stat=locierr)
       call proc_dealloc_outpt(filename,procname,arrayname,dsize, &
            'ch4',2,locierr,locqdie,kind(ch4),siz1,siz2)
       ! character(len=16)
    else if(present(ch16)) then
       if (allocated(ch16)) dsize = size(ch16)
       deallocate(ch16,stat=locierr)
       call proc_dealloc_outpt(filename,procname,arrayname,dsize, &
            'cr16',2,locierr,locqdie,kind(ch16),siz1,siz2)
    elseif(present(mcrlp)) then
       if (associated(mcrlp)) dsize = size(mcrlp)
#if KEY_PARALLEL==1
       call mpi_free_mem(mcrlp, locierr)
       if (locierr /= MPI_SUCCESS) then
          locierr = 1
       else
          locierr = 0
       endif
       nullify(mcrlp)
#else /**/
       deallocate(mcrlp,stat=locierr)
#endif 
       call proc_dealloc_outpt(filename,procname,arrayname,dsize, &
            'mcrlp',2,locierr,locqdie,kind(mcrlp),siz1,siz2)
    else
       ! write warning if no array match
       write(6,*) 'd2d No array matched list of available types.',filename,procname,arrayname
       !        if(qdie) then call wrndie...
    endif
    ! save error status
    if(present(ierr)) then
       ierr = locierr 
    endif
  end SUBROUTINE dealloc_2d
  !______________________________________________________________________ 
  !______________________________________________________________________
  !
  SUBROUTINE dealloc_3d(filename,procname,arrayname,siz1,siz2, &
       siz3,crl,cr4,cr8,rlg,intg,iby,ci2,ci4,ci8,ch1,ch2,ch4,ch6,ch8, &
       ch16,log,cmpx,qdie,ierr)
    !
    !      Handles explicit 3D deallocations   --RJP Nov 2008
    !
    implicit none
    !      
    character(len=*),intent(in) :: filename,procname,arrayname
    integer,intent(in) :: siz1,siz2,siz3
    integer,optional,intent(out) :: ierr
    logical,optional,intent(in) :: qdie
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
    integer :: hit, locierr,dsize,isize
    !***********************************************************************
    ! end of variable declarations
    !***********************************************************************
    dsize = 0
    !
    ! determine whether to die on errors
    locqdie = .true.
    if(present(qdie)) locqdie = qdie
    isize = siz1*siz2*siz3
    !----------------------------------------------------------------------
    ! check to see which array type is present in arguments
    ! chm_real
    if(present(crl)) then
       if (allocated(crl)) dsize = size(crl)
       deallocate(crl,stat=locierr)
       call proc_dealloc_outpt(filename,procname,arrayname,dsize, &
            'crl',3,locierr,locqdie,kind(crl),siz1,siz2,siz3)
       ! general integer
    else if(present(intg)) then
       if (allocated(intg)) dsize = size(intg)
       deallocate(intg,stat=locierr)
       call proc_dealloc_outpt(filename,procname,arrayname,dsize, &
            'intg',3,locierr,locqdie,kind(intg),siz1,siz2,siz3)
       ! general real
    else if(present(rlg)) then
       if (allocated(rlg)) dsize = size(rlg)
       deallocate(rlg,stat=locierr)
       call proc_dealloc_outpt(filename,procname,arrayname,dsize, &
            'rlg',3,locierr,locqdie,kind(rlg),siz1,siz2,siz3)
       ! character(len=6)
    else if(present(ch6)) then
       if (allocated(ch6)) dsize = size(ch6)
       deallocate(ch6,stat=locierr)
       call proc_dealloc_outpt(filename,procname,arrayname,dsize, &
            'ch6',3,locierr,locqdie,kind(ch6),siz1,siz2,siz3)
       ! character(len=8)
    else if(present(ch8)) then
       if (allocated(ch8)) dsize = size(ch8)
       deallocate(ch8,stat=locierr)
       call proc_dealloc_outpt(filename,procname,arrayname,dsize, &
            'ch8',3,locierr,locqdie,kind(ch8),siz1,siz2,siz3)
       !! chm_real4
    else if(present(cr4)) then
       if (allocated(cr4)) dsize = size(cr4)
       deallocate(cr4,stat=locierr)
       call proc_dealloc_outpt(filename,procname,arrayname,dsize, &
            'cr4',3,locierr,locqdie,kind(cr4),siz1,siz2,siz3)
       ! chm_int8
    else if(present(ci8)) then
       if (allocated(ci8)) dsize = size(ci8)
       deallocate(ci8,stat=locierr)
       call proc_dealloc_outpt(filename,procname,arrayname,dsize, &
            'ci8',3,locierr,locqdie,kind(ci8),siz1,siz2,siz3)
       ! logical
    else if(present(log)) then
       !         write(6,*) 'dsize for log3D before is ',dsize
       if (allocated(log)) dsize = size(log)
       !         write(6,*) 'dsize for log3D after is ',dsize
       deallocate(log,stat=locierr)
       call proc_dealloc_outpt(filename,procname,arrayname,dsize, &
            'log',3,locierr,locqdie,kind(log),siz1,siz2,siz3)
       ! chm_cmpx
    else if(present(cmpx)) then
       if (allocated(cmpx)) dsize = size(cmpx)
       deallocate(cmpx,stat=locierr)
       call proc_dealloc_outpt(filename,procname,arrayname,dsize, &
            'cmpx',3,locierr,locqdie,kind(cmpx),siz1,siz2,siz3)
       !less common types
       ! chm_real8
    else if(present(cr8)) then
       if (allocated(cr8)) dsize = size(cr8)
       deallocate(cr8,stat=locierr)
       call proc_dealloc_outpt(filename,procname,arrayname,dsize, &
            'cr8',3,locierr,locqdie,kind(cr8),siz1,siz2,siz3)
       ! chm_int4
    else if(present(ci4)) then
       if (allocated(ci4)) dsize = size(ci4)
       deallocate(ci4,stat=locierr)
       call proc_dealloc_outpt(filename,procname,arrayname,dsize, &
            'ci4',3,locierr,locqdie,kind(ci4),siz1,siz2,siz3)
       ! int_byte
    else if(present(iby)) then
       if (allocated(iby)) dsize = size(iby)
       deallocate(iby,stat=locierr)
       call proc_dealloc_outpt(filename,procname,arrayname,dsize, &
            'iby',3,locierr,locqdie,kind(iby),siz1,siz2,siz3)
       ! chm_int2
    else if(present(ci2)) then
       if (allocated(ci2)) dsize = size(ci2)
       deallocate(ci2,stat=locierr)
       call proc_dealloc_outpt(filename,procname,arrayname,dsize, &
            'ci2',3,locierr,locqdie,kind(ci2),siz1,siz2,siz3)
       ! character(len=1)
    else if(present(ch1)) then
       if (allocated(ch1)) dsize = size(ch1)
       deallocate(ch1,stat=locierr)
       call proc_dealloc_outpt(filename,procname,arrayname,dsize, &
            'ch1',3,locierr,locqdie,kind(ch1),siz1,siz2,siz3)
       ! character(len=2)
    else if(present(ch2)) then
       if (allocated(ch2)) dsize = size(ch2)
       deallocate(ch2,stat=locierr)
       call proc_dealloc_outpt(filename,procname,arrayname,dsize, &
            'ch2',3,locierr,locqdie,kind(ch2),siz1,siz2,siz3)
       ! character(len=4)
    else if(present(ch4)) then
       if (allocated(ch4)) dsize = size(ch4)
       deallocate(ch4,stat=locierr)
       call proc_dealloc_outpt(filename,procname,arrayname,dsize, &
            'ch4',3,locierr,locqdie,kind(ch4),siz1,siz2,siz3)
       ! character(len=16)
    else if(present(ch16)) then
       if (allocated(ch16)) dsize = size(ch16)
       deallocate(ch16,stat=locierr)
       call proc_dealloc_outpt(filename,procname,arrayname,dsize, &
            'ch16',3,locierr,locqdie,kind(ch16),siz1,siz2,siz3)
    else
       ! write warning if no array match
       write(6,*) 'd3d No array matched list of available types.',filename,procname,arrayname
       !        if(qdie) then call wrndie...
    endif
    ! save error status
    if(present(ierr)) then
       ierr = locierr 
    endif
  end SUBROUTINE dealloc_3d
  !______________________________________________________________________ 
  !______________________________________________________________________
  !
  SUBROUTINE dealloc_4d(filename,procname,arrayname,siz1,siz2, &
       siz3,siz4,crl,cr4,cr8,rlg,intg,iby,ci2,ci4,ci8,ch1,ch2,ch4,ch6,ch8, &
       ch16,log,cmpx,qdie,ierr)
    !
    !      Handles explicit 4D deallocations   --mfc jul 2009
    !
    implicit none
    !      
    character(len=*),intent(in) :: filename,procname,arrayname
    integer,intent(in) :: siz1,siz2,siz3,siz4
    integer,optional,intent(out) :: ierr
    logical,optional,intent(in) :: qdie
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
    integer :: hit, locierr,dsize,isize
    !***********************************************************************
    ! end of variable declarations
    !***********************************************************************
    dsize = 0
    !
    ! determine whether to die on errors
    locqdie = .true.
    if(present(qdie)) locqdie = qdie
    isize = siz1*siz2*siz3*siz4
    !----------------------------------------------------------------------
    ! check to see which array type is present in arguments
    ! chm_real
    if(present(crl)) then
       if (allocated(crl)) dsize = size(crl)
       deallocate(crl,stat=locierr)
       call proc_dealloc_outpt(filename,procname,arrayname,dsize, &
            'crl',3,locierr,locqdie,kind(crl),siz1,siz2,siz3,siz4)
       ! general integer
    else if(present(intg)) then
       if (allocated(intg)) dsize = size(intg)
       deallocate(intg,stat=locierr)
       call proc_dealloc_outpt(filename,procname,arrayname,dsize, &
            'intg',3,locierr,locqdie,kind(intg),siz1,siz2,siz3,siz4)
       ! general real
    else if(present(rlg)) then
       if (allocated(rlg)) dsize = size(rlg)
       deallocate(rlg,stat=locierr)
       call proc_dealloc_outpt(filename,procname,arrayname,dsize, &
            'rlg',3,locierr,locqdie,kind(rlg),siz1,siz2,siz3,siz4)
       ! character(len=6)
    else if(present(ch6)) then
       if (allocated(ch6)) dsize = size(ch6)
       deallocate(ch6,stat=locierr)
       call proc_dealloc_outpt(filename,procname,arrayname,dsize, &
            'ch6',3,locierr,locqdie,kind(ch6),siz1,siz2,siz3,siz4)
       ! character(len=8)
    else if(present(ch8)) then
       if (allocated(ch8)) dsize = size(ch8)
       deallocate(ch8,stat=locierr)
       call proc_dealloc_outpt(filename,procname,arrayname,dsize, &
            'ch8',3,locierr,locqdie,kind(ch8),siz1,siz2,siz3,siz4)
       !! chm_real4
    else if(present(cr4)) then
       if (allocated(cr4)) dsize = size(cr4)
       deallocate(cr4,stat=locierr)
       call proc_dealloc_outpt(filename,procname,arrayname,dsize, &
            'cr4',3,locierr,locqdie,kind(cr4),siz1,siz2,siz3,siz4)
       ! chm_int8
    else if(present(ci8)) then
       if (allocated(ci8)) dsize = size(ci8)
       deallocate(ci8,stat=locierr)
       call proc_dealloc_outpt(filename,procname,arrayname,dsize, &
            'ci8',3,locierr,locqdie,kind(ci8),siz1,siz2,siz3,siz4)
       ! logical
    else if(present(log)) then
       if (allocated(log)) dsize = size(log)
       deallocate(log,stat=locierr)
       call proc_dealloc_outpt(filename,procname,arrayname,dsize, &
            'log',3,locierr,locqdie,kind(log),siz1,siz2,siz3,siz4)
       ! chm_cmpx
    else if(present(cmpx)) then
       if (allocated(cmpx)) dsize = size(cmpx)
       deallocate(cmpx,stat=locierr)
       call proc_dealloc_outpt(filename,procname,arrayname,dsize, &
            'cmpx',3,locierr,locqdie,kind(cmpx),siz1,siz2,siz3,siz4)
       !less common types
       ! chm_real8
    else if(present(cr8)) then
       if (allocated(cr8)) dsize = size(cr8)
       deallocate(cr8,stat=locierr)
       call proc_dealloc_outpt(filename,procname,arrayname,dsize, &
            'cr8',3,locierr,locqdie,kind(cr8),siz1,siz2,siz3,siz4)
       ! chm_int4
    else if(present(ci4)) then
       if (allocated(ci4)) dsize = size(ci4)
       deallocate(ci4,stat=locierr)
       call proc_dealloc_outpt(filename,procname,arrayname,dsize, &
            'ci4',3,locierr,locqdie,kind(ci4),siz1,siz2,siz3,siz4)
       ! int_byte
    else if(present(iby)) then
       if (allocated(iby)) dsize = size(iby)
       deallocate(iby,stat=locierr)
       call proc_dealloc_outpt(filename,procname,arrayname,dsize, &
            'iby',3,locierr,locqdie,kind(iby),siz1,siz2,siz3,siz4)
       ! chm_int2
    else if(present(ci2)) then
       if (allocated(ci2)) dsize = size(ci2)
       deallocate(ci2,stat=locierr)
       call proc_dealloc_outpt(filename,procname,arrayname,dsize, &
            'ci2',3,locierr,locqdie,kind(ci2),siz1,siz2,siz3,siz4)
       ! character(len=1)
    else if(present(ch1)) then
       if (allocated(ch1)) dsize = size(ch1)
       deallocate(ch1,stat=locierr)
       call proc_dealloc_outpt(filename,procname,arrayname,dsize, &
            'ch1',3,locierr,locqdie,kind(ch1),siz1,siz2,siz3,siz4)
       ! character(len=2)
    else if(present(ch2)) then
       if (allocated(ch2)) dsize = size(ch2)
       deallocate(ch2,stat=locierr)
       call proc_dealloc_outpt(filename,procname,arrayname,dsize, &
            'ch2',3,locierr,locqdie,kind(ch2),siz1,siz2,siz3,siz4)
       ! character(len=4)
    else if(present(ch4)) then
       if (allocated(ch4)) dsize = size(ch4)
       deallocate(ch4,stat=locierr)
       call proc_dealloc_outpt(filename,procname,arrayname,dsize, &
            'ch4',3,locierr,locqdie,kind(ch4),siz1,siz2,siz3,siz4)
       ! character(len=16)
    else if(present(ch16)) then
       if (allocated(ch16)) dsize = size(ch16)
       deallocate(ch16,stat=locierr)
       call proc_dealloc_outpt(filename,procname,arrayname,dsize, &
            'ch16',3,locierr,locqdie,kind(ch16),siz1,siz2,siz3,siz4)
    else
       ! write warning if no array match
       write(6,*) 'd4d No array matched list of available types.',filename,procname,arrayname
       !        if(qdie) then call wrndie...
    endif
    ! save error status
    if(present(ierr)) then
       ierr = locierr 
    endif
  end SUBROUTINE dealloc_4d
  !______________________________________________________________________
  !______________________________________________________________________
  !
  SUBROUTINE dealloc_vchar(filename,procname,arrayname,siz1, &
       cha,chlen,qdie,ierr)
    !
    !     Special deallocation routine for variable (programmer specified)
    !     length character arrays.  To be used in conjunction with 
    !     alloc_vchar.
    !                                                 --RJP Nov 2008
    implicit none
    integer :: chlen
    integer,intent(in) :: siz1
    character(len=chlen),allocatable,dimension(:) :: cha
    character(len=*),intent(in) :: filename,procname,arrayname
    logical,optional,intent(in) :: qdie
    integer,optional,intent(out) :: ierr
    ! local variables
    logical :: locqdie
    integer :: locierr,dsize
    !***************************************************************
    dsize = 0
    ! test for presence of qdie
    locqdie = .true.
    if(present(qdie)) locqdie = qdie
    ! process allocation
    if (allocated(cha)) dsize = size(cha)
    deallocate(cha,stat=locierr)
    !
    call proc_dealloc_outpt(filename,procname,arrayname,dsize, &
         'vchar',1,locierr,locqdie,kind(cha),siz1)
    !
    ! save error status
    if(present(ierr)) then
       ierr = locierr
    endif
  end SUBROUTINE dealloc_vchar
  !______________________________________________________________________
  !______________________________________________________________________
  !
  SUBROUTINE proc_dealloc_outpt(filename,procname,arrayname, &
       dsize,arrtype,arrank,ierrp,qdiep,arkind,isiz1,isiz2,isiz3,isiz4)
    !
    !     This routine processes data associated with the memory
    !     deallocation for an array
    !
    !                                     --RJP Nov 2008
  use deallocdat
    character(len=*),intent(in) :: filename,procname,arrayname
    character(len=*),intent(in) :: arrtype
    integer,intent(in) :: arrank,ierrp,arkind
    logical,intent(in) :: qdiep 
    integer,intent(in) :: dsize  !determined size
    integer,intent(in) :: isiz1  !intended size
    integer,optional,intent(in) :: isiz2,isiz3,isiz4  !intended size
    integer :: isz2,isz3,isz4
    !
    !     local
    logical :: qdie
    !     qprndeallocf  true if printing deallocations on the fly
    !     qaccumallocdb   true if accumulating database
    !***************************************************************
    !
    isz2 = 1
    isz3 = 1
    isz4 = 1
    if(present(isiz2))isz2 = isiz2
    if(present(isiz3))isz3 = isiz3
    if(present(isiz4))isz4 = isiz4
    qdie = qdiep
    if(qnodealldie) qdie=.false.  !if set globally, don't die
    if (ierrp.gt.0) then
       call deallocerr(filename,procname,arrayname,qdie)
       if (qaccumdeallocdb) call accum_deallfail(filename,procname, &
            arrayname,dsize,arrtype,arrank,arkind,isiz1,isz2,isz3,isz4)
    else 
       if (qaccumdeallocdb) call accum_deallocdb(filename,procname, &
            arrayname,dsize,arrtype,arrank,arkind,isiz1,isz2,isz3,isz4) 
    endif
    if(qprndeallocf) call prndealloc_fly(filename,procname,arrayname, &
         dsize,arrtype,arrank,arkind,ierrp,isiz1,isz2,isz3,isz4)
    !
999 CONTINUE
  end SUBROUTINE proc_dealloc_outpt
  !______________________________________________________________________
  !______________________________________________________________________
  !
  !  ERROR PRINTOUT SUBROUTINES
  SUBROUTINE deallocerr(filename,procname,arrayname,qdie)
    ! printout for deallocation errors      --RJP Nov 2008
    character(len=*),intent(in) :: filename,procname,arrayname
    logical,intent(in) :: qdie
    !***************************************************************
    !
    WRITE(6,'(A25,1X,A10,1X,A6,1X,A10,A,A)') &
         'MEMORY DEALLOCATION ERROR @',arrayname,'SOURCE',filename,' PROCEDURE ',procname
    if(qdie) then
       STOP
       !       call wrndie(-5,
       !     & procname,'array not deallocated')
    endif
  end SUBROUTINE deallocerr
  !______________________________________________________________________
  !______________________________________________________________________
  !
  SUBROUTINE accum_deallocdb(filename,procname,arrayname, &
       dsize,arrtype,arrank,arkind,isiz1,isiz2,isiz3,isiz4)
    !
    !     Adds information for a successfully deallocated array to
    !     the database.  Initially, allocates memory for the db.
    !
    !                                            --RJP Nov 2008
  use deallocdat
    character(len=*),intent(in) :: filename,procname,arrayname
    character(len=*),intent(in) :: arrtype
    integer,intent(in) :: arrank,arkind
    integer,intent(in) :: dsize,isiz1
    integer,optional,intent(in) :: isiz2,isiz3,isiz4
    ! local variables
    integer :: ii,locsize
    !
    !***************************************************************
    !      dealldbarsz = 1000 !this should be set in test command
    if(.not.allocated(darnamar)) then
       write(6,*) 'allocating memory for database'
       allocate(dfilnamar(dealldbarsz),darnamar(dealldbarsz), &
            dartypar(dealldbarsz),dprocnamar(dealldbarsz), &
            darrankar(dealldbarsz), &
            darsizear(dealldbarsz),diarsizear(dealldbarsz), &
            darkindar(dealldbarsz))
       do ii = 1,dealldbarsz
          dfilnamar(ii)=' '
          darnamar(ii)=' '
          dartypar(ii)=' '
          dprocnamar(ii)=' '
          darrankar(ii)=0
          darsizear(ii)=0
          darkindar(ii)=0
       enddo
       ndeallocentr=0
    endif
    !
    locsize=isiz1
    if(present(isiz2)) locsize=locsize*isiz2
    if(present(isiz3)) locsize=locsize*isiz3
    if(present(isiz4)) locsize=locsize*isiz4
    !
    if(ndeallocentr.lt.dealldbarsz) then
       ndeallocentr = ndeallocentr + 1
       dfilnamar(ndeallocentr)=filename
       dprocnamar(ndeallocentr)=procname
       darnamar(ndeallocentr)=arrayname
       dartypar(ndeallocentr)=arrtype
       darrankar(ndeallocentr)=arrank
       darsizear(ndeallocentr)=dsize  !the (actual) overall size
       diarsizear(ndeallocentr)=locsize  !intended (passed) overall size
       darkindar(ndeallocentr)=arkind
    else
       write(6,'(A65)')  &
            'WARNING: # of dealloctd arrys > analysis limit (TEST command)'
    endif
    !
  end SUBROUTINE accum_deallocdb
  !______________________________________________________________________
  !______________________________________________________________________
  !
  SUBROUTINE accum_deallfail(filename,procname,arrayname, &
       dsize,arrtype,arrank,arkind,isiz1,isiz2,isiz3,isiz4)
    !
    !     Adds information for a failed deallocaton attempt to
    !     the database.  Initially, allocates memory for the db.
    !
    !                                                --RJP Nov 2008
  use deallocdat
    character(len=*),intent(in) :: filename,procname,arrayname
    character(len=*),intent(in) :: arrtype
    integer,intent(in) :: arrank,arkind
    integer,intent(in) :: dsize,isiz1
    integer,optional,intent(in) :: isiz2,isiz3,isiz4
    ! local variables
    integer :: ii,locsize
    !
    !***************************************************************
    !      fdealldbarsz = 1000 !this should be set in test command
    if(.not.allocated(fdarnamar)) then
       !        write(6,*) 'allocating memory for database (failed deall)'
       allocate(fdfilnamar(fdealldbarsz),fdarnamar(fdealldbarsz), &
            fdartypar(fdealldbarsz),fdprocnamar(fdealldbarsz), &
            fdarrankar(fdealldbarsz), &
            fdarsizear(fdealldbarsz),fdiarsizear(fdealldbarsz), &
            fdarkindar(fdealldbarsz))
       do ii = 1,fdealldbarsz
          fdfilnamar(ii)=' '
          fdarnamar(ii)=' '
          fdartypar(ii)=' '
          fdprocnamar(ii)=' '
          fdarrankar(ii)=0
          fdarsizear(ii)=0
          fdarkindar(ii)=0
       enddo
       nfdeallocentr = 0
    endif
    !
    locsize=isiz1
    if(present(isiz2)) locsize=locsize*isiz2
    if(present(isiz3)) locsize=locsize*isiz3
    if(present(isiz4)) locsize=locsize*isiz4
    !
    if(nfdeallocentr.lt.fdealldbarsz) then
       nfdeallocentr = nfdeallocentr + 1
       fdfilnamar(nfdeallocentr)=filename
       fdprocnamar(nfdeallocentr)=procname
       fdarnamar(nfdeallocentr)=arrayname
       fdartypar(nfdeallocentr)=arrtype
       fdarrankar(nfdeallocentr)=arrank
       fdarsizear(nfdeallocentr)=dsize  !the (actual) overall size
       fdiarsizear(nfdeallocentr)=locsize  !intended (passed) overall size
       fdarkindar(nfdeallocentr)=arkind !kind value
    else
       write(6,'(A65)')  &
            'WARNING: # of dealloctd arrys > analysis limit (TEST command)'
    endif
  end SUBROUTINE accum_deallfail
  !______________________________________________________________________
  !______________________________________________________________________
  !
  SUBROUTINE prn_deallocdb(qpsucc,qpfail)
    !
    !     Prints accumulated database info for either successful
    !     or failed deallocations (or both).         --RJP Nov 2008
    !
  use deallocdat
    logical,intent(in) :: qpsucc,qpfail
    ! local variables
    integer :: nar
    real(kind=chm_real) :: ssum,issum,msum,misum
    logical :: locqdiep
    !***************************************************************
    ! 
    !print successes
    if(qpsucc) then
       if(.not.(qaccumdeallocdb)) write(6,'(5X,A63)') &
            'WARNING:  NOT CURRENTLY ACCUMULATNG SUCCSSFUL DEALLOC DATABASE'
       if(ndeallocentr.gt.0) then
          write(6,'(5X,A70)') &
               '--------------------------------------------------------------'
          write(6,'(25X,A38)') 'SUCCESSFULLY MEMMOD-DEALLOCATED ARRAYS'
          write(6,'(5X,A70)') &
               '--------------------------------------------------------------'
          write(6, &
               '(9X,A8,1X,A6,5X,A7,1X,A8,2X,A6,3X,A4,1X,A12,5X,A13,2X,A4)') &
               'number','file','process','name', &
               'type','rank','size','intended size','kind'
          ssum = 0
          msum = 0
          issum = 0
          misum = 0
          do nar = 1,ndeallocentr
             msum = msum + darsizear(nar)*darkindar(nar)
             ssum = ssum + darsizear(nar)
             issum = issum + diarsizear(nar)
             misum = misum + diarsizear(nar)*darkindar(nar)
             call prn_deallocdpt('DEALCDB>',nar,dfilnamar(nar), &
                  dprocnamar(nar), &
                  darnamar(nar),dartypar(nar),darrankar(nar),darsizear(nar), &
                  diarsizear(nar),darkindar(nar))
          enddo
          write(6,'(A56,7X,F15.0,1X,F15.0)')  &
               'Totals:',ssum,issum
          write(6,'(14X,A47,2X,F15.0,1X,F15.0)') &
               'Estimated total bytes successfully deallocated:',msum,misum 
       else
          write(6,'(5X,A70)') &
               '--------------------------------------------------------------'
          write(6,'(27X,A34)') 'NO SUCCESSFUL MEMMOD DEALLOCATIONS'
          write(6,'(5X,A70)') &
               '--------------------------------------------------------------'
       endif !if number > 0
    endif !if printing successes
    ! print failures
    if(qpfail) then
       if(.not.(qaccumdeallocdb)) write(6,'(5X,A60)') &
            'WARNING:  NOT CURRENTLY ACCUMULATING FAILED DEALLOC DATABASE'
       if(nfdeallocentr.gt.0) then
          write(6,'(5X,A70)') &
               '--------------------------------------------------------------'
          write(6,'(26X,A35)') 'FAILED MEMMOD DEALLOCATION ATTEMPTS'
          write(6,'(5X,A70)') &
               '--------------------------------------------------------------'
          write(6, &
               '(9X,A8,1X,A6,5X,A7,1X,A8,2X,A6,3X,A4,1X,A12,5X,A13,2X,A4)') &
               'number','file','process','name', &
               'type','rank','size','intended size','kind'
          ssum = 0
          issum = 0
          msum = 0
          misum = 0
          do nar = 1,nfdeallocentr
             ssum = ssum + fdarsizear(nar)
             issum = issum + fdiarsizear(nar)
             msum = msum + fdarsizear(nar)*fdarkindar(nar)
             misum = misum + fdiarsizear(nar)*fdarkindar(nar)
             call prn_deallocdpt('DEAFLDB>',nar,fdfilnamar(nar), &
                  fdprocnamar(nar), &
                  fdarnamar(nar),fdartypar(nar),fdarrankar(nar),fdarsizear(nar), &
                  fdiarsizear(nar),fdarkindar(nar))
          enddo
          write(6,'(A56,7X,F15.0,1X,F15.0)') &
               'Totals:',ssum,issum
          write(6,'(12X,A49,2X,F15.0,1X,F15.0)') &
               'Estimated total bytes unsuccessfully deallocated:',msum, &
               misum
       else
          write(6,'(5X,A70)') &
               '--------------------------------------------------------------'
          write(6,'(29X,A30)') 'NO FAILED MEMMOD DEALLOCATIONS'
          write(6,'(5X,A70)') &
               '--------------------------------------------------------------'
       endif !if number > 0
    endif !if printing failures
    !
  end SUBROUTINE prn_deallocdb
  !______________________________________________________________________
  !______________________________________________________________________
  !
  SUBROUTINE prndealloc_fly(filename,procname,arrayname, &
       dsize,arrtype,arrank,arkind,ierrp,siz1,siz2,siz3,siz4)
    !
    !     Prints information about a deallocation (successful or failed)
    !     immediately after it has occurred--i.e. "on the fly"
    !                                                --RJP Nov 2008

  use deallocdat
    character(len=*),intent(in) :: filename,procname,arrayname
    character(len=*),intent(in) :: arrtype
    integer,intent(in) :: arrank,dsize,siz1,ierrp,arkind
    integer,optional,intent(in) :: siz2,siz3,siz4
    ! local variables
    integer :: locsize
    !***************************************************************
    !      
    locsize=siz1
    if(present(siz2)) locsize=locsize*siz2
    if(present(siz3)) locsize=locsize*siz3
    if(present(siz4)) locsize=locsize*siz4
    if(ierrp.eq.0) then
       write(6,'(A)') 'Successfully Deallocated Array' 
       write(6,'(9X,A8,1X,A6,5X,A7,1X,A8,2X,A6,3X,A4,1X,A12)') &
            'number','file','process','name', &
            'type','rank','size'
       !
       call prn_deallocdpt('DEALCOK>',ndeallocentr,filename,procname, &
            arrayname,arrtype,arrank,dsize,locsize,arkind)
    else
       write(6,'(A)') '****Could NOT Deallocate Array****'
       write(6,'(9X,A8,1X,A6,5X,A7,1X,A8,2X,A6,3X,A4,1X,A12)') &
            'number','file','process','name','type','rank','size'
       call prn_deallocdpt('DEALCFL>',nfdeallocentr,filename,procname, &
            arrayname,arrtype,arrank,dsize,locsize,arkind)
    endif
    if (dsize.ne.locsize) then
       write(6,'(A21,1X,I15,1X,A13,1X,I15,1X,A30)') &
            'DEALC WARNING: actual',dsize,'and intended ',locsize, &
            'szes for deallctn do not match'
    endif
  end SUBROUTINE prndealloc_fly
  !______________________________________________________________________
  !______________________________________________________________________
  SUBROUTINE prn_deallocdpt(label,num,afile,aproc,aname,atype,arank, &
       dsize,asize,akind,int1)
    !
    !     prints a single data point (line) of the allocation database
    !
    character(len=*),intent(in) :: afile,aproc,aname,label
    character(len=*),intent(in) :: atype
    integer,intent(in) :: num,arank,dsize,asize,akind
    integer,optional,intent(in) :: int1  !optional integer
    !
    if(.not.present(int1)) then
       write(6, &
            '(A8,1X,I9,1X,A8,1X,A10,1X,A10,1X,A6,1X,I4,1X,I15,1X, I15,1X,I3)') &
            label,num,afile,aproc,aname,atype,arank,dsize,asize,akind
       !
    else
       write(6, &
            '(A8,1X,I9,1X,A8,1X,A10,1X,A10,1X,A6,1X,I4,1X,I15,1X, I15,1X,I3,1X,I3)') &
            label,num,afile,aproc,aname,atype,arank,dsize,asize,akind,int1
    endif
    !
  END SUBROUTINE prn_deallocdpt
  !      
  !______________________________________________________________________
  !______________________________________________________________________
  SUBROUTINE reset_deallocdb
    !     initializes successful and failed allocation databases
    !
  use deallocdat
    integer :: ii
    ! initialize dealloc arrays
    WRITE(6,*) 'INITIALIZING DEALLOCATION DATABASES'
    if(allocated(dfilnamar)) then
       do ii = 1,dealldbarsz
          dfilnamar(ii)=' '
          darnamar(ii)=' '
          dartypar(ii)=' '
          dprocnamar(ii)=' '
          darrankar(ii)=0
          darsizear(ii)=0
          darkindar(ii)=0
       enddo
    endif
    ndeallocentr=0
    ! initialize fdealloc arrays
    if(allocated(fdfilnamar)) then
       do ii = 1,fdealldbarsz
          fdfilnamar(ii)=' '
          fdarnamar(ii)=' '
          fdartypar(ii)=' '
          fdprocnamar(ii)=' '
          fdarrankar(ii)=0
          fdarsizear(ii)=0
          fdarkindar(ii)=0
       enddo
    endif
    nfdeallocentr=0
    !
  END SUBROUTINE reset_deallocdb

  subroutine deallocate_chm_xyz(fil,routine,var,a,n)
    type(chm_xyz),allocatable,dimension(:) :: a
    character(len=*),intent(in) :: fil,routine,var
    integer,intent(in) :: n
    integer :: alloc_err,n2

    if(allocated(a))n2=size(a)
    deallocate(a,stat=alloc_err)
    
    call proc_dealloc_outpt(fil,routine,var,n,'chm_xyz',1,alloc_err,.false.,-999,n2)
    return
  end subroutine deallocate_chm_xyz

  subroutine chmdealloc_chm_array(fil,routine,var,n,p)
    type(chm_array),dimension(:),allocatable :: p
    character(len=*) :: fil,routine,var
    integer,intent(in) :: n
    integer alloc_err,n2

    if(allocated(p))n2 = size(p)
    deallocate(p,stat=alloc_err)
    call proc_dealloc_outpt(fil,routine,var,n,'chm_array',1,alloc_err,.false.,-999,n2)
  end subroutine chmdealloc_chm_array

  subroutine chmdealloc_chm_iptr(fil,routine,var,n,p)
    type(chm_iarray),dimension(:),allocatable :: p
    character(len=*) :: fil,routine,var
    integer,intent(in) :: n
    integer alloc_err,n2

    if(allocated(p))n2 = size(p)
    deallocate(p,stat=alloc_err)
    call proc_dealloc_outpt(fil,routine,var,n,'chm_array',1,alloc_err,.false.,-999,n2)
  end subroutine chmdealloc_chm_iptr


  subroutine chmdealloc_chm_xyz_mult(fil,routine,var,n1,n2,a)
    integer,intent(in) :: n1,n2
    type(chm_xyz),dimension(n1) :: a
    character(len=*),intent(in) :: fil,routine,var
    integer :: i

    do i=1,n1
       call dealloc_1d(fil,routine,var,n2,crl=a(i)%x)
       call dealloc_1d(fil,routine,var,n2,crl=a(i)%y)
       call dealloc_1d(fil,routine,var,n2,crl=a(i)%z)
    enddo
    return
  end subroutine chmdealloc_chm_xyz_mult
  
end MODULE deallocation

