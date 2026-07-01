module trunk
  use chm_kinds
  use dimens_fcm
  implicit none
  character(len=*),private,parameter :: file_name   ="trunk_ltm.src"
#if KEY_BLOCK==1 /*block*/
!  Table for nbond truncation...
!  IDX: Atom Type (vdW & elec).
!
  integer,PARAMETER :: MAXTAB=1500,MAXPAI=10000
  
  INTEGER PTABLE
  INTEGER,allocatable,dimension(:) :: IDX
  integer :: NDXM
  INTEGER,allocatable,dimension(:) :: NDX  ! (maxpai)
  real(chm_real),allocatable,dimension(:) ::  R02L  ! (maxtab)

contains

  subroutine allocate_trunk(natom)
    use memory
    integer,intent(in) :: natom
    character(len=*),parameter :: routine_name="allocate_trunk"
    if(.not.allocated(idx)) then
       call chmalloc(file_name,routine_name,'idx ',natom,intg=idx)
       call chmalloc(file_name,routine_name,'ndx ',maxpai,intg=ndx)
       call chmalloc(file_name,routine_name,'R02L',maxtab,crl=R02L)
    endif
    return
  end subroutine allocate_trunk

  subroutine deallocate_trunk(natom)
    use memory
    integer,intent(in) :: natom
    character(len=*),parameter :: routine_name="deallocate_trunk"
    if(allocated(idx)) then
       call chmdealloc(file_name,routine_name,'idx ',natom,intg=idx)
       call chmdealloc(file_name,routine_name,'ndx ',maxpai,intg=ndx)
       call chmdealloc(file_name,routine_name,'R02L',maxtab,crl=R02L)
    endif
    return
  end subroutine deallocate_trunk

#endif /* (block)*/
  
end module trunk

