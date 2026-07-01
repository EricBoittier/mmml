module travel
  use chm_kinds
  use dimens_fcm
  implicit none
  character(len=*),private,parameter :: file_name   ="travel_ltm.src"
  !CHARMM Element source/fcm/travel.fcm 1.1

  ! Common-block for interfacing CHARMM with TReK version 2.10
  ! By Stefan Fischer; July 5-2003.
  !
  ! This code is NOT compatible with TRAVEL.
  ! TRAVEL file-names used for compatibility with previous versions of CHARMM.
  !
  ! Include in  .source/main/iniall.src

#if KEY_TRAVEL==1 /*travel_fcm*/

  LOGICAL   QTRAV,LSCAL

  real(chm_real),dimension(:),allocatable :: XSCAL,YSCAL,ZSCAL

contains

  subroutine travel_iniall()
    qtrav=.false.
    return
  end subroutine travel_iniall

  subroutine allocate_travel_ltm(natom)
    use memory
    character(len=*),parameter :: routine_name="allocate_travel_ltm"
    integer :: natom
    call chmalloc(file_name,routine_name,'xscal ',natom,crl=xscal)
    call chmalloc(file_name,routine_name,'yscal ',natom,crl=yscal)
    call chmalloc(file_name,routine_name,'zscal ',natom,crl=zscal)

    return
  end subroutine allocate_travel_ltm

  subroutine deallocate_travel_ltm(natom)
    use memory
    character(len=*),parameter :: routine_name="deallocate_travel_ltm"
    integer :: natom
    call chmdealloc(file_name,routine_name,'xscal ',natom,crl=xscal)
    call chmdealloc(file_name,routine_name,'yscal ',natom,crl=yscal)
    call chmdealloc(file_name,routine_name,'zscal ',natom,crl=zscal)

    return
  end subroutine deallocate_travel_ltm

#endif /* (travel_fcm)*/
  !
end module travel

