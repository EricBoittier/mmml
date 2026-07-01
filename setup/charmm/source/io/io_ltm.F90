module io
  use chm_kinds
  use dimens_fcm
  !CHARMM Element source/fcm/io.fcm 1.1
  !
  !  The purpose of this file is to communicate
  !  information between different parts of the
  !  program what was [or was not!] read/written
  !
#if KEY_MMFF==1
  INTEGER,parameter :: NONE=0,SEQUENCE=1,MERCK=2
  INTEGER DATA_READ, DATA_WRITTEN
contains
  subroutine io_init
    data_read=none
    data_written=none
    return
  end subroutine io_init
#endif 
end module io

