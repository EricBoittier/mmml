module ctitla
  use chm_kinds
  !      Titles Common File
  !
  !     TITLEA is to be used for writing data files
  !     TITLEB is to be used for reading data files
  !
  !     NTITLA - The number of active lines in TITLEA
  !     NTITLB - The number of active lines in TITLEB
  !
  integer,parameter :: maxtit=32
  integer ntitla,ntitlb
  character(len=80) :: titlea(maxtit),titleb(maxtit)
contains
  subroutine ctitla_init
    ntitlb=0
    return
  end subroutine ctitla_init
end module ctitla

