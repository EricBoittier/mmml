module replica_ltm
  implicit none

  logical :: qRep
#if KEY_REPLICA==1
  integer :: nRepl
#endif 

contains
  subroutine replica_iniall
    qrep=.false.
    return
  end subroutine replica_iniall

end module replica_ltm

