module lupcom
  use chm_kinds
  use dimens_fcm

!
!     The LUP constraints communication
!
!     Purpose:
!     Passing LUP constraint info to minimization routines
!
!     I/O: lupopt.src
!
!     Variable  Purpose
!
!     QLUPCS         Flag set if LUP constraints are implemented
!     NLUPC          number of LUP constraints (default is 7)
!
      INTEGER NLUPC
      LOGICAL QLUPCS
!
contains
  subroutine lupcom_init
    qlupcs=.false.
    nlupc=0
    return
  end subroutine lupcom_init
end module lupcom

