!> C-callable echo of a pyCHARMM api call into the CHARMM output file.
!
! pyCHARMM functions that call a Fortran C-API entry point directly
! bypass the CHARMM script interpreter, so nothing is written to the
! CHARMM output to show that the command ran.  When api echo is enabled
! (see pycharmm.trace.set_api_echo), the Python layer calls api_trace
! with a short label for the call, and this writes it to OUTU so it
! appears in the CHARMM output file alongside the interpreter's own
! command echoes.  This is a debugging aid for pyCHARMM scripts: it lets
! you confirm, from the CHARMM output, that a given command was issued.
module api_trace_mod
  implicit none
contains

  !> Write one api-call label to the CHARMM output unit.
  !  c_msg is the label (need not be NUL terminated), n its length.
  !  Honours PRNLEV: nothing is written when output is suppressed
  !  (PRNLEV < 0).  The decision to call this at all is made on the
  !  Python side, so a normal run that has not enabled echo never gets
  !  here.
  subroutine api_trace(c_msg, n) bind(c)
    use, intrinsic :: iso_c_binding, only: c_char, c_int
    use api_util, only: c2f_string
    use stream, only: outu, prnlev
    implicit none
    character(kind=c_char, len=1) :: c_msg(*)
    integer(c_int), value, intent(in) :: n

    if (n <= 0) return
    if (prnlev < 0) return
    write(outu, '(2a)') ' PYCHARMM>  ', trim(c2f_string(c_msg, n))
  end subroutine api_trace

end module api_trace_mod
