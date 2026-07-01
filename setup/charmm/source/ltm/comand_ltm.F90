module comand
  use chm_kinds
  use dimens_fcm
!
!  COMLYN - The current unparsed part of the command line
!  COMLEN - The current length of the unparsed portion
!
!  Note: MXCMSZ may have to be reduced for certain machines (ibm),
!  or to save memory in non virtual machines.
!
      INTEGER,save :: COMLEN,comlen_save
      CHARACTER(len=MXCMSZ),save :: COMLYN,comlyn_save
!
contains

  subroutine sav_comlyn(c,lc)
    integer,intent(in) :: lc
    character(len=mxcmsz),intent(in) :: c

    comlyn_save(1:lc) = c(1:lc)
    comlen_save = lc

    return
  end subroutine sav_comlyn
end module comand

