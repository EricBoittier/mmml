module varcutm
  use chm_kinds
  use dimens_fcm
  implicit none
!
!     Variable cutoffs of LJ interaction depending on atom types
!
!     Purpose:
!
!     To store the list of vdw radii augmentations, which is dependent
!     on atom type and set using the 'scalar var' command, that
!     is used in allowing variable cutoffs of the LJ interactions.
!
!     Notes:    The index entry gives the domain of the data.
!
!     Variable  Index    Purpose
!
!     VARCUT    Atom     variable non-bond cut augmentation array
!
  character(len=*),private,parameter :: file_name = 'varcut_ltm.src'
  logical, save :: qvarcut = .false.
  real(chm_real),allocatable,dimension(:) :: varcut

contains

  subroutine setup_varcut(comlyn,comlen,natom)
    use string
    use stream
    use number
    use memory

    character(len=*),intent(inout) :: comlyn
    integer,intent(inout) :: comlen, natom

    character(len=4) :: wrd
    character(len=*),parameter :: routine_name = 'setup_varcut', &
         msg_prefix = '      VARCUT: ', &
         feature_name = 'Variable cutoff L-J'

    wrd = nexta4(comlyn,comlen)

    if (wrd == 'ON') then
       qvarcut = .true.
    else if (wrd == 'OFF') then
       qvarcut = .false.
    else
       if (prnlev >= 2) write (outu, '(2A)') msg_prefix, 'ON or OFF expected'
       return
    endif

    if (qvarcut .eqv. allocated(varcut)) then
       if (prnlev >= 2) write (outu, '(4A)') msg_prefix, feature_name, ' is already ', wrd
       return
    endif

    if (qvarcut) then
       call chmalloc(file_name,routine_name,'varcut',natom,crl=varcut)
       varcut = ZERO
       if (prnlev >= 2) write (outu,'(3A)') msg_prefix, feature_name, &
            ' is ON, setup with scalar commands'
    else
       call chmdealloc(file_name,routine_name,'varcut',natom,crl=varcut)
       if (prnlev >= 2) write (outu,'(3A)') msg_prefix, feature_name, &
            ' is OFF, cutoff array deallocated'
    endif

    return
  end subroutine setup_varcut

  subroutine get_varcut(buf, natom)
    use number
    real(chm_real), intent(out) :: buf(:)
    integer, intent(in) :: natom

    if (allocated(varcut)) then
       buf(1:natom) = varcut(1:natom)
    else
       buf(1:natom) = ZERO
    endif
  end subroutine get_varcut

  subroutine set_varcut(buf, natom)
    use memory
    real(chm_real), intent(in) :: buf(:)
    integer, intent(in) :: natom
    integer :: n

    if (.not. allocated(varcut)) then
       call chmalloc(file_name,'set_varcut','varcut',natom,crl=varcut)
       qvarcut = .true.
    else if (natom > size(varcut)) then
       call chmrealloc(file_name,'set_varcut','varcut',natom,crl=varcut)
    endif
    varcut(1:natom) = buf(1:natom)
  end subroutine set_varcut

end module varcutm

