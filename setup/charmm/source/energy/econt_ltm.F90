module econtmod
  use chm_kinds
  use dimens_fcm
  implicit none
  character(len=*),private,parameter :: file_name   ="econt_ltm.src"
!CHARMM Element source/fcm/econt.fcm 1.1
!
!  econt.fcm -  Energy contributions common block.
!
!  QECONT   - Flag to determine if partition analysis is on.
!  ECONT(*) - Partial energy contribution for each atom
!
!  QATERM   - flag to indicate energy term selection is used
!  ANSLCT(*)- Atom Selection for printing energy terms
!  
!  QAONLY   - Flag indicating that only energy terms where ALL
!                   atoms are selected be listed
!  QANBND   - Flag to print nonbond energy term table
!  QAUNIT   - Output unit number for printing data
!
!

      LOGICAL QECONT,QAONLY,QATERM,QANBND
      INTEGER,allocatable,dimension(:) :: ANSLCT
      integer :: QAUNIT
      real(chm_real),allocatable,dimension(:) :: ECONT

contains
  subroutine allocate_econt()
    use memory
    character(len=*),parameter :: routine_name="allocate_econt"
    call chmalloc(file_name,routine_name,'anslct ',maxaim,intg=anslct)
    call chmalloc(file_name,routine_name,'econt  ',maxaim, crl=econt)
    return
  end subroutine allocate_econt

  subroutine deallocate_econt()
    use memory
    character(len=*),parameter :: routine_name="deallocate_econt"
    call chmdealloc(file_name,routine_name,'anslct ',maxaim,intg=anslct)
    call chmdealloc(file_name,routine_name,'econt  ',maxaim, crl=econt)
    return
  end subroutine deallocate_econt

end module econtmod

