module exclm
  use chm_kinds
  use dimens_fcm
  character(len=*),private,parameter :: file_name   ="excl_ltm.src"
!CHARMM Element source/fcm/excl.fcm 1.1
!
!    EXTRA EXCLUSIONS (EXCL.FCM)
!
!    PURPOSE:
!
!    TO EXCLUDE USER SPECIFIED NONBONDED INTERACTIONS
!
!    DEBUG  -  DEBUGGING FLAG
!    LEXL   -  TRUE IF 'EXCL' KEYWORD PRESENT IN UPDATE COMMAND
!              NONBOND INTERACTIONS BETWEEN SELECTED ATOMS
!              WILL BE EXCLUDED
!              (NOT ACTIVE - RCZ 1991/05/13)
!    LEXS   -  TRUE IF 'EXSG' KEYWORD PRESENT IN UPDATE COMMAND
!              NONBOND INTERACTIONS BETWEEN SEGMENTS WILL BE EXCLUDED
!    MSEL   -  MULTIPLE SELECTION FLAGS (non active)
!    NEXS   -  NUMBER OF SELECTED SEGMENTS (in update.flx)
!    SLIST  -  LIST OF SELECTED SEGMENTS
!
!    ALL I,J PAIRS FOR EACH MSEL(I)=MSEL(J) GROUPS WILL BE EXCLUDED
!   (IN SUBROUTINE MAKINB / file update.flx) (non active)
!
!     NOTE: NOW(1991/05/13) ONLY EXCLUSION OF INTERSEGMENT
!           NONBONDED ENERGIES IS WORKING
!
      INTEGER     NEXS
      LOGICAL     LEXS
!
      CHARACTER(len=8),allocatable,dimension(:),save :: SLIST
!
contains
  subroutine allocate_excl_ltm()
    use memory
    character(len=*),parameter :: routine_name="allocate_excl_ltm"
    call chmalloc(file_name,routine_name,'slist',maxseg,ch8=slist)
    return
  end subroutine allocate_excl_ltm
end module exclm

