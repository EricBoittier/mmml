!
!     The following two subroutines have to be common for all
!     Then this  ##IF works OK
#if KEY_MPI==1
INTEGER FUNCTION NNODS()
  !-----------------------------------------------------------------------
  !     Becuse we call exec_stack or pvmini at the beginning everything
  !     is already defined in parallel.f90.
  use chm_kinds
  !
  use parallel
  implicit none
  !
  NNODS = NUMNOD
  RETURN
END FUNCTION NNODS
!
INTEGER FUNCTION MNODS()
  !-----------------------------------------------------------------------
  !     Becuse we call exec_stack or pvmini at the beginning everything
  !     is already defined!
  use chm_kinds
  !
  use parallel
  implicit none
  MNODS=MYNOD
  RETURN
END FUNCTION MNODS
#else /**/
SUBROUTINE NULL_SK
  RETURN
!
! Flex does not allow any lines after the 'end' statement (not even blanks)
!
END SUBROUTINE NULL_SK
#endif 

