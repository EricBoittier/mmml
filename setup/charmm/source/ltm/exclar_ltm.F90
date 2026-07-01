module mexclar
  use chm_kinds
  use dimens_fcm
!
!    EXCLUSION ARRAYS
!     These are two-dimensional tables of the exclusions. 
!     Each 2D table is stored in the form of a partially 
!     compressed 1D array.
!      --RJPetrella, 12.14.98
!
!  ATOM exclusion table array:
!
!  EXCDIM  the total number of elements in the exclusion array for atoms
!  GEXCDIM the total number of elements in the group exclusion array

!  HPAEXL      Heap pointer for atom exclusion list
!  HPGEXL      Heap pointer for group exclusion list
!  HPAEXP      Heap pointer for atom exclusion table base array
!  HPGEXP      Heap pointer for group exclusion table base array
!  HPNEXA,HPNEXG Heap pointers for arrays containing number of 
!                 exclusions per atom
!
!  QEXCTB      True if exclusion table has been made
!                                 --RJ Petrella
!
  integer,allocatable,dimension(:) :: HPAEXL,HPGEXL,HPAEXP,HPGEXP,HPNEXA,HPNEXG
      INTEGER EXCDIM,GEXCDIM
      LOGICAL QEXCTB
! 
end module mexclar

