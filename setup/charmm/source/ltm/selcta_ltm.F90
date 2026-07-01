module selctam
  use chm_kinds
  use chm_types
  use dimens_fcm
  !CHARMM Element source/fcm/selcta.fcm 1.1
  !
  !   Selection definitions common file.
  !
  !     MAXSKY - Maximun mumber of definitions
  !     MNAMSK - Maximum number of characters per definition
  !
  !     NUMSKY - Number of active definitions
  !     NAMSKY - Name of definitions
  !     LNAMSK - Length of each definition name
  !     PTRSKY - Heap pointer for selection array
  !     LENSKY - Number of atoms in selection array (for checking)
  !
  !
  INTEGER, PARAMETER :: MAXSKY=200,MNAMSK=20
  INTEGER :: NUMSKY
  INTEGER,dimension(maxsky) ::  LENSKY,LNAMSK
  CHARACTER(len=MNAMSK),dimension(maxsky) :: NAMSKY
  type(chm_iptr),dimension(maxsky),save :: PTRSKY

contains

  subroutine select_init()
    numsky=0
    return
  end subroutine select_init

end module selctam

