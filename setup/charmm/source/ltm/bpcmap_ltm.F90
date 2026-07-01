module bpcmap_mod
  use chm_kinds
 
     logical ::   QBPC = .FALSE.
     integer  ::  nbtp = 0
     integer,allocatable,dimension(:)         ::  BMAP_DIM, BMAP_UNT
     integer,allocatable,dimension(:,:)       ::  BMAP_IND
     real(chm_real),allocatable,dimension(:)  ::  BMAP_LAM

end module bpcmap_mod
