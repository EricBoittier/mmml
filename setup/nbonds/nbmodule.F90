module nb_module
  use chm_kinds
  implicit none
!
!     CCNBA - R**-12 vdw coefficients
!     CCNBB - R**-6  vdw coefficients
!     CCNBC - R**6   vdw coefficients (VSHIFT only)
!     CCNBD - R**0   vdw coefficients (VSHIFT only)
!     LCCNBA - real(chm_real) length of CCNBA and CCNBB
!     LCCNBD - real(chm_real) length of CCNBC and CCNBD (VSHIFT only)
!
  real(chm_real),allocatable,dimension(:),save ::  &
       CCNBA, CCNBB, CCNBC, CCNBD
  integer,save :: LCCNBA, LCCNBD
end module nb_module

