module ztypes
use chm_types
implicit none

 type confset
   integer,allocatable,dimension(:) :: LODOF,HIDOF,LOCNF,HICNF,MSTDF
   real(chm_real),allocatable,dimension(:) :: MSTDV, ENERG
 end type confset
 
end module ztypes
