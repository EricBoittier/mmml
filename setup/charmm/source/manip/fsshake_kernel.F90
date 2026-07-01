module fsshake_kernel
  use chm_kinds
  use number, only: one  ! for fsshake_kernel.inc dependency
  implicit none
  private

  ! Kernel routines for fsshake.src

  ! Public subroutines
  public settle_water_kernel_d0, fsshakph_kernel2_d0, fsshakph_kernel3_d0, fsshakph_kernel4_d0
#if KEY_DOMDEC == 1
  public settle_water_kernel_d1, fsshakph_kernel2_d1, fsshakph_kernel3_d1, fsshakph_kernel4_d1
#endif  /* KEY_DOMDEC == 1 */

contains

#define FSSHAKE_SUFFIX D0
#undef FSSHAKE_DD
#include "fsshake_kernel.inc"
#undef FSSHAKE_SUFFIX
  
#if KEY_DOMDEC == 1
#define FSSHAKE_SUFFIX D1
#define FSSHAKE_DD 1
#include "fsshake_kernel.inc"
#undef FSSHAKE_DD
#undef FSSHAKE_SUFFIX
#endif  /* KEY_DOMDEC == 1 */

end module fsshake_kernel
