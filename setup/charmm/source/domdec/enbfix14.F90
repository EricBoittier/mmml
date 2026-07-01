module enbfix14

#if KEY_DOMDEC==1 /* domdec */
  use, intrinsic :: iso_c_binding
  use chm_kinds
  use domdec_local_types,only:xyzq_sp_t, xyzq_dp_t
  use domdec_bonded_types,only:list14_t, list14thole_t
  !-----------------------------------------------------------------------
  ! These includes are for the subroutines in enbfix14_fortran.inc
  use number,only:zero, one
  use nblist_types,only:inex14_pair
  use domdec_common,only:simd_version, SIMD_SSE, SIMD_AVX, SIMD_AVX_FMA, &
       SIMD_AVX2, SIMD_AVX2_FMA, q_gpu
  use ewald_1m,only:kappa
  use enbxfast,only:elecmodel,vdwmodel
  use inbnd,only:e14fac
  use psf,only:qdrude
  !use consta,only:ccelec
  !-----------------------------------------------------------------------
  private

  enum, bind(C)
     enumerator :: NONE = 0
     enumerator :: VSH = 1, VSW = 2, VFSW = 3, VGSH = 4, VIPS = 6
     enumerator :: EWALD=4, CSHIFT=5, CFSWIT=6, CSHFT=7, CSWIT=8, RSWIT=9
     enumerator :: RSHFT=10, RSHIFT=11, RFSWIT=12, EIPS=121
  end enum

  ! Public subroutines
#if KEY_LJPME==1
  public ljpme14_ps, ljpme14_pd
#endif
  public ewald14_ps, ewald14_pd, ewald14_soft_ps, ewald14_soft_pd
  public thole14_ps, thole14_pd

  interface

     subroutine enb_fix_14_sse_in14_vsh_ewald(ii_start_in, ii_end_in, xx14list, &
          vdwtype, vdwparam, ron_in, roff_in, e14fac_in, &
          kappa_in, qq_scale_in, xyzq, scoordtab, force, sforce, &
          vdwpot, &
          coulpot) bind (C)
       import
       integer(c_int), intent(in) :: ii_start_in, ii_end_in
       type(list14_t), intent(in) :: xx14list(*)
       integer(c_int), intent(in) :: vdwtype
       real(c_float), intent(in) :: vdwparam(*), ron_in, roff_in, e14fac_in
       real(c_float), intent(in) :: kappa_in, qq_scale_in
       type(xyzq_sp_t), intent(in) :: xyzq
       real(c_float), intent(in) :: scoordtab(*)
       real(c_double), intent(inout) :: force(*), sforce(*)
       real(c_double), intent(inout) :: vdwpot
       real(c_double), intent(inout) :: coulpot
     end subroutine enb_fix_14_sse_in14_vsh_ewald

     subroutine enb_fix_14_sse_in14_vsw_ewald(ii_start_in, ii_end_in, xx14list, &
          vdwtype, vdwparam, ron_in, roff_in, e14fac_in, &
          kappa_in, qq_scale_in, xyzq, scoordtab, force, sforce, &
          vdwpot, &
          coulpot) bind (C)
       import
       integer(c_int), intent(in) :: ii_start_in, ii_end_in
       type(list14_t), intent(in) :: xx14list(*)
       integer(c_int), intent(in) :: vdwtype
       real(c_float), intent(in) :: vdwparam(*), ron_in, roff_in, e14fac_in
       real(c_float), intent(in) :: kappa_in, qq_scale_in
       type(xyzq_sp_t), intent(in) :: xyzq
       real(c_float), intent(in) :: scoordtab(*)
       real(c_double), intent(inout) :: force(*), sforce(*)
       real(c_double), intent(inout) :: vdwpot
       real(c_double), intent(inout) :: coulpot
     end subroutine enb_fix_14_sse_in14_vsw_ewald

     subroutine enb_fix_14_sse_in14_vfsw_ewald(ii_start_in, ii_end_in, xx14list, &
          vdwtype, vdwparam, ron_in, roff_in, e14fac_in, &
          kappa_in, qq_scale_in, xyzq, scoordtab, force, sforce, &
          vdwpot, &
          coulpot) bind (C)
       import
       integer(c_int), intent(in) :: ii_start_in, ii_end_in
       type(list14_t), intent(in) :: xx14list(*)
       integer(c_int), intent(in) :: vdwtype
       real(c_float), intent(in) :: vdwparam(*), ron_in, roff_in, e14fac_in
       real(c_float), intent(in) :: kappa_in, qq_scale_in
       type(xyzq_sp_t), intent(in) :: xyzq
       real(c_float), intent(in) :: scoordtab(*)
       real(c_double), intent(inout) :: force(*), sforce(*)
       real(c_double), intent(inout) :: vdwpot
       real(c_double), intent(inout) :: coulpot
     end subroutine enb_fix_14_sse_in14_vfsw_ewald

     subroutine enb_fix_14_sse_in14_vsh_none(ii_start_in, ii_end_in, xx14list, &
          vdwtype, vdwparam, ron_in, roff_in, e14fac_in, &
          kappa_in, qq_scale_in, xyzq, scoordtab, force, sforce, &
          vdwpot, &
          coulpot) bind (C)
       import
       integer(c_int), intent(in) :: ii_start_in, ii_end_in
       type(list14_t), intent(in) :: xx14list(*)
       integer(c_int), intent(in) :: vdwtype
       real(c_float), intent(in) :: vdwparam(*), ron_in, roff_in, e14fac_in
       real(c_float), intent(in) :: kappa_in, qq_scale_in
       type(xyzq_sp_t), intent(in) :: xyzq
       real(c_float), intent(in) :: scoordtab(*)
       real(c_double), intent(inout) :: force(*), sforce(*)
       real(c_double), intent(inout) :: vdwpot
       real(c_double), intent(inout) :: coulpot
     end subroutine enb_fix_14_sse_in14_vsh_none

     subroutine enb_fix_14_sse_in14_vsw_none(ii_start_in, ii_end_in, xx14list, &
          vdwtype, vdwparam, ron_in, roff_in, e14fac_in, &
          kappa_in, qq_scale_in, xyzq, scoordtab, force, sforce, &
          vdwpot, &
          coulpot) bind (C)
       import
       integer(c_int), intent(in) :: ii_start_in, ii_end_in
       type(list14_t), intent(in) :: xx14list(*)
       integer(c_int), intent(in) :: vdwtype
       real(c_float), intent(in) :: vdwparam(*), ron_in, roff_in, e14fac_in
       real(c_float), intent(in) :: kappa_in, qq_scale_in
       type(xyzq_sp_t), intent(in) :: xyzq
       real(c_float), intent(in) :: scoordtab(*)
       real(c_double), intent(inout) :: force(*), sforce(*)
       real(c_double), intent(inout) :: vdwpot
       real(c_double), intent(inout) :: coulpot
     end subroutine enb_fix_14_sse_in14_vsw_none

     subroutine enb_fix_14_sse_in14_vfsw_none(ii_start_in, ii_end_in, xx14list, &
          vdwtype, vdwparam, ron_in, roff_in, e14fac_in, &
          kappa_in, qq_scale_in, xyzq, scoordtab, force, sforce, &
          vdwpot, &
          coulpot) bind (C)
       import
       integer(c_int), intent(in) :: ii_start_in, ii_end_in
       type(list14_t), intent(in) :: xx14list(*)
       integer(c_int), intent(in) :: vdwtype
       real(c_float), intent(in) :: vdwparam(*), ron_in, roff_in, e14fac_in
       real(c_float), intent(in) :: kappa_in, qq_scale_in
       type(xyzq_sp_t), intent(in) :: xyzq
       real(c_float), intent(in) :: scoordtab(*)
       real(c_double), intent(inout) :: force(*), sforce(*)
       real(c_double), intent(inout) :: vdwpot
       real(c_double), intent(inout) :: coulpot
     end subroutine enb_fix_14_sse_in14_vfsw_none

     subroutine enb_fix_14_sse_in14_none_ewald(ii_start_in, ii_end_in, xx14list, &
          vdwtype, vdwparam, ron_in, roff_in, e14fac_in, &
          kappa_in, qq_scale_in, xyzq, scoordtab, force, sforce, &
          vdwpot, &
          coulpot) bind (C)
       import
       integer(c_int), intent(in) :: ii_start_in, ii_end_in
       type(list14_t), intent(in) :: xx14list(*)
       integer(c_int), intent(in) :: vdwtype
       real(c_float), intent(in) :: vdwparam(*), ron_in, roff_in, e14fac_in
       real(c_float), intent(in) :: kappa_in, qq_scale_in
       type(xyzq_sp_t), intent(in) :: xyzq
       real(c_float), intent(in) :: scoordtab(*)
       real(c_double), intent(inout) :: force(*), sforce(*)
       real(c_double), intent(inout) :: vdwpot
       real(c_double), intent(inout) :: coulpot
     end subroutine enb_fix_14_sse_in14_none_ewald

     subroutine enb_fix_14_sse_ex14_none_ewald(ii_start_in, ii_end_in, xx14list, &
          kappa_in, qq_scale_in, xyzq, scoordtab, force, sforce, &
          coulpot) bind (C)
       import
       integer(c_int), intent(in) :: ii_start_in, ii_end_in
       type(list14_t), intent(in) :: xx14list(*)
       real(c_float), intent(in) :: kappa_in, qq_scale_in
       type(xyzq_sp_t), intent(in) :: xyzq
       real(c_float), intent(in) :: scoordtab(*)
       real(c_double), intent(inout) :: force(*), sforce(*)
       real(c_double), intent(inout) :: coulpot
     end subroutine enb_fix_14_sse_ex14_none_ewald
  end interface

#endif /* domdec */

contains

#if KEY_DOMDEC==1 /* domdec */
  ! *
  ! * Quick and dirty approximation
  ! *
  real(chm_real4) function erfc_approx(x)
    implicit none
    ! Input
    real(chm_real4), intent(in) :: x
    ! Parameters
    real(chm_real4), parameter :: one = 1.0_chm_real4
    real(chm_real4), parameter :: a1 = 0.278393_chm_real4, a2 = 0.230389_chm_real4, &
         a3 = 0.000972_chm_real4, a4 = 0.078108_chm_real4
    ! Variables
    real(chm_real4) x2, x3, x4, sum, sum2, sum4

    x2 = x*x
    x3 = x2*x
    x4 = x3*x

    sum = one + a1*x + a2*x2 + a3*x3 + a4*x4
    sum2 = sum*sum
    sum4 = sum2*sum2
    
    erfc_approx = one - one/sum4

    return
  end function erfc_approx

!------------------ LJPME 1-4 --------------------
#if KEY_LJPME==1

#define LJPME14_CALL
#define DOUBLE_PREC
#define KERNEL_NAME ljpme14_pd
#include "enbfix14_kernel.inc"
#undef LJPME14_CALL
#undef DOUBLE_PREC
#undef KERNEL_NAME

#define LJPME14_CALL
#define SINGLE_PREC
#define KERNEL_NAME ljpme14_ps
#include "enbfix14_kernel.inc"
#undef LJPME14_CALL
#undef SINGLE_PREC
#undef KERNEL_NAME

#define DISPERSION_TERM

#define LOOKUP
#define EX14
#define DOUBLE_PREC
#define KERNEL_NAME enb_fix_14_lookup_ljpme14_pd
#include "enbfix14_kernel.inc"
#undef EX14
#undef DOUBLE_PREC
#undef KERNEL_NAME

#define EX14
#define SINGLE_PREC
#define KERNEL_NAME enb_fix_14_lookup_ljpme14_ps
#include "enbfix14_kernel.inc"
#undef EX14
#undef SINGLE_PREC
#undef KERNEL_NAME

#undef DISPERSION_TERM
#undef LOOKUP

#endif
!----------------End of LJPME 1-4 -----------------

!------------------ Ewald 1-4 --------------------

#define EWALD14_CALL
#define DOUBLE_PREC
#define KERNEL_NAME ewald14_pd
#include "enbfix14_kernel.inc"
#undef EWALD14_CALL
#undef DOUBLE_PREC
#undef KERNEL_NAME

#define EWALD14_CALL
#define SINGLE_PREC
#define KERNEL_NAME ewald14_ps
#include "enbfix14_kernel.inc"
#undef EWALD14_CALL
#undef SINGLE_PREC
#undef KERNEL_NAME

!------------------ Soft Ewald 1-4 --------------------
#define SOFT

#define EWALD14_CALL
#define DOUBLE_PREC
#define KERNEL_NAME ewald14_soft_pd
#include "enbfix14_kernel.inc"
#undef EWALD14_CALL
#undef DOUBLE_PREC
#undef KERNEL_NAME

#define EWALD14_CALL
#define SINGLE_PREC
#define KERNEL_NAME ewald14_soft_ps
#include "enbfix14_kernel.inc"
#undef EWALD14_CALL
#undef SINGLE_PREC
#undef KERNEL_NAME

#undef SOFT
!------------------- thole 1-4 ------------------

#define THOLE14_CALL
#define DOUBLE_PREC
#define KERNEL_NAME thole14_pd
#include "enbfix14_kernel.inc"
#undef THOLE14_CALL
#undef DOUBLE_PREC
#undef KERNEL_NAME

#define THOLE14_CALL
#define SINGLE_PREC
#define KERNEL_NAME thole14_ps
#include "enbfix14_kernel.inc"
#undef THOLE14_CALL
#undef SINGLE_PREC
#undef KERNEL_NAME

! ----------------- lookup ----------------
#define LOOKUP

#define IN14
#define DOUBLE_PREC
#define KERNEL_NAME enb_fix_14_lookup_in14_pd
#include "enbfix14_kernel.inc"
#undef IN14
#undef DOUBLE_PREC
#undef KERNEL_NAME

#define IN14
#define SINGLE_PREC
#define KERNEL_NAME enb_fix_14_lookup_in14_ps
#include "enbfix14_kernel.inc"
#undef IN14
#undef SINGLE_PREC
#undef KERNEL_NAME

#define IN14
#define IPSIN
#define DOUBLE_PREC
#define KERNEL_NAME enb_fix_14_lookup_in14ips_pd
#include "enbfix14_kernel.inc"
#undef IN14
#undef IPSIN
#undef DOUBLE_PREC
#undef KERNEL_NAME

#define IN14
#define IPSIN
#define SINGLE_PREC
#define KERNEL_NAME enb_fix_14_lookup_in14ips_ps
#include "enbfix14_kernel.inc"
#undef IN14
#undef IPSIN
#undef SINGLE_PREC
#undef KERNEL_NAME


#define IN14
#define DOUBLE_PREC
#define KERNEL_NAME enb_fix_14_lookup_ex14ips_pd
#include "enbfix14_kernel.inc"
#undef IN14
#undef DOUBLE_PREC
#undef KERNEL_NAME

#define IN14
#define SINGLE_PREC
#define KERNEL_NAME enb_fix_14_lookup_ex14ips_ps
#include "enbfix14_kernel.inc"
#undef IN14
#undef SINGLE_PREC
#undef KERNEL_NAME

#define EX14
#define DOUBLE_PREC
#define KERNEL_NAME enb_fix_14_lookup_ex14_pd
#include "enbfix14_kernel.inc"
#undef EX14
#undef DOUBLE_PREC
#undef KERNEL_NAME

#define EX14
#define SINGLE_PREC
#define KERNEL_NAME enb_fix_14_lookup_ex14_ps
#include "enbfix14_kernel.inc"
#undef EX14
#undef SINGLE_PREC
#undef KERNEL_NAME

#undef LOOKUP
! ------------------- soft lookup ----------------
#define LOOKUP
#define SOFT

#define IN14
#define DOUBLE_PREC
#define KERNEL_NAME enb_fix_14_lookup_in14_soft_pd
#include "enbfix14_kernel.inc"
#undef IN14
#undef DOUBLE_PREC
#undef KERNEL_NAME

#define IN14
#define SINGLE_PREC
#define KERNEL_NAME enb_fix_14_lookup_in14_soft_ps
#include "enbfix14_kernel.inc"
#undef IN14
#undef SINGLE_PREC
#undef KERNEL_NAME

#undef SOFT
#undef LOOKUP
! ------------------- no-lookup (math eval) ----------------

#define MATH_EVAL

#define IN14
#define VDW_VSH
#define DOUBLE_PREC
#define KERNEL_NAME enb_fix_14_in14_vsh_none_pd
#include "enbfix14_kernel.inc"
#undef IN14
#undef VDW_VSH
#undef DOUBLE_PREC
#undef KERNEL_NAME

#define IN14
#define VDW_VSW
#define DOUBLE_PREC
#define KERNEL_NAME enb_fix_14_in14_vsw_none_pd
#include "enbfix14_kernel.inc"
#undef IN14
#undef VDW_VSW
#undef DOUBLE_PREC
#undef KERNEL_NAME

#define IN14
#define VDW_VFSW
#define DOUBLE_PREC
#define KERNEL_NAME enb_fix_14_in14_vfsw_none_pd
#include "enbfix14_kernel.inc"
#undef IN14
#undef VDW_VFSW
#undef DOUBLE_PREC
#undef KERNEL_NAME

#define IN14
#define VDW_NONE
#define ELEC_EWALD
#define DOUBLE_PREC
#define KERNEL_NAME enb_fix_14_in14_none_ewald_pd
#include "enbfix14_kernel.inc"
#undef IN14
#undef VDW_NONE
#undef ELEC_EWALD
#undef DOUBLE_PREC
#undef KERNEL_NAME

#define IN14
#define VDW_VSH
#define ELEC_EWALD
#define DOUBLE_PREC
#define KERNEL_NAME enb_fix_14_in14_vsh_ewald_pd
#include "enbfix14_kernel.inc"
#undef IN14
#undef VDW_VSH
#undef ELEC_EWALD
#undef DOUBLE_PREC
#undef KERNEL_NAME

#define IN14
#define VDW_VSW
#define ELEC_EWALD
#define DOUBLE_PREC
#define KERNEL_NAME enb_fix_14_in14_vsw_ewald_pd
#include "enbfix14_kernel.inc"
#undef IN14
#undef VDW_VSW
#undef ELEC_EWALD
#undef DOUBLE_PREC
#undef KERNEL_NAME

#define IN14
#define VDW_VFSW
#define ELEC_EWALD
#define DOUBLE_PREC
#define KERNEL_NAME enb_fix_14_in14_vfsw_ewald_pd
#include "enbfix14_kernel.inc"
#undef IN14
#undef VDW_VFSW
#undef ELEC_EWALD
#undef DOUBLE_PREC
#undef KERNEL_NAME

! ----- Single precision -------

#define IN14
#define VDW_VSH
#define SINGLE_PREC
#define KERNEL_NAME enb_fix_14_in14_vsh_none_ps
#include "enbfix14_kernel.inc"
#undef IN14
#undef VDW_VSH
#undef SINGLE_PREC
#undef KERNEL_NAME

#define IN14
#define VDW_VSW
#define SINGLE_PREC
#define KERNEL_NAME enb_fix_14_in14_vsw_none_ps
#include "enbfix14_kernel.inc"
#undef IN14
#undef VDW_VSW
#undef SINGLE_PREC
#undef KERNEL_NAME

#define IN14
#define VDW_VFSW
#define SINGLE_PREC
#define KERNEL_NAME enb_fix_14_in14_vfsw_none_ps
#include "enbfix14_kernel.inc"
#undef IN14
#undef VDW_VFSW
#undef SINGLE_PREC
#undef KERNEL_NAME

#define IN14
#define VDW_NONE
#define ELEC_EWALD
#define SINGLE_PREC
#define KERNEL_NAME enb_fix_14_in14_none_ewald_ps
#include "enbfix14_kernel.inc"
#undef IN14
#undef VDW_NONE
#undef ELEC_EWALD
#undef SINGLE_PREC
#undef KERNEL_NAME

#define IN14
#define VDW_VSH
#define ELEC_EWALD
#define SINGLE_PREC
#define KERNEL_NAME enb_fix_14_in14_vsh_ewald_ps
#include "enbfix14_kernel.inc"
#undef IN14
#undef VDW_VSH
#undef ELEC_EWALD
#undef SINGLE_PREC
#undef KERNEL_NAME

#define IN14
#define VDW_VSW
#define ELEC_EWALD
#define SINGLE_PREC
#define KERNEL_NAME enb_fix_14_in14_vsw_ewald_ps
#include "enbfix14_kernel.inc"
#undef IN14
#undef VDW_VSW
#undef ELEC_EWALD
#undef SINGLE_PREC
#undef KERNEL_NAME

#define IN14
#define VDW_VFSW
#define ELEC_EWALD
#define SINGLE_PREC
#define KERNEL_NAME enb_fix_14_in14_vfsw_ewald_ps
#include "enbfix14_kernel.inc"
#undef IN14
#undef VDW_VFSW
#undef ELEC_EWALD
#undef SINGLE_PREC
#undef KERNEL_NAME

!---------------- 1-4 exclusions --------------

#define EX14
#define VDW_NONE
#define ELEC_EWALD
#define DOUBLE_PREC
#define KERNEL_NAME enb_fix_14_ex14_ewald_pd
#include "enbfix14_kernel.inc"
#undef EX14 
#undef VDW_NONE
#undef ELEC_EWALD
#undef DOUBLE_PREC
#undef KERNEL_NAME

#define EX14
#define VDW_NONE
#define ELEC_EWALD
#define SINGLE_PREC
#define KERNEL_NAME enb_fix_14_ex14_ewald_ps
#include "enbfix14_kernel.inc"
#undef EX14
#undef VDW_NONE
#undef ELEC_EWALD
#undef SINGLE_PREC
#undef KERNEL_NAME

#define EX14
#define VDW_NONE
#define ELEC_EWALD
#define SMALLR
#define DOUBLE_PREC
#define KERNEL_NAME enb_fix_14_ex14_ewald_smallr_pd
#include "enbfix14_kernel.inc"
#undef EX14
#undef VDW_NONE
#undef ELEC_EWALD
#undef SMALLR
#undef DOUBLE_PREC
#undef KERNEL_NAME

#define EX14
#define VDW_NONE
#define ELEC_EWALD
#define SMALLR
#define SINGLE_PREC
#define KERNEL_NAME enb_fix_14_ex14_ewald_smallr_ps
#include "enbfix14_kernel.inc"
#undef EX14
#undef VDW_NONE
#undef ELEC_EWALD
#undef SMALLR
#undef SINGLE_PREC
#undef KERNEL_NAME

#undef MATH_EVAL

#endif /* domdec */

end module enbfix14
