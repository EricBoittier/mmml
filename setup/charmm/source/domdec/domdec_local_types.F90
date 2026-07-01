module domdec_local_types
#if KEY_DOMDEC==1
  use chm_kinds
  use, intrinsic :: iso_c_binding
  implicit none
  public

  ! Double precision coordinate + charge (with q=w matches double4 in CUDA)
  type, bind(c) :: xyzq_dp_t
     real(c_double) x, y, z, q
#if KEY_LJPME==1
     real(c_double) c6
#endif
  end type xyzq_dp_t

  ! Single precision coordinate + charge (with q=w matches float4 in CUDA)
  type, bind(c) :: xyzq_sp_t
     real(c_float) x, y, z, q
#if KEY_LJPME==1
     real(c_float) c6
#endif
  end type xyzq_sp_t

  ! Definition of double/single precision arrays:
  ! dp = double precision
  ! sp = single precision
  type xyzq_dpsp_t
     type(xyzq_dp_t), pointer, dimension(:) :: dp
     type(xyzq_sp_t), pointer, dimension(:) :: sp
  end type xyzq_dpsp_t

  type crlarray_t
     real(chm_real), allocatable, dimension(:) :: array
  end type crlarray_t

#endif 
end module domdec_local_types

