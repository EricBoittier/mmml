module domdec_gpu_mod
#if KEY_DOMDEC_GPU==1 /*domdec_gpu*/
  use, intrinsic :: iso_c_binding
  implicit none
  public

  interface
     subroutine is_device_on(q_device_on) bind(C,name="is_device_on")
       import
       logical(c_int), intent(out) :: q_device_on
     end subroutine is_device_on

     subroutine set_q_test_gpu(h_q_test) bind(C,name="set_q_test_gpu")
       import
       logical(c_int), intent(in) :: h_q_test
     end subroutine set_q_test_gpu

     subroutine stop_device() bind(C,name="stop_device")
     end subroutine stop_device

     subroutine start_device(prnlev, mynod, numnod, gpuid) &
          bind(C,name="start_device")
       import
       integer(c_int), intent(in) :: prnlev, mynod, numnod, gpuid
     end subroutine start_device

     subroutine start_mdsim(ndirect, recip_on, bonded_on, nblist_on, q_direct_node, q_recip_node, &
          nx, ny, nz, ncoord, iblo14, inb14, nfftx, nffty, nfftz, forder, kappa, nblock, use_softcore, use_pmel, seed) bind(C,name="start_mdsim")
       import
       integer(c_int), intent(in) :: ndirect
       logical(c_int), intent(in) :: recip_on, bonded_on, nblist_on, q_direct_node, q_recip_node
       integer(c_int), intent(in) :: nx, ny, nz, ncoord, iblo14(*), inb14(*), nfftx, nffty, nfftz, forder
       real(c_double), intent(in) :: kappa
       integer(c_int), intent(in) :: nblock
       integer(c_int), intent(in) :: use_softcore
       integer(c_int), intent(in) :: use_pmel
       integer(c_int), intent(in) :: seed
     end subroutine start_mdsim

     subroutine stop_mdsim() bind(C,name="stop_mdsim")
     end subroutine stop_mdsim

     subroutine alloc_gpu_pinned_float4(cptr, n) bind(C,name="alloc_gpu_pinned_float4")
       import
       type(c_ptr), intent(out) :: cptr
       integer(c_int), intent(in) :: n
     end subroutine alloc_gpu_pinned_float4

     subroutine dealloc_gpu_pinned_float4(cptr) bind(C,name="dealloc_gpu_pinned_float4")
       import
       type(c_ptr), intent(inout) :: cptr
     end subroutine dealloc_gpu_pinned_float4

     subroutine alloc_gpu_pinned_int(cptr, n) bind(C,name="alloc_gpu_pinned_int")
       import
       type(c_ptr), intent(out) :: cptr
       integer(c_int), intent(in) :: n
     end subroutine alloc_gpu_pinned_int

     subroutine dealloc_gpu_pinned_int(cptr) bind(C,name="dealloc_gpu_pinned_int")
       import
       type(c_ptr), intent(inout) :: cptr
     end subroutine dealloc_gpu_pinned_int

     subroutine alloc_gpu_pinned_double(cptr, n) bind(C,name="alloc_gpu_pinned_double")
       import
       type(c_ptr), intent(out) :: cptr
       integer(c_int), intent(in) :: n
     end subroutine alloc_gpu_pinned_double

     subroutine dealloc_gpu_pinned_double(cptr) bind(C,name="dealloc_gpu_pinned_double")
       import
       type(c_ptr), intent(inout) :: cptr
     end subroutine dealloc_gpu_pinned_double

     subroutine alloc_gpu_pinned_float(cptr, n) bind(C,name="alloc_gpu_pinned_float")
       import
       type(c_ptr), intent(out) :: cptr
       integer(c_int), intent(in) :: n
     end subroutine alloc_gpu_pinned_float

     subroutine dealloc_gpu_pinned_float(cptr) bind(C,name="dealloc_gpu_pinned_float")
       import
       type(c_ptr), intent(inout) :: cptr
     end subroutine dealloc_gpu_pinned_float

     subroutine alloc_gpu_pinned_ientry(cptr, n) bind(C,name="alloc_gpu_pinned_ientry")
       import
       type(c_ptr), intent(out) :: cptr
       integer(c_int), intent(in) :: n
     end subroutine alloc_gpu_pinned_ientry

     subroutine dealloc_gpu_pinned_ientry(cptr) bind(C,name="dealloc_gpu_pinned_ientry")
       import
       type(c_ptr), intent(inout) :: cptr
     end subroutine dealloc_gpu_pinned_ientry

     subroutine alloc_gpu_pinned_tile_excl(cptr, n) bind(C,name="alloc_gpu_pinned_tile_excl")
       import
       type(c_ptr), intent(out) :: cptr
       integer(c_int), intent(in) :: n
     end subroutine alloc_gpu_pinned_tile_excl

     subroutine dealloc_gpu_pinned_tile_excl(cptr) bind(C,name="dealloc_gpu_pinned_tile_excl")
       import
       type(c_ptr), intent(inout) :: cptr
     end subroutine dealloc_gpu_pinned_tile_excl

     subroutine alloc_gpu_pinned_char(cptr, n) bind(C,name="alloc_gpu_pinned_char")
       import
       type(c_ptr), intent(out) :: cptr
       integer(c_int), intent(in) :: n
     end subroutine alloc_gpu_pinned_char

     subroutine dealloc_gpu_pinned_char(cptr) bind(C,name="dealloc_gpu_pinned_char")
       import
       type(c_ptr), intent(inout) :: cptr
     end subroutine dealloc_gpu_pinned_char

  end interface
#endif /* (domdec_gpu)*/
end module domdec_gpu_mod


