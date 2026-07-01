#ifndef NOCUDAC
module dummy_module
  ! APH 2014
  ! Empty dummy module. This is needed because when compiling without domdec_gpu, the compiler would
  ! come up with an empty archive file, which then cannot be linked
end module dummy_module
#endif /* NOCUDAC */
