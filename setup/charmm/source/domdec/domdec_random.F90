module domdec_random

#if KEY_DOMDEC_GPU==1
  use chm_kinds
  implicit none
  private

  integer :: n_random_buffer = 0
  real(chm_real4), allocatable, dimension(:) :: random_buffer

  ! public subroutines
  public generate_random_buffer_gpu

  ! public variables
  public random_buffer

contains

  subroutine check_random_buffer(natom)
    use memory
    implicit none
    integer, intent(in) :: natom
    integer :: nbuf

    nbuf = 3 * natom
    nbuf = nbuf + ior(nbuf, 1)

    if (nbuf .gt. n_random_buffer) then
      if (n_random_buffer .gt. 0) then
        call chmdealloc('domdec_random.src','check_random_buffer','random_buffer',n_random_buffer,cr4=random_buffer)
      endif
      call chmalloc('domdec_random.src','check_random_buffer','random_buffer',nbuf,cr4=random_buffer)

      n_random_buffer = nbuf
    endif

    return
  end subroutine check_random_buffer

  subroutine generate_random_buffer_gpu(natom)
    use domdec_util_gpu_mod, only : generate_rn_gpu
    integer, intent(in) :: natom

    call check_random_buffer(natom)
    call generate_rn_gpu(natom, random_buffer)

    return
  end subroutine generate_random_buffer_gpu

#endif /* KEY_DOMDEC_GPU */
end module domdec_random
