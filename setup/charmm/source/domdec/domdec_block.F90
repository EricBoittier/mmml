module domdec_block
  ! *
  ! * Storage module for DOMDEC BLOCK arrays with their init/uninit subroutines
  ! * Antti-Pekka Hynninen, Jan 2015.
  ! *
#if KEY_DOMDEC==1
#if KEY_BLOCK==1
  use chm_kinds
  use dimens_fcm
  implicit none
  private

  real(chm_real), allocatable, dimension(:) :: biflam_loc, biflam2_loc
#if KEY_DOMDEC_GPU==1
  integer, allocatable, dimension(:) :: blocktype
  real(chm_real4), allocatable :: bixlam_sp(:), fullblcoep_sp(:,:)
#endif
  
  ! Public variables
  public biflam_loc, biflam2_loc

  ! Public subroutines
  public init_block, uninit_block
#if KEY_DOMDEC_GPU==1
  public set_blocktype_to_gpu, set_block_params_to_gpu, combine_biflam_from_gpu
#endif
  
#endif
#endif
contains

#if KEY_DOMDEC==1
#if KEY_BLOCK==1
  ! *
  ! * (Re-)Allocate biflam_loc and biflam2_loc if needed.
  ! *
  subroutine init_block()
    use memory,only:chmalloc, chmdealloc
    use block_ltm,only:nblock
    implicit none

    ! biflam_loc
    if (allocated(biflam_loc)) then
       if (size(biflam_loc) /= nblock) then
          call chmdealloc('domdec_block.src','init_block','biflam_loc',size(biflam_loc),&
               crl=biflam_loc)
       endif
    endif
    if (.not.allocated(biflam_loc)) then
       call chmalloc('domdec_block.src','init_block','biflam_loc',nblock,crl=biflam_loc)
    endif

    ! biflam2_loc
    if (allocated(biflam2_loc)) then
       if (size(biflam2_loc) /= nblock) then
          call chmdealloc('domdec_block.src','init_block','biflam2_loc',size(biflam2_loc),&
               crl=biflam2_loc)
       endif
    endif
    if (.not.allocated(biflam2_loc)) then
       call chmalloc('domdec_block.src','init_block','biflam2_loc',nblock,crl=biflam2_loc)
    endif

    return
  end subroutine init_block

  ! *
  ! * De-allocate biflam_loc and biflam2_loc.
  ! *
  subroutine uninit_block()
    use memory,only:chmdealloc
    implicit none

    ! biflam_loc
    if (allocated(biflam_loc)) then
       call chmdealloc('domdec_block.src','uninit_block','biflam_loc',size(biflam_loc),&
            crl=biflam_loc)
    endif

    ! biflam2_loc
    if (allocated(biflam2_loc)) then
       call chmdealloc('domdec_block.src','uninit_block','biflam2_loc',size(biflam2_loc),&
            crl=biflam2_loc)
    endif

#if KEY_DOMDEC_GPU==1    
    ! blocktype
    if (allocated(blocktype)) then
       call chmdealloc('domdec_block.src','uninit_block','blocktype',size(blocktype),intg=blocktype)
    endif

    ! bixlam_sp
    if (allocated(bixlam_sp)) then
       call chmdealloc('domdec_block.src','uninit_block','bixlam_sp',size(bixlam_sp),cr4=bixlam_sp)
    endif

    ! fullblcoep_sp
    if (allocated(fullblcoep_sp)) then
       call chmdealloc('domdec_block.src','uninit_block','fullblcoep_sp',size(fullblcoep_sp,1),&
            size(fullblcoep_sp,2),cr4=fullblcoep_sp)
    endif
#endif
    
    return
  end subroutine uninit_block

#if KEY_DOMDEC_GPU==1
  ! *
  ! * Setup blocktype for GPU: blocktype[i] = iblckp[loc2glo_ind[i]] - 1
  ! * NOTE: this is done at every neighborlist build
  ! *
  subroutine set_blocktype_to_gpu(ncoord, loc2glo_ind)
    use memory,only:chmalloc, chmdealloc
    use psf,only:natom
    use block_ltm,only:iblckp
    use domdec_util_gpu_mod,only:copy_blocktype_to_gpu
    use lambdam,only:isitemld
    implicit none
    ! Input
    integer, intent(in) :: ncoord, loc2glo_ind(:)
    ! Variables
    integer i, ibl

    if (allocated(blocktype)) then
       if (size(blocktype) < ncoord) then
          call chmdealloc('domdec_block.src','set_blocktype_to_gpu','blocktype',&
               size(blocktype),intg=blocktype)
       endif
    endif
    if (.not.allocated(blocktype)) then
       call chmalloc('domdec_block.src','set_blocktype_to_gpu','blocktype',&
            min(natom,int(ncoord*1.5)),intg=blocktype)
    endif
    
!$omp parallel do private(i, ibl) schedule(static)
    do i=1,ncoord
       ibl = iblckp(loc2glo_ind(i)+1)
       ! blocktype = (isitemld(ibl) << 16) | (ibl-1)
       blocktype(i) = ior(ibl-1, ishft(isitemld(ibl), 16))
    enddo
!$omp end parallel do

    call copy_blocktype_to_gpu(ncoord, blocktype)
    
    return
  end subroutine set_blocktype_to_gpu

  ! *
  ! * Setup block parameters (fullblcoep, bixlam, biflam, biflam2) on GPU
  ! *
  subroutine set_block_params_to_gpu()
    use memory,only:chmalloc, chmdealloc
    use lambdam,only:bixlam, fullblcoep, biflam, biflam2
    use domdec_util_gpu_mod,only:copy_bixlam_to_gpu, copy_blockparam_to_gpu, copy_biflam_to_gpu
    use block_ltm,only:nblock
    implicit none
    ! Variables
    integer i, j

    ! bixlam_sp
    if (allocated(bixlam_sp)) then
       if (size(bixlam_sp) /= nblock) then
          call chmdealloc('domdec_block.src','set_block_params_to_gpu','bixlam_sp',&
               size(bixlam_sp),cr4=bixlam_sp)
       endif
    endif
    if (.not.allocated(bixlam_sp)) then
       call chmalloc('domdec_block.src','set_block_params_to_gpu','bixlam_sp',&
            nblock,cr4=bixlam_sp)
    endif

    ! fullblcoep_sp
    if (allocated(fullblcoep_sp)) then
       if (size(fullblcoep_sp,1) /= nblock) then
          call chmdealloc('domdec_block.src','uninit_block','fullblcoep_sp',&
               size(fullblcoep_sp,1),size(fullblcoep_sp,2),cr4=fullblcoep_sp)
       endif
    endif
    if (.not.allocated(fullblcoep_sp)) then
       call chmalloc('domdec_block.src','uninit_block','fullblcoep_sp',&
            nblock,nblock,cr4=fullblcoep_sp)
    endif

    
    ! Create single precision versions of the arrays
    do i=1,nblock
       bixlam_sp(i) = real(bixlam(i), kind=chm_real4)
    enddo

    do j=1,nblock
       do i=1,nblock
          fullblcoep_sp(i,j) = real(fullblcoep(i,j), kind=chm_real4)
       enddo
    enddo
    
    ! Copy to GPU
    call copy_bixlam_to_gpu(bixlam_sp)
    call copy_blockparam_to_gpu(fullblcoep_sp)
    call copy_biflam_to_gpu(biflam, biflam2)
    
    return
  end subroutine set_block_params_to_gpu

  ! *
  ! * Add GPU results to biflam and biflam2
  ! *
  subroutine combine_biflam_from_gpu()
    use lambdam,only:biflam, biflam2
    use domdec_util_gpu_mod,only:read_biflam_from_gpu
    use block_ltm,only:nblock
    implicit none
    integer i

    ! Read the results from GPU
    call read_biflam_from_gpu(biflam_loc, biflam2_loc)

    ! Add the results to the CPU results
    do i=1,nblock
       biflam(i) = biflam(i) + biflam_loc(i)
       biflam2(i) = biflam2(i) + biflam2_loc(i)
    enddo
    
    return
  end subroutine combine_biflam_from_gpu
  
#endif
  
#endif
#endif

end module domdec_block
