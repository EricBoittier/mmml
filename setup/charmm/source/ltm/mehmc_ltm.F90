module mehmc
  use chm_kinds
  use dimens_fcm
  character(len=*),private,parameter :: file_name   ="mehmc_ltm.src"
#if KEY_MEHMC==1 /*mehmc_fcm*/
!
!     Momentum-Enhanced Hybrid Monte Carlo
!
!     VMEAVi  The current  average velocities
!     VMEOLi  The previous average velocities
!     BVECMi  The B vector used to bias the initial velocities in HMC
!     RMEHMC  The weight of each MD step in the average velocities
!    
      real(chm_real),allocatable,dimension(:) ::  VMEAVX,VMEAVY,VMEAVZ,VMEOLX,VMEOLY,VMEOLZ, &
           BVECMX,BVECMY,BVECMZ
      real(chm_real)  RMEHMC
!
contains
  subroutine allocate_mehmc(natom)
    use memory
    integer csize, natom
    character(len=*),parameter :: routine_name="allocate_coordc"
    if(allocated(vmeavx).and.size(vmeavx)>natom) then
       return
    elseif(allocated(vmeavx)) then
       csize = size(vmeavx)
       call chmdealloc(file_name,routine_name,'VMEAVX',csize,crl=VMEAVX)
       call chmdealloc(file_name,routine_name,'VMEAVY',csize,crl=VMEAVY)
       call chmdealloc(file_name,routine_name,'VMEAVZ',csize,crl=VMEAVZ)
       call chmdealloc(file_name,routine_name,'VMEOLX',csize,crl=VMEOLX)
       call chmdealloc(file_name,routine_name,'VMEOLY',csize,crl=VMEOLY)
       call chmdealloc(file_name,routine_name,'VMEOLZ',csize,crl=VMEOLZ)
       call chmdealloc(file_name,routine_name,'BVECMX',csize,crl=BVECMX)
       call chmdealloc(file_name,routine_name,'BVECMY',csize,crl=BVECMY)
       call chmdealloc(file_name,routine_name,'BVECMZ',csize,crl=BVECMZ)
    endif
    call chmalloc(file_name,routine_name,'VMEAVX',maxa,crl=VMEAVX)
    call chmalloc(file_name,routine_name,'VMEAVY',maxa,crl=VMEAVY)
    call chmalloc(file_name,routine_name,'VMEAVZ',maxa,crl=VMEAVZ)
    call chmalloc(file_name,routine_name,'VMEOLX',maxa,crl=VMEOLX)
    call chmalloc(file_name,routine_name,'VMEOLY',maxa,crl=VMEOLY)
    call chmalloc(file_name,routine_name,'VMEOLZ',maxa,crl=VMEOLZ)
    call chmalloc(file_name,routine_name,'BVECMX',maxa,crl=BVECMX)
    call chmalloc(file_name,routine_name,'BVECMY',maxa,crl=BVECMY)
    call chmalloc(file_name,routine_name,'BVECMZ',maxa,crl=BVECMZ)
    return
  end subroutine allocate_mehmc
#endif /* (mehmc_fcm)*/

end module mehmc

