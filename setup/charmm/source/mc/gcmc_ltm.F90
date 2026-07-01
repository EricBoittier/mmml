module gcmc
  use chm_kinds
  use dimens_fcm
  implicit none
  character(len=*),private,parameter :: file_name   ="gcmc_ltm.src"

#if KEY_GCMC==1
!
!     Grand canonical Monte Carlo shared arrays
!
!     QGCMC  - flag indicating GCMC is active
!     GCMCON - flag indicating an atom is active
!     GCBLKR - flag indicating an atom occludes space in cavity-biased MC
!
      LOGICAL QGCMC
      logical,allocatable,dimension(:) :: GCMCON, GCBLKR


contains
  subroutine gcmc_init()
    qgcmc=.false.
    return
  end subroutine gcmc_init

  subroutine allocate_gcmc()
    use memory
    character(len=*),parameter :: routine_name="allocate_gcmc"
    call chmalloc(file_name,routine_name,'gcmcon ',maxa,log=gcmcon)
    call chmalloc(file_name,routine_name,'gcblkr ',maxa,log=gcblkr)
    gcmcon=.true.
    gcblkr = .false.
    return
  end subroutine allocate_gcmc

#endif 
end module gcmc

