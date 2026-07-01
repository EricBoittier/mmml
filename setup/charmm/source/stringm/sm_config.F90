! ********************************************************************************!
! Preprocessed by GNU CPP !
! ********************************************************************************!
!
! SM_CONFIG.MOD
!
! MODULE THAT CONTAINS CONFIGURATION OPTIONS FOR SMCV, FTSM AND SM0K
!
      module sm_config
#if (KEY_STRINGM==1) /*  automatically protect all code */
!
      use chm_kinds
!
!ccccc flags
      logical :: &
     & smcv_on=.false., repa_on=.false., restrained_on=.false., &
     & stat_on=.false., evolve_cv_on=.false., unrestrained_on=.false., &
     & voronoi_hist_on=.false., voronoi_allow_cross=.false., &
     & evolve_expo_on=.false., &
     & evolve_smooth_on=.false., evolve_aver_on=.false., &
     & evolve_bd_on=.false., &
     & voronoi_wrong_cell=.false., & ! this flag is on if the system belongs to a wrong cell for 2 consecutive steps.
     & string_noprint=.false., restraint_force_on=.false., &
     & planar_on=.false., repl_x_on=.false., compute_whereami=.false., &
     & ftsm_on=.false., confcons_on=.false., chirality_on=.false., &
     & steering_on=.false.
      integer :: evolve_freq=0, repa_freq=0, stat_freq=0, hist_freq=0, &
     & restrained_eq_steps=0, restrained_eq0=0, evolve_nskip=0, &
     & evolve_smooth_d=10, unrestrained_eq_steps=0, unrestrained_eq0=0, &
     & num_ave_cv_samples=0, vtime_offset=0, olditeration=0, & ! timestep offset for voronoi calculations
     & repl_x_freq=0, rextime_offset=0, voronoi_update_freq=0, &
     & voronoi_nocross_ini=0, confcons_freq=0, chirality_freq=0
!
      real(chm_real) :: evolve_step=0.0d0, evolve_expo_mem=1.00d0, &
     & evolve_bd_T=0d0
!
! cv index arrays (for parallelization)
! integer, pointer, dimension(:) ::
! & cv_send_displ, cv_send_count,
! & fr_send_displ, fr_send_count,
! & qt_send_displ, qt_send_count,
! & imap_displ, imap_count
      integer*4, pointer, dimension(:) :: &
     & cv_send_displ, cv_send_count, &
     & fr_send_displ, fr_send_count, &
     & qt_send_displ, qt_send_count, &
     & imap_displ, imap_count
! MPI derived types
! integer :: MPI_CV_TYPE2_, MPI_CV_TYPE3_, MPI_GRAD_TYPE_,
! & MPI_CV_TYPE3I_
! integer :: MPI_CV_TYPE2 , MPI_CV_TYPE3 , MPI_GRAD_TYPE,
! & MPI_CV_TYPE3I
      integer*4 :: MPI_CV_TYPE2_, MPI_CV_TYPE3_, MPI_GRAD_TYPE_, &
     & MPI_CV_TYPE3I_
      integer*4 :: MPI_CV_TYPE2 , MPI_CV_TYPE3 , MPI_GRAD_TYPE, &
     & MPI_CV_TYPE3I
! flags that control parallelization
      integer, parameter :: allgather_=1, gather_bcast_=1, hypercube_=3 ! internal constants
      integer :: allgather_method=hypercube_ ! specifies the method to use in ftsm allgather procs
!
      logical :: calc_cv_para=.false., & ! should CVs be calculated in parallel?
     & calc_fr_para=.false., & ! should frames be calculated in parallel?
     & calc_qt_para=.false., & ! should quaternions be calculated in parallel?
     & calc_voronoi_para=.true., & ! should the tessellation metric test be done in parallel?
     & calc_Mtensor_para=.true., &! should the calculation of the M tensor be done in parallel?
     & calc_Mtensor_fast=.true., &! should the calculation of the M tensor use the fast algorithm for sparse matrices?
     & inverse_LU=.true., & ! use LU decomposition for solving Ax=b (and for A inverse)
     & calc_bestfit_grad_para=.true. ! should the calculation of rotation gradients be in parallel?
! finite difference perturbation
      real(chm_real) :: finite_difference_d=0.000001d0
      real(chm_real) :: parallel_tolerance=1.0d-13
!
      integer, parameter :: sizeofreal=8
!
      integer, parameter :: izero=0, ione=1, itwo=2, ithree=3, ifour=4, ifive=5, isix=6 ! ugly fix for CHARMM constants
!
#endif /* automatically protect all code */
      end module sm_config
