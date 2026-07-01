! ********************************************************************************!
! Preprocessed by GNU CPP !
! ********************************************************************************!
!
! FTSM_VAR.MOD
!
! VARIABLES FOR THE FINITE TEMPERATURE STRING METHOD
!
      module ftsm_var
#if (KEY_STRINGM==1) /*  automatically protect all code */
      use chm_kinds
      use chm_types
      use number
      use ivector
      use ivector_list
      use rvector_list
! note that SMCV and FTSM should not be used simultaneously
! see module smcv_common for the description of the variables below
!
      use sm_var, only: &
     & nstring, mestring, &
     & repa_initialized, smcv_initialized, &
     & linear, spline, bspline, dst, linear_exact, interp_method, &
     & interp_methods, &
     & iterations, def, &
     & dst_cutoff, &
     & ds, curv, & ! arclength and curvature
     & stat_iteration_counter, &
     & output_rmsd0, &
     & output_arclength, &
     & output_curvature, &
     & output_fe, &
     & output_forces, &
     & output_rex_map, &
     & output_rex_log, &
     & output_M, &
!
 & Id3, &
!
     & stat_initialized, &
!
     & forces_fname, rex_fname, &
     & rmsd0_fname, s_fname, fe_fname, c_fname, M_fname, &
!
     & rmsd0_funit, s_funit, &
     & fe_funit, c_funit, &
     & forces_funit, rex_funit, vlog_funit, &
!
     & rform, sform, feform, cform, fform, rxlform, vlform, &
!
     & forces_flen, rmsd0_flen, s_flen, fe_flen, c_flen, rex_flen, &
     & M_flen, &
! for Voronoi dynamics 1/2013
     & output_voronoi_hist, output_voronoi_map, output_voronoi_log, &
     & voronoi_flen, voronoi_fname
!
      use sm_config, only: &
     & ftsm_on, repa_on, restrained_on, unrestrained_on, &
     & restraint_force_on, &
     & stat_on, &
     & string_noprint, restraint_force_on, &
     & calc_bestfit_grad_para, &
     & allgather_method, allgather_, gather_bcast_, hypercube_, &
     & evolve_freq, stat_freq, &
     & restrained_eq_steps, restrained_eq0, evolve_nskip, &
     & unrestrained_eq_steps, unrestrained_eq0, &
     & olditeration, &
     & evolve_aver_on, & ! in this context, whether the evolution corresponds to averaging the simulation structure
! ! when false, evolve_expo_on is used, which implies a fixed exponential filter width
     & evolve_expo_on, &
     & evolve_expo_mem, &
     & finite_difference_d, &
     & parallel_tolerance, &
     & repl_x_on, &
     & repl_x_freq, &
     & rextime_offset, &
! for Voronoi dynamics 1/2013
     & vtime_offset, &
     & voronoi_hist_on, voronoi_allow_cross, &
     & voronoi_update_freq, voronoi_nocross_ini, calc_voronoi_para, &
     & compute_whereami , ione, itwo, ithree ! ugly fix for CHARMM constants
!
!
       real(chm_real), pointer, save :: fe(:), feav(:) ! free energy arrays
       real(chm_real), save :: avforce(3) ! average parallel (1) and (2) perpendicular force;
       real(chm_real), save :: repl_x_temp ! temperature for replica exchange
       integer, save :: num_evolve_samples=0 ! number of samples in the averaged image
       integer, save :: max_evolve_samples=0 ! maximum number of allowed samples before wraparound (if > 0)
       integer, save :: num_force_samples=0 ! number of force samples
       integer, save :: num_M_samples=0 ! number of samples in the running average of M tensor
       integer, save :: num_J_samples=0 ! number of samples in the Jacobian
       logical, save :: ftsm_initialized=.false.
       logical, save :: ftsm_nbond_image_data_initialized=.false.
       logical, save :: evolve_ftsm_on=.false., & ! is string evolution on?
     & update_on=.false., & ! image updating on?
     & ftsm_mini_on=.false., & ! image minimization on?
     & ftsm_reconnect_on=.false. ! try to reconnect path during updates ?
       logical, save :: output_centers=.false. ! center of the transition tube (these are coordinate files)
       logical, save :: output_connect_map=.false. ! whether to output ftsm path connectivity map
       logical, save :: output_connect_log=.false. ! whether to output ftsm path connectivity log
       logical, save :: output_dist=.false. ! whether to output distance to the string (dpar, dprp, or drms)
       logical, save :: qdiffrot=.false. ! whether the orientation atoms are different from forcing atoms, in which case store them separately
       logical, save :: qorient=.false. ! whether images are to be oriented in the bestfit sense
       logical, save :: proj_on=.false. ! whether the string evolves only along the direction perpendicular to itself
       logical, save :: fe_curvature=.true. ! whether to compute curvature contribution to the free energy
!
       logical, save :: output_J=.false. ! whether to compute FTSM Jacobian
!
       character(len=80), save :: centers_fname='', dist_fname='', J_fname='', connect_fname='';
       integer, save :: centers_funit=-1, centers_flen=0, dist_funit=-1, dist_flen=0, &
& J_flen=0, connect_funit=-1, connect_flen=0;
       character(len=80), save :: cenform, distform, connectform
!
       integer, save :: norient=0, nforced=0, & ! orientation and forcing atoms
     & nany=0, nboth=0 ! any atoms, overlapping atoms
       integer, save :: update_freq=0 ! frequency for performing image update
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! coordinate variables below
       real(chm_real), save, pointer, dimension (:,:,:) :: &
     & r_f, & ! forcing (rall`s will not always be associated so beware)
     & r_o ! orientation
       real(chm_real), save, pointer, dimension(:,:,:,:,:) :: Mtensor ! metric tensor (stats only)
       real(chm_real), save, pointer, dimension(:,:) :: rcom ! COM coordinates
       real(chm_real), save :: Jacobian(2) ! FTSM Jacobian
! weights
       real(chm_real), save, pointer :: orientWeights(:), forcedWeights(:)
! indices
       integer, save, pointer :: iatom_o(:), iatom_f(:) ! contains psf atomid indices
       type (int_vector), pointer :: iatoms_a ! a vector of all of the atom indices usde by ftsm
! COM index lists (ftsm_com_on version)
       type (int_vlist), pointer :: iatoms_o, iatoms_f ! psf atomid lists for computing COMs
       type (real_vlist), pointer :: wgts_o, wgts_f ! weights corresponding to iatoms for computing COMs
!
       integer, save, pointer :: iatom_both(:,:) ! index pairs that correspond to the same PSF atoms
                                                                     ! in the case of COM coordinates (ftsm_com_on) iatom_both
                                                                     ! contains index pairs into atom lists
! define indices into r array:
       integer, parameter, public :: left=1, &
     & center=2, &
     & right=3, &
     & left_old=4, & ! old structures are saved to provide smooth transitions after evolution
     & center_old=5, &
     & right_old=6, &
     & center_new=7, &
     & ref=8, &
     & instant=9, &
     & rave=11, &
     & dummy=10, &
     & vpar=12, &
     & vperp=13, &
     & fpar=14, &
     & fperp=15, &
     & left_rot=16, &
     & center_rot=17, &
     & right_rot=18, &
     & left_cur=19, &
     & center_cur=20, &
     & right_cur=21, &
     & scratch=22, &
     & scratch2=23
!
       real(chm_real), save :: krms=0d0, kpara=0d0, kperp=0d0, &
     & dpar0=0.5d0,dperp0=0d0,drms0=0d0,dpar,dperp,drms,&
     & dperp0i, dperp0f, dperp_adjust_iter, & ! initial and final dperp (optional) , iterations to adjust to final (optional)
     & dpar0i, dpar0f, &
     & ftsm_ini_iteration=-1, & ! initial iteration (computed by ftsm_main)
     & fe_curv=0d0 ! instantaneous curvature contribution to the free energy
      logical, save :: qrms_upper_bound=.false.
      logical, save :: qkpara_angstrom=.false., & ! whether the distance is specified in Ang (unscaled vs scaled, as implemented originally)
     & qkperp_angstrom=.false., & ! as above
     & qdperp_angstrom=.false., &
     & qdperpf_angstrom=.false.
!
       integer, parameter, public :: num_sets=23 ! num of parameters above
       integer*4, save :: MPI_RTMD_TYPE, MPI_RTMD_TYPE_
!
       character(len=8), parameter, public :: real_format='(E23.15)'
       character(len=5), parameter, public :: int_format='(I10)'
       character(len=1), parameter, public :: tab=char(9)
!
! disabling for now -- limited compiler compatibility
! abstract interface ! for procedure ftsm_calc
! subroutine ftsm_calculate(x,y,z,deriv,t)
! use chm_kinds
! real(chm_real), intent(in) :: x(:), y(:), z(:)
! logical :: deriv
! real(chm_real), intent(in), optional :: t
! end subroutine ftsm_calculate
! end interface
!
       logical :: qver2 = .true. ! whether to use ftsm version 2
       logical :: ftsm_scaledist=.true. ! whether to use scaled RMSD in FTSM ; currently implemented only with ftsm version 2
       logical :: ftsm_com_on=.true. ! whether to use COM of atom groups in FTSM rather than individual atom positions
       logical :: ftsm_compute_qdiffrot=.true. ! whether need to determine if o/f atom groups are identical
       logical :: ftsm_reset_dpar0f=.false. ! whether to recompute reference location for dpar; valid only with ftsm in unscaled coordinates
! real(chm_real) :: ftsm_flim_coeff = 20d0 ! for limiting force magnitude if a degenerate fluctuating alignment creates large forces
       real(chm_real) :: ftsm_flim_coeff = -1d0 ! (<0 =>off) : for limiting force magnitude if a degenerate fluctuating alignment creates large forces
       real(chm_real), parameter :: sqrt3 = sqrt( three )
! (NOTE : ftsm paper describes unscaled version, but scaled version is ``neater`` because the projection reference is fixed at 1/2 rather
! than at L/(M-1), which changes because L -- the string length -- changes during evolution/reparametrization)
! procedure (ftsm_calculate), pointer :: ftsm_calc=>NULL()
!
#endif /* automatically protect all code */
      end module ftsm_var
!
