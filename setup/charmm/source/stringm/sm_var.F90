! ********************************************************************************!
! Preprocessed by GNU CPP !
! ********************************************************************************!
!
! SM_VAR.MOD
!
! COMMON FILE FOR STRING METHOD IN COLLECTIVE VARIABLES (AND FTSM)
!
      module sm_var
#if (KEY_STRINGM==1) /*  automatically protect all code */
!
      use chm_kinds
      use number
!
!ccccc SMCV initialization flag
      logical, save :: smcv_initialized=.false.
!ccccc number of replicas on the string
      integer, save :: nstring=-1, mestring=-1
!ccccc GENERAL VARS
      logical, save :: repa_initialized=.false.
! interpolation methods
      integer, parameter :: linear=1, spline=2, bspline=3, dst=4, linear_exact=5
      integer, parameter :: interp_methods(5) = (/linear, spline, bspline, dst, linear_exact/);
!
      real(chm_real), parameter :: Id3(3,3)=reshape( (/one, zero, zero, zero, one, zero, zero, zero, one/),(/3,3/))
!
      integer, save :: interp_method=0
      integer, save :: iterations=1 ! maximum interpolation iterations
      real(chm_real), save :: def=1.1d0 ! interpolation tolerance
      real(chm_real), save :: dst_cutoff=1.0d0 ! wavenumber truncation parameter for DST
      logical, save :: fixed_bc_0=.false., fixed_bc_1=.false. ! whether string endpoints are fixed
! arclength and curvature
      real(chm_real), save, pointer :: ds(:), curv(:) ! unavailable at first iteration
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!ccccc STATISTICS VARS FOR SMCV ccccccccccccccccccccccccccccccccccccccccccccccc
      integer, save :: stat_iteration_counter=0 ! how many times 'stat' has been called
      logical, save :: output_rmsd0=.false., &
     & output_dsdt=.false., &
     & output_arclength=.false., & ! output options
     & output_curvature=.false., &
     & output_fe=.false., & ! free energy
     & output_cv=.false., & ! col var
     & output_forces=.false., & ! forces
     & output_rmsd_ave=.false., & ! rmsd wrt to the average structure
     & output_voronoi_hist=.false., & ! output voronoi histograms (for FE computation)
     & output_voronoi_log=.false., & ! write complete log of crossing events
     & output_voronoi_map=.false., & ! write voronoi map (useful in case voronoi cell crossing is allowed)
     & output_work=.false., & ! output noneq. work in during restrained eq.
     & output_wgt=.false., & ! output CV weights
     & output_M=.false., & ! output M matrix
     & output_rex_log=.false., & ! output exchange log
     & output_rex_map=.false. ! output replica map
      integer, save, pointer :: rex_map(:)
      logical, save :: stat_initialized=.false.
!
      character(len=200), save :: cv_fname='',forces_fname='', &
     & rmsd0_fname='', dsdt_fname='', s_fname='', fe_fname='', &
     & rmsd_ave_fname='', voronoi_fname='', c_fname='', &
     & work_fname='', wgt_fname='', M_fname='', &
     & rex_fname='' ! output names
!
      integer, save :: rmsd0_funit=-1, dsdt_funit=-1, s_funit=-1, &
     & fe_funit=-1, rmsd_ave_funit=-1, c_funit=-1, &
     & work_funit=-1, cv_funit=-1, forces_funit=-1, wgt_funit,&
     & vlog_funit=-1, rex_funit=-1
!
      character(len=80) :: rform, dform, sform, feform, cform, wkform, &
     & cvform, fform, wgtform, vlform, rxlform, raform
!
      character(len=8), save :: work_tag
      integer, save :: num_average_samples=0 ! number of samples in the average set
      integer, save :: cv_flen=0, forces_flen=0, &
     & rmsd0_flen=0, dsdt_flen=0, s_flen=0, fe_flen=0, rmsd_ave_flen=0, &
     & voronoi_flen=0, c_flen=0, work_flen=0, wgt_flen=0, M_flen=0, &
     & rex_flen=0
!cccccccccccccccccccccccccccccccccc ADDITIONAL VARS ccccccccccccccccccccccccccccccccccccc
! integer, pointer, save :: posi_islct(:) ! maintains indices of CV atoms (POSI)
#endif /* automatically protect all code */
      end module sm_var
