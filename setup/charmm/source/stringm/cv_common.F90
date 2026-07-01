! ********************************************************************************!
! Preprocessed by GNU CPP !
! ********************************************************************************!
!
! CV_COMMON.MOD
!
! COMMON ROUTINES FOR COLLECTIVE VARIABLES
! SPECIFIC CV USE THIS MODULE
! ADDITIONAL STORAGE FOR CV TYPES SHOULD BE ALLOCATED IN CV%PRIV(I),
! WHERE I IS THE CV INDEX
!
!
      module cv_common
!
#if (KEY_STRINGM==1) /*  automatically protect all code */
!
      use chm_kinds
!
      use ivector_list
      use cv_types ! unfortunately, need to include this module because angle calculations require special treatment
!
      implicit none
!
      private
! declare pointer type (for arrays of pointer)
      type priv_ptr
       integer, pointer :: p(:), amap_ptr(:) ! amap_ptr contains a unique list of indices in the amap that the cv depends on
       real(chm_real), pointer :: pr(:)
      end type priv_ptr
!****************************************************************************************************************************
      type cv_base
!
      real(chm_real), dimension(:,:), pointer :: r ! cv coordinates
      real(chm_real), dimension(:,:,:), pointer :: M ! mass matrix: (running average); two slots for M, (:,:,1) and (:,:,2); two slots for M^{-1} (:,:,3-4)
      real(chm_real), dimension(:,:,:,:), pointer :: grad ! combined gradient array; introduced for parallelization (comm. efficiency)
      real(chm_real), dimension(:,:,:), pointer :: gradx ! gradient vector of CV w.r.t. x ; (:,:,2) is mass-weighted to compute M
      real(chm_real), dimension(:,:,:), pointer :: grady ! gradient vector of CV w.r.t. y ; (:,:,2) is mass-weighted to compute M
      real(chm_real), dimension(:,:,:), pointer :: gradz ! gradient vector of CV w.r.t. z ; (:,:,2) is mass-weighted to compute M
      integer, dimension(:), pointer :: type ! cv type (e.g. position, bond, angle, etc.)
      logical, dimension(:), pointer :: active ! if flag .false., cv will be computed, but force not added, unless this cv is a part of another function
! e.g. RMSD ; experimental as of 7.2010; to make this compatible witl all features, need to ignore cvs
! that are "inactive" in reparametrization, df computation, etc.; currently used for SMD
      type (priv_ptr), dimension(:), pointer :: priv ! cv data that is 'private' to each cv
      type (int_vlist) :: amap ! maps local indices to psf atom indices; also contains an inverse map: which CV make use of an index
      real(chm_real), dimension(:), pointer :: r_bc_0, r_bc_1 ! arrays for optional fixed b.c.
      real(chm_real), dimension(:), pointer :: k ! force constant
      real(chm_real) :: kpara ! force constant for forces parallel to string ; same for all CV (required for off-path sim.)
      real(chm_real) :: kperp ! force constant for forces perpendicular to string ; same for all CV
      real(chm_real), dimension(:), pointer :: gamma
      real(chm_real), dimension(:), pointer :: weight
      real(chm_real) :: wrss ! one over root-sum-square of the weight array
      integer :: num_cv=0 ! number of active collective vars
      integer :: num_hist=0 ! number of snapshots in the history
      integer :: beg_hist=1 ! index in the history that corresponds to the oldest snapshot (first)
      integer :: end_hist=0 ! index in the history that corresponds to the newest snapshot (last)
      integer :: num_run_ave=0 ! number of snapshots in the running average
      real(chm_real) :: dt=0 ! timestep for string evolution
      integer :: num_average_samples=0 ! number of samples in the accumulated CV average
      integer :: num_fe_samples=0 ! number of samples in the fe running average
      real(chm_real), dimension(:), pointer :: ds, fe, curvature, feav ! arclength,FE,curvature, avg. FE
      real(chm_real), dimension(:,:), pointer :: rall ! holds a complete set of CV for Voronoi T.
      real(chm_real), dimension(:,:,:), pointer :: Mall ! holds a complete set of M matrices & inverses (Voronoi T.)
      real(chm_real) :: work ! nonequilibrium work
      integer, dimension(:,:,:), pointer :: voronoi_data ! holds the nine arrays listed below
! local
      integer, dimension(:,:), pointer :: cross_attempt ! holds the history of crossing attempts in 1st column
      integer, dimension(:,:), pointer :: cross_accept ! history of successful crossing attempts in second column. For Voronoi Tessellation.
      integer, dimension(:), pointer :: voro_occupancy ! total # iterations voronoi cells are populated (watch out for max integer value !!!)
! global
      integer, dimension(:,:), pointer :: cross_attemptG
      integer, dimension(:,:), pointer :: cross_acceptG
      integer, dimension(:), pointer :: voro_occupancyG
! old global (for restarting)
      integer, dimension(:,:), pointer :: cross_attemptO
      integer, dimension(:,:), pointer :: cross_acceptO
      integer, dimension(:), pointer :: voro_occupancyO
!
      integer, dimension(:), pointer :: voronoi_map ! holds map between process rank and voronoi cell
      type (int_vector) :: voro_log ! logs the history of crossing attempts ( which cells and when ); local to each replica
      integer :: voronoi_whereami ! the voronoi cell this replica is inside
      real(chm_real) :: voronoi_cut ! voronoi cell cutoff in path-perpendicular direction (beyond which MD replicas are not allowed)
      type (int_vector) :: rex_log ! logs the history of replica exchanges ( which replica and when ); local to each replica
      integer, dimension(:), pointer :: rex_map ! holds the map between replica number and the process rank
      real(chm_real) :: rex_beta ! ensemble temperature in replica exchange (usually will be the same as the simulation temperature)
      end type cv_base
!
      public priv_ptr
      ! subroutines
      public cv_common_init ! initialize cv array
      public cv_common_done ! destroy cv array
      public cv_common_add ! add a CV
      public cv_common_fill ! set z=theta(x)
      public cv_common_grad_init ! initialize cv%grad arrays
      public cv_common_set_bc ! set boundary conditions on string (fixed or free)
      public cv_common_repa ! reparameterize
      public cv_common_print_local ! print out cv values (i.e. z) to separate files
      public cv_common_print_global ! print out cv values (i.e. z) to a combined file
      public cv_common_add_hist ! save an instance of theta(x)
      public cv_common_print_hist_local ! write history of computed cv values (i.e. theta(x)) to file
      public cv_common_print_hist_global ! write history of computed cv values (i.e. theta(x)) to file
      public cv_common_clear_hist ! clear history
      public cv_common_smooth_hist ! filter history & place in comp-set r(:,:,2)
      public cv_common_evolve_expo ! evolution using exponential averaging of history
      public cv_common_swap ! swap main-set and comp_set, i.e. r(:,:,1) and r(:,:,2)
      public cv_common_copy ! swap main-set and comp_set, i.e. r(:,:,1) and r(:,:,2)
      public cv_common_read_global ! read cv values from separate files
      public cv_common_read_local ! read cv values from separate files
      public cv_common_read_local_from_global ! read cv values from a particular column in a global file
      public cv_common_unwrap_angles ! make sure that along the string the angle does not change by more than 180 degrees
      public cv_common_evolve_smcv ! evolve string using average force a la SMCV
      public cv_common_evolve_sd ! evolve string using average force (simple SD)
      public cv_common_rmsd ! compute rmsd between two sets of CV
      public cv_common_set_kpara ! set parallel force constant k
      public cv_common_set_kperp ! set perpendicular force constant k
      public cv_common_set_k ! set force constant k
      public cv_common_set_w ! set weight
      public cv_common_set_active ! set active flag
      public cv_common_set_g ! set gamma
      public cv_common_set_dt ! set default timestep
      public cv_common_set_r ! set collective variable value
      public cv_common_print_curvature ! print out curvature
      public cv_common_print_ds ! print out ds
      public cv_common_print_fe ! print out fe
      public cv_common_print_feav ! print out fe averaged over the statistics interval
      public cv_common_print_forces ! write out forces on CV
      public cv_common_compute_fe_fd ! finite difference computation of free energy using integration
      public cv_common_update_ave ! update average CV set
      public cv_common_update_fe_ave ! update average work curve
      public cv_common_set_ave_samples ! change the number of samples associated with the average CV
      public cv_common_voronoi_compute ! determine which voronoi cell we are in
      public cv_common_voronoi_print_log ! print voronoi log
      public cv_common_voronoi_init ! initialize voronoi histogram matrix
      public cv_common_voronoi_update ! update voronoi cell centers
      public cv_common_voronoi_smart_update ! update vorohnoi cell centers sudh that the MD replicas remain inside their cells
      public cv_common_voronoi_done ! deallocate voronoi histogram matrix
      public cv_common_voronoi_set_cutoff ! set voronoi cell cutoff
      public cv_common_read_voro_map ! read voronoi map from file
      public cv_common_print_voro_map ! print voronoi map to file
      public cv_common_read_voro_data ! read voronoi crossing data from file
      public cv_common_print_voro_data ! print voronoi crossing to file
      public cv_common_neq_work_init ! initialize variables for computing work
      public cv_common_neq_get_work ! return work
      public cv_common_interpolate ! interpolate CV onto a different grid
      public cv_common_compute_wgt ! compute CV weights for interpolation and RMSD calculations
      public cv_common_print_wgt ! print CV weights
      public cv_common_read_wgt ! read in CV weights
      public cv_common_compute_dr ! compute tangent to the path
      public cv_common_print_dr ! print tangent to the path
      public cv_common_read_dr ! read tangent to the path
      public cv_common_compute_k ! calculate force constants using k_i=kpara*w_i^2
      public cv_common_print_M_global ! print metric tensor
      public cv_common_read_M_global ! read metric tensor
      public cv_common_compute_Minv ! compute inverse of metric tensor
      public cv_common_rex_init ! initialize replica exchange variables
      public cv_common_rex_done
      public cv_common_rex_set_temp
      public cv_common_rex_compute_dE ! compute energy difference
      public cv_common_rex_read_map ! read replica map from file
      public cv_common_rex_print_map ! print replica map
      public cv_common_rex_print_log ! print replica map
! variables
 character(len=132)::info(21)=(/'','','','','','','','','','','','','','','','','','','','',''/);! output buffer
      type (cv_base), public, save :: cv
!
      integer, public :: max_cv_common=200
      integer, parameter :: max_hist_base=1000
!
      logical, public, save :: cv_common_initialized=.false., &
     & cv_common_voronoi_initialized=.false., &
     & cv_common_voronoi_wrong_cell=.false., &
     & cv_common_grad_initialized=.false., & ! have the cv%grad arrays been allocated
     & cv_common_weights_initialized=.false., & ! has the weight array been initialized, obsolescent
     & cv_common_Minv_initialized=.false., & ! has the M-1 tensor been initialized
     & cv_common_dz_initialized=.false., & ! has the weight array been initialized
     & cv_common_k_initialized=.false., & ! added for off_path sampling
     & cv_common_kpara_initialized=.false., & ! parallel force constant (off-path)
     & cv_common_kperp_initialized=.false., & ! perpendicular force constant (off-path)
     & cv_common_rex_initialized=.false. ! exchange of adjacent replicas
!
      integer, public, save :: cv_common_fixed_0_bc=0, & ! first point fixed (not simulated)
     & cv_common_fixed_1_bc=0 ! last point fixed (not simulated)
!
      ! parameters
! define indices into r array:
      integer, parameter, public :: main=1, & ! current z
     & comp=2, & ! previous z
     & ref=3, & ! reference z (initial)
     & ref2=4, & ! reference z (average)
     & zcur=5, & ! z at current MD iteration (combination of main and comp)
     & zold=6, & ! z at previous MD iterations (combination of main and comp); currently not used
     & dz=7, & ! vector tangent to the path
     & runave=8, & ! running average between string steps
     & instant=9, & ! (instantaneous) theta(x)
     & previnst=10, & ! previous value of (instantaneous) theta(x)
     & forces=11, & ! average forces on z (dF/dz) for evolution
     & forces2=12 ! instantaneous forces (not used for evolution)
!
      integer, parameter, public :: main_offset=12 ! num of parameters above
!
      character(len=8), parameter, public :: real_format='(E23.15)'
      character(len=5), parameter, public :: int_format='(I10)'
!
      contains
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine cv_common_init(max_cv)
       use sm_var, only: nstring, smcv_initialized
      use number
      use multicom_aux;
!
       integer, optional :: max_cv
       integer :: nrep ! number of total replicas
       integer :: i
!
       interface
        subroutine smcv_init(maxcv); integer, optional :: maxcv ; end subroutine smcv_init
       end interface
!
       if (.not.smcv_initialized) call smcv_init() ! defines nstring
!
       if (.not.cv_common_initialized) then
        if (present(max_cv)) max_cv_common=max_cv
!
        nrep=nstring+cv_common_fixed_0_bc+cv_common_fixed_1_bc
        cv%num_cv=0
        call cv_common_clear_hist()
!
        allocate(cv%r(max_cv_common, max_hist_base+main_offset))
!
        if (cv_common_fixed_0_bc.eq.1) then
         allocate(cv%r_bc_0(max_cv_common))
         cv%r_bc_0=anum
        endif
!
        if (cv_common_fixed_1_bc.eq.1) then
         allocate(cv%r_bc_1(max_cv_common))
         cv%r_bc_1=anum
        endif
!
! metric M tensor : 1 -- running average in current averaging window (i.e. a short-time average)
! 2 -- reduced M from all slave processors (in parallel); think of as scratch
! 3 -- inverse of M ; by default, computed from M(4) -- see below
! 4 -- long-time average of M. In smcv evolution M(4)=M(1); expo evolution M(4) has long memory
        allocate(cv%M(max_cv_common,max_cv_common,4))
!
        allocate(cv%priv(max_cv_common)) ! allocate pointer array
        do i=1, max_cv_common
         nullify(cv%priv(i)%p) ! initialize pointer array
         nullify(cv%priv(i)%amap_ptr) ! nullify pointer to atom map (see above)
         nullify(cv%priv(i)%pr)
        enddo
        allocate(cv%type(max_cv_common))
        allocate(cv%active(max_cv_common))
        allocate(cv%k(max_cv_common))
        allocate(cv%gamma(max_cv_common))
        allocate(cv%weight(max_cv_common))
        allocate(cv%ds(nrep-1))
        allocate(cv%curvature(nrep-2))
        allocate(cv%fe(nrep))
        allocate(cv%feav(nrep))
        cv_common_initialized=.true.
        cv%r=anum
        cv%r(:,runave)=0d0 ! not necessary, but cleaner when CVs include angles which need to be averaged
        cv%r(:,previnst)=0d0 ! not necessary, but cleaner when CVs include angles
        cv%M=0d0
        cv%type=-1
        cv%active=.true. ! all cv active by default (and for compatibility with previous code)
        cv%k=0d0
        cv%kpara=0d0
        cv%kperp=0d0
        cv%gamma=0d0
        cv%weight=0d0 ! need to be initialized by user
        cv%ds=0d0
        cv%curvature=0d0
        cv%fe=0d0
        cv%feav=0d0
        cv%dt=0d0
        cv%work=0d0
        cv%num_average_samples=0
        cv%num_fe_samples=0
       endif
       end subroutine cv_common_init
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine cv_common_done()

       integer :: i
       cv%num_cv=0
       call cv_common_clear_hist()
       cv%dt=0d0
       if (cv_common_initialized) then
        deallocate(cv%r)
!
        if (cv_common_fixed_0_bc.eq.1) then
         deallocate(cv%r_bc_0)
        endif
!
        if (cv_common_fixed_1_bc.eq.1) then
         deallocate(cv%r_bc_1)
        endif
!
        do i=1, max_cv_common
         if (associated(cv%priv(i)%p)) &
     & deallocate(cv%priv(i)%p) ! free private memory
         if (associated(cv%priv(i)%amap_ptr)) &
     & deallocate(cv%priv(i)%amap_ptr) ! free pointer array to atom map
         if (associated(cv%priv(i)%pr)) &
     & deallocate(cv%priv(i)%pr) ! free private memory
        enddo
        call int_vlist_done(cv%amap) ! deallocate amap if necessasry
!
        deallocate(cv%M)
        if (associated(cv%grad)) deallocate(cv%grad)
        nullify(cv%gradx)
        nullify(cv%grady)
        nullify(cv%gradz)
! if (associated(cv%gradx)) deallocate(cv%gradx)
! if (associated(cv%grady)) deallocate(cv%grady)
! if (associated(cv%gradz)) deallocate(cv%gradz)
        deallocate(cv%type)
        deallocate(cv%active)
        deallocate(cv%k)
        deallocate(cv%gamma)
        deallocate(cv%weight)
        deallocate(cv%ds)
        deallocate(cv%curvature)
        deallocate(cv%fe)
        deallocate(cv%feav)
        cv_common_initialized=.false.
        cv_common_weights_initialized=.false.
        cv_common_Minv_initialized=.false.
        cv_common_voronoi_initialized=.false.
        cv_common_grad_initialized=.false.
        cv_common_dz_initialized=.false.
        cv_common_k_initialized=.false.
        cv_common_kpara_initialized=.false.
        cv_common_kperp_initialized=.false.
        cv_common_rex_initialized=.false.
       endif
       end subroutine cv_common_done
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       function cv_common_add(k,gamma,weight,type)
      use stream
      use number

!
       integer :: type
       real(chm_real), optional :: k, gamma, weight
! locals
       integer j
       integer cv_common_add ! returns index into cv array
       character(len=len("CV_COMMON_ADD>") ),parameter::whoami="CV_COMMON_ADD>";!macro
! do work
       if (.not.cv_common_initialized) call cv_common_init()
       j=cv%num_cv + 1
       if (j.le.max_cv_common) then
! assign cv values
        cv%num_cv=j
        cv%r(j,:)=anum ! undefined value
!
        cv%k(j)=0d0
        if (present(k)) then
         if (k.ge.0d0) then ! negative k are ignored
          cv%k(j)=k
         endif
        endif
!
        if (present(gamma)) then
         if (gamma.gt.0d0) then
          cv%gamma(j)=1d0/gamma; ! disallow zero
         else
          call wrndie(0,whoami,trim('NONPOSITIVE GAMMA SPECIFIED. WILL RESET TO 1.0.'))
          cv%gamma(j)=1d0;
         endif
        else
         cv%gamma(j)=1d0;
        endif
!
        cv%weight(j)=0d0
        if (present(weight)) then
         if (weight.gt.0d0) then ! negative weights are ignored
          cv%weight(j)=weight
         endif
        endif
        cv%type(j)=type
        cv_common_add=j
!
        if (any(cv%weight(1:cv%num_cv).le.0d0)) then
         cv_common_weights_initialized=.false.
        else
         cv_common_weights_initialized=.true.
        endif
        cv%wrss=1./sqrt( sum( cv%weight(1:cv%num_cv)**2) )
       else ! out of bounds
! maximum storage space exceeded; for now complain and exit; in the future, can reallocate
        call wrndie(0,whoami,trim('MAXIMUM NUMBER OF CV EXCEEDED. NOTHING DONE.'))
        cv_common_add=0
        return
       endif
       end function cv_common_add
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine cv_common_fill(i,val,c)
      use stream
!
       integer :: i
       real(chm_real) :: val
       integer, optional :: c
       integer :: c1
       character(len=len("CV_COMMON_FILL>") ),parameter::whoami="CV_COMMON_FILL>";!macro
! do work:
       if (present(c)) then
        if (c.lt.1.or.c.gt.main_offset+max_hist_base) then
         call wrndie(0,whoami,trim('OUT OF BOUNDS. NOTHING DONE.'))
         return
        else ! c valid
         c1=c
        endif
       else ! c was not passed
        c1=main
       endif
       cv%r(i,c1)=val
       end subroutine cv_common_fill
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine cv_common_grad_init()
! initialize cv%grad arrays

       if (associated(cv%grad)) deallocate(cv%grad)
! if (associated(cv%gradx)) deallocate(cv%gradx) ! delete old data if present
! if (associated(cv%grady)) deallocate(cv%grady) ! delete old data if present
! if (associated(cv%gradz)) deallocate(cv%gradz) ! delete old data if present
       allocate(cv%grad(cv%num_cv, cv%amap%last, 2, 3)) ! x,y,z components are indexed by last dimension
       cv%gradx=>cv%grad(:, :, :, 1);
       cv%grady=>cv%grad(:, :, :, 2);
       cv%gradz=>cv%grad(:, :, :, 3);
! allocate(cv%gradx(cv%num_cv, cv%amap%last, 2))
! allocate(cv%grady(cv%num_cv, cv%amap%last, 2))
! allocate(cv%gradz(cv%num_cv, cv%amap%last, 2))
! cv%gradx=0d0; cv%grady=0d0; cv%gradz=0d0
       cv%grad=0d0;
       cv_common_grad_initialized=.true.
       end subroutine cv_common_grad_init
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine cv_common_compute_Minv(inverse_LU, ind)
       use lu, only: inv_lu ! matrix inverse by LU decomposition
       use multidiag, only: inv_mdiag

       logical, optional, intent(in) :: inverse_LU
       integer, optional, intent(in) :: ind
       logical :: qLU=.true.
       integer :: bug, Mind
       if (present(inverse_LU)) then ; qLU=inverse_LU; endif
       if (present(ind)) then ; Mind=ind; else ; Mind=4 ; endif
! assume matrix M is regular
! if using multidiag, also assume it has nonzero diagonals
       if (qLU) then
        call inv_lu(cv%M(1:cv%num_cv,1:cv%num_cv,Mind), &
     & cv%M(1:cv%num_cv,1:cv%num_cv,3), cv%num_cv, bug) ! M inverse in M(3)
       else
        call inv_mdiag(cv%M(1:cv%num_cv,1:cv%num_cv,Mind), &
     & cv%M(1:cv%num_cv,1:cv%num_cv,3), cv%num_cv, bug)
       endif
!
       if (bug.eq.0) cv_common_Minv_initialized=.true.
!
       end subroutine cv_common_compute_Minv
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine cv_common_compute_wgt()
! compute weight array from M matrix
! assume that M is valid on each node
      use stream
      use multicom_aux;
      use mpi
      use parallel, only: psnd4, psnd8

!
#if (KEY_PARALLEL==1)
#if (KEY_SINGLE==1)
 integer :: mpifloat=MPI_REAL 
#endif
#if (KEY_SINGLE==0)
 integer :: mpifloat=MPI_REAL8 
#endif
#if (KEY_INTEGER8==0)
 integer :: mpiint=MPI_INTEGER 
#endif
#if (KEY_INTEGER8==1)
 integer :: mpiint=MPI_INTEGER8 
#endif
 integer :: mpichar=MPI_CHARACTER
 integer :: mpibool=MPI_LOGICAL
#endif
!
       real(chm_real) :: s, sum
       integer :: i, j, ierror
! local
!
       logical :: qroot
!
       real(chm_real) :: Mave(max_cv_common, cv%num_cv)
       character(len=len("CV_COMMON_COMPUTE_WGT>") ),parameter::whoami="CV_COMMON_COMPUTE_WGT>";!macro
!
       qroot=(MPI_COMM_STRNG.ne.MPI_COMM_NULL)
       if (qroot) then
       call MPI_ALLREDUCE(cv%M(1,1,3),Mave,max_cv_common*cv%num_cv, &
     & mpifloat, MPI_SUM, MPI_COMM_STRNG, ierror) ! use M inverse averaged along string
! 11/30/08: note: it is not completely clear to me how the weights should computed
        do j=1,cv%num_cv
         s=0d0; do i=1, cv%num_cv ; s=s+Mave(i,j)**2 ; enddo
! s=1d0*SIZE_STRNG/sqrt(s) ! average, since we used MPI_SUM above
         s=(s/SIZE_STRNG)**0.25d0 ! average [SIZE_STRNG], since we used MPI_SUM above
         cv%weight(j)=s ! for positions, s will be sqrt(mass), as expected
        enddo
        if (any(cv%weight(1:cv%num_cv).eq.0)) &
        call wrndie(0,whoami,trim('ZERO CV WEIGHT COMPUTED.'))
       endif ! qroot
!
! broadcast to slaves
!
       if (MPI_COMM_LOCAL.ne.MPI_COMM_NULL.and.SIZE_LOCAL.gt.1) then
#if (KEY_SINGLE==1)
        call PSND4(cv%weight(1:cv%num_cv),cv%num_cv) 
#endif
#if (KEY_SINGLE==0)
        call PSND8(cv%weight(1:cv%num_cv),cv%num_cv) 
#endif
       endif
!
       cv%wrss=1./sqrt( sum( cv%weight(1:cv%num_cv)**2) ) ! compute rms
       cv_common_weights_initialized=.true.
       end subroutine cv_common_compute_wgt
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine cv_common_print_wgt(iunit,fmt)
       use stream
! only root process must call
       integer iunit
       character(len=*), optional :: fmt
! local
       character(len=80) :: frm
       character(len=len("CV_COMMON_PRINT_WGT>") ),parameter::whoami="CV_COMMON_PRINT_WGT>";!macro
! begin
       if (.not.cv_common_weights_initialized) &
     & call wrndie(0,whoami,trim('CV WEIGHTS NOT INITIALIZED.'))
       if (.not.present(fmt)) then
        write(frm,'("(",I5,"E15.5)")') max(cv%num_cv,1)
       else
        frm=fmt
       endif
       write(iunit,frm) cv%weight(1:cv%num_cv)
       end subroutine cv_common_print_wgt
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine cv_common_read_wgt(iunit)
      use stream
      use multicom_aux;
      use mpi
      use parallel, only: psnd4, psnd8
!
#if (KEY_PARALLEL==1)
#if (KEY_SINGLE==1)
 integer :: mpifloat=MPI_REAL 
#endif
#if (KEY_SINGLE==0)
 integer :: mpifloat=MPI_REAL8 
#endif
#if (KEY_INTEGER8==0)
 integer :: mpiint=MPI_INTEGER 
#endif
#if (KEY_INTEGER8==1)
 integer :: mpiint=MPI_INTEGER8 
#endif
 integer :: mpichar=MPI_CHARACTER
 integer :: mpibool=MPI_LOGICAL
#endif
!
       integer iunit, ierror
       character(len=len("CV_COMMON_READ_WGT>") ),parameter::whoami="CV_COMMON_READ_WGT>";!macro
!
       if (MPI_COMM_STRNG.ne.MPI_COMM_NULL) then
        if (ME_STRNG.eq.0) then
         read(iunit,*) cv%weight(1:cv%num_cv) ! only root reads
         if (any(cv%weight(1:cv%num_cv).le.0)) &
     & call wrndie(0,whoami,trim('READ ZERO OR NEGATIVE CV WEIGHT'))
        endif ! ME_
        if (SIZE_STRNG.gt.1) &
     & call mpi_bcast(cv%weight,cv%num_cv,mpifloat,0,MPI_COMM_STRNG,ierror)
       endif ! MPI_COMM
! broadcast to slave nodes
       if (ME_LOCAL.ne.MPI_UNDEFINED.and.SIZE_LOCAL.gt.1) &
! & call MPI_BCAST(cv%weight, cv%num_cv, mpifloat,
! & 0,MPI_COMM_LOCAL,ierr)
#if (KEY_SINGLE==1)
     & call PSND4(cv%weight,cv%num_cv) 
#endif
#if (KEY_SINGLE==0)
     & call PSND8(cv%weight,cv%num_cv) 
#endif
!
       cv%wrss=1./sqrt( sum( cv%weight(1:cv%num_cv)**2) )
       cv_common_weights_initialized=.true.
       end subroutine cv_common_read_wgt
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine cv_common_compute_k()
       use stream
       integer :: i
       character(len=len("CV_COMMON_COMPUTE_K>") ),parameter::whoami="CV_COMMON_COMPUTE_K>";!macro
!
       if (cv_common_weights_initialized) then
        if (cv_common_kpara_initialized) then
         do i=1, cv%num_cv
          cv%k(i)=cv%kpara*(cv%weight(i)**2)
         enddo
         cv_common_k_initialized=.true. ! note: this flag only relevant for off-path sampling
         if (any(cv%k(1:cv%num_cv).lt.0)) &
     & call wrndie(0,whoami,trim('COMPUTED NEGATIVE FORCE CONSTANT'))
        else
         call wrndie(0,whoami,trim('PARALLEL FORCE CONSTANT NOT SET. NOTHING DONE.'))
        endif
       else
         call wrndie(0,whoami,trim('CV WEIGHTS NOT INITIALIZED. NOTHING DONE.'))
       endif
!
       end subroutine cv_common_compute_k
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine cv_common_compute_dr()
       use sm_var, only: nstring
       use multicom_aux;
       use mpi
       use parallel, only: psnd4, psnd8
       integer :: ierror
!
#if (KEY_PARALLEL==1)
#if (KEY_SINGLE==1)
 integer :: mpifloat=MPI_REAL 
#endif
#if (KEY_SINGLE==0)
 integer :: mpifloat=MPI_REAL8 
#endif
#if (KEY_INTEGER8==0)
 integer :: mpiint=MPI_INTEGER 
#endif
#if (KEY_INTEGER8==1)
 integer :: mpiint=MPI_INTEGER8 
#endif
 integer :: mpichar=MPI_CHARACTER
 integer :: mpibool=MPI_LOGICAL
#endif
! interface to 'compute_dr' utility routine
       interface
        subroutine compute_dr(rin,drout,wgt,n, &
     & d_arclength, curvature, r_bc_0, r_bc_1) ! arrays for fixed bc
!
      use chm_kinds
!
         integer :: n
         real(chm_real) :: rin(n), drout(n), wgt(n)
         real(chm_real) :: d_arclength(:), curvature(:)
         real(chm_real), optional :: r_bc_0(n), r_bc_1(n)
        end subroutine compute_dr
       end interface
!
       if (MPI_COMM_STRNG.ne.MPI_COMM_NULL.and. &
     & SIZE_STRNG.gt.1) then
        if (cv%num_cv.gt.0) then
         if (cv_common_fixed_0_bc.eq.1) then
          if (cv_common_fixed_1_bc.eq.1) then
           call compute_dr(RIN=cv%r(1:cv%num_cv,main), &
     & DROUT=cv%r(1:cv%num_cv,dz), WGT=cv%weight, &
     & N=cv%num_cv,D_ARCLENGTH=cv%ds,CURVATURE=cv%curvature, &
     & R_BC_0=cv%r_bc_0, R_BC_1=cv%r_bc_1)
          else
           call compute_dr(RIN=cv%r(1:cv%num_cv,main), &
     & DROUT=cv%r(1:cv%num_cv,dz),WGT=cv%weight, &
     & N=cv%num_cv,D_ARCLENGTH=cv%ds,CURVATURE=cv%curvature, &
     & R_BC_0=cv%r_bc_0)
          endif
         else
          if (cv_common_fixed_1_bc.eq.1) then
           call compute_dr(RIN=cv%r(1:cv%num_cv,main), &
     & DROUT=cv%r(1:cv%num_cv,dz),WGT=cv%weight, &
     & N=cv%num_cv,D_ARCLENGTH=cv%ds,CURVATURE=cv%curvature, &
     & R_BC_1=cv%r_bc_1)
          else
           call compute_dr(RIN=cv%r(1:cv%num_cv,main), &
     & DROUT=cv%r(1:cv%num_cv,dz),WGT=cv%weight, &
     & N=cv%num_cv,D_ARCLENGTH=cv%ds,CURVATURE=cv%curvature)
          endif
         endif
        endif
       endif ! root nodes
!
! send to slaves
       if (MPI_COMM_LOCAL.ne.MPI_COMM_NULL.and.SIZE_LOCAL.gt.1) then
#if (KEY_SINGLE==1)
        call PSND4(cv%r(1:cv%num_cv,dz),cv%num_cv)
        call PSND4(cv%ds,nstring-1)
        call PSND4(cv%curvature,nstring-2)
#else
        call PSND8(cv%r(1:cv%num_cv,dz),cv%num_cv)
        call PSND8(cv%ds,nstring-1)
        call PSND8(cv%curvature,nstring-2)
#endif
       endif
!
       cv_common_dz_initialized=.true.
! cv%r(1,dz) contains the vector tangent to cv%r(:,1) scaled
! by wgt
       end subroutine cv_common_compute_dr
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine cv_common_print_dr(iunit) ! global print
       integer :: iunit
       call cv_common_print_global(iunit,dz,.false.) ! do not print BC
       end subroutine cv_common_print_dr
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine cv_common_read_dr(iunit) ! global print
       integer :: iunit
! local
       integer :: bc0, bc1
! use a trick to read the tangent file
! note: cannot change BC between runs, because the tangents to the BC are not stored!
! in that case we will need to recompute using cv_common_compute_dr()
! for example, you cannot run off-path sampling with all replicas and then switch
! to fixed endponits, reading the tangent vector from the old file
       bc0=cv_common_fixed_0_bc; bc1=cv_common_fixed_1_bc ! store bc info
       cv_common_fixed_0_bc=0; cv_common_fixed_1_bc=0
       call cv_common_read_global(iunit,dz,.true.) ! make sure everything gets read from the dr file
! restore bc
       cv_common_fixed_0_bc=bc0; cv_common_fixed_1_bc=bc1 ! store bc info
!
       cv_common_dz_initialized=.true.
       end subroutine cv_common_read_dr
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine cv_common_repa(interp_method, def, iterations, dst_c)
       use sm_var, only: nstring
       use stream
       use multicom_aux;
       use mpi
       use parallel, only: psnd4, psnd8
!
#if (KEY_PARALLEL==1)
#if (KEY_SINGLE==1)
 integer :: mpifloat=MPI_REAL 
#endif
#if (KEY_SINGLE==0)
 integer :: mpifloat=MPI_REAL8 
#endif
#if (KEY_INTEGER8==0)
 integer :: mpiint=MPI_INTEGER 
#endif
#if (KEY_INTEGER8==1)
 integer :: mpiint=MPI_INTEGER8 
#endif
 integer :: mpichar=MPI_CHARACTER
 integer :: mpibool=MPI_LOGICAL
#endif
!
       integer :: interp_method, iterations, ierror
       real(chm_real), optional :: dst_c
       real(chm_real) :: def
       character(len=len("CV_COMMON_REPA>") ),parameter::whoami="CV_COMMON_REPA>";!macro
!
! interfaces to reparameterization routine
! needed because of the keyword calls
!
       interface
        subroutine interp_driver_sci(rin,rout,wgt,n, &
     & interp_method,tol,max_iterations,d_arclength, curvature, &
     & dst_cutoff,dr,r_bc_0, r_bc_1)
!
      use chm_kinds
!
        integer n
        real(chm_real) rin(n), rout(n), wgt(n)
        integer, intent(in) :: interp_method
        integer max_iterations
        real(chm_real) :: tol, d_arclength(:), curvature(:)
        real(chm_real), optional :: dst_cutoff
        real(chm_real) , optional :: dr(n), r_bc_0(n), r_bc_1(n)
        end subroutine interp_driver_sci
!
        subroutine interp_linear_exact(rin,rout,wgt,n, &
     & d_arclength, curvature, &
     & drout, &
     & r_bc_0, r_bc_1)
!
      use chm_kinds
!
        integer :: n
        real(chm_real) :: rin(n), rout(n), wgt(n)
        real(chm_real) :: d_arclength(:), curvature(:)
        real(chm_real), optional :: drout(n) ! optional computation of tangent
        real(chm_real) , optional :: r_bc_0(n), r_bc_1(n)
       end subroutine interp_linear_exact
!
       end interface
!
       if (.not.cv_common_weights_initialized) then
        call wrndie(0,whoami,trim('CV WEIGHTS NOT INITIALIZED'))
       endif
!
       if (MPI_COMM_STRNG.ne.MPI_COMM_NULL.and. &
     & SIZE_STRNG.gt.1) then
        if (cv%num_cv.gt.0) then ! in-place repa
         if (present(dst_c)) then
          if (cv_common_fixed_0_bc.eq.1) then
           if (cv_common_fixed_1_bc.eq.1) then
            call interp_driver_sci(RIN=cv%r(1:cv%num_cv,main), &
     & ROUT=cv%r(1:cv%num_cv,main),WGT=cv%weight, &
     & N=cv%num_cv,INTERP_METHOD=interp_method,TOL=def, &
     & MAX_ITERATIONS=iterations,D_ARCLENGTH=cv%ds, &
     & DR=cv%r(1:cv%num_cv,dz), &
     & CURVATURE=cv%curvature,DST_CUTOFF=dst_c, R_BC_0=cv%r_bc_0, &
     & R_BC_1=cv%r_bc_1)
           else
            call interp_driver_sci(RIN=cv%r(1:cv%num_cv,main), &
     & ROUT=cv%r(1:cv%num_cv,main),WGT=cv%weight, &
     & N=cv%num_cv,INTERP_METHOD=interp_method,TOL=def, &
     & MAX_ITERATIONS=iterations,D_ARCLENGTH=cv%ds, &
     & DR=cv%r(1:cv%num_cv,dz), &
     & CURVATURE=cv%curvature,DST_CUTOFF=dst_c, R_BC_0=cv%r_bc_0)
           endif
          else
           if (cv_common_fixed_1_bc.eq.1) then
            call interp_driver_sci(RIN=cv%r(1:cv%num_cv,main), &
     & ROUT=cv%r(1:cv%num_cv,main),WGT=cv%weight, &
     & N=cv%num_cv,INTERP_METHOD=interp_method,TOL=def, &
     & MAX_ITERATIONS=iterations,D_ARCLENGTH=cv%ds, &
     & DR=cv%r(1:cv%num_cv,dz), &
     & CURVATURE=cv%curvature,DST_CUTOFF=dst_c, R_BC_1=cv%r_bc_1)
           else
            call interp_driver_sci(RIN=cv%r(1:cv%num_cv,main), &
     & ROUT=cv%r(1:cv%num_cv,main),WGT=cv%weight, &
     & N=cv%num_cv,INTERP_METHOD=interp_method,TOL=def, &
     & MAX_ITERATIONS=iterations,D_ARCLENGTH=cv%ds, &
     & DR=cv%r(1:cv%num_cv,dz), &
     & CURVATURE=cv%curvature,DST_CUTOFF=dst_c)
           endif
          endif
         else
          if (cv_common_fixed_0_bc.eq.1) then
           if (cv_common_fixed_1_bc.eq.1) then
            call interp_driver_sci(RIN=cv%r(1:cv%num_cv,main), &
     & ROUT=cv%r(1:cv%num_cv,main),WGT=cv%weight, &
     & N=cv%num_cv,INTERP_METHOD=interp_method,TOL=def, &
     & MAX_ITERATIONS=iterations,D_ARCLENGTH=cv%ds, &
     & DR=cv%r(1:cv%num_cv,dz), &
     & CURVATURE=cv%curvature, R_BC_0=cv%r_bc_0, &
     & R_BC_1=cv%r_bc_1)
           else
            call interp_driver_sci(RIN=cv%r(1:cv%num_cv,main), &
     & ROUT=cv%r(1:cv%num_cv,main),WGT=cv%weight, &
     & N=cv%num_cv,INTERP_METHOD=interp_method,TOL=def, &
     & MAX_ITERATIONS=iterations,D_ARCLENGTH=cv%ds, &
     & DR=cv%r(1:cv%num_cv,dz), &
     & CURVATURE=cv%curvature,R_BC_0=cv%r_bc_0)
           endif
          else
           if (cv_common_fixed_1_bc.eq.1) then
            call interp_driver_sci(RIN=cv%r(1:cv%num_cv,main), &
     & ROUT=cv%r(1:cv%num_cv,main),WGT=cv%weight, &
     & N=cv%num_cv,INTERP_METHOD=interp_method,TOL=def, &
     & MAX_ITERATIONS=iterations,D_ARCLENGTH=cv%ds, &
     & DR=cv%r(1:cv%num_cv,dz), &
     & CURVATURE=cv%curvature,R_BC_1=cv%r_bc_1)
           else
            call interp_driver_sci(RIN=cv%r(1:cv%num_cv,main), &
     & ROUT=cv%r(1:cv%num_cv,main),WGT=cv%weight, &
     & N=cv%num_cv,INTERP_METHOD=interp_method,TOL=def, &
     & MAX_ITERATIONS=iterations,D_ARCLENGTH=cv%ds, &
     & DR=cv%r(1:cv%num_cv,dz), &
     & CURVATURE=cv%curvature)
           endif
          endif
         endif
! add current CV coordinates to running average -- this is now done in elsewhere
! call cv_common_update_ave()
! call moved to smcv_master
! if (cv_common_voronoi_initialized) call cv_common_voronoi_update() ! update V.cell nodes
        endif
       endif
!
! send to slaves
       if (MPI_COMM_LOCAL.ne.MPI_COMM_NULL.and.SIZE_LOCAL.gt.1) then
#if (KEY_SINGLE==1)
        call PSND4(cv%r(1:cv%num_cv,main),cv%num_cv)
        call PSND4(cv%r(1:cv%num_cv,dz),cv%num_cv)
        call PSND4(cv%ds,nstring-1)
        call PSND4(cv%curvature,nstring-2)
#else
        call PSND8(cv%r(1:cv%num_cv,main),cv%num_cv)
        call PSND8(cv%r(1:cv%num_cv,dz),cv%num_cv)
        call PSND8(cv%ds,nstring-1)
        call PSND8(cv%curvature,nstring-2)
#endif
       endif
!
       end subroutine cv_common_repa
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine cv_common_evolve_smcv(dt)
! not parallelized (yet?) because it should be cheaper to execute locally than to broadcast info
! in the future, may compute on roots, then broadcast
      use stream
      use consta
!
! the timestep can be specified optionally
! nskip ( the number of history elements to skip when computing the average) is also optional
       real(chm_real), optional :: dt
! locals
       real(chm_real) :: dummy
       integer :: i
       real(chm_real) :: delt
       character(len=len("CV_COMMON_EVOLVE_SMCV>") ),parameter::whoami="CV_COMMON_EVOLVE_SMCV>";!macro
! do work
       cv%r(:,comp)=cv%r(:,main) ! save current as "old" coordinates in column 2
       if (present(dt)) then; delt=dt; else ; delt=cv%dt; endif
!
       if (cv%num_run_ave.lt.1) then
        call wrndie(0,whoami,trim('NO SLICES IN THE AVERAGE. NOTHING DONE'))
        return
       endif
! compute average F.E. gradient and dump into force set
! assuming that the runnung average has been computed
       do i=1, cv%num_cv
!
        select case(cv%type(i))
        case(dihe_com);
          dummy=(cv%r(i,main) - cv%r(i,runave))
          dummy=modulo(dummy,TWOPI)
          if (dummy.gt.PI) dummy=dummy-TWOPI;
        case(angle_com); ! cannot tell between theta/-theta
          dummy=(abs(cv%r(i,main)) - cv%r(i,runave))
        case default
          dummy=(cv%r(i,main) - cv%r(i,runave))
        end select
!
        cv%r(i,forces)=cv%k(i)*dummy
       enddo
! compute free energy ( uses cv%r(:,forces) )
       call cv_common_compute_fe_fd()
! add fe to the fe running average
       call cv_common_update_fe_ave()
! now evolve string -- take the negative of the FE gradient
! (we are assuming that M has been computed)
! explicit Euler advancement (any reason to use more accuracy, e.g. A-B ?)
!
       do i=1, cv%num_cv
        cv%r(i,main)=cv%r(i,main)-delt*cv%gamma(i)* &
     & dot_product(cv%M(i,:,2),cv%r(:,forces))
! & cv%r(i,forces)
! write(600+whoiam,*) size(cv%M(i,:,1))
! write(600+whoiam,*) cv%r(1:cv%num_cv,forces)
! write(600+whoiam,*)
! & dot_product(cv%M(i,:,1),cv%r(:,forces)),
! & cv%r(i,forces)/cv%weight(i)
       enddo
! now reset history (number of slices in the running average)
       cv%num_run_ave=0
!
       end subroutine cv_common_evolve_smcv
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine cv_common_evolve_sd(dt) ! steepest descent evolution (no M) with temp; compare with cv_common_evolve_smcv
! temperature not added yet
      use stream
      use consta
! the timestep can be specified optionally
! nskip ( the number of history elements to skip when computing the average) is also optional
       real(chm_real), optional :: dt
       real(chm_real) :: dummy
       integer :: i
       real(chm_real) :: delt
       character(len=len("CV_COMMON_EVOLVE_BD>") ),parameter::whoami="CV_COMMON_EVOLVE_BD>";!macro
! do work
       cv%r(:,comp)=cv%r(:,main) ! save current as "old" coordinates in column 2
       if (present(dt)) then; delt=dt; else ; delt=cv%dt; endif
!
       if (cv%num_run_ave.lt.1) then
        call wrndie(0,whoami,trim('NO SLICES IN THE AVERAGE. NOTHING DONE'))
        return
       endif
! compute average F.E. gradient and dump into force set
! assuming that the runnung average has been computed
       do i=1, cv%num_cv
!
        select case(cv%type(i))
        case(dihe_com);
          dummy=(cv%r(i,main) - cv%r(i,runave))
          dummy=modulo(dummy,TWOPI)
          if (dummy.gt.PI) dummy=dummy-TWOPI;
        case(angle_com); ! cannot tell between theta/-theta
          dummy=(abs(cv%r(i,main)) - cv%r(i,runave))
        case default
          dummy=(cv%r(i,main) - cv%r(i,runave))
        end select
!
        cv%r(i,forces)=cv%k(i)*dummy
       enddo
! compute free energy ( uses cv%r(:,forces) )
       call cv_common_compute_fe_fd()
! add fe to the fe running average
       call cv_common_update_fe_ave()
! now evolve string -- take the negative of the FE gradient
!
       do i=1, cv%num_cv
        cv%r(i,main)=cv%r(i,main)-delt*cv%gamma(i)* &
! & dot_product(cv%M(i,:,2),cv%r(:,forces))
     & cv%r(i,forces) ! + temp (coming soon)
       enddo
! now reset history (number of slices in the running average)
       cv%num_run_ave=0
!
       end subroutine cv_common_evolve_sd
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine cv_common_evolve_expo(a,nskip)
! evolve CV using equation: z(n+1)=a * z(n) + (1-a) * <theta(x)>
      use stream
      use consta
!

       real(chm_real) :: a, a1
       integer, optional :: nskip
! locals
       real(chm_real) :: dummy, u, v
       integer :: numskip, i, j, k, n, k1, k2, k3, k4
       character(len=len("CV_COMMON_EVOLVE_EXPO>") ),parameter::whoami="CV_COMMON_EVOLVE_EXPO>";!macro
       real(chm_real) :: temp(max_hist_base)=0.0d0
!
! do work
! only valid for 0 <= a <= 1
       if (a.lt.0.or.a.gt.1) then
        call wrndie(0,whoami,trim('MEMORY PARAMETER MUST BE BETWEEN 0 AND 1'))
        return
       endif
       if (present(nskip)) then ; numskip=nskip; else; numskip=0; endif
       if (numskip.ge.cv%num_hist) then
        call wrndie(0,whoami,trim('SKIPPED ALL ELEMENTS. NOTHING DONE'))
        return
       endif
       if (cv%num_run_ave.lt.1) then
        call wrndie(0,whoami,trim('NO SLICES IN THE AVERAGE. NOTHING DONE'))
        return
       endif
! 12.08.08: made modifications: 1) compute forces 2) evolve using runave
! from Euler evolve:
! compute average F.E. gradient and dump into force set
! assuming that the runnung average has been computed
       do i=1, cv%num_cv
!
        select case(cv%type(i))
        case(dihe_com);
          dummy=(cv%r(i,main) - cv%r(i,runave))
          dummy=modulo(dummy,TWOPI)
          if (dummy.gt.PI) dummy=dummy-TWOPI;
        case(angle_com); ! cannot tell between theta/-theta
          dummy=(abs(cv%r(i,main)) - cv%r(i,runave))
        case default
          dummy=(cv%r(i,main) - cv%r(i,runave))
        end select
! compute forces in case that we are running restrained dynamics
        cv%r(i,forces)=cv%k(i)*dummy
       enddo
! compute free energy ( uses cv%r(:,forces) )
       call cv_common_compute_fe_fd()
! add fe to the fe running average
       call cv_common_update_fe_ave()
!
! cv%r(:,:,comp)=cv%r(:,:,main) ! save current as "old" coordinates in column 2
! precompute indices for faster copying
       k1=cv%beg_hist+main_offset
       k2=max_hist_base+main_offset
       k3=main_offset+1
       k4=k1-1
!cc
       a1=1d0-a
       n=cv%num_hist-numskip ! number of elements
       do i=1, cv%num_cv
        temp=(/ (cv%r(i,k),k=k1,k2),(cv%r(i,k),k=k3,k4) /) ! 'unwrap' history
!c need CV coordinates that are consistent with the current theta(x): simply take the last slice from above:
! cv%r(i,comp)=temp(cv%num_hist)
! (?) replace above line with :
! cv%r(i,comp)=cv%r(i,instant) ! this is better for alternating restrained/unrestrained equilibration
        cv%r(i,comp)=cv%r(i,main) ! better for restraints
!===== evolve i_th cv ======
!c cv%r(i,main)= a * cv%r(i,main) +
!c & a1 * sum(temp(numskip+1:cv%num_hist))/n

! VO 3.24.09: angle correction
        u=cv%r(i,main)
        v=u-cv%r(i,runave)
!
        select case(cv%type(i));
         case(dihe_com, angle_com, anglvec);
          v=modulo(v,TWOPI)
          if (v.gt.PI) v=v-TWOPI
        end select
!
        cv%r(i,main)= u - &
     & a1 * v
!cccccc from Euler evolve cccccccccccc
! compute average F.E. gradient and dump into force set
! assuming that the runnung average has been computed
! not clear what to do with M in this evolution scheme :
! on the one hand, M should be reset and recomputed when the cv`s change
! on the other, in fts-voronoi, I think we want to have a slowly varying M that includes
! a long-term average, otherwise the CVs (path) could converge, but not M, because it is reset !
! thus, store a long-term average in M(4)
! M:
        do j=1,i
         cv%M(i,j,4) = a * cv%M(i,j,4) + a1 * cv%M(i,j,2);
         cv%M(j,i,4) = cv%M(i,j,4)
        enddo ! j
       enddo ! i
!c reset history (necessary)
       call cv_common_clear_hist()
       cv%num_run_ave=0
! update voronoi cells
! call moved to smcv_master
! if (cv_common_voronoi_initialized) call cv_common_voronoi_update() ! update V.cell nodes
       end subroutine cv_common_evolve_expo
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine cv_common_set_bc(first,last)
! tells the module whether collective string operations include
! virtual fixed CV sets; virtual because the CV are not updated;
! can be viewed as boundary conditions on the string

       logical first, last
! clean up first
       call cv_common_done()
!
       if (first) then
        cv_common_fixed_0_bc=1
       else
        cv_common_fixed_0_bc=0
       endif
!
       if (last) then
        cv_common_fixed_1_bc=1
       else
        cv_common_fixed_1_bc=0
       endif
       end subroutine cv_common_set_bc
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       function cv_common_add_hist()
       use stream

       integer :: cv_common_add_hist
       character(len=len("CV_COMMON_ADD_HIST>") ),parameter::whoami="CV_COMMON_ADD_HIST>";!macro
! returns index into cv%r array; the addition is done elsewhere
       if (cv%num_cv.eq.0) then
         call wrndie(0,whoami,trim('NO CV DEFINED. NOTHING DONE'));
         return
       endif
!
       if (cv%num_hist.eq.max_hist_base) then ! maximum storage reached
        cv%beg_hist=mod(cv%beg_hist, max_hist_base)+1 ! wrap around (avoids copying)
        cv%end_hist=mod(cv%end_hist, max_hist_base)+1
       else
        cv%end_hist=cv%end_hist+1 ! location of the last slice
        cv%num_hist=cv%num_hist+1 ! total number of slices in history
       endif
       cv_common_add_hist=cv%end_hist+main_offset ! offset the reference coords
       end function cv_common_add_hist
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine cv_common_smooth_hist(d,nskip) ! not used in regular SMCV
! filter the history of CV values so as to obtain new restraining sets
! the new set is dumped into the main set (1)
! filter width delta can be specified optionally, default value is 10 (arbitrary)
! nskip (the number of history elements to skip when computing the average) is also optional
! this algorithm is entirely heuristic
!
       use stream
       use consta
!
       integer, optional :: d
       integer, optional :: nskip
! locals
       integer :: delta
       integer :: numskip, i, j, k, n, k1, k2, k3, k4
       real(chm_real) :: t(max_hist_base)=(/(1.0d0*i,i=1,max_hist_base)/)
       real(chm_real) :: temp(max_hist_base)=0.0d0, u ! for smoothing
       character(len=len("CV_COMMON_SMOOTH_HIST>") ),parameter::whoami="CV_COMMON_SMOOTH_HIST>";!macro
! interface to smooth2
!
       interface
        function smooth2(t,x,delta)
!
      use chm_kinds
!
        integer :: delta
        real(chm_real) :: t(:), x(:)
        real(chm_real), dimension(size(t)) :: smooth2
        end function smooth2
       end interface
!
! do work
       if (present(d)) then; delta=d; else ; delta=10; endif
       if (present(nskip)) then ; numskip=nskip; else; numskip=0; endif
       if (numskip.ge.cv%num_hist) then
        call wrndie(0,whoami,trim('SKIPPED ALL SLICES. NOTHING DONE'))
        return
       endif
! cv%r(:,comp)=cv%r(:,main) ! save current as "old" coordinates in column 2
! precompute indices for faster copying
       k1=cv%beg_hist+main_offset
       k2=max_hist_base+main_offset
       k3=main_offset+1
       k4=k1-1
!
       n=cv%num_hist-numskip ! number of elements
       do i=1, cv%num_cv
! 'unwrap' history
        temp=(/ (cv%r(i,k),k=k1,k2),(cv%r(i,k),k=k3,k4) /)
!
! VO 3.24.09: angle correction
        select case(cv%type(i));
         case(dihe_com, angle_com, anglvec);
          do j=numskip+2, cv%num_hist
           u=temp(j)-temp(j-1)
           u=modulo(u,TWOPI)
           if (u.gt.PI) u=u-TWOPI
           temp(j)=temp(j-1)+u
          enddo
        end select
!
        temp(1:n)=smooth2( &
     & t(numskip+1:cv%num_hist), &
     & temp(numskip+1:cv%num_hist), &
     & delta)
        cv%r(i,main)=temp(n) ! set the restraint value (z) to the
        cv%r(i,comp)=temp(n) ! the theta(x) are roughly consistent with these CV
! M: what to do about M in this case is unclear
! for now, setting the long-term average to the short-term evarage
        do j=1,i
          cv%M(i,j,4)=cv%M(i,j,2); cv%M(j,i,4)=cv%M(i,j,2)
        enddo
       enddo
! reset history (necessary)
       call cv_common_clear_hist()
       cv%num_run_ave=0
       end subroutine cv_common_smooth_hist
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine cv_common_print_hist_local(iunit, nskip) ! not used in regular SMCV
! assume that unit is prepared
       use stream

       integer :: iunit
       integer, optional :: nskip
! locals
       integer :: i, j, k, kbeg, kend, numskip
       character(len=80) :: fmt
       character(len=len("CV_COMMON_PRINT_HIST>") ),parameter::whoami="CV_COMMON_PRINT_HIST>";!macro

! do work
       if (present(nskip)) then ; numskip=nskip; else; numskip=0; endif
       if (numskip.ge.cv%num_hist) then
        call wrndie(0,whoami,trim('SKIPPED ALL ELEMENTS. NOTHING PRINTED'))
        return
       endif

       write(fmt,int_format) cv%num_cv
       kbeg=cv%beg_hist+numskip
       kend=cv%beg_hist+cv%num_hist ! 'unwrapped' end
       do k=kbeg, kend
        j=mod(k-1,max_hist_base)+1+main_offset
         write(iunit,'('//fmt//real_format//')')                        &
     & (cv%r(i,j),i=1,cv%num_cv) ! each line corresponds to a time slice
       enddo
       end subroutine cv_common_print_hist_local
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine cv_common_print_hist_global(iunit, nskip) ! not used in regular SMCV
! assume that unit is prepared
       use stream
       use mpi
       use multicom_aux;
!
#if (KEY_PARALLEL==1)
#if (KEY_SINGLE==1)
 integer :: mpifloat=MPI_REAL 
#endif
#if (KEY_SINGLE==0)
 integer :: mpifloat=MPI_REAL8 
#endif
#if (KEY_INTEGER8==0)
 integer :: mpiint=MPI_INTEGER 
#endif
#if (KEY_INTEGER8==1)
 integer :: mpiint=MPI_INTEGER8 
#endif
 integer :: mpichar=MPI_CHARACTER
 integer :: mpibool=MPI_LOGICAL
#endif
!
       integer :: iunit
       integer, optional :: nskip
! locals
       integer :: i, j, k, kbeg, kend, numskip, kout
       integer :: ierror
       character(len=80) :: fmt
       real(chm_real), pointer :: rtemp(:,:)
       real(chm_real), pointer :: rall(:,:,:)
       logical :: qroot
       character(len=len("CV_COMMON_PRINT_HIST>") ),parameter::whoami="CV_COMMON_PRINT_HIST>";!macro
!
! do work
       qroot=(MPI_COMM_STRNG.ne.MPI_COMM_NULL)
!
       if (qroot) then
        if (present(nskip)) then ; numskip=nskip; else; numskip=0; endif
        if (numskip.ge.cv%num_hist) then
         call wrndie(0,whoami,trim('SKIPPED ALL ELEMENTS. NOTHING PRINTED'))
         return
        endif
!
        kbeg=cv%beg_hist+numskip
        kend=cv%beg_hist+cv%num_hist-1 ! 'unwrapped' end
        kout=kend-kbeg+1
! allocate data array
        if (kout.gt.0) then
         allocate(rall(cv%num_cv,kout,SIZE_STRNG))
         allocate(rtemp(cv%num_cv,kout))
        else
         call wrndie(0,whoami,trim('SKIPPED ALL ELEMENTS. NOTHING PRINTED'))
         return
        endif ! kout
!
        do k=1, kout
         j=mod(k+kbeg-2,max_hist_base)+1+main_offset
         rtemp(:,k)=cv%r(1:cv%num_cv,j) ! each line corresponds to a time slice
        enddo
! gather all data on root (0)
        call MPI_GATHER(rtemp,cv%num_cv*kout,mpifloat, &
     & rall,cv%num_cv*kout,mpifloat,0,MPI_COMM_STRNG, &
     & ierror)
!
        if (ME_STRNG.eq.0) then ! root writes
         write(fmt,int_format) cv%num_cv
         do k=1,kout
          write(iunit,'("% ",'//int_format//')') k
          do i=1,cv%num_cv
           write(iunit,'('//fmt//real_format//')')                      &
     & (rall(i,k,j),j=1,SIZE_STRNG)
          enddo
         enddo
        endif ! ME
!
        if(associated(rtemp))deallocate(rtemp)
        if(associated(rall))deallocate(rall)
       endif ! qroot
       end subroutine cv_common_print_hist_global
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine cv_common_clear_hist()
       cv%num_hist=0; cv%beg_hist=1; cv%end_hist=0
       end subroutine cv_common_clear_hist
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine cv_common_print_local(iunit,col)
! assume that unit is prepared
! NOTE that this is a local print!
! does not print fixed BC
       use stream
       integer iunit
       integer, optional :: col
! locals
       integer :: i, c
       character(len=len("CV_COMMON_PRINT_LOCAL>") ),parameter::whoami="CV_COMMON_PRINT_LOCAL>";!macro
! do work
       if (.not.present(col)) then ; c=main ; else ; c=col ; endif
       if (c.lt.1.or.c.gt.(max_hist_base+main_offset)) then
        call wrndie(0,whoami,trim('OUT OF BOUNDS. NOTHING DONE.'))
        return
       endif
       do i=1, cv%num_cv
        write(iunit, real_format) cv%r(i,c)
       enddo
       end subroutine cv_common_print_local
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine cv_common_print_global(iunit,col,print_bc)
! assume that unit is prepared
! NOTE that this is a global print!
       use stream
       use multicom_aux;
       use mpi
!
#if (KEY_PARALLEL==1)
#if (KEY_SINGLE==1)
 integer :: mpifloat=MPI_REAL 
#endif
#if (KEY_SINGLE==0)
 integer :: mpifloat=MPI_REAL8 
#endif
#if (KEY_INTEGER8==0)
 integer :: mpiint=MPI_INTEGER 
#endif
#if (KEY_INTEGER8==1)
 integer :: mpiint=MPI_INTEGER8 
#endif
 integer :: mpichar=MPI_CHARACTER
 integer :: mpibool=MPI_LOGICAL
#endif
!
       integer iunit
       integer, optional :: col
       logical, optional :: print_bc
! locals
       integer :: i, k, c
       logical :: bc
       character(len=80) :: fmt
       real(chm_real), pointer :: rtemp(:,:) ! temporary array for holding coords
       integer :: ierror
       logical :: qroot
       character(len=len("CV_COMMON_PRINT_GLOBAL>") ),parameter::whoami="CV_COMMON_PRINT_GLOBAL>";!macro
! do work
       qroot=(MPI_COMM_STRNG.ne.MPI_COMM_NULL)
       if (qroot) then
!
        if (.not.present(col)) then ; c=main ; else ; c=col ; endif
        if (c.lt.1.or.c.gt.(cv%num_hist+main_offset)) then
         call wrndie(0,whoami,trim('OUT OF BOUNDS. NOTHING DONE.'))
         return
        endif
!
        bc=.true.
        if (present(print_bc)) bc=print_bc ! do we want bc printed if applicable?
!
! gather all data on root
        allocate(rtemp(cv%num_cv, SIZE_STRNG))
!
        call MPI_GATHER(cv%r(1,c),cv%num_cv,mpifloat, &
     & rtemp,cv%num_cv,mpifloat,0,MPI_COMM_STRNG, &
     & ierror)
!
        if (ME_STRNG.eq.0) then ! root writes
!
         if (bc) then ! write inner replicas + bc
          write(fmt,int_format) &
     & (SIZE_STRNG+cv_common_fixed_0_bc+cv_common_fixed_1_bc)
          if (cv_common_fixed_0_bc.eq.1) then
           if (cv_common_fixed_1_bc.eq.1) then
            do i=1,cv%num_cv
             write(iunit,'('//fmt//real_format//')')                    &
     & cv%r_bc_0(i), (rtemp(i,k),k=1,SIZE_STRNG), cv%r_bc_1(i)
            enddo
           else
            do i=1,cv%num_cv
             write(iunit,'('//fmt//real_format//')')                    &
     & cv%r_bc_0(i), (rtemp(i,k),k=1,SIZE_STRNG)
            enddo
           endif
          else
           if (cv_common_fixed_1_bc.eq.1) then
            do i=1,cv%num_cv
             write(iunit,'('//fmt//real_format//')')                    &
     & (rtemp(i,k),k=1,SIZE_STRNG), cv%r_bc_1(i)
            enddo
           else
            do i=1,cv%num_cv
             write(iunit,'('//fmt//real_format//')')                    &
     & (rtemp(i,k),k=1,SIZE_STRNG)
            enddo
           endif
          endif
         else ! bc; just write the inner replicas
          write(fmt,int_format) SIZE_STRNG
          do i=1,cv%num_cv
           write(iunit,'('//fmt//real_format//')')                      &
     & (rtemp(i,k),k=1,SIZE_STRNG)
          enddo
         endif ! print_bc
        endif ! ME=0
        deallocate(rtemp)
       endif ! qroot
!
       end subroutine cv_common_print_global
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine cv_common_unwrap_angles(col,ind)
       use sm_var, only: nstring, mestring ! all cpus need to know string length
       use stream
       use multicom_aux;
       use consta
       use mpi
!
#if (KEY_PARALLEL==1)
#if (KEY_SINGLE==1)
 integer :: mpifloat=MPI_REAL 
#endif
#if (KEY_SINGLE==0)
 integer :: mpifloat=MPI_REAL8 
#endif
#if (KEY_INTEGER8==0)
 integer :: mpiint=MPI_INTEGER 
#endif
#if (KEY_INTEGER8==1)
 integer :: mpiint=MPI_INTEGER8 
#endif
 integer :: mpichar=MPI_CHARACTER
 integer :: mpibool=MPI_LOGICAL
#endif
!
       integer, optional :: col, ind
! locals
       integer :: ibeg, iend
       integer :: i, j, c, nrep
       real(chm_real) :: dangle
       real(chm_real), pointer :: rtemp(:,:) ! temporary array for holding coords
       integer :: ierror
       character(len=len("CV_COMMON_UNWRAP_ANGLES>") ),parameter::whoami="CV_COMMON_UNWRAP_ANGLES>";!macro
! do work
       if (.not.present(col)) then ; c=main ; else ; c=col ; endif
       if (c.lt.1.or.c.gt.(cv%num_hist+main_offset)) then
        call wrndie(0,whoami,trim('OUT OF BOUNDS. NOTHING DONE.'))
        return
       endif
!
       if (.not.present(ind)) then
        ibeg=1 ; iend=cv%num_cv
       else
        if (ind.lt.1.or.ind.gt.cv%num_cv) then
         call wrndie(0,whoami,trim('OUT OF BOUNDS. NOTHING DONE.'))
         return
        endif
        ibeg=ind; iend=ind ;
       endif
!
       if (nstring.le.1) return
!
       nrep=nstring+cv_common_fixed_0_bc+cv_common_fixed_1_bc
       allocate(rtemp(cv%num_cv, nrep))
! gather all data on each root processor
       if (MPI_COMM_STRNG.ne.MPI_COMM_NULL.and.SIZE_STRNG.gt.1) &
     & call MPI_ALLGATHER(cv%r(1,c),cv%num_cv,mpifloat, &
     & rtemp(1,1+cv_common_fixed_0_bc), &
     & cv%num_cv,mpifloat,MPI_COMM_STRNG, &
     & ierror)
! broadcast rtemp to slaves:
       if (MPI_COMM_LOCAL.ne.MPI_COMM_NULL.and.SIZE_LOCAL.gt.1) &
     & call mpi_bcast(rtemp,cv%num_cv*nrep,mpifloat,0,MPI_COMM_LOCAL,ierror)
!
! loop over specified cv indices
       do i=ibeg, iend
        select case(cv%type(i))
         case(dihe_com, angle_com, anglvec);
!
          do j=2,nrep
           dangle=modulo(rtemp(i,j)-rtemp(i,j-1), TWOPI)
           if (dangle .gt. PI ) dangle = dangle - TWOPI
           rtemp(i,j)=rtemp(i,j-1)+dangle
          enddo
          cv%r(i,c)=rtemp(i,mestring+1+cv_common_fixed_0_bc)
          if (cv_common_fixed_1_bc.gt.0) cv%r_bc_1(i)=rtemp(i,nrep)
!
        end select
       enddo
! free memory
       deallocate(rtemp)
       end subroutine cv_common_unwrap_angles
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine cv_common_compute_fe_fd()
       use sm_var, only: nstring
       use stream
       use multicom_aux;
       use mpi
       use parallel, only: psnd4, psnd8
!
       integer :: ierror
       character(len=len("CV_COMMON_COMPUTE_FE_FD>") ),parameter::whoami="CV_COMMON_COMPUTE_FE_FD>";!macro
! define interface for work routine (calling by keyword)
       interface
        subroutine compute_work_fd(r,rbc0,rbc1,f,n,fe)
!
      use chm_kinds
!
        integer :: n
        real(chm_real) :: r(n), f(n)
        real(chm_real) :: fe(:)
        real(chm_real), optional :: rbc0(n), rbc1(n)
        end subroutine compute_work_fd
       end interface
!
       if (MPI_COMM_STRNG.ne.MPI_COMM_NULL.and. &
     & SIZE_STRNG.gt.1) then
        if (cv%num_cv.eq.0) &
     & call wrndie(0,whoami,trim('NO CV DEFINED.'))
        if (cv_common_fixed_0_bc.eq.1) then
         if (cv_common_fixed_1_bc.eq.1) then
          call compute_work_fd( &
     & R=cv%r(1:cv%num_cv,main),RBC0=cv%r_bc_0,RBC1=cv%r_bc_1, &
     & F=cv%r(1:cv%num_cv,forces),N=cv%num_cv,FE=cv%fe)
         else
          call compute_work_fd( &
     & R=cv%r(1:cv%num_cv,main),RBC0=cv%r_bc_0, &
     & F=cv%r(1:cv%num_cv,forces),N=cv%num_cv,FE=cv%fe)
         endif
        else
         if (cv_common_fixed_1_bc.eq.1) then
          call compute_work_fd( &
     & R=cv%r(1:cv%num_cv,main),RBC1=cv%r_bc_1, &
     & F=cv%r(1:cv%num_cv,forces),N=cv%num_cv,FE=cv%fe)
         else
          call compute_work_fd( &
     & R=cv%r(1:cv%num_cv,main), &
     & F=cv%r(1:cv%num_cv,forces),N=cv%num_cv,FE=cv%fe)
         endif
        endif
       endif
!
! send to slaves
       if (MPI_COMM_LOCAL.ne.MPI_COMM_NULL.and.SIZE_LOCAL.gt.1) then
#if (KEY_SINGLE==1)
        call PSND4(cv%r(1:cv%num_cv,forces),cv%num_cv)
        call PSND4(cv%fe,nstring)
#else
        call PSND8(cv%r(1:cv%num_cv,forces),cv%num_cv)
        call PSND8(cv%fe,nstring)
#endif
       endif
!
       end subroutine cv_common_compute_fe_fd
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine cv_common_print_forces(iunit) ! global
       integer :: iunit
       call cv_common_print_global(iunit,forces,.false.) ! do not print BC if they are present
       end subroutine cv_common_print_forces
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine cv_common_read_local(iunit,col)
! assume that unit is prepared
! for parallel, broadcast to slaves done by calling routine
       use stream
       integer :: iunit
       integer, optional :: col
       integer :: i, c
       character(len=len("CV_COMMON_READ_LOCAL>") ),parameter::whoami="CV_COMMON_READ_LOCAL>";!macro
! do work
! character(len=80) :: fmt
! write(fmt,int_format) max_cv_common
       if (.not.present(col)) then ; c=main ; else ; c=col ; endif
       if (c.lt.1.or.c.gt.(max_hist_base+main_offset)) then
        call wrndie(0,whoami,trim('OUT OF BOUNDS. NOTHING DONE.'))
        return
       endif
       do i=1,cv%num_cv
! read(iunit,'('//fmt//'F11.5)')
        read(iunit,*) & ! free-form read in case you want to edit entries by hand
     & cv%r(i,c)
       enddo
       end subroutine cv_common_read_local
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine cv_common_read_global(iunit,col,read_bc)
! assume that unit is prepared
       use sm_var, only : nstring
       use string
       use stream
       use multicom_aux;
       use number
       use mpi
!
#if (KEY_PARALLEL==1)
#if (KEY_SINGLE==1)
 integer :: mpifloat=MPI_REAL 
#endif
#if (KEY_SINGLE==0)
 integer :: mpifloat=MPI_REAL8 
#endif
#if (KEY_INTEGER8==0)
 integer :: mpiint=MPI_INTEGER 
#endif
#if (KEY_INTEGER8==1)
 integer :: mpiint=MPI_INTEGER8 
#endif
 integer :: mpichar=MPI_CHARACTER
 integer :: mpibool=MPI_LOGICAL
#endif
!
       integer :: iunit
       integer, optional :: col
       logical, optional :: read_bc
       integer :: i, k, c
       real(chm_real) :: rtemp(max_cv_common, nstring), dummy ! temporary array for holding coords
       integer :: ierror
       character(len=8) :: me ! for output
       logical :: bc
       logical :: qread
       character(len=len("CV_COMMON_READ_GLOBAL>") ),parameter::whoami="CV_COMMON_READ_GLOBAL>";!macro
!
       qread=(MPI_COMM_STRNG.ne.MPI_COMM_NULL.and.ME_STRNG.eq.0)
! do work
       bc=.false.
       rtemp=anum ! initialize
! character(len=80) :: fmt
! write(fmt,int_format) max_cv_common
       if (.not.present(col)) then ; c=main ; else ; c=col ; endif
       if (c.lt.1.or.c.gt.(cv%num_hist+main_offset)) then
        call wrndie(0,whoami,trim('OUT OF BOUNDS. NOTHING DONE.'))
        return
       endif
! read file
       if (c.eq.main) bc=.true. ! read bc if reading first (main coordinates)
       if (present(read_bc)) bc=read_bc ! do we want to read bc if applicable?
!
       if (qread) then ! root reads
!
        if (bc) then ! read inner replicas + bc
         if (cv_common_fixed_0_bc.eq.1) then
          if (cv_common_fixed_1_bc.eq.1) then
           do i=1,cv%num_cv
            read(iunit,*) &
     & cv%r_bc_0(i), (rtemp(i,k),k=1,SIZE_STRNG), cv%r_bc_1(i)
           enddo
          else
           do i=1,cv%num_cv
            read(iunit,*) &
     & cv%r_bc_0(i), (rtemp(i,k),k=1,SIZE_STRNG)
           enddo
          endif
         else
          if (cv_common_fixed_1_bc.eq.1) then
           do i=1,cv%num_cv
            read(iunit,*) &
     & (rtemp(i,k),k=1,SIZE_STRNG), cv%r_bc_1(i)
           enddo
          else
           do i=1,cv%num_cv
            read(iunit,*) &
     & (rtemp(i,k),k=1,SIZE_STRNG)
           enddo
          endif
         endif
        else ! just read the inner replicas and throw away the rest
         if (cv_common_fixed_0_bc.eq.1) then
          if (cv_common_fixed_1_bc.eq.1) then
           do i=1,cv%num_cv
            read(iunit,*) &
     & dummy, (rtemp(i,k),k=1,SIZE_STRNG), dummy
           enddo
          else
           do i=1,cv%num_cv
            read(iunit,*) &
     & dummy,(rtemp(i,k),k=1,SIZE_STRNG)
           enddo
          endif
         else
          if (cv_common_fixed_1_bc.eq.1) then
           do i=1,cv%num_cv
            read(iunit,*) &
     & (rtemp(i,k),k=1,SIZE_STRNG), dummy
           enddo
          else
           do i=1,cv%num_cv
            read(iunit,*) &
     & (rtemp(i,k),k=1,SIZE_STRNG)
           enddo
          endif
         endif
        endif ! read_bc
       endif ! root
! scatter all data on root
       if (MPI_COMM_STRNG.ne.MPI_COMM_NULL) then
        call MPI_SCATTER(rtemp, max_cv_common, mpifloat, &
     & cv%r(1,c),max_cv_common,mpifloat,0,MPI_COMM_STRNG, &
     & ierror)
! broadcast BC
        if (bc) then
         if (cv_common_fixed_0_bc.eq.1) &
     & call mpi_bcast(cv%r_bc_0,cv%num_cv,mpifloat,0,MPI_COMM_STRNG,ierror)
         if (cv_common_fixed_1_bc.eq.1) &
     & call mpi_bcast(cv%r_bc_1,cv%num_cv,mpifloat,0,MPI_COMM_STRNG,ierror)
        endif ! bc
! check for zero coordinate entries
        if (any(cv%r(1:cv%num_cv,c).eq.anum)) then
         write(me,'(I8)') ME_STRNG
         i=len(me)
         call trima(me, i)
      write(info(1),*)'SOME CV VALUES ARE UNDEFINED AFTER READING ON REPLICA',me,'.';call wrndie(0,whoami,trim(info(1)))
        endif
       endif ! MPI_COMM_STRNG
!
! broadcast to slaves in calling routine
       end subroutine cv_common_read_global
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine cv_common_read_local_from_global &
     & (iunit,numrep,replica,col)
! read data for a particular specified replica from a global file
! caller needs to provide number of replicas in the file (# cols)
! assume that unit is prepared
      use stream
      use string
      use multicom_aux;
      use number
      use mpi
!
#if (KEY_PARALLEL==1)
#if (KEY_SINGLE==1)
 integer :: mpifloat=MPI_REAL 
#endif
#if (KEY_SINGLE==0)
 integer :: mpifloat=MPI_REAL8 
#endif
#if (KEY_INTEGER8==0)
 integer :: mpiint=MPI_INTEGER 
#endif
#if (KEY_INTEGER8==1)
 integer :: mpiint=MPI_INTEGER8 
#endif
 integer :: mpichar=MPI_CHARACTER
 integer :: mpibool=MPI_LOGICAL
#endif
!
       integer :: iunit
       integer :: numrep ! number of replicas in the CV data file
       integer :: replica ! will read CV values corresponding to 'replica'
       integer, optional :: col
       integer :: i, k, c
       real(chm_real) :: rtemp(max_cv_common, numrep) ! temporary array for holding coords
       integer :: ierror
       character(len=8) :: me, which ! for output
       logical :: qroot
       character(len=len("CV_COMMON_READ_LOCAL_FROM_GLOBAL>") ),parameter::whoami="CV_COMMON_READ_LOCAL_FROM_GLOBAL>";!macro
!
       qroot=(MPI_COMM_STRNG.ne.MPI_COMM_NULL)
!
       rtemp=anum ! initialize
! character(len=80) :: fmt
! write(fmt,int_format) max_cv_common
       if (.not.present(col)) then ; c=main ; else ; c=col ; endif
       if (c.lt.1.or.c.gt.(cv%num_hist+main_offset)) then
        call wrndie(0,whoami,trim('OUT OF BOUNDS. NOTHING DONE.'))
        return
       endif
!
! read file
       if (qroot.and.ME_STRNG.eq.0) then ! root reads, then broadcasts
        do i=1,cv%num_cv
! read(iunit,'('//fmt//'D23.15)')
         read(iunit,*) &
     & (rtemp(i,k),k=1,numrep)
        enddo
       endif
!
! now broadcast CV values to other nodes
       if (qroot) then
        if (SIZE_STRNG.gt.1) &
     & call mpi_bcast(rtemp,max_cv_common*numrep,mpifloat,0,MPI_COMM_STRNG,ierror)
! "pick out" what I need
        if (replica .le. numrep .and. replica .gt. 0) then
         cv%r(1:cv%num_cv,c)=rtemp(1:cv%num_cv,replica)
        else
         write(me,'(I8)') ME_STRNG
         write(which,'(I8)') replica
write(info(1),*)'REPLICA',me,'ATTEMPTED TO READ NONEXISTENT COLUMN',which,'. NOTHING DONE.';call wrndie(0,whoami,trim(info(1)))
        endif ! replica
! check for zero coordinate entries
        if (any(cv%r(1:cv%num_cv,c).eq.0.0d0)) then
         write(me,'(I8)') ME_STRNG
         i=len(me)
         call trima(me, i)
write(info(1),*)'SOME CV COORDINATES ARE ZERO AFTER READING ON REPLICA ',me,'.';call wrndie(0,whoami,trim(info(1)))
        endif
       endif ! root
!
       end subroutine cv_common_read_local_from_global
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine cv_common_print_ds(iunit,fmt)
       use sm_var, only: nstring
       integer iunit
       character(len=*), optional :: fmt
! local
       character(len=80) :: frm
! begin
       if (.not.present(fmt)) then
        write(frm,'('//int_format//',A)')                               &
     & nstring+cv_common_fixed_0_bc+cv_common_fixed_1_bc, real_format
       else
        frm=fmt
       endif
!
       write(iunit,frm) cv%ds
       end subroutine cv_common_print_ds
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine cv_common_print_curvature(iunit,fmt)
       use sm_var, only: smcv_initialized, nstring
       integer iunit
       character(len=*), optional :: fmt
! local
       character(len=80) :: frm
       interface
        subroutine smcv_init(maxcv);; integer, optional :: maxcv ; end subroutine smcv_init
       end interface
! begin
       if (.not.smcv_initialized) call smcv_init()
       if (.not.present(fmt)) then
        write(frm,'('//int_format//',A)')                               &
     & nstring+cv_common_fixed_0_bc+cv_common_fixed_1_bc, real_format
       else
        frm=fmt
       endif
!
       write(iunit,frm) cv%curvature
       end subroutine cv_common_print_curvature
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine cv_common_print_fe(iunit,fmt)
       use sm_var, only: smcv_initialized, nstring
!
       integer iunit
       character(len=*), optional :: fmt
! local
       character(len=80) :: frm
       interface
        subroutine smcv_init(maxcv);; integer, optional :: maxcv ; end subroutine smcv_init
       end interface
! begin
       if (.not.smcv_initialized) call smcv_init()
       if (.not.present(fmt)) then
        write(frm, '('//int_format//',A)')                              &
     & nstring+cv_common_fixed_0_bc+cv_common_fixed_1_bc, real_format
       else
        frm=fmt
       endif
!
       write(iunit,frm) cv%fe
       end subroutine cv_common_print_fe
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine cv_common_print_feav(iunit,fmt)
       use sm_var, only: smcv_initialized, nstring
!
       integer iunit
       character(len=*), optional :: fmt
! local
       character(len=80) :: frm
! begin
       interface
        subroutine smcv_init(maxcv);; integer, optional :: maxcv ; end subroutine smcv_init
       end interface
       if (.not.smcv_initialized) call smcv_init()
       if (.not.present(fmt)) then
        write(frm, '('//int_format//',A)')                              &
     & nstring+cv_common_fixed_0_bc+cv_common_fixed_1_bc, real_format
       else
        frm=fmt
       endif
       write(iunit,frm) cv%feav
! reset number of fe samples so that the averaging starts over
       cv%num_fe_samples=0
       end subroutine cv_common_print_feav
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine cv_common_set_k(i,k)
       integer :: i
       real(chm_real) :: k
! no error checking: can set to anything you want!
       if (i.ge.0.and.i.le.cv%num_cv) cv%k(i)=k
       end subroutine cv_common_set_k
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine cv_common_set_active(i,flag)
       integer :: i
       logical :: flag
       if (i.ge.0.and.i.le.cv%num_cv) cv%active(i)=flag
       end subroutine cv_common_set_active
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine cv_common_set_kpara(k)
       use stream
       real(chm_real) :: k
!
       character(len=len("CV_COMMON_SET_KPARA>") ),parameter::whoami="CV_COMMON_SET_KPARA>";!macro
       if (k.lt.0d0) &
     &call wrndie(0,whoami,trim(' NEGATIVE FORCE CONSTANT SPECIFIED. NOTHING DONE.'))
       cv%kpara=k
       cv_common_kpara_initialized=.true.
       end subroutine cv_common_set_kpara
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine cv_common_set_kperp(k)
       use stream
       real(chm_real) :: k
!
       character(len=len("CV_COMMON_SET_KPERP>") ),parameter::whoami="CV_COMMON_SET_KPERP>";!macro
       if (k.lt.0d0) &
     & call wrndie(0,whoami,trim(' NEGATIVE FORCE CONSTANT SPECIFIED. NOTHING DONE.'))
       cv%kperp=k
       cv_common_kperp_initialized=.true.
       end subroutine cv_common_set_kperp
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine cv_common_set_g(i,gamma)
       integer :: i
       real(chm_real) :: gamma
!
       if (i.gt.0.and.i.le.cv%num_cv.and.gamma.gt.0d0) & ! disallow neagtive or zero values
     & cv%gamma(i)=1d0/gamma
       end subroutine cv_common_set_g
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine cv_common_set_w(i,weight)
       integer :: i
       real(chm_real) :: weight
!
! if (weight.le.0.0) return ! zero or negative weight breaks
       if (i.gt.0.and.i.le.cv%num_cv) then
        cv%weight(i)=weight
        cv_common_weights_initialized=.true. ! assume that the user will has initialized all weights !
       endif
!
       end subroutine cv_common_set_w
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine cv_common_set_r(i,z,col1)
       use stream
!
       integer :: i
       integer, optional :: col1
       real(chm_real) :: z
!
       integer :: col
       character(len=len("CV_COMMON_SET_R>") ),parameter::whoami="CV_COMMON_SET_R>";!macro
!
       if (present(col1)) then ; col=col1; else ; col=main; endif
       if (col.gt.0.and.col.le.main_offset+max_hist_base) then
        if (i.gt.0.and.i.le.cv%num_cv) then
         cv%r(i,col)=z
        else
         call wrndie(0,whoami,trim('INVALID COLUMN. NOTHING DONE.'))
        endif
       else
        call wrndie(0,whoami,trim('INVALID CV INDEX. NOTHING DONE.'))
       endif
       end subroutine cv_common_set_r
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine cv_common_set_dt(dt)
       real(chm_real) :: dt
! no error checking: can set to anything you want!
! locals
       cv%dt=dt
       end subroutine cv_common_set_dt
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine cv_common_copy(c1, c2)
! be careful -- you can do damage here!
       integer :: c1, c2
! local
       character(len=len("CV_COMMON_COPY>") ),parameter::whoami="CV_COMMON_COPY>";!macro
! start
       if (c1.gt.0.and.c1.le.main_offset+max_hist_base &
     & .and. &
     & c2.gt.0.and.c2.le.main_offset+max_hist_base &
     & .and.c1.ne.c2) then
        cv%r(:,c2)=cv%r(:,c1)
       else
        call wrndie(0,whoami,trim('INVALID COLUMN SPECIFIED'))
       endif
       end subroutine cv_common_copy
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine cv_common_swap(col1, col2)
! be careful -- you can do damage here!
       integer, optional :: col1, col2
       character(len=len("CV_COMMON_SWAP>") ),parameter::whoami="CV_COMMON_SWAP>";!macro
       integer :: c1, c2
! local
       real(chm_real) :: temp(max_cv_common)
! start
       if (present(col1)) then ; c1=col1; else ; c1=main; endif
       if (present(col2)) then ; c2=col2; else ; c2=comp; endif
       if (c1.gt.0.and.c1.le.main_offset+max_hist_base &
     & .and. &
     & c2.gt.0.and.c2.le.main_offset+max_hist_base &
     & .and.c1.ne.c2) then
        temp=cv%r(:,c1);
        cv%r(:,c1)=cv%r(:,c2)
        cv%r(:,c2)=temp
       else
        call wrndie(0,whoami,trim('INVALID COLUMN SPECIFIED'))
       endif
       end subroutine cv_common_swap
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       function cv_common_rmsd(col1, col2)
       use stream
       use consta
       integer, optional :: col1, col2
       integer :: c1, c2
! local
       real(chm_real) :: cv_common_rmsd, dummy, dangle
! real(chm_real), parameter :: pi=3.14159265358979 ! hardwired for angles
       integer :: i
       character(len=len("CV_COMMON_RMSD>") ),parameter::whoami="CV_COMMON_RMSD>";!macro
! start
       if (cv%num_cv.eq.0) then
        call wrndie(0,whoami,trim('NO CV DEFINED. EXITING.'))
        return
       endif
       if (.not.cv_common_weights_initialized) then
        call wrndie(0,whoami,trim('CV WEIGHTS NOT INITIALIZED.'))
       endif
!
! NOTE: might be a good idea to force the (re)computation of weights before this call
! if weights are indeed the mass matrix M.
!
       cv_common_rmsd=0.0
       if (present(col1)) then ; c1=col1; else ; c1=main; endif
       if (present(col2)) then ; c2=col2; else ; c2=comp; endif
       if (c1.gt.0.and.c1.le.(max_hist_base+main_offset) &
     & .and. &
     & c2.gt.0.and.c2.le.(max_hist_base+main_offset)) then
!
        dummy=0d0
        do i=1,cv%num_cv
! this is 'unclean', but for now keep the angle
! hardwiring here
         select case(cv%type(i))
         case(dihe_com);
            dangle=modulo(cv%r(i,c1)-cv%r(i,c2), TWOPI)
            if (dangle .gt. PI ) dangle = dangle - TWOPI
            dummy=dummy+(dangle*cv%weight(i))**2
         case(angle_com); ! cannot tell between theta/-theta
            dummy=dummy+((abs(cv%r(i,c1))-abs(cv%r(i,c2)))* &
     & cv%weight(i))**2
         case default
            dummy=dummy+((cv%r(i,c1)-cv%r(i,c2))*cv%weight(i))**2
         end select
        enddo
        cv_common_rmsd=sqrt(dummy/sum(cv%weight(1:cv%num_cv)**2))
!
! cv_common_rmsd=sqrt( ! take square root
! & dot_product(
! & (cv%r(1:cv%num_cv,c1)-cv%r(1:cv%num_cv,c2))**2, ! square
! & cv%weight(1:cv%num_cv)**2)
! & *(1.0d0/sum(cv%weight(1:cv%num_cv)**2)) ! weight, average
! & ) ! sqrt
       else
        call wrndie(0,whoami,trim('OUT OF BOUNDS. NOTHING DONE.'))
       endif
       end function cv_common_rmsd
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine cv_common_update_ave()
! local
       real(chm_real) :: t
! begin
       t=1.0d0*cv%num_average_samples/(cv%num_average_samples+1)
       cv%r(:,ref2)=t*cv%r(:,ref2)+(1.0d0-t)*cv%r(:,main)
       cv%num_average_samples=cv%num_average_samples+1
       end subroutine cv_common_update_ave
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine cv_common_update_fe_ave()
! local
       real(chm_real) :: t
! begin
       t=1.0d0*cv%num_fe_samples/(cv%num_fe_samples+1)
       cv%feav=t*cv%feav+(1.0d0-t)*cv%fe
       cv%num_fe_samples=cv%num_fe_samples+1
       end subroutine cv_common_update_fe_ave
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine cv_common_set_ave_samples(n)
       integer :: n
       if (n.ge.0) cv%num_average_samples=n
       end subroutine cv_common_set_ave_samples
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine cv_common_rex_init(temp)
       use sm_var, only: nstring, smcv_initialized
       use consta
!
       real(chm_real), optional :: temp
       real(chm_real) :: t
       integer :: i
!
       if (.not.cv_common_rex_initialized) then
        if (present(temp)) then ; t=temp; else ; t=300d0; endif
        if (t.gt.0) cv%rex_beta=1d0/(t*kboltz)
        if (.not.smcv_initialized) return
!
        allocate(cv%rex_map(nstring))
        cv%rex_map=(/ (i, i=0,nstring-1) /)
        call int_vector_reinit(cv%rex_log)
        cv_common_rex_initialized=.true.
       endif
!
       end subroutine cv_common_rex_init
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine cv_common_rex_done()
       if (cv_common_rex_initialized) then
        deallocate(cv%rex_map)
        call int_vector_done(cv%rex_log) ! destroy rex log
        cv_common_rex_initialized=.false.
       endif
       end subroutine cv_common_rex_done
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine cv_common_rex_set_temp(temp)
       use consta
       real(chm_real), optional :: temp
       real(chm_real) :: t
!
       if (.not.cv_common_rex_initialized) call cv_common_rex_init()
       if (present(temp)) then ; t=temp; else ; t=300d0; endif
       if (t.gt.0) cv%rex_beta=1d0/(t*kboltz)
!
       end subroutine cv_common_rex_set_temp
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine cv_common_rex_print_map(iunit,fmt)
       use sm_var, only: nstring
       use stream
! only root process should call
       integer :: iunit
       character(len=*), optional :: fmt
! local
       integer :: i
       character(len=80) :: frm
       character(len=len("CV_COMMON_PRINT_REX_MAP>") ),parameter::whoami="CV_COMMON_PRINT_REX_MAP>";!macro
! begin
       if (.not.cv_common_rex_initialized) then
        call wrndie(0,whoami,trim('REX NOT INITIALIZED. NOTHING DONE.'))
        return
       endif
!
       if (.not.present(fmt)) then
        write(frm,'("(",I5,"I5)")') nstring
       else
        frm=fmt
       endif
       write(iunit,frm) (/ (i, i=0,nstring-1) /)
       write(iunit,frm) cv%rex_map(1:nstring)
       end subroutine cv_common_rex_print_map
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine cv_common_rex_read_map(iunit)
       use sm_var, only: nstring
       use stream
       use multicom_aux;
       use mpi
       use parallel, only: psnd4, psnd8
!
       integer :: iunit, ierror
#if (KEY_PARALLEL==1)
#if (KEY_SINGLE==1)
 integer :: mpifloat=MPI_REAL 
#endif
#if (KEY_SINGLE==0)
 integer :: mpifloat=MPI_REAL8 
#endif
#if (KEY_INTEGER8==0)
 integer :: mpiint=MPI_INTEGER 
#endif
#if (KEY_INTEGER8==1)
 integer :: mpiint=MPI_INTEGER8 
#endif
 integer :: mpichar=MPI_CHARACTER
 integer :: mpibool=MPI_LOGICAL
#endif
!
       character(len=len("CV_COMMON_READ_REX_MAP>") ),parameter::whoami="CV_COMMON_READ_REX_MAP>";!macro
! begin
       if (.not.cv_common_rex_initialized) then
        call cv_common_rex_init()
       endif
!
       if (MPI_COMM_STRNG.ne.MPI_COMM_NULL) then
        if (ME_STRNG.eq.0) then
         read(iunit,*) cv%rex_map(1:nstring) ! first row contains indices 0 -- nstring-1
         read(iunit,*) cv%rex_map(1:nstring) ! second row is what we want
         if (any(cv%rex_map.lt.0)) call wrndie(0,whoami,trim('READ NEGATIVE RANK.'))
        endif ! ME_
        if (SIZE_STRNG.gt.1) &
     & call mpi_bcast(cv%rex_map,nstring,mpiint,0,MPI_COMM_STRNG,ierror)
       endif ! MPI_COMM
! broadcast to slave nodes
       if (ME_LOCAL.ne.MPI_UNDEFINED.and.SIZE_LOCAL.gt.1) &
! & call MPI_BCAST(cv%rex_map, nstring, MPI_INTEGER,
! & 0,MPI_COMM_LOCAL,ierr)
#if (KEY_INTEGER8==0)
     & call PSND4(cv%rex_map,nstring) 
#endif
#if (KEY_INTEGER8==1)
     & call PSND8(cv%rex_map,nstring) 
#endif
!
       end subroutine cv_common_rex_read_map
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine cv_common_rex_print_log(iunit, fmt)
! assume that unit is prepared
! NOTE that this is a global print!
      use stream
      use multicom_aux;
      use mpi
!
       integer iunit
       character(len=*), optional :: fmt
! local
       character(len=80) frm
       integer :: i
       integer*4 :: rex_log_size4(SIZE_STRNG)
       integer :: rex_log_size8(SIZE_STRNG)
       integer*4 :: rex_log_disp4(SIZE_STRNG)
       integer :: total_size
       integer, pointer, dimension(:) :: rex_log_all
       integer :: ierror
!
#if (KEY_PARALLEL==1)
#if (KEY_SINGLE==1)
 integer :: mpifloat=MPI_REAL 
#endif
#if (KEY_SINGLE==0)
 integer :: mpifloat=MPI_REAL8 
#endif
#if (KEY_INTEGER8==0)
 integer :: mpiint=MPI_INTEGER 
#endif
#if (KEY_INTEGER8==1)
 integer :: mpiint=MPI_INTEGER8 
#endif
 integer :: mpichar=MPI_CHARACTER
 integer :: mpibool=MPI_LOGICAL
#endif
! do work
! gather all data on root
       if (.not.cv_common_rex_initialized) call cv_common_rex_init()
!
       if (MPI_COMM_STRNG.ne.MPI_COMM_NULL.and.SIZE_STRNG.gt.1) then
! calculate size of logs
        rex_log_size8=0
        call MPI_ALLGATHER(cv%rex_log%last,1,mpiint, &
     & rex_log_size8,1,mpiint, &
     & MPI_COMM_STRNG,ierror)
        total_size=sum(rex_log_size8)
        rex_log_size4=rex_log_size8 ! type cast to 4 byte integer
! allocate space to hold entire log
        allocate(rex_log_all(total_size))
! calculate send displacements
        rex_log_disp4(1)=0;
        do i=1,SIZE_STRNG-1
         rex_log_disp4(i+1)=rex_log_disp4(i)+rex_log_size4(i)
        enddo
! now gather the logs
        call MPI_ALLGATHERV(cv%rex_log%i,cv%rex_log%last,mpiint, &
     & rex_log_all,rex_log_size4,rex_log_disp4,mpiint, &
     & MPI_COMM_STRNG,ierror)
!
        if (.not.present(fmt)) then
         frm='(2I5,I8)'
        else
         frm=fmt
        endif
!
      if(ME_STRNG.eq.0.and.total_size.gt.0) write(iunit,frm) rex_log_all
!
        call int_vector_reinit(cv%rex_log) ! erase log
        deallocate(rex_log_all)
       endif ! STRNG
       end subroutine cv_common_rex_print_log
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine cv_common_rex_compute_dE(rnew, dE)
       use consta
       real(chm_real) :: rnew(:)
       real(chm_real) :: dummy1, dummy2, dE
       integer :: i
! calculate string energies (code adopted from cv_common_evolve_smcv)
       dE=0d0
       do i=1, cv%num_cv
!
         select case(cv%type(i))
         case(dihe_com);
          dummy1=(rnew(i) - cv%r(i,instant)) ! new
          dummy1=modulo(dummy1,TWOPI)
          if (dummy1.gt.PI) dummy1=dummy1-TWOPI
!
          dummy2=(cv%r(i,main) - cv%r(i,instant)) ! old
          dummy2=modulo(dummy2,TWOPI)
          if (dummy2.gt.PI) dummy2=dummy2-TWOPI
         case(angle_com); ! cannot tell between theta/-theta
          dummy1=(abs(rnew(i)) - cv%r(i,instant))
!
          dummy2=(abs(cv%r(i,main)) - cv%r(i,instant))
         case default
          dummy1=(rnew(i) - cv%r(i,instant))
          dummy2=(cv%r(i,main) - cv%r(i,instant))
         end select
!
         dE=dE+cv%k(i)*(dummy1**2-dummy2**2) ! new energy - old energy
        enddo ! loop over CV
        dE=0.5d0*dE
!
       end subroutine cv_common_rex_compute_dE
!===================================================================================================
       subroutine cv_common_voronoi_init()
       call cv_common_voronoi_update()
       cv%voronoi_data=0
! cv%cross_acceptO=1 ! need a nonzero initial condition
       cv%voronoi_cut=9999d0
       cv%voronoi_whereami=-1
       cv%voronoi_map=-1
       end subroutine cv_common_voronoi_init
!===================================================================================================
       subroutine cv_common_voronoi_set_cutoff(cut)
       real(chm_real) :: cut
!
       if (cut.gt.0d0) then ! reject invalid values without warning
        if (.not. cv_common_voronoi_initialized) &
     & call cv_common_voronoi_init()
        cv%voronoi_cut=cut
       endif
       end subroutine cv_common_voronoi_set_cutoff
!===================================================================================================
       subroutine cv_common_print_voro_map(iunit,fmt)
       use sm_var, only: nstring
       use stream
       use multicom_aux;
       use mpi
!
       integer :: iunit !, ierr
       character(len=*), optional :: fmt
! local
!#include "mpitype.def"
!
       integer :: i
       character(len=80) :: frm
       character(len=len("CV_COMMON_PRINT_VORO_MAP>") ),parameter::whoami="CV_COMMON_PRINT_VORO_MAP>";!macro
! begin
       if (.not.cv_common_voronoi_initialized) call cv_common_voronoi_init()
!
! if (MPI_COMM_STRNG.ne.MPI_COMM_NULL.and.SIZE_STRNG.gt.1) then
! call MPI_ALLGATHER(cv%voronoi_whereami, 1, mpiint,
! & cv%voronoi_map, 1, mpiint, MPI_COMM_STRNG, ierr)
!
        if (ME_STRNG.eq.0) then
         if (.not.present(fmt)) then
          write(frm,'("(",I5,"I5)")') nstring
         else
          frm=fmt
         endif
         write(iunit,frm) (/ (i, i=1,nstring) /)
         write(iunit,frm) cv%voronoi_map(1:nstring)
        endif ! ME_STRNG
! endif ! MPI_COMM_STRNG
!
       end subroutine cv_common_print_voro_map
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine cv_common_read_voro_map(iunit)
       use sm_var, only: nstring
       use stream
       use multicom_aux;
       use mpi
       use parallel, only: psnd4, psnd8
#if (KEY_PARALLEL==1)
#if (KEY_SINGLE==1)
 integer :: mpifloat=MPI_REAL 
#endif
#if (KEY_SINGLE==0)
 integer :: mpifloat=MPI_REAL8 
#endif
#if (KEY_INTEGER8==0)
 integer :: mpiint=MPI_INTEGER 
#endif
#if (KEY_INTEGER8==1)
 integer :: mpiint=MPI_INTEGER8 
#endif
 integer :: mpichar=MPI_CHARACTER
 integer :: mpibool=MPI_LOGICAL
#endif
!
       integer :: iunit, ierror
       character(len=len("CV_COMMON_READ_VORO_MAP>") ),parameter::whoami="CV_COMMON_READ_VORO_MAP>";!macro
! begin
       if (.not.cv_common_voronoi_initialized) then
        call cv_common_voronoi_init()
       endif
!
       if (MPI_COMM_STRNG.ne.MPI_COMM_NULL) then
        if (ME_STRNG.eq.0) then
         read(iunit,*) cv%voronoi_map(1:nstring) ! first row contains indices 0 -- nstring-1
         read(iunit,*) cv%voronoi_map(1:nstring) ! second row is what we want
         if (any(cv%voronoi_map.lt.0)) call wrndie(0,whoami,trim('READ NEGATIVE RANK.'))
        endif ! ME_
        if (SIZE_STRNG.gt.1) then
         call mpi_bcast(cv%voronoi_map,nstring,mpiint,0,MPI_COMM_STRNG,ierror)
        endif
       endif ! MPI_COMM_STRNG
! broadcast to slave nodes
       if (ME_LOCAL.ne.MPI_UNDEFINED.and.SIZE_LOCAL.gt.1) &
! & call MPI_BCAST(cv%voronoi_map, nstring, MPI_INTEGER,
! & 0,MPI_COMM_LOCAL,ierr)
#if (KEY_INTEGER8==0)
     & call PSND4(cv%voronoi_map,nstring) 
#endif
#if (KEY_INTEGER8==1)
     & call PSND8(cv%voronoi_map,nstring) 
#endif
!
       end subroutine cv_common_read_voro_map
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine cv_common_voronoi_update()
       use sm_var, only: nstring, mestring, smcv_initialized
       use multicom_aux;
       use mpi
       use parallel, only: psnd4, psnd8
#if (KEY_PARALLEL==1)
#if (KEY_SINGLE==1)
 integer :: mpifloat=MPI_REAL 
#endif
#if (KEY_SINGLE==0)
 integer :: mpifloat=MPI_REAL8 
#endif
#if (KEY_INTEGER8==0)
 integer :: mpiint=MPI_INTEGER 
#endif
#if (KEY_INTEGER8==1)
 integer :: mpiint=MPI_INTEGER8 
#endif
 integer :: mpichar=MPI_CHARACTER
 integer :: mpibool=MPI_LOGICAL
#endif
!
! locals
       character(len=len("CV_COMMON_VORONOI_UPDATE>") ),parameter::whoami="CV_COMMON_VORONOI_UPDATE>";!macro
       integer :: ierror
       real(chm_real), pointer :: M(:,:)
       logical :: qroot
       interface
        subroutine smcv_init(maxcv);; integer, optional :: maxcv ; end subroutine smcv_init
       end interface
! begin
       if (.not. smcv_initialized) call smcv_init()
       if (.not.cv_common_voronoi_initialized) then
        allocate(cv%rall(cv%num_cv, nstring))
        allocate(cv%Mall(cv%num_cv, cv%num_cv, nstring))
        allocate(cv%voronoi_data(nstring,2*nstring+1,3))
! assign pointers -- local
        cv%cross_attempt =>cv%voronoi_data(:,1:nstring,1)
        cv%cross_accept =>cv%voronoi_data(:,nstring+1:2*nstring,1)
        cv%voro_occupancy=>cv%voronoi_data(:,2*nstring+1,1)
! assign pointers -- global
        cv%cross_attemptG =>cv%voronoi_data(:,1:nstring,2)
        cv%cross_acceptG =>cv%voronoi_data(:,nstring+1:2*nstring,2)
        cv%voro_occupancyG=>cv%voronoi_data(:,2*nstring+1,2)
! assign pointers -- data from previous run (old)
        cv%cross_attemptO =>cv%voronoi_data(:,1:nstring,3)
        cv%cross_acceptO =>cv%voronoi_data(:,nstring+1:2*nstring,3)
        cv%voro_occupancyO=>cv%voronoi_data(:,2*nstring+1,3)
!
        cv%voronoi_data=0 ! set all crossing data to zero
! cv%cross_acceptO=1
        cv%voronoi_cut=9999d0
        cv%voronoi_whereami=-1 ! -1 indicates this needs to be computed
!
        allocate(cv%voronoi_map(nstring))
        cv%voronoi_map=-1
!
        call int_vector_reinit(cv%voro_log)
        cv_common_voronoi_initialized=.true.
       endif
! gather all main coordinate sets (same code as in print_global)
       qroot=(MPI_COMM_STRNG.ne.MPI_COMM_NULL)
!
       if (qroot.and.SIZE_STRNG.gt.1) then
        call MPI_ALLGATHER(cv%r,cv%num_cv,mpifloat, &
     & cv%rall,cv%num_cv,mpifloat,MPI_COMM_STRNG, &
     & ierror)
! gather all M^-1 matrices
!
        if (.not.cv_common_Minv_initialized) then
         call wrndie(0,whoami,trim('TENSOR M^-1 HAS NOT BEEN SET.'))
        endif
        M => cv%M(1:cv%num_cv,1:cv%num_cv,3) ! for use in MPI call
!
        call MPI_ALLGATHER(M, &! cv%M(1:cv%num_cv,1:cv%num_cv,3),
     & cv%num_cv**2,mpifloat, &
     & cv%Mall,cv%num_cv**2,mpifloat,MPI_COMM_STRNG, &
     & ierror)
       endif
!
! broadcast to slaves
       if (MPI_COMM_LOCAL.ne.MPI_COMM_NULL.and.SIZE_LOCAL.gt.1) then
#if (KEY_SINGLE==1)
        call PSND4r(cv%rall,cv%num_cv*nstring)
        call PSND4r(cv%Mall,(cv%num_cv**2)*nstring)
#else
        call PSND8m(cv%rall,cv%num_cv*nstring)
        call PSND8m(cv%Mall,(cv%num_cv**2)*nstring)
#endif
       endif
!
       cv_common_voronoi_wrong_cell=.false.
       end subroutine cv_common_voronoi_update
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine cv_common_voronoi_smart_update(rnew, rold, m_iter)
       use sm_var, only: nstring, mestring, smcv_initialized
       use cv_types
       use stream
! cv%rall(:,:) is expected to have values from previous update that
! are consistent with rtemp; if this is not true, the routine
! just does a regular update and exits
       use multicom_aux;
       use consta
       use mpi
       use parallel, only: psnd4, psnd8
! vars
#if (KEY_PARALLEL==1)
#if (KEY_SINGLE==1)
 integer :: mpifloat=MPI_REAL 
#endif
#if (KEY_SINGLE==0)
 integer :: mpifloat=MPI_REAL8 
#endif
#if (KEY_INTEGER8==0)
 integer :: mpiint=MPI_INTEGER 
#endif
#if (KEY_INTEGER8==1)
 integer :: mpiint=MPI_INTEGER8 
#endif
 integer :: mpichar=MPI_CHARACTER
 integer :: mpibool=MPI_LOGICAL
#endif
!
       real(chm_real) :: rnew(cv%num_cv) ! current CV values (theta(x))
       real(chm_real) :: rold(cv%num_cv) ! new (possibly reflected) CV values (theta(xcomp))
       integer, optional :: m_iter
! locals
       real(chm_real), pointer :: M(:,:)
       real(chm_real) :: rall_new(cv%num_cv, nstring)
       real(chm_real) :: rall_temp(cv%num_cv, nstring)
       real(chm_real) :: Mall_new(cv%num_cv, cv%num_cv, nstring)
       real(chm_real) :: Mall_temp(cv%num_cv, cv%num_cv, nstring)
       integer :: correct_cell_me, correct_cell
       integer :: iter, max_iter
       integer, parameter :: default_iter=15
       integer :: ierror, me
       logical :: qroot
!
       character(len=len("CV_COMMON_VORONOI_SMART_UPDATE>") ),parameter::whoami="CV_COMMON_VORONOI_SMART_UPDATE>";!macro
!
! begin
!
       if (cv%num_cv.eq.0) then
         call wrndie(0,whoami,trim('NO CV DEFINED. NOTHING DONE.'))
         return
       endif
!
       qroot=(MPI_COMM_STRNG.ne.MPI_COMM_NULL)
!
       if (qroot.and.SIZE_STRNG.gt.1) then
! gather all main cv coordinate sets
         call MPI_ALLGATHER(cv%r,cv%num_cv,mpifloat, &
     & rall_new,cv%num_cv,mpifloat,MPI_COMM_STRNG, &
     & ierror)
! gather all M^-1 matrices
         M => cv%M(1:cv%num_cv,1:cv%num_cv,3)
!
         call MPI_ALLGATHER(M, &
     & cv%num_cv**2,mpifloat, &
     & Mall_new,cv%num_cv**2,mpifloat,MPI_COMM_STRNG, &
     & ierror)
       endif
!
       if (.not.cv_common_voronoi_initialized) then
        allocate(cv%rall(cv%num_cv, nstring))
        allocate(cv%Mall(cv%num_cv, cv%num_cv, nstring))
        cv%rall=rall_new
        cv%Mall=Mall_new
!
        allocate(cv%voronoi_data(nstring,2*nstring+1,3))
! assign pointers -- local
        cv%cross_attempt =>cv%voronoi_data(:,1:nstring,1)
        cv%cross_accept =>cv%voronoi_data(:,nstring+1:2*nstring,1)
        cv%voro_occupancy=>cv%voronoi_data(:,2*nstring+1,1)
! assign pointers -- global
        cv%cross_attemptG =>cv%voronoi_data(:,1:nstring,2)
        cv%cross_acceptG =>cv%voronoi_data(:,nstring+1:2*nstring,2)
        cv%voro_occupancyG=>cv%voronoi_data(:,2*nstring+1,2)
! assign pointers -- data from previous run (old)
        cv%cross_attemptO =>cv%voronoi_data(:,1:nstring,3)
        cv%cross_acceptO =>cv%voronoi_data(:,nstring+1:2*nstring,3)
        cv%voro_occupancyO=>cv%voronoi_data(:,2*nstring+1,3)
!
        cv%voronoi_data=0 ! set all crossing data to zero
! cv%cross_acceptO=1
        cv%voronoi_cut=9999d0
        cv%voronoi_whereami=-1 ! -1 indicates this needs to be computed
!
        allocate(cv%voronoi_map(nstring))
        cv%voronoi_map=-1
!
        call int_vector_reinit(cv%voro_log)
!
        cv_common_voronoi_initialized=.true.
       else
        me=mestring+1
        if (qroot) then
!
         if (.not.cv_common_Minv_initialized) then
          call wrndie(0,whoami,trim('TENSOR M^-1 HAS NOT BEEN SET.'))
         endif
!
! check if the old set is consistent
!
         correct_cell_me=cv_common_voronoi_check_aux &
     & (rold, cv%rall, cv%num_cv, nstring, cv%Mall, me)
!
! are the coords in rold consistent with the current V. cell centers?
         if (correct_cell_me.eq.0) then
! if old set is outside cell, it means that we just crossed into a
! neighbor cell; in this case, test rnew, not rold
! (repeat as above).
          rold=rnew
          correct_cell_me=cv_common_voronoi_check_aux &
     & (rold, cv%rall, cv%num_cv, nstring, cv%Mall, me)
! even if correct_cell is true now, the reversed momenta might not place the replica
! into the cell corresponding to the _evolved_ centers, so it may be best not to evolve this particular
! cell; this matter in any case is particular to charmm at this stage, so will think on it.-VO.
         endif ! .not. correct_cell_me
! pool all results
         call MPI_ALLREDUCE(correct_cell_me, correct_cell, 1, &
     & mpiint, MPI_MIN, MPI_COMM_STRNG, ierror)
         if (correct_cell.eq.0) then
          call wrndie(0,whoami,trim(' CURRENT VORONOI CELL CENTERS INCONSISTENT WITH COORDINATES.'))
! roll back CVs
           cv%r(1:cv%num_cv,1)=cv%rall(1:cv%num_cv,me)
!
         else ! correct_cell
! begin iterative adjustment
          if (present(m_iter)) then
           max_iter=m_iter
          else
           max_iter=default_iter
          endif
!
          iter=0
          rall_temp=rall_new
          Mall_temp=Mall_new
          do
! check if the new set is consistent
           correct_cell_me=cv_common_voronoi_check_aux &
     & (rold, rall_temp, cv%num_cv, nstring, Mall_temp, me)
! pool all results
           call MPI_ALLREDUCE(correct_cell_me, correct_cell, 1, &
     & mpiint, MPI_MIN, MPI_COMM_STRNG, ierror)
           if (correct_cell.eq.1) then
            cv%rall=rall_temp
            cv%Mall=Mall_temp
! moved below 1/2013
!! `roll back' cv's:
! if (iter.gt.0) then ! otherwise they are fine
! cv%r(1:cv%num_cv,1)=rall_temp(1:cv%num_cv,me)
! cv%M(1:cv%num_cv,1:cv%num_cv,3)=Mall_temp(:,:,me)
! endif
            exit
           elseif (iter.ge.max_iter) then
            call wrndie(0,whoami,trim(' MAXIMUM NUMBER OF ITERATIONS EXCEEDED.'))
! moved below 1/2013
!! reset cv`s to consistent Voronoi cell centers :
! cv%r(1:cv%num_cv,1)=cv%rall(1:cv%num_cv,me)
! cv%M(1:cv%num_cv,1:cv%num_cv,3)=cv%Mall(:,:,me)
            exit
           else
            rall_temp=0.5d0*(rall_temp+cv%rall)
            Mall_temp=0.5d0*(Mall_temp+cv%Mall)
           endif ! correct cell
           iter=iter+1
          enddo ! <empty>
         endif ! .not. correct_cell
!
        endif ! qroot
       endif ! .not. initialized
! broadcast to slaves
! (1.2013: are cv%r broadcast, in case they change ? where ?
! Answer: slaves do not need cv%r in voronoi; they only need cv%rall [ in voronoi_compute, which is parallel ], which is computed by roots & broadcast
! even in the case that the cv%r are evolving: the slaves will have incorrect evolution, but both reflections and image updates only use cv%r from roots
! however, for a clean look, I am setting cv%r = cv%rall(me) below (& commenting out above for roots)
       if (MPI_COMM_LOCAL.ne.MPI_COMM_NULL.and.SIZE_LOCAL.gt.1) then
#if (KEY_SINGLE==1)
        call PSND4r(cv%rall,cv%num_cv*nstring)
        call PSND4r(cv%Mall,(cv%num_cv**2)*nstring)
#else
        call PSND8m(cv%rall,cv%num_cv*nstring)
        call PSND8m(cv%Mall,(cv%num_cv**2)*nstring)
#endif
       endif
! moved possible reassignment here (all ranks participate)
       cv%r(1:cv%num_cv,1)=cv%rall(1:cv%num_cv,me)
       cv%M(1:cv%num_cv,1:cv%num_cv,3)=cv%Mall(:,:,me)
!
       end subroutine cv_common_voronoi_smart_update
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       integer function cv_common_voronoi_check_aux &
     & (r, rall, ncv, nstring, Mall, me)
! auxiliary function for use with voronoi_smart_update
! here, I only need to know whether the host cell is "me";
! in contrast, in voronoi_compute, I need to know what the host cell is,
! because that information is used to construct the MSM
! this is a straightforward implementation of eqs 29/30 (perhaps a bit inefficient but at the benefit of clarity)
!
       use consta
       integer :: ncv, nstring, me
       real(chm_real) :: r(ncv), rall(ncv, nstring), Mall(ncv, ncv, nstring)
! locals
       integer :: i,ii,j,jj,k
       real(chm_real) :: z(ncv), dummy, msd_n(nstring), msd_m(nstring)
       logical :: q
       do i=1,nstring
!
! preprocess differences (wrap angles)
        do j=1,ncv
         select case(cv%type(j))
         case(dihe_com);
            dummy=modulo(r(j)-rall(j,i), TWOPI)
            if (dummy .gt. PI ) dummy = dummy - TWOPI
         case(angle_com); ! cannot tell between theta/-theta
            dummy=abs(r(j))-abs(rall(j,i))
         case default
            dummy=r(j)-rall(j,i)
         end select
         z(j)=dummy
        enddo ! j
!
! compute two norms
        if (me.eq.i) then ! compute rmsd_me
         do ii=1, nstring
           dummy=0d0
           do j=1,ncv
            jj=j-1
            do k=1, jj
             dummy=dummy + &
     & z(j) * (Mall(j,k,me) + Mall(j,k,ii)) * z(k)
            enddo
           enddo
           dummy=2d0*dummy ! double contribution from off-diagonal terms
! contribution from diagonal terms
           do j=1,ncv
            dummy=dummy + &
     & z(j) * (Mall(j,j,me) + Mall(j,j,ii)) * z(j)
           enddo
           msd_n(ii)=0.5*dummy
         enddo ! ii=1,nstring
        else ! compute rmsd_i; computed only for different tensors (hence if/else)
! same code as above
         dummy=0d0
         do j=1,ncv
          jj=j-1
          do k=1, jj
           dummy=dummy + &
     & z(j) * (Mall(j,k,me)+Mall(j,k,i)) * z(k)
          enddo
         enddo
         dummy=2d0*dummy ! double contribution from off-diagonal terms
! contribution from diagonal terms
         do j=1,ncv
          dummy=dummy + &
     & z(j) * (Mall(j,j,me)+Mall(j,j,i)) * z(j)
         enddo
         msd_m(i)=0.5d0*dummy
        endif ! me.eq.i
!
       enddo ! i=1,nstring
!
       msd_m(me)=msd_n(me)
! check whether theta(x) lives in this cell
       do i=1, nstring
        q=(msd_n(i).le.msd_m(i))
        if (.not.q) exit
       enddo
!
       if (q.and.((msd_n(me)).lt.cv%voronoi_cut**2)) then
         cv_common_voronoi_check_aux=1
       else
         cv_common_voronoi_check_aux=0
       endif
!
       end function cv_common_voronoi_check_aux
!
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine cv_common_voronoi_done()
!
       if (cv_common_voronoi_initialized) then
        deallocate(cv%rall)
        deallocate(cv%Mall)
        deallocate(cv%voronoi_map)
        deallocate(cv%voronoi_data)
!
        nullify(cv%cross_attempt)
        nullify(cv%cross_accept)
        nullify(cv%voro_occupancy)
!
        nullify(cv%cross_attemptG)
        nullify(cv%cross_acceptG)
        nullify(cv%voro_occupancyG)
!
        nullify(cv%cross_attemptO)
        nullify(cv%cross_acceptO)
        nullify(cv%voro_occupancyO)
!
        call int_vector_done(cv%voro_log) ! destroy Voronoi T. log
        cv_common_voronoi_initialized=.false.
       endif
       end subroutine cv_common_voronoi_done
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       function cv_common_voronoi_compute(rtemp) ! rtemp calculated elsewhere and passed in
! this is a very straightforward implementation of eqations 29 and 30 in the string paper
       use cv_types
       use sm_var, only: nstring, mestring
       use sm_config, only: calc_voronoi_para
      use stream
      use multicom_aux;
      use number
      use consta
      use mpi
      use parallel, only: psnd4, psnd8
!
#if (KEY_PARALLEL==1)
#if (KEY_SINGLE==1)
 integer :: mpifloat=MPI_REAL 
#endif
#if (KEY_SINGLE==0)
 integer :: mpifloat=MPI_REAL8 
#endif
#if (KEY_INTEGER8==0)
 integer :: mpiint=MPI_INTEGER 
#endif
#if (KEY_INTEGER8==1)
 integer :: mpiint=MPI_INTEGER8 
#endif
 integer :: mpichar=MPI_CHARACTER
 integer :: mpibool=MPI_LOGICAL
#endif
! variables
       integer :: cv_common_voronoi_compute ! index of Voronoi cell that the system is in
       real(chm_real) :: rtemp(cv%num_cv) ! current CV values (theta(x))
! locals
       real(chm_real) :: msd(nstring,nstring), msdall(nstring,nstring) ! mean square distance
       real(chm_real) :: dummy, z(cv%num_cv)
       integer :: i, j, k, n, m, which(1), q(nstring), me
! parallelization
       logical :: qpara
       integer*4 :: send_displ(SIZE_LOCAL), send_count(SIZE_LOCAL)
       integer :: mbeg, mend, ierror
       character(len=len("CV_COMMON_VORONOI_COMPUTE>") ),parameter::whoami="CV_COMMON_VORONOI_COMPUTE>";!macro
!
       if (cv%num_cv.eq.0) then
         call wrndie(0,whoami,trim('NO CV DEFINED. NOTHING DONE.'))
         return
       endif
!
       if (.not.cv_common_Minv_initialized) then
        call wrndie(0,whoami,trim('TENSOR M^-1 HAS NOT BEEN SET.'))
       endif
! initialize, if necessary
       if (.not.cv_common_voronoi_initialized) call cv_common_voronoi_init()
! compute RMSD
! need to compute distances w.r.t. all M tensors on the string
       msd=zero
       msdall=zero
!
! compute norms in parallel: each slave node is assigned a set of matrices
! compute index limits (this could be done once in the smcv_init routine)
!
       qpara=(MPI_COMM_LOCAL.ne.MPI_COMM_NULL.and.SIZE_LOCAL.gt.1 &
     & .and.calc_voronoi_para)
       if (qpara) then
!
        j=ceiling(1.0d0*nstring/SIZE_LOCAL) ! max. number of CV assigned to slave node
!
        do i=1,SIZE_LOCAL
         send_displ(i)=nstring*min((i-1)*j,nstring-1) ! cannot exceed nstring
         send_count(i)=nstring*max(0,min(j,nstring-j*(i-1)))
        enddo
        me=ME_LOCAL+1
! indices are below
        mbeg=send_displ(me)/nstring+1
        mend=mbeg+send_count(me)/nstring-1
       else
        mbeg=1
        mend=nstring
       endif ! qpara
!
       do n=1, nstring ! loop over z
!ccccccccccccccccccccccccccccccccccccccccc preprocess differences (wrap angles)
        do j=1,cv%num_cv
! this is 'unclean', and I may move this routine 'up' but for now keep the angle
         select case(cv%type(j))
         case(dihe_com);
            dummy=modulo(rtemp(j)-cv%rall(j,n), TWOPI)
            if (dummy .gt. PI ) dummy = dummy - TWOPI
         case(angle_com); ! cannot tell between theta/-theta
            dummy=abs(rtemp(j))-abs(cv%rall(j,n))
         case default
            dummy=rtemp(j)-cv%rall(j,n)
         end select
         z(j)=dummy
        enddo ! j
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        do m=mbeg,mend ! loop over the M tensors
!cccc compute z_n M_m z_n below
!==============================
         dummy=0d0
         do i=1,cv%num_cv
          j=i-1
          do k=1, j ! M is symmetric, so do not need to iterate over all indices when computing norm
           dummy=dummy + &
     & z(i) * (cv%Mall(i,k,m)) * z(k)
          enddo
         enddo
         dummy=2d0*dummy ! double contribution from off-diagonal terms (because M is symmetric)
! contribution from diagonal terms
         do i=1,cv%num_cv
          dummy=dummy + &
     & z(i) * (cv%Mall(i,i,m)) * z(i)
         enddo
         msd(n,m)=dummy
!==============================
        enddo ! m=1,nstring
       enddo ! n=1,nstring
!
! pool norms on root (gather over matrices (m-index) ):
! VO 9.2013: note that the method below is a nonstandard MPI_IN_PLACE (sometimes will give errors)
! changing to proper MPI_IN_PLACE
       if (qpara) then
        if (ME_LOCAL.eq.0) then
         call MPI_GATHERV(MPI_IN_PLACE, send_count(me), &
     & mpifloat, &
     & msd, send_count, send_displ, &
     & mpifloat, 0, MPI_COMM_LOCAL, ierror)
        else ! slaves
         call MPI_GATHERV(msd(1,mbeg),send_count(me), &
     & mpifloat, &
     & msd, send_count, send_displ, &
     & mpifloat, 0, MPI_COMM_LOCAL, ierror)
        endif ! ME_LOCAL
       endif ! qpara
! root determines where theta(x) is
       if (ME_LOCAL.eq.0.or..not.qpara) then
         do n=1, nstring
          q(n)=1 ! assume replica lives in cell "n"
          do m=1, nstring
           if (msd(n,n)+msd(n,m).gt.msd(m,n)+msd(m,m)) then ! note that this is false for m=n, as should be
            q(n)=0 ! Voronoi test failed, n is not the cell !
            exit
           endif
          enddo
         enddo
         which=maxloc(q) ! maxloc needs an integer array;
         if (msd(which(1),which(1)).gt.cv%voronoi_cut**2) &
     & which(1)=-which(1) ! outside of the allowed region (special code)
       endif ! ME_LOCAL
!
       if (qpara) &
#if (KEY_INTEGER8==0)
     & call PSND4(which,1) 
#endif
#if (KEY_INTEGER8==1)
     & call PSND8(which,1) 
#endif
! write(600+ME_GLOBAL,*) q
! write(600+ME_GLOBAL,'(32G12.4)') msd
! write(600+ME_GLOBAL,*) rtemp
! stop
!
       cv_common_voronoi_compute=which(1)
!
       end function cv_common_voronoi_compute
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine cv_common_print_voro_data(iunit)
       use sm_var, only: nstring
! assume that unit is prepared
! NOTE that this is a global print!
       use stream
       use multicom_aux;
       use mpi
!
       integer :: iunit
! locals
       character(len=80) :: fmt
       integer :: j
       integer :: voro_data_all(nstring,2*nstring+1)
       integer :: ierror, type
! do work
! gather all data on root
#if (KEY_INTEGER8==0)
       type=MPI_INTEGER 
#endif
#if (KEY_INTEGER8==1)
       type=MPI_INTEGER8 
#endif
       if (.not.cv_common_voronoi_initialized) call cv_common_voronoi_init()
!
       if (MPI_COMM_STRNG.ne.MPI_COMM_NULL) then
        if (SIZE_STRNG.gt.1) then
         call MPI_ALLREDUCE(cv%voronoi_data(:,:,1),voro_data_all, &
     & nstring*(2*nstring+1),type,MPI_SUM,MPI_COMM_STRNG,ierror)
        else
         voro_data_all=cv%voronoi_data(:,:,1)
        endif ! SIZE
!
        if (ME_STRNG.eq.0) then ! string root writes
         write(fmt,int_format) nstring
         do j=1,nstring
          write(iunit,'('//fmt//int_format//')')                        &
     & voro_data_all(j,1:nstring)+cv%cross_attemptO(j,:) ! crossing attemps
         enddo
         write(iunit,'("%")') ! break
         do j=1,nstring
          write(iunit,'('//fmt//int_format//')')                        &
     & voro_data_all(j,nstring+1:2*nstring)+cv%cross_acceptO(j,:) ! crossing accepts
         enddo
         write(iunit,'("%")') ! break
         write(iunit,'('//fmt//int_format//')')                         &
     & voro_data_all(:,2*nstring+1)+cv%voro_occupancyO(:) ! occupancy
        endif ! ME
       endif ! MPI_COMM_STRNG
!
       end subroutine cv_common_print_voro_data
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine cv_common_read_voro_data(iunit)
       use sm_var, only: nstring
! assume that unit is prepared
       use multicom_aux;
       use mpi
       use parallel, only: psnd4, psnd8
!
       integer :: iunit
! locals
       integer :: j
       integer :: voro_data_all(SIZE_STRNG,2*SIZE_STRNG+1)
       integer :: ierror
#if (KEY_PARALLEL==1)
#if (KEY_SINGLE==1)
 integer :: mpifloat=MPI_REAL 
#endif
#if (KEY_SINGLE==0)
 integer :: mpifloat=MPI_REAL8 
#endif
#if (KEY_INTEGER8==0)
 integer :: mpiint=MPI_INTEGER 
#endif
#if (KEY_INTEGER8==1)
 integer :: mpiint=MPI_INTEGER8 
#endif
 integer :: mpichar=MPI_CHARACTER
 integer :: mpibool=MPI_LOGICAL
#endif
! do work
! gather all data on root
!
       if (.not.cv_common_voronoi_initialized) call cv_common_voronoi_init()
!
       if (MPI_COMM_STRNG.ne.MPI_COMM_NULL) then
        if (ME_STRNG.eq.0) then ! string root reads
         do j=1,nstring
          read(iunit,*) voro_data_all(j,1:nstring) ! crossing attemps
         enddo
         read(iunit,'(A)') ! break
         do j=1,nstring
          read(iunit,*) voro_data_all(j,nstring+1:2*nstring) ! crossing accepts
         enddo
         read(iunit,'(A)') ! break
         read(iunit,*) voro_data_all(:,2*nstring+1) ! occupancy
        endif ! ME
!
        cv%voronoi_data(:,:,3)=voro_data_all ! place into "old" position
!
        if (SIZE_STRNG.gt.1) then
         call mpi_bcast(cv%voronoi_data(:,:,3),SIZE_STRNG*(2*SIZE_STRNG+1),mpiint,0,MPI_COMM_STRNG,ierror)
        endif
       endif ! MPI_COMM_STRNG
! broadcast to slaves
       if (MPI_COMM_LOCAL.ne.MPI_COMM_NULL.and.SIZE_LOCAL.gt.1) &
#if (KEY_INTEGER8==0)
     & call PSND4r(cv%voronoi_data(:,:,3),nstring*(2*nstring+1)) 
#endif
#if (KEY_INTEGER8==1)
     & call PSND8m(cv%voronoi_data(:,:,3),nstring*(2*nstring+1)) 
#endif
!
       end subroutine cv_common_read_voro_data
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine cv_common_voronoi_print_log(iunit)
! assume that unit is prepared
! NOTE that this is a global print!
       use multicom_aux;
       use mpi
!
       integer :: iunit
! locals
       integer :: i
       integer*4 :: voro_log_size4(SIZE_STRNG)
       integer :: voro_log_size8(SIZE_STRNG)
       integer*4 :: voro_log_disp4(SIZE_STRNG)
       integer :: total_size
       integer, pointer, dimension(:) :: voro_log_all
       integer :: ierror
#if (KEY_PARALLEL==1)
#if (KEY_SINGLE==1)
 integer :: mpifloat=MPI_REAL 
#endif
#if (KEY_SINGLE==0)
 integer :: mpifloat=MPI_REAL8 
#endif
#if (KEY_INTEGER8==0)
 integer :: mpiint=MPI_INTEGER 
#endif
#if (KEY_INTEGER8==1)
 integer :: mpiint=MPI_INTEGER8 
#endif
 integer :: mpichar=MPI_CHARACTER
 integer :: mpibool=MPI_LOGICAL
#endif
! do work
! gather all data on root
!
       if (.not.cv_common_voronoi_initialized) call cv_common_voronoi_init()
!
       if (MPI_COMM_STRNG.ne.MPI_COMM_NULL.and.SIZE_STRNG.gt.1) then
! calculate size of logs
        voro_log_size8=0
! call MPI_GATHER(cv%voro_log%last,1,type,
! & voro_log_size8,1,type,
! & 0,MPI_COMM_WORLD,ierror)
        call MPI_ALLGATHER(cv%voro_log%last,1,mpiint, &
     & voro_log_size8,1,mpiint, &
     & MPI_COMM_STRNG,ierror)
        total_size=sum(voro_log_size8)
        voro_log_size4=voro_log_size8 ! type cast to 4 byte integer
! allocate space to hold entire log
        allocate(voro_log_all(total_size))
! calculate send displacements
        voro_log_disp4(1)=0;
        do i=1,SIZE_STRNG-1
         voro_log_disp4(i+1)=voro_log_disp4(i)+voro_log_size4(i)
        enddo
! now gather the logs
! call MPI_GATHERV(cv%voro_log%i,cv%voro_log%last,type,
! & voro_log_all,voro_log_size4,voro_log_disp4,type,
! & 0,MPI_COMM_WORLD,ierror)
        call MPI_ALLGATHERV(cv%voro_log%i,cv%voro_log%last,mpiint, &
     & voro_log_all,voro_log_size4,voro_log_disp4,mpiint,&
     & MPI_COMM_STRNG,ierror)
!
        if (ME_STRNG.eq.0) write(iunit) voro_log_all
!
        call int_vector_reinit(cv%voro_log) ! erase log
        deallocate(voro_log_all)
       endif ! STRNG
       end subroutine cv_common_voronoi_print_log
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine cv_common_read_M_global(ifile, ind)
       use sm_var, only: nstring
! print mass matrix; assume that cv%m(:,:,2) has what the user wants
      use multicom_aux;
      use mpi
      use parallel, only: psnd4, psnd8
!
#if (KEY_PARALLEL==1)
#if (KEY_SINGLE==1)
 integer :: mpifloat=MPI_REAL 
#endif
#if (KEY_SINGLE==0)
 integer :: mpifloat=MPI_REAL8 
#endif
#if (KEY_INTEGER8==0)
 integer :: mpiint=MPI_INTEGER 
#endif
#if (KEY_INTEGER8==1)
 integer :: mpiint=MPI_INTEGER8 
#endif
 integer :: mpichar=MPI_CHARACTER
 integer :: mpibool=MPI_LOGICAL
#endif
       integer :: ifile
       integer, optional :: ind ! M index
! locals
       integer :: ierror, j, Mind
       real(chm_real) :: M_all(cv%num_cv, cv%num_cv, nstring)
       real(chm_real) :: M_me(cv%num_cv, cv%num_cv)
!
       if (present(ind)) then ; Mind=ind ; else ; Mind=4 ; endif
!
       if (MPI_COMM_STRNG.ne.MPI_COMM_NULL) then
!
        if (ME_STRNG.eq.0) then ! root reads
         do j=1,nstring
          read(ifile,'(1X,I10)')
          read(ifile,*) M_all(:,:,j)
         enddo
        endif ! ME
        call MPI_SCATTER(M_all,cv%num_cv**2,mpifloat, &
     & M_me,cv%num_cv**2,mpifloat,0, &
     & MPI_COMM_STRNG,ierror)
       endif ! COMM_STRNG
! broadcast to slaves
       if (ME_LOCAL.ne.MPI_UNDEFINED.and.SIZE_LOCAL.gt.1) &
#if (KEY_SINGLE==1)
     & call PSND4r(M_me,cv%num_cv**2) 
#endif
#if (KEY_SINGLE==0)
     & call PSND8m(M_me,cv%num_cv**2) 
#endif
!
       cv%M(1:cv%num_cv, 1:cv%num_cv, Mind)=M_me
!
       if (Mind.eq.4) then
! set the short-time average to the long-time average
        cv%M(1:cv%num_cv, 1:cv%num_cv, 2)=M_me
        call cv_common_compute_Minv()
       elseif (Mind.eq.3) then
        cv_common_Minv_initialized=.true.
       endif
!
       end subroutine cv_common_read_M_global
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine cv_common_print_M_global(ifile, ind)
       use sm_var, only: nstring
! print mass matrix; assume that cv%m(:,:,4) has what the user wants (long-term average)
!
#if (KEY_MULTICOM==1)
      use multicom_aux; 
#endif
      use mpi ! deal with other platforms later
!
#if (KEY_PARALLEL==1)
#if (KEY_SINGLE==1)
 integer :: mpifloat=MPI_REAL 
#endif
#if (KEY_SINGLE==0)
 integer :: mpifloat=MPI_REAL8 
#endif
#if (KEY_INTEGER8==0)
 integer :: mpiint=MPI_INTEGER 
#endif
#if (KEY_INTEGER8==1)
 integer :: mpiint=MPI_INTEGER8 
#endif
 integer :: mpichar=MPI_CHARACTER
 integer :: mpibool=MPI_LOGICAL
#endif
       integer :: ifile
       integer, optional :: ind ! M index to print
! locals
       character(len=80) :: fmt
       integer :: isizeM, ierror, j, Mind
       real(chm_real) :: M_all(cv%num_cv, cv%num_cv, nstring)
       real(chm_real) :: M_me(cv%num_cv, cv%num_cv)
!
       if (MPI_COMM_STRNG.ne.MPI_COMM_NULL) then
!
        if (present(ind)) then ; Mind=ind ; else ; Mind=4 ; endif
!
        isizeM=cv%num_cv;
        M_me=cv%M(1:isizeM, 1:isizeM, Mind)
        isizeM=isizeM*isizeM
!
        call MPI_GATHER(M_me,isizeM,mpifloat, &
     & M_all,isizeM,mpifloat,0, &
     & MPI_COMM_STRNG,ierror)
! call MPI_ALLGATHER(M_me,isizeM,mpifloat,
! & M_all,isizeM,mpifloat,
! & MPI_COMM_STRNG,ierror)
        write(fmt,'(I5)') cv%num_cv
        if (ME_STRNG.eq.0) then ! root writes
         do j=1,nstring
          write(ifile,'("% ",I5)') j
          write(ifile,'('//fmt//real_format//')') M_all(:,:,j)
! write(ifile,'('//fmt//'E15.5)') M_all(:,:,j)
! write(ifile,'('//fmt//'F11.5)') cv%M(1:cv%num_cv,1:cv%num_cv,1)
         enddo
        endif ! ME
       endif ! MPI_COMM
!
       end subroutine cv_common_print_M_global
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine cv_common_neq_work_init()
! initialize variables for nonequilibrium work
       cv%work=0.
       cv%r(:,forces2)=0.
       cv%r(:,zcur)=cv%r(:,comp)
       cv%r(:,zold)=cv%r(:,comp)
       end subroutine cv_common_neq_work_init
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       function cv_common_neq_get_work()
! returns the value of work
       real(chm_real) :: cv_common_neq_get_work
       cv_common_neq_get_work=cv%work
       end function cv_common_neq_get_work
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine cv_common_interpolate(ifile, ofile, num_rep_in, &
     & num_rep_out, interp_method)
      use stream
! assume that both units ifile/ofile are prepared
      use multicom_aux;
      use number
      use mpi
!
       integer :: ifile, ofile, num_rep_in, num_rep_out, interp_method
! local declarations
       real(chm_real) :: rin(max_cv_common,num_rep_in), &
     & dr(max_cv_common,num_rep_in-1)
       real(chm_real) :: rout(max_cv_common,num_rep_out)
       real(chm_real) :: rr(num_rep_in), rr_out(num_rep_out), ds(num_rep_in-1), &
     & ds2(num_rep_in-1), s(num_rep_in), t(num_rep_out), &
     & rrpp(num_rep_in), dum
       integer :: i,k
       character(len=80) :: fmt
       logical :: qroot
       character(len=len("CV_COMMON_INTERPOLATE>") ),parameter::whoami="CV_COMMON_INTERPOLATE>";!macro
!
       integer, parameter :: linear=1, spline=2, bspline=3
!
       interface ! to linear interpolation routine
        subroutine linear_interp(xin,yin,nin,xout,yout,nout,dydxout)
!
      use chm_kinds
!
        integer :: nin, nout
        real(chm_real) :: xin(nin), yin(nin), xout(nout), yout(nout)
        real(chm_real), optional :: dydxout(nout) ! tangent computation
        real(chm_real) :: dydx(nout)
        end subroutine linear_interp
       end interface
!
       qroot=(MPI_COMM_STRNG.ne.MPI_COMM_NULL.and.ME_STRNG.eq.0)
!
       if (cv%num_cv.eq.0) then
        call wrndie(0,whoami,trim(' NO CV DEFINED. NOTHING DONE.'))
        return
       endif
!
       rin=0d0; dr=0d0; rout=0d0;
!
       if (qroot) then ! root does the work
        do i=1,cv%num_cv
         read(ifile,*) (rin(i,k),k=1,num_rep_in)
        enddo
       endif ! root
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! do the interpolation -- simple, not self-consistent
       if (qroot) then ! only root does the work
! check for undefined values
       if (any(rin(1:cv%num_cv,:).eq.anum)) &
     & call wrndie(0,whoami,trim('SOME CV VALUES ARE UNDEFINED'))
! compute arclength
        dr=rin(:,2:num_rep_in)-rin(:,1:num_rep_in-1)
        s(1)=0
        do i=1,num_rep_in-1
         ds2(i)=dot_product(dr(:,i)**2,cv%weight**2)
         ds(i)=sqrt(ds2(i))
         s(i+1)=s(i)+ds(i)
        enddo
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! normalize arclength
        do i=1,num_rep_in
         s(i)=s(i)/s(num_rep_in)
        enddo
!ccccccccccccccccccccccccc
! create uniform array
        do i=1,num_rep_out
         t(i)=1.0d0*(i-1)/(num_rep_out-1)
        enddo
!cccccccccccccc now interpolate variables cccccc
        if (interp_method.eq.spline) then
         do i=1,cv%num_cv
           rr=rin(i,:)
           call spline_cubic_set(num_rep_in,s,rr,0,0,0,0,rrpp)
           do k=1,num_rep_out
            call spline_cubic_val(num_rep_in,s,rr,rrpp,t(k), &
     & rout(i,k),dum,dum)
           enddo
         enddo
        elseif (interp_method.eq.bspline) then
         do i=1,cv%num_cv
           rr=rin(i,:)
           do k=1,num_rep_out
            call spline_b_val(num_rep_in,s,rr,t(k),rout(i,k))
           enddo
         enddo
        elseif (interp_method.eq.linear) then
         do i=1,cv%num_cv
           rr=rin(i,:)
           call linear_interp(s,rr,num_rep_in,t,rr_out,num_rep_out)
           rout(i,:)=rr_out
         enddo
        else
         call wrndie(0,whoami,trim('NO VALID INTERPOLATION METHODS SELECTED'))
        endif ! interp_method
!
! write output file
        write(fmt,int_format) num_rep_out
        do i=1,cv%num_cv
         write(ofile,'('//fmt//real_format//')')                        &
     & (rout(i,k),k=1,num_rep_out)
        enddo
       endif ! qroot
       end subroutine cv_common_interpolate
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
#endif /* automatically protect all code */
      end module cv_common
!
