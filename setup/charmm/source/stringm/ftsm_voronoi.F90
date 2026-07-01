! ********************************************************************************!
! Preprocessed by GNU CPP !
! ********************************************************************************!
!
! FTSM_VORONOI.MOD
!
! VORONOI TESSELLATION MODULE FOR THE FINITE TEMPERATURE STRING METHOD
! this module is modeled after Voronoi tessellation in SMCV
      module ftsm_voronoi
#if (KEY_STRINGM==1) /*  automatically protect all code */
!
#if (KEY_STRINGM==1)
!
      use chm_kinds
      use ivector
      use ftsm_var
!
      implicit none
!
      private
!
      logical, save, public :: ftsm_voronoi_initialized=.false.
      integer, dimension(:,:,:), pointer, save :: ftsm_voronoi_data ! holds the nine arrays listed below
! local
      integer, dimension(:,:), pointer, save :: cross_attempt ! holds the history of crossing attempts in 1st column
      integer, dimension(:,:), pointer, save :: cross_accept ! history of successful crossing attempts in second column.
      integer, dimension(:), pointer, save :: occupancy ! total # iterations voronoi cells are populated (watch out for max integer value !!!)
! global
      integer, dimension(:,:), pointer, save :: cross_attemptG
      integer, dimension(:,:), pointer, save :: cross_acceptG
      integer, dimension(:), pointer, save :: occupancyG
! old global (for restarting)
      integer, dimension(:,:), pointer, save :: cross_attemptO
      integer, dimension(:,:), pointer, save :: cross_acceptO
      integer, dimension(:), pointer, save :: occupancyO
!
      integer, dimension(:), pointer :: ftsm_voronoi_map ! holds map between process rank and voronoi cell
      type (int_vector), save :: ftsm_voro_log ! logs the history of crossing attempts ( which cells and when ); local to each replica
      integer :: ftsm_voronoi_whereami ! the voronoi cell this replica is inside
      real(chm_real) :: ftsm_voronoi_cut ! voronoi cell cutoff in path-perpendicular direction (beyond which MD replicas are not allowed)
!
      real(chm_real), save, pointer, dimension (:,:,:) :: rall_f, & ! forcing (rall`s will not always be associated so beware)
     & rall_o ! orientation
!
      public ftsm_voronoi_map
      public ftsm_voronoi_whereami
      public ftsm_voronoi_init ! initialize voronoi histogram matrix
      public ftsm_voronoi_done ! deallocate voronoi histogram matrix
      public ftsm_voronoi_compute ! determine which cell we are in using rall_o & rall_f
      public ftsm_voronoi_whereami_compute ! determine cell using x,y,z coordinates
      public ftsm_voronoi_check ! determine whether coordinates should be reversed (include allow_cross case)
      public ftsm_voronoi_update ! update voronoi cell centers
      public ftsm_voronoi_smart_update ! update voronoi cell centers in such a way that the MD replicas remain inside their cells
      public ftsm_voronoi_read_map ! read replica map from file
      public ftsm_voronoi_read_data ! read histograms and occupancy
      public ftsm_voronoi_print_map ! read replica map from file
      public ftsm_voronoi_print_log ! print voronoi log from which a MSM can be built
      public ftsm_voronoi_print_data ! print histograms and occupancy
      public ftsm_voronoi_set_cutoff ! set voronoi cell cutoff
!
      contains
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine ftsm_voronoi_done()
       if (ftsm_voronoi_initialized) then
        deallocate(ftsm_voronoi_map)
        deallocate(ftsm_voronoi_data)
!
        nullify(cross_attempt)
        nullify(cross_accept)
        nullify(occupancy)
!
        nullify(cross_attemptG)
        nullify(cross_acceptG)
        nullify(occupancyG)
!
        nullify(cross_attemptO)
        nullify(cross_acceptO)
        nullify(occupancyO)
!
        call int_vector_done(ftsm_voro_log) ! destroy Voronoi T. log
       endif
       if(associated(rall_f))deallocate(rall_f)
       if (qdiffrot) then ; if(associated(rall_o))deallocate(rall_o) ; endif
!
       ftsm_voronoi_initialized=.false.
       end subroutine ftsm_voronoi_done
!======================================================================================================
       recursive subroutine ftsm_voronoi_init()
!
       if (.not.ftsm_initialized) return ! cannot call ftsm_init because this creates a circular dependency
       if (.not.ftsm_voronoi_initialized) then ! do not reallocate data if already initialized
!
! allocate data for all replicas, if not already allocated elsewhere (voronoi is not the only module that is expected to use rall)
!
        if (.not. associated(rall_f)) allocate(rall_f(nforced,3,nstring))
        if (qorient) then
         if (qdiffrot) then
          if (.not. associated(rall_o)) allocate(rall_o(norient,3,nstring))
         else
          rall_o =>rall_f
         endif !qdiffrot
        endif ! qorient
!
        allocate(ftsm_voronoi_data(nstring,2*nstring+1,3))
! assign pointers -- local
        cross_attempt =>ftsm_voronoi_data(:,1:nstring,1)
        cross_accept =>ftsm_voronoi_data(:,nstring+1:2*nstring,1)
        occupancy=>ftsm_voronoi_data(:,2*nstring+1,1)
! assign pointers -- global
        cross_attemptG =>ftsm_voronoi_data(:,1:nstring,2)
        cross_acceptG =>ftsm_voronoi_data(:,nstring+1:2*nstring,2)
        occupancyG=>ftsm_voronoi_data(:,2*nstring+1,2)
! assign pointers -- data from previous run (old)
        cross_attemptO =>ftsm_voronoi_data(:,1:nstring,3)
        cross_acceptO =>ftsm_voronoi_data(:,nstring+1:2*nstring,3)
        occupancyO=>ftsm_voronoi_data(:,2*nstring+1,3)
!
        allocate(ftsm_voronoi_map(nstring))
!
        call int_vector_reinit(ftsm_voro_log)
!
       endif ! not initialized
!
       ftsm_voronoi_data=0 ! set all crossing data to zero
! cross_acceptO=1 ! zero by default
       ftsm_voronoi_cut=-9999d0
       ftsm_voronoi_whereami=-1 ! -1 indicates this needs to be computed
       ftsm_voronoi_map=-1
!
       ftsm_voronoi_initialized=.true. ! this must be set because it affects the behavior of voronoi_update
       call ftsm_voronoi_update(.true.) ! communicate orientation atoms
!
       end subroutine ftsm_voronoi_init
!===========================================================
       recursive subroutine ftsm_voronoi_update(bcast_orient_)
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
       logical, optional, intent(in) :: bcast_orient_ ! whether orientation coordinates should be broadcast (usually not b/c they only evolve via forced coords)
! locals
       logical :: bcast_orient
       integer :: i, ierror
! begin
       if (present(bcast_orient_)) then ; bcast_orient=bcast_orient_ ; else ; bcast_orient=.false. ; endif
       bcast_orient=bcast_orient.and.qorient.and.qdiffrot ! with qdiffrot false rall_o and rall_f point to the same thing
!
       if (.not.ftsm_voronoi_initialized) then
        call ftsm_voronoi_init() ! init will call update, so we are done after this call returns
       else
        if (MPI_COMM_STRNG.ne.MPI_COMM_NULL.and.SIZE_STRNG.gt.1) then ! roots only
         call MPI_ALLGATHER(r_f(:,:,center),3*nforced,mpifloat, &
     & rall_f,3*nforced,mpifloat,MPI_COMM_STRNG, ierror)
         if (bcast_orient) call MPI_ALLGATHER(r_o(:,:,center),3*norient,mpifloat, &
     & rall_o,3*norient,mpifloat,MPI_COMM_STRNG, ierror)
        endif ! roots only
! broadcast to slaves
        if (MPI_COMM_LOCAL.ne.MPI_COMM_NULL.and.SIZE_LOCAL.gt.1) then
#if (KEY_SINGLE==1)
          call PSND4r(rall_f,3*nforced*nstring)
          if (bcast_orient) call PSND4r(rall_o,3*norient*nstring)
#else
          call PSND8m(rall_f,3*nforced*nstring)
          if (bcast_orient) call PSND8m(rall_o,3*norient*nstring)
#endif
        endif ! MPI_COMM_LOCAL
!
        if (qorient.and.qdiffrot) then ; do i=1, nboth ; rall_o(iatom_both(2,i),:,:)=rall_f(iatom_both(1,i),:,:) ; enddo ; endif ! update overlap coords
! will assume that the coordinates have been shifted to their COM; this needs to be ensured in ftsm
       endif ! ftsm_voronoi_initialized
!
       end subroutine ftsm_voronoi_update
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine ftsm_voronoi_smart_update(x, y, z, xold, yold, zold, m_iter)
       use stream
! ftsm_rall(:,:) is expected to have values from previous update that
! are consistent with rtemp; if this is not true, the routine
! just does a regular update and exits
       use multicom_aux;
       use number
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
       real(chm_real), dimension(:) :: x, y, z, xold, yold, zold
       integer, optional :: m_iter
! locals
!
       real(chm_real), pointer, dimension(:,:,:) :: rall_new_f, rall_new_o, rall_temp_f, rall_temp_o
       real(chm_real), pointer, dimension(:,:) :: rfi, roi, rfi_old, roi_old
       real(chm_real), pointer, dimension(:) :: r_com, r_com_old, ow, wgts
       integer, pointer :: inds(:)
       real(chm_real) :: d
!
       logical :: correct_cell_me, correct_cell
       integer :: iter, max_iter
       integer, parameter :: default_iter=15
       integer :: i, j, k, ierror, me, ind
       logical :: qroot, qgrp
!
       character(len=len("FTSM_VORONOI_SMART_UPDATE>") ),parameter::whoami="FTSM_VORONOI_SMART_UPDATE>";!macro
!
! begin
       if (.not.ftsm_voronoi_initialized) then
        call ftsm_voronoi_init() ! init will call regular update; we are done after this
       else
!ccccccccccccccccc do some work ccccccccccccccccccccccccccccc
! allocate temporary arrays
        allocate(rall_new_f(nforced,3,nstring), rall_temp_f(nforced,3,nstring))
        if (qorient) then
         if (qdiffrot) then
          allocate(rall_new_o(norient,3,nstring), rall_temp_o(norient,3,nstring))
         else
          rall_new_o =>rall_new_f
          rall_temp_o =>rall_temp_f
         endif !qdiffrot
        endif ! qorient
!
        qroot=(MPI_COMM_STRNG.ne.MPI_COMM_NULL.and.SIZE_STRNG.gt.1)
        qgrp=(MPI_COMM_LOCAL.ne.MPI_COMM_NULL.and.SIZE_LOCAL.gt.1)
! gather all main coordinate sets in rall_new
        if (qroot) &
     & call MPI_ALLGATHER(r_f(:,:,center),3*nforced,mpifloat, &
     & rall_new_f,3*nforced,mpifloat,MPI_COMM_STRNG,ierror)
!
! broadcast to slaves
        if (MPI_COMM_LOCAL.ne.MPI_COMM_NULL.and.SIZE_LOCAL.gt.1) then
#if (KEY_SINGLE==1)
          call PSND4r(rall_new_f,3*nforced*nstring)
#else
          call PSND8m(rall_new_f,3*nforced*nstring)
#endif
        endif ! MPI_COMM_LOCAL
! update orientation coords if needed (all CPUs do this)
        if (qorient.and.qdiffrot) then ! update orientation coords (which only evolve via r_f) :
         rall_new_o=rall_o ! copy from previous update step
         do i=1, nboth ; rall_new_o(iatom_both(2,i),:,:)=rall_new_f(iatom_both(1,i),:,:) ; enddo ! update overlap coords
        endif ! qorient
!
        me=mestring+1
! load coordinates via macro
! forcing
        rfi=>r_f(:,:,instant); rfi_old=>r_f(:,:,dummy)
!
!
!
!
!
!
! | macro ftsm_load_fcor.def
! v
! ==========================
  if (ftsm_com_on) then
   do k=1, nforced
    rfi(k,:)=zero
    rfi_old(k,:)=zero
    inds=>iatoms_f%v(k)%i
    wgts=>wgts_f%v(k)%r
    do i=1, iatoms_f%v(k)%last
     ind=inds(i)
     d=wgts(i)
     !
     rfi(k,1)=rfi(k,1)+(d*x(ind));
     rfi(k,2)=rfi(k,2)+(d*y(ind));
     rfi(k,3)=rfi(k,3)+(d*z(ind));
     !
     rfi_old(k,1)=rfi_old(k,1)+(d*xold(ind));
     rfi_old(k,2)=rfi_old(k,2)+(d*yold(ind));
     rfi_old(k,3)=rfi_old(k,3)+(d*zold(ind));
    enddo ! i
   enddo ! k
  else
   do k=1, nforced
    ind=iatom_f(k)
    !
    rfi(k,1)=x(ind)
    rfi(k,2)=y(ind)
    rfi(k,3)=z(ind)
    !
    rfi_old(k,1)=xold(ind)
    rfi_old(k,2)=yold(ind)
    rfi_old(k,3)=zold(ind)
   enddo
  endif
!===========================
! ^
! | macro ftsm_load_fcor.def
! orientation
        roi=>r_o(:,:,instant); roi_old=>r_o(:,:,dummy)
        if (qorient) then ;
         if (qdiffrot) then
!
!
!
!
!
!
! | macro ftsm_load_fcor.def
! v
! ==========================
  if (ftsm_com_on) then
   do k=1, norient
    roi(k,:)=zero
    roi_old(k,:)=zero
    inds=>iatoms_o%v(k)%i
    wgts=>wgts_o%v(k)%r
    do i=1, iatoms_o%v(k)%last
     ind=inds(i)
     d=wgts(i)
     !
     roi(k,1)=roi(k,1)+(d*x(ind));
     roi(k,2)=roi(k,2)+(d*y(ind));
     roi(k,3)=roi(k,3)+(d*z(ind));
     !
     roi_old(k,1)=roi_old(k,1)+(d*xold(ind));
     roi_old(k,2)=roi_old(k,2)+(d*yold(ind));
     roi_old(k,3)=roi_old(k,3)+(d*zold(ind));
    enddo ! i
   enddo ! k
  else
   do k=1, norient
    ind=iatom_o(k)
    !
    roi(k,1)=x(ind)
    roi(k,2)=y(ind)
    roi(k,3)=z(ind)
    !
    roi_old(k,1)=xold(ind)
    roi_old(k,2)=yold(ind)
    roi_old(k,3)=zold(ind)
   enddo
  endif
!===========================
! ^
! | macro ftsm_load_fcor.def
         endif
         ow=>orientWeights
         r_com=>rcom(:,instant); r_com_old=>rcom(:,dummy)
! translate forced atoms to centroid
         r_com=zero; r_com_old=zero
         do j=1,3 ; do k=1, norient;
          r_com(j) = r_com(j)+ow(k)*roi(k,j) ; r_com_old(j) = r_com_old(j)+ow(k)*roi_old(k,j)
         enddo ; enddo
!
         rfi (:,1)=rfi (:,1)-r_com (1) ; rfi (:,2)=rfi (:,2)-r_com (2) ; rfi (:,3)=rfi (:,3)-r_com (3)
         rfi_old(:,1)=rfi_old(:,1)-r_com_old(1) ; rfi_old(:,2)=rfi_old(:,2)-r_com_old(2) ; rfi_old(:,3)=rfi_old(:,3)-r_com_old(3)
!
         if (qdiffrot) then
          roi (:,1)=roi (:,1)-r_com (1) ; roi (:,2)=roi (:,2)-r_com (2) ; roi (:,3)=roi (:,3)-r_com (3)
          roi_old(:,1)=roi_old(:,1)-r_com_old(1) ; roi_old(:,2)=roi_old(:,2)-r_com_old(2) ; roi_old(:,3)=roi_old(:,3)-r_com_old(3)
         endif ! qdiffrot
        endif ! qorient
!
! check if the old set is consistent with current
        correct_cell_me=(me.eq.ftsm_voronoi_compute(rfi_old, roi_old, rall_f, rall_o))
! are the coords in rold consistent with the current V. cell centers?
        if (.not.correct_cell_me) then
! if old set is outside cell, it means that we just crossed into a
! neighbor cell; in this case, test rnew, not rold
! repeat as above:
          rfi_old=>rfi; roi_old=>roi
!
          correct_cell_me=(me.eq.ftsm_voronoi_compute(rfi_old, roi_old, rall_f, rall_o))
        endif ! .not. correct_cell_me
! pool all results
        if (qroot) call MPI_ALLREDUCE(correct_cell_me, correct_cell, 1, &
     & mpibool, MPI_LAND, MPI_COMM_STRNG, ierror)
! broadcast to slaves
        if (qgrp) then ; call mpi_bcast(correct_cell,1,mpibool,0,MPI_COMM_LOCAL,ierror) ; endif
!
        if (.not.correct_cell) then
          call wrndie(0,whoami,trim(' CURRENT VORONOI CELL CENTERS INCONSISTENT WITH COORDINATES.'))
! roll back string
           r_f(:,:,center)=rall_f(:,:,me) ; if (qorient.and.qdiffrot) r_o(:,:,center)=rall_o(:,:,me)
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
          rall_temp_f=rall_new_f ; if (qorient.and.qdiffrot) rall_temp_o=rall_new_o
          do
! check if the new set is consistent
           correct_cell_me=(me.eq.ftsm_voronoi_compute(rfi_old, roi_old, rall_temp_f, rall_temp_o))
! pool all results
           if (qroot) call MPI_ALLREDUCE(correct_cell_me, correct_cell, 1, &
     & mpibool, MPI_LAND, MPI_COMM_STRNG, ierror)
! broadcast to slaves
           if (qgrp) then
!#ifdef 1
! call PSND4(correct_cell,1) !##.not.INTEGER8
! call PSND8(correct_cell,1) !##INTEGER8
!#else
! call mpi_bcast(correct_cell,1,mpiint,0,MPI_COMM_LOCAL,ierror)
!#endif
           endif ! qgrp
           call mpi_bcast(correct_cell,1,mpibool,0,MPI_COMM_LOCAL,ierror)
!
           if (correct_cell) then
            rall_f=rall_temp_f ; if (qorient.and.qdiffrot) rall_o=rall_temp_o
! `roll back' cv's:
            if (iter.gt.0) then ! otherwise they are fine
             r_f(:,:,center)=rall_f(:,:,me) ; if (qorient.and.qdiffrot) r_o(:,:,center)=rall_o(:,:,me)
            endif
            exit
           elseif (iter.ge.max_iter) then
            call wrndie(0,whoami,trim(' MAXIMUM NUMBER OF ITERATIONS EXCEEDED.'))
! reset cv`s to consistent Voronoi cell centers :
            r_f(:,:,center)=rall_f(:,:,me) ; if (qorient.and.qdiffrot) r_o(:,:,center)=rall_o(:,:,me)
            exit
           else
            rall_temp_f=half*(rall_temp_f+rall_f) ; if (qorient.and.qdiffrot) rall_temp_o=half*(rall_temp_o+rall_o)
           endif ! correct cell
           iter=iter+1
          enddo ! <empty>
        endif ! .not. correct_cell
!
       deallocate(rall_temp_f, rall_new_f)
       if (qorient.and.qdiffrot) then ; deallocate(rall_temp_o, rall_new_o) ; else ; nullify(rall_temp_o, rall_new_o) ; endif
!
       endif ! .not. initialized
!
! NOTE : no need to broadcast anything to slaves because they also execute above loop
!
       end subroutine ftsm_voronoi_smart_update
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! this is the function that computes the Voronoi distances
       function ftsm_voronoi_compute(rf, ro, rfall, roall)
      use stream
      use multicom_aux;
      use number
      use mpi
      use parallel, only: psnd4, psnd8
      use bestfit, only : eig3s, RMSBestFit, rmsd, norm3, veccross3
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
       integer :: ftsm_voronoi_compute ! index of Voronoi cell that the system is in
       real(chm_real), intent(in), pointer :: rf(:,:), ro(:,:), rfall(:,:,:), roall(:,:,:)
! locals
       real(chm_real), dimension(nforced,3) :: rf_rot, rfl_rot, rfr_rot ! for computing perpendicular projection
       real(chm_real), dimension(:,:), pointer :: rfl, rfc, rfr, rol, roc, ror
       real(chm_real) :: msd(nstring) ! mean square distance
       real(chm_real) :: d, d1, rho, rho1
       integer :: i, j, k, l, which(1)
       real(chm_real) :: u(3,3,nstring) ! array of nstring-rotation matrices for transforming ro to roall
! parallelization
       logical :: qpara
       integer*4 :: send_displ(SIZE_LOCAL), send_count(SIZE_LOCAL)
       integer :: mbeg, mend, ierror, me
!
! character(len=len("FTSM_VORONOI_COMPUTE>") ),parameter::whoami="FTSM_VORONOI_COMPUTE>";!macro
! initialize, if necessary
       if (.not.ftsm_voronoi_initialized) call ftsm_voronoi_init()
!
! compute RMSD
! need to compute distances w.r.t. all string images
       msd=zero
!
! compute norms in parallel: each slave node is assigned a set of matrices
! compute index limits (this could be done once in the smcv_init routine)
!
       qpara=(MPI_COMM_LOCAL.ne.MPI_COMM_NULL.and.SIZE_LOCAL.gt.1 &
     & .and.calc_voronoi_para)
       if (qpara) then
!
        j=ceiling(one*nstring/SIZE_LOCAL) ! max. number of calculations assigned to a slave node
!
        do i=1,SIZE_LOCAL
         send_displ(i)=min((i-1)*j,nstring-1) ! cannot exceed nstring
         send_count(i)=max(0,min(j,nstring-j*(i-1)))
        enddo
        me=ME_LOCAL+1
! indices are below
        mbeg=send_displ(me)+1
        mend=mbeg+send_count(me)-1
       else
        mbeg=1
        mend=nstring
       endif ! qpara
! call best fitting routine (obtain RMSD from eigenvalues)
! when running in parallel, each slave node computes norms for string replicas mbeg -- mend (and the current coor. set)
       do i=mbeg, mend
        rfc=>rfall(:,:,i)
        if (qorient) then
         roc=>roall(:,:,i)
         call RMSBestFit(ro,roc,orientWeights,u(:,:,i)) ; ! u fits ro onto roall (string)
!
         rf_rot=0d0
         do l=1,3; do k=1,3
          rf_rot(:,k) =rf_rot(:,k) + rfc(:,l)*u(l,k,i) ! rotate rfall (here rfc) to fit roall onto ro (taking u transpose)
         enddo ; enddo
!
        else
         rf_rot=rfc
        endif ! qorient
!
        msd(i)=rmsd(rf_rot,rf,forcedWeights,.false.) ! rf_rot : rotated string atoms
       enddo
! note: at the end of this loop, rf_rot does correspond to the string image closest to ro
! because the loop above is over _ALL_ string points (so always get the last point)
!
! pool norms (and rotation matrices) on local root
       if (qpara) then
        if (ME_LOCAL.eq.0) then
         call MPI_GATHERV(MPI_IN_PLACE, send_count(me), &
     & mpifloat, &
     & msd, send_count, send_displ, &
     & mpifloat, 0, MPI_COMM_LOCAL, ierror)
! send orientation matrices if tube constraint used
         if (qorient.and.ftsm_voronoi_cut.gt.zero) &
     & call MPI_GATHERV(MPI_IN_PLACE, 9*send_count(me), &
     & mpifloat, &
     & u, 9*send_count, 9*send_displ, &
     & mpifloat, 0, MPI_COMM_LOCAL, ierror)
!====================================================
        else ! slaves
         call MPI_GATHERV(msd(mbeg),send_count(me), &
     & mpifloat, &
     & msd, send_count, send_displ, &
     & mpifloat, 0, MPI_COMM_LOCAL, ierror)
! send orientation matrices if tube constraint used
         if (qorient.and.ftsm_voronoi_cut.gt.zero) &
     & call MPI_GATHERV(u(1,1,mbeg), 9*send_count(me), &
     & mpifloat, &
     & u, 9*send_count, 9*send_displ, &
     & mpifloat, 0, MPI_COMM_LOCAL, ierror)
        endif ! me_local
       endif ! qpara
!
       if (ME_LOCAL.eq.0.or..not.qpara) then
        which=minloc(msd) ! minloc needs an integer array;
! now that we know which cell the MD replica belongs to, compute the perpendicular distance to the string (project)
        if (ftsm_voronoi_cut.gt.zero) then
! communicate alignment matrices (assuming this is cheaper than recomputing)
          i=max(1,which(1)-1)
          j=min(nstring,which(1)+1)
          rfc=>rfall(:,:,which(1))
          rfl=>rfall(:,:,i)
          rfr=>rfall(:,:,j)
!
          if (qorient) then
           rol=>roall(:,:,i)
           ror=>roall(:,:,j)
!
           rfr_rot=0d0; rfl_rot=0d0; rf_rot=0d0
           do l=1,3; do k=1,3
            rf_rot(:,k) =rf_rot(:,k) +rfc(:,l)*u(l,k,which(1)) ! reorient center image
!
            rfl_rot(:,k)=rfl_rot(:,k)+rfl(:,l)*u(l,k,i) ! eqv. u^t x rfl, which rotates rfl so that rol fits ro
            rfr_rot(:,k)=rfr_rot(:,k)+rfr(:,l)*u(l,k,j)
           enddo; enddo
!
          else ! (not qorient)
           rfl_rot=rfl ; rfr_rot=rfr
          endif ! qorient
! compute projection
          rho=zero ; rho1=zero
          do k=1,3 ; do j=1,nforced
                    d = rf(j,k) -rf_rot(j,k) ! disp. vector to string point
                    d1= rfr_rot(j,k)-rfl_rot(j,k) ! disp. vector between neighboring string points
!
                    rho =rho + d *d1 *forcedWeights(j)
                    rho1=rho1 + d1*d1 *forcedWeights(j)
          enddo ; enddo
!
          if (rho1.le.RSMALL) then ; rho1=1d0 ; else ; rho1=1d0/rho1 ; endif ! a very unlikely event
!
! check cutoff
!
         if ( ( msd(which(1)) - rho*rho*rho1 ).gt.ftsm_voronoi_cut**2) &
     & which(1)=-which(1) ! outside of the allowed region (special code)
        endif ! cutoff > 0
       endif ! ME_LOCAL=0 / !qpara
!
       if (qpara) then
#if (KEY_INTEGER8==0)
        call PSND4(which,1) 
#endif
#if (KEY_INTEGER8==1)
        call PSND8(which,1) 
#endif
       endif
!
!aa
! if (ME_LOCAL.eq.0) then
! write(ME_STRNG+200,*) minloc(msd), rho, rho1, rho*rho*rho1, ftsm_voronoi_cut**2, which(1), qpara, mbeg, mend
! write(ME_STRNG+200,*) msd
! close(ME_STRNG+200)
! endif
! aa
!
       ftsm_voronoi_compute=which(1)
!
       end function ftsm_voronoi_compute
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine ftsm_voronoi_whereami_compute(x,y,z)
       use stream
       use number
       use multicom_aux; ! aa
!
       real(chm_real) :: x(:), y(:), z(:)
! locals
       real(chm_real), pointer, dimension(:,:) :: rfi, roi
       real(chm_real), pointer, dimension(:) :: r_com, ow, wgts
       integer :: i, j, k, ind
       integer, pointer :: inds(:)
       real(chm_real) :: d
!
! character(len=len("FTSM_VORONOI_WHEREAMI_COMPUTE>") ),parameter::whoami="FTSM_VORONOI_WHEREAMI_COMPUTE>";!macro
! make sure string coordinates have been defined
!
       if (.not.ftsm_voronoi_initialized) call ftsm_voronoi_init()
!
! populate instantaneous orientation and forcing coordinates
! load coordinates (code dup, alas)
! forcing
       rfi=>r_f(:,:,instant)
!
!
!
!
!
!
! | macro ftsm_load_fcor.def
! v
! ==========================
  if (ftsm_com_on) then
   do k=1, nforced
    rfi(k,:)=zero
    inds=>iatoms_f%v(k)%i
    wgts=>wgts_f%v(k)%r
    do i=1, iatoms_f%v(k)%last
     ind=inds(i)
     d=wgts(i)
     !
     rfi(k,1)=rfi(k,1)+(d*x(ind));
     rfi(k,2)=rfi(k,2)+(d*y(ind));
     rfi(k,3)=rfi(k,3)+(d*z(ind));
    enddo ! i
   enddo ! k
  else
   do k=1, nforced
    ind=iatom_f(k)
    !
    rfi(k,1)=x(ind)
    rfi(k,2)=y(ind)
    rfi(k,3)=z(ind)
   enddo
  endif
!===========================
! ^
! | macro ftsm_load_fcor.def
! orientation
       roi=>r_o(:,:,instant)
       if (qorient) then ;
         if (qdiffrot) then
!
!
!
!
!
!
! | macro ftsm_load_fcor.def
! v
! ==========================
  if (ftsm_com_on) then
   do k=1, norient
    roi(k,:)=zero
    inds=>iatoms_o%v(k)%i
    wgts=>wgts_o%v(k)%r
    do i=1, iatoms_o%v(k)%last
     ind=inds(i)
     d=wgts(i)
     !
     roi(k,1)=roi(k,1)+(d*x(ind));
     roi(k,2)=roi(k,2)+(d*y(ind));
     roi(k,3)=roi(k,3)+(d*z(ind));
    enddo ! i
   enddo ! k
  else
   do k=1, norient
    ind=iatom_o(k)
    !
    roi(k,1)=x(ind)
    roi(k,2)=y(ind)
    roi(k,3)=z(ind)
   enddo
  endif
!===========================
! ^
! | macro ftsm_load_fcor.def
         endif
         r_com=>rcom(:,instant)
         ow=>orientWeights
! translate forced atoms to centroid
         r_com=zero;
         do j=1,3 ; do k=1, norient;
          r_com(j) = r_com(j)+ow(k)*roi(k,j)
         enddo ; enddo
         rfi(:,1)=rfi(:,1)-r_com(1) ; rfi(:,2)=rfi(:,2)-r_com(2) ; rfi(:,3)=rfi(:,3)-r_com(3)
         if (qdiffrot) then
          roi(:,1)=roi(:,1)-r_com(1) ; roi(:,2)=roi(:,2)-r_com(2) ; roi(:,3)=roi(:,3)-r_com(3)
         endif ! qdiffrot
       endif ! qorient
!
       ftsm_voronoi_whereami=ftsm_voronoi_compute(rfi, roi, rall_f, rall_o)
!
       end subroutine ftsm_voronoi_whereami_compute
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine ftsm_voronoi_print_data(iunit)
! assume that unit is prepared
! NOTE that this is a global print!
       use stream
       use multicom_aux;
       use mpi
!
       integer iunit
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
!
       if (.not.ftsm_voronoi_initialized) call ftsm_voronoi_init()
!
       if (MPI_COMM_STRNG.ne.MPI_COMM_NULL) then
        if (SIZE_STRNG.gt.1) then
         call MPI_ALLREDUCE(ftsm_voronoi_data(:,:,1),voro_data_all, &
     & nstring*(2*nstring+1),type,MPI_SUM,MPI_COMM_STRNG,ierror)
        else
         voro_data_all=ftsm_voronoi_data(:,:,1)
        endif ! SIZE
!
        if (ME_STRNG.eq.0) then ! string root writes
         write(fmt,int_format) nstring
         do j=1,nstring
          write(iunit,'('//fmt//int_format//')')                        &
     & voro_data_all(j,1:nstring)+cross_attemptO(j,:) ! crossing attemps
         enddo
         write(iunit,'("%")') ! break
         do j=1,nstring
          write(iunit,'('//fmt//int_format//')')                        &
     & voro_data_all(j,nstring+1:2*nstring)+cross_acceptO(j,:) ! crossing accepts
         enddo
         write(iunit,'("%")') ! break
         write(iunit,'('//fmt//int_format//')')                         &
     & voro_data_all(:,2*nstring+1)+occupancyO(:) ! occupancy
        endif ! ME
       endif ! MPI_COMM_STRNG
!
       end subroutine ftsm_voronoi_print_data
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine ftsm_voronoi_read_data(iunit)
! assume that unit is prepared
       use multicom_aux;
       use mpi
       use parallel, only: psnd4, psnd8
!
       integer iunit
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
       if (.not.ftsm_voronoi_initialized) call ftsm_voronoi_init()
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
        ftsm_voronoi_data(:,:,3)=voro_data_all ! place into "old" position
!
        if (SIZE_STRNG.gt.1) then
         call mpi_bcast(ftsm_voronoi_data(:,:,3),SIZE_STRNG*(2*SIZE_STRNG+1),mpiint,0,MPI_COMM_STRNG,ierror)
        endif
       endif ! MPI_COMM_STRNG
! broadcast to slaves
       if (MPI_COMM_LOCAL.ne.MPI_COMM_NULL.and.SIZE_LOCAL.gt.1) &
#if (KEY_INTEGER8==0)
     & call PSND4r(ftsm_voronoi_data(:,:,3),nstring*(2*nstring+1)) 
#endif
#if (KEY_INTEGER8==1)
     & call PSND8m(ftsm_voronoi_data(:,:,3),nstring*(2*nstring+1)) 
#endif
!
       end subroutine ftsm_voronoi_read_data
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine ftsm_voronoi_print_log(iunit)
! assume that unit is prepared
! NOTE that this is a global print!
! this routine is redundant with ftsm_voronoi_print_hist
       use multicom_aux;
       use mpi
!
       integer iunit
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
       if (.not.ftsm_voronoi_initialized) call ftsm_voronoi_init()
!
       if (MPI_COMM_STRNG.ne.MPI_COMM_NULL.and.SIZE_STRNG.gt.1) then
! calculate size of logs
        voro_log_size8=0
! call MPI_GATHER(ftsm_voro_log%last,1,type,
! & voro_log_size8,1,type,
! & 0,MPI_COMM_WORLD,ierror)
        call MPI_ALLGATHER(ftsm_voro_log%last,1,mpiint, &
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
! call MPI_GATHERV(ftsm_voro_log%i,ftsm_voro_log%last,type,
! & voro_log_all,voro_log_size4,voro_log_disp4,type,
! & 0,MPI_COMM_WORLD,ierror)
        call MPI_ALLGATHERV(ftsm_voro_log%i,ftsm_voro_log%last,mpiint, &
     & voro_log_all,voro_log_size4,voro_log_disp4,mpiint,&
     & MPI_COMM_STRNG,ierror)
!
        if (ME_STRNG.eq.0) write(iunit) voro_log_all
!
        call int_vector_reinit(ftsm_voro_log) ! erase log
        deallocate(voro_log_all)
       endif ! STRNG
       end subroutine ftsm_voronoi_print_log
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      function ftsm_voronoi_check(x,y,z,itime)
! code based mainly on smcv_voronoi_compute
!
      use lu ! for computing FE
      use stream
      use multicom_aux;
      use mpi
      use clcg_mod, only: random; use reawri, only: iseed
      use number
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
      real(chm_real) :: x(:), y(:), z(:)
! real(chm_real) :: mass(size(x,1)) ! assumed size
      integer :: itime ! timestep -- needed by new version of Voronoi
! local var.
      real(chm_real), pointer, dimension(:,:) :: rfi, roi
      real(chm_real), pointer, dimension(:) :: r_com, ow, wgts
      integer :: i, j, k, l, m, which, me, ind
      integer*4 :: ierror, m_
      integer*4 :: length(nstring-1)
      integer*4 :: request(nstring-1)
      logical :: ftsm_voronoi_check ! returns false if the algorithm tells to revert momenta
!
      integer, pointer :: vtemp(:), vtemp2(:) ! for gathering Voronoi stats; this is an upper bound
      integer*4 :: stat(MPI_STATUS_SIZE)
      logical :: voronoi_update
      logical :: success, qgrp, qstring, ready(nstring-1), ok
      real(chm_real) :: P_accept_cross, d
      integer, pointer :: inds(:)
! for computing FE
! real(chm_real) :: flux(nstring, nstring), ! normalized probability flux
! & prob(nstring), ! probability of being in a Voronoi cell
! & netflux(nstring) ! net probability flux into a Voronoi cell
! integer :: permut(nstring), d, code ! for LU decomposition routine
!
! character(len=len("FTSM_VORONOI_CHECK>") ),parameter::whoami="FTSM_VORONOI_CHECK>";!macro
!
      if (.not.ftsm_voronoi_initialized) call ftsm_voronoi_init()
!
      qstring=(MPI_COMM_STRNG.ne.MPI_COMM_NULL).and.(SIZE_STRNG.gt.1)
      qgrp=(MPI_COMM_LOCAL.ne.MPI_COMM_NULL).and.(SIZE_LOCAL.gt.1)
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! pool crossing+occupancy
      voronoi_update=voronoi_allow_cross.and.(voronoi_update_freq.gt.0)
      if (voronoi_update) voronoi_update=(mod(itime,voronoi_update_freq).eq.0)
      if (voronoi_update) then
! only roots
         if (qstring) then
! pack
          allocate(vtemp(3*nstring*(2*nstring+1))) ! upper bound
          k=0
          do i=1, nstring
! do j=1, 2*nstring+1 ! send everything
           do j=2*nstring+1, 2*nstring+1 ! send just the occupancy: this means FE cannot be computed
            l=ftsm_voronoi_data(i,j,1) ! voronoi data hosts targets of cross_attemps, cross_accept,
            if (l.gt.0) then
             k=k+1; vtemp(k)=i; k=k+1; vtemp(k)=j; k=k+1; vtemp(k)=l
            endif
           enddo
          enddo
!
          if (ME_STRNG.ne.0) then
           call MPI_ISEND(vtemp, k, & ! send local attempts, accepts + occupancy to root packed in vtemp
     & mpiint, 0, ME_STRNG, MPI_COMM_STRNG, request(ME_STRNG), ierror)
          else ! rank 0
! wait for all messages to begin sending
           ready=.false.
           do while (any(.not.ready))
            do m=1,nstring-1
             if (.not.ready(m)) then
              ok=.false. ! this is necessary
              m_=m ! cast
              call MPI_IPROBE(m_, m_, MPI_COMM_STRNG, ok, stat, ierror)
              if (ok) then
! get message length
                call MPI_Get_count(stat,mpiint,length(m),ierror)
! write(600+ME_STRNG,*) ok, ready(m), m, length(m), stat
                ready(m)=.true.
              endif ! ok
             endif ! .not.ready(m)
            enddo ! m=1,nstring-1
           enddo ! now have all the lengths
! close(600+ME_STRNG) !aa
! begin receiving
           l=k+sum(length)
           allocate(vtemp2(l))
           do m=1,nstring-1
            ind=k+sum(length(1:m-1))+1
            call MPI_IRECV(vtemp2(ind), length(m), &
     & mpiint, m, m, MPI_COMM_STRNG, request(m), ierror)
           enddo ! m
! copy vtemp into vtemp2
           vtemp2(1:k)=vtemp(1:k)
          endif ! ME_STRNG
         endif ! MPI_COMM_STRNG
      endif ! voro_update
! do some work
!
! calculate instantaneous string orientation/forced coordinates (r_o, r_f) based on x, y, z (repeated code for now)
! note: in a typical V. calculation, this will be the only string routine called at every step, so we can assume
! the coordinates need to be calculated
! forcing
       rfi=>r_f(:,:,instant)
!
!
!
!
!
!
! | macro ftsm_load_fcor.def
! v
! ==========================
  if (ftsm_com_on) then
   do k=1, nforced
    rfi(k,:)=zero
    inds=>iatoms_f%v(k)%i
    wgts=>wgts_f%v(k)%r
    do i=1, iatoms_f%v(k)%last
     ind=inds(i)
     d=wgts(i)
     !
     rfi(k,1)=rfi(k,1)+(d*x(ind));
     rfi(k,2)=rfi(k,2)+(d*y(ind));
     rfi(k,3)=rfi(k,3)+(d*z(ind));
    enddo ! i
   enddo ! k
  else
   do k=1, nforced
    ind=iatom_f(k)
    !
    rfi(k,1)=x(ind)
    rfi(k,2)=y(ind)
    rfi(k,3)=z(ind)
   enddo
  endif
!===========================
! ^
! | macro ftsm_load_fcor.def
! orientation
       roi=>r_o(:,:,instant)
       if (qorient) then ;
         if (qdiffrot) then
!
!
!
!
!
!
! | macro ftsm_load_fcor.def
! v
! ==========================
  if (ftsm_com_on) then
   do k=1, norient
    roi(k,:)=zero
    inds=>iatoms_o%v(k)%i
    wgts=>wgts_o%v(k)%r
    do i=1, iatoms_o%v(k)%last
     ind=inds(i)
     d=wgts(i)
     !
     roi(k,1)=roi(k,1)+(d*x(ind));
     roi(k,2)=roi(k,2)+(d*y(ind));
     roi(k,3)=roi(k,3)+(d*z(ind));
    enddo ! i
   enddo ! k
  else
   do k=1, norient
    ind=iatom_o(k)
    !
    roi(k,1)=x(ind)
    roi(k,2)=y(ind)
    roi(k,3)=z(ind)
   enddo
  endif
!===========================
! ^
! | macro ftsm_load_fcor.def
         endif
         ow=>orientWeights
         r_com=>rcom(:,instant)
! translate forced atoms to centroid
         r_com=zero;
         do j=1,3 ; do k=1, norient;
          r_com(j) = r_com(j)+ow(k)*roi(k,j)
         enddo ; enddo
         rfi(:,1)=rfi(:,1)-r_com(1) ; rfi(:,2)=rfi(:,2)-r_com(2) ; rfi(:,3)=rfi(:,3)-r_com(3)
         if (qdiffrot) then
          roi(:,1)=roi(:,1)-r_com(1) ; roi(:,2)=roi(:,2)-r_com(2) ; roi(:,3)=roi(:,3)-r_com(3)
         endif ! qdiffrot
       endif ! qorient
! do the actual check later, first, check on the communication
!
      if (voronoi_update) then
! root waits for all messages; then sends back concatenated array
       if (qstring) then
        if (ME_STRNG.eq.0) then
! wait for all messages to arrive
         call MPI_WAITALL(nstring-1, request, MPI_STATUSES_IGNORE, ierror)
! now send received array to all cpus:
         do m=1, nstring-1
           call MPI_ISEND(vtemp2, k+sum(length), & ! send local attempts, accepts + occupancy to root packed in temp
     & mpiint, m, m, MPI_COMM_STRNG, request(m), ierror)
         enddo
        else ! other roots
! first make sure previous send was successful
         call MPI_WAIT(request(ME_STRNG), stat, ierror)
! test for message from root:
         ok=.false.
         do while (.not.ok)
          call MPI_IPROBE(0, ME_STRNG, MPI_COMM_STRNG, ok, stat, ierror)
         enddo
! a message is ready to be received : begin
! get message length
         call MPI_Get_count(stat,mpiint,length(1),ierror)
         l=length(1)
! begin receiving
         allocate(vtemp2(l))
         call MPI_IRECV(vtemp2, l, mpiint, &
     & 0, ME_STRNG, MPI_COMM_STRNG, request(ME_STRNG), ierror)
        endif ! ME
       endif ! string root
      endif ! voro_update
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! do some more work
      which=ftsm_voronoi_compute(rfi, roi, rall_f, rall_o)
      me=ftsm_voronoi_whereami
! receive the rest
      if (voronoi_update) then
        if (qstring) then
         if (ME_STRNG.ne.0) call MPI_WAIT(request(ME_STRNG), stat, ierror) ! make sure message is received
! unpack message (all string roots do this)
         ftsm_voronoi_data(:,:,2)=ftsm_voronoi_data(:,:,3) ! statistics from previous runs (if any)
         k=0
         do while (k.lt.l)
           k=k+1; i=vtemp2(k); k=k+1; j=vtemp2(k); k=k+1;
           ftsm_voronoi_data(i,j,2)=ftsm_voronoi_data(i,j,2)+vtemp2(k)
         enddo
         if (ME_STRNG.eq.0) call MPI_WAITALL(nstring-1, request, &
     & MPI_STATUSES_IGNORE, ierror)
         deallocate(vtemp, vtemp2)
        endif ! string roots
!
        if (MPI_COMM_STRNG.ne.MPI_COMM_NULL.and.SIZE_STRNG.eq.1) &
     & ftsm_voronoi_data(:,:,2)=ftsm_voronoi_data(:,:,1)
!
      endif ! voronoi_update
!
      if (which.lt.0) then
        ftsm_voronoi_check=.false. ! cutoff exceeded ; reflect ; do not update log
      elseif (which.eq.me) then ! stayed in the same cell as before
       occupancy(which)=occupancy(which)+1 ! update occupancy log
       ftsm_voronoi_check=.true.
      else ! crossed into a different cell
       cross_attempt(me,which)=cross_attempt(me,which)+1
!
       if (voronoi_allow_cross.and.itime.gt.voronoi_nocross_ini) then
! decide whether to allow the crossing
! roots decide
        if (MPI_COMM_STRNG.ne.MPI_COMM_NULL) then
! do i=1,nstring
! if (occupancyG(i).eq.0) occupancyG(i)=1
! enddo
! dummy=1d0/sum(occupancyG)
!************************* acceptance criterion ********************
! P_accept_cross=
! & (sum(1d0*cross_acceptG(:,me)/occupancyG(:) )*
! & occupancyG(me) ! note: cross_accept(i,i)=0
! & -(sum(cross_acceptG(me,:))-cross_acceptG(me,which)))/
! & (max(cross_attemptG(me,which),1))
!********************************************************************
! & ( sum(1d0*cross_attemptG(:,me)/occupancyG(:))-
! & sum(1d0*cross_attemptG(me,:))/occupancyG(me) )
! & /
! & ( sum(1d0*cross_attemptG(:,which)/occupancyG(:))-
! & sum(1d0*cross_attemptG(which,:))/occupancyG(which))
!********************************************************************
! & sum(1d0*cross_attemptG(:,me)-cross_attemptG(me,:))/
! & sum(1d0*cross_attemptG(:,which)-cross_attemptG(which,:))
!********************************************************************
! compute free energy from crossing attemps
! do i=2,nstring
! flux(i,:)=
! & 1d0*cross_attemptG(:,i)/occupancyG(:)
! flux(i,i)=
! & -1d0*sum(cross_attemptG(i,:))/occupancyG(i)
! if (flux(i,i-1).eq.0d0) flux(i,i-1)=
! & 1d0/occupancyG(i-1) ! crude regularization
! if (flux(i-1,i).eq.0d0) flux(i-1,i)=
! & 1d0/occupancyG(i)
! enddo
! netflux(2:nstring)=0d0;
! netflux(1)=1d0 ! boundary condition (p(1)=1)
! flux(1,1)=1d0; flux(1,2:nstring)=0d0 ! boundary condition
! prob=netflux;
! call ludcmp(flux, nstring, permut, d, code)
! call lubksb(flux, nstring, permut, prob)
!
         P_accept_cross=one &
     & * (2-abs(which-me)) & ! allow crosses only between adjacent cells (ad hoc)
! & *prob(me)/prob(which) ! free energy estimate
! & *exp(10d0*(occupancyG(me)/occupancyG(which)-1d0))
     & *(occupancyG(me)/max(occupancyG(which),1))
!********************************************************************
         if (P_accept_cross.ge.1) then
          success=.true.
         else
          success=(random(iseed).le.P_accept_cross)
         endif
        endif ! string roots
! write(600+ME_STRNG,*) P_accept_cross, success, ' * ', prob
! broadcast to slaves
        if (qgrp) call PSND4(success,1)
! if successful, update
        if (success) then
         cross_accept(me,which)=cross_accept(me,which)+1
         occupancy(which)=occupancy(which)+1
         ftsm_voronoi_whereami=which
        else
         occupancy(me)=occupancy(me)+1
        endif
!
        ftsm_voronoi_check=success
       else ! crossing not allowed, so reflect
        ftsm_voronoi_check=.false.
        occupancy(me)=occupancy(me)+1 ! update occupancy log
       endif ! voronoi_allow_cross
! update voronoi log.
       j=ME_STRNG+1
       i=int_vector_add(ftsm_voro_log, j) ! replica ID
       i=int_vector_add(ftsm_voro_log, me) ! from this cell
       i=int_vector_add(ftsm_voro_log, which) ! into this cell
       i=int_vector_add(ftsm_voro_log, ftsm_voronoi_whereami) ! now in this cell
       i=int_vector_add(ftsm_voro_log, itime+vtime_offset) ! at this time
      endif ! which<0
!
      end function ftsm_voronoi_check
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine ftsm_voronoi_set_cutoff(cut)
       use number
       real(chm_real) :: cut
       if (cut.gt.zero) then ! reject invalid values without warning
        if (.not. ftsm_voronoi_initialized) call ftsm_voronoi_init()
        ftsm_voronoi_cut=cut
       endif
       end subroutine ftsm_voronoi_set_cutoff
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine ftsm_voronoi_print_map(iunit,fmt)
       use stream
       use multicom_aux;
       use mpi
       integer :: iunit
! integer :: ierr
       character(len=*), optional :: fmt
! local
!#include "mpitype.def"
!
       integer :: i
       character(len=80) :: frm
       character(len=len("FTSM_VORONOI_PRINT_MAP>") ),parameter::whoami="FTSM_VORONOI_PRINT_MAP>";!macro
! begin
       if (.not.ftsm_voronoi_initialized) call ftsm_voronoi_init()
!
! if (MPI_COMM_STRNG.ne.MPI_COMM_NULL.and.SIZE_STRNG.gt.1) then
! call MPI_ALLGATHER(ftsm_voronoi_whereami, 1, mpiint,
! & ftsm_voronoi_map, 1, mpiint, MPI_COMM_STRNG, ierr)
!
! assume that the map has been communicated elsewhere
        if (ME_STRNG.eq.0) then
         if (.not.present(fmt)) then
          write(frm,'("(",I5,"I5)")') nstring
         else
          frm=fmt
         endif
         write(iunit,frm) (/ (i, i=1,nstring) /)
         write(iunit,frm) ftsm_voronoi_map(1:nstring)
        endif ! ME_STRNG
! endif ! MPI_COMM_STRNG
!
       end subroutine ftsm_voronoi_print_map
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine ftsm_voronoi_read_map(iunit)
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
       integer :: iunit, ierror
       character(len=len("FTSM_VORONOI_READ_MAP>") ),parameter::whoami="FTSM_VORONOI_READ_MAP>";!macro
! begin
       if (.not.ftsm_voronoi_initialized) call ftsm_voronoi_init()
!
       if (MPI_COMM_STRNG.ne.MPI_COMM_NULL) then
        if (ME_STRNG.eq.0) then
         read(iunit,*) ftsm_voronoi_map(1:nstring) ! first row contains indices 0 -- nstring-1
         read(iunit,*) ftsm_voronoi_map(1:nstring) ! second row is what we want
         if (any(ftsm_voronoi_map.lt.0)) call wrndie(0,whoami,trim(' READ NEGATIVE RANK.'))
        endif ! ME_
        if (SIZE_STRNG.gt.1) then
         call mpi_bcast(ftsm_voronoi_map,nstring,mpiint,0,MPI_COMM_STRNG,ierror)
        endif
       endif ! MPI_COMM_STRNG
! broadcast to slave nodes
       if (ME_LOCAL.ne.MPI_UNDEFINED.and.SIZE_LOCAL.gt.1) &
! & call MPI_BCAST(ftsm_voronoi_map, nstring, MPI_INTEGER,
! & 0,MPI_COMM_LOCAL,ierr)
#if (KEY_INTEGER8==0)
     & call PSND4(ftsm_voronoi_map,nstring) 
#endif
#if (KEY_INTEGER8==1)
     & call PSND8(ftsm_voronoi_map,nstring) 
#endif
!
       end subroutine ftsm_voronoi_read_map
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
#endif
#endif /* automatically protect all code */
      end module ftsm_voronoi
