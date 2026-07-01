! ********************************************************************************!
! Preprocessed by GNU CPP !
! ********************************************************************************!
! reparametrization module for finite-temperature string module
      module ftsm_rep
#if (KEY_STRINGM==1) /*  automatically protect all code */
!
#if (KEY_STRINGM==1)
!
      use chm_kinds
      use ftsm_var
      implicit none
!
      private
!
  character(len=132)::info(21)=(/'','','','','','','','','','','','','','','','','','','','',''/);! output buffer
!=====================================================================
! SUBROUTINES
      public ftsm_repa_init
      public ftsm_repa
      public ftsm_compute_arcl_curv
!
      contains
!=======================================================================
      subroutine ftsm_repa_init(comlyn, comlen)
!
      use stream
      use string
#if (KEY_MULTICOM==1)
      use multicom_aux; 
#endif
      use mpi
!
      CHARACTER(LEN=*) :: COMLYN
      integer :: COMLEN
      character(len=len("FTSM_REPA_INIT>") ),parameter::whoami="FTSM_REPA_INIT>";!macro
      logical :: qprint, qroot
!
      qroot=MPI_COMM_STRNG.ne.MPI_COMM_NULL
      qprint=qroot.and.ME_STRNG.eq.0
!
      if (qprint) then ; write(info,8002) whoami ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
      call smcv_repa_init(comlyn, comlen)
!
 8002 format(/A,' USING SMCV INITIALIZATION ROUTINE')
!
      end subroutine ftsm_repa_init
!======================================================================================
      subroutine ftsm_repa(qbcast)
! the rather convoluted approach in this routine stems from the original implementation
! idea to modify orientation coordinates only if required, that is, only those that are
! also forcing coordinates; coordinates that are _purely_ orientation are left untouched
! for this reason, if diffrot=f, ro1 points to rf (since rf is modified by interp), and
! otherwise ro1 is a copy of the ref coords
! ===> for ftsm version 2, we need to keep the string perfectly aligned because the
! neighbor images are not aligned with the instantaneous structure, as in version 1
!
      use string
#if (KEY_MULTICOM==1)
      use multicom_aux; 
#endif
      use mpi
      use stream
      use number
      use consta
      use bestfit, only : eig3s, RMSBestFit, rmsd, norm3, veccross3
      use ftsm_util
      use parallel, only: psnd4, psnd8
!
! local variables
!
      character(len=len("FTSM_REPA>") ),parameter::whoami="FTSM_REPA>";!macro
      logical :: qroot, qslave, qprint
      logical, optional :: qbcast
      real(chm_real) :: u(3,3)=Id3
      real(chm_real), pointer :: r_o_com(:) ! COM vector
      real(chm_real), pointer, dimension(:,:) :: ro, rf, ro1
      real(chm_real) :: w
      real(chm_real) :: weights(nforced,3) ! assuming nforced is reasonably defined
      integer*4 :: stat(MPI_STATUS_SIZE)
      integer :: i, ierror
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
      interface
        subroutine interp_driver_sci(rin,rout,wgt,n, &
     & interp_method,tol,max_iterations,d_arclength, curvature, &
     & dst_cutoff, dr,r_bc_0, r_bc_1)
      use chm_kinds
      use stream
        integer n
        real(chm_real) rin(n), rout(n), wgt(n)
        integer, intent(in) :: interp_method
        integer max_iterations
        real(chm_real) :: tol, d_arclength(:), curvature(:)
        real(chm_real), optional :: dst_cutoff
        real(chm_real), optional :: dr(n) ,r_bc_0(n), r_bc_1(n)
        end subroutine interp_driver_sci
!
        subroutine interp_linear_exact(rin,rout,wgt,n, &
     & d_arclength, curvature, &
     & drout, &
     & r_bc_0, r_bc_1)
      use chm_kinds
        integer :: n
        real(chm_real) :: rin(n), rout(n), wgt(n)
        real(chm_real) :: d_arclength(:), curvature(:)
        real(chm_real), optional :: drout(n) ! optional computation of tangent
        real(chm_real) , optional :: r_bc_0(n), r_bc_1(n) ! optional fixed bc data
       end subroutine interp_linear_exact
!
      end interface
!
      qroot =MPI_COMM_STRNG.ne.MPI_COMM_NULL.and.SIZE_STRNG.gt.1
      qslave=MPI_COMM_LOCAL.ne.MPI_COMM_NULL.and.SIZE_LOCAL.gt.1
      qprint=MPI_COMM_STRNG.ne.MPI_COMM_NULL.and.ME_STRNG.eq.0
!
      if (present(qbcast)) qslave=qslave.and.qbcast ! in case qbcast false do not broadcast to slaves
!
! check if the user has made an initialization call
!
      if (.not.repa_initialized) then
       call wrndie(0,whoami,trim('NO REPARAMETRIZATION OPTIONS SELECTED. NOTHING DONE.'))
       return
      endif
      if (qprint) then
       write(info,690) whoami ; if(prnlev.ge. 5) write(OUTU,'(A)') pack(info,info.ne.'');info='';
      endif
 690 format(A,' PERFORMING STRING REPARAMETRIZATION.')
!
! shorthand
      rf => r_f(:,:,center)
      ro => r_o(:,:,center)
      ro1 => r_o(:,:,dummy)
      r_o_com=>rcom(:,center)
!
      if (qroot) then
!
       if (qorient) then
! translate structure to centroid
! note: this should not be needed because the structure should always be kept COM-free
! but is kept in case things change in the future (overhead of repa is small)
        r_o_com=zero
        do i=1, norient
         w=orientWeights(i)
         r_o_com(1)=r_o_com(1) + w * ro(i,1)
         r_o_com(2)=r_o_com(2) + w * ro(i,2)
         r_o_com(3)=r_o_com(3) + w * ro(i,3)
        enddo
! orientation atoms ! comment out to avoid unneeded roundoff error with qdiffrot=T
! ro(:,1)=ro(:,1)-r_o_com(1)
! ro(:,2)=ro(:,2)-r_o_com(2)
! ro(:,3)=ro(:,3)-r_o_com(3)
! forced atoms (same as orientation atoms if qdiffrot F)
        if (qdiffrot) then
         rf(:,1)=rf(:,1)-r_o_com(1)
         rf(:,2)=rf(:,2)-r_o_com(2)
         rf(:,3)=rf(:,3)-r_o_com(3)
        endif
!ccccccccccc orientation ccccccc
! send/receive orientation structure
! this is a slow procedure, as the orientation is done
! sequentially
!
        if (mestring.gt.0) then
         call MPI_RECV(ro1,3*norient,mpifloat,mestring-1, 1, &
     & MPI_COMM_STRNG, stat, ierror)
! orient current structure
! call RMSBestFit(ro,ro1,orientWeights,u)
         call RMSBestFit(ro1,ro,orientWeights,u) ! this computes u^T, saving us a transposition
! transform current structure (ro) to overlap with reference (ro1)
! (if orientation is off, u=I)
         if (.not.qdiffrot) then ; ro1=>rf ! actually, rf/ro point to the same thing in this case
         elseif (qver2) then ; ro1=>ro ! modify orientation coords
         endif
         ro1(:,1)=ro(:,1)-r_o_com(1)
         ro1(:,2)=ro(:,2)-r_o_com(2)
         ro1(:,3)=ro(:,3)-r_o_com(3)
!
! u=transpose(u) ! see transposition above
         ro1=matmul(ro1,u)
         if (qdiffrot) rf=matmul(rf,u)
!
        else ! first replica enters below
! ro1=>ro ! make sure 1st replica sends the center, not dummy coords
! technically, the above line is correct (since ro is com-free; com is never subtracted with qdiffrot=false above)
! however, conceptually, the COM _should_ be subtracted as in the loop above (e.g. note ro1=>rf, which can modify both rf and ro)
! I also do not like the fact that the code below involves copying, which slows down the first replica!
         if (.not.qdiffrot) then ; ro1=>rf ! will remove the COM from actual ro/rf coordinates below (they are modified anyway by interp)
         elseif (qver2) then ; ro1=>ro ! modify orientation coords
         endif
         ro1(:,1)=ro(:,1)-r_o_com(1)
         ro1(:,2)=ro(:,2)-r_o_com(2)
         ro1(:,3)=ro(:,3)-r_o_com(3)
        endif ! mestring
!
        if (mestring.lt.nstring-1) then
         call mpi_send(ro1,3*norient,mpifloat,mestring+1, 1, &
     & MPI_COMM_STRNG, ierror)
        endif ! me
       endif ! qorient
!cccccccccccccc now call the appropriate interpolation subroutine
       weights(:,1)=forcedWeights
       weights(:,2)=forcedWeights
       weights(:,3)=forcedWeights
!
       if (interp_method.eq.linear_exact) then
        call interp_linear_exact(rf,rf,weights,3*nforced,ds,curv)
       else
        call interp_driver_sci(rf,rf,weights,3*nforced, &
     & interp_method,def,iterations,ds,curv,dst_cutoff)
       endif
!
       if (qorient.and..not.qver2) then
        u=transpose(u)
        if (mestring.gt.0) rf=matmul(rf, u) ! rotate back
! restore original COM
        rf(:,1)=rf(:,1)+r_o_com(1)
        rf(:,2)=rf(:,2)+r_o_com(2)
        rf(:,3)=rf(:,3)+r_o_com(3)
!
       endif ! orient
      endif ! root
!
! broadcast coordinates to slaves
      if (qslave) then
#if (KEY_SINGLE==1)
         call PSND4r(rf,3*nforced) 
#endif
#if (KEY_SINGLE==0)
         call PSND8m(rf,3*nforced) 
#endif
      endif
! update any orientation coordinates that have changes
      if (qorient.and.qdiffrot) call ftsm_update_overlap_coor(ione)
!
      call ftsm_save_com()
!
      end subroutine ftsm_repa
!======================================================================================
      subroutine ftsm_compute_arcl_curv(interp_method_) ! compute arclength and curvature
! much code duplication from ftsm_repa, unfortunately
! see that routine (ftsm_repa) below for more code comments
      use string, only : itoa
      use multicom_aux;
      use number
      use bestfit, only : eig3s, RMSBestFit, rmsd, norm3, veccross3
!
      character(len=len("FTSM_COMPUTE_ARCL_CURV>") ),parameter::whoami="FTSM_COMPUTE_ARCL_CURV>";!macro
      integer, optional :: interp_method_
!
      integer :: interp_m
      integer :: i, ierror
!
      real(chm_real) :: u(3,3)=Id3
      real(chm_real) :: r_o_com(3), w, weights(nforced,3)
      real(chm_real), target :: rf(nforced,3)
      real(chm_real), pointer :: ro(:,:), ro1(:,:)
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
      integer*4 :: stat(MPI_STATUS_SIZE)
!
      interface
        subroutine interp_driver_sci(rin,rout,wgt,n, &
     & interp_method,tol,max_iterations,d_arclength, curvature, &
     & dst_cutoff, dr,r_bc_0, r_bc_1)
      use chm_kinds
      use stream
        integer n
        real(chm_real) rin(n), rout(n), wgt(n)
        integer, intent(in) :: interp_method
        integer max_iterations
        real(chm_real) :: tol, d_arclength(:), curvature(:)
        real(chm_real), optional :: dst_cutoff
        real(chm_real), optional :: dr(n) ,r_bc_0(n), r_bc_1(n)
        end subroutine interp_driver_sci
!
        subroutine interp_linear_exact(rin,rout,wgt,n, &
     & d_arclength, curvature, &
     & drout, &
     & r_bc_0, r_bc_1)
      use chm_kinds
        integer :: n
        real(chm_real) :: rin(n), rout(n), wgt(n)
        real(chm_real) :: d_arclength(:), curvature(:)
        real(chm_real), optional :: drout(n) ! optional computation of tangent
        real(chm_real) , optional :: r_bc_0(n), r_bc_1(n) ! optional fixed bc data
       end subroutine interp_linear_exact
!
      end interface
!
      if (present(interp_method_)) then
       if (all(interp_methods.ne.interp_method_)) then
        write(info(1),'(A)') &
     & 'UNRECOGNIZED INTERPOLATION METHOD CODE ('//itoa(interp_method_)//'). WILL USE LINEAR INTERPOLATION.' ;
        call wrndie(0,whoami,trim(info(1)))
        interp_m = linear
       else
        interp_m = interp_method_
       endif
      elseif (repa_initialized) then
       interp_m = interp_method
      else
       interp_m=linear
      endif
!
      if ( MPI_COMM_STRNG.ne.MPI_COMM_NULL.and.SIZE_STRNG.gt.1 ) then
       if (qorient) then
        ro => r_o(:,:,center)
        ro1 => r_o(:,:,dummy)
! translate structure to centroid
        r_o_com=zero
        do i=1, norient
         w=orientWeights(i)
         r_o_com(1)=r_o_com(1) + w * ro(i,1)
         r_o_com(2)=r_o_com(2) + w * ro(i,2)
         r_o_com(3)=r_o_com(3) + w * ro(i,3)
        enddo
        if (qdiffrot) then
         rf(:,1)=r_f(:,1,center)-r_o_com(1)
         rf(:,2)=r_f(:,2,center)-r_o_com(2)
         rf(:,3)=r_f(:,3,center)-r_o_com(3)
        endif
!ccccccccccc orientation ccccccc
! send/receive orientation structure, orient sequentially
        if (mestring.gt.0) then
         call MPI_RECV(ro1,3*norient,mpifloat,mestring-1, 1, &
     & MPI_COMM_STRNG, stat, ierror)
! orient current structure
         call RMSBestFit(ro1,ro,orientWeights,u) ! u^T rotates ro onto ro1
! transform current structure to overlap with reference
! (if orientation is off, u=I)
         if (.not.qdiffrot) ro1=>rf
         ro1(:,1)=ro(:,1)-r_o_com(1)
         ro1(:,2)=ro(:,2)-r_o_com(2)
         ro1(:,3)=ro(:,3)-r_o_com(3)
!
         ro1=matmul(ro1,u)
         if (qdiffrot) rf=matmul(rf,u) ! otherwise already rotated
!
        else ! first replica enters below
         if (.not.qdiffrot) ro1=>rf ! will remove the COM from actual ro/rf coordinates below (they are modified anyway by interp)
         ro1(:,1)=ro(:,1)-r_o_com(1)
         ro1(:,2)=ro(:,2)-r_o_com(2)
         ro1(:,3)=ro(:,3)-r_o_com(3)
        endif ! mestring
!
        if (mestring.lt.nstring-1) then
         call mpi_send(ro1,3*norient,mpifloat,mestring+1, 1, &
     & MPI_COMM_STRNG, ierror)
        endif ! me
!
       else ! (.not. qorient)
!
        rf(:,1)=r_f(:,1,center)
        rf(:,2)=r_f(:,2,center)
        rf(:,3)=r_f(:,3,center)
!
       endif ! qorient
!cccccccccccccc now call the appropriate interpolation subroutine
       weights(:,1)=forcedWeights
       weights(:,2)=forcedWeights
       weights(:,3)=forcedWeights
!
       if (interp_m.eq.linear_exact) then
        call interp_linear_exact(rf,rf,weights,3*nforced,ds,curv)
       else
        i=0; ! request zero maximum iterations
        call interp_driver_sci(rf,rf,weights,3*nforced, &
     & interp_m,def,i,ds,curv,dst_cutoff)
       endif
      endif ! qroot
!
! NOTE : only roots know ds; broadcast to slaves elsewhere if needed
!
      end subroutine ftsm_compute_arcl_curv
!=================================================================================
#endif
#endif /* automatically protect all code */
      end module ftsm_rep
!
