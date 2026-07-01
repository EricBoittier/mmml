#if (KEY_STRINGM==1) /*  automatically protect all code */
! ********************************************************************************!
! Preprocessed by GNU CPP !
! ********************************************************************************!
! compute work along a path using TI,
! reparametrize string
! open file for iput/output
!
#if (KEY_STRINGM==1)
!
      subroutine compute_work_fd(x,xbc0,xbc1,f,n,work)
      use multicom_aux;
      use mpi
      use chm_kinds
!
      implicit none
      integer :: n
      real(chm_real) :: x(n), f(n)
      real(chm_real) :: work(:)
      real(chm_real), optional :: xbc0(n), xbc1(n)
! locals
      integer :: fbc0=0, fbc1=0 ! fixed bc local flags
      integer ierror, nall, i, nrep, me, nensem
!
! local arrays (f90)
      real(chm_real), pointer :: xall(:,:), fall(:,:)
      real(chm_real), pointer :: dxall(:,:), fall_c(:,:)
      real(chm_real), pointer :: wme(:)
      integer*4 :: s_disp(SIZE_STRNG), r_disp(SIZE_STRNG), &
     & s_count(SIZE_STRNG), r_count(SIZE_STRNG) ! have to declare these as integer*4 -- it`s a problem!
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
! begin
! bail if not a replica root; will syncronize slave nodes elsewhere
      if (MPI_COMM_STRNG.eq.MPI_COMM_NULL) return
!
      if (present(xbc0)) fbc0=1
      if (present(xbc1)) fbc1=1
!
      me=ME_STRNG
      nensem=SIZE_STRNG
      nrep=nensem+fbc0+fbc1
! allocate work array, now that we know nrep
      allocate(wme(nrep))
!
! compute in parallel
! determine displacements and allocate arays
      nall=ceiling(1.0d0*n/nrep)
      allocate(xall(nall,nrep), fall(nall,nrep), &
     & dxall(nall,nrep-1), fall_c(nall,nrep-1))
! initialize
      xall=0.0
      dxall=0.0
      fall=0.0
      fall_c=0.0
      do i=1,nensem
       s_disp(i)=min((i-1)*nall,n-1)
       r_disp(i)=(i-1+fbc0)*nall
       s_count(i)=max(0,min(nall,n-nall*(i-1)))
      enddo
      r_count=s_count(me+1)
!
      call mpi_alltoallv(x,s_count,s_disp,mpifloat, &
     & xall,r_count,r_disp,mpifloat, &
     & MPI_COMM_STRNG, ierror)
      call mpi_alltoallv(f,s_count,s_disp,mpifloat, &
     & fall,r_count,r_disp,mpifloat, &
     & MPI_COMM_STRNG, ierror)
! take care of BC as well
      if (fbc0.eq.1) &
     & xall(1:r_count(1),1)= &
     & xbc0(s_disp(me+1)+1:s_disp(me+1)+r_count(1))
!
      if (fbc1.eq.1) &
     & xall(1:r_count(1),nrep)= &
     & xbc1(s_disp(me+1)+1:s_disp(me+1)+r_count(1))
! now have partial coordinates and forces from all replicas
! compute tangent
      dxall=xall(:,2:nrep)-xall(:,1:nrep-1)
! compute force at midpoint
      fall_c=0.5*(fall(:,2:nrep)+fall(:,1:nrep-1))
! now evaluate individual contributions to the free energy
      wme(1)=0
      do i=1,nrep-1
! wme(i+1)=wme(i)+sum( dxall(:,i)*fall_c(:,i) )
       wme(i+1)=wme(i)+dot_product( dxall(:,i), fall_c(:,i) )
      enddo
!
! reduce work
      call mpi_allreduce(wme, work, nrep, mpifloat, &
     & MPI_SUM, MPI_COMM_STRNG, ierror)
! done!
!
      deallocate(xall, dxall, fall, fall_c, wme)
      end subroutine compute_work_fd
!
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine interp_driver(rin,rout,wgt,n, &
     & interp_method,d_arclength, curvature, &
     & r_bc_0, r_bc_1) ! provide additional arrays for fixed bc
! this routine should handle both ZTS/FTS, depending on what you
! feed into it
!
! however, it is not a general purpose splining routine because the
! number of points cannot change (because it corresponds to the
! number of CPUs in the ensemble)
!
      use multicom_aux;
      use mpi
      use chm_kinds
!
      implicit none
!
      integer n
      real(chm_real) :: rin(n), rout(n), wgt(n)
      integer :: interp_method
      real(chm_real) :: d_arclength(:), curvature(:)
      real(chm_real), optional :: r_bc_0(n), r_bc_1(n) ! optional fixed bc data
!
      integer, parameter :: linear=1, spline=2, bspline=3
!
! locals
      real(chm_real) :: rin_w(n) ! weighted by wgt
      real(chm_real) :: swgt(n) ! square root of weights
      integer :: fbc0=0, fbc1=0 ! fixed bc local flags
      integer :: ierror, nall, i, j, nrep, me, nensem
!
! local arrays (f90)
      real(chm_real), pointer :: rall(:,:), drall(:,:), dr2all(:,:)
!
      integer*4 :: s_disp(SIZE_STRNG), r_disp(SIZE_STRNG), &
     & s_count(SIZE_STRNG), r_count(SIZE_STRNG) ! have to declare these as integer*4 -- it`s a problem!
      real(chm_real), pointer :: rr(:), rrpp(:), dsn(:), ds(:), ds2me(:), &
     & ds2all(:), s(:), t(:), curv_me(:)
      real(chm_real) :: dum, wrs
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
      interface ! to linear interpolation routine
       subroutine linear_interp(xin,yin,nin,xout,yout,nout,dydxout)
       use chm_kinds
       integer :: nin, nout
       real(chm_real) :: xin(nin), yin(nin), xout(nout), yout(nout)
       real(chm_real), optional :: dydxout(nout) ! tangent computation
       real(chm_real) :: dydx(nout)
       end subroutine linear_interp
      end interface
!
! begin
! bail if not a replica root; will syncronize slave nodes elsewhere
      if (MPI_COMM_STRNG.eq.MPI_COMM_NULL) return
!
      if (present(r_bc_0)) fbc0=1
      if (present(r_bc_1)) fbc1=1
!
      me=ME_STRNG
      nensem=SIZE_STRNG
      nrep=nensem+fbc0+fbc1
! allocate work arrays, now that we know nrep
      allocate(rr(nrep), rrpp(nrep))
      allocate(ds(nrep-1), dsn(nrep-1), ds2me(nrep-1), ds2all(nrep-1), &
     & s(nrep),t(nrep),curv_me(nrep-2))
!
! compute arclength in parallel
! determine displacements and allocate arays
      nall=ceiling(1.0d0*n/nrep)
      allocate(rall(nall,nrep), drall(nall,nrep), dr2all(nall,nrep-2))
! initialize
      rall=0.0d0
      drall=0.0d0
      dr2all=0.0d0
!
      do i=1,nensem
       s_disp(i)=min((i-1)*nall,n-1) ! in case n<ncpu; want to keep valid address
       r_disp(i)=(i-1+fbc0)*nall ! fixed bc0 correction
       s_count(i)=max(0,min(nall,n-nall*(i-1)))
      enddo
      r_count=s_count(me+1)
!
! weight rin by wgt
      wrs=1./sum(wgt)
      swgt=sqrt(wgt*wrs) ! no checking for nonnegative numbers
      rin_w=rin*swgt
      call mpi_alltoallv(rin_w,s_count,s_disp,mpifloat, &
     & rall,r_count,r_disp,mpifloat, &
     & MPI_COMM_STRNG, ierror)
! take care of BC as well
      if (fbc0.eq.1) &
     & rall(1:r_count(1),1)= &
     & r_bc_0(s_disp(me+1)+1:s_disp(me+1)+r_count(1))* &
     & swgt(s_disp(me+1)+1:s_disp(me+1)+r_count(1))
!
      if (fbc1.eq.1) &
     & rall(1:r_count(1),nrep)= &
     & r_bc_1(s_disp(me+1)+1:s_disp(me+1)+r_count(1))* &
     & swgt(s_disp(me+1)+1:s_disp(me+1)+r_count(1))
!
! now have partial coordinates from ALL replicas
! compute local part of the tangent
!
      drall=rall(:,2:nrep)-rall(:,1:nrep-1)
      ds2me=sum(drall**2,1)
! reduce ds2
      call mpi_allreduce(ds2me, ds2all, nrep-1, mpifloat, &
     & MPI_SUM, MPI_COMM_STRNG, ierror)
! now can compute arclength
      s(1)=0d0
      do i=1,nrep-1
       ds(i)=sqrt(ds2all(i))
       s(i+1)=s(i)+ds(i)
      enddo
! save arclength and pass back to calling routine
      d_arclength=ds
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! normalize dr
      do i=1, nall
       drall(i,1:nrep-1)=drall(i,1:nrep-1)/ds(1:nrep-1)
      enddo
! compute curvature (second derivative, actually)
      dsn(1:nrep-2)=0.5d0*(ds(2:nrep-1)+ds(1:nrep-2)) ! dsc approximated at nodes, not centers
      do i=1, nall
       dr2all(i,1:nrep-2)=(drall(i,2:nrep-1)-drall(i,1:nrep-2)) &
     & /dsn(1:nrep-2)
      enddo
! compute magnitude
      curv_me=sum(dr2all**2,1)
! reduce from all cpus
      call mpi_allreduce(curv_me, curvature, nrep-2, mpifloat, &
     & MPI_SUM, MPI_COMM_STRNG, ierror)
! take square root
      curvature=sqrt(curvature)
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! normalize arclength array
      do i=1,nrep
       s(i)=s(i)/s(nrep)
      enddo
!ccccccccccccccccccccccccc
! create uniform array
      do i=1,nrep
       t(i)=1d0*(i-1)/(nrep-1)
      enddo
!cccccccccccccc now interpolate variables cccccc
      if (interp_method.eq.spline) then
       do i=1,nall
        rr=rall(i,:)
        call spline_cubic_set(nrep,s,rr,0,0,0,0,rrpp)
        do j=1,nrep
         call spline_cubic_val(nrep,s,rr,rrpp,t(j),rall(i,j),dum,dum)
        enddo
       enddo
      elseif (interp_method.eq.bspline) then
       do i=1,nall ! over all known atoms
        rr=rall(i,:)
        do j=1,nrep
         call spline_b_val(nrep,s,rr,t(j),rall(i,j)) ! overwrite data
        enddo
       enddo
      elseif (interp_method.eq.linear) then
       do i=1,nall
        rr=rall(i,:)
        call linear_interp(s,rr,nrep,t,rr,nrep)
        rall(i,:)=rr
       enddo
      else
       call wrndie(0,' INTERP_DRIVER>',trim('NO VALID INTERPOLATION METHODS SELECTED'))
      endif
!ccccc now scatter new coordinates into xout,yout,zout
      call mpi_alltoallv(rall,r_count,r_disp,mpifloat, &
     & rout,s_count,s_disp,mpifloat, &
     & MPI_COMM_STRNG, ierror)
! note: rout obtained in this way is scaled by wgt
! we need to unscale it:
      rout=rout/swgt
! done!
!cccccccccccccccccccccccccccccccccccccccccccccccccc
      deallocate(rall, drall, dr2all)
      deallocate(rr,rrpp,ds,dsn,ds2me,ds2all,s,t,curv_me)
!
      end subroutine interp_driver
!
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!-----------------SELF - CONSISTENT REPARAMETERIZATION ------
      subroutine interp_driver_sci(rin,rout,wgt,n, &
     & interp_method,tol,max_iterations,d_arclength, curvature, &
     & dst_cutoff, &
     & drout, & ! optional computation of tangent
     & r_bc_0, r_bc_1) ! provide additional arrays for fixed bc
!
      use stream
      use multicom_aux;
      use mpi
      use chm_kinds
      use number
      use consta
!
      implicit none
!
      integer :: n
      real(chm_real) :: rin(n), rout(n), wgt(n)
      integer, intent(in) :: interp_method
      integer :: max_iterations
      real(chm_real) :: tol, d_arclength(:), curvature(:)
      real(chm_real), optional :: dst_cutoff ! wavenumber cutoff for the sine transform
      real(chm_real), optional :: drout(n) ! optional computation of tangent
      real(chm_real) , optional :: r_bc_0(n), r_bc_1(n) ! optional fixed bc data
!
      integer, parameter :: linear=1, spline=2, bspline=3, dst=4
!
! locals
      real(chm_real) :: rin_w(n)
      real(chm_real) :: swgt(n)
      integer :: fbc0=0, fbc1=0 ! fixed bc local flags
      integer :: ierror, nall, i, j, nrep, me, nensem
!
! fine-grid interpolation
      integer :: nfine
      real(chm_real), pointer, dimension(:) :: sfine, tfine
!
! local arrays (f90)
      real(chm_real), pointer :: rall(:,:), drall(:,:), dr2all(:,:)
      real(chm_real), pointer :: rfine(:,:), drfine(:,:)
      real(chm_real), pointer :: rh(:) ! for sine transform
      real(chm_real), pointer :: sinvec(:,:) ! for sine transform
      real(chm_real), pointer :: sinvec_fine(:,:) ! for sine tranform
!
      integer*4 :: s_disp(SIZE_STRNG), r_disp(SIZE_STRNG), &
     & s_count(SIZE_STRNG), r_count(SIZE_STRNG) ! have to declare these as integer*4 -- it`s a problem!
!
      real(chm_real), pointer :: rr(:), rrpp(:), rrtan(:), &
     & ds(:), s(:), tuni(:), tnew(:), & ! for sine transform
     & curv_me(:) ! local contribution to curvature
!
      real(chm_real), pointer, dimension(:) :: rrfine, ds2me_fine, &
     & dsfine, ds2all_fine
!
      integer :: iteration
      integer :: npass ! for sine transform
      real(chm_real) :: def, dum
! real(chm_real), parameter :: pi=3.14159265358979323846d0 ! comment out if known from a module
      real(chm_real) :: r0, r1, wrs
 character(len=132)::info(21)=(/'','','','','','','','','','','','','','','','','','','','',''/);! output buffer
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
      character(len=len("INTERP_DRIVER_SCI>") ),parameter::whoami="INTERP_DRIVER_SCI>";!macro
!
      interface ! to linear interpolation routine
       subroutine linear_interp(xin,yin,nin,xout,yout,nout,dydxout)
       use chm_kinds
       implicit none
       integer :: nin, nout
       real(chm_real) :: xin(nin), yin(nin), xout(nout), yout(nout)
       real(chm_real), optional :: dydxout(nout) ! tangent computation
       real(chm_real) :: dydx(nout)
       end subroutine linear_interp
      end interface
!
! begin
! bail if not a replica root; will synchronize slave nodes elsewhere
      if (MPI_COMM_STRNG.eq.MPI_COMM_NULL) return
!
      if (present(r_bc_0)) fbc0=1
      if (present(r_bc_1)) fbc1=1
!
      me=ME_STRNG
      nensem=SIZE_STRNG
      nrep=nensem+fbc0+fbc1
!
! allocate work arrays now that we know nrep
      allocate(rr(nrep),rrtan(nrep),rrpp(nrep),ds(nrep-1), &
     & s(nrep), tuni(nrep), tnew(nrep), &
     & curv_me(nrep-1)) ! local contribution to curvature
! allocate fine arrays
      nfine=5*nrep
      allocate(sfine(nfine), tfine(nfine), &
     & rrfine(nfine), ds2me_fine(nfine-1), &
     & dsfine(nfine-1), ds2all_fine(nfine-1))
! initialization for sine transform
      if (interp_method.eq.dst) then
       if (.not. present(dst_cutoff)) dst_cutoff=0.5 ! by default, only the lower half of wn kept
       if (dst_cutoff.gt.1) &
       call wrndie(0,whoami,trim('SINE TRANSFORM CUTOFF > 1.0; WILL SET TO 0.5'));
       npass=min(nrep,ceiling(nrep*dst_cutoff))
       allocate(sinvec(nrep,nrep))
       allocate(rh(nrep)) ! allocate wavenumber array
       allocate(sinvec_fine(nfine,nrep))
      endif
!
! compute arclength in parallel
! determine displacements and allocate arrays
      nall=ceiling(1.0d0*n/nrep)
      allocate(rall(nall,nrep), drall(nall,nrep), dr2all(nall,nrep-1))
      allocate(rfine(nall, nfine), drfine(nall,nfine-1)) ! sci
! initialize
      rall=0d0
      rfine=0d0
      do i=1,nensem
       s_disp(i)=min((i-1)*nall,n-1) ! in case n<nrep
       r_disp(i)=(i-1+fbc0)*nall
       s_count(i)=max(0,min(nall,n-nall*(i-1))) ! how many CV I will send to CPU i
      enddo
      r_count=s_count(me+1) ! how many CV I am receiving
!
! scale rin by wgt
      wrs=sum(wgt) ; wrs=one/wrs
      swgt=sqrt(wgt*wrs)
      rin_w=rin*swgt
!
      call mpi_alltoallv(rin_w,s_count,s_disp,mpifloat, &
     & rall,r_count,r_disp,mpifloat, &
     & MPI_COMM_STRNG, ierror)
! take care of BC as well
      if (fbc0.eq.1) &
     & rall(1:r_count(1),1)= &
     & r_bc_0(s_disp(me+1)+1:s_disp(me+1)+r_count(1))* &
     & swgt(s_disp(me+1)+1:s_disp(me+1)+r_count(1))
!
      if (fbc1.eq.1) &
     & rall(1:r_count(1),nrep)= &
     & r_bc_1(s_disp(me+1)+1:s_disp(me+1)+r_count(1))* &
     & swgt(s_disp(me+1)+1:s_disp(me+1)+r_count(1))
!
! now have partial coordinates from all replicas
!
! interpolate onto the fine grid
!ccccccccccccccccccccccccccccccccccccccccccccccc
! create uniform array
      tuni(1)=zero
      if (nrep.gt.1) then
       dum=one/(nrep-1)
       do i=2,nrep-1
        tuni(i)=dum*(i-1)
       enddo
       tuni(nrep)=one
      endif
!cccccccccccccc now interpolate variables cccccc
! create uniform fine-grid array
      tfine(1)=zero
      if (nfine.gt.1) then
       dum=one/(nfine-1)
       do i=2,nfine-1
        tfine(i)=dum*(i-1)
       enddo
       tfine(nfine)=one
      endif
!cccccccccccccc now interpolate variables cccccc
      if (interp_method.eq.spline) then
       do i=1,nall
        rr=rall(i,:)
        call spline_cubic_set(nrep,tuni,rr,0,0,0,0,rrpp)
        do j=1,nfine
         call spline_cubic_val(nrep,tuni,rr,rrpp,tfine(j),rfine(i,j), &
     & dum,dum) ! throw away derivatives
        enddo
       enddo
!ccccccccccccccccccccccccccccccccccccccccccccccc
      elseif (interp_method.eq.bspline) then
       do i=1,nall ! over all known atoms
        rr=rall(i,:)
        do j=1,nfine
         call spline_b_val(nrep,tuni,rr,tfine(j),rfine(i,j)) ! overwrite
        enddo
       enddo
!ccccccccccccccccccccccccccccccccccccccccccccccc
      elseif (interp_method.eq.linear) then
       do i=1,nall
        rr=rall(i,:)
        call linear_interp(tuni,rr,nrep,tfine,rrfine,nfine)
        rfine(i,:)=rrfine
       enddo
!ccccccccccccccccccccccccccccccccccccccccccccccc
      elseif (interp_method.eq.dst) then
! pre-compute sine transform arrays
       do j=1,nrep
        sinvec(:,j)=sin(pi*(j-1)*tuni)
        sinvec_fine(:,j)=sin(pi*(j-1)*tfine)
       enddo
       do i=1,nall ! loop over all variables
        rr=rall(i,:) ! make a copy
        r0=rr(1); r1=rr(nrep)
        rr=rr-(r1-r0)*tuni-r0 ! remove linear part of array
        rh=2.0d0*matmul(transpose(sinvec),rr)/(nrep-1) ! forward transform
        rrfine=matmul(sinvec_fine(:,1:npass), &
     & rh(1:npass)) ! inverse transform
        rrfine=rrfine+r0+(r1-r0)*tfine ! restore linear component
        rfine(i,:)=rrfine ! copy to fine array
       enddo ! loop over variables
      else
        call wrndie(0,whoami,trim('NO VALID INTERPOLATION METHODS SELECTED'))
      endif
! now we have a finely spaced path
!
! compute local part of the arclength
      drfine=rfine(:,2:nfine)-rfine(:,1:nfine-1)
!
      ds2me_fine=sum(drfine**2,1)
! reduce ds2
      call mpi_allreduce(ds2me_fine, ds2all_fine, nfine-1, mpifloat, &
     & MPI_SUM, MPI_COMM_STRNG, ierror)
!
! compute (fine) arclength s
      sfine(1)=zero
      do i=1,nfine-1
       dsfine(i)=sqrt(ds2all_fine(i))
       sfine(i+1)=sfine(i)+dsfine(i)
      enddo
! and get s by linear interp:
      call linear_interp(tfine,sfine,nfine,tuni,s,nrep)
! normalize sfine
      sfine=sfine/sfine(nfine)
! compute ds
      ds=s(2:nrep)-s(1:nrep-1)
! deficit
      def=maxval(ds)/minval(ds)
! compute _approximate_ curvature on the coarse grid (2nd derivative)
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      dr2all=rall(:,2:nrep)-rall(:,1:nrep-1) ! first difference
      dr2all(:,1:nrep-2)=dr2all(:,2:nrep-1)-dr2all(:,1:nrep-2) ! second difference
      curv_me=sum(dr2all**2,1)
! reduce from all cpus
      call mpi_allreduce(curv_me, curvature, nrep-2, mpifloat, &
     & MPI_SUM, MPI_COMM_STRNG, ierror)
! take square root and divide by ds^2
      curvature=sqrt(curvature)/ds(1:nrep-2)/ds(2:nrep-1)
! to get the derivative with respect to a \in {0,1} parametrization, multiply by curve length
! (bear in mind that the norms here are smaller by sqrt(3) than atomic RMSDs)
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! begin convergence loop
      iteration=0
      do while (iteration.lt.max_iterations.and.def.gt.tol)
! get new r values
       iteration=iteration+1
       do i=1,nall
        rrfine=rfine(i,:)
        call linear_interp(sfine,rrfine,nfine,tuni,rr,nrep,rrtan)
        rall(i,:)=rr
        drall(i,:)=rrtan ! this may not be accurate for linear interpolation
       enddo
! compute drall differently for linear interpolation; dr is normalized at the end of routine
! only inner points get modified
! assuming that ds is uniform (which should be true at convergence)
! note: ds is normalized to unit length at the end
       if (interp_method.eq.linear) &
     & drall(:,2:nrep-1)=rall(:,3:nrep)-rall(:,1:nrep-2)
! now repeat: compute fine grid data, etc:
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       if (interp_method.eq.spline) then
        do i=1,nall
         rr=rall(i,:)
         call spline_cubic_set(nrep,tuni,rr,0,0,0,0,rrpp)
         do j=1,nfine
          call spline_cubic_val(nrep,tuni,rr,rrpp,tfine(j),rfine(i,j), &
     & dum,dum) ! throw away derivatives
         enddo
        enddo
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       elseif (interp_method.eq.bspline) then
        do i=1,nall ! over all known atoms
         rr=rall(i,:)
         do j=1,nfine
          call spline_b_val(nrep,tuni,rr,tfine(j),rfine(i,j))
         enddo
        enddo
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       elseif (interp_method.eq.linear) then
        do i=1,nall
         rr=rall(i,:)
         call linear_interp(tuni,rr,nrep,tfine,rrfine,nfine)
         rfine(i,:)=rrfine
        enddo
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       elseif (interp_method.eq.dst) then
! compute new parameter array
! call linear_interp(sfine,tfine,nfine,tuni,tnew,nrep) ! tnew is the new parameter array
! pre-compute sine transform array based on tnew (note that tnew is nonuniform)
       tnew=tuni ! works better, is there a bug?
       do j=1,nrep
        sinvec(:,j)=sin(pi*(j-1)*tnew)
       enddo
       do i=1,nall ! loop over all variables
        rr=rall(i,:) ! make a copy
        r0=rr(1); r1=rr(nrep)
        rr=rr-(r1-r0)*tnew-r0 ! remove linear part of array
        rh=2d0*matmul(transpose(sinvec),rr)/(nrep-1) ! forward transform
        rrfine=matmul(sinvec_fine(:,1:npass), &
     & rh(1:npass)) ! inverse transform
        rrfine=rrfine+r0+(r1-r0)*tfine ! restore linear component
        rfine(i,:)=rrfine ! copy to fine array
       enddo ! loop over variables
       else
        call wrndie(0,whoami,trim('NO VALID INTERPOLATION METHODS SELECTED'))
       endif
!ccccccccccccccccccccccccccccccccccccccccccccccccccccc
       drfine=rfine(:,2:nfine)-rfine(:,1:nfine-1)
       ds2me_fine=sum(drfine**2,1)
       call mpi_allreduce(ds2me_fine, ds2all_fine, nfine-1, mpifloat, &
     & MPI_SUM, MPI_COMM_STRNG, ierror)
!
       sfine(1)=zero
       do i=1,nfine-1
        dsfine(i)=sqrt(ds2all_fine(i))
        sfine(i+1)=sfine(i)+dsfine(i)
       enddo
       call linear_interp(tfine,sfine,nfine,tuni,s,nrep)
       sfine=sfine/sfine(nfine)
       ds=s(2:nrep)-s(1:nrep-1)
       def=maxval(ds)/minval(ds)
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! compute curvature on the coarse grid (exact only for constant ds)
       dr2all=rall(:,2:nrep)-rall(:,1:nrep-1)
       dr2all(:,1:nrep-2)=dr2all(:,2:nrep-1)-dr2all(:,1:nrep-2)
       curv_me=sum(dr2all**2,1)
! reduce from all cpus
       call mpi_allreduce(curv_me, curvature, nrep-2, mpifloat, &
     & MPI_SUM, MPI_COMM_STRNG, ierror)
! take square root and divide by ds^2
       curvature=sqrt(curvature)/ds(1:nrep-2)/ds(2:nrep-1)
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      enddo
! print message
      write(info,666) whoami, iteration, def; if(prnlev.ge. 4) write(OUTU,'(A)') pack(info,info.ne.'');info='';
 666 format(A,' ',I4,' ITERATIONS. DEF=',F10.5)
!
! save arclength to pass back to calling routine
      d_arclength=ds
!ccccc now scatter new coordinates into rout
      call mpi_alltoallv(rall,r_count,r_disp,mpifloat, &
     & rout,s_count,s_disp,mpifloat, &
     & MPI_COMM_STRNG, ierror)
! note: rall is scaled by wgt, so we need to unscale rout:
      rout=rout/swgt
!
! drall is computed inside the iteration loop; if there were no
! iterations, leave drout untouched; (also do nothing if drout not passed in)
      if (present(drout).and.(iteration.gt.0)) then
       call mpi_alltoallv(drall,r_count,r_disp,mpifloat, &
     & drout,s_count,s_disp,mpifloat, &
     & MPI_COMM_STRNG, ierror)
! note: drout is scaled by wgt; to unscale, uncomment below
! drout=drout/swgt
! normalize drout to unity
       dum=dot_product(drout,drout)
       drout=drout/sqrt(dum)
      endif
! done!
!cccccccccccccccccccccccccccccccccccccccccccccccccc
      deallocate(rall, drall, dr2all, rfine, drfine)
      deallocate(rr,rrtan,rrpp,s,ds,tuni,tnew,curv_me)
      deallocate(tfine,sfine,rrfine,ds2me_fine,dsfine,ds2all_fine)
      if (interp_method.eq.dst) &
     & deallocate(sinvec, sinvec_fine, rh)
      end subroutine interp_driver_sci
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! SAME ROUTINE AS ABOVE EXCEPT
! PERFORMED ON A SINGLE NODE TO REDUCE COMMUNICATION
!
      subroutine interp_driver_sci_root(rin,rout,wgt,n, &
     & interp_method,tol,max_iterations,d_arclength, curvature, &
     & dst_cutoff, &
     & drout, & ! optional computation of tangent
     & r_bc_0, r_bc_1) ! provide additional arrays for fixed bc
!
      use number
      use consta
      use stream
      use multicom_aux;
      use mpi
      use chm_kinds
!
      implicit none
!
      integer :: n
      real(chm_real) :: rin(n), rout(n), wgt(n)
      integer, intent(in) :: interp_method
      integer :: max_iterations
      real(chm_real) :: tol, d_arclength(:), curvature(:)
      real(chm_real), optional :: dst_cutoff ! wavenumber cutoff for the sine transform
      real(chm_real), optional :: drout(n) ! optional computation of tangent
      real(chm_real) , optional :: r_bc_0(n), r_bc_1(n) ! optional fixed bc data
!
      integer, parameter :: linear=1, spline=2, bspline=3, dst=4
!
! locals
      real(chm_real) :: rin_w(n)
      real(chm_real) :: swgt(n)
      integer :: fbc0=0, fbc1=0 ! fixed bc local flags
      integer :: ierror, nall, i, j, nrep, me, nensem
      logical :: qroot
!
! fine-grid interpolation
      integer :: nfine
      real(chm_real), pointer, dimension(:) :: sfine, tfine
!
! local arrays (f90)
      real(chm_real), pointer :: rall(:,:), drall(:,:), dr2all(:,:)
      real(chm_real), pointer :: rfine(:,:), drfine(:,:)
      real(chm_real), pointer :: rh(:) ! for sine transform
      real(chm_real), pointer :: sinvec(:,:) ! for sine transform
      real(chm_real), pointer :: sinvec_fine(:,:) ! for sine tranform
!
! integer*4 :: s_disp(SIZE_STRNG), r_disp(SIZE_STRNG), &
! & s_count(SIZE_STRNG), r_count(SIZE_STRNG) ! have to declare these as integer*4 -- it`s a problem!
!
      real(chm_real), pointer :: rr(:), rrpp(:), rrtan(:), &
     & ds(:), s(:), tuni(:), tnew(:) ! for sine transform
!
      real(chm_real), pointer, dimension(:) :: rrfine, &
     & dsfine, ds2_fine
!
      integer :: iteration
      integer :: npass ! for sine transform
      real(chm_real) :: def, dum
! real(chm_real), parameter :: pi=3.14159265358979323846
      real(chm_real) :: r0, r1, wrs
 character(len=132)::info(21)=(/'','','','','','','','','','','','','','','','','','','','',''/);! output buffer
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
      character(len=len("INTERP_DRIVER_SCI_ROOT>") ),parameter::whoami="INTERP_DRIVER_SCI_ROOT>";!macro
!
      interface ! to linear interpolation routine
       subroutine linear_interp(xin,yin,nin,xout,yout,nout,dydxout)
        use chm_kinds
       integer :: nin, nout
       real(chm_real) :: xin(nin), yin(nin), xout(nout), yout(nout)
       real(chm_real), optional :: dydxout(nout) ! tangent computation
       real(chm_real) :: dydx(nout)
       end subroutine linear_interp
      end interface
!
! begin
! bail if not a replica root; will synchronize slave nodes elsewhere
      if (MPI_COMM_STRNG.eq.MPI_COMM_NULL) return
      qroot=ME_STRNG.eq.0 ! global root
!
      if (present(r_bc_0)) fbc0=1
      if (present(r_bc_1)) fbc1=1
!
      me=ME_STRNG
      nensem=SIZE_STRNG
      nrep=nensem+fbc0+fbc1
!
! allocate work arrays now that we know nrep
      allocate(rr(nrep),rrtan(nrep),rrpp(nrep),ds(nrep-1), &
     & s(nrep), tuni(nrep), tnew(nrep))
! allocate fine arrays
      nfine=5*nrep
      allocate(sfine(nfine), tfine(nfine), &
     & rrfine(nfine), ds2_fine(nfine-1), &
     & dsfine(nfine-1))
! initialization for sine transform
      if (interp_method.eq.dst) then
       if (.not. present(dst_cutoff)) dst_cutoff=0.5 ! by default, only the lower half of wn kept
       if (dst_cutoff.gt.1) &
     & call wrndie(0,whoami,trim('SINE TRANSFORM CUTOFF > 1.0; WILL SET TO 0.5'))
       npass=min(nrep,ceiling(nrep*dst_cutoff))
       allocate(sinvec(nrep,nrep))
       allocate(rh(nrep)) ! allocate wavenumber array
       allocate(sinvec_fine(nfine,nrep))
      endif
!
! compute arclength
! determine displacements and allocate arrays
      nall=n ! all coordinates will be sent to root
      allocate(rall(nall,nrep), drall(nall,nrep), dr2all(nall,nrep-1))
      allocate(rfine(nall, nfine), drfine(nall,nfine-1)) ! sci
! initialize
      rall=zero
      rfine=zero
!
! scale rin by wgt
      wrs=one/sum(wgt)
      swgt=sqrt(wgt*wrs)
      rin_w=rin*swgt
!
! gather all data on root
      call mpi_gather(rin_w,n,mpifloat,rall(1,fbc0+1),n,mpifloat, &
     & 0,MPI_COMM_STRNG, ierror)
!
! the rest of reparametrization done by root
!
      if (qroot) then
! take care of BC
       if (fbc0.eq.1) rall(:,1) =r_bc_0*swgt
       if (fbc1.eq.1) rall(:,nrep)=r_bc_1*swgt
!
! interpolate onto the fine grid
!ccccccccccccccccccccccccccccccccccccccccccccccc
! create uniform array
       tuni(1)=zero
       if (nrep.gt.1) then
        do i=2,nrep-1
         tuni(i)=one*(i-1)/(nrep-1)
        enddo
        tuni(nrep)=one
       endif
!cccccccccccccc now interpolate variables cccccc
! create uniform fine-grid array
       tfine(1)=zero
       if (nfine.gt.1) then
        do i=2,nfine-1
         tfine(i)=one*(i-1)/(nfine-1)
        enddo
        tfine(nfine)=one
       endif
!cccccccccccccc now interpolate variables cccccc
       if (interp_method.eq.spline) then
        do i=1,nall
         rr=rall(i,:)
         call spline_cubic_set(nrep,tuni,rr,0,0,0,0,rrpp)
         do j=1,nfine
          call spline_cubic_val(nrep,tuni,rr,rrpp,tfine(j),rfine(i,j), &
     & dum,dum) ! throw away derivatives
         enddo
        enddo
!ccccccccccccccccccccccccccccccccccccccccccccccc
       elseif (interp_method.eq.bspline) then
        do i=1,nall ! over all known atoms
         rr=rall(i,:)
         do j=1,nfine
          call spline_b_val(nrep,tuni,rr,tfine(j),rfine(i,j)) ! overwrite
         enddo
        enddo
!ccccccccccccccccccccccccccccccccccccccccccccccc
       elseif (interp_method.eq.linear) then
        do i=1,nall
         rr=rall(i,:)
         call linear_interp(tuni,rr,nrep,tfine,rrfine,nfine)
         rfine(i,:)=rrfine
        enddo
!ccccccccccccccccccccccccccccccccccccccccccccccc
       elseif (interp_method.eq.dst) then
! pre-compute sine transform arrays
        do j=1,nrep
         sinvec(:,j)=sin(pi*(j-1)*tuni)
         sinvec_fine(:,j)=sin(pi*(j-1)*tfine)
        enddo
        do i=1,nall ! loop over all variables
         rr=rall(i,:) ! make a copy
         r0=rr(1); r1=rr(nrep)
         rr=rr-(r1-r0)*tuni-r0 ! remove linear part of array
         rh=two*matmul(transpose(sinvec),rr)/(nrep-1) ! forward transform
         rrfine=matmul(sinvec_fine(:,1:npass), &
     & rh(1:npass)) ! inverse transform
         rrfine=rrfine+r0+(r1-r0)*tfine ! restore linear component
         rfine(i,:)=rrfine ! copy to fine array
        enddo ! loop over variables
       else
        call wrndie(0,whoami,trim('NO VALID INTERPOLATION METHODS SELECTED'))
!
       endif
! now we have a finely spaced path
!
! compute arclength
       drfine=rfine(:,2:nfine)-rfine(:,1:nfine-1)
!
       ds2_fine=sum(drfine**2,1)
!
! compute (fine) arclength s
       sfine(1)=zero
       do i=1,nfine-1
        dsfine(i)=sqrt(ds2_fine(i))
        sfine(i+1)=sfine(i)+dsfine(i)
       enddo
! and get s by linear interp:
       call linear_interp(tfine,sfine,nfine,tuni,s,nrep)
! normalize sfine
       sfine=sfine/sfine(nfine)
! compute ds
       ds=s(2:nrep)-s(1:nrep-1)
! deficit
       def=maxval(ds)/minval(ds)
! compute _approximate_ curvature on the coarse grid (2nd derivative)
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       dr2all=rall(:,2:nrep)-rall(:,1:nrep-1)
       dr2all(:,1:nrep-2)=dr2all(:,2:nrep-1)-dr2all(:,1:nrep-2)
       curvature=sqrt(sum(dr2all**2,1))/ds(1:nrep-2)/ds(2:nrep-1)
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! begin convergence loop
       iteration=0
       do while (iteration.lt.max_iterations.and.def.gt.tol)
! get new r values
        iteration=iteration+1
        do i=1,nall
         rrfine=rfine(i,:)
         call linear_interp(sfine,rrfine,nfine,tuni,rr,nrep,rrtan)
         rall(i,:)=rr
         drall(i,:)=rrtan ! this may not be accurate for linear interpolation
        enddo
! compute drall differently for linear interpolation
! only inner points get modified
! assuming that ds is uniform (which should be true at convergence)
! note: ds is normalized to unit length at the end
        if (interp_method.eq.linear) &
     & drall(:,2:nrep-1)=rall(:,3:nrep)-rall(:,1:nrep-2)
! now repeat: compute fine grid data, etc:
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        if (interp_method.eq.spline) then
         do i=1,nall
          rr=rall(i,:)
          call spline_cubic_set(nrep,tuni,rr,0,0,0,0,rrpp)
          do j=1,nfine
           call spline_cubic_val(nrep,tuni,rr,rrpp,tfine(j),rfine(i,j), &
     & dum,dum) ! throw away derivatives
          enddo
         enddo
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        elseif (interp_method.eq.bspline) then
         do i=1,nall ! over all known atoms
          rr=rall(i,:)
          do j=1,nfine
           call spline_b_val(nrep,tuni,rr,tfine(j),rfine(i,j))
          enddo
         enddo
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        elseif (interp_method.eq.linear) then
         do i=1,nall
          rr=rall(i,:)
          call linear_interp(tuni,rr,nrep,tfine,rrfine,nfine)
          rfine(i,:)=rrfine
         enddo
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        elseif (interp_method.eq.dst) then
! compute new parameter array
! call linear_interp(sfine,tfine,nfine,tuni,tnew,nrep) ! tnew is the new parameter array
! pre-compute sine transform array based on tnew (note that tnew is nonuniform)
         tnew=tuni ! works better, is there a bug?
         do j=1,nrep
          sinvec(:,j)=sin(pi*(j-1)*tnew)
         enddo
         do i=1,nall ! loop over all variables
          rr=rall(i,:) ! make a copy
          r0=rr(1); r1=rr(nrep)
          rr=rr-(r1-r0)*tnew-r0 ! remove linear part of array
          rh=two*matmul(transpose(sinvec),rr)/(nrep-1) ! forward transform
          rrfine=matmul(sinvec_fine(:,1:npass), &
     & rh(1:npass)) ! inverse transform
          rrfine=rrfine+r0+(r1-r0)*tfine ! restore linear component
          rfine(i,:)=rrfine ! copy to fine array
         enddo ! loop over variables
        else
         call wrndie(0,whoami,trim('NO VALID INTERPOLATION METHODS SELECTED'))
        endif
!cccccccccccccccccccccccccccccccccccccccccccccccccccccc
        drfine=rfine(:,2:nfine)-rfine(:,1:nfine-1)
        ds2_fine=sum(drfine**2,1)
!
        sfine(1)=zero
        do i=1,nfine-1
         dsfine(i)=sqrt(ds2_fine(i))
         sfine(i+1)=sfine(i)+dsfine(i)
        enddo
        call linear_interp(tfine,sfine,nfine,tuni,s,nrep)
        sfine=sfine/sfine(nfine)
        ds=s(2:nrep)-s(1:nrep-1)
        def=maxval(ds)/minval(ds)
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! compute curvature on the coarse grid (exact only for constant ds)
        dr2all=rall(:,2:nrep)-rall(:,1:nrep-1)
        dr2all(:,1:nrep-2)=dr2all(:,2:nrep-1)-dr2all(:,1:nrep-2)
        curvature=sqrt(sum(dr2all**2,1))/ds(1:nrep-2)/ds(2:nrep-1)
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       enddo ! convergence loop
! print message
       write(info,666) whoami, iteration, def; if(prnlev.ge. 4) write(OUTU,'(A)') pack(info,info.ne.'');info='';
 666 format(A,' ',I4,' ITERATIONS. DEF=',F10.5)
!
! save arclength to pass back to calling routine
       d_arclength=ds
!
      endif ! qroot
!ccccc now scatter new coordinates into rout
      call mpi_scatter(rall(1,fbc0+1), n, mpifloat, rout, n ,mpifloat, &
     & 0, MPI_COMM_STRNG, ierror)
! note: may want to broadcast curvature and arclength
!
! note: rall is scaled by wgt, so we need to unscale rout:
      rout=rout/swgt
!
! drall is computed inside the iteration loop; if there were no
! iterations, leave drout untouched; (also do nothing if drout not passed in)
! known bug: some compilers report drout as present when it is not
! write(0,*) present(drout)
!
      if (present(drout)) then
       call mpi_bcast(iteration,1,mpiint,0,MPI_COMM_STRNG,ierror)
       if (iteration.gt.0) then
! write(0,*) 'sending drout'
        call mpi_scatter(drall(1,fbc0), n, mpifloat, drout, n ,mpifloat, &
     & 0, MPI_COMM_STRNG, ierror)
! note: drout is scaled by wgt; to unscale, uncomment below
! drout=drout/swgt
! normalize drout to unity
        dum=dot_product(drout,drout)
        drout=drout/sqrt(dum)
       endif
      endif ! drout present
! done!
!cccccccccccccccccccccccccccccccccccccccccccccccccc
      deallocate(rall, drall, dr2all, rfine, drfine)
      deallocate(rr,rrtan,rrpp,s,ds,tuni,tnew)
      deallocate(tfine,sfine,rrfine,ds2_fine,dsfine)
      if (interp_method.eq.dst) &
     & deallocate(sinvec, sinvec_fine, rh)
!
! write(me_global+100,*) present(drout), rout
! close(me_global+100)
! call MPI_BARRIER(MPI_COMM_STRNG,ierror)
! stop
      end subroutine interp_driver_sci_root
!
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!
      subroutine interp_linear_exact(rin,rout,wgt,n, &
     & d_arclength, curvature, &
     & drout, & ! optional computation of tangent
     & r_bc_0, r_bc_1) ! provide additional arrays for fixed bc
!
      use multicom_aux;
      use mpi
      use multidiag ! tridiagonal inversion
      use chm_kinds
!
      implicit none
!
      integer, parameter :: ione=1
      integer :: n
      real(chm_real) :: rin(n), rout(n), wgt(n)
      real(chm_real) :: d_arclength(:), curvature(:)
      real(chm_real), optional :: drout(n) ! optional computation of tangent
      real(chm_real) , optional :: r_bc_0(n), r_bc_1(n) ! optional fixed bc data
!
! locals
      real(chm_real) :: rin_w(n)
      real(chm_real) :: swgt(n)
      integer :: fbc0=0, fbc1=0 ! fixed bc local flags
      integer :: ierror, nall, i, nrep, me, nensem, im, iteration
      logical :: qroot
!
! local arrays (f90)
      real(chm_real), pointer :: rall(:,:), drall(:,:), &
     & rall_new(:,:), drall_new(:,:), &
     & ds2_new(:), cons(:), rhs(:), alpha(:), &
     & dalpha(:)
      real(chm_real), pointer :: a(:,:), ao(:), am(:), ap(:)
!
      real(chm_real) :: wrs, err, dum
      real(chm_real), parameter :: errtol = 1d-8
      integer , parameter :: max_iterations = 200
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
      character(len=len("INTERP_LINEAR_EXACT>") ),parameter::whoami="INTERP_LINEAR_EXACT>";!macro
!
! begin
! bail if not a replica root; will synchronize slave nodes elsewhere
      if (MPI_COMM_STRNG.eq.MPI_COMM_NULL) return
      qroot=ME_STRNG.eq.0 ! global root
!
      if (present(r_bc_0)) fbc0=1
      if (present(r_bc_1)) fbc1=1
!
      me=ME_STRNG
      nensem=SIZE_STRNG
      nrep=nensem+fbc0+fbc1
!
      nall=n ! all coordinates will be sent to root
      allocate(rall(nall,nrep), drall(nall,nrep-1), &
     & rall_new(nall,nrep), &
     & drall_new(nall,nrep-1), &
     & ds2_new(nrep-1), &
     & a(-1:1,nrep-2), &
     & cons(nrep-2), &
     & rhs(nrep-2), &
     & alpha(nrep), &
     & dalpha(nrep) )
!
      am=>a(-1,:); ao=>a(0,:); ap=>a(1,:)
!
! scale rin by wgt
      wrs=1./sum(wgt)
      swgt=sqrt(wgt*wrs)
      rin_w=rin*swgt
!
! gather all data on root
      call mpi_gather(rin_w,n,mpifloat,rall(1,fbc0+1),n,mpifloat, &
     & 0,MPI_COMM_STRNG, ierror)
!
! the rest of reparametrization done by root
!
      if (qroot) then
! take care of BC
       if (fbc0.eq.1) rall(:,1) =r_bc_0*swgt
       if (fbc1.eq.1) rall(:,nrep)=r_bc_1*swgt
!
       alpha = 0d0 ! initial guess is zero
       dalpha= 0d0
!
! compute tangents
       drall(:,1:nrep-2)=rall(:,3:nrep)-rall(:,1:nrep-2) ! skipping central point
       drall(:,nrep-1)=0d0 ! initialize to zero (otherwise might be NAN)
       rall_new(:,1)=rall(:,1) ; rall_new(:,nrep)=rall(:,nrep) ! boundary conditions
! begin convergence iteration loop here
       iteration=0
       do
       iteration=iteration+1
! linearize about current alphas
        do i=2, nrep ! reparametrization modifies inner replicas only (include last point trivially)
         im=i-1
         rall_new(:,i)=rall(:,i) + alpha (i) * drall(:,im)
         drall_new(:,im)=rall_new(:,i)-rall_new(:,im) ! new tangents
         ds2_new(im)=dot_product(drall_new(:,im),drall_new(:,im))
        enddo
! compute coefficients
! first equation
        am(1)=0d0
        ao(1)= - dot_product(drall_new(:,1) + drall_new(:,2),drall(:,1))
        ap(1)= dot_product(drall_new(:,2), drall(:,2))
        cons(1)= ds2_new(2)-ds2_new(1) ! constraint
        rhs(1)=ao(1)*alpha(2) + ap(1)*alpha(3) &
     & - 0.5d0 * cons(1)
! middle equations
        do i=2,nrep-3
         am(i)= dot_product(drall_new(:,i), drall(:,i-1)) ! ok
         ao(i)=-dot_product(drall_new(:,i) +drall_new(:,i+1),drall(:,i))
         ap(i)= dot_product(drall_new(:,i+1), drall(:,i+1)) ! ok
         cons(i)=ds2_new(i+1) - ds2_new(i)
         rhs(i)=am(i)*alpha(i) + ao(i)*alpha(i+1) + ap(i)*alpha(i+2) &
     & - 0.5d0 * ( cons(i) )
        enddo
! last equation
        i=nrep-2
        am(i)= dot_product(drall_new(:,i), drall(:,i-1))
        ao(i)=-dot_product(drall_new(:,i) + drall_new(:,i+1),drall(:,i))
        ap(i)= 0d0
        cons(i)=ds2_new(i+1) - ds2_new(i)
        rhs(i)=am(i)*alpha(i) + ao(i)*alpha(i+1) &
     & - 0.5d0 * ( cons(i) )
!
! determine whether the constraint is satisfied
        err=sqrt(dot_product(cons,cons)/(nrep-2))
!
! write(600+ME_GLOBAL,*) iteration, err, alpha ! aa
! write(700+ME_GLOBAL,*) iteration, err, cons ! aa
! write(800+ME_GLOBAL,*) iteration, err, ds2_new ! aa
! write(900+ME_GLOBAL,*) iteration, drall_new(:,nrep-1), ds2_new ! aa
! write(500+ME_GLOBAL,*) iteration, rall_new(:,nrep) ! aa
! write(400+ME_GLOBAL,*) iteration, rall(:,nrep) ! aa
!
        if (err.lt.errtol) then
         exit
        elseif (iteration.gt.max_iterations) then
 call wrndie(0,whoami,trim('NO CONVERGENCE AFTER MAXIMUM NUMBER OF ITERATIONS.'))
! close(600+ME_GLOBAL)
         rall=rall_new
         exit
        else
! call matrix solver to get new alpha values
         call mdiag(a,rhs,nrep-2,ione)
         dalpha(2:nrep-1)=rhs-alpha(2:nrep-1)
!
! limit dalpha
         do i=2,nrep-1
          alpha(i)=alpha(i)+sign(min(abs(dalpha(i)),0.4d0),dalpha(i))
         enddo
!
! where (abs(dalpha).gt.0.4d0)
! alpha = alpha + sign (0.4d0, dalpha) ;
! elsewhere
! alpha = alpha + dalpha
! end where
! dum=maxval(abs(dalpha))
! alpha=alpha + dalpha * min(1d0,1d0/dum)
!
        endif
       enddo ! convergence loop
!
       rall=rall_new
!
       d_arclength=sqrt(ds2_new)
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       drall(:,1:nrep-2)=drall_new(:,2:nrep-1)-drall_new(:,1:nrep-2)
       curvature=sqrt(sum(drall(:,1:nrep-2)**2,1))/ &
     & (d_arclength(1:nrep-2)*d_arclength(2:nrep-1))
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      endif ! qroot
!ccccc now scatter new coordinates into rout
      call mpi_scatter(rall(1,fbc0+1), n, mpifloat, rout, n ,mpifloat, &
     & 0, MPI_COMM_STRNG, ierror)
! note: may want to broadcast curvature and arclength
!
! note: rall is scaled by wgt, so we need to unscale rout:
      rout=rout/swgt
!
      if (present(drout)) then
        call mpi_scatter(drall_new(1,fbc0), n, mpifloat, drout, n ,mpifloat, &
     & 0, MPI_COMM_STRNG, ierror)
! note: drout is scaled by wgt; to unscale, uncomment below
! drout=drout/swgt
! normalize drout to unity
        dum=dot_product(drout,drout)
        drout=drout/sqrt(dum)
      endif ! drout present
! done!
!cccccccccccccccccccccccccccccccccccccccccccccccccc
      deallocate(rall,drall,rall_new,drall_new)
      deallocate(rhs,a,cons,ds2_new,alpha,dalpha)
!
      end subroutine interp_linear_exact
!
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!----------------- COMPUTE VECTOR TANGENT TO PATH (2ND ORDER FD)
      subroutine compute_dr(rin,drout,wgt,n, &
     & d_arclength, curvature, &
     & r_bc_0, r_bc_1) ! provide additional arrays for fixed bc
!
      use multicom_aux;
      use mpi
      use chm_kinds
      implicit none
!
      integer :: n
      real(chm_real) :: rin(n), drout(n), wgt(n)
      real(chm_real) :: d_arclength(:), curvature(:)
      real(chm_real), optional :: r_bc_0(n), r_bc_1(n) ! optional fixed bc data
!
! locals
      real(chm_real) :: rin_w(n) ! r weighted by wgt
      real(chm_real) :: swgt(n)
      integer :: fbc0=0, fbc1=0 ! fixed bc local flags
      integer ierror, nall, i, nrep, me, nensem ! 9.08: replaced ncpu with nrep[lica]
!
! local arrays (f90)
      real(chm_real), pointer :: rall(:,:), drall(:,:), dr2all(:,:)
!
      integer*4 :: s_disp(SIZE_STRNG), r_disp(SIZE_STRNG), &
     & s_count(SIZE_STRNG), r_count(SIZE_STRNG) ! have to declare these as integer*4 -- it`s a problem!
      real(chm_real), pointer :: ds(:), dsn(:), ds2me(:), &
     & ds2all(:), s(:), curv_me(:)
      real(chm_real) :: wrs
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
! begin
! bail if not a replica root; will synchronize slave nodes elsewhere
      if (MPI_COMM_STRNG.eq.MPI_COMM_NULL) return
!
      if (present(r_bc_0)) fbc0=1
      if (present(r_bc_1)) fbc1=1
!
      me=ME_STRNG
      nensem=SIZE_STRNG
      nrep=nensem+fbc0+fbc1
! allocate work arrays, now that we know nrep
      allocate(ds(nrep-1), dsn(nrep-1), ds2me(nrep), ds2all(nrep-1), &
     & s(nrep),curv_me(nrep-2))
!
! determine send/receive displacements and allocate arays
      nall=ceiling(1.0d0*n/nrep)
      allocate(rall(nall,nrep), drall(nall,nrep), dr2all(nall,nrep-2))
! initialize
      rall=0.0
      drall=0.0
      dr2all=0.0
      do i=1,nensem
       s_disp(i)=min((i-1)*nall,n-1) ! in case n<ncpu; want to keep valid address
       r_disp(i)=(i-1+fbc0)*nall ! fixed bc0 correction
       s_count(i)=max(0,min(nall,n-nall*(i-1)))
      enddo
      r_count=s_count(me+1)
!
! weight rin by wgt
      wrs=1./sum(wgt)
      swgt=sqrt(wgt*wrs)
      rin_w=rin*swgt
      call mpi_alltoallv(rin_w,s_count,s_disp,mpifloat, &
     & rall,r_count,r_disp,mpifloat, &
     & MPI_COMM_STRNG, ierror)
!
! take care of BC as well
      if (fbc0.eq.1) &
     & rall(1:r_count(1),1)= &
     & r_bc_0(s_disp(me+1)+1:s_disp(me+1)+r_count(1))* &
     & swgt(s_disp(me+1)+1:s_disp(me+1)+r_count(1))
!
      if (fbc1.eq.1) &
     & rall(1:r_count(1),nrep)= &
     & r_bc_1(s_disp(me+1)+1:s_disp(me+1)+r_count(1))* &
     & swgt(s_disp(me+1)+1:s_disp(me+1)+r_count(1))
!
! now have partial coordinates from ALL replicas
! compute local part of the tangent
!
      drall(:,1:nrep-1)=rall(:,2:nrep)-rall(:,1:nrep-1)
      ds2me=sum(drall**2,1)
! reduce ds2
      call mpi_allreduce(ds2me, ds2all, nrep-1, mpifloat, &
     & MPI_SUM, MPI_COMM_STRNG, ierror)
! now can compute arclength
      s(1)=0
      do i=1,nrep-1
       ds(i)=sqrt(ds2all(i))
       s(i+1)=s(i)+ds(i)
      enddo
! save arclength and pass back to calling routine
      d_arclength=ds
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! normalize dr
      do i=1, nall
       drall(i,1:nrep-1)=drall(i,1:nrep-1)/ds(1:nrep-1)
      enddo
! compute curvature (second derivative, actually)
      dsn(1:nrep-2)=0.5*(ds(2:nrep-1)+ds(1:nrep-2)) ! dsn approximated at nodes, not centers
      do i=1, nall
       dr2all(i,1:nrep-2)=(drall(i,2:nrep-1)-drall(i,1:nrep-2)) &
     & /dsn(1:nrep-2)
      enddo
! compute magnitude
      curv_me=sum(dr2all**2,1)
! reduce from all cpus
      call mpi_allreduce(curv_me, curvature, nrep-2, mpifloat, &
     & MPI_SUM, MPI_COMM_STRNG, ierror)
! take square root
      curvature=sqrt(curvature)
! massage tangent vector (compute at nodes, rather than centers)
      drall(:,nrep)=drall(:,nrep-1) ! define last point
! NOTE: the line below is a bug: F90 does this 'vector' operation sequentially
! drall(:,2:nrep-1)=0.5*(drall(:,1:nrep-2)+drall(:,2:nrep-1)) ! redefine intermediate points
      do i=nrep-1,2,-1
       drall(:,i)=0.5*(drall(:,i)+drall(:,i-1)) ! redefine intermediate points
      enddo
!
!ccccc now scatter tangent vector
      call mpi_alltoallv(drall,r_count,r_disp,mpifloat, &
     & drout,s_count,s_disp,mpifloat, &
     & MPI_COMM_STRNG, ierror)
! drout is scaled by wgt, we can unscale it here if desired:
! drout=drout/swgt
! scale new drout to unit length
      wrs=dot_product(drout,drout)
      drout=drout/sqrt(wrs)
! done!
!cccccccccccccccccccccccccccccccccccccccccccccccccc
      deallocate(ds,dsn,ds2me,ds2all,s,curv_me)
      deallocate(rall, drall, dr2all)
!
      end subroutine compute_dr
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!ccccccccccccccccccc open file (relatively) cleanly cccccccccccccccc
      subroutine open_file(ifile,fname,form,acc)
!
      use stream
      use string
      use dimens_fcm
      use multicom_aux;
      use chm_kinds
!
      use machio, only : vopen
!
      implicit none
      character(len=132)::info(21)=(/'','','','','','','','','','','','','','','','','','','','',''/);! output buffer
!
! for reading a file
      integer, intent(inout) :: ifile
      character(len=*), intent(in) :: fname, form, acc
      character(len=len(fname)) :: fname_, fname_bkp
      character(len=len(form)) :: form_
      character(len=len(acc)) :: acc_
!
! locals
      integer :: error,ifile2,flen,k
      logical :: OPENUN,QFORM,QWRITE, QERROR
      integer :: l
!
      integer lunass ! function
      character(len=len("OPEN_FILE>") ),parameter::whoami="OPEN_FILE>";!macro
!
!ccccccccccccccccccc begin ccccccccccccccccccc
      if (iolev.lt.0) return
      if (ME_LOCAL.ne.0) return ! allow only the replica roots to open files
!
      fname_=fname
      flen=len(fname_)
      call trima(fname_, flen)
!
      form_=form
      l=len(form_)
      call trima(form_, l)
!
      acc_=acc
! if the unit is undefined, request a file handle
       if (ifile.eq.-1) Ifile=LUNASS(90)
! inquire into unit status
       fname_bkp=fname_
       CALL VINQRE('UNIT',fname_bkp,flen,k,OPENUN,QFORM,QWRITE,ifile)
       IF (OPENUN) THEN
           IF(WRNLEV.GE.2) WRITE(info,'(2A)') &
     & whoami,' Unit already open.', &
     & ' The old file will be closed first.'
           CALL VCLOSE(ifile,'KEEP',ERROR)
       ENDIF
! check if the file is connected to another number
! when calling by file, length field is destroyed
       fname_bkp=fname_
       CALL VINQRE('FILE',fname_bkp,k,k,OPENUN,QFORM,QWRITE,ifile2)
       IF (OPENUN) THEN
            IF(WRNLEV.GE.2) WRITE(info,'(A,/,2A)') &
     & whoami, &
     & ' ***** WARNING ***** another unit is already ', &
     & ' assigned to the file - it will be disconnected first.'
            CALL VCLOSE(ifile2,'KEEP',ERROR)
       ELSE ! try lower case
            fname_bkp=fname_
            call cnvtlc(fname_bkp,flen)
            CALL VINQRE('FILE',fname_bkp,k,k,OPENUN,QFORM,QWRITE,ifile2)
            IF (OPENUN) THEN
               IF(WRNLEV.GE.2) WRITE(info,'(A,/,2A)') &
     & whoami, &
     & ' ***** WARNING ***** another unit is already ', &
     & ' assigned to the file - it will be disconnected first.'
               CALL VCLOSE(ifile2,'KEEP',ERROR)
            endif
       endif
       CALL VOPEN(Ifile,fname_,form_,acc_,QERROR &
     & ,0 &
& )
       IF(QERROR) then
            CALL WRNDIE(0,whoami, &
     & '"OPEN" not possible.')
       ENDIF
      end subroutine open_file
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! frames align moved here from cv_frames modules because of circular dependency problem
      subroutine frames_align_string(x,y,z,mass,min_rmsd,ind)
       use cv_common, only: cv, priv_ptr, main, comp, cv_common_rmsd
       use cv_frames, only : frames, frames_reset_calculate, &
     & frames_calc, frames_initialized
       use smcv_master, only : smcv_fill ! used for rmsd test
       use sm_var, only : nstring, mestring
!
       use multicom_aux;
       use stream
       use mpi
       use chm_kinds
!
       implicit none
!
       real(chm_real) :: x(:), y(:), z(:), mass(:)
       logical, optional :: min_rmsd
       integer, optional :: ind ! frame index
!
       logical :: qrmsd
       real(chm_real) :: A1(3,3), A2(3,3), A3(3,3), A4(3,3), o(3), xx, yy, zz, w
       real(chm_real), pointer, dimension(:) :: x0, x1, x2, x3, x4, &
     & y0, y1, y2, y3, y4, &
     & z0, z1, z2, z3, z4, &
     & m0
       real(chm_real) :: totm0, corr1, corr2, corr3, corr4, cmax, &
     & rmsd1, rmsd2, rmsd3, rmsd4, mrmsd ! for rmsd alignment
       integer, pointer, dimension (:) :: ind0
       integer :: i, j, ii, jj, ibeg, iend, ncom
       integer :: stat(MPI_STATUS_SIZE)
       integer :: error
!
 character(len=132)::info(21)=(/'','','','','','','','','','','','','','','','','','','','',''/);! output buffer
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
       character(len=len("FRAMES_ALIGN_STRING>") ),parameter::whoami="FRAMES_ALIGN_STRING>";!macro
! do work
!
! check for initialization
       if (.not.frames_initialized) then
         call wrndie(0,whoami,trim('NO FRAMES DEFINED. NOTHING DONE.'))
         return
       endif
!
! make sure the frame index is valid
       if (present(ind)) then ! reset ith frame
! check frame number:
         if (ind.lt.1.or.ind.gt.frames%num_frames) then
          call wrndie(0,whoami,trim('OUT OF BOUNDS. NOTHING DONE.'))
          return
         endif
!
         ibeg=ind; iend=ind
        else ! select all
         ibeg=1; iend=frames%num_frames
         if (ibeg.gt.iend) then
          call wrndie(0,whoami,trim('NO FRAMES DEFINED. NOTHING DONE.'))
          return
         endif
       endif
!cccc valid frames have been selected;
!
       if (MPI_COMM_STRNG.ne.MPI_COMM_NULL) then ! only replica roots do this part
!
!cccc check whether to test coordinates ccccccccccccccccccccc
        if (present(min_rmsd)) then; qrmsd=min_rmsd;
        else; qrmsd=.false.; endif
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!
        do i=ibeg, iend
! access private array of frame
         ncom=frames%priv(i)%p(1)
!cccc allocate necessary storage
         allocate(ind0(ncom),x0(ncom),y0(ncom),z0(ncom),m0(ncom)) ! absolute coordinates
! trial coordinates
         allocate(x1(ncom),y1(ncom),z1(ncom))
         allocate(x2(ncom),y2(ncom),z2(ncom))
         allocate(x3(ncom),y3(ncom),z3(ncom))
         allocate(x4(ncom),y4(ncom),z4(ncom))
! get frame axes
! (1) extract indices into the atom map
         ii=2; jj=ii+ncom-1; ind0=frames%priv(i)%p(ii:jj)
!ccc (2) now calculate (possibly redundant but safe)
         call frames_reset_calculate(.true.,i)
         call frames_calc(i,x,y,z,mass,.false.)
         call frames_reset_calculate(.true.,i)
! get center of mass
         o=frames%o(:,i)
! get matrix of engenvectors
         A1=frames%r(:,:,i);
!cccc generate equivalent axes (assuming no eigenvalue degeneracy)
! eigenvectors are unique up to a sign ; reversing two at a time
! preserves right-handedness
         A2(:,1)=-A1(:,1); A2(:,2)=-A1(:,2); A2(:,3)= A1(:,3);
         A3(:,1)=-A1(:,1); A3(:,2)= A1(:,2); A3(:,3)=-A1(:,3);
         A4(:,1)= A1(:,1); A4(:,2)=-A1(:,2); A4(:,3)=-A1(:,3);
! get coordinates
         do j=1, ncom;
          ii=cv%amap%i(ind0(j)) ! psf index
          x0(j)=x(ii)-o(1); y0(j)=y(ii)-o(2); z0(j)=z(ii)-o(3) ! subtracted COM
          m0(j)=mass(ii)
         enddo
         totm0=1./sum(m0)
! apply rotation (TRANSPOSE A) to get oriented coordinates
         do j=1,ncom
          w=sqrt(m0(j)*totm0) ! normalization
          xx=x0(j)*w ! mass-weighting
          yy=y0(j)*w
          zz=z0(j)*w
!
          x1(j)=A1(1,1)*xx+A1(2,1)*yy+A1(3,1)*zz
          y1(j)=A1(1,2)*xx+A1(2,2)*yy+A1(3,2)*zz
          z1(j)=A1(1,3)*xx+A1(2,3)*yy+A1(3,3)*zz
! alternative coordinates
          x2(j)=A2(1,1)*xx+A2(2,1)*yy+A2(3,1)*zz
          y2(j)=A2(1,2)*xx+A2(2,2)*yy+A2(3,2)*zz
          z2(j)=A2(1,3)*xx+A2(2,3)*yy+A2(3,3)*zz
! alternative coordinates
          x3(j)=A3(1,1)*xx+A3(2,1)*yy+A3(3,1)*zz
          y3(j)=A3(1,2)*xx+A3(2,2)*yy+A3(3,2)*zz
          z3(j)=A3(1,3)*xx+A3(2,3)*yy+A3(3,3)*zz
! alternative coordinates
          x4(j)=A4(1,1)*xx+A4(2,1)*yy+A4(3,1)*zz
          y4(j)=A4(1,2)*xx+A4(2,2)*yy+A4(3,2)*zz
          z4(j)=A4(1,3)*xx+A4(2,3)*yy+A4(3,3)*zz
!
         enddo
!ccccccccccccccccc at this stage try to select optimal axes based on the least (z-theta(x))
! call smcv master to fill cv, then loop over the cv, picking out the ones that depend
! on the frame in question; compute the appropriate rmsd and pick the orientation that
! gives the least rmsd
! NOTE: this is "unclean": this module has no business processing
! collective variables, I am including this code as a last resort to guess the correct frame
! vectors; in addition, I am calling the cv_rmsd routine, which uses weights array; the weights,
! however, would change if they are calculated from M, since M depends on d_theta/dx,
! which depend in the frame; for Cartesian positions, it seems, the derivatives would simply change sign
! so that there would not be a difference in M; for quaternions/anglvec variables this would be more
! complicated.
         if (qrmsd) then
          call smcv_fill(x,y,z,mass,comp) ! fill (overwrite) comparison array
          rmsd1=cv_common_rmsd(main,comp) ! note that I am assuming that the main column has the "CORRECT" z
!ccc calculate rmsd`s for all frames
          frames%r(:,:,i)=A2;
          call smcv_fill(x,y,z,mass,comp)
          rmsd2=cv_common_rmsd(main,comp)
!
          frames%r(:,:,i)=A3;
          call smcv_fill(x,y,z,mass,comp)
          rmsd3=cv_common_rmsd(main,comp)
!
          frames%r(:,:,i)=A4;
          call smcv_fill(x,y,z,mass,comp)
          rmsd4=cv_common_rmsd(main,comp)
!
! write(600+mestring,*) rmsd1, rmsd2, rmsd3, rmsd4
! close(600+mestring)
!
          cv%r(:,comp)=cv%r(:,main); ! set comp set to reasonable values in case user does nothing
!
          mrmsd=min(rmsd1, rmsd2, rmsd3, rmsd4)
! 0th replica (root) sets its frame based on rmsd
          if (mestring.eq.0) then
           if (mrmsd.eq.rmsd1) then
            frames%r(:,:,i)=A1;
           elseif (mrmsd.eq.rmsd2) then
            frames%r(:,:,i)=A2; x1=x2; y1=y2; z1=z2;
           elseif (mrmsd.eq.rmsd3) then
            frames%r(:,:,i)=A3; x1=x3; y1=y3; z1=z3;
           elseif (mrmsd.eq.rmsd4) then
            frames%r(:,:,i)=A4; x1=x4; y1=y4; z1=z4;
           endif ! mrmsd
          endif ! whoiam
         endif ! qrmsd
!
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!
!ccc (1) if we are replicas [1 .. nstring-1], first receive aligned coordinates from
!ccc replica on the left (0 .. nstring -2)
         if (mestring.gt.0) then
          call MPI_RECV(x0, ncom, mpifloat, mestring-1, &
     & 1, MPI_COMM_STRNG, stat, error)
          call MPI_RECV(y0, ncom, mpifloat, mestring-1, &
     & 2, MPI_COMM_STRNG, stat, error)
          call MPI_RECV(z0, ncom, mpifloat, mestring-1, &
     & 3, MPI_COMM_STRNG, stat, error)
!ccc now manipulate local axes to achieve best RMSD
          corr1=dot_product(x0,x1)+dot_product(y0,y1)+dot_product(z0,z1)
          corr2=dot_product(x0,x2)+dot_product(y0,y2)+dot_product(z0,z2)
          corr3=dot_product(x0,x3)+dot_product(y0,y3)+dot_product(z0,z3)
          corr4=dot_product(x0,x4)+dot_product(y0,y4)+dot_product(z0,z4)
!ccc pick the set of axis that yields optimal alignment
          cmax=max(corr1, corr2, corr3, corr4);
          if (cmax.eq.corr1) then;
           frames%r(:,:,i)=A1; ! this is already true
           if (qrmsd.and.mrmsd.ne.rmsd1) &
     & write(info,600) whoami,mestring,i,whoami; write(OUTU,'(A)') pack(info,info.ne.'');info='';
          elseif (cmax.eq.corr2) then
           frames%r(:,:,i)=A2; x1=x2; y1=y2; z1=z2;
           if (qrmsd.and.mrmsd.ne.rmsd2) &
     & write(info,600) whoami,mestring,i,whoami; write(OUTU,'(A)') pack(info,info.ne.'');info='';
          elseif (cmax.eq.corr3) then
           frames%r(:,:,i)=A3; x1=x3; y1=y3; z1=z3;
           if (qrmsd.and.mrmsd.ne.rmsd3) &
     & write(info,600) whoami,mestring,i,whoami; write(OUTU,'(A)') pack(info,info.ne.'');info='';
          elseif (cmax.eq.corr4) then
           frames%r(:,:,i)=A4; x1=x4; y1=y4; z1=z4;
           if (qrmsd.and.mrmsd.ne.rmsd4) &
     & write(info,600) whoami,mestring,i,whoami; write(OUTU,'(A)') pack(info,info.ne.'');info='';
          endif
 600 FORMAT(A,' FOR REPLICA ',I3,', FRAME ',I3,', STRING ALIGNMENT AND'&
     & ,/,A,' MIN. CV RMSD CRITERIA DIFFER.')
!cc done
         endif ! whoiam
!ccc (2) if we are replicas [0 .. nensem-2], send aligned local coordinates to
!ccc replica on the right (1 .. nensem -1)
         if (mestring.lt.nstring-1) then
          call MPI_SEND(x1, ncom, mpifloat, mestring+1, &
     & 1, MPI_COMM_STRNG, error)
          call MPI_SEND(y1, ncom, mpifloat, mestring+1, &
     & 2, MPI_COMM_STRNG, error)
          call MPI_SEND(z1, ncom, mpifloat, mestring+1, &
     & 3, MPI_COMM_STRNG, error)
         endif ! whoiam
! free memory
!
         deallocate(x0, y0, z0, m0, ind0, x1, y1, z1, &
     & x2, y2, z2, x3, y3, z3, x4, y4, z4)
        enddo ! over frame indices
!
       endif ! MPI_COMM_STRING .ne. MPI_COMM_NULL
!
! now broadcast relevant frames + cv to slave nodes
       if (MPI_COMM_LOCAL.ne.MPI_COMM_NULL.and.SIZE_LOCAL.gt.1) then
        call MPI_BCAST(frames%r(:,:,ibeg:iend), (iend-ibeg+1)*9, &
     & mpifloat, 0, MPI_COMM_LOCAL, error)
        call MPI_BCAST(cv%r(1,main), cv%num_cv, mpifloat, &
     & 0, MPI_COMM_LOCAL, error)
        call MPI_BCAST(cv%r(1,comp), cv%num_cv, mpifloat, &
     & 0, MPI_COMM_LOCAL, error)
       endif
!
! frames may have changed as a result of our 'acrobatics', but not the
! frame gradients; therefore, this is essential (especially on root nodes) :
       call frames_reset_calculate(.true.) ! says that frames should be recalculated
!
      end subroutine frames_align_string
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine frame_align_rmsd(x,y,z,mass,ind)
! calculate optimal frame axes, based on resulting RMSD between known CV and instantaneous CV
! this is a much simpler version of frames_align_string; currently used only in the SMCV interpolation routine
! note that I assume that the main array has the (preset) CV values
       use cv_common, only: cv, main, comp, cv_common_rmsd
       use cv_frames, only : frames, frames_reset_calculate, &
     & frames_calc, frames_initialized
       use smcv_master, only : smcv_fill ! used for rmsd test
!
      use mpi
      use multicom_aux;
      use chm_kinds
!
      implicit none
!
       real(chm_real) :: x(:), y(:), z(:), mass(:)
       integer, optional :: ind ! frame index
!
       real(chm_real) :: A1(3,3), A2(3,3), A3(3,3), A4(3,3)
       real(chm_real) :: rmsd1, rmsd2, rmsd3, rmsd4, mrmsd ! for rmsd alignment
       integer :: i, ibeg, iend, error
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
       character(len=len("FRAME_ALIGN_RMSD>") ),parameter::whoami="FRAME_ALIGN_RMSD>";!macro
! do work
!
! check for initialization
       if (.not.frames_initialized) then
         call wrndie(0,whoami,trim('NO FRAMES DEFINED. NOTHING DONE.'))
         return
       endif
!
! make sure the frame index is valid
       if (present(ind)) then ! reset ith frame
! check frame number:
         if (ind.lt.1.or.ind.gt.frames%num_frames) then
          call wrndie(0,whoami,trim('OUT OF BOUNDS. NOTHING DONE.'))
          return
         endif
!
         ibeg=ind; iend=ind
        else ! select all
         ibeg=1; iend=frames%num_frames
         if (ibeg.gt.iend) then
          call wrndie(0,whoami,trim('NO FRAMES DEFINED. NOTHING DONE.'))
          return
         endif
       endif
!cccc valid frames have been selected;
!
       if (MPI_COMM_STRNG.ne.MPI_COMM_NULL) then ! only replica roots do this part
!
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!
        do i=ibeg, iend
         call frames_reset_calculate(.true.,i)
         call frames_calc(i,x,y,z,mass,.false.)
         call frames_reset_calculate(.true.,i)
! get matrix of engenvectors
         A1=frames%r(:,:,i);
!cccc generate equivalent axes (assuming no eigenvalue degeneracy)
         A2(:,1)=-A1(:,1); A2(:,2)=-A1(:,2); A2(:,3)= A1(:,3);
         A3(:,1)=-A1(:,1); A3(:,2)= A1(:,2); A3(:,3)=-A1(:,3);
         A4(:,1)= A1(:,1); A4(:,2)=-A1(:,2); A4(:,3)=-A1(:,3);
!
         call smcv_fill(x,y,z,mass,comp) ! fill (overwrite) comparison array
         rmsd1=cv_common_rmsd(main,comp) ! note that I am assuming that the main column has the "CORRECT" z
!ccc calculate rmsd`s for all frames
         frames%r(:,:,i)=A2;
         call smcv_fill(x,y,z,mass,comp)
         rmsd2=cv_common_rmsd(main,comp)
!
         frames%r(:,:,i)=A3;
         call smcv_fill(x,y,z,mass,comp)
         rmsd3=cv_common_rmsd(main,comp)
!
         frames%r(:,:,i)=A4;
         call smcv_fill(x,y,z,mass,comp)
         rmsd4=cv_common_rmsd(main,comp)
!
! write(600+mestring,*) rmsd1, rmsd2, rmsd3, rmsd4
! close(600+mestring)
!
         cv%r(:,comp)=cv%r(:,main); ! set comp set to reasonable values in case user does nothing
!
         mrmsd=min(rmsd1, rmsd2, rmsd3, rmsd4)
!
         if (mrmsd.eq.rmsd1) then
            frames%r(:,:,i)=A1;
         elseif (mrmsd.eq.rmsd2) then
            frames%r(:,:,i)=A2;
         elseif (mrmsd.eq.rmsd3) then
            frames%r(:,:,i)=A3;
         elseif (mrmsd.eq.rmsd4) then
            frames%r(:,:,i)=A4;
         endif ! mrmsd
!
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!
        enddo ! over frame indices
!
       endif ! MPI_COMM_STRING .ne. MPI_COMM_NULL
!
! now broadcast relevant frames + cv to slave nodes
       if (MPI_COMM_LOCAL.ne.MPI_COMM_NULL.and.SIZE_LOCAL.gt.1) then
        call MPI_BCAST(frames%r(:,:,ibeg:iend), (iend-ibeg+1)*9, &
     & mpifloat, 0, MPI_COMM_LOCAL, error)
        call MPI_BCAST(cv%r(1,main), cv%num_cv, mpifloat, &
     & 0, MPI_COMM_LOCAL, error)
        call MPI_BCAST(cv%r(1,comp), cv%num_cv, mpifloat, &
     & 0, MPI_COMM_LOCAL, error)
       endif
!
! frames may have changed as a result of our 'acrobatics', but not the
! frame gradients; therefore, this is essential (especially on root nodes) :
       call frames_reset_calculate(.true.) ! says that frames should be recalculated
!
      end subroutine frame_align_rmsd
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine frame_align_voro(x,y,z,mass,ind)
! calculate optimal frame axes such that the current replica is in the voronoi cell
! specified in the voronoi map. This requires the voronoi map to be present.
! if replica cannot be made consitent with the map, a warning is issued
! currently works correctly if only one frame has been defined
! with more frames have to consider permutations
       use cv_common, only: cv, cv_common_voronoi_initialized
       use cv_frames, only : frames, frames_reset_calculate, &
     & frames_initialized
       use smcv_master, only : smcv_voronoi_whereami ! used for rmsd test
       use sm_var, only : mestring
!
       use stream
       use chm_kinds
!
       implicit none
!
       real(chm_real) :: x(:), y(:), z(:), mass(:)
       integer, optional :: ind ! frame index
!
       real(chm_real) :: A1(3,3), A2(3,3), A3(3,3), A4(3,3)
       integer :: where1, where2, where3, where4, me
       integer :: i, ibeg, iend
character(len=132)::info(21)=(/'','','','','','','','','','','','','','','','','','','','',''/);! output buffer
       character(len=len("FRAME_ALIGN_VORO>") ),parameter::whoami="FRAME_ALIGN_VORO>";!macro
! do work
!
! check for initialization
       if (.not.frames_initialized) then
         call wrndie(0,whoami,trim('NO FRAMES DEFINED. NOTHING DONE.'))
         return
       endif
!
       if (.not.cv_common_voronoi_initialized) then
         call wrndie(0,whoami,trim('VORONOI DATA NOT INITIALIZED. NOTHING DONE.'))
         return
       endif
!
! check if a valid map is present
       if (any(cv%voronoi_map.eq.-1)) then
         call wrndie(0,whoami,trim('VORONOI MAP CONTAINS INVALID ENTRIES. NOTHING DONE.'))
         return
       endif
!
       me=cv%voronoi_map(mestring+1)
!
! make sure the frame index is valid
       if (present(ind)) then ! reset ith frame
! check frame number:
         if (ind.lt.1.or.ind.gt.frames%num_frames) then
          call wrndie(0,whoami,trim('OUT OF BOUNDS. NOTHING DONE.'))
          return
         endif
!
         ibeg=ind; iend=ind
        else ! select all
         ibeg=1; iend=frames%num_frames
         if (ibeg.gt.iend) then
          call wrndie(0,whoami,trim('NO FRAMES DEFINED. NOTHING DONE.'))
          return
         endif
       endif
!cccc valid frames have been selected;
!
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!
       do i=ibeg, iend
         call smcv_voronoi_whereami(x,y,z,mass)
         where1=cv%voronoi_whereami
! matrix of engenvectors
         A1=frames%r(:,:,i);
!cccc generate equivalent axes (assuming no eigenvalue degeneracy)
         A2(:,1)=-A1(:,1); A2(:,2)=-A1(:,2); A2(:,3)= A1(:,3);
         A3(:,1)=-A1(:,1); A3(:,2)= A1(:,2); A3(:,3)=-A1(:,3);
         A4(:,1)= A1(:,1); A4(:,2)=-A1(:,2); A4(:,3)=-A1(:,3);
!
         frames%r(:,:,i)=A2;
         call smcv_voronoi_whereami(x,y,z,mass)
         where2=cv%voronoi_whereami
!
         frames%r(:,:,i)=A3;
         call smcv_voronoi_whereami(x,y,z,mass)
         where3=cv%voronoi_whereami
!
         frames%r(:,:,i)=A4;
         call smcv_voronoi_whereami(x,y,z,mass)
         where4=cv%voronoi_whereami
!
         if (me.eq.where1) then
            frames%r(:,:,i)=A1;
            cv%voronoi_whereami=me
         elseif (me.eq.where2) then
            frames%r(:,:,i)=A2;
            cv%voronoi_whereami=me
         elseif (me.eq.where3) then
            frames%r(:,:,i)=A3;
            cv%voronoi_whereami=me
         elseif (me.eq.where4) then
            frames%r(:,:,i)=A4;
            cv%voronoi_whereami=me
         else
            frames%r(:,:,i)=A1; ! default (unsafe)
            cv%voronoi_whereami=where1
            write(info,601) whoami,mestring ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
 601 FORMAT(A,' REPLICA ',I3,' IS OUTSIDE OF ASSIGNED CELL.')
         endif ! mrmsd
!
       enddo ! over frame indices
!
! frames may have changed as a result of our 'acrobatics', but not the
! frame gradients; therefore, this is essential (especially on root nodes) :
       call frames_reset_calculate(.true.) ! says that frames should be recalculated
!
      end subroutine frame_align_voro
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine hypercube_allgatherv(message,count,displ,type, &
     & comm, error, rank, size)
! custom in-place all-gatherv
!
      use mpi
      use multicom_aux;
      use chm_kinds
!
      implicit none
!
      real(chm_real) :: message(*)
      integer :: error
      integer*4 :: comm, rank, size, type, count(size), displ(size)
! local variables
      integer*4 :: step, partner, scount, sdispl, rcount, rdispl, ind
      integer*4 :: stat(MPI_STATUS_SIZE)
!
      step=1
      do while (step.lt.size)
       partner = rank - step * &
     & ( &
     & mod ( rank, step*2 ) & ! compute "group" id
     & / step * 2 - 1 & ! split each group into two subgroups and assign -1 to first, and +1 to second
     & )
       ind = rank - mod (rank, step) + 1
       sdispl = displ ( ind )
       scount = sum ( count ( ind : ind + step - 1 ) )
!
! write(ME_GLOBAL+500,'(5I6)') ! aa
! & step, rank, partner, ind, sdispl, scount ! aa
       ind = partner - mod (partner, step) + 1
       rdispl = displ ( ind )
       rcount = sum ( count ( ind : ind + step - 1 ) )
! write(ME_GLOBAL+500,'(5I6)') ! aa
! & step, rank, partner, ind, rdispl, rcount ! aa
       call mpi_sendrecv(message(sdispl+1),scount,type,partner,rank, &
     & message(rdispl+1),rcount,type,partner,partner, &
     & comm, stat, error)
       step = step*2
      enddo
! close(ME_GLOBAL+500)
!
      end subroutine hypercube_allgatherv
!
!=================================================================================
      function sm_get_column(cmd_, l, qcoltag, missing) result(C)
      use sm_var, only : smcv_initialized
      use ftsm_var, only : center, ftsm_ref=>ref, ftsm_initialized
      use cv_common, only : main, comp, smcv_ref=>ref
      use string
      implicit none
!
      character(len=*) :: cmd_
      integer :: l, missing
      logical :: qcoltag
!
      integer, parameter :: kl=16
      character(len=kl) :: keyword
      integer :: i, C
!
      C=missing ! default column
      if (qcoltag) then
       call gtrmwa(cmd_, l, 'COL', 3, keyword, kl, i)
       if (i.eq.0) return ! 'COL' not found
      else
       keyword=nexta8(cmd_,l)
      endif
      call cnvtuc(keyword,len(keyword))
      if (ftsm_initialized) then
       if(keyword(1:4).eq.'MAIN') then ; C=center;
       elseif(keyword(1:3).eq.'REF') then; C=ftsm_ref;
! elseif( (keyword(1:4).eq.'DYNA') .or.(keyword(1:3).eq.'OLD') ) then; C=center_old;
       else; i=kl; C=nexti(keyword, i);
       endif
      elseif (smcv_initialized) then
       if(keyword(1:4).eq.'MAIN') then ; C=main;
       elseif(keyword(1:3).eq.'REF') then; C=smcv_ref;
       elseif( (keyword(1:4).eq.'DYNA') .or.(keyword(1:3).eq.'OLD') ) then; C=comp;
       else; i=kl; C=nexti(keyword, i);
       endif
      endif
!
      end function sm_get_column
!=================================================================================
#endif
#endif /* automatically protect all code */
