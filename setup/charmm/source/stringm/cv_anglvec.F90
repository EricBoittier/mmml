! ********************************************************************************!
! Preprocessed by GNU CPP !
! ********************************************************************************!
!
! CV_ANGLVEC.MOD
!
! ROUTINES FOR COLLECTIVE VARIABLE `ANGLE` BETWEEN TWO VECTORS IN TWO COORDINATE FRAMES
! IF THE TWO FRAMES ARE THE SAME, THEN THE ABSOLUTE FRAME IS USED; NOTE THAT IF ONLY
! CONSTANT VECTORS ARE SPECIFIED,
! THIS CV TYPE GENERALIZES CV_ANGLE AND HAS MUCH MORE VERSATILITY; CV_ANGLE IS KEPT FOR ITS
! SIMPLICITY
!
! NOTE THAT WE CANNOT SIMULATE A PATH THAT CROSSES 0/180 WITHOUT ADDING A (4TH) REFERENCE POINT
! WHICH WOULD REMOVE REFLECTIONAL SYMMETRY, SO THAT THETA AND -THETA ARE THEN _DIFFERENT_ ANGLES
! MIGHT ADD AN ADDITIONAL COLLECTIVE VARIABLE TO TAKE CARE OF THIS IN THE FUTURE
      module cv_anglvec
!
#if (KEY_STRINGM==1) /*  automatically protect all code */
!
      use cv_common
      use cv_types
      use cv_frames
      use ivector_list ! vector class for storing atom lists
!
      implicit none
      private
      ! subroutines
      public cv_anglvec_add
      public cv_anglvec_calc
      public cv_anglvec_list
!
      contains
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       function cv_anglvec_add(atom_list,p,fr1,fr2,k,gamma,weight) ! note: i is the atom index in the PSF
       use stream
       integer :: fr1, fr2
       real(chm_real) :: k, gamma, weight
       real(chm_real), dimension(4,3) :: p
       type (int_vector), dimension(4) :: atom_list
! locals
       integer :: i, j, l, m, ind, num_int
       integer :: f1, f2, ii, jj, ncomf
       integer, pointer :: indf(:)
       logical :: found, cv_anglvec_add
       real(chm_real), parameter :: errtol = 1e-8
       real(chm_real) :: rn
       integer , dimension(4) :: ncom
       type (int_vector) :: unique_amap_ptr ! contains a unique list of atom map pointers (to speed up gradient computation, etc)
!
       character(len=len("CV_ANGLVEC_ADD>") ),parameter::whoami="CV_ANGLVEC_ADD>";!macro
!
! check that input options are valid
!cccccccccccccccc check frame indices ccccccccccccccccccccccccccccccccc
       if (fr1.ge.0.and.fr1.le.frames%num_frames) then
        f1=fr1
       else
        call wrndie(0,whoami,trim(' INVALID FRAME SPECIFIED. WILL USE FIXED COORDINATE SYSTEM.'))
        f1=0
       endif
!
       if (fr2.ge.0.and.fr2.le.frames%num_frames) then
        f2=fr2
       else
        call wrndie(0,whoami,trim(' INVALID FRAME SPECIFIED. WILL USE FIXED COORDINATE SYSTEM.'))
        f2=0
       endif
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       if ( (f1.eq.f2) .and. (f1.ne.0) ) then
        call wrndie(0,whoami,trim(' FRAMES ARE THE SAME. ABSOLUTE FRAME WILL BE USED.'))
        f1=0; f2=0;
       endif
!
       do i=1,4; ncom(i)=atom_list(i)%last; enddo
! make sure constant vectors do not have zero length
       if (ncom(1).eq.0.and.ncom(2).eq.0) then
        rn=sqrt( sum( (p(2,:)-p(1,:))**2) );
        if (rn.lt.errtol) then
          call wrndie(0,whoami,trim(' FIRST VECTOR HAS NEARLY ZERO LENGTH. NOTHING DONE'))
          return
        endif
       endif
       if (ncom(3).eq.0.and.ncom(4).eq.0) then
        rn=sqrt( sum( (p(4,:)-p(3,:))**2) );
        if (rn.lt.errtol) then
          call wrndie(0,whoami,trim(' SECOND VECTOR HAS NEARLY ZERO LENGTH. NOTHING DONE'))
          return
        endif
       endif
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!
! check for duplicate CV (exact identical entry only)
       found=.false.
       do l=1, cv%num_cv
        if (cv%type(l).eq.anglvec) then
         found=.true.
         do i=1,4
          if (found) found=(ncom(i).eq.cv%priv(l)%p(i))
         enddo
         ind=5
         do i=1,4
          do j=1,ncom(i)
           if (found) found= &
     & (atom_list(i)%i(j).eq.cv%amap%i(cv%priv(l)%p(ind)))
           ind=ind+1
          enddo
         enddo
         if (found) found= &
     & (f1.eq.cv%priv(l)%p(ind)) &
     & .and.(f2.eq.cv%priv(l)%p(ind+1))
! also loop over the *real* private array
         ind=1
         do i=1,4; do j=1,3
                    if (found) found=(p(i,j).eq.cv%priv(l)%pr(ind))
                    ind=ind+1
         enddo; enddo
!
        endif
        if (found) exit
       enddo
!
       if (.not.found) then ! (if found -- do nothing)
        l=cv_common_add(k,gamma,weight,anglvec) ! get a new cv index
        if (l.gt.0) then
! allocate private data
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! space needed:
! integer data:
         num_int = 4 + sum(ncom) + 2 ! number of ints needded for storage
!
         allocate(cv%priv(l)%p(num_int));
         cv%priv(l)%p(1:4)=ncom(1:4)
! now add atom indices
         ind=5
         do i=1,4
          do j=1,ncom(i)
           m=atom_list(i)%i(j)
           if (m.le.0) call wrndie(0,whoami,trim(' INVALID ATOM INDEX SPECIFIED.'))
           cv%priv(l)%p(ind)=int_vlist_uaddu(cv%amap,m,l) ! add indices into unique map
           m=int_vector_uadd(unique_amap_ptr,cv%priv(l)%p(ind))
           ind=ind+1
          enddo
         enddo
! add frames
         cv%priv(l)%p(ind)=f1
         cv%priv(l)%p(ind+1)=f2
! real data:
         allocate(cv%priv(l)%pr(12));
         do i=1,4; do j=1,3
                    cv%priv(l)%pr(j+(i-1)*3)=p(i,j)
         enddo; enddo
!
! 10.2010 VO *************************************************************************
! since the CV depends on atom coords through the definition of a frame,
! we have to add the relevant atom lists in cv%amap with the index of this cv (l)
         if (f1.gt.0) then
          ncomf=frames%priv(f1)%p(1); ! number of atoms that define the frame
          allocate(indf(ncomf)); ! allocate temporary array for indices into amap
          ii=2; jj=ii+ncomf-1; ! range in the private array that corresponds to indices
          indf=frames%priv(f1)%p(ii:jj) ! copy atom indices from amap into temporary array
          do j=1, ncomf ! loop over frame atoms (amap indices)
            i=int_vector_uadd(cv%amap%v(indf(j)),l) ! add index of this cv (l) to the list corr. to amap index indf(j)
            i=int_vector_uadd(unique_amap_ptr,indf(j)) ! add index of the frame atom to the list of unique atom labels
          enddo
          deallocate(indf)
         endif
!
         if (f2.gt.0) then
          ncomf=frames%priv(f2)%p(1);
          allocate(indf(ncomf));
          ii=2; jj=ii+ncomf-1;
          indf=frames%priv(f2)%p(ii:jj)
          do j=1, ncomf ! loop over frame atoms
            i=int_vector_uadd(cv%amap%v(indf(j)),l) ! add index of this cv
            i=int_vector_uadd(unique_amap_ptr,indf(j)) ! add index of the frame atom to the list of unique atom labels
          enddo
          deallocate(indf)
         endif
!*****************************************************************************************
!cccc done adding
!
         m=unique_amap_ptr%last ! number of unique atoms this cv depends on
         allocate(cv%priv(l)%amap_ptr(m+1)) ! add one to include length of list
         cv%priv(l)%amap_ptr(1)=m;
         cv%priv(l)%amap_ptr(2:m+1)=unique_amap_ptr%i(1:m)
         call int_vector_done(unique_amap_ptr)
!
         cv_anglvec_add=.true.
        else ! out of bounds
         call wrndie(0,whoami,trim(' ERROR ADDING ANGLVEC CV. NOTHING DONE.'))
         cv_anglvec_add=.false.
        endif
       else ! found
         call wrndie(0,whoami,trim(' ANGLVEC CV ALREADY PRESENT. NOTHING DONE.'))
         cv_anglvec_add=.false.
       endif
       end function cv_anglvec_add
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine cv_anglvec_calc(i,x,y,z,mass,fx,fy,fz, &
     & calctheta,deriv,addforce,fext)
       use consta
       use sm_var
!
       real(chm_real) :: x(:), y(:), z(:), &
     & fx(:), fy(:), fz(:), mass(:)
       real(chm_real), optional :: fext ! `external` force for planar dynamics (see smcv_addforce in smcv_master)
       integer :: i
       logical :: calctheta ! whether or not to calculate theta(x); if not, use theta=cv%r(i,instant)
                            ! note that if calctheta=.false., we dont calculate the derivatives!
       logical :: deriv ! whether or not to calculate derivatives w.r.t. x
       logical :: addforce ! whether or not to add forces on simulation atoms
!
       real(chm_real), parameter :: tol=1.0e-10
       real(chm_real) :: dummy
!
       integer :: j, ii, jj, &
     & fr1, fr2 ! frame indices
! variables for cv and derivative calculations
       real(chm_real), dimension(3) :: u, v, & ! the two vectors
     & u0, v0, & ! vectors transformed into the absolute (0) frame
     & fu, fv ! derivatives of angle w.r.t. vector components
       integer :: ncom1, ncom2, ncom3, ncom4, ncomf, ind! , indpsf
       integer, pointer, dimension(:) :: ind1, ind2, ind3, ind4, &
     & indf ! , indpsf1, indpsf2, indpsf3, indpsf4
       real(chm_real), pointer, dimension(:) :: x1, x2, x3, x4
       real(chm_real), pointer, dimension(:) :: y1, y2, y3, y4
       real(chm_real), pointer, dimension(:) :: z1, z2, z3, z4
       real(chm_real), pointer, dimension(:) :: m1, m2, m3, m4
       real(chm_real) :: theta, costh, sinth, &
     & xcom1, xcom2, xcom3, xcom4, ycom1, ycom2, ycom3, ycom4,&
     & zcom1, zcom2, zcom3, zcom4, &
! & ! vx, vy, vz, vn, ux, uy, uz, un,
     & v0x, v0y, v0z, v0n, u0x, u0y, u0z, u0n, &
     & f, w, totm1=0d0, totm2=0d0, totm3=0d0, totm4=0d0
       real(chm_real) :: A(3,3), B(3,3) ! transformation matrices (frame vectors)
       real(chm_real) :: dA(3,3) ! derivatives of transformation matrices

       logical :: qframe1=.false., qframe2=.false., qcom1, qcom2, qcom3, qcom4
       integer, pointer :: priv(:)
! do work:
! extract info
       ncom1=cv%priv(i)%p(1); ncom2=cv%priv(i)%p(2)
       ncom3=cv%priv(i)%p(3); ncom4=cv%priv(i)%p(4)
! determine which points are atom selections and which are constant
       qcom1=ncom1.gt.0; qcom2=ncom2.gt.0;
       qcom3=ncom3.gt.0; qcom4=ncom4.gt.0;
!
       allocate(ind1(ncom1),x1(ncom1),y1(ncom1),z1(ncom1),m1(ncom1))
       allocate(ind2(ncom2),x2(ncom2),y2(ncom2),z2(ncom2),m2(ncom2))
       allocate(ind3(ncom3),x3(ncom3),y3(ncom3),z3(ncom3),m3(ncom3))
       allocate(ind4(ncom4),x4(ncom4),y4(ncom4),z4(ncom4),m4(ncom4))
! allocate(indpsf1(ncom1),indpsf2(ncom2),indpsf3(ncom2),
! & indpsf4(ncom4))
!c
! extract indices into the atom map
       ii=5; jj=ii+ncom1-1; ind1=cv%priv(i)%p(ii:jj)
       ii=jj+1; jj=ii+ncom2-1; ind2=cv%priv(i)%p(ii:jj)
       ii=jj+1; jj=ii+ncom3-1; ind3=cv%priv(i)%p(ii:jj)
       ii=jj+1; jj=ii+ncom4-1; ind4=cv%priv(i)%p(ii:jj)
!
       do j=1, ncom1;
        ind=cv%amap%i(ind1(j)) ! actual psf index
! indpsf1(j)=ind ;
                      x1(j)=x(ind) ; y1(j)=y(ind) ; z1(j)=z(ind) ;
                      m1(j)=mass(ind)
       enddo
       do j=1, ncom2;
        ind=cv%amap%i(ind2(j))
! indpsf2(j)=ind;
                      x2(j)=x(ind); y2(j)=y(ind) ; z2(j)=z(ind)
                      m2(j)=mass(ind)
       enddo
       do j=1, ncom3;
        ind=cv%amap%i(ind3(j))
! indpsf3(j)=ind;
                      x3(j)=x(ind); y3(j)=y(ind) ; z3(j)=z(ind)
                      m3(j)=mass(ind)
       enddo
       do j=1, ncom4;
        ind=cv%amap%i(ind4(j))
! indpsf4(j)=ind;
                      x4(j)=x(ind); y4(j)=y(ind) ; z4(j)=z(ind)
                      m4(j)=mass(ind)
       enddo
       if (calctheta) then
! extract frames
        fr1=cv%priv(i)%p(jj+1) ! frame that corresponds to u
        fr2=cv%priv(i)%p(jj+2) ! frame that corresponds to v
        qframe1=fr1.gt.0; qframe1=fr2.gt.0;
! normalize masses
        if (qcom1) then ; totm1=1d0/sum(m1); m1=m1*totm1; endif
        if (qcom2) then ; totm2=1d0/sum(m2); m2=m2*totm2; endif
        if (qcom3) then ; totm3=1d0/sum(m3); m3=m3*totm3; endif
        if (qcom4) then ; totm4=1d0/sum(m4); m4=m4*totm4; endif
        m1=m1*totm1; m2=m2*totm2; m3=m3*totm3; m4=m4*totm4;
!
        xcom1=0.; xcom2=0.; xcom3=0.; xcom4=0.;
        ycom1=0.; ycom2=0.; ycom3=0.; ycom4=0.;
        zcom1=0.; zcom2=0.; zcom3=0.; zcom4=0.;
        do j=1, ncom1;
         xcom1=xcom1+x1(j)*m1(j);
         ycom1=ycom1+y1(j)*m1(j);
         zcom1=zcom1+z1(j)*m1(j);
        enddo
        do j=1, ncom2;
         xcom2=xcom2+x2(j)*m2(j);
         ycom2=ycom2+y2(j)*m2(j);
         zcom2=zcom2+z2(j)*m2(j);
        enddo
        do j=1, ncom3;
         xcom3=xcom3+x3(j)*m3(j);
         ycom3=ycom3+y3(j)*m3(j);
         zcom3=zcom3+z3(j)*m3(j);
        enddo
        do j=1, ncom4;
         xcom4=xcom4+x4(j)*m4(j);
         ycom4=ycom4+y4(j)*m4(j);
         zcom4=zcom4+z4(j)*m4(j);
        enddo
! now compute vectors
! constant contribution
        u=cv%priv(i)%pr(4:6)-cv%priv(i)%pr(1:3)
        v=cv%priv(i)%pr(10:12)-cv%priv(i)%pr(7:9)
! atomic contribution
        u(1)=u(1)+xcom2-xcom1 ; v(1)=v(1)+xcom4-xcom3
        u(2)=u(2)+ycom2-ycom1 ; v(2)=v(2)+ycom4-ycom3
        u(3)=u(3)+zcom2-zcom1 ; v(3)=v(3)+zcom4-zcom3
!
!
! ux=u(1); uy=u(2); uz=u(3);
! vx=v(1); vy=v(2); vz=v(3);
!
! un=sqrt(ux*ux+uy*uy+uz*uz);
! vn=sqrt(vx*vx+vy*vy+vz*vz);
! if (un.eq.0d0) un=1d0
! if (vn.eq.0d0) vn=1d0
!
        if (qframe1) then
         call frames_calc(fr1,x,y,z,mass,deriv)
         A=frames%r(:,:,fr1); ! transformation matrix
        else
         A=Id3
        endif
!
        if (qframe2) then
         call frames_calc(fr2,x,y,z,mass,deriv)
         B=frames%r(:,:,fr2); ! transformation matrix
        else
         B=Id3
        endif
!
        u0=matmul(A,u); u0x=u0(1); u0y=u0(2); u0z=u0(3);
        v0=matmul(B,v); v0x=v0(1); v0y=v0(2); v0z=v0(3);
!
        u0n=sqrt(u0x*u0x+u0y*u0y+u0z*u0z);
        v0n=sqrt(v0x*v0x+v0y*v0y+v0z*v0z);

        costh=(u0x*v0x+u0y*v0y+u0z*v0z)/(u0n*v0n)
        sinth=sqrt(1d0-costh*costh)
        theta=atan2(sinth, costh) ! although acos would do just fine since we can not tell between theta & -theta
!
        cv%r(i,previnst)=cv%r(i,instant)
        cv%r(i,instant)=theta
!cccccccccccccccccc now compute derivative
        if (deriv) then
! compute derivatives (of the angle) with respect to vector components
         u0n=1d0/u0n; u0x=u0x*u0n; u0y=u0y*u0n; u0z=u0z*u0n;
         v0n=1d0/v0n; v0x=v0x*v0n; v0y=v0y*v0n; v0z=v0z*v0n;
!
         f=-1d0/(max(sinth, tol)); ! avoid singularity at zero
!
         fu(1)=f*u0n*(v0x-costh*u0x);
         fu(2)=f*u0n*(v0y-costh*u0y);
         fu(3)=f*u0n*(v0z-costh*u0z);
!
         fv(1)=f*v0n*(u0x-costh*v0x);
         fv(2)=f*v0n*(u0y-costh*v0y);
         fv(3)=f*v0n*(u0z-costh*v0z);
!
! initialize derivative arrays
! do j=1,cv%amap%last
         priv=>cv%priv(i)%amap_ptr ! this should copy the pointers statically; the pointers point to the _same_ data
         do jj=2,priv(1)+1 ! only a subset of indices needs to be considered
          j=priv(jj)
!
          do ii=1,2
          cv%gradx(i,j,ii)=0d0;cv%grady(i,j,ii)=0d0;cv%gradz(i,j,ii)=0d0
          enddo
         enddo
!
         if (qframe1) then ! contribution from frame derivatives
! extract frame indices (need to compute gradients w.r.t atoms that define the frame!)
          ncomf=frames%priv(fr1)%p(1);
          allocate(indf(ncomf));
          ii=2; jj=ii+ncomf-1; indf=frames%priv(fr1)%p(ii:jj)
!
! loop over all atoms in the frame definition
          do j=1, ncomf ! loop over frame atoms
           ind=indf(j) ! index into atom map
           w=1d0/sqrt(mass(cv%amap%i(ind))) !
! gradx
           dA=frames%gradrx(:,:,ind,fr1)
           dummy=dot_product(fu,matmul(dA,u))
           cv%gradx(i,ind,1)=dummy
           cv%gradx(i,ind,2)=dummy*w
! grady
           dA=frames%gradry(:,:,ind,fr1)
           dummy=dot_product(fu,matmul(dA,u))
           cv%grady(i,ind,1)=dummy
           cv%grady(i,ind,2)=dummy*w
! gradz
           dA=frames%gradrz(:,:,ind,fr1)
           dummy=dot_product(fu,matmul(dA,u))
           cv%gradz(i,ind,1)=dummy
           cv%gradz(i,ind,2)=dummy*w
          enddo
          deallocate(indf);
         endif ! frame
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
         if (qframe2) then ! contribution from frame derivatives
! extract frame indices (need to compute gradients w.r.t atoms that define the frame!)
          ncomf=frames%priv(fr2)%p(1);
          allocate(indf(ncomf));
          ii=2; jj=ii+ncomf-1; indf=frames%priv(fr2)%p(ii:jj)
!
! loop over all atoms in the frame definition
          do j=1, ncomf ! loop over frame atoms
           ind=indf(j) ! index into atom map
           w=1d0/sqrt(mass(cv%amap%i(ind))) !
! gradx
           dA=frames%gradrx(:,:,ind,fr2)
           dummy=dot_product(fv,matmul(dA,v))
           cv%gradx(i,ind,1)=cv%gradx(i,ind,1)+dummy
           cv%gradx(i,ind,2)=cv%gradx(i,ind,2)+dummy*w
! grady
           dA=frames%gradry(:,:,ind,fr2)
           dummy=dot_product(fv,matmul(dA,v))
           cv%grady(i,ind,1)=cv%grady(i,ind,1)+dummy
           cv%grady(i,ind,2)=cv%grady(i,ind,2)+dummy*w
! gradz
           dA=frames%gradrz(:,:,ind,fr2)
           dummy=dot_product(fv,matmul(dA,v))
           cv%gradz(i,ind,1)=cv%gradz(i,ind,1)+dummy
           cv%gradz(i,ind,2)=cv%gradz(i,ind,2)+dummy*w
          enddo
          deallocate(indf);
         endif ! frame
!ccccccccccccccccccccccc contribution from atomic coordinates
! loop over all atom groups
! COM1
! loop over all relevant indices in the map
         do j=1,ncom1
          ind=ind1(j)
          w=sqrt(totm1/m1(j)) ! for mass-weighting
!
          f=-fu(1)*m1(j)
          cv%gradx(i,ind,1)=cv%gradx(i,ind,1)+f ! derivative w.r.t x component of atom indpsf1(j)
          cv%gradx(i,ind,2)=cv%gradx(i,ind,2)+f*w
!
          f=-fu(2)*m1(j)
          cv%grady(i,ind,1)=cv%grady(i,ind,1)+f
          cv%grady(i,ind,2)=cv%grady(i,ind,1)+f*w
!
          f=-fu(3)*m1(j)
          cv%gradz(i,ind,1)=cv%gradz(i,ind,1)+f
          cv%gradz(i,ind,2)=cv%gradz(i,ind,1)+f*w
         enddo
! COM2
         do j=1,ncom2
          ind=ind2(j)
          w=sqrt(totm2/m2(j))
!
          f=fu(1)*m2(j)
          cv%gradx(i,ind,1)=cv%gradx(i,ind,1)+f
          cv%gradx(i,ind,2)=cv%gradx(i,ind,2)+f*w
!
          f=fu(2)*m2(j)
          cv%grady(i,ind,1)=cv%grady(i,ind,1)+f
          cv%grady(i,ind,2)=cv%grady(i,ind,2)+f*w
!
          f=fu(3)*m2(j)
          cv%gradz(i,ind,1)=cv%gradz(i,ind,1)+f
          cv%gradz(i,ind,2)=cv%gradz(i,ind,2)+f*w
        enddo
! COM3
         do j=1,ncom3
          ind=ind3(j)
          w=sqrt(totm3/m3(j))
!
          f=-fv(1)*m3(j)
          cv%gradx(i,ind,1)=cv%gradx(i,ind,1)+f
          cv%gradx(i,ind,2)=cv%gradx(i,ind,2)+f*w
!
          f=-fv(2)*m3(j)
          cv%grady(i,ind,1)=cv%grady(i,ind,1)+f
          cv%grady(i,ind,2)=cv%grady(i,ind,2)+f*w
!
          f=-fv(3)*m3(j)
          cv%gradz(i,ind,1)=cv%gradz(i,ind,1)+f
          cv%gradz(i,ind,2)=cv%gradz(i,ind,2)+f*w
         enddo
! COM4
         do j=1,ncom4
          ind=ind4(j)
          w=sqrt(totm4/m4(j))
!
          f=fv(1)*m4(j)
          cv%gradx(i,ind,1)=cv%gradx(i,ind,1)+f
          cv%gradx(i,ind,2)=cv%gradx(i,ind,2)+f*w
!
          f=fv(2)*m4(j)
          cv%grady(i,ind,1)=cv%grady(i,ind,1)+f
          cv%grady(i,ind,2)=cv%grady(i,ind,2)+f*w
!
          f=fv(3)*m4(j)
          cv%gradz(i,ind,1)=cv%gradz(i,ind,1)+f
          cv%gradz(i,ind,2)=cv%gradz(i,ind,2)+f*w
         enddo
! done !!!
        endif ! deriv
       else ! calctheta
        theta=cv%r(i,instant)
       endif
!
! NOTE that the forces calculated here are NOT acting on the CV, as in
! the evolution subroutine, but on the simulation atoms
       dummy=abs(cv%r(i,zcur))-theta ! we cannot distinguish between theta and -theta (because of reflectional symmetry)
!
       f=cv%k(i)*dummy
       cv%r(i,forces2)=f
       if (addforce) then
        if (present(fext)) f=fext ! override for planar sampling
!
! **** older code -- commented out 3/6/11
! if (qframe1.or.qframe2) then
! loop over all atom indices (easier to do this way, since the frame and CV atom indices may not be disjoint)
! need to figure out a clever way of only looping over the relevant atoms. SEE NEW WAY BELOW
! do j=1,cv%amap%last
! indpsf=cv%amap%i(j)
! fx(indpsf)=fx(indpsf)-f*cv%gradx(i,j,1)
! fy(indpsf)=fy(indpsf)-f*cv%grady(i,j,1)
! fz(indpsf)=fz(indpsf)-f*cv%gradz(i,j,1)
! enddo
! else
! code from dihe_com: as in the other cases, atom lists cannot overlap ; will deal with this `bug` later
! loop over all groups
! COM1
! loop over all relevant indices in the map
! note: fx are actually components of grad V, not -grad V!
! do j=1,ncom1
! ind=ind1(j)
! indpsf=indpsf1(j) ! index into psf
! fx(indpsf)=fx(indpsf)-f*cv%gradx(i,ind,1)
! fy(indpsf)=fy(indpsf)-f*cv%grady(i,ind,1)
! fz(indpsf)=fz(indpsf)-f*cv%gradz(i,ind,1)
! enddo
! COM2
! do j=1,ncom2
! ind=ind2(j)
! indpsf=indpsf2(j)
! fx(indpsf)=fx(indpsf)-f*cv%gradx(i,ind,1)
! fy(indpsf)=fy(indpsf)-f*cv%grady(i,ind,1)
! fz(indpsf)=fz(indpsf)-f*cv%gradz(i,ind,1)
! enddo
! COM3
! do j=1,ncom3
! ind=ind3(j)
! indpsf=indpsf3(j)
! fx(indpsf)=fx(indpsf)-f*cv%gradx(i,ind,1)
! fy(indpsf)=fy(indpsf)-f*cv%grady(i,ind,1)
! fz(indpsf)=fz(indpsf)-f*cv%gradz(i,ind,1)
! enddo
! COM4
! do j=1,ncom4
! ind=ind4(j)
! indpsf=indpsf4(j)
! fx(indpsf)=fx(indpsf)-f*cv%gradx(i,ind,1)
! fy(indpsf)=fy(indpsf)-f*cv%grady(i,ind,1)
! fz(indpsf)=fz(indpsf)-f*cv%gradz(i,ind,1)
! enddo
! endif ! qframe
!
! more efficient and conceptually correct way:
!
        priv=>cv%priv(i)%amap_ptr
        do ii=2, priv(1)+1
         jj=priv(ii)
         j=cv%amap%i(jj) ! psf index
         fx(j)=fx(j)-f*cv%gradx(i,jj,1)
         fy(j)=fy(j)-f*cv%grady(i,jj,1)
         fz(j)=fz(j)-f*cv%gradz(i,jj,1)
        enddo
!
       endif ! addforce
! free memory
       deallocate(ind1, ind2, ind3, ind4, &
! & indpsf1, indpsf2, indpsf3, indpsf4
     & x1, x2, x3, x4, &
     & y1, y2, y3, y4, &
     & z1, z2, z3, z4, &
     & m1, m2, m3, m4)
       end subroutine cv_anglvec_calc
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine cv_anglvec_list(i)
       use stream
       use multicom_aux;
       use mpi
       use chutil, only : atomid
!
 character(len=132)::info(21)=(/'','','','','','','','','','','','','','','','','','','','',''/);! output buffer
!
       integer :: i
!
! variables for COM groups
       integer :: ncom1, ncom2, ncom3, ncom4, iatom
       integer, pointer, dimension(:) :: ind1, ind2, ind3, ind4
!
       character(len=8) :: sid, rid, ren, ac
       integer :: f1, f2, type
       integer :: j, k, ii, jj
       real(chm_real), dimension(4,3) :: p ! points that define vectors
!
       character(len=len("CV_ANGLVEC_LIST>") ),parameter::whoami="CV_ANGLVEC_LIST>";!macro
!
!
       if (MPI_COMM_STRNG.eq.MPI_COMM_NULL) return ! only replica heads stay
!
! check type just in case
       type=cv%type(i)
       if (type.ne.anglvec) then
        call wrndie(0,whoami,trim(' WRONG CV TYPE RECEIVED.'))
       endif
!
       if (ME_STRNG.eq.0) then
!
        ncom1=cv%priv(i)%p(1);
        ncom2=cv%priv(i)%p(2);
        ncom3=cv%priv(i)%p(3);
        ncom4=cv%priv(i)%p(4);
        allocate(ind1(ncom1),ind2(ncom2),ind3(ncom3),ind4(ncom4))

! extract indices into the atom map
        ii=5; jj=ii+ncom1-1; ind1=cv%priv(i)%p(ii:jj)
        ii=jj+1; jj=ii+ncom2-1; ind2=cv%priv(i)%p(ii:jj)
        ii=jj+1; jj=ii+ncom3-1; ind3=cv%priv(i)%p(ii:jj)
        ii=jj+1; jj=ii+ncom4-1; ind4=cv%priv(i)%p(ii:jj)
! extract frames
        f1=cv%priv(i)%p(jj+1)
        f2=cv%priv(i)%p(jj+2)
! extract constant points
        do j=1,4; do k=1,3
                    p(j,k)=cv%priv(i)%pr(k+(j-1)*3)
        enddo; enddo

! output:
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! first vector:
        write(info,'(A)') '\t ANGLE BETWEEN VECTORS, VECTOR 1' ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
        write(info,'(A,I5)') '\t COORDINATE FRAME: ',f1 ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
! first point:
        if (ncom1.eq.0) then ! constant point
         write(info,'(A,3(F11.5,", "))') '\t POINT 1, CONSTANT: ',p(1,:) ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
        else
         write(info,'(A)') '\t POINT 1, COM OF ATOM GROUP:' ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
         do j=1, ncom1;
          iatom=cv%amap%i(ind1(j)) ! actual psf index
          call atomid(iatom, sid, rid, ren, ac)
          write(info,667) '\t',j, iatom, sid, rid, ren, ac ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
         enddo
        endif
! second point:
        if (ncom2.eq.0) then ! constant point
         write(info,'(A,3(F11.5,", "))') '\t POINT 2, CONSTANT: ',p(2,:) ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
        else
         write(info,'(A)') '\t POINT 2, COM OF ATOM GROUP:' ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
         do j=1, ncom2;
          iatom=cv%amap%i(ind2(j)) ! actual psf index
          call atomid(iatom, sid, rid, ren, ac)
          write(info,667) '\t',j, iatom, sid, rid, ren, ac ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
         enddo
        endif
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! second vector:
        write(info,'(A)') '\t ANGLE BETWEEN VECTORS, VECTOR 2' ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
        write(info,'(A,I5)') '\t COORDINATE FRAME: ',f2 ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
! third point:
        if (ncom3.eq.0) then ! constant point
         write(info,'(A,3(F11.5,", "))') '\t POINT 3, CONSTANT: ',p(3,:) ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
        else
         write(info,'(A)') '\t POINT 3, COM OF ATOM GROUP:' ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
         do j=1, ncom3;
          iatom=cv%amap%i(ind3(j)) ! actual psf index
          call atomid(iatom, sid, rid, ren, ac)
          write(info,667) '\t',j, iatom, sid, rid, ren, ac ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
         enddo
        endif
! fourth point:
        if (ncom4.eq.0) then ! constant point
         write(info,'(A,3(F11.5,", "))') '\t POINT 4, CONSTANT: ',p(4,:) ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
        else
         write(info,'(A)') '\t POINT 4, COM OF ATOM GROUP:' ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
         do j=1, ncom4;
          iatom=cv%amap%i(ind4(j)) ! actual psf index
          call atomid(iatom, sid, rid, ren, ac)
          write(info,667) '\t',j, iatom, sid, rid, ren, ac ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
         enddo
        endif
       endif
!
 667 format(A,2I8,' ',4A)
!
       end subroutine cv_anglvec_list
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

#endif /* automatically protect all code */
      end module cv_anglvec
!
