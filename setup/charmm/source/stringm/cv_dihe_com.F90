! ********************************************************************************!
! Preprocessed by GNU CPP !
! ********************************************************************************!
!
! CV_DIHE_COM.MOD
!
! ROUTINES FOR COLLECTIVE VARIABLE `DIHEDRAL ANGLE` BETWEEN CENTERS OF MASS OF ATOMS
      module cv_dihe_com
#if (KEY_STRINGM==1) /*  automatically protect all code */
!
!
      use cv_common
      use cv_types
      use ivector ! vector class for storing atom lists
      use ivector_list
!
      implicit none
      private
      ! subroutines
      public cv_dihe_com_add
      public cv_dihe_com_calc
      public cv_dihe_com_list
      public cv_dihe_com_grad_dot_dr
      public cv_dihe_com_dcv_dot_grad
!
      contains
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       function cv_dihe_com_add(atom_list,k,gamma,weight) ! note: i is the atom index in the PSF
       use stream
       real(chm_real) :: k, gamma, weight
       type (int_vector), dimension(4) :: atom_list
! locals
       integer :: i, j, l, m, ind, num_int, ncom(4)
       logical :: found, cv_dihe_com_add
       type (int_vector) :: unique_amap_ptr ! contains a unique list of atom map pointers (to speed up gradient computation, etc)
!
       character(len=len("CV_DIHE_COM_ADD>") ),parameter::whoami="CV_DIHE_COM_ADD>";!macro
!
! check for duplicate CV (exact identical entry only)
       found=.false.
       do l=1, cv%num_cv
        if (cv%type(l).eq.dihe_com) then
         found=.true.
         do i=1,4
          ncom(i)=atom_list(i)%last
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
        endif
        if (found) exit
       enddo
!
       if (.not.found) then ! (if found -- do nothing)
        l=cv_common_add(k,gamma,weight,dihe_com) ! get a new cv index
        if (l.gt.0) then
! allocate private data
! space needed:
         do i=1,4 ; ncom(i)=atom_list(i)%last ; enddo
         num_int = 4 + sum(ncom) ! number of ints needed for storage
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
!
         m=unique_amap_ptr%last ! number of unique atoms this cv depends on
         allocate(cv%priv(l)%amap_ptr(m+1)) ! add one to include length of list
         cv%priv(l)%amap_ptr(1)=m;
         cv%priv(l)%amap_ptr(2:m+1)=unique_amap_ptr%i(1:m)
         call int_vector_done(unique_amap_ptr)
!
         cv_dihe_com_add=.true.
        else ! out of bounds
         call wrndie(0,whoami,trim(' ERROR ADDING DIHE_COM CV. NOTHING DONE.'))
         cv_dihe_com_add=.false.
        endif
       else ! found
         call wrndie(0,whoami,trim(' DIHE_COM CV ALREADY PRESENT. NOTHING DONE.'))
         cv_dihe_com_add=.false.
       endif
       end function cv_dihe_com_add
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine cv_dihe_com_calc(i,x,y,z,mass,fx,fy,fz, &
     & calctheta,deriv,addforce,fext)
       use consta
!
       real(chm_real) :: x(:), y(:), z(:), &
     & fx(:), fy(:), fz(:), mass(:)
       real(chm_real), optional :: fext ! `external` force for planar dynamics (see smcv_addforce in smcv_master)
       integer :: i
       logical :: calctheta ! whether or not to calculate theta(x); if not, use theta=cv%r(i,instant)
                            ! note that if calctheta=.false., we do not calculate the derivatives!
       logical :: deriv ! whether or not to calculate derivatives w.r.t. x
       logical :: addforce ! whether or not to add forces on simulation atoms
!
! real(chm_real), parameter :: pi=3.14159265358979 ! hardwired for angles
! real(chm_real), parameter :: twopi=2d0*pi
       real(chm_real) :: dummy
!
       integer, pointer, dimension(:) :: ind1, ind2, ind3, ind4
! & ,indpsf1, indpsf2, indpsf3, indpsf4
       real(chm_real), pointer, dimension(:) :: x1, x2, x3, x4
       real(chm_real), pointer, dimension(:) :: y1, y2, y3, y4
       real(chm_real), pointer, dimension(:) :: z1, z2, z3, z4
       real(chm_real), pointer, dimension(:) :: m1, m2, m3, m4
!
       integer :: ncom1, ncom2, ncom3, ncom4, ind! , indpsf
       integer :: j, ii, jj
! variables for cv and derivative calculations
       real(chm_real) :: theta, costh, sinth, &
     & xcom1, xcom2, xcom3, xcom4, ycom1, ycom2, ycom3, ycom4,&
     & zcom1, zcom2, zcom3, zcom4, &
     & dx12, dy12, dz12, dx23, dy23, dz23, dx34, dy34, dz34, &
     & vx, vy, vz, vn, ux, uy, uz, un, wx, wy, wz, wn, &
     & dcosdux, dcosduy, dcosduz, dcosdvx, dcosdvy, dcosdvz, &
     & dsindux, dsinduy, dsinduz, dsindwx, dsindwy, dsindwz, &
     & fx12, fy12, fz12, fx23, fy23, fz23, fx34, fy34, fz34, &
     & f1x, f1y, f1z, f2x, f2y, f2z, f3x, f3y, f3z, &
     & f4x, f4y, f4z, f, w, totm1, totm2, totm3, totm4
       integer, pointer :: priv(:)
! do work:
! look up atom index in the private array of CV i; obtain the PSF index from amap
       ncom1=cv%priv(i)%p(1)
       ncom2=cv%priv(i)%p(2)
       ncom3=cv%priv(i)%p(3)
       ncom4=cv%priv(i)%p(4)
!
       allocate(ind1(ncom1),x1(ncom1),y1(ncom1),z1(ncom1),m1(ncom1))
       allocate(ind2(ncom2),x2(ncom2),y2(ncom2),z2(ncom2),m2(ncom2))
       allocate(ind3(ncom3),x3(ncom3),y3(ncom3),z3(ncom3),m3(ncom3))
       allocate(ind4(ncom4),x4(ncom4),y4(ncom4),z4(ncom4),m4(ncom4))
! allocate(indpsf1(ncom1),indpsf2(ncom2),indpsf3(ncom2),
! & indpsf4(ncom4))
!
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
! now we have all the coordinates needed to compute dihedral and its derivative :
       if (calctheta) then
!
        totm1=1d0/sum(m1); totm2=1d0/sum(m2);
        totm3=1d0/sum(m3); totm4=1d0/sum(m4);
! normalize masses
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
! compute centers of mass
! xcom1=xcom1*totm1; ycom1=ycom1*totm1; zcom1=zcom1*totm1;
! xcom2=xcom2*totm2; ycom2=ycom2*totm2; zcom2=zcom2*totm2;
! xcom3=xcom3*totm3; ycom3=ycom3*totm3; zcom3=zcom3*totm3;
! xcom4=xcom4*totm4; ycom4=ycom4*totm4; zcom4=zcom4*totm4;
!
        dx12=xcom2-xcom1; dy12=ycom2-ycom1; dz12=zcom2-zcom1;
        dx23=xcom3-xcom2; dy23=ycom3-ycom2; dz23=zcom3-zcom2;
        dx34=xcom4-xcom3; dy34=ycom4-ycom3; dz34=zcom4-zcom3;
!
! u= - d34 x d23 = d23 x d34
! v= - d23 x d12 = d12 x d23
! w= d23 x v
!
! parts of the following code adopted from NAMD2 ComputeDihedrals.C
!
! note:
! costh = [ (d34 x d32) . (d23 x d21) / |(d34 x d23)| |(d23 x d12)| ] =
! = acos [ (u . v) / |u| |v| ] (definition of u and v)
        ux=dy23*dz34-dy34*dz23;
        uy=dz23*dx34-dz34*dx23;
        uz=dx23*dy34-dx34*dy23;
        un=sqrt(ux*ux+uy*uy+uz*uz);
!
        vx=dy12*dz23-dy23*dz12;
        vy=dz12*dx23-dz23*dx12;
        vz=dx12*dy23-dx23*dy12;
        vn=sqrt(vx*vx+vy*vy+vz*vz);
!
        wx=dy23*vz-vy*dz23;
        wy=dz23*vx-vz*dx23;
        wz=dx23*vy-vx*dy23;
        wn=sqrt(wx*wx+wy*wy+wz*wz);
!
        if (un.eq.0d0) un=1d0
        if (vn.eq.0d0) vn=1d0
        if (wn.eq.0d0) wn=1d0
        costh=(ux*vx+uy*vy+uz*vz)/(un*vn)
        sinth=(wx*ux+wy*uy+wz*uz)/(wn*un)
        theta=atan2(sinth, costh)
!
        cv%r(i,previnst)=cv%r(i,instant)
        cv%r(i,instant)=theta
!cccccccccccccccccc now compute derivative
        if (deriv) then
         un=1d0/un
         ux=ux*un; uy=uy*un; uz=uz*un;
! decide which form to use: 1/sin(theta) is singular for theta=n*pi
         if (abs(sinth).gt.0.1) then ! use simpler version (1/sin defined)
          vn=1d0/vn
          vx=vx*vn; vy=vy*vn; vz=vz*vn;
          dcosdux=un*(vx-costh*ux);
          dcosduy=un*(vy-costh*uy);
          dcosduz=un*(vz-costh*uz);
!
          dcosdvx=vn*(ux-costh*vx);
          dcosdvy=vn*(uy-costh*vy);
          dcosdvz=vn*(uz-costh*vz);
! calculate derivative w.r.t. `bonds` 12 23 34;
          f=-1d0/sinth;
          fx12 = f*(dy23*dcosdvz-dz23*dcosdvy);
          fy12 = f*(dz23*dcosdvx-dx23*dcosdvz);
          fz12 = f*(dx23*dcosdvy-dy23*dcosdvx);
!
          fx23 = f*(dy34*dcosduz-dz34*dcosduy- &
     & (dy12*dcosdvz-dz12*dcosdvy) )
          fy23 = f*(dz34*dcosdux-dx34*dcosduz- &
     & (dz12*dcosdvx-dx12*dcosdvz) )
          fz23 = f*(dx34*dcosduy-dy34*dcosdux- &
     & (dx12*dcosdvy-dy12*dcosdvx) )
!
          fx34 = f*(-dy23*dcosduz+dz23*dcosduy);
          fy34 = f*(-dz23*dcosdux+dx23*dcosduz);
          fz34 = f*(-dx23*dcosduy+dy23*dcosdux);
!
! write(600+whoiam,*) fx12, fy12, fz12
! write(600+whoiam,*) fx23, fy23, fz23
! write(600+whoiam,*) fx34, fy34, fz34
         else ! use less efficient form (1/cos defined)
          wn=1d0/wn
          wx=wx*wn; wy=wy*wn; wz=wz*wn;
!
          dsindwx=wn*(ux-sinth*wx);
          dsindwy=wn*(uy-sinth*wy);
          dsindwz=wn*(uz-sinth*wz);
!
          dsindux=un*(wx-sinth*ux);
          dsinduy=un*(wy-sinth*uy);
          dsinduz=un*(wz-sinth*uz);

          f=1d0/costh;
          fx12=f*(dsindwx*(dy23*dy23+dz23*dz23) &
     & -dsindwy*(dx23*dy23) &
     & -dsindwz*(dx23*dz23))
          fy12=f*(-dsindwx*(dx23*dy23) &
     & + dsindwy*(dx23*dx23+dz23*dz23) &
     & - dsindwz*(dy23*dz23))
          fz12=f*(-dsindwx*(dx23*dz23) &
     & - dsindwy*(dy23*dz23) &
     & + dsindwz*(dy23*dy23+dx23*dx23))
!
          fx23=f*(dy34*dsinduz-dz34*dsinduy &
     & -dsindwx*(dz23*dz12+dy23*dy12) &
     & +dsindwy*(2d0*dy12*dx23-dy23*dx12) &
     & +dsindwz*(2d0*dz12*dx23-dz23*dx12))
          fy23=f*(dz34*dsindux-dx34*dsinduz &
     & +dsindwx*(2d0*dx12*dy23-dx23*dy12) &
     & -dsindwy*(dx12*dx23+dz12*dz23) &
     & +dsindwz*(2d0*dz12*dy23-dz23*dy12))
          fz23=f*(dx34*dsinduy-dy34*dsindux &
     & +dsindwx*(2d0*dx12*dz23-dx23*dz12) &
     & +dsindwy*(2d0*dy12*dz23-dz12*dy23) &
     & -dsindwz*(dx12*dx23+dy12*dy23))
!
          fx34=f*(dsinduy*dz23-dsinduz*dy23)
          fy34=f*(dsinduz*dx23-dsindux*dz23)
          fz34=f*(dsindux*dy23-dsinduy*dx23)
! aardvark
! write(600+whoiam,*) '****************'
! write(600+whoiam,*) fx12, fy12, fz12
! write(600+whoiam,*) fx23, fy23, fz23
! write(600+whoiam,*) fx34, fy34, fz34
         endif ! sinth

!ccccccccc test fd -- PASSED
! f=0.00001d0
! dz34=dz34+f
!
! ux=dy23*dz34-dy34*dz23;
! uy=dz23*dx34-dz34*dx23;
! uz=dx23*dy34-dx34*dy23;
! un=sqrt(ux*ux+uy*uy+uz*uz);
!
! vx=dy12*dz23-dy23*dz12;
! vy=dz12*dx23-dz23*dx12;
! vz=dx12*dy23-dx23*dy12;
! vn=sqrt(vx*vx+vy*vy+vz*vz);
!
! wx=dy23*vz-vy*dz23;
! wy=dz23*vx-vz*dx23;
! wz=dx23*vy-vx*dy23;
! wn=sqrt(wx*wx+wy*wy+wz*wz);
!
! if (un.eq.0d0) un=1d0
! if (vn.eq.0d0) vn=1d0
! if (wn.eq.0d0) wn=1d0
! costh=(ux*vx+uy*vy+uz*vz)/(un*vn)
! sinth=(wx*ux+wy*uy+wz*uz)/(wn*un)
!
! write(600+whoiam,*) '****************'
! write(600+whoiam,*) theta, atan2(sinth, costh),
! & (atan2(sinth, costh)-theta)/f, f
!
! compute derivatives w.r.t COMs:
         f1x=-fx12; f1y=-fy12; f1z=-fz12;
         f2x= fx12-fx23; f2y= fy12-fy23; f2z= fz12-fz23;
         f3x= fx23-fx34; f3y= fy23-fy34; f3z= fz23-fz34;
         f4x= fx34; f4y= fy34; f4z= fz34;
! compute derivatives w.r.t. COM components
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
! loop over all groups
! COM1
! loop over all relevant indices in the map
         do j=1,ncom1
          ind=ind1(j)
          w=sqrt(totm1/m1(j)) ! for mass-weighting
!
          f=f1x*m1(j)
          cv%gradx(i,ind,1)=f ! derivative w.r.t x component of atom indpsf1(j)
          cv%gradx(i,ind,2)=f*w
!
          f=f1y*m1(j)
          cv%grady(i,ind,1)=f
          cv%grady(i,ind,2)=f*w
!
          f=f1z*m1(j)
          cv%gradz(i,ind,1)=f
          cv%gradz(i,ind,2)=f*w
         enddo
! COM2
         do j=1,ncom2
          ind=ind2(j)
          w=sqrt(totm2/m2(j))
!
          f=f2x*m2(j)
          cv%gradx(i,ind,1)=cv%gradx(i,ind,1)+f
          cv%gradx(i,ind,2)=cv%gradx(i,ind,2)+f*w
!
          f=f2y*m2(j)
          cv%grady(i,ind,1)=cv%grady(i,ind,1)+f
          cv%grady(i,ind,2)=cv%grady(i,ind,2)+f*w
!
          f=f2z*m2(j)
          cv%gradz(i,ind,1)=cv%gradz(i,ind,1)+f
          cv%gradz(i,ind,2)=cv%gradz(i,ind,2)+f*w
         enddo
! COM3
         do j=1,ncom3
          ind=ind3(j)
          w=sqrt(totm3/m3(j))
!
          f=f3x*m3(j)
          cv%gradx(i,ind,1)=cv%gradx(i,ind,1)+f
          cv%gradx(i,ind,2)=cv%gradx(i,ind,2)+f*w
!
          f=f3y*m3(j)
          cv%grady(i,ind,1)=cv%grady(i,ind,1)+f
          cv%grady(i,ind,2)=cv%grady(i,ind,2)+f*w
!
          f=f3z*m3(j)
          cv%gradz(i,ind,1)=cv%gradz(i,ind,1)+f
          cv%gradz(i,ind,2)=cv%gradz(i,ind,2)+f*w
         enddo
! COM4
         do j=1,ncom4
          ind=ind4(j)
          w=sqrt(totm4/m4(j))
!
          f=f4x*m4(j)
          cv%gradx(i,ind,1)=cv%gradx(i,ind,1)+f
          cv%gradx(i,ind,2)=cv%gradx(i,ind,2)+f*w
!
          f=f4y*m4(j)
          cv%grady(i,ind,1)=cv%grady(i,ind,1)+f
          cv%grady(i,ind,2)=cv%grady(i,ind,2)+f*w
!
          f=f4z*m4(j)
          cv%gradz(i,ind,1)=cv%gradz(i,ind,1)+f
          cv%gradz(i,ind,2)=cv%gradz(i,ind,2)+f*w
         enddo
        endif ! deriv
       else
        theta=cv%r(i,instant) ! assume valid theta has been computed in cv%r(i,instant)
       endif ! calctheta
!
! NOTE that the forces calculated here are NOT acting on the CV, as in
! the evolution subroutine, but on the simulation atoms
       dummy=modulo(cv%r(i,zcur)-theta,TWOPI) ! zcur contains reference coords (combination of main+comp)
       if (dummy.gt.PI) dummy=dummy-TWOPI ! wrap angle around
!
       f=cv%k(i)*dummy ! note that f is not divided by 1/sqrt(mass)
       cv%r(i,forces2)=f
       if (addforce) then
        if (present(fext)) f=fext ! override for planar sampling
! loop over all groups
! COM1
! loop over all relevant indices in the map
! NOTE: I am assuming that the sets are disjoint; otherwise some forces are added more than once; need to rewrite this! But how? FIXED BELOW
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
! indpsf=indpsf2(j) ! index into psf
! fx(indpsf)=fx(indpsf)-f*cv%gradx(i,ind,1)
! fy(indpsf)=fy(indpsf)-f*cv%grady(i,ind,1)
! fz(indpsf)=fz(indpsf)-f*cv%gradz(i,ind,1)
! enddo
! COM3
! do j=1,ncom3
! ind=ind3(j)
! indpsf=indpsf3(j) ! index into psf
! fx(indpsf)=fx(indpsf)-f*cv%gradx(i,ind,1)
! fy(indpsf)=fy(indpsf)-f*cv%grady(i,ind,1)
! fz(indpsf)=fz(indpsf)-f*cv%gradz(i,ind,1)
! enddo
! COM4
! do j=1,ncom4
! ind=ind4(j)
! indpsf=indpsf4(j) ! index into psf
! fx(indpsf)=fx(indpsf)-f*cv%gradx(i,ind,1)
! fy(indpsf)=fy(indpsf)-f*cv%grady(i,ind,1)
! fz(indpsf)=fz(indpsf)-f*cv%gradz(i,ind,1)
! enddo
! more efficient and conceptually correct way:
        priv=>cv%priv(i)%amap_ptr
        do ii=2, priv(1)+1
         jj=priv(ii)
         j=cv%amap%i(jj) ! psf index
         fx(j)=fx(j)-f*cv%gradx(i,jj,1)
         fy(j)=fy(j)-f*cv%grady(i,jj,1)
         fz(j)=fz(j)-f*cv%gradz(i,jj,1)
        enddo
       endif ! addforce
! free memory
       deallocate(ind1, ind2, ind3, ind4, &
! & indpsf1, indpsf2, indpsf3, indpsf4,
     & x1, x2, x3, x4, &
     & y1, y2, y3, y4, &
     & z1, z2, z3, z4, &
     & m1, m2, m3, m4)
       end subroutine cv_dihe_com_calc
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       real(chm_real) function cv_dihe_com_grad_dot_dr(i,dx,dy,dz)
!
       real(chm_real) :: dx(:), dy(:), dz(:)
       integer :: i
       real(chm_real) :: dummy
!
       integer, pointer, dimension(:) :: ind1, ind2, ind3, ind4
!
       integer :: ncom1, ncom2, ncom3, ncom4, ind, indpsf
       integer :: j, ii, jj
! do work:
! look up atom index in the private array of CV i; obtain the PSF index from amap
       ncom1=cv%priv(i)%p(1)
       ncom2=cv%priv(i)%p(2)
       ncom3=cv%priv(i)%p(3)
       ncom4=cv%priv(i)%p(4)
!
       allocate(ind1(ncom1), ind2(ncom2), ind3(ncom3), ind4(ncom4))
!
! extract indices into the atom map
       ii=5; jj=ii+ncom1-1; ind1=cv%priv(i)%p(ii:jj)
       ii=jj+1; jj=ii+ncom2-1; ind2=cv%priv(i)%p(ii:jj)
       ii=jj+1; jj=ii+ncom3-1; ind3=cv%priv(i)%p(ii:jj)
       ii=jj+1; jj=ii+ncom4-1; ind4=cv%priv(i)%p(ii:jj)
!
! loop over all groups
       dummy=0d0
! COM1
! loop over all relevant indices in the map
! assume non-overlapping selections
       do j=1,ncom1
         ind=ind1(j)
         indpsf=cv%amap%i(ind) ! actual psf index
         dummy=dummy+dx(indpsf)*cv%gradx(i,ind,1) &
     & +dy(indpsf)*cv%grady(i,ind,1) &
     & +dz(indpsf)*cv%gradz(i,ind,1)
       enddo
! COM2
       do j=1,ncom2
         ind=ind2(j)
         indpsf=cv%amap%i(ind) ! actual psf index
         dummy=dummy+dx(indpsf)*cv%gradx(i,ind,1) &
     & +dy(indpsf)*cv%grady(i,ind,1) &
     & +dz(indpsf)*cv%gradz(i,ind,1)
       enddo
! COM3
       do j=1,ncom3
         ind=ind3(j)
         indpsf=cv%amap%i(ind) ! actual psf index
         dummy=dummy+dx(indpsf)*cv%gradx(i,ind,1) &
     & +dy(indpsf)*cv%grady(i,ind,1) &
     & +dz(indpsf)*cv%gradz(i,ind,1)
       enddo
! COM4
       do j=1,ncom4
         ind=ind4(j)
         indpsf=cv%amap%i(ind) ! actual psf index
         dummy=dummy+dx(indpsf)*cv%gradx(i,ind,1) &
     & +dy(indpsf)*cv%grady(i,ind,1) &
     & +dz(indpsf)*cv%gradz(i,ind,1)
       enddo
! free memory
       deallocate(ind1, ind2, ind3, ind4)
       cv_dihe_com_grad_dot_dr=dummy
!
       end function cv_dihe_com_grad_dot_dr
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine cv_dihe_com_dcv_dot_grad(i,dx,dy,dz,dcv)
!
       real(chm_real) :: dx(:), dy(:), dz(:)
       integer :: i
       real(chm_real) :: dcv
!
       integer, pointer, dimension(:) :: ind1, ind2, ind3, ind4
!
       integer :: ncom1, ncom2, ncom3, ncom4, ind, indpsf
       integer :: j, ii, jj
! do work:
! look up atom index in the private array of CV i; obtain the PSF index from amap
       ncom1=cv%priv(i)%p(1)
       ncom2=cv%priv(i)%p(2)
       ncom3=cv%priv(i)%p(3)
       ncom4=cv%priv(i)%p(4)
!
       allocate(ind1(ncom1), ind2(ncom2), ind3(ncom3), ind4(ncom4))
!
! extract indices into the atom map
       ii=5; jj=ii+ncom1-1; ind1=cv%priv(i)%p(ii:jj)
       ii=jj+1; jj=ii+ncom2-1; ind2=cv%priv(i)%p(ii:jj)
       ii=jj+1; jj=ii+ncom3-1; ind3=cv%priv(i)%p(ii:jj)
       ii=jj+1; jj=ii+ncom4-1; ind4=cv%priv(i)%p(ii:jj)
!
! loop over all groups
! COM1
! loop over all relevant indices in the map
! assume non-overlapping selections
       do j=1,ncom1
         ind=ind1(j)
         indpsf=cv%amap%i(ind) ! actual psf index
         dx(indpsf)=dx(indpsf)+dcv*cv%gradx(i,ind,1)
         dy(indpsf)=dy(indpsf)+dcv*cv%grady(i,ind,1)
         dz(indpsf)=dz(indpsf)+dcv*cv%gradz(i,ind,1)
       enddo
! COM2
       do j=1,ncom2
         ind=ind2(j)
         indpsf=cv%amap%i(ind) ! actual psf index
         dx(indpsf)=dx(indpsf)+dcv*cv%gradx(i,ind,1)
         dy(indpsf)=dy(indpsf)+dcv*cv%grady(i,ind,1)
         dz(indpsf)=dz(indpsf)+dcv*cv%gradz(i,ind,1)
       enddo
! COM3
       do j=1,ncom3
         ind=ind3(j)
         indpsf=cv%amap%i(ind) ! actual psf index
         dx(indpsf)=dx(indpsf)+dcv*cv%gradx(i,ind,1)
         dy(indpsf)=dy(indpsf)+dcv*cv%grady(i,ind,1)
         dz(indpsf)=dz(indpsf)+dcv*cv%gradz(i,ind,1)
       enddo
! COM4
       do j=1,ncom4
         ind=ind4(j)
         indpsf=cv%amap%i(ind) ! actual psf index
         dx(indpsf)=dx(indpsf)+dcv*cv%gradx(i,ind,1)
         dy(indpsf)=dy(indpsf)+dcv*cv%grady(i,ind,1)
         dz(indpsf)=dz(indpsf)+dcv*cv%gradz(i,ind,1)
       enddo
! free memory
       deallocate(ind1, ind2, ind3, ind4)
!
       end subroutine cv_dihe_com_dcv_dot_grad
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine cv_dihe_com_list(i)
       use stream
       use multicom_aux;
       use mpi
       use chutil, only : atomid
!
 character(len=132)::info(21)=(/'','','','','','','','','','','','','','','','','','','','',''/);! output buffer
!
       integer :: i, j, type, ii, jj, iatom
       character(len=8) :: sid, rid, ren, ac
!
       integer :: ncom1, ncom2, ncom3, ncom4
       integer, pointer, dimension(:) :: ind1, ind2, ind3, ind4
       character(len=len("CV_DIHE_COM_LIST>") ),parameter::whoami="CV_DIHE_COM_LIST>";!macro
!
       if (MPI_COMM_STRNG.eq.MPI_COMM_NULL) return ! only replica heads stay
!
! check type just in case
       type=cv%type(i)
       if (type.ne.dihe_com) then
        call wrndie(0,whoami,trim(' WRONG CV TYPE RECEIVED.'))
       endif
!
       if (ME_STRNG.eq.0) then
        ncom1=cv%priv(i)%p(1)
        ncom2=cv%priv(i)%p(2)
        ncom3=cv%priv(i)%p(3)
        ncom4=cv%priv(i)%p(4)
!
        allocate(ind1(ncom1),ind2(ncom2),ind3(ncom3),ind4(ncom4))
!
! extract indices into the atom map
        ii=5; jj=ii+ncom1-1; ind1=cv%priv(i)%p(ii:jj)
        ii=jj+1; jj=ii+ncom2-1; ind2=cv%priv(i)%p(ii:jj)
        ii=jj+1; jj=ii+ncom3-1; ind3=cv%priv(i)%p(ii:jj)
        ii=jj+1; jj=ii+ncom4-1; ind4=cv%priv(i)%p(ii:jj)
!
        write(info,'(A)') '\t DIHEDRAL-COM, GROUP 1' ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
        do j=1, ncom1;
         iatom=cv%amap%i(ind1(j)) ! actual psf index
         call atomid(iatom, sid, rid, ren, ac)
         write(info,667) '\t',j, iatom, sid, rid, ren, ac ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
        enddo
        write(info,'(A)') '\t DIHEDRAL-COM, GROUP 2' ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
        do j=1, ncom2;
         iatom=cv%amap%i(ind2(j))
         call atomid(iatom, sid, rid, ren, ac)
         write(info,667) '\t',j, iatom, sid, rid, ren, ac ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
        enddo
        write(info,'(A)') '\t DIHEDRAL-COM, GROUP 3' ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
        do j=1, ncom3;
         iatom=cv%amap%i(ind3(j))
         call atomid(iatom, sid, rid, ren, ac)
         write(info,667) '\t',j, iatom, sid, rid, ren, ac ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
        enddo
        write(info,'(A)') '\t DIHEDRAL-COM, GROUP 4' ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
        do j=1, ncom4;
         iatom=cv%amap%i(ind4(j))
         call atomid(iatom, sid, rid, ren, ac)
         write(info,667) '\t',j, iatom, sid, rid, ren, ac ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
        enddo
        deallocate(ind1,ind2,ind3,ind4)
       endif
!
 667 format(A,2I8,' ',4A)
!
       end subroutine cv_dihe_com_list
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
#endif /* automatically protect all code */

      end module cv_dihe_com
!
