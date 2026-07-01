! ********************************************************************************!
! Preprocessed by GNU CPP !
! ********************************************************************************!
!
! CV_ANGLE_COM.MOD
!
! ROUTINES FOR COLLECTIVE VARIABLE `ANGLE` BETWEEN CENTERS OF MASS OF ATOMS
! NOTE THAT WE CANNOT SIMULATE A PATH THAT CROSSES 0/180 WITHOUT ADDING A (4TH) REFERENCE POINT
! WHICH WOULD REMOVE REFLECTIONAL SYMMETRY, SO THAT THETA AND -THETA ARE THEN _DIFFERENT_ ANGLES
! MIGHT ADD AN ADDITIONAL COLLECTIVE VARIABLE TO TAKE CARE OF THIS IN THE FUTURE
!
!
      module cv_angle_com
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
      public cv_angle_com_add
      public cv_angle_com_calc
      public cv_angle_com_list
!
      contains
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       function cv_angle_com_add(atom_list,k,gamma,weight) ! note: i is the atom index in the PSF
       use stream
       real(chm_real) :: k, gamma, weight
       type (int_vector), dimension(3) :: atom_list
! locals
       integer :: i, j, l, m, ind, num_int, ncom(3)
       logical :: found, cv_angle_com_add
       type (int_vector) :: unique_amap_ptr ! contains a unique list of atom map pointers (to speed up gradient computation, etc)
!
       character(len=len("CV_ANGLE_COM_ADD>") ),parameter::whoami="CV_ANGLE_COM_ADD>";!macro
!
! check for duplicate CV (exact identical entry only)
       found=.false.
       do l=1, cv%num_cv
        if (cv%type(l).eq.angle_com) then
         found=.true.
         do i=1,3
          ncom(i)=atom_list(i)%last
          if (found) found=(ncom(i).eq.cv%priv(l)%p(i))
         enddo
         ind=4
         do i=1,3
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
        l=cv_common_add(k,gamma,weight,angle_com) ! get a new cv index
        if (l.gt.0) then
! allocate private data
! space needed:
         do i=1,3 ; ncom(i)=atom_list(i)%last ; enddo
         num_int = 3 + sum(ncom) ! number of ints needded for storage
!
         allocate(cv%priv(l)%p(num_int));
         cv%priv(l)%p(1:3)=ncom(1:3)
! now add atom indices
         ind=4
         do i=1,3
          do j=1,ncom(i)
           m=atom_list(i)%i(j)
           if (m.le.0) call wrndie(0,whoami,trim(' INVALID ATOM INDEX SPECIFIED.'))
           cv%priv(l)%p(ind)=int_vlist_uaddu(cv%amap,m,l) ! add index m into unique map & return index; also associate cv l with atom index m
! collect unique atom indices (not psf indices, but indices into amap)
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
         cv_angle_com_add=.true.
        else ! out of bounds
         call wrndie(0,whoami,trim(' ERROR ADDING ANGLE_COM CV. NOTHING DONE.'))
         cv_angle_com_add=.false.
        endif
       else ! found
         call wrndie(0,whoami,trim(' ANGLE_COM CV ALREADY PRESENT. NOTHING DONE.'))
         cv_angle_com_add=.false.
       endif
       end function cv_angle_com_add
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine cv_angle_com_calc(i,x,y,z,mass,fx,fy,fz, &
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
       real(chm_real), parameter :: tol=1.0e-10
       real(chm_real) :: dummy
!
       integer, pointer, dimension(:) :: ind1, ind2, ind3
! & ,indpsf1, indpsf2, indpsf3
       real(chm_real), pointer, dimension(:) :: x1, x2, x3
       real(chm_real), pointer, dimension(:) :: y1, y2, y3
       real(chm_real), pointer, dimension(:) :: z1, z2, z3
       real(chm_real), pointer, dimension(:) :: m1, m2, m3
!
       integer :: ncom1, ncom2, ncom3, ind ! , indpsf
       integer :: j, ii, jj
! variables for cv and derivative calculations
       real(chm_real) :: theta, costh, sinth, &
     & xcom1, xcom2, xcom3, ycom1, ycom2, ycom3, &
     & zcom1, zcom2, zcom3, &
     & dx12, dy12, dz12, dx32, dy32, dz32, &
     & vx, vy, vz, vn, ux, uy, uz, un, &
     & fx12, fy12, fz12, fx32, fy32, fz32, &
     & f1x, f1y, f1z, f2x, f2y, f2z, f3x, f3y, f3z, &
     & f, w, totm1, totm2, totm3
       integer, pointer :: priv(:)
! do work:
! look up atom index in the private array of CV i; obtain the PSF index from amap
       ncom1=cv%priv(i)%p(1)
       ncom2=cv%priv(i)%p(2)
       ncom3=cv%priv(i)%p(3)
!
       allocate(ind1(ncom1),x1(ncom1),y1(ncom1),z1(ncom1),m1(ncom1))
       allocate(ind2(ncom2),x2(ncom2),y2(ncom2),z2(ncom2),m2(ncom2))
       allocate(ind3(ncom3),x3(ncom3),y3(ncom3),z3(ncom3),m3(ncom3))
! allocate(indpsf1(ncom1),indpsf2(ncom2),indpsf3(ncom2))
!
! extract indices into the atom map
       ii=4; jj=ii+ncom1-1; ind1=cv%priv(i)%p(ii:jj)
       ii=jj+1; jj=ii+ncom2-1; ind2=cv%priv(i)%p(ii:jj)
       ii=jj+1; jj=ii+ncom3-1; ind3=cv%priv(i)%p(ii:jj)
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
! now we have all the coordinates needed to compute angle and its derivative :
       if(calctheta) then
!
        totm1=1d0/sum(m1); totm2=1d0/sum(m2); totm3=1d0/sum(m3);
! normalize masses
        m1=m1*totm1; m2=m2*totm2; m3=m3*totm3;
!
        xcom1=0.; xcom2=0.; xcom3=0.;
        ycom1=0.; ycom2=0.; ycom3=0.;
        zcom1=0.; zcom2=0.; zcom3=0.;
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
! compute centers of mass
! xcom1=xcom1*totm1; ycom1=ycom1*totm1; zcom1=zcom1*totm1;
! xcom2=xcom2*totm2; ycom2=ycom2*totm2; zcom2=zcom2*totm2;
! xcom3=xcom3*totm3; ycom3=ycom3*totm3; zcom3=zcom3*totm3;
!
        dx12=xcom2-xcom1; dy12=ycom2-ycom1; dz12=zcom2-zcom1;
        dx32=xcom2-xcom3; dy32=ycom2-ycom3; dz32=zcom2-zcom3;
!
        ux=dx12; uy=dy12; uz=dz12;
        un=sqrt(ux*ux+uy*uy+uz*uz);
!
        vx=dx32; vy=dy32; vz=dz32;
        vn=sqrt(vx*vx+vy*vy+vz*vz);
!
        if (un.eq.0d0) un=1d0
        if (vn.eq.0d0) vn=1d0
        costh=(ux*vx+uy*vy+uz*vz)/(un*vn)
        sinth=sqrt(1d0-costh*costh)
        theta=atan2(sinth, costh) ! although acos would do just fine since we can not tell between theta & -theta
!
        cv%r(i,previnst)=cv%r(i,instant)
        cv%r(i,instant)=theta
!cccccccccccccccccc now compute derivative
        if (deriv) then
         un=1d0/un; ux=ux*un; uy=uy*un; uz=uz*un;
         vn=1d0/vn; vx=vx*vn; vy=vy*vn; vz=vz*vn;
!
         f=-1d0/(max(sinth, tol)); ! avoid singularity at zero
!
         fx12=f*un*(vx-costh*ux);
         fy12=f*un*(vy-costh*uy);
         fz12=f*un*(vz-costh*uz);
!
         fx32=f*vn*(ux-costh*vx);
         fy32=f*vn*(uy-costh*vy);
         fz32=f*vn*(uz-costh*vz);
!
! compute derivatives w.r.t COMs:
         f1x=-fx12; f1y=-fy12; f1z=-fz12;
         f2x= fx12+fx32; f2y= fy12+fy32; f2z= fz12+fz32;
         f3x=-fx32; f3y=-fy32; f3z=-fz32;
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
        endif ! deriv
       else
        theta=cv%r(i,instant)
       endif ! calctheta
!
! NOTE that the forces calculated here are NOT acting on the CV, as in
! the evolution subroutine, but on the simulation atoms
       dummy=abs(cv%r(i,zcur))-theta ! we cannot distinguish between theta and -theta (because of reflectional symmetry)
!
       f=cv%k(i)*dummy ! note that f is not divided by 1/sqrt(mass)
       cv%r(i,forces2)=f
       if (addforce) then
        if (present(fext)) f=fext ! override for planar sampling
! loop over all groups
! COM1
! loop over all relevant indices in the map
! note: fx are actually components of grad V, not -grad V!
! skip old way
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
! more efficient and conceptually correct way:
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
       deallocate(ind1, ind2, ind3, &
! & indpsf1, indpsf2, indpsf3,
     & x1, x2, x3, &
     & y1, y2, y3, &
     & z1, z2, z3, &
     & m1, m2, m3)
       end subroutine cv_angle_com_calc
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine cv_angle_com_list(i)
       use stream
       use string
       use multicom_aux;
       use mpi
       use chutil, only : atomid
!
       character(len=132)::info(21)=(/'','','','','','','','','','','','','','','','','','','','',''/);! output buffer
       integer :: i, j, type, ii, jj, iatom
       character(len=8) :: sid, rid, ren, ac
!
       integer :: ncom1, ncom2, ncom3
       integer, pointer, dimension(:) :: ind1, ind2, ind3
       character(len=len("CV_ANGLE_COM_LIST>") ),parameter::whoami="CV_ANGLE_COM_LIST>";!macro
!
       if (MPI_COMM_STRNG.eq.MPI_COMM_NULL) return ! only replica heads stay
!
! check type justt in case
       type=cv%type(i)
       if (type.ne.angle_com) then
        call wrndie(0,whoami,trim(' WRONG CV TYPE RECEIVED.'))
       endif
!
       if (ME_STRNG.eq.0) then
        ncom1=cv%priv(i)%p(1)
        ncom2=cv%priv(i)%p(2)
        ncom3=cv%priv(i)%p(3)
!
        allocate(ind1(ncom1),ind2(ncom2),ind3(ncom3))
!
! extract indices into the atom map
        ii=4; jj=ii+ncom1-1; ind1=cv%priv(i)%p(ii:jj)
        ii=jj+1; jj=ii+ncom2-1; ind2=cv%priv(i)%p(ii:jj)
        ii=jj+1; jj=ii+ncom3-1; ind3=cv%priv(i)%p(ii:jj)
!
        write(info,'(A)') '\t ANGLE-COM, GROUP 1' ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
        do j=1, ncom1;
         iatom=cv%amap%i(ind1(j)) ! actual psf index
         call atomid(iatom, sid, rid, ren, ac)
         write(info,667) '\t',j, iatom, sid, rid, ren, ac ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
        enddo
        write(info,'(A)') '\t ANGLE-COM, GROUP 2' ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
        do j=1, ncom2;
         iatom=cv%amap%i(ind2(j))
         call atomid(iatom, sid, rid, ren, ac)
         write(info,667) '\t',j, iatom, sid, rid, ren, ac ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
        enddo
        write(info,'(A)') '\t ANGLE-COM, GROUP 3' ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
        do j=1, ncom3;
         iatom=cv%amap%i(ind3(j))
         call atomid(iatom, sid, rid, ren, ac)
         write(info,667) '\t',j, iatom, sid, rid, ren, ac ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
        enddo
        deallocate(ind1,ind2,ind3)
       endif
!
 667 format(A,2I8,' ',4A)
!
       end subroutine cv_angle_com_list
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
#endif /* automatically protect all code */
      end module cv_angle_com
!
