! ********************************************************************************!
! Preprocessed by GNU CPP !
! ********************************************************************************!
!
! CV_CVRMS.MOD
!
! ROUTINES FOR COLLECTIVE VARIABLE `CVRMS` : DEFINED IN TERMS OF OTHER CV IN THE FOLLOWING WAY:
! (compatibility limited to steered dynamics as of 7.2010): z=sqrt( 1/N sum^N_i (z_i - z^0_i)^2 )
      module cv_cvrms
!
#if (KEY_STRINGM==1) /*  automatically protect all code */
!
      use cv_common
      use cv_types
      use ivector
!
      implicit none
      private
      ! subroutines
      public cv_cvrms_add
      public cv_cvrms_calc
      public cv_cvrms_list
!
      contains
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       function cv_cvrms_add(cv_list,k,gamma,weight) ! note: i is the atom index in the PSF
       use stream
       real(chm_real) :: k, gamma, weight
       type (int_vector) :: cv_list
! locals
       integer :: i, j, l, m, n, ind, num_int, ncv
       logical :: found, cv_cvrms_add
       type (int_vector) :: unique_amap_ptr ! contains a unique list of atom map pointers (to speed up gradient computation, etc)
!
       character(len=len("CV_CVRMS_ADD>") ),parameter::whoami="CV_CVRMS_ADD>";!macro
!
! check for duplicate CV (exact identical entry only)
       found=.false.
       do l=1, cv%num_cv
        if (cv%type(l).eq.cvrms) then
         found=.true.
         ncv=cv_list%last
         if (found) found=(ncv.eq.cv%priv(l)%p(1))
         ind=2
         do j=1,ncv
           if (found) found= &
     & (cv_list%i(j).eq.cv%priv(l)%p(ind))
           ind=ind+1
         enddo
        endif
        if (found) exit
       enddo
!
       if (.not.found) then ! (if found -- do nothing)
        l=cv_common_add(k,gamma,weight,cvrms) ! get a new cv index
        if (l.gt.0) then
! allocate private data
! space needed:
          ncv=cv_list%last
          num_int = 1 + ncv ! number of ints needed for storage
!
         allocate(cv%priv(l)%p(num_int));
         cv%priv(l)%p(1)=ncv
! now add slave CV indices
         ind=2
         do j=1,ncv
           m=cv_list%i(j)
           if ( (m.le.0) .or. (m.gt.cv%num_cv) .or. (m.eq.l) ) then
             call wrndie(0,whoami,trim(' CV INDEX OUT OF BOUNDS.'))
           elseif (cv%type(m).eq.cvrms) then
             call wrndie(0,whoami,trim(' SELECTED CV IS COLLECTIVE.'))
! elseif (.not.cv%active(m)) then ! this should be OK
! call wrndie(0,whoami,trim(' SELECTED CV IS MARKED INACTIVE.'))
           endif
           cv%priv(l)%p(ind)=m
           cv%active(m)=.false. ! mark slave CV inactive, so that it does not add forces on its own (default behaviour in master)
! copy (uniquely) the atom map of the slave cv
           do i=2,cv%priv(m)%amap_ptr(1)+1
            n=int_vector_uadd(unique_amap_ptr, cv%priv(m)%amap_ptr(i))
           enddo
!
           ind=ind+1
         enddo
         cv_cvrms_add=.true. ! note that I am not updating the cv%amap cvlist -- not sure what the desired behavior should be !
!
         m=unique_amap_ptr%last ! number of unique atoms this cv depends on
         allocate(cv%priv(l)%amap_ptr(m+1)) ! add one to include length of list
         cv%priv(l)%amap_ptr(1)=m;
         cv%priv(l)%amap_ptr(2:m+1)=unique_amap_ptr%i(1:m)
         call int_vector_done(unique_amap_ptr)
!
        else ! out of bounds
         call wrndie(0,whoami,trim(' ERROR ADDING CVRMS CV. NOTHING DONE.'))
         cv_cvrms_add=.false.
        endif
       else ! found
         call wrndie(0,whoami,trim(' CVRMS CV ALREADY PRESENT. NOTHING DONE.'))
         cv_cvrms_add=.false.
       endif
       end function cv_cvrms_add
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       recursive subroutine cv_cvrms_calc(i,x,y,z,mass,fx,fy,fz, &
     & calctheta,deriv,addforce,fext)
       use cv_posi_com, only: cv_posi_com_calc
       use cv_dist_com, only: cv_dist_com_calc
       use cv_dihe_com, only: cv_dihe_com_calc
       use cv_angle_com, only: cv_angle_com_calc
       use cv_anglvec, only: cv_anglvec_calc
       use cv_qcomp, only: cv_qcomp_calc
       use cv_rmsd, only: cv_rmsd_calc
       use cv_drmsd, only: cv_drmsd_calc
       use cv_proj, only: cv_proj_calc
!
       use consta
!
       real(chm_real) :: x(:), y(:), z(:), mass(:)
       real(chm_real) :: fx(:), fy(:), fz(:)
       real(chm_real), optional :: fext ! `external` force ; present here for nominal compatibility, but (should) not be used
       integer :: i ! cv index
       logical :: calctheta ! whether or not to calculate theta(x); if not, use theta=cv%r(i,instant)
                            ! note that if calctheta=.false., we don`t calculate the derivatives!
       logical :: deriv ! whether or not to calculate derivatives w.r.t. x
       logical :: addforce ! whether or not to add forces on simulation atoms
!
       real(chm_real) :: dummy
       integer , pointer :: priv(:)
!
       integer :: ncv
       integer, pointer :: ind(:)
       integer :: j, k, m, ii, jj
! variables for cv and derivative calculations
       real(chm_real) :: theta, f
       logical :: calctheta2, addforce2, deriv2, addforce_slaves ! for constituent force calculations
! do work:
! look up cv index in the private array of CV i
       ncv=cv%priv(i)%p(1)
!
       allocate(ind(ncv))
       ind=cv%priv(i)%p(2:ncv+1)
!
! because the collective CVs (i.e. those made up of other CVs) are defined only after the other (in particular, component) CVs have been defined,
! the component CVs should have their values and derivatives already computed (I assume this below)
       if (calctheta) then
        if (.not.deriv) then
         theta=0d0
         do jj=1, ncv
          j=ind(jj)
          select case(cv%type(j))
           case(dihe_com)
! minimum angle:
            dummy=(cv%r(j,instant)-cv%r(j,ref))
            dummy=modulo(dummy,TWOPI)
            if (dummy.gt.PI) dummy=dummy-TWOPI ! pick the smallest angular displacement
! minimum angular displacement from previous instant CV value (i.e. unwrapping):
! note that this required the "previnst" value at 1st timestep (should provide via charmm script: e.g set previnst(col 10) to ref values)
! dummy=cv%r(j,instant)-cv%r(j,previnst)
! dummy=modulo(dummy,TWOPI)
! if (dummy.gt.PI) dummy=dummy-TWOPI ! pick the smallest angular displacement
! dummy=(cv%r(j,previnst)+dummy)-cv%r(j,ref)
!
           case(angle_com) ! cannot tell between theta/-theta
            dummy=(abs(cv%r(j,instant))-abs(cv%r(j,ref)))
           case default
            dummy=(cv%r(j,instant)-cv%r(j,ref))
          end select
          theta=theta + dummy**2
         enddo
         theta=sqrt(theta/ncv) ! RMS
        else ! deriv=true
!cccccccccccccccccc compute values and derivatives in the same loop
! do j=1,cv%amap%last ! over all indices in the map
         priv=>cv%priv(i)%amap_ptr ! this should copy the pointers statically; the pointers point to the _same_ data
         do jj=2,priv(1)+1 ! only a subset of indices needs to be considered
          j=priv(jj)
!
          do ii=1,2
          cv%gradx(i,j,ii)=0d0;cv%grady(i,j,ii)=0d0;cv%gradz(i,j,ii)=0d0
          enddo
         enddo
!
         theta=0d0
         do jj=1, ncv
          j=ind(jj)
          select case(cv%type(j))
           case(dihe_com)
! minimum angle:
            dummy=(cv%r(j,instant)-cv%r(j,ref))
            dummy=modulo(dummy,TWOPI)
            if (dummy.gt.PI) dummy=dummy-TWOPI ! pick the smallest angular displacement
! minimum angular displacement from previous instant CV value (i.e. unwrapping):
! note that this required the "previnst" value at 1st timestep (should provide via charmm script: e.g set previnst(col 10) to ref values)
! dummy=cv%r(j,instant)-cv%r(j,previnst)
! dummy=modulo(dummy,TWOPI)
! if (dummy.gt.PI) dummy=dummy-TWOPI ! pick the smallest angular displacement
! dummy=(cv%r(j,previnst)+dummy)-cv%r(j,ref)
           case(angle_com) ! cannot tell between theta/-theta
            dummy=(abs(cv%r(j,instant))-abs(cv%r(j,ref)))
           case default
            dummy=(cv%r(j,instant)-cv%r(j,ref))
          end select
          theta=theta + dummy**2
! note that the gradient computation below will be very costly for many CVS (and inefficient if CVS use small atom subsets)
! should make the slave cvs contribute to this sum?
!
! cv%gradx(i,:,:)=cv%gradx(i,:,:)+dummy*cv%gradx(j,:,:)
! cv%grady(i,:,:)=cv%grady(i,:,:)+dummy*cv%grady(j,:,:)
! cv%gradz(i,:,:)=cv%gradz(i,:,:)+dummy*cv%gradz(j,:,:)
! potentially faster version:
!
          priv=>cv%priv(j)%amap_ptr ! contains pointers to private data of slave cv j
          do k=2,priv(1)+1 ! only a subset of indices needs to be considered
           m=priv(k)
           do ii=1,2
            cv%gradx(i,m,ii)=cv%gradx(i,m,ii)+dummy*cv%gradx(j,m,ii)
            cv%grady(i,m,ii)=cv%grady(i,m,ii)+dummy*cv%grady(j,m,ii)
            cv%gradz(i,m,ii)=cv%gradz(i,m,ii)+dummy*cv%gradz(j,m,ii)
           enddo ! ii
          enddo ! k -- all atom map indices of slave cv
!
         enddo ! j
         theta=sqrt(theta/ncv)
         dummy=1d0/theta/ncv
! cv%gradx(i,:,:)=cv%gradx(i,:,:)*dummy
! cv%grady(i,:,:)=cv%grady(i,:,:)*dummy
! cv%gradz(i,:,:)=cv%gradz(i,:,:)*dummy
! potentially faster version
         priv=>cv%priv(i)%amap_ptr ! contains pointers to private data of slave cv j
         do k=2,priv(1)+1 ! only a subset of indices needs to be considered
          m=priv(k)
          cv%gradx(i,m,:)=cv%gradx(i,m,:)*dummy
          cv%grady(i,m,:)=cv%grady(i,m,:)*dummy
          cv%gradz(i,m,:)=cv%gradz(i,m,:)*dummy
         enddo
        endif ! deriv
        cv%r(i,previnst)=cv%r(i,instant)
        cv%r(i,instant)=theta ! RMS
!
!
       else ! .not. calctheta
        theta=cv%r(i,instant)
       endif ! calctheta
! (2) add force via slave cv calc calls or by looping over the entire gradient array
! NOTE that the forces calculated here are NOT acting on the CV, as in
! the evolution subroutine, but on the simulation atoms
       dummy=cv%r(i,zcur)-theta ! zcur contains reference coords (combination of main+comp)
!
       f=cv%k(i)*dummy ! note that f is not divided by 1/sqrt(mass)
       cv%r(i,forces2)=f
       if (addforce) then
        if (present(fext)) f=fext ! override
!
        addforce_slaves=.false.
        if (addforce_slaves) then
! compute using slave addforce routines:
! THIS IS NOW OBSOLETE AND SHOULD BE REMOVED AT SOME POINT IN THE FUTURE
!
         f=f/theta/ncv ! constant factor coming from derivative of this CV
! loop over all constituent CV (NOTE: it might be faster to just loop over all the atoms and use the current gradients
! loop over all relevant indices in the map
! note: fx are actually components of grad V, not -grad V!
         calctheta2=.false.
         deriv2=.false.
         addforce2=.true.
!
         do jj=1,ncv
          j=ind(jj)
          select case(cv%type(j))
           case(posi_com_x, posi_com_y, posi_com_z);
            dummy=(cv%r(j,instant)-cv%r(j,ref))
            call cv_posi_com_calc(j,x,y,z,mass,fx,fy,fz, &
     & calctheta2,deriv2,addforce2,f*dummy)
           case(qcomp_1, qcomp_2, qcomp_3, qcomp_4);
            dummy=(cv%r(j,instant)-cv%r(j,ref))
            call cv_qcomp_calc(j,x,y,z,mass,fx,fy,fz, &
     & calctheta2,deriv2,addforce2,f*dummy)
           case(angle_com);
            dummy=(abs(cv%r(j,instant))-abs(cv%r(j,ref)))
            call cv_angle_com_calc(j,x,y,z,mass,fx,fy,fz, &
     & calctheta2,deriv2,addforce2,f*dummy)
           case(anglvec);
            dummy=(cv%r(j,instant)-cv%r(j,ref))
            call cv_anglvec_calc(j,x,y,z,mass,fx,fy,fz, &
     & calctheta2,deriv2,addforce2,f*dummy)
           case(dihe_com);
! minimum angle:
           dummy=(cv%r(j,instant)-cv%r(j,ref))
           dummy=modulo(dummy,TWOPI)
           if (dummy.gt.PI) dummy=dummy-TWOPI ! pick the smallest angular displacement
! minimum angular displacement from previous instant CV value (i.e. unwrapping):
! note that this required the "previnst" value at 1st timestep (should provide via charmm script: e.g set previnst(col 10) to ref values)
! dummy=cv%r(j,instant)-cv%r(j,previnst)
! dummy=modulo(dummy,TWOPI)
! if (dummy.gt.PI) dummy=dummy-TWOPI ! pick the smallest angular displacement
! dummy=(cv%r(j,previnst)+dummy)-cv%r(j,ref)
            call cv_dihe_com_calc(j,x,y,z,mass,fx,fy,fz, &
     & calctheta2,deriv2,addforce2,f*dummy)
           case(dist_com);
            dummy=(cv%r(j,instant)-cv%r(j,ref))
            call cv_dist_com_calc(j,x,y,z,mass,fx,fy,fz, &
     & calctheta2,deriv2,addforce2,f*dummy)
           case(rmsd);
            dummy=(cv%r(j,instant)-cv%r(j,ref))
            call cv_rmsd_calc(j,x,y,z,mass,fx,fy,fz, &
     & calctheta2,deriv2,addforce2,f*dummy)
           case(drmsd);
            dummy=(cv%r(j,instant)-cv%r(j,ref))
            call cv_drmsd_calc(j,x,y,z,mass,fx,fy,fz, &
     & calctheta2,deriv2,addforce2,f*dummy)
           case(proj);
            dummy=(cv%r(j,instant)-cv%r(j,ref))
            call cv_proj_calc(j,x,y,z,mass,fx,fy,fz, &
     & calctheta2,deriv2,addforce2,f*dummy)
           case(cvrms); ! in principle, possible, but not tested (or recommended)
            dummy=(cv%r(j,instant)-cv%r(j,ref))
            call cv_cvrms_calc(j,x,y,z,mass,fx,fy,fz, &
     & calctheta2,deriv2,addforce2,f*dummy)
           case default
            call wrndie(0,' CV_CVRMS_CALC>',trim('UNKNOWN CV SPECIFIED.'))
          end select
         enddo ! over all cv
!
        else ! loop over all atoms -- some force terms may be zero (hopefully, not many)
! do jj=1,cv%amap%last
         priv=>cv%priv(i)%amap_ptr
         do k=2, priv(1)+1
          jj=priv(k)
          j=cv%amap%i(jj) ! psf index
          fx(j)=fx(j)-f*cv%gradx(i,jj,1)
          fy(j)=fy(j)-f*cv%grady(i,jj,1)
          fz(j)=fz(j)-f*cv%gradz(i,jj,1)
         enddo
        endif ! addforce_slaves
!
       endif ! addforce
! free memory
       deallocate(ind)
       end subroutine cv_cvrms_calc
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine cv_cvrms_list(i)
       use stream
       use multicom_aux;
       use mpi
 character(len=132)::info(21)=(/'','','','','','','','','','','','','','','','','','','','',''/);! output buffer
!
       integer :: i, j, jj
!
       integer :: ncv
       integer, pointer, dimension(:) :: ind
       character(len=len("CV_CVRMS_LIST>") ),parameter::whoami="CV_CVRMS_LIST>";!macro
!
!
       if (MPI_COMM_STRNG.eq.MPI_COMM_NULL) return ! only replica heads stay
!
! check type just in case
       if (cv%type(i).ne.cvrms) then
        call wrndie(0,whoami,trim(' WRONG CV TYPE RECEIVED.'))
       endif
!
       if (ME_STRNG.eq.0) then
        ncv=cv%priv(i)%p(1)
!
        allocate(ind(ncv))
!
        ind=cv%priv(i)%p(2:ncv+1)
!
        write(info,'(A)') '\t CV RMSD FROM REFERENCE' ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
        do jj=1, ncv;
         j=ind(jj)
         write(info,667) '\t INDEX: ',j ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
        enddo
        deallocate(ind)
       endif
!
 667 format(2(A,I8),A,F11.5)
!
       end subroutine cv_cvrms_list
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
#endif /* automatically protect all code */
      end module cv_cvrms
!
