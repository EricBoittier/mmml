! ********************************************************************************!
! Preprocessed by GNU CPP !
! ********************************************************************************!
!
! CV_POSI_COM.MOD
!
! ROUTINES FOR COLLECTIVE VARIABLE 'X-POSITION-COM', 'Y-POSITION-COM', 'Z-POSITION-COM'
!
!
      module cv_posi_com
#if (KEY_STRINGM==1) /*  automatically protect all code */
!
#if (KEY_STRINGM==1)
!
      use cv_common
      use cv_types
      use cv_frames ! reference frames
      use ivector
      use ivector_list
!
      implicit none
      private
!
      public cv_posi_com_add
      public cv_posi_com_calc
      public cv_posi_com_list
!
      contains
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! this function does the job for x, y, and z positions
       function cv_posi_com_add(type,atom_list,k,gamma,weight,frame)
       use stream
       character(len=132)::info(21)=(/'','','','','','','','','','','','','','','','','','','','',''/);! output buffer
!
       real(chm_real) :: k, gamma, weight
       integer, optional :: frame ! reference frame
       type (int_vector) :: atom_list
! locals
       integer :: type, j, l, m, ind, num_int, ncom, ii, jj, ncomf
       integer, pointer :: indf(:)
       logical :: found, cv_posi_com_add
       character(len=1) :: pos
       type (int_vector) :: unique_amap_ptr ! contains a unique list of atom map pointers (to speed up gradient computation, etc)
!
       character(len=len("CV_POSI_COM_ADD>") ),parameter::whoami="CV_POSI_COM_ADD>";!macro
!
       select case (type)
        case (posi_com_x); pos='X'
        case (posi_com_y); pos='Y'
        case (posi_com_z); pos='Z'
        case default;
         call wrndie(0,whoami,trim('UNKNOWN CV TYPE. NOTHING DONE.'))
         return
       end select
!
! check for duplicate CV (exact identical entry only)
       found=.false.
       do l=1, cv%num_cv
        if (cv%type(l).eq.type) then
         ncom=atom_list%last
         found=ncom.eq.cv%priv(l)%p(1)
         ind=2
         do j=1,ncom
           if (found) found= &
     & (atom_list%i(j).eq.cv%amap%i(cv%priv(l)%p(ind)))
           ind=ind+1
         enddo
         if (found) found=(frame.eq.cv%amap%i(cv%priv(l)%p(ind)))
        endif
        if (found) exit
       enddo
!
       if (.not.found) then ! (if found -- do nothing)
        l=cv_common_add(k,gamma,weight,type) ! get a new cv index
        if (l.gt.0) then
! allocate private data
! space needed:
         ncom=atom_list%last
         num_int = 1 + ncom + 1 ! number of ints needded for storage (last entry for reference frame VO 2.27)
!
         allocate(cv%priv(l)%p(num_int));
         cv%priv(l)%p(1)=ncom
! now add atom indices
         ind=2
         do j=1,ncom
           m=atom_list%i(j)
           if (m.le.0) call wrndie(0,whoami,trim(' INVALID ATOM INDEX SPECIFIED.'))
           cv%priv(l)%p(ind)=int_vlist_uaddu(cv%amap,m,l) ! add indices into unique map (which also stores this cv index)
           m=int_vector_uadd(unique_amap_ptr,cv%priv(l)%p(ind))
           ind=ind+1
         enddo
! check if a reference frame is specified, if it is valid, record
         if (present(frame)) then
          if (frame.ge.0.and.frame.le.frames%num_frames) then
           cv%priv(l)%p(ind)=frame ! append frame index to end
! 10.2010 VO *************************************************************************
! since the CV depends on atom coords through the definition of a frame,
! we have to add the relevant atom lists in cv%amap with the index of this cv (l)
          ncomf=frames%priv(frame)%p(1);
          allocate(indf(ncomf));
          ii=2; jj=ii+ncomf-1;
          indf=frames%priv(frame)%p(ii:jj)
          do j=1, ncomf ! loop over frame atoms
            m=int_vector_uadd(cv%amap%v(indf(j)),l) ! add index of this cv
            m=int_vector_uadd(unique_amap_ptr,indf(j)) ! add index of the frame atom to the list of unique atom labels
          enddo
          deallocate(indf)
!*****************************************************************************************
          else
 call wrndie(0,whoami,trim(' INVALID FRAME SPECIFIED. WILL USE FIXED COORDINATE SYSTEM.'))
           cv%priv(l)%p(ind)=0
          endif
         else
          cv%priv(l)%p(ind)=0
         endif
!
         m=unique_amap_ptr%last ! number of unique atoms this cv depends on
         allocate(cv%priv(l)%amap_ptr(m+1)) ! add one to include length of list
         cv%priv(l)%amap_ptr(1)=m;
         cv%priv(l)%amap_ptr(2:m+1)=unique_amap_ptr%i(1:m)
         call int_vector_done(unique_amap_ptr)
!
         cv_posi_com_add=.true.
!
        else ! out of bounds
write(info(1),*)'ERROR ADDING POSITION_COM_',pos,' CV. NOTHING DONE.';call wrndie(0,whoami,trim(info(1)))
         cv_posi_com_add=.false.
        endif
       else ! found
write(info(1),*)' POSITION_COM_',pos,' CV ALREADY PRESENT. NOTHING DONE.';call wrndie(0,whoami,trim(info(1)))
         cv_posi_com_add=.false.
       endif
       end function cv_posi_com_add
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! this routine does the job for x, y, and z positions
       subroutine cv_posi_com_calc(i,x,y,z,mass,fx,fy,fz,calctheta, &
     & deriv,addforce,fext)
!
       real(chm_real) :: x(:), y(:), z(:), fx(:), fy(:), fz(:), mass(:)
       real(chm_real), optional :: fext ! `external` force for planar dynamics (see smcv_addforce in smcv_master)
       integer :: i, type ! cv index, cv type
       logical :: calctheta ! whether or not to calculate theta(x); if not, use theta=cv%r(i,instant)
                            ! note that if calctheta=.false., we don`t calculate the derivatives!
       logical :: deriv ! whether or not to calculate derivatives w.r.t. x
       logical :: addforce ! whether or not to add forces on simulation atoms
!
       real(chm_real) :: dummy
!
       integer, pointer, dimension(:) :: ind1, indf! , indpsf1
       real(chm_real), pointer, dimension(:,:) :: r1
       real(chm_real), pointer, dimension(:) :: m1
!
       integer :: ncom1, ncomf, ind ! , indpsf
       integer :: j, k, ii, jj, frame
! variables for cv and derivative calculations
       real(chm_real) :: com(3), f, w, totm1
       real(chm_real) :: A(3,3) ! orthogonal transformation matrix
       real(chm_real) :: dA(3,3) ! derivative of A
       real(chm_real) :: p(3) ! position in the local reference frame
       logical :: qframe ! is a local coordinate frame defined for this CV
       integer, pointer :: priv(:)
!
! get cv type
       type=cv%type(i)
       select case(type);
        case(posi_com_x); type=1
        case(posi_com_y); type=2
        case(posi_com_z); type=3
       end select
! do work:
! look up atom index in the private array of CV i; obtain the PSF index from amap
       ncom1=cv%priv(i)%p(1)
!
       allocate(ind1(ncom1),r1(3,ncom1),m1(ncom1))
! allocate(indpsf1(ncom1))
!
! extract indices into the atom map
       ii=2; jj=ii+ncom1-1; ind1=cv%priv(i)%p(ii:jj)
!
!cccccccccccccccccccc get coordinates ccccccccccccccccccc
       do j=1, ncom1;
         ind=cv%amap%i(ind1(j)) ! actual psf index
! indpsf1(j)=ind ;
         r1(1,j)=x(ind); r1(2,j)=y(ind); r1(3,j)=z(ind); m1(j)=mass(ind)
       enddo
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! now we have all the coordinates
       frame=cv%priv(i)%p(jj+1)
       qframe=(frame.gt.0)
!
       if (calctheta) then
        totm1=1d0/sum(m1);
! normalize masses
        m1=m1*totm1;
!
        com=0d0
!
        do j=1,ncom1;
         do k=1,3;
          com(k)=com(k)+r1(k,j)*m1(j); ! center of mass in absolute frame
         enddo
        enddo
!
        if (qframe) then ! if this position is relative to a local frame, obtain frame gradients
         call frames_calc(frame,x,y,z,mass,deriv)
         A=transpose(frames%r(:,:,frame)); ! transformation matrix
         com(:)=com(:)-frames%o(:,frame)
         p(:)=matmul(A,com); ! components in local frame
        else
         A=RESHAPE( (/1d0,0d0,0d0,0d0,1d0,0d0,0d0,0d0,1d0/),(/3,3/) ) ! for later use
         p(:)=com(:); ! trivial transformation
        endif
!
        cv%r(i,previnst)=cv%r(i,instant)
        cv%r(i,instant)=p(type)
!cccccccccccccccccc now compute derivative
        if (deriv) then
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
!
         if (qframe) then ! first, compute contribution from frame derivatives
! extract frame indices (need to compute gradients w.r.t atoms that define the frame!)
          ncomf=frames%priv(frame)%p(1);
          allocate(indf(ncomf));
!
          ii=2; jj=ii+ncomf-1;
!
! write(0,*) frame, ii,jj, size(frames%priv),
! & size(frames%priv(frame)%p), frames%priv(frame)%p(1)
! if(frame.gt.size(frames%priv).or.
! & jj.gt.size(frames%priv(frame)%p))
! & stop
!
!
           indf=frames%priv(frame)%p(ii:jj)
!
! loop over all atoms in the frame definition
!
          do j=1, ncomf ! loop over frame atoms
           ind=indf(j) ! index into atom map
           w=1d0/sqrt(mass(cv%amap%i(ind))) ! actual psf index
           f=frames%grado(ind,frame);
! gradx
           dA=transpose(frames%gradrx(:,:,ind,frame))
           dummy=dA(type,1)*com(1)+dA(type,2)*com(2)+dA(type,3)*com(3) &
     & -A(type,1)*f
           cv%gradx(i,ind,1)=dummy
           cv%gradx(i,ind,2)=dummy*w
! grady
           dA=transpose(frames%gradry(:,:,ind,frame))
           dummy=dA(type,1)*com(1)+dA(type,2)*com(2)+dA(type,3)*com(3) &
     & -A(type,2)*f
           cv%grady(i,ind,1)=dummy
           cv%grady(i,ind,2)=dummy*w
! gradz
           dA=transpose(frames%gradrz(:,:,ind,frame))
           dummy=dA(type,1)*com(1)+dA(type,2)*com(2)+dA(type,3)*com(3) &
     & -A(type,3)*f
           cv%gradz(i,ind,1)=dummy
           cv%gradz(i,ind,2)=dummy*w
          enddo ! over frame atoms
         endif ! frame
! aardvark
! write(600+whoiam,*) cv%gradx(i,:,1)
! write(600+whoiam,*) cv%grady(i,:,1)
! write(600+whoiam,*) cv%gradz(i,:,1)
! stop
! write(6,*) whoiam, 'computed frame component'
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! loop over all atoms in the CV COM
         do j=1,ncom1
          ind=ind1(j)
          w=sqrt(totm1/m1(j)) ! for mass-weighting
! gradx
          dummy=A(type,1)*m1(j)
          cv%gradx(i,ind,1)=cv%gradx(i,ind,1)+dummy
          cv%gradx(i,ind,2)=cv%gradx(i,ind,2)+dummy*w
! grady
          dummy=A(type,2)*m1(j)
          cv%grady(i,ind,1)=cv%grady(i,ind,1)+dummy
          cv%grady(i,ind,2)=cv%grady(i,ind,2)+dummy*w
! gradz
          dummy=A(type,3)*m1(j)
          cv%gradz(i,ind,1)=cv%gradz(i,ind,1)+dummy
          cv%gradz(i,ind,2)=cv%gradz(i,ind,2)+dummy*w
         enddo
! aardvark
! write(600+whoiam,*) cv%gradx(i,:,1)
! stop
! write(6,*) whoiam, 'computed CV component'
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        endif ! deriv
       else ! calctheta
        p(type)=cv%r(i,instant)
       endif
! aardvark
! write(6,*) whoiam, 'computed derivatives'
!
! NOTE that the forces calculated here are NOT acting on the CV, as in
! the evolution subroutine, but on the simulation atoms
       dummy=cv%r(i,zcur)-cv%r(i,instant) ! zcur contains reference coords (combination of main+comp)
!
       f=cv%k(i)*dummy ! note that f is not divided by 1/sqrt(mass)
       cv%r(i,forces2)=f
       if (addforce) then
        if (present(fext)) f=fext ! override for planar sampling
! if (qframe) then
! loop over all atom indices (easier to do this way, since the frame and CV atom indices may not be disjoint)
! need to figure out a clever way of only looping over the relevant atoms (SEE BELOW)
! write(600+whoiam,*) dummy, f
! do j=1,cv%amap%last
! indpsf=cv%amap%i(j)
! fx(indpsf)=fx(indpsf)-f*cv%gradx(i,j,1)
! fy(indpsf)=fy(indpsf)-f*cv%grady(i,j,1)
! fz(indpsf)=fz(indpsf)-f*cv%gradz(i,j,1)
! enddo
! else
! do j=1,ncom1 ! just loop over the atoms of this CV
! ind=ind1(j)
! indpsf=indpsf1(j) ! index into psf
! fx(indpsf)=fx(indpsf)-f*cv%gradx(i,ind,1)
! fy(indpsf)=fy(indpsf)-f*cv%grady(i,ind,1)
! fz(indpsf)=fz(indpsf)-f*cv%gradz(i,ind,1)
! enddo
! endif
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
       deallocate(ind1, & ! indpsf1,
     & r1,m1)
       if(associated(indf))deallocate(indf)
! aardvark
! write(6,*) whoiam, 'done with calc'
!
       end subroutine cv_posi_com_calc
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine cv_posi_com_list(i)
      use stream
      use multicom_aux;
      use mpi
      use chutil, only : atomid
!
       character(len=132)::info(21)=(/'','','','','','','','','','','','','','','','','','','','',''/);! output buffer
       integer :: i, j, type, ii, jj, iatom
       character(len=8) :: sid, rid, ren, ac
       character(len=1) :: pos
!
       integer :: ncom1, frame
       integer, pointer, dimension(:) :: ind1
       character(len=len("CV_POSI_COM_LIST>") ),parameter::whoami="CV_POSI_COM_LIST>";!macro
!
       if (MPI_COMM_STRNG.eq.MPI_COMM_NULL) return ! only replica heads stay
!
! check type just in case
       type=cv%type(i)
       select case (type)
        case (posi_com_x); pos='X'
        case (posi_com_y); pos='Y'
        case (posi_com_z); pos='Z'
        case default;
         call wrndie(0,whoami,trim(' UNKNOWN POSITION_COM CV. NOTHING DONE.'))
         return
       end select
!
       if (ME_STRNG.eq.0) then
        ncom1=cv%priv(i)%p(1)
!
        allocate(ind1(ncom1))
!
! extract indices into the atom map
        ii=2; jj=ii+ncom1-1; ind1=cv%priv(i)%p(ii:jj)
        frame=cv%priv(i)%p(jj+1)
!
        if (frame.ge.1) then
         write(info,'(A, I3)') &
     & '\t '//pos//' POSITION-COM RELATIVE TO LOCAL FRAME ',frame
         write(OUTU,'(A)') pack(info,info.ne.'');info='';
        else
         write(info,'(A)') '\t '//pos//' POSITION-COM'
         write(OUTU,'(A)') pack(info,info.ne.'');info='';
        endif
!
        do j=1, ncom1;
         iatom=cv%amap%i(ind1(j)) ! actual psf index
 call atomid(iatom, sid, rid, ren, ac)
         write(info,667) '\t',j, iatom, sid, rid, ren, ac ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
        enddo
        deallocate(ind1)
       endif
!
 667 format(A,2I8,' ',4A)
!
       end subroutine cv_posi_com_list
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
#endif
#endif /* automatically protect all code */
      end module cv_posi_com
!
