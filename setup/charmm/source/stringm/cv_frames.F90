! ********************************************************************************!
! Preprocessed by GNU CPP !
! ********************************************************************************!
!
! CV_FRAMES.MOD
!
! SEPARATE MODULE FOR STORING REFERENCE FRAMES
! CVs that use reference (i.e. non-fixed) coordinates will use this module
!
!$DEC ATTRIBUTE INLINE :: eig3s
      module cv_frames
#if (KEY_STRINGM==1) /*  automatically protect all code */
!
#if (KEY_STRINGM==1)
!
      use chm_kinds
      use ivector_list
      use cv_common, only: cv, priv_ptr, main, comp, cv_common_rmsd
      implicit none
      private
!
      type cv_frame
      real(chm_real), dimension(:,:), pointer :: o ! frame origin
      real(chm_real), dimension(:,:,:), pointer :: r ! frame vectors
      real(chm_real), dimension(:,:), pointer :: grado ! gradient vector of frame origin w.r.t. x y or z (the same) ;
      real(chm_real), dimension(:,:,:,:,:), pointer :: gradr ! combined gradient array (with some extra space for `packed` communication)
      real(chm_real), dimension(:,:,:,:), pointer :: gradrx ! gradient vector of frame vectors w.r.t. x ;
      real(chm_real), dimension(:,:,:,:), pointer :: gradry !
      real(chm_real), dimension(:,:,:,:), pointer :: gradrz !
      logical, dimension(:), pointer :: recalculate ! flag that indicates that op values should be recalculated
      logical, dimension(:), pointer :: recalculate_grad ! flag that indicates that op gradients should be recalculated
      type (priv_ptr), dimension(:), pointer :: priv ! `private` to each frame
      integer :: num_frames=0 ! number of active collective frames
      end type cv_frame
!
      ! subroutines
      public frames_init ! initialize frame arrays
      public frames_done ! destroy all frames
      public frames_add ! add a frame
      public frames_list ! list frames
      public frames_calc ! calculate frame vectors & their gradients w.r.t. atom positions
      public frames_grad_init ! initialize frames%grad arrays
      public frames_print_local ! print out current frames to separate files
      public frames_print_global ! print out current frames to a combined file
      public frames_read_local ! read frames from separate files
      public frames_read_global ! read frames from a combined file
      public frames_reset_calculate ! when called, resets 'recalculate' flags; will be called from e.g. smcv_master
      public frames_calc_align_comp ! subroutine which computes frames from the x-set consistently with the xcomp-set
! public frames_align ! iteratively invert frame vectors (v -> -v)
! to guess the best alignment along string &
! (optional) to mimimize DIST(z,theta(x)) ; moved to sm_util
!
      public cv_frame
      ! variables
      type (cv_frame), public, save :: frames
      logical, public, save :: frames_initialized=.false., &
     & frames_grad_initialized=.false. ! have the cv%grad arrays been allocated
!
      ! parameters
      integer, parameter, public :: max_frames=10
!
      contains
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine frames_init()
       integer :: i
!
       if (.not.frames_initialized) then
!
        frames%num_frames=0
!
        allocate(frames%o(3,max_frames))
        allocate(frames%r(3,3,max_frames))
        allocate(frames%priv(max_frames)) ! allocate private pointer array
        allocate(frames%recalculate(max_frames))
        allocate(frames%recalculate_grad(max_frames))
        do i=1, max_frames
         nullify(frames%priv(i)%p) ! initialize pointer array
         nullify(frames%priv(i)%pr) ! initialize pointer array
        enddo
        frames_initialized=.true.
        frames%o=0d0
        frames%r=0d0
        frames%recalculate=.true.
        frames%recalculate_grad=.true.
       endif
       end subroutine frames_init
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine frames_done()
       integer :: i
       frames%num_frames=0
       if (frames_initialized) then
        deallocate(frames%o)
        deallocate(frames%r)
!
        do i=1, max_frames
         if(associated(frames%priv(i)%p))deallocate(frames%priv(i)%p)
         if(associated(frames%priv(i)%pr))deallocate(frames%priv(i)%pr)
        enddo
!
        if(associated(frames%grado))deallocate(frames%grado)
        if(associated(frames%gradr))deallocate(frames%gradr)
        nullify(frames%gradrx)
        nullify(frames%gradry)
        nullify(frames%gradrz)
! if (associated(frames%gradrx)) deallocate(frames%gradrx)
! if (associated(frames%gradry)) deallocate(frames%gradry)
! if (associated(frames%gradrz)) deallocate(frames%gradrz)
        if (associated(frames%recalculate)) &
     & deallocate(frames%recalculate)
        if (associated(frames%recalculate_grad)) &
     & deallocate(frames%recalculate_grad)
        frames_initialized=.false.
        frames_grad_initialized=.false.
       endif
       end subroutine frames_done
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       function frames_add(atom_list)
       use stream
!
       type (int_vector) :: atom_list
! locals
       integer :: j, l, m, ind, num_int, ncom
       logical :: found
       integer :: frames_add
       character(len=len("FRAMES_ADD>") ),parameter::whoami="FRAMES_ADD>";!macro
!
! check for duplicate frame (exact identical entry only)
       found=.false.
       do l=1, frames%num_frames
        ncom=atom_list%last
        found=ncom.eq.frames%priv(l)%p(1)
        ind=2
        do j=1,ncom
          if (found) found= &
     & (atom_list%i(j).eq.cv%amap%i(frames%priv(l)%p(ind)))
          ind=ind+1
        enddo
        if (found) exit
       enddo
!
       if (.not.found) then ! (if found -- do nothing)
        if (.not.frames_initialized) call frames_init()
        l=frames%num_frames + 1
        if (l.le.max_frames) then
         frames%num_frames=l
! allocate private data
! space needed:
         ncom=atom_list%last
         num_int = 1 + ncom ! number of ints needed for storage
!
         allocate(frames%priv(l)%p(num_int));
         frames%priv(l)%p(1)=ncom
! now add atom indices
         ind=2
         do j=1,ncom
           m=atom_list%i(j)
           if (m.le.0) call wrndie(0,whoami,trim(' INVALID ATOM INDEX SPECIFIED.'))
           frames%priv(l)%p(ind)=int_vlist_uadd(cv%amap,m) ! add indices into map (note: the same map as the one used in CV)
           ind=ind+1
         enddo
         frames_add=l
         frames%r(:,:,l)=0d0 ! do NOT change this; otherwise the axes may not be right-handed (see calc)
         frames%o(:,l)=0d0
        else ! out of bounds
         call wrndie(0,whoami,trim(' ERROR ADDING FRAME. NOTHING DONE.'))
         frames_add=0
        endif
       else ! found
         call wrndie(0,whoami,trim(' IDENTICAL FRAME ALREADY PRESENT. NOTHING DONE.'))
         frames_add=0
       endif
       end function frames_add
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine frames_list()
       use stream
       use multicom_aux;
       use string
       use mpi
       use chutil, only : atomid
!
       integer :: i, j, ii, jj, iatom
       character(len=8) :: sid, rid, ren, ac
       character(len=len("FRAMES_LIST>") ),parameter::whoami="FRAMES_LIST>";!macro
character(len=132)::info(21)=(/'','','','','','','','','','','','','','','','','','','','',''/);! output buffer
!
       integer :: ncom1
       integer, pointer, dimension(:) :: ind1
!
       if (MPI_COMM_STRNG.eq.MPI_COMM_NULL) return ! only replica heads stay
!
       if (ME_STRNG.eq.0) then ! only 1st ensemble replica reports
!
! check for initialization
        if (.not.frames_initialized.or.frames%num_frames.eq.0) then
         call wrndie(0,whoami,trim('NO FRAMES DEFINED.'))
         return
        endif
!
        do i=1, frames%num_frames
         ncom1=frames%priv(i)%p(1)
         allocate(ind1(ncom1))
! extract psf indices from the atom map
         ii=2; jj=ii+ncom1-1; ind1=frames%priv(i)%p(ii:jj)
!
         write(info,'(A, I3)') '\t  ATOM LIST FOR COORDINATE FRAME #', i
         write(OUTU,'(A)') pack(info,info.ne.'');info='';
         do j=1, ncom1;
          iatom=cv%amap%i(ind1(j)) ! actual psf index
          call atomid(iatom, sid, rid, ren, ac)
          write(info,667) '\t',j, iatom, sid, rid, ren, ac ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
         enddo
         deallocate(ind1)
        enddo
       endif
!
 667 format(A,2I8,' ',4A)
!
       end subroutine frames_list
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine frames_calc(i,x,y,z,mass,deriv)
       use sm_var, only: mestring , Id3
       use bestfit, only : eig3s, RMSBestFit, rmsd, norm3, veccross3
       use stream
       use number
!
       integer :: i ! which frame to calculate
       real(chm_real) :: x(:), y(:), z(:), mass(:)
       logical :: deriv
!
       integer, pointer, dimension(:) :: ind1
       real(chm_real), pointer, dimension(:) :: x1, y1, z1, m1
!
       integer :: ncom1, ind
       integer :: j, k, p, q, r, ii, jj
! variables for cv and derivative calculations
       real(chm_real) :: xcom1, ycom1, zcom1, totm1
! real(chm_real) :: C(3,3) ! mass-weighted correlation matrix
! real(chm_real) :: invA(3,3), dum(3,3) ! auxiliary matrix
       real(chm_real), dimension(3,3) :: C, A1, A2, A3, A4, invA ! 5.21.09: align frames the `correct` way
       real(chm_real) :: a11, a12, a13, a21, a22, a23, a31, a32, a33
       real(chm_real) :: detA, dist, corr1, corr2, corr3, corr4, cmax, dum
       real(chm_real) :: mu(3), e ! eigenvalues
       real(chm_real) :: b(3), v(3) ! auxiliary vector
       real(chm_real) :: dVdC(3,3,3,3); ! derivative of axis( evector v w.r.t. correlation matrix Cij)
! real(chm_real) :: dVdCfd(3,3,3,3), w; ! aardvark
       real(chm_real), parameter :: tol=1e-8
!
       character(len=len("FRAMES_CALC>") ),parameter::whoami="FRAMES_CALC>";!macro
! check for initialization
       if (.not.frames_initialized) then
        call wrndie(0,whoami,trim(' NO FRAMES DEFINED. NOTHING DONE.'))
        return
       endif
!
! check frame number:
       if (i.lt.1.or.i.gt.frames%num_frames) then
        call wrndie(0,whoami,trim(' OUT OF BOUNDS. NOTHING DONE.'))
        return
       endif
!
! check flags
       if (.not.(frames%recalculate(i).or. &
     & (frames%recalculate_grad(i).and.deriv))) &
     & return
! do work:
! look up atom index in the private array of CV i; obtain the PSF index from amap
       ncom1=frames%priv(i)%p(1)
!
       allocate(ind1(ncom1),x1(ncom1),y1(ncom1),z1(ncom1),m1(ncom1))
!
! extract indices into the atom map
       ii=2; jj=ii+ncom1-1; ind1=frames%priv(i)%p(ii:jj)
!
       do j=1, ncom1;
        ind=cv%amap%i(ind1(j)) ! actual psf index
        x1(j)=x(ind); y1(j)=y(ind); z1(j)=z(ind); m1(j)=mass(ind)
       enddo
! now we have all the coordinates
       totm1=1d0/sum(m1);
!
       xcom1=0.; ycom1=0.; zcom1=0.;
       do j=1, ncom1;
        xcom1=xcom1+x1(j)*m1(j);
        ycom1=ycom1+y1(j)*m1(j);
        zcom1=zcom1+z1(j)*m1(j);
       enddo
       xcom1=xcom1*totm1;
       ycom1=ycom1*totm1;
       zcom1=zcom1*totm1;
! how much did the origin move from previous step?
       dist=sqrt( (frames%o(1,i)-xcom1)**2 + &
     & (frames%o(2,i)-ycom1)**2 + &
     & (frames%o(3,i)-zcom1)**2 )
       frames%o(1,i)=xcom1
       frames%o(2,i)=ycom1
       frames%o(3,i)=zcom1
!
       C=0d0; ! initialize correlation matrix
       do j=1, ncom1;
        x1(j)=x1(j)-xcom1;
        y1(j)=y1(j)-ycom1;
        z1(j)=z1(j)-zcom1;
! compute mass-weighted correlation matrix
        C(1,1)=C(1,1)+x1(j)*x1(j)*m1(j);
        C(1,2)=C(1,2)+x1(j)*y1(j)*m1(j);
        C(1,3)=C(1,3)+x1(j)*z1(j)*m1(j);
!
        C(2,2)=C(2,2)+y1(j)*y1(j)*m1(j);
        C(2,3)=C(2,3)+y1(j)*z1(j)*m1(j);
!
        C(3,3)=C(3,3)+z1(j)*z1(j)*m1(j);
!
        x1(j)=x1(j)*m1(j)
        y1(j)=y1(j)*m1(j)
        z1(j)=z1(j)*m1(j)
       enddo
! enforce symmetry
       C(2,1)=C(1,2);
       C(3,1)=C(1,3);
       C(3,2)=C(2,3);
! diagonalize the correlation matrix to obtain coordinate frame
       call eig3s(C, mu, A1); ! eigenvectors are the principal axes (invA)
! write(6,*) whoiam,' : evectors :', A1 ! aardvark
! write(6,*) whoiam,' : evalues :', mu
! write(6,*) whoiam,' orthogonal?', matmul(transpose(invA),invA) : YES
! compute overlap of the new axes with the old:
       detA=A1(1,1)*(A1(2,2)*A1(3,3)-A1(2,3)*A1(3,2))+ &
     & A1(1,2)*(A1(2,3)*A1(3,1)-A1(2,1)*A1(3,3))+ &
     & A1(1,3)*(A1(2,1)*A1(3,2)-A1(2,2)*A1(3,1))
       if (detA.gt.0) A1(:,1)=-A1(:,1) ! ensure a (NO, LEFT 3.31.09 adhoc, for compat. ) right-handed coordinate frame
!
!cccc generate equivalent axes (assuming no eigenvalue degeneracy)
       do ii=1,3
        dum=A1(ii,1)
        A2(ii,1)=-dum;
        A3(ii,1)=-dum;
        A4(ii,1)= dum;
        dum=A1(ii,2)
        A2(ii,2)=-dum;
        A3(ii,2)= dum;
        A4(ii,2)=-dum;
        dum=A1(ii,3)
        A2(ii,3)= dum;
        A3(ii,3)=-dum;
        A4(ii,3)=-dum;
       enddo
! this is what we are doing above:
! A2(:,1)=-A1(:,1); A2(:,2)=-A1(:,2); A2(:,3)= A1(:,3);
! A3(:,1)=-A1(:,1); A3(:,2)= A1(:,2); A3(:,3)=-A1(:,3);
! A4(:,1)= A1(:,1); A4(:,2)=-A1(:,2); A4(:,3)=-A1(:,3);
! now permute axes
       corr1=0d0; corr2=0d0; corr3=0d0; corr4=0d0;
       do ii=1,3
        do jj=1,3
         dum=frames%r(ii,jj,i)
         corr1=corr1+dum*A1(ii,jj)
         corr2=corr2+dum*A2(ii,jj)
         corr3=corr3+dum*A3(ii,jj)
         corr4=corr4+dum*A4(ii,jj)
        enddo
       enddo
!
       cmax=max(corr1, corr2, corr3, corr4);
       if (cmax.eq.corr1) then; frames%r(:,:,i)=A1;
       elseif (cmax.eq.corr2) then; frames%r(:,:,i)=A2;
       elseif (cmax.eq.corr3) then; frames%r(:,:,i)=A3;
       elseif (cmax.eq.corr4) then; frames%r(:,:,i)=A4;
       else; frames%r(:,:,i)=A1; ! should never be executed
       endif
!
! dpr=0d0;
! do j=1,3;
!c old ad hoc trick to ensure maximum overlap with the previous set of axes
!c this should not be needed for `good` frames
! w=dot_product(frames%r(:,j,i), A1(:,j));
! if (w.lt.0) then
! A1(:,j)=-A1(:,j)
! dpr=dpr-w
! else
! dpr=dpr+w
! endif
! enddo
!
       ! aardvark
! write(100+whoiam,*) dist, dpr ! ,mu,
! & A1(1,1)*(A1(2,2)*A1(3,3)-A1(2,3)*A1(3,2))+
! & A1(1,2)*(A1(2,3)*A1(3,1)-A1(2,1)*A1(3,3))+
! & A1(1,3)*(A1(2,1)*A1(3,2)-A1(2,2)*A1(3,1))
! write(100+whoiam,*) A1(1,:);
! write(100+whoiam,*) A1(2,:);
! write(100+whoiam,*) A1(3,:);
! write(100+whoiam,*) '********'
    ! aardvark
       frames%recalculate(i)=.false. ! indicate that the new axes are known
!cccccccccccccccccc now compute derivative cccccccccccccccccccc
       if (deriv.and.frames%recalculate_grad(i)) then
! compute derivatives w.r.t. COM components
! check for gradient initialization
        if (.not.frames_grad_initialized) &
     & call frames_grad_init()
!
! compute dAij/dCpq
!
! loop over principal axes (vectors)
        do j=1,3
! construct matrix; A=C-muI; this matrix is singular; so throw away first row, and put j_th eigenvector (v) in last row to
! enforce v*dv=0
         e=mu(j);
         v=frames%r(:,j,i);
!
         a11=C(2,1); a12=C(2,2)-e ; a13=C(2,3);
         a21=C(3,1); a22=C(3,2) ; a23=C(3,3)-e;
         a31=v(1); a32=v(2) ; a33=v(3);
!
! make sure A is invertible; if so, compute and store inverse
! otherwise complain and quit (will deal with exceptions later)
         detA = a11*(a22*a33-a23*a32)+ &
     & a12*(a23*a31-a21*a33)+ &
     & a13*(a21*a32-a22*a31)
!
         if (abs(detA).lt.tol) then
          call wrndie(0,whoami,trim(' EXCEPTION: A MATRIX IS SINGULAR. NOTHING DONE.'))
          return
         else
! compute inverse explicitly
          detA=1d0/detA;
          invA(1,1)=detA*(a22*a33-a23*a32);
          invA(1,2)=detA*(a13*a32-a12*a33);
          invA(1,3)=detA*(a23*a12-a22*a13);
          invA(2,1)=detA*(a23*a31-a21*a33);
          invA(2,2)=detA*(a11*a33-a13*a31);
          invA(2,3)=detA*(a13*a21-a11*a23);
          invA(3,1)=detA*(a21*a32-a22*a31);
          invA(3,2)=detA*(a12*a31-a11*a32);
          invA(3,3)=detA*(a11*a22-a12*a21);
         endif
! test inverse ! aardvark ! passed
! write(600+whoiam,*)
! & matmul(invA,
! & transpose(RESHAPE((/a11,a12,a13,a21,a22,a23,a31,a32,a33/),
! & (/3,3/)))) ! note: fills column-wise
! stop
!
! loop over all components in C_pq:
         do p=1,3
          do q=1,p
! construct right hand side vector (b)
           do r=2,3
             b(r-1)=v(q) * ( v(p)*v(r)-Id3(r,p) ) + & ! dC_pq contribution
     & (1 - Id3(p,q)) * &
     & v(p) * ( v(q)*v(r)-Id3(r,q) ) ! dC_qp contribution
           enddo
           b(3)=0d0;
! compute derivative dV/dC
! note: only entries for which p >= q are populated; in fact, we really are computing dV/dC_pq + dV/dC_qp (this is OK b/c Cij=Cji)
           dVdC(:,j,p,q)=matmul(invA,b);
! write(700+whoiam,*) j, p, q, dVdC(:,j,p,q)
          enddo ! q
         enddo ! p
        enddo ! j
! aardvark: compute derivatives dVdC by finite differences & compare : PASSED
! detA=0.00001
! do p=1,3
! do q=1,p
! invA=C;
! invA(p,q)=invA(p,q)+detA;
! invA(q,p)=invA(q,p)+(1-transfer(p.eq.q,0))*detA;
! call eig3s(invA, mu, dum);
! DVDCfd(:,:,p,q)=-(frames%r(:,:,i)-dum)/detA;
! enddo
! enddo
! write(700+whoiam,*) dVdC(:,:,1,:)
! write(600+whoiam,*) DVDCfd(:,:,1,:)
! write(600+whoiam,*) '*****************'
! stop
!ccccccccccccccccccc tested matlab code cccccccccccccccccccccccccccccccccccccccccccccccccccc
!dvdA1=zeros(3,3,3,3); % 1 - vec#; 2 - vec component; 3 - matrix row; 4 - matrix column
!
!for i=1:3
!%construct matrix
! e=e1(i,i); v=v1(:,i);
! M=[ A1 - eye(3)*e ; v` ]; M=M(2:4,:); % throw away 1st row
!%
! for p=1:3 % 1st index in A_pq
! for q=1:p
! b=[ v(q)*[ v(p)*v-([1;2;3]==p)] + (1-(p==q)) * v(p)*[ v(q)*v-([1;2;3]==q)]; 0.]; b=b(2:4);%
! dvdA1(i,:,p,q) = M\b ;
! end
! end
!end
!% note: only entries for which p >= q are populated
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!ccccccccccccccccccccccccccc contract against dC/dx (x are actual psf atom positions) ccccccccccccccc
!
        do q=1,ncom1
         ind=ind1(q)
! gradient of COM w.r.t. atom coordinates
         frames%grado(ind,i)=m1(q)*totm1
! loop over all axes components:
         do j=1,3
          do k=1,3
! gradient w.r.t x-position of atom q
           frames%gradrx(j,k,ind,i) = &
     & ( &
     & dVdC(j,k,1,1) *2d0* x1(q) + & ! remember, (1) dVdC(:,:,p,q) is nonzero only for p<=q; (2) dC/dx is sparse
     & dVdC(j,k,2,1) * y1(q) + & ! coordinates are premultiplied by mass above
     & dVdC(j,k,3,1) * z1(q) )
! gradient w.r.t y-position
           frames%gradry(j,k,ind,i) = &
     & ( &
     & dVdC(j,k,2,1) * x1(q) + &
     & dVdC(j,k,2,2) *2d0* y1(q) + &
     & dVdC(j,k,3,2) * z1(q) )
! gradient w.r.t z-position
           frames%gradrz(j,k,ind,i) = &
     & ( &
     & dVdC(j,k,3,1) * x1(q) + &
     & dVdC(j,k,3,2) * y1(q) + &
     & dVdC(j,k,3,3) *2d0* z1(q) )
          enddo ! k
         enddo ! j
        enddo ! q
! aardvark: test gradient calculation by FD: PASSED
! write(700+whoiam,*) dVdC(:,:,1,:)
! write(700+whoiam,*) frames%gradrz(:,:,:,i)
! write(700+whoiam,*) frames%grado(:,i)

! detA=0.0001
       if (.false.) then ! hardwired FD test
!
       do k=1,ncom1
!
        do j=1, ncom1;
         ind=cv%amap%i(ind1(j)) ! actual psf index
         x1(j)=x(ind); y1(j)=y(ind); z1(j)=z(ind); m1(j)=mass(ind)
        enddo
!
        x1(k)=x1(k)+detA; ! perturbation

! now we have all the coordinates
        totm1=1d0/sum(m1);
!
        xcom1=0d0; ycom1=0d0; zcom1=0d0;
        do j=1, ncom1;
         xcom1=xcom1+x1(j)*m1(j);
         ycom1=ycom1+y1(j)*m1(j);
         zcom1=zcom1+z1(j)*m1(j);
        enddo
        xcom1=xcom1*totm1;
        ycom1=ycom1*totm1;
        zcom1=zcom1*totm1;
!
        C=0; ! initialize correlation matrix
        do j=1, ncom1;
         x1(j)=x1(j)-xcom1;
         y1(j)=y1(j)-ycom1;
         z1(j)=z1(j)-zcom1;
! compute mass-weighted correlation matrix
         C(1,1)=C(1,1)+x1(j)*x1(j)*m1(j);
         C(1,2)=C(1,2)+x1(j)*y1(j)*m1(j);
         C(1,3)=C(1,3)+x1(j)*z1(j)*m1(j);
!
         C(2,2)=C(2,2)+y1(j)*y1(j)*m1(j);
         C(2,3)=C(2,3)+y1(j)*z1(j)*m1(j);
!
         C(3,3)=C(3,3)+z1(j)*z1(j)*m1(j);
!
         x1(j)=x1(j)*m1(j)
         y1(j)=y1(j)*m1(j)
         z1(j)=z1(j)*m1(j)
        enddo
! enforce symmetry
        C(2,1)=C(1,2);
        C(3,1)=C(1,3);
        C(3,2)=C(2,3);
! diagonalize the correlation matrix to obtain coordinate frame
        call eig3s(C, mu, invA); ! eigenvectors are the principal axes (invA)
        frames%gradrz(:,:,k,i)=(invA-frames%r(:,:,i))/detA;
!
        frames%grado(k,i)=(xcom1-frames%o(1,i))/detA;

! write(700+mestring,*) k, frames%grado(k,i)

       enddo ! k

! write(600+whoiam,*) frames%gradrx(:,:,:,i)
! write(600+whoiam,*) frames%gradry(:,:,:,i)
! write(600+whoiam,*) frames%gradrz(:,:,:,i)
! write(700+whoiam,*) frames%grado(:,i)
! stop

! aardvark: done test gradients using FD: PASSED
       endif ! hardwired FD test
!cccccccccccccccccccccccccc done with derivatives cccccccccccccccccccccccc
        frames%recalculate_grad(i)=.false. ! indicate that derivatives are known
       endif ! deriv
! aardvark
! write(6,*) whoiam, 'computed derivatives'
! stop
! free memory
       deallocate(ind1, x1, y1, z1, m1)
       end subroutine frames_calc
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine frames_grad_init()
! initialize frames%grad arrays
       if (associated(frames%grado)) deallocate(frames%grado) ! delete old data if present
       if (associated(frames%gradr)) deallocate(frames%gradr)
       nullify(frames%gradrx)
       nullify(frames%gradry)
       nullify(frames%gradrz)
!
       allocate(frames%grado(cv%amap%last,frames%num_frames))
       allocate(frames%gradr(3,3,cv%amap%last,4,frames%num_frames)) ! some extra space in the middle for parallel comm. (see smcv_master)
       frames%gradrx => frames%gradr(:,:,:,1,:)
       frames%gradry => frames%gradr(:,:,:,2,:)
       frames%gradrz => frames%gradr(:,:,:,3,:)
! allocate(frames%gradrx(3,3,cv%amap%last,frames%num_frames))
! allocate(frames%gradry(3,3,cv%amap%last,frames%num_frames))
! allocate(frames%gradrz(3,3,cv%amap%last,frames%num_frames))
! frames%gradrx=0d0; frames%gradry=0d0; frames%gradrz=0d0
       frames%gradr=0d0
       frames_grad_initialized=.true.
       end subroutine frames_grad_init
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine frames_print_local(iunit)
! assume that unit is prepared
! NOTE that this is a local print!
! use stream
       use multicom_aux;
       use mpi
!
       integer iunit
! locals
       integer :: i, j, k
! character(len=len("FRAMES_PRINT_LOCAL>") ),parameter::whoami="FRAMES_PRINT_LOCAL>";!macro
!
       if (MPI_COMM_STRNG.eq.MPI_COMM_NULL) return
! do work
       do i=1, frames%num_frames
! write(iunit,'("% ", I3)') i
        write(iunit,'(9F11.5)') ((frames%r(j,k,i), j=1,3), k=1,3) ! print vectors (columns) in sequence
       enddo
       end subroutine frames_print_local
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine frames_print_global(iunit)
! assume that unit is prepared
! NOTE that this is a global print!
! use stream
       use multicom_aux;
       use mpi
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
! locals
       integer :: i, j, k, m, n
! character(len=80 fmt
       real(chm_real) :: rtemp(3,3,frames%num_frames,SIZE_STRNG) ! temporary array
       integer :: ierror
! character(len=len("FRAMES_PRINT_GLOBAL>") ),parameter::whoami="FRAMES_PRINT_GLOBAL>";!macro
!
       if (MPI_COMM_STRNG.eq.MPI_COMM_NULL) return
! do work
!
! replica heads gather all data on root replica
       n=frames%num_frames
       if (SIZE_STRNG.gt.1) then
        call MPI_GATHER(frames%r(:,:,1:n), &
     & 9*n,mpifloat, &
     & rtemp,9*n,mpifloat,0,MPI_COMM_STRNG, &
     & ierror)
       else
        rtemp(:,:,:,1)=frames%r(:,:,1:n)
       endif
!
       if (ME_STRNG.eq.0) then ! root replica writes
!
! write(fmt,'(I10)') 9*SIZE_STRNG
        do k=1, n
! write(iunit,'("% ", I3)') k
! write(iunit,'('//fmt//'F11.5)') ! if SIZE_STRNG large, may not be able to put all on one line (record too long)
         write(iunit,'(9F11.5)') &
     & (((rtemp(i,j,k,m),i=1,3),j=1,3),m=1,SIZE_STRNG) ! sequentially write vectors of all replicas in one line
        enddo
       endif ! me_strng
       end subroutine frames_print_global
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine frames_read_local(iunit)
! assume that unit is prepared
! NOTE that this is a local read!
! use stream
       use multicom_aux;
       use mpi
!
       integer iunit
! locals
       integer :: i, j, k
       character(len=len("FRAMES_READ_LOCAL>") ),parameter::whoami="FRAMES_READ_LOCAL>";!macro
! do work
       if (MPI_COMM_STRNG.ne.MPI_COMM_NULL) then
! do work
        do i=1, frames%num_frames
         read(iunit,*) ((frames%r(j,k,i), j=1,3), k=1,3) ! read vectors (columns) in sequence
        enddo
       endif
       end subroutine frames_read_local
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine frames_read_global(iunit)
! assume that unit is prepared
! NOTE that this is a global read!
       use stream
       use multicom_aux;
       use mpi
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
! locals
       integer :: i, j, k, m, n
       real(chm_real) :: rtemp(3,3,frames%num_frames,SIZE_STRNG) ! temporary array
       integer :: ierror
       character(len=len("FRAMES_READ_GLOBAL>") ),parameter::whoami="FRAMES_READ_GLOBAL>";!macro
! do work
!
       if (MPI_COMM_STRNG.ne.MPI_COMM_NULL) then
        n=frames%num_frames ! number of frames
        if (ME_STRNG.eq.0) then ! root reads
         do k=1,n
          read(iunit,*) &
     & (((rtemp(i,j,k,m),i=1,3),j=1,3),m=1,SIZE_STRNG) ! sequentially read vectors (columns) of all replicas from one line
         enddo
        endif ! whoiam
! scatter all data
        if (SIZE_STRNG.gt.1) then
         call MPI_SCATTER(rtemp,9*n,mpifloat, &
     & frames%r(:,:,1:n),9*n,mpifloat,0,MPI_COMM_STRNG, &
     & ierror)
        else
         frames%r(:,:,1:n)=rtemp(:,:,:,1)
        endif
       endif
!
       end subroutine frames_read_global
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine frames_reset_calculate(grad,i)
       use stream
       integer, optional :: i
       logical :: grad
!
       character(len=len("FRAMES_RESET_CALCULATE>") ),parameter::whoami="FRAMES_RESET_CALCULATE>";!macro
!
! check for initialization
       if (.not.frames_initialized) then
! call wrndie(0,whoami,trim('NO FRAMES DEFINED. NOTHING DONE.'))
        return
       endif
!
       if (present(i)) then ! reset ith frame
! check frame number:
        if (i.lt.1.or.i.gt.frames%num_frames) then
         call wrndie(0,whoami,trim('OUT OF BOUNDS. NOTHING DONE.'))
         return
        endif
        frames%recalculate(i)=.true.
        if (grad) frames%recalculate_grad(i)=.true.
       else ! reset all
        frames%recalculate=.true.
        if (grad) frames%recalculate_grad=.true.
       endif
       end subroutine frames_reset_calculate
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! subroutine which computes frames from the x-set consistently with the xcomp-set
! necessary for the correct permutation of the local coordinate system frame
! if such a permutation is not ensured, CV values will be computed inconsistently
! uses other routines in this file + least-squares alignment (rtmd routines)
!
! this routine need not be parallelized, since it should only be used interactively
! note for parallelization: assume that all x,y,z values are correct
!
       subroutine frames_calc_align_comp(i,x,y,z, &
     & xcomp,ycomp,zcomp,mass,d)
       use bestfit, only : eig3s, RMSBestFit, rmsd, norm3, veccross3
       use stream
! vars
       integer :: i
       real(chm_real) :: x(:), y(:), z(:)
       real(chm_real) :: xcomp(:), ycomp(:), zcomp(:), mass(:)
       logical, optional :: d
! locals
       logical :: deriv
       integer, pointer, dimension(:) :: ind1
       real(chm_real), pointer, dimension(:) :: w
       real(chm_real), pointer, dimension(:,:) :: x1, x2
       real(chm_real) :: U(3,3), A1(3,3), A2(3,3), A3(3,3), A4(3,3), B(3,3)
       real(chm_real) :: corr1, corr2, corr3, corr4, cmax, wsum
!
       integer :: ncom1, ind, j, ii, jj
!
       character(len=len("FRAMES_CALC_ALIGN_COMP>") ),parameter::whoami="FRAMES_CALC_ALIGN_COMP>";!macro
! check for initialization
       if (.not.frames_initialized) then
        call wrndie(0,whoami,trim(' NO FRAMES DEFINED. NOTHING DONE.'))
        return
       endif
!
! check frame number:
       if (i.lt.1.or.i.gt.frames%num_frames) then
        call wrndie(0,whoami,trim(' OUT OF BOUNDS. NOTHING DONE.'))
        return
       endif
!
       if (present(d)) then ; deriv=d ; else ; deriv=.true. ; endif
! force computation of frame/derivative (override check flags)
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! do work:
! get atoms (adopted from frames_calc)
       ncom1=frames%priv(i)%p(1)
       allocate(ind1(ncom1),x1(ncom1,3),x2(ncom1,3))
       allocate(w(ncom1));
       ii=2; jj=ii+ncom1-1; ind1=frames%priv(i)%p(ii:jj)
       wsum=0d0;
       do j=1, ncom1;
        ind=cv%amap%i(ind1(j))
        x1(j,1)=x(ind); x1(j,2)=y(ind); x1(j,3)=z(ind);
        x2(j,1)=xcomp(ind); x2(j,2)=ycomp(ind); x2(j,3)=zcomp(ind);
        w(j)=mass(ind);
        wsum=wsum+w(j)
       enddo
       if (wsum.gt.1e-9) w=w/wsum ! normalize weights; this is NOT done by best_fit routine
! write(6,*) 'inside frames_align_comp'
! stop
! now we have all the coordinates for this frame; perform alignment
       call RMSBestFit(x1,x2,w,U); ! need rotation matrix !
! compute two sets of frames, based on xcomp & x; then use the matrix U
! to pick the `best` one
       call frames_reset_calculate(deriv,i)
       call frames_calc(i,xcomp,ycomp,zcomp,mass,deriv)
       B=frames%r(:,:,i); ! save frame
!
       call frames_reset_calculate(deriv,i)
       call frames_calc(i,x,y,z,mass,deriv)
       A1=frames%r(:,:,i); ! save frame
!cccc generate equivalent axes (assuming no eigenvalue degeneracy)
!cccc I am also assuming that frames_calc will generate the correct handedness (this is NOT guaranteed currently [see above]);
       A2(:,1)=-A1(:,1); A2(:,2)=-A1(:,2); A2(:,3)= A1(:,3);
       A3(:,1)=-A1(:,1); A3(:,2)= A1(:,2); A3(:,3)=-A1(:,3);
       A4(:,1)= A1(:,1); A4(:,2)=-A1(:,2); A4(:,3)=-A1(:,3);
! now permute axes
       corr1=sum(B*matmul(U,A1));
       corr2=sum(B*matmul(U,A2));
       corr3=sum(B*matmul(U,A3));
       corr4=sum(B*matmul(U,A4));
       cmax=max(corr1, corr2, corr3, corr4);
       if (cmax.eq.corr1) then; ! frames%r(:,:,i)=A1; ! nothing: the equality holds already
       elseif (cmax.eq.corr2) then; frames%r(:,:,i)=A2;
       elseif (cmax.eq.corr3) then; frames%r(:,:,i)=A3;
       elseif (cmax.eq.corr4) then; frames%r(:,:,i)=A4;
       endif
! recompute frame (& possibly derivatives); this is OK because frames_calc remembers last permutation
       call frames_reset_calculate(.true.,i)
       call frames_calc(i,x,y,z,mass,deriv)
! deallocate vars
       deallocate(ind1, x1, x2, w)
!
       end subroutine frames_calc_align_comp
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! moved to sm_util to avoid circular dependency problem
! subroutine frames_align_string(x,y,z,mass,min_rmsd,ind)
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
#endif
#endif /* automatically protect all code */
      end module cv_frames
!
