! ********************************************************************************!
! Preprocessed by GNU CPP !
! ********************************************************************************!
!
! CV_QUATERNION.MOD
!
! QUATERNION REPRESENTATION OF ROTATION BETWEEN TWO quat (DEPENDS ON FRAME MODULE)
      module cv_quaternion
#if (KEY_STRINGM==1) /*  automatically protect all code */
!
#if (KEY_STRINGM==1)
!
      use ivector
      use cv_common, only: cv, priv_ptr
      use cv_frames
!
      implicit none
      private
!
      type cv_quat
      real(chm_real), dimension(:,:), pointer :: q ! quaternion components
      real(chm_real), dimension(:,:,:,:), pointer :: gradq ! combined gradient array with some `extra` space; parallelization efficiency
      real(chm_real), dimension(:,:,:), pointer :: gradqx ! gradient vector of frame vectors w.r.t. x ;
      real(chm_real), dimension(:,:,:), pointer :: gradqy !
      real(chm_real), dimension(:,:,:), pointer :: gradqz !
      logical, dimension(:), pointer :: recalculate ! flag that indicates that op values should be recalculated
      logical, dimension(:), pointer :: recalculate_grad ! flag that indicates that op gradients should be recalculated
      type (priv_ptr), dimension(:), pointer :: priv ! `private` to each quaternion
      integer :: num_quat=0 ! number of active quaternions
      end type cv_quat
!
      ! subroutines
      public quat_init ! initialize quaternion arrays
      public quat_done ! destroy all quaternions
      public quat_add ! add a quaternnion
      public quat_calc ! calculate quaternions & their gradients w.r.t. atom positions
      public quat_list ! list quaternions
      public quat_grad_init ! initialize quat%grad arrays
      public quat_print_local ! print out quaternion values to separate files
      public quat_print_global ! print out quaternions values to a combined file
      public quat_reset_calculate ! when called, resets 'recalculate' flags; will be called from e.g. smcv_master
!
      ! variables
      type (cv_quat), public, save :: quat
      logical, public, save :: quat_initialized=.false., &
     & quat_grad_initialized=.false. ! have the cv%grad arrays been allocated
!
      ! parameters
      integer, parameter, public :: max_quat=20
!
      contains
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine quat_init()
!
       integer :: i
!
       if (.not.quat_initialized) then
!
        quat%num_quat=0
!
        allocate(quat%q(4,max_quat))
        allocate(quat%priv(max_quat)) ! allocate private pointer array
        allocate(quat%recalculate(max_quat))
        allocate(quat%recalculate_grad(max_quat))
        do i=1, max_quat
         nullify(quat%priv(i)%p) ! initialize integer pointer array
         nullify(quat%priv(i)%pr) ! initialize real pointer array
         nullify(quat%priv(i)%amap_ptr) ! nullify pointer to atom map (see above)
        enddo
        quat_initialized=.true.
        quat%q=0d0; quat%q(1,:)=1d0;
        quat%recalculate=.true.
        quat%recalculate_grad=.true.
       endif
       end subroutine quat_init
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine quat_done()
       integer :: i
       quat%num_quat=0
       if (quat_initialized) then
        deallocate(quat%q)
!
        do i=1, max_quat
         if (associated(quat%priv(i)%p)) &
     & deallocate(quat%priv(i)%p) ! free private memory
         if (associated(quat%priv(i)%amap_ptr)) &
     & deallocate(quat%priv(i)%amap_ptr) ! free private memory
         if (associated(quat%priv(i)%pr)) &
     & deallocate(quat%priv(i)%pr) ! free private memory
        enddo
!
        if (associated(quat%gradq)) deallocate(quat%gradq)
        nullify(quat%gradqx)
        nullify(quat%gradqy)
        nullify(quat%gradqz)
! if (associated(quat%gradqx)) deallocate(quat%gradqx)
! if (associated(quat%gradqy)) deallocate(quat%gradqy)
! if (associated(quat%gradqz)) deallocate(quat%gradqz)
        if (associated(quat%recalculate)) &
     & deallocate(quat%recalculate)
        if (associated(quat%recalculate_grad)) &
     & deallocate(quat%recalculate_grad)
        quat_initialized=.false.
        quat_grad_initialized=.false.
       endif
       end subroutine quat_done
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       function quat_add(fr1, fr2)
       use stream
       use number
!
       integer :: fr1, fr2
! locals
       integer :: j, l, num_int
       integer :: i, ii, jj, ncomf
       integer, pointer :: indf(:)
       integer :: f1, f2
       logical :: found
       integer :: quat_add
       type (int_vector) :: unique_amap_ptr ! contains a unique list of atom map pointers (to speed up gradient computation)
!
       character(len=len("QUAT_ADD>") ),parameter::whoami="QUAT_ADD>";!macro
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
       if ( f1.eq.f2 ) then
        call wrndie(0,whoami,trim(' ERROR: FRAMES CANNOT BE THE SAME. NOTHING DONE.'))
        return
       endif
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! check for duplicate quaternion
       found=.false.
       do l=1, quat%num_quat
        found=( (f1.eq.quat%priv(l)%p(1).and. &
     & f2.eq.quat%priv(l)%p(2)).or. &
! &
     & (f1.eq.quat%priv(l)%p(2).and. & ! frames interchanged (basically a quaternion and its inverse are considered equivalent)
     & f2.eq.quat%priv(l)%p(1)))
!
        if (found) exit
       enddo
!
       if (.not.found) then ! (if found -- do nothing)
        if (.not.quat_initialized) call quat_init()
        l=quat%num_quat + 1
        if (l.le.max_quat) then
         quat%num_quat=l
! allocate private data
! space needed:
         num_int = 2 ! number of ints needed for storage: just two frame indices
!
         allocate(quat%priv(l)%p(num_int));
         quat%priv(l)%p(1)=f1
         quat%priv(l)%p(2)=f2
         quat_add=l
         quat%q(:,l)=anum ! undefined value
! add pointers to indices in the atom map (faster gradient computation)
         ncomf=frames%priv(f1)%p(1);
         allocate(indf(ncomf));
         ii=2; jj=ii+ncomf-1;
         indf=frames%priv(f1)%p(ii:jj)
         do j=1, ncomf ! loop over frame atoms
          i=int_vector_uadd(unique_amap_ptr,indf(j)) ! add index of the frame atom to the list of unique atom labels
         enddo
         deallocate(indf)
!
         ncomf=frames%priv(f2)%p(1);
         allocate(indf(ncomf));
         ii=2; jj=ii+ncomf-1;
         indf=frames%priv(f2)%p(ii:jj)
         do j=1, ncomf ! loop over frame atoms
          i=int_vector_uadd(unique_amap_ptr,indf(j)) ! add index of the frame atom to the list of unique atom labels
         enddo
         deallocate(indf)
!
         i=unique_amap_ptr%last ! number of unique atoms this cv depends on
         allocate(quat%priv(l)%amap_ptr(i+1)) ! add one to include length of list
         quat%priv(l)%amap_ptr(1)=i;
         quat%priv(l)%amap_ptr(2:i+1)=unique_amap_ptr%i(1:i)
         call int_vector_done(unique_amap_ptr)
!
        else ! out of bounds
         call wrndie(0,whoami,trim(' ERROR ADDING QUATERNION. NOTHING DONE.'))
         quat_add=0
        endif
       else ! found
         call wrndie(0,whoami,trim(' EQUIVALENT QUATERNION ALREADY PRESENT. NOTHING DONE.'))
         quat_add=0
       endif
       end function quat_add
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine quat_list(i)
       use stream
       use multicom_aux;
       use mpi
!
 character(len=132)::info(21)=(/'','','','','','','','','','','','','','','','','','','','',''/);! output buffer
!
       integer :: i
!
       integer :: f1, f2
       character(len=len("CV_QUAT_LIST>") ),parameter::whoami="CV_QUAT_LIST>";!macro
!
!
       if (MPI_COMM_STRNG.eq.MPI_COMM_NULL) return ! only replica heads stay
!
       if (ME_STRNG.eq.0) then
        f1=quat%priv(i)%p(1)
        f2=quat%priv(i)%p(2)
!
        write(info,'(A)') '\t ORIENTATION QUATERNION BETWEEN TWO FRAMES' ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
        write(info,'(A,I5)') '\t FRAME: ',f1 ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
        write(info,'(A,I5)') '\t FRAME: ',f2 ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
       endif
!
       end subroutine quat_list
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine quat_calc(i,x,y,z,mass,deriv)
       use stream
       use consta
       use sm_var
!
       integer :: i ! which quaternion to calculate
       real(chm_real) :: x(:), y(:), z(:), mass(:)
       logical :: deriv
!
       real(chm_real) :: A(3,3), B(3,3), R(3,3) ! transformation matrices (frame vectors)
       real(chm_real) :: dA(3,3) ! matrix derivatives
       real(chm_real) :: qq(4) ! temporary quaternioin
       integer :: f1, f2, ii, j, jj, ncomf, ind
       integer, pointer, dimension(:) :: indf
       real(chm_real) :: a1, a2, trace, tr, dtr, s, s2
       logical :: qframe1, qframe2
       real(chm_real) :: RRm32, RRm13, RRm21, RRp12, RRp13, RRp23
       integer, pointer :: priv(:)
!
       real(chm_real), parameter :: tol=1e-2
! real(chm_real), parameter :: tol=0d0
!
! aardvark : FD
! real(chm_real) :: h, aa(3,3)
       character(len=len("QUAT_CALC>") ),parameter::whoami="QUAT_CALC>";!macro
!
! check for initialization
       if (.not.quat_initialized) then
        call wrndie(0,whoami,trim('NO QUATERNIONS DEFINED. NOTHING DONE.'))
        return
       endif
!
! check frame number:
       if (i.lt.1.or.i.gt.quat%num_quat) then
        call wrndie(0,whoami,trim('OUT OF BOUNDS. NOTHING DONE.'))
        return
       endif
!
! check flags: do work only if derivatives are unknown
       if (.not.(quat%recalculate(i).or. &
     & (quat%recalculate_grad(i).and.deriv))) &
     & return
! do work:
! extract frames
       f1=quat%priv(i)%p(1); f2=quat%priv(i)%p(2);
       qframe1=f1.gt.0; qframe2=f2.gt.0;
! calculate quaternion from the transformation matrix that takes frame 2[B] into frame 1[A]:
! 1: get transformation matrix
       if (qframe1) then
        call frames_calc(f1,x,y,z,mass,deriv)
        A=transpose(frames%r(:,:,f1))
       else
        A=Id3
       endif
!
       if (qframe2) then
        call frames_calc(f2,x,y,z,mass,deriv)
        B=frames%r(:,:,f2)
       else
        B=Id3
       endif
       R=matmul(A,B);
       RRm32=R(3,2)-R(2,3)
       RRm13=R(1,3)-R(3,1)
       RRm21=R(2,1)-R(1,2)
!
       RRp12=R(1,2)+R(2,1)
       RRp13=R(1,3)+R(3,1)
       RRp23=R(2,3)+R(3,2)
!
!
! compute quaternion: experimental code
! Martin Baker's quaternion page: Christian's post
!
! qq(1) = 0.5d0*sqrt( max( 0d0, 1d0 + R(1,1) + R(2,2) + R(3,3) ) )
! qq(2) = 0.5d0*sqrt( max( 0d0, 1d0 + R(1,1) - R(2,2) - R(3,3) ) )
! qq(3) = 0.5d0*sqrt( max( 0d0, 1d0 - R(1,1) + R(2,2) - R(3,3) ) )
! qq(4) = 0.5d0*sqrt( max( 0d0, 1d0 - R(1,1) - R(2,2) + R(3,3) ) )
! qq(2) = sign( qq(2), RRm32 )
! qq(3) = sign( qq(3), RRm13 )
! qq(4) = sign( qq(4), RRm21 )
!
! if (.false.) then ! skip
! 2 compute quaternion: (standard code)
       trace=1d0+R(1,1)+R(2,2)+R(3,3) ! trace (in graphics, the matrix is 4x4 so trace includes 1.0)
! if (.false.) then
! if (.true.) then
       if (trace.gt.tol) then
        tr=trace
        s=0.5d0/sqrt(tr)
! note: the notation is: 1-w, 2-x, 3-y, 4-z; where w is cos(theta) and x=sin(theta)*r_x, etc.
        qq(1)=s*tr
        qq(2)=s*(R(3,2)-R(2,3));
        qq(3)=s*(R(1,3)-R(3,1));
        qq(4)=s*(R(2,1)-R(1,2));
! elseif ( .false.) then
! elseif ( .true.) then
       elseif ( (R(1,1).gt.R(2,2)) .and. (R(1,1).gt.R(3,3)) ) then
        tr=1d0 + R(1,1) - R(2,2) - R(3,3) ! not the trace anymore, but reuse tr
        s=0.5d0/sqrt(tr)
        qq(1)=s*(R(3,2)-R(2,3));
        qq(2)=s*tr
        qq(3)=s*(R(1,2)+R(2,1));
        qq(4)=s*(R(1,3)+R(3,1));
! elseif ( .true.) then
! elseif ( .false.) then
       elseif ( R(2,2) .gt. R(3,3) ) then
        tr=1d0 + R(2,2) - R(1,1) - R(3,3)
        s=0.5d0/sqrt(tr)
        qq(1)=s*(R(1,3)-R(3,1));
        qq(2)=s*(R(1,2)+R(2,1));
        qq(3)=s*tr
        qq(4)=s*(R(2,3)+R(3,2));
       else
        tr=1d0 + R(3,3) - R(1,1) - R(2,2)
        s=0.5d0/sqrt(tr)
        qq(1)=s*(R(2,1)-R(1,2));
        qq(2)=s*(R(1,3)+R(3,1));
        qq(3)=s*(R(2,3)+R(3,2));
        qq(4)=s*tr
       endif
! endif ! skip
! invert so that `dot` product (cosine of angle) with the previous value is positive
! adhoc
       if ( qq(1)*quat%q(1,i)+ &
     & qq(2)*quat%q(2,i)+ &
     & qq(3)*quat%q(3,i)+ &
     & qq(4)*quat%q(4,i) .lt. 0) then
        quat%q(:,i)=-qq
        s=-s
       else
        quat%q(:,i)=qq
       endif
! the quaternion should be normalized; check: OK
       quat%recalculate(i)=.false.
!
!cccccccccccccccccc now compute derivatives cccccccccccccccccccc
       if (deriv.and.quat%recalculate_grad(i)) then
! check for gradient initialization
        if (.not.quat_grad_initialized) call quat_grad_init()
! initialize derivative arrays
! do j=1,cv%amap%last
        priv=>quat%priv(i)%amap_ptr
        do jj=2,priv(1)+1
         j=priv(jj)
         do ii=1,4
          quat%gradqx(ii,j,i)=0d0;
          quat%gradqy(ii,j,i)=0d0;
          quat%gradqz(ii,j,i)=0d0
         enddo
        enddo
!
! trace=1d0+R(1,1)+R(2,2)+R(3,3) ! already known
        s2=s*s
! if (.true.) then
! if (.false.) then
        if (trace.gt.tol) then ! same four cases as above
! tr=trace
! s=0.5d0/sqrt(tr);
! s2=s*s
!
         if (qframe1) then ! compute contribution from first frame
! extract frame indices (need to compute gradients w.r.t atoms that define the frame!)
          ncomf=frames%priv(f1)%p(1);
          allocate(indf(ncomf));
          ii=2; jj=ii+ncomf-1; indf=frames%priv(f1)%p(ii:jj)
          do j=1, ncomf ! loop over frame atoms
           ind=indf(j)
! gradx
           dA=matmul(transpose(frames%gradrx(:,:,ind,f1)), B)
           dtr=dA(1,1)+dA(2,2)+dA(3,3)
!
           a1=0.5d0*s*dtr
           a2=-4d0*s2*a1
!
           quat%gradqx(1,ind,i)=a1
           quat%gradqx(2,ind,i)=a2*RRm32+s*(dA(3,2)-dA(2,3))
           quat%gradqx(3,ind,i)=a2*RRm13+s*(dA(1,3)-dA(3,1))
           quat%gradqx(4,ind,i)=a2*RRm21+s*(dA(2,1)-dA(1,2))
! grady
           dA=matmul(transpose(frames%gradry(:,:,ind,f1)), B)
           dtr=dA(1,1)+dA(2,2)+dA(3,3)
!
           a1=0.5d0*s*dtr
           a2=-4d0*s2*a1
!
           quat%gradqy(1,ind,i)=a1
           quat%gradqy(2,ind,i)=a2*RRm32+s*(dA(3,2)-dA(2,3))
           quat%gradqy(3,ind,i)=a2*RRm13+s*(dA(1,3)-dA(3,1))
           quat%gradqy(4,ind,i)=a2*RRm21+s*(dA(2,1)-dA(1,2))

! gradz
           dA=matmul(transpose(frames%gradrz(:,:,ind,f1)), B)
           dtr=dA(1,1)+dA(2,2)+dA(3,3)
!
           a1=0.5d0*s*dtr
           a2=-4d0*s2*a1
!
           quat%gradqz(1,ind,i)=a1
           quat%gradqz(2,ind,i)=a2*RRm32+s*(dA(3,2)-dA(2,3))
           quat%gradqz(3,ind,i)=a2*RRm13+s*(dA(1,3)-dA(3,1))
           quat%gradqz(4,ind,i)=a2*RRm21+s*(dA(2,1)-dA(1,2))
          enddo
          deallocate(indf)
         endif ! qframe1
!
         if (qframe2) then ! compute contribution from second frame
! extract frame indices (need to compute gradients w.r.t atoms that define the frame!)
          ncomf=frames%priv(f2)%p(1);
          allocate(indf(ncomf));
          ii=2; jj=ii+ncomf-1; indf=frames%priv(f2)%p(ii:jj)

          do j=1, ncomf ! loop over frame atoms
           ind=indf(j)
! gradx
           dA=matmul(A,frames%gradrx(:,:,ind,f2))
           dtr=dA(1,1)+dA(2,2)+dA(3,3)
!
           a1=0.5d0*s*dtr
           a2=-4d0*s2*a1
!
           quat%gradqx(1,ind,i)=a1
           quat%gradqx(2,ind,i)=a2*RRm32+s*(dA(3,2)-dA(2,3))
           quat%gradqx(3,ind,i)=a2*RRm13+s*(dA(1,3)-dA(3,1))
           quat%gradqx(4,ind,i)=a2*RRm21+s*(dA(2,1)-dA(1,2))
! grady
           dA=matmul(A,frames%gradry(:,:,ind,f2))
           dtr=dA(1,1)+dA(2,2)+dA(3,3)
!
           a1=0.5d0*s*dtr
           a2=-4d0*s2*a1
!
           quat%gradqy(1,ind,i)=a1
           quat%gradqy(2,ind,i)=a2*RRm32+s*(dA(3,2)-dA(2,3))
           quat%gradqy(3,ind,i)=a2*RRm13+s*(dA(1,3)-dA(3,1))
           quat%gradqy(4,ind,i)=a2*RRm21+s*(dA(2,1)-dA(1,2))

! gradz
           dA=matmul(A,frames%gradrz(:,:,ind,f2))
           dtr=dA(1,1)+dA(2,2)+dA(3,3)
!
           a1=0.5d0*s*dtr
           a2=-4d0*s2*a1
!
           quat%gradqz(1,ind,i)=a1
           quat%gradqz(2,ind,i)=a2*RRm32+s*(dA(3,2)-dA(2,3))
           quat%gradqz(3,ind,i)=a2*RRm13+s*(dA(1,3)-dA(3,1))
           quat%gradqz(4,ind,i)=a2*RRm21+s*(dA(2,1)-dA(1,2))
          enddo
          deallocate(indf)
         endif ! qframe2
!
! elseif ( .false.) then
! elseif ( .true.) then
        elseif ( (R(1,1).gt.R(2,2)) .and. (R(1,1).gt.R(3,3)) ) then
! tr=1d0 + R(1,1) - R(2,2) - R(3,3) ! already known
! s=0.5d0/sqrt(tr)
! s2=s*s
!
         if (qframe1) then ! compute contribution from first frame
! extract frame indices (need to compute gradients w.r.t atoms that define the frame!)
          ncomf=frames%priv(f1)%p(1);
          allocate(indf(ncomf));
          ii=2; jj=ii+ncomf-1; indf=frames%priv(f1)%p(ii:jj)
!
          do j=1, ncomf ! loop over frame atoms
           ind=indf(j)
! gradx
           dA=matmul(transpose(frames%gradrx(:,:,ind,f1)), B)
           dtr=dA(1,1)-dA(2,2)-dA(3,3)
!
           a1=0.5d0*s*dtr
           a2=-4d0*s2*a1
!
           quat%gradqx(1,ind,i)=a2*RRm32+s*(dA(3,2)-dA(2,3))
           quat%gradqx(2,ind,i)=a1
           quat%gradqx(3,ind,i)=a2*RRp12+s*(dA(1,2)+dA(2,1))
           quat%gradqx(4,ind,i)=a2*RRp13+s*(dA(1,3)+dA(3,1))
! grady
           dA=matmul(transpose(frames%gradry(:,:,ind,f1)), B)
           dtr=dA(1,1)-dA(2,2)-dA(3,3)
!
           a1=0.5d0*s*dtr
           a2=-4d0*s2*a1
!
           quat%gradqy(1,ind,i)=a2*RRm32+s*(dA(3,2)-dA(2,3))
           quat%gradqy(2,ind,i)=a1
           quat%gradqy(3,ind,i)=a2*RRp12+s*(dA(1,2)+dA(2,1))
           quat%gradqy(4,ind,i)=a2*RRp13+s*(dA(1,3)+dA(3,1))
! gradz
           dA=matmul(transpose(frames%gradrz(:,:,ind,f1)), B)
           dtr=dA(1,1)-dA(2,2)-dA(3,3)
!
           a1=0.5d0*s*dtr
           a2=-4d0*s2*a1
!
           quat%gradqz(1,ind,i)=a2*RRm32+s*(dA(3,2)-dA(2,3))
           quat%gradqz(2,ind,i)=a1
           quat%gradqz(3,ind,i)=a2*RRp12+s*(dA(1,2)+dA(2,1))
           quat%gradqz(4,ind,i)=a2*RRp13+s*(dA(1,3)+dA(3,1))
          enddo
          deallocate(indf)
         endif
!
         if (qframe2) then ! compute contribution from first frame
! extract frame indices (need to compute gradients w.r.t atoms that define the frame!)
          ncomf=frames%priv(f2)%p(1);
          allocate(indf(ncomf));
          ii=2; jj=ii+ncomf-1; indf=frames%priv(f2)%p(ii:jj)
!
          do j=1, ncomf ! loop over frame atoms
           ind=indf(j)
! gradx
           dA=matmul(A,frames%gradrx(:,:,ind,f2))
           dtr=dA(1,1)-dA(2,2)-dA(3,3)
!
           a1=0.5d0*s*dtr
           a2=-4d0*s2*a1
!
           quat%gradqx(1,ind,i)=a2*RRm32+s*(dA(3,2)-dA(2,3))
           quat%gradqx(2,ind,i)=a1
           quat%gradqx(3,ind,i)=a2*RRp12+s*(dA(1,2)+dA(2,1))
           quat%gradqx(4,ind,i)=a2*RRp13+s*(dA(1,3)+dA(3,1))
! grady
           dA=matmul(A,frames%gradry(:,:,ind,f2))
           dtr=dA(1,1)-dA(2,2)-dA(3,3)
!
           a1=0.5d0*s*dtr
           a2=-4d0*s2*a1
!
           quat%gradqy(1,ind,i)=a2*RRm32+s*(dA(3,2)-dA(2,3))
           quat%gradqy(2,ind,i)=a1
           quat%gradqy(3,ind,i)=a2*RRp12+s*(dA(1,2)+dA(2,1))
           quat%gradqy(4,ind,i)=a2*RRp13+s*(dA(1,3)+dA(3,1))
! gradz
           dA=matmul(A,frames%gradrz(:,:,ind,f2))
           dtr=dA(1,1)-dA(2,2)-dA(3,3)
!
           a1=0.5d0*s*dtr
           a2=-4d0*s2*a1
!
           quat%gradqz(1,ind,i)=a2*RRm32+s*(dA(3,2)-dA(2,3))
           quat%gradqz(2,ind,i)=a1
           quat%gradqz(3,ind,i)=a2*RRp12+s*(dA(1,2)+dA(2,1))
           quat%gradqz(4,ind,i)=a2*RRp13+s*(dA(1,3)+dA(3,1))
          enddo
          deallocate(indf)
         endif ! qframe2
! elseif ( .true.) then
! elseif ( .false.) then
        elseif ( R(2,2) .gt. R(3,3) ) then
! tr=1d0 + R(2,2) - R(1,1) - R(3,3) ! not the trace anymore, but reuse tr
! s=0.5d0/sqrt(tr) ! already known
! s2=s*s
!
         if (qframe1) then ! compute contribution from first frame
! extract frame indices (need to compute gradients w.r.t atoms that define the frame!)
          ncomf=frames%priv(f1)%p(1);
          allocate(indf(ncomf));
          ii=2; jj=ii+ncomf-1; indf=frames%priv(f1)%p(ii:jj)
!
          do j=1, ncomf ! loop over frame atoms
           ind=indf(j)
! gradx
           dA=matmul(transpose(frames%gradrx(:,:,ind,f1)), B)
           dtr=dA(2,2)-dA(1,1)-dA(3,3)
!
           a1=0.5d0*s*dtr
           a2=-4d0*s2*a1
!
           quat%gradqx(1,ind,i)=a2*RRm13+s*(dA(1,3)-dA(3,1))
           quat%gradqx(2,ind,i)=a2*RRp12+s*(dA(1,2)+dA(2,1))
           quat%gradqx(3,ind,i)=a1
           quat%gradqx(4,ind,i)=a2*RRp23+s*(dA(2,3)+dA(3,2))
! grady
           dA=matmul(transpose(frames%gradry(:,:,ind,f1)), B)
           dtr=dA(2,2)-dA(1,1)-dA(3,3)
!
           a1=0.5d0*s*dtr
           a2=-4d0*s2*a1
!
           quat%gradqy(1,ind,i)=a2*RRm13+s*(dA(1,3)-dA(3,1))
           quat%gradqy(2,ind,i)=a2*RRp12+s*(dA(1,2)+dA(2,1))
           quat%gradqy(3,ind,i)=a1
           quat%gradqy(4,ind,i)=a2*RRp23+s*(dA(2,3)+dA(3,2))
! gradz
           dA=matmul(transpose(frames%gradrz(:,:,ind,f1)), B)
           dtr=dA(2,2)-dA(1,1)-dA(3,3)
!
           a1=0.5d0*s*dtr
           a2=-4d0*s2*a1
!
           quat%gradqz(1,ind,i)=a2*RRm13+s*(dA(1,3)-dA(3,1))
           quat%gradqz(2,ind,i)=a2*RRp12+s*(dA(1,2)+dA(2,1))
           quat%gradqz(3,ind,i)=a1
           quat%gradqz(4,ind,i)=a2*RRp23+s*(dA(2,3)+dA(3,2))
          enddo
          deallocate(indf)
         endif
!
         if (qframe2) then ! compute contribution from first frame
! extract frame indices (need to compute gradients w.r.t atoms that define the frame!)
          ncomf=frames%priv(f2)%p(1);
          allocate(indf(ncomf));
          ii=2; jj=ii+ncomf-1; indf=frames%priv(f2)%p(ii:jj)
!
          do j=1, ncomf ! loop over frame atoms
           ind=indf(j)
! gradx
           dA=matmul(A,frames%gradrx(:,:,ind,f2))
           dtr=dA(2,2)-dA(1,1)-dA(3,3)
!
           a1=0.5d0*s*dtr
           a2=-4d0*s2*a1
!
           quat%gradqx(1,ind,i)=a2*RRm13+s*(dA(1,3)-dA(3,1))
           quat%gradqx(2,ind,i)=a2*RRp12+s*(dA(1,2)+dA(2,1))
           quat%gradqx(3,ind,i)=a1
           quat%gradqx(4,ind,i)=a2*RRp23+s*(dA(2,3)+dA(3,2))
! grady
           dA=matmul(A,frames%gradry(:,:,ind,f2))
           dtr=dA(2,2)-dA(1,1)-dA(3,3)
!
           a1=0.5d0*s*dtr
           a2=-4d0*s2*a1
!
           quat%gradqy(1,ind,i)=a2*RRm13+s*(dA(1,3)-dA(3,1))
           quat%gradqy(2,ind,i)=a2*RRp12+s*(dA(1,2)+dA(2,1))
           quat%gradqy(3,ind,i)=a1
           quat%gradqy(4,ind,i)=a2*RRp23+s*(dA(2,3)+dA(3,2))
! gradz
           dA=matmul(A,frames%gradrz(:,:,ind,f2))
           dtr=dA(2,2)-dA(1,1)-dA(3,3)
!
           a1=0.5d0*s*dtr
           a2=-4d0*s2*a1
!
           quat%gradqz(1,ind,i)=a2*RRm13+s*(dA(1,3)-dA(3,1))
           quat%gradqz(2,ind,i)=a2*RRp12+s*(dA(1,2)+dA(2,1))
           quat%gradqz(3,ind,i)=a1
           quat%gradqz(4,ind,i)=a2*RRp23+s*(dA(2,3)+dA(3,2))
          enddo
          deallocate(indf)
         endif ! qframe2
!
        else
! tr=1d0 + R(3,3) - R(1,1) - R(2,2) ! not the trace anymore, but reuse tr
! s=0.5d0/sqrt(tr)
! s2=s*s
!
         if (qframe1) then ! compute contribution from first frame
! extract frame indices (need to compute gradients w.r.t atoms that define the frame!)
          ncomf=frames%priv(f1)%p(1);
          allocate(indf(ncomf));
          ii=2; jj=ii+ncomf-1; indf=frames%priv(f1)%p(ii:jj)
!
          do j=1, ncomf ! loop over frame atoms
           ind=indf(j)
! gradx
           dA=matmul(transpose(frames%gradrx(:,:,ind,f1)), B)
           dtr=dA(3,3)-dA(1,1)-dA(2,2)
!
           a1=0.5d0*s*dtr
           a2=-4d0*s2*a1
!
           quat%gradqx(1,ind,i)=a2*RRm21+s*(dA(2,1)-dA(1,2))
           quat%gradqx(2,ind,i)=a2*RRp13+s*(dA(1,3)+dA(3,1))
           quat%gradqx(3,ind,i)=a2*RRp23+s*(dA(2,3)+dA(3,2))
           quat%gradqx(4,ind,i)=a1
! grady
           dA=matmul(transpose(frames%gradry(:,:,ind,f1)), B)
           dtr=dA(3,3)-dA(1,1)-dA(2,2)
!
           a1=0.5d0*s*dtr
           a2=-4d0*s2*a1
!
           quat%gradqy(1,ind,i)=a2*RRm21+s*(dA(2,1)-dA(1,2))
           quat%gradqy(2,ind,i)=a2*RRp13+s*(dA(1,3)+dA(3,1))
           quat%gradqy(3,ind,i)=a2*RRp23+s*(dA(2,3)+dA(3,2))
           quat%gradqy(4,ind,i)=a1
! gradz
           dA=matmul(transpose(frames%gradrz(:,:,ind,f1)), B)
           dtr=dA(3,3)-dA(1,1)-dA(2,2)
!
           a1=0.5d0*s*dtr
           a2=-4d0*s2*a1
!
           quat%gradqz(1,ind,i)=a2*RRm21+s*(dA(2,1)-dA(1,2))
           quat%gradqz(2,ind,i)=a2*RRp13+s*(dA(1,3)+dA(3,1))
           quat%gradqz(3,ind,i)=a2*RRp23+s*(dA(2,3)+dA(3,2))
           quat%gradqz(4,ind,i)=a1
          enddo
          deallocate(indf)
         endif
!
         if (qframe2) then ! compute contribution from first frame
! extract frame indices (need to compute gradients w.r.t atoms that define the frame!)
          ncomf=frames%priv(f2)%p(1);
          allocate(indf(ncomf));
          ii=2; jj=ii+ncomf-1; indf=frames%priv(f2)%p(ii:jj)
!
          do j=1, ncomf ! loop over frame atoms
           ind=indf(j)
! gradx
           dA=matmul(A,frames%gradrx(:,:,ind,f2))
           dtr=dA(3,3)-dA(1,1)-dA(2,2)
!
           a1=0.5d0*s*dtr
           a2=-4d0*s2*a1
!
           quat%gradqx(1,ind,i)=a2*RRm21+s*(dA(2,1)-dA(1,2))
           quat%gradqx(2,ind,i)=a2*RRp13+s*(dA(1,3)+dA(3,1))
           quat%gradqx(3,ind,i)=a2*RRp23+s*(dA(2,3)+dA(3,2))
           quat%gradqx(4,ind,i)=a1
! grady
           dA=matmul(A,frames%gradry(:,:,ind,f2))
           dtr=dA(3,3)-dA(1,1)-dA(2,2)
!
           a1=0.5d0*s*dtr
           a2=-4d0*s2*a1
!
           quat%gradqy(1,ind,i)=a2*RRm21+s*(dA(2,1)-dA(1,2))
           quat%gradqy(2,ind,i)=a2*RRp13+s*(dA(1,3)+dA(3,1))
           quat%gradqy(3,ind,i)=a2*RRp23+s*(dA(2,3)+dA(3,2))
           quat%gradqy(4,ind,i)=a1
! gradz
           dA=matmul(A,frames%gradrz(:,:,ind,f2))
           dtr=dA(3,3)-dA(1,1)-dA(2,2)
!
           a1=0.5d0*s*dtr
           a2=-4d0*s2*a1
!
           quat%gradqz(1,ind,i)=a2*RRm21+s*(dA(2,1)-dA(1,2))
           quat%gradqz(2,ind,i)=a2*RRp13+s*(dA(1,3)+dA(3,1))
           quat%gradqz(3,ind,i)=a2*RRp23+s*(dA(2,3)+dA(3,2))
           quat%gradqz(4,ind,i)=a1
          enddo
          deallocate(indf)
         endif
        endif
!
!cccccccccccccccccccccccccc done with derivatives cccccccccccccccccccccccc
        quat%recalculate_grad(i)=.false. ! indicate that derivatives are known
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!
       endif ! deriv
!
       end subroutine quat_calc
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine quat_grad_init()
! initialize quat%grad arrays
       if(associated(quat%gradq))deallocate(quat%gradq)
! if (associated(quat%gradqx)) deallocate(quat%gradqx)
! if (associated(quat%gradqy)) deallocate(quat%gradqy)
! if (associated(quat%gradqz)) deallocate(quat%gradqz)
!
       ! the xyz components are in 3rd dimension; keeping quat index outside for comm.; add some room on in the middle (+1)
       ! might make computation slightly slower (?), but we should win on communication: we will put q(1:4,:) into qrad(:,1,4,:) and send grad
       allocate(quat%gradq(4,cv%amap%last,3+1,quat%num_quat))
       quat%gradqx => quat%gradq(:,:,1,:)
       quat%gradqy => quat%gradq(:,:,2,:)
       quat%gradqz => quat%gradq(:,:,3,:)
! allocate(quat%gradqx(4,cv%amap%last,quat%num_quat))
! allocate(quat%gradqy(4,cv%amap%last,quat%num_quat))
! allocate(quat%gradqz(4,cv%amap%last,quat%num_quat))
! quat%gradqx=0d0; quat%gradqy=0d0; quat%gradqz=0d0
       quat%gradq=0d0
       quat_grad_initialized=.true.
       end subroutine quat_grad_init
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine quat_print_local(iunit)
! assume that unit is prepared
! NOTE that this is a local print!
! use stream
       use multicom_aux;
       use mpi
!
       integer :: iunit
! locals
       integer :: i, j
       character(len=len("QUAT_PRINT_LOCAL>") ),parameter::whoami="QUAT_PRINT_LOCAL>";!macro
!
       if (MPI_COMM_STRNG.eq.MPI_COMM_NULL) return
! do work
       do i=1, quat%num_quat
        write(iunit,'("% ", I3)') i
        write(iunit,'(4F11.5)') (quat%q(j,i), j=1,4)
       enddo
       end subroutine quat_print_local
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine quat_print_global(iunit)
! assume that unit is prepared
! NOTE that this is a global print!
! use stream
!
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
       integer iunit
! locals
       integer :: i, k, m, n
       character(len=80) :: fmt
       real(chm_real) :: rtemp(4,quat%num_quat,SIZE_STRNG) ! temporary array
       integer :: ierror
       character(len=len("QUAT_PRINT_GLOBAL>") ),parameter::whoami="QUAT_PRINT_GLOBAL>";!macro
!
       if (MPI_COMM_STRNG.eq.MPI_COMM_NULL) return
!
! do work
! gather all data on root
       n=quat%num_quat
       if (SIZE_STRNG.gt.1) then
        call MPI_GATHER(quat%q(:,1:n), &
     & 4*n,mpifloat, &
     & rtemp,4*n,mpifloat,0,MPI_COMM_STRNG, &
     & ierror)
       else
        rtemp(:,:,1)=quat%q(:,1:n)
       endif
!
       if (ME_STRNG.eq.0) then ! root replica writes
!
        write(fmt,'(I10)') 4*SIZE_STRNG
        do k=1, n
         write(iunit,'("% ", I3)') k
         write(iunit,'('//fmt//'F11.5)')                                &
     & ((rtemp(i,k,m),i=1,4),m=1,SIZE_STRNG) ! sequentially write vectors of all replicas in one line
        enddo
       endif ! ME_
       end subroutine quat_print_global
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine quat_reset_calculate(grad,i)
       use stream
       integer, optional :: i
       logical :: grad
       character(len=len("QUAT_RESET_CALCULATE>") ),parameter::whoami="QUAT_RESET_CALCULATE>";!macro
!
! check for initialization
       if (.not.quat_initialized) then
! call wrndie(0,whoami,trim('NO quat DEFINED. NOTHING DONE.'))
        return
       endif
!
       if (present(i)) then ! reset ith frame
! check frame number:
        if (i.lt.1.or.i.gt.quat%num_quat) then
         call wrndie(0,whoami,trim('OUT OF BOUNDS. NOTHING DONE.'))
         return
        endif
        quat%recalculate(i)=.true.
        if (grad) quat%recalculate_grad(i)=.true.
       else ! reset all
        quat%recalculate=.true.
        if (grad) quat%recalculate_grad=.true.
       endif
       end subroutine quat_reset_calculate
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
#endif
#endif /* automatically protect all code */
      end module cv_quaternion
!
