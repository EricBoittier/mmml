! ********************************************************************************!
! Preprocessed by GNU CPP !
! ********************************************************************************!
!
! CV_QCOMP.MOD
!
! QUATERNION COMPONENTS; USES CV_QUATERNION
      module cv_qcomp
#if (KEY_STRINGM==1) /*  automatically protect all code */
!
#if (KEY_STRINGM==1)
!
      use cv_common
      use cv_types
      use cv_quaternion
      use cv_frames
      use ivector
      use ivector_list
!
      implicit none
      private
      ! subroutines
      public cv_qcomp_add
      public cv_qcomp_calc
      public cv_qcomp_list
!
      contains
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       function cv_qcomp_add(type,fr1,fr2,k,gamma,weight) ! note: i is the atom index in the PSF
       use stream
 character(len=132)::info(21)=(/'','','','','','','','','','','','','','','','','','','','',''/);! output buffer
       integer :: type, fr1, fr2
       real(chm_real) :: k, gamma, weight
! locals
       integer :: i, j, l
       integer :: f1, f2, ii, jj, ncomf
       integer, pointer :: indf(:)
       logical :: found, cv_qcomp_add
       character(len=1) :: pos
!
       character(len=len("CV_QCOMP_ADD>") ),parameter::whoami="CV_QCOMP_ADD>";!macro
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
        call wrndie(0,whoami,trim(' FRAMES CANNOT BE THE SAME. NOTHING DONE.'))
        return
       endif
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! check that the type is valid
       select case (type)
        case (qcomp_1); pos='1'
        case (qcomp_2); pos='2'
        case (qcomp_3); pos='3'
        case (qcomp_4); pos='4'
        case default;
         call wrndie(0,whoami,trim('UNKNOWN CV TYPE. NOTHING DONE.'))
         return
       end select
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! check if the quaternion corresponding to f1 and f2 is present
       do j=1, quat%num_quat
        found=( (f1.eq.quat%priv(j)%p(1).and. &
     & f2.eq.quat%priv(j)%p(2)).or. &
! &
     & (f1.eq.quat%priv(j)%p(2).and. & ! frames interchanged (a quaternion and its additive inverse are considered equivalent)
     & f2.eq.quat%priv(j)%p(1)))
        if (found) exit
       enddo
!
! add quaternion, if necessary
       if ( .not. found ) then
         j=quat_add(f1, f2)
         if (j.eq.0) then
          call wrndie(0,whoami,trim(' COULD NOT ADD QUATERNION. NOTHING DONE.'))
          cv_qcomp_add=.false.
          return
         endif
       endif
! check for duplicate CV (exact identical entry only)
       found=.false.
       do l=1, cv%num_cv
        if (cv%type(l).eq.type) then
         if (j.eq.cv%priv(l)%p(1)) then
    write(info(1),*)' QUAT_',pos,'CV ALREADY PRESENT. NOTHING DONE.';call wrndie(0,whoami,trim(info(1)))
            cv_qcomp_add=.false.
            return
         endif
        endif
       enddo
!
       l=cv_common_add(k,gamma,weight,type) ! get a new cv index
       if (l.gt.0) then
! add quaternion index
        allocate(cv%priv(l)%p(1));
        cv%priv(l)%p(1)=j
! 10.2010 VO *************************************************************************
! since the CV depends on atom coords through the definition of a frame,
! we have to add the relevant atom lists in cv%amap with the index of this cv (l)
        if (f1.gt.0) then
          ncomf=frames%priv(f1)%p(1);
          allocate(indf(ncomf));
          ii=2; jj=ii+ncomf-1;
          indf=frames%priv(f1)%p(ii:jj)
          do j=1, ncomf ! loop over frame atoms
            i=int_vector_uadd(cv%amap%v(indf(j)),l) ! add index of this cv
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
          enddo
          deallocate(indf)
        endif
!****************************************************************************************
        cv_qcomp_add=.true.
       else ! out of bounds
        write(info(1),*)' ERROR ADDING QCOMP_',pos,'CV. NOTHING DONE.';call wrndie(0,whoami,trim(info(1)))
        cv_qcomp_add=.false.
       endif
       end function cv_qcomp_add
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine cv_qcomp_calc(i,x,y,z,mass,fx,fy,fz, &
     & calctheta,deriv,addforce,fext)
!
       real(chm_real) :: x(:), y(:), z(:), &
     & fx(:), fy(:), fz(:), mass(:)
       real(chm_real), optional :: fext ! `external` force for planar dynamics (see smcv_addforce in smcv_master)
       integer :: i
       logical :: calctheta ! whether or not to calculate theta(x); if not, use theta=cv%r(i,instant)
                            ! note that if calctheta=.false., we do not calculate the derivatives either
       logical :: deriv ! whether or not to calculate derivatives w.r.t. x
       logical :: addforce ! whether or not to add forces on simulation atoms
!
       integer :: iq, indpsf, j, jj
       integer :: pos
       real(chm_real) :: dummy, w, f
!
       integer, pointer :: priv(:) ! will point to the amap pointer array of parent quaternion
!
       if (calctheta) then
! check that the type is valid
        select case (cv%type(i))
         case (qcomp_1); pos=1
         case (qcomp_2); pos=2
         case (qcomp_3); pos=3
         case (qcomp_4); pos=4
        end select
!
! do work:
! extract quaternion index
        iq=cv%priv(i)%p(1);
!
! calculate quaternion derivatives
        call quat_calc(iq, x, y, z, mass, deriv)
!
        cv%r(i,previnst)=cv%r(i,instant)
        cv%r(i,instant)=quat%q(pos,iq)
        if (deriv) then ! brute force way
!
! do j=1,cv%amap%last
         priv=>quat%priv(iq)%amap_ptr ! point to map pointers of parent quaternion
         do jj=2,priv(1)+1
          j=priv(jj)
          w=1d0/sqrt(mass(cv%amap%i(j))) ! mass weight
!
          dummy=quat%gradqx(pos,j,iq)
          cv%gradx(i,j,1)=dummy
          cv%gradx(i,j,2)=dummy*w
!
          dummy=quat%gradqy(pos,j,iq)
          cv%grady(i,j,1)=dummy
          cv%grady(i,j,2)=dummy*w
!
          dummy=quat%gradqz(pos,j,iq)
          cv%gradz(i,j,1)=dummy
          cv%gradz(i,j,2)=dummy*w
         enddo
        endif ! deriv
       endif ! calctheta
!
! NOTE that the forces calculated here are NOT acting on the CV, as in
! the evolution subroutine, but on the simulation atoms
       dummy=cv%r(i,zcur)-cv%r(i,instant) ! zcur contains reference coords (combination of main+comp)
!
       f=cv%k(i)*dummy ! note that f is not divided by 1/sqrt(mass)
       cv%r(i,forces2)=f
       if (addforce) then
        if (present(fext)) f=fext ! override for planar sampling
! loop over all atom indices
! need to figure out a clever way of only looping over the relevant atoms
! NEW LOOP BOUNDS BELOW (looping over only the relevant atom subset)
! do j=1,cv%amap%last
!
! extract quaternion index
        iq=cv%priv(i)%p(1);
        priv=>quat%priv(iq)%amap_ptr ! point to map pointers of parent quaternion
        do jj=2, priv(1)+1
          j=priv(jj)
          indpsf=cv%amap%i(j)
          fx(indpsf)=fx(indpsf)-f*cv%gradx(i,j,1)
          fy(indpsf)=fy(indpsf)-f*cv%grady(i,j,1)
          fz(indpsf)=fz(indpsf)-f*cv%gradz(i,j,1)
        enddo
       endif ! addforce
!
      end subroutine cv_qcomp_calc
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine cv_qcomp_list(i)
       use stream
#if (KEY_MULTICOM==1)
       use multicom_aux; 
#endif
       use mpi
 character(len=132)::info(21)=(/'','','','','','','','','','','','','','','','','','','','',''/);! output buffer
!
       integer :: i
       integer :: iq
       character(len=1) :: pos
       character(len=len("CV_QCOMP_LIST>") ),parameter::whoami="CV_QCOMP_LIST>";!macro
!
!
       if (MPI_COMM_STRNG.eq.MPI_COMM_NULL) return ! only replica heads stay
!
! check type
       select case (cv%type(i))
        case (qcomp_1); pos='W'
        case (qcomp_2); pos='X'
        case (qcomp_3); pos='Y'
        case (qcomp_4); pos='Z'
        case default;
         call wrndie(0,whoami,trim('UNKNOWN CV TYPE. NOTHING DONE.'))
         return
       end select
!
       if (ME_STRNG.eq.0) then
        iq=cv%priv(i)%p(1)
        write(info,'(A)') '\t '//pos//'-COMPONENT OF' ; 
        write(OUTU,'(A)') pack(info,info.ne.'');info='';
        call quat_list(iq)
       endif
!
       end subroutine cv_qcomp_list
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
#endif
#endif /* automatically protect all code */
      end module cv_qcomp
!
