! ********************************************************************************!
! Preprocessed by GNU CPP !
! ********************************************************************************!
! 2008 Victor Ovchinnikov; Harvard/MIT
! module to perform superposition of two structures
! An alternative to FROTU, the module also computes derivatives
! of the rotation matrix w.r.t. Cartesian coordinates
! Reference for derivatives:
! Ovchinnikov & Karplus, J. Phys. Chem. B 2012, Analysis and Elimination of a Bias in
! Targeted Molecular Dynamics Simulations of Biomolecules
! Basic algorithms: Best-fit alignment : W. Kabsch '78; diagonalization: Cramer's rule
!
! cannot inline module procedures
!!DEC$ ATTRIBUTES FORCEINLINE :: norm3
!!DEC$ ATTRIBUTES FORCEINLINE :: veccross3
!!DEC$ ATTRIBUTES FORCEINLINE :: unit3
!!DEC$ ATTRIBUTES FORCEINLINE :: com
!!DEC$ ATTRIBUTES FORCEINLINE :: rmsd
!

!
!
!
  module bestfit
#if (KEY_STRINGM==1) /*  automatically protect all code */
      !
      use number, only : RSMALL; use chm_kinds
      !
      implicit none
      private
      !
       logical, private :: qdebug=.false.
       logical, private :: qcheckdet=.true.
       logical, private :: qdetfailtrace=.true.
! logical, private :: qiter_refine_eval=.true.
! logical, private :: qiter_refine_evec=.true.
       logical, private :: qiter_refine_eval=.false.
       logical, private :: qiter_refine_evec=.false.
       logical, private :: qdiag_inverse=.true.
! logical, private :: qdiag_inverse=.false.
      !
      !parameters
       real(chm_real), parameter :: zero=0d0
       real(chm_real), parameter :: one=1d0
       real(chm_real), parameter :: two=2d0
       real(chm_real), parameter :: three=3d0
       real(chm_real), parameter :: half=0.5d0
       real(chm_real), parameter :: fifth=0.2d0
       real(chm_real), parameter :: oo3=0.3333333333333333d0
       real(chm_real), parameter :: oo6=0.1666666666666666d0
       real(chm_real), parameter :: oo9=0.1111111111111111d0
       real(chm_real), parameter :: oo27=0.037037037037037037d0
       real(chm_real), parameter :: pi=3.141592653589793d0
       real(chm_real), parameter :: twopi=6.283185307179586232d0
       real(chm_real), parameter :: fourpi=12.566370614359172464d0
       real(chm_real), parameter :: cos_2pi_o3=-half
       real(chm_real), parameter :: sin_2pi_o3= 0.866025403784439d0
       real(chm_real), parameter :: cos_4pi_o3=-half
       real(chm_real), parameter :: sin_4pi_o3=-0.866025403784439d0
       real(chm_real), parameter :: Kd(3,3)= &
     & RESHAPE( (/one,zero,zero,zero,one,zero,zero,zero,one/), (/3,3/) ) ! Kronecker delta
      !
      
      !
       public eig3s ! eigenvalues + eigenvectors of a symmetric 3x3 PosDef matrix
       public eig3s_ref ! extra steps to try for a better solution
       public RMSBestFit ! superpose one structure onto another (best-fit in the sense of weighted RMSD)
       public rmsd ! calculate weighted RMSD between two structures
      ! auxiliary routines
       public norm3
       private unit3
       public veccross3
       public com
       public setdebug
      !
       contains
      !###############################################################
       subroutine setdebug(switch);
       logical :: switch
       qdebug=switch
       end subroutine setdebug
      !###############################################################
       function norm3(v)
       real(chm_real) :: v(3), norm3
       norm3=sqrt(v(1)*v(1)+v(2)*v(2)+v(3)*v(3))
       if (norm3.lt.RSMALL * 0.0001d0) norm3=zero
       end function norm3
      !###############################################################
       function unit3(v)
       real(chm_real) :: v(3), norm, unit3(3)
      ! need to be careful with roundoff error
       norm=sqrt(v(1)*v(1)+v(2)*v(2)+v(3)*v(3))
       if (norm.lt.RSMALL * 0.0001d0) then; norm=one; else; norm=one/norm; endif
       unit3=(/v(1)*norm, v(2)*norm, v(3)*norm/)
       end function unit3
      !###############################################################
       function veccross3(v1,v2)
       real(chm_real), intent(in) :: v1(3), v2(3)
       real(chm_real) :: veccross3(3)
       veccross3=(/v1(2)*v2(3)-v1(3)*v2(2), &
     & v1(3)*v2(1)-v1(1)*v2(3), &
     & v1(1)*v2(2)-v1(2)*v2(1)/)
       end function veccross3
      !###############################################################
       function com(x,w,qdimswap_)
      ! note: assuming sum(w)=1
       real(chm_real) :: x(:,:), w(:), com(3)
       logical, optional, intent(in) :: qdimswap_; logical :: qdimswap; if (present(qdimswap_))then;qdimswap=qdimswap_;else;qdimswap=.false.;endif
       if (qdimswap) then
        com=matmul(x,w)
       else
        com(1)=dot_product(x(:,1),w); com(2)=dot_product(x(:,2),w); com(3)=dot_product(x(:,3),w)
       endif
!
       end function com
      !###############################################################
       function rmsd(x,y,w,qroot_,qdimswap_)
      ! note: assuming sum(w)=1
       real(chm_real) :: x(:,:), y(:,:), w(:), rmsd ! , wsum
       integer :: n, m, i, j
       logical, optional, intent(in) :: qroot_ ! whether or not to take the (expensive) square root
       logical :: qroot
       logical, optional, intent(in) :: qdimswap_; logical :: qdimswap; if (present(qdimswap_))then;qdimswap=qdimswap_;else;qdimswap=.false.;endif
       if (present(qroot_)) then ; qroot = qroot_ ; else ; qroot=.true. ; endif
       n=size(x,1); m=size(x,2); rmsd=zero ! wsum=zero
       if (qdimswap) then
        do i=1,m
         do j=1,n
          rmsd=rmsd+( ( x(j,i)-y(j,i) )**2 * w(i) )
         enddo;
! wsum=wsum+w(i)
        enddo
       else
        do i=1,n
         do j=1,m
          rmsd=rmsd+( ( x(i,j)-y(i,j) )**2 * w(i) )
         enddo;
! wsum=wsum+w(i)
        enddo
       endif
! if (wsum.lt.RSMALL * 0.0001d0) wsum=1d0
! rmsd=rmsd/wsum
       if (qroot) rmsd=sqrt(rmsd)
       end function rmsd
      !################################################################
       subroutine eig3s(M, eval, evec)
      ! Exact diagonalization of a 3X3 SYMMETRIC matrix
      ! The symmetry is used in the assumption that all of the eigenvalues are REAL
      !
       real(chm_real) :: M(3,3)
       real(chm_real) :: a11, a12, a13, a21, a22, a23, a31, a32, a33
       real(chm_real) :: a0, a1, a2, Qq, Rr, Dd, rootQ, rootQ3, theta
       real(chm_real) :: a,b,c,d,e,f,g,h,i,det13,det12
       real(chm_real) :: eval(3), evec(3,3), p(3), q(3), r(3), v(3), &
     & dummy, pn, qn, trace
       integer :: j,k
      !
       real(chm_real):: errtol_
      !
       errtol_=RSMALL * 0.0001d0
      !
      !
      ! determinant of M
! dummy= &
! & M(1,1)*(M(2,2)*M(3,3)-M(2,3)*M(3,2))+ &
! & M(1,2)*(M(2,3)*M(3,1)-M(2,1)*M(3,3))+ &
! & M(1,3)*(M(2,1)*M(3,2)-M(2,2)*M(3,1))
! write(666,*) ' Det(M) [must be >=0] : ', dummy
      !
      !
       a11=M(1,1); a12=M(1,2); a13=M(1,3)
       a21=M(2,1); a22=M(2,2); a23=M(2,3)
       a31=M(3,1); a32=M(3,2); a33=M(3,3)
      !
       trace = a11 + a22 + a33
      ! solve the characteristic equation
       a2=-(a11+a22+a33)
       a1= (a11*a22+a11*a33+a22*a33-a23*a32-a12*a21-a13*a31)
       a0=-(a11*a22*a33+a12*a23*a31+a13*a21*a32- &
     & a11*a23*a32-a13*a31*a22-a12*a21*a33)
      !
       Qq=oo3*a1 - oo9*a2*a2
       Rr=oo6*a2*a1 - half*a0 - oo27*a2**3
      ! Dd=Qq**3+Rr*Rr ! Discriminant: D == 0 implies theta=1 below
      !
       rootQ=sqrt(-Qq)
       rootQ3=rootQ**3
      !
      ! write(0,*) 'errtol:',errtol_
      ! write(0,*) 'D:', Dd
      ! write(0,*) 'Q:', Qq
      ! write(0,*) 'R:', Rr
      ! write(0,*) 'rootQ:', rootQ
      !
       if (rootQ.lt.errtol_) then ! for the indentity matrix
        eval(1)=-a2*oo3
        eval(2)=-a2*oo3
        eval(3)=-a2*oo3
       else
        Dd=Rr/rootQ3
        ! denormalize to 1 if very close; acos very sensitive to roundoff error
! if (abs(abs(Dd)-one).lt.errtol_) then
! theta=zero
! else
         theta=acos( max(-one, min(Dd,one)) ) ! protect from out-of-range arguments
! note: acos wil give a result that is between 0 and pi
! the sine of the result is therefore always positive (see b= below)
! endif
      !
! eval(1)=two*rootQ*cos( theta*oo3) -a2*oo3
! eval(2)=two*rootQ*cos((theta+twopi) *oo3)-a2*oo3
! eval(3)=two*rootQ*cos((theta+fourpi)*oo3)-a2*oo3
!
! try to speed things up: (15sec w/ intel) ; appears to make no difference
! a = cos( theta*oo3 ); b=sqrt(one-a*a)*sin_2pi_o3 ; ! use cosine sum
        a = cos( theta*oo3 ); b=sin( theta*oo3 ) *sin_2pi_o3 ; ! use cosine sum
        eval(1)=two*rootQ*(a )-a2*oo3
        eval(2)=two*rootQ*(a*cos_2pi_o3-b)-a2*oo3
        eval(3)=two*rootQ*(a*cos_2pi_o3+b)-a2*oo3
! eval(3)=two*rootQ*(a*cos_4pi_o3-b*sin_4pi_o3)-a2*oo3
       endif
      ! write(0,*) 'EVAL:', eval
      ! Dd=Rr/rootQ3
      ! write(0,'(A,5E30.20)') 'test:', theta, rootQ3, Dd, acos(Dd), acos(1d0)
      !############# truncate if within errtol of 0 ###################
       if (abs(eval(1)) .lt. errtol_) eval(1)=zero
       if (abs(eval(2)) .lt. errtol_) eval(2)=zero
       if (abs(eval(3)) .lt. errtol_) eval(3)=zero
      !############# sort eigenvalues in the order of DECREASING MAGNITUDE
       if (abs(eval(1)).lt.abs(eval(2))) then;
        dummy=eval(1); eval(1)=eval(2); eval(2)=dummy; endif
       if (abs(eval(1)).lt.abs(eval(3))) then;
        dummy=eval(1); eval(1)=eval(3); eval(3)=dummy; endif
       if (abs(eval(2)).lt.abs(eval(3))) then;
        dummy=eval(2); eval(2)=eval(3); eval(3)=dummy; endif
      !#################################################################
! ad hoc : try to refine eigenvalues using Newton-Raphson
! for most cases, turns out that the correction is already machine precision, so no improvement is possible
! ! write(0,*) 'EVAL:', eval
       if (qiter_refine_eval) then
        do j=1,2
         d=eval(j)
         if ( abs(d) .gt. errtol_ ) then ! do not attempt to refine zero evals
          do k=1,1
           a=(d**3+a2*d**2+a1*d+a0)/(three*d**2+two*a2*d+a1) * errtol_ ! newton-raphson iteration
! if (abs(a).lt.errtol_) a=0.9*sign(errtol_,a) ! make sure we take a finite step
          ! write(0,*) 'CHARACTERISTIC DETERMINANT:',d,a
           d=d-a
          enddo ! k
          eval(j)=d
         endif ! (abs(d)>0)
! stop
        enddo ! j
! compute last evalue from trace :
        eval(3)=trace - eval(1) - eval(2)
       endif
!
      !######################### now compute eigenvectors ############
      !
       do j=1,3
       ! change notation
        a=a11-eval(j); b=a12 ; c=a13
        d=a21 ; e=a22-eval(j); f=a23
        g=a31 ; h=a32 ; i=a33-eval(j)
      ! columnspace vectors:
        p=(/a, d, g/) ; pn=norm3(p)
        q=(/b, e, h/) ; qn=norm3(q)
        r=(/c, f, i/) ;
      !
        if (pn .lt.errtol_) then; v=(/one , zero, zero/) ! ;write(0,*) 198, p
        elseif (qn .lt.errtol_) then; v=(/zero, one, zero/) ! ;write(0,*) 199, q
        elseif (norm3(r).lt.errtol_) then; v=(/zero, zero, one /) ! ;write(0,*) 200, r
        elseif (abs(abs(dot_product(p,q)) - pn*qn) .gt. errtol_) then ! P || Q
      ! set z=1
      ! use Cramer`s rule
         det12=a*e-b*d
         det13=a*h-b*g
      !
         if (abs(det12).gt.abs(det13) .and. abs(det12).gt.errtol_) then
! det12=one/det12;
! v=unit3((/(b*f-c*e)*det12, (d*c-a*f)*det12, one/)) !;write(0,*) 209, det12, errtol_,&
! ! unit3( (/ (b*f-c*e)*det12, (d*c-a*f)*det12, one /) )
! more stable & faster (?):
          v=unit3((/(b*f-c*e), (d*c-a*f), det12/)) !;write(0,*) 209, j, det12, det13, &
                                                               ! ( (/ (b*f-c*e), (d*c-a*f), det12 /) ), v
         elseif (abs(det13) .gt. errtol_) then
! det13=one/det13;
! v=unit3((/(b*i-c*h)*det13, (g*c-a*i)*det13, one/)) !;write(0,*) 212, det13
          v=unit3((/(b*i-c*h), (g*c-a*i), det13/)) !;write(0,*) 212, j, det13
         endif
        else
         ! otherwise there is a vector with z = 0 try setting x=1
! if (abs(b) .gt. errtol_ .and. abs(a) .gt. errtol_ ) then; v=unit3((/one, -a/b, zero/)) !;write(0,*) 216, b
! elseif (abs(e) .gt. errtol_ .and. abs(d) .gt. errtol_ ) then; v=unit3((/one, -d/e, zero/)) !;write(0,*) 217, e
! elseif (abs(h) .gt. errtol_ .and. abs(g) .gt. errtol_ ) then; v=unit3((/one, -g/h, zero/)) !;write(0,*) 218, h
         if (abs(b) .gt. errtol_ ) then; v=unit3((/one, -a/b, zero/)) !;write(0,*) 216, j, b, v
         elseif (abs(e) .gt. errtol_ ) then; v=unit3((/one, -d/e, zero/)) !;write(0,*) 217, j, e, v
         elseif (abs(h) .gt. errtol_ ) then; v=unit3((/one, -g/h, zero/)) !;write(0,*) 218, j, h, v
         else ! Q=0 this should not be possible because P Q R are checked above !
           v=(/zero, one, zero/) !;write(0,*) 220, j
         endif
        endif
        evec(:,j)=v
! if the first two evectors are perpendicular, obtain the third by taking a cross product; then bail
! if (qdebug) write(666,*) " *EIG: ", evec(:,j)
!
        if (j.eq.2) then
! if (qdebug) write(666,*) " < ev1|ev2 > / errtol : ", dot_product(evec(:,1),evec(:,2)), errtol_
! if (abs(dot_product(evec(:,1),evec(:,2))).lt.errtol_) then
! note: for a symmetric matrix, different evectors are orthogonal, so use less strict criterion below
         if (abs(dot_product(evec(:,1),evec(:,2))).lt.half) then
          evec(:,3)=veccross3(evec(:,1),evec(:,2))
! obtain orthogonality to higher precision (?) :
! v=veccross3(evec(:,2),evec(:,3))
! evec(:,2)=veccross3(evec(:,3),evec(:,1))
! evec(:,1)=v;
!
! write(666,*) " *EIG: ", evec(:,3)
! return
          exit ! quit do loop to skip j=3 iteration, but do not return
         endif
        endif
       enddo ! j-loop
      !
      ! make sure that triple degeneracy is taken into account
       if (j.gt.2) then ! do not check if we bailed out
! if ((abs(abs(dot_product(evec(:,1),evec(:,2)))-one).lt.errtol_) &
! & .and.(abs(abs(dot_product(evec(:,2),evec(:,3)))-one).lt.errtol_))&
        if ((abs(abs(dot_product(evec(:,1),evec(:,2)))-one).lt.half) & ! see previous comment
     & .and.(abs(abs(dot_product(evec(:,2),evec(:,3)))-one).lt.half))&
     & then
         evec=Kd
        else
      ! make sure that double degeneracy is taken into account
      ! the only possible remaining case is that vectors 1 and 2 are parallel
         evec(:,1)=veccross3(evec(:,2),evec(:,3))
        endif
       endif
!
! optional power iterations to improve pesky eigenvectors (in some cases might make things worse)
       if (qiter_refine_evec) then
! first eigenvalue/vector pair
        if (abs(eval(1)).gt.errtol_) then
         a=evec(1,1); b=evec(2,1); c=evec(3,1)
         dummy=one/eval(1)
         do j=1,1
          d = a11*a + a12*b + a13*c; e = a21*a + a22*b + a23*c; f = a31*a + a32*b + a33*c; a=d*dummy ; b=e*dummy ; c=f*dummy
         enddo
         v=unit3((/a,b,c/)) ; evec(:,1)=v ;
! second eigenvalue/vector pair
         if (abs(eval(2)).gt.errtol_) then
          a=evec(1,1); b=evec(2,1); c=evec(3,1)
          d=evec(1,2); e=evec(2,2); f=evec(3,2)
          theta=one/eval(2)
          do j=1,1
           dummy=a*d+b*e+c*f;
           d=d-dummy*a ; e=e-dummy*b ; f=f-dummy*c
           g = a11*d + a12*e + a13*f; h = a21*d + a22*e + a23*f; i = a31*d + a32*e + a33*f; d=g*theta ; e=h*theta ; f=i*theta
          enddo ; r=unit3((/d,e,f/)) ; evec(:,2)=r ;
         endif
! obtain third vector using a cross product
         evec(:,3)=veccross3(evec(:,1),evec(:,2))
        endif
! if (qdebug) write(666,*) " *REF EIG: ", evec(:,1)
! if (qdebug) write(666,*) " *REF EIG: ", evec(:,2)
! if (qdebug) write(666,*) " *REF EIG: ", evec(:,3)
       endif ! qiter_refine
!
      !
       end subroutine eig3s
      !#########################################################
       subroutine RMSBestFit(x0,y0,w,u,ibeg_,iend_,gradu,eigval,qdimswap_)





      ! fit structure in x0 onto y0 in the sense of least RMSD
      ! implementation of the algorithm by Kabsch 1976 (plus reflection modification)
      ! 10.09: compute gradients of u w.r.t. y0-com(y0);
      ! because we are subtracting the COM from both x0 and y0, this is also the
      ! gradient w.r.t. the absolute coordinates: the matrix u is invariant to translation of
      ! x if COM(y) is 0, and invariant to translation of y if COM(x) is zero
      ! gradients are computed only for the indices ibeg -- iend for
      ! easy parallelization (which is, therefore, not done in this routine)
      ! note that the _entire_ Grad U array is assumed to be passed in
      ! NOTE 5/2/2014 : will stress again that the derivatives are computed wrt y0
      ! derivatives wrt x0 are obtained by transposition (take care to remove COM)
      !***************************************************************
      ! x0 -- orientation structure
      ! y0 -- template structure
      ! w -- fitting weights, e.g. mass weighting
      ! (note: not requiring that sum(w)=1, but this is HIGHLY recommended for FP precision)
      ! u -- rotation matrix [ such that x = u . x0 ]
      ! ibeg -- index of first atom with respect to whose coords the gradients are to be computed
      ! iend -- index of last atom with respect to whose coords the gradients are to be computed
      ! gradu -- gradient array, computed with respect to y0 (previously, erroneously, stated as w.r.t x0)
      ! eigval -- square root of eigenvalues of the correlation matrix
      ! qdimswap -- if false, the shapes of coordinate arrays are (1:natom,1:3); if true, (1:3,1:natom)
!
       real(chm_real), intent(in), target :: x0(:,:), y0(:,:), w(:)
       real(chm_real), intent(out) :: u(3,3)
       integer, optional, intent(in) :: ibeg_, iend_
       real(chm_real), optional, intent(out) :: gradu(:,:,:,:) ! gradient of rotation matrix wrt y0 (previously, erroneously, stated as w.r.t x0)
       real(chm_real), optional, intent(out) :: eigval(3) ! eigenvalues
!
       real(chm_real), pointer, dimension(:) :: x1, x2, x3, y1, y2, y3
       real(chm_real) :: xcom(3),ycom(3),r11,r12,r13,r21,r22,r23,r31,r32,r33, &
     & detr, x1w, x2w, x3w, y1s, y2s, y3s
       real(chm_real) :: r(3,3),rr(3,3),a(3,3),b(3,3),b0(3,3),mu(3),oomu(2), &
     & rmu(3), roomu(2)
       integer :: i, j, k, l, n
       integer :: ibeg=1, iend=0
       character(len=len("RMSBestFit>") ),parameter::whoami="RMSBestFit>";!macro
       character(len=132)::info(21)=(/'','','','','','','','','','','','','','','','','','','','',''/);! output buffer
! for derivatives:
       logical :: qgrad
       real(chm_real) :: invA(3,3),detA,e,v(3),c(3),dvdrr(3,3,3,3),dmudrr(3,3,3)
       real(chm_real) :: gradb(3,3,3)
       real(chm_real) :: a11, a12, a13, a21, a22, a23, a31, a32, a33
       integer :: p, q, s
       real(chm_real), allocatable :: grada(:,:,:,:), gradmu(:,:,:)
       real(chm_real) :: x, y, z, wgt, mu1, mu2, errtol_
       real(chm_real) :: G(3,3,3)
!
       logical, optional, intent(in) :: qdimswap_; logical :: qdimswap; if (present(qdimswap_))then;qdimswap=qdimswap_;else;qdimswap=.false.;endif
!
       errtol_=RSMALL * 0.0001d0
!
       if (qdebug) then ; write(666,*) whoami, ' ERRTOL:= ', errtol_ ; endif
!
       xcom=com(x0,w,qdimswap) ; ycom=com(y0,w,qdimswap)
!
! initialize
       r11=zero; r12=zero; r13=zero
       r21=zero; r22=zero; r23=zero
       r31=zero; r32=zero; r33=zero
!
       if (qdimswap) then
        n=size(x0, 2)
! build correlation matrix
        do i=1,n
         x1w = (x0(1,i)-xcom(1)) * w(i)
         x2w = (x0(2,i)-xcom(2)) * w(i)
         x3w = (x0(3,i)-xcom(3)) * w(i)
!
         y1s = (y0(1,i)-ycom(1))
         y2s = (y0(2,i)-ycom(2))
         y3s = (y0(3,i)-ycom(3))
!
         r11=r11+y1s*x1w
         r12=r12+y1s*x2w
         r13=r13+y1s*x3w
!
         r21=r21+y2s*x1w
         r22=r22+y2s*x2w
         r23=r23+y2s*x3w
!
         r31=r31+y3s*x1w
         r32=r32+y3s*x2w
         r33=r33+y3s*x3w
        enddo
       else
        n=size(x0, 1)
        do i=1,n
         x1w = (x0(i,1)-xcom(1)) * w(i)
         x2w = (x0(i,2)-xcom(2)) * w(i)
         x3w = (x0(i,3)-xcom(3)) * w(i)
!
         y1s = (y0(i,1)-ycom(1))
         y2s = (y0(i,2)-ycom(2))
         y3s = (y0(i,3)-ycom(3))
!
         r11=r11+y1s*x1w
         r12=r12+y1s*x2w
         r13=r13+y1s*x3w
!
         r21=r21+y2s*x1w
         r22=r22+y2s*x2w
         r23=r23+y2s*x3w
!
         r31=r31+y3s*x1w
         r32=r32+y3s*x2w
         r33=r33+y3s*x3w
        enddo
       endif ! qdimswap
!
       if (present(gradu)) then
        qgrad=.true.
        if (present(ibeg_)) then ; ibeg=max(1,ibeg_) ; else ; ibeg=1 ; endif
        if (present(iend_)) then ; iend=min(n,iend_) ; else ; iend=n ; endif
       else ; qgrad = .false.
       endif
!
       r=RESHAPE( (/r11,r21,r31,r12,r22,r32,r13,r23,r33/),(/3,3/))
! rr=matmul(transpose(r),r) ! r`r
        rr=matmul(transpose(r),r);
      ! diagonalize rr
       call eig3s(rr, mu, a) ! mu -- evalues a -- evectors
!
       if (qdiag_inverse) then
        detr= &
     & r(1,1)*(r(2,2)*r(3,3)-r(2,3)*r(3,2))+ &
     & r(1,2)*(r(2,3)*r(3,1)-r(2,1)*r(3,3))+ &
     & r(1,3)*(r(2,1)*r(3,2)-r(2,2)*r(3,1))
        if (detr*detr.gt.errtol_) then
! also diagonalize inverse matrix to get more accurate evalues
         u(1,1)=rr(2,2)*rr(3,3)-rr(2,3)*rr(3,2);
         u(1,2)=rr(1,3)*rr(3,2)-rr(1,2)*rr(3,3);
         u(1,3)=rr(2,3)*rr(1,2)-rr(2,2)*rr(1,3);
         u(2,1)=u(1,2)
         u(2,2)=rr(1,1)*rr(3,3)-rr(1,3)*rr(3,1);
         u(2,3)=rr(1,3)*rr(2,1)-rr(1,1)*rr(2,3);
         u(3,1)=u(1,3)
         u(3,2)=u(2,3)
         u(3,3)=rr(1,1)*rr(2,2)-rr(1,2)*rr(2,1);
         call eig3s(u, v, b) ! v -- evalues b -- evectors
         mu(3)=detr*detr/v(1) ! smallest eigenvalue
         mu(2)=rr(1,1) + rr(2,2) + rr(3,3) - mu(1) - mu(3) ! compute second ev from trace invairant
         a(:,3)=b(:,1) ! eigenvector corresponding to mu(3) / v(1)
         a(:,2)=veccross3(a(:,3),a(:,1)) ! obtain via cross product
        endif
       endif
!
! ad hoc treatment of degenerate (collinear) case
      if (abs(mu(2) + mu(3)).lt.errtol_) then ; mu(2)=zero ; mu(3)=zero ; endif; ! two zero evals appear symmetric around zero
! mu=max(mu,zero) ; ! prevent small negative values
!
       if (qdebug) then
        write(666,*) whoami, '> RR matrix:'
        write(666,'(3E30.23)') rr
        write(666,*) whoami, '> E-VALUES:'
        write(666,'(3E22.15)') mu
        write(666,*) whoami, '> E-VECTORS:'
        write(666,'(3E22.15)') a
! verify directly that matrix orthonormal:
        write(666,*) whoami, '> max | U x U^T - I | (should be 0):'
        write(666,'(3E22.15)') maxval ( abs ( matmul(a,transpose(a)) - Kd) )
! verify directly that diagonalization is correct:
        b0(:,1)=mu(1)*a(:,1);b0(:,2)=mu(2)*a(:,2);b0(:,3)=mu(3)*a(:,3)! aa
        write(666,*) whoami, '> max | RR - U x M x U^T | (should be 0):'
        write(666,'(3E22.15)') maxval ( abs ( matmul(b0,transpose(a))-rr) )
       endif
!
! b0=matmul(r,a)
       b0=matmul(r,a);
!
! handle three zero eigenvalues (single point)
! note that the eigenvalues are sorted in decreasing order
       if (mu(1) .lt. errtol_) then
        b=Kd; oomu=one; roomu=one; rmu=zero
        if (qgrad) then ; gradu(:,:,:,ibeg:iend)=zero ; qgrad=.false.; endif ! derivatives are ill-defined
!*********************************************************************
! handle two zero eigenvalues (line)
       elseif (mu(2) .lt. errtol_) then
        rmu(1)=sqrt(mu(1)); roomu(1)=one/rmu(1); oomu(1)=roomu(1)**2;
        b(:,1)=b0(:,1)*roomu(1)
        oomu(2)=one; roomu(2)=one; rmu(2)=zero
        if (qgrad) then ; gradu(:,:,:,ibeg:iend)=zero ; qgrad=.false.; endif ! derivatives are ill-defined
! b(:,2-3) are not unique, so we pick arbitrary vectors
! try (1,0,0)
        if ( abs(b(1,1)) .lt. half ) then ! scalar product of b with (1,0,0) is relatively small
      ! project b1 out of b2 :
         b(:,2)=&
     & unit3( (/ one-b(1,1)*b(1,1), -b(1,1)*b(2,1), -b(1,1)*b(3,1)/) )
        else ! use (0,1,0)
         b(:,2)=&
     & unit3( (/ -b(2,1)*b(1,1), one-b(2,1)*b(2,1), -b(2,1)*b(3,1)/) )
        endif
       else ! one or fewer zero eigenvalues
        rmu(1)=sqrt(mu(1)); roomu(1)=one/rmu(1); oomu(1)=roomu(1)**2;
        rmu(2)=sqrt(mu(2)); roomu(2)=one/rmu(2); oomu(2)=roomu(2)**2;
        b(:,1)=b0(:,1)*roomu(1)
        b(:,2)=b0(:,2)*roomu(2)
       endif
! note : the third eigenvalue is not explicitly used in the rotation calculation
       if (mu(3).lt.errtol_) then; rmu(3)=zero;
                             else; rmu(3)=sqrt(mu(3)); endif
! third vector is determined by the first two
       b(:,3)=veccross3(b(:,1),b(:,2))
!
! u=matmul(b, transpose(a)) ! this is the rotation matrix; apply using: x_new = U . x_old
       u=matmul(b,transpose(a));
! u(:,3)=veccross3(u(:,1),u(:,2))
!
       if (qcheckdet) then
        detr= &
     & u(1,1)*(u(2,2)*u(3,3)-u(2,3)*u(3,2))+ &
     & u(1,2)*(u(2,3)*u(3,1)-u(2,1)*u(3,3))+ &
     & u(1,3)*(u(2,1)*u(3,2)-u(2,2)*u(3,1))
! if ( abs(detr-one) .gt. errtol_ ) then
        if ( abs(detr-one) .gt. 1d-3 ) then ! this is a crude test, but should be sufficient
          if (qdetfailtrace) then ! print debugging info after a determinant check failure
! write(666,*) whoami, '> PLEASE SEND THIS FILE TO OVCHINNV@GMX.COM'
           write(666,*) whoami, '> DEBUG DUMP FROM BESTFIT.FTN'
           write(666,*) whoami, '> RR matrix:'
           write(666,'(3E30.22)') rr
           write(666,*) whoami, '> E-VALUES:'
           write(666,'(3E30.22)') mu
           write(666,*) whoami, '> E-VECTORS:'
           write(666,'(3E30.22)') a
! verify directly that matrix orthonormal:
           write(666,*) whoami, '> max | Ev x Ev^T - I | (should be 0):'
           write(666,'(1E30.22)') maxval ( abs ( matmul(a,transpose(a)) - Kd ) )
! verify directly that diagonalization is correct:
           b0(:,1)=mu(1)*a(:,1);b0(:,2)=mu(2)*a(:,2);b0(:,3)=mu(3)*a(:,3)! aa
           write(666,*) whoami, '> max | RR - Ev x M x Ev^T | (should be 0):'
           write(666,'(1E30.22)') maxval ( abs ( matmul(b0,transpose(a))-rr) )
! verify directly that the ROTATION matrix is orthonormal:
           write(666,*) whoami, '> max | U x U^T - I  | (should be 0):'
           write(666,'(1E30.22)') maxval ( abs ( matmul(u,transpose(u)) - Kd ) )
           if (qgrad) then ; gradu(:,:,:,ibeg:iend)=zero ; qgrad=.false.; endif ! skip derivatives
         endif !qdetfailtrace
         write(info(1),*)'ROTATION MATRIX (U) NOT UNITARY. DET[U]=',detr,'.';call wrndie(0,whoami,trim(info(1)))
!
        endif ! detr/=1
       endif ! qcheckdet
!
       if (qdebug) then
        write(666,*) whoami, '> ROTATION MATRIX:'
        write(666,'(3E30.22)') u
       endif
!
       if (present(eigval)) then
      ! return square root of eigenvalues
      ! determinant of r
        if (.not. qdiag_inverse) detr= & ! otherwise, already computed
     & r(1,1)*(r(2,2)*r(3,3)-r(2,3)*r(3,2))+ &
     & r(1,2)*(r(2,3)*r(3,1)-r(2,1)*r(3,3))+ &
     & r(1,3)*(r(2,1)*r(3,2)-r(2,2)*r(3,1))
        if (detr.lt.0) rmu(3)=-rmu(3)
        eigval=rmu
       endif
! compute gradient
       if (qgrad) then
        if (qdimswap) then
         x1=>x0(1,:); x2=>x0(2,:); x3=>x0(3,:);
         y1=>y0(1,:); y2=>y0(2,:); y3=>y0(3,:);
        else
         x1=>x0(:,1); x2=>x0(:,2); x3=>x0(:,3);
         y1=>y0(:,1); y2=>y0(:,2); y3=>y0(:,3);
        endif
! 1) compute derivative of A w.r.t. RR (and Mu w.r.t RR)
! 2) compute derivative of RR w.r.t. atom coordinates
! 3) combine 1 and 2 to produce the derivative of u w.r.t coordinates
       dvdrr=zero; dmudrr=zero ! unnecessary
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
        allocate(grada(3,3,3,ibeg:iend), gradmu(3,3,ibeg:iend))
!
! compute d[a]ij/d[rr]pq
!
! loop over eigenvectors a of rr
        do j=1,3
! construct matrix; a=rr-muI; this matrix is singular; so throw away first row, and put j_th eigenvector (v) in last row to
! enforce v*dv=0
         e=mu(j);
         v=a(:,j);
!
         a11=rr(2,1); a12=rr(2,2)-e ; a13=rr(2,3);
         a21=rr(3,1); a22=rr(3,2) ; a23=rr(3,3)-e;
         a31=v(1); a32=v(2) ; a33=v(3);
!
! make sure A is invertible; if so, compute and store inverse
! otherwise complain and quit (will deal with exceptions later)
         detA = a11*(a22*a33-a23*a32)+ &
     & a12*(a23*a31-a21*a33)+ &
     & a13*(a21*a32-a22*a31)
!
         if (abs(detA).lt.errtol_) then
          call wrndie(0,whoami,trim(' EXCEPTION: "A" MATRIX IS SINGULAR. WILL SET A^-1 TO ONE.'))
          invA=one
! return
         else
! compute inverse explicitly (do not need some entries)
          detA=one/detA;
          invA(1,1)=detA*(a22*a33-a23*a32);
          invA(1,2)=detA*(a13*a32-a12*a33);
! invA(1,3)=detA*(a23*a12-a22*a13);
          invA(2,1)=detA*(a23*a31-a21*a33);
          invA(2,2)=detA*(a11*a33-a13*a31);
! invA(2,3)=detA*(a13*a21-a11*a23);
          invA(3,1)=detA*(a21*a32-a22*a31);
          invA(3,2)=detA*(a12*a31-a11*a32);
! invA(3,3)=detA*(a11*a22-a12*a21);
         endif
! test inverse ! aardvark ! passed
! write(600+whoiam,*)
! & matmul(invA,
! & transpose(RESHAPE((/a11,a12,a13,a21,a22,a23,a31,a32,a33/),
! & (/3,3/)))) ! note: fills column-wise
! stop
!
! loop over all components in [rr]_pq:
         do p=1,3
          do q=1,p
! compute eigenvalue derivative (actually, dmu/drr_pq + dmu/drr_qp together); OK because rr is always symmetric, so rr_pq and rr_qp change identically
           dmudrr(j,p,q)=(two-Kd(p,q))*v(p)*v(q)
! construct right hand side vector (c)
           do s=2,3
             c(s-1)=v(q) * ( v(p)*v(s)-Kd(s,p) ) + & ! d[rr]_pq contribution
     & (one - Kd(p,q)) * &
     & v(p) * ( v(q)*v(s)-Kd(s,q) ) ! d[rr]_qp contribution
           enddo ! s
! c(3)=0d0;
! compute derivative dv/d[rr]
! note: only entries for which p >= q are populated; in fact, we really are computing dv/d[rr]_pq + dv/d[rr]_qp (this is OK b/c rr_ij=rr_ji)
! dvdrr(:,j,p,q)=matmul(invA,c);
            dvdrr(1,j,p,q)=invA(1,1)*c(1)+invA(1,2)*c(2);
            dvdrr(2,j,p,q)=invA(2,1)*c(1)+invA(2,2)*c(2);
            dvdrr(3,j,p,q)=invA(3,1)*c(1)+invA(3,2)*c(2);
! write(700+whoiam,*) j, p, q, dvdrr(:,j,p,q)
          enddo ! q
         enddo ! p
        enddo ! j
!
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! compute derivatives w.r.t. relative coordinates
        gradu(:,:,:,ibeg:iend)=zero
        do q=ibeg, iend
         x=x1(q)-xcom(1); y=x2(q)-xcom(2); z=x3(q)-xcom(3); wgt=w(q)
!
         do i=1,3
! compute eigenvalue derivatives
           a11=dmudrr(i,1,1)
           a21=dmudrr(i,2,1)
           a31=dmudrr(i,3,1)
           a22=dmudrr(i,2,2)
           a32=dmudrr(i,3,2)
           a33=dmudrr(i,3,3)
! gradient w.r.t x-position of atom q
           gradmu(i,1,q)=wgt * ( &
     & 2d0 * ( a11*x*r11 + a22*y*r12 + a33*z*r13 ) + &
     & a21 * ( x*r12 + y*r11 ) + &
     & a31 * ( x*r13 + z*r11 ) + &
     & a32 * ( y*r13 + z*r12 ) )
! gradient w.r.t y-position
           gradmu(i,2,q)=wgt * ( &
     & 2d0 * ( a11*x*r21 + a22*y*r22 + a33*z*r23 ) + &
     & a21 * ( x*r22 + y*r21 ) + &
     & a31 * ( x*r23 + z*r21 ) + &
     & a32 * ( y*r23 + z*r22 ) )
! gradient w.r.t z-position
           gradmu(i,3,q)=wgt * ( &
     & 2d0 * ( a11*x*r31 + a22*y*r32 + a33*z*r33 ) + &
     & a21 * ( x*r32 + y*r31 ) + &
     & a31 * ( x*r33 + z*r31 ) + &
     & a32 * ( y*r33 + z*r32 ) )
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
! compute eigenvector derivative
           do j=1,3
!
            a11=dvdrr(i,j,1,1)
            a21=dvdrr(i,j,2,1)
            a31=dvdrr(i,j,3,1)
            a22=dvdrr(i,j,2,2)
            a32=dvdrr(i,j,3,2)
            a33=dvdrr(i,j,3,3)
!
! gradient w.r.t x-position of atom q
!
            grada(i,j,1,q)=wgt * ( &
     & 2d0 * ( a11*x*r11 + a22*y*r12 + a33*z*r13 ) + &
     & a21 * ( x*r12 + y*r11 ) + &
     & a31 * ( x*r13 + z*r11 ) + &
     & a32 * ( y*r13 + z*r12 ) )
! gradient w.r.t y-position
            grada(i,j,2,q)=wgt * ( &
     & 2d0 * ( a11*x*r21 + a22*y*r22 + a33*z*r23 ) + &
     & a21 * ( x*r22 + y*r21 ) + &
     & a31 * ( x*r23 + z*r21 ) + &
     & a32 * ( y*r23 + z*r22 ) )
! gradient w.r.t z-position
            grada(i,j,3,q)=wgt * ( &
     & 2d0 * ( a11*x*r31 + a22*y*r32 + a33*z*r33 ) + &
     & a21 * ( x*r32 + y*r31 ) + &
     & a31 * ( x*r33 + z*r31 ) + &
     & a32 * ( y*r33 + z*r32 ) )
           enddo ! j
         enddo ! i
!%%%%%%%%%%%%%%%%%%%%%%%%% computed gradients of A, MU w.r.t. translated coordinates i.e. y=y0-ycom %%%%%%%%%%%%
! compute gradient of u w.r.t. relative(translated) coordinates
!
         G(:,:,1)=matmul(r,grada(:,:,1,q))
         G(:,:,2)=matmul(r,grada(:,:,2,q))
         G(:,:,3)=matmul(r,grada(:,:,3,q))
! first, compute gradients of B=RAM^{-1/2}
! do this for the first two column vectors corresponding to the two largest e-values
! obtain the third component by using orthogonality (this also treats mu(3)-->0).
         do k=1,2 ! columns of b
          mu1 = roomu(k); mu2=-half*oomu(k)
          a11=wgt * (x*a(1,k) + y*a(2,k) + z*a(3,k) ) ! dummy
          do l=1,3 ! component of gradient: 1:ddx 2:ddy 3:ddz
           do i=1,3 ! component of vector b_k
            gradb(i,k,l)=mu1*(mu2*gradmu(k,l,q)*b0(i,k)+ &
     & a11*Kd(i,l)+ &
     & G(i,k,l))
! make a contribution to gradu in this inner loop
            do j=1,3
             gradu(i,j,l,q)=gradu(i,j,l,q)+ &
     & gradb(i,k,l)*a(j,k)+grada(j,k,l,q)*b(i,k)
            enddo ! j
           enddo ! i
          enddo ! l
         enddo ! k
! treat b_3 separately below
         do l=1,3
          gradb(:,3,l)=veccross3(gradb(:,1,l),b(:,2))- &
     & veccross3(gradb(:,2,l),b(:,1))
! compute the rest of the sum
          do i=1,3; do j=1,3
            gradu(i,j,l,q)=gradu(i,j,l,q)+ &
     & gradb(i,3,l)*a(j,3)+grada(j,3,l,q)*b(i,3)
          enddo; enddo ! i,j
         enddo ! l
! OLD CODE; INACCURATE FOR MU_3 --> ZERO
! do i=1,3
! do j=1,3
! gradu(i,j,:,q)=0d0
! sum over all eigenvalues (k)
! note that b0=r*a above
! do k=1,3
! mu1=roomu(k); mu2=0.5d0*oomu(k)
! a11=wgt * (x*a(1,k) + y*a(2,k) + z*a(3,k) ) ! dummy
! do l=1,3
! gradu(i,j,l,q)=gradu(i,j,l,q) + mu1 * (
! & (-mu2*gradmu(k,l,q) * b0(i,k) +
! & G(i,k,l) +
! & Kd(i,l) * a11) * a(j,k) +
! & grada(j,k,l,q) * b0(i,k) )
! enddo ! l
! separately:
! w.r.t. x
! gradu(i,j,1,q)=gradu(i,j,1,q) + mu * (
! (-mu2*gradmu(k,1,q) * b0(i,k) +
! & G(i,k) +
! & Kd(i,1) * wgt * a11) * a(j,k) +
! & grada(j,k,1,q) * b0(i,k) )
! w.r.t. y
! gradu(i,j,2,q)=gradu(i,j,2,q) + mu * (
! (-mu2*gradmu(k,2,q) * b0(i,k) +
! & H(i,k) +
! & Kd(i,2) * wgt * a11) * a(j,k) +
! & grada(j,k,2,q) * b0(i,k) )
! w.r.t. z
! gradu(i,j,3,q)=gradu(i,j,3,q) + mu * (
! (-mu2*gradmu(k,3,q) * b0(i,k) +
! & I(i,k) +
! & Kd(i,3) * wgt * a11) * a(j,k) +
! & grada(j,k,3,q) * b0(i,k) )
! enddo ! k
! enddo ! j
! enddo ! i
! otherwise, parallelization is meaningless
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        enddo ! q
!================================================================
! direct finite-difference test of rotation gradients
!======================= finite difference test
        deallocate(grada, gradmu)
!
       endif ! qgrad
!
       end subroutine RMSBestFit
!=============================================================================
! subroutine that calls eigensolver twice
       subroutine eig3s_ref(rr, mu, a)
       real(chm_real) :: rr(3,3), a(3,3), mu(3)
       real(chm_real) :: detrr, rrinv(3,3), muinv(3), b(3,3)
!
       call eig3s(rr, mu, a) ! mu -- evalues a -- evectors
!
       if (qdiag_inverse) then
        detrr= &
     & rr(1,1)*(rr(2,2)*rr(3,3)-rr(2,3)*rr(3,2))+ &
     & rr(1,2)*(rr(2,3)*rr(3,1)-rr(2,1)*rr(3,3))+ &
     & rr(1,3)*(rr(2,1)*rr(3,2)-rr(2,2)*rr(3,1))
        if (detrr.gt.RSMALL * 0.0001d0) then
! also diagonalize inverse matrix to get more accurate evalues
         rrinv(1,1)=rr(2,2)*rr(3,3)-rr(2,3)*rr(3,2);
         rrinv(1,2)=rr(1,3)*rr(3,2)-rr(1,2)*rr(3,3);
         rrinv(1,3)=rr(2,3)*rr(1,2)-rr(2,2)*rr(1,3);
         rrinv(2,1)=rrinv(1,2)
         rrinv(2,2)=rr(1,1)*rr(3,3)-rr(1,3)*rr(3,1);
         rrinv(2,3)=rr(1,3)*rr(2,1)-rr(1,1)*rr(2,3);
         rrinv(3,1)=rrinv(1,3)
         rrinv(3,2)=rrinv(2,3)
         rrinv(3,3)=rr(1,1)*rr(2,2)-rr(1,2)*rr(2,1);
         call eig3s(rrinv, muinv, b) ! v -- evalues b -- evectors
         mu(3)=detrr/muinv(1) ! smallest eigenvalue
         mu(2)=rr(1,1) + rr(2,2) + rr(3,3) - mu(1) - mu(3) ! compute second ev from trace invairant
         a(:,3)=b(:,1) ! eigenvector corresponding to mu(3) / v(1)
         a(:,2)=veccross3(a(:,3),a(:,1)) ! obtain via cross product
        endif
       endif
       end subroutine eig3s_ref
!
#endif /* automatically protect all code */
  end module bestfit
!
