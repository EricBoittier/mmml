module gmmutil
  use chm_kinds
  use dimens_fcm
implicit none
#if KEY_GAMUS==1 /*gamus*/
!struct gmm_parameter {
!        size_t D;           // the dimensionality of the fitting space
!        size_t M;           // the number of Gaussians
!        double gamma;       // the constant offset of the probabilities.
!        double * pi;        // the vector of constants pi
!        double * m;         // the vector of vectors m
!        double * CR;        // the vector of matrices Cholesky R
!        double * Sinv;      // the vector of matrices S^(-1)
!        double * ln_det_S;  // the vector of logarithm of determinants of S
!        periodic * p;       // the periodicity information of the fit
!        double ln_like;     // the log of the likelihood of the training data.
!};

REAL(chm_real),parameter :: DLN2PI = 1.837877066409345483560659472811235279722794d0

type, public :: gmm_parameter
   integer ngauss,ndim
   real(chm_real) gamma,lnlike
   real(chm_real), allocatable :: weight(:),mean(:,:),r(:,:,:),sinv(:,:,:),lndet(:),minper(:),maxper(:) ! weights, logarithm of determinants 
   logical,allocatable :: qper(:) !flag if periodic
end type gmm_parameter
contains

subroutine givens(c,s,a,b)
use chm_kinds
use number
implicit none
real(chm_real) c,s,a,b,t
    if (b==zero) then
        c=one
        s=zero
    else if (abs(b)>abs(a)) then
        t = -a/b
        s = 1 / sqrt(1+t*t)
        c = s * t
    else if (abs(b)<=abs(a)) then
        t = -b/a
        c = 1 / sqrt(1+t*t)
        s = c * t
    endif
end subroutine givens

subroutine givens_QR_extra_row(n,R,extrarow)
use chm_kinds
use number
implicit none
integer n
real(chm_real) R(1:n,1:n),extrarow(1:n)
integer i,j
real(chm_real) c,s,a,b
     do i=1,n
        call givens(c,s,R(i,i),extrarow(i))
        do j=i,n
             a=R(i,j)
             b=extrarow(j)
             R(i,j)=c*a-s*b
             extrarow(j)=s*a+c*b
        end do
        extrarow(i)=zero
    end do
end subroutine givens_QR_extra_row

      real(chm_real) FUNCTION SUMLOG(A, B)
      use chm_kinds
      use number
      implicit none
      real(kind=chm_real) :: A, B, BA
!     swap if A<B
      IF (A.LT.B) THEN
         BA = A
         A = B
         B = BA
      ENDIF
      BA = ONE + EXP(B-A)
      SUMLOG = A + LOG ( BA )
      RETURN
      END FUNCTION SUMLOG


real(chm_real) function logsum_all(x,n)
use chm_kinds
use number
integer n
real(chm_real) x(*)
integer i
real(chm_real) aux,a
      a=maxval(x(1:n))
      !sum=zero
      !do i=1,n
      !     sum=sum+exp(x(i)-a) !argument guaranteed to be <=0 for all x(i)
      !end do
      aux=sum(exp(x(1:n)-a))
      logsum_all=a+log(aux)
end function logsum_all

subroutine alloc_gmm_parameter(ngauss,ndim,param)
use chm_kinds
use memory
use number
implicit none
integer ngauss,ndim
type(gmm_parameter) param
integer i
    param%ngauss = ngauss
    param%ndim = ndim
    call chmalloc('gmmutil.src','alloc_gmm_parameter','weight',ngauss,crl=param%weight)
    call chmalloc('gmmutil.src','alloc_gmm_parameter','mean',ndim,ngauss,crl=param%mean)
    call chmalloc('gmmutil.src','alloc_gmm_parameter','r',ndim,ndim,ngauss,crl=param%r)
    call chmalloc('gmmutil.src','alloc_gmm_parameter','sinv',ndim,ndim,ngauss,crl=param%sinv)
    call chmalloc('gmmutil.src','alloc_gmm_parameter','lndet',ngauss,crl=param%lndet)
    call chmalloc('gmmutil.src','alloc_gmm_parameter','minper',ndim,crl=param%minper)
    call chmalloc('gmmutil.src','alloc_gmm_parameter','maxper',ndim,crl=param%maxper)
    call chmalloc('gmmutil.src','alloc_gmm_parameter','qper',ndim,log=param%qper)
    do i=1,ndim
       param%qper(i)=.true.
       param%minper(i)=-one8ty
       param%maxper(i)=one8ty
    end do
end subroutine alloc_gmm_parameter

subroutine dealloc_gmm_parameter(param)
use chm_kinds
use memory
use number
implicit none
integer ngauss,ndim
type(gmm_parameter) param
integer i
    ngauss = param%ngauss
    ndim = param%ndim
    call chmdealloc('gmmutil.src','alloc_gmm_parameter','weight',ngauss,crl=param%weight)
    call chmdealloc('gmmutil.src','alloc_gmm_parameter','mean',ndim,ngauss,crl=param%mean)
    call chmdealloc('gmmutil.src','alloc_gmm_parameter','r',ndim,ndim,ngauss,crl=param%r)
    call chmdealloc('gmmutil.src','alloc_gmm_parameter','sinv',ndim,ndim,ngauss,crl=param%sinv)
    call chmdealloc('gmmutil.src','alloc_gmm_parameter','lndet',ngauss,crl=param%lndet)
    call chmdealloc('gmmutil.src','alloc_gmm_parameter','minper',ndim,crl=param%minper)
    call chmdealloc('gmmutil.src','alloc_gmm_parameter','maxper',ndim,crl=param%maxper)
    call chmdealloc('gmmutil.src','alloc_gmm_parameter','qper',ndim,log=param%qper)
end subroutine dealloc_gmm_parameter

!from param1 to param2
subroutine copy_gmm_parameter(param1,param2)
use chm_kinds
implicit none
type(gmm_parameter) param1,param2
      param2%ngauss = param1%ngauss
      param2%ndim = param1%ndim
      param2%gamma = param1%gamma
      param2%lnlike = param1%lnlike
      param2%weight = param1%weight
      param2%mean = param1%mean
      param2%r = param1%r
      param2%sinv = param1%sinv
      param2%lndet = param1%lndet
      param2%minper = param1%minper
      param2%maxper = param1%maxper
      param2%qper = param1%qper
end subroutine copy_gmm_parameter

subroutine read_gmm_parameter(param,iunit)
use chm_kinds
use stream
implicit none
integer iunit
type(gmm_parameter) :: param
integer i,j,k
#if KEY_ENSEMBLE==0
      if (iolev.gt.0) then  /*justin, for parallel*/
#endif
      READ(iunit,*) param%gamma
      DO I=1,param%ngauss
         READ(iunit,*) param%weight(i)
      END DO
      DO I=1,param%ngauss
         DO J=1,param%ndim
            READ(iunit,*) param%mean(j,i) 
         end do
      END DO
      DO I=1,param%ngauss
         DO J=1,param%ndim
            DO K=1,param%ndim
               READ(iunit,*) param%sinv(j,k,i)  
            END DO
         END DO
      END DO
      DO I=1,param%ngauss
         READ(iunit,*) param%lndet(i)
      END DO
#if KEY_ENSEMBLE==0
      end if  /*iolev*/
#endif
      RETURN
END SUBROUTINE READ_gmm_parameter


subroutine write_gmm_parameter(param,iunit)
            use chm_kinds
            use stream
            implicit none
            type(gmm_parameter) :: param
            integer :: iunit
            integer :: i,j,k
#if KEY_ENSEMBLE==0
            if (iolev.gt.0) then 
#endif
            if (iunit<=0) call wrndie(-2,'<GAMUSWRITE>','Invalid unit')
            WRITE(iunit,*) param%ngauss,param%ndim
            WRITE(iunit,*) param%gamma
            DO I=1,param%ngauss
               WRITE(iunit,*) param%weight(i)
            END DO
            DO I=1,param%ngauss
               DO J=1,param%ndim
                  WRITE(iunit,*) param%mean(j,i) 
               END DO
            END DO
            DO I=1,param%ngauss
               DO J=1,param%ndim
                  DO K=1,param%ndim
                     WRITE(iunit,*) param%sinv(j,k,i) 
                  END DO
               END DO
            END DO
            DO I=1,param%ngauss
               WRITE(iunit,*) param%lndet(i) 
            END DO
#if KEY_ENSEMBLE==0
            end if  /*iolev*/
#endif

end subroutine write_gmm_parameter


subroutine min_image(param,w)
use chm_kinds
implicit none
type(gmm_parameter) :: param
real(chm_real) :: w(*)
integer i
     do i=1,param%ndim !find minimum image
        if (param%qper(i)) then
            if (w(i)<param%minper(i)) w(i)=w(i)+param%maxper(i)-param%minper(i)
            if (w(i)>param%maxper(i)) w(i)=w(i)-param%maxper(i)+param%minper(i)
        endif
     end do
end subroutine min_image


real(chm_real) function log_pdf_gaussian_periodic(param,gauss,x,w)
use chm_kinds
implicit none
type(gmm_parameter) :: param
integer gauss
real(chm_real) :: x(*),w(*)
integer i
real(chm_real) sum
     w(1:param%ndim)=x(1:param%ndim)-param%mean(1:param%ndim,gauss)
     call min_image(param,w)
     sum = dot_product(w(1:param%ndim),matmul(param%sinv(:,:,gauss),w(1:param%ndim)))
     log_pdf_gaussian_periodic = -0.5*(param%ndim*Dln2Pi + param%lndet(gauss) + sum)
end function log_pdf_gaussian_periodic
     

#if KEY_PARALLEL==1
subroutine broadcast_gmm_parameter(param)
use chm_kinds
use parallel,only:psnd4,psnd8

implicit none
type(gmm_parameter) param
    call psnd4(param%ngauss,1)
    call psnd4(param%ndim,1)
    call psnd8(param%gamma,1)
    call psnd8(param%lnlike,1)
    call psnd8(param%weight,param%ngauss)
    call psnd8m(param%mean,param%ngauss*param%ndim)
    call psnd8m(param%r,param%ngauss*param%ndim*param%ndim)
    call psnd8m(param%sinv,param%ngauss*param%ndim*param%ndim)
    call psnd8(param%lndet,param%ngauss)
    call psnd8(param%minper,param%ndim)
    call psnd8(param%maxper,param%ndim)
    call psnd4(param%qper,param%ndim)
end subroutine broadcast_gmm_parameter

!Of the GMM parameter sets present on various processors, finds the one with the maximum likelihood, and broadcasts this to all the processors.
!Since the PSND4 and PSND8 subroutines broadcast only from node 0, these cannot be used directly.
!Instead, we use GSEN and GREC to send the maximum likelihood parameters to node 0 and then broadcast.
subroutine broadcast_max_likelihood(param)
use chm_kinds
use parallel
use memory
use number
use mpi
implicit none
type(gmm_parameter) param
integer nodemax,i
real(chm_real) maxlike
real(chm_real),allocatable :: likelihood(:)
     !Collect the maximum likelihood obtained on all the nodes, and find out which node obtained this maximum likelihood.
     call chmalloc('gmmutil.src','broacast_max_likelihood','likelihood',numnod,crl=likelihood,lbou=0)
     !write(*,*) "node ",mynod," my maximum likelihood ",param%lnlike," my gamma ",param%gamma
     likelihood = zero
     likelihood(mynod) = param%lnlike
     call gcomb(likelihood,numnod)
     !write(*,*) "node ",mynod," likelihood array ",likelihood
     nodemax = -1
     maxlike = -thosnd
     do i=0,numnod-1
        if (likelihood(i)>=maxlike) then
           nodemax=i
           maxlike=likelihood(i)
        end if
     end do
     !Send from this node to master node.
     if (nodemax /= master_node) then
          if (mynod == nodemax) then
              !send param to the master node. Lengths of the arrays must be specified in bytes rather than as a number of items.
              call gsen(master_node,201,param%ngauss,mpi_integer_size)
              call gsen(master_node,202,param%ndim,mpi_integer_size)
              call gsen(master_node,203,param%gamma,mpi_real8_size)
              call gsen(master_node,204,param%lnlike,mpi_real8_size)
              call gsen(master_node,205,param%weight,param%ngauss*mpi_real8_size)
              call gsen(master_node,206,param%mean,param%ngauss*param%ndim*mpi_real8_size)
              call gsen(master_node,207,param%r,param%ngauss*param%ndim*param%ndim*mpi_real8_size)
              call gsen(master_node,208,param%sinv,param%ngauss*param%ndim*param%ndim*mpi_real8_size)
              call gsen(master_node,209,param%lndet,param%ngauss*mpi_real8_size)
              call gsen(master_node,210,param%minper,param%ndim*mpi_real8_size)
              call gsen(master_node,211,param%maxper,param%ndim*mpi_real8_size)
              call gsen(master_node,212,param%qper,param%ndim*mpi_integer_size)
          else if (mynod == master_node) then
              !receive param from node nodemax
              call grec(nodemax,201,param%ngauss,mpi_integer_size)
              call grec(nodemax,202,param%ndim,mpi_integer_size)
              call grec(nodemax,203,param%gamma,mpi_real8_size)
              call grec(nodemax,204,param%lnlike,mpi_real8_size)
              call grec(nodemax,205,param%weight,param%ngauss*mpi_real8_size)
              call grec(nodemax,206,param%mean,param%ngauss*param%ndim*mpi_real8_size)
              call grec(nodemax,207,param%r,param%ngauss*param%ndim*param%ndim*mpi_real8_size)
              call grec(nodemax,208,param%sinv,param%ngauss*param%ndim*param%ndim*mpi_real8_size)
              call grec(nodemax,209,param%lndet,param%ngauss*mpi_real8_size)
              call grec(nodemax,210,param%minper,param%ndim*mpi_real8_size)
              call grec(nodemax,211,param%maxper,param%ndim*mpi_real8_size)
              call grec(nodemax,212,param%qper,param%ndim*mpi_integer_size)
          endif
     endif
     !Broadcast from master node to every other node.
     call broadcast_gmm_parameter(param)
     call chmdealloc('gmmutil.src','broacast_max_likelihood','likelihood',numnod,crl=likelihood)
end subroutine broadcast_max_likelihood

#endif 
 
#endif /* (gamus)*/
end module gmmutil

