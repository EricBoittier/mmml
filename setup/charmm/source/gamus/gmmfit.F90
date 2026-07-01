module gmmfit
#if KEY_GAMUS==1
use gmmutil
implicit none

!Subroutines for Gaussian mixture model fitting, according to 
!Bowers et al. Comput. Phys. Commun. 164, 311 (2008)
!Includes periodic boundary conditions, constraints on variance in each direction

!Fortran translation and code clean up by Justin Spiriti

contains

!Equation 7 from Bowers et al.
!also log-likelihood computation (equation 12) if qupdatelike = .true.
subroutine log_omega(param,omega,x,weight,work,qupdatelike)
use chm_kinds
use stream
use parallel
implicit none
type(gmm_parameter) param
real(chm_real) omega(1:param%ngauss)
integer ndata
real(chm_real) x(1:param%ndim),weight,work(1:param%ndim)
logical qupdatelike
integer i,ii
real(chm_real) lpdf,denom,ln_pi_g
      !write(*,*) "log_omega: ",x,weight
      do i=1,param%ngauss
          lpdf = log_pdf_gaussian_periodic(param,i,x,work)
          omega(i) = param%weight(i)+lpdf
          !write(*,*) "log_omega 2:",i,lpdf,param%weight(i),omega(i)
      end do
      denom=logsum_all(omega,param%ngauss)
      if (qupdatelike.and.(param%gamma>denom)) then
         ii=maxloc(omega,1)
         !write(outu,*) "log_omega 1: ",mynod,param%gamma,denom,ii,omega(ii),&
         !    param%weight(ii),param%mean(:,ii),param%lndet(ii),x
         !write(outu,*) "log_omega 2: ",mynod,omega
         param%gamma = denom
      end if
      do i=1,param%ngauss
         ln_pi_g = omega(i)
         omega(i) = omega(i) + weight - denom
         if (qupdatelike) param%lnlike = param%lnlike + exp(omega(i))*ln_pi_g
      end do
end subroutine log_omega

!Equations 8 and 12
subroutine update_W(param,Wmega,omega,ndata,x,weight,work)
use chm_kinds
use number
use parallel
implicit none
type(gmm_parameter) param
integer ndata
real(chm_real) Wmega(1:param%ngauss),omega(1:param%ngauss),x(1:param%ndim,1:ndata),weight(1:ndata)
real(chm_real) work(1:param%ndim)
integer i,j
     Wmega = -thosnd
     param%lnlike = zero
     param%gamma = zero
     do i=1,ndata
          call log_omega(param,omega,x(:,i),weight(i),work,.true.)
          do j=1,param%ngauss
               Wmega(j) = sumlog(Wmega(j),omega(j))
          end do
          !write(*,*) "update_W: omega = ",omega
          !write(*,*) "update_W: Wmega = ",Wmega
          !write(*,*) "update_W: node,i,lnlike = ",mynod,i,param%lnlike
     end do
end subroutine update_W

!Equation 10 with periodicity
subroutine update_m(param,Wmega,omega,ndata,x,weight,work,newmeans,rms_m)
use chm_kinds
use number
implicit none
type(gmm_parameter) param
integer ndata
real(chm_real) rms_m
real(chm_real) Wmega(1:param%ngauss),omega(1:param%ngauss),x(1:param%ndim,1:ndata),weight(1:ndata)
real(chm_real) work(1:param%ndim),newmeans(1:param%ndim,1:param%ngauss)
integer i,j,k
     newmeans=zero
     !write(*,*) "update_m: ",param%qper,param%minper,param%maxper
     do i=1,ndata
         call log_omega(param,omega,x(:,i),weight(i),work,.false.)
         do j=1,param%ngauss
            work=x(:,i)-param%mean(:,j)
            call min_image(param,work)
            do k=1,param%ndim
               newmeans(k,j) = newmeans(k,j) + exp(omega(j)-Wmega(j)) * work(k)
            end do
         end do
     end do
     rms_m = zero
     do j=1,param%ngauss
         call min_image(param,newmeans(:,j))
         rms_m = rms_m + dot_product(newmeans(:,j),newmeans(:,j))
         param%mean(:,j) = param%mean(:,j) + newmeans(:,j)
         call min_image(param,param%mean(:,j))
     end do
     rms_m = sqrt(rms_m / real(param%ngauss*param%ndim))
end subroutine update_m

!Equation 9
subroutine update_weight(param,Wmega,sum_w)
use chm_kinds
implicit none
type(gmm_parameter) param
real(chm_real) Wmega(1:param%ngauss),sum_w
     param%weight(1:param%ngauss)  = Wmega(1:param%ngauss) - sum_w
end subroutine update_weight

!Equation 17
subroutine update_R(param,Wmega,omega,ndata,x,weight,work,newr)
use chm_kinds
use number
implicit none
type(gmm_parameter) param
integer ndata
real(chm_real) Wmega(1:param%ngauss),omega(1:param%ngauss),x(1:param%ndim,1:ndata),weight(1:ndata)
real(chm_real) work(1:param%ndim),newr(1:param%ndim,1:param%ndim,1:param%ngauss)
integer i,j,k
     newr=zero
     do i=1,ndata
        call log_omega(param,omega,x(:,i),weight(i),work,.false.) 
        do j=1,param%ngauss
             work=x(:,i)-param%mean(:,j)
             call min_image(param,work)
             !change "work" into the needed extra row
             work(1:param%ndim) = work(1:param%ndim) * exp((omega(j)-Wmega(j))/two)
             !Add the extra row in by performing Givens rotations.
             call givens_QR_extra_row(param%ndim,newr(:,:,j),work)
        end do
     end do
     !write(*,*) "update_R:",newr(1,1,1),newr(1,2,1),newr(2,1,1),newr(2,2,1)
     param%r = newr
     !zero out lower triangle of r
     do i=1,param%ngauss
        do j=2,param%ndim
           param%r(j,1:j-1,i) = zero
        end do
     end do
end subroutine update_R

!Update S^-1 from its upper triangular Cholesky decomposition, S = R^T . R
!S^-1 = (R^-1) . (R^-1)^T
!LAPACK's DPOTRI performs both operations
subroutine update_Sinv(param)
use chm_kinds
use number
use stream
implicit none
type(gmm_parameter) param
integer i,j,k,info
real(chm_real) d
     do i=1,param%ngauss
        param%sinv(:,:,i) = param%r(:,:,i)
        !The original code had 'L' here.
        call DPOTRI('U',param%ndim,param%sinv(:,:,i),param%ndim,info)
        if (info<0) then
           write(outu,*) "LAPACK error: matrix ",i," info ",info
           call wrndie(-10,'<update_Sinv>','LAPACK error')
        endif
        !Need to fill in lower triangle of S^-1.
        do j=2,param%ndim
           param%sinv(j,1:j-1,i) = param%sinv(1:j-1,j,i)
        end do
        !check
        !if (param%ndim==2) then
        !   d=param%sinv(1,1,i)*param%sinv(2,2,i)-param%sinv(1,2,i)*param%sinv(2,1,i)
        !   if (d<=zero) then
        !      write(outu,*) "matrix not positive definite ",i
        !      write(outu,*) "r(1,1), r(1,2), r(2,2) = ",param%r(1,1,i),param%r(1,2,i),param%r(2,2,i) 
        !      write(outu,*) "s(1,1), s(1,2), s(2,1), s(2,2) = ",param%sinv(1,1,i),param%sinv(1,2,i),&
        !           param%sinv(2,1,i),param%sinv(2,2,i)
        !      call wrndie(-10,'<update_Sinv>','problem with matrix')
        !   endif
        !endif
     end do
end subroutine update_Sinv

!Update S^-1 and ln |S| 
subroutine update_S_from_CR(param)
use chm_kinds
use number
implicit none
type(gmm_parameter) param
integer i,j
     do i=1,param%ngauss
        param%lndet(i) = zero
        do j=1,param%ndim
           !if(param%r(j,j,i)<zero) call wrndie(-10,'<update_S_from_CR>','matrix not positive definite')
           param%lndet(i) = param%lndet(i) + log(abs(param%r(j,j,i)+tenm20))
        end do
        param%lndet(i) = param%lndet(i) * two
     end do
     call update_Sinv(param)
end subroutine update_S_from_CR


!Impose constraints on minimum and maximum variance of Gaussians.
!If S = R^T . R and R = U . sigma . V^T, then S = V . sigma . U^T . U . sigma . V^T = V . sigma^2 . V^T.
!It follows that the singular values of R are the standard deviations of the Gaussian in each orthogonal direction.
!We therefore perform a singular value decomposition on R, and force each singular value into the appropriate range
!The new variance-covariance matrix = V . sigma . U^T . U . sigma . V^T = V . sigma^2 . V^T
!To make a new upper triangular matrix we perform a QR decomposition on sigma . V^T = QR
!then V . sigma^2 . V^T = (QR)^T . QR = R^T . Q^T . Q . R = R^T . R
!The C++ code seems to have extra transpositions, I'm not exactly sure why.
subroutine constraint_on_CR(param,minsd,maxsd,u,s,vt,work)
use chm_kinds
use number
use stream
implicit none
type(gmm_parameter) param
real(chm_real) minsd,maxsd
real(chm_real) u(1:param%ndim,1:param%ndim),s(1:param%ndim),vt(1:param%ndim,1:param%ndim),work(*) 
!work is passed to LAPACK and should be 10 * ndim.
integer i,j,k,ndim,info
     ndim=param%ndim
     do i=1,param%ngauss
          !write(*,*) "constraint_on_CR: R = ",i,param%r(1,1,i),param%r(1,2,i),param%r(2,2,i)
          call DGESVD('A','A',ndim,ndim,param%r(:,:,i),ndim,s,u,ndim,vt,ndim,work,10*ndim,info)
          !write(*,*) "constraint_on_CR: U = ",i,u(1,1),u(1,2),u(2,1),u(2,2)
          !write(*,*) "constraint_on_CR: sigma = ",i,s(1),s(2)
          !write(*,*) "constraint_on_CR: VT = ",i,vt(1,1),vt(1,2),vt(2,1),vt(2,2)
          if (info<0) then
             write(outu,*) "LAPACK error: matrix ",i," info ",info
             call wrndie(-10,'<constraint_on_CR>','LAPACK error in DGESVD')
          endif
          do j=1,ndim
             if (s(j)<minsd) s(j)=minsd
             if (s(j)>maxsd) s(j)=maxsd
          end do
          !VT becomes sigma . VT
          do j=1,ndim
             do k=1,ndim
                vt(j,k) = s(j) * vt(j,k)
             end do
          end do
          !write(*,*) "constraint_on_CR: sigma.VT = ",i,vt(1,1),vt(1,2),vt(2,1),vt(2,2)
          !param%r(:,:,i) = matmul(u,vt)
          !write(*,*) "constraint_on_CR: R' = ",i,param%r(1,1,i),param%r(1,2,i),param%r(2,2,i)
          !reuse s, u to contain junk we don't need
          call DGEQRF(ndim,ndim,vt,ndim,s,work,10*ndim,info)
          if (info<0) then
             write(outu,*) "LAPACK error: matrix ",i," info ",info
             call wrndie(-10,'<constraint_on_CR>','LAPACK error in DGEQRF')
          endif
          !write(*,*) "constraint_on_CR: R'' = ",i,param%r(1,1,i),param%r(1,2,i),param%r(2,2,i)
          param%r(1:ndim,1:ndim,i) = vt(1:ndim,1:ndim)
          !zero out the lower triangle of r
          do j=2,ndim
             param%r(j,1:j-1,i) = zero
          end do
     end do
end subroutine constraint_on_CR

!Construct an initial guess for the GMM parameters given data.  Assumes the number of gaussians and dimensions are set.
!This sets the mean of each gaussian to a randomly selected data point and the standard deviation to 30 degrees in each direction.
!Each gaussian is initially equally weighted.
subroutine guess(param,ndata,x,weight)
use chm_kinds
use reawri, only: iseed
use number
use clcg_mod,only: random
use parallel
use rndnum
implicit none
type(gmm_parameter) param
integer ndata,ig,seed
real(chm_real) x(1:param%ndim,1:ndata),weight(1:ndata)
!for testing purposes, use the same set of data points to construct the guess every time
!integer test(1:5)
!parameter(test=(/100,200,300,400,500/))
integer i,j,k
     param%r = zero
     !imitate random number generation in DLNGV, source/dynamc/dynlng.src
     !ig=iseed
     !if (.not.qoldrng) then
     !   ig = 1
#if KEY_PARALLEL==1
     !   ig = mynodp 
#endif
     !   !write(*,*) "node ",mynod," using stream ",ig," seeds ",rngseeds
     !endif
     !write(*,*) mynod,iseed
     do i=1,param%ngauss
        !initial weight 1/ngauss
        param%weight(i) = -log(real(param%ngauss))
        !initial mean, randomly chosen data point
        j = int(random(iseed)*ndata) + 1
#if KEY_PARALLEL==1
        !This is the easiest way I know of to ensure that the randomly selected data points are different on each processor. 
        j = j+int(mynod*ndata/numnod) 
        j = mod(j,ndata) 
#endif 
        !write(*,*) "data point selection: ",mynod,i,j
        !j = mod(100 * i + mynod,ndata) + 1 !test(i)
        !write(90,*) mynod,i,j
        param%mean(1:param%ndim,i) = x(1:param%ndim,j)
        !inital standard deviation, 30 degrees in every direction
        do k=1,param%ndim
           param%r(k,k,i) = thirty
        end do
     end do
     call update_S_from_CR(param)
end subroutine guess


!Do everything necessary for a GMM fit of the data.
!assumes param has been allocated as needed
subroutine gmm_fit(ngauss,ndim,nrefinements,n_em_iterations,gammacutoff,min_sigma,max_sigma,ndata,x,weight,param)
use chm_kinds
use number
use parallel
use memory
use stream
implicit none
integer ngauss,ndim,nrefinements,n_em_iterations, ndata
real(chm_real) gammacutoff,min_sigma,max_sigma
real(chm_real) x(1:ndim,1:ndata),weight(1:ndata)
type(gmm_parameter) param
!Work arrays.
real(chm_real),allocatable :: newmeans(:,:),newr(:,:,:)
real(chm_real),allocatable :: work(:),work2(:),omega(:),Wmega(:),U(:,:),S(:),VT(:,:)
integer refinement,iter,i,j,refinements,actual_refinements,oldprnlev
real(chm_real) sum_w,old_ln_like,d_ln_like,rms_m,rms_pi,mindet
type(gmm_parameter) current !Current parameters in E-M iteration
        !Memory allocation.
        call chmalloc('gmmfit.src','gmm_fit','newmeans',ngauss,ndim,crl=newmeans)
        call chmalloc('gmmfit.src','gmm_fit','newr',ndim,ndim,ngauss,crl=newr)
        call chmalloc('gmmfit.src','gmm_fit','work',ndim,crl=work)
        call chmalloc('gmmfit.src','gmm_fit','work2',10*ndim,crl=work2)
        call chmalloc('gmmfit.src','gmm_fit','omega',ngauss,crl=omega)
        call chmalloc('gmmfit.src','gmm_fit','Wmega',ngauss,crl=Wmega)
        call chmalloc('gmmfit.src','gmm_fit','u',ndim,ndim,crl=u)
        call chmalloc('gmmfit.src','gmm_fit','s',ndim,crl=s)
        call chmalloc('gmmfit.src','gmm_fit','vt',ndim,ndim,crl=vt)
        call alloc_gmm_parameter(ngauss,ndim,current) 
        
        !How many fits to perform on this processor?
#if KEY_PARALLEL==1
        actual_refinements = int(real(nrefinements)/real(numnod))
        if (mynod<mod(nrefinements,numnod)) actual_refinements = actual_refinements + 1
        !write(outu,*) "node ",mynod," will perform ",actual_refinements," refinements."
        oldprnlev = prnlev
        plnod0 = prnlev
        if (mynod /= 0) plnod0 = 0
        call gbor(plnod0,1) !So that all nodes can output independently for the time being.
        !write(outu,*) "node ",mynod," prnlev,oldprnlev = ",prnlev,oldprnlev
        !write(outu,*) "node ",mynod," weights ",(weight(i),i=1,5),(x(1,i),i=1,5)
#else /**/
        actual_refinements = nrefinements
#endif 
        if (prnlev>=2) then
#if KEY_PARALLEL==1
           write(outu,*) "Node Refinement Iteration    ln Like  d(ln Like) RMS dev. of means"
           write(outu,*) "---- ---------- ---------    -------  ---------- -----------------"
#else /**/
           write(outu,*) "Refinement Iteration    ln Like  d(ln Like) RMS dev. of means"
           write(outu,*) "---------- ---------    -------  ---------- -----------------"
#endif 
        endif
        sum_w=logsum_all(weight,ndata)
        !write(*,*) "gmm_fit: sum_w = ",sum_w

        !Ensure that no parameters already present in param are considered
        param%lnlike = -thosnd
        !Loop over the refinements        
        do refinement=1,actual_refinements
           !Initial guess
           call guess(current,ndata,x,weight)

           !E-M iteration loop
           old_ln_like = zero
           rms_m = -one
           iter = 1
           do while (.true.) !exits coded in the loop
              !call write_gmm_parameter(current,6+iter)
              !call flush(6+iter)
              call update_W(current,Wmega,omega,ndata,x,weight,work)
              current%lnlike = current%lnlike / exp(sum_w)
              d_ln_like = current%lnlike - old_ln_like
              !write(outu,*) "omega: ",omega
              !write(outu,*) "Wmega: ",Wmega
              !exit if decreased likelihood
              if ((iter>=10).and.(d_ln_like<zero)) exit
              !Test plnod0 instead of prnlev here, because we specifically want output from every node.
#if KEY_PARALLEL==1
              if (plnod0>=2) write(outu,'(I4,1X,I10,1X,I9,1X,F10.4,1X,F11.4,1X,F17.4)') &
                    mynod,refinement,iter,current%lnlike,d_ln_like,rms_m
#else /* */
              if (prnlev>=2) write(outu,'(1X,I10,1X,I9,1X,F10.4,1X,F11.4,1X,F17.4)') &
                    refinement,iter,current%lnlike,d_ln_like,rms_m
              !write(outu,*) refinement,iter,current%lnlike,d_ln_like,rms_m
#endif 
              call update_m(current,Wmega,omega,ndata,x,weight,work,newmeans,rms_m)
              call update_R(current,Wmega,omega,ndata,x,weight,work,newr)
              call update_weight(current,Wmega,sum_w)
              call constraint_on_CR(current,min_sigma,max_sigma,u,s,vt,work2)
              call update_S_from_CR(current)

              !exit if reached iterations limit
              old_ln_like = current%lnlike
              iter = iter + 1
              if (iter > n_em_iterations) exit
           end do
           
           if (current%lnlike > param%lnlike) call copy_gmm_parameter(current,param)
              !write(outu,*) "node ",mynod," replacing parameters: ",current%lnlike,param%lnlike,&
              !   current%gamma,param%gamma
           !   call copy_gmm_parameter(current,param)
           !endif
        end do

#if KEY_PARALLEL==1
        prnlev = oldprnlev
        !This determines which refinement has the maximum likelihood over all the refinements on all the processors.
        call broadcast_max_likelihood(param) 
        !write(*,*) "node ",mynod," actual_refinements before = ",actual_refinements
        call igcomb(actual_refinements,1)
        !write(*,*) "node ",mynod," actual_refinements after = ",actual_refinements
#endif 
        if (param%lnlike < gammacutoff) param%lnlike = gammacutoff
        if (prnlev>2) then
            write(outu,*) "GAMUSREWEIGHT> A total of ",actual_refinements," fits were performed."
            write(outu,*) "GAMUSREWEIGHT> Overall maximum likelihood ",param%lnlike
            mindet = two * real(ndim) * log(min_sigma) + pt001
            do i=1,ngauss
               if (param%lndet(i)<mindet) then
                  write(outu,*) "GAMUSREWEIGHT> gaussian number ",i," is of minimum size in all directions"
                  write(outu,'(A,10(F15.8,1X))') "GAMUSREWEIGHT> the gaussian is centered at ",param%mean(:,i)
               endif
            end do
        end if
         
        !Clean up.
        call chmdealloc('gmmfit.src','gmm_fit','newmeans',ngauss,ndim,crl=newmeans)
        call chmdealloc('gmmfit.src','gmm_fit','newr',ndim,ndim,ngauss,crl=newr)
        call chmdealloc('gmmfit.src','gmm_fit','work',ndim,crl=work)
        call chmdealloc('gmmfit.src','gmm_fit','work2',10*ndim,crl=work2)
        call chmdealloc('gmmfit.src','gmm_fit','omega',ngauss,crl=omega)
        call chmdealloc('gmmfit.src','gmm_fit','Wmega',ngauss,crl=Wmega)
        call chmdealloc('gmmfit.src','gmm_fit','u',ndim,ndim,crl=u)
        call chmdealloc('gmmfit.src','gmm_fit','s',ndim,crl=s)
        call chmdealloc('gmmfit.src','gmm_fit','vt',ndim,ndim,crl=vt)
        call dealloc_gmm_parameter(current)
end subroutine gmm_fit
#endif
end module gmmfit
