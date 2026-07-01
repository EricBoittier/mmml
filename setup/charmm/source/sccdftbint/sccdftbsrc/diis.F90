#if KEY_DFTBMKL==1
subroutine diis(iprint,niter,mdiis,ndiis,nn,nndim,mxcdim, &
                qold,qnew,qdiff,qall,b,c,iextra)
!
!   diis algorithm(direct inversion in iterative space) 
!   to extrapolate a new qold for next iteration calculation
!   
!   extrapolated vector is returned in "qnew"
!   
!   ref: p.pulay chem. phys. lett. 73, 393 (1980)
!
!   qold/qnew: old/new parameter vector
!   qdiff: parameter difference matrix
!   qall:  parameter matrix
!  mdiis:  maximum diis iterations
!  ndiis:  current diis iteration index
!      b:  diis coefficient matrix
!      c:  contant coeffientvector
!
!   guishan zheng
!   harvard univ.
!   10/19/2014
!
        implicit none
        integer, parameter:: mau=20
        integer :: iprint,niter,mdiis,ndiis,nn,nndim,mxcdim,iextra
        real*8 :: qold(*), qnew(*), qdiff(nndim,*), qall(nndim,*), c(*)
        real*8 :: b(mxcdim,*) , zero, one
        integer ipiv(mau+1),info, i, j, k, np
        data zero/0.0d0/, one/1.0d0/
        save zero, one

        if(mau.lt.mxcdim) then
           write(*,*) 'diis: auxiliary matrix dim is lower than', mxcdim
           stop
        endif
        ndiis = ndiis + 1
! if scf iteration index is more than maximum diis iteration, then 
! move the diis vectors
        if(ndiis.gt.mdiis) then
           do 10 i = 1, mdiis-1
              qdiff(1:nn,i) = qdiff(1:nn,i+1)
              qall(1:nn,i) = qall(1:nn,i+1)
10         continue
        endif
!       calculate the current iteration gradient/error function
        np = min(ndiis,mdiis)
        qdiff(1:nn,np) = qnew(1:nn) - qold(1:nn)
        qall(1:nn,np) = qnew(1:nn)
!       construct b matrix and coefficient vector 
        b(1:np+1,1:np+1) = zero
        do 20 i = 1, np
        do 20 j = 1, i 
           b(i,j) = b(i,j) + sum(qdiff(:,i)*qdiff(:,j))
           b(j,i) = b(i,j)
20      continue
        do 30 i = 1, np+1
           b(i,np+1) = -one
           b(np+1,i) = -one
           c(i) = zero
30      continue
        b(np+1,np+1) = zero
        c(np+1) = -one
!       solve a*c=b 
        call dgesv(np+1,1,b,mxcdim,ipiv,c,mxcdim,info)
        if(info.ne.0) then
            write(*,*) 'diis: not successful in solving b*x=c'
            stop
        endif
!       extrapolation for a new vector
        qnew(1:nn) = zero
        if(iextra.eq.0) then
           do 40 j = 1, nn
              qnew(j) = qnew(j) + sum(c(1:np)*qall(j,1:np))
40         continue
        endif
        if(iprint.ge.6) then
           write(*,*) 'diis: ndiis=',ndiis,'np=',np
           write(*,*) 'diis: previous , current , and diff vector'
          write(*,'(3(d20.10,2x))') (qold(i),qnew(i),qdiff(i,np),i=1,nn)
           write(*,*) 'diis: solution coef.'
        endif
        return
end
#endif
