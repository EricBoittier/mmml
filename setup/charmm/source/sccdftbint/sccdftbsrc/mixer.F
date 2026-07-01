! MG_QC_UW1207: newly introduced file and subroutine for spin-polarization formalism
! purpose: organize charges to be fed to broyden-mixer
! MG_UW1210 (ldep)

       subroutine mixer(niter,almix,nn,qmold,qmat,lcolspin,ndim,lldim,
     &     lldep,lmax,izp,indl,qlold,ql,qlupold,qldownold,qlup,qldown,
     &     sccmix,gens,wsy)  ! Guanhua_puja_QC_UW1212
       use sccdftbsrc, only: maxtyp,nndim
       implicit none
       logical lcolspin,lldep
       integer niter,nn,ndim,lldim,lmax(maxtyp),izp(maxtyp)
       integer indl(nndim+1)
       real*8  almix,qmold(nndim),qmat(nndim),ql(3*nndim),qlold(3*nndim)
       real*8  qlup(3*nndim),qldown(3*nndim),qlupold(3*nndim)
       real*8  qldownold(3*nndim)

       real*8  qmix(6*nndim),qmixold(6*nndim)
       integer i,li,mi,mu,l
       integer sccmix,gens ! Guanhua_puja_QC_UW1212
       real*8  wsy,al_n    ! Guanhua_puja_QC_UW1212

! Guanhua_puja_QC_UW1212
       if (niter.eq.1) then
          if (sccmix.eq.0) al_n=1.0d0
          if (sccmix.eq.2) al_n=1.0d0
       else
          al_n=almix
       end if

       do i=1,6*nndim
         qmix(i)=0.0d0
         qmixold(i)=0.0d0
       enddo

!
!
! spinpolarized (change all q_al)
       if (lcolspin) then
         do i=1,lldim
           qmix(i)       = qlup(i)
           qmix(i+lldim) = qldown(i)
           qmixold(i)       = qlupold(i)
           qmixold(i+lldim) = qldownold(i)
         enddo
! Guanhua_puja_QC_UW1212
         if(sccmix.eq.0) then
           call simple_mix(al_n,2*lldim,qmixold,qmix)
#if KEY_DFTBMKL==1
         else if (sccmix.eq.1)then
           call anderson_mix(al_n,niter,gens,2*lldim,qmixold,qmix,wsy)
#endif 
         else  
           call broyden(niter,al_n,2*lldim,qmixold,qmix) 
         endif
!
         do i=1,lldim
           qlup(i)   = qmixold(i)
           qldown(i) = qmixold(i+lldim)
           qlupold(i)   = qmixold(i)
           qldownold(i) = qmixold(i+lldim)
           ql(i)     = qlup(i)+qldown(i)
         enddo
         do i = 1,nn
           qmat(i)=0.0d0
           do li = 1,lmax(izp(i))
             qmat(i) = qmat(i)+qlup(indl(i)+li)+qldown(indl(i)+li)
           enddo
         enddo
! spinunpolarized
       else
! and l-dependence Hubbard
         if (lldep) then

           do i=1,lldim
             qmix(i)=ql(i)
             qmixold(i)=qlold(i)
           enddo

           if(sccmix.eq.0) then
             call simple_mix(al_n,lldim,qmixold,qmix)
#if KEY_DFTBMKL==1
           else if (sccmix.eq.1)then
             call anderson_mix(al_n,niter,gens,lldim,qmixold,qmix,wsy)
#endif 
           else
             call broyden(niter,al_n,lldim,qmixold,qmix)
           endif

           do i=1,lldim
             ql(i) = qmixold(i)
             qlold(i) = qmixold(i)
           enddo
           do i = 1,nn
             qmat(i)=0.0d0
             do li = 1,lmax(izp(i))
               qmat(i) = qmat(i)+ql(indl(i)+li)
             enddo
           enddo
! no l-dependent Hubbard (standard DFTB2/3)
         else

           do i=1,nn
             qmix(i) = qmat(i)
             qmixold(i) = qmold(i)
           enddo

           if(sccmix.eq.0) then
             call simple_mix(al_n,nn,qmixold,qmix)
#if KEY_DFTBMKL==1
           else if (sccmix.eq.1)then
             call anderson_mix(al_n,niter,gens,nn,qmixold,qmix,wsy)
#endif 
           else
             call broyden(niter,al_n,nn,qmixold,qmix)
           endif

           do i = 1,nn
             qmat(i) = qmixold(i)
             qmold(i)= qmixold(i)
           end do
         endif
       endif

       end

! Guanhua_puja_QC_UW1212
       SUBROUTINE simple_mix(al_n,smdim,qmixold,qmix) 
       use sccdftbsrc, only: nndim
       implicit none
       real*8 qmixold(6*nndim),qmix(6*nndim)
       real*8 al_n
       integer smdim,i
       do i=1,smdim
         qmixold(i)=al_n*qmix(i)+(1.0d0-al_n)*qmixold(i)
       end do
       RETURN
       END
!

#if KEY_DFTBMKL==1
! Guanhua_puja_QC_UW1212
       SUBROUTINE ANDERSON_MIX(al_n,niter,gens,amdim,qmixold,qmix,wsy)
       use sccdftbsrc, only: nndim
       IMPLICIT NONE
!
! Anderson Mixing Subroutine; see Eyert paper for notation used
!
! Guanhua_QC_1002: add modification to break linear dependency near convergence region
       REAL*8 qmix(6*nndim),qmixold(6*nndim),ddot,amix,wsy
       INTEGER niter,i,j,k,l,m,n,gens,amdim,maxind,info
       REAL*8 xmiddle(6*nndim),fmiddle(6*nndim),theta(15),al_n
       REAL*8 A(15,15),B(15),helpa(6*nndim),helpb(6*nndim)
       REAL*8 helpc(6*nndim)
       INTEGER IPIV(15)
!       
! bookkeeping arrays for anderson mixing       
! array is large enough to hold maximal 16 x and F vectors
!
       REAL*8 tmix(3,2)
       LOGICAL exist
       REAL*8 anderson(6*nndim,32)
       SAVE anderson       
!       
!       write(*,*) 'Start Anderson',gens,niter

!      JZ_QC_UW_0611: Initialize A and B just to make sure
       DO i=1,15
         B(i)=0.0d0
         DO j=1,15
           A(i,j)=0.0d0
         ENDDO
       ENDDO

! puja- 'gens' here includes the current iteration
! so maximum no. of previous iterations allowed is 15

       IF (gens.GT.16) THEN
         write(*,*)'Too many generations for anderson mixing, using 16'
         gens=16
       END IF  
       DO i=1,6*nndim
         xmiddle(i)=0.0d0
         fmiddle(i)=0.0d0
       END DO
       IF (niter.LE.gens) THEN
         maxind=niter
         amix=0.01d0
       ELSE
         maxind=gens
         amix=al_n
       END IF  
!      write(*,*) 'max index is ',maxind,amix
       DO j=2,gens
         DO i=1,amdim
           anderson(i,j-1)=anderson(i,j)
           anderson(i,gens+j-1)=anderson(i,gens+j)
         END DO
       END DO

       DO i=1,amdim
         anderson(i,gens)=qmixold(i)
         anderson(i,2*gens)=qmix(i)-qmixold(i)
         helpc(i)=anderson(i,2*gens)
       END DO
!
! now start anderson; note that gens might have been changed
! in case of startup, see above
!
! setup system of equations for theta
!
!      write(*,*) 'Setting up linear equation system'
!       open(70,file='scfsum',form='formatted',status='unknown',
!     &      access='append')

! Puja-We should be solving a system of (maxind-1) linear equations
! since we need theta only for previous iterations, not current one

! Puja_QC_UW1212
       if(niter.eq.1) goto 200
!         DO i=1,amdim
!           qmixold(i)=anderson(i,gens)+0.5d0*anderson(i,2*gens)
!         END DO
!         RETURN
!       endif
!
 100   DO i=1,maxind-1  !Puja_QC_UW1212
         DO k=1,amdim
           helpa(k)=anderson(k,2*gens)-anderson(k,2*gens-i)
         END DO
         B(i)=DDOT(amdim,helpa,1,helpc,1)
         DO j=1,maxind-1
           DO k=1,amdim
             helpb(k)=anderson(k,2*gens)-anderson(k,2*gens-j)
           END DO
           A(i,j)=DDOT(amdim,helpa,1,helpb,1)
         END DO
       END DO
       DO j=1,maxind-1
         A(j,j)=A(j,j)*(1.0d0+wsy*wsy)
       END DO

!
! solve for theta
!
!       write(*,*) 'Solving linear equation system'
       
       CALL DGESV(maxind-1,1,A,15,IPIV,B,15,info)
      
       IF (INFO.LT.0) THEN
!         write(70,'(A)') 'Illegal value in mixer; STOP'
         write(*,'(A)') 'Illegal value in mixer; STOP'
         STOP
       END IF  

       IF (info.GT.0) THEN
         write(*,'(A)') '# WARNING: linear dependency in mixer, STOP'
!         DO i=1,maxind-3
!           theta(i)=0.0d0
!           theta(maxind-1)=1.0d0
!           amix=0.05d0
!         END DO
         STOP
       END IF

       DO i=1,maxind-1
         theta(i)=b(i)
       END DO

!       close(70)
!
! form xmiddle
! 
 200   CONTINUE  
       DO j=1,maxind-1
         DO i=1,amdim
           xmiddle(i)=xmiddle(i)+theta(j)*
     &       (anderson(i,gens-j)-anderson(i,gens))
         END DO
       END DO
       DO i=1,amdim
         xmiddle(i)=xmiddle(i)+anderson(i,gens)
       END DO
!
! form fmiddle
!
       DO j=1,maxind-1
         DO i=1,amdim
           fmiddle(i)=fmiddle(i)+theta(j)*
     &       (anderson(i,2*gens-j)-anderson(i,2*gens))
         END DO
       END DO
       DO i=1,amdim
         fmiddle(i)=fmiddle(i)+anderson(i,2*gens)
       END DO
!
! get final new guess vector
!
!       open(70,file='ANDERSON.LOG',form='formatted',status='unknown',
!     &      access='append')
!         write(70,'(A,I5,A,1PD10.2)')'ITER: ',
!     &     niter,', Change: ',DSQRT(DDOT(2*ndim,helpc,1,helpc,1))
!       close(70)
!       IF (DSQRT(DDOT(2*ndim,helpc,1,helpc,1)).LT.1.0d-02) THEN
!         amix=0.1d0
!       END IF  
!       IF (DSQRT(DDOT(2*ndim,helpc,1,helpc,1)).LT.1.0d-03) THEN
!         amix=0.3d0
!       END IF  
!       IF (DSQRT(DDOT(2*ndim,helpc,1,helpc,1)).LT.1.0d-04) THEN
!         amix=0.5d0
!       END IF  
       tmix(1,1)=1.0d-02
       tmix(1,2)=0.1d0
       tmix(2,1)=1.0d-03
       tmix(2,2)=0.3d0
       tmix(3,1)=1.0d-04
       tmix(3,2)=0.5d0
!       INQUIRE(FILE='MIXER_CONF.DAT',EXIST=EXIST)
!       IF (EXIST) THEN
!        OPEN(65,FILE='MIXER_CONF.DAT',FORM='FORMATTED',STATUS='UNKNOWN')
!         REWIND(65)
!         READ(65,*) tmix(1,1),tmix(1,2)
!         READ(65,*) tmix(2,1),tmix(2,2)
!         READ(65,*) tmix(3,1),tmix(3,2)
!         CLOSE(65)
!       ELSE
!        OPEN(65,FILE='MIXER_CONF.DAT',FORM='FORMATTED',STATUS='UNKNOWN')
!         REWIND(65)
!         WRITE(65,*) tmix(1,1),tmix(1,2)
!         WRITE(65,*) tmix(2,1),tmix(2,2)
!         WRITE(65,*) tmix(3,1),tmix(3,2)
!         CLOSE(65)
!       END IF  
       DO i=1,3
         IF (DSQRT(DDOT(amdim,helpc,1,helpc,1)).LT.tmix(i,1)) THEN
           amix=tmix(i,2)
         END IF
       END DO
       DO i=1,amdim
         qmixold(i)=xmiddle(i)+amix*fmiddle(i)
       END DO
       RETURN
       END
#endif 

