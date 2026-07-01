! PZ, MG_QC_UW1206 adapted
       SUBROUTINE sccpme2a(x,nn,xE,nE,zE,recbasis,vol,shiftE,ksgrd,
     &                     Kappa)

       use sccdftbsrc, only: NNDIM
       use sccdftb,    only: MAXPTC
       use sccpme_module
       use sccpmeutil
! ASC: Add parallel
#if KEY_PARALLEL==1
       use parallel
#endif

       implicit none

          real*8 zero
          parameter(zero=0.D0)
! Passed in
          integer nn,nE
          real*8 x(3,NNDIM),shiftE(NNDIM),ksgrd(3*NNDIM),Kappa
          real*8 xE(3,MAXPTC),ZE(MAXPTC)
          real*8 vol,recbasis(3,3)
       
! Local variables
          integer i,j,ii,atfrst,atlast
          real*8 recip(3,3),empot(NNDIM),pme_ksp_grad(3,NNDIM)
          integer siz_q,siztheta,sizdtheta,alloc_err
          integer nfftdim1, nfftdim2, nfftdim3
          integer xnsymm
          xnsymm=1
          empot=zero
          pme_ksp_grad=zero

          recip=recbasis*0.15915493866
          call get_fftdims(nfftdim1,nfftdim2,nfftdim3,i,j)
          siztheta  = (nn+nE)*xnsymm*forder
          sizdtheta = (nn+nE+1)*forder
          nattot=(nn+nE)*xnsymm
          SIZ_Q = 2*NFFTDIM1*NFFTDIM2*NFFTDIM3
          if(.not.allocated(qarray)) then
             allocate(qarray(siz_q),stat=alloc_err)
             if(alloc_err /= 0 ) write(0,*)"unable to allocate qarray"
          end if
          qarray=zero
          if(.not.allocated(qarray_mm)) then
             allocate(qarray_mm(siz_q),stat=alloc_err)
             if(alloc_err /= 0 ) 
     &           write(0,*)"unable to allocate qarray_mm"
          end if
!          if(allocated(fr1) .and. nattot.gt.size(fr1)) deallocate(fr1,fr2,fr3,cg1)
          if(.not. allocated(fr1)) then
            allocate(fr1(nattot),fr2(nattot),fr3(nattot),cg1(nattot),
     &      stat=alloc_err)
            if(alloc_err /= 0 ) write(0,*)"unable to allocate fr arrays"
          ! initialize
             fr1(1:nattot)=zero
             fr2(1:nattot)=zero
             fr3(1:nattot)=zero
             cg1(1:nattot)=zero
          end if
          if(.not.allocated(theta1)) then
             call allocate_bspline(nn+nE,nn+nE)
          endif
        
          call do_pme_ksp_scc_pot(nE,xE,nn,x,zE,recip,vol, 
     &                    kappa,forder,empot,pme_ksp_grad, 
     &                    sizfftab,sizffwrk,siztheta,siz_q, 
     &                    xnsymm,shiftE)
! ASC: Add parallel
#if KEY_PARALLEL==1
      if (numnod.gt.1) call GCOMB(empot,nn)
#endif
          shiftE(1:nn)=empot(1:nn)
          do i=1,nn
             ksgrd(3*i-2)=pme_ksp_grad(1,i)
             ksgrd(3*i-1)=pme_ksp_grad(2,i)
             ksgrd(3*i)=pme_ksp_grad(3,i)
          enddo ! i=1,nn
          return
       END SUBROUTINE sccpme2a
       SUBROUTINE do_pme_ksp_scc_pot(natom,xE,nquant,x,cg,recip,vol, 
     &                kappa,forder,empot,pme_ksp_grad,  
     &                 sizfftab,sizffwrk,siztheta,siz_q, 
     &                 xnsymm,shiftE)
       use sccdftb,    only: MAXPTC
       use sccpme_module
       use sccpmeutil,only:nxyslab,mxyslabs,mxzslabs,nfft1,nfft2, 
     &             nfft3,get_sc_fract,  
     &             get_fftdims,fft3d0rc,fft3d_zxyrc
       use exfunc
       use number
       use grape
       use memory
       implicit none
! Passed in
          integer        :: forder
          real*8         :: recip(3,3),vol,kappa
          real*8         :: x(3,*),xE(3,*),empot(*),pme_ksp_grad(3,*)
          real*8         :: cg(*)
          integer        :: natom,nquant,xnsymm
          integer        :: sizfftab,sizffwrk,siztheta,siz_q    ! sizes of some arrays
          real*8         :: scale,shiftE(*)
! Local variables
          integer nfftdim1,nfftdim2,nfftdim3,nfftable,nffwork
          integer latm
          integer igood, kbot, ktop, i0,i
          real*8         :: xx(MAXPTC),yy(MAXPTC),zz(MAXPTC),vir(6)
          real*8         :: cg_local(MAXPTC)
          logical        :: q_grad_and_pot

          q_grad_and_pot = .true.          ! .true. for potential calculation
                                           ! .false. for energy gradient calculation
          nattot=(natom+nquant)*xnsymm
          xx(1:natom)=xE(1,1:natom)
          xx(natom+1:natom+nquant)=x(1,1:nquant)
          yy(1:natom)=xE(2,1:natom)
          yy(natom+1:natom+nquant)=x(2,1:nquant)
          zz(1:natom)=xE(3,1:natom)
          zz(natom+1:natom+nquant)=x(3,1:nquant)
          cg_local(1:natom)=cg(1:natom)
          cg_local(natom+1:natom+nquant)=zero
          
          !  get some integer array dimensions
          call get_fftdims(nfftdim1,nfftdim2,nfftdim3,nfftable,nffwork)
          scale = 1.D0

          if(allocated(tmpy) .and. 2*nfftdim1.ne.size(tmpy)) 
     &      deallocate(tmpy)
          if(allocated(alpha) .and. nfft1.ne.size(alpha))
     &      deallocate(alpha,beta)
          if(.not.allocated(tmpy)) allocate(tmpy(2*nfftdim1))
          if(.not.allocated(alpha)) allocate(alpha(nfft1),beta(nfft1))
          call fft3d0rc(0,scale,qarray,nfftdim1,nfftdim2,tmpy,
     &      alpha,beta)
          if(xnsymm>1) call wrndie(-1,'<DO_PME_KSP_SCC_POT>',
     &      'not support xnsymm > 1')
          latm=nattot
          call chmalloc('long_sccpme.src','do_pme_ksp_scc_pot',
     &      'lmy_ks',latm,intg=lmy_ks)
          call chmalloc('long_sccpme.src','do_pme_ksp_scc_pot',
     &      'lmy_ks_inv',latm,intg=lmy_ks_inv)

          if(xnsymm.eq.1) then
             call fill_ch_grid(igood, kbot, ktop,nattot,cg_local,
#if KEY_BLOCK==1
     &           CG1,                                      
#endif
     &           xx,yy,zz,recip,natom,xnsymm,
     &           nfftdim1,nfftdim2,
     &           lmy_ks,latm,lmy_ks_inv,
     &           mxyslabs)
          ELSE
             if(xnsymm>1) call wrndie(-1,'<DO_PME_KSP_SCC_POT>',
     &        'not support xnsymm > 1')
          endif
          call fft_backrc(Qarray,nfftdim1,nfftdim2,nfftdim3,nffwork)
          qarray_mm(1:siz_q)=qarray(1:siz_q)
          call convol_fr_space_qm_mm(qfinit,rewcut,kappa,vol,recip,
     &                 nfftdim1,nfftdim2,nfftdim3,Qarray,Qarray_mm,vir,
     &                 q_grad_and_pot)
          call fft_forwardrc(Qarray,nfftdim1,nfftdim2,nfftdim3,nffwork)
          call potential_sumrc_qm_mm(igood, kbot, ktop, natom, nquant,
     &                 Qarray,empot,pme_ksp_grad, 
     &                 recip,vol,forder,nfftdim1,nfftdim2,nfftdim3,
     &                 lmy_ks_inv,latm,xnsymm)
!          deallocate(tmpy,alpha,beta)
          call chmdealloc('long_sccpme.src','do_pme_ksp_scc_pot',
     &      'lmy_ks',latm,intg=lmy_ks)
          call chmdealloc('long_sccpme.src','do_pme_ksp_scc_pot',
     &      'lmy_ks_inv',latm,intg=lmy_ks_inv)

          return
       END SUBROUTINE do_pme_ksp_scc_pot
       subroutine convol_fr_space_qm_mm(qfinit_local,
     &                  rewcut_local,ewaldcof,volume,recip,
     &                  nfftdim1,nfftdim2,nfftdim3, 
     &                  Qarray_local,Qarray_mm_local,vir,q_grad_and_pot)
       use sccpme_module
       use sccpmeutil,only:mxzslabs,mxzstart,
     &                      nfft1,nfft2,nfft3,bsp_mod1,bsp_mod2,bsp_mod3
       use number
       use consta
! ASC: add parallel
#if KEY_PARALLEL==1
       use parallel, only: mynod
#endif
       implicit none
! Passed in
          LOGICAL,intent(in) :: QFINIT_local
          real*8, intent(in) ::  REWCUT_local
          INTEGER,intent(in) :: NFFTDIM1,NFFTDIM2,NFFTDIM3
          real*8, intent(in) :: EWALDCOF,RECIP(9),VOLUME
          real*8, intent(inout) :: Qarray_local(*),Qarray_mm_local(*)
          real*8, intent(inout) :: vir(6)
          logical,intent(in) :: q_grad_and_pot             ! =.true., when computing potential.
                                                            ! =.false., when computing gradient.
! Local variables      
          real*8         :: FAC,ETERM,VTERM
          real*8         :: DEN1,DEN2,DEN3,DEN4
          INTEGER        :: K,K0,K1,K2,K3,K2Q,M1,M2,M3
          INTEGER        :: K1s,K2s,K3s,M1s,M2s,M3s
          INTEGER        :: IPT1,IPT2,IPT3,NF1,NF2,NF3
          real*8         :: MVAL,MCUT,MSQ,MSQR,STRUC2,VCORR,ESTR
          real*8         :: MHAT(3),MHATA(3),MHATB(3)
          real*8         :: MHATs(3)
          real*8         :: DENs,ETERMs,VTERMs,ESTRs,MSQs,STRUC2s,MSQRs
          LOGICAL        :: QFIN
          FAC = PI**2/EWALDCOF**2
          MCUT= TWO*PI*REWCUT_local
          QFIN=QFINIT_local
      
          NF1 = NFFT1/2
          IF(2*NF1 < NFFT1) NF1 = NF1+1
          NF2 = NFFT2/2
          IF(2*NF2 < NFFT2) NF2 = NF2+1
          NF3 = NFFT3/2
          IF(2*NF3 < NFFT3) NF3 = NF3+1
      
          DEN1 = ONE/(PI*VOLUME)

! ASC: add parallel
#if KEY_PMEPLSMA==1
#if KEY_PARALLEL==1
            if(mynod == 0)then
#endif
       qarray_local(1:2)   =zero
#if KEY_PARALLEL==1
        endif
#endif
#endif
          if(q_grad_and_pot) then            ! when computing potential contribution.
             IPT1=1
             do K2Q = 1, MXZSLABS
                K2=K2Q
! ASC: add parallel
#if KEY_PARALLEL==1
             IF(MYNOD > 0) K2 = K2Q + MXZSTART(MYNOD)
#endif
                M2 = K2 - 1
                IF(K2 > NF2) M2 = M2 - NFFT2
                DEN2       = DEN1*BSP_MOD2(K2)
                MHATA(1:3) = RECIP(4:6)*M2
      
                IPT2=IPT1
                do K1 = 1, NF1+1
                   M1 = K1 - 1
                   IF(K1 > NF1) M1 = M1 - NFFT1
                   DEN3       = DEN2*BSP_MOD1(K1)
                   MHATB(1:3) = MHATA(1:3)+RECIP(1:3)*M1
      
                   IPT3=IPT2
                   IF(K1+K2 == 2) THEN
                      K0=2
                      IPT3=IPT3+2
                   ELSE
                      K0=1
                   ENDIF
      
                   do K3 = K0,NFFT3
                      M3 = K3 - 1
                      IF(K3 > NF3) M3 = M3 - NFFT3
                      DEN4      = DEN3*BSP_MOD3(K3)
                      MHAT(1:3) = MHATB(1:3)+RECIP(7:9)*M3
                      MSQ       = MHAT(1)*MHAT(1)+MHAT(2)*MHAT(2) + 
     &                            MHAT(3)*MHAT(3)
                      MSQR      = ONE/MSQ
      
                      ETERM = EXP(-FAC*MSQ)*DEN4*MSQR
      
                      IF(QFIN) THEN
                         MVAL=MCUT*SQRT(MSQ)
                         ETERM=ETERM*(ONE-COS(MVAL))
                      ENDIF
                      !
                      Qarray_local(IPT3)  = ETERM * Qarray_local(IPT3)
                      Qarray_local(IPT3+1)= ETERM * Qarray_local(IPT3+1)
                      !
                      IPT3=IPT3+2
                   end do
                   IPT2=IPT2+NFFT3*2
                end do
                IPT1=IPT1+NFFT3*NFFTDIM1*2
             end do
          endif
          return
       end subroutine convol_fr_space_qm_mm

       subroutine potential_sumrc_qm_mm(igood,kbot, ktop,numatoms,natqm,
     &                qarray_local,ewd_potential,qm_atm_grad_comp_local,
     &                recip,volume,ordr,nfftdim1,nfftdim2,nfftdim3,     
     &                my_ks_inv,latm,xnsymm)
       use sccpme_module
       use sccpmeutil,only: mxystart,mxyslabs,mxzslabs, 
     &              nfft1,nfft2,nfft3, 
     &              theta1,theta2,theta3,dtheta1,dtheta2,dtheta3   
! ASC: add parallel
#if KEY_PARALLEL==1
       use parallel, only: mynod
#endif
       use number
       implicit none
! Passed in
          integer,intent(in) :: igood, kbot, ktop
          integer,intent(in) :: numatoms,natqm,ordr
          integer,intent(in) :: nfftdim1,nfftdim2,nfftdim3,xnsymm,latm
          real*8,intent(inout)   :: qarray_local(*), ewd_potential(*),
     &                              qm_atm_grad_comp_local(3,*)
          integer,intent(in) :: my_ks_inv(*)
          real*8 ,intent(in) :: recip(9),volume
! Local variables
          integer :: igoo,ig,iqm,n
          integer :: I,J,K,KQ,i_keep,j_keep,k_keep
          integer :: ITH1,ITH2,ITH3,IPT1,IPT2,IPT3
          integer :: rcskip,nfftdimrc                   ! RCFFT addition
          real*8  :: CFACT,Pot,P_tmp1,P_tmp2
          real*8  :: fxyz(3),vala(3),val(3),ufact,CFACT2
          real*8,parameter:: BOHRS_TO_A = 0.529177249D0
          real*8,parameter:: AU_TO_KCAL = 627.48981D0

          rcskip=1
          if(nfft1/2 /= (nfft1+1)/2)CALL WRNDIE(-5,'<grad_sumrc_qm_mm>',
     &     'fftx dimension not even ')
          nfftdimrc=nfft1+4
          !
          ! ufact      = AU_TO_KCAL*BOHRS_TO_A
          ufact = one
          if(xnsymm.eq.1)then
             CFACT=one                    ! CCELEC, since unit conversion
             CFACT2=ufact
          else
             CFACT=one/XNSYMM             ! CCELEC/XNSYMM
             CFACT2=ufact/XNSYMM
          end if
          !
          qm_atm_grad_comp_local(1:3,1:natqm)=zero
 
          loopig: do iqm=1,natqm
             n=numatoms+iqm               ! qm atoms are listed at the end of arrays
             igoo=my_ks_inv(n)
             ig=igoo

! ASC: Add parallel
#if KEY_PARALLEL==1
        ! skip if igoo .eq. 0
        if(igoo.le.0) cycle loopig
#endif

             K_keep = INT(FR3(igoo)) - ORDR + 1 + NFFT3
             J_keep = INT(FR2(igoo)) - ORDR + 1 + NFFT2
             I_keep = INT(FR1(igoo)) - ORDR + 1 + NFFT1
             K        = k_keep
             Pot      = ZERO
             fxyz(1:3)= zero
             do ITH3 = 1,ORDR
                K=K+1
                IF(K > NFFT3) K=K-NFFT3
                KQ=K
!ASC: Add parallel
#if KEY_PARALLEL==1
           if ( K  >=  KBOT .AND. K  <=  KTOP ) then
              KQ = K - MXYSTART(MYNOD)
#endif
                IPT1=(KQ-1)*NFFTDIM2 -1
                J = j_keep
                I = i_keep
                IF(I >= NFFT1) I=I-NFFT1
    
                P_tmp1 = THETA3(ITH3,ig)   ! NFFT1*NFFT2*NFFT3/volume, it may not be correct!
                vala(1)= NFFT1*THETA3(ITH3,ig)
                vala(2)= NFFT2*THETA3(ITH3,ig)
                vala(3)= NFFT3*DTHETA3(ITH3,igoo)
    
                do ITH2 = 1,ORDR
                   J=J+1
                   IF(J > NFFT2) J=J-NFFT2
    
                   P_tmp2 = P_tmp1*THETA2(ITH2,ig)
                   val(1) = vala(1)*THETA2(ITH2,ig)
                   val(2) = vala(2)*DTHETA2(ITH2,igoo)
                   val(3) = vala(3)*THETA2(ITH2,ig)
    
                   IPT2= rcskip*((IPT1+J)*NFFTDIMrc+I)+1
                   IPT3= IPT2 + rcskip*(NFFT1-I)
                   do ITH1 = 1,ORDR
                      Pot=Pot+P_tmp2*qarray_local(IPT2)*THETA1(ITH1,ig)
                      fxyz(1)=fxyz(1)+val(1)*qarray_local(IPT2)*
     &                  DTHETA1(ITH1,igoo)
                      fxyz(2)=fxyz(2)+val(2)*qarray_local(IPT2)*
     &                  THETA1(ITH1,ig)
                      fxyz(3)=fxyz(3)+val(3)*qarray_local(IPT2)*
     &                  THETA1(ITH1,ig)
                      IPT2=IPT2+rcskip
                      IF(IPT2 >= IPT3) IPT2=IPT2-NFFT1*rcskip

                   end do
                end do
!ASC: Add parallel
#if KEY_PARALLEL==1
          end if
#endif
             end do
             ewd_potential(iqm)= ewd_potential(iqm) + CFACT*Pot 
             qm_atm_grad_comp_local(1,iqm)=qm_atm_grad_comp_local(1,iqm)
     &      +CFACT2*(recip(1)*fxyz(1)+recip(4)*fxyz(2)+recip(7)*fxyz(3))
             qm_atm_grad_comp_local(2,iqm)=qm_atm_grad_comp_local(2,iqm)
     &      +CFACT2*(recip(2)*fxyz(1)+recip(5)*fxyz(2)+recip(8)*fxyz(3))
             qm_atm_grad_comp_local(3,iqm)=qm_atm_grad_comp_local(3,iqm)
     &      +CFACT2*(recip(3)*fxyz(1)+recip(6)*fxyz(2)+recip(9)*fxyz(3))
          enddo loopig
          return
       end subroutine potential_sumrc_qm_mm

       subroutine do_pme_ksp_grad_mm(nn,nE,x,xE,zE,qmchg,ksgrd_mm,
     &      recbasis,vol,Kappa)
       use sccpme_module
       use sccpmeutil
       use sccdftbsrc, only: NNDIM
       use sccdftb, only: MAXPTC
!ASC: Add parallel
#if KEY_PARALLEL==1
       use parallel
#endif
       implicit none

          real*8 zero
          parameter(zero=0.D0)
! Passed in
          integer nn,nE
          real*8 x(3,NNDIM),shiftE(NNDIM),Kappa
          real*8 xE(3,MAXPTC),ZE(MAXPTC),ksgrd_mm(3,MAXPTC),qmchg(NNDIM)
          real*8 vol,recbasis(3,3)

! Local variables
          integer i,j,ii,atfrst,atlast
          real*8 recip(3,3),empot(MAXPTC),pme_ksp_grad(3,MAXPTC)
          integer siz_q,siztheta,sizdtheta,alloc_err
          integer nfftdim1, nfftdim2, nfftdim3
          integer xnsymm
          xnsymm=1
          empot=zero
          pme_ksp_grad=zero

          recip=recbasis*0.15915493866
          call get_fftdims(nfftdim1,nfftdim2,nfftdim3,i,j)
          siztheta  = (nn+nE)*xnsymm*forder
          sizdtheta = (nn+nE+1)*forder
          nattot=(nn+nE)*xnsymm
          SIZ_Q = 2*NFFTDIM1*NFFTDIM2*NFFTDIM3
          if(.not.allocated(qarray)) then
             allocate(qarray(siz_q),stat=alloc_err)
             if(alloc_err /= 0 ) write(0,*)"unable to allocate qarray"
          end if
          qarray=zero
          if(.not.allocated(qarray_mm)) then
             allocate(qarray_mm(siz_q),stat=alloc_err)
             if(alloc_err /= 0 )write(0,*)"unable to allocate qarray_mm"
          end if
!          if(allocated(fr1) .and. nattot.gt.size(fr1)) deallocate(fr1,fr2,fr3,cg1)
          if(.not. allocated(fr1)) then
             allocate(fr1(nattot),fr2(nattot),fr3(nattot),cg1(nattot),
     &         stat=alloc_err)
             if(alloc_err /= 0 )write(0,*)"unable to allocate fr arrays"
          ! initialize
             fr1(1:nattot)=zero
             fr2(1:nattot)=zero
             fr3(1:nattot)=zero
             cg1(1:nattot)=zero
          end if
          if(.not.allocated(theta1)) then
             call allocate_bspline(nn+nE,nn+nE)
          endif

          call do_pme_ksp_scc_pot(nn,x,nE,xE,qmchg,recip,vol,
     &                    kappa,forder,empot,pme_ksp_grad,
     &                    sizfftab,sizffwrk,siztheta,siz_q,
     &                    xnsymm,shiftE)

! ASC: Add parallel
#if KEY_PARALLEL==1
      if (numnod.gt.1) call GCOMB(empot,nn)
      ! Note: Don't combine grad here, is combined in sccphi2f
      ! if (numnod.gt.1) call GCOMB(pme_ksp_grad,nE*3)
#endif
          do i=1,nE
             ksgrd_mm(1:3,i)=pme_ksp_grad(1:3,i)
          enddo ! i=1,nn
          return
       end subroutine do_pme_ksp_grad_mm


