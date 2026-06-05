module qmmmpme_module
  use chm_kinds
  use number
  use consta

  contains

#if KEY_MNDO97==1 /*mndo97*/
!=================================================================================================!
!========                          PME Section Begins Here                                ========!
!=================================================================================================!
  SUBROUTINE qm_pme_mm_grad(natom,nquant,iqmatoms,x,y,z,d_ewald_mm,cg,  &
                            scf_mchg_2,recip_vec,volume,kappa,ewvirial)
  !
  ! Compute the gradient at the QM and MM atom position due to
  ! the reciprocal space summation. Called after the scf convergence.
  !
  ! Calculate K space PME gradient at QM and MM atom position
  !
  use qm1_constant
  use qm1_info, only: qm_control_r
  use mndo97, only: lqmewd

#if KEY_PARALLEL==1
  use parallel
#endif
  use pme_module
  use pmeutil
  !
  use dimens_fcm
  use exfunc
  use number
  use inbnd
  use image
  use stream
  use memory
  !
  implicit none

  ! Passed in
  integer, intent(in)    :: natom,nquant
  integer, intent(in)    :: iqmatoms(nquant)

  real(chm_real),intent(in)    :: x(*),y(*),z(*),d_ewald_mm(3,*),cg(*)
  real(chm_real),intent(in)    :: kappa
  real(chm_real),intent(in)    :: scf_mchg_2(nquant),recip_vec(6),volume
  real(chm_real),intent(inout) :: ewvirial(9)

  ! Local variables
  integer        :: i,j,latm
  integer        :: alloc_err
  real(chm_real) :: recip(3,3),virial(6),cfact
  integer        :: siztheta, sizdtheta, siz_q
  integer        :: nfftdim1, nfftdim2, nfftdim3

  logical        :: ok

  if(qm_control_r%q_do_cpmd_pme) then
     ! these are deallocated in do_pme_ksp_qm_mm_grad.
     nattot=natom*xnsymm
     latm=nattot
     if(allocated(tmpy) ) deallocate(tmpy)
     if(allocated(alpha)) deallocate(alpha)
     if(allocated(beta) ) deallocate(beta)
     call chmdealloc('mndo97_pme_module.src','qm_pme_mm_grad','lmy_ks',latm,intg=lmy_ks)
     call chmdealloc('mndo97_pme_module.src','qm_pme_mm_grad','lmy_ks_inv',nattot,intg=lmy_ks_inv)

  else
     RECIP(1,1) = recip_vec(1)
     RECIP(2,2) = recip_vec(3)
     RECIP(3,3) = recip_vec(6)
     RECIP(1,2) = recip_vec(2)
     RECIP(2,1) = recip_vec(2)
     RECIP(1,3) = recip_vec(4)
     RECIP(3,1) = recip_vec(4)
     RECIP(2,3) = recip_vec(5)
     RECIP(3,2) = recip_vec(5)

     !-------------------------------------------------------------------
     ! INPUT
     !      NFFT1,NFFT2,NFFT3 are the (integer) dimensions of the charge grid array
     !      NATOM is number of atoms
     !      FORDER is the order of B-spline interpolation
     !      x,y,z:   atomic coords
     !      CG  atomic charges
     !      recip_vec: array of reciprocal unit cell vectors
     !      VOLUME: the volume of the unit cell
     !      KAPPA=ewald_coeff:   ewald convergence parameter
     ! OUTPUT
     !      siz_Q=3d charge grid array
     !      sizfftab is permanent 3d fft table storage
     !      sizffwrk is temporary 3d fft work storage
     !      siztheta is size of arrays theta1-3 dtheta1-3
     !      d_ewald_mm: forces incremented by k-space sum
     !      EWVIRIAL=virial:  virial due to k-space sum (valid for atomic scaling;
     !                rigid molecule virial needs a correction term not
     !                computed here
     !
     call get_fftdims(nfftdim1,nfftdim2,nfftdim3,i,j)
     siztheta  = natom*xnsymm*forder
     sizdtheta = (natom+1)*forder
     nattot=natom*xnsymm
#if KEY_PARALLEL==1
     SIZ_Q = max(2*NFFTDIM1*NFFTDIM2*mxyslabs,2*NFFTDIM1*NFFTDIM3*mxzslabs)
#else
     SIZ_Q = 2*NFFTDIM1*NFFTDIM2*NFFTDIM3
#endif
     !

     !==================QM-MM interactions==============================
#if KEY_COLFFT==1 /*colfft*/
     !==================COLUMN FFT METHOD ==============================
#if KEY_QUANTUM==1 || KEY_MNDO97==1 || KEY_SQUANTM==1 || KEY_GAMESS==1 || \
    KEY_GAMESSUK==1 || KEY_QCHEM==1 || KEY_QTURBO==1 || KEY_G09==1
     if(LQMEWD) call wrndie(-5,'<PME_QMMM>','QM/MM-PME do not support COLFFT.')
#endif
#else /*      (colfft)*/

     call do_pme_ksp_qm_mm_grad(natom,nquant,iqmatoms,forder,volume,kappa, &
                                recip,virial,x,y,z,d_ewald_mm,cg,cg1, &
                                qm_atm_grad_comp,scf_mchg_2, &
                                sizfftab,sizffwrk,siztheta,siz_q,xnsymm,maxsym,xsymop)
#endif /*        (colfft)*/

     !     for the virial contribution.
     cfact=CCELEC/(xnsymm**2)
     ewvirial(1) = ewvirial(1) - virial(1)*cfact
     ewvirial(2) = ewvirial(2) - virial(2)*cfact
     ewvirial(3) = ewvirial(3) - virial(3)*cfact
     ewvirial(4) = ewvirial(4) - virial(2)*cfact
     ewvirial(5) = ewvirial(5) - virial(4)*cfact
     ewvirial(6) = ewvirial(6) - virial(5)*cfact
     ewvirial(7) = ewvirial(7) - virial(3)*cfact
     ewvirial(8) = ewvirial(8) - virial(5)*cfact
     ewvirial(9) = ewvirial(9) - virial(6)*cfact
     !
  end if

  deallocate(fr1,fr2,fr3,cg1,stat=alloc_err)
  if(alloc_err /= 0 ) write(0,*)"unable to deallocate fr1,2,3"
  deallocate(qm_atm_grad_comp,stat=alloc_err)
  if(alloc_err /= 0 ) write(0,*)"unable to deallocate qm_atm_grad_comp"
  deallocate(qarray,stat=alloc_err)
  if(alloc_err /= 0 ) write(0,*)"unable to deallocate qarray"
  deallocate(qarray_mm,stat=alloc_err)
  if(alloc_err /= 0 ) write(0,*)"unable to deallocate qarray_mm"

  call deallocate_bspline()

  return
  END SUBROUTINE qm_pme_mm_grad


  SUBROUTINE do_pme_ksp_qm_mm_grad(natom,nquant,iqmatoms,forder,volume,ewald_coeff, &
                                   recip,virial,x,y,z,d_ewald_mm,cg,cg1_local,   &
                                   qm_atm_grad_comp_local,scf_mchg_2,  &
                                   sizfftab,sizffwrk,siztheta,siz_q,xnsymm,maxsyml,xsymop)

  use pme_module
  use pmeutil,only:nxyslab,mxyslabs,mxzslabs,nfft1,nfft2, &
                   nfft3,get_sc_fract, &
                   get_fftdims,fft3d0rc,fft3d_zxyrc
  ! OUTPUT
  !       eer:  ewald reciprocal or k-space  energy
  !       dx,dy,dz: forces incremented by k-space sum
  !       virial:  virial due to k-space sum (valid for atomic scaling;
  !                rigid molecule virial needs a correction term not
  !                computed here
  !
  use exfunc
  use number
  use parallel
  use stream
!!#if KEY_BLOCK==1
!!  use block_fcm 
!!#endif
  use memory
  use qm1_info, only : qm_control_r
  use gamess_fcm, only : IGMSEL

  implicit none

  integer        :: natom,nquant
  integer        :: iqmatoms(nquant),forder
  real(chm_real) :: recip(3,3),volume,ewald_coeff
  real(chm_real) :: x(*),y(*),z(*),d_ewald_mm(3,*),scf_mchg_2(nquant),virial(6), &
                    qm_atm_grad_comp_local(3,*)
  real(chm_real) :: cg(*)

  integer :: i,iqm
  integer :: sizfftab,sizffwrk,siztheta,siz_q        ! sizes of some arrays

  ! storage: These arrays can be tossed after leaving this routine
  real(chm_real),intent(in) :: cg1_local(*)

  integer :: xnsymm,maxsyml
  integer :: xsymop(3,4,xnsymm)

  integer :: nfftdim1,nfftdim2,nfftdim3,nfftable,nffwork
  integer :: latm
  integer :: kbot, ktop, i0, igood
  logical :: q_grad_and_pot,q_cpmd_local

  nattot=natom*xnsymm

  !=================================================================
  ! 1st for recip-pme gradient on qm atoms, put gradient into d_ewald_mm array.
  do i=1,nquant
     iqm=iqmatoms(i)
     d_ewald_mm(1,iqm)=d_ewald_mm(1,iqm)+scf_mchg_2(i)*qm_atm_grad_comp_local(1,i)
     d_ewald_mm(2,iqm)=d_ewald_mm(2,iqm)+scf_mchg_2(i)*qm_atm_grad_comp_local(2,i)
     d_ewald_mm(3,iqm)=d_ewald_mm(3,iqm)+scf_mchg_2(i)*qm_atm_grad_comp_local(3,i)
  end do
  !=================================================================

  !=================================================================
  ! Now, get the graient component for each mm atoms by qm atoms.

  !  get some integer array dimensions
  call get_fftdims(nfftdim1,nfftdim2,nfftdim3,nfftable,nffwork)

  !  make array for keeping track of atoms important to this processor
  latm=nattot   ! for consistency with pme.src

  ! Fill charge-grid array relevant for qm atoms only.
  q_grad_and_pot=.false.       ! for gradient calculation
  q_cpmd_local  =.false.
  call fill_ch_grid_qm_mm(kbot, ktop,nquant,scf_mchg_2, &
                          x,y,z,recip,natom,xnsymm, &
                          nfftdim1,nfftdim2,mxyslabs, &
                          lmy_ks_inv,latm,qm_control_r%qminb,q_grad_and_pot,q_cpmd_local)

  if(.not.allocated(tmpy))  allocate(tmpy(2*nfftdim1))
  if(.not.allocated(alpha)) allocate(alpha(nfft1))
  if(.not.allocated(beta))  allocate(beta(nfft1))

  call fft_backrc(Qarray,nfftdim1,nfftdim2,nfftdim3,nffwork)

  ! here: Qarray contains the Fourier transformed structure factor contribution
  !              from QM atoms.
  !       Qarray_mm contains the Fourier transformed structure factor
  !                 contribution from MM atoms.
  ! Now, compute virial component & calculate, (B*C)*F(qarray).
!!!      allocate(qarray_2(siz_q))
!!!      qarray_2=Qarray
  call convol_fr_space_qm_mm(qfinit,rewcut,ewald_coeff,volume,recip, &
                             nfftdim1,nfftdim2,nfftdim3, &
                             Qarray,Qarray_mm,virial,q_grad_and_pot, &
                             q_cpmd_local)

!!!      ! qm-qm:  if you want to use qm-qm virial only from pme calculation,
!!!      !         use the following routines.
!!!      vir_2=zero
!!!      call convol_fr_space_qm_mm(qfinit,rewcut,ewald_coeff,volume,recip, &
!!!                                 nfftdim1,nfftdim2,nfftdim3, &
!!!                                 Qarray_2,Qarray_2,vir_2,q_grad_and_pot, &
!!!                                 q_cpmd_local)
!!!      virial(1:6)=virial(1:6)+half*vir_2(1:6)
!!!      deallocate(qarray_2)

  ! now, forward fourier transform of qarray produces
  ! the convolution of theta(=F(B*C)) and Qarray.
  call fft_forwardrc(Qarray,nfftdim1,nfftdim2,nfftdim3,nffwork)

  call grad_sumrc_qm_mm(igood_qmmm,kbot,ktop,natom,nquant,     &
                        Qarray,cg,scf_mchg_2,d_ewald_mm,recip, &
                        forder,nfftdim1,nfftdim2,nfftdim3,lmy_ks,latm, &
                        xnsymm,igmsel)

  !=================================================================

  deallocate(tmpy)
  deallocate(alpha)
  deallocate(beta)
  call chmdealloc('mndo97_pme_module.src','do_pme_ksp_qm_mm_grad','lmy_ks',latm,intg=lmy_ks)
  call chmdealloc('mndo97_pme_module.src','do_pme_ksp_qm_mm_grad','lmy_ks_inv',nattot,intg=lmy_ks_inv)

  return
  END SUBROUTINE do_pme_ksp_qm_mm_grad


  !***********************************************************************
  subroutine grad_sumrc_qm_mm(igood, kbot, ktop,numatoms,numat,       &
                              qarray_local,cg,scf_mchg_2,d_ewald_mm,  &
                              recip,ordr,nfftdim1,nfftdim2,nfftdim3,  &
                              my_ks,latm,xnsymm,igmsel_local)
  !
  ! This routine compute the gradient at the QM and QM atom sites
  ! applied by all k-space terms.
  !
  ! I have to only assume xnsymm is 1. Other cases does not support yet.
  !
  ! Note: parent routine must call with
  ! numat
  ! IGMSEL (or igmsel_dual, depending on the actual calcualtions.)
  !

  use pme_module
  use pmeutil,only: mxystart,mxyslabs,mxzslabs, &
                    nfft1,nfft2,nfft3, &
                    theta1,theta2,theta3,dtheta1,dtheta2,dtheta3

  use number
  use dimens_fcm
  use parallel
  use stream
  use qm1_constant, only : EV,EVCAL,A0

  implicit none

  integer,intent(in) :: igood, kbot, ktop
  integer,intent(in) :: numatoms,numat,ordr
  integer,intent(in) :: nfftdim1,nfftdim2,nfftdim3,xnsymm
  real(chm_real)     :: qarray_local(*), cg(*), scf_mchg_2(numat), d_ewald_mm(3,*)
  integer,intent(in) :: latm,my_ks(latm),igmsel_local(*)

  real(chm_real),intent(in) :: recip(9)

  integer :: igoo,ig,iqm
  integer :: I,J,K,KQ,i_keep,j_keep,k_keep,n,ITH1,ITH2,ITH3,IPT1,IPT2,IPT3
  integer :: rcskip,nfftdimrc                   ! RCFFT addition
  real(chm_real) :: CFACT,fxyz(3),vala(3),val(3),ufact

  rcskip=1
  if(nfft1/2 /= (nfft1+1)/2) CALL WRNDIE(-5,'<grad_sumrc_qm_mm>','fftx dimension not even ')
  nfftdimrc=nfft1+4
  !
  ufact      = EV*EVCAL*A0 
  if(xnsymm == 1)then
     CFACT=ufact                    ! CCELEC, since unit conversion
  else
     CFACT=ufact/XNSYMM             ! CCELEC/XNSYMM
  end if
  !
  loopig: do ig = 1,igood
     n=my_ks(ig)
     if(xnsymm == 1)then
        igoo=ig
     else
        igoo=n
     endif

     ! only loop over mm atoms.
     ! probably, this is safer.
     if((igmsel_local(n).ne.1) .and. (igmsel_local(n).ne.2)) then
        K_keep = INT(FR3(igoo)) - ORDR + 1 + NFFT3
        J_keep = INT(FR2(igoo)) - ORDR + 1 + NFFT2
        I_keep = INT(FR1(igoo)) - ORDR + 1 + NFFT1

        K        = k_keep
        fxyz(1:3)= zero
        do ITH3 = 1,ORDR
           K=K+1
           IF(K > NFFT3) K=K-NFFT3
           KQ=K
#if KEY_PARALLEL==1
           if ( K  >=  KBOT .AND. K  <=  KTOP ) then
              KQ = K - MXYSTART(MYNOD)
#endif
              IPT1=(KQ-1)*NFFTDIM2 -1
              J = j_keep
              I = i_keep
              IF(I >= NFFT1) I=I-NFFT1

              vala(1)= cg(n)*NFFT1*THETA3(ITH3,ig)
              vala(2)= cg(n)*NFFT2*THETA3(ITH3,ig)
              vala(3)= cg(n)*NFFT3*DTHETA3(ITH3,igoo)

              do ITH2 = 1,ORDR
                 J=J+1
                 IF(J > NFFT2) J=J-NFFT2

                 val(1) = vala(1)*THETA2(ITH2,ig)
                 val(2) = vala(2)*DTHETA2(ITH2,igoo)
                 val(3) = vala(3)*THETA2(ITH2,ig)

                 IPT2= rcskip*((IPT1+J)*NFFTDIMrc+I)+1
                 IPT3= IPT2 + rcskip*(NFFT1-I)
                 do ITH1 = 1,ORDR
                    fxyz(1)=fxyz(1)+val(1)*qarray_local(IPT2)*DTHETA1(ITH1,igoo)
                    fxyz(2)=fxyz(2)+val(2)*qarray_local(IPT2)*THETA1(ITH1,ig)
                    fxyz(3)=fxyz(3)+val(3)*qarray_local(IPT2)*THETA1(ITH1,ig)

                    IPT2=IPT2+rcskip
                    IF(IPT2 >= IPT3) IPT2=IPT2-NFFT1*rcskip
                 end do
              end do
#if KEY_PARALLEL==1
           end if
#endif
        end do
        !
        d_ewald_mm(1,n)=d_ewald_mm(1,n)+CFACT*(recip(1)*fxyz(1)+recip(4)*fxyz(2)+recip(7)*fxyz(3))
        d_ewald_mm(2,n)=d_ewald_mm(2,n)+CFACT*(recip(2)*fxyz(1)+recip(5)*fxyz(2)+recip(8)*fxyz(3))
        d_ewald_mm(3,n)=d_ewald_mm(3,n)+CFACT*(recip(3)*fxyz(1)+recip(6)*fxyz(2)+recip(9)*fxyz(3))
     end if
  end do loopig
  RETURN
  END subroutine grad_sumrc_qm_mm


  !***********************************************************************
  subroutine convol_fr_space_qm_mm(qfinit_local,rewcut_local,ewaldcof,volume,recip, &
                        nfftdim1,nfftdim2,nfftdim3, &
                        Qarray_local,Qarray_mm_local,vir,q_grad_and_pot,q_cpmd)

  use pme_module
  use pmeutil,only:mxzslabs,mxzstart, &
                   nfft1,nfft2,nfft3,bsp_mod1,bsp_mod2,bsp_mod3
  !
  !-------------------------------------------------------------------
  !
  !  QFINIT   - Flag indicating a long range cutoff is to be applied
  !  REWCUT   - The long range cutoff diatance (only if QFINIT)
  !  Qarray   - The main charge grid & after this routine, it is
  !             F(Q)*(C*B), in which the forward 3DFFT produces the convolution of theta*Q.
  !  Qarray_mm- The fourier transformed structure factor from mm atoms.
  !             It will be used to compute virial contribution.
  !  EWALDCOF - The kappa value
  !  VOLUME   - The volume of the perodic system
  !  RECIP    - The inverse of the box length matrix
  !  VIR      - The virial contribution from qm/mm-pme interactions, in which
  !             this will be computed only q_grad_and_pot=.false., meaning
  !             after scf-converged and when computing the gradient.
  !  BSP_MOD1 - The inverse of the B-spline coefficients (a direction)
  !  BSP_MOD2 - The inverse of the B-spline coefficients (b direction)
  !  BSP_MOD3 - The inverse of the B-spline coefficients (c direction)
  !  NFFT1    - The number of grid points (a direction)
  !  NFFT2    - The number of grid points (b direction)
  !  NFFT3    - The number of grid points (c direction)
  !  NFFTDIM1 - The dimension of grid points (a direction)
  !  NFFTDIM2 - The dimension of grid points (b direction)
  !  NFFTDIM3 - The dimension of grid points (c direction)
  !
  !-------------------------------------------------------------------
  !
  !   Finite distance cutoff code added by B. Brooks - NIH - 3/28/98
  !        ref: E.L.Pollock&J. Glosli, Computer Physics Communications,
  !        95 (1996) 93-110.
  !
  use number
  use parallel

  implicit none

  LOGICAL,intent(in) :: QFINIT_local
  real(chm_real),intent(in) ::  REWCUT_local
  INTEGER,intent(in) :: NFFTDIM1,NFFTDIM2,NFFTDIM3
  !
  real(chm_real) :: EWALDCOF,VOLUME
  real(chm_real) :: RECIP(9),Qarray_local(*),Qarray_mm_local(*)
  real(chm_real) :: vir(6)
  logical, intent(in) :: q_grad_and_pot             ! =.true., when computing potential.
                                                    ! =.false., when computing gradient.
  logical, intent(in) :: q_cpmd


  real(chm_real) :: FAC,ETERM,VTERM
  real(chm_real) :: DEN1,DEN2,DEN3,DEN4
  INTEGER        :: K,K0,K1,K2,K3,K2Q,M1,M2,M3
  INTEGER        :: K1s,K2s,K3s,M1s,M2s,M3s
  INTEGER        :: IPT1,IPT2,IPT3,NF1,NF2,NF3
  real(chm_real) :: MVAL,MCUT,MSQ,MSQR,STRUC2,VCORR,ESTR
  real(chm_real) :: MHAT(3),MHATA(3),MHATB(3)
  real(chm_real) :: MHATs(3),DENs,ETERMs,VTERMs,ESTRs,MSQs,STRUC2s,MSQRs
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

#if KEY_PMEPLSMA==1
#if KEY_PARALLEL==1
  if(mynod == 0)then
#endif
     qarray_local(1:2)   =zero
     if(q_cpmd) qarray_mm_local(1:2)=zero
#if KEY_PARALLEL==1
  endif
#endif
#endif

  if(q_grad_and_pot) then            ! when computing potential contribution.
     IPT1=1
     do K2Q = 1, MXZSLABS
        K2=K2Q
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
              MSQ       = MHAT(1)*MHAT(1)+MHAT(2)*MHAT(2)+MHAT(3)*MHAT(3)
              MSQR      = ONE/MSQ

              ETERM = EXP(-FAC*MSQ)*DEN4*MSQR

              IF(QFIN) THEN
                 MVAL=MCUT*SQRT(MSQ)
                 ETERM=ETERM*(ONE-COS(MVAL))
              ENDIF
              !
              Qarray_local(IPT3)   = ETERM * Qarray_local(IPT3)
              Qarray_local(IPT3+1) = ETERM * Qarray_local(IPT3+1)

              if(q_cpmd) then
                 ! this is for the qm-mm only component, whereas Qarray_local has all.
                 Qarray_mm_local(IPT3)   = ETERM * Qarray_mm_local(IPT3)
                 Qarray_mm_local(IPT3+1) = ETERM * Qarray_mm_local(IPT3+1)
              end if
              !
              IPT3=IPT3+2
           end do
           IPT2=IPT2+NFFT3*2
        end do
        IPT1=IPT1+NFFT3*NFFTDIM1*2
     end do
  else                 ! when computing the gradient and virial contribution.
     !
#if KEY_PMEPLSMA==1
#if KEY_PARALLEL==1
     if(mynod == 0)then
#endif
        qarray_mm_local(1:2)=zero
#if KEY_PARALLEL==1
     endif
#endif
#endif
     vir(1:6)=zero
     IPT1=1
     do K2Q = 1, MXZSLABS
        K2=K2Q
#if KEY_PARALLEL==1
        IF(MYNOD > 0) K2 = K2Q + MXZSTART(MYNOD)
#endif

        M2 = K2 - 1
        IF(K2 > NF2) M2 = M2 - NFFT2
        DEN2       = DEN1*BSP_MOD2(K2)
        MHATA(1:3) = RECIP(4:6)*M2

        IPT2=IPT1
        K2s=mod(nfft2-K2+1,nfft2)+1
        do K1 = 1, NF1+1
           K1s=nfft1-K1+2
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
              K3s=mod(nfft3-K3+1,nfft3)+1
              M3 = K3 - 1
              IF(K3 > NF3) M3 = M3 - NFFT3
              DEN4      = DEN3*BSP_MOD3(K3)
              MHAT(1:3) = MHATB(1:3)+RECIP(7:9)*M3
              MSQ       = MHAT(1)*MHAT(1)+MHAT(2)*MHAT(2)+MHAT(3)*MHAT(3)
              MSQR      = ONE/MSQ

              ETERM = EXP(-FAC*MSQ)*DEN4*MSQR
              VTERM = TWO*(FAC+MSQR)
              STRUC2= Qarray_mm_local(ipt3  )*Qarray_local(ipt3  ) +  &
                      Qarray_mm_local(ipt3+1)*Qarray_local(ipt3+1)

              IF(QFIN) THEN
                 MVAL=MCUT*SQRT(MSQ)
                 VCORR=STRUC2*ETERM*SIN(MVAL)*MVAL*MSQR
                 ETERM=ETERM*(ONE-COS(MVAL))
                 VIR(1) = VIR(1) -VCORR*MHAT(1)*MHAT(1)
                 VIR(2) = VIR(2) -VCORR*MHAT(1)*MHAT(2)
                 VIR(3) = VIR(3) -VCORR*MHAT(1)*MHAT(3)
                 VIR(4) = VIR(4) -VCORR*MHAT(2)*MHAT(2)
                 VIR(5) = VIR(5) -VCORR*MHAT(2)*MHAT(3)
                 VIR(6) = VIR(6) -VCORR*MHAT(3)*MHAT(3)
              ENDIF

              ESTR = ETERM * STRUC2
              VIR(1) = VIR(1) + ESTR*(VTERM*MHAT(1)*MHAT(1) - ONE)
              VIR(2) = VIR(2) + ESTR*(VTERM*MHAT(1)*MHAT(2)      )
              VIR(3) = VIR(3) + ESTR*(VTERM*MHAT(1)*MHAT(3)      )
              VIR(4) = VIR(4) + ESTR*(VTERM*MHAT(2)*MHAT(2) - ONE)
              VIR(5) = VIR(5) + ESTR*(VTERM*MHAT(2)*MHAT(3)      )
              VIR(6) = VIR(6) + ESTR*(VTERM*MHAT(3)*MHAT(3) - ONE)

              ! for k1 > 1
              if(k1 > 1)then
                 DENs = DEN1*BSP_MOD3(K3s)*BSP_MOD2(K2s)*BSP_MOD1(K1s)
                 M1s = K1s - 1
                 IF(K1s > NF1) M1s = M1s - NFFT1
                 M2s = K2s - 1
                 IF(K2s > NF2) M2s = M2s - NFFT2
                 M3s = K3s - 1
                 IF(K3s > NF3) M3s = M3s - NFFT3

                 MHATs(1) =RECIP(1)*M1s+RECIP(4)*M2s+RECIP(7)*M3s
                 MHATs(2) =RECIP(2)*M1s+RECIP(5)*M2s+RECIP(8)*M3s
                 MHATs(3) =RECIP(3)*M1s+RECIP(6)*M2s+RECIP(9)*M3s
                 MSQs = MHATs(1)*MHATs(1)+MHATs(2)*MHATs(2)+MHATs(3)*MHATs(3)
                 MSQRs=ONE/MSQs
                 !
                 ETERMs = EXP(-FAC*MSQs)*DENs*MSQRs
                 VTERMs = TWO*(FAC+MSQRs)

                 STRUC2s= Qarray_mm_local(ipt3  )*Qarray_local(ipt3  ) +  &
                          Qarray_mm_local(ipt3+1)*Qarray_local(ipt3+1)

                 ESTRs = ETERMs * STRUC2s
                 VIR(1) = VIR(1) + ESTRs*(VTERMs*MHATs(1)*MHATs(1) - ONE)
                 VIR(2) = VIR(2) + ESTRs*(VTERMs*MHATs(1)*MHATs(2)      )
                 VIR(3) = VIR(3) + ESTRs*(VTERMs*MHATs(1)*MHATs(3)      )
                 VIR(4) = VIR(4) + ESTRs*(VTERMs*MHATs(2)*MHATs(2) - ONE)
                 VIR(5) = VIR(5) + ESTRs*(VTERMs*MHATs(2)*MHATs(3)      )
                 VIR(6) = VIR(6) + ESTRs*(VTERMs*MHATs(3)*MHATs(3) - ONE)
              end if

              !
              Qarray_local(IPT3)   = ETERM * Qarray_local(IPT3)
              Qarray_local(IPT3+1) = ETERM * Qarray_local(IPT3+1)
              !
              IPT3=IPT3+2
           end do
           IPT2=IPT2+NFFT3*2
        end do
        IPT1=IPT1+NFFT3*NFFTDIM1*2
     end do

!!!!      no need to do this, since it is between qm and mm pairs.
!!!!      vir(1:6)=half*vir(1:6)

  end if

  RETURN
  END subroutine convol_fr_space_qm_mm
!=================================================================================================!
!========                           PME Section Ends Here                                 ========!
!=================================================================================================!


#endif /*mndo97*/

end module qmmmpme_module
