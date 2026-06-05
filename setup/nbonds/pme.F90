module pme_module

  use chm_kinds
  implicit none

  real(chm_real),allocatable,dimension(:),save :: &
       tmpy,alpha,beta

  !-----------------------------------------------------------------------
  !     for the particle mesh Ewald summation.
  !
  !     This fcm containd the following:
  !
  !     QPME     - flag to use particle mesh ewald for kspace sum
  !     QFINIT   - Flag indicating a finite (non-periodic PME)
  !     REWCUT   - The cutoff distance for interactions when QFINIT
  !     REQCOR   - Scale factor for net charge correction (def 1.0)
  !     FORDER   - the order of Bspline interpolation. E.g. cubic
  !                is order 4 (default) fifth degree is order 6 etc.
  !                The order must be an even number and at least 4
  !
  !     Code has been modified to add BLOCK for decomposition
  !     of the energy
  !     Author:
  !           Hiqmet Kamberaj
  !           November 2007
  !

  real(chm_real),allocatable,dimension(:),save :: qarray
  real(chm_real),allocatable,dimension(:),save :: FR1,FR2,FR3,CG1
  REAL(chm_real),save ::  REWCUT,REQCOR

  integer,save :: nattot,ierr_allocate

  LOGICAL,save :: QPME,QFINIT
  !
  !av_080628 H kamberaj (November 2007)
#if KEY_BLOCK==1
  real(kind=chm_real), allocatable, save :: qtemp(:)
  integer, save                          :: ntot
#endif /*   av_080628 close BLOCK*/

  ! for indexing: it's used in do_pme_ksp
  integer,allocatable,dimension(:),save :: lmy_ks
#if KEY_SQUANTM==1 || KEY_MNDO97==1 || KEY_QCHEM==1 || KEY_QTURBO==1 || KEY_G09==1
  integer,allocatable,dimension(:),save :: lmy_ks_inv   
#endif

  ! for qm/mm-pme version
#if KEY_SQUANTM==1 || KEY_MNDO97==1 || KEY_QCHEM==1 || KEY_QTURBO==1 || KEY_G09==1
  ! for qm-qm interactions.
  real(chm_real),allocatable,dimension(:,:),save :: qm_atm_grad_comp
  integer, save :: igood_qmmm

  ! for virial.
  real(chm_real),allocatable,dimension(:),save :: qarray_mm

  ! for gradient.
  real(chm_real),allocatable,dimension(:,:,:,:),save :: r_qarray_mm
  integer,       allocatable,dimension(:,:,:,:),save :: i_qarray_mm
#endif 

  integer, allocatable, dimension(:) :: grid_atom_old

contains

  !****************************************************************
  !                        PMESH_CLEAR
  !****************************************************************
  SUBROUTINE PMESH_CLEAR
    use pmeutil
#if KEY_COLFFT==1
    use colfft,only:colfft_uninit        
#endif
    use memory,only:chmdealloc
#if KEY_LJPME==1
    use helpme_wrapper, only: helpme_destroy

    if(qljpme) call helpme_destroy(dpmeInstance)
    if(qcpme) call helpme_destroy(qpmeInstance)
#endif
    !
    !  This routine releases PME allocated space.
    !
    if (allocated(grid_atom_old)) then
       call chmdealloc('pme.src','pmesh_clear','grid_atom_old',&
            size(grid_atom_old),intg=grid_atom_old)
    endif

    if(allocated(bsp_mod1)) &
         deallocate(bsp_mod1,bsp_mod2,bsp_mod3)
    if(allocated(fft1_table)) &
         deallocate(fft1_table,fft2_table,fft3_table,ffwork)
    !
#if KEY_COLFFT==1
    call colfft_uninit()  
#endif
    RETURN
  END subroutine pmesh_clear

  ! *
  ! * Calculates PME self-energy
  ! *
  subroutine pme_self_energy(leself, qeself, kappa, cg, natomx, qdcm)
    use number
    use consta,only:pi,ccelec
#if KEY_BLOCK==1
    use block_ltm,only:qhybh,ihybh,hybhlb,qblock,iblckp,blcoee ! Y Huang 2017
    use lambdam,only:iqldm_pme,biflam,bixlam
#endif
#if KEY_BLOCK==1
    use energym,only:ewself,ETERMR,ETERMP ! symhyb -- Y Huang 2017
#endif
#if KEY_CHEQ==1
    use cheq,only:qcg,dch            
#endif
#if KEY_DOMDEC==1
    use domdec_common,only:q_domdec,natoml,atoml  
#endif
#if KEY_PARALLEL==1
    use parallel,only:iparpt,mynod,mynodp,numnod
#endif
    use psf,only:natom

    implicit none
    ! Input / Output
    real(chm_real), intent(out) :: leself
    logical, intent(in) :: qeself
    real(chm_real), intent(in) :: kappa, cg(*)
    ! Variables
    real(chm_real) eslf, ent
#if KEY_BLOCK==1
    real(chm_real) chf2(3),dslf(3),coef
    integer i,ibl,kk
#endif 
#if KEY_CHEQ==1
    real(chm_real) hij              
#endif
    integer ii, ia
    integer atfrst, atlast, nat
    logical qdcm ! FOR DCM MODULE
    integer atgrp,natomx ! FOR DCM MODULE

#if KEY_DOMDEC==1 /*domdec*/
    if (q_domdec) then
       atfrst = 1
       atlast = natoml
    else
#endif /* (domdec)*/
       if(qdcm) atlast=natomx !DCM MODULE
#if KEY_PARALLEL==1 /*paramain*/
#if KEY_PARAFULL==1 /*parfmain*/
       atfrst = 1 + iparpt(mynod)
       atlast = iparpt(mynodp)

       if(qdcm) then
         if(numnod.gt.natomx) then !catch more cores than charges
           if(mynod.eq.0) then
             atfrst=1
             atlast=natomx
           else
             atfrst=1
             atlast=0
           endif
         else !standard case
           atgrp=int(natomx/numnod)
           atfrst=atgrp*mynod+1
           atlast=atgrp*(mynod+1)
           if(mynod.eq.(numnod-1)) atlast=natomx
         endif
       endif

#elif KEY_PARASCAL==1 || KEY_SPACDEC==1 /*parfmain*/
       atfrst = 1
       atlast = natom
#endif /* (parfmain)*/
#else /* (paramain)*/
       atfrst = 1
       atlast = natom
#endif /* (paramain)*/
#if KEY_DOMDEC==1
    endif  
#endif

    nat = atlast - atfrst + 1

    self_e: if(qeself) then
       eslf = zero
       !.ab...HYBH.We can use the fact that energy.f90 is here.
#if KEY_BLOCK==1
       if (qhybh) then
          ihybh=ewself
          chf2(1)=one
          chf2(2)=(one-hybhlb)*(one-hybhlb)
          chf2(3)=hybhlb*hybhlb
          dslf(1)=zero
          dslf(2)=zero
          dslf(3)=zero
       endif
#endif 
          
#if KEY_CHEQ==1
       hij=two*kappa*ccelec/sqrt(pi)    
#endif
          
#if KEY_BLOCK==1
!$omp parallel do private(ia, ii, ent, ibl, kk, coef) reduction(+:eslf)
#else
!$omp parallel do private(ia, ii, ent) reduction(+:eslf)
#endif
       selfloop: do ia = atfrst, atlast
#if KEY_DOMDEC==1
          if (q_domdec) then
             ii = atoml(ia)
          else
#endif 
             ii = ia
#if KEY_DOMDEC==1
          endif  
#endif
          ent = cg(ii) * cg(ii)
#if KEY_BLOCK==1
          if (qblock) then
             ibl = iblckp(ii)
             if (iqldm_pme > 0) then
               coef = bixlam(ibl)
               if (ibl > 1) then
!$omp atomic
                 biflam(ibl) = biflam(ibl) - 2*ent*coef*kappa*ccelec/sqrt(pi)
!$omp end atomic
               endif
               ent = ent*coef*coef
             else
               kk  = ibl + ibl*(ibl-1)/2
               coef = blcoee(kk)
               ent = ent*coef
             endif
          endif
          if (qhybh) then
             i = iblckp(ii)
             if (i /= 1) then
                eslf = eslf + ent*chf2(i)
                dslf(i)=dslf(i)+ent
             else
                eslf = eslf + ent
             endif
          else
#endif 
             eslf = eslf + ent
             
#if KEY_BLOCK==1
          endif                        
#endif
#if KEY_CHEQ==1
          if (qcg) dch(ii)=dch(ii)-hij*cg(ii)     
#endif
       enddo selfloop
!$omp end parallel do
       
#if KEY_BLOCK==1
       if (qhybh) then
          dslf(2)=two*kappa*ccelec/sqrt(pi)*(one-hybhlb)*dslf(2)
          dslf(3)=-two*kappa*ccelec/sqrt(pi)*hybhlb*dslf(3)
!------------------------------------------------------------------
!      No block_fcm, No symhyb -- Y Huang 2017
!          call sumhyb(ihybh,dslf(2),dslf(3))
          ETERMR(ihybh)=ETERMR(ihybh)+dslf(2)
          ETERMP(ihybh)=ETERMP(ihybh)+dslf(3)
!------------------------------------------------------------------
       endif
#endif 
       leself = - eslf*kappa*ccelec/sqrt(pi)
    else  ! self_e
       leself = zero
    endif self_e

    return
  end subroutine pme_self_energy


  ! *
  ! * Calculates LJ-PME self-energy
  ! *
  subroutine ljpme_self_energy(selfen, qeself, kappa, c6s)
    use number
#if KEY_DOMDEC==1
    use domdec_common,only:q_domdec,natoml,atoml  
#endif
#if KEY_PARALLEL==1
    use parallel,only:iparpt,mynod,mynodp   
#endif
    use psf,only:natom
    implicit none
    ! Input / Output
    real(chm_real), intent(out) :: selfen
    logical, intent(in) :: qeself
    real(chm_real), intent(in) :: kappa, c6s(*)
    ! Variables
    real(chm_real) eslf, ent, prefac
    integer ii, ia
    integer atfrst, atlast, nat

#if KEY_DOMDEC==1 /*domdec*/
    if (q_domdec) then
       atfrst = 1
       atlast = natoml
    else
#endif /* (domdec)*/
#if KEY_PARALLEL==1 /*paramain*/
#if KEY_PARAFULL==1 /*parfmain*/
       atfrst = 1 + iparpt(mynod)
       atlast = iparpt(mynodp)
#elif KEY_PARASCAL==1 || KEY_SPACDEC==1 /*parfmain*/
       atfrst = 1
       atlast = natom
#endif /* (parfmain)*/
#else /* (paramain)*/
       atfrst = 1
       atlast = natom
#endif /* (paramain)*/
#if KEY_DOMDEC==1
    endif  
#endif

    nat = atlast - atfrst + 1
    prefac = kappa**6 / 12d0
    self_e: if(qeself) then
       eslf = zero
!$omp parallel do private(ia, ii, ent) reduction(+:eslf)
       selfloop: do ia = atfrst, atlast
#if KEY_DOMDEC==1
          if (q_domdec) then
             ii = atoml(ia)
          else
#endif 
             ii = ia
#if KEY_DOMDEC==1
          endif  
#endif
             eslf = eslf + c6s(ii) * c6s(ii)
       enddo selfloop
!$omp end parallel do
       selfen = eslf*prefac
    else  ! self_e
       selfen = zero
    endif self_e

    return
  end subroutine ljpme_self_energy

  !****************************************************************
  !                  PME
  !****************************************************************
  !
  SUBROUTINE PME(EKSUM,LESELF,EQCOR,EUTIL, &
       QEKSUM,QESELF,QEQCOR,QEUTIL, &
       X,Y,Z,DX,DY,DZ,NATOM,CG,CGTOT, &
#if KEY_LJPME==1
       ljc6, &
#endif
       ewvirial, kappa &
#if KEY_PHMD==1
     , QPHMD  &  ! PME-CPHMD -- Y Huang 2017
#endif /* phmd */
#if KEY_MNDO97==1
     , q_qmmm_pme_do   &
#endif
     , QDCM & ! DCM Module
       )
    !
    !
    ! QM/MM-Ewald
#if KEY_QUANTUM==1 
#if KEY_GAMESS==0 && KEY_GAMESSUK==0 && KEY_QTURBO==0 && \
    KEY_G09==0 && KEY_QCHEM==0 && KEY_MNDO97==0 && KEY_SCCDFTB==0 
    use quantm,only: LQMEWD, QSETUPKQ, QCGSC   
#endif
#endif
#if KEY_GAMESS==1 || KEY_GAMESSUK==1  || KEY_QTURBO==1 || \
    KEY_G09==1 || KEY_QCHEM==1 || KEY_MNDO97==1 || KEY_SCCDFTB==1 
#if KEY_SQUANTM==0
    use mndo97,only: lqmewd                    
#endif
#endif 
#if KEY_SQUANTM==1
    use squantm,only: lqmewd,ewmode            
#endif
    use memory    
    use erfcd_mod,only: erfcd
    use ewald_1m,only: erfmod
    use pmeutil
    !-----------------------------------------------------------------------
    !     This routine calculates non bonded interaction energies and
    !     forces via the Particle Mesh Ewald Summation
    !     original code by Tom Darden, implemented into CHARMM
    !     by Scott Feller and Bernie Brooks, NIH, Jan-1996
    !     Parallel 3D fft routines from Michael Crowley, Pittsburgh
    !     Supercomputer Center
    !-----------------------------------------------------------------------

#if KEY_COLFFT==1
    use colfft,only:colfft_init, do_colfft, filter_grid_atom
    use colfft_util,only:colfft_util_init
#endif
#if KEY_CHEQ==1
    use cheq, only: qcg, dch    
#endif
    use dimens_fcm
    use number
    use energym
    use inbnd
    use image
    use pbound
    use consta
    use iso_c_binding, only: c_size_t
    use stream
#if KEY_BLOCK==1
    use block_ltm  ! block_fcm -> block_ltm -- PME-CPHMD Y Huang 2017
    use lambdam, only: iqldm_pme
#endif
    use parallel,only:mynod, numnod
    use gbsw,only:QGBSW  ! PHMD
    use gbmv,only:QGBMV  ! PHMD
#if KEY_DOMDEC==1
    use domdec_r2d_comm,only:n_grid_atom, grid_atom_p
    use domdec_common,only:q_domdec, ndirect
#endif
#if KEY_DOMDEC_GPU==1
    use domdec_common,only:q_gpu, gpu_code_version, q_print_output, q_test, boxx, boxy, boxz
    use nbrecip_gpu_mod, only: calc_recip_gpu, calc_recip_block_gpu
    use reawri,only:qcnstp
#endif
#if KEY_LJPME==1
   use helpme_wrapper
   use pmeutil, only: dkappa, dorder, dfftx, dffty, dfftz, dpmeInstance, qpmeInstance,&
                      qmax, qmaxx, qmaxy, qmaxz, dmax, dmaxx, dmaxy, dmaxz,&
                      nfft1, nfft2, nfft3
#endif
    use prssre
#if KEY_MNDO97==1
    ! Also, compute the potential at the QM atom position due to the ewald sum of
    ! the MM atoms. Called once per energy evaluation, before doing the SCF cycle.
    !
    ! Basically, calculate K space PME potential at QM atom position.
    !
    use qm1_info,only: qm_control_r,qm_main_r,mm_main_r
    use qmmmewald_module, only : qmmm_ewald_r
#endif

    !
    real(chm_real) ::  EKSUM,LESELF,EQCOR,EUTIL
    real(chm_real) :: ewvirial(9),kappa
    LOGICAL QEKSUM,QESELF,QEQCOR,QEUTIL
    real(chm_real) ::  X(*),Y(*),Z(*),DX(*),DY(*),DZ(*)
    INTEGER NATOM
    real(chm_real) ::  CG(*),CGTOT
#if KEY_LJPME==1
    real(chm_real), intent(in) :: ljc6(*)
    real(chm_real) :: ljvirial(6), junk1, junk2
#endif
    INTEGER IONE
    real(chm_real) :: imxcent(1),imycent(1),imzcent(1)

#if KEY_CHEQ==1
    real(chm_real) ::    HIJ        
#endif
    real(chm_real) :: xtlinv(6),recip(3,3),virial(6), &
         vircor(6),eslf,cfact                        !av_080628
    real(chm_real) :: rdum(3),htmp(9),hwrk(9)
    integer siztheta, sizdtheta, siz_q
    integer nfftdim1, nfftdim2, nfftdim3

    integer i,j, ii, ipt, itrans, nn
    real(chm_real) :: tx,ty,tz,rs,s2,erfcx,drfc,ene,qd(1),dxf,dyf,dzf

#if KEY_PMEPLSMA==1
    real(chm_real) factor,ee_plasma      
#endif

#if KEY_BLOCK==1
    real(chm_real) CHF2(3),CURCG,CHFACT    
#endif
#if KEY_PHMD==1
    logical QPHMD  ! PME-CPHMD --  Y Huang 2017
#endif /* phmd */
    logical ok
    integer alloc_err
    integer xnsymm_t
    integer ia
    integer n_grid_atom_old
#if KEY_DOMDEC_GPU==1
    logical q_calc_energy, q_calc_virial
#endif
#if KEY_MNDO97==1
    integer :: nquant
    logical :: q_qmmm_pme_do
#endif
    logical :: qdcm ! DCM Module

    if (qboun) then
#if KEY_PBOUND==1
       ! APH: Assumes orthorhombic box
       call pbound_getvol(eprop(volume))        !APH: Note getvol requires use of images
       xtlinv(1) = boxinv
       xtlinv(2) = zero
       xtlinv(3) = boyinv
       xtlinv(4) = zero
       xtlinv(5) = zero
       xtlinv(6) = bozinv
       ok = .true.
#endif 
    else
       call getvol(eprop(volume))        !APH: Note getvol requires use of images
       call invt33s(xtlinv,xtlabc,ok)
    endif
    recip(1,1) = xtlinv(1)
    recip(2,2) = xtlinv(3)
    recip(3,3) = xtlinv(6)
    recip(1,2) = xtlinv(2)
    recip(2,1) = xtlinv(2)
    recip(1,3) = xtlinv(4)
    recip(3,1) = xtlinv(4)
    recip(2,3) = xtlinv(5)
    recip(3,2) = xtlinv(5)

#if KEY_LJPME==1
    ljvirial = 0d0
#endif
#if KEY_DOMDEC_GPU==1
#if KEY_LJPME==1
    if(qljpme .and. q_gpu) call wrndie(-5,'<PME>','LJPME not implemented for DOMDEC_GPU')
#endif /* LJPME */
    ! DOMDEC_GPU code version 2 calculates PME self energy on GPU
    if (.not.q_domdec .or. .not.q_gpu .or. gpu_code_version==1) &
#endif
       call pme_self_energy(leself, qeself, kappa, cg, natom, qdcm)
#if KEY_LJPME==1
       if(qljpme) call ljpme_self_energy(ETERM(LJSEL), QETERM(LJSEL), dkappa, ljc6)
#endif

#if KEY_MNDO97==1 /*mndo_and_others*/
    ! this should check later, if initialization has been done correctly.
    if(q_qmmm_pme_do .or. .not. lqmewd) then
       ewvirial(1:9)=zero
    end if
#else
#if KEY_QUANTUM==1 || KEY_SQUANTM==1 || KEY_GAMESS==1 || \
    KEY_GAMESSUK==1 || KEY_QCHEM==1 || KEY_QTURBO==1 || KEY_G09==1
    if(.not.lqmewd) then      
#endif
       ewvirial(1:9)=zero
#if KEY_QUANTUM==1 || KEY_SQUANTM==1 || KEY_GAMESS==1 || \
    KEY_GAMESSUK==1 || KEY_QCHEM==1 || KEY_QTURBO==1 || KEY_G09==1
    endif                     
#endif
#endif  /*mndo_and_others*/

#if KEY_MNDO97==1 
    if(q_qmmm_pme_do) then
       nquant = qm_main_r%numat
       qmmm_ewald_r%empot_pme(1:nquant)   =zero
       qmmm_ewald_r%empot_qm_pme(1:nquant)=zero
    endif
!!    ! put QM charges if cpmd run.
!!    if(lqmewd .and. qm_control_r%q_do_cpmd_pme) then
!!       ! put the mulliken charge on cg array
!!       do i=1,nquant
!!          nn     =iabs(qm_control_r%qminb(i))
!!          qm_control_r%cgqmmm(i) = cg(nn)
!!          cg(nn) = mm_main_r%qm_charges(i)
!!       end do
!!    end if
#endif

    eqcor=zero
    eutil=zero

    !-------------------------------------------------------------------
    ! INPUT
    !      NFFT1,NFFT2,NFFT3 are the (integer) dimensions of the charge grid array
    !      NATOM is number of atoms
    !      FORDER is the order of B-spline interpolation
    !      x,y,z:   atomic coords
    !      CG  atomic charges
    !      XTLINV=recip: array of reciprocal unit cell vectors
    !      VOLUME: the volume of the unit cell
    !      KAPPA=ewald_coeff:   ewald convergence parameter
    !      NFFT1,NFFT2,NFFT3: the dimensions of the charge grid array
    ! OUTPUT
    !      siz_Q=3d charge grid array
    !      sizfftab is permanent 3d fft table storage
    !      sizffwrk is temporary 3d fft work storage
    !      siztheta is size of arrays theta1-3 dtheta1-3
    !      EKSUM= eer:  ewald reciprocal or k-space  energy
    !      dx,dy,dz: forces incremented by k-space sum
    !      EWVIRIAL=virial:  virial due to k-space sum (valid for atomic scaling;
    !                rigid molecule virial needs a correction term not
    !                computed here
    !
    !   All pointers are the integer names of the real(chm_real) variables that
    !   will be filled
    !
    !   Get memory for scratch arrays, free them after summation.
    !
    !   Modified to add BLOCK code
    !   Author:
    !         Hiqmet Kamberaj
    !         November 2007
    !
    call get_fftdims(nfftdim1,nfftdim2,nfftdim3,i,j)
    if (qboun) then
       xnsymm_t = 1
    else
       xnsymm_t = xnsymm
    endif

    siztheta  = natom*xnsymm_t*forder
    sizdtheta = (natom+1)*forder
    nattot=natom*xnsymm_t
#if KEY_PARALLEL==1
    siz_q = max(2*nfftdim1*nfftdim2*mxyslabs, &
         2*nfftdim1*nfftdim3*mxzslabs)
#if KEY_BLOCK==1
    chfact=one/numnod                        
#endif
#else /**/
    siz_q = 2*nfftdim1*nfftdim2*nfftdim3
#if KEY_BLOCK==1
    chfact=one                               
    siz_q = max(2*nfftdim1*nfftdim2*mxyslabs, &
         2*nfftdim1*nfftdim3*mxzslabs)
#endif
#endif 
#if KEY_BLOCK==1
    if (qblock) then
       ntot=siz_q
       allocate(qtemp(ntot), stat = alloc_err)
       if (alloc_err /= 0)  &
            call wrndie(-1,'<PME>','unable to allocate QTEMP')
       siz_q = nblock*ntot
    endif
#endif /*  close BLOCK*/
    !
#if KEY_COLFFT==0
    allocate(qarray(siz_q),stat=alloc_err)
    if(alloc_err /= 0 ) write(0,*)"unable to allocate qarray"

#if KEY_MNDO97==1
    if(q_qmmm_pme_do) then
       allocate(qarray_mm(siz_q),stat=alloc_err)
       if(alloc_err /= 0 ) write(0,*)"unable to allocate qarray_mm"
       allocate(qm_atm_grad_comp(3,nquant))
    end if
#endif

    allocate(fr1(nattot),fr2(nattot),fr3(nattot),cg1(nattot), &
         stat=alloc_err)
    if(alloc_err /= 0 ) write(0,*)"unable to allocate fr arrays"
    call allocate_bspline(natom,nattot)
#endif 

    !
    do_pme: if(qeksum &
#if KEY_LJPME==1
             .or. qljpme &
#endif
              ) then
#if KEY_COLFFT==1 /*colfft*/
       !
       !==================COLUMN FFT METHOD ==============================
       !
#if KEY_BLOCK==1
       IF (QBLOCK .and. (iqldm_pme==0)) CALL WRNDIE(-5,'<PME>','BLOCK not implemented for COLFFT')
       IF (QHYBH) CALL WRNDIE(-5,'<PME>','HYBH not implemented for COLFFT')
#endif 
#if KEY_CHEQ==1
       if (qcg) call wrndie(-5,'<pme>','CHEQ not implemented for COLFFT')  
#endif

#if KEY_DOMDEC==1
       if (q_domdec) then
#if KEY_DOMDEC_GPU==1
          if (q_gpu .and. gpu_code_version==2) then
             ! Calculate colfft on GPU
!             IF (QBLOCK) CALL WRNDIE(-5,'<PME>','BLOCK not implemented on domdec GPU yet. Remove when fixed. -RLH 2017-04-13')
             q_calc_energy = (.not.qdyncall) .or. q_print_output .or. q_test
             q_calc_virial = (.not.qdyncall) .or. q_print_output .or. q_test .or. qcnstp
#if KEY_BLOCK==1
             if (iqldm_pme > 0) then
                call calc_recip_block_gpu(boxx, boxy, boxz, q_calc_energy, q_calc_virial)
             else
                call calc_recip_gpu(boxx, boxy, boxz, q_calc_energy, q_calc_virial)
             end if
#else
             call calc_recip_gpu(boxx, boxy, boxz, q_calc_energy, q_calc_virial)
#endif /* KEY_BLOCK */             
          else
#endif
#if KEY_LJPME==1
             if(qljpme) then
                if((ndirect .gt. 1) .and. ((nfft1 .ne. dfftx) .or. (nfft2 .ne. dffty) .or. (nfft3 .ne. dfftz) .or. (forder .ne. dorder))) then
                    call wrndie(-5,'<pme>',&
                          'Electrostatic and LJPME grids and splines much have the same dimensions for DOMDEC with ndir>1')
                endif
                ! There's a nondeterministic deadlock happning after handing off
                ! the communicator to helPME.  Just use colfft for now.
                !call helpme_setup(dpmeInstance, 6, dkappa, dorder, dfftx, dffty, dfftz, -1d0, 0)
                !call helpme_compute_EFV_rec(dpmeInstance, ETERM(LJREC), n_grid_atom, grid_atom_p, ljc6,&
                !                            X, Y, Z, DX, DY, DZ, ljvirial)
                ! Initialize colfft_util
                call colfft_init(natom, n_grid_atom)
                ! Calculate colfft
                call do_colfft(n_grid_atom, grid_atom_p, &
                     natom,nfftdim1,nfftdim2,nfftdim3, &
                     ETERM(LJREC),junk1,junk2, &
                     QETERM(LJREC),.false.,.false., .true., &
                     x,y,z,dx,dy,dz,ljc6,0d0,ljvirial,dkappa)
             endif
             if (qcpme) then
                call helpme_setup_compressed(qpmeInstance, 1, kappa, forder, nfft1, nfft2, nfft3,&
                                             qmaxx, qmaxy, qmaxz, CCELEC, 0)
                call helpme_compute_EFV_rec(qpmeInstance, eksum, n_grid_atom, grid_atom_p, cg,&
                                            X, Y, Z, DX, DY, DZ, ljvirial)
                eksum = eksum / CCELEC
                ljvirial = ljvirial / CCELEC
             else
#endif /*LJPME*/
                ! Initialize colfft_util
                call colfft_init(natom, n_grid_atom)
                ! Calculate colfft
                call do_colfft(n_grid_atom, grid_atom_p, &
                     natom,nfftdim1,nfftdim2,nfftdim3, &
                     eksum,eqcor,eutil, &
                     qeksum,qeqcor,qeutil, &
#if KEY_LJPME==1
                     .false., &
#endif
                     x,y,z,dx,dy,dz,cg,cgtot,virial,kappa)
#if KEY_LJPME==1
             endif
#endif /*LJPME*/

#if KEY_DOMDEC_GPU==1
          endif
#endif
       else
#endif
          if (allocated(grid_atom_old)) then
             if (size(grid_atom_old) < natom) then
                call chmdealloc('pme.src','pme','grid_atom_old',&
                     size(grid_atom_old),intg=grid_atom_old)
             endif
          endif
          if (.not.allocated(grid_atom_old)) then
             call chmalloc('pme.src','pme','grid_atom_old',natom,intg=grid_atom_old)
          endif
          ! Initialize colfft_util
          call colfft_util_init(numnod)
          ! Filter atom
          call filter_grid_atom(natom, nfft2, nfft3, forder, x, y, z, &
               n_grid_atom_old, grid_atom_old)
#if KEY_LJPME==1
          if (qcpme) then
             call helpme_setup_compressed(qpmeInstance, 1, kappa, forder, nfft1, nfft2, nfft3,&
                                          qmaxx, qmaxy, qmaxz, CCELEC, 0)
             call helpme_compute_EFV_rec(qpmeInstance, eksum, n_grid_atom_old, grid_atom_old, cg,&
                                         X, Y, Z, DX, DY, DZ, ljvirial)
             eksum = eksum / CCELEC
             ljvirial = ljvirial / CCELEC
          else
#endif /*LJPME*/
             ! Initialize colfft
             call colfft_init(natom, n_grid_atom_old)
             ! Calculate colfft
             call do_colfft(n_grid_atom_old, grid_atom_old, natom,nfftdim1,nfftdim2,nfftdim3, &
                  eksum,eqcor,eutil, &
                  qeksum,qeqcor,qeutil, &
#if KEY_LJPME==1
                     .false., &
#endif
                  x,y,z,dx,dy,dz,cg,cgtot,virial,kappa)
#if KEY_LJPME==1
          endif
          if(qljpme) then
             !call helpme_setup(dpmeInstance, 6, dkappa, dorder, dfftx, dffty, dfftz, -1d0, 0)
             !call helpme_compute_EFV_rec(dpmeInstance, ETERM(LJREC), n_grid_atom_old, grid_atom_old, ljc6,&
             !                            X, Y, Z, DX, DY, DZ, ljvirial)
             ! Initialize colfft_util
             call colfft_init(natom, n_grid_atom_old)
             ! Calculate colfft
             call do_colfft(n_grid_atom_old, grid_atom_old, &
                  natom,nfftdim1,nfftdim2,nfftdim3, &
                  ETERM(LJREC),junk1,junk2, &
                  QETERM(LJREC),.false.,.false., .true., &
                  x,y,z,dx,dy,dz,ljc6,0d0,ljvirial,dkappa)
          endif
#endif /*LJPME*/
#if KEY_DOMDEC==1
       endif
#endif

#else /*      (colfft)*/
#if KEY_BLOCK==1
       if(qhybh) ihybh=ewksum                            
#endif
       if(xnsymm == 0) call wrndie(-5,'<PME>', &
            'XNSYMM is zero: build crystal.')
       call do_pme_ksp(qfinit,rewcut,natom, &
            x,y,z,cg,recip,eprop(volume), &
            kappa,forder,eksum,dx,dy,dz,virial, &
#if KEY_CHEQ==1
            dch,qcg,                                              & 
#endif
            sizfftab,sizffwrk,siztheta,siz_q,cg1, &
            xnsymm,maxsym,xsymop   &
#if KEY_MNDO97==1
           ,q_qmmm_pme_do          &
#endif
           )
#endif /*        (colfft)*/
    endif do_pme
    !
    !------------- Charged systems ----------------------------------------
    if((abs(cgtot) > pt0001 .and. reqcor.gt.zero) &
#if KEY_BLOCK==1
         .or. qhybh                                  & 
#endif
         ) then
#if KEY_PMEPLSMA==1 /*pmeplsma*/
#if KEY_COLFFT==1
       call wrndie(-4,"pme.src:COLFFT","not set up for charged systems")    
#endif
#if KEY_BLOCK==1
       IF(.NOT.QHYBH) THEN                         
#endif
#if KEY_PARALLEL==1
          if(mynod == 0)then 
#endif
             factor = pi / (kappa*kappa*volume)
             ee_plasma = 0.5d0*factor*cgtot*cgtot*CCELEC
             EWVIRIAL(1) = EWVIRIAL(1) - ee_plasma
             EWVIRIAL(5) = EWVIRIAL(5) - ee_plasma
             EWVIRIAL(9) = EWVIRIAL(9) - ee_plasma
#if KEY_PARALLEL==1
          endif 
#endif
#if KEY_BLOCK==1
       ENDIF                                       
#endif

#else /* (pmeplsma)*/
#if KEY_COLFFT==1
       call wrndie(-4,"pme.src:COLFFT","not set up for charged systems")   
#endif
!AvdV bugfix -- moved "#endif /*   (pmeplsma)*/" below
!.ab.Revise if then etc. structure jul2021.
!---#if KEY_PMEPLSMA==1 && KEY_BLOCK==1
!---if(qhybh) then                              
!.ab.#endif
          if(qeqcor) then
             rdum(1)=zero
             rdum(2)=zero
             rdum(3)=zero
             qd(1)=cgtot*sqrt(reqcor)
#if KEY_BLOCK==1
             chf2(1)=cgtot*sqrt(reqcor)               
             if (qhybh) then
                ihybh=ewutil
                curcg=cgtot
                chfact=chfact* two*kappa*ccelec/sqrt(pi)

                !.ab. Derivative of Background correction is included here.
                !.ab. CHF2 is filled with orig charge for (1:system, 2:react, 3:prod).
                !.ab. Hope REQCOR=1....
                call do_pme_cor(natom,curcg,cg1,cg,chfact,chf2,qeutil)
                chf2(1)=curcg
                qd(1)=curcg
                ihybh=ewqcor
                !.ab.checked in SUMHYB.
                !           IF (EWQCOR /= 47)
                !     A          CALL WRNDIE(-5,'<PME>','EWQCOR should be 47. See code.')
             endif
#endif 
             !.ab.
             ! process the
             ione=1
             imxcent(1)=imxcen
             imycent(1)=imycen
             imzcent(1)=imzcen
             !.ab.            call do_pme_ksp(qfinit,rewcut,ione,imxcent,imycent,imzcent,
             !.ab.     &           qd,recip,eprop(volume),kappa,
             !.ab.     &           cg1,
             call do_pme_ksp(qfinit,rewcut,ione,imxcent,imycent,imzcent, &
#if KEY_BLOCK==1
                  CHF2,                                  & 
#endif
#if KEY_BLOCK==0
                  QD,                                    & 
#endif
                  recip,eprop(volume),kappa, &
                  forder,eqcor,rdum(1),rdum(2), &
                  rdum(3),vircor, &
#if KEY_CHEQ==1
                  dch,qcg,                               & 
#endif
                  sizfftab,sizffwrk,xnsymm*forder,siz_q, &
#if KEY_BLOCK==1
                  qd,                                    & 
#endif
#if KEY_BLOCK==0
                  cg1,                                   & 
#endif
                  xnsymm,maxsym,xsymop   &
#if KEY_MNDO97==1
                 ,.false.                &   ! for q_qmmm_pme_do not done.
#endif
                 )

          endif
#if KEY_PARALLEL==1
          if(mynod == 0) then                       
#endif
             if(qeutil) then
                eutil = cgtot**2*kappa*ccelec/sqrt(pi)
#if KEY_BLOCK==1
                if(qhybh) eutil = curcg**2*kappa*ccelec/sqrt(pi)  
#endif
                eslf=0.0
                if(norot) then
                   do itrans=1,ntrans
                      ipt=(itrans-1)*12
                      tx=imtrns(ipt+10)
                      ty=imtrns(ipt+11)
                      tz=imtrns(ipt+12)
                      s2=max(rsmall,tx*tx+ty*ty+tz*tz)
                      rs=sqrt(s2)
                      if(rs <= ctofnb) then
                         call erfcd(rs,kappa,erfcx,drfc,erfmod)
                         ene= erfcx/rs
                         eslf = eslf + ene
                         drfc = reqcor*half*cgtot**2*ccelec* (drfc+ene)/s2
                         ewvirial(1) = ewvirial(1) - drfc*tx*tx
                         ewvirial(2) = ewvirial(2) - drfc*ty*tx
                         ewvirial(3) = ewvirial(3) - drfc*tz*tx
                         ewvirial(4) = ewvirial(4) - drfc*tx*ty
                         ewvirial(5) = ewvirial(5) - drfc*ty*ty
                         ewvirial(6) = ewvirial(6) - drfc*tz*ty
                         ewvirial(7) = ewvirial(7) - drfc*tx*tz
                         ewvirial(8) = ewvirial(8) - drfc*ty*tz
                         ewvirial(9) = ewvirial(9) - drfc*tz*tz
                      endif
                   enddo
                   eutil=eutil-eslf*half*cgtot**2*ccelec
                else
                   CALL WRNDIE(-3,'<PME>', &
                        'Cannot do Ewald QTOT correction for rotational symmetry')
                endif
                eutil=eutil*reqcor
             endif
#if KEY_PARALLEL==1
          endif                                     
#endif
!AvdV bugfix -- moved "#endif /*   (pmeplsma)*/" here ((and removed a confusion comment))
#endif /*   (pmeplsma)*/
!.ab.#if KEY_BLOCK==1 && KEY_PMEPLSMA==1
!.ab.endif
!.ab.#endif
    endif  !---------------end of charged system correction -------------
#if KEY_BLOCK==1
    if (qblock) then
       if (allocated(qtemp)) then
          deallocate(QTEMP,stat = alloc_err)
          if (alloc_err /= 0)  &
               call wrndie(-1,'<PME>','unable to deallocate QTEMP')
       endif
    endif
#endif /*  close BLOCK*/
    !
#if KEY_COLFFT==0
#if KEY_MNDO97==1 /*skip_in_mndo97*/
    if(.not. q_qmmm_pme_do) then
#endif /*skip_in_mndo97*/
#if KEY_PHMD==1
    if(.not. QPHMD .or. ( QPHMD .and. (QGBSW .or. QGBMV))) then ! Y Huang 2017 
#endif /* phmd */
       deallocate(fr1,fr2,fr3,cg1,stat=alloc_err)
       if(alloc_err /= 0 ) write(0,*)"unable to deallocate fr1,2,3"
       deallocate(qarray,stat=alloc_err)
       if(alloc_err /= 0 ) write(0,*)"unable to deallocate qarray"
       call deallocate_bspline()
#if KEY_PHMD==1
    endif
#endif /* phmd */
#if KEY_MNDO97==1 /*skip_in_mndo97*/
    end if
#endif /*skip_in_mndo97*/
#endif 
    !
#if KEY_DOMDEC_GPU==1
    if (.not.q_domdec .or. .not.q_gpu .or. gpu_code_version==1) then
       ! When reciprocal forces are computed on GPU,
       ! we don't have the virial yet here. We do the below conversion
       ! in the CUDA code (in domdec_gpu/Grid.cu, Grid::get_energy_virial)
#endif
       cfact = ccelec/xnsymm_t**2
       !c      cfact = ccelec/xnsymm
       !
       if(qeksum) then
          eksum = eksum*cfact
          ! add in pressure contributions
          ewvirial(1) = ewvirial(1) - virial(1)*cfact 
          ewvirial(2) = ewvirial(2) - virial(2)*cfact
          ewvirial(3) = ewvirial(3) - virial(3)*cfact
          ewvirial(4) = ewvirial(4) - virial(2)*cfact
          ewvirial(5) = ewvirial(5) - virial(4)*cfact
          ewvirial(6) = ewvirial(6) - virial(5)*cfact
          ewvirial(7) = ewvirial(7) - virial(3)*cfact
          ewvirial(8) = ewvirial(8) - virial(5)*cfact
          ewvirial(9) = ewvirial(9) - virial(6)*cfact
#if KEY_LJPME==1
          if (qcpme) then
             ! Helpme ordering:  XX XY YY XZ YZ ZZ
             ewvirial(1) = ewvirial(1) + ljvirial(1)
             ewvirial(2) = ewvirial(2) + ljvirial(2)
             ewvirial(3) = ewvirial(3) + ljvirial(4)
             ewvirial(4) = ewvirial(4) + ljvirial(2)
             ewvirial(5) = ewvirial(5) + ljvirial(3)
             ewvirial(6) = ewvirial(6) + ljvirial(5)
             ewvirial(7) = ewvirial(7) + ljvirial(4)
             ewvirial(8) = ewvirial(8) + ljvirial(5)
             ewvirial(9) = ewvirial(9) + ljvirial(6)
          else
             ! CHARMM ordering:  XX XY XZ YY YZ ZZ
             ewvirial(1) = ewvirial(1) + ljvirial(1)
             ewvirial(2) = ewvirial(2) + ljvirial(2)
             ewvirial(3) = ewvirial(3) + ljvirial(3)
             ewvirial(4) = ewvirial(4) + ljvirial(2)
             ewvirial(5) = ewvirial(5) + ljvirial(4)
             ewvirial(6) = ewvirial(6) + ljvirial(5)
             ewvirial(7) = ewvirial(7) + ljvirial(3)
             ewvirial(8) = ewvirial(8) + ljvirial(5)
             ewvirial(9) = ewvirial(9) + ljvirial(6)
          endif
#endif
       else
          eksum=0.0
       endif
#if KEY_DOMDEC_GPU==1
    endif
#endif
    
#if KEY_PMEPLSMA==0
    if(qeqcor) then
       if(abs(cgtot) > pt0001 .and. reqcor.gt.zero) then
          eqcor = -eqcor*cfact
          ewvirial(1) = ewvirial(1) + vircor(1)*cfact
          ewvirial(2) = ewvirial(2) + vircor(2)*cfact
          ewvirial(3) = ewvirial(3) + vircor(3)*cfact
          ewvirial(4) = ewvirial(4) + vircor(2)*cfact
          ewvirial(5) = ewvirial(5) + vircor(4)*cfact
          ewvirial(6) = ewvirial(6) + vircor(5)*cfact
          ewvirial(7) = ewvirial(7) + vircor(3)*cfact
          ewvirial(8) = ewvirial(8) + vircor(5)*cfact
          ewvirial(9) = ewvirial(9) + vircor(6)*cfact
       endif
    endif
#endif 
    !
!!#if KEY_MNDO97==1
!!    if(lqmewd .and. qm_control_r%q_do_cpmd_pme) then
!!       ! put the mulliken charge on cg array
!!       do i=1,nquant
!!          nn     =iabs(qm_control_r%qminb(i))
!!          cg(nn) = qm_control_r%cgqmmm(i)
!!       end do
!!       ! correct energy: since ewald energy component includes qm charges.
!!#if KEY_PARALLEL==1
!!       if(mynod.eq.0) then
!!#endif
!!          if(qeksum) eksum = eksum - qm_control_r%E_ewald_corr  ! substract this amount.
!!#if KEY_PARALLEL==1
!!       end if
!!#endif
!!    end if
!!#endif

    return
  end subroutine pme
  
  !===========================================================================
  !            DO_PME_KSP
  !===========================================================================
  !
  subroutine do_pme_ksp(qfinit,rewcut, &
       natom,x,y,z,cg,recip,volume,ewald_coeff, &
       forder, &
       eer,dx,dy,dz,virial, &
#if KEY_CHEQ==1
       dch,qcg,                                               & 
#endif
       sizfftab,sizffwrk,siztheta,siz_q, &
       cg1,xnsymm,maxsyml,xsymop   &
#if KEY_MNDO97==1
      ,q_qmmm_pme_do               &
#endif
      )

  use new_timer,only:timer_start,timer_stop,timer_stpstrt,  & 
     T_fillg,T_FFT,T_scsum,T_grads                       
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
     !       Modified to add BLOCK code for energy decomposition
     !       Author:
     !             Hiqmet Kamberaj
     !             November 2007
     !
  use number
  use parallel  
  use stream
#if KEY_GRAPE==1
  use grape                  
#endif
#if KEY_BLOCK==1
  use block_ltm ! block_fcm -> block_ltm -- Y Huang 2017              
#endif
  use memory
#if KEY_MNDO97==1
  use mndo97,only: lqmewd
  use qm1_info,only: qm_control_r,qm_main_r,mm_main_r
  use qmmmewald_module, only : qmmm_ewald_r
  use gamess_fcm, only : IGMSEL
#endif
     !
    logical        :: qfinit
    real(chm_real) :: rewcut
    integer        :: forder
    real(chm_real) :: recip(3,3),volume,ewald_coeff
    real(chm_real) :: x(*),y(*),z(*),dx(*),dy(*),dz(*)
    real(chm_real) :: cg(*)
    real(chm_real) :: eer,virial(6)
#if KEY_CHEQ==1
    real(chm_real) :: dch(*)
    logical        :: qcg
#endif 
    integer natom
    integer l_rctable,l_qq
#if KEY_PARALLEL==1
    integer i
#else /**/
    integer i,j,k
#endif 
    ! sizes of some arrays
    integer   sizfftab,sizffwrk,siztheta,siz_q

    ! storage: These arrays can be tossed after leaving this routine
    real(chm_real),intent(in) :: cg1(*)

    integer xnsymm,maxsyml
    integer xsymop(3,4,xnsymm)

    real(chm_real) :: scale,qt
    integer nfftdim1,nfftdim2,nfftdim3,nfftable,nffwork
    integer latm
    integer igood, kbot, ktop, i0
#if KEY_BLOCK==1
    INTEGER    ::  KK                         
#endif
#if KEY_MNDO97==1
    integer    :: nquant
    logical    :: q_grad_and_pot, q_cpmd
    logical    :: q_qmmm_pme_do
#endif
    !.ab.Update nattot....
    nattot=natom*xnsymm
    !.ab.
    !
#if KEY_GRAPE==1 /*cuda1*/
    if((igrape/10) == 1)then
       call timer_start(T_fillg)                          
       call cuda_pme( &
             volume,ewald_coeff,recip,x,y,z,cg,nattot, &
#if KEY_CHEQ==1
             dch, qcg,                                 &  
#endif
             dx,dy,dz,eer)
       call timer_stpstrt(T_fillg,T_FFT)                  
       call timer_stpstrt(T_FFT,T_scsum)                  
       call timer_stpstrt(T_scsum,T_FFT)                  
       call timer_stpstrt(T_FFT,T_grads)                  
       call timer_stop(T_grads)                           
    ELSE
#endif /* (cuda1)*/
       !
       !  get some integer array dimensions
       call get_fftdims(nfftdim1,nfftdim2,nfftdim3,nfftable,nffwork)

       scale = ONE
       allocate(tmpy(2*nfftdim1),alpha(nfft1),beta(nfft1)) ! move here.
       !
       ! H Kamberaj: BLOCK code added, November 2007
#if KEY_BLOCK==1
       IF (QBLOCK) THEN
          call fft3d0rc(0,scale,qtemp,nfftdim1,nfftdim2,tmpy,alpha,beta)
       ELSE
#endif
          call fft3d0rc(0,scale,qarray,nfftdim1,nfftdim2,tmpy,alpha,beta)
#if KEY_BLOCK==1
       ENDIF
#endif
       !.ab.x/box and images + ch(img). (cg1 filled below if test.false).
       if(xnsymm > 1) &
            call get_sc_fract(fr1,fr2,fr3,cg1, &
            natom,nattot,x,y,z,recip, &
            xnsymm,maxsyml,xsymop,cg)


       !       make array for keeping track of atoms important to
       !         this processor
       latm=nattot
       call chmalloc('pme.src','do_pme_ksp','lmy_ks',latm,intg=lmy_ks)
#if KEY_SQUANTM==1 || KEY_MNDO97==1 || KEY_QCHEM==1 || KEY_QTURBO==1 || KEY_G09==1
       call chmalloc('pme.src','do_pme_ksp','lmy_ks_inv',nattot,intg=lmy_ks_inv) 
#endif
       !-------- symmetrical case ------------------------------
       !        fill frac coords and thetas in fill_ch_grid
       !              use min image charge array: cg
       call timer_start(T_fillg)                          
       !.ab.HybH.do some charge scaling...
       !.ab.Routine call changed to scale charges...
       !.ab. Note: if XNSYMM == 1: this make a scaled copy cg->cg1
       !.ab.       if XNSYMM >  1: CG1 & CHARGE have the same memloc
       !.ab.                       and this is a scaling
       if(xnsymm == 1) then
          call fill_ch_grid(igood, kbot, ktop,nattot,cg, &
#if KEY_BLOCK==1
               CG1,                                        & 
#endif
               x,y,z,recip,natom,xnsymm, &
               nfftdim1,nfftdim2, &
               lmy_ks,latm, &
#if KEY_SQUANTM==1 || KEY_MNDO97==1 || KEY_QCHEM==1 || KEY_QTURBO==1 || KEY_G09==1
               lmy_ks_inv,  &   
#endif
               mxyslabs)

       ELSE
          !-------- asymmetrical case ------------------------------
          !        fill frac coords  in GET_SC_FRACT but thetas in fill_ch_grid
          !           use the whole unit cell charge array: cg1
          call fill_ch_grid( &
               igood, kbot, ktop, &
               nattot,cg1, &
#if KEY_BLOCK==1
               CG1,                                        & 
#endif
               x,y,z,recip,natom,xnsymm, &
               nfftdim1,nfftdim2, &
               lmy_ks,latm, &
#if KEY_SQUANTM==1 || KEY_MNDO97==1 || KEY_QCHEM==1 || KEY_QTURBO==1 || KEY_G09==1
               lmy_ks_inv,  &   
#endif
               mxyslabs)
       endif
       call timer_stpstrt(T_fillg,T_FFT)                  

       ! move to above..
       !!allocate(tmpy(2*nfftdim1),alpha(nfft1),beta(nfft1))

#if KEY_MNDO97==1 /*mndo97*/
       if(q_qmmm_pme_do) then
          ! to use in gradient evaluation.
          igood_qmmm=igood
          nquant    = qm_main_r%numat

          q_cpmd        = qm_control_r%q_do_cpmd_pme
          q_grad_and_pot=.true.        ! for the computation of potential.
          if(q_cpmd) then
             ! qarray_mm is used to take only qarray qm-mm interaction potential.
             qarray_mm(1:siz_q)=qarray(1:siz_q)

             ! Fill charge-grid array relevant for qm atoms only.
             ! This will be used later for the computation of gradient component
             ! applied to each MM atoms by each QM atoms.
             !
             ! For q_cpmd case.. qarray contains qm-mm + qm-qm component.
             !
             call fill_ch_grid_qm_mm(kbot,ktop,nquant,mm_main_r%qm_charges, &
                                     x,y,z,recip,natom,xnsymm, &
                                     nfftdim1,nfftdim2,mxyslabs, &
                                     lmy_ks_inv,latm, &
                                     qm_control_r%qminb,q_grad_and_pot,q_cpmd)
             ! for qm-mm component
             call fft_backrc(Qarray_mm,nfftdim1,nfftdim2,nfftdim3,nffwork)
          else
             ! Fill charge-grid array relevant for qm atoms only.
             ! This will be used later for the computation of gradient component
             ! applied to each MM atoms by each QM atoms.
             call fill_ch_grid_qm_mm(kbot,ktop,nquant,mm_main_r%qm_charges, &
                                     x,y,z,recip,natom,xnsymm, &
                                     nfftdim1,nfftdim2,mxyslabs, &
                                     lmy_ks_inv,latm, &
                                     qm_control_r%qminb,q_grad_and_pot,q_cpmd)
          end if
       end if
#endif /*mndo97*/

       !
       ! H Kamberaj: Block code added, November 2007
#if KEY_BLOCK==1
       if (qblock) then
          if (.not. allocated(qtemp) .or. .not. allocated(qarray)) then
             call wrndie(-3, '<DO_PME_KSP>', 'QTEMP or QARRAY not allocated')
          endif
          if (prnlev  >  6) then
             if((ntot/=size(qtemp)).or.(nblock*ntot/=size(qarray)) ) then
                write(outu,'(4I10)') ntot,size(qtemp),nblock*ntot, &
                     size(qarray)
                CALL WRNDIE(-3,'<DO_PME_KSP>', &
                     'Mismatch of QTEMP array size')
             endif
          endif
          do kk=1, nblock
             i0=(kk-1)*ntot
             do i=1, ntot
                qtemp(i)=qarray(i0+i)
             enddo
             call fft_backrc(qtemp, &
                  nfftdim1,nfftdim2,nfftdim3,nffwork)
             do i=1, ntot
                qarray(i0+i)=qtemp(i)
             enddo
          enddo
       else
#endif
          ! .ab.Ok.TF^-1(Q)
          call fft_backrc(Qarray,nfftdim1,nfftdim2,nfftdim3,nffwork)
#if KEY_BLOCK==1
       endif
#endif
       !

#if KEY_MNDO97==1 /*mndo97*/
       if(q_qmmm_pme_do) then
          ! save for later virial calculation (none q_cpmd case).
          if(.not. q_cpmd) qarray_mm(1:siz_q)=Qarray(1:siz_q)
       end if
#endif /*mndo97*/

       call timer_stpstrt(T_FFT,T_scsum)                  
#if KEY_MNDO97==1
       ! also, compute (B*C)*F(qarray) in the routine.
#endif
       !.ab.Calculate.E.
       call scalar_sum(qfinit,rewcut, &
            ewald_coeff,volume,recip, &
            nfftdim1,nfftdim2,nfftdim3,eer,virial  &
#if KEY_MNDO97==1
           ,q_qmmm_pme_do   &
#endif
           )
       !.ab.On return: Q=FT(Q).B.C
       call timer_stpstrt(T_scsum,T_FFT)                  
       !
       ! H Kamberaj: Block code added, November 2007
#if KEY_BLOCK==1
       if (qblock) then
          do kk=1, nblock
             i0=(kk-1)*ntot
             do i=1, ntot
                qtemp(i)=qarray(i0+i)
             enddo
             call fft_forwardrc(Qtemp,nfftdim1,nfftdim2,nfftdim3,nffwork)
             do i=1, ntot
                qarray(i0+i)=qtemp(i)
             enddo
          enddo
       else
#endif
#if KEY_MNDO97==1
          ! now, forward fourier transform of qarray produces
          ! the convolution of theta(=F(B*C)) and Qarray.
#endif
          call fft_forwardrc(Qarray,nfftdim1,nfftdim2,nfftdim3,nffwork)
#if KEY_BLOCK==1
       endif
#endif /*    close block */

#if KEY_MNDO97==1 /*mndo97*/

       if(q_qmmm_pme_do) then
          ! forward fft of qarray_mm (for qm-qm component)
          if(q_cpmd) call fft_forwardrc(Qarray_mm,nfftdim1,nfftdim2,nfftdim3,nffwork)

          call potential_sumrc_qm_mm(igood, kbot, ktop, natom, nquant,                 &
                                     Qarray,Qarray_mm,                                 &
                                     qmmm_ewald_r%empot_pme,qmmm_ewald_r%empot_qm_pme, &
                                     qm_atm_grad_comp,                                 &
                                     recip,volume,forder,nfftdim1,nfftdim2,nfftdim3,   &
                                     lmy_ks_inv,latm,xnsymm,                           &
                                     igmsel,qm_control_r%mminb2(1:natom),qm_control_r%qminb, &
                                     q_cpmd)
       end if
#endif
       !
       !.ab.Now Q=(Theta_rec*Q). Essmann etal 1995 Eq.4.7.
       !!!deallocate(tmpy,alpha,beta)  <= moved to below.
       call timer_stpstrt(T_FFT,T_grads)                  
       !.ab.HYBH.make an other routine to avoid optimization isues.
#if KEY_BLOCK==1
       IF(.NOT.QHYBH) THEN                               
#endif
          call grad_sumrc( &
               igood, kbot, ktop, &
               natom,cg,recip, &
               dx,dy,dz, &
#if KEY_CHEQ==1
               dch,qcg,                                     & 
#endif
               forder,nfftdim1,nfftdim2,nfftdim3, &
               lmy_ks,latm, &
               xnsymm)
          !.ab.HybH part.note we use cg1 & cg.
#if KEY_BLOCK==1
       ELSE
          call grad_sumrch( &
               igood, kbot, ktop, &
               natom,cg,cg1,recip, &
               dx,dy,dz, &
#if KEY_CHEQ==1
               dch,qcg,                                      & 
#endif
               forder,nfftdim1,nfftdim2,nfftdim3, &
               lmy_ks,latm, &
               xnsymm)
       ENDIF
#endif 
       !.ab.
       call timer_stop(T_grads)                           
#if KEY_MNDO97==1 /*skip_memory*/
       if(.not. q_qmmm_pme_do) then
#endif            /*skip_memory*/
          deallocate(tmpy,alpha,beta)
          call chmdealloc('pme.src','do_pme_ksp','lmy_ks',latm,intg=lmy_ks)
#if KEY_SQUANTM==1 || KEY_MNDO97==1 || KEY_QCHEM==1 || KEY_QTURBO==1 || KEY_G09==1
          call chmdealloc('pme.src','do_pme_ksp','lmy_ks_inv',nattot,intg=lmy_ks_inv) 
#endif
#if KEY_MNDO97==1
       end if
#endif
       !
#if KEY_GRAPE==1
    ENDIF                 
#endif
    !
    RETURN
  END subroutine do_pme_ksp
  
!===========================================================================
!            DO_PME_PHMD
!===========================================================================
subroutine do_pme_phmd(dpH_Theta,pH_Theta,NTitr,Grplist,SP_GRP, &
                       natom,x,y,z,cg,Qstate1,Qstate2)

    !******************************************************************
    ! Calculate the first derivative of K-Space with respect to titration coordinates 
    !
    ! dpH_theta(NTitr) : Force on titration coordinates
    ! pH_theta(NTitr) : Titration coordinates
    ! GrpList(natom) : Titration number list
    ! SP_GRP(NTitr) : Titration type list
    ! Qstate1(2,natom) : charge for prot tautomers
    ! Qstate2(2,natom) : charge for deprot tautomers
    ! CG(natom) : Charge grid
    !
    ! Yandong Huang, July 2016
    !******************************************************************
    use number
    use energym
    use inbnd
    use parallel
    use stream
    use image                  !xtlabc
    use pmeutil                !mxyslabs,mxzslabs
    use consta,only:ccelec
    use prssre,only:getvol

    integer nfftdim1,nfftdim2,nfftdim3
    integer nfftable,nffwork,size_Q,nattot
    integer igood,kbot,ktop,latm
    integer NTitr,natom,aa,bb,G
    integer alloc_err,nstrt,nsolute,nskip

    integer,dimension(:) :: Grplist, SP_GRP
    integer,dimension(:),allocatable :: lmy_ks, lmy_ks_inv

    real(chm_real) :: cga,tmp,Lambda,X2
    real(chm_real) :: xtlinv(6),recip(3,3)

    real(chm_real),dimension(:) :: x, y, z, cg
    real(chm_real),dimension(:) :: dpH_Theta, pH_Theta
    real(chm_real),dimension(:,:) :: QState1, QState2
    real(chm_real),dimension(:),allocatable :: Qconv, dcg
    real(chm_real) :: chfact

    logical ok

    call getvol(eprop(volume))
    call invt33s(xtlinv,xtlabc,ok)
    recip(1,1) = xtlinv(1)
    recip(2,2) = xtlinv(3)
    recip(2,2) = xtlinv(3)
    recip(3,3) = xtlinv(6)
    recip(1,2) = xtlinv(2)
    recip(2,1) = xtlinv(2)
    recip(1,3) = xtlinv(4)
    recip(3,1) = xtlinv(4)
    recip(2,3) = xtlinv(5)
    recip(3,2) = xtlinv(5)

    ! NOTE: cut from source/nbonbds/pme.src:582
    !       all BLOCK sections were cut
    call get_fftdims(nfftdim1,nfftdim2,nfftdim3,nfftable,nffwork)
    nattot=natom*xnsymm

    ! NOTE: Construct dQarray for kspace derivative
    !       Qarray currently holds the Q*phi convolution


#if KEY_PARALLEL==1
    size_Q = max(2*nfftdim1*nfftdim2*mxyslabs, &
                 2*nfftdim1*nfftdim3*mxzslabs)
#if KEY_BLOCK==1
    chfact=one/numnod
#endif
#else /**/
    size_Q = 2*nfftdim1*nfftdim2*nfftdim3
#if KEY_BLOCK==1
    chfact=one
    size_Q = max(2*nfftdim1*nfftdim2*mxyslabs, &
         2*nfftdim1*nfftdim3*mxzslabs)
#endif
#endif

    allocate(Qconv(size_Q),dcg(natom),stat=alloc_err)
    if(alloc_err /= 0 ) write(0,*)"unable to allocate Qconv, dcg"
    DO aa=1,size_Q
        Qconv(aa) = Qarray(aa)
    ENDDO

#if KEY_PARALLEL==1
    latm=nattot
#else
    latm=nattot
#endif
    allocate(lmy_ks(latm),lmy_ks_inv(nattot),stat=alloc_err)
    if(alloc_err /= 0 ) write(0,*)"unable to allocate lmy_ks, lmy_ks_inv"

    nstrt=1
    nskip=1
    nsolute=natom ! NOTE: phmd.src:124

!##IF PARALLEL
!    nstrt=mynodp
!    nskip=numnod
!##ENDIF

    do bb=1,ntitr

        dcg(1:natom)=ZERO
        DO aa=nstrt,nsolute,nskip
            tmp=ZERO
            G = GrpList(aa) !! Question this

            ! NOTE: Check if titratable
            IF ((G .EQ. bb ) .or. (G .EQ. (bb-1))) THEN

                ! NOTE: Check for tautomerization lambdas
                IF ( SP_GRP(bb) .LE. 0 ) THEN
                    tmp = Qstate2(1,aa) - Qstate1(1,aa)                      !  dq/dlamb,x=1
                ELSEIF ((SP_GRP(bb) .EQ. 2) .or. (SP_GRP(bb) .EQ. 4)) THEN     !  x
                    Lambda = DSIN(pH_Theta(bb-1))**2                          !  lamb
                    tmp = Lambda*(Qstate2(1,aa) - Qstate2(2,aa)) + &         !  dq/dx
                           (1-Lambda)*(Qstate1(1,aa) - Qstate1(2,aa))
                ELSEIF ((SP_GRP(bb) .EQ. 1) .or. (SP_GRP(bb) .EQ. 3)) THEN     !  lamb
                    X2 = DSIN(pH_Theta(bb+1))**2                              !  x
                    tmp = X2*( Qstate2(1,aa)-Qstate1(1,aa) ) + &             !  dq/dlamb
                           (1-X2)*( Qstate2(2,aa)-Qstate1(2,aa) )
                ELSE ! 5 and above
                    tmp = Qstate2(1,aa) - Qstate1(1,aa)                      !  dq/dlamb,x=1
                ENDIF
                 dcg(aa)=tmp                                                 ! dq/dlamb.or.x
            ENDIF !titratable
       ENDDO !aa natom

        ! NOTE: CG1 only used if QHYBH for BLOCK, it's removed from the para list
        call fill_dch_grid(igood,kbot,ktop,nattot,dcg, &
#if KEY_BLOCK==1
             CG1,                                     &  !##BLOCK
#endif
             x,y,z,recip,natom,xnsymm, &
             nfftdim1,nfftdim2, &
             lmy_ks,latm, &
#if KEY_SQUANTM==1 || KEY_MNDO97==1 || KEY_QCHEM==1 || KEY_QTURBO==1 || KEY_G09==1
             lmy_ks_inv, &
#endif
             mxyslabs)

        tmp=ZERO

        DO aa=1,size_q
            ! NOTE: The array is mostly zero,
            !       this should save some time
            IF (Qarray(aa) .NE. 0.0) THEN
                tmp = tmp + Qarray(aa) * Qconv(aa)
            ENDIF
        ENDDO
        dpH_Theta(bb) = dpH_Theta(bb) + ccelec * tmp
    ENDDO !bb ntitr

    deallocate(Qconv,dcg,lmy_ks,lmy_ks_inv,stat=alloc_err)
    if(alloc_err /= 0 ) write(0,*)"unable to deallocate Qconv, dcg, lmy_ks, lmy_ks_inv"

    ! NOTE: Do here instead of in PME
    !       QPHMD must be set if this function is called (Now it's commented out in PME)
    deallocate(fr1,fr2,fr3,cg1,stat=alloc_err)
    if(alloc_err /= 0 ) write(0,*)"unable to deallocate fr1,2,3"
    deallocate(qarray,stat=alloc_err)
    if(alloc_err /= 0 ) write(0,*)"unable to deallocate qarray"
    call deallocate_bspline()

    RETURN
end subroutine do_pme_phmd

  !***********************************************************************
  !                 +----------------------------+
  !                 |        FILL_DCH_GRID        |
  !                 +----------------------------+
  !       Yandong Huang, based on fill_ch_grid
  !     Spread only non-zero dq/dlambda onto the grids to save CPU
  !***********************************************************************
  subroutine fill_dch_grid(igood, kbot, ktop,numatoms,charge, &
#if KEY_BLOCK==1
       cg1,                                  &
#endif
       x,y,z,recip,natom,xnsymm, &
       nfftdim1,nfftdim2, &
       my_ks,latm, &
#if KEY_SQUANTM==1 || KEY_MNDO97==1 || KEY_QCHEM==1 || KEY_QTURBO==1 || KEY_G09==1
       my_ks_inv, &
#endif
       nfftdim3)

  use pmeutil,only:nfft1,nfft2,nfft3,forder, &
     theta1,theta2,theta3, &
     dtheta1,dtheta2,dtheta3,fill_bspline &
#if KEY_PARALLEL==1
     ,mxystart,mxyslabs  &
#endif
     ; !ends the line when PARALLEL not on.
     !
     ! This routine fills the charge grid, Q.
     !
     !---------------------------------------------------------------------
     ! INPUT:
     !      numatoms:  number of atoms
     !      charge: the array of atomic charges
     !      nfft1,nfft2,nfft3: the charge grid dimensions
     !      nfftdim1,nfftdim2,nfftdim3: physical charge grid dims
     ! OUTPUT:
     !
     ! This routine fills the charge grid, Q.
     ! C.ab. scales charges (CG1) according to HybH scheme...
     !
     !---------------------------------------------------------------------
     ! INPUT:
     !      numatoms:  number of atoms
     !      charge: the array of atomic charges
     !      theta1,theta2,theta3: the spline coeff arrays
     !      fr1,fr2,fr3 the scaled and shifted fractional coords
     !      nfft1,nfft2,nfft3: the charge grid dimensions
     !      nfftdim1,nfftdim2,nfftdim3: physical charge grid dims
     ! OUTPUT:
     !      Q the charge grid
     !
     !      Modified to add BLOCK code
     !      Author:
     !            Hiqmet Kamberaj
     !            November 2007
     !
     !---------------------------------------------------------------------
     !
  use number
  use parallel
  use dimens_fcm
#if KEY_BLOCK==1
     !av_080626
  use block_ltm ! block_fcm -> block_ltm -- Y Huang 2017
  use stream
#endif
    integer numatoms,natom
    integer dim_q0,nfftdim1,nfftdim2,nfftdim3
    real(chm_real) :: x(*),y(*),z(*), recip(9)
    real(chm_real) :: charge(*)

#if KEY_BLOCK==1
    INTEGER       :: dim_q,INX,INXT,IBL
    !av_080626
    !.ab.HybH. and .av.BLOCK
    real(chm_real) :: cg1(*)
    INTEGER NUMA,REATOM
    real(chm_real) CHF(3)
    !.ab.Can't include exfunc.f90 because conflict name with order...
#endif
    !.ab.
    !
    integer n,ith1,ith2,ith3,i,j,k,kq,ipt1,ipt2,ipt3,ipt
    real(chm_real) :: prod,proda

    integer enumtasks,itask,kdel,kbot0
    real(chm_real) :: fr1n,fr2n,fr3n,w
    integer xnsymm,igoody, igdt
    integer igood, kbot, ktop
    integer latm,my_ks(latm)
#if KEY_SQUANTM==1 || KEY_MNDO97==1 || KEY_QCHEM==1 || KEY_QTURBO==1 || KEY_G09==1
    integer :: my_ks_inv(numatoms)
#endif
    integer nfftdimrc

    if(nfft1/2 /= (nfft1+1)/2)then
       call wrndie(-5,'<pme fill charge grid>' &
            ,'fftx dimension not even ')
    endif
    nfftdimrc=nfft1+4
    !
    igood=0
    igoody=0
    my_ks(1:latm)=0
#if KEY_SQUANTM==1 || KEY_MNDO97==1 || KEY_QCHEM==1 || KEY_QTURBO==1 || KEY_G09==1
    my_ks_inv(1:numatoms)=0
#endif
#if KEY_PARALLEL==1
    enumtasks = 1
    kdel = nfft3/enumtasks
    if ( kdel  ==  0 )kdel = 1
    !
    kbot0 = mxystart(mynod)
    kbot = kbot0 + 1
    ktop = kbot0 + mxyslabs
#else /**/
    kbot0 = 0
    kbot = 1
    ktop = nfft3
#endif

    !
    dim_q0 = 2*nfftdim1*nfftdim2*nfftdim3
    !
#if KEY_BLOCK==1
    if (qblock) then
       if (.not. allocated(qarray)) then
          call wrndie(-3, '<FILL_CH_GRID>', 'QARRAY not allocated')
       endif
       dim_q = nblock*dim_q0
       if (prnlev  >  6) then
          write(outu,'(A6,3I6)')"dim-q=",dim_q,nblock,size(qarray)
       endif
       qarray(1:dim_q)=zero
    else
       qarray(1:dim_q0)=zero
    endif
#else /**/
    qarray(1:dim_q0)=zero
#endif /*   close block*/
    !
    !
    !------------------------------------------
    !          MFC NOTE: THERE COULD BE A PRE-FILTER HERE TO ELIMINATE
    !                      MOST OF THE ATOMS AND KEEP A LIST OF APPROPRIATE
    !                      ATOMS EITHER FOR EACH PE, EACH X-Y SLAB,
    !                      OR EACH Q ELEMENT
    !          MFC NOte Note: Looks like I am doing that filter now....
    !------------------------------------------
    !.ab.HybH stuff.
#if KEY_BLOCK==1
    IF(QHYBH) THEN
       !.ab. Initialise -> les coefs. Charges CHF(1..3)
       CHF(1)=1.
       CHF(2)=1.-HYBHLB
       CHF(3)=HYBHLB
       REATOM=NUMATOMS/XNSYMM
    ENDIF
#endif
    !.ab.
    do n = 1,numatoms

       IF ( charge(n) .EQ. 0.0 ) THEN
          CYCLE  ! Skip zero elements -- Y Huang 2017
       ENDIF

       !av_080628
#if KEY_BLOCK==1
       if (qblock) then
          ibl = iblckp(N)
          inxt = (ibl-1)*dim_q0
       endif
#endif /*   close BLOCK*/
       !av_080628
#if KEY_BLOCK==1
       IF(QHYBH) THEN
          NUMA=MOD(N-1,REATOM)+1
          IBL = IBLCKP(NUMA)
          IF (IBL /= 1) THEN
             CG1(N)=CHARGE(N)*CHF(IBL)
          ELSE
             CG1(N)=CHARGE(N)
          ENDIF
          !.ab. it would be more elegant
          !.ab. to do   CG1(N)=CHARGE(NUMA)*CHF(IBL) ...
          !.ab. Note: if XNSYMM == 1: this make a copy(scaled)
          !.ab.       if XNSYMM >  1: CG1 & CHARGE have the same memloc
          !.ab.                       and this is a scaling
       ENDIF
#endif
       !.ab.

       if(xnsymm > 1)then
          fr3n=fr3(n)
       else
          w = x(n)*recip(7)+y(n)*recip(8)+z(n)*recip(9)
          fr3n = nfft3*(w - anint(w) + half)
       endif
       k = int(fr3n) - forder + 1 + nfft3

       kq = k          ! save k
       k = k + 1
       if(k > nfft3) k = k - nfft3
       igood=igood+1
       my_ks(igood)=n
#if KEY_SQUANTM==1 || KEY_MNDO97==1 || KEY_QCHEM==1 || KEY_QTURBO==1 || KEY_G09==1
       my_ks_inv(n)=igood
#endif
       if(n <= natom)igoody=igood
       if(xnsymm == 1)then
          w = x(n)*recip(1)+y(n)*recip(2)+z(n)*recip(3)
          fr1n = nfft1*(w - anint(w) + half)
          w = x(n)*recip(4)+y(n)*recip(5)+z(n)*recip(6)
          fr2n = nfft2*(w - anint(w) + half)
          fr1(igood)=fr1n
          fr2(igood)=fr2n
          fr3(igood)=fr3n
          igdt=igood
       else
          fr1n=fr1(n)
          fr2n=fr2(n)
          igdt=min(n,natom+1)
       endif
       w = fr1n-int(fr1n)
       call fill_bspline(w,forder,theta1(1,igood),dtheta1(1,igdt))
       w = fr2n-int(fr2n)
       call fill_bspline(w,forder,theta2(1,igood),dtheta2(1,igdt))
       w = fr3n-int(fr3n)
       call fill_bspline(w,forder,theta3(1,igood),dtheta3(1,igdt))
       k = kq          ! restore k

       !
       ! NOTE from Antti-Pekka Hynninen 2/13/2012:
       !
       ! BLOCK flag check moved outside inner loop
       ! => performance improvement
       !

#if KEY_BLOCK==1 /*block*/
       if (qblock) then
          do ith3 = 1,forder
             k=k+1
             if(k > nfft3) k=k-nfft3
             kq=k
#if KEY_PARALLEL==1
             if ( k  >=  kbot .and. k  <=  ktop )then
                kq = k - kbot0
#endif
#if KEY_BLOCK==1
                IF(QHYBH) THEN
                   proda = theta3(ith3,igood)*cg1(n)
                ELSE
#endif
                   proda = theta3(ith3,igood)*charge(n)
#if KEY_BLOCK==1
                ENDIF
#endif
                !
                j = int(fr2n) - forder + 1 + nfft2
                ipt1 = (kq-1)*nfftdim2 - 1
                !
                i = int(fr1n) - forder + 1 + nfft1
                if(i >= nfft1) i=i-nfft1
                !
                do ith2 = 1,forder
                   j=j+1
                   if(j > nfft2) j=j-nfft2
                   prod = theta2(ith2,igood)*proda
                   ipt2= ((ipt1+j)*nfftdimrc+i)+1
                   ipt3= ipt2 + (nfft1-i)
                   !
                   do ith1 = 1,forder
                      !
                      ! Block code: H Kamberaj, November 2007
                      INX = INXT + IPT2
                      if (inx  >  dim_q) then
                         call wrndie(-5, '<FILL_CH_GRID>', &
                              'Index INX  >  DIM_Q')
                      endif
                      Qarray(INX)=Qarray(INX)+THETA1(ITH1,IGOOD)*PROD
                      !
                      ipt2=ipt2+1
                      if(ipt2 >= ipt3) ipt2=ipt2-nfft1
                   enddo
                enddo
#if KEY_PARALLEL==1
             endif
#endif
             !       (check to see if space overflow)
             if ( igood  > latm) &
                  call wrndie(-5,'<pme>' &
                  ,'fill_ch_grid igood  > latm ')
          enddo
       else
#endif /* (block)*/
          do ith3 = 1,forder
             k=k+1
             if(k > nfft3) k=k-nfft3
             kq=k
#if KEY_PARALLEL==1
             if ( k  >=  kbot .and. k  <=  ktop )then
                kq = k - kbot0
#endif
#if KEY_BLOCK==1
                IF(QHYBH) THEN
                   proda = theta3(ith3,igood)*cg1(n)
                ELSE
#endif
                   proda = theta3(ith3,igood)*charge(n)
#if KEY_BLOCK==1
                ENDIF
#endif
                !
                j = int(fr2n) - forder + 1 + nfft2
                ipt1 = (kq-1)*nfftdim2 - 1
                !
                i = int(fr1n) - forder + 1 + nfft1
                if(i >= nfft1) i=i-nfft1
                !
                do ith2 = 1,forder
                   j=j+1
                   if(j > nfft2) j=j-nfft2
                   prod = theta2(ith2,igood)*proda
                   ipt2= ((ipt1+j)*nfftdimrc+i)+1
                   ipt3= ipt2 + (nfft1-i)
                   !
                   do ith1 = 1,forder
                      Qarray(IPT2) = Qarray(IPT2) + THETA1(ITH1,IGOOD)*PROD
                      ipt2=ipt2+1
                      if(ipt2 >= ipt3) ipt2=ipt2-nfft1
                   enddo
                enddo
#if KEY_PARALLEL==1
             endif
#endif
             !       (check to see if space overflow)
             if ( igood  > latm) &
                  call wrndie(-5,'<pme>' &
                  ,'fill_ch_grid igood  > latm ')
          enddo
#if KEY_BLOCK==1
       endif
#endif
    enddo
    !
    igood=igoody
    return
  end subroutine fill_dch_grid



  
  !***********************************************************************
  !                 +----------------------------+
  !                 |        FILL_CH_GRID        |
  !                 +----------------------------+
  !***********************************************************************
  subroutine fill_ch_grid(igood, kbot, ktop,numatoms,charge, &
#if KEY_BLOCK==1
       cg1,                                  & 
#endif
       x,y,z,recip,natom,xnsymm, &
       nfftdim1,nfftdim2, &
       my_ks,latm, &
#if KEY_SQUANTM==1 || KEY_MNDO97==1 || KEY_QCHEM==1 || KEY_QTURBO==1 || KEY_G09==1
       my_ks_inv, &   
#endif
       nfftdim3)

  use pmeutil,only:nfft1,nfft2,nfft3,forder, &
     theta1,theta2,theta3, &
     dtheta1,dtheta2,dtheta3,fill_bspline &
#if KEY_PARALLEL==1
     ,mxystart,mxyslabs  &  
#endif
     ; !ends the line when PARALLEL not on.
     !
     ! This routine fills the charge grid, Q.
     !
     !---------------------------------------------------------------------
     ! INPUT:
     !      numatoms:  number of atoms
     !      charge: the array of atomic charges
     !      nfft1,nfft2,nfft3: the charge grid dimensions
     !      nfftdim1,nfftdim2,nfftdim3: physical charge grid dims
     ! OUTPUT:
     !
     ! This routine fills the charge grid, Q.
     ! C.ab. scales charges (CG1) according to HybH scheme...
     !
     !---------------------------------------------------------------------
     ! INPUT:
     !      numatoms:  number of atoms
     !      charge: the array of atomic charges
     !      theta1,theta2,theta3: the spline coeff arrays
     !      fr1,fr2,fr3 the scaled and shifted fractional coords
     !      nfft1,nfft2,nfft3: the charge grid dimensions
     !      nfftdim1,nfftdim2,nfftdim3: physical charge grid dims
     ! OUTPUT:
     !      Q the charge grid
     !
     !      Modified to add BLOCK code
     !      Author:
     !            Hiqmet Kamberaj
     !            November 2007
     !
     !---------------------------------------------------------------------
     !
  use number
  use parallel
  use dimens_fcm
#if KEY_BLOCK==1
     !av_080626
  use block_ltm ! block_fcm -> block_ltm -- Y Huang 2017
  use stream
#endif 
    integer numatoms,natom
    integer dim_q0,nfftdim1,nfftdim2,nfftdim3
    real(chm_real) :: x(*),y(*),z(*), recip(9)
    real(chm_real) :: charge(*)

#if KEY_BLOCK==1
    INTEGER       :: dim_q,INX,INXT,IBL
    !av_080626
    !.ab.HybH. and .av.BLOCK
    real(chm_real) :: cg1(*)
    INTEGER NUMA,REATOM
    real(chm_real) CHF(3)
    !.ab.Can't include exfunc.f90 because conflict name with order...
#endif 
    !.ab.
    !
    integer n,ith1,ith2,ith3,i,j,k,kq,ipt1,ipt2,ipt3,ipt
    real(chm_real) :: prod,proda

    integer enumtasks,itask,kdel,kbot0
    real(chm_real) :: fr1n,fr2n,fr3n,w
    integer xnsymm,igoody, igdt
    integer igood, kbot, ktop
    integer latm,my_ks(latm)
#if KEY_SQUANTM==1 || KEY_MNDO97==1 || KEY_QCHEM==1 || KEY_QTURBO==1 || KEY_G09==1
    integer :: my_ks_inv(numatoms)   
#endif
    integer nfftdimrc

    if(nfft1/2 /= (nfft1+1)/2)then
       call wrndie(-5,'<pme fill charge grid>' &
            ,'fftx dimension not even ')
    endif
    nfftdimrc=nfft1+4
    !
    igood=0
    igoody=0
    my_ks(1:latm)=0
#if KEY_SQUANTM==1 || KEY_MNDO97==1 || KEY_QCHEM==1 || KEY_QTURBO==1 || KEY_G09==1
    my_ks_inv(1:numatoms)=0             
#endif
#if KEY_PARALLEL==1
    enumtasks = 1
    kdel = nfft3/enumtasks
    if ( kdel  ==  0 )kdel = 1
    !
    kbot0 = mxystart(mynod)
    kbot = kbot0 + 1
    ktop = kbot0 + mxyslabs
#else /**/
    kbot0 = 0
    kbot = 1
    ktop = nfft3
#endif 

    !
    dim_q0 = 2*nfftdim1*nfftdim2*nfftdim3
    !
#if KEY_BLOCK==1
    if (qblock) then
       if (.not. allocated(qarray)) then
          call wrndie(-3, '<FILL_CH_GRID>', 'QARRAY not allocated')
       endif
       dim_q = nblock*dim_q0
       if (prnlev  >  6) then
          write(outu,'(A6,3I6)')"dim-q=",dim_q,nblock,size(qarray)
       endif
       qarray(1:dim_q)=zero
    else
       qarray(1:dim_q0)=zero
    endif
#else /**/
    qarray(1:dim_q0)=zero
#endif /*   close block*/
    !
    !
    !------------------------------------------
    !          MFC NOTE: THERE COULD BE A PRE-FILTER HERE TO ELIMINATE
    !                      MOST OF THE ATOMS AND KEEP A LIST OF APPROPRIATE
    !                      ATOMS EITHER FOR EACH PE, EACH X-Y SLAB,
    !                      OR EACH Q ELEMENT
    !          MFC NOte Note: Looks like I am doing that filter now....
    !------------------------------------------
    !.ab.HybH stuff.
#if KEY_BLOCK==1
    IF(QHYBH) THEN
       !.ab. Initialise -> les coefs. Charges CHF(1..3)
       CHF(1)=1.
       CHF(2)=1.-HYBHLB
       CHF(3)=HYBHLB
       REATOM=NUMATOMS/XNSYMM
    ENDIF
#endif 
    !.ab.
    do n = 1,numatoms
       !av_080628
#if KEY_BLOCK==1
       if (qblock) then
          ibl = iblckp(N)
          inxt = (ibl-1)*dim_q0
       endif
#endif /*   close BLOCK*/
       !av_080628
#if KEY_BLOCK==1
       IF(QHYBH) THEN
          NUMA=MOD(N-1,REATOM)+1
          IBL = IBLCKP(NUMA)
          IF (IBL /= 1) THEN
             CG1(N)=CHARGE(N)*CHF(IBL)
          ELSE
             CG1(N)=CHARGE(N)
          ENDIF
          !.ab. it would be more elegant
          !.ab. to do   CG1(N)=CHARGE(NUMA)*CHF(IBL) ...
          !.ab. Note: if XNSYMM == 1: this make a copy(scaled)
          !.ab.       if XNSYMM >  1: CG1 & CHARGE have the same memloc
          !.ab.                       and this is a scaling
       ENDIF
#endif 
       !.ab.

       if(xnsymm > 1)then
          fr3n=fr3(n)
       else
          w = x(n)*recip(7)+y(n)*recip(8)+z(n)*recip(9)
          fr3n = nfft3*(w - anint(w) + half)
       endif
       k = int(fr3n) - forder + 1 + nfft3

       kq = k          ! save k
       k = k + 1
       if(k > nfft3) k = k - nfft3
       igood=igood+1
       my_ks(igood)=n
#if KEY_SQUANTM==1 || KEY_MNDO97==1 || KEY_QCHEM==1 || KEY_QTURBO==1 || KEY_G09==1
       my_ks_inv(n)=igood             
#endif
       if(n <= natom)igoody=igood
       if(xnsymm == 1)then
          w = x(n)*recip(1)+y(n)*recip(2)+z(n)*recip(3)
          fr1n = nfft1*(w - anint(w) + half)
          w = x(n)*recip(4)+y(n)*recip(5)+z(n)*recip(6)
          fr2n = nfft2*(w - anint(w) + half)
          fr1(igood)=fr1n
          fr2(igood)=fr2n
          fr3(igood)=fr3n
          igdt=igood
       else
          fr1n=fr1(n)
          fr2n=fr2(n)
          igdt=min(n,natom+1)
       endif
       w = fr1n-int(fr1n)
       call fill_bspline(w,forder,theta1(1,igood),dtheta1(1,igdt))
       w = fr2n-int(fr2n)
       call fill_bspline(w,forder,theta2(1,igood),dtheta2(1,igdt))
       w = fr3n-int(fr3n)
       call fill_bspline(w,forder,theta3(1,igood),dtheta3(1,igdt))
       k = kq          ! restore k

       !
       ! NOTE from Antti-Pekka Hynninen 2/13/2012:
       !
       ! BLOCK flag check moved outside inner loop 
       ! => performance improvement
       ! 

#if KEY_BLOCK==1 /*block*/
       if (qblock) then
          do ith3 = 1,forder
             k=k+1
             if(k > nfft3) k=k-nfft3
             kq=k
#if KEY_PARALLEL==1
             if ( k  >=  kbot .and. k  <=  ktop )then
                kq = k - kbot0
#endif 
#if KEY_BLOCK==1
                IF(QHYBH) THEN                            
                   proda = theta3(ith3,igood)*cg1(n)      
                ELSE                                      
#endif
                   proda = theta3(ith3,igood)*charge(n)
#if KEY_BLOCK==1
                ENDIF                                     
#endif
                !
                j = int(fr2n) - forder + 1 + nfft2
                ipt1 = (kq-1)*nfftdim2 - 1
                !
                i = int(fr1n) - forder + 1 + nfft1
                if(i >= nfft1) i=i-nfft1
                !
                do ith2 = 1,forder
                   j=j+1
                   if(j > nfft2) j=j-nfft2
                   prod = theta2(ith2,igood)*proda
                   ipt2= ((ipt1+j)*nfftdimrc+i)+1
                   ipt3= ipt2 + (nfft1-i)
                   !
                   do ith1 = 1,forder
                      !
                      ! Block code: H Kamberaj, November 2007
                      INX = INXT + IPT2
                      if (inx  >  dim_q) then
                         call wrndie(-5, '<FILL_CH_GRID>', &
                              'Index INX  >  DIM_Q')
                      endif
                      Qarray(INX)=Qarray(INX)+THETA1(ITH1,IGOOD)*PROD
                      !
                      ipt2=ipt2+1
                      if(ipt2 >= ipt3) ipt2=ipt2-nfft1
                   enddo
                enddo
#if KEY_PARALLEL==1
             endif
#endif 
             !       (check to see if space overflow)
             if ( igood  > latm) &
                  call wrndie(-5,'<pme>' &
                  ,'fill_ch_grid igood  > latm ')
          enddo
       else
#endif /* (block)*/
          do ith3 = 1,forder
             k=k+1
             if(k > nfft3) k=k-nfft3
             kq=k
#if KEY_PARALLEL==1
             if ( k  >=  kbot .and. k  <=  ktop )then
                kq = k - kbot0
#endif 
#if KEY_BLOCK==1
                IF(QHYBH) THEN                            
                   proda = theta3(ith3,igood)*cg1(n)      
                ELSE                                      
#endif
                   proda = theta3(ith3,igood)*charge(n)
#if KEY_BLOCK==1
                ENDIF                                     
#endif
                !
                j = int(fr2n) - forder + 1 + nfft2
                ipt1 = (kq-1)*nfftdim2 - 1
                !
                i = int(fr1n) - forder + 1 + nfft1
                if(i >= nfft1) i=i-nfft1
                !
                do ith2 = 1,forder
                   j=j+1
                   if(j > nfft2) j=j-nfft2
                   prod = theta2(ith2,igood)*proda
                   ipt2= ((ipt1+j)*nfftdimrc+i)+1
                   ipt3= ipt2 + (nfft1-i)
                   !
                   do ith1 = 1,forder
                      Qarray(IPT2) = Qarray(IPT2) + THETA1(ITH1,IGOOD)*PROD
                      ipt2=ipt2+1
                      if(ipt2 >= ipt3) ipt2=ipt2-nfft1
                   enddo
                enddo
#if KEY_PARALLEL==1
             endif
#endif 
             !       (check to see if space overflow)
             if ( igood  > latm) &
                  call wrndie(-5,'<pme>' &
                  ,'fill_ch_grid igood  > latm ')
          enddo
#if KEY_BLOCK==1
       endif   
#endif
    enddo
    !
    igood=igoody
    return
  end subroutine fill_ch_grid
  
  !***********************************************************************
  !                 +----------------------------+
  !                 |        FFT_BACK            |
  !                 +----------------------------+
  !***********************************************************************
  !--------------------------------------------------------------
  subroutine fft_backrc(array, &
       nfftdim1,nfftdim2,nfftdim3, &
       nffwork)
  use pmeutil,only:fft3d0rc
    !
    !
    real(chm_real) :: array(*),scale
    integer nfftdim1,nfftdim2,nfftdim3
    integer nffwork
    !
    integer isign
    !
    isign = -1
    scale = 1.d0
    call FFT3D0rc(isign,scale,array, &
         nfftdim1,nfftdim2, &
         tmpy,alpha,beta)
    return
  end subroutine fft_backrc
  
  !***********************************************************************
  !                 +----------------------------+
  !                 |        SCALAR_SUM          |
  !                 +----------------------------+
  !***********************************************************************
  subroutine scalar_sum( &
       qfinit,rewcut,ewaldcof,volume,recip, &
       nfftdim1,nfftdim2,nfftdim3,eer,vir   &
#if KEY_MNDO97==1
      ,q_qmmm_pme_do    &
#endif
      )

  use pmeutil,only:mxzslabs,mxzstart, &
     nfft1,nfft2,nfft3,bsp_mod1,bsp_mod2,bsp_mod3

     !
     !-------------------------------------------------------------------
     !
     !  QFINIT   - Flag indicating a long range cutoff is to be applied
     !  REWCUT   - The long range cutoff diatance (only if QFINIT)
     !  Q        - The main charge grid
     !  EWALDCOF - The kappa value
     !  VOLUME   - The volume of the perodic system
     !  RECIP    - The inverse of the box length matrix
     !  BSP_MOD1 - The inverse of the B-spline coefficients (a direction)
     !  BSP_MOD2 - The inverse of the B-spline coefficients (b direction)
     !  BSP_MOD3 - The inverse of the B-spline coefficients (c direction)
     !  NFFT1    - The number of grid points (a direction)
     !  NFFT2    - The number of grid points (b direction)
     !  NFFT3    - The number of grid points (c direction)
     !  NFFTDIM1 - The dimension of grid points (a direction)
     !  NFFTDIM2 - The dimension of grid points (b direction)
     !  NFFTDIM3 - The dimension of grid points (c direction)
     !  EER      - K-space energy returned
     !  VIR      - K-space virial returned
     !
     !  Modified to add BLOCK code for energy decomposition
     !           Author:
     !                 Hiqmet Kamberaj 
     !                 November 2007
     !
     !-------------------------------------------------------------------
     !
     !   Finite distance cutoff code added by B. Brooks - NIH - 3/28/98
     !        ref: E.L.Pollock&J. Glosli, Computer Physics Communications,
     !        95 (1996) 93-110.
     !
  use number
  use consta
  use parallel
#if KEY_BLOCK==1
  use block_ltm ! block_fcm -> block_ltm -- Y Huang 2017
#endif
#if KEY_MNDO97==1
  use mndo97,only: lqmewd
  use qm1_info,only: qm_control_r,qm_main_r,mm_main_r
#endif 
     !...##INCLUDE '~/charmm_fcm/pme_par.f90'
     !
    LOGICAL,intent(in) :: QFINIT
    real(chm_real),intent(in) ::  REWCUT
    INTEGER,intent(in) :: NFFTDIM1,NFFTDIM2,NFFTDIM3
    !
    real(chm_real) :: EWALDCOF,VOLUME
    real(chm_real) :: EER,VIR(6)
    real(chm_real) :: RECIP(9)
    !
    real(chm_real) :: FAC,ETERM,VTERM,ENERGY
    real(chm_real) :: DEN1,DEN2,DEN3,DEN4
    INTEGER K,K0,K1,K2,K3,M1,M2,M3,IND,JND,INDTOP
    INTEGER NF1,NF2,NF3,K2Q
    INTEGER IPT1,IPT2,IPT3
    real(chm_real) :: MHAT1,MHAT2,MHAT3,MSQ,STRUC2
    real(chm_real) :: MHAT1A,MHAT1B,MHAT2A,MHAT2B,MHAT3A,MHAT3B
    real(chm_real) :: VCORR,MVAL,MCUT,ESTR,MSQR
    LOGICAL QFIN
    integer k1s,k2s,k3s,m1s,m2s,m3s       
    real(chm_real) :: MHAT1s,MHAT2s,MHAT3s,MSQs,STRUC2s,msqrs
    real(chm_real) :: dens,eterms,vterms,estrs
    integer i
#if KEY_MNDO97==1
    logical :: q_cpmd
    logical :: q_qmmm_pme_do
#endif
    !av_080628
#if KEY_BLOCK==1
    INTEGER                 :: IBL,JBL,IB,JB,KK,INXI,INXJ
    REAL(kind=chm_real)     :: COEF

    !...##IF PARALLEL
    !      NTOT=2*NFFTDIM1*NFFTDIM2*MXZSLABS
    !...##ELSE
    !      NTOT=2*NFFTDIM1*NFFTDIM2*NFFTDIM3
    !...##ENDIF
#endif /*  close BLOCK*/
    !av_080628
#if KEY_MNDO97==1
    q_cpmd = .false.
    if(q_qmmm_pme_do) q_cpmd = qm_control_r%q_do_cpmd_pme
#endif
    FAC = PI**2/EWALDCOF**2
    MCUT= TWO*PI*REWCUT
    QFIN=QFINIT
    !
    NF1 = NFFT1/2
    IF(2*NF1 < NFFT1) NF1 = NF1+1
    NF2 = NFFT2/2
    IF(2*NF2 < NFFT2) NF2 = NF2+1
    NF3 = NFFT3/2
    IF(2*NF3 < NFFT3) NF3 = NF3+1
    !
    ENERGY = ZERO
    DO K = 1,6
       VIR(K) = ZERO
    ENDDO
    !
    DEN1 = ONE/PI/VOLUME
    !
    IPT1=1
    !...##IF PARALLEL (pll0)
#if KEY_PMEPLSMA==1
#if KEY_PARALLEL==1
    if(mynod == 0)then     
#endif
       qarray(1:2)=zero
#if KEY_MNDO97==1
       if(q_qmmm_pme_do) then
          if(qm_control_r%q_do_cpmd_pme) qarray_mm(1:2) = zero
       end if
#endif
#if KEY_PARALLEL==1
    endif                  
#endif
#endif 
#if KEY_BLOCK==1 /*block*/
    if (qblock) then
       DO K2Q = 1, MXZSLABS
          K2=K2Q
#if KEY_PARALLEL==1
          IF(MYNOD > 0) K2 = K2Q + MXZSTART(MYNOD)     
#endif
          M2 = K2 - 1
          IF(K2 > NF2) M2 = M2 - NFFT2
          DEN2 = DEN1*BSP_MOD2(K2)
          MHAT1A = RECIP(4)*M2
          MHAT2A = RECIP(5)*M2
          MHAT3A = RECIP(6)*M2
          IPT2=IPT1
          k2s=mod(nfft2-k2+1,nfft2)+1
          DO K1 = 1, NF1+1
             k1s=nfft1-k1+2
             M1 = K1 - 1
             IF(K1 > NF1) M1 = M1 - NFFT1
             DEN3 = DEN2*BSP_MOD1(K1)
             MHAT1B = MHAT1A+RECIP(1)*M1
             MHAT2B = MHAT2A+RECIP(2)*M1
             MHAT3B = MHAT3A+RECIP(3)*M1
             IPT3=IPT2
             IF(K1+K2 == 2) THEN
                K0=2
                IPT3=IPT3+2
             ELSE
                K0=1
             ENDIF
             DO K3 = K0,NFFT3
                k3s=mod(nfft3-k3+1,nfft3)+1 
                M3 = K3 - 1
                IF(K3 > NF3) M3 = M3 - NFFT3
                DEN4 = DEN3*BSP_MOD3(K3)
                MHAT1 = MHAT1B+RECIP(7)*M3
                MHAT2 = MHAT2B+RECIP(8)*M3
                MHAT3 = MHAT3B+RECIP(9)*M3
                MSQ = MHAT1*MHAT1+MHAT2*MHAT2+MHAT3*MHAT3
                MSQR=ONE/MSQ
                !
                ETERM = EXP(-FAC*MSQ)*DEN4*MSQR
                VTERM = TWO*(FAC+MSQR)
                !av_080628
                ! Energy decomposition: H Kamberaj (November 2007)
                STRUC2 = ZERO
                DO IB = 1,NBLOCK
                   INXI = (IB-1)*NTOT + IPT3
                   DO JB = 1, NBLOCK
                      JBL = JB
                      IBL = IB
                      IF (jbl  <  ibl) then
                         kk = ibl
                         ibl = jbl
                         jbl = kk
                      endif
                      KK = IBL+JBL*(JBL-1)/2
                      COEF = BLCOEE(KK)
                      INXJ = (JB-1)*NTOT + IPT3
                      STRUC2 = STRUC2 + COEF * ( &
                           qarray(INXI)*qarray(INXJ) + &
                           qarray(INXI+1)*qarray(INXJ+1) &
                           )
                   ENDDO
                ENDDO
                !av_080628
#if KEY_DEBUG==1
                IF(.NOT.QFINIT) THEN
                   IF(ABS(eterm * struc2) > 5.0D-9) THEN
                      write(22,834) K1,K2,K3,MHAT1,MHAT2,MHAT3,eterm,struc2
834                   FORMAT(3I4,5F16.10)
                   ENDIF
                ENDIF
#endif 
                !
                IF(QFIN) THEN
                   MVAL=MCUT*SQRT(MSQ)
                   VCORR=STRUC2*ETERM*SIN(MVAL)*MVAL*MSQR
                   ETERM=ETERM*(ONE-COS(MVAL))
                   VIR(1) = VIR(1) -VCORR*MHAT1*MHAT1
                   VIR(2) = VIR(2) -VCORR*MHAT1*MHAT2
                   VIR(3) = VIR(3) -VCORR*MHAT1*MHAT3
                   VIR(4) = VIR(4) -VCORR*MHAT2*MHAT2
                   VIR(5) = VIR(5) -VCORR*MHAT2*MHAT3
                   VIR(6) = VIR(6) -VCORR*MHAT3*MHAT3
                ENDIF
                !
                !
                ESTR = ETERM * STRUC2
                ENERGY = ENERGY + ESTR
                VIR(1) = VIR(1) + ESTR * (VTERM*MHAT1*MHAT1 - ONE)
                VIR(2) = VIR(2) + ESTR * (VTERM*MHAT1*MHAT2)
                VIR(3) = VIR(3) + ESTR * (VTERM*MHAT1*MHAT3)
                VIR(4) = VIR(4) + ESTR * (VTERM*MHAT2*MHAT2 - ONE)
                VIR(5) = VIR(5) + ESTR * (VTERM*MHAT2*MHAT3)
                VIR(6) = VIR(6) + ESTR * (VTERM*MHAT3*MHAT3 - ONE)
                !--------------------------------------------------------------------------
                !                  if(k1 < 0)then
                if(k1 > 1)then
                   DENs = DEN1*BSP_MOD3(K3s) &
                        *BSP_MOD2(K2s)*BSP_MOD1(K1s)
                   M1s = K1s - 1
                   IF(K1s > NF1) M1s = M1s - NFFT1
                   M2s = K2s - 1
                   IF(K2s > NF2) M2s = M2s - NFFT2
                   M3s = K3s - 1
                   IF(K3s > NF3) M3s = M3s - NFFT3
                   
                   MHAT1s =RECIP(1)*M1s+RECIP(4)*M2s+RECIP(7)*M3s
                   MHAT2s =RECIP(2)*M1s+RECIP(5)*M2s+RECIP(8)*M3s
                   MHAT3s =RECIP(3)*M1s+RECIP(6)*M2s+RECIP(9)*M3s
                   MSQs = MHAT1s*MHAT1s+MHAT2s*MHAT2s+MHAT3s*MHAT3s
                   MSQRs=ONE/MSQs
                   !
                   ETERMs = EXP(-FAC*MSQs)*DENs*MSQRs
                   VTERMs = TWO*(FAC+MSQRs)
                   !
                   !av_080628
                   ! Energy decomposition: H Kamberaj (November 2007)
                   STRUC2s = ZERO
                   DO IB = 1,NBLOCK
                      INXI = (IB-1)*NTOT + IPT3
                      DO JB = 1,NBLOCK
                         JBL=JB
                         IBL=IB
                         IF (JBL  <  IBL) THEN
                            KK = IBL
                            IBL = JBL
                            JBL = KK
                         ENDIF
                         KK = IBL + JBL*(JBL-1)/2
                         INXJ = (JB-1)*NTOT + IPT3
                         COEF = BLCOEE(KK)
                         STRUC2s = STRUC2s + COEF * ( &
                              Qarray(INXI)*Qarray(INXJ) + &
                              Qarray(INXI+1)*Qarray(INXJ+1) &
                              )
                      ENDDO
                   ENDDO
                   !av_080628
                   !
                   ESTRs = ETERMs * STRUC2s
                   ENERGY = ENERGY + ESTRs
                   VIR(1) = VIR(1) + &
                        ESTRs * (VTERMs*MHAT1s*MHAT1s - ONE)
                   VIR(2) = VIR(2) + &
                        ESTRs * (VTERMs*MHAT1s*MHAT2s)
                   VIR(3) = VIR(3) + &
                        ESTRs * (VTERMs*MHAT1s*MHAT3s)
                   VIR(4) = VIR(4) + &
                        ESTRs * (VTERMs*MHAT2s*MHAT2s - ONE)
                   VIR(5) = VIR(5) + &
                        ESTRs * (VTERMs*MHAT2s*MHAT3s)
                   VIR(6) = VIR(6) + &
                        ESTRs * (VTERMs*MHAT3s*MHAT3s - ONE)
                endif
                !--------------------------------------------------------------------------
                !
                !av_080628 Block code: H Kamberaj (November 2007)
                DO IB=1, NBLOCK
                   INXI = (IB-1)*NTOT + IPT3
                   Qarray(INXI)   = ETERM * Qarray(INXI)
                   Qarray(INXI+1) = ETERM * Qarray(INXI+1)
                ENDDO
                !
                IPT3=IPT3+2
             ENDDO
             IPT2=IPT2+NFFT3*2
          ENDDO
          IPT1=IPT1+NFFT3*NFFTDIM1*2
       ENDDO
    else
#endif /* (block)*/
       !
       ! No BLOCK keyword or qblock = .false.
       !
       DO K2Q = 1, MXZSLABS
          K2=K2Q
#if KEY_PARALLEL==1
          IF(MYNOD > 0) K2 = K2Q + MXZSTART(MYNOD)     
#endif
          M2 = K2 - 1
          IF(K2 > NF2) M2 = M2 - NFFT2
          DEN2 = DEN1*BSP_MOD2(K2)
          MHAT1A = RECIP(4)*M2
          MHAT2A = RECIP(5)*M2
          MHAT3A = RECIP(6)*M2
          IPT2=IPT1
          k2s=mod(nfft2-k2+1,nfft2)+1
          DO K1 = 1, NF1+1
             k1s=nfft1-k1+2
             M1 = K1 - 1
             IF(K1 > NF1) M1 = M1 - NFFT1
             DEN3 = DEN2*BSP_MOD1(K1)
             MHAT1B = MHAT1A+RECIP(1)*M1
             MHAT2B = MHAT2A+RECIP(2)*M1
             MHAT3B = MHAT3A+RECIP(3)*M1
             IPT3=IPT2
             IF(K1+K2 == 2) THEN
                K0=2
                IPT3=IPT3+2
             ELSE
                K0=1
             ENDIF
             DO K3 = K0,NFFT3
                k3s=mod(nfft3-k3+1,nfft3)+1 
                M3 = K3 - 1
                IF(K3 > NF3) M3 = M3 - NFFT3
                DEN4 = DEN3*BSP_MOD3(K3)
                MHAT1 = MHAT1B+RECIP(7)*M3
                MHAT2 = MHAT2B+RECIP(8)*M3
                MHAT3 = MHAT3B+RECIP(9)*M3
                MSQ = MHAT1*MHAT1+MHAT2*MHAT2+MHAT3*MHAT3
                MSQR=ONE/MSQ
                !
                ETERM = EXP(-FAC*MSQ)*DEN4*MSQR
                VTERM = TWO*(FAC+MSQR)
                !av_080628
                ! Energy decomposition: H Kamberaj (November 2007)
                STRUC2 = qarray(IPT3)*qarray(IPT3) +  &
                     qarray(IPT3+1)*qarray(IPT3+1)
                !av_080628
#if KEY_DEBUG==1
                IF(.NOT.QFINIT) THEN
                   IF(ABS(eterm * struc2) > 5.0D-9) THEN
                      write(22,835) K1,K2,K3,MHAT1,MHAT2,MHAT3,eterm,struc2
835                   FORMAT(3I4,5F16.10)
                   ENDIF
                ENDIF
#endif 
                !
                IF(QFIN) THEN
                   MVAL=MCUT*SQRT(MSQ)
                   VCORR=STRUC2*ETERM*SIN(MVAL)*MVAL*MSQR
                   ETERM=ETERM*(ONE-COS(MVAL))
                   VIR(1) = VIR(1) -VCORR*MHAT1*MHAT1
                   VIR(2) = VIR(2) -VCORR*MHAT1*MHAT2
                   VIR(3) = VIR(3) -VCORR*MHAT1*MHAT3
                   VIR(4) = VIR(4) -VCORR*MHAT2*MHAT2
                   VIR(5) = VIR(5) -VCORR*MHAT2*MHAT3
                   VIR(6) = VIR(6) -VCORR*MHAT3*MHAT3
                ENDIF
                !
                !
                ESTR = ETERM * STRUC2
                ENERGY = ENERGY + ESTR
                VIR(1) = VIR(1) + ESTR * (VTERM*MHAT1*MHAT1 - ONE)
                VIR(2) = VIR(2) + ESTR * (VTERM*MHAT1*MHAT2)
                VIR(3) = VIR(3) + ESTR * (VTERM*MHAT1*MHAT3)
                VIR(4) = VIR(4) + ESTR * (VTERM*MHAT2*MHAT2 - ONE)
                VIR(5) = VIR(5) + ESTR * (VTERM*MHAT2*MHAT3)
                VIR(6) = VIR(6) + ESTR * (VTERM*MHAT3*MHAT3 - ONE)
                !--------------------------------------------------------------------------
                !                  if(k1 < 0)then
                if(k1 > 1)then
                   DENs = DEN1*BSP_MOD3(K3s) &
                        *BSP_MOD2(K2s)*BSP_MOD1(K1s)
                   M1s = K1s - 1
                   IF(K1s > NF1) M1s = M1s - NFFT1
                   M2s = K2s - 1
                   IF(K2s > NF2) M2s = M2s - NFFT2
                   M3s = K3s - 1
                   IF(K3s > NF3) M3s = M3s - NFFT3
                   
                   MHAT1s =RECIP(1)*M1s+RECIP(4)*M2s+RECIP(7)*M3s
                   MHAT2s =RECIP(2)*M1s+RECIP(5)*M2s+RECIP(8)*M3s
                   MHAT3s =RECIP(3)*M1s+RECIP(6)*M2s+RECIP(9)*M3s
                   MSQs = MHAT1s*MHAT1s+MHAT2s*MHAT2s+MHAT3s*MHAT3s
                   MSQRs=ONE/MSQs
                   !
                   ETERMs = EXP(-FAC*MSQs)*DENs*MSQRs
                   VTERMs = TWO*(FAC+MSQRs)
                   !
                   !av_080628
                   ! Energy decomposition: H Kamberaj (November 2007)
                   STRUC2s = Qarray(IPT3)*qarray(ipt3) +  &
                        Qarray(IPT3+1)*qarray(ipt3+1)
                   !av_080628
                   !
                   ESTRs = ETERMs * STRUC2s
                   ENERGY = ENERGY + ESTRs
                   VIR(1) = VIR(1) + &
                        ESTRs * (VTERMs*MHAT1s*MHAT1s - ONE)
                   VIR(2) = VIR(2) + &
                        ESTRs * (VTERMs*MHAT1s*MHAT2s)
                   VIR(3) = VIR(3) + &
                        ESTRs * (VTERMs*MHAT1s*MHAT3s)
                   VIR(4) = VIR(4) + &
                        ESTRs * (VTERMs*MHAT2s*MHAT2s - ONE)
                   VIR(5) = VIR(5) + &
                        ESTRs * (VTERMs*MHAT2s*MHAT3s)
                   VIR(6) = VIR(6) + &
                        ESTRs * (VTERMs*MHAT3s*MHAT3s - ONE)
                endif
                Qarray(IPT3)   = ETERM * Qarray(IPT3)
                Qarray(IPT3+1) = ETERM * Qarray(IPT3+1)

#if KEY_MNDO97==1
                ! this is for the qm-mm only component, whereas Qarray_local has all.
                if(q_cpmd) then
                   Qarray_mm(IPT3)   = ETERM * Qarray_mm(IPT3)
                   Qarray_mm(IPT3+1) = ETERM * Qarray_mm(IPT3+1)
                end if
#endif
                !
                IPT3=IPT3+2
             ENDDO
             IPT2=IPT2+NFFT3*2
          ENDDO
          IPT1=IPT1+NFFT3*NFFTDIM1*2
       ENDDO
#if KEY_BLOCK==1
    endif  
#endif
    !
    EER = HALF * ENERGY
    !
    DO K = 1,6
       VIR(K) = HALF*VIR(K)
    ENDDO
    RETURN
  END subroutine scalar_sum
  
  !***********************************************************************
  !                 +----------------------------+
  !                 |        FFT_FORWARD         |
  !                 +----------------------------+
  !***********************************************************************
  subroutine fft_forwardrc(array,nfftdim1,nfftdim2,nfftdim3, &
       nffwork)

  use pmeutil,only:fft3d_zxyrc
    !
    !
    real(chm_real) :: array(*),scale
    integer nfftdim1,nfftdim2,nfftdim3
    integer nffwork
    !
    integer isign,i
    !
    isign = 1

    scale = 1.d0
    call fft3d_zxyrc(isign,scale,array, &
         nfftdim1,nfftdim2,array,nfftdim1,nfftdim2, &
         tmpy,alpha,beta)

    !
    return
  end subroutine fft_forwardrc

  
  !***********************************************************************
  !                 +----------------------------+
  !                 |        GRAD_SUM            |
  !                 +----------------------------+
  !
  ! Modified to add block code of energy decomposition
  ! Author:         
  !       Hiqmet Kamberaj
  !       November 2007
  !
  !***********************************************************************
  subroutine grad_sumrc( &
       igood, kbot, ktop, &
       numatoms,charge,recip, &
       fx,fy,fz, &
#if KEY_CHEQ==1
       dch,qcg,                & 
#endif
       ordr, &
       nfftdim1,nfftdim2,nfftdim3, &
       my_ks,latm,xnsymm)

  use pmeutil,only: mxystart,mxyslabs,mxzslabs, &
     nfft1,nfft2,nfft3, &
     theta1,theta2,theta3,dtheta1,dtheta2,dtheta3

  use number
  use dimens_fcm
  use consta
  use parallel
  use stream
#if KEY_BLOCK==1
  use block_ltm ! block_fcm -> block_ltm -- Y Huang 2017
#endif 
     !

    integer,intent(in) :: numatoms,ordr
    integer,intent(in) :: nfftdim1,nfftdim2,nfftdim3,xnsymm
    real(chm_real),intent(in) :: recip(9),charge(*)
    real(chm_real) :: FX(*),FY(*),FZ(*)
    integer,intent(in) :: latm,my_ks(latm)
    !
#if KEY_CHEQ==1
    real(chm_real) :: DCH(*),HIJ
    logical qcg
#endif 
    !...##INCLUDE '~/charmm_fcm/pme_par.f90'
    !
    !av_080628
    ! H Kamberaj (November 2007)
#if KEY_BLOCK==1
    INTEGER JB,IB,IBL,JBL,KK,I0
    real(chm_real)  COEF
#endif /* close BLOCK*/
    !av_080628

    integer igoo,ig
    real(chm_real) :: VAL1,VAL2,VAL3,VAL1A,VAL2A,VAL3A
    real(kind=chm_real) :: cg, cgs
    INTEGER IPT1,IPT2,IPT3
    !
    INTEGER KQ
    integer igood, kbot, ktop
    !
    INTEGER N,ITH1,ITH2,ITH3,I,J,K
    real(chm_real) :: F1,F2,F3,TERM,CFACT
    integer nfftdimrc
    if(nfft1/2 /= (nfft1+1)/2)then
       CALL WRNDIE(-5,'<PME fill charge grid>' &
            ,'fftx dimension not even ')
    endif
    nfftdimrc=nfft1+4
    !
    CFACT=CCELEC/XNSYMM
    !
    do ig = 1,igood
       n=my_ks(ig)
       if(xnsymm == 1)then
          igoo=ig
       else
          igoo=n
       endif
       !av_080628
       ! Block code: H Kamberaj (November 2007)
#if KEY_BLOCK==1
       IF (QBLOCK) THEN
          IF (N  >  NUMATOMS) &
               CALL WRNDIE(-5, '<GRAD-PME>', &
               'N GREATER THE NATOMS')
          IB = IBLCKP(N)
       ENDIF
#endif /*  close BLOCK*/
       !av_080628
       F1 = ZERO
       F2 = ZERO
       F3 = ZERO
#if KEY_CHEQ==1
       HIJ= ZERO
#endif 
       K = INT(FR3(igoo)) - ORDR + 1 + NFFT3
       !

#if KEY_BLOCK==1 /*block*/
       if (qblock) then
          DO ITH3 = 1,ORDR
             K=K+1
             IF(K > NFFT3) K=K-NFFT3
             KQ=K
#if KEY_PARALLEL==1
             IF ( K  >=  KBOT .AND. K  <=  KTOP )THEN
                KQ = K - MXYSTART(MYNOD)
#endif 
                VAL1A = CHARGE(N) * NFFT1 * THETA3(ITH3,ig)
                VAL2A = CHARGE(N) * NFFT2 * THETA3(ITH3,ig)
                VAL3A = CHARGE(N) * NFFT3 * DTHETA3(ITH3,igoo)
                !
                J = INT(FR2(igoo)) - ORDR + 1 + NFFT2
                IPT1=(KQ-1)*NFFTDIM2 -1
                !
                I = INT(FR1(igoo)) - ORDR + 1 + NFFT1
                IF(I >= NFFT1) I=I-NFFT1
                !
                DO ITH2 = 1,ORDR
                   J=J+1
                   IF(J > NFFT2) J=J-NFFT2
                   !
                   VAL1= VAL1A * THETA2(ITH2,ig)
                   VAL2= VAL2A * DTHETA2(ITH2,igoo)
                   VAL3= VAL3A * THETA2(ITH2,ig)
                   
                   IPT2= ((IPT1+J)*NFFTDIMrc+I)+1
                   IPT3= IPT2 + (NFFT1-I)
                   !
                   DO ITH1 = 1,ORDR
                      ! force is negative of grad
                      ! Block code: H Kamberaj (November 2007)
                      CG = ZERO
                      CGS= ZERO
                      DO JB=1, NBLOCK
                         IBL = IB
                         JBL = JB
                         IF (JBL  <  IBL) THEN
                            KK  = IBL
                            IBL = JBL
                            JBL = KK
                         ENDIF
                         KK = IBL + JBL*(JBL-1)/2
                         COEF = BLCOEE(KK)
                         I0 = (JB-1)*NTOT
                         IF ( (I0+IPT2)  >  (NTOT*NBLOCK) ) &
                              CALL WRNDIE(-5,'<PME-GRAD>', &
                              'I0 GREATER THAN MAX')
                         IF (IPT2  >  NTOT) &
                              CALL WRNDIE(-5,'<PME-GRAD>', &
                              'IPT2 GREATER THAN NTOT')

                         CGS = CGS + COEF * Qarray(I0+IPT2)
                         !
#if KEY_CHEQ==1
                         CG = CG + Qarray(I0+IPT2)
#endif 
                         !
                      ENDDO
                      F1 = F1 - VAL1 * cgs * DTHETA1(ITH1,igoo)
                      F2 = F2 - VAL2 * cgs * THETA1(ITH1,ig)
                      F3 = F3 - VAL3 * cgs * THETA1(ITH1,ig)
#if KEY_CHEQ==1
                      HIJ=HIJ+THETA1(ITH1,ig)*THETA2(ITH2,ig)* &
                           THETA3(ITH3,ig)*cg
#endif 
                      !
                      IPT2=IPT2+1
                      IF(IPT2 >= IPT3) IPT2=IPT2-NFFT1
                   ENDDO
                ENDDO
#if KEY_PARALLEL==1
             ENDIF
#endif 
          ENDDO
       else
#endif /* (block)*/
          !
          ! No BLOCK keyword defined or qblock = .false.
          !
          DO ITH3 = 1,ORDR
             K=K+1
             IF(K > NFFT3) K=K-NFFT3
             KQ=K
#if KEY_PARALLEL==1
             IF ( K  >=  KBOT .AND. K  <=  KTOP )THEN
                KQ = K - MXYSTART(MYNOD)
#endif 
                VAL1A = CHARGE(N) * NFFT1 * THETA3(ITH3,ig)
                VAL2A = CHARGE(N) * NFFT2 * THETA3(ITH3,ig)
                VAL3A = CHARGE(N) * NFFT3 * DTHETA3(ITH3,igoo)
                !
                J = INT(FR2(igoo)) - ORDR + 1 + NFFT2
                IPT1=(KQ-1)*NFFTDIM2 -1
                !
                I = INT(FR1(igoo)) - ORDR + 1 + NFFT1
                IF(I >= NFFT1) I=I-NFFT1
                !
                DO ITH2 = 1,ORDR
                   J=J+1
                   IF(J > NFFT2) J=J-NFFT2
                   !
                   VAL1= VAL1A * THETA2(ITH2,ig)
                   VAL2= VAL2A * DTHETA2(ITH2,igoo)
                   VAL3= VAL3A * THETA2(ITH2,ig)
                   
                   IPT2= ((IPT1+J)*NFFTDIMrc+I)+1
                   IPT3= IPT2 + (NFFT1-I)
                   !
                   DO ITH1 = 1,ORDR
                      CGS= Qarray(IPT2)
                      CG = CGS
                      F1 = F1 - VAL1 * cgs * DTHETA1(ITH1,igoo)
                      F2 = F2 - VAL2 * cgs * THETA1(ITH1,ig)
                      F3 = F3 - VAL3 * cgs * THETA1(ITH1,ig)
#if KEY_CHEQ==1
                      HIJ=HIJ+THETA1(ITH1,ig)*THETA2(ITH2,ig)* &
                           THETA3(ITH3,ig)*cg
#endif 
                      !
                      IPT2=IPT2+1
                      IF(IPT2 >= IPT3) IPT2=IPT2-NFFT1
                   ENDDO
                ENDDO
#if KEY_PARALLEL==1
             ENDIF
#endif 
          ENDDO
#if KEY_BLOCK==1
       endif  
#endif
       !
       FX(N) = FX(N) - CFACT*(RECIP(1)*F1+RECIP(4)*F2+RECIP(7)*F3)
       FY(N) = FY(N) - CFACT*(RECIP(2)*F1+RECIP(5)*F2+RECIP(8)*F3)
       FZ(N) = FZ(N) - CFACT*(RECIP(3)*F1+RECIP(6)*F2+RECIP(9)*F3)
#if KEY_CHEQ==1
       IF (CHARGE(N) /= 0.0 .and. qcg) THEN
          DCH(N)= DCH(N)+CCELEC*HIJ
       ENDIF
#endif 
    ENDDO
    RETURN
  END subroutine grad_sumrc

  !***********************************************************************
  !                 +----------------------------+
  !                 |        GRAD_SUMRCH         |
  !                 +----------------------------+
  !.ab. This is a copy of GRAD_SUMRCH with HybH scheme.
  !.ab. Note that this allow the calculation of dewald/dlambda
  !.ab. for charges=(chageReactant*(1-lambda)+ChargeProduit*lambda)
  !.ab. for use in Thermodynamics integrations. We use both original
  !.ab. charges (unscaled) and scaled ones.
  !***********************************************************************
#if KEY_BLOCK==1 /*block*/
  subroutine grad_sumrch( &
       igood, kbot, ktop, &
       numatoms,REACHG,charge,recip, &
       fx,fy,fz, &
#if KEY_CHEQ==1
       dch, qcg,               & 
#endif
       order, &
       nfftdim1,nfftdim2,nfftdim3, &
       my_ks,latm,xnsymm)

  use pmeutil,only: mxystart,mxyslabs,nfft1,nfft2,nfft3, &
     theta1,theta2,theta3,dtheta1,dtheta2,dtheta3

  use number
  use dimens_fcm
  use consta
  use parallel
     !...##INCLUDE '~/charmm_fcm/pme_par.f90'
     !.ab.
  use block_ltm  ! block_fcm -> block_ltm -- Y Huang 2017
  use energym,only:ETERMR,ETERMP ! sumhyb -- Y Huang 2017
     !

    integer,intent(in) :: numatoms,order
    integer,intent(in) :: nfftdim1,nfftdim2,nfftdim3,xnsymm
    real(chm_real),intent(in) :: recip(9),charge(*)
    !.ab.
    real(chm_real),intent(in) :: REACHG(*)
    !.ab.
    real(chm_real) :: FX(*),FY(*),FZ(*)
    integer,intent(in) :: latm,my_ks(latm)
    !
#if KEY_CHEQ==1
    real(chm_real) :: DCH(*),HIJ
    logical qcg
#endif 
    !...  use exfunc
    !.ab.Can't include exfunc.f90 because conflict name with order...
    real(chm_real) :: DPRD,DEW(3)
    INTEGER REATOM,NUMA,IBL
    !.ab.
    integer igoo,ig
    real(chm_real) :: VAL1,VAL2,VAL3,VAL1A,VAL2A,VAL3A
    INTEGER IPT1,IPT2,IPT3
    !
    INTEGER KQ
    integer igood, kbot, ktop
    !
    INTEGER N,ITH1,ITH2,ITH3,I,J,K
    real(chm_real) :: F1,F2,F3,TERM,CFACT
    integer rcskip,nfftdimrc
    rcskip=1
    if(nfft1/2 /= (nfft1+1)/2)then
       CALL WRNDIE(-5,'<PME fill charge grid>' &
            ,'fftx dimension not even ')
    endif
    nfftdimrc=nfft1+4
    !

    !
    CFACT=CCELEC/XNSYMM
    !.ab.Initialise INHYBH -> les coefs. Charges CHF(1..3)
    IF (.NOT.QHYBH) CALL WRNDIE(-5,'<GRAD_SUMB>', &
         'QHYBH should be true to call this routine')
    DEW(1)=.0
    DEW(2)=.0
    DEW(3)=.0
    REATOM=NUMATOMS/XNSYMM
    !.ab.
    !
    do ig = 1,igood
       n=my_ks(ig)
       if(xnsymm == 1)then
          igoo=ig
       else
          igoo=n
       endif
       !.ab. Get block here -> coef dcoef.
       !         NUMA=MOD(N-1,REATOM)+1
       !         IBL=IBLCKP(NUMA)
       IBL = IBLCKP(N)
       !.ab. for EWQCOR = 47 only one dummy atom.
       !.ab. Use prod (ibl=3) as dE/dl=(dcp/dl-dcr/dl)*c*fact (product)
       IF(IHYBH == 47) IBL=3
       !.ab.
       F1 = ZERO
       F2 = ZERO
       F3 = ZERO
#if KEY_CHEQ==1
       HIJ= ZERO
#endif 
       K = INT(FR3(igoo)) - ORDER + 1 + NFFT3
       !
       DO ITH3 = 1,ORDER
          K=K+1
          IF(K > NFFT3) K=K-NFFT3
          KQ=K
#if KEY_PARALLEL==1
          IF ( K  >=  KBOT .AND. K  <=  KTOP )THEN
             KQ = K - MXYSTART(MYNOD)
#endif 
             VAL1A = CHARGE(N) * NFFT1 * THETA3(ITH3,ig)
             VAL2A = CHARGE(N) * NFFT2 * THETA3(ITH3,ig)
             VAL3A = CHARGE(N) * NFFT3 * DTHETA3(ITH3,igoo)
             !
             J = INT(FR2(igoo)) - ORDER + 1 + NFFT2
             IPT1=(KQ-1)*NFFTDIM2 -1
             !
             I = INT(FR1(igoo)) - ORDER + 1 + NFFT1
             IF(I >= NFFT1) I=I-NFFT1
             !
             DO ITH2 = 1,ORDER
                J=J+1
                IF(J > NFFT2) J=J-NFFT2
                !
                VAL1= VAL1A * THETA2(ITH2,ig)
                VAL2= VAL2A * DTHETA2(ITH2,igoo)
                VAL3= VAL3A * THETA2(ITH2,ig)
                !.ab. Original charge are used. Linear coupling for elec... dQ/dl=+/-Q
                !.ab. sign given at the end...
                !.ab. Note if(ihybh=47)charge(1)<-reachg(1)but not 2,3.
                IF (IBL /= 1) &
                     DPRD=REACHG(N)*THETA2(ITH2,IG)*THETA3(ITH3,IG)
                !a.b.
                IPT2= rcskip*((IPT1+J)*NFFTDIMrc+I)+1
                IPT3= IPT2 + rcskip*(NFFT1-I)
                !
                DO ITH1 = 1,ORDER
                   !                                   force is negative of grad
                   F1 = F1 - VAL1 * Qarray(IPT2) * DTHETA1(ITH1,igoo)
                   F2 = F2 - VAL2 * Qarray(IPT2) * THETA1(ITH1,ig)
                   F3 = F3 - VAL3 * Qarray(IPT2) * THETA1(ITH1,ig)
                   !.ab.dEwald/dL.
                   IF (IBL /= 1) DEW(IBL)=DEW(IBL) &
                        +Qarray(IPT2)*DPRD*THETA1(ITH1,IG)
                   !.ab.
#if KEY_CHEQ==1
                   HIJ=HIJ+THETA1(ITH1,ig)*THETA2(ITH2,ig)* &
                        THETA3(ITH3,ig)*Qarray(IPT2)
#endif 
                   !
                   IPT2=IPT2+rcskip
                   IF(IPT2 >= IPT3) IPT2=IPT2-NFFT1*rcskip
                ENDDO
             ENDDO
#if KEY_PARALLEL==1
          ENDIF
#endif 
       ENDDO
       !
       FX(N) = FX(N) - CFACT*(RECIP(1)*F1+RECIP(4)*F2+RECIP(7)*F3)
       FY(N) = FY(N) - CFACT*(RECIP(2)*F1+RECIP(5)*F2+RECIP(8)*F3)
       FZ(N) = FZ(N) - CFACT*(RECIP(3)*F1+RECIP(6)*F2+RECIP(9)*F3)
#if KEY_CHEQ==1
       IF (CHARGE(N) /= 0.0 .and. qcg) THEN
          DCH(N)= DCH(N)+CCELEC*HIJ
       ENDIF
#endif 
    ENDDO
    !.ab.Wrapup dEwald/dlambda.
    !      if(numatoms == 1)write(6,*) NUMATOMS,NUMA,IBL,DEW(2),DEW(3)
    IF(IHYBH == 47) THEN
       !.ab.see Note above
       IF(REACHG(1) /= 0.)THEN
          !            write(6,*) 'RC: ',reachg(1),reachg(2),reachg(3)
          !            write(6,*) 'CC: ',charge(1),charge(2),charge(3)
          !.ab. the sign of eqcor is inverted latter in code. Invert derivatives.
          DEW(2)=-DEW(3)*REACHG(2)/REACHG(1)
          DEW(3)= DEW(3)*REACHG(3)/REACHG(1)
       ELSE
          DEW(3)= 0.
       ENDIF
    ELSE
       DEW(2)=-DEW(2)*CFACT/XNSYMM
       DEW(3)=DEW(3)*CFACT/XNSYMM
    ENDIF
    !.ab.I am so currious !
    !      if (xnsymm /= 1) write(6,*) 'XNSYMM: ',XNSYMM
    !.ab. Arbitrary distribution on inter block terms.
!------------------------------------------------------------------
!      No block_fcm, No symhyb -- Y Huang 2017
!          call sumhyb(ihybh,dew(2),dew(3))
          ETERMR(ihybh)=ETERMR(ihybh)+dew(2)
          ETERMP(ihybh)=ETERMP(ihybh)+dew(3)
!------------------------------------------------------------------
    !
    RETURN
  END subroutine grad_sumrch

  
  SUBROUTINE DO_PME_COR(NATOM,CURCG,CURCHG,ORGCHG,FACT,CGB,QEUTIL)
    !.ab. calculate Background correction.
    !.ab. CURCHG are supposed to have been scaled.
  use chm_kinds
  use block_ltm  ! block_fcm -> block_ltm -- Y Huang 2017
  use energym,only:ETERMR,ETERMP ! sumhyb -- Y Huang 2017
    !
    implicit none
    INTEGER NATOM
    real(chm_real)  CURCG,CURCHG(*),ORGCHG(*),FACT, CGB(*)
    LOGICAL QEUTIL
    !
    INTEGER I,IBL
    !
    CURCG=0.
    CGB(1)=0.
    CGB(2)=0.
    CGB(3)=0.
    DO I=1,NATOM
       CURCG=CURCG+CURCHG(I)
       IBL = IBLCKP(I)
       CGB(IBL)=CGB(IBL)+ORGCHG(I)
    ENDDO
    !      CGB(2)=-CGB(2)*CURCG*FACT
    !      CGB(3)= CGB(3)*CURCG*FACT
    !      IF(QEUTIL)CALL SUMHYB(IHYBH,CGB(2),CGB(3))
!-----------------------------------------------------------------------------
!      No block_fcm, No sumhyb -- Y Huang 2017
!    IF(QEUTIL)CALL SUMHYB(IHYBH,-CGB(2)*CURCG*FACT,CGB(3)*CURCG*FACT)
     IF(QEUTIL) THEN
        ETERMR(IHYBH)=ETERMR(IHYBH)-CGB(2)*CURCG*FACT
        ETERMP(IHYBH)=ETERMP(IHYBH)+CGB(3)*CURCG*FACT
     ENDIF
!-----------------------------------------------------------------------------
    !      write(6,*) 'BgCor: ',curcg,fact,CGB(2),CGB(3)
    !
    RETURN
  END SUBROUTINE DO_PME_COR

#endif /* (block)*/

#if KEY_MNDO97==1 /*mndo97*/
  !****************************************************************
  !                   QM/MM-PME (MNDO97) specific
  SUBROUTINE fill_ch_grid_qm_mm(kbot, ktop, numat, scf_mchg_2, &
                                x,y,z,recip,natom,xnsymm, &
                                nfftdim1,nfftdim2,nfftdim3, &
                                my_ks_inv,latm,qminb_local,q_grad_and_pot,q_cpmd)

  use pmeutil,only:nfft1,nfft2,nfft3,forder, &
                   mxystart,mxyslabs,  &  !##PARALLEL
                   theta1,theta2,theta3, &
                   dtheta1,dtheta2,dtheta3,fill_bspline
  !
  ! This routine fills the charge grid, Q for each qm atom. This will
  ! be called after calling fill_ch_grid.

  !
  ! I have to only assume xnsymm is 1. Other cases does not support yet.
  !

  !
  !---------------------------------------------------------------------
  ! INPUT:
  !      numat:  number of qm atoms
  !      nfft1,nfft2,nfft3: the charge grid dimensions
  !      nfftdim1,nfftdim2,nfftdim3: physical charge grid dims
  ! OUTPUT:
  !
  ! This routine fills the charge grid, Q.
  ! C.ab. scales charges (CG1) according to HybH scheme...
  !
  !---------------------------------------------------------------------
  ! INPUT:
  !      theta1,theta2,theta3: the spline coeff arrays
  !      fr1,fr2,fr3 the scaled and shifted fractional coords
  !      nfft1,nfft2,nfft3: the charge grid dimensions
  !      nfftdim1,nfftdim2,nfftdim3: physical charge grid dims
  ! OUTPUT:
  !      Q the charge grid
  !
  !---------------------------------------------------------------------
  !
  use number
  use parallel
  use dimens_fcm

  implicit none

  integer :: natom,numat
  integer :: nfftdim1,nfftdim2,nfftdim3
  real(chm_real) :: x(*),y(*),z(*), recip(9), scf_mchg_2(numat)
  integer :: qminb_local(*)
  integer :: latm,my_ks_inv(*)
  integer :: kbot, ktop, xnsymm
  logical :: q_grad_and_pot,q_cpmd

  real(chm_real) :: prod,proda
  real(chm_real) :: fr1n,fr2n,fr3n,w

  integer :: n,ith1,ith2,ith3,i,j,k,kq,ipt1,ipt2,ipt3,ipt,dim_q0,iqm
  integer :: enumtasks,itask,kdel,kbot0
  integer :: igood
  integer :: rcskip,nfftdimrc         ! rcfft addition

  rcskip=1
  if(nfft1/2 /= (nfft1+1)/2) call wrndie(-5,'<fill_ch_grid_qm_qm>','fftx dimension not even ')
  nfftdimrc=nfft1+4
  !
  igood=0
#if KEY_PARALLEL==1
  enumtasks = 1
  kdel = nfft3/enumtasks
  if ( kdel  ==  0 )kdel = 1
  !
  kbot0 = mxystart(mynod)
  kbot = kbot0 + 1
  ktop = kbot0 + mxyslabs
#else
  kbot0 = 0
  kbot = 1
  ktop = nfft3
#endif

  ! allocate arrays:
  if(q_grad_and_pot) then
     if(allocated(i_qarray_mm) .and. size(i_qarray_mm).ne.3*forder*numat) deallocate(i_qarray_mm)
     if(allocated(r_qarray_mm) .and. size(r_qarray_mm).ne.3*forder*numat) deallocate(r_qarray_mm)
     if(.not.allocated(i_qarray_mm)) allocate(i_qarray_mm(forder,forder,forder,numat))
     if(.not.allocated(r_qarray_mm)) allocate(r_qarray_mm(forder,forder,forder,numat))
  end if

  ! Initialization...
  dim_q0 = 2*nfftdim1*nfftdim2*nfftdim3
  if(q_grad_and_pot) then
     if(.not.q_cpmd) then
        i_qarray_mm        = 0                   ! i have to allocate this array first somewhere
        r_qarray_mm        = zero                ! same
     end if
  else
     qarray(1:dim_q0) = zero
  end if

  !------------------------------------------
  !          MFC NOTE: THERE COULD BE A PRE-FILTER HERE TO ELIMINATE
  !                      MOST OF THE ATOMS AND KEEP A LIST OF APPROPRIATE
  !                      ATOMS EITHER FOR EACH PE, EACH X-Y SLAB,
  !                      OR EACH Q ELEMENT
  !          MFC NOte Note: Looks like I am doing that filter now....
  !------------------------------------------
  if(q_grad_and_pot) then
     ! only loop over QM atoms, since it only filters qarray terms relevant for
     ! qm atoms.
     loopiqm: do iqm = 1,numat
        n=qminb_local(iqm)      ! pointer for the index in x/y/z arrays.
        igood=my_ks_inv(n)
        if(igood.le.0) cycle loopiqm

        w    = x(n)*recip(1)+y(n)*recip(2)+z(n)*recip(3)
        fr1n = nfft1*(w - anint(w) + half)
        w    = x(n)*recip(4)+y(n)*recip(5)+z(n)*recip(6)
        fr2n = nfft2*(w - anint(w) + half)
        w    = x(n)*recip(7)+y(n)*recip(8)+z(n)*recip(9)
        fr3n = nfft3*(w - anint(w) + half)

        k = int(fr3n) - forder + 1 + nfft3

        do ith3 = 1,forder
           k=k+1
           if(k > nfft3) k=k-nfft3
           kq=k
#if KEY_PARALLEL==1
           if ( k  >=  kbot .and. k  <=  ktop )then
              kq = k - kbot0
#endif
              proda = theta3(ith3,igood)                         ! not includ echarge here. charge(n)

              j = int(fr2n) - forder + 1 + nfft2
              ipt1 = (kq-1)*nfftdim2 - 1

              i = int(fr1n) - forder + 1 + nfft1
              if(i >= nfft1) i=i-nfft1

              do ith2 = 1,forder
                 j=j+1
                 if(j > nfft2) j=j-nfft2
                 prod = theta2(ith2,igood)*proda
                 ipt2= rcskip*((ipt1+j)*nfftdimrc+i)+1
                 ipt3= ipt2 + rcskip*(nfft1-i)

                 do ith1 = 1,forder
                    ! In the end, it will be something like:
                    ! Qarray(i_qarray_mm(i,j,k,iqm))= sum_iqm [r_qarray_mm(i,j,k,iqm)*qm_chareg(iqm)]
                    if(q_cpmd) then
                       !Qarray(ipt2)=Qarray(ipt2)+half*THETA1(ITH1,IGOOD)*PROD*scf_mchg_2(iqm)
                       qarray(ipt2)=qarray(ipt2)+THETA1(ITH1,IGOOD)*PROD*scf_mchg_2(iqm)
                    else
                       i_qarray_mm(ith1,ith2,ith3,iqm)=ipt2                     ! pointer for qarray_qm
                       r_qarray_mm(ith1,ith2,ith3,iqm)=THETA1(ITH1,IGOOD)*PROD  ! value for qarray_qm, contribution
                                                                                ! from iqm atom.
                    end if

                    ipt2=ipt2+rcskip
                    if(ipt2 >= ipt3) ipt2=ipt2-nfft1*rcskip
                 end do
              end do
#if KEY_PARALLEL==1
           end if
#endif
        end do
     end do loopiqm

  else           ! for gradient calculations.
     loopiiqm: do iqm = 1,numat
        n=qminb_local(iqm)      ! pointer for the index in x/y/z arrays.
        !if(my_ks_inv(n).le.0) cycle loopiiqm  ! should check
        do ith3 = 1,forder
           do ith2 = 1,forder
              do ith1 = 1,forder
                 ! In the end, it will be something like:
                 ! Qarray(i_qarray_mm(i,j,k,iqm))= sum_iqm [r_qarray_mm(i,j,k,iqm)*qm_chareg(iqm)]

                 ipt2 = i_qarray_mm(ith1,ith2,ith3,iqm)
                 if(ipt2.gt.0) then
                    Qarray(ipt2)=Qarray(ipt2)+r_qarray_mm(ith1,ith2,ith3,iqm)*scf_mchg_2(iqm)
                 end if
              end do
           end do
        end do
     end do   loopiiqm
  end if

  RETURN
  END SUBROUTINE fill_ch_grid_qm_mm

  SUBROUTINE potential_sumrc_qm_mm(igood, kbot, ktop,numatoms,numat,       &
                              qarray_local,qarray_mm_local,ewd_potential,ewd_potential_qm, &
                              qm_atm_grad_comp_local,                       &
                              recip,volume,ordr,nfftdim1,nfftdim2,nfftdim3, &
                              my_ks_inv,latm,xnsymm,                        &
                              igmsel_local,mminb2_local,qminb_local,q_cpmd)
  !
  ! This routine compute the potential at the QM atom sites applied by all MM atoms.
  !
  ! I have to only assume xnsymm is 1. Other cases, should not supported yet.
  !
  ! Note: parent routine must call with
  ! numat
  ! IGMSEL (or igmsel_dual, depending on the actual calcualtions.)
  ! MMINB2_dual(*,1 or 2)
  !

  use pmeutil,only: mxystart,mxyslabs,mxzslabs, &
                    nfft1,nfft2,nfft3, &
                    theta1,theta2,theta3,dtheta1,dtheta2,dtheta3

  use number
  use dimens_fcm
  use consta
  use parallel
  use stream
  use qm1_constant, only : EV,EVCAL,A0

  implicit none

  integer,intent(in) :: igood, kbot, ktop
  integer,intent(in) :: numatoms,numat,ordr
  integer,intent(in) :: nfftdim1,nfftdim2,nfftdim3,xnsymm,latm
  real(chm_real)     :: qarray_local(*),qarray_mm_local(*),   &
                        ewd_potential(*),ewd_potential_qm(*), &
                        qm_atm_grad_comp_local(3,*),volume
  integer,intent(in) :: my_ks_inv(*), &
                        igmsel_local(*),mminb2_local(*),qminb_local(*)
  real(chm_real),intent(in) :: recip(9)
  logical            :: q_cpmd

  integer :: igoo,ig,iqm,n
  integer :: I,J,K,KQ,i_keep,j_keep,k_keep,ITH1,ITH2,ITH3,IPT1,IPT2,IPT3
  integer :: rcskip,nfftdimrc                   ! RCFFT addition
  real(chm_real) :: CFACT,Pot,Pot_mm,P_tmp1,P_tmp2,fxyz(3),vala(3),val(3),ufact,CFACT2

  rcskip=1
  if(nfft1/2 /= (nfft1+1)/2) CALL WRNDIE(-5,'<grad_sumrc_qm_mm>','fftx dimension not even ')
  nfftdimrc=nfft1+4
  !
  ufact      = EV*EVCAL*A0
  if(xnsymm == 1)then
     CFACT=one                    ! CCELEC, since unit conversion
     CFACT2=ufact
  else
     CFACT=one/XNSYMM             ! CCELEC/XNSYMM
     CFACT2=ufact/XNSYMM
  end if
  !
  qm_atm_grad_comp_local(1:3,1:numat)=zero
  loopig: do iqm=1,numat
      n=qminb_local(iqm)      ! pointer for the index in x/y/z arrays.
      igoo=my_ks_inv(n)
      ig=igoo

#if KEY_PARALLEL==1
      ! skip if igoo .eq. 0
      if(igoo.le.0) cycle loopig      !##PARALLEL
#endif

      ! it is running for qm atoms.
      K_keep = INT(FR3(igoo)) - ORDR + 1 + NFFT3
      J_keep = INT(FR2(igoo)) - ORDR + 1 + NFFT2
      I_keep = INT(FR1(igoo)) - ORDR + 1 + NFFT1

      K        = k_keep
      Pot      = ZERO
      Pot_mm   = zero
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
               if(q_cpmd) then
                  do ITH1 = 1,ORDR
                     Pot    =Pot   +P_tmp2*qarray_local(IPT2)*THETA1(ITH1,ig)
                     Pot_mm =Pot_mm+P_tmp2*qarray_mm_local(IPT2)*THETA1(ITH1,ig)
                     IPT2=IPT2+rcskip
                     IF(IPT2 >= IPT3) IPT2=IPT2-NFFT1*rcskip
                  end do
               else
                  do ITH1 = 1,ORDR
                     Pot    =Pot+P_tmp2*qarray_local(IPT2)*THETA1(ITH1,ig)
                     fxyz(1)=fxyz(1)+val(1)*qarray_local(IPT2)*DTHETA1(ITH1,igoo)
                     fxyz(2)=fxyz(2)+val(2)*qarray_local(IPT2)*THETA1(ITH1,ig)
                     fxyz(3)=fxyz(3)+val(3)*qarray_local(IPT2)*THETA1(ITH1,ig)

                     IPT2=IPT2+rcskip
                     IF(IPT2 >= IPT3) IPT2=IPT2-NFFT1*rcskip
                  end do
               end if
            end do
#if KEY_PARALLEL==1
         end if
#endif
      end do
      !
      if(q_cpmd) then
         ! qm-mm component & qm-qm component
         ewd_potential(iqm)   = ewd_potential(iqm)    + CFACT*Pot_mm       ! mm only
         ewd_potential_qm(iqm)= ewd_potential_qm(iqm) + CFACT*(Pot-Pot_mm) ! qm only
      else
         ewd_potential(iqm)= ewd_potential(iqm) + CFACT*Pot
         qm_atm_grad_comp_local(1,iqm)=qm_atm_grad_comp_local(1,iqm)+ &
                                       CFACT2*(recip(1)*fxyz(1)+recip(4)*fxyz(2)+recip(7)*fxyz(3))
         qm_atm_grad_comp_local(2,iqm)=qm_atm_grad_comp_local(2,iqm)+ &
                                       CFACT2*(recip(2)*fxyz(1)+recip(5)*fxyz(2)+recip(8)*fxyz(3))
         qm_atm_grad_comp_local(3,iqm)=qm_atm_grad_comp_local(3,iqm)+ &
                                       CFACT2*(recip(3)*fxyz(1)+recip(6)*fxyz(2)+recip(9)*fxyz(3))
      end if
  end do loopig

  RETURN
  END SUBROUTINE potential_sumrc_qm_mm
  !                End of M/MM-PME (MNDO97) specific
  !****************************************************************
#endif /*mndo97*/

  !****************************************************************
  !                        PMESH_SETUP
  !****************************************************************
  SUBROUTINE PMESH_SETUP(NATOM,XNSYMM)
    !-----------------------------------------------------------------------
    !     This routine allocates space and defines variables for
    !     the Particle Mesh Ewald Summation
    !     original code by Tom Darden, implemented into CHARMM
    !     by Scott Feller and Bernie Brooks, NIH, Jan-1996
    !     Parallel 3D fft routines from Michael Crowley, Pittsburgh
    !     Supercomputing Center
    !-----------------------------------------------------------------------
    !
    use pmeutil,only:nfft1,nfft2,nfft3,sizfftab,sizffwrk,forder, &
         fft1_table,fft2_table,fft3_table,ffwork, &
         bsp_mod1,bsp_mod2,bsp_mod3, &
         mxyslabs,mxzslabs,get_fftdims &
         ,pll_fft_setup,load_bsp_mod
    
    use stream 
    use parallel
#if KEY_DOMDEC==1
    use domdec_common,only:q_domdec
    use domdec_dr_common,only:q_recip_node
#endif
#if KEY_DOMDEC_GPU==1
    use domdec_common,only:q_gpu, gpu_code_version
#endif
    
    INTEGER NATOM,XNSYMM
    !
    INTEGER NFFTDIM1,NFFTDIM2,NFFTDIM3,nfftable,NFFWORK
    INTEGER SIZ_Q,SIZHEAP,SIZTHETA,SIZDTHETA

    INTEGER MAXORDER,MAXNFFT
    PARAMETER (MAXORDER=25, MAXNFFT=2000)

    CALL GET_FFTDIMS(NFFTDIM1,NFFTDIM2,NFFTDIM3,NFFTABLE,NFFWORK)

    IF(PRNLEV > 6) WRITE(OUTU,46) nfft1,nfft2,nfft3,nfftdim1, &
         nfftdim2,nfftdim3,nfftable,nffwork
46  FORMAT(' <PME> FFT sizes =', 3I4, /, &
         ' <PME> FFT dims  =', 3I4, /, &
         ' <PME> FFT table =', 1I9, /, &
         ' <PME> FFT work  =', 1I9   )
    !          tripled for 3 FFTs
    SIZFFTAB = 3*NFFTABLE
    !          doubled for complex
    SIZFFWRK = 2*NFFWORK

    SIZTHETA  = NATOM*XNSYMM*FORDER
    SIZDTHETA = NATOM*FORDER
#if KEY_PARALLEL==1
    SIZ_Q = max(2*NFFTDIM1*NFFTDIM2*mxyslabs,  &
         2*NFFTDIM1*NFFTDIM3*mxzslabs)
#else /**/
    SIZ_Q = 2*NFFTDIM1*NFFTDIM2*NFFTDIM3
#endif 
    !
    SIZHEAP = NFFT1+NFFT2+NFFT3+SIZFFTAB + &
         SIZ_Q+3*SIZTHETA+3*SIZDTHETA+SIZFFWRK+4*NATOM*XNSYMM
    !
    IF(PRNLEV > 3) WRITE(OUTU,45) SIZHEAP
45  FORMAT(' <PME> Total heap storage needed =',I12)
    !
    !     allocate long-term heap space
#if KEY_DOMDEC==1
    if ((.not.q_domdec) .or. (q_domdec .and. q_recip_node)) then  
#endif
#if KEY_DOMDEC_GPU==1
       if (.not.q_gpu .or. gpu_code_version==1) then
#endif
          if(allocated(fft1_table))call pmesh_clear
          allocate(fft1_table(4*nfftdim1), fft2_table(4*nfftdim2), &
               fft3_table(4*nfftdim3),ffwork(sizffwrk))
          allocate(bsp_mod1(nfft1),bsp_mod2(nfft2),bsp_mod3(nfft3))
          CALL LOAD_BSP_MOD
#if KEY_DOMDEC_GPU==1
       endif
#endif
#if KEY_DOMDEC==1
    endif  
#endif

#if KEY_COLFFT==0
    call pll_fft_setup(nfftdim1,nfftdim2,nfftdim3)
#endif
    !
    !
    RETURN
  END subroutine pmesh_setup


end module pme_module




subroutine dumpq(a,n1,n2,n3,lun,p_type,n10,n20,n30)
  implicit none
  integer,intent(in) :: n1,n2,n3,lun
  integer,intent(in) :: p_type,n10,n20,n30
  integer :: print_type,n100,n200,n300
  double complex,intent(in) :: a(n1,n2,n3)
  integer :: i,j,k

  !      print *," MINS in dumpq",n10,n20,n30
  !      if( present(p_type) ) then
  print_type = p_type
  !      else
  !         print_type = 3
  !      endif
  !      print *," DUMPQ print type: ",print_type
  n100=0
  n200=0
  n300=0
  !      if( present(n10) ) then
  n100=n10
  !      endif
  !      if( present(n20)) then
  n200=n20
  !      endif
  !      if( present(n30)) then
  n300=n30
  !      endif
  write(lun,'(6i6)')n1,n100,n2,n200,n3,n300
  do k=1,n3
     do j=1,n2
        do i=1,n1
           select case(print_type)
           case(0)     
              write(lun,904)a(i,j,k)
           case(1)
              write(lun,903)i+n100,a(i,j,k)
           case(2)
              write(lun,902)i+n100,j+n200,a(i,j,k)
           case(3)
              write(lun,901)i+n100,j+n200,k+n300,a(i,j,k)
           case default
              call wrndie(-1,"DUMPQ<ew_fft.f>",  &
                   "bad printout type specified, allowed types 0,1,2,3")
           end select
        end do
     end do
  end do
901 format(3i4,2e17.8)
902 format(2i4,2e17.8)
903 format(i4,2e17.8)
904 format(2e17.8)
  return
end subroutine dumpq

