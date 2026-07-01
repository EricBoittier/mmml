! PZ, MG_QC_UW1206: copied from ../nbonds/pme.src
module sccpme_module

  use chm_kinds
  use prssre, only: getvol
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
#if KEY_SQUANTM==1 || KEY_QCHEM==1 || KEY_G09==1 || KEY_SCCDFTB==1
  integer,allocatable,dimension(:),save :: lmy_ks_inv   
#endif

  ! for qm/mm-pme version
#if KEY_SQUANTM==1 || KEY_QCHEM==1 || KEY_G09==1 || KEY_SCCDFTB==1
  ! for qm-qm interactions.
  real(chm_real),allocatable,dimension(:,:),save :: qm_atm_grad_comp
  integer, save :: igood_qmmm

  ! for virial.
  real(chm_real),allocatable,dimension(:),save :: qarray_mm

  ! for gradient.
  real(chm_real),allocatable,dimension(:,:,:,:),save :: r_qarray_mm
  integer,       allocatable,dimension(:,:,:,:),save :: i_qarray_mm
#endif 

contains

  !****************************************************************
  !                        PMESH_CLEAR
  !****************************************************************
  SUBROUTINE PMESH_CLEAR
    use sccpmeutil
    !
    !  This routine releases PME allocated space.
    !
    if(allocated(bsp_mod1)) &
         deallocate(bsp_mod1,bsp_mod2,bsp_mod3)
    if(allocated(fft1_table)) &
         deallocate(fft1_table,fft2_table,fft3_table,ffwork)
    !
    RETURN
  END subroutine pmesh_clear
 
  !****************************************************************
  !                  PME
  !****************************************************************
  !
  SUBROUTINE PME(EKSUM,LESELF,EQCOR,EUTIL, &
       QEKSUM,QESELF,QEQCOR,QEUTIL, &
       X,Y,Z,DX,DY,DZ,NATOM,CG,CGTOT, &
       ewvirial,kappa)
    !
    !
    ! QM/MM-Ewald
#if KEY_QUANTUM==1
    use quantm,only: LQMEWD, QSETUPKQ, QCGSC   
#endif
#if KEY_MNDO97==1
    use mndo97,only: lqmewd                    
#endif
#if KEY_GAMESS==1 || KEY_GAMESSUK==1 || KEY_G09==1
#if KEY_SQUANTM==0
    use mndo97,only: lqmewd                    
#endif
#endif 
#if KEY_SQUANTM==1
    use squantm,only: lqmewd,ewmode            
#endif
    
    use erfcd_mod,only: erfcd
    use ewald_1m,only: erfmod
    use sccpmeutil
    !-----------------------------------------------------------------------
    !     This routine calculates non bonded interaction energies and
    !     forces via the Particle Mesh Ewald Summation
    !     original code by Tom Darden, implemented into CHARMM
    !     by Scott Feller and Bernie Brooks, NIH, Jan-1996
    !     Parallel 3D fft routines from Michael Crowley, Pittsburgh
    !     Supercomputer Center
    !-----------------------------------------------------------------------

#if KEY_CHEQ==1
    use cheq, only: qcg,   &                  
#endif
#if KEY_CHEQ==1
         DCH,SUMDCH,DDCH                  
#endif

    use dimens_fcm
    use number
    use energym
    use inbnd
    use image
    use pbound
    use consta
    use stream
#if KEY_BLOCK==1
    use block_fcm         
#endif
    use parallel
    !
    real(chm_real) ::  EKSUM,LESELF,EQCOR,EUTIL
    real(chm_real) :: ewvirial(9),kappa
    LOGICAL QEKSUM,QESELF,QEQCOR,QEUTIL
    real(chm_real) ::  X(*),Y(*),Z(*),DX(*),DY(*),DZ(*)
    INTEGER NATOM
    real(chm_real) ::  CG(*),CGTOT
    INTEGER IONE
    real(chm_real) :: imxcent(1),imycent(1),imzcent(1)

#if KEY_CHEQ==1
    real(chm_real) ::    HIJ        
#endif
    real(chm_real) :: xtlinv(6),recip(3,3),virial(6), &
         vircor(6),eslf,cfact,ent                        !av_080628
    real(chm_real) :: rdum(3),htmp(9),hwrk(9)
    integer siztheta, sizdtheta, siz_q
    integer nfftdim1, nfftdim2, nfftdim3

    integer i,j, ii, ipt, itrans
    real(chm_real) :: tx,ty,tz,rs,s2,erfcx,drfc,ene,qd(1),dxf,dyf,dzf

#if KEY_PMEPLSMA==1
    real(chm_real) factor,ee_plasma      
#endif

#if KEY_BLOCK==1
    real(chm_real) CHF2(3),DSLF(3),CURCG,CHFACT    
#endif

#if KEY_BLOCK==1
    REAL(kind=chm_real)              :: COEF
    INTEGER                          :: IBL, JBL, KK,JB,IB
#endif /*   close BLOCK*/

    logical ok
    integer atfrst,atlast,nat
    integer alloc_err
    integer xnsymm_t
    integer ia

#if KEY_PARALLEL==1 /*paramain*/
#if KEY_PARAFULL==1 /*parfmain*/
       atfrst=1+iparpt(mynod)
       atlast=iparpt(mynodp)
#elif KEY_PARASCAL==1 || KEY_SPACDEC==1 /*parfmain*/
       atfrst=1
       atlast=natom
#endif /* (parfmain)*/
#else /* (paramain)*/
       atfrst=1
       atlast=natom
#endif /* (paramain)*/

    nat=atlast-atfrst+1

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

    self_e: if(qeself) then
       eslf = zero
       !.ab...HYBH.We can use the fact that energy.f90 is here.
#if KEY_BLOCK==1
       if (qhybh) then
          ihybh=ewself
          chf2(1)=1.
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

       selfloop: do ia = atfrst, atlast
          ii = ia

          ent = cg(ii) * cg(ii)

#if KEY_BLOCK==1
          if (qblock) then
             ibl = iblckp(ii)
             kk  = ibl + ibl*(ibl-1)/2
             coef = blcoep(kk)
             ent = ent*coef
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

#if KEY_BLOCK==1
       if (qhybh) then
          dslf(2)=two*kappa*ccelec/sqrt(pi)*(one-hybhlb)*dslf(2)
          dslf(3)=-two*kappa*ccelec/sqrt(pi)*hybhlb*dslf(3)
          call sumhyb(ihybh,dslf(2),dslf(3))
       endif
#endif 
       leself = - eslf*kappa*ccelec/sqrt(pi)
    else  ! self_e
       leself = 0.0_chm_real
    endif self_e

#if KEY_QUANTUM==1 || KEY_MNDO97==1 || KEY_SQUANTM==1 || KEY_GAMESS==1 || KEY_GAMESSUK==1 || KEY_G09==1
    if(.not.lqmewd) then      
#endif
       ewvirial(1:9)=zero
#if KEY_QUANTUM==1 || KEY_MNDO97==1 || KEY_SQUANTM==1 || KEY_GAMESS==1 || KEY_GAMESSUK==1 || KEY_G09==1
    endif
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
    allocate(qarray(siz_q),stat=alloc_err)
    if(alloc_err /= 0 ) write(0,*)"unable to allocate qarray"

    allocate(fr1(nattot),fr2(nattot),fr3(nattot),cg1(nattot), &
         stat=alloc_err)
    if(alloc_err /= 0 ) write(0,*)"unable to allocate fr arrays"
    call allocate_bspline(natom,nattot)
    !
    do_pme: if(qeksum) then
#if KEY_COLFFT==1 /*colfft*/
       call wrndie(-4,"pme.src:COLFFT", &
         "2012 COLFFT not yet implemented for sccdftbint")    

       !==================COLUMN FFT METHOD ==============================
       ! deprecated pre 2012 colfft methods 
       ! this needs to be reimplemented using the new colfft methods
! #if KEY_BLOCK==1
!        if(QHYBH) CALL WRNDIE(-5,'<PME>', 'HYBH not implemented forCOLFFT')
! #endif
!        call pme_column(natom,nfftdim1,nfftdim2,nfftdim3, &
!             eksum,leself,eqcor,eutil, &
!             qeksum,qeself,qeqcor,qeutil, &
!             x,y,z,dx,dy,dz,cg,cgtot,virial,kappa)
! #else /*      (colfft)*/
! #if KEY_BLOCK==1
!        if(qhybh) ihybh=ewksum                            
! #endif
!        if(xnsymm == 0) call wrndie(-5,'<PME>', &
!             'XNSYMM is zero: build crystal.')
!        call do_pme_ksp(qfinit,rewcut,natom, &
!             x,y,z,cg,recip,eprop(volume), &
!             kappa,forder,eksum,dx,dy,dz,virial, &
! #if KEY_CHEQ==1
!             dch,qcg,                                              & 
! #endif
!             sizfftab,sizffwrk,siztheta,siz_q,cg1, &
!             xnsymm,maxsym,xsymop)
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
#endif /*   (pmeplsma)*/
#if KEY_PMEPLSMA==1 && KEY_BLOCK==1
       if(qhybh) then                               
#endif
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
                  xnsymm,maxsym,xsymop)

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
          !...#.#.ENDIF   (pmeplsma)
       endif  !---------------end of charged system correction -------------
#if KEY_BLOCK==1 && KEY_PMEPLSMA==1
    ENDIF                                  
#endif
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
    deallocate(fr1,fr2,fr3,cg1,stat=alloc_err)
    if(alloc_err /= 0 ) write(0,*)"unable to deallocate fr1,2,3"
    deallocate(qarray,stat=alloc_err)
    if(alloc_err /= 0 ) write(0,*)"unable to deallocate qarray"
    call deallocate_bspline()
    !
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
    else
       eksum=0.0
    endif

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
       cg1,xnsymm,maxsyml,xsymop)

  use new_timer,only:timer_start,timer_stop,timer_stpstrt,  & 
     T_fillg,T_FFT,T_scsum,T_grads                       
  use sccpmeutil,only:nxyslab,mxyslabs,mxzslabs,nfft1,nfft2, &
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
  use block_fcm              
#endif
  use memory
     !
!!!     integer,allocatable,dimension(:) :: lmy_ks      ! this is moved to above.
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
#if KEY_PARALLEL==1
       !
       ! H Kamberaj: BLOCK code added, November 2007
#if KEY_BLOCK==1
       IF (QBLOCK) THEN
          call fft3d0rc(0,scale,qtemp, &
               nfftdim1,nfftdim2, &
               tmpy,alpha,beta)
       ELSE
          call fft3d0rc(0,scale,qarray, &
               nfftdim1,nfftdim2, &
               tmpy,alpha,beta)
       ENDIF
#else /**/
       call fft3d0rc(0,scale,qarray, &
            nfftdim1,nfftdim2, &
            tmpy,alpha,beta)

#endif 
       !
#else /**/
       !
       ! H KAMBERAJ, NOVEMBER 2007
#if KEY_BLOCK==1
       IF (QBLOCK) THEN
          call fft3d0rc(0,scale,qtemp, &
               nfftdim1,nfftdim2, &
               tmpy,alpha,beta)
       ELSE
          call fft3d0rc(0,scale,qarray, &
               nfftdim1,nfftdim2, &
               tmpy,alpha,beta)

       ENDIF
#else /**/
       call fft3d0rc(0,scale,qarray, &
            nfftdim1,nfftdim2, &
            tmpy,alpha,beta)
#endif 
       !
#endif 
       !.ab.x/box and images + ch(img). (cg1 filled below if test.false).
       if(xnsymm > 1) &
            call get_sc_fract(fr1,fr2,fr3,cg1, &
            natom,nattot,x,y,z,recip, &
            xnsymm,maxsyml,xsymop,cg)


       !       make array for keeping track of atoms important to
       !         this processor
#if KEY_PARALLEL==1
       latm=nattot 
#else /**/
       latm=nattot
#endif 
       call chmalloc('sccpme.src','do_pme_ksp','lmy_ks',latm,intg=lmy_ks)
#if KEY_SQUANTM==1 || KEY_QCHEM==1 || KEY_G09==1 || KEY_SCCDFTB==1
       call chmalloc('sccpme.src','do_pme_ksp','lmy_ks_inv',nattot,intg=lmy_ks_inv) 
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
#if KEY_SQUANTM==1 || KEY_QCHEM==1 || KEY_G09==1 || KEY_SCCDFTB==1
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
#if KEY_SQUANTM==1 || KEY_QCHEM==1 || KEY_G09==1 || KEY_SCCDFTB==1
               lmy_ks_inv,  &   
#endif
               mxyslabs)
       endif
       call timer_stpstrt(T_fillg,T_FFT)                  

       allocate(tmpy(2*nfftdim1),alpha(nfft1),beta(nfft1))
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
          call fft_backrc(Qarray, &
               nfftdim1,nfftdim2,nfftdim3,nffwork)
       endif
#else /**/
       !.ab.Ok.TF^-1(Q)
       call fft_backrc(Qarray, &
            nfftdim1,nfftdim2,nfftdim3,nffwork)
#endif /*   close block*/
       !

       call timer_stpstrt(T_FFT,T_scsum)                  
       !.ab.Calculate.E.
       call scalar_sum(qfinit,rewcut, &
            ewald_coeff,volume,recip, &
            nfftdim1,nfftdim2,nfftdim3,eer,virial)
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
          call fft_forwardrc(Qarray,nfftdim1,nfftdim2,nfftdim3,nffwork)
       endif
#else /**/
       call fft_forwardrc(Qarray,nfftdim1,nfftdim2,nfftdim3,nffwork)
#endif /*    close block */
       !
       !.ab.Now Q=(Theta_rec*Q). Essmann etal 1995 Eq.4.7.
       deallocate(tmpy,alpha,beta)
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
       call chmdealloc('sccpme.src','do_pme_ksp','lmy_ks',latm,intg=lmy_ks)
#if KEY_SQUANTM==1 || KEY_QCHEM==1 || KEY_G09==1 || KEY_SCCDFTB==1
       call chmdealloc('sccpme.src','do_pme_ksp','lmy_ks_inv',nattot,intg=lmy_ks_inv) 
#endif
       !
#if KEY_GRAPE==1
    ENDIF                 
#endif
    !
    RETURN
  END subroutine do_pme_ksp
  

  
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
#if KEY_SQUANTM==1 || KEY_QCHEM==1 || KEY_G09==1 || KEY_SCCDFTB==1
       my_ks_inv, &   
#endif
       nfftdim3)

  use sccpmeutil,only:nfft1,nfft2,nfft3,forder, &
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
  use block_fcm
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
    logical qfill
    integer latm,my_ks(latm)
#if KEY_SQUANTM==1 || KEY_QCHEM==1 || KEY_G09==1 || KEY_SCCDFTB==1
    integer :: my_ks_inv(numatoms)   
#endif
    integer rcskip,nfftdimrc

    rcskip=1
    if(nfft1/2 /= (nfft1+1)/2)then
       call wrndie(-5,'<pme fill charge grid>' &
            ,'fftx dimension not even ')
    endif
    nfftdimrc=nfft1+4
    !
    igood=0
    igoody=0
    my_ks(1:latm)=0
#if KEY_SQUANTM==1 || KEY_QCHEM==1 || KEY_G09==1 || KEY_SCCDFTB==1
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

       qfill = .true.

       if(xnsymm > 1)then
          fr3n=fr3(n)
       else
          w = x(n)*recip(7)+y(n)*recip(8)+z(n)*recip(9)
          fr3n = nfft3*(w - anint(w) + half)
       endif
       k = int(fr3n) - forder + 1 + nfft3

       do ith3 = 1,forder
          k=k+1
          if(k > nfft3) k=k-nfft3
          kq=k
#if KEY_PARALLEL==1
          if ( k  >=  kbot .and. k  <=  ktop )then
             kq = k - kbot0
#endif 
             if(qfill) then
                qfill = .false.
                igood=igood+1
                my_ks(igood)=n
#if KEY_SQUANTM==1 || KEY_QCHEM==1 || KEY_G09==1 || KEY_SCCDFTB==1
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
             endif
#if KEY_BLOCK==1
             IF(QHYBH) THEN                            
#endif
#if KEY_BLOCK==1
                proda = theta3(ith3,igood)*cg1(n)      
#endif
#if KEY_BLOCK==1
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
                ipt2= rcskip*((ipt1+j)*nfftdimrc+i)+1
                ipt3= ipt2 + rcskip*(nfft1-i)
                !
                do ith1 = 1,forder
                   !
                   ! Block code: H Kamberaj, November 2007
#if KEY_BLOCK==1
                   IF (QBLOCK) THEN
                      INX = INXT + IPT2
                      if (inx  >  dim_q) then
                         call wrndie(-5, '<FILL_CH_GRID>', &
                              'Index INX  >  DIM_Q')
                      endif
                      Qarray(INX)=Qarray(INX)+THETA1(ITH1,IGOOD)*PROD
                   ELSE
                      Qarray(IPT2)=Qarray(IPT2)+THETA1(ITH1,IGOOD)*PROD
                   ENDIF
#else /**/
                   Qarray(IPT2)=Qarray(IPT2)+THETA1(ITH1,IGOOD)*PROD
#endif /*  close BLOCK*/
                   !
                   ipt2=ipt2+rcskip
                   if(ipt2 >= ipt3) ipt2=ipt2-nfft1*rcskip
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
  use sccpmeutil,only:fft3d0rc
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
       nfftdim1,nfftdim2,nfftdim3,eer,vir)

  use sccpmeutil,only:mxzslabs,mxzstart, &
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
  use block_fcm
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
       qarray(1)=0.
       qarray(2)=0.
#if KEY_PARALLEL==1
    endif                  
#endif
#endif 
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
             !...##ELSE (pll0)
             !...##IF PMEPLSMA
             !         q(1)=0.
             !         q(2)=0.
             !...##ENDIF
             !         DO K3 = 1,NFFT3
             !            M3 = K3 - 1
             !            IF(K3 > NF3) M3 = M3 - NFFT3
             !            DEN2 = DEN1*BSP_MOD3(K3)
             !            MHAT1A = RECIP(7)*M3
             !            MHAT2A = RECIP(8)*M3
             !            MHAT3A = RECIP(9)*M3
             !            IPT2=IPT1
             !            DO K2 = 1, NFFT2
             !               M2 = K2 - 1
             !               IF(K2 > NF2) M2 = M2 - NFFT2
             !               DEN3 = DEN2*BSP_MOD2(K2)
             !               MHAT1B = MHAT1A+RECIP(4)*M2
             !               MHAT2B = MHAT2A+RECIP(5)*M2
             !               MHAT3B = MHAT3A+RECIP(6)*M2
             !               IPT3=IPT2
             !               IF(K3+K2 == 2) THEN
             !                  K0=2
             !                  IPT3=IPT3+2
             !               ELSE
             !                  K0=1
             !               ENDIF
             !               DO K1 = K0, NFFT1
             !                  M1 = K1 - 1
             !                  IF(K1 > NF1) M1 = M1 - NFFT1
             !                  DEN4 = DEN3*BSP_MOD1(K1)
             !                  MHAT1 = MHAT1B+RECIP(1)*M1
             !                  MHAT2 = MHAT2B+RECIP(2)*M1
             !                  MHAT3 = MHAT3B+RECIP(3)*M1
             !...##ENDIF   (pll0)
             MSQ = MHAT1*MHAT1+MHAT2*MHAT2+MHAT3*MHAT3
             MSQR=ONE/MSQ
             !
             ETERM = EXP(-FAC*MSQ)*DEN4*MSQR
             VTERM = TWO*(FAC+MSQR)
             !av_080628
             ! Energy decomposition: H Kamberaj (November 2007)
#if KEY_BLOCK==1
             IF (QBLOCK) THEN
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
                      COEF = BLCOEP(KK)
                      INXJ = (JB-1)*NTOT + IPT3
                      STRUC2 = STRUC2 + COEF * ( &
                           qarray(INXI)*qarray(INXJ) + &
                           qarray(INXI+1)*qarray(INXJ+1) &
                           )
                   ENDDO
                ENDDO
             ELSE
                STRUC2 = qarray(IPT3)*qarray(IPT3) +  &
                     qarray(IPT3+1)*qarray(IPT3+1)
             ENDIF
#else /**/
             STRUC2 = qarray(IPT3)*qarray(IPT3) +  &
                  qarray(IPT3+1)*qarray(IPT3+1)
#endif /*   close BLOCK*/
             !av_080628
#if KEY_DEBUG==1
             IF(.NOT.QFINIT) THEN
                IF(ABS(eterm * struc2) > 5.0D-9) THEN
                   write(22,834) K1,K2,K3,MHAT1,MHAT2,MHAT3,eterm,struc2
834                FORMAT(3I4,5F16.10)
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
#if KEY_BLOCK==1
                IF (QBLOCK) THEN
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
                         COEF = BLCOEP(KK)
                         STRUC2s = STRUC2s + COEF * ( &
                              Qarray(INXI)*Qarray(INXJ) + &
                              Qarray(INXI+1)*Qarray(INXJ+1) &
                              )
                      ENDDO
                   ENDDO
                ELSE
                   STRUC2s = Qarray(IPT3)*qarray(ipt3) +  &
                        Qarray(IPT3+1)*qarray(ipt3+1)
                ENDIF
#else /**/
                STRUC2s = Qarray(IPT3)*qarray(ipt3) +  &
                     Qarray(IPT3+1)*qarray(ipt3+1)
#endif /*     close BLOCK*/
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
#if KEY_BLOCK==1
             IF (QBLOCK) THEN
                DO IB=1, NBLOCK
                   INXI = (IB-1)*NTOT + IPT3
                   Qarray(INXI)   = ETERM * Qarray(INXI)
                   Qarray(INXI+1) = ETERM * Qarray(INXI+1)
                ENDDO
             ELSE
                Qarray(IPT3)   = ETERM * Qarray(IPT3)
                Qarray(IPT3+1) = ETERM * Qarray(IPT3+1)
             ENDIF
#else /*   av_080628*/
             Qarray(IPT3)   = ETERM * Qarray(IPT3)
             Qarray(IPT3+1) = ETERM * Qarray(IPT3+1)
#endif /*  close BLOCK av_080628*/
             !
             IPT3=IPT3+2
          ENDDO
          IPT2=IPT2+NFFT3*2
       ENDDO
       IPT1=IPT1+NFFT3*NFFTDIM1*2
    ENDDO
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

  use sccpmeutil,only:fft3d_zxyrc
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

  use sccpmeutil,only: mxystart,mxyslabs,mxzslabs, &
     nfft1,nfft2,nfft3, &
     theta1,theta2,theta3,dtheta1,dtheta2,dtheta3

  use number
  use dimens_fcm
  use consta
  use parallel
  use stream
#if KEY_BLOCK==1
  use block_fcm
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
    integer rcskip,nfftdimrc
    rcskip=1
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

                IPT2= rcskip*((IPT1+J)*NFFTDIMrc+I)+1
                IPT3= IPT2 + rcskip*(NFFT1-I)
                !
                DO ITH1 = 1,ORDR
                   !                                   force is negative of grad
                   ! Block code: H Kamberaj (November 2007)
#if KEY_BLOCK==1
                   IF (QBLOCK) THEN
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
                         COEF = BLCOEP(KK)
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
                   ELSE
                      CGS= Qarray(IPT2)
                      CG = CGS
                   ENDIF
#else /**/
                   CGS= Qarray(IPT2)
                   CG = CGS
#endif /*   close BLOCK*/
                   !

                   F1 = F1 - VAL1 * cgs * DTHETA1(ITH1,igoo)
                   F2 = F2 - VAL2 * cgs * THETA1(ITH1,ig)
                   F3 = F3 - VAL3 * cgs * THETA1(ITH1,ig)
#if KEY_CHEQ==1
                   HIJ=HIJ+THETA1(ITH1,ig)*THETA2(ITH2,ig)* &
                        THETA3(ITH3,ig)*cg
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

  use sccpmeutil,only: mxystart,mxyslabs,nfft1,nfft2,nfft3, &
     theta1,theta2,theta3,dtheta1,dtheta2,dtheta3

  use number
  use dimens_fcm
  use consta
  use parallel
     !...##INCLUDE '~/charmm_fcm/pme_par.f90'
     !.ab.
  use block_fcm
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
    CALL SUMHYB(IHYBH,DEW(2),DEW(3))
    !
    RETURN
  END subroutine grad_sumrch

  
  SUBROUTINE DO_PME_COR(NATOM,CURCG,CURCHG,ORGCHG,FACT,CGB,QEUTIL)
    !.ab. calculate Background correction.
    !.ab. CURCHG are supposed to have been scaled.
  use chm_kinds
  use block_fcm
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
    IF(QEUTIL)CALL SUMHYB(IHYBH,-CGB(2)*CURCG*FACT,CGB(3)*CURCG*FACT)
    !      write(6,*) 'BgCor: ',curcg,fact,CGB(2),CGB(3)
    !
    RETURN
  END SUBROUTINE DO_PME_COR

#endif /* (block)*/

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
    use sccpmeutil,only:nfft1,nfft2,nfft3,sizfftab,sizffwrk,forder, &
         fft1_table,fft2_table,fft3_table,ffwork, &
         bsp_mod1,bsp_mod2,bsp_mod3, &
         mxyslabs,mxzslabs,get_fftdims &
         ,pll_fft_setup,load_bsp_mod
    
    use stream 
    use parallel

#if KEY_COLFFT==1
    use colfft_util, only: colfft_util_init
#endif
    implicit none

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
    if(allocated(fft1_table))call pmesh_clear
    allocate(fft1_table(4*nfftdim1), fft2_table(4*nfftdim2), &
         fft3_table(4*nfftdim3),ffwork(sizffwrk))
    allocate(bsp_mod1(nfft1),bsp_mod2(nfft2),bsp_mod3(nfft3))
    
    CALL LOAD_BSP_MOD

#if KEY_COLFFT==1
    call colfft_util_init(numnod)
#else /**/
    call pll_fft_setup(nfftdim1,nfftdim2,nfftdim3)
#endif 
    !
    !
    RETURN
  END subroutine pmesh_setup


end module sccpme_module




