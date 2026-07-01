#if KEY_MNDO97==1 /*mndo97*/
!!!SUBROUTINE MNDENE(CTOT,X,Y,Z,DX,DY,DZ)
!!!  !-----------------------------------------------------------------------
!!!  !
!!!  ! Get energy and forces from MNDO97
!!!  !
!!!  use chm_kinds
!!!  use dimens_fcm
!!!  use number
!!!  use exfunc
!!!  use memory
!!!  !
!!!  !use bases_fcm
!!!  use consta
!!!  use contrl
!!!  use gamess_fcm
!!!  use inbnd
!!!  use mndo97
!!!  use mndgho
!!!  use ewald_1m, only: lewald,EWVIRIAL2
!!!  ! use erfcd_mod,only: EWLDT
!!!  use quantm, only : natom_check,xim,yim,zim
!!!  use nbndqm_mod, only : imattq
!!!  use psf
!!!  use stream
!!!  use prssre, only: getvol
!!!#if KEY_PARALLEL==1
!!!  use parallel
!!!#endif
!!!  !
!!!  use qm1_info, only : qm_control_r,qm_main_r,mm_main_r,qm_scf_main_r
!!!  use qmmm_interface, only : fill_mm_coords,fill_dist_qm_mm_array,             &
!!!                             scf_energy,scf_gradient,                          &
!!!                             qmmm_Ewald_setup_and_potential,qmmm_Ewald_gradient, &
!!!                             scf_energy_prep
!!!  use mndgho_module, only : qm_gho_info_r,HBDEF,DQLINK
!!!  use mndnbnd_module, only: ch2mnd
!!!  ! D3BJ & H4 corrections
!!!  use consta, only: BOHRR,AU_TO_EV,TOKCAL
!!!  use dftd3_mndo, only : calc_e_dftd3,calc_g_dftd3
!!!  use H4_mndo, only : q_h4corr,h4_correction
!!!
!!!  !
!!!  use leps
!!!
!!!  ! Adjust nonbonded group list for IMAGES and simple PBC.
!!!  use image
!!!#if KEY_PBOUND==1
!!!  use pbound
!!!#endif
!!!
!!! 
!!!  implicit none
!!!
!!!  real(chm_real) :: CTOT,X(*),Y(*),Z(*),DX(*),DY(*),DZ(*)
!!!
!!!  ! local variables
!!!  INTEGER :: ICALL, NTATOM
!!!  INTEGER :: I,J,II,N,IPT,N1,MNUM,INDS, INDF, PRNLEVQM2
!!!  real(chm_real) ::  ECLASS,EH4_corr
!!!  LOGICAL :: QDONE, QIMAGE,qfail
!!!
!!!  ! for leps/svb correction terms
!!!  real(chm_real) :: E_LEPS,DA_LEPS_local(3),DB_LEPS_local(3),DC_LEPS_local(3), &
!!!                    DD_LEPS_local(3),DE_LEPS_local(3),DF_LEPS_local(3)
!!!
!!!  ! for qm/mm-ewald setup
!!!  real(chm_real):: VOLUME,XTLINV(6)
!!!  logical:: qfrst,qcheck,OK
!!!
!!!  ! D3BJ & H4 corrections
!!!  ! Dispersion gradient contribution. Defined this way to
!!!  ! fit into the wrapper for Grimmes code.
!!!  real(chm_real)         :: e_dftd3, g_dftd3_norm
!!!  real(chm_real),pointer :: xyz_disp(:,:)=>Null(), &  ! 3,numat
!!!                            g_dftd3(:,:)=>Null()      ! 3,numat
!!!  real(chm_real),parameter :: r_BOHRR = one/BOHRR, &  ! 1/bohrr
!!!                              r_atoev=one/AU_TO_EV,&
!!!                              conv_grad=TOKCAL/BOHRR
!!!  real(chm_real):: boxsiz(3,3), e_disp_local
!!!  integer       :: nlat(3)
!!!  logical       :: qperiodic,lgrad
!!!  logical, save :: qdftd3_first=.true.
!!! 
!!!  ! return, if not setup qm/mm
!!!  if(.not.qm_control_r%ifqnt) return
!!!
!!!  ! initial check up
!!!  qimage =.FALSE.
!!!  ntatom = natom
!!!
!!!#if KEY_PBOUND==1
!!!  if(.not.qBoun) then
!!!#endif
!!!     if(ntrans.gt.0) then
!!!        !if(lgroup) then
!!!           qimage =.true.
!!!        !else
!!!        !   call wrndie(-1,'<MNDENE>', 'QM/MM do not interact with Images under Atom Based Cutoff.')
!!!        !end if
!!!     end if
!!!#if KEY_PBOUND==1
!!!  end if
!!!#endif
!!!
!!!  ! assign stack array for temporary coordinate
!!!  if(allocated(xim).and. size(xim).lt.natom) then
!!!     if(allocated(xim)) call chmdealloc('mndini.src','MNDENE','xim',size(xim),crl=xim)
!!!     if(allocated(yim)) call chmdealloc('mndini.src','MNDENE','yim',size(yim),crl=yim)
!!!     if(allocated(zim)) call chmdealloc('mndini.src','MNDENE','zim',size(zim),crl=zim)
!!!     natom_check = natom      ! for checking purpose
!!!  end if
!!!  if(.not.allocated(xim)) call chmalloc('mndini.src','MNDENE','xim',natom,crl=xim)
!!!  if(.not.allocated(yim)) call chmalloc('mndini.src','MNDENE','yim',natom,crl=yim)
!!!  if(.not.allocated(zim)) call chmalloc('mndini.src','MNDENE','zim',natom,crl=zim)
!!!
!!!  call SwapXYZ_image(natom,X,Y,Z,xim,yim,zim,imattq)
!!!  
!!!  ! update QM coordinates
!!!  do i=1,qm_main_r%numat
!!!     n=qm_control_r%qminb(i)
!!!     qm_main_r%qm_coord(1,i)=X(n)
!!!     qm_main_r%qm_coord(2,i)=Y(n)
!!!     qm_main_r%qm_coord(3,i)=Z(n)
!!!  end do
!!!
!!!!!#if KEY_GHO==1
!!!  ! update the hybridization matrix for GHO
!!!  if(qm_gho_info_r%q_gho) then
!!!     qfail=.true.
!!!     call hbdef(x,y,z,qm_gho_info_r%BT,qm_gho_info_r%BTM,qm_gho_info_r%DBTMMM, &
!!!                qm_gho_info_r%nqmlnk,qm_gho_info_r%mqm16,                      &
!!!                qm_gho_info_r%IQLINK,qm_gho_info_r%JQLINK,qm_gho_info_r%KQLINK,&
!!!                qfail)
!!!  end if
!!!!!#endif
!!!
!!!  ! non-bond list and prepare for QM/MM-interaction list
!!!  call ch2mnd(qm_main_r%numat,igmsel,xim,yim,zim,.false.)
!!!
!!!  ! copy mm coords for qm/mm calculations
!!!  call fill_mm_coords(natom,xim,yim,zim,cg, &
!!!                      mm_main_r%mm_coord,mm_main_r%mm_chrgs, &
!!!                      mm_main_r%qm_mm_pair_list,qm_control_r%mminb1)
!!!
!!!  ! incore preparation.
!!!  if(qm_main_r%rij_qm_incore .or. mm_main_r%rij_mm_incore) then
!!!     call fill_dist_qm_mm_array(qm_main_r%numat,mm_main_r%numatm,      &
!!!                                qm_main_r%qm_coord,mm_main_r%mm_coord, &
!!!                                mm_main_r%LQMEWD)
!!!  end if
!!!
!!!  if(mm_main_r%LQMEWD) then
!!!     ! get the volume and reciprocal space lattice vector
!!!     call getvol(volume)
!!!     call INVT33S(xtlinv,xtlabc,ok)
!!!
!!!     ! setup Ktable and Kvec
!!!     qcheck=.true.
!!!     call qmmm_Ewald_setup_and_potential(VOLUME,XTLINV,X,Y,Z,CG,qcheck)
!!!     !
!!!     if(.not.qcheck) call wrndie(-5,'<MNDENE>','The CHARMM will stop at MNDENE/qmmm_Ewald_setup.')
!!!  end if
!!!
!!!  icall = 0
!!!  call scf_energy(natom,xim,yim,zim,icall)
!!!  !
!!!  ! if icall returned as 0, it means scf not converged.
!!!  if(icall.eq.-1) call wrndie(0,'<MNDENE>','The QM/MM energy is not converged.')
!!!
!!!  ! now, compute gradient components and copy them into the main dx/dy/dz arrays.
!!!  call scf_gradient(natom,xim,yim,zim,dx,dy,dz)
!!!
!!!  !
!!!#if KEY_PARALLEL==1
!!!  if(mynod.eq.0) then
!!!#endif
!!!     CTOT=qm_control_r%E_total
!!!#if KEY_PARALLEL==1
!!!  end if
!!!#endif
!!!
!!!  ! compute Kspace gradient contribution, but not added here.
!!!  ! added in subroutine KSPACE (ewalf.src)
!!!  if(mm_main_r%LQMEWD) then
!!!     ! do virial initialization here, (check ewaldf.src and pme.src)
!!!     if(.not. mm_main_r%PMEwald) EWVIRIAL2(1:9)=zero
!!!
!!!     ! 
!!!     call qmmm_Ewald_gradient(X,Y,Z,DX,DY,DZ,CG,EWVIRIAL2,qcheck)
!!!  end if
!!!
!!!  ! D3BJ & H4 corrections
!!!  if(dispers) then
!!!     if(.not.associated(xyz_disp))  allocate(xyz_disp(3,qm_main_r%numat))
!!!     if(.not.associated(g_dftd3))   allocate(g_dftd3(3,qm_main_r%numat))
!!!     e_dftd3        = zero
!!!     g_dftd3_norm   = zero
!!!!#if KEY_PARALLEL==1
!!!!     if(MYNOD.EQ.0) then
!!!!#endif
!!!        do i=1,qm_main_r%numat
!!!           do j=1,3
!!!              xyz_disp(j,i)= qm_main_r%qm_coord(j,i)*r_BOHRR
!!!              g_dftd3(j,i) = zero
!!!           end do
!!!        end do
!!!!!!        if(l_disp) then
!!!!!!           ! for periodic boundary condition
!!!!!!           ! in the present case, assuming that QM regions stay within the primary box.
!!!!!!           qperiodic =.false.   ! for now turn off this.
!!!!!!           if(mm_main_r%LQMEWD) then
!!!!!!              boxsiz(1,1)=XTLABC(1)*r_BOHRR
!!!!!!              boxsiz(1,2)=XTLABC(2)*r_BOHRR
!!!!!!              boxsiz(1,3)=XTLABC(4)*r_BOHRR
!!!!!!              boxsiz(2,1)=XTLABC(2)*r_BOHRR
!!!!!!              boxsiz(2,2)=XTLABC(3)*r_BOHRR
!!!!!!              boxsiz(2,3)=XTLABC(5)*r_BOHRR
!!!!!!              boxsiz(3,1)=XTLABC(4)*r_BOHRR
!!!!!!              boxsiz(3,2)=XTLABC(5)*r_BOHRR
!!!!!!              boxsiz(3,3)=XTLABC(6)*r_BOHRR
!!!!!!              nlat(1:3)  =0               ! for now, correct one refers gamma_summind routine.
!!!!!!           end if
!!!!!!           call dispersion_egr(qm_main_r%numat,xyz_disp,g_dftd3,boxsiz,nlat,qperiodic)
!!!!!!           !!!e_dftd3 = e_dftd3 + Edis  ! energy for dispersion
!!!!!!           CTOT = CTOT + (Edis)*TOKCAL
!!!!!!           do i=1,qm_main_r%numat
!!!!!!              n=qm_control_r%qminb(i)
!!!!!!              dx(n) = dx(n) + g_dftd3(1,i)*conv_grad
!!!!!!              dy(n) = dy(n) + g_dftd3(2,i)*conv_grad
!!!!!!              dz(n) = dz(n) + g_dftd3(3,i)*conv_grad
!!!!!!              g_dftd3(j,i) = zero ! for next calc.
!!!!!!           end do
!!!!!!        end if
!!!
!!!        lgrad =.true.
!!!        if (lmndod2 .or. lmndod3) then
!!!           ! Initialize energy.
!!!           if (lgrad) then
!!!              ! Use variables from current routine: natom,xim,yim,zim,dx,dy,dz
!!!              call calc_g_dftd3(xyz_disp, e_dftd3, g_dftd3, qm_main_r%numat)
!!!           else
!!!              call calc_e_dftd3(xyz_disp, e_dftd3, qm_main_r%numat)
!!!           end if
!!!           !conv_grad: H/Bohr to kcal/mol/A.  and energy: H to kcal/mol.
!!!           do i=1,qm_main_r%numat
!!!              n=qm_control_r%qminb(i)
!!!              dx(n) = dx(n) + g_dftd3(1,i)*conv_grad
!!!              dy(n) = dy(n) + g_dftd3(2,i)*conv_grad
!!!              dz(n) = dz(n) + g_dftd3(3,i)*conv_grad
!!!           end do
!!!#if KEY_PARALLEL==1
!!!     if(numnod>1) call gcomb(e_dftd3,1)
!!!     if(MYNOD == 0) then
!!!#endif
!!!           CTOT = CTOT + e_dftd3*TOKCAL
!!!#if KEY_PARALLEL==1
!!!     end if
!!!#endif
!!!        end if
!!!!#if KEY_PARALLEL==1
!!!!     end if
!!!!#endif
!!!  end if
!!!  ! End of D3BJ
!!!
!!!  ! H4 correction
!!!  if(q_h4corr) then
!!!     call h4_correction(EH4_corr,natom,dx,dy,dz)
!!!#if KEY_PARALLEL==1
!!!     if(MYNOD.EQ.0) then
!!!#endif
!!!        CTOT = CTOT + EH4_corr
!!!#if KEY_PARALLEL==1
!!!     end if
!!!#endif
!!!  end if
!!!
!!!  ! LEPS and SVB correction part
!!!  if(QLEPS) THEN
!!!     ! only do from the master node.
!!!#if KEY_PARALLEL==1
!!!     if(MYNOD.EQ.0) then
!!!#endif
!!!        ! assign coords
!!!        XLA(1) = X(NTA)
!!!        XLA(2) = Y(NTA)
!!!        XLA(3) = Z(NTA)
!!!        XLB(1) = X(NTB)
!!!        XLB(2) = Y(NTB)
!!!        XLB(3) = Z(NTB)
!!!        XLC(1) = X(NTC)
!!!        XLC(2) = Y(NTC)
!!!        XLC(3) = Z(NTC)
!!!        if (SVB_DIM .EQ. 2) then
!!!           XLD(1) = X(NTD)
!!!           XLD(2) = Y(NTD)
!!!           XLD(3) = Z(NTD)
!!!           XLE(1) = X(NTE)
!!!           XLE(2) = Y(NTE)
!!!           XLE(3) = Z(NTE)
!!!           XLF(1) = X(NTF)
!!!           XLF(2) = Y(NTF)
!!!           XLF(3) = Z(NTF)
!!!        end if
!!!     
!!!        ! call SVB energy term
!!!        if(QSVB) then
!!!           if(SVB_DIM .EQ. 1) then
!!!              call QM_SVB1D(E_LEPS,DA_LEPS_local,DB_LEPS_local,DC_LEPS_local)
!!!           else if(SVB_DIM .EQ. 2) then
!!!              call QM_SVB2D(E_LEPS,DA_LEPS_local,DB_LEPS_local,DC_LEPS_local, &
!!!                            DD_LEPS_local,DE_LEPS_local,DF_LEPS_local)
!!!           end if
!!!        else
!!!           call CORRECT_LEPS(E_LEPS,DA_LEPS_local,DB_LEPS_local,DC_LEPS_local)
!!!        end if
!!!   
!!!        ! add energy/gradients.
!!!        CTOT = CTOT + E_LEPS
!!!        DX(nta) = DX(nta) + DA_LEPS_local(1)
!!!        DY(nta) = DY(nta) + DA_LEPS_local(2)
!!!        DZ(nta) = DZ(nta) + DA_LEPS_local(3)
!!!
!!!        DX(ntb) = DX(ntb) + DB_LEPS_local(1)
!!!        DY(ntb) = DY(ntb) + DB_LEPS_local(2)
!!!        DZ(ntb) = DZ(ntb) + DB_LEPS_local(3)
!!!
!!!        DX(ntc) = DX(ntc) + DC_LEPS_local(1)
!!!        DY(ntc) = DY(ntc) + DC_LEPS_local(2)
!!!        DZ(ntc) = DZ(ntc) + DC_LEPS_local(3)
!!!
!!!        if(SVB_DIM .EQ. 2) then
!!!           DX(ntd) = DX(ntd) + DD_LEPS_local(1)
!!!           DY(ntd) = DY(ntd) + DD_LEPS_local(2)
!!!           DZ(ntd) = DZ(ntd) + DD_LEPS_local(3)
!!!
!!!           DX(nte) = DX(nte) + DE_LEPS_local(1)
!!!           DY(nte) = DY(nte) + DE_LEPS_local(2)
!!!           DZ(nte) = DZ(nte) + DE_LEPS_local(3)
!!!
!!!           if(.not.SVB_DE) then
!!!              DX(ntf) = DX(ntf) + DF_LEPS_local(1)
!!!              DY(ntf) = DY(ntf) + DF_LEPS_local(2)
!!!              DZ(ntf) = DZ(ntf) + DF_LEPS_local(3)
!!!           end if
!!!        end if
!!!#if KEY_PARALLEL==1
!!!     end if
!!!#endif
!!!  end if
!!!
!!!!!#if KEY_GHO==1
!!!  ! determined the GHO-atom derivatives.
!!!  if(qm_gho_info_r%q_gho) then
!!!     ECLASS = ZERO
!!!     !
!!!     ! treat RHF and UHF differently ... PJ 12/2002
!!!     call dqlink(DX,DY,DZ,X,Y,Z,CG,qm_gho_info_r%FAOA,qm_gho_info_r%PHO,ECLASS, &
!!!                 qm_gho_info_r%nqmlnk,qm_gho_info_r%norbao,                     &
!!!                 qm_gho_info_r%lin_norbao,qm_gho_info_r%mqm16,                  &
!!!                 qm_scf_main_r%indx,qm_gho_info_r%IQLINK,qm_gho_info_r%JQLINK,  &
!!!                 qm_gho_info_r%BT,qm_gho_info_r%BTM,qm_gho_info_r%DBTMMM,       &
!!!                 qm_gho_info_r%UHFGHO)
!!!     !
!!!#if KEY_PARALLEL==1
!!!     if(numnod>1) call gcomb(ECLASS,1)
!!!     if(mynod.eq.0) then 
!!!#endif
!!!        CTOT = CTOT + ECLASS
!!!        qm_control_r%E_total=qm_control_r%E_total+ECLASS  ! also add her
!!!#if KEY_PARALLEL==1
!!!     end if
!!!#endif
!!!     !
!!!     ! I am not sure, it will produce correct results for UHF case. Need to check
!!!     ! if we are going to use UHF.
!!!     if(qm_gho_info_r%uhfgho) then
!!!        ! Include the gradient correction term for alpha and beta FOCK
!!!        ! matrices seperately. The repulsion energy between MM atoms
!!!        ! linked to the GHO boundary is included when the alpha
!!!        ! correction term is included ... PJ 12/2002
!!!        call dqlink(DX,DY,DZ,X,Y,Z,CG,qm_gho_info_r%FAOB,qm_gho_info_r%PBHO,ECLASS, &
!!!                    qm_gho_info_r%nqmlnk,qm_gho_info_r%norbao,                      &
!!!                    qm_gho_info_r%lin_norbao,qm_gho_info_r%mqm16,                   &
!!!                    qm_scf_main_r%indx,qm_gho_info_r%IQLINK,qm_gho_info_r%JQLINK,   &
!!!                    qm_gho_info_r%BT,qm_gho_info_r%BTM,qm_gho_info_r%DBTMMM,        &
!!!                    qm_gho_info_r%UHFGHO)
!!!     end if
!!!  end if
!!!!!#endif 
!!!
!!!  return
!!!  END SUBROUTINE MNDENE
!!!!-----------------------------------------------------------------------

  SUBROUTINE MNDENE_STEP1(X,Y,Z)
  !-----------------------------------------------------------------------
  !
  ! First step in the preparation for the QM/MM energy/gradients evaluation.
  ! In fact, this is the first step in the splitting of MNDENE subroutine.
  !
  use chm_kinds
  use dimens_fcm
  use memory
  !
  use contrl
  use gamess_fcm,only : igmsel
  use inbnd
  use mndo97
  use mndgho
  ! use erfcd_mod,only: EWLDT
  use quantm, only : natom_check,xim,yim,zim
  use nbndqm_mod, only : imattq
  use psf
  use stream
  !
  use qm1_info, only : qm_control_r,qm_main_r,mm_main_r
  use qmmm_interface, only : fill_mm_coords,fill_dist_qm_mm_array
  use mndgho_module, only : qm_gho_info_r,HBDEF
  use mndnbnd_module, only: ch2mnd

  ! Adjust nonbonded group list for IMAGES and simple PBC.
  use image
#if KEY_PBOUND==1
  use pbound, only: qBoun
#endif

  implicit none

  real(chm_real) :: X(*),Y(*),Z(*)

  ! local variables
  INTEGER :: NTATOM,I,N
  LOGICAL :: QIMAGE,qfail

  ! return, if not setup qm/mm
  if(.not.qm_control_r%ifqnt) return

  ! initial check up
  qimage =.FALSE.
  ntatom = natom

#if KEY_PBOUND==1
  if(.not.qBoun) then
#endif
     if(ntrans.gt.0) then
        !if(lgroup) then
           qimage =.true.
        !else
        !   call wrndie(-1,'<MNDENE>', 'QM/MM do not interact with Images under Atom Based Cutoff.')
        !end if
     end if
#if KEY_PBOUND==1
  end if
#endif

  ! assign stack array for temporary coordinate
  if(allocated(xim).and. size(xim).lt.natom) then
     if(allocated(xim)) call chmdealloc('mndini.src','MNDENE_STEP1','xim',size(xim),crl=xim)
     if(allocated(yim)) call chmdealloc('mndini.src','MNDENE_STEP1','yim',size(yim),crl=yim)
     if(allocated(zim)) call chmdealloc('mndini.src','MNDENE_STEP1','zim',size(zim),crl=zim)
     natom_check = natom      ! for checking purpose
  end if
  if(.not.allocated(xim)) call chmalloc('mndini.src','MNDENE_STEP1','xim',natom,crl=xim)
  if(.not.allocated(yim)) call chmalloc('mndini.src','MNDENE_STEP1','yim',natom,crl=yim)
  if(.not.allocated(zim)) call chmalloc('mndini.src','MNDENE_STEP1','zim',natom,crl=zim)

  call SwapXYZ_image(natom,X,Y,Z,xim,yim,zim,imattq)
  
  ! update QM coordinates
  do i=1,qm_main_r%numat
     n=qm_control_r%qminb(i)
     qm_main_r%qm_coord(1,i)=X(n)
     qm_main_r%qm_coord(2,i)=Y(n)
     qm_main_r%qm_coord(3,i)=Z(n)
  end do

  !! update the hybridization matrix for GHO
  !if(qm_gho_info_r%q_gho) then
  !   qfail=.true.
  !   call hbdef(x,y,z,qm_gho_info_r%BT,qm_gho_info_r%BTM,qm_gho_info_r%DBTMMM, &
  !              qm_gho_info_r%nqmlnk,qm_gho_info_r%mqm16,                      &
  !              qm_gho_info_r%IQLINK,qm_gho_info_r%JQLINK,qm_gho_info_r%KQLINK,&
  !              qfail)
  !end if
  
  ! non-bond list and prepare for QM/MM-interaction list
  call ch2mnd(qm_main_r%numat,igmsel,xim,yim,zim,.false.)
  !
  !! copy mm coords for qm/mm calculations
  !call fill_mm_coords(natom,xim,yim,zim,cg, &
  !                    mm_main_r%mm_coord,mm_main_r%mm_chrgs, &
  !                    mm_main_r%qm_mm_pair_list,qm_control_r%mminb1)
  !
  !! incore preparation.
  !if(qm_main_r%rij_qm_incore .or. mm_main_r%rij_mm_incore) then
  !   call fill_dist_qm_mm_array(qm_main_r%numat,mm_main_r%numatm,      &
  !                              qm_main_r%qm_coord,mm_main_r%mm_coord, &
  !                              mm_main_r%LQMEWD)
  !end if
  !

  return
  END SUBROUTINE MNDENE_STEP1
!-----------------------------------------------------------------------

  SUBROUTINE MNDENE_STEP2(X,Y,Z)
  !-----------------------------------------------------------------------
  !
  ! Second step in the preparation for the QM/MM energy/gradients evaluation.
  ! In fact, this is the second step in the splitting of MNDENE subroutine.
  !
  use chm_kinds
  !use gamess_fcm,only : igmsel
  use quantm, only : xim,yim,zim
  use psf,only: natom, cg
  use stream
  use prssre, only: getvol
  use image, only: xtlabc

  use qm1_info, only : qm_control_r,mm_main_r,qm_main_r
  use qmmm_interface, only : scf_energy_prep,qmmm_Ewald_setup_and_potential, &
                             fill_mm_coords,fill_dist_qm_mm_array
  use mndgho_module, only : qm_gho_info_r,HBDEF
  !use mndnbnd_module, only: ch2mnd

  implicit none
  real(chm_real) :: X(*),Y(*),Z(*)

  ! local variables
  integer       :: icall
  real(chm_real):: XTLINV(6),volume
  logical       :: qcheck,OK,qfail

  ! update the hybridization matrix for GHO
  if(qm_gho_info_r%q_gho) then
     qfail=.true.
     call hbdef(x,y,z,qm_gho_info_r%BT,qm_gho_info_r%BTM,qm_gho_info_r%DBTMMM, &
                qm_gho_info_r%nqmlnk,qm_gho_info_r%mqm16,                      &
                qm_gho_info_r%IQLINK,qm_gho_info_r%JQLINK,qm_gho_info_r%KQLINK,&
                qfail)
  end if

  !! non-bond list and prepare for QM/MM-interaction list
  !call ch2mnd(qm_main_r%numat,igmsel,xim,yim,zim,.false.)

  ! copy mm coords for qm/mm calculations
  call fill_mm_coords(natom,xim,yim,zim,cg, &
                      mm_main_r%mm_coord,mm_main_r%mm_chrgs, &
                      mm_main_r%qm_mm_pair_list,qm_control_r%mminb1)

  ! incore preparation.
  if(qm_main_r%rij_qm_incore .or. mm_main_r%rij_mm_incore) then
     call fill_dist_qm_mm_array(qm_main_r%numat,mm_main_r%numatm,      &
                                qm_main_r%qm_coord,mm_main_r%mm_coord, &
                                mm_main_r%LQMEWD)
  end if
 
  if(mm_main_r%LQMEWD) then
     ! get the volume and reciprocal space lattice vector
     call getvol(volume)
     call INVT33S(xtlinv,xtlabc,ok)
   
     ! setup Ktable and Kvec
     qcheck=.true.
     call qmmm_Ewald_setup_and_potential(VOLUME,XTLINV,X,Y,Z,CG,qcheck)
     !
     if(.not.qcheck) call wrndie(-5,'<MNDENE_STEP2>','The CHARMM will stop at MNDENE/qmmm_Ewald_setup.')
  end if

  icall = 0
  call scf_energy_prep(natom,xim,yim,zim)

  return
  END SUBROUTINE MNDENE_STEP2
!-----------------------------------------------------------------------

  SUBROUTINE MNDENE_STEP3(CTOT,X,Y,Z,DX,DY,DZ)
  !-----------------------------------------------------------------------
  !
  ! Third step in the preparation for the QM/MM energy/gradients evaluation.
  ! In fact, this is the third step in the splitting of MNDENE subroutine.
  !
  ! For the gradient part, it will be split into two sections for OpenMP.
  !
  use chm_kinds
  use number,only: zero
#if KEY_PARALLEL==1
  use parallel,only: mynod,numnod
#endif
  use psf,only: natom,cg
  use stream
  use quantm, only : xim,yim,zim
  use qm1_info, only : qm_control_r,qm_scf_main_r
  use qmmm_interface, only : scf_energy
!  use mndgho_module, only : qm_gho_info_r,DQLINK
!  use leps

  implicit none
  real(chm_real) :: CTOT,X(*),Y(*),Z(*),DX(*),DY(*),DZ(*)

  ! local variables
  integer :: icall
!  real(chm_real) :: ECLASS
!  
!  ! for leps/svb correction terms
!  real(chm_real) :: E_LEPS,DA_LEPS_local(3),DB_LEPS_local(3),DC_LEPS_local(3), &
!                    DD_LEPS_local(3),DE_LEPS_local(3),DF_LEPS_local(3)

  ! here, now call energy related routines.
  icall = 0
  call scf_energy(natom,xim,yim,zim,icall)
  !
  ! if icall returned as 0, it means scf not converged.
  if(icall.eq.-1) call wrndie(0,'<MNDENE_STEP3>','The QM/MM energy is not converged.')

  !
#if KEY_PARALLEL==1
  if(mynod.eq.0) then
#endif
     CTOT=qm_control_r%E_total
#if KEY_PARALLEL==1
  end if
#endif

!  ! LEPS and SVB correction part
!  if(QLEPS) THEN
!     ! only do from the master node.
!#if KEY_PARALLEL==1
!     if(MYNOD.EQ.0) then
!#endif
!        ! assign coords
!        XLA(1) = X(NTA)
!        XLA(2) = Y(NTA)
!        XLA(3) = Z(NTA)
!        XLB(1) = X(NTB)
!        XLB(2) = Y(NTB)
!        XLB(3) = Z(NTB)
!        XLC(1) = X(NTC)
!        XLC(2) = Y(NTC)
!        XLC(3) = Z(NTC)
!        if (SVB_DIM .EQ. 2) then
!           XLD(1) = X(NTD)
!           XLD(2) = Y(NTD)
!           XLD(3) = Z(NTD)
!           XLE(1) = X(NTE)
!           XLE(2) = Y(NTE)
!           XLE(3) = Z(NTE)
!           XLF(1) = X(NTF)
!           XLF(2) = Y(NTF)
!           XLF(3) = Z(NTF)
!        end if
!     
!        ! call SVB energy term
!        if(QSVB) then
!           if(SVB_DIM .EQ. 1) then
!              call QM_SVB1D(E_LEPS,DA_LEPS_local,DB_LEPS_local,DC_LEPS_local)
!           else if(SVB_DIM .EQ. 2) then
!              call QM_SVB2D(E_LEPS,DA_LEPS_local,DB_LEPS_local,DC_LEPS_local, &
!                            DD_LEPS_local,DE_LEPS_local,DF_LEPS_local)
!           end if
!        else
!           call CORRECT_LEPS(E_LEPS,DA_LEPS_local,DB_LEPS_local,DC_LEPS_local)
!        end if
!   
!        ! add energy/gradients.
!        CTOT = CTOT + E_LEPS
!        DX(nta) = DX(nta) + DA_LEPS_local(1)
!        DY(nta) = DY(nta) + DA_LEPS_local(2)
!        DZ(nta) = DZ(nta) + DA_LEPS_local(3)
!
!        DX(ntb) = DX(ntb) + DB_LEPS_local(1)
!        DY(ntb) = DY(ntb) + DB_LEPS_local(2)
!        DZ(ntb) = DZ(ntb) + DB_LEPS_local(3)
!
!        DX(ntc) = DX(ntc) + DC_LEPS_local(1)
!        DY(ntc) = DY(ntc) + DC_LEPS_local(2)
!        DZ(ntc) = DZ(ntc) + DC_LEPS_local(3)
!
!        if(SVB_DIM .EQ. 2) then
!           DX(ntd) = DX(ntd) + DD_LEPS_local(1)
!           DY(ntd) = DY(ntd) + DD_LEPS_local(2)
!           DZ(ntd) = DZ(ntd) + DD_LEPS_local(3)
!
!           DX(nte) = DX(nte) + DE_LEPS_local(1)
!           DY(nte) = DY(nte) + DE_LEPS_local(2)
!           DZ(nte) = DZ(nte) + DE_LEPS_local(3)
!
!           if(.not.SVB_DE) then
!              DX(ntf) = DX(ntf) + DF_LEPS_local(1)
!              DY(ntf) = DY(ntf) + DF_LEPS_local(2)
!              DZ(ntf) = DZ(ntf) + DF_LEPS_local(3)
!           end if
!        end if
!#if KEY_PARALLEL==1
!     end if
!#endif
!  end if
!
!  ! determined the GHO-atom derivatives.
!  if(qm_gho_info_r%q_gho) then
!     ECLASS = ZERO
!     !
!     ! treat RHF and UHF differently ... PJ 12/2002
!     call dqlink(DX,DY,DZ,X,Y,Z,CG,qm_gho_info_r%FAOA,qm_gho_info_r%PHO,ECLASS, &
!                 qm_gho_info_r%nqmlnk,qm_gho_info_r%norbao,                     &
!                 qm_gho_info_r%lin_norbao,qm_gho_info_r%mqm16,                  &
!                 qm_scf_main_r%indx,qm_gho_info_r%IQLINK,qm_gho_info_r%JQLINK,  &
!                 qm_gho_info_r%BT,qm_gho_info_r%BTM,qm_gho_info_r%DBTMMM,       &
!                 qm_gho_info_r%UHFGHO)
!#if KEY_PARALLEL==1
!     if(numnod>1) call gcomb(ECLASS,1)
!     if(mynod.eq.0) then
!#endif
!        CTOT = CTOT + ECLASS
!        qm_control_r%E_total=qm_control_r%E_total+ECLASS  ! also add her
!#if KEY_PARALLEL==1
!     end if
!#endif
!     !
!     ! I am not sure, it will produce correct results for UHF case. Need to check
!     ! if we are going to use UHF.
!     if(qm_gho_info_r%uhfgho) then
!        ! Include the gradient correction term for alpha and beta FOCK
!        ! matrices seperately. The repulsion energy between MM atoms
!        ! linked to the GHO boundary is included when the alpha
!        ! correction term is included ... PJ 12/2002
!        call dqlink(DX,DY,DZ,X,Y,Z,CG,qm_gho_info_r%FAOB,qm_gho_info_r%PBHO,ECLASS, &
!                    qm_gho_info_r%nqmlnk,qm_gho_info_r%norbao,                      &
!                    qm_gho_info_r%lin_norbao,qm_gho_info_r%mqm16,                   &
!                    qm_scf_main_r%indx,qm_gho_info_r%IQLINK,qm_gho_info_r%JQLINK,   &
!                    qm_gho_info_r%BT,qm_gho_info_r%BTM,qm_gho_info_r%DBTMMM,        &
!                    qm_gho_info_r%UHFGHO)
!     end if
!  end if
  !
  return
  END SUBROUTINE MNDENE_STEP3
!-----------------------------------------------------------------------

  SUBROUTINE MNDENE_STEP4(CTOT,X,Y,Z,dx,dy,dz)
  !-----------------------------------------------------------------------
  !
  ! Third step in the preparation for the QM/MM energy/gradients evaluation.
  ! In fact, this is the third step in the splitting of MNDENE subroutine.
  !
  ! For the gradient part, it will be split into two sections for OpenMP.
  !
  use chm_kinds
  use number,only: zero,one
#if KEY_PARALLEL==1
  use parallel,only: mynod, numnod, gcomb
#endif
  use psf,only: natom,cg
  use ewald_1m, only: EWVIRIAL2
  use quantm, only : xim,yim,zim
  use qm1_info, only : mm_main_r,qm_control_r,qm_scf_main_r,qm_main_r
  use qmmm_interface, only : scf_gradient,qmmm_Ewald_gradient
  use mndgho_module, only : qm_gho_info_r,DQLINK
  use leps

  ! D3BJ & H4 corrections
  use mndo97, only: dispers,lmndod2,lmndod3 ! ,Edis,l_disp
  use consta, only: BOHRR,AU_TO_EV,TOKCAL
  use dftd3_mndo, only : calc_e_dftd3,calc_g_dftd3
  use image, only : xtlabc
  ! h4 correction related
  use H4_mndo, only : q_h4corr,h4_correction

  implicit none
  real(chm_real) :: CTOT,X(*),Y(*),Z(*),dx(*),dy(*),dz(*)

  ! local variables
  logical:: qcheck
  real(chm_real) :: ECLASS,EH4_corr

  ! for leps/svb correction terms
  real(chm_real) :: E_LEPS,DA_LEPS_local(3),DB_LEPS_local(3),DC_LEPS_local(3), &
                    DD_LEPS_local(3),DE_LEPS_local(3),DF_LEPS_local(3)

  ! D3BJ & H4 corrections
  ! Dispersion gradient contribution. Defined this way to
  ! fit into the wrapper for Grimmes code.
  real(chm_real)         :: e_dftd3, g_dftd3_norm
  real(chm_real),pointer :: xyz_disp(:,:)=>Null(), &  ! 3,numat
                            g_dftd3(:,:)=>Null()      ! 3,numat
  real(chm_real),parameter :: r_BOHRR = one/BOHRR, &  ! 1/bohrr
                              r_atoev=one/AU_TO_EV,&
                              conv_grad=TOKCAL/BOHRR
  real(chm_real):: boxsiz(3,3), e_disp_local
  integer       :: nlat(3)
  logical       :: qperiodic,lgrad
  logical, save :: qdftd3_first=.true.
  integer       :: i,j,n

  !
  ! LEPS and SVB correction part
  if(QLEPS) THEN
     ! only do from the master node.
#if KEY_PARALLEL==1
     if(MYNOD.EQ.0) then
#endif
        ! assign coords
        XLA(1) = X(NTA)
        XLA(2) = Y(NTA)
        XLA(3) = Z(NTA)
        XLB(1) = X(NTB)
        XLB(2) = Y(NTB)
        XLB(3) = Z(NTB)
        XLC(1) = X(NTC)
        XLC(2) = Y(NTC)
        XLC(3) = Z(NTC)
        if (SVB_DIM .EQ. 2) then
           XLD(1) = X(NTD)
           XLD(2) = Y(NTD)
           XLD(3) = Z(NTD)
           XLE(1) = X(NTE)
           XLE(2) = Y(NTE)
           XLE(3) = Z(NTE)
           XLF(1) = X(NTF)
           XLF(2) = Y(NTF)
           XLF(3) = Z(NTF)
        end if
        ! call SVB energy term
        if(QSVB) then
           if(SVB_DIM .EQ. 1) then
              call QM_SVB1D(E_LEPS,DA_LEPS_local,DB_LEPS_local,DC_LEPS_local)
           else if(SVB_DIM .EQ. 2) then
              call QM_SVB2D(E_LEPS,DA_LEPS_local,DB_LEPS_local,DC_LEPS_local, &
                            DD_LEPS_local,DE_LEPS_local,DF_LEPS_local)
           end if
        else
           call CORRECT_LEPS(E_LEPS,DA_LEPS_local,DB_LEPS_local,DC_LEPS_local)
        end if

        ! add energy/gradients.
        CTOT = CTOT + E_LEPS
        DX(nta) = DX(nta) + DA_LEPS_local(1)
        DY(nta) = DY(nta) + DA_LEPS_local(2)
        DZ(nta) = DZ(nta) + DA_LEPS_local(3)

        DX(ntb) = DX(ntb) + DB_LEPS_local(1)
        DY(ntb) = DY(ntb) + DB_LEPS_local(2)
        DZ(ntb) = DZ(ntb) + DB_LEPS_local(3)

        DX(ntc) = DX(ntc) + DC_LEPS_local(1)
        DY(ntc) = DY(ntc) + DC_LEPS_local(2)
        DZ(ntc) = DZ(ntc) + DC_LEPS_local(3)

        if(SVB_DIM .EQ. 2) then
           DX(ntd) = DX(ntd) + DD_LEPS_local(1)
           DY(ntd) = DY(ntd) + DD_LEPS_local(2)
           DZ(ntd) = DZ(ntd) + DD_LEPS_local(3)

           DX(nte) = DX(nte) + DE_LEPS_local(1)
           DY(nte) = DY(nte) + DE_LEPS_local(2)
           DZ(nte) = DZ(nte) + DE_LEPS_local(3)

           if(.not.SVB_DE) then
              DX(ntf) = DX(ntf) + DF_LEPS_local(1)
              DY(ntf) = DY(ntf) + DF_LEPS_local(2)
              DZ(ntf) = DZ(ntf) + DF_LEPS_local(3)
           end if
        end if
#if KEY_PARALLEL==1
     end if
#endif
  end if

  ! determined the GHO-atom derivatives.
  if(qm_gho_info_r%q_gho) then
     ECLASS = ZERO
     !
     ! treat RHF and UHF differently ... PJ 12/2002
     call dqlink(DX,DY,DZ,X,Y,Z,CG,qm_gho_info_r%FAOA,qm_gho_info_r%PHO,ECLASS, &
                 qm_gho_info_r%nqmlnk,qm_gho_info_r%norbao,                     &
                 qm_gho_info_r%lin_norbao,qm_gho_info_r%mqm16,                  &
                 qm_scf_main_r%indx,qm_gho_info_r%IQLINK,qm_gho_info_r%JQLINK,  &
                 qm_gho_info_r%BT,qm_gho_info_r%BTM,qm_gho_info_r%DBTMMM,       &
                 qm_gho_info_r%UHFGHO)
#if KEY_PARALLEL==1
     if(numnod>1) call gcomb(ECLASS,1)
     if(mynod.eq.0) then
#endif
        CTOT = CTOT + ECLASS
        qm_control_r%E_total=qm_control_r%E_total+ECLASS  ! also add her
#if KEY_PARALLEL==1
     end if
#endif
     !
     ! I am not sure, it will produce correct results for UHF case. Need to check
     ! if we are going to use UHF.
     if(qm_gho_info_r%uhfgho) then
        ! Include the gradient correction term for alpha and beta FOCK
        ! matrices seperately. The repulsion energy between MM atoms
        ! linked to the GHO boundary is included when the alpha
        ! correction term is included ... PJ 12/2002
        call dqlink(DX,DY,DZ,X,Y,Z,CG,qm_gho_info_r%FAOB,qm_gho_info_r%PBHO,ECLASS, &
                    qm_gho_info_r%nqmlnk,qm_gho_info_r%norbao,                      &
                    qm_gho_info_r%lin_norbao,qm_gho_info_r%mqm16,                   &
                    qm_scf_main_r%indx,qm_gho_info_r%IQLINK,qm_gho_info_r%JQLINK,   &
                    qm_gho_info_r%BT,qm_gho_info_r%BTM,qm_gho_info_r%DBTMMM,        &
                    qm_gho_info_r%UHFGHO)
     end if
  end if

  ! now, compute gradient components and copy them into the main dx/dy/dz arrays.
  call scf_gradient(natom,xim,yim,zim,dx,dy,dz)
  !
  ! compute Kspace gradient contribution, but not added here.
  ! added in subroutine KSPACE (ewalf.src)
  if(mm_main_r%LQMEWD) then
     ! do virial initialization here, (check ewaldf.src and pme.src)
     if(.not. mm_main_r%PMEwald) EWVIRIAL2(1:9)=zero

     !
     call qmmm_Ewald_gradient(X,Y,Z,dx,dy,dz,CG,EWVIRIAL2,qcheck)
  end if
  !
  ! D3BJ & H4 corrections
  if(dispers) then
     if(.not.associated(xyz_disp))  allocate(xyz_disp(3,qm_main_r%numat))
     if(.not.associated(g_dftd3))   allocate(g_dftd3(3,qm_main_r%numat))
     e_dftd3        = zero
     g_dftd3_norm   = zero
!#if KEY_PARALLEL==1
!     if(MYNOD.EQ.0) then
!#endif
        do i=1,qm_main_r%numat
           do j=1,3
              xyz_disp(j,i)= qm_main_r%qm_coord(j,i)*r_BOHRR
              g_dftd3(j,i) = zero
           end do
        end do
!!!        if(l_disp) then
!!!           ! for periodic boundary condition
!!!           ! in the present case, assuming that QM regions stay within the primary box.
!!!           qperiodic =.false.   ! for now turn off this.
!!!           if(mm_main_r%LQMEWD) then
!!!              boxsiz(1,1)=XTLABC(1)*r_BOHRR
!!!              boxsiz(1,2)=XTLABC(2)*r_BOHRR
!!!              boxsiz(1,3)=XTLABC(4)*r_BOHRR
!!!              boxsiz(2,1)=XTLABC(2)*r_BOHRR
!!!              boxsiz(2,2)=XTLABC(3)*r_BOHRR
!!!              boxsiz(2,3)=XTLABC(5)*r_BOHRR
!!!              boxsiz(3,1)=XTLABC(4)*r_BOHRR
!!!              boxsiz(3,2)=XTLABC(5)*r_BOHRR
!!!              boxsiz(3,3)=XTLABC(6)*r_BOHRR
!!!              nlat(1:3)  =0               ! for now, correct one refers gamma_summind routine.
!!!           end if
!!!           call dispersion_egr(qm_main_r%numat,xyz_disp,g_dftd3,boxsiz,nlat,qperiodic)
!!!           !!!e_dftd3 = e_dftd3 + Edis  ! energy for dispersion
!!!           CTOT = CTOT + (Edis)*TOKCAL
!!!           do i=1,qm_main_r%numat
!!!              n=qm_control_r%qminb(i)
!!!              dx(n) = dx(n) + g_dftd3(1,i)*conv_grad
!!!              dy(n) = dy(n) + g_dftd3(2,i)*conv_grad
!!!              dz(n) = dz(n) + g_dftd3(3,i)*conv_grad
!!!              g_dftd3(j,i) = zero ! for next calc.
!!!           end do
!!!        end if

        lgrad =.true.
        if (lmndod2 .or. lmndod3) then
           ! Initialize energy.
           if (lgrad) then
              ! Use variables from current routine: natom,xim,yim,zim,dx,dy,dz
              call calc_g_dftd3(xyz_disp, e_dftd3, g_dftd3, qm_main_r%numat)
           else
              call calc_e_dftd3(xyz_disp, e_dftd3, qm_main_r%numat)
           end if
           !conv_grad: H/Bohr to kcal/mol/A.  and energy: H to kcal/mol.
           do i=1,qm_main_r%numat
              n=qm_control_r%qminb(i)
              dx(n) = dx(n) + g_dftd3(1,i)*conv_grad
              dy(n) = dy(n) + g_dftd3(2,i)*conv_grad
              dz(n) = dz(n) + g_dftd3(3,i)*conv_grad
              ! debug
              !write(6,*) g_dftd3(1,i)*conv_grad,g_dftd3(2,i)*conv_grad,g_dftd3(3,i)*conv_grad
           end do
#if KEY_PARALLEL==1
     if(numnod>1) call gcomb(e_dftd3,1)
     if(MYNOD == 0) then
#endif
           CTOT = CTOT + e_dftd3*TOKCAL
           ! debug
           !write(6,'(A,F13.8)') 'D3BJ energy:',e_dftd3*TOKCAL
#if KEY_PARALLEL==1
     end if
#endif
        end if
!#if KEY_PARALLEL==1
!     end if
!#endif
  end if
  ! End of D3BJ

  ! H4 correction
  if(q_h4corr) then
     call h4_correction(EH4_corr,natom,dx,dy,dz)
#if KEY_PARALLEL==1
     if(MYNOD.EQ.0) then
#endif
        CTOT = CTOT + EH4_corr
        ! debug
        !write(6,'(A,F13.8)') 'H4 energy:',EH4_corr
#if KEY_PARALLEL==1
     end if
#endif
  end if
  !
  return
  END SUBROUTINE MNDENE_STEP4
!-----------------------------------------------------------------------

  SUBROUTINE MNDENE_STEP4_1(X,Y,Z,dx,dy,dz)
  !-----------------------------------------------------------------------
  !
  ! Third step in the preparation for the QM/MM energy/gradients evaluation.
  ! In fact, this is the third step in the splitting of MNDENE subroutine.
  !
  ! For the gradient part, it will be split into two sections for OpenMP.
  !
  use chm_kinds
!  use number,only: zero
!#if KEY_PARALLEL==1
!  use parallel,only: mynod,numnod
!#endif
  use psf,only: natom !,cg
  use quantm, only : xim,yim,zim
!  use qm1_info, only : mm_main_r,qm_control_r,qm_scf_main_r
  use qmmm_interface, only : scf_gradient ! ,qmmm_Ewald_gradient

  implicit none
  real(chm_real) :: X(*),Y(*),Z(*),dx(*),dy(*),dz(*)

!  ! local variables
!  logical:: qcheck

  ! now, compute gradient components and copy them into the main dx/dy/dz arrays.
  call scf_gradient(natom,xim,yim,zim,dx,dy,dz)
  !
  ! compute Kspace gradient contribution, but not added here.
  ! added in subroutine KSPACE (ewalf.src)
  !if(mm_main_r%LQMEWD) then
  !   ! do virial initialization here, (check ewaldf.src and pme.src)
  !   if(.not. mm_main_r%PMEwald) EWVIRIAL2(1:9)=zero
  !
  !   !
  !   call qmmm_Ewald_gradient(X,Y,Z,dx,dy,dz,CG,EWVIRIAL2,qcheck)
  !end if
  !
  return
  END SUBROUTINE MNDENE_STEP4_1
!-----------------------------------------------------------------------

  SUBROUTINE MNDENE_STEP4_2(CTOT,X,Y,Z,dx,dy,dz)
  !-----------------------------------------------------------------------
  !
  ! Third step in the preparation for the QM/MM energy/gradients evaluation.
  ! In fact, this is the third step in the splitting of MNDENE subroutine.
  !
  ! For the gradient part, it will be split into two sections for OpenMP.
  !
  use chm_kinds
  use number,only: zero,one
#if KEY_PARALLEL==1
  use parallel,only: mynod, numnod, gcomb
#endif
  use psf,only: natom,cg
  use ewald_1m, only: EWVIRIAL2
  use quantm, only : xim,yim,zim
  use qm1_info, only : mm_main_r,qm_control_r,qm_scf_main_r,qm_main_r
  use qmmm_interface, only : qmmm_Ewald_gradient !,scf_gradient
  use mndgho_module, only : qm_gho_info_r,DQLINK
  use leps
  ! D3BJ & H4 corrections
  use mndo97, only: dispers,lmndod2,lmndod3 ! ,Edis,l_disp
  use consta, only: BOHRR,AU_TO_EV,TOKCAL
  use dftd3_mndo, only : calc_e_dftd3,calc_g_dftd3
  use image, only : xtlabc
  ! h4 correction related
  use H4_mndo, only : q_h4corr,h4_correction

  implicit none
  real(chm_real) :: CTOT,X(*),Y(*),Z(*),dx(*),dy(*),dz(*)

  ! local variables
  logical:: qcheck
  real(chm_real) :: ECLASS,EH4_corr

  ! for leps/svb correction terms
  real(chm_real) :: E_LEPS,DA_LEPS_local(3),DB_LEPS_local(3),DC_LEPS_local(3), &
                    DD_LEPS_local(3),DE_LEPS_local(3),DF_LEPS_local(3)
  ! D3BJ & H4 corrections
  ! Dispersion gradient contribution. Defined this way to 
  ! fit into the wrapper for Grimmes code.
  real(chm_real)         :: e_dftd3, g_dftd3_norm
  real(chm_real),pointer :: xyz_disp(:,:)=>Null(), &  ! 3,numat
                            g_dftd3(:,:)=>Null()      ! 3,numat
  real(chm_real),parameter :: r_BOHRR = one/BOHRR, &  ! 1/bohrr
                              r_atoev=one/AU_TO_EV,&
                              conv_grad=TOKCAL/BOHRR
  real(chm_real):: boxsiz(3,3), e_disp_local
  integer       :: nlat(3)
  logical       :: qperiodic,lgrad
  logical, save :: qdftd3_first=.true.
  integer       :: i,j,n

  !
  ! LEPS and SVB correction part
  if(QLEPS) THEN
     ! only do from the master node.
#if KEY_PARALLEL==1
     if(MYNOD.EQ.0) then
#endif
        ! assign coords
        XLA(1) = X(NTA)
        XLA(2) = Y(NTA)
        XLA(3) = Z(NTA)
        XLB(1) = X(NTB)
        XLB(2) = Y(NTB)
        XLB(3) = Z(NTB)
        XLC(1) = X(NTC)
        XLC(2) = Y(NTC)
        XLC(3) = Z(NTC)
        if (SVB_DIM .EQ. 2) then
           XLD(1) = X(NTD)
           XLD(2) = Y(NTD)
           XLD(3) = Z(NTD)
           XLE(1) = X(NTE)
           XLE(2) = Y(NTE)
           XLE(3) = Z(NTE)
           XLF(1) = X(NTF)
           XLF(2) = Y(NTF)
           XLF(3) = Z(NTF)
        end if
        ! call SVB energy term
        if(QSVB) then
           if(SVB_DIM .EQ. 1) then
              call QM_SVB1D(E_LEPS,DA_LEPS_local,DB_LEPS_local,DC_LEPS_local)
           else if(SVB_DIM .EQ. 2) then
              call QM_SVB2D(E_LEPS,DA_LEPS_local,DB_LEPS_local,DC_LEPS_local, &
                            DD_LEPS_local,DE_LEPS_local,DF_LEPS_local)
           end if
        else
           call CORRECT_LEPS(E_LEPS,DA_LEPS_local,DB_LEPS_local,DC_LEPS_local)
        end if

        ! add energy/gradients.
        CTOT = CTOT + E_LEPS
        DX(nta) = DX(nta) + DA_LEPS_local(1)
        DY(nta) = DY(nta) + DA_LEPS_local(2)
        DZ(nta) = DZ(nta) + DA_LEPS_local(3)

        DX(ntb) = DX(ntb) + DB_LEPS_local(1)
        DY(ntb) = DY(ntb) + DB_LEPS_local(2)
        DZ(ntb) = DZ(ntb) + DB_LEPS_local(3)

        DX(ntc) = DX(ntc) + DC_LEPS_local(1)
        DY(ntc) = DY(ntc) + DC_LEPS_local(2)
        DZ(ntc) = DZ(ntc) + DC_LEPS_local(3)

        if(SVB_DIM .EQ. 2) then
           DX(ntd) = DX(ntd) + DD_LEPS_local(1)
           DY(ntd) = DY(ntd) + DD_LEPS_local(2)
           DZ(ntd) = DZ(ntd) + DD_LEPS_local(3)

           DX(nte) = DX(nte) + DE_LEPS_local(1)
           DY(nte) = DY(nte) + DE_LEPS_local(2)
           DZ(nte) = DZ(nte) + DE_LEPS_local(3)

           if(.not.SVB_DE) then
              DX(ntf) = DX(ntf) + DF_LEPS_local(1)
              DY(ntf) = DY(ntf) + DF_LEPS_local(2)
              DZ(ntf) = DZ(ntf) + DF_LEPS_local(3)
           end if
        end if
#if KEY_PARALLEL==1
     end if
#endif
  end if

  ! determined the GHO-atom derivatives.
  if(qm_gho_info_r%q_gho) then
     ECLASS = ZERO
     !
     ! treat RHF and UHF differently ... PJ 12/2002
     call dqlink(DX,DY,DZ,X,Y,Z,CG,qm_gho_info_r%FAOA,qm_gho_info_r%PHO,ECLASS, &
                 qm_gho_info_r%nqmlnk,qm_gho_info_r%norbao,                     &
                 qm_gho_info_r%lin_norbao,qm_gho_info_r%mqm16,                  &
                 qm_scf_main_r%indx,qm_gho_info_r%IQLINK,qm_gho_info_r%JQLINK,  &
                 qm_gho_info_r%BT,qm_gho_info_r%BTM,qm_gho_info_r%DBTMMM,       &
                 qm_gho_info_r%UHFGHO)
#if KEY_PARALLEL==1
     if(numnod>1) call gcomb(ECLASS,1)
     if(mynod.eq.0) then
#endif
        CTOT = CTOT + ECLASS
        qm_control_r%E_total=qm_control_r%E_total+ECLASS  ! also add her
#if KEY_PARALLEL==1
     end if
#endif
     !
     ! I am not sure, it will produce correct results for UHF case. Need to check
     ! if we are going to use UHF.
     if(qm_gho_info_r%uhfgho) then
        ! Include the gradient correction term for alpha and beta FOCK
        ! matrices seperately. The repulsion energy between MM atoms
        ! linked to the GHO boundary is included when the alpha
        ! correction term is included ... PJ 12/2002
        call dqlink(DX,DY,DZ,X,Y,Z,CG,qm_gho_info_r%FAOB,qm_gho_info_r%PBHO,ECLASS, &
                    qm_gho_info_r%nqmlnk,qm_gho_info_r%norbao,                      &
                    qm_gho_info_r%lin_norbao,qm_gho_info_r%mqm16,                   &
                    qm_scf_main_r%indx,qm_gho_info_r%IQLINK,qm_gho_info_r%JQLINK,   &
                    qm_gho_info_r%BT,qm_gho_info_r%BTM,qm_gho_info_r%DBTMMM,        &
                    qm_gho_info_r%UHFGHO)
     end if
  end if

  ! now, compute gradient components and copy them into the main dx/dy/dz arrays.
  !call scf_gradient(natom,xim,yim,zim,dx,dy,dz)
  !
  ! compute Kspace gradient contribution, but not added here.
  ! added in subroutine KSPACE (ewalf.src)
  if(mm_main_r%LQMEWD) then
     ! do virial initialization here, (check ewaldf.src and pme.src)
     if(.not. mm_main_r%PMEwald) EWVIRIAL2(1:9)=zero

     !
     call qmmm_Ewald_gradient(X,Y,Z,dx,dy,dz,CG,EWVIRIAL2,qcheck)
  end if
  !
  ! D3BJ & H4 corrections
  if(dispers) then
     if(.not.associated(xyz_disp))  allocate(xyz_disp(3,qm_main_r%numat))
     if(.not.associated(g_dftd3))   allocate(g_dftd3(3,qm_main_r%numat))
     e_dftd3        = zero
     g_dftd3_norm   = zero
!#if KEY_PARALLEL==1
!     if(MYNOD.EQ.0) then
!#endif
        do i=1,qm_main_r%numat
           do j=1,3
              xyz_disp(j,i)= qm_main_r%qm_coord(j,i)*r_BOHRR
              g_dftd3(j,i) = zero
           end do
        end do
!!!        if(l_disp) then
!!!           ! for periodic boundary condition
!!!           ! in the present case, assuming that QM regions stay within the primary box.
!!!           qperiodic =.false.   ! for now turn off this.
!!!           if(mm_main_r%LQMEWD) then
!!!              boxsiz(1,1)=XTLABC(1)*r_BOHRR
!!!              boxsiz(1,2)=XTLABC(2)*r_BOHRR
!!!              boxsiz(1,3)=XTLABC(4)*r_BOHRR
!!!              boxsiz(2,1)=XTLABC(2)*r_BOHRR
!!!              boxsiz(2,2)=XTLABC(3)*r_BOHRR
!!!              boxsiz(2,3)=XTLABC(5)*r_BOHRR
!!!              boxsiz(3,1)=XTLABC(4)*r_BOHRR
!!!              boxsiz(3,2)=XTLABC(5)*r_BOHRR
!!!              boxsiz(3,3)=XTLABC(6)*r_BOHRR
!!!              nlat(1:3)  =0               ! for now, correct one refers gamma_summind routine.
!!!           end if
!!!           call dispersion_egr(qm_main_r%numat,xyz_disp,g_dftd3,boxsiz,nlat,qperiodic)
!!!           !!!e_dftd3 = e_dftd3 + Edis  ! energy for dispersion
!!!           CTOT = CTOT + (Edis)*TOKCAL
!!!           do i=1,qm_main_r%numat
!!!              n=qm_control_r%qminb(i)
!!!              dx(n) = dx(n) + g_dftd3(1,i)*conv_grad
!!!              dy(n) = dy(n) + g_dftd3(2,i)*conv_grad
!!!              dz(n) = dz(n) + g_dftd3(3,i)*conv_grad
!!!              g_dftd3(j,i) = zero ! for next calc.
!!!           end do
!!!        end if

        lgrad =.true.
        if (lmndod2 .or. lmndod3) then
           ! Initialize energy.
           if (lgrad) then
              ! Use variables from current routine: natom,xim,yim,zim,dx,dy,dz
              call calc_g_dftd3(xyz_disp, e_dftd3, g_dftd3, qm_main_r%numat)
           else
              call calc_e_dftd3(xyz_disp, e_dftd3, qm_main_r%numat)
           end if
           !conv_grad: H/Bohr to kcal/mol/A.  and energy: H to kcal/mol.
           do i=1,qm_main_r%numat
              n=qm_control_r%qminb(i)
              dx(n) = dx(n) + g_dftd3(1,i)*conv_grad
              dy(n) = dy(n) + g_dftd3(2,i)*conv_grad
              dz(n) = dz(n) + g_dftd3(3,i)*conv_grad
              ! debug
              !write(6,*) g_dftd3(1,i)*conv_grad,g_dftd3(2,i)*conv_grad,g_dftd3(3,i)*conv_grad
           end do
#if KEY_PARALLEL==1
     if(numnod>1) call gcomb(e_dftd3,1)
     if(MYNOD == 0) then
#endif
           CTOT = CTOT + e_dftd3*TOKCAL
           ! debug
           !write(6,'(A,F13.8)') 'D3BJ energy:',e_dftd3*TOKCAL
#if KEY_PARALLEL==1
     end if
#endif
        end if
!#if KEY_PARALLEL==1
!     end if
!#endif
  end if
  ! End of D3BJ

  ! H4 correction
  if(q_h4corr) then
     call h4_correction(EH4_corr,natom,dx,dy,dz)
#if KEY_PARALLEL==1
     if(MYNOD.EQ.0) then
#endif
        CTOT = CTOT + EH4_corr
        ! debug
        !write(6,'(A,F13.8)') 'H4 energy:',EH4_corr
#if KEY_PARALLEL==1
     end if
#endif
  end if
  !
  return
  END SUBROUTINE MNDENE_STEP4_2
!-----------------------------------------------------------------------



#else /* (mndo97)*/
  SUBROUTINE MNDENE(CTOT,X,Y,Z,DX,DY,DZ)
  use chm_kinds
      real(chm_real) CTOT,X(*),Y(*),Z(*),DX(*),DY(*),DZ(*)
      RETURN
  END SUBROUTINE MNDENE
#endif /* (mndo97)*/
