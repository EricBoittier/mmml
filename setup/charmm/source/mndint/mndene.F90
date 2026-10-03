#if KEY_MNDO97==1 /*mndo97*/
  SUBROUTINE MNDENE_prep(X,Y,Z)
  !-----------------------------------------------------------------------
  !
  ! First step: the preparation for the QM/MM energy/gradients evaluation.
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
  use qm1_info, only : qm_control_c,qm_main_c,mm_main_c,qm_gho_info_c, &
                       num_qm_system
  use qmmm_interface, only : fill_mm_coords,fill_dist_qm_mm_array,array_pointers
  use mndgho_module, only  : HBDEF
  use mndnbnd_module, only : ch2mnd

  ! Adjust nonbonded group list for IMAGES and simple PBC.
  use image
#if KEY_PBOUND==1
  use pbound, only: qBoun
#endif

  implicit none

  real(chm_real) :: X(natom),Y(natom),Z(natom)

  ! local variables
  INTEGER :: NTATOM,I,N
  LOGICAL :: QIMAGE,qfail

  ! return, if not setup qm/mm
  if(.not.qm_control_c%ifqnt) return

  ! first assign pointers for system 1 (irepl=1 or rs state)
  if(num_qm_system>1) call array_pointers(.true.,1)

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
  if(allocated(xim)) call chmdealloc('mndini.src','MNDENE_prep','xim',size(xim),crl=xim)
  if(allocated(yim)) call chmdealloc('mndini.src','MNDENE_prep','yim',size(yim),crl=yim)
  if(allocated(zim)) call chmdealloc('mndini.src','MNDENE_prep','zim',size(zim),crl=zim)
  call chmalloc('mndini.src','MNDENE_prep','xim',natom,crl=xim)
  call chmalloc('mndini.src','MNDENE_prep','yim',natom,crl=yim)
  call chmalloc('mndini.src','MNDENE_prep','zim',natom,crl=zim)

  call SwapXYZ_image(natom,X,Y,Z,xim,yim,zim,imattq)
  
  ! update QM coordinates
  do i=1,qm_main_c%numat
     n=qm_control_c%qminb(i)
     qm_main_c%qm_coord(1,i)=X(n)
     qm_main_c%qm_coord(2,i)=Y(n)
     qm_main_c%qm_coord(3,i)=Z(n)
  end do

  !! update the hybridization matrix for GHO
  !if(qm_gho_info_c%q_gho) then
  !   qfail=.true.
  !   call hbdef(x,y,z,qm_gho_info_c%BT,qm_gho_info_c%BTM,qm_gho_info_c%DBTMMM, &
  !              qm_gho_info_c%nqmlnk,qm_gho_info_c%mqm16,                      &
  !              qm_gho_info_c%IQLINK,qm_gho_info_c%JQLINK,qm_gho_info_c%KQLINK,&
  !              qfail)
  !end if

  ! non-bond list and prepare for QM/MM-interaction list
  call ch2mnd(qm_main_c%numat,igmsel,xim,yim,zim,.false.)
  !
  !! copy mm coords for qm/mm calculations
  !call fill_mm_coords(natom,xim,yim,zim,cg, &
  !                    mm_main_c%mm_coord,mm_main_c%mm_chrgs, &
  !                    mm_main_c%qm_mm_pair_list,qm_control_c%mminb1)
  !
  !! incore preparation.
  !if(qm_main_c%rij_qm_incore .or. mm_main_c%rij_mm_incore) then
  !   call fill_dist_qm_mm_array(qm_main_c%numat,mm_main_c%numatm,      &
  !                              qm_main_c%qm_coord,mm_main_c%mm_coord, &
  !                              mm_main_c%LQMEWD)
  !end if
  !

  return
  END SUBROUTINE MNDENE_prep
  !-----------------------------------------------------------------------


  SUBROUTINE MNDENE_main(CTOT,X,Y,Z,DX,DY,DZ)
  !-----------------------------------------------------------------------
  !
  ! The main routine to compute the QM/MM energy/gradients evaluation.
  !
  use chm_kinds
  use number,only: zero,one
  !use gamess_fcm,only : igmsel
#if KEY_PARALLEL==1
  use parallel,only: mynod,numnod,gcomb
#endif
  use psf,only: natom,cg
  use stream
  use prssre, only: getvol
  use image, only: xtlabc
  use ewald_1m, only: EWVIRIAL2
  use quantm, only : xim,yim,zim
  use qm1_info, only : qm_control_c,mm_main_c,qm_main_c,qm_scf_main_c,qm_gho_info_c, &
                       num_qm_system
  use qmmm_interface, only : qmmm_Ewald_setup_and_potential,fill_mm_coords,  &
                             fill_dist_qm_mm_array,scf_energy,scf_gradient,  &
                             qmmm_Ewald_gradient,array_pointers
  !use mndnbnd_module, only: ch2mnd
  use mndgho_module, only : HBDEF,DQLINK
  use leps

  ! D3BJ & H4 corrections
  use mndo97, only: dispers,q_h4corr,lmndod2,lmndod3 ! ,Edis,l_disp
  use consta, only: BOHRR,AU_TO_EV,TOKCAL
  use dftd3_mndo, only : calc_e_dftd3,calc_g_dftd3
  ! h4 correction related
  use H4_mndo, only : h4_correction
  ! mlayered qm/mm
  use mlay_mndo97, only: MNDENE_MLAYer
  use mmbonded_mod,only: q_mmbond,Energy_mmbond

  implicit none
  real(chm_real) :: CTOT,X(natom),Y(natom),Z(natom),DX(natom),DY(natom),DZ(natom)

  ! local variables
  integer :: icall
  real(chm_real):: XTLINV(6),volume
  logical       :: qcheck,OK,qfail
  logical,save  :: qfirst_call=.true.
  logical       :: q_mm_bond_c
  real(chm_real):: ECLASS,EH4_corr,E_MM_BOND_C

  ! for leps/svb correction terms
  real(chm_real) :: E_LEPS,DA_LEPS_local(3),DB_LEPS_local(3),DC_LEPS_local(3), &
                    DD_LEPS_local(3),DE_LEPS_local(3),DF_LEPS_local(3)

  ! D3BJ & H4 corrections
  ! Dispersion gradient contribution. Defined this way to
  ! fit into the wrapper for Grimmes code.
  real(chm_real)         :: e_dftd3, g_dftd3_norm
  real(chm_real),allocatable :: xyz_disp(:,:), &      ! 3,numat
                                g_dftd3(:,:)          ! 3,numat
  real(chm_real),parameter :: r_BOHRR = one/BOHRR, &  ! 1/bohrr
                              r_atoev=one/AU_TO_EV,&
                              conv_grad=TOKCAL/BOHRR
  real(chm_real):: boxsiz(3,3), e_disp_local
  integer       :: nlat(3)
  logical       :: qperiodic,lgrad
  logical, save :: qdftd3_first=.true.
  integer       :: i,j,n


  !
  ! update the hybridization matrix for GHO
  if(qm_gho_info_c%q_gho) then
     qfail=.true.
     call hbdef(x,y,z,qm_gho_info_c%BT,qm_gho_info_c%BTM,qm_gho_info_c%DBTMMM, &
                qm_gho_info_c%nqmlnk,qm_gho_info_c%mqm16,                      &
                qm_gho_info_c%IQLINK,qm_gho_info_c%JQLINK,qm_gho_info_c%KQLINK,&
                qfail)
  end if

  !! non-bond list and prepare for QM/MM-interaction list (see MNDENE_prep)
  !call ch2mnd(qm_main_c%numat,igmsel,xim,yim,zim,.false.)

  ! copy mm coords for qm/mm calculations
  call fill_mm_coords(natom,xim,yim,zim,cg, &
                      mm_main_c%mm_coord,mm_main_c%mm_chrgs, &
                      mm_main_c%qm_mm_pair_list,qm_control_c%mminb1)

  ! incore preparation.
  if(qm_main_c%rij_qm_incore .or. mm_main_c%rij_mm_incore) then
     call fill_dist_qm_mm_array(qm_main_c%numat,mm_main_c%numatm,      &
                                qm_main_c%qm_coord,mm_main_c%mm_coord, &
                                mm_main_c%LQMEWD)
  end if

  if(mm_main_c%LQMEWD) then
     ! get the volume and reciprocal space lattice vector
     call getvol(volume)
     call INVT33S(xtlinv,xtlabc,ok)

     ! setup Ktable and Kvec
     qcheck=.true.
     call qmmm_Ewald_setup_and_potential(VOLUME,XTLINV,X,Y,Z,CG,qcheck)
     !
     if(.not.qcheck) call wrndie(-5,'<MNDENE_main>','The CHARMM will stop at MNDENE/qmmm_Ewald_setup.')
  end if

  !================================================================
  ! here, now call scf energy related routines.
  icall = 0
  call scf_energy(natom,xim,yim,zim,icall,qfirst_call)
  if(icall /= -1 .and. .not. qfirst_call) qfirst_call =.false.
  !
  ! if icall returned as 0, it means scf not converged.
  if(icall == -1) call wrndie(0,'<MNDENE_main>','The QM/MM energy is not converged.')

  !
#if KEY_PARALLEL==1
  if(mynod == 0) then
#endif
     CTOT=qm_control_c%E_total
#if KEY_PARALLEL==1
  end if
#endif

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
  if(qm_gho_info_c%q_gho) then
     ECLASS = ZERO
     !
     ! treat RHF and UHF differently ... PJ 12/2002
     call dqlink(DX,DY,DZ,X,Y,Z,CG,qm_gho_info_c%FAOA,qm_gho_info_c%PHO,ECLASS, &
                 qm_gho_info_c%nqmlnk,qm_gho_info_c%norbao,                     &
                 qm_gho_info_c%lin_norbao,qm_gho_info_c%mqm16,                  &
                 qm_scf_main_c%indx,qm_gho_info_c%IQLINK,qm_gho_info_c%JQLINK,  &
                 qm_gho_info_c%BT,qm_gho_info_c%BTM,qm_gho_info_c%DBTMMM,       &
                 qm_gho_info_c%UHFGHO)
#if KEY_PARALLEL==1
     if(numnod>1) call gcomb(ECLASS,1)
     if(mynod == 0) then
#endif
        CTOT = CTOT + ECLASS
        qm_control_c%E_total=qm_control_c%E_total+ECLASS  ! also add her
#if KEY_PARALLEL==1
     end if
#endif
     !
     ! I am not sure, it will produce correct results for UHF case. Need to check
     ! if we are going to use UHF.
     if(qm_gho_info_c%uhfgho) then
        ! Include the gradient correction term for alpha and beta FOCK
        ! matrices seperately. The repulsion energy between MM atoms
        ! linked to the GHO boundary is included when the alpha
        ! correction term is included ... PJ 12/2002
        call dqlink(DX,DY,DZ,X,Y,Z,CG,qm_gho_info_c%FAOB,qm_gho_info_c%PBHO,ECLASS, &
                    qm_gho_info_c%nqmlnk,qm_gho_info_c%norbao,                      &
                    qm_gho_info_c%lin_norbao,qm_gho_info_c%mqm16,                   &
                    qm_scf_main_c%indx,qm_gho_info_c%IQLINK,qm_gho_info_c%JQLINK,   &
                    qm_gho_info_c%BT,qm_gho_info_c%BTM,qm_gho_info_c%DBTMMM,        &
                    qm_gho_info_c%UHFGHO)
     end if
  end if

  ! now, compute gradient components and copy them into the main dx/dy/dz arrays.
  call scf_gradient(natom,xim,yim,zim,dx,dy,dz)

  !
  ! compute Kspace gradient contribution, but not added here.
  ! added in subroutine KSPACE (ewalf.src)
  if(mm_main_c%LQMEWD) then
     ! do virial initialization here, (check ewaldf.src and pme.src)
     if(.not. mm_main_c%PMEwald) EWVIRIAL2(1:9)=zero

     !
     call qmmm_Ewald_gradient(X,Y,Z,dx,dy,dz,CG,EWVIRIAL2,qcheck)
  end if

  !
  ! D3BJ & H4 corrections
  if(dispers) then
     if(.not.allocated(xyz_disp))  allocate(xyz_disp(3,qm_main_c%numat))
     if(.not.allocated(g_dftd3))   allocate(g_dftd3(3,qm_main_c%numat))
     e_dftd3        = zero
     g_dftd3_norm   = zero
!#if KEY_PARALLEL==1
!     if(MYNOD.EQ.0) then
!#endif
        do i=1,qm_main_c%numat
           do j=1,3
              xyz_disp(j,i)= qm_main_c%qm_coord(j,i)*r_BOHRR
              g_dftd3(j,i) = zero
           end do
        end do
!!!        if(l_disp) then
!!!           ! for periodic boundary condition
!!!           ! in the present case, assuming that QM regions stay within the primary box.
!!!           qperiodic =.false.   ! for now turn off this.
!!!           if(mm_main_c%LQMEWD) then
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
!!!           call dispersion_egr(qm_main_c%numat,xyz_disp,g_dftd3,boxsiz,nlat,qperiodic)
!!!           !!!e_dftd3 = e_dftd3 + Edis  ! energy for dispersion
!!!           CTOT = CTOT + (Edis)*TOKCAL
!!!           do i=1,qm_main_c%numat
!!!              n=qm_control_c%qminb(i)
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
              call calc_g_dftd3(xyz_disp, e_dftd3, g_dftd3, qm_main_c%numat)
           else
              call calc_e_dftd3(xyz_disp, e_dftd3, qm_main_c%numat)
           end if
           !conv_grad: H/Bohr to kcal/mol/A.  and energy: H to kcal/mol.
           do i=1,qm_main_c%numat
              n=qm_control_c%qminb(i)
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
     if(allocated(xyz_disp))  deallocate(xyz_disp)
     if(allocated(g_dftd3))   deallocate(g_dftd3)
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

  ! for MM-bonded corrections... only to be used for mts ai-qm/mm methods
  if(q_mmbond) then
     E_MM_BOND_C = zero
     call Energy_mmbond(natom,E_MM_BOND_C,one,Xim,Yim,Zim,DX,DY,DZ)
     !
     ! copy back to the main energy array
#if KEY_PARALLEL==1
     if(MYNOD == 0) then
#endif
        CTOT = CTOT + E_MM_BOND_C
#if KEY_PARALLEL==1
     end if
#endif          
  end if

  ! nullify pointers, reassign to irepl=1, i.e., rs state
  if(num_qm_system>1) call array_pointers(.true.,1)


  ! mlayered qm/mm energies
  if(num_qm_system>1) call MNDENE_MLAYer(CTOT,xim,yim,zim,x,y,z,DX,DY,DZ)

  !
  return
  END SUBROUTINE MNDENE_main
!-----------------------------------------------------------------------

#else /* (mndo97)*/
  SUBROUTINE MNDENE(CTOT,X,Y,Z,DX,DY,DZ)
  use chm_kinds
      real(chm_real) CTOT,X(*),Y(*),Z(*),DX(*),DY(*),DZ(*)
      RETURN
  END SUBROUTINE MNDENE
#endif /* (mndo97)*/
