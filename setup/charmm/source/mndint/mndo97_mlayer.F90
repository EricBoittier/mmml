! MLAYered QM/MM method
module mlay_mndo97
  use chm_kinds
  use dimens_fcm

#if KEY_MNDO97==1 /*mndo97*/
  implicit none

  contains

  !---------------------------------------------------------------------
  subroutine mlay_memory_init(nrepl,qallocate)
     ! allocate type arrays
     use qm1_info, only: mlay_r,mlay_c
     implicit none
     integer :: nrepl
     logical :: qallocate

     ! first pointers, nullify
     if(associated(mlay_c))  nullify(mlay_c)

     ! deallocate memories
     if(allocated(mlay_r)) deallocate(mlay_r)

     ! allocate memories
     allocate(mlay_r(nrepl))

     ! pointers, as a default, num_qm_system=1
     ! later, mlay_c pointer is defined in array_pointers.
     mlay_c => mlay_r(1)
     return
  end subroutine mlay_memory_init

  !---------------------------------------------------------------------
  subroutine mlay_memory_allocate(qm_control_l,mm_main_l,mlay_l,qallocate)
     !
     ! allocate/deallocate arrays
     !
     use qm1_info, only : qm_control,mm_main,mlay_array

     implicit none
     type(mlay_array):: mlay_l
     TYPE(qm_control):: qm_control_l
     TYPE(mm_main)   :: mm_main_l
     logical         :: qallocate
     integer         :: ier=0

     ! deallocate if arrays are allocated.
     if(allocated(mlay_l%igmsel)) deallocate(mlay_l%igmsel,stat=ier)
     if(allocated(mlay_l%dx_repl)) deallocate(mlay_l%dx_repl,stat=ier)
     if(allocated(mlay_l%dy_repl)) deallocate(mlay_l%dy_repl,stat=ier)
     if(allocated(mlay_l%dz_repl)) deallocate(mlay_l%dz_repl,stat=ier)
     if(allocated(mlay_l%dx_repl_save)) deallocate(mlay_l%dx_repl_save,stat=ier)
     if(allocated(mlay_l%dy_repl_save)) deallocate(mlay_l%dy_repl_save,stat=ier)
     if(allocated(mlay_l%dz_repl_save)) deallocate(mlay_l%dz_repl_save,stat=ier)
     if(allocated(mlay_l%q_mm_flag)) deallocate(mlay_l%q_mm_flag,stat=ier)

     if(allocated(qm_control_l%igmsel))  deallocate(qm_control_l%igmsel,stat=ier)
     if(allocated(qm_control_l%AZNUC_local)) deallocate(qm_control_l%AZNUC_local,stat=ier)
     if(allocated(qm_control_l%cg_local))    deallocate(qm_control_l%cg_local,stat=ier)
     if(allocated(qm_control_l%CAATOM_local)) deallocate(qm_control_l%CAATOM_local,stat=ier)

     ! printing energy/gradients
     if(mlay_l%qmlay_print) then
        ! arrays
        if(allocated(mlay_l%dx_mpr_low))  deallocate(mlay_l%dx_mpr_low,stat=ier)
        if(allocated(mlay_l%dy_mpr_low))  deallocate(mlay_l%dy_mpr_low,stat=ier)
        if(allocated(mlay_l%dz_mpr_low))  deallocate(mlay_l%dz_mpr_low,stat=ier)
        if(allocated(mlay_l%dx_mpr_high)) deallocate(mlay_l%dx_mpr_high,stat=ier)
        if(allocated(mlay_l%dy_mpr_high)) deallocate(mlay_l%dy_mpr_high,stat=ier)
        if(allocated(mlay_l%dz_mpr_high)) deallocate(mlay_l%dz_mpr_high,stat=ier)
        if(allocated(mlay_l%mm_flag))     deallocate(mlay_l%mm_flag,stat=ier)
     end if

     ! now, allocate memory, only if qallocate==.true.
     if(qallocate) then
        allocate(mlay_l%igmsel(mm_main_l%natom),stat=ier)
        allocate(mlay_l%dx_repl(mm_main_l%natom),stat=ier)
        allocate(mlay_l%dy_repl(mm_main_l%natom),stat=ier)
        allocate(mlay_l%dz_repl(mm_main_l%natom),stat=ier)
        allocate(mlay_l%dx_repl_save(mm_main_l%natom),stat=ier)
        allocate(mlay_l%dy_repl_save(mm_main_l%natom),stat=ier)
        allocate(mlay_l%dz_repl_save(mm_main_l%natom),stat=ier)
        allocate(mlay_l%q_mm_flag(mm_main_l%natom),stat=ier)
        mlay_l%natom = mm_main_l%natom

        ! this array is defined in qm1_info and to allow access from other
        ! routines outside of mndo97
        allocate(qm_control_l%igmsel(mm_main_l%natom),stat=ier)
        allocate(qm_control_l%AZNUC_local(mm_main_l%natom),stat=ier)
        allocate(qm_control_l%cg_local(mm_main_l%natom),stat=ier)
        allocate(qm_control_l%CAATOM_local(mm_main_l%natom),stat=ier)

        ! for printing energy/gradients
        if(mlay_l%qmlay_print) then
           ! gradients
           allocate(mlay_l%dx_mpr_low(mm_main_l%natom),stat=ier)
           allocate(mlay_l%dy_mpr_low(mm_main_l%natom),stat=ier)
           allocate(mlay_l%dz_mpr_low(mm_main_l%natom),stat=ier)
           allocate(mlay_l%dx_mpr_high(mm_main_l%natom),stat=ier)
           allocate(mlay_l%dy_mpr_high(mm_main_l%natom),stat=ier)
           allocate(mlay_l%dz_mpr_high(mm_main_l%natom),stat=ier)
           allocate(mlay_l%mm_flag(mm_main_l%natom),stat=ier)
        end if
     end if
     return
  end subroutine mlay_memory_allocate

  !-----------------------------------------------------------------------
  subroutine MNDINI_MLAYer(COMLYN,COMLEN)
     !
     !     Initial setup options for Multi-layered QM/MM.
     !
     !     Kwangho Nam
     !
     use chm_kinds
     use dimens_fcm
     use number
     use memory
     use string
     use bases_fcm
     use datstr
     use code
     use coord
     use energym
     use gamess_fcm
     use inbnd
     use mndo97
     use quantm, only : natom_check,xim,yim,zim
     use nbndqm_mod
     use ewald_1m, only : lewald,kappa,erfmod
     !
     use param
     use psf
     use select
     use stream
     !
     use mndnbnd_module, only: ch2mnd_update
     use qm1_info, only      : num_qm_system,qm_main_c,qm_main_r,qm_control_c,qm_control_r, &
                               qm_scf_main_r,mm_main_r,mlay_c,mlay_r,                       &
                               dxlbomd_r,irepl_high
     use qmmm_interface, only: qmmm_init_set,qmmm_load_parameters_setup_qm_info,qmmm_Ewald_init, &
                               array_pointers,find_unique_qm
     use qmmmewald_module, only : qmmm_ewald_memory_init
     use gukini_mod
     !use leps
 
     !
#if KEY_PARALLEL==1
     use parallel  
#endif
#if KEY_FLUCQ==1
     use flucq, only : qfluc
#endif
#if KEY_REPLICA==1
     use replica_mod, only : qrep
#endif
#if KEY_BLOCK==1
     use block_fcm, only : qblock
#endif

     ! machine learning potential qm/mm
     use mndo97_mlp,only: setup_qmhub
     use qm1_info, only: qmmm_mlp_array ! for qmhub python-file i/o and/or dpmm (libtorch) interface ! arat2025sep15

     implicit none
     CHARACTER(len=*):: COMLYN
     INTEGER ::  COMLEN

     ! selection
     integer,allocatable,dimension(:) :: islct,jslct

     integer :: i,ii_option
     integer :: numat_local,natgho_local,EWMODE_local,NQMEWD_local,iopt_fdiss,imax_fdiss
     integer :: my_replica,nqmtheory,nqmcharge,nspin,iiunit,K_order,N_scf_step
     real(chm_real) :: cggho,scfconv,qmcharge
     real(chm_real) :: lambda_val,dlambda_val
     logical :: qlink_local,LQMEWD_local,NOPMEwald,QMMM_NoDiis, &
                QSRP_PhoT,QNoMemIncore,q_dxl_bomd,q_analysis,q_bond_order,q_m_charge,q_fockmd, &
                qmmbond_use,qmswtch_qmmm_local, &
                qmlay_print
     integer :: imlay_print

#if KEY_GAMESSUK==1
     integer :: init,icode,iver
     logical :: startup
#endif

     ! for MLP related
     logical :: qmhub_python,qmmm_mlp_only,qmhub_dpmm,qmhub_ml
     integer :: qmlp_mode ! arat2025sep15
     integer :: qmlp_lgpu

     !
#if KEY_BLOCK==1
     if(qblock) then
        call wrndie(-1,'<MNDINI_MLAYer>','BLOCK is not supported currenlty in MLAYer QM/MM. Ignored')
        return
     end if
#endif
#if KEY_REPLICA==1
     if(qrep) then
        call wrndie(-1,'<MNDINI_MLAYer>','MLAYer QM/MM do not work with Replicas. Ignored.')
        return
     end if
#endif

     ! current replica
     my_replica=gtrmi(COMLYN,COMLEN,'IREP',1)
     if(my_replica > num_qm_system) then
        call wrndie(-1,'<MNDINI_MLAYer>','No. of QM systems overflow. Ignored.')
        return
     else if(my_replica <= 1) then
        call wrndie(-1,'<MNDINI_MLAYer>','The current QM system must be > 1. Ignored.')
        return
     end if
     if(prnlev >= 2) then
        write(outu,19) 'MLAYer-QM/MM: multi-layered QM/MM option is activated.'
        write(outu,21) 'The QM region to be set:',my_replica
     end if
19   format(/,'MNDINI_MLAYer> ',A)
20   format(  'MNDINI_MLAYer> ',A)
21   format(  'MNDINI_MLAYer> ',A,I5)
22   format(  'MNDINI_MLAYer> ',A,I5,A)
23   format(  'MNDINI_MLAYer> ',A)
24   format(  'MNDINI_MLAYer> ',A,F12.5)
51   format(  'MNDINI_MLAYer> ',A,I1,A,I3,A)


     !=====================================================================
     ! check and allocate mlayer qm/mm related memories (mlay_array)
     ! to be done at the first time to call MNDINI_MLAYer.
     if(.not. (allocated(mlay_r))) call mlay_memory_init(num_qm_system,.true.)

     ! set the current array pointers to my_replica, including mlay_c to point mlay_r(irepl).
     call array_pointers(.true.,my_replica)

     ! local memory
     call chmalloc('mndo97_mlayer.F90','MNDINI_MLAYer','islct',natom,intg=islct)    ! for qm     atoms
     call chmalloc('mndo97_mlayer.F90','MNDINI_MLAYer','jslct',natom,intg=jslct)    ! for h-link atoms

     ! for main qm atom selection:
     call selcta(COMLYN,COMLEN,islct,x,y,z,wmain,.true.)

     ! do not support GHO and C-link atoms. for qm/mm boundary, only the H-link atoms are supported.
     ! note that LINK (qh_link) is used only for qm-mm bounds are cut, where both qm and mm atoms are
     !                          a subset of the qm region of the main/first qm region, i.e., the 
     !                          LINK sele ... end atoms are treated as mm atoms in the second qm region
     !                          (i.e., for high level qm/mm correction).
     ! thus, the LINK selected atoms are the host qm atoms for the qm-mm cut bonds, and
     !       the mm atoms are searched based on bond connectivity.
     mlay_c%num_h_link = 0
     mlay_c%lp_level   = 1   ! options for force project on the h-link atoms
     mlay_c%qh_Link    =(indxa(COMLYN,COMLEN,'LINK') > 0)
     if(mlay_c%qh_Link) then
        call selcta(COMLYN,COMLEN,jslct,x,y,z,wmain,.true.)

        ! how to hand forces on the H-link atom.
        ! read the LPLEV option to handle the force on H-link atom.
        mlay_c%lp_level = gtrmi(COMLYN,COMLEN,'LPLE',1)
        if(mlay_c%lp_level == 1) then
           if(prnlev>=2) write(outu,23) ' The force along H-QM atom will be projected out.'
        else if(mlay_c%lp_level == 2) then 
           if(prnlev>=2) write(outu,23) ' The force on H-link atom will be ignored.'
        else
           call wrndie(-1,'<MLAYer>','The default LPLEV will be used.')
           if(prnlev>=2) write(outu,23) ' The force along H-QM atom will be projected out.'
           mlay_c%lp_level   = 1
        end if
     end if

     ! multiple time step option
     ! Read the NSTEP for High level calculation during dynamics
     mlay_c%NHSTP = gtrmi(COMLYN,COMLEN,'NSTE',1)
     mlay_c%NMDSTP= 0
     if(mlay_c%NHSTP <= 0) then
        call wrndie(-1,'<MLAYer>','Specify NSTEP for High Level calculation during MD.')
        if(prnlev>=2) write(outu,21) ' Default NSTEP for High Level during MD:',mlay_c%NHSTP
     !!else if(mlay_c%NHSTP == 0) then
     !!   if(prnlev>=2) write(outu,21) ' High Level calculation will not be performed.'
     !!   mlay_c%qmlay_mts =.false.
     else if(mlay_c%NHSTP > 1) then
        if(prnlev>=2) then
           write(outu,23) ' The Multiple Time Step (MTS) option is used.'
           write(outu,22) ' High Level calculation will be updated at every ',mlay_c%NHSTP,'-th step during MD.'
        end if
        mlay_c%qmlay_mts =.true.     ! turn on the MTS option 
     else
        if(prnlev>=2) write(outu,22) ' High Level calculation will be performed at every MD step.'
     end if

     ! copy the info from the main qm region (replica 1) & set the defaults for not supported by MLAyer qm/mm method
     nqmtheory   = qm_control_r(1)%iqm_mode
     QSRP_PhoT   = qm_control_r(1)%qsrp_phot
     nqmcharge   = gtrmi(COMLYN,COMLEN,'KHAR',0)
     qmcharge    = real(nqmcharge)
     scfconv     = qm_scf_main_r(1)%SCFCRT
     nspin       = 0                              ! only singlet is allowed for now.
     QMMM_NoDiis =.not.(qm_control_r(1)%q_diis)
     QNoMemIncore= qm_main_r(1)%rij_qm_incore
     lambda_val  = GTRMF(COMLYN,COMLEN,'LAMB',one)  ! lambda for scaling mlay correction
     dlambda_val = GTRMF(COMLYN,COMLEN,'DLAM',zero) ! dlambda for changing lamdba during MD.
     if(lambda_val<one) then
        if(prnlev>=2) write(outu,24) ' High Level energy/gradients will be scaled by ',lambda_val
        mlay_c%lambda_scale = lambda_val
     else
        mlay_c%lambda_scale = one
     end if
     if(dlambda_val /= zero) then
        ! this will allow lambda changes slowly to zero during md, so we can determine
        ! work to change lambda from zero to one or one to zero.
        if(prnlev>=2) write(outu,24) ' The scaling factor (lambda) for high level energy/gradients will change by ',dlambda_val
        mlay_c%dlambda_scale = dlambda_val
     else
        mlay_c%dlambda_scale = zero
     end if

     ! printing energy/gradients (for MLP training)
     qmlay_print=(indxa(COMLYN, COMLEN, 'QPRI') > 0)
     if(qmlay_print) then
        imlay_print=gtrmi(COMLYN,COMLEN,'UPRI',6)  ! unit to print coordinates/gradients
        if(prnlev>=2) write(outu,21) ' The energy/coordiantes/gradients will be saved on unit ',imlay_print

        !
        mlay_c%qmlay_print=qmlay_print
        mlay_c%imlay_print=imlay_print
     else
        mlay_c%qmlay_print=.false.
     end if

     ! not supported by mlayered qm/mm (for the 2nd qm region)
     !dispers     =.false.
     !q_h4corr    =.false.
     !LQMEWD      =.false.
     !QLEPS       =.false.
     !qmmbond_use =.false.  ! MTS for mm-bonded/angle terms.
     !qmswtch_qmmm=.false.  ! use switching function? need to fix this later. see module nbndqm_mod

     !=====================================================================
     ! for options and others for high-level qm/mm method.
     ! i.e., do left over, but with limited options and setup for Q-Chem and others.
     !call setup_option_mlayer(COMLYN,COMLEN)
!#if KEY_QCHEM==1 || KEY_G09==1 || KEY_GAMESSUK==1 || KEY_GAMESS==1  /*gukini calls*/
!     ! mlayered qm/mm version of (shorter) gukini routine to setup
!     ! ai-qm/mm options.
!     call GUKINI_mlayer(COMLYN,COMLEN)
!#endif                                                              /*gukini calls*/

     !
     ! allocate mlay memories.
     call mlay_memory_allocate(qm_control_c,mm_main_r(1),mlay_c,.true.)

     ! 2nd qm region selection
     mlay_c%igmsel(1:mlay_c%natom) = 0  ! initialize
     call copsel_dual(numat_local,mlay_c%igmsel,islct,jslct,mlay_c%qh_Link)

     ! set qm/mm option values and allocate memory arrays for qm and mm atoms.
     natgho_local = 0
     qlink_local  =.false.  ! no gho
     LQMEWD_local =.false.  ! no qm/mm ewald
     EWMODE_local = 0
     NQMEWD_local = 0
     NOPMEwald    =.true.
  
     iiunit       = 6
     K_order      = 0
     N_scf_step   = 0
     iopt_fdiss   = 0
     imax_fdiss   = 0
     q_dxl_bomd   =.false.
     !q_fockmd     =.false.
     q_analysis   =.false.
     q_bond_order =.false.
     q_m_charge   =.false.
     qmswtch_qmmm_local=.false.

     ! should allow fockmd to be used for low-level 2nd qm region calculation?
     ! Fock matrix dynamics (ref: ).
     q_fockmd=(indxa(COMLYN,COMLEN,'FOCK') > 0)
     if(QMMM_NoDiis) q_fockmd=.false.   ! turn-off if DIIS is off.
     if(q_fockmd) then
        imax_fdiss = GTRMI(COMLYN,COMLEN,'MXDI',5)
        if(imax_fdiss <= 1) imax_fdiss = 5
        iopt_fdiss = GTRMI(COMLYN,COMLEN,'FOPT',2)
        if(.not.(iopt_fdiss==1 .or. iopt_fdiss==2)) iopt_fdiss=2
        if(prnlev >= 2) &
           write(outu,51) 'Fock matrix dynamics (option: ',iopt_fdiss,') is used with ',imax_fdiss,' max iterations.'
     end if

     call qmmm_init_set(nqmtheory,nqmcharge,nspin,numat_local,natgho_local,  &
                        natom,EWMODE_local,NQMEWD_local,                     &
                        qmcharge,scfconv,                                    &
                        qlink_local,QMMM_NoDiis,LQMEWD_local,NOPMEwald,      &
                        q_bond_order,q_m_charge,iiunit,                      &
                        q_dxl_bomd,K_order,N_scf_step,                       &
                        q_fockmd,iopt_fdiss,imax_fdiss,                      &
                        qmswtch_qmmm_local,QNoMemIncore)

     ! Get and print QM region info
     call Get_dual_QM_from_CHM(mlay_c%igmsel)

     ! initialize parameters, setup qm info, and allocate memories.
     call qmmm_load_parameters_setup_qm_info(QSRP_PhoT)

     !=============================================================================!
     ! new qm region definition....
     ! 
     ! perhaps, we user defined "RCUT" value?
     !
     ! get mm atoms ready for the qm/mm calcualtions
     ! 1) non-bonded list.
     if (useddt_nbond(bnbnd).and.numat_local > 0) call nbndqm(x,y,z)

     ! mm coordinates copied to xim,yim,zim arrays
     natom_check= natom      ! for checking purpose
     if(allocated(XIM)) call chmdealloc('mndo97_mlayer.F90','MNDINI_MLAYer','XIM',size(XIM),crl=XIM)
     if(allocated(YIM)) call chmdealloc('mndo97_mlayer.F90','MNDINI_MLAYer','YIM',size(YIM),crl=YIM)
     if(allocated(ZIM)) call chmdealloc('mndo97_mlayer.F90','MNDINI_MLAYer','ZIM',size(ZIM),crl=ZIM)
     !
     call chmalloc('mndo97_mlayer.F90','MNDINI_MLAYer','XIM',natom,crl=XIM)
     call chmalloc('mndo97_mlayer.F90','MNDINI_MLAYer','YIM',natom,crl=YIM)
     call chmalloc('mndo97_mlayer.F90','MNDINI_MLAYer','ZIM',natom,crl=ZIM)

     call SwapXYZ_image(natom,x,y,z,xim,yim,zim,imattq)
      
     ! ready the coordinates for qm/mm interface.
     ! this needs to be worked out....  (check for now 2023-0110)
     mlay_c%q_mm_flag(1:natom) =.false.
     call ch2mnd_update(qm_main_c%numat,mlay_c%igmsel,mlay_c%q_mm_flag,.true.)
     !=============================================================================!


     ! turn on the mlyered qm/mm method.
     qmlay_main           =.true.  ! main flag to turn on mlayered qm/mm (mndo97_ltm file)
     qmlay_mts            = mlay_c%qmlay_mts  ! MTS option is on.
     irepl_high           = my_replica        ! replica index for high-level qm/mm correction
     if(qmlay_mts) then
        nmlay_nmts        = mlay_c%NHSTP      ! md step freq. to update mts ai-qm/mm correction.
        nmlay_mdstp       = 0 ! mlay_c%NMDSTP ! md step counter.. (initialize here)
        qmlay_do_energy   =.true.             ! so, this will be used to control
     else
        nmlay_nmts        = 1
        nmlay_mdstp       = 0
     end if
     mlay_c%qmlay_high    =.true.  ! my_replica    is a high-level region.
                                   ! my_replica==1 is a low-level  region.
     mlay_c%qmlay_main    =.true.  ! turn on the main flag, meaning using mlayered qm/mm method.
     mlay_r(1)%qmlay_main =.true.  !
     mlay_r(1)%qmlay_high =.false. ! meaning low-level region.

     !! debug
     !if(prnlev>=2) then
     !   write(6,*) 'qmlay_main:',qmlay_main
     !   write(6,*) 'qmlay_mts :',qmlay_mts,mlay_c%qmlay_mts
     !   write(6,*) 'nmlay_nmts:',nmlay_nmts,mlay_c%NHSTP
     !   write(6,*) 'nmlay_mdstp:',nmlay_mdstp,mlay_c%NMDSTP
     !   write(6,*) 'irepl_high :',irepl_high,my_replica
     !   write(6,*) 'qmlay_do_energy:',qmlay_do_energy
     !   write(6,*) 'mlay_c%qmlay_high:',mlay_c%qmlay_high
     !   write(6,*) 'mlay_c%qmlay_main:',mlay_c%qmlay_main
     !   write(6,*) 'mlay_r(1)%qmlay_main:',mlay_r(1)%qmlay_main
     !   write(6,*) 'mlay_r(1)%qmlay_high:',mlay_r(1)%qmlay_high
     !end if

     ! for the primay qm/mm region.. reset several options not supported.
     !if(qm_control_r(1)%q_dxl_bomd) then
     !   if(prnlev>=2) write(outu,23) ' DXL-BOMD option is not supported. Ignored.'
     !   qm_control_r(1)%q_dxl_bomd =.false.
     !   dxlbomd_r(1)%Kth_sum_order =0
     !   if(allocated(dxlbomd_r(1)%cextr)) deallocate(dxlbomd_r(1)%cextr)
     !end if
     !if(qm_control_r(1)%q_fockmd) then
     !   if(prnlev>=2) write(outu,23) ' Fock-MD option is not supported. Ignored.'
     !   qm_control_r(1)%q_fockmd        =.false.
     !   qm_control_r(1)%q_do_fockmd_scf =.false.
     !end if

     ! deallocate local arrays
     call chmdealloc('mndo97_mlayer.F90','MNDINI_MLAYer','islct',natom,intg=islct)
     call chmdealloc('mndo97_mlayer.F90','MNDINI_MLAYer','jslct',natom,intg=jslct)

     ! for handling of h-link coords here.
     if(mlay_c%qh_link) call get_hlink_coords(xim,yim,zim,.true.,.true.)

     !======================= interface to gukini routines ========================!
#if KEY_QCHEM==1 || KEY_G09==1 || KEY_GAMESSUK==1 || KEY_GAMESS==1  /*gukini calls*/
     ! mlayered qm/mm version of (shorter) gukini routine to setup
     ! ai-qm/mm options.
     call GUKINI_mlayer(COMLYN,COMLEN)

     !!! need to work on this...
     !!call COPSEL(ISLCT,QQINP)

     ! This initialize gamess data 
     QINIGM=.TRUE.

     ! for high level qm methods.
     QMLAY_high =.true.     ! this is from gamess_fcm

     ! copy igmsel & cg array to be accessible from gukini related files.
     ! note: mm charges copied in Get_dual_QM_from_CHM for high-level qm/mm.
     do i=1,natom
        qm_control_c%igmsel(i) = mlay_c%igmsel(i)
        igmsel(i)              = mlay_c%igmsel(i)
        cg(i)                  = qm_control_c%cg_local(i)
     end do

     ! Modify QChem input and output filenames for replica/path and neb
#if KEY_QCHEM==1 || KEY_G09==1 /*qcini*/
     ! Get the info from the Q-chem input file
     if(qmused_qchem .or. qmused_g09) then
        !call QCHEMINI(NGAMES,mlay_c%igmsel)
        qmused_qchem =.true.
        NGAMES       = qm_main_c%numat  ! total no. of qm atoms for the high-level region.
     end if
#endif /* (qcini)*/
#if KEY_QTURBO==1 /*qtini*/
     !CALL QTURBOINI
     NGAMES = qm_main_c%numat  ! total no. of qm atoms for the high-level region.
#endif /* (qtini)*/

#if KEY_GAMESS==1
     if(qmused_gamess) then
        call CH2GMS(.TRUE.)
        call GAMESS
     end if
#endif

#if KEY_GAMESSUK==1 /*guk*/
     init         = 1
     startup      =.true.
     iver         = 5
     LQMEWD_local =.false.
     call GAMESS(init,icode,startup,LQMEWD_local,iver)  ! turn off LQMEWD
     if(iver /= -5) then
        if(prnlev>=2) write(6,*) iver
        call wrndie(-5,'<GAMESS-UK>','Code Version Mismatch')
     end if
#endif /* (guk)*/

     ! Report QM/MM repulsion energy, also when no derivatives involved
#if KEY_GAMESS==1
     if((prnlev>=2).and.qmused_gamess) call CGREPE(NATOM)
#endif

#else                                                               /*gukini calls*/
     call wrndie(-5,'<MNDINI_MLAYer>','Ab initio QM/MM code not compiled.')
#endif                                                              /*gukini calls*/
     !======================= End interface to gukini routines ====================!

     !======================= interface for MLP QM/MM routines ====================!
     ! machine learning potentials (QMHub Python File I/O and/or DPMM (LibTorch)) QM/MM
     ! PYTHon|DPMM MLPMode [int]                   ! between MLP or. delta-MLP
     !               PGPUid [int]                    ! Set CUDA_VISIBLE_DEVICES GPU id: 0 ~ ; -1 not use gpu
     qmhub_python   =(INDXA(COMLYN,COMLEN,'PYTH') > 0) ! use qmhub Python File I/O interface
     qmhub_dpmm     =(INDXA(COMLYN,COMLEN,'DPMM') > 0) ! or  DPMM (LibTorch) interface
     qmmm_mlp_only  =(INDXA(COMLYN,COMLEN,'ONLY') > 0) ! flag for only using MLP not QMLAY_high...
     qmhub_ml       = qmhub_python .or. qmhub_dpmm
     !qmlp_mode   =-1                                ! no MLP/delta-MLP is used
     !                                               ! 0: MLP; 1: delta-MLP
     qmlp_mode=gtrmi(COMLYN,COMLEN,'MLPM',-1)
     qmlp_lgpu=gtrmi(COMLYN,COMLEN,'PGPU',-1)
     if(qmhub_ml) then
        if(prnlev>=2) then
           if(qmhub_python) then
              write(outu,20) 'QMHub Python File I/O Interfacer Selected: Setup qmhub.ini control file in current directory'
           else if(qmhub_dpmm) then
              write(outu,20) 'QMHub DPMM (LibTorch) Interfacer Selected: Setup qmhub.ini control file in current directory'
           end if
           if(qmmm_mlp_only) write(outu,20) 'Only use MLP/delta-MLP.'
        end if
        if(qmlp_mode<0 .or. qmlp_mode>1) then
           if(prnlev>=2) write(outu,20) 'wrong MLP mode between MLP (0) and delta-MLP(1)'
           call wrndie(0,'<MNDINI_MLAYer>','Use default MLP mode (delta-MLP).')
           qmlp_mode = 1
        end if
        if(.not. (qmhub_dpmm .or. qmhub_python)) qmlp_lgpu = -1
        if(qmhub_dpmm .and. qmlp_lgpu >= 0) then
#if KEY_PARALLEL==1
           if(mynod==0) write(outu,21) 'QMHub DPMM (LibTorch) Interfacer use GPU for Model interence: GPU id:',qmlp_lgpu
           if(mynod/=0) qmlp_lgpu = -1
#else
           if(prnlev>=2) write(outu,21) 'QMHub DPMM (LibTorch) Interfacer use GPU for Model interence: GPU id:',qmlp_lgpu
#endif
        end if

        ! do setup
        call setup_qmhub(qmhub_ml,qmmm_mlp_only,qmhub_python,qmhub_dpmm,qmlp_mode,qmlp_lgpu,my_replica)
     end if
     !======================= End interface for MLP QM/MM routines ================!

     ! Turn on Logical flag that says that the PSF has been modified.
     ! (Refer code.f90)
     MUSTUP=.TRUE.

     ! reset these variables to avoid direct energy call to them.
     !if(qmused_qchem)    qmused_qchem=.false.
     !if(qmused_gamess)   qmused_gamess=.false.
     !if(qmused_g09)      qmused_g09=.false.
     !if(qmused_turbo)    qmused_turbo=.false.
     !if(qmused_nwchem)   qmused_nwchem=.false.
     !if(qmused_gamessuk) qmused_gamessuk=.false.

     ! restore original charges and igmsel values of the primary system
     ! here, cgqmmm contains a copy of the primary qm system.
     do i=1,natom
        igmsel(i) = qm_control_r(1)%igmsel(i)
        cg(i)     = qm_control_c%cgqmmm(i)
     end do

     ! for handling of h-link coords here.
     if(mlay_c%qh_link) call get_hlink_coords(xim,yim,zim,.false.,.false.)

     ! temporary memories =============================
     call chmdealloc('mndo97_mlayer.F90','MNDINI_MLAYer','XIM',size(XIM),crl=XIM)
     call chmdealloc('mndo97_mlayer.F90','MNDINI_MLAYer','YIM',size(YIM),crl=YIM)
     call chmdealloc('mndo97_mlayer.F90','MNDINI_MLAYer','ZIM',size(ZIM),crl=ZIM)
     !=================================================

     ! return & set pointer array back to replica==1
     call array_pointers(.true.,1)

     !
     comlen = 0
  
     return
  end subroutine MNDINI_MLAYer


  !-----------------------------------------------------------------------
  subroutine MNDENE_MLAYer(CTOT,X,Y,Z,x_ll,y_ll,z_ll,DX,DY,DZ)
     !
     ! Compute energy and gradients
     !
     use chm_kinds
     use number,only: zero,one,minone
     use gamess_fcm
#if KEY_PARALLEL==1
     use parallel,only: mynod,numnod
#endif
     use psf,only: natom,cg
     use stream
     use qm1_info, only : qm_control_c,qm_control_r,mm_main_c,qm_main_c,qm_scf_main_c, &
                          num_qm_system,mlay_c,mlay_r,irepl_high, &
                          qmmm_mlp
     use qmmm_interface, only : fill_mm_coords,fill_dist_qm_mm_array,scf_energy,scf_gradient,  &
                                array_pointers
     use mndnbnd_module, only : ch2mnd_update
     use mndo97,only : qmlay_energy_updated,qmlay_do_energy,qmlay_main,qmlay_mts, &
                       nmlay_mdstp,nmlay_nmts,qm_md_master
     use mmbonded_mod,only: q_mmbond,Energy_mmbond

     ! machine learning potential qm/mm
     use mndo97_mlp,only: qmhub_energy

     implicit none
     real(chm_real) :: CTOT,X(natom),Y(natom),Z(natom),DX(natom),DY(natom),DZ(natom), &
                       x_ll(natom),y_ll(natom),z_ll(natom)

     ! local variables
     integer :: i,j,n,icall,jj
     logical :: qcheck,OK,qfail
     logical,save:: qfirst_call=.true., q_do_seqmmm=.true.
     real(chm_real) :: E_MM_BOND_C,E_MLP,E_sqm

     !!!integer :: irepl_high  ! defined in qm1_info

     ! mlp energy...
     E_sqm = zero
     E_MLP = zero
     if(qmmm_mlp%qmmm_mlp) then
        ! for MLP case:  we will still call qm/mm-pme for the long-range contribution
        !                and substract qm/mm energy within the cutoff, plus MLP energy call for
        !                MLP contribution within the cutoff.
        ! E_qm/mm-total = E_qm/mm-pme + (E_MLP (cutoff) - E_qm/mm (cutoff))
        ! 
        !                so, first call the qm/mm-pme energy, followed by MLP & qm/mm cutoff energies
        !
        ! for delta-MLP case:
        !               first, call qm/mm-pme for the entire system's energy, plus
        !               MLP energy call for delta-MLP contribution.
        ! E_qm/mm-total = E_qm/mm-pme + E_delta-MLP (cutoff)
        !
        !                so, first call the qm/mm-pme energy, followed by delta-MLP cutoff energy
        !
        ! in the end,
        ! E_MLP & dx_mlp/dy_mlp/dz_mlp are MLP/delta-MLP only energy and gradients... so later,
        ! they can be used without recalculating them when using QMLAY_high

        ! check
        if(irepl_high <= 0) then
           if(prnlev >= 2) write(outu,20) 'No mlp/delta-mlp qm/mm replica is defined. Return.'
           return
        end if

        !
        ! nee to turn to irepl_high replica
        ! assign irepl arrays
        call array_pointers(.true.,irepl_high)

        !
        if(qmmm_mlp%qmlp_mode == 0) then
           ! for MLP: E_qm/mm-total = E_qm/mm-pme + (E_MLP (cutoff) - E_qm/mm (cutoff))
           do i=1,natom
              mlay_c%dx_repl(i) = zero
              mlay_c%dy_repl(i) = zero
              mlay_c%dz_repl(i) = zero

              !
              qmmm_mlp%dx_mlp(i)= zero
              qmmm_mlp%dy_mlp(i)= zero
              qmmm_mlp%dz_mlp(i)= zero

              !
              igmsel(i)         = mlay_c%igmsel(i)
              cg(i)             = qm_control_c%cg_local(i)
           end do
        else
           ! for delta-MLP: E_qm/mm-total = E_qm/mm-pme + E_delta-MLP (cutoff)
           do i=1,natom
              qmmm_mlp%dx_mlp(i)= zero
              qmmm_mlp%dy_mlp(i)= zero
              qmmm_mlp%dz_mlp(i)= zero

              !
              igmsel(i)         = mlay_c%igmsel(i)
              cg(i)             = qm_control_c%cg_local(i)
           end do
        end if

        ! h-link atom coordinates..
        if(mlay_c%qh_link) call get_hlink_coords(x,y,z,.true.,.false.)

        ! for se-qm/mm region ==========================================
        ! 1. update QM coordinates
        do i=1,qm_main_c%numat
           n=qm_control_c%qminb(i)
           qm_main_c%qm_coord(1,i)=X(n)
           qm_main_c%qm_coord(2,i)=Y(n)
           qm_main_c%qm_coord(3,i)=Z(n)
        end do

        ! 2. non-bond list and prepare for QM/MM-interaction list
        mlay_c%q_mm_flag(1:natom) =.false.
        call ch2mnd_update(qm_main_c%numat,mlay_c%igmsel,mlay_c%q_mm_flag,.false.)

        ! now, compute MLP or delta-MLP energy and gradients
        ! where MLP and delta-MLP values are "+" contribution to E_se-qm/mm-pme...
        ! Unit of the energy is converted in qmhub_energy.
#if KEY_PARALLEL==1
        if(mynod==0) then
#endif
        call qmhub_energy(E_MLP,x,y,z,qmmm_mlp%dx_mlp,qmmm_mlp%dy_mlp,qmmm_mlp%dz_mlp, &
                          qm_control_c%cg_local,NATOM)
#if KEY_PARALLEL==1
        end if
#endif


        ! for MLP case, we need se-qm/mm (cutoff) energy and gradients
        if(qmmm_mlp%qmlp_mode == 0) then
           ! 3. copy mm coords for qm/mm calculations
           call fill_mm_coords(natom,x,y,z,qm_control_c%cg_local, &
                               mm_main_c%mm_coord,mm_main_c%mm_chrgs, &
                               mm_main_c%qm_mm_pair_list,qm_control_c%mminb1)

           ! 4. incore preparation.
           if(qm_main_c%rij_qm_incore .or. mm_main_c%rij_mm_incore) then
              call fill_dist_qm_mm_array(qm_main_c%numat,mm_main_c%numatm,      &
                                         qm_main_c%qm_coord,mm_main_c%mm_coord, &
                                         mm_main_c%LQMEWD)
           end if

           ! here, now call scf energy related routines.
           icall = 0
           call scf_energy(natom,x,y,z,icall,qfirst_call)
           if(icall /= -1 .and. .not. qfirst_call) qfirst_call =.false.
           !
           ! if icall returned as 0, it means scf not converged.
           if(icall == -1) call wrndie(0,'<MNDENE_MLAYer>','The QM/MM energy is not converged.')

#if KEY_PARALLEL==1
           if(mynod == 0) then
#endif
              E_sqm =-qm_control_c%E_total  ! negative sign here.
#if KEY_PARALLEL==1
           end if
#endif
           ! compute gradient components and copy them into the main dx/dy/dz arrays,
           ! and inverse sign as it is for low-level qm/mm, so that
           ! E_corr = E_MLP (cutoff) - E_se-qm/mm (cutoff)
           ! dx_corr= dx_mlp - dx_sqm; dy_corr= dy_mlp - dy_sqm; dy_corr= dy_mlp - dy_sqm
           call scf_gradient(natom,x,y,z,mlay_c%dx_repl,mlay_c%dy_repl,mlay_c%dz_repl)
           do i=1,natom
              if(mlay_c%q_mm_flag(i)) then
                 mlay_c%dx_repl(i) = qmmm_mlp%dx_mlp(i) - mlay_c%dx_repl(i)
                 mlay_c%dy_repl(i) = qmmm_mlp%dy_mlp(i) - mlay_c%dy_repl(i)
                 mlay_c%dz_repl(i) = qmmm_mlp%dz_mlp(i) - mlay_c%dz_repl(i)
              end if
           end do

           ! for MM-bonded corrections... only to be used for mts ai-qm/mm methods
           E_MM_BOND_C = zero
           if(q_mmbond) then
              call Energy_mmbond(natom,E_MM_BOND_C,minone,X_ll,Y_ll,Z_ll, &
                                 mlay_c%dx_repl,mlay_c%dy_repl,mlay_c%dz_repl)
#if KEY_PARALLEL==1
              if(mynod == 0) then
#endif
              E_sqm = E_sqm + E_MM_BOND_C  ! note that the sign is aready taken care in
                                           ! Energy_mmbond
#if KEY_PARALLEL==1
              end if
#endif
           end if

           ! printing related (only done with MTS & MLP; delta-MLP case, it is done below).
           ! fill in mm and qm arrays, including energy and gradients for low-level
           if(mlay_c%qmlay_print .and. .not. qmmm_mlp%qmmm_mlp_only) then
              ! copy low-level energy (note the sign..
#if KEY_PARALLEL==1
              if(mynod == 0) then
#endif 
                 mlay_c%e_pr_low = - E_sqm
#if KEY_PARALLEL==1
              else
                 mlay_c%e_pr_low = zero
              end if
#endif
              ! first qm region
              mlay_c%mm_flag(1:natom)       = 0   ! 0: mm atoms outside cutoff
                                                  ! 1: mm atoms within  cutoff
                                                  ! 5: qm atoms
              do i=1,qm_main_c%numat
                 mlay_c%mm_flag(qm_control_c%qminb(i))   = 5     ! qm region
              end do

              ! second mm region
              do i=1,natom
                 mlay_c%dx_mpr_low(i)  = zero  ! low-level gradients
                 mlay_c%dy_mpr_low(i)  = zero
                 mlay_c%dz_mpr_low(i)  = zero
                 mlay_c%dx_mpr_high(i) = zero  ! high-level gradients
                 mlay_c%dy_mpr_high(i) = zero
                 mlay_c%dz_mpr_high(i) = zero
                 if(mlay_c%mm_flag(i) == 0 .and. mlay_c%q_mm_flag(i)) mlay_c%mm_flag(i) = 1 ! mm atom within the cutoff

                 ! copy low-level energy/gradients for qm/mm atoms 
                 ! note: 1) signs are changed as it was multiplied by "-" above.
                 !       2) atoms with mlay_c%mm_flag(i) == 5, QM atoms
                 !                                          1, MM atoms within the cutoff
                 if(mlay_c%q_mm_flag(i)) then
                    mlay_c%dx_mpr_low(i) = qmmm_mlp%dx_mlp(i) - mlay_c%dx_repl(i)
                    mlay_c%dy_mpr_low(i) = qmmm_mlp%dy_mlp(i) - mlay_c%dy_repl(i)
                    mlay_c%dz_mpr_low(i) = qmmm_mlp%dz_mlp(i) - mlay_c%dz_repl(i)
                 end if
              end do
           end if
        else
           ! copy delta-MLP gradients to dx_repl/dy_repl/dz_repl arrays
           do i=1,natom
              if(mlay_c%q_mm_flag(i)) then
                 mlay_c%dx_repl(i) = qmmm_mlp%dx_mlp(i)
                 mlay_c%dy_repl(i) = qmmm_mlp%dy_mlp(i)
                 mlay_c%dz_repl(i) = qmmm_mlp%dz_mlp(i)
              end if
           end do
        end if

        ! h-link atom coordinates & gradients.
        if(mlay_c%qh_link) then
           ! for coordinates
           call get_hlink_coords(x,y,z,.false.,.false.)

           ! for gradients, i.e., projection for h-link atoms..
           call put_hlink_grads(mlay_c%dx_repl,mlay_c%dy_repl,mlay_c%dz_repl,.true.)
        end if

        ! copy energy and gradients
#if KEY_PARALLEL==1
        if(mynod == 0) then
#endif
           CTOT = CTOT + (E_MLP + E_sqm) ! E_sqm sign done above
#if KEY_PARALLEL==1
        end if
#endif
        ! gradients
        ! MLP      : E_qm/mm-total = E_qm/mm-pme + (E_MLP (cutoff) - E_qm/mm (cutoff))
        ! delta-MLP: E_qm/mm-total = E_qm/mm-pme + E_delta-MLP (cutoff)
        do i=1,natom
           if(mlay_c%q_mm_flag(i)) then
              dx(i) = dx(i) + mlay_c%dx_repl(i)
              dy(i) = dy(i) + mlay_c%dy_repl(i)
              dz(i) = dz(i) + mlay_c%dz_repl(i)
           end if

           ! restore original charges and igmsel values of the primary system
           ! here, cgqmmm is a copy of the primary qm system.
           igmsel(i) = qm_control_r(1)%igmsel(i)
           cg(i)     = qm_control_c%cgqmmm(i)
        end do

        ! return with original pointers for all arrays
        call array_pointers(.true.,1)

        ! return not performing high-level QMLAY_high calc.
        if(qmmm_mlp%qmmm_mlp_only) return  ! as only use MLP not QMLAY_high.
     end if

     ! find high-level qm/mm replica
     if(QMLAY_high) then
        !!!irepl_high=0
        !!!if(allocated(mlay_r) .and. size(mlay_r) >=num_qm_system) then
        !!!   do i=2,num_qm_system
        !!!      if(mlay_r(i)%qmlay_high) irepl_high=i 
        !!!   end do
        !!!end if

        if(irepl_high <= 0) then
           if(prnlev >= 2) write(outu,20) 'No high-level qm/mm replica is defined. Return.'
           return
        end if
20      format('MNDENE_MLAYer> ',A)

        !===================================================================
        ! First, determine if MTS is used and update is needed at this energy call.
        if(qm_md_master .and. mlay_r(irepl_high)%qmlay_mts .and. qm_control_r(1)%md_run) then
           if((mod(nmlay_mdstp,nmlay_nmts) /= 0) .and. .not. qmlay_do_energy) then
              ! mts-ai-qm/mm is used & md simulation
              ! check whether this energy call is to update energy/gradients
#if KEY_PARALLEL==1
              if(mynod == 0) then
#endif
                 ! only update energy...
                 CTOT = CTOT + mlay_r(irepl_high)%E_value_save

                 ! not the gradient... here..
                 !do i=1,natom
                 !   dx(i) = dx(i) + mlay_r(irepl_high)%dx_repl_save(i)
                 !   dy(i) = dy(i) + mlay_r(irepl_high)%dy_repl_save(i)
                 !   dz(i) = dz(i) + mlay_r(irepl_high)%dz_repl_save(i)
                 !end do
#if KEY_PARALLEL==1
              end if
#endif
              ! return...
              qmlay_energy_updated =.false.  ! & skip the high-level qm/mm energy call.
                                          ! meaning gradients are not added to the main gradient arrays
              return
           end if
        end if

        !===================================================================
        ! Second, perform ai-qm/mm correction term calculation...

        ! setup first...
        ! assign irepl arrays
        call array_pointers(.true.,irepl_high)

        ! initialize energy and gradients for replicas & prepare charnges and others
        mlay_c%E_value = zero
        do i=1,natom
           mlay_c%dx_repl(i) = zero
           mlay_c%dy_repl(i) = zero
           mlay_c%dz_repl(i) = zero

           !
           igmsel(i)         = mlay_c%igmsel(i)
           cg(i)             = qm_control_c%cg_local(i)
        end do

        ! h-link atom coordinates..
        if(mlay_c%qh_link) call get_hlink_coords(x,y,z,.true.,.false.)

        !==========(1) Low level qm/mm calculation =========================
        ! (1) first do semi-qm/mm calculations.
        ! 1. update QM coordinates
        do i=1,qm_main_c%numat
           n=qm_control_c%qminb(i)
           qm_main_c%qm_coord(1,i)=X(n)
           qm_main_c%qm_coord(2,i)=Y(n)
           qm_main_c%qm_coord(3,i)=Z(n)
        end do

        ! 2. non-bond list and prepare for QM/MM-interaction list
        mlay_c%q_mm_flag(1:natom) =.false. 
        call ch2mnd_update(qm_main_c%numat,mlay_c%igmsel,mlay_c%q_mm_flag,.false.)

        ! check whether we need se-qm/mm calc.
        q_do_seqmmm =.true.
        ! MLP case, we can skip se-qm/mm (cutoff) energy and gradients;
        ! delta-MLP case, we need se-qm/mm (cutoff) energy and gradients
        if(qmmm_mlp%qmmm_mlp .and. qmmm_mlp%qmlp_mode == 0) q_do_seqmmm =.false.

        if(q_do_seqmmm) then
           ! 3. copy mm coords for qm/mm calculations
           call fill_mm_coords(natom,x,y,z,qm_control_c%cg_local, &
                               mm_main_c%mm_coord,mm_main_c%mm_chrgs, &
                               mm_main_c%qm_mm_pair_list,qm_control_c%mminb1)

           ! 4. incore preparation.
           if(qm_main_c%rij_qm_incore .or. mm_main_c%rij_mm_incore) then
              call fill_dist_qm_mm_array(qm_main_c%numat,mm_main_c%numatm,      &
                                         qm_main_c%qm_coord,mm_main_c%mm_coord, &
                                         mm_main_c%LQMEWD)
           end if

           ! here, now call scf energy related routines.
           icall = 0
           call scf_energy(natom,x,y,z,icall,qfirst_call)
           if(icall /= -1 .and. .not. qfirst_call) qfirst_call =.false.
           !
           ! if icall returned as 0, it means scf not converged.
           if(icall == -1) call wrndie(0,'<MNDENE_MLAYer>','The QM/MM energy is not converged.') 

#if KEY_PARALLEL==1
           if(mynod == 0) then
#endif
              mlay_c%E_value=-qm_control_c%E_total
#if KEY_PARALLEL==1
           end if
#endif
           ! compute gradient components and copy them into the main dx/dy/dz arrays,
           ! and inverse sign as it is for low-level qm/mm, so that
           ! E_corr = E_high  - E_low
           ! dx_corr= dx_high - dx_low; dy_corr= dy_high - dy_low; dy_corr= dy_high - dy_low
           call scf_gradient(natom,x,y,z,mlay_c%dx_repl,mlay_c%dy_repl,mlay_c%dz_repl)
           do i=1,natom
              if(mlay_c%q_mm_flag(i)) then
                 mlay_c%dx_repl(i) = -mlay_c%dx_repl(i)
                 mlay_c%dy_repl(i) = -mlay_c%dy_repl(i)
                 mlay_c%dz_repl(i) = -mlay_c%dz_repl(i)
              end if
           end do

           ! for MM-bonded corrections... only to be used for mts ai-qm/mm methods
           E_MM_BOND_C = zero
           if(q_mmbond) then
              call Energy_mmbond(natom,E_MM_BOND_C,minone,X_ll,Y_ll,Z_ll, &
                                 mlay_c%dx_repl,mlay_c%dy_repl,mlay_c%dz_repl)
#if KEY_PARALLEL==1
              if(mynod == 0) then
#endif
              mlay_c%E_value = mlay_c%E_value + E_MM_BOND_C  ! note that the sign is aready taken care in
                                                             ! Energy_mmbond

              !!! just for updating low-level energy term
              !!qm_control_c%E_total = qm_control_c%E_total - E_MM_BOND_C ! low-level energy
#if KEY_PARALLEL==1
              end if
#endif
           end if

           ! printing related (only done with MTS & delta-MLP case; MLP case, it is done above).
           ! fill in mm and qm arrays, including energy and gradients for low-level
           if(mlay_c%qmlay_print) then
              ! copy low-level energy (note the sign..
#if KEY_PARALLEL==1
              if(mynod == 0) then
#endif 
                 mlay_c%e_pr_low = - mlay_c%E_value
#if KEY_PARALLEL==1
              else
                 mlay_c%e_pr_low = zero
              end if
#endif
              ! first qm region
              mlay_c%mm_flag(1:natom)       = 0   ! 0: mm atoms outside cutoff
                                                  ! 1: mm atoms within  cutoff
                                                  ! 5: qm atoms
              do i=1,qm_main_c%numat
                 mlay_c%mm_flag(qm_control_c%qminb(i))   = 5     ! qm region
              end do

              ! second mm region
              do i=1,natom
                 mlay_c%dx_mpr_low(i)  = zero  ! low-level gradients
                 mlay_c%dy_mpr_low(i)  = zero
                 mlay_c%dz_mpr_low(i)  = zero
                 mlay_c%dx_mpr_high(i) = zero  ! high-level gradients
                 mlay_c%dy_mpr_high(i) = zero
                 mlay_c%dz_mpr_high(i) = zero
                 if(mlay_c%mm_flag(i) == 0 .and. mlay_c%q_mm_flag(i)) mlay_c%mm_flag(i) = 1 ! mm atom within the cutoff

                 ! copy low-level energy/gradients for qm/mm atoms 
                 ! note: 1) signs are changed as it was multiplied by "-" above.
                 !       2) atoms with mlay_c%mm_flag(i) == 5, QM atoms
                 !                                          1, MM atoms within the cutoff
                 if(mlay_c%q_mm_flag(i)) then
                    mlay_c%dx_mpr_low(i) = -mlay_c%dx_repl(i)
                    mlay_c%dy_mpr_low(i) = -mlay_c%dy_repl(i)
                    mlay_c%dz_mpr_low(i) = -mlay_c%dz_repl(i)
                 end if
              end do
           end if
        end if  ! q_do_seqmmm

        ! for MLP/delta-MLP qm/mm handling.. (MLP/delta-MLP energy and gradients are computed above.)
        if(qmmm_mlp%qmmm_mlp) then
           ! delta-MLP: correction: E_se-qm/mm (cutoff) + E_delta-MLP (cutoff)
           ! note sign on mlay_c%E_value & dx_repl/dy_repl/dz_repl
           ! while MLP: correction: E_MLP only
#if KEY_PARALLEL==1
           if(mynod == 0) then
#endif
              mlay_c%E_value = mlay_c%E_value - E_MLP
              do i=1,natom
                 if(mlay_c%q_mm_flag(i)) then
                    mlay_c%dx_repl(i) = mlay_c%dx_repl(i) - qmmm_mlp%dx_mlp(i)
                    mlay_c%dy_repl(i) = mlay_c%dy_repl(i) - qmmm_mlp%dy_mlp(i)
                    mlay_c%dz_repl(i) = mlay_c%dz_repl(i) - qmmm_mlp%dz_mlp(i)
                 end if
              end do
#if KEY_PARALLEL==1
           end if
#endif
        end if

        !==========(2) High level qm/mm calculation ========================
        call HighQM_ene_mlayer(mlay_c%E_value,x,y,z,mlay_c%dx_repl,mlay_c%dy_repl,mlay_c%dz_repl, &
                               qm_control_c%cg_local,NATOM)


        ! printing related
        ! copy high-level energy/gradients
        if(mlay_c%qmlay_print) then
#if KEY_PARALLEL==1
           if(mynod == 0) then
#endif 
              ! since E_value = E_high - (E_low + E_MLP)
              mlay_c%e_pr_high = mlay_c%E_value + mlay_c%e_pr_low + E_MLP
#if KEY_PARALLEL==1
           else
              mlay_c%e_pr_high = zero
           end if
#endif
           ! copy high-level qm gradients for qm/mm atoms
           j = 0
           if(qmmm_mlp%qmmm_mlp) then
              do i=1,natom
                 ! note: 1) dx/y/z_repl are high-level dx/y/z - low-level dx/y/z
                 !       2) atoms with mlay_c%mm_flag(i) == 5, QM atoms
                 !
                 if(mlay_c%q_mm_flag(i)) then
                    j                     = j + 1
                    mlay_c%dx_mpr_high(i) = mlay_c%dx_repl(i) + mlay_c%dx_mpr_low(i) + qmmm_mlp%dx_mlp(i)
                    mlay_c%dy_mpr_high(i) = mlay_c%dy_repl(i) + mlay_c%dy_mpr_low(i) + qmmm_mlp%dy_mlp(i)
                    mlay_c%dz_mpr_high(i) = mlay_c%dz_repl(i) + mlay_c%dz_mpr_low(i) + qmmm_mlp%dz_mlp(i)
                 end if
              end do
           else
              do i=1,natom
                 ! note: 1) dx/y/z_repl are high-level dx/y/z - low-level dx/y/z
                 !       2) atoms with mlay_c%mm_flag(i) == 5, QM atoms
                 !
                 if(mlay_c%q_mm_flag(i)) then
                    j                     = j + 1
                    mlay_c%dx_mpr_high(i) = mlay_c%dx_repl(i) + mlay_c%dx_mpr_low(i)
                    mlay_c%dy_mpr_high(i) = mlay_c%dy_repl(i) + mlay_c%dy_mpr_low(i)
                    mlay_c%dz_mpr_high(i) = mlay_c%dz_repl(i) + mlay_c%dz_mpr_low(i)
                 end if
              end do
           end if
#if KEY_PARALLEL==1
           ! merging gradients, copying here before h-link projections..
           call VDGSUM(mlay_c%dx_mpr_low, mlay_c%dy_mpr_low, mlay_c%dz_mpr_low, 0)
           call VDGSUM(mlay_c%dx_mpr_high,mlay_c%dy_mpr_high,mlay_c%dz_mpr_high,0)

           ! this is not needed and dx/y/z_mlp is only non zero for mynod==0 (master node).
           ! all other nodes, they are zero. so, be cautious.
           !if(qmmm_mlp%qmmm_mlp) then
           !   call VDGSUM(qmmm_mlp%dx_mlp,qmmm_mlp%dy_mlp,qmmm_mlp%dz_mlp,0)
           !end if

           ! printing to imlay_print
           if(mynod == 0) then
#endif
              ! printing 
              jj = mlay_c%imlay_print           ! high-level energy, low-level energy
              if(qmmm_mlp%qmmm_mlp) then
                 ! ai-qm/mm with MLP/delta-MLP
                 write(jj,'(A,3(2X,F20.10))') '!E:',mlay_c%e_pr_high,mlay_c%e_pr_low,E_MLP

                 ! qm region
                 write(jj,'(A,2X,I10)') '!QM region:',qm_main_c%numat
                 do i=1,qm_main_c%numat
                    n = qm_control_c%qminb(i)
                                  ! i,ele no,qmcharge,x,y,z,dx/y/z_high,dx/y/z_low,dx/y/z_mlp
                    write(jj,510) i,qm_main_c%nat(i),qm_control_r(1)%cgqmmm(n),x(n),y(n),z(n), &
                                  mlay_c%dx_mpr_high(n),mlay_c%dy_mpr_high(n),mlay_c%dz_mpr_high(n), &
                                  mlay_c%dx_mpr_low(n), mlay_c%dy_mpr_low(n), mlay_c%dz_mpr_low(n), &
                                  qmmm_mlp%dx_mlp(n),   qmmm_mlp%dy_mlp(n),   qmmm_mlp%dz_mlp(n)
                 end do

                 ! mm region, skip qm region
                 write(jj,'(A,2X,I10)') '!MM region:',j-qm_main_c%numat ! natom-qm_main_c%numat
                 do i=1,natom
                    !if(mlay_c%mm_flag(i) /= 5) then
                    if(mlay_c%mm_flag(i) == 1) then
                       !i,mm_flag, mm_charge,x/y/z/,dx/y/z_high,dx/y/z_low,dx/y/z_mlp
                       write(jj,510) i,mlay_c%mm_flag(i),qm_control_c%cg_local(i),x(i),y(i),z(i), &
                                     mlay_c%dx_mpr_high(i),mlay_c%dy_mpr_high(i),mlay_c%dz_mpr_high(i), &
                                     mlay_c%dx_mpr_low(i), mlay_c%dy_mpr_low(i), mlay_c%dz_mpr_low(i), &
                                     qmmm_mlp%dx_mlp(i),   qmmm_mlp%dy_mlp(i),   qmmm_mlp%dz_mlp(i)
                    end if
                 end do

              else
                 ! ai-qm/mm without MLP/delta-MLP
                 write(jj,'(A,2(2X,F20.10))') '!E:',mlay_c%e_pr_high,mlay_c%e_pr_low

                 ! qm region
                 write(jj,'(A,2X,I10)') '!QM region:',qm_main_c%numat
                 do i=1,qm_main_c%numat
                    n = qm_control_c%qminb(i)
                                  ! i,ele no,qmcharge,x,y,z,dx/y/z_high,dx/y/z_low
                    write(jj,500) i,qm_main_c%nat(i),qm_control_r(1)%cgqmmm(n),x(n),y(n),z(n), &
                                  mlay_c%dx_mpr_high(n),mlay_c%dy_mpr_high(n),mlay_c%dz_mpr_high(n), &
                                  mlay_c%dx_mpr_low(n), mlay_c%dy_mpr_low(n), mlay_c%dz_mpr_low(n)
                 end do

                 ! mm region, skip qm region
                 write(jj,'(A,2X,I10)') '!MM region:',j-qm_main_c%numat ! natom-qm_main_c%numat
                 do i=1,natom
                    !if(mlay_c%mm_flag(i) /= 5) then
                    if(mlay_c%mm_flag(i) == 1) then
                       !i,mm_flag, mm_charge,x/y/z/,dx/y/z_high,dx/y/z_low
                       write(jj,500) i,mlay_c%mm_flag(i),qm_control_c%cg_local(i),x(i),y(i),z(i), &
                                     mlay_c%dx_mpr_high(i),mlay_c%dy_mpr_high(i),mlay_c%dz_mpr_high(i), &
                                     mlay_c%dx_mpr_low(i), mlay_c%dy_mpr_low(i), mlay_c%dz_mpr_low(i)
                    end if
                 end do
              end if
              write(jj,'(A)') '!End'
#if KEY_PARALLEL==1
           end if
#endif
500 format(I7,I3,F10.6, 9(F17.10))
510 format(I7,I3,F10.6,12(F17.10))
        end if

        ! h-link atom coordinates & gradients.
        if(mlay_c%qh_link) then
           ! for coordinates
           call get_hlink_coords(x,y,z,.false.,.false.)

           ! for gradients
           call put_hlink_grads(mlay_c%dx_repl,mlay_c%dy_repl,mlay_c%dz_repl,.true.)
        end if

        !===================================================================
        !if not mts, copy energy and gradients to the main array.
        if( qm_md_master .and. mlay_r(irepl_high)%qmlay_mts .and. &
           (qm_control_r(1)%md_run .or. .not. qmlay_do_energy) ) then
           ! ai-qm/mm energy/gradient corrections are computed, but
           ! gradients are not updated here..
#if KEY_PARALLEL==1
           if(mynod == 0) then
#endif
              if(prnlev >=2 ) then
                 if(qmmm_mlp%qmmm_mlp) then
                    if(qmmm_mlp%qmlp_mode == 0) then
                       ! MLP case
                       write(6,'(1X,A,F18.5,2X,A,F18.5,2X,A,F18.5,2X,A,F8.5,2X,A,F15.5)') &
                                'E_high :',mlay_c%E_value+E_MLP,                          &
                                'E_MLP  :',E_MLP,                                         &
                                'E_corr(U)=E_high-E_low:',mlay_c%E_value,                 &
                                'lambda=',mlay_c%lambda_scale,                            &
                                'E_seqmmm=',qm_control_c%E_total
                    else
                       ! delta-MLP case
                       if(q_mmbond) then
                          write(6,'(1X,A,F18.5,2X,A,F15.5,1X,A,F12.5,A,1X,A,F18.5,A,2X,A,F18.5,2X,A,F8.5)')&
                                'E_high :',mlay_c%E_value+qm_control_c%E_total-E_MM_BOND_C+E_MLP,   &
                                'E_low  :',qm_control_c%E_total,                              &
                                '(+E_low_mm:',-1.0d0*E_MM_BOND_C,')',                         &
                                '(+E_MLP  :', E_MLP,')',                                      &
                                'E_corr(U)=E_high-E_low:',mlay_c%E_value,                     &
                                'lambda=',mlay_c%lambda_scale
                       else
                          write(6,'(1X,A,F18.5,2X,A,F15.5,1X,A,F18.5,A,2X,A,F18.5,2X,A,F8.5)')             &
                                'E_high :',mlay_c%E_value+qm_control_c%E_total-E_MM_BOND_C+E_MLP,   &
                                'E_low  :',qm_control_c%E_total,                              &
                                '(+E_MLP  :',E_MLP,')',                                       &
                                'E_corr(U)=E_high-E_low:',mlay_c%E_value,                     &
                                'lambda=',mlay_c%lambda_scale
                       end if
                    end if
                 else
                    if(q_mmbond) then
                       write(6,'(1X,A,F18.5,2X,A,F15.5,1X,A,F12.5,A,2X,A,F18.5,2X,A,F8.5)')&
                             'E_high :',mlay_c%E_value+qm_control_c%E_total-E_MM_BOND_C,   &
                             'E_low  :',qm_control_c%E_total,                              &
                             '(+E_low_mm:',-1.0d0*E_MM_BOND_C,')',                         &
                             'E_corr(U)=E_high-E_low:',mlay_c%E_value,                     &
                             'lambda=',mlay_c%lambda_scale
                    else
                       write(6,'(1X,A,F18.5,2X,A,F15.5,2X,A,F18.5,2X,A,F8.5)')             &
                             'E_high :',mlay_c%E_value+qm_control_c%E_total-E_MM_BOND_C,   &
                             'E_low  :',qm_control_c%E_total,                              &
                             'E_corr(U)=E_high-E_low:',mlay_c%E_value,                     &
                             'lambda=',mlay_c%lambda_scale
                    end if
                 end if
              end if
              ! lambda scaling factor applied.
              mlay_c%E_value = mlay_c%lambda_scale*mlay_c%E_value
              CTOT           = CTOT + mlay_c%E_value
#if KEY_PARALLEL==1
              !write(6,*) 'mndo 2:',nmlay_mdstp,mlay_c%E_value
           end if

           ! in parallel, they need to be merged.
           call VDGSUM(mlay_c%dx_repl,mlay_c%dy_repl,mlay_c%dz_repl,0)
#endif
           ! copy gradients to save array
           mlay_c%E_value_save  = mlay_c%E_value
           do i=1,natom
              mlay_c%dx_repl_save(i) = mlay_c%lambda_scale*mlay_c%dx_repl(i)
              mlay_c%dy_repl_save(i) = mlay_c%lambda_scale*mlay_c%dy_repl(i)
              mlay_c%dz_repl_save(i) = mlay_c%lambda_scale*mlay_c%dz_repl(i)

              ! restore original charges and igmsel values of the primary system
              ! here, cgqmmm is a copy of the primary qm system.
              igmsel(i) = qm_control_r(1)%igmsel(i)
              cg(i)     = qm_control_c%cgqmmm(i)
           end do
           qmlay_energy_updated =.false. ! gradients are not added to the main array.

        else
           ! regular energy/gradient calls.
           ! without mts, it should be always...
           ! with    mts, not during md simulation loop! 
           ! so, total energy & gradients are updated to the main energy and dx/dy/dz arrays.
#if KEY_PARALLEL==1
           if(mynod == 0) then
#endif
              if(prnlev >=2 ) then
                 if(qmmm_mlp%qmmm_mlp) then
                    if(qmmm_mlp%qmlp_mode == 0) then
                       ! MLP case
                       write(6,'(1X,A,F18.5,2X,A,F18.5,2X,A,F18.5,2X,A,F8.5,2X,A,F15.5)') &
                                'E_high :',mlay_c%E_value+E_MLP,                          &
                                'E_MLP  :',E_MLP,                                         &
                                'E_corr   =E_high-E_MLP:',mlay_c%E_value,                 &
                                'lambda=',mlay_c%lambda_scale,                            &
                                'E_seqmmm=',qm_control_c%E_total

                       ! delta-MLP case
                       else if(q_mmbond) then
                          write(6,'(1X,A,F18.5,2X,A,F15.5,1X,A,F12.5,A,1X,A,F18.5,A,2X,A,F18.5,2X,A,F8.5)')&
                                'E_high :',mlay_c%E_value+qm_control_c%E_total-E_MM_BOND_C+E_MLP,   &
                                'E_low  :',qm_control_c%E_total,                              &
                                '(+E_low_mm:',-1.0d0*E_MM_BOND_C,')',                         &
                                '(+E_MLP  :',E_MLP,')',                                       &
                                'E_corr   =E_high-E_low:',mlay_c%E_value,                     &
                                'lambda=',mlay_c%lambda_scale
                       else
                          write(6,'(1X,A,F18.5,2X,A,F15.5,1X,A,F18.5,A,2X,A,F18.5,2X,A,F8.5)')      &
                                'E_high :',mlay_c%E_value+qm_control_c%E_total-E_MM_BOND_C+E_MLP,   &
                                'E_low  :',qm_control_c%E_total,                              &
                                '(+E_MLP  :',E_MLP,')',                                       &
                                'E_corr   =E_high-E_low:',mlay_c%E_value,                     &
                                'lambda=',mlay_c%lambda_scale
                       end if
                 else
                    if(q_mmbond) then
                       write(6,'(1X,A,F18.5,2X,A,F15.5,1X,A,F12.5,A,2X,A,F18.5,2X,A,F8.5)')&
                             'E_high :',mlay_c%E_value+qm_control_c%E_total-E_MM_BOND_C,   &
                             'E_low  :',qm_control_c%E_total,                              &
                             '(+E_low_mm:',-1.0d0*E_MM_BOND_C,')',                         &
                             'E_corr   =E_high-E_low:',mlay_c%E_value,                     &
                             'lambda=',mlay_c%lambda_scale
                    else
                       write(6,'(1X,A,F18.5,2X,A,F15.5,2X,A,F18.5,2X,A,F8.5)')             &
                             'E_high :',mlay_c%E_value+qm_control_c%E_total-E_MM_BOND_C,   &
                             'E_low  :',qm_control_c%E_total,                              &
                             'E_corr   =E_high-E_low:',mlay_c%E_value,                     &
                             'lambda=',mlay_c%lambda_scale
                    end if
                 end if
              end if
              ! lambda scaling factor applied.
              mlay_c%E_value = mlay_c%lambda_scale*mlay_c%E_value
              CTOT           = CTOT + mlay_c%E_value
#if KEY_PARALLEL==1
              !write(6,*) 'mndo 3:',nmlay_mdstp,mlay_c%E_value
           end if
#endif
           ! gradients
           do i=1,natom
              if(mlay_c%q_mm_flag(i)) then
                 dx(i) = dx(i) + mlay_c%lambda_scale*mlay_c%dx_repl(i)
                 dy(i) = dy(i) + mlay_c%lambda_scale*mlay_c%dy_repl(i)
                 dz(i) = dz(i) + mlay_c%lambda_scale*mlay_c%dz_repl(i)
              end if

              ! restore original charges and igmsel values of the primary system
              ! here, cgqmmm is a copy of the primary qm system.
              igmsel(i) = qm_control_r(1)%igmsel(i)
              cg(i)     = qm_control_c%cgqmmm(i)
           end do

#if KEY_PARALLEL==1
           ! for the parallel purpose. dx_repl/dy_repl/dz_repl are summed over nodes.
           if(qm_md_master .and. qmlay_main .and. qmlay_mts) then
              call VDGSUM(mlay_c%dx_repl,mlay_c%dy_repl,mlay_c%dz_repl,0)
           end if
#endif
           qmlay_energy_updated =.true.  ! gradients are updated to the main array.
        end if

        !only during md, update lambda value for the next md step.
        if(qm_control_r(1)%md_run .and. mlay_c%dlambda_scale /= zero) then
           if(mlay_c%dlambda_scale > zero) then
              ! increase lamdba value.
              mlay_c%lambda_scale = mlay_c%lambda_scale + mlay_c%dlambda_scale
              if(mlay_c%lambda_scale > one) mlay_c%lambda_scale = one ! maximum
           else
              ! decrease lambda value.
              mlay_c%lambda_scale = mlay_c%lambda_scale + mlay_c%dlambda_scale
              if(mlay_c%lambda_scale < zero)  mlay_c%lambda_scale = zero ! minimum
           end if
        end if

        ! return with original pointers for all arrays
        call array_pointers(.true.,1)
        !===================================================================
     end if

     return
  end subroutine MNDENE_MLAYer
  

  !-----------------------------------------------------------------------
  !subroutine setup_option_mlayer(COMLYN,COMLEN)
  !   !
  !   implicit none 
  !   CHARACTER(len=*):: COMLYN
  !   INTEGER ::  COMLEN
  !
  !   return
  !end subroutine setup_option_mlayer


  !-----------------------------------------------------------------------
  subroutine HighQM_ene_mlayer(GTOT,X,Y,Z,DX,DY,DZ,CGX,NATOM)
     !
     ! local copy of GUKENE_mlayer to call different routines in gukini.F90
     !
     use chm_kinds
     use dimens_fcm
     use consta
     use number
     use gamess_fcm
     use stream
     use mndo97
     use parallel
     use scalar_module
     use qm1_info, only : qm_control_c,qm_main_c
     use gukini_mod

     real(chm_real) GTOT,X(*),Y(*),Z(*),DX(*),DY(*),DZ(*)
     real(chm_real) CGX(*)
     INTEGER NATOM
     LOGICAL lexist
     !     
#if KEY_GAMESSUK==1
     INTEGER INIT,ICODE,IVER,N1
     LOGICAL STARTUP
     CHARACTER(len=3) TMPSTR
     CHARACTER(len=40) MSG
#endif
#if KEY_GAMESS==1 || KEY_QCHEM==1 || KEY_QTURBO==1 || KEY_G09==1 /*gamess*/
     INTEGER ICHARM, N1
     !     
     real(chm_real) E, EG
     COMMON /FUNCT/ E, EG(3*MAXGMS)
     !     
     INTEGER(gms_int) NAT,ICH,MUL,NUM,NX,NE,NA,NB,IAN
     real(chm_real) ZAN, C
     COMMON /INFOA / NAT,ICH,MUL,NUM,NX,NE,NA,NB,ZAN(MAXGMS),C(3,MAXGMS),IAN(MAXGMS)
     !
     real(chm_real) HMLTN
     LOGICAL MPCWFN
     COMMON /MPCLNK/ HMLTN,MPCWFN
     !     
     real(chm_real) QMMMRP, EDIESL
     real(chm_real),parameter :: RBR=ONE/BOHRR
     INTEGER IPT,N
     logical mopac
#endif /*  (gamess)*/

     LOGICAL QDONE,LQMEWD_local
     CHARACTER CFN*17,CRN*13
     INTEGER I,natqm_2,iatom

     !     
     !     Are there any QM atoms?
     if(NGAMES == 0) return
     !
     CFN='gukini_mlayer.src'
     CRN='GUKENE_mlayer'

     !     
     ! Separate to parallel/parallel
     BLFACTOR=ONE
     !
     ! Zero the QM charges, in case they are not calculated
     do i=1,ngames
        QMMUL(i) = ZERO
        QMLOW(i) = ZERO
        QMKOL(i) = ZERO
     end do
     !
     !
     ! Counter for true QM atoms
     natqm_2 = qm_main_c%numat
     !
     ! This forces a restart for all components of this job
     !
#if KEY_GAMESSUK==1
     STARTUP = QINIGM
#endif 

     ! GAMESS
#if KEY_GAMESS==1                           /*gamess*/
     ! Update coordinates
     n = 0
     do i=1,natqm_2
        iatom  = iabs(qm_control_c%qminb(i))
        n      = n + 1
        c(1,n) = x(iatom)*RBR
        c(2,n) = y(iatom)*RBR
        c(3,n) = z(iatom)*RBR
     end do

     call ch2gms(.FALSE.)
     call cgrep(NATOM,QMMMRP,DX,DY,DZ)
     if(prnlev >= 6) write(outu,'(A,2F17.6)') 'QM/MM repulsion is (au,kcal) ',QMMMRP,QMMMRP*TOKCAL
     !     
     ! From gamess/grd1.src in case of other WFns.
     if(NCHMAT /= 0) then
        do i=1,NCHMAT
           DXELMM(i)=ZERO
           DYELMM(i)=ZERO
           DZELMM(i)=ZERO
        end do
     end if
     !     
     QINIGM=.FALSE.
     call gamess

     ! If this is DIESEL calculation then get the energy
     CALL DIESELE(E)

     ! energy
#if KEY_PARALLEL==1
     if(mynod == 0) then
#endif
        GTOT=GTOT + E*TOKCAL*BLFACTOR+QMMMRP*TOKCAL
#if KEY_PARALLEL==1
     else
        GTOT=ZERO
     end if
#endif

     ! gradients: MM atoms, without igmsel(i)=5, unless BLUR !!
     n = 0
     do i=natqm_2+1,natom
        mm = qm_control_c%mminb1(i)
        if(mm>0) then
           n = n + 1
           dx(mm) = dx(mm) - DXELMM(N)*TOKCAL*RBR*BLFACTOR
           dy(mm) = dy(mm) + DYELMM(N)*TOKCAL*RBR*BLFACTOR
           dz(mm) = dz(mm) + DZELMM(N)*TOKCAL*RBR*BLFACTOR

           !if(prnlev > 6) then
           !   write(outu,334) I,N, - DXELMM(N)*TOKCAL*RBR &
           !                      , - DYELMM(N)*TOKCAL*RBR &
           !                      , - DZELMM(N)*TOKCAL*RBR
           !end if
        end if
     end do

     ! gradients: QM atoms (in parallel they are already summed up!)
#if KEY_PARALLEL==1
     if(mynod == 0) then
#endif 
        do i=1,natqm_2
           iatom  = iabs(qm_control_c%qminb(i))
           ipt=3*(i-1)+1
           dx(iatom) = dx(iatom) + EG(ipt)  *TOKCAL*RBR*BLFACTOR
           dy(iatom) = dy(iatom) + EG(ipt+1)*TOKCAL*RBR*BLFACTOR
           dz(iatom) = dz(iatom) + EG(ipt+2)*TOKCAL*RBR*BLFACTOR

           !if(prnlev > 6) then
           !   write(outu,334) I,i, EG(IPT)  *TOKCAL*RBR &
           !                      , EG(IPT+1)*TOKCAL*RBR &
           !                      , EG(IPT+2)*TOKCAL*RBR
           !end if
        end do
#if KEY_PARALLEL==1
     end if
#endif
334  FORMAT(2I10,3F14.6,3F14.9)
#endif                                      /*gamess*/

     ! GAMESS-UK
#if KEY_GAMESSUK==1                         /*gamessuk*/
     QINIGM=.FALSE.
     INIT=0
     IVER=5
     LQMEWD_local = .false.
     call gamess(INIT,ICODE,STARTUP,LQMEWD_local,IVER)

     if(icode /= 0)then
        if(icode == 1)then
           msg = 'SCF convergence failure '//tmpstr
        else if(icode /= 0)then
           write(tmpstr,'(i3)')icode
           write(outu,*)'Return code '//tmpstr
        end if
        call wrndie(-1,'<GAMESS-UK>',msg)
     end if

     !     
     ! Recover Energy and gradient here
     ! quantum charges written into first (QM) section of CGQMMM 
     call gms2chm(GTOT,DX,DY,DZ,CGQMMM,GMSMAP,BLFACTOR,NATOM)
#endif                                      /*gamessuk*/

     ! Qchem
#if KEY_QCHEM==1 && KEY_GAMESS==0 
     call QCHEM_mlayer(E,X,Y,Z,DX,DY,DZ,CGX)
     GTOT=GTOT + E*TOKCAL

     !     We obtain E,DX,DY,DZ from Q-chem output
     !     We put X,Y,Z to input for Q-chem
#endif 

!!     ! G09
!!#if KEY_G09==1
!!     call qg09(E,DX,DY,DZ,CGX,AMASSX,IACX)
!!     GTOT=GTOT + E*TOKCAL
!!#endif
!!
!!     ! Turbomole
!!#if KEY_QTURBO==1
!!     CALL QTURBO(E,DX,DY,DZ,CGX,AMASSX,IACX,NDD1,DD1,QSECD,IUPT,JUPT)
!!     GTOT=GTOT + E*TOKCAL
!!
!!     !     We obtain E,DX,DY,DZ from TURBOMOLE output
!!     !     We put X,Y,Z to input for TURBOMOLE
!!#endif 

!!     ! Qchem
!!#if KEY_QCHEM==1 && KEY_GAMESS==0
!!     ! Reading QM Charges from charges.dat file. May not be used.
!!     if(QQCHARG) then
!!        if(QQCRP) then
!!           inquire(file=FILCHRG(1:LCH),exist=lexist)
!!        else
!!           inquire(file='charges.dat',exist=lexist)
!!        end if
!!        if(lexist) then
!!           !write(*,*)'READING MULIKEN CHARGES'
!!           if(QQCRP) then
!!              open(unit=11,file=FILCHRG(1:LCH),status='old')
!!           else
!!              open(unit=11,file='charges.dat' ,status='old')
!!           end if
!!           rewind(11)
!!           if(mynodg == 0) then
!!
!!             ! need to work on later when we deal with qchem routine
!!             do i=1, natom
!!                if((igmsel(i) == 1).or.(igmsel(i) == 2)) then
!!                   read(11,'(3X,F19.16)') QMMUL(I)
!!                   CGX(I)=QMMUL(I)
!!                else
!!#if KEY_PARALLEL==1
!!                   if(mynodg > 0) CGX(I)=ZERO
!!#endif
!!                end if
!!             end do
!!           end if
!!#if KEY_PARALLEL==1
!!           call gcomb(CGX,NATOM)
!!#endif
!!        end if
!!        if(.not. QSMBP) call unlink('charges.dat')
!!     end if
!!#endif 

#if KEY_GAMESSUK==1 || KEY_GAMESS==1 || KEY_QCHEM==1 || KEY_QTURBO==1 || KEY_G09==1
     call GETQMCHG_mlayer(natqm_2,cgx,qm_control_c%qminb)
#endif

     return
  end subroutine HighQM_ene_mlayer

  !-----------------------------------------------------------------------
  subroutine Get_dual_QM_from_CHM(igmselm)
     !
     !     Find the atoms defined as QM atoms and get them ready for MNDO97 and ai/dft method
     !
     use chm_kinds
     use dimens_fcm
     use number
     use exfunc
     use param
     use psf
     use rtf, only: atct
     use stream
     use gamess_fcm
     use mndo97
     use linkatom, only: findel
     ! 
     use qm1_info, only : qm_control_r,qm_control_c,qm_main_c,mm_main_c

     implicit none
     !
     !
     logical :: qlink,clink
     integer :: igmselm(*)
     !
     !charcater(len=10),allocatable,dimension(:) :: aatom
     !real(chm_real),allocatable,dimension(:)    :: azunc
     real(chm_real) :: azunc
     !
     integer:: i,n,nslct,natmm,natlnk,natlnkh,nlatq,ii
     character(len=6) :: ele
     logical          :: qprt
     !
     QPRT=.TRUE.

     ! fill qminb array
     nlatq  =0
     do i=1,mm_main_c%natom
        ! igmselm(i)==1 : pure qm atom
        !           ==2 : QQH h-link atom
        !           ==3 : h-link atom (only for high qm-region)
        !           ==0 : pure mm atom
        !           ==5 : mm atom excluded from qm/mm calculation 
        if(igmselm(i)==1 .or. igmselm(i)==2 .or. igmselm(i)==3) then
           nlatq       = nlatq+1
           qm_control_c%qminb(nlatq)= i
        end if
     end do

     ! make a copy of the MM charges 
     if(qgmrem) then
        !qm_control_c%cgqmmm(1:mm_main_c%natom) = qm_control_r(1)%cgqmmm(1:mm_main_c%natom)
        do i=1,mm_main_c%natom
           if(igmselm(i)==1 .or. igmselm(i)==2 .or. igmselm(i)==3) then
              qm_control_c%cg_local(i)=zero
           else
              qm_control_c%cg_local(i)= qm_control_r(1)%cgqmmm(i)
           end if

           ! this is saved here for the (modified) psf charges
           qm_control_c%cgqmmm(i)  = cg(i)
        end do
     end if
  
     ! then, assign nuclear charges:
     !call chmalloc('mndo97_mlayer.F90','get_dual_qm_from_chm','azunc',qm_main_c%numat,crl=azunc)
     do i=1,qm_main_c%numat
        ii= qm_control_c%qminb(i)
        call findel(ATCT(iac(ii)),amass(ii),ii,ELE,azunc,QPRT)
        !
        ! assign neclear charges
        if(igmselm(ii)==2 .or. igmselm(ii)==3) then
           ! H atom
           qm_main_c%nat(i)                 = 1
           qm_control_c%AZNUC_local(i)      = one
           qm_control_c%CAATOM_local(i)(1:6)= ' H    '
        else 
           ! regular qm atoms
           qm_main_c%nat(i)                 = int(azunc)
           qm_control_c%AZNUC_local(i)      = azunc
           qm_control_c%CAATOM_local(i)(1:6)= ELE
        end if
     end do
     !
     ! no. of mm atoms (for the high-level qm/mm calc.)
     natmm=natom-qm_main_c%numat
     !
     ! number of QQ H-link atoms
     natlnk=0
     do i = 1,natom
        if(igmselm(i) == 2) natlnk=natlnk+1
     end do
     ! number of (QM-MM) H-link atoms
     natlnkh=0
     do i = 1,natom
        if(igmselm(i) == 3) natlnkh=natlnkh+1
     end do

     !
     ! Write out atomic information
     if(prnlev >= 2) then
        write (outu,'(/,1x,A,/)') ' Get_dual_QM_from_CHM> Some atoms will be treated quantum mechanically.'
        write (outu,'(4(8X,A,I5,/),/)') &
             ' The number of quantum mechanical atoms                  = ',qm_main_c%numat, &
             ' The number of QM/MM QQH-link atoms  (from low-level qm) = ',NATLNK, &
             ' The number of QM/MM H-link(*) atoms (for high-level qm) = ',NATLNKH, &
             ' The number of molecular mechanical atoms                = ',NATMM
     end if
     !
     ! clean-up memory.
     !call chmdealloc('mndo97_mlayer.F90','get_dual_qm_from_chm','azunc',qm_main_c%numat,crl=azunc)

     return
  end subroutine Get_dual_QM_from_CHM


  !-----------------------------------------------------------------------
  subroutine copsel_dual(numat,igmselm,islct,jslct,hlink)
     !
     !     Copies selection vector to common block for MLAYered QM/MM
     !
     !     IGMSEL(I) = 5  MM atom to be excluded from QM/MM interaction (for QQ atom case.)
     !     IGMSEL(I) = 3  H-Link atom replaces MM atoms (note that this atom is not selected in islct).
     !     IGMSEL(I) = 2  QQ H-link atom
     !     IGMSEL(I) = 1  QM atom
     !     IGMSEL(I) = 0  MM atom
     !
     !     MM atom in position close to link atom is excluded from interaction
     !     of external charges to QM region. Instead of this atom is already
     !     a link atom so no need for two atoms in one place!
     !
     use chm_kinds
     use exfunc
     use dimens_fcm
     use gamess_fcm
     use stream
     use psf
     use number
     ! use mndo97
     use chutil,only:getres,atomid
     use qm1_info

     !
     implicit none
     !
     integer :: numat
     integer :: igmselm(natom),islct(natom),jslct(natom)
     logical :: hlink
     !
     !
     integer :: i,j,i1,i2,j1,n,is,iq,nlatq
     character(len=4) :: SID, RID, REN, AC
     logical :: lnflag,qglnk,qclnk
     integer :: ln
     integer :: num_h_host,ncnt
     integer,allocatable :: ihostguest_tmp(:,:)
     !

     ! check if the irepl (2nd) qm region is a subset of the primary (1st) qm region
     numat      = 0
     num_h_host = 0
     do i=1,natom
        if(islct(i)==1) then
           if(.not. (igmsel(i)==1 .or. igmsel(i)==2)) then
              call wrndie(-5,'<copsel_dual>','2nd QM region should be a subset of 1st QM region.')
           end if

           ! 2nd qm region
           numat        = numat + 1
           igmselm(i)   = islct(i)
           if(igmsel(i) == 1) igmselm(i) = 1 ! pure qm atom
           if(igmsel(i) == 2) igmselm(i) = 2 ! QQH atom
        else
           ! mm atoms
           igmselm(i) = 0                    ! (pure) mm atoms (note below for h-link)
                                             ! see below for MM atoms connected to QQH atom
        end if
     end do

     ! for mm atoms connected to QQH link atom
     !do i=1,nbond
     !   i1=ib(i)
     !   i2=jb(i)
     do i=1,qm_bond_r(1)%nbond_qm
        i1 = qm_bond_r(1)%i_mm_bond(1,i)
        i2 = qm_bond_r(1)%i_mm_bond(2,i)
        ! i1 == qm atom .and. i2 == excluded mm atom
        if(islct(i1) == 1 .and. igmsel(i2) == 5) then
           igmselm(i2) = 5                   ! mm atoms execluded from qm/mm interaction
        end if

        ! i2 == qm atom .and. i1 == excluded mm atom
        if(islct(i2) == 1 .and. igmsel(i1) == 5) then
           igmselm(i1) = 5                   ! mm atoms execluded from qm/mm interaction
        end if
     end do

     ! find h-link atoms for qm-mm cut, where a new h-link atom to be
     ! introduced along the qm-mm bond, from the qm atom side, but the
     ! introduced h-link atom replaces the mm atom only during the the qm/mm
     ! calculation but not during the mm energy calculation.
     !
     ! note that QQH H-link atom already presents along the qm-mm bond, it
     ! must skip for the particular qm-mm bond.
     if(hlink) then
        ! many mm atoms (<10) can be connected to any qm atom.
        ncnt = 0
        do i=1,natom
           if(jslct(i)==1) then         ! this do loop does not check QQH atom.
              ncnt = ncnt + 1           ! so, it must be checked below.
              if(islct(i) /=1) call wrndie(-5,'<copsel_dual>', &
                 'H-link selection must be a subset of the main QM region.')
           end if
        end do
        if(ncnt > 0) then
           allocate(ihostguest_tmp(2,10*ncnt))
           ihostguest_tmp = 0

           ! count no. of h-link atoms
           num_h_host = 0
           !loopnb: do i=1,nbond
           !   i1=ib(i)
           !   i2=jb(i)
           loopnb: do i=1,qm_bond_r(1)%nbond_qm
              i1 = qm_bond_r(1)%i_mm_bond(1,i)
              i2 = qm_bond_r(1)%i_mm_bond(2,i)

              ! if jslct-ed atom is qqh atom, it will be skipped.
              if((jslct(i1) == 1 .and. igmselm(i1) == 2) .or. &
                 (jslct(i2) == 1 .and. igmselm(i2) == 2)) then
                 cycle loopnb
              end if

              ! if i1 is qm atom and i2 is mm atom (not qm atom)
              ! this also natually skips igmsel == 5 case.
              if(jslct(i1) == 1 .and. igmselm(i2) == 0) then
                 num_h_host = num_h_host + 1
                 ihostguest_tmp(1,num_h_host) = i1 ! qm atom (link host)
                 ihostguest_tmp(2,num_h_host) = i2 ! mm atom (link guest)
                 !igmselm(i2)                  = 3  ! h-link atom in place of mm atom
                 cycle loopnb
              end if

              ! if i2 is qm atom and i1 is mm atom (not qm atom)
              if(jslct(i2) == 1 .and. igmselm(i1) == 0) then
                 num_h_host = num_h_host + 1
                 ihostguest_tmp(1,num_h_host) = i2 ! qm atom (link host)
                 ihostguest_tmp(2,num_h_host) = i1 ! mm atom (link guest)
                 !igmselm(i1)                  = 3  ! h-link atom in place of mm atom
                 cycle loopnb
              end if
           end do loopnb

           !
           if(num_h_host >= 1) then
              if(allocated(mlay_c%ihostguest)) deallocate(mlay_c%ihostguest)
              if(allocated(mlay_c%xyz_hlink))  deallocate(mlay_c%xyz_hlink)
              if(allocated(mlay_c%qcheck_tmp)) deallocate(mlay_c%qcheck_tmp)
              if(allocated(mlay_c%r_h_ref))    deallocate(mlay_c%r_h_ref)
              allocate(mlay_c%ihostguest(2,num_h_host))
              allocate(mlay_c%xyz_hlink(6,num_h_host))
              allocate(mlay_c%qcheck_tmp(num_h_host))
              allocate(mlay_c%r_h_ref(num_h_host))
              mlay_c%num_h_link = num_h_host
              mlay_c%ihostguest(1:2,1:num_h_host) = ihostguest_tmp(1:2,1:num_h_host)
              mlay_c%r_h_ref(1:num_h_host)        = zero
           else
              ! no qm-mm bonds are selected.
              call wrndie(0,'<copsel_dual>','No H-link atoms selected.')
              mlay_c%num_h_link = 0
              mlay_c%qh_link    =.false.  ! turn off
           end if

           ! h-link atoms (mm atoms in the main array but replace with h-link atom)
           numat = numat + num_h_host 
           do i=1,num_h_host
              igmselm(mlay_c%ihostguest(2,i)) = 3  ! h-link atom to replace mm/link guest atom
           end do

           !
           deallocate(ihostguest_tmp)
        else
           call wrndie(0,'<copsel_dual>','No H-link atoms selected.')
           mlay_c%num_h_link = 0
           mlay_c%qh_link    =.false.  ! turn off
        end if
     end if

     !
     if(prnlev>=2) then
        write(outu,118)
        write(outu,120) 'Mlayered QM/MM Set. Info for the 2nd QM region'
        write(outu,120) 'Classical atoms excluded from the 2nd QM calculation'
     end if
118  format('------------------------------------------------')
120  format('MNDINI_MLAYer> ',A,':')
122  format(10X,I5,4(1X,A4))
123  format(10X,I5,4(1X,A4),1X,'*')
125  format(10X,'NONE.')
     n=0
     do i=1,natom
        if(igmselm(i)==5) then
           call atomid(i,sid,rid,ren,ac)
           if(prnlev>=2) write(outu,122) I,SID,RID,REN,AC
           n=n+1
        end if
     end do
     if(prnlev>=2) then
        if(n==0) write(outu,125)
        write(outu,120) 'Quantum mechanical atoms (pure QM)'
     end if
     n=0
     do i=1,natom
        if(igmselm(i)==1) then
           ! pure qm atom
           call atomid(i,sid,rid,ren,ac)
           if(prnlev>=2) write(outu,122) I,SID,RID,REN,AC
           n=n+1
        end if
     end do
     if(prnlev>=2) then
        if(n==0) write(outu,125)
        write(outu,120) 'Quantum mechanical QQHydrogen link atoms'
     end if
     n=0
     do i=1,natom
        if(igmselm(i)==2) then
           ! QQH link atom
           call atomid(i,sid,rid,ren,ac)
           if(prnlev>=2) write(outu,122) I,SID,RID,REN,AC
           n=n+1
        end if
     end do
     if(hlink) then
        if(prnlev>=2) then
           if(n==0) write(outu,125)
           write(outu,120) 'Quantum mechanical H-link atoms to replace MM atoms (*)'
        end if
        n=0
        do i=1,natom
           if(igmselm(i)==3) then
              ! H-link atom only for the 2nd qm region
              call atomid(i,sid,rid,ren,ac)
              if(prnlev>=2) write(outu,123) I,SID,RID,REN,AC
              n=n+1
           end if
        end do
     end if
     if(prnlev>=2) then
        if(n==0) write(outu,125)
        write(outu,118)
     end if

     ! do the work for high-level qm/mm prep.
     NGAMES=numat

     ! all for partial charges on any qm atom
     n = 0
     loopqm: do i=1,natom
        if(iabs(igmselm(i))==1 .or. iabs(igmselm(i))==2 .or. iabs(igmselm(i))==3) then
           n = n + 1
           !cc               LNFLAG=.FALSE.
           !ccc first check the flag for QINP for all atoms.
           !cc               if(QQINP) then
           !cc                  if(.not.LNFLAG) FQQCHG(N)=CG(i)
           !cc               else

           if(n <= MAXGMS) then
              FQQCHG(n)=-THOSND
           else
              if(prnlev>=2) write(outu,'(A,I8)') 'Number of QM atoms exceeded (REPLICA?)',n
              cycle loopqm
           end if

           !cc               end if
           !
           ! only output for the currrent qm region.
           if(igmselm(i) > 0) then
              if(prnlev >= 6) write(outu,'(A,2I10,A,F15.5)') 'MLAYer: ATOM(',i,n,') has QNUC: ',FQQCHG(n)
           end if
        end if
     end do loopqm

     return
  end subroutine copsel_dual


  subroutine get_hlink_coords(x,y,z,Lxyz_get,Lref_save)
    !
    ! calculate the position of h-link atoms based on the
    ! qm (host) and mm (guest) atom and their coordinates.
    !
    ! Lxyz_get : .true.   put h-link atom at (relatively) 1.0 A away from qm host
    !            .false.  restore the original mm (guest) atom position.
    ! Lref_save: .true. to store reference values.
    !
    use chm_kinds
    use dimens_fcm
    use number
    use qm1_info, only: mlay_c

    implicit none
    real(chm_real):: x(*),y(*),z(*)
    logical ::  Lxyz_get,Lref_save

    ! local
    integer :: i,iatom,jatom
    real(chm_real) :: r_xyz(3)

    ! locate H-link atom at 1.0d0 distance away from xyz(iatom).
    ! we need to work on it to fluctuate based on the QM-MM bond
    ! distance.
    if(Lxyz_get) then
       ! for a first call, check the distances and save it for later uses...
       ! so, it stores reference values for r_h distances for qm-mm pairs
       if(Lref_save) then
          do i=1,mlay_c%num_h_link
             iatom    = mlay_c%ihostguest(1,i)
             jatom    = mlay_c%ihostguest(2,i)
             r_xyz(1) = x(jatom) - x(iatom)
             r_xyz(2) = y(jatom) - y(iatom)
             r_xyz(3) = z(jatom) - z(iatom)
             ! scaling factor
             mlay_c%r_h_ref(i)=one/sqrt(r_xyz(1)**2+r_xyz(2)**2+r_xyz(3)**2)
          end do
       end if

       do i=1,mlay_c%num_h_link 
          iatom                 = mlay_c%ihostguest(1,i)
          jatom                 = mlay_c%ihostguest(2,i)
          mlay_c%xyz_hlink(1,i) = x(iatom)
          mlay_c%xyz_hlink(2,i) = y(iatom)
          mlay_c%xyz_hlink(3,i) = z(iatom)

          mlay_c%xyz_hlink(4,i) = x(jatom)
          mlay_c%xyz_hlink(5,i) = y(jatom)
          mlay_c%xyz_hlink(6,i) = z(jatom)

          x(jatom) = x(iatom)+mlay_c%r_h_ref(i)*(mlay_c%xyz_hlink(4,i)-mlay_c%xyz_hlink(1,i))
          y(jatom) = y(iatom)+mlay_c%r_h_ref(i)*(mlay_c%xyz_hlink(5,i)-mlay_c%xyz_hlink(2,i))
          z(jatom) = z(iatom)+mlay_c%r_h_ref(i)*(mlay_c%xyz_hlink(6,i)-mlay_c%xyz_hlink(3,i))
       end do
    else
       ! restore the coordinates of mm (guest) atoms
       do i=1,mlay_c%num_h_link
          jatom    = mlay_c%ihostguest(2,i)
          x(jatom) = mlay_c%xyz_hlink(4,i)
          y(jatom) = mlay_c%xyz_hlink(5,i)
          z(jatom) = mlay_c%xyz_hlink(6,i)
       end do
    end if

    return
  end subroutine get_hlink_coords


  subroutine put_hlink_grads(dx,dy,dz,qnoproj)
    ! 
    ! This subroutine projects out the force along the H-link atom and
    ! QM atoms bonded to it.
    !
    use chm_kinds
    use dimens_fcm
    use number
    use qm1_info, only: mlay_c

    implicit none
    real(chm_real) :: dx(*),dy(*),dz(*)
    logical        :: qnoproj

    integer :: i,iic,ii,iatom,jatom
    real(chm_real) ::  SP,S2,SF2,SS,S_xyz(3),SMF(3,2),P_xyz(3), &
                       r_xyz(3),r_xyz_dist,prjf
    logical :: done
    real(chm_real),parameter :: R_TOLI=0.0001d0,FACTF = R_TOLI*R_TOLI*1.0d-6

    !
    !FACTF = 1.0d-3
    !FACTF = R_TOLI*R_TOLI*FACTF*FACTF   ! R_TOLI^2 * FACTF^2

    if(Qnoproj) then
       ! do not project forces
       ! as R_H = R_QM + g*(R_MM - R_QM)
       !    gradient contriution is
       !    dE/dR_QM = dE/dR_H * (1-g); dE/dR_MM = dE/dR_H * g
       do iic=1,mlay_c%num_h_link
          iatom = mlay_c%ihostguest(1,iic)
          jatom = mlay_c%ihostguest(2,iic)

          ! For link host atom (qm-atom)
          prjf      =(one-mlay_c%r_h_ref(iic))
          dx(iatom) = dx(iatom)+prjf*dx(jatom)
          dy(iatom) = dy(iatom)+prjf*dy(jatom)
          dz(iatom) = dz(iatom)+prjf*dz(jatom)

          ! For link guest atom (mm-atom)
          prjf      = mlay_c%r_h_ref(iic)
          dx(jatom) = dx(jatom)*prjf
          dy(jatom) = dy(jatom)*prjf
          dz(jatom) = dz(jatom)*prjf
       end do
    else
       mlay_c%qcheck_tmp(1:mlay_c%num_h_link)=.FALSE.
       loopdo1: do
          do iic=1,mlay_c%num_h_link
             ! Find the unit direction vector
             r_xyz(1:3)= mlay_c%xyz_hlink(4:6,iic)-mlay_c%xyz_hlink(1:3,iic)
             r_xyz_dist= one/SQRT(r_xyz(1)**2+r_xyz(2)**2+r_xyz(3)**2)
             SMF(1:3,1)=-r_xyz(1:3)*r_xyz_dist  ! for i-th atom
             SMF(1:3,2)= r_xyz(1:3)*r_xyz_dist  ! for j-th atom
 
             SP = zero
             S2 = zero
             SF2= zero
             do i=1,2
                ii        =mlay_c%ihostguest(i,iic)
                S_xyz(1:3)=SMF(1:3,i)
                P_xyz(1)  =dx(ii)*S_xyz(1)
                P_xyz(2)  =dy(ii)*S_xyz(2)
                P_xyz(3)  =dz(ii)*S_xyz(3)
                SP        =SP + P_xyz(1)   +P_xyz(2)   +P_xyz(3)
                S2        =S2 + S_xyz(1)**2+S_xyz(2)**2+S_xyz(3)**2
                SF2       =SF2+ dx(ii)**2  +dy(ii)**2  +dz(ii)**2
             end do

             SS = S2*SF2*FACTF               ! refer FACTF above.
             if(SP*SP < SS) mlay_c%qcheck_tmp(iic)=.true.
             SP=SP/S2
             !
             ! ... Subtract parallel contribution
             !
             do i=1,2
                ii    =mlay_c%ihostguest(i,iic)
                dx(ii)=dx(ii)-SP*SMF(1,i)
                dy(ii)=dy(ii)-SP*SMF(2,i)
                dz(ii)=dz(ii)-SP*SMF(3,i)
             end do
          end do ! iic=1,mlay_c%num_h_link
          !
          ! ... End of loop over constraints: Check convergence
          !
          DONE = .TRUE.
          do iic=1,mlay_c%num_h_link
             DONE=(DONE.AND. mlay_c%qcheck_tmp(iic))
          end do
          if(done) exit loopdo1 
       end do loopdo1
    end if

    return
  end subroutine put_hlink_grads

#endif            /*mndo97*/
end module mlay_mndo97
