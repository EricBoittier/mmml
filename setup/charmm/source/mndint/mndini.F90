#if KEY_MNDO97==1 /*mndo97*/
SUBROUTINE MNDINI(COMLYN,COMLEN)
  !-----------------------------------------------------------------------
  !     This is the interface for running modified version of MNDO97 with 
  !     CHARMM for QM/MM calculations
  !
  !     Kwangho Nam, February 2012.
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
  use mndgho
  use quantm, only : natom_check,xim,yim,zim
  use nbndqm_mod
  use ewald_1m, only : lewald,kappa,erfmod
  !!!use erfcd_mod,only : erfmod
  use pme_module, only : QPME
  !
  use param
  use psf
  use select
  use stream
  !
  use mndnbnd_module, only: ch2mnd
  use qmmm_interface, only : qmmm_init_set,qmmm_load_parameters_setup_qm_info,qmmm_Ewald_init, &
                             qm_cpmd_init,qm_lookup_init
  use qm1_info, only  : qm_main_r,mm_main_r,qm_control_r
  ! D3BJ & H4 corrections
  use dftd3_mndo, only: init_dftd3,r_autoang
  use H4_mndo, only : q_h4corr,h4_correction_setup
  !
  use mndgho_module, only : GHOHYB
  use qm1_energy_module,only: dr_width,rval_min,rval_max,dr_width_beta,rval_min_beta,rval_max_beta, &
                              find_unique_qm

  !
  use leps

  !
  !     Adjust nonbonded group list for IMAGES.
  use image
  !     Adjust nonbonded group list for simple pbc.
#if KEY_PBOUND==1
  use pbound
#endif
  !
#if KEY_FLUCQ==1
  !     Check for FLUCQ
  use flucq 
#endif
  !
  implicit none
  !
  !
  CHARACTER(len=*):: COMLYN
  INTEGER ::  COMLEN
  !
  INTEGER :: i,natgho,natclink,EWMODE,NQMEWD,kmaxq,ksqmaxq,kmaxxq,kmaxyq,kmaxzq
  integer :: nqmtheory,nqmcharge,nspin
  LOGICAL :: QDONE, QIMAGE, ISNBDS, qcheck,LEWMODE,NOPMEwald,QMMM_NoDiis,clink
  logical :: QSRP_PhoT, QNoMemIncore, q_cpmd, q_fast_pme, q_lookup_setup, q_lookup_read, q_lookup_first, &
             q_dxl_bomd,q_analysis,q_bond_order,q_m_charge,q_fockmd
  integer :: i_lookup_unit,K_order,N_scf_step,iiunit,imax_fdiss,iopt_fdiss
  real(chm_real) :: cggho,scfconv,qmcharge

  integer,allocatable,dimension(:) :: islct,jslct,lslct
  ! D3BJ & H4 corrections
  integer :: n, iunit_disp
  real(chm_real) :: xdisp(3,nndim)
  logical :: q_d3bj_read

  !
#if KEY_FLUCQ==1
  if(qfluc) call wrndie(-2,'<MNDINI>', 'FLUCQ is not implmented into MNDO97.')
#endif
  !
  ! Initial check up
  qimage =.false.
  if(.not.useddt_nbond(bnbnd)) call wrndie(-3,'<MNDINI>','Nonbond data structure is not defined.')
#if KEY_PBOUND==1
  if(.not.qBoun) then
#endif
     if(ntrans.gt.0) then
        !if(lgroup) then
           qimage =.true.
        !   if(.not.useddt_image(bimag)) call wrndie(-3,'<MNDINI>', &
        !        'Image nonbond data structure is not defined.')
        !else
        !   call wrndie(-1,'<MNDINI>','QM/MM do not interact with Images under Atom Based Cutoff.')
        !end if
     end if
#if KEY_PBOUND==1
  end if
#endif
  !
  qmused = .true.      ! safe here since next are the allocations, ...
  call allocate_gamess ! try to reduce from MAXA to NATOM
  !new
  call chmalloc('mndini.src','MNDINI','islct',natom,intg=islct)    ! for qm     atoms
  call chmalloc('mndini.src','MNDINI','jslct',natom,intg=jslct)    ! for C-link atoms
  call chmalloc('mndini.src','MNDINI','lslct',natom,intg=lslct)    ! for GHO    atoms
  
  ! for main qm atom selection:
  call selcta(COMLYN,COMLEN,islct,x,y,z,wmain,.true.)

  ! MNDO97 already has several link atom options
  ! 1) GHO atoms
  qlink = (indxa(COMLYN, COMLEN, 'GLNK') .gt. 0)
  if (qlink) then
     if(prnlev.ge.2) write(outu,22) 'GLNK: GHO boundary atoms are used'
     call selcta(COMLYN,COMLEN,lslct,x,y,z,wmain,.true.)
     qcheck=.true.
     call ghohyb(natom,islct,lslct,NBOND,IB,JB,cggho,X,Y,Z,CG,QCHECK,qlink)
     if(.not.qcheck) call wrndie(-5,'<MNDINI>','The program will stop at GHOHYB.')
  end if
  ! 
  ! 2) Connection atom (Currenlty only C-connection atom)
  clink  = (indxa(COMLYN,COMLEN,'CLNK') .gt. 0)
  if(clink) then
     if(qlink) then
        call wrndie(-1,'<MNDINI>','GLNK and CLNK are not compatable. Ignore CLNK')
        clink=.false.
     else
        if(prnlev.ge.2) write(outu,22) 'CLNK: Connection atoms are used'
        call selcta(COMLYN,COMLEN,jslct,x,y,z,wmain,.true.)
     end if
  end if

  !***********************************************************************
  ! Determine QM method/QM charge/Spin State/SCF convergence
  !
  ! Here's default values
  ! QMTHEORY: MNDO (1); AM1 (2); PM3 (3); AM1/d (4); MNDO/d(5)
  !                                       AMDD : AM1/d
  !                                       MNDD : MNDO/d
  !
  nqmtheory=2   ! default QM model: AM1
  nqmcharge=0   ! default charge 0
  scfconv=TENM8 ! default scf convergence
  nspin=0       ! default spin state: singlet. (see explanation of imult, in qm1_info.f
  ! QM method
  if(index(COMLYN,'MNDO') .ne. 0) nqmtheory=1
  if(index(COMLYN,'AM1')  .ne. 0) nqmtheory=2
  if(index(COMLYN,'PM3')  .ne. 0) nqmtheory=3
  if(index(COMLYN,'AMDD') .ne. 0) nqmtheory=4  ! experimental method.
  if(index(COMLYN,'MNDD') .ne. 0) nqmtheory=5

  ! for AM1/d-PhoT parameters.
  QSRP_PhoT=.false.              ! 
  if(nqmtheory.eq.2 .or. nqmtheory.eq.4) then
     QSRP_PhoT=index(COMLYN,'PHOT').ne.0  
     if(QSRP_PhoT .and. prnlev .ge. 2) then
        write(outu,22) 'AM1/d-PhoT: Specific reactions parameters will be used for H,O, and P atoms.'
     end if
  end if
  !
  ! QM charge
  nqmcharge=gtrmi(COMLYN,COMLEN,'CHAR',0)
  qmcharge =real(nqmcharge)
  !
  ! QM scf convergence
  scfconv  =gtrmf(COMLYN,COMLEN,'SCFC',TENM8)
  !
  ! QM spin
  if((index(COMLYN,'TRIPLET').ne.0).or.(index(COMLYN,'TRIP').ne.0)) nspin=3
  ! Doublet spin state not work with Restricted QM methods
  if((index(COMLYN,'DOUBLET').ne.0).or.(index(COMLYN,'DOUB').ne.0)) nspin=2
  !
  if(nspin.gt.0) then
     call wrndie(-5,'<MNDINI>','Other than singlet is yet supported.')
     nspin=0       ! default spin state: singlet.
  end if
  !**********************************************************************

  ! other options
  qgmrem=(indxa(COMLYN,COMLEN,'REMO').gt.0)
  if(prnlev .ge. 2) then
     if(qgmrem) then
        write(outu,22) 'REMOve: Classical energies within QM atoms are removed.'
     else
        write(outu,22) 'No REMOve: Classical energies within QM atoms are retained.'
     end if
  end if

  qgmexg=(indxa(COMLYN,COMLEN,'EXGR').gt.0)
  if(prnlev .ge. 2) then
     if(qgmexg) then
        write(outu,22) 'EXGRoup: QM/MM Electrostatics for link host groups removed.'
     else
        write(outu,22) 'No EXGRoup: QM/MM Elec. for link atom host only is removed.'
     end if
  end if
22 format('MNDINT> ',A)

  ! for DIIS converger.
  QMMM_NoDiis =(indxa(COMLYN,COMLEN,'NDIS').gt.0)
  if(prnlev.ge.2.and.QMMM_NoDiis) write(outu,22) 'No Diis: DIIS converger will be turned off.'

  ! for memory inlining.
  QNoMemIncore=(indxa(COMLYN,COMLEN,'NOIN').gt.0)
  if(prnlev.ge.2) then
     if(QNoMemIncore) then
        write(outu,22) 'No Memory Incore: All rij will not be saved on memory.'
     else
        write(outu,22) 'Memory Incore: All rij will be saved on memory.'
     end if
  end if   

  ! D3BJ & H4 corrections
  dispers=(INDXA(COMLYN,COMLEN,'D3BJ').GT.0)  ! (INDXA(COMLYN,COMLEN,'DISP').GT.0)
  if(dispers) then
!!!     l_disp=(INDXA(COMLYN,COMLEN,'DISE').GT.0) ! dispersion energy
!!!     if(l_disp) then
!!!        if (prnlev.ge.2) write(outu,22) "DISP: Dispersion among QM atoms included"
!!!        iunit_disp=GTRMI(COMLYN,COMLEN,'UDIS',54)  ! unit for dispersion.inp file read
!!!     end if
     ! Grimmes DFT-D3 two body dispersion
     lmndod2=(INDXA(COMLYN,COMLEN,'TWOBOD') > 0)

     ! Grimmes DFT-D3 three body dispersion
     lmndod3=(INDXA(COMLYN,COMLEN,'THREEBOD') > 0)
     if(.not. lmndod3) lmndod2 =.true.  ! default

     ! User supplied parameters?
     q_d3bj_read=(INDXA(COMLYN,COMLEN,'D3PARAM') > 0)

     if(prnlev.ge.2) then
        if (lmndod3) then
           WRITE(OUTU,22) "D3BJ: Using THREE-body D3(BJ)+E_abc dispersion correction."
        else if (LMNDOD2) then
           WRITE(OUTU,22) "D3BJ: Using TWO-body D3(BJ) dispersion correction."
        end if
        if(q_d3bj_read) write(OUTU,22) 'D3BJ: User supplied parameters are used.'
     end if
  end if
  !

  ! Activation of LQMEWD when LEWALD and QGMREM..only this case
  !
  ! Possible option.
  ! EWMODE     : 1 Ewald QM/MM-SCF.
  !                The MM within cutoff do interact with QM atoms as regular
  !                QM/MM interaction, and apply Ewald correction potential
  !                into diagonal in FOCK matrix.
  !            : 2 (not used). All qm/mm interactions are represented by interaction with
  !                            digonal elements in the Fock matrix, i.e., the qm/mm interactions
  !                            are by muliken charge - mm charge Coulombic (1/r) interactions.
  !            : 3 (only with qm/mm-ewald or -pme). The off-diagonal elements of the Fock matrix
  !                 interact with mm charges by conventional way (i.e., multipole - monopole interaction),
  !                 while the diagonal elements by 1/r manner (i.e., the muliken - mm charge 1/r 
  !                 interaction.
  ! NQMEWD     : 0 Use Mulliken charges on QM atoms to represent charges on
  !                image atoms.
  if(LEWALD.and.(.noT.qgmrem)) call wrndie(-1,'<MNDINI>','QM/MM-Ewald is not compatable without REMO.')
  !
  ! Initialization
  LQMEWD=.false.
  CGMM  = zero
  if(LEWALD.and.qgmrem) LQMEWD=.true.  ! turn on 
  !
  if(LQMEWD) then
     if(prnlev.ge.2) write(outu,22) 'Ewald with QM/MM Option has been Activated.'
     NQMEWD = 0
     LEWMODE = .true.
     NOPMEwald=(indxa(COMLYN,COMLEN,'NOPM').gt.0)  ! don't use PMEwald option.
     !EWMODE = 1
     EWMODE = GTRMI(COMLYN,COMLEN,'NEWD',1)
     if(.not.(EWMODE == 1 .or. EWMODE == 3)) EWMODE = 1 ! use default and ignore other options.

     ! sanity check: only use this when QPME==.true.
     if(.not.QPME) NOPMEwald=.true.
     !
     if(prnlev.ge.2) then      ! write information.
        write(outu,22) 'Default Ewald with QM/MM Option uses Mulliken Charges'
        if(EWMODE == 1) then
           write(outu,22) 'MM within cutoff interact with regular way with QM'
           write(outu,22) 'MM from Ewald Sum interact with diagonal elements in QM'
        else if(EWMODE == 3) then
           write(outu,22) 'MM within cutoff interact with regular way with the off-diagonal elements of QM'
           write(outu,22) 'while it interacts with the digonal element by 1/r manner.'
           write(outu,22) 'MM from Ewald Sum interact with diagonal elements in QM'
        end if
        if(NOPMEwald) then
           write(outu,22) 'Regular Ewald summation will be carried out for QM/MM-Ewald.'
        else
           write(outu,22) 'PMEwald option will be used for QM/MM-Ewald (QM/MM-PMEwald).'
        end if
     end if
     !
     ! Now, for Ewald in QM/MM SCF
     ! default values
     kmaxq  = 5
     ksqmaxq= 27  ! (kmaxq-2)**3
     !
     kmaxq  =gtrmi(COMLYN,COMLEN,'KMAX',kmaxq)
     if(kmaxq.gt.2) ksqmaxq= (kmaxq-2)**3   ! otherwise, use dafault value.
     !
     ! if explicitly specified.
     kmaxxq =GTRMI(COMLYN,COMLEN,'KMXX',kmaxq)
     kmaxyq =GTRMI(COMLYN,COMLEN,'KMXY',kmaxq)
     kmaxzq =GTRMI(COMLYN,COMLEN,'KMXZ',kmaxq)
     ksqmaxq=GTRMI(COMLYN,COMLEN,'KSQM',ksqmaxq)

     ! it should not be here, as igmsel is not set.
     !! Check the total charge for QM/MM-Ewald
     !do i=1,natom
     !   if(igmsel(i).eq.5.or.igmsel(i).eq.0) cgmm=cgmm+cg(i)
     !end do
     !if(qlink) cgmm = cgmm + cggho

  else
     ! in the case of no Ewald with QM/MM
     EWMODE = 0
     NOPMEwald=.true.
     LEWMODE = .false.   ! this may not be used.
  end if
  ! 
  ! Cutoff based on group-based cutoff.
  ! 1. In the non-bond list, differently from that of the default (default) group-based list, 
  !    in which any MM group that is within the cutoff distance from any QM group
  !    is included, the list is splitted such that each QM group has different MM list, which
  !    is within the cutoff update distance. (see routine GUQME3_by_group in qmnbnd.src.)
  ! 2. When evaluating qm-mm interactions, for each QM group, any MM group in the list
  !    is exlcuded in the interaction evaluation if that MM group is away more than 
  !    the cutoff distance.
  ! 3. If the following switch is on, the interaction between the dist_on and dist_off 
  !    is switched off at the dist_off distance. (This is the default with q_cut_by_group=.true.
  ! 4. When LQMEWD is true, the interaction energy between the i (QM) and j (MM) atom pair
  !    is 
  ! 
  !    E_ij (r_ij) = Sw(r_ij)*E_qm-mm-model(r_ij) + (1-Sw(r_ij))*E_qm-mm-long-distance-model(r_ij)
  !
  !    where E_qm-mm-model is the regular QM-MM interaction model, and 
  !          E_qm-mm-long-distance-model is the QM-Mulliken-charge and MM charge interaction,
  !          which is used in the QM/MM-Ewald and QM/MM-PME long interaction model.
  !
  q_cut_by_group=(indxa(COMLYN,COMLEN,'BYGR').gt.0)
  if(.not.qimage .and. q_cut_by_group) then
     call wrndie(0,'<MNDINI>','QM-MM BYGRoup option is only available wiht IMAGE setup. Ignore BYGRoup.')
     q_cut_by_group = .false.
  end if
  if(q_cut_by_group .and. prnlev >= 2) write(outu,22) &
    'Group-based cutoff: QM/MM electrostatic interaction is evaluated based on group-pair cutoff.'

  ! cutoff switching
  qmswtch_qmmm=(indxa(COMLYN,COMLEN,'SWIT').gt.0)
  if(qmswtch_qmmm .and. prnlev >= 2) then
     if(lqmewd) then
        write(outu,22) 'SWITch: QM/MM electrostatic Switching function is used as E =S*E_qm/mm + (1-S)*E_qm/mm-pme.'
     else
        write(outu,22) 'SWITch: QM/MM electrostatic Switching function is used as E =S*E_qm/mm.'
     end if
  end if

  ! CPMD for the propagation of density (see below for setup.)
  q_cpmd=(indxa(COMLYN,COMLEN,'CPMD').gt.0)
  q_fast_pme=.false.   ! default is not using.
  if(q_cpmd .and. LQMEWD .and. .not. NOPMEwald) then
     q_fast_pme=(indxa(COMLYN,COMLEN,'FPME').gt.0)
     if(q_fast_pme .and. prnlev.ge.2) then      ! write information.
        write(outu,22) 'When MD, QM-QM interaction is evaluated using PME routine.'
     end if
  end if
  !

  ! DXL-BOMD (AMN Niklasson, JCP (2009) 130:214109 & Guishan Zheng, Harvard Univ. 05/19/2010)
  q_dxl_bomd=(indxa(COMLYN,COMLEN,'DXLB') > 0)
  if(q_dxl_bomd) then
     K_order   = GTRMI(COMLYN,COMLEN,'NORD',0) 
     N_scf_step= GTRMI(COMLYN,COMLEN,'NSTE',200)  ! number of scf cycle per each md step.
     if(N_scf_step <= 0) N_scf_step = 200         ! default number of scf cycle.
     if(K_order < 3 .or. K_order > 9) then
        q_dxl_bomd =.false.
     else if(prnlev >= 2) then
        write(outu,21) 'Extended Lagrangian with dissipation with K sum over:',K_order, &
                       ' with ',N_scf_step,' scf cycle.'
     end if
  end if
21 format('MNDINT> ',A,I3,A,I3,A)

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
51 format('MNDINT> ',A,I1,A,I3,A)

  ! for Loop-up table type approach.
  mm_main_r%q_lookup=(indxa(COMLYN,COMLEN,'LOOK').gt.0)
  mm_main_r%q_lookup_beta =(indxa(COMLYN,COMLEN,'BETA').gt.0)
  mm_main_r%i_lookup_unit = 100
  if(mm_main_r%q_lookup) then
     mm_main_r%q_lookup_setup=(indxa(COMLYN,COMLEN,'SETU').gt.0)
     mm_main_r%q_lookup_read =(indxa(COMLYN,COMLEN,'RESD').gt.0)
     mm_main_r%i_lookup_unit = GTRMI(COMLYN,COMLEN,'IUNT',100)
     if(prnlev.ge.2) then
        if(mm_main_r%q_lookup_beta) write(outu,22) 'QM-QM beta interactions are used Look-up table.'
        write(outu,22) 'QM-MM interactions are evaluated with Look-up table.'
        if(mm_main_r%q_lookup_setup) write(outu,22) 'Look-up tables are set and saved to file.'
        if(mm_main_r%q_lookup_read) write(outu,22) 'Look-up tables are read from the file.'
     end if
     if(mm_main_r%q_lookup_setup) then
        dr_width=gtrmf(COMLYN,COMLEN,'WIDT',zero)
        rval_min=gtrmf(COMLYN,COMLEN,'RMIN',zero)
        rval_max=gtrmf(COMLYN,COMLEN,'RMAX',THOSND)

        if(mm_main_r%q_lookup_beta) then   
           dr_width_beta=gtrmf(COMLYN,COMLEN,'BWID',zero)
           rval_min_beta=gtrmf(COMLYN,COMLEN,'BMIN',zero)
           rval_max_beta=gtrmf(COMLYN,COMLEN,'BMAX',THOSND)
        end if
     end if
  end if

  ! analysis
  q_analysis=(INDXA(COMLYN,COMLEN,'ANAL').gt.0)
  q_bond_order=.false.
  q_m_charge  =.false.
  iiunit      = 6     ! default output unit.
  if(q_analysis) then
     iiunit      = GTRMI(COMLYN,COMLEN,'IPUT',6)  ! output unit
     q_bond_order=(INDXA(COMLYN,COMLEN,'BOND').gt.0)
     q_m_charge  =(INDXA(COMLYN,COMLEN,'MULL').gt.0)
     if(prnlev >= 2) then
        if(q_bond_order) write(outu,22) 'QM Bond order analysis will be performed.'
        if(q_m_charge  ) write(outu,22) 'QM Mulliken charge analysis will be performed.'
     end if
  end if

  ! for OpenMP/MPI controls.
  num_cpus=GTRMI(COMLYN,COMLEN,'NCPU',4)
#if KEY_PARALLEL==1
  if(prnlev >= 2) then
     write(outu,51) 'The OpenMP/MPI switches occur in ',num_cpus,' number of MPIs.'
  end if
#endif

  ! LEPS and SVB correction part
  QLEPS = (INDXA(COMLYN,COMLEN,'LEPS').GT.0)
  if(QLEPS) CALL SETLEPS(COMLYN,COMLEN)

  ! Turn on Logical flag that says that the PSF has been modified.
  ! (Refer code.f90)
  MUSTUP=.TRUE.

  ! Assign stack array for temporary coordinate
  if(allocated(XIM)) call chmdealloc('mndini.src','MNDINI','XIM',size(XIM),crl=XIM)
  if(allocated(YIM)) call chmdealloc('mndini.src','MNDINI','YIM',size(YIM),crl=YIM)
  if(allocated(ZIM)) call chmdealloc('mndini.src','MNDINI','ZIM',size(ZIM),crl=ZIM)
  !
  call chmalloc('mndini.src','MNDINI','XIM',natom,crl=XIM)
  call chmalloc('mndini.src','MNDINI','YIM',natom,crl=YIM)
  call chmalloc('mndini.src','MNDINI','ZIM',natom,crl=ZIM)

  !=====================================================================
  ! start building up qm/mm parts.
  ! 1) fill igmsel array
  igmsel(1:size(igmsel)) = 0  ! initialize
  call COPSEL_mndo97(numat,natgho,islct,jslct,lslct,clink,qlink)

  ! 2) set qm/mm option values and allocate memory arrays for qm and mm atoms.
  call qmmm_init_set(nqmtheory,nqmcharge,nspin,numat,natgho,  &
                     natom,EWMODE,NQMEWD,                     &
                     qmcharge,scfconv,                        &
                     qlink,QMMM_NoDiis,LQMEWD,NOPMEwald,      &
                     q_bond_order,q_m_charge,iiunit,          &
                     q_dxl_bomd,K_order,N_scf_step,           &
                     q_fockmd,iopt_fdiss,imax_fdiss,          &
                     q_cut_by_group,qmswtch_qmmm,q_cpmd,QNoMemIncore)

  ! 3) Get and print QM region info:
  call Get_QM_from_CHM(qlink,clink,jslct,lslct)

  ! 4) initialize parameters, setup qm info, and allocate memories.
  call qmmm_load_parameters_setup_qm_info(QSRP_PhoT)

  ! 5) initialize CPMD info, and allocate memories.
  if(q_cpmd) then
     if(prnlev.ge.2) write(outu,22) 'CPMD: CPMD for the propagation of density is used.'
     call qm_cpmd_init(q_cpmd,COMLYN,COMLEN)
  end if

  ! 6) Get MM atoms ready for the QM/MM calculations
  ! 6-1) set non-bonded list and prepare for QM/MM-interaction list 
  !      to setup QM/MM-non-bonded list
  if (useddt_nbond(bnbnd).and.numat.GT.0) call nbndqm(x,y,z)

  ! 6-2) mm coordinates copied to xim,yim,zim arrays
  natom_check= natom      ! for checking purpose
  call SwapXYZ_image(natom,x,y,z,xim,yim,zim,imattq)
  
  ! 6-3) ready the coordinates for qm/mm interface.
  call ch2mnd(qm_main_r%numat,igmsel,xim,yim,zim,.true.)

  !=====================================================================
  !
  if(LQMEWD) then
     qcheck=.true.
     call qmmm_Ewald_init(natom,numat,erfmod,igmsel,kmaxXq,kmaxYq,kmaxZq,KSQmaxq,kappa, &
                          q_fast_pme,qcheck)
     if(.not.qcheck) call wrndie(-5,'<MNDINI>','The CHARMM will stop at qmmm_Ewald_init.')
     !
     ! Check the total charge for QM/MM-Ewald (PME usage)
     cgmm=zero
     do i=1,natom
        if(igmsel(i).eq.5.or.igmsel(i).eq.0) cgmm=cgmm+cg(i)
     end do
!!#if KEY_GHO==1
     if(qlink) cgmm=cgmm+cggho
!!#endif
  end if

  if(mm_main_r%q_lookup) then
     if(mm_main_r%q_lookup_setup) then
        q_lookup_first =.true.
     else if(mm_main_r%q_lookup_read) then
        q_lookup_first =.false.
     end if
     call qm_lookup_init(dr_width,rval_min,rval_max, &
                         dr_width_beta,rval_min_beta,rval_max_beta, &
                         mm_main_r%i_lookup_unit,q_lookup_first)
  end if

  ! D3BJ & H4 corrections
  if(dispers)then
     do i=1,qm_main_r%numat
        n                      =qm_control_r%qminb(i)
        qm_main_r%qm_coord(1,i)=X(n)
        qm_main_r%qm_coord(2,i)=Y(n)
        qm_main_r%qm_coord(3,i)=Z(n)
        xdisp(1:3,i)           =qm_main_r%qm_coord(1:3,i)*r_autoang
     end do
     !if(prnlev.ge.2) write(outu,22) "Reading Dispersion parameters"

     ! before calling this, determine the total number of different QM atom types.
     ! which is "ntype"  (in sccdftb, this is based on SCCTYP == izp.)
     call find_unique_qm(ntype)
!!!     if(l_disp) call dispersionread(iunit_disp,qm_main_r%numat,xdisp)

     if (lmndod2 .or. lmndod3) then
        ! lcpe =.false. for now.
        call init_dftd3(qm_main_r%nat,qm_main_r%numat,lmndod2,lmndod3, &
                        q_d3bj_read,COMLYN,COMLEN)
     end if
  end if

  ! KN 10/31/2018: Add H4 correction term
  q_h4corr = (INDXA(COMLYN,COMLEN,'H4CO').GT.0)  ! H4COrrection
  if(q_h4corr) then
     if(prnlev.ge.2) write(outu,'(/,"H4 CORR: H4 correction is used for QM atoms.")')
     call h4_correction_setup(COMLYN,COMLEN)
  end if
  !


  ! This initializes MNDO97 data and performs one gradient calculation
  !
  !call mndo97_sub(MNDOAR,LEN,natom,natom,xim,yim,zim)
  !
  !if(mm_main_r%LQMEWD) then
  !   if(mm_main_r%EWMODE.gt.0) then
  !      do i=1,qm_main_r%numat
  !         CGQMMM(iabs(mminb1(i)))= mm_main_r%qm_charges(i)  ! chag(i)
  !      end do
  !   else
  !      do i=1,numat
  !         CGQMMM(iabs(mminb1(i)))= CHARGQM(i)
  !      end do
  !   end do
  !end if

  !
  ! free memory allocations.
  call chmdealloc('mndini.src','MNDINI','islct',natom,intg=islct)    ! for qm   atoms
  call chmdealloc('mndini.src','MNDINI','jslct',natom,intg=jslct)    ! for link atoms
  call chmdealloc('mndini.src','MNDINI','lslct',natom,intg=lslct)    ! for GHO  atoms
  !
  comlen = 0
  !
  return
END SUBROUTINE MNDINI
!
SUBROUTINE Get_QM_from_CHM(QLINK,CLINK,JSLCT,LSLCT)
  !----------------------------------------------------------------------
  !     Find the atoms defined as QM atoms and get them ready for MNDO97.
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
  use qm1_info, only : qm_control_r,qm_main_r,mm_main_r
  use mndgho_module, only : qm_gho_info_r

  implicit none
  !
  !
  logical :: qlink,clink
  integer :: jslct(*),lslct(*)
  !
  !charcater(len=10),allocatable,dimension(:) :: aatom
  !real(chm_real),allocatable,dimension(:)    :: azunc
  real(chm_real) :: azunc
  !
  integer:: i,n,nslct,natmm,natlnk,nato,naca,nacg,nlatq,ii
  character(len=6) :: ele
  logical          :: qprt
  !
  QPRT=.TRUE.

  ! first, fill qminb array, in which GHO atoms are last:
  nlatq  =0
  if(qlink) then
     ! pure QM atoms first
     do i=1,mm_main_r%natom
        if((igmsel(i).eq.1 .or.  igmsel(i).eq.2) .and. lslct(i).eq.0) then
           nlatq       = nlatq+1
           qm_control_r%qminb(nlatq)= i
        end if
     end do
     ! gho atoms last, as it is needed for gho-expansion etc.
     do i=1,mm_main_r%natom
        if(lslct(i).eq.1) then
           nlatq        = nlatq+1
           qm_control_r%qminb(nlatq) = i
        end if
     end do
  else
     do i=1,mm_main_r%natom
        if(igmsel(i).eq.1.or.igmsel(i).eq.2) then
           nlatq       = nlatq+1
           qm_control_r%qminb(nlatq)= i
        end if
     end do
  end if 

  !
  ! zero charges on QM atoms to remove from MM term.
  if(qgmrem) then
     do i=1,mm_main_r%natom
        qm_control_r%cgqmmm(i) = cg(i)
        if(igmsel(i).eq.1.or.igmsel(i).eq.2) cg(i) = zero
     end do
  end if
  
  ! then, assign nuclear charges:
  !call chmalloc('mndini.src','Get_QM_from_CHM','azunc',qm_main_r%numat,crl=azunc)
  do i=1,qm_main_r%numat
     ii= qm_control_r%qminb(i)
     CALL FINDEL(ATCT(iac(ii)),amass(ii),ii,ELE,azunc,QPRT)
     !
     ! assign neclear charges
     if(qlink.and.lslct(ii).eq.1) then
        ! gho atom: 85
        qm_main_r%nat(i)=85
     else if(clink.and.jslct(ii).eq.1) then
        ! connection atom: 86
        qm_main_r%nat(i)=86
     else 
        ! regular qm atoms
        qm_main_r%nat(i)=int(azunc)
     end if
  end do
  !
  natmm=natom-qm_main_r%numat
  !
  ! number of H-link atoms
  natlnk=0
  do i = 1,natom
     if (igmsel(i).eq.2) natlnk=natlnk+1
  end do
  ! number of Adjusted connecion atoms
  naca=0
  if(clink) then
     do i=1,natom
        if(jslct(i).eq.1) naca=naca+1
     end do
  end if
  ! number of GHO atoms
  if(qlink) then
     nacg=qm_gho_info_r%nqmlnk
  else
     nacg=0
  end if

  !
  ! Write out atomic information
  if(prnlev.gt.2) then
     write (outu,'(/,1x,A,/)') ' Get_QM_from_CHM> Some atoms will be treated quantum mechanically.'
     write (outu,'(5(8X,A,I5,/),/)') &
          ' The number of quantum mechanical atoms   = ',qm_main_r%numat, &
          ' The number of Adjusted Connection atoms  = ',NACA, &
          ' The number of GHO atoms                  = ',NACG, &
          ' The number of QM/MM H-link atoms         = ',NATLNK, &
          ' The number of molecular mechanical atoms = ',NATMM
  end if
  !
  ! clean-up memory.
  !call chmdealloc('mndini.src','Get_QM_from_CHM','azunc',qm_main_r%numat,crl=azunc)

  return
END SUBROUTINE Get_QM_from_CHM
!
SUBROUTINE COPSEL_mndo97(numat,NATGHO,ISLCT,JSLCT,LSLCT,CLINK,QGLNK)
  !-----------------------------------------------------------------------
  !     Copies selection vector to common block 
  !     so it may be used by GAMESS/CADPAC/MNDO97 interface
  !     Call this routine only once and retain definition
  !     of QM, MM, and link atoms throughout the calculation.
  !     We call this from GAMINI/CADINI/MNDINI which is called from charmm/charmm.src
  !
  !     IGMSEL(I) = 5  MM atom to be excluded from QM/MM interaction
  !     IGMSEL(I) = 2  QQ H-Link atom
  !     IGMSEL(I) = 1  QM atom
  !     IGMSEL(I) = 0  MM atom
  !
  !     Not yet supported, but it will come soon (How soon?)
  !     IGMSEL(I) = -1 QM atom  (other replica)
  !     IGMSEL(I) = -2 Link atom (other replica)
  !     IGMSEL(I) = -5 MM atom to be excluded from its QM/MM
  !                    interaction (other replica)
  !     IGMSEL(I) = -6 MM atom (other replica)
  !
  !     MM atom in position close to link atom is excluded from interaction
  !     of external charges to QM region. Instead of this atom is already
  !     a link atom so no need for two atoms in one place!
  !
  use chm_kinds
  use exfunc
  use dimens_fcm
  !...  use coord
  use gamess_fcm
  use stream
  use psf
  use number
  ! use mndo97
  use chutil,only:getres,atomid
  
  use qm1_info, only : qm_main_r,mm_main_r
  use mndgho_module, only : qm_gho_info_r

  !
  implicit none
  !
  integer :: numat,natgho
  integer ::ISLCT(*),JSLCT(*),LSLCT(*)
  logical :: CLINK,QGLNK
  !
  !
  integer :: i,j,i1,i2,n,is,iq,nlatq,numgho
  character(len=4) :: SID, RID, REN, AC
  logical :: lnflag
  integer :: ln
  !
  ! fill igmsel array for qm atoms.
  do i=1, natom
     igmsel(i)=islct(i)
     if (ATYPE(i)(1:2) == 'QQ')   igmsel(i)=2
     if (clink.and.jslct(i).eq.1) igmsel(i)=1  ! Connection atom
     if (qglnk.and.lslct(i).eq.1) igmsel(i)=1  ! gho atom
  end do
  !
  ! find number of qm and gho atoms.
  nlatq = 0
  numgho= 0  ! count for gho atoms
  if(qglnk) then
     ! pure QM atoms first
     do i=1, natom
        if((igmsel(i).eq.1 .or.  igmsel(i).eq.2) .and. lslct(i).eq.0) then
           nlatq        = nlatq+1
        end if
     end do
     ! gho atoms last, as it is needed for gho-expansion etc.
     do i=1, natom
        if(lslct(i).eq.1) then
           nlatq        = nlatq+1
           numgho       = numgho+1
        end if
     end do
  else
     do i=1, natom
        if(igmsel(i).eq.1.or.igmsel(i).eq.2) then
           nlatq       = nlatq+1
        end if
     end do
  end if
  !
  if(nlatq.le.0) call wrndie(-1,'<COPSEL_mndo97>','No quantum mechanical atoms selected.')

  numat=nlatq
  natgho=numgho   ! number of gho atoms.

  !
  !     Check if link atom is connected to any of its neighbors. If
  !     yes then that atom will not be included in QM/MM interaction.
  !     This is sometimes necessary to prevent opposite charge collision,
  !     since QM cannot prevent this to happen.
  !
  !
  do i=1,nbond
     i1=ib(i)
     i2=jb(i)
     !
     ! For connection atom approach, link host atom or group should be
     ! removed from QM/MM SCF procedure
     if(igmsel(i1).eq.1.and.igmsel(i2).eq.0) then
        if(.not.(qglnk.and.lslct(i1).eq.1)) then
           if(qgmexg) then
              !                 remove the entire group
              j=getres(i2,igpbs,ngrp)
              do j=igpbs(j)+1,igpbs(j+1)
                 if(igmsel(j).eq.0) igmsel(j)=5
              end do
           else
              !                 remove the link host atom
              if(igmsel(i2).eq.0) igmsel(i2)=5
           end if
        end if
     else if(igmsel(i1).eq.0.and.igmsel(i2).eq.1) then
        if(.not.(qglnk.and.lslct(i2).eq.1)) then 
           if(qgmexg) then
              !                 remove the entire group
              j=getres(i1,igpbs,ngrp)
              do j=igpbs(j)+1,igpbs(j+1)
                 if(igmsel(j).eq.0) igmsel(j)=5
              end do
           else
              !                 remove the link host atom
              if(igmsel(i1).eq.0) igmsel(i1)=5
           end if
        end if
     end if
     !
     ! For the QQ-link hydrogen atom
     if (igmsel(i1).eq.2) then
        !           Don't change QM atoms
        if(qgmexg) then
           !              remove the entire group
           j=getres(i2,igpbs,ngrp)
           do j=igpbs(j)+1,igpbs(j+1)
              if(igmsel(j).eq.0) igmsel(j)=5
           end do
        else
           !              remove the link host atom
           if(igmsel(i2).eq.0) igmsel(i2)=5
        end if
     end if
     if (igmsel(i2).eq.2) then
        if(qgmexg) then
           !              remove the entire group
           j=getres(i1,igpbs,ngrp)
           do j=igpbs(j)+1,igpbs(j+1)
              if(igmsel(j).eq.0) igmsel(j)=5
           end do
        else
           !              remove the link host atom
           if(igmsel(i1).eq.0) igmsel(i1)=5
        end if
     end if
  end do
  !
  if(prnlev.ge.2) then
     write(outu,118)
     write(outu,120) 'Classical atoms excluded from the QM calculation' 
  end if
118 format('------------------------------------------------')
120 format('MNDINT: ',A,':')
122 format(10X,I5,4(1X,A4))
123 format(10X,I5,4(1X,A4),1X,'*')
124 format(10X,'NONE.')
  n=0
  do i=1,natom
     if(igmsel(i).eq.5) then
        call atomid(i,sid,rid,ren,ac)
        if(prnlev.ge.2) write(outu,122) I,SID,RID,REN,AC
        n=n+1
     end if
  end do
  if(prnlev.ge.2) then
     if(n.eq.0) write(outu,124)
     if(clink) then
        write(outu,120) 'Quantum mechanical atoms, (* is Connection atom)'
     else
        write(outu,120) 'Quantum mechanical atoms'
     end if
  end if
  n=0
  if(clink) then
     do i=1,natom
        if(igmsel(i).eq.1) then
           call atomid(i,sid,rid,ren,ac)
           if(jslct(i).eq.1) then
              if(prnlev.ge.2) write(outu,123) I,SID,RID,REN,AC
           else
              if(prnlev.ge.2) write(outu,122) I,SID,RID,REN,AC
           end if
           n=n+1
        end if
     end do
  else
     do i=1,natom
        if(igmsel(i).eq.1) then
           call atomid(i,sid,rid,ren,ac)
           if(prnlev.ge.2) write(outu,122) I,SID,RID,REN,AC
           n=n+1
        end if
     end do
  end if
  if(prnlev.ge.2) then
     if(n.eq.0) write(outu,124)
     write(outu,120) 'Quantum mechanical Hydrogen link atoms'
  end if
  n=0
  do i=1,natom
     if(igmsel(i).eq.2) then
        call atomid(i,sid,rid,ren,ac)
        if(prnlev.ge.2) write(outu,122) I,SID,RID,REN,AC
        n=n+1
     end if
  end do
  if(qglnk) then
     if(prnlev.ge.2) then
        if(n.eq.0) write(outu,124)
        write(outu,120) 'Quantum mechanical GHO atoms'
     end if
     n=0
     do i=1,natom
        if(lslct(i).eq.1) then
           call atomid(i,sid,rid,ren,ac)
           if(prnlev.ge.2) write(outu,122) I,SID,RID,REN,AC
           n=n+1
        end if
     end do
  end if
  if(prnlev.ge.2) then
     if(n.eq.0) write(outu,124)
     write(outu,118)
  end if
  !
  ! the following is moved to the subroutine Get_QM_from_CHM.
  !!
  !!! Zero charges on QM atoms to remove from MM term.
  !!if(qgmrem) then
  !!   do i=1,natom
  !!      ! cgqmmm(i) = cg(i)
  !!      if(igmsel(i).eq.1.or.igmsel(i).eq.2) cg(i) = zero
  !!   end do
  !!end if
  !
  return
END SUBROUTINE COPSEL_mndo97


SUBROUTINE PBCHECK(DXYZI)
  !-----------------------------------------------------------------------
  !
  use chm_kinds
  use dimens_fcm
  use number,only : zero,half,one
  use pbound
  implicit none
  real(chm_real) :: CORR
  real(chm_real) :: DXYZI(3)
  integer:: ij
#if KEY_PBOUND==1
  if(qBoun) then 
     if(qCUBoun.or.qTOBoun) then
        dxyzi(1) = BOXINV * dxyzi(1)
        dxyzi(2) = BOYINV * dxyzi(2)
        dxyzi(3) = BOZINV * dxyzi(3)
        do ij=1,3
           if(dxyzi(ij).gt.half) then
              dxyzi(ij)=dxyzi(ij)-one
           else if(dxyzi(ij).lt.-half) then
              dxyzi(ij)=dxyzi(ij)+one
           end if
        end do
        if(qTOBoun) then
           corr = half*AINT(R75*(ABS(dxyzi(1)) + ABS(dxyzi(3)) + ABS(dxyzi(3))))
           do ij=1,3
              dxyzi(ij) = dxyzi(ij) - SIGN(corr, dxyzi(ij))
           end do
        end if
        dxyzi(1) = XSIZE * dxyzi(1)
        dxyzi(2) = YSIZE * dxyzi(2)
        dxyzi(3) = ZSIZE * dxyzi(3)
     else
        call PBMove(dxyzi(1), dxyzi(2), dxyzi(3))
     end if
  end if
#endif
  return
END SUBROUTINE PBCHECK

SUBROUTINE VZERO(v,n)
  !-----------------------------------------------------------------------
  ! zeroes out a vector of length n
  !
  use chm_kinds
  use number
  implicit none

  integer        :: n,i
  real(chm_real) :: v(n)

  v(1:n)=zero
  return
END SUBROUTINE VZERO

#else /* (mndo97)*/
  SUBROUTINE MNDINI(COMLYN,COMLEN)
     character(len=*) COMLYN
     integer   COMLEN
     call wrndie(-1,'<MNDINI>','MNDO97 code not compiled.')
     return
  END SUBROUTINE MNDINI
#endif /* (mndo97)*/
