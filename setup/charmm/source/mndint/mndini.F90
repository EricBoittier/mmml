#if KEY_MNDO97==1 /*mndo97*/
SUBROUTINE MNDINI_main(COMLYN,COMLEN)
  use chm_kinds
  use string
  use mlay_mndo97, only: MNDINI_MLAYer

  implicit none
  CHARACTER(len=*):: COMLYN
  INTEGER ::  COMLEN
  !
  ! local variables
  !logical :: q_update

  !q_update = (indxa(COMLYN, COMLEN, 'UPDA') > 0)  ! do update?
  if(indxa(COMLYN, COMLEN, 'UPDA') > 0) then
     ! do update of qm model paramerters
     !
     ! usage:
     ! MNDO UPDAte PARAmeter IUNIT [int] ALLReplica
     !                                   REPLica  [int]
     call MNDINI_update(COMLYN,COMLEN)
  else if(indxa(COMLYN, COMLEN, 'MLAY') >0) then
     !
     ! If mlayered qm/mm method is used, the total qm/mm energy
     ! E_qm/mm (total) = E_low-levl qm/mm (PBC) + dE_corr (high-level)
     !
     ! where dE_corr = [ E_high-level qm/mm (cutoff) - E_low-level qm/mm (cutoff) ]
     !
     ! The high-level correction term, i.e., dE_corr, can also be
     ! applied via the multiple time step (MTS) approach.
     !
     !
     ! do setup for many layers (or multiple) qm regions.
     !
     ! usage:
     ! MNDO97 MLAY IREPL [int] [mlay-specific] [atom-sele] LINK [atom-sele] 
     !             QCHEM ... [QChem options]  (see qchem.info)
     !             QPRInt  UPRInt [int]             ! prnting option
     !             PYTHon|DPMM MLPMode [int] ONLY PGPUid [int]
     !
     ! IREPL [int]: the irepl no. [int] for high-level qm region.
     !              for multi-layered qm/mm, the high region must not be
     !              the primary (irepl==1) region, preferably irepl==2.
     !
     ! [atom-sele]: the first atom selection is for the high-level qm region.
     ! LINK [atom-sele]: selection for the H-link atom for the qm/mm boundary 
     !                   (used for high-level region).
     !                   this should be a subset of the first selection.
     !
     ! mlay-specific:
     !!! KHARge [real] CUTOff  RCUT [real] NSTEp [int]  LPLE [int] 
     ! KHARge [real] NSTEp [int] LAMBda [real]
     ! Q-Chem or other ab initio package options
     !
     !!! CUTOff       : cutoff option to use for high-level qm/mm non-bond interactions
     !!! RCUT [real]  : cutoff distance for CUTOptions
     ! NSTEp [int]  : NSTEP for high-level correction (for MTS calculation)
     ! LAMBda [real]: Lambda scaling factor for the mlayered qm/mm energy/gradients. 
     !!! LPLE  [int]  : Options to handle the force on H-link atom, when LINK is used.
     !
     ! MLP QM/MM specific:
     ! note           : ML potentials will be added to the underlying low-level methods.
     !                  This can be combined with high-level ai-qm/mm methods using 
     !                  MTS (mts=0, meaning no high-level calculations done ever!)
     ! PYTHon|DPMM    : use QMHub Python File I/O or DPMM LibTorch based interface
     ! MLPMode [int[  : 0: MLP; 1: delta-MLP methods
     ! ONLY           : use MLP/delta-MLP QM/MM not MTS QMLAY_high.
     !                  So, the compilation & setup must be done the same way to setup
     !                      QMLAY_high, just not performing high-level correction calc.
     ! PGPUid [int]   : use GPU for MLP calc. & GPU id: 0 ~ ; -1 not use GPU (i.e., CPU only)
     !
     call MNDINI_MLAYer(COMLYN,COMLEN)
  else
     ! do setup
     ! It is a general se-QM/MM setup where you have an option to allocate more than 1 
     ! qm region. The first qm region (defined) is the default qm region as in normal
     ! qm/mm simulation, which is called as a primary qm region.
     call MNDINI(COMLYN,COMLEN)
  end if
  !
  return
END SUBROUTINE MNDINI_main

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
  use mndgho, only : QLINK
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
  use qm1_info, only      : num_qm_system,qm_main_c,mndo97_memory_init ! ,qm_control_c
  use qmmm_interface, only: qmmm_init_set,qmmm_load_parameters_setup_qm_info,qmmm_Ewald_init, &
                            array_pointers,find_unique_qm
  use qmmmewald_module, only : qmmm_ewald_memory_init
  ! D3BJ & H4 corrections
  use dftd3_mndo, only: init_dftd3,r_autoang
  use H4_mndo, only : h4_memory_init,h4_correction_setup
  !
  use mndgho_module, only : GHOHYB
  !use qm1_energy_module,only: find_unique_qm
#if KEY_MTS==1    /*mts*/
  use mmbonded_mod,only : Setup_mmbond,Setup_mmbond_coupling
  use tbmts,only : QTBMTS
#endif /*mts*/

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
  logical :: QSRP_PhoT,QNoMemIncore,q_dxl_bomd,q_analysis,q_bond_order,q_m_charge,q_fockmd
  integer :: K_order,N_scf_step,iiunit,imax_fdiss,iopt_fdiss
  real(chm_real) :: cggho,scfconv,qmcharge

  integer,allocatable,dimension(:) :: islct,jslct,lslct
  integer,allocatable,dimension(:) :: islct_2,jslct_2,lslct_2
  ! D3BJ & H4 corrections
  integer :: n, iunit_disp
  !!real(chm_real) :: xdisp(3,nndim)
  logical :: q_d3bj_read

  ! for MTS
  logical :: qmm_bond,qmm_angle,qmm_dihe,qmm_imph,qmmbond_use,qmm_morse,qmm_print, &
             qmm_bond_corr
  logical :: q_mm_write,q_mm_read,q_coupling
  integer :: n_mm_replica,iunit_bond_write
  real(chm_real) :: scale_force

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
  ! initializations
  qmused = .true.      ! safe here since next are the allocations, ...
  call allocate_gamess ! try to reduce from MAXA to NATOM

  ! first read the no. of qm regions to define
  ! something like this
  ! ... NUQMregion [int]
  num_qm_system = gtrmi(COMLYN,COMLEN,'NUQM',1)  ! number of qm regions
  if(num_qm_system>1) then
     if(prnlev >= 2) write(outu,24) 'No. of QM regions to be defined (multi-layered and others):',num_qm_system

     ! first re-allocate and init the qm related memories and others
     ! this will overwrite the inital call of mndo97_memory_init in mndo97_iniall
     ! at the beginning of charmm run.
     ! this routine also sets the current point to the first replica. so the call of array_pointers
     ! is not necessary.
     call mndo97_memory_init(num_qm_system,.true.)

     ! for qm/mm-ewald
     call qmmm_ewald_memory_init(num_qm_system,.true.)

     ! for h4 related type arrays
     call h4_memory_init(num_qm_system,.true.)
  
     !! for the first sytem, set array pointers to point the first qm region.
     !call array_pointers(.true.,1)
  end if

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
  if(indxa(COMLYN,COMLEN,'MNDO') .ne. 0) nqmtheory=1
  if(indxa(COMLYN,COMLEN,'AM1')  .ne. 0) nqmtheory=2
  if(indxa(COMLYN,COMLEN,'PM3')  .ne. 0) nqmtheory=3
  if(indxa(COMLYN,COMLEN,'AMDD') .ne. 0) nqmtheory=4  ! experimental method.
  if(indxa(COMLYN,COMLEN,'MNDD') .ne. 0) nqmtheory=5

  ! for AM1/d-PhoT parameters.
  QSRP_PhoT=.false.              ! 
  if(nqmtheory.eq.2 .or. nqmtheory.eq.4) then
     QSRP_PhoT=indxa(COMLYN,COMLEN,'PHOT').ne.0
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
  if(indxa(COMLYN,COMLEN,'TRIP').ne.0) nspin=3
  ! Doublet spin state not work with Restricted QM methods
  if(indxa(COMLYN,COMLEN,'DOUB').ne.0) nspin=2
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
     if(.not. lmndod3) lmndod2 =.true.  ! default, two-body dispersion

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

  ! KN 10/31/2018: Add H4 correction term
  q_h4corr = (INDXA(COMLYN,COMLEN,'H4CO') > 0)  ! H4COrrection
  if(q_h4corr .and. prnlev >= 2) then
     write(outu,'(/,"H4 CORR: H4 correction is used for QM atoms.")')
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
  if(LEWALD.and.(.not.qgmrem)) call wrndie(-1,'<MNDINI>','QM/MM-Ewald is not compatable without REMO.')
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
  ! 1. In the non-bond list (i.e., the default (default) group-based list), any MM group
  !    that is within the cutoff distance from any QM group is included.
  ! 2. When evaluating qm-mm interactions, for each QM group, any MM group in the list
  !    is evaluated for their interaction with each QM group if that MM group is within  
  !    the cutoff distance.
  ! 3. If the switch option is on, the interaction between the dist_on and dist_off 
  !    is switched off at the dist_off distance. 
  ! 4. When LQMEWD is true, the interaction energy between the i (QM) and j (MM) atom pair
  !    is 
  ! 
  !    E_ij (r_ij) = Sw(r_ij)*E_qm-mm-model(r_ij) + (1-Sw(r_ij))*E_qm-mm-long-distance-model(r_ij)
  !
  !    where E_qm-mm-model is the regular QM-MM interaction model, and 
  !          E_qm-mm-long-distance-model is the QM-Mulliken-charge and MM charge interaction,
  !          which is used in the QM/MM-Ewald and QM/MM-PME long interaction model.

  ! cutoff switching
  qmswtch_qmmm=(indxa(COMLYN,COMLEN,'SWIT').gt.0)
  if(qmswtch_qmmm .and. prnlev >= 2) then
     if(lqmewd) then
        write(outu,22) 'SWITch: QM/MM electrostatic Switching function is used as E =S*E_qm/mm + (1-S)*E_qm/mm-pme.'
     else
        write(outu,22) 'SWITch: QM/MM electrostatic Switching function is used as E =S*E_qm/mm.'
     end if
  end if

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
!  num_cpus=GTRMI(COMLYN,COMLEN,'NCPU',4)
!#if KEY_PARALLEL==1
!  if(prnlev >= 2) then
!     write(outu,51) 'The OpenMP/MPI switches occur in ',num_cpus,' number of MPIs.'
!  end if
!#endif

  ! LEPS and SVB correction part
  QLEPS = (INDXA(COMLYN,COMLEN,'LEPS').GT.0)
  if(QLEPS) CALL SETLEPS(COMLYN,COMLEN)


  ! for MTS-MMBond part for qm/mm (only works with multple time step approaches).
  ! usage
  ! ... MTSMmbond BOND ANGL WRIT OUNI [int] - ! DIHE IMDI; write ounit [int] write bond infos..
  !                         READ              ! read will be always start from OUNI unit by the number of NUQM replicas.
  !                                           ! the following COUPling terms are handled at Setup_mmbond
  !                         COUP CONSt [int] repeat [[int] [int] [real]] ! first, integer = no. of coupling terms to repeat
  !                                                                      ! [int] [int] [real] H_coupling(i,j) = [real]
  !               NUMQ [int] -                ! no. of QM mm region (for EVB) for memory allocation
  !               BONCorrection               ! only to be used for mm-bonded (freq) corrections used in
  !                                             mts ai-qm/mm calc. (note that this only be used for BOND, not angle/dihe/impr.
  qmmbond_use = (INDXA(COMLYN,COMLEN,'MTSM').GT.0)  ! MTSMmbond
  scale_force     = one  ! default, no scale
  qmm_bond        =.false.
  qmm_angle       =.false.
  qmm_dihe        =.false.
  qmm_imph        =.false.
  qmm_morse       =.false.
  q_mm_write      =.false.
  q_mm_read       =.false.
  qmm_bond_corr   =.false.
  iunit_bond_write= 0    ! defaults to write mm bonds info
  n_mm_replica    = 1    ! deaultts for no. qm mm evb replicas
  if(qmmbond_use) then
#if KEY_MTS==1
     ! for now, only valence terms (not including UB term for the angle).
     ! and, only two step MTS (short-time scale for MM-bonded terms and
     ! longer-time scale for the entire energy).
     qmm_bond_corr =(INDXA(COMLYN,COMLEN,'BONC') > 0)
     if(qmm_bond_corr) then
        ! only support bond terms.. perhaps, for now also allow MORSe terms but may not be used.
        qmm_bond      =(INDXA(COMLYN,COMLEN,'BOND') > 0)
     else
        qmm_bond      =(INDXA(COMLYN,COMLEN,'BOND') > 0)
        qmm_angle     =(INDXA(COMLYN,COMLEN,'ANGL') > 0)
        qmm_dihe      =(INDXA(COMLYN,COMLEN,'DIHE') > 0)
        qmm_imph      =(INDXA(COMLYN,COMLEN,'IMDI') > 0) ! improper dihedral (mts)
     end if
     !
     !if(QTBMTS) then
        if((qmm_bond.or.qmm_angle.or.qmm_dihe.or.qmm_imph) .and. prnlev>=2) &
        write(outu,22) 'MTSMbond is used for MTS-SE QM/MM simulations (use with MTS).'
     !else
     !   qmmbond_use =.false.
     !   call wrndie(-2,'<MNDINI>', 'MTS option not setup. It only works with MTS.')
     !end if

     !
     n_mm_replica=gtrmi(COMLYN,COMLEN,'NUMQ',1)    ! no. of qm mm evb replicas (e.g., rs, ps, int)
                                                   ! n_mm_replica>=2, assume EVB type potentials.
     q_mm_write  =(INDXA(COMLYN,COMLEN,'WRIT') > 0)! write mm bonds info
     q_mm_read   =(INDXA(COMLYN,COMLEN,'READ') > 0)! read mm bonds info (should be saved previously 
                                                   ! using WRITE OUNI [int]
     if(.not. q_mm_read .and. n_mm_replica>1) n_mm_replica = 1 ! reset to ignore NUQM input...
     if(q_mm_write) then
        iunit_bond_write=gtrmi(COMLYN,COMLEN,'OUNI',6)  ! write unit
        if(prnlev>=2) write(outu,24) 'MTSMbond: mm bonds and angles information will be written to:',iunit_bond_write
     else if(q_mm_read) then
        ! note that read will start from unit "OUNI" by the no. of NUQM replicas.
        ! for all data files must be available and open (from WRITe OUNI [int] ...)
        iunit_bond_write=gtrmi(COMLYN,COMLEN,'OUNI',5)  ! read unit
        if(prnlev>=2) then
           write(outu,24) 'MTSMbond: mm bonds and angles information will be from:',iunit_bond_write
           if(n_mm_replica>=2) then
              write(outu,24) 'MTSMbond: mm bond/angle energy will be evaluated by EVB approach.'
           end if
        end if
     end if

     ! for Morse-type bond and angle energies
     if(.not. qmm_bond_corr) &
        qmm_morse = (INDXA(COMLYN,COMLEN,'MORS') > 0)
     if(qmm_morse .and. prnlev >= 2) write(outu,22) 'MTSMbond: Some Bonds are treated by the Morse-type MM Energy form.'

     ! printing option
     qmm_print = (INDXA(COMLYN,COMLEN,'PSVB') > 0)
     if(qmm_print .and. prnlev >= 2) write(outu,22) 'MTSMbond: Print SVB energy.'

     ! for MM force constant scaling.
     scale_force=gtrmf(COMLYN,COMLEN,'SCAL',one)  ! scale force constant.
#else
     call wrndie(-2,'<MNDINI>', 'MTSMmbond only works with MTS. So, ignored.')
#endif
  end if
23 format('MNDINT> ',A,I6,A)
24 format('MNDINT> ',A,I6)

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

  ! 1.1) MTS-SE QM/MM (MTS-MMBond) parts.
#if KEY_MTS==1    /*mts*/
  if(qmmbond_use) then
     call Setup_mmbond(COMLYN,COMLEN,qmm_bond,qmm_angle,qmm_dihe,qmm_imph,qmm_morse,qmm_print, &
                       qmm_bond_corr, &
                       n_mm_replica,q_mm_write,q_mm_read,iunit_bond_write,scale_force)
  end if
#endif /*mts*/

  ! 2) set qm/mm option values and allocate memory arrays for qm and mm atoms.
  call qmmm_init_set(nqmtheory,nqmcharge,nspin,numat,natgho,  &
                     natom,EWMODE,NQMEWD,                     &
                     qmcharge,scfconv,                        &
                     qlink,QMMM_NoDiis,LQMEWD,NOPMEwald,      &
                     q_bond_order,q_m_charge,iiunit,          &
                     q_dxl_bomd,K_order,N_scf_step,           &
                     q_fockmd,iopt_fdiss,imax_fdiss,          &
                     qmswtch_qmmm,QNoMemIncore)

  ! 3) Get and print QM region info:
  call Get_QM_from_CHM(qlink,clink,jslct,lslct)

  ! 4) initialize parameters, setup qm info, and allocate memories.
  call qmmm_load_parameters_setup_qm_info(QSRP_PhoT)

  ! 5) Get MM atoms ready for the QM/MM calculations
  ! 5-1) set non-bonded list and prepare for QM/MM-interaction list 
  !      to setup QM/MM-non-bonded list
  if (useddt_nbond(bnbnd).and.numat.GT.0) call nbndqm(x,y,z)

  ! 5-2) mm coordinates copied to xim,yim,zim arrays
  natom_check= natom      ! for checking purpose
  call SwapXYZ_image(natom,x,y,z,xim,yim,zim,imattq)
  
  ! 5-3) ready the coordinates for qm/mm interface.
  call ch2mnd(qm_main_c%numat,igmsel,xim,yim,zim,.true.)

  !=====================================================================
  !
  if(LQMEWD) then
     qcheck=.true.
     call qmmm_Ewald_init(natom,numat,erfmod,igmsel,kmaxXq,kmaxYq,kmaxZq,KSQmaxq,kappa, &
                          qcheck)
     if(.not.qcheck) call wrndie(-5,'<MNDINI>','The CHARMM will stop at qmmm_Ewald_init.')
     !
     ! Check the total charge for QM/MM-Ewald (PME usage)
     cgmm=zero
     do i=1,natom
        if(igmsel(i).eq.5.or.igmsel(i).eq.0) cgmm=cgmm+cg(i)
     end do
     if(qlink) cgmm=cgmm+cggho
  end if

  ! D3BJ & H4 corrections
  if(dispers)then
     !do i=1,qm_main_c%numat
     !   n                      =qm_control_c%qminb(i)
     !   qm_main_c%qm_coord(1,i)=X(n)
     !   qm_main_c%qm_coord(2,i)=Y(n)
     !   qm_main_c%qm_coord(3,i)=Z(n)
     !   xdisp(1:3,i)           =qm_main_c%qm_coord(1:3,i)*r_autoang
     !end do
     !if(prnlev >= 2) write(outu,22) "Reading Dispersion parameters"

     ! before calling this, determine the total number of different QM atom types.
     ! which is "ntype"  (in sccdftb, this is based on SCCTYP == izp.)
     call find_unique_qm(ntype)
!!!     if(l_disp) call dispersionread(iunit_disp,qm_main_c%numat,xdisp)

     if (lmndod2 .or. lmndod3) then
        ! lcpe =.false. for now.
        call init_dftd3(qm_main_c%nat,qm_main_c%numat,lmndod2,lmndod3, &
                        q_d3bj_read,COMLYN,COMLEN)
     end if
  end if

  ! KN 10/31/2018: Add H4 correction term
  if(q_h4corr) call h4_correction_setup(COMLYN,COMLEN)
  !

  ! 1.2) MTS-SE QM/MM (MTS-MMBond) parts. Do the remaining part for H coupling values.
#if KEY_MTS==1    /*mts*/
  if(qmmbond_use .and. n_mm_replica>1) then
     q_coupling = (INDXA(COMLYN,COMLEN,'COUP') > 0)
     if(q_coupling) call Setup_mmbond_coupling(COMLYN,COMLEN,q_coupling)
  end if
#endif /*mts*/

  ! nullify pointers, skip for now, for a single qm system
  if(num_qm_system>1) call array_pointers(.true.,1)

  !
  ! free memory allocations.
  call chmdealloc('mndini.src','MNDINI','XIM',size(XIM),crl=XIM)
  call chmdealloc('mndini.src','MNDINI','YIM',size(YIM),crl=YIM)
  call chmdealloc('mndini.src','MNDINI','ZIM',size(ZIM),crl=ZIM)

  call chmdealloc('mndini.src','MNDINI','islct',natom,intg=islct)    ! for qm   atoms
  call chmdealloc('mndini.src','MNDINI','jslct',natom,intg=jslct)    ! for link atoms
  call chmdealloc('mndini.src','MNDINI','lslct',natom,intg=lslct)    ! for GHO  atoms
  !
  comlen = 0
  !
  return
END SUBROUTINE MNDINI

SUBROUTINE MNDINI_update(COMLYN,COMLEN)
  !-----------------------------------------------------------------------
  !     This is the interface for running modified version of MNDO97 with 
  !     CHARMM for QM/MM calculations
  !
  !     Kwangho Nam, February 2012.
  !
  use chm_kinds
  use dimens_fcm
  use number
  use stream
  use string, only : nexta4,gtrmi,indxa
  use qmmm_interface, only: parameter_update
  use qm1_info,only : num_qm_system
  !

  implicit none
  CHARACTER(len=*):: COMLYN
  INTEGER ::  COMLEN
  !
  character(len=4) :: keyword
  integer :: inunit, irepl
  logical :: q_all_replica

  ! get the keyword
  keyword=nexta4(comlyn,comlen)
  if(keyword(1:4) == 'PARA') then
     ! do update the semi-empirical parameters
     !
     ! usage
     ! MNDO UPDAte PARAmeter IUNIT [int] ALLReplica
     !                                   REPLica  [int]

     ! Read unit for the input parameters. The file must be already open to read the file.
     inunit = gtrmi(COMLYN,COMLEN,'IUNI',0)
     if(inunit <= 0) then
        call wrndie(-1,'<MNDINI_update>','No file to read parameters is specified. Ignored.')
        !return
     else
        ! 
        q_all_replica=(indxa(COMLYN,COMLEN,'ALLR') > 0)

        ! irepl = 0 means, update them for all qm regions (num_qm_system)
        !       > 0 do separately for each qm region defined by irepl
        if(q_all_replica) then
           if(prnlev>=2) write(outu,20) 'UPDA PARAm> Parameters updated for all replicas'
           irepl=0
        else
           irepl=gtrmi(COMLYN,COMLEN,'IREP',0) ! update parameters for each replica
           if(irepl > num_qm_system) then
              call wrndie(-1,'<MNDINI_update>','Wrong IREPlica number. Ignored.')
              return
           else
              if(prnlev>=2) write(outu,20) 'UPDA PARAm> Parameters updated for replica: ',irepl
           end if
        end if
        !
        ! now call to update parameters
        !rewind inunit
        call parameter_update(irepl,inunit)
     end if
  else
     ! leave out for later use.
     call wrndie(1,'<MNDINI_update>','Do nothing?')

     continue
  end if
20 format('MNDINI_update> ',A,I4)
  !
  return
END SUBROUTINE MNDINI_update

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
  use qm1_info, only : qm_control_c,qm_main_c,mm_main_c,qm_gho_info_c

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
     do i=1,mm_main_c%natom
        if((igmsel(i).eq.1 .or.  igmsel(i).eq.2) .and. lslct(i).eq.0) then
           nlatq       = nlatq+1
           qm_control_c%qminb(nlatq)= i
        end if
     end do
     ! gho atoms last, as it is needed for gho-expansion etc.
     do i=1,mm_main_c%natom
        if(lslct(i).eq.1) then
           nlatq        = nlatq+1
           qm_control_c%qminb(nlatq) = i
        end if
     end do
  else
     do i=1,mm_main_c%natom
        if(igmsel(i).eq.1.or.igmsel(i).eq.2) then
           nlatq       = nlatq+1
           qm_control_c%qminb(nlatq)= i
        end if
     end do
  end if 

  !
  ! zero charges on QM atoms to remove from MM term.
  if(qgmrem) then
     do i=1,mm_main_c%natom
        qm_control_c%cgqmmm(i) = cg(i)
        if(igmsel(i).eq.1.or.igmsel(i).eq.2) cg(i) = zero
     end do
  end if
  
  ! then, assign nuclear charges:
  !call chmalloc('mndini.src','Get_QM_from_CHM','azunc',qm_main_c%numat,crl=azunc)
  do i=1,qm_main_c%numat
     ii= qm_control_c%qminb(i)
     CALL FINDEL(ATCT(iac(ii)),amass(ii),ii,ELE,azunc,QPRT)
     !
     ! assign neclear charges
     if(qlink.and.lslct(ii).eq.1) then
        ! gho atom: 85
        qm_main_c%nat(i)=85
     else if(clink.and.jslct(ii).eq.1) then
        ! connection atom: 86
        qm_main_c%nat(i)=86
     else 
        ! regular qm atoms
        qm_main_c%nat(i)=int(azunc)
     end if
  end do
  !
  natmm=natom-qm_main_c%numat
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
     nacg=qm_gho_info_c%nqmlnk
  else
     nacg=0
  end if

  !
  ! finally, make a local copy of igmsel array (to be used in 
  ! mlayered qm/mm & high-level qm/mm calculations.
  qm_control_c%igmsel(1:natom) = igmsel(1:natom)

  !
  ! Write out atomic information
  if(prnlev.gt.2) then
     write (outu,'(/,1x,A,/)') ' Get_QM_from_CHM> Some atoms will be treated quantum mechanically.'
     write (outu,'(5(8X,A,I5,/),/)') &
          ' The number of quantum mechanical atoms   = ',qm_main_c%numat, &
          ' The number of Adjusted Connection atoms  = ',NACA, &
          ' The number of GHO atoms                  = ',NACG, &
          ' The number of QM/MM H-link atoms         = ',NATLNK, &
          ' The number of molecular mechanical atoms = ',NATMM
  end if
  !
  ! clean-up memory.
  !call chmdealloc('mndini.src','Get_QM_from_CHM','azunc',qm_main_c%numat,crl=azunc)

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
  !use mndo97
  use qm1_info,only : qm_bond_c !  qm_control_c
  use chutil,only:getres,atomid
  
  !
  implicit none
  !
  integer :: numat,natgho
  integer :: ISLCT(*),JSLCT(*),LSLCT(*)
  logical :: CLINK,QGLNK
  !
  ! local variables
  integer :: i,j,i1,i2,j1,n,is,iq,nlatq,numgho
  character(len=4) :: SID, RID, REN, AC
  logical :: lnflag
  integer :: ln
  !
  integer :: nbonds_qm_local
  integer,allocatable :: i_mm_bond_local(:,:)
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
        if((igmsel(i)==1 .or.  igmsel(i)==2) .and. lslct(i)==0) then
           nlatq        = nlatq+1
        end if
     end do
     ! gho atoms last, as it is needed for gho-expansion etc.
     do i=1, natom
        if(lslct(i)==1) then
           nlatq        = nlatq+1
           numgho       = numgho+1
        end if
     end do
  else
     do i=1, natom
        if(igmsel(i)==1.or.igmsel(i)==2) then
           nlatq       = nlatq+1
        end if
     end do
  end if
  !
  if(nlatq <= 0) call wrndie(-1,'<COPSEL_mndo97>','No quantum mechanical atoms selected.')

  numat=nlatq
  natgho=numgho   ! number of gho atoms.

  !
  !     Check if link atom is connected to any of its neighbors. If
  !     yes then that atom will not be included in QM/MM interaction.
  !     This is sometimes necessary to prevent opposite charge collision,
  !     since QM cannot prevent this to happen.
  !
  nbonds_qm_local = 0
  allocate(i_mm_bond_local(2,nbond))
  do i=1,nbond
     i1=ib(i)
     i2=jb(i)
     !
     ! For connection atom approach, link host atom or group should be
     ! removed from QM/MM SCF procedure
     if(igmsel(i1)==1 .and. igmsel(i2)==0) then
        if(.not.(qglnk .and. lslct(i1)==1)) then
           if(qgmexg) then
              !                 remove the entire group
              j=getres(i2,igpbs,ngrp)
              do j1=igpbs(j)+1,igpbs(j+1)
                 if(igmsel(j1)==0) igmsel(j1)=5
              end do
           else
              !                 remove the link host atom
              if(igmsel(i2)==0) igmsel(i2)=5
           end if
        end if
     else if(igmsel(i1)==0 .and. igmsel(i2)==1) then
        if(.not.(qglnk .and. lslct(i2)==1)) then 
           if(qgmexg) then
              !                 remove the entire group
              j=getres(i1,igpbs,ngrp)
              do j1=igpbs(j)+1,igpbs(j+1)
                 if(igmsel(j1)==0) igmsel(j1)=5
              end do
           else
              !                 remove the link host atom
              if(igmsel(i1)==0) igmsel(i1)=5
           end if
        end if
     end if
     !
     ! For the QQ-link hydrogen atom
     if (igmsel(i1)==2) then
        !           Don't change QM atoms
        if(qgmexg) then
           !              remove the entire group
           j=getres(i2,igpbs,ngrp)
           do j1=igpbs(j)+1,igpbs(j+1)
              if(igmsel(j1)==0) igmsel(j1)=5
           end do
        else
           !              remove the link host atom
           if(igmsel(i2)==0) igmsel(i2)=5
        end if
     end if
     if (igmsel(i2)==2) then
        if(qgmexg) then
           !              remove the entire group
           j=getres(i1,igpbs,ngrp)
           do j1=igpbs(j)+1,igpbs(j+1)
              if(igmsel(j1)==0) igmsel(j1)=5
           end do
        else
           !              remove the link host atom
           if(igmsel(i1)==0) igmsel(i1)=5
        end if
     end if

     ! to save mm bonds information for the qm region.
     if((igmsel(i1)==1 .or. igmsel(i1)==2 .or. igmsel(i1)==5) .or. &
        (igmsel(i2)==1 .or. igmsel(i2)==2 .or. igmsel(i2)==5)) then
        nbonds_qm_local = nbonds_qm_local + 1
        i_mm_bond_local(1,nbonds_qm_local) = i1
        i_mm_bond_local(2,nbonds_qm_local) = i2
     end if
  end do
  !
  if(nbonds_qm_local>0) then
     qm_bond_c%nbond_qm = nbonds_qm_local
     allocate(qm_bond_c%i_mm_bond(2,nbonds_qm_local))

     !
     do i=1,nbonds_qm_local
        qm_bond_c%i_mm_bond(1,i) = i_mm_bond_local(1,i)
        qm_bond_c%i_mm_bond(2,i) = i_mm_bond_local(2,i)
     end do
     qm_bond_c%qbond_qm =.true.
  end if
  deallocate(i_mm_bond_local)
  !
  if(prnlev>=2) then
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
     if(igmsel(i)==5) then
        call atomid(i,sid,rid,ren,ac)
        if(prnlev>=2) write(outu,122) I,SID,RID,REN,AC
        n=n+1
     end if
  end do
  if(prnlev>=2) then
     if(n==0) write(outu,124)
     if(clink) then
        write(outu,120) 'Quantum mechanical atoms, (* is Connection atom)'
     else
        write(outu,120) 'Quantum mechanical atoms'
     end if
  end if
  n=0
  if(clink) then
     do i=1,natom
        if(igmsel(i)==1) then
           call atomid(i,sid,rid,ren,ac)
           if(jslct(i)==1) then
              if(prnlev>=2) write(outu,123) I,SID,RID,REN,AC
           else
              if(prnlev>=2) write(outu,122) I,SID,RID,REN,AC
           end if
           n=n+1
        end if
     end do
  else
     do i=1,natom
        if(igmsel(i)==1) then
           call atomid(i,sid,rid,ren,ac)
           if(prnlev>=2) write(outu,122) I,SID,RID,REN,AC
           n=n+1
        end if
     end do
  end if
  if(prnlev>=2) then
     if(n==0) write(outu,124)
     write(outu,120) 'Quantum mechanical Hydrogen link atoms'
  end if
  n=0
  do i=1,natom
     if(igmsel(i)==2) then
        call atomid(i,sid,rid,ren,ac)
        if(prnlev>=2) write(outu,122) I,SID,RID,REN,AC
        n=n+1
     end if
  end do
  if(qglnk) then
     if(prnlev>=2) then
        if(n==0) write(outu,124)
        write(outu,120) 'Quantum mechanical GHO atoms'
     end if
     n=0
     do i=1,natom
        if(lslct(i)==1) then
           call atomid(i,sid,rid,ren,ac)
           if(prnlev>=2) write(outu,122) I,SID,RID,REN,AC
           n=n+1
        end if
     end do
  end if
  if(prnlev>=2) then
     if(n==0) write(outu,124)
     write(outu,118)
  end if
  !
  ! the following is moved to the subroutine Get_QM_from_CHM.
  !
  ! finally, make a local copy of igmsel array (to be used in 
  ! mlayered qm/mm & high-level qm/mm calculations.
  !qm_control_c%igmsel(1:natom) = igmsel(1:natom)
  !
  !!
  !!! Zero charges on QM atoms to remove from MM term.
  !!if(qgmrem) then
  !!   do i=1,natom
  !!      ! cgqmmm(i) = cg(i)
  !!      if(igmsel(i)==1.or.igmsel(i)==2) cg(i) = zero
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
