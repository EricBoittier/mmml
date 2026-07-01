module gamess_fcm
  use chm_kinds
  use dimens_fcm
  use quantm,only:qmused_quantum ! not sure if needed here ??
  implicit none
  !
  !     Defines the data necessary for a GAMESS calculation on a system.
  !
  ! NOTE: DO NOT MODIFY, UNLESS THE PARENT CODE IN GAMESS IS ALSO CHANGED!
  !         
  !  
  !     Variable  Index    Purpose 
  ! 
  !     MAXGMS             Maximum number of GAMESS atoms allowed
  !     MXAO               Maximum number of basis functions
  !     IGMSEL             Selection for calls from gamess
  !           = 5  MM atom to be excluded from QM/MM interaction
  !           = 2  Link atom
  !           = 1  QM atom
  !           = 0  MM atom
  !           = -1 QM atom  (other replica)
  !           = -2 Link atom (other replica)
  !           = -5 MM atom to be excluded from its QM/MM
  !                    interaction (other replica)
  !           = -6 MM atom (other replica)
  !
  !     QGMREM             Flag to remove all classical energies
  !                        within QM atoms
  !     QGMEXG             Exclude all atoms of link host atom group
  !                        from QM/MM electrostatic energy term.
  ! for DIV - guanhua_puja_QC_UW1212
  !     QGMDIV             Divide the charge on the MM link host atom
  !                        equally among other atoms in the same group
  !     QBLUCH             Flag controlling use of blurred charges
  !     QRECALL            Flag for determining where to get blur charge
  !     RECALLINT          Specify which recall integer to use
  !     QNOGU              Flag for initial guess control
  !     QINIGM             Flag for initialization of GAMESS code
  !
  !     NGAMES             Number of QM atoms for ab initio program
  !     NQQCHG             Counter for QINP
  !     QQCHG              QINP charges for nuclei
  !     FQQCHG             Charges on the nuclei from CG
  !
  !     KDIESL             Is this a DIESEL run (MRCI program from Peyerimhoff)
  !     EFCORE             Frozen core energy for DIESEL.
  !
  !     NQMLINK           Number of link QM atoms here
  !
  !     QCUTFLAG          Flag if cutoff keyword is present
  !
  !     QMUSED            Flag if the run is using any of the QM methods
  !                       This is needed since some parts of CHARMM use
  !                       data structures from individual QM methods that
  !                       are not properly allocated/initialized and
  !                       we are moving away from the static allocations :-)
  !
  INTEGER,allocatable,dimension(:) :: IGMSEL
  INTEGER,allocatable,dimension(:) :: NDIV
  real(chm_real),allocatable,dimension(:) :: zlhost
  INTEGER(gms_int) :: NGAMES,NQQCHG,RECALLINT


  logical :: qmused = .false.
  logical :: qmused_qchem = .false.
  logical :: qmused_g09 = .false.
  logical :: qmused_turbo = .false.
  logical :: qmused_nwchem = .false.
  logical :: qmused_gamess = .false.
  logical :: qmused_gamessuk = .false.
  logical :: qmused_squantm = .false.
  logical :: qmused_sccdftb = .false.
  logical :: qmused_mndo97 = .false.
  logical :: qmused_cadpac = .false.

  ! can be common ?
  LOGICAL QQMGRP
  INTEGER NQMEXL,MQMGRP
  integer,allocatable,dimension(:) ::LQMGRP
  real(chm_real) RCUTQM
  INTEGER,allocatable,dimension(:) :: IMMLST
  !
#if KEY_NWCHEM
  integer*8 geom_handle,rtdb_handle
#endif
#if KEY_GAMESS==1
  integer,PARAMETER :: MAXGMS = 2000,MXAO=8192
#elif KEY_QCHEM==1 || KEY_QTURBO==1 || KEY_G09==1
  integer,PARAMETER :: MAXGMS = 5000  ! In QCHEM this is not used as the max # of atoms
#else /**/
  integer,PARAMETER :: MAXGMS = 500
#endif 
  !

#if KEY_GAMESS==1 || KEY_QUANTUM==1 || KEY_CADPAC==1 || KEY_SCCDFTB==1 || \
    KEY_QCHEM==1 || KEY_MNDO97==1 || KEY_SQUANTM==1 || KEY_QTURBO==1 || \
    KEY_G09==1 /*qmstuff_fcm*/

#if KEY_GAMESSUK==0
  LOGICAL QGMREM,QINIGM
  LOGICAL(gms_int) QGMEXG,QBLUCH,QNOGU,QFMO,QRECALL,QCUTFLAG
  LOGICAL(gms_int) QGMDIV   ! for DIV - guanhua_puja_QC_UW1212
  !
  ! VERY IMPORTANT:
  ! this common and few more must stay here, since they are the same
  ! in the original gamess code
  ! Only needed for GAMESS(US)
#if KEY_GAMESS==1
  COMMON /GAMESL/QGMREM,QGMEXG,QINIGM,QBLUCH,QNOGU,QFMO,QRECALL, &
       QCUTFLAG,QGMDIV      ! for DIV - guanhua_puja_QC_UW1212
#endif

! for LITE  INTEGER(gms_int) :: NGAMES,NQQCHG,RECALLINT
  real(chm_real) :: QQCHG(MAXGMS),FQQCHG(MAXGMS)
  COMMON /CGAMES/ NGAMES,NQQCHG,QQCHG,FQQCHG,RECALLINT

! for DIV - guanhua_puja_QC_UW1212
!  INTEGER(gms_int) :: NDIV(MAXAIM)
!  real(chm_real) :: zlhost(MAXAIM)
!  common /divch/ ndiv, zlhost
 
#endif 
  !
#if KEY_SQUANTM==1 || KEY_MNDO97==1
  INTEGER KDIESL  ! this is integer*4, below is integer*8 - rename: (FIXME)
  real(chm_real) EFCORE
#endif
#if KEY_QCHEM==1 || KEY_SQUANTM==1
#if KEY_GAMESS==0
  integer KHFDFT,KGUES
#endif
#endif
#if KEY_MNDO97==0 && KEY_SQUANTM==0 /*nonmndo97_stuff*/
  !
  INTEGER(gms_int) KDIESL
  real(chm_real) EFCORE
  COMMON /DIESEL/ EFCORE,KDIESL
  !
  !     QC: Add # of link atoms here for later reference in vibran.
  INTEGER NQMLINK
  !     -----------------------------------------------------------------
  !     QC: ALSO ADD: 
  !     IMAGE RELATED INFO (ALTHOUGHT CURRENTLY NOT USED, TO BE REFINED)
  !     RCUTQM         QM/MM interaction cut-off
  !     LQMPB          If Duplicate QM image - whether keep QM/MM force on
  !                    the MM images
  !     NPRIMM         Number of Primary MM atoms the QM atoms see
  !     LQMSHF         If use QM shift  for QM/MM ele.
  !     LQMSWT         If use QM switch for QM/MM ele.
  !     LQMGRP         If use group based
  !     LOGICAL LQMPB,LQMSHF,LQMSWT,LQMGRP
  !     INTEGER NPRIMM,NTOTMM

  !     QC_UW_031205 obsolete variables
  !     ,NPRIMM,NTOTMM,LQMPB,LQMSHF,
  !    &               LQMSWT,LQMGRP
  !     
  !     ----------------------------------------------------------------
  !     QC: UW_031205 flag if a group is QM
  !-----------------------------------------------------------------------
#if KEY_GAMESS==1 /*gamess_stuff*/
  !--##IFN CADPAC SCCDFTB (gamess_stuff)
  !
  !     UZNUC          Atomic number
  !     CUNIQ          coordinates of QM atoms for GAMESS
  !     NATREL         number of QM atoms for GAMESS
  !     UATOM          GAMESS atom names
  !
  real(chm_real)  UZNUC,CUNIQ
  INTEGER(gms_int) NATREL
  CHARACTER(len=10) UATOM
  !
  COMMON /COORDN/ UZNUC(MAXGMS),CUNIQ(MAXGMS,3),NATREL,UATOM(MAXGMS)
  !
  !
  !-----------------------------------------------------------------------
  !     Blurred charges variables:
  !CC     MAXBLU      Maximum number of blurred charges, 
  !CC                 must not exceed MXATM=500 in GAMESS
  !     MAXBLU      Maximum number of blurred charges.
  !     NBLUCH      Number of blurred charges
  !     IBLUCH      Pointer array of which charges are blurred
  !     EBLUCH      Values of exponents for blurred charges
  !                 calculated from WMAIN: ebluch=(a.u./wmain)**2
  !     CBLUCH      Preexponential factor calculated from CG
  !                 CBLUCH=CG/WMAIN**3/2/sqrt(PI**3)
  !     CGBLCH      Array to store CG values for blurred charges
  !                 so CG can be put to zero.
  !     SGBLCH      Array to store Sigma values for blurred charges
  !                 To simplify formulae.
  !
  INTEGER,PARAMETER :: MAXBLU=360720
  INTEGER(gms_int) :: NBLUCH,IBLUCH(MAXBLU)    ! this could be protected ???
  real(chm_real),dimension(maxblu) :: EBLUCH,CBLUCH
  !
  COMMON /BLUR/EBLUCH,CBLUCH,IBLUCH,NBLUCH
  !
  !     Also needed in the GAMESS:
  real(chm_real) :: CGBLCH(MAXBLU), SGBLCH(MAXBLU)
  COMMON /MBLUR/ CGBLCH, SGBLCH
  !
  !     This COMMON block is the same as in gamess.src
  !
  !     MXCHRM         Maximum nubmer of MM atoms (here this is  SIZE)
  !                    This number has to be specified also in gamess.src
  !     NCHMAT         Number of MM atoms in QM calculation
  !     KCHRMM         Flag to tell GAMESS if it is in CHARMM environment
  !     {X,Y,Z}CHM     Coordinates of perturbing charges from MM to QM
  !     D{X,Y,Z}ELMM   Forces from QM electrons to MM atoms
  !     QCHM           Atomic charges of MM atoms
  !     KGUES          Initial guess flag
  !     KHFDFT         Initial HF/DFT switch flag
  !
  ! the following are used for QCHEM, too
  ! but they need to be different than for QCHEM, because of integer*4 vs integer*8
  ! RENAME them!! (FIXME)
  INTEGER(gms_int) NCHMAT,KCHRMM,KHFDFT,KGUES
  !     This is currently in GAMESS and they must match!
  integer,PARAMETER :: MXCHRM=360720

  real(chm_real) :: &
       XCHM(MXCHRM),YCHM(MXCHRM),ZCHM(MXCHRM), &
       DXELMM(MXCHRM),DYELMM(MXCHRM),DZELMM(MXCHRM), &
       QCHM(MXCHRM)
  common /chmgms/ xchm,ychm,zchm,dxelmm,dyelmm,dzelmm,qchm, &
       nchmat,kchrmm,kgues,khfdft

#endif /* (gamess_stuff)*/
#endif /* (nonmndo97_stuff)*/
#endif /* (qmstuff_fcm)*/
#if KEY_GAMESSUK==1
  !
  !     IGMSEL             Selection for calls from gamess
  !     QGMREM             Flag to remove all classical energies
  !                        within QM atoms
  !     QGMEXG             Exclude all atoms of link host atom group
  !                        from QM/MM electrostatic energy term.
  !     QINIGM             Flag for initialization of GAMESS code
  !     QBLUCH             Flag controlling use of blurred charges
  !
  !     NGAMES             Number of QM atoms for GAMESS
  !     NGAMES             Number of QM atoms for ab initio program
  !     NQQCHG             Counter for QINP
  !     QQCHG              QINP charges for nuclei
  !     FQQCHG             Charges on the nuclei from CG
  !     A2MASS             Copy of AMASS (for lonepair)
  !     NCHMAT             Number of QM atoms
  !     NBLUCH             Number of blurred charges
  !     gmsmap             Mapping of atoms in CHARMM and GAMESS-UK
  !
  integer, allocatable, dimension(:) :: igmsel

  LOGICAL QGMREM,QGMEXG,QINIGM,QBLUCH,QCUTFLAG

  INTEGER NGAMES,NQQCHG,RECALLINT
  real(chm_real), allocatable :: qqchg(:), fqqchg(:), a2mass(:)

  integer :: nchmat, nbluch
  integer, allocatable :: gmsmap(:)
  !
  !     UZNUC is useful for GAMESS-UK too
  !
  integer, parameter :: MAXGMS = 500
  real(chm_real),save ::  UZNUC(MAXGMS),CUNIQ(MAXGMS,3)
  INTEGER,save :: NATREL
  CHARACTER(len=10),save :: UATOM(MAXGMS)
  !
  !      COMMON /COORDN/ UZNUC(MAXGMS),CUNIQ(MAXGMS,3),NATREL,UATOM(MAXGMS)
  !
#endif 
  !cc##ENDIF (qmstuff_fcm)
  !
  !-----------------------------------------------------------------------
  !
  !
  !     IFRAG                 Current fragment index
  !     NFRAG                 Number of fragments in the system
  !     QFRAG                 Was the fragment command issued
  !     QXFRAG                Freagment excluded on this pocessor
  !     OPTCNT  Counts number of optimization steps of the Q-Chem
  !             and SCCDFTB QM/MM calculation
  !
  INTEGER IFRAG,NFRAG,OPTCNT
  LOGICAL QFRAG,QXFRAG
  !
  !     QMMUL    MAXGMS    Mulliken charges from abinitio calculation
  !     QMLOW    MAXGMS    Lowdin charges from  abinitio calculation
  !     QMKOL    MAXGMS    Merz-Kollman charges from abinitio calculation
  !     QMCMUL   MAXA      Mulliken charges from abinitio calculation
  !     QMCLOW   MAXA      Lowdin charges from  abinitio calculation
  !     QMCKOL   MAXA      Merz-Kollman charges from abinitio calculation
  !
  ! dont change - used in GAMESS code:
  real(chm_real) QMMUL(MAXGMS),QMLOW(MAXGMS),QMKOL(MAXGMS)
  real(chm_real),dimension(:),allocatable :: qmckol,qmclow,qmcmul

  !     QMLAY_high         Flag to use high level QM/MM in multi-layered qm/mm.
  !
  LOGICAL QMLAY_high

  !     QMP2        Do MP2 Calculation
  !     QLMP2       Do Local MP2 Calculation
  !     QCCSD       Do CCSD Calculation
  !     QQCIS       Do QCIS Calculation
  !     QRIMP2      Do RIMP2 Calculation
  !     QSOSMP2     Do SOS-MP2 Calculation
  !     QMOSMP2     Do MOS-MP2 Calculation
  !     QSCSMP2     Do SCS-MP2 Calculation
  !     QQCOORD     Pass Coordinates from Q-Chem back to CHARMM
  !     QRESTART    Determine if Q-Chem restarts first step from saved orbitals
  !     QSAVORB     Save orbitals from Q-Chem calculation 
  !     QQCLJ       Use Lennard-Jones parameters in Q-Chem calculation 
  !     ECOOR       Correction energy used in micro-iteration procedure 
  !     QMICRO      Controls Micro-iteration procedure 
  !     QFRSTMICIT  First micro-iteration step? 
  !     QRESET      Reset Q-Chem options 
  !     QQCHARG     Determines if QM Charges will be read in... 
  !     QQCPARA     Tell Q-Chem to use a specific number of procs
  !     QCPARA      How many procs Q-Chem will use 
  !     QREADHESS   Flag to determine if Hessian is read in or computed 
  !     QSAVEHESS   Flag to determine if Hessian is saved 
  !     QREADGRAD   Flag to determine if Gradient is read in or computed
  !     QSAVEGRAD   Flag to determine if Gradient is saved
  !     QINIQC      Q-Chem Initialization flag
  !     FILCHRG     File location of charges.dat file 
  !     QCHEMCNTRL  Q-Chem Control File
  !     LCH         Length of charges.dat filename 
  !     QNRAP       Turns on Newton-Rhaphson optimization with Q-Chem 
  !     QPRNTINP    Determines if $rem information should be echoed
  !     QSAVEINP    Save Q-Chem input files
  !     QSAVEOUT    Save Q-Chem output files
  !     QPCM        Turn on/off QM/MM/PCM
  !     MAPFF       Map atoms for QM/MM/PCM parameters
  !     NREStart    Number of times to restart Q-Chem if a job fails
  !     QWRITeinp   Determines if Q-Chem writes inputs files to disk    
  !     QOPENMP     Run Q-Chem in a multithreaded way (-nt not -np) 
  !     QMIXED      Setup a mixed basis set calculation for Q-Chem
  !     BASIS1      Basis set 1 to use in mixed basis calculation 
  !     BASIS2      Basis set 2 to use in mixed basis calculation 
  !     QQCSCRATCH  Save Q-Chem scratch files in a custom directory
  !     QCSCRATCH   Directory where Q-Chem scratch files are saved 
  !     QCSCRATCHOLD Directory where Q-Chem scratch files were last saved
  !     QQCHEM      Is this calculation using Q-Chem? 
  !     QQEWALD     USE John Herbert's QM/MM EWALD code
  !     QMESS       USE MESS QM/MM energy estimation
  !     NROOts      Number of roots for MESS-H 
  !     QMALpha     alpha / kappa parameter for QM part of QM/MM ewald 
  !     MMALpha     alpha / kappa parameter for MM part of QM/MM ewald 
  !     QQRESD      Send RESDistance calculation to Q-Chem
  !     QQCONS      Use only single distance for constraint in Q-Chem
  !     qcffatmtyp  Atom types actually used (only needed for QM/MM EWALD)
  !     QQNOLKATMS  Disable all link atoms from Q-Chem calculation (this is
  !                 needed for QM/MM MSCALE+VIBRAN        

  LOGICAL QMP2,QLMP2,QCCSD,QCIS,QRIMP2,QSOSMP2,QMOSMP2,QSCSMP2, &
       QQCOORD,QRESTART,QSAVORB,QQCLJ,QMICRO,QFRSTMICIT,QRESET,  &
       QQCHARG,QQCPARA,QREADHESS,QSAVEHESS,QINIQC,QNRAP,QPRNTINP, & 
       QSAVEINP,QSAVEOUT,QPCM,QREADGRAD,QSAVEGRAD,QWRITEINP,QOPENMP, & 
       QMIXED,QQCSCRATCH,QQCHEM,QQEWALD,QMESS,QQRESD,QQCONS,QQNOLKATMS
  real(chm_real) :: ECORR(128),QMALpha,MMALpha
  real(chm_real), allocatable :: xtmpgrad(:), ytmpgrad(:), ztmpgrad(:)
  INTEGER QCPARA,LCH,NREStart,NROOts
  CHARACTER(len=255) FILCHRG,QCHEMCNTRL,QCSCRATCH,QCSCRATCHOLD
  CHARACTER(len=10) BASIS1,BASIS2
  integer :: qcnblk
  integer, allocatable :: qcblkreorder(:), qcblksizes(:), mapff(:)
  integer, allocatable :: qcffatmtyp(:)

  !
#if KEY_REPLICA==1 /*replica_stuff*/
  !
  !     BLFACTOR              Block factor on the QM reagion
  !     IREPQM                First replica to compute on this node
  !     KREPQM                Number of nodes available for one replica
  !     NREPQM                Number of QM replicas
  !     NPGRP                 Number of parallel groups
  !     NREPQMNOD             Number of replicas per node
  !     numnod_save           saved NUMNOD from CHARMM
  !     mynod_save            saved MYNOD from CHARMM
  !
  real(chm_real)  BLFACTOR
  INTEGER IREPQM,KREPQM,NREPQM
  integer NPGRP, NREPQMNOD, &
       numnod_save, mynod_save
#endif /* (replica_stuff)*/

  character(len=*),private,parameter :: file_name = "gamess_ltm.src"
  !
  !----------------------------------------------------------------------
  !
contains
  subroutine gamess_ltm_init()

    qmused = .false.
    qmused_qchem = .false.
    qmused_g09 = .false.
    qmused_turbo = .false.
    qmused_nwchem = .false.
    qmused_gamess = .false.
    qmused_gamessuk = .false.
    qmused_squantm = .false.
    qmused_sccdftb = .false.
    qmused_mndo97 = .false.
    qmused_cadpac = .false.

#if KEY_GAMESS==1
    QGMREM=.FALSE.
#endif

    NGAMES=0
    NQQCHG=0
    IFRAG=0
    RECALLINT=-1

    QMLAY_high=.false.     

    OPTCNT=0
    QMP2=.false.
    QLMP2=.false.
    QCCSD=.false.
    QCIS=.false.
    QRIMP2=.false.
    QSOSMP2=.false.
    QMOSMP2=.false.
    QSCSMP2=.false.
    QCPARA=-1
    QFRSTMICIT=.false.
    QRESET=.false.
    QREADHESS=.false.
    QSAVEHESS=.false.
    QREADGRAD=.false.
    QSAVEGRAD=.false.
    QNRAP=.false.
    QINIQC=.true.
    QSAVEINP=.false.     ! Save Q-Chem input files
    QSAVEOUT=.false.     ! Save Q-Chem output files
    QPCM=.false.         ! Disable QM/MM/PCM by default
    QPRNTINP=.true.      ! Determine if Q-Chem information get printed 
    QWRITeinp=.false.    ! Does Q-Chem write / save input files then quit? 
    QOPENMP=.false.      ! Run Q-Chem multi-threaded 
    QMIXED=.false.       ! Is Q-Chem using a mixed basis set? 
    NREStart=0           ! Number of times to restart Q-Chem if a job fails
    QQCSCRATCH=.false.
    QQCHEM=.false.
!   QMALpha=ZERO
!   MMALpha=ZERO

    return
  end subroutine gamess_ltm_init

  subroutine allocate_gamess()
    use memory
    use quantm,only:qmused_quantum
    implicit none
    character(len=*),parameter :: routine_name="allocate_gamess"
    if(qmused_quantum.or.qmused_qchem.or.qmused_g09.or.qmused_turbo.or. &
         qmused_gamess.or.qmused_sccdftb.or.qmused_mndo97.or.qmused_squantm &
         .or.qmused_cadpac.or.qmused_nwchem) then
       if(allocated(igmsel))call chmdealloc(file_name,routine_name,'igmsel',maxaim,intg=igmsel)
       if(allocated(ndiv))call chmdealloc(file_name,routine_name,'ndiv',maxaim,intg=ndiv)
       if(allocated(zlhost))call chmdealloc(file_name,routine_name,'zlhost',maxaim,crl=zlhost)
       if(allocated(lqmgrp))call chmdealloc(file_name,routine_name,'lqmgrp',maxgrp,intg=lqmgrp)
       if(allocated(immlst))call chmdealloc(file_name,routine_name,'immlst',maxa,intg=immlst)
       call chmalloc(file_name,routine_name,'igmsel',maxaim,intg=igmsel)
       call chmalloc(file_name,routine_name,'ndiv',maxaim,intg=ndiv)
       call chmalloc(file_name,routine_name,'zlhost',maxaim,crl=zlhost)
       call chmalloc(file_name,routine_name,'lqmgrp',maxgrp,intg=lqmgrp)
       call chmalloc(file_name,routine_name,'immlst',maxa,intg=immlst)
    endif
#if KEY_GAMESSUK==1
    call chmalloc(file_name,routine_name,'igmsel',MAXA,intg=igmsel)
    call chmalloc(file_name,routine_name,'qqchg',MAXA,crl=qqchg)
    call chmalloc(file_name,routine_name,'fqqchg',MAXA,crl=fqqchg)
    call chmalloc(file_name,routine_name,'a2mass',MAXA,crl=a2mass)
    call chmalloc(file_name,routine_name,'gmsmap',MAXA,intg=gmsmap)
#endif 
    if(qmused_quantum.or.qmused_qchem.or.qmused_g09.or.qmused_turbo.or. &
         qmused_gamess.or.qmused_sccdftb.or.qmused_mndo97.or.qmused_squantm &
         .or.qmused_gamessuk.or.qmused_nwchem) then
       if(allocated(qmckol))call chmdealloc(file_name,routine_name,'qmckol',maxa,crl=qmckol)
       if(allocated(qmclow))call chmdealloc(file_name,routine_name,'qmclow',maxa,crl=qmclow)
       if(allocated(qmcmul))call chmdealloc(file_name,routine_name,'qmcmul',maxa,crl=qmcmul)
       call chmalloc(file_name,routine_name,'qmckol',maxa,crl=qmckol)
       call chmalloc(file_name,routine_name,'qmclow',maxa,crl=qmclow)
       call chmalloc(file_name,routine_name,'qmcmul',maxa,crl=qmcmul)
    endif
    if(qmused_qchem.or.qmused_turbo.or.qmused_g09) then
       if(allocated(xtmpgrad))call chmdealloc(file_name,routine_name,'xtmpgrad',MAXA,crl=xtmpgrad)
       if(allocated(ytmpgrad))call chmdealloc(file_name,routine_name,'ytmpgrad',MAXA,crl=ytmpgrad)
       if(allocated(ztmpgrad))call chmdealloc(file_name,routine_name,'ztmpgrad',MAXA,crl=ztmpgrad)
       if(allocated(qcblkreorder))call chmdealloc(file_name,routine_name,'qcblkreorder',MAXA,intg=qcblkreorder)
       if(allocated(qcblksizes))call chmdealloc(file_name,routine_name,'qcblksizes',MAXA,intg=qcblksizes)
       if(allocated(mapff))call chmdealloc(file_name,routine_name,'mapff',MAXA,intg=mapff)
       call chmalloc(file_name,routine_name,'xtmpgrad',MAXA,crl=xtmpgrad)
       call chmalloc(file_name,routine_name,'ytmpgrad',MAXA,crl=ytmpgrad)
       call chmalloc(file_name,routine_name,'ztmpgrad',MAXA,crl=ztmpgrad)
       call chmalloc(file_name,routine_name,'qcblkreorder',MAXA,intg=qcblkreorder)
       call chmalloc(file_name,routine_name,'qcblksizes',MAXA,intg=qcblksizes)
       call chmalloc(file_name,routine_name,'mapff',MAXA,intg=mapff)
    endif
    return
  end subroutine allocate_gamess

end module gamess_fcm

