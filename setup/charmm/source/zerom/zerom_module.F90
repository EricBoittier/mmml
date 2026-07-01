module zerom_ltmmod
  use chm_kinds
  !  Contains common blocks for zero module
  !
  !                          --RJ Petrella 6.1.06
  !  Variables    Description
  !  ---------     ---------
  !  QZEROMEM       true if initital memory allocated
  !  QZEROMOD     true if entered the zero module
  !  QDOFDE       true if read dof definitions
  !  QSUBDE       true if read sub definitions
  !  QDISTDE      true if read in dist constraint definitions
  !  QZNCUT       true if abs energy cutoff
  !  QRECUT       true if "recutting" conformers
  !  QZBINS       true if binning energies
  !  QMISSG       true if allow missing conformers in conf file
  !  QREDUN       true if allow same dof read in for > 1 subspace
  !  QWZCMP       true if compressing comformer output to file
  !  QZMINI       true if structure to assume global minimum
  !  QZVCUT       true if have a variable cutoff
  !  QSSDIS       true if allow disordered subspace numbers in conf file
  !  QICFILL      true if ic table has been filled
  !  QZSWRIT       true if writing search output to file
  !  QZCFIX       true if cons fixing non-moving atoms
  !  QMINPRNT     true if printing out each minimum conformer to a file
  !  QREDUSS      true if not allowing subspaces to appear > 1 time/search
  !  QZRMSD       true if calculating rmsd's 
  !  QZWRTRJ      true if writing .dcd (trajectory) files
  !  QZENERG      true if calculating energies
  !  QZORIEN      true if reorientation being done with rmsd
  !  QICBREV      true if building coordinates from ic table in reverse
  !  QSUBNUL      true if there is one or more null subspaces
  !  QDOFCONS     true if there are dof (ic) constraints
  !  QICCNA       true if at least one dof constraint loaded
  !  DDEFME       memory estimate for dof definitions
  !  NVALME       memory estimate for master dof list
  !  NCNFME       memory estimate for conformers
  !  NSUBME       memory estimate for subspaces
  !  NSATME       memory estimate for substructure atom list
  !  NDISTME      memory estimate for distance constraints
  !  NDISATME     memory estimate for atoms in all distance constraints
  !  NDOFCME      memory estimate for # of dof constraints
  !  NZBINS       number of energy bins
  !  ZBINSZ       energy bin size
  !  MXCNSZ       memory for max size of conformer (# of dof)
  !  NTHDDF       count of dof definitions
  !  ICELST       count of entries in master dof list
  !  NSSSLD       count of loaded subspaces
  !  TSUBAT       current length of substruc atom list
  !  TINITA       curr length of substr atom initialization list
  !  TDISTAT      current length of distance constraint atom list
  !  NSUBDE       count of substructure definitions
  !  NDISTDE      count of distance constraints
  !  NDOFCON      count of internal coordinate (dof) constraints
  !  NALIAS       count of subspace aliases
  !  RECUTF       tolerance above minimum for energy recut
  !
  !  NCONFO       total number of conformers read
  !  NSUBSP       total number of subspaces read
  !  NMASTR       length of master dof list as read 
  !  NALLDF       number of distinct dofs read
  !  NMINORSS     number of "minor" subspaces
  !  ZTMINO       number of minor subspaces taken in a search
  !  
  !  ZENCUT       cutoff energy for storing or writing out conformers
  !  ZVATOL       tolerance for variable energy cutoff
  !  ZWRUNI       unit for writing out conformers in search
  !  ZWRTUNI      unit for writing out dcd files
  !  ZTNSTEP      number of structures to be written to dcd
  !  ZMINUNI      unit for writing out minimum-energy conformers
  !  ZSTEPN       grid-point counter for each zsearch
  !  ZSVFRC       frac of gridpnts in search to be saved (for memory) 
  !  ZNCOMB       number of combinations or grids
  !  NSSTAG       number assigned to product subspace
  !  LDSSMNE      minimum energy for set of loaded subspaces
  !  NDISTLD      number of loaded distance constraints
  !  NDOFCLD      number of loaded ic (dof) constraints
  !  Z1MNSTP      number of steps in 1st-order minimization
  !  Z1STPRN      print interval for conjugant gradient minimization
  !  
  !  Z1STTYP      =0 for SD; = 1 for Cong Grad
  !  QZ1STMIN     true if 1st-order minimization requested
  !  Z1MSTP       step size for 1st order minimization
  !  
  !
  ! ---- heap pointers ----------------------------------
  !  HPMDFL      for master dof list array
  !  HPMDFV      for master dof value array
  !  HPLODF      for lo dof pointer
  !  HPHIDF      for hi dof pointer
  !  HPLOCN      for lo conformer pointer
  !  HPHICN      for hi conformer pointer
  !  HPLOIE      for lo pointer into ic line entry list
  !  HPHIIE      for hi pointer into ic line entry list
  !  HPLNLS      for list of ic line numbers 
  !  HPCLST      for list of ic column numbers
  !  HPICTY      for list of ic entry types
  !  HPSATB      for substructure atom base array
  !  HPSATE      for substructure atom base array
  !  HPSALS      for substructure atom list
  !  HPSIAB      for subs atom init base array 
  !  HPSIAE      for subs atom init base array
  !  HPINIL      for subs atom init list 
  !  HPLDSS      for loaded subspace list
  !  HPALIA      for subspace aliases
  !  HPALRF      for aliased subspaces
  !  HPADDF      for distinct dofs encountered in read
  !  HPFLCI      for flagging active atoms
  !  HPFLIN      for initialization of atom positions (forward)
  !  HPFLINR      for initialization of atom positions (reverse)
  !  HPDOIN      for initialization of atom positions (forward)
  !  HPDOINR      for initialization of atom positions (reverse)
  !  HPINVL      for initial values of dofs
  !  HPFSTC      for array of first-conformers
  !  HPZBNS      for energy binning array
  !  HPLCSS      for base array of grid-indexed subspace lists
  !  HPHCSS      
  !  HPCBSS      for grid-indexed subspace lists
  !  HPZNCB      for relative positions of grids
  !  HPGMNV      for global minimum dof values of 1 search
  !  HPLDSMN     for minimum dof value of all loaded subspaces
  !  HPMINSS     for array of minor subspaces
  !  HPIMISS     for array of flags designating minor subspaces
  ! for distances
  !  HPDISAB1, HPDISAE1 for distance constraint base arrays
  !  HPDISAB2, HPDISAE2  same, for second set of constraint arrays
  !  HPDIATL     for distance constraint atom list
  !  HPLDDIS     for loaded distance constraint array
  !  HPLDDFC     for loaded ic constraints array
  !  HPDISGAR    for lower limits of distance constraint bounds
  !  HPDISLAR    for upper limits of distance constraint bounds
  !  HPQICBF     for forward ic build flags (flags are 1 or 0) 
  !  HPDFCLAR,HPDFCGAR for lower and upper limits of dof cons bounds 
  !  HPDFCMP     for map of degree of freedom constraints
  ! ---------------------------------------------------------
  ! ---------------------------------------------------------
  !      
  integer,allocatable,dimension(:) :: HPSSNM_hv
  integer,allocatable,dimension(:) :: HPMDFL_hv
  real(chm_real),allocatable,dimension(:) :: HPMDFV_hv
  integer,allocatable,dimension(:) :: HPLODF_hv
  integer,allocatable,dimension(:) :: HPHIDF_hv
  integer,allocatable,dimension(:) :: HPLOCN_hv
  integer,allocatable,dimension(:) :: HPHICN_hv
  integer,allocatable,dimension(:) :: HPLOIE_hv
  integer,allocatable,dimension(:) :: HPHIIE_hv
  real(chm_real),allocatable,dimension(:) :: HPGMNV_hv
  real(chm_real),allocatable,dimension(:) :: HPLDSMN_hv
  integer,allocatable,dimension(:) :: HPLNLS_hv
  integer,allocatable,dimension(:) :: HPCLST_hv
  integer,allocatable,dimension(:) :: HPSATB_hv
  integer,allocatable,dimension(:) :: HPSATE_hv
  integer,allocatable,dimension(:) :: HPSALS_hv
  integer,allocatable,dimension(:) :: HPSIAB_hv
  integer,allocatable,dimension(:) :: HPSIAE_hv
  integer,allocatable,dimension(:) :: HPINIL_hv
  integer,allocatable,dimension(:) :: HPLDSS_hv
  integer,allocatable,dimension(:) :: HPALIA_hv
  integer,allocatable,dimension(:) :: HPALRF_hv
  integer,allocatable,dimension(:) :: HPADDF_hv
  integer,allocatable,dimension(:) :: HPFLCI_hv
  integer,allocatable,dimension(:) :: HPFLIN_hv
  integer,allocatable,dimension(:) :: HPDOIN_hv
  integer,allocatable,dimension(:) :: HPFLINR_hv
  integer,allocatable,dimension(:) :: HPDOINR_hv
  real(chm_real),allocatable,dimension(:) :: HPINVL_hv
  real(chm_real),allocatable,dimension(:) :: HPDFCGAR_hv
  real(chm_real),allocatable,dimension(:) :: HPDFCLAR_hv
  integer,allocatable,dimension(:) :: HPDFCMP_hv
  integer,allocatable,dimension(:) :: HPLDDFC_hv
  integer,allocatable,dimension(:) :: HPFSTC_hv
  integer,allocatable,dimension(:) :: HPZBNS_hv
  integer,allocatable,dimension(:) :: HPZNCB_hv
  integer,allocatable,dimension(:) :: HPLCSS_hv
  integer,allocatable,dimension(:) :: HPHCSS_hv
  integer,allocatable,dimension(:) :: HPCBSS_hv

  integer,allocatable,dimension(:) :: HPDISAE1_hv
  integer,allocatable,dimension(:) :: HPDIATL_hv
  integer,allocatable,dimension(:) :: HPDISAB1_hv
  integer,allocatable,dimension(:) :: HPDISAB2_hv
  integer,allocatable,dimension(:) :: HPDISAE2_hv
  real(chm_real),allocatable,dimension(:) :: HPDISGAR_hv
  real(chm_real),allocatable,dimension(:) :: HPDISLAR_hv
  integer,allocatable,dimension(:) :: HPIMISS_hv

  integer,allocatable,dimension(:) :: HPLDDIS_hv
  integer,allocatable,dimension(:) :: HPQICBF_hv
  integer,allocatable,dimension(:) :: HPMINSS_hv


  INTEGER DDEFME,NVALME,NCNFME,NSUBME,NSATME,NZBINS
  INTEGER NDISTME,NDISATME,TDISTAT,NDISTDE
  INTEGER MXCNSZ,NTHDDF,ICELST,NSSSLD,TSUBAT,TINITA
  INTEGER NSUBDE
  INTEGER NALIAS,NCONFO,NSUBSP,NMASTR
  INTEGER NALLDF,ZWRUNI,ZSTEPN,ZNCOMB
  INTEGER NSSTAG,ZMINUNI,NMINORSS,ZTMINO,NDISTLD,ZWRTUNI
  INTEGER ZTNSTEP,Z1MNSTP,Z1STTYP,Z1STPRN,NDOFCON
  INTEGER NDOFCME,NDOFCLD
  ! pointer integers
  INTEGER HPMDFL,HPMDFV,HPLODF,HPHIDF,HPLOCN
  INTEGER HPHICN,HPLOIE,HPHIIE,HPLNLS,HPCLST,HPICTY
  INTEGER HPSATB,HPSATE,HPSALS
  INTEGER HPSIAB,HPSIAE,HPINIL,HPLDSS
  INTEGER HPALIA,HPALRF,HPADDF,HPFLCI
  INTEGER HPFLIN,HPFLINR,HPDOIN,HPDOINR,HPINVL,HPFSTC,HPZBNS
  INTEGER HPLCSS,HPHCSS,HPCBSS,HPZNCB,HPGMNV,HPLDSMN, &
       HPMINSS,HPIMISS,HPDISAB1,HPDISAE1,HPDISAB2,HPDISAE2, &
       HPDIATL,HPLDDIS,HPDISGAR,HPDISLAR,HPQICBF
  INTEGER HPDFCLAR,HPDFCGAR,HPDFCMP,HPLDDFC 
  !      
  !
  ! logicals
  LOGICAL QZEROMEM,QZEROMOD,QDOFDE,QSUBDE,QZNCUT,QRECUT,QZBINS 
  LOGICAL QMISSG,QREDUN,QWZCMP,QZMINI,QZVCUT,QSSDIS
  LOGICAL QICFILL,QZSWRIT,QZCFIX,QMINPRNT,QREDUSS,QZORIEN, &
       QZRMSD,QDISTDE,QZWRTRJ,QZENERG,QICBREV,QZ1STMIN,QSUBNUL, &
       QDOFCONS,QICCNA
  ! reals
  real(chm_real) ZENCUT,RECUTF,ZBINSZ,ZSVFRC,ZVATOL,LDSSMNE,Z1MSTP

#if KEY_ZEROM==1
contains

  subroutine zerom_iniall()
    implicit none
    ! zero module variables
    qsubde=.false.
    qdofde=.false.
    qzeromem=.false.
    qzeromod=.false.
    qdistde=.false.
    qicfill=.false.
    icelst=0
    nthddf=0
    ddefme=0
    nvalme=0
    ncnfme=0
    nsubme=0
    nsatme=0
    nsssld=0
    tsubat=0
    tinita=0
    nsubde=0
    nconfo=0
    nsubsp=0
    nmastr=0
    zstepn=0
    ndistde=0
    return
  end subroutine zerom_iniall
#endif /*  ZEROM*/

end module zerom_ltmmod

