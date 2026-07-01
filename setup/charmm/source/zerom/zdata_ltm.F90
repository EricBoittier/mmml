module zdata_mod
  use chm_kinds
  !  Contains common blocks for zero module
  !
  !                          --RJ Petrella 6.1.06
  !  Variables    Description
  !  ---------     ---------
  !  QZEROMEM       true if initital memory allocated
  !  QZMOD     true if entered the zero module
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
  !  NICELST       count of entries in master dof list
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
  !  QNOZMEMCL    !true if not clearing memory at end command
  !  CONFLLEN     !for parallel runs, length of (combined) conformer file
  !  QZTIME      !true if timing runs (def=false)
  !  
  !
  ! ---- arrays----------------------------------------------
  !  HPMDFL     master dof list array
  !  HPMDFV     master dof value array
  !  MSTDOFW,MSTDFVW    master dof and val, uncompressed
  !  HPLODF     lo dof pointer (into master)
  !  HPHIDF     hi dof pointer
  !  LODOFLW     lo dof pointer, into master, uncompressed
  !  HIDOFLW     hi dof pointer, into master, uncompressed
  !  LOCONFW    same as first NSUBSP entries of LOCONF,HICONF  
  !  HICONFW     (i.e. lo and hi conformer numbers of each ss)
  !  HPLOCN     lo conformer pointer
  !  HPHICN     hi conformer pointer
  !  HPLOIE     lo pointer into ic line entry list
  !  HPHIIE     hi pointer into ic line entry list
  !  HPLNLS     list of ic line numbers 
  !  HPCLST     list of ic column numbers
  !  HPICTY     list of ic entry types
  !  HPSATB     substructure atom base array
  !  HPSATE     substructure atom base array
  !  HPSALS     substructure atom list
  !  HPSIAB     subs atom init base array 
  !  HPSIAE     subs atom init base array
  !  HPINIL     subs atom init list 
  !  HPLDSS     loaded subspace list
  !  HPALIA     subspace aliases
  !  HPALRF     aliased subspaces
  !  HPADDF     distinct dofs encountered in read
  !  HPFLCI     flagging active atoms
  !  HPFLIN     initialization of atom positions (forward)
  !  HPFLINR     initialization of atom positions (reverse)
  !  HPDOIN     initialization of atom positions (forward)
  !  HPDOINR     initialization of atom positions (reverse)
  !  HPINVL     initial values of dofs
  !  HPFSTC     array of first-conformers
  !  HPZBNS     energy binning array
  !  HPLCSS     base array of grid-indexed subspace lists
  !  HPHCSS      
  !  HPCBSS     grid-indexed subspace lists
  !  HPZNCB     relative positions of grids
  !  HPGMNV     global minimum dof values of 1 search
  !  HPLDSMN    minimum dof value of all loaded subspaces
  !  HPMINSS    array of minor subspaces
  !  HPIMISS    array of flags designating minor subspaces
  ! for distances
  !  HPDISAB1, HPDISAE1 distance constraint base arrays
  !  HPDISAB2, HPDISAE2  same, for second set of constraint arrays
  !  HPDIATL    distance constraint atom list
  !  HPLDDIS    loaded distance constraint array
  !  HPLDDFC    loaded ic constraints array
  !  HPDISGAR   lower limits of distance constraint bounds
  !  HPDISLAR   upper limits of distance constraint bounds
  !  HPQICBF    forward ic build flags (flags are 1 or 0) 
  !  HPDFCLAR,HPDFCGAR for lower and upper limits of dof cons bounds 
  !  HPDFCMP    map of degree of freedom constraints
  !  ZENERGY      energies of conformers read in
  !  LNOREV     if true, no reverse phase in ic build. default is true
  !  QZDATCOMRA  if true, communication of results data is asynchronous
  !  ZRECVND   ! receiving node for async results data
  !  QZRECVND   !true if I am receiving node for async results data
  !  QZSENDND   !true if I am sending node of async results
  !  CHUNKSZ   !number of conformers communicated in a message
  !  ALLPNTS8  !number of points in a search (all grids)
  ! ---------------------------------------------------------
  ! ---------------------------------------------------------
  !      

  integer(chm_int8),save :: ALLPNTS8
  integer,allocatable,dimension(:) :: HPSSNM_hv,HPMDFL_hv
  real(chm_real),allocatable,dimension(:) :: HPMDFV_hv
! for uncompressed conformer arrays
!  integer,allocatable,dimension(:),save,target :: MSTDOFW
!  real(chm_real),allocatable,dimension(:),save,target :: MSTDFVW
!  integer,allocatable,dimension(:),save,target :: LODOFLW,HIDOFLW,LOCONFW,HICONFW
!
  integer,allocatable,dimension(:),save :: HPLODF_hv,HPHIDF_hv,HPLOCN_hv,HPHICN_hv,HPLOIE_hv, &
    HPHIIE_hv, HPLNLS_hv,HPCLST_hv,HPSATB_hv,HPSATE_hv,HPSALS_hv,HPSIAB_hv,HPSIAE_hv, &
    HPINIL_hv,HPLDSS_hv,HPALIA_hv,HPALRF_hv,HPADDF_hv,HPFLCI_hv,HPFLIN_hv,HPDOIN_hv, &
    HPFLINR_hv,HPDOINR_hv
  
  real(chm_real),allocatable,dimension(:),save :: HPGMNV_hv,HPLDSMN_hv,HPINVL_hv,HPDFCGAR_hv, &
    HPDFCLAR_hv
  integer,allocatable,dimension(:),save :: HPDFCMP_hv,HPLDDFC_hv,HPFSTC_hv,HPZBNS_hv,HPLCSS_hv, &
  HPHCSS_hv,HPCBSS_hv,HPDISAE1_hv,HPDIATL_hv,HPDISAB1_hv,HPDISAB2_hv,HPDISAE2_hv

!  integer,allocatable,dimension(:) :: HPZNCB_hv
  integer(chm_int8),allocatable,dimension(:),save :: HPZNCB_hv

  real(chm_real),allocatable,dimension(:),save :: HPDISGAR_hv,HPDISLAR_hv
  integer,allocatable,dimension(:),save :: HPIMISS_hv,HPLDDIS_hv,HPQICBF_hv,HPMINSS_hv,NRNDCNF

!  real(chm_real),allocatable,dimension(:),save :: ZENERGY

  integer,save :: DDEFME,NVALME,NCNFME,NSUBME,NSATME,NZBINS
  integer,save :: NDISTME,NDISATME,TDISTAT,NDISTDE
  integer,save :: MXCNSZ,NTHDDF,NICELST,NSSSLD,TSUBAT,TINITA
  integer,save :: NSUBDE
  integer,save :: NALIAS,NCONFO,NSUBSP,NMASTR
  integer,save :: NALLDF,ZWRUNI,ZSTEPN,ZNCOMB
  integer,save :: NSSTAG,ZMINUNI,NMINORSS,ZTMINO,NDISTLD,ZWRTUNI
  integer,save :: ZTNSTEP,Z1MNSTP,Z1STTYP,Z1STPRN,NDOFCON
  integer,save :: NDOFCME,NDOFCLD,NMASTRW
#if KEY_PARALLEL==1
  integer,save :: CONFLLEN  
  integer,save :: CHUNKSZ
#endif

  ! pointer integers
  integer,save :: HPMDFL,HPMDFV,HPLODF,HPHIDF,HPLOCN
  integer,save :: HPHICN,HPLOIE,HPHIIE,HPLNLS,HPCLST,HPICTY
  integer,save :: HPSATB,HPSATE,HPSALS
  integer,save :: HPSIAB,HPSIAE,HPINIL,HPLDSS
  integer,save :: HPALIA,HPALRF,HPADDF,HPFLCI
  integer,save :: HPFLIN,HPFLINR,HPDOIN,HPDOINR,HPINVL,HPFSTC,HPZBNS
  integer,save :: HPZNCB,HPGMNV,HPLDSMN, &
       HPMINSS,HPIMISS,HPDISAB1,HPDISAE1,HPDISAB2,HPDISAE2, &
       HPDIATL,HPLDDIS,HPDISGAR,HPDISLAR,HPQICBF
  integer,save :: HPDFCLAR,HPDFCGAR,HPDFCMP,HPLDDFC 

  ! logicals
  logical,save :: QZEROMEM,QZMOD,QDOFDE,QSUBDE,QZNCUT,QRECUT,QZBINS 
  logical,save :: QMISSG,QREDUN,QWZCMP,QZMINI,QZVCUT,QSSDIS
  logical,save :: QICFILL,QZSWRIT,QZCFIX,QMINPRNT,QREDUSS,QZORIEN, &
       QDISTDE,QZWRTRJ,QZENERG,QICBREV,QZ1STMIN,QSUBNUL, &
       QDOFCONS,QICCNA
  logical,save :: QZRMSD=.false.
  logical,save :: QNOZMEMCL=.false.
  logical,save :: QZDATCH=.false.  !true when data has been changed 
  logical,save :: LNOREV=.true.,QZTIME=.false.
  ! reals
  real(chm_real),save :: ZENCUT,RECUTF,ZBINSZ,ZSVFRC,ZVATOL,LDSSMNE,Z1MSTP

#if KEY_PARALLEL==1
  integer,save :: OUTLNCNT,MAXLNCNT 
  integer,dimension(:),allocatable,save :: NSSTAG_AR,NEWCONF_AR,MSTDOF_AR
  integer,dimension(:),allocatable,save :: NSSTAG_GLB,NEWCONF_GLB,MSTDOF_GLB
  real(chm_real),dimension(:),allocatable,save :: CURVAL_AR, MYPOTEN_AR, RMST1_AR
  real(chm_real),dimension(:),allocatable,save :: CURVAL_GLB, MYPOTEN_GLB, RMST1_GLB
  logical,save :: QZDATCOMRA = .false.,QZRECVND=.false.,QZSENDND=.false.
  integer(chm_int4) :: ZRECVND
#endif
! random conformer selection
  integer :: NRNDCNT !number of subspaces to randomize
  integer,allocatable,dimension(:) :: NRANDAR
  logical :: QALLSSR=.false. !true if randomizing same # conf in all ss
  logical :: QSEED=.false.  !true if seed set
  logical :: QZRANDM = .false. !true if random conformer selection
  integer :: NALLRAND !if all ss randomized, how many conformers in each
  integer,save :: INSEED !initial seed for randomization
  logical :: QINSEED=.true. !true if seed specified at least once
!  integer,allocatable,dimension(:),target :: HICONFR,LOCONFR,LODOFLR,HIDOFLR,MSTDOFR
!  real(chm_real),allocatable,dimension(:),target :: MSTDFVR,ZENERGYR
  integer,save :: LTMCONF  !temporary

#if KEY_ZEROM==1
contains

  subroutine zerom_iniall()
    implicit none
    ! zero module variables
    QSUBDE=.false.
    QDOFDE=.false.
    QZEROMEM=.false.
    QZMOD=.false.
    QDISTDE=.false.
    QICFILL=.false.
    NICELST=0
    NTHDDF=0
    DDEFME=0
    NVALME=0
    NCNFME=0
    NSUBME=0
    NSATME=0
    NSSSLD=0
    TSUBAT=0
    TINITA=0
    NSUBDE=0
    NCONFO=0
    NSUBSP=0
    NMASTR=0
    ZSTEPN=0
    NDISTDE=0
    return
  end subroutine zerom_iniall
#endif

end module zdata_mod
