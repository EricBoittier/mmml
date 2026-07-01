module actclus_mod

  use dimens_fcm
  use chm_kinds
     implicit none
  character(len=*),private,parameter :: file_name   ="actclus_ltm.src"

  !  This module contains the cluster 
  !  information, the active atom, group, and cluster 
  !  arrays, and heap pointers for the cluster-cluster
  !  pairlists.
  !  It also contains a small number of parameters used
  !  in the BYCC algorithm
  !
  !                          --RJ Petrella 1.6.99 
  !  Variables    Description
  !  ---------     ---------
  ! ---- active atom information: ---------------------------
  !  ACTV        the array of active atoms 
  !  NACTVE      the number of active atoms
  !  ACAFLG      active atom flag array (1 or 0 for all atoms)
  ! ----- cluster structure variables:  ---------------------
  !  CLNUM       the number of clusters
  !  CLPRTN      array containing the number of atoms in each cluster
  !  JCLUS       list of atom numbers corresponding to 
  !              ordered clusters 
  !  CLHIGH      array pointing into JCLUS 
  !              (gives high index associated with each cluster)
  !  CLUSNUM     arrays contains cluster # of each atom
  ! ----- active cluster structure variables (analogous to above): 
  !  NACTC       # of active clusters
  !  ACLPTN      array containing # of active atoms in each cluster
  !  JACLS       list of atom numbers corresponding to 
  !              active ordered clusters 
  !  ACLHI       array pointing into JACLS
  ! ---- active group structure variables (analogous to above):
  ! ACTVG        array of active groups
  ! NACTG        # of active groups
  ! ACGFLG       active group flag array (1 or 0 for all groups)
  !
  ! ---- active bonds, angle, dihedrals, and impropers
  ! ACTBON       active bonds array
  ! NACTBND      number of active bonds
  ! ACTANGL      active angles array
  ! NACTANG      number of active angles
  ! ACTDIHE      active dihedrals
  ! NACTDIH      numbe of active dihedrals
  ! ACTIMPR      active impropers
  ! NACTIMP      number of active impropers
  ! MXACTBND     maximum number of active bonds, angles, dihe, impr
  ! MXBPAT       maximum number of bonds per atom in clustering alg
  !
  ! ---- heap variables for cluster-cluster pairlists--------
  !  HPMXCN      heap pointer for near cluster-cluster pairs 
  !  HPMXCD      heap pointer for distant cluster-cluster pairs
  !  MAXACN      size of allocated space for near pairs
  !  MAXACD      size of allocated space for distant pairs
  !
  ! -----flags---------------------------
  !  QCLSMD     true if the clusters have been made (initially false)
  !  QNBACTON   true if active atom, clus, grp selections have been made
  !  QLBYCC     same as LBYCC in inbnd.fcm (true if list done with BYCC)
  !  QBACTON    true if the active bond, angle, dihe, and impr arrays
  !             have been made
  !  QUREYBR =  passed in energy routines for actbonds
  !             when Urey Bradley term calculated
  ! ---------------------------------------------------------
  ! ----- some parameters for BYCC algorithm: ---------------
  !  MARGL       user-defined limit for cluster margin width
  ! ---------------------------------------------------------
  !  small stencil used in bycc algorithm:
  !  BBX,BBY,BBZ  the displacements on the cubical grid from 
  !              the reference cube (in x, y, and z directions)
  !               that specify a particular cube position.
  !  NATOMST  stored structure variables used in testing for changes in topology
  !  NRESST    or connectivity 
  !  NSEGST 
  !  NBONDST 
  !  NTHETAST 
  !  NPHIST 
  !  NIMPHIST 
  !      
  integer,allocatable,dimension(:) :: HPMXCD,HPMXCN
  integer :: MXACTBND
  !
  real(chm_real),save :: MARGL
  !
  LOGICAL,save ::  QCLSMD,QLBYCC,QNBACTON,QBACTON,QUREYBR
#if KEY_PARALLEL==1
  LOGICAL,save :: QPARAZ  /*true if zmod running in parallel*/
#endif
  !
  integer,parameter :: MXBPAT=6

  integer,allocatable,dimension(:),save :: ACTV,ACAFLG,CLPRTN,CLHIGH,JCLUS, &
       ACLPTN,JACLS,ACLHI,CLUSNUM,ICACTVF,ICACTVR

  integer,save :: BBX(27),BBY(27),BBZ(27), &
       NACTVE,CLNUM, NACTC, MAXACN,MAXACD,nactg
  integer,save,allocatable,dimension(:) :: ACTVG,ACGFLG

  integer,save :: NATOMST,NRESST,NSEGST,NBONDST, &
       NTHETAST,NPHIST,NIMPHIST,NACTBND, NACTANG, NACTDIH, NACTIMP,NICACTVF,NICACTVR
  integer,save,dimension(:),allocatable :: ACTBOND,ACTANGL,ACTDIHE,ACTIMPR

contains
#if KEY_ACTBOND==1
  subroutine actbond_iniall()
    qbacton=.false.
    qureybr=.false.
#if KEY_PARALLEL==1
    QPARAZ=.FALSE.  
#endif
    return
  end subroutine actbond_iniall
#endif

  subroutine actclus_init()
    QCLSMD = .FALSE.
    NACTVE = 0
    NACTC = 0
    NACTG = 0
    QLBYCC = .FALSE.  !if bycc for neighborlist generation
    QNBACTON = .FALSE.
    MAXACD=0
    MAXACN=0
    return
  end subroutine actclus_init

!  subroutine allocate_actclus()
!    use memory
!    character(len=*),parameter :: routine_name="allocate_actclus"
!    mxactbnd = maxa*2
!    return
!    call chmalloc(file_name,routine_name,'ACTBOND',mxactbnd,intg=ACTBOND)
!    call chmalloc(file_name,routine_name,'ACTANGL',mxactbnd,intg=ACTANGL)
!    call chmalloc(file_name,routine_name,'ACTDIHE',mxactbnd,intg=ACTDIHE)
!    call chmalloc(file_name,routine_name,'ACTIMPR',mxactbnd,intg=ACTIMPR)
!  end subroutine allocate_actclus

end module actclus_mod

