module distri
  use chm_kinds
  use dimens_fcm
#if KEY_RISM==1
  use rism
  implicit none
  !------------------------------------------------------------------
  !                Distribution function common block
  !------------------------------------------------------------------
  !
  !     PVV, PUV, PUU :  pointers for the inter- and intra-molecular
  !     functions.
  !     The functions are stored in (DVECT X DPAIR) matricies
  !     with the left element labeling the point in r- or k-space
  !     and the right element specifying the site-site pair. Along
  !     any particular row each function is stored as follows:
  ! -----------------------------------------------------------------
  !   row   : ( V-V   U1-V   U1-U1   U2-V   U2-U1   U2-U2 )
  !
  !  pointer  : PVV  PUV(1)  PUU(1) PUV(2)  PUU(2)  PUU(3)
  ! -----------------------------------------------------------------
  !
  !     NSITV       number of solvent atomic sites
  !     NSITU(DU)   number of solute atomic sites; DU parameterizes
  !                 the solute
  !     NPV       = NSITV*(NSITV+1)/2 : # of intramolecular solvent
  !                 site pairs
  !     NPVV      = NTYPV(NTYPV+1)/2  : # of distinct  intermolecular
  !                 solvent site pairs
  !     NPUV(DU)  = NTYPU(DU)*NTYPV   : # of distinct  solute-solvent
  !                 site pairs
  !     NPU(DU)   = NSITU(DU)*(NSITU(DU)+1)/2  : # of intramolecular
  !                 solute site pairs
  !     NPUU(DUU) = # of distinct  solute-solute site pairs
  !
  !     INTV(DSITV,DSITV)    Index for the intramolecular solvent site
  !                          pairs
  !     INTVV(DSITV,DSITV)   Index for the intermolecular distinct 
  !                          solvent site pairs
  !     INTU(DSITU,DSITU,DU) Index for the intramolecular solute  site
  !                          pairs
  !     INTUV(DSITU,DSITV,DU) Index for the solvent-solute distinct 
  !                          site pairs
  !                          pairs
  !     INTUU(DSITU,DSITU,DUU) Index for the solute-solute distinct 
  !                          site pairs
  !     ITYPE(DSITE)         Array that stores the type of the atomic 
  !                          site; Sites with the same vdw parameters
  !                          and coulomb charges correspond to the same
  !                          ITYPE
  !
  real(chm_real),allocatable,dimension(:,:) :: ipusr
  real(chm_real),allocatable,dimension(:,:) :: ipphik
  real(chm_real),allocatable,dimension(:,:) :: ipgr
  real(chm_real),allocatable,dimension(:,:) :: ipxvvk
  real(chm_real),allocatable,dimension(:,:) :: ipcsr
  real(chm_real),allocatable,dimension(:,:) :: ipcsk
  real(chm_real),allocatable,dimension(:,:) :: ipdgr
  real(chm_real),allocatable,dimension(:,:) :: ipdcsr
  INTEGER PVV,PUV(DU),PUU(DUU),NSITV,NSITU(DU),NPV,NPVV,NPUV(DU), &
       NPU(DU),NPUU(DUU),INTV(DSITV,DSITV),INTVV(DSITV,DSITV), &
       INTU(DSITU,DSITU,DU),INTUV(DSITU,DSITV,DU), &
       INTUU(DSITU,DSITU,DUU),ITYPE(DSITE)
  !
  COMMON / DISTRB / PVV,PUV,PUU,NSITV,NSITU,NPV,NPVV,NPUV, &
       NPU,NPUU,INTV,INTVV,INTU,INTUV,INTUU,ITYPE
  !
  !  Intermolecular functions
  !     05-Aug-93 We now allocate the following arrays in the HEAP.
  !      USR      Auxiliary array that can store the short range 
  !               potential, the cavity potential, the potential of mean
  !               force or the bridge function, depending on the 
  !               specifications of the iteration cycle
  !      PHIK     Fourier transform of the Coulomb potential (-beta*phik)
  !      GR       radial distribution function
  !      XVVK     Fourier transform of the solvent-solvent susceptibility
  !      CSR      short range direct correlation function
  !      CSK      Fourier transform of the short range direct 
  !               correlation function
  !      DGR      stores the change in the solvent gr due to the solute
  !      DCSR     stores the change in the solvent dcsr due to the solute

  integer nsolv,nsolu,ndu,nduu,nprvv,npruu,npruv

  !  Atom-Atom pair names
  character(len=13) :: prnam(dpair)

#endif 
  !
end module distri

