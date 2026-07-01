module  mndo97
  use chm_kinds
  use dimens_fcm
#if KEY_MNDO97==1
!!
!!     Defines the data necessary for a GAMESS calculation on a system.
!!
!! NOTE: DO NOT MODIFY, UNLESS THE PARENT CODE IN GAMESS IS ALSO CHANGED!
!!         
!!  
!!     Variable  Index    Purpose 
!! 
!!     IGMSEL             Selection for calls from cadpac
!!     KCHRMM         Flag to tell MNDO97 if it is in CHARMM environment
!!     MAXCEN         Maximum number of atoms that CADPAC works with
!!     LEN             size of the array MNDOAR for the dynamic memory allocation
!!
!! MNDO97 common blocks
!      real(chm_real) :: SCFCAL, DHCORE       ! external routines   
!      EXTERNAL SCFCAL, DHCORE
!
!! Double to accomendate full DIIS for large systems, 0319PJ05
!      integer, parameter :: LEN=2000000, &     ! memory array size
!                            LM1=600, &         ! maximum QM atoms. 
!                            LM1M=40000*2
!
!! ***LM1M size should be considered to provide IMAGE atoms.
!!    MAXAIM (maximum atoms including image) = 2*SIZE
!!    SIZE   = 60,000 for  LARGE version
!!            240,000 for XLARGE version
!!             25,000 for MEDIUM version
!!    This LM1M should be mirrored in MNDO97 files.
!
!      real(chm_real) :: MNDOAR(LEN)
!
!      integer        :: NABCWT(LM1*3), NNWT(LM1), NATWT, NVAR
!      real(chm_real) :: AAWT(3,LM1), DUMWT(3,LM1)
!      real(chm_real) :: CORD(3,LM1)
!      real(chm_real) :: DENER, CGQM(3,LM1+LM1M)
!      real(chm_real) :: COORDM(3,LM1M), CHARGM(LM1M)
!      INTEGER        :: IN2(200)
!
!! BG Modify mndo for input on any file
!      INTEGER M97IU, KCHRMM
!!=================================================================================
!! namkh 09/10/03
!!     QMPRNT        : Overall control of print
!!     PRNLEVQM      : Control print level
!!                     1 ~ 4 currently
!!     PRNOP         : Output unit for Printing of output information from MNDO97
!!     NELECQM       : Number of valence electrons
!!     CHARMGQM(LM1) : Mulliken Charges of QM atoms
!!     DENSQM(LM1)   : Atomic Electron Density of QM atoms
!!     DIPOLEQM(4)   : Total Dipole Moment of QM part,
!!                     1,2,3...are respectively X,Y,Z axis
!!                     4 is total value
!!     EIPQM         : Ionization Potential 
!!     EVPQM         : First Vertical Energy
!!     EHOMO(2)      : HOMO-1 and HOMO Energy
!!     ELUMO(2)      : LUMO and LUMO+1 Energy
!!     BNDORQM(5050) : only support up to 100 QM atoms currently
!!                     if requires more than 100, then N*(N+1) / 2 will be needed value
!      LOGICAL QMPRNT
!      INTEGER PRNLEVQM, PRNOP
!      INTEGER NELECQM
!      real(chm_real)  CHARGQM(LM1), DENSQM(LM1)
!      real(chm_real)  DIPOLEQM(4), EIPQM, EVPQM, EHOMO(2), ELUMO(2)
!      real(chm_real)  BNDORQM(5050)
!! if you change the number "5050" to bigger number..
!! then change the parameter LMQMN in file "MBONDS.f"
!!=================================================================================
!!
#endif
#if KEY_QCHEM==1 || KEY_GAMESS==1 || KEY_GAMESSUK==1 || KEY_QTURBO==1 || KEY_G09==1
#if KEY_SQUANTM==0 && KEY_MNDO97==0 /*nosquantmstuff*/
!!=================================================================================
!! namkh 02/14/04
!! For the QM/MM electrostatics with Cutoff distance and Ewald Sum method
!!     It is only work with Group based Cutoff option
!!     LQMEWD        : Logical to Ewald or PME with QM/MM
!!     LQMEWD2       : Alias of LQMEWD (bad coding)
!!     NQMPME        : The mode of QM/MM in Ewald or PME
!!                     1: use MM partial charges (it should be specified in psf)
!!                     2: use Mulliken Atomic partial charges from QM
!!     EWMODE        : The mode of Ewald in QM/MM SCF procedure
!!                     0: MM from Ewald Sum doesn't polarize QM atom in SCF
!!                     1: MM from Ewald Sum polarize QM atom in SCF,
!!                        and MM atoms in cutoff polarize QM atom based on 
!!                        M.J.Field et al.
!!                     2: MM from Ewald Sum polarize QM atom in SCF,
!!                        but MM atoms in cutoff also polarize only diagonal elements
!!                        in Fock matrix 
!!     NQMGRP        : The group number of QM group
!!     NMAXGRP       : Maximum number of QM groups. Currently upto 30
!!     NATMTOT       : Total number of atoms in the periodic box
!!     QMINB(LM1)    : The QM atom number in atomic sequence
!!     MMINB(LM1+LM1M) : The atom discription for the mapping of atom number
!!                       1~LM1 : QM atom number in the charmm that will be passed onto
!!                               MNDO97 subroutine
!!                       LM1~Last : '+' with atom number within cutoff distance
!!                                  '-' with atom number that is not included cutoff
!!     MMINB2(LM1+LM1M): MMINB2(I) points the position in the MMINB array, that is pointer.
!!                       MMINB2(I) = J, and MMINB(J)='+ or - I' for MM atom
!!                       If I is QM atom, MMINB2(I) = -J
!!      **MMINB and MMINB2 should change more clever way
!!
!!     IMATTQ(MAXAIM): Mapping index for Images into Primary image and vice versa.
!!                     1      ~NATOM : mapped into images (image transformation number)
!!                   : refer "nbndqm_ltm.src"
!!
!!     EMPOT(LM1)      : Potential from Ewald Sum on QM atom sites
!!     CGQMMM(LM1+LM1M): Charges to be passed to MNDO97 for the MM atoms. These charges 
!!                       will only interact with the QM atoms in the non-bonded interaction. 
!!                       The MM charges in array CG are used for MM-MM interactions.
!! Refer ewald.fcm
!!
#if KEY_QCHEM==1 || KEY_GAMESS==1 || KEY_GAMESSUK==1 || KEY_QTURBO==1 || KEY_G09==1
      INTEGER,parameter:: LM1=6000, LM1M=40000*2
!#endif
!
      INTEGER, parameter :: NMAXGRP=201

      integer :: NQMGRP(NMAXGRP), MMINB(LM1+LM1M)

      real(chm_real), save :: CGQMMM(LM1+LM1M)                           
      integer,        save :: QMINB(LM1),MMINB2(LM1+LM1M),NCUTOFF        

      logical, save :: LQMEWD2                                           

      real(chm_real):: CHAG(LM1), EMPOT(LM1) ,ESLF(LM1)
#if KEY_MNDO97==0
      logical       :: LQMEWD
      integer       :: EWMODE
#endif

      integer       :: NQMPME

      logical, save :: LEWMODE, QSETUPKQ                                 
!
      integer, save :: TOTKQ,MAXKVQ,KMAXQ,KSQMAXQ,KMAXXQ,KMAXYQ,KMAXZQ, &
                       OKMAXQ,OKSQMAXQ,OKMAXXQ,OKMAXYQ,OKMAXZQ
!
      integer,save :: NATM_old
      real(chm_real),allocatable,dimension(:),save :: PKVECQ
      real(chm_real),allocatable,dimension(:),save :: PKTABXCQ
      real(chm_real),allocatable,dimension(:),save :: PKTABXSQ
      real(chm_real),allocatable,dimension(:),save :: PKTABYCQ
      real(chm_real),allocatable,dimension(:),save :: PKTABYSQ
      real(chm_real),allocatable,dimension(:),save :: PKTABZCQ
      real(chm_real),allocatable,dimension(:),save :: PKTABZSQ
#if KEY_MNDO97==0
      real(chm_real), save :: CGMM  ! works for MNDO97.
#endif

!
!     gradient from Kspace sum
      real(chm_real),allocatable,dimension(:,:),save :: PDXYZ
      real(chm_real),allocatable,dimension(:,:),save :: DXYZ_local
      logical, save :: QFIRSTD                                           

! for non-bond list generation
!
      !if KEY_QCHEM==1 || KEY_GAMESS==1 || KEY_GAMESSUK==1 || KEY_QTURBO==1 || KEY_G09==1
#if KEY_MNDO97==0
      integer, save :: NUMAT
#endif
#endif
#else /*   (nosquantmstuff)*/
#if KEY_QCHEM==1 || KEY_GAMESS==1 || KEY_GAMESSUK==1 || KEY_QTURBO==1 || KEY_G09==1
      INTEGER,parameter :: LM1=6000, LM1M=40000*2
#if KEY_MNDO97==0
      logical, save :: LEWMODE
#endif
      real(chm_real),save :: CGQMMM(LM1+LM1M)                            
#endif /*  */
#endif /*  (nosquantmstuff)*/
#endif
!
#if KEY_MNDO97==1
      integer,save :: numat,num_cpus
#if KEY_SQUANTM==0 /* not squantm */
      logical,save :: lqmewd   ! temporary
#endif /* not squantm */
      real(chm_real), save :: CGMM              ! works for MNDO97.
!      INTEGER NUMAT, NAT(LM1), NFIRST(LM1), NLAST(LM1)
!      CHARACTER(len=80) KTITLE, KOMENT
!      CHARACTER(len=2) ELEMNT(107)
#endif
! D3BJ & H4 corrections
#if KEY_MNDO97==1
      ! Enables Dispersion energy (?): DISP option in scc-dftb
!      logical,save        :: l_disp
      ! Enables Grimme's DFT-D3 two-body dispersion (D3(BJ))
      logical,save        :: lmndod2
      ! Enables Grimme's DFT-D3 three-body dispersion
      logical,save        :: lmndod3
      ! ---------------------------------------------
      ! Dispersion related [nolonger in blockscc]
      real(chm_real)      :: Edis
      logical,save        :: dispers
      integer, parameter  :: nndim=650,maxtyp=100
      real(chm_real),save :: Adis,Bdis,Cdis,r0dis,rvdis
      real(chm_real),save :: C6(NNDIM,NNDIM),Rvdw(NNDIM,NNDIM)
      integer,save        :: izp(nndim),ntype
#endif
! end
!=================================================================================
!
contains
  subroutine mndo97_iniall()
#if KEY_MNDO97==1
    numat = 0
    lqmewd=.false.
!    lqmewd2=.false.
!    maxkvq=0
!    qsetupkq=.false.
!    natm_old =0
    num_cpus=4   ! number of cpus to switch in openMP/MPI routine.
    ! Grimme's DFT-D3
    lmndod2 =.false.
    lmndod3 =.false.
!    l_disp  =.false.
    dispers =.false.
#endif
    return
  end subroutine mndo97_iniall
end module  mndo97
