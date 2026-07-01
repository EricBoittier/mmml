
module sccdftbsrc !Data required in SCCDFTB calculations
  use chm_kinds
  use dimens_fcm

#if KEY_DFTBPLUS==1
  use charmm_dftb, only: CharmmDftb
  use external_charges, only: ExternalCharges
  use klopman_ohno, only: KlopmanOhno
  use gsbpscc_potential, only: gsbpPot
#endif

#if KEY_SCCDFTB==1
! first the original maxinc
      integer, parameter :: nndim=650
      integer, parameter :: maxint=160
      integer, parameter :: mdim=1650
      integer, parameter :: maxtyp=100
      integer, parameter :: maxtab=1002
      integer, parameter :: ldim=9
      integer, parameter :: maxsiz=mdim
      integer, parameter :: maxtp=maxtyp

#if KEY_DFTBPLUS==1
      ! Instance of DFTB+
      type(CharmmDftb) :: dftb_plus
      type(ExternalCharges), target :: dftb_external_charges
      type(KlopmanOhno), target :: dftb_klopman_ohno
      type(gsbpPot), target :: dftb_gsbp_pot
#endif

      character(200) :: all_skfiles(maxtyp,maxtyp)
      logical :: ldftbplus = .false.
      integer :: dmeth
!     ---------------------------------------------
!
!     dylcao  [no longer in blockscc]
!
      real(chm_real) skhtab(10,maxtab,maxtyp,maxtyp)
      real(chm_real) skstab(10,maxtab,maxtyp,maxtyp)
      real(chm_real) skself(3,maxtyp), sr(maxtyp,maxtyp)
      integer dim(maxtyp,maxtyp)
      real(chm_real) espin(maxtyp)

!     repulsive potential related?
      real(chm_real) coeff(6,maxint,maxtyp,maxtyp)
      real(chm_real) xr(2,maxint,maxtyp,maxtyp)
      real(chm_real) efkt(3,maxtyp,maxtyp),cutoff(maxtyp,maxtyp)
      integer numint(maxtyp,maxtyp)

!     SPR Repulsive potential (Gaussian)
      real(chm_real) gbump(3,maxtyp,maxtyp),ronsw(maxtyp,maxtyp)
      real(chm_real) deltar(maxtyp,maxtyp)
      logical lgausw

!!     Hbond [moved from sccdftb_ltm]
!      logical lscchb
!      real*8 kl1


!     ------------------------------------------
!     SCC Nonbond [moved from sccdftb_ltm]
      real(chm_real)  sccfnb
      logical qsccnb,qsccs,qsccsh

!     Control parameters [moved from sccdftb_ltm]
      real(chm_real) TELEC,SCFTOL,QMAT(nndim)
      INTEGER MXITSCF
      LOGICAL LPRVEC

! Guanhua_puja_QC_UW1212
      INTEGER SCCMIX,GENS
      real(chm_real) WSY
! end

!     SCCDIPOLE  LSCCDIP if dipole output is invoked
!     SCCDIPOLE  UDIP    fort for output
!     SCCDIPOLE  LSCFCHG A tighter convergence criterion based on Mulliken

      LOGICAL LSCCDIP,LSCFCHG
      INTEGER UDIP
!
!     ---------------------------------------------
!     More molecular info [no longer in blockscc]
      integer, target :: izp2(nndim)
      integer, target :: lmax(maxtyp)
      integer nbeweg
      real(chm_real) nel

      integer ntype

!     Hubbard
      real(chm_real) qzeroscc(maxtyp,4), uhubb(maxtyp,3)

!     qdiff
      real(chm_real) :: qdiff(nndim)

!     Hubbard derivatives and hbond [sccdftb_ltm]
!     ---------- Hubbard derivative -----------
      logical luhder,luhgau
      real(chm_real) uhder(maxtyp,3)
      real(chm_real) v0_hgau,alp_hgau,q0_hgau
!     MG+Guanhua_QC_UW1205
      logical lscchb,izpxh(maxtyp),lscc3rd
      real(chm_real) kl1
      !common /scchb/ lscchb,kl1,izpxh
      !common /scc3/  lscc3rd
      logical lldep  ! MG_UW1210

!     ---------------------------------------------
!     Dispersion related [nolonger in blockscc]
      real(chm_real) edis
      logical dispers
!     Jeez: Horrible names from dispr!
      real(chm_real) Adis,Bdis,Cdis,r0dis,rvdis
      real(chm_real) C6(NNDIM,NNDIM),Rvdw(NNDIM,NNDIM)
!
!     PBC related [in sccdftb_ltm]
!     logical period
!     integer nlat(3)
!     real(chm_real) boxsiz(3,3), xinvbox(3,3), xnullvec(3)

!     machine accuracy [no longer in blockscc_ltm]
      real(chm_real) racc

!     eglcao I/O [no longer in blockscc]
      integer writeout

      ! Enables Grimme's DFT-D3 two-body dispersion (D3(BJ))
      logical lsccdftd2

      ! Enables Grimme's DFT-D3 three-body dispersion
      logical lsccdftd3

      ! ASC_UW2014
      ! Enables the CPE correction for DFTD3
      logical lcpe
      logical lcpe0
      logical lcpeq
!     ---------------------------------------------
!     MG_QC_UW1206: KO-scheme from Guanhua
      real(chm_real), target :: kalpha(maxtyp)
      real(chm_real), target :: kbeta(maxtyp)
      real(chm_real) :: mmuh(maxtyp*10)
      real(chm_real), target :: mmuhub(120240)
      logical lcdko
      integer nmmtype
      character(len=6) mmname(maxtyp*10)
      !common /kopara/ lcdko,kalpha,kbeta,nmmtype,mmname,mmuh
      !save /kopara/

!     ----------------------------------------------
!     PZ, MG_QC_UW1206:  Particle mesh ewald (PME)
      logical qsccpme
      integer fftx,ffty,fftz,fftorder
      real(chm_real) ksgrd(3*NNDIM)
      !common/sccpme/qsccpme,fftx,ffty,fftz,fftorder,ksgrd(3*nndim)

!     ----------------------------------------------
!     MG_QC_UW1206: spin-polarization, MG_UW1210 ldep added
      logical lcolspin
      real(chm_real) unpe                 ! number of unpaired electrons
      real(chm_real) nelup,neldown        ! number of up/down-electrons
      real(chm_real) wspin(maxtyp,3,3)    ! atomic spin constants
      real(chm_real) ql(3*nndim)          ! total l-shell dependent Mulliken charge
      real(chm_real) qlup(3*nndim)        !       for up-electrons
      real(chm_real) qldown(3*nndim)      !       for down-electrons

!     -------------------------------------------------
!     GUISHAN_ZHENG 03/17/2013: DXL-BOMD (Lagrangian formalism of BOMD with
!     dissipation)
      integer, parameter :: MSTEP = 15
      real(chm_real), parameter :: scftight = 1.0d-10
      real(chm_real) :: qmats(NNDIM,MSTEP)     ! previous steps' charge vector
      real(chm_real) :: qaux(NNDIM,MSTEP)      ! auxiliary charge vector
      real(chm_real) :: scftol0
!     -------------------------------------------------
!     0: previous step.  101-117 DXL-BOMD(JCP, 135, 044122(2011))
      integer :: iguess, idamp                 ! initial guess option and # of damping steps
!     -------------------------------------------------

!     QC_UW1710: based on XL_QC_UW1505: NBO
      logical qnbo
      integer elenuc(nndim)


#endif
end module sccdftbsrc

#if KEY_DFTBPLUS==0
! Satisfy the dependency checker trivially.
! These are declared in the external library, but the configure
! scripts searches in CHARMM's source, thus it will NEVER find
! any external modules. The only way to circumvent this is to
! insert dummy module definitions that will never compile.
      module dftbplus_interface
      end module dftbplus_interface
      module charmm_dftb
      end module charmm_dftb
      module external_charges
      end module external_charges
      module klopman_ohno
      end module klopman_ohno
      module gsbpscc_potential
      end module gsbpscc_potential
#endif /* KEY_DFTBPLUS==0 */
