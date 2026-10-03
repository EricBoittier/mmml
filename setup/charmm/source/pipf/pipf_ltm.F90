module pipfm
  use chm_kinds
  use dimens_fcm
  implicit none

!     PIPF information
!
!     Purpose: Stores dipole convergence criteria and max iteration
!              and other variables used by the PIPF utility
!
!     Variable        Purpose
!
!     QPIPF           flag to do pipf
!
! Iterative dipole:
!     DTHRES          induced dipole convergence (in deby)
!     ITRMX           maximum iteration
!     INDP            pointer to induced dipoles
!
! Dipole dynamics:
!     QPFDYN          flag to do dipole dynamics (false means iterative)
!     QMINV           flag to calculate dipole by matrix inversion
!     QMPOL           flag to calculate molecular polarizibility, working
!                     only with matrix inversion procedure
!     QVPOL           vabrational analysis with polarization
!     IUMAS           pointer to dipole mass in heap
!     UMAS            fictitous mass dipoles of selected atoms
!     NHFLAG          Nose-Hoover dipole heat bath option
!     TSTAU           temperature (K) for the Maxwell-Boltzmann draw of
!                     the initial induced-dipole velocities ("tsta"
!                     keyword).  TSTAU<=0 leaves the dipoles cold.
!     QPFVST          one-shot flag: initial dipole velocities still need
!                     to be seeded (armed for a fresh-start run, cleared
!                     after PFVSEED runs on the first PFDYN step)
!     IPFVSD          RNG stream number for the dipole-velocity draw
!                     (captured from the dynamics ISEED)
!     IPFVGO          Gaussian option for the draw (captured from IASVEL)
!     QPFBA           flag of whether PFBA keyword has been found
!     QDYFST          flag of the first dynamic step
!     IUINDSV         pointer to dipole moment saved in heap 
!                     (dipole in a unit of e*A)
!     NUFRS           inital induced dipole for first dynamics step
!     NPFPR           number of primary cell atoms 
!     NPFIM           number of atoms including image atoms
!     QUEANG          flag to calculate the average angle between
!                     the dynamical induced dipole with the total
!                     electric field. 
!     IESAV           the heap index to collect the total electric field
!                     if one calculates the average U-E angle for an
!                     system with images present
!
! General:
!     PFCTOF          energy cut-off for polarization
!     NPDAMP          option for damping removing 1-4 interaction and damp
!                     1-5 interaction by a damping function 
!                     0 - no damping (default)
!                     1 - Thole's roh2, used by Ren&Ponder)
!                     2 - Thole's roh4
!     PFMODE          the mode for iterative procedure
!                     0 - start from 0 dipole
!                     1 - start from last dynamical step
!     QFSTDP          first dynamic step
!     NAVDIP          number of atoms within each molecule to compute
!                     average dipoles
!     QPFEX           option to exclude 1-4 interaction
!

!---mfc--- Needs allocation routine

      INTEGER ITRMX
      real(chm_real)  DTHRES

      LOGICAL QPIPF,QPFDYN,QMINV,QMPOL,QVPOL,PFBASETUP,QPFBA,QDYFST, &
              QUEANG,QPFEX,QFSTDP
      ! One-shot first-PFDYN-step flag (fresh start only): draw the
      ! initial induced-dipole velocities from a Maxwell-Boltzmann
      ! distribution at temperature TSTAU and seed the Verlet history
      ! UINDO=UIND-VUIND*DELTA (cold, UINDO=UIND, when TSTAU<=0).  See
      ! PFVSEED/PFDYN.  Defaults .FALSE. so restart runs (which carry
      ! UINDO from the restart file) are never overridden.
      LOGICAL :: QPFVST = .FALSE.
      ! RNG stream number and Gaussian option for the dipole-velocity
      ! draw, captured from the dynamics ISEED/IASVEL in DCNTRL.
      INTEGER IPFVSD,IPFVGO

      real(chm_real) UMAS,TSTAU

!yw      real(chm_real),pointer,dimension(:,:) :: DUINDSV,iuindsv
      real(chm_real),allocatable,dimension(:,:) :: UIND,DUIND,IESAV,INDP
      real(chm_real),allocatable,dimension(:) :: IUMAS
      
      INTEGER NHFLAG,NUFRS,NPFPR,NPFIM, &
              IFRSTAPP,NATOMPP,IFRSTAIP,NATOMIP

      INTEGER NPFBATHS,NDGFBPF(10),IFSTBPF(10),ILSTBPF(10)

      real(chm_real),dimension(10) :: &
           KEUBATH,PFNHSBATH,PFNHSOBATH,PFNHMBATH,PFTEMPBATH

      real(chm_real)  PFCTOF
 
      INTEGER NPDAMP,PFMODE

      real(chm_real)  DPFAC

      INTEGER NAVDIP

contains
  subroutine pipf_iniall()      
    QPIPF = .FALSE.
    return
  end subroutine pipf_iniall
end module pipfm

