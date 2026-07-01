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
!     TSTAU           initial dipole velocity
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

