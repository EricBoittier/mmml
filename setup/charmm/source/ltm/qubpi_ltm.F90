module qubpi
  use chm_kinds
  use dimens_fcm
!CHARMM Element source/fcm/qmlink.fcm 1.1
#if KEY_QUANTUM==1 || KEY_SCCDFTB==1 || KEY_SQUANTM==1 || KEY_GAMESSUK==1 || KEY_QCHEM==1 || KEY_QTURBO==1
!
!     Defines the data necessary for a QM/PI calculation 
!     for FEP from classical to quantum potential energy surface.
!     Jiali Gao, 9/15/01
!     Dan T. Major, 23/12/2010
!  
!     Variable  Index    Purpose 
! 
!     QCP                Use QCP method
!     BQCP               Use BQCP method - automatically sets BISE keyword
!     SQCP               Use SQCP method - automatically sets STAGE keyword
!     QFEP               Use PI-FEP/UM method
!     QCOP               Use QCOPI method - quantum-classical open path-integral calulation - 
!                        automatically sets STAGE and OSTAGE keywords
!     NPIATM             Number of QM/PI nuclei
!     IPIATM(i)          PSF atom number of ith PI nuclei
!     IQMPIATM(i)        Array of QM atoms that are also PI atoms
!     QNUQM(i)           Array specifying method (BQCP,...)
!     NBEADSQ            number of quasiparticles (beads) of spring
!     NBEADSQQ           effective # of beads - same as NBEADSQ unless open PI
!     MCCONF             number of Monte Carlo path configurations
!                           for each CL configuration
!     MCEQUIL            number of MC paths for equilibration
!     NBMOVE             number of beads to be changed in each MC move
!     NBIN_UNIT          input file unit for all beads (initial guess)
!     NBOUT_UNIT         output file unit for all beads (at end of MC/PI)
!     NBPDB_UNIT         output pdb file for all beads 
!     NQUBOUT            unit for print out of QM-MC/PI results
!     NQUBOUT2           unit for print out of QM-MC/PI results of isotope #2
!
!     FPITEMP
!     BETA_FPI           = 1/kT
!     LAMBDA0
!     LAMBDA             De Broglie wavelength for free particles
!     LAMBDA2            For Metropolis, account for multibead moves (smaller displacement)
!     TIAC               Logical for Takahashi-Imada action
!     CHAC               Logical for Chin action
!     HOFAC              Pre-mult. factor for higher-order (TI or Chin) action
!     CHFACN             Inverse value of factorization level in Chin (typically 1/3)
!     PIMCMV             mult. factor for atom moves (multiplies LAMBDA in Metropolis MC)
!     MSQRAT             Square root of ratio btw masses (FEP)
!     BISE               Bisect or not - QCP with BISE sets BQCP keyword
!     STAGE              Staging algorithm - QCP with staging sets SQCP keyword
!     OSTAGE             Open-chain PI sampling
!     KLEV               Level of bisecting (2**klev - 1 beads moved)
!     NOEWALD            Do not use Ewald summation
! The following 2 options only work with the old mopac code (obsolete)
!     FENER              Use fast EAMPAC routine (QUBEAMPAC) avoiding gradient calculation
!     FFOCK              Use fast Fock matrix updating
!
!     QRXN               Reaction coordinate analysis or not
!     RXNA,RXNB,RXNC     3 atoms defining reaction coordinate
!     ISTR               Isotropic sampling of open chain PI along A---C axis (donor-acceptor atoms)
!     MAXBIN             Max number of bins allowed in analysis of reaction coordinate 
!     BINWIDTH           Width of bins
!     RESLN              Resolution in reaction coordinate analysis
!
!     MAXPIATM           Number of QM//QM atoms
!
!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

      integer, parameter :: MINBEADS=2,MAXBEADS=1024,MAXBIN=2000,MAXPIATM=100
      integer, parameter :: MAXNUQM=6,QCP=1,BQCP=2,SQCP=3,BFEP=4,SFEP=5,QCOP=6  ! Additional QUB methods defined here

      integer, save :: NPIATM,IPIATM(MAXPIATM),IQMPIATM(MAXPIATM),NBEADSQ,NBEADSQQ, &        
!                       MCCONF,MCEQUIL,NBMOVE,IBMOVE(MAXPIATM),NBIN_UNIT, &
                       MCCONF,MCEQUIL,NBMOVE,NBIN_UNIT, &
                       NBOUT_UNIT,NBPDB_UNIT,NQUBOUT,NQUBOUT2,NMOMDIS,NMOMDIS2,KLEV,RXNA,RXNB,RXNC
      real(chm_real), save :: FPITEMP,BETA_FPI,FPIFAC,BINWIDTH,RESLN,PIMCMV,CHFACN,T0,T1,U0, &      
                              LAMBDA0(MAXPIATM),LAMBDA(MAXPIATM),LAMBDA2(MAXPIATM), &
                              MSQRAT(MAXPIATM),HOFAC(MAXPIATM)
      logical, save :: BISE,STAGE,OSTAGE,QFEP,TIAC,CHAC,QRXN,ISTR,FENER,FFOCK,NOEWALD,QNUQM(MAXNUQM)                 

!
!  MONTE CARLO RANDOM NUMBER GENERATOR SEED
!
      INTEGER, save :: IRN                                                          
!
#endif 
end module qubpi

