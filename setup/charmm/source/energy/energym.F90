module energym
  use chm_kinds
  use dimens_fcm
  implicit none
!=======================================================================
!
!     ENERGY.FCM stores the energy data structure.
!
!=======================================================================
!
!  LENENP - Maximum number of energy related properties.
!  LENENT - Maximum number of energy terms.
!  LENENV - Maximum number of pressure/virial terms.
!
!  !!WARNING!!: These lengths need to be changed carefully.  The dynamic
!               restart file will be modified when these are changed.
!               LENENP and LENENT may be increased (not decreased), but
!               neither may exceed 120.  LENENV should remain at 50.
!               Examine the code in dynamc/dynio.src(READYN).
!
      integer, parameter :: LENENP = 60, LENENT = 128, LENENV = 50
!
!=======================================================================
! . Energy array indexes.
!
! !!! IMPORTANT NOTE: Any changes here must be mirrored below and also
!                     in energy/eutil.src(ENERIN) if the value will
!                     be accessible to the parser!!!
!    Note: If pointers are changed, then old dynamic restart files where
!    these values are required will be unusable.
!
      integer,PARAMETER :: TOTE   =  1, TOTKE  =  2, EPOT   =  3, TEMPS  =  4, &
                 GRMS   =  5, BPRESS =  6, PJNK1  =  7, PJNK2  =  8, &
                 PJNK3  =  9, PJNK4  = 10, HFCTE  = 11, HFCKE  = 12, &
                 EHFC   = 13, EWORK  = 11, VOLUME = 15, PRESSE = 16, &
                 PRESSI = 17, VIRI   = 18, VIRE   = 19, VIRKE  = 20, &
                 TEPR   = 21, PEPR   = 22, KEPR   = 23, KEPR2  = 24, &
                              DROFFA = 26, XTLTE  = 27, XTLKE  = 28, &
                 XTLPE  = 29, XTLTEM = 30, XTLPEP = 31, XTLKEP = 32, &
                 XTLKP2 = 33, &
                 TOT4   = 37, TOTK4  = 38, EPOT4  = 39, TEM4   = 40, &
                 MbMom  = 41, BodyT  = 42, PartT  = 43
#if KEY_ACE==1
      integer,PARAMETER ::SELF   = 45, SCREEN = 46, COUL   = 47,       & 
#endif
#if KEY_ACE==1
                 SOLV   = 48, INTER  = 49                                
#endif
#if KEY_FLUCQ==1
      integer,PARAMETER :: FQKIN  = 50                                
#endif
#if KEY_CHEQ==1
      integer,PARAMETER :: CGKE   = 51, CGPOT  = 52, ALLK= 53, ALLP = 54, &
                 ALLE   = 55, SDMIN  = 56
#endif 
#if KEY_PIPF==1
      integer,PARAMETER :: DIPK   = 57, DIPT  = 58 
#endif
#if KEY_PHMD==1
      integer,PARAMETER :: PHKIN  = 59             
#endif
!     
!      TOTE    - total energy including EHFC
!      TOTKE   - total kinetic energy (u**2)
!      EPOT    - potential energy
!      TEMPS   - temperature
!      GRMS    - rms gradient
!      BPRESS  - boundary pressure
!      PJNK1   - utility (see various minimization methods)
!      PJNK2   - utility
!      PJNK3   - utility
!      PJNK4   - utility
!      HFCTE   - high frequency corrected total energy
!      HFCKE   - high frequency corrected kinetic energy (u**2)
!      EHFC    - high frequency correction energy
!      EWORK   - work term (fv) !! note this is equivalenced to HFCTE
!      VOLUME  - unit cell volume, or specified volume
!      PRESSE  - external pressure
!      PRESSI  - internal pressure
!      VIRI    - internal virial
!      VIRE    - external virial
!      VIRKE   - virial energy (fr)
!      TEPR    - previous total energy
!      PEPR    - previous potential energy
!      KEPR    - previous kinetic energy (Un-1/2**2)
!      KEPR2   - second previous kinetic energy  (Un-3/2**2)
!      DEPRES  - energy lost each step due to constant pressure method
!      DROFFA  - Geometric average of DROFF values in the MMFP code
!      TOT4    - total 4-D energy
!      TOTK4   - total 4-D kinetic energy
!      EPOT4   - potential 4-D energy
!      TEM4    - 4-D temperature
#if KEY_ACE==1
!      SELF    - self energies
!      SCREEN  - screening of coulombic interactions
!      COUL    - coulomb interactions
!      SOLV    - solvation energy
!      INTER   - interaction energy
#endif 
! SAPATEL
#if KEY_CHEQ==1
!       charge dynamics variables:
!      CGKE    - kintic energy derived from charge fluctuation
!      CGPOT   - cg potential (cg dependent part of the DFT energy)
!      ALLK    - total Kinetic, normal and charge
!      ALLP    - total Potential, normal and charge
!      ALLE    - total energy, ALLK+ALLP
#endif 
! SAPATEL
#if KEY_GRID==1
!      GrvdW   - grid-based vdW energy
!      GrElec  - grid-based elec energy
!      GrHBON  - grid-based hydrogen bond energy   !! Yujin
#endif 
#if KEY_FLUCQ==1
!      FQKIN   - Fluctuating charge kinetic energy              
#endif
#if KEY_CMAP==1
!      CMAP    - Two-dihedral cross term                        
#endif
#if KEY_LRVDW==1
!      ELRC    - Long range vdW energy correction               
#endif
#if KEY_OVERLAP==1
!      QOVLAP  - Energy for Atomic Overlap                      
#endif
#if KEY_PIPF==1
!      DIPK    - PIPF dipole kinetic energy
!      DIPT    - PIPF dipole temperature
#endif 
#if KEY_PHMD==1
!     PHKIN    - lambda kinetic energy in PHMD
#endif 
#if KEY_EMAP==1
!     EEMAP    -  energy from map objects
#endif 
#if KEY_CONSHELIX==1
!      ECHDL   - Helix-helix distrance restraint energy
!              - Two centers of Mass distance restraint energy
!      ECHAN   - Helix-helix cross angle restraint energy
!              - Helix-helix hinge angle restraint energy
!      ECHTN   - Helix tilt angle restraint energy
!      ECHRN   - Helix rotation angle restraint energy
#endif 
!
! !!! IMPORTANT NOTE: Any changes here must be mirrored below
!                     and also in energy/eutil.src(ENERIN) !!!
!    Note: If energy term pointers are changed, then old dynamic restart
!    files where these values are required will be unusable.
!
 
      integer,PARAMETER :: &
           BOND   =  1, ANGLE  =  2, UREYB  =  3, DIHE   =  4, &
           IMDIHE =  5, VDW    =  6, ELEC   =  7, HBOND  =  8, &
           USER   =  9, CHARM  = 10, CDIHE  = 11, CINTCR = 12, &
           CQRT   = 13, NOE    = 14, SBNDRY = 15, IMVDW  = 16, &
           IMELEC = 17, IMHBND = 18, EWKSUM = 19, EWSELF = 20, &
           EXTNDE = 21, RXNFLD = 22, ST2    = 23, IMST2  = 24, &
           TSM    = 25, QMEL   = 26, QMVDW  = 27, ASP    = 28, &
           EHARM  = 29, GEO    = 30, MDIP   = 31, PINT   = 32
#if KEY_RPATH==1
      integer,PARAMETER ::PRMS   = 33, PANG   = 34       
#endif
      integer,PARAMETER ::SSBP   = 35, BK4D   = 36, &
           SHEL   = 37, RESD   = 38, SHAP   = 39, STRB   = 40, &
           OOPL   = 41, PULL   = 42, POLAR  = 43, DMC    = 44, &
           RGY    = 45, EWEXCL = 46, EWQCOR = 47, EWUTIL = 48, &
           PBELEC = 49, PBNP   = 50, MbDefrm= 51, MbElec = 52, &
           STRSTR = 53, BNDBND = 54, BNDTW  = 55, EBST   = 56, &
           MBST   = 57, BBT    = 58, SST    = 59, GBEnr  = 60, &
           GSBP   = 64

#if KEY_HMCOM==1
      integer,PARAMETER :: HMCM   = 61            
#endif
      integer,PARAMETER :: ADUMB  = 62            ! ad050629
#if KEY_ACE==1
      integer,PARAMETER :: HYDR   = 63            
#endif
#if KEY_FLUCQ==1
      integer,PARAMETER :: FQPOL  = 65             
#endif
#if KEY_GRID==1
      integer,PARAMETER :: GrvdW  = 66, GrElec = 67, GrHBON = 111 !! Yujin wu 
#endif
#if KEY_SASAE==1
      integer,PARAMETER :: SASTRM = 68             
#endif
      integer,PARAMETER :: CMAP   = 69
      integer,PARAMETER :: ELRC   = 70
#if KEY_OVERLAP==1
      integer,PARAMETER :: QOVLAP = 71             
#endif

#if KEY_PIPF==1
      integer,PARAMETER :: PIPF   = 72             
#endif
      integer,PARAMETER :: UMBR   = 73             ! ad050629
      integer,PARAMETER :: rushRepu = 74, rushPhob = 75, rushHbnd = 76 &
               , rushBdon = 77, rushBacc = 78, rushArom = 79
#if KEY_PHMD==1
      integer,PARAMETER :: PHEnr  = 80             
#endif
#if KEY_MMPT==1
       integer,PARAMETER :: MMPT   = 83             
#endif
      integer,PARAMETER :: VMOD   = 84
#if KEY_RMD==1
      integer,PARAMETER :: CROS   = 85             
#endif
#if KEY_PNM==1
      integer,PARAMETER :: PNME   = 86             
#endif
#if KEY_FACTS==1
      integer,PARAMETER :: IFCTPOL = 87             
#endif
#if KEY_FACTS==1
      integer,PARAMETER :: IFCTNPL = 88             
#endif
#if KEY_RPATH==1
      integer,PARAMETER :: PATH   = 89              
#endif

#if KEY_EPMF==1
      integer,PARAMETER :: PMF1D = 90, PMF2D = 91  
#endif
#if KEY_PRIMO==1
      integer,PARAMETER :: PRIMO = 92  
#endif
#if KEY_VALBOND==1
      integer,PARAMETER :: VALB = 93               
#endif
#if KEY_EDS==1
      integer,PARAMETER :: EDS = 94    
#endif
#if KEY_CONSHELIX==1
      integer,PARAMETER :: ECHDL = 95, ECHAN = 96, ECHTN = 97, &
                  ECHRN = 98
#endif 
      integer,parameter :: cpuck = 99              !puckering restraint
      integer,parameter :: SMBP = 100              !JZ_UW1207
#if KEY_GAMUS==1
      integer,parameter :: GAMUS = 101             
#endif
#if KEY_EMAP==1
      integer,PARAMETER :: EEMAP = 102             
#endif
#if KEY_MRMD==1
      integer,PARAMETER :: MRMD = 103
#endif
#if KEY_LARMORD==1
      integer, parameter :: cspres = 104
#endif
#if KEY_SSNMR==1
      integer,PARAMETER :: ECS = 105
#endif
#if KEY_RDC==1
      integer,PARAMETER :: ERDC = 106
#endif
#if KEY_DHDGB==1
!AP/MF
      integer,parameter :: DEFE = 107
#endif
#if KEY_MSMMPT==1
       integer,PARAMETER :: MSMMPT   = 108
#endif
#if KEY_LJPME==1
      integer, parameter :: LJSEL = 108
      integer, parameter :: LJREC = 109
      integer, parameter :: LJEXC = 110
#endif
      integer, parameter :: PCHARM = 112 ! principal component harmonic restraints
!
      integer,PARAMETER :: VEXX =  1, VEXY =  2, VEXZ =  3, VEYX =  4, &
                  VEYY =  5, VEYZ =  6, VEZX =  7, VEZY =  8, &
                  VEZZ =  9, &
                  VIXX = 10, VIXY = 11, VIXZ = 12, VIYX = 13, &
                  VIYY = 14, VIYZ = 15, VIZX = 16, VIZY = 17, &
                  VIZZ = 18, &
                  PEXX = 19, PEXY = 20, PEXZ = 21, PEYX = 22, &
                  PEYY = 23, PEYZ = 24, PEZX = 25, PEZY = 26, &
                  PEZZ = 27, &
                  PIXX = 28, PIXY = 29, PIXZ = 30, PIYX = 31, &
                  PIYY = 32, PIYZ = 33, PIZX = 34, PIZY = 35, &
                  PIZZ = 36
!
! . Energy term character arrays.
      CHARACTER(len=4) ::  CEPROP(lenenp), CETERM(lenent), CEPRSS(lenenv)
! . Energy term logical arrays.
      logical :: QEPROP(LENENP), QETERM(LENENT), QEPRSS(LENENV)
      logical QDYNCALL
! . Energy arrays.
      real(chm_real) :: EPROP(LENENP), ETERM(LENENT), EPRESS(LENENV)
!
#if KEY_BLOCK==1
!.ab.HYBridHamiltonian.
      real(chm_real)  :: ETERMR(LENENT),ETERMP(LENENT), &
                      EPROPR(LENENP),EPROPP(LENENP)
#endif 

!=======================================================================
! . Miscellaneous energy variables :
! .    ECALLS is incremented by ICALL in ENERGY(,,,,ICALL). Used for updating.
! .    TOT1ST counts all calls where the first  derivative gets computed.
! .    TOT2ND counts all calls where the second derivative gets computed.
      INTEGER  ECALLS, TOT1ST, TOT2ND
      real(chm_real)   EOLD, FITA, DRIFTA, EAT0A, CORRA, FITP, DRIFTP, &
               EAT0P, CORRP
#if KEY_LRVDW==1
      real(chm_real) PVLRC                                    
#endif
!
!=======================================================================
! Names are set in energy/eutil.src(ENERIN) (Make sure it's correct!)
!
! ENERGY PROPERTY NAMES
! . Initialise the energy property names.
!     (TOTE)   = 'TOTE'  - total energy
!     (TOTKE)  = 'TOTK'  - total kinetic energy
!     (EPOT)   = 'ENER'  - total potential energy
!     (TEMPS)  = 'TEMP'  - temperature (from KE)
!     (GRMS)   = 'GRMS'  - rms gradient
!     (BPRESS) = 'BPRE'  - boundary pressure applied
!     (HFCTE)  = 'HFCT'  - total energy with HFC
!     (HFCKE)  = 'HFCK'  - total kinetic energy with HFC
!     (EHFC)   = 'EHFC'  - high frequency correction energy
!     (EWORK)  = 'EHYS'  - slow growth hysteresis energy correction
!     (VOLUME) = 'VOLU'  - the volume of the system.
!     (PRESSE) = 'PRSE'  - the pressure calculated from the external virial.
!     (PRESSI) = 'PRSI'  - the pressure calculated from the internal virial.
!     (VIRE)   = 'VIRE'  - the external virial.
!     (VIRI)   = 'VIRI'  - the internal virial.
!     (VIRKE)  = 'VIRK'  - the virial "kinetic energy".
#if KEY_ACE==1
!     (SELF)   = 'SELF'  - self energies
!     (SCREEN) = 'SCRE'  - screening of coulombic interactions
!     (COUL)   = 'COUL'  - coulomb interactions
!     (SOLV)   = 'SOLV'  - total electrostatic solvation energy (ACE)
!     (INTER)  = 'INTE'  - total electrostatic interaction energy (ACE)
#endif 
#if KEY_FLUCQ==1
!     (FQKIN)  = 'FQKI'  - Fluctuating charge kinetic energy  
#endif
!
! ENERGY TERM NAMES:
!     (BOND)   = 'BOND'  - bond (1-2) energy
!     (ANGLE)  = 'ANGL'  - angle (1-3) energy
!     (UREYB)  = 'UREY'  - additional 1-3 urey bradley energy
!     (DIHE)   = 'DIHE'  - dihedral 1-4 energy
!     (IMDIHE) = 'IMPR'  - improper planar of chiral energy
#if KEY_CMAP==1
!     (CMAP)   = 'CMAP'  - crossterm
#endif 
!     (VDW)    = 'VDW '  - van der waal energy
!     (ELEC)   = 'ELEC'  - electrostatic energy
!     (HBOND)  = 'HBON'  - hydrogen bonding energy
!     (USER)   = 'USER'  - user supplied energy term
!     (CHARM)  = 'HARM'  - harmonic positional restraint energy
!     (CDIHE)  = 'CDIH'  - dihedral restraint energy
!     (CPUCK)  = 'CPUC'  - puckering restraint energy
!     (CINTCR) = 'CIC '  - internal coordinate restraint energy
!     (CQRT)   = 'CDRO'  - droplet restraint energy (approx const press)
!     (HMCM)   = 'HMCM'  - center of masses harmonic restraint
!     (PATH)   = 'PATH'  - path restraint
!     (EPMF1D) = 'EPMF1D'- 1D PMF potential
!     (EPMF2D) = 'EPMF2D'- 2D PMF potential
!     (PRIMO)  = 'PRIMO' - PRIMO model specific potential terms
!     (NOE)    = 'NOE'   - general distance restraint energy (for NOE)
!     (SBNDRY) = 'SBOU'  - solvent boundary lookup table energy
!     (IMVDW)  = 'IMNB'  - primary-image van der waal energy
!     (IMELEC) = 'IMEL'  - primary-image electrostatic energy
!     (IMHBND) = 'IMHB'  - primary-image hydrogen bond energy
!     (EWKSUM) = 'EWKS'  - Ewald k-space sum energy term
!     (EWSELF) = 'EWSE'  - Ewald self energy term
!     (EXTNDE) = 'EXTE'  - extended electrostatic energy
!     (RXNFLD) = 'RXNF'  - reaction field electrostatic energy
!     (ST2)    = 'ST2'   - ST2 water-water energy
!     (IMST2)  = 'IMST'  - primary-image ST2 water-water energy
!     (TSM)    = 'TSM'   - TMS free energy term
!     (QMEL)   = 'QMEL'  - Quantum (QM) energy with QM/MM electrostatics
!     (QMVDW)  = 'QMVD'  - Quantum (QM/MM) van der Waal term
!     (ASP)    = 'ASP'   - Atomic solvation parameter (surface) energy
!     (EHARM)  = 'EHAR'  - Restraint term for Implicit Euler integration
!     (GEO)    = 'GEO '  - Mean-Field-Potential energy term
!     (MDIP)   = 'MDIP'  - Dipole Mean-Field-Potential energy term
!     (PINT)   = 'PINT'  - Path integral energy
!     (PRMS)   = 'PRMS'  - Replica/Path RMS deviation energy 
!     (PANG)   = 'PANG'  - Replica/Path RMS angle deviation energy 
!     (SSBP)   = 'SSBP'  - Solvent boundary potential energy
!     (BK4D)   = 'BK4D'  - 4-D energy
!     (SHEL)   = 'SHEL'  -
!     (RESD)   = 'RESD'  - Restrained Distance energy
!     (SHAP)   = 'SHAP'  - Shape restraint energy
!     (STRB)   = 'STRB'  - Strech-Bend coupling energy (MMFF and CFF)
!     (OOPL)   = 'OOPL'  - Out-off-plane energy (MMFF and CFF)
#if KEY_OVERLAP==1
!     (OLAP)   = 'OLAP'  - Atomic overlap energy                  
#endif
!     (PULL)   = 'PULL'  - Pulling force energy
!     (POLAR)  = 'POLA'  - Polarizable water energy
!     (DMC )   = 'DMC '  - Distance map restraint energy
!     (RGY )   = 'RGY '  - Radius of Gyration restraint energy
!     (EWEXCL) = 'EWEX'  - Ewald exclusion correction energy
!     (EWQCOR) = 'EWQC'  - Ewald total charge correction energy
!     (EWUTIL) = 'EWUT'  - Ewald utility energy term (for misc. corrections)
!     (PBELEC) = 'PBEL'  - Poisson-Boltzmann electrostatic solvation energy
!     (PBNP)   = 'PBNP'  - Poisson-Boltzmann nonpolar solvation energy
!     (PHENR)  = 'PHEN'  - pH dependent biasing potential
!     (GBENR)  = 'GBEN'  - Generalized Born Energy
!     (GSBP)   = 'GSBP'  - Generalized Solvent Boundary Potential energy
!     (SMBP)   = 'SMBP'  - Solvent Macromolecule Boundary Potential energy
!     (MBDEFRM)= 'MBDE'  - Body mode deformation energy
!     (STRSTR) = 'STRS'  - Stretch-Stretch coupling energy (CFF)
!     (BNDBND) = 'BNDB'  - Bend-Bend coupling energy (CFF)
!     (BNDTW)  = 'BNDT'  - Bend-Twist coupling energy (CFF)
!     (EBST)   = 'EBST'  - End-Bond-Stretch-Twist coupling energy (CFF)
!     (MBST)   = 'MBST'  - Middle-Bond-Stretch-Twist coupling energy (CFF)
!     (BBT)    = 'BBT '  - Bend-Bend-Twist coupling energy (CFF)
!     (SST)    = 'SST '  - Stretch-Stretch-Twist coupling energy (CFF)
!     (ADUMB)  = 'ADUM'  - Adaptive umbrella potential
#if KEY_ACE==1
!     (HYDR)   = 'HYDR'  - hydrophobic effect                     
#endif
#if KEY_GRID==1
!     (GrvdW)  = 'GRVD'  - Grid-based vdW energy                  
#endif
#if KEY_GRID==1
!     (GrElec) = 'GREL'  - Grid-based elec energy                 
#endif
#if KEY_GRID==1
!     (GrHBON) = 'GRHB'  - Grid-based hydrogen bond energy            !! Yujin                
#endif
#if KEY_FLUCQ==1
!     (FQPOL)  = 'FQPO'  - Fluctuating charge polarisation energy 
#endif
#if KEY_SASAE==1
!     (SASTRM) = 'SASL'  - SASA mean solvation energy             
#endif
#if KEY_LRVDW==1
!     (ELRC)   = 'ELRC'  - Long-Range Van derWaals Correction     
#endif
#if KEY_PIPF==1
!     (PIPF)   = 'PIPF'  - Polarizable intermolecular potential   
#endif
!     (UMBR)   = 'UMBR'  - Umbrella potential for RXNCOR
!     (rushRepu) = 'RREP' - RUSH repulsion energy                 
!     (rushPhob) = 'RPHO' - RUSH hydrophobic energy               
!     (rushHbnd) = 'RHBN' - RUSH intra-molecular h-bond energy    
!     (rushBdon) = 'RBDO' - RUSH donor - solvent h-bond energy    
!     (rushBacc) = 'RBAC' - RUSH acceptor - solvent h-bond energy 
!     (rushArom) = 'RARO' - RUSH aromatic - aromatic energy       
#if KEY_MMPT==1
!     (MMPT)     = 'MMPT' - MMPT energy
#endif
#if KEY_MSMMPT==1
!     (MSMMPT)     = 'MSPT' - MSMMPT energy                           
#endif
#if KEY_RMD==1
!     (CROS)     = 'CROS' - Energy correction in surface crossing 
#endif
#if KEY_MRMD==1
!     (MRMD)     = 'ERMD' - MRMD energy correction
#endif
#if KEY_VALBOND==1
!     (VALB)     = 'VALB' - VALBOND bending energy             
#endif
#if KEY_PNM==1
!     (PNME)     = 'PNME' - Plastic network energy                
#endif
#if KEY_FACTS==1
!     (IFCTPOL) = 'FCTP'  - FACTS electrostatic solvation free energy 
#endif
#if KEY_FACTS==1
!     (IFCTNPL) = 'FCTN'  - FACTS nonpolar solvation free energy      
#endif
#if KEY_DHDGB==1
!AP/MF
!      (DEFE) = 'DEFE' - DHDGB implicit membrane deformation energy
#endif
!
! Energy Pressure/Virial Terms:
!     (External Virial   - VEXX) = 'VEXX'
!     (                  - VEXY) = 'VEXY'
!     (                  - VEXZ) = 'VEXZ'
!     (                  - VEYX) = 'VEYX'
!     (                  - VEYY) = 'VEYY'
!     (                  - VEYZ) = 'VEYZ'
!     (                  - VEZX) = 'VEZX'
!     (                  - VEZY) = 'VEZY'
!     (                  - VEZZ) = 'VEZZ'
!     (Internal Virial   - VIXX) = 'VIXX'
!     (                  - VIXY) = 'VIXY'
!     (                  - VIXZ) = 'VIXZ'
!     (                  - VIYX) = 'VIYX'
!     (                  - VIYY) = 'VIYY'
!     (                  - VIYZ) = 'VIYZ'
!     (                  - VIZX) = 'VIZX'
!     (                  - VIZY) = 'VIZY'
!     (                  - VIZZ) = 'VIZZ'
!     (External Pressure - PEXX) = 'PEXX'
!     (                  - PEXY) = 'PEXY'
!     (                  - PEXZ) = 'PEXZ'
!     (                  - PEYX) = 'PEYX'
!     (                  - PEYY) = 'PEYY'
!     (                  - PEYZ) = 'PEYZ'
!     (                  - PEZX) = 'PEZX'
!     (                  - PEZY) = 'PEZY'
!     (                  - PEZZ) = 'PEZZ'
!     (Internal Pressure - PIXX) = 'PIXX'
!     (                  - PIXY) = 'PIXY'
!     (                  - PIXZ) = 'PIXZ'
!     (                  - PIYX) = 'PIYX'
!     (                  - PIYY) = 'PIYY'
!     (                  - PIYZ) = 'PIYZ'
!     (                  - PIZX) = 'PIZX'
!     (                  - PIZY) = 'PIZY'
!     (                  - PIZZ) = 'PIZZ'
!
!
!=======================================================================
! TSM accumulation variables
#if KEY_TSM==1
      real(chm_real) TSMTRM(LENENT),TSMTMP(LENENT)
#endif 
!
#if KEY_HQBM==1
! HQBM and AFM variables
!      real(chm_real) EHQBM,EAFM
!      LOGICAL HQBM,LAFM
! moved variables to afm module
      real(chm_real) EHQBM
      LOGICAL HQBM
#endif 
#if KEY_CONSHELIX==1
   Logical LQSTPRT
#endif 

#if KEY_DOMDEC==1
  ! Auxiliary data
  integer nauxdata
  real(chm_real), allocatable, dimension(:) :: auxdata
#endif 

 contains

   !> Explicit interface wrapper for old_ENERGY.
   subroutine energy(x, y, z, dx, dy, dz, bnbnd, bimag, icall, ndd, dd &
#if KEY_DHDGB==1
!AP/MF
    ,SDEFIN,DS_DHDGBOUT &
#endif
     )
      use chm_types
      use number
      use memory,only:chmalloc,chmdealloc
#if KEY_DHDGB==1
!AP/MF
     use dhdgb,only:qfhdgb
#endif

      real(chm_real), intent(in) :: x(:), y(:), z(:)
      real(chm_real), intent(inout) :: dx(:), dy(:), dz(:)
#if KEY_DHDGB==1
!AP/MF
      REAL(chm_real), intent(in), optional :: sdefin(:)
      REAL(chm_real), intent(inout), optional :: ds_dhdgbout(:)
#endif
      type(nonbondDataStructure), intent(in) :: bnbnd
      type(imageDataStructure), intent(in) :: bimag
      integer, intent(in) :: icall
      ! only where caller wants second derivatives
      integer, intent(in), optional :: ndd
      real(chm_real), intent(inout), optional :: dd(:)
      ! Variables
      integer,allocatable,dimension(:) :: iupt,kupt
#if KEY_DHDGB==1
!AP/MF
    IF (.NOT. QFHDGB) THEN
#endif
      if (present(ndd) .and. present(dd)) then
         ! ndd ~= floor(sqrt(2 * size(dd)))
         call chmalloc('energym.src','energy','iupt',ndd,intg=iupt)
         call chmalloc('energym.src','energy','kupt',ndd,intg=kupt)
         call old_ENERGY(x, y, z, dx, dy, dz, bnbnd, bimag, &
               ndd, dd, iupt, kupt, .true., icall &
#if KEY_DHDGB==1
!AP/MF
              , [0d0], [0d0] &
#endif
)
         call chmdealloc('energym.src','energy','iupt',ndd,intg=iupt)
         call chmdealloc('energym.src','energy','kupt',ndd,intg=kupt)
      else
         call old_ENERGY(x, y, z, dx, dy, dz, bnbnd, bimag, &
               0, [ZERO], [0], [0], .false., icall &
#if KEY_DHDGB==1
!AP/MF
              , [0d0], [0d0] &
#endif
)
      endif
#if KEY_DHDGB==1
!AP/MF
    ELSE
     if (present(sdefin) .and. present(ds_dhdgbout)) then
      if (present(ndd) .and. present(dd)) then
         ! ndd ~= floor(sqrt(2 * size(dd)))
         call old_ENERGY(x, y, z, dx, dy, dz, bnbnd, bimag, &
              ndd, dd, [0], [0], .true., icall &
              , sdefin, ds_dhdgbout &
             )
       else
         call old_ENERGY(x, y, z, dx, dy, dz, bnbnd, bimag, &
               0, [ZERO], [0], [0], .false., icall &
              , sdefin, ds_dhdgbout &
             )  
       endif
     else
      QFHDGB=.FALSE. 
      if (present(ndd) .and. present(dd)) then
         ! ndd ~= floor(sqrt(2 * size(dd)))
         call old_ENERGY(x, y, z, dx, dy, dz, bnbnd, bimag, &
              ndd, dd, [0], [0], .true., icall &
              , [0d0], [0d0] &
             )
       else
         call old_ENERGY(x, y, z, dx, dy, dz, bnbnd, bimag, &
               0, [ZERO], [0], [0], .false., icall &
              , [0d0], [0d0] &
             )
       endif
     endif
    ENDIF
#endif
   end subroutine energy

!!$   ! *
!!$   ! * Adds data to auxdata -array
!!$   ! *
!!$   subroutine add_to_auxdata(nauxdata, auxdata, dmc_rho, vpress)
!!$     use chm_kinds
!!$     use domdec_common,only:q_domdec, q_cons_node
!!$     !  use energym,only:eterm, qeterm, dmc, umbr, adumb
#if KEY_DMCONS==1
!!$     use dmcons,only:qdmc            
#endif
#if KEY_RXNCOR==1
!!$     use rxncom,only:rxnind          
#endif
#if KEY_ADUMB==1
!!$     use umb, only: numbr, coum      
#endif
!!$     implicit none
!!$     ! Input / Output
!!$     integer, intent(inout) :: nauxdata
!!$     real(chm_real), allocatable, dimension(:), intent(inout) :: auxdata
!!$     real(chm_real), intent(in) :: dmc_rho, vpress(9)
!!$     ! Variables
!!$     logical q_add_vpress
!!$     
!!$     if (q_domdec .and. q_cons_node) then
!!$        q_add_vpress = .false.
!!$##IF DMCONS
!!$        if (qdmc) then
!!$           call realloc(nauxdata+2, auxdata)
!!$           auxdata(nauxdata+1) = eterm(dmc)
!!$           auxdata(nauxdata+2) = dmc_rho
!!$           nauxdata = nauxdata + 2
!!$           q_add_vpress = .true.
!!$        endif
!!$##ENDIF
!!$##IF RXNCOR
!!$        if (rxnind /= 0 .and. qeterm(umbr)) then
!!$           call realloc(nauxdata+1, auxdata)
!!$           auxdata(nauxdata+1) = eterm(umbr)
!!$           nauxdata = nauxdata + 1
!!$           q_add_vpress = .true.
!!$        endif
!!$##ENDIF
!!$##IF ADUMB
!!$        !JMS 7/2012 -- insert energy and reaction coordinates for adaptive umbrella
!!$        if (numbr > 0 ) then
!!$           call realloc(nauxdata+1+numbr, auxdata)
!!$           auxdata(nauxdata+1) = eterm(adumb)
!!$           nauxdata = nauxdata + 1
!!$           auxdata(nauxdata+1:nauxdata+numbr) = coum(1:numbr)
!!$           nauxdata = nauxdata + numbr
!!$        endif
!!$##ENDIF
!!$        ! Add vpress
!!$        if (q_add_vpress) then
!!$           call realloc(nauxdata+9, auxdata)
!!$           auxdata(nauxdata+1:nauxdata+9) = vpress(1:9)
!!$           nauxdata = nauxdata + 9
!!$        endif
!!$     endif
!!$     
!!$     return
!!$   contains
!!$     ! *
!!$     ! * Reallocates auxdata() -array as needed
!!$     ! *
!!$     subroutine realloc(new_size, auxdata)
!!$       use memory,only:chmrealloc
!!$       implicit none
!!$       ! Input / Output
!!$       integer, intent(in) :: new_size
!!$       real(chm_real), allocatable, dimension(:), intent(inout) :: auxdata
!!$       
!!$       if (size(auxdata) < new_size) then
!!$          call chmrealloc('energy.src','add_to_auxdata','auxdata',new_size,crl=auxdata)
!!$       endif
!!$       
!!$       return
!!$     end subroutine realloc
!!$   end subroutine add_to_auxdata

end module energym

