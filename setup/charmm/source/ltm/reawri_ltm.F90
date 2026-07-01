module reawri
  use chm_kinds
  use dimens_fcm
  implicit none
!
!   DYNAMICS VARIABLES COMMON FILE
!
!       Most of the variables in this common file are written to the
!       dynamics restart file.
!
!      IUNREA  - Unit to read restart file from
!      IUNWRI  - Unit to write restart file
!      IUNCRD  - Unit to write the trajectory
!      IUNVEL  - Unit to write the velocities
!      IUNXYZ  - Unit to write in XYZ coordinate (for MOLDEN, R*8!!)
!      IUNQMC  - Unit to write Mulliken, Lowdin, Kollman charges
!      NSAVC   - Frequency for saving coordinates
!      NSAVV   - Frequency for saving velocities
!      NSAVX   - Frequency for saving XYZ coordinates
!      NSAVQ   - Frequency for saving ab initio charges
!      MXYZ    - For XYZ format (1=coor,2=coor+vel,3=coor+vel+force)
!      NSTEP   - Number of steps for the current run
!      JHSTRT  - Number of steps in all previous runs
!      KUNIT   - Unit to write the energy file
!      IUPTEN  - Unit to write the pressure tensor 
!      IPTFRQ  - Frequency for saving the pressure tensor
!      IUNOS   - Unit to write Nose S-value
!      NSNOS   - Frequency for saving Nose s-value
!      ISEED   - Random seed value (continually updated)
!      DELTA   - Dynamics time step in AKMA units
!      TIMEST  - Dynamics time step in picoseconds
!      FIRSTT  - Initial temperature (used in heating)
!      FINALT  - Final temperature
!      TEMINC  - Temperature increment for heating
!      TWINDL  - Low temperature tolerance for equilibration
!      TWINDH  - High temperature tolerance for equilibration
!      ECHECK  - Total energy change tolerance for a given step
!      TSTRUC  - The temperature of the initial structure
!
! for constant pressure
!      QCNSTP  - Logical flag constant pressure
!      QNPT    - Flag for Nose-Hoove temp control with constant pressure       
!      QCNSTPL - Flag for linear modification of the pressure
!      QCNSTT  - Logical flag constant temperature
!      QCNSTTX - Logical flag exact constant temperature (constant KE)
!      QCBYGR  - Logical flag for scaling coordinates by group centers
!      QPSVEL  - Logical flag for velocity scaling due to piston moves
!      PWMAT(6)- Piston mass matrix
!      PWINV(6)- Inverse piston mass matrix
!      PRFWD(6)- Piston random force scale factor
!      PALPHA  - Piston velocity decay factor
!      PBFACT  - Piston force scale factor due to friction
!      PVFACT  - Piston velocity scale factor due to fricton
!
!      REFP(6) - Reference pressure tensor (lower triangle)
!      REFPI(6)- Initial   pressure tensor (lower triangle)
!      REFPF(6)- Final     pressure tensor (lower triangle)
!      PTYPE   - Which pressure couples (.TRUE.= external pressure)
!                              (default=.FALSE.= internal pressure)
!      PITER   - Integer determining # of iterations to get consistency
!      TITER   - Number of temperature iterations to get consistency
!      TMASS   - Thermal piston mass
!      REFT    - Nose-Hoover target temperature  
!      QSURF   - Flag for constant surface tension dynamics
!      SURFT   - Target surface tension value 
!      QCONZ   - Flag for constant z-spacing with constant surface tension
!      QISO    - Flag for isotropic pressure scaling
!      GAM     - the product of gamma, timestep, and timfac
!      KBT     - Boltzmann constant times bath temperature
!
! for constant temperature
!      TCOUPL  - Temperature coupling coefficient
!      TREF    - Reference temperature
!
!=======================================================================
! for print out potential to get free energy with BLOCK
!
!      IBLCKFEP  Unit to write the Potential energy for post-proceeding
!      NBLCKFEP  Frequency for saving Potential energy
!=======================================================================
       INTEGER IBLCKFEP, NBLCKFEP

! for Hybrid Monte Carlo (ARD 00-08-15)
!      HMCKEI  - The initial kinetic energy 
#if KEY_MC==1
       real(chm_real)  HMCKEI 
#endif

! logicals
      LOGICAL QCNSTP,QCNSTPL,QCNSTT,QCNSTTX,PTYPE,QCBYGR,QPSVEL, &
              QNPT,QSURF,QCONZ,QISO
! integers
      INTEGER IUNREA, IUNWRI, IUNCRD, IUNVEL, NSAVC, NSAVV, NSTEP, &
              JHSTRT, KUNIT, ISEED, IUNOS, NSNOS, PITER, TITER, &
              NSAVX, MXYZ, IUNXYZ, IUNQMC, NSAVQ, IUPTEN, IPTFRQ
! reals
      real(chm_real)  DELTA, TIMEST, FIRSTT, FINALT, TWINDL, TWINDH, &
              TEMINC, ECHECK, REFP(6), REFPI(6), REFPF(6), TCOUPL, TREF, &
              PWMAT(6), PWINV(6), PRFWD(6), PALPHA, PBFACT, PVFACT, &
              TMASS, REFT, SURFT, GAM, KBT, TSTRUC 
!
      INTEGER RPXX,RPYY,RPZZ,RPXY,RPXZ,RPYZ
      PARAMETER (RPXX=1,RPYY=3,RPZZ=6,RPXY=2,RPXZ=4,RPYZ=5)
#if KEY_DHDGB==1
!AP/MF
     INTEGER IUDHDGB
#endif
!
contains
  subroutine reawri_init
    use number
    use consta,only:roomt
    iunrea=-1
    iunwri=-1
    iuncrd=-1
    iunvel=-1
    iunxyz=-1
    iunqmc=-1
    nsavc=10
    nsavv=10
    nsavx=10
    nsavq=10
    nstep=100
    nsnos=10
    jhstrt=0
    kunit=-1
    iupten=-1
    iunos=-1
#if KEY_DHDGB==1
!AP/MF
    IUDHDGB=-1
#endif
    delta = ZERO
    timest = 0.001 ! beware of roundoff
    firstt = ZERO
    finalt=roomt
    tstruc=fmark
    twindl=-10.0_chm_real
    twindh=10.0_chm_real
    teminc=5.0_chm_real
    echeck=20.0_chm_real
    qcnstp = .false.
    qcnstt = .false.
    qcnsttx= .false.
    iseed=314159
    return
  end subroutine reawri_init

end module reawri

