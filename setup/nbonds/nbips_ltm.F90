module nbips
  use chm_kinds
  use dimens_fcm

#if KEY_NBIPS==1 /*nbips_fcm*/
!
!   Common file for Isotropic Periodic Sum (IPS) method
!
!      QIPS     -    Using IPS for nonbonded energy calculation
!      QIPSINI  -    Initiatiation of IPS calculation
!      QAIPS     -   Using anisotropic IPS calculation
!      QIPSFIN   -   Using anisotropic IPS calculation for finite systems
!      QIPSONLY -    Using IPS nonbonded routine for IPS calculation
!      QIPSUPD -     Update IPS system calculation
!      QIPSNETCG -   Use net charge to calculate boundary electrostatic IPS energy
!      QIPSEXP -   Use double exponential potential for VDW interation
!
!      IPSX     -  Homogeneity index along X axis.
!      IPSY     -  Homogeneity index along Y axis.
!      IPSZ     -  Homogeneity index along Z axis.
!
!      NIPSFRQ -    The number of steps between update system energies
!      NIPSCNT -    The number of steps between update system energies
!      IPSUPD -     The number of IPS cutoff updating
!      IPSSIZ -     Grid size for anisotropic IPS calculation
!
!      IPSWNB  -    Heap index for vdw factor of each atom
!
!      IPSQ   -    Heap index for charge grid
!      IPSPQ  -    Heap index for electrostatic IPS potential
!      IPSQXX -    Heap index for electrostatic IPS XX tensor
!      IPSQXY -    Heap index for electrostatic IPS XY tensor
!      IPSQXZ -    Heap index for electrostatic IPS XZ tensor
!      IPSQYY -    Heap index for electrostatic IPS YY tensor
!      IPSQYZ -    Heap index for electrostatic IPS YZ tensor
!      IPSQZZ -    Heap index for electrostatic IPS ZZ tensor
!      IPSW   -    Heap index for VDW grid
!      IPSPW  -    Heap index for VDW IPS potential
!      IPSWXX -    Heap index for VDW IPS XX tensor
!      IPSWXY -    Heap index for VDW IPS XY tensor
!      IPSWXZ -    Heap index for VDW IPS XZ tensor
!      IPSWYY -    Heap index for VDW IPS YY tensor
!      IPSWYZ -    Heap index for VDW IPS YZ tensor
!      IPSWZZ -    Heap index for VDW IPS ZZ tensor
!
!      IPSSFY -    Heap index for atomic system force at y direction
!      IPSSFZ -    Heap index for atomic system force at z direction
!      IPSFXC -    Heap index for atomic cutoff force at x direction
!      IPSFYC -    Heap index for atomic cutoff force at y direction
!      IPSFZC -    Heap index for atomic cutoff force at z direction
!
!      RIPS    -    The radius of the spheric boundary 
!      RAIPS    -   The input anisotropic radius 
!
!      VBIPS     -  Volume of the Simulation system
!      DVBIPS    -  Volume change allowance
!
!      GIPSX    -   Default grid size in x-axis
!      GIPSY    -   Default grid size in y-axis
!      GIPSZ    -   Default grid size in z-axis

!      AEXPPOT  -   Double exponential potential repulsive exponent constant 
!      CEXPPOT  -   Double exponential potential attractive exponent constant 
!
!      AIPSE(0:6)  -  parameters for electrostatic IPS energy
!      BIPSE(6)  -  parameters for electrostatic IPS forces
!      AIPSVC(0:6) -  parameter for L-J 6 term IPS energy
!      BIPSVC(6) -  parameter for L-J 6 term IPS forces
!      AIPSVA(0:6) -  parameter for L-J 12 term IPS energy
!      BIPSVA(6) -  parameter for L-J 12 term IPS forces
!
!      PIPSE0    -     Self electrostatic energy difference
!      PIPSVA0    -    Self vdw 12-term energy difference
!      PIPSVC0    -    Self vdw 6-term energy difference
!
!      PIPSEC    -     Cutoff electrostatic energy difference
!      PIPSVAC    -    Cutoff vdw 12-term energy difference
!      PIPSVCC    -    Cutoff vdw 6-term energy difference
!
!      EIPSSEL    -    Total system electrostatic energy
!      EIPSSNB    -    Total system vdw energy
!
!      EIPSAEL    -    Total aips electrostatic energy
!      EIPSANB    -    Total aips vdw energy
!      VIRAIPS    -    Total aips viral
!
!      CGSUM      -    Total charge
!      CIJSUM     -    Total vdw r6 coefficients
!      AIJSUM     -    Total vdw r12 coefficients
!
!      VIRIPS      -  system  virial 
!      PIPSVIR(9)  -  Pressure tensor matrix 
!      URIPS(9)    -  Rotation matrix for finite systems
!      UVIPS(9)    -  Inverse rotation matrix for finite systems
!
!
      INTEGER NIPSFRQ,NIPSCNT,IPSUPD,IPSSIZ
      INTEGER MIPSX,MIPSY,MIPSZ,MIPSO
      INTEGER IPSQ,IPSQXX,IPSQXY,IPSQXZ,IPSQYY,IPSQYZ,IPSQZZ
      INTEGER IPSW,IPSWXX,IPSWXY,IPSWXZ,IPSWYY,IPSWYZ,IPSWZZ
      INTEGER IPSWNB
!
      real(chm_real) IPSX,IPSY,IPSZ
      real(chm_real) VBIPS,DVBIPS,GIPSX,GIPSY,GIPSZ
      real(chm_real) RIPS,RIPS2,RIPSR,RIPS2R,RIPS6R,RIPS12R,RAIPS
      real(chm_real) SRIPS,AEXPPOT,CEXPPOT
      real(chm_real) AIPSE(0:6),AIPSVC(0:6),AIPSVA(0:6)
      real(chm_real) BIPSE(6),BIPSVC(6),BIPSVA(6),BBIPSE(6),BBIPSVC(6),BBIPSVA(6)
      real(chm_real) PIPSE0,PIPSVC0,PIPSVA0,PIPSEC,PIPSVCC,PIPSVAC
      real(chm_real) EIPSSNB,EIPSSEL,VIRIPS,EIPSANB,EIPSAEL,VIRAIPS
      real(chm_real) PIPSVIR(9),URIPS(9),UVIPS(9)
      real(chm_real) A2SUM,C2SUM,AIJSUM,CIJSUM,CG2SUM,CGSUM
      real(chm_real) PA2SUM,PC2SUM,PAIJSUM,PCIJSUM,PCGSUM,PCG2SUM
      real(chm_real) RA2SUM,RC2SUM,RAIJSUM,RCIJSUM,RCGSUM,RCG2SUM
!
      LOGICAL QIPS,QIPSINI
      LOGICAL QIPSUPD,QIPSONLY,QAIPS,QIPSFIN,QIPS2D,QIPSNETCG,QIPSEXP
!
      Common /NBIPSI/NIPSFRQ,NIPSCNT,IPSUPD,IPSSIZ, &
        MIPSX,MIPSY,MIPSZ,MIPSO, &
        IPSQ,IPSQXX,IPSQXY,IPSQXZ,IPSQYY,IPSQYZ,IPSQZZ, &
        IPSW,IPSWXX,IPSWXY,IPSWXZ,IPSWYY,IPSWYZ,IPSWZZ, &
        IPSWNB 
!
      Common /NBIPSR/IPSX,IPSY,IPSZ,VBIPS,DVBIPS,GIPSX,GIPSY,GIPSZ, &
            RIPS,RIPS2,RIPSR,RIPS2R,RIPS6R,RIPS12R,RAIPS, &
            AIPSE,AIPSVC,AIPSVA, &
            BIPSE,BIPSVC,BIPSVA, &
            PIPSE0,PIPSVC0,PIPSVA0,PIPSEC,PIPSVCC,PIPSVAC, &
            EIPSSNB,EIPSSEL,VIRIPS,EIPSANB,EIPSAEL,VIRAIPS, &
            PIPSVIR,URIPS,UVIPS,A2SUM,C2SUM,CG2SUM,AIJSUM,CIJSUM,CGSUM, &
            PA2SUM,PC2SUM,PAIJSUM,PCIJSUM,PCGSUM,PCG2SUM,  &
            RA2SUM,RC2SUM,RAIJSUM,RCIJSUM,RCGSUM,RCG2SUM,  &
            SRIPS,AEXPPOT,CEXPPOT
      Common /NBIPSL/QIPS,QIPSINI, & 
         QIPSUPD,QIPSONLY,QAIPS,QIPSFIN,QIPS2D,QIPSNETCG,QIPSEXP
!
#if KEY_SAVEFCM==1
      SAVE /NBIPSI/
      SAVE /NBIPSR/
      SAVE /NBIPSL/
#endif 
#endif /* (nbips_fcm)*/
!
end module nbips

